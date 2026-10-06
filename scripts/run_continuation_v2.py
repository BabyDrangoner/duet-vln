#!/usr/bin/env python3
"""Run a frozen continuation comparator on full navigation or a fixed train panel.

Top-level metrics always describe natural navigation. Perturbed conditions are
reported separately. Official validation requires a previously registered,
single-use full-split access; this command does not register or close its ledger.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from vln_improve.continuation_learning import ContinuationComparator
from vln_improve.continuation_v2 import SCHEMA, SCHEDULES, CausalContinuationHook
from vln_improve.pipeline import backup_mount_identity, validate_backup_root
from vln_improve.protocol import file_sha256, object_sha256, resolve_config
from vln_improve.study_ledger import StudyLedger

HEAD_SCHEMA = 'e3_continuation_head_v2'
REPORT_SCHEMA = 'e3_continuation_navigation_v2'
CODE_FILES = ('scripts/run_continuation_v2.py', 'src/vln_improve/continuation_v2.py',
    'src/vln_improve/continuation_learning.py', 'src/vln_improve/continuation_probe.py',
    'src/vln_improve/features.py', 'src/vln_improve/protocol.py',
    'src/vln_improve/intervention_runtime.py', 'src/vln_improve/pipeline.py',
    'src/vln_improve/study_ledger.py', 'scripts/run_duet.py', 'scripts/prepare_duet.py')
ARM_MODES = {'relative-history': ('relative', True), 'absolute-history': ('absolute', True),
    'teacher-history': ('teacher', True), 'relative-nohistory': ('relative', False),
    'relative-olddata': ('relative', True)}
ASSET_FIELDS = ('base_checkpoint_sha256', 'feature_sha256',
                'train_annotation_sha256', 'connectivity_sha256')


def code_identity():
    files = {name: file_sha256(ROOT/name) for name in CODE_FILES}
    return files, object_sha256(files)


def load_head(path, *, identity=False):
    """Load plain tensor weights; the policy receives none of this metadata."""
    import torch
    if path is None:
        if not identity:
            raise ValueError('navigation requires --head or --identity')
        torch.manual_seed(0)
        model = ContinuationComparator().eval()
        return model, {'schema': HEAD_SCHEMA, 'arm': 'identity-zero', 'epoch': 0,
            'model_config': model.config, 'provenance': {}, 'thresholds': {'sr': 0., 'spl': 0.}}
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema') != HEAD_SCHEMA:
        raise ValueError('unsupported continuation head schema')
    if payload.get('arm') not in ARM_MODES or type(payload.get('epoch')) is not int or payload['epoch'] < 0:
        raise ValueError('invalid continuation head arm or epoch')
    if not isinstance(payload.get('provenance'), dict):
        raise ValueError('head lacks training provenance')
    if not re.fullmatch(r'[0-9a-f]{64}', payload['provenance'].get('fit_manifest_sha256', '')):
        raise ValueError('head lacks a verified fit manifest identity')
    thresholds = payload.get('thresholds')
    if not isinstance(thresholds, dict) or set(thresholds) != {'sr', 'spl'} or any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in thresholds.values()
    ):
        raise ValueError('head thresholds must contain finite nonnegative sr and spl')
    cfg = payload.get('model_config')
    if not isinstance(cfg, dict) or set(cfg) != {'feature_dim', 'hidden_dim', 'mode', 'history'}:
        raise ValueError('invalid comparator model configuration')
    if cfg['feature_dim'] != 1549 or (cfg['mode'], cfg['history']) != ARM_MODES[payload['arm']]:
        raise ValueError('head arm or features differ from the navigation interface')
    model = ContinuationComparator(**cfg)
    weights = payload.get('state_dict')
    if not isinstance(weights, dict) or not all(
        isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in weights.values()
    ):
        raise ValueError('head state contains invalid or nonfinite weights')
    model.load_state_dict(weights, strict=True)
    if identity and any(bool(p.detach().ne(0).any()) for p in model.comparison[-1].parameters()):
        raise ValueError('identity requires a zero-output head')
    if not identity and payload['epoch'] < 1:
        raise ValueError('trained navigation requires a positive checkpoint epoch')
    return model.eval(), {key: value for key, value in payload.items() if key != 'state_dict'}


def read_panel(path, split):
    if path is None:
        return None
    if split != 'train_dev':
        raise ValueError('dataset panels are restricted to train_dev')
    manifest = json.loads(Path(path).read_text())
    selection = manifest.get('selection', {})
    ids, scans = selection.get('instr_ids'), selection.get('scan_ids')
    if (manifest.get('schema') != SCHEMA or manifest.get('complete') is not True
            or selection.get('split') != 'train_dev'
            or not isinstance(ids, list) or not ids
            or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids)
            or selection.get('count') != len(ids)
            or not isinstance(scans, list) or not scans or len(set(scans)) != len(scans)
            or selection.get('conditions') != list(SCHEDULES)
            or type(selection.get('seed')) is not int or selection['seed'] < 0):
        raise ValueError('invalid complete fixed train_dev panel')
    return manifest


def select_evaluation_rows(rows, split, panel=None):
    by_id = {r['instr_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate evaluation instruction IDs')
    if panel is None:
        expected = 2890 if split == 'train_dev' else 2349
        if len(rows) != expected:
            raise ValueError(f'full {split} requires exactly {expected} instructions')
        return sorted(rows, key=lambda r: r['instr_id'])
    if split != 'train_dev':
        raise ValueError('official validation cannot use an instruction panel')
    ids = panel['selection']['instr_ids']
    if any(key not in by_id for key in ids):
        raise ValueError('panel contains an instruction outside train_dev')
    selected = [by_id[key] for key in sorted(ids)]
    if sorted({r['scan'] for r in selected}) != sorted(panel['selection']['scan_ids']):
        raise ValueError('panel scene inventory differs from real annotations')
    return selected


def validate_provenance(provenance, actual, *, name):
    """Bind trained/cached inputs to the actual assets and causal hook version."""
    if not isinstance(provenance, dict):
        raise ValueError(name+' lacks collection provenance')
    required = (*ASSET_FIELDS, 'model_config_sha256', 'partition_seed', 'dev_fraction', 'upstream_lock')
    for key in required:
        if key not in provenance or provenance[key] != actual[key]:
            raise ValueError(name+' collection provenance differs: '+key)
    collected_source = provenance.get('source_files', {})
    for key in ('src/vln_improve/continuation_v2.py', 'src/vln_improve/continuation_probe.py',
                'src/vln_improve/features.py', 'src/vln_improve/protocol.py', 'scripts/run_duet.py'):
        if collected_source.get(key) != file_sha256(ROOT/key):
            raise ValueError(name+' causal collection source differs: '+key)


def validate_registration(args, code_sha):
    if args.split != 'val_unseen':
        if args.access_id or args.ledger or args.execution_backup_root:
            raise ValueError('train_dev cannot consume a validation access')
        return None
    if (args.identity or args.dataset_manifest is not None or not args.head
            or not args.access_id or not args.ledger or not args.experiment or not args.execution_backup_root):
        raise ValueError('val_unseen requires full natural navigation, frozen head and registered access')
    study = json.loads(args.study.read_text())
    if study['baseline']['selection_episodes'] != 2349:
        raise ValueError('study does not describe full 2349-episode validation')
    access = StudyLedger(args.ledger, study).lookup(args.access_id)
    if access['outcome'] is not None:
        raise ValueError('completed or failed validation access cannot be rerun')
    expected = {'category': args.category, 'split': 'val_unseen', 'label_use': 'evaluation',
        'parameter_fitting_split': 'train_fit', 'subset': False, 'subset_ids': [],
        'config_sha256': file_sha256(args.experiment), 'checkpoint_sha256': file_sha256(args.head),
        'code_sha256': code_sha, 'seed': args.seed, 'expected_episodes': 2349}
    if any(access['registration']['request'].get(key) != value for key, value in expected.items()):
        raise ValueError('registered validation identity differs from the requested navigation')
    return access['registration']


def validate_formal_head(metadata):
    provenance = metadata.get('provenance', {})
    if (provenance.get('scope') != 'training_experiment'
            or 'engineering_smoke' in str(metadata.get('selection_status', ''))):
        raise ValueError('official validation cannot use an engineering-smoke head')
    for name, split in (('fit_selection', 'train_fit'), ('dev_selection', 'train_dev')):
        selection = provenance.get(name, {})
        if selection.get('split') != split or selection.get('smoke') is not False:
            raise ValueError('official validation requires explicitly formal fit/dev caches')


def exclusive_json(path, value):
    """Publish a complete JSON file atomically without replacing any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def validation_execution(args, registration, code_sha):
    if registration is None:
        yield None
        return
    if validate_registration(args, code_sha) != registration:
        raise ValueError('validation registration changed before execution')
    backup = args.execution_backup_root.expanduser().resolve()
    mount = backup_mount_identity(backup, backend='filesystem')
    def check():
        validate_backup_root(backup, backend='filesystem', expected_identity=mount)
    suffix = object_sha256([registration['study_id'], args.access_id])
    root = args.ledger.resolve().with_name(args.ledger.name+'.executions')
    local = root/(suffix+'.claim.json')
    remote = backup/'validation-executions'/(suffix+'.claim.json')
    if local.exists() or remote.exists():
        raise ValueError('validation access already executed; a rerun needs a new access ID')
    payload = {'schema': 'continuation_validation_execution_v2', 'access_id': args.access_id,
        'registration': registration, 'code_sha256': code_sha, 'head_sha256': file_sha256(args.head),
        'ledger_sha256': file_sha256(args.ledger), 'output': str(args.output.resolve()),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid()}
    exclusive_json(local, payload)
    check()
    exclusive_json(remote, payload)
    check()
    if file_sha256(local) != file_sha256(remote):
        raise ValueError('validation claim backup failed readback')
    claim = {'local': str(local), 'backup': str(remote), 'sha256': file_sha256(local)}
    def outcome(status, **extra):
        value = {'status': status, 'access_id': args.access_id, 'claim_sha256': claim['sha256'], **extra}
        exclusive_json(local.with_name(suffix+'.outcome.json'), value)
        check()
        exclusive_json(remote.with_name(suffix+'.outcome.json'), value)
        check()
    try:
        if validate_registration(args, code_sha) != registration:
            raise ValueError('validation registration changed after execution claim')
        yield claim
    except BaseException as error:
        try:
            outcome('failed', error_type=type(error).__name__, error=str(error))
        except BaseException as recording_error:
            error.add_note('Validation claim remains; outcome recording failed: '+str(recording_error))
        raise
    else:
        outcome('completed', report_sha256=file_sha256(args.output))


def summarize(episodes):
    if not episodes:
        raise ValueError('cannot summarize an empty navigation condition')
    mapping = {'sr': ('success', 100), 'spl': ('spl', 100), 'oracle_sr': ('oracle_success', 100),
        'nDTW': ('nDTW', 100), 'SDTW': ('SDTW', 100), 'CLS': ('CLS', 100),
        'nav_error': ('nav_error', 1), 'oracle_error': ('oracle_error', 1),
        'action_steps': ('action_steps', 1), 'steps': ('trajectory_steps', 1),
        'lengths': ('trajectory_lengths', 1)}
    for episode in episodes:
        if episode['success'] not in (0, 1) or not 0 <= episode['spl'] <= 1:
            raise ValueError('invalid navigation success/SPL fractions')
        if any(not math.isfinite(episode[key]) for key, _ in mapping.values()):
            raise ValueError('nonfinite navigation metric')
    return {name: scale*sum(e[key] for e in episodes)/len(episodes)
            for name, (key, scale) in mapping.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head', type=Path)
    p.add_argument('--split', choices=('train_dev', 'val_unseen'))
    p.add_argument('--dataset-manifest', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--identity', action='store_true')
    p.add_argument('--config', type=Path, default=ROOT/'configs/r2r.json')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--study', type=Path, default=ROOT/'configs/research_study.json')
    p.add_argument('--ledger', type=Path)
    p.add_argument('--access-id')
    p.add_argument('--experiment', type=Path)
    p.add_argument('--category', choices=('pilot', 'confirmatory'), default='pilot')
    p.add_argument('--execution-backup-root', type=Path)
    p.add_argument('--print-code-sha256', action='store_true')
    args = p.parse_args(argv)
    for name in ('head', 'dataset_manifest', 'output', 'config', 'study', 'ledger',
                 'experiment', 'execution_backup_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    files, source_sha = code_identity()
    if args.print_code_sha256:
        print(source_sha)
        return 0
    if args.split is None or args.output is None or args.seed < 0:
        p.error('--split, --output and a nonnegative seed are required')
    if args.output.exists():
        raise ValueError('refusing to overwrite a navigation report')
    registration = validate_registration(args, source_sha)
    object_manifest_sha = file_sha256(args.dataset_manifest) if args.dataset_manifest else None
    panel = read_panel(args.dataset_manifest, args.split)
    head_sha = file_sha256(args.head) if args.head else None
    model, head_metadata = load_head(args.head, identity=args.identity)
    if args.split == 'val_unseen':
        validate_formal_head(head_metadata)
    cfg = resolve_config(args.config, ROOT)
    if (cfg['model']['batch_size'] != 1 or cfg['model']['max_action_len'] != 15
            or cfg['model']['fusion'] != 'dynamic' or cfg['model']['angle_feat_size'] != 4):
        raise ValueError('continuation navigation requires the frozen batch1 dynamic 15-decision DUET')
    import run_duet
    import torch
    lock = run_duet.verify()
    data = Path(cfg['dataset_root'])
    actual = {'base_checkpoint_sha256': file_sha256(cfg['base_checkpoint']),
        'feature_sha256': file_sha256(data/'R2R/features/pth_vit_base_patch16_224_imagenet.hdf5'),
        'train_annotation_sha256': file_sha256(data/'R2R/annotations/R2R_train_enc.json'),
        'connectivity_sha256': object_sha256({f.name: file_sha256(f) for f in sorted((data/'R2R/connectivity').glob('*_connectivity.json'))}),
        'model_config_sha256': file_sha256(args.config), 'partition_seed': cfg['partition_seed'],
        'dev_fraction': cfg['dev_fraction'], 'upstream_lock': lock}
    if args.head:
        validate_provenance(head_metadata['provenance'].get('fit_provenance'), actual, name='head')
    if panel:
        validate_provenance(panel.get('provenance'), actual, name='panel')
    if not torch.cuda.is_available():
        raise RuntimeError('real navigation requires NVIDIA CUDA')
    model = model.cuda().eval()
    conditions = list(SCHEDULES) if panel else ['natural']
    condition_seed = panel['selection']['seed'] if panel else args.seed
    sys.path.insert(0, str(run_duet.DEFAULT_DEST/'map_nav_src'))
    import r2r.agent as upstream
    original_class, original_partition, original_cli = upstream.GMapNavAgent, run_duet.select_partition, run_duet.parse_cli
    selection = None
    outcomes = {condition: [] for condition in conditions}
    identity_checked = 0
    def partition(rows, split, fraction, partition_seed):
        nonlocal selection
        if split != args.split:
            raise ValueError('unexpected evaluation split')
        chosen = select_evaluation_rows(original_partition(rows, split, fraction, partition_seed), split, panel)
        selection = {'split': split, 'instr_ids': sorted(r['instr_id'] for r in chosen),
            'scan_ids': sorted({r['scan'] for r in chosen}), 'conditions': conditions,
            'count': len(chosen), 'subset': panel is not None, 'condition_seed': condition_seed}
        return chosen
    class Agent(original_class):
        def execute(self, row, condition, *, comparator, pure_baseline=False):
            self.env.data = [copy.deepcopy(row)]
            self.env.reset_epoch(shuffle=False)
            self.scanvp_cands, self.feedback = {}, 'argmax'
            old_hook, old_move = self.decision_hook, self.make_equiv_action
            hook = None if pure_baseline else CausalContinuationHook(self, condition=condition,
                seed=condition_seed, predictor=comparator, sr_threshold=head_metadata['thresholds']['sr'],
                spl_threshold=head_metadata['thresholds']['spl'])
            self.decision_hook = hook
            if hook is not None:
                self.make_equiv_action = hook.make_equiv_action
            try:
                with torch.no_grad():
                    trajectories = super().rollout(reset=True)
                if len(trajectories) != 1 or trajectories[0]['instr_id'] != row['instr_id']:
                    raise ValueError('rollout instruction differs')
                result = hook.finish(trajectories) if hook else {'path': trajectories[0]['path'],
                    'intervention': None, 'perturbations': []}
            finally:
                self.decision_hook, self.make_equiv_action = old_hook, old_move
            scan, target = self.env.gt_trajs[row['instr_id']]
            if scan != row['scan']:
                raise ValueError('evaluation scene differs')
            metrics = {k: float(v) for k, v in self.env._eval_item(scan, result['path'], target).items()}
            return {'instr_id': row['instr_id'], 'scan_id': scan, 'condition': condition,
                'trajectory': result['path'], 'metrics': metrics, 'intervention': result['intervention'],
                'perturbations': result['perturbations'],
                'decision_count': result.get('decision_count', int(metrics['action_steps']))}

        def test(self, use_dropout=False, feedback='argmax', **kwargs):
            nonlocal identity_checked
            if use_dropout or feedback != 'argmax' or kwargs:
                raise ValueError('frozen inference settings changed')
            rows = self.env.data
            self.results = {}
            try:
                for number, row in enumerate(sorted(rows, key=lambda r: r['instr_id']), 1):
                    for condition in conditions:
                        baseline = self.execute(row, condition, comparator=None,
                            pure_baseline=condition == 'natural') if args.identity else None
                        result = self.execute(row, condition, comparator=model)
                        if baseline is not None:
                            if (result['trajectory'] != baseline['trajectory'] or result['metrics'] != baseline['metrics']
                                    or result['intervention'] is not None):
                                raise AssertionError('zero comparator changed original full navigation')
                            identity_checked += 1
                        outcomes[condition].append(result)
                        if condition == 'natural':
                            self.results[row['instr_id']] = {'instr_id': row['instr_id'], 'path': result['trajectory'], 'details': {}}
                    if number % 16 == 0:
                        print(json.dumps({'navigation_instructions': number, 'conditions': len(conditions),
                                          'identity_checked_rollouts': identity_checked}), flush=True)
            finally:
                self.env.data = rows
                self.env.reset_epoch(shuffle=False)
    raw = args.output.with_name(args.output.stem+'.duet-driver-'+str(time.time_ns())+'.json')
    with validation_execution(args, registration, source_sha) as claim:
        began = time.monotonic()
        try:
            upstream.GMapNavAgent, run_duet.select_partition = Agent, partition
            run_duet.parse_cli = lambda: SimpleNamespace(mode='baseline', config=args.config, split=args.split,
                output=raw, cache=None, head=None, limit=None, seed=args.seed)
            run_duet.main()
        finally:
            upstream.GMapNavAgent, run_duet.select_partition, run_duet.parse_cli = original_class, original_partition, original_cli
        if code_identity()[0] != files or (args.head and file_sha256(args.head) != head_sha):
            raise ValueError('navigation code or frozen head changed during evaluation')
        if panel and file_sha256(args.dataset_manifest) != object_manifest_sha:
            raise ValueError('fixed panel changed during evaluation')
        raw_report = json.loads(raw.read_text())
        condition_reports = {}
        for condition, rows in outcomes.items():
            if len(rows) != selection['count'] or sorted(r['instr_id'] for r in rows) != selection['instr_ids']:
                raise ValueError('navigation condition did not cover the fixed instruction inventory')
            episodes = [dict(instr_id=r['instr_id'], scan_id=r['scan_id'], **r['metrics']) for r in rows]
            condition_reports[condition] = {'summary': summarize(episodes), 'episodes': episodes,
                'trajectories': [{'instr_id': r['instr_id'], 'trajectory': r['trajectory']} for r in rows],
                'interventions': [{k: r[k] for k in ('instr_id', 'scan_id', 'intervention', 'perturbations', 'decision_count')} for r in rows],
                'intervention_count': sum(r['intervention'] is not None for r in rows)}
        natural = condition_reports['natural']
        if any(abs(natural['summary'][key]-raw_report['summary'][key]) > 1e-8 for key in natural['summary']):
            raise ValueError('independent per-episode means differ from DUET natural summary')
        report = {'schema': REPORT_SCHEMA, 'metadata': dict(raw_report['metadata'],
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            mode='continuation_v2_identity' if args.identity else 'continuation_v2_navigation',
            subset=panel is not None, head_sha256=head_sha, head_metadata=head_metadata,
            dataset_manifest_sha256=object_manifest_sha, selection=selection,
            code_files=files, code_sha256=source_sha, validation_access_id=args.access_id,
            validation_execution=claim, identity_checked_rollouts=identity_checked,
            zero_comparator_full_trajectory_identity=True if args.identity else None),
            'summary': natural['summary'], 'episodes': natural['episodes'], 'trajectories': natural['trajectories'],
            'conditions': condition_reports, 'resources': dict(raw_report['resources'],
                total_attempt_seconds=time.monotonic()-began, condition_rollouts=selection['count']*len(conditions),
                identity_reference_rollouts=identity_checked),
            'claim': 'Natural navigation and fixed training stress conditions reported separately; no oracle action selection.'}
        exclusive_json(args.output, report)
        print(json.dumps({'output': str(args.output), 'summary': report['summary'],
                          'identity_checked_rollouts': identity_checked}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
