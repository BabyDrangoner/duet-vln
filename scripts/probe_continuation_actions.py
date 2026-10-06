#!/usr/bin/env python3
"""Frozen training-only E3 full-continuation opportunity probe; no model training."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vln_improve.protocol import file_sha256, object_sha256, resolve_config
from vln_improve.continuation_probe import (ContinuationHook, ProbeTaskStore, assert_anchor,
    candidate_actions, opportunity_summary, select_state_steps)
from vln_improve.intervention_runtime import select_records, verified_copy
from vln_improve.pipeline import (atomic_json, backup_mount_identity, validate_backup_root,
                                  validate_separate_roots)

CODE_FILES = ('scripts/probe_continuation_actions.py', 'src/vln_improve/continuation_probe.py',
              'scripts/run_duet.py', 'scripts/prepare_duet.py', 'src/vln_improve/protocol.py',
              'src/vln_improve/intervention_runtime.py', 'src/vln_improve/pipeline.py')


def code_identity():
    return {name: file_sha256(ROOT/name) for name in CODE_FILES}


def normalize_storage_paths(args):
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.backup_dir = Path(args.backup_dir).expanduser().resolve()
    validate_separate_roots(args.output_dir, args.backup_dir)


def validate_request(args, spec, cfg):
    if spec.get('schema') != 'e3_continuation_probe_v1':
        raise ValueError('wrong frozen probe specification')
    p = spec['probe']
    if args.split not in ('train_fit', 'train_dev') or args.split not in p['allowed_splits']:
        raise ValueError('probe is restricted to training partitions')
    count = p['engineering_smoke_instruction_count_per_split'] if args.engineering_smoke else p['instruction_count_per_split']
    if args.limit != count or args.seed != spec['seed']:
        raise ValueError('limit/seed differ from frozen probe; use 64 or explicit engineering smoke 2')
    if (p['selection'] != 'scene_stratified' or p['conditions'] != ['natural', 'perturb_step2']
            or p['state_steps'] != {'natural': [0, 3, 'terminal'], 'perturb_step2': [3, 6, 'terminal']}
            or p['perturbed_states_min_step'] != 3 or p['max_action_len'] != 15
            or p['max_branches_per_state'] != 4 or p['max_selected_states_per_condition'] != 3
            or p['train_model'] or p['official_validation_access'] or p['adaptive_expansion']):
        raise ValueError('implemented probe semantics differ from frozen specification')
    if cfg['model']['batch_size'] != 1 or cfg['model']['fusion'] != 'dynamic' or cfg['model']['max_action_len'] != 15:
        raise ValueError('requires dynamic frozen DUET, batch size 1, original 15-decision budget')


def validate_asset_pins(provenance, specification):
    pins = specification.get('asset_pins', {})
    names = ('base_checkpoint_sha256', 'feature_sha256', 'train_annotation_sha256', 'connectivity_sha256')
    if any(not isinstance(pins.get(k), str) or len(pins[k]) != 64 or provenance.get(k) != pins[k] for k in names):
        raise ValueError('actual DUET assets differ from frozen checkpoint/features/annotations/connectivity pins')


def compact_branch(result, reference):
    result = copy.deepcopy(result)
    result['is_anchor'] = result['target_action'] == reference['states'][result['target_step']]['executed_action']
    if result['is_anchor']:
        assert_anchor(result, reference)
        result['anchor_metrics_equal'] = True
    result['action_trace'] = [{'step': s['step'], 'viewpoint': s['viewpoint'],
        'executed_action': s['executed_action'], 'original_stop_probability': s['original_stop_probability']}
        for s in result.pop('states')]
    result['baseline_metrics'] = reference['metrics']
    result['rescue'] = result['metrics']['success'] > reference['metrics']['success']
    result['harm'] = result['metrics']['success'] < reference['metrics']['success']
    result['spl_gain'] = result['metrics']['spl']-reference['metrics']['spl']
    result['label_only'] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/r2r.json')
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--split', choices=('train_fit', 'train_dev'), required=True)
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--backup-dir', type=Path, required=True)
    parser.add_argument('--backup-backend', choices=('drive', 'filesystem'), default='drive')
    parser.add_argument('--engineering-smoke', action='store_true')
    parser.add_argument('--deadline-seconds', type=float, default=36000)
    args = parser.parse_args(argv)
    normalize_storage_paths(args)
    spec = json.loads(args.experiment.read_text())
    cfg = resolve_config(args.config, ROOT)
    validate_request(args, spec, cfg)
    if args.deadline_seconds <= 0 or args.deadline_seconds > 36000:
        raise ValueError('deadline must be positive and at most 10 hours')
    backup_mount = backup_mount_identity(args.backup_dir, backend=args.backup_backend)
    backup_kind = backup_mount['verification']
    def check_backup():
        return validate_backup_root(args.backup_dir, backend=args.backup_backend, expected_identity=backup_mount)
    import run_duet
    lock = run_duet.verify()
    import torch
    dataset = Path(cfg['dataset_root'])
    source = code_identity()
    provenance = {'base_checkpoint_sha256': file_sha256(cfg['base_checkpoint']),
        'feature_sha256': file_sha256(dataset/'R2R/features/pth_vit_base_patch16_224_imagenet.hdf5'),
        'train_annotation_sha256': file_sha256(dataset/'R2R/annotations/R2R_train_enc.json'),
        'connectivity_sha256': object_sha256({f.name: file_sha256(f) for f in sorted((dataset/'R2R/connectivity').glob('*_connectivity.json'))}),
        'model': cfg['model'], 'partition_seed': cfg['partition_seed'], 'dev_fraction': cfg['dev_fraction'],
        'upstream_lock': lock, 'source_files': source, 'source_sha256': object_sha256(source),
        'experiment_sha256': file_sha256(args.experiment), 'model_config_file_sha256': file_sha256(args.config),
        'backup_backend': args.backup_backend,
        'torch_version': str(torch.__version__)}
    validate_asset_pins(provenance, spec)
    sys.path.insert(0, str(run_duet.DEFAULT_DEST/'map_nav_src'))
    import r2r.agent as upstream
    original_class, original_parse, original_partition = upstream.GMapNavAgent, run_duet.parse_cli, run_duet.select_partition
    store = None
    selection = None
    references, branches, expected_tasks = [], [], []
    start = time.monotonic()
    stop_requested = False
    previous_signals = {}
    attempted_tasks = resumed_tasks = 0

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    def boundary():
        if stop_requested or time.monotonic()-start >= args.deadline_seconds:
            raise InterruptedError('probe stopped at durable task boundary; resume with identical arguments')
        check_backup()

    def partition(data, split, dev_fraction, partition_seed):
        nonlocal store, selection
        if split != args.split:
            raise ValueError('unexpected partition access')
        rows = original_partition(data, split, dev_fraction, partition_seed)
        chosen = select_records(rows, 'scene_stratified', args.limit, args.seed)
        if len(chosen) != args.limit:
            raise ValueError('selected instruction count differs from frozen request')
        selection = {'split': args.split, 'scope': 'engineering_smoke' if args.engineering_smoke else 'training_opportunity_probe',
            'seed': args.seed, 'limit': args.limit, 'selection': 'scene_stratified',
            'instr_ids': sorted(r['instr_id'] for r in chosen), 'scan_ids': sorted({r['scan'] for r in chosen}),
            'instruction_set_sha256': object_sha256(sorted(r['instr_id'] for r in chosen)),
            'conditions': spec['probe']['conditions']}
        store = ProbeTaskStore(args.output_dir/'tasks', args.backup_dir/'tasks',
            {'provenance': provenance, 'selection': selection},
            check_backup=check_backup)
        return chosen

    class Agent(original_class):
        def _probe_rollout(self, row, condition, reference=None, target_step=None, target_action=None):
            self.env.data = [copy.deepcopy(row)]
            self.env.reset_epoch(shuffle=False)
            self.scanvp_cands = {}
            self.feedback = 'argmax'
            hook = ContinuationHook(self, condition=condition, seed=args.seed, reference=reference,
                                    target_step=target_step, target_action=target_action)
            old_hook, old_move = self.decision_hook, self.make_equiv_action
            self.decision_hook, self.make_equiv_action = hook, hook.make_equiv_action
            began = time.monotonic()
            try:
                with torch.no_grad():
                    trajectories = super().rollout(reset=True)
                result = hook.finish(trajectories)
            finally:
                self.decision_hook, self.make_equiv_action = old_hook, old_move
            # Labels are accessed only after the policy, complete continuation and
            # discovered-graph historical return have finished.
            scan, target_path = self.env.gt_trajs[result['instr_id']]
            if scan != result['scan_id']:
                raise ValueError('completed rollout scene differs from evaluator identity')
            result['metrics'] = {k: float(v) for k, v in self.env._eval_item(scan, result['path'], target_path).items()}
            if result['metrics']['success'] not in (0, 1) or not 0 <= result['metrics']['spl'] <= 1:
                raise ValueError('evaluator success/SPL must be finite fractions')
            result['rollout_seconds'] = time.monotonic()-began
            if reference is not None:
                result = compact_branch(result, reference)
            return result

        def test(self, use_dropout=False, feedback='argmax', **kwargs):
            nonlocal attempted_tasks, resumed_tasks
            if use_dropout or feedback != 'argmax' or kwargs:
                raise ValueError('probe cannot change frozen inference settings')
            for model in self.models:
                model.eval()
            original_data = self.env.data
            self.results = {}
            def execute(task, row, condition, **branch_kw):
                nonlocal attempted_tasks, resumed_tasks
                expected_tasks.append(task)
                cached = store.get(task)
                if cached is not None:
                    resumed_tasks += 1
                    return cached
                boundary()
                try:
                    result = self._probe_rollout(row, condition, **branch_kw)
                    store.put(task, result)
                except Exception as error:
                    store.fail(task, error)
                    raise
                attempted_tasks += 1
                print(json.dumps({'probe_task_complete': task, 'new_tasks': attempted_tasks,
                                  'resumed_tasks': resumed_tasks, 'seconds': time.monotonic()-start}), flush=True)
                return result
            try:
                for row in sorted(original_data, key=lambda x: x['instr_id']):
                    for condition in spec['probe']['conditions']:
                        task = {'kind': 'reference', 'instr_id': row['instr_id'], 'condition': condition}
                        reference = execute(task, row, condition)
                        references.append(reference)
                        if condition == 'natural':
                            self.results[row['instr_id']] = {'instr_id': row['instr_id'], 'path': reference['path'], 'details': {}}
                        for step in select_state_steps(reference['states'], condition):
                            for action in candidate_actions(reference['states'][step]):
                                task = {'kind': 'branch', 'instr_id': row['instr_id'], 'condition': condition,
                                        'target_step': step, 'target_action': action}
                                branches.append(execute(task, row, condition, reference=reference,
                                                        target_step=step, target_action=action))
            finally:
                self.env.data = original_data
                self.env.reset_epoch(shuffle=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.output_dir/f'natural-reference-attempt-{time.time_ns()}.raw.json'
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_signals[sig] = signal.signal(sig, stop)
        upstream.GMapNavAgent, run_duet.select_partition = Agent, partition
        run_duet.parse_cli = lambda: SimpleNamespace(mode='baseline', config=args.config, split=args.split,
            output=raw, cache=None, head=None, limit=args.limit, seed=args.seed)
        run_duet.main()
        if code_identity() != source or file_sha256(args.experiment) != provenance['experiment_sha256']:
            raise ValueError('source/config changed while probe was running')
        store.close(expected_tasks)
        report = {'schema': 'e3_continuation_probe_summary_v1', 'complete': True,
            'label_only': True, 'training_only': True, 'selection': selection, 'provenance': provenance,
            'backup_kind': backup_kind, 'backup_backend': args.backup_backend, 'backup_mount_identity': backup_mount,
            'task_manifest_sha256': file_sha256(args.output_dir/'tasks/manifest.json'),
            'reference_rollouts': len(references), 'branch_rollouts': len(branches),
            'anchor_checks': sum(b['is_anchor'] for b in branches),
            'all_anchor_checks_passed': all(b.get('anchor_metrics_equal') for b in branches if b['is_anchor']),
            'all_prefix_checks_passed': all(len(b['prefix_checks']) == b['target_step']+1 for b in branches),
            'max_prefix_absolute_logit_error': max((c['max_absolute_logit_error'] for b in branches for c in b['prefix_checks']), default=0.0),
            'opportunity': opportunity_summary(references, branches),
            'resources': {'this_attempt_seconds': time.monotonic()-start, 'new_tasks': attempted_tasks,
                'resumed_tasks': resumed_tasks, 'recorded_all_rollout_seconds': sum(r['rollout_seconds'] for r in references+branches),
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'gpu': torch.cuda.get_device_name(0)}}
        destination = args.output_dir/'probe-summary.json'
        if not destination.exists() and (args.backup_dir/'probe-summary.json').exists():
            verified_copy(args.backup_dir/'probe-summary.json', destination)
        # Resume of an already completed probe leaves its original summary immutable.
        if destination.exists():
            previous = json.loads(destination.read_text())
            if (not previous.get('complete') or previous['provenance'] != provenance
                    or previous['selection'] != selection or previous['opportunity'] != report['opportunity']
                    or previous['task_manifest_sha256'] != report['task_manifest_sha256']):
                raise ValueError('completed probe summary differs')
        else:
            atomic_json(destination, report)
        check_backup()
        verified_copy(destination, args.backup_dir/'probe-summary.json')
        verified_copy(raw, args.backup_dir/raw.name)
        print(json.dumps({'summary': str(destination), 'opportunity': report['opportunity']}, indent=2), flush=True)
    except BaseException as error:
        status = {'complete': False, 'error': str(error), 'resumable_tasks': len(store.manifest['tasks']) if store else 0,
                  'new_tasks': attempted_tasks, 'resumed_tasks': resumed_tasks, 'time_ns': time.time_ns()}
        status_path = args.output_dir/f'attempt-status-{status["time_ns"]}.json'
        atomic_json(status_path, status)
        try:
            check_backup()
            verified_copy(status_path, args.backup_dir/status_path.name)
        except Exception:
            pass
        raise
    finally:
        upstream.GMapNavAgent, run_duet.parse_cli, run_duet.select_partition = original_class, original_parse, original_partition
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
