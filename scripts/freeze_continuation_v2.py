#!/usr/bin/env python3
"""Publish all five completed E3 arms together before any validation access.

This command reads training artifacts only. Failed cached-development candidates
remain the prescribed final/strictest heads and are marked ineligible. It never
opens a validation ledger, changes a gate, or chooses another checkpoint.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

import torch
import train_continuation_v2 as training
import run_continuation_v2 as navigation
from vln_improve.continuation_learning import ContinuationComparator
from vln_improve.continuation_v2 import SCHEMA as DATA_SCHEMA, SCHEDULES
from vln_improve.intervention_runtime import verified_copy
from vln_improve.pipeline import backup_mount_identity, validate_backup_root, validate_separate_roots
from vln_improve.protocol import file_sha256, object_sha256

SCHEMA = 'e3_continuation_joint_freeze_v1'
ARMS = ('relative-history', 'absolute-history', 'teacher-history', 'relative-nohistory', 'relative-olddata')
FREEZER_FILES = ('scripts/freeze_continuation_v2.py', 'src/vln_improve/intervention_runtime.py',
                 'src/vln_improve/pipeline.py', 'src/vln_improve/protocol.py')


def require(condition, message):
    if not condition: raise ValueError(message)


def read_json(path):
    def reject(value): raise ValueError('nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), label+' is not a SHA256')


def near(a, b, label):
    require(type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b)
            and abs(a-b) <= 1e-12, label+' differs')


def verify_selection_history(report, config, dev_count):
    expected = [(epoch, sr, spl) for epoch in config['candidate_epochs']
                for sr in config['sr_thresholds'] for spl in config['spl_thresholds']]
    history = report['dev_history']
    require([(r['epoch'], r['sr_threshold'], r['spl_threshold']) for r in history] == expected,
            'completed development epoch/gate inventory differs')
    eligible = []
    for candidate in history:
        conditions = candidate['conditions']
        require(set(conditions) == set(SCHEDULES), 'development condition inventory differs')
        for value in conditions.values():
            require(value['episodes'] == dev_count, 'development condition episode count differs')
            for name in ('successes', 'baseline_successes', 'interventions'):
                require(type(value[name]) is int and 0 <= value[name] <= dev_count, 'invalid development '+name)
            near(value['sr'], value['successes']/dev_count, 'development SR')
            near(value['baseline_sr'], value['baseline_successes']/dev_count, 'development baseline SR')
            for name, ceiling in (('spl', 'sr'), ('baseline_spl', 'baseline_sr')):
                require(type(value[name]) in (int, float) and math.isfinite(value[name])
                        and 0 <= value[name] <= value[ceiling]+1e-12, 'invalid development SPL')
        natural = conditions['natural']
        passes = (natural['sr'] >= natural['baseline_sr']-1e-12
                  and natural['spl'] >= natural['baseline_spl']-1e-12)
        require(candidate['eligible'] is passes, 'recorded development eligibility differs from natural gate')
        hard = [conditions[name] for name in ('early_two', 'late_three')]
        require(candidate['hard_net_successes'] == sum(v['successes']-v['baseline_successes'] for v in hard),
                'recorded hard net successes differ')
        near(candidate['hard_spl'], sum(v['spl'] for v in hard)/2, 'recorded hard SPL')
        require(candidate['interventions'] == sum(v['interventions'] for v in conditions.values()),
                'recorded intervention count differs')
        if passes: eligible.append(candidate)
    def key(item):
        return (item['hard_net_successes'], item['hard_spl'], -item['interventions'],
                -item['epoch'], item['sr_threshold'], item['spl_threshold'])
    best = max(eligible, key=key) if eligible else None
    require(report['best_selection'] == best, 'training report did not retain the fixed best development candidate')
    return best


def verify_data_identity(head, report, config, experiment, experiment_sha):
    provenance = head['provenance']
    require(head.get('scope') == provenance.get('scope') == 'training_experiment', 'head is not a formal training experiment')
    require(head['data_identity'] == report['data_identity'], 'head/report data identities differ')
    identities = report['data_identity']
    for kind, split in (('fit', 'train_fit'), ('dev', 'train_dev')):
        identity = identities[kind]; selection = identity['selection']; collected = identity['provenance']
        digest(identity['manifest_sha256'], kind+' manifest SHA')
        digest(identity['bundle_inventory_sha256'], kind+' bundle inventory SHA')
        require(provenance[kind+'_manifest_sha256'] == identity['manifest_sha256']
                and provenance[kind+'_selection'] == selection and provenance[kind+'_provenance'] == collected,
                'head provenance does not bind '+kind+' cache')
        ids, scenes = selection['instr_ids'], selection['scan_ids']
        expected = experiment['instructions'][split]
        require(type(selection['smoke']) is bool and selection['smoke'] is False
                and selection['split'] == split and selection['seed'] == 0
                and len(ids) == expected == selection['count'] and len(set(ids)) == len(ids)
                and scenes and len(set(scenes)) == len(scenes) and selection['conditions'] == list(SCHEDULES),
                'formal '+kind+' selection differs from fixed experiment')
        require(collected.get('schema') == DATA_SCHEMA and collected.get('feature_dtype') == 'float16'
                and collected['experiment_sha256'] == experiment_sha, 'collection experiment/schema differs')
        for name, expected_pin in experiment['asset_pins'].items():
            require(collected[name] == expected_pin, 'collection asset pin differs: '+name)
    a, b = identities['fit']['selection'], identities['dev']['selection']
    require(not set(a['instr_ids']).intersection(b['instr_ids']) and not set(a['scan_ids']).intersection(b['scan_ids']),
            'fit/dev instruction or scene overlap')
    require(identities['fit']['provenance'] == identities['dev']['provenance'], 'fit/dev collection provenance differs')
    support = config['support_gate']; fixed = experiment['training_plan']
    count = fixed['required_fit_rescue_instructions_before_formal_training']
    scans = fixed['required_fit_rescue_scans_before_formal_training']
    require(support['passed'] is True and support['required_instructions'] == count and support['required_scans'] == scans
            and type(support['unique_rescuable_fit_instructions']) is int
            and count <= support['unique_rescuable_fit_instructions'] <= len(a['instr_ids'])
            and type(support['rescuable_fit_scans']) is int
            and scans <= support['rescuable_fit_scans'] <= len(a['scan_ids']), 'formal fit support gate was not met')
    return {'fit': identities['fit'], 'dev': identities['dev']}


def verify_arm(root, arm, experiment, experiment_sha, source):
    head_path, report_path = root/arm/'selected-head.pt', root/arm/'training-report.json'
    require(head_path.is_file() and report_path.is_file() and not head_path.is_symlink() and not report_path.is_symlink(),
            'all five arms need regular selected-head and completed report files: '+arm)
    head_sha, report_sha = file_sha256(head_path), file_sha256(report_path)
    raw = head_path.read_bytes()
    import hashlib
    require(hashlib.sha256(raw).hexdigest() == head_sha, 'head changed while reading')
    head = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    report = read_json(report_path)
    require(report.get('schema') == training.SCHEMA and report.get('status') == 'complete'
            and report.get('arm') == arm and report.get('pending_dev') is False, 'training report is not complete for '+arm)
    require(head.get('schema') == 'e3_continuation_head_v2' and head.get('arm') == arm, 'portable head schema/arm differs')
    config = report['config']; fixed = experiment['training_plan']
    resources = report['resources']
    allocated, reserved = resources['cuda_peak_allocated_bytes'], resources['cuda_peak_reserved_bytes']
    require(type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved,
            'completed report lacks CUDA allocation evidence')
    require(head['config'] == config and config['seed'] == 0 and config['device'] == 'cuda'
            and config.get('engineering_smoke') is False, 'freeze requires CUDA seed0 formal training')
    for name in ('epochs', 'candidate_epochs', 'sr_thresholds', 'spl_thresholds'):
        require(config[name] == fixed[name], 'training configuration differs from fixed '+name)
    require(config['feature_dim'] == 1549 and report['completed_epochs'] == config['epochs']
            and [h['epoch'] for h in report['training_history']] == list(range(1, config['epochs']+1)),
            'training epoch or feature inventory incomplete')
    require(report['code_identity'] == head['code_identity'] == head['provenance']['source_files'] == source
            and head['provenance']['source_sha256'] == object_sha256(source), 'head/report training source differs from current frozen source')
    require(head['provenance']['training_config_sha256'] == object_sha256(config), 'head training configuration SHA differs')
    data = verify_data_identity(head, report, config, experiment, experiment_sha)
    best = verify_selection_history(report, config, len(data['dev']['selection']['instr_ids']))
    selected = report['selected_checkpoint']; final = report['final_checkpoint']
    require(selected['export_file'] == 'selected-head.pt' and selected['head_sha256'] == head_sha,
            'selected exported head SHA differs from training report')
    require(head['epoch'] == selected['epoch'] and head['global_step'] == selected['global_step']
            and 0 < selected['global_step'] <= report['global_step']
            and final['epoch'] == config['epochs'] and final['global_step'] == report['global_step'],
            'selected/final training checkpoint position differs')
    require(selected['eligible'] is (best is not None), 'selected checkpoint eligibility differs')
    if best is not None:
        status = 'passed_cached_dev_gate'
        thresholds = {'sr': best['sr_threshold'], 'spl': best['spl_threshold']}
        require(head['selection'] == best and head['epoch'] == best['epoch'], 'head is not the recorded best cached-dev candidate')
    else:
        status = 'failed_cached_dev_gate_fixed_final_strictest'
        thresholds = {'sr': max(config['sr_thresholds']), 'spl': max(config['spl_thresholds'])}
        require(head['selection'] is None and all(selected[k] == final[k] for k in
                ('checkpoint_id', 'epoch', 'global_step', 'head_sha256', 'head_relative_path')),
                'failed candidate must retain fixed final checkpoint')
    require(head['selection_status'] == selected['status'] == status
            and head['thresholds'] == thresholds
            and {'sr': selected['sr_threshold'], 'spl': selected['spl_threshold']} == thresholds,
            'exported head gate/status differs from frozen selection')
    mode, history, _ = training.ARMS[arm]
    require(head['model_config'] == {'feature_dim': 1549, 'hidden_dim': config['hidden_dim'], 'mode': mode, 'history': history},
            'head model/arm configuration differs')
    weights = head['state_dict']
    require(isinstance(weights, dict) and all(isinstance(t, torch.Tensor) and t.dtype == torch.float32
            and bool(torch.isfinite(t).all()) for t in weights.values()), 'head contains invalid/nonfinite weights')
    model = ContinuationComparator(**head['model_config'])
    model.load_state_dict(weights, strict=True)
    require(file_sha256(head_path) == head_sha and file_sha256(report_path) == report_sha, 'training artifacts changed during freeze verification')
    entry = {'arm': arm, 'head_file': str(Path(arm)/'selected-head.pt'), 'head_sha256': head_sha,
        'training_report_file': str(Path(arm)/'training-report.json'), 'training_report_sha256': report_sha,
        'epoch': head['epoch'], 'global_step': head['global_step'], 'thresholds': thresholds,
        'model_config': head['model_config'], 'eligible': selected['eligible'], 'selection_status': status,
        'selection': best, 'fit_manifest_sha256': data['fit']['manifest_sha256'],
        'dev_manifest_sha256': data['dev']['manifest_sha256'], 'training_config_sha256': object_sha256(config),
        'training_source_sha256': object_sha256(source), 'training_source_files': source}
    return entry, data, config, [(head_path, head_sha), (report_path, report_sha)]



def copy_artifact(source, target, expected_sha, check_backup):
    """Copy verified original bytes, then atomically link without overwriting."""
    check_backup()
    require(not target.parent.is_symlink() and not target.is_symlink(), 'frozen artifact path is a symlink')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(file_sha256(target) == expected_sha, 'existing frozen artifact differs')
        return
    temporary = target.with_name('.freeze-'+uuid.uuid4().hex+'.tmp')
    try:
        verified_copy(source, temporary)
        require(file_sha256(temporary) == expected_sha, 'source changed during artifact copy')
        check_backup()
        try: os.link(temporary, target)
        except FileExistsError:
            require(not target.is_symlink() and file_sha256(target) == expected_sha, 'existing frozen artifact differs')
        directory = os.open(target.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
        check_backup()
        require(file_sha256(target) == expected_sha, 'frozen artifact readback differs')
    finally:
        temporary.unlink(missing_ok=True)


def publish_plan(path, plan):
    raw = (json.dumps(plan, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    require(not path.is_symlink(), 'frozen plan path is a symlink')
    if path.exists():
        require(path.read_bytes() == raw, 'existing frozen plan differs; use a new freeze directory')
        return
    temporary = path.with_name('.plan-'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def freeze(training_root, output_dir, backup_root, experiment_path, check_backup):
    """All verification precedes publication; retries may reuse identical bytes."""
    training_root, output_dir, backup_root, experiment_path = map(Path, (training_root, output_dir, backup_root, experiment_path))
    for a, b in ((output_dir, backup_root), (training_root, output_dir), (training_root, backup_root)):
        validate_separate_roots(a, b)
    check_backup()
    spec = read_json(experiment_path); experiment_sha = file_sha256(experiment_path)
    require(spec['schema'] == DATA_SCHEMA and spec['seed'] == 0 and spec['training_plan']['arms'] == list(ARMS)
            and spec['instructions']['train_fit'] in (512, 2048) and spec['instructions']['train_dev'] == 128,
            'experiment is not the complete fixed five-arm protocol')
    source = training.code_identity()
    navigation_source, navigation_sha = navigation.code_identity()
    freezer_source = {name: file_sha256(ROOT/name) for name in FREEZER_FILES}
    entries, shared_data, shared_config, originals = [], None, None, []
    for arm in ARMS:
        entry, data, config, files = verify_arm(training_root, arm, spec, experiment_sha, source)
        require(shared_data is None or shared_data == data, 'five arms use different fit/dev caches')
        require(shared_config is None or shared_config == config, 'five arms use different training configurations')
        shared_data, shared_config = data, config
        entries.append(entry); originals.extend(files)
    plan = {'schema': SCHEMA, 'status': 'frozen', 'training_only': True, 'seed': 0, 'arms': list(ARMS),
        'training_root': str(training_root.resolve()), 'output_dir': str(output_dir.resolve()),
        'backup_root': str(backup_root.resolve()), 'experiment_file': 'experiment.json',
        'experiment_sha256': experiment_sha, 'data_identity': shared_data, 'training_config': shared_config,
        'training_source_files': source, 'training_source_sha256': object_sha256(source),
        'navigation_source_files': navigation_source, 'navigation_source_sha256': navigation_sha,
        'freezer_source_files': freezer_source, 'freezer_source_sha256': object_sha256(freezer_source),
        'entries': entries, 'policy': 'All five selected head byte streams and thresholds are frozen together. Ineligible heads retain the prescribed final/strictest candidate. No official-validation access is registered or evaluated by this plan.'}
    # Refuse conflicts before copying any artifact, including a prior plan.
    for directory in (output_dir, backup_root):
        path = directory/'plan.json'
        if path.exists(): require(read_json(path) == plan, 'existing frozen plan differs; use a new freeze directory')
    def unchanged():
        require(training.code_identity() == source and navigation.code_identity() == (navigation_source, navigation_sha)
                and file_sha256(experiment_path) == experiment_sha
                and {name: file_sha256(ROOT/name) for name in FREEZER_FILES} == freezer_source,
                'source or experiment changed while freezing')
        require(all(file_sha256(path) == digest_value for path, digest_value in originals), 'training artifact changed while freezing')
    unchanged()
    for directory in (backup_root, output_dir):
        check_backup()
        directory.mkdir(parents=True, exist_ok=True)
        copy_artifact(experiment_path, directory/'experiment.json', experiment_sha, check_backup)
        check_backup()
        for entry in entries:
            for field in ('head_file', 'training_report_file'):
                relative = entry[field]; source_path = training_root/relative
                expected = entry['head_sha256'] if field == 'head_file' else entry['training_report_sha256']
                copy_artifact(source_path, directory/relative, expected, check_backup)
    unchanged(); check_backup()
    for directory in (backup_root, output_dir):
        require(file_sha256(directory/'experiment.json') == experiment_sha, 'frozen experiment readback differs')
        for entry in entries:
            require(file_sha256(directory/entry['head_file']) == entry['head_sha256']
                    and file_sha256(directory/entry['training_report_file']) == entry['training_report_sha256'],
                    'complete frozen artifact inventory readback differs')
    publish_plan(backup_root/'plan.json', plan)
    check_backup(); unchanged()
    publish_plan(output_dir/'plan.json', plan)
    check_backup()
    require(file_sha256(output_dir/'plan.json') == file_sha256(backup_root/'plan.json'), 'frozen plan backup readback differs')
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [p.expanduser().resolve() for p in (args.training_root, args.output_dir, args.backup_root, args.experiment)]
    backup = paths[2]
    mount = backup_mount_identity(backup, backend='filesystem')
    def check(): validate_backup_root(backup, backend='filesystem', expected_identity=mount)
    torch.set_num_threads(1)
    plan = freeze(*paths, check_backup=check)
    print(json.dumps({'status': plan['status'], 'plan': str(paths[1]/'plan.json'),
                      'arms': len(plan['entries']), 'eligible': sum(e['eligible'] for e in plan['entries'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
