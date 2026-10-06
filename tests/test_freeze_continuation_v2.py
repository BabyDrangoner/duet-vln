"""Five-arm publication is all-or-none and retains failed fixed candidates."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts import freeze_continuation_v2 as freeze
from vln_improve.continuation_learning import ContinuationComparator
from vln_improve.protocol import file_sha256, object_sha256


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_training(tmp_path, failed_arm='relative-olddata'):
    root, output, backup = tmp_path/'training', tmp_path/'frozen', tmp_path/'backup'
    root.mkdir(); backup.mkdir()
    experiment = tmp_path/'experiment.json'
    experiment.write_bytes((freeze.ROOT/'configs/e3_continuation_v2.json').read_bytes())
    spec = json.loads(experiment.read_text()); experiment_sha = file_sha256(experiment)
    source = freeze.training.code_identity()
    provenance = dict(spec['asset_pins'], schema=freeze.DATA_SCHEMA, feature_dtype='float16',
                      experiment_sha256=experiment_sha)
    identities = {}
    for kind, n, scenes, split in [('fit', 512, 20, 'train_fit'), ('dev', 128, 12, 'train_dev')]:
        selection = {'split': split, 'count': n, 'instr_ids': [kind+str(i) for i in range(n)],
            'scan_ids': [kind+'-scene-'+str(i) for i in range(scenes)], 'smoke': False, 'seed': 0,
            'conditions': list(freeze.SCHEDULES)}
        identities[kind] = {'manifest_sha256': ('a' if kind == 'fit' else 'b')*64,
            'bundle_inventory_sha256': ('c' if kind == 'fit' else 'd')*64,
            'selection': selection, 'provenance': provenance}
    config = freeze.training.default_config()
    config.update(hidden_dim=8, engineering_smoke=False, support_gate={
        'unique_rescuable_fit_instructions': 40, 'rescuable_fit_scans': 15,
        'required_instructions': 32, 'required_scans': 12, 'passed': True,
        'scope': 'engineering_support_gate_not_statistical_power_or_navigation_performance'})
    for arm in freeze.ARMS:
        local = root/arm; local.mkdir()
        failed = arm == failed_arm
        mode, history, _ = freeze.training.ARMS[arm]
        model = ContinuationComparator(hidden_dim=8, mode=mode, history=history)
        candidates = []
        for epoch in config['candidate_epochs']:
            for sr in config['sr_thresholds']:
                conditions = {}
                for condition in freeze.SCHEDULES:
                    natural = condition == 'natural'; success = 128 if natural else 100
                    conditions[condition] = {'episodes': 128, 'successes': success,
                        'baseline_successes': 128 if natural else 96, 'sr': success/128,
                        'baseline_sr': 1. if natural else .75, 'spl': (.97 if failed else .98) if natural else .6,
                        'baseline_spl': .98 if natural else .5, 'interventions': 0 if natural else 4}
                candidates.append({'epoch': epoch, 'sr_threshold': sr, 'spl_threshold': 0.,
                    'conditions': conditions, 'eligible': not failed, 'hard_net_successes': 8,
                    'hard_spl': .6, 'interventions': 12})
        best = None if failed else copy.deepcopy(candidates[4])
        selected_epoch = 20 if failed else 5; selected_step = selected_epoch*10
        thresholds = {'sr': .4, 'spl': 0.}
        status = 'failed_cached_dev_gate_fixed_final_strictest' if failed else 'passed_cached_dev_gate'
        data = dict(copy.deepcopy(identities), retained_records=1000, full_records=1000, records_content_sha256='e'*64)
        head_provenance = {'scope': 'training_experiment', 'source_files': source,
            'source_sha256': object_sha256(source), 'training_config_sha256': object_sha256(config)}
        for kind in ('fit', 'dev'):
            head_provenance[kind+'_manifest_sha256'] = data[kind]['manifest_sha256']
            head_provenance[kind+'_selection'] = copy.deepcopy(data[kind]['selection'])
            head_provenance[kind+'_provenance'] = copy.deepcopy(data[kind]['provenance'])
        head = {'schema': 'e3_continuation_head_v2', 'arm': arm, 'model_config': model.config,
            'state_dict': model.state_dict(), 'epoch': selected_epoch, 'global_step': selected_step,
            'data_identity': data, 'code_identity': source, 'config': copy.deepcopy(config),
            'provenance': head_provenance, 'thresholds': thresholds, 'selection': best,
            'scope': 'training_experiment', 'selection_status': status}
        torch.save(head, local/'selected-head.pt'); head_sha = file_sha256(local/'selected-head.pt')
        final = {'checkpoint_id': 'final', 'epoch': 20, 'global_step': 200,
            'head_sha256': head_sha if failed else 'f'*64, 'head_relative_path': 'snapshots/final/head.pt'}
        selected = dict(final if failed else {'checkpoint_id': 'best', 'epoch': 5, 'global_step': 50,
            'head_sha256': head_sha, 'head_relative_path': 'snapshots/best/head.pt'},
            export_file='selected-head.pt', sr_threshold=.4, spl_threshold=0., status=status, eligible=not failed)
        report = {'schema': freeze.training.SCHEMA, 'status': 'complete', 'arm': arm,
            'config': config, 'global_step': 200, 'completed_epochs': 20, 'pending_dev': False,
            'data_identity': data, 'code_identity': source,
            'training_history': [{'epoch': epoch} for epoch in range(1, 21)],
            'dev_history': candidates, 'best_selection': best, 'selected_checkpoint': selected,
            'final_checkpoint': final, 'resources': {'cuda_peak_allocated_bytes': 1024, 'cuda_peak_reserved_bytes': 2048}}
        (local/'training-report.json').write_text(json.dumps(report))
    return root, output, backup, experiment


def change_artifact(root, mutation, arm='relative-history'):
    local = root/arm; head_path = local/'selected-head.pt'; report_path = local/'training-report.json'
    head = torch.load(head_path, map_location='cpu', weights_only=True); report = json.loads(report_path.read_text())
    mutation(head, report)
    torch.save(head, head_path)
    report['selected_checkpoint']['head_sha256'] = file_sha256(head_path)
    if arm == 'relative-olddata': report['final_checkpoint']['head_sha256'] = file_sha256(head_path)
    report_path.write_text(json.dumps(report))


def run(paths, check=lambda: None):
    return freeze.freeze(*paths, check_backup=check)


def test_publishes_all_five_original_byte_streams_and_failed_head_without_reselection(tmp_path):
    paths = make_training(tmp_path); root, output, backup, _ = paths
    plan = run(paths)
    assert plan['arms'] == list(freeze.ARMS) and len(plan['entries']) == 5
    assert (plan['navigation_source_files'], plan['navigation_source_sha256']) == freeze.navigation.code_identity()
    assert [e['eligible'] for e in plan['entries']] == [True, True, True, True, False]
    assert plan['entries'][-1]['epoch'] == 20 and plan['entries'][-1]['thresholds'] == {'sr': .4, 'spl': 0.}
    assert (output/'plan.json').read_bytes() == (backup/'plan.json').read_bytes()
    for arm in freeze.ARMS:
        for file in ('selected-head.pt', 'training-report.json'):
            assert (root/arm/file).read_bytes() == (output/arm/file).read_bytes() == (backup/arm/file).read_bytes()
    original = (output/'plan.json').read_bytes()
    assert run(paths) == plan
    assert (output/'plan.json').read_bytes() == original


def test_missing_fifth_arm_cannot_publish_a_four_arm_plan(tmp_path):
    paths = make_training(tmp_path); (paths[0]/freeze.ARMS[-1]/'selected-head.pt').unlink()
    with pytest.raises(ValueError, match='all five arms'): run(paths)
    assert not (paths[1]/'plan.json').exists() and not (paths[2]/'plan.json').exists()
    assert not list(paths[2].glob('*/selected-head.pt'))


@pytest.mark.parametrize('defect,match', [
    ('paused', 'not complete'), ('cuda_evidence', 'CUDA allocation evidence'), ('cpu', 'CUDA seed0'), ('seed', 'CUDA seed0'),
    ('engineering', 'formal training experiment'), ('smoke', 'formal fit selection'),
    ('mode', 'model/arm'), ('nan', 'nonfinite weights'), ('threshold', 'gate/status'),
    ('eligibility', 'eligibility differs'), ('unselected_best', 'fixed best'),
    ('missing_candidate', 'epoch/gate inventory'), ('source', 'current frozen source'),
    ('weak_support', 'support gate'), ('epoch', 'checkpoint position'),
    ('data_overlap', 'overlap'), ('manifest', 'different fit/dev caches'),
    ('candidate_lie', 'eligibility differs'), ('different_config', 'different training configurations')])
def test_rejects_semantic_training_or_selection_corruption_even_with_recomputed_head_sha(tmp_path, defect, match):
    paths = make_training(tmp_path)
    def change(head, report):
        if defect == 'paused': report['status'] = 'paused'
        elif defect == 'cuda_evidence': report['resources']['cuda_peak_allocated_bytes'] = None
        elif defect in ('cpu', 'seed', 'weak_support', 'different_config'):
            if defect == 'cpu': report['config']['device'] = 'cpu'
            elif defect == 'seed': report['config']['seed'] = 1
            elif defect == 'different_config': report['config']['lr'] *= 2
            else: report['config']['support_gate']['unique_rescuable_fit_instructions'] = 31
            head['config'] = copy.deepcopy(report['config'])
            head['provenance']['training_config_sha256'] = object_sha256(head['config'])
        elif defect == 'engineering': head['scope'] = head['provenance']['scope'] = 'engineering_smoke'
        elif defect == 'smoke':
            head['data_identity']['fit']['selection']['smoke'] = True
            report['data_identity'] = copy.deepcopy(head['data_identity'])
            head['provenance']['fit_selection'] = copy.deepcopy(head['data_identity']['fit']['selection'])
        elif defect == 'mode': head['model_config']['mode'] = 'absolute'
        elif defect == 'nan': next(iter(head['state_dict'].values())).flatten()[0] = torch.nan
        elif defect == 'threshold':
            head['thresholds']['sr'] = .2; report['selected_checkpoint']['sr_threshold'] = .2
        elif defect == 'eligibility': report['selected_checkpoint']['eligible'] = False
        elif defect == 'unselected_best': report['best_selection'] = copy.deepcopy(report['dev_history'][0])
        elif defect == 'missing_candidate': report['dev_history'].pop()
        elif defect == 'source':
            altered = copy.deepcopy(report['code_identity']); altered['scripts/train_continuation_v2.py'] = '1'*64
            report['code_identity'] = head['code_identity'] = head['provenance']['source_files'] = altered
            head['provenance']['source_sha256'] = object_sha256(altered)
        elif defect == 'epoch': head['epoch'] += 1
        elif defect == 'data_overlap':
            selection = head['data_identity']['dev']['selection']
            selection['scan_ids'][0] = head['data_identity']['fit']['selection']['scan_ids'][0]
            head['provenance']['dev_selection'] = copy.deepcopy(selection)
            report['data_identity'] = copy.deepcopy(head['data_identity'])
        elif defect == 'manifest':
            head['data_identity']['fit']['manifest_sha256'] = '2'*64
            report['data_identity'] = copy.deepcopy(head['data_identity'])
            head['provenance']['fit_manifest_sha256'] = '2'*64
        elif defect == 'candidate_lie': report['dev_history'][0]['eligible'] = False
    change_artifact(paths[0], change)
    with pytest.raises(ValueError, match=match): run(paths)
    assert not (paths[1]/'plan.json').exists() and not (paths[2]/'plan.json').exists()


def test_failed_candidate_cannot_be_replaced_with_an_earlier_or_less_strict_head(tmp_path):
    paths = make_training(tmp_path)
    def change(head, report):
        head['epoch'] = 5; head['global_step'] = 50
        report['selected_checkpoint']['epoch'] = 5; report['selected_checkpoint']['global_step'] = 50
    change_artifact(paths[0], change, arm='relative-olddata')
    with pytest.raises(ValueError, match='fixed final checkpoint'): run(paths)


def test_preexisting_different_head_bytes_are_never_replaced(tmp_path):
    paths = make_training(tmp_path); target = paths[2]/freeze.ARMS[0]/'selected-head.pt'
    target.parent.mkdir(); target.write_bytes(b'existing different frozen head')
    with pytest.raises(ValueError, match='existing frozen artifact differs'): run(paths)
    assert target.read_bytes() == b'existing different frozen head'
    assert not (paths[1]/'plan.json').exists() and not (paths[2]/'plan.json').exists()


def test_backup_failure_during_copy_never_publishes_plan(tmp_path):
    paths = make_training(tmp_path); calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls >= 6: raise RuntimeError('mount disappeared')
    with pytest.raises(RuntimeError, match='mount disappeared'): run(paths, check)
    assert not (paths[1]/'plan.json').exists() and not (paths[2]/'plan.json').exists()


def test_changed_source_report_during_copy_cannot_publish_plan(tmp_path, monkeypatch):
    paths = make_training(tmp_path); original = freeze.verified_copy; fired = False
    def copy_and_change(source, target):
        nonlocal fired
        original(source, target)
        if not fired:
            fired = True
            (paths[0]/freeze.ARMS[-1]/'training-report.json').write_text('{}')
    monkeypatch.setattr(freeze, 'verified_copy', copy_and_change)
    with pytest.raises(ValueError, match='artifact changed|source changed during artifact copy'): run(paths)
    assert not (paths[1]/'plan.json').exists() and not (paths[2]/'plan.json').exists()


def test_changed_plan_never_overwrites_a_previous_joint_freeze(tmp_path):
    paths = make_training(tmp_path); run(paths)
    before = (paths[1]/'plan.json').read_bytes()
    def change(head, report): report['new_audit_note'] = 'later report version'
    change_artifact(paths[0], change)
    with pytest.raises(ValueError, match='existing frozen plan differs'): run(paths)
    assert (paths[1]/'plan.json').read_bytes() == before


def test_cli_requires_existing_filesystem_backup_with_fixed_mount_identity(tmp_path, monkeypatch):
    paths = make_training(tmp_path); calls = []
    def identify(path, **kwargs):
        calls.append(('identify', path, kwargs)); return {'mount_id': 'frozen'}
    def check(path, **kwargs): calls.append(('check', path, kwargs))
    monkeypatch.setattr(freeze, 'backup_mount_identity', identify)
    monkeypatch.setattr(freeze, 'validate_backup_root', check)
    assert freeze.main(['--training-root', str(paths[0]), '--output-dir', str(paths[1]),
                        '--backup-root', str(paths[2]), '--experiment', str(paths[3])]) == 0
    assert calls[0][2] == {'backend': 'filesystem'}
    assert all(c[2] == {'backend': 'filesystem', 'expected_identity': {'mount_id': 'frozen'}} for c in calls[1:])


def test_identical_symlink_is_not_a_durable_frozen_copy(tmp_path):
    paths = make_training(tmp_path); target = paths[2]/freeze.ARMS[0]/'selected-head.pt'
    target.parent.mkdir(); target.symlink_to(paths[0]/freeze.ARMS[0]/'selected-head.pt')
    with pytest.raises(ValueError, match='symlink'): run(paths)
    assert not (paths[1]/'plan.json').exists() and not (paths[2]/'plan.json').exists()


def test_concurrent_different_artifact_is_never_overwritten(tmp_path, monkeypatch):
    source, target = tmp_path/'source.pt', tmp_path/'target.pt'
    source.write_bytes(b'our frozen bytes'); expected = file_sha256(source)
    original = freeze.verified_copy
    def competing_publish(src, tmp):
        original(src, tmp)
        target.write_bytes(b'competing frozen bytes')
    monkeypatch.setattr(freeze, 'verified_copy', competing_publish)
    with pytest.raises(ValueError, match='existing frozen artifact differs'):
        freeze.copy_artifact(source, target, expected, lambda: None)
    assert target.read_bytes() == b'competing frozen bytes'


def test_navigation_code_change_after_verification_prevents_joint_publication(tmp_path, monkeypatch):
    paths = make_training(tmp_path); original = freeze.navigation.code_identity; calls = 0
    def changing_identity():
        nonlocal calls
        calls += 1
        files, digest = original()
        return (files, digest) if calls == 1 else (files, '1'*64)
    monkeypatch.setattr(freeze.navigation, 'code_identity', changing_identity)
    with pytest.raises(ValueError, match='source or experiment changed'): run(paths)
    assert not (paths[1]/'plan.json').exists() and not (paths[2]/'plan.json').exists()
