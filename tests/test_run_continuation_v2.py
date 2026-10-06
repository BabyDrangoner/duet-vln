"""Evaluation contracts and a synthetic end-to-end navigation driver."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_continuation_probe_integration_audit import KnownGraph, NoGoalObservation
from vln_improve.continuation_learning import ContinuationComparator
from vln_improve.protocol import file_sha256, object_sha256
from vln_improve.study_ledger import StudyLedger

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_continuation_v2', ROOT/'scripts/run_continuation_v2.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def payload():
    model = ContinuationComparator(hidden_dim=8)
    return {'schema': module.HEAD_SCHEMA, 'model_config': model.config,
        'state_dict': model.state_dict(), 'arm': 'relative-history', 'epoch': 5,
        'thresholds': {'sr': .1, 'spl': 0.}, 'extra_audit_field': 'allowed',
        'provenance': {'fit_manifest_sha256': 'a'*64, 'scope': 'training_experiment',
            'fit_selection': {'split': 'train_fit', 'smoke': False},
            'dev_selection': {'split': 'train_dev', 'smoke': False}}}


def panel():
    return {'schema': module.SCHEMA, 'complete': True, 'selection': {
        'split': 'train_dev', 'instr_ids': ['a', 'b'], 'scan_ids': ['scene'], 'count': 2,
        'conditions': list(module.SCHEDULES), 'seed': 0, 'smoke': True}, 'provenance': {}}


def test_head_roundtrip_keeps_frozen_thresholds_and_extra_audit_metadata(tmp_path):
    content = payload(); path = tmp_path/'head.pt'; torch.save(content, path)
    model, metadata = module.load_head(path)
    assert metadata['thresholds'] == {'sr': .1, 'spl': 0.}
    assert metadata['extra_audit_field'] == 'allowed'
    assert 'state_dict' not in metadata and not model.training
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, content['state_dict'][name], rtol=0, atol=0)


@pytest.mark.parametrize('defect', ['schema', 'arm', 'features', 'nan', 'threshold', 'fit_hash', 'missing_weight', 'epoch'])
def test_head_rejects_incompatible_or_untrained_payload(tmp_path, defect):
    content = payload()
    if defect == 'schema': content['schema'] = 'endpoint-head'
    elif defect == 'arm': content['arm'] = 'teacher-history'
    elif defect == 'features': content['model_config']['feature_dim'] = 10
    elif defect == 'nan': content['state_dict']['comparison.2.bias'][0] = torch.nan
    elif defect == 'threshold': content['thresholds']['sr'] = float('nan')
    elif defect == 'fit_hash': content['provenance'].pop('fit_manifest_sha256')
    elif defect == 'missing_weight': content['state_dict'].pop('comparison.2.bias')
    else: content['epoch'] = 0
    path = tmp_path/'head.pt'; torch.save(content, path)
    with pytest.raises((ValueError, RuntimeError)):
        module.load_head(path)


def test_identity_builds_zero_head_and_rejects_nonzero_trained_output(tmp_path):
    model, metadata = module.load_head(None, identity=True)
    assert all(not p.detach().ne(0).any() for p in model.comparison[-1].parameters())
    assert metadata['epoch'] == 0
    content = payload(); content['state_dict']['comparison.2.weight'][0, 0] = 1
    path = tmp_path/'head.pt'; torch.save(content, path)
    with pytest.raises(ValueError, match='zero-output'):
        module.load_head(path, identity=True)


def test_formal_validation_rejects_engineering_or_unbound_training_scope():
    metadata = payload()
    module.validate_formal_head(metadata)
    for key, value in (('scope', 'engineering_smoke'), ('fit_selection', {'split': 'train_fit', 'smoke': True}),
                       ('dev_selection', {'split': 'train_dev'})):
        bad = copy.deepcopy(metadata); bad['provenance'][key] = value
        with pytest.raises(ValueError, match='official validation'):
            module.validate_formal_head(bad)


def test_provenance_binds_assets_and_causal_source():
    actual = {key: 'a'*64 for key in module.ASSET_FIELDS}
    actual.update(model_config_sha256='b'*64, partition_seed=0, dev_fraction=.2,
                  upstream_lock={'commit': 'fixed'})
    provenance = dict(actual, source_files={name: file_sha256(ROOT/name) for name in module.CODE_FILES})
    module.validate_provenance(provenance, actual, name='head')
    changed = copy.deepcopy(provenance); changed['feature_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='feature_sha256'):
        module.validate_provenance(changed, actual, name='head')
    changed = copy.deepcopy(provenance); changed['source_files']['src/vln_improve/continuation_v2.py'] = 'c'*64
    with pytest.raises(ValueError, match='causal collection source'):
        module.validate_provenance(changed, actual, name='head')


def test_panel_selection_uses_exact_ids_and_scenes_not_result_labels(tmp_path):
    content = panel(); path = tmp_path/'dataset-manifest.json'; path.write_text(json.dumps(content))
    selected = module.read_panel(path, 'train_dev')
    rows = [{'instr_id': key, 'scan': 'scene'} for key in ('c', 'b', 'a')]
    assert [r['instr_id'] for r in module.select_evaluation_rows(rows, 'train_dev', selected)] == ['a', 'b']
    with pytest.raises(ValueError, match='restricted to train_dev'):
        module.read_panel(path, 'val_unseen')
    with pytest.raises(ValueError, match='outside train_dev'):
        module.select_evaluation_rows(rows[:2], 'train_dev', selected)
    rows[-1]['scan'] = 'other'
    with pytest.raises(ValueError, match='scene inventory'):
        module.select_evaluation_rows(rows, 'train_dev', selected)


@pytest.mark.parametrize('defect', ['incomplete', 'duplicates', 'conditions', 'count', 'split'])
def test_panel_rejects_changed_inventory(tmp_path, defect):
    content = panel()
    if defect == 'incomplete': content['complete'] = False
    elif defect == 'duplicates': content['selection']['instr_ids'] = ['a', 'a']
    elif defect == 'conditions': content['selection']['conditions'] = ['natural']
    elif defect == 'count': content['selection']['count'] = 3
    else: content['selection']['split'] = 'train_fit'
    path = tmp_path/'panel.json'; path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='fixed train_dev panel'):
        module.read_panel(path, 'train_dev')


@pytest.mark.parametrize('split,count', [('train_dev', 2890), ('val_unseen', 2349)])
def test_full_navigation_cannot_silently_become_subset(split, count):
    rows = [{'instr_id': str(i), 'scan': 's'} for i in range(count)]
    assert len(module.select_evaluation_rows(rows, split)) == count
    with pytest.raises(ValueError, match='exactly'):
        module.select_evaluation_rows(rows[:-1], split)


@pytest.fixture
def registration(tmp_path):
    study = json.loads((ROOT/'configs/research_study.json').read_text())
    study_path = tmp_path/'study.json'; study_path.write_text(json.dumps(study))
    head = tmp_path/'head.pt'; head.write_bytes(b'frozen-head')
    experiment = tmp_path/'method.json'; experiment.write_text('{"method":"fixed"}')
    backup = tmp_path/'backup'; backup.mkdir()
    args = SimpleNamespace(split='val_unseen', identity=False, dataset_manifest=None,
        head=head, output=tmp_path/'result.json', access_id='V0100', ledger=tmp_path/'ledger.jsonl',
        study=study_path, experiment=experiment, execution_backup_root=backup, category='pilot', seed=0)
    ledger = StudyLedger(args.ledger, study)
    ledger.register(access_id=args.access_id, category='pilot', variant_id='continuation-frozen',
        config_sha256=file_sha256(experiment), checkpoint_sha256=file_sha256(head), code_sha256='c'*64,
        purpose='unit-test registration', split='val_unseen', seed=0, expected_episodes=2349)
    return args, ledger


def test_validation_registration_binds_code_head_method_seed_and_complete_split(registration):
    args, _ = registration
    assert module.validate_registration(args, 'c'*64)['access_id'] == args.access_id
    for key, value in (('identity', True), ('dataset_manifest', Path('subset.json')), ('head', None),
                       ('seed', 1), ('execution_backup_root', None)):
        changed = copy.copy(args); setattr(changed, key, value)
        with pytest.raises(ValueError):
            module.validate_registration(changed, 'c'*64)
    with pytest.raises(ValueError, match='identity differs'):
        module.validate_registration(args, 'd'*64)
    args.head.write_bytes(b'changed-head')
    with pytest.raises(ValueError, match='identity differs'):
        module.validate_registration(args, 'c'*64)


def test_training_split_does_not_consume_ledger_and_failed_access_cannot_restart(registration):
    args, ledger = registration
    changed = copy.copy(args); changed.split = 'train_dev'
    with pytest.raises(ValueError, match='cannot consume'):
        module.validate_registration(changed, 'c'*64)
    ledger.finish(args.access_id, status='failed', metrics={}, resources={'seconds': 0},
                  decision='test failure', error='test interruption')
    with pytest.raises(ValueError, match='cannot be rerun'):
        module.validate_registration(args, 'c'*64)


def test_execution_claim_is_permanent_even_if_output_name_changes(registration, monkeypatch):
    args, _ = registration
    row = module.validate_registration(args, 'c'*64)
    monkeypatch.setattr(module, 'backup_mount_identity', lambda *a, **k: {'mount': 'fixed'})
    monkeypatch.setattr(module, 'validate_backup_root', lambda *a, **k: 'verified')
    with module.validation_execution(args, row, 'c'*64) as claim:
        module.exclusive_json(args.output, {'result': 'synthetic'})
    assert Path(claim['local']).read_bytes() == Path(claim['backup']).read_bytes()
    args.output = args.output.with_name('different-output.json')
    with pytest.raises(ValueError, match='already executed'):
        with module.validation_execution(args, row, 'c'*64):
            pytest.fail('same registered access started twice')


def test_execution_failure_leaves_permanent_failed_claim(registration, monkeypatch):
    args, _ = registration
    row = module.validate_registration(args, 'c'*64)
    monkeypatch.setattr(module, 'backup_mount_identity', lambda *a, **k: {'mount': 'fixed'})
    monkeypatch.setattr(module, 'validate_backup_root', lambda *a, **k: 'verified')
    with pytest.raises(RuntimeError, match='interrupted'):
        with module.validation_execution(args, row, 'c'*64) as claim:
            raise RuntimeError('interrupted')
    outcome = Path(claim['local']).with_name(Path(claim['local']).name.replace('.claim.', '.outcome.'))
    assert json.loads(outcome.read_text())['status'] == 'failed'
    with pytest.raises(ValueError, match='already executed'):
        with module.validation_execution(args, row, 'c'*64):
            pytest.fail('failed run reused its access')


def test_exclusive_json_never_replaces_existing_report(tmp_path):
    destination = tmp_path/'report.json'
    module.exclusive_json(destination, {'original': 1})
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        module.exclusive_json(destination, {'replacement': 2})
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))


def metric_row(success=1.):
    return {'success': success, 'spl': .5*success, 'oracle_success': success, 'nDTW': .8,
        'SDTW': .8*success, 'CLS': .7, 'nav_error': 1., 'oracle_error': .5,
        'action_steps': 9., 'trajectory_steps': 8., 'trajectory_lengths': 10.}


def test_report_metric_units_are_percentages_with_unmodified_path_length():
    result = module.summarize([metric_row(1.), metric_row(0.)])
    assert result['sr'] == 50. and result['spl'] == 25.
    assert result['lengths'] == 10. and result['action_steps'] == 9.


def test_synthetic_driver_full_identity_and_fixed_stress_panel(tmp_path, monkeypatch):
    """Exercise main, its Agent subclass, raw DUET reference and actual causal hook.

    Only simulator/model loading and CUDA placement are replaced. Outputs remain
    temporary synthetic test artifacts, not navigation benchmark evidence.
    """
    import run_duet
    import vln_improve.continuation_v2 as hook_module
    rows = [{'instr_id': 'a', 'scan': 'scene'}]
    class Env:
        def __init__(self):
            self.data = copy.deepcopy(rows)
            self.gt_trajs = {'a': ('scene', ['v0', 'v1'])}
        def reset_epoch(self, **kwargs): pass
        def _eval_item(self, *args): return metric_row()
        @property
        def graphs(self): raise AssertionError('policy read the complete graph')
    class BaseAgent:
        def __init__(self):
            self.args = SimpleNamespace(max_action_len=15, fusion='dynamic')
            self.env, self.decision_hook, self.feedback = Env(), None, 'argmax'
        def make_equiv_action(self, actions, graphs, observations, trajectories):
            if actions[0] is not None:
                trajectories[0]['path'].append(graphs[0].path(observations[0]['viewpoint'], actions[0]))
        def rollout(self, reset=True):
            trajectory = {'instr_id': 'a', 'path': [['v0']]}
            graph = KnownGraph(); graph.node_positions = {}
            visited, current = set(), 'v0'
            for step in range(15):
                visited.add(current)
                ids = [None]+['v'+str(i) for i in range(max(int(v[1:]) for v in visited)+3)]
                for vp in ids[1:]: graph.node_positions[vp] = (float(vp[1:]), 0., 0.)
                mask = torch.tensor([[vp in visited for vp in ids]])
                logits = torch.full((1, len(ids)), -torch.inf); logits[0, 0] = 6. if step >= 8 else -2.
                preferred = next(vp for vp in ids[1:] if vp not in visited and int(vp[1:]) > int(current[1:]))
                for index, vp in enumerate(ids[1:], 1):
                    if vp not in visited: logits[0, index] = 3. if vp == preferred else 1.
                nav = {'gmap_vpids': [ids], 'vp_cand_vpids': [ids], 'gmap_masks': torch.ones_like(mask),
                    'gmap_visited_masks': mask, 'vp_nav_masks': ~mask, 'no_vp_left': [False],
                    'gmap_pos_fts': torch.zeros(1, len(ids), 7)}
                outputs = {'fused_logits': logits, 'global_logits': logits.clone(), 'local_logits': logits.clone(),
                    'gmap_embeds': torch.zeros(1, len(ids), 768), 'vp_embeds': torch.zeros(1, len(ids), 768)}
                observation = NoGoalObservation(instr_id='a', scan='scene', viewpoint=current,
                    heading=0., elevation=0., viewIndex=0, gt_path='forbidden', distance='forbidden')
                if self.decision_hook:
                    outputs = self.decision_hook(nav, outputs, [observation], [False], step, [trajectory])
                graph.node_stop_scores[current] = {'stop': float(outputs['fused_logits'].softmax(1)[0, 0])}
                action = None if step == 14 else ids[int(outputs['fused_logits'].argmax())]
                self.make_equiv_action([action], [graph], [observation], [trajectory])
                if action is None:
                    endpoint = max(graph.node_stop_scores, key=lambda vp: graph.node_stop_scores[vp]['stop'])
                    if endpoint != current: trajectory['path'].append(graph.path(current, endpoint))
                    return [trajectory]
                current = action
            raise AssertionError('synthetic simulator did not terminate')
    package, upstream = ModuleType('r2r'), ModuleType('r2r.agent')
    package.agent = upstream; upstream.GMapNavAgent = BaseAgent
    monkeypatch.setitem(sys.modules, 'r2r', package)
    monkeypatch.setitem(sys.modules, 'r2r.agent', upstream)
    monkeypatch.setattr(run_duet, 'verify', lambda: {'commit': 'fixed'})
    monkeypatch.setattr(run_duet, 'select_partition', lambda rows, *args: rows)
    def fake_main():
        cli = run_duet.parse_cli()
        chosen = run_duet.select_partition(rows, cli.split, .2, 0)
        agent = upstream.GMapNavAgent(); agent.env.data = copy.deepcopy(chosen)
        agent.test(use_dropout=False, feedback='argmax')
        module.exclusive_json(cli.output, {'metadata': {'split': cli.split, 'num_episodes': len(chosen)},
            'summary': module.summarize([metric_row()]), 'resources': {'rollout_seconds': 0.,
                'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0}})
    monkeypatch.setattr(run_duet, 'main', fake_main)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(ContinuationComparator, 'cuda', lambda self: self)
    model_config = json.loads((ROOT/'configs/r2r.json').read_text())
    model_config.update(dataset_root=str(tmp_path/'datasets'), base_checkpoint=str(tmp_path/'base'))
    config_file = tmp_path/'config.json'; config_file.write_text(json.dumps(model_config))
    files = [tmp_path/'base', tmp_path/'datasets/R2R/features/pth_vit_base_patch16_224_imagenet.hdf5',
        tmp_path/'datasets/R2R/annotations/R2R_train_enc.json', tmp_path/'datasets/R2R/connectivity/scene_connectivity.json']
    for file in files: file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(b'synthetic')
    value = panel(); value['selection'].update(instr_ids=['a'], count=1)
    value['provenance'] = dict(zip(module.ASSET_FIELDS, [file_sha256(file) for file in files[:3]]+
        [object_sha256({files[3].name: file_sha256(files[3])})]))
    value['provenance'].update(model_config_sha256=file_sha256(config_file), partition_seed=model_config['partition_seed'],
        dev_fraction=model_config['dev_fraction'], upstream_lock={'commit': 'fixed'},
        source_files={name: file_sha256(ROOT/name) for name in module.CODE_FILES})
    manifest = tmp_path/'dataset-manifest.json'; manifest.write_text(json.dumps(value))
    destination = tmp_path/'identity.json'
    command = ['--identity', '--split', 'train_dev', '--dataset-manifest', str(manifest),
               '--config', str(config_file), '--output', str(destination)]
    assert module.main(command) == 0
    report = json.loads(destination.read_text())
    assert report['metadata']['identity_checked_rollouts'] == 4
    assert report['metadata']['zero_comparator_full_trajectory_identity'] is True
    assert set(report['conditions']) == set(module.SCHEDULES)
    assert all(value['intervention_count'] == 0 for value in report['conditions'].values())
    assert report['summary'] == report['conditions']['natural']['summary']
    assert report['metadata']['subset'] is True
    assert upstream.GMapNavAgent is BaseAgent  # Global hook patched back after completion.
    with pytest.raises(ValueError, match='overwrite'):
        module.main(command)
    exported = payload(); exported['provenance']['fit_provenance'] = value['provenance']
    trained_path = tmp_path/'selected-head.pt'; torch.save(exported, trained_path)
    trained_command = [item for item in command if item != '--identity']
    trained_command[-1] = str(tmp_path/'trained-navigation.json')
    trained_command += ['--head', str(trained_path)]
    assert module.main(trained_command) == 0
    trained_report = json.loads((tmp_path/'trained-navigation.json').read_text())
    assert trained_report['metadata']['head_sha256'] == file_sha256(trained_path)
    assert trained_report['metadata']['identity_checked_rollouts'] == 0
    assert trained_report['metadata']['head_metadata']['thresholds'] == exported['thresholds']
    # A regression that changes an action must fail real trajectory identity.
    command[-1] = str(tmp_path/'broken-identity.json')
    monkeypatch.setattr(hook_module, 'choose_action', lambda *args: 1)
    with pytest.raises(AssertionError, match='changed original full navigation'):
        module.main(command)
    assert upstream.GMapNavAgent is BaseAgent
    assert not (tmp_path/'broken-identity.json').exists()
