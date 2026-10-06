"""Synthetic-only audit tests: no benchmark annotations or navigation access."""
from __future__ import annotations

import copy
import json
import math
import random

import pytest

from scripts import report_continuation_v2 as report


def make_reports():
    graph = {'A': {'B': 4.}, 'B': {'A': 4., 'C': 4.},
             'C': {'B': 4., 'D': 4.}, 'D': {'C': 4.}}
    truth = {'1_0': ('scene', ['A', 'B', 'C']),
             '1_1': ('scene', ['A', 'B', 'C'])}
    paths = [{'instr_id': instr, 'trajectory': [['A'], ['B'], ['C']]}
             for instr in truth]
    episodes = [{'instr_id': row['instr_id'], 'scan_id': 'scene',
                 **report.independent_metrics(row['trajectory'], truth[row['instr_id']][1],
                                              graph, report.Distance(graph))} for row in paths]
    model = {'dataset': 'r2r', 'batch_size': 1, 'fusion': 'dynamic',
             'angle_feat_size': 4, 'max_action_len': 15}
    metadata = {'dataset': 'r2r', 'feature_id': '1'*64,
        'base_checkpoint_sha256': '2'*64, 'upstream_commit': 'f'*40,
        'max_action_len': 15, 'feedback': 'argmax',
        'model_config_sha256': report.objsha({k: v for k, v in model.items() if k != 'batch_size'}),
        'partition_seed': 20261003, 'dev_fraction': .2,
        'train_annotation_sha256': '3'*64, 'connectivity_sha256': '4'*64,
        'source_and_integration_sha256': '5'*64, 'split': 'train_dev',
        'protocol_sha256': '6'*64, 'subset': False, 'model': model,
        'seed': 0, 'num_episodes': len(truth), 'gpu': 'synthetic GPU',
        'feature_schema': {'dim': 768}, 'mode': 'baseline'}
    baseline = {'metadata': metadata, 'episodes': episodes, 'trajectories': paths,
                'summary': report.summarize(episodes), 'resources': {
                    'rollout_seconds': 1., 'cuda_peak_allocated_bytes': 1024,
                    'cuda_peak_reserved_bytes': 2048}}
    training_source = {'scripts/train_continuation_v2.py': 'a'*64}
    navigation_source = {'scripts/run_continuation_v2.py': 'b'*64}
    config = {'seed': 0, 'device': 'cuda', 'epochs': 20, 'hidden_dim': 128}
    provenance = {'feature_sha256': metadata['feature_id'],
        **{k: metadata[k] for k in ('base_checkpoint_sha256', 'train_annotation_sha256',
                                   'connectivity_sha256', 'partition_seed', 'dev_fraction')},
        'upstream_lock': {'commit': metadata['upstream_commit']}}
    identities = {kind: {'manifest_sha256': char*64,
                         'selection': {'split': 'train_'+kind, 'instr_ids': [kind+'_0']},
                         'provenance': copy.deepcopy(provenance)}
                  for kind, char in [('fit', 'c'), ('dev', 'd')]}
    plan = {'schema': 'e3_continuation_joint_freeze_v1', 'status': 'frozen',
        'training_only': True, 'seed': 0, 'arms': list(report.ARMS),
        'entries': [], 'training_config': config, 'data_identity': identities,
        'training_source_files': training_source,
        'training_source_sha256': report.objsha(training_source),
        'navigation_source_files': navigation_source,
        'navigation_source_sha256': report.objsha(navigation_source)}
    methods = {}
    for arm in report.ARMS:
        mode = arm.split('-')[0]
        selected = {'epoch': 5, 'eligible': True, 'sr_threshold': .1, 'spl_threshold': .01}
        head = {'schema': 'e3_continuation_head_v2', 'arm': arm, 'epoch': 5,
            'global_step': 50, 'thresholds': {'sr': .1, 'spl': .01},
            'model_config': {'feature_dim': 1549, 'hidden_dim': 128,
                             'mode': mode, 'history': arm != 'relative-nohistory'},
            'selection_status': 'passed_cached_dev_gate', 'selection': selected,
            'config': copy.deepcopy(config), 'code_identity': copy.deepcopy(training_source),
            'data_identity': copy.deepcopy(identities), 'scope': 'training_experiment',
            'provenance': {'scope': 'training_experiment',
                'source_files': copy.deepcopy(training_source),
                'source_sha256': report.objsha(training_source),
                'training_config_sha256': report.objsha(config)}}
        for kind, identity in identities.items():
            head['provenance'].update({kind+'_manifest_sha256': identity['manifest_sha256'],
                kind+'_selection': copy.deepcopy(identity['selection']),
                kind+'_provenance': copy.deepcopy(identity['provenance'])})
        entry = {k: copy.deepcopy(head[k]) for k in ('arm', 'epoch', 'global_step',
            'thresholds', 'model_config', 'selection_status', 'selection')}
        entry.update(head_sha256=report.objsha([arm, 'head']), eligible=True,
            fit_manifest_sha256=identities['fit']['manifest_sha256'],
            dev_manifest_sha256=identities['dev']['manifest_sha256'],
            training_config_sha256=report.objsha(config),
            training_source_files=copy.deepcopy(training_source),
            training_source_sha256=report.objsha(training_source))
        plan['entries'].append(entry)
        method = copy.deepcopy(baseline)
        method['schema'] = 'e3_continuation_navigation_v2'
        method['metadata'].update(mode='continuation_v2_navigation', head_metadata=head,
            head_sha256=entry['head_sha256'], code_files=copy.deepcopy(navigation_source),
            code_sha256=report.objsha(navigation_source), dataset_manifest_sha256=None,
            validation_access_id=None, validation_execution=None,
            torch_version='synthetic torch', cuda_version='synthetic CUDA',
            selection={'split': 'train_dev', 'instr_ids': sorted(truth), 'scan_ids': ['scene'],
                'conditions': ['natural'], 'count': len(truth), 'subset': False, 'condition_seed': 0})
        method['conditions'] = {'natural': {k: copy.deepcopy(method[k])
                                           for k in ('episodes', 'trajectories', 'summary')}}
        method['conditions']['natural'].update(intervention_count=0, interventions=[
            {'instr_id': instr, 'scan_id': 'scene', 'perturbations': [],
             'decision_count': 3, 'intervention': None} for instr in truth])
        method['resources'].update(condition_rollouts=len(truth), identity_reference_rollouts=0)
        methods[arm] = method
    return baseline, methods, plan, truth, {'scene': graph}


def validate(fixture):
    return report.validate_reports(*fixture[:4])


def set_path(fixture, name, instr, path):
    baseline, methods, _, truth, graphs = fixture
    item = baseline if name == 'baseline' else methods[name]
    for row in item['trajectories']:
        if row['instr_id'] == instr:
            row['trajectory'] = copy.deepcopy(path)
    scan, reference = truth[instr]
    metrics = report.independent_metrics(path, reference, graphs[scan], report.Distance(graphs[scan]))
    for row in item['episodes']:
        if row['instr_id'] == instr:
            row.update(metrics)
    item['summary'] = report.summarize(item['episodes'])
    if name != 'baseline':
        item['conditions']['natural'].update({k: copy.deepcopy(item[k])
                                             for k in ('episodes', 'trajectories', 'summary')})


def decision(fixture, *, instr='1_0', step=1, action='D', count=3):
    natural = fixture[1][report.ARMS[0]]['conditions']['natural']
    natural['intervention_count'] = 1
    row = next(row for row in natural['interventions'] if row['instr_id'] == instr)
    row.update(decision_count=count, intervention={'step': step, 'action': action,
        'baseline_action': 'C', 'scores': [[0., 0.], [.3, .2]]})
    return row


def test_all_five_complete_reports_recompute_every_metric():
    fixture = make_reports()
    episodes, paths, summaries = validate(fixture)
    audit = report.graph_checks(episodes, paths, fixture[3], fixture[4])
    assert audit['status'] == 'passed'
    assert audit['episodes'] == 12
    assert audit['metric_values_recomputed'] == 12*12
    assert audit['physical_transitions_checked'] == 12*2
    assert all(value['sr'] == 100. and value['spl'] == 100. for value in summaries.values())


@pytest.mark.parametrize('kind', ['training', 'navigation'])
@pytest.mark.parametrize('defect', ['digest', 'empty', 'nonhex'])
def test_frozen_source_inventory_must_be_nonempty_and_self_consistent(kind, defect):
    fixture = make_reports(); plan = fixture[2]
    if defect == 'digest': plan[kind+'_source_sha256'] = 'e'*64
    elif defect == 'empty':
        plan[kind+'_source_files'] = {}
        plan[kind+'_source_sha256'] = report.objsha({})
    else:
        plan[kind+'_source_files'] = {'source.py': 'z'*64}
        plan[kind+'_source_sha256'] = report.objsha(plan[kind+'_source_files'])
    with pytest.raises(ValueError): validate(fixture)


@pytest.mark.parametrize('key,value', [('seed', 1), ('seed', False), ('training_only', False),
    ('schema', 'other'), ('status', 'partial'), ('arms', list(reversed(report.ARMS)))])
def test_wrong_freeze_protocol_is_rejected(key, value):
    fixture = make_reports(); fixture[2][key] = value
    with pytest.raises(ValueError): validate(fixture)


@pytest.mark.parametrize('key,value', [('dataset', 'rxr'), ('seed', 1), ('seed', False),
    ('feedback', 'sample'), ('max_action_len', 16), ('feature_id', 'e'*64),
    ('base_checkpoint_sha256', 'e'*64), ('train_annotation_sha256', 'e'*64),
    ('connectivity_sha256', 'e'*64), ('partition_seed', 9), ('dev_fraction', .3),
    ('upstream_commit', 'e'*40), ('model_config_sha256', 'e'*64)])
def test_shared_wrong_baseline_identity_cannot_be_laundered_by_all_methods(key, value):
    fixture = make_reports()
    for item in [fixture[0], *fixture[1].values()]: item['metadata'][key] = value
    with pytest.raises(ValueError): validate(fixture)


@pytest.mark.parametrize('key,value', [('batch_size', 2), ('fusion', 'local'),
                                      ('angle_feat_size', 8), ('max_action_len', 16)])
def test_shared_wrong_model_protocol_rejected_even_with_consistent_model_digest(key, value):
    fixture = make_reports()
    for item in [fixture[0], *fixture[1].values()]:
        model = item['metadata']['model']; model[key] = value
        item['metadata']['model_config_sha256'] = report.objsha({k: v for k, v in model.items() if k != 'batch_size'})
    with pytest.raises(ValueError): validate(fixture)


@pytest.mark.parametrize('key,value', [('split', 'train_fit'), ('instr_ids', ['1_0']),
    ('scan_ids', ['another']), ('conditions', ['natural', 'early_two']),
    ('count', 1), ('subset', True), ('condition_seed', 1)])
def test_full_selection_matches_exact_annotation_inventory(key, value):
    fixture = make_reports()
    fixture[1][report.ARMS[0]]['metadata']['selection'][key] = value
    with pytest.raises(ValueError, match='selection'): validate(fixture)


@pytest.mark.parametrize('key,value', [('epoch', 10), ('global_step', 100),
    ('thresholds', {'sr': .5, 'spl': .01}), ('model_config', {}),
    ('selection_status', 'failed_cached_dev_gate_fixed_final_strictest'),
    ('selection', None), ('schema', 'legacy_head')])
def test_every_exported_head_position_and_selection_field_is_bound_to_freeze(key, value):
    fixture = make_reports()
    fixture[1][report.ARMS[0]]['metadata']['head_metadata'][key] = value
    with pytest.raises(ValueError, match='head/threshold'): validate(fixture)


@pytest.mark.parametrize('defect', ['head_hash', 'head_config', 'head_source', 'provenance_source',
    'provenance_source_hash', 'provenance_config_hash', 'fit_manifest', 'dev_manifest',
    'fit_selection', 'dev_selection', 'fit_provenance', 'dev_provenance', 'head_data',
    'navigation_source', 'navigation_hash', 'engineering', 'stress_manifest'])
def test_head_data_training_and_navigation_provenance_cannot_drift(defect):
    fixture = make_reports(); meta = fixture[1][report.ARMS[0]]['metadata']
    head = meta['head_metadata']; provenance = head['provenance']
    if defect == 'head_hash': meta['head_sha256'] = 'e'*64
    elif defect == 'head_config': head['config']['epochs'] = 21
    elif defect == 'head_source': head['code_identity'] = {'other.py': 'e'*64}
    elif defect == 'provenance_source': provenance['source_files'] = {'other.py': 'e'*64}
    elif defect == 'provenance_source_hash': provenance['source_sha256'] = 'e'*64
    elif defect == 'provenance_config_hash': provenance['training_config_sha256'] = 'e'*64
    elif defect in ('fit_manifest', 'dev_manifest'): provenance[defect+'_sha256'] = 'e'*64
    elif defect.endswith('_selection'): provenance[defect] = {}
    elif defect.endswith('_provenance'): provenance[defect] = {}
    elif defect == 'head_data': head['data_identity']['fit']['manifest_sha256'] = 'e'*64
    elif defect == 'navigation_source': meta['code_files'] = {'other.py': 'e'*64}
    elif defect == 'navigation_hash': meta['code_sha256'] = 'e'*64
    elif defect == 'engineering': head['scope'] = provenance['scope'] = 'engineering_smoke'
    elif defect == 'stress_manifest': meta['dataset_manifest_sha256'] = 'e'*64
    with pytest.raises(ValueError): validate(fixture)


@pytest.mark.parametrize('defect', ['missing_arm', 'duplicate_episode', 'missing_trajectory',
    'wrong_scene', 'summary', 'natural_copy', 'extra_condition', 'missing_decision',
    'perturbation', 'intervention_count', 'train_dev_access'])
def test_complete_report_inventory_and_natural_copies_are_verified(defect):
    fixture = make_reports(); method = fixture[1][report.ARMS[0]]
    natural = method['conditions']['natural']
    if defect == 'missing_arm': fixture[1].pop(report.ARMS[-1])
    elif defect == 'duplicate_episode': method['episodes'].append(copy.deepcopy(method['episodes'][0]))
    elif defect == 'missing_trajectory': method['trajectories'].pop()
    elif defect == 'wrong_scene': method['episodes'][0]['scan_id'] = 'other'
    elif defect == 'summary': method['summary']['sr'] -= 1.
    elif defect == 'natural_copy': natural['summary']['sr'] -= 1.
    elif defect == 'extra_condition': method['conditions']['early_two'] = copy.deepcopy(natural)
    elif defect == 'missing_decision': natural['interventions'].pop()
    elif defect == 'perturbation': natural['interventions'][0]['perturbations'] = [{'step': 1}]
    elif defect == 'intervention_count': natural['intervention_count'] = 1
    elif defect == 'train_dev_access': method['metadata']['validation_access_id'] = 'V99'
    with pytest.raises(ValueError): validate(fixture)


def test_no_intervention_cannot_hide_changed_but_geometrically_legal_path():
    fixture = make_reports()
    set_path(fixture, report.ARMS[0], '1_0', [['A'], ['B'], ['C', 'D']])
    with pytest.raises(ValueError, match='no-intervention'): validate(fixture)


def test_intervention_prefix_action_and_fallback_path_are_accepted():
    fixture = make_reports()
    set_path(fixture, report.ARMS[0], '1_0', [['A'], ['B'], ['C', 'D'], ['C']])
    decision(fixture)
    episodes, paths, _ = validate(fixture)
    result = report.graph_checks(episodes, paths, fixture[3], fixture[4])
    assert result['physical_transitions_checked'] == 26
    assert episodes[report.ARMS[0]]['1_0']['spl'] == .5
    assert episodes[report.ARMS[0]]['1_0']['DTW'] == 4.
    assert episodes[report.ARMS[0]]['1_0']['nDTW'] == pytest.approx(math.exp(-4./9.))
    assert episodes[report.ARMS[0]]['1_0']['CLS'] == .5


@pytest.mark.parametrize('defect', ['prefix', 'wrong_action', 'no_change', 'bad_step',
    'too_late', 'bad_count', 'extra_segments', 'nan_scores', 'missing_scores', 'stop_continues'])
def test_intervention_log_must_agree_with_actual_path(defect):
    fixture = make_reports()
    set_path(fixture, report.ARMS[0], '1_0', [['A'], ['B'], ['C', 'D']])
    row = decision(fixture); choice = row['intervention']
    if defect == 'prefix':
        set_path(fixture, report.ARMS[0], '1_0', [['A'], ['B', 'C'], ['D']])
    elif defect == 'wrong_action': choice['action'] = 'A'
    elif defect == 'no_change': choice['action'] = choice['baseline_action']
    elif defect == 'bad_step': choice['step'] = -1
    elif defect == 'too_late': choice['step'] = 14
    elif defect == 'bad_count': row['decision_count'] = True
    elif defect == 'extra_segments': row['decision_count'] = 1
    elif defect == 'nan_scores': choice['scores'][0][0] = math.nan
    elif defect == 'missing_scores': choice['scores'] = []
    elif defect == 'stop_continues': choice['action'] = None
    with pytest.raises(ValueError): validate(fixture)


def test_stop_intervention_with_no_return_and_one_return_segment():
    for path in ([['A'], ['B']], [['A'], ['B'], ['A']]):
        fixture = make_reports()
        set_path(fixture, report.ARMS[0], '1_0', path)
        decision(fixture, step=1, action=None, count=2)
        episodes, paths, _ = validate(fixture)
        assert report.graph_checks(episodes, paths, fixture[3], fixture[4])['status'] == 'passed'


@pytest.mark.parametrize('defect', ['nonedge', 'start_segment', 'budget', 'metric', 'extra_metric', 'missing_metric'])
def test_independent_geometry_rejects_invalid_edges_budget_and_metrics(defect):
    fixture = make_reports(); episodes, paths, _ = validate(fixture)
    path = paths['baseline']['1_0']['trajectory']; row = episodes['baseline']['1_0']
    if defect == 'nonedge': path[1] = ['D']
    elif defect == 'start_segment': path[0] = ['A', 'B']
    elif defect == 'budget': path[:] = [['A']]+[['B'], ['A']]*8
    elif defect == 'metric': row['spl'] -= .01
    elif defect == 'extra_metric': row['undocumented_metric'] = 1.
    elif defect == 'missing_metric': row.pop('DTW')
    with pytest.raises(ValueError): report.graph_checks(episodes, paths, fixture[3], fixture[4])


def test_geometry_counts_cross_segment_edges_and_strict_three_meter_success():
    graph = {'A': {'B': 3.}, 'B': {'A': 3., 'C': 3.}, 'C': {'B': 3.}}
    truth = {'x': ('s', ['A', 'B', 'C'])}
    trajectory = [['A'], ['B', 'C'], ['B']]
    metrics = report.independent_metrics(trajectory, truth['x'][1], graph, report.Distance(graph))
    assert metrics['trajectory_lengths'] == 9.
    assert metrics['trajectory_steps'] == 3
    assert metrics['nav_error'] == 3. and metrics['success'] == 0
    assert metrics['oracle_success'] == 1 and metrics['spl'] == 0
    episodes = {'baseline': {'x': {'instr_id': 'x', 'scan_id': 's', **metrics}}}
    paths = {'baseline': {'x': {'instr_id': 'x', 'trajectory': trajectory}}}
    assert report.graph_checks(episodes, paths, truth, {'s': graph})['physical_transitions_checked'] == 3


def test_paired_statistics_use_method_minus_baseline_and_episode_weighted_scene_resampling(monkeypatch):
    base, method, base_paths, method_paths = {}, {}, {}, {}
    for i in range(4):
        instr = str(i); rescue = i < 3
        before, after = (0, 1) if rescue else (1, 0)
        base[instr] = {'scan_id': 'large' if rescue else 'small', 'success': before,
            'spl': before*.5, 'nDTW': before*.8, 'nav_error': 8.-before*8,
            'trajectory_lengths': 8.}
        method[instr] = dict(base[instr], success=after, spl=after*.5, nDTW=after*.8,
                            nav_error=8.-after*8, trajectory_lengths=10.)
        base_paths[instr] = {'trajectory': [['A'], ['C' if before else 'B']]}
        method_paths[instr] = {'trajectory': [['A'], ['C' if after else 'B']]}
    sampled_distributions = []
    original_quantile = report.paired_comparison.__globals__['quantile']
    def capture_distribution(values, q):
        sampled_distributions.append(list(values))
        return original_quantile(values, q)
    monkeypatch.setitem(report.paired_comparison.__globals__, 'quantile', capture_distribution)
    observed = report.paired_comparison(base, method, base_paths, method_paths, replicates=137, seed=23)
    assert observed['metrics']['sr']['delta_pp'] == 50.
    assert observed['metrics']['spl']['delta_pp'] == 25.
    assert observed['metrics']['nDTW']['delta_pp'] == pytest.approx(40.)
    assert observed['paired_counts']['rescues'] == 3 and observed['paired_counts']['harms'] == 1
    assert observed['delta_sr_pp_from_counts'] == 50.
    assert observed['endpoint_change_percent'] == 100.
    assert observed['mean_differences_m'] == {'nav_error': -4., 'trajectory_lengths': 2.}
    rng = random.Random(23); samples = []
    for _ in range(137):
        scenes = [rng.randrange(2) for _ in range(2)]
        numerator = sum(3 if scene == 0 else -1 for scene in scenes)
        denominator = sum(3 if scene == 0 else 1 for scene in scenes)
        samples.append(100*numerator/denominator)
    samples.sort()
    # Checking the entire sampled distribution catches scene-average weighting
    # even when both approaches happen to produce the same CI endpoints.
    assert sorted(sampled_distributions[0]) == pytest.approx(samples)
    assert 50. in sampled_distributions[0] and 0. not in sampled_distributions[0]
    def quantile(q):
        position = (len(samples)-1)*q; lo = math.floor(position); hi = math.ceil(position)
        return samples[lo]+(samples[hi]-samples[lo])*(position-lo)
    assert observed['metrics']['sr']['ci95_pp'] == pytest.approx([quantile(.025), quantile(.975)])
    reverse = report.paired_comparison(method, base, method_paths, base_paths, replicates=137, seed=23)
    assert reverse['metrics']['sr']['delta_pp'] == -50.
    assert reverse['paired_counts']['rescues'] == 1 and reverse['paired_counts']['harms'] == 3


def test_paired_comparison_rejects_different_instruction_scenes():
    fixture = make_reports(); episodes, paths, _ = validate(fixture)
    method = episodes[report.ARMS[0]]; method['1_0']['scan_id'] = 'other'
    with pytest.raises(ValueError):
        report.paired_comparison(episodes['baseline'], method, paths['baseline'], paths[report.ARMS[0]])


def test_read_frozen_hashes_exact_decoded_bytes_and_detects_second_read_mutation(tmp_path):
    path = tmp_path/'input.json'; path.write_text('{"n": 1}\n'); identities = {}
    assert report.read_frozen(path, identities) == {'n': 1}
    assert identities[path.resolve()] == report.sha(path)
    path.write_text('{"n": 2}\n')
    with pytest.raises(ValueError, match='input changed'): report.read_frozen(path, identities)


def test_read_frozen_rejects_nonfinite_json(tmp_path):
    path = tmp_path/'bad.json'; path.write_text('{"x": NaN}')
    with pytest.raises(ValueError, match='nonfinite'): report.read_frozen(path, {})


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
    return path


def make_main_inputs(tmp_path, monkeypatch):
    fixture = make_reports(); baseline, methods, plan, _, _ = fixture
    paths = {'baseline': tmp_path/'baseline.json', 'methods_dir': tmp_path/'methods',
        'frozen_plan': tmp_path/'frozen/plan.json', 'annotations': tmp_path/'annotations.json',
        'connectivity': tmp_path/'connectivity', 'navigation_config': tmp_path/'navigation-config.json',
        'output': tmp_path/'report.json'}
    partition = baseline['metadata']['partition_seed']
    other = next('unused'+str(i) for i in range(100)
                 if report.objsha([partition, 'unused'+str(i)]) > report.objsha([partition, 'scene']))
    dump(paths['annotations'], [
        {'path_id': 1, 'scan': 'scene', 'path': ['A', 'B', 'C'], 'instructions': ['one', 'two']},
        {'path_id': 2, 'scan': other, 'path': ['A', 'B', 'C'], 'instructions': ['excluded fit scene']}])
    annotation_sha = report.sha(paths['annotations'])
    graph_file = paths['connectivity']/'scene_connectivity.json'; nodes = ['A', 'B', 'C', 'D']
    rows = []
    for i, node in enumerate(nodes):
        pose = [0.]*16; pose[3] = 4.*i
        rows.append({'image_id': node, 'included': True, 'pose': pose,
                     'unobstructed': [abs(i-j) == 1 for j in range(len(nodes))]})
    dump(graph_file, rows)
    connectivity_sha = report.objsha({graph_file.name: report.sha(graph_file)})
    config = {k: baseline['metadata'][k] for k in ('model', 'partition_seed', 'dev_fraction')}
    dump(paths['navigation_config'], config)
    pins = {'feature_sha256': baseline['metadata']['feature_id'],
        'base_checkpoint_sha256': baseline['metadata']['base_checkpoint_sha256'],
        'train_annotation_sha256': annotation_sha, 'connectivity_sha256': connectivity_sha}
    spec = {'seed': 0, 'training_plan': {'arms': list(report.ARMS)}, 'asset_pins': pins,
            'target': {'baseline_sr_percent': 70., 'baseline_spl_percent': 60.,
                       'sr_gain_percentage_points_min': 5., 'spl_gain_percentage_points_min': 0.}}
    experiment = paths['frozen_plan'].parent/'experiment.json'; dump(experiment, spec)
    plan.update(experiment_file='experiment.json', experiment_sha256=report.sha(experiment))
    updates = {**pins, 'model_config_sha256': report.sha(paths['navigation_config']),
               'experiment_sha256': plan['experiment_sha256']}
    for kind in ('fit', 'dev'): plan['data_identity'][kind]['provenance'].update(updates)
    for item in [baseline, *methods.values()]:
        item['metadata'].update(train_annotation_sha256=annotation_sha, connectivity_sha256=connectivity_sha)
    for entry in plan['entries']:
        arm = entry['arm']; head = methods[arm]['metadata']['head_metadata']
        for kind in ('fit', 'dev'):
            head['data_identity'][kind] = copy.deepcopy(plan['data_identity'][kind])
            head['provenance'][kind+'_provenance'] = copy.deepcopy(plan['data_identity'][kind]['provenance'])
        for kind, filename in [('head', 'selected-head.pt'), ('training_report', 'training-report.json')]:
            path = paths['frozen_plan'].parent/arm/filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((arm+' synthetic frozen '+kind).encode())
            entry[kind+'_file'] = str(path.relative_to(paths['frozen_plan'].parent))
            entry[kind+'_sha256'] = report.sha(path)
        methods[arm]['metadata']['head_sha256'] = entry['head_sha256']
        dump(paths['methods_dir']/(arm+'.json'), methods[arm])
    dump(paths['baseline'], baseline); dump(paths['frozen_plan'], plan)
    monkeypatch.setitem(report.ANNOTATIONS, 'train_dev', annotation_sha)
    monkeypatch.setitem(report.EXPECTED_COUNTS, 'train_dev', 2)
    source = tmp_path/'fake_source/scripts/source.py'; source.parent.mkdir(parents=True)
    source.write_text('# synthetic report dependency\n')
    monkeypatch.setattr(report, 'ROOT', source.parent.parent)
    monkeypatch.setattr(report, 'SOURCE_FILES', ('source.py',))
    argv = [token for key, path in paths.items() for token in ('--'+key.replace('_', '-'), str(path))]
    return fixture, paths, argv, {'graph': graph_file, 'source': source, 'experiment': experiment}


def test_main_recomputes_complete_training_navigation_and_preserves_output(tmp_path, monkeypatch):
    fixture, paths, argv, extras = make_main_inputs(tmp_path, monkeypatch)
    assert report.main(argv) == 0
    result = json.loads(paths['output'].read_text())
    assert result['schema'] == 'e3_continuation_complete_navigation_report_v1'
    assert result['status'] == 'passed' and result['scope'] == 'complete_natural_train_dev'
    assert result['goal_checks_val_unseen_only'] is None
    assert len(result['comparisons']) == 5 and result['graph_audit']['episodes'] == 12
    assert result['resources']['new_navigation_episodes'] == 0
    assert result['frozen_plan_sha256'] == report.sha(paths['frozen_plan'])
    assert len(result['frozen_artifact_sha256']) == 10
    assert result['runtime_metadata']['baseline']['torch_version'] is None
    assert result['source_sha256'] == {'source.py': report.sha(extras['source'])}
    before = paths['output'].read_bytes()
    with pytest.raises(ValueError, match='preserve existing'): report.main(argv)
    assert paths['output'].read_bytes() == before


@pytest.mark.parametrize('value', [False, 1, None])
def test_frozen_eligibility_must_be_boolean_and_agree_with_selection_status(value):
    fixture = make_reports(); fixture[2]['entries'][0]['eligible'] = value
    with pytest.raises(ValueError): validate(fixture)


def test_failed_formal_head_remains_present_and_ineligible_in_complete_report(tmp_path, monkeypatch):
    fixture, paths, argv, _ = make_main_inputs(tmp_path, monkeypatch)
    entry = fixture[2]['entries'][-1]
    method = fixture[1][report.ARMS[-1]]; head = method['metadata']['head_metadata']
    status = 'failed_cached_dev_gate_fixed_final_strictest'
    entry.update(eligible=False, selection_status=status, selection=None, epoch=20, global_step=200)
    head.update(selection_status=status, selection=None, epoch=20, global_step=200)
    dump(paths['frozen_plan'], fixture[2]); dump(paths['methods_dir']/(report.ARMS[-1]+'.json'), method)
    assert report.main(argv) == 0
    result = json.loads(paths['output'].read_text())
    assert result['frozen_heads'][report.ARMS[-1]]['eligible'] is False
    assert result['frozen_heads'][report.ARMS[-1]]['selection_status'] == status
    assert report.ARMS[-1] in result['comparisons']


@pytest.mark.parametrize('kind', ['head', 'training_report'])
def test_main_rejects_frozen_artifact_byte_changes_before_analysis(kind, tmp_path, monkeypatch):
    fixture, paths, argv, _ = make_main_inputs(tmp_path, monkeypatch)
    entry = fixture[2]['entries'][0]
    artifact = paths['frozen_plan'].parent/entry[kind+'_file']
    artifact.write_bytes(b'changed frozen bytes')
    with pytest.raises(ValueError, match='frozen artifact SHA'): report.main(argv)
    assert not paths['output'].exists()


@pytest.mark.parametrize('kind', ['plan', 'graph', 'new_graph', 'annotations', 'navigation_config',
                                 'head', 'training_report', 'baseline', 'method', 'source', 'experiment'])
def test_main_rejects_input_changes_during_analysis(kind, tmp_path, monkeypatch):
    fixture, paths, argv, extras = make_main_inputs(tmp_path, monkeypatch)
    targets = {**extras, **paths, 'plan': paths['frozen_plan'],
        'head': paths['frozen_plan'].parent/fixture[2]['entries'][0]['head_file'],
        'training_report': paths['frozen_plan'].parent/fixture[2]['entries'][0]['training_report_file'],
        'method': paths['methods_dir']/(report.ARMS[0]+'.json'),
        'new_graph': paths['connectivity']/'new_connectivity.json'}
    original = report.graph_checks
    def check_and_change(*args):
        result = original(*args)
        target = targets[kind]
        target.write_bytes((target.read_bytes() if target.exists() else b'[]')+b'\n')
        return result
    monkeypatch.setattr(report, 'graph_checks', check_and_change)
    with pytest.raises(ValueError, match='changed during analysis'): report.main(argv)
    assert not paths['output'].exists()


@pytest.mark.parametrize('field,value', [('rollout_seconds', 0.), ('rollout_seconds', math.nan),
    ('cuda_peak_allocated_bytes', 0), ('cuda_peak_allocated_bytes', True),
    ('cuda_peak_reserved_bytes', 1), ('condition_rollouts', 3), ('identity_reference_rollouts', 1)])
def test_method_resources_prove_expected_rollout_count_and_gpu_execution(field, value):
    fixture = make_reports(); fixture[1][report.ARMS[0]]['resources'][field] = value
    with pytest.raises(ValueError): validate(fixture)


@pytest.mark.parametrize('field', ['torch_version', 'cuda_version'])
def test_method_requires_explicit_runtime_version(field):
    fixture = make_reports(); fixture[1][report.ARMS[0]]['metadata'].pop(field)
    with pytest.raises(ValueError, match='runtime versions'): validate(fixture)


@pytest.mark.parametrize('field', ['torch_version', 'cuda_version'])
def test_method_runtime_versions_must_match_across_arms_and_explicit_baseline(field):
    fixture = make_reports()
    fixture[1][report.ARMS[0]]['metadata'][field] = 'different runtime'
    with pytest.raises(ValueError, match='runtime versions differ'): validate(fixture)
    fixture = make_reports(); fixture[0]['metadata'][field] = 'different runtime'
    with pytest.raises(ValueError, match='baseline runtime version differs'): validate(fixture)
    fixture[0]['metadata'][field] = fixture[1][report.ARMS[0]]['metadata'][field]
    validate(fixture)


@pytest.mark.parametrize('value', ['', None])
def test_shared_gpu_metadata_requires_nonempty_hardware_identity(value):
    fixture = make_reports()
    for item in [fixture[0], *fixture[1].values()]: item['metadata']['gpu'] = value
    with pytest.raises(ValueError): validate(fixture)


def make_claims(tmp_path):
    fixture = make_reports(); methods, plan = fixture[1:3]
    plan['experiment_sha256'] = 'f'*64
    hashes, claim_files = {}, []
    for n, (arm, method) in enumerate(methods.items()):
        meta = method['metadata']; meta['validation_access_id'] = 'V'+str(n+1)
        local, backup = [tmp_path/root/(arm+'.claim.json') for root in ('local', 'backup')]
        claim = {'schema': 'continuation_validation_execution_v2',
            'access_id': meta['validation_access_id'], 'head_sha256': meta['head_sha256'],
            'code_sha256': meta['code_sha256'], 'ledger_sha256': 'e'*64,
            'registration': {'request': {'split': 'val_unseen', 'label_use': 'evaluation',
                'parameter_fitting_split': 'train_fit', 'subset': False, 'subset_ids': [],
                'config_sha256': plan['experiment_sha256'], 'checkpoint_sha256': meta['head_sha256'],
                'code_sha256': meta['code_sha256'], 'seed': 0, 'expected_episodes': 2}}}
        dump(local, claim); dump(backup, claim)
        meta['validation_execution'] = {'local': str(local), 'backup': str(backup), 'sha256': report.sha(local)}
        hashes[arm] = report.objsha(['synthetic navigation report', arm])
        outcome = {'status': 'completed', 'access_id': meta['validation_access_id'],
            'claim_sha256': report.sha(local), 'report_sha256': hashes[arm]}
        for path in (local, backup): dump(path.with_name(arm+'.outcome.json'), outcome)
        claim_files.append((local, backup))
    return methods, plan, hashes, claim_files


def test_validation_claims_bind_five_separate_completed_reports_and_backup_bytes(tmp_path):
    methods, plan, hashes, files = make_claims(tmp_path); identities = {}
    report.validation_claims(methods, identities, hashes, plan, 2)
    assert len(identities) == 20


@pytest.mark.parametrize('defect', ['backup', 'head', 'code', 'access', 'ledger', 'fitting_split',
    'episodes', 'config', 'incomplete', 'report_sha', 'outcome_backup'])
def test_validation_claims_reject_misbound_or_incomplete_execution(defect, tmp_path):
    methods, plan, hashes, files = make_claims(tmp_path)
    local, backup = files[0]; meta = methods[report.ARMS[0]]['metadata']
    if defect in ('incomplete', 'report_sha', 'outcome_backup'):
        first = local.with_name(report.ARMS[0]+'.outcome.json')
        second = backup.with_name(first.name); outcome = json.loads(first.read_text())
        if defect == 'incomplete': outcome['status'] = 'running'
        elif defect == 'report_sha': outcome['report_sha256'] = '9'*64
        else: outcome['access_id'] = 'V999'
        dump(first, outcome)
        if defect != 'outcome_backup': dump(second, outcome)
    else:
        claim = json.loads(local.read_text())
        if defect == 'backup': backup.write_bytes(backup.read_bytes()+b'\n')
        else:
            if defect == 'head': claim['head_sha256'] = '9'*64
            elif defect == 'code': claim['code_sha256'] = '9'*64
            elif defect == 'access': claim['access_id'] = 'V999'
            elif defect == 'ledger': claim['ledger_sha256'] = 'not a hash'
            elif defect == 'fitting_split': claim['registration']['request']['parameter_fitting_split'] = 'val_unseen'
            elif defect == 'episodes': claim['registration']['request']['expected_episodes'] = 3
            elif defect == 'config': claim['registration']['request']['config_sha256'] = '9'*64
            dump(local, claim); dump(backup, claim)
            meta['validation_execution']['sha256'] = report.sha(local)
    with pytest.raises(ValueError): report.validation_claims(methods, {}, hashes, plan, 2)
