#!/usr/bin/env python3
"""Audit complete natural navigation and report all five frozen comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'src'))
from audit_continuation_probe import Distance, close, flat, load_graphs, objsha, require, sha
from audit_continuation_v2 import independent_metrics
from report_intervention_loop import paired_comparison
from run_continuation_v2 import summarize, exclusive_json

ARMS = ('relative-history', 'absolute-history', 'teacher-history', 'relative-nohistory', 'relative-olddata')
SHARED = ('dataset', 'feature_id', 'base_checkpoint_sha256', 'upstream_commit',
    'max_action_len', 'feedback', 'model_config_sha256', 'partition_seed', 'dev_fraction',
    'train_annotation_sha256', 'connectivity_sha256', 'source_and_integration_sha256',
    'split', 'protocol_sha256', 'subset', 'model', 'seed', 'num_episodes', 'gpu', 'feature_schema')
ANNOTATIONS = {'train_dev': '8ffdfd5a5c5efeef56af7883ebac28d854a64161f1d0a3d0e52ba75086b726b8',
               'val_unseen': '29110ed14c22cba6ba12bfc2e5f4d3bfdc27a253ff47f55a7a06ab97c9b71d13'}
EXPECTED_COUNTS = {'train_dev': 2890, 'val_unseen': 2349}
SOURCE_FILES = ('report_continuation_v2.py', 'audit_continuation_v2.py',
                'audit_continuation_probe.py', 'report_intervention_loop.py', 'run_continuation_v2.py')


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), label+': invalid SHA256')


def read_frozen(path, identities, *, json_value=True):
    """Hash the bytes actually decoded, then recheck all inputs before output."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    value = hashlib.sha256(raw).hexdigest()
    require(path not in identities or identities[path] == value, 'input changed during analysis: '+str(path))
    identities[path] = value
    if not json_value:
        return value
    def reject(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(raw, parse_constant=reject)


def indexed(rows, expected, label):
    require(isinstance(rows, list) and rows and all(isinstance(r, dict)
            and isinstance(r.get('instr_id'), str) and r['instr_id'] for r in rows), label+': invalid rows')
    values = {r['instr_id']: r for r in rows}
    require(len(values) == len(rows) and set(values) == expected, label+': incomplete/duplicate instruction inventory')
    return values


def validate_reports(baseline, methods, plan, truth):
    try:
        return _validate_reports(baseline, methods, plan, truth)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('incomplete or invalid navigation/freeze metadata: '+str(error)) from error


def _validate_reports(baseline, methods, plan, truth):
    require(tuple(methods) == ARMS and plan['arms'] == list(ARMS), 'all five frozen arms are required')
    require(plan['schema'] == 'e3_continuation_joint_freeze_v1' and plan['status'] == 'frozen', 'invalid joint freeze')
    require(plan['training_only'] is True and type(plan['seed']) is int and plan['seed'] == 0, 'invalid frozen scope/seed')
    for kind in ('navigation', 'training'):
        files = plan[kind+'_source_files']
        require(isinstance(files, dict) and files, 'missing frozen source inventory')
        for value in files.values(): digest(value, kind+' source')
        require(objsha(files) == plan[kind+'_source_sha256'], 'invalid frozen '+kind+' identity')
    require([e['arm'] for e in plan['entries']] == list(ARMS), 'frozen entries differ')
    require(baseline['metadata']['mode'] == 'baseline' and baseline['metadata']['subset'] is False,
            'full original DUET baseline required')
    require(baseline['metadata']['num_episodes'] == len(truth), 'full annotation count differs')
    base = baseline['metadata']; model = base['model']
    require(base['dataset'] == 'r2r' and type(base['seed']) is int and base['seed'] == 0
            and base['split'] in ANNOTATIONS and base['feedback'] == 'argmax' and base['max_action_len'] == 15
            and model['batch_size'] == 1 and model['fusion'] == 'dynamic' and model['angle_feat_size'] == 4
            and model['max_action_len'] == 15 and isinstance(base['gpu'], str) and base['gpu'],
            'baseline is not the frozen natural navigation protocol')
    require(base['model_config_sha256'] == objsha({k: v for k, v in model.items() if k != 'batch_size'}),
            'baseline model configuration hash differs')
    for key in SHARED:
        require(key in base, 'missing shared protocol field: '+key)
        if key.endswith('sha256') or key == 'feature_id': digest(base[key], key)
    fitted = plan['data_identity']['fit']['provenance']
    require(plan['data_identity']['dev']['provenance'] == fitted, 'frozen fit/dev provenance differs')
    for left, right in (('feature_id', 'feature_sha256'), ('base_checkpoint_sha256', 'base_checkpoint_sha256'),
                        ('train_annotation_sha256', 'train_annotation_sha256'), ('connectivity_sha256', 'connectivity_sha256'),
                        ('partition_seed', 'partition_seed'), ('dev_fraction', 'dev_fraction')):
        require(base[left] == fitted[right], 'baseline differs from frozen '+right)
    require(base['upstream_commit'] == fitted['upstream_lock']['commit'], 'baseline upstream commit differs')
    entries = {entry['arm']: entry for entry in plan['entries']}
    episodes, paths, summaries = {}, {}, {}
    for name, report in {'baseline': baseline, **methods}.items():
        for key in SHARED:
            require(key in report['metadata'] and report['metadata'][key] == base[key], name+': shared protocol differs: '+key)
        episodes[name] = indexed(report['episodes'], set(truth), name+'/episodes')
        paths[name] = indexed(report['trajectories'], set(truth), name+'/trajectories')
        for instr, row in episodes[name].items():
            require(row['scan_id'] == truth[instr][0], name+': annotation scene differs')
        costs = report['resources']
        require(type(costs['rollout_seconds']) in (int, float) and math.isfinite(costs['rollout_seconds'])
                and costs['rollout_seconds'] > 0, 'missing/invalid rollout timing')
        allocated, reserved = costs['cuda_peak_allocated_bytes'], costs['cuda_peak_reserved_bytes']
        require(type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved,
                'missing/invalid navigation CUDA allocation evidence')
        summaries[name] = summarize(list(episodes[name].values()))
        require(set(summaries[name]) == set(report['summary']), 'summary fields differ')
        for metric, value in summaries[name].items():
            close(value, report['summary'][metric], name+'/summary/'+metric)
        if name == 'baseline':
            continue
        meta, entry = report['metadata'], entries[name]
        head = meta['head_metadata']
        require(entry['selection_status'] in ('passed_cached_dev_gate', 'failed_cached_dev_gate_fixed_final_strictest')
                and type(entry['eligible']) is bool
                and entry['eligible'] is (entry['selection_status'] == 'passed_cached_dev_gate'),
                'frozen selection status/eligibility differs')
        require(all(isinstance(meta.get(key), str) and meta[key] for key in ('torch_version', 'cuda_version')),
                'method lacks explicit runtime versions')
        require(costs['condition_rollouts'] == len(truth) and costs['identity_reference_rollouts'] == 0,
                'method performed extra or missing navigation episodes')
        require(report['schema'] == 'e3_continuation_navigation_v2'
                and meta['mode'] == 'continuation_v2_navigation', 'wrong method navigation mode')
        digest(meta['head_sha256'], 'head')
        require(meta['head_sha256'] == entry['head_sha256'] and head['schema'] == 'e3_continuation_head_v2'
                and head['arm'] == name and all(head[key] == entry[key] for key in
                ('epoch', 'global_step', 'thresholds', 'model_config', 'selection_status', 'selection')),
                'frozen head/threshold identity differs')
        require(head['config'] == plan['training_config']
                and objsha(head['config']) == entry['training_config_sha256'], 'frozen training configuration differs')
        require(head['code_identity'] == plan['training_source_files'] == entry['training_source_files']
                and entry['training_source_sha256'] == plan['training_source_sha256'], 'frozen training source differs')
        provenance = head['provenance']
        require(provenance['source_files'] == plan['training_source_files']
                and provenance['source_sha256'] == plan['training_source_sha256']
                and provenance['training_config_sha256'] == entry['training_config_sha256'], 'head training provenance differs')
        for kind in ('fit', 'dev'):
            identity = plan['data_identity'][kind]
            require(head['data_identity'][kind] == identity
                    and provenance[kind+'_manifest_sha256'] == identity['manifest_sha256'] == entry[kind+'_manifest_sha256']
                    and provenance[kind+'_provenance'] == identity['provenance']
                    and provenance[kind+'_selection'] == identity['selection'], 'head '+kind+' identity differs')
        require(meta['code_files'] == plan['navigation_source_files']
                and meta['code_sha256'] == plan['navigation_source_sha256'], 'frozen navigation source differs')
        require(head['scope'] == provenance['scope'] == 'training_experiment'
                and 'engineering_smoke' not in head['selection_status'], 'engineering head is not a formal method')
        require(meta['dataset_manifest_sha256'] is None, 'full navigation cannot use a stress panel')
        selection = meta['selection']
        require(selection['split'] == base['split'] and selection['subset'] is False
                and selection['count'] == len(truth) and selection['instr_ids'] == sorted(truth)
                and selection['scan_ids'] == sorted({v[0] for v in truth.values()})
                and selection['conditions'] == ['natural'] and selection['condition_seed'] == base['seed'],
                'full natural selection inventory differs')
        if base['split'] == 'train_dev':
            require(meta['validation_access_id'] is None and meta['validation_execution'] is None,
                    'train_dev must not claim a validation access')
        else:
            require(isinstance(meta['validation_access_id'], str)
                    and re.fullmatch(r'V[0-9]+', meta['validation_access_id']), 'missing validation access')
            digest(meta['validation_execution']['sha256'], 'validation execution')
        require(set(report['conditions']) == {'natural'}, 'full navigation must contain natural condition only')
        natural = report['conditions']['natural']
        for key in ('episodes', 'trajectories', 'summary'):
            require(natural[key] == report[key], 'natural copy differs')
        decisions = indexed(natural['interventions'], set(truth), name+'/interventions')
        for instr, decision in decisions.items():
            require(decision['scan_id'] == truth[instr][0] and not decision['perturbations'], 'natural run contains perturbations')
            require(type(decision['decision_count']) is int and 1 <= decision['decision_count'] <= 15, 'decision budget differs')
            path = paths[name][instr]['trajectory']
            require(len(path) in (decision['decision_count'], decision['decision_count']+1), 'decision count/path segments differ')
            choice = decision['intervention']
            if choice is not None:
                require(type(choice['step']) is int and 0 <= choice['step'] < min(14, decision['decision_count']), 'invalid intervention time')
                require(choice['action'] != choice['baseline_action'], 'intervention did not change action')
                require(path[:choice['step']+1] == paths['baseline'][instr]['trajectory'][:choice['step']+1],
                        'trajectory changed before first intervention')
                scores = choice['scores']
                require(isinstance(scores, list) and 1 <= len(scores) <= 4 and all(isinstance(row, list)
                        and len(row) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in row)
                        for row in scores), 'invalid recorded intervention scores')
                if choice['action'] is None:
                    require(decision['decision_count'] == choice['step']+1, 'STOP intervention did not terminate')
                else:
                    require(isinstance(choice['action'], str) and choice['action']
                            and choice['step']+1 < len(path) and path[choice['step']+1][-1] == choice['action'],
                            'intervention action differs from executed segment')
            else:
                require(path == paths['baseline'][instr]['trajectory'], 'no-intervention trajectory differs from baseline')
        require(natural['intervention_count'] == sum(d['intervention'] is not None for d in decisions.values()), 'intervention count differs')
    if base['split'] == 'val_unseen':
        require(len({r['metadata']['validation_access_id'] for r in methods.values()}) == len(ARMS),
                'five methods must have separate validation accesses')
    require(len({(r['metadata']['torch_version'], r['metadata']['cuda_version']) for r in methods.values()}) == 1,
            'method runtime versions differ')
    for key in ('torch_version', 'cuda_version'):
        if key in base:
            require(all(r['metadata'][key] == base[key] for r in methods.values()), 'baseline runtime version differs')
    return episodes, paths, summaries


def graph_checks(episodes, paths, truth, graphs):
    distances = {s: Distance(g) for s, g in graphs.items()}
    errors, count, transitions = {}, 0, 0
    for name in episodes:
        errors[name] = {}
        for instr, (scan, reference) in truth.items():
            path = paths[name][instr]['trajectory']
            nodes = flat(path)
            require(path[0] == [reference[0]], 'wrong navigation starting segment')
            require(len(path) <= 16, 'navigation exceeds the decision/fallback segment budget')
            require(all(a == b or b in graphs[scan].get(a, {}) for a, b in zip(nodes, nodes[1:])),
                    'nonexecutable navigation edge')
            values = independent_metrics(path, reference, graphs[scan], distances[scan])
            require(set(episodes[name][instr]) == {'instr_id', 'scan_id', *values}, 'per-episode metric fields differ')
            for key, value in values.items():
                close(value, episodes[name][instr][key], name+':'+instr+':'+key, errors[name])
                count += 1
            transitions += len(nodes)-1
    return {'status': 'passed', 'episodes': sum(len(e) for e in episodes.values()),
        'metric_values_recomputed': count, 'physical_transitions_checked': transitions,
        'maximum_metric_errors': errors}


def frozen_artifacts(plan, directory, identities):
    require(plan['experiment_file'] == 'experiment.json', 'invalid frozen experiment filename')
    experiment_path = directory/'experiment.json'
    spec = read_frozen(experiment_path, identities)
    require(identities[experiment_path.resolve()] == plan['experiment_sha256'], 'frozen experiment differs')
    require(spec['seed'] == plan['seed'] and spec['training_plan']['arms'] == list(ARMS),
            'frozen experiment arm/seed differs')
    fitted = plan['data_identity']['fit']['provenance']
    require(fitted['experiment_sha256'] == plan['experiment_sha256'], 'collection experiment differs from joint freeze')
    for name, value in spec['asset_pins'].items():
        require(fitted[name] == value, 'frozen experiment asset pin differs: '+name)
    inventory = {}
    for entry in plan['entries']:
        for kind, filename in (('head', 'selected-head.pt'), ('training_report', 'training-report.json')):
            relative = str(Path(entry['arm'])/filename)
            require(entry[kind+'_file'] == relative, 'frozen artifact path differs')
            value = read_frozen(directory/relative, identities, json_value=False)
            require(value == entry[kind+'_sha256'], 'frozen artifact SHA differs: '+relative)
            inventory[relative] = value
    return spec, inventory


def validation_claims(methods, identities, input_sha, plan, episodes):
    """Check completed, separately registered accesses using their saved bytes."""
    for arm, report in methods.items():
        meta = report['metadata']; execution = meta['validation_execution']
        local, backup = Path(execution['local']), Path(execution['backup'])
        claim = read_frozen(local, identities); mirror = read_frozen(backup, identities)
        require(claim == mirror and identities[local.resolve()] == identities[backup.resolve()] == execution['sha256'],
                'validation claim/backup identity differs')
        require(claim['schema'] == 'continuation_validation_execution_v2'
                and claim['access_id'] == meta['validation_access_id']
                and claim['head_sha256'] == meta['head_sha256'] and claim['code_sha256'] == meta['code_sha256'],
                'validation claim does not bind evaluated head/code/access')
        digest(claim['ledger_sha256'], 'validation ledger snapshot')
        request = claim['registration']['request']
        expected = {'split': 'val_unseen', 'label_use': 'evaluation', 'parameter_fitting_split': 'train_fit',
            'subset': False, 'subset_ids': [], 'config_sha256': plan['experiment_sha256'],
            'checkpoint_sha256': meta['head_sha256'], 'code_sha256': meta['code_sha256'],
            'seed': 0, 'expected_episodes': episodes}
        require(all(request.get(k) == v for k, v in expected.items()), 'validation registration differs')
        outcomes = []
        for path in (local, backup):
            require(path.name.endswith('.claim.json'), 'invalid validation claim filename')
            outcome_path = path.with_name(path.name[:-len('.claim.json')]+'.outcome.json')
            outcomes.append(read_frozen(outcome_path, identities))
        require(outcomes[0] == outcomes[1] and outcomes[0]['status'] == 'completed'
                and outcomes[0]['access_id'] == meta['validation_access_id']
                and outcomes[0]['claim_sha256'] == execution['sha256']
                and outcomes[0]['report_sha256'] == input_sha[arm], 'validation completion/report binding differs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'methods-dir', 'frozen-plan', 'annotations', 'connectivity', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--navigation-config', type=Path, default=ROOT/'configs/r2r.json')
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'preserve existing analysis')
    started = time.monotonic()
    identities = {}
    source_identity = {name: read_frozen(ROOT/'scripts'/name, identities, json_value=False) for name in SOURCE_FILES}
    plan = read_frozen(args.frozen_plan, identities)
    baseline = read_frozen(args.baseline, identities)
    split = baseline['metadata']['split']
    require(split in ANNOTATIONS, 'annotation split/asset identity differs')
    annotations = read_frozen(args.annotations, identities)
    require(identities[args.annotations.resolve()] == ANNOTATIONS[split], 'annotation split/asset identity differs')
    truth = {}
    for row in annotations:
        for number in range(len(row['instructions'])):
            instr = f"{row['path_id']}_{number}"
            require(instr not in truth, 'duplicate annotation instruction')
            truth[instr] = (row['scan'], row['path'])
    if split == 'train_dev':
        meta = baseline['metadata']
        scans = sorted({scan for scan, _ in truth.values()}, key=lambda s: objsha([meta['partition_seed'], s]))
        require(len(scans) >= 2 and type(meta['dev_fraction']) in (int, float)
                and 0 < meta['dev_fraction'] < 1, 'invalid scene partition configuration')
        n = min(len(scans)-1, max(1, round(len(scans)*meta['dev_fraction'])))
        selected = set(scans[:n])
        truth = {i: value for i, value in truth.items() if value[0] in selected}
    require(len(truth) == EXPECTED_COUNTS[split], 'expected full split inventory differs')
    inputs = {'baseline': args.baseline, **{arm: args.methods_dir/(arm+'.json') for arm in ARMS}}
    methods = {arm: read_frozen(inputs[arm], identities) for arm in ARMS}
    input_sha = {name: identities[path.resolve()] for name, path in inputs.items()}
    episodes, paths, summaries = validate_reports(baseline, methods, plan, truth)
    spec, artifacts = frozen_artifacts(plan, args.frozen_plan.parent, identities)
    navigation_config = read_frozen(args.navigation_config, identities)
    fitted = plan['data_identity']['fit']['provenance']
    require(identities[args.navigation_config.resolve()] == fitted['model_config_sha256'],
            'frozen navigation config file differs')
    for key in ('model', 'partition_seed', 'dev_fraction'):
        require(navigation_config[key] == baseline['metadata'][key], 'navigation configuration metadata differs: '+key)
    graph_files = set(args.connectivity.glob('*_connectivity.json'))
    for path in graph_files: read_frozen(path, identities, json_value=False)
    graphs, identity = load_graphs(args.connectivity, {s for s, _ in truth.values()}, baseline['metadata']['connectivity_sha256'])
    geometry = graph_checks(episodes, paths, truth, graphs)
    comparisons = {arm: paired_comparison(episodes['baseline'], episodes[arm], paths['baseline'], paths[arm],
                                          replicates=2000, seed=0) for arm in ARMS}
    if split == 'val_unseen': validation_claims(methods, identities, input_sha, plan, len(truth))
    target = spec['target']
    gain_sr, gain_spl = target['sr_gain_percentage_points_min'], target['spl_gain_percentage_points_min']
    goals = {arm: {'against_original_fixed_target': summaries[arm]['sr'] >= target['baseline_sr_percent']+gain_sr
                  and summaries[arm]['spl'] >= target['baseline_spl_percent']+gain_spl,
                  'against_current_paired_baseline': comparisons[arm]['metrics']['sr']['delta_pp'] >= gain_sr
                  and comparisons[arm]['metrics']['spl']['delta_pp'] >= gain_spl} for arm in ARMS} if split == 'val_unseen' else None
    require(graph_files == set(args.connectivity.glob('*_connectivity.json')), 'graph inventory changed during analysis')
    require(all(sha(path) == value for path, value in identities.items()), 'input changed during analysis')
    report = {'schema': 'e3_continuation_complete_navigation_report_v1', 'status': 'passed', 'split': split,
        'scope': 'complete_natural_'+split, 'summaries': summaries, 'comparisons': comparisons,
        'frozen_heads': {entry['arm']: {key: entry[key] for key in ('head_sha256', 'epoch', 'global_step',
            'thresholds', 'model_config', 'eligible', 'selection_status')} for entry in plan['entries']},
        'goal_checks_val_unseen_only': goals, 'target': target, 'graph_audit': geometry,
        'input_sha256': input_sha, 'frozen_plan_sha256': identities[args.frozen_plan.resolve()],
        'frozen_artifact_sha256': artifacts, 'experiment_sha256': plan['experiment_sha256'],
        'annotations_sha256': identities[args.annotations.resolve()], 'connectivity_identity': identity,
        'navigation_config_sha256': identities[args.navigation_config.resolve()],
        'runtime_metadata': {name: {key: value['metadata'].get(key) for key in
            ('gpu', 'torch_version', 'cuda_version', 'protocol_sha256')} for name, value in {'baseline': baseline, **methods}.items()},
        'navigation_resources': {name: value['resources'] for name, value in {'baseline': baseline, **methods}.items()},
        'bootstrap': {'unit': 'scan_id', 'paired': True, 'resamples': 2000, 'seed': 0,
            'confidence': .95, 'aggregation': 'episode weighted, whole scenes resampled with multiplicity',
            'multiplicity': 'five descriptive pointwise intervals; no multiple-comparison correction'},
        'resources': {'cpu_wall_seconds': time.monotonic()-started, 'new_navigation_episodes': 0},
        'source_sha256': source_identity,
        'limitations': ['One training seed does not establish robustness.',
            'train_dev uses houses already seen by the pretrained DUET backbone.',
            'val_unseen is exposed development data, not the official blind test.',
            'Legacy baseline omits explicit runtime versions; matching full protocol digests are retained, missing fields remain null.',
            'Recorded rollout timings are observations; concurrent machine load was not controlled for a speed benchmark.',
            'Path/metric verification alone cannot prove every internal policy computation.']}
    exclusive_json(args.output, report)
    print(json.dumps({'output': str(args.output), 'summaries': summaries, 'goal_checks': goals}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
