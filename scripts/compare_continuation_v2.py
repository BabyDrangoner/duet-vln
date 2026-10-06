#!/usr/bin/env python3
"""Verify cached causal decisions against actual fixed-panel navigation.

This check performs no navigation or parameter selection. A frozen model chooses
from observations first; only then are executed branch paths and metrics read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

import torch
import run_continuation_v2 as navigation
from train_continuation_v2 import load_dataset
from vln_improve.continuation_v2 import SCHEDULES, choose_action
from vln_improve.pipeline import backup_mount_identity, validate_backup_root
from vln_improve.protocol import file_sha256, object_sha256


def require(value, message):
    if not value:
        raise ValueError(message)


def index(rows, expected, label):
    require(isinstance(rows, list), label+': rows must be a list')
    result = {r['instr_id']: r for r in rows}
    require(len(result) == len(rows) and set(result) == expected,
            label+': duplicated, missing or extra instructions')
    return result


def near(actual, expected, label, tolerance=1e-8):
    require(type(actual) in (int, float) and type(expected) in (int, float)
            and math.isfinite(actual) and math.isfinite(expected)
            and abs(actual-expected) <= tolerance, label+': numeric value differs')


def choose_cached(model, bundle, thresholds):
    """Labels and future paths are accessed only after the first trigger is set."""
    choice = None
    for record in sorted(bundle['records'], key=lambda row: row['step']):
        scores = model.score_record(record)
        candidate = choose_action(scores, model.mode, thresholds['sr'], thresholds['spl'])
        if candidate:
            choice = {'step': record['step'], 'baseline_action': record['candidate_actions'][0],
                      'action': record['candidate_actions'][candidate],
                      'scores': scores.detach().cpu().tolist()}
            break
    if choice is None:
        return bundle['reference'], None
    branches = [b for b in bundle['branches']
                if (b['target_step'], b['target_action']) == (choice['step'], choice['action'])]
    require(len(branches) == 1, 'chosen action must have one executed continuation')
    return branches[0], choice


def compare(dataset, model, thresholds, report, *, head_sha256, identity=False):
    metadata = report['metadata']
    selection = dataset.manifest['selection']
    require(report.get('schema') == navigation.REPORT_SCHEMA, 'unsupported navigation report')
    require(selection['split'] == metadata.get('split') == 'train_dev'
            and metadata.get('subset') is True, 'comparison requires a fixed train_dev panel')
    require(metadata.get('dataset_manifest_sha256') == dataset.identity['manifest_sha256'],
            'dataset manifest SHA differs')
    require(metadata.get('head_sha256') == head_sha256, 'frozen head SHA differs')
    require(metadata.get('head_metadata', {}).get('thresholds') == thresholds,
            'frozen head thresholds differ')
    if identity:
        require(head_sha256 is None and metadata.get('zero_comparator_full_trajectory_identity') is True,
                'identity report did not verify raw DUET navigation')
        require(metadata.get('identity_checked_rollouts') == selection['count']*len(SCHEDULES),
                'identity reference rollout coverage differs')
    require(metadata.get('mode') == ('continuation_v2_identity' if identity else 'continuation_v2_navigation'),
            'navigation mode differs')
    require(metadata.get('code_files') == navigation.code_identity()[0]
            and metadata.get('code_sha256') == navigation.code_identity()[1], 'navigation source differs')
    actual_selection = metadata.get('selection', {})
    for key in ('instr_ids', 'scan_ids', 'conditions', 'count', 'split'):
        require(actual_selection.get(key) == selection[key], 'panel selection differs: '+key)
    require(actual_selection.get('condition_seed') == selection['seed'], 'perturbation seed differs')
    ids = set(selection['instr_ids'])
    require(set(report['conditions']) == set(SCHEDULES), 'condition coverage differs')
    all_indexes = {}
    for condition, value in report['conditions'].items():
        all_indexes[condition] = {name: index(value[name], ids, condition+'/'+name)
                                 for name in ('episodes', 'trajectories', 'interventions')}
    for key in ('summary', 'episodes', 'trajectories'):
        require(report[key] == report['conditions']['natural'][key], 'natural top-level copy differs: '+key)
    grouped, errors, checked, seen = defaultdict(list), {}, 0, set()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for bundle in dataset.bundles:
                ref = bundle['reference']
                condition, instr = ref['condition'], ref['instr_id']
                require((condition, instr) not in seen, 'duplicate cached condition/instruction')
                seen.add((condition, instr))
                expected, choice = choose_cached(model, bundle, thresholds)
                rows = all_indexes[condition]
                episode = rows['episodes'][instr]
                require(episode['scan_id'] == ref['scan_id'], 'episode scene differs')
                require(rows['trajectories'][instr]['trajectory'] == expected['path'],
                        f'{condition}/{instr}: complete executed trajectory differs')
                require(set(episode) == {'instr_id', 'scan_id', *expected['metrics']}, 'metric inventory differs')
                for key, value in expected['metrics'].items():
                    near(episode[key], value, condition+'/'+instr+'/'+key)
                    errors[key] = max(errors.get(key, 0.), abs(episode[key]-value))
                trace = rows['interventions'][instr]
                require(trace['scan_id'] == ref['scan_id'], 'intervention scene differs')
                require(trace['perturbations'] == ref['perturbations'], 'scheduled perturbations differ')
                require(trace['decision_count'] == expected['decision_count'], 'decision budget differs')
                actual = trace['intervention']
                require((actual is None) == (choice is None), 'intervention presence differs')
                if choice is not None:
                    require(set(actual) == set(choice), 'intervention fields differ')
                    for key in ('step', 'baseline_action', 'action'):
                        require(actual[key] == choice[key], 'first causal intervention differs: '+key)
                    require(len(actual['scores']) == len(choice['scores']), 'candidate score count differs')
                    for left, right in zip(actual['scores'], choice['scores']):
                        require(len(left) == len(right) == 2, 'score dimensions differ')
                        for a, b in zip(left, right):
                            near(a, b, 'cached/online score', tolerance=1e-6)
                baseline = ref['metrics']
                grouped[condition].append({'instr_id': instr, 'scan_id': ref['scan_id'],
                    **expected['metrics'], 'baseline_success': baseline['success'],
                    'baseline_spl': baseline['spl'], 'intervened': choice is not None})
                checked += 1
    finally:
        model.train(was_training)
    require(seen == {(c, i) for c in SCHEDULES for i in ids}, 'cache coverage differs')
    summaries = {}
    for condition, rows in grouped.items():
        summary = navigation.summarize(rows)
        actual = report['conditions'][condition]
        require(set(actual['summary']) == set(summary), 'summary metric inventory differs')
        for key, value in summary.items():
            near(actual['summary'][key], value, 'condition summary/'+key)
        interventions = sum(r['intervened'] for r in rows)
        require(interventions == actual['intervention_count'], 'intervention count differs')
        summaries[condition] = {'episodes': len(rows), 'summary': summary,
            'baseline_sr': 100*math.fsum(r['baseline_success'] for r in rows)/len(rows),
            'baseline_spl': 100*math.fsum(r['baseline_spl'] for r in rows)/len(rows),
            'interventions': interventions,
            'rescued': sum(r['success'] > r['baseline_success'] for r in rows),
            'harmed': sum(r['success'] < r['baseline_success'] for r in rows)}
    return {'schema': 'e3_continuation_online_cached_comparison_v1', 'status': 'passed',
        'scope': 'fixed_train_dev_panel_not_official_validation', 'conditions': summaries,
        'checked_complete_trajectories': checked, 'all_first_actions_match': True,
        'all_complete_paths_match': True, 'maximum_metric_errors': errors,
        'head_sha256': head_sha256, 'dataset_manifest_sha256': dataset.identity['manifest_sha256'],
        'score_tolerance': 1e-6, 'metric_tolerance': 1e-8,
        'claim': 'Cache-selected frozen policy matches actual online execution; conditions are separate, and perturbed cases are not independent instructions.'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--navigation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--head', type=Path)
    group.add_argument('--identity', action='store_true')
    args = p.parse_args(argv)
    require(not args.output.exists(), 'comparison output already exists')
    manifest = json.loads((args.cache/'dataset-manifest.json').read_text())
    backup = Path(manifest['backup_root'])
    mount = backup_mount_identity(backup, backend='filesystem')
    check = lambda: validate_backup_root(backup, backend='filesystem', expected_identity=mount)
    dataset = load_dataset(args.cache, 'train_dev', check_backup=check)
    head_sha = file_sha256(args.head) if args.head else None
    model, metadata = navigation.load_head(args.head, identity=args.identity)
    require(not args.head or file_sha256(args.head) == head_sha, 'head changed while loading')
    model.to(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source = {str(p.relative_to(ROOT)): file_sha256(p) for p in
              (Path(__file__), ROOT/'scripts/run_continuation_v2.py', ROOT/'scripts/train_continuation_v2.py')}
    nav_sha = file_sha256(args.navigation)
    nav_report = json.loads(args.navigation.read_text())
    runtime = {'torch_version': str(torch.__version__), 'device': args.device,
               'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None}
    require(nav_report['metadata']['torch_version'] == runtime['torch_version'], 'Torch runtime differs')
    if args.device == 'cuda':
        require(nav_report['metadata']['gpu'] == runtime['gpu'], 'GPU runtime differs')
    result = compare(dataset, model, metadata['thresholds'], nav_report,
                     head_sha256=head_sha, identity=args.identity)
    require(file_sha256(args.navigation) == nav_sha and
            (not args.head or file_sha256(args.head) == head_sha), 'input changed while comparing')
    result.update(navigation_report_sha256=nav_sha, comparator_source_files=source,
                  comparator_source_sha256=object_sha256(source), runtime=runtime)
    navigation.exclusive_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'status': result['status'],
                      'checked_complete_trajectories': result['checked_complete_trajectories']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
