"""Reject equal-average navigation reports with different causal executions."""
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts import compare_continuation_v2 as comparison


class Scores(torch.nn.Module):
    mode = 'relative'

    def score_record(self, record):
        return record['features']


def metrics(success, spl):
    return dict(success=float(success), spl=spl, nav_error=1. if success else 5.,
        oracle_success=float(success), oracle_error=1. if success else 5.,
        action_steps=2., trajectory_steps=2., trajectory_lengths=4.,
        DTW=2., nDTW=.7, SDTW=.7*success, CLS=.6)


def fixture(identity=False):
    conditions = list(comparison.SCHEDULES)
    selection = dict(split='train_dev', instr_ids=['a', 'b'], scan_ids=['scene'],
                     count=2, conditions=conditions, seed=0)
    bundles = []
    for instr in selection['instr_ids']:
        for condition in conditions:
            successful = instr == 'b'
            ref = dict(instr_id=instr, scan_id='scene', condition=condition,
                metrics=metrics(successful, .7 if successful else 0.),
                path=[['start'], ['keep'], ['end']], decision_count=3,
                perturbations=[] if condition == 'natural' else [{'step': 2, 'action': 'p'}])
            scores = [[0., 0.], [0., 0.]] if successful or identity else [[0., 0.], [.3, .4]]
            record = dict(features=torch.tensor(scores), step=7,
                          candidate_actions=['keep', 'rescue'])
            branches = [dict(target_step=7, target_action='rescue', metrics=metrics(1, .8),
                path=[['start'], ['rescue'], ['goal']], decision_count=3)]
            bundles.append(dict(reference=ref, records=[record], branches=branches))
    dataset = SimpleNamespace(manifest={'selection': selection}, bundles=bundles,
                              identity={'manifest_sha256': 'm'*64})
    model, thresholds = Scores(), dict(sr=.1 if not identity else 0., spl=0.)
    report = {'schema': comparison.navigation.REPORT_SCHEMA,
        'metadata': dict(split='train_dev', subset=True, head_sha256=None if identity else 'h'*64,
            dataset_manifest_sha256='m'*64, head_metadata={'thresholds': thresholds},
            mode='continuation_v2_identity' if identity else 'continuation_v2_navigation',
            zero_comparator_full_trajectory_identity=True if identity else None,
            identity_checked_rollouts=8 if identity else 0,
            selection=dict(selection, condition_seed=0),
            code_files=comparison.navigation.code_identity()[0],
            code_sha256=comparison.navigation.code_identity()[1]), 'conditions': {}}
    for condition in conditions:
        episodes, trajectories, interventions = [], [], []
        for bundle in bundles:
            ref = bundle['reference']
            if ref['condition'] != condition:
                continue
            chosen, choice = comparison.choose_cached(model, bundle, thresholds)
            episodes.append(dict(instr_id=ref['instr_id'], scan_id='scene', **chosen['metrics']))
            trajectories.append(dict(instr_id=ref['instr_id'], trajectory=chosen['path']))
            interventions.append(dict(instr_id=ref['instr_id'], scan_id='scene',
                intervention=choice, perturbations=ref['perturbations'], decision_count=3))
        report['conditions'][condition] = dict(episodes=episodes, trajectories=trajectories,
            interventions=interventions, intervention_count=sum(i['intervention'] is not None for i in interventions),
            summary=comparison.navigation.summarize(episodes))
    refresh_natural(report)
    return dataset, model, thresholds, copy.deepcopy(report)


def refresh_natural(report):
    for key in ('summary', 'episodes', 'trajectories'):
        report[key] = copy.deepcopy(report['conditions']['natural'][key])


def run(data, model, thresholds, report, identity=False):
    return comparison.compare(data, model, thresholds, report,
                              head_sha256=None if identity else 'h'*64, identity=identity)


def test_all_paths_actions_metrics_and_summary_match_with_rescue_counts():
    data, model, thresholds, report = fixture()
    model.train()
    result = run(data, model, thresholds, report)
    assert result['status'] == 'passed' and model.training
    assert result['checked_complete_trajectories'] == 8
    assert result['conditions']['natural']['rescued'] == 1
    assert result['conditions']['natural']['harmed'] == 0
    assert all(v == 0 for v in result['maximum_metric_errors'].values())


def test_zero_head_and_no_intervention_cases_use_original_references():
    result = run(*fixture(identity=True), identity=True)
    assert all(c['interventions'] == 0 for c in result['conditions'].values())


def test_identity_requires_all_raw_reference_rollouts():
    data, model, thresholds, report = fixture(identity=True)
    report['metadata']['identity_checked_rollouts'] = 7
    with pytest.raises(ValueError, match='identity reference rollout coverage'):
        run(data, model, thresholds, report, identity=True)


@pytest.mark.parametrize('defect', ['path', 'metric', 'first_action', 'step', 'baseline_action',
    'presence', 'score', 'perturbation', 'decision_count', 'duplicate', 'missing', 'extra',
    'head', 'manifest', 'threshold', 'source', 'selection', 'seed', 'condition',
    'summary', 'top_level', 'intervention_count', 'scene', 'mode'])
def test_semantically_wrong_report_is_rejected_even_without_checksum_failure(defect):
    data, model, thresholds, report = fixture()
    value = report['conditions']['natural']
    if defect == 'path': value['trajectories'][0]['trajectory'][-1] = ['wrong']
    elif defect == 'metric': value['episodes'][0]['spl'] -= .1
    elif defect in ('first_action', 'step', 'baseline_action'):
        key = 'action' if defect == 'first_action' else defect
        value['interventions'][0]['intervention'][key] = 'wrong'
    elif defect == 'presence': value['interventions'][0]['intervention'] = None
    elif defect == 'score': value['interventions'][0]['intervention']['scores'][1][0] += .01
    elif defect == 'perturbation': value['interventions'][0]['perturbations'] = [{'step': 0, 'action': 'p'}]
    elif defect == 'decision_count': value['interventions'][0]['decision_count'] += 1
    elif defect == 'duplicate': value['episodes'].append(copy.deepcopy(value['episodes'][0]))
    elif defect == 'missing': value['trajectories'].pop()
    elif defect == 'extra': value['interventions'].append(dict(instr_id='extra'))
    elif defect == 'head': report['metadata']['head_sha256'] = 'other'
    elif defect == 'manifest': report['metadata']['dataset_manifest_sha256'] = 'other'
    elif defect == 'threshold': report['metadata']['head_metadata']['thresholds'] = dict(sr=.5, spl=0.)
    elif defect == 'source': report['metadata']['code_sha256'] = 'other'
    elif defect == 'selection': report['metadata']['selection']['count'] = 3
    elif defect == 'seed': report['metadata']['selection']['condition_seed'] = 1
    elif defect == 'condition': report['conditions'].pop('late_three')
    elif defect == 'summary': value['summary']['sr'] -= 1
    elif defect == 'intervention_count': value['intervention_count'] += 1
    elif defect == 'scene': value['episodes'][0]['scan_id'] = 'other'
    elif defect == 'mode': report['metadata']['mode'] = 'other'
    refresh_natural(report)
    if defect == 'top_level': report['summary']['sr'] -= 1
    with pytest.raises(ValueError):
        run(data, model, thresholds, report)


def test_equal_aggregate_sr_does_not_hide_per_instruction_metric_swap():
    data, model, thresholds, report = fixture(identity=True)
    rows = report['conditions']['natural']['episodes']
    rows[0]['success'], rows[1]['success'] = rows[1]['success'], rows[0]['success']
    refresh_natural(report)
    with pytest.raises(ValueError, match='numeric value differs'):
        run(data, model, thresholds, report, identity=True)


def test_labels_cannot_change_first_chosen_action_and_later_states_are_not_scored():
    data, model, thresholds, _ = fixture()
    bundle = data.bundles[0]
    # Sorting needs timestamps, while prediction must never read later features.
    class LaterRecord(dict):
        def __getitem__(self, key):
            if key != 'step': raise AssertionError('future observation read')
            return super().__getitem__(key)
    bundle['records'].append(LaterRecord(step=10))
    _, before = comparison.choose_cached(model, bundle, thresholds)
    bundle['branches'][0]['metrics'] = metrics(0, 0.)
    _, after = comparison.choose_cached(model, bundle, thresholds)
    assert before == after and before['step'] == 7 and before['action'] == 'rescue'


def test_missing_selected_branch_and_duplicate_cached_case_fail():
    data, model, thresholds, report = fixture()
    data.bundles[0]['branches'].append(copy.deepcopy(data.bundles[0]['branches'][0]))
    with pytest.raises(ValueError, match='one executed continuation'):
        run(data, model, thresholds, report)
    data, model, thresholds, report = fixture()
    data.bundles.append(copy.deepcopy(data.bundles[0]))
    with pytest.raises(ValueError, match='duplicate cached'):
        run(data, model, thresholds, report)
