"""Semantic checks using an observable-only synthetic DUET rollout.

These checks exercise action execution and storage faults. They are not a
navigation benchmark or evidence of an improvement in SR/SPL.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_continuation_probe import state
from test_continuation_probe_integration_audit import (
    KnownGraph, NoEvaluator, NoGoalObservation,
)
from vln_improve import continuation_v2 as module
from vln_improve.continuation_probe import candidate_actions
from vln_improve.continuation_v2 import (
    SCHEMA, CausalContinuationHook, choose_action, eligible_state,
    load_bundle, save_bundle, selected_steps,
)


class Observation(NoGoalObservation):
    def get(self, key, default=None):
        if key in {'gt_path', 'distance', 'goal', 'reference_path'}:
            raise AssertionError('policy read a ground-truth field: ' + key)
        return super().get(key, default)


class AlwaysIntervene:
    mode = 'relative'

    def __init__(self, *, stop=False):
        self.calls = []
        self.stop = stop

    def __call__(self, features, history, progress):
        self.calls.append(tuple(x.detach().clone() for x in (features, history, progress)))
        scores = features.new_zeros((len(features), 2))
        scores[-1 if self.stop else 1] = scores.new_tensor([1., .5])
        return scores


def run_driver(*, condition='natural', reference=None, target_step=None,
               target_action=None, collect_features=False, predictor=None,
               natural_stop_step=11, no_left_step=None):
    trajectory = {'instr_id': 'train-only', 'path': [['v0']]}
    graph = KnownGraph()
    graph.node_positions = {}
    events = []

    def original_move(actions, graphs, observations, trajectories):
        if actions[0] is not None:
            trajectories[0]['path'].append(graphs[0].path(observations[0]['viewpoint'], actions[0]))

    # The policy receives no environment graph or evaluator. Teacher labels are
    # generated only when training features are explicitly requested.
    def teacher(observations, ids, ended, *, visited_masks, **kwargs):
        if not collect_features:
            raise AssertionError('online inference accessed training teacher')
        if kwargs['t'] >= natural_stop_step:
            return torch.tensor([0])
        return torch.tensor([next(i for i in range(1, len(ids[0])) if not visited_masks[0, i])])

    agent = SimpleNamespace(args=SimpleNamespace(max_action_len=15, fusion='dynamic'),
        feedback='argmax', make_equiv_action=original_move, env=NoEvaluator(),
        _teacher_action_r4r=teacher)
    hook = CausalContinuationHook(agent, condition=condition, seed=0,
        reference=reference, target_step=target_step, target_action=target_action,
        collect_features=collect_features, predictor=predictor)
    visited, current = set(), 'v0'
    for step in range(15):
        current_index = int(current[1:])
        visited.add(current)
        # Keep every previously discovered frontier after a backward detour.
        # Rebuilding the inventory only from the current node would create a
        # spurious early STOP unrelated to the hook under test.
        frontier = max(int(node[1:]) for node in visited)+3
        ids = [None] + ['v'+str(i) for i in range(frontier)]
        for node in ids[1:]:
            graph.node_positions[node] = (float(node[1:]), 0., 0.)
        mask = torch.ones(1, len(ids), dtype=torch.bool)
        visited_mask = torch.tensor([[vp in visited for vp in ids]], dtype=torch.bool)
        logits = torch.full((1, len(ids)), -torch.inf)
        logits[0, 0] = 1. if step == 0 else -2.
        preferred = next(vp for vp in ids[1:]
                         if vp not in visited and int(vp[1:]) > current_index)
        for i, vp in enumerate(ids[1:], 1):
            if vp not in visited:
                logits[0, i] = 3. if vp == preferred else 1.
        if step >= natural_stop_step:
            logits[0, 0] = 6.
        no_left = step == no_left_step
        # Time/action markers let the test identify past chosen actions and the
        # current pre-action token without deriving expected history from hook state.
        markers = torch.arange(len(ids), dtype=torch.float32) + 100*step
        embeds = torch.stack((markers, markers+.25), -1).unsqueeze(0)
        nav = {'gmap_vpids': [ids], 'vp_cand_vpids': [ids], 'gmap_masks': mask,
            'gmap_visited_masks': visited_mask, 'vp_nav_masks': ~visited_mask,
            'gmap_pos_fts': torch.zeros(1, len(ids), 3), 'no_vp_left': [no_left]}
        original = {'fused_logits': logits, 'global_logits': logits.clone(),
            'local_logits': logits.clone(), 'gmap_embeds': embeds, 'vp_embeds': embeds.clone()}
        before = {k: v.clone() for k, v in original.items()}
        ob = Observation(instr_id='train-only', scan='synthetic-graph', viewpoint=current,
            heading=0., elevation=0., viewIndex=12, gt_path='FORBIDDEN', distance='FORBIDDEN')
        changed = hook(nav, original, [ob], [False], step, [trajectory])
        for name, value in before.items():
            assert torch.equal(original[name], value), 'hook mutated frozen DUET output: '+name
        assert changed['gmap_embeds'] is original['gmap_embeds']
        probs = changed['fused_logits'].softmax(1)
        graph.node_stop_scores[current] = {'stop': float(probs[0, 0])}
        action = None if no_left or step == 14 else ids[int(changed['fused_logits'].argmax())]
        hook.make_equiv_action([action], [graph], [ob], [trajectory])
        events.append({'step': step, 'returned_original_object': changed is original,
            'action': action, 'stop_after_restore': graph.node_stop_scores[current]['stop']})
        if action is None:
            anchor = max(graph.node_stop_scores, key=lambda vp: graph.node_stop_scores[vp]['stop'])
            if anchor != current:
                trajectory['path'].append(graph.path(current, anchor))
            return hook.finish([trajectory]), events, hook
        current = action
    raise AssertionError('driver exceeded the original 15-decision budget')


@pytest.mark.parametrize('condition,expected', [
    ('natural', [0, 3, 11]), ('perturb_step2', [3, 6, 11]),
    ('early_two', [3, 6, 11]), ('late_three', [7, 10, 11]),
])
def test_online_eligibility_depends_on_current_step_stop_and_completed_schedule(condition, expected):
    states = [state(i, raw_argmax_action=None if i == 11 else 'a') for i in range(12)]
    assert selected_steps({'condition': condition, 'states': states}) == expected
    assert not eligible_state(state(14, forced_terminal=True, raw_argmax_action=None), condition)
    assert not eligible_state(state(expected[0], forced_terminal=True), condition)


@pytest.mark.parametrize('condition,target', [
    ('natural', 0), ('natural', 3), ('natural', 11),
    ('early_two', 3), ('early_two', 6), ('late_three', 7), ('late_three', 10),
])
def test_original_action_anchor_reproduces_all_observed_states_and_complete_path(condition, target):
    reference, _, _ = run_driver(condition=condition)
    action = reference['states'][target]['executed_action']
    branch, _, _ = run_driver(condition=condition, reference=reference,
                             target_step=target, target_action=action)
    assert branch['path'] == reference['path']
    assert branch['states'] == reference['states']
    assert branch['terminal_stop_scores'] == reference['terminal_stop_scores']
    assert len(branch['prefix_checks']) == target+1
    assert all(item['max_absolute_logit_error'] == 0 for item in branch['prefix_checks'])


@pytest.mark.parametrize('condition,schedule,target', [
    ('early_two', [1, 2], 3), ('late_three', [2, 4, 6], 7),
])
def test_all_perturbations_are_shared_prefix_and_post_intervention_is_pure_duet(condition, schedule, target):
    reference, _, _ = run_driver(condition=condition)
    assert [item['step'] for item in reference['perturbations']] == schedule
    alternative = next(a for a in candidate_actions(reference['states'][target])
        if a is not None and a != reference['states'][target]['executed_action'])
    branch, events, _ = run_driver(condition=condition, reference=reference,
                                  target_step=target, target_action=alternative)
    assert branch['states'][:target] == reference['states'][:target]
    assert branch['states'][target]['executed_action'] == alternative
    assert all(s['executed_action'] == s['raw_argmax_action'] for s in branch['states'][target+1:])
    assert all(event['returned_original_object'] for event in events[target+1:])
    for step in schedule:
        assert events[step]['stop_after_restore'] == branch['states'][step]['original_stop_probability']
    with pytest.raises(ValueError, match='follow every scheduled perturbation'):
        run_driver(condition=condition, reference=reference, target_step=schedule[-1],
                   target_action=None)


def test_causal_history_excludes_current_action_and_future_tokens_and_preserves_cached_snapshot():
    _, _, hook = run_driver(collect_features=True)
    early = hook.records[0]
    assert early['history_features'].shape[0] == 1
    assert early['history_features'][0, 0].item() == 0
    third = hook.records[3]
    assert third['history_features'][:, 0].tolist() == [0., 2., 100., 103., 200., 204., 300.]
    assert third['progress'][2].item() == pytest.approx(3/100, abs=2e-5)
    assert all(token < 301 for token in third['history_features'][:, 0].tolist())
    assert len(hook.history) > third['history_features'].shape[0]
    assert early['history_features'].shape[0] == 1  # Later rollout did not mutate the saved prefix.


def test_online_prediction_has_no_goal_teacher_or_future_reference_and_changes_only_once():
    predictor = AlwaysIntervene()
    result, events, _ = run_driver(predictor=predictor)
    assert len(predictor.calls) == 1
    assert result['intervention']['step'] == 0
    assert result['states'][0]['executed_action'] != result['states'][0]['raw_argmax_action']
    assert predictor.calls[0][1][:, 0].tolist() == [0.]
    assert all(event['returned_original_object'] for event in events[1:])
    assert all(s['executed_action'] == s['raw_argmax_action'] for s in result['states'][1:])


def test_online_inputs_exactly_match_quantized_training_prefix_inputs():
    class KeepPredictor(AlwaysIntervene):
        def __call__(self, features, history, progress):
            super().__call__(features, history, progress)
            return features.new_zeros((len(features), 2))
    reference, _, cached_hook = run_driver(collect_features=True)
    predictor = KeepPredictor()
    deployed, _, _ = run_driver(predictor=predictor)
    assert deployed['path'] == reference['path']
    assert deployed['intervention'] is None
    assert len(predictor.calls) == len(cached_hook.records)
    for step, actual in zip(sorted(cached_hook.records), predictor.calls):
        expected = cached_hook.records[step]
        for name, value in zip(('features', 'history_features', 'progress'), actual):
            assert value.dtype == torch.float16
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_perturbed_online_prediction_waits_until_last_external_override():
    predictor = AlwaysIntervene()
    result, events, _ = run_driver(condition='late_three', predictor=predictor)
    assert [item['step'] for item in result['perturbations']] == [2, 4, 6]
    assert result['intervention']['step'] == 7
    assert len(predictor.calls) == 1
    assert predictor.calls[0][1].shape[0] == 15
    assert all(event['returned_original_object'] for event in events[8:])


def test_forced_stop_preserves_original_probability_and_historical_return():
    reference, _, _ = run_driver()
    branch, events, _ = run_driver(reference=reference, target_step=3, target_action=None)
    assert branch['path'][-1] == ['v2', 'v1', 'v0']
    assert events[-1]['stop_after_restore'] == branch['states'][-1]['original_stop_probability']
    assert events[-1]['stop_after_restore'] < .01


def test_forced_terminal_never_produces_moving_branch_or_invented_late_state():
    for options in ({'natural_stop_step': 99}, {'natural_stop_step': 99, 'no_left_step': 3}):
        reference, _, _ = run_driver(**options)
        last = reference['states'][-1]
        assert last['forced_terminal'] and reference['decision_count'] <= 15
        assert last['step'] not in selected_steps(reference)
        with pytest.raises(ValueError, match='follow every scheduled perturbation'):
            run_driver(reference=reference, target_step=last['step'], target_action=None, **options)
    early, _, _ = run_driver(condition='late_three', natural_stop_step=1)
    assert early['perturbations'] == []
    assert selected_steps(early) == []


def test_corrupt_prefix_fails_before_executing_requested_branch():
    reference, _, _ = run_driver(condition='early_two')
    reference = copy.deepcopy(reference)
    reference['states'][2]['logits'][0] += .1
    with pytest.raises(ValueError, match='prefix replay differs'):
        run_driver(condition='early_two', reference=reference, target_step=3,
                   target_action=reference['states'][3]['executed_action'])


def test_predictor_cannot_receive_training_teacher_or_reference_rollouts():
    with pytest.raises(ValueError, match='prediction cannot access training labels'):
        run_driver(predictor=AlwaysIntervene(), collect_features=True)
    reference, _, _ = run_driver()
    with pytest.raises(ValueError, match='prediction cannot access training labels'):
        run_driver(predictor=AlwaysIntervene(), reference=reference, target_step=0,
                   target_action=reference['states'][0]['executed_action'])


def test_score_gates_keep_baseline_and_absolute_subtraction_matches_relative_decisions():
    assert choose_action(torch.zeros(4, 2), 'relative', 0., 0.) == 0
    assert choose_action(torch.tensor([[9., 9.], [.4, -.1], [-.1, .8]]), 'relative', 0., 0.) == 0
    gains = torch.tensor([[0., 0.], [.3, .1], [.2, .2], [.9, -.01]])
    assert choose_action(gains, 'relative', .1, 0.) == 1
    assert choose_action(gains+torch.tensor([.4, .6]), 'absolute', .1, 0.) == 1
    assert choose_action(gains, 'relative', .3, 0.) == 0
    assert choose_action(torch.tensor([[0., 0.], [.2, .0]]), 'relative', 0., 0.) == 1


def test_teacher_selection_uses_predicted_action_probability_without_outcome_channel():
    values = torch.tensor([[0., 1e6], [2., -1e6], [-1., 9e6]])
    assert choose_action(values, 'teacher', 0., 0.) == 1
    assert choose_action(values, 'teacher', .99, 0.) == 0
    assert choose_action(torch.zeros(3, 2), 'teacher', 0., 0.) == 0


def test_predicted_sr_has_lexicographic_priority_then_spl_and_stable_index():
    scores = torch.tensor([[0., 0.], [.11, 0.], [.10, .90]])
    assert choose_action(scores, 'relative', 0., 0.) == 1
    scores = torch.tensor([[0., 0.], [.11, .20], [.11, .30], [.11, .30]])
    assert choose_action(scores, 'relative', 0., 0.) == 2


@pytest.mark.parametrize('scores', [torch.zeros(3), torch.zeros(2, 3), torch.tensor([[0., float('nan')]])])
def test_invalid_predictions_rejected(scores):
    with pytest.raises(ValueError, match='invalid action scores'):
        choose_action(scores, 'relative', 0., 0.)


def bundle():
    return {'schema': SCHEMA, 'records': [{'features': torch.arange(6).reshape(2, 3).half()}]}


def test_bundle_empty_local_restores_verified_tensor_bytes(tmp_path):
    local, backup = tmp_path/'local', tmp_path/'persistent'
    checks = []
    pointer = save_bundle(local, backup, {'instr_id': 'i', 'step': 3}, bundle(), lambda: checks.append(True))
    assert len(checks) == 2
    (local/pointer['file']).unlink()
    restored = load_bundle(tmp_path/'cold', backup, pointer, lambda: checks.append(True))
    assert len(checks) == 4
    torch.testing.assert_close(restored['records'][0]['features'], bundle()['records'][0]['features'])
    assert (tmp_path/'cold'/pointer['file']).read_bytes() == (backup/pointer['file']).read_bytes()


def test_corrupt_persistent_bundle_is_rejected_even_when_local_copy_is_good(tmp_path):
    local, backup = tmp_path/'local', tmp_path/'persistent'
    pointer = save_bundle(local, backup, {'instr_id': 'i'}, bundle(), lambda: None)
    (backup/pointer['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='persistent bundle corrupt'):
        load_bundle(local, backup, pointer)


def test_corrupt_local_bundle_is_not_silently_trusted(tmp_path):
    local, backup = tmp_path/'local', tmp_path/'persistent'
    pointer = save_bundle(local, backup, {'instr_id': 'i'}, bundle(), lambda: None)
    (local/pointer['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='local bundle corrupt'):
        load_bundle(local, backup, pointer)
    # Removing a diagnosed bad local copy permits recovery from the verified backup.
    (local/pointer['file']).unlink()
    restored = load_bundle(local, backup, pointer)
    assert restored['schema'] == SCHEMA


def test_bundle_pointer_schema_count_and_path_escape_are_rejected(tmp_path):
    local, backup = tmp_path/'local', tmp_path/'persistent'
    pointer = save_bundle(local, backup, {'instr_id': 'i'}, bundle(), lambda: None)
    with pytest.raises(ValueError, match='identity differs'):
        load_bundle(local, backup, dict(pointer, records=2))
    with pytest.raises(ValueError, match='unsafe bundle path'):
        load_bundle(local, backup, dict(pointer, file='../outside.pt'))
    wrong = bundle(); wrong['schema'] = 'wrong'
    wrong_pointer = save_bundle(local, backup, {'instr_id': 'j'}, wrong, lambda: None)
    with pytest.raises(ValueError, match='identity differs'):
        load_bundle(local, backup, wrong_pointer)


def test_bundle_copy_failure_keeps_local_recovery_point_without_success_pointer(tmp_path, monkeypatch):
    local, backup = tmp_path/'local', tmp_path/'persistent'
    def fail(*args):
        raise OSError('persistent disk unavailable')
    monkeypatch.setattr(module, 'verified_copy', fail)
    with pytest.raises(OSError, match='disk unavailable'):
        save_bundle(local, backup, {'instr_id': 'i'}, bundle(), lambda: None)
    files = list(local.glob('bundle-*.pt'))
    assert len(files) == 1
    assert torch.load(files[0], weights_only=True)['schema'] == SCHEMA
    assert not backup.exists()


def test_bundle_postcopy_mount_failure_is_not_acknowledged(tmp_path):
    checks = []
    def checker():
        checks.append(True)
        if len(checks) == 2:
            raise RuntimeError('backup mount identity changed')
    with pytest.raises(RuntimeError, match='mount identity changed'):
        save_bundle(tmp_path/'local', tmp_path/'persistent', {'instr_id': 'i'}, bundle(), checker)
    assert len(list((tmp_path/'local').glob('bundle-*.pt'))) == 1
