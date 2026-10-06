"""Independent probe-hook checks with a small observable-only rollout driver.

The driver reproduces upstream action/STOP bookkeeping and a known graph path;
it is not a navigation benchmark or evidence of task improvement.
"""
import copy
import math
from types import SimpleNamespace

import pytest
import torch

from vln_improve.continuation_probe import (
    ContinuationHook, candidate_actions, select_state_steps,
)


class NoGoalObservation(dict):
    def __getitem__(self, key):
        if key in {'gt_path', 'distance', 'goal', 'reference_path'}:
            raise AssertionError('policy read a ground-truth field: ' + key)
        return super().__getitem__(key)


class NoEvaluator:
    def __getattr__(self, key):
        raise AssertionError('policy tried to access evaluator: ' + key)


class KnownGraph:
    def __init__(self):
        self.node_stop_scores = {}
        self.graph = self

    def path(self, source, target):
        # Forward moves of one/two nodes are known edges. Returns use known
        # consecutive edges. As in DUET, omit source and include target.
        a, b = int(source[1:]), int(target[1:])
        if a == b:
            return []
        if b > a:
            return [target]
        return ['v'+str(i) for i in range(a-1, b-1, -1)]


def run_driver(*, condition='natural', reference=None, target_step=None,
               target_action=None, natural_stop_step=4, no_left_step=None):
    trajectory = {'instr_id': 'train-only', 'path': [['v0']]}
    graph = KnownGraph()
    events = []

    def original_move(actions, graphs, observations, trajectories):
        if actions[0] is not None:
            trajectories[0]['path'].append(graphs[0].path(observations[0]['viewpoint'], actions[0]))

    agent = SimpleNamespace(args=SimpleNamespace(max_action_len=15, fusion='dynamic'),
                            feedback='argmax', make_equiv_action=original_move,
                            env=NoEvaluator())
    hook = ContinuationHook(agent, condition=condition, seed=0, reference=reference,
                            target_step=target_step, target_action=target_action)
    visited = set()
    current = 'v0'
    for step in range(15):
        index = int(current[1:])
        visited.add(current)
        ids = [None] + ['v'+str(i) for i in range(index+3)]
        mask = torch.ones(1, len(ids), dtype=torch.bool)
        visited_mask = torch.tensor([[vp in visited for vp in ids]], dtype=torch.bool)
        logits = torch.full((1, len(ids)), -torch.inf)
        logits[0, 0] = 1. if step == 0 else -2.
        for i, vp in enumerate(ids[1:], 1):
            if vp not in visited:
                logits[0, i] = 3. if vp == 'v'+str(index+1) else 1.
        if step >= natural_stop_step:
            logits[0, 0] = 6.
        no_left = step == no_left_step
        nav = {'gmap_vpids': [ids], 'gmap_masks': mask,
               'gmap_visited_masks': visited_mask, 'no_vp_left': [no_left]}
        sentinel = object()
        original = {'fused_logits': logits, 'gmap_embeds': sentinel}
        ob = NoGoalObservation(instr_id='train-only', scan='synthetic-graph', viewpoint=current,
                               heading=0., elevation=0., viewIndex=12,
                               gt_path='FORBIDDEN', distance='FORBIDDEN')
        changed = hook(nav, original, [ob], [False], step, [trajectory])
        assert changed['gmap_embeds'] is sentinel
        assert torch.equal(original['fused_logits'], logits)
        probs = torch.softmax(changed['fused_logits'], 1)
        graph.node_stop_scores[current] = {'stop': float(probs[0, 0])}
        action = ids[int(changed['fused_logits'].argmax())]
        if no_left or step == 14:
            action = None
        hook.make_equiv_action([action], [graph], [ob], [trajectory])
        events.append({'step': step, 'returned_original_object': changed is original,
                       'action': action, 'stop_after_restore': graph.node_stop_scores[current]['stop']})
        if action is None:
            anchor = max(graph.node_stop_scores, key=lambda vp: graph.node_stop_scores[vp]['stop'])
            if anchor != current:
                trajectory['path'].append(graph.path(current, anchor))
            return hook.finish([trajectory]), events
        current = action
    raise AssertionError('driver failed to finish within original budget')


def test_forced_stop_preserves_old_endpoint_evidence_and_charges_return():
    reference, _ = run_driver()
    branch, events = run_driver(reference=reference, target_step=3, target_action=None)
    assert branch['states'][-1]['viewpoint'] == 'v3'
    assert branch['states'][-1]['original_stop_probability'] < .01
    assert events[-1]['stop_after_restore'] == branch['states'][-1]['original_stop_probability']
    assert branch['path'][-1] == ['v2', 'v1', 'v0']
    assert branch['terminal_stop_scores']['v3'] < branch['terminal_stop_scores']['v0']
    assert branch['path'][0] == ['v0']  # Keep and charge the real executed prefix.


def test_forced_move_from_natural_stop_then_pure_duet_and_original_budget():
    reference, _ = run_driver()
    terminal = reference['states'][-1]
    alternative = next(a for a in candidate_actions(terminal) if a is not None)
    branch, events = run_driver(reference=reference, target_step=4, target_action=alternative)
    assert branch['states'][4]['executed_action'] == alternative
    assert branch['states'][4]['original_stop_probability'] > .9
    assert events[4]['stop_after_restore'] == branch['states'][4]['original_stop_probability']
    assert branch['decision_count'] == 6
    assert all(x['returned_original_object'] for x in events[5:])
    assert branch['decision_count'] <= 15


def test_step14_and_no_frontier_reject_fake_moving_branches():
    for options in ({'natural_stop_step': 99}, {'natural_stop_step': 99, 'no_left_step': 3}):
        reference, _ = run_driver(**options)
        terminal = reference['states'][-1]
        assert terminal['forced_terminal']
        assert candidate_actions(terminal) == [None]
        assert terminal['raw_argmax_action'] is not None
        with pytest.raises(ValueError, match='outside frozen candidate'):
            run_driver(reference=reference, target_step=terminal['step'],
                       target_action=terminal['raw_argmax_action'], **options)
        anchor, _ = run_driver(reference=reference, target_step=terminal['step'],
                               target_action=None, **options)
        assert anchor['anchor_trajectory_and_all_states_equal']


@pytest.mark.parametrize('target', [0, 3, 4])
def test_natural_original_action_anchor_reproduces_complete_rollout(target):
    reference, _ = run_driver()
    action = reference['states'][target]['executed_action']
    branch, _ = run_driver(reference=reference, target_step=target, target_action=action)
    assert branch['anchor_trajectory_and_all_states_equal']
    assert branch['path'] == reference['path']
    assert branch['terminal_stop_scores'] == reference['terminal_stop_scores']


def test_perturbation_is_shared_prefix_and_never_reapplied_after_target():
    reference, _ = run_driver(condition='perturb_step2', natural_stop_step=7)
    assert reference['perturbation']['applied']
    assert select_state_steps(reference['states'], 'perturb_step2') == [3, 6, 7]
    target = 3
    anchor, _ = run_driver(condition='perturb_step2', reference=reference, target_step=target,
                          target_action=reference['states'][target]['executed_action'], natural_stop_step=7)
    assert anchor['anchor_trajectory_and_all_states_equal']
    alternative = next(a for a in candidate_actions(reference['states'][target])
                       if a is not None and a != reference['states'][target]['executed_action'])
    branch, events = run_driver(condition='perturb_step2', reference=reference, target_step=target,
                               target_action=alternative, natural_stop_step=7)
    assert [s['executed_action'] for s in branch['states'][:3]] == [s['executed_action'] for s in reference['states'][:3]]
    assert all(x['returned_original_object'] for x in events[target+1:])
    assert all(s['executed_action'] == s['raw_argmax_action'] for s in branch['states'][target+1:])


def test_early_termination_has_no_invented_post_perturbation_state():
    reference, _ = run_driver(condition='perturb_step2', natural_stop_step=1)
    assert not reference['perturbation']['applied']
    assert select_state_steps(reference['states'], 'perturb_step2') == []
    with pytest.raises(ValueError, match='outside frozen state'):
        run_driver(condition='perturb_step2', reference=reference, target_step=1,
                   target_action=None, natural_stop_step=1)


def test_replay_state_mismatch_fails_before_forcing_action():
    reference, _ = run_driver()
    tampered = copy.deepcopy(reference)
    tampered['states'][1]['viewpoint'] = 'unexpected'
    with pytest.raises(ValueError, match='prefix replay differs: viewpoint'):
        run_driver(reference=tampered, target_step=3,
                   target_action=tampered['states'][3]['executed_action'])
