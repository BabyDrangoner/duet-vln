"""Causal features and replay for the second, trainable continuation experiment."""
from __future__ import annotations

import copy
import io
import math
from pathlib import Path

import torch

from .continuation_probe import ContinuationHook, assert_anchor, assert_prefix_state, candidate_actions
from .features import build_features
from .intervention_runtime import atomic_bytes, verified_copy
from .protocol import file_sha256, object_sha256

SCHEMA = 'e3_causal_continuation_v2'
SCHEDULES = {'natural': (), 'perturb_step2': (2,), 'early_two': (1, 2), 'late_three': (2, 4, 6)}


def eligible_state(state, condition):
    schedule = SCHEDULES[condition]
    first = max(schedule) + 1 if schedule else 0
    fixed = (first, first + 3)
    return (state['step'] >= first and not state['forced_terminal']
            and (state['step'] in fixed or state['raw_argmax_action'] is None))


def selected_steps(reference):
    return [s['step'] for s in reference['states'] if eligible_state(s, reference['condition'])]


def choose_action(scores, mode, sr_threshold, spl_threshold):
    """Scores alone determine the action; index zero explicitly keeps DUET."""
    if scores.ndim != 2 or scores.shape[1] != 2 or not torch.isfinite(scores).all():
        raise ValueError('invalid action scores')
    if mode == 'absolute':
        scores = scores - scores[:1]
    elif mode == 'teacher':
        p = scores[:, 0].softmax(0)
        gains = p - p[0]
        allowed = gains > sr_threshold
        allowed[0] = False
        return int(gains.masked_fill(~allowed, -torch.inf).argmax()) if allowed.any() else 0
    elif mode != 'relative':
        raise ValueError('unknown comparator mode')
    allowed = (scores[:, 0] > sr_threshold) & (scores[:, 1] >= spl_threshold)
    allowed[0] = False
    indices = allowed.nonzero(as_tuple=False).flatten().tolist()
    return max(indices, key=lambda i: (float(scores[i, 0]), float(scores[i, 1]), -i)) if indices else 0


class CausalContinuationHook(ContinuationHook):
    """Every perturbation precedes intervention; after intervention DUET is frozen."""
    def __init__(self, agent, *, condition='natural', seed=0, reference=None,
                 target_step=None, target_action=None, collect_features=False,
                 predictor=None, sr_threshold=0.0, spl_threshold=0.0):
        super().__init__(agent, condition='natural', seed=seed)
        if condition not in SCHEDULES:
            raise ValueError('unknown continuation condition')
        if (reference is None) != (target_step is None):
            raise ValueError('branch requires reference and target step')
        if reference is not None:
            if reference['condition'] != condition or reference['seed'] != seed:
                raise ValueError('branch condition or seed differs')
            if target_step not in selected_steps(reference):
                raise ValueError('intervention must follow every scheduled perturbation')
            if target_action not in candidate_actions(reference['states'][target_step]):
                raise ValueError('illegal branch action')
        if predictor is not None and (reference is not None or collect_features):
            raise ValueError('prediction cannot access training labels or reference futures')
        self.condition = condition
        self.replay, self.branch_step, self.branch_action = reference, target_step, target_action
        self.collect_features, self.predictor = collect_features, predictor
        self.sr_threshold, self.spl_threshold = sr_threshold, spl_threshold
        self.records, self.history = {}, []
        self.applied_changes, self.intervention = [], None
        self._last_features = None
        self.executed_length = 0.0

    def __call__(self, nav_inputs, nav_outs, observations, ended, step, trajectories):
        # The parent records immutable raw scores and restores raw STOP evidence.
        condition = self.condition
        self.condition = 'natural'
        try:
            super().__call__(nav_inputs, nav_outs, observations, ended, step, trajectories)
        finally:
            self.condition = condition
        state = self.states[-1]
        selected = state['expected_executed_action']
        needed = self.collect_features or self.predictor is not None
        if needed:
            features, _, _ = build_features(nav_inputs, nav_outs)
            # Identical quantization during cache creation and online deployment.
            self._last_features = features[0].detach().half()
            self.history.append(self._last_features[0].clone())
        if self.replay is not None and step <= self.branch_step:
            expected = self.replay['states'][step]
            error = assert_prefix_state(state, expected)
            self.prefix_checks.append({'step': step, 'max_absolute_logit_error': error})
            selected = expected['executed_action'] if step < self.branch_step else self.branch_action
            if step == self.branch_step:
                self.target_reached = True
        elif self.replay is None and step in SCHEDULES[condition]:
            alternatives = [i for i, v in enumerate(state['candidate_vpids']) if i > 0
                and state['valid_mask'][i] and not state['visited_mask'][i]
                and state['logits'][i] is not None and v != state['raw_argmax_action']]
            if alternatives and not state['forced_terminal']:
                alternatives.sort(key=lambda i: state['candidate_vpids'][i])
                key = [SCHEMA, state['instr_id'], self.seed, condition, step,
                       [state['candidate_vpids'][i] for i in alternatives]]
                pick = alternatives[int(object_sha256(key), 16) % len(alternatives)]
                selected = state['candidate_vpids'][pick]
                self.applied_changes.append({'step': step, 'action': selected})
        if needed and eligible_state(state, condition):
            state_for_candidates = dict(state, executed_action=selected)
            actions = candidate_actions(state_for_candidates)
            indices = [state['candidate_vpids'].index(action) for action in actions]
            # Prefix length uses only traversed edges, never a target or unseen view.
            length = self.executed_length
            record = {'features': self._last_features[indices].clone(),
                      'history_features': torch.stack(self.history),
                      'progress': self._last_features.new_tensor([step/14, (15-step)/15, length/100, len(actions)/4]),
                      'candidate_actions': actions, 'step': step,
                      'instr_id': state['instr_id'], 'scan_id': state['scan_id'], 'condition': condition}
            if self.collect_features:
                # Target is saved separately; policy-facing feature construction is over.
                teacher = int(self.agent._teacher_action_r4r(observations,
                    nav_inputs['gmap_vpids'], ended, visited_masks=nav_inputs['gmap_visited_masks'],
                    imitation_learning=False, t=step, traj=trajectories)[0])
                record['teacher_target'] = indices.index(teacher) if teacher in indices else -1
                self.records[step] = {k: v.detach().cpu().half() if isinstance(v, torch.Tensor) else v
                                      for k, v in record.items()}
            if self.predictor is not None and self.intervention is None:
                scores = self.predictor(record['features'], record['history_features'], record['progress'])
                index = choose_action(scores, self.predictor.mode, self.sr_threshold, self.spl_threshold)
                if index:
                    selected = actions[index]
                    self.intervention = {'step': step, 'baseline_action': actions[0], 'action': selected,
                                         'scores': scores.detach().cpu().tolist()}
        if state['forced_terminal'] and selected is not None:
            raise ValueError('cannot move at forced termination')
        state['expected_executed_action'] = selected
        if selected == state['raw_argmax_action'] or state['forced_terminal']:
            return nav_outs
        index = state['candidate_vpids'].index(selected)
        if not state['valid_mask'][index] or state['visited_mask'][index] or state['logits'][index] is None:
            raise ValueError('selected action is not executable')
        changed = dict(nav_outs, fused_logits=nav_outs['fused_logits'].clone())
        changed['fused_logits'][0].fill_(-torch.inf)
        changed['fused_logits'][0, index] = 0
        return changed

    def make_equiv_action(self, actions, graphs, observations, trajectories):
        result = super().make_equiv_action(actions, graphs, observations, trajectories)
        # These positions belong to the discovered GraphMap. No evaluator graph,
        # shortest-to-goal distance, reference route, or future observation is read.
        if self.collect_features or self.predictor is not None:
            flat = [v for segment in trajectories[0]['path'] for v in segment]
            positions = graphs[0].node_positions
            self.executed_length = sum(math.dist(positions[a], positions[b])
                                       for a, b in zip(flat, flat[1:]) if a != b)
        if self._last_features is not None and actions[0] is not None:
            index = self.states[-1]['candidate_vpids'].index(actions[0])
            self.history.append(self._last_features[index].clone())
        return result

    def finish(self, trajectories):
        result = super().finish(trajectories)
        result.update(schema=SCHEMA, perturbations=self.applied_changes,
                      target_step=self.branch_step, target_action=self.branch_action,
                      intervention=self.intervention)
        if self.replay is not None:
            if not self.target_reached:
                raise ValueError('branch target not reached')
            if self.branch_action == self.replay['states'][self.branch_step]['executed_action']:
                assert_anchor(result, self.replay, metrics=False)
        return result


def save_bundle(local, backup, task, bundle, check_backup):
    """Write immutable tensor payload before acknowledging its small task pointer."""
    stream = io.BytesIO()
    torch.save(bundle, stream)
    raw = stream.getvalue()
    import hashlib
    digest = hashlib.sha256(raw).hexdigest()
    name = f'bundle-{object_sha256(task)}-{digest[:16]}.pt'
    check_backup()
    path = Path(local)/name
    atomic_bytes(path, raw)
    verified_copy(path, Path(backup)/name)
    check_backup()
    return {'file': name, 'sha256': digest, 'bytes': len(raw), 'records': len(bundle['records'])}


def load_bundle(local, backup, pointer, check_backup=lambda: None):
    name = pointer['file']
    if Path(name).name != name:
        raise ValueError('unsafe bundle path')
    check_backup()
    cloud, path = Path(backup)/name, Path(local)/name
    if cloud.stat().st_size != pointer['bytes'] or file_sha256(cloud) != pointer['sha256']:
        raise ValueError('persistent bundle corrupt')
    if not path.exists():
        verified_copy(cloud, path)
    if path.stat().st_size != pointer['bytes'] or file_sha256(path) != pointer['sha256']:
        raise ValueError('local bundle corrupt')
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value['schema'] != SCHEMA or len(value['records']) != pointer['records']:
        raise ValueError('bundle identity differs')
    check_backup()
    return value
