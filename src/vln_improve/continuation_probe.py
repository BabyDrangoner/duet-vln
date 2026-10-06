"""Training-only single-action counterfactuals with the complete frozen DUET continuation.

Policy-facing code never reads goals or evaluator state. Outcomes are attached by
the caller only after the simulator has completed the full executed trajectory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .intervention_runtime import atomic_bytes, verified_copy
from .protocol import file_sha256, object_sha256

SCHEMA = 'e3_continuation_probe_tasks_v1'
MAX_ACTION_LEN = 15


def select_state_steps(states, condition):
    if condition not in ('natural', 'perturb_step2') or not states:
        raise ValueError('invalid condition or empty reference')
    if [s['step'] for s in states] != list(range(len(states))):
        raise ValueError('reference decisions must be contiguous from zero')
    requested = [0, 3] if condition == 'natural' else [3, 6]
    requested.append(states[-1]['step'])
    minimum = 0 if condition == 'natural' else 3
    return sorted({s for s in requested if minimum <= s < len(states)})


def candidate_actions(state):
    """Original executed action, two raw-logit alternatives, then STOP."""
    baseline = state['executed_action']
    if state['forced_terminal']:
        if baseline is not None:
            raise ValueError('forced terminal must execute STOP')
        return [None]
    ids, logits = state['candidate_vpids'], state['logits']
    legal = [i for i in range(1, len(ids)) if state['valid_mask'][i]
             and not state['visited_mask'][i] and logits[i] is not None
             and math.isfinite(logits[i])]
    if baseline is not None and baseline not in [ids[i] for i in legal]:
        raise ValueError('reference executed an illegal action')
    other = sorted((i for i in legal if ids[i] != baseline), key=lambda i: (-logits[i], ids[i]))[:2]
    candidates = [baseline] + [ids[i] for i in other] + [None]
    return list(dict.fromkeys(candidates))


def assert_prefix_state(actual, expected, *, atol=1e-5, rtol=1e-5):
    """Fail closed on replay divergence before allowing a forced action."""
    exact = ('instr_id', 'scan_id', 'step', 'viewpoint', 'view_index',
             'candidate_vpids', 'valid_mask', 'visited_mask', 'no_vp_left',
             'forced_terminal', 'prefix_path', 'raw_argmax_action')
    for key in exact:
        if actual[key] != expected[key]:
            raise ValueError(f'prefix replay differs: {key} at step {actual["step"]}')
    for key in ('heading', 'elevation'):
        if not math.isclose(actual[key], expected[key], abs_tol=1e-7, rel_tol=1e-7):
            raise ValueError(f'prefix replay differs: {key}')
    maximum = 0.0
    if len(actual['logits']) != len(expected['logits']):
        raise ValueError('prefix replay differs: logit length')
    for a, b in zip(actual['logits'], expected['logits']):
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError('prefix replay differs: finite logit mask')
        else:
            maximum = max(maximum, abs(a-b))
            if not math.isclose(a, b, abs_tol=atol, rel_tol=rtol):
                raise ValueError('prefix replay differs: original logits')
    if not math.isclose(actual['original_stop_probability'], expected['original_stop_probability'],
                        abs_tol=atol, rel_tol=rtol):
        raise ValueError('prefix replay differs: original STOP probability')
    return maximum


class ContinuationHook:
    """Observe raw DUET scores, replay a prefix, and change at most one new action."""
    def __init__(self, agent, *, condition='natural', seed=0, reference=None,
                 target_step=None, target_action=None):
        if condition not in ('natural', 'perturb_step2'):
            raise ValueError('unknown probe condition')
        if agent.args.max_action_len != MAX_ACTION_LEN or agent.args.fusion != 'dynamic':
            raise ValueError('probe requires dynamic DUET with the original 15 decisions')
        if getattr(agent, 'feedback', None) != 'argmax':
            raise ValueError('probe requires greedy frozen DUET')
        if (reference is None) != (target_step is None):
            raise ValueError('a branch needs both reference and target step')
        if reference is not None:
            if reference.get('condition') != condition or reference.get('seed') != seed:
                raise ValueError('branch reference condition/seed differs')
            if target_step not in select_state_steps(reference['states'], condition):
                raise ValueError('branch target outside frozen state selection')
            if target_action not in candidate_actions(reference['states'][target_step]):
                raise ValueError('branch action outside frozen candidate selection')
        self.agent, self.condition, self.seed = agent, condition, seed
        self.reference, self.target_step, self.target_action = reference, target_step, target_action
        self.original_move = agent.make_equiv_action
        self.states, self.prefix_checks = [], []
        self.target_reached = False
        self.perturbation = copy.deepcopy(reference['perturbation']) if reference is not None else {
            'scheduled_step': 2, 'applied': False, 'reason': 'step_not_reached'}
        self.terminal_stop_scores = None

    def __call__(self, nav_inputs, nav_outs, observations, ended, step, trajectories):
        if len(observations) != 1 or bool(ended[0]) or step != len(self.states):
            raise ValueError('probe needs one active, contiguous trajectory')
        ob = observations[0]
        logits = nav_outs['fused_logits'][0]
        ids = list(nav_inputs['gmap_vpids'][0])
        if len(ids) != len(logits) or not ids or ids[0] is not None:
            raise ValueError('unexpected global action inventory')
        finite = torch.isfinite(logits)
        if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()) or not bool(finite[0]):
            raise ValueError('invalid original DUET action logits')
        no_left = bool(nav_inputs['no_vp_left'][0])
        forced_terminal = no_left or step == MAX_ACTION_LEN-1
        state = {'instr_id': str(ob['instr_id']), 'scan_id': str(ob['scan']), 'step': int(step),
                 'viewpoint': str(ob['viewpoint']), 'heading': float(ob['heading']),
                 'elevation': float(ob['elevation']), 'view_index': int(ob['viewIndex']),
                 'candidate_vpids': ids,
                 'valid_mask': nav_inputs['gmap_masks'][0].bool().cpu().tolist(),
                 'visited_mask': nav_inputs['gmap_visited_masks'][0].bool().cpu().tolist(),
                 'logits': [float(v) if bool(finite[i]) else None for i, v in enumerate(logits)],
                 'no_vp_left': no_left, 'forced_terminal': forced_terminal,
                 'raw_argmax_action': ids[int(logits.argmax())],
                 'original_stop_probability': float(torch.softmax(logits, 0)[0]),
                 'prefix_path': copy.deepcopy(trajectories[0]['path'])}
        selected = None if forced_terminal else state['raw_argmax_action']
        replace = False
        if self.reference is not None and step <= self.target_step:
            expected = self.reference['states'][step]
            error = assert_prefix_state(state, expected)
            self.prefix_checks.append({'step': step, 'max_absolute_logit_error': error})
            selected = expected['executed_action'] if step < self.target_step else self.target_action
            replace = True
            if step == self.target_step:
                self.target_reached = True
        elif self.reference is None and self.condition == 'perturb_step2' and step == 2:
            alternatives = [i for i in range(1, len(ids)) if bool(finite[i])
                and state['valid_mask'][i] and not state['visited_mask'][i]
                and ids[i] != state['raw_argmax_action']]
            alternatives.sort(key=lambda i: ids[i])
            if alternatives and not forced_terminal:
                pick = int(object_sha256([state['instr_id'], self.seed, 'perturb_step2',
                                         [ids[i] for i in alternatives]]), 16) % len(alternatives)
                selected, replace = ids[alternatives[pick]], True
                self.perturbation.update(applied=True, reason='applied',
                    original_action=state['raw_argmax_action'], forced_action=selected)
            else:
                self.perturbation['reason'] = 'no_executable_alternative'
        if forced_terminal and selected is not None:
            raise ValueError('moving action cannot execute at forced terminal')
        state['expected_executed_action'] = selected
        self.states.append(state)
        if not replace or selected == state['raw_argmax_action']:
            return nav_outs
        idx = ids.index(selected)
        if not state['valid_mask'][idx] or state['visited_mask'][idx] or not bool(finite[idx]):
            raise ValueError('cannot force an illegal action')
        changed = dict(nav_outs)
        changed['fused_logits'] = nav_outs['fused_logits'].clone()
        changed['fused_logits'][0].fill_(-torch.inf)
        changed['fused_logits'][0, idx] = 0
        return changed

    def make_equiv_action(self, actions, graphs, observations, trajectories):
        state = self.states[-1]
        if len(actions) != 1 or actions[0] != state['expected_executed_action']:
            raise ValueError('upstream did not execute the requested action')
        result = self.original_move(actions, graphs, observations, trajectories)
        # Upstream records softmax after our hook. Restore raw evidence BEFORE its
        # historical STOP fallback, including forced STOP and forced moving actions.
        graph = graphs[0]
        graph.node_stop_scores[state['viewpoint']]['stop'] = state['original_stop_probability']
        state['executed_action'] = actions[0]
        if actions[0] is None:
            self.terminal_stop_scores = {str(k): float(v['stop']) for k, v in graph.node_stop_scores.items()}
        return result

    def finish(self, trajectories):
        if (len(trajectories) != 1 or not self.states or len(self.states) > MAX_ACTION_LEN
                or self.states[-1].get('executed_action', 'missing') is not None
                or any('executed_action' not in s for s in self.states)):
            raise ValueError('incomplete original-budget rollout')
        if self.reference is not None and not self.target_reached:
            raise ValueError('branch target was never reached')
        trajectory = trajectories[0]
        if str(trajectory['instr_id']) != self.states[0]['instr_id']:
            raise ValueError('rollout instruction identity differs')
        result = {'instr_id': self.states[0]['instr_id'], 'scan_id': self.states[0]['scan_id'],
                  'condition': self.condition, 'seed': self.seed, 'path': copy.deepcopy(trajectory['path']),
                  'states': self.states, 'decision_count': len(self.states),
                  'perturbation': self.perturbation, 'prefix_checks': self.prefix_checks,
                  'terminal_stop_scores': self.terminal_stop_scores,
                  'target_step': self.target_step, 'target_action': self.target_action}
        if self.reference is not None and self.target_action == self.reference['states'][self.target_step]['executed_action']:
            assert_anchor(result, self.reference, metrics=False)
            result['anchor_trajectory_and_all_states_equal'] = True
        return result


def assert_anchor(branch, reference, *, metrics=True):
    if branch['path'] != reference['path'] or branch['decision_count'] != reference['decision_count']:
        raise ValueError('original-action anchor full trajectory differs')
    if 'states' in branch:
        for a, b in zip(branch['states'], reference['states']):
            assert_prefix_state(a, b)
            if a['executed_action'] != b['executed_action']:
                raise ValueError('original-action anchor action differs')
    if metrics:
        a, b = branch['metrics'], reference['metrics']
        if set(a) != set(b) or any(not math.isclose(a[k], b[k], rel_tol=1e-12, abs_tol=1e-12) for k in a):
            raise ValueError('original-action anchor full-route metrics differ')


class ProbeTaskStore:
    """Cloud journal acknowledges only immutable complete, read-back-verified tasks."""
    def __init__(self, root, backup, identity, *, check_backup=None):
        self.root, self.backup, self.identity = Path(root), Path(backup), identity
        self.check_backup = check_backup or (lambda: None)
        self.check_backup()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backup.mkdir(parents=True, exist_ok=True)
        self.identity_sha = object_sha256(identity)
        cloud, local = self.backup/'manifest.json', self.root/'manifest.json'
        if cloud.exists():
            self.manifest = json.loads(cloud.read_text())
            if (self.manifest.get('schema') != SCHEMA or self.manifest.get('identity') != identity
                    or self.manifest.get('identity_sha256') != self.identity_sha):
                raise ValueError('probe resume source/assets/config/selection identity differs')
            atomic_bytes(local, cloud.read_bytes())
            for key in self.manifest['tasks']:
                self.get_by_key(key)
        else:
            if local.exists():
                raise ValueError('local journal has no verified cloud counterpart')
            self.manifest = {'schema': SCHEMA, 'identity': identity, 'identity_sha256': self.identity_sha,
                             'tasks': {}, 'complete': False}
            self._journal()

    def _journal(self):
        self.check_backup()
        raw = (json.dumps(self.manifest, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        atomic_bytes(self.backup/'manifest.json', raw)
        if hashlib.sha256((self.backup/'manifest.json').read_bytes()).hexdigest() != hashlib.sha256(raw).hexdigest():
            raise ValueError('cloud probe journal read-back failed')
        atomic_bytes(self.root/'manifest.json', raw)

    def get_by_key(self, key):
        entry = self.manifest['tasks'].get(key)
        if entry is None:
            return None
        self.check_backup()
        name = entry['file']
        if Path(name).name != name:
            raise ValueError('unsafe task filename')
        cloud, local = self.backup/name, self.root/name
        if not cloud.is_file() or file_sha256(cloud) != entry['sha256']:
            raise ValueError('acknowledged cloud task missing or corrupt')
        if local.exists() and file_sha256(local) != entry['sha256']:
            raise ValueError('local task differs from acknowledged checksum')
        if not local.exists():
            atomic_bytes(local, cloud.read_bytes())
        record = json.loads(local.read_text())
        if (record.get('status') != 'complete' or record.get('identity_sha256') != self.identity_sha
                or object_sha256(record.get('task')) != key
                or record.get('payload_sha256') != object_sha256({k: v for k, v in record.items() if k != 'payload_sha256'})):
            raise ValueError('failed or mismatched probe task cannot be reused')
        return record['result']

    def get(self, task):
        key = object_sha256(task)
        # Recover a fully atomic cloud object if eviction occurred between its
        # verified write and journal commit. Partial/failed objects are rejected.
        if key not in self.manifest['tasks']:
            self.check_backup()
            name = f'task-{key}.json'
            cloud = self.backup/name
            if cloud.exists():
                record = json.loads(cloud.read_text())
                if (record.get('status') != 'complete' or record.get('identity_sha256') != self.identity_sha
                        or record.get('task') != task or not isinstance(record.get('result'), dict)
                        or record.get('payload_sha256') != object_sha256({k: v for k, v in record.items() if k != 'payload_sha256'})):
                    raise ValueError('unacknowledged cloud object is not a complete matching task')
                self.manifest['tasks'][key] = {'file': name, 'sha256': file_sha256(cloud)}
                # Local copies may be interrupted/obsolete; the atomic cloud
                # object is authoritative only after the validation above.
                atomic_bytes(self.root/name, cloud.read_bytes())
                self._journal()
        return self.get_by_key(key)

    def put(self, task, result):
        key = object_sha256(task)
        existing = self.get(task)
        if existing is not None:
            if existing != result:
                raise ValueError('completed task changed')
            return
        if self.manifest['complete']:
            raise ValueError('cannot append to completed probe')
        record = {'status': 'complete', 'identity_sha256': self.identity_sha, 'task': task, 'result': result}
        record['payload_sha256'] = object_sha256(record)
        raw = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        name = f'task-{key}.json'
        atomic_bytes(self.root/name, raw)
        self.check_backup()
        verified_copy(self.root/name, self.backup/name)
        self.manifest['tasks'][key] = {'file': name, 'sha256': file_sha256(self.root/name)}
        self._journal()

    def fail(self, task, error):
        """Diagnostic only: failures never enter the completed-task journal."""
        record = {'status': 'failed', 'identity_sha256': self.identity_sha,
                  'task': task, 'error': str(error), 'time_ns': time.time_ns()}
        raw = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        name = f'failure-{object_sha256(task)}-{record["time_ns"]}.json'
        atomic_bytes(self.root/name, raw)
        self.check_backup()
        verified_copy(self.root/name, self.backup/name)

    def close(self, expected_tasks):
        expected = {object_sha256(x) for x in expected_tasks}
        if expected != set(self.manifest['tasks']):
            raise ValueError('probe task coverage incomplete or unexpected')
        for key in expected:
            self.get_by_key(key)
        self.manifest['complete'] = True
        self._journal()


def opportunity_summary(references, branches):
    """Descriptive label-only bound; the unit is original instruction per condition."""
    out = {'warning': 'Hindsight training-scene opportunity only; not learned method performance or unseen prediction.',
           'conditions': {}}
    for result in references + branches:
        m = result['metrics']
        if m['success'] not in (0, 1) or not math.isfinite(m['spl']) or not 0 <= m['spl'] <= 1:
            raise ValueError('success and SPL outcomes must be finite fractions')
    for condition in ('natural', 'perturb_step2'):
        refs = [r for r in references if r['condition'] == condition]
        if not refs:
            continue
        if len({r['instr_id'] for r in refs}) != len(refs):
            raise ValueError('duplicate original instruction within condition')
        selected, rescue_ids, harm_ids, spl_ids = [], [], [], []
        rescue_branches = harm_branches = branch_count = changed_count = 0
        for ref in refs:
            options = [b for b in branches if b['instr_id'] == ref['instr_id'] and b['condition'] == condition]
            baseline = ref['metrics']; outcomes = [baseline] + [b['metrics'] for b in options]
            eligible = [m for m in outcomes if m['spl'] >= baseline['spl']-1e-12]
            selected.append(max(eligible, key=lambda m: (m['success'], m['spl'])))
            rescues = sum(m['success'] > baseline['success'] for m in outcomes[1:])
            harms = sum(m['success'] < baseline['success'] for m in outcomes[1:])
            if rescues: rescue_ids.append(ref['instr_id'])
            if harms: harm_ids.append(ref['instr_id'])
            if any(m['spl'] > baseline['spl']+1e-12 for m in outcomes[1:]): spl_ids.append(ref['instr_id'])
            rescue_branches += rescues; harm_branches += harms; branch_count += len(options)
            changed_count += sum(not b['is_anchor'] for b in options)
        n = len(refs)
        out['conditions'][condition] = {'instructions': n, 'scenes': len({r['scan_id'] for r in refs}),
            'baseline_sr_percent': 100*sum(r['metrics']['success'] for r in refs)/n,
            'baseline_spl_percent': 100*sum(r['metrics']['spl'] for r in refs)/n,
            'label_only_oracle_sr_percent': 100*sum(m['success'] for m in selected)/n,
            'label_only_oracle_spl_percent': 100*sum(m['spl'] for m in selected)/n,
            'oracle_constraint': 'per-instruction SPL >= own reference SPL; one forced action then full DUET',
            'rescuable_failed_instructions': len(rescue_ids), 'rescuable_instr_ids': rescue_ids,
            'harmful_branch_instructions': len(harm_ids), 'harmful_instr_ids': harm_ids,
            'spl_improvable_instructions': len(spl_ids), 'spl_improvable_instr_ids': spl_ids,
            'branches_including_anchors': branch_count, 'alternative_branches': changed_count,
            'rescue_branches_nonindependent': rescue_branches, 'harm_branches_nonindependent': harm_branches,
            'actual_perturbations': sum(r['perturbation']['applied'] for r in refs),
            'actual_perturbation_rate': sum(r['perturbation']['applied'] for r in refs)/n,
            'instructions_without_selected_states': sum(not select_state_steps(r['states'], condition) for r in refs)}
    out['unique_original_instructions_union_conditions'] = len({r['instr_id'] for r in references})
    return out
