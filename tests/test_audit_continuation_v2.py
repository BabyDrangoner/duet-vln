"""Independent cache auditing rejects semantic errors even with fresh SHA pins."""
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit_continuation_v2', ROOT/'scripts/audit_continuation_v2.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture_bundle(condition='natural', stop_at=3):
    """Explicit complete Euclidean graph: all candidates visible at every node."""
    count = max(7, stop_at+4)
    nodes = [f'v{i:02}' for i in range(count)]
    graph = {a: {b: float(4*abs(i-j)) for j, b in enumerate(nodes) if a != b} for i, a in enumerate(nodes)}
    distance = audit.Distance(graph)
    gt = nodes[:4]
    truth = {'1_0': ('scene', gt)}
    path, states, perturbations, stop_scores = [[nodes[0]]], [], [], {}
    visited = []
    features_by_step = {}
    for t in range(stop_at+1):
        current = path[-1][-1]; visited.append(current)
        ids = [None] + visited + [n for n in nodes if n not in visited]
        raw = None if t == stop_at else next(n for n in nodes if n not in visited)
        forced = t == 14
        logits = [4. if raw is None else -2.] + [None if n in visited else (5. if n == raw else 0.) for n in ids[1:]]
        finite = [x for x in logits if x is not None]
        prob = math.exp(logits[0])/math.fsum(math.exp(x) for x in finite)
        selected = None if forced else raw
        if t in audit.SCHEDULES[condition] and not forced:
            alternatives = sorted(n for n in nodes if n not in visited and n != raw)
            if alternatives:
                key = [audit.SCHEMA, '1_0', 0, condition, t, alternatives]
                selected = alternatives[int(audit.objsha(key), 16) % len(alternatives)]
                perturbations.append({'step': t, 'action': selected})
        state = {'instr_id': '1_0', 'scan_id': 'scene', 'step': t, 'viewpoint': current,
            'heading': 0., 'elevation': 0., 'view_index': 12,
            'candidate_vpids': ids, 'valid_mask': [True]*len(ids),
            'visited_mask': [False]+[n in visited for n in ids[1:]], 'logits': logits,
            'no_vp_left': False, 'forced_terminal': forced,
            'raw_argmax_action': raw, 'original_stop_probability': prob,
            'expected_executed_action': selected, 'executed_action': selected,
            'prefix_path': copy.deepcopy(path)}
        states.append(state); stop_scores[current] = prob
        feature_rows = {}
        for index, node in enumerate(ids):
            if logits[index] is None: continue
            token = torch.zeros(1549, dtype=torch.float16)
            token[:768] = 100*t + index
            token[768:1536] = 1
            token[1536] = token[1538] = -1
            token[1537] = max(-30, logits[index] - math.log(math.fsum(math.exp(x) for x in finite)))
            token[1539] = 1; token[1540] = float(node is None)
            token[1541] = math.log1p(len(finite))
            feature_rows[node] = token
        features_by_step[t] = feature_rows
        if selected is not None: path.append([selected])
    assert states[-1]['executed_action'] is None, 'fixture must finish after all perturbations'
    fallback = max(stop_scores, key=stop_scores.get)
    if path[-1][-1] != fallback: path.append([fallback])
    reference = {'schema': audit.SCHEMA, 'instr_id': '1_0', 'scan_id': 'scene', 'seed': 0,
        'condition': condition, 'path': path, 'states': states, 'decision_count': len(states),
        'target_step': None, 'target_action': None, 'prefix_checks': [],
        'terminal_stop_scores': stop_scores, 'perturbations': perturbations,
        'perturbation': {'applied': False, 'reason': 'unused legacy field'}, 'intervention': None,
        'metrics': audit.independent_metrics(path, gt, graph, distance)}
    branches, records = [], []
    for step in audit.selected_steps(reference):
        state = states[step]; actions = audit.candidates(state); outcomes = []
        for action in actions:
            branch = {k: copy.deepcopy(v) for k, v in reference.items() if k != 'states'}
            branch.update(target_step=step, target_action=action, perturbations=[],
                prefix_checks=[{'step': t, 'max_absolute_logit_error': 0.} for t in range(step+1)],
                is_anchor=action == state['executed_action'])
            if branch['is_anchor']:
                trace = [{k: s[k] for k in ('step', 'viewpoint', 'executed_action')} for s in states]
            else:
                branch['path'] = copy.deepcopy(path[:step+1])
                trace = [{k: s[k] for k in ('step', 'viewpoint', 'executed_action')} for s in states[:step]]
                trace.append({'step': step, 'viewpoint': state['viewpoint'], 'executed_action': action})
                scores = {s['viewpoint']: s['original_stop_probability'] for s in states[:step+1]}
                if action is not None:
                    branch['path'].append([action])
                    trace.append({'step': step+1, 'viewpoint': action, 'executed_action': None})
                    scores[action] = .999
                fallback = max(scores, key=scores.get)
                if branch['path'][-1][-1] != fallback: branch['path'].append([fallback])
                branch['terminal_stop_scores'] = scores
                branch['decision_count'] = len(trace)
            branch['action_trace'] = trace
            branch['metrics'] = audit.independent_metrics(branch['path'], gt, graph, distance)
            outcomes.append([branch['metrics']['success'], branch['metrics']['spl']]); branches.append(branch)
        history = []
        for t in range(step+1):
            history.append(features_by_step[t][None])
            if t < step: history.append(features_by_step[t][states[t]['executed_action']])
        if state['viewpoint'] == gt[-1]: teacher = None
        else:
            legal = [v for i, v in enumerate(state['candidate_vpids']) if i and not state['visited_mask'][i]]
            teacher = min(legal, key=lambda v: distance(state['viewpoint'], v)+distance(v, gt[-1]))
        prefix = audit.flat(state['prefix_path'])
        length = sum(graph[a][b] for a, b in zip(prefix, prefix[1:]))
        records.append({'instr_id': '1_0', 'scan_id': 'scene', 'condition': condition,
            'step': step, 'candidate_actions': actions,
            'features': torch.stack([features_by_step[step][a] for a in actions]),
            'history_features': torch.stack(history),
            'progress': torch.tensor([step/14, (15-step)/15, length/100, len(actions)/4], dtype=torch.float16),
            'teacher_target': actions.index(teacher) if teacher in actions else -1,
            'utilities': torch.tensor(outcomes, dtype=torch.float32)})
    bundle = {'schema': audit.SCHEMA, 'split': 'train_fit', 'reference': reference,
              'branches': branches, 'records': records}
    return bundle, truth, {'scene': graph}, {'scene': distance}


def check(bundle, truth, graphs, distances):
    return audit.audit_bundle(bundle, truth, graphs, distances)


@pytest.mark.parametrize('condition,stop_at', [('natural', 3), ('perturb_step2', 8), ('early_two', 8), ('late_three', 12)])
def test_complete_causal_bundles_pass_all_four_schedules(condition, stop_at):
    args = fixture_bundle(condition, stop_at)
    result = check(*args)
    assert result['anchors'] == len(args[0]['records'])
    assert result['branches'] > result['anchors']
    assert result['perturbations'] == len(audit.SCHEDULES[condition])


def test_forced_terminal_at_original_fifteen_decisions_is_never_a_training_state():
    args = fixture_bundle('late_three', 14)
    assert args[0]['reference']['decision_count'] == 15
    assert [r['step'] for r in args[0]['records']] == [7, 10]
    check(*args)


def test_analytic_metrics_charge_return_and_duplicate_segment_boundaries():
    graph = {'a': {'b': 4.}, 'b': {'a': 4., 'c': 4.}, 'c': {'b': 4.}}
    distance = audit.Distance(graph)
    # Reaching the goal is insufficient if historical STOP returns to a.
    metrics = audit.independent_metrics([['a'], ['b', 'c'], ['b', 'a']], ['a', 'b', 'c'], graph, distance)
    assert metrics['trajectory_lengths'] == 16.
    assert metrics['trajectory_steps'] == 4
    assert metrics['nav_error'] == 8.
    assert metrics['success'] == metrics['spl'] == metrics['SDTW'] == 0.
    assert metrics['oracle_success'] == 1.
    assert metrics['DTW'] == 12.
    assert metrics['nDTW'] == pytest.approx(math.exp(-12/9))
    assert metrics['CLS'] == .5
    repeated = audit.independent_metrics([['a'], ['a', 'b'], ['b', 'c']], ['a', 'b', 'c'], graph, distance)
    assert repeated['trajectory_lengths'] == 8.
    assert repeated['trajectory_steps'] == 4
    assert repeated['nDTW'] == repeated['CLS'] == repeated['spl'] == 1.
    with pytest.raises(ValueError, match='nonexecutable edge'):
        audit.independent_metrics([['a'], ['c']], ['a', 'b', 'c'], graph, distance)


@pytest.mark.parametrize('field', audit.METRICS)
def test_recomputes_every_metric_instead_of_trusting_collector(field):
    args = fixture_bundle(); args[0]['branches'][0]['metrics'][field] += .02
    with pytest.raises(ValueError, match=field): check(*args)


@pytest.mark.parametrize('defect,match', [
    ('utility', 'utility SR'), ('spl_utility', 'utility SPL'),
    ('half_utilities', 'utilities'), ('fp32_features', 'features'),
    ('nan', 'finiteness'), ('extra_history', 'causal history'),
    ('fused_logprob', 'fused log'), ('action_count', 'action count'),
    ('current_stop', 'STOP identity'), ('reordered_history', 'fused log'),
    ('rewritten_embedding', 'rewrites'), ('past_action_embedding', 'past executed'),
    ('current_stop_embedding', 'current STOP'), ('progress', 'progress differs'),
    ('candidate_order', 'candidates/order'), ('teacher', 'teacher'),
    ('removed_record', 'coverage/order'), ('removed_branch', 'branch coverage'),
    ('prefix_route', 'branch changed the executed prefix'),
    ('anchor_route', 'anchor full trajectory'), ('prefix_action', 'prefix/target action'),
    ('prefix_logits', 'prefix-logit'), ('post_perturbation', 'new perturbation'),
    ('raw_stop', 'reference raw STOP'), ('terminal_table', 'per-decision evidence'),
    ('mask', 'visited mask'), ('argmax', 'raw argmax'), ('budget', 'decision budget'),
    ('unseen_candidate', 'candidate inventory'), ('before_eligible', 'branch coverage')])
def test_rejects_semantic_corruptions(defect, match):
    args = fixture_bundle(); bundle = args[0]; ref = bundle['reference']
    record = bundle['records'][0]; later = bundle['records'][1]
    if defect == 'utility': record['utilities'][0, 0] = 1-record['utilities'][0, 0]
    elif defect == 'spl_utility': record['utilities'][0, 1] += .1
    elif defect == 'half_utilities': record['utilities'] = record['utilities'].half()
    elif defect == 'fp32_features': record['features'] = record['features'].float()
    elif defect == 'nan': record['features'][0, 0] = torch.nan
    elif defect == 'extra_history': record['history_features'] = record['history_features'].repeat(2, 1)
    elif defect == 'fused_logprob': record['features'][0, 1537] -= .1
    elif defect == 'action_count': record['features'][0, 1541] += .1
    elif defect == 'current_stop': record['features'][-1, 1540] = 0
    elif defect == 'reordered_history': later['history_features'][[0, 1]] = later['history_features'][[1, 0]]
    elif defect == 'rewritten_embedding': later['history_features'][0, 0] += 1
    elif defect == 'past_action_embedding': later['history_features'][1, 0] += 1
    elif defect == 'current_stop_embedding': later['history_features'][-1, 0] += 1
    elif defect == 'progress': later['progress'][2] += .25
    elif defect == 'candidate_order': record['candidate_actions'][1:3] = reversed(record['candidate_actions'][1:3])
    elif defect == 'teacher': record['teacher_target'] = record['candidate_actions'].index(None)
    elif defect == 'removed_record': bundle['records'].pop()
    elif defect == 'removed_branch': bundle['branches'].pop()
    elif defect == 'prefix_route':
        branch = next(b for b in bundle['branches'] if b['target_step'] == 3)
        # Insert a physically legal detour but preserve all decision endpoints;
        # refresh metrics so only the independent common-prefix check catches it.
        branch['path'][1] = ['v04', 'v00', 'v01']
        branch['metrics'] = audit.independent_metrics(branch['path'], args[1]['1_0'][1], args[2]['scene'], args[3]['scene'])
    elif defect == 'anchor_route':
        branch = bundle['branches'][0]; branch['path'][1] = ['v04', 'v00', 'v01']
        branch['metrics'] = audit.independent_metrics(branch['path'], args[1]['1_0'][1], args[2]['scene'], args[3]['scene'])
    elif defect == 'prefix_action':
        # Keep rollout endpoints valid, but override the designated target.
        branch = bundle['branches'][1]; branch['target_action'] = record['candidate_actions'][0]
        # Exercise branch checker directly, after explicit independent metrics.
        with pytest.raises(ValueError, match=match):
            audit.audit_branch(branch, ref, ref['metrics'], branch['metrics'], {})
        return
    elif defect == 'prefix_logits': bundle['branches'][0]['prefix_checks'][0]['max_absolute_logit_error'] = .1
    elif defect == 'post_perturbation': bundle['branches'][0]['perturbations'] = [{'step': 1, 'action': 'v02'}]
    elif defect == 'raw_stop': ref['states'][0]['original_stop_probability'] = .8
    elif defect == 'terminal_table':
        ref['terminal_stop_scores']['v00'] += .001
    elif defect == 'mask': ref['states'][0]['visited_mask'][1] = False
    elif defect == 'argmax': ref['states'][0]['raw_argmax_action'] = None
    elif defect == 'budget': ref['decision_count'] = 16
    elif defect == 'unseen_candidate': ref['states'][0]['candidate_vpids'][2] = 'unseen'
    elif defect == 'before_eligible': bundle['branches'][0]['target_step'] = 1
    with pytest.raises(ValueError, match=match): check(*args)


def test_multistep_perturbation_hash_and_array_are_independently_checked():
    args = fixture_bundle('early_two', 8)
    args[0]['reference']['perturbations'].pop()
    with pytest.raises(ValueError, match='deterministic schedule'): check(*args)
    args = fixture_bundle('early_two', 8)
    args[0]['reference']['seed'] = 123
    with pytest.raises(ValueError, match='fixed perturbation'): check(*args)


def save_pair(tmp_path, bundle):
    local, backup = tmp_path/'local', tmp_path/'backup'
    local.mkdir(); backup.mkdir()
    raw = io.BytesIO(); torch.save(bundle, raw); raw = raw.getvalue()
    digest = audit.hashlib.sha256(raw).hexdigest()
    task = {'instr_id': '1_0', 'condition': 'natural'}
    name = f'bundle-{audit.objsha(task)}-{digest[:16]}.pt'
    for directory in (local, backup): (directory/name).write_bytes(raw)
    return local, backup, {'task': task, 'sha256': digest, 'file': name, 'bytes': len(raw), 'records': len(bundle['records'])}


def test_valid_sha_on_semantically_corrupted_labels_does_not_make_them_valid(tmp_path):
    args = fixture_bundle(); args[0]['records'][0]['utilities'][0, 0] = 0
    local, backup, pointer = save_pair(tmp_path, args[0])
    decoded = audit.read_bundle(local, backup, pointer)
    with pytest.raises(ValueError, match='utility SR'):
        check(decoded, *args[1:])


def test_corrupt_or_missing_backup_is_rejected_without_recovery_writes(tmp_path):
    local, backup, pointer = save_pair(tmp_path, fixture_bundle()[0])
    original = (local/pointer['file']).read_bytes()
    (backup/pointer['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'): audit.read_bundle(local, backup, pointer)
    assert (local/pointer['file']).read_bytes() == original
    assert (backup/pointer['file']).read_bytes() == b'corrupt'
    (backup/pointer['file']).unlink()
    with pytest.raises(ValueError, match='missing'): audit.read_bundle(local, backup, pointer)
    assert not (backup/pointer['file']).exists()


def test_tensor_pointer_cannot_escape_immutable_directories(tmp_path):
    local, backup, pointer = save_pair(tmp_path, fixture_bundle()[0])
    pointer['file'] = '../escape.pt'
    with pytest.raises(ValueError, match='filename/task'): audit.read_bundle(local, backup, pointer)


def test_selection_reconstructs_disjoint_scenes_and_deterministic_shards(tmp_path):
    rows = [{'path_id': s*10+i, 'scan': f'scene{s}', 'path': ['a', 'b'], 'instructions': ['one', 'two']}
            for s in range(5) for i in range(2)]
    path = tmp_path/'train.json'; path.write_text(json.dumps(rows))
    provenance = {'partition_seed': 7, 'dev_fraction': .2, 'seed': 0}
    scans = sorted({r['scan'] for r in rows}, key=lambda s: audit.objsha([7, s]))
    pool = [f'{r["path_id"]}_{i}' for r in rows if r['scan'] != scans[0] for i in range(2)]
    lookup = {f'{r["path_id"]}_{i}': r['scan'] for r in rows for i in range(2)}
    scene_order = sorted(scans[1:], key=lambda s: audit.objsha([0, s]))
    per_scene = {s: sorted([i for i in pool if lookup[i] == s], key=lambda i: audit.objsha([0, i])) for s in scene_order}
    full = sorted(per_scene[s][d] for d in range(2) for s in scene_order)
    selected = full[1::3]
    selection = {'split': 'train_fit', 'selection': 'scene_stratified', 'seed': 0,
        'instr_ids': selected, 'count': len(selected), 'scan_ids': sorted({lookup[i] for i in selected}),
        'conditions': list(audit.SCHEDULES), 'full_instr_ids': full, 'shard_index': 1, 'shard_count': 3}
    assert set(audit.select_truth(path, selection, provenance, 8)) == set(selected)
    selection['instr_ids'][0] = next(i for i in pool if i not in selected)
    with pytest.raises(ValueError, match='selection differs'): audit.select_truth(path, selection, provenance, 8)


def test_source_identity_accepts_one_historical_snapshot_but_rejects_mixed_versions(tmp_path, monkeypatch):
    files = ('a.py', 'b.py'); monkeypatch.setattr(audit, 'COLLECTOR_FILES', files)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    def commit():
        subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-qm', 'fixture'], check=True)
    for name in files: (tmp_path/name).write_text('old\n')
    commit(); old = {name: audit.sha(tmp_path/name) for name in files}
    for name in files: (tmp_path/name).write_text('new\n')
    commit()
    provenance = {'source_files': old, 'source_sha256': audit.objsha(old)}
    assert audit.verify_source(tmp_path, provenance)['status'] == 'complete_historical_git_snapshot_matches'
    provenance['source_files']['b.py'] = audit.sha(tmp_path/'b.py')
    provenance['source_sha256'] = audit.objsha(provenance['source_files'])
    with pytest.raises(ValueError, match='complete Git snapshot'): audit.verify_source(tmp_path, provenance)


def journal_pair(tmp_path):
    local, backup = tmp_path/'local', tmp_path/'backup'
    for directory in (local, backup): (directory/'tasks').mkdir(parents=True)
    task = {'instr_id': '1_0', 'condition': 'natural'}
    pointer = {'file': 'bundle.pt', 'sha256': 'a'*64, 'bytes': 123, 'records': 1}
    manifest = {'selection': {'split': 'train_fit'}, 'provenance': {}, 'bundles': [dict(pointer, task=task)]}
    identity = {'schema': audit.SCHEMA, 'selection': manifest['selection'], 'provenance': manifest['provenance']}
    record = {'status': 'complete', 'identity_sha256': audit.objsha(identity), 'task': task, 'result': pointer}
    record['payload_sha256'] = audit.objsha(record)
    key = audit.objsha(task); name = f'task-{key}.json'
    raw = json.dumps(record).encode()
    journal = {'schema': 'e3_continuation_probe_tasks_v1', 'complete': True, 'identity': identity,
        'identity_sha256': audit.objsha(identity), 'tasks': {key: {'file': name, 'sha256': audit.hashlib.sha256(raw).hexdigest()}}}
    for directory in (local, backup):
        (directory/'tasks'/name).write_bytes(raw)
        (directory/'tasks/manifest.json').write_text(json.dumps(journal))
    return local, backup, manifest


def test_task_journal_has_to_bind_manifest_pointer_even_if_both_copies_agree(tmp_path):
    local, backup, manifest = journal_pair(tmp_path)
    assert audit.verify_journal(local, backup, manifest)['both_copies_verified']
    # Recompute both copies' entry digests and payload digest, but make them
    # disagree with the completed dataset manifest's immutable tensor pointer.
    for directory in (local, backup):
        path = directory/'tasks/manifest.json'; journal = json.loads(path.read_text())
        key, entry = next(iter(journal['tasks'].items()))
        record_path = directory/'tasks'/entry['file']; record = json.loads(record_path.read_text())
        record['result']['records'] += 1
        record['payload_sha256'] = audit.objsha({k: v for k, v in record.items() if k != 'payload_sha256'})
        record_path.write_text(json.dumps(record)); entry['sha256'] = audit.sha(record_path)
        path.write_text(json.dumps(journal))
    with pytest.raises(ValueError, match='task pointer differs'):
        audit.verify_journal(local, backup, manifest)


def test_missing_completed_tasks_cannot_be_hidden_by_shortening_dataset_manifest(tmp_path):
    local, backup, manifest = journal_pair(tmp_path)
    manifest['bundles'] = []
    with pytest.raises(ValueError, match='journal coverage'):
        audit.verify_journal(local, backup, manifest)


def test_geometry_label_agreement_does_not_allow_an_unobserved_transit_edge():
    args = fixture_bundle(); branch = args[0]['branches'][0]
    # Both edges are globally legal. v04->v01 was not observed at the start.
    branch['path'][1] = ['v04', 'v01']
    branch['metrics'] = audit.independent_metrics(branch['path'], args[1]['1_0'][1], args[2]['scene'], args[3]['scene'])
    with pytest.raises(ValueError, match='not yet observed'):
        audit.audit_rollout(branch, args[1], args[2], args[3], {})


def test_branch_stop_after_intervention_keeps_prefix_raw_probability():
    args = fixture_bundle(); ref = args[0]['reference']
    branch = next(b for b in args[0]['branches'] if b['target_step'] == 0 and b['target_action'] is None)
    branch['terminal_stop_scores'][ref['states'][0]['viewpoint']] = 1.
    with pytest.raises(ValueError, match='prefix raw STOP'):
        audit.audit_branch(branch, ref, ref['metrics'], branch['metrics'], {})
