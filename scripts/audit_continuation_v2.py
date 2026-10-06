#!/usr/bin/env python3
"""Read-only, independent geometry and tensor audit of completed E3 v2 caches.

No navigation environment, collector, or policy implementation is imported.
Only training annotations are accepted. CPU torch is used to decode safe tensor
bundles and inspect their precision, never to run the navigation model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import torch

# Also permits running a temporary copy under outputs with --root PROJECT.
ROOT = Path(__file__).resolve().parents[1]
if '--root' in sys.argv:
    ROOT = Path(sys.argv[sys.argv.index('--root') + 1]).expanduser().resolve()
sys.path.insert(0, str(ROOT/'scripts'))
from audit_continuation_probe import (CONNECTIVITY_SHA, TRAIN_SHA, Distance,
    candidates, close, flat, load_graphs, objsha, read, recompute, require, sha)

SCHEMA = 'e3_causal_continuation_v2'
AUDIT_SCHEMA = 'e3_causal_continuation_independent_audit_v1'
SCHEDULES = {'natural': (), 'perturb_step2': (2,), 'early_two': (1, 2), 'late_three': (2, 4, 6)}
METRICS = ('success', 'spl', 'nav_error', 'trajectory_lengths', 'oracle_success',
           'oracle_error', 'trajectory_steps', 'action_steps', 'DTW', 'nDTW', 'SDTW', 'CLS')
FEATURE_DIM = 1549
ASSET_PINS = {'train_annotation_sha256': TRAIN_SHA, 'connectivity_sha256': CONNECTIVITY_SHA,
    'base_checkpoint_sha256': 'c1ed3ed27e1acbffec1fa81fbdd06e16498398809a7f80f349043ba74c21ba94',
    'feature_sha256': 'ae0ff208349e6a12096fe47c7b045e42b93e9c047da50d19dd17e04ec8d31bbc'}
COLLECTOR_FILES = ('scripts/collect_continuation_v2.py', 'scripts/prepare_duet.py',
    'scripts/run_duet.py', 'src/vln_improve/continuation_probe.py',
    'src/vln_improve/continuation_v2.py', 'src/vln_improve/features.py',
    'src/vln_improve/intervention_runtime.py', 'src/vln_improve/pipeline.py',
    'src/vln_improve/protocol.py')
LIMITATIONS = [
    'Compact branch traces do not retain post-intervention raw logits. Future DUET argmax, raw STOP probabilities, and the absence of additional score overrides cannot be independently proved from these logs.',
    'Raw graph adjacency represents the standard complete panoramic R2R observation. The audit verifies route edges were discoverable from decision viewpoints; raw simulator observations were not retained.',
    'Embedding values cannot be rederived without rerunning DUET. History length, scalar/action identities, shared cached prefixes, and selected-state tokens are checked; arbitrary hidden target information in embeddings cannot be excluded by cache inspection alone.',
    'Checkpoint and feature asset hashes are checked against frozen provenance pins, but the large model and feature files are not rehashed. Training annotations and every connectivity file are rehashed.',
    'Rescuable counts describe supervised counterfactual opportunities, not learned navigation improvement.'
]


def selected_steps(reference):
    schedule = SCHEDULES[reference['condition']]
    first = max(schedule) + 1 if schedule else 0
    return [s['step'] for s in reference['states'] if s['step'] >= first
            and not s['forced_terminal']
            and (s['step'] in (first, first + 3) or s['raw_argmax_action'] is None)]


def independent_metrics(path, reference, graph, distance):
    """Charge every physical edge and historical return; rolling-row exact DTW."""
    value, transitions = recompute(path, reference, graph, distance)
    nodes = flat(path)
    previous = [0.] + [math.inf] * len(reference)
    for node in nodes:
        current = [math.inf]
        for j, target in enumerate(reference, 1):
            current.append(distance(node, target) + min(previous[j], current[-1], previous[j-1]))
        previous = current
    dtw = previous[-1]
    ndtw = math.exp(-dtw/(3*len(reference)))
    coverage = math.fsum(math.exp(-min(distance(goal, node) for node in nodes)/3)
                         for goal in reference)/len(reference)
    expected = coverage * math.fsum(distance(a, b) for a, b in zip(reference, reference[1:]))
    denominator = expected + abs(expected - value['trajectory_lengths'])
    require(denominator > 0, 'undefined CLS for zero-length annotation and trajectory')
    value.update(trajectory_steps=transitions, action_steps=len(path)-1,
                 DTW=dtw, nDTW=ndtw, SDTW=value['success']*ndtw, CLS=coverage*expected/denominator)
    return value


def finite_probability(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, label)


def audit_rollout(result, truth, graphs, distances, errors):
    instr = result['instr_id']
    require(instr in truth, 'instruction absent from training annotations')
    scan, reference = truth[instr]
    require(result.get('schema') == SCHEMA and result['scan_id'] == scan, 'rollout schema/scene mismatch')
    require(result.get('intervention') is None, 'collection contains a learned policy intervention')
    graph = graphs[scan]
    trace = result['states'] if 'states' in result else result['action_trace']
    path, count = result['path'], result['decision_count']
    require(type(count) is int and 1 <= count <= 15 and len(trace) == count, 'decision budget/count mismatch')
    require([s['step'] for s in trace] == list(range(count)), 'noncontiguous decision trace')
    require(len(path) in (count, count+1), 'extra/missing executed route segments')
    require(path[0] == [reference[0]], 'initial segment contains more than start')
    nodes = flat(path)
    require(trace[-1]['executed_action'] is None, 'last actual decision is not STOP')
    visited, discovered, known_edges = set(), set(), set()
    for t, state in enumerate(trace):
        current = state['viewpoint']
        require(current in graph and current == path[t][-1], 'decision viewpoint differs from executed prefix')
        visited.add(current)
        discovered.update([current, *graph[current]])
        known_edges.update((current, other) for other in graph[current])
        known_edges.update((other, current) for other in graph[current])
        action = state['executed_action']
        if t < count-1:
            require(action is not None, 'trajectory continued after STOP')
            require(action in discovered-visited, 'moving action was unseen or already visited')
            require(path[t+1][-1] == action, 'action endpoint differs from actual route')
        if t+1 < len(path):
            segment = [current] + path[t+1]
            for a, b in zip(segment, segment[1:]):
                require(a == b or (a, b) in known_edges, 'route uses an edge not yet observed')
    scores = result['terminal_stop_scores']
    require(isinstance(scores, dict) and list(scores) == list(dict.fromkeys(s['viewpoint'] for s in trace)),
            'terminal STOP table differs from decision viewpoints/order')
    for value in scores.values(): finite_probability(value, 'invalid original STOP probability')
    endpoint = max(scores, key=scores.__getitem__)
    require(nodes[-1] == endpoint, 'endpoint differs from historical raw-STOP fallback')
    require((len(path) == count) == (trace[-1]['viewpoint'] == endpoint), 'fallback segment presence differs')
    metrics = independent_metrics(path, reference, graph, distances[scan])
    require(set(result['metrics']) == set(METRICS), 'recorded metric inventory differs')
    for name in METRICS:
        close(metrics[name], result['metrics'][name], instr+':'+name, errors)
    return metrics


def audit_reference(reference, graph):
    require(reference['condition'] in SCHEDULES, 'unknown perturbation condition')
    require(reference['target_step'] is None and reference['target_action'] is None
            and reference['prefix_checks'] == [], 'reference carries a branch target')
    visited, discovered, expected_perturbations, scores = set(), set(), [], {}
    for state in reference['states']:
        t = state['step']; ids = state['candidate_vpids']; logits = state['logits']
        current = state['viewpoint']
        visited.add(current); discovered.update([current, *graph[current]])
        require(ids and ids[0] is None and len(set(ids)) == len(ids), 'invalid candidate identities')
        require(set(ids[1:]) == discovered, 'candidate inventory differs from observed panoramic graph')
        require(len(ids) == len(logits) == len(state['valid_mask']) == len(state['visited_mask']),
                'reference action dimensions differ')
        require(all(type(v) is bool for v in state['valid_mask']+state['visited_mask']), 'nonboolean masks')
        require(all(state['valid_mask']), 'single-item global action inventory has an invalid mask')
        require(state['visited_mask'] == [v in visited for v in ids], 'visited mask differs from causal decision history')
        legal = [i for i, v in enumerate(ids) if v not in visited]
        require(all((logits[i] is not None) == (i in legal) for i in range(len(ids))), 'logit mask differs from legal actions')
        require(all(type(logits[i]) in (int, float) and math.isfinite(logits[i]) for i in legal), 'nonfinite reference logits')
        require(state['no_vp_left'] is (len(legal) == 1), 'frontier exhaustion differs from observed graph')
        require(state['forced_terminal'] is (state['no_vp_left'] or t == 14), 'forced termination differs from original budget')
        raw = ids[max(legal, key=lambda i: logits[i])]
        require(state['raw_argmax_action'] == raw, 'raw argmax differs from logits')
        peak = max(logits[i] for i in legal)
        probability = math.exp(logits[0]-peak)/math.fsum(math.exp(logits[i]-peak) for i in legal)
        close(probability, state['original_stop_probability'], 'reference raw STOP', abs_tol=1e-6, rel_tol=1e-5)
        action = None if state['forced_terminal'] else raw
        if t in SCHEDULES[reference['condition']] and not state['forced_terminal']:
            alternatives = sorted((ids[i] for i in legal if i and ids[i] != raw))
            if alternatives:
                key = [SCHEMA, reference['instr_id'], reference['seed'], reference['condition'], t, alternatives]
                action = alternatives[int(objsha(key), 16) % len(alternatives)]
                expected_perturbations.append({'step': t, 'action': action})
        require(state['expected_executed_action'] == state['executed_action'] == action,
                'reference differs from fixed perturbation/raw-policy action')
        require(state['prefix_path'] == reference['path'][:t+1], 'saved pre-action prefix differs from actual route')
        require(state['instr_id'] == reference['instr_id'] and state['scan_id'] == reference['scan_id'], 'state identity differs')
        scores[current] = state['original_stop_probability']
    require(reference['terminal_stop_scores'] == scores, 'reference STOP table lacks per-decision evidence')
    require(reference['perturbations'] == expected_perturbations, 'perturbations differ from deterministic schedule')


def audit_branch(branch, reference, baseline, metrics, errors):
    target, action = branch['target_step'], branch['target_action']
    states, trace = reference['states'], branch['action_trace']
    require(branch['instr_id'] == reference['instr_id'] and branch['scan_id'] == reference['scan_id']
            and branch['condition'] == reference['condition'] and branch['seed'] == reference['seed'], 'branch reference identity differs')
    require(type(target) is int and target in selected_steps(reference), 'branch target outside causal selection')
    require(action in candidates(states[target]), 'branch target outside legal candidate inventory')
    require(target < len(trace) and branch['path'][:target+1] == states[target]['prefix_path'], 'branch changed the executed prefix')
    for t in range(target+1):
        require(trace[t]['viewpoint'] == states[t]['viewpoint'], 'branch prefix viewpoint differs')
        expected = states[t]['executed_action'] if t < target else action
        require(trace[t]['executed_action'] == expected, 'branch prefix/target action differs')
        close(branch['terminal_stop_scores'][trace[t]['viewpoint']], states[t]['original_stop_probability'],
              'branch prefix raw STOP', abs_tol=1e-5, rel_tol=1e-5)
    require(branch['perturbations'] == [], 'replay unexpectedly applied a new perturbation')
    checks = branch['prefix_checks']
    require([c['step'] for c in checks] == list(range(target+1)), 'missing recorded prefix-logit check')
    for check in checks:
        value = check['max_absolute_logit_error']
        scale = max(abs(v) for v in states[check['step']]['logits'] if v is not None)
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e-5*(1+scale),
                'recorded prefix-logit check failed')
    anchor = action == states[target]['executed_action']
    require(branch['is_anchor'] is anchor, 'anchor label differs from actual action')
    if anchor:
        require(branch['path'] == reference['path'] and branch['decision_count'] == reference['decision_count'],
                'anchor full trajectory differs')
        require(trace == [{k: s[k] for k in ('step', 'viewpoint', 'executed_action')} for s in states], 'anchor full action trace differs')
        require(branch['terminal_stop_scores'] == reference['terminal_stop_scores'], 'anchor raw STOP evidence differs')
        for name in METRICS: close(metrics[name], baseline[name], 'anchor:'+name, errors)


def tensor(value, shape, dtype, label):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.layout == torch.strided
            and list(value.shape) == list(shape) and value.dtype == dtype
            and bool(torch.isfinite(value).all()), label+': invalid tensor shape/dtype/finiteness')


def fp16_close(actual, expected, label):
    # Half rounding plus the preceding float32 log-softmax/distance operations.
    close(float(actual), expected, label, abs_tol=2e-6, rel_tol=5.1e-4)


def audit_token(token, state, action):
    ids, logits = state['candidate_vpids'], state['logits']
    require(action in ids and logits[ids.index(action)] is not None, 'history/action token is not legal at its time')
    finite = [v for v in logits if v is not None]
    peak = max(finite)
    logp = logits[ids.index(action)] - peak - math.log(math.fsum(math.exp(v-peak) for v in finite))
    fp16_close(token[1537], max(-30., min(0., logp)), 'feature fused log probability')
    require(float(token[1540]) == float(action is None), 'feature STOP identity differs')
    fp16_close(token[1541], math.log1p(len(finite)), 'feature legal action count')
    require(all(-30 <= float(token[i]) <= 0 for i in (1536, 1537, 1538)), 'invalid log probability feature')
    require(float(token[1539]) in (0., 1.), 'invalid local-action flag')
    if not token[1539]:
        require(not bool(token[768:1536].ne(0).any()) and float(token[1538]) == 0., 'absent local action has nonzero aligned features')


def audit_records(records, reference, metrics_by_branch, graph, distance, target):
    steps = selected_steps(reference)
    require([r['step'] for r in records] == steps, 'selected-state tensor coverage/order differs')
    prior = []
    for record in records:
        t = record['step']; state = reference['states'][t]; actions = candidates(state); k = len(actions)
        require(record['candidate_actions'] == actions, 'selected-state candidates/order differ')
        require(record['instr_id'] == reference['instr_id'] and record['scan_id'] == reference['scan_id']
                and record['condition'] == reference['condition'], 'tensor identity differs')
        features, history, progress, utilities = (record[n] for n in ('features', 'history_features', 'progress', 'utilities'))
        tensor(features, (k, FEATURE_DIM), torch.float16, 'features')
        tensor(history, (2*t+1, FEATURE_DIM), torch.float16, 'causal history')
        tensor(progress, (4,), torch.float16, 'progress')
        tensor(utilities, (k, 2), torch.float32, 'utilities')
        for i, action in enumerate(actions):
            audit_token(features[i], state, action)
            expected = metrics_by_branch[(t, action)]
            close(float(utilities[i, 0]), expected['success'], 'utility SR', abs_tol=0, rel_tol=0)
            close(float(utilities[i, 1]), expected['spl'], 'utility SPL', abs_tol=6e-8, rel_tol=6e-8)
        for j in range(t+1):
            audit_token(history[2*j], reference['states'][j], None)
            if j < t: audit_token(history[2*j+1], reference['states'][j], reference['states'][j]['executed_action'])
        require(torch.equal(history[-1], features[actions.index(None)]), 'current history token differs from current STOP context')
        for previous in prior:
            old_t = previous['step']; old_history = previous['history_features']
            require(torch.equal(history[:len(old_history)], old_history), 'history rewrites a saved causal prefix')
            require(torch.equal(history[2*old_t+1], previous['features'][0]), 'history action token differs from past executed baseline')
        prefix = flat(state['prefix_path'])
        length = math.fsum(0. if a == b else graph[a][b] for a, b in zip(prefix, prefix[1:]))
        expected_progress = torch.tensor([t/14, (15-t)/15, length/100, k/4], dtype=torch.float16)
        require(torch.equal(progress, expected_progress), 'progress differs from observable executed prefix')
        teacher = record['teacher_target']
        require(type(teacher) is int and -1 <= teacher < k, 'teacher target outside candidate inventory')
        if state['viewpoint'] == target:
            require(teacher == actions.index(None), 'teacher must STOP exactly at annotated goal')
        else:
            legal = [v for i, v in enumerate(state['candidate_vpids']) if i and not state['visited_mask'][i]]
            costs = {v: distance(state['viewpoint'], v)+distance(v, target) for v in legal}
            best = min(costs.values())
            optimal = {v for v, cost in costs.items() if math.isclose(cost, best, rel_tol=1e-12, abs_tol=2e-10)}
            require((teacher == -1 and bool(optimal-set(actions))) or
                    (teacher >= 0 and actions[teacher] in optimal), 'teacher target differs from independent SPL expert')
        prior.append(record)


def audit_bundle(bundle, truth, graphs, distances, errors=None):
    errors = {} if errors is None else errors
    require(bundle.get('schema') == SCHEMA and bundle.get('split') in ('train_fit', 'train_dev'), 'bundle schema/split differs')
    reference = bundle['reference']; scan = reference['scan_id']
    baseline = audit_rollout(reference, truth, graphs, distances, errors)
    audit_reference(reference, graphs[scan])
    expected = [(t, a) for t in selected_steps(reference) for a in candidates(reference['states'][t])]
    actual = [(b['target_step'], b['target_action']) for b in bundle['branches']]
    require(actual == expected, 'branch coverage/order differs from selected candidates')
    branch_metrics = {}
    for branch in bundle['branches']:
        metrics = audit_rollout(branch, truth, graphs, distances, errors)
        audit_branch(branch, reference, baseline, metrics, errors)
        branch_metrics[(branch['target_step'], branch['target_action'])] = metrics
    audit_records(bundle['records'], reference, branch_metrics, graphs[scan], distances[scan], truth[reference['instr_id']][1][-1])
    return {'instr_id': reference['instr_id'], 'scan_id': scan, 'condition': reference['condition'],
            'reference_metrics': baseline, 'records': len(bundle['records']), 'branches': len(bundle['branches']),
            'anchors': len(selected_steps(reference)), 'perturbations': len(reference['perturbations']),
            'rescuable': any(m['success'] > baseline['success'] for m in branch_metrics.values()),
            'executed_transitions': baseline['trajectory_steps'] + sum(m['trajectory_steps'] for m in branch_metrics.values())}


def select_truth(annotation, selection, provenance, expected_count):
    truth, scans = {}, set()
    for row in read(annotation):
        scans.add(row['scan'])
        for i in range(len(row['instructions'])):
            instr = f'{row["path_id"]}_{i}'
            require(instr not in truth, 'duplicate training instruction')
            truth[instr] = (row['scan'], row['path'])
    ordered = sorted(scans, key=lambda s: objsha([provenance['partition_seed'], s]))
    require(0 < provenance['dev_fraction'] < 1 and len(ordered) > 1, 'invalid training scene partition')
    ndev = min(len(ordered)-1, max(1, round(len(ordered)*provenance['dev_fraction'])))
    dev = set(ordered[:ndev]); split = selection['split']
    require(split in ('train_fit', 'train_dev'), 'audit accepts only training partitions')
    by_scan = {}
    for instr, (scan, _) in truth.items():
        if (scan in dev) == (split == 'train_dev'): by_scan.setdefault(scan, []).append(instr)
    seed = selection['seed']
    require(seed == provenance['seed'] and selection['selection'] == 'scene_stratified', 'selection seed/rule differs')
    for ids in by_scan.values(): ids.sort(key=lambda instr: objsha([seed, instr]))
    order = sorted(by_scan, key=lambda scan: objsha([seed, scan]))
    sequence = [by_scan[scan][d] for d in range(max(map(len, by_scan.values()))) for scan in order if d < len(by_scan[scan])]
    full = sorted(sequence[:expected_count])
    require(len(full) == expected_count, 'insufficient training selection pool')
    sharded = any(k in selection for k in ('full_instr_ids', 'shard_index', 'shard_count'))
    if sharded:
        shard, count = selection['shard_index'], selection['shard_count']
        require(type(shard) is int and type(count) is int and 1 <= count <= 4 and 0 <= shard < count,
                'invalid deterministic shard')
        require(selection['full_instr_ids'] == full, 'full shard selection differs from independent reconstruction')
        expected = full[shard::count]
    else: expected = full
    require(selection['instr_ids'] == expected and selection['count'] == len(expected), 'instruction selection differs from independent reconstruction')
    require(selection['scan_ids'] == sorted({truth[i][0] for i in expected}), 'selected scene inventory differs')
    require(selection['conditions'] == list(SCHEDULES), 'condition inventory differs')
    return {i: truth[i] for i in expected}


def safe_file(directory, name):
    require(isinstance(name, str) and name not in ('', '.', '..') and Path(name).name == name, 'unsafe cache filename')
    path = Path(directory)/name
    require(not path.is_symlink() and path.is_file(), 'missing or symlinked immutable cache file')
    return path


def verified_pair(local, backup, name, digest=None, size=None):
    """Verify both extant copies without invoking collector recovery or writing."""
    paths = [safe_file(d, name) for d in (local, backup)]
    digests = [sha(p) for p in paths]
    require(digests[0] == digests[1] and (digest is None or digests[0] == digest), 'local/backup immutable checksum differs')
    if size is not None:
        require(type(size) is int and size > 0 and all(p.stat().st_size == size for p in paths), 'immutable bundle size differs')
    return paths[0]


def read_bundle(local, backup, pointer):
    name, digest = pointer['file'], pointer['sha256']
    require(re.fullmatch(r'[0-9a-f]{64}', digest) is not None, 'invalid tensor checksum')
    require(name == f'bundle-{objsha(pointer["task"])}-{digest[:16]}.pt', 'bundle filename/task identity differs')
    path = verified_pair(local, backup, name, digest, pointer['bytes'])
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'bundle changed while reading')
    bundle = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    require(len(bundle['records']) == pointer['records'], 'bundle record count differs from manifest')
    return bundle


def verify_journal(directory, backup, manifest):
    path = verified_pair(directory/'tasks', backup/'tasks', 'manifest.json')
    journal = read(path)
    identity = {'schema': SCHEMA, 'selection': manifest['selection'], 'provenance': manifest['provenance']}
    require(journal.get('schema') == 'e3_continuation_probe_tasks_v1' and journal.get('complete') is True
            and journal['identity'] == identity and journal['identity_sha256'] == objsha(identity), 'task journal identity/completion differs')
    expected_keys = {objsha(p['task']) for p in manifest['bundles']}
    require(set(journal['tasks']) == expected_keys, 'dataset/task journal coverage differs')
    for pointer in manifest['bundles']:
        key = objsha(pointer['task']); entry = journal['tasks'][key]
        require(entry['file'] == f'task-{key}.json', 'task filename differs')
        record = read(verified_pair(directory/'tasks', backup/'tasks', entry['file'], entry['sha256']))
        require(record['status'] == 'complete' and record['identity_sha256'] == objsha(identity)
                and record['task'] == pointer['task'], 'task identity differs')
        require(record['payload_sha256'] == objsha({k: v for k, v in record.items() if k != 'payload_sha256'}), 'task payload digest differs')
        require(record['result'] == {k: v for k, v in pointer.items() if k != 'task'}, 'task pointer differs from dataset manifest')
    return {'sha256': sha(path), 'tasks': len(expected_keys), 'both_copies_verified': True}



def task_inventory(manifest):
    selection = manifest['selection']
    expected = {(instr, condition) for instr in selection['instr_ids'] for condition in SCHEDULES}
    actual = [(pointer['task']['instr_id'], pointer['task']['condition']) for pointer in manifest['bundles']]
    require(len(actual) == len(expected) and set(actual) == expected,
            'complete instruction-condition coverage differs or contains duplicate tasks')
    return {key: pointer for key, pointer in zip(actual, manifest['bundles'])}


def verify_merger_source(root, digest):
    name = 'scripts/merge_continuation_v2.py'
    require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest), 'invalid merger source SHA')
    if sha(root/name) == digest:
        return {'file': name, 'sha256': digest, 'status': 'current_file_matches'}
    revisions = subprocess.run(['git', '-C', str(root), 'log', '--all', '--format=%H', '--', name],
                               check=True, capture_output=True, text=True).stdout.splitlines()
    for commit in revisions:
        result = subprocess.run(['git', '-C', str(root), 'show', f'{commit}:{name}'], capture_output=True)
        if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == digest:
            return {'file': name, 'sha256': digest, 'status': 'historical_file_matches', 'commit': commit}
    raise ValueError('merger source SHA has no matching current or historical implementation')


def verify_collection_chain(root, directory, backup, manifest, truth):
    """Derived caches retain the real shard journals instead of inventing one."""
    merged = task_inventory(manifest)
    if 'merge_provenance' not in manifest:
        return dict(verify_journal(directory, backup, manifest), kind='original_collection_journal')
    merge = manifest['merge_provenance']
    require(set(merge) == {'source_sha256', 'shard_manifests'}, 'merge provenance inventory differs')
    merger = verify_merger_source(root, merge['source_sha256'])
    sources = merge['shard_manifests']
    require(isinstance(sources, dict) and 1 <= len(sources) <= 4, 'empty or invalid merge source inventory')
    manifests, journals, paths = [], [], set()
    source_pointers = {}
    for location, digest in sources.items():
        require(isinstance(location, str) and location, 'invalid source shard location')
        source = (root/Path(location)).resolve()
        require(source not in paths and source not in (directory, backup), 'duplicate or recursive source shard location')
        paths.add(source)
        source_manifest = safe_file(source, 'dataset-manifest.json')
        require(sha(source_manifest) == digest, 'source shard manifest SHA differs')
        original = read(source_manifest)
        require(original.get('schema') == SCHEMA and original.get('complete') is True
                and 'merge_provenance' not in original, 'merge source is not an original completed shard')
        original_backup = Path(original['backup_root'])
        require(original_backup.is_absolute(), 'source shard backup root must be absolute')
        original_backup = original_backup.resolve()
        require(original_backup != source and not source.is_relative_to(original_backup)
                and not original_backup.is_relative_to(source), 'source shard backup overlaps local collection')
        verified_pair(source, original_backup, 'dataset-manifest.json', digest)
        require(original['provenance'] == manifest['provenance'], 'source shard collection provenance differs')
        pointers = task_inventory(original)
        require(not set(source_pointers).intersection(pointers), 'duplicate instruction-condition across source shards')
        journal = verify_journal(source, original_backup, original)
        for pointer in pointers.values():
            # The derived manifest must point to the exact original serialized
            # bytes. Never load/re-save or repair any source/cache object here.
            verified_pair(source/'bundles', original_backup/'bundles', pointer['file'], pointer['sha256'], pointer['bytes'])
        source_pointers.update(pointers)
        selection = original['selection']
        require(selection['count'] == len(selection['instr_ids'])
                and all(instr in truth for instr in selection['instr_ids'])
                and selection['scan_ids'] == sorted({truth[instr][0] for instr in selection['instr_ids']}),
                'source shard instruction/scene inventory differs')
        require(sha(source_manifest) == digest == sha(original_backup/'dataset-manifest.json'), 'source manifest changed while auditing')
        manifests.append(original)
        journals.append({'root': str(source), 'dataset_manifest_sha256': digest, 'journal': journal,
                         'shard_index': selection['shard_index'], 'instructions': len(selection['instr_ids'])})
    first = manifests[0]['selection']; full = first['full_instr_ids']; count = first['shard_count']
    require(type(count) is int and 1 <= count <= 4 and len(manifests) == count
            and {m['selection']['shard_index'] for m in manifests} == set(range(count)), 'missing or duplicate source shard indices')
    require(isinstance(full, list) and full and full == sorted(set(full)), 'invalid full source instruction inventory')
    seen = set()
    for original in manifests:
        selection = original['selection']; index = selection['shard_index']; ids = selection['instr_ids']
        require(type(index) is int and selection['shard_count'] == count and selection['full_instr_ids'] == full
                and all(selection[k] == first[k] for k in ('split', 'conditions', 'smoke', 'seed', 'selection')),
                'source shard selection identity differs')
        require(ids == full[index::count] and not seen.intersection(ids), 'source shards are not the fixed disjoint instruction partition')
        seen.update(ids)
    require(seen == set(full), 'source shards miss full instruction inventory')
    expected_selection = dict(first, instr_ids=full, count=len(full), shard_count=1, shard_index=0,
                              scan_ids=sorted({scan for m in manifests for scan in m['selection']['scan_ids']}))
    require(manifest['selection'] == expected_selection, 'merged selection differs from complete source shards')
    require(merged == source_pointers, 'merged pointer set differs from source shard pointers')
    return {'kind': 'derived_merge_chain', 'status': 'passed', 'merger_source': merger,
            'shards': sorted(journals, key=lambda item: item['shard_index']),
            'source_bundles': len(source_pointers), 'source_bundle_copies_verified': True,
            'original_journals_verified': True, 'synthetic_merged_journal': False}


def verify_source(root, provenance):
    source = provenance['source_files']
    require(set(source) == set(COLLECTOR_FILES) and objsha(source) == provenance['source_sha256'], 'source provenance inventory/digest differs')
    current = {name: sha(root/name) for name in source}
    changed = [name for name in source if current[name] != source[name]]
    if not changed: return {'status': 'current_files_match', 'source_sha256': objsha(source)}
    # A later audit may run after a collector update. Accept only one complete
    # historical Git snapshot matching every recorded source, never mixed files.
    history = subprocess.run(['git', '-C', str(root), 'log', '--all', '--format=%H', '--', *source],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    for commit in history:
        matches = True
        for name, expected in source.items():
            result = subprocess.run(['git', '-C', str(root), 'show', f'{commit}:{name}'], capture_output=True)
            if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected:
                matches = False; break
        if matches:
            return {'status': 'complete_historical_git_snapshot_matches', 'commit': commit,
                    'source_sha256': objsha(source), 'current_different_files': changed}
    raise ValueError('recorded collector source has no matching current files or complete Git snapshot')


def audit(root, dataset_manifest, experiment, config):
    started = time.monotonic(); root = Path(root).resolve(); manifest_path = Path(dataset_manifest).resolve()
    directory = manifest_path.parent; original_manifest_sha = sha(manifest_path)
    manifest = read(manifest_path)
    require(manifest.get('schema') == SCHEMA and manifest.get('complete') is True, 'dataset manifest incomplete or wrong schema')
    backup = Path(manifest['backup_root'])
    require(backup.is_absolute(), 'backup root must be absolute')
    backup = backup.resolve()
    require(backup != directory and not backup.is_relative_to(directory)
            and not directory.is_relative_to(backup), 'backup root overlaps collection')
    verified_pair(directory, backup, manifest_path.name)
    provenance, selection = manifest['provenance'], manifest['selection']
    require(provenance.get('schema') == SCHEMA and provenance.get('feature_dtype') == 'float16', 'provenance schema/precision differs')
    for name, pin in ASSET_PINS.items(): require(provenance[name] == pin, 'frozen asset pin differs: '+name)
    require(sha(experiment) == provenance['experiment_sha256'] and sha(config) == provenance['model_config_sha256'], 'frozen configuration identity differs')
    spec, cfg = read(experiment), read(config)
    require(spec['schema'] == SCHEMA and spec['conditions'] == list(SCHEDULES) and spec['asset_pins'] == ASSET_PINS, 'experiment protocol/pins differ')
    rules = spec['data_rules']
    require(rules['selection'] == 'scene_stratified' and rules['feature_dtype'] == 'float16'
            and rules['max_action_len'] == 15 and rules['interventions_per_episode'] == 1
            and rules['perturb_schedules'] == {k: list(v) for k, v in SCHEDULES.items()}, 'frozen data rules differ')
    require(cfg['model']['max_action_len'] == 15 and cfg['model']['batch_size'] == 1
            and cfg['model']['fusion'] == 'dynamic' and cfg['model']['enc_full_graph'] is True
            and cfg['model']['expert_policy'] == 'spl', 'unsupported model/teacher protocol')
    require(provenance['seed'] == spec['seed'] and provenance['partition_seed'] == cfg['partition_seed']
            and provenance['dev_fraction'] == cfg['dev_fraction'], 'partition/collection seed differs from frozen configuration')
    require(type(selection['smoke']) is bool and selection['split'] in ('train_fit', 'train_dev'), 'invalid training-only selection scope')
    data = root/Path(cfg['dataset_root'])
    annotation, connectivity = data/'R2R/annotations/R2R_train_enc.json', data/'R2R/connectivity'
    require(sha(annotation) == TRAIN_SHA, 'raw training annotation identity differs')
    count = spec['smoke_instructions_per_split'] if selection['smoke'] else spec['instructions'][selection['split']]
    truth = select_truth(annotation, selection, provenance, count)
    graphs, graph_identity = load_graphs(connectivity, selection['scan_ids'], CONNECTIVITY_SHA)
    distances = {scan: Distance(graph) for scan, graph in graphs.items()}
    source = verify_source(root, provenance)
    upstream = provenance['upstream_lock']
    require(upstream == read(root/'configs/upstream.json')
            and upstream['commit'] == '93e8b233164bc079a6db48b8a0a78d123ec8de41', 'upstream source lock differs')
    for name, identities in upstream['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe upstream path')
        require(sha(root/'third_party/VLN-DUET'/name) == identities['prepared'], 'prepared upstream source differs: '+name)
    journal = verify_collection_chain(root, directory, backup, manifest, truth)
    errors, rows = {}, []
    for pointer in manifest['bundles']:
        bundle = read_bundle(directory/'bundles', backup/'bundles', pointer)
        require(bundle['split'] == selection['split'] and bundle['reference']['seed'] == selection['seed'], 'bundle split/seed differs')
        require({k: bundle['reference'][k] for k in ('instr_id', 'condition')} == pointer['task'], 'bundle task differs')
        rows.append(audit_bundle(bundle, truth, graphs, distances, errors))
    require(sha(manifest_path) == sha(backup/manifest_path.name) == original_manifest_sha, 'manifest changed while auditing')
    per_condition = {}
    for condition in SCHEDULES:
        selected = [r for r in rows if r['condition'] == condition]
        per_condition[condition] = {'bundles': len(selected), 'records': sum(r['records'] for r in selected),
            'branches': sum(r['branches'] for r in selected), 'anchors': sum(r['anchors'] for r in selected),
            'reference_metrics': {name: math.fsum(r['reference_metrics'][name] for r in selected)/len(selected) for name in METRICS},
            'rescuable_unique_instructions': len({r['instr_id'] for r in selected if r['rescuable']})}
    return {'schema': AUDIT_SCHEMA, 'status': 'passed', 'created_at': datetime.now(timezone.utc).isoformat(),
        'training_only': True, 'label_only': True, 'dataset_manifest': str(manifest_path), 'dataset_manifest_sha256': original_manifest_sha,
        'split': selection['split'], 'smoke': selection['smoke'], 'selection': selection,
        'assets': {'train_annotation_sha256': sha(annotation), 'connectivity': graph_identity},
        'source': source, 'journal': journal, 'both_bundle_copies_verified': True,
        'bundles': len(rows), 'records': sum(r['records'] for r in rows), 'branches': sum(r['branches'] for r in rows),
        'anchors': sum(r['anchors'] for r in rows), 'executed_transitions': sum(r['executed_transitions'] for r in rows),
        'rescuable_unique_instructions': len({r['instr_id'] for r in rows if r['rescuable']}),
        'rescuable_scans': len({r['scan_id'] for r in rows if r['rescuable']}),
        'maximum_metric_errors': errors, 'per_condition': per_condition,
        'auditor_source_sha256': {Path(__file__).name: sha(__file__), 'audit_continuation_probe.py': sha(ROOT/'scripts/audit_continuation_probe.py')},
        'legacy_perturbation_field': 'Ignored: v2 uses perturbations, checked against SCHEDULES and actual actions.',
        'metric_fields_recomputed': list(METRICS),
        'metric_values_recomputed': len(METRICS)*(len(rows)+sum(r['branches'] for r in rows)),
        'metric_tolerance': {'absolute': 2e-10, 'relative': 2e-12},
        'resources': {'gpu_seconds': 0, 'new_navigation_episodes': 0, 'official_validation_reads': 0},
        'limitations': LIMITATIONS, 'seconds': time.monotonic()-started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'audit output already exists')
    torch.set_num_threads(1)
    report = audit(args.root, args.dataset_manifest,
                   args.experiment or args.root/'configs/e3_continuation_v2.json',
                   args.config or args.root/'configs/r2r.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'split', 'bundles', 'records', 'branches', 'anchors', 'seconds')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
