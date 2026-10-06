"""Terminal E2 adapter and verified resumable training-only cache storage."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import torch

from .endpoint_intervention import (InterventionInputs, build_intervention_inputs,
    build_intervention_targets, collate_interventions, select_intervention)
from .protocol import file_sha256, object_sha256

CACHE_SCHEMA = 'e2_intervention_cache_v1'
RECORD_SCHEMA = 'e2_intervention_record_v1'


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as out:
        out.write(raw); out.flush(); os.fsync(out.fileno())
    os.replace(tmp, path)


def verified_copy(source, target):
    source, target = Path(source), Path(target)
    expected = file_sha256(source)
    if target.exists():
        if file_sha256(target) != expected:
            raise ValueError(f'cloud file differs: {target}')
        return
    atomic_bytes(target, source.read_bytes())
    if file_sha256(target) != expected:
        raise ValueError(f'cloud read-back failed: {target}')


def record_inputs(record):
    x = record['inputs']
    return InterventionInputs(tuple(x['candidate_vpids']), x['baseline_index'],
        x['node_features'], x['terminal_context'], x['scalar_features'])


def validate_record(record):
    if record.get('schema') != RECORD_SCHEMA or not record.get('instr_id') or not record.get('scan_id'):
        raise ValueError('invalid intervention record identity')
    inputs = record_inputs(record)
    collate_interventions([inputs])  # validates tensor dimensions, masks and anchor
    n, anchor = len(inputs.candidate_vpids), inputs.baseline_index
    y, u = record['targets'], record['utilities']
    if (y.shape != (n, 2) or u.shape != y.shape or not torch.isfinite(y).all()
            or not torch.isfinite(u).all() or not torch.equal(y, u-u[anchor])
            or not ((u >= 0) & (u <= 1)).all()
            or record['baseline_endpoint'] != inputs.candidate_vpids[anchor]):
        raise ValueError('invalid intervention utilities or relative targets')
    metrics = record['candidate_metrics']
    if len(metrics) != n or len(record['candidate_paths']) != n:
        raise ValueError('candidate paths/metrics count differs')
    expected = torch.tensor([[x['success'], x['spl']] for x in metrics], dtype=u.dtype)
    if not torch.equal(expected, u):
        raise ValueError('candidate full-route metrics differ from utilities')


def load_records(cache_dir, *, require_complete=True):
    root = Path(cache_dir)
    m = json.loads((root/'manifest.json').read_text())
    if m.get('schema') != CACHE_SCHEMA or (require_complete and not m.get('complete')):
        raise ValueError('incomplete or incompatible intervention cache')
    seen = set()
    for shard in m['shards']:
        name = shard['file']
        if Path(name).name != name:
            raise ValueError('unsafe shard path')
        path = root/name
        if not path.exists() and m.get('backup_root'):
            path = Path(m['backup_root'])/name
        if path.stat().st_size != shard['bytes'] or file_sha256(path) != shard['sha256']:
            raise ValueError(f'cache shard checksum mismatch: {name}')
        records = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(records, list) or len(records) != shard['records']:
            raise ValueError('cache shard record count differs')
        if [r['instr_id'] for r in records] != shard['instr_ids']:
            raise ValueError('cache shard instruction inventory differs')
        for r in records:
            validate_record(r)
            condition=m['collection'].get('condition')
            if condition is not None and r['condition']!=condition:
                raise ValueError('record condition differs from cache collection')
            scans=m['collection'].get('scan_ids')
            if scans is not None and r['scan_id'] not in scans:
                raise ValueError('record scene differs from cache collection')
            if r['instr_id'] in seen:
                raise ValueError('duplicate cache instruction')
            seen.add(r['instr_id'])
            yield r
    if m.get('complete') and sorted(seen) != sorted(m['collection']['instr_ids']):
        raise ValueError('completed cache does not cover fixed instruction selection')


class InterventionCacheWriter:
    """Commit each shard to cloud, then journal, then prune acknowledged local shards."""
    def __init__(self, root, backup, provenance, collection, *, shard_size=64, keep_local_shards=2):
        self.root, self.backup = Path(root), Path(backup)
        if shard_size < 1 or keep_local_shards < 0:
            raise ValueError('invalid cache retention')
        self.root.mkdir(parents=True, exist_ok=True)
        self.backup.mkdir(parents=True, exist_ok=True)
        self.shard_size, self.keep_local = shard_size, keep_local_shards
        self.pending, self.seen = [], set()
        identity = {'schema': CACHE_SCHEMA, 'provenance': provenance, 'collection': collection,
                    'backup_root': str(self.backup.resolve())}
        cloud, local = self.backup/'manifest.json', self.root/'manifest.json'
        # Cloud journal is authoritative. An interrupted unacknowledged local shard
        # is never treated as collected data; it is deterministically recollected.
        if cloud.exists():
            m = json.loads(cloud.read_text())
            if any(m.get(k) != v for k,v in identity.items()):
                raise ValueError('cache resume identity differs')
            atomic_bytes(local, cloud.read_bytes())
            self.manifest = m
            for r in load_records(self.root, require_complete=False):
                self.seen.add(r['instr_id'])
        else:
            if local.exists():
                raise ValueError('local manifest exists without verified cloud journal')
            self.manifest = dict(identity, shards=[], complete=False)
            self._journal()

    def _journal(self):
        raw = (json.dumps(self.manifest, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
        atomic_bytes(self.backup/'manifest.json', raw)
        if hashlib.sha256((self.backup/'manifest.json').read_bytes()).hexdigest() != hashlib.sha256(raw).hexdigest():
            raise ValueError('cloud cache journal verification failed')
        atomic_bytes(self.root/'manifest.json', raw)

    def add(self, record):
        validate_record(record)
        instr = record['instr_id']
        condition=self.manifest['collection'].get('condition')
        if condition is not None and record['condition']!=condition:
            raise ValueError('record condition differs from cache collection')
        if instr in self.seen:
            return
        if self.manifest['complete'] or instr not in self.manifest['collection']['instr_ids']:
            raise ValueError('unexpected instruction or closed cache')
        self.seen.add(instr); self.pending.append(record)
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        index = len(self.manifest['shards'])
        tmp = self.root/f'shard-{index:05d}.pt.tmp'
        with tmp.open('wb') as f:
            torch.save(self.pending, f); f.flush(); os.fsync(f.fileno())
        name = f'shard-{index:05d}-{file_sha256(tmp)[:16]}.pt'
        path = self.root/name
        os.replace(tmp, path)
        # A prior interrupted cloud write may have the same shard filename.
        # Only byte-identical contents can be acknowledged; preserve differences.
        verified_copy(path, self.backup/name)
        self.manifest['shards'].append({'file':name, 'sha256':file_sha256(path),
            'bytes':path.stat().st_size, 'records':len(self.pending),
            'instr_ids':[r['instr_id'] for r in self.pending]})
        self._journal(); self.pending=[]
        retained = self.manifest['shards'][-self.keep_local:] if self.keep_local else []
        keep = {x['file'] for x in retained}
        for s in self.manifest['shards']:
            p=self.root/s['file']
            if s['file'] not in keep and p.exists():
                if file_sha256(self.backup/s['file']) != s['sha256']:
                    raise ValueError('refusing cleanup of unverified cloud shard')
                p.unlink()

    def close(self):
        self.flush()
        if self.seen != set(self.manifest['collection']['instr_ids']):
            raise ValueError('cache incomplete; journal retained for resume')
        self.manifest['complete']=True
        self._journal()
        return self.manifest


def select_records(records, selection='lexicographic', limit=None, seed=0):
    if selection == 'lexicographic':
        ordered=sorted(records, key=lambda r:r['instr_id'])
    elif selection == 'scene_stratified':
        by_scan={}
        for r in records:
            by_scan.setdefault(r['scan'], []).append(r)
        for rows in by_scan.values():
            rows.sort(key=lambda r:object_sha256([seed,r['instr_id']]))
        scans=sorted(by_scan, key=lambda s:object_sha256([seed,s]))
        ordered=[]
        for j in range(max(map(len,by_scan.values()),default=0)):
            ordered.extend(by_scan[s][j] for s in scans if j<len(by_scan[s]))
    else:
        raise ValueError('unknown deterministic selection')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    return ordered if limit is None else ordered[:limit]


class TerminalIntervention:
    """Observe terminal state; decide before consulting labels; execute full return."""
    def __init__(self, agent, *, condition='natural', seed=0, model=None,
                 predict_gains=None, collect=False, baseline_paths=None):
        if condition not in ('natural','perturb_step2') or (condition != 'natural' and not collect):
            raise ValueError('perturbation is training collection only')
        self.agent,self.condition,self.seed=agent,condition,seed
        self.model,self.predict_gains,self.collect=model,predict_gains,collect
        self.baseline_paths=baseline_paths
        self.original_move=agent.make_equiv_action
        self.visits={};self.stop={};self.terminal={};self.result=None;self.record=None
        self.perturbation={'scheduled_step':2,'applied':False}

    def __call__(self, nav_inputs, nav_outs, observations, ended, step, trajectories):
        if len(observations)!=1 or bool(ended[0]):
            raise ValueError('E2 requires exactly one active observation')
        ob=observations[0];vp=str(ob['viewpoint'])
        if vp in self.visits:
            raise ValueError('decision viewpoint revisited')
        self.visits[vp]=int(step)
        self.stop[vp]=float(torch.softmax(nav_outs['fused_logits'],1)[0,0])
        self.terminal={'inputs':nav_inputs,'outputs':nav_outs}
        if self.condition=='perturb_step2' and step==2 and not bool(nav_inputs['no_vp_left'][0]):
            logits=nav_outs['fused_logits'][0]
            original=int(logits.argmax())
            legal=[i for i in range(1,len(logits)) if bool(torch.isfinite(logits[i]))
                   and bool(nav_inputs['gmap_masks'][0,i])
                   and not bool(nav_inputs['gmap_visited_masks'][0,i])]
            alternatives=[i for i in legal if i!=original]
            if alternatives:
                ids=nav_inputs['gmap_vpids'][0]
                alternatives.sort(key=lambda i:ids[i])
                pick=int(object_sha256([str(ob['instr_id']),self.seed,'perturb_step2',[ids[i] for i in alternatives]]),16)%len(alternatives)
                selected=alternatives[pick]
                changed=dict(nav_outs);changed['fused_logits']=nav_outs['fused_logits'].clone()
                changed['fused_logits'][0].fill_(-torch.inf)
                changed['fused_logits'][0,selected]=0
                self.perturbation.update(applied=True,original_action=ids[original],forced_action=ids[selected])
                return changed
        return nav_outs

    def make_equiv_action(self, actions, graphs, observations, trajectories):
        result=self.original_move(actions,graphs,observations,trajectories)
        graph,ob=graphs[0],observations[0]
        # Preserve the original policy's STOP evidence even on the forced step.
        graph.node_stop_scores[ob['viewpoint']]['stop']=self.stop[str(ob['viewpoint'])]
        if actions[0] is not None:
            return result
        instr,scan,current=map(str,(ob['instr_id'],ob['scan'],ob['viewpoint']))
        nodes=list(graph.node_stop_scores)
        if set(nodes)!=set(self.visits):
            raise ValueError('observed-node whitelist differs from original STOP inventory')
        stops={v:float(graph.node_stop_scores[v]['stop']) for v in nodes}
        anchor=max(nodes,key=stops.__getitem__)
        prefix=copy.deepcopy(trajectories[0]['path'])
        flat=[v for segment in prefix for v in segment]
        length=math.fsum(math.dist(graph.node_positions[a],graph.node_positions[b]) for a,b in zip(flat,flat[1:]))
        returns={v:float(graph.graph.distance(current,v)) if v!=current else 0.0 for v in nodes}
        inputs=build_intervention_inputs(self.terminal['inputs'],self.terminal['outputs'],
            observed_vpids=nodes,visit_steps=self.visits,stop_probabilities=stops,
            baseline_vpid=anchor,termination_vpid=current,prefix_length_m=length,return_distances_m=returns)
        paths=[]
        for v in nodes:
            p=copy.deepcopy(prefix)
            if v!=current:p.append(graph.graph.path(current,v))
            paths.append(p)
        original_path=paths[inputs.baseline_index]
        if self.baseline_paths is not None and original_path!=self.baseline_paths.get(instr):
            raise ValueError('online prefix or original historical fallback differs from frozen baseline')
        chosen=anchor;gains=None
        if self.model is not None:
            device=next(self.model.parameters()).device
            batch=collate_interventions([inputs],device=device)
            with torch.no_grad():gains=self.predict_gains(self.model,batch)[0]
            chosen=select_intervention(gains,inputs.candidate_vpids,anchor,valid_mask=batch.valid_mask[0])
        # Freeze the decision before any ground-truth access below.
        self.result={'instr_id':instr,'scan_id':scan,'condition':self.condition,
            'candidate_vpids':nodes,'baseline_endpoint':anchor,'selected_endpoint':chosen,
            'termination_endpoint':current,'endpoint_changed':chosen!=anchor,
            'prefix_path':prefix,'baseline_trajectory':original_path,
            'online_path_and_termination_parity':True if self.baseline_paths is not None else None,
            'selected_trajectory':paths[nodes.index(chosen)],'perturbation':self.perturbation,
            'predicted_gains':gains.detach().cpu().tolist() if gains is not None else None}
        if self.collect:
            metrics=[self.agent.env._eval_item(scan,p,ob['gt_path']) for p in paths]
            distances=self.agent.env.shortest_distances[scan]
            ref=math.fsum(distances[a][b] for a,b in zip(ob['gt_path'],ob['gt_path'][1:]))
            target=build_intervention_targets(
                candidate_goal_distances_m=torch.tensor([m['nav_error'] for m in metrics],dtype=torch.float64),
                candidate_total_lengths_m=torch.tensor([m['trajectory_lengths'] for m in metrics],dtype=torch.float64),
                reference_path_length_m=ref,baseline_index=inputs.baseline_index)
            utilities=torch.tensor([[m['success'],m['spl']] for m in metrics],dtype=torch.float64)
            if not torch.allclose(target,utilities-utilities[inputs.baseline_index],rtol=0,atol=2e-12):
                raise ValueError('target differs from full executed-route evaluator')
            self.record={'schema':RECORD_SCHEMA,'instr_id':instr,'scan_id':scan,'condition':self.condition,
                'inputs':{'candidate_vpids':nodes,'baseline_index':inputs.baseline_index,
                    'node_features':inputs.node_features.detach().cpu().float(),
                    'terminal_context':inputs.terminal_context.detach().cpu().float(),
                    'scalar_features':inputs.scalar_features.detach().cpu().float()},
                'targets':utilities-utilities[inputs.baseline_index],'utilities':utilities,
                'candidate_metrics':[{k:float(v) for k,v in m.items()} for m in metrics],
                'candidate_paths':paths,'baseline_endpoint':anchor,'termination_endpoint':current,
                'prefix_path':prefix,'perturbation':copy.deepcopy(self.perturbation)}
            validate_record(self.record)
        if chosen!=anchor:
            for v in nodes:graph.node_stop_scores[v]['stop']=float(v==chosen)
        return result

    def finish(self, trajectories):
        if (self.result is None or len(trajectories)!=1
                or trajectories[0]['instr_id']!=self.result['instr_id']
                or trajectories[0]['path']!=self.result['selected_trajectory']):
            raise ValueError('actual complete navigation differs from intended endpoint execution')
        return self.result
