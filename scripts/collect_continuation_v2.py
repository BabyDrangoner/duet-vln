#!/usr/bin/env python3
"""Collect fixed training partitions with causal inputs and complete branch labels."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from vln_improve.continuation_v2 import (SCHEMA, SCHEDULES, CausalContinuationHook,
                                        selected_steps, save_bundle, load_bundle)
from vln_improve.continuation_probe import ProbeTaskStore, candidate_actions, assert_anchor
from vln_improve.intervention_runtime import select_records, verified_copy
from vln_improve.pipeline import atomic_json, backup_mount_identity, validate_backup_root, validate_separate_roots
from vln_improve.protocol import file_sha256, object_sha256, resolve_config

CODE_FILES = ('scripts/collect_continuation_v2.py', 'src/vln_improve/continuation_v2.py',
              'src/vln_improve/continuation_probe.py', 'src/vln_improve/features.py',
              'src/vln_improve/intervention_runtime.py', 'src/vln_improve/pipeline.py',
              'src/vln_improve/protocol.py', 'scripts/run_duet.py', 'scripts/prepare_duet.py')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/r2r.json')
    p.add_argument('--split', choices=('train_fit', 'train_dev'), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--backup-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--deadline-seconds', type=float, default=36000)
    p.add_argument('--stop-after-bundles', type=int)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    args = p.parse_args(argv)
    args.output_dir, args.backup_dir = args.output_dir.resolve(), args.backup_dir.expanduser().resolve()
    validate_separate_roots(args.output_dir, args.backup_dir)
    spec = json.loads(args.experiment.read_text())
    if spec['schema'] != SCHEMA or spec['conditions'] != list(SCHEDULES):
        raise ValueError('wrong frozen data experiment')
    count = spec['smoke_instructions_per_split'] if args.smoke else spec['instructions'][args.split]
    seed = spec['seed']
    if not 1 <= args.shard_count <= 4 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('invalid deterministic shard')
    cfg = resolve_config(args.config, ROOT)
    if (cfg['model']['batch_size'] != 1 or cfg['model']['max_action_len'] != 15
            or cfg['model']['fusion'] != 'dynamic' or args.deadline_seconds <= 0):
        raise ValueError('requires batch1 dynamic DUET with original 15 decisions')
    mount = backup_mount_identity(args.backup_dir, backend='filesystem')
    def check():
        validate_backup_root(args.backup_dir, backend='filesystem', expected_identity=mount)
    source = {name: file_sha256(ROOT/name) for name in CODE_FILES}
    import run_duet
    import torch
    lock = run_duet.verify()
    data = Path(cfg['dataset_root'])
    provenance = {'base_checkpoint_sha256': file_sha256(cfg['base_checkpoint']),
        'feature_sha256': file_sha256(data/'R2R/features/pth_vit_base_patch16_224_imagenet.hdf5'),
        'train_annotation_sha256': file_sha256(data/'R2R/annotations/R2R_train_enc.json'),
        'connectivity_sha256': object_sha256({f.name: file_sha256(f) for f in sorted((data/'R2R/connectivity').glob('*_connectivity.json'))}),
        'upstream_lock': lock, 'source_files': source, 'source_sha256': object_sha256(source),
        'experiment_sha256': file_sha256(args.experiment), 'model_config_sha256': file_sha256(args.config),
        'partition_seed': cfg['partition_seed'], 'dev_fraction': cfg['dev_fraction'], 'seed': seed,
        'torch_version': str(torch.__version__), 'feature_dtype': 'float16', 'schema': SCHEMA}
    for name, expected in spec['asset_pins'].items():
        if provenance[name] != expected:
            raise ValueError('asset differs from frozen pin: '+name)
    sys.path.insert(0, str(run_duet.DEFAULT_DEST/'map_nav_src'))
    import r2r.agent as upstream
    original_class, original_partition, original_cli = upstream.GMapNavAgent, run_duet.select_partition, run_duet.parse_cli
    store = selection = None
    stopped = False
    start = time.monotonic()
    new_bundles, pointers, expected_tasks, descriptive = 0, [], [], []
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    def boundary():
        check()
        if (stopped or time.monotonic()-start > args.deadline_seconds
                or (args.stop_after_bundles is not None and new_bundles >= args.stop_after_bundles)):
            raise InterruptedError('paused after durable bundle; resume identical arguments')
    def partition(rows, split, fraction, partition_seed):
        nonlocal store, selection
        if split != args.split:
            raise ValueError('unexpected split access')
        chosen = select_records(original_partition(rows, split, fraction, partition_seed),
                                'scene_stratified', count, seed)
        if len(chosen) != count:
            raise ValueError('selection count differs')
        full_ids = sorted(r['instr_id'] for r in chosen)
        chosen = sorted(chosen, key=lambda r:r['instr_id'])[args.shard_index::args.shard_count]
        selection = {'split': split, 'instr_ids': sorted(r['instr_id'] for r in chosen),
            'scan_ids': sorted({r['scan'] for r in chosen}), 'conditions': spec['conditions'],
            'count': len(chosen), 'full_instr_ids': full_ids,
            'shard_index': args.shard_index, 'shard_count': args.shard_count,
            'smoke': args.smoke, 'seed': seed, 'selection': 'scene_stratified'}
        store = ProbeTaskStore(args.output_dir/'tasks', args.backup_dir/'tasks',
            {'schema': SCHEMA, 'provenance': provenance, 'selection': selection}, check_backup=check)
        return chosen
    class Agent(original_class):
        def execute(self, row, condition, reference=None, step=None, action=None):
            self.env.data = [copy.deepcopy(row)]
            self.env.reset_epoch(shuffle=False)
            self.scanvp_cands, self.feedback = {}, 'argmax'
            hook = CausalContinuationHook(self, condition=condition, seed=seed,
                reference=reference, target_step=step, target_action=action,
                collect_features=reference is None)
            old_hook, old_move = self.decision_hook, self.make_equiv_action
            self.decision_hook, self.make_equiv_action = hook, hook.make_equiv_action
            try:
                with torch.no_grad():
                    trajectories = super().rollout(reset=True)
                result = hook.finish(trajectories)
            finally:
                self.decision_hook, self.make_equiv_action = old_hook, old_move
            scan, target = self.env.gt_trajs[result['instr_id']]
            result['metrics'] = {k: float(v) for k,v in self.env._eval_item(scan, result['path'], target).items()}
            if reference is not None:
                result['is_anchor'] = action == reference['states'][step]['executed_action']
                if result['is_anchor']:
                    assert_anchor(result, reference)
                # Features are never collected along a counterfactual future.
                result['action_trace'] = [{'step': s['step'], 'viewpoint': s['viewpoint'],
                    'executed_action': s['executed_action']} for s in result.pop('states')]
            return result, hook.records
        def test(self, use_dropout=False, feedback='argmax', **kwargs):
            nonlocal new_bundles
            if use_dropout or feedback != 'argmax' or kwargs:
                raise ValueError('inference settings changed')
            original_data = self.env.data
            self.results = {}
            try:
                for row in sorted(original_data, key=lambda r:r['instr_id']):
                    for condition in spec['conditions']:
                        task = {'instr_id': row['instr_id'], 'condition': condition}
                        expected_tasks.append(task)
                        pointer = store.get(task)
                        if pointer is not None:
                            bundle = load_bundle(args.output_dir/'bundles', args.backup_dir/'bundles', pointer, check)
                        else:
                            boundary()
                            reference, records = self.execute(row, condition)
                            branches = []
                            for step in selected_steps(reference):
                                outcomes = []
                                for action in candidate_actions(reference['states'][step]):
                                    branch, _ = self.execute(row, condition, reference, step, action)
                                    branches.append(branch)
                                    outcomes.append([branch['metrics']['success'], branch['metrics']['spl']])
                                record = records[step]
                                record['utilities'] = torch.tensor(outcomes, dtype=torch.float32)
                                if record['candidate_actions'] != candidate_actions(reference['states'][step]):
                                    raise ValueError('feature candidate order differs from label actions')
                            if set(records) != set(selected_steps(reference)):
                                raise ValueError('causal feature coverage differs')
                            bundle = {'schema': SCHEMA, 'split': args.split, 'reference': reference,
                                      'branches': branches, 'records': [records[t] for t in sorted(records)]}
                            pointer = save_bundle(args.output_dir/'bundles', args.backup_dir/'bundles', task, bundle, check)
                            store.put(task, pointer)
                            new_bundles += 1
                        ref = bundle['reference']
                        if condition == 'natural':
                            self.results[row['instr_id']] = {'instr_id': row['instr_id'], 'path': ref['path'], 'details': {}}
                        pointers.append(dict(pointer, task=task))
                        descriptive.append({'instr_id': row['instr_id'], 'scan_id': row['scan'],
                            'condition': condition, 'baseline_success': ref['metrics']['success'],
                            'baseline_spl': ref['metrics']['spl'], 'records': len(bundle['records']),
                            'rescuable': any(b['metrics']['success'] > ref['metrics']['success'] for b in bundle['branches']),
                            'branches': len(bundle['branches']), 'anchors': sum(b['is_anchor'] for b in bundle['branches']),
                            'perturbations_applied': len(ref['perturbations'])})
                        if len(pointers) % 8 == 0:
                            progress = {'complete': False, 'bundles': len(pointers), 'expected_bundles': len(original_data)*len(SCHEDULES),
                                'new_bundles': new_bundles, 'seconds': time.monotonic()-start,
                                'rescuable_unique_instructions': len({r['instr_id'] for r in descriptive if r['rescuable']})}
                            atomic_json(args.output_dir/'progress.json', progress)
                            print(json.dumps(progress), flush=True)
            finally:
                self.env.data = original_data
                self.env.reset_epoch(shuffle=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.output_dir/f'natural-reference-{time.time_ns()}.json'
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        upstream.GMapNavAgent, run_duet.select_partition = Agent, partition
        run_duet.parse_cli = lambda: SimpleNamespace(mode='baseline', config=args.config, split=args.split,
            output=raw, cache=None, head=None, limit=count, seed=seed)
        run_duet.main()
        if source != {name: file_sha256(ROOT/name) for name in CODE_FILES}:
            raise ValueError('collector source changed during capture')
        if provenance['experiment_sha256'] != file_sha256(args.experiment):
            raise ValueError('experiment changed during capture')
        store.close(expected_tasks)
        manifest = {'schema': SCHEMA, 'complete': True, 'selection': selection,
                    'provenance': provenance, 'backup_root': str(args.backup_dir), 'bundles': pointers}
        for name, value in [('dataset-manifest.json', manifest), ('collection-summary.json', {
                'schema': SCHEMA, 'complete': True, 'label_only': True, 'selection': selection,
                'rows': descriptive, 'records': sum(r['records'] for r in descriptive),
                'branches': sum(r['branches'] for r in descriptive), 'anchors': sum(r['anchors'] for r in descriptive),
                'all_anchors_and_prefixes_passed': True,
                'rescuable_unique_instructions': len({r['instr_id'] for r in descriptive if r['rescuable']}),
                'rescuable_scans': len({r['scan_id'] for r in descriptive if r['rescuable']})})]:
            atomic_json(args.output_dir/name, value)
            check()
            verified_copy(args.output_dir/name, args.backup_dir/name)
            check()
        print(json.dumps({'complete': True, 'bundles': len(pointers), 'seconds': time.monotonic()-start}), flush=True)
        return 0
    except InterruptedError as error:
        print(json.dumps({'paused': True, 'reason': str(error), 'new_bundles': new_bundles}), flush=True)
        return 75
    finally:
        upstream.GMapNavAgent, run_duet.select_partition, run_duet.parse_cli = original_class, original_partition, original_cli
        for s, h in handlers.items():
            signal.signal(s, h)


if __name__ == '__main__':
    raise SystemExit(main())
