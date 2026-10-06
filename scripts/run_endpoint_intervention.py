#!/usr/bin/env python3
"""Collect resumable E2 train caches or execute a frozen endpoint head in DUET."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vln_improve.protocol import file_sha256,object_sha256,resolve_config
from vln_improve.intervention_runtime import (InterventionCacheWriter,TerminalIntervention,
    load_records,select_records,verified_copy,atomic_bytes)

COLLECTION_CODE=("scripts/run_endpoint_intervention.py","src/vln_improve/intervention_runtime.py",
 "src/vln_improve/endpoint_intervention.py","scripts/run_duet.py","scripts/prepare_duet.py",
 "src/vln_improve/protocol.py")
EVALUATION_CODE=COLLECTION_CODE+("src/vln_improve/intervention_training.py",
 "scripts/train_endpoint_intervention.py","src/vln_improve/checkpoint_store.py",
 "src/vln_improve/resumable.py","src/vln_improve/endpoint_pairs.py",
 "src/vln_improve/pipeline.py","src/vln_improve/run_assets.py",
 "scripts/evaluate_endpoint_groups.py","src/vln_improve/study_ledger.py")


def code_identity(evaluation=False):
    files={p:file_sha256(ROOT/p) for p in (EVALUATION_CODE if evaluation else COLLECTION_CODE)}
    return files,object_sha256(files)


def cache_report(records, provenance, collection, resources):
    episodes=[];trajectories=[]
    for r in sorted(records,key=lambda r:r['instr_id']):
        i=r['inputs']['baseline_index']
        episodes.append(dict(instr_id=r['instr_id'],scan_id=r['scan_id'],**r['candidate_metrics'][i]))
        trajectories.append({'instr_id':r['instr_id'],'trajectory':r['candidate_paths'][i]})
    mean=lambda key:sum(r[key] for r in episodes)/len(episodes)
    summary={key:100*mean(metric) for key,metric in [('sr','success'),('spl','spl'),('nDTW','nDTW'),('SDTW','SDTW'),('CLS','CLS'),('oracle_sr','oracle_success')]}
    summary.update({'nav_error':mean('nav_error'),'oracle_error':mean('oracle_error'),'action_steps':mean('action_steps'),
                    'steps':mean('trajectory_steps'),'lengths':mean('trajectory_lengths')})
    return {'metadata':dict(provenance,mode='e2_collect',split=collection['split'],
                num_episodes=len(episodes),subset=collection['limit'] is not None,
                condition=collection['condition'],collection=collection),
            'summary':summary,'episodes':episodes,'trajectories':trajectories,'resources':resources}


def validate_head_identity(meta, specification, experiment_sha, provenance, seed):
    from vln_improve.intervention_training import code_identity as training_code_identity
    training = meta.get('config', {})
    keys = ('seed','epochs','batch_size','hidden_dim','lr','weight_decay','monitor_every_epochs','risk_weight')
    if (training.get('experiment_sha256') != experiment_sha or training.get('seed') != seed
            or any(training.get(k) != specification[k] for k in keys)
            or meta.get('arm') not in specification['arms'] or training.get('arm') != meta['arm']
            or (specification.get('arm') is not None and meta['arm'] != specification['arm'])
            or type(meta.get('epoch')) is not int or meta['epoch'] < 1
            or meta.get('code_identity') != training_code_identity()):
        raise ValueError('head training config/source differs from frozen experiment')
    identity=meta.get('data_identity', {})
    sources=[*identity.get('fit', []), identity.get('dev')]
    if len(sources)!=3 or any(not isinstance(x,dict) for x in sources):
        raise ValueError('head lacks two fit caches and one natural development cache')
    expected_provenance=dict(provenance)
    expected_provenance['experiment_sha256']=specification.get('collection_config_sha256',experiment_sha)
    for source in sources:
        if source.get('provenance') != expected_provenance:
            raise ValueError('head training cache provenance differs from navigation assets/source')
        if source.get('collection',{}).get('scope')!='research':
            raise ValueError('engineering-smoke cache cannot support formal navigation head')
    c=specification['collection']
    fit=identity['fit'];dev=identity['dev']
    if ({x['collection']['condition'] for x in fit}!=set(c['fit_conditions'])
            or any(x['records']!=c['fit_instruction_count'] or x['collection']['split']!='train_fit' for x in fit)
            or fit[0]['collection']['instr_ids']!=fit[1]['collection']['instr_ids']
            or dev['records']!=c['dev_instruction_count'] or dev['collection']['condition']!='natural'
            or dev['collection']['split']!='train_dev'):
        raise ValueError('head cache coverage differs from frozen formal experiment')


def validate_collection_request(args, specification):
    if args.seed != specification['seed']:
        raise ValueError('collection seed differs from frozen experiment')
    if args.engineering_smoke:
        if args.mode!='collect' or args.split not in ('train_fit','train_dev') or args.limit is None:
            raise ValueError('engineering smoke requires an explicit training-only collection subset')
        return
    if args.mode!='collect':return
    c=specification['collection']
    if args.split=='train_fit':
        if (args.limit!=c['fit_instruction_count'] or args.selection!=c['fit_selection']
                or args.condition not in c['fit_conditions']):
            raise ValueError('fit collection differs from frozen instruction/condition plan')
    elif args.limit is not None or args.condition!='natural':
        raise ValueError('formal natural development collection must cover the full split')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('collect','eval'))
    p.add_argument('--split',choices=('train_fit','train_dev','val_unseen'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/r2r.json')
    p.add_argument('--experiment',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--backup',type=Path)
    p.add_argument('--cache',type=Path)
    p.add_argument('--condition',choices=('natural','perturb_step2'),default='natural')
    p.add_argument('--selection',choices=('lexicographic','scene_stratified'),default='lexicographic')
    p.add_argument('--limit',type=int)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--shard-size',type=int,default=64)
    p.add_argument('--keep-local-shards',type=int,default=2)
    p.add_argument('--head',type=Path)
    p.add_argument('--baseline-report',type=Path)
    p.add_argument('--study',type=Path,default=ROOT/'configs/research_study.json')
    p.add_argument('--ledger',type=Path)
    p.add_argument('--access-id')
    p.add_argument('--category',choices=('pilot','confirmatory'),default='pilot')
    p.add_argument('--execution-backup-root',type=Path)
    p.add_argument('--print-code-sha256',action='store_true')
    p.add_argument('--engineering-smoke',action='store_true')
    p.add_argument('--allow-local-backup-for-tests',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.print_code_sha256:
        print(code_identity(evaluation=True)[1]);return
    if any(x is None for x in (args.mode,args.split,args.output,args.backup,args.experiment)):
        p.error('mode, split, output, backup and experiment are required')
    if args.limit is not None and args.limit<1:p.error('limit must be positive')
    if args.mode=='collect' and (args.split not in ('train_fit','train_dev') or args.cache is None or args.head):
        p.error('collection requires a training split, cache, and no head')
    if args.mode=='eval' and (not args.head or not args.baseline_report or args.condition!='natural'):
        p.error('evaluation requires a frozen head, baseline report and natural condition')
    if args.split=='train_dev' and args.condition!='natural':p.error('train_dev must remain natural')
    if args.output.exists():raise ValueError('refusing to overwrite a completed report')
    if args.allow_local_backup_for_tests and (not args.engineering_smoke or args.mode!='collect' or args.split not in ('train_fit','train_dev')):
        raise ValueError('local backup exception requires a training-only engineering smoke')
    specification=json.loads(args.experiment.read_text())
    validate_collection_request(args,specification)
    files,code_sha=code_identity(evaluation=args.mode=='eval')
    from evaluate_endpoint_groups import validate_registration,validation_execution
    registration=validate_registration(args,code_sha)
    from vln_improve.pipeline import validate_backup_root,atomic_json
    backup_kind=validate_backup_root(args.backup,allow_local=args.allow_local_backup_for_tests)
    cfg=resolve_config(args.config,ROOT)
    if cfg['model']['batch_size']!=1:raise ValueError('E2 adapter requires batch_size=1')
    import run_duet
    lock=run_duet.verify()
    import torch
    dataset=Path(cfg['dataset_root'])
    provenance={'base_checkpoint_sha256':file_sha256(cfg['base_checkpoint']),
        'feature_sha256':file_sha256(dataset/'R2R/features/pth_vit_base_patch16_224_imagenet.hdf5'),
        'train_annotation_sha256':file_sha256(dataset/'R2R/annotations/R2R_train_enc.json'),
        'connectivity_sha256':object_sha256({f.name:file_sha256(f) for f in sorted((dataset/'R2R/connectivity').glob('*_connectivity.json'))}),
        'model':cfg['model'],'partition_seed':cfg['partition_seed'],'dev_fraction':cfg['dev_fraction'],
        'upstream_lock':lock,'collection_code_sha256':code_identity(False)[1],
        'collection_code_files':code_identity(False)[0],
        'experiment_sha256':file_sha256(args.experiment),'torch_version':str(torch.__version__)}
    context=validation_execution(args,registration,code_sha) if args.split=='val_unseen' else nullcontext(None)
    with context as claim:
        baseline_paths=None
        if args.baseline_report:
            reference=json.loads(args.baseline_report.read_text())
            if reference['metadata']['split']!=args.split or reference['metadata'].get('base_checkpoint_sha256')!=provenance['base_checkpoint_sha256']:
                raise ValueError('reference baseline split/checkpoint differs')
            baseline_paths={r['instr_id']:r['trajectory'] for r in reference['trajectories']}
        model=predict=None
        if args.mode=='eval':
            from vln_improve.intervention_training import load_head,predict_gains
            model=load_head(args.head,device='cuda');model.eval();predict=predict_gains
            meta=model.training_metadata
            validate_head_identity(meta,specification,file_sha256(args.experiment),provenance,args.seed)
        sys.path.insert(0,str(run_duet.DEFAULT_DEST/'map_nav_src'))
        import r2r.agent as upstream
        original_class,original_parse,original_partition=upstream.GMapNavAgent,run_duet.parse_cli,run_duet.select_partition
        writer=None;selection=None;completed={};start=time.monotonic()
        def partition(data,split,dev_fraction,partition_seed):
            nonlocal writer,selection
            rows=original_partition(data,split,dev_fraction,partition_seed)
            chosen=select_records(rows,args.selection,args.limit,args.seed)
            selection={'scope':'engineering_smoke' if args.engineering_smoke else 'research','split':args.split,'condition':args.condition,'seed':args.seed,'selection':args.selection,
                'limit':args.limit,'instr_ids':sorted(r['instr_id'] for r in chosen),
                'scan_ids':sorted({r['scan'] for r in chosen}),
                'instruction_set_sha256':object_sha256(sorted(r['instr_id'] for r in chosen)),
                'perturbation_rule':'step2; sorted finite valid unvisited nonSTOP alternatives excluding argmax; SHA([instr_id,seed,perturb_step2,alternative_vpids]) modulo count',
                'selection_rule':'scene and within-scene instruction SHA([seed,id]) ascending; round-robin scene depth'}
            if args.mode=='collect' and not args.engineering_smoke:
                c=specification['collection']
                count=c['fit_instruction_count'] if args.split=='train_fit' else c['dev_instruction_count']
                scenes=c['fit_expected_scenes'] if args.split=='train_fit' else c['dev_expected_scenes']
                if len(chosen)!=count or len(selection['scan_ids'])!=scenes:
                    raise ValueError('selected instruction/scene count differs from frozen plan')
            if writer is None and args.mode=='collect':
                writer=InterventionCacheWriter(args.cache,args.backup/'cache',provenance,selection,
                    shard_size=args.shard_size,keep_local_shards=args.keep_local_shards)
                pending=[r for r in chosen if r['instr_id'] not in writer.seen]
                if not pending:
                    raise CacheAlreadyComplete()
                return pending
            return chosen
        class Agent(original_class):
            def rollout(self,*positional,**kwargs):
                if positional or getattr(self,'decision_hook',None) is not None:raise ValueError('unexpected existing hook')
                hook=TerminalIntervention(self,condition=args.condition,seed=args.seed,model=model,
                    predict_gains=predict,collect=args.mode=='collect',baseline_paths=baseline_paths)
                old_move=self.make_equiv_action
                self.decision_hook,self.make_equiv_action=hook,hook.make_equiv_action
                try:result=super().rollout(**kwargs)
                finally:self.decision_hook,self.make_equiv_action=None,old_move
                decision=hook.finish(result)
                instr=decision['instr_id']
                if instr in completed and completed[instr]!=decision:raise ValueError('repeated trajectory differs')
                completed[instr]=decision
                if writer is not None:writer.add(hook.record)
                if len(completed)%64==0:print(json.dumps({'completed_this_attempt':len(completed),'already_acknowledged':len(writer.seen) if writer else None}),flush=True)
                return result
        raw=args.output.with_name(args.output.stem+'.attempt-'+str(time.time_ns())+'.raw.json')
        try:
            upstream.GMapNavAgent,run_duet.select_partition=Agent,partition
            # Our fixed selection occurs before original sorting; no second limit.
            run_duet.parse_cli=lambda:SimpleNamespace(mode='baseline',config=args.config,split=args.split,
                output=raw,cache=None,head=None,limit=None,seed=0)
            try:run_duet.main()
            except CacheAlreadyComplete:pass
        finally:
            upstream.GMapNavAgent,run_duet.parse_cli,run_duet.select_partition=original_class,original_parse,original_partition
        if args.mode=='collect':
            manifest=writer.close()
            records=list(load_records(args.cache))
            report=cache_report(records,provenance,selection,{'this_attempt_seconds':time.monotonic()-start,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'shards':len(manifest['shards'])})
            report['support']={'episodes':len(records),'candidates':sum(len(r['utilities']) for r in records),
                'baseline_successes':sum(float(r['utilities'][r['inputs']['baseline_index'],0]) for r in records),
                'episodes_with_positive_spl_gain':sum(bool((r['targets'][:,1]>0).any()) for r in records),
                'positive_spl_candidates':sum(int((r['targets'][:,1]>0).sum()) for r in records),
                'perturbations_applied':sum(bool(r['perturbation']['applied']) for r in records)}
        else:
            report=json.loads(raw.read_text())
            expected=set(selection['instr_ids'])
            if set(completed)!=expected or {r['instr_id'] for r in report['episodes']}!=expected:
                raise ValueError('evaluation instruction inventory differs')
            report['endpoint_decisions']=[completed[k] for k in sorted(completed)]
            report['metadata'].update(mode='e2_endpoint_intervention',head_sha256=file_sha256(args.head),
                head_metadata=model.training_metadata,baseline_report_sha256=file_sha256(args.baseline_report),
                experiment_sha256=file_sha256(args.experiment),access_id=args.access_id,
                validation_execution=claim,all_online_path_and_termination_parity=True)
        report['metadata'].update(e2_code_files=files,e2_code_sha256=code_sha,backup_verification=backup_kind)
        atomic_json(args.output,report)
        verified_copy(args.output,args.backup/args.output.name)
        print(json.dumps({'output':str(args.output),'summary':report['summary'],'support':report.get('support')}),flush=True)


class CacheAlreadyComplete(Exception):pass

if __name__=='__main__':main()
