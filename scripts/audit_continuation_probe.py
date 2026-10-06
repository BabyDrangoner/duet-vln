#!/usr/bin/env python3
"""Independent training-only audit of completed E3 continuation probe tasks.

Rebuild graphs from raw training connectivity and recompute full-route metrics.
No collector imports, rollout, fitting, validation annotations, or new outcome
slices. Compact branch logs do not retain future raw logits; that proof limit
is reported explicitly instead of being claimed as an independent success.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import time

TRAIN_SHA='8ffdfd5a5c5efeef56af7883ebac28d854a64161f1d0a3d0e52ba75086b726b8'
CONNECTIVITY_SHA='ef4a536632ad2d76b94c5b7ee41fffcc470a33844f7e7a9ca048739cdaa7d6af'
FIELDS=('success','spl','nav_error','trajectory_lengths','oracle_success','oracle_error')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objsha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    def reject(value):raise ValueError('nonfinite JSON '+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def close(a,b,label,errors=None,*,abs_tol=2e-10,rel_tol=2e-12):
    require(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b),label+': nonfinite/non-numeric')
    require(math.isclose(a,b,abs_tol=abs_tol,rel_tol=rel_tol),f'{label}: independently recomputed {a}, recorded {b}')
    if errors is not None:errors[label.rsplit(':',1)[-1]]=max(errors.get(label.rsplit(':',1)[-1],0.),abs(a-b))


def flat(path):
    require(isinstance(path,list) and path and all(isinstance(s,list) and s for s in path),'empty trajectory or segment')
    nodes=[v for s in path for v in s]
    require(all(isinstance(v,str) and v for v in nodes),'invalid trajectory node')
    return nodes


class Distance:
    def __init__(self,graph):self.graph,self.cache=graph,{}
    def __call__(self,a,b):
        require(a in self.graph and b in self.graph,'path node absent from raw graph')
        if a not in self.cache:
            values,queue={a:0.},[(0.,a)]
            while queue:
                d,node=heapq.heappop(queue)
                if d!=values[node]:continue
                for other,cost in self.graph[node].items():
                    candidate=d+cost
                    if candidate<values.get(other,math.inf):
                        values[other]=candidate;heapq.heappush(queue,(candidate,other))
            self.cache[a]=values
        require(b in self.cache[a],'disconnected route/reference')
        return self.cache[a][b]


def load_graphs(directory,scans,expected_inventory):
    files=sorted(Path(directory).glob('*_connectivity.json'))
    inventory={p.name:sha(p) for p in files}
    require(objsha(inventory)==expected_inventory,'complete raw connectivity identity differs')
    graphs={}
    for scan in sorted(scans):
        rows=read(Path(directory)/(scan+'_connectivity.json'))
        graph={r['image_id']:{} for r in rows if r['included']}
        for i,row in enumerate(rows):
            if not row['included']:continue
            require(len(row['unobstructed'])==len(rows),'connectivity dimensions differ')
            for j,connected in enumerate(row['unobstructed']):
                other=rows[j]
                if not connected or not other['included']:continue
                require(other['unobstructed'][i],'asymmetric raw connectivity')
                weight=math.sqrt(math.fsum((row['pose'][k]-other['pose'][k])**2 for k in (3,7,11)))
                require(math.isfinite(weight) and weight>=0,'invalid raw graph edge length')
                graph[row['image_id']][other['image_id']]=weight
        require(graph,'empty selected graph')
        graphs[scan]=graph
    return graphs,{'inventory_sha256':objsha(inventory),'files':len(inventory),
        'used_scan_sha256':{s:inventory[s+'_connectivity.json'] for s in sorted(scans)}}


def recompute(path,reference,graph,distance):
    nodes=flat(path)
    require(nodes[0]==reference[0],'executed path does not begin at instruction start')
    for a,b in zip(nodes,nodes[1:]):
        require(a in graph and b in graph and (a==b or b in graph[a]),f'nonexecutable edge {a}->{b}')
    # Sum all recorded edges, including segment boundaries and final fallback.
    length=math.fsum(0. if a==b else graph[a][b] for a,b in zip(nodes,nodes[1:]))
    ref_length=math.fsum(distance(a,b) for a,b in zip(reference,reference[1:]))
    error=distance(nodes[-1],reference[-1]);success=float(error<3.)
    # Every physically executed node counts, including transit inside an action
    # segment and the final historical return. This is not decision-node history.
    oracle_error=min(distance(node,reference[-1]) for node in nodes)
    return {'success':success,'spl':success*ref_length/max(ref_length,length,.01),
        'nav_error':error,'trajectory_lengths':length,
        'oracle_success':float(oracle_error<3.),'oracle_error':oracle_error},len(nodes)-1


def selection_steps(states,condition):
    require(condition in ('natural','perturb_step2'),'unknown condition')
    minimum=0 if condition=='natural' else 3
    steps=[0,3] if condition=='natural' else [3,6]
    steps.append(len(states)-1)
    return sorted({t for t in steps if minimum<=t<len(states)})


def candidates(state):
    base=state['executed_action']
    if state['forced_terminal']:
        require(base is None,'forced terminal did not STOP')
        return [None]
    ids,scores=state['candidate_vpids'],state['logits']
    indices=[i for i in range(1,len(ids)) if state['valid_mask'][i]
             and not state['visited_mask'][i] and scores[i] is not None and math.isfinite(scores[i])]
    require(base is None or base in [ids[i] for i in indices],'illegal original executed action')
    alternatives=sorted((i for i in indices if ids[i]!=base),key=lambda i:(-scores[i],ids[i]))[:2]
    return list(dict.fromkeys([base]+[ids[i] for i in alternatives]+[None]))


def trace_of(result):
    return result['states'] if 'states' in result else result['action_trace']


def audit_reference_policy(result):
    states=result['states'];seen=[];perturb_applied=False;perturb_action=None
    for s in states:
        t=s['step'];ids=s['candidate_vpids'];scores=s['logits']
        require(ids and ids[0] is None and len(set(ids))==len(ids),'invalid reference candidate identities')
        require(len(ids)==len(scores)==len(s['valid_mask'])==len(s['visited_mask']),'reference action dimensions differ')
        require(all(type(v) is bool for v in s['valid_mask']+s['visited_mask']),'nonboolean masks')
        require(s['valid_mask'][0] and not s['visited_mask'][0],'invalid STOP mask')
        legal=[i for i in range(len(ids)) if s['valid_mask'][i] and not s['visited_mask'][i]]
        require(all((scores[i] is not None)==(i in legal) for i in range(len(ids))),'finite mask differs from legal actions')
        require(all(type(scores[i]) in (int,float) and math.isfinite(scores[i]) for i in legal),'invalid raw reference logits')
        winner=max(legal,key=lambda i:scores[i]);raw=ids[winner]
        require(s['raw_argmax_action']==raw,'reference raw argmax inconsistent with logits')
        peak=max(scores[i] for i in legal)
        probability=math.exp(scores[0]-peak)/math.fsum(math.exp(scores[i]-peak) for i in legal)
        close(probability,s['original_stop_probability'],'reference raw STOP',abs_tol=1e-6,rel_tol=1e-5)
        require(s['forced_terminal']==(s['no_vp_left'] or t==14),'forced terminal does not match original budget')
        selected=None if s['forced_terminal'] else raw
        if result['condition']=='perturb_step2' and t==2 and not s['forced_terminal']:
            alternatives=sorted((i for i in legal if i and ids[i]!=raw),key=lambda i:ids[i])
            if alternatives:
                index=int(objsha([result['instr_id'],result['seed'],'perturb_step2',[ids[i] for i in alternatives]]),16)%len(alternatives)
                selected=ids[alternatives[index]];perturb_applied=True;perturb_action=selected
        require(s['executed_action']==selected,'reference action differs from fixed raw-policy/perturbation rule')
        require(s['expected_executed_action']==selected,'reference requested action differs from executed action')
        prefix=result['path'][:t+1]
        require(s['prefix_path']==prefix,'reference saved pre-action prefix differs from actual path')
        seen.append(s['viewpoint'])
    require(bool(result['perturbation']['applied'])==perturb_applied,'actual perturbation flag differs')
    if perturb_applied:require(result['perturbation']['forced_action']==perturb_action,'perturbation chosen action differs')


def audit_rollout(result,truth,graphs,distances,errors):
    instr=result['instr_id'];require(instr in truth,'instruction absent from training annotations')
    scan,reference=truth[instr]
    require(result['scan_id']==scan,'rollout/annotation scene mismatch')
    trace=trace_of(result);count=result['decision_count'];path=result['path']
    require(type(count) is int and 1<=count<=15 and len(trace)==count,'decision budget/count mismatch')
    require([s['step'] for s in trace]==list(range(count)),'noncontiguous decision trace')
    require(len(path) in (count,count+1),'extra/missing actual route segments')
    require(path[0]==[reference[0]],'initial segment must contain only start node')
    require(trace[-1]['executed_action'] is None,'last actual decision is not STOP')
    scores={}
    for t,state in enumerate(trace):
        require(state['viewpoint']==path[t][-1],'decision viewpoint differs from actual executed prefix')
        probability=state['original_stop_probability']
        require(type(probability) in (int,float) and math.isfinite(probability) and 0<=probability<=1,'invalid recorded original STOP probability')
        scores[state['viewpoint']]=probability
        if t<count-1:
            require(state['executed_action'] is not None,'trajectory continued after STOP')
            require(path[t+1][-1]==state['executed_action'],'action endpoint differs from executed segment')
    require(scores==result['terminal_stop_scores'],'terminal STOP table differs from per-decision evidence')
    anchor=max(scores,key=scores.__getitem__)
    require(flat(path)[-1]==anchor,'actual endpoint differs from original-evidence historical fallback')
    require((len(path)==count)==(trace[-1]['viewpoint']==anchor),'fallback segment presence inconsistent')
    recomputed,transitions=recompute(path,reference,graphs[scan],distances[scan])
    for field in FIELDS:close(recomputed[field],result['metrics'][field],instr+':'+field,errors)
    close(float(transitions),result['metrics']['trajectory_steps'],instr+':trajectory_steps',errors)
    close(float(len(path)-1),result['metrics']['action_steps'],instr+':action_steps',errors)
    if 'states' in result:audit_reference_policy(result)
    return recomputed,transitions


def audit_branch(branch,reference,recomputed,baseline,errors):
    target=branch['target_step'];action=branch['target_action'];states=reference['states'];trace=branch['action_trace']
    require(branch['condition']==reference['condition'] and branch['seed']==reference['seed'],'branch reference condition/seed mismatch')
    require(target in selection_steps(states,reference['condition']),'branch target outside fixed observable state selection')
    require(action in candidates(states[target]),'branch action outside fixed observable candidate selection')
    require(target<len(trace),'branch never reached selected target')
    require(branch['path'][:target+1]==states[target]['prefix_path'],'branch changed the actual shared prefix')
    for t in range(target+1):
        require(trace[t]['viewpoint']==states[t]['viewpoint'],'branch/reference prefix viewpoint differs')
        close(trace[t]['original_stop_probability'],states[t]['original_stop_probability'],'branch prefix STOP',abs_tol=1e-5,rel_tol=1e-5)
        expected=states[t]['executed_action'] if t<target else action
        require(trace[t]['executed_action']==expected,'branch prefix/target action differs')
    checks=branch['prefix_checks']
    require([r['step'] for r in checks]==list(range(target+1)),'missing recorded raw-prefix check')
    for entry in checks:
        error=entry['max_absolute_logit_error']
        values=[abs(v) for v in states[entry['step']]['logits'] if v is not None]
        bound=1e-5*(1+max(values))
        require(type(error) in (int,float) and math.isfinite(error) and 0<=error<=bound,'recorded raw-prefix logit check invalid')
    anchor=action==states[target]['executed_action']
    require(branch['is_anchor'] is anchor,'anchor label differs from actual target action')
    if anchor:
        require(branch['path']==reference['path'] and branch['decision_count']==reference['decision_count'],'anchor full actual trajectory differs')
        require(branch.get('anchor_trajectory_and_all_states_equal') is True and branch.get('anchor_metrics_equal') is True,'anchor implementation checks missing')
        require([(s['step'],s['viewpoint'],s['executed_action']) for s in trace]
                ==[(s['step'],s['viewpoint'],s['executed_action']) for s in states],'anchor full action trace differs')
        for field in FIELDS:close(recomputed[field],baseline[field],'anchor:'+field,errors)
    for field in FIELDS:close(baseline[field],branch['baseline_metrics'][field],'branch baseline:'+field,errors)
    require(branch['rescue']==(recomputed['success']>baseline['success']),'rescue flag disagrees with independently computed SR')
    require(branch['harm']==(recomputed['success']<baseline['success']),'harm flag disagrees with independently computed SR')
    close(recomputed['spl']-baseline['spl'],branch['spl_gain'],'branch:spl_gain',errors)
    require(branch.get('label_only') is True,'counterfactual must remain label-only')


def truth_and_selection(annotation,selection,provenance):
    rows=read(annotation);truth={};scans=set()
    for row in rows:
        scan=row['scan'];scans.add(scan)
        for i in range(len(row['instructions'])):
            instr=f'{row["path_id"]}_{i}'
            require(instr not in truth,'duplicate training instruction')
            truth[instr]=(scan,row['path'])
    ordered=sorted(scans,key=lambda s:objsha([provenance['partition_seed'],s]))
    fraction=provenance['dev_fraction'];require(0<fraction<1,'invalid scene partition')
    ndev=min(len(ordered)-1,max(1,round(len(ordered)*fraction)))
    dev=set(ordered[:ndev]);split=selection['split']
    require(split in ('train_fit','train_dev'),'only training partitions may be audited')
    pool={i:t for i,t in truth.items() if (t[0] in dev)==(split=='train_dev')}
    by_scan={}
    for instr,(scan,_) in pool.items():by_scan.setdefault(scan,[]).append(instr)
    seed=selection['seed']
    for instrs in by_scan.values():instrs.sort(key=lambda i:objsha([seed,i]))
    scene_order=sorted(by_scan,key=lambda s:objsha([seed,s]))
    ordered_ids=[]
    for depth in range(max(map(len,by_scan.values()))):
        ordered_ids.extend(by_scan[s][depth] for s in scene_order if depth<len(by_scan[s]))
    expected=sorted(ordered_ids[:selection['limit']])
    require(selection['selection']=='scene_stratified' and selection['instr_ids']==expected,'selected IDs differ from independent fixed-selection reconstruction')
    require(selection['instruction_set_sha256']==objsha(expected),'instruction identity digest differs')
    require(selection['scan_ids']==sorted({truth[i][0] for i in expected}),'selected scene identity differs')
    return {i:truth[i] for i in expected}


def audit(root,probe_dir):
    started=time.monotonic();root,probe_dir=Path(root),Path(probe_dir)
    manifest_path=probe_dir/'tasks/manifest.json';manifest=read(manifest_path)
    require(manifest.get('schema')=='e3_continuation_probe_tasks_v1' and manifest.get('complete') is True,'probe task manifest must be complete')
    identity=manifest['identity'];require(objsha(identity)==manifest['identity_sha256'],'manifest identity fingerprint invalid')
    provenance,selection=identity['provenance'],identity['selection']
    require(selection.get('split') in ('train_fit','train_dev'),'validation audit is forbidden')
    require(selection.get('conditions')==['natural','perturb_step2'],'fixed conditions differ')
    require(selection.get('scope') in ('engineering_smoke','training_opportunity_probe'),'unknown probe scope')
    expected_count=2 if selection['scope']=='engineering_smoke' else 64
    require(selection.get('limit')==expected_count and selection.get('seed')==0,'frozen count or seed differs')
    cfg=read(root/'configs/r2r.json');require(sha(root/'configs/r2r.json')==provenance['model_config_file_sha256'],'baseline configuration SHA differs')
    dataset=Path(cfg['dataset_root']);dataset=dataset if dataset.is_absolute() else root/dataset
    annotation=dataset/'R2R/annotations/R2R_train_enc.json'
    require(sha(annotation)==provenance['train_annotation_sha256']==TRAIN_SHA,'raw training annotation SHA differs')
    require(provenance['connectivity_sha256']==CONNECTIVITY_SHA,'raw connectivity pin differs')
    require(provenance['model']==cfg['model'] and provenance['partition_seed']==cfg['partition_seed']
            and provenance['dev_fraction']==cfg['dev_fraction'],'cache partition/model differs from frozen baseline config')
    require(provenance['model']['max_action_len']==15 and provenance['model']['fusion']=='dynamic','original action budget/fusion changed')
    experiment=root/'configs/e3_continuation_probe_v1.json'
    require(sha(experiment)==provenance['experiment_sha256'],'frozen probe specification SHA differs')
    require(objsha(provenance['source_files'])==provenance['source_sha256'],'recorded source identity invalid')
    for name,expected in provenance['source_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe source path')
        require(sha(root/name)==expected,'executed collector source differs: '+name)
    truth=truth_and_selection(annotation,selection,provenance)
    graphs,graph_identity=load_graphs(dataset/'R2R/connectivity',{v[0] for v in truth.values()},CONNECTIVITY_SHA)
    distances={s:Distance(g) for s,g in graphs.items()}
    references,branches,tasks={},{},{}
    for key,entry in manifest['tasks'].items():
        name=entry['file'];require(Path(name).name==name,'unsafe task filename')
        path=probe_dir/'tasks'/name
        require(path.is_file() and sha(path)==entry['sha256'],'completed task SHA differs: '+name)
        record=read(path);task=record['task'];result=record['result']
        require(record.get('status')=='complete' and record['identity_sha256']==manifest['identity_sha256'],'incomplete/mixed-identity task')
        require(objsha(task)==key,'task request fingerprint differs')
        require(task['instr_id']==result['instr_id'] and task['condition']==result['condition'],'task/result identity mismatch')
        require(result['seed']==selection['seed'],'rollout seed differs')
        require(result['instr_id'] in truth and result['condition'] in selection['conditions'],'task outside selected training inventory')
        if task['kind']=='reference':
            require(set(task)=={'kind','instr_id','condition'},'unexpected reference task fields')
            require(result['target_step'] is None and result['target_action'] is None,'reference contains intervention target')
            references[(result['instr_id'],result['condition'])]=result
        else:
            require(task['kind']=='branch' and set(task)=={'kind','instr_id','condition','target_step','target_action'},'unexpected task kind/fields')
            require(task['target_step']==result['target_step'] and task['target_action']==result['target_action'],'branch target/task identity differs')
            branches[key]=result
        tasks[key]=task
    expected_refs={(i,c) for i in truth for c in selection['conditions']}
    require(set(references)==expected_refs,'missing selected reference episode/condition')
    expected_tasks={}
    for (instr,condition),reference in references.items():
        task={'kind':'reference','instr_id':instr,'condition':condition};expected_tasks[objsha(task)]=task
        for step in selection_steps(reference['states'],condition):
            for action in candidates(reference['states'][step]):
                task={'kind':'branch','instr_id':instr,'condition':condition,'target_step':step,'target_action':action}
                expected_tasks[objsha(task)]=task
    require(tasks==expected_tasks,'task inventory does not cover exactly the fixed observable candidate plan')
    errors={};reference_metrics={};transitions=0;anchor_count=0;longest=0
    for key,result in references.items():
        metrics,edges=audit_rollout(result,truth,graphs,distances,errors)
        reference_metrics[key]=metrics;transitions+=edges;longest=max(longest,result['decision_count'])
    for result in branches.values():
        key=(result['instr_id'],result['condition'])
        metrics,edges=audit_rollout(result,truth,graphs,distances,errors)
        audit_branch(result,references[key],metrics,reference_metrics[key],errors)
        transitions+=edges;longest=max(longest,result['decision_count']);anchor_count+=int(result['is_anchor'])
    summary_path=probe_dir/'probe-summary.json'
    if summary_path.exists():
        summary=read(summary_path)
        require(summary.get('complete') is True and summary['task_manifest_sha256']==sha(manifest_path),'summary does not bind complete task journal')
        require(summary['selection']==selection and summary['provenance']==provenance,'summary/task provenance differs')
        require(summary['reference_rollouts']==len(references) and summary['branch_rollouts']==len(branches)
                and summary['anchor_checks']==anchor_count,'summary task counts differ')
    return {'schema':'e3_independent_training_probe_audit_v2','status':'passed',
        'scope':'Training-only fixed probe artifact audit; no new rollout, outcome slices, fitting, or validation access.',
        'recorded_utc':datetime.now(timezone.utc).isoformat(),'audit_source_sha256':sha(__file__),
        'split':selection['split'],'probe_scope':selection['scope'],'instructions':len(truth),'scenes':len(graphs),
        'reference_rollouts_checked':len(references),'branch_rollouts_checked':len(branches),
        'anchor_full_path_checks':anchor_count,'metric_values_recomputed':len(FIELDS)*(len(references)+len(branches)),
        'metric_fields_recomputed':list(FIELDS),
        'actual_adjacent_transitions_checked':transitions,'maximum_decision_count':longest,
        'max_absolute_metric_errors':errors,'metric_tolerance':{'absolute':2e-10,'relative':2e-12},
        'artifact_identity':{'manifest_sha256':sha(manifest_path),'identity_sha256':manifest['identity_sha256'],
            'training_annotation_sha256':sha(annotation),'connectivity':graph_identity,
            'probe_summary_sha256':sha(summary_path) if summary_path.exists() else None},
        'checks':{'fixed_training_scene_partition_and_instruction_selection':True,'exact_task_inventory_and_sha256':True,
            'all_adjacent_edges_physically_legal':True,'complete_path_including_returns_charged':True,
            'sr_spl_ne_length_recomputed_from_raw_graph':True,'original_15_decision_budget':True,
            'full_executed_path_oracle_success_and_error_recomputed_from_raw_graph':True,
            'reference_actions_match_stored_raw_logits_and_fixed_perturbation':True,
            'recorded_original_stop_evidence_and_historical_endpoint_consistent':True,
            'branch_shared_prefix_and_target_action_match':True,'anchor_entire_path_and_metrics_match':True},
        'trace_semantics':{'reference_raw_actions_and_stop_softmax':'independently checked against retained raw reference logits',
            'oracle_proximity':'minimum raw-graph distance to the annotated goal over every node in the complete executed path, including transit and historical return; oracle_success uses the strict distance < 3 m rule, not decision-node proximity or policy goal recognition',
            'branch_prefix_and_forced_target':'independently checked using action trace, reference states and complete actual path',
            'branch_future_unmodified_duet_argmax':'not independently verifiable from compact branch trace: future raw argmax/logits are not retained',
            'branch_future_original_stop_probability':'recorded value is checked for range, terminal-table consistency and prefix agreement; future raw-softmax equality is not independently provable',
            'collector_source':'recorded implementation file SHAs matched current files; code specifies one forced target followed by pure DUET, which is an implementation constraint rather than an independent future-logit audit',
            'known_graph_only_routing':'raw graph edge legality is verified; compact logs do not retain every revealed edge inventory, so discovery-time availability is not independently reconstructed'},
        'resources':{'cpu_seconds':time.monotonic()-started,'gpu_seconds':0,'new_navigation_episodes':0,'official_validation_reads':0}}


def selfcheck():
    # Synthetic chain with 2 m edges and a 6 m reference; none of these are study outcomes.
    graph={x:{} for x in 'ABCD'}
    for a,b in zip('ABC','BCD'):graph[a][b]=graph[b][a]=2.
    distance=Distance(graph);truth={'synthetic':('s',list('ABCD'))};graphs={'s':graph};distances={'s':distance}
    def state(t,v,ids,logits,visited,action,prefix):
        finite=[x for x in logits if x is not None];peak=max(finite)
        prob=math.exp(logits[0]-peak)/sum(math.exp(x-peak) for x in finite)
        return {'instr_id':'synthetic','scan_id':'s','step':t,'viewpoint':v,'heading':0.,'elevation':0.,'view_index':12,
            'candidate_vpids':ids,'logits':logits,'valid_mask':[True]*len(ids),'visited_mask':visited,
            'no_vp_left':False,'forced_terminal':False,'raw_argmax_action':ids[max(range(len(ids)),key=lambda i:-math.inf if logits[i] is None else logits[i])],
            'original_stop_probability':prob,'prefix_path':prefix,'executed_action':action,'expected_executed_action':action}
    states=[state(0,'A',[None,'A','B','C'],[0.,None,2.,1.],[False,True,False,False],'B',[['A']]),
            state(1,'B',[None,'A','B','C','D'],[4.,None,None,1.,0.],[False,True,True,False,False],None,[['A'],['B']])]
    def metrics(path):
        m,n=recompute(path,list('ABCD'),graph,distance)
        return dict(m,trajectory_steps=float(n),action_steps=float(len(path)-1))
    reference={'instr_id':'synthetic','scan_id':'s','condition':'natural','seed':0,'path':[['A'],['B']],
        'states':states,'decision_count':2,'perturbation':{'applied':False},'target_step':None,'target_action':None,
        'terminal_stop_scores':{s['viewpoint']:s['original_stop_probability'] for s in states}}
    reference['metrics']=metrics(reference['path']);errors={}
    base,_=audit_rollout(reference,truth,graphs,distances,errors)
    require(base['oracle_error']==4. and base['oracle_success']==0.,'synthetic never-entered path oracle differs')
    trace=[{k:s[k] for k in ('step','viewpoint','executed_action','original_stop_probability')} for s in states]
    trace[1]['executed_action']='C'
    trace.append({'step':2,'viewpoint':'C','executed_action':None,'original_stop_probability':.99})
    branch={'instr_id':'synthetic','scan_id':'s','condition':'natural','seed':0,'path':[['A'],['B'],['C']],
        'action_trace':trace,'decision_count':3,'terminal_stop_scores':{s['viewpoint']:s['original_stop_probability'] for s in trace},
        'target_step':1,'target_action':'C','is_anchor':False,'label_only':True,
        'prefix_checks':[{'step':0,'max_absolute_logit_error':0.},{'step':1,'max_absolute_logit_error':0.}],
        'baseline_metrics':reference['metrics'],'rescue':True,'harm':False,'spl_gain':1.}
    branch['metrics']=metrics(branch['path'])
    computed,_=audit_rollout(branch,truth,graphs,distances,errors)
    audit_branch(branch,reference,computed,base,errors)
    require(computed['oracle_error']==2. and computed['oracle_success']==1.,'synthetic successful endpoint oracle differs')
    anchor=copy.deepcopy(branch)
    anchor.update(path=copy.deepcopy(reference['path']),action_trace=[{k:s[k] for k in ('step','viewpoint','executed_action','original_stop_probability')} for s in states],
        decision_count=2,terminal_stop_scores=reference['terminal_stop_scores'].copy(),target_action=None,is_anchor=True,
        metrics=reference['metrics'].copy(),rescue=False,harm=False,spl_gain=0.,
        anchor_trajectory_and_all_states_equal=True,anchor_metrics_equal=True)
    anchor_values,_=audit_rollout(anchor,truth,graphs,distances,errors)
    audit_branch(anchor,reference,anchor_values,base,errors)
    threshold_graph={a:{b:1.5 for b in edges} for a,edges in graph.items()}
    boundary=recompute([['A'],['B']],list('ABCD'),threshold_graph,Distance(threshold_graph))[0]
    require(boundary['success']==0. and boundary['oracle_success']==0. and boundary['oracle_error']==3.,
        'SR and oracle SR boundary must be strictly below 3 m')
    # Explicit return and repeated segment boundary must both be charged.
    returned=[['A'],['B','C'],['B','A']]
    require(recompute(returned,list('ABCD'),graph,distance)[0]['trajectory_lengths']==8.,'synthetic return undercharged')
    transit=recompute(returned,list('ABCD'),graph,distance)[0]
    require(transit['nav_error']==6. and transit['success']==0. and transit['oracle_error']==2. and transit['oracle_success']==1.,
        'oracle must include a transit node even when the final endpoint fails')
    goal_return=recompute([['A'],['B','C','D'],['C','B','A']],list('ABCD'),graph,distance)[0]
    require(goal_return['nav_error']==6. and goal_return['success']==0. and goal_return['oracle_error']==0. and goal_return['oracle_success']==1.,
        'historical return must not erase earlier exact-goal proximity')
    require(recompute([['A'],['A','B']],list('ABCD'),graph,distance)[0]['trajectory_lengths']==2.,'duplicate segment boundary mishandled')
    failures=0
    for defect in ('metric','illegal_edge','budget','wrong_prefix','false_anchor','stop_evidence','oracle_success','oracle_error'):
        bad=copy.deepcopy(branch)
        if defect=='metric':bad['metrics']['spl']=.75
        elif defect=='illegal_edge':bad['path']=[['A'],['B'],['D']]
        elif defect=='budget':bad['decision_count']=16
        elif defect=='wrong_prefix':bad['path']=[['A'],['C'],['C']]
        elif defect=='false_anchor':bad['is_anchor']=True
        elif defect=='oracle_success':bad['metrics']['oracle_success']=0.
        elif defect=='oracle_error':bad['metrics']['oracle_error']=3.
        else:bad['terminal_stop_scores']['C']=.01
        try:
            m,_=audit_rollout(bad,truth,graphs,distances,{})
            audit_branch(bad,reference,m,base,{})
        except ValueError:failures+=1
        else:raise AssertionError('did not reject synthetic defect '+defect)
    return {'status':'passed','synthetic':True,'synthetic_checks':18,'rejected_corruptions':failures,
            'metric_fields_recomputed':list(FIELDS),
            'notice':'Synthetic implementation checks only; no probe or validation files read.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--probe-dir',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--selfcheck',action='store_true')
    args=p.parse_args(argv)
    if args.selfcheck:
        require(args.root is None and args.probe_dir is None and args.output is None,'selfcheck cannot read or write experiment paths')
        print(json.dumps(selfcheck(),indent=2));return
    require(args.root is not None and args.probe_dir is not None,'root and probe-dir are required')
    output=args.output or args.probe_dir/'audit.json'
    require(not output.exists(),'preserve previous audit: output already exists')
    result=audit(args.root,args.probe_dir)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'status':result['status'],'output':str(output),'sha256':sha(output),
        'reference_rollouts':result['reference_rollouts_checked'],'branch_rollouts':result['branch_rollouts_checked']}))


if __name__=='__main__':main()
