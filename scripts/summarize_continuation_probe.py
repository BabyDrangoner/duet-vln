#!/usr/bin/env python3
"""Describe complete E3 training-only probe opportunities; standard library only.

Checks cryptographic identities and complete frozen branch inventories. Uses
recorded full-route metrics; independent geometric metric recomputation belongs
to audit_continuation_probe.py. Never trains, runs navigation, or accesses validation.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('natural', 'perturb_step2')
ASSET_KEYS = ('base_checkpoint_sha256', 'feature_sha256', 'train_annotation_sha256', 'connectivity_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def selected_steps(states, condition):
    require(states and [s['step'] for s in states] == list(range(len(states))), 'noncontiguous reference states')
    requested = [0, 3] if condition == 'natural' else [3, 6]
    requested.append(len(states)-1)
    minimum = 0 if condition == 'natural' else 3
    return sorted({x for x in requested if minimum <= x < len(states)})


def candidates(state):
    base = state['executed_action']
    if state['forced_terminal']:
        require(base is None, 'forced terminal has moving baseline')
        return [None]
    ids, logits = state['candidate_vpids'], state['logits']
    require(len(ids) == len(logits) == len(state['valid_mask']) == len(state['visited_mask']), 'candidate dimensions differ')
    require(ids and ids[0] is None and len(set(ids)) == len(ids), 'candidate identities invalid')
    legal = [i for i in range(1, len(ids)) if state['valid_mask'][i] and not state['visited_mask'][i]
             and logits[i] is not None and math.isfinite(logits[i])]
    require(base is None or base in [ids[i] for i in legal], 'illegal baseline action')
    other = sorted((i for i in legal if ids[i] != base), key=lambda i: (-logits[i], ids[i]))[:2]
    return list(dict.fromkeys([base]+[ids[i] for i in other]+[None]))


def valid_metrics(metrics):
    require(metrics['success'] in (0, 1), 'success is not a binary fraction')
    require(metrics['oracle_success'] in (0, 1), 'oracle_success is not a binary fraction')
    require(type(metrics['spl']) in (int, float) and math.isfinite(metrics['spl']) and 0 <= metrics['spl'] <= 1,
            'SPL is not a finite fraction')
    require(metrics['success'] <= metrics['oracle_success'], 'final success cannot exceed full-path oracle success')


def validate_records(identity, records, expected_count=64):
    """Validate only training records and reconstruct every expected reference/branch."""
    selection = identity['selection']
    require(selection['split'] in ('train_fit', 'train_dev'), 'validation/test data is forbidden')
    require(selection['scope'] == 'training_opportunity_probe', 'engineering samples cannot enter formal feasibility')
    require(selection['limit'] == expected_count and len(selection['instr_ids']) == expected_count
            and len(set(selection['instr_ids'])) == expected_count, 'requires exactly 64 selected instructions per split')
    require(selection['selection'] == 'scene_stratified' and selection['conditions'] == list(CONDITIONS), 'selection/conditions differ')
    require(selection['instruction_set_sha256'] == objsha(sorted(selection['instr_ids'])), 'selection instruction digest differs')
    refs, branches, actual_tasks = {}, [], set()
    for record in records:
        task, result = record['task'], record['result']
        key = objsha(task)
        require(key not in actual_tasks, 'duplicate task')
        actual_tasks.add(key)
        instr, condition = task['instr_id'], task['condition']
        require(instr in selection['instr_ids'] and condition in CONDITIONS, 'unexpected instruction/condition')
        require(result['instr_id'] == instr and result['condition'] == condition
                and result['scan_id'] in selection['scan_ids'] and result['seed'] == selection['seed'],
                'task result instruction/condition/scene/seed differs')
        require(type(result['decision_count']) is int and 1 <= result['decision_count'] <= 15, 'invalid 15-decision budget')
        require(isinstance(result['path'], list) and result['path'], 'missing complete executed path')
        valid_metrics(result['metrics'])
        if task['kind'] == 'reference':
            require(set(task) == {'kind', 'instr_id', 'condition'}, 'unexpected reference task fields')
            require((instr, condition) not in refs, 'duplicate reference')
            require(result['target_step'] is None and result['target_action'] is None, 'reference has intervention target')
            require(len(result['states']) == result['decision_count'], 'reference trace count differs')
            require(result['states'][-1]['executed_action'] is None, 'reference lacks terminal STOP')
            refs[instr, condition] = result
        elif task['kind'] == 'branch':
            require(set(task) == {'kind', 'instr_id', 'condition', 'target_step', 'target_action'}, 'unexpected branch task fields')
            require(result['target_step'] == task['target_step'] and result['target_action'] == task['target_action'], 'branch target differs')
            require(result['label_only'] is True, 'branch must be labelled counterfactual data')
            branches.append(result)
        else:
            raise ValueError('unknown probe task kind')
    require(set(refs) == {(instr, c) for instr in selection['instr_ids'] for c in CONDITIONS}, 'reference coverage incomplete')
    require({r['scan_id'] for r in refs.values()} == set(selection['scan_ids']), 'reference scene coverage differs')
    require(all(refs[i, 'natural']['scan_id'] == refs[i, 'perturb_step2']['scan_id'] for i in selection['instr_ids']),
            'same original instruction differs in scene across conditions')
    expected = set()
    for (instr, condition), reference in refs.items():
        expected.add(objsha({'kind': 'reference', 'instr_id': instr, 'condition': condition}))
        for step in selected_steps(reference['states'], condition):
            for action in candidates(reference['states'][step]):
                expected.add(objsha({'kind': 'branch', 'instr_id': instr, 'condition': condition,
                                    'target_step': step, 'target_action': action}))
    require(actual_tasks == expected, 'complete manifest does not cover exact frozen branch inventory')
    for b in branches:
        ref = refs[b['instr_id'], b['condition']]
        require(b['scan_id'] == ref['scan_id'] and b['perturbation'] == ref['perturbation'], 'branch/reference identity differs')
        require(b['baseline_metrics'] == ref['metrics'], 'branch baseline metrics differ')
        require(b['rescue'] == (b['metrics']['success'] > ref['metrics']['success'])
                and b['harm'] == (b['metrics']['success'] < ref['metrics']['success']), 'stored rescue/harm flags differ')
        require(math.isclose(b['spl_gain'], b['metrics']['spl']-ref['metrics']['spl'], abs_tol=1e-12, rel_tol=1e-12), 'stored SPL gain differs')
        anchor = b['target_action'] == ref['states'][b['target_step']]['executed_action']
        require(b['is_anchor'] is anchor, 'branch anchor flag differs')
        require([c['step'] for c in b['prefix_checks']] == list(range(b['target_step']+1)), 'incomplete prefix checks')
        require(all(type(c['max_absolute_logit_error']) in (int,float)
                    and math.isfinite(c['max_absolute_logit_error']) and c['max_absolute_logit_error'] >= 0
                    for c in b['prefix_checks']), 'invalid prefix check errors')
        if anchor:
            require(b['path'] == ref['path'] and b['metrics'] == ref['metrics']
                    and b.get('anchor_metrics_equal') is True and b.get('anchor_trajectory_and_all_states_equal') is True,
                    'original-action anchor differs or lacks checks')
    return list(refs.values()), branches


def load_probe(directory, expected_split, experiment):
    directory = Path(directory)
    summary_path, manifest_path = directory/'probe-summary.json', directory/'tasks/manifest.json'
    summary, manifest = read(summary_path), read(manifest_path)
    require(summary.get('schema') == 'e3_continuation_probe_summary_v1' and summary.get('complete') is True
            and summary.get('training_only') is True and summary.get('label_only') is True, 'not a completed training-only probe summary')
    require(manifest.get('schema') == 'e3_continuation_probe_tasks_v1' and manifest.get('complete') is True, 'incomplete probe manifest')
    require(summary['task_manifest_sha256'] == sha(manifest_path), 'summary manifest SHA mismatch')
    identity = manifest['identity']; provenance = identity['provenance']
    require(manifest['identity_sha256'] == objsha(identity), 'manifest identity SHA mismatch')
    require(summary['provenance'] == provenance and summary['selection'] == identity['selection'], 'summary/manifest identities differ')
    require(identity['selection']['split'] == expected_split and expected_split in ('train_fit','train_dev'), 'wrong or forbidden split')
    spec = read(experiment)
    require(spec['schema'] == 'e3_continuation_probe_v1' and provenance['experiment_sha256'] == sha(experiment), 'frozen experiment identity differs')
    require(identity['selection']['seed'] == spec['seed'], 'experiment seed differs')
    require(all(provenance[k] == spec['asset_pins'][k] for k in ASSET_KEYS), 'frozen DUET asset pins differ')
    require(provenance['source_sha256'] == objsha(provenance['source_files']), 'source inventory identity differs')
    records = []
    for key, entry in sorted(manifest['tasks'].items()):
        name = entry['file']
        require(Path(name).name == name and name == f'task-{key}.json', 'unsafe/mismatched task filename')
        path = directory/'tasks'/name
        require(sha(path) == entry['sha256'], 'task file SHA mismatch')
        record = read(path)
        require(record.get('status') == 'complete' and record.get('identity_sha256') == manifest['identity_sha256'], 'failed or wrong-identity task')
        require(objsha(record['task']) == key, 'task identity SHA mismatch')
        require(record.get('payload_sha256') == objsha({k:v for k,v in record.items() if k != 'payload_sha256'}), 'task payload SHA mismatch')
        records.append(record)
    refs, branches = validate_records(identity, records)
    require(summary['reference_rollouts'] == len(refs) and summary['branch_rollouts'] == len(branches), 'summary task counts differ')
    require(summary['all_anchor_checks_passed'] is True and summary['all_prefix_checks_passed'] is True, 'summary has failed engineering checks')
    return identity, refs, branches, {'directory':str(directory.resolve()), 'summary_sha256':sha(summary_path),
        'manifest_sha256':sha(manifest_path), 'identity_sha256':manifest['identity_sha256'],
        'experiment_sha256':sha(experiment), 'verified_task_files':len(records)}


def summarize_condition(references, branches):
    refs = {r['instr_id']:r for r in references}
    require(len(refs) == len(references), 'condition has duplicate original instructions')
    rescue, harm, positive_spl, earlier, terminal, never_entered, entered = (set() for _ in range(7))
    alternatives = [b for b in branches if not b['is_anchor']]
    for b in alternatives:
        instr = b['instr_id']; ref = refs[instr]
        if b['metrics']['success'] > ref['metrics']['success']:
            rescue.add(instr)
            (earlier if b['target_step'] < ref['states'][-1]['step'] else terminal).add(instr)
            (never_entered if ref['metrics']['oracle_success'] == 0 else entered).add(instr)
        if b['metrics']['success'] < ref['metrics']['success']:
            harm.add(instr)
        if b['metrics']['spl'] > ref['metrics']['spl']+1e-12:
            positive_spl.add(instr)
    def count(ids):
        return {'instructions':len(ids), 'scenes':len({refs[i]['scan_id'] for i in ids})}
    n = len(refs)
    return {'N_original_instructions':n, 'scenes':len({r['scan_id'] for r in references}),
        'baseline_failed_instructions':sum(r['metrics']['success']==0 for r in references),
        'baseline_sr_percent':100*sum(r['metrics']['success'] for r in references)/n,
        'baseline_spl_percent':100*sum(r['metrics']['spl'] for r in references)/n,
        'rescuable':count(rescue), 'harmful_alternative':count(harm), 'positive_delta_spl':count(positive_spl),
        'rescued_when_reference_full_path_never_entered_success_radius':count(never_entered),
        'rescued_when_reference_full_path_already_entered_success_radius':count(entered),
        'rescuable_at_least_one_earlier_step':count(earlier),
        'rescuable_only_at_reference_terminal_step':count(rescue-earlier),
        'rescuable_at_reference_terminal_step_including_earlier_overlap':count(terminal),
        'actual_perturbed_instructions':sum(bool(r['perturbation']['applied']) for r in references),
        'reference_rollouts':n, 'branch_rollouts_including_anchors':len(branches),
        'alternative_branch_rollouts_nonindependent':len(alternatives),
        'scope_note':'Counts classify only the fixed selected states and candidate inventory; no exhaustive reachability claim.'}


def analyze(fit_directory, dev_directory, experiment):
    result = {'schema':'e3_training_opportunity_feasibility_v1', 'training_only':True, 'label_only':True,
              'inputs':{}, 'splits':{}}
    identities, original_sets, scene_sets = [], [], []
    for split, directory in [('train_fit',fit_directory),('train_dev',dev_directory)]:
        identity, refs, branches, evidence = load_probe(directory,split,experiment)
        identities.append(identity); original_sets.append({r['instr_id'] for r in refs}); scene_sets.append({r['scan_id'] for r in refs})
        result['inputs'][split] = evidence
        result['splits'][split] = {c:summarize_condition([r for r in refs if r['condition']==c],
                                                       [b for b in branches if b['condition']==c]) for c in CONDITIONS}
    require(identities[0]['provenance'] == identities[1]['provenance'], 'fit/dev source and asset provenance differ')
    require(not (original_sets[0] & original_sets[1]) and not (scene_sets[0] & scene_sets[1]), 'fit/dev instruction or scene partitions overlap')
    union = len(original_sets[0] | original_sets[1])
    require(union == 128, 'expected exactly 128 unique original instructions across fit/dev and conditions')
    result['global'] = {'unique_original_instructions_union_splits_and_conditions':union,
                        'reference_condition_rollouts':256,
                        'branches_are_independent_evidence':False,
                        'natural_and_perturbed_conditions_share_original_instructions':True}
    result['scientific_interpretation'] = {
        'facts':[
            {'claim':'Opportunity is measured on 128 selected training-scene instructions, with two conditions per instruction.',
             'support':'Exact fixed selection and all branch inventories verified against complete task manifests.'},
            {'claim':'Earlier-step rescue identifies selected cases with an executable single-action correction before the original terminal decision.',
             'support':'Every counted branch executes one chosen action and then a complete frozen DUET continuation.'},
            {'claim':'Reference oracle_success describes proximity anywhere on the full executed path, including transit and historical return.',
             'support':'Its categories cannot be called observed decision-node history or evidence that the policy recognized the goal.'},
            {'claim':'Terminal-only rescue means no tested earlier selected state/candidate rescued that instruction.',
             'support':'This finite probe does not exclude rescue at other untested states or actions.'}],
        'metric_evidence':'Recorded full-route metrics; this script checks identity, coverage and metric consistency but does not independently recompute geometry.',
        'learned_policy_evaluated':False, 'new_official_validation_access':False,
        'unseen_sr_plus_5pp_established':False, 'automatic_promotion_threshold':None,
        'limitations':['Ground-truth outcomes choose hindsight categories; no inference-time ability has been demonstrated.',
                       'Natural and perturbed results must be interpreted separately; perturbation recovery cannot establish unseen improvement.',
                       'The original instructions share scenes, so they are not statistically independent scene-level replicates.']}
    return result


def self_check():
    """Synthetic classification, exact inventory and corruption checks; no dataset."""
    def ref(instr, success, spl, oracle):
        states = [{'step':t, 'executed_action':'a' if t<3 else None, 'forced_terminal':False,
                   'candidate_vpids':[None,'a','b','c'], 'logits':[1,4,3,2],
                   'valid_mask':[True]*4,'visited_mask':[False]*4} for t in range(4)]
        return {'instr_id':instr,'scan_id':'scene-'+instr,'condition':'natural','seed':0,'path':[['start']],
            'states':states,'decision_count':4,'target_step':None,'target_action':None,
            'perturbation':{'applied':False},'metrics':{'success':success,'spl':spl,'oracle_success':oracle}}
    refs = [ref('early',0,0,0),ref('terminal',0,0,1),ref('harm',1,.8,1)]
    def branch(r,step,success,spl):
        return dict(instr_id=r['instr_id'],scan_id=r['scan_id'],condition='natural',target_step=step,
                    is_anchor=False,metrics={'success':success,'spl':spl,'oracle_success':1})
    branches = [branch(refs[0],0,1,.4),branch(refs[0],3,1,.5),branch(refs[1],3,1,.3),
                branch(refs[2],0,0,0),branch(refs[2],3,1,.9)]
    s = summarize_condition(refs,branches)
    require(s['rescuable']['instructions']==2 and s['rescuable']['scenes']==2, 'synthetic rescue dedup failed')
    require(s['rescuable_at_least_one_earlier_step']['instructions']==1
            and s['rescuable_only_at_reference_terminal_step']['instructions']==1, 'synthetic temporal categories failed')
    require(s['rescued_when_reference_full_path_never_entered_success_radius']['instructions']==1
            and s['rescued_when_reference_full_path_already_entered_success_radius']['instructions']==1, 'synthetic transit categories failed')
    require(s['harmful_alternative']['instructions']==1 and s['positive_delta_spl']['instructions']==3, 'synthetic harm/SPL failed')
    require(selected_steps(refs[0]['states'],'natural')==[0,3]
            and selected_steps(refs[0]['states'][:2],'perturb_step2')==[], 'synthetic selected states failed')
    # Build all 64 x 2 references and exact branches, validate end-to-end reading.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); experiment = root/'experiment.json'
        spec = {'schema':'e3_continuation_probe_v1','seed':0,'asset_pins':{k:'a'*64 for k in ASSET_KEYS}}
        write(experiment,spec)
        provenance = dict(spec['asset_pins'],experiment_sha256=sha(experiment),source_files={'synthetic':'b'*64})
        provenance['source_sha256']=objsha(provenance['source_files'])
        directories=[]
        for split in ('train_fit','train_dev'):
            directory=root/split; directories.append(directory)
            ids=[f'{split}-{i}' for i in range(64)]
            selection={'split':split,'scope':'training_opportunity_probe','limit':64,'instr_ids':ids,
                'scan_ids':['scene-'+i for i in ids],'seed':0,'selection':'scene_stratified','conditions':list(CONDITIONS),
                'instruction_set_sha256':objsha(sorted(ids))}
            identity={'provenance':provenance,'selection':selection}; identity_sha=objsha(identity)
            manifest={'schema':'e3_continuation_probe_tasks_v1','complete':True,'identity':identity,'identity_sha256':identity_sha,'tasks':{}}
            nr=nb=0
            def add(task,r):
                record={'status':'complete','identity_sha256':identity_sha,'task':task,'result':r}
                record['payload_sha256']=objsha(record); key=objsha(task); name=f'task-{key}.json'
                write(directory/'tasks'/name,record)
                manifest['tasks'][key]={'file':name,'sha256':sha(directory/'tasks'/name)}
            for instr in ids:
                for condition in CONDITIONS:
                    r=ref(instr,1,.8,1); r['condition']=condition
                    task={'kind':'reference','instr_id':instr,'condition':condition}; add(task,r); nr+=1
                    for step in selected_steps(r['states'],condition):
                        for action in candidates(r['states'][step]):
                            b=copy.deepcopy(r); b.pop('states'); b.update(target_step=step,target_action=action,
                                label_only=True,baseline_metrics=r['metrics'],rescue=False,harm=False,spl_gain=0.,
                                is_anchor=action==r['states'][step]['executed_action'],
                                prefix_checks=[{'step':t,'max_absolute_logit_error':0.} for t in range(step+1)],
                                anchor_metrics_equal=True,anchor_trajectory_and_all_states_equal=True)
                            add({'kind':'branch','instr_id':instr,'condition':condition,'target_step':step,'target_action':action},b); nb+=1
            write(directory/'tasks/manifest.json',manifest)
            write(directory/'probe-summary.json',{'schema':'e3_continuation_probe_summary_v1','complete':True,
                'training_only':True,'label_only':True,'task_manifest_sha256':sha(directory/'tasks/manifest.json'),
                'provenance':provenance,'selection':selection,'reference_rollouts':nr,'branch_rollouts':nb,
                'all_anchor_checks_passed':True,'all_prefix_checks_passed':True})
        report=analyze(*directories,experiment)
        require(report['global']['unique_original_instructions_union_splits_and_conditions']==128, 'union dedup failed')
        corrupt_path=next((directories[0]/'tasks').glob('task-*.json'))
        original_record=read(corrupt_path)
        corrupt=copy.deepcopy(original_record); corrupt['result']['metrics']['spl']=.1; write(corrupt_path,corrupt)
        try:
            analyze(*directories,experiment)
        except ValueError as error:
            require('task file SHA' in str(error), 'unexpected corruption test failure')
        else:
            raise ValueError('task corruption was accepted')
        manifest_path=directories[0]/'tasks/manifest.json'
        summary_path=directories[0]/'probe-summary.json'
        current_manifest=read(manifest_path)
        corrupt_key=objsha(corrupt['task'])
        current_manifest['tasks'][corrupt_key]['sha256']=sha(corrupt_path)
        write(manifest_path,current_manifest)
        current_summary=read(summary_path); current_summary['task_manifest_sha256']=sha(manifest_path)
        write(summary_path,current_summary)
        try:
            analyze(*directories,experiment)
        except ValueError as error:
            require('payload SHA' in str(error), 'payload corruption test did not reach envelope verification')
        else:
            raise ValueError('corrupt payload with updated outer checksum was accepted')
        write(corrupt_path,original_record)
        current_manifest['tasks'][corrupt_key]['sha256']=sha(corrupt_path)
        branch_key=next(k for k,e in current_manifest['tasks'].items()
                        if read(directories[0]/'tasks'/e['file'])['task']['kind']=='branch')
        del current_manifest['tasks'][branch_key]
        write(manifest_path,current_manifest)
        current_summary['task_manifest_sha256']=sha(manifest_path); write(summary_path,current_summary)
        try:
            analyze(*directories,experiment)
        except ValueError as error:
            require('branch inventory' in str(error), 'missing branch did not fail inventory check')
        else:
            raise ValueError('incomplete but resealed task inventory was accepted')
        try:
            load_probe(directories[1],'val_unseen',experiment)
        except ValueError:
            pass
        else:
            raise ValueError('validation split was accepted')
    return {'self_check':'passed','checks':['instruction_dedup','scene_counts','transit_proximity_categories',
        'early_vs_terminal_rescue','harm_and_positive_spl','exact_64_both_conditions_and_branch_inventory',
        '128_original_instruction_union','manifest_and_payload_identity','corruption_rejected',
        'resealed_payload_corruption_rejected','resealed_incomplete_inventory_rejected','validation_rejected']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-probe-dir',type=Path)
    p.add_argument('--dev-probe-dir',type=Path)
    p.add_argument('--experiment',type=Path,default=ROOT/'configs/e3_continuation_probe_v1.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--self-check',action='store_true')
    args=p.parse_args()
    if args.self_check:
        print(json.dumps(self_check(),indent=2)); return
    if any(v is None for v in (args.fit_probe_dir,args.dev_probe_dir,args.output)):
        p.error('--fit-probe-dir, --dev-probe-dir and --output are required')
    require(not args.output.exists(), 'refusing to overwrite a feasibility report')
    report=analyze(args.fit_probe_dir,args.dev_probe_dir,args.experiment)
    report['analysis_script_sha256']=sha(__file__)
    write(args.output,report)
    print(json.dumps({'output':str(args.output),'global':report['global']},indent=2))


if __name__=='__main__':
    main()
