import copy
import json
from types import SimpleNamespace
import pytest
import torch
from vln_improve.intervention_runtime import (InterventionCacheWriter,load_records,
    record_inputs,select_records,TerminalIntervention,RECORD_SCHEMA)
from vln_improve.protocol import object_sha256


def record(instr='one'):
    u=torch.tensor([[1.,.8],[1.,.6]],dtype=torch.float64)
    return {'schema':RECORD_SCHEMA,'instr_id':instr,'scan_id':'scan','condition':'natural',
        'inputs':{'candidate_vpids':['A','B'],'baseline_index':0,
            'node_features':torch.zeros(2,768),'terminal_context':torch.zeros(1536),
            'scalar_features':torch.zeros(2,10)},'targets':u-u[0],'utilities':u,
        'candidate_metrics':[{'success':1.,'spl':.8},{'success':1.,'spl':.6}],
        'candidate_paths':[[['A']],[['A'],['B']]],'baseline_endpoint':'A',
        'termination_endpoint':'B','prefix_path':[['A'],['B']]}


def test_cache_prunes_only_verified_cloud_and_resumes_empty_local(tmp_path):
    local,cloud=tmp_path/'local',tmp_path/'cloud'
    coll={'split':'train_fit','instr_ids':['one','two','three']}
    w=InterventionCacheWriter(local,cloud,{'fixed':1},coll,shard_size=1,keep_local_shards=0)
    w.add(record());w.add(record('two'))
    assert not list(local.glob('*.pt'))
    w2=InterventionCacheWriter(tmp_path/'restored',cloud,{'fixed':1},coll,shard_size=1)
    assert w2.seen=={'one','two'}
    w2.add(record('three'));w2.close()
    rows=list(load_records(tmp_path/'restored'))
    assert [r['instr_id'] for r in rows]==['one','two','three']
    assert record_inputs(rows[0]).candidate_vpids==('A','B')


def test_cache_rejects_incomplete_corruption_and_identity_change(tmp_path):
    local,cloud=tmp_path/'local',tmp_path/'cloud'
    coll={'split':'train_fit','instr_ids':['one','two']}
    w=InterventionCacheWriter(local,cloud,{'fixed':1},coll,shard_size=1)
    w.add(record())
    with pytest.raises(ValueError,match='incomplete'):list(load_records(local))
    with pytest.raises(ValueError,match='identity'):InterventionCacheWriter(local,cloud,{'fixed':2},coll)
    shard=local/w.manifest['shards'][0]['file']
    shard.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):list(load_records(local,require_complete=False))


def test_cache_rejects_misaligned_utility(tmp_path):
    w=InterventionCacheWriter(tmp_path/'a',tmp_path/'b',{}, {'instr_ids':['one']})
    r=record();r['targets'][1,1]=.2
    with pytest.raises(ValueError,match='utilities'):w.add(r)


def test_scene_selection_balances_and_reproduces():
    rows=[{'scan':s,'instr_id':f'{s}-{i}'} for s in ('A','B','C') for i in range(6)]
    a=select_records(rows,'scene_stratified',8,0)
    b=select_records(list(reversed(rows)),'scene_stratified',8,0)
    assert a==b
    assert sorted(sum(x['scan']==s for x in a) for s in ('A','B','C'))==[2,3,3]


def test_fixed_perturbation_uses_only_observable_valid_alternative_and_keeps_original_tokens():
    agent=SimpleNamespace(make_equiv_action=lambda *a:None)
    h=TerminalIntervention(agent,condition='perturb_step2',collect=True)
    inputs={'gmap_vpids':[[None,'A','B','C','D']], 'gmap_masks':torch.tensor([[1,1,1,1,0]],dtype=torch.bool),
        'gmap_visited_masks':torch.tensor([[0,1,0,0,0]],dtype=torch.bool),'no_vp_left':[False]}
    outputs={'fused_logits':torch.tensor([[4.,-torch.inf,2.,1.,-torch.inf]]),'gmap_embeds':torch.zeros(1,5,768)}
    class Guard(dict):
        def __getitem__(self,k):
            assert k not in ('gt_path','goal','distance')
            return super().__getitem__(k)
    ob=Guard(instr_id='one',viewpoint='A',gt_path='forbidden')
    changed=h(inputs,outputs,[ob],[False],2,[])
    ids=['B','C'];pick=int(object_sha256(['one',0,'perturb_step2',ids]),16)%2
    assert inputs['gmap_vpids'][0][int(changed['fused_logits'].argmax())]==ids[pick]
    assert h.stop['A']==pytest.approx(float(torch.softmax(outputs['fused_logits'],1)[0,0]))
    assert changed['gmap_embeds'] is outputs['gmap_embeds']
    assert outputs['fused_logits'][0,0]==4
    assert h.perturbation['applied']


def terminal_fixture(*,collect=False,model=None,predict=None):
    class Graph:
        node_stop_scores={'A':{'stop':.9},'B':{'stop':.2}}
        node_positions={'A':(0.,0.,0.),'B':(1.,0.,0.)}
        def __init__(self):self.graph=self
        def path(self,a,b):return [a,b]
        def distance(self,a,b):return 1.
    class Env:
        def _eval_item(self,*args):raise AssertionError('inference attempted GT metric access')
    agent=SimpleNamespace(make_equiv_action=lambda *a:None,env=Env())
    h=TerminalIntervention(agent,collect=collect,model=model,predict_gains=predict)
    h.visits={'A':0,'B':1};h.stop={'A':.9,'B':.2}
    h.terminal={'inputs':{'gmap_vpids':[[None,'A','B']],
        'gmap_masks':torch.ones(1,3,dtype=torch.bool),'gmap_visited_masks':torch.tensor([[0,1,1]],dtype=torch.bool),
        'vp_masks':torch.ones(1,1,dtype=torch.bool),'vp_cand_vpids':[[None]]},
        'outputs':{'gmap_embeds':torch.zeros(1,3,768),'vp_embeds':torch.zeros(1,1,768)}}
    class Ob(dict):
        def __getitem__(self,k):
            if k=='gt_path':raise AssertionError('inference read GT')
            return super().__getitem__(k)
    ob=Ob(instr_id='one',scan='s',viewpoint='B')
    return h,Graph(),ob


def test_inference_has_no_ground_truth_and_complete_return_is_verified():
    h,g,ob=terminal_fixture()
    traj=[{'instr_id':'one','path':[['A'],['B']]}]
    h.make_equiv_action([None],[g],[ob],traj)
    assert h.result['selected_endpoint']=='A'
    with pytest.raises(ValueError,match='actual complete'):h.finish(traj)
    traj[0]['path'].append(['B','A'])
    h.finish(traj)


def test_intervention_modifies_only_terminal_fallback_and_checks_actual_path():
    model=torch.nn.Linear(1,1)
    h,g,ob=terminal_fixture(model=model,predict=lambda m,b:torch.tensor([[[0.,0.],[0.,.1]]]))
    traj=[{'instr_id':'one','path':[['A'],['B']]}]
    h.make_equiv_action([None],[g],[ob],traj)
    assert h.result['selected_endpoint']=='B'
    assert max(g.node_stop_scores,key=lambda k:g.node_stop_scores[k]['stop'])=='B'
    h.finish(traj)
