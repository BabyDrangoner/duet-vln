"""Fail-closed bindings before navigation or collection starts."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('run_endpoint_intervention',ROOT/'scripts/run_endpoint_intervention.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixtures(monkeypatch):
    from vln_improve import intervention_training
    monkeypatch.setattr(intervention_training,'code_identity',lambda:{'source':'fixed'})
    specification={'seed':0,'epochs':20,'batch_size':64,'hidden_dim':128,'lr':1e-4,'weight_decay':.01,
        'monitor_every_epochs':2,'risk_weight':0.,'arms':['relative','absolute'],'arm':'relative',
        'collection_config_sha256':'master',
        'collection':{'fit_conditions':['natural','perturb_step2'],'fit_instruction_count':2,
            'dev_instruction_count':1,'fit_selection':'scene_stratified'}}
    provenance={'experiment_sha256':'arm','asset':'same'}
    cache_provenance={'experiment_sha256':'master','asset':'same'}
    def source(split,condition,ids):
        return {'records':len(ids),'provenance':cache_provenance.copy(),
            'collection':{'scope':'research','split':split,'condition':condition,'instr_ids':ids}}
    config={k:specification[k] for k in ('seed','epochs','batch_size','hidden_dim','lr','weight_decay','monitor_every_epochs','risk_weight')}
    config.update(experiment_sha256='arm',arm='relative')
    meta={'config':config,'epoch':2,'arm':'relative','code_identity':{'source':'fixed'},
        'data_identity':{'fit':[source('train_fit','natural',['a','b']),source('train_fit','perturb_step2',['a','b'])],
                         'dev':source('train_dev','natural',['c'])}}
    return meta,specification,provenance


def test_head_requires_exact_arm_experiment_and_explicit_master_collection_link(monkeypatch):
    meta,specification,provenance=fixtures(monkeypatch)
    module.validate_head_identity(meta,specification,'arm',provenance,0)
    specification.pop('collection_config_sha256')
    with pytest.raises(ValueError,match='provenance'):module.validate_head_identity(meta,specification,'arm',provenance,0)


@pytest.mark.parametrize('defect',['config','arm','source','asset','smoke','coverage'])
def test_head_rejects_mismatched_or_smoke_training(monkeypatch,defect):
    meta,specification,provenance=fixtures(monkeypatch)
    if defect=='config':meta['config']['experiment_sha256']='other'
    elif defect=='arm':meta['arm']='absolute';meta['config']['arm']='absolute'
    elif defect=='source':meta['code_identity']={}
    elif defect=='asset':meta['data_identity']['fit'][0]['provenance']['asset']='other'
    elif defect=='smoke':meta['data_identity']['dev']['collection']['scope']='engineering_smoke'
    elif defect=='coverage':meta['data_identity']['fit'][1]['collection']['instr_ids']=['a','z']
    with pytest.raises(ValueError):module.validate_head_identity(meta,specification,'arm',provenance,0)


def test_smoke_override_is_training_only_and_explicit(monkeypatch):
    _,specification,_=fixtures(monkeypatch)
    args=SimpleNamespace(mode='collect',split='train_fit',seed=0,engineering_smoke=False,
        limit=1,selection='scene_stratified',condition='natural')
    with pytest.raises(ValueError,match='frozen'):module.validate_collection_request(args,specification)
    args.engineering_smoke=True
    module.validate_collection_request(args,specification)
    args.mode='eval'
    with pytest.raises(ValueError,match='training-only'):module.validate_collection_request(args,specification)
