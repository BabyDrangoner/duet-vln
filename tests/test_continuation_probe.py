import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vln_improve.continuation_probe import (ProbeTaskStore, assert_prefix_state,
    candidate_actions, opportunity_summary, select_state_steps)


def state(step=0, action='a', **kwargs):
    value = dict(instr_id='i', scan_id='s', step=step, viewpoint=f'v{step}',
        heading=0., elevation=0., view_index=0, candidate_vpids=[None,'a','b','c','d','visited'],
        valid_mask=[True]*6, visited_mask=[False]*5+[True],
        logits=[0., 4., 3., 3., 2., None], no_vp_left=False,
        forced_terminal=False, prefix_path=[['v0']], raw_argmax_action='a',
        original_stop_probability=.01, executed_action=action)
    value.update(kwargs)
    return value


def test_candidate_top_two_original_action_and_stop():
    assert candidate_actions(state()) == ['a', 'b', 'c', None]
    assert candidate_actions(state(action=None)) == [None, 'a', 'b']
    assert candidate_actions(state(action=None, forced_terminal=True)) == [None]
    with pytest.raises(ValueError, match='forced terminal'):
        candidate_actions(state(forced_terminal=True))


def test_state_selection_retains_early_perturb_without_interventions():
    assert select_state_steps([state(0), state(1)], 'natural') == [0,1]
    assert select_state_steps([state(0), state(1)], 'perturb_step2') == []
    assert select_state_steps([state(i) for i in range(15)], 'natural') == [0,3,14]
    assert select_state_steps([state(i) for i in range(15)], 'perturb_step2') == [3,6,14]
    assert select_state_steps([state(i) for i in range(4)], 'perturb_step2') == [3]


@pytest.mark.parametrize('field,value', [('viewpoint','changed'), ('candidate_vpids',[None,'x']),
    ('valid_mask',[True,False]), ('visited_mask',[True,False]), ('prefix_path',[['wrong']]),
    ('heading',1.), ('logits',[0.,4.1,3.,3.,2.,None]), ('original_stop_probability',.2)])
def test_prefix_rejects_divergence(field,value):
    reference = state()
    changed = copy.deepcopy(reference); changed[field] = value
    with pytest.raises(ValueError, match='prefix replay differs'):
        assert_prefix_state(changed, reference)


def test_store_empty_local_resume_corruption_and_failed_not_accepted(tmp_path):
    root, backup = tmp_path/'first', tmp_path/'cloud'
    task = {'kind':'branch','instr_id':'i','target_step':3,'target_action':None}
    store = ProbeTaskStore(root, backup, {'split':'train_fit','assets':'sha'})
    store.fail(task, ValueError('bad prefix'))
    assert store.get(task) is None
    value = {'path':[['start'],['start','middle','return']], 'metrics':{'success':1.,'spl':.4}}
    store.put(task, value)
    restored = ProbeTaskStore(tmp_path/'empty', backup, {'split':'train_fit','assets':'sha'})
    assert restored.get(task) == value
    restored.close([task])
    with pytest.raises(ValueError, match='identity differs'):
        ProbeTaskStore(tmp_path/'wrong', backup, {'split':'train_dev','assets':'sha'})
    entry = next(iter(restored.manifest['tasks'].values()))
    (backup/entry['file']).write_text('{}')
    with pytest.raises(ValueError, match='corrupt'):
        restored.get(task)


def test_store_recovers_atomic_cloud_object_before_journal_ack(tmp_path, monkeypatch):
    task, result = {'instr_id':'i'}, {'path':[['a']]}
    store = ProbeTaskStore(tmp_path/'first', tmp_path/'cloud', {'fixed':1})
    monkeypatch.setattr(store, '_journal', lambda: (_ for _ in ()).throw(RuntimeError('evicted')))
    with pytest.raises(RuntimeError, match='evicted'):
        store.put(task, result)
    restored = ProbeTaskStore(tmp_path/'empty', tmp_path/'cloud', {'fixed':1})
    assert restored.get(task) == result
    restored.close([task])


def test_store_rejects_parseable_corrupt_unacknowledged_cloud_payload(tmp_path, monkeypatch):
    task, result = {'instr_id':'i'}, {'path':[['a']]}
    store = ProbeTaskStore(tmp_path/'first', tmp_path/'cloud', {'fixed':1})
    monkeypatch.setattr(store, '_journal', lambda: (_ for _ in ()).throw(RuntimeError('evicted')))
    with pytest.raises(RuntimeError):
        store.put(task, result)
    path = next((tmp_path/'cloud').glob('task-*.json'))
    corrupt = json.loads(path.read_text()); corrupt['result']['path'] = [['corrupted']]
    path.write_text(json.dumps(corrupt))
    restored = ProbeTaskStore(tmp_path/'empty', tmp_path/'cloud', {'fixed':1})
    with pytest.raises(ValueError, match='unacknowledged cloud object'):
        restored.get(task)


def test_oracle_one_actual_branch_per_original_instruction_and_spl_constraint():
    refs = [dict(instr_id='i',scan_id='s',condition='natural',metrics={'success':0.,'spl':0.},
                 perturbation={'applied':False},states=[state(action=None)]),
            dict(instr_id='j',scan_id='s',condition='natural',metrics={'success':1.,'spl':.8},
                 perturbation={'applied':False},states=[state(action=None)])]
    branches = [dict(instr_id='i',scan_id='s',condition='natural',metrics={'success':1.,'spl':.3},is_anchor=False),
                dict(instr_id='i',scan_id='s',condition='natural',metrics={'success':1.,'spl':.4},is_anchor=False),
                dict(instr_id='j',scan_id='s',condition='natural',metrics={'success':0.,'spl':0.},is_anchor=False),
                dict(instr_id='j',scan_id='s',condition='natural',metrics={'success':1.,'spl':.7},is_anchor=False)]
    report = opportunity_summary(refs,branches)['conditions']['natural']
    assert report['baseline_sr_percent'] == 50
    assert report['label_only_oracle_sr_percent'] == 100
    assert report['label_only_oracle_spl_percent'] == pytest.approx(60)
    assert report['rescuable_failed_instructions'] == 1
    assert report['rescue_branches_nonindependent'] == 2
    assert report['harmful_branch_instructions'] == 1


def test_cli_rejects_nontraining_and_unfrozen_request():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('probe_cli',root/'scripts/probe_continuation_actions.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    experiment = json.loads((root/'configs/e3_continuation_probe_v1.json').read_text())
    cfg = json.loads((root/'configs/r2r.json').read_text())
    args = SimpleNamespace(split='train_fit',limit=2,seed=0,engineering_smoke=True)
    module.validate_request(args,experiment,cfg)
    pins = {k: 'a'*64 for k in ('base_checkpoint_sha256', 'feature_sha256', 'train_annotation_sha256', 'connectivity_sha256')}
    module.validate_asset_pins(pins, {'asset_pins': pins})
    with pytest.raises(ValueError, match='frozen'):
        module.validate_asset_pins(dict(pins, feature_sha256='b'*64), {'asset_pins': pins})
    args.split='val_unseen'
    with pytest.raises(ValueError,match='training'):
        module.validate_request(args,experiment,cfg)
    args.split='train_fit'; args.limit=3
    with pytest.raises(ValueError,match='limit/seed'):
        module.validate_request(args,experiment,cfg)
