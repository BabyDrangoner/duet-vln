import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('merge_v2',Path(__file__).resolve().parents[1]/'scripts/merge_continuation_v2.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def shards():
 out=[]
 ids=['a','b','c','d']
 for i in range(2):
  selected=ids[i::2]
  out.append({'schema':module.SCHEMA,'complete':True,'provenance':{'source':'frozen'},
   'selection':{'shard_count':2,'shard_index':i,'full_instr_ids':ids,'instr_ids':selected,
      'split':'train_fit','conditions':['natural','early_two'],'smoke':False,'seed':0},
   'bundles':[{'task':{'instr_id':n,'condition':c}} for n in selected for c in ['natural','early_two']]})
 return out

def test_merge_complete_disjoint_instruction_and_condition_coverage():
 assert module.validate_shards(shards())==['a','b','c','d']

@pytest.mark.parametrize('kind',['missing_shard','repeated_shard','missing_bundle','duplicate_bundle','wrong_source','wrong_membership','wrong_split','unfinished'])
def test_merge_rejects_incomplete_or_mismatched_shards(kind):
 rows=shards()
 if kind=='missing_shard':rows.pop()
 elif kind=='repeated_shard':rows[1]=copy.deepcopy(rows[0])
 elif kind=='missing_bundle':rows[1]['bundles'].pop()
 elif kind=='duplicate_bundle':rows[1]['bundles'].append(copy.deepcopy(rows[1]['bundles'][0]))
 elif kind=='wrong_source':rows[1]['provenance']['source']='changed'
 elif kind=='wrong_membership':rows[1]['selection']['instr_ids']=['a','b']
 elif kind=='wrong_split':rows[1]['selection']['split']='train_dev'
 elif kind=='unfinished':rows[1]['complete']=False
 with pytest.raises(ValueError):module.validate_shards(rows)
