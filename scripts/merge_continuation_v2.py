#!/usr/bin/env python3
"""Verify complete disjoint shards and copy their unchanged tensor bundles."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vln_improve.continuation_v2 import SCHEMA, load_bundle
from vln_improve.intervention_runtime import verified_copy
from vln_improve.pipeline import atomic_json, backup_mount_identity, validate_backup_root, validate_separate_roots
from vln_improve.protocol import file_sha256


def validate_shards(manifests):
    if not manifests:
        raise ValueError('no shards')
    first = manifests[0]
    count = first['selection']['shard_count']
    if len(manifests) != count or {m['selection']['shard_index'] for m in manifests} != set(range(count)):
        raise ValueError('incomplete shard indices')
    expected = first['selection']['full_instr_ids']
    seen = set()
    for m in manifests:
        s = m['selection']
        if (m.get('schema') != SCHEMA or not m.get('complete') or m['provenance'] != first['provenance']
                or s['full_instr_ids'] != expected or s['shard_count'] != count
                or any(s[k] != first['selection'][k] for k in ('split','conditions','smoke','seed'))):
            raise ValueError('shard identity differs')
        ids = s['instr_ids']
        if ids != expected[s['shard_index']::count] or seen.intersection(ids):
            raise ValueError('shard instruction membership differs')
        tasks = [(x['task']['instr_id'], x['task']['condition']) for x in m['bundles']]
        required = {(i,c) for i in ids for c in s['conditions']}
        if len(tasks) != len(required) or set(tasks) != required:
            raise ValueError('incomplete or duplicated condition coverage')
        seen.update(ids)
    if seen != set(expected):
        raise ValueError('incomplete full instruction inventory')
    return expected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards', type=Path, nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--backup-dir', type=Path, required=True)
    a=p.parse_args()
    a.output_dir=a.output_dir.resolve();a.backup_dir=a.backup_dir.expanduser().resolve()
    validate_separate_roots(a.output_dir,a.backup_dir)
    mount=backup_mount_identity(a.backup_dir,backend='filesystem')
    def check():validate_backup_root(a.backup_dir,backend='filesystem',expected_identity=mount)
    manifests=[json.loads((r/'dataset-manifest.json').read_text()) for r in a.shards]
    expected=validate_shards(manifests)
    rows,pointers=[],[]
    for root,m in zip(a.shards,manifests):
        old_backup=Path(m['backup_root'])
        old_mount=backup_mount_identity(old_backup,backend='filesystem')
        def old_check():validate_backup_root(old_backup,backend='filesystem',expected_identity=old_mount)
        for pointer in m['bundles']:
            bundle=load_bundle(root/'bundles',old_backup/'bundles',pointer,old_check)
            if (bundle['reference']['instr_id'] != pointer['task']['instr_id']
                    or bundle['reference']['condition'] != pointer['task']['condition']):
                raise ValueError('payload task differs')
            check();old_check()
            source=root/'bundles'/pointer['file']
            verified_copy(source,a.backup_dir/'bundles'/pointer['file'])
            check()
            verified_copy(source,a.output_dir/'bundles'/pointer['file'])
            pointers.append(pointer)
        summary=json.loads((root/'collection-summary.json').read_text())
        if not summary['all_anchors_and_prefixes_passed']:
            raise ValueError('failed shard integrity')
        rows.extend(summary['rows'])
    pointers.sort(key=lambda r:(r['task']['instr_id'],r['task']['condition']))
    rows.sort(key=lambda r:(r['instr_id'],r['condition']))
    selection=dict(manifests[0]['selection'],instr_ids=expected,count=len(expected),
        scan_ids=sorted({s for m in manifests for s in m['selection']['scan_ids']}),shard_count=1,shard_index=0)
    out=dict(manifests[0],selection=selection,backup_root=str(a.backup_dir),bundles=pointers,
        merge_provenance={'source_sha256':file_sha256(__file__),
            'shard_manifests':{str(r):file_sha256(r/'dataset-manifest.json') for r in a.shards}})
    summary={'schema':SCHEMA,'complete':True,'label_only':True,'selection':selection,'rows':rows,
        'records':sum(r['records'] for r in rows),'branches':sum(r['branches'] for r in rows),
        'anchors':sum(r['anchors'] for r in rows),'all_anchors_and_prefixes_passed':True,
        'rescuable_unique_instructions':len({r['instr_id'] for r in rows if r['rescuable']}),
        'rescuable_scans':len({r['scan_id'] for r in rows if r['rescuable']})}
    for name,value in [('dataset-manifest.json',out),('collection-summary.json',summary)]:
        check();atomic_json(a.output_dir/name,value)
        verified_copy(a.output_dir/name,a.backup_dir/name);check()
    print(json.dumps({'complete':True,'instructions':len(expected),'bundles':len(pointers)}))


if __name__=='__main__':main()
