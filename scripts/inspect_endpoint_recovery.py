#!/usr/bin/env python3
"""Read only the known E1 study's cloud metadata before any recovery mutation.

No tensor cache is downloaded or loaded. These observations do not replace the
strict cache loaders or CheckpointStore verification required before resuming.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        result=hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):result.update(chunk)
        return result.hexdigest()


def read_record(path):
    if not path.exists():return {'exists':False}
    if path.is_symlink() or not path.is_file():raise ValueError('unexpected metadata file: '+str(path))
    return {'exists':True,'sha256':sha(path),'record':json.loads(path.read_text())}


def inspect(root):
    root=root.resolve(strict=True)
    report={'schema':'endpoint_recovery_inspection_v1',
        'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cloud_root':str(root),'drive_mount_present':os.path.ismount('/content/drive'),
        'feature_payloads_read':False,'training_or_navigation_started':False,
        'scope':'metadata inspection; strict cache/checkpoint verification still required',
        'records':{},'caches':{},'runs':{}}
    for name in ('full-pairs-status.json','controls-smoke-v2-audit.json','controls-full-v2-status.json',
                 'controls-full-v2-decision.json','e1-full-data-lock.json','e1-seed0-queue-status.json',
                 'e1-seed0-status.json','e1-seed0-comparison.json'):
        report['records'][name]=read_record(root/name)
    for family in ('endpoint-pair-full','endpoint-controls-full-v2'):
        for suffix in ('train-fit','train-dev'):
            name=family+'-'+suffix;directory=root/name
            value={'exists':directory.is_dir()}
            if directory.is_dir():
                value['pair_commit_markers_present']=sum((p/'COMMITTED.json').is_file() for p in directory.glob('pair-*'))
                value['collection']=read_record(directory/'COLLECTION.json')
                value['manifest']=read_record(directory/'manifest.json')
                value['commit']=read_record(directory/'COMMITTED.json')
                value['root_manifest_hash_verified']=(value['manifest']['exists'] and value['commit']['exists']
                    and value['commit']['record']=={'manifest_sha256':value['manifest']['sha256']})
            report['caches'][name]=value
    report['resume_acceptance']=read_record(root/'e1-resume-acceptance-v1/resume-acceptance.json')
    for arm in ('C1','C2','C3','M'):
        name='e1-seed0-'+arm;directory=root/name
        report['runs'][name]={'summary':read_record(directory/'training-summary.json'),
            'dev_final':read_record(directory/'dev-final.json'),'refs':read_record(directory/'refs.json'),
            'snapshot_directories':[p.name for p in sorted((directory/'snapshots').glob('step-*')) if p.is_dir()]}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud-root',type=Path,default=Path('/content/drive/MyDrive/VLN-Research/studies/arrival-evidence-20261003'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=inspect(args.cloud_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'output':str(args.output),'sha256':sha(args.output),
        'records_present':[k for k,v in report['records'].items() if v['exists']],
        'caches':{k:{field:v.get(field) for field in ('exists','pair_commit_markers_present','root_manifest_hash_verified')} for k,v in report['caches'].items()}}))


if __name__=='__main__':main()
