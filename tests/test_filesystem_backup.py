"""Explicit durable-filesystem backend; no test bypass in real backend tests."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from vln_improve.checkpoint_store import BackupError, CheckpointStore
from vln_improve.pipeline import (MountedCheckpointStore, backup_mount_identity,
    run_pipeline, validate_backup_root, validate_config, validate_separate_roots)
from test_pipeline import durable_run, fixed_evaluator, restored_state, tensors_equal


@pytest.fixture
def mount_table(monkeypatch):
    """Mock only kernel mount reporting; exercise actual disk files/checkpoints."""
    content = {'text':'1 0 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n'}
    original_is_file, original_read_text = Path.is_file, Path.read_text
    def is_file(path):
        return True if str(path)=='/proc/self/mountinfo' else original_is_file(path)
    def read_text(path,*args,**kwargs):
        return content['text'] if str(path)=='/proc/self/mountinfo' else original_read_text(path,*args,**kwargs)
    monkeypatch.setattr(Path,'is_file',is_file)
    monkeypatch.setattr(Path,'read_text',read_text)
    return content


def test_default_drive_guard_does_not_accept_ext4_filesystem(tmp_path,mount_table):
    with pytest.raises(RuntimeError,match='Google Drive is not mounted'):
        validate_backup_root(tmp_path)
    assert validate_backup_root(tmp_path,backend='filesystem')=='filesystem-mount-readback-sha256'


def test_filesystem_root_must_preexist_no_implicit_creation(tmp_path,mount_table):
    path=tmp_path/'absent'
    with pytest.raises(RuntimeError,match='existing directory'):
        validate_backup_root(path,backend='filesystem')
    assert not path.exists()
    with pytest.raises(ValueError,match='test-only'):
        validate_backup_root(tmp_path,backend='filesystem',allow_local=True)


@pytest.mark.parametrize('kind',['tmpfs','ramfs','overlay','fuse.overlayfs','aufs','rootfs','squashfs','proc'])
def test_filesystem_refuses_ephemeral_mounts(tmp_path,mount_table,kind):
    mount_table['text']=f'1 0 0:1 / / rw - {kind} none rw\n'
    with pytest.raises(RuntimeError,match='persistent filesystem'):
        validate_backup_root(tmp_path,backend='filesystem')


@pytest.mark.parametrize('kind,source',[('ext4','/dev/sdb'),('9p','drvfs'),('drvfs','C:')])
def test_wsl_filesystem_types_and_mount_identity(tmp_path,mount_table,kind,source):
    mount_table['text']=f'1 0 8:2 / / rw - {kind} {source} rw\n'
    identity=backup_mount_identity(tmp_path,backend='filesystem')
    assert identity['filesystem']==kind and identity['source']==source and identity['device']=='8:2'
    assert identity['mount_point']=='/' and identity['mount_root']=='/'
    validate_backup_root(tmp_path,backend='filesystem',expected_identity=identity)


def test_deepest_mount_drop_to_parent_rejected(tmp_path,mount_table):
    escaped=str(tmp_path).replace(' ','\\040')
    mount_table['text']+=f'2 1 8:2 / {escaped} rw - ext4 /dev/sdb rw\n'
    identity=backup_mount_identity(tmp_path,backend='filesystem')
    assert identity['mount_point']==str(tmp_path)
    mount_table['text']='1 0 8:1 / / rw - ext4 /dev/sda1 rw\n'
    with pytest.raises(RuntimeError,match='mount identity changed'):
        validate_backup_root(tmp_path,backend='filesystem',expected_identity=identity)


def test_stacked_same_path_mounts_fail_closed(tmp_path,mount_table):
    mount_table['text']+='2 1 0:2 / / rw - tmpfs tmpfs rw\n'
    with pytest.raises(RuntimeError,match='stacked backup mounts'):
        validate_backup_root(tmp_path,backend='filesystem')


def test_same_device_remount_changes_process_mount_identity(tmp_path,mount_table):
    identity=backup_mount_identity(tmp_path,backend='filesystem')
    assert identity['mount_id']=='1'
    mount_table['text']='2 0 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n'
    with pytest.raises(RuntimeError,match='mount identity changed'):
        validate_backup_root(tmp_path,backend='filesystem',expected_identity=identity)


def test_mount_readonly_or_ram_block_source_rejected(tmp_path,mount_table):
    mount_table['text']='1 0 8:1 / / ro - ext4 /dev/sda1 ro\n'
    with pytest.raises(RuntimeError,match='read-only'):
        validate_backup_root(tmp_path,backend='filesystem')
    mount_table['text']='1 0 8:1 / / rw - ext4 /dev/zram0 rw\n'
    with pytest.raises(RuntimeError,match='persistent filesystem'):
        validate_backup_root(tmp_path,backend='filesystem')


@pytest.mark.parametrize('nested',['same','backup_inside_local','local_inside_backup'])
def test_nonoverlap_guard_before_store_creation(tmp_path,nested):
    local=tmp_path/'local'; backup=tmp_path/'backup'
    if nested=='same': backup=local
    elif nested=='backup_inside_local': backup=local/'backup'
    else: local=backup/'local'
    with pytest.raises(ValueError,match='non-nested'):
        validate_separate_roots(local,backup)
    with pytest.raises(ValueError,match='non-nested'):
        CheckpointStore(local,backup)
    assert not local.exists() and not backup.exists()


def test_probe_normalizes_tilde_before_validation_and_storage():
    path=Path(__file__).resolve().parents[1]/'scripts/probe_continuation_actions.py'
    spec=importlib.util.spec_from_file_location('filesystem_probe_cli',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args=SimpleNamespace(output_dir=Path('~/vln-local/example'),backup_dir=Path('~/vln-backups/example'))
    module.normalize_storage_paths(args)
    assert args.output_dir==(Path.home()/'vln-local/example').resolve()
    assert args.backup_dir==(Path.home()/'vln-backups/example').resolve()


def test_filesystem_pipeline_new_run_and_empty_local_restore(durable_run,mount_table):
    project, config=durable_run
    config['backup_backend']='filesystem'
    backup_root=Path(config['backup_root']); backup_root.mkdir()
    assert not (backup_root/config['run_id']).exists()
    reference=copy.deepcopy(config); reference['run_id']='continuous'
    run_pipeline(reference,project_root=project,evaluator=fixed_evaluator)
    expected,_,_=restored_state(reference)
    paused=run_pipeline(config,project_root=project,evaluator=fixed_evaluator,stop_after_steps=1)
    assert paused['backup_backend']=='filesystem' and paused['backup_mount_identity']['filesystem']=='ext4'
    shutil.rmtree(config['local_root'])
    shutil.rmtree(project/'cache')
    config['local_root']+='-restored'
    # A WSL reboot may assign a new block device. New invocation revalidates the
    # current mount, while persistent store/checkpoint identities remain fixed.
    mount_table['text']='1 0 8:9 / / rw,relatime - ext4 /dev/sdz rw\n'
    result=run_pipeline(config,project_root=project,evaluator=fixed_evaluator,require_resume=True)
    actual,_,_=restored_state(config)
    assert result['status']=='complete' and result['resumed']
    assert actual['identity']['backup_backend']=='filesystem'
    assert actual['backup_mount_identity']['device']=='8:9'
    for key in ('head','optimizer','history','epoch_totals','record_cursor','global_step'):
        tensors_equal(actual['trainer'][key],expected['trainer'][key])


def test_backend_change_or_invalid_backend_rejected(durable_run,mount_table):
    project,config=durable_run
    config['backup_backend']='filesystem'; Path(config['backup_root']).mkdir()
    run_pipeline(config,project_root=project,evaluator=fixed_evaluator,stop_after_steps=1)
    config['backup_backend']='drive'
    with pytest.raises(ValueError,match='changed'):
        run_pipeline(config,project_root=project,evaluator=fixed_evaluator,require_resume=True,allow_local_backup=True)
    config['backup_backend']='guess'
    with pytest.raises(ValueError,match='backup_backend'):
        validate_config(config)


def test_mount_loss_after_copy_preserves_local_snapshots_and_backup_pointer(tmp_path,mount_table,monkeypatch):
    root=tmp_path/'backup'; root.mkdir()
    identity=backup_mount_identity(root,backend='filesystem')
    check=lambda:validate_backup_root(root,backend='filesystem',expected_identity=identity)
    store=MountedCheckpointStore(tmp_path/'local',root/'run',keep_local=1,keep_backup=1,backup_check=check)
    first=store.save({'step':1},{'head':1},step=1,is_best=True)
    old_pointer=(store.backup_dir/'latest.json').read_bytes()
    original=CheckpointStore._copy_snapshot
    def lose_mount(self,source,target):
        value=original(self,source,target)
        mount_table['text']='1 0 8:2 / / rw - ext4 /dev/replacement rw\n'
        return value
    monkeypatch.setattr(CheckpointStore,'_copy_snapshot',lose_mount)
    with pytest.raises(BackupError,match='mount identity changed'):
        store.save({'step':2},{'head':2},step=2,is_best=True)
    local_ids={p.name for p in (store.local_dir/'snapshots').glob('step-*')}
    assert len(local_ids)==2 and first in local_ids
    assert (store.backup_dir/'latest.json').read_bytes()==old_pointer


def test_regular_backup_write_failure_keeps_unbacked_local_snapshot(tmp_path,mount_table,monkeypatch):
    root=tmp_path/'backup'; root.mkdir()
    identity=backup_mount_identity(root,backend='filesystem')
    store=MountedCheckpointStore(tmp_path/'local',root/'run',keep_local=1,keep_backup=1,
        backup_check=lambda:validate_backup_root(root,backend='filesystem',expected_identity=identity))
    first=store.save({'step':1},{'head':1},step=1,is_best=True)
    old_pointer=(store.backup_dir/'latest.json').read_bytes()
    def fail(*args): raise OSError('filesystem write failed')
    monkeypatch.setattr(shutil,'copyfile',fail)
    with pytest.raises(BackupError,match='local checkpoint retained'):
        store.save({'step':2},{'head':2},step=2)
    assert len(list((store.local_dir/'snapshots').glob('step-*')))==2
    assert (store.backup_dir/'latest.json').read_bytes()==old_pointer
    assert json.loads(old_pointer)['checkpoint_id']==first
