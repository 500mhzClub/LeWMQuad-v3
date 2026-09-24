"""Atomic replacements retain artifact bytes and preserve interrupted evidence."""
import hashlib
import os
from copy import deepcopy
import pytest
from types import SimpleNamespace
from scripts import consolidate_go2_training_artifact_duplicates_v1 as subject
from scripts.inventory_go2_training_artifact_duplicates_v1 import duplicate_groups,metadata


def fixture(tmp_path):
    rows=[]
    for name in ('a','b','c'):
        path=tmp_path/name;path.write_bytes(b'original training artifact\x00'*100)
        rows.append(dict(root='synthetic',path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metadata=metadata(path)))
    group=duplicate_groups(rows)[0]
    return group,lambda row:tmp_path/row['path']


def test_consolidation_keeps_every_path_and_byte_with_complete_journal(tmp_path):
    group,resolve=fixture(tmp_path);before=deepcopy(group);events=[]
    subject.consolidate_group(group,events.append,sequence=0,resolver=resolve)
    paths=[resolve(r) for r in [group['canonical'],*group['duplicate_paths']]]
    assert all(p.read_bytes()==paths[0].read_bytes() for p in paths)
    assert len({p.stat().st_ino for p in paths})==1
    assert all(p.stat().st_nlink==3 for p in paths)
    assert [e['stage'] for e in events]==['intent','complete','intent','complete']
    assert group==before and sorted(p.name for p in tmp_path.iterdir())==['a','b','c']


@pytest.mark.parametrize('fault',['source_bytes','target_bytes','mode','symlink','extra_link','temporary_exists'])
def test_changed_reviewed_inputs_reject_before_replacement(tmp_path,fault):
    group,resolve=fixture(tmp_path);source=resolve(group['canonical']);target=resolve(group['duplicate_paths'][0])
    if fault=='source_bytes':source.write_bytes(b'changed')
    elif fault=='target_bytes':target.write_bytes(b'changed')
    elif fault=='mode':target.chmod(0o600)
    elif fault=='symlink':target.unlink();target.symlink_to(source)
    elif fault=='extra_link':os.link(target,tmp_path/'external')
    else:(tmp_path/'.lewm-consolidate-v1-0-0').write_bytes(b'unrelated')
    events=[];before=target.lstat().st_ino
    with pytest.raises(ValueError):subject.consolidate_group(group,events.append,sequence=0,resolver=resolve)
    assert target.lstat().st_ino==before and events==[]


def test_failure_after_link_keeps_original_target_and_intent_for_recovery(tmp_path,monkeypatch):
    group,resolve=fixture(tmp_path);target=resolve(group['duplicate_paths'][0]);before=target.stat().st_ino;events=[]
    def fail(*args):raise OSError('synthetic interruption before replacement')
    monkeypatch.setattr(subject.os,'replace',fail)
    with pytest.raises(OSError):subject.consolidate_group(group,events.append,sequence=0,resolver=resolve)
    assert target.stat().st_ino==before and [e['stage'] for e in events]==['intent']
    assert (tmp_path/'.lewm-consolidate-v1-0-0').is_file()


def test_target_change_during_link_rejects_without_overwriting_it(tmp_path,monkeypatch):
    group,resolve=fixture(tmp_path);target=resolve(group['duplicate_paths'][0]);original=subject.os.link;events=[]
    def changed(*args,**kwargs):
        original(*args,**kwargs);target.write_bytes(b'concurrent external edit')
    monkeypatch.setattr(subject.os,'link',changed)
    with pytest.raises(ValueError):subject.consolidate_group(group,events.append,sequence=0,resolver=resolve)
    assert target.read_bytes()==b'concurrent external edit' and [e['stage'] for e in events]==['intent']


def test_execute_requires_explicit_authorization_before_admission(monkeypatch):
    monkeypatch.setattr('sys.argv',['consolidate'])
    monkeypatch.setattr(subject,'digest',lambda p:'a'*64)
    monkeypatch.setattr(subject,'verify_sources',lambda p:None)
    monkeypatch.setattr(subject,'admit',lambda:pytest.fail('admission must not imply approval'))
    with pytest.raises(ValueError,match='explicit reviewed user authorization'):subject.main()


@pytest.mark.parametrize('kind',['development','open_artifact','unidentified','protected_service'])
def test_quiescence_rejects_workers_but_reports_unrelated_descriptor_gaps(monkeypatch,kind):
    class Process:
        pid=99999999
        def uids(self):return SimpleNamespace(real=os.getuid())
        def cmdline(self):
            if kind=='unidentified':raise subject.psutil.AccessDenied(self.pid)
            return ['/env/genesis_rocm_0_4_6_v1/bin/python','-'] if kind=='development' else ['/usr/lib/systemd/systemd','--user']
        def open_files(self):
            if kind in ('development','protected_service'):raise subject.psutil.AccessDenied(self.pid)
            return [SimpleNamespace(path=str(subject.BASE/next(iter(subject.INPUTS))/'some.npz'))]
        def memory_maps(self,**kwargs):return []
    monkeypatch.setattr(subject.psutil,'process_iter',lambda:[Process()])
    if kind=='protected_service':
        result=subject.require_quiet_consumers()
        assert len(result['inaccessible_owned_descriptors'])==1 and not result['global_process_quiescence_proven']
    else:
        with pytest.raises(ValueError):subject.require_quiet_consumers()
