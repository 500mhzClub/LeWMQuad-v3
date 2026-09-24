"""Tiny-probe source/fixture tests; never creates services or induces OOM."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from scripts import tracking_kernel_scope_development as kernel
from scripts import run_go2_tracking_keeper_memory_probe_v1 as mod
from scripts import probe_go2_tracking_keeper_memory_development as child
from scripts import independent_tracking_memory_supervision_development as memory
from scripts import navigation_artifact_root_development as custody


def definition(mode):
    return dict(mode=mode,unit=kernel.PROBE_UNITS[mode],profile=kernel.profile(kernel.PROBE_UNITS[mode]),
        allocation_bytes=8*1024**2 if mode=='fit' else 128*1024**2,
        source_sha256={child.SOURCE:'a'*64,child.KERNEL_SOURCE:'b'*64})


def log_fixture(mode):
    d=definition(mode);unit=d['unit']
    group=f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}'
    controls={'memory.max':str(64*1024**2),'memory.swap.max':'0','memory.oom.group':'1','pids.max':'16'}
    def event(stage,**kw):return dict(schema='tracking_keeper_memory_probe_event.v1',mode=mode,stage=stage,**kw)
    events=[event('SCOPE_ADMITTED',role='parent',native_execution=False,
        scope=dict(unit=unit,pid=101,cgroup=group,controls=controls)),
        event('LEAF_STARTED',parent_pid=101,leaf_pid=102),
        event('SCOPE_ADMITTED',role='leaf',native_execution=False,
        scope=dict(unit=unit,pid=102,cgroup=group,controls=deepcopy(controls))),
        event('ALLOCATION_REQUEST',pid=102,bytes=d['allocation_bytes'])]
    if mode=='fit':events.extend([event('ALLOCATION_RETURNED',pid=102,bytes=d['allocation_bytes'],
        current_bytes=24*1024**2,peak_bytes=24*1024**2),event('PARENT_RETURNED',returncode=0)])
    return d,events


def encode(mode,events,manager=True):
    tail='Finished with result: oom-kill\n' if mode=='overflow' and manager else 'Finished with result: success\n'
    raw=('\n'.join(json.dumps(e) for e in events)+'\n'+tail).encode()
    evidence=dict(child_handle_terminal=True,log_complete=True,log_retained_bytes=len(raw),
        log_total_bytes=len(raw),log_omitted_bytes=0,systemd_run_returncode=0 if mode=='fit' else 1)
    return evidence,raw


@pytest.mark.parametrize('mode',['fit','overflow'])
def test_parent_leaf_and_manager_outcomes_are_distinct(mode):
    d,events=log_fixture(mode);evidence,raw=encode(mode,events)
    r=mod.inspect_log(d,evidence,raw)
    assert r['parent_pid']==101 and r['leaf_pid']==102
    assert r['manager_oom_classification_observed']==(mode=='overflow')
    assert not r['native_workload_fit_proved']


@pytest.mark.parametrize('fault',['no_parent','no_leaf','no_request','wrong_allocation','wrong_group',
    'unlimited','swap','no_group_kill','wrong_child','missing_manager','returned',
    'wrong_mode','zero_exit','omitted','out_of_order','foreign_user'])
def test_arbitrary_failure_or_import_oom_is_not_admitted_group_oom(fault):
    d,events=log_fixture('overflow')
    if fault=='no_parent':events.pop(0)
    elif fault=='no_leaf':events.pop(2)
    elif fault=='no_request':events.pop(3)
    elif fault=='wrong_allocation':events[3]['bytes']=8*1024**2
    elif fault=='wrong_group':events[2]['scope']['cgroup']+='/other'
    elif fault=='unlimited':events[2]['scope']['controls']['memory.max']='max'
    elif fault=='swap':events[2]['scope']['controls']['memory.swap.max']='max'
    elif fault=='no_group_kill':events[2]['scope']['controls']['memory.oom.group']='0'
    elif fault=='wrong_child':events[1]['leaf_pid']=103
    elif fault=='returned':events.append(dict(schema='tracking_keeper_memory_probe_event.v1',mode='overflow',stage='PARENT_RETURNED',returncode=1))
    elif fault=='wrong_mode':events[3]['mode']='fit'
    elif fault=='out_of_order':events[2],events[3]=events[3],events[2]
    elif fault=='foreign_user':
        for e in (events[0],events[2]):e['scope']['cgroup']=e['scope']['cgroup'].replace(f'user-{os.getuid()}.slice','user-99999.slice')
    evidence,raw=encode('overflow',events,manager=fault!='missing_manager')
    if fault=='zero_exit':evidence['systemd_run_returncode']=0
    elif fault=='omitted':evidence['log_complete']=False;evidence['log_omitted_bytes']=1
    with pytest.raises(ValueError):mod.inspect_log(d,evidence,raw)


@pytest.mark.parametrize('unit',[kernel.CHALLENGE_UNIT,*kernel.PROBE_UNITS.values()])
def test_shared_command_prefix_exact_profiles_no_old_probe_or_scope(unit):
    p=kernel.profile(unit);cmd=kernel.service_prefix(unit)
    assert '--property=MemoryMax='+str(p['memory_bytes']) in cmd
    assert '--property=TasksMax='+str(p['tasks']) in cmd
    assert '--property=RuntimeMaxSec='+str(p['runtime_seconds']) in cmd
    assert '--property=OOMPolicy=kill' in cmd and '--property=Restart=no' in cmd
    assert '--scope' not in cmd and '--remain-after-exit' not in cmd
    with pytest.raises(ValueError):kernel.service_prefix('lewm-tracking-memory-fit-20260907-v1.service')


def test_lightweight_shared_admission_and_child_import_without_ml_in_fresh_process():
    code="from scripts import tracking_kernel_scope_development; from scripts import probe_go2_tracking_keeper_memory_development; import sys; assert not any(x in sys.modules for x in ('numpy','torch','genesis')); print('LIGHTWEIGHT_IMPORT_VERIFIED')"
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=False,timeout=15)
    assert p.returncode==0 and p.stdout.strip()=='LIGHTWEIGHT_IMPORT_VERIFIED'


def test_ordinary_child_rejects_before_allocation_or_new_process(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('no allocation or child before actual scope admission')
    monkeypatch.setattr(child.subprocess,'Popen',forbidden)
    monkeypatch.setattr(child,'bytearray',forbidden,raising=False)
    with pytest.raises(ValueError,match='unbounded'):child.run('fit','parent','a'*64,'b'*64)
    with pytest.raises(ValueError,match='unbounded'):child.run('overflow','leaf','a'*64,'b'*64)


def test_parent_flushes_identity_before_one_shot_grant_and_wait(monkeypatch):
    order=[]
    class Pipe:
        def write(self,raw):
            assert order==['LEAF_STARTED'] and raw==child.GRANT
            order.append('grant');return len(raw)
        def close(self):order.append('close')
    def wait():
        assert order==['LEAF_STARTED','grant','close'];order.append('wait');return 0
    monkeypatch.setattr(child.subprocess,'Popen',lambda *a,**kw:NS(pid=123,stdin=Pipe(),wait=wait))
    monkeypatch.setattr(child,'emit',lambda stage,**kw:order.append(stage))
    assert child.run_parent('fit',['synthetic'])==0
    assert order==['LEAF_STARTED','grant','close','wait','PARENT_RETURNED']


@pytest.mark.parametrize('raw',[child.GRANT,b'',b'WRONG\n',b'x'*65])
def test_leaf_requires_exact_bounded_grant_before_allocation(monkeypatch,raw):
    monkeypatch.setattr(child.sys,'stdin',NS(buffer=io.BytesIO(raw)))
    if raw==child.GRANT:child.wait_for_allocation_grant()
    else:
        with pytest.raises(ValueError,match='grant'):child.wait_for_allocation_grant()


@pytest.fixture
def roots(tmp_path,monkeypatch):
    monkeypatch.setattr(custody,'BASE',tmp_path)
    outputs={m:tmp_path/f'go2_tiny_{m}_fixture_attempt_001' for m in kernel.PROBE_UNITS}
    monkeypatch.setattr(memory,'PROBE_OUTPUTS',outputs)
    monkeypatch.setattr(memory.shutil,'disk_usage',lambda _:NS(free=100*1024**3))
    monkeypatch.setattr(mod,'verify_sources',lambda _:None)
    return outputs


@pytest.mark.parametrize('mode',['fit','overflow'])
def test_new_probe_store_cannot_consume_challenge_root_and_completed_attempt_cannot_retry(roots,monkeypatch,mode):
    d,events=log_fixture(mode);evidence,raw=encode(mode,events);calls=[]
    def relay(command,store):
        calls.append(command);p=store.fresh_path('unit.log');p.write_bytes(raw)
        return evidence|{'log_sha256':memory.sha256(p)}
    monkeypatch.setattr(memory,'relay_process',relay)
    r=mod.execute(d,dict(pid=os.getpid(),cgroup='/synthetic'),None)
    assert r['status'].endswith('VERIFIED') and not r['native_execution']
    assert {p.name for p in roots[mode].iterdir()}=={'request.json','unit.log','terminal.json'}
    custody.verify_artifacts(roots[mode],r['output_sha256'])
    with pytest.raises(ValueError):mod.execute(d,{},None)
    assert len(calls)==1
    assert not memory.OUTPUT.exists() and not memory.INNER_OUTPUT.exists()


def test_failed_probe_retains_actual_log_and_never_calls_failure_oom(roots,monkeypatch):
    d,events=log_fixture('overflow');evidence,raw=encode('overflow',[],manager=False)
    def relay(command,store):
        p=store.fresh_path('unit.log');p.write_bytes(raw)
        return evidence|{'log_sha256':memory.sha256(p)}
    monkeypatch.setattr(memory,'relay_process',relay)
    with pytest.raises(ValueError):mod.execute(d,{},None)
    r=json.loads((roots['overflow']/'terminal.json').read_text())
    assert r['status']=='TINY_KEEPER_PROBE_FAILED' and r['child_handle_terminal']
    custody.verify_artifacts(roots['overflow'],r['output_sha256'])


def test_interrupted_probe_observation_is_not_workload_termination(roots,monkeypatch):
    def relay(*a):raise KeyboardInterrupt('synthetic interrupted observation')
    monkeypatch.setattr(memory,'relay_process',relay)
    with pytest.raises(KeyboardInterrupt):mod.execute(definition('fit'),{},None)
    r=json.loads((roots['fit']/'terminal.json').read_text())
    assert r['status']=='TINY_KEEPER_INTERRUPTED_CHILD_STATE_UNVERIFIED'
    assert not r['child_handle_terminal']


def test_overflow_requires_completed_same_source_fit(roots,monkeypatch):
    d,events=log_fixture('fit');evidence,raw=encode('fit',events)
    def relay(command,store):
        p=store.fresh_path('unit.log');p.write_bytes(raw)
        return evidence|{'log_sha256':memory.sha256(p)}
    monkeypatch.setattr(memory,'relay_process',relay)
    with pytest.raises(ValueError):mod.fit_receipt(definition('overflow'))
    mod.execute(d,{},None)
    assert mod.fit_receipt(definition('overflow'))
    wrong=definition('overflow');wrong['source_sha256'][child.SOURCE]='d'*64
    with pytest.raises(ValueError,match='same-source'):mod.fit_receipt(wrong)
