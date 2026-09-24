"""Containment admission and outside evidence tests; no native simulation or units.

Two tiny subprocess tests exercise the real pipe/exit path, including SIGKILL.
They do not induce OOM, instantiate systemd units or establish native memory fit.
"""
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from scripts import independent_tracking_memory_supervision_development as mod
from scripts import supervise_go2_independent_tracking_challenge_v1 as keeper
from scripts import navigation_artifact_root_development as custody


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    group=f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{mod.UNIT}'
    values={'/proc/self/cgroup':'0::'+group+'\n','/proc/self/oom_score_adj':'0'}
    for name,value in {'memory.max':str(mod.MEMORY_BYTES),'memory.swap.max':'0',
            'memory.oom.group':'1','pids.max':str(mod.TASKS)}.items():
        values['/sys/fs/cgroup'+group+'/'+name]=value
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**kw:values[str(p)] if str(p) in values else original(p,*a,**kw))
    return group,values


@pytest.mark.parametrize('fault',[None,'outside','child_group','foreign_user','nonunified',
    'memory_unlimited','memory_changed','swap','oom_group','tasks','oom_exempt'])
def test_effective_kernel_scope_before_native_work(kernel,fault):
    group,values=kernel
    if fault=='outside':values['/proc/self/cgroup']='0::/user.slice/old-collector.scope\n'
    elif fault=='child_group':values['/proc/self/cgroup']='0::'+group+'/unreviewed\n'
    elif fault=='foreign_user':values['/proc/self/cgroup']='0::'+group.replace(f'user-{os.getuid()}.slice','user-99999.slice')
    elif fault=='nonunified':values['/proc/self/cgroup']='1:memory:'+group
    elif fault=='memory_unlimited':values['/sys/fs/cgroup'+group+'/memory.max']='max'
    elif fault=='memory_changed':values['/sys/fs/cgroup'+group+'/memory.max']=str(mod.MEMORY_BYTES+1)
    elif fault=='swap':values['/sys/fs/cgroup'+group+'/memory.swap.max']='max'
    elif fault=='oom_group':values['/sys/fs/cgroup'+group+'/memory.oom.group']='0'
    elif fault=='tasks':values['/sys/fs/cgroup'+group+'/pids.max']='16'
    elif fault=='oom_exempt':values['/proc/self/oom_score_adj']='-1000'
    if fault:
        with pytest.raises(ValueError):mod.own_scope()
    else:
        r=mod.own_scope()
        assert r['cgroup']==group and r['controls']['memory.max']==str(8*1024**3)
        with pytest.raises(ValueError,match='outside'):mod.outside_scope()


@pytest.mark.parametrize('text',['0::relative','0::/a/../b','0::/a//b','0::/a\n1:memory:/b','0::/a\n\n0::/b'])
def test_noncanonical_or_multiple_groups_rejected(text):
    with pytest.raises(ValueError):mod.unified_group(text)


@pytest.mark.parametrize('state,code,valid',[
    ('not-found\n',0,True),('not-found\n',1,True),('loaded\n',0,False),
    ('error\n',1,False),('',1,False),('not-found\n',4,False)])
def test_fresh_unit_check_cannot_replace_existing_or_assume_absence(monkeypatch,state,code,valid):
    calls=[]
    def run(command,**kw):
        calls.append((command,kw));return NS(returncode=code,stdout=state)
    monkeypatch.setattr(mod.subprocess,'run',run)
    if valid:mod.require_fresh_unit()
    else:
        with pytest.raises(ValueError):mod.require_fresh_unit()
    assert calls[0][0]==['/usr/bin/systemctl','--user','show',mod.UNIT,'--property=LoadState','--value']
    assert calls[0][1]['timeout']==10


def test_fixed_service_command_has_no_unbounded_fallback_and_covers_whole_parent():
    c=keeper.challenge
    command=mod.service_command(c.PYTHON,c.SOURCE,c.ENVIRONMENT,'a'*64,'b'*64,'c'*64)
    assert command[0]=='/usr/bin/systemd-run'
    for option in ('--user','--wait','--pipe','--collect','--service-type=exec',
        '--property=MemoryMax=8589934592','--property=MemorySwapMax=0',
        '--property=OOMPolicy=kill','--property=TasksMax=512',
        '--property=KillMode=control-group','--property=Restart=no','--scoped-parent'):
        assert option in command
    assert '--scope' not in command and '--remain-after-exit' not in command
    assert command[-6:]==['--definition-sha256','a'*64,'--study-result-sha256','b'*64,
        '--supervisor-request-sha256','c'*64]
    assert mod.contract()['parent_replay_scoring_and_native_children_in_same_scope']
    assert not mod.contract()['workload_fit_proved']
    with pytest.raises(ValueError):mod.service_command(c.PYTHON,'other.py',c.ENVIRONMENT,'a'*64,'b'*64,'c'*64)
    with pytest.raises(ValueError):mod.service_command(c.PYTHON,c.SOURCE,c.ENVIRONMENT,'x','b'*64,'c'*64)


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(custody,'BASE',tmp_path)
    root=tmp_path/'go2_outside_fixture_attempt_001'
    monkeypatch.setattr(mod,'OUTPUT',root)
    monkeypatch.setattr(mod.shutil,'disk_usage',lambda _:NS(free=100*1024**3))
    return root


def test_exclusive_bounded_evidence_roster_and_reserve(evidence,monkeypatch):
    store=mod.EvidenceStore();h=store.save('request.json',{'synthetic':True})
    custody.verify_artifacts(evidence,{'request.json':h})
    with pytest.raises(ValueError):store.save('request.json',{})
    with pytest.raises(ValueError):store.fresh_path('../outside')
    with pytest.raises(ValueError):mod.EvidenceStore()
    monkeypatch.setattr(mod,'FILE_BYTES',10)
    with pytest.raises(ValueError,match='bounded'):store.save('terminal.json',{'oversized':'record'})
    monkeypatch.setattr(mod.shutil,'disk_usage',lambda _:NS(free=0))
    with pytest.raises(ValueError,match='reserve'):store.fresh_path('unit.log')
    assert not (evidence/'unit.log').exists() and not (evidence/'terminal.json').exists()


@pytest.mark.parametrize('killed',[False,True])
def test_real_pipe_exit_evidence_survives_child_sigkill(evidence,killed):
    store=mod.EvidenceStore()
    code="import os,signal; print('synthetic child evidence',flush=True)"
    if killed:code+='; os.kill(os.getpid(),signal.SIGKILL)'
    r=mod.relay_process([sys.executable,'-c',code],store)
    assert r['systemd_run_returncode']==(-signal.SIGKILL if killed else 0)
    assert r['child_handle_terminal'] and r['log_complete'] and r['log_omitted_bytes']==0
    assert (evidence/'unit.log').read_bytes()==b'synthetic child evidence\n'
    assert not r['failure_cause_inferred_from_exit_code']
    store.save('terminal.json',r)
    assert json.loads((evidence/'terminal.json').read_text())==r


def test_bounded_log_preserves_terminal_tail_but_never_claims_complete(evidence,monkeypatch):
    monkeypatch.setattr(mod,'FILE_BYTES',32)
    payload=b'a'*16+b'b'*100+b'TERMINAL_EVIDENCE'
    process=NS(stdout=io.BytesIO(payload),wait=lambda:1)
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:process)
    r=mod.relay_process(['synthetic'],mod.EvidenceStore())
    assert (evidence/'unit.log').read_bytes()==payload[:16]+payload[-16:]
    assert not r['log_complete'] and r['log_retained_bytes']==32 and r['log_omitted_bytes']==len(payload)-32


def test_read_interruption_does_not_wait_or_claim_child_termination(evidence,monkeypatch):
    class Broken:
        def read(self,n):raise KeyboardInterrupt('synthetic outside interruption')
        def close(self):pass
    def no_wait():raise AssertionError('must not wait on interrupted observation')
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:NS(stdout=Broken(),wait=no_wait))
    with pytest.raises(KeyboardInterrupt):mod.relay_process(['synthetic'],mod.EvidenceStore())
    assert (evidence/'unit.log').is_file()


@pytest.mark.parametrize('fault',[None,'definition','study','scope','terminal','changed_request','missing_supervisor'])
def test_inside_admission_requires_bound_request_and_live_outside_keeper(evidence,monkeypatch,fault):
    store=mod.EvidenceStore();row=mod.request('a'*64,'b'*64)
    if fault=='definition':row['definition_sha256']='d'*64
    elif fault=='study':row['study_result_sha256']='d'*64
    elif fault=='scope':row['scope_contract']['memory_max_bytes']+=1
    elif fault=='missing_supervisor':row['supervisor']['pid']=999999999
    h=store.save('request.json',row)
    if fault=='terminal':store.save('terminal.json',{})
    if fault=='changed_request':h='0'*64
    monkeypatch.setattr(mod,'own_scope',lambda:dict(synthetic=True))
    if fault:
        with pytest.raises((ValueError,FileNotFoundError)):mod.admit_request(h,'a'*64,'b'*64)
    else:assert mod.admit_request(h,'a'*64,'b'*64)['request_sha256']==h


@pytest.fixture
def supervisor(evidence,monkeypatch):
    c=keeper.challenge
    inner=evidence.parent/'go2_inside_fixture_attempt_001'
    monkeypatch.setattr(c,'OUTPUT',inner);monkeypatch.setattr(mod,'INNER_OUTPUT',inner)
    monkeypatch.setattr(mod,'require_fresh_unit',lambda:None)
    d={'synthetic':True};completed={'study_result_sha256':'b'*64}
    monkeypatch.setattr(c,'preflight',lambda *a:(d,completed))
    monkeypatch.setattr(c,'verify_ordered_launch',lambda _:None)
    calls=[]
    def relay(command,store):
        calls.append(command)
        req=json.loads((evidence/'request.json').read_text())
        request_sha=mod.sha256(evidence/'request.json')
        inner.mkdir()
        launch=dict(definition=d,completed_learning=completed,memory_supervision=dict(request_sha256=request_sha))
        (inner/'launch.json').write_text(json.dumps(launch))
        result=dict(status='NATIVE_TRACKING_CHALLENGE_COLLECTION_AND_EVALUATIONS_COMPLETE',
            definition_sha256='a'*64,output_sha256={'launch.json':mod.sha256(inner/'launch.json')},
            full_challenge_pass=False,independent_result_verification_complete=False,
            navigation_qualified=False,real_time_qualified=False,goal_achieved=False)
        (inner/'challenge_result.json').write_text(json.dumps(result))
        (evidence/'unit.log').write_bytes(b'synthetic exit evidence\n')
        return dict(systemd_run_returncode=0,child_handle_terminal=True,log_total_bytes=24,
            log_retained_bytes=24,log_omitted_bytes=0,log_complete=True,
            log_sha256=mod.sha256(evidence/'unit.log'),failure_cause_inferred_from_exit_code=False)
    monkeypatch.setattr(mod,'relay_process',relay)
    return evidence,inner,calls,relay


@pytest.mark.parametrize('fault',[None,'nonzero','truncated_log','missing_result','changed_launch','unbound_launch','promoted',
    'failure_file','interrupted','spawn_failure'])
def test_keeper_requires_actual_exit_and_bound_complete_result_and_never_retries(supervisor,monkeypatch,fault):
    root,inner,calls,original=supervisor
    def relay(command,store):
        if fault in ('interrupted','spawn_failure'):
            calls.append(command)
            if fault=='interrupted':raise KeyboardInterrupt('synthetic interrupted observation')
            raise OSError('synthetic failed to create process')
        r=original(command,store)
        if fault=='nonzero':r['systemd_run_returncode']=1
        elif fault=='truncated_log':r['log_complete']=False;r['log_omitted_bytes']=100
        elif fault=='missing_result':(inner/'challenge_result.json').unlink()
        elif fault=='changed_launch':(inner/'launch.json').write_text('{}')
        elif fault=='unbound_launch':
            p=inner/'challenge_result.json';v=json.loads(p.read_text());v['output_sha256']={};p.write_text(json.dumps(v))
        elif fault=='promoted':
            p=inner/'challenge_result.json';v=json.loads(p.read_text());v['goal_achieved']=True;p.write_text(json.dumps(v))
        elif fault=='failure_file':(inner/'failure.json').write_text('{}')
        return r
    monkeypatch.setattr(mod,'relay_process',relay)
    if fault:
        with pytest.raises((ValueError,KeyboardInterrupt,OSError)):keeper.supervise('a'*64,'b'*64)
    else:
        r=keeper.supervise('a'*64,'b'*64)
        assert r['status']=='SCOPED_CHALLENGE_COLLECTION_AND_EVALUATIONS_COMPLETE'
        assert not r['goal_achieved']
    terminal=json.loads((root/'terminal.json').read_text())
    assert len(calls)==1 and not terminal['retry_performed']
    if fault in ('interrupted','spawn_failure'):
        assert terminal['status']=='SUPERVISOR_INTERRUPTED_CHILD_STATE_UNVERIFIED'
        assert not terminal['child_handle_terminal']
    elif fault:
        assert terminal['status']=='SCOPED_COMMAND_FAILED' and terminal['child_handle_terminal']
    with pytest.raises(ValueError):keeper.supervise('a'*64,'b'*64)
    assert len(calls)==1


def test_missing_resource_review_stops_keeper_before_output_or_service(evidence,monkeypatch):
    def missing(*a):raise ValueError('native resource review not complete')
    monkeypatch.setattr(keeper.challenge,'preflight',missing)
    def forbidden():raise AssertionError('must not even inspect unit after failed scientific preflight')
    monkeypatch.setattr(mod,'require_fresh_unit',forbidden)
    with pytest.raises(ValueError,match='review'):keeper.supervise('a'*64,'b'*64)
    assert not evidence.exists()


def test_terminal_evidence_uses_remaining_file_reservation_not_whole_three_file_budget(evidence,monkeypatch):
    store=mod.EvidenceStore()
    monkeypatch.setattr(mod.shutil,'disk_usage',lambda _:NS(free=mod.RESERVE_BYTES+mod.FILE_BYTES))
    assert store.save('terminal.json',dict(synthetic=True))


def test_parent_and_worker_reject_unbounded_direct_calls_before_any_output(evidence,monkeypatch):
    c=keeper.challenge
    monkeypatch.setattr(c,'OUTPUT',evidence)
    def reject():raise ValueError('unbounded execution forbidden')
    monkeypatch.setattr(mod,'own_scope',reject)
    with pytest.raises(ValueError,match='unbounded'):c.execute({}, {}, {})
    with pytest.raises(ValueError,match='unbounded'):c.worker(c.TRIALS[0],'a'*64,'b'*64)
    assert not evidence.exists()
