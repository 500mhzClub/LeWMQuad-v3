"""Verify whole-history review and process/admission boundaries before launch."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import run_go2_body_projection_external_profile_v1 as run
from scripts import verify_go2_body_projection_external_profile_completion_v1 as check
from lewm.tests import test_external_body_projection_profile_replay_development as fixtures


@pytest.fixture
def completed_replay(monkeypatch,tmp_path):
    _,_,prior_rows,prior,_=fixtures.fixture(monkeypatch,tmp_path)
    report=run.replay.replay(prior_rows,prior)
    rows=[json.loads(line) for line in (tmp_path/'comparison.jsonl').read_text().splitlines()]
    return report,rows,prior_rows,prior


def test_entire_actual_replay_report_and_rows_review(completed_replay):
    run.validate_report(*completed_replay)


@pytest.mark.parametrize('fault', ['state','model','scope','window','marker','timing','row_timing',
    'input','decision','missing_row','forecast_count','new_claim','hold'])
def test_changed_complete_replay_cannot_pass(completed_replay,fault):
    report,rows,prior_rows,prior=deepcopy(completed_replay)
    if fault=='state':report['observed_state_checks'][-1]['state_sha256']='bad'
    elif fault=='model':report['model_state_sha256']='bad'
    elif fault=='scope':report['real_time_qualified']=True
    elif fault=='window':rows[3]['profiled_window']='late_navigation'
    elif fault=='marker':report['windows']['early_navigation']['python_stack_markers'].pop()
    elif fault=='timing':report['windows']['early_navigation']['observations'][0]['controller_wall_s_with_profiling']+=1
    elif fault=='row_timing':rows[1000]['controller_wall_s']=float('nan')
    elif fault=='input':rows[1000]['public_input_sha256']='bad'
    elif fault=='decision':rows[-1]['candidate_decision_sha256']='bad'
    elif fault=='missing_row':rows.pop()
    elif fault=='forecast_count':report['raw_model_forecast_comparisons']=1424
    elif fault=='new_claim':report['new_claim']=True
    elif fault=='hold':report['windows']['repeated_hold']['observations'][0]['action']='forward'
    with pytest.raises(ValueError):run.validate_report(report,rows,prior_rows,prior)


def test_existing_attempt_rejects_before_sources_or_processes(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'OUTPUT',tmp_path)
    monkeypatch.setattr(run,'validate_root',lambda *a,**kw:None)
    monkeypatch.setattr(run,'sources_and_resources',lambda:pytest.fail('exclusive gate comes first'))
    with pytest.raises(ValueError,match='exclusive'):run.parent()


def test_source_preflight_never_creates_output_or_child(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'OUTPUT',tmp_path/'absent')
    monkeypatch.setattr(run,'validate_root',lambda *a,**kw:None)
    monkeypatch.setattr(run,'sources_and_resources',lambda:({'source':'a'*64},{'synthetic':True}))
    monkeypatch.setattr(run,'create_output',lambda *a:pytest.fail('preflight must not create'))
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**kw:pytest.fail('preflight must not launch'))
    run.parent(True)
    assert not run.OUTPUT.exists()


def test_child_rejects_wrong_parent_before_raw_admission(monkeypatch):
    monkeypatch.setattr(run,'authenticate_launch',lambda sha:{'owner':{'pid':-1}})
    monkeypatch.setattr(run.admission,'admit_completed',lambda:pytest.fail('ownership before raw/model reads'))
    with pytest.raises(ValueError,match='owning parent'):run.child('a'*64)


def test_wait_ignores_partial_ready_lines_and_collects_complete_line(monkeypatch,tmp_path):
    path=tmp_path/'out.txt';path.write_text('READY incomplete')
    process=SimpleNamespace(poll=lambda:None)
    calls=[]
    def sleep(value):
        calls.append(value);path.write_text('READY complete\n')
    monkeypatch.setattr(run.time,'sleep',sleep)
    assert run.wait_for_line(path,'READY ',process)=='READY complete'
    assert calls==[.2]


def test_wait_rejects_ended_process_without_readiness(tmp_path):
    path=tmp_path/'out.txt';path.write_text('incomplete')
    with pytest.raises(RuntimeError,match='ended before readiness'):
        run.wait_for_line(path,'READY ',SimpleNamespace(poll=lambda:1))


def test_final_checker_rejects_live_parent_before_actual_admission(monkeypatch,tmp_path):
    monkeypatch.setattr(check,'OUTPUT',tmp_path/'absent')
    monkeypatch.setattr(run,'authenticate_launch',lambda sha:{'owner':{}})
    monkeypatch.setattr(run,'owner_live',lambda o:True)
    monkeypatch.setattr(check,'verify_artifacts',lambda *a:pytest.fail('owner comes first'))
    monkeypatch.setattr(sys,'argv',['check','--result-sha256','a'*64,'--launch-sha256','b'*64])
    with pytest.raises(ValueError,match='parent must be ended'):check.main()


@pytest.mark.parametrize('pid',[True,0,-1,'123'])
def test_profiler_command_requires_original_integer_child(pid):
    with pytest.raises(ValueError):run.profiler_command(pid)


def test_profiler_command_has_only_exact_target_and_filters():
    command=run.profiler_command(123)
    assert command[-2:]==['--pid','123']
    assert '--gil' not in command and '--nonblocking' not in command and '--subprocesses' not in command
    assert '--idle' in command and command[command.index('--rate')+1]=='100'
