"""Exclusive dispatch and terminal failure handling; no native or actual replay."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts import replay_go2_extended_return_budget_controller_prefix_v1 as job


def setup(tmp_path,monkeypatch,fault=None):
    output=tmp_path/'attempt';calls=[];admissions=[];sources={'synthetic':'frozen'}
    admission={'frames':4014,'input':'same'}
    report=dict(frames=4004,budget_only_preboundary_decisions_supported=fault!='negative',
        boundary={'stop_reason':'FIRST_NORMALIZED_DECISION_DIFFERENCE'})
    for key,value in job.run.ENV.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(job.run.cv2.ocl,'useOpenCL',lambda:False)
    monkeypatch.setattr(job,'OUTPUT',output)
    monkeypatch.setattr(job.run,'validate_root',lambda *a,**k:None)
    monkeypatch.setattr(job.run,'hardware',lambda:{'memory_available_bytes':64*1024**3})
    def prepared(seeds):
        assert job.SOURCE in seeds and job.PROTOCOL in seeds
        calls.append('sources');return sources
    monkeypatch.setattr(job.inputs,'prepared_sources',prepared)
    def resources():
        calls.append('resources')
        if fault=='resources':raise ValueError('resources rejected')
        return job.run.hardware()
    monkeypatch.setattr(job,'resources',resources)
    def idle():
        calls.append('idle')
        if fault=='busy':raise ValueError('CPU busy')
    monkeypatch.setattr(job,'cpu_idle',idle)
    def admit(sha,source):
        calls.append('admit');admissions.append(sha);assert source==sources
        if fault=='admission':raise ValueError('original proof incomplete')
        return admission|({'changed':True} if fault=='reauthentication' and len(admissions)==2 else {})
    monkeypatch.setattr(job.inputs,'admit',admit)
    monkeypatch.setattr(job.pair,'require_admission',lambda receipt:calls.append('admission_shape'))
    monkeypatch.setattr(job.run,'create_output',lambda p:p.mkdir())
    def write(path,value):
        with path.open('x') as f:json.dump(value,f)
    monkeypatch.setattr(job.run,'write_json',write)
    monkeypatch.setattr(job.run,'digest',lambda p:hashlib.sha256(p.read_bytes()).hexdigest())
    def verify(root,ids):
        calls.append('verify_outputs')
        assert all(job.run.digest(root/name)==sha for name,sha in ids.items())
    monkeypatch.setattr(job.run,'verify_artifacts',verify)
    monkeypatch.setattr(job.run,'verify',lambda source:calls.append('verify_sources'))
    def replay(receipt,*,output):
        calls.append('replay');assert receipt==admission
        if fault=='replay':raise ValueError('model evidence changed')
        for name in (job.pair.pipeline.stream.NAME,'state_checks.json','identities.json','resource_monitor.jsonl'):
            (output/name).write_text('synthetic prefix evidence\n')
        return deepcopy(report)
    monkeypatch.setattr(job.pair,'replay',replay)
    def check(actual,receipt,*,output):
        calls.append('checker');assert actual==report and receipt==admission
        if fault=='checker':raise ValueError('saved evidence changed')
    monkeypatch.setattr(job.pair,'check_output',check)
    return output,calls,admissions,report


def test_source_preflight_does_not_admit_inputs_resources_or_create_output(tmp_path,monkeypatch):
    output,calls,admissions,_=setup(tmp_path,monkeypatch)
    job.main(source_only=True)
    assert calls==['sources'] and not admissions and not output.exists()


@pytest.mark.parametrize('fault',[None,'negative'])
def test_complete_positive_or_negative_comparison_binds_all_evidence(tmp_path,monkeypatch,fault):
    output,calls,admissions,report=setup(tmp_path,monkeypatch,fault)
    job.main('a'*64)
    result=json.loads((output/'result.json').read_text());launch=json.loads((output/'launch.json').read_text())
    assert result['status']=='EXTENDED_RETURN_BUDGET_CONTROLLER_PREFIX_V1_COMPLETE'
    assert result['report']==report and result['scientific_budget_only_prefix_supported'] is (fault is None)
    assert launch['planned_state_frames'][-2:]==[4002,4003]
    assert result['original_native_and_timing_inputs_reauthenticated_before_and_after']
    assert not result['native_execution'] and not result['goal_achieved']
    assert len(result['artifact_sha256'])==6 and admissions==['a'*64]*2
    assert calls==['sources','resources','idle','admit','admission_shape','resources','idle',
        'replay','checker','admit','verify_outputs','verify_sources']
    before=list(calls)
    with pytest.raises(ValueError,match='exclusive'):job.main('a'*64)
    assert calls==before


@pytest.mark.parametrize('fault',['resources','busy','admission'])
def test_prelaunch_rejection_creates_no_attempt(tmp_path,monkeypatch,fault):
    output,calls,_,_=setup(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError):job.main('a'*64)
    assert not output.exists() and 'replay' not in calls


@pytest.mark.parametrize('fault',['replay','checker','reauthentication'])
def test_terminal_execution_failure_is_preserved_and_cannot_retry(tmp_path,monkeypatch,fault):
    output,calls,_,_=setup(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError):job.main('a'*64)
    failure=json.loads((output/'failure.json').read_text())
    assert failure['status']=='TERMINAL_EXTENDED_RETURN_PREFIX_FAILURE' and not failure['automatic_retry']
    assert (output/'launch.json').exists() and not (output/'result.json').exists()
    with pytest.raises(ValueError,match='exclusive'):job.main('a'*64)


def test_missing_actual_completed_result_does_not_dispatch(tmp_path,monkeypatch):
    output,calls,_,_=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='actual completed'):job.main()
    assert calls==['sources'] and not output.exists()
