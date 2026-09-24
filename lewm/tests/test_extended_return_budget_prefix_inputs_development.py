"""Authenticated completed timing and native identity, without controller replay."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts import extended_return_budget_prefix_inputs_development as job
from lewm.tests.test_measured_plane_full_history_timing_development import population


def setup(tmp_path,monkeypatch):
    sources={'frozen_source':'b'*64};calls=[]
    admission=dict(frames=4014,learned_result_sha256=job.NATIVE_RESULT_SHA,
        model_state_sha256=job.MODEL_SHA,original_physical_prefix_reconstructed=True)
    launch=dict(source_sha256=sources,owner={'pid':job.TIMING_PID,'created':job.TIMING_CREATED},
        boot_id=job.native_inputs.BOOT_ID,input_admission=admission)
    def write(name,value):
        (tmp_path/name).write_text(json.dumps(value))
    def digest(name):return hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
    write('launch.json',launch);monkeypatch.setattr(job,'TIMING_LAUNCH_SHA',digest('launch.json'))
    rows,states=population(4014)
    report=job.timing.pair.comparison.summarize(rows,states,frames=4014,model_sha=job.MODEL_SHA,
        input_result_sha=job.NATIVE_RESULT_SHA)
    (tmp_path/'comparison.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write('state_checks.json',states);write('report.json',report)
    (tmp_path/'resource_monitor.jsonl').write_text('synthetic bound timing resource file\n')
    result=dict(status='MEASURED_PLANE_CHAINED_SINGLE_PASS_FULL_HISTORY_V1_COMPLETE',
        source_sha256=sources,artifact_sha256={n:digest(n) for n in job.TIMING_ARTIFACTS},report=report,
        complete_output_and_public_packets_rechecked=True,original_raw_inputs_reauthenticated_before_and_after=True,
        original_physical_prefix_reconstructed_before_and_after=True,native_execution=False,
        navigation_outcomes_inferred=False,navigation_qualified=False,real_time_qualified=False,
        hardware_qualified=False,goal_achieved=False)
    write('result.json',result)
    monkeypatch.setattr(job.timing,'OUTPUT',tmp_path)
    monkeypatch.setattr(job.run,'read_json',lambda root,name:json.loads((root/name).read_text()))
    monkeypatch.setattr(job.run,'owner_live',lambda owner:False)
    monkeypatch.setattr(job.run,'verify',lambda source:calls.append('sources'))
    def verify(root,ids):
        calls.append('artifacts')
        for name,sha in ids.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise ValueError('changed artifact')
    monkeypatch.setattr(job.run,'verify_artifacts',verify)
    def native(sha,source):
        calls.append('native_admission');assert sha==job.NATIVE_RESULT_SHA
        return deepcopy(admission)
    monkeypatch.setattr(job.native_inputs,'admit',native)
    return sources,admission,result,calls,write,digest


def test_complete_timing_accounting_and_same_native_admission_are_required(tmp_path,monkeypatch):
    sources,admission,result,calls,write,digest=setup(tmp_path,monkeypatch)
    sha=digest('result.json');actual=job.admit(sha,sources)
    assert actual['completed_chained_timing_result_sha256']==sha
    assert actual['complete_timing_accounting_reconstructed'] and actual['preceding_timing_owner_ended']
    assert not actual['preceding_timing_controller_execution_repeated']
    assert all(actual[k]==v for k,v in admission.items())
    assert calls.count('native_admission')==1 and calls[-1]=='artifacts'


@pytest.mark.parametrize('fault',['live','failure','status','scope','source_union','missing_artifact',
    'result_binding','report_copy','population','state','native_admission','result_sha'])
def test_incomplete_changed_or_live_timing_proof_cannot_dispatch(tmp_path,monkeypatch,fault):
    sources,admission,result,calls,write,digest=setup(tmp_path,monkeypatch)
    if fault=='live':monkeypatch.setattr(job.run,'owner_live',lambda owner:True)
    elif fault=='failure':write('failure.json',{'terminal':True})
    elif fault=='status':result['status']='RUNNING'
    elif fault=='scope':result['native_execution']=True
    elif fault=='source_union':sources=dict(sources,frozen_source='c'*64)
    elif fault=='missing_artifact':del result['artifact_sha256']['comparison.jsonl']
    elif fault=='result_binding':result['artifact_sha256']['comparison.jsonl']='c'*64
    elif fault=='report_copy':result['report']['frames']=4013
    elif fault=='population':
        p=tmp_path/'comparison.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
        result['artifact_sha256']['comparison.jsonl']=digest('comparison.jsonl')
    elif fault=='state':
        p=tmp_path/'state_checks.json';states=json.loads(p.read_text());states.pop();write('state_checks.json',states)
        result['artifact_sha256']['state_checks.json']=digest('state_checks.json')
    elif fault=='native_admission':
        monkeypatch.setattr(job.native_inputs,'admit',lambda *a:admission|{'changed':True})
    write('result.json',result);sha='not-a-sha' if fault=='result_sha' else digest('result.json')
    with pytest.raises(ValueError):job.admit(sha,sources)
    if fault in ('live','failure','result_sha'):assert 'native_admission' not in calls


def test_source_preparation_merges_both_bound_ancestries_without_admitting_results(tmp_path,monkeypatch):
    sources,_,_,calls,_,_=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(job.native_inputs,'native_launch',lambda:{'source_sha256':{'native':'d'*64}})
    def discover(seeds,prior):
        assert job.SOURCE in seeds and job.TEST in seeds and 'caller.py' in seeds
        assert prior==sources|{'native':'d'*64}
        return prior|{'caller.py':'e'*64}
    monkeypatch.setattr(job,'discover_sources',discover)
    result=job.prepared_sources(('caller.py',))
    assert result['caller.py']=='e'*64 and 'native_admission' not in calls
