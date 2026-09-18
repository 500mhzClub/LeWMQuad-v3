"""Admission orchestration over verified synthetic native and replay receipts."""
from copy import deepcopy

import pytest

from scripts import extended_return_budget_native_inputs_development as job


def setup(tmp_path,monkeypatch):
    root=tmp_path/'prefix';root.mkdir();native_root=tmp_path/'native';native_root.mkdir()
    sources={'native.py':'b'*64,job.replay.SOURCE:'c'*64,job.replay.PROTOCOL:'d'*64};calls=[]
    name=job.native.CASE[0];old_sha=job.replay.inputs.NATIVE_RESULT_SHA
    outcomes=dict(verified_round_trip=False,native_evaluation={'native_round_trip_candidate_pass':False},
        strict_physical_visibility_pass=True,hard_measurement_failed_frames=[],renderer_capture_audit={'frames':4014})
    record=dict(status=job.native.WORKER_STATUS,collection={'decisions':4014,'schedule_terminal':'MISSION_TICK_BUDGET_EXHAUSTED'},
        model_state_sha256=job.replay.pair.MODEL_SHA,**outcomes)
    audit=outcomes|dict(raw_sensor_reconstruction_pass=True,raw_command_audit_pass=True,
        raw_model_command_replay_pass=True,model_state_unchanged=True)
    native_ids={name+'/context_decisions.jsonl.gz':'e'*64,name+'_worker_terminal.json':'f'*64,name+'_audit.json':'1'*64}
    admission=dict(completed_chained_timing_result_sha256='2'*64,original_context_sha256='e'*64,
        complete_raw_artifact_roster_sha256=job.run.fingerprint(native_ids|{'result.json':old_sha}))
    launch=dict(owner={'pid':1},boot_id=job.run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        source_sha256=sources,protocol=job.replay.PROTOCOL,baseline_navigation_ticks=4000,candidate_navigation_ticks=8000,
        maximum_prefix_observations=4004,planned_state_frames=job.replay.pair.expected_states(4004),
        stop_at_first_normalized_decision_difference=True,preceding_timing_and_native_owners_ended=True,
        cpu_replay_serialization_checked=True,native_execution=False,automatic_retry=False,input_admission=admission)
    report={'eligible':'synthetic complete checked prefix'}
    result=dict(status='EXTENDED_RETURN_BUDGET_CONTROLLER_PREFIX_V1_COMPLETE',source_sha256=sources,
        artifact_sha256={n:'3'*64 for n in job.PREFIX_ARTIFACTS},report=report,
        original_native_result_sha256=old_sha,completed_chained_timing_result_sha256='2'*64,
        complete_output_and_public_packets_rechecked=True,original_native_and_timing_inputs_reauthenticated_before_and_after=True,
        original_physical_prefix_reconstructed_before_and_after=True,scientific_budget_only_prefix_supported=True,
        changed_command_executed=False,native_execution=False,automatic_retry=False,navigation_qualified=False,
        real_time_qualified=False,hardware_qualified=False,goal_achieved=False)
    saved={(root,'launch.json'):launch,(root,'result.json'):result,(root,'report.json'):deepcopy(report),
        (native_root,'result.json'):{'artifact_sha256':native_ids},
        (native_root,name+'_worker_terminal.json'):deepcopy(record),(native_root,name+'_audit.json'):audit}
    monkeypatch.setattr(job.replay,'OUTPUT',root);monkeypatch.setattr(job.native,'OUTPUT',native_root)
    monkeypatch.setattr(job.run,'read_json',lambda root,name:deepcopy(saved[(root,name)]))
    monkeypatch.setattr(job.run,'digest',lambda p:'3'*64)
    monkeypatch.setattr(job.run,'owner_live',lambda owner:False)
    monkeypatch.setattr(job.run,'verify_artifacts',lambda root,ids:calls.append(('artifacts',root,deepcopy(ids))))
    monkeypatch.setattr(job.run,'verify',lambda source:calls.append(('sources',deepcopy(source))))
    monkeypatch.setattr(job.old_inputs,'native_launch',lambda:{'owner':{'pid':2},'source_sha256':{'native.py':'b'*64}})
    monkeypatch.setattr(job.old_inputs,'worker_live',lambda:False)
    monkeypatch.setattr(job.old_inputs,'require_result',lambda *a:deepcopy(record))
    def boundary(actual):
        assert actual==report
        return {'frames':4004,'intervention':4003,'physics_samples':200900}
    monkeypatch.setattr(job.prefix,'boundary',boundary)
    monkeypatch.setattr(job.replay.pair,'require_admission',lambda actual:calls.append(('admission_shape',deepcopy(actual))))
    monkeypatch.setattr(job.replay.pair,'check_output',lambda *a,**k:calls.append(('public_reconstruction',)))
    return sources,saved,record,calls,root,native_root


def test_completed_audit_is_reused_and_bound_inputs_are_checked_without_reexecution(tmp_path,monkeypatch):
    sources,saved,record,calls,root,native_root=setup(tmp_path,monkeypatch)
    receipt=job.admit('a'*64,sources)
    assert 'public_reconstruction' not in [r[0] for r in calls]
    assert receipt['completed_public_prefix_audit_reused']
    assert not receipt['initial_public_prefix_reconstruction_required']
    assert receipt['boundary']['physics_samples']==200900 and receipt['original_owners_ended']
    assert not receipt['prior_controller_or_physics_execution_repeated']
    assert not receipt['prior_native_physical_prefix_reconstruction_repeated']
    assert not receipt['native_execution_started']
    calls.clear();job.verify_bound(receipt,sources)
    assert 'public_reconstruction' not in [r[0] for r in calls]
    assert any(r[0]=='artifacts' and r[1]==native_root and len(r[2])==4 for r in calls)
    assert any(r[0]=='artifacts' and r[1]==root and len(r[2])==7 for r in calls)


@pytest.mark.parametrize('fault',['live_prefix','live_native','live_worker','failure','scope','budget','source',
    'missing_source','negative','missing_artifact','report','context','worker','audit','sha'])
def test_live_incomplete_or_changed_admission_is_rejected(tmp_path,monkeypatch,fault):
    sources,saved,record,calls,root,native_root=setup(tmp_path,monkeypatch)
    launch=saved[(root,'launch.json')];result=saved[(root,'result.json')]
    if fault=='live_prefix':monkeypatch.setattr(job.run,'owner_live',lambda owner:owner['pid']==1)
    elif fault=='live_native':monkeypatch.setattr(job.run,'owner_live',lambda owner:owner['pid']==2)
    elif fault=='live_worker':monkeypatch.setattr(job.old_inputs,'worker_live',lambda:True)
    elif fault=='failure':(root/'failure.json').write_text('{}')
    elif fault=='scope':result['native_execution']=True
    elif fault=='budget':launch['candidate_navigation_ticks']=4000
    elif fault=='source':sources=sources|{'native.py':'4'*64}
    elif fault=='missing_source':del launch['source_sha256'][job.replay.SOURCE]
    elif fault=='negative':result['scientific_budget_only_prefix_supported']=False
    elif fault=='missing_artifact':del result['artifact_sha256']['identities.json']
    elif fault=='report':saved[(root,'report.json')]={'changed':True}
    elif fault=='context':launch['input_admission']['original_context_sha256']='4'*64
    elif fault=='worker':saved[(native_root,job.native.CASE[0]+'_worker_terminal.json')]['changed']=True
    elif fault=='audit':saved[(native_root,job.native.CASE[0]+'_audit.json')]['raw_sensor_reconstruction_pass']=False
    with pytest.raises(ValueError):job.admit('invalid' if fault=='sha' else 'a'*64,sources)
    assert 'public_reconstruction' not in [r[0] for r in calls]


def test_changed_later_receipt_and_changed_completed_audit_reject(tmp_path,monkeypatch):
    sources,saved,record,calls,root,native_root=setup(tmp_path,monkeypatch)
    receipt=job.admit('a'*64,sources)
    changed=deepcopy(receipt);changed['model_state_sha256']='4'*64
    with pytest.raises(ValueError,match='identities changed'):job.verify_bound(changed,sources)
    saved[(root,'result.json')]['complete_output_and_public_packets_rechecked']=False
    with pytest.raises(ValueError):job.admit('a'*64,sources)


def test_source_preparation_requires_the_whole_prospective_replay_source_and_test_set(monkeypatch):
    def prepare(seeds):
        assert {job.SOURCE,job.TEST,job.replay.SOURCE,job.replay.PROTOCOL,*job.replay.TESTS,'native_caller.py'}<=set(seeds)
        return {'source':'bound'}
    monkeypatch.setattr(job.replay.inputs,'prepared_sources',prepare)
    assert job.prepared_sources(('native_caller.py',))=={'source':'bound'}
