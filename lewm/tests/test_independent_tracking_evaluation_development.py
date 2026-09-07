"""Real paired-observer phase and numerical scoring with an injected raw auditor.

The raw-auditor fixture is not native reconstruction evidence. It tests phase
ordering, error mathematics, populations and failure retention without Genesis.
"""
import json
from copy import deepcopy
import numpy as np
import pytest

import scripts.independent_tracking_cohort_development as cohort
import scripts.independent_tracking_evaluation_development as scoring
from lewm.tests.test_independent_tracking_cohort_development import collected
from lewm.independent_tracking_challenge_development import TRIALS,specification
from lewm.independent_tracking_coverage_development import measured_coverage


@pytest.fixture
def sensor_phase(collected,monkeypatch):
    store,sha,_=collected;sensor_sha,phase=cohort.replay_population(store,sha)
    visited=[]
    def raw_audit(output,trial,result,protocol):
        # Independent guard in the fixture ensures the callback is not invoked
        # until every trial's sensor stream is durably represented in the phase.
        marker=cohort.read(output,'sensor_phase_complete.json')
        assert set(marker['reports'])==set(TRIALS) and all(r['frames']==3 for r in marker['reports'].values())
        visited.append(trial)
        n=result['physics_samples'];s=specification(trial);x,y,yaw=s['geometry']['spawn_se2_world']
        pose=np.zeros((n,7));pose[:,:3]=[x,y,.375];pose[:,5]=np.sin(yaw/2);pose[:,6]=np.cos(yaw/2)
        raw=dict(timestamp_s=np.arange(1,n+1)*.002,base_pose_world=pose,base_twist_world=np.zeros((n,6)))
        coverage=measured_coverage(s['direction'],raw['timestamp_s'],pose,raw['base_twist_world'],
            completed_ticks=result['completed_ticks'],schedule_complete=False,physical_stop=result['physical_stop'],acquisition_stop=None)
        return raw,dict(coverage=coverage,sensors=dict(depth_checks=[dict(within1mm=True,
            physical_visibility=dict(passes_sampled_physical_visibility=True))]*3),
            synthetic_raw_auditor=True,independent_observations_verified=False)
    monkeypatch.setattr(scoring,'_raw_audit',raw_audit)
    return store,sha,sensor_sha,phase,visited,raw_audit


def test_complete_sensor_population_precedes_all_native_callbacks_and_errors_are_scored(sensor_phase):
    store,sha,sensor_sha,phase,visited,_=sensor_phase
    r=scoring.evaluate_population(store,sha,sensor_sha,'a'*64)
    assert visited==list(TRIALS) and set(r['scores'])==set(TRIALS)
    assert r['all_candidate_pose_allocations_met'] and not r['all_intended_motion_covered']
    assert not r['full_challenge_pass'] and not r['goal_achieved'] and not r['independent_observations_verified']
    for s in r['scores'].values():
        for a in cohort.ARMS:
            assert s['arms'][a]['position_m']['count']==3
            assert s['arms'][a]['incremental_position_m']['count']==2
            assert s['arms'][a]['position_m']['maximum']<1e-8
            assert s['arms'][a]['orientation_rad']['maximum']<1e-8
        assert s['bridged_frame_errors']['position_m']['count']==0
        assert s['bridged_frame_errors']['position_m']['maximum'] is None


def test_native_motion_error_is_not_hidden_by_complete_tracking_availability(sensor_phase,monkeypatch):
    store,sha,sensor_sha,_,_,original=sensor_phase
    def displaced(*a):
        raw,audit=original(*a);raw['base_pose_world'][849,0]+=.03
        return raw,audit
    monkeypatch.setattr(scoring,'_raw_audit',displaced)
    r=scoring.evaluate_population(store,sha,sensor_sha,'a'*64)
    assert not r['all_candidate_pose_allocations_met'] and not r['full_challenge_pass']
    for s in r['scores'].values():
        assert s['arms']['temporal_anchor']['position_m']['maximum']==pytest.approx(.03,abs=1e-8)
        assert s['availability']['both']==3


@pytest.mark.parametrize('fault',['wrong_hash','missing_trial','wrong_collection'])
def test_failed_sensor_admission_cannot_invoke_native_auditor(sensor_phase,fault):
    import hashlib
    store,sha,sensor_sha,phase,visited,_=sensor_phase
    if fault=='wrong_hash':sensor_sha='0'*64
    else:
        if fault=='missing_trial':del phase['reports'][TRIALS[-1]]
        else:phase['collection_sha256']='0'*64
        p=store.output/'sensor_phase_complete.json';p.write_bytes(cohort.encode(phase))
        sensor_sha=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError):scoring.evaluate_population(store,sha,sensor_sha,'a'*64)
    assert not visited and store.failed and not (store.output/'result.json').exists()
    assert cohort.read(store.output,'failure.json')['stage']=='native_audit_or_admission'


def test_native_failure_keeps_prior_scores_and_cannot_retry(sensor_phase,monkeypatch):
    store,sha,sensor_sha,_,visited,original=sensor_phase
    def fail(output,trial,*a):
        if trial==TRIALS[1]:raise ValueError('synthetic raw-audit failure')
        return original(output,trial,*a)
    monkeypatch.setattr(scoring,'_raw_audit',fail)
    with pytest.raises(ValueError,match='synthetic'):scoring.evaluate_population(store,sha,sensor_sha,'a'*64)
    assert store.failed and visited==[TRIALS[0]]
    assert (store.output/(TRIALS[0]+'_audit.json')).is_file() and not (store.output/'result.json').exists()
    with pytest.raises(ValueError,match='restart'):scoring.evaluate_population(store,sha,sensor_sha,'a'*64)
    assert visited==[TRIALS[0]]


def test_clock_association_failure_stops_before_scoring_that_frame(sensor_phase,monkeypatch):
    store,sha,sensor_sha,_,_,original=sensor_phase
    def mismatch(*a):
        raw,audit=original(*a);raw['timestamp_s'][799]+=.001
        return raw,audit
    monkeypatch.setattr(scoring,'_raw_audit',mismatch)
    with pytest.raises(ValueError,match='clock'):scoring.evaluate_population(store,sha,sensor_sha,'a'*64)
    p=store.output/(TRIALS[0]+'_evaluation.jsonl')
    assert len(p.read_text().splitlines())==1 and store.failed
    assert not (store.output/'result.json').exists()


def test_missing_candidate_poses_keep_null_errors_and_common_denominators(collected,monkeypatch):
    store,sha,_=collected;candidate=cohort.MODELS['temporal_anchor']
    class Missing(candidate):
        def observe(self,*a,**k):
            r=super().observe(*a,**k)
            if self.model.frame>=1:r['current_pose']=None;r['terminal_failure']='synthetic dropout'
            return r
    monkeypatch.setitem(cohort.MODELS,'temporal_anchor',Missing)
    sensor_sha,phase=cohort.replay_population(store,sha)
    def raw_audit(output,trial,result,protocol):
        n=result['physics_samples'];p=np.zeros((n,7));p[:,6]=1
        return dict(timestamp_s=np.arange(1,n+1)*.002,base_pose_world=p),dict(
            coverage=dict(intended_motion_covered=False),sensors=dict(depth_checks=[]),synthetic_raw_auditor=True)
    monkeypatch.setattr(scoring,'_raw_audit',raw_audit)
    r=scoring.evaluate_population(store,sha,sensor_sha,'a'*64)
    for t,s in r['scores'].items():
        assert s['arms']['original']['position_m']['count']==3
        assert s['arms']['original']['paired_position_m']['count']==1
        assert s['arms']['temporal_anchor']['position_m']['count']==1
        assert not s['empirical_local_pose_allocation_met']['temporal_anchor']
        rows=[json.loads(x) for x in (store.output/(t+'_evaluation.jsonl')).read_text().splitlines()]
        assert rows[1]['errors']['temporal_anchor'] is None and rows[2]['errors']['temporal_anchor'] is None


@pytest.mark.parametrize('fault',[None,'friction_clock','raster_order','raw_request','selector'])
def test_raw_audit_wires_real_selector_command_and_coverage_checks(collected,monkeypatch,fault):
    """Lower native reconstruction callbacks are mocked; command/coverage math is real."""
    from lewm.independent_tracking_challenge_development import decision
    store,_,samples=collected;trial=TRIALS[0];r=store.episodes[trial]['result'];spec=specification(trial)
    n=r['physics_samples'];x,y,yaw=spec['geometry']['spawn_se2_world']
    pose=np.zeros((n,7));pose[:,:3]=[x,y,.375];pose[:,5]=np.sin(yaw/2);pose[:,6]=np.cos(yaw/2)
    raw=dict(timestamp_s=np.arange(1,n+1)*.002,base_pose_world=pose,base_twist_world=np.zeros((n,6)),
             requested_command=np.zeros((n,3),np.float64),applied_command=np.zeros((n,3)),
             phase=np.r_[np.zeros(750,np.uint8),np.ones(n-750,np.uint8)])
    rows=[];tape=[]
    for tick in range(3):
        d=decision('left',tick,samples[tick][0])
        rows.append(dict(tick=tick,observation_index=tick,pre_sample_index=749+50*tick,decision=d,
            acquisition_wall_ms=1.,selection_wall_ms=1.,acquisition_selection_deadline_missed=False,
            observer_computation_included=False,real_time_qualified=False))
        tape.append(dict(tick=tick,phase=d['phase'],role=d['role'],requested_command=d['requested_command'],
            pre_sample_index=749+50*tick,post_sample_index=min(799+50*tick,n-1),completed=tick<2,
            dispatch_and_physics_wall_ms=1.))
    friction=[dict(stage='before_settle',physics_steps=0)]+[
        dict(stage='before_decision',tick=i,physics_steps=750+50*i) for i in range(3)]+[dict(stage='terminal',physics_steps=n)]
    for f in friction:f.update(solver_friction=1.,solver_ratio=np.ones((1,28)).tolist())
    if fault=='friction_clock':friction[2]['physics_steps']+=1
    if fault=='raw_request':raw['requested_command'][750,0]=.1
    if fault=='selector':rows[0]['decision']['requested_command']=[.12,0.,0.]
    directory=store.output/trial
    for name,value in [('friction_checks.json',friction),('tracking_decisions.json',rows),('command_tape.json',tape),
                       ('actuator_identity.json',dict(effective={'kp':1})),('terminal_actuator_gains.json',{'kp':1}),
                       ('native_guard_rows.json',[])]:
        (directory/name).write_bytes(cohort.encode(value))
    cameras=[dict(physical_sample_index=749+50*i,world_from_optical=np.eye(4).tolist()) for i in range(3)]
    strict=dict(passes_sampled_physical_visibility=True)
    sensors=dict(depth_checks=[dict(within1mm=True,physical_visibility=strict)]*3)
    monkeypatch.setattr(scoring,'audit_sensors',lambda *a:(raw,{}, {}, {},cameras,[],object(),sensors))
    monkeypatch.setattr(scoring,'audit_setup',lambda *a:dict(mocked_native_setup=True))
    monkeypatch.setattr(scoring,'audit_stops',lambda *a:dict(sample_index=n-1,reason='DISALLOWED_CONTACT'))
    monkeypatch.setattr(scoring,'IntentReturnRGBDReplay',cohort.IntentReturnRGBDReplay)
    monkeypatch.setattr(scoring,'read_npz',lambda *a:dict(optical_depth_m=np.full((480,640),2.,np.float32)))
    monkeypatch.setattr(scoring,'evaluate_footprint',lambda *a,**k:dict(original_strict_score=strict))
    for i in range(3):
        raster=dict(physical_sample_index=749+50*i,order=dict(order='floor_first',roles=['floor','walls'],surfaces={'floor':0,'walls':1}),
                    precision=dict(subpixel_bits=8,depth_target_depth_bits=24,rgb_target_samples=1,rgb_target_sample_positions=[[.5,.5]]))
        if fault=='raster_order' and i==1:raster['order']['order']='wall_first'
        (directory/f'raster_{i:04d}.json').write_bytes(cohort.encode(raster))
    if fault:
        with pytest.raises((ValueError,AssertionError)):scoring._raw_audit(store.output,trial,r,'a'*64)
    else:
        actual,audit=scoring._raw_audit(store.output,trial,r,'a'*64)
        assert actual is raw and audit['command']['command_accounting_verified']
        assert not audit['coverage']['intended_motion_covered'] and audit['prefix']['status']=='MISSING_DEPARTURE_PREFIX'
        assert len(audit['raster_readbacks'])==len(audit['footprints'])==3
        assert not audit['independent_observations_verified']
