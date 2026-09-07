"""Synthetic complete88-stream phase, actual observers, mocked native auditor.

Each case receives a fresh explicit copy of only the synthetic fixture roster;
this is not a runtime resume, source export or copying recorded native evidence.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts import independent_tracking_stress_cohort_development as mod
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_evaluation_development as scoring
from scripts import navigation_artifact_root_development as authority
from lewm.independent_tracking_challenge_development import TRIALS,MAX_FRAMES
from lewm.independent_tracking_stress_development import SCENARIOS,ARMS,PairedStressObserver,ONSET_FRAME
from lewm.tests.test_independent_tracking_cohort_development import episode
from lewm.tests.test_rgbd_correspondence_motion_development import packets,texture


def shifted(item):
    keys={'measured_ns','available_ns','decision_ns','image_ns','sensor_anchor_ns'}
    def shift(v):
        if isinstance(v,dict):return {k:val-100_000_000 if k in keys else shift(val) for k,val in v.items()}
        if isinstance(v,tuple):return tuple(shift(x) for x in v)
        return deepcopy(v)
    p,d,f,now=item
    return (*shift((p,d,f)),now-100_000_000)


@pytest.fixture(scope='module')
def template(tmp_path_factory):
    directory=tmp_path_factory.mktemp('stress_synthetic_template')
    samples=[shifted(x) for x in packets([texture()]*3)]
    class Reader:
        def __init__(self,path):self.frames=list(range(3))
        def packet(self,index):return deepcopy(samples[index])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(authority,'BASE',directory);patch.setattr(base,'IntentReturnRGBDReplay',Reader)
        root=directory/'go2_stress_template_attempt_001';root.mkdir();store=mod.StressCohortStore(root)
        for trial in TRIALS:store.admit_episode(trial,*episode(root,trial))
        collection_sha=store.complete_collection()
        base_sha,_=base.replay_population(store,collection_sha)
        stress_sha,phase=mod.replay_stress_population(store,collection_sha,base_sha)
    return store,collection_sha,base_sha,stress_sha,phase,Reader


@pytest.fixture
def prepared(template,tmp_path,monkeypatch):
    source,collection_sha,base_sha,stress_sha,phase,Reader=template
    monkeypatch.setattr(authority,'BASE',tmp_path);monkeypatch.setattr(base,'IntentReturnRGBDReplay',Reader)
    root=tmp_path/'go2_stress_case_attempt_001';root.mkdir();store=mod.StressCohortStore(root)
    bindings=dict(source.hashes)
    for trial,entry in source.episodes.items():
        bindings.update({trial+'/'+n:sha for n,sha in entry['receipt']['artifact_sha256'].items()})
    for name,sha in bindings.items():
        p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        assert not any(v=='sealed' or v=='sealed_test.json' or v.startswith('sealed_') for v in p.parts)
        src=source.output/p;assert src.resolve()==src and hashlib.sha256(src.read_bytes()).hexdigest()==sha
        destination=root/p;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,destination)
    # Restore only fresh synthetic fixture metadata, not a production resume API.
    store.hashes=deepcopy(source.hashes);store.sizes=deepcopy(source.sizes);store.episodes=deepcopy(source.episodes)
    return store,collection_sha,base_sha,stress_sha,deepcopy(phase)


def rebind(store,phase,trial=None,scenario=None,changed=None):
    if changed is not None:
        name=mod.stream_name(trial,scenario)
        (store.output/name).write_bytes(b''.join(base.encode(row) for row in changed))
        phase['reports'][trial][scenario]['estimates_sha256']=hashlib.sha256((store.output/name).read_bytes()).hexdigest()
    (store.output/mod.PHASE).write_bytes(base.encode(phase))
    return hashlib.sha256((store.output/mod.PHASE).read_bytes()).hexdigest()


def raw_fixture(visited):
    def audit(output,trial,result,protocol):
        marker=base.read(output,mod.PHASE)
        assert set(marker['reports'])==set(TRIALS)
        assert all(set(v)==set(SCENARIOS) for v in marker['reports'].values())
        visited.append(trial);n=result['physics_samples'];pose=np.zeros((n,7));pose[:,6]=1.
        return dict(timestamp_s=np.arange(1,n+1)*.002,base_pose_world=pose),dict(
            coverage=dict(intended_motion_covered=False),sensors=dict(depth_checks=[]),synthetic_raw_auditor=True)
    return audit


def test_all88_streams_reconstructed_without_parsing_native_sentinels(prepared):
    store,c,b,s,phase=prepared
    episodes,base_phase,actual=mod.admit_complete_sensor_phase(store.output,c,b,s)
    assert actual==phase and len(episodes)==8
    for trial in TRIALS:
        for scenario in SCENARIOS:
            r=phase['reports'][trial][scenario]
            assert r['frames']==3 and r['availability']['both']==3
            assert r['changed_packet_frames']==0 and not r['onset_recorded']
            assert all(not e['onset_update_attempted'] for e in r['exposure'].values())
    assert not actual['native_coordinates_parsed'] and not actual['independent_observations_verified']
    with pytest.raises(ValueError):mod.StressCohortStore(store.output)


@pytest.mark.parametrize('fault',['missing_scenario','missing_trial','summary','definition','budget','source_binding',
                                 'transform_binding','intervention','pose_shape','update','injection','frame',
                                 'privileged_row','float_frame'])
def test_rebound_mutations_rejected_before_first_native_callback(prepared,monkeypatch,fault):
    store,c,b,s,phase=prepared;trial=TRIALS[0];scenario=SCENARIOS[1]
    if fault=='missing_scenario':del phase['reports'][TRIALS[-1]][SCENARIOS[-1]]
    elif fault=='missing_trial':del phase['reports'][TRIALS[-1]]
    elif fault=='summary':phase['reports'][trial][scenario]['exposure']['original']['updates_with_changed_packet']=1
    elif fault=='definition':phase['definition_sha256']='0'*64
    elif fault=='budget':phase['resource_contract']['total_bytes']+=1
    else:
        rr=list(mod.rows(store.output,mod.stream_name(trial,scenario)));r=rr[0]
        if fault=='source_binding':r['source_packet_sha256']='0'*64
        elif fault=='transform_binding':r['intervened_packet_sha256']='0'*64
        elif fault=='intervention':r['intervention']['onset_reached']=True
        elif fault=='pose_shape':r['arms']['original']['pose']['position_initial_body_m']=[0.,0.]
        elif fault=='update':r['arms']['original']['observer_update_attempted']=False
        elif fault=='injection':r['arms']['original']['reference_injections']=[dict(kind='retained_reference_denied',reference_frame=0)]
        elif fault=='frame':r['frame']=1
        elif fault=='privileged_row':r['world_pose']=[0.,0.,0.]
        elif fault=='float_frame':r['frame']=0.
        s=rebind(store,phase,trial,scenario,rr)
    s=rebind(store,phase)
    visited=[];monkeypatch.setattr(scoring,'_raw_audit',raw_fixture(visited))
    with pytest.raises((ValueError,AssertionError)):
        mod.evaluate_complete_population(store,c,b,s,'a'*64)
    assert visited==[] and store.failed and not (store.output/'result.json').exists()
    assert base.read(store.output,'failure.json')['stage']=='complete_stress_native_audit_or_admission'


def test_nominal_stress_pose_must_match_base_even_if_both_are_finite(prepared):
    store,c,b,s,phase=prepared;t=TRIALS[0];scenario='nominal'
    rr=list(mod.rows(store.output,mod.stream_name(t,scenario)))
    rr[0]['arms']['original']['pose']['position_initial_body_m'][0]=.001
    s=rebind(store,phase,t,scenario,rr)
    with pytest.raises(ValueError,match='nominal'):mod.admit_complete_sensor_phase(store.output,c,b,s)


def test_all_native_callbacks_follow_full_phase_and_all_stress_errors_are_scored(prepared,monkeypatch):
    store,c,b,s,_=prepared;visited=[];monkeypatch.setattr(scoring,'_raw_audit',raw_fixture(visited))
    result=mod.evaluate_complete_population(store,c,b,s,'a'*64)
    assert visited==list(TRIALS) and set(result['stress_scores'])==set(TRIALS)
    assert result['stress_arms_evaluated'] and not result['full_challenge_pass']
    assert not result['goal_achieved'] and not result['independent_observations_verified']
    for cases in result['stress_scores'].values():
        for score in cases.values():
            assert score['arms']['temporal_anchor']['position_m']['count']==3
            assert score['arms']['temporal_anchor']['position_m']['maximum']<1e-8
            assert score['arms']['temporal_anchor']['incremental_position_m']['count']==2
            assert score['bridged_frame_errors']['position_m']['maximum'] is None


def test_native_error_survives_good_availability_and_no_fault_onset(prepared,monkeypatch):
    store,c,b,s,_=prepared;original=raw_fixture([])
    def displaced(*a):
        raw,audit=original(*a);raw['base_pose_world'][799,0]=.03
        return raw,audit
    monkeypatch.setattr(scoring,'_raw_audit',displaced)
    result=mod.evaluate_complete_population(store,c,b,s,'a'*64)
    for trial in TRIALS:
        for score in result['stress_scores'][trial].values():
            assert not score['empirical_local_pose_allocation_met']['temporal_anchor']
            assert score['arms']['temporal_anchor']['position_m']['maximum']==pytest.approx(.03,abs=1e-8)


def test_terminal_native_failure_retains_partial_scores_and_denies_retry(prepared,monkeypatch):
    store,c,b,s,_=prepared;visited=[];original=raw_fixture(visited)
    def failed(output,trial,*a):
        if trial==TRIALS[1]:raise ValueError('synthetic native failure')
        return original(output,trial,*a)
    monkeypatch.setattr(scoring,'_raw_audit',failed)
    with pytest.raises(ValueError,match='synthetic'):mod.evaluate_complete_population(store,c,b,s,'a'*64)
    assert visited==[TRIALS[0]] and store.failed
    assert (store.output/mod.stream_name(TRIALS[0],SCENARIOS[-1],True)).is_file()
    assert not (store.output/'result.json').exists()
    with pytest.raises(ValueError,match='restart'):mod.evaluate_complete_population(store,c,b,s,'a'*64)


def test_stress_replay_exception_retains_partial_stream_and_cannot_resume(prepared,monkeypatch):
    store,c,b,s,_=prepared
    # Remove only explicit stress outputs from this fresh synthetic fixture to
    # construct its base-only phase; never touch an actual experiment directory.
    names={mod.PHASE}|{mod.stream_name(t,scenario) for t in TRIALS for scenario in SCENARIOS}
    for name in names:
        (store.output/name).unlink();del store.hashes[name];del store.sizes[name]
    original=mod.PairedStressObserver
    class Broken(original):
        def observe(self,packet):
            if self.frame==1:raise RuntimeError('synthetic stress observer exception')
            return super().observe(packet)
    monkeypatch.setattr(mod,'PairedStressObserver',Broken)
    with pytest.raises(RuntimeError,match='synthetic'):mod.replay_stress_population(store,c,b)
    name=mod.stream_name(TRIALS[0],'nominal')
    assert len(list(mod.rows(store.output,name)))==1 and name in store.hashes
    assert store.failed and not (store.output/mod.PHASE).exists()
    assert base.read(store.output,'failure.json')['stage']=='stress_replay_or_admission'
    with pytest.raises(ValueError,match='restart'):mod.replay_stress_population(store,c,b)


def test_actual_roster_resource_envelope_and_bounded_reader(prepared,monkeypatch):
    store,_,_,_,_=prepared;contract=mod.resource_contract()
    assert len(store.allowed)==contract['streams']+contract['metadata_files']==214
    assert contract['streams']==192 and contract['metadata_files']==22
    assert contract['worst_case_bound_bytes']<=52*1024**3
    assert contract['worst_case_bound_bytes']>44*1024**3
    name=mod.stream_name(TRIALS[0],SCENARIOS[0],True)
    monkeypatch.setattr(base,'MAX_ROW',10)
    with pytest.raises(ValueError):
        with store.stream(name) as stream:store.append(stream,dict(exceeds='row budget'))
    (store.output/name).write_bytes(b'x'*11+b'\n')
    with pytest.raises(ValueError,match='bounded'):list(mod.rows(store.output,name))
    monkeypatch.setattr(mod,'TOTAL_BYTES',store.used)
    with pytest.raises(ValueError):store.check(1)


@pytest.fixture(scope='module')
def exercised():
    model=PairedStressObserver('anchor_absence_1');audit=mod.StressAudit('anchor_absence_1')
    onset=None;prefix=None
    for frame,packet in enumerate(packets([texture()]*(ONSET_FRAME+3))):
        packet=shifted(packet);row=json.loads(base.encode(model.observe(packet)));row['availability']=mod.category(row)
        if frame==ONSET_FRAME:prefix=deepcopy(audit);onset=(row,packet)
        audit.observe(row,packet)
    return audit.summary(),prefix,onset


def test_exercised_reference_fault_and_rejoin_are_not_confused_with_unreached_onset(exercised):
    summary,_,_=exercised
    assert summary['onset_recorded'] and summary['changed_packet_frames']==0
    assert summary['arms']['original']['first_failure']==84
    assert summary['arms']['temporal_anchor']['available']==87
    assert summary['exposure']['original']['frames_with_reference_injection']==1
    assert summary['exposure']['temporal_anchor']['frames_with_reference_injection']==1
    assert summary['continuity']['total_bridged_frames']==1
    assert summary['continuity']['bridge_spans'][0]['outcome']=='ANCHOR_REJOINED'
    assert not summary['independent_observations'] and not summary['navigation_qualified']


@pytest.mark.parametrize('fault',['duplicate_event','wrong_scenario','wrong_reference','missing_failure','hidden_skip'])
def test_reconstruct_actual_onset_rejects_false_injection_provenance(exercised,fault):
    _,prefix,(row,packet)=exercised;audit=deepcopy(prefix);row=deepcopy(row)
    r=row['arms']['original']
    if fault=='duplicate_event':r['reference_injections']*=2
    elif fault=='wrong_scenario':row['scenario']='nominal'
    elif fault=='wrong_reference':r['reference_injections'][0]['reference_frame']=84
    elif fault=='missing_failure':r['failure']=None
    elif fault=='hidden_skip':r['observer_update_attempted']=False
    with pytest.raises(ValueError):audit.observe(row,packet)


def test_empty_cohort_retains88_empty_streams_and_cannot_claim_coverage(tmp_path,monkeypatch):
    monkeypatch.setattr(authority,'BASE',tmp_path)
    root=tmp_path/'go2_empty_stress_attempt_001';root.mkdir();store=mod.StressCohortStore(root)
    for trial in TRIALS:store.admit_episode(trial,*episode(root,trial,0))
    c=store.complete_collection()
    def forbidden(*a):raise AssertionError('empty population must not construct reader')
    monkeypatch.setattr(base,'IntentReturnRGBDReplay',forbidden)
    b,_=base.replay_population(store,c);s,phase=mod.replay_stress_population(store,c,b)
    mod.admit_complete_sensor_phase(store.output,c,b,s)
    assert all(r['frames']==0 and not r['onset_recorded'] and not r['complete_requested_tape']
        for cases in phase['reports'].values() for r in cases.values())
    visited=[];monkeypatch.setattr(scoring,'_raw_audit',raw_fixture(visited))
    result=mod.evaluate_complete_population(store,c,b,s,'a'*64)
    assert visited==list(TRIALS) and not result['all_candidate_base_pose_allocations_met']
    for cases in result['stress_scores'].values():
        for score in cases.values():
            assert score['arms']['original']['position_m']['maximum'] is None
            assert not score['empirical_local_pose_allocation_met']['temporal_anchor']
