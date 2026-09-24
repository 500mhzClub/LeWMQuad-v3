"""Complete synthetic cohort receipts and real-observer sensor-only phase.

Native artifact bytes are deliberate non-NPZ sentinels: hashing is permitted,
parsing them would fail. A mock reader supplies valid synthetic causal packets,
not measurements from eight physical scenes. No native independence is claimed.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

import scripts.independent_tracking_cohort_development as mod
import scripts.navigation_artifact_root_development as authority
from scripts.independent_tracking_artifacts_development import EpisodeStore
from lewm.independent_tracking_challenge_development import TRIALS,specification
from lewm.tests.test_rgbd_correspondence_motion_development import packets,texture
from lewm.tests.independent_tracking_native_contact_fixtures import report_for_count


def result(trial,frames=3):
    return dict(status='TRACKING_TAPE_REQUIRES_RAW_AUDIT',trial=trial,initialized=True,
        setup_admitted=bool(frames),setup_checked=bool(frames),schedule_complete=False,
        physical_stop='DISALLOWED_CONTACT' if frames else 'BODY_STABILITY_LIMIT',acquisition_stop=None,
        infrastructure_failure=None,secondary_failures=[],command_ticks=frames,completed_ticks=max(0,frames-1),
        decisions=frames,physics_samples=750+50*(frames-1)+1 if frames else 1,rgbd_frames=frames,
        native_contact_integrity=report_for_count(750+50*(frames-1)+1 if frames else 1),
        lifecycle_wall_ms={},tracker_required_for_commands=False,native_state_used_for_commands=False,
        observer_executed=False,native_evaluation_executed=False,sensor_reconstruction_verified=False,
        navigation_qualified=False,real_time_qualified=False)


def episode(root,trial,frames=3):
    store=EpisodeStore(root,trial);r=result(trial,frames)
    (store.directory/'visual_meshes').mkdir()
    for name in mod.expected_episode_names(r):
        if name=='specification.json':store.json(name,specification(trial))
        elif name=='result.json':store.json(name,r)
        elif name=='policy_observations.json':
            store.json(name,dict(frames=[dict(rgb_file=f'rgb_{i:04d}.png',image_ns=1_500_000_000+i*100_000_000,
                                          decision_ns=1_500_000_000+i*100_000_000) for i in range(frames)]))
        else:store.write_bytes(name,b'synthetic unparsed artifact sentinel')
    return r,store.verify()


@pytest.fixture
def collected(tmp_path,monkeypatch):
    monkeypatch.setattr(authority,'BASE',tmp_path)
    root=tmp_path/'go2_cohort_synthetic_attempt_001';root.mkdir()
    store=mod.CohortStore(root)
    for trial in TRIALS:store.admit_episode(trial,*episode(root,trial))
    sha=store.complete_collection()
    # Shift the existing synthetic packet timeline from1.6s to1.5s, preserving
    # every relative measured/available/history relationship and pixel value.
    keys={'measured_ns','available_ns','decision_ns','image_ns','sensor_anchor_ns'}
    def shift(v):
        if isinstance(v,dict):return {k:val-100_000_000 if k in keys else shift(val) for k,val in v.items()}
        if isinstance(v,tuple):return tuple(shift(x) for x in v)
        return deepcopy(v)
    samples=[]
    for p,d,f,now in packets([texture()]*3):samples.append((*shift((p,d,f)),now-100_000_000))
    class Reader:
        def __init__(self,path):self.frames=list(range(3))
        def packet(self,index):return deepcopy(samples[index])
    monkeypatch.setattr(mod,'IntentReturnRGBDReplay',Reader)
    return store,sha,samples


def test_complete_receipts_do_not_parse_native_arrays(collected):
    store,sha,_=collected
    marker,episodes=mod.verify_collection(store.output,sha)
    assert marker['trials']==list(TRIALS) and len(episodes)==8
    assert all(e['result']['physical_stop'] for e in episodes.values())
    assert not marker['sensor_reconstruction_verified'] and not marker['navigation_qualified']
    with pytest.raises(ValueError):mod.CohortStore(store.output)


def test_real_frozen_observers_complete_all_sensor_streams_before_native_admission(collected):
    store,sha,_=collected
    with pytest.raises(ValueError):mod.admit_native_evaluation(store.output,sha,'0'*64)
    sensor_sha,phase=mod.replay_population(store,sha)
    episodes,admitted=mod.admit_native_evaluation(store.output,sha,sensor_sha)
    assert phase==admitted and len(episodes)==8
    for trial,r in phase['reports'].items():
        assert r['frames']==3 and r['availability']==dict(both=3,original_only=0,temporal_anchor_only=0,neither=0)
        assert all(a['available']==3 for a in r['arms'].values())
        assert not r['native_coordinates_parsed'] and not r['independent_observations_verified']
    assert not phase['navigation_qualified'] and not phase['independent_observations_verified']
    with pytest.raises(ValueError):mod.replay_population(store,sha)  # No overwrite/retry.


@pytest.mark.parametrize('fault',['missing_receipt','frame_count','bytes','extra_artifact','spec','failure','clock','partial_internal'])
def test_receipt_or_input_mutations_fail_before_reader_construction(collected,monkeypatch,fault):
    store,sha,_=collected;trial=TRIALS[0];row=deepcopy(store.episodes[trial])
    if fault=='missing_receipt':
        (store.output/(trial+'_receipt.json')).unlink()
    elif fault=='frame_count':row['result']['rgbd_frames']=2
    elif fault=='bytes':row['receipt']['artifact_bytes']+=1
    elif fault=='extra_artifact':row['receipt']['artifact_sha256']['unplanned.json']='a'*64
    elif fault=='spec':(store.output/trial/'specification.json').write_bytes(b'{}')
    elif fault=='failure':row['result']['infrastructure_failure']='hidden failure'
    elif fault=='clock':
        name=trial+'/policy_observations.json';m=mod.read(store.output,name);m['frames'][0]['decision_ns']+=1
        (store.output/name).write_bytes(mod.encode(m))
    elif fault=='partial_internal':row['receipt']['failed_internal']=True
    if fault in ('frame_count','bytes','extra_artifact','failure','partial_internal'):
        (store.output/(trial+'_receipt.json')).write_bytes(mod.encode(row))
    def forbidden(*args):raise AssertionError('reader must not be constructed')
    monkeypatch.setattr(mod,'IntentReturnRGBDReplay',forbidden)
    with pytest.raises(ValueError):mod.replay_population(store,sha)
    assert not (store.output/'sensor_phase_complete.json').exists()


def test_short_cohort_and_wrong_trial_order_cannot_get_collection_marker(tmp_path,monkeypatch):
    monkeypatch.setattr(authority,'BASE',tmp_path)
    root=tmp_path/'go2_short_cohort_attempt_001';root.mkdir();store=mod.CohortStore(root)
    with pytest.raises(ValueError):store.complete_collection()
    r,receipt=episode(root,TRIALS[1])
    with pytest.raises(ValueError):store.admit_episode(TRIALS[1],r,receipt)
    assert not (root/'collection_complete.json').exists()


def test_original_consumer_cannot_mutate_candidate_input(collected,monkeypatch):
    store,sha,_=collected;original=mod.MODELS['original']
    class Mutating(original):
        def observe(self,p,d,f,**kw):
            r=super().observe(p,d,f,**kw)
            p['image']['rgb'][:]=0;d['depth_m'][:]=0;f['values'][:]=999
            return r
    monkeypatch.setitem(mod.MODELS,'original',Mutating)
    sensor_sha,phase=mod.replay_population(store,sha)
    mod.admit_native_evaluation(store.output,sha,sensor_sha)
    assert all(r['arms']['temporal_anchor']['available']==3 for r in phase['reports'].values())


def test_candidate_missingness_keeps_full_population_and_original_denominators(collected,monkeypatch):
    store,sha,_=collected;candidate=mod.MODELS['temporal_anchor']
    class Missing(candidate):
        def observe(self,*a,**k):
            r=super().observe(*a,**k)
            if self.model.frame>=1:r['current_pose']=None;r['terminal_failure']='synthetic dropout'
            return r
    monkeypatch.setitem(mod.MODELS,'temporal_anchor',Missing)
    sensor_sha,phase=mod.replay_population(store,sha)
    mod.admit_native_evaluation(store.output,sha,sensor_sha)
    for r in phase['reports'].values():
        assert r['availability']==dict(both=1,original_only=2,temporal_anchor_only=0,neither=0)
        assert r['arms']['temporal_anchor']==dict(frames=3,available=1,first_failure=1)


def test_exception_in_sensor_phase_retains_partial_stream_without_native_admission(collected,monkeypatch):
    store,sha,_=collected;candidate=mod.MODELS['temporal_anchor']
    class Broken(candidate):
        def observe(self,*a,**k):
            if self.model.frame>=0:raise RuntimeError('synthetic observer exception')
            return super().observe(*a,**k)
    monkeypatch.setitem(mod.MODELS,'temporal_anchor',Broken)
    with pytest.raises(RuntimeError):mod.replay_population(store,sha)
    assert store.failed and TRIALS[0]+'_estimates.jsonl' in store.hashes
    assert not (store.output/'sensor_phase_complete.json').exists()
    with pytest.raises(ValueError):mod.admit_native_evaluation(store.output,sha,'0'*64)


@pytest.mark.parametrize('fault',['denominator','missing_trial','wrong_collection','row_frame','row_timing','row_pose'])
def test_rebound_but_inconsistent_sensor_metadata_or_rows_are_rejected(collected,fault):
    store,sha,_=collected;sensor_sha,phase=mod.replay_population(store,sha)
    trial=TRIALS[0];r=phase['reports'][trial]
    if fault=='denominator':r['arms']['original']['available']-=1
    elif fault=='missing_trial':del phase['reports'][TRIALS[-1]]
    elif fault=='wrong_collection':phase['collection_sha256']='0'*64
    else:
        p=store.output/r['estimates_file'];rows=[json.loads(line) for line in p.read_text().splitlines()]
        if fault=='row_frame':rows[-1]['frame']=0
        elif fault=='row_timing':rows[-1]['arms']['original']['observer_wall_ms']=-1
        else:rows[-1]['arms']['original']['pose']['position_initial_body_m']=[0.,0.]
        p.write_bytes(b''.join(mod.encode(row) for row in rows));r['estimates_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    p=store.output/'sensor_phase_complete.json';p.write_bytes(mod.encode(phase));sensor_sha=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises((ValueError,AssertionError)):mod.admit_native_evaluation(store.output,sha,sensor_sha)


def test_empty_physical_stop_trial_remains_in_complete_population(tmp_path,monkeypatch):
    monkeypatch.setattr(authority,'BASE',tmp_path)
    root=tmp_path/'go2_empty_cohort_attempt_001';root.mkdir();store=mod.CohortStore(root)
    for t in TRIALS:store.admit_episode(t,*episode(root,t,frames=0))
    sha=store.complete_collection()
    def forbidden(*args):raise AssertionError('no reader for zero frames')
    monkeypatch.setattr(mod,'IntentReturnRGBDReplay',forbidden)
    sensor_sha,phase=mod.replay_population(store,sha)
    mod.admit_native_evaluation(root,sha,sensor_sha)
    assert set(phase['reports'])==set(TRIALS) and all(r['frames']==0 for r in phase['reports'].values())
    assert not phase['navigation_qualified'] and not phase['independent_observations_verified']


def test_serialized_metadata_row_and_total_budgets_are_enforced(collected,monkeypatch):
    store,_,_=collected
    monkeypatch.setattr(mod,'MAX_METADATA',10)
    with pytest.raises(ValueError):store.save('result.json',dict(message='too long'))
    monkeypatch.setattr(mod,'MAX_ROW',10)
    with pytest.raises(ValueError):
        with store.stream(TRIALS[0]+'_evaluation.jsonl') as stream:store.append(stream,dict(message='too long'))
    assert store.failed and (store.output/(TRIALS[0]+'_evaluation.jsonl')).is_file()
    monkeypatch.setattr(mod,'TOTAL_BYTES',store.used)
    with pytest.raises(ValueError):store.check(1)
