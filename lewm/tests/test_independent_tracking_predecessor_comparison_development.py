"""Pure nonidentity counterexamples and authenticated-phase ordering tests."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from scripts import independent_tracking_predecessor_comparison_development as mod
from scripts import independent_tracking_stress_cohort_development as phase
from scripts import independent_tracking_cohort_development as base
from scripts import navigation_artifact_root_development as custody
from lewm.tests.test_rgbd_correspondence_motion_development import packets,texture
from lewm.tests.test_independent_tracking_stress_cohort_development import shifted


def wall(position=(1.,2.,3.),size=(1.,2.,3.),quat=(1.,0.,0.,0.),name='wall'):
    return dict(native_position=list(position),native_box_size=list(size)+[0.]*4,
        native_quaternion_wxyz=list(quat),native_name=name,pack_object={'material_id':'label'},
        fixed=True,collision_enabled=True,native_collision_boxes=1)


@pytest.fixture(scope='module')
def data():
    samples=[shifted(p) for p in packets([texture()]*9)]
    class Reader:
        def __init__(self,count=9):self.frames=[{'decision_ns':1_500_000_000+i*100_000_000} for i in range(count)]
        def packet(self,index):return deepcopy(samples[index])
    raw=dict(timestamp_s=np.arange(1,1151)*.002,base_pose_world=np.tile([0.,0.,.375,0.,0.,0.,1.],(1150,1)))
    cameras=[dict(physical_sample_index=749+50*i,
        rgb_sha256=hashlib.sha256(p[0]['image']['rgb'].tobytes()).hexdigest()) for i,p in enumerate(samples)]
    w=mod.extract_witness(raw,[wall()],Reader(),cameras)
    return raw,cameras,Reader,w,samples


def test_geometry_ignores_labels_order_quaternion_sign_and_equivalent_box_axes():
    first=[wall(),wall(position=(4.,5.,6.))]
    renamed=deepcopy(first[::-1]);renamed[0]['native_name']='different'
    renamed[0]['pack_object']['material_id']='not geometry'
    renamed[1]['native_quaternion_wxyz']=[-1.,0.,0.,0.]
    assert mod.same_geometry(mod.geometry_witness(first),mod.geometry_witness(renamed))
    equivalent=wall(size=(2.,1.,3.),quat=(np.sqrt(.5),0.,0.,np.sqrt(.5)))
    assert mod.same_geometry(mod.geometry_witness([wall()]),mod.geometry_witness([equivalent]))


def test_native_numeric_noise_is_not_new_geometry_but_resolved_changes_are():
    a=mod.geometry_witness([wall()]);b=mod.geometry_witness([wall(position=(1.+1e-7,2.,3.))])
    assert mod.same_geometry(a,b)
    b=mod.geometry_witness([wall(position=(1.01,2.,3.))]);assert not mod.same_geometry(a,b)
    assert not mod.same_geometry(a,mod.geometry_witness([wall(),wall(position=(4.,0.,0.))]))


@pytest.mark.parametrize('fault',['no_collision','not_fixed','multiple_boxes','zero_size','nan','bad_quaternion'])
def test_invalid_native_geometry_is_not_a_novel_scene(fault):
    row=wall()
    if fault=='no_collision':row['collision_enabled']=False
    elif fault=='not_fixed':row['fixed']=False
    elif fault=='multiple_boxes':row['native_collision_boxes']=2
    elif fault=='zero_size':row['native_box_size'][0]=0
    elif fault=='nan':row['native_position'][0]=float('nan')
    else:row['native_quaternion_wxyz']=[2.,0.,0.,0.]
    with pytest.raises(ValueError):mod.geometry_witness([row])


def test_value_identity_ignores_metadata_and_separates_control_from_sensing(data):
    packet=deepcopy(data[4][0]);before=mod.sensor_value_identity(packet)
    packet[0]['image']['calibration_id']='changed label'
    packet[0]['sensor_state']['identity']=(99,99,99)
    packet[1]['available_ns']+=1
    # This is only the value-comparison helper. The acquisition reader must
    # reject invalid metadata, not admit these fabricated sensor contracts.
    assert mod.sensor_value_identity(packet)==before
    packet[0]['sensor_state']['control']['applied_command']['values'][:]=1.
    after=mod.sensor_value_identity(packet)
    assert after['control']!=before['control']
    assert all(after[k]==before[k] for k in ('rgb','depth','body','fast_gyro'))


def test_identical_actual_prefix_is_not_rescored_as_new_evidence(data):
    witness=data[3];r=mod.compare_witnesses(witness,deepcopy(witness))
    assert r['native_box_inventory_matches'] and r['native_pose_prefix_matches']
    assert not r['nonidentity_checks_pass'] and all(not v for v in r['unequal_value_frames'].values())


@pytest.mark.parametrize('changed',['control','pose','start','geometry','later_view','only_initial_rgb','only_initial_depth'])
def test_each_incomplete_novelty_argument_fails_the_combined_check(data,changed):
    a=data[3];b=deepcopy(a)
    if changed=='control':b['sensor_values'][0]['control']='0'*64
    elif changed=='pose':b['native_pose_prefix_sha256']='0'*64
    elif changed=='start':b['actual_initial_pose_world'][0]+=.1
    elif changed=='geometry':b['native_geometry']=mod.geometry_witness([wall(position=(3.,0.,0.))])
    else:
        b['actual_initial_pose_world'][0]+=.1
        b['native_geometry']=mod.geometry_witness([wall(position=(3.,0.,0.))])
        if changed=='later_view':b['sensor_values'][1]['rgb']=b['sensor_values'][1]['depth']='0'*64
        elif changed=='only_initial_rgb':b['sensor_values'][0]['rgb']='0'*64
        else:b['sensor_values'][0]['depth']='0'*64
    assert not mod.compare_witnesses(a,b)['nonidentity_checks_pass']


def test_complete_nonidentity_check_still_does_not_claim_independence(data):
    a=data[3];b=deepcopy(a);b['actual_initial_pose_world'][0]+=.1
    b['native_geometry']=mod.geometry_witness([wall(position=(3.,0.,0.))])
    b['sensor_values'][0]['rgb']=b['sensor_values'][0]['depth']='0'*64
    r=mod.compare_witnesses(a,b)
    assert r['nonidentity_checks_pass']
    assert not r['independent_observations_verified'] and not r['maze_generalization_established']
    assert not r['rigid_transform_or_topology_isomorphism_test'] and not r['navigation_qualified']


@pytest.mark.parametrize('fault',['clock','camera_index','rgb_binding','native_quaternion','native_shape'])
def test_prefix_requires_actual_clock_content_and_pose_associations(data,fault):
    raw,cameras,Reader,_,_=data;raw=deepcopy(raw);cameras=deepcopy(cameras)
    if fault=='clock':raw['timestamp_s'][799]+=.001
    elif fault=='camera_index':cameras[0]['physical_sample_index']=750
    elif fault=='rgb_binding':cameras[0]['rgb_sha256']='0'*64
    elif fault=='native_quaternion':raw['base_pose_world'][750,3:]=0
    else:raw['base_pose_world']=raw['base_pose_world'][:,:6]
    with pytest.raises(ValueError):mod.extract_witness(raw,[wall()],Reader(),cameras)


@pytest.mark.parametrize('missing',['samples','frames','geometry'])
def test_incomplete_prefix_is_unavailable_not_nonidentical(data,missing):
    raw,cameras,Reader,full,_=data;raw=deepcopy(raw);static=[wall()];count=9
    if missing=='samples':raw={k:v[:749] for k,v in raw.items()}
    elif missing=='frames':count=8;cameras=cameras[:8]
    else:static=[]
    partial=mod.extract_witness(raw,static,Reader(count),cameras)
    assert partial['status']=='MISSING_COMPLETE_GEOMETRY_START_OR_SENSOR_PREFIX'
    assert mod.compare_witnesses(full,partial)['status']=='UNAVAILABLE_COMPARISON'


@pytest.mark.parametrize('fault',['wrong_hash','missing_binding','audit_failure'])
def test_predecessor_file_or_terminal_authority_failure_precedes_any_reader(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(custody,'BASE',tmp_path)
    root=tmp_path/'go2_prior_fixture_attempt_001';root.mkdir()
    result=dict(status='ROOM_RETURN_PULSE_COLLECTION_TERMINAL',absent_expected_artifacts=[],
        conditions={t:dict(physics_samples=1150,rgbd_frames=9) for t in mod.PRIOR_TRIALS},
        artifact_sha256={},goal_achieved=False)
    audit=dict(status='RAW_RETURN_AUDIT_PASS',conditions={t:dict(raw_sensor_audit_pass=True,
        physics_samples=1150,raw_depth_checks=9) for t in mod.PRIOR_TRIALS},goal_achieved=False)
    if fault=='audit_failure':audit['status']='FAIL'
    descriptor=dict(root=root.name)
    for key,name,data in [('launch','launch.json',dict(output_root=str(root))),
                          ('result','result.json',result),('audit','raw_return_audit.json',audit)]:
        (root/name).write_bytes(base.encode(data));descriptor[key]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if fault=='wrong_hash':descriptor['result']='0'*64
    monkeypatch.setattr(mod,'PREDECESSORS',{'fixture':descriptor})
    def forbidden(*a):raise AssertionError('no reader/native parser before bound complete authority')
    monkeypatch.setattr(mod,'IntentReturnRGBDReplay',forbidden);monkeypatch.setattr(mod,'_read_pose',forbidden)
    with pytest.raises(ValueError):mod.load_predecessors()


def test_selected_roster_is_narrow_and_rejects_protected_trial_names():
    paths=mod._selected_names('nominal_left',1138)
    assert len(paths)==25 and 'nominal_left/rgb_0008.png' in paths
    assert 'nominal_left/rgb_0009.png' not in paths
    for name in ('sealed','sealed_test.json','../nominal_left','nominal_left/extra'):
        with pytest.raises(ValueError):mod._selected_names(name,1138)


def test_new_native_access_is_impossible_before_complete_stress_admission(monkeypatch):
    calls=[]
    def rejected(*a):calls.append('phase');raise ValueError('incomplete stress phase')
    def forbidden(*a):raise AssertionError('native parser/predecessor load invoked too early')
    monkeypatch.setattr(phase,'admit_complete_sensor_phase',rejected)
    monkeypatch.setattr(mod,'_read_pose',forbidden);monkeypatch.setattr(mod,'load_predecessors',forbidden)
    with pytest.raises(ValueError,match='incomplete'):mod.compare_new_population(None,'a'*64,'b'*64,'c'*64,'d'*64)
    assert calls==['phase']


def test_missing_completed_raw_score_result_also_blocks_native_comparison(tmp_path,monkeypatch):
    monkeypatch.setattr(custody,'BASE',tmp_path)
    root=tmp_path/'go2_new_fixture_attempt_001';root.mkdir()
    monkeypatch.setattr(phase,'admit_complete_sensor_phase',lambda *a: ({},{},{}))
    def forbidden(*a):raise AssertionError('native parser invoked before completed result')
    monkeypatch.setattr(mod,'_read_pose',forbidden);monkeypatch.setattr(mod,'load_predecessors',forbidden)
    with pytest.raises(ValueError):mod.compare_new_population(root,'a'*64,'b'*64,'c'*64,'d'*64)


def test_population_comparison_keeps_all48_pairs_and_remains_unqualified(data,tmp_path,monkeypatch):
    """Native/phase callbacks mocked: wiring and denominator evidence only."""
    from lewm.independent_tracking_challenge_development import TRIALS
    monkeypatch.setattr(custody,'BASE',tmp_path)
    root=tmp_path/'go2_comparison_fixture_attempt_001';root.mkdir()
    raw,cameras,Reader,witness,samples=data;raw=deepcopy(raw);raw['base_pose_world'][:,0]+=.1
    class Changed(Reader):
        def __init__(self,path):super().__init__()
        def packet(self,index):
            p,d,f,now=super().packet(index)
            p['image']['rgb'][0,0]^=np.uint8(255);d['depth_m']+=np.float32(.01)
            d['rgb_sha256']=hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
            return p,d,f,now
    changed=Changed(None)
    cameras=[dict(physical_sample_index=749+50*i,
        rgb_sha256=hashlib.sha256(changed.packet(i)[0]['image']['rgb'].tobytes()).hexdigest()) for i in range(9)]
    hashes={}
    for trial in TRIALS:
        (root/trial).mkdir()
        for name,value in [('static_objects.json',[wall(position=(10.,0.,0.))]),('camera_audit.json',cameras)]:
            (root/trial/name).write_bytes(base.encode(value))
        name=trial+'_audit.json';(root/name).write_bytes(base.encode({'synthetic':True}))
        hashes[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    result=dict(status='EIGHT_TRIAL_BASE_AND_FIXED_STRESS_RAW_AUDIT_AND_SCORING_COMPLETE',
        collection_sha256='a'*64,base_phase_sha256='b'*64,stress_phase_sha256='c'*64,
        trials=list(TRIALS),output_sha256=hashes)
    (root/'result.json').write_bytes(base.encode(result));sha=hashlib.sha256((root/'result.json').read_bytes()).hexdigest()
    order=[]
    def admission(*a):
        order.append('all_sensor_phases')
        return {t:dict(result=dict(rgbd_frames=9,setup_checked=True)) for t in TRIALS},{},{}
    def native(output,name):
        assert order[0]=='all_sensor_phases';order.append(name)
        return deepcopy(raw)
    monkeypatch.setattr(phase,'admit_complete_sensor_phase',admission)
    monkeypatch.setattr(mod,'_read_pose',native);monkeypatch.setattr(mod,'IntentReturnRGBDReplay',Changed)
    monkeypatch.setattr(mod,'load_predecessors',lambda:dict(cohorts={
        cohort:dict(witnesses={t:deepcopy(witness) for t in mod.PRIOR_TRIALS}) for cohort in ('inner','intent')}))
    report=mod.compare_new_population(root,'a'*64,'b'*64,'c'*64,sha)
    assert len(report['comparisons'])==8 and all(len(p)==6 for p in report['comparisons'].values())
    assert report['all_six_predecessor_nonidentity_checks_pass']
    assert order[-1]=='all_sensor_phases' and len(order)==10
    assert not report['independent_observations_verified'] and not report['full_challenge_pass']
    assert not report['navigation_qualified'] and not report['goal_achieved']
