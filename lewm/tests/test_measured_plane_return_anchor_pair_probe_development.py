"""Boundary and failure-accounting checks for the fixed recorded-data probe."""
import gzip
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts import probe_go2_measured_plane_return_anchor_pairs_v1 as probe


def boundary_stream(directory, *, fault=None):
    with gzip.open(directory/'context_decisions.jsonl.gz', 'wt') as output:
        for tick in range(3124):
            bridge = 3103 <= tick <= 3112
            status = ('MEASURED_INCREMENT_BRIDGE' if bridge else
                'MEASURED_BRIDGE_BUDGET_EXHAUSTED' if tick == 3113 else 'ANCHOR_MEASUREMENT')
            count = tick-3102 if bridge else 0
            visual = dict(status='CURRENT_VISUAL_POSE', continuity_evidence=dict(
                status=status, bridge_frames=count), camera_selection=dict(auxiliary_attempted=tick==3113,
                    primary_continuity=dict(status=status)), reference_selection=dict(
                        attempts=[dict(reference_frame=frame) for frame in probe.REFERENCES]))
            row = dict(tick=tick, observation_index=tick, decision=dict(
                terminal='SENSOR_OR_MODEL_FAILURE' if tick>=3113 else None, failure=None,
                requested_command=[0.,0.,0.], original_visual_evidence=visual))
            if fault == 'reference' and tick == 3103:
                visual['reference_selection']['attempts'].pop()
            if fault == 'clock' and tick == 3103: row['observation_index'] += 1
            if fault == 'bridge' and tick == 3103: visual['continuity_evidence']['bridge_frames'] = 2
            if fault == 'camera' and tick == 3113: visual['camera_selection']['auxiliary_attempted'] = False
            if fault == 'population' and tick == 3123: continue
            output.write(json.dumps(row)+'\n')


@pytest.mark.parametrize('fault', ['reference','clock','bridge','camera','population'])
def test_changed_recorded_boundary_rejected(tmp_path, fault):
    boundary_stream(tmp_path, fault=fault)
    with pytest.raises(ValueError): probe.recorded_boundary(tmp_path)


def test_exact_recorded_boundary_and_raw_input_roster(tmp_path):
    boundary_stream(tmp_path)
    rows = probe.recorded_boundary(tmp_path)
    assert [row['tick'] for row in rows] == [3102,3103,3112,3113]
    names = probe.input_names()
    assert len(names) == len(set(names)) == 7+22*4
    assert 'auxiliary_rgb_3092.png' in names and 'auxiliary_depth_3113.npz' in names
    assert not any('3114' in name for name in names)


def fit_fixture(monkeypatch, *, stage=None, repeat_fault=False):
    arrays = (np.zeros((12,3)), np.ones((12,3)), np.zeros((12,2)), np.ones((12,2)))
    features = {frame:{camera:object() for camera in probe.CAMERAS} for frame in probe.FRAMES}
    gyros = {frame:np.eye(3) for frame in probe.FRAMES}; calls = []
    def descriptor(a,b):
        if stage == 'association': raise probe.SensorContractError('insufficient rigid-pose matches')
        return arrays
    chains = []
    def chained(sequence):
        chains.append(sequence)
        values = tuple(a.copy() for a in arrays)
        if repeat_fault and len(chains)==2: values[0][0,0] = 1
        return values,dict(endpoint_depth_pairs=12)
    def register(*args, **kwargs):
        calls.append(kwargs)
        if stage == 'endpoint_registration': raise probe.SensorContractError('rigid fit rejected')
        return np.eye(3),np.array([.01,.02,.03]),np.ones(12,bool),dict(inliers=12)
    monkeypatch.setattr(probe,'matched_points',descriptor)
    monkeypatch.setattr(probe,'chained_points',chained)
    monkeypatch.setattr(probe,'register',register)
    return features,gyros,calls,chains


@pytest.mark.parametrize('camera', probe.CAMERAS)
@pytest.mark.parametrize('method', ['descriptor','chained'])
def test_existing_methods_and_camera_endpoint_adapter(monkeypatch, camera, method):
    features,gyros,calls,chains=fit_fixture(monkeypatch)
    row=probe.fit_pair(features,gyros,3100,3113,camera,method)
    assert row['qualified'] and row['failure'] is None and row['failure_stage'] is None
    R,t=(np.eye(3),np.array([.01,.02,.03]))
    if camera=='auxiliary': R,t=probe.pose_in_body(R,t)
    np.testing.assert_array_equal(row['reference_body_from_current_body'],R)
    np.testing.assert_array_equal(row['translation_reference_body_m'],t)
    assert calls[0]['mode']=='joint' and calls[0]['frame']==3113
    assert len(row['endpoint_arrays'])==4
    if method=='chained':
        assert len(chains)==2
        assert [item[0] for item in chains[0]]==list(range(3100,3114))
        assert chains[0][0][2] is features[3100][camera]


@pytest.mark.parametrize('stage', ['association','endpoint_registration'])
def test_negative_pair_is_preserved(monkeypatch, stage):
    features,gyros,_,_=fit_fixture(monkeypatch,stage=stage)
    row=probe.fit_pair(features,gyros,3100,3113,'primary','descriptor')
    assert row['qualified'] is False and row['failure_stage']==stage and row['failure']


def test_nondeterministic_association_is_execution_failure(monkeypatch):
    features,gyros,_,_=fit_fixture(monkeypatch,repeat_fault=True)
    with pytest.raises(ValueError,match='byte-exactly'):
        probe.fit_pair(features,gyros,3100,3113,'primary','chained')


@pytest.mark.parametrize('args', [(3101,3113,'primary','chained'),(3100,3114,'primary','chained'),
    (3100,3113,'third','chained'),(3100,3113,'primary','new_rule')])
def test_extra_pairs_or_rules_rejected(args):
    with pytest.raises(ValueError): probe.fit_pair({}, {}, *args)


def test_main_admits_fixed_inputs_and_accounts_for_all_negative_pairs(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();directory=source/probe.CASE;directory.mkdir()
    output=tmp_path/'output';base=tmp_path
    collection=dict(decisions=3124,rgbd_frames=3124,auxiliary_frames=3124,
        schedule_terminal='SENSOR_OR_MODEL_FAILURE',terminal_zero_ticks=10)
    (directory/'result.json').write_text(json.dumps(collection))
    (source/'launch.json').write_text(json.dumps(dict(source_sha256={})))
    boundary_stream(directory)
    for name in probe.input_names():
        path=directory/name
        if not path.exists():path.write_text('[]' if name=='auxiliary_camera_audit.json' else '{}')
    (directory/'auxiliary_camera_audit.json').write_text(json.dumps([{}]*3124))
    monkeypatch.setattr(probe,'INPUT',source);monkeypatch.setattr(probe,'OUTPUT',output)
    monkeypatch.setattr(probe,'BASE',base)
    monkeypatch.setattr(probe,'LAUNCH_SHA',probe.digest(source/'launch.json'))
    monkeypatch.setattr(probe,'COLLECTION_SHA',probe.digest(directory/'result.json'))
    monkeypatch.setattr(probe,'validate_root',lambda root,**kwargs:root)
    monkeypatch.setattr(probe,'discover_sources',lambda *args:{})
    monkeypatch.setattr(probe,'verify_sources',lambda *args:None)
    monkeypatch.setattr(probe,'psutil',NS(virtual_memory=lambda:NS(available=64*1024**3)))
    monkeypatch.setattr(probe,'shutil',NS(disk_usage=lambda root:NS(free=100*1024**3)))
    monkeypatch.setattr(probe,'ExtendedBudgetRGBDReplay',lambda root:NS(frames=[None]*3124,
        packet=lambda frame:({'image':{'rgb':None}},None,None,1_500_000_000+frame*100_000_000)))
    monkeypatch.setattr(probe,'public_acquisition',lambda row:row)
    monkeypatch.setattr(probe,'rgb_packet',lambda *args,**kwargs:({'rgb':None},None))
    monkeypatch.setattr(probe,'CornerSupportFeatureFrame',lambda *args:NS(witness=lambda:{}))
    class Orientation:
        samples_integrated=0
        def begin(self,*args,**kwargs):return {'rotation_initial_body_from_current_body':np.eye(3)}
        def step(self,*args,**kwargs):
            self.samples_integrated+=50
            return self.begin()
    monkeypatch.setattr(probe,'FastRelativeOrientation',Orientation)
    calls=[]
    def fit(features,gyros,reference,current,camera,method):
        calls.append((current,camera,reference,method))
        return dict(reference_frame=reference,current_frame=current,camera=camera,method=method,
            qualified=False,failure='synthetic negative')
    monkeypatch.setattr(probe,'fit_pair',fit)
    probe.main()
    report=json.loads((output/'result.json').read_text())
    assert len(calls)==len(set(calls))==len(report['pairs'])==64
    assert report['gyro']['public_packets_validated']==22
    assert report['gyro']['samples_integrated']==1050
    assert report['scope']['all_pair_failures_preserved'] is True
    for key in ('native_final_audit_verified','pose_admitted','navigation_recovered',
            'measured_plane_refinement_applied','full_observer_replayed','goal_achieved'):
        assert report['scope'][key] is False
    with pytest.raises(ValueError,match='exclusive'):probe.main()
