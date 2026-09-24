from copy import deepcopy
from fractions import Fraction
from functools import partial
import itertools
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.later_floor_evidence_development import (
    LaterFloorEvidence, map_box, floor_squares, sphere_hits, resolve_sphere_query,
    PRIMARY_CALIBRATION, AUXILIARY_CALIBRATION)
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.measured_floor_partition_development import FOOT_IDS
from lewm.frame_cached_floor_map_development import FrameCachedFloorMemory, FrameCachedJointFloorRoundTripController
from lewm.later_floor_resolution_controller_development import (
    LaterResolvedFloorMemory, LaterFloorResolutionRoundTripController)
from lewm.tests.test_frame_floor_cache_development import equal
from lewm.tests.test_joint_floor_registered_controller_development import packets
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual

BOX = np.array([[.012, .012, -.302], [.018, .018, -.298]])


def observations(frame, primary=(), auxiliary=()):
    Q, q = reference_pose(np.eye(3), np.zeros(3))
    return [dict(cells=np.asarray(cells, dtype=np.int64).reshape(-1,2), witness=dict(
        camera=camera, frame=frame, measured_ns=1_500_000_000+frame*100_000_000,
        calibration_id=calibration, rgb_sha256='a'*64, depth_sha256='b'*64,
        rotation_map_from_reference=R.tolist(), position_map_m=p.tolist()))
        for camera, cells, calibration, R, p in (
            ('primary', primary, PRIMARY_CALIBRATION, np.eye(3), np.zeros(3)),
            ('auxiliary', auxiliary, AUXILIARY_CALIBRATION, Q, q))]


def record(ledger, frame, primary=(), auxiliary=()):
    ledger.record_pair(frame, 1_500_000_000+frame*100_000_000, np.eye(3), -.3,
        observations(frame, primary, auxiliary))


def test_outward_box_contains_exact_rational_corner_transforms():
    for angle in (0., .003, -.4, 1.2):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        bounds = np.array([[1.01, -.42, -.31], [1.031, -.401, -.299]])
        result = map_box(bounds, R)
        for point in itertools.product(*zip(*bounds)):
            for axis in range(3):
                exact = sum(Fraction(float(R[axis,j]))*Fraction(float(point[j])) for j in range(3))
                assert Fraction(float(result[0,axis])) <= exact <= Fraction(float(result[1,axis]))


def test_closed_grid_boundary_includes_both_sides_and_height_is_conservative():
    cells, reason = floor_squares(np.array([[.05,.02,-.3],[.05,.02,-.3]]), -.3)
    assert reason is None and cells == [(0,0),(1,0)]
    for height in (-.28, -.32):
        assert floor_squares(np.array([[.02,.02,height],[.02,.02,height]]), -.3)[0] is None
    assert floor_squares(np.array([[1e100,.02,-.3],[1e100,.02,-.3]]), -.3)[0] is None


def test_same_frame_evidence_does_not_resolve_but_later_complete_view_does():
    ledger = LaterFloorEvidence(); record(ledger, 0, [(0,0)])
    assert not ledger.resolve(BOX, 0, now_ns=ledger.now_ns)['resolved']
    record(ledger, 1, auxiliary=[(0,0)])
    result = ledger.resolve(BOX, 0, now_ns=ledger.now_ns)
    assert result['resolved'] and result['later_single_view']['camera'] == 'auxiliary'
    assert result['later_single_view']['frame'] == 1
    assert not result['ground_support_approved'] and not result['calibrated_pose_or_sensor_bounds']
    assert not result['original_return_classification_changed']


def test_separate_camera_or_frame_coverage_cannot_be_combined_into_one_view():
    ledger = LaterFloorEvidence(); record(ledger, 0)
    bounds = np.array([[.041,.012,-.302],[.059,.018,-.298]])
    record(ledger, 1, [(0,0)], [(1,0)])
    assert not ledger.resolve(bounds, 0, now_ns=ledger.now_ns)['resolved']
    record(ledger, 2, [(1,0)], [(0,0)])
    assert not ledger.resolve(bounds, 0, now_ns=ledger.now_ns)['resolved']
    record(ledger, 3, auxiliary=[(0,0),(1,0)])
    assert ledger.resolve(bounds, 0, now_ns=ledger.now_ns)['later_single_view']['frame'] == 3


def test_new_ambiguous_sample_invalidates_all_older_resolution_witnesses():
    ledger = LaterFloorEvidence(); record(ledger, 0); record(ledger, 1, [(0,0)])
    assert ledger.resolve(BOX, 0, now_ns=ledger.now_ns)['resolved']
    assert not ledger.resolve(BOX, 1, now_ns=ledger.now_ns)['resolved']
    record(ledger, 2)
    assert not ledger.resolve(BOX, 1, now_ns=ledger.now_ns)['resolved']
    record(ledger, 3, [(0,0)])
    assert ledger.resolve(BOX, 1, now_ns=ledger.now_ns)['later_single_view']['frame'] == 3


@pytest.mark.parametrize('fault', ['clock','order','calibration','hash','rgb_pair','transform','duplicate_cell','outside_cell'])
def test_bad_pair_rejected_before_any_ledger_publication(fault):
    ledger = LaterFloorEvidence(); views = observations(0, [(0,0)])
    if fault == 'clock': views[1]['witness']['measured_ns'] += 1
    elif fault == 'order': views.reverse()
    elif fault == 'calibration': views[1]['witness']['calibration_id'] = PRIMARY_CALIBRATION
    elif fault == 'hash': views[0]['witness']['depth_sha256'] = 'bad'
    elif fault == 'rgb_pair': views[1]['witness']['rgb_sha256'] = 'c'*64
    elif fault == 'transform': views[1]['witness']['position_map_m'][0] += .001
    elif fault == 'duplicate_cell': views[0]['cells'] = np.array([[0,0],[0,0]])
    else: views[0]['cells'] = np.array([[100,0]])
    with pytest.raises(ValueError): ledger.record_pair(0,1_500_000_000,np.eye(3),-.3,views)
    assert ledger.frame == -1 and not ledger._records and not ledger._cell_observations


def test_observation_inputs_and_returned_witnesses_cannot_mutate_history():
    ledger = LaterFloorEvidence(); record(ledger,0)
    views = observations(1,[(0,0)])
    ledger.record_pair(1,1_600_000_000,np.eye(3),-.3,views)
    views[0]['cells'][:] = 20; views[0]['witness']['rgb_sha256'] = 'f'*64
    a = ledger.resolve(BOX,0,now_ns=ledger.now_ns)
    a['later_single_view']['rgb_sha256'] = 'e'*64
    assert ledger.resolve(BOX,0,now_ns=ledger.now_ns)['later_single_view']['rgb_sha256'] == 'a'*64
    with pytest.raises(ValueError): ledger.resolve(BOX,0,now_ns=ledger.now_ns+1)
    with pytest.raises(ValueError): ledger.resolve(BOX,2,now_ns=ledger.now_ns)
    with pytest.raises(ValueError): record(ledger,1)


def test_every_sphere_hit_is_accounted_for_and_unresolved_bounds_remain():
    index = MeasuredSampleBoundsIndex()
    points = np.array([[.024,.02,-.3],[.026,.02,-.284]])
    index.insert(points, dict(frame=0,measured_ns=1_500_000_000))
    before = deepcopy(index.__dict__)
    ledger = LaterFloorEvidence(); record(ledger,0); record(ledger,1,[(0,0)])
    center = [.025,.02,-.299]; original = index.intersect_sphere(center,.022)
    assert original['intersecting_voxels'] == len(sphere_hits(index,center,.022)) == 2
    revised, proof = resolve_sphere_query(index,original,center,.022,ledger,now_ns=ledger.now_ns)
    assert proof['resolved_intersections'] == proof['remaining_intersections'] == 1
    assert len(proof['enclosures']) == 2 and revised['intersecting_voxels'] == 1
    assert revised['first_cell'] != original['first_cell']
    equal(before,index.__dict__)
    bad = deepcopy(original); bad['intersecting_voxels'] = 1
    with pytest.raises(ValueError): resolve_sphere_query(index,bad,center,.022,ledger,now_ns=ledger.now_ns)


def test_nonfoot_shapes_remain_blocking_and_original_queries_survive(monkeypatch):
    memory = LaterResolvedFloorMemory(identity=(0,0,0))
    memory.route = [{},{}]; memory.rotation=np.eye(3);memory.position=np.zeros(3);memory.joints=np.zeros(12)
    memory.partition.other.insert(np.array([[.016,.016,-.3]]),dict(frame=0,measured_ns=1_500_000_000))
    record(memory.later_floor_evidence,0);record(memory.later_floor_evidence,1,[(0,0)])
    centres = {key:([.016,.016,-.3] if i==0 else [1.+i,0.,-.3]) for i,key in enumerate(FOOT_IDS)}
    geometry = SimpleNamespace(_shapes=[dict(shape_id=k,kind='sphere',dimensions=[.022]) for k in FOOT_IDS],
        supports=lambda joints,R:dict(shapes=[dict(shape_id=k,center_body_m=v) for k,v in centres.items()]))
    original = dict(shapes=[dict(shape_id=k,**memory.partition.other.intersect_sphere(c,.022)) for k,c in centres.items()],
        auxiliary_shapes=[dict(shape_id=k,**memory.confirmed_auxiliary_partition.other.intersect_sphere(c,.022)) for k,c in centres.items()])
    trunk = dict(shape_id='trunk',intersecting_voxels=1)
    original['shapes'].append(trunk);original['auxiliary_shapes'].append(trunk)
    snapshot=deepcopy(original)
    monkeypatch.setattr(FrameCachedFloorMemory,'footprint',lambda *args,**kwargs:original)
    result=memory.footprint(geometry,[0.,0.],0.,now_ns=1_600_000_000)
    assert result['shapes'][0]['intersecting_voxels']==0 and result['possible_intersection']
    assert result['shapes'][-1]==trunk==result['auxiliary_shapes'][-1]
    assert not result['non_foot_contacts_exempted'] and not result['unresolved_contacts_exempted']
    equal(result['original_contact_check_before_later_floor_resolution'],snapshot);equal(original,snapshot)


def test_public_packet_mapping_stays_exact_and_records_two_views(monkeypatch):
    monkeypatch.setattr(fixture,'visual',partial(visual,origin=1_500_000_000))
    kwargs=dict(public_mission=dict(goal_initial_body_xy_m=[1.,0.],return_initial_body_xy_m=[0.,0.],require_return_after_goal=True),
        navigation_ticks=100,condition='jepa',variant='full',persistent=True)
    old=FrameCachedJointFloorRoundTripController(None,None,**kwargs)
    new=LaterFloorResolutionRoundTripController(None,None,**kwargs)
    previous=None
    for frame in range(2):
        p,d,a,raw,now=packets(frame,previous)
        for controller in (old,new):controller.motion=SimpleNamespace(observe=lambda *args,**kw:deepcopy(raw))
        x=old.observe(p,d,None,now_ns=now,auxiliary_depth=a);y=new.observe(p,d,None,now_ns=now,auxiliary_depth=a)
        assert x['terminal'] is None and y['terminal'] is None, y.get('failure')
        y.pop('later_measured_floor_contact_resolution_enabled');y['controller']=x['controller'];equal(x,y)
        assert new.memory.later_floor_evidence.frame==frame
        assert len(new.memory.later_floor_evidence._records)==2*(frame+1)
        assert new.mapper.frame_geometry is None and new.memory.frame_geometry is None
        previous=raw
