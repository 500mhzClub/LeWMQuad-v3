"""Exact derivative scope and complete geometry behavior around reused projection."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
import pytest

from lewm.body_projected_floor_geometry_development import BodyProjectedFloorGeometry
from lewm.frame_cached_floor_geometry_development import FloorFrameGeometry
from lewm.current_primary_floor_plane_development import measured_points, ROWS, COLUMNS
from lewm.retained_floor_patch_development import RetainedFloorPatches
from lewm.tests.test_current_primary_floor_plane_development import depth_plane
from lewm.tests.test_frame_floor_cache_development import equal

DERIVATIVES = json.loads(Path('docs/go2_body_projected_floor_geometry_source_derivatives_2026-09-11.json').read_text())


def source_node(path, names):
    source = Path(path).read_text(); body = ast.parse(source).body
    for name in names.split('.'):
        node = next(n for n in body if getattr(n, 'name', None) == name); body = node.body
    return source, node


@pytest.mark.parametrize('entry', DERIVATIVES['derivatives'], ids=lambda entry:entry['target'])
def test_only_declared_projection_and_provider_calls_change(entry):
    source, node = source_node(entry['source'], entry['node'])
    assert hashlib.sha256(source.encode()).hexdigest() == entry['source_sha256']
    segment = ast.get_source_segment(source, node)
    lines = segment.splitlines()
    segment = lines[0]+'\n'+textwrap.dedent('\n'.join(lines[1:]))
    segment = lines[0]+'\n'+textwrap.indent(segment.split('\n', 1)[1], '    ')
    for before,after in entry['replacements']:
        assert segment.count(before) == 1
        segment = segment.replace(before,after)
    expected = ast.parse(segment).body[0]
    _,actual = source_node(DERIVATIVES['target_file'],entry['target'])
    for item in (expected,actual):
        if ast.get_docstring(item) is not None: item.body[0].value.value = ast.get_docstring(item)
    assert ast.dump(actual) == ast.dump(expected)


def exercise(geometry, depth, valid, rotation, position, height):
    outputs = [geometry.floor_coverage(depth,valid,rotation,position,height),
        geometry.sampled_floor_patch(depth,valid,rotation,position,height,ROWS,COLUMNS)]
    plane = geometry.primary_floor_plane(depth,valid,rotation,position,height)
    outputs.append(plane)
    outputs.append(geometry.confirm_auxiliary_floor(depth,valid,rotation,position,height,ROWS,COLUMNS,plane))
    history = RetainedFloorPatches()
    witness = dict(frame=0,measured_ns=1_500_000_000,rgb_sha256='a'*64,depth_sha256='b'*64)
    geometry.append_patch(history,depth,valid,rotation,position,height,witness)
    outputs.append(history.frames)
    return outputs,history


@pytest.mark.parametrize('height', [-.32,-.31])
@pytest.mark.parametrize('tilt', [0., .015])
def test_all_consumer_outputs_and_retained_prefix_bytes_remain_exact(height,tilt):
    depth,valid = depth_plane(height=height)
    c,s = np.cos(tilt),np.sin(tilt); rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    position = np.array([.003,-.002,.001])
    old,new = FloorFrameGeometry(),BodyProjectedFloorGeometry()
    expected,old_history=exercise(old,depth,valid,rotation,position,height)
    actual,new_history=exercise(new,depth,valid,rotation,position,height)
    equal(actual,expected); equal(new.counts(),old.counts())
    assert new.body_projection_cache.misses == 1 and new.body_projection_cache.hits >= 4
    new.close(); old.close()
    assert new._entries == {} and new.body_projection_cache._entries == {}
    equal(new_history.frames,old_history.frames)
    equal(new_history.coverage([[1.5,0.],[0.,0.]]),old_history.coverage([[1.5,0.],[0.,0.]]))


def test_different_up_vectors_reuse_projection_but_recompute_original_classification():
    depth,valid=depth_plane();old,new=FloorFrameGeometry(),BodyProjectedFloorGeometry()
    for tilt in (0.,.025):
        c,s=np.cos(tilt),np.sin(tilt);rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        equal(new.floor_coverage(depth,valid,rotation,np.zeros(3),-.32),
              old.floor_coverage(depth,valid,rotation,np.zeros(3),-.32))
    assert new.counts() == old.counts() == dict(hits=0,misses=2,uncached=0)
    assert new.body_projection_cache.counts() == dict(hits=1,misses=1,uncached=0)


def test_measured_points_validate_pose_and_retain_transform_bytes():
    depth,_=depth_plane();new=BodyProjectedFloorGeometry()
    for yaw in (0.,.04):
        c,s=np.cos(yaw),np.sin(yaw);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        position=np.array([.02,-.01,.003])
        equal(new.measured_points(depth,rotation,position),measured_points(depth,rotation,position))
    assert new.body_projection_cache.counts() == dict(hits=1,misses=1,uncached=0)


@pytest.mark.parametrize('method', ['floor_coverage','sampled_floor_patch','primary_floor_plane','confirm_auxiliary_floor','append_patch'])
@pytest.mark.parametrize('fault', ['nonzero_invalid','nan_depth','valid_dtype','depth_shape','rotation','position','height'])
def test_invalid_request_after_success_reaches_original_error_before_projection(monkeypatch,method,fault):
    depth,valid=depth_plane();old,new=FloorFrameGeometry(),BodyProjectedFloorGeometry()
    for geometry in (old,new):geometry.floor_coverage(depth,valid,np.eye(3),np.zeros(3),-.32)
    rotation=np.eye(3);position=np.zeros(3);height=-.32
    if fault=='nonzero_invalid':depth[0,0]=1.;valid[0,0]=False
    elif fault=='nan_depth':depth[0,0]=np.nan
    elif fault=='valid_dtype':valid=valid.astype(int)
    elif fault=='depth_shape':depth=depth[:10]
    elif fault=='rotation':rotation[0,0]=2.
    elif fault=='position':position=np.zeros(2)
    elif fault=='height':height=float('nan')
    monkeypatch.setattr(new,'body_projection',lambda *a:pytest.fail('original validation must precede projection'))
    def call(geometry):
        args=(depth,valid,rotation,position,height)
        if method=='sampled_floor_patch':return geometry.sampled_floor_patch(*args,ROWS,COLUMNS)
        if method=='confirm_auxiliary_floor':return geometry.confirm_auxiliary_floor(*args,ROWS,COLUMNS,{'available':False})
        if method=='append_patch':return geometry.append_patch(RetainedFloorPatches(),*args,dict(frame=0,measured_ns=1_500_000_000))
        return getattr(geometry,method)(*args)
    with pytest.raises((ValueError,TypeError,IndexError,KeyError)) as expected:call(old)
    with pytest.raises(type(expected.value)) as actual:call(new)
    assert str(actual.value)==str(expected.value)
    new.close();assert new.body_projection_cache._entries == {} and new._entries == {}


@pytest.mark.parametrize('fault', ['normal','offset'])
def test_late_plane_rejection_after_cached_patch_keeps_original_error(fault):
    depth,valid=depth_plane();old,new=FloorFrameGeometry(),BodyProjectedFloorGeometry()
    plane=dict(available=True,normal_map=[0.,0.,1.],offset_m=.32)
    if fault=='normal':plane['normal_map']=[0.,0.,2.]
    else:plane['offset_m']=float('nan')
    args=(depth,valid,np.eye(3),np.zeros(3),-.32,ROWS,COLUMNS,plane)
    with pytest.raises(ValueError) as expected:old.confirm_auxiliary_floor(*args)
    with pytest.raises(ValueError) as actual:new.confirm_auxiliary_floor(*args)
    assert str(actual.value)==str(expected.value)
    assert new.body_projection_cache.misses==1 and new.body_projection_cache.hits==1
    new.close();assert new.body_projection_cache._entries=={}


def test_new_camera_depth_and_mutated_depth_recompute_bounded_projection():
    old,new=FloorFrameGeometry(),BodyProjectedFloorGeometry()
    for height in (-.32,-.31,-.30):
        depth,valid=depth_plane(height=height)
        equal(new.floor_coverage(depth,valid,np.eye(3),np.zeros(3),height),
              old.floor_coverage(depth,valid,np.eye(3),np.zeros(3),height))
    assert new.body_projection_cache.counts()==dict(hits=0,misses=3,uncached=1)
    assert len(new.body_projection_cache._entries)==2
    new.close()
    with pytest.raises(ValueError,match='closed'):new.measured_points(depth,np.eye(3),np.zeros(3))
