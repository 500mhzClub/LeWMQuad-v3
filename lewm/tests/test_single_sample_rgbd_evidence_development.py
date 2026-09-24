import ast
from copy import deepcopy
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, INTRINSICS
from lewm.depth_geometry_evaluation_development import evaluate_depth as physical_reference
from lewm.visual_surface_depth_evaluation_development import check_floor_identity, expected_optical_depth, evaluate_depth
from scripts import single_sample_rgbd_session_development as native
from scripts import run_go2_single_sample_rgbd_observation_development_v1 as runner
from scripts import run_go2_rgbd_observation_development_v1 as original
from scripts.rgbd_session_development import RGBDSession


def floor_record():
    return {'visual_local_vertices_m': [[-500., -500., -.005], [500., -500., -.005],
        [500., 500., -.005], [-500., 500., -.005]], 'visual_faces': [[0, 1, 2], [0, 2, 3]],
        'visual_position_world_m': [0., 0., 0.], 'visual_quaternion_wxyz': [1., 0., 0., 0.],
        'collision_position_world_m': [0., 0., 0.], 'collision_quaternion_wxyz': [1., 0., 0., 0.],
        'collision_plane_data': [0., 0., 1., 0., 0., 0., 0.], 'collision_enabled': True, 'scope': 'evaluation only'}


def test_floor_geometry_accepts_equivalent_shared_or_unshared_vertices():
    record = floor_record()
    assert check_floor_identity(record)['rendered_floor_below_collision_m'] == .005
    record['visual_local_vertices_m'] = np.asarray(record['visual_local_vertices_m'])[record['visual_faces']].reshape(-1, 3).tolist()
    record['visual_faces'] = [[0, 1, 2], [3, 4, 5]]
    assert check_floor_identity(record)['physical_clearance_qualified'] is False


@pytest.mark.parametrize('fault', ['shift', 'collision', 'hole', 'extent', 'winding', 'normal', 'nan'])
def test_false_floor_identity_cannot_pass(fault):
    record = floor_record()
    if fault == 'shift': record['visual_position_world_m'][2] = .005
    if fault == 'collision': record['collision_enabled'] = False
    if fault == 'hole': record['visual_faces'][1] = [0, 1, 2]
    if fault == 'extent': record['visual_local_vertices_m'][0][0] = -499.
    if fault == 'winding': record['visual_faces'][1] = [0, 3, 2]
    if fault == 'normal': record['collision_plane_data'][2] = -1.
    if fault == 'nan': record['visual_local_vertices_m'][0][2] = np.nan
    with pytest.raises(ValueError): check_floor_identity(record)


def test_visual_surface_and_collision_plane_remain_different_claims():
    transform = np.array(BODY_FROM_OPTICAL); transform[2, 3] += .32
    for spec in runner.trials():
        ref = expected_optical_depth(spec['geometry']['wall_boxes'], transform)
        depth = np.full((480, 640), 200., np.float32)
        depth[np.ix_(ref['rows'], ref['columns'])] = np.where(np.isfinite(ref['expected_depth_m']), ref['expected_depth_m'], 200.)
        result = evaluate_depth(depth, spec['geometry']['wall_boxes'], transform, marker_case=spec['marker_case'])
        assert result['passes_declared_depth_check'] and result['visual_floor_z_m'] == -.005
        assert not physical_reference(depth, spec['geometry']['wall_boxes'], transform,
                                      marker_case=spec['marker_case'])['checks']['metric_optical_depth_within_5mm']


def test_collector_control_body_and_policy_packet_path_unchanged():
    def syntax(f): return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(f))))
    assert syntax(runner.collect) == syntax(original.collect)
    assert runner.RouteSession is native.SingleSampleRGBDSession
    for name in ('capture_observation', 'persist_observations', 'command_tick', 'settle_recorded', '_sample'):
        assert getattr(native.SingleSampleRGBDSession, name) is getattr(RGBDSession, name)
    for a, b in zip(runner.trials(), original.trials(), strict=True):
        assert {k: v for k, v in a.items() if k != 'scene_id'} == {k: v for k, v in b.items() if k != 'scene_id'}
    assert len(runner.verify_native()) == 10


def capture_fixture(tmp_path, monkeypatch, *, clock_fault=False, multisample=False):
    session = object.__new__(native.SingleSampleRGBDSession)
    calls = []
    class Camera:
        _raytracer = _batch_renderer = None
        res = (640, 480); intrinsics = np.array(INTRINSICS); near = .05; far = 200.; fov = 62.
        transform = np.eye(4)
        def set_pose(self, **kwargs): pass
        def render(self, **kwargs):
            calls.append(kwargs)
            if kwargs['rgb']: return np.zeros((480, 640, 3), np.uint8), None, None, None
            if clock_fault: session.ctx.runner._sim_time_ns += 2_000_000
            return None, np.full((480, 640), 2., np.float32), None, None
    robot = SimpleNamespace(get_pos=lambda: np.array([0., 0., .32]), get_quat=lambda: np.array([1., 0., 0., 0.]))
    session.ctx = SimpleNamespace(build=SimpleNamespace(robot=robot, camera=Camera()),
        runner=SimpleNamespace(_sim_time_ns=1_500_000_000, _extract_rgb=lambda x: x[0]),
        pack=SimpleNamespace(camera=SimpleNamespace(rpy_body_rad=[0., 0., 0.], xyz_body_m=[.326, 0., .043])))
    session.samples = [{'timestamp_s': 1.5}]; session.depth_manifest = []; session.depth_audit = []
    monkeypatch.setattr(native, 'floor_identity', lambda _: floor_record())
    monkeypatch.setattr(native, 'sampling_readback', lambda _: {'draw_framebuffer_is_single_sample_target': not multisample,
        'draw_framebuffer_is_multisample_target': multisample, 'samples': 4 if multisample else 0,
        'sample_buffers': 1 if multisample else 0, 'multisample_enabled': multisample, 'pixel_scale': 1})
    return session, calls


def test_capture_has_separate_render_calls_same_clock_and_raw_depth(tmp_path, monkeypatch):
    session, calls = capture_fixture(tmp_path, monkeypatch)
    session.capture_fixed_rgb(tmp_path, 'rgb_0000')
    assert calls == [dict(rgb=True, depth=False, segmentation=False, normal=False),
                     dict(rgb=False, depth=True, segmentation=False, normal=False)]
    meta = session.depth_audit[0]
    assert meta['same_render_call_as_rgb'] is False and meta['same_physics_and_camera_as_rgb'] is True
    assert meta['physics_clock_before_after_ns'] == [1_500_000_000]*2
    assert np.load(tmp_path/'native_depth_0000.npz')['optical_depth_m'][0, 0] == 2.


@pytest.mark.parametrize('fault', ['clock', 'multisample'])
def test_capture_rejects_intervening_physics_or_wrong_native_sampling(tmp_path, monkeypatch, fault):
    session, _ = capture_fixture(tmp_path, monkeypatch, clock_fault=fault == 'clock', multisample=fault == 'multisample')
    with pytest.raises(ValueError): session.capture_fixed_rgb(tmp_path, 'rgb_0000')
    assert not session.depth_audit and not (tmp_path/'native_depth_0000.npz').exists()
