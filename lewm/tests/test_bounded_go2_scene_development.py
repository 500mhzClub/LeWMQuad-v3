import ast
from copy import deepcopy
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.physical_semantics import world_from_optical
from lewm_genesis import scene_builder as original
from lewm_genesis import bounded_scene_builder_development as bounded
from scripts.bounded_floor_physical_init_development import BoundedFloorPhysicalInit
from scripts import bounded_rgbd_session_development as session
from scripts.moving_rgbd_session_development import MovingRGBDSession
from scripts.single_sample_rgbd_session_development import SingleSampleRGBDSession
from scripts.run_physical_graph_edge_handoff_qualification_v1 import _GenesisPhysicalSession
from lewm.tests.test_single_sample_rgbd_evidence_development import capture_fixture
from lewm.tests.test_aligned_floor_development import identity


def tree(function): return ast.parse(textwrap.dedent(inspect.getsource(function)))


def test_native_init_changes_only_explicit_builder_import():
    old = tree(_GenesisPhysicalSession.__init__)
    replacements = 0
    for node in ast.walk(old):
        if isinstance(node, ast.ImportFrom) and node.module == 'lewm_genesis.scene_builder':
            node.module = 'lewm_genesis.bounded_scene_builder_development'; replacements += 1
    assert replacements == 1
    assert ast.dump(old) == ast.dump(tree(BoundedFloorPhysicalInit.__init__))


def test_all_nonfloor_builder_statements_preserved():
    old, new = tree(original.build_scene_from_pack), tree(bounded.build_scene_from_pack)
    # Inspect/remove only the declared guards, split-floor construction and
    # returned floor handles. Every other statement, including wall/robot/
    # material/camera construction and native build, must match the original.
    old_body, new_body = old.body[0].body, new.body[0].body
    floor_index = next(i for i, n in enumerate(old_body) if isinstance(n, ast.Expr)
                       and ast.unparse(n).startswith('scene.add_entity(gs.morphs.Plane()'))
    assert ast.unparse(new_body[1]).startswith("if n_envs != 1 or backend != 'cpu'")
    assert ast.unparse(new_body[2]) == 'domain = floor_domain(pack)'
    del new_body[1:3]
    assert ast.unparse(new_body[floor_index]).startswith('collision_floor = scene.add_entity(gs.morphs.Plane(visualization=False)')
    assert 'plane_size=(32.0, 32.0)' in ast.unparse(new_body[floor_index + 1])
    new_body[floor_index:floor_index + 2] = [deepcopy(old_body[floor_index])]
    assert ast.unparse(new_body[-2]) == 'read_extent_identity(collision_floor, visual_floor, 32.0)'
    assert 'collision_floor=collision_floor' in ast.unparse(new_body[-1])
    new_body[-2:] = [deepcopy(old_body[-1])]
    assert ast.dump(old) == ast.dump(new)


def test_capture_changes_only_finite_support_check_and_floor_identity():
    old = tree(SingleSampleRGBDSession.capture_fixed_rgb)
    body = old.body[0].body
    pos = next(i for i, n in enumerate(body) if ast.unparse(n).startswith('camera.set_pose('))
    body.insert(pos, ast.parse('check_capture_domain(self.ctx.build, world_from_optical(camera_position, forward, up))').body[0])
    for n in ast.walk(old):
        if isinstance(n, ast.Name) and n.id == 'floor_identity': n.id = 'bounded_floor_identity'
    assert ast.dump(old) == ast.dump(tree(session.BoundedRGBDSession.capture_fixed_rgb))


def test_c3_replaces_init_without_changing_gait_sensor_or_sample_methods():
    expected = list(MovingRGBDSession.__mro__)
    expected.insert(expected.index(_GenesisPhysicalSession), BoundedFloorPhysicalInit)
    assert session.BoundedRGBDSession.__mro__ == (session.BoundedRGBDSession, *expected)
    for name in ('command_tick', 'settle_recorded', '_sample', 'execute_requested_ticks',
                 'capture_observation', 'persist_observations', '_disallowed_contact'):
        assert getattr(session.BoundedRGBDSession, name) is getattr(MovingRGBDSession, name)


def pack():
    return SimpleNamespace(world_bounds_xy_m=((-4., -4.), (4., 4.)),
        robot=SimpleNamespace(spawn_xyz_m=(0., 0., .375)), camera_extrinsic_jitter=None,
        camera=SimpleNamespace(xyz_body_m=(.326, 0., .043), rpy_body_rad=(0., 0., 0.),
        native_resolution=(640, 480), fov_axis='horizontal', fov_deg=78.323, near_m=.05, far_m=200.))


def test_declared_workspace_and_camera_frustum_fit_finite_visual():
    domain = bounded.floor_domain(pack())
    assert domain['minimum_declared_support_margin_m'] > 4.
    for yaw in np.linspace(-np.pi, np.pi, 9):
        pose = world_from_optical([4., 4., .35], [np.cos(yaw), np.sin(yaw), 0.], [0., 0., 1.])
        assert bounded.check_capture_domain(None, pose)['finite_visual_support_verified']


@pytest.mark.parametrize('fault', ['large', 'inverted', 'nan', 'spawn', 'mount', 'jitter', 'fov', 'near', 'roll'])
def test_unreviewed_domain_rejected(fault):
    p = pack()
    if fault == 'large': p.world_bounds_xy_m = ((-10., -10.), (10., 10.))
    if fault == 'inverted': p.world_bounds_xy_m = ((4., 4.), (-4., -4.))
    if fault == 'nan': p.world_bounds_xy_m = ((np.nan, -4.), (4., 4.))
    if fault == 'spawn': p.robot.spawn_xyz_m = (5., 0., .3)
    if fault == 'mount': p.camera.xyz_body_m = (np.nan, 0., 0.)
    if fault == 'jitter': p.camera_extrinsic_jitter = object()
    if fault == 'fov': p.camera.fov_deg = 80.
    if fault == 'near': p.camera.near_m = .1
    if fault == 'roll': p.camera.rpy_body_rad = (0., .1, 0.)
    with pytest.raises(ValueError): bounded.floor_domain(p)


@pytest.mark.parametrize('fault', ['outside', 'nonrigid', 'nan', 'shape'])
def test_actual_capture_cannot_extrapolate_visual_support(fault):
    T = world_from_optical([14., 0., .35] if fault == 'outside' else [0., 0., .35], [1., 0., 0.], [0., 0., 1.])
    if fault == 'nonrigid': T[0, 0] = 2.
    if fault == 'nan': T[0, 0] = np.nan
    if fault == 'shape': T = T[:3]
    with pytest.raises(ValueError): bounded.check_capture_domain(None, T)


def test_contact_roles_exclude_visual_plane_and_preserve_existing_support_roles(monkeypatch):
    Plane = type('Plane', (), {})
    def entity(idx, name, plane=False):
        return SimpleNamespace(name=name, morph=Plane() if plane else object(), links=[SimpleNamespace(idx=idx, name=name)])
    floor, visual, robot, wall = entity(0, 'ground', True), entity(1, 'visual', True), entity(2, 'FL_calf'), entity(3, 'wall')
    build = SimpleNamespace(robot=robot, collision_floor=floor, visual_floor=visual,
                            scene=SimpleNamespace(entities=[floor, visual, robot, wall]))
    obj = object.__new__(session.BoundedRGBDSession)
    obj.ctx = SimpleNamespace(build=build); obj.geometry = {'wall_boxes': [{'wall_id': 'wall'}]}
    monkeypatch.setattr(session, 'bounded_floor_identity', lambda _: {})
    obj._contact_topology = obj._build_contact_topology()
    assert obj._contact_topology == {'robot': {2}, 'support': {2}, 'ground': {0}}
    obj.install_contact_identity()
    assert obj.object_ids == {0: 'ground_plane', 3: 'wall'} and 1 not in obj.link_names
    build.scene.entities.append(entity(4, 'unknown'))
    with pytest.raises(ValueError): obj.install_contact_identity()


def test_capture_preserves_raw_depth_and_uses_separate_identity(tmp_path, monkeypatch):
    old, calls = capture_fixture(tmp_path, monkeypatch)
    obj = object.__new__(session.BoundedRGBDSession); obj.__dict__.update(old.__dict__)
    row = identity()
    monkeypatch.setattr(session, 'bounded_floor_identity', lambda _: row)
    monkeypatch.setattr(session, 'sampling_readback', lambda _: {'draw_framebuffer_is_single_sample_target': True,
        'draw_framebuffer_is_multisample_target': False, 'samples': 0, 'sample_buffers': 0,
        'multisample_enabled': False, 'pixel_scale': 1})
    obj.capture_fixed_rgb(tmp_path, 'rgb_0000')
    with np.load(tmp_path/'native_depth_0000.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(f['optical_depth_m'], np.full((480, 640), 2., np.float32))
    assert len(calls) == 2 and obj._floor_identity == row
