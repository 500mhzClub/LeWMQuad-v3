from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from lewm.fresh_fused_maze_scene_development import specification, pack, EDGES, MARKER_CELL, PHYSICS_SEED, APPEARANCE_SEED
from lewm.rgb_marker_beacon_development import observe_marker
from lewm.whole_task_metrics_development import marker_centres_occluded
from lewm_genesis.fresh_maze_scene_development import mission_surfaces, independently_seeded_surfaces
from scripts.fresh_maze_physical_init_development import MissionPhysicalInit
from scripts.fresh_maze_session_development import MissionRGBDSession, marker_pixel_pairs, validate_command, priors
from scripts.rgbd_shadow_motion_physical_init_development import AppearancePhysicalInit
from scripts.whole_task_physics_session_development import GeometryFreePhysicalSample
from lewm.tests.test_observed_traversal_controller_development import Stream


def test_fresh_pack_matches_declared_geometry_seed_spawn_and_hidden_marker():
    s = specification()
    p = pack(s)
    assert p.physics_seed == PHYSICS_SEED and p.visual_seed == APPEARANCE_SEED
    assert p.robot.spawn_xyz_m == (0., 0., .375)
    assert len(p.static_objects) == len(s['geometry']['wall_boxes'])
    for obj, box in zip(p.static_objects, s['geometry']['wall_boxes'], strict=True):
        assert obj.object_id == box['wall_id']
        assert list(obj.center_xyz_m) == box['centre_xyz']
        assert list(obj.size_xyz_m) == box['size_xyz']
        assert obj.yaw_rad == box['yaw_rad'] and obj.material_id == box['material_id']
    assert marker_centres_occluded(s, [.326, 0., .418])
    assert not marker_centres_occluded(s, [3.6, -3.6, .43])


def test_scene_is_connected_with_junction_dead_ends_and_noninitial_marker():
    neighbors = {}
    for a, b in EDGES:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    seen = {(0, 0)}
    while True:
        reached = seen | set().union(*(neighbors[p] for p in seen))
        if reached == seen: break
        seen = reached
    assert seen == set(neighbors) and MARKER_CELL != (0, 0)
    assert max(map(len, neighbors.values())) == 4
    assert sum(len(n) == 1 for n in neighbors.values()) >= 3


@pytest.mark.parametrize('field', ['geometry', 'procedural_seed', 'appearance_seed'])
def test_initializer_cannot_silently_build_a_different_pack(field):
    s = specification()
    if field == 'geometry': s[field]['spawn_se2_world'][0] += .1
    else: s[field] += 1
    with pytest.raises(ValueError): pack(s)


@pytest.mark.parametrize('arm', ['neutral', 'repeated', 'distinctive'])
def test_marker_color_only_preserves_all_triangle_geometry_and_nonmarker_appearance(arm):
    boxes = specification()['geometry']['wall_boxes']
    before = independently_seeded_surfaces(boxes, arm, APPEARANCE_SEED)
    after = mission_surfaces(boxes, arm, APPEARANCE_SEED)
    for (an, a), (bn, b) in zip(before, after, strict=True):
        assert an == bn
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.faces, b.faces)
        if 'marker_red' in an: assert np.all(b.visual.vertex_colors == [230, 15, 15, 255])
        elif 'marker_blue' in an: assert np.all(b.visual.vertex_colors == [15, 15, 230, 255])
        else: np.testing.assert_array_equal(a.visual.vertex_colors, b.visual.vertex_colors)


def test_raw_pixel_assay_has_same_marker_predicate_as_sensor_detector():
    p, _, now = Stream().frame(0)
    rgb = p['image']['rgb']
    rgb[120:220, 280:310] = [230, 15, 15]
    rgb[120:220, 315:345] = [15, 15, 230]
    assert marker_pixel_pairs(rgb) == [r['bbox_xyxy'] for r in observe_marker(p, now_ns=now)['detections']]
    rgb[:] = 128
    assert marker_pixel_pairs(rgb) == []


def test_mro_selects_new_initializer_and_preserves_geometry_free_native_execution():
    mro = MissionRGBDSession.__mro__
    assert MissionPhysicalInit in mro and AppearancePhysicalInit not in mro
    assert next(c for c in mro if 'execute_requested_ticks' in c.__dict__) is GeometryFreePhysicalSample
    assert MissionRGBDSession.capture_fixed_rgb.__qualname__.startswith('AppearanceRGBDSession.')


@pytest.mark.parametrize('command', [[.21, 0, 0], [-.01, 0, 0], [0, .01, 0], [0, 0, .36], [float('nan'), 0, 0], [0, 0]])
def test_out_of_contract_commands_cannot_reach_native_runner(command):
    with pytest.raises(ValueError): validate_command(command)


def test_zero_release_and_limits_remain_explicit_and_setup_does_not_become_map():
    assert validate_command([0., 0., 0.]) == [0., 0., 0.]
    assert validate_command([.2, 0., -.35]) == [.2, 0., -.35]
    velocity, region = priors('a'*64)
    assert velocity.anchor_ns == region.anchor_ns == 1_500_000_000
    assert region.valid_until_ns == 2_500_000_000
    assert velocity.radius_m_s == .02
