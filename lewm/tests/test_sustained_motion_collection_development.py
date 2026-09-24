from copy import deepcopy

import numpy as np
import pytest

from lewm.sustained_motion_collection_development import (
    specification, pack, schedule, validate_command, TRIALS, SEGMENTS, TAIL_TICKS)
from lewm_genesis.bounded_scene_builder_development import floor_domain
from scripts.sustained_motion_physical_init_development import SustainedPhysicalInit
from scripts.sustained_motion_session_development import SustainedMotionSession
from scripts.rgbd_shadow_motion_physical_init_development import AppearancePhysicalInit
from scripts.whole_task_physics_session_development import GeometryFreePhysicalSample


@pytest.mark.parametrize('trial', TRIALS)
def test_frozen_scene_has_one_wide_plane_wall_and_finite_capture_domain(trial):
    s = specification(trial); p = pack(s)
    assert len(p.static_objects) == 1
    wall = p.static_objects[0]
    assert wall.object_id == 'wide_front' and wall.size_xyz_m == (.08, 16., .6)
    assert wall.center_xyz_m == (4.5, 0., .3) and wall.yaw_rad == 0.
    assert p.robot.spawn_xyz_m[:2] == tuple(s['geometry']['spawn_se2_world'][:2])
    assert p.robot.spawn_xyz_m[2] == .375 and p.physics_seed == s['procedural_seed']
    assert p.visual_seed == s['appearance_seed'] and floor_domain(p)['minimum_declared_support_margin_m'] > .01


def test_trials_are_distinct_predeclared_configs_not_a_relabelled_rerun():
    a, b = [pack(specification(t)) for t in TRIALS]
    assert a.physics_seed != b.physics_seed and a.visual_seed != b.visual_seed
    assert a.robot.spawn_xyz_m != b.robot.spawn_xyz_m and a.robot.spawn_quat_wxyz != b.robot.spawn_quat_wxyz
    assert a.manifest_sha256 != b.manifest_sha256
    assert a.static_objects == b.static_objects  # Not independent maze layouts.


@pytest.mark.parametrize('field', ['procedural_seed', 'appearance_seed', 'geometry', 'trial'])
def test_mutated_specification_cannot_reach_native_builder(field):
    s = deepcopy(specification('fit'))
    if field == 'geometry': s[field]['wall_boxes'][0]['size_xyz'][1] = 2.
    elif field == 'trial': s[field] = 'validation'
    else: s[field] += 1
    with pytest.raises(ValueError): pack(s)


def test_fixed_tape_has_sustained_travel_two_turns_brakes_and_tail():
    tape = schedule()
    assert len(tape) == 330 and TAIL_TICKS == 5
    assert [r['segment'] for r in tape[:10]] == ['initial_weak_observation']*10
    forward = [r for r in tape if r['segment']=='sustained_forward']
    assert len(forward) == 150 and all(r['requested_command']==[.12, 0., 0.] for r in forward)
    assert sum(r['requested_command'][2] > 0 for r in tape) == 50
    assert sum(r['requested_command'][2] < 0 for r in tape) == 50
    assert all(r['requested_command'] == [0., 0., 0.] for r in tape[-10:])
    for r in tape: validate_command(r['requested_command'])
    tape[0]['requested_command'][0] = 9.
    assert schedule()[0]['requested_command'] == [0., 0., 0.]
    assert len(SEGMENTS) == 9


@pytest.mark.parametrize('command', [[.121,0,0], [-.01,0,0], [0,.01,0], [0,0,.251], [np.nan,0,0]])
def test_out_of_protocol_command_rejected(command):
    with pytest.raises(ValueError): validate_command(command)


def test_mro_uses_new_scene_and_preserves_physical_sensor_wrappers():
    mro = SustainedMotionSession.__mro__
    assert SustainedPhysicalInit in mro and AppearancePhysicalInit not in mro
    assert next(c for c in mro if 'execute_requested_ticks' in c.__dict__) is GeometryFreePhysicalSample
    assert SustainedMotionSession.capture_fixed_rgb.__qualname__.startswith('AppearanceRGBDSession.')


def test_complete_artifact_roster_has_unique_paired_frames_and_native_evidence():
    from scripts.run_go2_sustained_observed_floor_motion_development_v1 import artifact_names
    names = artifact_names(336)
    assert len(names) == len(set(names)) == 1036
    for prefix, suffix in [('rgb', 'png'), ('depth', 'npz'), ('native_depth', 'npz')]:
        assert all(f'{prefix}_{i:04d}.{suffix}' in names for i in range(336))
        assert f'{prefix}_0336.{suffix}' not in names
    assert 'native_contacts.npz' in names and 'native_guard_rows.json' in names
    assert 'setup_checks.json' in names and 'command_tape.json' in names
    assert 'visual_meshes/wide_front_visual.ply' in names
