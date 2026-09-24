"""Raw sensor/command replay plus evaluator-only downstream goal verification."""
import json
import numpy as np
from lewm.surface_memory_goal_probe_development import SurfaceMemoryGoalProbe

from lewm.learned_goal_probe_development import ( GOAL_XY_M,
    WARMUP_TICKS, NAVIGATION_TICKS, DRAIN_TICKS)
from lewm.geometry_progress_layout_family_development import specification
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.physical_execution_development import rotation_xyzw
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.near_field_sensor_audit_development import audit_sensors, read_json
from scripts.geometry_progress_family_audit_development import audit_rasters_and_footprints
from scripts.read_go2_geometry_progress_commands_v1 import audit_setup, audit_stops
from scripts.geometry_progress_family_episode_development import RESERVE


def audit_commands(raw, tape, rows, result):
    n = len(raw['timestamp_s'])
    assert len(tape) == result['command_ticks'] <= WARMUP_TICKS + NAVIGATION_TICKS + DRAIN_TICKS
    assert len(rows) == result['decisions'] and len(tape) <= len(rows) <= len(tape)+1
    assert result['completed_ticks'] == sum(t['completed'] for t in tape)
    assert all(t['completed'] for t in tape[:-1])
    if tape and not tape[-1]['completed']:
        assert result['physical_stop'] is not None
    drain = 0
    for i, item in enumerate(tape):
        decision = rows[i]['decision']
        assert item['tick'] == rows[i]['tick'] == i
        assert item['requested_command'] == decision['requested_command']
        phase, role = ((3, 'terminal_zero_drain') if decision['terminal'] is not None else
            (1, 'causal_history_warmup') if i < WARMUP_TICKS else (2, 'online_learned_goal_command'))
        assert (item['phase'], item['role']) == (phase, role)
        if phase == 3:
            assert item['requested_command'] == [0., 0., 0.]
            drain += int(item['completed'])
        a, b = item['pre_sample_index'], item['post_sample_index']
        assert a == 749+50*i and type(b) is int and a <= b <= a+50 and b < n
        if item['completed']:
            assert b == a+50
        for key in ('requested_command', 'applied_command', 'post_slew_applied_command'):
            assert raw[key].dtype == np.float64
        request = np.asarray(item['requested_command'], np.float64)
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1], np.tile(request, (b-a, 1)))
        prior = raw['applied_command'][a].astype(np.float32)
        delta = np.array([.25, 0., .35], dtype=np.float32)
        applied = np.clip(request.astype(np.float32), prior-delta, prior+delta).astype(np.float64)
        for key in ('applied_command', 'post_slew_applied_command'):
            np.testing.assert_array_equal(raw[key][a+1:b+1], np.tile(applied, (b-a, 1)))
        np.testing.assert_array_equal(raw['phase'][a+1:b+1], np.full(b-a, phase))
    assert result['terminal_zero_ticks'] == drain <= DRAIN_TICKS
    assert n == min(n, 750) + sum(t['post_sample_index']-t['pre_sample_index'] for t in tape)
    np.testing.assert_array_equal(raw['requested_command'][:min(n, 750)], np.zeros((min(n, 750), 3)))
    if result['physical_stop'] is None and result['acquisition_stop'] is None:
        assert result['schedule_terminal'] is not None and drain == DRAIN_TICKS
        assert rows[-1]['decision']['terminal'] == result['schedule_terminal']
        assert len(rows) == len(tape)+1


def native_goal(raw, result):
    if len(raw['base_pose_world']) < 750:
        return dict(verified_goal_reached=False, terminal_distance_m=None)
    poses = raw['base_pose_world']
    R0 = rotation_xyzw(poses[749, 3:])
    local = (poses[749:, :3]-poses[749, :3]) @ R0
    distances = np.linalg.norm(local[:, :2]-np.asarray(GOAL_XY_M), axis=1)
    last = min(501, len(distances))
    quiet = (last == 501 and bool((distances[-last:] <= .06).all())
        and bool((np.linalg.norm(raw['base_twist_world'][-last:, :3], axis=1) <= .05).all())
        and bool((raw['requested_command'][-500:] == 0.).all()))
    verified = bool(quiet and result['schedule_terminal'] == 'OBSERVED_GOAL_CANDIDATE'
        and result['terminal_zero_ticks'] == DRAIN_TICKS and not raw['physics_contact'].any()
        and result['physical_stop'] is None and result['acquisition_stop'] is None)
    return dict(verified_goal_reached=verified, terminal_distance_m=float(distances[-1]),
        minimum_distance_m=float(distances.min()), terminal_displacement_initial_body_xy_m=local[-1, :2].tolist(),
        native_one_second_arrival_and_quiet_pass=bool(quiet), evaluator_only=True,
        native_goal_radius_m=.06, native_quiet_speed_m_s=.05)


def audit(trial, result, definition, *, input_root, model, robot_geometry, persistent, episode_name):
    directory = input_root / episode_name
    spec = specification(trial)
    assert read_json(directory, 'specification.json') == spec and read_json(directory, 'result.json') == result
    raw, contacts, topology, roles, cameras, _, geometry, sensors = audit_sensors(directory, spec, result)
    assert len(raw['timestamp_s']) <= 750+50*(WARMUP_TICKS+NAVIGATION_TICKS+DRAIN_TICKS)
    friction = read_json(directory, 'friction_checks.json')
    for f in friction:
        np.testing.assert_allclose(f['solver_friction'], spec['friction_mu'], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(f['solver_ratio'], np.ones((1, 28)))
    assert friction[0]['stage'] == 'before_settle' and friction[0]['physics_steps'] == 0
    assert friction[-1]['stage'] == 'terminal' and friction[-1]['physics_steps'] == len(raw['timestamp_s'])
    assert read_json(directory, 'actuator_identity.json')['effective'] == read_json(directory, 'terminal_actuator_gains.json')
    rows = read_json(directory, 'context_decisions.json')
    tape = read_json(directory, 'command_tape.json')
    assert len(rows) <= len(friction)-2 <= len(rows)+1
    for tick, f in enumerate(friction[1:-1]):
        assert f['stage'] == 'before_decision' and f['tick'] == tick and f['physics_steps'] == 750+50*tick
    before = state_digest(model.state_dict())
    controller = SurfaceMemoryGoalProbe(model, robot_geometry, persistent=persistent)
    reader = IntentReturnRGBDReplay(directory) if cameras else None
    selections = []
    errors = []
    for tick, row in enumerate(rows):
        assert row['tick'] == row['observation_index'] == tick
        assert row['pre_sample_index'] == cameras[tick]['physical_sample_index'] == 749+50*tick
        assert row['resource_free_bytes'] >= RESERVE
        p, d, f, now = reader.packet(tick)
        replay = controller.observe(p, d, f, now_ns=now)
        assert json.loads(json.dumps(replay)) == row['decision'], ('raw controller replay', tick)
        if replay['new_selection'] is not None:
            selections.append(dict(tick=tick, **replay['new_selection']))
        evidence = replay['evidence']
        if evidence is not None and evidence['current_pose'] is not None:
            pose = raw['base_pose_world']
            actual = rotation_xyzw(pose[749, 3:]).T @ (pose[749+50*tick, :3]-pose[749, :3])
            estimate = np.asarray(evidence['current_pose']['position_initial_body_m'])
            errors.append(float(np.linalg.norm(actual[:2]-estimate[:2])))
    assert state_digest(model.state_dict()) == before and all(p.grad is None for p in model.parameters())
    assert len(rows) <= len(cameras) <= len(rows)+1
    assert result['tracker_required_for_commands'] is True and result['native_state_used_for_commands'] is False
    audit_commands(raw, tape, rows, result)
    setup = audit_setup(directory, raw, contacts, topology, geometry, result, definition)
    stop = audit_stops(raw, contacts, roles, friction, setup, read_json(directory, 'native_guard_rows.json'), result)
    footprints = audit_rasters_and_footprints(directory, spec, cameras, sensors)
    return dict(trial=trial, raw_sensor_reconstruction_pass=True, raw_model_command_replay_pass=True,
        model_state_unchanged=True, goal=native_goal(raw, result), physical_stop=stop,
        controller_terminal=result['schedule_terminal'], frames=len(cameras), physics_samples=len(raw['timestamp_s']),
        online_selections=selections, selection_count=len(selections),
        observed_pose_xy_errors_m=errors, depth_checks=sensors['depth_checks'], footprint_checks=footprints,
        hard_measurement_failed_frames=[i for i, f in enumerate(footprints)
            if not f['stable_interior_metric_pass'] or f['near_occlusion_failure']],
        strict_physical_visibility_pass=bool(cameras and all(f['original_strict_score']['passes_sampled_physical_visibility'] for f in footprints)),
        observation_and_control_wall_ms=[r['observation_and_control_wall_ms'] for r in rows],
        iteration_with_command_wall_ms=[r['iteration_with_command_wall_ms'] for r in rows if 'iteration_with_command_wall_ms' in r],
        navigation_qualified=False, hardware_qualified=False, independent_maze_evaluation=False)
