"""Raw sensor/command replay plus evaluator-only downstream goal verification."""
import json
from scripts.overlap_retention_goal_audit_development import audit_commands,native_goal
import numpy as np
from lewm.causal_residual_final_goal_development import CausalResidualFinalGoalProbe

from lewm.learned_goal_probe_development import ( GOAL_XY_M,
    WARMUP_TICKS, NAVIGATION_TICKS, DRAIN_TICKS)
from lewm.geometry_progress_layout_family_development import specification
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.physical_execution_development import rotation_xyzw
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.near_field_sensor_audit_development import audit_sensors, read_json
from scripts.auxiliary_downward45_sensor_audit_development import audit_rasters_and_footprints,audit_auxiliary
from scripts.auxiliary_downward45_packet_replay_development import packet as auxiliary_packet,public_acquisition
from scripts.read_go2_geometry_progress_commands_v1 import audit_setup, audit_stops
from scripts.geometry_progress_family_episode_development import RESERVE






def audit(trial, result, definition, *, input_root, model, robot_geometry, persistent, episode_name, condition, variant):
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
    auxiliary_rows=read_json(directory,'auxiliary_camera_audit.json')
    auxiliary_audit=audit_auxiliary(directory,spec,raw,cameras,topology,result)
    before = state_digest(model.state_dict())
    controller = CausalResidualFinalGoalProbe(model, robot_geometry, persistent=persistent, condition=condition, variant=variant)
    reader = IntentReturnRGBDReplay(directory) if cameras else None
    selections = []
    errors = []
    for tick, row in enumerate(rows):
        assert row['tick'] == row['observation_index'] == tick
        assert row['pre_sample_index'] == cameras[tick]['physical_sample_index'] == 749+50*tick
        assert row['resource_free_bytes'] >= RESERVE
        p, d, f, now = reader.packet(tick)
        auxiliary=auxiliary_packet(directory,tick,p,public_acquisition(auxiliary_rows[tick]),now_ns=now)
        replay = controller.observe(p, d, f, now_ns=now,auxiliary_depth=auxiliary)
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
        auxiliary_sensor_audit=auxiliary_audit,
        hard_measurement_failed_frames=[i for i, f in enumerate(footprints)
            if not f['stable_interior_metric_pass'] or f['near_occlusion_failure'] or not auxiliary_audit[i]['auxiliary_visibility_pass']],
        strict_physical_visibility_pass=bool(cameras and all(f['original_strict_score']['passes_sampled_physical_visibility'] for f in footprints)
            and all(r['auxiliary_visibility_pass'] for r in auxiliary_audit)),
        observation_and_control_wall_ms=[r['observation_and_control_wall_ms'] for r in rows],
        iteration_with_command_wall_ms=[r['iteration_with_command_wall_ms'] for r in rows if 'iteration_with_command_wall_ms' in r],
        navigation_qualified=False, hardware_qualified=False, independent_maze_evaluation=False)


