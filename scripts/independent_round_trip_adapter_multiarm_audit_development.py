"""Full public-sensor/controller replay for each fixed independent-study arm.

Physical geometry, contact, visibility and round-trip evaluation stay private.
"""
import json
from collections import Counter
import numpy as np
from lewm.independent_round_trip_multiarm_contract_development import (
    require_collection, treatment, verify_model, replay_receipt)
from lewm.independent_round_trip_comparison_study_development import require_case
from scripts.independent_round_trip_adapter_controller_factory_development import create
from scripts.navigation_artifact_root_development import validate_root, artifact_path
from scripts.renderer_witness_maze_probe_comparison_development import renderer_audit
from lewm.independent_round_trip_layouts_development import specification, public_mission
from lewm.novel_maze_round_trip_contract_development import (
    NAVIGATION_TICKS, MAX_COMMAND_TICKS, RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES)
from lewm.independent_round_trip_evaluation_development import evaluate
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.physical_execution_development import rotation_xyzw
from scripts.near_field_sensor_audit_development import audit_sensors, read_json
from scripts.auxiliary_downward45_sensor_audit_development import audit_rasters_and_footprints, audit_auxiliary
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.novel_maze_round_trip_command_audit_development import audit_commands as audit_learned_commands
from scripts.reactive_nominal_maze_command_audit_development import audit_commands as audit_reactive_commands
from scripts.maze_decision_stream_development import read_rows
from scripts.audit_go2_independent_pulse_context_pilot_v1 import audit_setup, audit_stops


def audit(case, result, definition, *, input_root, robot_geometry, correction_admission=None):
    arm = require_case(case); require_collection(case, result)
    layout_index = case.layout_index
    input_root = validate_root(input_root)
    directory = artifact_path(input_root, case.name+'/result.json').parent
    spec = specification(layout_index); mission = public_mission(layout_index)
    assert read_json(directory, 'specification.json') == spec and read_json(directory, 'result.json') == result
    assert read_json(directory, 'public_mission.json') == mission and result['navigation_ticks'] == NAVIGATION_TICKS
    raw, contacts, topology, roles, cameras, _, geometry, sensors = audit_sensors(directory, spec, result)
    assert len(raw['timestamp_s']) <= 750+50*MAX_COMMAND_TICKS
    np.testing.assert_allclose(raw['base_pose_world'][0, :2], spec['geometry']['spawn_se2_world'][:2], atol=.002, rtol=0)
    R = rotation_xyzw(raw['base_pose_world'][0, 3:])
    assert abs(np.arctan2(R[1, 0], R[0, 0])) < .002
    friction = read_json(directory, 'friction_checks.json')
    for row in friction:
        np.testing.assert_allclose(row['solver_friction'], spec['friction_mu'], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(row['solver_ratio'], np.ones((1, 28)))
    assert friction[0]['stage'] == 'before_settle' and friction[0]['physics_steps'] == 0
    assert friction[-1]['stage'] == 'terminal' and friction[-1]['physics_steps'] == len(raw['timestamp_s'])
    assert read_json(directory, 'actuator_identity.json')['effective'] == read_json(directory, 'terminal_actuator_gains.json')
    for tick, row in enumerate(friction[1:-1]):
        assert row['stage'] == 'before_decision' and row['tick'] == tick and row['physics_steps'] == 750+50*tick
    assert result['additional_auxiliary_rgb_for_motion'] is True
    assert result['measured_floor_transport_enabled'] is True and result['renderer_witnesses_recorded'] is True
    auxiliary_rows = read_json(directory, 'auxiliary_camera_audit.json')
    auxiliary_audit = audit_auxiliary(directory, spec, raw, cameras, topology, result)
    controller, model = create(case, robot_geometry, correction_admission=correction_admission)
    verify_model(case, model)
    reader = IntentReturnRGBDReplay(directory) if cameras else None
    rows = []; errors = []; actions = Counter(); arrival_transitions = []; latest_mission = None
    for tick, row in enumerate(read_rows(directory)):
        assert row['tick'] == row['observation_index'] == tick
        assert row['pre_sample_index'] == cameras[tick]['physical_sample_index'] == 749+50*tick
        assert row['resource_free_bytes'] >= RESERVE_BYTES+PERSISTENCE_HEADROOM_BYTES
        p, d, f, now = reader.packet(tick)
        image, auxiliary = packet(directory, tick, p, public_acquisition(auxiliary_rows[tick]), now_ns=now)
        replay = controller.observe(p, d, f, now_ns=now, auxiliary_depth=auxiliary, auxiliary_rgb=image)
        assert json.loads(json.dumps(replay)) == row['decision'], ('complete raw controller replay', tick)
        latest_mission = replay['mission_receipt']
        if latest_mission and latest_mission.get('arrival_confirmed_this_frame') and replay['evidence'] is not None:
            arrival_transitions.append(dict(tick=tick, mission=latest_mission))
        if replay['new_selection'] is not None: actions[str(replay['new_selection']['action'])] += 1
        evidence = replay['evidence']
        if evidence is not None and evidence['current_pose'] is not None:
            pose = raw['base_pose_world']
            actual = rotation_xyzw(pose[749, 3:]).T@(pose[749+50*tick, :3]-pose[749, :3])
            estimate = np.asarray(evidence['current_pose']['position_initial_body_m'])
            errors.append(float(np.linalg.norm(actual[:2]-estimate[:2])))
        # The complete decision is compared above and retained in the compressed
        # stream. Keep only the fields needed by the independent command audit.
        compact = {k: row[k] for k in row if k != 'decision'}
        compact['decision'] = {k: replay[k] for k in ('requested_command', 'terminal')}
        rows.append(compact)
    verify_model(case, model)
    assert len(rows) <= len(friction)-2 <= len(rows)+1
    assert len(rows) <= len(cameras) <= len(rows)+1 and latest_mission == result['mission_receipt']
    assert result['tracker_required_for_commands'] and not result['native_state_used_for_commands']
    with (directory/'decision_stream_timing.jsonl').open() as source:
        stream_timing = [json.loads(line) for line in source]
    assert len(stream_timing) == len(rows)
    for tick, (timing, row) in enumerate(zip(stream_timing, rows, strict=True)):
        assert timing['tick'] == tick
        assert all(np.isfinite(timing[k]) and timing[k] >= 0 for k in
            ('decision_receipt_write_wall_ms', 'iteration_with_receipt_wall_ms'))
        assert timing['iteration_with_receipt_wall_ms'] >= row.get('iteration_with_command_wall_ms', row['observation_and_control_wall_ms'])
    command_audit = audit_learned_commands if arm.model_name is not None else audit_reactive_commands
    command_audit(raw, read_json(directory, 'command_tape.json'), rows, result)
    setup = audit_setup(directory, raw, contacts, topology, geometry, result, definition)
    stops = audit_stops(raw, contacts, roles, friction, setup, read_json(directory, 'native_guard_rows.json'), result)
    footprints = audit_rasters_and_footprints(directory, spec, cameras, sensors)
    failed = [i for i, f in enumerate(footprints) if not f['stable_interior_metric_pass']
        or f['near_occlusion_failure'] or not auxiliary_audit[i]['auxiliary_visibility_pass']]
    visibility = bool(cameras and all(f['original_strict_score']['passes_sampled_physical_visibility'] for f in footprints)
        and all(r['auxiliary_visibility_pass'] for r in auxiliary_audit))
    renderer = renderer_audit(directory)
    outcome = evaluate(raw, latest_mission or dict(arrivals=[], terminal=None), result, layout_index=layout_index)
    verified = bool(outcome['native_round_trip_candidate_pass'] and visibility and not failed)
    return dict(**treatment(case), **replay_receipt(case),
        renderer_capture_audit=renderer, measured_floor_transport_enabled=True,
        raw_sensor_reconstruction_pass=True, additional_auxiliary_rgb_reconstructed=True,
        raw_command_audit_pass=True, physical_stop=stops,
        native_evaluation=outcome, verified_round_trip=verified, observed_arrival_transitions=arrival_transitions,
        strict_physical_visibility_pass=visibility, hard_measurement_failed_frames=failed,
        selected_actions=dict(actions), observed_pose_xy_errors_m=errors, depth_checks=sensors['depth_checks'],
        footprint_checks=footprints, auxiliary_sensor_audit=auxiliary_audit,
        observation_and_control_wall_ms=[r['observation_and_control_wall_ms'] for r in rows],
        iteration_with_command_wall_ms=[r['iteration_with_command_wall_ms'] for r in rows if 'iteration_with_command_wall_ms' in r],
        iteration_with_receipt_wall_ms=[r['iteration_with_receipt_wall_ms'] for r in stream_timing],
        decision_receipt_write_wall_ms=[r['decision_receipt_write_wall_ms'] for r in stream_timing],
        navigation_qualified=False, hardware_qualified=False,
        real_time_qualified=False, learned_planning_or_memory_advantage_established=False)
