"""Independent saved-data replay and physical accounting; never executes physics."""
import json

import cv2
import numpy as np
from PIL import Image

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.depth_proposal_navigation_development import DepthProposalNavigation
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries, nonfoot_ground_contact_indices
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.rgbd_shadow_motion_development import POINT_HYPOTHESES
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot, initial_ground_support_witness
from lewm.whole_task_metrics_development import marker_centres_occluded, reduce_whole_task
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_startup_observation_turn_development_v1 import native_materials
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.fresh_maze_session_development import priors, marker_pixel_pairs, validate_command
from scripts.fresh_maze_turn_conflict_audit_development import diagnose
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.rgbd_shadow_motion_raw_audit_development import audit_sensors, read_json, contact_packet, classify
from scripts.run_go2_fresh_fused_maze_development_v1 import OUTPUT, PROTOCOL, specification, preflight, artifact_names
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json

IDENTITIES = dict(launch='54c9e50bba03f6e74dce97d2338d96176b3436269066e1ad7deca94db5e8ab30',
                  result='5ede0552ccd31681f6cbb12ba441005dbf9e221289cba36c50546d2c31faa1f9')
SOURCES = ('scripts/audit_go2_fresh_fused_maze_development_v1.py',
           'scripts/fresh_maze_turn_conflict_audit_development.py')


def exact(actual, expected):
    assert json.loads(json.dumps(actual, allow_nan=False)) == expected


def commands(raw, decisions, tape, cameras):
    """This audit targets the completed, unfaulted first attempt, not arbitrary runs."""
    assert len(decisions) == 219 and len(tape) == 223
    assert all(t['phase'] == 1 for t in tape[:218]) and all(t['phase'] == 2 for t in tape[218:])
    for field in ('requested_command', 'applied_command'):
        np.testing.assert_array_equal(raw[field][:750], np.zeros((750, 3)))
    np.testing.assert_array_equal(raw['phase'][:750], np.zeros(750))
    np.testing.assert_array_equal(raw['post_slew_applied_command'], raw['applied_command'])
    for i, row in enumerate(decisions):
        assert row['tick'] == row['observation_index'] == i
        assert row['decision_ns'] == 1_500_000_000+i*100_000_000
        assert row['pre_sample_index'] == cameras[i]['physical_sample_index'] == 749+50*i
        assert row['failure'] is None and row['executed'] == (i < 218)
        assert row['controller']['terminal'] == (i == 218)
        assert row['end_perf_counter_ns'] >= row['start_perf_counter_ns']
        assert row['outer_wall_ms'] == (row['end_perf_counter_ns']-row['start_perf_counter_ns'])/1e6
        if i < 218:
            exact(tape[i]['requested_command'], validate_command(row['controller']['requested_command']))
            assert tape[i]['decision_tick'] == i
    previous = 749
    for i, entry in enumerate(tape):
        assert entry['completed'] and entry['pre_sample_index'] == previous
        end = entry['post_sample_index']; assert end-previous == 50
        if i >= 218:
            assert entry['tail_tick'] == i-218 and entry['requested_command'] == [0., 0., 0.]
        sl = slice(previous+1, end+1)
        np.testing.assert_array_equal(raw['requested_command'][sl], np.tile(entry['requested_command'], (50, 1)))
        np.testing.assert_array_equal(raw['phase'][sl], np.full(50, entry['phase']))
        applied = raw['applied_command'][previous]+np.clip(np.asarray(entry['requested_command'], np.float32)
            -raw['applied_command'][previous], [-.25, 0., -.35], [.25, 0., .35])
        np.testing.assert_allclose(raw['applied_command'][sl], np.tile(applied, (50, 1)), atol=1e-7, rtol=0)
        previous = end
    assert previous == len(raw['timestamp_s'])-1
    periods = np.diff([d['start_perf_counter_ns'] for d in decisions])/1e6
    tail = raw['base_twist_world'][-100:]
    return dict(active_ticks=218, zero_tail_ticks=5, every_physics_sample_accounted=True,
        full_cycle_wall_ms=dict(minimum=float(periods.min()), median=float(np.median(periods)),
            maximum=float(periods.max()), exceeding100ms=int((periods > 100).sum()), samples=len(periods)),
        tail_last200ms_max_xy_speed_m_s=float(np.linalg.norm(tail[:, :2], axis=1).max()),
        tail_last200ms_max_yaw_speed_rad_s=float(np.abs(tail[:, 5]).max()),
        tail_displacement_m=float(np.linalg.norm(raw['base_pose_world'][-1, :3]-raw['base_pose_world'][-251, :3])))


def audit():
    cv2.setNumThreads(1)
    own = {n: digest(ROOT/n) for n in SOURCES}
    bound = {str((OUTPUT/(n+'.json')).relative_to(ROOT)): h for n, h in IDENTITIES.items()}
    verify_bindings(bound | own)
    launch, result = [read_json(OUTPUT, n+'.json') for n in ('launch', 'result')]
    verify(launch); exact(preflight(), launch)
    spec, mission = specification(), result['mission']
    exact(launch['specification'], spec)
    assert set(result['artifact_sha256']) == {'mission/'+n for n in artifact_names(mission)}
    bound |= {str((OUTPUT/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    verify_bindings(bound)
    directory = OUTPUT/'mission'
    exact(read_json(directory, 'result.json'), mission)
    exact(read_json(directory, 'specification.json'), spec)
    decisions, tails = [read_json(directory, n) for n in ('task_decisions.json', 'tail_observations.json')]
    velocity, region = priors(launch['source_sha256'][PROTOCOL])
    controller = DepthProposalNavigation(ArticulatedCollisionGeometry(URDF), memory_arm=spec['memory_arm'],
                                         prior=velocity, hypotheses=POINT_HYPOTHESES)
    # No evaluator raw poses, contacts, cameras or scene layout enter this replay.
    for i, saved in enumerate(decisions):
        policy, depth = load_rgbd_observation(directory, i)
        fast = load_fast_packet(directory, i); now = policy['sensor_state']['decision_ns']
        actual = controller.observe_rgbd(policy, fast, depth, now_ns=now)
        exact(actual, saved['controller'])
        assert controller._context is None and controller.regions.memory.pending is None
        if i % 20 == 0:
            print('EXACT_CONTROLLER_REPLAY '+str(i+1), flush=True)
    cameras = read_json(directory, 'camera_audit.json')
    diagnosis = diagnose(controller, policy, cameras)
    exact(controller.regions.clearance(policy, now_ns=now), decisions[-1]['controller']['child']['sampled_turn_volume'])
    print('TERMINAL_DIAGNOSIS '+json.dumps({k: v for k, v in diagnosis.items() if k not in ('views', 'bands')}), flush=True)
    for i, saved in enumerate(tails, len(decisions)):
        assert saved['observation_index'] == i and saved['failure'] is None and not saved['estimator_not_reinvoked']
        policy, depth = load_rgbd_observation(directory, i)
        now = policy['sensor_state']['decision_ns']; assert saved['measured_ns'] == now
        exact(controller.observe_stopping_tail(policy, load_fast_packet(directory, i), depth, now_ns=now), saved['observation'])
    exact(controller.ledgers(), read_json(directory, 'task_ledgers.json'))
    exact(controller.memory_snapshot(), read_json(directory, 'task_memory.json'))
    print('EXACT_CONTROLLER_AND_TAIL_REPLAY_PASS', flush=True)
    raw, contacts, topology, roles, cameras, relatives, geometry, sensor_report = audit_sensors(directory, spec, mission)
    assert all(r['within1mm'] for r in sensor_report['depth_checks'])
    print('RAW_SENSOR_CONTACT_DEPTH_RECONSTRUCTION_PASS', flush=True)
    tape = read_json(directory, 'command_tape.json')
    command_report = commands(raw, decisions, tape, cameras)
    setup = read_json(directory, 'setup_checks.json'); native = read_json(directory, 'startup_native_robot_geometry.json')
    terminal_native = read_json(directory, 'terminal_native_robot_geometry.json')
    pose = raw['base_pose_world'][749]; q = raw['joint_position'][749]; R = rotation_xyzw(pose[3:])
    feet = match_native_foot_geometries(native, geometry, q, pose); exact(feet, setup['feet'])
    match_native_foot_geometries(terminal_native, geometry, raw['joint_position'][-1], raw['base_pose_world'][-1])
    exact(native_materials(native), native_materials(terminal_native))
    exact(read_json(directory, 'actuator_identity.json')['effective'], read_json(directory, 'terminal_actuator_gains.json'))
    check = check_setup_snapshot(velocity, region, identity=(0, 0, 0), measured_ns=1_500_000_000,
        position_world_m=pose[:3], rotation_world_from_initial_body=R, velocity_world_m_s=raw['base_twist_world'][749, :3],
        native_static_boxes=read_json(directory, 'static_objects.json'),
        expected_nonfloor_names=tuple(b['wall_id'] for b in spec['geometry']['wall_boxes']), geometry=geometry, joint_position=q)
    exact(check, setup['setup']); assert check['velocity_and_nonfloor_setup_checks_pass']
    support = initial_ground_support_witness(classify(contact_packet(contacts, 749), topology),
        expected_support_groups=['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'], ground_link_ids=sorted(topology['ground_link_ids']),
        geometry=geometry, joint_position=q, position_world_m=pose[:3], rotation_world_from_body=R)
    exact(support, setup['support']); assert support['initial_native_support_witness_present']
    guard_rows = read_json(directory, 'native_guard_rows.json')
    assert len(guard_rows) == len(raw['timestamp_s'])-750
    for i, saved in enumerate(guard_rows, 750):
        violations = nonfoot_ground_contact_indices(contact_packet(contacts, i), robot_geom_ids=[r['geom_id'] for r in native],
            foot_geom_ids=sorted(feet['native_foot_geom_to_shape']), ground_geom_ids=roles['physical_ground_geom_ids'])
        speed = float(np.linalg.norm(raw['base_twist_world'][i, :3]))
        exact(dict(sample_index=i, measured_ns=int(round(raw['timestamp_s'][i]*1e9)),
                   nonfoot_ground_contact_indices=violations, base_speed_m_s=speed,
                   evaluator_supervision_not_policy_input=True), saved)
        assert not violations and speed <= .3
    visibility = read_json(directory, 'marker_visibility_assay.json')
    exact(marker_pixel_pairs(np.asarray(Image.open(directory/'marker_visibility_assay.png'))), visibility['marker_pairs'])
    assert visibility['marker_pairs'] and visibility['clocks_before_after'][0] == visibility['clocks_before_after'][1]
    assert not any(visibility[k] for k in ('controller_received_image', 'body_mounted_sensor', 'physics_advanced'))
    occluded = marker_centres_occluded(spec, np.asarray(cameras[0]['world_from_optical'])[:3, 3]); assert occluded
    response = reduce_whole_task(raw, 749, decisions, terminal=controller.status, stop_reason=None,
                                sensor_fault=None, initial_marker_occluded=occluded)
    response['tail_sensor_fault'] = False
    exact(response, mission['response'])
    # Independent direct physical endpoints, not a controller home claim.
    displacement = np.linalg.norm(raw['base_pose_world'][-1, :2]-raw['base_pose_world'][749, :2])
    assert displacement > 2 and not raw['physics_contact'].any() and not response['physical_task_success']
    verify(launch); exact(preflight(), launch); verify_bindings(bound | own)
    return dict(status='FRESH_MAZE_EXACT_REPLAY_RAW_AUDIT_PASS_MISSION_FAILED', auditor_source_sha256=own,
        identity_sha256=IDENTITIES, verified_sources=len(launch['source_sha256']), verified_inputs=len(launch['input_sha256']),
        verified_artifacts=len(result['artifact_sha256']), exact_controller_frames=len(decisions), exact_tail_frames=len(tails),
        sensor_audit=sensor_report, commands=command_report, terminal_turn_diagnosis=diagnosis,
        physical_response=response, learned_navigation_policy=False, jepa_contribution_tested=False,
        hardware_qualified=False, navigation_qualified=False)


if __name__ == '__main__':
    target = OUTPUT/'raw_artifact_audit.json'
    if target.exists() or target.is_symlink():
        raise ValueError('new independent audit only; no overwrite')
    report = audit()
    write_json(target, report)
    print(report['status'], flush=True)
