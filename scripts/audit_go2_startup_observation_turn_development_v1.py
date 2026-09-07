"""Independent raw replay and bounded outcome accounting; never runs physics."""
import json

import numpy as np

from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries, nonfoot_ground_contact_indices
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot, initial_ground_support_witness
from lewm.startup_observation_turn_development import StartupObservationTurn, MAX_BASE_SPEED_M_S
from lewm_genesis.floor_extent_precision_development import check_extent_identity
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.audit_go2_bounded_floor_robot_interface_native_box_reader_development_v1 import same, require
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_startup_observation_turn_development_v1 import OUTPUT, PROTOCOL, specification, verify_extensions, artifact_names
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_observation_turn_session_development import make_priors
from scripts.startup_raw_sensor_audit_development import read_json, audit_sensors, contact_packet, classify
from scripts.startup_source_inventory_development import allowed_relative


def json_same(expected, actual):
    same(json.loads(json.dumps(expected, allow_nan=False)), actual)


def audit_commands(raw, tape, decisions, result):
    """Account for every sample, partial final tick, decision and zero tail."""
    n = len(raw['timestamp_s']); require(750 <= n <= 1900, 'bounded physical population')
    require(len(tape) == result['requested_ticks'] and len(decisions) == result['controller_decisions'] <= 21, 'bounded action records')
    same(raw['requested_command'][:750], np.zeros((750, 3)))
    same(raw['applied_command'][:750], np.zeros((750, 3))); same(raw['phase'][:750], np.zeros(750))
    same(raw['post_slew_applied_command'], raw['applied_command'])
    active = [e for e in tape if e['phase'] == 1]; tail = [e for e in tape if e['phase'] == 2]
    require(len(active) <= 20 and len(tail) <= 3 and tape == active+tail, 'controller ticks followed only by tail')
    terminal = False
    for i, item in enumerate(decisions):
        row = item['decision']; require(not terminal, 'no decision after terminal')
        same(item['observation_index'], i); same(row['decision_ns'], 1_500_000_000+i*100_000_000)
        terminal = row['terminal']
        if terminal:
            same(row['requested_command'], [0., 0., 0.]); require(i == len(active), 'terminal at command boundary')
        elif i < len(active):
            same(active[i]['decision_index'], i); same(active[i]['requested_command'], row['requested_command'])
        else:
            require(i == len(decisions)-1 and result['physical_stop_reason'] is not None, 'unexecuted decision requires recorded stop')
    require(not active or len(decisions) >= len(active), 'all actions have decisions')
    require(not tail or terminal, 'stopping tail requires terminal controller')
    previous = 749; completed_tail = 0
    for i, entry in enumerate(tape):
        same(entry['tick'], i); same(entry['pre_sample_index'], previous)
        end = entry['post_sample_index']; size = end-previous
        require(type(end) is int and 0 <= size <= 50 and end < n, 'bounded tick sample interval')
        require(size == 50 or (i == len(tape)-1 and result['physical_stop_reason'] is not None), 'partial tick only at physical stop')
        require(np.isfinite(entry['execution_wall_ms']) and entry['execution_wall_ms'] >= 0, 'finite execution timing')
        if entry['phase'] == 2:
            same(entry['decision_index'], None); same(entry['requested_command'], [0., 0., 0.])
            completed_tail += int(size == 50)
        if size:
            sl = slice(previous+1, end+1)
            same(raw['requested_command'][sl], np.tile(entry['requested_command'], (size, 1)))
            same(raw['phase'][sl], np.full(size, entry['phase']))
            applied = raw['applied_command'][previous] + np.clip(
                np.asarray(entry['requested_command'], np.float32)-raw['applied_command'][previous],
                [-.25, 0., -.35], [.25, 0., .35])
            np.testing.assert_allclose(raw['applied_command'][sl], np.tile(applied, (size, 1)), atol=1e-7, rtol=0)
        previous = end
    require(previous == n-1, 'no unaccounted physics')
    same(completed_tail, result['stopping_tail_ticks'])
    if result['physical_stop_reason'] is None:
        require(terminal and completed_tail == 3, 'unstopped acquisition requires terminal and complete real tail')
    same(result['controller_status'], decisions[-1]['decision']['status'] if decisions else None)
    return dict(active_ticks=len(active), complete_zero_tail_ticks=completed_tail,
                terminal_recorded=bool(terminal), command_reconstruction_pass=True)


def stopping_metrics(raw, tape):
    tail = [e for e in tape if e['phase'] == 2]
    if not tail: return dict(complete=False, final_quiet_window=False)
    start = tail[0]['pre_sample_index']; end = tail[-1]['post_sample_index']
    complete = len(tail) == 3 and end-start == 150
    # Require the entire final 100 ms to be quiet, not just a favourable endpoint.
    window = raw['base_twist_world'][max(start+1, end-49):end+1]
    if not len(window): return dict(complete=False, final_quiet_window=False)
    speed = np.linalg.norm(window[:, :3], axis=1); angular = np.linalg.norm(window[:, 3:], axis=1)
    return dict(complete=bool(complete), final_quiet_window=bool(complete and np.max(speed) <= .05 and np.max(angular) <= .1),
        final_window_max_speed_m_s=float(np.max(speed)), final_window_max_angular_speed_rad_s=float(np.max(angular)),
        stopping_displacement_m=float(np.linalg.norm(raw['base_pose_world'][end, :3]-raw['base_pose_world'][start, :3])))


def native_materials(rows):
    fields = ('geom_id', 'link_id', 'link_name', 'geom_type', 'data', 'friction', 'solver_parameters')
    return sorted(({k: r[k] for k in fields} for r in rows), key=lambda r: r['geom_id'])


def padded_body_inside_setup(geometry, region, initial_pose, pose, joints):
    rotation = rotation_xyzw(initial_pose[3:])
    position = rotation.T @ (np.asarray(pose[:3])-initial_pose[:3])
    shapes = geometry.supports(joints, rotation.T @ rotation_xyzw(pose[3:]))['shapes']
    return all(np.all(s['lower']+position-.04 > np.asarray(region.lower_initial_body_m))
               and np.all(s['upper']+position+.04 < np.asarray(region.upper_initial_body_m)) for s in shapes)


def audit():
    launch = read_json(OUTPUT, 'launch.json'); result = read_json(OUTPUT, 'result.json')
    require(result['status'] == 'ACQUISITION_COMPLETE_AUDIT_REQUIRED', 'auditable acquisition required')
    same(launch['specification'], specification())
    expected = set(artifact_names(result['rgbd_frames']))
    require(set(result['artifact_sha256']) <= expected and set(result['absent_expected_artifacts']) == expected-set(result['artifact_sha256']), 'exact artifact inventory accounting')
    for name in result['artifact_sha256']: allowed_relative(name)
    bindings = launch['source_sha256'] | launch['input_sha256'] | {
        str((OUTPUT / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    # Bind launch/result themselves throughout replay, as well as their children.
    bindings |= {str((OUTPUT / p).relative_to(ROOT)): digest(OUTPUT / p) for p in ('launch.json', 'result.json')}
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    raw, contacts, topology, roles, cameras, relatives, geometry, sensor_report = audit_sensors(OUTPUT, specification(), result)
    decisions = read_json(OUTPUT, 'startup_decisions.json'); tape = read_json(OUTPUT, 'startup_command_tape.json')
    command_report = audit_commands(raw, tape, decisions, result)
    gains = read_json(OUTPUT, 'actuator_identity.json'); same(gains['effective'], read_json(OUTPUT, 'terminal_actuator_gains.json'))
    check_extent_identity(read_json(OUTPUT, 'startup_floor_identity.json'), 32.)
    same(read_json(OUTPUT, 'startup_floor_identity.json'), read_json(OUTPUT, 'floor_visual_collision_identity.json'))
    initial = raw['base_pose_world'][749]; rotation = rotation_xyzw(initial[3:]); q = raw['joint_position'][749]
    definition = launch['source_sha256'][PROTOCOL]; velocity, region = make_priors(1_500_000_000, definition)
    native = read_json(OUTPUT, 'startup_native_robot_geometry.json')
    feet = match_native_foot_geometries(native, geometry, q, initial)
    setup = check_setup_snapshot(velocity, region, identity=(0, 0, 0), measured_ns=velocity.anchor_ns,
        position_world_m=initial[:3], rotation_world_from_initial_body=rotation,
        velocity_world_m_s=raw['base_twist_world'][749, :3], native_static_boxes=read_json(OUTPUT, 'static_objects.json'),
        expected_nonfloor_names=tuple(v for k, v in topology['environment_object_ids'].items() if int(k) not in topology['ground_link_ids']),
        geometry=geometry, joint_position=q)
    support = initial_ground_support_witness(classify(contact_packet(contacts, 749), topology),
        expected_support_groups=['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'], ground_link_ids=topology['ground_link_ids'],
        geometry=geometry, joint_position=q, position_world_m=initial[:3], rotation_world_from_body=rotation)
    checks = dict(setup=setup, support=support, feet=feet, sample_index=749, definition_sha256=definition,
                  native_plane_identity_verified=True, scope='evaluator-only initial setup; not runtime sensor data')
    json_same(checks, read_json(OUTPUT, 'startup_checks.json'))
    require(setup['velocity_and_nonfloor_setup_checks_pass'] and support['initial_native_support_witness_present'], 'admitted actual setup')
    admission = dict(schema='startup_setup_admission_development.v1', identity=(0, 0, 0), anchor_ns=velocity.anchor_ns,
        definition_sha256=definition, checks_sha256=digest(OUTPUT / 'startup_checks.json'),
        velocity_and_nonfloor_checks_pass=True, initial_native_support_witness_present=True)
    json_same(admission, read_json(OUTPUT, 'startup_admission.json'))
    guard = dict(robot_geom_ids=[r['geom_id'] for r in native], foot_geom_ids=sorted(feet['native_foot_geom_to_shape']),
                 ground_geom_ids=roles['physical_ground_geom_ids'])
    guards = []; violations = []; unstable = []; region_violations = []
    for i in range(750, len(raw['timestamp_s'])):
        speed = float(np.linalg.norm(raw['base_twist_world'][i, :3]))
        bad = nonfoot_ground_contact_indices(contact_packet(contacts, i), **guard)
        R = rotation_xyzw(raw['base_pose_world'][i, 3:])
        if not padded_body_inside_setup(geometry, region, initial, raw['base_pose_world'][i], raw['joint_position'][i]):
            region_violations.append(i)
        body_bad = raw['base_pose_world'][i, 2] < .15 or max(abs(np.arctan2(R[2,1], R[2,2])), abs(np.arcsin(np.clip(R[2,0], -1, 1)))) > .70
        if body_bad: unstable.append(i)
        if bad or speed > MAX_BASE_SPEED_M_S: violations.append(i)
        # Inherited guard raises first; additional guard cannot log that final row.
        if not (body_bad or raw['physics_contact'][i]):
            guards.append(dict(sample_index=i, measured_ns=int(round(raw['timestamp_s'][i]*1e9)),
                base_speed_m_s=speed, nonfoot_ground_contact_indices=bad))
    same(guards, read_json(OUTPUT, 'startup_guard_rows.json'))
    all_stops = sorted(set(violations+unstable+np.flatnonzero(raw['physics_contact']).tolist()))
    require(not all_stops or (all_stops[0] == len(raw['timestamp_s'])-1 and result['physical_stop_reason'] is not None), 'no execution after raw physical stop')
    if all_stops:
        i = all_stops[0]
        if raw['physics_contact'][i]: reason = 'DISALLOWED_CONTACT'
        elif i in unstable: reason = 'BODY_STABILITY_LIMIT'
        elif nonfoot_ground_contact_indices(contact_packet(contacts, i), **guard): reason = 'STARTUP_NONFOOT_GROUND_CONTACT'
        else: reason = 'STARTUP_BASE_SPEED_ASSUMPTION_VIOLATED'
        same(result['physical_stop_reason'], reason)
    else:
        require(result['physical_stop_reason'] not in ('DISALLOWED_CONTACT', 'BODY_STABILITY_LIMIT',
            'STARTUP_NONFOOT_GROUND_CONTACT', 'STARTUP_BASE_SPEED_ASSUMPTION_VIOLATED'), 'physical stop needs raw cause')
    material_verified = False
    if 'terminal_native_robot_geometry.json' in result['artifact_sha256']:
        terminal_native = read_json(OUTPUT, 'terminal_native_robot_geometry.json')
        json_same(match_native_foot_geometries(terminal_native, geometry, raw['joint_position'][-1], raw['base_pose_world'][-1]),
                  read_json(OUTPUT, 'terminal_foot_identity.json'))
        same(native_materials(native), native_materials(terminal_native)); material_verified = True
    controller = StartupObservationTurn(geometry, velocity_prior=velocity, region_prior=region, admission=admission)
    envelope_rows = []
    for item in decisions:
        frame = item['observation_index']; policy, depth = load_rgbd_observation(OUTPUT, frame)
        now = policy['sensor_state']['decision_ns']
        row = controller.observe(policy, depth, relatives[frame]['observer'], now_ns=now)
        json_same(row, item['decision'])
        if 'command_plus_stop_extent_m' not in row: continue
        centre = np.asarray(controller.memory._rays.fusion['position_initial_body_m'])
        start = cameras[frame]['physical_sample_index']
        end = min(len(raw['timestamp_s'])-1, start+200)
        indices = np.arange(start, end+1)
        actual = (raw['base_pose_world'][indices, :3]-initial[:3]) @ rotation
        dt = raw['timestamp_s'][indices]-raw['timestamp_s'][start]
        # Evaluate actual motion against the current proxy plus explicit speed cap.
        excess = np.linalg.norm(actual-centre, axis=1)-(row['combined_pose_scale_m']+MAX_BASE_SPEED_M_S*dt)
        envelope_rows.append(dict(observation_index=frame, sampled_through_ns=int(round(raw['timestamp_s'][end]*1e9)),
            full_horizon_recorded=bool(end == start+200), max_position_envelope_excess_m=float(excess.max()),
            sampled_motion_within_envelope=bool(excess.max() <= 1e-12)))
    stop = stopping_metrics(raw, tape)
    timings = read_json(OUTPUT, 'startup_timings.json')
    same([r['observation_index'] for r in timings['captures']], list(range(len(cameras))))
    same([r['observation_index'] for r in timings['controller']], [r['observation_index'] for r in decisions])
    for rows, key in ((timings['captures'], 'acquisition_and_depth_observer_ms'), (timings['controller'], 'controller_wall_ms')):
        require(all(np.isfinite(r[key]) and r[key] >= 0 for r in rows), 'finite complete timing records')
    require(np.isfinite(result['total_acquisition_wall_ms']) and result['total_acquisition_wall_ms'] > 0, 'whole collector wall time')
    within_setup_time = raw['timestamp_s'][-1]*1e9 <= region.valid_until_ns
    success = bool(controller.status == 'COMPLETE_OBSERVATION_TURN' and result['physical_stop_reason'] is None
        and not result['absent_expected_artifacts'] and stop['final_quiet_window'] and not all_stops and material_verified
        and not region_violations and within_setup_time
        and all(r['within1mm'] for r in sensor_report['depth_checks'])
        and all(r['sampled_motion_within_envelope'] for r in envelope_rows))
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    return dict(status='RAW_REPLAY_COMPLETE', local_observation_turn_success=success,
        controller_status=controller.status, first_rank3_ns=controller.first_rank3_ns,
        physical_stop_reason=result['physical_stop_reason'], physics_samples=len(raw['timestamp_s']), rgbd_frames=len(cameras),
        **command_report, **sensor_report, setup_reconstruction_exact=True, controller_reconstruction_exact=True,
        terminal_material_identity_verified=material_verified, physical_stop_sample_indices=all_stops,
        padded_body_outside_setup_sample_indices=region_violations, complete_trace_within_setup_expiry=bool(within_setup_time),
        envelope_diagnostics=envelope_rows, stopping=stop, wall_times=timings,
        total_acquisition_wall_ms=result['total_acquisition_wall_ms'],
        execution_wall_ms=[r['execution_wall_ms'] for r in tape],
        scope='one conditional development observation turn; not maze, calibration or hardware evidence',
        future_dynamics_validated=False, contact_model_validated=False, real_time_qualified=False, navigation_qualified=False)


def main():
    target = OUTPUT / 'raw_artifact_audit.json'
    require(not target.exists(), 'one-shot audit output; no overwrite')
    report = audit(); write_json(target, report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__': main()
