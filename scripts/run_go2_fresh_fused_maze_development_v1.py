"""One fresh closed-loop complete-maze development attempt; no resume/retry."""
import json
import shutil
import time

import cv2
import numpy as np

from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.depth_proposal_navigation_development import DepthProposalNavigation
from lewm.fresh_fused_maze_scene_development import specification, pack
from lewm.rgbd_shadow_motion_development import POINT_HYPOTHESES
from lewm.rgb_marker_beacon_development import observe_marker
from lewm.whole_task_metrics_development import marker_centres_occluded, reduce_whole_task
from lewm.whole_task_navigation_development import MAX_SECONDS
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.fresh_maze_session_development import MissionRGBDSession, admit_setup, marker_visibility_assay, validate_command
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.probe_go2_depth_proposal_navigation_interface_development_v1 import OUTPUT as PREVIOUS
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_observation_turn_session_development import latest_policy
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = ROOT/'.generated/go2_fresh_fused_maze_development_v1_attempt_001'
PROTOCOL = 'docs/go2_fresh_fused_maze_development_v1_2026-09-06.md'
SEEDS = (PROTOCOL, 'scripts/run_go2_fresh_fused_maze_development_v1.py',
         'lewm/tests/test_fresh_fused_maze_execution_development.py')
IDENTITIES = {'launch.json': 'd39e01184f1bc3526b0a32859ce2b6e9f2fa7f9b650643b1c47952ed8c324ea3',
    'result.json': '434700298aab6afc7001d9247d3e708e0a3ff90628c51a3ef26ab0a75adb0e03',
    'interface_audit.json': '72f408c118e028ea09ae9c97f4adff2486a60486dfe9a2e71e66cd76b898f79f'}


def preflight():
    bindings = {str((PREVIOUS/n).relative_to(ROOT)): h for n, h in IDENTITIES.items()}
    verify_bindings(bindings)
    old = read_json(PREVIOUS, 'launch.json')
    verify(old)
    previous = read_json(PREVIOUS, 'result.json')
    bindings |= {str((PREVIOUS/n).relative_to(ROOT)): h for n, h in previous['artifact_sha256'].items()}
    sources = discover_sources(SEEDS, old['source_sha256'])
    launch = old | dict(source_sha256=sources, input_sha256=old['input_sha256'] | bindings,
        specification=specification(), maximum_mission_seconds=MAX_SECONDS, terminal_zero_ticks=5,
        scope='one actual sensor-controlled fresh maze; externally guarded simulation, no future-gait/hardware qualification')
    verify(launch)
    return launch


def chain(error):
    result = []
    while error is not None:
        result.append(str(error))
        error = error.__cause__
    return result


def collect(spec, definition):
    output = OUTPUT/'mission'
    output.mkdir()
    write_json(output/'specification.json', spec)
    session = controller = None
    decisions, tape, tails = [], [], []
    stop = fault = terminal = None
    initial_occluded = False
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    try:
        session = MissionRGBDSession(spec, output)
        session.install_contact_identity()
        build = session.ctx.build
        gains = configure_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
        write_json(output/'actuator_identity.json', gains)
        write_json(output/'floor_roles.json', dict(
            physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        marker_visibility_assay(session)
        try:
            session.settle_recorded()
            velocity = admit_setup(session, definition)
            controller = DepthProposalNavigation(ArticulatedCollisionGeometry(URDF), memory_arm=spec['memory_arm'],
                                                  prior=velocity, hypotheses=POINT_HYPOTHESES)
            for tick in range(MAX_SECONDS*10+1):
                start_perf = time.perf_counter_ns()
                index = session.capture_current()
                policy = latest_policy(session)
                now = policy['sensor_state']['decision_ns']
                if tick == 0:
                    initial_occluded = marker_centres_occluded(spec, np.asarray(session.image_audit[0]['world_from_optical'])[:3, 3])
                    if not initial_occluded or observe_marker(policy, now_ns=now)['detections']:
                        raise ValueError('fresh mission must start with an actually hidden marker')
                row = dict(tick=tick, observation_index=index, pre_sample_index=len(session.samples)-1,
                           decision_ns=now, controller=None, failure=None, executed=False,
                           start_perf_counter_ns=start_perf)
                decisions.append(row)
                try:
                    row['controller'] = controller.observe_rgbd(policy, session.fast_buffer.packet(now_ns=now),
                                                                session.latest_depth, now_ns=now)
                except SensorContractError as error:
                    row['failure'] = chain(error)
                    fault = dict(tick=tick, observation_index=index, pre_sample_index=len(session.samples)-1,
                                 reason=row['failure'], decision_ns=now)
                    terminal = 'FAILED_SENSOR'
                try:
                    if fault is not None or row['controller']['terminal']:
                        terminal = terminal or row['controller']['status']
                        break
                    command = validate_command(row['controller']['requested_command'])
                    if tick == MAX_SECONDS*10:
                        raise ValueError('controller failed to terminate at mission budget')
                    item = dict(phase=1, decision_tick=tick, pre_sample_index=len(session.samples)-1,
                                requested_command=command, completed=False)
                    tape.append(item)
                    session.phase = 1
                    row['executed'] = True  # Started; completion is separately recorded.
                    try:
                        session.command_tick(command)
                        item['completed'] = True
                    finally:
                        item['post_sample_index'] = len(session.samples)-1
                finally:
                    row['end_perf_counter_ns'] = time.perf_counter_ns()
                    row['outer_wall_ms'] = (row['end_perf_counter_ns']-start_perf)/1e6
                if tick % 10 == 0:
                    print(json.dumps(dict(tick=tick, decision_ns=now, stage=controller.stage,
                        command=command, position_scale_m=row['controller']['sensor_fusion']['position_error_scale_m'])), flush=True)
            for tick in range(5):
                start_perf = time.perf_counter_ns()
                item = dict(phase=2, tail_tick=tick, pre_sample_index=len(session.samples)-1,
                            requested_command=[0., 0., 0.], completed=False)
                tape.append(item)
                session.phase = 2
                try:
                    session.command_tick([0., 0., 0.])
                    item['completed'] = True
                    index = session.capture_current()
                    policy = latest_policy(session)
                    now = policy['sensor_state']['decision_ns']
                    tail = dict(observation_index=index, measured_ns=now, observation=None,
                                failure=None, estimator_not_reinvoked=controller.sensor_memory.failed)
                    if not controller.sensor_memory.failed:
                        try:
                            tail['observation'] = controller.observe_stopping_tail(policy,
                                session.fast_buffer.packet(now_ns=now), session.latest_depth, now_ns=now)
                        except SensorContractError as error:
                            tail['failure'] = chain(error)
                    tails.append(tail)
                finally:
                    item['post_sample_index'] = len(session.samples)-1
                    item['outer_wall_ms'] = (time.perf_counter_ns()-start_perf)/1e6
        except PhysicalStop as error:
            stop = str(error)
            terminal = 'PHYSICAL_STOP'
            if controller is not None:
                controller.finish_physical_stop(now_ns=int(round(session.samples[-1]['timestamp_s']*1e9)))
        terminal_gains = read_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist())
        if terminal_gains != gains['effective']:
            raise ValueError('actuator gains changed')
        write_json(output/'terminal_actuator_gains.json', terminal_gains)
        write_json(output/'terminal_native_robot_geometry.json', capture_native_robot_geometry(build.robot))
        write_json(output/'terminal_environment_identity.json', appearance_environment_identity(session))
        raw = {k: np.stack([r[k] for r in session.samples]) for k in session.samples[0]}
        valid_decisions = [r for r in decisions if r['controller'] is not None]
        if valid_decisions:
            response = reduce_whole_task(raw, 749, valid_decisions, terminal=terminal or controller.status,
                stop_reason=stop, sensor_fault=fault, initial_marker_occluded=initial_occluded)
            response['tail_sensor_fault'] = any(r['failure'] is not None for r in tails)
            response['physical_task_success'] &= not response['tail_sensor_fault']
        else:
            response = dict(physical_task_success=False, controller_terminal=terminal,
                            stop_reason=stop, sensor_fault=fault, completed_setup_and_decision=False)
        result = dict(status='FRESH_MAZE_ATTEMPT_TERMINAL_AUDIT_REQUIRED', response=response,
            physics_samples=len(session.samples), rgbd_frames=len(session.model_manifest),
            control_decisions=len(decisions), terminal_zero_ticks=sum(t['phase']==2 and t['completed'] for t in tape),
            initial_marker_centres_occluded=initial_occluded, controller_selects_commands=True,
            learned_navigation_policy=False, future_gait_envelope_validated=False,
            external_native_supervision=True, hardware_qualified=False, navigation_qualified=False)
        write_json(output/'result.json', result)
        return result
    finally:
        if session is not None:
            try:
                session.persist(output)
                session.persist_observations(output)
                write_json(output/'task_decisions.json', decisions)
                write_json(output/'command_tape.json', tape)
                write_json(output/'tail_observations.json', tails)
                write_json(output/'native_guard_rows.json', session.guard_rows)
                write_json(output/'task_ledgers.json', None if controller is None else controller.ledgers())
                write_json(output/'task_memory.json', None if controller is None else controller.memory_snapshot())
            finally:
                session.ctx.build.scene.destroy()
        shutdown_genesis()


def artifact_names(result):
    fixed = ['specification.json', 'actuator_identity.json', 'floor_roles.json', 'terminal_actuator_gains.json',
        'terminal_native_robot_geometry.json', 'terminal_environment_identity.json', 'result.json',
        'physics_trace.npz', 'native_contacts.npz', 'contact_events.json', 'contact_topology.json',
        'ideal_sensor_samples.npz', 'policy_histories.npz', 'policy_observations.json', 'camera_audit.json',
        'depth_observations.json', 'depth_camera_audit.json', 'fast_gyro_samples.npz', 'fast_gyro_histories.npz',
        'floor_visual_collision_identity.json', 'static_objects.json', 'startup_native_robot_geometry.json', 'setup_checks.json',
        'task_decisions.json', 'command_tape.json', 'tail_observations.json', 'native_guard_rows.json',
        'task_ledgers.json', 'task_memory.json', 'marker_visibility_assay.png', 'marker_visibility_assay.json']
    return fixed + [f'{prefix}_{i:04d}.{suffix}' for i in range(result['rgbd_frames'])
        for prefix, suffix in [('rgb', 'png'), ('depth', 'npz'), ('native_depth', 'npz')]] + [
        f'visual_meshes/{name}_visual.ply' for name in ['ground', *[o.object_id for o in pack(specification()).static_objects]]]


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('fresh fixed output required; no resume/retry')
    if shutil.disk_usage(OUTPUT.parent).free < 10*1024**3:
        raise ValueError('at least ten GiB free required')
    cv2.setNumThreads(1)
    launch = preflight()
    OUTPUT.mkdir()
    write_json(OUTPUT/'launch.json', launch)
    try:
        result = collect(specification(), launch['source_sha256'][PROTOCOL])
        verify(launch)
        names = ['mission/'+n for n in artifact_names(result)]
        if any(not (OUTPUT/n).is_file() for n in names):
            raise ValueError('missing declared mission artifact')
        final = dict(status='FRESH_CLOSED_LOOP_MAZE_ATTEMPT_COMPLETE_AUDIT_REQUIRED', mission=result,
                     artifact_sha256={n:digest(OUTPUT/n) for n in names}, independent_layout_trials=1,
                     navigation_qualified=False)
        write_json(OUTPUT/'result.json', final)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_FRESH_MAZE_INFRASTRUCTURE_FAILURE', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
