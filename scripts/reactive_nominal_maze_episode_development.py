"""Fresh observed-controller maze execution with streamed complete decisions."""
import shutil
import time
import json
from lewm.reactive_nominal_round_trip_controller_development import ReactiveNominalRoundTripController
from lewm.novel_maze_round_trip_scene_development import specification, public_mission
from lewm.novel_maze_round_trip_contract_development import (
    WARMUP_TICKS, DRAIN_TICKS, NAVIGATION_TICKS, MAX_OBSERVATIONS,
    RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES)
from lewm.reactive_nominal_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES
from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.support_friction_challenge_development import native_friction
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.novel_maze_round_trip_session_development import NovelMazeRoundTripSession
from scripts.geometry_progress_family_episode_development import artifacts as primary_artifacts
from scripts.maze_decision_stream_development import writer, NAME as DECISIONS
from scripts.pulse_context_setup_development import admit_context_setup
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import write_json
from scripts.navigation_artifact_root_development import BASE, validate_root


def collect(layout_index, definition, *, output, geometry, episode_name):
    validate_root(output); directory = output/episode_name; directory.mkdir()
    spec = specification(layout_index); mission = public_mission(layout_index)
    write_json(directory/'specification.json', spec); write_json(directory/'public_mission.json', mission)
    session = None; tape = []; friction = []; decisions = 0; latest_mission = None
    stop = acquisition_stop = terminal = None; admitted = False; drain = 0
    initial_free = shutil.disk_usage(BASE).free
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    try:
        session = NovelMazeRoundTripSession(spec, directory); session.install_contact_identity()
        build = session.ctx.build
        gains = configure_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
        write_json(directory/'actuator_identity.json', gains)
        write_json(directory/'floor_roles.json', dict(
            physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        friction.append(dict(stage='before_settle', **native_friction(build, spec['friction_mu'])))
        controller = ReactiveNominalRoundTripController(geometry, public_mission=mission,
            navigation_ticks=NAVIGATION_TICKS)
        with writer(directory) as raw_append, (directory/'decision_stream_timing.jsonl').open('x') as stream_timing:
            def append(row):
                before_write = time.perf_counter_ns()
                raw_append(row)
                after_write = time.perf_counter_ns()
                stream_timing.write(json.dumps(dict(tick=row['tick'],
                    decision_receipt_write_wall_ms=(after_write-before_write)/1e6,
                    iteration_with_receipt_wall_ms=(after_write-start)/1e6))+'\n')
                stream_timing.flush()
            try:
                session.settle_recorded(); session.capture_current()
                admit_context_setup(session, definition); admitted = True
                for tick in range(MAX_OBSERVATIONS):
                    free = shutil.disk_usage(BASE).free
                    if free < RESERVE_BYTES+PERSISTENCE_HEADROOM_BYTES or initial_free-free >= COLLECTION_ALLOWANCE_BYTES:
                        acquisition_stop = 'STORAGE_RESERVE_STOP'; break
                    friction.append(dict(stage='before_decision', tick=tick, **native_friction(build, spec['friction_mu'])))
                    start = time.perf_counter_ns()
                    try:
                        p, d, f, auxiliary, now = session.sensor_packets(); packet_ns = time.perf_counter_ns()
                        selected = controller.observe(p, d, f, now_ns=now, auxiliary_depth=auxiliary)
                    except (ValueError, TypeError, KeyError) as error:
                        acquisition_stop = 'PACKET_CONTRACT_STOP: '+str(error); break
                    controlled_ns = time.perf_counter_ns(); latest_mission = selected['mission_receipt']
                    row = dict(tick=tick, observation_index=len(session.model_manifest)-1,
                        pre_sample_index=len(session.samples)-1, decision=selected,
                        acquisition_wall_ms=(packet_ns-start)/1e6, controller_wall_ms=(controlled_ns-packet_ns)/1e6,
                        observation_and_control_wall_ms=(controlled_ns-start)/1e6, resource_free_bytes=free)
                    if selected['terminal'] is not None:
                        terminal = selected['terminal']
                        if drain == DRAIN_TICKS:
                            append(row); decisions += 1; break
                        phase, role = 3, 'terminal_zero_drain'
                    elif tick < WARMUP_TICKS:
                        phase, role = 1, 'causal_history_warmup'
                    else:
                        phase, role = 2, 'online_reactive_round_trip_command'
                    session.phase = phase
                    item = dict(tick=tick, requested_command=selected['requested_command'], phase=phase, role=role,
                        pre_sample_index=len(session.samples)-1, post_sample_index=None, completed=False)
                    tape.append(item)
                    try:
                        session.command_tick(item['requested_command']); item['completed'] = True
                        if terminal is not None: drain += 1
                    finally:
                        item['post_sample_index'] = len(session.samples)-1
                        row['iteration_with_command_wall_ms'] = (time.perf_counter_ns()-start)/1e6
                        append(row); decisions += 1
            except PhysicalStop as error:
                stop = str(error)
        friction.append(dict(stage='terminal', **native_friction(build, spec['friction_mu'])))
        terminal_gains = read_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist())
        if terminal_gains != gains['effective']: raise ValueError('frozen gait gains changed')
        write_json(directory/'terminal_actuator_gains.json', terminal_gains)
        write_json(directory/'terminal_native_robot_geometry.json', capture_native_robot_geometry(build.robot))
        write_json(directory/'terminal_environment_identity.json', appearance_environment_identity(session))
        result = dict(status='REACTIVE_NOMINAL_MAZE_TERMINAL_AUDIT_REQUIRED', layout_index=layout_index,
            physical_stop=stop, acquisition_stop=acquisition_stop, schedule_terminal=terminal,
            terminal_zero_ticks=drain, setup_admitted=admitted, setup_checked=(directory/'setup_checks.json').is_file(),
            departure_present=decisions>WARMUP_TICKS, command_ticks=len(tape), completed_ticks=sum(t['completed'] for t in tape),
            physics_samples=len(session.samples), rgbd_frames=len(session.model_manifest),
            auxiliary_frames=len(session.auxiliary_audit), decisions=decisions, mission_receipt=latest_mission,
            navigation_ticks=NAVIGATION_TICKS, tracker_required_for_commands=True,
            native_state_used_for_commands=False, navigation_qualified=False,
            storage_initial_free_bytes=initial_free, storage_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
            storage_persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES)
        write_json(directory/'result.json', result)
        print('REACTIVE_NOMINAL_MAZE_COLLECTED', result, flush=True)
        return result
    finally:
        if session is not None:
            try:
                session.persist(directory); session.persist_observations(directory)
                write_json(directory/'command_tape.json', tape)
                write_json(directory/'native_guard_rows.json', session.guard_rows)
                write_json(directory/'friction_checks.json', friction)
            finally: session.ctx.build.scene.destroy()
        shutdown_genesis()


def artifacts(layout_index, result):
    names = [DECISIONS if n == 'context_decisions.json' else n for n in primary_artifacts('', result)]
    return names+['public_mission.json', 'auxiliary_camera_audit.json', 'decision_stream_timing.jsonl']+[
        f'auxiliary_{kind}_{i:04d}.{suffix}' for i in range(result['auxiliary_frames'])
        for kind, suffix in (('depth', 'npz'), ('rgb', 'png'))]

