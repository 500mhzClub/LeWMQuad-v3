"""Fresh native mission with online model decisions and retained terminal drain."""
import shutil
import time
from lewm.observation_replan_goal_probe_development import ObservationReplanGoalProbe

from lewm.learned_goal_probe_development import WARMUP_TICKS, NAVIGATION_TICKS, DRAIN_TICKS
from lewm.geometry_progress_layout_family_development import specification
from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.support_friction_challenge_development import native_friction
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.geometry_progress_family_session_development import GeometryProgressFamilySession
from scripts.geometry_progress_family_episode_development import artifacts, RESERVE
from scripts.pulse_context_setup_development import admit_context_setup
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import write_json
from scripts.navigation_artifact_root_development import BASE, validate_root


def collect(trial, definition, *, output, model, geometry, persistent, episode_name, condition, variant):
    validate_root(output)
    directory = output / episode_name
    directory.mkdir()
    spec = specification(trial)
    write_json(directory / 'specification.json', spec)
    session = None
    rows, tape, friction = [], [], []
    stop = acquisition_stop = terminal = None
    admitted = False
    drain = 0
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    try:
        session = GeometryProgressFamilySession(spec, directory)
        session._runtime['high_level'] = 'observation_replan_goal_probe_v1; online model plus measured surfaces'
        session.install_contact_identity()
        build = session.ctx.build
        gains = configure_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
        write_json(directory / 'actuator_identity.json', gains)
        write_json(directory / 'floor_roles.json', dict(
            physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        friction.append(dict(stage='before_settle', **native_friction(build, spec['friction_mu'])))
        controller = ObservationReplanGoalProbe(model, geometry, persistent=persistent, condition=condition, variant=variant)
        try:
            session.settle_recorded()
            session.capture_current()
            admit_context_setup(session, definition)
            admitted = True
            for tick in range(WARMUP_TICKS + NAVIGATION_TICKS + DRAIN_TICKS + 1):
                free = shutil.disk_usage(BASE).free
                if free < RESERVE:
                    acquisition_stop = 'STORAGE_RESERVE_STOP'
                    break
                friction.append(dict(stage='before_decision', tick=tick, **native_friction(build, spec['friction_mu'])))
                start = time.perf_counter_ns()
                try:
                    p, d, f, now = session.sensor_packets()
                    packet_ns = time.perf_counter_ns()
                    selected = controller.observe(p, d, f, now_ns=now)
                except (ValueError, TypeError, KeyError) as error:
                    acquisition_stop = 'PACKET_CONTRACT_STOP: ' + str(error)
                    break
                controlled_ns = time.perf_counter_ns()
                row = dict(tick=tick, observation_index=len(session.model_manifest)-1,
                    pre_sample_index=len(session.samples)-1, decision=selected,
                    acquisition_wall_ms=(packet_ns-start)/1e6, controller_wall_ms=(controlled_ns-packet_ns)/1e6,
                    observation_and_control_wall_ms=(controlled_ns-start)/1e6, resource_free_bytes=free)
                rows.append(row)
                if selected['terminal'] is not None:
                    terminal = selected['terminal']
                    if drain == DRAIN_TICKS:
                        break
                    phase, role = 3, 'terminal_zero_drain'
                elif tick < WARMUP_TICKS:
                    phase, role = 1, 'causal_history_warmup'
                else:
                    phase, role = 2, 'online_learned_goal_command'
                session.phase = phase
                item = dict(tick=tick, requested_command=selected['requested_command'], phase=phase, role=role,
                    pre_sample_index=len(session.samples)-1, post_sample_index=None, completed=False)
                tape.append(item)
                try:
                    session.command_tick(item['requested_command'])
                    item['completed'] = True
                    if terminal is not None:
                        drain += 1
                finally:
                    item['post_sample_index'] = len(session.samples)-1
                    row['iteration_with_command_wall_ms'] = (time.perf_counter_ns()-start)/1e6
        except PhysicalStop as error:
            stop = str(error)
        friction.append(dict(stage='terminal', **native_friction(build, spec['friction_mu'])))
        terminal_gains = read_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist())
        if terminal_gains != gains['effective']:
            raise ValueError('frozen gait gains changed')
        write_json(directory / 'terminal_actuator_gains.json', terminal_gains)
        write_json(directory / 'terminal_native_robot_geometry.json', capture_native_robot_geometry(build.robot))
        write_json(directory / 'terminal_environment_identity.json', appearance_environment_identity(session))
        result = dict(status='OBSERVATION_REPLAN_GOAL_PROBE_TERMINAL_AUDIT_REQUIRED', trial=trial,
            physical_stop=stop, acquisition_stop=acquisition_stop, schedule_terminal=terminal,
            terminal_zero_ticks=drain, setup_admitted=admitted,
            setup_checked=(directory/'setup_checks.json').is_file(),
            departure_present=any(r['tick']==WARMUP_TICKS for r in rows),
            command_ticks=len(tape), completed_ticks=sum(t['completed'] for t in tape),
            physics_samples=len(session.samples), rgbd_frames=len(session.model_manifest), decisions=len(rows),
            tracker_required_for_commands=True, native_state_used_for_commands=False, navigation_qualified=False)
        write_json(directory / 'result.json', result)
        print('COMMITMENT_POSE_GOAL_PROBE', trial, result, flush=True)
        return result
    finally:
        if session is not None:
            try:
                session.persist(directory)
                session.persist_observations(directory)
                write_json(directory / 'context_decisions.json', rows)
                write_json(directory / 'command_tape.json', tape)
                write_json(directory / 'native_guard_rows.json', session.guard_rows)
                write_json(directory / 'friction_checks.json', friction)
            finally:
                session.ctx.build.scene.destroy()
        shutdown_genesis()

