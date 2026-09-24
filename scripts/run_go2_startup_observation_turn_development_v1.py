"""One fresh bounded startup observation action with exact recorded setup admission."""
import json
import time

import numpy as np

from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.causal_sensor_state import SensorContractError
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries
from lewm.startup_observation_turn_development import StartupObservationTurn
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.probe_go2_measured_plane_obstacle_memory_development import DIRECTORY as PREVIOUS, IDENTITIES
from scripts.run_go2_aligned_floor_interface_development_v1 import native_bindings, verify_native_bindings
from scripts.run_go2_causal_rgb_body_capture_development_v1 import probe_spec
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_single_sample_rgbd_observation_development_v1 import NATIVE_ROOT
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_observation_turn_session_development import StartupObservationSession, admit_setup, latest_policy
from scripts.startup_source_inventory_development import discover_sources


OUTPUT = ROOT / '.generated/go2_startup_observation_turn_development_v1_attempt_001'
PROTOCOL = 'docs/go2_startup_observation_turn_development_v1_2026-09-06.md'
SEEDS = ('scripts/run_go2_startup_observation_turn_development_v1.py',
    'scripts/audit_go2_startup_observation_turn_development_v1.py',
    'lewm/tests/test_startup_observation_turn_execution_development.py',
    'lewm/tests/test_startup_observation_turn_development.py',
    'lewm/tests/test_native_foot_geometry_evaluation_development.py', PROTOCOL)
NATIVE_EXTRA = ('engine/entities/rigid_entity/rigid_geom.py', 'engine/entities/rigid_entity/rigid_entity.py',
                'utils/urdf.py', 'constants.py')


def extension_bindings():
    return {str(NATIVE_ROOT / p): digest(NATIVE_ROOT / p) for p in NATIVE_EXTRA}


def verify_extensions(expected):
    if extension_bindings() != expected: raise ValueError('native geometry readback implementation changed')


def specification():
    spec = probe_spec(0)
    spec.update(scene_id='go2-startup-observation-turn-development-v1',
                family='SETUP_CONDITIONED_OBSERVATION_TURN', procedural_seed=2026090602)
    return spec


def collect(definition_sha256):
    initialize_genesis(backend='cpu', seed=2026090602, logging_level='warning')
    session = None; decisions = []; tape = []; timings = []; stop = None; tail = 0; controller = None
    try:
        session = StartupObservationSession(specification(), OUTPUT); session.install_contact_identity()
        build = session.ctx.build
        gains = configure_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
        write_json(OUTPUT / 'actuator_identity.json', gains)
        write_json(OUTPUT / 'floor_domain.json', build.floor_domain)
        write_json(OUTPUT / 'floor_roles.json', dict(
            physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            visual_only_link_ids=[int(l.idx) for l in build.visual_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_collision_geom_count=len(build.visual_floor.geoms)))

        def step(command, phase, decision_index):
            session.phase = phase
            entry = dict(tick=len(tape), pre_sample_index=len(session.samples)-1,
                requested_command=list(command), phase=phase, decision_index=decision_index)
            tape.append(entry); start = time.perf_counter()
            try: session.command_tick(command)
            finally:
                entry['post_sample_index'] = len(session.samples)-1
                entry['execution_wall_ms'] = 1000 * (time.perf_counter() - start)

        try:
            session.settle_recorded(); session.capture_current()
            geometry, velocity, region, admission = admit_setup(session, definition_sha256)
            controller = StartupObservationTurn(geometry, velocity_prior=velocity, region_prior=region, admission=admission)
            for index in range(21):
                session.capture_current(); policy = latest_policy(session)
                stamp = int(policy['sensor_state']['decision_ns']); start = time.perf_counter()
                row = controller.observe(policy, session.latest_depth, session.relative_observations[-1]['observer'], now_ns=stamp)
                timings.append(dict(observation_index=len(session.packet_rows)-1,
                                    controller_wall_ms=1000 * (time.perf_counter() - start)))
                decisions.append(dict(observation_index=len(session.packet_rows)-1, decision=row))
                print(json.dumps(dict(decision_index=index, decision=row)), flush=True)
                if row['terminal']: break
                if index == 20: raise PhysicalStop('STARTUP_CONTROLLER_TICK_LIMIT')
                step(row['requested_command'], 1, index)
            # A controller terminal is followed by real zero-command dynamics,
            # never by pretending frozen simulation constitutes a physical stop.
            for _ in range(3):
                step((0., 0., 0.), 2, None); tail += 1; session.capture_current()
            terminal_rows = capture_native_robot_geometry(build.robot)
            write_json(OUTPUT / 'terminal_native_robot_geometry.json', terminal_rows)
            raw = session.samples[-1]
            write_json(OUTPUT / 'terminal_foot_identity.json', match_native_foot_geometries(
                terminal_rows, geometry, raw['joint_position'], raw['base_pose_world']))
        except (PhysicalStop, SensorContractError) as error:
            stop = str(error)
        session.persist(OUTPUT); session.persist_observations(OUTPUT)
        terminal = read_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist())
        write_json(OUTPUT / 'terminal_actuator_gains.json', terminal)
        if terminal != gains['effective']: raise ValueError('actuator gain drift')
        return dict(physics_samples=len(session.samples), rgbd_frames=len(session.packet_rows),
            relative_observer_frames=len(session.relative_observations), controller_decisions=len(decisions),
            requested_ticks=len(tape), stopping_tail_ticks=tail, physical_stop_reason=stop,
            controller_status=controller.status if controller else None,
            navigation_qualified=False, real_time_qualified=False)
    finally:
        try:
            if session is not None:
                try:
                    if session.samples and not (OUTPUT / 'physics_trace.npz').exists():
                        session.persist(OUTPUT); session.persist_observations(OUTPUT)
                    write_json(OUTPUT / 'startup_decisions.json', decisions)
                    write_json(OUTPUT / 'startup_command_tape.json', tape)
                    write_json(OUTPUT / 'startup_timings.json', dict(captures=session.capture_timings, controller=timings))
                    write_json(OUTPUT / 'startup_guard_rows.json', session.startup_guard_rows)
                finally: session.ctx.build.scene.destroy()
        finally: shutdown_genesis()


def artifact_names(frames):
    names = ['actuator_identity.json', 'terminal_actuator_gains.json', 'static_objects.json',
        'floor_domain.json', 'floor_roles.json', 'physics_trace.npz', 'native_contacts.npz', 'contact_events.json',
        'contact_topology.json', 'ideal_sensor_samples.npz', 'policy_histories.npz', 'policy_observations.json',
        'camera_audit.json', 'fast_gyro_samples.npz', 'fast_gyro_histories.npz', 'depth_observations.json',
        'depth_camera_audit.json', 'relative_state_observations.json', 'floor_visual_collision_identity.json',
        'startup_floor_identity.json', 'startup_native_robot_geometry.json', 'startup_checks.json',
        'startup_admission.json', 'terminal_native_robot_geometry.json', 'terminal_foot_identity.json',
        'startup_decisions.json', 'startup_command_tape.json', 'startup_timings.json', 'startup_guard_rows.json']
    return names + [name for i in range(frames) for name in
                   (f'rgb_{i:04d}.png', f'native_depth_{i:04d}.npz', f'depth_{i:04d}.npz')]


def main():
    if OUTPUT.exists(): raise ValueError('fixed fresh one-shot startup observation trial only')
    identities = {str((PREVIOUS / p).relative_to(ROOT)): h for p, h in IDENTITIES.items()}
    verify_bindings(identities)
    old = json.loads((PREVIOUS / 'launch.json').read_text()); result = json.loads((PREVIOUS / 'result.json').read_text())
    inputs = old['input_sha256'] | identities | {str((PREVIOUS / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    sources = discover_sources(SEEDS, old['source_sha256'])
    verify_bindings(sources | inputs); verify_native_bindings(old['native_sha256'])
    extras = extension_bindings(); verify_extensions(extras)
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.json', dict(source_sha256=sources, input_sha256=inputs,
        native_sha256=old['native_sha256'], native_geometry_sha256=extras, specification=specification(),
        scope='one fresh conditional startup observation turn; not maze navigation or training'))
    try:
        started = time.perf_counter()
        row = collect(sources[PROTOCOL])
        row['total_acquisition_wall_ms'] = 1000 * (time.perf_counter()-started)
        verify_bindings(sources | inputs); verify_native_bindings(old['native_sha256']); verify_extensions(extras)
        leaves = artifact_names(row['rgbd_frames'])
        write_json(OUTPUT / 'result.json', dict(status='ACQUISITION_COMPLETE_AUDIT_REQUIRED', **row,
            artifact_sha256={p: digest(OUTPUT / p) for p in leaves if (OUTPUT / p).is_file()},
            absent_expected_artifacts=[p for p in leaves if not (OUTPUT / p).is_file()]))
        print('STARTUP_OBSERVATION_TURN_ACQUISITION_COMPLETE', flush=True)
    except Exception as error:
        write_json(OUTPUT / 'result.json', dict(status='TERMINAL_FAILURE', error=repr(error), navigation_qualified=False))
        raise


if __name__ == '__main__': main()
