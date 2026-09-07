"""Two predeclared supervised physical tapes; no estimator-selected commands."""
import json
import time

import cv2

from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.rgbd_shadow_motion_development import ShadowObserver
from lewm.sustained_motion_collection_development import specification, schedule, TRIALS, TAIL_TICKS
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.fresh_maze_session_development import admit_setup
from scripts.probe_go2_finite_rgbd_motion_errors_development_v1 import OUTPUT as PREVIOUS
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_observation_turn_session_development import latest_policy
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.sustained_motion_session_development import SustainedMotionSession

OUTPUT = ROOT/'.generated/go2_sustained_observed_floor_motion_development_v1_attempt_001'
PROTOCOL = 'docs/go2_sustained_observed_floor_motion_development_v1_2026-09-06.md'
SEEDS = (PROTOCOL, 'scripts/run_go2_sustained_observed_floor_motion_development_v1.py',
         'lewm/tests/test_sustained_motion_collection_development.py')
IDENTITIES = {'launch.json': '0c0cd27c661b14e1dfb6e5eb606929ae1329700931cd203239a6374065064558',
    'result.json': 'd8146a104bb5408234679ebf77cc592e30fc1b5883f33d78821787460f14299e',
    'shared_row_analysis.json': '6263944993c1c881d8ba9bb709bafd84db536ecf3d046f61c1eff2ccd7e13534'}


def preflight():
    bindings = {str((PREVIOUS/n).relative_to(ROOT)): h for n, h in IDENTITIES.items()}
    verify_bindings(bindings)
    old = read_json(PREVIOUS, 'launch.json'); verify(old)
    result = read_json(PREVIOUS, 'result.json'); audit = read_json(PREVIOUS, 'shared_row_analysis.json')
    verify_bindings(audit['source_sha256']|audit['input_sha256'])
    bindings |= {str((PREVIOUS/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    launch = old | dict(source_sha256=discover_sources(SEEDS, old['source_sha256']),
        input_sha256=old['input_sha256']|bindings|audit['source_sha256'], diagnostic_protocol=PROTOCOL,
        trial_specifications={t: specification(t) for t in TRIALS}, motion_schedule=schedule(),
        terminal_zero_ticks=TAIL_TICKS,
        scope='two supervised development collection trials; no learned navigation, error calibration or prospective action qualification')
    verify(launch); return launch


def artifact_names(frames):
    fixed = ('specification.json', 'actuator_identity.json', 'floor_roles.json',
        'terminal_actuator_gains.json', 'terminal_native_robot_geometry.json', 'terminal_environment_identity.json',
        'result.json', 'physics_trace.npz', 'native_contacts.npz', 'contact_events.json', 'contact_topology.json',
        'ideal_sensor_samples.npz', 'policy_histories.npz', 'policy_observations.json', 'camera_audit.json',
        'depth_observations.json', 'depth_camera_audit.json', 'fast_gyro_samples.npz', 'fast_gyro_histories.npz',
        'floor_visual_collision_identity.json', 'static_objects.json', 'startup_native_robot_geometry.json',
        'setup_checks.json', 'shadow_observations.json', 'command_tape.json', 'native_guard_rows.json',
        'visual_meshes/ground_visual.ply', 'visual_meshes/wide_front_visual.ply')
    return list(fixed)+[f'{prefix}_{i:04d}.{suffix}' for i in range(frames)
        for prefix, suffix in (('rgb', 'png'), ('depth', 'npz'), ('native_depth', 'npz'))]


def collect(trial, definition_sha256):
    output = OUTPUT/trial; output.mkdir(); spec = specification(trial); write_json(output/'specification.json', spec)
    session = shadow = None; rows = []; tape = []; physical_stop = None; completed = tail = 0
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    try:
        session = SustainedMotionSession(spec, output); session.install_contact_identity(); build = session.ctx.build
        gains = configure_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
        write_json(output/'actuator_identity.json', gains)
        write_json(output/'floor_roles.json', dict(
            physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        def observe_current():
            session.capture_current(); p = latest_policy(session); now = p['sensor_state']['decision_ns']
            start = time.perf_counter()
            result = shadow.observe(p, session.latest_depth, session.fast_buffer.packet(now_ns=now), now_ns=now)
            rows.append(dict(observation_index=len(session.model_manifest)-1, shadow=result,
                             observer_wall_ms=1000*(time.perf_counter()-start)))
        try:
            session.settle_recorded(); session.capture_current()
            velocity = admit_setup(session, definition_sha256); shadow = ShadowObserver(velocity); observe_current()
            stimulus = schedule(); commands = stimulus+[dict(segment='terminal_zero_tail', phase=10,
                requested_command=[0., 0., 0.]) for _ in range(TAIL_TICKS)]
            for index, command in enumerate(commands):
                session.phase = command['phase']
                item = command|dict(tick=index, pre_sample_index=len(session.samples)-1,
                    post_sample_index=None, completed=False, start_perf_counter_ns=time.perf_counter_ns())
                tape.append(item)
                try:
                    session.command_tick(command['requested_command'])
                    item['command_finished_perf_counter_ns'] = time.perf_counter_ns()
                    if index < len(stimulus): completed += 1
                    else: tail += 1
                    observe_current(); item['completed'] = True
                finally:
                    item['post_sample_index'] = len(session.samples)-1
                    item['end_perf_counter_ns'] = time.perf_counter_ns()
                    item['outer_wall_ms'] = (item['end_perf_counter_ns']-item['start_perf_counter_ns'])/1e6
                if index % 10 == 0 or index == len(commands)-1:
                    print(json.dumps(dict(trial=trial, tick=index, segment=command['segment'],
                        time_s=session.samples[-1]['timestamp_s'], shadow_status=rows[-1]['shadow']['status'])), flush=True)
        except PhysicalStop as error:
            physical_stop = str(error)  # No more physics in this trial, including tail.
        terminal = read_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist())
        if terminal != gains['effective']: raise ValueError('actuator gains changed')
        write_json(output/'terminal_actuator_gains.json', terminal)
        write_json(output/'terminal_native_robot_geometry.json', capture_native_robot_geometry(build.robot))
        write_json(output/'terminal_environment_identity.json', appearance_environment_identity(session))
        result = dict(status='PHYSICAL_TAPE_COMPLETE_AUDIT_REQUIRED' if completed==330 and tail==5 and physical_stop is None
                          else 'PHYSICAL_TAPE_INCOMPLETE_AUDIT_REQUIRED', trial=trial,
            completed_motion_ticks=completed, completed_zero_tail_ticks=tail, physical_stop=physical_stop,
            shadow_failure=None if shadow is None else shadow.failure,
            shadow_successful_frames=0 if shadow is None else shadow.successes,
            physics_samples=len(session.samples), rgbd_frames=len(session.model_manifest),
            estimator_selects_commands=False, prospective_envelope_qualified=False,
            fitting_or_validation_performance_scored=False, navigation_qualified=False)
        write_json(output/'result.json', result); return result
    finally:
        if session is not None:
            try:
                session.persist(output); session.persist_observations(output)
                write_json(output/'shadow_observations.json', rows); write_json(output/'command_tape.json', tape)
                write_json(output/'native_guard_rows.json', session.guard_rows)
            finally: session.ctx.build.scene.destroy()
        shutdown_genesis()


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('fresh exclusive collection only')
    cv2.setNumThreads(1); launch = preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json', launch)
    results = {}
    try:
        for trial in TRIALS:
            verify(launch)
            results[trial] = collect(trial, launch['source_sha256'][PROTOCOL])
        verify(launch)
        names = [f'{trial}/{n}' for trial, row in results.items() for n in artifact_names(row['rgbd_frames'])]
        present = [n for n in names if (OUTPUT/n).is_file()]
        result = dict(status='SUSTAINED_MOTION_ACQUISITION_COMPLETE_AUDIT_REQUIRED', trials=results,
            artifact_sha256={n: digest(OUTPUT/n) for n in present}, absent_expected_artifacts=sorted(set(names)-set(present)),
            independent_new_physical_trials=2, independent_geometry_count=1, estimator_selects_commands=False,
            action_model_fitted=False, error_model_calibrated=False, navigation_qualified=False, goal_achieved=False)
        write_json(OUTPUT/'result.json', result); print(json.dumps(results), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_SUSTAINED_COLLECTION_INFRASTRUCTURE_FAILURE',
            reason=str(error), completed_trials=results)); raise


if __name__ == '__main__': main()
