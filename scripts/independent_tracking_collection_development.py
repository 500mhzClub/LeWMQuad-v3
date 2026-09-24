"""One fixed raw observation tape with stop-only supervision, no observer control.

This is a callable collection component, not an authorized/frozen launcher. A
future cohort launcher must enforce source/input bindings, all-trial budgets,
resource ownership and sensor/evaluation phase ordering before calling it.
"""
import time

from lewm.independent_tracking_challenge_development import (
    validate_specification, MAX_TICKS, MAX_FRAMES, MAX_PHYSICS_SAMPLES, decision)
from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.support_friction_challenge_development import native_friction
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.independent_tracking_session_development import IndependentTrackingSession
from scripts.independent_tracking_artifacts_development import (
    StorageStop, MESH_BYTES, FRAME_BYTES, SETUP_BYTES, frame_names)
from scripts.independent_tracking_snapshot_development import persist_snapshot
from scripts.pulse_context_setup_development import admit_context_setup
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop


def collect_episode(store, spec, definition_sha256):
    validate_specification(spec)
    if store.directory.name != spec['trial']:
        raise ValueError('exact store/trial pairing required')
    if (not isinstance(definition_sha256, str) or len(definition_sha256) != 64
            or any(c not in '0123456789abcdef' for c in definition_sha256)):
        raise ValueError('frozen protocol SHA256 required')
    session = None; initialized = False; gains = None
    rows = []; tape = []; friction = []; admitted = False
    physical_stop = acquisition_stop = failure = None
    secondary_failures = []; complete = False; lifecycle = {}
    store.json('specification.json', spec)

    def elapsed(start):
        return (time.perf_counter_ns() - start) / 1e6

    def record_secondary(stage, operation):
        start = time.perf_counter_ns()
        try: operation()
        except Exception as error: secondary_failures.append(dict(stage=stage, error=repr(error)))
        finally: lifecycle[stage + '_wall_ms'] = elapsed(start)

    def packets():
        index = len(session.model_manifest)
        if index >= MAX_FRAMES: raise ValueError('capture frame budget exhausted')
        names = frame_names(index)
        if index == 0: names += ('floor_visual_collision_identity.json',)
        start = time.perf_counter_ns()
        with store.external(names, FRAME_BYTES):
            packet = session.sensor_packets()
        if (len(session.model_manifest) != index + 1 or len(session.depth_manifest) != index + 1
                or len(session.fast_packets) != index + 1):
            raise ValueError('complete new RGB/body/depth/gyro packet required')
        return packet, elapsed(start)

    try:
        start = time.perf_counter_ns()
        initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        with store.external(('visual_meshes/ground_visual.ply', 'visual_meshes/wall_union_visual.ply'), MESH_BYTES):
            session = IndependentTrackingSession(spec, store.directory)
            initialized = True
        lifecycle['construction_wall_ms'] = elapsed(start)
        session.install_contact_identity()
        build = session.ctx.build
        gains = configure_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
        store.json('actuator_identity.json', gains)
        store.json('floor_roles.json', dict(
            physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        friction.append(dict(stage='before_settle', **native_friction(build, spec['friction_mu'])))
        start = time.perf_counter_ns()
        try: session.settle_recorded()
        finally: lifecycle['settling_wall_ms'] = elapsed(start)
        first_packet, first_wall = packets()
        start = time.perf_counter_ns()
        try:
            with store.external(('static_objects.json', 'startup_native_robot_geometry.json', 'setup_checks.json'), SETUP_BYTES):
                admit_context_setup(session, definition_sha256)
        finally: lifecycle['setup_audit_wall_ms'] = elapsed(start)
        admitted = True
        for tick in range(MAX_TICKS + 1):
            store.check(FRAME_BYTES)
            friction.append(dict(stage='before_decision', tick=tick, **native_friction(build, spec['friction_mu'])))
            (p, _depth, _fast, now), acquisition_ms = (first_packet, first_wall) if tick == 0 else packets()
            start = time.perf_counter_ns()
            selected = decision(spec['direction'], tick, p)
            selection_ms = elapsed(start)
            if now != selected['decision_ns']:
                raise ValueError('capture/selector clock mismatch')
            rows.append(dict(tick=tick, observation_index=len(session.model_manifest)-1,
                pre_sample_index=len(session.samples)-1, decision=selected,
                acquisition_wall_ms=acquisition_ms, selection_wall_ms=selection_ms,
                acquisition_selection_deadline_missed=acquisition_ms+selection_ms > 100.,
                observer_computation_included=False, real_time_qualified=False))
            if selected['terminal']:
                complete = True; break
            item = dict(tick=tick, requested_command=selected['requested_command'], phase=selected['phase'],
                role=selected['role'], pre_sample_index=len(session.samples)-1,
                post_sample_index=None, completed=False, dispatch_and_physics_wall_ms=None)
            tape.append(item); session.phase = selected['phase']
            frames_before = len(session.model_manifest)
            start = time.perf_counter_ns()
            try:
                session.command_tick(item['requested_command']); item['completed'] = True
            finally:
                item['post_sample_index'] = len(session.samples)-1
                item['dispatch_and_physics_wall_ms'] = elapsed(start)
            if len(session.model_manifest) != frames_before:
                raise ValueError('command dispatch unexpectedly captured an unbudgeted frame')
            if len(session.samples) > MAX_PHYSICS_SAMPLES:
                raise ValueError('physics sample budget exceeded')
    except PhysicalStop as error:
        physical_stop = str(error)
    except StorageStop as error:
        acquisition_stop = str(error)
    except Exception as error:
        failure = repr(error)
    finally:
        if initialized:
            record_secondary('terminal_contact_integrity',lambda:session.contact_integrity.finish(len(session.samples)))
            def terminal_native():
                build = session.ctx.build
                friction.append(dict(stage='terminal', **native_friction(build, spec['friction_mu'])))
                if gains is not None:
                    terminal = read_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist())
                    store.json('terminal_actuator_gains.json', terminal)
                    if terminal != gains['effective']: raise ValueError('gait gains changed')
                store.json('terminal_native_robot_geometry.json', capture_native_robot_geometry(build.robot))
                environment = appearance_environment_identity(session)
                store.json('terminal_environment_identity.json', environment)
                if 'floor_visual_collision_identity.json' not in store.bindings:
                    store.json('floor_visual_collision_identity.json', environment)
            record_secondary('terminal_native', terminal_native)
            record_secondary('persist_snapshot', lambda: persist_snapshot(session, store))
            record_secondary('persist_guards', lambda: store.json('native_guard_rows.json', session.guard_rows))
        record_secondary('persist_decisions', lambda: store.json('tracking_decisions.json', rows))
        record_secondary('persist_tape', lambda: store.json('command_tape.json', tape))
        record_secondary('persist_friction', lambda: store.json('friction_checks.json', friction))
        if initialized:
            record_secondary('scene_destroy', session.ctx.build.scene.destroy)
        record_secondary('genesis_shutdown', shutdown_genesis)
    result = dict(status='TRACKING_TAPE_REQUIRES_RAW_AUDIT', trial=spec['trial'],
        initialized=initialized, setup_admitted=admitted, setup_checked='setup_checks.json' in store.bindings,
        schedule_complete=complete, physical_stop=physical_stop, acquisition_stop=acquisition_stop,
        infrastructure_failure=failure, secondary_failures=secondary_failures,
        command_ticks=len(tape), completed_ticks=sum(t['completed'] for t in tape), decisions=len(rows),
        physics_samples=len(session.samples) if initialized else 0,
        rgbd_frames=len(session.model_manifest) if initialized else 0,
        native_contact_integrity=session.contact_integrity.report() if initialized else None,
        lifecycle_wall_ms=lifecycle, tracker_required_for_commands=False, native_state_used_for_commands=False,
        observer_executed=False, native_evaluation_executed=False,
        sensor_reconstruction_verified=False, navigation_qualified=False, real_time_qualified=False)
    if failure is not None or secondary_failures or store.failed_internal or store.failed_external and physical_stop is None:
        result['status'] = 'TERMINAL_TRACKING_COLLECTION_INFRASTRUCTURE_FAILURE'
    if result['status'] == 'TERMINAL_TRACKING_COLLECTION_INFRASTRUCTURE_FAILURE':
        store.json('failure.json', result)
    store.json('result.json', result)
    receipt = store.verify()
    return result, receipt
