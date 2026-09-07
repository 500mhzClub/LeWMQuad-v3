"""Fresh fixed Go2 command/acquisition interface assay, not maze navigation."""
import json
import numpy as np

from lewm.actuator_gain_development import configure_gains, read_gains
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.bounded_rgbd_session_development import BoundedRGBDSession
from scripts.run_go2_causal_rgb_body_capture_development_v1 import probe_spec
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_marker_beacon_development_v1 import static_identity
from scripts.run_go2_floor_extent_precision_development_v1 import OUTPUT as PREVIOUS, ROOT
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = ROOT / '.generated/go2_bounded_floor_robot_interface_development_v1_attempt_001'
SOURCES = ('lewm_genesis/lewm_genesis/bounded_scene_builder_development.py',
    'scripts/bounded_floor_physical_init_development.py', 'scripts/bounded_rgbd_session_development.py',
    'lewm/tests/test_bounded_go2_scene_development.py',
    'scripts/run_go2_bounded_floor_robot_interface_development_v1.py',
    'scripts/audit_go2_bounded_floor_robot_interface_development_v1.py',
    'docs/go2_bounded_floor_robot_interface_development_v1_2026-09-06.md')
COMMANDS = [(0., 0., 0.)] * 5 + [(.2, 0., 0.)] * 5 + [(0., 0., .3)] * 5 + [(0., 0., -.3)] * 5 + [(0., 0., 0.)] * 5


def specification():
    spec = probe_spec(0)
    spec.update(scene_id='go2-bounded-floor-robot-interface-development-v1',
                family='BOUNDED_FLOOR_ROBOT_INTERFACE_DEVELOPMENT', procedural_seed=2026090601)
    return spec


def collect(spec):
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    session = None
    try:
        session = BoundedRGBDSession(spec, OUTPUT)
        session.install_contact_identity()
        gains = configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
        write_json(OUTPUT / 'actuator_identity.json', gains)
        write_json(OUTPUT / 'static_objects.json', static_identity(session))
        build = session.ctx.build
        write_json(OUTPUT / 'floor_domain.json', build.floor_domain)
        write_json(OUTPUT / 'floor_roles.json', {
            'physical_ground_link_ids': [int(l.idx) for l in build.collision_floor.links],
            'visual_only_link_ids': [int(l.idx) for l in build.visual_floor.links],
            'physical_ground_geom_ids': [int(g.idx) for g in build.collision_floor.geoms],
            'visual_collision_geom_count': len(build.visual_floor.geoms),
            'scope': 'evaluation-only actual plane roles, never sensor input'})
        stop = None; completed_ticks = 0
        try:
            session.settle_recorded()
            session.capture_current()
            for i, command in enumerate(COMMANDS):
                session.phase = 1 + i // 5
                session.command_tick(command)
                session.capture_current()
                completed_ticks += 1
                print(json.dumps({'completed_command_ticks': completed_ticks, 'physics_samples': len(session.samples),
                                  'rgbd_frames': len(session.packet_rows)}), flush=True)
        except PhysicalStop as error:
            stop = str(error)
        raw = session.persist(OUTPUT); session.persist_observations(OUTPUT)
        terminal = read_gains(build.robot, session.ctx.runner._leg_dof_idx.tolist())
        write_json(OUTPUT / 'terminal_actuator_gains.json', terminal)
        if terminal != gains['effective']: raise ValueError('actuator gain drift')
        return {'completed_command_ticks': completed_ticks, 'physics_samples': len(session.samples),
                'rgbd_frames': len(session.packet_rows), 'stop_reason': stop,
                'any_native_disallowed_contact': bool(raw['physics_contact'].any()),
                'whole_tape_completed': stop is None and completed_ticks == 25 and len(session.samples) == 2000}
    except Exception:
        if session is not None and session.samples and not (OUTPUT / 'physics_trace.npz').exists():
            session.persist(OUTPUT); session.persist_observations(OUTPUT)
        raise
    finally:
        try:
            if session is not None: session.ctx.build.scene.destroy()
        finally: shutdown_genesis()


def main():
    if OUTPUT.exists(): raise ValueError('fixed fresh one-shot Go2 interface run only')
    identities = {str((PREVIOUS / p).relative_to(ROOT)): h for p, h in (
        ('launch.json', 'aad642ac9576cb3075bf132e97b01d3227c50767b9f7c56c02cce3b749b91f58'),
        ('result.json', '4dc9a5915fa9503efa4a66f58d98fd2bdbf3fb73997dc38da8eb946115e7d27b'),
        ('raw_artifact_audit.json', '4832196c35509b1e594e0a7806b31a9d307a6b45250daabe904f6349d64b7e15'))}
    verify_bindings(identities)
    old = json.loads((PREVIOUS / 'launch.json').read_text())
    previous_result = json.loads((PREVIOUS / 'result.json').read_text())
    inputs = old['input_sha256'] | identities | {str((PREVIOUS / p).relative_to(ROOT)): h for p, h in previous_result['artifact_sha256'].items()}
    if set(SOURCES) & set(old['source_sha256']): raise ValueError('new source paths required')
    sources = old['source_sha256'] | {p: digest(ROOT / p) for p in SOURCES}
    verify_bindings(sources | inputs); verify_native_bindings(old['native_sha256'])
    OUTPUT.mkdir(); spec = specification()
    write_json(OUTPUT / 'launch.json', {'source_sha256': sources, 'input_sha256': inputs,
        'native_sha256': old['native_sha256'], 'specification': spec, 'commands': COMMANDS,
        'scope': 'fresh bounded-floor Go2 interface and contact diagnostics, not navigation or training'})
    try:
        row = collect(spec)
        verify_bindings(sources | inputs); verify_native_bindings(old['native_sha256'])
        leaves = ['actuator_identity.json', 'terminal_actuator_gains.json', 'static_objects.json',
            'floor_domain.json', 'floor_roles.json', 'physics_trace.npz', 'native_contacts.npz', 'contact_events.json',
            'contact_topology.json', 'ideal_sensor_samples.npz', 'policy_histories.npz', 'policy_observations.json',
            'camera_audit.json', 'fast_gyro_samples.npz', 'fast_gyro_histories.npz', 'depth_observations.json',
            'depth_camera_audit.json', 'relative_state_observations.json']
        if row['rgbd_frames']: leaves.append('floor_visual_collision_identity.json')
        leaves += [name for i in range(row['rgbd_frames']) for name in
                   (f'rgb_{i:04d}.png', f'native_depth_{i:04d}.npz', f'depth_{i:04d}.npz')]
        write_json(OUTPUT / 'result.json', {'status': 'ACQUISITION_COMPLETE_AUDIT_REQUIRED', **row,
            'artifact_sha256': {p: digest(OUTPUT / p) for p in leaves}, 'navigation_qualified': False})
        print('BOUNDED_FLOOR_ROBOT_INTERFACE_ACQUISITION_COMPLETE', flush=True)
    except Exception as error:
        if not (OUTPUT / 'result.json').exists():
            write_json(OUTPUT / 'result.json', {'status': 'TERMINAL_FAILURE', 'error': repr(error)})
        raise


if __name__ == '__main__': main()
