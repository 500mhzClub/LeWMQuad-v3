"""Reconstruct the fixed new robot-interface assay from native raw evidence."""
import json
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import from_native_depth, INTRINSICS
from lewm.physical_execution_development import rotation_xyzw
from lewm.physical_semantics import world_from_optical
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.simulated_body_observation_development import IdealBodySensor, BodyObservationBuffer, JOINT_NAMES
from lewm.simulated_fast_gyro_development import IdealFastGyro
from lewm.fast_gyro_development import FastGyroBuffer
from lewm.safety.contact_attribution import attribute_contacts
from lewm.visual_surface_depth_evaluation_development import expected_optical_depth
from lewm_genesis.bounded_scene_builder_development import check_capture_domain
from lewm_genesis.floor_extent_precision_development import check_extent_identity
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.run_go2_contact_attributed_execution_development_v1 import CONTACT_FIELDS
from scripts.run_go2_bounded_floor_robot_interface_development_v1 import OUTPUT, ROOT, COMMANDS, specification
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.run_go2_successive_choice_maze_development_v1 import write_json


def require(value, message):
    if not value: raise ValueError(message)


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        require(set(a) == set(b), 'matching replay fields')
        for key in a: same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        require(len(a) == len(b), 'matching replay sequence lengths')
        for x, y in zip(a, b, strict=True): same(x, y)
    else: require(a == b, 'matching replay values')


def read_npz(name):
    with np.load(OUTPUT / name, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def audit():
    launch = json.loads((OUTPUT / 'launch.json').read_text())
    result = json.loads((OUTPUT / 'result.json').read_text())
    require(result['status'] == 'ACQUISITION_COMPLETE_AUDIT_REQUIRED', 'completed or stopped acquisition required')
    same(launch['specification'], specification()); same(launch['commands'], COMMANDS)
    bindings = launch['source_sha256'] | launch['input_sha256']
    bindings |= {str((OUTPUT / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256'])
    raw = read_npz('physics_trace.npz'); contacts = read_npz('native_contacts.npz')
    n = len(raw['timestamp_s']); require(n == result['physics_samples'] and n > 0, 'physical sample population')
    ns = np.rint(raw['timestamp_s'] * 1e9).astype(np.int64)
    same(ns, np.arange(1, n + 1) * 2_000_000)
    require(all(len(v) == n and np.isfinite(v).all() for v in raw.values()), 'finite complete physical fields')
    requested = np.repeat(np.asarray([(0., 0., 0.)] * 15 + COMMANDS), 50, axis=0)[:n]
    same(raw['requested_command'], requested)
    for start in range(0, n, 50):
        require(np.all(raw['applied_command'][start:start+50] == raw['applied_command'][start]), 'constant applied command within tick')
    gains = json.loads((OUTPUT / 'actuator_identity.json').read_text())
    same(gains['effective'], json.loads((OUTPUT / 'terminal_actuator_gains.json').read_text()))
    topology = json.loads((OUTPUT / 'contact_topology.json').read_text())
    roles = json.loads((OUTPUT / 'floor_roles.json').read_text())
    physical, visual = set(roles['physical_ground_link_ids']), set(roles['visual_only_link_ids'])
    require(physical and visual and not physical & visual and roles['visual_collision_geom_count'] == 0, 'distinct native plane roles')
    require(set(topology['ground_link_ids']) == physical and not visual & set(map(int, topology['environment_object_ids'])), 'physical-only ground attribution')
    require(len(contacts['frame_offsets']) == n + 1, 'contact frame population')
    same(contacts['frame_timestamp_s'], raw['timestamp_s'])
    events = json.loads((OUTPUT / 'contact_events.json').read_text()); require(len(events) == n, 'contact event population')
    support_force = []
    for i in range(n):
        start, end = contacts['frame_offsets'][i:i+2]
        packet = {k: contacts[k][start:end][None] for k in CONTACT_FIELDS}
        classified = attribute_contacts(packet, environment_index=0,
            robot_link_ids=topology['robot_link_ids'], support_link_ids=topology['support_link_ids'],
            ground_link_ids=topology['ground_link_ids'], link_names={int(k): v for k, v in topology['link_names'].items()},
            environment_object_ids={int(k): v for k, v in topology['environment_object_ids'].items()})
        disallowed = [r for r in classified if r['disallowed']]
        require(bool(disallowed) == bool(raw['physics_contact'][i]), 'native contact stop replay')
        same(events[i]['disallowed_contacts'], disallowed)
        require(events[i]['sample_index'] == i and events[i]['phase'] == int(raw['phase'][i]), 'contact event clock/phase')
        support_force.append(sum(r['force_magnitude_n'] for r in classified if r['environment_link_id'] in physical))
    static = json.loads((OUTPUT / 'static_objects.json').read_text())
    walls = specification()['geometry']['wall_boxes']; require(len(static) == len(walls), 'all native walls recorded')
    for row, expected in zip(static, walls, strict=True):
        require(row['native_name'] == expected['wall_id'] and row['fixed'] and row['collision_enabled'], 'native fixed wall identity')
        np.testing.assert_allclose(row['native_position'], expected['centre_xyz'], atol=2e-7, rtol=0)
        np.testing.assert_allclose(row['native_box_size'], expected['size_xyz'], atol=2e-7, rtol=0)
        yaw = expected['yaw_rad']
        np.testing.assert_allclose(row['native_quaternion_wxyz'], [np.cos(yaw/2), 0., 0., np.sin(yaw/2)], atol=2e-7, rtol=0)
    cameras = json.loads((OUTPUT / 'camera_audit.json').read_text())
    depth_cameras = json.loads((OUTPUT / 'depth_camera_audit.json').read_text())
    require(len(cameras) == len(depth_cameras) == result['rgbd_frames'], 'paired camera population')
    if cameras: check_extent_identity(json.loads((OUTPUT / 'floor_visual_collision_identity.json').read_text()), 32.)
    by_sample = {c['physical_sample_index']: i for i, c in enumerate(cameras)}
    body, fast = IdealBodySensor(), IdealFastGyro()
    buffer, fast_buffer = BodyObservationBuffer((0, 0, 0)), FastGyroBuffer((0, 0, 0))
    body_saved, fast_saved = read_npz('ideal_sensor_samples.npz'), read_npz('fast_gyro_samples.npz')
    body_index = 0; depth_results = []
    geometry = ArticulatedCollisionGeometry(URDF); primitive_gaps = []; shape_ids = None
    for i, stamp in enumerate(ns):
        pose = raw['base_pose_world'][i]; twist = raw['base_twist_world'][i]
        f, valid = fast.sample(measured_ns=int(stamp), quaternion_xyzw=pose[3:], angular_velocity_world=twist[3:])
        same(f, fast_saved['values'][i]); same(valid, fast_saved['valid'][i])
        require(fast_saved['measured_ns'][i] == stamp == fast_saved['available_ns'][i], '500Hz sensor clocks')
        fast_buffer.append(f, valid, measured_ns=int(stamp), available_ns=int(stamp))
        if stamp % 20_000_000 == 0:
            values = body.sample(measured_ns=int(stamp), quaternion_xyzw=pose[3:], velocity_world=twist[:3],
                angular_velocity_world=twist[3:], joint_position=raw['joint_position'][i],
                joint_velocity=raw['joint_velocity'][i], joint_names=JOINT_NAMES)
            buffer.append_sensors(values, int(stamp))
            require(body_saved['measured_ns'][body_index] == stamp, '50Hz sensor clocks')
            for name, pair in values.items():
                for j, field in enumerate(('values', 'valid')): same(pair[j], body_saved[f'{name}_{field}'][body_index])
            body_index += 1
        if stamp % 100_000_000 == 0: buffer.append_applied_command(raw['applied_command'][i], int(stamp))
        shapes = geometry.supports(raw['joint_position'][i], rotation_xyzw(pose[3:]))['shapes']
        if shape_ids is None: shape_ids = [s['shape_id'] for s in shapes]
        primitive_gaps.append([s['lower'][2] + pose[2] for s in shapes])
        if i not in by_sample: continue
        frame = by_sample[i]; c = cameras[frame]; dc = depth_cameras[frame]
        policy, depth = load_rgbd_observation(OUTPUT, frame)
        same(buffer.packet(policy['image']['rgb'], int(stamp)), policy)
        same(fast_buffer.packet(now_ns=int(stamp)), load_fast_packet(OUTPUT, frame))
        native = read_npz(f'native_depth_{frame:04d}.npz')['optical_depth_m']
        same(from_native_depth(native, policy, measured_ns=int(stamp), available_ns=int(stamp), now_ns=int(stamp)), depth)
        R = rotation_xyzw(pose[3:]); T = world_from_optical(pose[:3] + R @ [.326, 0., .043], R[:, 0], R[:, 2])
        np.testing.assert_allclose(c['world_from_optical'], T, atol=1e-12, rtol=0)
        check_capture_domain(None, T)
        require(dc['physics_clock_before_after_ns'] == [int(stamp)]*2, 'no intervening physics at RGBD capture')
        np.testing.assert_allclose(dc['native_intrinsics'], INTRINSICS, atol=1e-7, rtol=0)
        require(dc['native_near_m'] == .05 and dc['native_far_m'] == 200., 'fixed camera clip')
        require(dc['sampling_readback'] == {'draw_framebuffer_is_single_sample_target': True,
            'draw_framebuffer_is_multisample_target': False, 'samples': 0, 'sample_buffers': 0,
            'multisample_enabled': False, 'pixel_scale': 1}, 'single-sample depth target')
        ref = expected_optical_depth(walls, T, floor_z_m=0.)
        expected = ref['expected_depth_m']; use = ref['surface_interior'] & (expected > .22) & (expected < 4.98)
        measured = native[np.ix_(ref['rows'], ref['columns'])]
        error = np.abs(measured[use] - expected[use])
        require(len(error) >= 1000 and np.isfinite(error).all(), 'sufficient finite interior depth rays')
        row = {'frame': frame, 'rays': len(error), 'maximum_error_m': float(error.max()), 'within1mm': bool(error.max() <= .001)}
        for name, mask in (('floor', ref['object_index'] == 0), ('wall', ref['object_index'] != 0)):
            selected = use & mask; errors = np.abs(measured[selected] - expected[selected])
            row[name + '_rays'] = int(selected.sum()); row[name + '_max_error_m'] = float(errors.max()) if len(errors) else None
        depth_results.append(row)
    gaps = np.asarray(primitive_gaps)
    feet = {f'{leg}_foot:0' for leg in ('FL', 'FR', 'RL', 'RR')}
    require(feet <= set(shape_ids), 'exact foot identities present')
    summaries = [{'shape_id': name, 'foot_sphere': name in feet,
                  'minimum_gap_m': float(gaps[:, j].min()),
                  'minimum_active_gap_m': float(gaps[750:, j].min()) if n > 750 else None,
                  'samples_below_zero': int((gaps[:, j] < 0).sum())} for j, name in enumerate(shape_ids)]
    complete = n == 2000 and len(cameras) == 26 and result['whole_tape_completed'] and result['stop_reason'] is None
    require(not complete or np.array_equal([c['physical_sample_index'] for c in cameras], np.arange(749, 2000, 50)), 'complete planned camera cadence')
    interface = bool(complete and not raw['physics_contact'].any() and all(r['within1mm'] for r in depth_results))
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256'])
    return {'status': 'RAW_ARTIFACTS_VERIFIED_INTERFACE_REPORTED', 'interface_check_pass': interface,
        'whole_tape_completed': bool(complete), 'physics_samples': n, 'rgbd_frames': len(cameras),
        'sensor_and_contact_reconstruction_exact': True, 'depth_checks': depth_results,
        'primitive_gap_diagnostics': summaries, 'maximum_summed_ground_contact_force_magnitudes_n': float(max(support_force)),
        'scope': 'new robot interface and nominal URDF/evaluator contact diagnostics only',
        'contact_model_validated': False, 'sensor_calibrated': False, 'navigation_qualified': False}


def main():
    report = audit()
    write_json(OUTPUT / 'raw_artifact_audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
