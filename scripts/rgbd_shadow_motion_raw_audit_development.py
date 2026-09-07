"""Raw-only sensor/contact replay for fresh shadow motion (no simulation)."""
import hashlib
import json

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import from_native_depth, INTRINSICS
from lewm.depth_relative_motion_development import DepthRelativeState
from lewm.fast_gyro_development import FastGyroBuffer
from lewm.physical_execution_development import rotation_xyzw
from lewm.physical_semantics import world_from_optical
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.safety.contact_attribution import attribute_contacts
from lewm.simulated_body_observation_development import IdealBodySensor, BodyObservationBuffer, JOINT_NAMES
from lewm.simulated_fast_gyro_development import IdealFastGyro
from lewm.visual_surface_depth_evaluation_development import expected_optical_depth
from lewm_genesis.bounded_scene_builder_development import check_capture_domain
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.audit_go2_bounded_floor_robot_interface_native_box_reader_development_v1 import same, require, check_native_box_size
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.run_go2_contact_attributed_execution_development_v1 import CONTACT_FIELDS


def read_json(directory, name):
    return json.loads((directory / name).read_text())


def read_npz(directory, name):
    with np.load(directory / name, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def contact_packet(contacts, index):
    lo, hi = contacts['frame_offsets'][index:index+2]
    return {k: contacts[k][lo:hi] for k in CONTACT_FIELDS}


def classify(packet, topology):
    return attribute_contacts({k: v[None] for k, v in packet.items()}, environment_index=0,
        robot_link_ids=topology['robot_link_ids'], support_link_ids=topology['support_link_ids'],
        ground_link_ids=topology['ground_link_ids'], link_names={int(k): v for k, v in topology['link_names'].items()},
        environment_object_ids={int(k): v for k, v in topology['environment_object_ids'].items()})


def audit_sensors(directory, spec, result):
    raw = read_npz(directory, 'physics_trace.npz'); contacts = read_npz(directory, 'native_contacts.npz')
    n = len(raw['timestamp_s'])
    require(n == result['physics_samples'] and n >= 750, 'complete settling required for startup sensor audit')
    ns = np.rint(raw['timestamp_s'] * 1e9).astype(np.int64)
    same(ns, np.arange(1, n+1) * 2_000_000)
    require(all(len(v) == n and np.isfinite(v).all() for v in raw.values()), 'finite complete physics')
    topology = read_json(directory, 'contact_topology.json'); roles = read_json(directory, 'floor_roles.json')
    physical, visual = set(roles['physical_ground_link_ids']), set(roles['visual_only_link_ids'])
    require(physical and visual and not physical & visual and roles['visual_collision_geom_count'] == 0, 'distinct floor roles')
    require(set(topology['ground_link_ids']) == physical and not visual & set(map(int, topology['environment_object_ids'])), 'physical-only contact topology')
    same(topology['native_environment_count'], 1); same(topology['selected_environment_index'], 0)
    offsets = contacts['frame_offsets']
    require(offsets.shape == (n+1,) and np.issubdtype(offsets.dtype, np.integer)
            and offsets[0] == 0 and np.all(np.diff(offsets) >= 0), 'contact offsets')
    require(all(len(contacts[k]) == offsets[-1] for k in CONTACT_FIELDS), 'complete contact fields')
    same(contacts['frame_timestamp_s'], raw['timestamp_s'])
    events = read_json(directory, 'contact_events.json'); require(len(events) == n, 'contact event population')
    force = []
    for i in range(n):
        classified = classify(contact_packet(contacts, i), topology)
        disallowed = [r for r in classified if r['disallowed']]
        same(bool(disallowed), bool(raw['physics_contact'][i]))
        same(events[i], dict(sample_index=i, timestamp_s=float(raw['timestamp_s'][i]),
                            phase=int(raw['phase'][i]), disallowed_contacts=disallowed))
        require(all(r['force_status'] == 'measured' for r in classified), 'measured native forces')
        force.append(sum(r['force_magnitude_n'] for r in classified if r['environment_link_id'] in physical))
    walls = spec['geometry']['wall_boxes']; static = read_json(directory, 'static_objects.json')
    require(len(static) == len(walls), 'complete wall inventory')
    for row, expected in zip(static, walls, strict=True):
        require(row['native_name'] == expected['wall_id'] and row['fixed'] and row['collision_enabled']
                and row['native_collision_boxes'] == 1, 'fixed native wall identity')
        np.testing.assert_allclose(row['native_position'], expected['centre_xyz'], atol=2e-7, rtol=0)
        check_native_box_size(row['native_box_size'], expected['size_xyz'])
        yaw = expected['yaw_rad']
        np.testing.assert_allclose(row['native_quaternion_wxyz'], [np.cos(yaw/2), 0., 0., np.sin(yaw/2)], atol=2e-7, rtol=0)
    cameras = read_json(directory, 'camera_audit.json'); dcameras = read_json(directory, 'depth_camera_audit.json')
    relatives = []  # Independently reconstructed raw-depth comparator, not a collector output.
    require(len(cameras) == len(dcameras) == result['rgbd_frames'] > 0, 'complete paired sensor frames')
    same(read_json(directory, 'floor_visual_collision_identity.json'), read_json(directory, 'terminal_environment_identity.json'))
    indices = [c['physical_sample_index'] for c in cameras]
    require(indices == sorted(set(indices)) and indices[0] == 749 and indices[-1] < n, 'ordered camera samples')
    require(all((i+1) % 50 == 0 for i in indices), 'startup decision-clock captures')
    by_sample = {sample: frame for frame, sample in enumerate(indices)}
    body, fast, relative = IdealBodySensor(), IdealFastGyro(), DepthRelativeState()
    buffer, fbuffer = BodyObservationBuffer((0, 0, 0)), FastGyroBuffer((0, 0, 0))
    bs, fs = read_npz(directory, 'ideal_sensor_samples.npz'), read_npz(directory, 'fast_gyro_samples.npz')
    require(all(len(v) == n for v in fs.values()) and all(len(v) == n//10 for v in bs.values()), 'raw sensor populations')
    geometry = ArticulatedCollisionGeometry(URDF); gaps = []; names = None; bindex = 0; depth_results = []
    for i, stamp in enumerate(ns):
        pose, twist = raw['base_pose_world'][i], raw['base_twist_world'][i]
        values, valid = fast.sample(measured_ns=int(stamp), quaternion_xyzw=pose[3:], angular_velocity_world=twist[3:])
        same(values, fs['values'][i]); same(valid, fs['valid'][i])
        require(fs['measured_ns'][i] == fs['available_ns'][i] == stamp, 'fast sensor clocks')
        fbuffer.append(values, valid, measured_ns=int(stamp), available_ns=int(stamp))
        if stamp % 20_000_000 == 0:
            body_values = body.sample(measured_ns=int(stamp), quaternion_xyzw=pose[3:], velocity_world=twist[:3],
                angular_velocity_world=twist[3:], joint_position=raw['joint_position'][i],
                joint_velocity=raw['joint_velocity'][i], joint_names=JOINT_NAMES)
            buffer.append_sensors(body_values, int(stamp)); same(bs['measured_ns'][bindex], stamp)
            for name, pair in body_values.items():
                for j, field in enumerate(('values', 'valid')): same(pair[j], bs[f'{name}_{field}'][bindex])
            bindex += 1
        if stamp % 100_000_000 == 0: buffer.append_applied_command(raw['applied_command'][i], int(stamp))
        R = rotation_xyzw(pose[3:]); shapes = geometry.supports(raw['joint_position'][i], R)['shapes']
        if names is None: names = [s['shape_id'] for s in shapes]
        gaps.append([s['lower'][2] + pose[2] for s in shapes])
        if i not in by_sample: continue
        frame = by_sample[i]; c, dc = cameras[frame], dcameras[frame]
        same(dc['physical_sample_index'], i)
        same(c['timestamp_s'], float(raw['timestamp_s'][i])); same(dc['timestamp_s'], c['timestamp_s'])
        policy, depth = load_rgbd_observation(directory, frame)
        same(buffer.packet(policy['image']['rgb'], int(stamp)), policy)
        packet = fbuffer.packet(now_ns=int(stamp)); same(packet, load_fast_packet(directory, frame))
        observed = relative.observe(policy, depth, packet, now_ns=int(stamp))
        relatives.append(dict(observation_index=frame, observer=observed))
        native = read_npz(directory, f'native_depth_{frame:04d}.npz')['optical_depth_m']
        same(from_native_depth(native, policy, measured_ns=int(stamp), available_ns=int(stamp), now_ns=int(stamp)), depth)
        same(hashlib.sha256(native.tobytes()).hexdigest(), dc['native_depth_sha256'])
        same(hashlib.sha256(policy['image']['rgb'].tobytes()).hexdigest(), c['rgb_sha256'])
        T = world_from_optical(pose[:3] + R @ [.326, 0., .043], R[:, 0], R[:, 2])
        np.testing.assert_allclose(c['world_from_optical'], T, atol=1e-12, rtol=0)
        check_capture_domain(None, T); same(dc['physics_clock_before_after_ns'], [int(stamp)]*2)
        np.testing.assert_allclose(dc['native_intrinsics'], INTRINSICS, atol=1e-7, rtol=0)
        require(dc['native_near_m'] == .05 and dc['native_far_m'] == 200., 'fixed camera clipping')
        same(dc['sampling_readback'], dict(draw_framebuffer_is_single_sample_target=True,
            draw_framebuffer_is_multisample_target=False, samples=0, sample_buffers=0, multisample_enabled=False, pixel_scale=1))
        ref = expected_optical_depth(walls, T, floor_z_m=0.)
        expected = ref['expected_depth_m']; use = ref['surface_interior'] & (expected > .22) & (expected < 4.98)
        measured = native[np.ix_(ref['rows'], ref['columns'])]; errors = np.abs(measured[use]-expected[use])
        require(len(errors) >= 1000 and np.isfinite(errors).all(), 'sufficient finite interior depth rays')
        row = dict(frame=frame, rays=len(errors), maximum_error_m=float(errors.max()), within1mm=bool(errors.max() <= .001))
        for name, mask in (('floor', ref['object_index'] == 0), ('wall', ref['object_index'] != 0)):
            selected = use & mask; e = np.abs(measured[selected]-expected[selected])
            row[name+'_rays'] = int(selected.sum()); row[name+'_max_error_m'] = float(e.max()) if len(e) else None
        depth_results.append(row)
    gaps = np.asarray(gaps)
    report = dict(sensor_contact_reconstruction_exact=True, raw_depth_comparator_recomputed=True, depth_checks=depth_results,
        maximum_summed_ground_contact_force_magnitudes_n=float(max(force)),
        primitive_gap_diagnostics=[dict(shape_id=name, minimum_gap_m=float(gaps[:, j].min()),
            minimum_active_gap_m=float(gaps[750:, j].min()) if n > 750 else None) for j, name in enumerate(names)])
    return raw, contacts, topology, roles, cameras, relatives, geometry, report
