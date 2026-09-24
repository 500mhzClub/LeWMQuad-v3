"""Recompute scene/ray/contact evidence from this fixed assay's raw artifacts."""
import hashlib
import json
import math

import numpy as np
from PIL import Image

from lewm.causal_depth_observation_development import INTRINSICS
from lewm.physical_semantics import world_from_optical
from lewm.visual_surface_depth_evaluation_development import expected_optical_depth
from lewm_genesis.aligned_floor_development import check_aligned_floor_identity
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_aligned_floor_interface_development_v1 import OUTPUT, ROOT, VIEWS, verify_native_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import write_json


def require(value, message):
    if not value: raise ValueError(message)


def audit():
    launch = json.loads((OUTPUT / 'launch.json').read_text())
    result = json.loads((OUTPUT / 'result.json').read_text())
    require(result['status'] == 'ACQUISITION_COMPLETE_AUDIT_REQUIRED', 'completed fresh acquisition required')
    bindings = launch['source_sha256'] | launch['input_sha256']
    bindings |= {str((OUTPUT / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256'])
    floor = check_aligned_floor_identity(json.loads((OUTPUT / 'floor_identity.json').read_text()))
    physical = json.loads((OUTPUT / 'physical_identity.json').read_text())
    require(physical['sphere_radius_m'] == .022 and physical['dt_s'] == .002, 'fixed sphere and timing')
    require(physical['floor_geom_idx'] != physical['sphere_geom_idx'], 'distinct physical identities')
    with np.load(OUTPUT / 'sphere_trace.npz', allow_pickle=False) as trace:
        positions = trace['position_world_m']; steps = trace['step']
    require(positions.shape == (500, 3) and np.isfinite(positions).all()
            and np.array_equal(steps, np.arange(1, 501)), 'complete finite physics trace')
    require(np.max(np.abs(positions[:, :2] - [-1., 0.])) < .001, 'sphere remains behind camera')
    gap = positions[:, 2] - .022
    require(np.min(gap) >= -.001 and np.max(np.abs(gap[-100:])) <= .001, 'sphere collision-surface settling')
    contacts = json.loads((OUTPUT / 'contacts.json').read_text())
    require(len(contacts) == 500, 'all physics contact records required')
    touched = []
    for i, row in enumerate(contacts):
        require(row['step'] == i + 1, 'contact/physics clocks')
        a, b = np.asarray(row['geom_a']), np.asarray(row['geom_b'])
        require(a.shape == b.shape and a.ndim == 1, 'contact pair arrays')
        force_a, force_b, p = (np.asarray(row[k], float).reshape(-1, 3) for k in ('force_a', 'force_b', 'position'))
        require(len(force_a) == len(a) == len(force_b) == len(p) and
                all(np.isfinite(v).all() for v in (force_a, force_b, p)), 'finite contact evidence')
        plane, ball = physical['floor_geom_idx'], physical['sphere_geom_idx']
        require(np.all(((a == plane) & (b == ball)) | ((a == ball) & (b == plane))), 'actual plane/sphere pair')
        require(np.allclose(force_a, -force_b, atol=1e-9, rtol=0), 'opposite contact forces')
        touched.append(bool(len(a) and np.linalg.norm(force_a, axis=1).max() > 0))
    require(all(touched[-100:]), 'last100 samples have native plane contacts')
    cameras = json.loads((OUTPUT / 'cameras.json').read_text()); depth_checks = []
    require(len(cameras) == 4, 'fixed four views')
    for i, row in enumerate(cameras):
        require(row['view'] == i and row['yaw_pitch_rad'] == list(VIEWS[i])
                and row['step_before_after'] == [500, 500], 'fixed camera and no intervening physics')
        yaw, pitch = VIEWS[i]
        forward = [math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)]
        expected_pose = world_from_optical([0., 0., .35], forward, [0., 0., 1.])
        require(np.array_equal(row['world_from_optical'], expected_pose), 'preregistered camera geometry')
        require(np.allclose(np.asarray(row['native_world_from_opengl']) @ np.diag([1., -1., -1., 1.]),
                            expected_pose, atol=1e-7, rtol=0), 'native camera pose readback')
        require(np.allclose(row['native_intrinsics'], INTRINSICS, atol=1e-7, rtol=0)
                and row['near_far_m'] == [.05, 200.], 'native calibration')
        require(row['sampling'] == {'draw_framebuffer_is_single_sample_target': True,
            'draw_framebuffer_is_multisample_target': False, 'samples': 0, 'sample_buffers': 0,
            'multisample_enabled': False, 'pixel_scale': 1}, 'actual single-sample depth target')
        with np.load(OUTPUT / f'depth_{i}.npz', allow_pickle=False) as z: depth = z['optical_depth_m']
        with Image.open(OUTPUT / f'rgb_{i}.png') as image: rgb = np.asarray(image)
        require(depth.shape == (480, 640) and depth.dtype == np.float32
                and rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8, 'native formats')
        require(hashlib.sha256(depth.tobytes()).hexdigest() == row['depth_sha256']
                and hashlib.sha256(rgb.tobytes()).hexdigest() == row['rgb_sha256'], 'raw RGBD identity')
        ref = expected_optical_depth([], row['world_from_optical'], floor_z_m=0.)
        expected = ref['expected_depth_m']; mask = (expected > .22) & (expected < 4.98)
        errors = np.abs(depth[np.ix_(ref['rows'], ref['columns'])][mask] - expected[mask])
        require(len(errors) > 1000 and np.isfinite(errors).all() and errors.max() <= .001,
                'physical-plane ray depths within declared1mm')
        depth_checks.append({'view': i, 'rays': len(errors), 'maximum_error_m': float(errors.max()),
                             'mean_error_m': float(errors.mean())})
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256'])
    return {'status': 'PASS_ALIGNED_FLOOR_INTERFACE_ONLY', 'surface_identity': floor,
            'physics_steps': 500, 'settled_contact_samples': sum(touched[-100:]),
            'minimum_sphere_gap_m': float(gap.min()), 'maximum_settled_gap_error_m': float(np.abs(gap[-100:]).max()),
            'depth_checks': depth_checks, 'navigation_qualified': False, 'contact_model_validated': False,
            'scope': 'static scene geometry, rendered plane depths and sphere contact only; no Go2 mission or hardware'}


def main():
    report = audit()
    write_json(OUTPUT / 'raw_artifact_audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
