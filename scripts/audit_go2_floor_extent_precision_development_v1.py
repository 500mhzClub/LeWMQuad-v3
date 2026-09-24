"""Bound raw artifact recomputation; negative precision outcomes are retained."""
import hashlib
import json

import numpy as np
from PIL import Image

from lewm.causal_depth_observation_development import INTRINSICS
from lewm_genesis.floor_extent_precision_development import (
    EXTENTS_M, VIEWS, camera_pose, check_extent_identity, evaluate,
)
from scripts.run_go2_floor_extent_precision_development_v1 import OUTPUT, ROOT
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import write_json


def require(value, message):
    if not value: raise ValueError(message)


def audit():
    launch = json.loads((OUTPUT / 'launch.json').read_text())
    result = json.loads((OUTPUT / 'result.json').read_text())
    require(result['status'] == 'ACQUISITION_COMPLETE_AUDIT_REQUIRED', 'completed acquisition')
    require(launch['extents_m'] == list(EXTENTS_M) and launch['views'] == [list(v) for v in VIEWS], 'fixed design')
    bindings = launch['source_sha256'] | launch['input_sha256']
    bindings |= {str((OUTPUT / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256'])
    rows = []; poses = []
    for extent in EXTENTS_M:
        directory = OUTPUT / f'extent_{int(extent)}'
        surface = check_extent_identity(json.loads((directory / 'floor_identity.json').read_text()), extent)
        cameras = json.loads((directory / 'cameras.json').read_text())
        require(len(cameras) == 8, 'complete fixed viewpoint population')
        native_poses = []
        for i, c in enumerate(cameras):
            require(c['view'] == i and c['view_parameters'] == list(VIEWS[i]) and c['step_before_after'] == [0, 0], 'fixed camera/clock')
            require(np.array_equal(c['world_from_optical'], camera_pose(VIEWS[i])), 'requested camera transform')
            native = np.asarray(c['native_world_from_opengl'])
            require(np.allclose(native @ np.diag([1., -1., -1., 1.]), camera_pose(VIEWS[i]), atol=2e-7, rtol=0), 'actual native camera pose')
            native_poses.append(native)
            require(np.allclose(c['native_intrinsics'], INTRINSICS, atol=1e-7, rtol=0) and c['near_far_m'] == [.05, 200.], 'fixed native camera')
            require(c['sampling'] == {'draw_framebuffer_is_single_sample_target': True,
                'draw_framebuffer_is_multisample_target': False, 'samples': 0, 'sample_buffers': 0,
                'multisample_enabled': False, 'pixel_scale': 1}, 'single-sample framebuffer')
            with np.load(directory / f'depth_{i}.npz', allow_pickle=False) as f:
                require(set(f.files) == {'optical_depth_m', 'normalized_depth'}, 'exact raw depth leaves')
                depth = f['optical_depth_m']; raw = f['normalized_depth']
            with Image.open(directory / f'rgb_{i}.png') as image: rgb = np.asarray(image)
            require(rgb.dtype == np.uint8 and rgb.shape == (480, 640, 3), 'native RGB format')
            for value, field in ((depth, 'depth_sha256'), (raw, 'raw_buffer_sha256'), (rgb, 'rgb_sha256')):
                require(hashlib.sha256(value.tobytes()).hexdigest() == c[field], 'raw acquisition binding')
            measured = evaluate(VIEWS[i], extent, depth, raw)
            require(measured['native_buffer_reconstruction_exact'], 'raw buffer reconstructs every native pixel exactly')
            rows.append({'extent_m': extent, 'view': i, 'surface': surface, **measured})
        poses.append(np.stack(native_poses))
    require(np.array_equal(poses[0], poses[1]), 'identical native poses in paired scenes')
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256'])
    bounded = rows[8:]; control = rows[:8]
    return {'status': 'RAW_ARTIFACTS_VERIFIED_PRECISION_REPORTED', 'views': rows,
        'bounded32_native_within1mm_count': sum(r['native_within1mm'] for r in bounded),
        'control1000_native_within1mm_count': sum(r['native_within1mm'] for r in control),
        'bounded32_all_views_within1mm': all(r['native_within1mm'] for r in bounded),
        'bounded32_native_error_lower_count': sum(b['native_max_error_m'] < a['native_max_error_m'] for a, b in zip(control, bounded, strict=True)),
        'sensor_calibrated': False, 'contact_model_validated': False, 'navigation_qualified': False}


def main():
    report = audit()
    write_json(OUTPUT / 'raw_artifact_audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
