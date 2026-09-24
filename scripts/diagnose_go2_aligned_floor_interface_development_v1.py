"""Read-only decomposition AFTER the strict sphere-impact audit failed.

Preserves that failure. No threshold changes, new simulation, rescore, policy
input, contact permission or calibration claim. Prints saved surface/render
checks separately from impact penetration so next work targets the right model.
"""
import json
from pathlib import Path

import numpy as np

from lewm.causal_depth_observation_development import INTRINSICS
from lewm.visual_surface_depth_evaluation_development import expected_optical_depth
from lewm_genesis.aligned_floor_development import check_aligned_floor_identity
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_aligned_floor_interface_development_v1 import OUTPUT, ROOT, verify_native_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import digest


def main():
    own = digest(Path(__file__))
    launch = json.loads((OUTPUT / 'launch.json').read_text())
    result = json.loads((OUTPUT / 'result.json').read_text())
    bindings = launch['source_sha256'] | launch['input_sha256']
    bindings |= {str((OUTPUT / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256'])
    surface = check_aligned_floor_identity(json.loads((OUTPUT / 'floor_identity.json').read_text()))
    with np.load(OUTPUT / 'sphere_trace.npz', allow_pickle=False) as f:
        positions = f['position_world_m']; steps = f['step']
    if positions.shape != (500, 3) or not np.array_equal(steps, np.arange(1, 501)) or not np.isfinite(positions).all():
        raise ValueError('complete finite trace required')
    gap = positions[:, 2] - .022
    contacts = json.loads((OUTPUT / 'contacts.json').read_text())
    touched = [bool(len(row['geom_a']) and np.linalg.norm(row['force_a'], axis=1).max() > 0) for row in contacts]
    print(json.dumps({'surface': surface, 'minimum_gap_m': float(gap.min()),
        'minimum_gap_step': int(gap.argmin() + 1), 'samples_below_minus1mm': int((gap < -.001).sum()),
        'maximum_settled_error_m': float(np.abs(gap[-100:]).max()),
        'settled_contact_samples': sum(touched[-100:]),
        'strict_all_trace1mm_audit_failed': bool(gap.min() < -.001),
        'source_sha256': own}), flush=True)
    cameras = json.loads((OUTPUT / 'cameras.json').read_text())
    if len(cameras) != 4: raise ValueError('four saved cameras required')
    for row in cameras:
        i = row['view']
        with np.load(OUTPUT / f'depth_{i}.npz', allow_pickle=False) as f: d = f['optical_depth_m']
        ref = expected_optical_depth([], row['world_from_optical'], floor_z_m=0.)
        old = expected_optical_depth([], row['world_from_optical'], floor_z_m=-.005)
        mask = (ref['expected_depth_m'] > .22) & (ref['expected_depth_m'] < 4.98)
        measured = d[np.ix_(ref['rows'], ref['columns'])][mask]
        error = np.abs(measured - ref['expected_depth_m'][mask])
        legacy = np.abs(measured - old['expected_depth_m'][mask])
        if not len(error) or not np.isfinite(error).all(): raise ValueError('finite saved depth rays required')
        print(json.dumps({'view': i, 'physical_plane_rays': int(mask.sum()),
            'maximum_physical_plane_depth_error_m': float(error.max()),
            'mean_physical_plane_depth_error_m': float(error.mean()),
            'maximum_legacy_minus5mm_plane_depth_error_m': float(legacy.max()),
            'native_intrinsics_match': bool(np.allclose(row['native_intrinsics'], INTRINSICS, atol=1e-7, rtol=0)),
            'step_before_after': row['step_before_after'], 'sampling': row['sampling']}), flush=True)
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256'])
    if digest(Path(__file__)) != own: raise ValueError('diagnostic source changed')
    print(json.dumps({'status': 'FAILED_ASSAY_DECOMPOSITION_COMPLETE', 'all_bindings_preserved': True,
        'original_strict_audit_failure_preserved': True, 'navigation_qualified': False,
        'scope': 'read-only post-failure decomposition; not a new acceptance or physical qualification'}), flush=True)


if __name__ == '__main__': main()
