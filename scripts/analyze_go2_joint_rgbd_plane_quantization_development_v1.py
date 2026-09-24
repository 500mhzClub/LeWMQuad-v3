"""Isolate observed-cell perturbation rounding; never upgrade a runtime sensor."""
import json

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.measured_plane_obstacle_memory_development import MeasuredPlaneHypothesis
from lewm.paired_rgbd_physical_plane_development import minimum_gaps
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_fresh_fused_maze_development_v1 import exact
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import OUTPUT, INPUT, STEPS, preflight
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json

IDENTITIES = {'launch.json': '5361a89c73762d789cc431c3b4a35c24c440916bb1af17bf9720c22dde1d808a',
    'result.json': '12cd0cba81d4600f5777bbc811500cc80172f8e475a31343f35c50dbf35f6fe6',
    'numerical_comparison.json': '45a722a379e12d54791def074b1dc052ffe739d949d3bf8c3e1ba924f47e915b'}


def analyze():
    own = 'scripts/analyze_go2_joint_rgbd_plane_quantization_development_v1.py'
    source = {own: digest(ROOT/own)}
    bound = {str((OUTPUT/n).relative_to(ROOT)): h for n, h in IDENTITIES.items()}
    verify_bindings(bound | source)
    launch, result = [read_json(OUTPUT, n) for n in ('launch.json', 'result.json')]
    exact(preflight(), launch); verify(launch)
    bound |= {str((OUTPUT/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    verify_bindings(bound)
    steps = [read_json(OUTPUT, f'step_{i:02d}.json')['physical_gap_relations']['initial'] for i in range(2)]
    initial, depth = load_rgbd_observation(INPUT/'mission', 0)
    terminal, _ = load_rgbd_observation(INPUT/'mission', 218)
    decisions = read_json(INPUT/'mission', 'task_decisions.json'); end = decisions[-1]['controller']
    R = np.asarray(end['global_orientation']['rotation_initial_body_from_current_body'])
    p = np.asarray(end['sensor_fusion']['position_initial_body_m'])
    q = terminal['sensor_state']['sensed']['joints']['values'][-1, :12]
    mean = initial['sensor_state']['sensed']['specific_force']['values'].mean(axis=0)
    up = (9.81*mean/np.linalg.norm(mean))/9.81
    frame = PreparedFloorFrame(depth['depth_m'], depth['valid'], up)
    cell = MeasuredPlaneHypothesis.from_frame(frame).cell_for(frame)
    assert list(cell) == steps[0]['hypothesis_cells_rc'][0] == steps[1]['hypothesis_cells_rc'][0]
    cells = np.asarray(cell)[None]+np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    yy, xx = cells.T; z = depth['depth_m'][yy, xx]
    assert depth['valid'][yy, xx].all()
    T = np.asarray(BODY_FROM_OPTICAL); geometry = ArticulatedCollisionGeometry(URDF)
    ids = [s['shape_id'] for s in geometry.supports(q, np.eye(3))['shapes']]
    rows = []
    for mode in ('actual_float32_perturbation', 'diagnostic_unrounded_float64_perturbation'):
        factors = []
        for step in STEPS:
            columns = []
            for loading in (.001*z.astype(float), np.full(4, .001)):
                values = []
                for sign in (1., -1.):
                    perturbed = z.astype(float)+sign*step*loading
                    if mode == 'actual_float32_perturbation': perturbed = perturbed.astype(np.float32)
                    optical = np.column_stack((perturbed*(xx+.5-320)/FOCAL, perturbed*(yy+.5-240)/FOCAL, perturbed))
                    patch = optical@T[:3, :3].T+T[:3, 3]
                    a = patch[0]; n = np.cross(patch[1]-a, patch[2]-a); n /= np.linalg.norm(n)
                    if n@up < 0: n = -n
                    values.append(minimum_gaps(geometry, q, a, n, R, p))
                columns.append((values[0]-values[1])/(2*step))
            factors.append(np.stack(columns, axis=1))
        if mode == 'actual_float32_perturbation':
            for observed, saved in zip(factors, steps, strict=True):
                np.testing.assert_allclose(observed, np.asarray(saved['floor_only_gap_factor_m'])[:, :2], atol=1e-12, rtol=0)
        rows.append(dict(mode=mode, first_step_factors_m=factors[0].tolist(), second_step_factors_m=factors[1].tolist(),
            per_primitive_step_difference_m=np.linalg.norm(factors[0]-factors[1], axis=1).tolist(),
            maximum_step_difference_m=float(np.linalg.norm(factors[0]-factors[1], axis=1).max())))
    verify(launch); exact(preflight(), launch); verify_bindings(bound | source)
    return dict(status='INITIAL_MEASURED_CELL_ROUNDING_CAUSAL_DIAGNOSTIC_COMPLETE', source_sha256=source,
        identity_sha256=IDENTITIES, shape_ids=ids, source_ids=steps[0]['source_ids'][:2], hypothesis_cell_rc=cell,
        original_float32_saved_floor_factors_reconstructed=True, original_measured_pixel_ranges_m=z.tolist(),
        comparisons=rows, original_sensor_quantization_and_runtime_unchanged=True,
        diagnostic_float64_not_installed=True, better_plane_estimator_validated=False,
        covariance_calibrated=False, navigation_action_permitted=False)


if __name__ == '__main__':
    target = OUTPUT/'plane_quantization_diagnostic.json'
    if target.exists() or target.is_symlink(): raise ValueError('fresh independent rounding diagnosis only')
    result = analyze(); write_json(target, result); print(result['status'], flush=True)
