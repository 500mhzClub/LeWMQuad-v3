"""Read-only raw-registration joint-sensitivity successor diagnostic.

Fixed reused development trace and declared masking/bias-source hypotheses.
No physical result is rescored and no covariance is used to approve motion.
"""
import json
import time

import numpy as np

from lewm.correlated_moment_sensitivity_development import RawRgbdMomentSensitivity, RAW_SHAPES
from lewm.depth_inertial_moment_fusion_development import MomentDepthInertialState
from lewm.depth_inertial_stress_development import perturb
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_correlated_moment_sensitivity_development import PREDECESSOR, DIRECTORY


def main():
    launch = json.loads(PREDECESSOR.read_text())
    bindings = launch['source_sha256'] | launch['input_sha256'] | launch['artifact_sha256']
    verify_bindings(bindings)
    names = ['force_y_bias_from_tick80_scale_0.02_m_s2', 'gyro_z_shared_bias_scale_0.001_rad_s']
    models = [RawRgbdMomentSensitivity(names, difference_step=s) for s in (1e-3, 5e-4)]
    reference = MomentDepthInertialState()
    first_ns = None
    maximum_step_difference = 0.
    timings = []
    for tick in range(181):
        policy, depth = load_rgbd_observation(DIRECTORY, tick)
        fast = load_fast_packet(DIRECTORY, tick)
        now = policy['sensor_state']['decision_ns']
        if first_ns is None: first_ns = now
        p, d = perturb(policy, depth, 'narrow_depth', first_ns=first_ns)
        nominal = reference.observe(p, d, fast, now_ns=now)
        loadings = {name: np.zeros((*shape, 2)) for name, shape in RAW_SHAPES.items()}
        times = p['sensor_state']['sensed']['specific_force']['measured_ns']
        loadings['specific_force'][times >= first_ns + 8_000_000_000, 1, 0] = .02
        loadings['gyro'][:, 2, 1] = .001
        loadings['fast_gyro'][:, 2, 1] = .001
        start = time.perf_counter()
        results = [model.observe(p, d, fast, loadings) for model in models]
        timings.append(time.perf_counter() - start)
        assert all(result['nominal_fusion'] == nominal['fusion'] for result in results)
        difference = float(np.max(np.abs(results[0]['pose_error_factor'] - results[1]['pose_error_factor'])))
        maximum_step_difference = max(maximum_step_difference, difference)
        if tick == 79:
            for model in models: model.retain('before_depth_weakness')
        if tick in (79, 93, 140, 180):
            relative = models[0].relative_moments('before_depth_weakness', [[1, 0, -.3]])
            joint = relative['joint_pose_covariance']
            # Deliberately wrong independent-endpoint comparison, diagnostic
            # only: expose the effect of discarding the actual cross block.
            independent = joint.copy(); independent[:6, 6:] = 0.; independent[6:, :6] = 0.
            j = relative['jacobian'][0]
            print(json.dumps({'tick': tick, 'depth_rank': nominal['fusion']['depth_rank'],
                              'within_old_proxy_budget': nominal['fusion']['usable_under_declared_proxy_budget'],
                              'raw_position_factors_m': results[0]['pose_error_factor'][:3].tolist(),
                              'difference_step_disagreement': difference,
                              'relative_point_std_xyz_m': np.sqrt(np.maximum(0., np.diag(relative['point_covariance_m2'][0]))).tolist(),
                              'incorrect_independent_endpoint_std_xyz_m': np.sqrt(np.maximum(0., np.diag(j @ independent @ j.T))).tolist(),
                              'two_paired_diagnostics_elapsed_ms': 1000 * timings[-1],
                              'calibrated': False, 'navigation_qualified': False}), flush=True)
    verify_bindings(bindings)
    print(json.dumps({'status': 'RAW_DIAGNOSTIC_COMPLETE', 'frames': 181,
                      'all_nominal_fusion_records_exact': True,
                      'maximum_difference_step_disagreement': maximum_step_difference,
                      'two_paired_diagnostics_median_ms': 1000 * float(np.median(timings)),
                      'two_paired_diagnostics_maximum_ms': 1000 * max(timings),
                      'depth_noise_loading_zero': True,
                      'scope': 'two declared bias sources, reused masked development trace; not calibrated covariance or navigation'}), flush=True)


if __name__ == '__main__':
    main()
