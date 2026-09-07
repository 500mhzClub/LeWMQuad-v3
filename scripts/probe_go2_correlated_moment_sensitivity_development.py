"""Read-only derivative diagnostic on an already declared development fault.

No output directory, physics, training, navigation, qualification, rescoring or
checkpoint access. The paired raw-gyro observers expose what a frozen-rank
conditional calculation omits. Post-budget estimates remain diagnostics only.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from lewm.correlated_moment_sensitivity_development import CorrelatedMomentSensitivity, SHAPES
from lewm.depth_inertial_moment_fusion_development import MomentDepthInertialState
from lewm.depth_inertial_stress_development import perturb
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / '.generated/go2_depth_inertial_moment_replay_development_v1_attempt_001/launch.json'
DIRECTORY = ROOT / ('.generated/go2_depth_floor_hold_navigation_development_v1_attempt_001/'
                    'go2-depth-floor-hold-navigation-development-v1-north_dogleg')


def main():
    launch = json.loads(PREDECESSOR.read_text())
    bindings = launch['source_sha256'] | launch['input_sha256'] | launch['artifact_sha256']
    verify_bindings(bindings)
    names = ['force_y_bias_from_tick80_scale_0.02_m_s2', 'gyro_z_shared_bias_scale_0.001_rad_s']
    steps = (1e-3, 5e-4)
    models = [CorrelatedMomentSensitivity(names, difference_step=s) for s in steps]
    reference = MomentDepthInertialState()
    raw_gyro_models = [MomentDepthInertialState(), MomentDepthInertialState()]
    first_ns = None
    maximum_step_difference = maximum_missing_registration = 0.
    rank_changes = 0
    for tick in range(181):
        policy, depth = load_rgbd_observation(DIRECTORY, tick)
        fast = load_fast_packet(DIRECTORY, tick)
        now = policy['sensor_state']['decision_ns']
        if first_ns is None: first_ns = now
        p, d = perturb(policy, depth, 'narrow_depth', first_ns=first_ns)
        nominal = reference.observe(p, d, fast, now_ns=now)
        loadings = {name: np.zeros((*shape, 2)) for name, shape in SHAPES.items()}
        times = p['sensor_state']['sensed']['specific_force']['measured_ns']
        loadings['specific_force'][times >= first_ns + 8_000_000_000, 1, 0] = .02
        loadings['gyro'][:, 2, 1] = .001
        loadings['fast_gyro'][:, 2, 1] = .001
        results = [model.observe(p, nominal['depth_state'], fast, loadings) for model in models]
        assert all(result['nominal_fusion'] == nominal['fusion'] for result in results)
        difference = float(np.max(np.abs(results[0]['pose_error_factor'] - results[1]['pose_error_factor'])))
        maximum_step_difference = max(maximum_step_difference, difference)
        raw_positions = []
        for sign, model in zip((1., -1.), raw_gyro_models, strict=True):
            shifted, shifted_fast = deepcopy(p), deepcopy(fast)
            shifted['sensor_state']['sensed']['gyro']['values'][:, 2] += sign * steps[0] * .001
            shifted_fast['values'][:, 2] += sign * steps[0] * .001
            result = model.observe(shifted, d, shifted_fast, now_ns=now)
            rank_changes += result['fusion']['depth_rank'] != nominal['fusion']['depth_rank']
            raw_positions.append(np.asarray(result['fusion']['position_initial_body_m']))
        full_gyro_factor = (raw_positions[0] - raw_positions[1]) / (2 * steps[0])
        missing = float(np.linalg.norm(full_gyro_factor - results[0]['pose_error_factor'][:3, 1]))
        maximum_missing_registration = max(maximum_missing_registration, missing)
        if tick in (0, 79, 93, 140, 180):
            for model in models: model.retain(f'tick{tick}')
        if tick in (79, 93, 140, 180):
            relative = models[0].relative_moments('tick79', [[1, 0, -.3]])
            print(json.dumps({'tick': tick, 'depth_rank': nominal['fusion']['depth_rank'],
                              'within_old_proxy_budget': nominal['fusion']['usable_under_declared_proxy_budget'],
                              'conditional_position_factors_m': results[0]['pose_error_factor'][:3].tolist(),
                              'raw_registration_gyro_position_factor_m': full_gyro_factor.tolist(),
                              'missing_registration_response_norm_m': missing,
                              'difference_step_agreement': difference,
                              'relative_point_covariance_to_tick79_m2': relative['point_covariance_m2'].tolist(),
                              'calibrated': False, 'navigation_qualified': False}), flush=True)
    verify_bindings(bindings)
    print(json.dumps({'status': 'DIAGNOSTIC_COMPLETE', 'frames': 181,
                      'all_nominal_fusion_records_exact': True,
                      'maximum_difference_step_disagreement': maximum_step_difference,
                      'maximum_omitted_registration_response_m': maximum_missing_registration,
                      'perturbed_registration_rank_changes': rank_changes,
                      'scope': 'conditional source sensitivities and raw-registration counterexample; not navigation or calibrated covariance'}), flush=True)


if __name__ == '__main__':
    main()
