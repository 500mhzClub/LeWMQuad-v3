"""Read-only exactness/timing comparison of paired registration implementations.

Original and previously declared narrow-depth development traces, fixed three
source hypotheses, no experiment output or physical-result rescoring.
"""
import json
import time

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.certified_registration_reuse_development import CertifiedRgbdMomentSensitivity
from lewm.correlated_moment_sensitivity_development import RawRgbdMomentSensitivity, RAW_SHAPES
from lewm.depth_inertial_stress_development import perturb
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_correlated_moment_sensitivity_development import PREDECESSOR, DIRECTORY


def exact(actual, expected):
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            if not np.array_equal(actual[key], value): raise ValueError('paired array differs: ' + key)
        elif actual[key] != value:
            raise ValueError('paired field differs: ' + key)


def describe(values):
    return {'median_ms': 1000 * float(np.median(values)), 'maximum_ms': 1000 * max(values)}


def main(*, factory=CertifiedRgbdMomentSensitivity):
    launch = json.loads(PREDECESSOR.read_text())
    bindings = launch['source_sha256'] | launch['input_sha256'] | launch['artifact_sha256']
    verify_bindings(bindings)
    names = ['shared_roll_gyro_bias_0.001_rad_s', 'force_y_bias_0.02_m_s2_from_tick80',
             'shared_depth_range_scale_0.002']
    for condition in ('original', 'narrow_depth'):
        models = [RawRgbdMomentSensitivity(names, difference_step=.01),
                  factory(names, difference_step=.01)]
        times = [[], []]; certified = searched = extra = fallback = 0
        first_ns = None; completed = 0; terminal = None
        for tick in range(181):
            p, d = load_rgbd_observation(DIRECTORY, tick); f = load_fast_packet(DIRECTORY, tick)
            now = p['sensor_state']['decision_ns']
            if first_ns is None: first_ns = now
            if condition != 'original': p, d = perturb(p, d, condition, first_ns=first_ns)
            loading = {name: np.zeros((*shape, 3)) for name, shape in RAW_SHAPES.items()}
            loading['gyro'][:, 0, 0] = .001; loading['fast_gyro'][:, 0, 0] = .001
            ts = p['sensor_state']['sensed']['specific_force']['measured_ns']
            loading['specific_force'][ts >= first_ns + 8_000_000_000, 1, 1] = .02
            loading['depth_m'][..., 2] = .002 * d['depth_m']
            results = [None, None]; faults = [None, None]
            # Alternate order to avoid always warming inputs for one method.
            for index in ((0, 1) if tick % 2 else (1, 0)):
                start = time.perf_counter()
                try: results[index] = models[index].observe(p, d, f, loading)
                except SensorContractError as error: faults[index] = str(error)
                times[index].append(time.perf_counter() - start)
            if any(faults):
                if faults[0] != faults[1]: raise ValueError('fault outcome differs between implementations')
                terminal = {'tick': tick, 'matched_fault': faults[0]}
                break
            exact(results[1], results[0]); completed += 1
            for row in results[1]['registration_query_accounting']:
                certified += row.get('certified_queries', 0); searched += row.get('searched_queries', 0)
                extra += row.get('reference_queries', 0); fallback += row.get('lineage_fallback_iterations', 0)
            if tick in (0, 79):
                for model in models: model.retain(f'tick{tick}')
            if tick in (79, 93, 140, 180):
                exact(models[1].relative_moments('tick79', [[1., 0, -.3]]),
                      models[0].relative_moments('tick79', [[1., 0, -.3]]))
                print(json.dumps({'condition': condition, 'tick': tick, 'paired_outputs_exact': True,
                                  'original_ms': 1000 * times[0][-1], 'accelerated_ms': 1000 * times[1][-1],
                                  'certified_queries_cumulative': certified,
                                  'searched_queries_cumulative': searched,
                                  'additional_reference_queries_cumulative': extra,
                                  'lineage_fallback_iterations_cumulative': fallback,
                                  'shared_surface_hits': results[1]['shared_surface_hits'],
                                  'shared_cloud_hits': results[1]['shared_cloud_hits'],
                                  'within_old_proxy_budget': results[0]['nominal_fusion']['usable_under_declared_proxy_budget'],
                                  'navigation_qualified': False}), flush=True)
        print(json.dumps({'condition': condition, 'status': 'EXACT_DIAGNOSTIC_COMPLETE',
                          'accelerated_implementation': factory.__name__,
                          'completed_frames': completed, 'terminal_fault': terminal,
                          'original': describe(times[0]), 'accelerated': describe(times[1]),
                          'certified_queries': certified, 'searched_queries': searched,
                          'additional_reference_queries': extra, 'lineage_fallback_iterations': fallback,
                          'scope': 'paired computation equivalence, not uncertainty calibration or navigation'}), flush=True)
    verify_bindings(bindings)
    print('PREDECESSOR_BINDINGS_PRESERVED', flush=True)


if __name__ == '__main__':
    main()
