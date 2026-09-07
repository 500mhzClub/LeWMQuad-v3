"""Read-only exact-reference comparison of compiled primitive beam queries.

The original nominal observer records are loaded, not recomputed. Observation
times exclude live registration/acquisition/gait/control. Warm-mask repeats are
the SAME observation, not evidence about the next control tick or changed seed.
"""
import json
import time

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.primitive_obstacle_memory_development import PrimitiveObstacleMemory
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.probe_go2_correlated_moment_sensitivity_development import PREDECESSOR, DIRECTORY


def exact(a, b):
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], np.ndarray): np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        else: assert a[key] == b[key], key


def main():
    launch = json.loads(PREDECESSOR.read_text())
    bindings = launch['source_sha256'] | launch['input_sha256'] | launch['artifact_sha256']
    verify_bindings(bindings)
    records = json.loads((DIRECTORY / 'relative_state_observations.json').read_text())
    start = time.perf_counter()
    model = PrimitiveObstacleMemory(ArticulatedCollisionGeometry(URDF), normal_error=.002,
                                     up_error=.001, plane_offset_error=.001, range_error_m=.001,
                                     beam_backend='compiled')
    print(json.dumps({'consumer_construction_and_warmup_ms': 1000 * (time.perf_counter() - start)}), flush=True)
    observations = []
    for tick in range(181):
        p, d = load_rgbd_observation(DIRECTORY, tick); now = p['sensor_state']['decision_ns']
        start = time.perf_counter(); model.observe(p, d, records[tick]['observer'], now_ns=now)
        observations.append(time.perf_counter() - start)
        if tick not in (80, 140, 180): continue
        rows = []; seconds = []
        for backend in ('reference', 'compiled', 'compiled'):
            start = time.perf_counter(); rows.append(model.query_current_primitives(now_ns=now, beam_backend=backend))
            seconds.append(time.perf_counter() - start)
        exact(rows[0], rows[1]); exact(rows[0], rows[2])
        row = rows[0]; ids = np.asarray(row['shape_ids'])
        print(json.dumps({'tick': tick, 'all_reference_compiled_and_repeat_fields_exact': True,
                          'views': len(row['views']), 'reference_compiled_repeat_ms': [1000 * s for s in seconds],
                          'conditionally_clear_primitives': ids[row['conditional_clearance']].tolist(),
                          'foot_contact_candidates': ids[row['foot_contact_candidate']].tolist(),
                          'non_floor_conflicts': ids[row['non_floor_conflict']].tolist(),
                          'floor_penetrations': ids[row['floor_penetration']].tolist(),
                          'all_primitives_conditionally_clear': row['all_primitives_conditionally_clear'],
                          'cached_plane_families': sum(len(f._plane_family_cache) for f in model._prepared.values()),
                          'scanned_cells': sum(v['scanned_pixels'] for v in row['views']),
                          'contact_permitted': False, 'navigation_qualified': False}), flush=True)
    verify_bindings(bindings)
    print(json.dumps({'status': 'PRIMITIVE_BEAM_KERNEL_COMPARISON_COMPLETE', 'frames': 181,
                      'median_observation_subset_ms': 1000 * float(np.median(observations)),
                      'all_predecessor_bindings_verified_before_and_after': True,
                      'scope': 'exact computational comparison; no full-loop timing, navigation, training or hardware claim'}), flush=True)


if __name__ == '__main__': main()
