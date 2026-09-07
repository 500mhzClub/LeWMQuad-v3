"""Read-only observation-bound current-primitive consumer diagnostic.

Saved original nominal depth records supply the observer input. Timing therefore
excludes live nominal registration, acquisition and gait/control; it is not a
full-loop benchmark. No physics, navigation rerun or physical-result rescoring.
"""
import json
import time

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.primitive_obstacle_memory_development import PrimitiveObstacleMemory
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.probe_go2_correlated_moment_sensitivity_development import PREDECESSOR, DIRECTORY


def main():
    launch = json.loads(PREDECESSOR.read_text())
    bindings = launch['source_sha256'] | launch['input_sha256'] | launch['artifact_sha256']
    verify_bindings(bindings)
    records = json.loads((DIRECTORY / 'relative_state_observations.json').read_text())
    model = PrimitiveObstacleMemory(ArticulatedCollisionGeometry(URDF), normal_error=.002,
                                     up_error=.001, plane_offset_error=.001, range_error_m=.001)
    observe_seconds = []
    for tick in range(181):
        p, d = load_rgbd_observation(DIRECTORY, tick); now = p['sensor_state']['decision_ns']
        start = time.perf_counter()
        model.observe(p, d, records[tick]['observer'], now_ns=now)
        observe_seconds.append(time.perf_counter() - start)
        if tick not in (80, 140, 180): continue
        start = time.perf_counter(); row = model.query_current_primitives(now_ns=now)
        query_seconds = time.perf_counter() - start
        ids = np.asarray(row['shape_ids'])
        print(json.dumps({'tick': tick, 'retained_views': len(row['views']),
                          'conditionally_clear_primitives': ids[row['conditional_clearance']].tolist(),
                          'foot_contact_candidates': ids[row['foot_contact_candidate']].tolist(),
                          'non_floor_conflicts': ids[row['non_floor_conflict']].tolist(),
                          'floor_penetrations': ids[row['floor_penetration']].tolist(),
                          'view_diagnostics': row['views'],
                          'all_primitives_conditionally_clear': row['all_primitives_conditionally_clear'],
                          'observation_subset_ms': 1000 * observe_seconds[-1],
                          'query_ms': 1000 * query_seconds, 'contact_permitted': False,
                          'future_gait_qualified': False, 'navigation_qualified': False}), flush=True)
    verify_bindings(bindings)
    print(json.dumps({'status': 'PRIMITIVE_OBSTACLE_MEMORY_DIAGNOSTIC_COMPLETE', 'frames': 181,
                      'median_observation_subset_ms': 1000 * float(np.median(observe_seconds)),
                      'all_predecessor_bindings_verified_before_and_after': True,
                      'scope': 'current measured posture and supplied sensor-error model only; no full-loop timing or navigation'}), flush=True)


if __name__ == '__main__': main()
