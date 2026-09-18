"""Compare routing semantics and timing on four saved observed-map queries."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from lewm import axis_aligned_fine_connectivity_development as fast
from lewm import cached_fine_connectivity_development as old
from lewm.cached_fine_goal_route_development import cached_graph_geometry
from lewm.fine_stored_obstacle_routing_development import _cached_clearance
from scripts import run_go2_polygon_floor_repeatability_development as run


def scientific(value):
    return {k: ({kk: vv for kk, vv in v.items() if kk != 'routing_s'}
                if k == 'fine_goal_route' else v) for k, v in value.items()}


def main():
    root = run.BASE / run.root_name(1) / 'fine_goal_routing_profile_v1'
    source = root / 'result.json'
    output = root / 'axis_geometry_verification_v1.json'
    if output.exists():
        raise ValueError('preserve completed verification')
    saved = json.loads(source.read_text())
    fast.warmup()
    rows = []
    for row in saved['rows']:
        snapshot = SimpleNamespace(**{k: frozenset(map(tuple, row[k]))
            for k in ('floor', 'occupied', 'fine_occupied')})
        position = np.asarray(row['position_map_xy_m'])
        goal = np.asarray(row['goal_map_xy_m'])
        initial = dict(status='DIAGNOSTIC_FINE_GOAL_QUERY', nominal_radius_m=.45)
        results, timings = {}, {'original': [], 'axis': []}
        # Both orders, with the same geometry/search cache resets per query.
        for name in ('original', 'axis', 'axis', 'original'):
            for cache in (old.search_graph, fast.search_graph, old.floor_index,
                    cached_graph_geometry, fast.axis_graph_geometry, _cached_clearance):
                cache.cache_clear()
            call = old.fine_goal_route if name == 'original' else fast.fine_goal_route
            started = time.perf_counter()
            result = call(snapshot, position, goal, initial)
            timings[name].append((time.perf_counter() - started) * 1000)
            assert scientific(result) == scientific(row['result'])
            results[name] = scientific(result)
        assert results['original'] == results['axis']
        rows.append(dict(frame=row['frame'], results_identical=True,
            cold_query_ms=timings, result=results['axis']))
        print('AXIS_ROUTING_VERIFIED', row['frame'], timings, flush=True)
    paths = ('lewm/axis_aligned_fine_connectivity_development.py',
        'scripts/verify_go2_axis_fine_routing_development.py',
        'lewm/tests/test_axis_aligned_fine_connectivity_development.py')
    output.write_text(json.dumps(dict(rows=rows, source_sha256={p:
        hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        all_four_saved_query_results_identical=True,
        same_observed_geometry_costs_clearance_and_tie_breaking=True,
        native_navigation_executed=False, alternative_navigation_outcome_proven=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
