"""New continuous-development runs in the declared local predecessor fixtures."""
import copy

from lewm.observed_traversal_scene_development import trials as initial_trials
from lewm.observed_continuation_development import METHODS


def trials():
    result = []
    for initial in initial_trials():
        if initial['method'] != 'always_stop': continue
        case = initial['case_index']
        first = copy.deepcopy(initial['geometry'])
        second = copy.deepcopy(first)
        second['source_node'] = copy.deepcopy(first['target_node'])
        second['target_node'] = {'node_id': 'cell_1_1', 'centre_world': [1.28, 1.28],
            'boundary_polygon_world': [[.68, .68], [1.88, .68], [1.88, 1.88], [.68, 1.88]]}
        second['selected_directed_edge'] = {'edge_id': 'observed-continuation-second-edge',
            'source_node_id': 'cell_1_0', 'target_node_id': 'cell_1_1',
            'opening_segment_world': [[1.88, .64], [.68, .64]],
            'opening_normal_world': [0., 1.],
            'edge_region_polygon_world': copy.deepcopy(second['target_node']['boundary_polygon_world'])}
        second['teacher_route_polyline_world'] = [[1.28, 0.], [1.28, .64], [1.28, 1.28]]
        for method in METHODS:
            spec = copy.deepcopy(initial)
            spec.update(scene_id=f'observed-continuation-development-v1-{case:02d}-{method}',
                family='OBSERVED_CONTINUATION_DEVELOPMENT', method=method,
                procedural_seed=2026092800+case,
                evaluation_leg_geometries=[copy.deepcopy(first), copy.deepcopy(second)])
            result.append(spec)
    return result
