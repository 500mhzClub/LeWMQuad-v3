"""Two fresh connected branch-choice layouts, paired across memory conditions.

Geometry is construction/evaluation only. The navigation runtime never receives
these layouts, marker locations, canonical cells or connection labels.
"""
from copy import deepcopy
import math

from lewm.physical_execution_development import build_case
from lewm.whole_task_navigation_development import MEMORY_ARMS

PITCH = 1.44
LAYOUTS = (
    {'name': 'north_dogleg', 'edges': (((0, 0), (1, 0)), ((1, 0), (2, 0)),
        ((1, 0), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (2, 2))),
     'marker_cell': (2, 2), 'marker_facing_direction': (1, 0)},
    {'name': 'south_branch', 'edges': (((0, 0), (1, 0)), ((1, 0), (1, 1)),
        ((1, 1), (1, 2)), ((1, 0), (2, 0)), ((2, 0), (2, -1)), ((2, -1), (2, -2))),
     'marker_cell': (2, -2), 'marker_facing_direction': (0, -1)})


def trial_specs():
    result = []
    for index, layout in enumerate(LAYOUTS):
        edges = {frozenset(edge) for edge in layout['edges']}
        cells = {cell for edge in layout['edges'] for cell in edge}
        walls = {}
        for cell in sorted(cells):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (cell[0]+dx, cell[1]+dy)
                if frozenset((cell, neighbor)) in edges:
                    continue
                mid = (2*cell[0]+dx, 2*cell[1]+dy)
                key = (mid, abs(dx))
                walls[key] = {'wall_id': f'task-wall-{mid[0]}-{mid[1]}-{abs(dx)}',
                              'centre_xyz': [mid[0]*PITCH/2, mid[1]*PITCH/2, .7],
                              'size_xyz': [.08, PITCH+.08, 1.4] if dx else [PITCH+.08, .08, 1.4],
                              'yaw_rad': 0., 'material_id': 'NEUTRAL_WALL'}
        boxes = [walls[k] for k in sorted(walls)]
        dx, dy = layout['marker_facing_direction']
        cx, cy = [v*PITCH for v in layout['marker_cell']]
        centre = [cx+dx*(PITCH/2-.10), cy+dy*(PITCH/2-.10)]
        for color, sign in (('red', 1.), ('blue', -1.)):
            boxes.append({'wall_id': 'marker_'+color+'_panel',
                          'centre_xyz': [centre[0]-dy*.14*sign, centre[1]+dx*.14*sign, .55],
                          'size_xyz': [.08, .25, .50], 'yaw_rad': math.atan2(dy, dx),
                          'material_id': 'landmark_'+color})
        for arm in MEMORY_ARMS:
            spec = build_case('straight', 1.)
            spec['geometry'] = {'spawn_se2_world': [0., 0., 0.], 'wall_boxes': deepcopy(boxes)}
            spec.update(scene_id=f'go2-whole-task-development-v1-{layout["name"]}-{arm}',
                        family='WHOLE_TASK_RGB_MEMORY_DEVELOPMENT', procedural_seed=2026100300+index,
                        case_index=index, layout_name=layout['name'], memory_arm=arm, method='fixed_forward')
            spec['evaluation_layout'] = {'cells': [list(c) for c in sorted(cells)],
                                         'edges': [[list(a), list(b)] for a, b in layout['edges']],
                                         'marker_cell': list(layout['marker_cell']), 'pitch_m': PITCH}
            result.append(spec)
    return result
