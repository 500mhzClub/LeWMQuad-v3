"""Six fixed stationary physical marker probes, not a navigation benchmark."""
from lewm.physical_execution_development import build_case

CASES = ('positive', 'absent', 'occluded', 'red_only', 'reversed', 'separated')


def box(name, centre, size, material='NEUTRAL_WALL'):
    return {'wall_id': name, 'centre_xyz': centre, 'size_xyz': size,
            'yaw_rad': 0., 'material_id': material}


def probe_spec(index):
    if type(index) is not int or not 0 <= index < len(CASES):
        raise ValueError('fixed marker probe index required')
    case = CASES[index]
    spec = build_case('straight', 1.)
    walls = [box(name, centre, size) for name, centre, size in (
        ('arena_front', [2.54, 0., .7], [.08, 5.16, 1.4]),
        ('arena_back', [-2.54, 0., .7], [.08, 5.16, 1.4]),
        ('arena_left', [0., 2.54, .7], [5., .08, 1.4]),
        ('arena_right', [0., -2.54, .7], [5., .08, 1.4]))]
    # Camera optical right is negative body y. Fixed collision boxes are used
    # for both panels; these are additional objects, never substituted walls.
    if case != 'absent':
        spacing = .60 if case == 'separated' else .14
        red_y = -spacing if case == 'reversed' else spacing
        walls.append(box('marker_red_panel', [1.5, red_y, .55], [.08, .25, .50], 'landmark_red'))
        if case != 'red_only':
            walls.append(box('marker_blue_panel', [1.5, -red_y, .55], [.08, .25, .50], 'landmark_blue'))
    if case == 'occluded':
        walls.append(box('marker_occluder', [1.05, 0., .7], [.08, 1.2, 1.4]))
    spec['geometry']['wall_boxes'] = walls
    spec.update(scene_id=f'go2-marker-beacon-development-v1-{case}', family='PHYSICAL_RGB_MARKER_PROBE',
                procedural_seed=2026100200, case_index=index, marker_case=case)
    return spec


def trials():
    return [probe_spec(i) for i in range(len(CASES))]
