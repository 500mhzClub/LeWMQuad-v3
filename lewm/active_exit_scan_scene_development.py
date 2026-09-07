"""Fresh physical scan specimens; geometry is NEVER passed to the observer."""
import copy

from lewm.multijunction_routes_development import route_spec

DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
MOTIFS = (('dead_end', 1), ('corner', 3), ('tee', 11), ('cross', 15))


def scan_scenes():
    result = []
    for motif, mask in MOTIFS:
        for width in (.9, 1.2):
            for heading in (-.15, .15):
                index = len(result)
                spec = route_spec('left_right', width)
                pitch = width + .08
                open_directions = [d for i, d in enumerate(DIRECTIONS) if mask & (1 << i)]
                cells = {(0, 0), *open_directions}
                links = {tuple(sorted(((0, 0), d))) for d in open_directions}
                walls = {}
                for cell in sorted(cells):
                    for direction in DIRECTIONS:
                        neighbor = (cell[0] + direction[0], cell[1] + direction[1])
                        if tuple(sorted((cell, neighbor))) in links:
                            continue
                        midpoint = (2 * cell[0] + direction[0], 2 * cell[1] + direction[1])
                        key = (midpoint, abs(direction[0]))
                        walls[key] = {'wall_id': f'scan-wall-{midpoint[0]}-{midpoint[1]}-{abs(direction[0])}',
                                      'centre_xyz': [midpoint[0] * pitch / 2, midpoint[1] * pitch / 2, .3],
                                      'size_xyz': [.08, pitch + .08, .6] if direction[0] else [pitch + .08, .08, .6],
                                      'yaw_rad': 0., 'material_id': 'NEUTRAL_WALL'}
                geometry = copy.deepcopy(spec['geometry'])
                geometry['spawn_se2_world'] = [0., 0., heading]
                geometry['wall_boxes'] = [walls[key] for key in sorted(walls)]
                spec.update(scene_id=f'active-exit-scan-development-v1-{index:02d}-{motif}',
                            family='ACTIVE_EXIT_SCAN_DEVELOPMENT', case_index=index,
                            procedural_seed=2026092500 + index, motif=motif, width_m=width,
                            initial_heading_rad=heading, geometry=geometry,
                            evaluation_open_directions=[list(d) for d in open_directions],
                            evaluation_opening_midpoints_world=[[d[0] * pitch / 2, d[1] * pitch / 2] for d in open_directions])
                # Remove unused route history so this specimen cannot masquerade
                # as an executed multi-junction route or a prepopulated runtime map.
                for key in ('route_geometries', 'route_cells', 'graph_connections'):
                    spec.pop(key, None)
                result.append(spec)
    return result
