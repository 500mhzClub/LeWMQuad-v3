"""Four local development fixtures, not runtime graph or destination inputs."""
import copy

from lewm.multijunction_routes_development import route_spec, connection

METHODS = ('always_stop', 'directional_gait', 'direct_direct', 'supervised_rollout', 'jepa_rollout')


def trials():
    rows = []
    for destination in ('corner', 'tee'):
        for offset, heading in ((-.06, -.1), (.06, .1)):
            case = len(rows) // len(METHODS)
            base = route_spec('left_right', 1.2)
            links = {connection((0, 0), (1, 0)), connection((1, 0), (1, 1))}
            if destination == 'tee': links.add(connection((1, 0), (2, 0)))
            cells = {c for edge in links for c in edge}
            walls = {}
            for cell in sorted(cells):
                for d in ((1,0),(-1,0),(0,1),(0,-1)):
                    neighbor = (cell[0]+d[0], cell[1]+d[1])
                    if connection(cell, neighbor) in links: continue
                    mid = (2*cell[0]+d[0], 2*cell[1]+d[1])
                    key = (mid, abs(d[0]))
                    walls[key] = {'wall_id': f'traversal-wall-{mid[0]}-{mid[1]}-{abs(d[0])}',
                        'centre_xyz': [mid[0]*.64, mid[1]*.64, .3],
                        'size_xyz': [.08,1.36,.6] if d[0] else [1.36,.08,.6],
                        'yaw_rad': 0., 'material_id': 'NEUTRAL_WALL'}
            geometry = copy.deepcopy(base['geometry'])
            geometry['wall_boxes'] = [walls[k] for k in sorted(walls)]
            geometry['spawn_se2_world'] = [0.,offset,heading]
            for method in METHODS:
                spec = {k: copy.deepcopy(v) for k,v in base.items() if k not in ('route_geometries','route_cells','graph_connections')}
                spec.update(scene_id=f'observed-traversal-development-v1-{case:02d}-{method}',
                    family='OBSERVED_TRAVERSAL_DEVELOPMENT', case_index=case, procedural_seed=2026092700+case,
                    destination_motif=destination, initial_offset_m=offset, initial_heading_rad=heading,
                    method=method, geometry=copy.deepcopy(geometry))
                rows.append(spec)
    return rows
