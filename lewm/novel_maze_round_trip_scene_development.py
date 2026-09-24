"""Four fixed prospective mazes; geometry/routes are evaluator-only evidence.

Import performs no input materialization, model access or native execution.
The controller-facing mission contains instructed coordinates, never topology.
"""
from collections import deque
from dataclasses import replace
import hashlib
import json
import random
from lewm.counterfactual_maze_development import topology_identity
from lewm_genesis.scene_loader import StaticObject
from scripts.probe_go2_rgbd_motion_scene_development_v1 import pack as base_pack

LAYOUT_COUNT = 4
PITCH_M = 1.3
WALL_THICKNESS_M = .08
START = (-1, 0)
CELLS = tuple((x, y) for x in range(-1, 3) for y in range(-1, 3))
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def edge(a, b):
    return tuple(sorted((a, b)))


def graph(index):
    if type(index) is not int or not 0 <= index < LAYOUT_COUNT:
        raise ValueError('four fixed prospective maze indices required')
    rng = random.Random(2026091600 + index)
    links = {edge(START, (0, 0))}
    parent = {c: c for c in CELLS}
    def root(c):
        while parent[c] != c: c = parent[c]
        return c
    def merge(a, b): parent[root(a)] = root(b)
    merge(START, (0, 0))
    candidates = [edge(c, (c[0]+dx, c[1]+dy)) for c in CELLS
        for dx, dy in ((1, 0), (0, 1))
        if (c[0]+dx, c[1]+dy) in parent
        and START not in (c, (c[0]+dx, c[1]+dy))]
    rng.shuffle(candidates)
    for a, b in candidates:
        if root(a) != root(b): links.add(edge(a, b)); merge(a, b)
    if len(links) != len(CELLS)-1 or len({root(c) for c in CELLS}) != 1:
        raise ValueError('connected tree required; no resampling')
    return tuple(sorted(links))


def evaluator_route(links):
    """Farthest-cell task definition; this route is never a policy input."""
    adjacency = {c: [] for c in CELLS}
    for a, b in links: adjacency[a].append(b); adjacency[b].append(a)
    parent = {START: None}; distance = {START: 0}; queue = deque([START])
    while queue:
        c = queue.popleft()
        for other in sorted(adjacency[c]):
            if other not in parent:
                parent[other] = c; distance[other] = distance[c]+1; queue.append(other)
    if len(parent) != len(CELLS): raise ValueError('complete connected maze required')
    goal = min(CELLS, key=lambda c: (-distance[c], c))
    route = []; here = goal
    while here is not None: route.append(here); here = parent[here]
    return list(reversed(route))


def specification(index):
    links = graph(index); route = evaluator_route(links); walls = {}
    for x, y in CELLS:
        for dx, dy in NEIGHBOURS:
            if edge((x, y), (x+dx, y+dy)) in links: continue
            mx, my = 2*x+dx, 2*y+dy; key = (mx, my, abs(dx))
            walls[key] = dict(wall_id=f'novel_wall_{mx}_{my}_{abs(dx)}',
                centre_xyz=[mx*PITCH_M/2, my*PITCH_M/2, .7],
                size_xyz=[WALL_THICKNESS_M, PITCH_M+WALL_THICKNESS_M, 1.4] if dx
                    else [PITCH_M+WALL_THICKNESS_M, WALL_THICKNESS_M, 1.4],
                yaw_rad=0., material_id='NEUTRAL_WALL')
    return dict(scene_id=f'novel-maze-round-trip-development-v1-{index:02d}',
        layout_index=index, family='NOVEL_MAZE_ROUND_TRIP_DEVELOPMENT',
        data_role='prospective_navigation_development', procedural_seed=2026091600+index,
        appearance_seed=2026091610+index, appearance_arm='distinctive', friction_mu=1.,
        render_near_m=.005, visual_surface_contract='floor_first_variable_height_union_walls',
        geometry=dict(spawn_se2_world=[START[0]*PITCH_M, START[1]*PITCH_M, 0.],
            wall_boxes=[walls[k] for k in sorted(walls)]),
        evaluation_layout=dict(cells=[list(c) for c in CELLS],
            edges=[[list(a), list(b)] for a, b in links], goal_cell=list(route[-1]),
            shortest_outbound_route=[list(c) for c in route],
            shortest_outbound_distance_m=(len(route)-1)*PITCH_M,
            topology_sha256_dihedral=topology_identity(links), pitch_m=PITCH_M),
        native_execution=False, navigation_qualified=False)


def public_mission(index):
    """Coordinate instruction only: visit the goal, dwell, then return home.

    Coordinates do not assert free space or reveal a route. Collector/controller
    integration must preserve this boundary; no such integration is claimed here.
    """
    goal = evaluator_route(graph(index))[-1]
    return dict(goal_initial_body_xy_m=[(goal[i]-START[i])*PITCH_M for i in range(2)],
        return_initial_body_xy_m=[0., 0.], require_return_after_goal=True)


def pack(spec):
    if spec != specification(spec['layout_index']):
        raise ValueError('exact fixed prospective maze specification required')
    original = base_pack()
    objects = tuple(StaticObject(object_id=b['wall_id'], kind='wall',
        center_xyz_m=tuple(b['centre_xyz']), size_xyz_m=tuple(b['size_xyz']),
        yaw_rad=b['yaw_rad'], material_id=b['material_id']) for b in spec['geometry']['wall_boxes'])
    spawn = (*spec['geometry']['spawn_se2_world'][:2], .375)
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return replace(original, scene_id=spec['scene_id'], family=spec['family'],
        difficulty_tier='MULTIJUNCTION_OUTBOUND_AND_RETURN', manifest_sha256=identity,
        static_objects=objects, physics_seed=spec['procedural_seed'], topology_seed=spec['procedural_seed'],
        visual_seed=spec['appearance_seed'], world_bounds_xy_m=((-2.1, -2.1), (3.4, 3.4)),
        camera=replace(original.camera, near_m=.005),
        robot=replace(original.robot, spawn_xyz_m=spawn, spawn_quat_wxyz=(1., 0., 0., 0.)))
