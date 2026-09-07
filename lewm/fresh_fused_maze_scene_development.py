"""Fresh junction/backtracking scene; construction and evaluation only."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math

from lewm_genesis.scene_loader import StaticObject
from scripts.probe_go2_rgbd_motion_scene_development_v1 import pack as base_pack

PITCH = 1.8
PHYSICS_SEED = 2026090610
APPEARANCE_SEED = 2026090611
EDGES = (((0, 0), (1, 0)), ((1, 0), (2, 0)), ((1, 0), (1, 1)),
         ((1, 1), (0, 1)), ((1, 0), (1, -1)), ((1, -1), (2, -1)),
         ((2, -1), (2, -2)))
MARKER_CELL = (2, -2)
MARKER_DIRECTION = (0, -1)


def specification():
    edges = {frozenset(e) for e in EDGES}
    cells = {p for e in EDGES for p in e}
    walls = {}
    for x, y in sorted(cells):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if frozenset(((x, y), (x+dx, y+dy))) in edges:
                continue
            mid = (2*x+dx, 2*y+dy)
            key = (mid, abs(dx))
            walls[key] = dict(wall_id=f'fresh-wall-{mid[0]}-{mid[1]}-{abs(dx)}',
                centre_xyz=[mid[0]*PITCH/2, mid[1]*PITCH/2, .7],
                size_xyz=[.08, PITCH+.08, 1.4] if dx else [PITCH+.08, .08, 1.4],
                yaw_rad=0., material_id='NEUTRAL_WALL')
    boxes = [walls[k] for k in sorted(walls)]
    dx, dy = MARKER_DIRECTION
    cx, cy = (p*PITCH for p in MARKER_CELL)
    centre = [cx+dx*(PITCH/2-.10), cy+dy*(PITCH/2-.10)]
    for color, sign in (('red', 1.), ('blue', -1.)):
        boxes.append(dict(wall_id='marker_'+color+'_panel',
            centre_xyz=[centre[0]-dy*.14*sign, centre[1]+dx*.14*sign, .55],
            size_xyz=[.08, .25, .50], yaw_rad=math.atan2(dy, dx), material_id='landmark_'+color))
    return dict(scene_id='go2-fresh-fused-maze-development-v1',
        family='DEVELOPMENT_COMPLETE_MAZE', procedural_seed=PHYSICS_SEED,
        appearance_seed=APPEARANCE_SEED, appearance_arm='distinctive', memory_arm='episodic',
        method='depth_proposal_fused_navigation', case_index=0, layout_name='fork_backtrack_hidden_south',
        geometry=dict(spawn_se2_world=[0., 0., 0.], wall_boxes=boxes),
        evaluation_layout=dict(cells=[list(p) for p in sorted(cells)],
            edges=[[list(a), list(b)] for a, b in EDGES], marker_cell=list(MARKER_CELL), pitch_m=PITCH))


def pack(spec):
    if spec != specification():
        raise ValueError('exact fresh source-declared mission specification required')
    objects = tuple(StaticObject(object_id=b['wall_id'], kind='wall',
        center_xyz_m=tuple(b['centre_xyz']), size_xyz_m=tuple(b['size_xyz']),
        yaw_rad=b['yaw_rad'], material_id=b['material_id']) for b in spec['geometry']['wall_boxes'])
    original = base_pack()
    identity = hashlib.sha256(json.dumps(dict(objects=[asdict(o) for o in objects],
        spawn=[0., 0., .375], yaw=0.), sort_keys=True).encode()).hexdigest()
    return replace(original, scene_id=spec['scene_id'], family=spec['family'],
        difficulty_tier='JUNCTION_BACKTRACKING_DEVELOPMENT', manifest_sha256=identity,
        static_objects=objects, physics_seed=PHYSICS_SEED, topology_seed=PHYSICS_SEED,
        visual_seed=APPEARANCE_SEED, world_bounds_xy_m=((-2., -6.), (6., 4.)),
        robot=replace(original.robot, spawn_xyz_m=(0., 0., .375), spawn_quat_wxyz=(1., 0., 0., 0.)))
