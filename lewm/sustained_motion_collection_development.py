"""Fixed supervised development motion; not a navigation or safety policy."""
from dataclasses import replace
import hashlib
import json
import math

import numpy as np

from lewm_genesis.scene_loader import StaticObject
from scripts.probe_go2_rgbd_motion_scene_development_v1 import pack as predecessor_pack
from scripts.run_go2_causal_rgb_body_capture_development_v1 import probe_spec

TRIALS = ('fit', 'validation')
DEFINITIONS = {
    'fit': dict(physics_seed=2026090622, appearance_seed=2026090624, spawn=(-.5, -.3, 0.)),
    'validation': dict(physics_seed=2026090623, appearance_seed=2026090625, spawn=(-.5, .3, .04)),
}
SEGMENTS = (
    ('initial_weak_observation', 10, (0., 0., 0.)),
    ('sustained_forward', 150, (.12, 0., 0.)),
    ('forward_brake', 10, (0., 0., 0.)),
    ('left_turn', 50, (0., 0., .25)),
    ('left_brake', 10, (0., 0., 0.)),
    ('post_turn_forward', 30, (.10, 0., 0.)),
    ('post_turn_brake', 10, (0., 0., 0.)),
    ('right_turn', 50, (0., 0., -.25)),
    ('right_brake', 10, (0., 0., 0.)),
)
TAIL_TICKS = 5


def schedule():
    return [dict(segment=segment, phase=i+1, requested_command=list(command))
            for i, (segment, count, command) in enumerate(SEGMENTS) for _ in range(count)]


def specification(trial):
    if trial not in TRIALS: raise ValueError('fixed fit or validation trial required')
    d = DEFINITIONS[trial]; spec = probe_spec(0)
    return spec | dict(scene_id='sustained-observed-floor-v1-'+trial,
        family='SUPERVISED_SUSTAINED_MOTION_DEVELOPMENT', trial=trial,
        procedural_seed=d['physics_seed'], appearance_arm='distinctive', appearance_seed=d['appearance_seed'],
        geometry=spec['geometry'] | dict(spawn_se2_world=list(d['spawn']), wall_boxes=[dict(
            wall_id='wide_front', centre_xyz=[4.5, 0., .3], size_xyz=[.08, 16., .6],
            yaw_rad=0., material_id='NEUTRAL_WALL')]))


def pack(spec):
    trial = spec.get('trial')
    if trial not in TRIALS or spec != specification(trial):
        raise ValueError('exact frozen sustained-motion specification required')
    base = predecessor_pack(); d = DEFINITIONS[trial]; x, y, yaw = d['spawn']
    wall = StaticObject(object_id='wide_front', kind='wall', center_xyz_m=(4.5, 0., .3),
        size_xyz_m=(.08, 16., .6), yaw_rad=0., material_id='NEUTRAL_WALL')
    return replace(base, scene_id=spec['scene_id'], family=spec['family'],
        manifest_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        physics_seed=d['physics_seed'], topology_seed=d['physics_seed'], visual_seed=d['appearance_seed'],
        world_bounds_xy_m=((-8., -8.), (8., 8.)), static_objects=(wall,),
        robot=replace(base.robot, spawn_xyz_m=(x, y, .375),
            spawn_quat_wxyz=(math.cos(yaw/2), 0., 0., math.sin(yaw/2))))


def validate_command(command):
    value = np.asarray(command, float)
    if (value.shape != (3,) or not np.isfinite(value).all() or not 0 <= value[0] <= .12
            or value[1] != 0. or abs(value[2]) > .25):
        raise ValueError('declared bounded forward/yaw collection command required')
    return value.tolist()
