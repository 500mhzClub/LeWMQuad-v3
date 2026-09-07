"""Prospective long-motion observation challenge, never a navigation policy.

Scene construction is separate from the sensor-only command selector. New
labels/seeds alone do not establish new geometry or independent observations.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math

from lewm.longer_motion_collection_development import specification as old_spec, pack as old_pack
from lewm.simulated_body_observation_development import validate_policy_packet
from lewm_genesis.scene_loader import StaticObject

SCENES = ('offset_niche', 'unequal_baffles')
SUPPORTS = ('nominal', 'lower_friction')
DIRECTIONS = ('left', 'right')
TRIALS = tuple(f'{scene}_{support}_{direction}' for scene in SCENES
               for support in SUPPORTS for direction in DIRECTIONS)
DT_NS = 100_000_000
SETTLE_NS = 1_500_000_000
PHYSICS_PER_TICK = 50
SETTLE_SAMPLES = 750
# Requested yaw integrals are +/-3.15rad, not measured half-turns.
SEGMENTS = (
    ('initial_hold', 20, (0., 0., 0.)),
    ('approach', 40, (.12, 0., 0.)),
    ('approach_brake', 20, (0., 0., 0.)),
    ('turn_out', 126, (0., 0., .25)),
    ('turn_out_brake', 20, (0., 0., 0.)),
    ('translated_view', 30, (.12, 0., 0.)),
    ('translation_brake', 20, (0., 0., 0.)),
    ('turn_back', 126, (0., 0., -.25)),
    ('final_hold', 40, (0., 0., 0.)),
)
MAX_TICKS = sum(count for _, count, _ in SEGMENTS)
MAX_FRAMES = MAX_TICKS + 1
MAX_PHYSICS_SAMPLES = SETTLE_SAMPLES + PHYSICS_PER_TICK * MAX_TICKS


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _wall(name, x, y, sx, sy):
    return dict(wall_id=name, centre_xyz=[x, y, .7], size_xyz=[sx, sy, 1.4],
                yaw_rad=0., material_id='NEUTRAL_WALL')


def geometry(scene):
    if scene not in SCENES:
        raise ValueError('explicit independent tracking scene required')
    if scene == 'offset_niche':
        spawn = [-.83, -.47, .17]
        walls = [_wall('west', -2.8, 0., .08, 4.88), _wall('east', 2.8, 0., .08, 4.88),
                 _wall('south', 0., -2.4, 5.68, .08), _wall('north', 0., 2.4, 5.68, .08),
                 _wall('niche_back', 1.85, 1.35, 1.98, .08),
                 _wall('niche_side', .90, 1.90, .08, 1.18)]
    else:
        spawn = [-.61, .39, -.23]
        walls = [_wall('west', -3.1, 0., .08, 4.28), _wall('east', 3.1, 0., .08, 4.28),
                 _wall('south', 0., -2.1, 6.28, .08), _wall('north', 0., 2.1, 6.28, .08),
                 _wall('east_baffle', 2.1, .90, .08, 1.55),
                 _wall('west_baffle', -2.2, -.75, .08, .95),
                 _wall('north_baffle', -.30, 1.70, 1.15, .08)]
    return dict(spawn_se2_world=spawn, wall_boxes=walls)


def physical_geometry_identity(walls):
    """Ignore wall names, material labels, ordering, seeds and split labels.

    This is an exact metric-box identity, NOT graph-isomorphism rejection or
    proof of different visible pixels. Native geometry/prefix audits are needed.
    """
    rows = [dict(centre_xyz=[float(v) for v in w['centre_xyz']],
                 size_xyz=[float(v) for v in w['size_xyz']], yaw_rad=float(w['yaw_rad']))
            for w in walls]
    return hashlib.sha256(canonical(sorted(rows, key=canonical)).encode()).hexdigest()


def specification(trial):
    if trial not in TRIALS:
        raise ValueError('one of the eight fixed independent tracking trials required')
    scene, support, direction = next((s, c, d) for s in SCENES for c in SUPPORTS
                                    for d in DIRECTIONS if trial == f'{s}_{c}_{d}')
    scene_index = SCENES.index(scene)
    g = geometry(scene)
    return old_spec('fit') | dict(
        trial=trial, scene_id='independent-tracking-v1-' + trial,
        family='INDEPENDENT_LONG_MOTION_OBSERVATION_CHALLENGE', scene=scene,
        data_role='development_challenge', condition=support, direction=direction,
        procedural_seed=2026091201 + scene_index, appearance_seed=2026091203 + scene_index,
        appearance_arm='distinctive', friction_mu=1. if support == 'nominal' else .15,
        render_near_m=.005, public_depth_range_m=[.2, 5.], geometry=g,
        geometry_identity=physical_geometry_identity(g['wall_boxes']),
        hidden_robot_ideal_camera=True, hardware_calibrated=False,
        maximum_command_ticks=MAX_TICKS, maximum_rgbd_frames=MAX_FRAMES,
        maximum_physics_samples=MAX_PHYSICS_SAMPLES,
        navigation_qualified=False, model_training=False)


def validate_specification(spec):
    if not isinstance(spec, dict) or 'trial' not in spec or canonical(spec) != canonical(specification(spec['trial'])):
        raise ValueError('exact independent tracking specification required')


def pack(spec):
    validate_specification(spec)
    base = old_pack(old_spec('fit'))
    x, y, yaw = spec['geometry']['spawn_se2_world']
    objects = tuple(StaticObject(object_id=w['wall_id'], kind='wall',
        center_xyz_m=tuple(w['centre_xyz']), size_xyz_m=tuple(w['size_xyz']),
        yaw_rad=w['yaw_rad'], material_id=w['material_id']) for w in spec['geometry']['wall_boxes'])
    return replace(base, scene_id=spec['scene_id'], family=spec['family'], static_objects=objects,
        camera=replace(base.camera, near_m=.005), physics_seed=spec['procedural_seed'],
        topology_seed=spec['procedural_seed'], visual_seed=spec['appearance_seed'],
        manifest_sha256=hashlib.sha256(canonical(spec).encode()).hexdigest(),
        robot=replace(base.robot, spawn_xyz_m=(x, y, .375),
                      spawn_quat_wxyz=(math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))),
        physics_randomization=replace(base.physics_randomization, floor_friction_mu=spec['friction_mu']))


def schedule(direction):
    if direction not in DIRECTIONS:
        raise ValueError('fixed left/right challenge direction required')
    sign = 1 if direction == 'left' else -1
    return [dict(phase=phase, role=name, requested_command=[c[0], c[1], sign * c[2]])
            for phase, (name, count, c) in enumerate(SEGMENTS, 1) for _ in range(count)]


def decision(direction, tick, policy):
    """No native pose, geometry, friction, depth, tracker or goal input."""
    if type(tick) is not int or not 0 <= tick <= MAX_TICKS:
        raise ValueError('bounded integer command tick required')
    rows = schedule(direction)
    validate_policy_packet(policy)
    now = SETTLE_NS + tick * DT_NS
    if (policy['sensor_state']['decision_ns'] != now or policy['image']['measured_ns'] != now
            or tuple(policy['sensor_state']['identity']) != (0, 0, 0)):
        raise ValueError('same-episode exact acquisition clock required')
    row = deepcopy(rows[tick]) if tick < MAX_TICKS else dict(
        phase=10, role='terminal', requested_command=[0., 0., 0.])
    return row | dict(tick=tick, decision_ns=now, terminal=tick == MAX_TICKS,
        tracker_required=False, native_state_used=False, navigation_qualified=False)
