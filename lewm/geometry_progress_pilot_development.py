"""Prospective training-only geometry/progress pilot; no learned control claim."""
from dataclasses import replace
import hashlib
import json
import random
import numpy as np
import torch
from lewm.longer_motion_collection_development import specification as old_spec, pack as old_pack
from lewm.command_pulse_response_development import validate_command
from lewm.simulated_body_observation_development import validate_policy_packet
from lewm_genesis.scene_loader import StaticObject

WARMUP_TICKS = 3
HORIZON_TICKS = 40
GOAL_BODY_XY = (1.2, 0.)
MIN_PROGRESS_M = .15
GEOMETRIES = ('left_open', 'right_open')
APPEARANCES = (2026090911, 2026090912)
ACTIONS = ('hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn')


def assignments():
    cells = [(g, s, a) for g in GEOMETRIES for s in APPEARANCES for a in ACTIONS]
    random.Random(2026090913).shuffle(cells)
    return {f'episode_{i:03d}': dict(geometry=g, appearance_seed=s, action=a)
            for i, (g, s, a) in enumerate(cells)}


TRIALS = tuple(assignments())


def geometry(kind):
    if kind not in GEOMETRIES:
        raise ValueError('fixed mirrored obstruction required')
    sign = -1 if kind == 'left_open' else 1
    boxes = [dict(wall_id='front_partial_panel', centre_xyz=[.65, sign*.45, .35],
                  size_xyz=[.08, 1., .7], yaw_rad=0., material_id='NEUTRAL_WALL')]
    for name, center, size in (
        ('rear', [-1.5, 0., .7], [.08, 4., 1.4]),
        ('far', [2.5, 0., .7], [.08, 4., 1.4]),
        ('left', [.5, 2., .7], [4., .08, 1.4]),
        ('right', [.5, -2., .7], [4., .08, 1.4]),
    ):
        boxes.append(dict(wall_id=name, centre_xyz=center, size_xyz=size,
                          yaw_rad=0., material_id='NEUTRAL_WALL'))
    return dict(spawn_se2_world=[0., 0., 0.], wall_boxes=boxes)


def specification(trial):
    if trial not in TRIALS:
        raise ValueError('exact randomized pilot episode required')
    cell = assignments()[trial]
    # Opaque episode names are independent of command labels. No action is
    # supplied to the physical scene, camera, history encoder, or mesh builder.
    return old_spec('fit') | dict(scene_id='geometry-progress-v1-'+trial,
        family='GEOMETRY_PROGRESS_TRAINING_PILOT', trial=trial,
        layout_id='geometry-progress-v1-'+cell['geometry'], data_role='train',
        procedural_seed=2026090910, appearance_seed=cell['appearance_seed'],
        friction_mu=1., geometry=geometry(cell['geometry']))


def pack(spec):
    if spec != specification(spec['trial']):
        raise ValueError('exact frozen geometry-progress specification required')
    base = old_pack(old_spec('fit'))
    objects = tuple(StaticObject(object_id=b['wall_id'], kind='wall', center_xyz_m=tuple(b['centre_xyz']),
        size_xyz_m=tuple(b['size_xyz']), yaw_rad=b['yaw_rad'], material_id=b['material_id'])
        for b in spec['geometry']['wall_boxes'])
    return replace(base, scene_id=spec['scene_id'], family=spec['family'], static_objects=objects,
        physics_seed=spec['procedural_seed'], topology_seed=spec['procedural_seed'], visual_seed=spec['appearance_seed'],
        manifest_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        robot=replace(base.robot, spawn_xyz_m=(0., 0., .375), spawn_quat_wxyz=(1., 0., 0., 0.)),
        physics_randomization=replace(base.physics_randomization, floor_friction_mu=1.))


def candidate_commands(action):
    if action not in ACTIONS:
        raise ValueError('fixed six-action pilot bank required')
    zero = (0., 0., 0.)
    if action == 'hold': commands = [zero]*40
    elif action == 'forward': commands = [(.20, 0., 0.)]*30+[zero]*10
    elif action in ('left_arc', 'right_arc'):
        yaw = .45 if action == 'left_arc' else -.45
        commands = [(.16, 0., yaw)]*20+[(.16, 0., 0.)]*10+[zero]*10
    else:
        yaw = .45 if action == 'left_turn' else -.45
        commands = [(0., 0., yaw)]*30+[zero]*10
    return [validate_command(c) for c in commands]


def timed_candidate(action):
    """Exact four-second plans accepted by the existing JEPA interface."""
    values = torch.tensor(candidate_commands(action), dtype=torch.float32)/torch.tensor([.3, 1., .5])
    return values.reshape(8, 5, 3), torch.ones((8, 5), dtype=torch.bool)


def schedule(action):
    return [dict(phase=1, role='common_quiet_history', requested_command=[0., 0., 0.]) for _ in range(WARMUP_TICKS)]+[
        dict(phase=2 if i < 30 else 3, role='candidate_prefix' if i < 30 else 'candidate_brake', requested_command=c)
        for i, c in enumerate(candidate_commands(action))]


def decision(action, tick, policy):
    if type(tick) is not int or not 0 <= tick <= WARMUP_TICKS+HORIZON_TICKS:
        raise ValueError('bounded integer pilot decision tick required')
    validate_policy_packet(policy)
    now = 1_500_000_000+tick*100_000_000
    if (policy['sensor_state']['decision_ns'] != now or policy['image']['measured_ns'] != now
            or tuple(policy['sensor_state']['identity']) != (0, 0, 0)):
        raise ValueError('same episode actual decision clock required')
    rows = schedule(action)
    row = rows[tick] if tick < len(rows) else dict(phase=9, role='terminal', requested_command=[0., 0., 0.])
    return row | dict(tick=tick, decision_ns=now, terminal=tick == len(rows),
        tracker_required=False, native_state_used=False, navigation_qualified=False)


def progress_outcome(final_body_xy, *, complete, disallowed_contact, physical_stop, acquisition_stop):
    """Native evaluator displacement only; truncated horizons cannot succeed.

    This is local distance reduction, not passage traversal or maze arrival.
    Missing terminal displacement is retained instead of filled from commands.
    """
    flags = (complete, disallowed_contact)
    if any(type(f) is not bool for f in flags):
        raise ValueError('explicit boolean completion/contact required')
    progress = None
    if final_body_xy is not None:
        xy = np.asarray(final_body_xy, float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('finite measured planar displacement required')
        progress = float(np.linalg.norm(GOAL_BODY_XY)-np.linalg.norm(np.asarray(GOAL_BODY_XY)-xy))
    if complete and (progress is None or physical_stop is not None or acquisition_stop is not None):
        raise ValueError('complete horizon requires measured endpoint and no stop')
    success = bool(complete and not disallowed_contact and progress >= MIN_PROGRESS_M)
    return dict(complete_horizon=complete, contact=disallowed_contact, progress_m=progress,
        progress_threshold_m=MIN_PROGRESS_M, successful_progress=success,
        physical_stop=physical_stop, acquisition_stop=acquisition_stop,
        evidence_role='NATIVE_TRAINING_PILOT_LOCAL_PROGRESS_ONLY')


def panel_informativeness(rows):
    """All 24 episodes required; no winning subset or success-dependent retries."""
    expected = {(g, s, a) for g in GEOMETRIES for s in APPEARANCES for a in ACTIONS}
    actual = [(r['geometry'], r['appearance_seed'], r['action']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('complete unique balanced pilot cohort required')
    by = {key: r['outcome']['successful_progress'] for key, r in zip(actual, rows)}
    reversals = []
    for s in APPEARANCES:
        reversals.append(dict(appearance_seed=s, mirrored_arc_success_reversal=bool(
            by['left_open', s, 'left_arc'] and not by['left_open', s, 'right_arc']
            and by['right_open', s, 'right_arc'] and not by['right_open', s, 'left_arc'])))
    controls_fail = not any(by[g, s, a] for g in GEOMETRIES for s in APPEARANCES
                            for a in ('hold', 'left_turn', 'right_turn'))
    constant = [a for a in ACTIONS if all(by[g, s, a] for g in GEOMETRIES for s in APPEARANCES)]
    return dict(mirrored_reversals=reversals, nonprogress_controls_fail=controls_fail,
        constant_success_actions=constant, informative_for_next_dataset=bool(
            all(r['mirrored_arc_success_reversal'] for r in reversals) and controls_fail and not constant),
        independent_maze_layouts=0, model_trained=False, navigation_qualified=False)
