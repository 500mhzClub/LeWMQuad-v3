"""Four new local layouts/tasks fixed before any native outcome."""
from lewm import geometry_progress_layout_family_development as family
from lewm.eligible_floor_registration_development import bind

# Same local obstruction family, new parameter combinations, two opening sides.
# These are not complete mazes or four unrelated environment families.
TASKS = (
    ('fresh_00', 'left_open', .78, .98, -.07, .72, 15, 5),
    ('fresh_01', 'right_open', .75, 1.18, -.10, .90, 15, 5),
    ('fresh_02', 'left_open', .61, .88, -.03, .60, 10, 10),
    ('fresh_03', 'right_open', .82, 1.02, -.08, .78, 10, 10),
)


def specification(trial):
    row = next(r for r in TASKS if r[0] == trial)
    name, side, x, length, inner, height, _, _ = row
    geometry = family.pilot_geometry(side)
    sign = -1 if side == 'left_open' else 1
    geometry['wall_boxes'][0].update(centre_xyz=[x, sign*(length/2+inner), height/2],
        size_xyz=[.08, length, height])
    base = family.specification(family.TRIALS[0])
    return base | dict(scene_id='fresh-local-visual-goal-'+name, trial=name,
        family='FRESH_LOCAL_VISUAL_GOAL_DEVELOPMENT', layout_id=name,
        data_role='fresh_development_transfer', appearance_seed=2026090940,
        geometry=geometry)


pack = bind(family.pack, specification=specification)


def goal_commands(trial):
    row = next(r for r in TASKS if r[0] == trial)
    sign = 1 if row[1] == 'left_open' else -1
    # Goal setup only: three quiet ticks, then a fixed twenty-tick trajectory.
    # First pair arcs then translates; second pair arcs then turns in place.
    tail = [.2, 0., 0.] if row[6] == 15 else [0., 0., sign*.45]
    return [[0., 0., 0.]]*3+[[.16, 0., sign*.45]]*row[6]+[tail]*row[7]
