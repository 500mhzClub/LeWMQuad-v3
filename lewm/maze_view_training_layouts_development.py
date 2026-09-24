"""Training-only native maze views, disjoint from the prospective cohort."""
from dataclasses import replace
from functools import lru_cache
import math

from lewm import dense_world_model_maze_layouts_development as prospective
from lewm.eligible_floor_registration_development import bind

generator = prospective.generator
LAYOUT_COUNT = 4
CASE_COUNT = 16
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def prior_graphs():
    rows = prospective.prior_graphs()
    rows.extend(dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
                for s in prospective.build_inventory()['layouts'])
    assert len(rows) == 109
    return rows


def make_spec(index, links, identity, candidate_index):
    spec = bind(generator.make_spec, PHYSICS_SEED_BASE=2026102200,
                APPEARANCE_SEED_BASE=2026102300)(index, links, identity, candidate_index)
    return spec | dict(scene_id=f'maze-view-training-v1-{index:02d}',
                       family='MAZE_VIEW_TRAINING_DEVELOPMENT', data_role='train')


@lru_cache(maxsize=1)
def build_inventory():
    return bind(generator.build_inventory, prior_graphs=prior_graphs,
                make_spec=make_spec, LAYOUT_COUNT=LAYOUT_COUNT,
                CONSTRUCTION_SEED=2026092204)() | dict(
        schema='maze_view_training_layouts_development.v1', data_role='train',
        prospective_cohort_excluded=True, selection_used_prediction_errors=False)


@lru_cache(maxsize=1)
def specifications():
    """One view per cardinal heading/layout, chosen only from maze geometry."""
    cases = []
    for layout, base in enumerate(build_inventory()['layouts']):
        adjacency = {tuple(c): [] for c in base['evaluation_layout']['cells']}
        for a, b in base['evaluation_layout']['edges']:
            adjacency[tuple(a)].append(tuple(b))
            adjacency[tuple(b)].append(tuple(a))
        used = set()
        for direction_index, (dx, dy) in enumerate(DIRECTIONS):
            preferred_degree = (1, 2, 3, 2)[direction_index]
            candidates = [c for c, neighbours in adjacency.items()
                          if (c[0]+dx, c[1]+dy) in neighbours]
            cell = min(candidates, key=lambda c: (
                c in used, abs(len(adjacency[c])-preferred_degree), c))
            used.add(cell)
            yaw = math.atan2(dy, dx)
            case = len(cases)
            cases.append(base | dict(
                scene_id=f'maze-view-training-v1-case-{case:02d}', layout_index=case,
                training_maze_index=layout, procedural_seed=2026102400+case,
                training_context=dict(cell=list(cell), heading_rad=yaw,
                    open_neighbour=[cell[0]+dx, cell[1]+dy], degree=len(adjacency[cell])),
                geometry=base['geometry'] | dict(spawn_se2_world=[
                    cell[0]*generator.PITCH_M, cell[1]*generator.PITCH_M, yaw])))
    assert len(cases) == CASE_COUNT
    return tuple(cases)


def specification(index):
    if type(index) is not int or not 0 <= index < CASE_COUNT:
        raise ValueError('one of sixteen fixed training contexts required')
    return specifications()[index]


def pack(spec):
    definition = bind(generator.pack, specification=specification)(spec)
    x, y, yaw = spec['geometry']['spawn_se2_world']
    return replace(definition, robot=replace(definition.robot,
        spawn_xyz_m=(x, y, .375),
        spawn_quat_wxyz=(math.cos(yaw/2), 0., 0., math.sin(yaw/2))))
