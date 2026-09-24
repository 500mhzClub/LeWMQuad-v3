"""Prospective evaluation-only maze views, separate from readout training."""
from dataclasses import replace
from functools import lru_cache
import math

from lewm import maze_view_training_layouts_development as training
from lewm.eligible_floor_registration_development import bind

generator = training.generator
LAYOUT_COUNT, CASE_COUNT = 2, 8


def prior_graphs():
    rows = training.prior_graphs()
    rows.extend(dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
                for s in training.build_inventory()['layouts'])
    assert len(rows) == 113
    return rows


def make_spec(index, links, identity, candidate_index):
    spec = bind(generator.make_spec, PHYSICS_SEED_BASE=2026102500,
                APPEARANCE_SEED_BASE=2026102600)(index, links, identity, candidate_index)
    return spec | dict(scene_id=f'maze-view-transfer-v1-{index:02d}',
        family='MAZE_VIEW_TRANSFER_DEVELOPMENT', data_role='development_transfer')


@lru_cache(maxsize=1)
def build_inventory():
    return bind(generator.build_inventory, prior_graphs=prior_graphs,
        make_spec=make_spec, LAYOUT_COUNT=LAYOUT_COUNT,
        CONSTRUCTION_SEED=2026092301)() | dict(
            schema='maze_view_transfer_layouts_development.v1',
            data_role='development_transfer', training_layouts_excluded=True,
            final_evaluation=False, selection_used_prediction_errors=False)


@lru_cache(maxsize=1)
def specifications():
    # Reuse the geometry-only cardinal-view selection, without reading outcomes.
    raw = bind(training.specifications.__wrapped__, build_inventory=build_inventory,
               CASE_COUNT=CASE_COUNT)()
    cases = []
    for case, original in enumerate(raw):
        spec = dict(original)
        spec['evaluation_maze_index'] = spec.pop('training_maze_index')
        spec['evaluation_context'] = spec.pop('training_context')
        spec.update(scene_id=f'maze-view-transfer-v1-case-{case:02d}',
                    procedural_seed=2026102700+case)
        cases.append(spec)
    return tuple(cases)


def specification(index):
    if type(index) is not int or not 0 <= index < CASE_COUNT:
        raise ValueError('one of eight fixed evaluation contexts required')
    return specifications()[index]


def pack(spec):
    definition = bind(generator.pack, specification=specification)(spec)
    x, y, yaw = spec['geometry']['spawn_se2_world']
    return replace(definition, robot=replace(definition.robot,
        spawn_xyz_m=(x, y, .375),
        spawn_quat_wxyz=(math.cos(yaw/2), 0., 0., math.sin(yaw/2))))
