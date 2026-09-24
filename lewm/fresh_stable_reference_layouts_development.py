"""Four prospective development mazes, disjoint from the explicit 56-layout registry.

Construction is fixed before native outcomes. Only instructed goal coordinates
leave the scene/evaluator boundary; graph structure is never a planner input.
"""
from functools import lru_cache

from lewm import independent_round_trip_layouts_development as previous
from lewm.eligible_floor_registration_development import bind

LAYOUT_COUNT = 4
CONSTRUCTION_SEED = 2026091407


def prior_graphs():
    rows = previous.prior_graphs()
    rows.extend(dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
        for s in previous.build_inventory()['layouts'])
    return rows


def make_spec(index, links, identity, candidate_index):
    spec = bind(previous.make_spec, PHYSICS_SEED_BASE=2026093400,
        APPEARANCE_SEED_BASE=2026093500)(index, links, identity, candidate_index)
    return spec | dict(scene_id=f'fresh-stable-reference-development-v1-{index:02d}',
        family='FRESH_STABLE_REFERENCE_DEVELOPMENT')


@lru_cache(maxsize=1)
def build_inventory():
    result = bind(previous.build_inventory, prior_graphs=prior_graphs,
        make_spec=make_spec, LAYOUT_COUNT=LAYOUT_COUNT,
        CONSTRUCTION_SEED=CONSTRUCTION_SEED)()
    return result | dict(schema='fresh_stable_reference_layouts_development.v1',
        frozen_controller_transfer=True, final_evaluation=False)


def specification(index):
    if type(index) is not int or not 0 <= index < LAYOUT_COUNT:
        raise ValueError('one of the four fixed fresh development layouts required')
    return build_inventory()['layouts'][index]


public_mission = bind(previous.public_mission, specification=specification)
pack = bind(previous.pack, specification=specification)
