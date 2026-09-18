"""Four prospective mazes beyond the explicit 64-layout development registry."""
from functools import lru_cache

from lewm.eligible_floor_registration_development import bind
from lewm import post_repeatability_transfer_layouts_development as recent

LAYOUT_COUNT = 4
CONSTRUCTION_SEED = 2026091517
generator = recent.recent.previous


def prior_graphs():
    rows = recent.prior_graphs()
    rows.extend(dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
        for s in recent.build_inventory()['layouts'])
    if len(rows) != 64:
        raise ValueError('explicit sixty-four-layout development registry required')
    return rows


def make_spec(index, links, identity, candidate_index):
    spec = bind(generator.make_spec, PHYSICS_SEED_BASE=2026097600,
        APPEARANCE_SEED_BASE=2026097700)(index, links, identity, candidate_index)
    return spec | dict(scene_id=f'post-training-comparison-development-v1-{index:02d}',
        family='POST_TRAINING_COMPARISON_DEVELOPMENT')


@lru_cache(maxsize=1)
def build_inventory():
    result = bind(generator.build_inventory, prior_graphs=prior_graphs,
        make_spec=make_spec, LAYOUT_COUNT=LAYOUT_COUNT,
        CONSTRUCTION_SEED=CONSTRUCTION_SEED)()
    return result | dict(schema='post_training_comparison_layouts_development.v1',
        final_evaluation=False, selection_used_runtime_outcomes=False)


def specification(index):
    if type(index) is not int or not 0 <= index < LAYOUT_COUNT:
        raise ValueError('one of four fixed post-training development layouts required')
    return build_inventory()['layouts'][index]


public_mission = bind(generator.public_mission, specification=specification)
pack = bind(generator.pack, specification=specification)
