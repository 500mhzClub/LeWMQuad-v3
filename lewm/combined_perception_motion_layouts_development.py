"""Four fresh development mazes for the combined perception/motion comparison."""
from functools import lru_cache
from lewm.eligible_floor_registration_development import bind
from lewm import post_training_comparison_layouts_development as previous

LAYOUT_COUNT = 4
CONSTRUCTION_SEED = 2026091541
generator = previous.generator


def prior_graphs():
    rows = previous.prior_graphs()
    rows.extend(dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
        for s in previous.build_inventory()['layouts'])
    if len(rows) != 68: raise ValueError('explicit 68-layout development registry required')
    return rows


def make_spec(index, links, identity, candidate_index):
    spec = bind(generator.make_spec, PHYSICS_SEED_BASE=2026097800,
        APPEARANCE_SEED_BASE=2026097900)(index, links, identity, candidate_index)
    return spec | dict(scene_id=f'combined-perception-motion-development-v1-{index:02d}',
        family='COMBINED_PERCEPTION_MOTION_DEVELOPMENT')


@lru_cache(maxsize=1)
def build_inventory():
    result = bind(generator.build_inventory, prior_graphs=prior_graphs,
        make_spec=make_spec, LAYOUT_COUNT=LAYOUT_COUNT, CONSTRUCTION_SEED=CONSTRUCTION_SEED)()
    return result | dict(schema='combined_perception_motion_layouts_development.v1',
        final_evaluation=False, selection_used_runtime_outcomes=False)


def specification(index):
    if type(index) is not int or not 0 <= index < LAYOUT_COUNT:
        raise ValueError('one of four fixed fresh development mazes required')
    return build_inventory()['layouts'][index]


public_mission = bind(generator.public_mission, specification=specification)
pack = bind(generator.pack, specification=specification)
