"""Two fixed new development mazes for the training-seed comparison."""
from functools import lru_cache
from lewm.eligible_floor_registration_development import bind
from lewm import stopping_projection_transfer_layouts_development as previous

LAYOUT_COUNT=2
CONSTRUCTION_SEED=2026091554
generator=previous.generator


def prior_graphs():
    rows=previous.prior_graphs()
    rows.extend(dict(name=s['scene_id'],edges=s['evaluation_layout']['edges'])
        for s in previous.build_inventory()['layouts'])
    if len(rows)!=76:raise ValueError('explicit 76-layout development registry required')
    return rows


def make_spec(index,links,identity,candidate_index):
    spec=bind(generator.make_spec,PHYSICS_SEED_BASE=2026098200,
        APPEARANCE_SEED_BASE=2026098300)(index,links,identity,candidate_index)
    return spec|dict(scene_id=f'multiseed-navigation-development-v1-{index:02d}',
        family='MULTISEED_NAVIGATION_DEVELOPMENT')


@lru_cache(maxsize=1)
def build_inventory():
    result=bind(generator.build_inventory,prior_graphs=prior_graphs,make_spec=make_spec,
        LAYOUT_COUNT=LAYOUT_COUNT,CONSTRUCTION_SEED=CONSTRUCTION_SEED)()
    return result|dict(schema='multiseed_navigation_layouts_development.v1',
        final_evaluation=False,selection_used_runtime_outcomes=False)


def specification(index):
    if type(index) is not int or not 0<=index<LAYOUT_COUNT:
        raise ValueError('one of two fixed development mazes required')
    return build_inventory()['layouts'][index]


public_mission=bind(generator.public_mission,specification=specification)
pack=bind(generator.pack,specification=specification)
