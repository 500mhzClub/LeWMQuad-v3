"""Four prospective maze layouts, selected before dense-model navigation."""
from functools import lru_cache

from lewm.eligible_floor_registration_development import bind
from lewm import return_routing_memory_layouts_development as previous

generator=previous.generator
LAYOUT_COUNT=4


def prior_graphs():
    rows=previous.prior_graphs()
    rows.extend(dict(name=s['scene_id'],edges=s['evaluation_layout']['edges'])
                for s in previous.build_inventory()['layouts'])
    assert len(rows)==105
    return rows


def make_spec(index,links,identity,candidate_index):
    spec=bind(generator.make_spec,PHYSICS_SEED_BASE=2026101800,
              APPEARANCE_SEED_BASE=2026101900)(index,links,identity,candidate_index)
    return spec|dict(scene_id=f'dense-world-model-navigation-development-v1-{index:02d}',
                     family='DENSE_WORLD_MODEL_NAVIGATION_DEVELOPMENT')


@lru_cache(maxsize=1)
def build_inventory():
    return bind(generator.build_inventory,prior_graphs=prior_graphs,make_spec=make_spec,
                LAYOUT_COUNT=LAYOUT_COUNT,CONSTRUCTION_SEED=2026091803)()|dict(
        schema='dense_world_model_maze_layouts_development.v1',final_evaluation=False,
        selection_used_runtime_outcomes=False,selection_used_dense_predictor_results=False)


def specification(index):
    if type(index) is not int or not 0<=index<LAYOUT_COUNT:
        raise ValueError('one of four fixed prospective dense-model layouts required')
    return build_inventory()['layouts'][index]


public_mission=bind(generator.public_mission,specification=specification)
pack=bind(generator.pack,specification=specification)
