"""Two prospective mazes for the return-leg routing-memory experiment."""
from functools import lru_cache

from lewm.eligible_floor_registration_development import bind
from lewm import sparse_corner_replication_layouts_development as previous

generator = previous.generator
LAYOUT_COUNT = 2


def prior_graphs():
    rows = previous.prior_graphs()
    rows.extend(dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
        for s in previous.build_inventory()['layouts'])
    assert len(rows) == 103
    return rows


def make_spec(index, links, identity, candidate_index):
    spec = bind(generator.make_spec, PHYSICS_SEED_BASE=2026101400,
        APPEARANCE_SEED_BASE=2026101500)(index, links, identity, candidate_index)
    return spec | dict(scene_id=f'return-routing-memory-development-v1-{index:02d}',
        family='RETURN_ROUTING_MEMORY_DEVELOPMENT')


@lru_cache(maxsize=1)
def build_inventory():
    return bind(generator.build_inventory, prior_graphs=prior_graphs,
        make_spec=make_spec, LAYOUT_COUNT=LAYOUT_COUNT, CONSTRUCTION_SEED=2026091781)() | dict(
            schema='return_routing_memory_layouts_development.v1',
            final_evaluation=False, selection_used_runtime_outcomes=False)


def specification(index):
    if type(index) is not int or not 0 <= index < LAYOUT_COUNT:
        raise ValueError('one of two prospective return-memory layouts required')
    return build_inventory()['layouts'][index]


public_mission = bind(generator.public_mission, specification=specification)
pack = bind(generator.pack, specification=specification)
