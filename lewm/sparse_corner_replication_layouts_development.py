"""Three prospective development mazes for the frozen sparse-corner controller."""
from functools import lru_cache

from lewm.eligible_floor_registration_development import bind
from lewm import route_turn_memory_transfer_layouts_development as previous

generator = previous.generator
LAYOUT_COUNT = 3


def prior_graphs():
    rows = previous.prior_graphs()
    rows.extend(dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
        for s in previous.build_inventory()['layouts'])
    if len(rows) != 100:
        raise ValueError('explicit 100-layout development registry required')
    return rows


def make_spec(index, links, identity, candidate_index):
    spec = bind(generator.make_spec, PHYSICS_SEED_BASE=2026101200,
        APPEARANCE_SEED_BASE=2026101300)(index, links, identity, candidate_index)
    return spec | dict(scene_id=f'sparse-corner-replication-development-v1-{index:02d}',
        family='SPARSE_CORNER_REPLICATION_DEVELOPMENT')


@lru_cache(maxsize=1)
def build_inventory():
    return bind(generator.build_inventory, prior_graphs=prior_graphs,
        make_spec=make_spec, LAYOUT_COUNT=LAYOUT_COUNT, CONSTRUCTION_SEED=2026091761)() | dict(
            schema='sparse_corner_replication_layouts_development.v1',
            final_evaluation=False, selection_used_runtime_outcomes=False)


def specification(index):
    if type(index) is not int or not 0 <= index < LAYOUT_COUNT:
        raise ValueError('one of three fixed prospective development layouts required')
    return build_inventory()['layouts'][index]


public_mission = bind(generator.public_mission, specification=specification)
pack = bind(generator.pack, specification=specification)
