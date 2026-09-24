"""Fixed fresh development geometry for the route-turn memory comparison."""
from functools import lru_cache

from lewm.eligible_floor_registration_development import bind
from lewm import recovery_repaired_navigation_replication_layouts_development as previous

generator = previous.generator
LAYOUT_COUNT = 2


def prior_graphs():
    rows = previous.prior_graphs()
    rows.extend(dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
        for s in previous.build_inventory()['layouts'])
    if len(rows) != 98:
        raise ValueError('explicit 98-layout development registry required')
    return rows


def make_spec(index, links, identity, candidate_index):
    spec = bind(generator.make_spec, PHYSICS_SEED_BASE=2026101000,
        APPEARANCE_SEED_BASE=2026101100)(index, links, identity, candidate_index)
    return spec | dict(scene_id=f'route-turn-memory-transfer-development-v1-{index:02d}',
        family='ROUTE_TURN_MEMORY_TRANSFER_DEVELOPMENT')


@lru_cache(maxsize=1)
def build_inventory():
    return bind(generator.build_inventory, prior_graphs=prior_graphs,
        make_spec=make_spec, LAYOUT_COUNT=LAYOUT_COUNT, CONSTRUCTION_SEED=2026091751)() | dict(
            schema='route_turn_memory_transfer_layouts_development.v1',
            final_evaluation=False, selection_used_runtime_outcomes=False)


def specification(index):
    if type(index) is not int or not 0 <= index < LAYOUT_COUNT:
        raise ValueError('one of two fixed fresh development layouts required')
    return build_inventory()['layouts'][index]


public_mission = bind(generator.public_mission, specification=specification)
pack = bind(generator.pack, specification=specification)
