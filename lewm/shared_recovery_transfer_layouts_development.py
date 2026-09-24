"""Two new development mazes for controllers sharing local recovery."""
from functools import lru_cache
from lewm.eligible_floor_registration_development import bind
from lewm import multiseed_navigation_layouts_development as previous

LAYOUT_COUNT = 2
CONSTRUCTION_SEED = 2026091555
generator = previous.generator


def prior_graphs():
    rows = previous.prior_graphs()
    rows.extend(dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
        for s in previous.build_inventory()['layouts'])
    if len(rows) != 78:
        raise ValueError('explicit 78-layout development registry required')
    return rows


def make_spec(index, links, identity, candidate_index):
    spec = bind(generator.make_spec, PHYSICS_SEED_BASE=2026098400,
        APPEARANCE_SEED_BASE=2026098500)(index, links, identity, candidate_index)
    return spec | dict(scene_id=f'shared-recovery-transfer-development-v1-{index:02d}',
        family='SHARED_RECOVERY_TRANSFER_DEVELOPMENT')


@lru_cache(maxsize=1)
def build_inventory():
    result = bind(generator.build_inventory, prior_graphs=prior_graphs, make_spec=make_spec,
        LAYOUT_COUNT=LAYOUT_COUNT, CONSTRUCTION_SEED=CONSTRUCTION_SEED)()
    return result | dict(schema='shared_recovery_transfer_layouts_development.v1',
        final_evaluation=False, selection_used_runtime_outcomes=False)


def specification(index):
    if type(index) is not int or not 0 <= index < LAYOUT_COUNT:
        raise ValueError('one of two fixed development mazes required')
    return build_inventory()['layouts'][index]


public_mission = bind(generator.public_mission, specification=specification)
pack = bind(generator.pack, specification=specification)
