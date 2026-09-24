"""Four new development layouts; scene structure remains evaluator-only."""
from functools import lru_cache

from lewm.eligible_floor_registration_development import bind
from lewm import fresh_stable_reference_layouts_development as recent


LAYOUT_COUNT = 4
CONSTRUCTION_SEED = 2026091417


def prior_graphs():
    rows = recent.prior_graphs()
    rows.extend(dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
        for s in recent.build_inventory()['layouts'])
    if len(rows) != 60:
        raise ValueError('explicit sixty-layout development registry required')
    return rows


def make_spec(index, links, identity, candidate_index):
    spec = bind(recent.previous.make_spec, PHYSICS_SEED_BASE=2026097400,
        APPEARANCE_SEED_BASE=2026097500)(index, links, identity, candidate_index)
    return spec | dict(scene_id=f'post-repeatability-transfer-development-v1-{index:02d}',
        family='POST_REPEATABILITY_TRANSFER_DEVELOPMENT')


@lru_cache(maxsize=1)
def build_inventory():
    result = bind(recent.previous.build_inventory, prior_graphs=prior_graphs,
        make_spec=make_spec, LAYOUT_COUNT=LAYOUT_COUNT,
        CONSTRUCTION_SEED=CONSTRUCTION_SEED)()
    return result | dict(schema='post_repeatability_transfer_layouts_development.v1',
        final_evaluation=False, selection_used_runtime_outcomes=False)


def specification(index):
    if type(index) is not int or not 0 <= index < LAYOUT_COUNT:
        raise ValueError('one of four fixed development transfer layouts required')
    return build_inventory()['layouts'][index]


public_mission = bind(recent.previous.public_mission, specification=specification)
pack = bind(recent.previous.pack, specification=specification)
