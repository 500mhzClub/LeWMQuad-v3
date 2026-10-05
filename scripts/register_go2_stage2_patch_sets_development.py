"""Register the stage-2 patch layout sets, a selected family of routes with long straights (Andrew, 5 October 2026).

Same procedure as the C3-v3 and sealed_test_v2 registrations:
- the capability generator's candidate rule and acceptance checks (lewm/independent_round_trip_layouts_development.py:
  candidate, identities, evaluator_route, the route-structure rule, rejection of prior and repeated abstract
  topologies and grid embeddings);
- the capability make_spec and episode rules;
- structural checks only (no physics, rendering or model);
- hash-bound files and a registry.
Exclusions are the same as sealed_test_v2's: the capability prior graphs, all 90 capability layouts (dev, validation,
prelim_test_v1), c3v2_sets_v1 and c3v3_sets_v1.
sealed_test_v2 cannot be excluded. Its seeds are private and the folder is sealed, so overlap with it is NOT
checked here. That must be settled before any of these layouts trains a model.

Selection: every unique accepted layout within the generator's 10,000-candidate bound is built in order, with
episodes 0 and 1. A layout qualifies when its first such episode places patches covering at least 20% of the route
under the stage-2 v2 rule (lewm/dev_dynamics_patches_v2_development.py, seeded as at runtime). Qualifying layouts take
roles in generation order:
- stage2_fit: 24;
- stage2_heldout: 6;
- stage2_eval: 20.
These are global indices 0-49, one episode each (the qualifying one). Seeds are public (development sets).
Bounded construction can exhaust before 50 layouts qualify. As in the earlier procedures, a new attempt then uses new
seeds: construction seeds 2026100501, 2026100502, ... in fixed order, each with its own episode stream. The first that
yields 50 is used, and every attempt's count is recorded. The choice depends on structural counts only, never on
runtime outcomes.
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil

from lewm import decision_headroom_json_v42_development as output
from lewm import dev_dynamics_patches_v2_development as patches
from lewm import independent_round_trip_layouts_development as generator
from lewm.eligible_floor_registration_development import bind
from scripts import generate_go2_navigation_capability_sets_development as capability
from scripts.generate_go2_sealed_test_v2_development import excluded_graphs

ATTEMPTS = 10


def seeds_for(attempt):
    return dict(construction_seed=2026100501+attempt, physics_seed_base=2026114700+100*attempt,
                appearance_seed_base=2026114800+100*attempt, episode_seed_base=2026114900+100*attempt)
ROLES = (('stage2_fit', 24), ('stage2_heldout', 6), ('stage2_eval', 20))
EPISODES = (0, 1)
MIN_COVERAGE = .20
ROOT_NAME = 'stage2_sets_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accepted_layouts(excluded, SEEDS):
    """All unique layouts the generator's acceptance rules admit within its candidate bound, in order."""
    previous = [generator.identities(g['edges']) for g in excluded]
    old_top = {p['abstract_topology_code'] for p in previous}
    old_metric = {p['metric_code'] for p in previous}
    top, metric, rng = set(), set(), random.Random(SEEDS['construction_seed'])
    out, rejections = [], {}
    for number in range(generator.MAXIMUM_CANDIDATES):
        links = generator.candidate(rng)
        identity = generator.identities(links)
        route = generator.evaluator_route(links)
        directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(route, route[1:])]
        turns = sum(a != b for a, b in zip(directions, directions[1:]))
        reason = None
        if len(route) < 7 or turns < 2:
            reason = 'insufficient_declared_route_structure'
        elif identity['abstract_topology_code'] in old_top:
            reason = 'prior_abstract_topology'
        elif identity['metric_code'] in old_metric:
            reason = 'prior_grid_embedding'
        elif identity['abstract_topology_code'] in top:
            reason = 'repeated_abstract_topology'
        elif identity['metric_code'] in metric:
            reason = 'repeated_grid_embedding'
        if reason:
            rejections[reason] = rejections.get(reason, 0)+1
            continue
        top.add(identity['abstract_topology_code'])
        metric.add(identity['metric_code'])
        out.append((number, links, identity))
    return out, rejections


def select(protocol, excluded, SEEDS):
    candidates, rejections = accepted_layouts(excluded, SEEDS)
    episode_protocol = copy.deepcopy(protocol)
    episode_protocol['generator'].update(episode_seed_base=SEEDS['episode_seed_base'],
                                         physics_seed_base=SEEDS['physics_seed_base'])
    need = sum(n for _, n in ROLES)
    chosen, examined = [], 0
    for number, links, identity in candidates:
        if len(chosen) == need:
            break
        examined += 1
        index = len(chosen)
        role = next(name for name, end in zip([r for r, _ in ROLES], (24, 30, 50)) if index < end)
        spec = bind(generator.make_spec, PHYSICS_SEED_BASE=SEEDS['physics_seed_base'],
                    APPEARANCE_SEED_BASE=SEEDS['appearance_seed_base'])(index, links, identity, number)
        spec = spec | dict(scene_id=f'navigation-stage2-patch-{index:02d}', family='NAVIGATION_STAGE2_LONG_STRAIGHTS_SELECTED',
                           data_role=role)
        for episode in EPISODES:
            packet = capability.episode(spec, episode, episode_protocol)
            runtime_spec = spec | dict(geometry=spec['geometry'] | dict(spawn_se2_world=packet['home_se2_world']))
            placement = patches.place_patches(runtime_spec, packet, seed=patches.placement_seed(role, index, episode))
            if placement['coverage'] >= MIN_COVERAGE:
                chosen.append(dict(index=index, role=role, episode=episode, spec=spec, packet=packet,
                                   candidate_index=number, coverage=placement['coverage'], patches=len(placement['patches'])))
                break
    return candidates, rejections, chosen, examined


def main():
    protocol = json.loads(capability.PROTOCOL.read_text())
    assert capability.digest(capability.PROTOCOL.read_bytes()) == capability.PROTOCOL_SHA
    base = Path(protocol['output_root'])
    if shutil.disk_usage(base).free < protocol['caps']['recovery_reserve_bytes']+64*1024**2:
        raise RuntimeError('filesystem reserve')
    folders = {name: base/'sets'/name for name, _ in ROLES}
    if any(f.exists() for f in folders.values()) or (base/f'{ROOT_NAME}_registry.json').exists():
        raise SystemExit('stage-2 sets already registered')
    output.install(base)
    excluded = excluded_graphs(base)
    excluded_identities = {generator.identities(g['edges'])['metric_sha256'] for g in excluded}
    attempts = []
    for attempt in range(ATTEMPTS):
        SEEDS = seeds_for(attempt)
        candidates, rejections, chosen, examined = select(protocol, excluded, SEEDS)
        attempts.append(dict(seeds=SEEDS, unique_layouts_available=len(candidates), qualifying=len(chosen)))
        if len(chosen) == sum(n for _, n in ROLES):
            break
    else:
        raise SystemExit(f'no attempt reached 50 qualifying layouts: {attempts}')
    need = sum(n for _, n in ROLES)
    identities = [generator.identities(c['spec']['evaluation_layout']['edges']) for c in chosen]
    checks = dict(
        layout_count=len(chosen) == need,
        indices_contiguous=[c['index'] for c in chosen] == list(range(need)),
        unique_topologies=len({i['abstract_topology_code'] for i in identities}) == need,
        unique_embeddings=len({i['metric_code'] for i in identities}) == need,
        disjoint_from_excluded=not ({i['metric_sha256'] for i in identities} & excluded_identities),
        episode_structural_rules=all(c['packet']['initial_wall_occlusion'] and c['packet']['endpoint_clearance_qualified']
                                     for c in chosen),
        coverage_at_least_20_percent=all(c['coverage'] >= MIN_COVERAGE for c in chosen),
        no_physics_or_rendering=not any(c['packet']['runtime_executed'] or c['packet']['rendering_performed'] for c in chosen))
    if not all(checks.values()):
        raise ValueError('structural check failed: '+json.dumps({k: v for k, v in checks.items() if not v}))
    entries = []
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=False)
    for c in chosen:
        folder = folders[c['role']]
        packet = c['packet'] | dict(role=c['role'])
        entries.append(dict(maze_id=c['index'], role=c['role'], episode=c['episode'], candidate_index=c['candidate_index'],
                            patch_coverage=c['coverage'], patch_count=c['patches'],
                            maze=capability.write(folder/f"maze_{c['index']:02d}.json", c['spec']),
                            episodes=[capability.write(folder/f"episode_{c['index']:02d}_{c['episode']}.json", packet)]))
    readback = all(sha(r['path']) == r['sha256'] for e in entries for r in [e['maze'], *e['episodes']])
    registry = dict(
        schema='navigation_stage2_patch_sets_registry.v1', status='REGISTERED', family='selected: routes with long straights',
        seeds=SEEDS, attempts=attempts, protocol_sha256=capability.PROTOCOL_SHA, roles={name: n for name, n in ROLES},
        selection_rule=f'first qualifying episode (0, then 1) with stage-2 v2 placement coverage >= {MIN_COVERAGE}',
        excluded_graphs=len(excluded),
        excluded_sources=['capability prior graphs', 'capability registry (90)', 'c3v2_sets_v1', 'c3v3_sets_v1'],
        not_checked='sealed_test_v2 (private seeds, sealed folder): overlap must be settled before any training use',
        unique_layouts_available=len(candidates), layouts_examined=examined, generator_rejections=rejections,
        structural_checks=checks, hash_readback=readback, entries=entries)
    public = capability.write(base/f'{ROOT_NAME}_registry.json', registry)
    print(json.dumps(dict(status='REGISTERED', registry=public, unique_layouts_available=len(candidates),
                          layouts_examined=examined, checks_passed=all(checks.values()), hash_readback=readback,
                          coverage=[round(c['coverage'], 3) for c in chosen])))


if __name__ == '__main__':
    main()
