"""Generate the rigorous-phase sealed test set, sealed_test_v2 (Andrew, 1 October 2026).

Andrew declassified the capability test set (layouts 30-89, now sets/prelim_test_v1) for the
preliminary run and asked for a fresh 60-maze sealed set:
- same generator (`independent_round_trip_layouts_development.build_inventory` with the
  capability `make_spec` and `episode` rules, two episodes per maze);
- same exclusions, extended to every maze built since: the capability prior graphs, all 90
  capability layouts (development, validation and prelim_test_v1), the fresh-check set
  (c3v2_sets_v1) and the C3-v3 round set (c3v3_sets_v1). The generator rejects any candidate
  whose abstract topology or grid embedding matches an excluded graph;
- register it, hash it and run structural checks only. No physics, rendering or model.

Custody: the construction seed and the three seed bases are drawn from os.urandom inside this
process, written only inside the sealed folder, and never printed, so the model-facing
account cannot regenerate the set. The sealed folder `<capability root>/sets/sealed_test_v2`
is protected by AGENTS.md (`sealed_*`) and stays untouched until the rigorous phase.
Outside it, only a public receipt is written: file hashes and sizes, counts, aggregate
rejection counts and structural-check booleans; no seed, geometry or episode content.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from lewm import decision_headroom_json_v42_development as output
from lewm import independent_round_trip_layouts_development as generator
from lewm.eligible_floor_registration_development import bind
from scripts import generate_go2_navigation_capability_sets_development as capability

NAME, COUNT, EPISODES = 'sealed_test_v2', 60, (0, 1)
PREVIOUS = ('c3v2_sets_v1', 'c3v3_sets_v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def excluded_graphs(base):
    graphs = list(capability.prior_graphs())
    registry = json.loads((base/'registry.json').read_text())
    assert len(registry['entries']) == 90
    for entry in registry['entries']:
        # Layouts 30-89 were declassified and moved to sets/prelim_test_v1 (AGENTS.md exception).
        path = Path(entry['maze']['path'].replace('/sets/sealed_test_v1/', '/sets/prelim_test_v1/'))
        assert sha(path) == entry['maze']['sha256'], path
        spec = json.loads(path.read_text())
        graphs.append(dict(name=spec['scene_id'], edges=spec['evaluation_layout']['edges']))
    for name in PREVIOUS:
        for entry in json.loads((base/name/'registry.json').read_text())['entries']:
            assert sha(entry['maze']['path']) == entry['maze']['sha256']
            spec = json.loads(Path(entry['maze']['path']).read_text())
            graphs.append(dict(name=spec['scene_id'], edges=spec['evaluation_layout']['edges']))
    return graphs


def main():
    protocol = json.loads(capability.PROTOCOL.read_text())
    assert capability.digest(capability.PROTOCOL.read_bytes()) == capability.PROTOCOL_SHA
    base = Path(protocol['output_root'])
    folder = base/'sets'/NAME
    receipt_path = base/f'{NAME}_receipt.json'
    if shutil.disk_usage(base).free < protocol['caps']['recovery_reserve_bytes']+64*1024**2:
        raise RuntimeError('filesystem reserve')
    output.install(base)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        excluded = excluded_graphs(base)
        excluded_identities = {generator.identities(g['edges'])['metric_sha256'] for g in excluded}
        seeds = {k: int.from_bytes(os.urandom(4), 'big') for k in
                 ('construction_seed', 'physics_seed_base', 'appearance_seed_base', 'episode_seed_base')}

        def make_spec(index, links, identity, candidate_index):
            spec = bind(generator.make_spec, PHYSICS_SEED_BASE=seeds['physics_seed_base'],
                        APPEARANCE_SEED_BASE=seeds['appearance_seed_base'])(index, links, identity, candidate_index)
            return spec | dict(scene_id=f'navigation-sealed-v2-{index:02d}', family='NAVIGATION_CAPABILITY_SAME_MAZE_FAMILY',
                               data_role=NAME)
        inventory = bind(generator.build_inventory, prior_graphs=lambda: excluded, make_spec=make_spec,
                         LAYOUT_COUNT=COUNT, CONSTRUCTION_SEED=seeds['construction_seed'])()
        episode_protocol = copy.deepcopy(protocol)
        episode_protocol['generator'].update(episode_seed_base=seeds['episode_seed_base'], physics_seed_base=seeds['physics_seed_base'])

        # Structural checks, in memory, before anything is written.
        layouts = inventory['layouts']
        identities = [generator.identities(s['evaluation_layout']['edges']) for s in layouts]
        packets = {(s['layout_index'], ep): capability.episode(s, ep, episode_protocol) for s in layouts for ep in EPISODES}
        checks = dict(
            layout_count=len(layouts) == COUNT,
            indices_contiguous=[s['layout_index'] for s in layouts] == list(range(COUNT)),
            unique_topologies=len({i['abstract_topology_code'] for i in identities}) == COUNT,
            unique_embeddings=len({i['metric_code'] for i in identities}) == COUNT,
            disjoint_from_excluded=not ({i['metric_sha256'] for i in identities} & excluded_identities),
            episodes_per_maze=all((s['layout_index'], ep) in packets for s in layouts for ep in EPISODES),
            episode_structural_rules=all(p['initial_wall_occlusion'] and p['endpoint_clearance_qualified'] for p in packets.values()),
            roles=all(s['data_role'] == NAME for s in layouts) and all(p['role'] == NAME for p in packets.values()),
            no_physics_or_rendering=not any(p['runtime_executed'] or p['rendering_performed'] for p in packets.values()))
        if not all(checks.values()):
            raise ValueError('structural check failed: '+json.dumps({k: v for k, v in checks.items() if not v}))

        entries = []
        for spec in layouts:
            index = spec['layout_index']
            row = dict(maze_id=index, maze=capability.write(folder/f'maze_{index:02d}.json', spec), episodes=[])
            for ep in EPISODES:
                row['episodes'].append(capability.write(folder/f'episode_{index:02d}_{ep}.json', packets[(index, ep)]))
            entries.append(row)
        sealed_registry = capability.write(folder/'registry.json', dict(
            schema='navigation_sealed_test_v2_registry.v1', seeds=seeds, protocol_sha256=capability.PROTOCOL_SHA,
            counts=dict(mazes=COUNT, episodes_per_maze=len(EPISODES)), excluded_graphs=len(excluded),
            excluded_sources=['capability prior graphs', 'capability registry (90)', *PREVIOUS],
            candidates_examined=inventory['candidates_examined'], rejection_counts=inventory['rejection_counts'],
            structural_checks=checks, entries=entries))
        capability.write(folder/'construction_evidence.json', dict(structural_rejections=inventory['structural_rejections']))
        written = sorted(p.name for p in folder.iterdir())
        readback = all(sha(r['path']) == r['sha256'] for e in entries for r in [e['maze'], *e['episodes']])
        receipt = dict(schema='navigation_sealed_test_v2_public_receipt.v1', status='REGISTERED',
            sealed_folder=str(folder), sealed_registry=sealed_registry,
            files=[dict(name=Path(r['path']).name, sha256=r['sha256'], bytes=r['bytes']) for e in entries for r in [e['maze'], *e['episodes']]],
            file_count=len(written), expected_file_count=COUNT*(1+len(EPISODES))+2, hash_readback=readback,
            counts=dict(mazes=COUNT, episodes=COUNT*len(EPISODES)), excluded_graphs=len(excluded),
            candidates_examined=inventory['candidates_examined'], rejection_counts=inventory['rejection_counts'],
            structural_checks=checks, seeds_disclosed=False, geometry_or_episode_content_disclosed=False,
            physics_executed=False, rendering_performed=False, model_loaded=False,
            custody='sealed until the rigorous phase; AGENTS.md sealed_* rule applies; model-facing account never opens the folder')
        public = capability.write(receipt_path, receipt)
        print(json.dumps(dict(status='REGISTERED', receipt=public, file_count=len(written), hash_readback=readback,
                              structural_checks_passed=all(checks.values()), excluded_graphs=len(excluded),
                              candidates_examined=inventory['candidates_examined'])))
    except BaseException as exc:
        emptied = folder.exists() and not any(folder.iterdir())
        if emptied:
            folder.rmdir()  # nothing was written; a new attempt draws new random seeds
        capability.write(base/f'{NAME}_generation_failure_{os.getpid()}.json',
                         dict(status='STOP', error=repr(exc)[:300], sealed_folder_removed_empty=emptied))
        raise


if __name__ == '__main__':
    main()
