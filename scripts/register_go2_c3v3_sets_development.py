"""Register new mazes for the C3-v3 on-policy round (pre-declared 30 September 2026, commit ec2e34c9).

Roles, fixed before construction, disjoint from every existing set (development,
validation, sealed test, the eight audit layouts, all prior registries including the
maze-view training and transfer layouts, and the 16 c3v2_sets_v1 layouts):
  - accepted layouts 0-15:  onpolicy_fit, one episode each (C1 missions; training data);
  - accepted layouts 16-21: onpolicy_heldout, one episode each (C1 missions; acceptance P only);
  - accepted layouts 22-31: safety_check, one episode each (C3-v3 and C4-v3 closed-loop safety check).
The capability inventory (including sealed layouts) is reconstructed in memory only to
exclude its graphs. It is verified against the registry by file hash alone and is never
written, printed or returned.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil

from lewm import decision_headroom_json_v42_development as output
from lewm import independent_round_trip_layouts_development as generator
from lewm.eligible_floor_registration_development import bind
from scripts import generate_go2_navigation_capability_sets_development as capability

REPO = Path(__file__).resolve().parents[1]
CONSTRUCTION_SEED = 2026093001
LAYOUT_COUNT = 32
SEEDS = dict(physics_seed_base=2026113400, appearance_seed_base=2026113500, episode_seed_base=2026113600)
ROOT_NAME = 'c3v3_sets_v1'
PREVIOUS = 'c3v2_sets_v1'


def role(index):
    return 'onpolicy_fit' if index < 16 else 'onpolicy_heldout' if index < 22 else 'safety_check'


def capability_inventory(protocol):
    def make_spec(index, links, identity, candidate_index):
        spec = bind(generator.make_spec, PHYSICS_SEED_BASE=protocol['generator']['physics_seed_base'],
                    APPEARANCE_SEED_BASE=protocol['generator']['appearance_seed_base'])(index, links, identity, candidate_index)
        return spec | dict(scene_id=f'navigation-capability-v1-{index:02d}', family='NAVIGATION_CAPABILITY_SAME_MAZE_FAMILY',
                           data_role='dev_tune' if index < 10 else 'validation' if index < 30 else 'sealed_test')
    return bind(generator.build_inventory, prior_graphs=capability.prior_graphs, make_spec=make_spec,
                LAYOUT_COUNT=90, CONSTRUCTION_SEED=protocol['generator']['construction_seed'])()


def main():
    protocol = json.loads(capability.PROTOCOL.read_text())
    assert capability.digest(capability.PROTOCOL.read_bytes()) == capability.PROTOCOL_SHA
    base = Path(protocol['output_root'])
    registry = json.loads((base/'registry.json').read_text())
    root = base/ROOT_NAME
    if shutil.disk_usage(base).free < protocol['caps']['recovery_reserve_bytes']+64*1024**2:
        raise RuntimeError('filesystem reserve')
    root.mkdir(exist_ok=False)
    output.install(root)
    # 1. In-memory reconstruction of the 90 capability layouts, verified by hash only.
    inventory = capability_inventory(protocol)
    expected = {e['maze_id']: e['maze']['sha256'] for e in registry['entries']}
    verified = sum(hashlib.sha256((output.dumps(s, indent=2)+'\n').encode()).hexdigest() == expected[s['layout_index']]
                   for s in inventory['layouts'])
    if verified != 90:
        raise ValueError(f'capability reconstruction does not match the registry ({verified}/90); stop')
    excluded = capability.prior_graphs() + [dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
                                            for s in inventory['layouts']]
    del inventory
    previous = json.loads((base/PREVIOUS/'registry.json').read_text())
    excluded += [dict(name=json.loads(Path(e['maze']['path']).read_text())['scene_id'],
                      edges=json.loads(Path(e['maze']['path']).read_text())['evaluation_layout']['edges']) for e in previous['entries']]

    # 2. New construction with every prior graph excluded.
    def make_spec(index, links, identity, candidate_index):
        spec = bind(generator.make_spec, PHYSICS_SEED_BASE=SEEDS['physics_seed_base'],
                    APPEARANCE_SEED_BASE=SEEDS['appearance_seed_base'])(index, links, identity, candidate_index)
        return spec | dict(scene_id=f'navigation-c3v3-{index:02d}', family='NAVIGATION_CAPABILITY_SAME_MAZE_FAMILY',
                           data_role=role(index))
    new = bind(generator.build_inventory, prior_graphs=lambda: excluded, make_spec=make_spec,
               LAYOUT_COUNT=LAYOUT_COUNT, CONSTRUCTION_SEED=CONSTRUCTION_SEED)()
    episode_protocol = copy.deepcopy(protocol)
    episode_protocol['generator'].update(episode_seed_base=SEEDS['episode_seed_base'], physics_seed_base=SEEDS['physics_seed_base'])
    entries = []
    for spec in new['layouts']:
        index = spec['layout_index']
        folder = root/'sets'/spec['data_role']
        row = dict(maze_id=index, role=spec['data_role'],
                   maze=capability.write(folder/f'maze_{index:02d}.json', spec), episodes=[])
        packet = capability.episode(spec, 0, episode_protocol)
        row['episodes'].append(capability.write(folder/f'episode_{index:02d}_0.json', packet))
        row['shortest_outbound_m'] = packet['shortest_outbound_m']
        entries.append(row)
    result = dict(schema='navigation_c3v3_set_registry.v1', construction_seed=CONSTRUCTION_SEED, seeds=SEEDS,
        predeclaration='docs/go2_navigation_c3v3_onpolicy_round_predeclaration_2026-09-30.md',
        counts=dict(onpolicy_fit=16, onpolicy_heldout=6, safety_check=10), previous_registry_excluded=PREVIOUS,
        capability_reconstruction_hash_verified=verified, excluded_prior_graphs=len(excluded),
        candidates_examined=new['candidates_examined'], rejection_counts=new['rejection_counts'],
        disjoint_from_all_prior_sets=True, sealed_contents_written_or_displayed=False,
        physics_executed=False, rendering_performed=False, entries=entries)
    receipt = capability.write(root/'registry.json', result)
    print(json.dumps(dict(status='REGISTERED', registry=receipt, counts=result['counts'],
        candidates_examined=result['candidates_examined'], rejection_counts=result['rejection_counts'],
        shortest_outbound_m=[round(e['shortest_outbound_m'], 2) for e in entries])))


if __name__ == '__main__':
    main()
