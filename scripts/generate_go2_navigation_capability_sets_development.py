"""One preregistered geometry-only construction; sealed contents never displayed.

Only this new-generation owner constructs/checks the new test packets in memory.
No historical sealed files, model, camera or simulator is used.
"""
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
from lewm import independent_round_trip_layouts_development as generator
from lewm import maze_view_transfer_layouts_development as transfer
from lewm import decision_headroom_json_v42_development as output
from lewm.decision_headroom_reference_development import ReferenceGeometry
from lewm.eligible_floor_registration_development import bind

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / 'docs/go2_navigation_capability_preregistration_v1_2026-09-25.json'
PROTOCOL_SHA = 'b6a84db2f304282d9d7ec9327ac21420d6e8c3e262de20178c7f243791a887d6'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
    # Readback is performed by the common converter. Test bytes are hashed here
    # without exposing or returning their geometry or episode contents.
    return dict(path=str(path), sha256=digest(path.read_bytes()), bytes=path.stat().st_size)


def prior_graphs():
    previous = transfer.prior_graphs() + [dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
        for s in transfer.build_inventory()['layouts']]
    audit = json.loads((REPO / 'docs/go2_decision_headroom_v4_layouts_2026-09-23.json').read_text())
    previous += [dict(name=r['specification']['scene_id'], edges=r['specification']['evaluation_layout']['edges'])
                 for r in audit['layouts']]
    unique = {}
    for row in previous:
        key = generator.identities(row['edges'])['metric_sha256']
        unique.setdefault(key, row)
    return list(unique.values())


def walls_for_reference(spec):
    return [dict(center=np.asarray(w['centre_xyz'][:2]), size=np.asarray(w['size_xyz'][:2]), yaw=w['yaw_rad'])
            for w in spec['geometry']['wall_boxes']]


def occluded(start, end, walls):
    # Exact slab intersection for the existing axis-aligned wall family.
    start, end = np.asarray(start), np.asarray(end)
    for wall in walls:
        if wall['yaw'] != 0:
            raise ValueError('generator family must have axis-aligned walls')
        low = wall['center'] - wall['size']/2
        high = wall['center'] + wall['size']/2
        enter, leave = 0., 1.
        for k in (0, 1):
            delta = end[k] - start[k]
            if abs(delta) < 1e-14:
                if not low[k] <= start[k] <= high[k]:
                    enter, leave = 1., 0.
                    break
            else:
                a, b = sorted(((low[k]-start[k])/delta, (high[k]-start[k])/delta))
                enter, leave = max(enter, a), min(leave, b)
        if enter <= leave:
            return True
    return False


def episode(spec, episode_index, protocol):
    index = spec['layout_index']
    seed = protocol['generator']['episode_seed_base'] + 2*index + episode_index
    rng = random.Random(seed)
    cells = protocol['generator']['cells']
    pairs = [(a,b) for a in cells for b in cells if a != b]
    rng.shuffle(pairs)
    walls = walls_for_reference(spec)
    cfg = protocol['definitions']['spl']
    for a, b in pairs:
        start, target = np.asarray(a)*1.3, np.asarray(b)*1.3
        geo = ReferenceGeometry(walls, protocol['generator']['world_bounds_xy_m'], target,
            radius_m=cfg['inflation_radius_m'], clearance_m=cfg['additional_clearance_m'],
            resolution_m=cfg['grid_resolution_m'])
        if float(min(geo.footprint_clearance(start), geo.footprint_clearance(target))) + geo.radius < .5:
            continue
        if not occluded(start, target, walls):
            continue
        distance = geo.distance_and_heading(start)
        if not distance['valid'] or distance['distance_m'] <= protocol['episodes']['minimum_start_beacon_geodesic_m']:
            continue
        heading = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(heading), math.sin(heading)
        goal_body = np.array([[c,s],[-s,c]]) @ (target-start)
        return dict(schema='navigation_capability_episode.v1', maze_id=index, episode_index=episode_index,
            episode_id=f'{index:02d}/{episode_index}', role=spec['data_role'], seed=seed,
            simulation_seed=protocol['generator']['physics_seed_base']+2*index+episode_index,
            home_se2_world=[*start.tolist(), heading], beacon_xy_world=target.tolist(),
            mission=dict(goal_initial_body_xy_m=goal_body.tolist(), return_initial_body_xy_m=[0.,0.],
                         require_return_after_goal=True),
            shortest_outbound_m=distance['distance_m'], shortest_return_m=distance['distance_m'],
            initial_wall_occlusion=True, endpoint_clearance_qualified=True,
            runtime_executed=False, rendering_performed=False)
    raise ValueError(f'No pair satisfies fixed structural episode rules for maze {index}; no replacement allowed')


def main():
    if digest(PROTOCOL.read_bytes()) != PROTOCOL_SHA:
        raise ValueError('preregistration identity changed')
    protocol = json.loads(PROTOCOL.read_text())
    root = Path(protocol['output_root'])
    for disk, reserve in ((root.parent, protocol['caps']['recovery_reserve_bytes']),
                          (REPO, protocol['caps']['workspace_reserve_bytes'])):
        if shutil.disk_usage(disk).free < reserve + 32*1024**2:
            raise RuntimeError('filesystem reserve prevents bounded geometry output')
    root.mkdir(exist_ok=False)
    output.install(root)
    def make_spec(index, links, identity, candidate_index):
        spec = bind(generator.make_spec, PHYSICS_SEED_BASE=protocol['generator']['physics_seed_base'],
                    APPEARANCE_SEED_BASE=protocol['generator']['appearance_seed_base'])(index, links, identity, candidate_index)
        return spec | dict(scene_id=f'navigation-capability-v1-{index:02d}',
            family='NAVIGATION_CAPABILITY_SAME_MAZE_FAMILY',
            data_role='dev_tune' if index < 10 else 'validation' if index < 30 else 'sealed_test')
    try:
        inventory = bind(generator.build_inventory, prior_graphs=prior_graphs, make_spec=make_spec,
            LAYOUT_COUNT=90, CONSTRUCTION_SEED=protocol['generator']['construction_seed'])()
        public = []
        for spec in inventory['layouts']:
            role = spec['data_role']
            folder = root / 'sets' / ('sealed_test_v1' if role == 'sealed_test' else role)
            row = dict(maze_id=spec['layout_index'], role=role,
                maze=write(folder / f'maze_{spec["layout_index"]:02d}.json', spec), episodes=[])
            for ep in (0,1):
                packet = episode(spec, ep, protocol)
                row['episodes'].append(write(folder / f'episode_{spec["layout_index"]:02d}_{ep}.json', packet))
            public.append(row)
        result = dict(schema='navigation_capability_registry.v1', protocol_sha256=PROTOCOL_SHA,
            counts=dict(dev_tune=10, validation=20, sealed_test=60), episodes_per_maze=2,
            candidates_examined=inventory['candidates_examined'], rejection_counts=inventory['rejection_counts'],
            prior_registry_count=inventory['prior_source_layout_count'],
            disjoint_from_prior_and_audit=True, unique_topology_and_embedding=True,
            structural_checks_passed=True, physics_executed=False, rendering_performed=False, entries=public)
        receipt = write(root / 'registry.json', result)
        # Rejections include candidate geometry identities: keep these sealed as
        # construction evidence, never include them in ordinary source discovery.
        write(root / 'sets/sealed_test_v1/construction_evidence.json',
              dict(structural_rejections=inventory['structural_rejections']))
        print(json.dumps(dict(status='REGISTERED', registry=receipt, counts=result['counts'])))
    except BaseException as exc:
        write(root / 'generation_failure.json', dict(status='STOP', error=repr(exc), retry_authorized=False))
        raise


if __name__ == '__main__':
    main()
