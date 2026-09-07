#!/usr/bin/env python3
"""Fresh native contact-packet diagnostic; no Go2, camera, model or dataset."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'lewm_genesis', ROOT / 'lewm_worlds'):
    sys.path.insert(0, str(path))

from lewm.safety.contact_attribution import attribute_contacts
from lewm.safety.contact_hazard_ontology_v1 import DISALLOWED_CONTACT_FORCE_FLOOR_N

SEED, DT, STEPS = 20260905, 0.002, 100
PUBLIC_FIELDS = ('geom_a', 'geom_b', 'link_a', 'link_b', 'force_a', 'force_b', 'position', 'valid_mask')


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def native_force_norm(packet, robot_links, index):
    """Float64 scalar reference from exactly one native contact/environment.

    Do not compute a float32 NumPy norm and compare its rounded output against
    a float64 adapter at a 1e-9 tolerance. math.hypot provides a separate scalar
    calculation while retaining the native vector components exactly.
    """
    side = 'a' if int(packet['link_a'][0, index]) in robot_links else 'b'
    return math.hypot(*(float(value) for value in packet[f'force_{side}'][0, index]))


def collect(name, output):
    import genesis as gs
    from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
    from scripts.run_physical_graph_edge_handoff_qualification_v1 import _GenesisPhysicalSession

    initialize_genesis(backend='cpu', seed=SEED, logging_level='warning')
    scene = None
    raw_frames, records = {}, []
    native_axis_ok, force_norm_ok = True, True
    try:
        gravity = (0, 0, 0) if name == 'wall_impact' else (0, 0, -9.81)
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, gravity=gravity), show_viewer=False)
        ground = scene.add_entity(gs.morphs.Plane())
        wall = None
        if name != 'ground_support':
            wall_x = 0.3 if name == 'wall_impact' else 0.2
            wall = scene.add_entity(gs.morphs.Box(pos=(wall_x, 0, 0.5), size=(0.2, 1, 1), fixed=True))
        z = 0.5 if name == 'wall_impact' else 0.1
        robot = scene.add_entity(gs.morphs.Box(pos=(0, 0, z), size=(0.2, 0.2, 0.2)))
        scene.build(n_envs=1)
        robot_links = {int(link.idx) for link in robot.links}
        ground_links = {int(link.idx) for link in ground.links}
        wall_links = {int(link.idx) for link in wall.links} if wall is not None else set()
        names = {int(link.idx): str(link.name) for entity in (robot, ground, wall) if entity is not None for link in entity.links}
        objects = {**{index: 'ground_plane' for index in ground_links},
                   **{index: 'diagnostic_wall' for index in wall_links}}
        if name == 'wall_impact':
            robot.set_dofs_velocity([1, 0, 0, 0, 0, 0])
        for step in range(STEPS):
            scene.step()
            native = robot.get_contacts(exclude_self_contact=False)
            packet = {key: np.array(array(native[key]), copy=True) for key in PUBLIC_FIELDS if key in native}
            for key, value in packet.items():
                raw_frames[f'step_{step:03d}__{key}'] = value
            native_axis_ok &= (packet['link_a'].ndim == 2 and packet['link_a'].shape[0] == 1
                               and packet['valid_mask'].shape == packet['link_a'].shape
                               and packet['valid_mask'].dtype == np.bool_)
            rows = attribute_contacts(packet, environment_index=0, robot_link_ids=robot_links,
                                       support_link_ids=robot_links, ground_link_ids=ground_links,
                                       link_names=names, environment_object_ids=objects)
            for row in rows:
                index = row['contact_index']
                magnitude = native_force_norm(packet, robot_links, index)
                force_norm_ok &= abs(magnitude - row['force_magnitude_n']) <= 1e-9
            stub = SimpleNamespace(
                ctx=SimpleNamespace(build=SimpleNamespace(robot=SimpleNamespace(get_contacts=lambda **_: packet)),
                                    runner=SimpleNamespace(_as_np=lambda value: value)),
                _contact_topology={'robot': robot_links, 'support': robot_links, 'ground': ground_links})
            old = _GenesisPhysicalSession._disallowed_contact(stub)
            records.append({'step': step + 1, 'time_s': (step + 1) * DT,
                            'legacy_disallowed': old, 'corrected_disallowed': any(row['disallowed'] for row in rows),
                            'contacts': rows})
        np.savez_compressed(output / f'{name}_native_packets.npz', **raw_frames)
        (output / f'{name}_attribution.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
        contacts = [row for frame in records for row in frame['contacts']]
        summary = {
            'steps': len(records), 'native_one_environment_axis_and_mask': bool(native_axis_ok),
            'per_pair_force_norm_agreement': bool(force_norm_ok),
            'measured_nonzero_support_contacts': sum(row['environment_link_id'] in ground_links and row['force_magnitude_n'] > DISALLOWED_CONTACT_FORCE_FLOOR_N for row in contacts),
            'measured_nonzero_wall_contacts': sum(row['environment_link_id'] in wall_links and row['force_magnitude_n'] > DISALLOWED_CONTACT_FORCE_FLOOR_N for row in contacts),
            'disallowed_contacts': sum(row['disallowed'] for row in contacts),
            'wall_nonzero_all_disallowed': all(row['disallowed'] for row in contacts if row['environment_link_id'] in wall_links and row['force_magnitude_n'] > DISALLOWED_CONTACT_FORCE_FLOOR_N),
            'old_new_disagreement_steps': [row['step'] for row in records if row['legacy_disallowed'] != row['corrected_disallowed']],
        }
        return summary
    except BaseException:
        # Preserve acquired diagnostic frames even if this fresh assay fails.
        if raw_frames and not (output / f'{name}_native_packets.npz').exists():
            np.savez_compressed(output / f'{name}_native_packets.npz', **raw_frames)
        raise
    finally:
        try:
            if scene is not None:
                scene.destroy()
        finally:
            shutdown_genesis()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.absolute()
    if any(part == 'sealed' or part == 'sealed_test.json' or part.startswith('sealed_') for part in output.parts):
        parser.error('protected output path forbidden')
    output.mkdir(exist_ok=False)
    sources = ('scripts/run_contact_packet_semantics_development_v1.py',
               'lewm/safety/contact_attribution.py', 'lewm/safety/contact_hazard_ontology_v1.py',
               'scripts/run_physical_graph_edge_handoff_qualification_v1.py',
               'lewm_genesis/lewm_genesis/scene_builder.py',
               'docs/go2_contact_measurement_integrity_2026-09-05.md')
    report = {'schema': 'contact_packet_semantics_development.v1',
              'scope': 'fresh primitive diagnostic; no Go2/model/dataset; historical material unchanged',
              'seed': SEED, 'physics_dt_s': DT, 'steps_per_case': STEPS, 'n_envs': 1,
              'versions': {name: importlib.metadata.version(name) for name in ('genesis-world', 'numpy', 'torch')},
              'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}}
    (output / 'launch.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        cases = {}
        for name in ('ground_support', 'wall_impact', 'mixed_touch'):
            print(f'collecting {name}', flush=True)
            cases[name] = collect(name, output)
            (output / f'{name}_summary.json').write_text(json.dumps(cases[name], indent=2) + '\n')
        checks = {
            'native_batched_axis_and_mask': all(row['native_one_environment_axis_and_mask'] for row in cases.values()),
            'native_per_pair_force_agreement': all(row['per_pair_force_norm_agreement'] for row in cases.values()),
            'nonzero_ground_support_observed': cases['ground_support']['measured_nonzero_support_contacts'] > 0,
            'ground_support_allowed': cases['ground_support']['disallowed_contacts'] == 0,
            'nonzero_wall_contact_observed': cases['wall_impact']['measured_nonzero_wall_contacts'] > 0,
            'nonzero_wall_contact_disallowed': cases['wall_impact']['wall_nonzero_all_disallowed'],
        }
        report.update(cases=cases, checks=checks, status='PASS' if all(checks.values()) else 'CHECK_FAILURE')
    except Exception as exc:
        traceback.print_exc()
        report.update(status='INFRASTRUCTURE_FAILURE', error=f'{type(exc).__name__}: {exc}')
    report['artifact_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ('ground_support', 'wall_impact', 'mixed_touch')
        for path in (output / f'{name}_native_packets.npz', output / f'{name}_attribution.json', output / f'{name}_summary.json')
        if path.is_file()}
    (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key in ('status', 'checks', 'cases', 'error')}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
