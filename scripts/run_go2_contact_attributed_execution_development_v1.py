#!/usr/bin/env python3
"""Eight fresh Go2 execution cases with per-contact evidence; development only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'lewm_genesis', ROOT / 'lewm_worlds'):
    sys.path.insert(0, str(path))

from lewm.physical_execution_development import KINDS, WIDTHS, build_case, evaluate_execution, rotation_xyzw
from lewm.physical_semantics import world_from_optical
from lewm.safety.contact_attribution import attribute_contacts
from scripts import run_physical_graph_edge_handoff_qualification_v1 as BASE

CONTACT_FIELDS = ('geom_a', 'geom_b', 'link_a', 'link_b', 'force_a', 'force_b', 'position', 'valid_mask')


class PhysicalStop(RuntimeError):
    pass


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class AttributedSession(BASE._GenesisPhysicalSession):
    """Reuse physics/gait/teacher source without invoking a frozen experiment."""

    def __init__(self, spec):
        self.phase = 0
        self.samples, self.packets, self.packet_times, self.contact_events = [], [], [], []
        super().__init__(spec, backend='cpu')

    def install_contact_identity(self):
        self.link_names = {int(link.idx): str(link.name) for entity in self.ctx.build.scene.entities for link in entity.links}
        self.object_ids = {}
        expected_walls = {row['wall_id'] for row in self.geometry['wall_boxes']}
        found = set()
        for entity in self.ctx.build.scene.entities:
            if entity is self.ctx.build.robot:
                continue
            if type(entity.morph).__name__ == 'Plane':
                name = 'ground_plane'
            else:
                name = str(entity.name)
                if name not in expected_walls:
                    raise ValueError(f'unresolved environment object identity: {name}')
                found.add(name)
            self.object_ids.update({int(link.idx): name for link in entity.links})
        if found != expected_walls:
            raise ValueError('wall/link identity coverage is incomplete')

    def _disallowed_contact(self):
        native = self.ctx.build.robot.get_contacts(exclude_self_contact=False)
        packet = {key: np.array(array(native[key]), copy=True) for key in CONTACT_FIELDS}
        if packet['link_a'].ndim != 2 or packet['link_a'].shape[0] != 1:
            raise ValueError('expected one explicit native environment axis')
        self.packets.append(packet)
        self.packet_times.append(self.sample_time)
        rows = attribute_contacts(packet, environment_index=0,
            robot_link_ids=self._contact_topology['robot'],
            support_link_ids=self._contact_topology['support'],
            ground_link_ids=self._contact_topology['ground'],
            link_names=self.link_names, environment_object_ids=self.object_ids)
        if any(row['force_status'] != 'measured' for row in rows):
            raise ValueError('Go2 contact force unavailable')
        disallowed = [row for row in rows if row['disallowed']]
        self.contact_events.append({'sample_index': len(self.samples), 'timestamp_s': self.sample_time,
                                    'phase': self.phase, 'disallowed_contacts': disallowed})
        return bool(disallowed)

    def _sample(self, requested, applied, timestamp_s):
        self.sample_time = float(timestamp_s)
        row = super()._sample(requested, applied, timestamp_s)
        row['phase'] = np.uint8(self.phase)
        self.samples.append(row)
        if not all(np.isfinite(np.asarray(value)).all() for value in row.values()):
            raise ValueError('nonfinite physical sample')
        rotation = rotation_xyzw(row['base_pose_world'][3:7])
        roll = np.arctan2(rotation[2,1], rotation[2,2])
        pitch = np.arcsin(np.clip(-rotation[2,0], -1, 1))
        if row['physics_contact']:
            raise PhysicalStop('DISALLOWED_CONTACT')
        if row['base_pose_world'][2] < .15 or max(abs(roll), abs(pitch)) > .70:
            raise PhysicalStop('BODY_STABILITY_LIMIT')
        return row

    def settle_recorded(self):
        BASE._reset_robot_to_fixed_spawn_compat(self.ctx.runner)
        for state in self.ctx.runner.episode_states:
            state.episode_step = 0
        self.ctx.episode_ticks = self.ctx.ticks_executed = self.ctx.policy_steps = 0
        self.ctx.episode_start_reset_count = int(self.ctx.runner.episode_states[0].reset_count)
        self.ctx.reset_in_last_block = False
        self.execute_requested_ticks([[0.,0.,0.]] * 15, record=True)

    def capture_fixed_rgb(self, output, name):
        from PIL import Image
        robot, camera = self.ctx.build.robot, self.ctx.build.camera
        position = array(robot.get_pos()).reshape(-1,3)[0]
        quat = array(robot.get_quat()).reshape(-1,4)[0]
        rotation = rotation_xyzw(quat[[1,2,3,0]])
        mount = self.ctx.pack.camera
        if not np.allclose(mount.rpy_body_rad, 0, rtol=0, atol=1e-12):
            raise ValueError('this study specifies the existing zero-RPY camera mount')
        camera_position = position + rotation @ np.asarray(mount.xyz_body_m)
        forward, up = rotation[:,0], rotation[:,2]
        camera.set_pose(pos=camera_position, lookat=camera_position+forward, up=up)
        rgb = self.ctx.runner._extract_rgb(camera.render())
        if rgb is None:
            raise ValueError('camera returned no image')
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = rgb[...,:3]
        if rgb.shape != (480,640,3) or rgb.dtype != np.uint8:
            raise ValueError(f'native RGB mismatch: {rgb.shape} {rgb.dtype}')
        Image.fromarray(rgb).save(output / f'{name}.png')
        return {'timestamp_s': float(self.samples[-1]['timestamp_s']),
                'world_from_optical': world_from_optical(camera_position, forward, up).tolist(),
                'rigid_mount_no_obstacle_adjustment': True,
                'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}

    def persist(self, output):
        arrays = {key: np.stack([row[key] for row in self.samples]) for key in self.samples[0]} if self.samples else {}
        np.savez_compressed(output / 'physics_trace.npz', **arrays)
        if self.packets:
            sizes = [packet['link_a'].shape[1] for packet in self.packets]
            native = {key: np.concatenate([packet[key][0] for packet in self.packets], axis=0) for key in CONTACT_FIELDS}
            native['frame_offsets'] = np.cumsum([0,*sizes], dtype=np.int64)
            native['frame_timestamp_s'] = np.asarray(self.packet_times)
            np.savez_compressed(output / 'native_contacts.npz', **native)
        (output / 'contact_events.json').write_text(json.dumps(self.contact_events, indent=2, allow_nan=False) + '\n')
        (output / 'contact_topology.json').write_text(json.dumps({
            'link_names': self.link_names, 'environment_object_ids': self.object_ids,
            'robot_link_ids': sorted(self._contact_topology['robot']),
            'support_link_ids': sorted(self._contact_topology['support']),
            'ground_link_ids': sorted(self._contact_topology['ground']),
            'native_environment_count': 1, 'selected_environment_index': 0,
            'packing': 'public native fields selected at environment 0, concatenated over frames with offsets',
        }, indent=2) + '\n')
        return arrays


def collect_case(spec, output):
    from lewm_genesis.scene_builder import shutdown_genesis
    session, stop_reason, images = None, None, {}
    try:
        session = AttributedSession(spec)
        session.install_contact_identity()
        try:
            session.settle_recorded()
            images['initial'] = session.capture_fixed_rgb(output, 'initial_rgb')
            session.phase = 1
            session.execute_teacher()
            session.phase = 2
            session.execute_requested_ticks([[0.,0.,0.]] * 5, record=True)
        except PhysicalStop as exc:
            stop_reason = str(exc)
        images['final'] = session.capture_fixed_rgb(output, 'final_rgb')
        arrays = session.persist(output)
        if len(session.packets) != len(session.samples):
            raise ValueError('native contact/physics sample alignment mismatch')
        active = arrays['phase'] != 0
        crossing = None
        if active.any():
            edge = spec['geometry']['selected_directed_edge']
            try:
                crossing = BASE.canonical_port_crossing(arrays['base_pose_world'][active],
                    np.zeros(np.count_nonzero(active), dtype=np.uint8), edge['opening_segment_world'],
                    edge['opening_normal_world'], [], sustained_samples=100)
            except BASE.ExperimentError as exc:
                if str(exc) not in ('teacher trace never crosses the canonical directed port',
                                    'teacher enters a competing physical port first',
                                    'teacher did not remain beyond the port for 100 physics samples'):
                    raise
        result = evaluate_execution(spec, arrays, stop_reason=stop_reason, crossing=crossing)
        result['images'] = images
        result['physics_samples'] = len(session.samples)
        result['first_disallowed_contact'] = next((row for row in session.contact_events if row['disallowed_contacts']), None)
        return result
    except Exception:
        if session is not None and not (output / 'physics_trace.npz').exists():
            session.persist(output)
        raise
    finally:
        try:
            if session is not None:
                session.ctx.build.scene.destroy()
        finally:
            shutdown_genesis()


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.absolute()
    if any(part == 'sealed' or part == 'sealed_test.json' or part.startswith('sealed_') for part in output.parts):
        parser.error('protected output path forbidden')
    output.mkdir(exist_ok=False)
    specs = [build_case(kind, width) for kind in KINDS for width in WIDTHS]
    source_paths = ('scripts/run_go2_contact_attributed_execution_development_v1.py',
        'lewm/physical_execution_development.py', 'lewm/physical_semantics.py',
        'lewm/safety/contact_attribution.py', 'lewm/safety/contact_hazard_ontology_v1.py',
        'scripts/run_physical_graph_edge_handoff_qualification_v1.py',
        'lewm_genesis/lewm_genesis/rollout.py', 'lewm_genesis/lewm_genesis/scene_builder.py',
        'lewm_genesis/lewm_genesis/scene_loader.py', 'config/go2_platform_manifest.yaml',
        'config/go2_primitive_registry.yaml', 'docs/go2_contact_attributed_execution_development_v1_2026-09-05.md')
    platform = yaml.safe_load((ROOT / 'config/go2_platform_manifest.yaml').read_text())
    policy = platform['locomotion']['policy_artifact']
    gait_bindings = {}
    for path_key, digest_key in (('path','sha256'),('cfg_path','cfg_sha256')):
        path = ROOT / policy[path_key]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != policy[digest_key]:
            raise ValueError('gait artifact does not match platform binding')
        gait_bindings[policy[path_key]] = digest
    launch = {'schema': 'go2_contact_attributed_execution_development.v1', 'case_specs': specs,
        'source_sha256': {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in source_paths},
        'gait_sha256': gait_bindings,
        'versions': {name: importlib.metadata.version(name) for name in ('genesis-world','torch','numpy')},
        'teacher': BASE.CONTRACT.TEACHER_CONTROLLER_AUTHORITY,
        'scope': 'fresh development oracle execution; not RGB/JEPA navigation or a resumed qualification stream'}
    (output / 'launch.json').write_text(json.dumps(launch, indent=2, allow_nan=False) + '\n')
    rows = []
    status = 'COMPLETE'
    try:
        for spec in specs:
            case_dir = output / spec['scene_id']
            case_dir.mkdir(exist_ok=False)
            print(f"collecting {spec['scene_id']}", flush=True)
            row = collect_case(spec, case_dir)
            row['scene_id'] = spec['scene_id']
            row['artifact_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (case_dir/'physics_trace.npz',case_dir/'native_contacts.npz',case_dir/'contact_events.json',
                             case_dir/'contact_topology.json',case_dir/'initial_rgb.png',case_dir/'final_rgb.png') if path.exists()}
            (case_dir / 'result.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
            rows.append(row)
            print(json.dumps({'scene_id': row['scene_id'], 'status': row['status'], 'stop_reason': row['stop_reason'],
                              'failed_checks': [key for key,value in row['checks'].items() if not value]}), flush=True)
    except Exception as exc:
        traceback.print_exc()
        status = 'INFRASTRUCTURE_FAILURE'
        (output/'failure.json').write_text(json.dumps({'error': f'{type(exc).__name__}: {exc}', 'completed_cases': len(rows)}, indent=2) + '\n')
    report = {'status': status, 'completed_cases': len(rows), 'planned_cases': len(specs),
              'successful_cases': sum(row['status']=='SUCCESS' for row in rows), 'cases': rows,
              'launch_sha256': hashlib.sha256((output/'launch.json').read_bytes()).hexdigest()}
    (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key:value for key,value in report.items() if key != 'cases'}, indent=2))
    return 0 if status == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
