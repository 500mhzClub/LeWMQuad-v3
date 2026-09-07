"""One-shot aligned-floor scene assay; no robot, policy, training or old rerun."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from lewm.causal_depth_observation_development import FOCAL, INTRINSICS
from lewm.physical_semantics import world_from_optical
from lewm_genesis.aligned_floor_development import add_aligned_floor, aligned_floor_identity
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.probe_go2_correlated_moment_sensitivity_development import PREDECESSOR, ROOT
from scripts.run_go2_single_sample_rgbd_observation_development_v1 import verify_native, NATIVE_ROOT
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = ROOT / '.generated/go2_aligned_floor_interface_development_v1_attempt_001'
SOURCES = ('lewm_genesis/lewm_genesis/aligned_floor_development.py',
           'lewm/tests/test_aligned_floor_development.py',
           'scripts/run_go2_aligned_floor_interface_development_v1.py',
           'scripts/audit_go2_aligned_floor_interface_development_v1.py',
           'docs/go2_aligned_floor_interface_development_v1_2026-09-05.md')
VIEWS = ((0., 0.), (0., -.15), (.4, -.15), (-.4, -.15))


def array(value):
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    return np.asarray(value)


def native_bindings():
    return verify_native() | {str(NATIVE_ROOT / p): digest(NATIVE_ROOT / p)
                             for p in ('options/morphs.py', 'engine/scene.py')}


def verify_native_bindings(bindings):
    if native_bindings() != bindings: raise ValueError('native implementation changed')


def collect():
    import genesis as gs
    initialize_genesis(backend='cpu', seed=2026090501, logging_level='warning')
    scene = None
    try:
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.002, gravity=(0., 0., -9.81)),
                         show_viewer=False, renderer=gs.renderers.Rasterizer())
        collision, appearance = add_aligned_floor(scene, gs)
        sphere = scene.add_entity(gs.morphs.Sphere(radius=.022, pos=(-1., 0., .10)))
        camera = scene.add_camera(res=(640, 480), pos=(0., 0., .35), lookat=(1., 0., .35),
                                  up=(0., 0., 1.), fov=math.degrees(2 * math.atan(240 / FOCAL)),
                                  near=.05, far=200., GUI=False)
        scene.build()
        identity = aligned_floor_identity(collision, appearance)
        write_json(OUTPUT / 'floor_identity.json', identity)
        write_json(OUTPUT / 'physical_identity.json', {'floor_geom_idx': collision.geoms[0].idx,
            'sphere_geom_idx': sphere.geoms[0].idx, 'sphere_radius_m': .022,
            'dt_s': scene.dt, 'scope': 'evaluation-only sphere contact; not Go2 sensor data'})
        positions = []; contacts = []
        for step in range(500):
            scene.step()
            positions.append(array(sphere.get_pos()).reshape(3).copy())
            contacts.append({'step': step + 1, **{k: array(v).tolist() for k, v in
                sphere.get_contacts(with_entity=collision).items()
                if k in ('geom_a', 'geom_b', 'position', 'force_a', 'force_b')}})
        np.savez_compressed(OUTPUT / 'sphere_trace.npz', position_world_m=np.asarray(positions),
                            step=np.arange(1, 501))
        write_json(OUTPUT / 'contacts.json', contacts)
        rows = []
        for i, (yaw, pitch) in enumerate(VIEWS):
            position = np.array([0., 0., .35])
            forward = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
            camera.set_pose(pos=position, lookat=position + forward, up=(0., 0., 1.))
            before = int(scene.t); pose = np.array(camera.transform, copy=True)
            rgb_result = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
            depth_result = camera.render(rgb=False, depth=True, segmentation=False, normal=False)
            if before != scene.t or not np.array_equal(pose, camera.transform):
                raise ValueError('intervening physics or camera motion')
            if (any(v is not None for v in rgb_result[1:]) or depth_result[0] is not None
                    or any(v is not None for v in depth_result[2:])):
                raise ValueError('separate requested modalities required')
            rgb, depth = np.asarray(rgb_result[0]), np.asarray(depth_result[1])
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or depth.shape != (480, 640) or depth.dtype != np.float32:
                raise ValueError('native RGBD format differs')
            Image.fromarray(rgb).save(OUTPUT / f'rgb_{i}.png')
            np.savez_compressed(OUTPUT / f'depth_{i}.npz', optical_depth_m=depth)
            rows.append({'view': i, 'yaw_pitch_rad': [yaw, pitch], 'step_before_after': [before, int(scene.t)],
                         'world_from_optical': world_from_optical(position, forward, [0., 0., 1.]).tolist(),
                         'native_world_from_opengl': pose.tolist(),
                         'native_intrinsics': np.asarray(camera.intrinsics).tolist(),
                         'near_far_m': [camera.near, camera.far], 'sampling': sampling_readback(camera),
                         'depth_sha256': hashlib.sha256(depth.tobytes()).hexdigest(),
                         'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
            print(json.dumps({'rendered_view': i, 'physics_step': int(scene.t)}), flush=True)
        if aligned_floor_identity(collision, appearance) != identity:
            raise ValueError('floor geometry identity drift')
        write_json(OUTPUT / 'cameras.json', rows)
    finally:
        try:
            if scene is not None: scene.destroy()
        finally: shutdown_genesis()


def main():
    if OUTPUT.exists(): raise ValueError('one fresh fixed assay only; no retry or resume')
    old = json.loads(PREDECESSOR.read_text())
    inherited = old['source_sha256'] | old['input_sha256'] | old['artifact_sha256']
    verify_bindings(inherited)
    if set(SOURCES) & set(old['source_sha256']): raise ValueError('must not replace frozen source')
    sources = {p: digest(ROOT / p) for p in SOURCES}
    verify_bindings(sources)
    native = native_bindings()
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.json', {'source_sha256': old['source_sha256'] | sources,
        'input_sha256': old['input_sha256'] | old['artifact_sha256'], 'native_sha256': native,
        'scope': 'fresh aligned-floor interface only; no Go2 mission, policy, training or hardware'})
    try:
        collect()
        verify_bindings(inherited | sources); verify_native_bindings(native)
        leaves = ['floor_identity.json', 'physical_identity.json', 'sphere_trace.npz', 'contacts.json', 'cameras.json']
        leaves += [f'{kind}_{i}.{ext}' for i in range(4) for kind, ext in (('rgb', 'png'), ('depth', 'npz'))]
        write_json(OUTPUT / 'result.json', {'status': 'ACQUISITION_COMPLETE_AUDIT_REQUIRED',
                   'artifact_sha256': {p: digest(OUTPUT / p) for p in leaves},
                   'navigation_qualified': False, 'contact_model_validated': False})
        print('ALIGNED_FLOOR_ACQUISITION_COMPLETE', flush=True)
    except Exception as error:
        if not (OUTPUT / 'result.json').exists():
            write_json(OUTPUT / 'result.json', {'status': 'TERMINAL_FAILURE', 'error': repr(error)})
        raise


if __name__ == '__main__': main()
