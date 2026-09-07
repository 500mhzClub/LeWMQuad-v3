"""One-shot paired visual-extent/native-conversion assay; no policy or physics."""
import hashlib
import json
import math

import numpy as np
from PIL import Image

from lewm.causal_depth_observation_development import FOCAL
from lewm_genesis.floor_extent_precision_development import (
    EXTENTS_M, VIEWS, camera_pose, add_extent_floor, read_extent_identity,
)
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.run_go2_aligned_floor_interface_development_v1 import (
    OUTPUT as PREVIOUS, ROOT, native_bindings, verify_native_bindings,
)
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = ROOT / '.generated/go2_floor_extent_precision_development_v1_attempt_001'
SOURCES = ('lewm_genesis/lewm_genesis/floor_extent_precision_development.py',
           'lewm/tests/test_floor_extent_precision_development.py',
           'scripts/run_go2_floor_extent_precision_development_v1.py',
           'scripts/audit_go2_floor_extent_precision_development_v1.py',
           'docs/go2_floor_extent_precision_development_v1_2026-09-05.md')


def raw_depth_buffer(camera):
    from OpenGL.GL import (glGetIntegerv, glBindFramebuffer, glReadPixels,
        GL_READ_FRAMEBUFFER_BINDING, GL_READ_FRAMEBUFFER, GL_DEPTH_COMPONENT, GL_FLOAT)
    rasterizer = camera._rasterizer
    target = rasterizer._camera_targets[camera.uid]
    context = rasterizer._renderer
    context.make_current()
    try:
        previous = int(glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING))
        try:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, target._main_fb)
            raw = glReadPixels(0, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT)
            data = np.frombuffer(raw, dtype=np.float32) if isinstance(raw, bytes) else np.asarray(raw)
            if data.dtype != np.float32 or data.size != 480 * 640:
                raise ValueError('native normalized depth buffer format')
            return data.reshape(480, 640)[::-1].copy()
        finally: glBindFramebuffer(GL_READ_FRAMEBUFFER, previous)
    finally: context.make_uncurrent()


def collect(extent, directory):
    import genesis as gs
    initialize_genesis(backend='cpu', seed=2026090502, logging_level='warning')
    scene = None
    try:
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.002), show_viewer=False,
                         renderer=gs.renderers.Rasterizer())
        collision, visual = add_extent_floor(scene, gs, extent)
        camera = scene.add_camera(res=(640, 480), pos=(0., 0., .35), lookat=(1., 0., .35),
             up=(0., 0., 1.), fov=math.degrees(2 * math.atan(240 / FOCAL)), near=.05, far=200., GUI=False)
        scene.build()
        identity = read_extent_identity(collision, visual, extent)
        write_json(directory / 'floor_identity.json', identity)
        rows = []
        for i, view in enumerate(VIEWS):
            pose = camera_pose(view)
            camera.set_pose(pos=pose[:3, 3], lookat=pose[:3, 3] + pose[:3, 2], up=-pose[:3, 1])
            before = int(scene.t); native_pose = np.array(camera.transform, copy=True)
            rgb_row = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
            d_row = camera.render(rgb=False, depth=True, segmentation=False, normal=False)
            sampling = sampling_readback(camera); raw = raw_depth_buffer(camera)
            if before != 0 or scene.t != 0 or not np.array_equal(native_pose, camera.transform):
                raise ValueError('no intervening physics or camera motion')
            if any(v is not None for v in rgb_row[1:]) or d_row[0] is not None or any(v is not None for v in d_row[2:]):
                raise ValueError('only separate requested RGB and depth')
            rgb, depth = np.asarray(rgb_row[0]), np.asarray(d_row[1])
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or depth.shape != (480, 640) or depth.dtype != np.float32:
                raise ValueError('native RGBD encoding')
            Image.fromarray(rgb).save(directory / f'rgb_{i}.png')
            np.savez_compressed(directory / f'depth_{i}.npz', optical_depth_m=depth, normalized_depth=raw)
            rows.append({'view': i, 'view_parameters': list(view), 'world_from_optical': pose.tolist(),
                'native_world_from_opengl': native_pose.tolist(), 'step_before_after': [before, int(scene.t)],
                'native_intrinsics': np.asarray(camera.intrinsics).tolist(), 'near_far_m': [camera.near, camera.far],
                'sampling': sampling, 'depth_sha256': hashlib.sha256(depth.tobytes()).hexdigest(),
                'raw_buffer_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
            print(json.dumps({'extent_m': extent, 'rendered_view': i, 'physics_step': int(scene.t)}), flush=True)
        if read_extent_identity(collision, visual, extent) != identity: raise ValueError('native floor drift')
        write_json(directory / 'cameras.json', rows)
    finally:
        try:
            if scene is not None: scene.destroy()
        finally: shutdown_genesis()


def main():
    if OUTPUT.exists(): raise ValueError('fixed fresh one-shot comparison only')
    identities = {str((PREVIOUS / p).relative_to(ROOT)): h for p, h in (
        ('launch.json', '3112197fe089b3fceacfa39385ab3fe12b4f527ba8a47831cd98996a648cabc9'),
        ('result.json', 'bc8d6ed7cc07e9ceb8ede789a4bd38359fed16dc85f0f9e31dd9d511a8f27e50'))}
    verify_bindings(identities)
    old = json.loads((PREVIOUS / 'launch.json').read_text())
    result = json.loads((PREVIOUS / 'result.json').read_text())
    inputs = old['input_sha256'] | identities | {str((PREVIOUS / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    if set(SOURCES) & set(old['source_sha256']): raise ValueError('new source identities required')
    sources = old['source_sha256'] | {p: digest(ROOT / p) for p in SOURCES}
    verify_bindings(sources | inputs); verify_native_bindings(old['native_sha256'])
    native = native_bindings()
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.json', {'source_sha256': sources, 'input_sha256': inputs, 'native_sha256': native,
        'extents_m': list(EXTENTS_M), 'views': [list(v) for v in VIEWS],
        'scope': 'new paired render extent/conversion diagnostic; no physical contact, navigation or training'})
    try:
        leaves = []
        for extent in EXTENTS_M:
            name = f'extent_{int(extent)}'; directory = OUTPUT / name; directory.mkdir()
            collect(extent, directory)
            leaves += [f'{name}/floor_identity.json', f'{name}/cameras.json']
            leaves += [f'{name}/{kind}_{i}.{ext}' for i in range(8) for kind, ext in (('rgb', 'png'), ('depth', 'npz'))]
        verify_bindings(sources | inputs); verify_native_bindings(native)
        write_json(OUTPUT / 'result.json', {'status': 'ACQUISITION_COMPLETE_AUDIT_REQUIRED',
            'artifact_sha256': {p: digest(OUTPUT / p) for p in leaves}, 'navigation_qualified': False})
        print('FLOOR_EXTENT_PRECISION_ACQUISITION_COMPLETE', flush=True)
    except Exception as error:
        if not (OUTPUT / 'result.json').exists():
            write_json(OUTPUT / 'result.json', {'status': 'TERMINAL_FAILURE', 'error': repr(error)})
        raise


if __name__ == '__main__': main()
