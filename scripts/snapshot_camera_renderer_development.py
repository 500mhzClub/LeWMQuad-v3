"""Render captured configurations in a separate, non-stepping scene process.

Native configurations stay within the sensor adapter. Its consumer receives
only rendered pixels, camera transforms for recording, and acquisition times.
"""
from pathlib import Path
import time
import numpy as np


def initialize(spec, output):
    import cv2
    import torch
    from lewm_genesis.scene_builder import initialize_genesis
    from scripts.run_go2_stopping_projection_transfer_development import FreshCameraSession
    global _session
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    directory = Path(output); directory.mkdir()
    _session = FreshCameraSession(spec, directory, noise_layout_index=spec['layout_index'], noise_sigma_mm=2)
    # No settle, gait command, or physics step runs in the renderer process.
    _session._render_pair()


def ready():
    return dict(ready=True, physics_steps=0, identity=_session._static_identity())


def render(qpos, measured_ns):
    started = time.perf_counter_ns()
    before = _session.ctx.build.scene._t
    _session.ctx.build.robot.set_qpos(np.asarray(qpos), zero_velocity=True)
    _session.ctx.runner._sim_time_ns = measured_ns
    # No scene.step occurs here, so explicitly refresh visual geometry and
    # shadow bounds. The renderer otherwise reuses its unchanged scene tick.
    _session.ctx.build.camera._rasterizer.update_scene(force_render=True)
    images, transforms = _session._render_pair()
    if _session.ctx.build.scene._t != before:
        raise ValueError('snapshot renderer must never step physics')
    return dict(images=images, transforms=transforms, measured_ns=measured_ns,
        rendering_wall_ns=time.perf_counter_ns()-started)
