"""Fixed-pose pixel equality and Python heartbeat during paired camera reads."""
import hashlib
import argparse
import json
from pathlib import Path
import statistics
import threading
import time

import cv2
import numpy as np
import torch

from scripts import run_go2_persistent_visual_learning_comparison_development as previous
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from lewm_genesis.nogil_readback_development import readback_variants, renderer_variants, select_readbacks
from lewm.actuator_gain_development import configure_gains


def pixel_hashes(images):
    return [[hashlib.sha256(a.tobytes()).hexdigest() for a in pair] for pair in images]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward-pass', action='store_true')
    args = parser.parse_args()
    part = 'forward_pass' if args.forward_pass else 'readback'
    output = previous.study.BASE/f'go2_renderer_{part}_gil_fixed_pose_v1_attempt_001'
    output.mkdir(exist_ok=False)
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    previous.study.cohort.stable.floor.configure()
    spec = previous.layouts.specification(3)
    initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    session = None
    try:
        native = output/'native'; native.mkdir()
        session = previous.FreshCameraSession(spec, native, noise_layout_index=3, noise_sigma_mm=2)
        session.install_contact_identity()
        configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(),
            session.ctx.policy.env_cfg, 'checkpoint')
        session.settle_recorded()
        images, transforms = session._render_pair()
        reference = pixel_hashes(images)
        initial_physics = (int(session.ctx.runner._sim_time_ns), len(session.samples))
        jit, originals, replacements, treatment = (renderer_variants(session.ctx.build.camera, ('_forward_pass',))
            if args.forward_pass else readback_variants(session.ctx.build.camera))
        rows = []
        # A-B-A-B distinguishes the code treatment from a single warm-up drift.
        for mode in ('original', 'nogil', 'original', 'nogil'):
            select_readbacks(jit, originals if mode == 'original' else replacements)
            stamps = []; stopped = threading.Event(); ready = threading.Event()
            def heartbeat():
                stamps.append(time.perf_counter_ns()); ready.set()
                while not stopped.wait(.001):
                    stamps.append(time.perf_counter_ns())
                stamps.append(time.perf_counter_ns())
            thread = threading.Thread(target=heartbeat); thread.start(); ready.wait()
            elapsed = []
            try:
                for _ in range(10):
                    start = time.perf_counter_ns(); current, pose = session._render_pair()
                    elapsed.append((time.perf_counter_ns()-start)/1e6)
                    assert pixel_hashes(current) == reference and pose == transforms
                    assert (int(session.ctx.runner._sim_time_ns), len(session.samples)) == initial_physics
            finally:
                stopped.set(); thread.join()
            gaps = np.diff(stamps)/1e6
            row = dict(mode=mode, render_pair_median_ms=statistics.median(elapsed),
                heartbeat_gaps_ms=gaps.tolist(), heartbeat_max_ms=float(gaps.max()),
                heartbeat_p95_ms=float(np.quantile(gaps, .95)), pixels_identical=True,
                physics_unchanged=True)
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='heartbeat_gaps_ms'}), flush=True)
        sources = {}
        for name in (__file__, 'lewm_genesis/lewm_genesis/nogil_readback_development.py'):
            data=Path(name).read_bytes(); sources[name]=hashlib.sha256(data).hexdigest()
            (output/Path(name).name).write_bytes(data)
        result = dict(status='PASS', rows=rows, treatment=treatment, reference_pixel_sha256=reference,
            source_sha256=sources, navigation_tested=False, deployment_timing_qualified=False,
            physics_steps_during_comparison=0, original_readbacks_restored=True)
        select_readbacks(jit, originals)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n')
        raise
    finally:
        if session is not None:
            session.ctx.build.scene.destroy()
        shutdown_genesis()


if __name__ == '__main__':
    main()
