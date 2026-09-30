"""Regenerate the consumed RGB frames of the C3-v3 round's on-policy C1 missions by verified replay.

Pre-declared 30 Sep 2026 (commit ec2e34c9), section 3. The missions retained frame hashes, not pixels.
This runs pass 1 of the capability video renderer, unchanged
(`render_go2_capability_v4_video_development.verify`), and saves nothing else:
- It re-simulates each mission from its seed with the recorded prediction-slot outputs.
- It asserts every consumed-packet hash, selected action, dispatch command, applied command,
  native trace value, published pose and mission row against the log.
- It keeps the consumed primary RGB as `ego_frames/`.
No simulator, controller or harness code changes. Validation and sealed sets are untouched.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from lewm import navigation_capability_active_wall_development as wall
from scripts import render_go2_capability_v4_video_development as renderer
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.run_go2_capability_completed_support_v4_gate_erratum_continuation_development import ENVIRONMENT

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
OUT = BASE/'c3v3_onpolicy_replays'
PATTERNS = ('c3v3_onpolicy_C1_m*_ep0_attempt001',)


def replay(assignment):
    source_root = BASE/'runs'/assignment
    config = json.loads((source_root/'config.json').read_text())
    assert config['harness_sha256'] == owner.sha(owner.FREEZE) and config['controller'] == 'C1'
    packet = json.loads((source_root/'episode.json').read_text())
    assert packet['role'] in ('onpolicy_fit', 'onpolicy_heldout')
    output.install(BASE)
    root = OUT/assignment
    root.mkdir(parents=True, exist_ok=False)
    budget = owner.Budget(BASE, json.loads(owner.PROTOCOL.read_text()))
    spec = json.loads((source_root/'specification.json').read_text())
    requests = json.loads((source_root/'requests.json').read_text())
    frames = json.loads((source_root/'native/in_memory_camera_observations.json').read_text())['frames']
    with np.load(source_root/'native/physics_trace.npz', allow_pickle=False) as a:
        trace = {k: a[k].copy() for k in a.files}
    started = time.monotonic()
    try:
        renderer.source.previous.warmup()
        renderer.source.previous.study.cohort.stable.floor.configure()
        renderer.source.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        fidelity = renderer.verify(source_root, root, budget, spec, packet, trace, requests, frames)
        owner.save(root/'replay_verification.json', fidelity | dict(wall_s=time.monotonic()-started, renderer_sha256=owner.sha(renderer.__file__),
                                                                    script_sha256=owner.sha(__file__)))
    except BaseException as exc:
        owner.save(root/'failure.json', dict(reason=repr(exc), traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        renderer.source.shutdown_genesis()


def child(assignment):
    with (OUT/f'{assignment}.log').open('x') as log:
        code = subprocess.run([sys.executable, '-m', 'scripts.replay_go2_c3v3_onpolicy_frames_development', '--run', assignment],
                              stdout=log, stderr=subprocess.STDOUT, env=os.environ | ENVIRONMENT).returncode
    return assignment, code


def main(workers):
    assert Path.cwd().resolve() == owner.REPO
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [d.name for pattern in PATTERNS for d in sorted((BASE/'runs').glob(pattern))
            if (d/'episode_evaluation.json').exists() and not (OUT/d.name).exists()]
    print(json.dumps(dict(replays=todo)), flush=True)
    with wall.job(BASE, 'C3-v3 round: on-policy frame replays'), ProcessPoolExecutor(workers) as pool:
        for assignment, code in pool.map(child, todo):
            print(json.dumps(dict(assignment=assignment, exit=code,
                                  verified=(OUT/assignment/'replay_verification.json').exists())), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    replay(a.run) if a.run else main(a.workers)
