"""Unmarked re-render of the held-out stage-2 recordings, for the offline marker control (development; Andrew, 5 October 2026).

Andrew (evening): the unmarked control is run offline at the forecast level, in place of closed-loop unmarked missions.
The question is whether the refit models use the tint to anticipate slip (approach and entry bins) or only react to slip
visible in recent motion.

**Method.** Each held-out recording (role stage2_heldout, contributing, verified replay) is re-simulated from its seed with
the same friction field and placement but `marked=False`, and the logged requested command is applied at every tick
instead of running the controller. Rendering does not feed back into physics, so the trajectory is the original one; this
is asserted at every tick:
- the applied command equals the logged one;
- every native trace value equals the logged trace (exactly);
- each consumed frame's measured time and physical sample index equal the logged ones.
The depth-noise record (hashes of the noisy depth packets) is compared but not asserted: on held-out maze 27 it first
differs at frame 48, the first frame in which tinted quads are visible (0.01% of pixels), although the noise is seeded by
(seed, layout, frame, camera) only and the floor geometry is unchanged, so the packet hash evidently covers RGB-linked
content. The differences are counted. C3 and C4 consume RGB and commands only, and the physics is asserted exact.
Only the RGB differs by design, and only where tinted floor quads are visible. The consumed primary RGB is saved as
ego_frames/NNNN.png, as in the marked replays. Missions ended by the stall stop or by a final-frame failure are handled as
in scripts/replay_go2_stage2_recording_frames_development.py.

Outputs: `<capability root>/stage2_unmarked_rerenders/<assignment>/` (ego_frames, rerender_verification.json).

Usage: render_go2_stage2_unmarked_frames_development.py [--workers N] | --run ASSIGNMENT
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

from lewm import decision_headroom_json_v42_development as output
from scripts import replay_go2_stage2_recording_frames_development as marked
from scripts.build_go2_stage2_feature_cache_development import contributing

BASE, owner, renderer, source = marked.BASE, marked.owner, marked.renderer, marked.source
OUT = BASE/'stage2_unmarked_rerenders'
COHORT = 's2rec2'


def install_unmarked(pin):
    from lewm.dev_dynamics_patches_v3_development import patch_session
    d = pin['dynamics']
    assert d['patches_version'] == 'v3' and d['placement_version'] == 'v2' and d['marked']
    owner.make_session = patch_session(owner.make_session, d['placement'], d['mu'], False)


def rerender(assignment):
    source_root = BASE/'runs'/assignment
    pin = json.loads((source_root/'launch_pin.json').read_text())
    install_unmarked(pin)
    output.install(BASE)
    root = OUT/assignment
    root.mkdir(parents=True, exist_ok=False)
    ego = root/'ego_frames'
    ego.mkdir()
    spec = json.loads((source_root/'specification.json').read_text())
    requests = json.loads((source_root/'requests.json').read_text())
    frames = json.loads((source_root/'native/in_memory_camera_observations.json').read_text())['frames']
    with np.load(source_root/'native/physics_trace.npz', allow_pickle=False) as a:
        trace = {k: a[k].copy() for k in a.files}
    truncated = 'post_sample_index' not in requests[-1]
    final_frame = (not truncated and (source_root/'failure.json').exists() and len(frames) == (len(requests)+4)//5+1)
    started = time.monotonic()
    session = None
    counts = dict(frames=0, steps=0, rgb_differs=0, depth_record_differs=0)

    def consume(index):
        session.sensor_packets()
        camera = session.captured_pairs[-1]
        record, logged = camera['consumed_hash_record'], frames[index]
        for key in ('measured_ns', 'physical_sample_index'):
            if record[key] != logged[key]:
                raise ValueError(f'frame {index}: {key} differs')
        counts['rgb_differs'] += record['pixel_sha256'] != logged['pixel_sha256']
        counts['depth_record_differs'] += record['live_depth_noise'] != logged['live_depth_noise']
        Image.fromarray(np.asarray(camera['images'][0][0])).save(ego/f'{index:04d}.png')
        camera['images'] = []
        counts['frames'] += 1
    try:
        renderer.source.previous.warmup()
        renderer.source.previous.study.cohort.stable.floor.configure()
        renderer.source.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        clock = source.UntimedSimulationClock()
        (root/'native').mkdir()
        session = renderer.start_session(spec, root/'native', full_frames=True)
        for key, values in trace.items():
            np.testing.assert_array_equal(np.stack([r[key] for r in session.samples]), values[:len(session.samples)])
        session.physics_clock_callback = clock.advance
        for tick, expected in enumerate(requests):
            now = int(session.ctx.runner._sim_time_ns)
            clock.advance(now)
            if now != expected['simulator_ns']:
                raise ValueError(f'clock differs at tick {tick}')
            if tick % 5 == 0:
                consume(tick//5)
            if truncated and tick == len(requests)-1:
                break
            session.phase = 2
            applied = session.command_policy_step(expected['requested_command'])
            np.testing.assert_array_equal(applied, expected['applied_command'])
            end = expected['post_sample_index']
            for key, values in trace.items():
                np.testing.assert_array_equal(np.stack([r[key] for r in session.samples[-10:]]), values[end-9:end+1])
            counts['steps'] += 1
        if final_frame:
            clock.advance(int(session.ctx.runner._sim_time_ns))
            consume(len(frames)-1)
        if counts['frames'] != len(frames):
            raise ValueError('frame count differs')
        clock.close()
        owner.save(root/'rerender_verification.json', dict(
            passed=True, marked=False, source_marked=True, truncated_by_stall_stop=truncated, ended_by_final_frame_failure=final_frame,
            frames=counts['frames'], frames_with_different_rgb=counts['rgb_differs'],
            frames_with_different_depth_record=counts['depth_record_differs'], forced_steps=counts['steps'], exact_native_trace_values=True, wall_s=time.monotonic()-started,
            script_sha256=owner.sha(__file__)))
    except BaseException as exc:
        owner.save(root/'failure.json', dict(reason=repr(exc), traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        if session is not None:
            session.ctx.build.scene.destroy()
        renderer.source.shutdown_genesis()


def child(assignment):
    with (OUT/f'{assignment}.log').open('x') as log:
        code = subprocess.run([sys.executable, '-m', 'scripts.render_go2_stage2_unmarked_frames_development', '--run', assignment],
                              stdout=log, stderr=subprocess.STDOUT, env=os.environ | marked.ENVIRONMENT).returncode
    return assignment, code


def main(workers):
    assert marked.Path.cwd().resolve() == owner.REPO
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((BASE/'dev_cohorts'/COHORT/'config.json').read_text())
    todo = []
    for _arm, _set, _maze, _ep, assignment in config['plan']:
        run = BASE/'runs'/assignment
        verification = marked.OUT/assignment/'replay_verification.json'
        if (run/'planning.json').exists() and contributing(run) and verification.exists() \
                and json.loads((run/'episode.json').read_text())['role'] == 'stage2_heldout' and not (OUT/assignment).exists():
            todo.append(assignment)
    print(json.dumps(dict(rerenders=todo)), flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for assignment, code in pool.map(child, todo):
            print(json.dumps(dict(assignment=assignment, exit=code)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run')
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    rerender(a.run) if a.run else main(a.workers)
