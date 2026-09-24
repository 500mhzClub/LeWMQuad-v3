"""Collect bounded native training views; no prospective maze data or fitting."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import psutil

from lewm import maze_view_training_layouts_development as layouts
from lewm.eligible_floor_registration_development import bind
from scripts import collect_go2_full_heading_training_development as previous

OUTPUT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/'
              'navigation_development_artifacts_v1/go2_maze_view_training_v1_attempt_001')
TRIALS = tuple(f'maze_view_{i:02d}' for i in range(layouts.CASE_COUNT))
CPU_GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7))
save, digest = previous.fit.save, previous.fit.digest


class MazeTrainingPhysicalInit(previous.IndependentRoundTripPhysicalInit):
    def __init__(self, spec, *, backend):
        bind(previous.IndependentRoundTripPhysicalInit.__init__,
             specification=layouts.specification, pack=layouts.pack)(self, spec, backend=backend)
        self._runtime['high_level'] = 'fixed training excitation; no navigation policy'


class TrainingSession(previous.RGBNavigationRetentionMixin, previous.NogilDrawingMixin,
        previous.LiveDepthNoiseMixin, previous.CompactDepthRetentionMixin,
        previous.LzmaRawDepthPairedCameraSession, MazeTrainingPhysicalInit):
    pass


def schedule():
    blocks = [('hold', 10), ('left_turn', 10), ('right_turn', 20),
              ('left_turn', 10), ('hold', 10), ('forward', 20), ('hold', 10),
              ('left_arc', 10), ('right_arc', 10), ('hold', 10),
              ('right_turn', 10), ('left_turn', 20), ('right_turn', 10), ('hold', 10)]
    tape, phases = [], []
    for block, (action, count) in enumerate(blocks):
        tape.extend([previous.candidate_commands(action)[0]]*count)
        phases.extend([f'block_{block:02d}_{action}']*count)
    assert len(tape) == 170
    return tape, phases


def prepare():
    free = shutil.disk_usage(OUTPUT.parent).free
    assert free > 4*1024**3 and psutil.virtual_memory().available > 16*1024**3
    tape, phases = schedule()
    specs = list(layouts.specifications())
    # Check the pack's actual robot initialization, not only the metadata.
    for spec in specs:
        definition = layouts.pack(spec)
        np.testing.assert_allclose(definition.robot.spawn_xyz_m[:2],
                                   spec['geometry']['spawn_se2_world'][:2], atol=0, rtol=0)
        yaw = spec['geometry']['spawn_se2_world'][2]
        q = definition.robot.spawn_quat_wxyz
        assert abs(np.arctan2(2*q[0]*q[3], 1-2*q[3]**2)-yaw) < 1e-12
        assert spec['data_role'] == 'train'
    OUTPUT.mkdir(exist_ok=False)
    save(OUTPUT/'plan.json', dict(trials=TRIALS, specifications=specs,
        inventory=layouts.build_inventory(), tape=tape, phases=phases,
        source_sha256=digest(__file__), dependencies={str(Path(p).resolve()): digest(p)
            for p in (previous.__file__, layouts.__file__)},
        data_role='train', workers=2, cpu_groups=CPU_GROUPS,
        free_bytes=free, minimum_free_bytes=2*1024**3,
        estimated_collection_bytes=2*1024**3, depth_arrays_retained=False,
        render_robot=True, fixed_tape_training_not_navigation=True,
        prospective_cohort_excluded=True, no_retry_or_outcome_based_selection=True,
        sample_rule='All contact-free 100--800-ms windows with departure >=10 and all eight horizons present.',
        planned_learning='Matched original-data versus half-original/half-new readout fit; same architecture, initialization, normalization and updates; encoder/predictor frozen.',
        limitation='Maze geometry, appearance and motion/view coverage change together.'))
    print('MAZE_VIEW_PREPARED', len(specs), len(tape)+1, flush=True)


def run(case):
    plan = json.loads((OUTPUT/'plan.json').read_text())
    for path, identity in plan['dependencies'].items():
        assert digest(path) == identity
    # Reuse the completed native collector, including contacts and failure persistence.
    bind(previous.run, OUTPUT=OUTPUT, TRIALS=TRIALS,
         specification=layouts.specification, TrainingSession=TrainingSession,
         __file__=__file__)(case)


def worker(index):
    for case in range(index, layouts.CASE_COUNT, 2):
        with (OUTPUT/f'case_{case:02d}_process.log').open('x') as log:
            subprocess.run([sys.executable, __file__, '--case', str(case)],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        print('MAZE_VIEW_WORKER_CASE', index, case, flush=True)


def summarize():
    samples, cases = [], []
    for case in range(layouts.CASE_COUNT):
        directory = OUTPUT/f'case_{case:02d}'
        result = json.loads((directory/'result.json').read_text())
        assert result['data_role'] == 'train'
        frames = json.loads((directory/'in_memory_camera_observations.json').read_text())['frames']
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as data:
            poses, contact = data['base_pose_world'].copy(), data['physics_contact'].copy()
        count = 0
        for frame in range(10, len(frames)-8):
            start = frames[frame]['physical_sample_index']
            last = frames[frame+8]['physical_sample_index']
            if contact[:last+1].any():
                continue
            R = previous.rotation_xyzw(poses[start, 3:])
            for horizon in range(1, 9):
                end = frames[frame+horizon]['physical_sample_index']
                assert frames[frame+horizon]['measured_ns']-frames[frame]['measured_ns'] == horizon*100_000_000
                relative = R.T @ previous.rotation_xyzw(poses[end, 3:])
                delta = (poses[end, :3]-poses[start, :3]) @ R
                samples.append(dict(sample_id=f'maze_view/case_{case:02d}/frame_{frame}/h{horizon}',
                    case=case, frame=frame, horizon_ms=horizon*100, data_role='train',
                    current_rgb=str(directory/f'rgb_{frame:04d}.png'),
                    future_rgb=str(directory/f'rgb_{frame+horizon:04d}.png'),
                    motion=[float(delta[0]), float(delta[1]),
                            float(np.arctan2(relative[1, 0], relative[0, 0]))]))
                count += 1
        cases.append(result | dict(training_windows=count))
    save(OUTPUT/'samples.json', samples)
    result = dict(status='COMPLETE', cases=cases, training_windows=len(samples),
        complete_recordings=sum(c['status']=='COMPLETE' for c in cases),
        physical_stops=sum(c['status']=='PHYSICAL_STOP' for c in cases),
        samples_sha256=digest(OUTPUT/'samples.json'), training_only=True,
        no_model_fit_yet=True, prospective_cohort_excluded=True)
    save(OUTPUT/'result.json', result)
    print('MAZE_VIEW_COLLECTION_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--case', type=int, choices=range(layouts.CASE_COUNT))
    group.add_argument('--worker', type=int, choices=(0, 1))
    group.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.case is not None: run(args.case)
    elif args.worker is not None: worker(args.worker)
    else: summarize()
