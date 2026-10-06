"""Collect training-role rest-start and in-place-turn recordings on the C3-v2 recording mazes.

Reuses the completed maze-view training collector (session, contacts, persistence) with a
fixed tape of repeated starts from rest: every motion block (forward, arcs, in-place turns)
follows a 1.2-s hold. No navigation policy, no model, no outcome-based selection or retry.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import psutil

from lewm import c3v2_recording_layouts_development as layouts
from lewm.eligible_floor_registration_development import bind
from scripts import collect_go2_full_heading_training_development as previous
from scripts import collect_go2_maze_view_training_development as maze_view

OUTPUT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/'
              'go2_navigation_capability_v1_attempt_001/c3v2_rest_turn_recordings_v1')
TRIALS = tuple(f'c3v2_rest_turn_{i:02d}' for i in range(layouts.CASE_COUNT))
save, digest = previous.fit.save, previous.fit.digest


class RecordingPhysicalInit(previous.IndependentRoundTripPhysicalInit):
    def __init__(self, spec, *, backend):
        bind(previous.IndependentRoundTripPhysicalInit.__init__,
             specification=layouts.specification, pack=layouts.pack)(self, spec, backend=backend)
        self._runtime['high_level'] = 'fixed rest-start training excitation; no navigation policy'


class TrainingSession(previous.RGBNavigationRetentionMixin, previous.NogilDrawingMixin,
        previous.LiveDepthNoiseMixin, previous.CompactDepthRetentionMixin,
        previous.LzmaRawDepthPairedCameraSession, RecordingPhysicalInit):
    pass


def schedule():
    blocks = [('hold', 12), ('forward', 8), ('hold', 12), ('left_turn', 10), ('hold', 12), ('right_arc', 8),
              ('hold', 12), ('right_turn', 10), ('hold', 12), ('left_arc', 8), ('hold', 12), ('forward', 8),
              ('hold', 12), ('right_turn', 15), ('hold', 12), ('left_turn', 15), ('hold', 12)]
    tape, phases = [], []
    for block, (action, count) in enumerate(blocks):
        tape.extend([previous.candidate_commands(action)[0]]*count)
        phases.extend([f'block_{block:02d}_{action}']*count)
    assert len(tape) == 190
    return tape, phases


def prepare():
    free = shutil.disk_usage(OUTPUT.parent).free
    assert free > 16*1024**3 and psutil.virtual_memory().available > 16*1024**3
    tape, phases = schedule()
    specs = list(layouts.specifications())
    for spec in specs:
        definition = layouts.pack(spec)
        np.testing.assert_allclose(definition.robot.spawn_xyz_m[:2], spec['geometry']['spawn_se2_world'][:2], atol=0, rtol=0)
        assert spec['data_role'] == 'train'
    OUTPUT.mkdir(exist_ok=False)
    save(OUTPUT/'plan.json', dict(trials=TRIALS, specifications=specs, tape=tape, phases=phases,
        source_sha256=digest(__file__), dependencies={str(Path(p).resolve()): digest(p)
            for p in (previous.__file__, maze_view.__file__, layouts.__file__)},
        data_role='train', splits={s['layout_index']: s['recording_split'] for s in specs},
        free_bytes=free, minimum_free_bytes=2*1024**3, depth_arrays_retained=False, render_robot=True,
        fixed_tape_training_not_navigation=True, no_retry_or_outcome_based_selection=True,
        sample_rule='All contact-free 100-800-ms windows with departure >=10 frames and all eight horizons present (as maze-view training).',
        approval='Andrew Knowles 29 September 2026: C3 readout training-data change; recordings only on mazes outside every existing set.'))
    print('C3V2_RECORDING_PREPARED', len(specs), len(tape)+1, flush=True)


def run(case):
    plan = json.loads((OUTPUT/'plan.json').read_text())
    for path, identity in plan['dependencies'].items():
        assert digest(path) == identity
    bind(previous.run, OUTPUT=OUTPUT, TRIALS=TRIALS, specification=layouts.specification,
         TrainingSession=TrainingSession, __file__=__file__)(case)


def worker(index, workers):
    for case in range(index, layouts.CASE_COUNT, workers):
        with (OUTPUT/f'case_{case:02d}_process.log').open('x') as log:
            subprocess.run([sys.executable, __file__, '--case', str(case)], stdout=log, stderr=subprocess.STDOUT, check=True)
        print('C3V2_RECORDING_CASE', index, case, flush=True)


def summarize():
    # Same window rule and target convention as the maze-view training summary.
    bind(maze_view.summarize, OUTPUT=OUTPUT, layouts=layouts)()
    samples = json.loads((OUTPUT/'samples.json').read_text())
    split = json.loads((OUTPUT/'plan.json').read_text())['splits']
    counts = {}
    for row in samples:
        counts[split[str(row['case'])]] = counts.get(split[str(row['case'])], 0)+1
    print('C3V2_RECORDING_SPLIT_WINDOWS', json.dumps(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--case', type=int, choices=range(layouts.CASE_COUNT))
    group.add_argument('--worker', type=int)
    group.add_argument('--summarize', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.case is not None: run(args.case)
    elif args.worker is not None: worker(args.worker, args.workers)
    else: summarize()
