"""PRELIMINARY videos from logged preliminary-test runs (Andrew, 1 October 2026).

Renders with the unchanged V4 video renderer (`render_go2_capability_v4_video_development.py`).
Pass 1 is its deterministic replay, which asserts every decision, dispatch, trace value,
published pose and mission row against the log. Pass 2 is the chase camera. Bound in for the
development-mode preliminary run:
- the runtime is composed with the logged run's own development fixes, from its dev_run.json:
  recovery off = the pose-loss record only, which changes no decision; recovery on = all six;
- the owner's budget is the development budget (no programme window; reserve checks unchanged);
- labels: the header says PRELIMINARY (prelim_test_v1, declassified set, not a sealed
  result) and recovery on or off; C0 is marked C0* (relaxed-check copy of the owner's
  harness); files are named <ctrl>_prelim<maze>_ep0_recovery-<on|off>.
"""
import argparse
import inspect
import json
from pathlib import Path

import cv2

from lewm.dev_harness_fixes_development import compose
from lewm.eligible_floor_registration_development import bind
from lewm.navigation_capability_completed_support_development import CompletedSupportRuntimeMixin
from scripts import render_go2_capability_v4_video_development as renderer
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.run_go2_dev_mission_development import DevBudget

STEM_OLD = 'stem = f\'{config["controller"]}_{"dev" if pipeline_test else "val"}{packet["maze_id"]:02d}_ep{packet["episode_index"]}\''
STATUS_OLD = "status='PIPELINE_TEST_REPLAY_VERIFIED' if pipeline_test else 'CAPABILITY_VIDEO_REPLAY_VERIFIED'"


def header_for(arm, recovery, outcome):
    note = ' | C0*: relaxed-check copy of owner harness' if arm == 'C0' else ''
    return f'PRELIMINARY (prelim_test_v1, declassified; not sealed) | recovery {recovery.upper()} | {outcome}{note}'


def prelim_main(recovery, outcome):
    """The renderer's main with only the file stem and metadata status relabelled."""
    source = inspect.getsource(renderer.main)
    assert source.count(STEM_OLD) == 1 and source.count(STATUS_OLD) == 1, 'renderer main changed'
    source = source.replace(STEM_OLD, 'stem = f\'{config["controller"]}_prelim{packet["maze_id"]:02d}_ep{packet["episode_index"]}_recovery-' + recovery + '\'')
    source = source.replace(STATUS_OLD, "status='PRELIMINARY_VIDEO_REPLAY_VERIFIED', preliminary_label=PRELIM_LABEL")
    namespace = dict(renderer.__dict__)
    exec(compile(source, renderer.__file__, 'exec'), namespace)
    return namespace['main']


class PrelimCanvas(renderer.Canvas):
    header = None

    def draw(self, chase, pose, elapsed):
        canvas = super().draw(chase, pose, elapsed)
        y = 770+6*44
        cv2.rectangle(canvas, (660, y-30), (1920, y+12), (20, 20, 20), -1)
        cv2.putText(canvas, self.header, (670, y), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 225, 120), 1, cv2.LINE_AA)
        return canvas


def main(source_root, label, rate):
    dev = json.loads((source_root/'dev_run.json').read_text())
    assert dev['set'] == 'prelim_test', 'preliminary-test runs only'
    recovery = dev.get('recovery') or ('on' if 'terminal' in dev['fixes'] else 'off')
    evaluation = json.loads((source_root/'episode_evaluation.json').read_text())
    outcome = 'round trip SUCCESS' if evaluation['round_trip_success'] else 'round trip FAILED (labelled failure example)'
    PrelimCanvas.header = header_for(dev['controller'], recovery, outcome)
    owner.Budget = DevBudget
    mixin = compose(dev['fixes'], CompletedSupportRuntimeMixin)
    verify = bind(renderer.verify, CompletedSupportRuntimeMixin=mixin)
    run = prelim_main(recovery, outcome)
    run = bind(run, verify=verify, Canvas=PrelimCanvas,
               PRELIM_LABEL=dict(header=PrelimCanvas.header, fixes=dev['fixes'], recovery=recovery, c3_decoder=dev.get('c3_decoder'),
                                 owner_run=dev.get('owner_run'), source_dev_run=dev))
    run(source_root, label, rate, True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--rate', required=True, help='HUD line: the controller\'s preliminary success rate')
    a = p.parse_args()
    main(a.source_root, a.label, a.rate)
