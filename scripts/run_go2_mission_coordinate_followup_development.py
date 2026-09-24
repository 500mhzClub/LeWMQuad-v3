"""Four fixed original/consistent terminal-metric debugging assignments."""
import argparse
import hashlib
from functools import partial
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.mission_coordinate_runtime_development import MissionCoordinateMixin
from scripts import run_go2_neural_rgb_transfer_development as pilot

ASSIGNMENTS = (
    (0, 'seed_2026091402_no_rgb_jepa', 'original'),
    (0, 'seed_2026091402_no_rgb_jepa', 'consistent'),
    (1, 'seed_2026091402_full_jepa', 'consistent'),
    (1, 'seed_2026091402_full_jepa', 'original'),
)
ROOT = 'go2_mission_coordinate_{mode}_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


class MissionCoordinateRuntime(MissionCoordinateMixin, pilot.InputCheckedRuntime):
    pass


def mark(name, value):
    if name == 'launch.json':
        sources = (__file__, 'lewm/mission_coordinate_runtime_development.py',
            'lewm/mission_coordinate_metric_development.py',
            'lewm/delayed_action_planning_development.py',
            'lewm/waypoint_alignment_planning_development.py',
            'lewm/predictive_arrival_hold_development.py',
            'lewm/arrival_entry_terminal_priority_development.py')
        value = value | dict(experiment='mission_coordinate_followup_v1',
            comparison='original_vs_consistent_terminal_position_metric', coordinate_mode=MODE,
            planned_conditions=['original', 'consistent'], planned_native_assignments=4,
            fixed_sequential_assignments=list(ASSIGNMENTS),
            new_independent_development_layout=False, exposed_development_layout=True,
            repeated_exposed_development_maze=True,
            layout_novelty_scope='targeted_cases_from_completed_neural_rgb_pilot',
            actual_runtime_class='MissionCoordinateRuntime',
            training_seed_selection_used_runtime_outcomes=True,
            targeted_controller_debugging=True, generalization_claimed=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                                   for p in sources})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(pilot.annotate, ARM=ARM,
         RAW_WRITE=bind(mark, MODE=MODE, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assignment', type=int, choices=range(1, 5), required=True)
    args = parser.parse_args()
    index, arm, mode = ASSIGNMENTS[args.assignment-1]
    base = pilot.study.cohort.stable.source.BASE
    if args.assignment > 1:
        i, a, m = ASSIGNMENTS[args.assignment-2]
        if not (base/ROOT.format(index=i, arm=a, mode=m)/'continuous_native_arrival_evaluation.json').is_file():
            raise ValueError('evaluate preceding fixed assignment first')
    if shutil.disk_usage(base).free < 4*1024**3:
        raise ValueError('four GiB free required for one full-budget recording')
    gyro = SimpleNamespace(**(vars(pilot.study.cohort.gyro) |
        dict(initialize_obstacles=pilot.reference.previous.initialize_obstacles)))
    cohort = SimpleNamespace(**(vars(pilot.study.cohort) | dict(gyro=gyro)))
    previous_argv = sys.argv
    sys.argv = [previous_argv[0], '--layout-index', str(index), '--arm', arm]
    try:
        bind(pilot.study.main, ROOT=ROOT.replace('{mode}', mode), ARMS=pilot.ARMS,
            layouts=pilot.layouts, INVENTORY=pilot.INVENTORY, INVENTORY_SHA256=pilot.INVENTORY_SHA256,
            FreshCameraSession=pilot.FreshCameraSession, annotate=bind(annotate, MODE=mode),
            cohort=cohort, initialize_pose=pilot.reference.initialize_pose,
            initialize_registration=pilot.reference.previous.initialize_registration,
            SeededPredictiveRuntime=partial(MissionCoordinateRuntime, coordinate_mode=mode))()
    finally:
        sys.argv = previous_argv


if __name__ == '__main__':
    main()
