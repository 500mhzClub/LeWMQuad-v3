"""Reuse physical/input evaluations and check the recorded coordinate treatment."""
import argparse
import json
from types import SimpleNamespace
import numpy as np

from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_neural_rgb_transfer_development as pilot
from scripts import run_go2_mission_coordinate_followup_development as study
from scripts.compare_continuous_navigation_arms_development import path, read


def evaluate(assignment):
    index, arm, mode = study.ASSIGNMENTS[assignment-1]
    experiment = SimpleNamespace(**(vars(pilot.experiment) |
        dict(ROOT=study.ROOT.replace('{mode}', mode))))
    result = bind(pilot.evaluate, experiment=experiment)(index, arm)
    root = path(study.ROOT.format(index=index, arm=arm, mode=mode))
    plans = [p for p in read(root, 'planning.json') if 'selection' in p]
    active = 0
    for plan in plans:
        s = plan['selection']; receipt = s['mission_coordinate_metric']
        if receipt['mode'] != mode:
            raise ValueError('recorded coordinate mode differs')
        if receipt['applied_to_terminal_position']:
            if mode != 'consistent' or s['position_distance_metric'] != 'mission_initial_xy':
                raise ValueError('incorrect active coordinate treatment')
            expected = np.asarray(receipt['planning_goal_initial_xy_m'])-receipt['current_initial_xy_m']
            actual = np.asarray(s['position_metric_matrix']) @ s['waypoint_body_xy_m']
            if not np.allclose(actual, expected, rtol=0., atol=1e-12):
                raise ValueError('terminal target does not match captured mission goal')
            active += 1
        elif 'position_metric_matrix' in s:
            raise ValueError('position metric leaked into an inactive plan')
    treatment = dict(mode=mode, selected_plans=len(plans), terminal_metric_plans=active,
        recorded_coordinate_treatment_verified=bool(plans),
        corrected_treatment_exercised=active>0, independent_arrivals_evaluated=True)
    pilot.previous.previous.save_or_read(root, 'actual_mission_coordinate_treatment_v1.json', treatment)
    print(json.dumps(treatment), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assignment', type=int, choices=range(1, 5), required=True)
    evaluate(parser.parse_args().assignment)
