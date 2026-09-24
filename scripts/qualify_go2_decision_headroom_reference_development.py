"""Score only the prewritten analytical reference-cost sanity cases.

This is not a comparative audit-row runner and consumes no pilot branch data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil

from lewm.decision_headroom_reference_development import ReferenceGeometry, reference_cost


EXPECTED = Path('docs/go2_decision_headroom_reference_sanity_v1_2026-09-23.json')
CAPS = Path('docs/go2_decision_headroom_phase1_caps_v1_2026-09-23.json')


def analytic_trace(case, template, action):
    times = np.arange(401, dtype=np.int64) * 2_000_000
    t = times / 1e9
    yaw = case['start_yaw']
    c, s = np.cos(yaw), np.sin(yaw)
    displacement = np.array([[c, -s], [s, c]]) @ template['local_translation_m'][action]
    final_yaw = yaw + template['yaw_delta_rad'][action]
    final_velocity = template['terminal_speed_m_s'][action] * np.array([np.cos(final_yaw), np.sin(final_yaw)])
    duration = .5
    u = np.clip((t - .3) / duration, 0, 1)
    # Cubic Hermite motion after the common prefix, with explicit final twist.
    position_fraction = 3 * u**2 - 2 * u**3
    endpoint_derivative_fraction = u**3 - u**2
    xy = np.asarray(case['start_xy']) + position_fraction[:, None] * displacement + endpoint_derivative_fraction[:, None] * duration * final_velocity
    velocity = ((6 * u - 6 * u**2)[:, None] * displacement / duration
                + (3 * u**2 - 2 * u)[:, None] * final_velocity)
    delta_yaw = template['yaw_delta_rad'][action]
    final_omega = template['terminal_yaw_rate_rad_s'][action]
    headings = yaw + position_fraction * delta_yaw + endpoint_derivative_fraction * duration * final_omega
    omegas = (6 * u - 6 * u**2) * delta_yaw / duration + (3 * u**2 - 2 * u) * final_omega
    return dict(offset_ns=times, xy=xy, yaw=headings, velocity_xy=velocity,
        yaw_rate=omegas, disallowed_contact=np.zeros(len(t), dtype=bool))


def qualify():
    caps = json.loads(CAPS.read_text())
    root = Path(caps['output_root'])
    stage_a = root.parent / 'go2_maze_view_readout_navigation_v1_attempt_001'
    if not (stage_a / 'result.json').is_file():
        raise RuntimeError('Stage A still pending; retain the recorded Phase 1 start condition')
    owner = json.loads((stage_a / 'process.json').read_text())
    try:
        process = psutil.Process(owner['pid'])
        if abs(process.create_time() - owner['created']) < .01 and process.status() != psutil.STATUS_ZOMBIE:
            raise RuntimeError('Stage A coordinator still live; wait for its exit')
    except psutil.NoSuchProcess:
        pass
    from scripts.run_go2_decision_headroom_pilot_development import PilotBudget
    admission = json.loads((root/'pilot_execution_admission.json').read_text())
    budget = PilotBudget.attach(root, admission)
    expected = json.loads(EXPECTED.read_text())
    assert expected['case_count'] == len(expected['cases']) == caps['collection_caps']['sanity_cases']
    destination = root / 'reference_sanity_v1'
    destination.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    parameters = expected['draft_cost_parameters']
    outcomes = []
    try:
        for case in expected['cases']:
            budget.check('reference_sanity_case', force=True)
            geometry = ReferenceGeometry(case['walls'], [[-3., -3.], [3., 3.]], case['target_xy'],
                radius_m=parameters['footprint_radius_m'], clearance_m=parameters['minimum_clearance_m'],
                resolution_m=parameters['geodesic_grid_resolution_m'])
            costs = {action:reference_cost(geometry, analytic_trace(case, expected['branch_template'], action),
                        arrival_settling=case['arrival_settling'], parameters=parameters)
                     for action in expected['branch_template']['local_translation_m']}
            valid = {a:r['cost_s'] for a,r in costs.items() if r['acceptable'] and r['cost_s'] is not None}
            best = min(valid.values()) if valid else None
            winners = [a for a,v in valid.items() if v <= best + 1e-8]
            optimal_correct = set(winners) == set(case['expected_optimal_actions'])
            exclusions_correct = all(not costs[a]['acceptable'] for a in case['expected_unacceptable_actions'])
            route = geometry.distance_and_heading(np.asarray(case['start_xy']))
            detour_correct = ('geodesic_requirement' not in case or (route['valid'] and not route['direct_segment']))
            outcomes.append(dict(case_id=case['case_id'], category=case['category'],
                expected_optimal_actions=case['expected_optimal_actions'], actual_optimal_actions=winners,
                costs=costs, source_reference_route=route,
                optimal_correct=optimal_correct, exclusions_correct=exclusions_correct,
                detour_correct=detour_correct, passed=optimal_correct and exclusions_correct and detour_correct))
        result = dict(status='PASS' if all(r['passed'] for r in outcomes) else 'FAIL',
            expected_sha256=hashlib.sha256(EXPECTED.read_bytes()).hexdigest(),
            caps_sha256=hashlib.sha256(CAPS.read_bytes()).hexdigest(),
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__), Path('lewm/decision_headroom_reference_development.py'))},
            cases=len(outcomes), passed=sum(r['passed'] for r in outcomes), outcomes=outcomes,
            wall_s=time.monotonic()-started, cpu_s=time.process_time()-cpu_started,
            new_physics=False, model_or_controller_comparisons=False,
            physical_restoration_qualified=False, phase2_authorized=False)
        with (destination / 'result.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'outcomes'}, indent=2))
    except BaseException as error:
        with (destination / 'failure.json').open('x') as stream:
            json.dump(dict(reason=repr(error), cases_completed=len(outcomes), outcomes=outcomes), stream, indent=2)
            stream.write('\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    qualify()
