"""CPU-only decision diagnostic from already completed dense branch forecasts.

Each recorded successor image serves as a reachable visual goal. Choose among
the three forecasts by feature MSE to that goal, then measure endpoint error
using recorded target-only motion. This is retrospective, not navigation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


WINDOWS = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_short_pulse_learning_v1_attempt_001/windows.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(source, windows=WINDOWS):
    result = json.loads(source.read_text())
    assert result['status'] == 'COMPLETE'
    rows = [r for r in json.loads(windows.read_text()) if r['observation_frame'] == 13]
    by_trial = {r['trial']: r for r in rows}
    assert len(rows) == len(by_trial) == 36
    groups = []
    for group in result['groups']:
        trials = group['trials']
        assert len(trials) == 3
        selected = [by_trial[t] for t in trials]
        assert all(r['available'] and r['data_role'] == group['role'] for r in selected)
        targets = [r['targets'][4] for r in selected]
        assert all(t['motion_valid'] and t['offset_ns'] == 500_000_000
                   and t['future_observation_index'] == 18 for t in targets)
        poses = [t['motion'] for t in targets]
        models = {}
        for arm, record in group['models'].items():
            matrix = record['mse_matrix']  # rows: candidate; columns: goal
            assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
            assert all(math.isfinite(v) for row in matrix for v in row)
            goals = []
            for j in range(3):
                costs = [matrix[i][j] for i in range(3)]
                choices = [i for i, value in enumerate(costs) if value == min(costs)]
                position = [math.hypot(poses[i][0]-poses[j][0],
                                       poses[i][1]-poses[j][1])*1000 for i in choices]
                yaw = [abs(math.atan2(math.sin(poses[i][2]-poses[j][2]),
                                     math.cos(poses[i][2]-poses[j][2])))*180/math.pi
                       for i in choices]
                goals.append(dict(goal_trial=trials[j], selected_trials=[trials[i] for i in choices],
                    expected_position_error_mm=mean(position), expected_heading_error_deg=mean(yaw),
                    expected_correct_action=float(j in choices)/len(choices)))
            models[arm] = dict(goals=goals, **{k: mean(g[k] for g in goals) for k in (
                'expected_position_error_mm', 'expected_heading_error_deg', 'expected_correct_action')})
        groups.append(dict(role=group['role'], cluster=group['cluster'],
                           prefix_action=group['prefix_action'], models=models))
    summaries = {}
    for role in ('train', 'geometry_transfer'):
        role_groups = [g for g in groups if g['role'] == role]
        assert len(role_groups) == 6
        summaries[role] = {}
        for arm in role_groups[0]['models']:
            summaries[role][arm] = {
                k: mean(g['models'][arm][k] for g in role_groups) for k in (
                    'expected_position_error_mm', 'expected_heading_error_deg', 'expected_correct_action')}
            summaries[role][arm]['groups_better_position_than_persistence'] = sum(
                g['models'][arm]['expected_position_error_mm'] <
                g['models']['persistence']['expected_position_error_mm'] for g in role_groups)
    return dict(status='COMPLETE', source=str(source), source_sha256=digest(source),
        windows_sha256=digest(windows), script_sha256=digest(Path(__file__)),
        horizon_ms=500, goals_per_role=18, matched_groups_per_role=6,
        layouts_per_role=2, summaries=summaries, groups=groups,
        selection='minimum forecast-to-goal feature MSE; uniform probability over exact ties',
        physical_labels_used_for_selection=False, goal_images_are_recorded_future_images=True,
        navigation_tested=False, new_model_execution=False,
        limitations=['retrospective reachable visual-goal diagnostic on exposed pulse branches',
            '18 goals per role are dependent; six history groups across two geometries',
            'millimetre-scale endpoint differences; not sustained-command navigation',
            'uniform tie expectation is an analytic reference, not a deployed policy',
            'no uncertainty interval or JEPA-training superiority claim'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = score(args.source)
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value['summaries'], indent=2))
