"""Report prospective predictor continuation outcomes, including native stops."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts import run_go2_dense_task_goal_pilot_development as pilot

RESULT = Path('docs/go2_dense_task_goal_pilot_result_2026-09-17.json')


def main(recovery_path=None):
    assert not RESULT.exists()
    recovery = json.loads(Path(recovery_path).read_text()) if recovery_path else None
    if recovery:
        assert recovery['original_plan_sha256'] == pilot.pilot.base.digest(pilot.PLAN)
    rows = []
    for case,(scene,_,arm) in enumerate(pilot.CASES):
        directory = Path(recovery['case_directories'][str(case)]) if recovery else pilot.OUTPUT/f'case_{case:02d}'
        assert not (directory/'failure.json').exists()
        result = json.loads((directory/'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['arm'] == arm and result['scene'] == scene
        decisions = json.loads((directory/'decisions.json').read_text())
        assert len(decisions) == result['decisions']
        assert all(d['predictor_arm'] == arm and d['cost_kind'] == 'learned_physical_goal_metric' for d in decisions)
        reference_case = 0 if case < 2 else 3
        previous = pilot.previous.OUTPUT/f'case_{reference_case:02d}'
        for frame in range(11):
            assert (directory/f'rgb_{frame:04d}.png').read_bytes() == (previous/f'rgb_{frame:04d}.png').read_bytes()
        streak = longest = 0
        for frame in result['camera_goal_errors'][10:]:
            streak = streak+1 if frame['within_goal'] else 0; longest = max(longest,streak)
        assert result['goal_reached'] == bool(longest >= 3 and not result['disallowed_contact'] and result['physical_stop'] is None)
        row = {k:result[k] for k in ('case','scene','arm','goal_reached','final_within_goal','completed_budget',
            'physical_stop','disallowed_contact','decisions','minimum_xy_error_m','final_xy_error_m','final_yaw_error_deg','wall_s')}
        row.update(maximum_consecutive_goal_frames=longest,initial_rgb_exact=True,
            result_path=str(directory/'result.json'),
            result_sha256=pilot.pilot.base.digest(directory/'result.json'))
        rows.append(row)
        if recovery and case in recovery['reexecute_cases']:
            interrupted = pilot.OUTPUT/f'case_{case:02d}'
            old_decisions = json.loads((interrupted/'decisions.json').read_text())
            for old,new in zip(old_decisions,decisions,strict=False):
                assert old['action'] == new['action'] and old['requested_commands'] == new['requested_commands']
                np.testing.assert_allclose(old['costs'],new['costs'],rtol=0,atol=1e-6)
            assert len(decisions) >= len(old_decisions)
            with np.load(interrupted/'physics_trace.npz',allow_pickle=False) as old, np.load(directory/'physics_trace.npz',allow_pickle=False) as new:
                for key in ('base_pose_world','applied_command','physics_contact'):
                    np.testing.assert_array_equal(old[key],new[key][:len(old[key])])
            row['interrupted_execution_prefix_reproduced']=True
    summaries = {}
    for arm in ('dense_action','metric_action'):
        selected = [r for r in rows if r['arm'] == arm]
        summaries[arm] = dict(trials=len(selected),transient_goal_visits=sum(r['goal_reached'] for r in selected),
            final_within_goal=sum(r['final_within_goal'] for r in selected),
            contacts=sum(r['disallowed_contact'] for r in selected),
            mean_final_xy_cm=float(np.mean([r['final_xy_error_m'] for r in selected]))*100)
    old = json.loads(Path('docs/go2_dense_metric_goal_pilot_result_2026-09-17.json').read_text())
    report = dict(status='COMPLETE',cases=rows,summaries=summaries,planned_task_cells=4,terminal_task_records=4,
        predecessor_summaries=old['summaries']['learned_metric'],
        reused_blind_reference=[r for r in old['cases'] if r['cost']=='learned_metric' and r['arm']=='no_future_action'],
        plan_sha256=pilot.pilot.base.digest(pilot.PLAN),encoder_and_goal_metric_unchanged=True,
        storage_interrupted_attempts=recovery['interrupted_attempts'] if recovery else [],
        total_native_attempts=6 if recovery else 4,
        recovery_plan_sha256=pilot.pilot.base.digest(recovery_path) if recovery else None,
        matched_extra_training_budget=True,full_maze_navigation=False,hardware_validated=False,
        limitations=['two exposed related local tasks, one seed','blind uniform-tie control is not a strong reactive baseline',
            'blind reference trajectories are reused, not additional replicates','extra physical supervision; no isolated JEPA training claim',
            'simulation paused during inference'])
    pilot.pilot.base.save(RESULT,report)
    print('TASK_GOAL_PILOT_COMPLETE',json.dumps(summaries),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recovery');args=parser.parse_args();main(args.recovery)
