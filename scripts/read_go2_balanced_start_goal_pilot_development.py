"""Retain all local outcomes and distinguish new trials from reused baselines."""
import json
from pathlib import Path

import numpy as np

from scripts import run_go2_balanced_start_goal_pilot_development as pilot

RESULT=Path('docs/go2_balanced_start_goal_pilot_result_2026-09-17.json')


def main():
    assert not RESULT.exists()
    rows=[]
    for case,(scene,goal,arm) in enumerate(pilot.CASES):
        directory=pilot.root(case)/f'case_{case:02d}'
        assert not (directory/'failure.json').exists()
        result=json.loads((directory/'result.json').read_text())
        decisions=json.loads((directory/'decisions.json').read_text())
        assert result['status']=='COMPLETE' and len(decisions)==result['decisions']
        assert result['arm']==arm and result['scene']==scene
        assert all(d['predictor_arm']==arm and d['goal_cost_model']=='cross_trajectory_goal_metric' for d in decisions)
        for d in decisions:
            if d['arrival_latched']:
                assert d['action']=='hold'
            else:
                assert d['cost_kind']=='learned_physical_goal_metric'
                if arm.endswith('no_future_action'):
                    assert len(d['tied_indices'])==6 and len(set(d['costs']))==1
        reference=pilot.previous.previous.OUTPUT/f'case_{2*(case//3):02d}'
        for frame in range(11):
            assert (directory/f'rgb_{frame:04d}.png').read_bytes()==(reference/f'rgb_{frame:04d}.png').read_bytes()
        first=next((d['tick'] for d in decisions if d['arrival_latched']),None)
        if first is not None:assert all(d['arrival_latched'] for d in decisions if d['tick']>=first)
        errors=result['camera_goal_errors']
        row={k:result[k] for k in ('case','scene','arm','goal_reached','final_within_goal','completed_budget',
                                   'physical_stop','disallowed_contact','decisions','minimum_xy_error_m',
                                   'final_xy_error_m','final_yaw_error_deg','wall_s')}
        row.update(first_latched_tick=first,false_arrival_latch=first is not None and not errors[first]['within_goal'],
                   actions=[d['action'] for d in decisions],initial_reference_rgb_exact=True,
                   result_path=str(directory/'result.json'),result_sha256=pilot.pilot.base.digest(directory/'result.json'))
        rows.append(row)
    summaries={}
    for arm in pilot.ARMS:
        selected=[r for r in rows if r['arm']==arm]
        summaries[arm]=dict(trials=len(selected),
            final_arrivals=sum(r['final_within_goal'] and r['completed_budget'] and not r['disallowed_contact'] for r in selected),
            transient_visits=sum(r['goal_reached'] for r in selected),contacts=sum(r['disallowed_contact'] for r in selected),
            false_arrival_latches=sum(r['false_arrival_latch'] for r in selected),
            mean_final_xy_cm=float(np.mean([r['final_xy_error_m'] for r in selected]))*100,
            mean_final_yaw_deg=float(np.mean([r['final_yaw_error_deg'] for r in selected])))
    reference=json.loads(Path('docs/go2_cross_trajectory_goal_pilot_result_2026-09-17.json').read_text())
    feedback=json.loads(Path('docs/go2_fresh_visual_goal_comparison_result_2026-09-17.json').read_text())
    report=dict(status='COMPLETE',cases=rows,summaries=summaries,new_trials=12,
                reused_parent_cross_metric_summary=reference['summary'],
                reused_direct_feedback_summary=feedback['summaries']['direct_feedback'],
                mixed_action_blind_new_trial=False,
                action_blind_weights_do_not_change_shared_uniform_tie_and_observed_arrival_policy=True,
                plan_sha256=pilot.pilot.base.digest(pilot.PLAN),
                full_maze_navigation=False,real_time_qualified=False,hardware_validated=False,
                limitations=['four exposed local tasks','one training seed','same known arrival-latch failure',
                             'coverage and future-action input effects, not JEPA encoder-objective isolation'])
    pilot.pilot.base.save(RESULT,report)
    print('BALANCED_GOAL_PILOT_COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
