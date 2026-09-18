"""Matched live comparison of raw versus learned visual goal costs."""
import json
from pathlib import Path

import numpy as np

from scripts import run_go2_dense_metric_goal_pilot_development as current

RESULT=Path('docs/go2_dense_metric_goal_pilot_result_2026-09-17.json')
FIGURE=Path('docs/go2_dense_metric_goal_pilot_2026-09-17.png')


def main():
    assert not RESULT.exists() and not FIGURE.exists()
    all_rows=[];curves={};comparisons=[]
    for cost,root in (('raw_mse',current.previous.OUTPUT),('learned_metric',current.OUTPUT)):
        for case,(_,_,arm) in enumerate(current.previous.CASES):
            directory=root/f'case_{case:02d}'
            assert not (directory/'failure.json').exists()
            result=json.loads((directory/'result.json').read_text())
            assert result['status']=='COMPLETE' and result['arm']==arm
            decisions=json.loads((directory/'decisions.json').read_text())
            assert len(decisions)==result['decisions']
            if cost=='learned_metric':
                assert all(d['cost_kind']=='learned_physical_goal_metric' for d in decisions)
            if arm=='no_future_action':assert all(len(d['tied_indices'])==6 for d in decisions)
            length=longest=0
            for frame in result['camera_goal_errors'][10:]:
                length=length+1 if frame['within_goal'] else 0;longest=max(longest,length)
            assert result['goal_reached']==bool(longest>=3 and not result['disallowed_contact'] and result['physical_stop'] is None)
            row={k:result[k] for k in ('case','arm','scene','goal_reached','final_within_goal','completed_budget',
                'physical_stop','disallowed_contact','decisions','minimum_xy_error_m','final_xy_error_m','final_yaw_error_deg','wall_s')}
            row.update(cost=cost,maximum_consecutive_goal_frames=longest,
                result_sha256=current.previous.base.digest(directory/'result.json'))
            all_rows.append(row);curves[cost,case]=result['camera_goal_errors']
    for case in range(4):
        old=current.previous.OUTPUT/f'case_{case:02d}';new=current.OUTPUT/f'case_{case:02d}'
        # First forecast must be unchanged: same checkpoint, exact initial
        # images/controls and action candidates; only its scoring differs.
        first_old=json.loads((old/'decision_01.json').read_text())
        first_new=json.loads((new/'decision_01.json').read_text())
        np.testing.assert_allclose(first_new['raw_goal_mse_costs'],first_old['costs'],rtol=0,atol=1e-6)
        for frame in range(11):
            assert (old/f'rgb_{frame:04d}.png').read_bytes()==(new/f'rgb_{frame:04d}.png').read_bytes()
        blind_identical=None
        if current.previous.CASES[case][2]=='no_future_action':
            with np.load(old/'physics_trace.npz',allow_pickle=False) as a, np.load(new/'physics_trace.npz',allow_pickle=False) as b:
                for key in ('base_pose_world','applied_command','physics_contact'):
                    np.testing.assert_array_equal(a[key],b[key])
            blind_identical=True
        comparisons.append(dict(case=case,initial_rgb_exact=True,first_raw_forecast_costs_match=True,
                                blind_native_trajectory_exact=blind_identical))
    summaries={}
    for cost in ('raw_mse','learned_metric'):
        summaries[cost]={}
        for arm in ('action','no_future_action'):
            rows=[r for r in all_rows if r['cost']==cost and r['arm']==arm]
            summaries[cost][arm]=dict(trials=2,transient_goal_visits=sum(r['goal_reached'] for r in rows),
                final_within_goal=sum(r['final_within_goal'] for r in rows),
                contacts=sum(r['disallowed_contact'] for r in rows),completed_budgets=sum(r['completed_budget'] for r in rows),
                mean_final_xy_cm=float(np.mean([r['final_xy_error_m'] for r in rows]))*100)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for col,(active,blind) in enumerate(((0,1),(3,2))):
        for cost,case,label,color,style in (
                ('raw_mse',active,'Raw MSE + action','#777777','--'),
                ('learned_metric',active,'Learned metric + action','#1765a3','-'),
                ('learned_metric',blind,'Action-blind (identical trajectory)','#bd621d','-')):
            values=curves[cost,case];times=[v['time_s']-1.5 for v in values]
            axes[0,col].plot(times,[100*v['xy_error_m'] for v in values],label=label,color=color,ls=style)
            axes[1,col].plot(times,[v['yaw_error_deg'] for v in values],label=label,color=color,ls=style)
        axes[0,col].set(title=f'Exposed transfer geometry {col+1}',ylabel='Goal position error (cm)')
        axes[1,col].set(ylabel='Goal heading error (degrees)',xlabel='Elapsed simulated time after settling (s)')
        axes[0,col].axhline(3,color='#27843b',lw=1,ls=':')
        axes[1,col].axhline(5,color='#27843b',lw=1,ls=':')
        for ax in axes[:,col]:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Prospective goal-cost intervention: unchanged encoder, predictor and native task\nDevelopment simulation pauses during inference; transient visits and final arrival are separate')
    fig.tight_layout();fig.savefig(FIGURE,dpi=160);plt.close(fig)
    report=dict(status='COMPLETE',summaries=summaries,cases=all_rows,matched_controls=comparisons,
        plan_sha256=current.previous.base.digest(current.PLAN),figure=str(FIGURE),
        new_native_trials=4,encoder_and_predictors_unchanged=True,goal_cost_only_intervention=True,
        goal_pose_not_used_for_control=True,full_maze_navigation=False,hardware_validated=False,
        limitations=['two previously exposed related local tasks; single model and execution seed',
            'goal-metric uses extra physical supervision; does not isolate JEPA objective',
            'blind tie-randomization is not a strong reactive baseline',
            'simulation paused during inference; not real-time qualification'])
    current.previous.base.save(RESULT,report)
    print('METRIC_GOAL_COMPARISON_COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
