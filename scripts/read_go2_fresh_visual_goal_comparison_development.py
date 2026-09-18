"""Fresh task outcomes; distinguish approach from learned arrival recognition."""
import json
from pathlib import Path
import re

import numpy as np

from scripts import run_go2_fresh_visual_goal_comparison_development as pilot

RESULT = Path('docs/go2_fresh_visual_goal_comparison_result_2026-09-17.json')


def main():
    assert not RESULT.exists()
    rows=[]
    for case in range(8):
        root=pilot.OUTPUT/f'case_{case:02d}'
        assert not (root/'failure.json').exists()
        result=json.loads((root/'result.json').read_text())
        decisions=json.loads((root/'decisions.json').read_text())
        assert result['status']=='COMPLETE' and len(decisions)==result['decisions']
        controller='world_model' if case%2==0 else 'direct_feedback'
        if controller=='direct_feedback':
            assert result['observed_forecast_windows']==0
            assert all(d['costs'] is None and d['forecast_horizon_ms'] is None for d in decisions)
        else:
            assert all(d['cost_kind']=='learned_physical_goal_metric' for d in decisions if not d['arrival_latched'])
        pair=pilot.OUTPUT/f'case_{case-case%2:02d}'
        for frame in range(11):
            assert (root/f'rgb_{frame:04d}.png').read_bytes()==(pair/f'rgb_{frame:04d}.png').read_bytes()
        assert not any(re.fullmatch(r'(?:native_)?depth_[0-9]+\.npz',p.name) for p in root.iterdir())
        assert json.loads((root/'recording_policy.json').read_text())['depth_recorded'] is False
        first=next((d['tick'] for d in decisions if d['arrival_latched']),None)
        if first is not None:assert all(d['arrival_latched'] and d['action']=='hold' for d in decisions if d['tick']>=first)
        errors=result['camera_goal_errors'];longest=streak=0
        for e in errors[10:]:
            streak=streak+1 if e['within_goal'] else 0;longest=max(longest,streak)
        observed_goal=[errors[d['tick']]['within_goal'] for d in decisions if first is None or d['tick']<=first]
        normalized=[(e['xy_error_m']/.03)**2+(e['yaw_error_deg']/5)**2 for e in errors[10:]]
        row={k:result[k] for k in ('case','scene','goal_reached','final_within_goal','completed_budget',
            'physical_stop','disallowed_contact','decisions','initial_xy_error_m','initial_yaw_error_deg',
            'minimum_xy_error_m','final_xy_error_m','final_yaw_error_deg','wall_s','observed_forecast_windows')}
        row.update(controller=controller,first_latched_tick=first,
            actual_within_at_latch=errors[first]['within_goal'] if first is not None else None,
            false_arrival_latch=first is not None and not errors[first]['within_goal'],
            observed_within_goal_before_latch=any(observed_goal),
            maximum_consecutive_goal_frames=longest,minimum_normalized_pose_cost=min(normalized),
            actions=[d['action'] for d in decisions],initial_pair_rgb_exact=True,
            result_path=str(root/'result.json'),result_sha256=pilot.pilot.base.digest(root/'result.json'))
        rows.append(row)
    summaries={}
    for controller in ('world_model','direct_feedback'):
        selected=[r for r in rows if r['controller']==controller]
        summaries[controller]=dict(trials=len(selected),
            contact_free_final_arrivals=sum(r['final_within_goal'] and r['completed_budget'] and not r['disallowed_contact'] for r in selected),
            transient_goal_visits=sum(r['goal_reached'] for r in selected),
            contacts=sum(r['disallowed_contact'] for r in selected),
            false_arrival_latches=sum(r['false_arrival_latch'] for r in selected),
            mean_final_xy_cm=float(np.mean([r['final_xy_error_m'] for r in selected]))*100,
            mean_final_heading_deg=float(np.mean([r['final_yaw_error_deg'] for r in selected])))
    report=dict(status='COMPLETE',cases=rows,summaries=summaries,
        plan_sha256=pilot.pilot.base.digest(pilot.PLAN),fresh_tasks=4,new_controller_trials=8,
        controllers_frozen_before_task_setup=True,no_training_or_tuning_on_fresh_tasks=True,
        depth_recorded=False,full_maze_navigation=False,real_time_qualified=False,hardware_validated=False,
        limitations=['same local obstruction family and appearance seed; not four unrelated maze environments',
            'one checkpoint and feedback law; no isolated JEPA-training advantage',
            'goal setup trajectories are not navigation evidence',
            'known irreversible arrival latch is shared by both controllers'])
    pilot.pilot.base.save(RESULT,report)
    print(json.dumps(dict(summaries=summaries,cases=rows),indent=2),flush=True)


if __name__=='__main__':main()
