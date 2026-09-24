"""Compare the mixed-pair goal metric against retained within-only controls."""
import json
from pathlib import Path

import numpy as np

from scripts import run_go2_cross_trajectory_goal_pilot_development as pilot

RESULT=Path('docs/go2_cross_trajectory_goal_pilot_result_2026-09-17.json')


def main():
    assert not RESULT.exists()
    reference=json.loads(Path('docs/go2_fresh_visual_goal_comparison_result_2026-09-17.json').read_text())
    evaluation=json.loads(Path('docs/go2_cross_trajectory_goal_metric_evaluation_2026-09-17.json').read_text())
    rows=[]
    for case in range(4):
        root=pilot.OUTPUT/f'case_{case:02d}';assert not (root/'failure.json').exists()
        result=json.loads((root/'result.json').read_text());decisions=json.loads((root/'decisions.json').read_text())
        assert result['status']=='COMPLETE' and len(decisions)==result['decisions']
        assert all(d['goal_cost_model']=='cross_trajectory_goal_metric' for d in decisions)
        assert all(d['cost_kind']=='learned_physical_goal_metric' for d in decisions if not d['arrival_latched'])
        original=pilot.previous.OUTPUT/f'case_{2*case:02d}'
        for frame in range(11):
            assert (root/f'rgb_{frame:04d}.png').read_bytes()==(original/f'rgb_{frame:04d}.png').read_bytes()
        if case==1:
            expected=evaluation['groups'][0]['models']['mixed_pairs']['predicted']
            np.testing.assert_allclose(decisions[0]['costs'],expected['costs'],rtol=1e-6,atol=1e-5)
            assert decisions[0]['action'] in expected['chosen_actions']
        first=next((d['tick'] for d in decisions if d['arrival_latched']),None)
        errors=result['camera_goal_errors'];longest=streak=0
        if first is not None:assert all(d['arrival_latched'] and d['action']=='hold' for d in decisions if d['tick']>=first)
        for e in errors[10:]:
            streak=streak+1 if e['within_goal'] else 0;longest=max(longest,streak)
        row={k:result[k] for k in ('case','scene','goal_reached','final_within_goal','completed_budget',
            'physical_stop','disallowed_contact','decisions','minimum_xy_error_m','final_xy_error_m','final_yaw_error_deg','wall_s')}
        row.update(first_latched_tick=first,false_arrival_latch=first is not None and not errors[first]['within_goal'],
            maximum_consecutive_goal_frames=longest,actions=[d['action'] for d in decisions],
            initial_reference_rgb_exact=True,result_path=str(root/'result.json'),result_sha256=pilot.pilot.base.digest(root/'result.json'),
            within_only_reference=reference['cases'][2*case],direct_feedback_reference=reference['cases'][2*case+1])
        rows.append(row)
    summary=dict(trials=4,contact_free_final_arrivals=sum(r['final_within_goal'] and r['completed_budget'] and not r['disallowed_contact'] for r in rows),
        transient_goal_visits=sum(r['goal_reached'] for r in rows),contacts=sum(r['disallowed_contact'] for r in rows),
        false_arrival_latches=sum(r['false_arrival_latch'] for r in rows),
        mean_final_xy_cm=float(np.mean([r['final_xy_error_m'] for r in rows]))*100,
        mean_final_heading_deg=float(np.mean([r['final_yaw_error_deg'] for r in rows])))
    report=dict(status='COMPLETE',summary=summary,cases=rows,reference_summaries=reference['summaries'],
        plan_sha256=pilot.pilot.base.digest(pilot.PLAN),references_reused_not_new_trials=True,
        training_budget_matched=True,encoder_predictor_arrival_rule_unchanged=True,
        full_maze_navigation=False,real_time_qualified=False,hardware_validated=False,
        limitations=['four exposed related local tasks, one fit seed','goal-supervision change, not JEPA objective isolation',
            'independent prospective navigation still required'])
    pilot.pilot.base.save(RESULT,report);print('CROSS_GOAL_PILOT_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
