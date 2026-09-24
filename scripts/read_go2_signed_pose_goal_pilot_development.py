"""Paired exposed-task outcomes after replacing only the predicted-goal cost."""
import json
from pathlib import Path

import numpy as np

from scripts import run_go2_signed_pose_goal_pilot_development as pilot

RESULT=Path('docs/go2_signed_pose_goal_pilot_result_2026-09-17.json')


def main():
    assert not RESULT.exists()
    original=json.loads(Path('docs/go2_fresh_visual_goal_comparison_result_2026-09-17.json').read_text())
    rows=[]
    for case in range(4):
        root=pilot.OUTPUT/f'case_{case:02d}';assert not (root/'failure.json').exists()
        result=json.loads((root/'result.json').read_text())
        decisions=json.loads((root/'decisions.json').read_text())
        assert result['status']=='COMPLETE' and len(decisions)==result['decisions']
        assert all(d['goal_cost_model']=='squared_signed_goal_pose' for d in decisions)
        assert all(d['cost_kind']=='squared_signed_goal_pose' for d in decisions if not d['arrival_latched'])
        reference=pilot.previous.OUTPUT/f'case_{2*case:02d}'
        for frame in range(11):
            assert (root/f'rgb_{frame:04d}.png').read_bytes()==(reference/f'rgb_{frame:04d}.png').read_bytes()
        if case==1:
            diagnostic=json.loads(Path('docs/go2_signed_goal_cost_diagnostic_2026-09-17.json').read_text())
            np.testing.assert_allclose(decisions[0]['costs'],[r['predicted_signed_cost'] for r in diagnostic['rows']],rtol=1e-6,atol=1e-6)
            assert decisions[0]['action']==diagnostic['predicted_choice']
        first=next((d['tick'] for d in decisions if d['arrival_latched']),None)
        if first is not None:assert all(d['arrival_latched'] and d['action']=='hold' for d in decisions if d['tick']>=first)
        errors=result['camera_goal_errors'];longest=streak=0
        for e in errors[10:]:
            streak=streak+1 if e['within_goal'] else 0;longest=max(longest,streak)
        row={k:result[k] for k in ('case','scene','goal_reached','final_within_goal','completed_budget','physical_stop',
            'disallowed_contact','decisions','minimum_xy_error_m','final_xy_error_m','final_yaw_error_deg','wall_s')}
        row.update(first_latched_tick=first,false_arrival_latch=first is not None and not errors[first]['within_goal'],
            maximum_consecutive_goal_frames=longest,actions=[d['action'] for d in decisions],
            initial_reference_rgb_exact=True,result_path=str(root/'result.json'),result_sha256=pilot.pilot.base.digest(root/'result.json'),
            previous_world_model=original['cases'][2*case],direct_feedback_reference=original['cases'][2*case+1])
        rows.append(row)
    summary=dict(trials=4,contact_free_final_arrivals=sum(r['final_within_goal'] and r['completed_budget'] and not r['disallowed_contact'] for r in rows),
        transient_goal_visits=sum(r['goal_reached'] for r in rows),contacts=sum(r['disallowed_contact'] for r in rows),
        false_arrival_latches=sum(r['false_arrival_latch'] for r in rows),
        mean_final_xy_cm=float(np.mean([r['final_xy_error_m'] for r in rows]))*100,
        mean_final_heading_deg=float(np.mean([r['final_yaw_error_deg'] for r in rows])))
    report=dict(status='COMPLETE',summary=summary,cases=rows,reference_summaries=original['summaries'],
        references_reused_not_new_trials=True,plan_sha256=pilot.pilot.base.digest(pilot.PLAN),
        no_training=True,cost_only_intervention=True,full_maze_navigation=False,real_time_qualified=False,hardware_validated=False,
        limitations=['four exposed tasks informed the intervention; independent confirmation required',
            'encoder/predictor/arrival rule fixed; no isolated JEPA representation-training benefit',
            'same goal readout as direct feedback, but different future versus current feature inputs'])
    pilot.pilot.base.save(RESULT,report)
    print(json.dumps(dict(summary=summary,cases=[{k:r[k] for k in ('case','scene','goal_reached','final_within_goal','disallowed_contact','final_xy_error_m','final_yaw_error_deg','first_latched_tick','false_arrival_latch')} for r in rows]),indent=2),flush=True)


if __name__=='__main__':main()
