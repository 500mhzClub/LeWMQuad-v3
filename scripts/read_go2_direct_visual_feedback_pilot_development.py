"""Compare actual outcomes, retaining unsuccessful feedback trajectories."""
import json
from pathlib import Path

from scripts import run_go2_direct_visual_feedback_pilot_development as pilot

RESULT = Path('docs/go2_direct_visual_feedback_pilot_result_2026-09-17.json')


def main():
    assert not RESULT.exists()
    plan = json.loads(pilot.PLAN.read_text())
    rows = []
    for case, path in enumerate(plan['case_directories']):
        directory = Path(path)
        assert not (directory/'failure.json').exists()
        result = json.loads((directory/'result.json').read_text())
        decisions = json.loads((directory/'decisions.json').read_text())
        observations = json.loads((directory/'observed_costs.json').read_text())
        assert result['status'] == 'COMPLETE' and result['observed_forecast_windows'] == 0
        assert len(decisions) == result['decisions']
        assert all(d['forecast_horizon_ms'] is None and d['costs'] is None for d in decisions)
        assert all('previous_selected_forecast_mse' not in r for r in observations)
        reference = pilot.previous.OUTPUT/f'case_{case:02d}'
        for frame in range(11):
            assert (directory/f'rgb_{frame:04d}.png').read_bytes() == (reference/f'rgb_{frame:04d}.png').read_bytes()
        first_latch = next((d['tick'] for d in decisions if d['arrival_latched']), None)
        if first_latch is not None:
            assert all(d['arrival_latched'] and d['action']=='hold' for d in decisions if d['tick']>=first_latch)
        longest = streak = 0
        for frame in result['camera_goal_errors'][10:]:
            streak = streak+1 if frame['within_goal'] else 0
            longest = max(longest,streak)
        row = {k:result[k] for k in ('case','scene','goal_reached','final_within_goal','completed_budget',
            'physical_stop','disallowed_contact','decisions','final_xy_error_m','final_yaw_error_deg','wall_s')}
        row.update(first_latched_tick=first_latch, maximum_consecutive_goal_frames=longest,
            actual_within_goal_at_latch=result['camera_goal_errors'][first_latch]['within_goal'] if first_latch is not None else None,
            actions=[d['action'] for d in decisions],initial_rgb_exact=True, observed_forecast_windows=0,
            result_path=str(directory/'result.json'),result_sha256=pilot.pilot.base.digest(directory/'result.json'))
        rows.append(row)
    old = json.loads(Path('docs/go2_dense_visual_arrival_pilot_result_2026-09-17.json').read_text())
    report = dict(status='COMPLETE',cases=rows,world_model_reference=old['cases'],
        world_model_reference_reused=True,plan_sha256=pilot.pilot.base.digest(pilot.PLAN),
        feedback_final_arrivals=sum(r['final_within_goal'] and r['completed_budget'] and not r['disallowed_contact'] for r in rows),
        world_model_final_arrivals=old['final_arrivals'],full_maze_navigation=False,real_time_qualified=False,
        limitations=['two exposed related tasks; feedback rule fixed before these runs',
            'same frozen JEPA encoder/readout in both controllers: no isolated JEPA-training benefit',
            'one feedback law and one predictor seed; not a general planner superiority result',
            'world-model reference reused, not two additional trials'])
    pilot.pilot.base.save(RESULT,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':main()
