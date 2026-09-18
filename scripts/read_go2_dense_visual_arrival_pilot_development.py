"""Evaluate observed-image arrival independently of the controller's estimates."""
import json
from pathlib import Path

import numpy as np

from scripts import run_go2_dense_visual_arrival_pilot_development as pilot

RESULT = Path('docs/go2_dense_visual_arrival_pilot_result_2026-09-17.json')


def main():
    assert not RESULT.exists()
    rows = []
    for case, reference_case in enumerate((0, 3)):
        directory = pilot.OUTPUT/f'case_{case:02d}'
        reference = pilot.previous.OUTPUT/f'case_{reference_case:02d}'
        assert not (directory/'failure.json').exists()
        result = json.loads((directory/'result.json').read_text())
        decisions = json.loads((directory/'decisions.json').read_text())
        old = json.loads((reference/'decisions.json').read_text())
        assert result['status'] == 'COMPLETE' and len(decisions) == result['decisions']
        latched = [d for d in decisions if d['arrival_latched']]
        first_tick = latched[0]['tick'] if latched else None
        for d, before in zip(decisions, old):
            if d['arrival_latched']:
                assert d['controller_mode'] == 'terminal_hold' and d['action'] == 'hold'
                assert d['requested_commands'] == [[0., 0., 0.]]*5
            else:
                assert d['action'] == before['action']
                assert d['requested_commands'] == before['requested_commands']
                np.testing.assert_allclose(d['costs'], before['costs'], rtol=0, atol=1e-6)
        if first_tick is not None:
            assert all(d['arrival_latched'] for d in decisions if d['tick'] >= first_tick)
        prefix_end = first_tick if first_tick is not None else 10
        for frame in range(prefix_end+1):
            assert (directory/f'rgb_{frame:04d}.png').read_bytes() == (reference/f'rgb_{frame:04d}.png').read_bytes()
        errors = result['camera_goal_errors']
        longest = streak = 0
        for frame in errors[10:]:
            streak = streak+1 if frame['within_goal'] else 0
            longest = max(longest, streak)
        row = {k:result[k] for k in ('case', 'scene', 'goal_reached', 'final_within_goal',
            'completed_budget', 'disallowed_contact', 'physical_stop', 'decisions',
            'final_xy_error_m', 'final_yaw_error_deg', 'wall_s')}
        row.update(first_latched_tick=first_tick, hold_decisions=len(latched),
            actual_within_goal_at_latch=errors[first_tick]['within_goal'] if latched else None,
            all_frames_within_after_latch=all(v['within_goal'] for v in errors[first_tick:]) if latched else None,
            maximum_consecutive_goal_frames=longest, matched_planning_prefix=True,
            exact_rgb_prefix_through_tick=prefix_end,
            result_path=str(directory/'result.json'), result_sha256=pilot.pilot.base.digest(directory/'result.json'))
        rows.append(row)
    report = dict(status='COMPLETE', cases=rows, plan_sha256=pilot.pilot.base.digest(pilot.PLAN),
        final_arrivals=sum(r['final_within_goal'] and r['completed_budget'] and not r['disallowed_contact'] for r in rows),
        full_maze_navigation=False, real_time_qualified=False, hardware_validated=False,
        limitations=['two repeatedly exposed related development tasks',
            'intervention motivated by these tasks; independent confirmation remains necessary',
            'learned goal recognition added; predictor unchanged',
            'no strong reactive baseline or isolated JEPA training advantage established'])
    pilot.pilot.base.save(RESULT, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
