"""Summarize all four fixed live trials, retaining every failure and overshoot."""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.run_go2_dense_visual_goal_pilot_development import OUTPUT, PLAN, CASES

RESULT = Path('docs/go2_dense_visual_goal_pilot_result_2026-09-17.json')
FIGURE = Path('docs/go2_dense_visual_goal_pilot_2026-09-17.png')


def main():
    assert not RESULT.exists() and not FIGURE.exists()
    rows, raw = [], []
    for case, (scene, goal, arm) in enumerate(CASES):
        directory = OUTPUT/f'case_{case:02d}'
        assert not (directory/'failure.json').exists()
        r = json.loads((directory/'result.json').read_text())
        observations = json.loads((directory/'observed_costs.json').read_text())
        decisions = json.loads((directory/'decisions.json').read_text())
        assert r['status'] == 'COMPLETE' and r['arm'] == arm and r['scene'] == scene
        assert r['decisions'] == len(decisions)
        assert all(d['commit_ticks'] == 5 and d['forecast_horizon_ms'] == 500 for d in decisions)
        if arm == 'no_future_action':
            assert all(len(d['tied_indices']) == 6 for d in decisions)
        run = maximum = 0
        for g in r['camera_goal_errors'][10:]:
            run = run+1 if g['within_goal'] else 0
            maximum = max(run, maximum)
        assert r['goal_reached'] == bool(maximum >= 3 and not r['disallowed_contact'] and r['physical_stop'] is None)
        valid = [o for o in observations if 'previous_selected_forecast_mse' in o]
        fields = ('case', 'arm', 'scene', 'goal_reached', 'final_within_goal', 'completed_budget',
            'physical_stop', 'disallowed_contact', 'decisions', 'initial_xy_error_m', 'initial_yaw_error_deg',
            'minimum_xy_error_m', 'final_xy_error_m', 'final_yaw_error_deg', 'wall_s',
            'mean_planning_wall_s', 'mean_encoding_wall_s')
        row = {k:r[k] for k in fields}
        row.update(maximum_consecutive_goal_frames=maximum, forecast_windows=len(valid),
            factual_mse_own_executed_windows=float(np.mean([v['previous_selected_forecast_mse'] for v in valid])),
            persistence_mse_own_executed_windows=float(np.mean([v['previous_persistence_mse'] for v in valid])),
            result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest())
        rows.append(row)
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as archive:
            trajectory = archive['base_pose_world'][:, :2].copy()
        raw.append((r, trajectory, json.loads((directory/'specification.json').read_text())))
    initial_matches = []
    for a, b in ((0, 1), (2, 3)):
        same = all((OUTPUT/f'case_{a:02d}'/f'rgb_{f:04d}.png').read_bytes() ==
                   (OUTPUT/f'case_{b:02d}'/f'rgb_{f:04d}.png').read_bytes() for f in range(11))
        assert same
        initial_matches.append(same)
    summaries = {}
    for arm in ('action', 'no_future_action'):
        selected = [r for r in rows if r['arm'] == arm]
        summaries[arm] = dict(trials=len(selected), transient_goal_visits=sum(r['goal_reached'] for r in selected),
            final_within_goal=sum(r['final_within_goal'] for r in selected),
            disallowed_contacts=sum(r['disallowed_contact'] for r in selected),
            completed_budgets=sum(r['completed_budget'] for r in selected))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {'action':'#1765a3', 'no_future_action':'#bd621d'}
    for col, pair in enumerate(((0, 1), (3, 2))):
        ax, curve = axes[0, col], axes[1, col]
        r, _, spec = raw[pair[0]]
        for box in spec['geometry']['wall_boxes']:
            x,y,_ = box['centre_xyz']; sx,sy,_ = box['size_xyz']
            assert box['yaw_rad'] == 0
            ax.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,color='#cccccc',zorder=0))
        goal = r['goal_pose_evaluator_only']
        ax.scatter(*goal[:2],marker='*',s=130,color='#27843b',label='Supplied goal pose (evaluation)')
        for case in pair:
            result, xy, _ = raw[case]; arm = result['arm']; color = colors[arm]
            label = 'Action-conditioned' if arm == 'action' else 'Action-blind'
            ax.plot(xy[:,0],xy[:,1],color=color,label=label)
            ax.scatter(*xy[-1],color=color,marker='x',s=55)
            g = result['camera_goal_errors']
            curve.plot([v['time_s']-1.5 for v in g],[100*v['xy_error_m'] for v in g],color=color,label=label)
        ax.scatter(0,0,color='black',s=25,label='Start')
        ax.set(xlim=(-.15,1.6),ylim=(-.25,1.3),aspect='equal',xlabel='World X (m)',ylabel='World Y (m)',
               title=f'Exposed transfer geometry {col+1}')
        curve.axhline(3,color='#27843b',ls='--',lw=1,label='Position tolerance (heading also required)')
        curve.set(xlabel='Elapsed simulated time after settling (s)',ylabel='Distance to goal (cm)')
        curve.grid(alpha=.2)
    axes[0,0].legend(fontsize=8); axes[1,0].legend(fontsize=8)
    fig.suptitle('Live dense visual planning approaches the goal, then overshoots\nFixed four-trial development pilot; simulation pauses during inference')
    fig.tight_layout(); fig.savefig(FIGURE,dpi=160); plt.close(fig)
    report = dict(status='COMPLETE', summaries=summaries, cases=rows,
        plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(), paired_quiet_history_rgb_exactly_equal=initial_matches,
        figure=str(FIGURE), all_four_executions_terminal=True, full_maze_navigation=False,
        robust_goal_reaching_established=False, action_blind_control_is_not_strong_reactive_baseline=True,
        original_interface_failures_preserved='go2_dense_visual_goal_pilot_v1_attempt_001/terminal_failure_summary.json',
        interpretation='Action model approaches both goals and avoids contact, but neither run ends at the goal. One transient visit meets the predeclared three-frame criterion.',
        next_question='Does actual future-image goal cost prefer braking near the goal? Use matched native alternatives to separate predictor ranking from goal-cost failure.',
        limitations=['two exposed local layouts, same nominal target motion; not independent complete mazes',
            'supplied single goal image, no exploration/memory/backtracking experiment',
            'feature errors across arms use different realized trajectories and are not a matched accuracy comparison',
            'action inference plus encoding averages about 0.63 seconds per 0.5-second step; no real-time qualification'])
    with RESULT.open('x') as stream:
        json.dump(report,stream,indent=2); stream.write('\n')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
