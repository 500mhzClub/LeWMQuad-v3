"""PRELIMINARY: does a calibrated margin make the route target collapse onto the robot? (route-versus-check clash;
Andrew, 3 October 2026)

Routing plans over the map at nominal clearance. The route-target lookahead
(`clearance_lookahead_development.clear_route_target`) and the action check both see remembered-wall distances
reduced by the margin. The lookahead walks the route points and stops at the first whose straight segment from the
robot falls below min(0.48 m, start clearance). Where the route runs with nominal clearance between 0.48 m and
0.48 m + margin, the walk stops at the robot's own cell: the target collapses onto the robot, every translation
overshoots it, and turns or holds score higher.

Route-following decisions are those with a lookahead receipt (route cells present, no view or scan heading). Per decision:
- collapsed target: the lookahead changed the target and it lies within 0.10 m of the robot;
- original target blocked only by the margin: the original target's logged (margin-reduced) segment clearance is
  below the requirement but would pass with the margin added back. An intermediate route point may also block; the
  route cells are not logged, so that case is reported as undetermined.
Reported for all route-following decisions, for those inside a stall (>= 120 s without translation), and for stall
samples labelled 'a translation passes the check but is not selected' by analyse_go2_reserve_trap_development.

Usage: diagnose_go2_margin_route_check_clash_development.py --cohorts margin_p95 ... [--markdown OUT] [--json OUT]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from scripts.analyse_go2_reserve_trap_development import mission as trap_mission
from scripts.diagnose_go2_forecast_sensitivity_failures_development import BASE

COLLAPSE_M, TURNS = .10, ('left_turn', 'right_turn')
NOT_SELECTED = 'a translation passes the check but is not selected'
LABEL = 'PRELIMINARY (prelim_test_v1 mazes 30-49; development mode)'


def mission(job):
    cohort, controller, maze, assignment = job
    root = BASE/'runs'/assignment
    ev = json.loads((root/'episode_evaluation.json').read_text())
    plans = sorted((r for r in json.loads((root/'planning.json').read_text()) if 'selection' in r), key=lambda r: r['measured_ns'])
    trap = trap_mission(assignment)
    stall = trap['stall']
    span = (stall['begin_s'], stall['end_s']) if stall else None
    not_selected = {x['time_s'] for x in (stall or {}).get('samples', []) if x['mechanism'] == NOT_SELECTED}
    rows = []
    for r in plans:
        if 'lookahead' not in r:
            continue
        s, look = r['selection'], r['lookahead']
        margin = (s.get('dev_clearance_margin') or {}).get('margin_m') or 0.
        waypoint = s.get('waypoint_body_xy_m')
        distance = None if waypoint is None else float(np.hypot(*waypoint[:2]))
        collapsed = bool(look.get('target_changed') and distance is not None and distance < COLLAPSE_M)
        original, required = look.get('original_shortcut_clearance_m'), look.get('required_shortcut_clearance_m')
        blocked_by_margin = None
        if look.get('target_changed') and original is not None and required is not None:
            blocked_by_margin = bool(original < required <= original+margin) if original < required else None
        t = round(r['measured_ns']/1e9-1.5, 1)
        memory = {c['action']: c for c in s.get('memory_forecast_candidates', [])}
        rows.append(dict(time_s=t, action=r['action'], collapsed=collapsed, waypoint_m=distance,
                         target_changed=bool(look.get('target_changed')), route_index=look.get('route_index'),
                         blocked_by_margin=blocked_by_margin, margin_m=margin,
                         translation_clear=any(memory.get(a, {}).get('nominal_predicted_path_clear') for a in ('forward', 'left_arc', 'right_arc')),
                         in_stall=bool(span and span[0] <= t <= span[1]),
                         not_selected_sample=any(abs(t-x) < .05 for x in not_selected)))
    flips = 0
    stall_actions = [x['action'] for x in rows if x['in_stall'] and x['action'] in TURNS]
    flips = sum(a != b for a, b in zip(stall_actions, stall_actions[1:]))
    return dict(cohort=cohort, controller=controller, maze=maze, assignment=assignment, round_trip=bool(ev['round_trip_success']),
                stall=None if not span else dict(begin_s=span[0], end_s=span[1], turn_decisions=len(stall_actions), turn_direction_flips=flips),
                decisions=rows)


def share(rows, key='collapsed'):
    return (sum(bool(x[key]) for x in rows)/len(rows), len(rows)) if rows else (None, 0)


def table(results):
    groups = {}
    for m in results:
        groups.setdefault((m['cohort'], m['controller']), []).append(m)
    pct = lambda v: '-' if v[0] is None else f'{100*v[0]:.1f}% of {v[1]}'
    lines = [f'**Route target collapsing onto the robot under a calibrated margin. {LABEL}**', '',
             f'Route-following decisions only. Collapsed = the lookahead changed the target and it lies within {100*COLLAPSE_M:.0f} cm of the robot. '
             'Blocked only by the margin = the original target fails the margin-reduced requirement but passes with the margin added back '
             '(share of collapsed decisions whose cause is determinable from the receipt).', '',
             '| Cohort | Ctrl | Margin (cm) | Collapsed, all route-following decisions | Collapsed, inside stalls | Collapsed, stall samples where a clear translation was not selected '
             '| Collapsed with a translation clear | Of collapsed: original blocked only by the margin · determinable | Turn-direction flips per stalled turn decision |',
             '|---|---|---:|---|---|---|---|---|---:|']
    for (cohort, controller), ms in sorted(groups.items()):
        rows = [x for m in ms for x in m['decisions']]
        stall = [x for x in rows if x['in_stall']]
        sample = [x for x in rows if x['not_selected_sample']]
        collapsed = [x for x in rows if x['collapsed']]
        determinable = [x for x in collapsed if x['blocked_by_margin'] is not None]
        margin = max((x['margin_m'] for x in rows), default=0.)
        turns = sum(m['stall']['turn_decisions'] for m in ms if m['stall'])
        flips = sum(m['stall']['turn_direction_flips'] for m in ms if m['stall'])
        lines.append(f"| {cohort} | {controller} | {100*margin:.2f} | {pct(share(rows))} | {pct(share(stall))} | {pct(share(sample))} | "
                     f"{pct(share(collapsed, 'translation_clear'))} | "
                     f"{sum(x['blocked_by_margin'] for x in determinable)} · {len(determinable)} of {len(collapsed)} | "
                     f"{'-' if not turns else f'{flips/turns:.2f}'} |")
    return '\n'.join(lines)


def main(cohorts, out_md, out_json, workers):
    jobs = []
    for cohort in cohorts:
        for job in json.loads((BASE/'dev_cohorts'/cohort/'config.json').read_text())['plan']:
            if (BASE/'runs'/job[4]/'episode_evaluation.json').exists():
                jobs.append((cohort, job[0], job[2], job[4]))
    with Pool(workers) as pool:
        results = pool.map(mission, jobs, chunksize=1)
    text = table(results)
    print(text)
    if out_md:
        Path(out_md).write_text(text+'\n')
    if out_json:
        Path(out_json).write_text(json.dumps(dict(label=LABEL, missions=results), indent=1)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cohorts', nargs='+', required=True)
    p.add_argument('--markdown')
    p.add_argument('--json')
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    main(a.cohorts, a.markdown, a.json, a.workers)
