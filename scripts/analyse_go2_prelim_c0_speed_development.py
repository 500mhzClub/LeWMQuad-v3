"""PRELIMINARY: why is C0 (oracle) slower to the beacon than C1? (Andrew, 1 October 2026)

On the same mazes and recovery setting, outbound leg only (start to the logged beacon arrival
frame), for each controller:
- time to beacon, actual path versus shortest path (the detour ratio = actual / shortest);
- decisions, the hold rate and the chosen move mix (forward, arcs, turns, hold);
- time in scan or view mode (the planner's view requests and frontier views, during which
  translations are excluded) and the fraction of decisions routed to a frontier rather than to
  the beacon cell;
- the mean commanded forward speed while translating, and zero-command time;
- recovery interventions.
Per maze, so one outlier cannot carry the mean, then paired differences (controller minus C1).

C0 forecasts the executed motion exactly (oracle), so any extra time against C1 comes from
what the selector does with perfect forecasts (its progress objective, clearance reserves and
routing), not from prediction accuracy.

Usage: analyse_go2_prelim_c0_speed_development.py [--controllers C0 C1] [--cohorts ...]
"""
import argparse
from collections import Counter, defaultdict
import json
import statistics as st

from scripts.report_go2_prelim_results_development import collect, BASE

MOVES = ('forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn', 'hold')


def outbound(assignment):
    run = BASE/'runs'/assignment
    ev = json.loads((run/'episode_evaluation.json').read_text())
    beacon = next((a for a in ev['arrivals'] if a['phase'] == 'OUTBOUND' and a['passed']), None)
    if beacon is None:
        return None
    end = beacon['frame']
    rows = [r for r in json.loads((run/'planning.json').read_text()) if 'selection' in r and r['frame'] <= end]
    requests = json.loads((run/'requests.json').read_text())
    end_ns = 1_500_000_000+end*100_000_000
    leg = [q for q in requests if q['now_ns'] <= end_ns]
    moving = [q for q in leg if any(q['requested_command'][:2])]
    actions = Counter(r['action'] for r in rows)
    scan = sum(1 for r in rows if r['selection'].get('scan_heading_error_rad') is not None)
    frontier = sum(1 for r in rows if r.get('route_status') == 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER')
    o = ev['outbound']
    return dict(time_s=o['elapsed_s'], path_m=o['actual_path_m'], shortest_m=o['shortest_path_m'],
                detour=o['actual_path_m']/o['shortest_path_m'], decisions=len(rows),
                mix={a: actions.get(a, 0)/len(rows) for a in MOVES}, scan=scan/len(rows), frontier=frontier/len(rows),
                zero_s=sum(1 for q in leg if not any(q['requested_command']))*0.02,
                forward_mps=st.mean(q['requested_command'][0] for q in moving) if moving else None,
                translating_s=len(moving)*0.02)


def main(controllers, cohorts):
    rows = collect(cohorts)
    by = defaultdict(dict)
    for r in rows:
        if r.get('round_trip') is not None:
            by[(r['recovery'], r['maze'])][r['controller']] = r
    a, b = controllers
    lines = [f'**Why is {a} slower to the beacon than {b}? PRELIMINARY (prelim_test_v1), outbound leg, same mazes**', '',
             '| Maze | Recovery | Ctrl | Time to beacon (s) | Path / shortest (m) | Detour | Decisions | Hold | Forward | Arcs | Turns | '
             'Scan/view mode | Routed to frontier | Translating (s) | Zero command (s) | Recovery events |', '|'+'---|'*16]
    diffs = defaultdict(list)
    for (rec, maze), v in sorted(by.items()):
        if a not in v or b not in v or not (v[a].get('beacon') and v[b].get('beacon')):
            continue
        stats = {}
        for c in (a, b):
            s = outbound(v[c]['assignment'])
            if s is None:
                break
            stats[c] = s
            events = v[c]['deadlock_escapes']+v[c]['stall_reroutes']+v[c]['backups']+v[c]['latch_timeouts']+v[c]['terminal_spin_breaks']
            m = s['mix']
            lines.append(f"| {maze} | {rec} | {c}{'*' if c == 'C0' else ''} | {s['time_s']:.1f} | {s['path_m']:.2f} / {s['shortest_m']:.2f} | {s['detour']:.2f} | "
                         f"{s['decisions']} | {m['hold']:.2f} | {m['forward']:.2f} | {m['left_arc']+m['right_arc']:.2f} | {m['left_turn']+m['right_turn']:.2f} | "
                         f"{s['scan']:.2f} | {s['frontier']:.2f} | {s['translating_s']:.1f} | {s['zero_s']:.1f} | {events} |")
        if len(stats) == 2:
            for key in ('time_s', 'detour', 'decisions', 'scan', 'frontier', 'translating_s', 'zero_s'):
                diffs[key].append(stats[a][key]-stats[b][key])
            for move in MOVES:
                diffs['mix_'+move].append(stats[a]['mix'][move]-stats[b]['mix'][move])
    n = len(diffs['time_s'])
    lines += ['', f'Paired differences, {a} minus {b}, over {n} mazes where both reached the beacon (mean; median):', '']
    for key, values in diffs.items():
        lines.append(f'- {key}: {st.mean(values):+.3f}; {st.median(values):+.3f}')
    print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--controllers', nargs=2, default=('C0', 'C1'))
    p.add_argument('--cohorts', nargs='+', default=('prelim_trial', 'prelim_on', 'prelim_off'))
    a = p.parse_args()
    main(a.controllers, a.cohorts)
