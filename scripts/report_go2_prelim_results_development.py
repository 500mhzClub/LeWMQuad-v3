"""PRELIMINARY results for the preliminary-test run (Andrew, 1 October 2026).

Everything here is labelled preliminary: prelim_test_v1 is the declassified former sealed set,
and the rigorous phase uses sealed_test_v2. Reads finished missions from the given cohorts
(recovery on or off as recorded in each cohort's config) and reports:
- per controller and recovery setting: missions, round-trip success, SPL, times, recovery counts
  per mission, stall (hold) rates, contacts, hard/operating violations and minimum clearance;
- per mission: the same with every recovery intervention and pose-correction count;
- recovery on versus off on the mazes run both ways (paired, per controller), with the
  missions whose outcome flips.
SPL: per leg from the frozen reader; round trip = success x (shortest outbound + return) /
max(actual outbound + return, shortest), and 0 for a failed round trip.

Usage: report_go2_prelim_results_development.py COHORT [COHORT ...] [--markdown OUT] [--json OUT]
"""
import argparse
from collections import defaultdict
import json
import statistics as st
from pathlib import Path

from scripts.summarise_go2_dev_cohorts_development import BASE, mission_row

LABEL = 'PRELIMINARY (prelim_test_v1, development mode; not a sealed-set result)'
C0_NOTE = ("C0*: run on a copy of the owner's harness in which only the C0 maze-ID check is relaxed to admit the "
           'preliminary-test IDs (the frozen owner limits C0 to IDs below 20); every other line is the owner\'s.')


def mission(assignment, controller, recovery, maze):
    row = mission_row(assignment, controller) | dict(recovery=recovery, maze=maze)
    path = BASE/'runs'/assignment/'episode_evaluation.json'
    if not path.exists() or row.get('round_trip') is None:
        return row
    ev = json.loads(path.read_text())
    out, ret = ev.get('outbound') or {}, ev.get('return_leg') or {}
    shortest = (out.get('shortest_path_m') or 0)+(ret.get('shortest_path_m') or 0)
    actual = (out.get('actual_path_m') or 0)+(ret.get('actual_path_m') or 0)
    stall = ev['stall_by_phase']
    row.update(outbound_spl=out.get('spl'), return_spl=ret.get('spl'),
               round_trip_spl=(shortest/max(actual, shortest) if row['round_trip'] and shortest else 0.),
               outbound_s=out.get('elapsed_s'), return_s=ret.get('elapsed_s'),
               total_s=(out.get('elapsed_s') or 0)+(ret.get('elapsed_s') or 0) if row['round_trip'] else None,
               outbound_hold_rate=stall.get('OUTBOUND', {}).get('rate'), return_hold_rate=stall.get('RETURN', {}).get('rate'))
    return row


def collect(names):
    rows = []
    for name in names:
        config = json.loads((BASE/'dev_cohorts'/name/'config.json').read_text())
        recovery = config.get('recovery') or ('on' if 'terminal' in (config.get('fixes') or []) else 'off')
        for arm, _set, maze, _episode, assignment in config['plan']:
            if (BASE/'runs'/assignment).exists():
                rows.append(mission(assignment, arm, recovery, maze) | dict(cohort=name))
    return rows


def f(value, digits=2):
    return '-' if value is None else f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def summary(rows):
    lines = [f'**{LABEL}**', '', '| Ctrl | Recovery | Read | Round trips | Success | SPL | Median time (s, successes) | Missions with recovery | '
             'Deadlock escapes / mission | Stall reroutes / mission | Back-ups / mission | Latch timeouts / mission | Spin breaks / mission | '
             'Outbound hold rate | Return hold rate | Contacts | Hard | Operating | Min clearance (m) |', '|'+'---|'*19]
    groups = defaultdict(list)
    for r in rows:
        groups[(r['controller'], r['recovery'])].append(r)
    for (ctrl, rec), rs in sorted(groups.items()):
        read = [r for r in rs if r.get('round_trip') is not None]
        if not read:
            continue
        n = len(read)
        wins = [r for r in read if r['round_trip']]
        times = [r['total_s'] for r in wins if r.get('total_s')]
        helped = sum(1 for r in read if r['deadlock_escapes'] or r['stall_reroutes'] or r['backups'] or r['latch_timeouts'] or r['terminal_spin_breaks'])
        per = lambda key: f(sum(r[key] for r in read)/n)
        clear = [r['min_clearance_m'] for r in read if r.get('min_clearance_m') is not None]
        lines.append(f"| {ctrl}{'*' if ctrl == 'C0' else ''} | {rec} | {n} | {len(wins)} | {len(wins)/n:.2f} | {f(mean([r.get('round_trip_spl') for r in read]))} | "
                     f"{f(st.median(times), 0) if times else '-'} | {helped} | {per('deadlock_escapes')} | {per('stall_reroutes')} | {per('backups')} | "
                     f"{per('latch_timeouts')} | {per('terminal_spin_breaks')} | {f(mean([r.get('outbound_hold_rate') for r in read]), 3)} | "
                     f"{f(mean([r.get('return_hold_rate') for r in read]), 3)} | {sum(r.get('contacts') or 0 for r in read)} | "
                     f"{sum(r.get('hard') or 0 for r in read)} | {sum(r.get('operating') or 0 for r in read)} | {f(min(clear) if clear else None, 3)} |")
    if any(r['controller'] == 'C0' for r in rows):
        lines += ['', C0_NOTE]
    return '\n'.join(lines)


def paired(rows):
    by = defaultdict(dict)
    for r in rows:
        if r.get('round_trip') is not None:
            by[(r['controller'], r['maze'])][r['recovery']] = r
    lines = ['', f'**Recovery on versus off, same mazes. {LABEL}**', '',
             '| Ctrl | Paired mazes | Success on | Success off | SPL on | SPL off | Only on succeeds | Only off succeeds | Contacts on | Contacts off |',
             '|'+'---|'*10]
    ctrls = sorted({c for c, _ in by})
    for ctrl in ctrls:
        pairs = [v for (c, _m), v in sorted(by.items()) if c == ctrl and 'on' in v and 'off' in v]
        if not pairs:
            continue
        on = [p['on'] for p in pairs]
        off = [p['off'] for p in pairs]
        only_on = [p['on']['maze'] for p in pairs if p['on']['round_trip'] and not p['off']['round_trip']]
        only_off = [p['on']['maze'] for p in pairs if p['off']['round_trip'] and not p['on']['round_trip']]
        lines.append(f"| {ctrl} | {len(pairs)} | {sum(bool(r['round_trip']) for r in on)} | {sum(bool(r['round_trip']) for r in off)} | "
                     f"{f(mean([r.get('round_trip_spl') for r in on]))} | {f(mean([r.get('round_trip_spl') for r in off]))} | "
                     f"{only_on or '-'} | {only_off or '-'} | {sum(r.get('contacts') or 0 for r in on)} | {sum(r.get('contacts') or 0 for r in off)} |")
    return '\n'.join(lines)


def per_mission(rows):
    lines = ['', f'**Per mission. {LABEL}**', '', '| Maze | Ctrl | Recovery | Round trip | SPL | Time (s) | Deadlock escapes | Stall reroutes | '
             'Back-ups | Latch timeouts | Spin breaks | Pose corrections | Holds out/ret | Contacts | Min clearance (m) |', '|'+'---|'*15]
    for r in sorted(rows, key=lambda r: (r['maze'], r['controller'], r['recovery'])):
        lines.append(f"| {r['maze']} | {r['controller']}{'*' if r['controller'] == 'C0' else ''} | {r['recovery']} | {f(r.get('round_trip'))} | {f(r.get('round_trip_spl'))} | "
                     f"{f(r.get('total_s'), 0)} | {r['deadlock_escapes']} | {r['stall_reroutes']} | {r['backups']} | {r['latch_timeouts']} | "
                     f"{r['terminal_spin_breaks']} | {f(r.get('pose_corrections'))} | {f(r.get('outbound_hold_rate'), 2)}/{f(r.get('return_hold_rate'), 2)} | "
                     f"{f(r.get('contacts'))} | {f(r.get('min_clearance_m'), 3)} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='+')
    p.add_argument('--markdown')
    p.add_argument('--json')
    p.add_argument('--no-missions', action='store_true')
    a = p.parse_args()
    rows = collect(a.names)
    text = summary(rows)+'\n'+paired(rows)+('' if a.no_missions else '\n'+per_mission(rows))
    print(text)
    if a.markdown:
        Path(a.markdown).write_text(text+'\n')
    if a.json:
        Path(a.json).write_text(json.dumps(dict(label=LABEL, rows=rows), indent=1, default=str))
