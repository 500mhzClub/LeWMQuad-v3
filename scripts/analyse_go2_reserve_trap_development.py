"""PRELIMINARY: "can't translate out once inside the reserve" — wall-bound stalls, any controller (Andrew, 2 October 2026).

A stall is the longest span without any applied translating command, if it lasts at least 120 s.
Inside it, every 5 s, for the decision being executed:
(a) the centre's clearance to remembered walls (forecast controllers: the hold candidate's logged
    path clearance; C2: its logged current stored clearance) and to the true walls, and whether
    it is below the 0.48-m translation requirement (0.45-m disc + 0.03-m reserve) and below the
    0.45-m disc;
(b) whether any translation candidate (forward, left arc, right arc) would have increased that
    clearance. Forecast controllers: from each candidate's logged segment clearances (the check's
    own 8 distances); the frozen reserve-recovery rule would pass it only if its prefix (first
    three) is above 0.45 m, it never decreases after the prefix, and it ends above 0.48 m. C2 has
    no forecast: a 0.1-m step at the current heading is tested against the true walls (on the reserve_exit_v1
    harness, C2's nominal-path check rows are used as the forecast controllers' are);
plus whether the decision was in scan mode (translations are then excluded by the view
requirement, whatever the clearance).

Mechanism per sample:
- inside the disc: centre clearance <= 0.45 m, so no translation can pass the check (C2's rule then
  makes every action ineligible, turns included);
- inside the reserve, none increases: 0.45-0.48 m and every translation candidate loses clearance;
- inside the reserve, one increases but fails recovery (ends <= 0.48 m or dips after the prefix);
- scan mode: translations excluded by the view requirement;
- other.
- a translation passes the check but is not selected (holds or turns outscore it);
A "can't translate out once inside the reserve" trap is a stall dominated by the first three.
With a calibrated margin, the disc and the requirement are shifted out by the margin (the check subtracts it).

Usage: analyse_go2_reserve_trap_development.py --runs COHORT[:CONTROLLER[:MAZE]] ... [--markdown OUT] [--json OUT]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.analyse_go2_forecast_sensitivity_error_budget_development import planar_yaw
from scripts.diagnose_go2_forecast_sensitivity_failures_development import BASE, wall_distance

STALL_S, SAMPLE_S, DISC_M, REQUIRED_M = 120., 5., .45, .48
TRANSLATIONS = ('forward', 'left_arc', 'right_arc')
LABEL = 'PRELIMINARY (prelim_test_v1 mazes 30-49; development mode)'


def stall_span(t, translating):
    edges = np.flatnonzero(np.diff(np.concatenate(([1], translating.astype(int), [1]))))
    spans = [(t[min(a, len(t)-1)], t[min(b, len(t))-1]) for a, b in zip(edges[::2], edges[1::2])]
    best = max(spans, key=lambda s: s[1]-s[0], default=None)
    return best if best and best[1]-best[0] >= STALL_S else None


def recovery_ok(d, margin=0.):
    prefix = min(d[:3])
    return bool(DISC_M+margin < prefix <= REQUIRED_M+margin and min(d[3:]) >= prefix and d[7] > REQUIRED_M+margin)


def mission(assignment):
    root = BASE/'runs'/assignment
    read = lambda n: json.loads((root/n).read_text())
    arm = read('config.json')['controller']
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as f:
        ts, P, applied = f['timestamp_s'].copy(), f['base_pose_world'].copy(), f['applied_command'].copy()
    t = ts-1.5
    span = stall_span(t, np.any(applied[:, :2] != 0, axis=1))
    ev = read('episode_evaluation.json')
    row = dict(assignment=assignment, controller=arm, maze=read('episode.json')['maze_id'], round_trip=bool(ev['round_trip_success']), stall=None)
    if span is None:
        return row
    distance = wall_distance(read('specification.json')['geometry']['wall_boxes'])
    plans = sorted((r for r in read('planning.json') if 'selection' in r), key=lambda r: r['measured_ns'])
    samples, next_t = [], span[0]
    for plan in plans:
        tp = plan['measured_ns']/1e9-1.5
        if tp < next_t or tp > span[1]:
            continue
        next_t = tp+SAMPLE_S
        s = plan['selection']
        i = min(int(np.searchsorted(ts, plan['measured_ns']/1e9)), len(ts)-1)
        true = float(distance(P[i, :2])[0])
        scan = s.get('scan_heading_error_rad') is not None
        out = dict(time_s=round(tp, 1), action=plan['action'], scan=scan, true_centre_m=round(true, 3))
        check_rows = s.get('memory_forecast_candidates') or (s.get('c2_nominal_path_check') or {}).get('rows')
        if check_rows:  # forecast controllers; C2 on the reserve_exit_v1 harness (nominal command paths)
            memory = {c['action']: c for c in check_rows}
            margin = (s.get('dev_clearance_margin') or {}).get('margin_m') or 0.
            hold = memory['hold'].get('minimum_predicted_path_clearance_m')
            remembered = None if hold is None else hold+margin
            moves = []
            for a in TRANSLATIONS:
                d = memory[a].get('segment_clearances_m')
                if d is None or any(x is None for x in d):
                    continue
                d = [x+margin for x in d]
                moves.append(dict(action=a, prefix=round(min(d[:3]), 3), end=round(d[7], 3), increases=d[7] > min(d[:3])+1e-3,
                                  recovery_ok=recovery_ok(d, margin), clear=bool(memory[a]['nominal_predicted_path_clear'])))
            out.update(remembered_centre_m=None if remembered is None else round(remembered, 3), moves=moves, margin_m=margin)
            increases = any(m['increases'] for m in moves)
        else:  # C2: no forecast; test a 0.1-m step at the current heading against the true walls
            margin = 0.
            remembered = s.get('current_stored_clearance_m')
            yaw = planar_yaw(P[i])
            step = float(distance(P[i, :2]+.1*np.array([np.cos(yaw), np.sin(yaw)]))[0])
            increases = step > true+1e-3
            out.update(remembered_centre_m=None if remembered is None else round(remembered, 3), step_forward_true_m=round(step, 3))
        if 'c2_nominal_path_check' in s:  # C2 eligibility = its own rule AND the nominal-path check
            out['remembered_centre_m'] = remembered = s.get('current_stored_clearance_m')
            rule = {c['action']: c for c in s['candidates']}
            any_translation_clear = any(rule[a]['eligible'] for a in TRANSLATIONS)
        else:
            any_translation_clear = any(m.get('clear') for m in out.get('moves', [])) if 'moves' in out else bool(s.get('current_nominal_disk_clear'))
        # With a calibrated margin the check subtracts it from every remembered distance, so the disc and the
        # requirement move out by the margin (remembered is reported margin-free).
        if remembered is not None and remembered <= DISC_M+margin:
            mechanism = 'inside the disc'
        elif remembered is not None and remembered <= REQUIRED_M+margin and not any_translation_clear:
            mechanism = 'inside the reserve, a translation increases but fails recovery' if increases else 'inside the reserve, none increases'
        elif scan:
            mechanism = 'scan mode'
        elif any_translation_clear:
            mechanism = 'a translation passes the check but is not selected'
        else:
            mechanism = 'other'
        out.update(increases=bool(increases), mechanism=mechanism)
        samples.append(out)
    onset = samples[0] if samples else None
    row['stall'] = dict(begin_s=round(float(span[0]), 1), end_s=round(float(span[1]), 1), samples=samples, onset=onset,
                        mechanisms=dict(Counter(x['mechanism'] for x in samples)))
    return row


def table(rows):
    lines = [f'**Wall-bound stalls: can the robot translate out once inside the reserve? {LABEL}**', '',
             'Onset = first sample of the stall. Remembered centre clearance: forecast controllers, the hold candidate\'s logged path clearance; '
             'C2, its current stored clearance. Translation requirement 0.48 m (disc 0.45 m + reserve 0.03 m), plus any calibrated margin. Increases = a translation candidate '
             'ends with more clearance than its prefix (C2: a 0.1-m step at the current heading, true walls).', '',
             '| Run | Ctrl | Maze | Round trip | Stall (s) | Onset centre clearance: remembered · true (m) | Below 0.48 · below 0.45 at onset '
             '| Samples where some translation increases clearance | Scan-mode samples | Mechanism (samples) |',
             '|---|---|---:|---|---|---|---|---|---:|---|']
    for r in rows:
        s = r['stall']
        if not s:
            lines.append(f"| {r['cohort']} | {r['controller']} | {r['maze']} | {r['round_trip']} | none | - | - | - | - | - |")
            continue
        o, sm = s['onset'], s['samples']
        rem = o['remembered_centre_m'] if o else None
        lines.append(f"| {r['cohort']} | {r['controller']} | {r['maze']} | {r['round_trip']} | {s['begin_s']:.0f}-{s['end_s']:.0f} | "
                     f"{'-' if rem is None else f'{rem:.3f}'} · {o['true_centre_m']:.3f} | "
                     f"{'-' if rem is None else ('yes' if rem <= REQUIRED_M+o.get('margin_m', 0.) else 'no')} · {'-' if rem is None else ('yes' if rem <= DISC_M+o.get('margin_m', 0.) else 'no')} | "
                     f"{sum(x['increases'] for x in sm)}/{len(sm)} | {sum(x['scan'] for x in sm)} | "
                     f"{'; '.join(f'{k} {v}' for k, v in sorted(s['mechanisms'].items(), key=lambda kv: -kv[1]))} |")
    return '\n'.join(lines)


def main(specs, out_md, out_json):
    rows = []
    for spec in specs:
        cohort, *rest = spec.split(':')
        config = json.loads((BASE/'dev_cohorts'/cohort/'config.json').read_text())
        for job in config['plan']:
            if rest and job[0] != rest[0]:
                continue
            if len(rest) > 1 and job[2] != int(rest[1]):
                continue
            if (BASE/'runs'/job[4]/'episode_evaluation.json').exists():
                rows.append(dict(mission(job[4]), cohort=cohort))
    text = table(rows)
    print(text)
    if out_md:
        Path(out_md).write_text(text+'\n')
    if out_json:
        Path(out_json).write_text(json.dumps(dict(label=LABEL, missions=rows), indent=1)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--markdown')
    p.add_argument('--json')
    a = p.parse_args()
    main(a.runs, a.markdown, a.json)
