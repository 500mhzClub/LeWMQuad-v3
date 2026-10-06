"""PRELIMINARY: a per-decision look at C3's recovery-off holds (Andrew, 1 October 2026).

For every decision where C3 (recovery off: its own choices on the frozen harness) selected
hold, from the planning log and the frozen reader's hold classification:
- predicted 700-ms travel (the planner's scored horizon) for forward and for hold: C3's own
  forecast (the neural raw forecast it scored with), and the command-history forecast the
  planner computes alongside it for every candidate (a kinematic reference);
- predicted minimum path clearance for forward and hold against the requirement (0.45-m
  nominal disk plus the translation prediction-error reserve);
- which check excluded each moving action: the view restriction (not in the scan action
  space), the forecast clearance gate, the planned-stopping projection, or none, in which
  case it was eligible and outscored by hold;
- the reader's hold category.

Usage: analyse_go2_prelim_c3_holds_development.py RUN [RUN ...] [--markdown OUT]
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS

H = 6  # 700 ms
MOVES = ACTIONS[1:]


def decisions(run):
    holds = {h['frame']: h for h in json.loads((run/'episode_evaluation.json').read_text())['hold_details']}
    out = []
    for r in json.loads((run/'planning.json').read_text()):
        if 'selection' not in r or r['action'] != 'hold':
            continue
        s, mc = r['selection'], r.get('motion_correction') or {}
        raw = np.asarray(mc.get('raw_forecast_xy_m')) if mc.get('raw_forecast_xy_m') is not None else None
        kin = np.asarray(mc['command_history_forecast_xy_yaw'])[..., :2] if mc.get('command_history_forecast_xy_yaw') is not None else None
        memory = {c['action']: c for c in s.get('memory_forecast_candidates', [])}
        scan = {c['action'] for c in s.get('scan_utilities') or []} if s.get('scan_utilities') else None
        stop = {c['action']: c.get('projection_clear') for c in (s.get('planned_stopping_projection') or {}).get('candidates', [])}
        utility = {c['action']: c['utility_m'] for c in s['candidates']}
        excluded = {}
        for a in MOVES:
            if scan is not None and a not in scan:
                excluded[a] = 'view restriction'
            elif a in memory and not memory[a].get('nominal_predicted_path_clear', False):
                excluded[a] = 'forecast clearance'
            elif stop.get(a) is False:
                excluded[a] = 'stopping projection'
            else:
                excluded[a] = 'eligible, outscored by hold' if utility.get(a, -1) <= utility.get('hold', 0) else 'eligible, other override'
        travel = lambda f, a: None if f is None else float(np.linalg.norm(f[ACTIONS.index(a), H]))
        out.append(dict(frame=r['frame'], category=holds.get(r['frame'], {}).get('category'),
                        override=holds.get(r['frame'], {}).get('override_reason'),
                        forward_c3_m=travel(raw, 'forward'), hold_c3_m=travel(raw, 'hold'),
                        forward_kinematic_m=travel(kin, 'forward'), hold_kinematic_m=travel(kin, 'hold'),
                        forward_clearance_m=memory.get('forward', {}).get('minimum_predicted_path_clearance_m'),
                        hold_clearance_m=memory.get('hold', {}).get('minimum_predicted_path_clearance_m'),
                        required_m=memory.get('forward', {}).get('required_path_clearance_m'),
                        hold_clear=memory.get('hold', {}).get('nominal_predicted_path_clear'),
                        excluded=excluded))
    return out


def stats(values):
    v = np.asarray([x for x in values if x is not None], float)
    return '-' if not len(v) else f'{np.median(v):.3f} ({np.percentile(v, 10):.3f}-{np.percentile(v, 90):.3f})'


def report(run, rows):
    lines = [f'### {run if isinstance(run, str) else run.name} (PRELIMINARY; C3 recovery off): {len(rows)} hold decisions', '',
             f"- Reader categories: {dict(Counter(r['category'] for r in rows))}; overrides: {dict(Counter(r['override'] for r in rows if r['override']))}",
             '- Predicted 700-ms travel, median (10th-90th percentile), m:',
             f"  - forward: C3 {stats(r['forward_c3_m'] for r in rows)}; command-history reference {stats(r['forward_kinematic_m'] for r in rows)}",
             f"  - hold: C3 {stats(r['hold_c3_m'] for r in rows)}; command-history reference {stats(r['hold_kinematic_m'] for r in rows)}",
             f"- Predicted minimum path clearance, m: forward {stats(r['forward_clearance_m'] for r in rows)}; hold {stats(r['hold_clearance_m'] for r in rows)}; "
             f"required {stats(r['required_m'] for r in rows)}; hold itself passes the gate at {sum(bool(r['hold_clear']) for r in rows)}/{len(rows)} decisions",
             '', '| Action | ' + ' | '.join(['view restriction', 'forecast clearance', 'stopping projection', 'eligible, outscored by hold', 'eligible, other override']) + ' |',
             '|---|---:|---:|---:|---:|---:|']
    for a in MOVES:
        c = Counter(r['excluded'][a] for r in rows)
        lines.append(f'| {a} | ' + ' | '.join(str(c.get(k, 0)) for k in ('view restriction', 'forecast clearance', 'stopping projection',
                                                                         'eligible, outscored by hold', 'eligible, other override')) + ' |')
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--markdown')
    p.add_argument('--pooled', action='store_true', help='also report all given runs pooled')
    a = p.parse_args()
    per = [(r, decisions(r)) for r in a.runs]
    text = '\n\n'.join(report(r, d) for r, d in per)
    if a.pooled:
        text += '\n\n'+report(f'All {len(per)} missions pooled', [x for _r, d in per for x in d])
    print(text)
    if a.markdown:
        Path(a.markdown).write_text(text+'\n')
