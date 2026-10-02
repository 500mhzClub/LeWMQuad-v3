"""Closed-loop prediction accuracy of a driving controller's motion forecasts (development).

Andrew (30 Sep): judge the decoder by closed-loop prediction accuracy across movement types,
then by driving; closed-loop driving decides. This scores the forecasts a controller actually
made while driving, against physical truth, without replaying anything.

For each planning decision at observation time t:
- the forecast is the selected candidate's 700-ms XY displacement (the planner's scored
  horizon; `scoring_endpoint_offset_ns`) from the decision's logged forecast: the neural raw
  forecast for C3/C4, the command-history forecast for C1, or the degraded forecast in a
  forecast-sensitivity run;
- it is scored only if the requested commands actually executed over [t, t+700 ms), sampled
  at the middle of each 100-ms step from the physics trace, equal the committed prefix followed by the selected
  action's command (so the forecast describes what ran; vetoed or replaced plans are skipped);
- truth is the base displacement over the same 700 ms in the body frame at t, from the
  physics trace;
- the movement type comes from the executed 8-step tape and the 1.5-s command history, with the
  feature-cache rule (hold, rest start, turn, cruise, steady arc, switch).
Per type: n, median predicted/true translation (true >= 10 mm), median and RMS XY error.

Usage: score_go2_dev_closed_loop_prediction_development.py COHORT [COHORT ...] [--json OUT]
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from scripts.build_go2_dev_c3_feature_cache_development import category
from scripts.summarise_go2_dev_cohorts_development import BASE

HORIZON_STEPS, STEP_NS = 7, 100_000_000
CATEGORIES = ('hold', 'rest_start', 'turn', 'cruise', 'arc_steady', 'switch')


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def decisions(run):
    with np.load(run/'native/physics_trace.npz', allow_pickle=False) as z:
        stamps = np.rint(z['timestamp_s']*1e9).astype(np.int64)
        pose, requested = z['base_pose_world'].copy(), z['requested_command'].copy()

    def at(ns):
        i = int(np.searchsorted(stamps, ns))
        return i if i < len(stamps) and stamps[i] == ns else None

    out = []
    for row in json.loads((run/'planning.json').read_text()):
        if 'selection' not in row or 'motion_correction' not in row:
            continue
        t = row['measured_ns']
        # Commands are sampled mid-step: a plan dispatched at t+300 ms is not yet visible in the
        # physics sample at exactly t+300 ms.
        steps = [at(t+k*STEP_NS+STEP_NS//2) for k in range(-15, 8)]
        start_i, end = at(t), at(t+HORIZON_STEPS*STEP_NS)
        if any(s is None for s in steps) or start_i is None or end is None:
            continue
        executed = requested[steps[15:]]
        past = requested[steps[:15]]
        s, mc = row['selection'], row['motion_correction']
        command = candidate_commands(s['action'])[0]
        planned = np.asarray(list(row['committed_prefix'])+[command]*4, float)
        if not np.allclose(executed[:HORIZON_STEPS], planned[:HORIZON_STEPS], atol=1e-9):
            continue
        source = mc.get('prediction_source')
        if mc.get('dev_degraded_forecast_xy_yaw') is not None:  # forecast-sensitivity runs: the degraded forecast is what was scored
            forecast = np.asarray(mc['dev_degraded_forecast_xy_yaw'])[..., :2]
        elif source == 'command_history':
            forecast = np.asarray(mc['command_history_forecast_xy_yaw'])[..., :2]
        else:
            forecast = np.asarray(mc['raw_forecast_xy_m'])
        predicted = forecast[ACTIONS.index(s['action']), HORIZON_STEPS-1]
        start = pose[start_i]
        yaw = yaw_of(start[3:])
        d = pose[end][:2]-start[:2]
        true = np.array([math.cos(yaw)*d[0]+math.sin(yaw)*d[1], -math.sin(yaw)*d[0]+math.cos(yaw)*d[1]])
        out.append(dict(frame=row['frame'], category=category(executed, past), predicted=predicted.tolist(), true=true.tolist(),
                        source=source))
    return out


def metrics(rows):
    out = {}
    for c in ('all',)+CATEGORIES:
        rs = [r for r in rows if c == 'all' or r['category'] == c]
        if not rs:
            continue
        p, t = np.asarray([r['predicted'] for r in rs]), np.asarray([r['true'] for r in rs])
        pt, tt, e = np.linalg.norm(p, axis=1), np.linalg.norm(t, axis=1), np.linalg.norm(p-t, axis=1)
        moving = tt >= .010
        out[c] = dict(n=len(rs), median_ratio=float(np.median(pt[moving]/tt[moving])) if moving.any() else None,
                      median_xy_mm=float(np.median(e))*1000, rmse_xy_mm=float(np.sqrt(np.mean(e**2)))*1000)
    return out


def cohort_runs(name):
    config = json.loads((BASE/'dev_cohorts'/name/'config.json').read_text())
    return [(arm, BASE/'runs'/assignment) for arm, _s, _m, _e, assignment in config['plan'] if (BASE/'runs'/assignment/'planning.json').exists()]


def main(names, out):
    report = {}
    for name in names:
        by = defaultdict(list)
        scored = defaultdict(lambda: [0, 0])
        for arm, run in cohort_runs(name):
            rows = decisions(run)
            by[arm] += rows
            scored[arm][0] += len(rows)
            scored[arm][1] += sum(1 for r in json.loads((run/'planning.json').read_text()) if 'selection' in r)
        for arm, rows in sorted(by.items()):
            m = metrics(rows)
            report[f'{name}/{arm}'] = dict(scored_decisions=scored[arm][0], all_decisions=scored[arm][1], metrics=m)
            cells = '  '.join(f"{c}:{x['n']} r{'-' if x['median_ratio'] is None else f'{x['median_ratio']:.2f}'} e{x['median_xy_mm']:.0f}"
                              for c, x in m.items())
            print(f'{name:28s} {arm}  scored {scored[arm][0]}/{scored[arm][1]}  {cells}')
    if out:
        Path(out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='+')
    p.add_argument('--json')
    a = p.parse_args()
    main(a.names, a.json)
