"""Stage-2 primary measure: closed-loop forecast error by distance to the patch edge (development; Andrew, 5 October 2026).

The decisions scored are those of scripts/score_go2_dev_closed_loop_prediction_development.py. A decision counts only
if the requested commands over [t, t+700 ms) equal the committed prefix followed by the selected command. Its forecast
is the selected candidate's 700-ms XY displacement, from the logged forecast:
- C1A's adapted forecast;
- otherwise C1's command-history forecast;
- otherwise C3/C4's raw neural forecast.
Truth is the physics-trace displacement in the body frame at t.

Each decision is binned by the body centre's signed distance d to the nearest patch boundary at t (negative inside),
and by whether d falls over the horizon (moving inward):
- on_patch: d <= -0.3 m (the feet are on the patch);
- entry: -0.3 < d <= 0.3 m, moving inward;
- exit: -0.3 < d <= 0.3 m, moving outward;
- approach: 0.3 < d <= 1.5 m, moving inward;
- off_patch: everything else.
There is also a 0.25-m distance curve over [-1.5, 1.5] m. Reported per cohort and controller, with the marked or
unmarked condition read from each run's native/dynamics_patches.json. v1 whole-cell patches are scored as cell squares.

Usage: score_go2_dev_patch_edge_prediction_development.py COHORT [COHORT ...] [--json OUT]
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from lewm.dev_dynamics_patches_v2_development import signed_edge_distance
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from scripts.build_go2_dev_c3_feature_cache_development import category
from scripts.score_go2_dev_closed_loop_prediction_development import HORIZON_STEPS, STEP_NS, cohort_runs, yaw_of
from scripts.summarise_go2_dev_cohorts_development import BASE

PITCH_M = 1.3
BINS = ('approach', 'entry', 'on_patch', 'exit', 'off_patch')
EDGES = np.arange(-1.5, 1.5001, .25)


def rects_of(run):
    path = run/'native/dynamics_patches.json'
    if not path.exists():
        return None, None
    record = json.loads(path.read_text())
    placement = record['placement']
    if 'patches' in placement:
        rects = [p['rect'] for p in placement['patches']]
    else:
        rects = [[x*PITCH_M-PITCH_M/2, x*PITCH_M+PITCH_M/2, y*PITCH_M-PITCH_M/2, y*PITCH_M+PITCH_M/2]
                 for x, y in placement['patch_cells']]
    return rects, bool(record['marked'])


def edge_bin(d0, d1):
    inward = d1 <= d0
    if d0 <= -.3:
        return 'on_patch'
    if d0 <= .3:
        return 'entry' if inward else 'exit'
    if d0 <= 1.5 and d1 < d0:
        return 'approach'
    return 'off_patch'


def decisions(run, rects):
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
        steps = [at(t+k*STEP_NS+STEP_NS//2) for k in range(-15, 8)]
        start_i, end = at(t), at(t+HORIZON_STEPS*STEP_NS)
        if any(s is None for s in steps) or start_i is None or end is None:
            continue
        executed, past = requested[steps[15:]], requested[steps[:15]]
        s, mc = row['selection'], row['motion_correction']
        planned = np.asarray(list(row['committed_prefix'])+[candidate_commands(s['action'])[0]]*4, float)
        if not np.allclose(executed[:HORIZON_STEPS], planned[:HORIZON_STEPS], atol=1e-9):
            continue
        if mc.get('dev_adaptive_forecast_xy_yaw') is not None:
            forecast = np.asarray(mc['dev_adaptive_forecast_xy_yaw'])[..., :2]
        elif mc.get('prediction_source') == 'command_history':
            forecast = np.asarray(mc['command_history_forecast_xy_yaw'])[..., :2]
        else:
            forecast = np.asarray(mc['raw_forecast_xy_m'])
        predicted = forecast[ACTIONS.index(s['action']), HORIZON_STEPS-1]
        start = pose[start_i]
        yaw = yaw_of(start[3:])
        d = pose[end][:2]-start[:2]
        true = np.array([math.cos(yaw)*d[0]+math.sin(yaw)*d[1], -math.sin(yaw)*d[0]+math.cos(yaw)*d[1]])
        d0, d1 = signed_edge_distance(start[:2], rects), signed_edge_distance(pose[end][:2], rects)
        out.append(dict(frame=row['frame'], movement=category(executed, past), predicted=predicted.tolist(),
                        true=true.tolist(), edge_m=d0, bin=edge_bin(d0, d1)))
    return out


def summary(rows, key):
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    for name, rs in groups.items():
        p, t = np.asarray([r['predicted'] for r in rs]), np.asarray([r['true'] for r in rs])
        pt, tt, e = np.linalg.norm(p, axis=1), np.linalg.norm(t, axis=1), np.linalg.norm(p-t, axis=1)
        moving = tt >= .010
        out[str(name)] = dict(n=len(rs), median_ratio=float(np.median(pt[moving]/tt[moving])) if moving.any() else None,
                              median_xy_mm=float(np.median(e))*1000, rmse_xy_mm=float(np.sqrt(np.mean(e**2)))*1000)
    return out


def main(names, out):
    report = {}
    for name in names:
        by = defaultdict(list)
        for arm, run in cohort_runs(name):
            rects, marked = rects_of(run)
            if rects is None:
                continue
            for r in decisions(run, rects):
                by[(arm, 'marked' if marked else 'unmarked')].append(r)
        for (arm, condition), rows in sorted(by.items()):
            for r in rows:
                i = int(np.clip(np.searchsorted(EDGES, r['edge_m'])-1, 0, len(EDGES)-2))
                r['distance_bin'] = f'{EDGES[i]:+.2f}..{EDGES[i+1]:+.2f}' if -1.5 <= r['edge_m'] <= 1.5 else 'beyond 1.5 m'
            report[f'{name}/{arm}/{condition}'] = dict(decisions=len(rows), by_edge_bin=summary(rows, 'bin'),
                                                       by_distance=summary(rows, 'distance_bin'),
                                                       on_patch_by_movement=summary([r for r in rows if r['bin'] == 'on_patch'], 'movement'))
            cells = '  '.join(f"{b}:{m['n']} e{m['median_xy_mm']:.0f} r{'-' if m['median_ratio'] is None else format(m['median_ratio'], '.2f')}"
                              for b in BINS if (m := report[f'{name}/{arm}/{condition}']['by_edge_bin'].get(b)))
            print(f'{name:26s} {arm:4s} {condition:8s} n={len(rows):5d}  {cells}')
    if out:
        Path(out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='+')
    p.add_argument('--json')
    a = p.parse_args()
    main(a.names, a.json)
