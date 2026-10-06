"""Stage-2 recording contexts per mission (development; Andrew, 5 October 2026).

Andrew: "build the training set only from decisions with meaningful motion (exclude holds and latched-veto ticks), and
report how many patch-approach and patch-entry contexts we get per mission".

A decision (a planning record with a selection) is a usable context when, over its 700-ms window [t, t + 700 ms):
- the requested commands are not all hold, so there is meaningful motion; and
- no dispatch tick has reason CURRENT_OBSERVED_OBSTACLE_VETO or COMMAND_WINDOW_VETO_LATCHED.

Each usable context is binned by the body centre's signed distance to the nearest strip edge, as the stage-2 scorer
does (scripts/score_go2_dev_patch_edge_prediction_development.py): approach, entry, on_patch, exit or off_patch.

Missions with a contact stop are listed, but under the on-policy rule they contribute nothing ("a mission with any
contact contributes nothing").

Each mission's ending is classified:
- round trip;
- contact stop;
- stall stop, either a "strip trap" (the centre within 0.45 m of a wall over the stall window, with the dispatch veto
  latched on more than half its ticks) or "other stall";
- other failure.

Usage: count_go2_stage2_recording_contexts_development.py COHORT --json OUT
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from lewm.dev_dynamics_patches_v2_development import signed_edge_distance
from scripts.score_go2_dev_closed_loop_prediction_development import cohort_runs
from scripts.score_go2_dev_patch_edge_prediction_development import edge_bin

STEP_NS, HORIZON = 100_000_000, 7
VETOES = {'CURRENT_OBSERVED_OBSTACLE_VETO', 'COMMAND_WINDOW_VETO_LATCHED'}


def wall_distance(xy, boxes):
    best = np.inf
    for b in boxes:
        c = np.asarray(b['centre_xyz'][:2])
        hx, hy, a = b['size_xyz'][0]/2, b['size_xyz'][1]/2, b['yaw_rad']
        d = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])@(np.asarray(xy)-c)
        best = min(best, math.hypot(max(abs(d[0])-hx, 0), max(abs(d[1])-hy, 0)))
    return best


def mission(run):
    with np.load(run/'native/physics_trace.npz', allow_pickle=False) as z:
        stamps = np.rint(z['timestamp_s']*1e9).astype(np.int64)
        pose, requested = z['base_pose_world'].copy(), z['requested_command'].copy()
    rects = [p['rect'] for p in json.loads((run/'native/dynamics_patches.json').read_text())['placement']['patches']]
    boxes = json.loads((run/'specification.json').read_text())['geometry']['wall_boxes']
    requests = json.loads((run/'requests.json').read_text())
    vetoed = np.array([r['now_ns'] for r in requests if r.get('reason') in VETOES], np.int64)
    index = {t: i for i, t in enumerate(stamps)}
    bins, total, usable = Counter(), 0, 0
    for row in json.loads((run/'planning.json').read_text()):
        if 'selection' not in row:
            continue
        total += 1
        t = row['measured_ns']
        steps = [index.get(t+k*STEP_NS+STEP_NS//2) for k in range(HORIZON)]
        start, end = index.get(t), index.get(t+HORIZON*STEP_NS)
        if None in steps or start is None or end is None:
            continue
        if not np.any(np.abs(requested[steps]) > 1e-9):
            continue
        if np.any((vetoed >= t) & (vetoed < t+HORIZON*STEP_NS)):
            continue
        usable += 1
        bins[edge_bin(signed_edge_distance(pose[start, :2], rects), signed_edge_distance(pose[end, :2], rects))] += 1
    stall = (run/'native/dev_recording_stall_stop.json').exists()
    result_path = run/'result.json'
    failure = (run/'failure.json').exists() and json.loads((run/'failure.json').read_text()).get('reason', '')
    evaluation = run/'episode_evaluation.json'
    success = evaluation.exists() and json.loads(evaluation.read_text()).get('round_trip_success')
    if success:
        ending = 'round trip'
    elif failure and 'DISALLOWED_CONTACT' in failure:
        ending = 'contact stop'
    elif stall:
        window = stamps >= stamps[-1]-60_000_000_000
        near = np.mean([wall_distance(p[:2], boxes) < .45 for p in pose[window][::250]])
        latched = np.mean([r.get('reason') in VETOES for r in requests if r['now_ns'] >= stamps[-1]-60_000_000_000] or [0])
        ending = 'stall stop: strip trap' if near > .5 and latched > .5 else 'stall stop: other'
    else:
        ending = 'other failure'
    return dict(run=run.name, ending=ending, decisions=total, usable=usable, bins=dict(bins),
                contributes=ending != 'contact stop', simulated_s=float(stamps[-1]/1e9))


def main(name, out):
    rows = [mission(run) for _, run in cohort_runs(name)]
    keep = [r for r in rows if r['contributes']]
    total = Counter()
    for r in keep:
        total.update(r['bins'])
    report = dict(cohort=name, missions=len(rows), endings=dict(Counter(r['ending'] for r in rows)),
                  contributing_missions=len(keep), usable_contexts=sum(r['usable'] for r in keep),
                  contexts_by_bin=dict(total), per_mission=rows)
    if out:
        Path(out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'per_mission'}, indent=1))
    for r in rows:
        b = r['bins']
        print(f"{r['run'][-22:]:22s} {r['ending']:24s} {r['simulated_s']:5.0f} s  usable {r['usable']:4d}/{r['decisions']:4d}"
              f"  approach {b.get('approach', 0):3d}  entry {b.get('entry', 0):3d}  on_patch {b.get('on_patch', 0):4d}"
              f"  exit {b.get('exit', 0):3d}{'' if r['contributes'] else '  (excluded: contact)'}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--json')
    a = p.parse_args()
    main(a.name, a.json)
