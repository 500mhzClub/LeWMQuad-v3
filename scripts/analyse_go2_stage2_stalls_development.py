"""PRELIMINARY: stage-2 stalls under the pessimistic-unknown rule, and whether their blocking cells were ever observable (Andrew, 2 October 2026).

For each mission of a stage-2 cohort (look-around exemption; unseen cells block within 0.425 m
+ the controller's e_f bound of the forecast centre path):
- Stall: the longest span without any applied translating command, if it lasts at least 120 s.
  Start stall: no translation before it began; later stall otherwise.
- What blocked translation, at decisions inside the stall (every 5 s) where every translation
  candidate was blocked: per candidate, the logged effective clearance against the forecast path's
  true-wall clearance (remembered walls sit about 1.5 cm closer) and its clearance to reconstructed
  unseen cells plus the rule's offset. Wall-bound if the logged value matches the wall within 2 cm
  of that bias, otherwise unseen-bound (the live map's unseen set is not logged, so an unseen-bound
  decision may not be reproduced by the reconstruction).
- Blocking cells, for unseen-bound decisions the reconstruction reproduces: never-observed 5-cm cells, neither seen by the depth cameras so far (ray-cast
  reconstruction along the true trajectory, validated against the logged remembered clearance)
  nor seeded by the rule (the 0.5-m start disc, counting a cell only if its whole square lies
  inside; the traversed track, cells within 0.20 m of past positions), lying within the blocking
  radius of any translation candidate's forecast centre path placed at the true pose.
- Observable without translating: whether either depth camera could see each blocking cell from
  the stall position at any heading (72 headings, 5 degrees apart, the body's current height,
  pitch and roll), inside its frustum and 0.2-5 m optical depth and not hidden behind a wall (ray
  cast to the cell centre). Cells that are never observable this way can only be cleared by
  translating, which the rule forbids.

Usage: analyse_go2_stage2_stalls_development.py --cohort stage2_c1_lookaround_p95 [--markdown OUT] [--json OUT] [--workers 2]
"""
import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from lewm.auxiliary_downward45_depth_observation_development import body_from_optical as auxiliary_from_optical
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL, HEIGHT, MAX_DEPTH_M, MIN_DEPTH_M, WIDTH
from lewm.physical_execution_development import rotation_xyzw
from scripts.analyse_go2_forecast_sensitivity_error_budget_development import PASSED, planar_yaw, rot
from scripts.analyse_go2_unknown_cells_development import CELL_M, Scene
from scripts.diagnose_go2_forecast_sensitivity_failures_development import ACTIONS, BASE, assignments

STALL_S, SAMPLE_S = 120., 5.
START_CLEAR_M, TRACK_M, BODY_REACH_M = .50, .20, .425
CHECK_REQUIRED_M, MAP_BIAS_M, MATCH_M = .48, .015, .02  # translation requirement; remembered walls sit ~1.5 cm closer than true
TRANSLATIONS = ('forward', 'left_arc', 'right_arc')
YAWS = np.radians(np.arange(0, 360, 5))
CAMERAS = tuple(np.asarray(T, float) for T in (BODY_FROM_OPTICAL, auxiliary_from_optical()))
BOUNDS = json.loads((Path(__file__).resolve().parents[1]/'docs/go2_navigation_calibrated_margins_2026-10-02.json').read_text())['controllers']
LABEL = 'PRELIMINARY (prelim_test_v1 mazes 30-49, recovery off, coverage-rule fix; development mode)'


def observable(scene, pose, point):
    """Can either depth camera see the world point from this position at some heading, unoccluded?"""
    tilt = rotation_xyzw(pose[3:])
    current = planar_yaw(pose)
    for yaw in YAWS:
        c, s = math.cos(yaw-current), math.sin(yaw-current)
        Q = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])@tilt
        for T in CAMERAS:
            origin = pose[:3]+Q@T[:3, 3]
            x, y, z = T[:3, :3].T@(Q.T@(point-origin))
            if z <= 0 or not MIN_DEPTH_M <= z <= MAX_DEPTH_M or abs(FOCAL*x/z) > WIDTH/2 or abs(FOCAL*y/z) > HEIGHT/2:
                continue
            first, _ = scene.cast(origin, ((point-origin)/z)[None, :])
            if first[0] >= z-.01:
                return True
    return False


def mission(args):
    cohort, assignment = args
    root = BASE/'runs'/assignment
    read = lambda n: json.loads((root/n).read_text())
    config, spec = read('config.json'), read('specification.json')
    arm = config['controller']
    dev = read('dev_run.json')
    radius = (dev.get('pessimistic_unknown') or {}).get('blocking_radius_m') or BODY_REACH_M+BOUNDS[arm]['p95']['margin_m']
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as f:
        ts, P, applied = f['timestamp_s'].copy(), f['base_pose_world'].copy(), f['applied_command'].copy()
    t = ts-1.5
    translating = np.any(applied[:, :2] != 0, axis=1)
    # Longest span without translation.
    edges = np.flatnonzero(np.diff(np.concatenate(([1], translating.astype(int), [1]))))
    spans = [(t[min(a, len(t)-1)], t[min(b, len(t))-1]) for a, b in zip(edges[::2], edges[1::2])] if len(edges) else []
    spans = [(a, b) for a, b in spans if b > a]
    first_translation = float(t[np.argmax(translating)]) if translating.any() else None
    ev = read('episode_evaluation.json')
    row = dict(assignment=assignment, maze=read('episode.json')['maze_id'], round_trip=bool(ev['round_trip_success']), radius_m=radius,
               first_translation_s=first_translation, stall=None)
    longest = max(spans, key=lambda s: s[1]-s[0], default=None)
    if longest is None or longest[1]-longest[0] < STALL_S:
        return row
    begin, end = longest
    kind = 'start' if first_translation is None or first_translation >= begin else 'later'
    at = lambda s: min(int(np.searchsorted(ts, s)), len(ts)-1)
    plans = sorted((r for r in read('planning.json') if 'selection' in r), key=lambda r: r['measured_ns'])
    scene = Scene(spec['geometry']['wall_boxes'])
    rule_seed = np.zeros(scene.shape, bool)
    half_diagonal = CELL_M*math.sqrt(2)/2
    samples, frame_i = [], 0
    targets = list(np.arange(begin+5, end, SAMPLE_S))
    for plan in plans:
        tp = plan['measured_ns']/1e9-1.5
        while frame_i <= plan['frame'] and 1.5+frame_i*.1 <= ts[-1]:
            pose = P[at(1.5+frame_i*.1)]
            if frame_i == 0:
                scene.seed(pose[:2], START_CLEAR_M-half_diagonal, rule_seed)
            scene.seed(pose[:2], TRACK_M, rule_seed)
            scene.observe(pose)
            frame_i += 1
        if not targets or tp < targets[0]:
            continue
        targets.pop(0)
        memory = {c['action']: c for c in plan['selection'].get('memory_forecast_candidates') or []}
        if any(memory.get(a, {}).get('nominal_predicted_path_clear') for a in TRANSLATIONS):
            continue
        mc = plan.get('motion_correction') or {}
        forecast = np.asarray(mc.get('applied_prediction_after_yaw_ablation') or mc['command_history_forecast_xy_yaw'], float)
        i = at(plan['measured_ns']/1e9)
        yaw = planar_yaw(P[i])
        unseen_mask = ~scene.observed & ~rule_seed
        squares = scene.origin+np.argwhere(unseen_mask)*CELL_M
        offset = CHECK_REQUIRED_M-radius
        candidates, blocking = [], set()
        for a in TRANSLATIONS:
            path = P[i, :2]+np.vstack((np.zeros(2), forecast[ACTIONS.index(a), :, :2]))@rot(yaw).T
            dense = np.vstack([p0+(p1-p0)*np.linspace(0, 1, 8)[:, None] for p0, p1 in zip(path[:-1], path[1:])])
            wall = float(scene.distance(dense).min())
            gaps = np.linalg.norm(np.maximum(np.maximum(squares[None]-dense[:, None], dense[:, None]-(squares[None]+CELL_M)), 0.), axis=-1).min(axis=0) \
                if len(squares) else np.array([])
            unseen = float(gaps.min()) if len(gaps) else None
            logged = memory[a]['minimum_predicted_path_clearance_m']
            binding = 'remembered wall' if logged is not None and logged >= wall-MAP_BIAS_M-MATCH_M else 'unseen (live map)'
            reproduced = bool(unseen is not None and unseen+offset < CHECK_REQUIRED_M)
            if binding != 'remembered wall' and reproduced:
                blocking.update(map(tuple, np.argwhere(unseen_mask)[gaps < radius].tolist()))
            candidates.append(dict(action=a, logged_m=logged, true_wall_m=round(wall, 4), unseen_plus_offset_m=None if unseen is None else round(unseen+offset, 4),
                                   binding=binding, unseen_reproduced=reproduced))
        seen = []
        for c in blocking:
            point = np.append(scene.origin+(np.array(c)+.5)*CELL_M, 0.)
            heights = (0.,) if not scene.occupied[c] else (.05, .15, .30, .45, .60)
            seen.append(any(observable(scene, P[i], np.array([point[0], point[1], h])) for h in heights))
        samples.append(dict(time_s=round(tp, 1), centre_from_start_m=round(float(np.linalg.norm(P[i, :2]-P[at(1.5), :2])), 3), candidates=candidates,
                            binding=('remembered wall' if all(c['binding'] == 'remembered wall' for c in candidates) else 'unseen (live map)'),
                            blocking_cells=len(blocking), observable_cells=int(sum(seen))))
    row['stall'] = dict(kind=kind, begin_s=round(float(begin), 1), end_s=round(float(end), 1), duration_s=round(float(end-begin), 1), samples=samples)
    return row


def table(rows):
    stalls = [r for r in rows if r['stall']]
    lines = [f'**Stage-2 stalls and whether their blocking cells were ever observable without translating. {LABEL}**', '',
             f"{len(rows)} missions; {len(stalls)} stalled (a span of at least {STALL_S:.0f} s without translation): "
             f"{sum(r['stall']['kind'] == 'start' for r in stalls)} at the start, {sum(r['stall']['kind'] == 'later' for r in stalls)} later. "
             f"Round trips {sum(r['round_trip'] for r in rows)}/{len(rows)}.", '',
             '| Maze | Round trip | Stall | Span (s) | Sampled stalled decisions | Bound by remembered wall · by unseen cells | Unseen-bound reproduced by reconstruction '
             '| Reproduced blocking cells observable from the stall position | Centre from start (m) |',
             '|---:|---|---|---|---:|---|---|---|---:|']
    for r in sorted(rows, key=lambda r: r['maze']):
        s = r['stall']
        if not s:
            lines.append(f"| {r['maze']} | {r['round_trip']} | none | - | - | - | - | - | - |")
            continue
        sm = s['samples']
        unseen = [x for x in sm if x['binding'] != 'remembered wall']
        reproduced = [x for x in unseen if any(c['unseen_reproduced'] for c in x['candidates'] if c['binding'] != 'remembered wall')]
        cells = sum(x['blocking_cells'] for x in sm)
        lines.append(f"| {r['maze']} | {r['round_trip']} | {s['kind']} | {s['begin_s']:.0f}-{s['end_s']:.0f} | {len(sm)} | "
                     f"{len(sm)-len(unseen)} · {len(unseen)} | {len(reproduced)}/{len(unseen)} | {sum(x['observable_cells'] for x in sm)}/{cells} | "
                     f"{max((x['centre_from_start_m'] for x in sm), default=float('nan')):.2f} |")
    return '\n'.join(lines)


def main(cohort, out_md, out_json, workers):
    with Pool(workers) as pool:
        rows = pool.map(mission, assignments(cohort), chunksize=1)
    text = table(rows)
    print(text)
    if out_md:
        Path(out_md).write_text(text+'\n')
    if out_json:
        Path(out_json).write_text(json.dumps(dict(label=LABEL, missions=rows), indent=1)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cohort', required=True)
    p.add_argument('--markdown')
    p.add_argument('--json')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    main(a.cohort, a.markdown, a.json, a.workers)
