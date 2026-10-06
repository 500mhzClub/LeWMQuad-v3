"""PRELIMINARY: closest approaches to walls and whether the depth stop could see them (Andrew, 2 October 2026).

For the forecast-sensitivity safety test (uniform scale 0.25x and 0.5x, under-prediction),
with every other cohort for comparison. Read-only over each mission's preserved records.

Approaches: per mission, the 5 smallest native separations (the frozen reader's 500-Hz
lower bound) while moving (a non-zero applied command), each at least 1 s from the others.

At each approach:
- Closest body part: the reader's articulated collision model (27 URDF primitives, the
  frozen V4 safety evaluator's support computation) gives every primitive's separation from
  every wall box; the closest primitive and wall are taken, and checked against the logged
  separation. Leg = any FL/FR/RL/RR link, body = base and head. Front / side / rear = the
  direction from the base centre to the nearest wall point in the body frame (front within
  45 degrees of straight ahead, rear within 45 degrees of straight behind, side otherwise).
- Executing move: from the applied command (forward, arc, turn in place, other).
- In the depth camera's view: the depth stop uses only the current images of the two
  forward depth cameras (primary: 78 x 63 degrees, level, 0.33 m ahead of the base centre;
  auxiliary: same lens pitched 45 degrees down, 0.35 m ahead), valid optical depth 0.2-5 m,
  points 0.03-0.65 m above the floor. The nearest wall point is in view if the wall at that
  point is inside either camera's frustum and depth range at any height in that band
  (0.05, 0.15, 0.30, 0.45, 0.60 m). Out of view means outside the field of view or closer
  than the 0.2-m minimum depth; occlusion is not modelled (walls are the nearest surfaces).
  Out-of-view approaches are ones the depth stop structurally cannot catch.

Planner disc check (Andrew: what does the remembered-map check test?). Every planner-stage
clearance filter in the frozen V4 chain (memory forecast clearance, turn and translation
reserves, stopping projection, recovery modes) tests a 0.45-m disc centred on the base, swept
along the forecast's centre positions only (heading unused), against remembered fine
obstacle cells; turns and translations need 0.48 m (a 0.03-m reserve). It is not the
articulated swept volume. At each approach: the base centre's distance to the true walls
(disc held in truth if >= 0.45 m), the closest part's reach toward the wall (centre
distance minus separation; a part reaching beyond 0.45 m sticks out of the disc), and the
remembered clearance the planner computed for the executing move's forecast path. Reach toward
the wall is the exact support of all primitives along the centre-to-nearest-wall-point direction
(the centre distance minus the reader's separation overstates it near wall edges and corners,
because that separation is a lower bound). Over the
whole mission (10 Hz, moving samples), the articulated body's horizontal reach from the base
centre (support of all 27 primitives in 16 horizontal directions): how often and how far any
part extends beyond 0.45 m and 0.48 m, by move.

Clearance loss, per cohort, against this experiment's clean baseline: each approach's
deficit is how far it falls below the clean baseline's median approach separation; the share
of the summed deficit, and of approaches closer than the clean baseline's 10th percentile,
that is out of view. Thresholds fixed before reading any scale-cohort approach.

Usage: analyse_go2_forecast_sensitivity_close_approaches_development.py [--cohorts sens_...] [--json OUT] [--markdown OUT] [--workers 8]
"""
import argparse
from collections import Counter
import json
import math
from multiprocessing import Pool
from pathlib import Path
import statistics as st

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.auxiliary_downward45_depth_observation_development import body_from_optical as auxiliary_from_optical
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL, HEIGHT, MAX_DEPTH_M, MIN_DEPTH_M, WIDTH
from lewm.physical_execution_development import rotation_xyzw
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.diagnose_go2_forecast_sensitivity_failures_development import BASE, LABEL, assignments, order, spec_of, wall_distance

PER_MISSION, SPACING_S = 5, 1.
BAND_HEIGHTS_M = (.05, .15, .30, .45, .60)
CAMERAS = {'primary': np.linalg.inv(np.asarray(BODY_FROM_OPTICAL, float)), 'auxiliary': np.linalg.inv(np.asarray(auxiliary_from_optical(), float))}
SAFETY_TEST = ('scale:0.25', 'scale:0.5', 'turnscale:0.5', 'turnscale:0.25')
PARTS = [f'{s} · {p}' for s in ('front', 'side', 'rear') for p in ('body', 'leg')]
MOVES = ('forward', 'arc', 'turn in place', 'other')
MOVE_GROUPS = {'in-place turns': ('turn in place',), 'forward / arcs': ('forward', 'arc'), 'other': ('other',)}
DISC_M, TURN_REQUIRED_M = .45, .48
REACH_DIRECTIONS = np.array([[math.cos(a), math.sin(a), 0.] for a in np.linspace(0, 2*math.pi, 16, endpoint=False)])


class Walls:
    """Per-primitive, per-wall separation lower bounds with the frozen V4 evaluator's support computation."""

    def __init__(self, walls):
        self.model = ArticulatedCollisionGeometry(URDF)
        self.centres = np.array([w['centre_xyz'] for w in walls], float)
        self.halves = np.array([w['size_xyz'] for w in walls], float)/2
        self.axes = np.array([[[math.cos(w['yaw_rad']), math.sin(w['yaw_rad']), 0], [-math.sin(w['yaw_rad']), math.cos(w['yaw_rad']), 0], [0, 0, 1]]
                              for w in walls])
        self.normals = self.axes.reshape(-1, 3)
        self.projection = np.einsum('wij,wj->wi', self.axes, self.centres)

    def reach(self, pose, joints):
        """Largest horizontal distance of any primitive from the base centre, and its link."""
        Q = rotation_xyzw(pose[3:])
        support = self.model.supports(joints, REACH_DIRECTIONS@Q)
        upper = np.array([s['upper'] for s in support['shapes']])
        k = np.unravel_index(np.argmax(upper), upper.shape)
        return float(upper[k]), support['shapes'][k[0]]['link']

    def reach_toward(self, pose, joints, direction_world):
        """Horizontal extent of the body from the base centre along one world direction (exact support)."""
        d = np.array([direction_world[0], direction_world[1], 0.])
        d /= np.linalg.norm(d)
        support = self.model.supports(joints, d[None, :]@rotation_xyzw(pose[3:]))
        return float(max(np.asarray(s['upper']).max() for s in support['shapes']))

    def closest(self, pose, joints):
        Q = rotation_xyzw(pose[3:])
        support = self.model.supports(joints, self.normals@Q)
        translation = self.normals@pose[:3]
        n = len(support['shapes'])
        lo = (np.array([s['lower'] for s in support['shapes']])+translation).reshape(n, -1, 3)
        hi = (np.array([s['upper'] for s in support['shapes']])+translation).reshape(n, -1, 3)
        separation = np.maximum(lo-(self.projection+self.halves), self.projection-self.halves-hi).max(axis=2)
        shape, wall = np.unravel_index(np.argmin(separation), separation.shape)
        row = support['shapes'][shape]
        centre = pose[:3]+Q@np.asarray(row['center_body_m'])
        local = self.axes[wall]@(centre-self.centres[wall])
        point = self.centres[wall]+self.axes[wall].T@np.clip(local, -self.halves[wall], self.halves[wall])
        return float(separation[shape, wall]), row['link'], point, Q


def visibility(point_xy, pose, Q):
    """'in view', or why not, for the wall at point_xy across the depth stop's height band."""
    reason = 'outside field of view'
    for h in BAND_HEIGHTS_M:
        body = Q.T@(np.array([point_xy[0], point_xy[1], h])-pose[:3])
        for name, optical_from_body in CAMERAS.items():
            x, y, z = (optical_from_body@np.append(body, 1.))[:3]
            if z <= 0 or abs(FOCAL*x/z) > WIDTH/2 or abs(FOCAL*y/z) > HEIGHT/2:
                continue
            if MIN_DEPTH_M <= z <= MAX_DEPTH_M:
                return 'in view', name
            if z < MIN_DEPTH_M:
                reason = 'closer than minimum depth'
    return reason, None


def move_type(command):
    vx, vy, wz = command
    if vx > 0 and abs(vy) < 1e-9:
        return 'forward' if abs(wz) < 1e-9 else 'arc'
    if vx == 0 and vy == 0 and wz != 0:
        return 'turn in place'
    return 'other'


def mission(args):
    cohort, assignment = args
    root = BASE/'runs'/assignment
    spec = json.loads((root/'specification.json').read_text())
    walls = Walls(spec['geometry']['wall_boxes'])
    centre_distance = wall_distance(spec['geometry']['wall_boxes'])
    requests = json.loads((root/'requests.json').read_text())
    request_s = np.array([q['simulator_ns'] for q in requests])/1e9
    plans = {r['measured_ns']: r for r in json.loads((root/'planning.json').read_text()) if 'selection' in r}
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as f:
        ts, poses, joints, applied = f['timestamp_s'].copy(), f['base_pose_world'].copy(), f['joint_position'].copy(), f['applied_command'].copy()
    with np.load(root/'native_clearance_summary_arrays.npz', allow_pickle=False) as c:
        ct, sep = c['timestamp_s'].copy(), c['separation_lower_m'].copy()
    index = np.clip(np.searchsorted(ts, ct), 0, len(ts)-1)
    moving = np.any(applied[index] != 0, axis=1)
    dt = np.diff(ts, append=ts[-1])
    sampled = applied[::25]  # 50-ms resolution for the time split, moving samples only
    kinds = np.array([move_type(c) if np.any(c != 0) else 'stationary' for c in sampled])
    move_time = {m: float(dt[::25][kinds == m].sum()*25) for m in MOVES}
    chosen = []
    for j in np.flatnonzero(moving)[np.argsort(sep[moving], kind='stable')]:
        if all(abs(ct[j]-ct[k]) >= SPACING_S for k in chosen):
            chosen.append(j)
            if len(chosen) == PER_MISSION:
                break
    rows = []
    for j in chosen:
        i = index[j]
        separation, link, point, Q = walls.closest(poses[i], joints[i])
        direction = Q.T@(point-poses[i, :3])
        angle = math.degrees(math.atan2(direction[1], direction[0]))
        sector = 'front' if abs(angle) <= 45 else 'rear' if abs(angle) >= 135 else 'side'
        part = 'leg' if link[:3] in ('FL_', 'FR_', 'RL_', 'RR_') else 'body'
        view, camera = visibility(point[:2], poses[i], Q)
        centre = float(centre_distance(poses[i, :2])[0])
        q = requests[max(0, int(np.searchsorted(request_s, ct[j], side='right'))-1)]
        plan = plans.get(q.get('command_observation_ns'))
        remembered = None
        if plan is not None:
            memory = {c['action']: c for c in plan['selection'].get('memory_forecast_candidates') or []}
            remembered = (memory.get(plan['action']) or {}).get('minimum_predicted_path_clearance_m')
        rows.append(dict(time_s=round(float(ct[j])-1.5, 3), separation_m=float(sep[j]), recomputed_m=separation, link=link,
                         part=f'{sector} · {part}', bearing_deg=round(angle, 1), move=move_type(applied[i]), command=applied[i].tolist(),
                         view=view, camera=camera, centre_to_wall_m=centre, reach_m=walls.reach_toward(poses[i], joints[i], point[:2]-poses[i, :2]),
                         dispatch_reason=q['reason'], planned_action=plan['action'] if plan else None, remembered_clearance_m=remembered))
    reach = []
    for i in range(0, len(ts), 50):
        if np.any(applied[i] != 0):
            value, link = walls.reach(poses[i], joints[i])
            reach.append(dict(move=move_type(applied[i]), reach_m=value, link=link))
    return dict(cohort=cohort, assignment=assignment, approaches=rows, move_time_s=move_time, reach=reach)


def summarise(approaches, reference_median, reference_p10):
    seps = [a['separation_m'] for a in approaches]
    out = [a for a in approaches if a['view'] != 'in view']
    deficit = lambda a: max(0., reference_median-a['separation_m'])
    total = sum(deficit(a) for a in approaches)
    close = [a for a in approaches if a['separation_m'] < reference_p10]
    return dict(n=len(approaches), worst=min(seps), p10=float(np.percentile(seps, 10)), median=st.median(seps),
                out_of_view=len(out)/len(approaches), deficit_out_of_view=sum(deficit(a) for a in out)/total if total else None,
                deficit_mean_mm=1000*total/len(approaches), close=len(close),
                close_out_of_view=sum(a['view'] != 'in view' for a in close)/len(close) if close else None)


def table(results, cohorts):
    by = {c: [a for r in results if r['cohort'] == c for a in r['approaches']] for c in cohorts}
    clean = next(c for c in cohorts if spec_of(c) == 'none')
    seps = [a['separation_m'] for a in by[clean]]
    reference_median, reference_p10 = st.median(seps), float(np.percentile(seps, 10))
    worst_check = max(abs(a['separation_m']-a['recomputed_m']) for c in cohorts for a in by[c])
    pct = lambda v: '-' if v is None else f'{100*v:.0f}%'
    lines = [f'**Closest approaches while moving (worst {PER_MISSION} per mission, ≥ {SPACING_S:.0f} s apart) and whether the depth stop could see them. {LABEL}**', '',
             f'Out of view = the nearest wall point, anywhere in the depth stop\'s 0.03–0.65 m height band, was outside both forward depth cameras\' '
             f'fields of view or closer than their 0.2-m minimum depth: approaches the depth stop structurally cannot catch. Clearance loss is '
             f'measured against this run\'s clean baseline: deficit = how far an approach falls below the clean median approach '
             f'({100*reference_median:.1f} cm); close = closer than the clean 10th percentile ({100*reference_p10:.1f} cm). '
             f'Closest-primitive recomputation matches the logged separation to within {1000*worst_check:.2f} mm.', '',
             '| Condition | Missions | Approaches | Separation: worst · p10 · median (cm) | Out of view (all approaches) '
             '| Mean deficit below clean median (mm) · share out of view | Close approaches · share out of view |',
             '|---|---:|---:|---|---:|---|---|']
    for c in cohorts:
        if not by[c]:
            continue
        s = summarise(by[c], reference_median, reference_p10)
        lines.append(f"| {spec_of(c)} | {sum(r['cohort'] == c for r in results)} | {s['n']} | {100*s['worst']:.1f} · {100*s['p10']:.1f} · {100*s['median']:.1f} | "
                     f"{pct(s['out_of_view'])} | {s['deficit_mean_mm']:.1f} · {pct(s['deficit_out_of_view'])} | {s['close']} · {pct(s['close_out_of_view'])} |")
    lines += ['', f'**Clearance loss by executing move: in-place turns versus forward / arcs. {LABEL}**', '',
              'Share of moving time from the applied commands. Deficit share = the move\'s part of the cohort\'s summed deficit below the clean median approach.', '',
              '| Condition | Move | Share of moving time | Approaches | Separation: worst · median (cm) | Mean deficit (mm) | Share of deficit '
              '| Close approaches · share out of view |',
              '|---|---|---:|---:|---|---:|---:|---|']
    for c in cohorts:
        if not by[c]:
            continue
        time = Counter()
        for r in results:
            if r['cohort'] == c:
                time.update(r['move_time_s'])
        moving_time = sum(time.values()) or 1
        deficit = lambda a: max(0., reference_median-a['separation_m'])
        total = sum(deficit(a) for a in by[c]) or 1
        for group, moves in MOVE_GROUPS.items():
            rows = [a for a in by[c] if a['move'] in moves]
            if not rows and group == 'other':
                continue
            close = [a for a in rows if a['separation_m'] < reference_p10]
            lines.append(f"| {spec_of(c)} | {group} | {sum(time[m] for m in moves)/moving_time:.2f} | {len(rows)} | "
                         + (f"{100*min(a['separation_m'] for a in rows):.1f} · {100*st.median(a['separation_m'] for a in rows):.1f} | "
                            f"{1000*sum(deficit(a) for a in rows)/len(rows):.1f} | {pct(sum(deficit(a) for a in rows)/total)} | " if rows else '- | - | - | ')
                         + f"{len(close)} · {pct(sum(a['view'] != 'in view' for a in close)/len(close)) if close else '-'} |")
    lines += ['', f'**The planner\'s disc check at each approach. {LABEL}**', '',
              'Every planner-stage clearance filter tests a 0.45-m disc on the forecast centre path (0.48 m for turns and translations) against '
              'remembered obstacles; heading and the articulated body are not modelled. Reach = how far the closest part extends from the base '
              'centre toward the nearest wall point (exact support of all 27 primitives in that direction). Disc broken in truth = the base centre was within 0.45 m '
              'of a true wall; planner believed clear = the remembered clearance of the executing move\'s forecast path was above 0.45 m.', '',
              '| Condition | Approaches | Rear-calf approaches | Rear-calf reach: median · p95 · max (cm) | Part outside the 0.45-m disc '
              '| Centre to true wall: worst · median (cm) | Disc broken in truth · of those, planner believed clear |',
              '|---|---:|---:|---|---:|---|---|']
    for c in cohorts:
        if not by[c]:
            continue
        rear = [a for a in by[c] if a['link'] in ('RL_calf', 'RR_calf')]
        broken = [a for a in by[c] if a['centre_to_wall_m'] < DISC_M]
        believed = [a for a in broken if a['remembered_clearance_m'] is not None and a['remembered_clearance_m'] > DISC_M]
        reach = [a['reach_m'] for a in rear]
        lines.append(f"| {spec_of(c)} | {len(by[c])} | {len(rear)} | "
                     + (f"{100*st.median(reach):.1f} · {100*float(np.percentile(reach, 95)):.1f} · {100*max(reach):.1f} | " if reach else '- | ')
                     + f"{pct(sum(a['reach_m'] > DISC_M for a in by[c])/len(by[c]))} | "
                     f"{100*min(a['centre_to_wall_m'] for a in by[c]):.1f} · {100*st.median(a['centre_to_wall_m'] for a in by[c]):.1f} | "
                     f"{len(broken)} · {len(believed)} |")
    lines += ['', f'**Articulated body reach versus the 0.45-m disc while moving (10 Hz samples). {LABEL}**', '',
              'Reach = the largest horizontal distance of any of the 27 collision primitives from the base centre. Beyond 0.45 m, a part sticks out '
              'of the disc the planner checks; beyond 0.48 m, out of the disc plus the turn/translation reserve.', '',
              '| Condition | Move | Samples | Reach: median · p95 · max (cm) | Beyond 0.45 m | Beyond 0.48 m | Farthest link when beyond 0.45 m |',
              '|---|---|---:|---|---:|---:|---|']
    for c in cohorts:
        samples = [x for r in results if r['cohort'] == c for x in r.get('reach', [])]
        for group, moves in MOVE_GROUPS.items():
            rows = [x for x in samples if x['move'] in moves]
            if not rows:
                continue
            values = np.array([x['reach_m'] for x in rows])
            far = Counter(x['link'] for x in rows if x['reach_m'] > DISC_M)
            lines.append(f"| {spec_of(c)} | {group} | {len(rows)} | {100*np.median(values):.1f} · {100*np.percentile(values, 95):.1f} · {100*values.max():.1f} | "
                         f"{pct(float(np.mean(values > DISC_M)))} | {pct(float(np.mean(values > TURN_REQUIRED_M)))} | "
                         f"{', '.join(f'{k} {v}' for k, v in far.most_common(2)) or '-'} |")
    focus = [c for c in cohorts if spec_of(c) in ('none', *SAFETY_TEST) and by[c]]
    for c in focus:
        approaches = by[c]
        lines += ['', f'**{spec_of(c)}: closest part and executing move ({len(approaches)} approaches). {LABEL}**', '',
                  '| Closest part (direction · leg or body) | Approaches | Median separation (cm) | Out of view | Why out of view: outside FOV · closer than 0.2 m |',
                  '|---|---:|---:|---:|---|']
        for part in PARTS:
            rows = [a for a in approaches if a['part'] == part]
            if not rows:
                continue
            why = Counter(a['view'] for a in rows)
            lines.append(f"| {part} | {len(rows)} | {100*st.median(a['separation_m'] for a in rows):.1f} | "
                         f"{pct(sum(a['view'] != 'in view' for a in rows)/len(rows))} | {why['outside field of view']} · {why['closer than minimum depth']} |")
        lines += ['', '| Executing move | Approaches | Median separation (cm) | Out of view |', '|---|---:|---:|---:|']
        for move in MOVES:
            rows = [a for a in approaches if a['move'] == move]
            if rows:
                lines.append(f"| {move} | {len(rows)} | {100*st.median(a['separation_m'] for a in rows):.1f} | "
                             f"{pct(sum(a['view'] != 'in view' for a in rows)/len(rows))} |")
        links = Counter(a['link'] for a in approaches)
        lines += ['', 'Closest links: '+', '.join(f'{k} {v}' for k, v in links.most_common(6))+'.']
    return '\n'.join(lines)


def main(cohorts, out_json, out_md, workers):
    cohorts = sorted(cohorts or [p.name for p in (BASE/'dev_cohorts').glob('sens_*') if (p/'config.json').exists()], key=order)
    jobs = [j for c in cohorts for j in assignments(c)]
    with Pool(workers) as pool:
        results = pool.map(mission, jobs, chunksize=1)
    text = table(results, cohorts)
    print(text)
    if out_json:
        Path(out_json).write_text(json.dumps(dict(label=LABEL, missions=results), indent=1)+'\n')
    if out_md:
        Path(out_md).write_text(text+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cohorts', nargs='*')
    p.add_argument('--json')
    p.add_argument('--markdown')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    main(a.cohorts, a.json, a.markdown, a.workers)
