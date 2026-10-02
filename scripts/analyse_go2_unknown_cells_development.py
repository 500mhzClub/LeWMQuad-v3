"""PRELIMINARY: how often a planned forecast path crosses never-observed cells near the robot (Andrew, 2 October 2026).

The planner's clearance checks measure distance to remembered occupied cells only, so a cell
that has never been observed counts as free. Neither the map nor the depth images were kept,
so observation coverage is reconstructed offline: both depth cameras (the depth stop's
calibrations: primary level 78 x 63 degrees, auxiliary pitched 45 degrees down; optical depth
0.2-5 m) are ray-cast at every 10-Hz frame from the true camera pose against the true floor
plane and wall boxes. A 5-cm cell is observed once a ray's first return lands in it, on the
floor or on a wall between 0.03 and 0.65 m (the mapper's height band). Sim depth is exact, so
this is the geometric observation set, sampled at 32 x 24 rays per camera (the mapper uses a
4-pixel stride); it is validated by comparing the reconstructed remembered clearance (distance
from the decision's forecast centre path to observed wall returns) with the clearance the
planner logged.

Per executed moving decision (the selected move's forecast centre path, the one the check
used, placed at the true pose):
- reach = 5-cm cells within 0.45 m of the path (the checked disc) and within 0.5 m of the
  robot's current centre;
- unknown in reach: any never-observed cell there; share of decisions, and cells per decision;
- hidden wall in reach: a never-observed cell there that a true wall occupies;
- passed only because unknown is free: the path's true clearance is below 0.45 m while its
  clearance to the observed wall returns is above 0.45 m;
- where the unknown cells lie relative to the robot (front, side, rear) and the move;
- for a pessimistic-unknown variant (never-observed cells within reach count as occupied),
  what it would still block after seeding the start disc (0.50 m: the episode generator
  rejects spawns closer than 0.5 m to a wall) and the robot's own track (cells within 0.20 m of
  any past base-centre position, under the body core) as known free.

Usage: analyse_go2_unknown_cells_development.py [--cohorts ...] [--json OUT] [--markdown OUT] [--workers 3]
"""
import argparse
from collections import Counter
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lewm.auxiliary_downward45_depth_observation_development import body_from_optical as auxiliary_from_optical
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL, MAX_DEPTH_M, MIN_DEPTH_M
from lewm.physical_execution_development import rotation_xyzw
from scripts.analyse_go2_forecast_sensitivity_error_budget_development import PASSED, planar_yaw, rot
from scripts.diagnose_go2_forecast_sensitivity_failures_development import ACTIONS, BASE, assignments, order, spec_of, wall_distance

CELL_M, FINE_M = .05, .01
START_FREE_M, TRACK_FREE_M = .50, .20  # first estimate (2 Oct): seeded known-free areas (spawn clearance >= 0.5 m by the generator)
BODY_REACH_M, REVISED_SCOPE_M = .425, .80  # revised rule (Andrew, 2 Oct): start reach disc 0.425 m + track; block within reach + bound
BOUNDS = json.loads((Path(__file__).resolve().parents[1]/'docs/go2_navigation_calibrated_margins_2026-10-02.json').read_text())['controllers']
DISC_M, NEAR_ROBOT_M = .45, .50
BAND = (.03, .65)
COLUMNS, ROWS = 32, 24
TURNS = ('left_turn', 'right_turn')
LABEL = 'PRELIMINARY (prelim_test_v1 mazes 30-49; development mode)'


def camera_rays():
    u = (np.arange(COLUMNS)+.5)*640/COLUMNS
    v = (np.arange(ROWS)+.5)*480/ROWS
    uu, vv = np.meshgrid(u, v)
    optical = np.stack(((uu-320)/FOCAL, (vv-240)/FOCAL, np.ones_like(uu)), axis=-1).reshape(-1, 3)
    out = []
    for T in (np.asarray(BODY_FROM_OPTICAL, float), np.asarray(auxiliary_from_optical(), float)):
        out.append((T[:3, 3], optical@T[:3, :3].T))  # origin and direction (optical-depth parametrised) in the body frame
    return out


RAYS = camera_rays()


class Scene:
    def __init__(self, walls):
        self.centre = np.array([w['centre_xyz'] for w in walls], float)
        self.half = np.array([w['size_xyz'] for w in walls], float)/2
        self.cos = np.array([math.cos(w['yaw_rad']) for w in walls])
        self.sin = np.array([math.sin(w['yaw_rad']) for w in walls])
        lo = (self.centre[:, :2]-np.linalg.norm(self.half[:, :2], axis=1)[:, None]).min(axis=0)-.6
        hi = (self.centre[:, :2]+np.linalg.norm(self.half[:, :2], axis=1)[:, None]).max(axis=0)+.6
        self.origin = lo
        self.shape = np.ceil((hi-lo)/CELL_M).astype(int)
        self.observed = np.zeros(self.shape, bool)
        self.seeded = np.zeros(self.shape, bool)
        self.seeded_reach = np.zeros(self.shape, bool)
        self.centres = self.origin+(np.indices(self.shape).transpose(1, 2, 0)+.5)*CELL_M
        self.fine = set()
        self.distance = wall_distance(walls)
        centres = self.origin+(np.indices(self.shape).reshape(2, -1).T+.5)*CELL_M
        self.occupied = (self.distance(centres) <= CELL_M*math.sqrt(2)/2).reshape(self.shape)

    def cast(self, origin, directions):
        """First return of each ray (optical-depth parameter); floor z = 0, wall boxes from the specification."""
        n = len(directions)
        s_floor = np.where(directions[:, 2] < 0, -origin[2]/np.where(directions[:, 2] < 0, directions[:, 2], -1), np.inf)
        o = origin[None, :]-self.centre  # (W, 3)
        lx = self.cos[:, None]*o[:, 0, None]+self.sin[:, None]*o[:, 1, None]
        ly = -self.sin[:, None]*o[:, 0, None]+self.cos[:, None]*o[:, 1, None]
        lz = np.broadcast_to(o[:, 2, None], lx.shape)
        dx = self.cos[:, None]*directions[None, :, 0]+self.sin[:, None]*directions[None, :, 1]
        dy = -self.sin[:, None]*directions[None, :, 0]+self.cos[:, None]*directions[None, :, 1]
        dz = np.broadcast_to(directions[None, :, 2], dx.shape)
        enter, leave = np.zeros((len(self.centre), n)), np.full((len(self.centre), n), np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            for local, d, h in ((lx, dx, self.half[:, 0, None]), (ly, dy, self.half[:, 1, None]), (lz, dz, self.half[:, 2, None])):
                a, b = (-h-local)/d, (h-local)/d
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                parallel = d == 0
                inside = (np.abs(local) <= h)
                lo = np.where(parallel, np.where(inside, -np.inf, np.inf), lo)
                hi = np.where(parallel, np.where(inside, np.inf, -np.inf), hi)
                enter, leave = np.maximum(enter, lo), np.minimum(leave, hi)
        s_wall = np.where(leave >= enter, enter, np.inf).min(axis=0)
        return np.minimum(s_floor, s_wall), s_wall < s_floor

    def observe(self, pose):
        Q = rotation_xyzw(pose[3:])
        for origin_b, dirs_b in RAYS:
            origin = pose[:3]+Q@origin_b
            dirs = dirs_b@Q.T
            s, wall = self.cast(origin, dirs)
            ok = (s >= MIN_DEPTH_M) & (s <= MAX_DEPTH_M)
            points = origin+s[ok, None]*dirs[ok]
            wall = wall[ok]
            keep = ~wall | ((points[:, 2] > BAND[0]) & (points[:, 2] < BAND[1]))
            points, wall = points[keep], wall[keep]
            idx = np.floor((points[:, :2]-self.origin)/CELL_M).astype(int)
            inside = np.all((idx >= 0) & (idx < self.shape), axis=1)
            self.observed[idx[inside, 0], idx[inside, 1]] = True
            self.fine.update(map(tuple, np.floor(points[wall, :2]/FINE_M).astype(int)))

    def seed(self, centre, radius, grid=None):
        """Mark cells within radius of a body-centre position as known free (start disc, own track)."""
        grid = self.seeded if grid is None else grid
        lo = np.maximum(np.floor((centre-radius-self.origin)/CELL_M).astype(int), 0)
        hi = np.minimum(np.ceil((centre+radius-self.origin)/CELL_M).astype(int), self.shape)
        block = self.centres[lo[0]:hi[0], lo[1]:hi[1]]
        grid[lo[0]:hi[0], lo[1]:hi[1]] |= np.linalg.norm(block-centre, axis=-1) <= radius

    def unseen_distance(self, path, centre):
        """Distance from the path to the nearest never-observed, unseeded (revised seeding) 5-cm cell square within scope."""
        lo = np.maximum(np.floor((centre-REVISED_SCOPE_M-self.origin)/CELL_M).astype(int), 0)
        hi = np.minimum(np.ceil((centre+REVISED_SCOPE_M-self.origin)/CELL_M).astype(int), self.shape)
        corner = self.origin+np.indices((hi[0]-lo[0], hi[1]-lo[1])).transpose(1, 2, 0)*CELL_M+lo*CELL_M
        mask = ~self.observed[lo[0]:hi[0], lo[1]:hi[1]] & ~self.seeded_reach[lo[0]:hi[0], lo[1]:hi[1]]
        mask &= np.linalg.norm(corner+CELL_M/2-centre, axis=-1) <= REVISED_SCOPE_M
        squares = corner[mask]
        if not len(squares):
            return None
        gap = np.maximum(np.maximum(squares[None]-path[:, None], path[:, None]-(squares[None]+CELL_M)), 0.)
        return float(np.linalg.norm(gap, axis=-1).min())

    def cells_near(self, path, centre):
        lo = np.floor((path.min(axis=0)-DISC_M-self.origin)/CELL_M).astype(int)
        hi = np.ceil((path.max(axis=0)+DISC_M-self.origin)/CELL_M).astype(int)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, self.shape)
        ii, jj = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing='ij')
        ii, jj = ii.ravel(), jj.ravel()
        xy = self.origin+(np.stack((ii, jj), axis=1)+.5)*CELL_M
        to_path = cKDTree(path).query(xy)[0]
        near = (to_path <= DISC_M) & (np.linalg.norm(xy-centre, axis=1) <= NEAR_ROBOT_M)
        return ii[near], jj[near], xy[near]


def mission(args):
    cohort, assignment = args
    root = BASE/'runs'/assignment
    read = lambda n: json.loads((root/n).read_text())
    spec = read('specification.json')
    scene = Scene(spec['geometry']['wall_boxes'])
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as f:
        ts, P = f['timestamp_s'].copy(), f['base_pose_world'].copy()
    at = lambda t: min(int(np.searchsorted(ts, t)), len(ts)-1)
    plans = sorted((r for r in read('planning.json') if 'selection' in r), key=lambda r: r['measured_ns'])
    requests = read('requests.json')
    executed = {q['command_observation_ns'] for q in requests if q['reason'] == PASSED and any(q['requested_command'])}
    frames = sorted({r['frame'] for r in plans})
    rows = []
    frame_i = 0
    last_frame = int(round((ts[-1]-1.5)/.1))
    tree, tree_size = None, -1
    for plan in plans:
        # Integrate every 10-Hz frame up to and including this decision's observation.
        while frame_i <= min(plan['frame'], last_frame):
            pose = P[at(1.5+frame_i*.1)]
            if frame_i == 0:
                scene.seed(pose[:2], START_FREE_M)
                scene.seed(pose[:2], BODY_REACH_M, scene.seeded_reach)
            scene.seed(pose[:2], TRACK_FREE_M)
            scene.seed(pose[:2], TRACK_FREE_M, scene.seeded_reach)
            scene.observe(pose)
            frame_i += 1
        if plan['measured_ns'] not in executed or plan['action'] == 'hold':
            continue
        mc = plan.get('motion_correction') or {}
        key = next((k for k in ('dev_degraded_forecast_xy_yaw', 'applied_prediction_after_yaw_ablation', 'command_history_forecast_xy_yaw')
                    if mc.get(k) is not None), None)
        if key is None:
            continue
        f = np.vstack((np.zeros(2), np.asarray(mc[key], float)[ACTIONS.index(plan['action']), :, :2]))
        i = at(plan['measured_ns']/1e9)
        yaw = planar_yaw(P[i])
        segments = [a+(b-a)*np.linspace(0, 1, 6)[:, None] for a, b in zip(f[:-1], f[1:])]
        path = P[i, :2]+np.vstack(segments)@rot(yaw).T
        ii, jj, xy = scene.cells_near(path, P[i, :2])
        unknown = ~scene.observed[ii, jj]
        hidden = unknown & scene.occupied[ii, jj]
        bearing = np.degrees(np.arctan2(*(((xy-P[i, :2])@rot(yaw))[:, ::-1].T))) if len(xy) else np.array([])
        sector = np.where(np.abs(bearing) <= 45, 'front', np.where(np.abs(bearing) >= 135, 'rear', 'side'))
        true_clear = float(scene.distance(path).min())
        if len(scene.fine) != tree_size:
            tree = cKDTree((np.array(list(scene.fine))+.5)*FINE_M) if scene.fine else None
            tree_size = len(scene.fine)
        observed_clear = float(tree.query(path)[0].min()) if tree is not None else None
        memory = {c['action']: c for c in plan['selection'].get('memory_forecast_candidates') or []}
        logged = (memory.get(plan['action']) or {}).get('minimum_predicted_path_clearance_m')
        rows.append(dict(time_s=round(plan['measured_ns']/1e9-1.5, 1), action=plan['action'], cells=int(len(ii)), unknown=int(unknown.sum()),
                         unknown_after_seeding=int((unknown & ~scene.seeded[ii, jj]).sum()),
                         unseen_distance_revised=scene.unseen_distance(path, P[i, :2]), hidden=int(hidden.sum()),
                         unknown_sectors=dict(Counter(sector[unknown])), true_clearance=true_clear, observed_clearance=observed_clear,
                         logged_clearance=logged, unknown_only_pass=bool(true_clear < DISC_M and (observed_clear is None or observed_clear > DISC_M))))
    return dict(cohort=cohort, assignment=assignment, controller=read('config.json').get('controller'), decisions=rows)


def table(results):
    groups = {}
    for r in results:
        groups.setdefault((r['cohort'], r['controller']), []).extend(r['decisions'])
    pct = lambda v: '-' if v is None else f'{100*v:.1f}%'
    lines = [f'**Planned forecast paths crossing never-observed cells near the robot. {LABEL}**', '',
             'Per executed moving decision: reach = 5-cm cells within 0.45 m of the selected move\'s forecast centre path (the checked disc) and within '
             '0.5 m of the robot. Unknown = never observed by either depth camera so far (reconstructed by ray casting). Hidden wall = an unknown '
             'reach cell a true wall occupies. Passed only because unknown is free = true path clearance < 45 cm while clearance to the observed wall '
             'returns is > 45 cm. Validation: reconstructed vs logged remembered clearance.', '',
             '| Cohort | Controller | Decisions | Unknown in reach: all moves · turns · forward/arcs · after the first 30 s | Unknown cells per decision (of reach) '
             '| Unknown cells by direction: front · side · rear | Still unknown in reach after seeding start disc and own track (pessimistic variant would block) '
             '| Hidden wall in reach | Passed only because unknown is free '
             '| Reconstructed − logged clearance: median · p95 abs (cm) |',
             '|---|---|---:|---|---|---|---|---:|---:|---|']
    for (cohort, controller), ds in sorted(groups.items(), key=lambda kv: (kv[0][0].startswith('sens_'), order(kv[0][0]) if kv[0][0].startswith('sens_') else (0, 0), kv[0][0], kv[0][1])):
        if not ds:
            continue
        share = lambda rows: (sum(d['unknown'] > 0 for d in rows)/len(rows)) if rows else None
        share2 = lambda rows: (sum(d['unknown_after_seeding'] > 0 for d in rows)/len(rows)) if rows else None
        turns = [d for d in ds if d['action'] in TURNS]
        trans = [d for d in ds if d['action'] not in TURNS]
        sectors = Counter()
        for d in ds:
            sectors.update(d['unknown_sectors'])
        total = sum(sectors.values()) or 1
        diff = [d['observed_clearance']-d['logged_clearance'] for d in ds if d['observed_clearance'] is not None and d['logged_clearance'] is not None
                and d['logged_clearance'] < 1.5]
        label = spec_of(cohort) if cohort.startswith('sens_') else cohort
        lines.append(f"| {label} | {controller} | {len(ds)} | {pct(share(ds))} · {pct(share(turns))} · {pct(share(trans))} · "
                     f"{pct(share([d for d in ds if d['time_s'] >= 30]))} | "
                     f"{np.mean([d['unknown'] for d in ds]):.1f} of {np.mean([d['cells'] for d in ds]):.0f} | "
                     f"{sectors['front']/total:.2f} · {sectors['side']/total:.2f} · {sectors['rear']/total:.2f} | "
                     f"{pct(sum(d['unknown_after_seeding'] > 0 for d in ds)/len(ds))} (turns {pct(share2(turns))}, after 30 s {pct(share2([d for d in ds if d['time_s'] >= 30]))}) | "
                     f"{pct(sum(d['hidden'] > 0 for d in ds)/len(ds))} | {pct(sum(d['unknown_only_pass'] for d in ds)/len(ds))} | "
                     + (f"{100*np.median(diff):+.1f} · {100*np.percentile(np.abs(diff), 95):.1f} |" if diff else '- |'))
    return '\n'.join(lines)


def revised_table(results):
    """Offline replay of the revised pessimistic-unknown rule on every logged executed moving decision."""
    groups = {}
    for r in results:
        groups.setdefault((r['cohort'], r['controller']), []).extend(r['decisions'])
    pct = lambda k, n: '-' if not n else f'{k}/{n} ({100*k/n:.1f}%)'
    lines = [f'**Offline replay of the revised pessimistic-unknown rule. {LABEL}**', '',
             'Seeded known free: the start reach disc (0.425 m, by cell centre) and the traversed track (0.20 m). A decision is blocked if a '
             'never-observed 5-cm cell square lies within body reach (0.425 m) + the controller\'s calibrated e_f bound of its forecast centre path '
             '(placed at the true pose; observation reconstructed by ray casting). Unsafe pass = true path clearance < 45 cm while clearance to '
             'observed walls > 45 cm. Blocked counts are for the selected move only; the live planner would then choose among the others.', '',
             '| Cohort | Controller | Decisions | Blocked at p95 radius: all · first 30 s · after 30 s | Blocked at p99 radius: all · first 30 s · after 30 s '
             '| Unsafe passes | Unsafe passes blocked: p95 · p99 |',
             '|---|---|---:|---|---|---:|---|']
    for (cohort, controller), ds in sorted(groups.items(), key=lambda kv: (kv[0][0].startswith('sens_'), order(kv[0][0]) if kv[0][0].startswith('sens_') else (0, 0), kv[0][0], kv[0][1])):
        if controller not in BOUNDS or not ds:
            continue
        cells = []
        for level in ('p95', 'p99'):
            radius = BODY_REACH_M+BOUNDS[controller][level]['margin_m']
            blocked = lambda rows: sum(d['unseen_distance_revised'] is not None and d['unseen_distance_revised'] < radius for d in rows)
            early, late = [d for d in ds if d['time_s'] < 30], [d for d in ds if d['time_s'] >= 30]
            cells.append(f"{pct(blocked(ds), len(ds))} · {pct(blocked(early), len(early))} · {pct(blocked(late), len(late))}")
        unsafe = [d for d in ds if d['unknown_only_pass']]
        hit = lambda level: sum(d['unseen_distance_revised'] is not None and d['unseen_distance_revised'] < BODY_REACH_M+BOUNDS[controller][level]['margin_m']
                                for d in unsafe)
        label = spec_of(cohort) if cohort.startswith('sens_') else cohort
        lines.append(f"| {label} | {controller} | {len(ds)} | {cells[0]} | {cells[1]} | {len(unsafe)} | "
                     f"{pct(hit('p95'), len(unsafe))} · {pct(hit('p99'), len(unsafe))} |")
    return '\n'.join(lines)


def main(cohorts, out_json, out_md, workers):
    jobs = [j for c in cohorts for j in assignments(c)]
    with Pool(workers) as pool:
        results = pool.map(mission, jobs, chunksize=1)
    text = table(results)+'\n\n'+revised_table(results)
    print(text)
    if out_json:
        Path(out_json).write_text(json.dumps(dict(label=LABEL, missions=results), indent=1)+'\n')
    if out_md:
        Path(out_md).write_text(text+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cohorts', nargs='+', required=True)
    p.add_argument('--json')
    p.add_argument('--markdown')
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    main(a.cohorts, a.json, a.markdown, a.workers)
