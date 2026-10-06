"""PRELIMINARY: failure mechanism and safety for every forecast-sensitivity cohort (Andrew, 2 October 2026).

Question: do bad forecasts make the robot freeze or make it unsafe, and how much does the
last-moment depth stop catch? Read-only over each cohort's preserved mission records.

Failure mechanism, per failed mission. Progress uses the true-map geodesic of the earlier V4
timeout diagnosis (0.46-m inflation + 5-mm clearance, 2-cm grid). On the active leg (outbound
if the beacon was never reached, otherwise return), over the final 120 s:
- progressing at budget run-out: the geodesic distance to the active target fell >= 0.25 m;
- otherwise stall / no-move: translating commands applied < 10% of the window, split into
  turning in place (turn-only commands >= 25% of the window) and holding;
- otherwise wandering / detour without progress: it translated but got no closer.
A mission ended by visual pose loss is labelled pose loss, with its behaviour before the loss.
Separately, where it failed: at the target (ended within 0.25 m of the active target, arrival
never confirmed; arrival needs the pose estimate within 2 cm, the true position within 4 cm
and a 1-s dwell with zero commands) or en route. Thresholds for progress and translation were
fixed before reading any sensitivity failure; the at-target location and the holding / turning
split were added after the first failures read ended within centimetres of the beacon while
turning in place. The same behaviour labels are applied to every 30-s window of every mission
(progress threshold 0.1 m, the earlier "stuck interval" rule), giving time shares.

Success two ways (Andrew, 2 October): strict (the frozen arrival rule) and reached (the true
base came within 0.25 m of the beacon and later within 0.25 m of home), so settling failures
are not counted as navigation failures. The harness starts the return leg only after a
confirmed beacon arrival, so an outbound settling failure can never reach home; reached
beacon (within 0.25 m on the outbound) is also reported for that reason. Pose loss (the visual
tracker failed) is a shared-system failure, not a forecast failure, and is counted on its own.

Safety, per mission: disallowed contact samples, hard (5 mm) and operating (20 mm) clearance
violations and the minimum wall separation (native ground truth, the frozen reader); and the
distribution of the native separation sampled at 500 Hz, over the whole mission and while a
translating command was applied (worst, 5th percentile, median).

Last-moment depth stop: the dispatch check of the requested-speed disk connector (0.45 m)
against the current depth image (CURRENT_OBSERVED_OBSTACLE_VETO, or
CURRENT_STOPPING_MARGIN_VETO where that layer runs). It does not use the forecast. One veto
latches the rest of its command window, so one veto record = one stopped command window.
Vetoes of windows whose plan was a hold stop nothing (the robot parked within 0.45 m of a
wall fails the disk check even at zero speed), so stops are counted on windows that planned a
move. Other dispatch vetoes (stale observation, late first dispatch) are counted separately.
Counterfactual for each depth stop on a planned translation (Andrew: is under-prediction the
one direction where forecast error costs safety?): from the true pose at the catch, the
remaining motion of the planned move to the end of its forecast (800 ms after the decision's
observation) is taken from C1's clean forecast, which matches physics to about 1 cm here, as
the path the robot would have swept without the stop; the degraded forecast gives the path
the planner believed. Separation along each path = native separation at the catch plus the
change in the base centre's distance to the true wall boxes (first-order: exact for a
translation toward a planar wall face). Would-be outcome: contact (<= 0), hard violation
(< 5 mm), operating violation (< 20 mm) or none.

Planner-stage filters, which do use the forecast: per decision, how many of the five moves
the forecast-based memory clearance check rejects (nominal predicted path not clear), and the
share of decisions where it rejects all five. Planner holds by the frozen reader's hold
records: arrival settling, hold won on score, a move blocked by forecast-based clearance (a
latched recovery turn blocked, the stopping projection, or no eligible movement), or the
coverage rule.

Usage: diagnose_go2_forecast_sensitivity_failures_development.py [--cohorts sens_...] [--json OUT] [--markdown OUT] [--workers 8]
"""
import argparse
from collections import Counter
import json
import math
from multiprocessing import Pool
from pathlib import Path
import statistics as st

import numpy as np

from lewm.decision_headroom_reference_development import ReferenceGeometry
from scripts.report_go2_prelim_results_development import wilson

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO/'docs/go2_navigation_capability_completed_support_v4_2026-09-27.json'
BASE = Path(json.loads(PROTOCOL.read_text())['output_root'])
LABEL = 'PRELIMINARY (prelim_test_v1 mazes 30-49, C1, recovery off, coverage-rule fix; development mode)'
FINAL_S, FINAL_PROGRESS_M, WINDOW_S, WINDOW_PROGRESS_M, MOVE_FRACTION, TURN_FRACTION, AT_TARGET_M = 120., .25, 30., .1, .10, .25, .25
DEPTH_STOPS = ('CURRENT_OBSERVED_OBSTACLE_VETO', 'CURRENT_STOPPING_MARGIN_VETO')
OTHER_VETOES = ('CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE', 'FIRST_DISPATCH_TOO_LATE', 'OUTSIDE_COMMITTED_INTERVAL')
MOVES = ('forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn')
ACTIONS = ('hold',)+MOVES
TRANSLATIONS = ('forward', 'left_arc', 'right_arc')
REACH_M = .25
EDGES = np.arange(-.05, 3.0005, .001)
BEHAVIOURS = ('holding', 'turning in place', 'wandering', 'progressing')


def hold_group(h):
    if h.get('override_reason') == 'PREDICTIVE_ARRIVAL_HOLD':
        return 'arrival settling'
    if h['category'] == 'movement_lost_recorded_score_or_tie':
        return 'on score'
    if h.get('override_reason') == 'TRANSLATION_FOOTPRINT_COVERAGE':
        return 'coverage rule'
    if h.get('override_reason') in ('LATCHED_RECOVERY_TURN_BLOCKED', 'PLANNED_STOPPING_PROJECTION') or h['category'] == 'no_eligible_movement':
        return 'forecast clearance'
    return 'other'


def yaw_of(q):
    """Yaw from a Genesis base pose row [x, y, z, qx, qy, qz, qw]."""
    return math.atan2(2*(q[6]*q[5]+q[3]*q[4]), 1-2*(q[4]**2+q[5]**2))


def reach_from(ep, xy):
    """Reached: within REACH_M of the beacon, then later within REACH_M of home (true base position)."""
    near_beacon = np.flatnonzero(np.linalg.norm(xy-np.asarray(ep['beacon_xy_world'], float), axis=1) <= REACH_M)
    home = np.linalg.norm(xy-np.asarray(ep['home_se2_world'][:2], float), axis=1) <= REACH_M
    return dict(reached_beacon=bool(len(near_beacon)), reached=bool(len(near_beacon)) and bool(home[near_beacon[0]:].any()))


def reach(assignment):
    root = BASE/'runs'/assignment
    ep = json.loads((root/'episode.json').read_text())
    ev = json.loads((root/'episode_evaluation.json').read_text())
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as f:
        xy = f['base_pose_world'][:, :2].copy()
    return reach_from(ep, xy) | dict(pose_loss=is_pose_loss(ev))


def is_pose_loss(ev):
    return 'measured visual pose unavailable' in str(ev.get('source_error') or '')


def wall_distance(walls):
    """Distance from points (n, 2) to the nearest wall box (negative inside a box)."""
    def distance(points):
        points = np.atleast_2d(points)
        best = np.full(len(points), np.inf)
        for w in walls:
            c, s = math.cos(w['yaw_rad']), math.sin(w['yaw_rad'])
            local = (points-np.asarray(w['centre_xyz'][:2]))@np.array([[c, -s], [s, c]])
            q = np.abs(local)-np.asarray(w['size_xyz'][:2])/2
            outside = np.linalg.norm(np.maximum(q, 0), axis=1)
            best = np.minimum(best, outside+np.minimum(np.max(q, axis=1), 0))
        return best
    return distance


def counterfactual(stop, plan, P, ts, ct, sep, distance):
    """Separation along the path the planned translation would have swept without the depth stop."""
    a = ACTIONS.index(plan['action'])
    mc = plan['motion_correction']
    tc, obs = stop['now_ns']/1e9, plan['measured_ns']/1e9
    i = min(np.searchsorted(ts, tc), len(ts)-1)
    xy, heading = P[i, :2], yaw_of(P[i])
    sep_catch = float(sep[min(np.searchsorted(ct, tc), len(ct)-1)])
    base = float(distance(xy)[0])
    out = dict(time_s=round(tc-1.5, 2), action=plan['action'], separation_at_catch_m=sep_catch,
               remaining_window_s=(stop.get('command_expires_ns', plan['measured_ns']+700_000_000)-stop['now_ns'])/1e9)
    for name, key in (('true', 'command_history_forecast_xy_yaw'), ('believed', 'dev_degraded_forecast_xy_yaw')):
        f = np.asarray(mc.get(key) or mc['command_history_forecast_xy_yaw'], float)[a]
        f = np.vstack((np.zeros(3), f))
        f[:, 2] = np.unwrap(f[:, 2])
        steps = np.arange(9.)
        s = np.linspace(min(max((tc-obs)/.1, 0.), 8.), 8., 33)
        x, y, h = (np.interp(s, steps, f[:, k]) for k in range(3))
        c, sn = math.cos(h[0]), math.sin(h[0])
        rel = np.stack((x-x[0], y-y[0]), axis=1)@np.array([[c, -sn], [sn, c]])
        c, sn = math.cos(heading), math.sin(heading)
        world = xy+rel@np.array([[c, sn], [-sn, c]])
        out[f'{name}_min_separation_m'] = sep_catch+float(np.min(distance(world)-base))
        out[f'{name}_travel_m'] = float(np.linalg.norm(rel[-1]))
    m = out['true_min_separation_m']
    out['would_be'] = 'contact' if m <= 0 else 'hard violation' if m < .005 else 'operating violation' if m < .02 else 'none'
    return out


def remaining_fn(geometry, native, t):
    def remaining(s):
        point = native[min(np.searchsorted(t, s), len(t)-1), :2]
        value = geometry.distance_and_heading(point)
        if value.get('valid'):
            return float(value['distance_m'])
        # Centre inside the inflated wall margin: nearest valid point within 0.5 m plus the offset.
        for r in np.arange(.02, .52, .02):
            best = min((v['distance_m'] for a in np.linspace(0, 2*np.pi, 32, endpoint=False)
                        if (v := geometry.distance_and_heading(point+r*np.array([np.cos(a), np.sin(a)]))).get('valid')), default=None)
            if best is not None:
                return float(best+r)
        return None
    return remaining


def mission(args):
    cohort, assignment = args
    root = BASE/'runs'/assignment
    read = lambda n: json.loads((root/n).read_text())
    ep, ev, planning, requests, spec = (read(n) for n in ('episode.json', 'episode_evaluation.json', 'planning.json', 'requests.json', 'specification.json'))
    s = ev['safety']
    plans = {r['measured_ns']: r for r in planning if 'selection' in r}
    reasons = Counter(q['reason'] for q in requests)
    stops = Counter()
    for q in requests:
        if q['reason'] in DEPTH_STOPS:
            plan = plans.get(q.get('command_observation_ns'))
            stops['unmatched' if plan is None else 'hold' if plan['action'] == 'hold' else 'turn' if plan['action'] in ('left_turn', 'right_turn') else 'translate'] += 1
    rejected = []
    for r in plans.values():
        memory = {c['action']: c for c in r['selection'].get('memory_forecast_candidates') or []}
        if memory:
            rejected.append(sum(not memory[a].get('nominal_predicted_path_clear', True) for a in MOVES if a in memory))
    row = dict(cohort=cohort, assignment=assignment, maze=ep['maze_id'], success=bool(ev['round_trip_success']), beacon=bool(ev['beacon_success']),
               error=ev.get('source_error'), contacts=ev['disallowed_contact_samples'] or 0,
               hard=s['hard']['confirmed_violation_samples'], operating=s['operating']['confirmed_violation_samples'],
               min_clearance_m=s['hard']['minimum_separation_lower_m'], depth_stops=dict(stops),
               move_windows=sum(r['action'] != 'hold' for r in plans.values()), decisions=len(plans),
               other_vetoes=sum(reasons[r] for r in OTHER_VETOES), reasons=dict(reasons),
               forecast_rejected_moves=rejected, holds=dict(Counter(hold_group(h) for h in ev['hold_details'])))
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as f:
        ts = f['timestamp_s'].copy()
        native = f['base_pose_world'].copy()
        applied = f['applied_command'].copy()
    t = ts-1.5
    end = float(t[-1])
    with np.load(root/'native_clearance_summary_arrays.npz', allow_pickle=False) as c:
        ct, sep = c['timestamp_s'].copy(), c['separation_lower_m'].copy()
    translating = np.any(applied[:, :2] != 0, axis=1)
    moving_sample = translating[np.clip(np.searchsorted(ts, ct), 0, len(ts)-1)]
    hist = lambda v: [[int(i), int(n)] for i, n in enumerate(np.histogram(np.clip(v, EDGES[0], EDGES[-1]-1e-9), bins=EDGES)[0]) if n]
    distance = wall_distance(spec['geometry']['wall_boxes'])
    catches = []
    for q in requests:
        plan = plans.get(q.get('command_observation_ns')) if q['reason'] in DEPTH_STOPS else None
        if plan is not None and plan['action'] in TRANSLATIONS:
            catches.append(counterfactual(q, plan, native, ts, ct, sep, distance))
    row.update(reach_from(ep, native[:, :2]), pose_loss=is_pose_loss(ev), clearance_hist_all=hist(sep), clearance_hist_translating=hist(sep[moving_sample]),
               separation_min_m=float(sep.min()), separation_min_translating_m=float(sep[moving_sample].min()) if moving_sample.any() else None,
               translation_windows=sum(r['action'] in TRANSLATIONS for r in plans.values()), catches=catches)
    walls = [dict(center=w['centre_xyz'][:2], size=w['size_xyz'][:2], yaw=w['yaw_rad']) for w in spec['geometry']['wall_boxes']]
    geometry = lambda xy: ReferenceGeometry(walls, [[-2.1, -2.1], [3.4, 3.4]], xy, radius_m=.46, clearance_m=.005, resolution_m=.02)
    targets = {'OUTBOUND': np.asarray(ep['beacon_xy_world'], float), 'RETURN': np.asarray(ep['home_se2_world'][:2], float)}
    remaining = {leg: remaining_fn(geometry(xy), native, t) for leg, xy in targets.items()}
    beacon = next((a['frame']*.1 for a in ev['arrivals'] if a['phase'] == 'OUTBOUND' and a['passed']), None)
    dt = np.diff(t, append=t[-1])
    turning = ~translating & (applied[:, 2] != 0)

    def split(lo, hi):
        m = (t >= lo) & (t < hi)
        span = max(hi-lo, 1e-9)
        return float(dt[m & translating].sum()/span), float(dt[m & turning].sum()/span)

    def behaviour(lo, hi, leg, threshold):
        a, b = remaining[leg](lo), remaining[leg](hi)
        progress = None if a is None or b is None else a-b
        move, turn = split(lo, hi)
        kind = ('progressing' if progress is not None and progress >= threshold else
                ('turning in place' if turn >= TURN_FRACTION else 'holding') if move < MOVE_FRACTION else 'wandering')
        return kind, progress, move, turn
    legs = [('OUTBOUND', 0., beacon if beacon is not None else end)] + ([('RETURN', beacon, end)] if beacon is not None else [])
    shares = Counter()
    for leg, lo, hi in legs:
        for w in np.arange(lo, hi-1e-8, WINDOW_S):
            shares[behaviour(w, min(w+WINDOW_S, hi), leg, WINDOW_PROGRESS_M)[0]] += min(w+WINDOW_S, hi)-w
    row.update(time_s=end, window_time_s={k: float(v) for k, v in shares.items()}, translating_fraction=split(0., end)[0],
               turning_fraction=split(0., end)[1])
    if not row['success']:
        leg, lo, hi = legs[-1]
        kind, progress, move, turn = behaviour(max(lo, end-FINAL_S), end, leg, FINAL_PROGRESS_M)
        distance = np.linalg.norm(native[(t >= lo) & (t <= hi), :2]-targets[leg], axis=1)
        actual = ev['return_leg' if leg == 'RETURN' else 'outbound']['actual_path_m']
        row.update(mechanism='pose loss' if row['pose_loss'] else 'other error' if row['error'] else kind, behaviour=kind, active_leg=leg,
                   location='at target' if distance[-1] <= AT_TARGET_M else 'en route',
                   closest_to_target_m=float(distance.min()), final_distance_to_target_m=float(distance[-1]),
                   time_within_25cm_s=float((distance <= .25).sum()*np.median(np.diff(t))),
                   final_progress_m=progress, final_translating=move, final_turning=turn, remaining_m=remaining[leg](end),
                   leg_detour=actual/ep['shortest_outbound_m'] if actual else None)
    return row


def assignments(cohort):
    config = json.loads((BASE/'dev_cohorts'/cohort/'config.json').read_text())
    return [(cohort, job[4]) for job in config['plan'] if (BASE/'runs'/job[4]/'episode_evaluation.json').exists()]


def spec_of(cohort):
    return json.loads((BASE/'dev_cohorts'/cohort/'config.json').read_text()).get('forecast_degradation') or 'none'


def order(cohort):
    kind, _, value = spec_of(cohort).partition(':')
    return (('none', 'noise', 'scale', 'fwdscale', 'turnscale').index(kind), float(value or 0))


def percentiles(rows, key, qs=(5, 50)):
    counts = np.zeros(len(EDGES)-1)
    for r in rows:
        for i, n in r[key]:
            counts[i] += n
    if not counts.sum():
        return [None]*len(qs)
    cumulative = np.cumsum(counts)/counts.sum()
    return [float(EDGES[np.searchsorted(cumulative, q/100)]+.0005) for q in qs]


def table(rows, cohorts):
    f = lambda v, d=2: '-' if v is None else f'{v:.{d}f}'
    cm = lambda v: '-' if v is None else f'{100*v:.1f}'
    ci = lambda k, n: f'{k}/{n} · {k/n:.2f} ({wilson(k, n)[0]:.2f}–{wilson(k, n)[1]:.2f})'
    success = [f'**Success and failure mechanism per cohort. {LABEL}**', '',
               'Strict = the frozen arrival rule. Reached = the true base came within 0.25 m of the beacon and later within 0.25 m of home; the '
               'harness starts the return only after a confirmed beacon arrival, so reached beacon (outbound only) is also shown. Pose loss = the '
               'visual tracker failed and ended the mission: a shared-system failure, not a forecast failure. Forecast-attributable failures are '
               'labelled on the final 120 s of the active leg: progressing (target geodesic fell ≥ 0.25 m), else no-move (translating < 10% of the '
               'time: turning in place if turn-only ≥ 25%, else holding), else wandering (translated, got no closer). At target = ended within '
               '0.25 m of the active target. Time shares label every 30-s window of every mission the same way (0.1 m).', '',
               '| Condition | Done | Strict success (Wilson 95%) | Reached beacon and home (Wilson 95%) | Reached beacon | Pose loss (shared system) '
               '| Forecast-attributable failures: holding · turning in place · wandering · progressing | Of those, ended at target '
               '| Time share: progressing · holding · turning in place · wandering | Translating time |',
               '|---|---:|---|---|---:|---:|---|---:|---|---:|']
    safety = [f'**Safety and the last-moment depth stop per cohort. {LABEL}**', '',
              'Separation = native robot-to-wall separation (lower bound) sampled at 500 Hz. Depth stop = command window vetoed at dispatch by the '
              'current depth image (forecast-independent), split by the planned command; vetoes of planned holds stop nothing. Would-be outcome of '
              'each catch on a planned translation without the stop: the remaining motion of the planned move taken from C1\'s clean forecast '
              '(about 1 cm from physics), separation tracked against the true walls (contact ≤ 0, hard < 5 mm, operating < 20 mm).', '',
              '| Condition | Contacts (samples · missions) | Hard · operating violations | Separation, whole mission: worst · p5 · median (cm) '
              '| Separation while translating: worst · p5 · median (cm) | Depth stops on planned translations (per 100 translation windows) · turns · holds '
              '| Translation catches without the stop: contact · hard · operating · none | Other dispatch vetoes |',
              '|---|---|---|---|---|---|---|---:|']
    filters = [f'**Planner-stage filters (these use the forecast). {LABEL}**', '',
               'Forecast-rejected moves = of the five moves, how many the forecast-based memory clearance check rejected per decision. Planner '
               'holds by the frozen reader\'s hold records.', '',
               '| Condition | Forecast-rejected moves per decision (of 5) · all 5 rejected | Planner holds per decision '
               '| Holds: arrival settling · on score · forecast clearance · coverage rule |',
               '|---|---|---:|---|']
    for c in cohorts:
        rs = [r for r in rows if r['cohort'] == c]
        if not rs:
            continue
        n = len(rs)
        failed = [r for r in rs if not r['success'] and not r['pose_loss']]
        mech = Counter(r['mechanism'] for r in failed)
        share = Counter()
        stops, holds = Counter(), Counter()
        for r in rs:
            share.update(r['window_time_s'])
            stops.update(r['depth_stops'])
            holds.update(r['holds'])
        total = sum(share.values()) or 1
        success.append(f"| {spec_of(c)} | {n}/20 | {ci(sum(r['success'] for r in rs), n)} | {ci(sum(r['reached'] for r in rs), n)} | "
                       f"{sum(r['reached_beacon'] for r in rs)}/{n} | {sum(r['pose_loss'] for r in rs)} | "
                       f"{' · '.join(str(mech[m]) for m in BEHAVIOURS)} | {sum(r['location'] == 'at target' for r in failed)}/{len(failed)} | "
                       f"{' · '.join(f'{share[b]/total:.2f}' for b in ('progressing', 'holding', 'turning in place', 'wandering'))} | "
                       f"{st.mean(r['translating_fraction'] for r in rs):.2f} |")
        would = Counter(k['would_be'] for r in rs for k in r['catches'])
        windows = sum(r['translation_windows'] for r in rs)
        p5, p50 = percentiles(rs, 'clearance_hist_all')
        q5, q50 = percentiles(rs, 'clearance_hist_translating')
        worst_moving = min((r['separation_min_translating_m'] for r in rs if r['separation_min_translating_m'] is not None), default=None)
        safety.append(f"| {spec_of(c)} | {sum(r['contacts'] for r in rs)} · {sum(r['contacts'] > 0 for r in rs)} | "
                      f"{sum(r['hard'] for r in rs)} · {sum(r['operating'] for r in rs)} | "
                      f"{cm(min(r['separation_min_m'] for r in rs))} · {cm(p5)} · {cm(p50)} | {cm(worst_moving)} · {cm(q5)} · {cm(q50)} | "
                      f"{stops['translate']} ({100*stops['translate']/max(1, windows):.2f}) · {stops['turn']} · {stops['hold']} | "
                      f"{would['contact']} · {would['hard violation']} · {would['operating violation']} · {would['none']} | "
                      f"{sum(r['other_vetoes'] for r in rs)} |")
        rejected = [x for r in rs for x in r['forecast_rejected_moves']]
        all_holds = sum(holds.values()) or 1
        filters.append(f"| {spec_of(c)} | {st.mean(rejected):.2f} · {sum(x == 5 for x in rejected)/len(rejected):.2f} | "
                       f"{sum(holds.values())/max(1, sum(r['decisions'] for r in rs)):.2f} | "
                       f"{' · '.join(f'{holds[g]/all_holds:.2f}' for g in ('arrival settling', 'on score', 'forecast clearance', 'coverage rule'))} |")
    catches = sorted(((r, k) for r in rows for k in r['catches']), key=lambda x: (order(x[0]['cohort']), x[0]['maze'], x[1]['time_s']))
    catch_lines = ['', f'**Every depth stop on a planned translation, with its would-be outcome. {LABEL}**', '',
                   'True path = C1\'s clean forecast of the planned move from the catch to 800 ms after the decision\'s observation; believed path = '
                   'the degraded forecast the planner scored. Uniform scale 0.25× and 0.5× are the safety test (under-prediction).', '',
                   '| Condition | Maze | Time (s) | Planned move | Separation at catch (cm) | Travel to forecast end: true · believed (cm) '
                   '| Min separation without the stop: true path · believed path (cm) | Would-be outcome |',
                   '|---|---:|---:|---|---:|---|---|---|']
    for r, k in catches:
        catch_lines.append(f"| {spec_of(r['cohort'])} | {r['maze']} | {k['time_s']:.1f} | {k['action']} | {cm(k['separation_at_catch_m'])} | "
                           f"{cm(k['true_travel_m'])} · {cm(k['believed_travel_m'])} | {cm(k['true_min_separation_m'])} · {cm(k['believed_min_separation_m'])} | "
                           f"{k['would_be']} |")
    if not catches:
        catch_lines.append('| (none) | | | | | | | |')
    failed = sorted((r for r in rows if not r['success']), key=lambda r: (order(r['cohort']), r['maze']))
    detail = ['', f'**Every failed mission. {LABEL}**', '',
              '| Condition | Maze | Mechanism (behaviour before loss) | Where | Active leg | Reached beacon · home | Remaining at end (m) '
              '| Closest to target (m) · time within 25 cm (s) | Progress in final 120 s (m) | Final 120 s: translating · turning only '
              '| Leg path / shortest | Contacts | Min clearance (cm) |',
              '|---|---:|---|---|---|---|---:|---|---:|---|---:|---:|---:|']
    for r in failed:
        mech = f"pose loss, shared system ({r['behaviour']})" if r['pose_loss'] else r['mechanism']
        detail.append(f"| {spec_of(r['cohort'])} | {r['maze']} | {mech} | {r['location']} | {r['active_leg'].lower()} | "
                      f"{'yes' if r['reached_beacon'] else 'no'} · {'yes' if r['reached'] else 'no'} | {f(r['remaining_m'])} | "
                      f"{r['closest_to_target_m']:.2f} · {r['time_within_25cm_s']:.0f} | {f(r['final_progress_m'])} | "
                      f"{f(r['final_translating'])} · {f(r['final_turning'])} | {f(r['leg_detour'], 1)} | {r['contacts']} | {cm(r['min_clearance_m'])} |")
    return '\n'.join(success+['']+safety+['']+filters+catch_lines+detail)


def main(cohorts, out_json, out_md, workers):
    cohorts = sorted(cohorts or [p.name for p in (BASE/'dev_cohorts').glob('sens_*') if (p/'config.json').exists()], key=order)
    jobs = [j for c in cohorts for j in assignments(c)]
    with Pool(workers) as pool:
        rows = pool.map(mission, jobs, chunksize=1)
    text = table(rows, cohorts)
    print(text)
    if out_json:
        Path(out_json).write_text(json.dumps(dict(label=LABEL, rows=rows), indent=1, default=float)+'\n')
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
