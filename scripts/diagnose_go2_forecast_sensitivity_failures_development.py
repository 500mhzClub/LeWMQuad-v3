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

Safety, per mission: disallowed contact samples, hard (5 mm) and operating (20 mm) clearance
violations and the minimum wall separation (native ground truth, the frozen reader).

Last-moment depth stop: the dispatch check of the requested-speed disk connector (0.45 m)
against the current depth image (CURRENT_OBSERVED_OBSTACLE_VETO, or
CURRENT_STOPPING_MARGIN_VETO where that layer runs). It does not use the forecast. One veto
latches the rest of its command window, so one veto record = one stopped command window.
Vetoes of windows whose plan was a hold stop nothing (the robot parked within 0.45 m of a
wall fails the disk check even at zero speed), so stops are counted on windows that planned a
move. Other dispatch vetoes (stale observation, late first dispatch) are counted separately.

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
        t = f['timestamp_s']-1.5
        native = f['base_pose_world'].copy()
        applied = f['applied_command'].copy()
    end = float(t[-1])
    walls = [dict(center=w['centre_xyz'][:2], size=w['size_xyz'][:2], yaw=w['yaw_rad']) for w in spec['geometry']['wall_boxes']]
    geometry = lambda xy: ReferenceGeometry(walls, [[-2.1, -2.1], [3.4, 3.4]], xy, radius_m=.46, clearance_m=.005, resolution_m=.02)
    targets = {'OUTBOUND': np.asarray(ep['beacon_xy_world'], float), 'RETURN': np.asarray(ep['home_se2_world'][:2], float)}
    remaining = {leg: remaining_fn(geometry(xy), native, t) for leg, xy in targets.items()}
    beacon = next((a['frame']*.1 for a in ev['arrivals'] if a['phase'] == 'OUTBOUND' and a['passed']), None)
    dt = np.diff(t, append=t[-1])
    translating = np.any(applied[:, :2] != 0, axis=1)
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
        row.update(mechanism='pose loss' if row['error'] else kind, behaviour=kind, active_leg=leg,
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


def table(rows, cohorts):
    f = lambda v, d=2: '-' if v is None else f'{v:.{d}f}'
    lines = [f'**Failure mechanism per cohort. {LABEL}**', '',
             'Failed missions labelled on the final 120 s of the active leg: progressing (target geodesic fell ≥ 0.25 m), else no-move '
             '(translating < 10% of the time: turning in place if turn-only ≥ 25%, else holding), else wandering (translated, got no closer); '
             'pose loss = the visual tracker failed and ended the mission. At target = ended within 0.25 m of the active target without a '
             'confirmed arrival. Time shares label every 30-s window of every mission the same way (0.1 m).', '',
             '| Condition | Done | Success | Failures: holding · turning in place · wandering · progressing · pose loss | Of failures, ended at target '
             '| Time share: progressing · holding · turning in place · wandering | Translating time |',
             '|---|---:|---|---|---:|---|---:|']
    safety = [f'**Safety, depth stops and planner filters per cohort. {LABEL}**', '',
              'Depth stop = command window vetoed at dispatch by the current depth image (forecast-independent); counted on windows that '
              'planned a move (vetoes of planned holds stop nothing). Forecast-rejected moves = of the five moves, how many the forecast-based '
              'memory clearance check rejected per decision. Planner holds by the frozen reader\'s hold records.', '',
              '| Condition | Contacts (samples · missions) | Hard · operating violations | Min clearance: worst · median | Depth stops on planned moves '
              '(per 100 move windows) · on planned holds | Other dispatch vetoes | Forecast-rejected moves per decision (of 5) · all 5 rejected '
              '| Planner holds per decision | Holds: arrival settling · on score · forecast clearance · coverage rule |',
              '|---|---|---|---|---|---:|---|---:|---|']
    for c in cohorts:
        rs = [r for r in rows if r['cohort'] == c]
        if not rs:
            continue
        k = sum(r['success'] for r in rs)
        lo, hi = wilson(k, len(rs))
        failed = [r for r in rs if not r['success']]
        mech = Counter(r['mechanism'] for r in failed)
        share = Counter()
        for r in rs:
            share.update(r['window_time_s'])
        total = sum(share.values()) or 1
        lines.append(f"| {spec_of(c)} | {len(rs)}/20 | {k}/{len(rs)} · {k/len(rs):.2f} ({lo:.2f}–{hi:.2f}) | "
                     f"{' · '.join(str(mech[m]) for m in (*BEHAVIOURS, 'pose loss'))} | {sum(r['location'] == 'at target' for r in failed)}/{len(failed)} | "
                     f"{' · '.join(f'{share[b]/total:.2f}' for b in ('progressing', 'holding', 'turning in place', 'wandering'))} | "
                     f"{st.mean(r['translating_fraction'] for r in rs):.2f} |")
        stops = Counter()
        holds = Counter()
        for r in rs:
            stops.update(r['depth_stops'])
            holds.update(r['holds'])
        moving = stops['translate']+stops['turn']
        rejected = [n for r in rs for n in r['forecast_rejected_moves']]
        clear = [r['min_clearance_m'] for r in rs if r['min_clearance_m'] is not None]
        decisions = sum(r['decisions'] for r in rs) or 1
        all_holds = sum(holds.values()) or 1
        safety.append(f"| {spec_of(c)} | {sum(r['contacts'] for r in rs)} · {sum(r['contacts'] > 0 for r in rs)} | "
                      f"{sum(r['hard'] for r in rs)} · {sum(r['operating'] for r in rs)} | {100*min(clear):.1f} · {100*st.median(clear):.1f} cm | "
                      f"{moving} ({100*moving/max(1, sum(r['move_windows'] for r in rs)):.2f}; translate {stops['translate']}, turn {stops['turn']}) · "
                      f"{stops['hold']} | {sum(r['other_vetoes'] for r in rs)} | "
                      f"{st.mean(rejected):.2f} · {sum(n == 5 for n in rejected)/len(rejected):.2f} | {sum(holds.values())/decisions:.2f} | "
                      f"{' · '.join(f'{holds[g]/all_holds:.2f}' for g in ('arrival settling', 'on score', 'forecast clearance', 'coverage rule'))} |")
    failed = sorted((r for r in rows if not r['success']), key=lambda r: (order(r['cohort']), r['maze']))
    detail = ['', f'**Every failed mission. {LABEL}**', '',
              '| Condition | Maze | Mechanism (behaviour before loss) | Where | Active leg | Remaining at end (m) | Closest to target (m) · time within 25 cm (s) '
              '| Progress in final 120 s (m) | Final 120 s: translating · turning only | Leg path / shortest | Depth stops on moves · on holds | Contacts | Min clearance |',
              '|---|---:|---|---|---|---:|---|---:|---|---:|---|---:|---:|']
    for r in failed:
        mech = r['mechanism'] if r['mechanism'] != 'pose loss' else f"pose loss ({r['behaviour']})"
        stops = r['depth_stops']
        detail.append(f"| {spec_of(r['cohort'])} | {r['maze']} | {mech} | {r['location']} | {r['active_leg'].lower()} | {f(r['remaining_m'])} | "
                      f"{r['closest_to_target_m']:.2f} · {r['time_within_25cm_s']:.0f} | {f(r['final_progress_m'])} | "
                      f"{f(r['final_translating'])} · {f(r['final_turning'])} | {f(r['leg_detour'], 1)} | "
                      f"{stops.get('translate', 0)+stops.get('turn', 0)} · {stops.get('hold', 0)} | {r['contacts']} | {100*r['min_clearance_m']:.1f} cm |")
    return '\n'.join(lines+['']+safety+detail)


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
