"""PRELIMINARY: safety as an explicit error budget, checked against every close approach (Andrew, 2 October 2026).

    min body clearance >= (disc radius - max body reach) - e_f - e_p - e_m

- Disc radius: every planner-stage clearance check requires the forecast centre path to stay
  more than 0.45 m from remembered obstacle cells (0.48 m with the turn/translation reserve;
  0.45 m in the reserve-recovery modes).
- Max body reach: 42.5 cm, the largest horizontal distance of any of the 27 collision
  primitives from the base centre over all moving samples (close-approach analysis).
- e_f, centre-position forecast error: the forecast centre path the check used (the degraded
  forecast where there is one) against the true centre path, both anchored at the true pose
  at the decision's observation; per decision, the largest error over the 800-ms horizon.
- e_p, tracker error: the registered visual pose estimate against the true pose at the
  decision, in the initial body frame: horizontal position error plus the heading error
  times the forecast path's largest extent (its effect on the checked path).
- e_m, map error: remembered cells are 1-cm squares and distances are measured to the
  squares, so quantisation and inflation contribute nothing that is not conservative; what
  remains is walls misplaced by the pose error when they were observed, or never observed.
  Measured per decision as e_pm - e_p (clipped at 0), where e_pm = remembered clearance of
  the executed move's forecast path (logged) minus the true clearance of the same path
  placed at the true pose against the true walls.

Check against data: for each close approach (worst 5 per mission while moving, >= 1 s apart),
the bound from that decision's actual e_f (at the approach time), e_p and e_m, with the
generic disc (0.45 m) and reach (42.5 cm). It assumes the executing command came from a
decision whose move passed the check (remembered clearance > 0.45 m) and that the approach
falls within that decision's 800-ms forecast. The executing move is the last dispatched non-zero
command: after a hold is requested, the slew limiter winds the previous move down over the
next ticks, and those approaches are attributed to that move. Violations are reported against the reader's
separation lower bound and its upper bound (a violation of the upper bound is definite). The
tight form, true clearance of the checked path minus e_f minus the body's largest reach at
that instant, is a strict geometric lower bound; its slack shows how much of the budget's
slack is the remembered clearance above 0.45 m and the reach below 42.5 cm.

Usage: analyse_go2_forecast_sensitivity_error_budget_development.py [--cohorts sens_...] [--json OUT] [--markdown OUT] [--workers 3]
"""
import argparse
from collections import Counter
import json
import math
from multiprocessing import Pool
from pathlib import Path
import statistics as st

import numpy as np

from lewm.physical_execution_development import rotation_xyzw
from scripts.analyse_go2_forecast_sensitivity_close_approaches_development import PER_MISSION, SPACING_S, Walls, move_type
from scripts.diagnose_go2_forecast_sensitivity_failures_development import ACTIONS, BASE, LABEL, assignments, order, spec_of, wall_distance

DISC_M, RESERVE_DISC_M, MAX_REACH_M = .45, .48, .425
HORIZON_S, STEP_S = .8, .1
PASSED = 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED'
TURNS = ('left_turn', 'right_turn')
DIRECTIONS_64 = np.array([[math.cos(a), math.sin(a), 0.] for a in np.linspace(0, 2*math.pi, 64, endpoint=False)])


def planar_yaw(q):
    return math.atan2(2*(q[6]*q[5]+q[3]*q[4]), 1-2*(q[4]**2+q[5]**2))


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def max_reach(walls, pose, joints):
    """Largest horizontal extent of the body from the base centre in any of 64 directions (exact support)."""
    Q = rotation_xyzw(pose[3:])
    support = walls.model.supports(joints, DIRECTIONS_64@Q)
    return float(max(np.asarray(s['upper']).max() for s in support['shapes']))


def mission(args):
    cohort, assignment = args
    root = BASE/'runs'/assignment
    read = lambda n: json.loads((root/n).read_text())
    spec = read('specification.json')
    distance = wall_distance(spec['geometry']['wall_boxes'])
    walls = Walls(spec['geometry']['wall_boxes'])
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as f:
        ts, P, joints, applied = f['timestamp_s'].copy(), f['base_pose_world'].copy(), f['joint_position'].copy(), f['applied_command'].copy()
    with np.load(root/'native_clearance_summary_arrays.npz', allow_pickle=False) as c:
        ct, lower, upper = c['timestamp_s'].copy(), c['separation_lower_m'].copy(), c['separation_upper_m'].copy()
    at = lambda t: min(int(np.searchsorted(ts, t)), len(ts)-1)
    poses = {p['frame']: p['registered_pose'] for p in read('poses.json') if p.get('registered_pose')}
    i0 = at(1.5)
    p0, R0 = P[i0, :3], rotation_xyzw(P[i0, 3:])
    plans = {r['measured_ns']: r for r in read('planning.json') if 'selection' in r}
    requests = read('requests.json')
    executed = {q['command_observation_ns'] for q in requests if q['reason'] == PASSED and any(q['requested_command'])}

    def path_clearance(points):
        dense = np.vstack([a+(b-a)*np.linspace(0, 1, 12)[:, None] for a, b in zip(points[:-1], points[1:])])
        return float(distance(dense).min())

    def decision(plan):
        t_o = plan['measured_ns']/1e9
        i = at(t_o)
        a = ACTIONS.index(plan['action'])
        mc = plan.get('motion_correction') or {}
        if mc.get('dev_degraded_forecast_xy_yaw') is not None:  # sensitivity runs: the degraded forecast is what the check used
            forecast = np.asarray(mc['dev_degraded_forecast_xy_yaw'], float)[a, :, :2]
        elif mc.get('applied_prediction_after_yaw_ablation') is not None:  # the prediction the planner applied (C1, C3, C4)
            forecast = np.asarray(mc['applied_prediction_after_yaw_ablation'], float)[a, :, :2]
        elif mc.get('command_history_forecast_xy_yaw') is not None:
            forecast = np.asarray(mc['command_history_forecast_xy_yaw'], float)[a, :, :2]
        else:
            return None  # no forecast logged (C2 is reactive)
        f = np.vstack((np.zeros(2), forecast))
        yaw_o = planar_yaw(P[i])
        placed = P[i, :2]+f[:, :2]@rot(yaw_o).T
        true_path = np.array([P[at(t_o+k*STEP_S), :2] for k in range(9)])
        e_f = float(np.linalg.norm(placed-true_path, axis=1).max())
        pose = poses.get(plan['frame'])
        e_p = heading = None
        if pose is not None:
            true_rel = R0.T@(P[i, :3]-p0)
            R_true = R0.T@rotation_xyzw(P[i, 3:])
            R_est = np.asarray(pose['rotation_initial_body_from_current_body'])
            heading = wrap(math.atan2(R_est[1, 0], R_est[0, 0])-math.atan2(R_true[1, 0], R_true[0, 0]))
            lever = float(np.linalg.norm(f[:, :2], axis=1).max())
            e_p = float(np.linalg.norm(np.asarray(pose['position_initial_body_m'][:2])-true_rel[:2]))+abs(heading)*lever
        selection = plan['selection']
        memory = {c['action']: c for c in selection.get('memory_forecast_candidates') or []}
        row = memory.get(plan['action']) or {}
        remembered = row.get('minimum_predicted_path_clearance_m')
        margin = (selection.get('dev_clearance_margin') or {}).get('margin_m') or 0.
        if remembered is not None:
            remembered += margin  # calibrated-margin runs log distances reduced by the margin
        brk = selection.get('dev_terminal_spin_break')
        intervention = bool(selection.get('dev_backup') or selection.get('dev_backup_aborted') or selection.get('dev_deadlock_escape')
                            or (brk and not (brk.get('kind') == 'position_scoring_burst' and brk.get('distance_m') == 0.))
                            or (selection.get('clearance_turn') or {}).get('event') == 'DEV_LATCH_TIMEOUT_RELEASED')
        true_clearance = path_clearance(placed)
        e_pm = None if remembered is None else remembered-true_clearance
        return dict(t_o=t_o, action=plan['action'], forecast=f, placed=placed, yaw_o=yaw_o, i=i, e_f=e_f, e_p=e_p,
                    heading_deg=None if heading is None else math.degrees(heading), remembered=remembered, true_clearance=true_clearance,
                    e_pm=e_pm, e_m=None if e_pm is None or e_p is None else max(0., e_pm-e_p), mode=row.get('clearance_check_mode'),
                    intervention=intervention, margin=margin)
    decisions = []
    for ns in sorted(executed):
        plan = plans.get(ns)
        if plan is not None and plan['action'] != 'hold':
            d = decision(plan)
            if d is None:
                continue
            decisions.append({k: d[k] for k in ('action', 'e_f', 'e_p', 'heading_deg', 'remembered', 'true_clearance', 'e_pm', 'e_m', 'mode',
                                                'intervention', 'margin')})
    # Close approaches, as in the close-approach analysis.
    index = np.clip(np.searchsorted(ts, ct), 0, len(ts)-1)
    moving = np.any(applied[index] != 0, axis=1)
    chosen = []
    for j in np.flatnonzero(moving)[np.argsort(lower[moving], kind='stable')]:
        if all(abs(ct[j]-ct[k]) >= SPACING_S for k in chosen):
            chosen.append(j)
            if len(chosen) == PER_MISSION:
                break
    request_s = np.array([q['simulator_ns'] for q in requests])/1e9
    approaches = []
    for j in chosen:
        t = float(ct[j])
        i = index[j]
        q = requests[max(0, int(np.searchsorted(request_s, t, side='right'))-1)]
        # The executing move is the last dispatched non-zero command: after a hold is requested,
        # the slew limiter winds the previous move's command down over the following ticks.
        last = next((requests[k] for k in range(int(np.searchsorted(request_s, t, side='right'))-1, -1, -1)
                     if requests[k]['reason'] == PASSED and any(requests[k]['requested_command'])), None)
        plan = plans.get(last.get('command_observation_ns')) if last is not None else None
        winding_down = not any(q['requested_command'])
        _, link, point, _ = walls.closest(P[i], joints[i])
        out = dict(time_s=round(t-1.5, 3), separation_lower_m=float(lower[j]), separation_upper_m=float(upper[j]), link=link,
                   move=move_type(applied[i]), dispatch_reason=q['reason'], centre_to_wall_m=float(distance(P[i, :2])[0]),
                   reach_m=walls.reach_toward(P[i], joints[i], point[:2]-P[i, :2]), max_reach_m=max_reach(walls, P[i], joints[i]),
                   winding_down=winding_down)
        if plan is None or plan['action'] == 'hold':
            out.update(status='executing command not from a checked move', planned_action=None if plan is None else plan['action'])
            approaches.append(out)
            continue
        d = decision(plan)
        if d is None:
            out.update(status='no forecast logged')
            approaches.append(out)
            continue
        s = (t-d['t_o'])/STEP_S
        within = 0 <= s <= HORIZON_S/STEP_S
        sc = min(max(s, 0.), 8.)
        point_f = np.array([np.interp(sc, np.arange(9.), d['forecast'][:, k]) for k in range(2)])  # centre only
        q_star = P[d['i'], :2]+rot(d['yaw_o'])@point_f
        e_f_t = float(np.linalg.norm(q_star-P[i, :2]))
        checked = d['remembered'] is None or d['remembered'] > DISC_M
        e_p, e_m = d['e_p'], d['e_m']
        bound = None if e_p is None else (DISC_M-MAX_REACH_M)-e_f_t-e_p-(e_m or 0.)
        tight = d['true_clearance']-out['max_reach_m']-e_f_t
        out.update(status='checked move within forecast' if checked and within else
                   'checked move beyond its 800-ms forecast' if checked else 'move with remembered clearance <= 0.45 m',
                   planned_action=d['action'], age_s=round(t-d['t_o'], 3), check_mode=d['mode'], remembered_m=d['remembered'],
                   true_path_clearance_m=d['true_clearance'], e_f_m=e_f_t, e_p_m=e_p, heading_deg=d['heading_deg'], e_pm_m=d['e_pm'], e_m_m=e_m,
                   bound_m=bound, tight_bound_m=tight)
        approaches.append(out)
    return dict(cohort=cohort, assignment=assignment, controller=read('config.json').get('controller'), decisions=decisions, approaches=approaches)


def q(values, p):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, p)) if values else None


def table(results, cohorts):
    cm = lambda v, d=1: '-' if v is None else f'{100*v:.{d}f}'
    lines = [f'**Safety error budget: min body clearance ≥ (disc − max reach) − e_f − e_p − e_m. {LABEL}**', '',
             f'Disc 45 cm (48 cm with the turn/translation reserve), max reach {100*MAX_REACH_M:.1f} cm: static margin '
             f'{100*(DISC_M-MAX_REACH_M):.1f} cm ({100*(RESERVE_DISC_M-MAX_REACH_M):.1f} cm at full reserve). Terms over every executed moving decision: '
             'e_f = largest centre forecast error over the 800-ms check horizon (forecast the check used vs truth, both from the true pose); '
             'e_p = tracker position error plus heading error times the path extent; e_m = map error beyond the tracker\'s share '
             '(remembered minus true clearance of the same path, minus e_p, clipped at 0); 1-cm cells measured to their squares add no quantisation error. '
             'Budget at p95 = static margin minus the three p95 terms.', '',
             '| Condition | Decisions | e_f: median · p95 · max (cm) | e_f in-place turns · forward/arcs, p95 (cm) | e_p: median · p95 · max (cm) '
             '| Heading error p95 (deg) | e_m: median · p95 · max (cm) | Remembered − true path clearance: median · p95 (cm) | Budget at p95 terms (cm) '
             '| Observed worst approach (cm) |',
             '|---|---:|---|---|---|---:|---|---|---:|---:|']
    for c in cohorts:
        ds = [d for r in results if r['cohort'] == c for d in r['decisions']]
        if not ds:
            continue
        ef, ep, em = [d['e_f'] for d in ds], [d['e_p'] for d in ds], [d['e_m'] for d in ds]
        turn = [d['e_f'] for d in ds if d['action'] in TURNS]
        trans = [d['e_f'] for d in ds if d['action'] not in TURNS]
        budget = None if q(ep, 95) is None else (DISC_M-MAX_REACH_M)-q(ef, 95)-q(ep, 95)-q(em, 95)
        worst = min((a['separation_lower_m'] for r in results if r['cohort'] == c for a in r['approaches']), default=None)
        heading95 = q([abs(d['heading_deg']) for d in ds if d['heading_deg'] is not None], 95)
        lines.append(f"| {spec_of(c)} | {len(ds)} | {cm(q(ef, 50))} · {cm(q(ef, 95))} · {cm(max(ef))} | {cm(q(turn, 95))} · {cm(q(trans, 95))} | "
                     f"{cm(q(ep, 50))} · {cm(q(ep, 95))} · {cm(max(v for v in ep if v is not None))} | "
                     f"{'-' if heading95 is None else f'{heading95:.2f}'} | "
                     f"{cm(q(em, 50))} · {cm(q(em, 95))} · {cm(max((v for v in em if v is not None), default=None))} | "
                     f"{cm(q([d['e_pm'] for d in ds], 50))} · {cm(q([d['e_pm'] for d in ds], 95))} | {cm(budget)} | {cm(worst)} |")
    lines += ['', f'**Each close approach against the budget\'s lower bound from that decision\'s actual errors. {LABEL}**', '',
              'Bound = (45 − 42.5 cm) − e_f at the approach time − e_p − e_m of the decision whose command was executing. It applies when that '
              'decision\'s move passed the check (remembered clearance > 45 cm) and the approach falls within its 800-ms forecast. Violation = observed '
              'separation below the bound (lower-bound separation; definite if the upper bound is also below). Slack = observed minus bound. Tight '
              'bound = true clearance of the checked path − e_f − the body\'s actual largest reach at that instant (any direction).', '',
              '| Condition | Approaches | Bound applies (of which in slew wind-down) | Beyond the forecast · not from a checked move | Violations: lower-bound sep · definite '
              '| Slack: min · median (cm) | Tight-bound slack: min · median (cm) | Approach terms, median: e_f · e_p · e_m (cm) |',
              '|---|---:|---:|---|---|---|---|---|']
    violations = []
    for c in cohorts:
        aps = [a for r in results if r['cohort'] == c for a in r['approaches']]
        if not aps:
            continue
        applies = [a for a in aps if a['status'] == 'checked move within forecast' and a.get('bound_m') is not None]
        bad = [a for a in applies if a['separation_lower_m'] < a['bound_m']]
        definite = [a for a in bad if a['separation_upper_m'] < a['bound_m']]
        violations += [(c, a) for a in bad]
        status = Counter(a['status'] for a in aps)
        slack = [a['separation_lower_m']-a['bound_m'] for a in applies]
        tight = [a['separation_upper_m']-a['tight_bound_m'] for a in applies]
        lines.append(f"| {spec_of(c)} | {len(aps)} | {len(applies)} ({sum(a['winding_down'] for a in applies)}) | {status['checked move beyond its 800-ms forecast']} · "
                     f"{status['executing command not from a checked move']+status['move with remembered clearance <= 0.45 m']} | "
                     f"{len(bad)} · {len(definite)} | {cm(min(slack, default=None))} · {cm(q(slack, 50))} | {cm(min(tight, default=None))} · {cm(q(tight, 50))} | "
                     f"{cm(q([a['e_f_m'] for a in applies], 50))} · {cm(q([a['e_p_m'] for a in applies], 50))} · {cm(q([a['e_m_m'] for a in applies], 50))} |")
    lines += ['', f'**Violations of the bound. {LABEL}**', '']
    if violations:
        lines += ['| Condition | Time (s) | Move | Separation lower · upper (cm) | Bound (cm) | e_f · e_p · e_m (cm) | Remembered · true path clearance (cm) | Reach (cm) |',
                  '|---|---:|---|---|---:|---|---|---:|']
        for c, a in violations:
            lines.append(f"| {spec_of(c)} | {a['time_s']:.1f} | {a['move']} | {cm(a['separation_lower_m'])} · {cm(a['separation_upper_m'])} | {cm(a['bound_m'])} | "
                         f"{cm(a['e_f_m'])} · {cm(a['e_p_m'])} · {cm(a['e_m_m'])} | {cm(a['remembered_m'])} · {cm(a['true_path_clearance_m'])} | {cm(a['reach_m'])} |")
    else:
        lines.append('None: every approach the bound applies to has observed separation at or above it.')
    other = [(c, a) for c in cohorts for r in results if r['cohort'] == c for a in r['approaches'] if a['status'] != 'checked move within forecast']
    if other:
        lines += ['', f'**Approaches the bound does not apply to. {LABEL}**', '',
                  '| Condition | Time (s) | Move | Status | Planned action | Age of decision (s) | Separation (cm) | Centre to wall (cm) |',
                  '|---|---:|---|---|---|---:|---:|---:|']
        for c, a in other:
            lines.append(f"| {spec_of(c)} | {a['time_s']:.1f} | {a['move']} | {a['status']} | {a.get('planned_action') or '-'} | "
                         f"{a.get('age_s', '-')} | {cm(a['separation_lower_m'])} | {cm(a['centre_to_wall_m'])} |")
    return '\n'.join(lines)


NEAR_WALL_M = .60


def controller_table(results):
    """Centre forecast error by cohort and controller, overall and near walls (true path clearance < 0.60 m)."""
    groups = {}
    for r in results:
        groups.setdefault((r['cohort'], r['controller']), []).extend(r['decisions'])
    cm = lambda v: '-' if v is None else f'{100*v:.1f}'
    lines = [f'**Centre-position forecast error e_f by controller: overall and near walls. {LABEL}**', '',
             f'Near walls = the checked path\'s true clearance below {100*NEAR_WALL_M:.0f} cm (within 15 cm of the 45-cm disc, where the check binds). '
             'Budget at a percentile = static margin 2.5 cm minus e_f, e_p and e_m at that percentile (near-wall decisions).', '',
             '| Cohort | Controller | Decisions | e_f overall: p50 · p95 · p99 (cm) | Near-wall decisions | e_f near walls: p50 · p95 · p99 (cm) '
             '| e_p near walls p95 · p99 (cm) | e_m near walls p95 · p99 (cm) | Budget near walls at p95 · p99 (cm) |',
             '|---|---|---:|---|---:|---|---|---|---|']
    for (cohort, controller), ds in sorted(groups.items(), key=lambda kv: (kv[0][0].startswith('sens_'), order(kv[0][0]) if kv[0][0].startswith('sens_') else (0, 0), kv[0][0], kv[0][1])):
        if not ds:
            continue
        near = [d for d in ds if d['true_clearance'] < NEAR_WALL_M]
        ef = [d['e_f'] for d in ds]
        nf, np_, nm = [d['e_f'] for d in near], [d['e_p'] for d in near], [d['e_m'] for d in near]
        budget = lambda pc: None if not near or q(np_, pc) is None else (DISC_M-MAX_REACH_M)-q(nf, pc)-q(np_, pc)-(q(nm, pc) or 0.)
        label = spec_of(cohort) if cohort.startswith('sens_') else cohort
        lines.append(f"| {label} | {controller} | {len(ds)} | {cm(q(ef, 50))} · {cm(q(ef, 95))} · {cm(q(ef, 99))} | {len(near)} | "
                     f"{cm(q(nf, 50))} · {cm(q(nf, 95))} · {cm(q(nf, 99))} | {cm(q(np_, 95))} · {cm(q(np_, 99))} | {cm(q(nm, 95))} · {cm(q(nm, 99))} | "
                     f"{cm(budget(95))} · {cm(budget(99))} |")
    return '\n'.join(lines)


def main(cohorts, out_json, out_md, workers, by_controller=False):
    cohorts = cohorts or sorted([p.name for p in (BASE/'dev_cohorts').glob('sens_*') if (p/'config.json').exists()], key=order)
    jobs = [j for c in cohorts for j in assignments(c)]
    with Pool(workers) as pool:
        results = pool.map(mission, jobs, chunksize=1)
    text = controller_table(results) if by_controller else table(results, sorted(cohorts, key=order))
    print(text)
    if out_json:
        Path(out_json).write_text(json.dumps(dict(label=LABEL, missions=results), indent=1, default=float)+'\n')
    if out_md:
        Path(out_md).write_text(text+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cohorts', nargs='*')
    p.add_argument('--json')
    p.add_argument('--markdown')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--by-controller', action='store_true', help='e_f by cohort and controller, overall and near walls (works for prelim_* cohorts)')
    a = p.parse_args()
    main(a.cohorts, a.json, a.markdown, a.workers, a.by_controller)
