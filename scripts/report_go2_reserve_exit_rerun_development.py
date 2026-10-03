"""PRELIMINARY: exits and remaining stalls on the reserve_exit_v1 harness (Andrew, 2-3 October 2026).

For a cohort run on the next harness version:
- exits taken per controller: decisions whose final command is a translation that passed only by the reserve exit (an
  exit the check selected but a later rule replaced is counted separately, by that rule: stopping projection, coverage
  rule, clearance-turn latch, other), the missions using them, the check's own clearance at the
  start and end of each exit path, and the true centre clearance at the decision and 1 s later;
- last-moment depth stops on exits (CURRENT_OBSERVED_OBSTACLE_VETO / CURRENT_STOPPING_MARGIN_VETO on an exit's window);
- every remaining stall (>= 120 s without translation), with its mechanism from analyse_go2_reserve_trap_development;
- contacts and minimum clearance per controller.
Success, SPL and the paired comparison against C1 come from scripts/report_go2_prelim_results_development.py.

Usage: report_go2_reserve_exit_rerun_development.py --cohorts NAME ... [--markdown OUT] [--json OUT]
"""
import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from scripts.analyse_go2_reserve_trap_development import TRANSLATIONS, mission as trap_mission
from scripts.diagnose_go2_forecast_sensitivity_failures_development import BASE, DEPTH_STOPS, wall_distance

LABEL = 'PRELIMINARY (reserve_exit_v1 harness; development mode)'
TRAP = ('inside the disc', 'inside the reserve, none increases', 'inside the reserve, a translation increases but fails recovery')


def exit_row(selection):
    """The exit row if the final command is a translation that passed only by the reserve exit, else None."""
    if 'c2_nominal_path_check' in selection:
        rows = {r['action']: r for r in selection['c2_nominal_path_check']['rows']}
    else:
        rows = {r['action']: r for r in selection.get('memory_forecast_candidates') or []}
    row = rows.get(selection['action'])
    return row if selection['action'] in TRANSLATIONS and row is not None and row.get('reserve_exit_path_clear') else None


def overridden_exit(selection):
    """Why an exit the clearance check selected did not become the command (forecast controllers), or None."""
    if not selection.get('selected_reserve_exit') or exit_row(selection) is not None:
        return None
    projection = selection.get('planned_stopping_projection') or {}
    coverage = selection.get('coverage_view_request') or {}
    if projection.get('changed') and projection.get('before_action') in TRANSLATIONS:
        return 'stopping projection'
    if coverage.get('status') == 'TRANSLATION_REQUIRES_COVERAGE':
        return 'coverage rule'
    if (selection.get('clearance_turn') or {}).get('active'):
        return 'clearance-turn latch'
    return 'other'


def mission(job):
    cohort, controller, maze, assignment = job
    root = BASE/'runs'/assignment
    read = lambda n: json.loads((root/n).read_text())
    ev, spec, requests = read('episode_evaluation.json'), read('specification.json'), read('requests.json')
    plans = [r for r in read('planning.json') if 'selection' in r]
    stops = {q.get('command_observation_ns') for q in requests if q['reason'] in DEPTH_STOPS}
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as f:
        ts, pose = f['timestamp_s'].copy(), f['base_pose_world'].copy()
    distance = wall_distance(spec['geometry']['wall_boxes'])
    centre = lambda t_s: float(distance(pose[min(int(np.searchsorted(ts, t_s)), len(ts)-1), :2])[0])
    exits, overridden = [], Counter()
    for r in plans:
        why = overridden_exit(r['selection'])
        if why:
            overridden[why] += 1
        row = exit_row(r['selection'])
        if row is None:
            continue
        t = r['measured_ns']/1e9
        d = row['segment_clearances_m']
        exits.append(dict(time_s=round(t-1.5, 1), action=r['action'], check_start_m=d[0], check_end_m=d[-1],
                          true_centre_m=centre(t), true_centre_after_1s_m=centre(t+1.), depth_stopped=r['measured_ns'] in stops))
    trap = trap_mission(assignment)
    stall = trap['stall']
    mechanism = None
    if stall and stall['samples']:
        counts = Counter(x['mechanism'] for x in stall['samples'])
        mechanism = 'reserve trap' if sum(counts[k] for k in TRAP)/len(stall['samples']) >= .5 else max(counts.items(), key=lambda kv: kv[1])[0]
    s = ev['safety']
    return dict(cohort=cohort, controller=controller, maze=maze, assignment=assignment, round_trip=bool(ev['round_trip_success']),
                contacts=ev['disallowed_contact_samples'] or 0, hard=s['hard']['confirmed_violation_samples'],
                min_clearance_m=s['hard']['minimum_separation_lower_m'], decisions=len(plans), exits=exits, overridden_exits=dict(overridden),
                stall=None if not stall else dict(begin_s=stall['begin_s'], end_s=stall['end_s'], mechanisms=stall['mechanisms'],
                                                  onset=stall['onset'], mechanism=mechanism))


def table(results):
    cm = lambda v: f'{100*v:.1f}'
    lines = [f'**Exits taken and remaining stalls. {LABEL}**', '',
             'Exit = the selected move passed the check only by the reserve exit (clearance never decreasing, ending higher). '
             'Check start/end = the check\'s own centre-path clearance (remembered walls). True centre = distance from the base centre '
             'to the nearest true wall at the decision and 1 s later.', '',
             '| Ctrl | Missions | Round trips | Contacts · hard | Min clearance (cm) | Exit decisions · missions using exits | '
             'Exit check clearance start → end, median (cm) | True centre at exit → 1 s later, median (cm) | Exits stopped by the depth stop | '
             'Exits selected by the check but overridden, by rule | Stalls · reserve trap |',
             '|---|---:|---:|---|---:|---|---|---|---:|---|---|']
    by = {}
    for m in results:
        by.setdefault(m['controller'], []).append(m)
    for controller in sorted(by):
        ms = by[controller]
        ex = [e for m in ms for e in m['exits']]
        med = lambda k: '-' if not ex else cm(float(np.median([e[k] for e in ex])))
        stalls = [m for m in ms if m['stall']]
        lines.append(f"| {controller} | {len(ms)} | {sum(m['round_trip'] for m in ms)} | {sum(m['contacts'] for m in ms)} · {sum(m['hard'] for m in ms)} | "
                     f"{cm(min(m['min_clearance_m'] for m in ms))} | {len(ex)} · {sum(bool(m['exits']) for m in ms)} | "
                     f"{med('check_start_m')} → {med('check_end_m')} | {med('true_centre_m')} → {med('true_centre_after_1s_m')} | "
                     f"{sum(e['depth_stopped'] for e in ex)} | {dict(sum((Counter(m['overridden_exits']) for m in ms), Counter())) or '-'} | "
                     f"{len(stalls)} · {sum(m['stall']['mechanism'] == 'reserve trap' for m in stalls)} |")
    lines += ['', '**Every remaining stall.**', '', '| Ctrl | Maze | Round trip | Stall (s) | Mechanism | Onset remembered · true centre clearance (m) | Samples by label |',
              '|---|---:|---|---|---|---|---|']
    for m in sorted(results, key=lambda m: (m['controller'], m['maze'])):
        st = m['stall']
        if not st:
            continue
        o = st['onset'] or {}
        rem = o.get('remembered_centre_m')
        lines.append(f"| {m['controller']} | {m['maze']} | {m['round_trip']} | {st['begin_s']:.0f}-{st['end_s']:.0f} | {st['mechanism']} | "
                     f"{'-' if rem is None else f'{rem:.3f}'} · {o.get('true_centre_m', float('nan')):.3f} | "
                     f"{'; '.join(f'{k} {v}' for k, v in sorted(st['mechanisms'].items(), key=lambda kv: -kv[1]))} |")
    return '\n'.join(lines)


def main(cohorts, out_md, out_json, workers):
    jobs = []
    for cohort in cohorts:
        for job in json.loads((BASE/'dev_cohorts'/cohort/'config.json').read_text())['plan']:
            if (BASE/'runs'/job[4]/'episode_evaluation.json').exists():
                jobs.append((cohort, job[0], job[2], job[4]))
    with Pool(workers) as pool:
        results = pool.map(mission, jobs, chunksize=1)
    text = table(results)
    print(text)
    if out_md:
        Path(out_md).write_text(text+'\n')
    if out_json:
        Path(out_json).write_text(json.dumps(dict(label=LABEL, missions=results), indent=1, default=float)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cohorts', nargs='+', required=True)
    p.add_argument('--markdown')
    p.add_argument('--json')
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    main(a.cohorts, a.markdown, a.json, a.workers)
