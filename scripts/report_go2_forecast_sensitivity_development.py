"""PRELIMINARY: forecast-sensitivity dose-response (Andrew, 2 October 2026).

C1 drives prelim mazes 30-49 with recovery off and the coverage-rule fix, its forecasts degraded
by noise or by a scale bias (`--degrade`, see `degradation_mixin`). For each level:
- driving: round-trip success (Wilson 95%), SPL, median round-trip time, hold rate, contacts;
- the forecast error the planner actually acted on, measured exactly as for C3 and C4 (the
  selected move's 700-ms forecast against physics truth): median error and median ratio.
C3 and C4 are marked by their measured closed-loop error in the preliminary recovery-off run.
C3's marker on the noise axis is approximate: its errors are structured (it under-predicts starts
and turns), not white noise, which the structured-error series (fwdscale, turnscale) copy.

Choice-change rate (Andrew, 2 October): at each decision, whether the degraded forecast selects a
different candidate than the clean forecast would on the same state. Recomputed offline from the
logged degraded and clean (command-history) forecasts with the planner's pure scorer (distance
progress minus contact, plus the heading-alignment term at the logged scale; the scan-mode
choice among hold and turns when the planner was scanning). This is the scoring-stage choice:
the later clearance, coverage and latch filters depend on the live map and are not recomputed.
The recomputation is validated against the logged utilities of the degraded forecast.

All conditions are compared against this experiment's own clean baseline (C1, coverage-rule
fix, recovery off), not the preliminary results: success difference with a maze-level paired
bootstrap 95% interval and discordant counts, SPL and median-time differences.
Writes a markdown table and a 2x2 plot.

Usage: report_go2_forecast_sensitivity_development.py [--markdown OUT] [--plot OUT.png]
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

from lewm.delayed_action_planning_development import score_delayed_predictions
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.mission_coordinate_metric_development import position_distance
from scripts.report_go2_prelim_results_development import BASE, bootstrap_mean, collect, mean, wilson
from scripts.score_go2_dev_closed_loop_prediction_development import cohort_runs, decisions, metrics

LABEL = 'PRELIMINARY (prelim_test_v1 mazes 30-49, recovery off, coverage-rule fix; development mode)'
REFERENCE = ('prelim_off',)  # C3 and C4 measured closed-loop error, recovery off


DELAY, COMMIT = 3, 4


def as_prediction(xy_yaw):
    a = np.asarray(xy_yaw, float)
    p = np.zeros((6, 8, 5))
    p[..., :2], p[..., 2], p[..., 3], p[..., 4] = a[..., :2], np.sin(a[..., 2]), np.cos(a[..., 2]), -1000.
    return p


def utilities(p, selection):
    """The planner's scoring-stage utilities (score_waypoint_alignment at the logged alignment scale)."""
    goal = np.asarray(selection['waypoint_body_xy_m'], float)
    metric = selection.get('position_metric_matrix')
    base = score_delayed_predictions(p, goal, delay_ticks=DELAY, commit_ticks=COMMIT, position_metric_matrix=metric)
    u = np.array([c['utility_m'] for c in base['candidates']])
    begin, end = DELAY-1, DELAY+COMMIT-1
    errors = []
    for index in (begin, end):
        delta = goal-p[:, index, :2]
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        yaw = np.arctan2(p[:, index, 2], p[:, index, 3])
        errors.append(np.abs(np.arctan2(np.sin(bearing-yaw), np.cos(bearing-yaw))))
    return u+selection['alignment_scale_m']*(errors[0]-errors[1])


def scan_choice(p, scan_error):
    begin, end = DELAY-1, DELAY+COMMIT-1
    scores = []
    for a in ('hold', 'left_turn', 'right_turn'):
        i = ACTIONS.index(a)
        yaw = math.atan2(p[i, end, 2], p[i, end, 3])-math.atan2(p[i, begin, 2], p[i, begin, 3])
        remaining = math.atan2(math.sin(scan_error-yaw), math.cos(scan_error-yaw))
        scores.append((.35*(abs(scan_error)-abs(remaining)), a))
    return max(scores)[1]


def choice_change(name):
    """(rate, decisions, max |recomputed - logged| utility) for one cohort's C1 runs."""
    changed = total = 0
    worst = 0.
    for arm, run in cohort_runs(name):
        for r in json.loads((run/'planning.json').read_text()):
            if 'selection' not in r or not r.get('motion_correction'):
                continue
            s, mc = r['selection'], r['motion_correction']
            if s.get('alignment_scale_m') is None or s.get('waypoint_body_xy_m') is None:
                continue
            clean = as_prediction(mc['command_history_forecast_xy_yaw'])
            degraded = as_prediction(mc['dev_degraded_forecast_xy_yaw']) if mc.get('dev_degraded_forecast_xy_yaw') is not None else clean
            ud, uc = utilities(degraded, s), utilities(clean, s)
            worst = max(worst, float(np.max(np.abs(ud-np.array([c.get('position_contact_utility_m', 0)+c.get('predicted_alignment_progress_m', 0)
                                                                for c in s['candidates']])))))
            if s.get('scan_heading_error_rad') is not None:
                a, b = scan_choice(degraded, s['scan_heading_error_rad']), scan_choice(clean, s['scan_heading_error_rad'])
            else:
                a, b = ACTIONS[int(np.argmax(ud))], ACTIONS[int(np.argmax(uc))]
            changed += a != b
            total += 1
    return (changed/total if total else None), total, worst


def condition(name):
    config = json.loads((BASE/'dev_cohorts'/name/'config.json').read_text())
    return config.get('forecast_degradation') or 'none'


def measured(name, arm='C1'):
    rows = [d for a, run in cohort_runs(name) if a == arm for d in decisions(run)]
    m = metrics(rows).get('all', {})
    return m.get('median_xy_mm'), m.get('median_ratio'), m.get('n')


def main(markdown, plot):
    names = sorted(p.name for p in (BASE/'dev_cohorts').iterdir() if p.name.startswith('sens_') and (p/'result.json').exists())
    table = []
    for name in names:
        rows = [r for r in collect([name]) if r.get('round_trip') is not None]
        if not rows:
            continue
        k = sum(bool(r['round_trip']) for r in rows)
        lo, hi = wilson(k, len(rows))
        times = [r['total_s'] for r in rows if r['round_trip'] and r.get('total_s')]
        err, ratio, n = measured(name)
        rate, decisions_n, worst = choice_change(name)
        spec = condition(name)
        kind, value = ('none', 0.) if spec == 'none' else (spec.split(':')[0], float(spec.split(':')[1]))
        table.append(dict(name=name, spec=spec, kind=kind, value=value, missions=len(rows), successes=k, success=k/len(rows), lo=lo, hi=hi,
                          spl=mean([r.get('round_trip_spl') for r in rows]), time=st.median(times) if times else None,
                          hold=mean([r.get('outbound_hold_rate') for r in rows]), contacts=sum(r.get('contacts') or 0 for r in rows),
                          min_clearance=min((r['min_clearance_m'] for r in rows if r.get('min_clearance_m') is not None), default=None),
                          error_mm=err, ratio=ratio, scored=n, choice_change=rate, choice_decisions=decisions_n, recompute_max_abs=worst,
                          rows=rows))
    refs = {}
    for arm in ('C3', 'C4'):
        rows = [d for name in REFERENCE for a, run in cohort_runs(name) if a == arm for d in decisions(run)]
        m = metrics(rows)['all']
        refs[arm] = dict(error_mm=m['median_xy_mm'], ratio=m['median_ratio'])
    base = next((r for r in table if r['kind'] == 'none'), None)
    if base:
        by_maze = {r['maze']: r for r in base['rows']}
        for r in table:
            pairs = [(x, by_maze[x['maze']]) for x in r['rows'] if x['maze'] in by_maze]
            d = bootstrap_mean([int(bool(a['round_trip']))-int(bool(b['round_trip'])) for a, b in pairs])
            both = [(a, b) for a, b in pairs if a['round_trip'] and b['round_trip']]
            r['vs_base'] = dict(success=d, only_base=sum(1 for a, b in pairs if b['round_trip'] and not a['round_trip']),
                                only_cond=sum(1 for a, b in pairs if a['round_trip'] and not b['round_trip']),
                                spl=bootstrap_mean([a['round_trip_spl']-b['round_trip_spl'] for a, b in both]),
                                time=bootstrap_mean([a['total_s']-b['total_s'] for a, b in both if a.get('total_s') and b.get('total_s')]))
    order = lambda r: (r['kind'] != 'none', r['kind'], r['value'])
    lines = [f'**Forecast-sensitivity dose-response. {LABEL}**', '',
             'Rows: C1 with its forecasts degraded. "Measured forecast error" is the selected move\'s 700-ms forecast against physics truth while driving '
             '(median error · median predicted/true ratio), the same measure used for C3 and C4.', '',
             '| Condition | Missions | Success (Wilson 95%) | SPL | Median round trip (s) | Outbound hold rate | Contacts | Min clearance | Measured forecast error | Choice-change rate | vs clean baseline: success diff (95% CI) · only base/only cond | SPL diff | Time diff (s) |',
             '|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|---|---|']
    for r in sorted(table, key=order):
        lines.append(f"| {r['spec']} | {r['missions']} | {r['successes']}/{r['missions']} · {r['success']:.2f} ({r['lo']:.2f}–{r['hi']:.2f}) | "
                     f"{r['spl']:.2f} | {'-' if r['time'] is None else f'{r['time']:.0f}'} | {r['hold']:.3f} | {r['contacts']} | "
                     f"{'-' if r['min_clearance'] is None else f'{100*r['min_clearance']:.1f} cm'} | "
                     f"{'-' if r['error_mm'] is None else f'{r['error_mm']:.0f} mm · {r['ratio']:.2f}'} | "
                     f"{'-' if r['choice_change'] is None else f'{r['choice_change']:.3f}'} | {vs(r, 'success')} · {r.get('vs_base', {}).get('only_base', '-')}/{r.get('vs_base', {}).get('only_cond', '-')} | "
                     f"{vs(r, 'spl')} | {vs(r, 'time', 1)} |")
    worst = max((r['recompute_max_abs'] for r in table), default=0.)
    lines += ['', f'Choice-change recomputation reproduces the logged degraded-forecast utilities to within {worst:.1e} m.']
    lines += ['', f"Reference, measured while driving in the preliminary recovery-off run: C3 {refs['C3']['error_mm']:.0f} mm · "
                  f"{refs['C3']['ratio']:.2f}; C4 {refs['C4']['error_mm']:.0f} mm · {refs['C4']['ratio']:.2f}."]
    text = '\n'.join(lines)
    print(text)
    if markdown:
        Path(markdown).write_text(text+'\n')
    if plot:
        draw(table, refs, plot)


def vs(r, key, digits=2):
    m = r.get('vs_base', {}).get(key)
    return '-' if not m or m[0] is None else f'{m[0]:+.{digits}f} ({m[1]:+.{digits}f} to {m[2]:+.{digits}f})'


def draw(table, refs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = [r for r in table if r['kind'] == 'none']
    green, amber, blue, plum = '#0e6a57', '#8a4b0b', '#3b6fb6', '#b63b6f'

    def series(ax, rows, xkey):
        x = [r[xkey] for r in rows]
        ax.plot(x, [r['success'] for r in rows], 'o-', color=green, label='success')
        ax.fill_between(x, [r['lo'] for r in rows], [r['hi'] for r in rows], color=green, alpha=.15, linewidth=0)
        ax.plot(x, [r['spl'] for r in rows], 's--', color=amber, label='SPL')

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), dpi=150)
    noise = sorted([r for r in table if r['kind'] == 'noise']+base, key=lambda r: r['error_mm'] or 0)
    series(axes[0, 0], noise, 'error_mm')
    axes[0, 0].set_xscale('symlog', linthresh=10)
    axes[0, 0].set_xlabel('measured forecast error while driving (median, mm, 700 ms)', fontsize=8)
    axes[0, 0].set_title('White noise on displacement and heading', fontsize=9)
    axes[0, 0].axvline(refs['C4']['error_mm'], color=blue, linestyle=':', linewidth=1.5)
    axes[0, 0].text(refs['C4']['error_mm'], .06, ' C4', color=blue, fontsize=8)
    axes[0, 0].axvline(refs['C3']['error_mm'], color=plum, linestyle=':', linewidth=1.5)
    axes[0, 0].text(refs['C3']['error_mm'], .14, ' ~C3 (approx.: errors not white noise)', color=plum, fontsize=7)
    scale = sorted([r for r in table if r['kind'] == 'scale']+[dict(b, value=1.0) for b in base], key=lambda r: r['value'])
    series(axes[0, 1], scale, 'value')
    axes[0, 1].set_xlabel('uniform forecast scale factor (all moves)', fontsize=8)
    axes[0, 1].set_title('Uniform scale bias', fontsize=9)
    for arm, colour, y in (('C4', blue, .06), ('C3', plum, .14)):
        axes[0, 1].axvline(refs[arm]['ratio'], color=colour, linestyle=':', linewidth=1.5)
        axes[0, 1].text(refs[arm]['ratio'], y, f' {arm} (overall ratio)', color=colour, fontsize=7)
    ax = axes[1, 0]
    for kind, colour, marker, label in (('fwdscale', green, 'o', 'forward/arcs under-predicted'), ('turnscale', amber, '^', 'turns over-predicted')):
        rows = sorted([r for r in table if r['kind'] == kind]+[dict(b, value=1.0) for b in base], key=lambda r: r['value'])
        ax.plot([r['value'] for r in rows], [r['success'] for r in rows], marker+'-', color=colour, label=f'{label}: success')
        ax.plot([r['value'] for r in rows], [r['spl'] for r in rows], marker+'--', color=colour, alpha=.6, label=f'{label}: SPL')
    ax.set_xlabel('scale factor on the affected moves only', fontsize=8)
    ax.set_title('Structured errors (copying C3)', fontsize=9)
    ax = axes[1, 1]
    markers = dict(none='*', noise='o', scale='s', fwdscale='D', turnscale='^')
    for r in table:
        if r['choice_change'] is None:
            continue
        ax.plot(r['choice_change'], r['success'], markers[r['kind']], color=green)
        ax.plot(r['choice_change'], r['spl'], markers[r['kind']], color=amber, alpha=.7)
        ax.annotate(r['spec'], (r['choice_change'], r['success']), fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('choice-change rate (scoring stage, per decision)', fontsize=8)
    ax.set_title('Success (green) and SPL (amber) against choice change', fontsize=9)
    for a in axes.flat:
        a.set_ylim(0, 1.05)
        a.grid(alpha=.3)
        a.tick_params(labelsize=8)
    axes[0, 0].set_ylabel('rate', fontsize=8)
    axes[1, 0].set_ylabel('rate', fontsize=8)
    axes[0, 0].legend(fontsize=7, loc='lower left')
    axes[1, 0].legend(fontsize=6, loc='lower left')
    fig.suptitle('C1 with degraded forecasts: 20 preliminary mazes, recovery off, coverage-rule fix (PRELIMINARY)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--markdown')
    p.add_argument('--plot')
    a = p.parse_args()
    main(a.markdown, a.plot)
