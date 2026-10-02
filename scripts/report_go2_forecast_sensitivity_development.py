"""PRELIMINARY: forecast-sensitivity dose-response (Andrew, 2 October 2026).

C1 drives prelim mazes 30-49 with recovery off and the coverage-rule fix, its forecasts degraded
by noise or by a scale bias (`--degrade`, see `degradation_mixin`). For each level:
- driving: round-trip success (Wilson 95%), SPL, median round-trip time, hold rate, contacts;
- the forecast error the planner actually acted on, measured exactly as for C3 and C4 (the
  selected move's 700-ms forecast against physics truth): median error and median ratio.
C3 and C4 are marked by their measured closed-loop error in the preliminary recovery-off run.
Writes a markdown table and a two-panel plot (success and SPL against measured error for the
noise series, and against the scale factor).

Usage: report_go2_forecast_sensitivity_development.py [--markdown OUT] [--plot OUT.png]
"""
import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from scripts.report_go2_prelim_results_development import BASE, collect, mean, wilson
from scripts.score_go2_dev_closed_loop_prediction_development import cohort_runs, decisions, metrics

LABEL = 'PRELIMINARY (prelim_test_v1 mazes 30-49, recovery off, coverage-rule fix; development mode)'
REFERENCE = ('prelim_off',)  # C3 and C4 measured closed-loop error, recovery off


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
        spec = condition(name)
        kind, value = ('none', 0.) if spec == 'none' else (spec.split(':')[0], float(spec.split(':')[1]))
        table.append(dict(name=name, spec=spec, kind=kind, value=value, missions=len(rows), successes=k, success=k/len(rows), lo=lo, hi=hi,
                          spl=mean([r.get('round_trip_spl') for r in rows]), time=st.median(times) if times else None,
                          hold=mean([r.get('outbound_hold_rate') for r in rows]), contacts=sum(r.get('contacts') or 0 for r in rows),
                          min_clearance=min((r['min_clearance_m'] for r in rows if r.get('min_clearance_m') is not None), default=None),
                          error_mm=err, ratio=ratio, scored=n))
    refs = {}
    for arm in ('C3', 'C4'):
        rows = [d for name in REFERENCE for a, run in cohort_runs(name) if a == arm for d in decisions(run)]
        m = metrics(rows)['all']
        refs[arm] = dict(error_mm=m['median_xy_mm'], ratio=m['median_ratio'])
    order = lambda r: (r['kind'] != 'none', r['kind'], r['value'])
    lines = [f'**Forecast-sensitivity dose-response. {LABEL}**', '',
             'Rows: C1 with its forecasts degraded. "Measured forecast error" is the selected move\'s 700-ms forecast against physics truth while driving '
             '(median error · median predicted/true ratio), the same measure used for C3 and C4.', '',
             '| Condition | Missions | Success (Wilson 95%) | SPL | Median round trip (s) | Outbound hold rate | Contacts | Min clearance | Measured forecast error |',
             '|---|---:|---|---:|---:|---:|---:|---:|---|']
    for r in sorted(table, key=order):
        lines.append(f"| {r['spec']} | {r['missions']} | {r['successes']}/{r['missions']} · {r['success']:.2f} ({r['lo']:.2f}–{r['hi']:.2f}) | "
                     f"{r['spl']:.2f} | {'-' if r['time'] is None else f'{r['time']:.0f}'} | {r['hold']:.3f} | {r['contacts']} | "
                     f"{'-' if r['min_clearance'] is None else f'{100*r['min_clearance']:.1f} cm'} | "
                     f"{'-' if r['error_mm'] is None else f'{r['error_mm']:.0f} mm · {r['ratio']:.2f}'} |")
    lines += ['', f"Reference, measured while driving in the preliminary recovery-off run: C3 {refs['C3']['error_mm']:.0f} mm · "
                  f"{refs['C3']['ratio']:.2f}; C4 {refs['C4']['error_mm']:.0f} mm · {refs['C4']['ratio']:.2f}."]
    text = '\n'.join(lines)
    print(text)
    if markdown:
        Path(markdown).write_text(text+'\n')
    if plot:
        draw(table, refs, plot)


def draw(table, refs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = [r for r in table if r['kind'] == 'none']
    noise = sorted([r for r in table if r['kind'] == 'noise']+base, key=lambda r: r['error_mm'] or 0)
    scale = sorted([r for r in table if r['kind'] == 'scale']+[dict(b, value=1.0) for b in base], key=lambda r: r['value'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for ax, rows, xkey, xlabel in ((axes[0], noise, 'error_mm', 'measured forecast error while driving (median, mm, 700 ms)'),
                                   (axes[1], scale, 'value', 'forecast scale factor (predicted motion × S)')):
        x = [r[xkey] for r in rows]
        ax.plot(x, [r['success'] for r in rows], 'o-', color='#0e6a57', label='success')
        ax.fill_between(x, [r['lo'] for r in rows], [r['hi'] for r in rows], color='#0e6a57', alpha=.15, linewidth=0)
        ax.plot(x, [r['spl'] for r in rows], 's--', color='#8a4b0b', label='SPL')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(alpha=.3)
        ax.tick_params(labelsize=8)
    axes[0].set_xscale('symlog', linthresh=10)
    for arm, colour in (('C4', '#3b6fb6'), ('C3', '#b63b6f')):
        axes[0].axvline(refs[arm]['error_mm'], color=colour, linestyle=':', linewidth=1.5)
        axes[0].text(refs[arm]['error_mm'], .05, f' {arm}', color=colour, fontsize=8)
        axes[1].axvline(refs[arm]['ratio'], color=colour, linestyle=':', linewidth=1.5)
        axes[1].text(refs[arm]['ratio'], .05, f' {arm}', color=colour, fontsize=8)
    axes[0].set_title('Noise on displacement and heading', fontsize=9)
    axes[1].set_title('Scale bias', fontsize=9)
    axes[0].set_ylabel('rate', fontsize=8)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('C1 with degraded forecasts, 20 preliminary mazes, recovery off (PRELIMINARY)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--markdown')
    p.add_argument('--plot')
    a = p.parse_args()
    main(a.markdown, a.plot)
