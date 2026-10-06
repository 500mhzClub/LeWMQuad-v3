"""PRELIMINARY: rescore the preliminary run under tighter time budgets (Andrew, 2 October 2026).

The controllers never see the mission budget (480 s), so success within a budget T can be read
off existing runs: a round trip succeeds within T if it succeeded and its home arrival
(outbound plus return time) is at most T. Budget-limited SPL is the round-trip SPL for such
missions and 0 otherwise. For each budget, per controller and recovery setting: success with a
Wilson 95% interval and SPL; and against C1 on the same mazes: the success difference with a
maze-level paired bootstrap 95% interval and the discordant counts.

Usage: rescore_go2_prelim_budgets_development.py [--budgets 240 300 360 480] [--markdown OUT]
"""
import argparse
from collections import defaultdict
from pathlib import Path

from scripts.report_go2_prelim_results_development import LABEL, bootstrap_mean, collect, mean, wilson


def within(r, budget):
    return bool(r.get('round_trip')) and r.get('total_s') is not None and r['total_s'] <= budget


def main(budgets, cohorts, out):
    rows = [r for r in collect(cohorts) if r.get('round_trip') is not None]
    groups = defaultdict(list)
    by = defaultdict(dict)
    for r in rows:
        groups[(r['controller'], r['recovery'])].append(r)
        by[(r['recovery'], r['maze'])][r['controller']] = r
    lines = [f'**Success within a time budget. {LABEL}**', '',
             'A round trip counts if home was reached by the budget; SPL is 0 otherwise. Controllers never see the budget.', '',
             '| Ctrl | Recovery | Missions | ' + ' | '.join(f'≤ {b} s: success (Wilson 95%) · SPL' for b in budgets) + ' |',
             '|---|---|---:|' + '---|'*len(budgets)]
    for (ctrl, rec), rs in sorted(groups.items()):
        cells = []
        for b in budgets:
            k = sum(within(r, b) for r in rs)
            lo, hi = wilson(k, len(rs))
            spl = mean([r['round_trip_spl'] if within(r, b) else 0. for r in rs])
            cells.append(f'{k}/{len(rs)} · {k/len(rs):.2f} ({lo:.2f}–{hi:.2f}) · {spl:.2f}')
        lines.append(f"| {ctrl}{'*' if ctrl == 'C0' else ''} | {rec} | {len(rs)} | " + ' | '.join(cells) + ' |')
    lines += ['', f'**Against C1 on the same mazes, success-within-budget difference (other minus C1, paired bootstrap 95%) and '
                  f'discordant mazes (only C1 / only other). {LABEL}**', '',
              '| Ctrl vs C1 | Recovery | Mazes | ' + ' | '.join(f'≤ {b} s' for b in budgets) + ' |',
              '|---|---|---:|' + '---|'*len(budgets)]
    for rec in ('on', 'off'):
        for ctrl in ('C0', 'C2', 'C3', 'C4'):
            pairs = [v for (rr, _m), v in sorted(by.items()) if rr == rec and ctrl in v and 'C1' in v]
            if not pairs:
                continue
            cells = []
            for b in budgets:
                x = [within(v[ctrl], b) for v in pairs]
                c1 = [within(v['C1'], b) for v in pairs]
                d, lo, hi = bootstrap_mean([int(a)-int(c) for a, c in zip(x, c1)])
                cells.append(f'{d:+.2f} ({lo:+.2f} to {hi:+.2f}) · {sum(c and not a for a, c in zip(x, c1))}/{sum(a and not c for a, c in zip(x, c1))}')
            lines.append(f"| {ctrl}{'*' if ctrl == 'C0' else ''} | {rec} | {len(pairs)} | " + ' | '.join(cells) + ' |')
    text = '\n'.join(lines)
    print(text)
    if out:
        Path(out).write_text(text+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--budgets', nargs='+', type=float, default=(240., 300., 360., 480.))
    p.add_argument('--cohorts', nargs='+', default=('prelim_trial', 'prelim_on', 'prelim_off'))
    p.add_argument('--markdown')
    a = p.parse_args()
    main([int(b) if b.is_integer() else b for b in a.budgets], a.cohorts, a.markdown)
