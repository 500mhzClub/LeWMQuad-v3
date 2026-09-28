"""Markdown tables for the capability report from the qualification analysis JSON."""
import argparse
import json
from pathlib import Path


def pct(x):
    return '—' if x is None else f'{100*x:.0f}%'


def ci(c, scale=100, fmt='{:.0f}'):
    return '—' if c is None else f'[{fmt.format(scale*c[0])}, {fmt.format(scale*c[1])}]'


def main(path):
    d = json.loads(path.read_text())
    c = d['controllers']
    arms = [a for a in ('C1', 'C2', 'C3', 'C4', 'C0') if a in c]
    print('| Controller | Round trips | 95% CI | Beacon | Home | Contacts | Capable |')
    print('|---|---|---|---|---|---|---|')
    for a in arms:
        s = c[a]
        capable = 'diagnostic' if s['capable'] is None else ('**yes**' if s['capable'] else '**no**')
        print(f"| {a} | {s['round_trips']}/{s['episodes']} ({pct(s['round_trip_rate']['mean'])}) | {ci(s['round_trip_rate']['ci95'])} | "
              f"{s['beacon']}/{s['episodes']} | {s['home']}/{s['episodes']} | {s['disallowed_contact_samples']} | {capable} |")
    print()
    print('| Controller | SPL out [CI] | SPL return [CI] | Time to beacon, median (IQR) | Time to home, median (IQR) | Stall out / return (holds/decisions) | Latency median / p95 | Wall per mission (median) |')
    print('|---|---|---|---|---|---|---|---|')
    for a in arms:
        s = c[a]
        tb, th = s['time_to_beacon_s'], s['time_to_home_s']
        so, sr = s['stall']['OUTBOUND'], s['stall']['RETURN']
        lat = s['decision_latency_s']
        f = lambda t: '—' if t['median'] is None else f"{t['median']:.0f} s ({t['iqr'][0]:.0f}–{t['iqr'][1]:.0f})"
        print(f"| {a} | {s['spl_outbound']['mean']:.2f} {ci(s['spl_outbound']['ci95'], 1, '{:.2f}')} | {s['spl_return']['mean']:.2f} {ci(s['spl_return']['ci95'], 1, '{:.2f}')} | "
              f"{f(tb)} | {f(th)} | {pct(so['per_maze_mean'])} ({so['holds']}/{so['selected']}) / {pct(sr['per_maze_mean'])} ({sr['holds']}/{sr['selected']}) | "
              f"{lat['median']:.2f} / {lat['p95']:.2f} s | {s['wall_s_per_episode']['median']:.0f} s |")
    print()
    print('| Controller | Hard viol. | Hard unresolved | Operating viol. | FK interval failures (hard/op) | Min separation |')
    print('|---|---|---|---|---|---|')
    for a in arms:
        s = c[a]
        print(f"| {a} | {s['hard_violation_samples']} | {s['hard_unresolved_samples']} | {s['operating_violation_samples']} | "
              f"{s['fk_interval_failures']['hard']}/{s['fk_interval_failures']['operating']} | {1000*s['minimum_separation_lower_m']:.0f} mm |")
    print()
    eps = d['per_episode']
    mazes = sorted({e['maze'] for e in eps})
    print('| Maze | ' + ' | '.join(arms) + ' |')
    print('|---|' + '---|'*len(arms))
    for m in mazes:
        cells = []
        for a in arms:
            e = next((e for e in eps if e['maze'] == m and e['controller'] == a), None)
            if e is None:
                cells.append('')
            else:
                mark = 'RT' if e['round_trip'] else ('B only' if e['beacon'] else 'fail')
                cells.append(f"{mark} {e['simulated_s']:.0f}s")
        print(f'| {m:02d}/0 | ' + ' | '.join(cells) + ' |')
    print()
    for k, v in d['paired_round_trip_differences_vs_C1'].items():
        print(f"- {k}: {100*v['mean']:+.0f} points {ci(v['ci95'])}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('analysis', type=Path)
    main(p.parse_args().analysis)
