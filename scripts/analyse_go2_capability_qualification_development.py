"""Capability-qualification analysis for the frozen V4 harness (pre-registered).

Unit: the maze (one validation episode per maze). 95% intervals from the
pre-registered paired maze bootstrap (10,000 replicates, seed 2026092519,
percentile). C1-C4 share one resample of the 20 validation mazes; C0 is
resampled over its own ten. Raw counts accompany every rate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO/'docs/go2_navigation_capability_completed_support_v4_2026-09-27.json'
ARMS = ('C0', 'C1', 'C2', 'C3', 'C4')
PHASES = ('OUTBOUND', 'RETURN')


def load(base, rows):
    episodes = []
    for row in rows:
        root = base/'runs'/row['assignment']
        e = json.loads((root/'episode_evaluation.json').read_text())
        timings = json.loads((root/'stage_timings.json').read_text()) if (root/'stage_timings.json').exists() else []
        latencies = [t['wall_ns']/1e9 for t in timings if t.get('stage') == 'planning']
        result = json.loads((root/'result.json').read_text())
        prefix = root/'oracle_prefix_erratum_evaluation.json'
        stall = e.get('stall_by_phase') or {}
        episodes.append(dict(
            controller=row['controller'], assignment=row['assignment'], episode=e['episode_id'], maze=int(e['episode_id'].split('/')[0]),
            beacon=bool(e['beacon_success']), home=bool(e['home_success']), round_trip=bool(e['round_trip_success']),
            spl_outbound=e['outbound']['spl'], spl_return=e['return_leg']['spl'],
            time_to_beacon_s=e['outbound']['elapsed_s'] if e['beacon_success'] else None,
            time_to_home_s=e['return_leg']['elapsed_s'] if e['home_success'] else None,
            path_outbound_m=e['outbound']['actual_path_m'], path_return_m=e['return_leg']['actual_path_m'],
            shortest_outbound_m=e['outbound']['shortest_path_m'], shortest_return_m=e['return_leg']['shortest_path_m'],
            contacts=e['disallowed_contact_samples'], hard=e['safety']['hard']['confirmed_violation_samples'],
            hard_unresolved=e['safety']['hard']['unresolved_sampled_samples'],
            operating=e['safety']['operating']['confirmed_violation_samples'],
            operating_unresolved=e['safety']['operating']['unresolved_sampled_samples'],
            fk_interval_failures_hard=e['safety']['hard']['interval_robustness_failure_count'],
            fk_interval_failures_operating=e['safety']['operating']['interval_robustness_failure_count'],
            minimum_separation_lower_m=e['safety']['hard']['minimum_separation_lower_m'],
            holds={p: (stall.get(p) or {}).get('holds', 0) for p in PHASES},
            selected={p: (stall.get(p) or {}).get('selected_plans', 0) for p in PHASES},
            latencies=latencies, wall_s=e['wall_s'], simulated_s=result['simulated_s'],
            wall_per_simulated_s=e.get('wall_seconds_per_simulated_second'),
            taxonomy=e['failure_and_stall_taxonomy'], source_error=e.get('source_error'),
            prefix=json.loads(prefix.read_text()) if prefix.exists() else None))
    return episodes


def interval(values, draws):
    stats = np.nanmean(values[draws], axis=1)
    return [float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))]


def summarise(episodes, draws_by_maze):
    out = {}
    for arm in ARMS:
        rows = sorted([e for e in episodes if e['controller'] == arm], key=lambda e: e['maze'])
        if not rows:
            continue
        mazes = [e['maze'] for e in rows]
        # Pre-registered draws for the full 20 (C1-C4) and 10 (C0) maze sets; an incomplete set
        # (only while a cohort is unfinished) uses the same seed over its own mazes.
        draws = draws_by_maze.get(tuple(mazes))
        if draws is None:
            draws = np.random.default_rng(2026092519).integers(0, len(mazes), size=(10000, len(mazes)))

        def metric(key):
            values = np.array([np.nan if e[key] is None else float(e[key]) for e in rows])
            return dict(mean=float(np.nanmean(values)) if np.isfinite(values).any() else None,
                        ci95=interval(values, draws) if np.isfinite(values).any() else None)
        stall = {}
        for phase in PHASES:
            rates = np.array([e['holds'][phase]/e['selected'][phase] if e['selected'][phase] else np.nan for e in rows])
            stall[phase] = dict(holds=sum(e['holds'][phase] for e in rows), selected=sum(e['selected'][phase] for e in rows),
                per_maze_mean=float(np.nanmean(rates)) if np.isfinite(rates).any() else None,
                ci95=interval(rates, draws) if np.isfinite(rates).any() else None)
        pooled = np.concatenate([e['latencies'] for e in rows]) if any(e['latencies'] for e in rows) else np.array([])
        beacon_times = [e['time_to_beacon_s'] for e in rows if e['time_to_beacon_s'] is not None]
        home_times = [e['time_to_home_s'] for e in rows if e['time_to_home_s'] is not None]
        n = len(rows)
        trips = sum(e['round_trip'] for e in rows)
        contacts = sum(e['contacts'] for e in rows)
        out[arm] = dict(
            episodes=n, mazes=mazes, beacon=sum(e['beacon'] for e in rows), home=sum(e['home'] for e in rows), round_trips=trips,
            beacon_rate=metric('beacon'), home_rate=metric('home'), round_trip_rate=metric('round_trip'),
            spl_outbound=metric('spl_outbound'), spl_return=metric('spl_return'),
            time_to_beacon_s=dict(n=len(beacon_times), median=float(np.median(beacon_times)) if beacon_times else None,
                iqr=[float(np.percentile(beacon_times, 25)), float(np.percentile(beacon_times, 75))] if beacon_times else None),
            time_to_home_s=dict(n=len(home_times), median=float(np.median(home_times)) if home_times else None,
                iqr=[float(np.percentile(home_times, 25)), float(np.percentile(home_times, 75))] if home_times else None),
            disallowed_contact_samples=contacts, hard_violation_samples=sum(e['hard'] for e in rows),
            hard_unresolved_samples=sum(e['hard_unresolved'] for e in rows), operating_violation_samples=sum(e['operating'] for e in rows),
            operating_unresolved_samples=sum(e['operating_unresolved'] for e in rows),
            fk_interval_failures=dict(hard=sum(e['fk_interval_failures_hard'] for e in rows), operating=sum(e['fk_interval_failures_operating'] for e in rows)),
            minimum_separation_lower_m=float(min(e['minimum_separation_lower_m'] for e in rows)),
            stall=stall,
            decision_latency_s=dict(decisions=int(pooled.size), median=float(np.median(pooled)) if pooled.size else None,
                p95=float(np.percentile(pooled, 95)) if pooled.size else None),
            wall_s_per_episode=dict(median=float(np.median([e['wall_s'] for e in rows])), mean=float(np.mean([e['wall_s'] for e in rows])),
                total_h=float(sum(e['wall_s'] for e in rows)/3600)),
            wall_per_simulated_s=float(np.median([e['wall_per_simulated_s'] for e in rows if e['wall_per_simulated_s']])),
            taxonomy=dict(sum((Counter(e['taxonomy']) for e in rows), Counter())),
            failures=[dict(episode=e['episode'], beacon=e['beacon'], home=e['home'], simulated_s=e['simulated_s'],
                source_error=e['source_error'], taxonomy=e['taxonomy']) for e in rows if not e['round_trip']],
            prefix_erratum=None if arm != 'C0' else dict(
                no_matching_branch_decisions=sum(e['prefix']['no_matching_branch_decisions'] for e in rows),
                vetoed_selections=sum(e['prefix']['vetoed_selections'] for e in rows),
                comparable_rows=sum(e['prefix']['comparable_rows'] for e in rows),
                maximum_position_error_m=max(e['prefix']['comparable_maximum_position_error_m'] for e in rows),
                maximum_yaw_error_deg=max(e['prefix']['comparable_maximum_yaw_error_deg'] for e in rows)),
            capable=None if arm == 'C0' else bool(trips >= 16 and contacts == 0),
            criterion='round-trip success >= 16/20 (80%) and zero disallowed contacts; point estimate')
    return out


def paired_differences(episodes, draws, reference='C1'):
    table = {}
    base = {e['maze']: e['round_trip'] for e in episodes if e['controller'] == reference}
    for arm in ('C2', 'C3', 'C4'):
        other = {e['maze']: e['round_trip'] for e in episodes if e['controller'] == arm}
        mazes = sorted(set(base) & set(other))
        if len(mazes) != 20:
            continue
        diff = np.array([float(other[m])-float(base[m]) for m in mazes])
        table[f'{arm}-{reference}'] = dict(mean=float(diff.mean()), ci95=interval(diff, draws), paired_mazes=len(mazes))
    return table


def main(out):
    protocol = json.loads(PROTOCOL.read_text())
    base = Path(protocol['output_root'])
    cohort = base/'cohorts/v4_completed_support_validation'
    config = json.loads((cohort/'config.json').read_text())
    results = sorted(cohort.glob('result*.json'))
    result = json.loads(results[-1].read_text()) if results else dict(harness_sha256=config['harness_sha256'], complete=False, stops=[])
    # Rows from every closed assignment (original owner and resumptions), controller from the fixed plan.
    rows = [json.loads((cohort/f'{a}_result.json').read_text()) | dict(controller=arm)
            for arm, maze, a in config['assignments'] if (cohort/f'{a}_result.json').exists()]
    result['complete'] = len(rows) == len(config['assignments'])
    episodes = load(base, rows)
    boot = protocol['qualification']['inference']
    rng = np.random.default_rng(boot['seed'])
    draws20 = rng.integers(0, 20, size=(boot['replicates'], 20))
    rng0 = np.random.default_rng(boot['seed'])
    draws10 = rng0.integers(0, 10, size=(boot['replicates'], 10))
    draws_by_maze = {tuple(range(10, 30)): draws20, tuple(range(10, 20)): draws10}
    summary = summarise(episodes, draws_by_maze)
    record = dict(schema='navigation_capability_qualification_analysis.v1', harness='v4_completed_support',
        harness_sha256=result['harness_sha256'], cohort_complete=result['complete'], cohort_stops=result['stops'],
        bootstrap=dict(unit='maze', replicates=boot['replicates'], seed=boot['seed'], confidence=.95, method='percentile',
            pairing='C1-C4 share one resample of mazes 10-29; C0 resampled over 10-19'),
        controllers=summary, paired_round_trip_differences_vs_C1=paired_differences(episodes, draws20),
        per_episode=[{k: v for k, v in e.items() if k not in ('latencies', 'prefix')} for e in episodes],
        label='Capability qualification, not paper results; C3/C4 training-render provenance unverified')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(record, stream, indent=1)
        stream.write('\n')
    print(json.dumps({arm: dict(round_trips=f"{s['round_trips']}/{s['episodes']}", contacts=s['disallowed_contact_samples'],
        capable=s['capable']) for arm, s in summary.items()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
