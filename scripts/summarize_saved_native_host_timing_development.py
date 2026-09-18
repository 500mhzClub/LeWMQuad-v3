"""Describe recorded host timing without rerunning sensors, models or physics."""
import argparse
import json
import numpy as np
from scripts.compare_continuous_navigation_arms_development import path, read


def distribution_ns(values):
    a = np.asarray(values, dtype=float) / 1e6
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('nonempty finite recorded timing population required')
    return dict(samples=len(a), median_ms=float(np.median(a)),
        p95_ms=float(np.percentile(a, 95)), p99_ms=float(np.percentile(a, 99)),
        maximum_ms=float(a.max()))


def summarize(root):
    launch = read(root, 'launch.json')
    if launch['deadline_clock'] != 'measured_simulation':
        raise ValueError('this analysis describes measured-simulation runs')
    # Completed success and failed assignments are both included.
    if not (root/'continuous_native_arrival_evaluation.json').exists():
        raise ValueError('independently evaluated completed recording required')
    requests = read(root, 'requests.json')
    release_log_available = (root/'measured_latency_releases.json').exists()
    releases = read(root, 'measured_latency_releases.json') if release_log_available else []
    acquisitions = read(root, 'acquisitions.json')
    plans = [p for p in read(root, 'planning.json') if 'selection' in p]
    lag = [r['simulator_lag_ns'] for r in requests]
    if any(r['released_ns'] < r['earliest_release_ns'] for r in releases):
        raise ValueError('worker result released before its measured service duration')
    return dict(root_name=root.name,
        acquisition=distribution_ns([r['measured_acquisition_wall_ns'] for r in acquisitions]),
        service_by_release_stage={stage: distribution_ns([
            r['measured_service_ns'] for r in releases if r['stage'] == stage])
            for stage in sorted({r['stage'] for r in releases})},
        service_release_log_available=release_log_available,
        service_records_may_be_cumulative_within_worker_item=True,
        host_minus_simulation_lag=distribution_ns(lag),
        requests_above_existing_wall_mode_20ms_lag_limit=sum(t > 20_000_000 for t in lag),
        selected_plans=len(plans), on_time_plans=sum(bool(p['on_time']) for p in plans),
        measured_simulation_plan_age=distribution_ns([
            p['completed_ns']-p['measured_ns'] for p in plans]),
        result_wall_s=read(root, 'result.json')['wall_s'] if (root/'result.json').exists() else None,
        result_simulation_s=read(root, 'result.json')['simulation_s'] if (root/'result.json').exists() else None,
        every_worker_release_respects_measured_service=True if release_log_available else None,
        wall_mode_outcome_inferred=False, host_real_time_qualified=False,
        robot_hardware_timing_measured=False,
        explanation='Lag is measured before each simulated command service. Exceeding the existing wall-mode limit describes this recording; rerunning with wall deadlines changes commands and trajectories.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root-name', required=True)
    root = path(parser.parse_args().root_name)
    target = root/'host_timing_diagnostic_v1.json'
    if target.exists(): raise ValueError('preserve completed timing analysis')
    result = summarize(root)
    with target.open('x') as sink: json.dump(result, sink, indent=2)
    print(json.dumps(result))


if __name__ == '__main__': main()
