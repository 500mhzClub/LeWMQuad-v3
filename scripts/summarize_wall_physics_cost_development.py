"""Summarize measured native service overload and optional component timers."""
import argparse
import json
import statistics
from scripts.compare_continuous_navigation_arms_development import path, read


def stats(values):
    return dict(samples=len(values), mean_ms=statistics.mean(values)/1e6,
        median_ms=statistics.median(values)/1e6, maximum_ms=max(values)/1e6)


def summarize(root):
    requests = read(root, 'requests.json')
    components = (read(root, 'command_component_timings.json')
        if (root/'command_component_timings.json').exists() else None)
    if components is not None:
        assert len(components) == len(requests)
        assert all(a['simulator_ns'] == b['simulator_ns'] for a,b in zip(requests, components))
        assert all(r['physics_step_calls'] == r['sensor_guard_recording_calls'] == 10
            and r['gait_inference_calls'] == 1 for r in components)
    windows = []
    epoch = 1_500_000_000
    last = (requests[-1]['simulator_ns']-epoch)//1_000_000_000
    for start in range(0, int(last)+1, 20):
        indexes = [i for i,r in enumerate(requests)
            if start*10**9 <= r['simulator_ns']-epoch < (start+20)*10**9]
        rows = [requests[i] for i in indexes]
        window = dict(start_simulation_s=start, requests=len(rows),
            native_service=stats([r['physical_service_completed_wall_ns']-
                r['physical_service_started_wall_ns'] for r in rows]),
            host_lag=stats([r['simulator_lag_ns'] for r in rows]))
        if components is not None:
            subset = [components[i] for i in indexes]
            fields = ('physics_step_wall_ns', 'sensor_guard_recording_wall_ns',
                'gait_inference_wall_ns', 'total_wall_ns', 'owner_thread_cpu_ns')
            window['components'] = {k:stats([r[k] for r in subset]) for k in fields}
            window['wall_minus_owner_thread_cpu'] = stats([
                r['total_wall_ns']-r['owner_thread_cpu_ns'] for r in subset])
        windows.append(window)
    return dict(root_name=root.name, service_period_ms=20, windows=windows,
        complete_ordered_component_receipts=components is not None,
        request_cost=stats([r['request_finished_wall_ns']-r['request_started_wall_ns'] for r in requests]),
        post_request_pre_service_gap=stats([r['physical_service_started_wall_ns']-
            r['request_finished_wall_ns'] for r in requests]),
        owner_cpu_excludes_native_worker_threads=True,
        wall_minus_owner_cpu_is_not_exclusively_scheduler_wait=True,
        one_execution_not_a_causal_speed_comparison=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    root=path(parser.parse_args().root_name);result=summarize(root)
    with (root/'wall_physics_cost_diagnostic_v1.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
