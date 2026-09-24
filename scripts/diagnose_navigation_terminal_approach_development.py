"""Separate observed terminal approach from route duration in completed comparisons."""
import argparse
import json
from scripts.compare_continuous_navigation_arms_development import path, read


def phases(root):
    mission = read(root, 'mission.json')
    requests = read(root, 'requests.json')
    result = {}
    for phase in ('OUTBOUND', 'RETURN'):
        rows = [r for r in mission if r.get('evaluated_phase') == phase]
        close = [r for r in rows if r.get('observed_goal_distance_m') is not None
            and r['observed_goal_distance_m'] <= .1]
        if not close:
            result[phase] = dict(entered_10cm=False)
            continue
        first = close[0]
        arrivals = [r for r in rows if r.get('arrival_confirmed_this_frame')]
        last = arrivals[0] if arrivals else rows[-1]
        start, end = first['measured_ns'], last['measured_ns']
        commands = dict(translation=0., turn_only=0., zero=0.)
        for request in requests:
            now = request['now_ns']
            if not start <= now < end:
                continue
            command = request['applied_command']
            kind = 'translation' if any(command[:2]) else 'turn_only' if command[2] else 'zero'
            commands[kind] += min(20_000_000, end-now)/1e9
        observed = [r['observed_goal_distance_m'] for r in rows
            if start <= r['measured_ns'] <= end and r.get('observed_goal_distance_m') is not None]
        result[phase] = dict(entered_10cm=True, first_close_frame=first['frame'],
            final_frame=last['frame'], observed_arrival=bool(arrivals),
            measured_approach_seconds=(end-start)/1e9,
            minimum_observed_distance_m=min(observed), maximum_observed_distance_m=max(observed),
            left_10cm_after_first_entry=any(d>.1 for d in observed),
            command_seconds=commands, accounted_command_seconds=sum(commands.values()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison-root-name', required=True)
    root = path(parser.parse_args().comparison_root_name)
    report = read(root, 'result.json')
    output = dict(conditions={row['condition']:phases(path(row['root_name'])) for row in report['rows']},
        definition='First observed distance <= 0.10 m through observed arrival, or phase end if unsuccessful',
        physical_arrival_checks_remain_separate=True, zero_command_is_not_physical_stationarity=True,
        recorded_trajectory_diagnostic_only=True)
    with (root/'terminal_approach_diagnostic_v1.json').open('x') as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
