"""Replay the unchanged local-memory state on recorded failure decisions."""
from copy import deepcopy
import argparse
import json
import math

import numpy as np

from lewm.interrupted_route_turn_memory_development import InterruptedRouteTurnMemory, TURNS
from lewm.selected_route_turn_memory_development import SelectedRouteTurnMemory
from lewm.clearance_turn_recovery_development import wrap
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts import run_go2_sparse_corner_replication_development as run


def main(probe=False):
    root = run.BASE / run.root_name(6)
    read = lambda name: json.loads((root / name).read_text())
    retention = root / 'depth_retention.json'
    if retention.exists():
        assert read('depth_retention.json')['full_sensor_replay_available']
    plans = read('planning.json')
    assert {r['phase'] for r in read('mission.json')} == {'OUTBOUND'}
    assert {r['mission_generation'] for r in plans if 'mission_generation' in r} == {0}
    poses = {r['frame']: r['registered_pose'] for r in read('poses.json')}
    support = {r['frame']: r for r in read('visual_support_recovery.json')}
    # Recreate the fixed map axes from the same first public force history.
    policy, *_ = NoisyPublicReplay(root / 'native').packet(0)
    force = policy['sensor_state']['sensed']['specific_force']
    command = policy['sensor_state']['control']['applied_command']
    assert force['valid'].all() and command['valid'].all()
    assert not np.any(np.abs(command['values']) > 1e-8)
    up = force['values'].mean(0)
    assert 8 <= np.linalg.norm(up) <= 12
    up = up / np.linalg.norm(up)
    forward = np.array([1., 0., 0.]) - up * up[0]
    forward /= np.linalg.norm(forward)
    B = np.stack((forward, np.cross(up, forward), up))
    memory = InterruptedRouteTurnMemory()
    rows, events, changes = [], [], []
    for plan in plans:
        frame = plan['frame']
        selection = plan['selection']
        pose = poses[frame]
        position = B @ np.asarray(pose['position_initial_body_m'])
        rotation = B @ np.asarray(pose['rotation_initial_body_from_current_body'])
        heading = math.atan2(rotation[1, 0], rotation[0, 0])
        receipt = support[frame]
        weak = receipt['trigger_ns'] if receipt['recovery_active'] else None
        if weak is not None:
            error = wrap(receipt['target_heading_rad'] - heading)
            assert abs(wrap(error - selection['scan_heading_error_rad'])) < 1e-12
        attempt = deepcopy(memory.attempt)
        new_trigger = weak is not None and weak > memory.last_trigger
        terminal = selection['prefix_aware_terminal_approach']['terminal_mode']
        if probe and not terminal:
            candidate = SelectedRouteTurnMemory()
            candidate.__dict__ = deepcopy(memory.__dict__)
            revised = candidate.select(deepcopy(selection), position, heading,
                plan.get('mission_generation', 0), weak)
            if revised['action'] != selection['action']:
                assert weak is None and 'scan_utilities' not in selection
                assert next(r for r in selection['memory_forecast_candidates']
                    if r['action'] == revised['action'])['nominal_predicted_path_clear']
                changes.append(dict(frame=frame, recorded=selection['action'],
                    revised=revised['action'], on_time=plan['on_time'],
                    memory=revised['visual_route_turn_memory']))
        result = selection if terminal else memory.select(deepcopy(selection), position,
            heading, plan.get('mission_generation', 0), weak)
        assert result['action'] == plan['action']
        assert result.get('visual_route_turn_memory') == selection.get('visual_route_turn_memory')
        if new_trigger:
            events.append(dict(frame=frame, trigger_ns=weak,
                attempt_direction=None if attempt is None else attempt['direction'],
                signed_progress_rad=None if attempt is None else
                    attempt['direction'] * wrap(heading - attempt['start_heading_rad']),
                distance_from_attempt_m=None if attempt is None else
                    float(np.linalg.norm(position[:2] - attempt['position'])),
                failed_record_added=attempt is not None and any(
                    f['trigger_ns'] == weak for f in memory.failed)))
        error = math.atan2(*selection['waypoint_body_xy_m'][::-1])
        target = wrap(heading + error)
        failed = sorted({f['direction'] for f in memory.failed
            if memory.nearby(f, position[:2], target)})
        by_action = {r['action']: r for r in selection['memory_forecast_candidates']}
        action = plan['action']
        other = 'right_turn' if action == 'left_turn' else 'left_turn'
        missed = (not terminal and weak is None and 'scan_utilities' not in selection
            and action in TURNS and TURNS[action] in failed
            and -TURNS[action] not in failed
            and by_action[other]['nominal_predicted_path_clear'])
        rows.append(dict(frame=frame, action=action,
            preferred_action=selection.get('before_memory_filter_action'),
            recovery_active=weak is not None, failed_directions=failed,
            selected_failed_turn_with_clear_unfailed_opposite=bool(missed),
            recorded_memory_active='visual_route_turn_memory' in selection))
    result = dict(schema='replication_turn_memory_replay.v1',
        all_recorded_actions_and_memory_flags_matched=True, plans=len(plans),
        map_axes_reconstructed_from_public_force=True,
        all_recorded_recovery_heading_errors_matched=True,
        trigger_events=events, rows=rows,
        recorded_failure_events=sum(e['failed_record_added'] for e in events),
        missed_clear_opposite_frames=[r['frame'] for r in rows
            if r['selected_failed_turn_with_clear_unfailed_opposite']],
        native_state_used=False, navigation_reexecuted=False,
        isolated_recorded_state_probes=changes if probe else None,
        counterfactual_state_propagated=False,
        alternative_navigation_outcome_proven=False)
    filename = 'turn_memory_selected_turn_probe_v1.json' if probe else 'turn_memory_failure_replay_v1.json'
    with (root / filename).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'trigger_events')}))
    if not probe:
        print('TRIGGER_EVENTS', json.dumps(events))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected-turn-probe', action='store_true')
    main(parser.parse_args().selected_turn_probe)
