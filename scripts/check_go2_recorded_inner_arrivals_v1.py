"""Read-only fixed-table reachability at all recorded completion states.

This is not counterfactual execution: changed commands would change subsequent
images, body states, budget consumption and future goal anchors. No native
pose or fitted correction enters the candidate generation.
"""
import json
from dataclasses import fields
import numpy as np
from lewm.inner_goal_pulse_rollout_development import plan
from lewm.coupled_pulse_rollout_development import compose
from lewm.anchored_pulse_servo_development import AnchoredPulseServo
from lewm.sensor_anchored_goal_development import AnchoredGoal
from scripts.fixed_nominal_pulse_table_development import load_fixed_table
from scripts.replay_go2_balanced_features_v1 import INPUT, OUTPUT
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest


def check():
    launch = read_json(OUTPUT, 'launch.json')
    sources = discover_sources(('scripts/check_go2_recorded_inner_arrivals_v1.py',
        'lewm/inner_goal_pulse_feedback_development.py',
        'lewm/inner_goal_room_return_development.py',
        'lewm/tests/test_inner_goal_pulse_development.py'), launch['source_sha256'])
    verify_bindings(sources | launch['input_sha256'])
    verify_artifacts(INPUT, launch['external_input_sha256'])
    table = load_fixed_table()
    reports = {}
    for condition in ('nominal_left', 'nominal_right', 'lower_friction_left'):
        rows = read_json(INPUT / condition, 'servo_decisions.json')
        selected = []
        for row in rows:
            local = row['decision']['execution']['local_decision']
            if local is not None and local['terminal'] == 'VISUAL_TARGET_SEQUENCE_COMPLETE':
                selected.append((row, local))
        # No completed low-friction goal: retain its final available stop state.
        if not selected:
            local = rows[-1]['decision']['execution']['local_decision']
            if local and local['diagnostic'].get('position_initial_body_m') is not None:
                selected.append((rows[-1], local))
        reports[condition] = []
        for row, local in selected:
            d, goal = local['diagnostic'], local['goal']
            start = [*d['position_initial_body_m'][:2], d['unwrapped_yaw_rad']]
            target = [*goal['target_xy'], d['target_unwrapped_yaw_rad']]
            remaining = 35 - local['pulse_count']
            mission_remaining = 140 - row['decision']['execution']['mission_pulses']
            horizon = min(24, remaining, mission_remaining)
            record = dict(tick=row['tick'], old_terminal=local['terminal'],
                anchor_frame=goal['anchor_frame'], start=start, target=target,
                old_visual_position_error_m=d['position_error_m'],
                remaining_local_pulses=remaining, remaining_mission_pulses=mission_remaining,
                remaining_local_ticks=1000-(local['decision_ns']-goal['anchor_ns'])//100000000,
                remaining_mission_ticks=3600-row['tick'])
            if horizon <= 0:
                record.update(candidate=None, status='NO_REMAINING_PULSE_BUDGET')
            else:
                candidate = plan(table, start, target, yaw_mode='winding', horizon=horizon)
                decoded = {f.name: goal[f.name] for f in fields(AnchoredGoal)}
                decoded['identity'] = tuple(decoded['identity'])
                checker = AnchoredPulseServo(AnchoredGoal(**decoded))
                state, rejected = np.array(start), []
                for step, index in enumerate(candidate['action_indices']):
                    state = compose(state, table.effects[index].delta_xy_yaw)
                    if checker._excursion(np.r_[state[:2], d['position_initial_body_m'][2]]):
                        rejected.append(step)
                record.update(candidate=candidate,
                    predicted_endpoint_excursion_rejections=rejected,
                    minimum_additional_ticks_including_final_hold=candidate['minimum_command_ticks']+10,
                    status='OFFLINE_FIXED_TABLE_CANDIDATE_ONLY')
            reports[condition].append(record)
    verify_bindings(sources | launch['input_sha256'])
    verify_artifacts(INPUT, launch['external_input_sha256'])
    return dict(status='RECORDED_ARRIVAL_INNER_REGION_DIAGNOSTIC', conditions=reports,
        replay_launch_sha256=digest(OUTPUT/'launch.json'),
        new_source_sha256={p: h for p, h in sources.items() if p not in launch['source_sha256']},
        native_pose_input=False, physical_execution=False,
        counterfactual_success_established=False, goal_achieved=False)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
