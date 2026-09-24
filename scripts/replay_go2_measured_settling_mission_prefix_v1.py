"""Stop a saved-observation mission comparison at its first behavior difference."""
import json
from lewm.observed_round_trip_mission_development import ObservedRoundTripMission
from lewm.measured_settling_round_trip_mission_development import MeasuredSettlingRoundTripMission
from lewm.joint_floor_registered_evidence_development import current_joint_floor_registered_pose
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from scripts.diagnose_go2_return_transition_matches_v1 import INPUT, CASE, IDENTITIES
from scripts.navigation_artifact_root_development import BASE, create_output, verify_artifacts, artifact_path
from scripts.maze_decision_stream_development import read_rows
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = BASE / 'go2_measured_settling_mission_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_measured_settling_mission_prefix_v1_2026-09-09.md'
BINDINGS = IDENTITIES | {CASE+'/context_decisions.jsonl.gz':
    '23510c874165007e055cbd1fc49346df38fbad3b30b9d96ea68f22fd3f536aa9'}
BEHAVIOR_FIELDS = ('phase', 'active_goal_initial_body_xy_m', 'hold_required', 'terminal')


def main():
    if not __debug__:
        raise ValueError('assertions required')
    verify_artifacts(INPUT, BINDINGS)
    native = json.loads(artifact_path(INPUT, 'launch.json').read_text())
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_measured_settling_mission_prefix_v1.py',
        'lewm/measured_settling_round_trip_controller_development.py',
        'lewm/tests/test_measured_settling_round_trip_development.py'), native['source_sha256'])
    source_check(sources); resources = hardware()
    if resources['memory_available_bytes'] < 2*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+64*1024**2:
        raise ValueError('bounded saved-mission replay resources unavailable')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(input_sha256=BINDINGS, source_sha256=sources, hardware=resources,
        maximum_frames=1880, stopping_fields=list(BEHAVIOR_FIELDS), native_audit_pending_at_launch=True,
        cpu_processes=1, numerical_threads=1, native_scene_workers=0, model_training=False,
        observer_rerun=False, controller_rerun=False, model_loaded=False,
        minimum_available_ram_bytes=2*1024**3, output_allowance_bytes=64*1024**2,
        os_resource_limits_enforced=False,
        concurrency_reason='one bounded mission-state replay beside the existing native raw audit'))
    print('MEASURED_SETTLING_MISSION_PREFIX_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        old, candidate = [cls(public_mission(0), navigation_ticks=NAVIGATION_TICKS)
            for cls in (ObservedRoundTripMission, MeasuredSettlingRoundTripMission)]
        previous_command = [0., 0., 0.]; count = 0; first_counter = first_behavior = None
        terminal = None; last = None
        with (OUTPUT/'mission_receipts.jsonl').open('x') as stream:
            for row in read_rows(INPUT/CASE):
                i = row['tick']; d = row['decision']; now = 1_500_000_000+i*100_000_000
                if i >= 1880:
                    raise ValueError('bounded original mission population exceeded')
                if d['terminal'] is not None:
                    terminal = dict(frame=i, terminal=d['terminal']); break
                p, _, pose = current_joint_floor_registered_pose(d['evidence'], identity=(0,0,0), now_ns=now)
                assert pose['frame'] == i
                original = old.advance(p[:2], frame=i, now_ns=now, previous_requested_command=previous_command)
                assert original == d['mission_receipt'], 'original mission receipt differs at '+str(i)
                revised = candidate.advance(p, frame=i, now_ns=now, previous_requested_command=previous_command)
                assert revised['failure'] is None, revised['failure']
                if first_counter is None and revised['quiet_intervals'] != original['quiet_intervals']:
                    first_counter = i
                differences = [k for k in BEHAVIOR_FIELDS if revised[k] != original[k]]
                last = dict(frame=i, original=original, candidate=revised,
                    actual_previous_requested_command=previous_command, behavior_differences=differences)
                stream.write(json.dumps(last, allow_nan=False)+'\n'); stream.flush(); count += 1
                if i%200 == 0:
                    print('MEASURED_SETTLING_MISSION_PREFIX_FRAME', i, flush=True)
                if differences:
                    first_behavior = i; break
                previous_command = d['requested_command']
        verify_artifacts(INPUT, BINDINGS); source_check(sources)
        write_json(OUTPUT/'result.json', dict(status='MEASURED_SETTLING_MISSION_PREFIX_COMPLETE',
            source_sha256=sources, artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','mission_receipts.jsonl')},
            frames=count, first_quiet_counter_difference=first_counter,
            first_mission_behavior_difference=first_behavior, original_terminal=terminal, last=last,
            original_mission_receipts_exact=True, current_registered_pose_witnesses_validated=True,
            input_bytes_unchanged=True, stopped_before_later_decisions=first_behavior is not None,
            full_controller_replay=False, raw_sensor_audit_replaced=False,
            continuous_speed_certified=False, navigation_qualified=False))
        print('MEASURED_SETTLING_MISSION_PREFIX_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(reason=repr(error)))
        raise


if __name__ == '__main__':
    main()
