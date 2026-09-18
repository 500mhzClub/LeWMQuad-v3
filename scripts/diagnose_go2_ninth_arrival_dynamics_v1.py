"""Evaluator-only settling evidence at every saved late-episode frame."""
import json
import numpy as np
from lewm.physical_execution_development import rotation_xyzw
from scripts.navigation_artifact_root_development import BASE, artifact_path, verify_artifacts, create_output
from scripts.diagnose_go2_return_transition_matches_v1 import INPUT, CASE, IDENTITIES
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import source_check, hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = BASE / 'go2_ninth_arrival_dynamics_v1_attempt_001'
BINDINGS = IDENTITIES | {CASE + '/physics_trace.npz':
    'ce14c1c1cad3dccb3db81bca554a12142f7dbe247ec39c32473153df3c17fc49'}


def main():
    if not __debug__:
        raise ValueError('assertions required')
    verify_artifacts(INPUT, BINDINGS)
    launch = json.loads(artifact_path(INPUT, 'launch.json').read_text())
    collection = json.loads(artifact_path(INPUT, CASE + '/result.json').read_text())
    sources = discover_sources(('scripts/diagnose_go2_ninth_arrival_dynamics_v1.py',), launch['source_sha256'])
    source_check(sources); resources = hardware()
    if resources['memory_available_bytes'] < 1024**3 or resources['artifact_free_bytes'] < 40*1024**3+1024**2:
        raise ValueError('bounded dynamics diagnosis resources unavailable')
    create_output(OUTPUT)
    write_json(OUTPUT / 'launch.json', dict(input_sha256=BINDINGS, source_sha256=sources,
        frame_range_inclusive=[1850, 1880], full_population_no_selected_window=True,
        hardware=resources, cpu_processes=1, numerical_threads=1, native_scene_workers=0,
        evaluator_only=True, model_training=False, input_native_state_to_controller=False,
        minimum_available_ram_bytes=1024**3, output_allowance_bytes=1024**2,
        os_resource_limits_enforced=False))
    try:
        with np.load(artifact_path(INPUT, CASE + '/physics_trace.npz'), allow_pickle=False) as z:
            pose, twist, request, times = [z[n] for n in
                ('base_pose_world', 'base_twist_world', 'requested_command', 'timestamp_s')]
        assert len(pose) == collection['physics_samples'] == 94750
        assert np.allclose(times, np.arange(1, len(times)+1)*.002, atol=5e-9, rtol=0)
        arrival = collection['mission_receipt']['arrivals'][0]
        assert arrival['frame'] == 1866 and arrival['phase'] == 'OUTBOUND'
        local = (pose[:, :3]-pose[749, :3])@rotation_xyzw(pose[749, 3:])
        target = np.asarray(arrival['target_initial_body_xy_m'])
        distance = np.linalg.norm(local[:, :2]-target, axis=1)
        speed = np.linalg.norm(twist[:, :3], axis=1)
        rows = []
        for frame in range(1850, 1881):
            end = 749+50*frame; start = end-500
            assert 0 <= start < end < len(pose)
            peak = start+int(np.argmax(speed[start:end+1]))
            quiet_speed = bool((speed[start:end+1] <= .05).all())
            quiet_command = bool((request[start+1:end+1] == 0.).all())
            within = bool((distance[start:end+1] <= .06).all())
            rows.append(dict(frame=frame, maximum_100ms_speed_m_s=float(speed[end-49:end+1].max()),
                maximum_100ms_horizontal_speed_m_s=float(np.linalg.norm(twist[end-49:end+1, :2], axis=1).max()),
                maximum_100ms_vertical_speed_m_s=float(np.abs(twist[end-49:end+1, 2]).max()),
                current_speed_m_s=float(speed[end]), current_goal_distance_m=float(distance[end]),
                previous_100ms_request_zero=bool((request[end-49:end+1] == 0.).all()),
                one_second_maximum_speed_m_s=float(speed[peak]),
                one_second_peak_sample=peak, one_second_peak_twist_world=twist[peak].tolist(),
                one_second_speed_pass=quiet_speed, one_second_request_pass=quiet_command,
                one_second_distance_pass=within, one_second_arrival_pass=quiet_speed and quiet_command and within,
                observed_arrival_claim=frame == arrival['frame']))
        verify_artifacts(INPUT, BINDINGS); source_check(sources)
        write_json(OUTPUT / 'result.json', dict(status='NINTH_ARRIVAL_DYNAMICS_DIAGNOSIS_COMPLETE',
            artifact_sha256={'launch.json': digest(OUTPUT / 'launch.json')}, source_sha256=sources,
            windows=rows, input_bytes_unchanged=True, evaluator_only=True,
            original_arrival_relabelled=False, raw_sensor_command_audit_replaced=False,
            navigation_qualified=False, verified_round_trip=False))
        print('NINTH_ARRIVAL_DYNAMICS_DIAGNOSIS_COMPLETE', digest(OUTPUT / 'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT / 'failure.json', dict(reason=repr(error)))
        raise


if __name__ == '__main__':
    main()
