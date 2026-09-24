"""Fresh complete-controller replay through the first floor-transport intervention."""
import argparse
from itertools import islice
import json
import time
import cv2
import torch

from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
from lewm.measured_floor_transport_development import current_measured_floor_pose, SCHEMA
from lewm.measured_floor_transport_prefix_development import compare_prior, admit_native
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.run_go2_dual_camera_settled_maze_pilot_v1 import OUTPUT as INPUT, CASE, verify_inputs as verify_native
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.maze_decision_stream_development import read_rows, writer, NAME as STREAM
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT = BASE/'go2_measured_floor_transport_prefix_v1_attempt_001'
DIAGNOSIS = BASE/'go2_dual_camera_floor_extent_diagnosis_v1_attempt_001'
DIAGNOSIS_RESULT = '1677bd7be89e5dd85c2e467693575aa5144f3f72fcef09ab16e8afa38b513a35'
PROTOCOL = 'docs/go2_measured_floor_transport_prefix_v1_2026-09-09.md'
MAX_OUTPUT_BYTES = 1024**3


def verify_all(launch):
    source_check(launch['source_sha256'])
    verify_artifacts(INPUT, launch['native_artifact_sha256'])
    verify_artifacts(DIAGNOSIS, launch['diagnosis_artifact_sha256'])
    verify_native(read_json(INPUT, 'launch.json'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--native-result-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive floor-transport prefix required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    verify_artifacts(INPUT, {'result.json': args.native_result_sha256})
    native = read_json(INPUT, 'result.json')
    native_ids = {'result.json': args.native_result_sha256, **native['artifact_sha256']}
    verify_artifacts(INPUT, native_ids)
    admit_native(native, read_json(INPUT, CASE[0]+'_audit.json'), case=CASE[0])
    verify_artifacts(DIAGNOSIS, {'result.json': DIAGNOSIS_RESULT})
    diagnosis = read_json(DIAGNOSIS, 'result.json')
    if (diagnosis['status'] != 'DUAL_CAMERA_FLOOR_EXTENT_DIAGNOSIS_COMPLETE'
            or diagnosis['original_rejection_reconstructed'] is not True
            or diagnosis['first_failure_frame'] != 1904 or diagnosis['thresholds_changed'] is not False):
        raise ValueError('completed unchanged floor-extent diagnosis required')
    diagnosis_ids = {'result.json': DIAGNOSIS_RESULT, **diagnosis['artifact_sha256']}
    verify_artifacts(DIAGNOSIS, diagnosis_ids)
    selected = read_json(DIAGNOSIS, 'launch.json')['closed_input_sha256']
    if len(selected) != 65 or any(native_ids.get(n) != h for n, h in selected.items()):
        raise ValueError('all selected diagnosis inputs must match the final independently audited artifact map')
    old = read_json(INPUT, 'launch.json'); inherited = dict(old['source_sha256'])
    for n, h in diagnosis['source_sha256'].items():
        if n in inherited and inherited[n] != h: raise ValueError('incompatible frozen diagnosis source: '+n)
        inherited[n] = h
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_measured_floor_transport_prefix_v1.py',
        'lewm/tests/test_measured_floor_transport_development.py',
        'lewm/tests/test_measured_floor_transport_prefix_development.py'), inherited)
    resources = hardware()
    launch = dict(protocol=PROTOCOL, output_root=str(OUTPUT), source_sha256=sources,
        native_artifact_sha256=native_ids, diagnosis_artifact_sha256=diagnosis_ids,
        correction_admission=old['correction_admission'], model_state_sha256=old['prefix_report']['model_state_sha256'],
        hardware=resources, frames=1905, expected_first_intervention_frame=1904,
        diagnosis_bindings_match_final_native_artifacts=True, diagnosis_binding_count=len(selected),
        cpu_processes=1, numerical_threads=1, opencv_threads=1, native_scene_workers=0,
        memory_admission_bytes=8*1024**3, output_allowance_bytes=MAX_OUTPUT_BYTES,
        minimum_free_bytes=RESERVE_BYTES, os_resource_limits_enforced=False,
        concurrency_reason='one CPU complete-controller replay; independent bounded renderer probe may overlap after resource inspection',
        native_execution=False, model_training=False, map_performance_candidate_adopted=False,
        navigation_qualified=False, real_time_qualified=False, goal_achieved=False)
    verify_all(launch)
    memory_ok = resources['memory_available_bytes'] >= 8*1024**3
    storage_ok = resources['artifact_free_bytes'] >= RESERVE_BYTES+MAX_OUTPUT_BYTES
    if args.preflight_only:
        print('MEASURED_FLOOR_TRANSPORT_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            memory_admission_pass=memory_ok, storage_admission_pass=storage_ok,
            completed_inputs_and_sources_verified=True, diagnosis_bindings_match_final_native_artifacts=True,
            output_created=False)), flush=True); return
    if not memory_ok or not storage_ok: raise ValueError('floor transport replay resource admission failed')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('MEASURED_FLOOR_TRANSPORT_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        name, index, variant, condition, model_name = CASE
        model, c, v = load_assigned(launch['correction_admission'], model_name)
        assert (c, v) == (condition, variant)
        before = state_digest(model.state_dict()); assert before == launch['model_state_sha256']
        controller = MeasuredFloorTransportController(model, ArticulatedCollisionGeometry(URDF),
            public_mission=public_mission(index), navigation_ticks=NAVIGATION_TICKS,
            persistent=True, condition=c, variant=v)
        directory = INPUT/name; reader = IntentReturnRGBDReplay(directory)
        acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
        count = 0; prior = None
        with writer(OUTPUT) as append:
            for i, saved in enumerate(islice(read_rows(directory), 1905)):
                if saved['tick'] != i: raise ValueError('ordered fixed controller prefix required')
                p, d, f, now = reader.packet(i)
                image, auxiliary = packet(directory, i, p, public_acquisition(acquisitions[i]), now_ns=now)
                inputs = fingerprint((p, d, f, auxiliary, image, now)); began = time.perf_counter_ns()
                candidate = controller.observe(p, d, f, now_ns=now, auxiliary_depth=auxiliary, auxiliary_rgb=image)
                elapsed = (time.perf_counter_ns()-began)/1e6
                normalized = json.loads(json.dumps(candidate, allow_nan=False))
                if fingerprint((p, d, f, auxiliary, image, now)) != inputs:
                    raise ValueError('public input arrays mutated at frame '+str(i))
                try:
                    if i < 1904:
                        compare_prior(saved, normalized, tape[i]['requested_command'], frame=i)
                        prior = saved['decision']
                    else:
                        original = saved['decision']; evidence = candidate['evidence']
                        if (original['failure'] != 'admitted combined measured moments must reconstruct exactly'
                                or original['evidence'] is not None or original['terminal'] != 'SENSOR_OR_MODEL_FAILURE'
                                or normalized['original_visual_evidence'] != original['original_visual_evidence']):
                            raise ValueError('same raw current visual evidence and original floor failure required')
                        current_measured_floor_pose(evidence, identity=(0, 0, 0), now_ns=now)
                        if (evidence['schema'] != SCHEMA or normalized['terminal'] is not None
                                or normalized['failure'] is not None or normalized['mission_receipt']['phase'] != 'RETURN'
                                or normalized['evidence']['floor_transport']['anchor'] != prior['evidence']
                                or evidence['floor_transport']['correction']['anchor_frame'] != 1903):
                            raise ValueError('active RETURN from exactly the last admitted floor anchor required')
                        write_json(OUTPUT/'intervention.json', dict(frame=i, original_failed_decision=original,
                            candidate_decision=normalized, current_pose_validated=True,
                            first_intervention=True, following_recorded_observations_consumed=False))
                except Exception:
                    write_json(OUTPUT/'mismatch.json', dict(frame=i, original=saved['decision'], candidate=normalized)); raise
                append(dict(tick=i, decision=normalized, preintervention_complete_decision_exact=i<1904,
                    input_arrays_unchanged=True, controller_wall_ms=elapsed)); count += 1
                if (OUTPUT/STREAM).stat().st_size > MAX_OUTPUT_BYTES//2: raise ValueError('compressed output headroom exceeded')
                if i%100 == 0: print('MEASURED_FLOOR_TRANSPORT_FRAME', i, flush=True)
        assert count == 1905 and state_digest(model.state_dict()) == before
        assert all(p.grad is None for p in model.parameters())
        verify_all(launch)
        bindings = {n:digest(OUTPUT/n) for n in ('launch.json', STREAM, 'intervention.json')}
        verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='MEASURED_FLOOR_TRANSPORT_PREFIX_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, frames=count, first_intervention_frame=1904,
            complete_preintervention_decisions_exact_outside_validated_labels=True,
            all_1904_prior_actual_commands_exact=True, raw_visual_evidence_exact_at_intervention=True,
            current_transport_pose_available=True, active_return_at_intervention=True,
            final_requested_command=normalized['requested_command'], model_state_unchanged=True,
            model_state_sha256=before, public_input_arrays_unchanged=True,
            diagnosis_bindings_match_final_native_artifacts=True,
            following_recorded_observations_consumed=False, native_execution=False,
            navigation_qualified=False, real_time_qualified=False, goal_achieved=False,
            wall_s=time.perf_counter()-start, hardware_after=hardware()))
        print('MEASURED_FLOOR_TRANSPORT_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(reason=repr(error))); raise


if __name__ == '__main__': main()
