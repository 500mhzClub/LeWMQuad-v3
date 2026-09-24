"""Read-only reconstruction of the raw-audited frame-504 floor rejection."""
import argparse
from itertools import islice
import json
import time
import cv2
from lewm.floor_transport_conflict_readout_development import reconstruct, FAILURE
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES
from scripts.run_go2_direct_flow_maze01_pilot_v1 import OUTPUT as INPUT, CASE, verify_inputs as verify_native
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT = BASE/'go2_direct_flow_maze01_floor_conflict_v1_attempt_001'
PROTOCOL = 'docs/go2_direct_flow_maze01_floor_conflict_v1_2026-09-09.md'
BOUNDARY = 504
ALLOWANCE = 128*1024**2


def admit(result, launch, audit):
    if (result['status'] != 'DIRECT_FLOW_MAZE01_PILOT_V1_COMPLETE' or len(result['conditions']) != 1
            or launch['planned_case'] != list(CASE) or launch['implementation_class'] != 'DirectFlowFloorTransportController'
            or launch['prefix_report']['model_state_sha256'] != MODEL_STATE):
        raise ValueError('completed exact maze1 tracking attempt required')
    record = result['conditions'][0]
    if (record['case'] != CASE[0] or record['layout_index'] != 1 or audit['layout_index'] != 1
            or record['status'] != 'DIRECT_FLOW_MAZE01_COLLECTED_AND_RAW_AUDITED'
            or record['model_state_unchanged'] is not True):
        raise ValueError('complete matching collection and raw audit required')
    require_raw_audit(record, audit, learned=True)
    expected = dict(common_prefix_frames=215, first_intervention_frame=214, physical_prefix_samples=11450,
        physical_and_public_prefix_exact=True, all_preintervention_observed_state_exact=True,
        all_preintervention_requested_commands_exact=True, complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=211, all_compared_raw_model_forecasts_exact=True,
        original_intervention_command=[0., 0., 0.], candidate_intervention_command=[0., 0., .45],
        candidate_intervention_command_completed=True, following_physical_outcomes_compared=False,
        unexecuted_outcomes_inferred=False)
    for key, value in expected.items():
        actual = record['prefix_comparison'][key]
        if actual != value or type(actual) is not type(value): raise ValueError('complete physical prefix required: '+key)
    if (record['collection']['rgbd_frames'] != 515 or record['collection']['completed_ticks'] != 514
            or record['collection']['schedule_terminal'] != 'SENSOR_OR_MODEL_FAILURE'):
        raise ValueError('exact completed frame-504 failure population required')


def verify_all(launch):
    verify(launch)
    verify_artifacts(INPUT, launch['diagnostic_input_bindings'])
    verify_native(read_json(INPUT, 'launch.json'))


def terminal_pair(rows):
    prior = last = None; count = 0
    for i, row in enumerate(islice(rows, BOUNDARY+1)):
        if row['tick'] != i or row['observation_index'] != i or row['pre_sample_index'] != 749+50*i:
            raise ValueError('exact ordered original observation endpoints required')
        d = row['decision']
        if i < BOUNDARY and (d['terminal'] is not None or d['failure'] is not None):
            raise ValueError('no earlier terminal or failure permitted')
        prior, last = last, row; count += 1
    if count != BOUNDARY+1 or prior is None: raise ValueError('complete prefix through first failure required')
    d = last['decision']
    if (d['terminal'] != 'SENSOR_OR_MODEL_FAILURE' or d['failure'] != FAILURE or d['evidence'] is not None
            or d['requested_command'] != [0., 0., 0.] or d['original_visual_evidence'] is None
            or prior['decision']['evidence'] is None):
        raise ValueError('exact observed floor conflict and preceding admitted evidence required')
    return prior, last


def diagnose():
    directory = INPUT/CASE[0]
    prior, terminal = terminal_pair(read_rows(directory))
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    if len(reader.frames) != 515 or len(acquisitions) != 515 or len(tape) != 514:
        raise ValueError('complete fixed paired observation population required')
    for row in (prior, terminal):
        item = tape[row['tick']]
        if not item['completed'] or item['requested_command'] != row['decision']['requested_command']:
            raise ValueError('actual completed boundary requests required')
    p, d, f, now = reader.packet(BOUNDARY)
    image, auxiliary = packet(directory, BOUNDARY, p, public_acquisition(acquisitions[BOUNDARY]), now_ns=now)
    inputs = (prior['decision']['evidence'], terminal['decision']['original_visual_evidence'], p, d, auxiliary, image)
    before = fingerprint((inputs, f))
    report = reconstruct(*inputs, now_ns=now)
    if fingerprint((inputs, f)) != before: raise ValueError('diagnosis mutated admitted evidence or public arrays')
    return report|dict(public_inputs_and_saved_evidence_unchanged=True,
        original_decisions_checked=BOUNDARY+1, raw_packet_frames_loaded=[BOUNDARY],
        following_decisions_or_packets_consumed=False, model_loaded=False, model_training=False,
        native_execution=False, reused_development_layout=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--native-result-sha256', required=True); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive floor conflict diagnosis required')
    verify_artifacts(INPUT, {'result.json':args.native_result_sha256}); result = read_json(INPUT, 'result.json')
    ids = dict(result['artifact_sha256']); ids['result.json'] = args.native_result_sha256
    verify_artifacts(INPUT, ids); old = read_json(INPUT, 'launch.json'); verify_native(old)
    admit(result, old, read_json(INPUT, CASE[0]+'_audit.json'))
    sources = discover_sources((PROTOCOL, 'scripts/diagnose_go2_direct_flow_maze01_floor_conflict_v1.py',
        'lewm/tests/test_floor_transport_conflict_readout_development.py'), result['source_sha256'])
    launch = old|dict(protocol=PROTOCOL, output_root=str(OUTPUT), source_sha256=sources,
        diagnostic_input_bindings=ids, implementation_class='unchanged MeasuredFloorTransportRegistration diagnostic',
        original_implementation_class=old['implementation_class'],
        native_result_sha256=args.native_result_sha256, first_failure_frame=BOUNDARY,
        model_loaded=False, model_training=False, native_execution=False, native_scene_workers=0,
        diagnosis_workers=1, numerical_threads=1, memory_admission_bytes=8*1024**3,
        output_allowance_bytes=ALLOWANCE, original_controller_reexecuted=False,
        saved_raw_visual_witnesses_already_raw_audited=True, original_failure_required=FAILURE)
    verify_all(launch); resources = hardware(); launch['hardware'] = resources
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+ALLOWANCE:
        raise ValueError('bounded floor conflict diagnosis resources unavailable')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('DIRECT_FLOW_FLOOR_CONFLICT_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        report = diagnose(); verify_all(launch)
        bindings = {'launch.json':digest(OUTPUT/'launch.json')}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='DIRECT_FLOW_MAZE01_FLOOR_CONFLICT_V1_COMPLETE',
            report=report, source_sha256=sources, artifact_sha256=bindings, wall_s=time.perf_counter()-started,
            native_result_sha256=args.native_result_sha256, model_loaded=False, model_training=False,
            native_execution=False, navigation_qualified=False, goal_achieved=False))
        print('DIRECT_FLOW_FLOOR_CONFLICT_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_DIRECT_FLOW_FLOOR_CONFLICT_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
