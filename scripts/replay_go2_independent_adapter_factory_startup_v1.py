"""Exercise all four real factory/controller paths on an old four-packet startup."""
from itertools import islice
import json
import time
import cv2
import numpy as np
import torch
from lewm.independent_round_trip_comparison_study_development import CASES, require_case
from lewm.independent_round_trip_multiarm_contract_development import verify_model
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.all_phase_adapter_prefix_comparison_development import compare_observed
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.observation_horizon_predictive_selection_development import candidate_inputs
from lewm.observation_horizon_input_ablation_development import transform_inputs
from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import independent_round_trip_controller_factory_development as old_factory
from scripts import independent_round_trip_adapter_controller_factory_development as new_factory
from scripts import replay_go2_all_phase_planner_adapter_startup_v2 as original
from scripts import all_phase_adapter_native_evidence_development as evidence
from scripts.all_phase_residual_maze02_native_inputs_development import verify_bound_inputs
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.maze_decision_stream_development import writer, NAME
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.replay_go2_residual_current_observation_planning_prefix_v1 import state_tree
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_independent_adapter_factory_startup_v1_attempt_001'
SOURCE = 'scripts/replay_go2_independent_adapter_factory_startup_v1.py'
PROTOCOL = 'docs/go2_independent_adapter_factory_startup_v1_2026-09-10.md'
TEST = 'lewm/tests/test_independent_round_trip_adapter_integration_development.py'
VERIFICATION = 'docs/go2_independent_round_trip_multiarm_integration_verification_2026-09-10.json'
VERIFICATION_SHA = '3e825e87a43b7958be94a57db64a02cf8a354185d3c369288905eb35304bc76a'
QUEUE = BASE/'go2_reached_frontier_maze03_native_wait_v1_attempt_001'
QUEUE_SHA = '68c10ea5a869d6236975372a525dc4586ba7ba16cbeefb17ff0fbb2b57c07a74'
INPUT_CASE = 'all_phase_full_jepa_residual_maze_02'


def run_case(case, admission):
    arm = require_case(case); geometry = ArticulatedCollisionGeometry(URDF)
    pairs = [factory.create(case, geometry, correction_admission=admission) for factory in (old_factory, new_factory)]
    old_controller, old_model = pairs[0]; new_controller, new_model = pairs[1]
    for model in (old_model, new_model): verify_model(case, model)
    if type(old_controller) is not type(new_controller) or old_controller is new_controller:
        raise ValueError('same original assigned controller class and fresh instances required')
    directory = original.INPUT/INPUT_CASE; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    output = OUTPUT/case.arm_name; output.mkdir(); boundary = None; all_forward = False
    with writer(output) as append:
        for frame in range(4):
            if not tape[frame]['completed'] or tape[frame]['tick'] != frame:
                raise ValueError('actual original completed startup commands required')
            p, d, fast, now = reader.packet(frame)
            image, auxiliary = packet(directory, frame, p, public_acquisition(acquisitions[frame]), now_ns=now)
            public = fingerprint((p, d, fast, auxiliary, image, now))
            old, new = [c.observe(p, d, fast, now_ns=now, auxiliary_depth=auxiliary, auxiliary_rgb=image)
                for c in (old_controller, new_controller)]
            if public != fingerprint((p, d, fast, auxiliary, image, now)):
                raise ValueError('raw public inputs changed')
            if frame < 3 and (old != new or new['requested_command'] != tape[frame]['requested_command']
                    or new['requested_command'] != [0., 0., 0.] or new['terminal'] is not None):
                raise ValueError('all actual three zero warmups and paired complete decisions must match')
            contacts = [fingerprint(state_tree(c.memory)) for c in (old_controller, new_controller)]
            if (contacts[0] != contacts[1] or old_controller.mapper.floor != new_controller.mapper.floor
                    or old_controller.mapper.occupied != new_controller.mapper.occupied):
                raise ValueError('unchanged observed map and retained contact evidence required')
            if arm.name == 'reactive':
                if (old != new or old_model is not None or new_model is not None
                        or new['candidate_future_outcomes_evaluated'] is not False):
                    raise ValueError('complete unchanged reactive no-model path required')
            else:
                compare_observed(old, new, frame=frame)
                if frame == 3:
                    selection = new['new_selection']
                    if new['terminal'] is not None or not selection or 'prediction' not in selection:
                        raise ValueError('actual learned selector must receive compatible model forecasts')
                    inputs = transform_inputs(candidate_inputs(causal_history_tensors(list(new_controller.history), now)),
                        input_variant=arm.variant)
                    with torch.inference_mode(): before = old_model(**inputs); after = new_model(**inputs)
                    if set(before) != set(after): raise ValueError('complete expanded forward outputs required')
                    for key in before: torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
                    np.testing.assert_array_equal(selection['prediction'], before['rollout_outcomes'].numpy())
                    all_forward = True
            append(dict(tick=frame, decision=new, old_factory_decision=old, public_input_sha256=public,
                public_input_arrays_unchanged=True, complete_retained_contact_state_sha256=contacts[0],
                complete_retained_contact_state_equal=True, same_assigned_study_public_mission=True,
                original_native_decision_reconstruction_claimed=False))
            if frame == 3:
                boundary = dict(original_factory_terminal=old['terminal'], adapter_factory_terminal=new['terminal'],
                    original_factory_command=old['requested_command'], adapter_factory_command=new['requested_command'],
                    selected_action=None if new['new_selection'] is None else new['new_selection']['action'])
    for model in (old_model, new_model): verify_model(case, model)
    return dict(arm=case.arm_name, assigned_case=case.name, original_public_packet_source=INPUT_CASE,
        controller_class=type(new_controller).__name__, frames=4, boundary=boundary,
        model_state_sha256=arm.model_state_sha256, model_state_unchanged=True,
        all_original_expanded_forward_outputs_exact=all_forward,
        actual_planner_forecast_exact=all_forward, reactive_path_unchanged=arm.name == 'reactive',
        no_packet_after_new_planning_command_consumed=True, command_executed=False,
        study_public_mission_instantiated=True, independent_layout_sensor_data_consumed=False,
        independent_layout_navigation_execution=False, navigation_verified=False)


def main():
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive four-arm startup attempt required')
    if digest(ROOT/VERIFICATION) != VERIFICATION_SHA: raise ValueError('exact earlier multiarm source verification required')
    previous = json.loads((ROOT/VERIFICATION).read_text())
    verify_artifacts(QUEUE, {'launch.json': QUEUE_SHA})
    old_result = read_json(original.INPUT, 'result.json')
    verify_artifacts(original.INPUT, {'result.json': evidence.ORIGINAL_SHA, 'launch.json': evidence.ORIGINAL_LAUNCH})
    ids = old_result['artifact_sha256'] | {'result.json': evidence.ORIGINAL_SHA}
    inherited = merge_sources(previous['source_sha256'], read_json(QUEUE, 'launch.json')['source_sha256'], old_result['source_sha256'])
    sources = discover_sources((SOURCE, PROTOCOL, TEST, VERIFICATION,
        'scripts/independent_round_trip_adapter_multiarm_episode_development.py',
        'scripts/independent_round_trip_adapter_multiarm_audit_development.py'), inherited)
    verify(sources); verify_artifacts(original.INPUT, ids)
    old_launch = read_json(original.INPUT, 'launch.json'); verify_bound_inputs(old_launch['input_admission'], sources)
    resources = hardware()
    if resources['memory_available_bytes'] < 32*1024**3 or resources['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('four-arm startup replay resource envelope unavailable')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, original_artifact_sha256=ids,
        original_result_sha256=evidence.ORIGINAL_SHA, original_raw_startup_case=INPUT_CASE,
        planned_cases=[vars(c) for c in CASES[:4]], output_root=str(OUTPUT), protocol=PROTOCOL, hardware=resources,
        study_public_mission_instantiated=True, independent_layout_sensor_data_consumed=False,
        full_original_training_and_coefficient_verifiers_rerun=False, native_execution=False, model_training=False))
    print('INDEPENDENT_ADAPTER_FACTORY_STARTUP_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True); started = time.perf_counter()
    try:
        reports = []
        for case in CASES[:4]:
            report = run_case(case, old_launch['input_admission']['correction_admission']); reports.append(report)
            print('INDEPENDENT_ADAPTER_FACTORY_STARTUP_CASE', case.arm_name, json.dumps(report['boundary']), flush=True)
        verify(sources); verify_artifacts(original.INPUT, ids); verify_bound_inputs(old_launch['input_admission'], sources)
        outputs = {'launch.json': digest(OUTPUT/'launch.json')}
        for case in CASES[:4]: outputs[case.arm_name+'/'+NAME] = digest(OUTPUT/case.arm_name/NAME)
        verify_artifacts(OUTPUT, outputs)
        write_json(OUTPUT/'result.json', dict(status='INDEPENDENT_ADAPTER_FACTORY_STARTUP_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=outputs, reports=reports, wall_s=time.perf_counter()-started,
            native_execution=False, model_training=False, new_layout_sensor_data_consumed=False,
            new_layout_navigation_execution=False, goal_achieved=False))
        print('INDEPENDENT_ADAPTER_FACTORY_STARTUP_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_INDEPENDENT_ADAPTER_FACTORY_STARTUP_FAILURE', reason=repr(error))); raise


if __name__ == '__main__': main()
