"""Exact four-observation controller replay for all six completed failed workers."""
from itertools import islice
import json
import time
import cv2
import numpy as np
import torch

from lewm.all_phase_planner_model_adapter_development import AllPhasePlannerModel
from lewm.training_translation_bias_development import TrainingTranslationBiasModel
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.observation_horizon_predictive_selection_development import candidate_inputs
from lewm.observation_horizon_input_ablation_development import transform_inputs
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.all_phase_residual_maze02_study_development import CASES, WORKER_STATUS, require_case
from scripts.all_phase_translation_bias_model_admission_development import load_assigned
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.maze_decision_stream_development import read_rows, writer, NAME
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.replay_go2_residual_current_observation_planning_prefix_v1 import state_tree
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

INPUT = BASE/'go2_all_phase_residual_maze02_matched_native_v1_attempt_001'
OUTPUT = BASE/'go2_all_phase_planner_adapter_startup_v1_attempt_001'
SOURCE = 'scripts/replay_go2_all_phase_planner_adapter_startup_v1.py'
PROTOCOL = 'docs/go2_all_phase_planner_adapter_startup_v1_2026-09-10.md'
TEST = 'lewm/tests/test_all_phase_planner_model_adapter_development.py'
QUEUE = BASE/'go2_all_phase_residual_maze02_native_wait_v1_attempt_001'
QUEUE_SHA = '6e96b6bc78f08f8dd7b5af3ad25e48b78ca915a5c3b5ede37efe0bc8a7be9b5e'
EXPECTED_FAILURE = 'training-only translation wrapper in evaluation mode required'


def admit_workers():
    launch = read_json(INPUT, 'launch.json'); ids = {'launch.json': digest(INPUT/'launch.json')}
    if (launch['planned_cases'] != [list(case) for case in CASES]
            or launch['implementation_class'] != 'ResidualAnchoredContinuationController'
            or launch['input_admission']['native_result_sha256'] != '330ae2381254538f43f5bf1d5374ba20ebea657b7d2749477468276e1d5ee723'):
        raise ValueError('exact original six-model launch and admitted predecessor required')
    records = []
    for case in CASES:
        name = case[0]; terminal_name = name+'_worker_terminal.json'
        record = read_json(INPUT, terminal_name); report = read_json(INPUT, name+'_audit.json')
        require_case(case, record, report)
        if (record['status'] != WORKER_STATUS or record['collection']['decisions'] != 14
                or record['collection']['completed_ticks'] != 13 or record['verified_round_trip'] is not False):
            raise ValueError('complete original four-observation-plus-drain failure required')
        ids.update(record['artifact_sha256']); ids[terminal_name] = digest(INPUT/terminal_name)
        records.append(record)
    verify_artifacts(INPUT, ids)
    return launch, ids, records


def replay_case(case, launch):
    name, index, variant, condition, model_name = case
    source, c, v = load_assigned(launch['input_admission']['correction_admission'], model_name)
    adapted = AllPhasePlannerModel(source); assigned = launch['assigned_model_states'][model_name]
    if (c, v) != (condition, variant) or any(state_digest(model.state_dict()) != assigned for model in (source, adapted)):
        raise ValueError('same original assigned expanded model state required')
    if source.training or isinstance(source, TrainingTranslationBiasModel) or not isinstance(adapted, TrainingTranslationBiasModel):
        raise ValueError('original exact interface mismatch and corrected interface required')
    options = dict(public_mission=public_mission(index), navigation_ticks=NAVIGATION_TICKS,
        condition=condition, variant=variant, persistent=True)
    geometry = ArticulatedCollisionGeometry(URDF)
    controllers = [ResidualAnchoredContinuationController(model, geometry, **options) for model in (source, adapted)]
    directory = INPUT/name; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    dest = OUTPUT/name; dest.mkdir(); reports = []; final = None
    with writer(dest) as append:
        for row in islice(read_rows(directory), 4):
            frame = row['tick']
            if not tape[frame]['completed'] or controllers[0].residual.pending != controllers[1].residual.pending:
                raise ValueError('actual completed original command and identical prior residual forecast required')
            p, d, fast, now = reader.packet(frame)
            image, auxiliary = packet(directory, frame, p, public_acquisition(acquisitions[frame]), now_ns=now)
            before = fingerprint((p, d, fast, auxiliary, image, now))
            old, new = [controller.observe(p, d, fast, now_ns=now, auxiliary_depth=auxiliary, auxiliary_rgb=image)
                for controller in controllers]
            if json.loads(json.dumps(old)) != row['decision']:
                raise ValueError('complete original raw controller decision does not reproduce: '+str(frame))
            if old['requested_command'] != tape[frame]['requested_command']:
                raise ValueError('recorded command differs from reproduced original request')
            if before != fingerprint((p, d, fast, auxiliary, image, now)):
                raise ValueError('public input arrays changed')
            for key in ('evidence', 'original_visual_evidence', 'memory_receipt', 'mission_receipt',
                    'observed_goal_distance_m', 'floor_partition_receipt', 'auxiliary_floor_partition_receipt'):
                if old[key] != new[key]: raise ValueError('observed state changed outside model interface: '+key)
            contacts = [fingerprint(state_tree(controller.memory)) for controller in controllers]
            if contacts[0] != contacts[1] or controllers[0].mapper.floor != controllers[1].mapper.floor or controllers[0].mapper.occupied != controllers[1].mapper.occupied:
                raise ValueError('same sensor, mapping and retained contact state required')
            if frame < 3 and old != new:
                raise ValueError('all three complete warmup decisions must match exactly')
            if frame == 3:
                if old['failure'] != EXPECTED_FAILURE or old['terminal'] != 'SENSOR_OR_MODEL_FAILURE':
                    raise ValueError('exact original first planning interface failure required')
                if new['terminal'] is not None or new['new_selection'] is None or 'prediction' not in new['new_selection']:
                    raise ValueError('adapter must actually reach the unchanged planner with forecasts')
                history = causal_history_tensors(list(controllers[1].history), now)
                inputs = transform_inputs(candidate_inputs(history), input_variant=variant)
                with torch.inference_mode(): original_output = source(**inputs); adapted_output = adapted(**inputs)
                if set(original_output) != set(adapted_output): raise ValueError('complete forecast output keys required')
                for key in original_output:
                    torch.testing.assert_close(original_output[key], adapted_output[key], rtol=0, atol=0)
                head = 'direct_outcomes' if condition == 'direct' else 'rollout_outcomes'
                np.testing.assert_array_equal(new['new_selection']['prediction'], original_output[head].numpy())
                final = dict(frame=frame, old_failure=old['failure'], original_terminal=old['terminal'],
                    candidate_terminal=new['terminal'], original_requested_command=old['requested_command'],
                    candidate_requested_command=new['requested_command'], selected_action=new['new_selection']['action'],
                    all_original_expanded_forward_tensors_exact=True, planner_forecast_matches_original_expanded_forward=True,
                    recorded_predecessor_forecasts_present=False)
            receipt = dict(tick=frame, decision=new, original_complete_decision_reconstructed=True,
                public_input_sha256=before, public_input_arrays_unchanged=True,
                original_requested_command=old['requested_command'], original_terminal=old['terminal'],
                complete_retained_contact_state_sha256=contacts[0], complete_retained_contact_state_equal=True)
            append(receipt); reports.append(receipt)
    if len(reports) != 4 or final is None: raise ValueError('complete four-observation first-divergence prefix required')
    for model in (source, adapted):
        if state_digest(model.state_dict()) != assigned or any(p.grad is not None for p in model.parameters()):
            raise ValueError('model state or gradients changed')
    return dict(case=name, model_name=model_name, model_state_sha256=assigned, frames=4,
        first_terminal_difference=3, boundary=final, model_state_unchanged=True,
        complete_original_warmup_decisions_exact=True, complete_original_decisions_reconstructed=True,
        no_recorded_observation_after_divergence_consumed=True, observed_and_contact_state_unchanged=True,
        controller_class_and_all_planning_constraints_unchanged=True,
        command_executed=False, navigation_verified=False)


def main():
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive adapter startup replay required')
    verify_artifacts(QUEUE, {'launch.json': QUEUE_SHA}); queued = read_json(QUEUE, 'launch.json')
    old, ids, workers = admit_workers()
    if any(queued['source_sha256'].get(name) != sha for name, sha in old['source_sha256'].items()):
        raise ValueError('unchanged original queued source bindings required')
    sources = discover_sources((SOURCE, PROTOCOL, TEST), queued['source_sha256']); verify(sources)
    resources = hardware()
    if resources['memory_available_bytes'] < 32*1024**3 or resources['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('bounded replay resources unavailable')
    create_output(OUTPUT)
    launch = dict(source_sha256=sources, original_artifact_sha256=ids,
        original_workers_complete_and_raw_audited=True, original_top_level_completion_required_by_this_replay=False,
        queued_launch_sha256=QUEUE_SHA, output_root=str(OUTPUT), protocol=PROTOCOL, hardware=resources,
        planned_cases=[list(case) for case in CASES], model_training=False, native_execution=False,
        source_mutation=False, full_original_input_verifiers_rerun=False)
    write_json(OUTPUT/'launch.json', launch); start = time.perf_counter()
    print('ALL_PHASE_ADAPTER_STARTUP_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    try:
        reports = []
        for case in CASES:
            report = replay_case(case, old); reports.append(report)
            print('ALL_PHASE_ADAPTER_STARTUP_CASE', json.dumps(report['boundary'] | {'case': case[0]}), flush=True)
        verify(sources); verify_artifacts(INPUT, ids)
        if admit_workers() != (old, ids, workers): raise ValueError('original completed worker evidence changed')
        outputs = {'launch.json': digest(OUTPUT/'launch.json')}
        for case in CASES: outputs[case[0]+'/'+NAME] = digest(OUTPUT/case[0]/NAME)
        verify_artifacts(OUTPUT, outputs)
        write_json(OUTPUT/'result.json', dict(status='ALL_PHASE_PLANNER_ADAPTER_STARTUP_COMPLETE',
            source_sha256=sources, artifact_sha256=outputs, reports=reports, wall_s=time.perf_counter()-start,
            models=6, original_frames_per_model=4, first_terminal_difference=3,
            all_selected_forecasts_and_complete_original_outputs_exact=True,
            raw_public_startup_used=True, model_training=False, native_execution=False,
            unexecuted_outcomes_inferred=False, navigation_qualified=False, goal_achieved=False))
        print('ALL_PHASE_ADAPTER_STARTUP_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_ALL_PHASE_ADAPTER_STARTUP_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
