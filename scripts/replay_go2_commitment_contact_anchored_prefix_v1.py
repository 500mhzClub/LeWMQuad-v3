"""Fresh original/candidate supervised controllers through one changed request."""
import argparse
from copy import deepcopy
from itertools import islice
import json
import time
import cv2
import torch
from lewm.commitment_contact_anchored_controller_development import (
    CommitmentContactAnchoredController, ordinary_commitment_contact)
from lewm.commitment_contact_anchored_prefix_development import PrefixComparison
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.all_phase_residual_maze02_study_development import require_case
from scripts import run_go2_all_phase_adapter_maze02_matched_native_v1 as original
from scripts import check_go2_commitment_contact_anchored_saved_prefix_v1 as saved
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

OUTPUT = BASE/'go2_commitment_contact_anchored_raw_prefix_v1_attempt_001'
SOURCE = 'scripts/replay_go2_commitment_contact_anchored_prefix_v1.py'
PROTOCOL = 'docs/go2_commitment_contact_anchored_raw_prefix_v1_2026-09-10.md'
TEST = 'lewm/tests/test_commitment_contact_anchored_raw_runner_development.py'
LAUNCH_SHA = saved.LAUNCH_SHA
SAVED_SHA = 'f20100955ca2e4bb39c91a9375cb270ff48731d5e721aacf8d5b314821381808'
MODEL_SHA = saved.MODEL_SHA
CASE = (saved.CASE, 2, 'full', 'supervised_rollout', saved.MODEL)
FRAMES = 4
SEEDS = (SOURCE, PROTOCOL, TEST,
    'docs/go2_commitment_contact_anchored_saved_prefix_result_2026-09-10.md',
    'docs/go2_commitment_contact_anchored_saved_prefix_verification_2026-09-10.json')


def saved_inputs():
    verify_artifacts(saved.OUTPUT, {'result.json': SAVED_SHA})
    result = read_json(saved.OUTPUT, 'result.json')
    verify(result['source_sha256']); verify_artifacts(saved.OUTPUT, result['artifact_sha256'])
    boundary = read_json(saved.OUTPUT, 'first_boundary.json')
    comparisons = [json.loads(line) for line in (saved.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    if (result['status'] != 'COMMITMENT_CONTACT_ANCHORED_SAVED_PREFIX_V1_COMPLETE'
            or result['consumed_frames'] != FRAMES or result['first_changed_command_frame'] != FRAMES-1
            or result['candidate_next_observation_consumed'] is not False
            or boundary['frame'] != FRAMES-1 or boundary['consumed_frames'] != FRAMES
            or len(comparisons) != FRAMES
            or any(c['frame'] != i or c['changed'] is not (i == FRAMES-1) for i,c in enumerate(comparisons))
            or boundary['original_requested_command'] != [0., 0., .45]
            or boundary['candidate_requested_command'] != [.2, 0., 0.]
            or ordinary_commitment_contact(boundary['original_selection']) != boundary['expected_candidate_selection']
            or saved.identity(boundary['original_selection']) != comparisons[-1]['original_selection_sha256']
            or saved.identity(boundary['expected_candidate_selection']) != comparisons[-1]['candidate_selection_sha256']):
        raise ValueError('complete fixed saved four-observation boundary required')
    return result, boundary, comparisons


def admit_worker(worker_sha, sources):
    name = CASE[0]; terminal = name+'_worker_terminal.json'
    verify_artifacts(original.OUTPUT, {'launch.json': LAUNCH_SHA, terminal: worker_sha})
    launch = read_json(original.OUTPUT, 'launch.json')
    if CASE != original.CASES[1] or any(sources.get(n) != h for n,h in launch['source_sha256'].items()):
        raise ValueError('unchanged exact original second adapter case required')
    record = read_json(original.OUTPUT, terminal)
    ids = record['artifact_sha256'] | {'launch.json': LAUNCH_SHA, terminal: worker_sha,
        name+'_worker.log': record['worker_log_sha256']}
    verify_artifacts(original.OUTPUT, ids)
    report = read_json(original.OUTPUT, name+'_audit.json'); require_case(CASE, record, report)
    expected = {name+'/'+n for n in original.artifacts(CASE[1], record['collection'])}
    expected.update((name+'_audit.json', name+'_startup_comparison.json', name+'_readout.json'))
    if (not expected.issubset(ids)
            or record['collection'] != read_json(original.OUTPUT/name, 'result.json')
            or record['model_state_sha256'] != MODEL_SHA
            or launch['assigned_model_states'][CASE[4]] != MODEL_SHA
            or original.compare_adapter_startup(CASE, original.OUTPUT/name) != record['startup_comparison']
            or read_json(original.OUTPUT, name+'_startup_comparison.json') != record['startup_comparison']
            or read_json(original.OUTPUT, name+'_readout.json') != record['readout']):
        raise ValueError('complete original collection, model, raw audit and actual adapter boundary required')
    original.verify_inputs(launch, full=True)
    return dict(original_worker_terminal_sha256=worker_sha, original_artifact_sha256=ids,
        original_case=CASE[0], original_worker_complete_and_raw_audited=True,
        original_full_input_verifier_reexecuted=True, complete_six_case_parent_required=False)


def resources_for(resources):
    if resources['memory_available_bytes'] < 48*1024**3 or resources['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('48GiB RAM and 40+1GiB artifact envelope required')


def check_pending(controller, decision, *, frame, now_ns):
    selection = decision['new_selection']; residual = controller.residual
    expected = None
    if decision['terminal'] is None and selection and 'prediction' in selection:
        matches = [i for i, action in enumerate(ACTIONS)
            if decision['requested_command'] == candidate_commands(action)[0]]
        if len(matches) != 1 or residual.pose is None:
            raise ValueError('one selected original command and current observed pose required')
        i = matches[0]
        expected = dict(tick=frame, measured_ns=now_ns, action=ACTIONS[i],
            requested_command=list(decision['requested_command']),
            predicted_body_xy_m=deepcopy(selection['prediction'][i][0][:2]), **deepcopy(residual.pose))
    if residual.pending != expected:
        raise ValueError('pending forecast must describe exactly the selected command and current observed pose')


def observed_state(controller):
    residual = {k:v for k,v in vars(controller.residual).items() if k != 'pending'}
    return state_tree(dict(memory=controller.memory, floor=controller.mapper.floor,
        occupied=controller.mapper.occupied, residual=residual, history=controller.history))


def replay():
    _, boundary, expected = saved_inputs()
    old_launch = read_json(original.OUTPUT, 'launch.json')
    models = [original.assigned_model(old_launch, CASE) for _ in range(2)]
    if (models[0] is models[1] or any(state_digest(m.state_dict()) != MODEL_SHA for m in models)
            or any(a.data_ptr() == b.data_ptr() for a,b in zip(models[0].parameters(), models[1].parameters()))):
        raise ValueError('fresh independently stored identical assigned supervised models required')
    options = dict(public_mission=public_mission(CASE[1]), navigation_ticks=NAVIGATION_TICKS,
        condition=CASE[3], variant=CASE[2], persistent=True)
    geometry = ArticulatedCollisionGeometry(URDF)
    controllers = (ResidualAnchoredContinuationController(models[0], geometry, **options),
        CommitmentContactAnchoredController(models[1], geometry, **options))
    directory = original.OUTPUT/CASE[0]; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    frames = forecasts = 0; last = None; comparator = PrefixComparison()
    with writer(OUTPUT) as append:
        for row in islice(read_rows(directory), FRAMES):
            frame = row['tick']; comparison = expected[frame]
            if (frame != frames or saved.identity(row) != comparison['original_row_sha256']
                    or frame >= len(tape) or tape[frame]['completed'] is not True
                    or tape[frame]['tick'] != frame or tape[frame]['pre_sample_index'] != 749+50*frame
                    or tape[frame]['post_sample_index'] != 799+50*frame):
                raise ValueError('exact original saved row and completed command endpoints required')
            if controllers[0].residual.pending != controllers[1].residual.pending:
                raise ValueError('identical full preceding pending forecasts required')
            p, d, fast, now = reader.packet(frame)
            image, auxiliary = packet(directory, frame, p, public_acquisition(acquisitions[frame]), now_ns=now)
            public_sha = fingerprint((p,d,fast,auxiliary,image,now)); decisions = []
            for controller in controllers:
                decision = controller.observe(p,d,fast,now_ns=now,auxiliary_depth=auxiliary,auxiliary_rgb=image)
                if public_sha != fingerprint((p,d,fast,auxiliary,image,now)):
                    raise ValueError('each controller must preserve all public input arrays')
                decisions.append(json.loads(json.dumps(decision, allow_nan=False)))
                check_pending(controller, decisions[-1], frame=frame, now_ns=now)
            old, new = decisions
            if old != row['decision']:
                raise ValueError('complete original raw controller decision does not reproduce: '+str(frame))
            check = comparator.compare(old, new, tape[frame]['requested_command'], frame=frame)
            if (saved.identity(new['new_selection']) != comparison['candidate_selection_sha256']
                    or check['requested_command_changed'] is not (frame == FRAMES-1)
                    or check['stop'] is not (frame == FRAMES-1)):
                raise ValueError('exact first saved intervention boundary required')
            states = [fingerprint(observed_state(c)) for c in controllers]
            if states[0] != states[1]:
                raise ValueError('complete observed memory map residual and model history must remain exact')
            append(dict(tick=frame, decision=new, original_requested_command=tape[frame]['requested_command'],
                original_complete_decision_reconstructed=True, comparison=check, public_input_sha256=public_sha,
                public_input_arrays_unchanged=True, complete_retained_observed_state_sha256=states[0],
                complete_retained_observed_state_equal=True, selected_pending_forecasts_checked=True))
            frames += 1; forecasts += int(check['raw_model_forecasts_compared']); last = new
            print('COMMITMENT_CONTACT_ANCHORED_RAW_FRAME', frame, 'changed', check['requested_command_changed'], flush=True)
    if (frames != FRAMES or forecasts != 1 or comparator.first_command_difference != FRAMES-1
            or last['requested_command'] != boundary['candidate_requested_command']):
        raise ValueError('full four-observation raw prefix must end at the declared changed request')
    if any(state_digest(m.state_dict()) != MODEL_SHA or any(p.grad is not None for p in m.parameters()) for m in models):
        raise ValueError('assigned model weights or gradients changed')
    return dict(frames=frames, raw_model_forecast_comparisons=forecasts, first_changed_command_frame=FRAMES-1,
        original_requested_command=boundary['original_requested_command'], candidate_requested_command=last['requested_command'],
        original_complete_decisions_reconstructed=True, candidate_matches_saved_selection_boundary=True,
        complete_retained_observed_state_exact=True, selected_pending_forecasts_checked=True,
        model_state_sha256=MODEL_SHA, model_state_unchanged=True, public_input_arrays_unchanged=True,
        no_observation_after_changed_request_consumed=True, native_execution=False,
        unexecuted_outcomes_inferred=False, navigation_verified=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    parser.add_argument('--original-worker-terminal-sha256'); args = parser.parse_args()
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive fixed raw prefix attempt required')
    result, _, _ = saved_inputs()
    sources = discover_sources(SEEDS, result['source_sha256'])
    verify(sources); resources = hardware(); resources_for(resources)
    if args.source_preflight_only:
        print('COMMITMENT_CONTACT_ANCHORED_RAW_SOURCE_PREFLIGHT_PASS', len(sources), flush=True); return
    if not args.original_worker_terminal_sha256: raise ValueError('explicit completed original worker hash required')
    print('COMMITMENT_CONTACT_ANCHORED_RAW_INPUT_ADMISSION_STARTED', flush=True)
    admission = admit_worker(args.original_worker_terminal_sha256, sources)
    resources = hardware(); resources_for(resources); verify(sources); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_admission=admission,
        saved_selection_result_sha256=SAVED_SHA, original_launch_sha256=LAUNCH_SHA,
        model_state_sha256=MODEL_SHA, output_root=str(OUTPUT), protocol=PROTOCOL, hardware=resources,
        source_mutation=False, native_execution=False, model_training=False))
    print('COMMITMENT_CONTACT_ANCHORED_RAW_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = replay(); verify(sources)
        if admit_worker(args.original_worker_terminal_sha256, sources) != admission:
            raise ValueError('complete original admitted evidence changed')
        saved_inputs()
        ids = {name:digest(OUTPUT/name) for name in ('launch.json', NAME)}; verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='COMMITMENT_CONTACT_ANCHORED_RAW_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report,
            original_worker_terminal_sha256=args.original_worker_terminal_sha256,
            wall_s=time.perf_counter()-start, native_execution=False, goal_achieved=False))
        print('COMMITMENT_CONTACT_ANCHORED_RAW_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_COMMITMENT_CONTACT_ANCHORED_RAW_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
