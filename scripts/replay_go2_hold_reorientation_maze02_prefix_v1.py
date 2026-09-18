"""Raw full-controller replay through the fixed first hold-reorientation request."""
import argparse
from itertools import islice
import json
import time
import cv2
import torch
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.hold_reorientation_controller_development import HoldReorientationController
from lewm.hold_reorientation_prefix_comparison_development import compare_step
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.all_phase_residual_maze02_study_development import require_case
from scripts import run_go2_all_phase_adapter_maze02_matched_native_v1 as original
from scripts import check_go2_hold_reorientation_saved_prefix_v1 as saved
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

OUTPUT = BASE/'go2_hold_reorientation_maze02_prefix_v1_attempt_001'
SOURCE = 'scripts/replay_go2_hold_reorientation_maze02_prefix_v1.py'
PROTOCOL = 'docs/go2_hold_reorientation_maze02_prefix_v1_2026-09-10.md'
TEST = 'lewm/tests/test_hold_reorientation_prefix_comparison_development.py'
LAUNCH_SHA = '97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a'
SAVED_SHA = 'ea832425482ec3f82e3a91407b5a9aa0029f1bad466213e64b8c83cce18f651c'
MODEL_SHA = '35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a'
CASE = ('all_phase_full_jepa_residual_maze_02', 2, 'full', 'jepa', 'seed_2026091001_full_jepa')
FRAMES = 406


def saved_inputs():
    verify_artifacts(saved.OUTPUT, {'result.json': SAVED_SHA})
    result = read_json(saved.OUTPUT, 'result.json')
    verify(result['source_sha256']); verify_artifacts(saved.OUTPUT, result['artifact_sha256'])
    if (result['status'] != 'HOLD_REORIENTATION_SAVED_PREFIX_V1_COMPLETE'
            or result['frames'] != FRAMES or result['first_changed_command_frame'] != FRAMES-1
            or result['input_result_sha256'] != saved.INPUT_SHA
            or result['candidate_next_observation_consumed'] is not False):
        raise ValueError('exact fixed first saved-selection boundary required')
    boundary = read_json(saved.OUTPUT, 'first_boundary.json')
    comparisons = [json.loads(x) for x in (saved.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    if (len(comparisons) != FRAMES or boundary['frame'] != FRAMES-1
            or any(c['frame'] != i or c['changed'] is not (i == FRAMES-1) for i,c in enumerate(comparisons))
            or saved.identity(boundary['original_selection']) != comparisons[-1]['original_selection_sha256']
            or saved.identity(boundary['candidate_selection']) != comparisons[-1]['candidate_selection_sha256']):
        raise ValueError('complete bound saved prefix and first changed command required')
    return result, boundary, comparisons


def admit_worker(worker_sha, sources):
    name = CASE[0]; terminal = name+'_worker_terminal.json'
    verify_artifacts(original.OUTPUT, {'launch.json': LAUNCH_SHA, terminal: worker_sha})
    launch = read_json(original.OUTPUT, 'launch.json')
    if CASE != original.CASES[0] or any(sources.get(n) != h for n,h in launch['source_sha256'].items()):
        raise ValueError('unchanged exact original first adapter case required')
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
    if resources['memory_available_bytes'] < 32*1024**3 or resources['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('32GiB RAM and 40+1GiB artifact envelope required')


def replay():
    _, boundary, expected = saved_inputs()
    old_launch = read_json(original.OUTPUT, 'launch.json')
    models = [original.assigned_model(old_launch, CASE) for _ in range(2)]
    if any(state_digest(m.state_dict()) != MODEL_SHA for m in models):
        raise ValueError('fresh identical assigned expanded JEPA models required')
    options = dict(public_mission=public_mission(CASE[1]), navigation_ticks=NAVIGATION_TICKS,
        condition=CASE[3], variant=CASE[2], persistent=True)
    geometry = ArticulatedCollisionGeometry(URDF)
    controllers = (ResidualAnchoredContinuationController(models[0], geometry, **options),
        HoldReorientationController(models[1], geometry, **options))
    directory = original.OUTPUT/CASE[0]; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    frames = forecasts = 0; last = None
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
            public_sha = fingerprint((p,d,fast,auxiliary,image,now))
            old, new = [c.observe(p,d,fast,now_ns=now,auxiliary_depth=auxiliary,auxiliary_rgb=image)
                for c in controllers]
            if json.loads(json.dumps(old)) != row['decision']:
                raise ValueError('complete original raw controller decision does not reproduce: '+str(frame))
            wanted = boundary['candidate_selection'] if frame == FRAMES-1 else row['decision']['new_selection']
            check = compare_step(old, new, tape[frame]['requested_command'], frame=frame, expected_selection=wanted)
            if (saved.identity(new['new_selection']) != comparison['candidate_selection_sha256']
                    or check['requested_command_changed'] is not (frame == FRAMES-1)):
                raise ValueError('exact first saved intervention boundary required')
            if public_sha != fingerprint((p,d,fast,auxiliary,image,now)):
                raise ValueError('raw public input arrays changed')
            contacts = [fingerprint(state_tree(c.memory)) for c in controllers]
            if (contacts[0] != contacts[1] or controllers[0].mapper.floor != controllers[1].mapper.floor
                    or controllers[0].mapper.occupied != controllers[1].mapper.occupied):
                raise ValueError('complete observed contact memory and accumulated planning cells must remain exact')
            append(dict(tick=frame, decision=new, original_requested_command=tape[frame]['requested_command'],
                original_complete_decision_reconstructed=True, comparison=check, public_input_sha256=public_sha,
                public_input_arrays_unchanged=True, complete_retained_contact_state_sha256=contacts[0],
                complete_retained_contact_state_equal=True))
            frames += 1; forecasts += int(check['raw_model_forecasts_compared']); last = new
            if frame % 50 == 0 or check['requested_command_changed']:
                print('HOLD_REORIENTATION_RAW_FRAME', frame, 'changed', check['requested_command_changed'], flush=True)
    if frames != FRAMES or last['requested_command'] != boundary['candidate_requested_command']:
        raise ValueError('full fixed raw prefix must end at the declared changed request')
    if any(state_digest(m.state_dict()) != MODEL_SHA or any(p.grad is not None for p in m.parameters()) for m in models):
        raise ValueError('assigned model weights or gradients changed')
    return dict(frames=frames, raw_model_forecast_comparisons=forecasts,
        first_changed_command_frame=FRAMES-1, original_requested_command=boundary['original_requested_command'],
        candidate_requested_command=last['requested_command'], original_complete_decisions_reconstructed=True,
        candidate_matches_saved_selection_boundary=True, observed_map_contact_and_executed_residual_state_exact=True,
        model_state_sha256=MODEL_SHA, model_state_unchanged=True, public_input_arrays_unchanged=True,
        no_observation_after_changed_request_consumed=True, native_execution=False,
        unexecuted_outcomes_inferred=False, navigation_verified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-preflight-only', action='store_true')
    parser.add_argument('--original-worker-terminal-sha256')
    args = parser.parse_args()
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive fixed raw prefix attempt required')
    result, _, _ = saved_inputs()
    sources = discover_sources((SOURCE, PROTOCOL, TEST,
        'lewm/tests/test_hold_reorientation_raw_runner_development.py'), result['source_sha256']); verify(sources)
    resources = hardware(); resources_for(resources)
    if args.source_preflight_only:
        print('HOLD_REORIENTATION_RAW_SOURCE_PREFLIGHT_PASS', len(sources), flush=True); return
    if not args.original_worker_terminal_sha256: raise ValueError('explicit completed original worker hash required')
    admission = admit_worker(args.original_worker_terminal_sha256, sources)
    resources = hardware(); resources_for(resources)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_admission=admission,
        saved_selection_result_sha256=SAVED_SHA, original_launch_sha256=LAUNCH_SHA,
        model_state_sha256=MODEL_SHA, output_root=str(OUTPUT), protocol=PROTOCOL, hardware=resources,
        source_mutation=False, native_execution=False, model_training=False))
    print('HOLD_REORIENTATION_RAW_PREFIX_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = replay()
        verify(sources)
        if admit_worker(args.original_worker_terminal_sha256, sources) != admission:
            raise ValueError('complete original admitted evidence changed')
        saved_inputs()
        ids = {n: digest(OUTPUT/n) for n in ('launch.json', NAME)}; verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='HOLD_REORIENTATION_MAZE02_RAW_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report,
            original_worker_terminal_sha256=args.original_worker_terminal_sha256,
            wall_s=time.perf_counter()-start, native_execution=False, goal_achieved=False))
        print('HOLD_REORIENTATION_RAW_PREFIX_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_HOLD_REORIENTATION_RAW_PREFIX_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
