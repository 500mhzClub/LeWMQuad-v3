"""Prospective shared sensor prefix for the fully nonpredictive comparator."""
import argparse
from contextlib import closing
import json
import time

import psutil
import torch

from lewm.measured_plane_comparator_controllers_development import MeasuredPlaneReactiveController
from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.geometry_progress_pilot_development import candidate_commands
from scripts import nominal_measured_plane_native_inputs_development as inputs
from scripts.nominal_measured_plane_maze_development import forbid_model_forward
from scripts.startup_source_inventory_development import discover_sources

previous = inputs.prefix.replay
job, run = previous.job, previous.run
SOURCE = 'scripts/replay_go2_measured_plane_reactive_prefix_v1.py'
TEST = 'lewm/tests/test_measured_plane_reactive_prefix_development.py'
PROTOCOL = 'docs/go2_measured_plane_reactive_prefix_v1_2026-09-11.md'
OUTPUT = run.BASE/'go2_measured_plane_reactive_prefix_v1_attempt_001'
MAX_FRAMES = 123
SHARED_KEYS = ('original_visual_evidence', 'evidence', 'memory_receipt', 'mission_receipt',
    'observed_goal_distance_m', 'auxiliary_floor_partition_receipt')
FALSE_FLAGS = ('learned_model_used', 'candidate_future_outcomes_evaluated',
    'predictive_surface_or_path_gates_applied', 'learned_residual_used',
    'isolated_prediction_ranking_ablation')


def require_reactive(decision):
    if (decision['controller'] != 'measured_plane_reactive_controller_v1'
            or decision['measured_plane_constrained_estimator'] is not True
            or decision['fully_nonpredictive_controller'] is not True
            or decision['reactive_is_whole_method_comparison'] is not True
            or any(decision[k] is not False for k in FALSE_FLAGS)):
        raise ValueError('explicit fully nonpredictive whole-method comparator required')
    selection = decision['new_selection']
    if selection is not None:
        if ('prediction' in selection or 'nominal_path_checks' in selection
                or any(selection[k] is not False for k in ('learned_model_used',
                    'candidate_future_outcomes_evaluated', 'predictive_surface_or_path_gates_applied',
                    'command_integrated_pose_used', 'native_state_used'))
                or selection['current_geometry_checked'] is not True):
            raise ValueError('reactive selection must use current observed geometry without future prediction')
        action = selection['action']
        command = [0., 0., 0.] if action is None else list(candidate_commands(action)[0])
        if selection['requested_command'] != command:
            raise ValueError('reactive selection must name its actual bounded action')
        if decision['terminal'] is None and decision['requested_command'] != command:
            raise ValueError('reactive request must follow its selected action')
    elif decision['terminal'] is None:
        if (decision['mission_receipt'].get('hold_required') is not True
                or decision['requested_command'] != [0., 0., 0.]):
            raise ValueError('only an explicit mission hold may omit reactive selection')
    if decision['terminal'] is not None and decision['requested_command'] != [0., 0., 0.]:
        raise ValueError('terminal reactive controller must hold')


def compare(baseline, candidate, reference, *, frame):
    if type(frame) is not int or not 0 <= frame < MAX_FRAMES:
        raise ValueError('bounded causal frame required')
    if run.canonical(baseline) != run.canonical(reference):
        raise ValueError('complete original measured-plane decision must reproduce')
    if (baseline['controller'] != 'measured_plane_residual_continuation_controller_v1'
            or baseline['measured_plane_constrained_estimator'] is not True):
        raise ValueError('original learned measured-plane baseline required')
    require_reactive(candidate)
    for key in SHARED_KEYS:
        if run.canonical(baseline[key]) != run.canonical(candidate[key]):
            raise ValueError('same complete measured observation, map and mission required: '+key)
    for decision in (baseline, candidate):
        if decision['terminal'] is None:
            if decision['tick'] != frame or decision['failure'] is not None:
                raise ValueError('current nonterminal frame required')
        elif decision['requested_command'] != [0., 0., 0.]:
            raise ValueError('terminal controller must request zero')
    changed = baseline['requested_command'] != candidate['requested_command']
    terminal = baseline['terminal'] is not None or candidate['terminal'] is not None
    stop = changed or terminal or frame == MAX_FRAMES-1
    return dict(frame=frame, complete_original_decision_equal=True,
        complete_shared_observed_receipts_equal=True, requested_command_changed=changed,
        terminal_boundary=terminal, stop=stop,
        stop_reason='FIRST_CHANGED_COMMAND_OR_TERMINAL' if changed or terminal else
            'FIXED_VERIFIED_PREFIX_END' if stop else None,
        reactive_is_whole_method_comparison=True, isolated_prediction_ranking_ablation=False,
        future_constraint_gates_matched=False, changed_command_executed=False,
        following_changed_command_observation_consumed=False, navigation_qualified=False)


def result_report(count, forecasts, check, baseline, candidate):
    return dict(frames=count, learned_forecasts=forecasts, actual_model_forward_calls=[forecasts, 0],
        reactive_model_instantiated=False, reactive_residual_instantiated=False,
        boundary_comparison=check, boundary_baseline=baseline, boundary_candidate=candidate,
        model_state_sha256=job.MODEL_SHA, model_state_unchanged=True,
        fully_nonpredictive_arm=True, predictive_feasibility_gates_matched=False,
        isolated_prediction_ranking_ablation=False, retrospective_navigation_outcomes_inferred=False,
        native_execution=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False)


def check_output(report):
    count = forecasts = 0; stopped = False
    with closing(run.pipeline.read_rows(OUTPUT)) as rows, closing(run.pipeline.read_rows(job.OUTPUT)) as references:
        for row in rows:
            if stopped or row['tick'] != count or count >= MAX_FRAMES:
                raise ValueError('complete ordered output only through first comparison boundary required')
            reference = next(references)
            if (reference['tick'] != count or row['public_inputs_unchanged'] is not True
                    or row['public_packet_sha256'] != reference['public_packet_sha256']
                    or row['reactive_model_forward_blocked'] is not True):
                raise ValueError('same complete public packet and reactive forward guard required')
            check = compare(row['baseline'], row['decision'], reference['decision'], frame=count)
            if check != row['comparison']: raise ValueError('comparison receipt must reconstruct exactly')
            forecasts += int(bool(row['baseline']['new_selection'] and 'prediction' in row['baseline']['new_selection']))
            count += 1; stopped = check['stop']
    if not count or not stopped: raise ValueError('complete explicit causal boundary required')
    if run.canonical(result_report(count, forecasts, check, row['baseline'], row['decision'])) != run.canonical(report):
        raise ValueError('complete report must reconstruct from all output decisions')


def prepared_sources():
    proof = inputs.completed_prefix()
    sources = discover_sources((SOURCE, TEST, PROTOCOL,
        'lewm/tests/test_measured_plane_comparator_controllers_development.py'), proof['source_sha256'])
    run.verify(sources)
    return sources


def replay():
    model = job.native.assigned_model(run.read_json(job.native.OUTPUT, 'launch.json'))
    if model.training or job.state_digest(model.state_dict()) != job.MODEL_SHA:
        raise ValueError('original corrected evaluation model required')
    shared = dict(public_mission=job.public_mission(2), navigation_ticks=4000)
    baseline_controller = MeasuredPlaneResidualController(model, job.ArticulatedCollisionGeometry(job.URDF),
        condition=job.native.CASE[3], variant=job.native.CASE[2], persistent=True, **shared)
    reactive = MeasuredPlaneReactiveController(job.ArticulatedCollisionGeometry(job.URDF), **shared)
    if hasattr(reactive, 'model') or hasattr(reactive, 'residual'):
        raise ValueError('reactive controller must not contain a model or learned residual')
    calls = [0]
    def record(module, args, output): calls[0] += 1
    handle = model.register_forward_hook(record)
    directory = job.native.OUTPUT/job.native.CASE[0]
    reader = run.pipeline.ExtendedBudgetRGBDReplay(directory)
    acquisitions = run.read_json(directory, 'auxiliary_camera_audit.json')
    tape = run.read_json(directory, 'command_tape.json')
    count = forecasts = 0
    try:
        with closing(run.pipeline.read_rows(job.OUTPUT)) as references, run.pipeline.writer(OUTPUT) as append:
            for reference in references:
                frame = count
                if frame >= MAX_FRAMES or reference['tick'] != frame:
                    raise ValueError('bounded complete original prefix required')
                job.command_endpoint(dict(tick=frame, observation_index=frame, pre_sample_index=749+50*frame,
                    decision=reference['original']), tape[frame], frame)
                p, d, f, now = reader.packet(frame)
                image, aux = run.pipeline.rgb_packet(directory, frame, p,
                    run.public_acquisition(acquisitions[frame]), now_ns=now)
                public = p, d, f, image, aux; before = run.fingerprint(public)
                if before != reference['public_packet_sha256']:
                    raise ValueError('same verified public observation required')
                baseline = baseline_controller.observe(p, d, f, now_ns=now, auxiliary_rgb=image, auxiliary_depth=aux)
                if run.fingerprint(public) != before: raise ValueError('baseline changed public inputs')
                with forbid_model_forward(model):
                    candidate = reactive.observe(p, d, f, now_ns=now, auxiliary_rgb=image, auxiliary_depth=aux)
                if run.fingerprint(public) != before: raise ValueError('reactive controller changed public inputs')
                baseline, candidate = [json.loads(run.canonical(d)) for d in (baseline, candidate)]
                check = compare(baseline, candidate, reference['decision'], frame=frame)
                if not check['stop'] and baseline['requested_command'] != tape[frame]['requested_command']:
                    raise ValueError('only a final prospective command may depart from executed history')
                append(dict(tick=frame, baseline=baseline, decision=candidate, comparison=check,
                    public_packet_sha256=before, public_inputs_unchanged=True, reactive_model_forward_blocked=True))
                if (OUTPUT/'context_decisions.jsonl.gz').stat().st_size > job.MAX_OUTPUT_BYTES:
                    raise ValueError('original 2-GiB compressed output allowance exceeded')
                count += 1
                forecasts += int(bool(baseline['new_selection'] and 'prediction' in baseline['new_selection']))
                if calls != [forecasts]: raise ValueError('exact observed baseline model call count required')
                if frame % 25 == 0 or check['stop']:
                    print('MEASURED_PLANE_REACTIVE_FRAME', frame, check['stop_reason'], flush=True)
                if check['stop']: break
    finally:
        handle.remove()
    if not count or not check['stop']: raise ValueError('complete explicit comparison boundary required')
    if job.state_digest(model.state_dict()) != job.MODEL_SHA or any(p.grad is not None for p in model.parameters()):
        raise ValueError('assigned model must remain unchanged without gradients')
    return result_report(count, forecasts, check, baseline, candidate)


def main(source_only=False):
    if (not __debug__ or any(run.os.environ.get(k) != v for k, v in run.ENV.items()) or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU environment required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive new reactive prefix; no retry')
    sources = prepared_sources(); hardware = job.resources()
    if source_only:
        print('MEASURED_PLANE_REACTIVE_PREFLIGHT', len(sources), json.dumps(hardware), flush=True); return
    scheduling = previous.previous.cpu_slot(); previous.previous.admit(sources)
    inputs.completed_prefix()
    run.cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    hardware = job.resources(); run.create_output(OUTPUT); process = psutil.Process()
    run.write_json(OUTPUT/'launch.json', dict(source_sha256=sources, hardware=hardware, scheduling=scheduling,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        maximum_frames=MAX_FRAMES, stop_at_first_changed_command_or_terminal=True,
        model_state_sha256=job.MODEL_SHA, reactive_model_instantiated=False, reactive_residual_instantiated=False,
        fully_nonpredictive_arm=True, isolated_prediction_ranking_ablation=False,
        native_execution=False, automatic_retry=False))
    print('MEASURED_PLANE_REACTIVE_LAUNCHED', run.digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        report = replay(); check_output(report); previous.previous.admit(sources); inputs.completed_prefix()
        run.write_json(OUTPUT/'report.json', report)
        ids = {n: run.digest(OUTPUT/n) for n in ('launch.json', 'context_decisions.jsonl.gz', 'report.json')}
        run.verify_artifacts(OUTPUT, ids); run.verify(sources)
        run.write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_REACTIVE_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, wall_s=time.perf_counter()-start,
            original_raw_inputs_reauthenticated_before_and_after=True,
            native_execution=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False))
        print('MEASURED_PLANE_REACTIVE_COMPLETE', run.digest(OUTPUT/'result.json'), report['boundary_comparison'], flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MEASURED_PLANE_REACTIVE_PREFIX_FAILURE',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    main(parser.parse_args().source_preflight_only)
