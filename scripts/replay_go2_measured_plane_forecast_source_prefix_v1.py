"""Learned versus nominal forecasts on the exact shared physical prefix."""
import argparse
from contextlib import closing
import json
import time

import psutil
import torch

from lewm.measured_plane_comparator_controllers_development import MeasuredPlaneForecastSourceController
from lewm.geometry_progress_pilot_development import candidate_commands
from scripts import replay_go2_measured_plane_single_pass_prefix_v1 as previous
from scripts.startup_source_inventory_development import discover_sources

job, run = previous.job, previous.run
SOURCE = 'scripts/replay_go2_measured_plane_forecast_source_prefix_v1.py'
TEST = 'lewm/tests/test_measured_plane_forecast_source_prefix_development.py'
PROTOCOL = 'docs/go2_measured_plane_forecast_source_prefix_v1_2026-09-11.md'
OUTPUT = run.BASE/'go2_measured_plane_forecast_source_prefix_v1_attempt_001'
MODES = ('frozen_world_model', 'nominal_requested_twist')
MAX_FRAMES = 123


def learned_reference(decision):
    if (decision['controller'] != 'measured_plane_forecast_source_controller_v1'
            or decision['assigned_forecast_source'] != MODES[0]
            or decision['shared_observed_residual_correction_retained'] is not True
            or decision['fully_nonpredictive_controller'] is not False):
        raise ValueError('explicit learned forecast-source baseline required')
    result = decision.copy()
    for key in ('assigned_forecast_source', 'shared_observed_residual_correction_retained',
                'fully_nonpredictive_controller'):
        result.pop(key)
    result['controller'] = 'measured_plane_residual_continuation_controller_v1'
    if result['new_selection'] is not None:
        selection = result['new_selection'].copy()
        if 'prediction' in selection:
            provenance = selection.pop('forecast_provenance')
            if (provenance['forecast_source'] != MODES[0]
                    or provenance['frozen_model_forward_called'] is not True):
                raise ValueError('actual learned model provenance required')
        result['new_selection'] = selection
    return result


def compare(baseline, candidate, reference, *, frame):
    if type(frame) is not int or not 0 <= frame < MAX_FRAMES:
        raise ValueError('bounded causal comparison frame required')
    if run.canonical(learned_reference(baseline)) != run.canonical(reference):
        raise ValueError('complete original measured-plane decision must reproduce')
    if (candidate['controller'] != 'measured_plane_forecast_source_controller_v1'
            or candidate['assigned_forecast_source'] != MODES[1]
            or candidate['measured_plane_constrained_estimator'] is not True
            or candidate['shared_observed_residual_correction_retained'] is not True
            or candidate['fully_nonpredictive_controller'] is not False):
        raise ValueError('same controller with explicitly nominal predictive source required')
    for key in ('original_visual_evidence', 'evidence', 'memory_receipt', 'mission_receipt'):
        if run.canonical(candidate[key]) != run.canonical(baseline[key]):
            raise ValueError('same complete observation, map and mission evidence required: '+key)
    for decision, mode in ((baseline, MODES[0]), (candidate, MODES[1])):
        if decision['terminal'] is None:
            if decision['tick'] != frame or decision['failure'] is not None:
                raise ValueError('current nonterminal controller frame required')
        elif decision['requested_command'] != [0., 0., 0.]:
            raise ValueError('terminal controller must request zero command')
        selection = decision['new_selection']
        if frame >= 3 and decision['terminal'] is None and (not selection or 'prediction' not in selection):
            if (decision['mission_receipt'].get('hold_required') is not True
                    or decision['requested_command'] != [0., 0., 0.]):
                raise ValueError('only an explicit mission hold may omit a forecast')
        if selection and 'prediction' in selection:
            source = selection['forecast_provenance']
            learned = mode == MODES[0]
            if (source['forecast_source'] != mode
                    or source['frozen_model_forward_called'] is not learned
                    or source['learned_forecasts_used'] is not learned
                    or source['nominal_requested_twist_forecasts_used'] is not (not learned)
                    or selection['model_prediction_corrected'] is not learned
                    or selection['translation_bias_training_only'] is not learned):
                raise ValueError('actual forecast-source provenance must survive selection')
            action = selection['action']
            requested = [0., 0., 0.] if action is None else list(candidate_commands(action)[0])
            if decision['terminal'] is None and decision['requested_command'] != requested:
                raise ValueError('request must follow actual selected action')
    changed = baseline['requested_command'] != candidate['requested_command']
    terminal = baseline['terminal'] != candidate['terminal'] or candidate['terminal'] is not None
    return dict(frame=frame, complete_original_decision_equal=True,
        complete_observation_map_mission_receipts_equal=True, requested_command_changed=changed,
        terminal_boundary=terminal, stop=changed or terminal or frame == MAX_FRAMES-1,
        stop_reason='FIRST_CHANGED_COMMAND_OR_TERMINAL' if changed or terminal else
            'FIXED_VERIFIED_PREFIX_END' if frame == MAX_FRAMES-1 else None,
        both_arms_predictive=True, isolated_planning_on_off_comparison=False,
        residual_values_required_equal=False, changed_command_executed=False,
        following_changed_command_observation_consumed=False, navigation_qualified=False)


def prepared_sources():
    proof = previous.inputs.completed_prefix()
    sources = discover_sources((SOURCE, TEST, PROTOCOL,
        'lewm/tests/test_measured_plane_comparator_controllers_development.py',
        str(previous.inputs.prefix.completed.OUTPUT.relative_to(run.ROOT))), proof['source_sha256'])
    run.verify(sources)
    return sources


def result_report(count, forecasts, check, baseline, candidate):
    return dict(frames=count, learned_forecasts=forecasts, actual_model_forward_calls=[forecasts, 0],
        boundary_comparison=check, boundary_baseline=baseline, boundary_candidate=candidate,
        model_state_sha256=job.MODEL_SHA, model_states_unchanged=True,
        nominal_source_is_nonlearned_predictive=True, fully_nonpredictive_arm=False,
        retrospective_navigation_outcomes_inferred=False, native_execution=False,
        navigation_qualified=False, real_time_qualified=False, goal_achieved=False)


def check_output(report):
    count = forecasts = 0
    stopped = False
    with closing(run.pipeline.read_rows(OUTPUT)) as rows, closing(run.pipeline.read_rows(job.OUTPUT)) as references:
        for row in rows:
            if stopped or row['tick'] != count or count >= MAX_FRAMES:
                raise ValueError('complete ordered output only through the first comparison boundary required')
            reference = next(references)
            if (reference['tick'] != count or row['public_inputs_unchanged'] is not True
                    or row['public_packet_sha256'] != reference['public_packet_sha256']):
                raise ValueError('same complete verified input packet required in every output')
            check = compare(row['baseline'], row['decision'], reference['decision'], frame=count)
            if check != row['comparison']: raise ValueError('comparison receipt must reconstruct exactly')
            count += 1
            selection = row['baseline']['new_selection']
            forecasts += int(bool(selection and 'prediction' in selection))
            stopped = check['stop']
    if not count or not stopped: raise ValueError('complete explicit causal boundary required')
    expected = result_report(count, forecasts, check, row['baseline'], row['decision'])
    if run.canonical(expected) != run.canonical(report):
        raise ValueError('complete report must reconstruct from all original output decisions')


def replay():
    models = [job.native.assigned_model(run.read_json(job.native.OUTPUT, 'launch.json')) for _ in MODES]
    storage = [{v.untyped_storage().data_ptr() for v in m.state_dict().values() if v.numel()} for m in models]
    if (models[0] is models[1] or not storage[0].isdisjoint(storage[1])
            or any(m.training or job.state_digest(m.state_dict()) != job.MODEL_SHA for m in models)):
        raise ValueError('two independent original corrected evaluation models required')
    calls = [0, 0]
    def hook(index):
        def record(module, args, output): calls[index] += 1
        return record
    handles = [model.register_forward_hook(hook(i)) for i, model in enumerate(models)]
    options = dict(public_mission=job.public_mission(2), navigation_ticks=4000,
        condition=job.native.CASE[3], variant=job.native.CASE[2], persistent=True)
    controllers = [MeasuredPlaneForecastSourceController(model, job.ArticulatedCollisionGeometry(job.URDF),
        forecast_source=mode, **options) for model, mode in zip(models, MODES, strict=True)]
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
                public = p, d, f, image, aux
                before = run.fingerprint(public)
                if before != reference['public_packet_sha256']:
                    raise ValueError('same complete verified public observation required')
                decisions = []
                for controller in controllers:
                    decisions.append(controller.observe(p, d, f, now_ns=now,
                        auxiliary_rgb=image, auxiliary_depth=aux))
                    if run.fingerprint(public) != before: raise ValueError('controller changed public inputs')
                baseline, candidate = [json.loads(run.canonical(value)) for value in decisions]
                check = compare(baseline, candidate, reference['decision'], frame=frame)
                append(dict(tick=frame, baseline=baseline, decision=candidate, comparison=check,
                    public_packet_sha256=before, public_inputs_unchanged=True))
                if (OUTPUT/'context_decisions.jsonl.gz').stat().st_size > job.MAX_OUTPUT_BYTES:
                    raise ValueError('original 2-GiB compressed output allowance exceeded')
                count += 1
                forecasts += int(bool(baseline['new_selection'] and 'prediction' in baseline['new_selection']))
                if calls != [forecasts, 0]:
                    raise ValueError('exact observed baseline forwards and zero nominal model forwards required')
                if frame % 25 == 0 or check['stop']:
                    print('MEASURED_PLANE_FORECAST_SOURCE_FRAME', frame, check['stop_reason'], flush=True)
                if check['stop']: break
    finally:
        for handle in handles: handle.remove()
    if not count or not check['stop']: raise ValueError('complete explicit comparison boundary required')
    if any(job.state_digest(m.state_dict()) != job.MODEL_SHA
            or any(p.grad is not None for p in m.parameters()) for m in models):
        raise ValueError('both assigned models must remain unchanged without gradients')
    return result_report(count, forecasts, check, baseline, candidate)


def main(source_only=False):
    if (not __debug__ or any(run.os.environ.get(k) != v for k, v in run.ENV.items())
            or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU environment required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive new forecast-source replay; no retry')
    sources = prepared_sources(); hardware = job.resources()
    if source_only:
        print('MEASURED_PLANE_FORECAST_SOURCE_PREFLIGHT', len(sources), json.dumps(hardware), flush=True); return
    scheduling = previous.cpu_slot(); previous.admit(sources)
    if run.owner_live(run.read_json(previous.OUTPUT, 'launch.json')['owner']):
        raise ValueError('preceding combined-controller replay must end before another full CPU replay')
    hardware = job.resources(); run.cv2.setNumThreads(1)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    run.create_output(OUTPUT); process = psutil.Process()
    run.write_json(OUTPUT/'launch.json', dict(source_sha256=sources, hardware=hardware, scheduling=scheduling,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        modes=list(MODES), maximum_frames=MAX_FRAMES, stop_at_first_changed_command_or_terminal=True,
        model_state_sha256=job.MODEL_SHA, native_execution=False, automatic_retry=False))
    print('MEASURED_PLANE_FORECAST_SOURCE_LAUNCHED', run.digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        report = replay(); check_output(report); previous.admit(sources)
        run.write_json(OUTPUT/'report.json', report)
        ids = {n: run.digest(OUTPUT/n) for n in ('launch.json', 'context_decisions.jsonl.gz', 'report.json')}
        run.verify_artifacts(OUTPUT, ids); run.verify(sources)
        run.write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_FORECAST_SOURCE_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, wall_s=time.perf_counter()-start,
            original_raw_inputs_reauthenticated_before_and_after=True,
            native_execution=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False))
        print('MEASURED_PLANE_FORECAST_SOURCE_COMPLETE', run.digest(OUTPUT/'result.json'),
            report['boundary_comparison'], flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MEASURED_PLANE_FORECAST_SOURCE_PREFIX_FAILURE',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    main(parser.parse_args().source_preflight_only)
