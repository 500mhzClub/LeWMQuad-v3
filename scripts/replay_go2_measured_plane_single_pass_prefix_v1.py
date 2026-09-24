"""Paired performance integration on the complete verified causal prefix."""
import argparse
from contextlib import closing
import json
import math
import statistics
import time

import psutil
import torch

from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.measured_plane_single_pass_controller_development import MeasuredPlaneSinglePassController
from scripts.measured_plane_single_pass_comparison_development import normalize, state, STATE_TYPE_PATHS
from scripts import measured_plane_native_inputs_development as inputs
from scripts import await_go2_deferred_memo_completion_v1 as cpu
from scripts.startup_source_inventory_development import discover_sources

job, run = inputs.job, inputs.run
SOURCE = 'scripts/replay_go2_measured_plane_single_pass_prefix_v1.py'
TEST = 'lewm/tests/test_measured_plane_single_pass_prefix_development.py'
CONTROLLER_TEST = 'lewm/tests/test_measured_plane_single_pass_controller_development.py'
PROTOCOL = 'docs/go2_measured_plane_single_pass_prefix_v1_2026-09-11.md'
OUTPUT = run.BASE/'go2_measured_plane_single_pass_prefix_v1_attempt_001'
SINGLE_PROOF = 'docs/go2_single_pass_body_projected_completion_verification_2026-09-11.json'
SINGLE_SHA = 'f9833329610096f9d5776ccc208d0de33988d8ebd4e6cdff4895004aa5190f1e'
CPU_WAIT_SHA = '0beb4e02fcdbc1c86b07773c88f21176c99a6ce1a21ee237f073da634c25da26'
FRAMES = 123
STATE_FRAMES = (0, 3, 61, 122)


def sources():
    proof = inputs.completed_prefix()
    run.verify({SINGLE_PROOF: SINGLE_SHA})
    single = json.loads((run.ROOT/SINGLE_PROOF).read_text())
    if single['status'] != 'SINGLE_PASS_BODY_PROJECTED_COMPLETION_VERIFIED':
        raise ValueError('completed single-pass integration evidence required')
    inherited = dict(proof['source_sha256'])
    for name, sha in single['source_sha256'].items():
        if name in inherited and inherited[name] != sha:
            raise ValueError('source ancestry conflict')
        inherited[name] = sha
    result = discover_sources((SOURCE, TEST, CONTROLLER_TEST, PROTOCOL, SINGLE_PROOF,
        str(inputs.prefix.completed.OUTPUT.relative_to(run.ROOT))), inherited)
    run.verify(result)
    return result


def cpu_slot():
    run.verify_artifacts(cpu.OUTPUT, {'launch.json': CPU_WAIT_SHA})
    launch = run.read_json(cpu.OUTPUT, 'launch.json')
    if launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('same CPU owner boot required')
    if run.owner_live(launch['owner']) or run.owner_live(launch['original_owner']):
        raise ValueError('original full replay and its completion checker must end first')
    for root in (cpu.OUTPUT, cpu.job.OUTPUT):
        if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
            raise ValueError('preserve failed preceding replay/checker without bypass')
    result_sha = run.digest(cpu.OUTPUT/'result.json')
    result = run.read_json(cpu.OUTPUT, 'result.json')
    if (result['status'] != 'DEFERRED_MEMO_COMPLETION_WAIT_V1_COMPLETE'
            or result['artifact_sha256']['launch.json'] != CPU_WAIT_SHA
            or result['source_sha256'] != launch['source_sha256']
            or result['verification_executed_once'] is not True
            or result['completion_receipt'] != str(cpu.checker.OUTPUT.relative_to(run.ROOT))):
        raise ValueError('complete exact preceding CPU verification required')
    run.verify(result['source_sha256'])
    run.verify_artifacts(cpu.OUTPUT, result['artifact_sha256'] | {'result.json': result_sha})
    run.verify({result['completion_receipt']: result['completion_sha256']})
    proof = json.loads(cpu.checker.OUTPUT.read_text())
    if (proof['status'] != 'DEFERRED_MEMO_SINGLE_PASS_COMPLETION_VERIFIED'
            or proof['result_sha256'] != result['original_result_sha256']):
        raise ValueError('same actual completed CPU replay required')
    return dict(waiter_result_sha256=result_sha, completion_sha256=result['completion_sha256'],
        original_owners_ended=True, scheduling_evidence_only=True,
        pending_copier_optimization_adopted=False)


def admit(bindings):
    proof = inputs.completed_prefix()
    worker_name = str(job.worker.OUTPUT.relative_to(run.ROOT))
    run.verify({worker_name: job.WORKER_ADMISSION_SHA, SINGLE_PROOF: SINGLE_SHA})
    worker = json.loads(job.worker.OUTPUT.read_text())
    if (worker['status'] != 'EXTENDED_BUDGET_COMPLETED_WORKER_ADMITTED'
            or worker['model_state_sha256'] != job.MODEL_SHA
            or worker['model_state_unchanged'] is not True
            or run.owner_live(worker['original_worker'])):
        raise ValueError('same ended original worker and unchanged assigned model required')
    run.verify_artifacts(job.native.OUTPUT, worker['artifact_sha256'])
    run.verify(bindings)
    return proof


def summarize(rows, states):
    if len(rows) != FRAMES or [r['frame'] for r in rows] != list(range(FRAMES)):
        raise ValueError('complete ordered 123-observation prefix required')
    for row in rows:
        frame = row['frame']
        if (row['execution_order'] != ([0, 1] if frame % 2 == 0 else [1, 0])
                or row['complete_reference_decision_equal'] is not True
                or row['complete_normalized_candidate_equal'] is not True
                or row['public_inputs_unchanged'] is not True
                or row['forecast_compared'] is not (frame >= 3)
                or row['reference_decision_sha256'] != row['baseline_decision_sha256']):
            raise ValueError('complete ordered equal decisions and input evidence required')
        for key in ('baseline_controller_s', 'candidate_controller_s'):
            if type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError('finite positive controller times required')
    if ([s['frame'] for s in states] != list(STATE_FRAMES)
            or any(s['retained_observed_state_equal'] is not True for s in states)):
        raise ValueError('all four complete retained-state comparisons required')
    before = [r['baseline_controller_s'] for r in rows[3:]]
    after = [r['candidate_controller_s'] for r in rows[3:]]
    return dict(frames=FRAMES, raw_model_forecast_comparisons=FRAMES-3,
        observed_state_checks=states, normalized_state_type_paths=STATE_TYPE_PATHS,
        baseline='MeasuredPlaneResidualController', candidate='MeasuredPlaneSinglePassController',
        timing=dict(observations=len(before), baseline_median_s=statistics.median(before),
            candidate_median_s=statistics.median(after), baseline_total_s=math.fsum(before),
            candidate_total_s=math.fsum(after), baseline_over_100ms=sum(t > .1 for t in before),
            candidate_over_100ms=sum(t > .1 for t in after)),
        model_state_sha256=job.MODEL_SHA, model_state_unchanged=True,
        following_changed_command_observation_consumed=False,
        controller_observe_only_timed=True, sensor_acquisition_timed=False,
        isolated_benchmark=False, native_execution=False, navigation_qualified=False,
        real_time_qualified=False, goal_achieved=False)


def replay():
    launch = run.read_json(job.native.OUTPUT, 'launch.json')
    models = [job.native.assigned_model(launch) for _ in range(2)]
    storage = [{v.untyped_storage().data_ptr() for v in m.state_dict().values() if v.numel()} for m in models]
    if (models[0] is models[1] or not storage[0].isdisjoint(storage[1])
            or any(job.state_digest(m.state_dict()) != job.MODEL_SHA or m.training for m in models)):
        raise ValueError('two independent original evaluation-only corrected models required')
    options = dict(public_mission=job.public_mission(2), navigation_ticks=4000,
        condition=job.native.CASE[3], variant=job.native.CASE[2], persistent=True)
    controllers = [cls(model, job.ArticulatedCollisionGeometry(job.URDF), **options)
        for cls, model in zip((MeasuredPlaneResidualController, MeasuredPlaneSinglePassController), models, strict=True)]
    directory = job.native.OUTPUT/job.native.CASE[0]
    reader = run.pipeline.ExtendedBudgetRGBDReplay(directory)
    acquisitions = run.read_json(directory, 'auxiliary_camera_audit.json')
    tape = run.read_json(directory, 'command_tape.json')
    rows, states = [], []
    with closing(run.pipeline.read_rows(job.OUTPUT)) as references, (OUTPUT/'comparison.jsonl').open('x') as stream:
        for reference in references:
            frame = len(rows)
            if frame >= FRAMES or reference['tick'] != frame:
                raise ValueError('no observations beyond the verified causal boundary')
            p, d, f, now = reader.packet(frame)
            image, aux = run.pipeline.rgb_packet(directory, frame, p,
                run.public_acquisition(acquisitions[frame]), now_ns=now)
            public = p, d, f, image, aux
            before = run.fingerprint(public)
            if before != reference['public_packet_sha256']:
                raise ValueError('same complete verified public packet required')
            job.command_endpoint(dict(tick=frame, observation_index=frame, pre_sample_index=749+50*frame,
                decision=reference['original']), tape[frame], frame)
            order = [0, 1] if frame % 2 == 0 else [1, 0]
            decisions, elapsed = [None, None], [None, None]
            for index in order:
                start = time.perf_counter()
                decisions[index] = controllers[index].observe(p, d, f, now_ns=now,
                    auxiliary_rgb=image, auxiliary_depth=aux)
                elapsed[index] = time.perf_counter()-start
                if run.fingerprint(public) != before:
                    raise ValueError('controller changed public input packets')
            baseline, candidate = [json.loads(run.canonical(value)) for value in decisions]
            normalized = normalize(candidate)
            if (baseline != reference['decision'] or normalized != baseline or baseline['terminal'] is not None):
                run.write_json(OUTPUT/'decision_mismatch.json', dict(frame=frame,
                    reference=reference['decision'], baseline=baseline, candidate=candidate))
                raise ValueError('complete reference, baseline and candidate equality failed at '+str(frame))
            if frame < FRAMES-1 and baseline['requested_command'] != tape[frame]['requested_command']:
                raise ValueError('only the final prospective command may differ from executed history')
            selection = baseline['new_selection']
            row = dict(frame=frame, execution_order=order,
                baseline_controller_s=elapsed[0], candidate_controller_s=elapsed[1],
                public_packet_sha256=before, reference_decision_sha256=run.fingerprint(reference['decision']),
                baseline_decision_sha256=run.fingerprint(baseline), candidate_decision_sha256=run.fingerprint(candidate),
                complete_reference_decision_equal=True, complete_normalized_candidate_equal=True,
                public_inputs_unchanged=True, forecast_compared=bool(selection and 'prediction' in selection))
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); rows.append(row)
            if frame in STATE_FRAMES:
                hashes = [run.fingerprint(state(c)) for c in controllers]
                if hashes[0] != hashes[1]:
                    raise ValueError('full retained observed state changed at '+str(frame))
                states.append(dict(frame=frame, state_sha256=hashes[0], retained_observed_state_equal=True))
            if frame % 25 == 0 or frame == FRAMES-1:
                print('MEASURED_PLANE_SINGLE_PASS_FRAME', frame, flush=True)
    if any(job.state_digest(m.state_dict()) != job.MODEL_SHA
            or any(p.grad is not None for p in m.parameters()) for m in models):
        raise ValueError('assigned model weights or gradients changed')
    return summarize(rows, states)


def check_output(report):
    with (OUTPUT/'comparison.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    if run.canonical(summarize(rows, report['observed_state_checks'])) != run.canonical(report):
        raise ValueError('complete output and timing report must reconstruct')
    with closing(run.pipeline.read_rows(job.OUTPUT)) as references:
        for row, reference in zip(rows, references, strict=True):
            if (row['frame'] != reference['tick']
                    or row['public_packet_sha256'] != reference['public_packet_sha256']
                    or row['reference_decision_sha256'] != run.fingerprint(reference['decision'])):
                raise ValueError('every output row must bind the complete original decision and packet')


def main(preflight=False):
    if (not __debug__ or any(run.os.environ.get(k) != v for k, v in run.ENV.items())
            or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic single-thread CPU environment required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive combined-controller replay; no retry or resume')
    bindings = sources()
    hardware = job.resources()
    if preflight:
        print('MEASURED_PLANE_SINGLE_PASS_SOURCE_PREFLIGHT', len(bindings), hardware, flush=True)
        return
    scheduling = cpu_slot()
    proof = admit(bindings)
    run.cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    hardware = job.resources()
    run.create_output(OUTPUT)
    process = psutil.Process()
    run.write_json(OUTPUT/'launch.json', dict(source_sha256=bindings, hardware=hardware,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        controller_completion_sha256=inputs.prefix.COMPLETION_SHA,
        reference_artifact_sha256=proof['artifact_sha256'], single_pass_completion_sha256=SINGLE_SHA,
        model_state_sha256=job.MODEL_SHA, environment=run.ENV, scheduling=scheduling,
        frames=FRAMES, state_frames=list(STATE_FRAMES), automatic_retry=False, native_execution=False))
    print('MEASURED_PLANE_SINGLE_PASS_LAUNCHED', run.digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        report = replay(); check_output(report); admit(bindings)
        if cpu_slot() != scheduling:
            raise ValueError('preceding CPU completion changed during execution')
        run.write_json(OUTPUT/'report.json', report)
        ids = {name: run.digest(OUTPUT/name) for name in ('launch.json', 'comparison.jsonl', 'report.json')}
        run.verify_artifacts(OUTPUT, ids); run.verify(bindings)
        run.write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_SINGLE_PASS_PREFIX_V1_COMPLETE',
            source_sha256=bindings, artifact_sha256=ids, report=report, wall_s=time.perf_counter()-start,
            complete_output_rechecked=True, original_raw_artifacts_reauthenticated_before_and_after=True,
            native_execution=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False))
        print('MEASURED_PLANE_SINGLE_PASS_COMPLETE', run.digest(OUTPUT/'result.json'), report['timing'], flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MEASURED_PLANE_SINGLE_PASS_PREFIX_FAILURE',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    main(parser.parse_args().source_preflight_only)
