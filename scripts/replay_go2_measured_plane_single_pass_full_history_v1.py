"""Compare the complete learned native history with the single-pass controller."""
import argparse
from contextlib import closing
import json
import time

import psutil
import torch

from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.measured_plane_single_pass_controller_development import MeasuredPlaneSinglePassController
from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import measured_plane_full_history_timing_development as comparison
from scripts import nominal_measured_plane_native_inputs_development as inputs
from scripts import reactive_measured_plane_native_prefix_development as cpu
from scripts import replay_go2_measured_plane_single_pass_prefix_v1 as short
from scripts.startup_source_inventory_development import discover_sources

run = inputs.run
native = inputs.learned
original = native.original
SOURCE = 'scripts/replay_go2_measured_plane_single_pass_full_history_v1.py'
TESTS = ('lewm/tests/test_measured_plane_full_history_timing_development.py',
    'lewm/tests/test_measured_plane_full_history_replay_development.py')
PROTOCOL = 'docs/go2_measured_plane_single_pass_full_history_v1_2026-09-11.md'
OUTPUT = run.BASE/'go2_measured_plane_single_pass_full_history_v1_attempt_001'
SHORT_LAUNCH_SHA = '7b0b3bf417344ee90d3f74ecc6c2c613623119f98a4ee8273b8f6c3f61e4dc42'
SHORT_RESULT_SHA = '5d0e1b47ca80b8c19474dd7a371957867c71ed9835cc97bc342b19752a22f087'
MAX_OUTPUT_BYTES = 2*1024**3


def completed_short_prefix():
    root = short.OUTPUT
    run.verify_artifacts(root, {'launch.json': SHORT_LAUNCH_SHA, 'result.json': SHORT_RESULT_SHA})
    launch = run.read_json(root, 'launch.json'); result = run.read_json(root, 'result.json')
    if (run.owner_live(launch['owner']) or (root/'failure.json').exists() or (root/'failure.json').is_symlink()
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or result['status'] != 'MEASURED_PLANE_SINGLE_PASS_PREFIX_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256']['launch.json'] != SHORT_LAUNCH_SHA
            or result['original_raw_artifacts_reauthenticated_before_and_after'] is not True
            or result['complete_output_rechecked'] is not True):
        raise ValueError('complete ended original combined-controller prefix required')
    run.verify(result['source_sha256']); run.verify_artifacts(root, result['artifact_sha256'])
    if result['report'] != run.read_json(root, 'report.json'):
        raise ValueError('same complete combined-controller prefix report required')
    short.check_output(result['report'])
    return result


def prepared_sources(seeds=()):
    original_launch = inputs.learned_launch(); prefix = completed_short_prefix(); replay = cpu.completed_prefix()
    sources = discover_sources((SOURCE, PROTOCOL, *TESTS, *seeds), merge_sources(
        original_launch['source_sha256'], prefix['source_sha256'], replay['source_sha256']))
    run.verify(sources)
    return sources


def resources():
    hardware = run.hardware()
    if hardware['memory_available_bytes'] < 64*1024**3 or hardware['artifact_free_bytes'] < 43*1024**3:
        raise ValueError('64 GiB available RAM and 43 GiB artifact space required for the full pair and native reserve')
    return hardware


def admit(result_sha, sources):
    admitted = inputs.admit(result_sha, sources)
    result = run.read_json(native.OUTPUT, 'result.json'); record = result['conditions'][0]
    count = record['collection']['decisions']; comparison.state_frames(count)
    name = original.CASE[0]
    if admitted['learned_result_sha256'] != result_sha:
        raise ValueError('same complete learned episode required')
    for proof in (completed_short_prefix(), cpu.completed_prefix()):
        if any(sources.get(k) != v for k, v in proof['source_sha256'].items()):
            raise ValueError('complete immutable completed replay sources required')
    return dict(learned_result_sha256=result_sha, learned_launch_sha256=inputs.LEARNED_LAUNCH_SHA,
        frames=count, native_case=name, native_owner_ended=True, complete_raw_audit_verified=True,
        original_native_scientific_success_required=False,
        original_schedule_terminal=record['collection']['schedule_terminal'],
        original_verified_round_trip=record['verified_round_trip'],
        complete_raw_artifact_roster_sha256=run.fingerprint(admitted['learned_artifact_sha256']),
        original_context_sha256=admitted['learned_artifact_sha256'][name+'/context_decisions.jsonl.gz'],
        short_prefix_result_sha256=SHORT_RESULT_SHA, preceding_cpu_result_sha256=cpu.RESULT_SHA)


def packets(directory, count):
    reader = run.pipeline.ExtendedBudgetRGBDReplay(directory)
    acquisitions = run.read_json(directory, 'auxiliary_camera_audit.json')
    if len(acquisitions) not in (count, count+1):
        raise ValueError('complete original auxiliary acquisition population required')
    for frame in range(count):
        p, d, f, now = reader.packet(frame)
        image, aux = run.pipeline.rgb_packet(directory, frame, p,
            run.public_acquisition(acquisitions[frame]), now_ns=now)
        yield (p, d, f, image, aux), now


def check_output(report, admission):
    count = admission['frames']; directory = native.OUTPUT/original.CASE[0]
    tape = run.read_json(directory, 'command_tape.json'); rows = []
    with (OUTPUT/'comparison.jsonl').open() as stream, closing(run.pipeline.read_rows(directory)) as reference, \
            closing(packets(directory, count)) as public:
        for frame, (line, old, (packet, now)) in enumerate(zip(stream, reference, public, strict=True)):
            if frame >= count: raise ValueError('no output beyond the complete original episode')
            row = json.loads(line); comparison.reference_endpoint(old, tape, frame, count)
            if (row['frame'] != frame or row['public_packet_sha256'] != run.fingerprint(packet)
                    or row['observation_now_ns'] != now
                    or row['original_decision_sha256'] != run.fingerprint(old['decision'])):
                raise ValueError('every output must bind its complete actual decision and reconstructed sensor packet')
            rows.append(row)
    states = run.read_json(OUTPUT, 'state_checks.json')
    expected = comparison.summarize(rows, states, frames=count, model_sha=inputs.job.MODEL_SHA,
        input_result_sha=admission['learned_result_sha256'])
    if run.canonical(expected) != run.canonical(report):
        raise ValueError('complete timings, populations, states and scope must reconstruct')


def replay(admission):
    count = admission['frames']; check_frames = comparison.state_frames(count)
    models = [original.assigned_model() for _ in range(2)]
    storage = [{v.untyped_storage().data_ptr() for v in m.state_dict().values() if v.numel()} for m in models]
    if (models[0] is models[1] or not storage[0].isdisjoint(storage[1])
            or any(m.training or original.state_digest(m.state_dict()) != inputs.job.MODEL_SHA for m in models)):
        raise ValueError('two independent unchanged original evaluation models required')
    options = dict(public_mission=original.public_mission(2), navigation_ticks=4000,
        condition=original.CASE[3], variant=original.CASE[2], persistent=True)
    controllers = [cls(model, original.ArticulatedCollisionGeometry(inputs.job.URDF), **options)
        for cls, model in zip((MeasuredPlaneResidualController, MeasuredPlaneSinglePassController), models, strict=True)]
    calls = [0, 0]
    def record(index):
        def hook(module, args, output): calls[index] += 1
        return hook
    handles = [model.register_forward_hook(record(i)) for i, model in enumerate(models)]
    directory = native.OUTPUT/original.CASE[0]; tape = run.read_json(directory, 'command_tape.json')
    rows, states = [], []
    try:
        with closing(run.pipeline.read_rows(directory)) as references, closing(packets(directory, count)) as public, \
                (OUTPUT/'comparison.jsonl').open('x') as stream, (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            for frame, (reference, (packet, now)) in enumerate(zip(references, public, strict=True)):
                if frame >= count: raise ValueError('no observation beyond complete original episode')
                comparison.reference_endpoint(reference, tape, frame, count)
                before = run.fingerprint(packet); p, d, f, image, aux = packet
                order = [0, 1] if frame % 2 == 0 else [1, 0]
                elapsed, decisions = [None, None], [None, None]; previous_calls = calls.copy()
                for index in order:
                    start = time.perf_counter_ns()
                    decisions[index] = controllers[index].observe(p, d, f, now_ns=now,
                        auxiliary_rgb=image, auxiliary_depth=aux)
                    elapsed[index] = (time.perf_counter_ns()-start)/1e9
                    if run.fingerprint(packet) != before: raise ValueError('controller changed public inputs')
                baseline, candidate = [json.loads(run.canonical(d)) for d in decisions]
                normalized = comparison.normalize(candidate)
                if baseline != reference['decision'] or normalized != baseline:
                    run.write_json(OUTPUT/'decision_mismatch.json', dict(frame=frame,
                        original=reference['decision'], baseline=baseline, candidate=candidate))
                    raise ValueError('complete original and normalized controller decisions differ at '+str(frame))
                forwards = [a-b for a, b in zip(calls, previous_calls, strict=True)]
                if forwards[0] != forwards[1] or any(v not in (0, 1) for v in forwards):
                    raise ValueError('same actual per-observation model calls required')
                row = dict(frame=frame, execution_order=order, baseline_controller_s=elapsed[0],
                    candidate_controller_s=elapsed[1], observation_now_ns=now, public_packet_sha256=before,
                    original_decision_sha256=run.fingerprint(reference['decision']),
                    baseline_decision_sha256=run.fingerprint(baseline), candidate_decision_sha256=run.fingerprint(candidate),
                    normalized_candidate_decision_sha256=run.fingerprint(normalized),
                    complete_original_decision_equal=True, complete_normalized_candidate_equal=True,
                    public_inputs_unchanged=True, actual_model_forward_calls=forwards, forecast_compared=bool(forwards[0]))
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); rows.append(row)
                if frame in check_frames:
                    hashes = [run.fingerprint(comparison.observed_state(controller)) for controller in controllers]
                    if hashes[0] != hashes[1]: raise ValueError('retained observed state differs at '+str(frame))
                    states.append(dict(frame=frame, state_sha256=hashes[0], observed_state_equal=True))
                if frame % 100 == 0 or frame == count-1:
                    memory = psutil.virtual_memory(); disk = psutil.disk_usage(run.BASE)
                    monitor.write(json.dumps(dict(frame=frame, rss_bytes=psutil.Process().memory_info().rss,
                        memory_available_bytes=memory.available, artifact_free_bytes=disk.free))+'\n'); monitor.flush()
                    if memory.available < 8*1024**3 or disk.free < 41*1024**3:
                        raise ValueError('retain minimum RAM and existing native disk reserve')
                    print('MEASURED_PLANE_FULL_HISTORY_FRAME', frame, 'of', count, flush=True)
                if (OUTPUT/'comparison.jsonl').stat().st_size > MAX_OUTPUT_BYTES:
                    raise ValueError('bounded comparison output exceeded 2 GiB')
    finally:
        for handle in handles: handle.remove()
    if len(rows) != count:
        raise ValueError('complete original observation population required')
    if any(original.state_digest(m.state_dict()) != inputs.job.MODEL_SHA
            or any(p.grad is not None for p in m.parameters()) for m in models):
        raise ValueError('both assigned models must remain unchanged without gradients')
    run.write_json(OUTPUT/'state_checks.json', states)
    return comparison.summarize(rows, states, frames=count, model_sha=inputs.job.MODEL_SHA,
        input_result_sha=admission['learned_result_sha256'])


def main(result_sha=None, source_only=False):
    if (not __debug__ or any(run.os.environ.get(k) != v for k, v in run.ENV.items()) or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU environment required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive full-history comparison; no retry or resume')
    sources = prepared_sources(); hardware = resources()
    if source_only:
        print('MEASURED_PLANE_FULL_HISTORY_SOURCE_PREFLIGHT', len(sources), json.dumps(hardware), flush=True); return
    if not result_sha: raise ValueError('actual completed learned native result SHA required')
    admission = admit(result_sha, sources); hardware = resources()
    run.cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    run.create_output(OUTPUT); process = psutil.Process()
    run.write_json(OUTPUT/'launch.json', dict(source_sha256=sources, hardware=hardware, input_admission=admission,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        state_frames=comparison.state_frames(admission['frames']), actual_complete_population_required=True,
        native_execution=False, automatic_retry=False))
    print('MEASURED_PLANE_FULL_HISTORY_LAUNCHED', run.digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        report = replay(admission); check_output(report, admission)
        if admit(result_sha, sources) != admission: raise ValueError('complete original native inputs changed')
        run.write_json(OUTPUT/'report.json', report)
        ids = {n: run.digest(OUTPUT/n) for n in ('launch.json', 'comparison.jsonl', 'state_checks.json',
            'resource_monitor.jsonl', 'report.json')}
        run.verify_artifacts(OUTPUT, ids); run.verify(sources)
        run.write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_SINGLE_PASS_FULL_HISTORY_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, wall_s=time.perf_counter()-start,
            complete_output_and_public_packets_rechecked=True, original_raw_inputs_reauthenticated_before_and_after=True,
            native_execution=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False))
        print('MEASURED_PLANE_FULL_HISTORY_COMPLETE', run.digest(OUTPUT/'result.json'), report['timing']['all_observations'], flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MEASURED_PLANE_SINGLE_PASS_FULL_HISTORY_FAILURE',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--learned-result-sha256')
    parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args(); main(args.learned_result_sha256, args.source_preflight_only)
