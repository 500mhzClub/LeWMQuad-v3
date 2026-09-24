"""Paired current-native controller replay ending at the first changed command."""
import argparse
from contextlib import closing
import json
import time

import psutil
import torch

from lewm.measured_plane_chained_anchor_development import MeasuredPlaneChainedAnchorController
from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import measured_plane_chained_controller_prefix_comparison_development as comparison
from scripts import replay_go2_measured_plane_single_pass_full_history_v1 as full
from scripts import await_go2_measured_plane_full_history_timing_v1 as timing_waiter
from scripts import probe_go2_measured_plane_return_anchor_pairs_v1 as probe
from scripts.startup_source_inventory_development import discover_sources

run, inputs, native, original = full.run, full.inputs, full.native, full.original
SOURCE = 'scripts/replay_go2_measured_plane_chained_controller_prefix_v1.py'
TESTS = ('lewm/tests/test_measured_plane_chained_controller_prefix_development.py',
    'lewm/tests/test_measured_plane_chained_anchor_development.py')
PROTOCOL = 'docs/go2_measured_plane_chained_controller_prefix_v1_2026-09-12.md'
OUTPUT = run.BASE/'go2_measured_plane_chained_controller_prefix_v1_attempt_001'
PROBE_LAUNCH_SHA = '1abd1409776931187bdc2ce4959d57e3d47146c4350222f71f06f2377bcfc267'
PROBE_RESULT_SHA = 'd3a01cc5b667b7038964adf04ee135179af8a35c7d8bf0ed875b5baae5831c6e'
CPU_LAUNCH_SHA = 'c63f9bccb0cdfc273b1d3fd721310da5fac384f1350dcfc4625098fd103184a4'
CPU_OWNER = dict(pid=2924370, created=1789167167.42, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B',
    'scripts/await_go2_measured_plane_full_history_timing_v1.py'])
MAX_OUTPUT_BYTES = 2*1024**3


def completed_probe():
    root = probe.OUTPUT
    run.verify_artifacts(root, {'launch.json':PROBE_LAUNCH_SHA, 'result.json':PROBE_RESULT_SHA})
    launch = run.read_json(root, 'launch.json'); result = run.read_json(root, 'result.json')
    if ((root/'failure.json').exists() or (root/'failure.json').is_symlink()
            or result['status'] != 'MEASURED_PLANE_RETURN_ANCHOR_PAIR_PROBE_COMPLETE'
            or result['launch_sha256'] != PROBE_LAUNCH_SHA
            or result['input_sha256'] != launch['input_sha256']
            or launch['native_launch_sha256'] != inputs.LEARNED_LAUNCH_SHA
            or launch['collection_sha256'] != probe.COLLECTION_SHA
            or len(result['pairs']) != 64):
        raise ValueError('same completed fixed pair diagnostic required')
    run.verify(launch['source_sha256'])
    run.verify_artifacts(native.OUTPUT, {original.CASE[0]+'/'+name:sha
        for name,sha in launch['input_sha256'].items()})
    return launch, result


def cpu_launch():
    root = timing_waiter.OUTPUT
    run.verify_artifacts(root, {'launch.json':CPU_LAUNCH_SHA})
    launch = run.read_json(root, 'launch.json')
    if (launch['owner'] != CPU_OWNER
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('exact reserved full-history timing owner and boot required')
    run.verify(launch['source_sha256'])
    return launch


def prepared_sources(seeds=()):
    learned = inputs.learned_launch(); paired, _ = completed_probe(); cpu = cpu_launch()
    sources = discover_sources((SOURCE, PROTOCOL, *TESTS, *seeds), merge_sources(
        learned['source_sha256'], paired['source_sha256'], cpu['source_sha256']))
    run.verify(sources)
    return sources


def cpu_slot(native_sha, cpu_sha, sources):
    """Verify the ended prior work for ordering, without rerunning its science."""
    launch = cpu_launch(); root = timing_waiter.OUTPUT
    if run.owner_live(CPU_OWNER): raise ValueError('reserved timing waiter must finish and end first')
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve timing failure; no automatic ordering bypass')
    run.verify_artifacts(root, {'result.json':cpu_sha})
    result = run.read_json(root, 'result.json'); report = result['report']
    if (result['status'] != 'MEASURED_PLANE_FULL_HISTORY_TIMING_WAIT_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256'].get('launch.json') != CPU_LAUNCH_SHA
            or set(result['artifact_sha256']) != {'launch.json','events.jsonl','replay_stdout.log','completion.json'}
            or any(sources.get(k) != v for k,v in result['source_sha256'].items())
            or report['native_result_sha256'] != native_sha
            or report['complete_actual_native_history'] is not True
            or report['original_owner_ended'] is not True
            or report['complete_rows_states_and_public_packets_reconstructed'] is not True
            or report['original_raw_artifacts_reauthenticated'] is not True):
        raise ValueError('complete exact timing waiter and same native input required')
    run.verify_artifacts(root, result['artifact_sha256'])
    if run.read_json(root, 'completion.json') != report:
        raise ValueError('complete saved timing-waiter receipt required')
    child = full.OUTPUT; child_launch = run.read_json(child, 'launch.json')
    if (run.owner_live(child_launch['owner'])
            or child_launch['boot_id'] != launch['boot_id']
            or (child/'failure.json').exists() or (child/'failure.json').is_symlink()):
        raise ValueError('ended successful preceding CPU replay required')
    run.verify_artifacts(child, {'result.json':report['replay_result_sha256']})
    replay = run.read_json(child, 'result.json')
    if (replay['status'] != 'MEASURED_PLANE_SINGLE_PASS_FULL_HISTORY_V1_COMPLETE'
            or replay['source_sha256'] != child_launch['source_sha256']
            or any(sources.get(k) != v for k,v in replay['source_sha256'].items())
            or replay['complete_output_and_public_packets_rechecked'] is not True
            or replay['original_raw_inputs_reauthenticated_before_and_after'] is not True
            or child_launch['input_admission']['learned_result_sha256'] != native_sha
            or replay['report']['frames'] != report['frames']
            or set(replay['artifact_sha256']) != {'launch.json','comparison.jsonl','state_checks.json','resource_monitor.jsonl','report.json'}):
        raise ValueError('same complete prior replay for CPU ordering required')
    run.verify_artifacts(child, replay['artifact_sha256'])
    if run.read_json(child, 'report.json') != replay['report']:
        raise ValueError('complete saved prior replay report required')
    return dict(timing_waiter_result_sha256=cpu_sha, preceding_replay_result_sha256=report['replay_result_sha256'],
        preceding_waiter_and_replay_owners_ended=True, timing_improvement_required=False,
        preceding_replay_science_reexecuted=False)


def admit(native_sha, cpu_sha, sources):
    admitted = inputs.admit(native_sha, sources)
    ordered = cpu_slot(native_sha, cpu_sha, sources)
    launch, _ = completed_probe()
    if any(sources.get(k) != v for k,v in launch['source_sha256'].items()):
        raise ValueError('complete pair-probe source ancestry required')
    result = run.read_json(native.OUTPUT, 'result.json'); record = result['conditions'][0]
    count = record['collection']['decisions']
    if (count != 3124 or record['collection']['schedule_terminal'] != 'SENSOR_OR_MODEL_FAILURE'
            or run.digest(native.OUTPUT/original.CASE[0]/'result.json') != probe.COLLECTION_SHA):
        raise ValueError('same complete original negative episode required')
    return dict(learned_result_sha256=native_sha, learned_launch_sha256=inputs.LEARNED_LAUNCH_SHA,
        frames=count, model_state_sha256=inputs.job.MODEL_SHA,
        complete_raw_artifact_roster_sha256=run.fingerprint(admitted['learned_artifact_sha256']),
        original_context_sha256=admitted['learned_artifact_sha256'][original.CASE[0]+'/context_decisions.jsonl.gz'],
        complete_raw_native_audit_verified=True, pair_probe_result_sha256=PROBE_RESULT_SHA,
        original_native_scientific_success_required=False, **ordered)


def report_for(count, totals, common, first_anchor, check, old, candidate):
    return dict(frames=count, actual_model_forward_calls=totals, common_forecast_comparisons=common,
        first_recorded_candidate_anchor_reacquisition_frame=first_anchor, boundary_comparison=check,
        boundary_original=old, boundary_candidate=candidate, model_state_sha256=inputs.job.MODEL_SHA,
        two_fresh_independent_models=True, models_unchanged_without_gradients=True,
        complete_original_decisions_reproduced=True, public_packets_unchanged=True,
        candidate_observer_independently_replayed=False,
        following_changed_command_outcome_consumed=False, changed_command_executed=False,
        native_execution=False, navigation_recovered=False, real_time_qualified=False,
        hardware_qualified=False, goal_achieved=False)


def check_output(report, admission):
    maximum = admission['frames']; directory = native.OUTPUT/original.CASE[0]
    tape = run.read_json(directory, 'command_tape.json')
    count = common = 0; totals = [0,0]; first_anchor = None; stopped = False
    with closing(run.pipeline.read_rows(OUTPUT)) as rows, closing(run.pipeline.read_rows(directory)) as originals, \
            closing(full.packets(directory, maximum)) as packets:
        for row in rows:
            if stopped or count >= report['frames']: raise ValueError('no recorded output after comparison boundary')
            recorded = next(originals); packet, now = next(packets)
            full.comparison.reference_endpoint(recorded, tape, count, maximum)
            if (row['tick'] != count or row['public_packet_sha256'] != run.fingerprint(packet)
                    or row['observation_now_ns'] != now or row['public_inputs_unchanged'] is not True):
                raise ValueError('same actual reconstructed public packet for every output required')
            old, candidate = row['original'], row['decision']
            check = comparison.compare(old,candidate,recorded['decision'],frame=count,
                maximum_frames=maximum,model_calls=row['actual_model_forward_calls'])
            if check != row['comparison']: raise ValueError('complete recorded comparison must reconstruct')
            totals = [a+b for a,b in zip(totals,check['actual_model_forward_calls'],strict=True)]
            common += int(check['original_forecast_compared'])
            if first_anchor is None and check['candidate_anchor_reacquisition_recorded']: first_anchor = count
            count += 1; stopped = check['stop']
    if not count or not stopped: raise ValueError('explicit first changed command or terminal boundary required')
    expected = report_for(count,totals,common,first_anchor,check,old,candidate)
    if run.canonical(expected) != run.canonical(report): raise ValueError('whole report must reconstruct')


def replay(admission):
    maximum = admission['frames']; models = [original.assigned_model() for _ in range(2)]
    storage = [{v.untyped_storage().data_ptr() for v in model.state_dict().values() if v.numel()} for model in models]
    if (models[0] is models[1] or not storage[0].isdisjoint(storage[1])
            or any(model.training or original.state_digest(model.state_dict()) != inputs.job.MODEL_SHA
                or any(p.grad is not None for p in model.parameters()) for model in models)):
        raise ValueError('two fresh independent original corrected evaluation models required')
    options = dict(public_mission=original.public_mission(2), navigation_ticks=4000,
        condition=original.CASE[3], variant=original.CASE[2], persistent=True)
    controllers = [cls(model,original.ArticulatedCollisionGeometry(inputs.job.URDF),**options)
        for cls,model in zip((MeasuredPlaneResidualController,MeasuredPlaneChainedAnchorController),models,strict=True)]
    calls = [0,0]
    def record(index):
        def hook(module,args,output): calls[index] += 1
        return hook
    handles = [model.register_forward_hook(record(i)) for i,model in enumerate(models)]
    directory = native.OUTPUT/original.CASE[0]; tape = run.read_json(directory,'command_tape.json')
    count = common = 0; first_anchor = None; stopped = False
    try:
        with closing(run.pipeline.read_rows(directory)) as references, closing(full.packets(directory,maximum)) as packets, \
                run.pipeline.writer(OUTPUT) as append, (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            for recorded,(packet,now) in zip(references,packets,strict=True):
                frame = count
                full.comparison.reference_endpoint(recorded,tape,frame,maximum)
                before = run.fingerprint(packet); p,d,f,image,aux = packet; previous = calls.copy(); decisions = []
                for controller in controllers:
                    decision = controller.observe(p,d,f,now_ns=now,auxiliary_rgb=image,auxiliary_depth=aux)
                    decisions.append(json.loads(run.canonical(decision)))
                    if run.fingerprint(packet) != before: raise ValueError('controller changed actual public inputs')
                old,candidate = decisions; invoked = [a-b for a,b in zip(calls,previous,strict=True)]
                try:
                    check = comparison.compare(old,candidate,recorded['decision'],frame=frame,
                        maximum_frames=maximum,model_calls=invoked)
                except Exception as error:
                    append(dict(tick=frame,original=old,decision=candidate,comparison_failure=repr(error)))
                    raise
                append(dict(tick=frame,original=old,decision=candidate,comparison=check,
                    public_packet_sha256=before,observation_now_ns=now,public_inputs_unchanged=True,
                    actual_model_forward_calls=invoked))
                count += 1; common += int(check['original_forecast_compared'])
                if first_anchor is None and check['candidate_anchor_reacquisition_recorded']: first_anchor = frame
                if frame % 100 == 0 or check['stop']:
                    memory = psutil.virtual_memory(); disk = psutil.disk_usage(run.BASE)
                    monitor.write(json.dumps(dict(frame=frame,rss_bytes=psutil.Process().memory_info().rss,
                        memory_available_bytes=memory.available,artifact_free_bytes=disk.free))+'\n');monitor.flush()
                    if memory.available < 8*1024**3 or disk.free < 41*1024**3:
                        raise ValueError('retain RAM and native artifact reserve')
                    print('MEASURED_PLANE_CHAINED_CONTROLLER_FRAME',frame,check['stop_reason'],flush=True)
                if (OUTPUT/'context_decisions.jsonl.gz').stat().st_size > MAX_OUTPUT_BYTES:
                    raise ValueError('bounded paired output allowance exceeded')
                if check['stop']: stopped=True;break
    finally:
        for handle in handles: handle.remove()
    if not count or not stopped: raise ValueError('explicit comparison boundary required')
    if any(original.state_digest(model.state_dict()) != inputs.job.MODEL_SHA
            or any(p.grad is not None for p in model.parameters()) for model in models):
        raise ValueError('original assigned models must remain unchanged without gradients')
    return report_for(count,calls,common,first_anchor,check,old,candidate)


def main(native_sha=None,cpu_sha=None,source_only=False):
    if (not __debug__ or any(run.os.environ.get(k) != v for k,v in run.ENV.items()) or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU environment required')
    run.validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive controller replay; no retry or resume')
    sources = prepared_sources(); hardware = full.resources()
    if source_only:
        print('MEASURED_PLANE_CHAINED_SOURCE_PREFLIGHT',len(sources),json.dumps(hardware),flush=True);return
    if not native_sha or not cpu_sha: raise ValueError('actual completed native and timing-waiter result identities required')
    admission = admit(native_sha,cpu_sha,sources); hardware = full.resources()
    run.cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    run.create_output(OUTPUT);process=psutil.Process()
    run.write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_admission=admission,hardware=hardware,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid,created=process.create_time(),command=process.cmdline()),
        maximum_frames=admission['frames'],maximum_output_bytes=MAX_OUTPUT_BYTES,
        stop_at_first_changed_command_or_terminal=True,native_execution=False,automatic_retry=False))
    print('MEASURED_PLANE_CHAINED_CONTROLLER_LAUNCHED',run.digest(OUTPUT/'launch.json'),flush=True)
    start=time.perf_counter()
    try:
        report=replay(admission);check_output(report,admission)
        if admit(native_sha,cpu_sha,sources)!=admission:raise ValueError('complete original inputs changed')
        run.write_json(OUTPUT/'report.json',report)
        ids={name:run.digest(OUTPUT/name) for name in ('launch.json','context_decisions.jsonl.gz','resource_monitor.jsonl','report.json')}
        run.verify_artifacts(OUTPUT,ids);run.verify(sources)
        run.write_json(OUTPUT/'result.json',dict(status='MEASURED_PLANE_CHAINED_CONTROLLER_PREFIX_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,wall_s=time.perf_counter()-start,
            complete_output_and_consumed_public_packets_rechecked=True,
            original_raw_inputs_reauthenticated_before_and_after=True,
            native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('MEASURED_PLANE_CHAINED_CONTROLLER_COMPLETE',run.digest(OUTPUT/'result.json'),report['boundary_comparison'],flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MEASURED_PLANE_CHAINED_CONTROLLER_PREFIX_FAILURE',
            reason=repr(error),automatic_retry=False,original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--learned-result-sha256')
    parser.add_argument('--timing-waiter-result-sha256');parser.add_argument('--source-preflight-only',action='store_true')
    args=parser.parse_args();main(args.learned_result_sha256,args.timing_waiter_result_sha256,args.source_preflight_only)
