"""Ended-owner verification of every consumed raw packet and recorded pose row.

Does not rerun visual fitting, infer unexecuted commands, or admit native
completion. Both scientific-negative and complete planned histories are kept.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from lewm.causal_sensor_state import _identity
from lewm.measured_floor_transport_development import current_measured_floor_pose
from scripts import replay_go2_measured_plane_observer_history_v1 as run
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/verify_go2_measured_plane_observer_history_v1.py'
TEST = 'lewm/tests/test_measured_plane_observer_history_completion_development.py'
OUTPUT = run.ROOT/'docs/go2_measured_plane_observer_history_completion_2026-09-11.json'
LAUNCH_SHA = '8f09edbb77d103e3fe37e6f021da16be810a1696b588dea2264e98489f30afe1'
ARTIFACTS = {'launch.json', 'progress.jsonl', 'context_decisions.jsonl.gz', 'report.json'}


def visual_identity(value):
    if type(value.get('identity')) is not list:
        raise ValueError('serialized visual identity must be an explicit JSON array')
    return value | dict(identity=_identity(tuple(value['identity'])))


def floor_identity(value):
    if type(value.get('identity')) is not list:
        raise ValueError('serialized floor identity must be an explicit JSON array')
    restored = value | dict(identity=_identity(tuple(value['identity'])),
        original_visual_evidence=visual_identity(value['original_visual_evidence']))
    if 'floor_transport' in value:
        restored['floor_transport'] = value['floor_transport'] | dict(
            anchor=floor_identity(value['floor_transport']['anchor']))
    return restored


def original_owner_ended(launch):
    if (launch['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or run.owner_live(launch['owner'])):
        raise ValueError('original observer owner must be ended on its recorded boot')


def check_row(row, recorded, *, frame, public):
    if (set(row) != {'tick','original','candidate','original_floor','original_floor_error','candidate_floor','comparison'}
            or type(row['tick']) is not int or row['tick'] != frame):
        raise ValueError('complete exact sequential observer row required')
    run.compare_original(recorded, row['original'], row['original_floor'], row['original_floor_error'], frame)
    policy, depth, fast, image, auxiliary = public
    now = 1_500_000_000+frame*100_000_000
    candidate = row['candidate']
    for evidence in (row['original'], candidate):
        if evidence['decision_ns'] != now:
            raise ValueError('exact actual decision clock required')
        if evidence['terminal_failure'] is None:
            run.current_dual_camera_pose(visual_identity(evidence), policy, image, auxiliary,
                identity=(0,0,0), now_ns=now)
    for floor in (row['original_floor'], row['candidate_floor']):
        if floor is not None:
            current_measured_floor_pose(floor_identity(floor), identity=(0,0,0), now_ns=now)
    error = row['comparison']['candidate_floor_failure']
    if (candidate['terminal_failure'] is not None and (row['candidate_floor'] is not None or error != 'visual observer terminal')
            or candidate['terminal_failure'] is None and ((error is None) != (row['candidate_floor'] is not None))
            or error is not None and (type(error) is not str or not error)):
        raise ValueError('candidate visual/floor failures require exact explicit negative accounting')
    stop = ('CANDIDATE_VISUAL_FAILURE' if candidate['terminal_failure'] is not None else
        'CANDIDATE_FLOOR_FAILURE' if error is not None else 'FIXED_HISTORY_END' if frame == run.FRAMES-1 else None)
    expected = dict(frame=frame, original_visual_exact=True, original_floor_exact_or_terminal_reproduced=True,
        candidate_visual_failure=candidate['terminal_failure'], candidate_floor_failure=error,
        stop_reason=stop, raw_packet_sha256=run.fingerprint(public),
        original_requested_command=recorded['decision']['requested_command'], command_selected=False)
    if run.canonical(row['comparison']) != run.canonical(expected):
        raise ValueError('entire comparison and actual raw packet fingerprint must reconstruct')
    if (candidate['measured_plane_constrained_estimator'] is not True
            or candidate['original_global_floor_gate_unchanged'] is not True
            or candidate['original_temporal_gate_values_unchanged'] is not True
            or candidate['reference_history_reset'] is not False
            or candidate['measured_plane_evidence_current'] != (candidate['current_pose'] is not None)):
        raise ValueError('exact candidate estimator and negative-scope labels required')
    return expected


def reconstructed_report(count, refined, missing, row):
    check = row['comparison']; stop = check['stop_reason']
    if not 1 <= count <= run.FRAMES or stop not in ('CANDIDATE_VISUAL_FAILURE','CANDIDATE_FLOOR_FAILURE','FIXED_HISTORY_END'):
        raise ValueError('complete declared stopping boundary required')
    if stop == 'FIXED_HISTORY_END' and count != run.FRAMES:
        raise ValueError('fixed history end cannot abbreviate the planned history')
    return dict(frames=count, planned_frames=run.FRAMES, stop_reason=stop,
        complete_planned_history=count == run.FRAMES, refined_selected_pairs=refined,
        missing_floor_original_selected_pairs=missing, final_comparison=check,
        final_candidate_visual=row['candidate'], final_candidate_floor=row['candidate_floor'],
        full_original_evidence_reproduced_for_consumed_prefix=True,
        actual_public_fast_gyro_history_replayed=True, independent_histories_from_frame_zero=True,
        original_global_floor_gate_unchanged=True, candidate_failure_preserved=stop != 'FIXED_HISTORY_END',
        fixed_executed_trajectory_diagnostic=True, candidate_commands_selected=False,
        hypothetical_navigation_outcome_inferred=False, model_loaded=False, mapper_replayed=False,
        controller_replayed=False, native_completion_admitted=False, native_execution=False,
        navigation_recovered=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False)


def verify_result(result_sha):
    if (not __debug__ or any(run.os.environ.get(k) != v for k,v in run.ENV.items())
            or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU verification environment required')
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive completion receipt required')
    if not isinstance(result_sha, str) or re.fullmatch('[0-9a-f]{64}', result_sha) is None:
        raise ValueError('actual completed result SHA-256 required')
    root = run.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('execution failure must be preserved')
    run.verify_artifacts(root, {'launch.json':LAUNCH_SHA})
    launch = run.read_json(root, 'launch.json')
    original_owner_ended(launch)
    run.verify_artifacts(root, {'result.json':result_sha})
    result = run.read_json(root, 'result.json')
    ids = result['artifact_sha256']
    if (set(result) != {'status','source_sha256','artifact_sha256','report','wall_s',
                'native_completion_admitted','native_execution','navigation_qualified','goal_achieved'}
            or set(ids) != ARTIFACTS or ids['launch.json'] != LAUNCH_SHA
            or result['source_sha256'] != launch['source_sha256']
            or result['status'] != 'MEASURED_PLANE_OBSERVER_HISTORY_V1_COMPLETE'
            or any(result[k] is not False for k in ('native_completion_admitted','native_execution','navigation_qualified','goal_achieved'))
            or type(result['wall_s']) not in (int,float) or not math.isfinite(result['wall_s']) or result['wall_s'] <= 0):
        raise ValueError('exact completed launch, result and negative scope required')
    run.verify_artifacts(root, ids)
    if ((root/'context_decisions.jsonl.gz').stat().st_size > run.MAX_OUTPUT_BYTES
            or (root/'progress.jsonl').stat().st_size > 4*1024**2):
        raise ValueError('declared bounded complete replay outputs required')
    sources = discover_sources((SOURCE, TEST), launch['source_sha256'])
    run.verify(sources)
    inputs = launch['input_artifact_sha256']
    run.verify_artifacts(run.diagnosis.NATIVE, inputs)
    run.admit_cpu()
    directory = run.diagnosis.NATIVE/run.diagnosis.CASE
    reader = run.pipeline.ExtendedBudgetRGBDReplay(directory)
    acquisitions = run.read_json(directory, 'auxiliary_camera_audit.json')
    if len(reader.frames) != 3848 or len(acquisitions) != 3848:
        raise ValueError('same full persisted acquisition population required')
    count = refined = missing = 0
    progress = []
    stopped = False
    with closing(run.pipeline.read_rows(root)) as rows, closing(run.pipeline.read_rows(directory)) as recorded:
        for row in rows:
            if stopped or count >= run.FRAMES: raise ValueError('no rows after declared diagnostic stop')
            policy, depth, fast, now = reader.packet(count)
            image, auxiliary = run.pipeline.rgb_packet(directory, count, policy,
                run.public_acquisition(acquisitions[count]), now_ns=now)
            check = check_row(row, next(recorded), frame=count, public=(policy,depth,fast,image,auxiliary))
            candidate = row['candidate']; pair = candidate['measured_plane_selected_pair']
            refined += int(candidate['terminal_failure'] is None and pair is not None and pair['applied'])
            missing += int(candidate['terminal_failure'] is None and pair is not None and not pair['applied'])
            stopped = check['stop_reason'] is not None
            if count % 50 == 0 or stopped: progress.append(check)
            count += 1
    if not count or not stopped: raise ValueError('complete stream through declared boundary required')
    expected = reconstructed_report(count, refined, missing, row)
    if (run.canonical(expected) != run.canonical(result['report'])
            or run.canonical(expected) != run.canonical(run.read_json(root, 'report.json'))
            or [json.loads(line) for line in (root/'progress.jsonl').read_text().splitlines()] != progress):
        raise ValueError('complete report and every progress row must reconstruct')
    run.verify(sources); run.verify_artifacts(root, ids | {'result.json':result_sha})
    run.verify_artifacts(run.diagnosis.NATIVE, inputs); original_owner_ended(launch)
    run.write_json(OUTPUT, dict(status='MEASURED_PLANE_OBSERVER_HISTORY_COMPLETION_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, result_sha256=result_sha,
        original_launch_sha256=LAUNCH_SHA, artifact_sha256=ids, report=expected,
        complete_output_stream_checked=True, actual_consumed_raw_packets_reconstructed=count,
        original_recorded_visual_and_floor_evidence_checked=True, original_owner_ended=True,
        candidate_pose_and_floor_composition_rechecked=True, original_inputs_rehashed_before_and_after=True,
        visual_fitting_reexecuted=False, floor_pixels_refitted=False,
        native_completion_admitted=False, native_execution=False, navigation_recovered=False,
        real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
    print('MEASURED_PLANE_OBSERVER_COMPLETION_VERIFIED', run.digest(OUTPUT), count, expected['stop_reason'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-sha256', required=True)
    verify_result(parser.parse_args().result_sha256)
