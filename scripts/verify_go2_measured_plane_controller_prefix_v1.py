"""Authenticate the completed controller comparison and its causal boundary."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import math
import re

from scripts import replay_go2_measured_plane_controller_prefix_v1 as job
from scripts.startup_source_inventory_development import discover_sources

run = job.run
SOURCE = 'scripts/verify_go2_measured_plane_controller_prefix_v1.py'
TEST = 'lewm/tests/test_measured_plane_controller_prefix_completion_development.py'
OUTPUT = run.ROOT/'docs/go2_measured_plane_controller_prefix_completion_2026-09-11.json'
LAUNCH_SHA = '23a08dfaabd751f84aab26c066aac656c46571feb7a029df005c462a2a96913a'
ARTIFACTS = {'launch.json','context_decisions.jsonl.gz','report.json'}


def ended(launch):
    if launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('same recorded boot required')
    if run.owner_live(launch['owner']): raise ValueError('original controller replay must have ended')


def check_stream(rows,originals,observers,packet,tape):
    count = forecasts = 0
    stopped = False
    for row in rows:
        if stopped or count >= run.FRAMES:
            raise ValueError('output after declared causal boundary')
        if set(row) != {'tick','original','decision','comparison','public_packet_sha256',
                'public_inputs_unchanged','original_requested_command'} or row['tick'] != count:
            raise ValueError('complete ordered controller comparison row required')
        recorded,observer = next(originals),next(observers)
        job.command_endpoint(recorded,tape[count],count)
        public = packet(count)
        before = run.fingerprint(public)
        if (observer['tick'] != count or row['public_packet_sha256'] != before
                or observer['comparison']['raw_packet_sha256'] != before
                or row['public_inputs_unchanged'] is not True
                or row['original_requested_command'] != tape[count]['requested_command']):
            raise ValueError('same authenticated public packet and actual original command required')
        old,new = row['original'],row['decision']
        check = job.compare(old,new,recorded['decision'],observer,frame=count)
        if run.canonical(check) != run.canonical(row['comparison']):
            raise ValueError('comparison receipt must reconstruct')
        count += 1; forecasts += int(check['original_forecast_compared'])
        stopped = check['stop']
    if not count or not stopped: raise ValueError('complete explicit comparison boundary required')
    return dict(frames=count,raw_forecast_comparisons=forecasts,boundary_comparison=check,
        boundary_original=old,boundary_candidate=new,model_state_sha256=job.MODEL_SHA,
        two_fresh_models=True,model_states_unchanged=True,original_controller_fully_reproduced=True,
        candidate_observer_and_floor_fully_reproduced=True,source_trajectory_is_development_only=True,
        following_changed_command_outcome_consumed=False,changed_command_executed=False,
        native_parent_completion_admitted=False,full_training_ancestry_reexecuted=False,
        assigned_snapshot_and_correction_reverified_on_load=True,
        native_execution=False,navigation_recovered=False,real_time_qualified=False,
        hardware_qualified=False,goal_achieved=False)


def verify_result(result_sha):
    if (not __debug__ or any(run.os.environ.get(k) != v for k,v in run.ENV.items())
            or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU verification environment required')
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive completion receipt required')
    if not isinstance(result_sha,str) or re.fullmatch('[0-9a-f]{64}',result_sha) is None:
        raise ValueError('actual completed result SHA-256 required')
    root = job.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('original execution failure must be preserved')
    run.verify_artifacts(root,{'launch.json':LAUNCH_SHA})
    launch = run.read_json(root,'launch.json'); ended(launch)
    run.verify_artifacts(root,{'result.json':result_sha})
    result = run.read_json(root,'result.json'); ids = result['artifact_sha256']
    if (set(result) != {'status','source_sha256','artifact_sha256','report','wall_s',
                'native_execution','navigation_qualified','goal_achieved'}
            or result['status'] != 'MEASURED_PLANE_CONTROLLER_PREFIX_V1_COMPLETE'
            or set(ids) != ARTIFACTS or ids['launch.json'] != LAUNCH_SHA
            or result['source_sha256'] != launch['source_sha256']
            or any(result[k] is not False for k in ('native_execution','navigation_qualified','goal_achieved'))
            or type(result['wall_s']) not in (int,float) or not math.isfinite(result['wall_s']) or result['wall_s'] <= 0):
        raise ValueError('exact original launch, completed result and negative scope required')
    run.verify_artifacts(root,ids)
    if (root/'context_decisions.jsonl.gz').stat().st_size > job.MAX_OUTPUT_BYTES:
        raise ValueError('bounded full output required')
    proof = json.loads(job.completed.OUTPUT.read_text())
    native_proof = json.loads(job.worker.OUTPUT.read_text())
    job.require_positive_observer(proof)
    job.verify_inputs(launch['source_sha256'],proof,native_proof,
        launch['observer_completion_sha256'],launch['observer_waiter_result_sha256'])
    sources = discover_sources((SOURCE,TEST),launch['source_sha256']); run.verify(sources)
    directory = job.native.OUTPUT/job.native.CASE[0]
    reader = run.pipeline.ExtendedBudgetRGBDReplay(directory)
    acquisitions = run.read_json(directory,'auxiliary_camera_audit.json')
    tape = run.read_json(directory,'command_tape.json')
    def packet(frame):
        policy,depth,fast,now = reader.packet(frame)
        image,aux = run.pipeline.rgb_packet(directory,frame,policy,
            run.public_acquisition(acquisitions[frame]),now_ns=now)
        return policy,depth,fast,image,aux
    with closing(run.pipeline.read_rows(root)) as rows, \
            closing(run.pipeline.read_rows(directory)) as originals, \
            closing(run.pipeline.read_rows(run.OUTPUT)) as observers:
        expected = check_stream(rows,originals,observers,packet,tape)
    if (run.canonical(expected) != run.canonical(result['report'])
            or run.canonical(expected) != run.canonical(run.read_json(root,'report.json'))):
        raise ValueError('complete report must reconstruct from every comparison row')
    run.verify(sources); run.verify_artifacts(root,ids | {'result.json':result_sha})
    job.verify_inputs(launch['source_sha256'],proof,native_proof,
        launch['observer_completion_sha256'],launch['observer_waiter_result_sha256'])
    ended(launch)
    run.write_json(OUTPUT,dict(status='MEASURED_PLANE_CONTROLLER_PREFIX_COMPLETION_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
        original_launch_sha256=LAUNCH_SHA,result_sha256=result_sha,artifact_sha256=ids,report=expected,
        original_owner_ended=True,complete_output_stream_checked=True,
        actual_consumed_raw_packets_reconstructed=expected['frames'],
        complete_original_decisions_and_candidate_observer_evidence_compared=True,
        first_changed_command_boundary_rechecked=True,
        model_state_assertion_from_authenticated_frozen_runner=True,
        model_inference_reexecuted=False,visual_fitting_reexecuted=False,
        native_execution=False,navigation_recovered=False,real_time_qualified=False,
        hardware_qualified=False,goal_achieved=False))
    print('MEASURED_PLANE_CONTROLLER_COMPLETION_VERIFIED',run.digest(OUTPUT),
        expected['frames'],expected['boundary_comparison'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--result-sha256',required=True)
    verify_result(parser.parse_args().result_sha256)
