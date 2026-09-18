"""Paired raw controller replay and unprofiled timing for isolated receipt copies."""
import argparse
from itertools import islice
import json
import math
import statistics
import time
import cv2
import torch
from lewm.receipt_copied_anchored_controller_development import ReceiptCopiedAnchoredController
from scripts import profile_go2_adapter_controller_windows_v1 as profile
from scripts.replay_go2_residual_current_observation_planning_prefix_v1 import state_tree
from scripts.maze_decision_stream_development import read_rows
from scripts.navigation_artifact_root_development import create_output, validate_root, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT = profile.reference.BASE/'go2_receipt_copied_anchored_prefix_v1_attempt_001'
PROFILE_SHA = '8be3553ba54c67827790a281aaf3a08bd2facbc8aff39537dd7f353a2b3b3fb0'
SOURCE = 'scripts/replay_go2_receipt_copied_anchored_prefix_v1.py'
PROTOCOL = 'docs/go2_receipt_copied_anchored_v1_2026-09-10.md'
TEST = 'lewm/tests/test_receipt_copied_anchored_replay_development.py'
FRAMES = 405
STATE_FRAMES = (3, 12, 395, 404)


def normalize_candidate(decision):
    if (decision.get('controller') != 'receipt_copied_residual_anchored_continuation_controller_v1'
            or decision.get('anchored_selection_receipt_copy_enabled') is not True):
        raise ValueError('explicit isolated receipt-copy implementation required')
    result = decision.copy()
    result.pop('anchored_selection_receipt_copy_enabled')
    result['controller'] = 'residual_anchored_continuation_controller_v1'
    return result


def execution_order(frame):
    if type(frame) is not int or not 0 <= frame < FRAMES:
        raise ValueError('fixed ordered prefix frame required')
    return (0, 1) if frame % 2 == 0 else (1, 0)


def timing_summary(rows):
    if [r['frame'] for r in rows] != list(range(FRAMES)):
        raise ValueError('complete paired timing population required')
    for row in rows:
        if tuple(row['execution_order']) != execution_order(row['frame']):
            raise ValueError('fixed alternating controller execution order required')
        if any(type(row[key]) not in (float, int) or not math.isfinite(row[key]) or row[key] <= 0
                for key in ('original_controller_s', 'candidate_controller_s')):
            raise ValueError('finite positive paired controller timings required')
    windows = {}
    for name, (first, last) in {'post_warmup_prefix': (3, 404), **profile.WINDOWS}.items():
        selected = rows[first:last+1]
        old = [r['original_controller_s'] for r in selected]
        new = [r['candidate_controller_s'] for r in selected]
        windows[name] = dict(first_frame=first, last_frame=last, observations=len(selected),
            original_median_s=statistics.median(old), candidate_median_s=statistics.median(new),
            original_total_s=sum(old), candidate_total_s=sum(new),
            median_ratio=statistics.median(old)/statistics.median(new),
            total_ratio=sum(old)/sum(new),
            original_over_100ms=sum(t > .1 for t in old), candidate_over_100ms=sum(t > .1 for t in new))
    return windows


def profile_inputs():
    verify_artifacts(profile.OUTPUT, {'result.json': PROFILE_SHA})
    result = read_json(profile.OUTPUT, 'result.json')
    if result['status'] != 'ADAPTER_CONTROLLER_WINDOWS_PROFILE_V1_COMPLETE':
        raise ValueError('completed exact original controller profile required')
    verify(result['source_sha256']); verify_artifacts(profile.OUTPUT, result['artifact_sha256'])
    return result


def replay():
    reference = profile.reference; original = reference.original; case = reference.CASE
    old_launch = read_json(original.OUTPUT, 'launch.json')
    models = [original.assigned_model(old_launch, case) for _ in range(2)]
    if any(profile.state_digest(m.state_dict()) != reference.MODEL_SHA for m in models):
        raise ValueError('two fresh identical assigned models required')
    if any(a.data_ptr() == b.data_ptr() for a, b in zip(models[0].parameters(), models[1].parameters())):
        raise ValueError('independent model storage required')
    options = dict(public_mission=profile.public_mission(case[1]), navigation_ticks=profile.NAVIGATION_TICKS,
        condition=case[3], variant=case[2], persistent=True)
    geometry = profile.ArticulatedCollisionGeometry(reference.URDF)
    controllers = (profile.ResidualAnchoredContinuationController(models[0], geometry, **options),
        ReceiptCopiedAnchoredController(models[1], geometry, **options))
    directory = original.OUTPUT/case[0]; reader = profile.IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    reference_rows = [json.loads(line) for line in (profile.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    if len(reference_rows) != FRAMES: raise ValueError('complete original profiled prefix required')
    timings = []; states = []; forecasts = 0
    with (OUTPUT/'comparison.jsonl').open('x') as stream:
        for row in islice(read_rows(directory), FRAMES):
            frame = row['tick']
            if (frame != len(timings) or tape[frame]['tick'] != frame or tape[frame]['completed'] is not True
                    or tape[frame]['pre_sample_index'] != 749+50*frame or tape[frame]['post_sample_index'] != 799+50*frame):
                raise ValueError('ordered exact original completed command endpoints required')
            p,d,fast,now = reader.packet(frame)
            image,aux = profile.packet(directory,frame,p,profile.public_acquisition(acquisitions[frame]),now_ns=now)
            before = profile.fingerprint((p,d,fast,aux,image,now))
            if before != reference_rows[frame]['public_input_sha256']:
                raise ValueError('same original public packet required')
            decisions = [None, None]; elapsed = [None, None]
            order = execution_order(frame)
            for index in order:
                start = time.perf_counter()
                decisions[index] = controllers[index].observe(p,d,fast,now_ns=now,auxiliary_depth=aux,auxiliary_rgb=image)
                elapsed[index] = time.perf_counter()-start
                if profile.fingerprint((p,d,fast,aux,image,now)) != before:
                    raise ValueError('each controller must preserve all public input arrays')
            old, new = [json.loads(json.dumps(d)) for d in decisions]
            if (old != row['decision'] or normalize_candidate(new) != old or old['terminal'] is not None
                    or old['requested_command'] != tape[frame]['requested_command']
                    or reference.saved.identity(old) != reference_rows[frame]['original_decision_sha256']):
                raise ValueError('complete original and candidate raw decisions must match: '+str(frame))
            selection = old['new_selection']; forecasts += int(bool(selection and 'prediction' in selection))
            record = dict(frame=frame, execution_order=list(order), original_controller_s=elapsed[0],
                candidate_controller_s=elapsed[1], public_input_sha256=before,
                original_decision_sha256=reference.saved.identity(old), candidate_decision_sha256=reference.saved.identity(new),
                complete_original_decision_reconstructed=True, candidate_normalized_decision_exact=True,
                public_input_arrays_unchanged=True)
            stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush(); timings.append(record)
            if frame in STATE_FRAMES:
                witnesses = [profile.fingerprint(state_tree(dict(memory=c.memory, floor=c.mapper.floor,
                    occupied=c.mapper.occupied, residual=c.residual, history=c.history))) for c in controllers]
                if witnesses[0] != witnesses[1]: raise ValueError('complete retained observed state must match')
                states.append(dict(frame=frame,state_sha256=witnesses[0],complete_retained_observed_state_equal=True))
            if frame % 50 == 0: print('RECEIPT_COPIED_ANCHORED_RAW_FRAME',frame,flush=True)
    if len(timings) != FRAMES or forecasts != 402 or [r['frame'] for r in states] != list(STATE_FRAMES):
        raise ValueError('complete fixed paired prefix, forecasts and state checks required')
    if any(profile.state_digest(m.state_dict()) != reference.MODEL_SHA or any(p.grad is not None for p in m.parameters()) for m in models):
        raise ValueError('model state or gradients changed')
    return dict(frames=FRAMES,raw_model_forecast_comparisons=forecasts,observed_state_checks=states,
        model_state_sha256=reference.MODEL_SHA,model_state_unchanged=True,complete_original_decisions_reconstructed=True,
        complete_normalized_candidate_decisions_exact=True,public_input_arrays_unchanged=True,
        timing_windows=timing_summary(timings),alternating_execution_order=True,profiling_enabled=False,
        controller_observe_only_timed=True,sensor_acquisition_timed=False,isolated_benchmark=False,
        no_observation_405_consumed=True,native_execution=False,real_time_qualified=False,navigation_qualified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-preflight-only',action='store_true');args=parser.parse_args()
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive paired receipt-copy prefix required')
    preceding=profile_inputs()
    seeds=(SOURCE,PROTOCOL,TEST,'lewm/tests/test_receipt_copied_anchored_development.py',
        'docs/go2_adapter_controller_windows_profile_result_2026-09-10.md',
        'docs/go2_adapter_controller_windows_profile_verification_2026-09-10.json')
    sources=discover_sources(seeds,preceding['source_sha256']);verify(sources)
    resources=profile.reference.hardware();profile.resources_for(resources)
    if args.source_preflight_only:
        print('RECEIPT_COPIED_ANCHORED_SOURCE_PREFLIGHT_PASS',len(sources),flush=True);return
    print('RECEIPT_COPIED_ANCHORED_INPUT_ADMISSION_STARTED',flush=True)
    admission=profile.reference.admit_worker(profile.WORKER_SHA,sources)
    resources=profile.reference.hardware();profile.resources_for(resources);verify(sources);create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,profile_result_sha256=PROFILE_SHA,
        input_admission=admission,frames=FRAMES,state_frames=STATE_FRAMES,protocol=PROTOCOL,hardware=resources,
        model_state_sha256=profile.reference.MODEL_SHA,native_execution=False,model_training=False,
        profiling_enabled=False,imported_module_globals_mutated=False))
    print('RECEIPT_COPIED_ANCHORED_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    try:
        report=replay();verify(sources)
        if profile.reference.admit_worker(profile.WORKER_SHA,sources)!=admission:
            raise ValueError('original complete input admission changed')
        profile_inputs()
        ids={name:digest(OUTPUT/name) for name in ('launch.json','comparison.jsonl')};verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='RECEIPT_COPIED_ANCHORED_PREFIX_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,profile_result_sha256=PROFILE_SHA,report=report,
            wall_s=time.perf_counter()-start,native_execution=False,goal_achieved=False))
        print('RECEIPT_COPIED_ANCHORED_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_RECEIPT_COPIED_ANCHORED_PREFIX_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
