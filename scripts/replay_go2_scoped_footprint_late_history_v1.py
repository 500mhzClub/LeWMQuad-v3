"""Paired incremental footprint-reuse replay against the frozen-footprint baseline."""
import argparse
from itertools import islice
import json
import math
import os
from pathlib import Path
import statistics
import time

import cv2
import psutil
import torch

from lewm.frozen_footprint_anchored_controller_development import FrozenFootprintAnchoredController
from lewm.scoped_footprint_anchored_controller_development import ScopedFootprintAnchoredController, CONTROLLER, FLAG
from scripts import profile_go2_frozen_footprint_late_history_v1 as profile
from scripts.replay_go2_residual_current_observation_planning_prefix_v1 import state_tree
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT = BASE/'go2_scoped_footprint_late_history_v1_attempt_001'
SOURCE = 'scripts/replay_go2_scoped_footprint_late_history_v1.py'
TEST = 'lewm/tests/test_scoped_footprint_late_history_replay_development.py'
PROTOCOL = 'docs/go2_scoped_footprint_late_history_replay_v1_2026-09-10.md'
PREPARATION = 'docs/go2_scoped_footprint_reuse_preparation_2026-09-10.json'
PREPARATION_SHA = 'bde0ec83c546cdd212db3b47b542d259a189aa5ab18971af271b242389c2992f'
PROFILE_LAUNCH_SHA = '0a87515a70adc3ee27039d01ec70a79809329cea77afcd2baae49c327a8599b3'
PROFILE_PID = 2738041
PROFILE_CREATED = 1789069525.94
PROFILE_ARGV = ['.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', profile.SOURCE]
BOOT = '1264d80f-6e46-4fcd-b2fd-2a5d7b964c73'
FRAMES = 1428
STATE_FRAMES = (3, 12, 395, 404, 1173, 1418, 1427)
WINDOWS = {'all_navigation': (3, 1427), **profile.WINDOWS}


def prepared_sources():
    verify({PREPARATION: PREPARATION_SHA})
    preparation = json.loads((ROOT/PREPARATION).read_text())
    if preparation['status'] != 'SCOPED_FOOTPRINT_REUSE_SOURCE_CHECKED_NOT_RAW_REPLAYED':
        raise ValueError('checked exact reuse implementation required')
    verify(preparation['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION,
        'docs/go2_scoped_footprint_reuse_result_2026-09-10.md'), preparation['source_sha256'])
    verify(sources)
    return sources


def profile_owner_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise ValueError('original profile boot identity required')
    try:
        process = psutil.Process(PROFILE_PID)
        if process.create_time() != PROFILE_CREATED or process.cmdline() != PROFILE_ARGV:
            raise ValueError('profile PID identity changed; do not infer original completion')
    except psutil.NoSuchProcess:
        return
    raise ValueError('original late-history profiler remains live')


def completed_profile(result_sha, sources):
    profile_owner_ended()
    if (profile.OUTPUT/'failure.json').exists():
        raise ValueError('original profile terminal failure cannot be replaced by completion')
    verify_artifacts(profile.OUTPUT, {'launch.json': PROFILE_LAUNCH_SHA, 'result.json': result_sha})
    result = read_json(profile.OUTPUT, 'result.json')
    launch = read_json(profile.OUTPUT, 'launch.json')
    expected = {'launch.json', 'comparison.jsonl'} | {n+s for n in profile.WINDOWS for s in ('.prof', '.json')}
    if (result['status'] != 'FROZEN_FOOTPRINT_LATE_HISTORY_PROFILE_V1_COMPLETE'
            or set(result['artifact_sha256']) != expected
            or result['artifact_sha256']['launch.json'] != PROFILE_LAUNCH_SHA
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())):
        raise ValueError('complete exact original late-history profile required')
    verify(result['source_sha256']); verify_artifacts(profile.OUTPUT, result['artifact_sha256'])
    report = result['report']
    if (report['frames'] != FRAMES or report['raw_model_forecast_comparisons'] != FRAMES-3
            or report['model_state_sha256'] != profile.reference.MODEL_SHA
            or report['no_observation_1428_consumed'] is not True
            or report['complete_normalized_candidate_decisions_exact'] is not True
            or report['model_state_unchanged'] is not True
            or result['sensing_scope'] != profile.sensing_scope()):
        raise ValueError('complete original forecasts and preserved negative sensing scope required')
    rows = [json.loads(line) for line in (profile.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    if len(rows) != FRAMES:
        raise ValueError('complete profiled input/decision identity population required')
    for i,row in enumerate(rows):
        if row['frame'] != i or any(row[k] is not True for k in (
                'candidate_normalized_decision_exact', 'complete_original_decision_reconstructed',
                'public_input_arrays_unchanged')):
            raise ValueError('complete ordered successful reference comparison rows required')
    return rows


def resources_for(resources):
    if (resources['memory_available_bytes'] < 64*1024**3
            or resources['artifact_free_bytes'] < 41*1024**3 or resources['physical_cpus'] < 4):
        raise ValueError('64GiB available RAM, 40+1GiB disk and four physical CPUs required')


def normalize_candidate(decision):
    if decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit selection-scoped footprint reuse implementation required')
    result = decision.copy(); result.pop(FLAG)
    result['controller'] = 'residual_anchored_continuation_controller_v1'
    return result


def execution_order(frame):
    if type(frame) is not int or not 0 <= frame < FRAMES:
        raise ValueError('fixed ordered paired frame required')
    return (0, 1) if frame % 2 == 0 else (1, 0)


def timing_summary(rows):
    if [r['frame'] for r in rows] != list(range(FRAMES)):
        raise ValueError('complete paired timing population required')
    for row in rows:
        if tuple(row['execution_order']) != execution_order(row['frame']):
            raise ValueError('fixed alternating execution order required')
        for key in ('baseline_controller_s', 'candidate_controller_s'):
            if type(row[key]) not in (float, int) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError('finite positive paired controller times required')
    result = {}
    for name,(first,last) in WINDOWS.items():
        baseline = [r['baseline_controller_s'] for r in rows[first:last+1]]
        candidate = [r['candidate_controller_s'] for r in rows[first:last+1]]
        result[name] = dict(first_frame=first, last_frame=last, observations=len(baseline),
            baseline_median_s=statistics.median(baseline), candidate_median_s=statistics.median(candidate),
            baseline_total_s=math.fsum(baseline), candidate_total_s=math.fsum(candidate),
            total_ratio=math.fsum(baseline)/math.fsum(candidate),
            baseline_over_100ms=sum(t > .1 for t in baseline), candidate_over_100ms=sum(t > .1 for t in candidate))
    return result


def replay(reference_rows):
    reference = profile.reference; original = reference.original; case = reference.CASE
    models = [original.assigned_model(read_json(original.OUTPUT, 'launch.json'), case) for _ in range(2)]
    if any(profile.state_digest(m.state_dict()) != reference.MODEL_SHA for m in models):
        raise ValueError('two fresh unchanged assigned models required')
    # Include parameters and buffers, not only paired parameter indices.
    storage = [{v.untyped_storage().data_ptr() for v in m.state_dict().values() if v.numel()} for m in models]
    if models[0] is models[1] or not storage[0].isdisjoint(storage[1]):
        raise ValueError('independent model tensor storage required')
    options = dict(public_mission=profile.public_mission(case[1]), navigation_ticks=profile.NAVIGATION_TICKS,
        condition=case[3], variant=case[2], persistent=True)
    geometry = profile.ArticulatedCollisionGeometry(reference.URDF)
    controllers = (FrozenFootprintAnchoredController(models[0], geometry, **options),
                   ScopedFootprintAnchoredController(models[1], geometry, **options))
    directory = original.OUTPUT/case[0]; reader = profile.IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    if len(reference_rows) != FRAMES:
        raise ValueError('complete original profiled history required')
    timings = []; states = []; forecasts = 0
    with (OUTPUT/'comparison.jsonl').open('x') as stream:
        for row in islice(profile.read_rows(directory), FRAMES):
            frame = row['tick']
            if (frame != len(timings) or tape[frame]['tick'] != frame or tape[frame]['completed'] is not True
                    or tape[frame]['pre_sample_index'] != 749+50*frame or tape[frame]['post_sample_index'] != 799+50*frame):
                raise ValueError('ordered exact original command endpoints required')
            p,d,fast,now = reader.packet(frame)
            image,aux = profile.packet(directory, frame, p, profile.public_acquisition(acquisitions[frame]), now_ns=now)
            before = profile.fingerprint((p,d,fast,aux,image,now))
            if before != reference_rows[frame]['public_input_sha256']:
                raise ValueError('same original raw public packets required')
            decisions = [None, None]; elapsed = [None, None]; order = execution_order(frame)
            for index in order:
                start = time.perf_counter()
                decisions[index] = controllers[index].observe(p,d,fast,now_ns=now,auxiliary_depth=aux,auxiliary_rgb=image)
                elapsed[index] = time.perf_counter()-start
                if profile.fingerprint((p,d,fast,aux,image,now)) != before:
                    raise ValueError('each controller must preserve public input arrays')
            baseline, candidate = [json.loads(json.dumps(value)) for value in decisions]
            normalized = profile.normalize_candidate(baseline)
            if (normalized != row['decision'] or normalize_candidate(candidate) != normalized
                    or normalized['terminal'] is not None
                    or normalized['requested_command'] != tape[frame]['requested_command']
                    or reference.saved.identity(normalized) != reference_rows[frame]['original_decision_sha256']
                    or reference.saved.identity(baseline) != reference_rows[frame]['candidate_decision_sha256']):
                raise ValueError('complete original, baseline and candidate decisions must agree at '+str(frame))
            selection = normalized['new_selection']; forecasts += int(bool(selection and 'prediction' in selection))
            record = dict(frame=frame, execution_order=list(order), baseline_controller_s=elapsed[0],
                candidate_controller_s=elapsed[1], public_input_sha256=before,
                original_decision_sha256=reference.saved.identity(normalized),
                baseline_decision_sha256=reference.saved.identity(baseline),
                candidate_decision_sha256=reference.saved.identity(candidate),
                complete_original_decision_reconstructed=True, candidate_normalized_decision_exact=True,
                public_input_arrays_unchanged=True)
            stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush(); timings.append(record)
            if frame in STATE_FRAMES:
                # These objects have identical types; no hidden-state type
                # normalization is allowed. Selector class identity is omitted.
                hashes = [profile.fingerprint(state_tree(dict(memory=c.memory, floor=c.mapper.floor,
                    occupied=c.mapper.occupied, residual=c.residual, history=c.history))) for c in controllers]
                if hashes[0] != hashes[1]:
                    raise ValueError('retained observed memory/map/residual/history state changed')
                states.append(dict(frame=frame, state_sha256=hashes[0], retained_observed_state_equal=True))
            if frame % 50 == 0: print('SCOPED_FOOTPRINT_PAIRED_FRAME',frame,flush=True)
    if len(timings) != FRAMES or forecasts != FRAMES-3 or [r['frame'] for r in states] != list(STATE_FRAMES):
        raise ValueError('complete fixed paired history, forecasts and state checks required')
    if any(profile.state_digest(m.state_dict()) != reference.MODEL_SHA
            or any(p.grad is not None for p in m.parameters()) for m in models):
        raise ValueError('model weights or gradients changed')
    return dict(frames=FRAMES, raw_model_forecast_comparisons=forecasts, observed_state_checks=states,
        state_scope=['memory', 'mapper.floor', 'mapper.occupied', 'residual', 'history'], normalized_state_type_paths=[],
        model_state_sha256=reference.MODEL_SHA, model_state_unchanged=True,
        complete_original_decisions_reconstructed=True, complete_normalized_candidate_decisions_exact=True,
        public_input_arrays_unchanged=True, timing_windows=timing_summary(timings),
        baseline='FrozenFootprintAnchoredController', candidate='ScopedFootprintAnchoredController',
        incremental_reuse_comparison=True, alternating_execution_order=True, profiling_enabled=False,
        controller_observe_only_timed=True, sensor_acquisition_timed=False, isolated_benchmark=False,
        no_observation_1428_consumed=True, native_execution=False, real_time_qualified=False, navigation_qualified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-preflight-only', action='store_true')
    parser.add_argument('--profile-result-sha256')
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    if any(os.environ.get(k) != v for k,v in dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0').items()):
        raise ValueError('fixed single-thread settings and hash seed required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive paired replay; no retry or resume')
    sources = prepared_sources()
    resources = profile.reference.hardware(); resources_for(resources)
    if args.source_preflight_only:
        print('SCOPED_FOOTPRINT_PAIRED_SOURCE_PREFLIGHT_PASS',len(sources),flush=True); return
    if not args.profile_result_sha256: raise ValueError('completed late-history profile SHA required')
    reference_rows = completed_profile(args.profile_result_sha256,sources)
    scope = profile.sensing_scope()
    print('SCOPED_FOOTPRINT_PAIRED_FULL_INPUT_ADMISSION_STARTED',flush=True)
    admission = profile.reference.admit_worker(profile.WORKER_SHA,sources)
    resources = profile.reference.hardware(); resources_for(resources); verify(sources); profile_owner_ended()
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources, input_admission=admission,
        profile_result_sha256=args.profile_result_sha256, profile_launch_sha256=PROFILE_LAUNCH_SHA,
        sensing_scope=scope, hardware=resources, frames=FRAMES, state_frames=STATE_FRAMES,
        baseline='FrozenFootprintAnchoredController', candidate='ScopedFootprintAnchoredController',
        incremental_reuse_comparison=True, native_execution=False, model_training=False))
    print('SCOPED_FOOTPRINT_PAIRED_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = replay(reference_rows); verify(sources)
        if (profile.reference.admit_worker(profile.WORKER_SHA,sources) != admission
                or completed_profile(args.profile_result_sha256,sources) != reference_rows
                or profile.sensing_scope() != scope):
            raise ValueError('original complete input admission or reference history changed')
        ids = {n:digest(OUTPUT/n) for n in ('launch.json','comparison.jsonl')}; verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='SCOPED_FOOTPRINT_LATE_HISTORY_PAIRED_REPLAY_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, sensing_scope=scope,
            wall_s=time.perf_counter()-start, native_execution=False, goal_achieved=False))
        print('SCOPED_FOOTPRINT_PAIRED_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SCOPED_FOOTPRINT_PAIRED_REPLAY_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__':
    main()
