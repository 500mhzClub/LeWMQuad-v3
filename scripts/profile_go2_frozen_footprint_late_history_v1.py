"""Rebuild the original causal history and profile fixed late controller work.

The development tape contains the original visibility failure at frame 1173.
Preserving its decisions establishes neither sensing validity nor navigation.
"""
import argparse
from collections import deque
import cProfile
from itertools import islice
import json
import os
import pstats
import time
import cv2
import numpy as np
import torch

from scripts import profile_go2_frozen_footprint_controller_windows_v1 as previous
from scripts.profile_go2_frozen_footprint_controller_windows_v1 import (
    FrozenFootprintAnchoredController, normalize_candidate, public_mission, NAVIGATION_TICKS,
    IntentReturnRGBDReplay, ArticulatedCollisionGeometry, state_digest, reference,
    read_rows, packet, public_acquisition, fingerprint, profile_summary, resources_for)
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from lewm.independent_reactive_floor_transport_study_development import merge_sources

OUTPUT = BASE/'go2_frozen_footprint_late_history_profile_v1_attempt_001'
SOURCE = 'scripts/profile_go2_frozen_footprint_late_history_v1.py'
TEST = 'lewm/tests/test_frozen_footprint_late_history_profile_development.py'
PROTOCOL = 'docs/go2_frozen_footprint_late_history_profile_v1_2026-09-10.md'
WORKER_SHA = '617056f19ba4928aa9ff7738616947e6e63a387cc6046353e30617ce50afa57e'
PROFILE_SHA = 'c636eb55c13f02624b73680295ab3f70d7faac00680d7e66dd820b870cfb9866'
TIMING = 'docs/go2_completed_adapter_native_timing_2026-09-10.json'
TIMING_SHA = '6d5e21e96d916cc3d0ab7bee540121b25e06f230e4b630a307ae665bca4d2d82'
TIMING_CHECK = 'docs/go2_completed_adapter_native_timing_verification_2026-09-10.json'
TIMING_CHECK_SHA = '4273c250cc3df21835dfae4dec15318cdf6a65c716a7244489265fef30b0913e'
WINDOWS = {'early_navigation':(3,12), 'repeated_hold':(395,404), 'late_navigation':(1418,1427)}
FRAMES = 1428


def prepared_sources():
    verify_artifacts(previous.OUTPUT, {'result.json':PROFILE_SHA})
    old = read_json(previous.OUTPUT, 'result.json')
    if old['status'] != 'FROZEN_FOOTPRINT_CONTROLLER_WINDOWS_PROFILE_V1_COMPLETE':
        raise ValueError('complete original optimized-controller profile required')
    verify_artifacts(previous.OUTPUT, old['artifact_sha256']); verify(old['source_sha256'])
    verify({TIMING:TIMING_SHA, TIMING_CHECK:TIMING_CHECK_SHA})
    timing = json.loads((ROOT/TIMING).read_text()); check = json.loads((ROOT/TIMING_CHECK).read_text())
    if (check['result_sha256'] != TIMING_SHA or check['source_sha256'] != timing['source_sha256']
            or check['observations'] != 4543 or check['statistics_groups'] != 464):
        raise ValueError('complete independently checked full-history timing diagnosis required')
    inherited = merge_sources(old['source_sha256'], timing['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL, TIMING, TIMING_CHECK,
        'docs/go2_completed_adapter_native_timing_result_2026-09-10.md'), inherited)
    verify(sources)
    return sources


def sensing_scope():
    root = reference.original.OUTPUT; terminal = reference.CASE[0]+'_worker_terminal.json'
    verify_artifacts(root, {terminal:WORKER_SHA})
    record = read_json(root, terminal)
    if (record['strict_physical_visibility_pass'] is not False
            or record['hard_measurement_failed_frames'] != [1173]
            or record['verified_round_trip'] is not False):
        raise ValueError('exact original negative navigation and visibility evidence required')
    return dict(original_strict_physical_visibility_pass=False,
        original_hard_measurement_failed_frames=[1173], failure_frame_inside_profiled_history=True,
        original_verified_round_trip=False, known_invalid_sensing_retained=True,
        qualified_sensing_prefix_claimed=False, navigation_verified=False)


def state_sizes(controller):
    """Read only bounded-depth container/array sizes; no state equality claim."""
    def describe(value, depth):
        row = dict(type=type(value).__module__+'.'+type(value).__name__)
        if isinstance(value, np.ndarray): row.update(shape=list(value.shape), bytes=value.nbytes)
        elif isinstance(value, torch.Tensor): row.update(shape=list(value.shape), bytes=value.nelement()*value.element_size())
        elif isinstance(value, (dict,list,tuple,set,deque)): row['entries'] = len(value)
        elif depth and hasattr(value, '__dict__'):
            row['fields'] = {name:describe(child, depth-1) for name,child in vars(value).items()}
        return row
    return {name:describe(getattr(controller,name),2) for name in ('memory','mapper','history','residual')}


def replay():
    original = reference.original; case = reference.CASE
    model = original.assigned_model(read_json(original.OUTPUT, 'launch.json'), case)
    if state_digest(model.state_dict()) != reference.MODEL_SHA:
        raise ValueError('unchanged assigned JEPA model required')
    controller = FrozenFootprintAnchoredController(model, ArticulatedCollisionGeometry(reference.URDF),
        public_mission=public_mission(case[1]), navigation_ticks=NAVIGATION_TICKS,
        condition=case[3], variant=case[2], persistent=True)
    directory = original.OUTPUT/case[0]; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    profiles = {name:cProfile.Profile() for name in WINDOWS}; captured = {name:[] for name in WINDOWS}
    count = forecasts = 0; sizes = {}
    with (OUTPUT/'comparison.jsonl').open('x') as stream:
        for row in islice(read_rows(directory), FRAMES):
            frame = row['tick']
            if (frame != count or tape[frame]['tick'] != frame or tape[frame]['completed'] is not True
                    or tape[frame]['pre_sample_index'] != 749+50*frame or tape[frame]['post_sample_index'] != 799+50*frame):
                raise ValueError('ordered completed original command endpoints required')
            p,d,fast,now = reader.packet(frame)
            image,aux = packet(directory,frame,p,public_acquisition(acquisitions[frame]),now_ns=now)
            before = fingerprint((p,d,fast,aux,image,now))
            window = next((n for n,(first,last) in WINDOWS.items() if first <= frame <= last), None)
            profiler = profiles[window] if window else None
            start = time.perf_counter()
            if profiler is not None: profiler.enable()
            try: actual = controller.observe(p,d,fast,now_ns=now,auxiliary_depth=aux,auxiliary_rgb=image)
            finally:
                if profiler is not None: profiler.disable()
            elapsed = time.perf_counter()-start
            actual = json.loads(json.dumps(actual))
            candidate_sha = reference.saved.identity(actual)
            actual = normalize_candidate(actual)
            if (actual != row['decision'] or actual['requested_command'] != tape[frame]['requested_command']
                    or actual['terminal'] is not None or before != fingerprint((p,d,fast,aux,image,now))):
                raise ValueError('complete original decision or public inputs changed at frame '+str(frame))
            selection = actual['new_selection']; action = None if selection is None else selection['action']
            if window == 'repeated_hold' and (action != 'hold' or actual['mission_receipt']['hold_required']):
                raise ValueError('original discretionary hold window required')
            if window:
                captured[window].append(dict(frame=frame, action=action, controller_wall_s_with_profiling=elapsed))
                if frame == WINDOWS[window][1]: sizes[window] = state_sizes(controller)
            stream.write(json.dumps(dict(frame=frame, original_decision_sha256=reference.saved.identity(actual),
                candidate_decision_sha256=candidate_sha, candidate_normalized_decision_exact=True,
                public_input_sha256=before, complete_original_decision_reconstructed=True,
                public_input_arrays_unchanged=True, profiled_window=window,
                controller_wall_s=elapsed), allow_nan=False)+'\n')
            count += 1; forecasts += int(bool(selection and 'prediction' in selection))
            if frame % 50 == 0: print('LATE_HISTORY_CONTROLLER_PROFILE_FRAME',frame,flush=True)
    if count != FRAMES or forecasts != FRAMES-3:
        raise ValueError('complete 1428-observation history and 1425 forecasts required')
    if state_digest(model.state_dict()) != reference.MODEL_SHA or any(p.grad is not None for p in model.parameters()):
        raise ValueError('profiled model state or gradients changed')
    summaries = {}
    for name, profiler in profiles.items():
        first,last = WINDOWS[name]
        if [r['frame'] for r in captured[name]] != list(range(first,last+1)):
            raise ValueError('all three fixed ten-observation windows required')
        summary = profile_summary(pstats.Stats(profiler).stats)
        profiler.dump_stats(str(OUTPUT/(name+'.prof'))); write_json(OUTPUT/(name+'.json'), summary)
        summaries[name] = dict(observations=captured[name], total_exclusive_profiled_s=summary['total_exclusive_profiled_s'],
            top_functions_by_cumulative_time=summary['functions'][:15],
            top_modules_by_exclusive_time=summary['modules_by_exclusive_time'][:15])
    return dict(frames=count, raw_model_forecast_comparisons=forecasts, windows=summaries,
        state_size_snapshots=sizes, retained_state_identity_established=False,
        complete_original_decisions_reconstructed=True, model_state_sha256=reference.MODEL_SHA,
        model_state_unchanged=True, sensor_acquisition_profiled=False, controller_observe_only_profiled=True,
        last_replayed_observation=FRAMES-1, no_observation_1428_consumed=True,
        profiler_overhead_removed=False, isolated_benchmark=False, speedup_established=False,
        native_execution=False, policy_changed=False, real_time_qualified=False, navigation_qualified=False,
        invocation_frozen_footprint_receipts=True, normalization_outside_profiled_region=True,
        complete_normalized_candidate_decisions_exact=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    if any(os.environ.get(k) != v for k,v in dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0').items()):
        raise ValueError('fixed thread counts and Python hash seed required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive late-history profile; no retry or resume')
    sources = prepared_sources(); scope = sensing_scope()
    resources = reference.hardware(); resources_for(resources)
    if args.source_preflight_only:
        print('LATE_HISTORY_PROFILE_SOURCE_PREFLIGHT_PASS',len(sources),json.dumps(resources),flush=True); return
    print('LATE_HISTORY_PROFILE_FULL_INPUT_ADMISSION_STARTED',len(sources),flush=True)
    admission = reference.admit_worker(WORKER_SHA, sources)
    resources = reference.hardware(); resources_for(resources); verify(sources)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_admission=admission, sensing_scope=scope,
        original_profile_result_sha256=PROFILE_SHA, timing_diagnosis_sha256=TIMING_SHA,
        model_state_sha256=reference.MODEL_SHA, windows=WINDOWS, frames=FRAMES, hardware=resources,
        native_execution=False, policy_changed=False, model_training=False,
        invocation_frozen_footprint_receipts=True, normalization_outside_profiled_region=True))
    print('LATE_HISTORY_PROFILE_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = replay(); verify(sources)
        if reference.admit_worker(WORKER_SHA, sources) != admission or sensing_scope() != scope:
            raise ValueError('complete original input or retained failure evidence changed')
        names = ['launch.json','comparison.jsonl']+[name+suffix for name in WINDOWS for suffix in ('.prof','.json')]
        ids = {n:digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='FROZEN_FOOTPRINT_LATE_HISTORY_PROFILE_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, sensing_scope=scope,
            wall_s=time.perf_counter()-start, native_execution=False, goal_achieved=False))
        print('LATE_HISTORY_PROFILE_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_LATE_HISTORY_PROFILE_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
