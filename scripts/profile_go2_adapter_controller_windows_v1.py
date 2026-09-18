"""Profile fixed controller windows while exactly replaying an audited prefix."""
import argparse
import cProfile
from itertools import islice
import json
import math
from pathlib import Path
import pstats
import time
import cv2
import torch
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import replay_go2_hold_reorientation_maze02_prefix_v1 as reference
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts, _ordinary
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT = BASE/'go2_adapter_controller_windows_profile_v1_attempt_001'
SOURCE = 'scripts/profile_go2_adapter_controller_windows_v1.py'
PROTOCOL = 'docs/go2_adapter_controller_windows_profile_v1_2026-09-10.md'
TEST = 'lewm/tests/test_adapter_controller_profile_development.py'
OWNER_ROOT = BASE/'go2_hold_reorientation_maze02_native_wait_v1_attempt_001'
OWNER_SHA = 'e27675df102b072b62f4351483363ab9ee80e9e0244e1193c398392f716a47f6'
WORKER_SHA = '617056f19ba4928aa9ff7738616947e6e63a387cc6046353e30617ce50afa57e'
WINDOWS = {'early_navigation': (3, 12), 'repeated_hold': (395, 404)}
FRAMES = 405


def profile_summary(stats):
    functions = []; modules = {}
    for (filename, line, function), (primitive_calls, calls, own, cumulative, _) in stats.items():
        _ordinary(Path(filename))
        if (type(line) is not int or type(primitive_calls) is not int or type(calls) is not int
                or not 0 <= primitive_calls <= calls or not math.isfinite(own) or not math.isfinite(cumulative)
                or own < 0 or cumulative < 0):
            raise ValueError('finite nonnegative profiler counts and times required')
        functions.append(dict(filename=filename, line=line, function=function, primitive_calls=primitive_calls,
            calls=calls, self_time_s=own, cumulative_time_s=cumulative))
        modules[filename] = modules.get(filename, 0.) + own
    if not functions: raise ValueError('nonempty fixed-window profile required')
    return dict(functions=sorted(functions, key=lambda r: (-r['cumulative_time_s'], r['filename'], r['line'], r['function'])),
        modules_by_exclusive_time=[dict(filename=n, self_time_s=t) for n,t in sorted(modules.items(), key=lambda r: (-r[1],r[0]))],
        total_exclusive_profiled_s=sum(r['self_time_s'] for r in functions),
        cumulative_times_overlap=True, cumulative_times_must_not_be_summed=True,
        profiler_overhead_removed=False, isolated_benchmark=False, real_time_qualified=False)


def resources_for(resources):
    if (resources['memory_available_bytes'] < 48*1024**3
            or resources['artifact_free_bytes'] < 41*1024**3 or resources['physical_cpus'] < 4):
        raise ValueError('48GiB available RAM, 40+1GiB artifact envelope and four physical CPUs required')


def replay():
    original = reference.original; case = reference.CASE
    model = original.assigned_model(read_json(original.OUTPUT, 'launch.json'), case)
    if state_digest(model.state_dict()) != reference.MODEL_SHA: raise ValueError('unchanged assigned JEPA model required')
    controller = ResidualAnchoredContinuationController(model, ArticulatedCollisionGeometry(reference.URDF),
        public_mission=public_mission(case[1]), navigation_ticks=NAVIGATION_TICKS,
        condition=case[3], variant=case[2], persistent=True)
    directory = original.OUTPUT/case[0]; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    profiles = {name:cProfile.Profile() for name in WINDOWS}; captured = {name:[] for name in WINDOWS}
    count = forecasts = 0
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
            if (actual != row['decision'] or actual['requested_command'] != tape[frame]['requested_command']
                    or actual['terminal'] is not None or before != fingerprint((p,d,fast,aux,image,now))):
                raise ValueError('profiling must preserve complete original raw decisions and public arrays: '+str(frame))
            selection = actual['new_selection']; action = None if selection is None else selection['action']
            if window == 'repeated_hold' and (action != 'hold' or actual['mission_receipt']['hold_required']):
                raise ValueError('the fixed later window must reconstruct original discretionary holds')
            if window: captured[window].append(dict(frame=frame, action=action, controller_wall_s_with_profiling=elapsed))
            stream.write(json.dumps(dict(frame=frame, original_decision_sha256=reference.saved.identity(actual),
                public_input_sha256=before, complete_original_decision_reconstructed=True,
                public_input_arrays_unchanged=True, profiled_window=window,
                controller_wall_s=elapsed), allow_nan=False)+'\n')
            count += 1; forecasts += int(bool(selection and 'prediction' in selection))
            if frame % 50 == 0: print('ADAPTER_CONTROLLER_PROFILE_FRAME',frame,flush=True)
    if count != FRAMES or forecasts != 402: raise ValueError('complete 405-observation prefix required')
    if state_digest(model.state_dict()) != reference.MODEL_SHA or any(p.grad is not None for p in model.parameters()):
        raise ValueError('profiled model state or gradients changed')
    summaries = {}
    for name, profiler in profiles.items():
        first,last = WINDOWS[name]
        if [r['frame'] for r in captured[name]] != list(range(first,last+1)):
            raise ValueError('both fixed ten-observation windows required')
        summary = profile_summary(pstats.Stats(profiler).stats)
        profiler.dump_stats(str(OUTPUT/(name+'.prof')))
        write_json(OUTPUT/(name+'.json'), summary)
        summaries[name] = dict(observations=captured[name], total_exclusive_profiled_s=summary['total_exclusive_profiled_s'],
            top_functions_by_cumulative_time=summary['functions'][:15],
            top_modules_by_exclusive_time=summary['modules_by_exclusive_time'][:15])
    return dict(frames=count, raw_model_forecast_comparisons=forecasts, windows=summaries,
        complete_original_decisions_reconstructed=True, model_state_sha256=reference.MODEL_SHA,
        model_state_unchanged=True, sensor_acquisition_profiled=False, controller_observe_only_profiled=True,
        no_observation_405_consumed=True, profiler_overhead_removed=False, isolated_benchmark=False,
        native_execution=False, policy_changed=False, real_time_qualified=False, navigation_qualified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-preflight-only',action='store_true');args=parser.parse_args()
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive fixed-window profile required')
    verify_artifacts(OWNER_ROOT,{'launch.json':OWNER_SHA})
    sources=discover_sources((SOURCE,PROTOCOL,TEST),read_json(OWNER_ROOT,'launch.json')['source_sha256']);verify(sources)
    resources=reference.hardware();resources_for(resources)
    if args.source_preflight_only:
        print('ADAPTER_CONTROLLER_PROFILE_SOURCE_PREFLIGHT_PASS',len(sources),flush=True);return
    print('ADAPTER_CONTROLLER_PROFILE_INPUT_ADMISSION_STARTED',flush=True)
    admission=reference.admit_worker(WORKER_SHA,sources);resources_for(reference.hardware())
    verify(sources);create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_admission=admission,
        model_state_sha256=reference.MODEL_SHA,windows=WINDOWS,frames=FRAMES,hardware=resources,
        owner_launch_sha256=OWNER_SHA,native_execution=False,policy_changed=False,model_training=False))
    print('ADAPTER_CONTROLLER_PROFILE_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    try:
        report=replay();verify(sources)
        if reference.admit_worker(WORKER_SHA,sources)!=admission:raise ValueError('complete original admission changed')
        names=['launch.json','comparison.jsonl']+[name+suffix for name in WINDOWS for suffix in ('.prof','.json')]
        ids={n:digest(OUTPUT/n) for n in names};verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='ADAPTER_CONTROLLER_WINDOWS_PROFILE_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,wall_s=time.perf_counter()-start,
            native_execution=False,goal_achieved=False))
        print('ADAPTER_CONTROLLER_PROFILE_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ADAPTER_CONTROLLER_PROFILE_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
