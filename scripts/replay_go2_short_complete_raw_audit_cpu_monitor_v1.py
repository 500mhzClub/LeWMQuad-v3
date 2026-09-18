"""Replay one complete old failed episode under CPU instrumentation."""
import argparse
import contextlib
import json
import os
import time
import traceback

import cv2
import torch
from scripts import replay_go2_independent_factory_cpu_monitor_startup_v1 as startup
from scripts import run_go2_all_phase_residual_maze02_matched_native_v1 as original
from scripts.independent_round_trip_audit_cpu_monitor_development import AuditCPUMonitor
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE = 'scripts/replay_go2_short_complete_raw_audit_cpu_monitor_v1.py'
TEST = 'lewm/tests/test_short_complete_raw_audit_cpu_monitor_development.py'
PROTOCOL = 'docs/go2_short_complete_raw_audit_cpu_monitor_v1_2026-09-11.md'
PREPARATION = 'docs/go2_independent_factory_cpu_monitor_startup_preparation_2026-09-11.json'
PREPARATION_SHA = '8bc7e9561187c44b26cb76ea761f74ad3bd0b4c6ce00ac74122919e770a44b96'
OUTPUT = BASE/'go2_short_complete_raw_audit_cpu_monitor_v1_attempt_001'
REFERENCE_SHA = 'a08496e1d62ec6e00ffae3d85729cc7cf069c2fddcb60d421910c83d30556e80'
REFERENCE_LAUNCH_SHA = '8260991892f61dd36d75ac001e2198215b19a2531f535bfc381f86c6282c46eb'
CASE = ('all_phase_full_jepa_residual_maze_02', 2, 'full', 'jepa', 'seed_2026091001_full_jepa')


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']; verify(inherited)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), inherited)
    verify(sources)
    return sources


def require_episode(collection, expected):
    fixed = dict(layout_index=2, schedule_terminal='SENSOR_OR_MODEL_FAILURE',
        command_ticks=13, completed_ticks=13, physics_samples=1400, rgbd_frames=14,
        auxiliary_frames=14, decisions=14, terminal_zero_ticks=10)
    if any(type(collection.get(k)) is not type(v) or collection[k] != v for k,v in fixed.items()):
        raise ValueError('the complete original 14-observation interface-failure episode is required')
    if (expected['verified_round_trip'] is not False or expected['selected_actions'] != {}
            or len(expected['observation_and_control_wall_ms']) != 14
            or expected['raw_model_command_replay_pass'] is not True):
        raise ValueError('complete original negative raw-audit report required')


def admit(sources):
    # This authenticates the already completed reference and its bound inputs;
    # it does not repeat the entire training ancestry admission or any inference.
    startup.admit_reference(sources)
    verify_artifacts(original.OUTPUT, {'result.json':REFERENCE_SHA, 'launch.json':REFERENCE_LAUNCH_SHA})
    reference = read_json(original.OUTPUT, 'result.json')
    launch = read_json(original.OUTPUT, 'launch.json')
    if (reference['status'] != 'ALL_PHASE_RESIDUAL_MAZE02_MATCHED_NATIVE_V1_COMPLETE'
            or reference['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in reference['source_sha256'].items())):
        raise ValueError('unchanged completed original source-bound cohort required')
    verify_artifacts(original.OUTPUT, reference['artifact_sha256'])
    original.verify_inputs(launch)
    worker = read_json(original.OUTPUT, CASE[0]+'_worker_terminal.json')
    expected = read_json(original.OUTPUT, CASE[0]+'_audit.json')
    collection = read_json(original.OUTPUT, CASE[0]+'/result.json')
    if worker['status'] != original.WORKER_STATUS or worker['collection'] != collection:
        raise ValueError('original completed worker and raw collection must agree')
    original.require_case(CASE, worker, expected)
    require_episode(collection, expected)
    return launch, collection, expected


def run_original(launch, collection):
    model = original.assigned_model(launch, CASE)
    return original.audit(CASE[1], collection, launch['source_sha256'][original.PROTOCOL],
        input_root=original.OUTPUT, model=model,
        robot_geometry=original.ArticulatedCollisionGeometry(original.URDF),
        episode_name=CASE[0], condition=CASE[3], variant=CASE[2])


def monitored_audit(launch, collection, expected):
    require_episode(collection, expected)
    monitor = AuditCPUMonitor(); started = time.perf_counter()
    try:
        with monitor:
            report = run_original(launch, collection)
    finally:
        if monitor.closed: write_json(OUTPUT/'cpu_monitor.json', monitor.summary())
    measured = monitor.summary()
    write_json(OUTPUT/'audit.json', report)
    if (not measured['scope_returned_without_error'] or measured['violations']
            or measured['initial_opencl_enabled'] is not False
            or measured['torch_operation_count'] <= 0):
        raise ValueError('complete monitored CPU raw audit with OpenCL disabled required')
    if json.loads(json.dumps(report)) != expected:
        raise ValueError('complete original raw-audit report changed')
    return dict(complete_original_audit_report_exact=True, original_failed_navigation_preserved=True,
        observations=14, physics_samples=1400, command_intervals=13,
        model_loading_and_geometry_construction_monitored=True, wall_s=time.perf_counter()-started,
        cpu_monitor=measured, independent_multiarm_raw_audit_qualified=False,
        overlap_execution_permitted=False, command_executed=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('original raw-audit assertions must remain enabled')
    env = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
    if any(os.environ.get(k) != v for k,v in env.items()) or cv2.ocl.useOpenCL():
        raise ValueError('fixed CPU environment and OpenCL disabled before import required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive bounded audit; no retry or resume')
    sources = prepared_sources(); resources = original.hardware()
    if resources['memory_available_bytes'] < 32*1024**3 or resources['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('bounded audit resource allowance unavailable')
    if args.source_preflight_only:
        print('SHORT_COMPLETE_RAW_AUDIT_CPU_SOURCE_PREFLIGHT_PASS', len(sources), flush=True); return
    launch, collection, expected = admit(sources)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, reference_result_sha256=REFERENCE_SHA,
        reference_launch_sha256=REFERENCE_LAUNCH_SHA, case=list(CASE), environment=env, hardware=resources,
        original_auditor_called_directly=True, original_input_root=str(original.OUTPUT),
        native_execution=False, independent_layout_sensor_data_consumed=False))
    print('SHORT_COMPLETE_RAW_AUDIT_CPU_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    try:
        with (OUTPUT/'worker.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            report = monitored_audit(launch, collection, expected)
        if admit(sources) != (launch, collection, expected): raise ValueError('original raw inputs changed')
        verify(sources)
        ids = {n:digest(OUTPUT/n) for n in ('launch.json','worker.log','audit.json','cpu_monitor.json')}
        verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='SHORT_COMPLETE_RAW_AUDIT_CPU_MONITOR_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report,
            reference_result_sha256=REFERENCE_SHA, complete_original_raw_inputs_rehashed_before_and_after=True,
            independent_multiarm_raw_audit_qualified=False, overlap_execution_permitted=False,
            native_execution=False, independent_layout_sensor_data_consumed=False, goal_achieved=False))
        print('SHORT_COMPLETE_RAW_AUDIT_CPU_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        names = [n for n in ('launch.json','worker.log','audit.json','cpu_monitor.json') if (OUTPUT/n).is_file()]
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_SHORT_COMPLETE_RAW_AUDIT_CPU_FAILURE',
            reason=repr(error), traceback=traceback.format_exc(), artifact_sha256={n:digest(OUTPUT/n) for n in names},
            automatic_retry=False, evidence_preserved=True, native_execution=False))
        raise


if __name__ == '__main__': main()
