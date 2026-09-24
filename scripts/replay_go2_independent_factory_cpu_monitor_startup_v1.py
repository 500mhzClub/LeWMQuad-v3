"""Four old packets per fixed arm under the CPU monitor; no new commands."""
import argparse
import json
import os
import time
import traceback
from types import FunctionType

import cv2
import torch
from scripts import replay_go2_independent_adapter_factory_startup_v1 as original
from scripts.independent_round_trip_audit_cpu_monitor_development import AuditCPUMonitor
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.maze_decision_stream_development import read_rows, NAME
from lewm.independent_round_trip_comparison_study_development import CASES, require_case

OUTPUT = BASE/'go2_independent_factory_cpu_monitor_startup_v1_attempt_001'
SOURCE = 'scripts/replay_go2_independent_factory_cpu_monitor_startup_v1.py'
TEST = 'lewm/tests/test_independent_factory_cpu_monitor_startup_development.py'
PROTOCOL = 'docs/go2_independent_factory_cpu_monitor_startup_v1_2026-09-11.md'
PREPARATION = 'docs/go2_independent_round_trip_audit_cpu_monitor_preparation_2026-09-11.json'
PREPARATION_SHA = '4b7877693272a42910f5b79d06d58b9e1a468ef7a6aad68dadf78e6b743ab97f'
REFERENCE_SHA = 'cd0af02bdbe978e714e55c1c0253f1fa01df88c4554eeea887deae84e40ae6db'
REFERENCE_LAUNCH_SHA = '2cbd89c0be82ddb26957393c9c299c62b1442c59dbfb8284fdabf1ae62aac9d6'


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']; verify(inherited)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), inherited); verify(sources)
    return sources


def require_reference(result, launch, sources):
    if (result['status'] != 'INDEPENDENT_ADAPTER_FACTORY_STARTUP_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or launch['original_result_sha256'] != original.evidence.ORIGINAL_SHA
            or launch['original_raw_startup_case'] != original.INPUT_CASE
            or launch['planned_cases'] != [vars(c) for c in CASES[:4]]
            or set(result['artifact_sha256']) != {'launch.json', *(case.arm_name+'/'+NAME for case in CASES[:4])}
            or result['artifact_sha256']['launch.json'] != REFERENCE_LAUNCH_SHA):
        raise ValueError('complete original four-arm factory startup required')
    for key in ('native_execution', 'model_training', 'new_layout_sensor_data_consumed',
            'new_layout_navigation_execution', 'goal_achieved'):
        if result[key] is not False: raise ValueError('original bounded startup scope required')
    if len(result['reports']) != 4: raise ValueError('all four original reports required')
    for case, report in zip(CASES[:4], result['reports'], strict=True):
        arm = require_case(case)
        if (report['arm'] != case.arm_name or report['assigned_case'] != case.name
                or report['frames'] != 4 or report['model_state_sha256'] != arm.model_state_sha256
                or report['model_state_unchanged'] is not True
                or report['no_packet_after_new_planning_command_consumed'] is not True
                or report['command_executed'] is not False
                or report['independent_layout_sensor_data_consumed'] is not False):
            raise ValueError('same original assigned startup reports required')


def admit_reference(sources):
    verify_artifacts(original.OUTPUT, {'result.json':REFERENCE_SHA,'launch.json':REFERENCE_LAUNCH_SHA})
    result = read_json(original.OUTPUT, 'result.json'); launch = read_json(original.OUTPUT, 'launch.json')
    require_reference(result, launch, sources)
    verify(result['source_sha256']); verify_artifacts(original.OUTPUT, result['artifact_sha256'])
    verify_artifacts(original.original.INPUT, launch['original_artifact_sha256'])
    input_launch = read_json(original.original.INPUT, 'launch.json')
    verify_artifacts(original.original.INPUT, {'launch.json':original.evidence.ORIGINAL_LAUNCH})
    original.verify_bound_inputs(input_launch['input_admission'], sources)
    rows = {case.arm_name:list(read_rows(original.OUTPUT/case.arm_name)) for case in CASES[:4]}
    if any(len(value) != 4 or [r['tick'] for r in value] != list(range(4)) for value in rows.values()):
        raise ValueError('exact four-frame reference streams required')
    return result, launch, input_launch, rows


def isolated_case():
    if original.run_case.__closure__ is not None: raise ValueError('closure-free original startup body required')
    return FunctionType(original.run_case.__code__, dict(original.run_case.__globals__, OUTPUT=OUTPUT),
        name=original.run_case.__name__)


def monitored_case(case, admission, expected_report, expected_rows):
    monitor = AuditCPUMonitor(); started = time.perf_counter()
    try:
        with monitor: report = isolated_case()(case, admission)
    finally:
        if monitor.closed:
            write_json(OUTPUT/(case.arm_name+'_cpu_monitor.json'), monitor.summary())
    measured = monitor.summary()
    if (not measured['scope_returned_without_error'] or measured['violations']
            or measured['initial_opencl_enabled'] is not False
            or (case.arm_name != 'reactive' and measured['torch_operation_count'] <= 0)):
        raise ValueError('complete monitored CPU factory execution with disabled OpenCL required')
    rows = list(read_rows(OUTPUT/case.arm_name))
    if report != expected_report or rows != expected_rows:
        raise ValueError('CPU monitoring or explicit device settings changed complete saved startup evidence')
    return dict(arm=case.arm_name, original_report=report, cpu_monitor=measured,
        complete_saved_startup_decisions_exact=True, frames=4, wall_s=time.perf_counter()-started,
        full_raw_auditor_qualified=False, overlap_execution_permitted=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('original audit assertions must remain enabled')
    expected = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
    if any(os.environ.get(k) != v for k,v in expected.items()) or cv2.ocl.useOpenCL():
        raise ValueError('fixed CPU settings and OpenCL disabled before interpreter import required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive monitored startup; no retry or resume')
    sources = prepared_sources(); resources = original.hardware()
    if resources['memory_available_bytes'] < 32*1024**3 or resources['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('bounded startup resource allowance unavailable')
    if args.source_preflight_only:
        print('INDEPENDENT_FACTORY_CPU_MONITOR_SOURCE_PREFLIGHT_PASS', len(sources), flush=True); return
    reference, old_launch, input_launch, rows = admit_reference(sources)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, reference_result_sha256=REFERENCE_SHA,
        reference_launch_sha256=REFERENCE_LAUNCH_SHA, original_artifact_sha256=old_launch['original_artifact_sha256'],
        planned_cases=[vars(c) for c in CASES[:4]], hardware=resources, environment=expected,
        original_startup_function_body_reused=True, imported_module_globals_mutated=False,
        packets_per_arm=4, native_execution=False, full_raw_auditor_qualification_claimed=False))
    print('INDEPENDENT_FACTORY_CPU_MONITOR_STARTUP_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    reports = []; active = None
    try:
        for case, expected_report in zip(CASES[:4], reference['reports'], strict=True):
            active = case.arm_name
            reports.append(monitored_case(case, input_launch['input_admission']['correction_admission'],
                expected_report, rows[case.arm_name]))
            print('INDEPENDENT_FACTORY_CPU_MONITOR_ARM_COMPLETE', active, flush=True)
        if admit_reference(sources) != (reference, old_launch, input_launch, rows):
            raise ValueError('original reference or input admission changed')
        verify(sources)
        names = ['launch.json']
        for case in CASES[:4]: names.extend([case.arm_name+'/'+NAME, case.arm_name+'_cpu_monitor.json'])
        ids = {n:digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, ids)
        result = dict(status='INDEPENDENT_FACTORY_CPU_MONITOR_STARTUP_V1_COMPLETE', source_sha256=sources,
            artifact_sha256=ids, reports=reports, reference_result_sha256=REFERENCE_SHA,
            original_startup_outputs_unchanged=True, full_raw_auditor_qualified=False,
            overlap_execution_permitted=False, native_execution=False,
            independent_layout_sensor_data_consumed=False, command_executed=False, goal_achieved=False)
        write_json(OUTPUT/'result.json', result)
        print('INDEPENDENT_FACTORY_CPU_MONITOR_STARTUP_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_INDEPENDENT_FACTORY_CPU_MONITOR_STARTUP_FAILURE',
            active_arm=active, reports=reports, reason=repr(error), traceback=traceback.format_exc(),
            automatic_retry=False, evidence_preserved=True, native_execution=False))
        raise


if __name__ == '__main__': main()
