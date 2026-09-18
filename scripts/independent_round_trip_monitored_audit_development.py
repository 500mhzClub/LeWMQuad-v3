"""CPU-instrumented successor around the unchanged separate raw-audit body."""
import hashlib
import json
import os
import sys
import time
from types import FunctionType, SimpleNamespace

import cv2
from scripts import independent_round_trip_separate_audit_development as separate
from scripts import independent_round_trip_audit_process_development as parent
from scripts import independent_round_trip_audit_cpu_monitor_development as cpu
from scripts import independent_round_trip_collection_process_development as lifecycle
from scripts import independent_round_trip_collection_handoff_development as handoff
from scripts import independent_round_trip_population_case_evidence_development as evidence
from scripts.navigation_artifact_root_development import validate_root, verify_artifacts, artifact_path
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/independent_round_trip_monitored_audit_development.py'
TEST = 'lewm/tests/test_independent_round_trip_monitored_audit_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_monitored_audit_v1_2026-09-11.md'
PREPARATION = 'docs/go2_short_complete_raw_audit_cpu_monitor_preparation_2026-09-11.json'
PREPARATION_SHA = '3ce009f25a7555cef2770fcd0759aef039010e0ccf9ce7497c41a08e48c2c0b2'
MONITOR = '_audit_cpu_monitor.json'
EXECUTION = separate.EXECUTION
WORKER_STATUS = separate.WORKER_STATUS
FAILED = separate.FAILED
ENVIRONMENT = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
FIXED = dict(actual_independent_auditor_monitored=True, cpu_receipt_required_at_parent_acceptance=True,
    opencl_disabled_before_import=True, monitoring_failure_rejects_case=True,
    original_audit_body_unchanged=True, os_device_access_isolated=False)


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']; verify(inherited)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), inherited); verify(sources)
    return sources


def isolated(function, **overrides):
    if function.__closure__ is not None: raise ValueError('closure-free original function body required')
    clone = FunctionType(function.__code__, dict(function.__globals__, **overrides),
        function.__name__, function.__defaults__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def report_hash(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def require_launch(output, launch_sha):
    verify_artifacts(output, {'launch.json':launch_sha})
    launch = evidence.read_json(output, 'launch.json')
    names = (SOURCE, TEST, PROTOCOL, cpu.SOURCE, cpu.TEST, cpu.PROTOCOL,
        separate.SOURCE, separate.TEST, separate.PROTOCOL, parent.SOURCE, parent.TEST, parent.PROTOCOL)
    if (json.dumps(launch.get('audit_cpu_monitor'), sort_keys=True) != json.dumps(FIXED, sort_keys=True)
            or any(n not in launch['source_sha256'] for n in names)):
        raise ValueError('fixed monitored auditor and frozen worker/parent sources required')
    verify(launch['source_sha256'])
    return launch


def require_environment():
    if not __debug__ or any(os.environ.get(k) != v for k,v in ENVIRONMENT.items()) or cv2.ocl.useOpenCL():
        raise ValueError('assertions and fixed CPU environment with OpenCL disabled before import required')


def fresh(output, case):
    separate.fresh(output, case)
    path = output/(case.name+MONITOR)
    if path.exists() or path.is_symlink(): raise ValueError('preserve original CPU receipt; no retry')


def audit_worker(output, case, confirmation_sha, launch_sha, reference_sha, verifier):
    output = validate_root(output); fresh(output, case); require_environment()
    launch = require_launch(output, launch_sha)
    invoked = False

    def monitored(*args, **kwargs):
        nonlocal invoked
        if invoked: raise ValueError('exactly one full original audit call required')
        invoked = True
        confirmation = evidence.read_json(output, case.name+lifecycle.CONFIRMATION)
        owner = separate.require_audit_child(confirmation)
        monitor = cpu.AuditCPUMonitor(); report = None; started = time.perf_counter()
        try:
            with monitor: report = separate.audit(*args, **kwargs)
            if separate.require_audit_child(confirmation) != owner:
                raise ValueError('same original audit child required after monitored scope')
            return report
        finally:
            measured = monitor.summary() if monitor.closed else None
            encoding_error = None; report_sha = None
            try:
                if report is not None: report_sha = report_hash(report)
            except (TypeError, ValueError) as error:
                encoding_error = error
            success = bool(measured and measured['scope_returned_without_error'] and report_sha is not None)
            write_json(output/(case.name+MONITOR), dict(
                status='INDEPENDENT_RAW_AUDIT_CPU_SCOPE_COMPLETE' if success else 'INDEPENDENT_RAW_AUDIT_CPU_SCOPE_FAILED',
                case=case.name, launch_sha256=launch_sha, reference_worker_sha256=reference_sha,
                collection_confirmation_sha256=confirmation_sha, source_sha256=launch['source_sha256'],
                boot_id=handoff.BOOT, audit_owner=owner, monitor=measured,
                complete_raw_audit_report_sha256=report_sha,
                report_encoding_error=None if encoding_error is None else repr(encoding_error),
                wall_s=time.perf_counter()-started, original_auditor_called_directly=True,
                native_collection_called=False, os_device_access_isolated=False,
                overlap_execution_permitted=False, automatic_retry=False))
            if encoding_error is not None:
                raise ValueError('complete finite JSON raw-audit report required') from encoding_error

    returned = isolated(separate.audit_worker, audit=monitored)(output, case, confirmation_sha,
        launch_sha, reference_sha, verifier)
    if returned['status'] == WORKER_STATUS:
        if not invoked: raise ValueError('original auditor was not monitored')
        returned = dict(returned, cpu_monitor_sha256=digest(artifact_path(output, case.name+MONITOR)))
    return returned


def require_monitor(record, execution, report):
    expected = dict(status='INDEPENDENT_RAW_AUDIT_CPU_SCOPE_COMPLETE',
        original_auditor_called_directly=True, native_collection_called=False,
        os_device_access_isolated=False, overlap_execution_permitted=False, automatic_retry=False,
        report_encoding_error=None)
    expected.update({k:execution[k] for k in ('case','launch_sha256','reference_worker_sha256',
        'collection_confirmation_sha256','source_sha256','boot_id','audit_owner')})
    expected['complete_raw_audit_report_sha256'] = report_hash(report)
    separate.same_fields(record, expected, 'same original source-bound monitored raw audit required')
    measured = record['monitor']
    if not isinstance(measured, dict): raise ValueError('completed CPU monitor record required')
    required = dict(scope_returned_without_error=True, violations=[], initial_opencl_enabled=False,
        opencl_disabled_inside_scope=True, original_opencl_setting_restored=True,
        tensor_inputs_outputs_and_device_requests_checked=True, genesis_runtime_calls_prohibited=True,
        accelerator_initialization_prohibited=True, new_python_threads_prohibited=True,
        os_device_access_isolated=False, full_raw_auditor_qualified=False, overlap_execution_permitted=False)
    separate.same_fields(measured, required, 'complete original CPU instrumentation without violations required')
    for key in ('torch_operation_count','python_calls','native_python_calls'):
        if type(measured[key]) is not int or measured[key] < 0:
            raise ValueError('nonnegative integer monitor counters required')
    operations = measured['torch_operations']
    if (not isinstance(operations, dict) or any(type(v) is not int or v < 0 for v in operations.values())
            or sum(operations.values()) != measured['torch_operation_count']
            or measured['python_calls'] == 0):
        raise ValueError('consistent executed monitor counters required')
    if type(record['wall_s']) not in (int,float) or not 0 <= record['wall_s'] < float('inf'):
        raise ValueError('finite measured CPU scope duration required')


def read_completed_audit(output, case, returned, confirmation_sha, launch_sha, reference_sha, verifier):
    output = validate_root(output); require_launch(output, launch_sha)
    name = case.name+MONITOR; monitor_sha = returned['cpu_monitor_sha256']
    verify_artifacts(output, {name:monitor_sha})
    admitted = separate.read_completed_audit(output, case, returned, confirmation_sha,
        launch_sha, reference_sha, verifier)
    record = evidence.read_json(output, name)
    report = evidence.read_json(output, case.name+'_audit.json')
    require_monitor(record, admitted['execution'], report)
    admitted['artifact_sha256'][name] = monitor_sha
    verify_artifacts(output, admitted['artifact_sha256'])
    admitted['actual_raw_audit_cpu_scope_authenticated'] = True
    return admitted


def parent_register(process, output, case, confirmation_sha, launch_sha, reference_sha, verifier):
    output = validate_root(output); require_launch(output, launch_sha); fresh(output, case)
    return parent.register(process, output, case, confirmation_sha, launch_sha, reference_sha, verifier)


def parent_accept(ticket, returned, verifier):
    # The original parent still verifies its owned process, normal zero exit,
    # registration, complete raw evidence and all hashes before writing acceptance.
    return isolated(parent.accept, separate=sys.modules[__name__])(ticket, returned, verifier)


def parent_interface():
    return SimpleNamespace(Ticket=parent.Ticket, _tickets=parent._tickets,
        REGISTRATION=parent.REGISTRATION, COMPLETION=parent.COMPLETION,
        register=parent_register, accept=parent_accept,
        SOURCE=parent.SOURCE, TEST=parent.TEST, PROTOCOL=parent.PROTOCOL)
