"""Authenticate bounded CPU evidence and enforce the monitored study driver.

This is the overlap verifier only. Final input, queue and policy admission
remain mandatory through the separate population verifier.
"""
from copy import deepcopy
import json

from scripts import independent_round_trip_monitored_audit_development as monitored
from scripts import independent_round_trip_monitored_staged_population_development as driver
from scripts import replay_go2_independent_factory_cpu_monitor_startup_v1 as startup
from scripts import replay_go2_short_complete_raw_audit_cpu_monitor_v1 as short
from scripts import independent_round_trip_population_runtime_development as runtime
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.maze_decision_stream_development import read_rows

SOURCE = 'scripts/independent_round_trip_audit_overlap_admission_development.py'
TEST = 'lewm/tests/test_independent_round_trip_audit_overlap_admission_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_audit_overlap_admission_v1_2026-09-11.md'
PREPARATION = 'docs/go2_independent_round_trip_monitored_audit_preparation_2026-09-11.json'
PREPARATION_SHA = '9a194add2beba9e189fd48e50779b685ba1bea8876439d99440f78472a477b37'
STARTUP_SHA = 'd7810ed1ad163a5c490cdf2302af8678bca82f71e0414d122cc89dce4b8794f3'
STARTUP_LAUNCH = 'fd1d226e0364400fec17abcbde792b81d51ac1b4770dc68aa139fa9717665e69'
SHORT_SHA = 'f7118d416212d5a4758f15274e8c8596f68382b07ea8ab40592ca5922b87bacb'
SHORT_LAUNCH = '66ed12f6245c5840dc4cde0dcdc5c3eba8668924bd1f96f4280645167a2edc12'
EVIDENCE = dict(monitored_runtime_preparation_sha256=PREPARATION_SHA,
    factory_cpu_startup_result_sha256=STARTUP_SHA, complete_short_raw_audit_result_sha256=SHORT_SHA,
    evidence_scope='bounded_completed_probes_and_mandatory_runtime_monitoring',
    prior_complete_independent_multiarm_audit_claimed=False, os_device_access_isolated=False,
    measured_collection_audit_speedup_claimed=False)
ENTRYPOINT = dict(source=driver.SOURCE, function='run_population')


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']; verify(inherited)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), inherited); verify(sources)
    return sources


def require_equal(actual, expected, message):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(message)


def require_cpu(measured, *, learned):
    if not isinstance(measured, dict): raise ValueError('completed CPU monitor required')
    expected = dict(scope_returned_without_error=True, violations=[], initial_opencl_enabled=False,
        opencl_disabled_inside_scope=True, original_opencl_setting_restored=True,
        tensor_inputs_outputs_and_device_requests_checked=True, genesis_runtime_calls_prohibited=True,
        accelerator_initialization_prohibited=True, new_python_threads_prohibited=True,
        os_device_access_isolated=False, full_raw_auditor_qualified=False, overlap_execution_permitted=False)
    monitored.separate.same_fields(measured, expected, 'successful bounded CPU evidence with unchanged scope required')
    for name in ('torch_operation_count','python_calls','native_python_calls'):
        if type(measured[name]) is not int or measured[name] < 0:
            raise ValueError('nonnegative CPU measurement counters required')
    operations = measured['torch_operations']
    if (not isinstance(operations, dict) or any(type(v) is not int or v < 0 for v in operations.values())
            or sum(operations.values()) != measured['torch_operation_count']
            or measured['python_calls'] <= 0 or (learned and measured['torch_operation_count'] <= 0)):
        raise ValueError('actual consistent bounded CPU execution counters required')


def completed(root, result_sha, launch_sha, sources):
    verify_artifacts(root, {'result.json':result_sha,'launch.json':launch_sha})
    result = read_json(root, 'result.json'); launch = read_json(root, 'launch.json')
    if (result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or result['artifact_sha256']['launch.json'] != launch_sha):
        raise ValueError('original completed source and launch bindings required')
    verify(result['source_sha256']); verify_artifacts(root, result['artifact_sha256'])
    return result, launch


def require_startup(result, reference, reports, streams, original_streams):
    monitored.separate.same_fields(result, dict(status='INDEPENDENT_FACTORY_CPU_MONITOR_STARTUP_V1_COMPLETE',
        reference_result_sha256=startup.REFERENCE_SHA, original_startup_outputs_unchanged=True,
        full_raw_auditor_qualified=False, overlap_execution_permitted=False, native_execution=False,
        independent_layout_sensor_data_consumed=False, command_executed=False, goal_achieved=False),
        'original bounded four-arm CPU startup required')
    if len(result['reports']) != 4 or len(reference['reports']) != 4:
        raise ValueError('all four fixed factory arms required')
    for case, report, original in zip(startup.CASES[:4], result['reports'], reference['reports'], strict=True):
        arm = case.arm_name
        monitored.separate.same_fields(report, dict(arm=arm, original_report=original,
            complete_saved_startup_decisions_exact=True, frames=4, full_raw_auditor_qualified=False,
            overlap_execution_permitted=False), 'same complete original factory report required')
        if report['cpu_monitor'] != reports[arm]: raise ValueError('same saved per-arm monitor required')
        require_cpu(reports[arm], learned=arm != 'reactive')
        if (streams[arm] != original_streams[arm] or len(streams[arm]) != 4
                or [r['tick'] for r in streams[arm]] != list(range(4))):
            raise ValueError('complete original four-packet streams required for every arm')


def require_short(result, monitor, audit, original):
    monitored.separate.same_fields(result, dict(status='SHORT_COMPLETE_RAW_AUDIT_CPU_MONITOR_V1_COMPLETE',
        reference_result_sha256=short.REFERENCE_SHA, complete_original_raw_inputs_rehashed_before_and_after=True,
        independent_multiarm_raw_audit_qualified=False, overlap_execution_permitted=False,
        native_execution=False, independent_layout_sensor_data_consumed=False, goal_achieved=False),
        'complete original bounded raw-audit CPU probe required')
    monitored.separate.same_fields(result['report'], dict(complete_original_audit_report_exact=True,
        original_failed_navigation_preserved=True, observations=14, physics_samples=1400,
        command_intervals=13, model_loading_and_geometry_construction_monitored=True,
        cpu_monitor=monitor, independent_multiarm_raw_audit_qualified=False,
        overlap_execution_permitted=False, command_executed=False), 'complete original short raw audit required')
    require_cpu(monitor, learned=True)
    if audit != original or audit['verified_round_trip'] is not False:
        raise ValueError('complete original failed-navigation audit report must be preserved')


def admit_proofs(sources, *, full=False):
    if type(full) is not bool: raise ValueError('explicit full-admission boolean required')
    verify(sources)
    result, launch = completed(startup.OUTPUT, STARTUP_SHA, STARTUP_LAUNCH, sources)
    reference, original_launch = completed(startup.original.OUTPUT, startup.REFERENCE_SHA,
        startup.REFERENCE_LAUNCH_SHA, sources)
    startup.require_reference(reference, original_launch, sources)
    if (launch['reference_result_sha256'] != startup.REFERENCE_SHA
            or launch['reference_launch_sha256'] != startup.REFERENCE_LAUNCH_SHA):
        raise ValueError('same completed four-arm startup reference required')
    arms = startup.CASES[:4]
    reports = {c.arm_name:read_json(startup.OUTPUT,c.arm_name+'_cpu_monitor.json') for c in arms}
    streams = {c.arm_name:list(read_rows(startup.OUTPUT/c.arm_name)) for c in arms}
    original_streams = {c.arm_name:list(read_rows(startup.original.OUTPUT/c.arm_name)) for c in arms}
    require_startup(result, reference, reports, streams, original_streams)
    short_result, short_launch = completed(short.OUTPUT, SHORT_SHA, SHORT_LAUNCH, sources)
    # The quick path authenticates only the exact original reference report,
    # rather than rehashing all six old episodes on every scheduling check.
    verify_artifacts(short.original.OUTPUT, {'result.json':short.REFERENCE_SHA})
    old = read_json(short.original.OUTPUT, 'result.json')
    name = short.CASE[0]+'_audit.json'
    verify_artifacts(short.original.OUTPUT, {name:old['artifact_sha256'][name]})
    original = read_json(short.original.OUTPUT, name)
    if short_launch['reference_result_sha256'] != short.REFERENCE_SHA:
        raise ValueError('same completed original short-episode reference required')
    require_short(short_result, read_json(short.OUTPUT,'cpu_monitor.json'),
        read_json(short.OUTPUT,'audit.json'), original)
    if full:
        # Reuses the full bounded-input verifier from the completed short probe.
        # It includes the original factory inputs and corrected-model bindings.
        # It never reruns an auditor, model, native scene or training ancestry.
        _, _, reconstructed_original = short.admit(sources)
        if reconstructed_original != original: raise ValueError('original full-admission report changed')
    for root, r, sha in ((startup.OUTPUT,result,STARTUP_SHA),(short.OUTPUT,short_result,SHORT_SHA)):
        verify_artifacts(root, r['artifact_sha256'] | {'result.json':sha})
    verify(sources)
    return dict(evidence=deepcopy(EVIDENCE), bounded_completed_probe_outputs_verified=True,
        full_original_bound_inputs_rechecked=full, new_independent_data_consumed=False,
        prior_complete_independent_multiarm_audit_claimed=False, population_execution_permitted=False)


def verify_overlap(launch, *, full=False):
    if type(full) is not bool: raise ValueError('explicit full-admission boolean required')
    monitored.require_environment()
    verify({PREPARATION:PREPARATION_SHA})
    frozen = json.loads((ROOT/PREPARATION).read_text())['source_sha256']
    sources = launch['source_sha256']
    if any(sources.get(n) != h for n,h in frozen.items()):
        raise ValueError('complete exact frozen monitored runtime source bindings required')
    if any(n not in sources for n in (SOURCE,TEST,PROTOCOL,PREPARATION)):
        raise ValueError('overlap verifier and its evidence must be source-bound')
    verify(sources)
    require_equal(launch.get('audit_cpu_monitor'), monitored.FIXED, 'mandatory actual audit monitoring required')
    require_equal(launch.get('staged_runtime'), driver.staged.FIXED, 'fixed one-collector/one-auditor bounds required')
    require_equal(launch.get('overlap_evidence'), EVIDENCE, 'exact bounded evidence and honest qualification scope required')
    require_equal(launch.get('population_entrypoint'), ENTRYPOINT, 'monitored population entrypoint required')
    require_equal(launch.get('overlap_verifier'), dict(source=SOURCE,function='verify_overlap'),
        'exact source-bound overlap verifier required')
    runtime.require_verifier(verify_overlap, dict(launch, runtime_verifier=launch['overlap_verifier']))
    runtime.require_verifier(driver.run_population, dict(launch, runtime_verifier=ENTRYPOINT))
    admit_proofs(sources, full=full)
    verify(sources)


def run_population(output, launch_sha, final_verifier):
    """Use the actual fixed monitored driver; final admission remains mandatory."""
    return driver.run_population(output, launch_sha, final_verifier, verify_overlap)
