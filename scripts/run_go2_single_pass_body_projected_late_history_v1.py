"""Exclusive full paired replay and ended-owner completion verification."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from types import FunctionType
import time

import psutil

from scripts import single_pass_body_projected_replay_development as harness
from scripts import body_projection_external_profile_admission_development as original_admission
from scripts.verify_go2_body_projected_tiled_controller_completion_v1 import check_rows
from scripts.status_go2_navigation_and_external_profile_v2 import snapshot
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.navigation_artifact_root_development import create_output, validate_root, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

SOURCE = 'scripts/run_go2_single_pass_body_projected_late_history_v1.py'
TEST = 'lewm/tests/test_single_pass_body_projected_runner_development.py'
PROTOCOL = 'docs/go2_single_pass_body_projected_late_history_v1_2026-09-11.md'
FAILURE_AUDIT = 'docs/go2_external_profile_v2_terminal_failure_audit_2026-09-11.json'
FAILURE_AUDIT_SHA = 'd3a959fbc92a1792319513c3d96ac7fea749464113e875b0349cf389a42d4eb8'
OUTPUT = harness.OUTPUT
COMPLETION = ROOT/'docs/go2_single_pass_body_projected_completion_verification_2026-09-11.json'
ENVIRONMENT = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
STATUS = 'SINGLE_PASS_BODY_PROJECTED_LATE_HISTORY_V1_COMPLETE'


def private_admission():
    namespace = vars(original_admission).copy()
    namespace['OUTPUT'] = OUTPUT
    for name in ('capture_verification', 'admit_completed'):
        original = getattr(original_admission, name)
        if original.__closure__ is not None:
            raise ValueError('closure-free original admission required')
        clone = FunctionType(original.__code__, namespace, name, original.__defaults__)
        clone.__kwdefaults__ = original.__kwdefaults__
        namespace[name] = clone
    return namespace['admit_completed']()


def sources():
    witness = original_admission.reference_witness()
    verify({FAILURE_AUDIT: FAILURE_AUDIT_SHA})
    audit = json.loads((ROOT/FAILURE_AUDIT).read_text())
    failed_root = Path(audit['root'])
    verify_artifacts(failed_root, {name: value['sha256'] for name, value in audit['artifacts'].items()})
    verify(audit['document_sha256'])
    seeds = (SOURCE, TEST, PROTOCOL, FAILURE_AUDIT, harness.TEST,
        'lewm/tests/test_single_pass_body_projected_controller_development.py',
        'lewm/tests/test_single_pass_sample_bounds_development.py')
    bindings = discover_sources(seeds, witness['source_sha256'])
    verify(bindings)
    return bindings


def slot_available():
    status = snapshot()
    if any(owner['state'] != 'ended' for owner in status['profile']['owners'].values()):
        raise ValueError('all four original profiler owners must be ended')


def hardware():
    resources = harness.original.profile.reference.hardware()
    harness.original.resources_for(resources)
    return resources


def expected_report(prior_report, timing):
    expected = deepcopy(prior_report)
    for key in ('incremental_body_projection_comparison',
            'both_controllers_use_original_tiled_classification_and_receipt_handling'):
        if expected.pop(key) is not True:
            raise ValueError('original completed body-projection scope required')
    expected.update(baseline='BodyProjectedTiledController', candidate='SinglePassBodyProjectedController',
        timing_windows=timing, normalized_state_type_paths=harness.STATE_TYPE_PATHS,
        incremental_reuse_comparison=False, incremental_single_pass_bounds_comparison=True,
        persistent_memory_type_unchanged=True, original_packed_insertion_unchanged=True,
        both_controllers_use_original_body_projection_and_receipt_handling=True,
        imported_module_globals_mutated=False)
    return expected


def validate_report(report, rows, prior):
    timing = check_rows(rows, prior[3])
    if fingerprint(report) != fingerprint(expected_report(prior[1]['report'], timing)):
        raise ValueError('complete report, timings and all seven state witnesses must reconstruct')
    return timing


def fixed_launch():
    return dict(reference_verification_sha256=original_admission.VERIFICATION_SHA,
        reference_result_sha256=original_admission.RESULT_SHA,
        reference_launch_sha256=original_admission.LAUNCH_SHA,
        environment=ENVIRONMENT, frames=1428, state_frames=list(harness.original.STATE_FRAMES),
        normalized_state_type_paths=harness.STATE_TYPE_PATHS,
        baseline='BodyProjectedTiledController', candidate='SinglePassBodyProjectedController',
        native_execution=False, model_training=False, automatic_retry=False)


def launch(preflight_only=False):
    if (not __debug__ or any(os.environ.get(key) != value for key, value in ENVIRONMENT.items())
            or harness.original.cv2.ocl.useOpenCL()):
        raise ValueError('assertions, deterministic environment and disabled OpenCL required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive single-pass replay; no retry or resume')
    bindings = sources(); slot_available(); resources = hardware()
    if preflight_only:
        print('SINGLE_PASS_BODY_PROJECTED_PREFLIGHT', len(bindings), json.dumps(resources), flush=True)
        return
    print('SINGLE_PASS_BODY_PROJECTED_ORIGINAL_INPUT_ADMISSION_STARTED', flush=True)
    prior = private_admission()
    verify(bindings); slot_available(); resources = hardware()
    create_output(OUTPUT)
    process = psutil.Process()
    write_json(OUTPUT/'launch.json', fixed_launch() | dict(source_sha256=bindings, hardware=resources,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline())))
    print('SINGLE_PASS_BODY_PROJECTED_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    harness.original.cv2.setNumThreads(1); harness.original.torch.set_num_threads(1)
    harness.original.torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = harness.replay(prior[3], prior[1]['report'])
        rows = [json.loads(line) for line in (OUTPUT/'comparison.jsonl').read_text().splitlines()]
        validate_report(report, rows, prior)
        if private_admission() != prior:
            raise ValueError('complete original inputs changed during paired replay')
        verify(bindings)
        artifacts = {name: digest(OUTPUT/name) for name in ('launch.json', 'comparison.jsonl')}
        verify_artifacts(OUTPUT, artifacts)
        write_json(OUTPUT/'result.json', dict(status=STATUS, source_sha256=bindings,
            artifact_sha256=artifacts, report=report, sensing_scope=prior[1]['sensing_scope'],
            original_inputs_reauthenticated_before_and_after=True, wall_s=time.perf_counter()-start,
            native_execution=False, real_time_qualified=False, navigation_qualified=False, goal_achieved=False))
        print('SINGLE_PASS_BODY_PROJECTED_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_SINGLE_PASS_BODY_PROJECTED_FAILURE',
            reason=repr(error), automatic_retry=False, evidence_preserved=True))
        raise


def verify_completion(result_sha):
    if COMPLETION.exists() or COMPLETION.is_symlink():
        raise ValueError('exclusive completion verification required')
    if (OUTPUT/'failure.json').exists() or (OUTPUT/'failure.json').is_symlink():
        raise ValueError('failed replay cannot be accepted')
    verify_artifacts(OUTPUT, {'result.json': result_sha})
    result = json.loads((OUTPUT/'result.json').read_text())
    artifacts = result['artifact_sha256']
    if set(artifacts) != {'launch.json', 'comparison.jsonl'}:
        raise ValueError('complete exact paired artifact set required')
    verify_artifacts(OUTPUT, artifacts)
    launch_record = json.loads((OUTPUT/'launch.json').read_text())
    if (launch_record['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or owner_live(launch_record['owner'])):
        raise ValueError('original replay owner must be ended on its recorded boot')
    bindings = sources()
    if (result['status'] != STATUS or result['source_sha256'] != launch_record['source_sha256']
            or result['source_sha256'] != bindings
            or any(result[k] is not False for k in ('native_execution', 'real_time_qualified',
                'navigation_qualified', 'goal_achieved'))
            or result['original_inputs_reauthenticated_before_and_after'] is not True
            or type(result['wall_s']) not in (int, float) or not math.isfinite(result['wall_s'])
            or result['wall_s'] <= 0
            or fingerprint({k: launch_record[k] for k in fixed_launch()}) != fingerprint(fixed_launch())):
        raise ValueError('exact original execution, sources and diagnostic scope required')
    prior = private_admission()
    rows = [json.loads(line) for line in (OUTPUT/'comparison.jsonl').read_text().splitlines()]
    timing = validate_report(result['report'], rows, prior)
    if result['sensing_scope'] != prior[1]['sensing_scope']:
        raise ValueError('original negative sensing scope required')
    verify(bindings); verify_artifacts(OUTPUT, artifacts | {'result.json': result_sha})
    if owner_live(launch_record['owner']): raise ValueError('original owner unexpectedly live')
    total = timing['all_navigation']
    write_json(COMPLETION, dict(status='SINGLE_PASS_BODY_PROJECTED_COMPLETION_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=bindings,
        result_sha256=result_sha, artifact_sha256=artifacts, original_owner=launch_record['owner'],
        original_owner_ended=True, actual_original_raw_model_inputs_reauthenticated=True,
        complete_rows=1428, raw_model_forecasts=1425,
        observed_state_checks=result['report']['observed_state_checks'], timing_windows=timing,
        total_navigation_time_reduction_percent=100*(1-total['candidate_total_s']/total['baseline_total_s']),
        sensing_scope=result['sensing_scope'], native_execution=False, real_time_qualified=False,
        navigation_qualified=False, goal_achieved=False))
    print('SINGLE_PASS_BODY_PROJECTED_COMPLETION_VERIFIED', digest(COMPLETION), flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--source-preflight-only', action='store_true')
    group.add_argument('--verify-result-sha256')
    args = parser.parse_args()
    if args.verify_result_sha256: verify_completion(args.verify_result_sha256)
    else: launch(args.source_preflight_only)


if __name__ == '__main__': main()
