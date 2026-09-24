"""Post-hoc coverage only; reuse exact admitted sensor bytes, no native rerun.

The prior diagnostic reconstructed all 96 streams before native access. This
reader requires that exact completed diagnostic and reauthenticates all its
inputs and all 96 stream bindings. It does not reconstruct transforms again,
run the incomplete full raw audit, score tracking, or pass the original attempt.
"""
import hashlib
import json
import time

import numpy as np

from lewm.representation_aware_tracking_coverage_development import coverage_with_sensor_rotation_convention
from scripts import read_go2_independent_tracking_quaternion_diagnostic_v1 as prior
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_stress_cohort_development as stress
from scripts.navigation_artifact_root_development import artifact_path, verify_artifacts

DIAGNOSTIC = 'docs/go2_independent_tracking_quaternion_diagnostic_2026-09-07.json'
DIAGNOSTIC_SHA = '44f7d6d5307320d2acca00178747693dbfe109179de008bb694e8f723b3897ff'


def admitted_inputs():
    path = prior.ROOT / DIAGNOSTIC
    base.require(path.resolve() == path and path.is_file(), 'exact diagnostic document required')
    data = path.read_bytes()
    base.require(hashlib.sha256(data).hexdigest() == DIAGNOSTIC_SHA, 'completed diagnostic identity changed')
    evidence = json.loads(data)
    base.require(evidence['status'] == 'FAILED_TRACKING_QUATERNION_DIAGNOSTIC_COMPLETE'
        and evidence['sensor_streams_admitted_before_native_arrays'] == 96
        and evidence['inner_bindings'] == prior.INNER_BINDINGS
        and evidence['outside_bindings'] == prior.OUTSIDE_BINDINGS
        and evidence['definition_sha256'] == prior.DEFINITION
        and set(evidence['reports']) == set(base.TRIALS), 'exact complete admitted diagnostic required')
    verify_artifacts(prior.OUTPUT, prior.INNER_BINDINGS)
    verify_artifacts(prior.OUTSIDE, prior.OUTSIDE_BINDINGS)
    definition = base.read(prior.OUTPUT, 'launch.json')['definition']
    base.require(prior.definition_identity(definition) == prior.DEFINITION, 'same source/configuration required')
    prior.source_bindings(definition)
    _, episodes = base.verify_collection(prior.OUTPUT, prior.INNER_BINDINGS['collection_complete.json'])
    phase = base.read(prior.OUTPUT, 'sensor_phase_complete.json')
    treated = base.read(prior.OUTPUT, stress.PHASE)
    bindings = {}
    for trial in base.TRIALS:
        for name, report in [('base', phase['reports'][trial]), *treated['reports'][trial].items()]:
            expected = trial + '_estimates.jsonl' if name == 'base' else stress.stream_name(trial, name)
            base.require(report['estimates_file'] == expected, 'exact admitted stream name')
            bindings[expected] = report['estimates_sha256']
    base.require(len(bindings) == 96, 'complete 96-stream identity population required')
    verify_artifacts(prior.OUTPUT, bindings)
    return episodes, bindings, definition


def readout():
    started = time.monotonic()
    episodes, bindings, definition = admitted_inputs()
    reports = {}
    for trial in base.TRIALS:
        with np.load(artifact_path(prior.OUTPUT, trial + '/physics_trace.npz'), allow_pickle=False) as z:
            poses, times, twists = z['base_pose_world'], z['timestamp_s'], z['base_twist_world']
        result = episodes[trial]['result']
        base.require(len(poses) == result['physics_samples'], 'same complete native population')
        reports[trial] = coverage_with_sensor_rotation_convention(base.specification(trial)['direction'],
            times, poses, twists, **{k: result[k] for k in (
                'completed_ticks', 'schedule_complete', 'physical_stop', 'acquisition_stop')})
    prior.source_bindings(definition)
    base.verify_collection(prior.OUTPUT, prior.INNER_BINDINGS['collection_complete.json'])
    verify_artifacts(prior.OUTPUT, prior.INNER_BINDINGS | bindings)
    verify_artifacts(prior.OUTSIDE, prior.OUTSIDE_BINDINGS)
    base.require(hashlib.sha256((prior.ROOT / DIAGNOSTIC).read_bytes()).hexdigest() == DIAGNOSTIC_SHA,
                 'diagnostic identity changed during readout')
    return dict(status='POSTHOC_SENSOR_CONVENTION_COVERAGE_READOUT_COMPLETE',
        prior_diagnostic_sha256=DIAGNOSTIC_SHA, definition_sha256=prior.DEFINITION,
        original_inner_bindings=prior.INNER_BINDINGS, original_outside_bindings=prior.OUTSIDE_BINDINGS,
        same_admitted_sensor_streams_reauthenticated=96, sensor_transforms_reexecuted=False,
        reports=reports, elapsed_seconds=time.monotonic() - started,
        original_failure_preserved=True, original_arrays_modified=False,
        original_attempt_resumed=False, native_simulator_executed=False,
        raw_physics_sensor_audit_completed=False, tracking_accuracy_verified=False,
        full_challenge_pass=False, navigation_qualified=False, goal_achieved=False)


if __name__ == '__main__':
    print(json.dumps(readout(), sort_keys=True, allow_nan=False))
