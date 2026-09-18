"""Read-only diagnosis of the exact failed V1; no retry, rescore or repair."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from lewm.tracking_quaternion_precision_diagnostic_development import summarize_quaternions
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_stress_cohort_development as stress
from scripts.navigation_artifact_root_development import BASE, validate_root, artifact_path, verify_artifacts
from scripts.startup_source_inventory_development import allowed_relative

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = BASE / 'go2_independent_tracking_challenge_v1_attempt_001'
OUTSIDE = BASE / 'go2_independent_tracking_supervision_v1_attempt_001'
INNER_BINDINGS = {
    'launch.json': 'ac4f532d0b8cf6e3cab4e984f7a5845ebc0ac3f8d225ed6c696a9bdc5f79c4f2',
    'collection_complete.json': 'eb94e4db44120ab5eb90c500fd4d832d5fa8bc216e83399aff56af45dbabac7d',
    'sensor_phase_complete.json': '5c9d451f35fe64e87219234878e7445bb0285fbb5bf72ded8be9b86ff5c30348',
    'stress_sensor_phase_complete.json': '671aa02549a748ca34af9258142fc698b920974baf529e245cd1dc691547e163',
    'failure.json': 'd0902842b1446882430093c4969ed78127a50256e0b66195207a40ce631a7486',
}
OUTSIDE_BINDINGS = {
    'terminal.json': '0cb3f84351b721ee4eb50db1ee837655c2a1acfc376410009893871dccd0c37d',
    'request.json': '4693ef9ccf3dbea5e11346f1a5c5bceb243108e97fba22a5e8e1b65ba557bfa1',
    'unit.log': '359c8e27ceed84d6012e3935b7d12a22173674c115c86fa23a935a577e05d8a1',
}
DEFINITION = '223056ac7ddcb47b9d1a4b1b188028761fb15f56443f02fda28ecfa668326744'


def definition_identity(definition):
    # The frozen study/challenge identity includes the terminal newline.
    return hashlib.sha256(base.encode(definition)).hexdigest()


def source_bindings(definition):
    bindings = definition['source_sha256']
    base.require(len(bindings) == 825, 'exact frozen source population required')
    for name, sha in bindings.items():
        p = ROOT / allowed_relative(name)
        base.require(p.resolve() == p and p.is_file(), 'ordinary nonsymlink source required')
        with p.open('rb') as f:
            base.require(hashlib.file_digest(f, 'sha256').hexdigest() == sha, 'frozen source changed: ' + name)


def diagnose():
    started = time.monotonic()
    validate_root(OUTPUT); validate_root(OUTSIDE)
    verify_artifacts(OUTPUT, INNER_BINDINGS); verify_artifacts(OUTSIDE, OUTSIDE_BINDINGS)
    failure = base.read(OUTPUT, 'failure.json'); terminal = base.read(OUTSIDE, 'terminal.json')
    base.require(failure['status'] == 'TERMINAL_TRACKING_COHORT_PHASE_FAILURE'
        and failure['stage'] == 'complete_stress_native_audit_or_admission'
        and failure['reason'] == "ValueError('unit native quaternions required')"
        and terminal['status'] == 'SCOPED_COMMAND_FAILED'
        and terminal['child_handle_terminal'] is True and terminal['systemd_run_returncode'] == 1
        and terminal['log_complete'] is True and terminal['log_omitted_bytes'] == 0,
        'exact retained terminal failure required')
    launch = base.read(OUTPUT, 'launch.json'); definition = launch['definition']
    identity = definition_identity(definition)
    base.require(identity == launch['definition_sha256'] == terminal['definition_sha256'] == DEFINITION,
                 'frozen source/configuration identity required')
    source_bindings(definition)
    print('QUATERNION_DIAGNOSTIC_COMPLETE_SENSOR_ADMISSION_START', flush=True)
    episodes, phase, treated = stress.admit_complete_sensor_phase(OUTPUT,
        INNER_BINDINGS['collection_complete.json'], INNER_BINDINGS['sensor_phase_complete.json'],
        INNER_BINDINGS[stress.PHASE])
    print('QUATERNION_DIAGNOSTIC_COMPLETE_SENSOR_ADMISSION_FINISHED', flush=True)
    reports = {}
    for trial in base.TRIALS:
        # Exact receipt-bound array only, after all 96 sensor streams admitted.
        path = artifact_path(OUTPUT, trial + '/physics_trace.npz')
        with np.load(path, allow_pickle=False) as z:
            pose, timestamps = z['base_pose_world'], z['timestamp_s']
        base.require(pose.shape == (episodes[trial]['result']['physics_samples'], 7),
                     'recorded sample count and pose shape required')
        reports[trial] = summarize_quaternions(pose[:, 3:], timestamps)
    source_bindings(definition)
    base.verify_collection(OUTPUT, INNER_BINDINGS['collection_complete.json'])
    verify_artifacts(OUTPUT, INNER_BINDINGS); verify_artifacts(OUTSIDE, OUTSIDE_BINDINGS)
    verify_artifacts(OUTPUT, {r['estimates_file']: r['estimates_sha256']
        for r in phase['reports'].values()} | {r['estimates_file']: r['estimates_sha256']
        for trial in treated['reports'].values() for r in trial.values()})
    return dict(status='FAILED_TRACKING_QUATERNION_DIAGNOSTIC_COMPLETE',
        definition_sha256=DEFINITION, inner_bindings=INNER_BINDINGS, outside_bindings=OUTSIDE_BINDINGS,
        sensor_streams_admitted_before_native_arrays=96, reports=reports,
        elapsed_seconds=time.monotonic() - started,
        original_failure_preserved=True, original_attempt_resumed=False, original_artifacts_modified=False,
        native_simulator_executed=False, observer_inference_recomputed=False,
        raw_sensor_physics_audit_completed=False, learning_prerequisite_reauthenticated=False,
        native_trajectory_validated=False, tracking_accuracy_verified=False,
        full_challenge_pass=False, navigation_qualified=False, goal_achieved=False)


if __name__ == '__main__':
    print(json.dumps(diagnose(), sort_keys=True, allow_nan=False))
