"""Explicit corrected-input admission and original transfer authentication."""
import json
from pathlib import Path
from scripts.check_go2_all_phase_training_inputs_v1 import OUTPUT as CHECK, TARGETS, OLD_INPUTS, TARGET_SHA, OLD_INPUT_SHA
from scripts.observation_horizon_fit_inputs_development import authenticate as original_authenticate, stream as original_stream
from scripts.all_phase_study_stream_development import CheckedAllPhaseTrainingStream, AllPhaseStudyStream
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.run_go2_successive_choice_maze_development_v1 import digest
from scripts.startup_raw_sensor_audit_development import read_json

CHECK_SHA = 'ef48950b7987eaf9310ba8124a00c2e6e13c9b84b7cda6a362ac7ddb6ecd63fb'
CORRECTION = Path('docs/go2_all_phase_training_inputs_scope_correction_2026-09-10.json')
CORRECTION_SHA = '8fcfdc678b721a84ed53c2545ec55284e2185d7b1e6bc5ca0dc26da4832e2dca'


def corrected_inputs():
    """Verify exact completed artifacts and explicitly admit preserved scope errors.

    This does not replace full original collection authentication in authenticate.
    """
    if CORRECTION.is_symlink() or digest(CORRECTION) != CORRECTION_SHA:
        raise ValueError('exact bound expanded-input scope correction required')
    correction = json.loads(CORRECTION.read_text())
    verify_artifacts(CHECK, {'result.json': CHECK_SHA})
    result = read_json(CHECK, 'result.json')
    verify_artifacts(CHECK, result['artifact_sha256'])
    launch = read_json(CHECK, 'launch.json'); verify_ordered_launch(launch)
    if (result['status'] != 'ALL_PHASE_TRAINING_INPUTS_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['all_phase_target_result_sha256'] != TARGET_SHA
            or result['original_input_result_sha256'] != OLD_INPUT_SHA
            or correction['checker_result_sha256'] != CHECK_SHA
            or correction['checker_launch_sha256'] != result['artifact_sha256']['launch.json']
            or correction['checker_artifact_sha256'] != result['artifact_sha256']):
        raise ValueError('exact completed result, launch and scope-correction bindings required')
    source_ids = dict(launch['source_sha256'])
    for path, sha in (
        (launch['protocol'], correction['checker_protocol_sha256']),
        ('scripts/all_phase_training_policy_stream_development.py', correction['stream_source_sha256']),
        ('scripts/check_go2_all_phase_training_inputs_v1.py', correction['checker_source_sha256'])):
        if source_ids.get(path) != sha:
            raise ValueError('correction must bind actual executed protocol and stream sources')
    if (any(launch[k] is not v for k, v in correction['preserved_inherited_launch_fields'].items())
            or launch['private_training_future_materialization'] is not True
            or launch['geometry_transfer_future_materialization'] is not False
            or correction['actual_current_checker_scope']['inference_future_packet_reads'] != 0
            or correction['actual_current_checker_scope']['geometry_transfer_future_materialization'] is not False
            or correction['original_artifacts_modified'] is not False
            or correction['scientific_definition_changed'] is not False):
        raise ValueError('preserved incorrect flags and explicit actual training-only future scope required')
    report = result['report']
    expected = dict(context_slots=4800, materialized_training_contexts=4010,
        original_available_training_inputs_exact=408, all_causal_and_training_inputs_exact=True,
        geometry_transfer_future_materialization=False, native_artifacts_opened_by_stream=False,
        optimizer_updates=0, parameter_updates=0, matched_retraining_completed=False)
    if any(report[k] != v for k, v in expected.items()):
        raise ValueError('complete unchanged expanded input population and zero fitting required')
    verify_artifacts(TARGETS, launch['all_phase_target_artifact_sha256'])
    verify_artifacts(OLD_INPUTS, launch['original_input_artifact_sha256'])
    if (launch['all_phase_target_artifact_sha256']['result.json'] != TARGET_SHA
            or launch['original_input_artifact_sha256']['result.json'] != OLD_INPUT_SHA):
        raise ValueError('exact target and original input artifacts required')
    return launch, result, correction


def authenticate():
    """Full predecessor admission, including unchanged native collection receipts."""
    original_definition, _, _ = original_authenticate(OLD_INPUT_SHA)
    launch, result, correction = corrected_inputs()
    if any(launch['source_sha256'].get(p) != h for p, h in original_definition['source_sha256'].items()):
        raise ValueError('original and expanded input source closures disagree')
    return launch, result, correction


def stream(*, maximum_cache_bytes=0):
    """Caller authenticates before/after; each reader verifies consumed leaves."""
    launch, _, _ = corrected_inputs()
    training = CheckedAllPhaseTrainingStream(read_json(TARGETS, 'windows.json'),
        launch['consumed_training_policy_sha256'], read_json(CHECK, 'tensor_index.json'),
        maximum_cache_bytes=maximum_cache_bytes)
    study = AllPhaseStudyStream(training, original_stream(OLD_INPUT_SHA))
    expected = {('train', 'family'): 1664, ('train', 'switch'): 2346,
        ('geometry_transfer', 'family'): 348, ('geometry_transfer', 'switch'): 72}
    if any(len(study.view.indices(role, source=source)) != count
            for (role, source), count in expected.items()):
        raise ValueError('exact completed expanded training and original transfer populations required')
    return study
