"""Exact completed input-check admission and fresh private family stream."""
from lewm.family_transition_bootstrap_scope_development import admit_scope
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView
from scripts.geometry_progress_family_policy_stream_development import FamilyPolicyStream
from scripts.check_go2_family_transition_bootstrap_inputs_v1 import OUTPUT as CHECK, INPUT, DERIVATION, COLLECTION_SHA, CAUSAL_SHA
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify

CHECK_SHA = '92b5fa6066715bad218b9d2cecdd853a10ff2448e93ab886d5af1214de09a01e'


def authenticate():
    verify_artifacts(CHECK, {'result.json': CHECK_SHA})
    result = read_json(CHECK, 'result.json')
    if (result['status'] != 'FAMILY_TRANSITION_BOOTSTRAP_INPUTS_COMPLETE'
            or result['transition_fit_inputs_validated'] is not True or result['optimizer_steps'] != 0
            or result['original_failures_preserved'] is not True or result['original_outcomes_changed'] is not False):
        raise ValueError('complete unchanged new-scope input check required')
    verify_artifacts(CHECK, result['artifact_sha256'])
    launch = read_json(CHECK, 'launch.json')
    verify_artifacts(INPUT, launch['collection_sha256']); verify_artifacts(DERIVATION, launch['causal_sha256'])
    collection, derived = read_json(INPUT, 'result.json'), read_json(DERIVATION, 'result.json')
    if launch['collection_sha256']['result.json'] != COLLECTION_SHA or launch['causal_sha256']['result.json'] != CAUSAL_SHA:
        raise ValueError('exact original collection/derivation required')
    if admit_scope(collection, derived) != result['scope'] or result['scope'] != launch['scope']:
        raise ValueError('unchanged transition prediction scope required')
    definition = read_json(INPUT, 'launch.json') | dict(source_sha256=result['source_sha256'])
    verify(definition)
    schedule = read_json(CHECK, 'training_schedule.json')
    view = FamilyWindowView(read_json(DERIVATION, 'windows.json'))
    if schedule != view.schedule(updates=1200, batch_size=6, seed=2026091001):
        raise ValueError('exact fixed common family schedule required')
    return definition, result, schedule


def stream():
    # The caller authenticates all source/data receipts before and after work;
    # the policy stream separately hashes every file it consumes on admission.
    view = FamilyWindowView(read_json(DERIVATION, 'windows.json'))
    collection = read_json(INPUT, 'result.json')
    return FamilyPolicyStream(view, output=INPUT, bindings=collection['artifact_sha256'],
        tensor_index=read_json(DERIVATION, 'tensor_index.json'))
