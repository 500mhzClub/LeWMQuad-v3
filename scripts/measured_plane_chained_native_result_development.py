"""Reconstruct the actual dynamic intervention when accepting a native audit.

This prepares result validation for a future separately admitted native run.
It starts no controller, model, worker, or scene. The launcher must authenticate
the completed replay and the complete native artifact rosters separately.
"""
from scripts import measured_plane_chained_native_prefix_development as prefix
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit

SOURCE = 'scripts/measured_plane_chained_native_result_development.py'
TEST = 'lewm/tests/test_measured_plane_chained_native_result_development.py'


def prefix_result(collection, report, *, prior, current):
    availability = prefix.prefix_availability(collection, report)
    if not availability['boundary_interval_available_by_collection_counts']:
        return dict(status='CHAINED_INTERVENTION_NOT_COMPLETED',
            collection_availability=availability, full_raw_audit_retained=True,
            actual_paired_execution_compared=False, navigation_verified=False,
            unexecuted_outcomes_inferred=False)
    # Counts alone cannot establish the intervention. Reconstruct the complete
    # raw physics, public packets and both decisions through the actual boundary.
    reconstructed = prefix.compare(prior, current, report)
    return dict(status='COMPLETE_CHAINED_ACTUAL_PREFIX',
        collection_availability=availability, full_raw_audit_retained=True,
        actual_paired_execution_compared=True, **reconstructed)


def require_worker(record, audit, report, *, case, worker_status, prior, current):
    expected = dict(status=worker_status, case=case[0], layout_index=case[1],
        variant=case[2], condition=case[3], model_name=case[4],
        model_state_sha256=prefix.replay.inputs.job.MODEL_SHA,
        model_state_unchanged=True, measured_plane_constrained_estimator=True)
    if ('failure' in record or any(type(record.get(k)) is not type(v) or record[k] != v
            for k, v in expected.items())):
        raise ValueError('complete assigned chained native worker and unchanged model required')
    collection = record['collection']
    if (type(collection.get('navigation_ticks')) is not int or collection['navigation_ticks'] != 4000
            or collection['status'] != 'RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED'
            or type(collection.get('decisions')) is not int or not 1 <= collection['decisions'] <= 4014):
        raise ValueError('same bounded full native collection required')
    require_raw_audit(record, audit, learned=True)
    success = bool(audit['native_evaluation']['native_round_trip_candidate_pass']
        and audit['strict_physical_visibility_pass'] and not audit['hard_measurement_failed_frames'])
    if record['verified_round_trip'] is not success:
        raise ValueError('unchanged joint physical and strict sensing success criteria required')
    actual = prefix_result(collection, report, prior=prior, current=current)
    if prefix.run.canonical(record['prefix_comparison']) != prefix.run.canonical(actual):
        raise ValueError('saved dynamic intervention must reconstruct from complete actual raw evidence')
    if not actual['actual_paired_execution_compared'] and success:
        raise ValueError('retain early negative without an intervention or success claim')
    return actual
