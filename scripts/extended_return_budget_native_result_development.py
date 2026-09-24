"""Join complete native outcome, resource evidence and the actual budget prefix."""
from scripts import extended_return_budget_native_prefix_development as prefix
from scripts import extended_return_budget_resource_audit_development as resource_audit
from scripts import extended_return_budget_native_inputs_development as inputs
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit

run=inputs.run


def prefix_result(collection,admission,*,output,case,current_bindings):
    availability=prefix.prefix_availability(collection,admission['prefix_report'])
    if not availability['boundary_interval_available_by_collection_counts']:
        return dict(status='EXTENDED_BUDGET_INTERVENTION_NOT_COMPLETED',collection_availability=availability,
            full_raw_audit_retained=True,actual_paired_execution_compared=False,
            navigation_verified=False,unexecuted_outcomes_inferred=False)
    actual=prefix.compare(inputs.native.OUTPUT/inputs.native.CASE[0],output/case[0],admission['prefix_report'],
        prior_bindings=admission['original_native_artifact_sha256'],current_bindings=current_bindings,
        replay_bindings=admission['prefix_artifact_sha256'])
    return dict(status='COMPLETE_EXTENDED_BUDGET_ACTUAL_PREFIX',collection_availability=availability,
        full_raw_audit_retained=True,actual_paired_execution_compared=True,**actual)


def require_worker(record,audit,admission,*,output,case,worker_status,prefix_receipt=None):
    expected=dict(status=worker_status,case=case[0],layout_index=case[1],variant=case[2],condition=case[3],model_name=case[4],
        model_state_sha256=inputs.replay.pair.MODEL_SHA,model_state_unchanged=True,
        measured_plane_constrained_estimator=True,extended_return_budget_enabled=True,
        single_pass_timing_change_adopted=True,sampled_resource_guards_enabled=True)
    if 'failure' in record or any(type(record.get(k)) is not type(v) or record[k]!=v for k,v in expected.items()):
        raise ValueError('complete assigned longer native worker and unchanged model required')
    collection=record['collection']
    if (type(collection.get('navigation_ticks')) is not int or collection['navigation_ticks']!=8000
            or collection['status']!='RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED'
            or type(collection.get('decisions')) is not int or not 1<=collection['decisions']<=8014):
        raise ValueError('bounded complete 8000-step native collection required')
    require_raw_audit(record,audit,learned=True)
    success=bool(audit['native_evaluation']['native_round_trip_candidate_pass']
        and audit['strict_physical_visibility_pass'] and not audit['hard_measurement_failed_frames'])
    if record['verified_round_trip'] is not success:
        raise ValueError('unchanged joint physical and strict sensor success criteria required')
    resources=resource_audit.check(output,case[0],collection)
    if run.canonical(resources)!=run.canonical(record['resource_audit']):
        raise ValueError('both complete lifecycle resource audits must reconstruct')
    if prefix_receipt is None:
        prefix_receipt=prefix_result(collection,admission,output=output,case=case,
            current_bindings=record['artifact_sha256'])
    if run.canonical(prefix_receipt)!=run.canonical(record['prefix_comparison']):
        raise ValueError('saved physical prefix must reproduce the actual admitted intervention')
    if not prefix_receipt['actual_paired_execution_compared'] and success:
        raise ValueError('retain early negative without claiming an unverified intervention or success')
    for suffix,value in (('_audit.json',audit),('_resource_audit.json',resources),('_prefix_comparison.json',prefix_receipt)):
        if run.canonical(run.read_json(output,case[0]+suffix))!=run.canonical(value):
            raise ValueError('same complete closed worker evidence required: '+suffix)
    return prefix_receipt
