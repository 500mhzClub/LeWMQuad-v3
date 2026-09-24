"""Require completed audits and the same closed collection before prospective use."""
from scripts.dual_camera_native_prefix_comparison_development import prefix_shape


def admit_prefix(prefix):
    prefix_shape(prefix)
    if (prefix['status']!='DUAL_CAMERA_CONTROLLER_PREFIX_V2_COMPLETE'
            or prefix['final_terminal'] is not None or prefix['final_failure'] is not None
            or prefix['final_registered_pose_available'] is not True
            or prefix['following_recorded_observations_consumed'] is not False
            or prefix['native_audit_replaced'] is not False or prefix['native_execution'] is not False):
        raise ValueError('successful bounded camera intervention required before new native execution')
    for key in ('all_preintervention_requested_commands_exact',
            'complete_preintervention_decisions_exact_outside_added_modality_metadata',
            'model_state_unchanged','failed_json_identity_attempt_preserved','controller_implementation_unchanged',
            'completed_native_audit_and_matching_collection_bindings_required_before_next_native'):
        if prefix[key] is not True:raise ValueError('required completed prefix invariant: '+key)


def admit_predecessor(native, audit, prefix_launch, *, case):
    if (native['status']!='SETTLED_BOUNDARY_MAZE_PILOT_V2_COMPLETE' or len(native['conditions'])!=1
            or native['conditions'][0]['case']!=case
            or native['conditions'][0]['status']!='SETTLED_BOUNDARY_MAZE_COLLECTED_AND_RAW_AUDITED'
            or native['conditions'][0]['prefix_comparison']['physical_and_public_prefix_exact'] is not True):
        raise ValueError('completed tenth native audit and physical prefix required')
    for key in ('raw_sensor_reconstruction_pass','raw_model_command_replay_pass','raw_command_audit_pass','model_state_unchanged'):
        if audit[key] is not True:raise ValueError('completed predecessor raw invariant: '+key)
    if audit['verified_round_trip'] is not False or audit['strict_physical_visibility_pass'] is not False:
        raise ValueError('preserve the predecessor failed qualification outcome')
    bindings=prefix_launch['collected_input_sha256']
    if not bindings or 'launch.json' not in bindings or case+'/result.json' not in bindings:
        raise ValueError('explicit native launch and closed-collection bindings required')
    for name,h in bindings.items():
        if native['artifact_sha256'].get(name)!=h:
            raise ValueError('final native audit differs from prefix collection: '+name)
    return dict(completed_native_audit_matched_all_prefix_collection_bindings=True,
        matched_artifact_count=len(bindings),predecessor_qualification_failure_preserved=True)
