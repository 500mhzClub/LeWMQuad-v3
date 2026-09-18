"""Separate unchanged observed state from the original failure's null display."""
FAILURE = 'training-only translation wrapper in evaluation mode required'


def compare_observed(old, new, *, frame):
    if type(frame) is not int or not 0 <= frame <= 3:
        raise ValueError('fixed four-observation adapter prefix required')
    for key in ('evidence', 'original_visual_evidence', 'memory_receipt', 'mission_receipt',
            'floor_partition_receipt', 'auxiliary_floor_partition_receipt'):
        if old[key] != new[key]:
            raise ValueError('observed state changed outside model interface: '+key)
    if frame < 3:
        if old != new: raise ValueError('complete original warmup decisions required')
    elif (old['terminal'] != 'SENSOR_OR_MODEL_FAILURE' or old['failure'] != FAILURE
            or old['observed_goal_distance_m'] is not None or new['terminal'] is not None
            or new['observed_goal_distance_m'] != new['mission_receipt']['observed_goal_distance_m']):
        raise ValueError('original failure-null and unchanged measured mission distance required')
    return dict(observed_pose_map_contact_and_mission_exact=True,
        original_failure_display_distance=old['observed_goal_distance_m'],
        candidate_display_distance=new['observed_goal_distance_m'],
        failure_display_distance_is_observed_pose=False)
