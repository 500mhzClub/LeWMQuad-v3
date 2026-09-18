"""Independent matrix accounting for joint continuity's saved witnesses.

No sensor inference or native truth. Checks composition against actual earlier
accepted poses and recomputes reported rotation disagreement; it cannot prove
the correctness of correspondence matches or calibrate gyro/depth uncertainty.
"""
import numpy as np
from lewm.independent_tracking_numerical_verification_development import require, same, _rotation, _angle


def verify_rotations(rows):
    accepted = {}; checked = paired = 0; failure = None; count = 0
    for row in rows:
        frame = row['frame']; arm = row['arms']['temporal_anchor']
        require(frame == count and count < 443, 'complete bounded joint witness chronology')
        count += 1
        pose, e = arm['pose'], arm['continuity']
        if failure is not None:
            require(pose is None and arm['failure'] == failure, 'no joint resurrection or rewritten failure')
            continue
        if pose is None:
            require(arm['failure'] is not None, 'explicit joint failure required')
            failure = arm['failure']
        if isinstance(e, dict) and 'rotation_measurement_witnesses' in e:
            witnesses = e['rotation_measurement_witnesses']
            require(type(witnesses) is list and len(witnesses) <= 9
                and e['rotation_fitting_mode'] == 'joint' and e['gyro_bias_estimated'] is False,
                'bounded explicit joint fitting witnesses')
            for w in witnesses:
                ref = w['reference_frame']
                require(ref in accepted and w['current_frame'] == frame
                    and w['reference_measured_ns'] == accepted[ref]['measured_ns']
                    and w['fitting_mode'] == 'joint' and w['candidate_envelope_passed'] is True
                    and w['witness_alone_grants_pose'] is False, 'qualified actual historical reference required')
                reference = _rotation(w['reference_rotation_initial_body_from_reference_body'])
                relative = _rotation(w['fitted_rotation_reference_body_from_current_body'])
                composed = _rotation(w['composed_rotation_initial_body_from_current_body'])
                gyro = _rotation(w['gyro_rotation_reference_body_from_current_body'])
                same(reference.tolist(), accepted[ref]['rotation_initial_body_from_current_body'], tolerance=1e-12)
                same((reference @ relative).tolist(), composed.tolist(), tolerance=1e-12)
                same(_angle(gyro.T @ relative), w['gyro_disagreement_rad'], tolerance=1e-12)
                require(w['gyro_disagreement_rad'] <= .10, 'joint gyro consistency gate retained')
                checked += 1
            anchor = e['selected_anchor_rotation_witness']; increment = e['incremental_rotation_witness']
            require((anchor is not None) == e['anchor_available']
                and (increment is not None) == e['incremental_available']
                and e['incremental_rotation_witness_saved'] == (increment is not None),
                'actual anchor/increment availability and witnesses agree')
            for w in (anchor, increment):
                if w is not None: require(w in witnesses, 'selected witness must be a recorded qualified fit')
            if increment is not None:
                require(increment['reference_frame'] == frame - 1, 'immediate previous rotation required')
            if anchor is not None and increment is not None:
                a = _rotation(anchor['composed_rotation_initial_body_from_current_body'])
                b = _rotation(increment['composed_rotation_initial_body_from_current_body'])
                same(_angle(a.T @ b), e['disagreement_rad'], tolerance=1e-12)
                paired += 1
            if pose is not None:
                selected = anchor if anchor is not None else increment
                require(selected is not None, 'accepted noninitial pose needs measured rotation')
                same(selected['composed_rotation_initial_body_from_current_body'],
                     pose['rotation_initial_body_from_current_body'], tolerance=1e-12)
                same(selected['position_initial_body_m'], pose['position_initial_body_m'], tolerance=1e-12)
        if pose is not None:
            require(pose['mode'] == 'joint' and pose['gyro_role'] == 'consistency_monitor_only', 'joint pose semantics')
            accepted[frame] = pose
    return dict(frames=count, qualified_rotation_witnesses_checked=checked,
        anchor_increment_rotation_disagreements_recomputed=paired,
        pose_history_composition_verified=True, correspondence_inference_recomputed=False,
        native_truth_used=False, gyro_bias_estimated=False, navigation_qualified=False)
