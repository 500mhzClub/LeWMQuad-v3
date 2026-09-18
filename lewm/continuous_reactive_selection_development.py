"""Instantaneous six-action feedback for the continuous comparison."""
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.six_action_reactive_controller_development import MAX_FORWARD,MAX_YAW,HEADING_GAIN


def select_reactive(goal_body_xy, *, scan_error=None, current_clearance_m=None):
    goal=np.asarray(goal_body_xy,float)
    if goal.shape!=(2,) or not np.isfinite(goal).all():
        raise ValueError('finite currently observed waypoint direction required')
    if scan_error is not None and not math.isfinite(scan_error):
        raise ValueError('finite current measured viewing error required')
    if current_clearance_m is not None and (not math.isfinite(current_clearance_m) or current_clearance_m<0):
        raise ValueError('finite nonnegative observed clearance required')
    clear=current_clearance_m is None or current_clearance_m>.45+1e-12
    translation=scan_error is None and np.linalg.norm(goal)>1e-8
    error=math.atan2(goal[1],goal[0]) if scan_error is None else scan_error
    desired=[0.,0.,0.]
    if clear and (translation or scan_error is not None):
        desired[0]=MAX_FORWARD*max(0.,math.cos(error)) if translation else 0.
        desired[2]=float(np.clip(HEADING_GAIN*error,-MAX_YAW,MAX_YAW))
    rows=[]
    for action in ACTIONS:
        command=candidate_commands(action)[0]
        eligible=clear and (not any(command[:2]) or translation)
        distance=((command[0]-desired[0])/MAX_FORWARD)**2+((command[2]-desired[2])/MAX_YAW)**2
        rows.append(dict(action=action,requested_command=command,eligible=bool(eligible),
            normalized_command_distance_squared=float(distance)))
    eligible=[i for i,row in enumerate(rows) if row['eligible']]
    index=min(eligible,key=lambda i:rows[i]['normalized_command_distance_squared']) if eligible else ACTIONS.index('hold')
    action=ACTIONS[index]
    return dict(action=action,action_index=index,requested_command=candidate_commands(action)[0],
        candidates=rows,waypoint_body_xy_m=goal.tolist(),scan_heading_error_rad=scan_error,
        desired_instantaneous_command=desired,current_stored_clearance_m=current_clearance_m,
        current_nominal_disk_clear=clear,heading_gain_per_s=HEADING_GAIN,
        learned_model_used=False,candidate_future_outcomes_evaluated=False,
        command_integrated_pose_used=False,predicted_feasibility_used=False,
        current_depth_dispatch_check_still_required=True,
        rule='nearest_six_action_primitive_to_current_waypoint_feedback')
