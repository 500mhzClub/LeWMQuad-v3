"""Fixed paired turn assay semantics; raw physical pose is evaluation only."""
import math
import numpy as np

from lewm.physical_execution_development import rotation_xyzw

TARGETS=(('left90',math.pi/2),('right90',-math.pi/2),('left180',math.pi))


def timed_decision(tick,target):
    active_ticks=math.ceil(abs(target)/(.35*.1))
    return {'status':'TURNING_TIMED' if tick<active_ticks else ('SETTLING_TIMED' if tick<active_ticks+3 else 'COMPLETE_TIMED'),
        'requested_command':[0.,0.,math.copysign(.35,target) if tick<active_ticks else 0.]}


def reduce_turn(raw,start,decisions,target,controller_terminal,stop_reason):
    origin=raw['base_pose_world'][start]
    initial=rotation_xyzw(origin[3:]); final=rotation_xyzw(raw['base_pose_world'][-1,3:])
    relative=initial.T@final
    yaw=math.atan2(relative[1,0],relative[0,0]); error=abs(math.atan2(math.sin(target-yaw),math.cos(target-yaw)))
    drift=np.linalg.norm(raw['base_pose_world'][start:,:2]-origin[:2],axis=1)
    release=np.flatnonzero(raw['phase']==2)
    window=release[-100:]
    settled=bool(len(release)==250 and len(window)==100
        and np.all(np.linalg.norm(raw['base_twist_world'][window,:2],axis=1)<=.1)
        and np.all(np.abs(raw['base_twist_world'][window,5])<=.25))
    roll=math.atan2(final[2,1],final[2,2]); pitch=math.atan2(-final[2,0],math.hypot(final[2,1],final[2,2]))
    estimates=[]
    for row in decisions:
        output=row['controller']
        if output.get('relative_heading_rad') is None: continue
        measured=initial.T@rotation_xyzw(raw['base_pose_world'][row['pre_sample_index'],3:])
        truth=math.atan2(measured[1,0],measured[0,0]); delta=output['relative_heading_rad']-truth
        estimates.append(abs(math.atan2(math.sin(delta),math.cos(delta))))
    checks={'controller_finished':controller_terminal in ('COMPLETE','COMPLETE_TIMED'),
        'no_physical_stop':stop_reason is None,'no_contact':not bool(raw['physics_contact'].any()),
        'heading_error_at_most_0p12':error<=.12,'maximum_xy_drift_at_most_0p15':float(drift.max())<=.15,
        'release_motion':settled,'terminal_body_stable':bool(raw['base_pose_world'][-1,2]>=.2 and abs(roll)<=.5 and abs(pitch)<=.5)}
    return {'checks':checks,'physical_task_success':all(checks.values()),'final_relative_heading_rad':yaw,
        'final_heading_error_rad':error,'maximum_xy_drift_m':float(drift.max()),
        'maximum_gyro_heading_error_rad':max(estimates) if estimates else None,
        'gyro_reference_agreement':max(estimates)<=.04 if estimates else None,
        'controller_terminal':controller_terminal,'stop_reason':stop_reason,
        'total_post_settle_seconds':float(raw['timestamp_s'][-1]-raw['timestamp_s'][start])}
