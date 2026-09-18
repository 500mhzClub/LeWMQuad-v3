"""Evaluator-only targets for forecast prefixes whose commands really executed."""
import math
import numpy as np
from lewm.physical_execution_development import rotation_xyzw


def diagnose(prediction,planned_commands,tape,poses,*,tick):
    prediction=np.asarray(prediction,float);poses=np.asarray(poses,float)
    planned=np.asarray(planned_commands,float)
    if (prediction.shape!=(8,5) or planned.shape!=(8,3) or poses.ndim!=2 or poses.shape[1]!=7
            or not np.isfinite(prediction).all() or not np.isfinite(planned).all()
            or not np.isfinite(poses).all() or type(tick) is not int or tick<0):
        raise ValueError('finite eight-horizon forecast, plan and native evaluator poses required')
    start=749+50*tick
    if start>=len(poses):raise ValueError('forecast start sample missing')
    R=rotation_xyzw(poses[start,3:]);results=[]
    for h,command in enumerate(planned):
        j=tick+h;end=start+50*(h+1)
        if j>=len(tape):break
        row=tape[j]
        if row['tick']!=j or row['pre_sample_index']!=749+50*j:
            raise ValueError('exact recorded command clock required')
        if (not row['completed'] or row['post_sample_index']!=799+50*j
                or not np.array_equal(np.asarray(row['requested_command'],float),command)):break
        if end>=len(poses):raise ValueError('completed command lacks native endpoint')
        actual=R.T@(poses[end,:3]-poses[start,:3])
        relative=R.T@rotation_xyzw(poses[end,3:]);yaw=math.atan2(relative[1,0],relative[0,0])
        pyaw=math.atan2(prediction[h,2],prediction[h,3])
        results.append(dict(horizon_ns=(h+1)*100_000_000,start_sample=start,end_sample=end,
            actual_body_xy_m=actual[:2].tolist(),predicted_body_xy_m=prediction[h,:2].tolist(),
            xy_residual_predicted_minus_actual_m=(prediction[h,:2]-actual[:2]).tolist(),
            xy_error_m=float(np.linalg.norm(prediction[h,:2]-actual[:2])),actual_yaw_rad=yaw,
            predicted_yaw_rad=pyaw,yaw_error_rad=float((pyaw-yaw+math.pi)%(2*math.pi)-math.pi),
            complete_executed_command_prefix_exact=True))
    return results
