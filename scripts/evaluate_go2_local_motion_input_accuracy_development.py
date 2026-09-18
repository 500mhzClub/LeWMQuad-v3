"""Post-hoc accuracy of public motion features; native state is truth only."""
from collections import defaultdict
import json
import numpy as np
from lewm.physical_execution_development import rotation_xyzw
from scripts.pre_switch_training_data_development import ROOTS
from scripts.derive_go2_causal_local_motion_inputs_development import OUTPUT


def main():
    destination=OUTPUT/'posthoc_feature_accuracy.json'
    if destination.exists():raise ValueError('preserve feature accuracy result')
    result=json.loads((OUTPUT/'result.json').read_text())
    if result['status']!='COMPLETE':raise ValueError('freeze features before native accuracy evaluation')
    groups=defaultdict(list)
    for line in (OUTPUT/'features.jsonl').read_text().splitlines():
        row=json.loads(line);groups[row['source'],row['trial']].append(row)
    errors=defaultdict(list);missing=defaultdict(int)
    for (source,trial),rows in groups.items():
        with np.load(ROOTS[source]/trial/'physics_trace.npz',allow_pickle=False) as archive:
            pose=archive['base_pose_world']
            for row in rows:
                role=row['data_role']
                if not row['history_available']:
                    missing[role]+=1;continue
                frame=row['frame'];now=749+50*frame
                R=rotation_xyzw(pose[now,3:]);origin=pose[now,:3]
                predicted=np.asarray(row['history_features']).reshape(3,4)
                values=[]
                for i,previous in enumerate(range(frame-3,frame)):
                    at=749+50*previous
                    delta=R.T@(pose[at,:3]-origin);relative=R.T@rotation_xyzw(pose[at,3:])
                    yaw=np.arctan2(relative[1,0],relative[0,0])
                    measured_yaw=np.arctan2(predicted[i,2],1+predicted[i,3])
                    difference=measured_yaw-yaw
                    values.append([float(np.linalg.norm(predicted[i,:2]-delta[:2])),
                                   float(np.arctan2(np.sin(difference),np.cos(difference)))])
                errors[role].append(values)
    summary={}
    for role,values in errors.items():
        v=np.asarray(values)
        summary[role]=dict(available=len(v),missing=missing[role],
            by_past_lag_ms={str(lag):dict(xy_rmse_mm=1000*float(np.sqrt(np.mean(v[:,i,0]**2))),
                xy_p95_mm=1000*float(np.percentile(v[:,i,0],95)),xy_max_mm=1000*float(v[:,i,0].max()),
                yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(v[:,i,1]**2)))))
                for i,lag in enumerate((300,200,100))})
    report=dict(status='COMPLETE',summary=summary,features_sha256=result['features_sha256'],
        observer_received_native_state=False,accuracy_evaluated_only_after_features_frozen=True,
        unavailable_histories_excluded_only_from_error_metrics=True,
        no_error_threshold_used_to_select_features=True,calibrated_error_bound_established=False)
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
