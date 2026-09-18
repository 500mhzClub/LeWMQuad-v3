"""Paired 150/300-feature timing and post-estimation pose accuracy."""
import argparse
from contextlib import closing
import json
import time
import cv2
import numpy as np
import psutil
import torch

from lewm.feature_budget_150_tracker_development import FeatureBudget150VisualMotion
from lewm.feature_budget_300_tracker_development import FeatureBudget300VisualMotion
from lewm.feature_budget_100_tracker_development import FeatureBudget100VisualMotion
from lewm.physical_execution_development import rotation_xyzw
from lewm.joint_rgbd_rigid_pose_development import angle
from scripts import probe_delayed_action_models_development as source


def main(count,attempt=1,budgets=(150,300)):
    output=source.BASE/f'go2_feature{budgets[0]}_{budgets[1]}_{count}frames_recorded_v1_attempt_{attempt:03d}'
    if output.exists():raise ValueError('preserve existing experiment')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    output.mkdir()
    def write(name,value):
        with (output/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    owner=psutil.Process()
    write('launch.json',dict(owner=dict(pid=owner.pid,created=owner.create_time()),frames=count,
        input=str(source.INPUT),arms=list(budgets),alternating_execution_order=True,
        native_pose_used_for_estimation=False,geometry_gates_unchanged=True,shared_host=True))
    classes={100:FeatureBudget100VisualMotion,150:FeatureBudget150VisualMotion,300:FeatureBudget300VisualMotion}
    trackers={budget:classes[budget]() for budget in budgets}
    reader=source.source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
    auxiliary=json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
    records=[];failure=None;started=time.perf_counter()
    try:
        with closing(source.source.packets.read_rows(source.INPUT)) as recorded,(output/'frames.jsonl').open('x') as log:
            for frame in range(count):
                row=next(recorded)
                assert row['tick']==frame and row['pre_sample_index']==749+50*frame
                p,d,f,now=reader.packet(frame)
                rgb,aux=source.source.packets.rgb_packet(source.INPUT,frame,p,
                    source.source.public_acquisition(auxiliary[frame]),now_ns=now)
                record=dict(frame=frame,arms={})
                for budget in (budgets if frame%2==0 else tuple(reversed(budgets))):
                    began=time.perf_counter()
                    result=trackers[budget].observe(p,d,f,now_ns=now,auxiliary_rgb=rgb,auxiliary_depth=aux)
                    elapsed=time.perf_counter()-began
                    if result.get('current_pose') is None or result.get('failure') is not None:
                        write('tracker_failure_evidence.json',dict(frame=frame,budget=budget,evidence=result))
                        raise ValueError(f'{budget}-feature tracker failed at frame {frame}')
                    features=[result['last_accepted_feature_witness']['selected_features'],result['auxiliary_feature_witness']['selected_features']]
                    assert max(features)<=budget
                    record['arms'][str(budget)]=dict(service_s=elapsed,pose=result['current_pose'],features=features)
                records.append(record);log.write(json.dumps(record)+'\n');log.flush()
                if frame%100==0:print('PAIRED_FEATURE_FRAME',frame,flush=True)
    except BaseException as error:
        failure=repr(error);write('failure.json',dict(reason=failure,accepted_pairs=len(records)))
    # Estimation is over before opening evaluator-only native poses.
    with np.load(source.INPUT/'physics_trace.npz',allow_pickle=False) as raw:poses=raw['base_pose_world']
    origin=poses[749];R0=rotation_xyzw(origin[3:]);metrics={}
    for budget in map(str,budgets):
        errors=[];rotations=[];times=[]
        for row in records:
            actual=row['arms'][budget];native=poses[749+50*row['frame']]
            p=(native[:3]-origin[:3])@R0;R=R0.T@rotation_xyzw(native[3:]);pose=actual['pose']
            errors.append(float(np.linalg.norm(np.asarray(pose['position_initial_body_m'])-p)))
            rotations.append(angle(np.asarray(pose['rotation_initial_body_from_current_body']).T@R))
            if row['frame']>=3:times.append(actual['service_s'])
        metrics[budget]={k:dict(median=float(np.median(v)),maximum=max(v),total=sum(v))
            for k,v in dict(position_error_m=errors,rotation_error_rad=rotations,service_s=times).items() if v}
    report=dict(status='PAIRED_FEATURE_COMPARISON_COMPLETE' if failure is None else 'PAIRED_FEATURE_COMPARISON_FAILED',
        requested_pairs=count,accepted_pairs=len(records),failure=failure,metrics=metrics,
        native_pose_loaded_only_after_estimation=True,wall_s=time.perf_counter()-started,
        navigation_executed=False,real_time_qualified=False)
    if records:report['tracking_time_reduction_percent']=100*(1-metrics[str(budgets[0])]['service_s']['total']/metrics[str(budgets[1])]['service_s']['total'])
    write('result.json' if failure is None else 'partial_result.json',report)
    print(json.dumps(report),flush=True)
    if failure is not None:raise RuntimeError(failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--frames',type=int,choices=(201,3440),required=True)
    parser.add_argument('--attempt',type=int,choices=(1,2),default=1)
    parser.add_argument('--budgets',choices=('150,300','100,150'),default='150,300')
    args=parser.parse_args();main(args.frames,args.attempt,tuple(map(int,args.budgets.split(','))))
