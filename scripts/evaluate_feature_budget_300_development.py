"""Speed and evaluator-only pose error for a smaller feature population."""
from contextlib import closing
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

from lewm.batched_patch_tracker_development import BatchedPatchVisualMotion
from lewm.feature_budget_300_tracker_development import FeatureBudget300VisualMotion
from lewm.physical_execution_development import rotation_xyzw
from lewm.joint_rgbd_rigid_pose_development import angle
from scripts.compare_full_consensus_recorded_tracker_development import without_work_counts
from scripts import probe_delayed_action_models_development as source

COUNT=201
OUTPUT=source.BASE/'go2_feature_budget300_recorded_prefix_v1_attempt_001'


def write(name,value):
    with (OUTPUT/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    assert not OUTPUT.exists() and psutil.virtual_memory().available>16*1024**3
    assert json.loads((source.ROOT/'result.json').read_text())['verified_round_trip'] is True
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    OUTPUT.mkdir();owner=psutil.Process()
    write('launch.json',dict(owner=dict(pid=owner.pid,created=owner.create_time()),observations=COUNT,
        input=str(source.INPUT),paired_first_observations=61,corner_budget=300,per_cell=25,
        native_pose_loaded_only_after_tracking=True,source_result_sha256=source.source.digest(source.ROOT/'result.json'),
        sources={p:source.source.digest(Path(p)) for p in ('lewm/feature_budget_300_tracker_development.py',
            'scripts/evaluate_feature_budget_300_development.py')},native_execution=False,automatic_retry=False))
    candidate=FeatureBudget300VisualMotion();baseline=BatchedPatchVisualMotion()
    reader=source.source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
    auxiliary=json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
    records=[];failure=None;started_all=time.perf_counter()
    try:
        with closing(source.source.packets.read_rows(source.INPUT)) as recorded,(OUTPUT/'frames.jsonl').open('x') as log:
            for frame in range(COUNT):
                row=next(recorded);assert row['tick']==frame
                expected=row['decision']['original_visual_evidence']
                p,d,f,now=reader.packet(frame)
                rgb,aux=source.source.packets.rgb_packet(source.INPUT,frame,p,
                    source.source.public_acquisition(auxiliary[frame]),now_ns=now)
                arms=[('candidate',candidate)]
                if frame<61:
                    arms.append(('baseline',baseline))
                    if frame%2:arms.reverse()
                times={};actual=None
                for label,tracker in arms:
                    started=time.perf_counter()
                    result=tracker.observe(p,d,f,now_ns=now,auxiliary_rgb=rgb,auxiliary_depth=aux)
                    times[label]=time.perf_counter()-started
                    if result.get('current_pose') is None or result.get('failure') is not None:
                        write('tracker_failure_evidence.json',dict(frame=frame,arm=label,evidence=result))
                        raise ValueError(f'{label} pose unavailable at frame {frame}')
                    if label=='baseline':
                        assert without_work_counts(json.loads(json.dumps(result)))==without_work_counts(expected)
                    else:actual=result
                pose=actual['current_pose'];old=expected['current_pose']
                record=dict(frame=frame,measured_ns=now,pre_sample_index=row['pre_sample_index'],
                    candidate_s=times['candidate'],baseline_s=times.get('baseline'),
                    position_initial_body_m=pose['position_initial_body_m'],
                    rotation_initial_body_from_current_body=pose['rotation_initial_body_from_current_body'],
                    original_position_initial_body_m=old['position_initial_body_m'],
                    original_rotation_initial_body_from_current_body=old['rotation_initial_body_from_current_body'],
                    primary_features=actual['last_accepted_feature_witness']['selected_features'],
                    auxiliary_features=actual['auxiliary_feature_witness']['selected_features'])
                assert record['primary_features']<=300 and record['auxiliary_features']<=300
                records.append(record);log.write(json.dumps(record)+'\n');log.flush()
                if frame%100==0:print('FEATURE_BUDGET300_FRAME',frame,flush=True)
    except BaseException as error:
        failure=dict(reason=repr(error),accepted_observations=len(records),automatic_retry=False)
        write('failure.json',failure)
    # Native state is first accessed here, after estimation has stopped.
    with np.load(source.INPUT/'physics_trace.npz',allow_pickle=False) as raw:poses=raw['base_pose_world']
    origin=poses[749];R0=rotation_xyzw(origin[3:])
    for record in records:
        assert record['pre_sample_index']==749+50*record['frame']
        native=poses[record['pre_sample_index']];p=(native[:3]-origin[:3])@R0
        R=R0.T@rotation_xyzw(native[3:])
        for label,prefix in (('candidate',''),('original','original_')):
            record[label+'_position_error_m']=float(np.linalg.norm(np.asarray(record[prefix+'position_initial_body_m'])-p))
            record[label+'_rotation_error_rad']=angle(np.asarray(record[prefix+'rotation_initial_body_from_current_body']).T@R)
    paired=[r for r in records if r['frame']>=3 and r['baseline_s'] is not None]
    old=sum(r['baseline_s'] for r in paired);new=sum(r['candidate_s'] for r in paired)
    metrics={k:dict(median=float(np.median([r[k] for r in records])),maximum=max(r[k] for r in records))
        for k in ('candidate_position_error_m','original_position_error_m','candidate_rotation_error_rad',
            'original_rotation_error_rad','candidate_s')} if records else {}
    report=dict(status='FEATURE_BUDGET300_RECORDED_COMPLETE' if failure is None else 'FEATURE_BUDGET300_RECORDED_FAILED',
        requested_observations=COUNT,accepted_observations=len(records),failure=failure,records=records,metrics=metrics,
        paired_timed_observations=len(paired),paired_baseline_s=old,paired_candidate_s=new,
        paired_reduction_percent=None if old==0 else 100*(1-new/old),wall_s=time.perf_counter()-started_all,
        native_pose_loaded_only_after_tracking=True,native_pose_used_for_estimation=False,
        native_trace_sha256=source.source.digest(source.INPUT/'physics_trace.npz'),
        geometric_and_temporal_gate_values_unchanged=True,pose_equality_to_600_features_claimed=False,
        shared_host=True,acquisition_included_in_timing=False,navigation_executed=False,real_time_qualified=False)
    write('result.json' if failure is None else 'partial_result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    if failure is not None:raise RuntimeError(failure['reason'])


if __name__=='__main__':main()
