"""Recover causal motion inputs for fixed training and development populations.

Public RGB-D/gyro only; no physics trace, native pose or future packet read.
Two-mm synthetic depth noise uses the same per-pixel recipe as live studies.
"""
from collections import defaultdict,Counter
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from lewm.causal_local_rgbd_motion_development import CausalLocalRGBDMotion
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.pre_switch_training_data_development import BASE,ROOTS,load_training_rows
from scripts.prepare_go2_pre_switch_transfer_development import OUTPUT as TRANSFER

OUTPUT=BASE/'go2_causal_local_motion_inputs_v1_attempt_001'
NOISE_SEED=2026091414


def noisy_depth(depth,*,layout,frame):
    rng=np.random.default_rng(np.random.SeedSequence([NOISE_SEED,layout,frame,0]))
    values=depth['depth_m'].copy();valid=depth['valid'].copy()
    values[valid]+=rng.normal(0.,.002,int(valid.sum())).astype(np.float32)
    valid&=(values>=.2)&(values<=5.);values[~valid]=0.
    return depth|dict(depth_m=values,valid=valid)


def main():
    if OUTPUT.exists():raise ValueError('preserve causal feature derivation')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    train=load_training_rows();transfer=[r for r in json.loads((TRANSFER/'windows.json').read_text()) if r['available']]
    if len(train)!=4514 or len(transfer)!=924:raise ValueError('fixed sample populations required')
    groups=defaultdict(list)
    for row in train+transfer:groups[row['source'],row['trial']].append(row)
    OUTPUT.mkdir();identities={};reports=[];counts=Counter();elapsed=[];started=time.monotonic()
    launch=dict(train_contexts=len(train),transfer_contexts=len(transfer),recordings=len(groups),
        input_definition='three consecutive RGBD/gyro pair increments, four causal frames',
        depth_noise_sigma_m=.002,depth_noise_seed=NOISE_SEED,
        ideal_recorded_gyro=True,hardware_noise_calibrated=False,global_navigation_pose_reset=False,
        native_state_input=False,future_packet_input=False,
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
            (__file__,'lewm/causal_local_rgbd_motion_development.py')})
    (OUTPUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    try:
        with (OUTPUT/'features.jsonl').open('x') as sink:
            for (source,trial),selected in sorted(groups.items()):
                root=ROOTS[source]/trial
                role=selected[0]['data_role'];cluster=selected[0]['cluster']
                if any(r['data_role']!=role or r['cluster']!=cluster for r in selected):
                    raise ValueError('one role and cluster per recording')
                meta_name='branch_specification.json' if source=='switch' else 'specification.json'
                meta=json.loads((root/meta_name).read_text())
                geometry=meta['geometry'] if source=='switch' else meta['layout_id']
                appearance=meta['appearance_seed']
                layout=int(cluster.split('_')[1])*4+2*int(geometry.endswith('right_open'))+(appearance-2026090940)
                desired={r['observation_horizon_receipt']['departure_tick']:r for r in selected}
                if len(desired)!=len(selected):raise ValueError('distinct causal departure frames')
                maximum=max(desired);observer=CausalLocalRGBDMotion();failures=[];available=0
                for name in (meta_name,'policy_observations.json','policy_histories.npz',
                             'depth_observations.json','fast_gyro_histories.npz'):
                    identities[str((root/name).relative_to(BASE))]=hashlib.sha256((root/name).read_bytes()).hexdigest()
                for frame in range(maximum+1):
                    for name in (f'rgb_{frame:04d}.png',f'depth_{frame:04d}.npz'):
                        identities[str((root/name).relative_to(BASE))]=hashlib.sha256((root/name).read_bytes()).hexdigest()
                    policy,depth=load_rgbd_observation(root,frame);fast=load_fast_packet(root,frame)
                    before=time.perf_counter()
                    value=observer.observe(policy,noisy_depth(depth,layout=layout,frame=frame),fast,
                                           now_ns=policy['sensor_state']['decision_ns'])
                    elapsed.append(time.perf_counter()-before)
                    if value['pair_failure']:failures.append(dict(frame=frame,reason=value['pair_failure']))
                    if frame in desired:
                        row=desired[frame]
                        receipt=row['observation_horizon_receipt']
                        if value['measured_ns']!=receipt['departure_ns']:
                            raise ValueError('same observation-time feature required')
                        record=dict(sample_id=row['sample_id'],source=source,trial=trial,data_role=role,
                            cluster=cluster,frame=frame,measured_ns=value['measured_ns'],
                            history_available=value['history_available'],history_features=value['history_features'],
                            history_frames=list(range(frame-3,frame+1)),noise_layout_key=layout)
                        sink.write(json.dumps(record)+'\n')
                        counts[role+'_contexts']+=1;counts[role+'_available']+=value['history_available']
                        available+=value['history_available']
                sink.flush()
                reports.append(dict(source=source,trial=trial,data_role=role,contexts=len(selected),
                    available_histories=available,frames_processed=maximum+1,pair_failures=failures))
                if len(reports)%10==0:print('LOCAL_MOTION_RECORDINGS',len(reports),dict(counts),flush=True)
        report=dict(status='COMPLETE',**counts,recordings=len(reports),trials=reports,
            frames_processed=len(elapsed),pair_failures=sum(len(r['pair_failures']) for r in reports),
            measured_feature_ms=dict(median=1000*float(np.median(elapsed)),p95=1000*float(np.percentile(elapsed,95)),
                maximum=1000*float(max(elapsed))),
            input_sha256=identities,features_sha256=hashlib.sha256((OUTPUT/'features.jsonl').read_bytes()).hexdigest(),
            wall_s=time.monotonic()-started,missing_features_are_null=True,missingness_not_filled_from_native_state=True,
            full_loop_timing_established=False,model_trained=False)
        (OUTPUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('trials','input_sha256')},indent=2))
    except Exception as error:
        (OUTPUT/'failure.json').write_text(json.dumps(dict(reason=repr(error),completed_recordings=len(reports),counts=dict(counts))))
        raise


if __name__=='__main__':main()
