"""Replay recorded frames through floor reacquisition without navigation."""
import argparse
import json
import time
import cv2
import torch
from lewm.retained_depth_cache_tracking_development import RetainedDepthCacheMotion
from lewm.coherent_reference_refresh_development import CoherentReferenceRefreshMotion
from lewm.floor_reacquisition_development import ReacquiringFloorRegistration,UNAVAILABLE
from lewm.partial_floor_height_development import read_pose
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import path
from scripts.diagnose_go2_depth_noise_failures_development import save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root-name',default='go2_current_plane_heading_reactive_noise_2mm_native_layout03_4800_v1_attempt_001')
    parser.add_argument('--tracker',choices=('baseline','coherent_refresh'),default='baseline')
    args=parser.parse_args();root=path(args.root_name)
    if (root/'depth_retention.json').exists():raise ValueError('full retained recording required')
    output=root/'floor_reacquisition_replay_v1';output.mkdir()
    recorded=json.loads((root/'poses.json').read_text())
    count=len(json.loads((root/'native/in_memory_camera_observations.json').read_text())['frames'])
    configure();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=NoisyPublicReplay(root/'native')
    motion={'baseline':RetainedDepthCacheMotion,'coherent_refresh':CoherentReferenceRefreshMotion}[args.tracker]()
    registration=ReacquiringFloorRegistration();rows=[];matched=0;started=time.monotonic()
    for frame in range(count):
        p,d,fast,rgb,a,now=reader.packet(frame)
        raw=motion.observe(p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=a,now_ns=now)
        evidence=registration.observe(p,d,a,raw,now_ns=now)
        missing=evidence.get('status')==UNAVAILABLE
        if not missing:read_pose(evidence,identity=(0,0,0),now_ns=now)
        if frame<len(recorded):
            for label,value in [('raw_pose',raw['current_pose']),('registered_pose',evidence['current_pose'])]:
                for k in ('frame','measured_ns','reference_frame','position_initial_body_m',
                        'rotation_initial_body_from_current_body','rgb_sha256','depth_sha256'):
                    if value[k]!=recorded[frame][label][k]:
                        raise ValueError(f'original pose changed at {frame}: {label}.{k}')
            matched+=1
        rows.append(dict(frame=frame,status=evidence['status'],pose_available=not missing,
            registration_frame=registration.frame,registration_failed=registration.failed))
        if frame%100==0:print('REACQUISITION_REPLAY_FRAME',frame,flush=True)
    result=dict(tracker=args.tracker,recorded_frames=count,consumed_frames=len(rows),matched_original_pose_pairs=matched,
        accepted_frames=sum(r['pose_available'] for r in rows),
        unavailable_frames=[r['frame'] for r in rows if not r['pose_available']],
        final_registration_failed=registration.failed,elapsed_s=time.monotonic()-started,
        rows=rows,controller_executed=False,physics_executed=False,native_pose_used=False,
        recorded_trajectory_unchanged=True,live_recovery_demonstrated=False)
    save(output,'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
