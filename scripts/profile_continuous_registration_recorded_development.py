"""Replay only saved public sensors from the failed continuous mission."""
import cProfile
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import io
import json
from pathlib import Path
import pstats
import time
import cv2
import numpy as np
from PIL import Image
import torch
from lewm.simulated_body_observation_development import SCHEMAS,CAMERA_CALIBRATION
from lewm.causal_depth_observation_development import from_native_depth
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth as auxiliary_depth
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.fast_gyro_development import SCHEMA_ID,SCHEMA as FAST_SCHEMA,CALIBRATION
from lewm.feature_budget_150_tracker_development import FeatureBudget150VisualMotion
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneFloorRegistration


def main(executor=None,routing=False):
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_continuous_round_trip_native_layout00_v1_attempt_003')
    prefix='registration_process_profile' if executor is not None else 'registration_profile'
    if routing:prefix='routing_profile'
    directory=root/'native';output=root/(prefix+'.json')
    if output.exists():raise ValueError('preserve completed profile')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    with np.load(directory/'policy_histories.npz',allow_pickle=False) as z:body={k:z[k] for k in z.files}
    with np.load(directory/'fast_gyro_histories.npz',allow_pickle=False) as z:fast={k:z[k] for k in z.files}
    expected=json.loads((root/'poses.json').read_text())
    tracker=FeatureBudget150VisualMotion();registration=SampledPlaneFloorRegistration()
    if executor is not None:
        from lewm.process_registered_round_trip_development import RegistrationProxy
        registration=RegistrationProxy(executor)
    profile=cProfile.Profile();rows=[]
    if routing:
        from lewm.multirate_routing_map_development import MultirateRoutingMap
        from lewm.observed_floor_waypoint_development import propose
        mapper=MultirateRoutingMap()
    for frame in range(len(expected)):
        now=int(body['decision_ns'][frame])
        state=dict(identity=(0,0,0),image_ns=now,decision_ns=now,sensor_anchor='decision',sensor_anchor_ns=now,sensed={},control={})
        for s in SCHEMAS:
            state[s.role][s.name]={**{k:body[f'{s.name}_{k}'][frame].copy() for k in ('values','valid','measured_ns','available_ns')},
                'channels':s.channels,'units':s.units,'calibration_id':s.calibration_id}
        with Image.open(directory/f'rgb_{frame:04d}.png') as image:rgb=np.array(image)
        p=dict(image=dict(rgb=rgb,measured_ns=now,available_ns=now,calibration_id=CAMERA_CALIBRATION),sensor_state=state)
        with np.load(directory/f'primary_depth_{frame:04d}.npz',allow_pickle=False) as z:
            d=from_native_depth(z['native_optical_depth_m'],p,measured_ns=now,available_ns=now,now_ns=now)
        with np.load(directory/f'auxiliary_depth_{frame:04d}.npz',allow_pickle=False) as z:
            a=auxiliary_depth(z['native_optical_depth_m'],p,measured_ns=now,available_ns=now,now_ns=now)
        with Image.open(directory/f'auxiliary_rgb_{frame:04d}.png') as image:aux_rgb=np.array(image)
        rgb=from_captured_rgb(aux_rgb,a,p,measured_ns=now,available_ns=now,now_ns=now)
        f={k:v[frame].copy() for k,v in fast.items()}|dict(schema=SCHEMA_ID,identity=(0,0,0),decision_ns=now,
            calibration_id=CALIBRATION,channels=FAST_SCHEMA.channels,units=FAST_SCHEMA.units)
        raw=tracker.observe(p,d,f,now_ns=now,auxiliary_rgb=rgb,auxiliary_depth=a)
        assert raw['current_pose']==expected[frame]['raw_pose']
        began=time.perf_counter()
        if not routing:profile.enable()
        evidence=registration.observe(p,d,a,raw,now_ns=now)
        profile.disable();elapsed=time.perf_counter()-began
        assert evidence['current_pose']==expected[frame]['registered_pose']
        if not routing:rows.append(dict(frame=frame,registration_ms=elapsed*1000))
        elif frame%4==0:
            snapshot=mapper.update(p,d,evidence,auxiliary_depth=a,measured_ns=now)
            goal=(np.array(snapshot.map_from_initial)@np.array([0.,2.6,0.]))[:2]
            began=time.perf_counter();profile.enable()
            result=propose(snapshot.floor,snapshot.occupied,snapshot.position_map[:2],goal)
            profile.disable()
            rows.append(dict(frame=frame,routing_ms=(time.perf_counter()-began)*1000,
                floor=sorted(snapshot.floor),occupied=sorted(snapshot.occupied),
                position=snapshot.position_map[:2],goal=goal.tolist(),result=result))
        if frame%40==0:print('REGISTRATION_PROFILE_FRAME',frame,flush=True)
    stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(30)
    with (root/(prefix+'.txt')).open('x') as f:f.write(stream.getvalue())
    report=dict(frames=len(rows),exact_recorded_raw_and_registered_poses=True,native_state_read=False,
        mean_ms=float(np.mean([r['routing_ms' if routing else 'registration_ms'] for r in rows])),rows=rows)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(stream.getvalue(),flush=True)
    print('PROFILE_COMPLETE',report['mean_ms'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--registration-process',action='store_true')
    parser.add_argument('--routing',action='store_true')
    args=parser.parse_args()
    if args.registration_process:
        from lewm.process_registered_round_trip_development import initialize_registration,registration_ready
        with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=initialize_registration) as executor:
            assert executor.submit(registration_ready).result()
            main(executor,routing=args.routing)
    else:main(routing=args.routing)
