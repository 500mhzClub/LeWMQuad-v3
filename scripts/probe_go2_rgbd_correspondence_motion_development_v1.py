"""Frozen A/B diagnostic; causal sensor estimates precede native-only scoring."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_correspondence_motion_development import RGBDCorrespondenceMotion,RULES
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_action_motion_validation_failure_development_v1 import IDENTITIES as B_IDENTITIES
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.fit_go2_action_response_development_v1 import DATA as A,IDENTITIES as A_IDENTITIES
from scripts.run_go2_action_motion_validation_development_v1 import OUTPUT as B
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_startup_observation_turn_development_v1 import verify_extensions
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_rgbd_correspondence_motion_diagnostic_v1_attempt_001'
PROTOCOL='docs/go2_rgbd_correspondence_motion_diagnostic_v1_2026-09-06.md'
SEEDS=('scripts/probe_go2_rgbd_correspondence_motion_development_v1.py',
       'lewm/tests/test_rgbd_correspondence_motion_development.py',PROTOCOL)
CV_BINARY=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/lib/python3.12/site-packages/cv2/cv2.abi3.so')
CV_SHA256='f7746177b49e0368c73142ab92d3d5a4a9ce989aecce7178d7e05c58679349c9'


def verify_opencv(binding):
    # Exact installed dependency only; ordinary source guards remain unchanged.
    if (binding!={str(CV_BINARY):CV_SHA256} or CV_BINARY.resolve()!=CV_BINARY
            or Path(cv2.__file__).resolve().parent!=CV_BINARY.parent or digest(CV_BINARY)!=CV_SHA256):
        raise ValueError('reviewed installed OpenCV binary changed')


def preflight():
    identities={str((path/n).relative_to(ROOT)):h for path,ids in ((A,A_IDENTITIES),(B,B_IDENTITIES)) for n,h in ids.items()}
    verify_bindings(identities)
    old=read_json(B,'launch.json');inputs=old['input_sha256']|identities
    result=read_json(B,'result.json')
    inputs|={str((B/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    sources=discover_sources(SEEDS,old['source_sha256'])
    definition=dict(source_sha256=sources,input_sha256=inputs,native_sha256=old['native_sha256'],
        native_geometry_sha256=old['native_geometry_sha256'],opencv_version=cv2.__version__,
        opencv_binary_sha256={str(CV_BINARY):CV_SHA256},rules=RULES,
        scope='one recorded development diagnostic; no physics, controller resume, refit or navigation qualification')
    verify(definition);return definition


def verify(launch):
    verify_bindings(launch['source_sha256']|launch['input_sha256'])
    verify_native_bindings(launch['native_sha256']);verify_extensions(launch['native_geometry_sha256'])
    verify_opencv(launch['opencv_binary_sha256'])
    if cv2.__version__!=launch['opencv_version'] or RULES!=launch['rules']:raise ValueError('frozen implementation/rules changed')


def replay(path):
    decisions=read_json(path,'motion_decisions.json');model=RGBDCorrespondenceMotion();rows=[]
    # All estimates use only current/past deployment-valid sensor packets.
    for frame,item in enumerate(decisions):
        assert item['observation_index']==frame
        policy,depth=load_rgbd_observation(path,frame);fast=load_fast_packet(path,frame);now=policy['sensor_state']['decision_ns']
        start=time.perf_counter();row=model.observe(policy,depth,fast,now_ns=now)
        row['observer_wall_ms']=1000*(time.perf_counter()-start);row['observation_index']=frame;rows.append(row)
    # Persist predictions BEFORE loading evaluator pose or original depth result.
    label='A' if path==A else 'B';write_json(OUTPUT/(label+'_sensor_predictions.json'),rows)
    cameras=read_json(path,'camera_audit.json');original=read_json(path,'relative_state_observations.json')
    with np.load(path/'physics_trace.npz',allow_pickle=False) as raw:
        for frame,row in enumerate(rows):
            if not frame:continue
            i,j=[cameras[f]['physical_sample_index'] for f in (frame-1,frame)]
            before=raw['base_pose_world'][i];after=raw['base_pose_world'][j]
            true=rotation_xyzw(before[3:]).T@(after[:3]-before[:3]);motion=original[frame]['observer']['motion']
            predicted=row['motion']['translation_previous_body_m']
            weak=np.asarray(motion['weak_directions_previous_body']).reshape(-1,3)
            projection=np.asarray(motion['observable_projection_previous_body_m'])
            row['evaluation']=dict(original_plane_depth_rank=motion['rank'],
                actual_translation_previous_body_m=true.tolist(),
                original_observable_projection_error_m=float(np.linalg.norm((np.eye(3)-weak.T@weak)@true-projection)),
                rgbd_translation_error_m=float(np.linalg.norm(np.asarray(predicted)-true)) if predicted is not None else None,
                rgbd_weak_component_error_m=float(np.linalg.norm(weak@(np.asarray(predicted)-true))) if predicted is not None and len(weak) else None)
    return rows


def summary(rows):
    transitions=rows[1:];accepted=[r for r in transitions if r['motion']['translation_previous_body_m'] is not None]
    return dict(transitions=len(transitions),accepted=len(accepted),rejected=len(transitions)-len(accepted),
        mean_error_m=float(np.mean([r['evaluation']['rgbd_translation_error_m'] for r in accepted])) if accepted else None,
        maximum_error_m=max((r['evaluation']['rgbd_translation_error_m'] for r in accepted),default=None),
        mean_observer_wall_ms=float(np.mean([r['observer_wall_ms'] for r in transitions])) if transitions else None)


def main():
    if OUTPUT.exists():raise ValueError('fresh fixed diagnostic output required; no retry/overwrite')
    cv2.setNumThreads(1)
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    try:
        rows={label:replay(path) for label,path in (('A',A),('B',B))}
        weak=[r for r in rows['B'] if 4_100_000_000<=r['measured_ns']<=5_300_000_000]
        # summary expects an anchor to exclude; preserve all 13 weak transitions.
        summaries={label:summary(values) for label,values in rows.items()}
        summaries['B_weak_interval']=summary([rows['B'][0],*weak])
        verify(launch)
        write_json(OUTPUT/'scored_rows.json',rows)
        report=dict(status='RECORDED_RGBD_CORRESPONDENCE_DIAGNOSTIC_COMPLETE',summaries=summaries,
            artifact_sha256={n:digest(OUTPUT/n) for n in ('A_sensor_predictions.json','B_sensor_predictions.json','scored_rows.json')},
            original_controller_results_unchanged=True,independent_validation=False,physics_executed=False,
            failed_controller_resumed=False,navigation_qualified=False)
        write_json(OUTPUT/'result.json',report);print(json.dumps(report,allow_nan=False),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='DIAGNOSTIC_FAILED',error=repr(error)));raise


if __name__=='__main__':main()
