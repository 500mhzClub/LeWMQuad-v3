"""Profile the measured tracker on recorded cameras without running navigation."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import time
import cv2
import numpy as np
from scripts.in_memory_public_replay_development import PublicReplay
from lewm.gyro_consensus_visual_motion_development import GyroConsensusVisualMotion
from lewm.two_cm_floor_extent_development import configure

BASE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root-name',required=True)
    parser.add_argument('--output-name',default='gyro_tracking_profile_v1')
    parser.add_argument('--profile-start',type=int,default=3260)
    parser.add_argument('--max-frames',type=int,default=3280)
    parser.add_argument('--reuse-chained-flow',action='store_true')
    parser.add_argument('--orthonormal-gyro',action='store_true')
    parser.add_argument('--jit-floor',action='store_true')
    args=parser.parse_args()
    for value in (args.root_name,args.output_name):
        if Path(value).name!=value or value.startswith('sealed'):raise ValueError('ordinary development basename required')
    root=BASE/args.root_name;output=root/args.output_name;output.mkdir()
    configure();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    reader=PublicReplay(root/'native');model=GyroConsensusVisualMotion()
    if args.reuse_chained_flow:
        from lewm.reused_flow_gyro_visual_motion_development import ReusedFlowGyroVisualMotion
        model=ReusedFlowGyroVisualMotion()
    if args.orthonormal_gyro:
        from lewm.orthonormal_gyro_visual_motion_development import OrthonormalGyroVisualMotion
        model=OrthonormalGyroVisualMotion()
    if args.jit_floor:
        from lewm.jit_floor_gyro_visual_motion_development import JitFloorGyroVisualMotion
        model=JitFloorGyroVisualMotion()
    expected=json.loads((root/'poses.json').read_text())
    profile=cProfile.Profile();rows=[];failure=None;first_difference=None
    max_position_difference=0.;max_rotation_difference=0.;reference_differences=[]
    with (output/'frames.jsonl').open('x') as sink:
        for frame in range(args.max_frames):
            p,d,fast,rgb,auxiliary,now=reader.packet(frame)
            profiled=frame>=args.profile_start
            if profiled:profile.enable()
            start=time.perf_counter()
            try:
                raw=model.observe(p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=auxiliary,now_ns=now)
            finally:
                elapsed=time.perf_counter()-start
                if profiled:profile.disable()
            if raw['current_pose'] is None:
                failure=dict(frame=frame,reason=raw['terminal_failure'])
                with (output/'failure_snapshot.json').open('x') as f:json.dump(raw,f,indent=2)
                break
            if frame<len(expected) and raw['current_pose']!=expected[frame]['raw_pose'] and first_difference is None:
                first_difference=frame
                print('FIRST_RAW_POSE_DIFFERENCE',frame,flush=True)
            if frame<len(expected):
                actual=raw['current_pose'];old=expected[frame]['raw_pose']
                max_position_difference=max(max_position_difference,float(np.linalg.norm(
                    np.asarray(actual['position_initial_body_m'])-old['position_initial_body_m'])))
                max_rotation_difference=max(max_rotation_difference,float(np.max(np.abs(
                    np.asarray(actual['rotation_initial_body_from_current_body'])-old['rotation_initial_body_from_current_body']))))
                if actual['reference_frame']!=old['reference_frame']:reference_differences.append(frame)
            row=dict(frame=frame,tracking_s=elapsed,profiled=profiled,
                reference_frame=raw['current_pose']['reference_frame'],
                selected_camera=raw['camera_selection'].get('selected_camera'),
                direct_flow=bool(model.model.last_direct_flow_fallback),
                chained_flow=bool(model.model.last_chained_anchor_fallback))
            rows.append(row);sink.write(json.dumps(row)+'\n');sink.flush()
            if frame%100==0:print('PROFILE_FRAME',frame,'tracking_s',round(elapsed,4),flush=True)
    profile.dump_stats(output/'tracking.prof')
    stream=io.StringIO();pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(40)
    (output/'profile.txt').write_text(stream.getvalue())
    report=dict(frames=len(rows),failure=failure,first_raw_pose_difference=first_difference,
        native_state_used=False,profiled_from_frame=args.profile_start,rows=rows,
        reuse_chained_flow=args.reuse_chained_flow or args.orthonormal_gyro or args.jit_floor,
        orthonormal_gyro=args.orthonormal_gyro or args.jit_floor,
        compiled_floor_candidate_predicates=args.jit_floor,
        maximum_position_difference_m=max_position_difference,
        maximum_rotation_element_difference=max_rotation_difference,
        reference_choice_difference_frames=reference_differences)
    if args.reuse_chained_flow or args.orthonormal_gyro or args.jit_floor:
        memo=model.model.flow_memo
        report['flow_memo']=dict(hits=memo.hits,misses=memo.misses,
            retained_images=len(memo.images),retained_links=len(memo.links))
    with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print('PROFILE_RESULT',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    print(stream.getvalue(),flush=True)


if __name__=='__main__':main()
