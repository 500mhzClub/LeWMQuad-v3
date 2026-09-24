"""Reproduce a retained noisy run's visual tracking failure from actual sensors."""
import argparse
import json
import time
import cv2
import torch
import numpy as np
from lewm.retained_depth_cache_tracking_development import RetainedDepthCacheMotion,RetainedDepthCachePose
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import path
from scripts.diagnose_go2_depth_noise_failures_development import save


def install_consensus_trace(model, rows):
    """Observe final refit inputs/rejections; execute the unchanged estimator."""
    from lewm import gyro_consensus_pair_pose_development as consensus
    from lewm.development_support_tracker_development import use
    original = consensus.consensus_refit

    def traced(candidate, gyro, **kwargs):
        if model.frame < model.trace_start_frame:
            return original(candidate, gyro, **kwargs)
        reg = candidate['registration']
        row = dict(frame=model.frame, camera=kwargs['camera'],
            reference_frame=candidate['reference'].frame, registration=reg,
            gyro_rotation=gyro, thresholds_changed=False)
        try:
            result = original(candidate, gyro, **kwargs)
        except Exception as error:
            row.update(accepted=False, error=repr(error))
            tb = error.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code is original.__code__:
                    state = tb.tb_frame.f_locals
                    row['refit_rounds'] = state.get('rounds')
                    row['terminal_mask'] = np.asarray(state['mask']).tolist()
                    row['terminal_inliers'] = int(np.sum(state['mask']))
                    row['terminal_translation_m'] = state.get('t')
                tb = tb.tb_next
            rows.append(row)
            raise
        row.update(accepted=True, final_registration=result['registration'])
        rows.append(row)
        return result

    consensus._candidate_with_consensus = use(consensus._candidate_with_consensus, refit=traced)


class TracedPose(RetainedDepthCachePose):
    def __init__(self,*args,trace_start_frame=1855,**kwargs):
        super().__init__(*args,**kwargs)
        self.trace_start_frame=trace_start_frame
        self.outer_candidate_trace=[]

    def _candidate(self,ref,current,G):
        if self.frame<self.trace_start_frame:return super()._candidate(ref,current,G)
        row=dict(frame=self.frame,camera=self.camera,reference_frame=ref.frame,
            direct_mode=getattr(self,'_direct_flow_mode',None),chain_mode=getattr(self,'_chain_mode',None))
        try:
            candidate=super()._candidate(ref,current,G)
        except Exception as error:
            row.update(accepted=False,error=repr(error))
            self.outer_candidate_trace.append(row)
            raise
        row.update(accepted=True,position=candidate['p'].tolist(),
            registration_mode=candidate['registration'].get('mode'))
        self.outer_candidate_trace.append(row)
        return candidate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trace-candidates',action='store_true')
    parser.add_argument('--trace-consensus',action='store_true')
    parser.add_argument('--root-name',default='go2_declared_floor_gap_reactive_noise_2mm_native_layout03_4800_v1_attempt_001')
    args=parser.parse_args();trace=args.trace_candidates or args.trace_consensus
    root=path(args.root_name)
    output=root/('terminal_consensus_trace_v1' if args.trace_consensus else
        'terminal_candidate_trace_v1' if trace else 'terminal_tracking_replay_v1');output.mkdir()
    recorded={r['frame']:r['raw_pose'] for r in json.loads((root/'poses.json').read_text())}
    count=len(json.loads((root/'native/in_memory_camera_observations.json').read_text())['frames'])
    configure();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=NoisyPublicReplay(root/'native');motion=RetainedDepthCacheMotion()
    if trace:motion.model=TracedPose(activation_frame=0,trace_start_frame=max(recorded,default=0))
    consensus_rows=[]
    if args.trace_consensus:install_consensus_trace(motion.model,consensus_rows)
    matched=0;started=time.monotonic()
    for frame in range(count):
        p,d,fast,rgb,a,now=reader.packet(frame)
        raw=motion.observe(p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=a,now_ns=now)
        if raw.get('current_pose') is None or raw.get('failure') is not None:
            save(output,'terminal_raw_snapshot.json',raw)
            save(output,'terminal_measured_plane.json',motion.model.last_measured_plane)
            result=dict(failure_frame=frame,recorded_frames=count,
                matched_published_raw_poses=matched,published_raw_poses=len(recorded),
                failure=raw.get('terminal_failure',raw.get('failure')),elapsed_s=time.monotonic()-started,
                raw_sensor_replay=True,native_pose_used=False,physics_or_controller_executed=False)
            if trace:save(output,'outer_candidate_trace.json',motion.model.outer_candidate_trace)
            if args.trace_consensus:save(output,'consensus_trace.json',consensus_rows)
            save(output,'result.json',result);print(json.dumps(result),flush=True);return
        if frame in recorded:
            for k in ('frame','measured_ns','reference_frame','position_initial_body_m',
                    'rotation_initial_body_from_current_body','rgb_sha256','depth_sha256'):
                if raw['current_pose'][k]!=recorded[frame][k]:
                    raise ValueError(f'raw pose replay mismatch at {frame}: {k}')
            matched+=1
        if frame%300==0:print('TRACKING_REPLAY_FRAME',frame,flush=True)
    raise ValueError('recorded tracking failure did not reproduce')


if __name__=='__main__':main()
