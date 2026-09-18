"""Recover the exact tracking rejection from the retained cache-trial sensors."""
import json
import time
import numpy as np

from lewm import process_mapped_runtime_development as process
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.replay_go2_cadenced_maze00_failure_development import FailureViewProbePose
from scripts.run_go2_cached_fine_connectivity_development import BASE,ROOT,collection


def main():
    root=BASE/ROOT;output=root/'tracking_failure_replay_v1';output.mkdir(exist_ok=False)
    read=lambda name:json.loads((root/name).read_text())
    recorded={r['frame']:r['raw_pose'] for r in read('poses.json')}
    count=len(read('native/in_memory_camera_observations.json')['frames'])
    # Use the launcher's actual pose-worker initialization and its global setup.
    collection.previous.baseline.initialize_pose(str(output))
    motion=process._motion;motion.model=FailureViewProbePose();motion.model.probe_stored_fits=True
    reader=NoisyPublicReplay(root/'native');rows=[];matched=0;began=time.monotonic();failure=None
    for frame in range(count):
        policy,depth,fast,rgb,auxiliary,now=reader.packet(frame)
        raw=motion.observe(policy,depth,fast,auxiliary_rgb=rgb,auxiliary_depth=auxiliary,now_ns=now)
        pose=raw.get('current_pose')
        row=dict(frame=frame,reference_frames=[r.frame for r in motion.model.references],
            selected_reference=None if pose is None else pose['reference_frame'],
            reference_selection=motion.model.last_selection,
            revisit_attempt=motion.model.last_revisit_attempt,
            feature_support={camera:raw.get(key) for camera,key in
                (('primary','last_accepted_feature_witness'),('auxiliary','auxiliary_feature_witness'))})
        rows.append(row)
        if pose is None or raw.get('failure') is not None:
            failure=dict(frame=frame,failure=raw.get('failure'),terminal_failure=raw.get('terminal_failure'),
                view_probe=motion.model.failure_view_probe)
            (output/'terminal_raw_snapshot.json').write_text(json.dumps(raw,indent=2)+'\n')
            break
        if frame in recorded:
            for key in ('position_initial_body_m','rotation_initial_body_from_current_body'):
                np.testing.assert_array_equal(pose[key],recorded[frame][key])
            assert pose['mode']==recorded[frame]['mode'] and pose['reference_frame']==recorded[frame]['reference_frame']
            matched+=1
        if frame%200==0:print('TRACKING_REPLAY',frame,round(time.monotonic()-began,1),flush=True)
    result=dict(schema='cache_trial_tracking_failure_replay.v1',matched_recorded_raw_poses=matched,
        recorded_raw_poses=len(recorded),all_recorded_raw_poses_matched=matched==len(recorded),
        failure=failure,rows=rows,wall_seconds=time.monotonic()-began,
        delivered_noisy_depth_digests_verified=True,native_state_used=False,
        navigation_reexecuted=False,stored_view_probe_is_diagnostic_only=True,
        failed_tracker_not_restarted=True)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('failure','rows')},indent=2))
    print('FAILURE',json.dumps(failure),flush=True)


if __name__=='__main__':main()
