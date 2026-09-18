"""Same recorded failure case, independently available short motion histories."""
import json
import time
import cv2
import torch
from lewm.causal_local_rgbd_motion_development import CausalLocalRGBDMotion
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_training_visual_motion_development import SOURCE,BASE

OUTPUT=BASE/'go2_local_training_motion_probe_v1_attempt_001'


def main():
    if OUTPUT.exists():raise ValueError('preserve local-history probe')
    branch=json.loads((SOURCE/'branch_specification.json').read_text())
    if branch['data_role']!='train':raise ValueError('training-only probe')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    count=len(json.loads((SOURCE/'policy_observations.json').read_text())['frames'])
    observer=CausalLocalRGBDMotion();rows=[];started=time.monotonic();OUTPUT.mkdir()
    try:
        for frame in range(count):
            policy,depth=load_rgbd_observation(SOURCE,frame)
            value=observer.observe(policy,depth,load_fast_packet(SOURCE,frame),
                                   now_ns=policy['sensor_state']['decision_ns'])
            rows.append(dict(frame=frame,**value))
        result=dict(status='COMPLETE',source=str(SOURCE),recorded_frames=count,
            available_histories=sum(r['history_available'] for r in rows),possible_histories=max(0,count-3),
            pair_failures=[dict(frame=r['frame'],reason=r['pair_failure']) for r in rows if r['pair_failure']],
            missing_history_frames=[r['frame'] for r in rows if r['frame']>=3 and not r['history_available']],
            at_previous_global_failure=rows[40],global_tracking_recovered=False,
            native_state_read=False,hardware_validated=False,wall_s=time.monotonic()-started)
        (OUTPUT/'features.json').write_text(json.dumps(rows,indent=2)+'\n')
        (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    except Exception as error:
        (OUTPUT/'failure.json').write_text(json.dumps(dict(reason=repr(error),attempted_frames=len(rows))))
        raise


if __name__=='__main__':main()
