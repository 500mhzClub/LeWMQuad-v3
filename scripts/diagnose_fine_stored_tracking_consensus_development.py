"""Record the exact consensus rejection on the failed public-sensor frame."""
from pathlib import Path
import json
import sys
import argparse
import cv2
import numpy as np
import torch
from lewm.optional_plane_refinement_development import OptionalPlaneVisualMotion
from lewm.joint_rgbd_rigid_pose_development import cells,RULES,RIGID_RULES
from scripts.in_memory_public_replay_development import PublicReplay


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root-name',default='go2_fine_stored_obstacle_native_layout00_v1_attempt_001')
    args=parser.parse_args()
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):
        raise ValueError('ordinary artifact basename required')
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')/args.root_name
    output=root/'consensus_failure_diagnostic.json'
    if output.exists():raise ValueError('preserve diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=PublicReplay(root/'native');tracker=OptionalPlaneVisualMotion();failures=[]
    failed_frame=len(json.loads((root/'poses.json').read_text()))
    def trace(frame,event,arg):
        if (event=='exception' and frame.f_code.co_name=='register'
                and frame.f_code.co_filename.endswith('/full_consensus_early_exit_development.py')
                and str(arg[1])=='rigid consensus fraction, grid support or displacement rejected'):
            v=frame.f_locals;mask=v['mask']
            failures.append(dict(lifted_matches=len(v['a']),inliers=int(mask.sum()),
                inlier_fraction=float(mask.mean()),reference_grid_cells=cells(v['ua'][mask]),
                current_grid_cells=cells(v['ub'][mask]),translation_m=float(np.linalg.norm(v['t'])),
                accepted_reference_pixels=v['ua'][mask].tolist(),accepted_current_pixels=v['ub'][mask].tolist(),
                reference_points=v['a'].tolist(),current_points=v['b'].tolist(),
                reference_pixels=v['ua'].tolist(),current_pixels=v['ub'].tolist()))
        return trace
    for frame in range(failed_frame+1):
        p,d,f,rgb,a,now=reader.packet(frame)
        if frame==failed_frame:sys.settrace(trace)
        try:raw=tracker.observe(p,d,f,now_ns=now,auxiliary_rgb=rgb,auxiliary_depth=a)
        finally:sys.settrace(None)
        if frame%100==0:print('CONSENSUS_DIAGNOSTIC_FRAME',frame,flush=True)
        if raw['current_pose'] is None:break
    report=dict(frame=frame,status=raw['status'],failures=failures,rules=RULES,rigid_rules=RIGID_RULES,
        public_sensor_replay_only=True,native_state_read=False,estimator_decisions_unchanged=True)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('CONSENSUS_DIAGNOSTIC_COMPLETE',frame,[{k:v for k,v in r.items() if not isinstance(v,list)} for r in failures],flush=True)


if __name__=='__main__':main()
