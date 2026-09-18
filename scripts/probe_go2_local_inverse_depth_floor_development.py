"""Fixed seven-frame floor-only probe; no tracker or navigation execution."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.replay_go2_depth_noise_tracking_development import (
    BASE, PublicReplay, perturbed_packet, configure, SEED)
from scripts.diagnose_go2_depth_noise_failures_development import save
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.jit_floor_candidates_development import measured_candidates as original, warmup
from lewm.local_inverse_depth_floor_development import measured_candidates as local, WINDOW
from lewm.joint_measured_floor_plane_development import fit_joint_plane


def main():
    output = BASE/'go2_local_inverse_depth_floor_probe_v1_attempt_001'
    output.mkdir()
    roster = [(i, 0) for i in range(4)] + [(0, 236), (2, 200), (3, 519)]
    save(output, 'launch.json', dict(roster=roster, sigma_mm=[0,2], seed=SEED,
        window=WINDOW, floor_only=True, thresholds_changed=False,
        raw_depth_not_replaced=True, source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in map(Path, (__file__, 'lewm/local_inverse_depth_floor_development.py'))}))
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); warmup()
    rows=[]
    for layout, frame in roster:
        root=BASE/f'go2_routing_memory_persistent_native_layout{layout:02d}_4800_v1_attempt_001'
        reader=PublicReplay(root/'native'); initial=reader.packet(0)
        up=initial[0]['sensor_state']['sensed']['specific_force']['values'].mean(0)
        up=up/np.linalg.norm(up)
        # Hold orientation fixed across the two candidate algorithms and noise
        # levels. This is the recorded public-sensor visual estimate, not truth.
        pose=json.loads((root/'poses.json').read_text())[frame]['raw_pose']
        up=np.asarray(pose['rotation_initial_body_from_current_body']).T@up
        packet=initial if frame==0 else reader.packet(frame)
        for sigma in (0,2):
            p,d,g,rgb,a,now=perturbed_packet(packet,layout=layout,frame=frame,sigma_m=sigma/1000)
            for name, extractor in (('original',original),('local_inverse_depth',local)):
                clouds=[extractor(depth['depth_m'],depth['valid'],E,up)[0]
                    for depth,E in ((d,np.asarray(BODY_FROM_OPTICAL)),(a,body_from_optical()))]
                plane=fit_joint_plane(*clouds,up)
                row=dict(layout_index=layout,frame=frame,sigma_mm=sigma,algorithm=name,plane=plane)
                rows.append(row)
                print(layout,frame,sigma,name,plane['candidate_count'],plane['available'],plane['reason'],flush=True)
    save(output,'result.json',dict(rows=rows,native_physics_used=False,
        full_tracker_executed=False,closed_loop_navigation_test=False,
        orientation_from_original_recorded_public_visual_pose=True,
        estimated_local_depth_is_not_raw_pixel_depth=True))


if __name__=='__main__':
    main()
