"""Compare joint and gyro-conditioned fits on the two failed adjacent pairs."""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
from scripts.replay_go2_depth_noise_tracking_development import BASE, PublicReplay, perturbed_packet
from scripts.diagnose_go2_depth_noise_failures_development import save
from lewm.feature_budget_150_tracker_development import FeatureFrame150
from lewm.batched_patch_agreement_development import tracked_points
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.conditioned_support_tracker_development import register
from lewm.orthonormal_gyro_visual_motion_development import OrthonormalFastOrientation
from lewm.auxiliary_reference_pose_adapter_development import gyro_in_reference


def main():
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    for layout, frame in ((2, 200), (3, 519)):
        root = BASE/f'go2_routing_memory_persistent_native_layout{layout:02d}_4800_v1_attempt_001'
        output = root/'depth_noise_2mm_failure_diagnostic_v1'
        reader = PublicReplay(root/'native')
        packets = [reader.packet(f) for f in (frame-1, frame)]
        gyro = OrthonormalFastOrientation()
        gyro.begin(packets[0][0], packets[0][2], now_ns=packets[0][-1])
        G = np.asarray(gyro.step(packets[1][0], packets[1][2],
            now_ns=packets[1][-1])['rotation_initial_body_from_current_body'])
        rows = []
        for sigma in (0, 2):
            pair = [perturbed_packet(p, layout=layout, frame=f, sigma_m=sigma/1000)
                for p, f in zip(packets, (frame-1, frame), strict=True)]
            for camera in ('primary', 'auxiliary'):
                features = [FeatureFrame150(p[0]['image']['rgb'], p[1]) if camera == 'primary'
                    else FeatureFrame150(p[3]['rgb'], p[4]) for p in pair]
                for association in ('descriptor', 'direct_flow'):
                    if association == 'descriptor':
                        values = matched_points(*features); receipt = None
                    else:
                        values, receipt = tracked_points(*features)
                    for mode in ('joint', 'gyro'):
                        row = dict(sigma_mm=sigma, camera=camera, association=association,
                            fitting_mode=mode, lifted_matches=len(values[0]),
                            feature_witnesses=[f.witness() for f in features], association_receipt=receipt)
                        try:
                            R, t, mask, evidence = register(*values,
                                gyro_rotation=G if camera == 'primary' else gyro_in_reference(G),
                                mode=mode, frame=frame)
                            row.update(accepted=True, evidence=evidence,
                                rotation=R.tolist(), translation=t.tolist())
                        except ValueError as error:
                            row.update(accepted=False, reason=str(error))
                        rows.append(row)
                        print(layout, sigma, camera, association, mode,
                            row['accepted'], row.get('reason', row.get('evidence', {}).get('inliers')), flush=True)
        save(output, 'adjacent_pair_fit_probe.json', dict(layout_index=layout, frame=frame,
            rows=rows, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            native_physics_used=False, full_tracker_executed=False,
            floor_registration_executed=False, pose_admission_claimed=False,
            gyro_integrated_from_current_public_interval=True,
            absolute_geometric_thresholds_unchanged=True))


if __name__ == '__main__':
    main()
