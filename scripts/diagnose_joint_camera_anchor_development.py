"""Probe retained-anchor fits at the initial-survey tracking failure using public sensors."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from lewm.conditioned_support_150_tracker_development import ConditionedSupport150VisualMotion, FeatureFrame150
from lewm.batched_patch_tracker_development import chained_points
from lewm.joint_camera_registration_development import register_views
from lewm.joint_rgbd_rigid_pose_development import angle
from scripts.in_memory_public_replay_development import PublicReplay


def main():
    root = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_initial_panorama_motion_residual_native_layout00_v1_attempt_001')
    output = root/'joint_camera_anchor_diagnostic.json'
    if output.exists():
        raise ValueError('preserve existing diagnostic')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    reader = PublicReplay(root/'native'); tracker = ConditionedSupport150VisualMotion()
    for frame in range(71):
        p, d, f, rgb, aux, now = reader.packet(frame)
        raw = tracker.observe(p, d, f, now_ns=now, auxiliary_rgb=rgb, auxiliary_depth=aux)
        if raw['current_pose'] is None:
            raise ValueError(f'unexpected earlier failure {frame}')
        if frame % 10 == 0:
            print('REPLAY_FRAME', frame, flush=True)
    model = tracker.model
    p, d, f, rgb, aux, now = reader.packet(71)
    G = np.asarray(model.gyro.step(p, f, now_ns=now)['rotation_initial_body_from_current_body'])
    current = dict(primary=FeatureFrame150(p['image']['rgb'], d), auxiliary=FeatureFrame150(rgb['rgb'], aux))
    model.frame = 71; model._cache_images(current, now)
    rows = []
    for ref in model.references:
        row = dict(reference_frame=ref.frame, current_frame=71)
        try:
            views = []
            for camera in ('primary', 'auxiliary'):
                sequence = [(i, model._image_history[i][0], model._image_history[i][1][camera])
                    for i in range(ref.frame, 72)]
                sequence[0] = (ref.frame, ref.measured_ns, ref.features[camera])
                values, _ = chained_points(sequence)
                views.append(values)
            row['camera_matches'] = [len(v[0]) for v in views]
            R, t, mask, receipt, _ = register_views(*views, gyro_rotation=ref.gyro.T@G, frame=71)
            position = ref.position+ref.rotation@t
            rotation = ref.rotation@R
            row.update(qualified=True, registration=receipt, position=position.tolist(),
                rotation=rotation.tolist(), translation_since_previous_m=float(np.linalg.norm(position-model.last_p)),
                rotation_since_previous_rad=angle(model.last_R.T@rotation))
        except (ValueError, RuntimeError) as error:
            row.update(qualified=False, failure=str(error))
        rows.append(row); print('JOINT_ANCHOR', json.dumps(row), flush=True)
    with output.open('x') as f:
        json.dump(dict(native_state_used=False, pose_admitted=False, rows=rows), f, indent=2)


if __name__ == '__main__':
    main()
