"""Read-only endpoint fit diagnosis; no tracker thresholds or poses changed."""
import json
from pathlib import Path

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.direct_corner_flow_association_development import tracked_points
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.auxiliary_reference_pose_adapter_development import gyro_in_reference
from lewm.joint_rgbd_rigid_pose_development import register, angle, cells, RULES
from scripts import check_gapped_camera_recorded_prefix_development as source


def main():
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    reader = source.source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
    auxiliary = json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
    gyro = FastRelativeOrientation(); images = {}
    for frame in range(687, 691):
        p, d, f, now = reader.packet(frame)
        attitude = (gyro.begin if frame == 687 else gyro.step)(p, f, now_ns=now)
        if frame in (687, 690):
            rgb, depth = source.source.packets.rgb_packet(source.INPUT, frame, p,
                source.source.public_acquisition(auxiliary[frame]), now_ns=now)
            images[frame] = dict(primary=CornerSupportFeatureFrame(p['image']['rgb'], d),
                auxiliary=CornerSupportFeatureFrame(rgb['rgb'], depth))
    G = np.asarray(attitude['rotation_initial_body_from_current_body'])
    results = {}
    for camera in ('primary', 'auxiliary'):
        values, association = tracked_points(images[687][camera], images[690][camera])
        row = dict(association=association, reference_features=images[687][camera].witness(),
            current_features=images[690][camera].witness())
        try:
            _, _, _, row['registration'] = register(*values,
                gyro_rotation=G if camera == 'primary' else gyro_in_reference(G),
                mode='joint', frame=345)
            row['accepted'] = True
        except SensorContractError as error:
            row.update(accepted=False, failure=str(error))
            tb = error.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code is register.__code__:
                    state = tb.tb_frame.f_locals
                    if 'mask' in state:
                        mask = state['mask']; a, b, ua, ub = values
                        row['rejected_fit'] = dict(inliers=int(mask.sum()),
                            inlier_fraction=float(mask.mean()),
                            reference_grid_cells=cells(ua[mask]), current_grid_cells=cells(ub[mask]),
                            translation_norm_m=float(np.linalg.norm(state['t'])),
                            rotation_rad=angle(state['R']),
                            residual_rms_m=float(np.sqrt(np.mean(state['residual'][mask]**2))),
                            reference_inlier_pixels=ua[mask].tolist(), current_inlier_pixels=ub[mask].tolist())
                tb = tb.tb_next
        results[camera] = row
    report = dict(source_frames=[687, 690], processed_fit_frame=345,
        source_sha256=source.source.digest(Path(__file__)), rules=RULES, cameras=results,
        original_fit_unchanged=True, rejected_pose_admitted=False,
        raw_native_pose_loaded=False, new_navigation=False)
    output = Path('docs/go2_gapped_camera_frame690_fit_diagnosis_2026-09-13.json')
    with output.open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    for camera, row in results.items():
        print(camera, json.dumps({k:v for k,v in row.items() if k != 'rejected_fit'}))
        print('fit', json.dumps({k:v for k,v in row.get('rejected_fit', {}).items() if 'pixels' not in k}))


if __name__ == '__main__': main()
