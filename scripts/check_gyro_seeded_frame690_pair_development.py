"""Compare original and gyro-seeded matching on the diagnosed failed pair."""
import json
from pathlib import Path

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.direct_corner_flow_association_development import tracked_points as original
from lewm.gyro_seeded_corner_flow_development import tracked_points as seeded
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.auxiliary_reference_pose_adapter_development import gyro_in_reference, body_from_reference
from lewm.joint_rgbd_rigid_pose_development import register
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
    G = np.asarray(attitude['rotation_initial_body_from_current_body']); results = []
    for camera in ('primary', 'auxiliary'):
        R = G; t = np.zeros(3)
        if camera == 'auxiliary':
            A, offset = body_from_reference()
            R = gyro_in_reference(G); t = A.T@(G@offset-offset)
        for mode in ('original', 'gyro_seeded'):
            left, right = images[687][camera], images[690][camera]
            values, association = (original(left, right) if mode == 'original' else
                seeded(left, right, rotation=R, translation=t))
            row = dict(camera=camera, mode=mode, association=association)
            try:
                _, _, _, row['registration'] = register(*values,
                    gyro_rotation=R, mode='joint', frame=345)
                row['accepted'] = True
            except SensorContractError as error:
                row.update(accepted=False, failure=str(error))
            results.append(row)
    report = dict(source_frames=[687, 690], processed_fit_frame=345, results=results,
        source_sha256=source.source.digest(Path(__file__)),
        association_source_sha256=source.source.digest(Path('lewm/gyro_seeded_corner_flow_development.py')),
        original_fit_limits_unchanged=True, gyro_initialization_selected_after_failure=True,
        camera_translation_hypothesis='zero_body_translation_with_fixed_camera_lever_arm',
        raw_native_pose_loaded=False, tracker_pose_admitted=False, new_navigation=False)
    output = Path('docs/go2_gyro_seeded_frame690_pair_2026-09-13.json')
    with output.open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
