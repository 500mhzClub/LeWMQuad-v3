"""Descriptive classification probe at the cited public contact frames only."""
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.auxiliary_downward45_depth_observation_development import body_points
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.current_primary_floor_plane_development import primary_floor_plane, confirm_auxiliary_floor
from lewm.joint_visual_surface_memory_development import VOXEL_M
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.view_reentry_auxiliary_floor_diagnosis_development import diagnose


def probe(directory):
    prior = diagnose(directory); frames = {f['frame']: f for f in prior['cited_frames']}
    reader = IntentReturnRGBDReplay(directory); acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    reports = []
    for row in read_rows(directory):
        frame = row['tick']
        if frame not in frames: continue
        decision = row['decision']; memory = decision['memory_receipt']; pose = decision['evidence']['current_pose']
        B = np.asarray(memory['map_from_initial']); R = np.asarray(pose['rotation_initial_body_from_current_body'])
        p = np.asarray(pose['position_initial_body_m']); floor = memory['floor_height_map_m']
        policy, depth, _, now = reader.packet(frame)
        plane = primary_floor_plane(depth['depth_m'], depth['valid'], B@R, B@p, floor)
        auxiliary = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
        Q, q = reference_pose(B@R, B@p); cloud = body_points(auxiliary, policy, now_ns=now, stride=4)
        mask, receipt = confirm_auxiliary_floor(auxiliary['depth_m'], auxiliary['valid'], Q, q,
            floor, cloud['rows'], cloud['columns'], plane)
        assert receipt['original_floor_count'] == memory['auxiliary_receipt']['current_floor_returns']
        points = cloud['points_body_m'][cloud['valid']]@R.T+p; keys = np.floor(points/VOXEL_M).astype(int)
        pixels = np.stack(np.meshgrid(cloud['rows'], cloud['columns'], indexing='ij'), axis=-1)[cloud['valid']]
        revised = mask[cloud['valid']]; checks = []
        for voxel in frames[frame]['voxels']:
            selected = np.all(keys == voxel['cell'], axis=1)
            for sample in voxel['cited_latest_frame_other_samples']:
                match = selected & np.all(pixels == sample['pixel_row_column'], axis=1)
                if int(match.sum()) != 1: raise ValueError('unique cited public sample required')
                index = int(np.flatnonzero(match)[0])
                np.testing.assert_array_equal(points[index], sample['point_initial_body_m'])
                checks.append(dict(cell=voxel['cell'], pixel_row_column=sample['pixel_row_column'],
                    original_classification='other_or_unknown', confirmed_by_current_primary_plane=bool(revised[index])))
        reports.append(dict(frame=frame, primary_plane=plane, auxiliary_classification_receipt=receipt,
            cited_sample_checks=checks))
    assert len(reports) == len(frames)
    return dict(frames=reports, original_classes_or_commands_changed=False,
        all_historical_voxel_samples_reclassified=False, native_pose_used=False,
        original_terminal_constraint_resolution_proven=False, prospective_command_replay_required=True,
        alternative_execution_inferred=False, navigation_verified=False)
