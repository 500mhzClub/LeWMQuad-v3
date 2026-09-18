"""Trace final auxiliary contact witnesses to their cited public depth frames.

This inspects the latest contributing frame for each first intersecting voxel,
not every historical sample or every intersecting voxel. No class is changed.
"""
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.auxiliary_downward45_depth_observation_development import body_points
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.observed_geometry_refinement_development import sampled_floor_patch
from lewm.floor_footprint_bounds_development import observed_floor_cell_index
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.joint_visual_surface_memory_development import VOXEL_M
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json


def diagnose(directory):
    final = None
    for row in read_rows(directory):
        selection = row['decision'].get('new_selection') or {}
        if selection.get('prediction') and selection['action'] is None:
            final = row
    if final is None:
        raise ValueError('actual final infeasible learned selection required')
    selection = final['decision']['new_selection']; witnesses = {}; candidates = []
    for candidate, surface, path in zip(selection['candidates'], selection['surface_checks'],
            selection['nominal_path_checks'], strict=True):
        hits = []
        for shape in surface['auxiliary_shapes']:
            if not shape['intersecting_voxels']: continue
            frame = shape['first_bounds_latest_frame']; cell = tuple(shape['first_cell'])
            if type(frame) is not int or not 0 <= frame <= final['tick']:
                raise ValueError('causal cited auxiliary witness required')
            witnesses.setdefault(frame, set()).add(cell)
            hits.append(dict(shape_id=shape['shape_id'], first_cell=list(cell), latest_frame=frame,
                first_bounds_m=shape['first_bounds_m'], count=shape['intersecting_voxels']))
        candidates.append(dict(action=candidate['action'], primary_conflict=surface['primary_possible_intersection'],
            auxiliary_conflict=surface['auxiliary_possible_intersection'], auxiliary_first_witnesses=hits,
            all_nominal_segments_clear=path['all_predicted_segments_nominally_clear']))
    if not witnesses: raise ValueError('auxiliary conflict witnesses required')
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); frames = []
    for row in read_rows(directory):
        frame = row['tick']
        if frame not in witnesses: continue
        decision = row['decision']; pose = decision['evidence']['current_pose']; memory = decision['memory_receipt']
        B = np.asarray(memory['map_from_initial']); R = np.asarray(pose['rotation_initial_body_from_current_body'])
        p = np.asarray(pose['position_initial_body_m']); floor = memory['floor_height_map_m']
        policy, _, _, now = reader.packet(frame)
        depth = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
        cloud = body_points(depth, policy, now_ns=now, stride=4)
        points = cloud['points_body_m'][cloud['valid']]@R.T+p
        Q, q = reference_pose(B@R, B@p)
        patch = sampled_floor_patch(depth['depth_m'], depth['valid'], Q, q, floor, cloud['rows'], cloud['columns'])
        mask = patch['measured_floor_patch'][cloud['valid']]
        receipt = memory['auxiliary_receipt']
        assert len(points) == receipt['current_returns'] and int(mask.sum()) == receipt['current_floor_returns']
        assert int((~mask).sum()) == receipt['current_other_returns']
        index = observed_floor_cell_index(depth['depth_m'], depth['valid'], Q[2])
        rr, cc = np.meshgrid(cloud['rows'], cloud['columns'], indexing='ij')
        pixels = np.column_stack((rr[cloud['valid']], cc[cloud['valid']]))
        yy, xx = np.indices((480, 640)); z = depth['depth_m']; T = np.asarray(BODY_FROM_OPTICAL)
        optical = np.stack((z*(xx+.5-320)/FOCAL, z*(yy+.5-240)/FOCAL, z), axis=-1)
        heights = (optical@T[:3,:3].T+T[:3,3])@Q[2]+q[2]
        keys = np.floor(points/VOXEL_M).astype(int); records = []
        for cell in sorted(witnesses[frame]):
            selected = np.flatnonzero(np.all(keys == cell, axis=1) & ~mask)
            if not len(selected): raise ValueError('cited latest nonfloor voxel has no reconstructed nonfloor return')
            samples = []
            for i in selected:
                r, c = pixels[i]; local = heights[r-1:r+2, c-1:c+2]-floor
                normals = bool(index['ground_cells'][r-1:r+1, c-1:c+1].all())
                nine_valid = bool(depth['valid'][r-1:r+2, c-1:c+2].all())
                band = bool(nine_valid and (np.abs(local) <= .01).all())
                assert bool(mask[i]) == bool(normals and band)
                samples.append(dict(pixel_row_column=[int(r), int(c)], point_initial_body_m=points[i].tolist(),
                    all_four_quads_pass_ground_geometry=normals, all_nine_pixels_valid=nine_valid,
                    all_nine_heights_within_original_band=band,
                    nine_height_residual_range_m=[float(local.min()), float(local.max())],
                    original_classification='other_or_unknown'))
            records.append(dict(cell=list(cell), cited_latest_frame_other_samples=samples))
        frames.append(dict(frame=frame, depth_sha256=receipt['depth_sha256'], floor_height_map_m=floor,
            complete_frame_classification_counts_exact=True, voxels=records))
    assert len(frames) == len(witnesses)
    return dict(final_infeasible_tick=final['tick'], candidates=candidates, cited_frames=frames,
        scope='latest contributing public frame of each first auxiliary conflict voxel only',
        all_historical_samples_reconstructed=False, native_pose_or_segmentation_used=False,
        classes_changed=False, alternative_execution_inferred=False, navigation_verified=False)
