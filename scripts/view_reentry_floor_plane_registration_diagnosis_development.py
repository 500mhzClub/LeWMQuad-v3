"""Compare initial and contemporary measured planes without changing a policy.

Least-squares plane fits are descriptive evidence, not floor truth or a contact
exemption. Initial candidates retain the original height band; contemporary
candidates use only the original measured ground-mesh tests.
"""
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.floor_footprint_bounds_development import observed_floor_cell_index
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.view_reentry_auxiliary_floor_diagnosis_development import diagnose as contact_diagnosis


def fit_plane(points):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise ValueError('at least 100 finite measured plane candidates required')
    center = points.mean(0); delta = points-center
    values, vectors = np.linalg.eigh(delta.T@delta/len(points))
    if values[1] <= 1e-8:
        raise ValueError('two-dimensional measured span required')
    normal = vectors[:, 0]
    if normal[2] < 0: normal = -normal
    offset = -float(normal@center); errors = points@normal+offset
    return dict(points=len(points), normal_map=normal.tolist(), offset_m=offset,
        covariance_eigenvalues_m2=values.tolist(), max_absolute_residual_m=float(np.abs(errors).max()),
        rms_residual_m=float(np.sqrt(np.mean(errors**2))),
        tilt_from_retained_up_rad=float(np.arctan2(np.linalg.norm(normal[:2]), normal[2])),
        physical_floor_certified=False, policy_admission_performed=False)


def diagnose(directory):
    contacts = contact_diagnosis(directory)
    indices = {0, *(f['frame'] for f in contacts['cited_frames'])}
    rows = {r['tick']: r for r in read_rows(directory) if r['tick'] in indices}
    if set(rows) != indices: raise ValueError('complete initial and cited observations required')
    reader = IntentReturnRGBDReplay(directory); acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    fits = []; T = np.asarray(BODY_FROM_OPTICAL)
    for frame in sorted(indices):
        decision = rows[frame]['decision']; memory = decision['memory_receipt']; pose = decision['evidence']['current_pose']
        B = np.asarray(memory['map_from_initial']); R = np.asarray(pose['rotation_initial_body_from_current_body'])
        p = np.asarray(pose['position_initial_body_m']); floor = memory['floor_height_map_m']
        policy, primary, _, now = reader.packet(frame)
        auxiliary = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
        for name, depth, Q, q in (
                ('primary', primary, B@R, B@p),
                ('auxiliary', auxiliary, *reference_pose(B@R, B@p))):
            index = observed_floor_cell_index(depth['depth_m'], depth['valid'], Q[2])
            rr, cc = np.nonzero(index['ground_cells']); z = depth['depth_m'][rr, cc]
            optical = np.column_stack((z*(cc+.5-320)/FOCAL, z*(rr+.5-240)/FOCAL, z))
            points = (optical@T[:3,:3].T+T[:3,3])@Q.T+q
            if frame == 0: points = points[np.abs(points[:, 2]-floor) <= .01]
            fit = fit_plane(points); normal = np.asarray(fit['normal_map']); comparisons = []
            for witness in contacts['cited_frames']:
                if frame not in (0, witness['frame']): continue
                for voxel in witness['voxels']:
                    samples = voxel['cited_latest_frame_other_samples']
                    query = np.asarray([s['point_initial_body_m'] for s in samples])@B.T
                    residual = query@normal+fit['offset_m']
                    comparisons.append(dict(witness_frame=witness['frame'], cell=voxel['cell'], samples=len(samples),
                        signed_residual_to_fitted_plane_range_m=[float(residual.min()), float(residual.max())]))
            fits.append(fit | dict(frame=frame, sensor=name, initial_height_band_applied=frame == 0,
                witness_comparisons=comparisons))
    return dict(contact_diagnosis=contacts, plane_fits=fits,
        current_primary_plane_uses_no_auxiliary_fit_samples=True,
        native_pose_or_segmentation_used=False, classifications_or_commands_changed=False,
        hypothesis_only=True, navigation_verified=False)
