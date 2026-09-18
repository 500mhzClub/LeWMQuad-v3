"""Reconstruct the selected primary pair at the persisted floor rejection.

This does not replay the full observer or admit native completion. It tests
whether the unmodified image registration reproduces the saved selected pose,
and measures that local transform's disagreement with both measured planes.
"""
from datetime import datetime, timezone
import json
import cv2
import numpy as np

from scripts import diagnose_go2_extended_budget_floor_boundary_v1 as diagnosis
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.joint_rgbd_rigid_pose_development import register, angle, RIGID_RULES

SOURCE = 'scripts/probe_go2_extended_budget_terminal_visual_pair_v1.py'
OUTPUT = ROOT/'docs/go2_extended_budget_terminal_visual_pair_provisional_2026-09-11.json'
DIAGNOSIS_SHA = 'cd40fa9b1b726708c32db1d1bfcdaf6b31d524c0da14238b2b9402f4479fc61a'


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive selected-pair probe required')
    if digest(diagnosis.OUTPUT) != DIAGNOSIS_SHA:
        raise ValueError('exact original floor reconstruction required')
    prior = json.loads(diagnosis.OUTPUT.read_text())
    sources = discover_sources((SOURCE,), prior['source_sha256'])
    verify(sources)
    verify_artifacts(diagnosis.NATIVE, prior['artifact_sha256'])
    cv2.setNumThreads(1)
    if cv2.ocl.useOpenCL():
        raise ValueError('CPU-only original OpenCV path required')
    rows = diagnosis.select_rows(diagnosis.NATIVE/diagnosis.CASE)
    reader = diagnosis.ExtendedBudgetRGBDReplay(diagnosis.NATIVE/diagnosis.CASE)
    features, poses = {}, {}
    for frame in (3836, 3837):
        raw = rows[frame]['decision']['original_visual_evidence']
        poses[frame] = raw['current_pose']
        if (raw['camera_selection']['selected_camera'] != 'primary'
                or raw['camera_selection']['auxiliary_attempted'] is not False
                or poses[frame]['promoted_keyframe'] is not True):
            raise ValueError('original selected primary measurements and promotions required')
        policy, depth, _, _ = reader.packet(frame)
        features[frame] = CornerSupportFeatureFrame(policy['image']['rgb'], depth)
        if features[frame].witness() != raw['last_accepted_feature_witness']:
            raise ValueError('exact original primary feature populations required')
    ref, cur = poses[3836], poses[3837]
    if cur['reference_frame'] != 3836:
        raise ValueError('actual terminal selected reference required')
    Ra = np.asarray(ref['rotation_initial_body_from_current_body'])
    pa = np.asarray(ref['position_initial_body_m'])
    G = (np.asarray(ref['gyro_rotation_initial_body_from_current_body']).T
        @ np.asarray(cur['gyro_rotation_initial_body_from_current_body']))
    points = matched_points(features[3836], features[3837])
    R, t, mask, reg = register(*points, gyro_rotation=G, mode='joint', frame=3837)
    global_R, global_p = Ra@R, pa+Ra@t
    if (not np.array_equal(global_R, cur['rotation_initial_body_from_current_body'])
            or not np.array_equal(global_p, cur['position_initial_body_m'])):
        raise ValueError('original selected raw global pose must reconstruct exactly')
    if (np.linalg.norm(global_p-pa) > RIGID_RULES['maximum_increment_translation_m']
            or angle(Ra.T@global_R) > RIGID_RULES['maximum_increment_rotation_rad']):
        raise ValueError('original consecutive pose envelope required')
    witness = rows[3837]['decision']['original_visual_evidence']['continuity_evidence']['selected_anchor_rotation_witness']
    for name in ('inliers', 'inlier_fraction', 'reference_grid_cells', 'current_grid_cells',
            'residual_rms_m', 'gyro_disagreement_rad'):
        if reg[name] != witness[name]:
            raise ValueError('selected original registration metric differs: '+name)
    planes = {row['frame']: row['joint_plane'] for row in prior['raw_depth_floor_reconstructions']}
    na, nb = np.asarray(planes[3836]['normal_body']), np.asarray(planes[3837]['normal_body'])
    da, db = planes[3836]['offset_body_m'], planes[3837]['offset_body_m']
    transformed_normal = R@nb
    local_height = db-da-float(na@t)
    local_tilt = float(np.arctan2(np.linalg.norm(np.cross(transformed_normal, na)), transformed_normal@na))
    fits = {row['frame']: row for row in prior['raw_depth_floor_reconstructions']}
    total_height_change = (fits[3837]['normal_translation_correction_m']
        - fits[3836]['normal_translation_correction_m'])
    verify(sources)
    verify_artifacts(diagnosis.NATIVE, prior['artifact_sha256'])
    if digest(diagnosis.OUTPUT) != DIAGNOSIS_SHA:
        raise ValueError('original floor diagnosis changed')
    report = dict(status='PROVISIONAL_ORIGINAL_TERMINAL_PRIMARY_PAIR_RECONSTRUCTED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        floor_diagnosis_sha256=DIAGNOSIS_SHA, artifact_sha256=prior['artifact_sha256'],
        reference_frame=3836, current_frame=3837, camera='primary', registration=reg,
        relative_rotation=R.tolist(), translation_reference_body_m=t.tolist(),
        feature_witnesses={str(f): v.witness() for f, v in features.items()},
        original_selected_global_position_and_rotation_byte_exact=True,
        original_selected_registration_metrics_exact=True, original_increment_envelopes_passed=True,
        local_plane_normal_translation_disagreement_m=local_height,
        local_plane_normal_rotation_disagreement_rad=local_tilt,
        accumulated_normal_translation_correction_change_m=total_height_change,
        previous_accumulated_correction_m=fits[3836]['normal_translation_correction_m'],
        current_accumulated_correction_m=fits[3837]['normal_translation_correction_m'],
        repeated_pair_registration=False, gyro_values_from_bound_original_visual_witnesses=True,
        gyro_history_reintegrated=False, full_observer_replayed=False, raw_pixels_used=True,
        native_pose_used=False, alternative_estimator_evaluated=False,
        native_completion_admitted=False, controller_changed=False, native_execution=False,
        navigation_recovered=False, navigation_qualified=False, goal_achieved=False)
    write_json(OUTPUT, report)
    print('TERMINAL_VISUAL_PAIR_RECONSTRUCTED', digest(OUTPUT), len(sources), flush=True)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256', 'artifact_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
