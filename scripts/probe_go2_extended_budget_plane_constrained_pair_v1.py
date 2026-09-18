"""One predeclared pair: floor-constrained fit on original accepted inliers.

Retains every original point and reports the original geometric checks without
searching another pair or claiming observer recovery from this local result.
"""
from datetime import datetime, timezone
import json
import cv2
import numpy as np

from scripts import probe_go2_extended_budget_terminal_visual_pair_v1 as original
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.joint_rgbd_rigid_pose_development import register, inliers, angle, RIGID_RULES
from lewm.rgbd_correspondence_motion_development import RULES, cells
from lewm.measured_plane_rigid_fit_development import fit

SOURCE = 'scripts/probe_go2_extended_budget_plane_constrained_pair_v1.py'
TEST = 'lewm/tests/test_measured_plane_rigid_fit_development.py'
OUTPUT = ROOT/'docs/go2_extended_budget_plane_constrained_pair_provisional_2026-09-11.json'
ORIGINAL_SHA = '9796462c7206ebc8525b8e6b571097e12ee14da0c965eefb365c8aa045de73dd'


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive single-pair diagnostic required')
    if digest(original.OUTPUT) != ORIGINAL_SHA:
        raise ValueError('exact reconstructed original selected pair required')
    prior = json.loads(original.OUTPUT.read_text())
    diagnosis = original.diagnosis
    if digest(diagnosis.OUTPUT) != original.DIAGNOSIS_SHA:
        raise ValueError('exact raw floor reconstruction required')
    floor = json.loads(diagnosis.OUTPUT.read_text())
    sources = discover_sources((SOURCE, TEST), prior['source_sha256'])
    verify(sources)
    verify_artifacts(diagnosis.NATIVE, prior['artifact_sha256'])
    cv2.setNumThreads(1)
    if cv2.ocl.useOpenCL(): raise ValueError('CPU original OpenCV required')
    rows = diagnosis.select_rows(diagnosis.NATIVE/diagnosis.CASE)
    reader = diagnosis.ExtendedBudgetRGBDReplay(diagnosis.NATIVE/diagnosis.CASE)
    features = {}
    for frame in (3836, 3837):
        policy, depth, _, _ = reader.packet(frame)
        features[frame] = CornerSupportFeatureFrame(policy['image']['rgb'], depth)
        if features[frame].witness() != prior['feature_witnesses'][str(frame)]:
            raise ValueError('original exact feature populations required')
    ref, cur = [rows[f]['decision']['original_visual_evidence']['current_pose'] for f in (3836, 3837)]
    G = (np.asarray(ref['gyro_rotation_initial_body_from_current_body']).T
        @ np.asarray(cur['gyro_rotation_initial_body_from_current_body']))
    a, b, ua, ub = matched_points(features[3836], features[3837])
    R0, t0, mask, reg = register(a, b, ua, ub, gyro_rotation=G, mode='joint', frame=3837)
    if (reg != prior['registration'] or R0.tolist() != prior['relative_rotation']
            or t0.tolist() != prior['translation_reference_body_m']):
        raise ValueError('exact original pair fit and accepted point mask required')
    planes = {r['frame']: r['joint_plane'] for r in floor['raw_depth_floor_reconstructions']}
    R, t, fitted = fit(a[mask], b[mask], planes[3836], planes[3837])
    keep, residual = inliers(a, b, ua, ub, R, t)
    retained = keep & mask
    checks = dict(all_original_inliers_still_pass=bool(np.all(keep[mask])),
        minimum_matches=bool(retained.sum() >= RULES['minimum_matches']),
        minimum_original_population_inlier_fraction=bool(retained.mean() >= RULES['minimum_inlier_fraction']),
        minimum_grid_cells=bool(min(cells(ua[retained]), cells(ub[retained])) >= RULES['minimum_grid_cells']),
        reference_translation=bool(np.linalg.norm(t) <= RIGID_RULES['maximum_reference_translation_m']),
        incremental_translation=bool(np.linalg.norm(t) <= RIGID_RULES['maximum_increment_translation_m']),
        incremental_rotation=bool(angle(R) <= RIGID_RULES['maximum_increment_rotation_rad']),
        gyro_disagreement=bool(angle(G.T@R) <= RIGID_RULES['maximum_gyro_disagreement_rad']))
    verify(sources)
    verify_artifacts(diagnosis.NATIVE, prior['artifact_sha256'])
    if digest(original.OUTPUT) != ORIGINAL_SHA or digest(diagnosis.OUTPUT) != original.DIAGNOSIS_SHA:
        raise ValueError('original diagnoses changed')
    report = dict(status='PROVISIONAL_FIXED_PAIR_MEASURED_PLANE_FIT_EVALUATED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        original_pair_sha256=ORIGINAL_SHA, original_floor_diagnosis_sha256=original.DIAGNOSIS_SHA,
        artifact_sha256=prior['artifact_sha256'], reference_frame=3836, current_frame=3837,
        original_lifted_matches=len(a), original_inliers=int(mask.sum()),
        original_inliers_retained=int(retained.sum()), original_inliers_removed=False,
        original_residual_rms_m=reg['residual_rms_m'], candidate_fit=fitted,
        original_pair_gate_results=checks, all_listed_pair_gates_pass=all(checks.values()),
        candidate_gyro_disagreement_rad=angle(G.T@R),
        candidate_relative_rotation=R.tolist(), candidate_translation_reference_body_m=t.tolist(),
        relative_rotation_change_rad=angle(R0.T@R), translation_change_m=float(np.linalg.norm(t-t0)),
        exact_original_pair_reconstructed=True, original_raw_input_identity_rechecked=True,
        fixed_original_inlier_set_used=True, new_robust_consensus_selected=False,
        single_posthoc_development_pair=True, native_completion_admitted=False,
        previous_accumulated_drift_repaired=False, retained_reference_history_recomputed=False,
        original_global_floor_rejection_overridden=False, observer_admitted=False,
        controller_changed=False, native_execution=False, navigation_recovered=False,
        real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
    write_json(OUTPUT, report)
    print('FIXED_PAIR_MEASURED_PLANE_FIT', digest(OUTPUT), len(sources), flush=True)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256', 'artifact_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
