"""Numerical consistency/step comparison; no fitted bounds or sensor calibration."""
import numpy as np


def gap_metrics(relation):
    ids = relation['shape_ids']; sources = relation['source_ids']
    joint, pose, floor = [np.asarray(relation[k], float) for k in
                         ('joint_gap_factor_m', 'pose_only_gap_factor_m', 'floor_only_gap_factor_m')]
    if (len(ids) != len(set(ids)) or not ids or len(sources) != len(set(sources)) or not sources
            or any(x.shape != (len(ids), len(sources)) for x in (joint, pose, floor))):
        raise ValueError('aligned nonempty primitive/source factors required')
    observed = np.asarray(relation['all_perturbed_footprints_observed'])
    if observed.shape != (len(ids),) or observed.dtype != bool:
        raise ValueError('explicit paired physical footprint evidence required')
    finite = np.isfinite(joint).all(axis=1) & np.isfinite(pose).all(axis=1) & np.isfinite(floor).all(axis=1)
    if np.any(observed & ~finite): raise ValueError('observed relation cannot have missing numerical factors')
    joint_var = np.sum(joint**2, axis=1); independent_var = np.sum(pose**2+floor**2, axis=1)
    for calculated, key in ((joint_var, 'joint_gap_variance_m2'), (independent_var, 'incorrect_independent_gap_variance_m2')):
        np.testing.assert_allclose(calculated, np.asarray(relation[key], float), rtol=1e-12, atol=1e-20, equal_nan=True)
    rows = []
    for i, sid in enumerate(ids):
        rows.append(dict(shape_id=sid, paired_footprint_observed=bool(observed[i]), numeric_factors_present=bool(finite[i]),
            joint_one_source_unit_scale_m=float(np.sqrt(joint_var[i])) if finite[i] else None,
            incorrect_independent_one_source_unit_scale_m=float(np.sqrt(independent_var[i])) if finite[i] else None,
            joint_minus_split_factor_norm_m=float(np.linalg.norm(joint[i]-pose[i]-floor[i])) if finite[i] else None,
            calibrated_coverage=False, motion_permission=False))
    return rows


def compare(left, right):
    if (left['shape_ids'] != right['shape_ids'] or left['source_ids'] != right['source_ids']
            or left['stored_measured_ns'] != right['stored_measured_ns']
            or left['current_measured_ns'] != right['current_measured_ns']):
        raise ValueError('identical primitive/source/observation identities required across steps')
    a, b = gap_metrics(left), gap_metrics(right)
    left_joint, right_joint = [np.asarray(r['joint_gap_factor_m'], float) for r in (left, right)]
    rows = []
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        finite = x['numeric_factors_present'] and y['numeric_factors_present']
        delta = float(np.linalg.norm(left_joint[i]-right_joint[i])) if finite else None
        scale = max(x['joint_one_source_unit_scale_m'], y['joint_one_source_unit_scale_m']) if finite else None
        rows.append(dict(shape_id=x['shape_id'], step_left=x, step_right=y,
            joint_factor_step_difference_norm_m=delta,
            relative_factor_step_difference=None if not finite else (delta/scale if scale else 0.),
            both_steps_footprint_observed=x['paired_footprint_observed'] and y['paired_footprint_observed'],
            step_stability_accepted=False, calibrated_bound_selected=False))
    return rows
