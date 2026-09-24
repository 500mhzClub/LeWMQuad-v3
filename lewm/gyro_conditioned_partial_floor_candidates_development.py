"""Prune weak-extent floor candidates against their existing gyro prior.

The original selector fits its own normal before the partial-height consumer
checks a gyro-conditioned normal. These need not select the same inliers.
This development fallback only removes original candidates when the full
plane lacks two-axis extent; it preserves all original fully accepted planes.
"""
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.floor_pose_registration_development import unit
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from lewm.partial_floor_height_development import MISSING_EXTENT
from lewm.robust_height_floor_candidates_development import (
    select_clouds as original, PairedHeightCandidates,
    RESIDUAL_M, MINIMUM_CLUSTER_FRACTION, MAXIMUM_REFINEMENTS)


def select_clouds(clouds, up_body):
    masks, receipt = original(clouds, up_body)
    arrays = [np.asarray(p, float) for p in clouds]
    up = unit(up_body)
    plane = fit_joint_plane(*(p[m] for p, m in zip(arrays, masks)), up)
    if plane['available'] or plane['reason'] != MISSING_EXTENT:
        return masks, receipt
    revised = [m.copy() for m in masks]
    minimum = max(100, MINIMUM_CLUSTER_FRACTION * sum(map(len, arrays)))
    for step in range(MAXIMUM_REFINEMENTS):
        selected = [p[m] for p, m in zip(arrays, revised)]
        points = np.concatenate(selected)
        if len(points) < minimum:
            return masks, receipt
        offset = float(points.mean(0) @ up)
        residuals = [np.abs(p @ up - offset) for p in selected]
        maximum = max((float(r.max()) for r in residuals if len(r)), default=0.)
        if maximum <= RESIDUAL_M:
            if all(np.array_equal(a, b) for a, b in zip(masks, revised)):
                return masks, receipt
            count = sum(int(m.sum()) for m in revised)
            return tuple(revised), receipt | dict(
                reason='weak_extent_candidates_conditioned_on_existing_gyro_prior',
                selected_count=count, excluded_count=sum(map(len, arrays))-count,
                before_gyro_pruning_selected_count=receipt['selected_count'],
                gyro_conditioned_pruning_steps=step,
                gyro_conditioned_maximum_residual_m=maximum,
                gyro_conditioned_normal_body=up.tolist(),
                gyro_conditioned_offset_body_m=-offset,
                only_original_candidates_removed=True,
                current_full_normal_measured=False,
                downstream_acceptance_thresholds_changed=False)
        for mask, residual in zip(revised, residuals):
            mask[np.flatnonzero(mask)[residual > RESIDUAL_M]] = False
    return masks, receipt


class GyroConditionedPartialFloorCandidates(PairedHeightCandidates):
    __call__ = bind(PairedHeightCandidates.__call__, select_clouds=select_clouds)
