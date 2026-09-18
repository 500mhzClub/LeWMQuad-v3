"""Weak-extent candidates conditioned on registration's transported normal."""
from functools import partial
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.floor_pose_registration_development import proper, unit
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from lewm.partial_floor_height_development import MISSING_EXTENT
from lewm.robust_height_floor_candidates_development import (
    select_clouds as original, PairedHeightCandidates,
    RESIDUAL_M, MINIMUM_CLUSTER_FRACTION, MAXIMUM_REFINEMENTS)
from lewm.robust_height_floor_tracking_development import RobustHeightFloorRegistration, DESCRIPTION
from lewm.floor_reacquisition_development import ReacquiringFloorRegistration


def transported_normal(anchor, raw):
    """Same rotation composition as the unchanged downstream witness reader.

    The anchor is retained internal registration state. The existing observe
    and evidence-reader paths still validate raw input and the entire anchor.
    This calculation supplies no independent admission of either input.
    """
    A = np.asarray(anchor['current_pose']['rotation_initial_body_from_current_body'])
    B = np.asarray(anchor['original_visual_evidence']['current_pose']['rotation_initial_body_from_current_body'])
    R = np.asarray(raw['current_pose']['rotation_initial_body_from_current_body'])
    normal = unit(anchor['floor_registration']['reference']['joint_plane']['normal_body'])
    return proper(proper(A @ B.T) @ R).T @ normal


def select_clouds(clouds, up_body, *, residual_normal):
    masks, receipt = original(clouds, up_body)
    arrays = [np.asarray(p, float) for p in clouds]
    # Pool selection and missing-plane metadata retain their original up.
    plane = fit_joint_plane(*(p[m] for p, m in zip(arrays, masks)), up_body)
    if plane['available'] or plane['reason'] != MISSING_EXTENT:
        return masks, receipt
    normal = np.asarray(residual_normal, float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1., atol=1e-10, rtol=0):
        raise ValueError('unit transported floor-reference normal required')
    revised = [m.copy() for m in masks]
    minimum = max(100, MINIMUM_CLUSTER_FRACTION * sum(map(len, arrays)))
    for step in range(MAXIMUM_REFINEMENTS):
        selected = [p[m] for p, m in zip(arrays, revised)]
        count = sum(map(len, selected))
        if count < minimum: return masks, receipt
        # Match the partial-height consumer's camera-weighted mean order.
        mean = sum(len(p)*p.mean(0) for p in selected if len(p)) / count
        offset = float(mean @ normal)
        residuals = [np.abs(p @ normal-offset) for p in selected]
        maximum = max(float(r.max()) for r in residuals if len(r))
        if maximum <= RESIDUAL_M:
            if all(np.array_equal(a,b) for a,b in zip(masks,revised)): return masks,receipt
            return tuple(revised), receipt | dict(
                reason='weak_extent_candidates_conditioned_on_transported_reference',
                selected_count=count, excluded_count=sum(map(len, arrays))-count,
                before_transport_pruning_selected_count=receipt['selected_count'],
                transport_conditioned_pruning_steps=step,
                transport_conditioned_maximum_residual_m=maximum,
                transport_conditioned_normal_body=normal.tolist(),
                transport_conditioned_offset_body_m=-offset,
                only_original_candidates_removed=True, current_full_normal_measured=False,
                downstream_acceptance_thresholds_changed=False)
        for mask,residual in zip(revised,residuals):
            mask[np.flatnonzero(mask)[residual > RESIDUAL_M]] = False
    return masks, receipt


class TransportConditionedCandidates(PairedHeightCandidates):
    def __init__(self, primary, auxiliary, *, residual_normal):
        super().__init__(primary, auxiliary)
        self.residual_normal = np.asarray(residual_normal, float).copy()

    def __call__(self, *args):
        return bind(PairedHeightCandidates.__call__, select_clouds=partial(
            select_clouds, residual_normal=self.residual_normal))(self, *args)


class _TransportRegistration(RobustHeightFloorRegistration):
    def observe(self, policy, primary, auxiliary, raw, *, now_ns):
        factory = PairedHeightCandidates if self.anchor is None else partial(
            TransportConditionedCandidates, residual_normal=transported_normal(self.anchor, raw))
        return bind(RobustHeightFloorRegistration.observe, PairedHeightCandidates=factory,
            DESCRIPTION=DESCRIPTION | dict(weak_extent_pruning_uses_transported_reference_normal=True,
                original_pool_up_and_plane_metadata_preserved=True))(
                    self, policy, primary, auxiliary, raw, now_ns=now_ns)


class TransportConditionedReacquiringRegistration(ReacquiringFloorRegistration, _TransportRegistration):
    pass
