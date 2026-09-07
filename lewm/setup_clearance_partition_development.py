"""Keep supplied initial clearance separate from residual observation queries.

This partitions one box into at most one setup-covered box and six residual
boxes. It never fills their convex hull, declares residuals clear, or establishes
ground support. The complete original query remains subject to observed vetoes.
"""
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.continuous_startup_handoff_development import ContinuousStartupHandoff
from lewm.setup_region_prior_development import SetupRegionPrior


def partition_setup_clearance(region, lower, upper, radius, *, identity, now_ns, through_ns, observed_conflict):
    if (not isinstance(region, SetupRegionPrior) or type(now_ns) is not int or type(through_ns) is not int
            or through_ns < now_ns or type(observed_conflict) is not bool
            or type(radius) not in (float, int) or not np.isfinite(radius) or radius < 0):
        raise SensorContractError('explicit finite setup query, horizon, radius and observed veto required')
    low, high = np.asarray(lower, float), np.asarray(upper, float)
    if low.shape != (3,) or high.shape != (3,): raise SensorContractError('one initial-body query box required')
    # Reuse identity, finite bounds and radius validation without treating the
    # full query's containment boolean as a partial-intersection proof.
    region.query([low], [high], [radius], identity=identity, now_ns=now_ns, observed_conflict=[observed_conflict])
    with np.errstate(over='ignore', invalid='ignore'):
        low, high = low-radius, high+radius
    if not np.isfinite([low, high]).all(): raise SensorContractError('representable expanded query required')
    original = [low.tolist(), high.tolist()]
    active = region.anchor_ns <= now_ns <= through_ns <= region.valid_until_ns
    covered = []; residual = [original]
    if active and not observed_conflict:
        # Shrink the supplied region by a numerical guard, never expand it.
        scale = max(1., float(np.max(np.abs([low, high, region.lower_initial_body_m, region.upper_initial_body_m]))))
        margin = 1e-10+128*np.finfo(float).eps*scale
        inner_low = np.maximum(low, np.asarray(region.lower_initial_body_m)+margin)
        inner_high = np.minimum(high, np.asarray(region.upper_initial_body_m)-margin)
        if np.all(inner_low <= inner_high):
            covered = [[inner_low.tolist(), inner_high.tolist()]]; residual = []
            work_low, work_high = low.copy(), high.copy()
            for axis in range(3):
                if work_low[axis] < inner_low[axis]:
                    cut = work_high.copy(); cut[axis] = inner_low[axis]
                    residual.append([work_low.tolist(), cut.tolist()]); work_low[axis] = inner_low[axis]
                if work_high[axis] > inner_high[axis]:
                    cut = work_low.copy(); cut[axis] = inner_high[axis]
                    residual.append([cut.tolist(), work_high.tolist()]); work_high[axis] = inner_high[axis]
    return dict(identity=identity, measured_ns=now_ns, through_ns=through_ns,
        setup_evidence_sha256=region.setup_evidence_sha256, setup_region_active_through_horizon=active,
        whole_query_for_observed_veto=original, observed_conflict=observed_conflict,
        setup_covered_boxes=covered, requires_sensor_evidence_boxes=residual,
        entire_query_conditionally_setup_nonfloor_clear=bool(covered and not residual and not observed_conflict),
        supplied_evidence_is_observed=False, residual_is_known_clear=False,
        ground_support_permission=False, future_gait_qualified=False, navigation_action_permitted=False)


def current_body_setup_partition(owner, *, now_ns):
    """Bind current measured posture and unchanged fusion scales to the prior.

No future posture or sweep is inferred. The additional rotation displacement
uses the all-posture radius with the existing uncalibrated orientation scale;
it does not turn that scale into a calibrated bound.
"""
    if not isinstance(owner, ContinuousStartupHandoff): raise SensorContractError('live continuous state owner required')
    snapshot = owner.navigation_snapshot(now_ns=now_ns)
    rays, memory = owner._memory._rays, owner._memory
    angle = rays.fusion['assumptions']['scale_multiplier'] * np.sqrt(rays.fusion['orientation_variance_proxy_rad2'])
    rotation_scale = 2*owner._startup.radius*np.sin(min(float(angle), np.pi)/2)
    radius = float(.04+rays.fusion['position_error_scale_m']+rotation_scale)
    shapes = memory._geometry.supports(memory._joints, rays.rotation)['shapes']
    observed = snapshot['observed_current_posture']
    conflicts = set(observed['non_floor_conflict']) | set(observed['floor_penetration'])
    rows = [dict(shape_id=s['shape_id'], partition=partition_setup_clearance(owner._region,
        s['lower']+rays.position, s['upper']+rays.position, radius,
        identity=rays.identity, now_ns=now_ns, through_ns=now_ns,
        observed_conflict=s['shape_id'] in conflicts)) for s in shapes]
    return dict(measured_ns=now_ns, current_measured_posture_only=True, point_expansion_m=radius,
        setup_conditionally_covered_shapes=[r['shape_id'] for r in rows if r['partition']['entire_query_conditionally_setup_nonfloor_clear']],
        shapes_requiring_additional_observation=[r['shape_id'] for r in rows if r['partition']['requires_sensor_evidence_boxes']],
        observed_current_posture=observed, per_shape=rows, supplied_error_scales_validated=False,
        ground_support_permission=False, future_gait_qualified=False, navigation_action_permitted=False)
