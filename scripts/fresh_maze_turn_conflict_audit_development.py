"""Read-only terminal ray diagnosis. No changed decision or motion permission."""
import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.uncertain_ray_memory_development import query_envelopes, transport_radius


def window(point, radius):
    transform = np.asarray(BODY_FROM_OPTICAL)
    optical = (point-transform[:3, 3])@transform[:3, :3]
    low, high = optical[2]-radius, optical[2]+radius
    if high+.04 < .2:
        return None
    if low <= 0:
        left, right, top, bottom = 0, 639, 0, 479
    else:
        denominators = np.maximum([low, high], 1e-12)
        u = FOCAL*(optical[0]+radius*np.array([-1, 1]))[:, None]/denominators+319.5
        v = FOCAL*(optical[1]+radius*np.array([-1, 1]))[:, None]/denominators+239.5
        left, right = int(np.floor(np.clip(u.min(), -2, 641))), int(np.floor(np.clip(u.max(), -2, 641)))+1
        top, bottom = int(np.floor(np.clip(v.min(), -2, 481))), int(np.floor(np.clip(v.max(), -2, 481)))+1
    complete = low >= .2 and high <= 5 and left >= 1 and right <= 638 and top >= 1 and bottom <= 478
    left, right, top, bottom = max(0, left), min(639, right), max(0, top), min(479, bottom)
    if left > right or top > bottom:
        return None
    return left, right, top, bottom, low, high, complete


def diagnose(controller, policy, cameras):
    """Evaluator camera poses classify hits ONLY after sensor-only replay."""
    rays = controller.sensor_memory.rays
    volume = controller.regions.volume(policy)
    points, roles = volume['points_body_m'], volume['ground_support_allowed']
    now = policy['sensor_state']['decision_ns']
    original = rays.query(points, roles, now_ns=now)
    frames = list(rays.frames)
    if frames[-1]['measured_ns'] != now:
        frames.append(rays.latest_frame)
    by_ns = {int(round(c['timestamp_s']*1e9)): np.asarray(c['world_from_optical']) for c in cameras}
    reference = points@rays.rotation.T+rays.position
    conflict_union = np.zeros(len(points), bool)
    zero_free = conflict_union.copy(); zero_support = conflict_union.copy(); zero_conflict = conflict_union.copy()
    floor_hits = np.zeros(len(points), int); nonfloor_hits = floor_hits.copy()
    rows = []
    for stored in frames:
        body = points if stored['measured_ns'] == now else (reference-stored['position'])@stored['rotation']
        radius = transport_radius(points, rays.latest_frame, stored)
        evidence = stored['evidence']
        measured = query_envelopes(evidence, body, radius, roles)
        independent = query_envelopes(evidence, body, radius, roles, backend='reference')
        for key in ('free', 'observed_ground_support', 'contradictory_or_near_surface'):
            np.testing.assert_array_equal(measured[key], independent[key])
        zero = query_envelopes(evidence, body, np.zeros(len(points)), roles)
        zero_free |= zero['free']; zero_support |= zero['observed_ground_support']; zero_conflict |= zero['contradictory_or_near_surface']
        conflict = measured['contradictory_or_near_surface']; conflict_union |= conflict
        counts = dict(incomplete_window=0, support_role_denied=0, non_ground_pixels_in_window=0,
                      invalid_pixels_in_window=0, height_spread_exceeds6mm=0, height_and_radius_exceeds60mm=0,
                      floor_only_near_hits=0, any_nonfloor_near_hits=0)
        T = by_ns[stored['measured_ns']]
        for i in np.flatnonzero(conflict):
            left, right, top, bottom, low, high, complete = window(body[i], radius[i])
            sl = np.s_[top:bottom+1, left:right+1]
            d, valid, ground, h = [evidence[k][sl] for k in ('depth', 'valid', 'ground', 'height')]
            hit = valid & (d >= low-.04) & (d <= high+.04)
            assert hit.any()
            yy, xx = np.nonzero(hit); z = d[yy, xx]
            optical = np.column_stack((z*(xx+left+.5-320)/FOCAL, z*(yy+top+.5-240)/FOCAL, z))
            world = optical@T[:3, :3].T+T[:3, 3]
            floor = np.abs(world[:, 2]) <= .001
            floor_hits[i] += int(floor.sum()); nonfloor_hits[i] += int((~floor).sum())
            counts['floor_only_near_hits'] += int(floor.all())
            counts['any_nonfloor_near_hits'] += int(not floor.all())
            counts['incomplete_window'] += int(not complete)
            counts['support_role_denied'] += int(not roles[i])
            counts['non_ground_pixels_in_window'] += int(not ground.all())
            counts['invalid_pixels_in_window'] += int(not valid.all())
            counts['height_spread_exceeds6mm'] += int(np.ptp(h) > .006)
            qh = body[i]@evidence['up']
            counts['height_and_radius_exceeds60mm'] += int(max(abs(qh-h.min()), abs(qh-h.max()))+radius[i] > .06)
        rows.append(dict(measured_ns=stored['measured_ns'], current_view=stored['measured_ns'] == now,
                         conflict_samples=int(conflict.sum()), maximum_radius_m=float(radius.max()),
                         zero_radius_conflict_samples=int(zero['contradictory_or_near_surface'].sum()),
                         overlapping_failure_counts=counts))
    np.testing.assert_array_equal(conflict_union, original['contradictory_or_near_surface'])
    return dict(sample_points=len(points), conflict_samples=int(conflict_union.sum()),
        unknown_samples=int(original['unknown_or_blocked'].sum()),
        conflicted_support_role_samples=int((conflict_union & roles).sum()),
        conflicted_non_support_role_samples=int((conflict_union & ~roles).sum()),
        samples_with_floor_near_hits=int((floor_hits > 0).sum()),
        samples_with_nonfloor_near_hits=int((nonfloor_hits > 0).sum()),
        compiled_reference_decisions_exact=True, bands=volume['bands'], views=rows,
        zero_radius_diagnostic_only=dict(conflict_samples=int(zero_conflict.sum()),
            unknown_samples=int((~((zero_free | zero_support) & ~zero_conflict)).sum()),
            pose_estimates_unchanged=True, uncertainty_removal_is_not_validated=True),
        evaluator_floor_classification='near-hit reconstructed world abs(z)<=1mm; diagnostic only',
        controller_modified=False, motion_permitted=False, future_gait_qualified=False)
