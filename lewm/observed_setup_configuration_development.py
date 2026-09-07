"""Observation-bound non-floor clearance for supplied articulated configurations.

An evaluated configuration is NOT a predicted gait or a swept trajectory. This
consumer combines an explicit initial-region condition with actual depth-backed
residual-box evidence, retaining every view's whole-query obstacle veto. Ground
relations remain separate candidates, never contact or navigation permission.
"""
import hashlib
from itertools import product

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.continuous_startup_handoff_development import ContinuousStartupHandoff
from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.primitive_obstacle_memory_development import non_floor_box_evidence
from lewm.setup_clearance_partition_development import partition_setup_clearance
from lewm.uncertain_ray_memory_development import transport_radius

_CORNERS = np.asarray(list(product((0, 1), repeat=3)), bool)


def inverse_pose_box(box, stored):
    """Initial-frame box -> conservative observed-body AABB and pose allowance.

The query box already contains its own stated uncertainty. Add the stored
camera/body-pose proxy independently; do not assume cancellation for a supplied
initial-frame configuration. This is conservative proxy accounting, not a newly
calibrated covariance or physical bound.
"""
    low, high = np.asarray(box, float)
    vertices = np.where(_CORNERS, high, low)
    position = np.asarray(stored['position']); rotation = np.asarray(stored['rotation'])
    translation = float(stored['position_scale_m']); angle = float(stored['orientation_scale_rad'])
    if (low.shape != (3,) or high.shape != (3,) or np.any(low > high)
            or position.shape != (3,) or rotation.shape != (3, 3)
            or not all(np.isfinite(v).all() for v in (vertices, position, rotation, translation, angle))
            or translation < 0 or angle < 0 or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-12, rtol=0)
            or abs(np.linalg.det(rotation)-1) > 1e-12):
        raise SensorContractError('finite query and proper observed pose with nonnegative scales required')
    lever = float(np.linalg.norm(vertices-position, axis=1).max())+translation
    radius = translation+2*np.sin(min(angle, np.pi)/2)*lever
    body = (vertices-position)@rotation
    lo, hi = body.min(axis=0)-radius, body.max(axis=0)+radius
    if not all(np.isfinite(v).all() for v in (lo, hi, radius)):
        raise SensorContractError('representable observation-frame enclosure required')
    return lo, hi, float(radius)


def query_configuration(owner, position, rotation, joints, point_error_m, *, now_ns, through_ns,
                        backend='compiled', reference='initial_body'):
    if not isinstance(owner, ContinuousStartupHandoff): raise SensorContractError('live continuous owner required')
    owner.navigation_snapshot(now_ns=now_ns)
    p, R, q = [np.asarray(v, float) for v in (position, rotation, joints)]
    if (p.shape != (3,) or R.shape != (3, 3) or q.shape != (12,)
            or not all(np.isfinite(v).all() for v in (p, R, q))
            or not np.allclose(R.T@R, np.eye(3), atol=1e-12, rtol=0) or abs(np.linalg.det(R)-1) > 1e-12
            or type(point_error_m) not in (int, float) or not np.isfinite(point_error_m) or point_error_m < 0
            or type(now_ns) is not int or type(through_ns) is not int or through_ns < now_ns
            or backend not in ('compiled', 'reference') or reference not in ('initial_body', 'current_body')):
        raise SensorContractError('proper finite supplied configuration, error, clock and backend required')
    memory, rays = owner._memory, owner._memory._rays
    if memory._last_ns != now_ns: raise SensorContractError('current prepared observations required')
    geometry = memory._geometry
    requested_position, requested_rotation = p.copy(), R.copy()
    query_error = float(point_error_m)
    if reference == 'current_body':
        angle = rays.latest_frame['orientation_scale_rad']
        query_error += (rays.fusion['position_error_scale_m']
            +2*np.sin(min(float(angle), np.pi)/2)*(owner._startup.radius+float(np.linalg.norm(p))))
        p, R = rays.position+rays.rotation@p, rays.rotation@R
    shapes = geometry.supports(q, R)['shapes']; ids = tuple(s['shape_id'] for s in shapes)
    physical = [[s['lower']+p, s['upper']+p] for s in shapes]
    current_local_shapes = geometry.supports(q, requested_rotation)['shapes'] if reference == 'current_body' else None
    partitions = [partition_setup_clearance(owner._region, *box, float(query_error+.04),
        identity=rays.identity, now_ns=now_ns, through_ns=through_ns, observed_conflict=False) for box in physical]
    # Full query first, then exact residuals. Every full query is checked in
    # every retained view, even when the prior covers it completely.
    boxes = [r['whole_query_for_observed_veto'] for r in partitions]
    shape_index = list(range(len(ids))); residual_indices = []
    for i, row in enumerate(partitions):
        indices = []
        for box in row['requires_sensor_evidence_boxes']:
            indices.append(len(boxes)); boxes.append(box); shape_index.append(i)
        residual_indices.append(indices)
    shape_index = np.asarray(shape_index, int)
    clear_sources = [[] for _ in boxes]; conflict_sources = [[] for _ in ids]
    penetration_sources = [[] for _ in ids]; contact_sources = [[] for _ in ids]; views = []
    frames = list(rays.frames)
    if frames[-1]['measured_ns'] != now_ns: frames.append(rays.latest_frame)
    stamps = [f['measured_ns'] for f in frames]
    if stamps != sorted(set(stamps)) or any(t > now_ns or t < owner._region.anchor_ns for t in stamps):
        raise SensorContractError('ordered same-episode retained observation clocks required')
    for stored in frames:
        stamp = stored['measured_ns']; frame = memory._prepared[stamp]
        hypothesis = memory._hypotheses[stamp]; cell = hypothesis.cell_for(frame)
        raw_hash = hashlib.sha256(stored['evidence']['depth'].tobytes()+stored['evidence']['valid'].tobytes()).hexdigest()
        if raw_hash != frame.depth_sha256: raise SensorContractError('prepared and retained raw depth identities differ')
        transformed = [inverse_pose_box(box, stored) for box in boxes]
        if reference == 'current_body':
            # Preserve the established common-pose cancellation for the actual
            # whole physical query. Initial-region residual boxes are fixed in
            # the initial frame and retain conservative absolute-pose accounting.
            local_shapes = geometry.supports(q, np.asarray(stored['rotation']).T@R)['shapes']
            t = (p-stored['position'])@stored['rotation']
            for i, shape in enumerate(local_shapes):
                local = current_local_shapes[i]
                points_current = np.where(_CORNERS, local['upper'], local['lower'])+requested_position
                allowance = float(transport_radius(points_current, rays.latest_frame, stored).max())
                error = point_error_m+allowance+.04
                transformed[i] = (shape['lower']+t-error, shape['upper']+t+error, allowance)
        low, high = np.asarray([b[0] for b in transformed]), np.asarray([b[1] for b in transformed])
        # Full-query pose allowances also bound each contained physical primitive.
        errors = {sid: point_error_m+transformed[i][2] for i, sid in enumerate(ids)}
        plane = None; exemption = np.zeros(len(ids), bool); foot_candidate = exemption.copy()
        covered = exemption.copy(); penetration = exemption.copy()
        if cell is not None:
            floor = frame.query(geometry, q, cell,
                rotation_observation_from_body=np.asarray(stored['rotation']).T@R,
                translation_observation_from_body=(p-stored['position'])@stored['rotation'],
                point_error_by_shape=errors, floor_backend='cached' if backend == 'compiled' else 'reference', **memory._errors)
            plane = (floor['plane_anchor_observation_m'], floor['plane_normal_observation'])
            for i, gap in enumerate(floor['gap_bounds']['primitives']):
                if gap['shape_id'] != ids[i]: raise SensorContractError('exact physical primitive ordering required')
                gl, gh = gap['minimum_gap_lower_m'], gap['minimum_gap_upper_m']
                covered[i] = floor['floor_coverage'][ids[i]]
                foot_candidate[i] = ids[i] in FOOT_SHAPES and gl <= 0 <= gh
                exemption[i] = gl > 0 or foot_candidate[i]
                penetration[i] = covered[i] and gh < 0
        clear, conflict = np.zeros(len(boxes), bool), np.zeros(len(boxes), bool)
        scanned = 0
        for allowed in (False, True):
            selected = np.flatnonzero(exemption[shape_index] == allowed)
            if not len(selected): continue
            observed = non_floor_box_evidence(frame, low[selected], high[selected], plane=plane if allowed else None,
                normal_error=memory._errors['normal_error'], plane_offset_error=memory._errors['plane_offset_error'],
                range_error_m=memory._range_error, backend=backend)
            clear[selected] = observed['non_floor_clearance']; conflict[selected] = observed['non_floor_conflict']
            scanned += int(observed['scanned_pixels'].sum())
        for i in range(len(ids)):
            # Only the whole physical query supplies obstacle vetoes. Residual
            # enclosures may contain extra uncertainty combinations; a near
            # return there prevents that residual's positive clearance but is
            # not promoted into an observation of the actual physical shape.
            if conflict[i]: conflict_sources[i].append(stamp)
            if penetration[i]: penetration_sources[i].append(stamp)
            if covered[i] and foot_candidate[i] and clear[i] and not conflict[i]: contact_sources[i].append(stamp)
        for i in np.flatnonzero(clear & ~conflict): clear_sources[int(i)].append(stamp)
        views.append(dict(measured_ns=stamp, depth_sha256=frame.depth_sha256, hypothesis_cell_rc=cell,
            full_queries_checked=len(ids), boxes_checked=len(boxes), scanned_pixels=scanned,
            floor_covered_primitives=int(covered.sum()), relation_compatible_floor_exemptions=int(exemption.sum())))
    rows = []
    for i, sid in enumerate(ids):
        veto = bool(conflict_sources[i] or penetration_sources[i])
        residual = residual_indices[i]
        supplied = bool(partitions[i]['setup_covered_boxes']) and not veto
        conditional = not veto and all(clear_sources[j] for j in residual)
        # A nonempty physical query is either setup-covered, residual, or both.
        if not partitions[i]['setup_covered_boxes'] and not residual:
            raise SensorContractError('nonempty configuration query has no coverage decomposition')
        rows.append(dict(shape_id=sid, partition_before_observed_veto=partitions[i],
            residual_clearance_sources=[list(clear_sources[j]) for j in residual],
            whole_box_clearance_sources=list(clear_sources[i]), obstacle_veto_sources=conflict_sources[i],
            floor_penetration_sources=penetration_sources[i], supplied_clearance_used=supplied,
            conditional_nonfloor_clearance=bool(conditional),
            observed_foot_contact_candidate=bool(contact_sources[i] and not veto),
            observed_foot_candidate_sources=contact_sources[i] if not veto else []))
    return dict(measured_ns=now_ns, through_ns=through_ns, identity=rays.identity,
        supplied_configuration=dict(reference=reference, position_in_reference_m=requested_position.tolist(),
            rotation_reference_from_body=requested_rotation.tolist(),
            resolved_position_initial_body_m=p.tolist(), resolved_rotation_initial_body_from_body=R.tolist(),
            joint_position_rad=q.tolist(), point_error_m=float(point_error_m), geometry_padding_m=.04,
            initial_partition_point_error_m=query_error),
        primitives=rows, observation_bindings=views,
        all_primitives_conditionally_nonfloor_clear=all(r['conditional_nonfloor_clearance'] for r in rows),
        static_scene_assumed=True, supplied_error_scales_validated=False,
        configuration_is_execution_prediction=False, continuous_swept_volume_established=False,
        ground_support_permission=False, future_gait_qualified=False, navigation_action_permitted=False)


def query_current_configuration(owner, *, now_ns, backend='compiled'):
    owner.navigation_snapshot(now_ns=now_ns)
    result = query_configuration(owner, [0.,0.,0.], np.eye(3), owner._memory._joints, 0.,
                                 now_ns=now_ns, through_ns=now_ns, backend=backend, reference='current_body')
    return result | {'current_measured_configuration': True}
