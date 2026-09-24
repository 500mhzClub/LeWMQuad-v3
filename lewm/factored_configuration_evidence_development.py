"""Separate observed obstacle, floor relation and visibility evidence.

New development consumer; the recorded predecessor is unchanged. Plane-family
classification is not contact permission. No future motion or calibrated-error
claim follows from a supplied static configuration.
"""
import hashlib
from itertools import product

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.continuous_startup_handoff_development import ContinuousStartupHandoff
from lewm.observed_setup_configuration_development import inverse_pose_box
from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.primitive_obstacle_memory_development import non_floor_box_evidence
from lewm.setup_clearance_partition_development import partition_setup_clearance
from lewm.uncertain_ray_memory_development import transport_radius

_CORNERS = np.asarray(list(product((0, 1), repeat=3)), bool)


def ground_summary(shape_id, witnesses):
    """Do not turn a wide interval into a contradiction of a narrower one.

    Positive separation requires one COMPLETE observed footprint. We do not
    merge partial floor footprints, fit a common plane, or average intervals.
    The residual checks at common physical-box vertices are necessary
    compatibility checks, not proof of a shared surface or calibrated bounds.
    Every covered penetration and incompatible covered-plane witness is kept.
    """
    covered = [w for w in witnesses if w['floor_coverage']]
    separated = [w['measured_ns'] for w in covered if w['gap']['minimum_gap_lower_m'] > 0]
    penetrated = [w['measured_ns'] for w in covered if w['gap']['minimum_gap_upper_m'] < 0]
    ambiguous = [w['measured_ns'] for w in covered
                 if w['gap']['minimum_gap_lower_m'] <= 0 <= w['gap']['minimum_gap_upper_m']]
    incompatible = []
    for i, left in enumerate(covered):
        for right in covered[i+1:]:
            lower = np.maximum(left['plane_vertex_residual_lower_m'], right['plane_vertex_residual_lower_m'])
            upper = np.minimum(left['plane_vertex_residual_upper_m'], right['plane_vertex_residual_upper_m'])
            if np.any(lower > upper): incompatible.append([left['measured_ns'], right['measured_ns']])
    veto = bool(penetrated or incompatible)
    if incompatible: status = 'INCOMPATIBLE_COVERED_GROUND_EVIDENCE'
    elif penetrated: status = 'OBSERVED_PENETRATION_UNDER_SUPPLIED_MODEL'
    elif separated: status = 'SINGLE_VIEW_OBSERVED_SEPARATION_UNDER_SUPPLIED_MODEL'
    elif ambiguous: status = 'FOOT_CONTACT_CANDIDATE_ONLY' if shape_id in FOOT_SHAPES else 'GROUND_INTERSECTION_POSSIBLE'
    else: status = 'UNKNOWN_FLOOR_COVERAGE'
    return dict(status=status, observed_separation_sources=separated,
        observed_penetration_sources=penetrated, possible_intersection_sources=ambiguous,
        incompatible_covered_plane_pairs=incompatible,
        physical_floor_separation_observed=bool(separated and not veto),
        observed_foot_contact_candidate=bool(shape_id in FOOT_SHAPES and ambiguous and not veto),
        common_surface_identity_established=False, contact_permitted=False)


def query_factored_configuration(owner, position, rotation, joints, point_error_m, *,
                                now_ns, through_ns, backend='compiled'):
    """Query a supplied CURRENT-BODY-relative configuration, not an action.

    Non-floor evidence always uses its bound measured plane family when one
    exists; physical ground intersection is reported separately. All residual
    volume still needs positive complete-box evidence. The initial region
    supplies non-floor clearance only and cannot grant ground support.
    """
    if not isinstance(owner, ContinuousStartupHandoff): raise SensorContractError('live continuous owner required')
    owner.navigation_snapshot(now_ns=now_ns)
    offset, local_R, q = [np.asarray(v, float) for v in (position, rotation, joints)]
    if (offset.shape != (3,) or local_R.shape != (3, 3) or q.shape != (12,)
            or not all(np.isfinite(v).all() for v in (offset, local_R, q))
            or not np.allclose(local_R.T@local_R, np.eye(3), atol=1e-12, rtol=0)
            or abs(np.linalg.det(local_R)-1) > 1e-12
            or type(point_error_m) not in (float, int) or not np.isfinite(point_error_m) or point_error_m < 0
            or type(now_ns) is not int or type(through_ns) is not int or through_ns < now_ns
            or backend not in ('compiled', 'reference')):
        raise SensorContractError('proper finite current-body configuration, error, clock and backend required')
    memory, rays = owner._memory, owner._memory._rays
    if memory._last_ns != now_ns: raise SensorContractError('current prepared observations required')
    geometry = memory._geometry
    p, R = rays.position+rays.rotation@offset, rays.rotation@local_R
    angle = float(rays.latest_frame['orientation_scale_rad'])
    global_error = float(point_error_m+rays.fusion['position_error_scale_m']
        +2*np.sin(min(angle, np.pi)/2)*(owner._startup.radius+float(np.linalg.norm(offset))))
    initial_shapes = geometry.supports(q, R)['shapes']
    local_shapes = geometry.supports(q, local_R)['shapes']
    body_shapes = geometry.supports(q, np.eye(3))['shapes']
    ids = tuple(s['shape_id'] for s in initial_shapes)
    partitions = [partition_setup_clearance(owner._region, s['lower']+p, s['upper']+p, global_error+.04,
        identity=rays.identity, now_ns=now_ns, through_ns=through_ns, observed_conflict=False) for s in initial_shapes]
    boxes = [row['whole_query_for_observed_veto'] for row in partitions]
    residual_indices = []
    for row in partitions:
        indices = []
        for box in row['requires_sensor_evidence_boxes']:
            indices.append(len(boxes)); boxes.append(box)
        residual_indices.append(indices)
    clear_sources = [[] for _ in boxes]; conflict_sources = [[] for _ in ids]
    ground_witnesses = [[] for _ in ids]; bindings = []
    frames = list(rays.frames)
    if frames[-1]['measured_ns'] != now_ns: frames.append(rays.latest_frame)
    stamps = [f['measured_ns'] for f in frames]
    if stamps != sorted(set(stamps)) or any(t > now_ns or t < owner._region.anchor_ns for t in stamps):
        raise SensorContractError('ordered same-episode observations required')
    for stored in frames:
        stamp = stored['measured_ns']; frame = memory._prepared[stamp]
        cell = memory._hypotheses[stamp].cell_for(frame)
        raw_hash = hashlib.sha256(stored['evidence']['depth'].tobytes()+stored['evidence']['valid'].tobytes()).hexdigest()
        if raw_hash != frame.depth_sha256: raise SensorContractError('prepared and raw depth identities differ')
        obs_R = np.asarray(stored['rotation']).T@R
        obs_p = (p-stored['position'])@stored['rotation']
        observed_shapes = geometry.supports(q, obs_R)['shapes']
        transformed = [inverse_pose_box(box, stored) for box in boxes]
        errors = {}
        for i, shape in enumerate(observed_shapes):
            points = np.where(_CORNERS, local_shapes[i]['upper'], local_shapes[i]['lower'])+offset
            allowance = float(transport_radius(points, rays.latest_frame, stored).max())
            error = point_error_m+allowance
            errors[ids[i]] = error
            transformed[i] = (shape['lower']+obs_p-(error+.04), shape['upper']+obs_p+(error+.04), allowance)
        floor = None if cell is None else frame.query(geometry, q, cell,
            rotation_observation_from_body=obs_R, translation_observation_from_body=obs_p,
            point_error_by_shape=errors, floor_backend='cached' if backend == 'compiled' else 'reference', **memory._errors)
        plane = None if floor is None else (floor['plane_anchor_observation_m'], floor['plane_normal_observation'])
        observed = non_floor_box_evidence(frame, np.asarray([b[0] for b in transformed]),
            np.asarray([b[1] for b in transformed]), plane=plane,
            normal_error=memory._errors['normal_error'], plane_offset_error=memory._errors['plane_offset_error'],
            range_error_m=memory._range_error, backend=backend)
        for j in np.flatnonzero(observed['non_floor_clearance'] & ~observed['non_floor_conflict']):
            clear_sources[int(j)].append(stamp)
        for i, sid in enumerate(ids):
            if observed['non_floor_conflict'][i]: conflict_sources[i].append(stamp)
            gap = None if floor is None else floor['gap_bounds']['primitives'][i]
            if gap is not None and gap['shape_id'] != sid: raise SensorContractError('physical primitive ordering mismatch')
            witness = dict(measured_ns=stamp, depth_sha256=frame.depth_sha256, hypothesis_cell_rc=cell,
                gap=gap, floor_coverage=False if floor is None else bool(floor['floor_coverage'][sid]),
                plane_vertex_residual_lower_m=None, plane_vertex_residual_upper_m=None)
            if floor is not None:
                # Common vertices in the SAME supplied articulated body frame.
                a = np.asarray(floor['gap_bounds']['plane_anchor_body_m'])
                n = np.asarray(floor['gap_bounds']['plane_normal_body'])
                vertices = np.where(_CORNERS, body_shapes[i]['upper'], body_shapes[i]['lower'])
                nominal = (vertices-a)@n
                # These are virtual AABB vertices, not material points. A
                # physical endpoint error e can move each coordinate support
                # by e, hence the corresponding box corner by sqrt(3)*e.
                # Transport the virtual vertices themselves: rotating a
                # body-frame AABB can extend beyond the primitive's directly
                # computed current-frame AABB used for physical queries.
                vertex_transport = transport_radius(vertices@local_R.T+offset, rays.latest_frame, stored)
                vertex_error = np.sqrt(3.)*point_error_m+vertex_transport
                error = (memory._errors['normal_error']*np.linalg.norm(vertices-a, axis=1)
                    +memory._errors['plane_offset_error']+(1+memory._errors['normal_error'])*vertex_error)
                rounding = 1e-12+128*np.finfo(float).eps*(np.abs(nominal)+np.linalg.norm(vertices-a,axis=1)+error)
                witness['plane_vertex_residual_lower_m'] = (nominal-error-rounding).tolist()
                witness['plane_vertex_residual_upper_m'] = (nominal+error+rounding).tolist()
            ground_witnesses[i].append(witness)
        bindings.append(dict(measured_ns=stamp, depth_sha256=frame.depth_sha256, hypothesis_cell_rc=cell,
            boxes_checked=len(boxes), scanned_pixels=int(observed['scanned_pixels'].sum()),
            floor_classification_available=plane is not None))
    rows = []
    for i, sid in enumerate(ids):
        ground = ground_summary(sid, ground_witnesses[i])
        nonfloor = not conflict_sources[i] and all(clear_sources[j] for j in residual_indices[i])
        if not partitions[i]['setup_covered_boxes'] and not residual_indices[i]:
            raise SensorContractError('nonempty physical query must have a coverage decomposition')
        ground_veto = bool(ground['observed_penetration_sources'] or ground['incompatible_covered_plane_pairs'])
        rows.append(dict(shape_id=sid, partition_before_observed_veto=partitions[i],
            supplied_nonfloor_clearance_used=bool(partitions[i]['setup_covered_boxes'] and not conflict_sources[i]),
            nonfloor_conflict_sources=conflict_sources[i], whole_box_nonfloor_clearance_sources=clear_sources[i],
            residual_nonfloor_clearance_sources=[clear_sources[j] for j in residual_indices[i]],
            conditional_nonfloor_clearance=bool(nonfloor), ground=ground, ground_witnesses=ground_witnesses[i],
            conditional_observed_separation=bool(nonfloor and ground['physical_floor_separation_observed'] and not ground_veto),
            contact_candidate_with_nonfloor_clearance=bool(nonfloor and ground['observed_foot_contact_candidate']),
            contact_permitted=False))
    return dict(identity=rays.identity, measured_ns=now_ns, through_ns=through_ns,
        supplied_configuration=dict(reference='current_body', position_m=offset.tolist(), rotation=local_R.tolist(),
            joints_rad=q.tolist(), point_error_m=float(point_error_m), initial_partition_point_error_m=global_error,
            padding_m=.04), primitives=rows, observation_bindings=bindings,
        all_primitives_conditionally_nonfloor_clear=all(r['conditional_nonfloor_clearance'] for r in rows),
        all_primitives_observed_separated=all(r['conditional_observed_separation'] for r in rows),
        static_scene_assumed=True, supplied_error_scales_validated=False,
        common_ground_surface_identity_established=False, configuration_is_execution_prediction=False,
        continuous_swept_volume_established=False, ground_support_permission=False,
        future_gait_qualified=False, navigation_action_permitted=False)
