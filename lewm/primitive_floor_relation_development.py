"""One-sided physical floor separation, distinct from padding and contact.

Development geometry only. Supplied error sets and measured-plane coverage
remain conditional. A plane calculation cannot clear other obstacles, invent
unseen floor, establish foot contact, or certify a future gait.
"""
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError


FOOT_SHAPES = frozenset(f'{leg}_foot:0' for leg in ('FL', 'FR', 'RL', 'RR'))


def primitive_floor_gap_bounds(geometry, joints, anchor, normal, *, normal_error,
                               plane_offset_error, point_error_by_shape, padding_m=.04):
    """Bound each UNPADDED primitive's minimum signed floor-plane residual.

    Let g0=min_p n dot (p-a), evaluated by the exact primitive support function.
    For ||n'-n||<=en, |b'|<=eb and point errors ||dp||<=ep, every point residual
    changes by at most en*L + eb + (||n||+en)*ep, where L bounds max||p-a||.
    Consequently the MINIMUM residual lies in [g0-error,g0+error]. This is
    not the range between the bottom and top of the primitive. A negative
    upper bound means even the best allowed model has an intersecting point.
    True plane normals are assumed unit for distances and isotropic padding.

    Point-error bounds must include all intended kinematic and relative-pose
    uncertainty. They are neither estimated nor certified by this function.
    Padding is reported separately and never used as penetration tolerance.
    """
    if not isinstance(geometry, ArticulatedCollisionGeometry):
        raise SensorContractError('verified Go2 primitive geometry required')
    a, n = np.asarray(anchor, float), np.asarray(normal, float)
    errors = np.asarray([normal_error, plane_offset_error, padding_m], float)
    if (a.shape != (3,) or n.shape != (3,) or not np.isfinite(a).all()
            or not np.isfinite(n).all() or abs(np.linalg.norm(n) - 1) > 1e-12
            or errors.shape != (3,) or not np.isfinite(errors).all() or np.any(errors < 0)
            or normal_error >= 1):
        raise SensorContractError('finite anchor, unit normal and nonnegative bounded errors/padding required')
    along = geometry.supports(joints, n[None])['shapes']
    boxes = geometry.supports(joints, np.eye(3))['shapes']
    ids = {row['shape_id'] for row in along}
    if (not isinstance(point_error_by_shape, dict) or set(point_error_by_shape) != ids
            or any(np.asarray(v).shape != () or not np.isfinite(v) or v < 0
                   for v in point_error_by_shape.values())):
        raise SensorContractError('one explicit nonnegative point-error bound per physical primitive required')
    rows = []
    for shape, box in zip(along, boxes, strict=True):
        if shape['shape_id'] != box['shape_id']:
            raise SensorContractError('primitive support identity mismatch')
        with np.errstate(over='ignore', invalid='ignore'):
            extent = np.maximum(np.abs(np.asarray(box['lower']) - a), np.abs(np.asarray(box['upper']) - a))
            lever = float(np.linalg.norm(extent))
            nominal_gap = float(shape['lower'][0] - n @ a)
            error = (normal_error * lever + plane_offset_error
                     + (np.linalg.norm(n) + normal_error) * point_error_by_shape[shape['shape_id']])
        if not np.isfinite([lever, nominal_gap, error]).all():
            raise SensorContractError('representable primitive-plane bounds required')
        rounding = 1e-12 + 128 * np.finfo(float).eps * (abs(nominal_gap) + lever + error)
        lo, hi = float(nominal_gap - error - rounding), float(nominal_gap + error + rounding)
        rows.append({'shape_id': shape['shape_id'], 'link': shape['link'], 'kind': shape['kind'],
                     'nominal_minimum_gap_m': nominal_gap, 'minimum_gap_lower_m': lo,
                     'minimum_gap_upper_m': hi, 'physical_error_allowance_m': float(error),
                     'padded_minimum_gap_lower_m': lo - padding_m,
                     'padded_minimum_gap_upper_m': hi - padding_m,
                     'padding_m': float(padding_m),
                     'contact_candidate_geometry': shape['shape_id'] in FOOT_SHAPES,
                     'point_error_m': float(point_error_by_shape[shape['shape_id']])})
    return {'primitives': rows, 'plane_normal_body': n.tolist(), 'plane_anchor_body_m': a.tolist(),
            'normal_error': float(normal_error), 'plane_offset_error_m': float(plane_offset_error),
            'supplied_error_bounds_validated': False, 'floor_coverage_established': False,
            'contact_permitted': False, 'navigation_qualified': False, 'future_gait_qualified': False}


def assess_primitive_floor_relation(bounds, *, floor_coverage, non_floor_clearance):
    """Keep floor geometry, observed coverage and other obstacles independent.

    Both evidence dictionaries require exact primitive identities and actual
    bool values. Their producer must establish whole-primitive/envelope evidence
    under the SAME pose/plane errors; a point seed or a favourable single view
    is insufficient. This pure combiner does not validate that provenance.
    Foot intersections are only candidates requiring independent contact and
    model evidence, never cleared automatically. Mixed groups require EVERY
    member's clearance; being grouped with a foot confers no contact role.
    """
    rows = bounds['primitives']
    ids = {r['shape_id'] for r in rows}
    if len(ids) != len(rows) or not rows:
        raise SensorContractError('nonempty unique physical primitive identities required')
    for evidence in (floor_coverage, non_floor_clearance):
        if not isinstance(evidence, dict) or set(evidence) != ids or any(type(v) is not bool for v in evidence.values()):
            raise SensorContractError('exact primitive-keyed boolean floor and non-floor evidence required')
    assessed = []
    for row in rows:
        sid = row['shape_id']
        lo, hi = row['minimum_gap_lower_m'], row['minimum_gap_upper_m']
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise SensorContractError('finite ordered minimum-gap bounds required')
        # Never trust a mutable/generic ground-role flag for contact geometry.
        foot = sid in FOOT_SHAPES and row['link'] == sid.split(':')[0] and row['kind'] == 'sphere'
        separated = lo > 0
        certain_penetration = hi < 0
        possible_contact = lo <= 0 <= hi
        if not floor_coverage[sid]: status = 'UNKNOWN_FLOOR_COVERAGE'
        elif certain_penetration: status = 'PENETRATION_UNDER_EVERY_SUPPLIED_MODEL'
        elif not separated:
            status = 'FOOT_CONTACT_CANDIDATE_ONLY' if foot and possible_contact else 'NON_CONTACT_INTERSECTION_POSSIBLE'
        elif not non_floor_clearance[sid]: status = 'NON_FLOOR_CLEARANCE_UNRESOLVED'
        else: status = 'SEPARATED_UNDER_SUPPLIED_MODEL'
        assessed.append({'shape_id': sid, 'status': status,
                         'physical_floor_separated_under_supplied_model': separated,
                         'possible_foot_contact': bool(foot and possible_contact),
                         'penetration_under_every_supplied_model': certain_penetration,
                         'floor_coverage': floor_coverage[sid], 'non_floor_clearance': non_floor_clearance[sid],
                         'conditional_clearance': bool(separated and floor_coverage[sid] and non_floor_clearance[sid]),
                         'contact_permitted': False})
    return {'primitives': assessed, 'all_primitives_conditionally_clear': all(r['conditional_clearance'] for r in assessed),
            'contact_permitted': False, 'evidence_provenance_validated': False,
            'navigation_qualified': False, 'future_gait_qualified': False}
