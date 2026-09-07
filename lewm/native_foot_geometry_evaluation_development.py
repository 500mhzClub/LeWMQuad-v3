"""Evaluator-only native foot-geometry identity and ground-contact diagnostics."""
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.physical_execution_development import rotation_xyzw
from lewm.primitive_floor_relation_development import FOOT_SHAPES


def match_native_foot_geometries(rows, geometry, joint_position, base_pose_world):
    """Match four native spheres by group, radius, and actual world centre.

Do not infer geometry IDs from their ordering or a merged support-link label.
The sphere native format is seven slots: radius followed by six reserved zeros.
This is an identity check at one pose, not contact/compliance qualification.
"""
    if not isinstance(geometry, ArticulatedCollisionGeometry) or not isinstance(rows, list) or len(rows) != 27:
        raise ValueError('verified 27-primitive robot and complete native geometry roster required')
    ids = [r['geom_id'] for r in rows]
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('unique explicit native geometry IDs required')
    pose = np.asarray(base_pose_world, float)
    if pose.shape != (7,) or not np.isfinite(pose).all(): raise ValueError('actual finite root pose required')
    R = rotation_xyzw(pose[3:]); matches = {}; maximum = 0.
    shapes = {s['shape_id']: s for s in geometry.supports(joint_position, np.eye(3))['shapes']}
    for row in rows:
        data, centre = np.asarray(row['data'], float), np.asarray(row['position_world_m'], float)
        if (data.shape != (7,) or centre.shape != (3,) or not np.isfinite(data).all()
                or not np.isfinite(centre).all() or type(row['link_id']) is not int or row['link_id'] < 0
                or not isinstance(row['link_name'], str) or not row['link_name']
                or row['geom_type'] not in ('SPHERE', 'ELLIPSOID', 'CYLINDER', 'CAPSULE', 'BOX', 'MESH')):
            raise ValueError('finite typed native robot geometry required')
    for sid in sorted(FOOT_SHAPES):
        group = sid[:2] + '_calf'
        candidates = [r for r in rows if r['link_name'] == group and r['geom_type'] == 'SPHERE']
        if len(candidates) != 1: raise ValueError('exactly one native foot sphere in each expected calf group required')
        row = candidates[0]; data = np.asarray(row['data'], float)
        if abs(data[0] - .022) > 1e-8 or not np.array_equal(data[1:], np.zeros(6)):
            raise ValueError('calibrated 22-mm foot radius and exact native reserved slots required')
        expected = pose[:3] + R @ np.asarray(shapes[sid]['center_body_m'])
        error = float(np.max(np.abs(np.asarray(row['position_world_m']) - expected)))
        if error > 2e-6: raise ValueError('native foot centre differs from actual-pose URDF reference')
        maximum = max(maximum, error); matches[row['geom_id']] = sid
    return dict(native_foot_geom_to_shape=matches, maximum_foot_centre_coordinate_error_m=maximum,
        exact_four_foot_geometry_identities_verified=True, native_geometry_count=len(rows),
        contact_model_validated=False, contact_permitted=False, navigation_qualified=False)


def nonfoot_ground_contact_indices(packet, *, robot_geom_ids, foot_geom_ids, ground_geom_ids):
    """Additional native check; retain the existing non-ground/body contact guard.

This distinguishes a foot sphere from another geom on the same calf link.
Zero-load and invalid rows are not positive contact evidence. Self-contact and
non-ground contacts are outside THIS additional check, not globally permitted.
"""
    sets = []
    for ids in (robot_geom_ids, foot_geom_ids, ground_geom_ids):
        values = tuple(ids)
        if not values or any(type(i) is not int or i < 0 for i in values) or len(set(values)) != len(values):
            raise ValueError('explicit unique nonnegative geometry identity sets required')
        sets.append(set(values))
    robot, feet, ground = sets
    if not feet <= robot or robot & ground: raise ValueError('disjoint actual robot/ground and contained feet required')
    a, b, valid = [np.asarray(packet[k]) for k in ('geom_a', 'geom_b', 'valid_mask')]
    fa, fb = [np.asarray(packet[k], float) for k in ('force_a', 'force_b')]
    if (a.ndim != 1 or b.shape != a.shape or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
            or valid.shape != a.shape or valid.dtype != bool or fa.shape != a.shape + (3,) or fb.shape != fa.shape):
        raise ValueError('unbatched aligned native geom/contact-force fields required')
    violations = []
    for i in np.flatnonzero(valid):
        if a[i] < 0 or b[i] < 0 or not np.isfinite([fa[i], fb[i]]).all():
            raise ValueError('valid native contacts need finite forces and geometry IDs')
        if a[i] in robot and b[i] in ground: geom, force = int(a[i]), fa[i]
        elif b[i] in robot and a[i] in ground: geom, force = int(b[i]), fb[i]
        else: continue
        norm = float(np.linalg.norm(force))
        if not np.isfinite(norm): raise ValueError('representable native force norm required')
        if geom not in feet and norm > 0: violations.append(int(i))
    return violations
