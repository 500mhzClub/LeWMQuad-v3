"""Evaluator-only startup checks; never imported by a sensor/policy producer.

Checks the supplied velocity ball and finite non-floor-clear prism against an
explicit complete static-box inventory. Native contact witnesses are separate
from prospective foot-contact permissions and hardware support measurements.
"""
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.physical_execution_development import rotation_xyzw
from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.setup_region_prior_development import SetupRegionPrior
from lewm.setup_velocity_prior_development import SetupVelocityPrior


def vector(value, n, label):
    v = np.asarray(value, float)
    if v.shape != (n,) or not np.isfinite(v).all(): raise ValueError('finite ' + label + ' required')
    return v


def rotation(value):
    r = np.asarray(value, float)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-10, rtol=0)
            or abs(np.linalg.det(r) - 1) > 1e-10):
        raise ValueError('proper finite evaluator rotation required')
    return r


def box_separation_lower_bound(centre_a, axes_a, half_a, centre_b, axes_b, half_b):
    """15-axis OBB separating-axis test, including edge cross products.

Positive output certifies separation under the supplied exact box model.
Touching/near-degenerate arithmetic remains non-positive. A positive projected
interval gap is a lower bound on Euclidean distance, not the exact distance.
"""
    a, b = vector(centre_a, 3, 'box centre'), vector(centre_b, 3, 'box centre')
    ra, rb = rotation(axes_a), rotation(axes_b)
    ha, hb = vector(half_a, 3, 'box half extents'), vector(half_b, 3, 'box half extents')
    if np.any(ha <= 0) or np.any(hb <= 0): raise ValueError('positive box extents required')
    directions = [*ra.T, *rb.T, *(np.cross(u, v) for u in ra.T for v in rb.T)]
    gaps = []
    for direction in directions:
        norm = np.linalg.norm(direction)
        if norm <= 1e-10: continue
        n = direction / norm
        with np.errstate(over='ignore', invalid='ignore'):
            gap = abs(n @ (b - a)) - np.abs(n @ ra) @ ha - np.abs(n @ rb) @ hb
            scale = np.abs(a).sum() + np.abs(b).sum() + ha.sum() + hb.sum()
            allowance = 1e-10 + 256 * np.finfo(float).eps * scale
        if not np.isfinite([gap, allowance]).all(): raise ValueError('representable box separation required')
        gaps.append(float(gap - allowance))
    return max(gaps)


def check_setup_snapshot(velocity_prior, region_prior, *, identity, measured_ns,
                         position_world_m, rotation_world_from_initial_body,
                         velocity_world_m_s, native_static_boxes, expected_nonfloor_names,
                         geometry, joint_position):
    """Only the setup epoch may validate the setup priors; no ongoing pose output."""
    if (not isinstance(velocity_prior, SetupVelocityPrior) or not isinstance(region_prior, SetupRegionPrior)
            or not isinstance(identity, tuple) or any(type(x) is not int for x in identity)
            or identity != velocity_prior.identity or identity != region_prior.identity
            or type(measured_ns) is not int or measured_ns != velocity_prior.anchor_ns
            or measured_ns != region_prior.anchor_ns
            or velocity_prior.setup_evidence_sha256 != region_prior.setup_evidence_sha256
            or not isinstance(geometry, ArticulatedCollisionGeometry)):
        raise ValueError('matching exact setup identity/epoch/provenance and verified geometry required')
    position = vector(position_world_m, 3, 'setup position')
    r = rotation(rotation_world_from_initial_body)
    velocity = vector(velocity_world_m_s, 3, 'setup velocity')
    names = tuple(expected_nonfloor_names)
    if (not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names)
            or not isinstance(native_static_boxes, list) or len(native_static_boxes) != len(names)
            or {x['native_name'] for x in native_static_boxes} != set(names)):
        raise ValueError('complete unique independently enumerated native static-box roster required')
    low, high = np.asarray(region_prior.lower_initial_body_m), np.asarray(region_prior.upper_initial_body_m)
    centre, half = position + r @ ((low + high) / 2), (high - low) / 2
    margins = {}
    for box in native_static_boxes:
        if (box['fixed'] is not True or box['collision_enabled'] is not True
                or type(box['native_collision_boxes']) is not int or box['native_collision_boxes'] != 1):
            raise ValueError('exactly one fixed collision box per enumerated object required')
        size = vector(box['native_box_size'], 7, 'native BOX data')
        if np.any(size[:3] <= 0) or not np.array_equal(size[3:], np.zeros(4)):
            raise ValueError('positive native box size and exact reserved zeros required')
        qw, qx, qy, qz = vector(box['native_quaternion_wxyz'], 4, 'native orientation')
        axes = rotation_xyzw([qx, qy, qz, qw])
        margins[box['native_name']] = box_separation_lower_bound(centre, r, half,
            box['native_position'], axes, size[:3] / 2)
    error = float(np.linalg.norm(r.T @ velocity - velocity_prior.mean_initial_body_m_s))
    if not np.isfinite(error): raise ValueError('representable initial velocity error required')
    shapes = geometry.supports(joint_position, np.eye(3))['shapes']
    inside = region_prior.query([s['lower'] for s in shapes], [s['upper'] for s in shapes],
        [.04] * len(shapes), identity=identity, now_ns=measured_ns, observed_conflict=[False] * len(shapes))
    body_inside = bool(inside['conditional_setup_non_floor_clearance'].all())
    velocity_ok = error <= velocity_prior.radius_m_s
    clear = all(gap > 0 for gap in margins.values())
    return dict(schema='setup_snapshot_evaluator_development.v1', measured_ns=measured_ns,
        identity=identity, setup_evidence_sha256=velocity_prior.setup_evidence_sha256,
        velocity_ball_contains_reference=velocity_ok, initial_velocity_error_m_s=error,
        native_nonfloor_region_clear=clear, native_box_separation_lower_bounds_m=margins,
        initial_27_primitives_with_4cm_padding_inside_region=body_inside,
        velocity_and_nonfloor_setup_checks_pass=bool(velocity_ok and clear and body_inside),
        evidence_role='EVALUATOR_ONLY_SETUP_SNAPSHOT', native_static_inventory_assumed_complete=True,
        support_established=False, contact_model_validated=False, navigation_qualified=False,
        runtime_sensor_measurement=False, setup_claim_valid_beyond_declared_region=False)


def initial_ground_support_witness(contact_rows, *, expected_support_groups, ground_link_ids,
                                   geometry, joint_position, position_world_m,
                                   rotation_world_from_body):
    """Instantaneous native support-group force and nominal non-foot gap check.

    Merged calf-group contacts are not individually identified foot collision geoms.
    This witness never sets contact_model_validated or future gait permission.
    The caller must separately verify the native world-z-up plane at z=0.
"""
    groups = tuple(expected_support_groups)
    expected = {'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'}
    ground = tuple(ground_link_ids)
    if (set(groups) != expected or len(groups) != 4 or not ground
            or len(set(ground)) != len(ground) or any(type(i) is not int or i < 0 for i in ground)
            or not isinstance(geometry, ArticulatedCollisionGeometry)):
        raise ValueError('exact native support groups, ground identity and calibrated URDF required')
    position = vector(position_world_m, 3, 'body position'); r = rotation(rotation_world_from_body)
    normal_forces = {g: 0. for g in groups}; other = []
    for row in contact_rows:
        if row['force_status'] != 'measured': raise ValueError('native measured support forces required')
        force = vector(row['force_on_robot_world_n'], 3, 'contact force')
        name = row['robot_link_name']; environment = row['environment_link_id']
        magnitude = float(np.linalg.norm(force))
        if (type(environment) is not int or environment < 0 or type(row['disallowed']) is not bool
                or not isinstance(name, str) or not name or not np.isfinite(magnitude)):
            raise ValueError('finite force magnitude and explicit contact identity required')
        if name in expected and environment in ground and not row['disallowed']:
            normal_forces[name] += float(force[2])
        elif magnitude > 0: other.append(dict(robot_link_name=name, environment_link_id=environment))
    if not np.isfinite(list(normal_forces.values())).all(): raise ValueError('representable support force sums required')
    shapes = geometry.supports(joint_position, r)['shapes']
    gaps = {s['shape_id']: float(s['lower'][2] + position[2]) for s in shapes if s['shape_id'] not in FOOT_SHAPES}
    all_groups = all(f > 0 for f in normal_forces.values())
    nonfeet_above = all(g > 0 for g in gaps.values())
    return dict(positive_normal_force_all_four_native_support_groups=all_groups,
        normal_force_by_native_support_group_n=normal_forces,
        other_loaded_contact_rows=other, all_nominal_nonfoot_primitives_above_z0_plane=nonfeet_above,
        minimum_nominal_nonfoot_gap_m=min(gaps.values()),
        initial_native_support_witness_present=bool(all_groups and nonfeet_above and not other),
        exact_foot_collision_geom_identity_verified=False, instantaneous_only=True,
        contact_model_validated=False, contact_permitted=False, navigation_qualified=False)
