"""Evaluator-only opaque-surface geometry. Never repair policy pixels using this."""
import math
import numpy as np
from lewm.causal_depth_observation_development import FOCAL

def expected_optical_depth(boxes, world_from_optical, *, stride=8, floor_z_m=0.):
    """Physical first opaque surface; deliberately independent of renderer clipping."""
    if type(stride) is not int or not 1 <= stride <= 480:
        raise ValueError('positive bounded integer pixel stride required')
    transform = np.asarray(world_from_optical, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.array_equal(transform[3], [0., 0., 0., 1.])
            or not np.allclose(transform[:3,:3].T @ transform[:3,:3], np.eye(3), atol=1e-7, rtol=0)
            or abs(np.linalg.det(transform[:3,:3])-1) > 1e-7):
        raise ValueError('proper finite optical transform required')
    if not np.isfinite(floor_z_m) or transform[2,3] <= floor_z_m:
        raise ValueError('camera must be strictly above opaque floor')
    rows, columns = np.arange(stride//2, 480, stride), np.arange(stride//2, 640, stride)
    u, v = np.meshgrid(columns+.5, rows+.5)
    rays = np.stack(((u-320)/FOCAL, (v-240)/FOCAL, np.ones_like(u)), axis=-1)
    origin = transform[:3, 3]
    direction = rays@transform[:3, :3].T
    distances, margins, names = [], [], []
    ground = np.divide(floor_z_m-origin[2], direction[..., 2], out=np.full(u.shape, np.inf), where=direction[..., 2] < -1e-12)
    distances.append(np.where(ground > 0., ground, np.inf))
    margins.append(np.full(u.shape, np.inf)); names.append('ground_plane')
    for box in boxes:
        angle = box['yaw_rad']; c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        p = (origin-np.asarray(box['centre_xyz']))@rotation
        d = direction@rotation
        half = np.asarray(box['size_xyz'])/2
        if (half.shape != (3,) or p.shape != (3,) or not np.isfinite(half).all()
                or not np.isfinite(p).all() or not np.isfinite(rotation).all() or np.any(half <= 0)):
            raise ValueError('finite positive upright boxes required')
        if np.all(np.abs(p) <= half):
            raise ValueError('camera lies in/on opaque solid; cannot certify visibility')
        moving = np.abs(d) > 1e-12
        a = np.divide(-half-p, d, out=np.full_like(d, -np.inf), where=moving)
        b = np.divide(half-p, d, out=np.full_like(d, np.inf), where=moving)
        entry = np.max(np.minimum(a, b), axis=-1)
        leave = np.min(np.maximum(a, b), axis=-1)
        outside_parallel = np.any((~moving) & (np.abs(p) > half), axis=-1)
        visible = ~outside_parallel & (entry > 0.) & (entry <= leave)
        hit = p+np.where(visible, entry, 0.)[..., None]*d
        # Exclude points within2cm of a box edge, fixed before actual rendering.
        face_distances = np.sort(np.abs(half-np.abs(hit)), axis=-1)
        margins.append(np.where(visible, face_distances[..., 1], 0.))
        distances.append(np.where(visible, entry, np.inf)); names.append(box['wall_id'])
    distances = np.stack(distances)
    winner = np.argmin(distances, axis=0)
    expected = np.min(distances, axis=0)
    margin = np.take_along_axis(np.stack(margins), winner[None], axis=0)[0]
    return {'rows': rows, 'columns': columns, 'expected_depth_m': expected, 'object_index': winner,
            'object_names': names, 'surface_interior': margin > .02}


def evaluate_visibility(native_depth, boxes, transform, *, render_near_m, stride=8):
    """Sampled physical first-surface check, including below-near occluders.

    Report failures instead of discarding clipped rays or treating them as free.
    This is not a full-image visibility proof or a hardware calibration.
    """
    if not np.isfinite(render_near_m) or render_near_m <= 0:
        raise ValueError('finite positive native render near plane required')
    native = np.asarray(native_depth)
    if native.shape != (480, 640) or native.dtype != np.float32:
        raise ValueError('native float32 optical metres required')
    ref = expected_optical_depth(boxes, transform, stride=stride)
    expected = ref['expected_depth_m']
    measured = native[np.ix_(ref['rows'], ref['columns'])]
    finite = np.isfinite(expected)
    clipped = finite & (expected <= render_near_m)
    use = finite & ref['surface_interior'] & (expected < 4.98)
    errors = np.abs(measured[use]-expected[use])
    # Compare near-surface metric error too, independent of public validity.
    within = bool(len(errors) >= 1000 and np.isfinite(errors).all() and errors.max() <= .001)
    below_public = finite & (expected < .199)
    falsely_valid = below_public & np.isfinite(measured) & (measured >= .2) & (measured <= 5.)
    passed = within and not clipped.any() and not falsely_valid.any()
    return dict(passes_sampled_physical_visibility=bool(passed), sampled_rays=int(expected.size),
        compared_interior_rays=int(use.sum()), clipped_opaque_rays=int(clipped.sum()),
        false_public_valid_near_rays=int(falsely_valid.sum()), within1mm=within,
        maximum_error_m=float(errors.max()) if len(errors) and np.isfinite(errors).all() else None,
        below_public_range_rays=int(below_public.sum()), render_near_m=float(render_near_m),
        stride=stride, scope='sampled opaque geometry only; no self-occlusion or hardware qualification')

