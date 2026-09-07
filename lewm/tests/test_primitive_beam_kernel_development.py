import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.primitive_beam_kernel_development import warm_primitive_beam_kernel
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.primitive_obstacle_memory_development import (
    cached_plane_family_mask, plane_family_cells, non_floor_box_evidence, PrimitiveObstacleMemory)
from lewm.tests.test_primitive_obstacle_memory_development import (
    LOW, HIGH, PLANE, rectangle, scene, observed_stream, ArticulatedCollisionGeometry, URDF, FOCAL)


@pytest.fixture(scope='module', autouse=True)
def warm():
    warm_primitive_beam_kernel()


def compare(frame, low=LOW, high=HIGH, *, plane=PLANE, error=.001, en=.002, eb=.001):
    kwargs = dict(plane=plane, normal_error=en, plane_offset_error=eb, range_error_m=error)
    a = non_floor_box_evidence(frame, low, high, backend='reference', **kwargs)
    b = non_floor_box_evidence(frame, low, high, backend='compiled', **kwargs)
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], np.ndarray): np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        else: assert a[key] == b[key]
    return a


@pytest.mark.parametrize('case', ['floor', 'wall', 'overhang', 'missing', 'occluder', 'far',
                                  'last_row', 'last_column', 'border_floor', 'early_out', 'empty', 'camera_crossing'])
def test_compiled_matches_every_output_on_geometric_and_boundary_cases(case):
    d, valid = scene(); low, high = LOW, HIGH
    x0, y0, x1, y1 = rectangle()
    if case == 'wall': d[y0-2:y1+3, x0-2:x1+3] = 1.17
    elif case == 'overhang':
        rows = np.arange(y0-2, y1+3)
        d[y0-2:y1+3, x0-2:x1+3] = ((.043 + .27) * FOCAL / (rows - 239.5))[:, None]
    elif case == 'missing': d[y0+1, x0+1] = 0.; valid[y0+1, x0+1] = False
    elif case in ('occluder', 'far', 'early_out', 'last_row', 'last_column'):
        d[:] = .7 if case == 'occluder' else 4.; valid[:] = True
    if case in ('last_column', 'early_out'):
        low, high = [[1.4, -2., -.3]], [[1.6, .1, -.24]]
        if case == 'last_column': d[350, 639] = 1.17
    elif case == 'last_row':
        d[479, 320] = 1.17; low, high = [[1.4, -.1, -2.]], [[1.6, .1, -.24]]
    elif case == 'border_floor': low, high = [[.7, -.1, -.5]], [[1.2, .1, -.27]]
    elif case == 'empty': low = high = np.empty((0, 3))
    elif case == 'camera_crossing': low, high = [[.2, -.2, -.2]], [[.9, .2, .2]]
    compare(PreparedFloorFrame(d, valid, [0, 0, 1]), low, high)


@pytest.mark.parametrize('seed', [671, 1283, 3117])
def test_random_boxes_and_missing_depth_match_reference(seed):
    rng = np.random.default_rng(seed)
    d, valid = scene()
    rr, cc = rng.integers(0, 480, 100), rng.integers(0, 640, 100)
    d[rr, cc] = 0.; valid[rr, cc] = False
    centres = rng.uniform([.1, -.9, -.6], [3., .9, .3], (20, 3))
    widths = rng.uniform(.001, .2, (20, 3))
    frame = PreparedFloorFrame(d, valid, [0, 0, 1])
    compare(frame, centres-widths, centres+widths, plane=None)
    compare(frame, centres-widths, centres+widths)


def test_full_cached_plane_mask_matches_direct_cells_and_is_bounded_immutable():
    d, valid = scene(); frame = PreparedFloorFrame(d, valid, [0, 0, 1])
    rng = np.random.default_rng(3159)
    cells = np.column_stack((rng.integers(0, 480, 2000), rng.integers(0, 640, 2000)))
    cells = np.vstack((cells, [[479, 0], [479, 639], [0, 639]]))
    for en, eb in ((.002, .001), (0., 0.), (.00001, .000001)):
        cached = cached_plane_family_mask(frame, *PLANE, normal_error=en, plane_offset_error=eb)
        direct = plane_family_cells(frame, cells, *PLANE, normal_error=en, plane_offset_error=eb)
        np.testing.assert_array_equal(cached[cells[:, 0], cells[:, 1]], direct)
        assert cached_plane_family_mask(frame, *PLANE, normal_error=en, plane_offset_error=eb) is cached
        with pytest.raises(ValueError): cached[300, 300] = True
    assert len(frame._plane_family_cache) == 2


@pytest.mark.parametrize('case', ['floor', 'missing', 'step', 'zero_family'])
def test_cached_floor_prefix_matches_direct_whole_primitive_triangles(case):
    model = ArticulatedCollisionGeometry(URDF); q = np.repeat([0., .8, -1.5], 4)
    errors = {r['shape_id']: .001 for r in model.supports(q, np.eye(3))['shapes']}
    d, valid = scene()
    if case == 'missing': d[360, 290] = 0.; valid[360, 290] = False
    if case == 'step': d[360, 290] += .03
    frame = PreparedFloorFrame(d, valid, [0, 0, 1])
    kwargs = dict(rotation_observation_from_body=np.eye(3), translation_observation_from_body=[1.5, 0, 0],
                  normal_error=0. if case == 'zero_family' else .002, up_error=.001,
                  plane_offset_error=0. if case == 'zero_family' else .001, point_error_by_shape=errors)
    a = frame.query(model, q, [354, 319], floor_backend='reference', **kwargs)
    b = frame.query(model, q, [354, 319], floor_backend='cached', **kwargs)
    for key in a:
        if isinstance(a[key], np.ndarray): np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        else: assert a[key] == b[key]
    assert len(frame._plane_family_cache) == 1
    mask, prefix = next(iter(frame._plane_family_cache.values()))
    with pytest.raises(ValueError): prefix[1, 1] = 0
    assert not mask.flags.writeable


def test_range_interval_arithmetic_is_float64_and_scalar_type_independent():
    d = np.full((480, 640), 1.23, np.float32); valid = np.ones_like(d, bool)
    er = .001
    true_lower = float(d[0, 0]) - er
    rounded_lower = float(np.float32(d[0, 0] - np.float32(er)))
    assert true_lower > rounded_lower
    optical_far = (true_lower + rounded_lower) / 2
    high = [[.326 + optical_far, .04, -.26]]
    frame = PreparedFloorFrame(d, valid, [0, 0, 1])
    for value in (er, np.float64(er)):
        result = compare(frame, LOW, high, plane=None, error=value)
        assert result['non_floor_clearance'].all()


def test_zero_error_family_and_range_boundary_match():
    d, valid = scene(); frame = PreparedFloorFrame(d, valid, [0, 0, 1])
    for er in (0., np.nextafter(0., 1.), .001): compare(frame, error=er, en=0., eb=0.)


def test_consumer_backends_agree_including_view_work_counts():
    model = PrimitiveObstacleMemory(ArticulatedCollisionGeometry(URDF), normal_error=.002,
                                    up_error=.001, plane_offset_error=.001, range_error_m=.001,
                                    beam_backend='compiled')
    for p, d, r, now in observed_stream(): model.observe(p, d, r, now_ns=now)
    a = model.query_current_primitives(now_ns=now, beam_backend='reference')
    b = model.query_current_primitives(now_ns=now, beam_backend='compiled')
    for key in a:
        if isinstance(a[key], np.ndarray): np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        else: assert a[key] == b[key]
    with pytest.raises(SensorContractError): model.query_current_primitives(now_ns=now, beam_backend='unknown')
