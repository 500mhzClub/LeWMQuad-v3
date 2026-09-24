"""Exact retained-witness ordering and pixel/depth boundary comparisons."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest
from lewm.batched_retained_floor_patch_development import (
    BatchedRetainedFloorPatches, projected_rectangles, FRAME_BATCH)
from lewm.retained_floor_patch_development import RetainedFloorPatches, T, FOCAL
from lewm.tests.test_retained_floor_patch_development import memory
from lewm.tests.test_observed_geometry_refinement_development import floor


def prefix(bad):
    result = np.zeros((480, 640), np.int32)
    result[1:, 1:] = bad.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    result.flags.writeable = False
    return result


GOOD = prefix(np.zeros((479, 639), dtype=bool))
BAD = prefix(np.ones((479, 639), dtype=bool))


def frame(i, pixels=GOOD, *, yaw=0., p=(0., 0., 0.)):
    c, s = np.cos(yaw), np.sin(yaw)
    return dict(R=np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]), p=np.asarray(p, float),
        floor_height=-.32, prefix=pixels, witness=dict(frame=i, measured_ns=i*100_000_000, evidence=['original']))


def paired(frames):
    old = RetainedFloorPatches(); new = BatchedRetainedFloorPatches()
    old.frames = frames; new.frames = frames
    return old, new


def scalar_projection(f, xy, radius):
    corners = xy[:, None, :]+radius*np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    world = np.concatenate((corners, np.full((*corners.shape[:-1], 1), f['floor_height'])), axis=-1)
    points = (world-f['p'])@f['R']; camera = (points-T[:3, 3])@T[:3, :3]; z = camera[..., 2]
    uv = camera[..., :2]/np.maximum(z[..., None], 1e-12)*FOCAL+[319.5, 239.5]
    lo = uv.min(1); hi = uv.max(1)
    margin = 1e-9+64*np.finfo(float).eps*np.maximum(np.abs(lo), np.abs(hi))
    lo -= margin; hi += margin
    visible = ((z >= .2)&(z <= 5.)).all(1)&(lo >= 0).all(1)&(hi < [639, 479]).all(1)
    return np.floor(np.clip(lo, -1, 640)).astype(int), np.floor(np.clip(hi, -1, 640)).astype(int), visible


def test_original_admission_and_storage_are_inherited():
    assert BatchedRetainedFloorPatches.__init__ is RetainedFloorPatches.__init__
    assert BatchedRetainedFloorPatches.append is RetainedFloorPatches.append
    assert hashlib.sha256(Path('lewm/retained_floor_patch_development.py').read_bytes()).hexdigest() == (
        'd8d774fa3401b40ad60ae5a9d0d085616881b8d5656d26590345e54874693a58')
    assert FRAME_BATCH == 32


@pytest.mark.parametrize('count', [0, 1, 31, 32, 33, 65, 130])
@pytest.mark.parametrize('queries', [0, 1, 4, 128])
def test_history_and_query_batch_edges_preserve_complete_outputs(count, queries):
    frames = [frame(i, GOOD if i in [31, 64, 129] else BAD) for i in range(count)]
    old, new = paired(frames)
    xy = np.column_stack((np.linspace(.1, 4.8, queries), np.linspace(-.2, .2, queries)))
    assert new.coverage(xy) == old.coverage(xy)


@pytest.mark.parametrize('radius', [.001, .022, .1])
def test_rotated_translated_projection_rectangles_match_exactly(radius):
    rng = np.random.default_rng(2026091007)
    frames = [frame(i, yaw=rng.uniform(-np.pi, np.pi), p=rng.uniform(-2., 2., 3)) for i in range(32)]
    xy = rng.uniform(-4.9, 4.9, (128, 2))
    a, b, visible = projected_rectangles(frames, xy, radius)
    for i, f in enumerate(frames):
        expected = scalar_projection(f, xy, radius)
        for actual, wanted in zip((a[i], b[i], visible[i]), expected, strict=True):
            np.testing.assert_array_equal(actual, wanted)
    old, new = paired(frames)
    assert new.coverage(xy, radius) == old.coverage(xy, radius)


def test_mixed_earliest_witnesses_and_returned_evidence_are_independent():
    left = np.zeros((479, 639), bool); left[:, 320:] = True
    right = np.zeros((479, 639), bool); right[:, :320] = True
    frames = [frame(0, prefix(left)), frame(1, BAD), frame(2, prefix(right)), frame(3, GOOD)]
    old, new = paired(frames)
    xy = [[1., .1], [1., -.1], [1., 0.], [0., 0.]]
    expected = old.coverage(xy); actual = new.coverage(xy)
    assert actual == expected
    witnesses = [r['coverage_witness']['witness']['frame'] for r in actual if r['complete_nominal_foot_patch']]
    assert set(witnesses) == {0, 2, 3}
    for result in actual:
        if result['coverage_witness']:
            result['coverage_witness']['witness']['evidence'].append('changed')
    assert all(f['witness']['evidence'] == ['original'] for f in frames)
    assert all(not f['prefix'].flags.writeable for f in frames)


def test_actual_floor_pixels_keep_invalid_interior_and_unknown_regions():
    depth, valid = floor(); admitted = memory(depth, valid)
    _, new = paired(admitted.frames)
    assert new.coverage([[1., 0.], [0., 0.], [1.2, .1]]) == admitted.coverage([[1., 0.], [0., 0.], [1.2, .1]])
    box = admitted.coverage([[1., 0.]])[0]['coverage_witness']['pixel_rectangle']
    x = (box[0][0]+box[1][0])//2; y = (box[0][1]+box[1][1])//2
    depth[y, x] = 0.; valid[y, x] = False
    damaged = memory(depth, valid); _, new = paired(damaged.frames)
    assert new.coverage([[1., 0.]]) == damaged.coverage([[1., 0.]])
    assert not new.coverage([[1., 0.]])[0]['complete_nominal_foot_patch']


def test_later_pixel_images_are_never_queried_after_all_witnesses_found():
    class RejectRead:
        def __getitem__(self, key): raise AssertionError('later image queried')
    old, new = paired([frame(0), frame(1, RejectRead())])
    assert new.coverage([[1., 0.]]) == old.coverage([[1., 0.]])


def test_general_three_axis_rotations_and_depth_pixel_boundaries():
    rng = np.random.default_rng(2026091011)
    frames = [frame(0), frame(1, p=(-1., 0., 0.))]
    for i in range(2, 32):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0: q[:, -1] *= -1
        f = frame(i, p=rng.uniform(-2., 2., 3)); f['R'] = q; frames.append(f)
    x = np.asarray([T[0, 3]+.2-.022, T[0, 3]+.2+.022, T[0, 3]+5.-.022-1., 1.])
    x = np.concatenate((x, np.nextafter(x, -np.inf), np.nextafter(x, np.inf)))
    xy = np.column_stack((x, np.linspace(-.2, .2, len(x))))
    actual = projected_rectangles(frames, xy, .022)
    for i, f in enumerate(frames):
        expected = scalar_projection(f, xy, .022)
        for values, wanted in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(values[i], wanted)
    old, new = paired(frames)
    assert old.coverage(xy) == new.coverage(xy)


@pytest.mark.parametrize('fault', ['overflow', 'missing_metadata'])
def test_unused_later_projection_error_cannot_defeat_earliest_witness(fault):
    later = frame(1, p=(1e308, 1e308, 1e308)) if fault == 'overflow' else frame(1)
    if fault == 'missing_metadata': later.pop('R')
    old, new = paired([frame(0), later])
    with np.errstate(all='raise'):
        assert new.coverage([[1., 0.]]) == old.coverage([[1., 0.]])


def test_required_projection_error_is_still_raised():
    old, new = paired([frame(0, BAD), frame(1, p=(1e308, 1e308, 1e308))])
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError): old.coverage([[1., 0.]])
        with pytest.raises(FloatingPointError): new.coverage([[1., 0.]])


def test_full_admitted_history_limit_keeps_last_single_frame_witness():
    frames = [frame(i, GOOD if i == 4095 else BAD) for i in range(4096)]
    old, new = paired(frames)
    actual = new.coverage([[1., 0.]])
    assert actual == old.coverage([[1., 0.]])
    assert actual[0]['coverage_witness']['witness']['frame'] == 4095


@pytest.mark.parametrize('xy,radius', [([1., 0.], .022), ([[5., 0.]], .022),
    ([[np.nan, 0.]], .022), ([[1., 0.]], 0.), ([[1., 0.]], .101), ([[1., 0.]], np.inf),
    (np.zeros((129, 2)), .022)])
def test_original_query_rejections_are_unchanged(xy, radius):
    old, new = paired([frame(0)])
    with pytest.raises(ValueError) as baseline: old.coverage(xy, radius)
    with pytest.raises(ValueError) as candidate: new.coverage(xy, radius)
    assert str(baseline.value) == str(candidate.value)
