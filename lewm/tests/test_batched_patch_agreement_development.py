import cv2
import numpy as np

from lewm import batched_patch_agreement_development as batch
from lewm.direct_corner_flow_association_development import patch_agrees, tracked_points
from lewm.tests.test_chained_corner_flow_association_development import scene


def test_mixed_patches_boundaries_and_blank_images_match_scalar_decisions():
    rng = np.random.default_rng(2026091301)
    left = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    right = left.copy(); right[:, 320:] = 128
    p = rng.uniform([0, 0], [639, 479], (600, 2)).astype(np.float32)
    p[:6] = [[5, 5], [634, 474], [4.99, 5], [634.01, 10], [30, 474.01], [320, 240]]
    for image, q in ((left, p), (right, p), (right, p+[.25, -.2])):
        expected = np.asarray([patch_agrees(left, image, a, b) for a,b in zip(p, q, strict=True)])
        np.testing.assert_array_equal(batch.patches_agree(left, image, p, q), expected)
    assert batch.patches_agree(left, right, np.empty((0, 2)), np.empty((0, 2))).shape == (0,)


def test_exact_decision_boundary_uses_original_scalar_test(monkeypatch):
    rng = np.random.default_rng(13)
    image = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    point = np.asarray([[100.25, 200.5]], np.float32)
    patch = cv2.getRectSubPix(image, (11, 11), tuple(map(float, point[0]))).astype(float)
    patch -= patch.mean()
    monkeypatch.setitem(batch.FLOW_RULES, 'minimum_patch_std', float(patch.std()))
    calls = []
    def scalar(*args): calls.append(1); return patch_agrees(*args)
    monkeypatch.setattr(batch, 'patch_agrees', scalar)
    expected = patch_agrees(image, image, point[0], point[0])
    assert batch.patches_agree(image, image, point, point).tolist() == [expected]
    assert calls == [1]


def test_entire_moving_image_association_matches_original():
    rows = scene(2)
    expected, receipt = tracked_points(rows[0][2], rows[1][2])
    actual, actual_receipt = batch.tracked_points(rows[0][2], rows[1][2])
    for x, y in zip(expected, actual, strict=True): np.testing.assert_array_equal(x, y)
    assert actual_receipt == receipt
