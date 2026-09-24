import cv2
import numpy as np
from lewm.subpixel_corner_support_features_development import SubpixelCornerSupportFeatureFrame


def depth(valid=True):
    return dict(depth_m=np.full((480, 640), 2., np.float32), valid=np.full((480, 640), valid, bool))


def image():
    # A sampled, blurred step intersection has a known subpixel corner at .5.
    gray = np.full((480, 640), 20, np.uint8)
    gray[160:, 240:] = 230
    gray = cv2.GaussianBlur(gray, (5, 5), .7)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_refines_known_corner_without_mutating_inputs():
    rgb, d = image(), depth()
    old_rgb, old_depth, old_valid = rgb.copy(), d['depth_m'].copy(), d['valid'].copy()
    result = SubpixelCornerSupportFeatureFrame(rgb, d)
    points = np.asarray([k.pt for k in result.keypoints])
    assert len(points) == 1
    assert np.linalg.norm(points[0] - [239.5, 159.5]) < .2
    assert np.array_equal(rgb, old_rgb) and np.array_equal(d['depth_m'], old_depth)
    assert np.array_equal(d['valid'], old_valid)
    assert result.witness()['matching_thresholds_unchanged']
    assert sum(result.cell_counts) == len(result.keypoints)


def test_invalid_depth_cannot_supply_features():
    result = SubpixelCornerSupportFeatureFrame(image(), depth(False))
    assert not result.keypoints and result.descriptors is None
    assert result.witness()['selected_features'] == 0


def test_blank_image_does_not_invent_features():
    result = SubpixelCornerSupportFeatureFrame(np.zeros((480, 640, 3), np.uint8), depth())
    assert not result.keypoints and result.witness()['integer_selected_features'] == 0


def test_refined_surface_jump_is_rejected(monkeypatch):
    # Force a bounded candidate refinement onto an invalid neighbouring pixel.
    d = depth(); d['valid'][160, 240] = False
    original = cv2.cornerSubPix
    def refine(gray, points, *args):
        result = original(gray, points, *args)
        result[:] = [239.5, 159.5]
        return result
    monkeypatch.setattr(cv2, 'cornerSubPix', refine)
    result = SubpixelCornerSupportFeatureFrame(image(), d)
    assert not result.keypoints
