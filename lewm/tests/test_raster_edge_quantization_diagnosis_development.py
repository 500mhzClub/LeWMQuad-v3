import numpy as np
import pytest
from lewm.raster_edge_quantization_diagnosis_development import edge_distance, projected_edge_witnesses
from lewm.tests.test_raster_footprint_visibility_development import fixture


def test_signed_distance_detects_small_projected_edge_side_change():
    point = [.5002, .5]; uv = np.array([[.5003, 0.], [.5003, 1.]])
    a = edge_distance(point, uv); b = edge_distance(point, np.rint(uv*256)/256)
    assert a['distance_pixels'] == pytest.approx(.0001)
    assert a['signed_line_distance_pixels']*b['signed_line_distance_pixels'] < 0
    assert a['closest_segment_fraction'] == pytest.approx(.5)
    assert edge_distance([2., 2.], [[0., 0.], [0., 0.]])['signed_line_distance_pixels'] is None


def test_projected_witnesses_are_geometric_and_never_certify_raster_behavior():
    boxes, T, _, _ = fixture()
    a = projected_edge_witnesses(boxes, T, [244, 324], 8)
    b = projected_edge_witnesses(boxes[::-1], T, [244, 324], 8)
    assert a == b and a[0]['original']['distance_pixels'] < 1.
    assert all(not x['native_rounding_rule_proven'] and not x['native_depth_explained'] for x in a)
    with pytest.raises(ValueError): projected_edge_witnesses(boxes, T, [480, 324], 8)
    with pytest.raises(ValueError): projected_edge_witnesses(boxes, T, [244, 324], 0)
