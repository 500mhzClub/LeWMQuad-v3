import pytest

from scripts.probe_go2_rgbd_fused_navigation_interface_development_v2 import assert_serialized_equal


def test_serialized_tuple_prior_is_exactly_the_same_json_evidence():
    assert_serialized_equal({'prior': {'mean': (0., 0., 0.)}}, {'prior': {'mean': [0., 0., 0.]}})


@pytest.mark.parametrize('saved', [{'x': .001000000000001}, {'x': .001, 'extra': False}, {}])
def test_no_numeric_tolerance_or_field_omission(saved):
    with pytest.raises(AssertionError): assert_serialized_equal({'x': .001}, saved)


def test_nonfinite_state_is_not_serializable_scientific_evidence():
    with pytest.raises(ValueError): assert_serialized_equal({'x': float('nan')}, {'x': None})
