import pytest
from lewm.native_link_segmentation_map_development import decode


def test_native_background_is_preserved_and_never_a_robot_label():
    result,ids=decode({0:-1,1:(2,3),2:(4,7),3:(4,8)},4)
    assert result=={'0':-1,'1':[2,3],'2':[4,7],'3':[4,8]} and ids==[2,3]


@pytest.mark.parametrize('mapping',({1:(4,7)},{0:0,1:(4,7)},{0:-1,1:4},
    {0:-1,1:(4,-1)},{0:-1,1:(4,7,9)},{0:-1,1:(2,3)}))
def test_malformed_or_missing_robot_identity_is_rejected(mapping):
    with pytest.raises(ValueError):decode(mapping,4)
