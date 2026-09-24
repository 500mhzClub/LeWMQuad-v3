import numpy as np
import pytest

from lewm.counterfactual_prefix_matching_development import rgb_difference


def test_sparse_render_difference_is_reported_not_hidden():
    a=np.zeros((480,640,3),dtype=np.uint8)
    b=a.copy(); b[279,592]=15
    value=rgb_difference(a,b)
    assert value['changed_pixels']==1 and value['maximum_channel_difference']==15
    assert value['rms_8bit']<1 and value['changed_pixel_fraction']<.001


def test_broad_low_amplitude_change_cannot_masquerade_as_sparse_aliasing():
    a=np.zeros((480,640,3),dtype=np.uint8)
    value=rgb_difference(a,a+1)
    assert value['rms_8bit']==1 and value['changed_pixel_fraction']==1


def test_wrong_native_shape_is_rejected():
    with pytest.raises(ValueError): rgb_difference(np.zeros((1,1,3),dtype=np.uint8),np.zeros((1,1,3),dtype=np.uint8))
