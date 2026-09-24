import numpy as np
import pytest
from scripts.read_go2_executed_commitment_six_model_errors_v1 import relative_motion


def test_body_frame_translation_and_wrapped_relative_yaw():
    q=np.sqrt(.5)
    a=[2.,3.,1.,0.,0.,q,q]
    b=[1.,5.,1.,0.,0.,1.,0.]
    assert np.allclose(relative_motion(a,b),[2.,1.,np.pi/2],atol=1e-12)
    assert np.allclose(relative_motion(a,a),[0.,0.,0.],atol=1e-12)


def test_invalid_native_evaluator_input_rejected():
    with pytest.raises(ValueError):relative_motion([0]*6,[0]*7)
    with pytest.raises(ValueError):relative_motion([np.nan]*7,[0]*7)
