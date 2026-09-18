"""Exact fit reuse cannot leak mutations or ignore changed observations/rules."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.frame_registration_memo_development import FrameRegistrationMemo
from lewm.conditioned_support_tracker_development import _register, bind, CONDITIONED_RULES
from lewm.tests.test_gyro_conditioned_pair_pose_development import example


def inputs():
    candidate,R,_=example('primary');r=candidate['registration']
    arrays=[np.asarray(r[name]) for name in ('reference_inlier_points_body_m',
        'current_inlier_points_body_m','reference_inlier_pixels','current_inlier_pixels')]
    return arrays,dict(gyro_rotation=R,mode='joint',frame=14)


def test_fit_and_mask_are_exact_and_caller_mutation_is_isolated():
    arrays,kw=inputs();memo=FrameRegistrationMemo(enabled=True)
    expected=_register(*arrays,**kw)
    first=memo.call(_register,*arrays,**kw)
    first[0][0,0]=999;first[2][:]=False;first[3]['inliers']=-1
    second=memo.call(_register,*arrays,**kw)
    for actual,wanted in zip(second[:3],expected[:3]):np.testing.assert_array_equal(actual,wanted)
    assert second[3]==expected[3] and memo.hits==1


def test_changed_current_observation_is_recomputed():
    arrays,kw=inputs();memo=FrameRegistrationMemo(enabled=True)
    first=memo.call(_register,*arrays,**kw)
    changed=deepcopy(arrays);changed[0][:,0]+=.001
    second=memo.call(_register,*changed,**kw)
    assert not np.array_equal(first[1],second[1]) and memo.hits==0


def test_changed_support_rules_cannot_reuse_a_success():
    arrays,kw=inputs();memo=FrameRegistrationMemo(enabled=True)
    memo.call(_register,*arrays,**kw)
    strict=bind(_register,RULES=CONDITIONED_RULES|dict(minimum_matches=100))
    with pytest.raises(ValueError,match='insufficient rigid-pose matches'):
        memo.call(strict,*arrays,**kw)
    assert memo.hits==0


def test_cached_rejection_is_fresh_and_preserves_reason():
    arrays,kw=inputs();memo=FrameRegistrationMemo(enabled=True)
    short=[a[:2] for a in arrays];failures=[]
    for _ in range(2):
        with pytest.raises(ValueError) as error:memo.call(_register,*short,**kw)
        failures.append(error.value)
    assert str(failures[0])==str(failures[1]) and failures[0] is not failures[1]
    assert memo.failure_hits==1
