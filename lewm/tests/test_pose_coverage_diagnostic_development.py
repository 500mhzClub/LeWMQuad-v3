from copy import deepcopy

import numpy as np
import pytest

from lewm.bounded_depth_surface_development import BoundedDepthSurface
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.pose_coverage_diagnostic_development import query_coverage


def state(rotation=None):
    return dict(rotation_initial_body_from_current_body=(np.eye(3) if rotation is None else rotation).tolist(),
                position_initial_body_m=[0.,0.,0.])


def test_pose_and_surface_unavailability_never_become_clearance():
    assert query_coverage(None,None,None,None,None)['status']=='POSE_UNAVAILABLE'
    s=state(); result=query_coverage(None,None,None,s,None)
    assert result['status']=='SURFACE_UNAVAILABLE' and result['floor_coverage'] is None


def test_real_rotation_tolerance_mismatch_is_explicit_not_relaxed():
    R=np.eye(3); R[0,0]+=1e-10
    np.testing.assert_array_equal(proper(R),R)
    surface=object.__new__(BoundedDepthSurface); surface.status='BOUNDED_MEASURED_SURFACE_AVAILABLE'
    s=state(R); before=deepcopy(s)
    result=query_coverage(surface,None,None,s,None)
    assert s==before and result['status']=='COVERAGE_CONTRACT_REJECTED' and result['floor_coverage'] is None
    assert result['reason']=='proper finite relative transform and supported backend required'
    assert result['orthogonality_max_abs']>1e-12 and result['determinant_abs_error']>1e-12
    # The helper neither mutates the pose nor weakens the original predicate.
    with pytest.raises(ValueError): surface.query(None,None,rotation_observation_from_body=R,
        translation_observation_from_body=[0.,0.,0.],point_error_by_shape={})


def test_success_forwards_exact_query_and_preserves_negative_coverage():
    class Surface:
        status='BOUNDED_MEASURED_SURFACE_AVAILABLE'
        def query(self,geometry,joints,**kwargs):
            assert geometry=='synthetic' and joints==[1.]
            np.testing.assert_array_equal(kwargs['rotation_observation_from_body'],np.eye(3))
            assert kwargs['translation_observation_from_body']==[0.,0.,0.]
            assert kwargs['point_error_by_shape']=={'a':0.}
            return {'floor_coverage':{'a':False}}
    result=query_coverage(Surface(),'synthetic',[1.],state(),{'a':0.})
    assert result['floor_coverage']=={'a':False}
    assert result['status']=='CONDITIONAL_ZERO_ADDITIONAL_ERROR_COVERAGE'


def test_unexpected_exception_remains_fatal():
    class Surface:
        status='BOUNDED_MEASURED_SURFACE_AVAILABLE'
        def query(self,*args,**kwargs): raise RuntimeError('synthetic infrastructure fault')
    with pytest.raises(RuntimeError,match='synthetic infrastructure fault'):
        query_coverage(Surface(),None,None,state(),{})


def test_successor_preserves_original_model_identity():
    from scripts.probe_go2_longer_motion_frozen_pose_development_v1 import RigidRGBDKeyframePose as original
    from scripts.probe_go2_longer_motion_pose_coverage_status_development_v1 import RigidRGBDKeyframePose as successor
    assert original is successor


def test_successor_preserves_original_scoring_and_surface_initialization():
    import inspect
    from scripts import probe_go2_longer_motion_frozen_pose_development_v1 as original
    from scripts import probe_go2_longer_motion_pose_coverage_status_development_v1 as successor
    for name in ('score','coverage_summary','initial_up'):
        assert inspect.getsource(getattr(original,name))==inspect.getsource(getattr(successor,name))
