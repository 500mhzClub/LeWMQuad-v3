from copy import deepcopy

import numpy as np
import pytest

from scripts.audit_go2_depth_inertial_fusion_replay_development_v1 import audit_rows, matrix
from scripts.run_go2_depth_inertial_fusion_replay_development_v1 import score
from lewm.tests.test_depth_inertial_fusion_development import initialized,frame,state


def fixture():
    stream,model=initialized(); records=[]; outputs=[]
    # A fresh estimator includes its anchor in this small audit fixture.
    from lewm.depth_inertial_fusion_development import WeakSubspaceIntegrator
    model=WeakSubspaceIntegrator()
    from lewm.tests.test_observed_traversal_controller_development import Stream
    stream=Stream()
    for tick in range(3):
        p=frame(stream,tick); s=state(p,None if tick==0 else [.01,0,0],[[0,1,0]] if tick==2 else ())
        outputs.append(model.observe(p,s)); records.append({'observation_index':tick,'observer':s})
    poses=np.array([[.01*k,0,0,0,0,0,1] for k in range(3)])
    return score(outputs,poses),records,poses


def test_independent_quaternion_handedness():
    s=2**-.5
    np.testing.assert_allclose(matrix([0,0,s,s])@[1,0,0],[0,1,0],atol=1e-12)


def test_independent_complete_audit():
    result,records,poses=fixture()
    assert audit_rows(result,records,poses)['weak_intervals']==1


@pytest.mark.parametrize('fault',['metric','component','position','kind','summary'])
def test_audit_rejects_corrupted_result(fault):
    result,records,poses=fixture(); result=deepcopy(result)
    if fault=='metric': result['rows'][1]['position_error_m']=.2
    elif fault=='component': result['rows'][2]['fusion']['translation_previous_body_m'][0]=.02
    elif fault=='position': result['rows'][2]['fusion']['position_initial_body_m'][1]=.02
    elif fault=='kind': result['rows'][2]['fusion']['kind']='DEPTH_CONSTRAINED_TRANSLATION'
    elif fault=='summary': result['weak_intervals']=0
    with pytest.raises(ValueError): audit_rows(result,records,poses)
