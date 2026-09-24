from copy import deepcopy

import numpy as np

from scripts.run_go2_depth_inertial_fusion_replay_development_v1 import score, STUDIES, COUNTS


def test_fixed_complete_development_population():
    assert len(STUDIES)==5 and sum(COUNTS)==3610
    assert STUDIES[-1]=='depth_floor_hold'


def test_evaluator_detects_step_position_and_proxy_failure_without_mutating_predictions():
    outputs=[{'position_initial_body_m':[0.,0.,0.],'translation_previous_body_m':None,
        'position_error_scale_m':0.,'usable_under_declared_proxy_budget':True,'kind':'INITIAL_RELATIVE_ANCHOR'},
        {'position_initial_body_m':[.12,0.,0.],'translation_previous_body_m':[.12,0.,0.],
        'position_error_scale_m':.01,'usable_under_declared_proxy_budget':True,
        'kind':'INERTIALLY_PREDICTED_WEAK_COMPONENT'}]
    before=deepcopy(outputs); poses=np.array([[0,0,0,0,0,0,1],[.02,0,0,0,0,0,1]],dtype=float)
    result=score(outputs,poses)
    assert outputs==before
    assert result['weak_intervals']==1 and result['proxy_exceedances']==1
    assert not result['checks']['step_error_at_most_1cm']
    assert not result['checks']['maximum_position_error_at_most_5cm']
    assert not result['checks']['no_proxy_exceedance']
    assert not result['passes_declared_replay_checks']


def test_evaluator_uses_initial_and_previous_body_frames_not_world_coordinates():
    s=2**-.5
    poses=np.array([[2,3,0,0,0,s,s],[2,3.1,0,0,0,s,s]])
    outputs=[{'position_initial_body_m':[0,0,0],'translation_previous_body_m':None,
        'position_error_scale_m':0.,'usable_under_declared_proxy_budget':True,'kind':'INITIAL_RELATIVE_ANCHOR'},
        {'position_initial_body_m':[.1,0,0],'translation_previous_body_m':[.1,0,0],
        'position_error_scale_m':.001,'usable_under_declared_proxy_budget':True,'kind':'DEPTH_CONSTRAINED_TRANSLATION'}]
    result=score(outputs,poses)
    assert result['passes_declared_replay_checks']
    assert result['maximum_step_error_m']<1e-12
