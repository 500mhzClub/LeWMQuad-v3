from pathlib import Path

import numpy as np

from scripts.run_go2_depth_inertial_moment_replay_development_v1 import (
    OUTPUT,NEW_SOURCES,STUDIES,COUNTS,CASES,FRAME_COUNT,score)
from scripts.audit_go2_depth_inertial_moment_replay_development_v1 import assumption_diagnostics


def test_distinct_bound_source_and_fixed_population():
    assert OUTPUT.name=='go2_depth_inertial_moment_replay_development_v1_attempt_001'
    assert len(STUDIES)==5 and sum(COUNTS)==3610 and len(CASES)==4 and FRAME_COUNT==181
    assert all(Path(p).is_file() for p in NEW_SOURCES)
    assert 'lewm/depth_inertial_moment_fusion_development.py' in NEW_SOURCES


def test_error_check_cannot_substitute_for_velocity_and_acceleration_assumptions():
    # Exact position can coexist with a poor velocity/acceleration state.
    rows=[{'position_initial_body_m':[0,0,0],'translation_previous_body_m':None,
        'position_error_scale_m':0,'usable_under_declared_proxy_budget':True,'kind':'INITIAL_RELATIVE_ANCHOR'},
        {'position_initial_body_m':[.01,0,0],'translation_previous_body_m':[.01,0,0],
        'position_error_scale_m':.01,'usable_under_declared_proxy_budget':True,'kind':'DEPTH_CONSTRAINED_TRANSLATION',
        'depth_rank':3,'velocity_initial_body_m_s':[.2,0,0],'acceleration_initial_body_m_s2':[.1,0,0]}]
    poses=np.array([[0,0,0,0,0,0,1],[.01,0,0,0,0,0,1]])
    result=score(rows,poses); diag=assumption_diagnostics(result,poses,np.array([[.1,0,0],[.1,0,0]]))
    assert result['passes_declared_replay_checks']
    assert diag['full_depth_velocity_errors_above_0_005']==1
    assert diag['interval_acceleration_errors_above_0_02']==1
    assert not diag['operating_envelope_validated']
