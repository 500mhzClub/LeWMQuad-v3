import copy

import numpy as np
import pytest

from lewm.tests.test_temporal_prediction_metrics_development import fixture
from lewm.temporal_prediction_metrics_development import prediction_report,initial_decisions
from scripts.audit_go2_temporal_rgb_body_learning_comparison_development_v1 import check_primary,check_choices


def test_scalar_primary_audit_rejects_corrupted_layout_or_macro_scores():
    p,b=fixture()
    for m in b['metadata']: m['offset_ns']=500_000_000
    report=prediction_report(p,b); check_primary(p,b,report)
    corrupt=copy.deepcopy(report); corrupt['later']['layout_macro']['contact_brier']+=.1
    with pytest.raises(ValueError,match='macro'): check_primary(p,b,corrupt)
    corrupt=copy.deepcopy(report); corrupt['later']['layouts'][0]['position_error_m']=.1
    with pytest.raises(ValueError,match='layout metric'): check_primary(p,b,corrupt)


def test_scalar_primary_audit_handles_no_motion_and_empty_later_subset():
    p,b=fixture(); report=prediction_report(p,b); check_primary(p,b,report)
    for m in b['metadata']: m['offset_ns']=500_000_000
    b['targets']['motion_valid'][:]=False; b['targets']['motion'][:]=np.nan
    check_primary(p,b,prediction_report(p,b))


def test_scalar_choice_audit_rejects_changed_action():
    _,b=fixture(5)
    for i,m in enumerate(b['metadata']): m.update(layout_id='one',action_index=i)
    b['known_action_valid'][:]=True
    b['targets']['motion_valid'][:]=True; b['targets']['contact_valid'][:]=True
    b['targets']['motion'][:]=0; b['targets']['contact'][:]=0
    p=np.zeros((5,8,5)); p[...,3]=1.; p[...,4]=-30.
    result=initial_decisions(p,b); check_choices(p,b,result)
    result['rows'][0]['chosen_action_index']=1
    with pytest.raises(ValueError,match='chosen'): check_choices(p,b,result)
