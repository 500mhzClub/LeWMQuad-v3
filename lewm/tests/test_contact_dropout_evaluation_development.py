from copy import deepcopy
import numpy as np
import pytest

from lewm.contact_dropout_evaluation_development import (
    SUPPORT_MODES, support_index, contact_window, dropout_spans,
    displacement_error, summarize_windows, stats)
from lewm.causal_sensor_state import SensorContractError
from scripts.probe_go2_friction_frozen_rgbd_dropout_v1 import observe_once


def rows(missing=()):
    return [dict(measured_ns=1_500_000_000+i*20_000_000,
                 modes={m: dict(consensus_velocity_body_m_s=None if i in missing else [1., 0., 0.])
                        for m in SUPPORT_MODES}) for i in range(11)]


def test_endpoint_dropout_counts_in_both_intervals_without_zero_imputation():
    data=rows([5]); before=deepcopy(data); index=support_index(data)
    for start in (1_500_000_000, 1_600_000_000):
        result=contact_window(index,start,start+100_000_000)
        assert result['unavailable_samples']==1 and not result['contact_complete']
        assert result['unavailable_measured_ns']==[1_600_000_000]
    assert data==before


@pytest.mark.parametrize('fault', ['gap', 'duplicate', 'mixed_modes', 'nan'])
def test_support_clock_and_availability_contract(fault):
    data=rows()
    if fault=='gap': data.pop(3)
    elif fault=='duplicate': data[3]['measured_ns']=data[2]['measured_ns']
    elif fault=='mixed_modes': data[3]['modes']['stationary_centre']['consensus_velocity_body_m_s']=None
    else: data[3]['modes']['stationary_centre']['consensus_velocity_body_m_s']=[np.nan,0,0]
    with pytest.raises(ValueError): support_index(data)


def test_all_six_samples_required_and_only_100ms_allowed():
    index=support_index(rows()); del index[1_540_000_000]
    with pytest.raises(ValueError): contact_window(index,1_500_000_000,1_600_000_000)
    with pytest.raises(ValueError): contact_window(index,1_600_000_000,1_680_000_000)


def test_missing_spans_keep_boundaries_and_sample_counts():
    actual=dropout_spans(support_index(rows([0,1,4,8,9,10])))
    assert [r['missing_samples'] for r in actual]==[2,1,3]
    assert actual[-1]['last_missing_ns']==1_700_000_000
    assert dropout_spans(support_index(rows()))==[]


def test_relative_error_subtracts_offset_and_preserves_missingness():
    a={'position_initial_body_m':[3.,4.,5.]}; b={'position_initial_body_m':[3.1,4.,5.]}
    assert displacement_error(a,b,[0.,0.,0.],[.1,0.,0.])<1e-14
    assert displacement_error(a,b,[0.,0.,0.],[.13,0.,0.])==pytest.approx(.03)
    assert displacement_error(a,None,[0,0,0],[0,0,0]) is None


def test_missing_visual_errors_not_disguised_as_accuracy():
    result=summarize_windows([
        dict(contact_complete=False,displacement_error_m=None),
        dict(contact_complete=False,displacement_error_m=.02),
        dict(contact_complete=True,displacement_error_m=.01)])
    assert result['contact_dropout']['windows']==2
    assert result['contact_dropout']['visual_unavailable']==1
    assert result['contact_dropout']['displacement_error_m']['mean']==.02
    assert stats([])==dict(count=0,mean=None,p95=None,maximum=None)
    with pytest.raises(ValueError): stats([np.nan])


def test_visual_failure_is_terminal_without_reinvocation():
    class Broken:
        calls=0
        def observe(self,*args,**kwargs):
            self.calls+=1
            raise SensorContractError('synthetic loss of image support')
    model=Broken(); item,failure=observe_once(model,None,{}, {}, {},1500000000,0)
    assert item['status']=='TERMINAL_FAILURE' and item['state'] is None
    later,kept=observe_once(model,failure,{}, {}, {},1600000000,1)
    assert later['status']=='NOT_REINVOKED_AFTER_FAILURE' and kept==failure and model.calls==1
