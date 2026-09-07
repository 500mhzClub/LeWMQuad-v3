from copy import deepcopy

import numpy as np
import pytest

from lewm.depth_inertial_stress_development import perturb,CASES
from lewm.tests.test_depth_local_surfaces_development import packet
from lewm.tests.test_observed_traversal_controller_development import Stream


@pytest.mark.parametrize('case',CASES)
def test_explicit_fault_does_not_modify_original_packets_or_rgb(case):
    stream=Stream()
    for k in range(81): p,_,_=stream.frame(k)
    _,d,_=packet('front',tick=80); before=deepcopy(d); force=p['sensor_state']['sensed']['specific_force']['values'].copy()
    new_p,new_d=perturb(p,d,case,first_ns=1_600_000_000)
    assert np.array_equal(d['valid'],before['valid']) and np.array_equal(d['depth_m'],before['depth_m'])
    assert np.array_equal(p['sensor_state']['sensed']['specific_force']['values'],force)
    assert np.array_equal(new_p['image']['rgb'],p['image']['rgb'])
    assert not new_d['valid'][:,:280].any() and not new_d['valid'][:,360:].any()
    assert np.all(new_d['depth_m'][~new_d['valid']]==0)
    if case=='depth_dropout': assert not new_d['valid'].any()
    else: assert np.array_equal(new_d['valid'][:,280:360],d['valid'][:,280:360])


def test_bias_history_is_time_causal_and_immutable_across_overlap():
    stream=Stream(); last=None
    for tick in range(85):
        p=stream.frame(tick)[0]; _,d,_=packet('front',tick=tick)
        new,_=perturb(p,d,'narrow_depth_bias_0_2',first_ns=1_600_000_000)
        force=new['sensor_state']['sensed']['specific_force']
        if last is not None:
            for i,t in enumerate(force['measured_ns']):
                match=np.flatnonzero(last['measured_ns']==t)
                if len(match): assert np.array_equal(force['values'][i],last['values'][match[0]])
        if tick<80: assert np.all(force['values'][:,1]==0.)
        if tick==80: assert force['values'][-1,1]==.2 and np.all(force['values'][:-1,1]==0.)
        last=deepcopy(force)


def test_narrow_depth_recovers_at_fixed_endpoint_not_an_outcome_dependent_time():
    stream=Stream()
    for tick in range(141): p=stream.frame(tick)[0]
    _,d,_=packet('front',tick=140)
    _,new=perturb(p,d,'narrow_depth',first_ns=1_600_000_000)
    assert np.array_equal(new['valid'],d['valid'])
