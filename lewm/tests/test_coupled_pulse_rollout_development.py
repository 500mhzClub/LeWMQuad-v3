"""Analytic/synthetic planner tests, not physical execution evidence."""
from dataclasses import FrozenInstanceError
import math
import numpy as np
import pytest
from lewm.coupled_pulse_rollout_development import COMMANDS,PulseEffect,PulseTable,compose,plan


def table():
    return PulseTable(tuple(PulseEffect(c,t,(.02*t if c[0] else 0.,
        -.005*t*np.sign(c[2]), .04*t*np.sign(c[2])),2)
        for c in COMMANDS for t in (2,5)), 'a'*64)


def test_start_frame_translation_and_unwrapped_yaw_composition():
    np.testing.assert_allclose(compose([1,2,math.pi/2],[.1,.02,math.pi]), [.98,2.1,1.5*math.pi])


def test_no_action_at_goal_is_not_execution_or_clearance():
    r=plan(table(),[0,0,0],[0,0,0])
    assert r['status']=='PREDICTED_GOAL_CANDIDATE' and r['action_indices']==[]
    assert not r['motion_permission'] and not r['body_sweep_checked'] and not r['learned_jepa_used']


@pytest.mark.parametrize('sign',[-1,1])
def test_winding_respects_signed_half_turn(sign):
    r=plan(table(),[0,0,0],[0,0,sign*math.pi],yaw_mode='winding',horizon=35)
    assert r['status']=='PREDICTED_GOAL_CANDIDATE',r
    assert abs(r['predicted_endpoint'][2]-sign*math.pi)<=.05
    state=np.zeros(3)
    for i in r['action_indices']:state=compose(state,table().effects[i].delta_xy_yaw)
    np.testing.assert_allclose(state,r['predicted_endpoint'],atol=1e-12)
    assert any(table().effects[i].command[0] for i in r['action_indices'])


def test_orientation_does_not_silently_promise_winding():
    assert plan(table(),[0,0,0],[0,0,2*math.pi])['action_indices']==[]
    r=plan(table(),[0,0,0],[0,0,2*math.pi],yaw_mode='winding',horizon=1)
    assert r['status']=='SEARCH_EXHAUSTED' and r['action_indices']


def test_turn_translation_included_and_supported_actions_only():
    r=plan(table(),[0,0,0],[.3,.1,.8],horizon=24)
    assert r['status']=='PREDICTED_GOAL_CANDIDATE',r
    assert all(table().effects[i].command in COMMANDS for i in r['action_indices'])
    assert r['minimum_command_ticks']==sum(table().effects[i].ticks+20 for i in r['action_indices'])


def test_no_reachable_endpoints_returns_no_false_solution():
    r=plan(table(),[0,0,0],[1,0,0],maximum_excursion_m=.001)
    assert r['status']=='SEARCH_EXHAUSTED' and not r['action_indices']


def test_composition_equivariance_and_inputs_not_mutated():
    p=np.array([.2,-.1,.7]);d=np.array([.03,-.02,.12]);p0=p.copy();d0=d.copy()
    frame=[2.,-3.,1.2]
    np.testing.assert_allclose(compose(frame,compose(p,d)),compose(compose(frame,p),d),atol=1e-12)
    np.testing.assert_array_equal(p,p0);np.testing.assert_array_equal(d,d0)


def test_deterministic_bounded_search_and_endpoint_excursion():
    kwargs=dict(horizon=10,beam_width=16,maximum_excursion_m=.15)
    a=plan(table(),[0,0,0],[.4,0,0],**kwargs)
    assert a==plan(table(),[0,0,0],[.4,0,0],**kwargs)
    assert a['expanded_nodes']<=10*16*6 and a['status']=='SEARCH_EXHAUSTED'
    p=np.zeros(3)
    for i in a['action_indices']:
        p=compose(p,table().effects[i].delta_xy_yaw)
        assert np.linalg.norm(p[:2])<=.15


@pytest.mark.parametrize('kwargs',[dict(horizon=36),dict(beam_width=0),dict(yaw_mode='shortest_signed'),dict(maximum_excursion_m=float('nan'))])
def test_invalid_search_configuration_rejected(kwargs):
    with pytest.raises(ValueError):plan(table(),[0,0,0],[.4,0,0],**kwargs)


def test_table_and_effect_custody_validation_and_immutability():
    with pytest.raises(ValueError):PulseTable(table().effects[:-1]+(table().effects[0],),'a'*64)
    with pytest.raises(ValueError):PulseEffect((0.,0.,.03),2,(0.,0.,.01),2)
    with pytest.raises(ValueError):PulseEffect(COMMANDS[0],2,(float('nan'),0,0),2)
    with pytest.raises(ValueError):PulseTable(table().effects,'unbound')
    with pytest.raises(FrozenInstanceError):table().effects[0].ticks=5
    with pytest.raises(ValueError):plan(table(),[0,0,float('inf')],[0,0,0])
