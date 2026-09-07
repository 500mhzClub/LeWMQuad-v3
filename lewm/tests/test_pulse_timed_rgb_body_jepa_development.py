"""Timing/causality/gradient tests only; no learned scientific result."""
from copy import deepcopy
import pytest
import torch
from lewm.pulse_timed_rgb_body_jepa_development import PulseTimedRGBBodyJEPA,pulse_brake_plan,validate_timed_plan
from lewm.coupled_pulse_rollout_development import COMMANDS


def data():
    torch.manual_seed(90691)
    history={'rgb':torch.rand(2,4,3,96,128),'body':torch.randn(2,4,20,63),'control':torch.randn(2,4,15,7)}
    pairs=[pulse_brake_plan(COMMANDS[0],2),pulse_brake_plan(COMMANDS[2],5)]
    return history,torch.stack([p[0] for p in pairs]),torch.stack([p[1] for p in pairs])


def test_short_and_long_pulses_have_true_future_times_without_invented_braking():
    h,p,v=data();active,offsets=validate_timed_plan(p,v,2)
    assert offsets.tolist()==[[500000000,1000000000,1500000000,2000000000,2200000000,0,0,0],
                              [500000000,1000000000,1500000000,2000000000,2500000000,0,0,0]]
    assert v.flatten(1).sum(1).tolist()==[22,25]
    assert p[0].reshape(40,3)[2:22].count_nonzero()==0
    assert not v[0].flatten()[22:].any()


def test_existing_model_rejects_partial_prefix_new_model_predicts_it():
    from lewm.temporal_rgb_body_jepa_development import validate_plan
    h,p,v=data()
    with pytest.raises(ValueError,match='partial'):validate_plan(p,v,2)
    m=PulseTimedRGBBodyJEPA(16);r=m(h,p,v)
    assert r['prediction_valid'].sum().item()==10
    for key in ('future_latents','direct_outcomes','rollout_outcomes'):
        assert torch.isfinite(r[key]).all() and r[key][~r['prediction_valid']].count_nonzero()==0


def test_later_plan_extension_cannot_change_earlier_boundary_predictions():
    h,p,v=data();m=PulseTimedRGBBodyJEPA(16).eval()
    long=p.clone();mask=torch.ones_like(v);long[:,5:,:,0]=-.5
    with torch.no_grad():short=m(h,p,v);extended=m(h,long,mask)
    for key in ('future_latents','direct_outcomes','rollout_outcomes'):
        torch.testing.assert_close(short[key][:,:4],extended[key][:,:4],rtol=0,atol=1e-6)
    assert short['target_offsets_ns'][0,4]!=extended['target_offsets_ns'][0,4]


def test_equal_zero_values_but_known_duration_changes_partial_endpoint():
    h,p,v=data();m=PulseTimedRGBBodyJEPA(16).eval();extended=v.clone();extended[0,4]=True
    with torch.no_grad():a=m(h,p,v);b=m(h,p,extended)
    assert not torch.equal(a['future_latents'][0,4],b['future_latents'][0,4])
    assert a['target_offsets_ns'][0,4].item()==2200000000 and b['target_offsets_ns'][0,4].item()==2500000000


@pytest.mark.parametrize('mode',['empty','gap','unknown_nonzero','nan','lateral','range','mask_dtype'])
def test_bad_tick_contracts_rejected(mode):
    h,p,v=data()
    if mode=='empty':v[0]=False;p[0]=0
    if mode=='gap':v[0,0,1]=False;p[0,0,1]=0
    if mode=='unknown_nonzero':p[0,7,0,0]=.2
    if mode=='nan':p[0,0,0,0]=float('nan')
    if mode=='lateral':p[0,0,0,1]=.2
    if mode=='range':p[0,0,0,0]=2
    if mode=='mask_dtype':v=v.float()
    with pytest.raises(ValueError):validate_timed_plan(p,v,2)


def test_no_targets_or_native_inputs_and_target_encoder_stays_frozen():
    h,p,v=data();m=PulseTimedRGBBodyJEPA(16)
    with pytest.raises(TypeError):m(h,p,v,targets={})
    changed=deepcopy(h);changed['native_pose']=torch.zeros(2,7)
    with pytest.raises(ValueError,match='undeclared'):m(changed,p,v)
    r=m(h,p,v);loss=r['future_latents'].square().mean()+r['rollout_outcomes'].square().mean()+r['direct_outcomes'].square().mean()
    loss.backward()
    assert all(x.grad is None for x in m.target_encoder.parameters())
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for name,x in m.named_parameters() if not name.startswith('target_encoder.'))


def test_public_rollout_helpers_use_partial_timing_not_inherited_full_blocks():
    h,p,v=data();m=PulseTimedRGBBodyJEPA(16).eval()
    with torch.no_grad():
        r=m(h,p,v);future=m.predict_latents(r['latent'],p,v)
        decoded=m.decode_rollout(r['latent'],future,r['prediction_valid'],r['target_offsets_ns'])
    torch.testing.assert_close(future,r['future_latents'],rtol=0,atol=0)
    torch.testing.assert_close(decoded,r['rollout_outcomes'],rtol=0,atol=0)
    with pytest.raises(TypeError):m.decode_rollout(r['latent'],future,r['prediction_valid'])
