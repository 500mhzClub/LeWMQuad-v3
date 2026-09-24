"""Matched exposure and gradients, not evidence of learned predictive quality."""
from copy import deepcopy
import pytest
import torch
from lewm.pulse_timed_learning_development import training_loss,active_parameters,CONDITIONS,join_sample
from lewm.pulse_timed_rgb_body_jepa_development import PulseTimedRGBBodyJEPA,validate_timed_plan
from lewm.tests.test_pulse_timed_rgb_body_jepa_development import data


def batch():
    history,blocks,mask=data();active,offsets=validate_timed_plan(blocks,mask,2)
    future={k:torch.rand((2,8,*v.shape[2:])) for k,v in history.items()}
    for v in future.values():v[~active]=float('nan')
    motion=torch.full((2,8,3),float('nan'));motion[active]=.1
    contact=torch.full((2,8),float('nan'));contact[active]=0.
    return dict(inputs=dict(observation_history=history,known_action_blocks=blocks,known_action_valid=mask),
        targets=dict(future_observations=future,future_valid=active.clone(),motion=motion,motion_valid=active.clone(),
            contact=contact,contact_valid=active.clone(),target_offsets_ns=offsets))


@pytest.mark.parametrize('condition',CONDITIONS)
def test_partial_objectives_reach_only_active_modules(condition):
    model=PulseTimedRGBBodyJEPA(16);b=batch();loss,parts=training_loss(model,b,condition);loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active_parameters(model,condition))
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert ('latent_prediction' in parts)==(condition=='jepa')
    if condition=='direct':assert all(p.grad is None for p in model.transition.parameters())


def test_shared_image_population_and_shared_loss_terms():
    b=batch();model=PulseTimedRGBBodyJEPA(16);populations=[];terms=[]
    for condition in CONDITIONS:
        sizes=[];hook=model.encoder.register_forward_pre_hook(lambda m,args:sizes.append(len(args[0]['rgb'])))
        _,parts=training_loss(model,b,condition);hook.remove();populations.append(sizes);terms.append(parts)
    assert populations==[[8,10]]*3
    for k in ('direct_outcome','variance','covariance'):assert terms[0][k]==terms[1][k]==terms[2][k]


def test_contact_image_and_missing_rgb_native_motion_masks_are_independent():
    b=batch();t=b['targets'];t['motion_valid'][0,4]=False;t['motion'][0,4]=float('nan');t['contact'][0,4]=1.
    t['future_valid'][1,4]=False
    for v in t['future_observations'].values():v[1,4]=float('nan')
    for c in CONDITIONS:assert torch.isfinite(training_loss(PulseTimedRGBBodyJEPA(16),b,c)[0])


def test_absent_future_images_keep_shared_past_regularization_no_latent_loss():
    b=batch();b['targets']['future_valid'][:]=False
    for v in b['targets']['future_observations'].values():v[:]=float('nan')
    loss,parts=training_loss(PulseTimedRGBBodyJEPA(16),b,'jepa')
    assert torch.isfinite(loss) and 'latent_prediction' not in parts


@pytest.mark.parametrize('mode',['time','unknown','contact_motion','contact_range','privileged','nan_valid'])
def test_bad_target_or_inference_contract_rejected(mode):
    b=batch();t=b['targets']
    if mode=='time':t['target_offsets_ns'][0,4]=2_500_000_000
    elif mode=='unknown':t['future_valid'][0,7]=True
    elif mode=='contact_motion':t['contact'][0,0]=1.
    elif mode=='contact_range':t['contact'][0,0]=.5
    elif mode=='privileged':b['inputs']['native_pose']=torch.zeros(2,7)
    else:t['motion'][0,0]=float('nan')
    with pytest.raises(ValueError):training_loss(PulseTimedRGBBodyJEPA(16),b,'jepa')


def test_targets_cannot_change_forward_predictions():
    b=batch();model=PulseTimedRGBBodyJEPA(16).eval()
    with torch.no_grad():a=model(**b['inputs'])
    changed=deepcopy(b)
    for v in changed['targets']['future_observations'].values():v.fill_(1.)
    changed['targets']['motion'].fill_(100.)
    with torch.no_grad():c=model(**changed['inputs'])
    for k in a:torch.testing.assert_close(a[k],c[k],rtol=0,atol=0)


def test_actual_recorded_observation_native_join_and_matched_objectives():
    from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
    from lewm.recorded_pulse_native_targets_development import RecordedPulseNativeTargets
    from lewm.pulse_timed_observation_pairing_development import observation_pair_tensors
    from scripts.check_go2_pulse_timed_pairing_v1 import INPUT,OUTPUT
    from scripts.startup_raw_sensor_audit_development import read_json,read_npz
    windows=read_json(OUTPUT,'windows.json');samples=[]
    for c in ('nominal_left','lower_friction_left'):
        w=next(w for w in windows if w['condition']==c)
        pair=observation_pair_tensors(IntentReturnRGBDReplay(INPUT/c),w)
        native=RecordedPulseNativeTargets(read_npz(INPUT/c,'physics_trace.npz')).labels(w)
        samples.append(join_sample(pair,native))
        bad=deepcopy(native);bad['target_offsets_ns'][4]+=100_000_000
        with pytest.raises(ValueError,match='clock'):join_sample(pair,bad)
    def stack(items):
        if isinstance(items[0],dict):return {k:stack([x[k] for x in items]) for k in items[0]}
        return torch.stack(items)
    b=stack(samples);model=PulseTimedRGBBodyJEPA(16)
    for condition in CONDITIONS:
        loss,parts=training_loss(model,b,condition)
        assert torch.isfinite(loss) and parts['direct_outcome']>=0
    # No optimizer step, model fitting, checkpoint or performance comparison.
