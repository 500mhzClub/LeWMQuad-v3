import copy

import pytest
import torch

from lewm.temporal_rgb_body_jepa_development import TemporalRGBBodyJEPA,validate_plan
from lewm.temporal_rgb_body_learning_development import training_loss,active_parameters,layout_batches,per_window_outcome_loss


def batch():
    torch.manual_seed(31)
    history={'rgb':torch.rand(2,4,3,96,128),'body':torch.randn(2,4,20,63),'control':torch.randn(2,4,15,7)}
    mask=torch.zeros(2,8,5,dtype=torch.bool); mask[0,:2]=True; mask[1,:4]=True
    plans=torch.zeros(2,8,5,3); plans[mask]=torch.tensor([.5,0.,.25])
    valid=mask.all(-1)
    future={k:torch.rand(2,8,*v.shape[2:]) for k,v in history.items()}
    for v in future.values(): v[~valid]=float('nan')
    motion=torch.full((2,8,3),float('nan')); motion[valid]=.1
    contact=torch.full((2,8),float('nan')); contact[valid]=0.
    return {'observation_history':history,'known_action_blocks':plans,'known_action_valid':mask,
        'targets':{'future_observations':future,'future_valid':valid,'motion':motion,'motion_valid':valid.clone(),
            'contact':contact,'contact_valid':valid.clone()},'metadata':[]}


def forward(model,data): return model(data['observation_history'],data['known_action_blocks'],data['known_action_valid'])


def test_outputs_explicitly_mask_unavailable_future_and_use_history():
    data=batch(); model=TemporalRGBBodyJEPA(16).eval()
    with torch.no_grad():
        original=forward(model,data)
        changed=copy.deepcopy(data)
        for v in changed['observation_history'].values(): v[:,0]=0.
        alternative=forward(model,changed)
    active=data['known_action_valid'].all(-1)
    assert torch.equal(original['prediction_valid'],active)
    assert not torch.equal(original['latent'],alternative['latent'])
    for key in ('future_latents','direct_outcomes','rollout_outcomes'):
        assert torch.isfinite(original[key]).all()
        assert torch.count_nonzero(original[key][~active])==0


def test_known_prefix_predictions_invariant_to_later_known_commands_and_plan_length():
    data=batch(); model=TemporalRGBBodyJEPA(16).eval()
    full=copy.deepcopy(data); full['known_action_valid'][:]=True
    full['known_action_blocks'][:,4:,:,0]=-.5
    with torch.no_grad():
        short=forward(model,data); long=forward(model,full)
    for key in ('future_latents','direct_outcomes','rollout_outcomes'):
        assert torch.allclose(short[key][short['prediction_valid']],long[key][short['prediction_valid']],atol=1e-6,rtol=0)


def test_predictions_do_not_accept_targets_or_privileged_metadata():
    data=batch(); model=TemporalRGBBodyJEPA(16)
    with pytest.raises(TypeError): model(**data)
    data['observation_history']['world_pose']=torch.zeros(2,4,7)
    with pytest.raises(ValueError,match='undeclared'): forward(model,data)


@pytest.mark.parametrize('mode',['empty','gap','partial','nonzero_unknown','nan','lateral','range','dtype'])
def test_bad_plan_contracts_rejected(mode):
    data=batch(); p=data['known_action_blocks']; v=data['known_action_valid']
    if mode=='empty': v[0]=False; p[0]=0.
    if mode=='gap': v[0,3]=True
    if mode=='partial': v[0,1,0]=False; p[0,1,0]=0.
    if mode=='nonzero_unknown': p[0,7,0,0]=.1
    if mode=='nan': p[0,0,0,0]=float('nan')
    if mode=='lateral': p[0,0,0,1]=.1
    if mode=='range': p[0,0,0,0]=2.
    if mode=='dtype': v=v.float()
    with pytest.raises(ValueError): validate_plan(p,v,2)


@pytest.mark.parametrize('condition',['direct','supervised_rollout','jepa'])
def test_matched_losses_backpropagate_only_active_modules_and_never_target(condition):
    data=batch(); model=TemporalRGBBodyJEPA(16)
    loss,parts=training_loss(model,data,condition); loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active_parameters(model,condition))
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert ('latent_prediction' in parts)==(condition=='jepa')
    if condition=='direct': assert all(p.grad is None for p in model.transition.parameters())


def test_equal_observation_exposure_and_shared_loss_terms_before_updates():
    data=batch(); model=TemporalRGBBodyJEPA(16); populations=[]; terms=[]
    for condition in ('direct','supervised_rollout','jepa'):
        sizes=[]
        hook=model.encoder.register_forward_pre_hook(lambda module,args:sizes.append(len(args[0]['rgb'])))
        _,parts=training_loss(model,data,condition); hook.remove()
        populations.append(sizes); terms.append(parts)
    assert populations==[[8,6]]*3
    for key in ('direct_outcome','variance','covariance'):
        assert terms[0][key]==terms[1][key]==terms[2][key]


def test_target_ema_and_invalid_horizon_supervision():
    data=batch(); model=TemporalRGBBodyJEPA(16)
    initial=next(model.target_encoder.parameters()).clone()
    with torch.no_grad(): next(model.encoder.parameters()).add_(1.)
    model.update_target(.9)
    assert torch.allclose(next(model.target_encoder.parameters()),initial+.1,atol=1e-6)
    data['targets']['contact_valid'][0,7]=True
    with pytest.raises(ValueError,match='outside known plan'): training_loss(model,data,'jepa')


def test_outcome_loss_weights_windows_not_remaining_horizon_count():
    data=batch(); t=data['targets']; prediction=torch.zeros(2,8,5)
    prediction[1,:,4]=2.
    args=[t[k] for k in ('motion','contact','motion_valid','contact_valid')]
    full=per_window_outcome_loss(prediction,*args)
    separate=[per_window_outcome_loss(prediction[i:i+1],*[a[i:i+1] for a in args]) for i in range(2)]
    assert full==pytest.approx(float((separate[0]+separate[1])/2))


def metadata():
    return [{'layout_id':f'train-{layout:02d}','action_index':a,'offset_ns':offset,'data_role':'train'}
        for layout in range(16) for a in range(5) for offset in (0,500_000_000)]


def test_schedule_keeps_layouts_unique_and_all_actions_balanced_deterministically():
    rows=metadata(); schedule=list(layout_batches(rows,0,2026091700))
    assert schedule==list(layout_batches(rows,0,2026091700))
    assert schedule!=list(layout_batches(rows,1,2026091700))
    assert len(schedule)==5
    for indices in schedule: assert len({rows[i]['layout_id'] for i in indices})==16
    for layout in {r['layout_id'] for r in rows}:
        assert sorted(rows[i]['action_index'] for indices in schedule for i in indices if rows[i]['layout_id']==layout)==list(range(5))


@pytest.mark.parametrize('mode',['validation','duplicate','missing_action','missing_layout'])
def test_schedule_rejects_role_leakage_or_missing_population(mode):
    rows=metadata()
    if mode=='validation': rows[0]['data_role']='validation'
    if mode=='duplicate': rows.append(rows[0].copy())
    if mode=='missing_action': rows=[r for r in rows if not (r['layout_id']=='train-00' and r['action_index']==0)]
    if mode=='missing_layout': rows=[r for r in rows if r['layout_id']!='train-00']
    with pytest.raises(ValueError): list(layout_batches(rows,0,1))
