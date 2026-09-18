"""Check horizon conditioning on actual training RGB and a retained predictor."""
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from lewm.horizon_conditioned_dense_predictor_development import HorizonConditionedDensePredictor
from scripts import train_go2_balanced_start_predictor_development as fit

RESULT=Path('docs/go2_horizon_predictor_initial_check_2026-09-18.json')


def main():
    assert not RESULT.exists()
    torch.set_num_threads(4);torch.manual_seed(2026091801)
    samples,paths,_=fit.dataset();sample=samples[-1]
    features,encoding=fit.encode([Path(paths[i]) for i in sample['frames']])
    context=features[:3].float().cuda()[None]
    target=features[3].float().cuda()[None]
    control=torch.tensor([sample['control']],device='cuda')
    actions=torch.zeros(1,8,2,device='cuda')
    actions[:,:5]=torch.tensor(sample['action'],device='cuda').reshape(1,5,2)
    actions[:,5:]=float('nan')
    horizon=torch.tensor([5],device='cuda')
    mask=torch.ones(1,768,dtype=torch.bool,device='cuda')
    parent=fit.load('mixed_action').cuda()
    model=HorizonConditionedDensePredictor(copy.deepcopy(parent)).cuda().eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=parent(context,actions[:,:5].reshape(1,10),mask,control=control)
        actual=model(context,actions,horizon,mask,control=control)
    discrepancy=float((expected.float()-actual.float()).abs().max())
    assert torch.equal(expected,actual),discrepancy
    # Test temporal causality for every supported endpoint, including ignored NaNs.
    for h in range(1,9):
        a=torch.randn(2,8,2);ticks=torch.full((2,),h,dtype=torch.long)
        b=a.clone();b[:,h:]=float('nan')
        assert torch.equal(model.condition(a,ticks),model.condition(b,ticks))
        changed=a.clone();changed[:,h-1]+=1
        assert not torch.equal(model.condition(a,ticks),model.condition(changed,ticks))
        blind=model.condition(a,ticks,action_blind=True)
        assert torch.equal(blind,model.condition(changed,ticks,action_blind=True))
        assert torch.equal(blind[:,:16],torch.zeros_like(blind[:,:16]))
        torch.testing.assert_close(blind[:,-1],torch.full((2,),(h-5)/5))
    # The new time/late-command channels must receive gradients before fitting.
    model.requires_grad_(True)
    actions=torch.full((1,8,2),.1,device='cuda');horizon.fill_(8)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prediction=model(context,actions,horizon,mask,control=control)
    loss=F.l1_loss(F.layer_norm(prediction.float(),(1024,)),target)
    loss.backward()
    gradient=model.backbone.action[0].extension.weight.grad
    assert torch.isfinite(gradient).all() and (gradient.abs().sum(0)>0).all()
    result=dict(status='PASS',training_sample_id=sample['sample_id'],
        source_sha256={p:fit.digest(p) for p in (__file__,'lewm/horizon_conditioned_dense_predictor_development.py')},
        parent_checkpoint_sha256=fit.digest(fit.OUTPUT/'mixed_action_final.pt'),
        native_500ms_initial_output_max_abs_error=discrepancy,
        post_target_command_invariance_horizons=list(range(1,9)),
        action_blind_retains_target_time=True,new_input_gradient_l1=gradient.abs().sum(0).tolist(),
        added_parameters=gradient.numel(),encoding=encoding,optimizer_steps=0,
        transfer_used=False,new_navigation=False)
    fit.save(RESULT,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
