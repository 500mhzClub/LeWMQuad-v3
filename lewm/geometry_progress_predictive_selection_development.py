"""Explicit learned-outcome selection for the fixed progressing candidate bank.

This is a local selector component, not an authorized control loop. A caller
must authenticate a fitted model and compatible causal observations. In
particular, quiet-start training does not validate replanning while moving.
Contact scores are uncalibrated model outputs; utility is not a safety bound.
Native poses, geometry, outcome labels and future images have no input here.
"""
import math
import time
import numpy as np
import torch
from lewm.geometry_progress_pilot_development import ACTIONS,timed_candidate,candidate_commands
from lewm.cumulative_pulse_contact_development import CumulativePulseRGBBodyJEPA
from lewm.independent_pulse_input_ablation_development import transform_inputs


def score_candidates(prediction,*,goal_body_xy_m,contact_penalty_m):
    """Choose max predicted terminal distance reduction minus contact cost.

    All six complete four-second plans participate. Stable ties follow ACTIONS
    (hold first). The caller fixes the nonnegative contact cost before outcomes.
    """
    p=np.asarray(prediction)
    goal=np.asarray(goal_body_xy_m,float)
    if p.shape!=(6,8,5) or not np.issubdtype(p.dtype,np.floating) or not np.isfinite(p).all():
        raise ValueError('finite complete six-candidate eight-horizon predictions required')
    if goal.shape!=(2,) or not np.isfinite(goal).all():raise ValueError('finite body-frame local goal required')
    if isinstance(contact_penalty_m,bool) or not math.isfinite(contact_penalty_m) or contact_penalty_m<0:
        raise ValueError('explicit finite nonnegative contact cost required')
    if (np.diff(p[:,:,4],axis=1)<-1e-6).any():raise ValueError('cumulative contact logits cannot decrease')
    logits=p[:,-1,4].astype(float)
    probability=np.exp(-np.logaddexp(0.,-logits))
    progress=np.linalg.norm(goal)-np.linalg.norm(goal-p[:,-1,:2],axis=1)
    utility=progress-contact_penalty_m*probability
    if not np.isfinite(utility).all():raise ValueError('finite utility required')
    chosen=int(np.argmax(utility))
    return dict(action=ACTIONS[chosen],action_index=chosen,
        requested_command=candidate_commands(ACTIONS[chosen])[0],
        candidates=[dict(action=a,predicted_progress_m=float(progress[i]),
            predicted_contact_score=float(probability[i]),utility_m=float(utility[i])) for i,a in enumerate(ACTIONS)],
        goal_body_xy_m=goal.tolist(),contact_penalty_m=float(contact_penalty_m),
        score_contract='terminal_distance_reduction_minus_contact_cost',
        prediction_horizon_ns=4_000_000_000,contact_probability_calibrated=False,
        native_state_used=False,navigation_qualified=False)


def candidate_inputs(observation_history):
    shapes={'rgb':(4,3,96,128),'body':(4,20,63),'control':(4,15,7)}
    if not isinstance(observation_history,dict) or set(observation_history)!=set(shapes):
        raise ValueError('only causal RGB/body/control histories accepted')
    for k,shape in shapes.items():
        value=observation_history[k]
        if (not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=torch.float32
                or value.device.type!='cpu' or not torch.isfinite(value).all()):
            raise ValueError('finite unbatched CPU causal history tensors required')
    blocks,valid=zip(*(timed_candidate(a) for a in ACTIONS),strict=True)
    return dict(observation_history={k:v.unsqueeze(0).expand(6,*v.shape) for k,v in observation_history.items()},
        known_action_blocks=torch.stack(blocks),known_action_valid=torch.stack(valid))


@torch.inference_mode()
def select(model,observation_history,*,head,input_variant,goal_body_xy_m,contact_penalty_m):
    """Run genuine action-conditioned predictions; return all forecasts and cost.

    This helper neither loads nor admits a checkpoint and never trains a model.
    Receipt authentication and deployment-distribution checks belong to caller.
    """
    if not isinstance(model,CumulativePulseRGBBodyJEPA) or model.training:
        raise ValueError('explicit cumulative-event model in evaluation mode required')
    if head not in ('direct_outcomes','rollout_outcomes'):raise ValueError('explicit trained prediction head required')
    start=time.perf_counter_ns()
    inputs=transform_inputs(candidate_inputs(observation_history),input_variant=input_variant)
    output=model(**inputs)
    expected=torch.arange(1,9,dtype=torch.int64).mul(500_000_000).expand(6,8)
    if (output['prediction_valid'].dtype!=torch.bool or output['prediction_valid'].shape!=(6,8)
            or not output['prediction_valid'].all() or not torch.equal(output['target_offsets_ns'],expected)):
        raise ValueError('all prospective action horizons must be returned exactly')
    prediction=output[head].cpu().numpy()
    result=score_candidates(prediction,goal_body_xy_m=goal_body_xy_m,contact_penalty_m=contact_penalty_m)
    return result|dict(head=head,input_variant=input_variant,prediction=prediction.tolist(),
        selection_wall_ms=(time.perf_counter_ns()-start)/1e6,model_checkpoint_admitted_by_this_helper=False)
