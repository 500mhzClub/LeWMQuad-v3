"""Original causal candidate interface with admitted training-only XY offsets."""
import time
import torch
from lewm.training_translation_bias_development import TrainingTranslationBiasModel
from lewm.observation_horizon_predictive_selection_development import candidate_inputs,score_candidates
from lewm.observation_horizon_input_ablation_development import transform_inputs

@torch.inference_mode()
def select(model,observation_history,*,head,input_variant,goal_body_xy_m,contact_penalty_m):
    """Run genuine action-conditioned predictions; return all forecasts and cost.

    This helper neither loads nor admits a checkpoint and never trains a model.
    Receipt authentication and deployment-distribution checks belong to caller.
    """
    if not isinstance(model,TrainingTranslationBiasModel) or model.training:
        raise ValueError('training-only translation wrapper in evaluation mode required')
    if head not in model.corrected_heads:raise ValueError('explicit trained prediction head required')
    start=time.perf_counter_ns()
    inputs=transform_inputs(candidate_inputs(observation_history),input_variant=input_variant)
    output=model(**inputs)
    expected=torch.arange(1,9,dtype=torch.int64).mul(100_000_000).expand(6,8)
    if (output['prediction_valid'].dtype!=torch.bool or output['prediction_valid'].shape!=(6,8)
            or not output['prediction_valid'].all() or not torch.equal(output['target_offsets_ns'],expected)):
        raise ValueError('all prospective action horizons must be returned exactly')
    prediction=output[head].cpu().numpy()
    result=score_candidates(prediction,goal_body_xy_m=goal_body_xy_m,contact_penalty_m=contact_penalty_m)
    return result|dict(first_prediction_horizon_ns=100_000_000,target_offsets_ns=expected[0].tolist(),head=head,input_variant=input_variant,prediction=prediction.tolist(),
        selection_wall_ms=(time.perf_counter_ns()-start)/1e6,model_checkpoint_admitted_by_this_helper=False,model_prediction_corrected=True,
        translation_bias_training_only=True,translation_bias_xy_m=getattr(model,head+'_xy_bias').cpu().tolist())

