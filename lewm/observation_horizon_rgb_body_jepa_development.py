"""Fresh eight-step RGB/body/control model with actual 100-ms action blocks."""
import torch
from torch import nn
from lewm.cumulative_pulse_contact_development import CumulativePulseRGBBodyJEPA,cumulative_outcomes
from lewm.observation_horizon_plan_development import validate_plan


class ObservationHorizonRGBBodyJEPA(CumulativePulseRGBBodyJEPA):
    def __init__(self,latent_dim=32):
        super().__init__(latent_dim)
        # One three-axis requested command and one validity bit per step.
        self.transition=nn.Sequential(nn.Linear(latent_dim+4,256),nn.SiLU(),nn.Linear(256,latent_dim))
        self.direct_plan=nn.GRU(4,128,batch_first=True)

    def predict_latents(self,z,blocks,valid):
        active,_=validate_plan(blocks,valid,len(z))
        tokens=torch.cat([blocks.flatten(2),valid.to(z.dtype)],-1)
        if tokens.device!=z.device or tokens.dtype!=z.dtype:raise ValueError('common model/plan dtype and device required')
        future=[]
        for i in range(8):
            selected=active[:,i];updated=z.clone()
            if selected.any():updated[selected]=z[selected]+self.transition(torch.cat([z[selected],tokens[selected,i]],-1))
            z=updated;future.append(torch.where(selected[:,None],z,torch.zeros_like(z)))
        return torch.stack(future,1)

    def forward(self,observation_history,known_action_blocks,known_action_valid):
        batch=observation_history['rgb'].shape[0]
        active,offsets=validate_plan(known_action_blocks,known_action_valid,batch)
        z,_=self.encode_history(observation_history)
        future=self.predict_latents(z,known_action_blocks,known_action_valid)
        tokens=torch.cat([known_action_blocks.flatten(2),known_action_valid.to(z.dtype)],-1)
        seconds=offsets.to(z.dtype).unsqueeze(-1)/1e9
        planned,_=self.direct_plan(tokens)
        raw=self.direct_decode(torch.cat([z[:,None].expand(-1,8,-1),planned,seconds],-1))
        return dict(latent=z,future_latents=future,prediction_valid=active,target_offsets_ns=offsets,
            direct_outcomes=cumulative_outcomes(raw,active,offsets),
            rollout_outcomes=self.decode_rollout(z,future,active,offsets))
