"""One-factor visual-only EMA target for the existing multimodal JEPA model."""
import torch
from lewm.command_history_residual_learning_development import (
    CommandHistoryResidualModel, CommandHistoryResidualTrainer)
from lewm.pulse_timed_learning_development import active_parameters
from lewm.pulse_timed_training_runner_development import state_digest


class VisualTargetModel(CommandHistoryResidualModel):
    @torch.no_grad()
    def target(self, observation):
        # Same EMA visual trunk and fusion projection, with the 64 body and
        # 32 control embedding coordinates identically zero. No future body
        # or control values are read. Causal-context encoding is unchanged.
        visual = self.target_encoder.visual(observation['rgb'])
        return self.target_encoder.fuse(torch.cat((visual, visual.new_zeros((len(visual),96))), -1))


class VisualTargetTrainer(CommandHistoryResidualTrainer):
    def __init__(self, *, seed, latent_dim=32, learning_rate=1e-4, ema_momentum=.99):
        super().__init__('jepa', seed=seed, latent_dim=latent_dim,
                         learning_rate=learning_rate, ema_momentum=ema_momentum)
        expected_initial = self.initial_sha256
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = VisualTargetModel(latent_dim).cpu()
        self.initial_sha256 = state_digest(self.model.state_dict())
        assert self.initial_sha256 == expected_initial
        self.parameters = active_parameters(self.model, 'jepa')
        self.optimizer = torch.optim.AdamW(self.parameters, lr=learning_rate, weight_decay=0.)

    def checkpoint(self):
        value = super().checkpoint()
        value['schema'] = 'visual_target_jepa_training_development.v1'
        value['latent_target'] = 'EMA_visual_then_fusion_with_zero_body_control_embeddings'
        return value
