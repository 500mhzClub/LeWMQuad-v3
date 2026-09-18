"""Preserve expanded-model arithmetic/state while providing the planner's API."""
from copy import deepcopy
import torch
from torch import nn

from lewm.all_phase_translation_bias_development import AllPhaseTranslationBiasModel
from lewm.training_translation_bias_development import TrainingTranslationBiasModel, HEADS
from lewm.pulse_timed_training_runner_development import state_digest


class AllPhasePlannerModel(TrainingTranslationBiasModel):
    """An explicit interface adapter, not another correction fit or checkpoint.

    The existing planner requires the TrainingTranslationBiasModel interface.
    Expanded correction receipts have their own already-validated schema; they
    are never relabeled as old receipts or fed through the old fit admission.
    Callers must authenticate the original expanded-model admission and state.
    """
    def __init__(self, source):
        if (type(source) is not AllPhaseTranslationBiasModel or source.training or source.base.training
                or any(parameter.grad is not None for parameter in source.parameters())
                or not source.corrected_heads or any(head not in HEADS for head in source.corrected_heads)):
            raise ValueError('original evaluation-only expanded correction model without gradients required')
        before = state_digest(source.state_dict())
        nn.Module.__init__(self)
        self.base = deepcopy(source.base)
        self.corrected_heads = tuple(source.corrected_heads)
        for head in self.corrected_heads:
            bias = getattr(source, head+'_xy_bias')
            if bias.shape != (8, 2) or bias.dtype != torch.float32 or not torch.isfinite(bias).all():
                raise ValueError('exact finite expanded float32 correction buffer required')
            self.register_buffer(head+'_xy_bias', bias.detach().clone())
        self.eval()
        if state_digest(self.state_dict()) != before or state_digest(source.state_dict()) != before:
            raise ValueError('adapter must preserve every original state key and tensor byte')

    # Reuse the exact expanded forward method; only the supported interface
    # and ownership of copied state differ. Training remains forbidden by the
    # inherited evaluation-only train method.
    forward = AllPhaseTranslationBiasModel.forward
