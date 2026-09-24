"""Reuse one observation encoding for an explicitly broadcast candidate batch."""
import torch

from lewm.live_planning_stage_profile_development import LivePlanningStageProfileRuntime


def install_shared_history_encoding(model):
    original = model.encode_history
    receipt = dict(shared_calls=0, ordinary_calls=0)

    def encode(history):
        # A zero batch stride proves all candidates alias the same history.
        # Distinct histories, training, and malformed inputs retain the original
        # implementation, including its validation. No cross-call cache exists.
        shared = (not model.training and not torch.is_grad_enabled()
                  and isinstance(history, dict) and set(history) == {'rgb', 'body', 'control'}
                  and all(isinstance(v, torch.Tensor) and v.ndim >= 2
                          and v.shape[0] == 6 and v.stride(0) == 0 for v in history.values()))
        if not shared:
            receipt['ordinary_calls'] += 1
            return original(history)
        z, encoded = original({k: v[:1] for k, v in history.items()})
        receipt['shared_calls'] += 1
        return z.expand(6, -1), encoded.expand(6, -1, -1)

    model.encode_history = encode
    return receipt


class SharedCandidateHistoryRuntime(LivePlanningStageProfileRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_history_receipt = install_shared_history_encoding(self.model)
