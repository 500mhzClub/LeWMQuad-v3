from __future__ import annotations

import torch

from scripts.train_jepa_pooled_lewm_control import _padded_actions, _pooled_context_length


def test_pooled_control_aligns_h_actions_with_h_plus_one_frames() -> None:
    actions = torch.randn(2, 2, 15)

    padded = _padded_actions(actions)

    assert padded.shape == (2, 3, 15)
    assert torch.equal(padded[:, :2], actions)
    assert torch.count_nonzero(padded[:, -1]) == 0
    assert _pooled_context_length({"active_blocks": [[0.0], [0.0]]}) == 3
