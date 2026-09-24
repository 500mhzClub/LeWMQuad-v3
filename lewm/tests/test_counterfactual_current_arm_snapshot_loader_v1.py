from __future__ import annotations

import copy

import pytest


torch = pytest.importorskip("torch")

from scripts.dev_probe_counterfactual_action_fidelity import (  # noqa: E402
    load_predictor_arm_state_v1,
)


class _TinyTemporalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.predictor_position = torch.nn.Parameter(torch.zeros(1, 2))
        self.predictor_mask_token = torch.nn.Parameter(torch.zeros(1, 2))
        self.predictor_blocks = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.predictor_norm = torch.nn.LayerNorm(2)
        self.predictor_output = torch.nn.Linear(2, 2)
        self.action_embedding = torch.nn.Embedding(9, 2)
        self.time_embedding = torch.nn.Embedding(3, 2)
        self.temporal_gru = torch.nn.GRU(2, 2, batch_first=True)


def _payload(
    model: _TinyTemporalModel,
    *,
    schema: str = "lewm_go2_world_model_action_alignment_successor_v1_snapshot_v1",
) -> dict[str, object]:
    arm = {
        name: torch.ones_like(value, device="cpu")
        for name, value in model.state_dict().items()
        if name in {"predictor_position", "predictor_mask_token"}
        or name.startswith((
            "predictor_blocks.",
            "predictor_norm.",
            "predictor_output.",
            "action_embedding.",
            "time_embedding.",
            "temporal_gru.",
        ))
    }
    return {
        "schema": schema,
        "status": "COMPLETE",
        "arm": "alignment",
        "update": 900,
        "arm_state_dict": arm,
    }


@pytest.mark.parametrize(
    "schema",
    (
        "lewm_go2_world_model_action_alignment_successor_v1_snapshot_v1",
        "lewm_go2_world_model_progression_v1_snapshot_v1",
    ),
)
def test_current_arm_snapshot_loads_without_overwriting_frozen_encoder(
    schema: str,
) -> None:
    model = _TinyTemporalModel()
    encoder_before = {
        name: value.detach().clone()
        for name, value in model.encoder.state_dict().items()
    }
    identity = load_predictor_arm_state_v1(
        model, _payload(model, schema=schema), expected_update=900
    )
    assert identity["state_key"] == "arm_state_dict"
    assert identity["selected_arm"] == "alignment"
    assert all(
        torch.equal(value, encoder_before[name])
        for name, value in model.encoder.state_dict().items()
    )
    assert torch.equal(
        model.state_dict()["predictor_position"],
        torch.ones_like(model.state_dict()["predictor_position"]),
    )


def test_current_arm_snapshot_rejects_partial_inventory() -> None:
    model = _TinyTemporalModel()
    payload = copy.deepcopy(_payload(model))
    payload["arm_state_dict"].pop("predictor_position")
    with pytest.raises(ValueError, match="inventory"):
        load_predictor_arm_state_v1(model, payload, expected_update=900)
