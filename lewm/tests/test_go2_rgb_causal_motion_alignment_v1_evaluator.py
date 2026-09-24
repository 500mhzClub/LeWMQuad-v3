from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_go2_rgb_causal_motion_alignment_v1.py"


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass(frozen=True)
class _BatchPayload:
    tensor: torch.Tensor


class _Accumulator:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)

    def finalize(self) -> dict[str, int]:
        return {"update_count": len(self.updates)}


def test_wrong_rgb_retains_target_incoming_motion_and_cold_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_motion_alignment_evaluator")
    family = "rough_local_dynamics"
    ids_by_family = {family: ["cold-target", "warm-target"]}
    predecessor_by_target = {
        "warm-target": {
            "family": family,
            "current_endpoint_sha256": "warm-predecessor",
            "next_endpoint_sha256": "warm-target",
            "primitive": "yaw_left",
        }
    }
    population = {
        "unique_endpoints": 2,
        "warm_endpoints": 1,
        "cold_endpoints": 1,
    }
    monkeypatch.setattr(
        module._BASE,
        "_selection_temporal_index",
        lambda _pairs: (
            ids_by_family,
            predecessor_by_target,
            population,
        ),
    )
    role_counts = dict(module.contract.SELECTION_ROLE_COUNTS)
    role_counts["unique_endpoints"] = 2
    role_counts["warm_endpoints"] = 1
    monkeypatch.setattr(
        module.contract, "SELECTION_ROLE_COUNTS", role_counts
    )
    monkeypatch.setattr(module.contract, "MICROBATCH_SIZE", 2)

    codes = {
        "cold-target": 10.0,
        "warm-target": 20.0,
        "warm-predecessor": 30.0,
    }

    def frame(endpoint_id: str, *, role: str, arm: str, stage: str):
        assert role == "checkpoint_selection"
        assert arm == "causal_motion_alignment_v1"
        assert stage == "synthetic_integrity"
        code = codes[endpoint_id]
        return {
            "endpoint_id": endpoint_id,
            "image": torch.tensor([code]),
            "camera_origin": torch.full((3,), code + 100.0),
            "camera_basis": torch.full((3, 3), code + 200.0),
            "ground": torch.tensor(code + 300.0),
            "pixel_hit": torch.tensor(code),
            "pixel_distance": torch.tensor(code + 1.0),
            "ground_in_frustum": torch.tensor(code + 2.0),
            "ground_clear": torch.tensor(code + 3.0),
            "raster_label": torch.tensor(int(code)),
        }

    def supervision(frames, device):
        assert device == torch.device("cpu")
        stack = lambda name: torch.stack([item[name] for item in frames])
        return SimpleNamespace(
            pixel_hit_mask=stack("pixel_hit"),
            pixel_first_hit_distance_m=stack("pixel_distance"),
            ground_support_in_frustum=stack("ground_in_frustum"),
            ground_support_clear_to_target=stack("ground_clear"),
            target_raster_labels=stack("raster_label"),
        )

    model_calls: list[dict[str, torch.Tensor]] = []

    class Model:
        model_config = SimpleNamespace(v4_pixel_ray_chunk_size=7)

        def forward_temporal_frame(self, **kwargs):
            model_calls.append({
                name: value.clone()
                for name, value in kwargs.items()
                if isinstance(value, torch.Tensor)
            })
            return SimpleNamespace(
                evidence=_BatchPayload(kwargs["current_image"].clone())
            )

    runtime = SimpleNamespace(
        torch=torch,
        MetricAccumulator=_Accumulator,
        derive_targets=lambda **kwargs: _BatchPayload(
            kwargs["pixel_hit_mask"].clone()
        ),
        soft_rasterize=lambda evidence, **_kwargs: _BatchPayload(
            evidence.tensor.clone()
        ),
        loss_adapter=SimpleNamespace(
            observable_camera_ray_v4_loss_v4=(
                lambda *args, **kwargs: SimpleNamespace(
                    total=torch.tensor(0.25)
                )
            )
        ),
    )
    vocabulary = module.PRIMITIVE_VOCABULARY
    table = torch.stack([
        torch.tensor([float(index), float(index) + 0.1, float(index) + 0.2])
        for index in range(len(vocabulary))
    ])
    trainer = SimpleNamespace(
        inputs=SimpleNamespace(frame=frame),
        supervision=supervision,
        _single_frame_pair=lambda online: online,
        _flatten_physical=lambda correct, wrong: {
            "correct": correct,
            "wrong": wrong,
        },
        _motion_primitive_to_index={
            primitive: index for index, primitive in enumerate(vocabulary)
        },
        _motion_nominal_table=table,
    )

    _metrics, _warm, loss, observed_population = (
        module._motion_physical_metrics(
            runtime,
            trainer,
            Model(),
            (),
            torch.device("cpu"),
            arm="causal_motion_alignment_v1",
            stage="synthetic_integrity",
        )
    )
    assert observed_population is population
    assert loss == pytest.approx(0.25)
    assert len(model_calls) == 2
    correct, wrong = model_calls
    assert correct["previous_image"].flatten().tolist() == [10.0, 30.0]
    assert correct["current_image"].flatten().tolist() == [10.0, 20.0]
    assert wrong["previous_image"].flatten().tolist() == [30.0, 10.0]
    assert wrong["current_image"].flatten().tolist() == [20.0, 10.0]

    expected_nominal = torch.stack([
        torch.zeros(3),
        table[vocabulary.index("yaw_left")],
    ])
    expected_previous_basis = torch.stack([
        torch.full((3, 3), 210.0),
        torch.full((3, 3), 230.0),
    ])
    expected_current_basis = torch.stack([
        torch.full((3, 3), 210.0),
        torch.full((3, 3), 220.0),
    ])
    for call in model_calls:
        assert torch.equal(
            call["nominal_delta_current_frame"], expected_nominal
        )
        assert torch.equal(
            call["previous_camera_basis_body_fru"],
            expected_previous_basis,
        )
        assert torch.equal(
            call["target_camera_basis_body_fru"],
            expected_current_basis,
        )
        assert call["history_valid"].tolist() == [False, True]

    # The wrong arm rotates visual packets, but it does not use the mapped
    # endpoint's primitive or attitude as the target condition.
    assert torch.equal(
        correct["nominal_delta_current_frame"],
        wrong["nominal_delta_current_frame"],
    )
    assert torch.equal(
        correct["previous_camera_basis_body_fru"],
        wrong["previous_camera_basis_body_fru"],
    )
