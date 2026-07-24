from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_go2_rgb_causal_temporal_perception_v1.py"


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


def test_wrong_arm_replaces_complete_rgb_history_but_retains_target_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_temporal_v1_evaluator_history")
    family = "rough_local_dynamics"
    ids_by_family = {family: ["cold-target", "warm-target"]}
    predecessor_by_target = {
        "warm-target": {
            "family": family,
            "current_endpoint_sha256": "warm-predecessor",
        }
    }
    population = {
        "unique_endpoints": 2,
        "warm_endpoints": 1,
        "cold_endpoints": 1,
    }
    monkeypatch.setattr(
        module,
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
        assert arm == "causal_temporal_perception_v1"
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

    supervised_endpoint_batches: list[list[str]] = []

    def supervision(frames, device):
        assert device == torch.device("cpu")
        supervised_endpoint_batches.append(
            [item["endpoint_id"] for item in frames]
        )
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

        def forward_temporal_frame(
            self,
            previous_image,
            current_image,
            origin,
            basis,
            ground,
            history_valid,
        ):
            model_calls.append({
                "previous_image": previous_image.clone(),
                "current_image": current_image.clone(),
                "origin": origin.clone(),
                "basis": basis.clone(),
                "ground": ground.clone(),
                "history_valid": history_valid.clone(),
            })
            return SimpleNamespace(
                evidence=_BatchPayload(current_image.clone())
            )

    derive_calls: list[dict[str, torch.Tensor]] = []
    raster_calls: list[dict[str, torch.Tensor]] = []

    def derive_targets(**kwargs):
        derive_calls.append({
            name: value.clone() for name, value in kwargs.items()
        })
        return _BatchPayload(kwargs["pixel_hit_mask"].clone())

    def soft_rasterize(
        evidence,
        *,
        camera_origin_body_m,
        camera_basis_body_fru,
        pixel_ray_chunk_size,
    ):
        assert pixel_ray_chunk_size == 7
        raster_calls.append({
            "origin": camera_origin_body_m.clone(),
            "basis": camera_basis_body_fru.clone(),
        })
        return _BatchPayload(evidence.tensor.clone())

    runtime = SimpleNamespace(
        torch=torch,
        MetricAccumulator=_Accumulator,
        derive_targets=derive_targets,
        soft_rasterize=soft_rasterize,
        loss_adapter=SimpleNamespace(
            observable_camera_ray_v4_loss_v4=(
                lambda *args, **kwargs: SimpleNamespace(
                    total=torch.tensor(0.25)
                )
            )
        ),
    )
    trainer = SimpleNamespace(
        inputs=SimpleNamespace(frame=frame),
        supervision=supervision,
        _single_frame_pair=lambda online: online,
        _flatten_physical=lambda correct, wrong: {
            "correct": correct,
            "wrong": wrong,
        },
    )

    _metrics, _warm_metrics, loss, observed_population = (
        module._temporal_physical_metrics(
            runtime,
            trainer,
            Model(),
            (),
            torch.device("cpu"),
            arm="causal_temporal_perception_v1",
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

    target_origin = torch.stack([
        frame(
            "cold-target",
            role="checkpoint_selection",
            arm="causal_temporal_perception_v1",
            stage="synthetic_integrity",
        )["camera_origin"],
        frame(
            "warm-target",
            role="checkpoint_selection",
            arm="causal_temporal_perception_v1",
            stage="synthetic_integrity",
        )["camera_origin"],
    ])
    target_basis = torch.stack([
        torch.full((3, 3), 210.0),
        torch.full((3, 3), 220.0),
    ])
    target_ground = torch.tensor([310.0, 320.0])
    for call in model_calls:
        assert torch.equal(call["origin"], target_origin)
        assert torch.equal(call["basis"], target_basis)
        assert torch.equal(call["ground"], target_ground)
        assert call["history_valid"].tolist() == [False, True]
    assert wrong["current_image"][1].item() == codes["cold-target"]
    assert bool(wrong["history_valid"][1]) is True

    assert supervised_endpoint_batches == [
        ["cold-target", "warm-target"]
    ]
    assert len(derive_calls) == 1
    assert derive_calls[0]["pixel_hit_mask"].tolist() == [10.0, 20.0]
    assert len(raster_calls) == 2
    assert all(
        torch.equal(call["origin"], target_origin)
        and torch.equal(call["basis"], target_basis)
        for call in raster_calls
    )


def test_warm_only_metrics_cannot_change_checkpoint_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_temporal_v1_evaluator_warm_control")
    primary = {"primary": {"fixed": True}}
    warm_holder = {"value": {"warm": {"score": -1_000_000.0}}}
    evaluation = {
        "complete_physical_scope_count": 0,
        "margin_count": module.contract.MARGIN_COUNT,
        "passed_margin_count": 0,
        "total_shortfall": 100.0,
        "worst_margin": -1.0,
        "rough_motion": {
            "pixel_balanced_accuracy": 0.0,
            "ground_balanced_accuracy": 0.0,
            "depth_p95_m": 2.0,
        },
    }
    evaluated_inputs: list[object] = []

    def evaluate_physical_scopes(scopes):
        evaluated_inputs.append(scopes)
        return dict(evaluation)

    def temporal_metrics(*args, **kwargs):
        return (
            primary,
            warm_holder["value"],
            0.5,
            {"unique_endpoints": 2, "warm_endpoints": 1},
        )

    monkeypatch.setattr(
        module.contract,
        "evaluate_physical_scopes",
        evaluate_physical_scopes,
    )
    monkeypatch.setattr(
        module, "_temporal_physical_metrics", temporal_metrics
    )
    monkeypatch.setattr(module, "_state_sha", lambda *args: "state")
    monkeypatch.setattr(
        module, "_subset_sha", lambda *args: "frozen-state"
    )

    class Model:
        def eval(self):
            return self

        def train(self):
            return self

    def run_once():
        metric = module._evaluate(
            SimpleNamespace(),
            SimpleNamespace(),
            Model(),
            (),
            torch.device("cpu"),
            update=1_000,
            frozen_sha256="frozen-state",
        )
        control = module.contract.checkpoint_control_decision(
            update=1_000,
            evaluation=metric["evaluation"],
            integrity_pass=metric["integrity_pass"],
        )
        return metric, control

    first_metric, first_control = run_once()
    warm_holder["value"] = {"warm": {"score": 1_000_000.0}}
    second_metric, second_control = run_once()

    assert evaluated_inputs == [primary, primary]
    assert (
        first_metric["warm_scopes_informational_only"]
        != second_metric["warm_scopes_informational_only"]
    )
    assert first_metric["evaluation"] == second_metric["evaluation"]
    assert first_control == second_control
    assert first_control["action"] == module.contract.CONTROL_FAIL
