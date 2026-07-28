from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts import (
    calibrate_go2_rgb_swept_progress_survival_joint_jepa_v4_physical_evidence as runner,
)


def _confident_logits(labels: torch.Tensor) -> torch.Tensor:
    logits = torch.full(
        (labels.shape[0], 3, *labels.shape[1:]),
        -8.0,
        dtype=torch.float32,
    )
    return logits.scatter_(1, labels[:, None], 8.0)


def test_real_calibration_and_full_threshold_grid_pass_on_separable_evidence() -> None:
    labels = torch.tensor(
        [
            [
                [0, 0, 1, 1],
                [0, 2, 1, 1],
                [2, 2, 1, 0],
                [0, 1, 2, 2],
            ],
            [
                [1, 0, 1, 2],
                [0, 2, 1, 1],
                [2, 1, 0, 0],
                [0, 1, 2, 1],
            ],
        ],
        dtype=torch.long,
    )
    science = runner._fit_select_score(
        _confident_logits(labels),
        labels,
        _confident_logits(labels),
        labels,
        provenance={"role": "probability_calibration", "fixture": "separable"},
    )
    artifact = science["calibration"]
    fit = artifact["fit"]
    fit_data = artifact["provenance"]["fit_data"]
    assert fit["sample_count"] == labels.numel()
    assert fit["balancing"] == fit["subsampling"] == "none"
    assert fit_data["masked_out_cell_count"] == 0
    assert fit_data["dropped_valid_cell_count"] == 0
    assert science["threshold_selection"]["candidate_count"] == 2_016
    assert science["threshold_selection"]["passing_candidate_count"] > 0
    assert science["gate"]["passed"] is True


class _SemanticOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("receipt", torch.tensor([1.0]))
        self.predictor_calls = 0

    def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        return torch.zeros((rgb.shape[0], 64, 64, 64), dtype=torch.float32)

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros((latent.shape[0], 3, 64, 64), dtype=torch.float32)

    def predict_all_actions_with_survival(self, latent: torch.Tensor) -> Any:
        self.predictor_calls += 1
        raise AssertionError("calibration must not call the predictor")


class _RecordingLoader:
    def __init__(self) -> None:
        self.requests: list[tuple[str, list[str]]] = []

    def endpoint_batch(
        self,
        endpoint_ids: list[str],
        device: torch.device,
        *,
        role: str,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.requests.append((role, list(endpoint_ids)))
        return (
            torch.zeros((len(endpoint_ids), 3, 112, 112), dtype=torch.float32),
            torch.zeros((len(endpoint_ids), 64, 64), dtype=torch.long),
        )


def test_collection_uses_ordered_next_endpoints_and_semantic_path_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    role = "probability_calibration"
    monkeypatch.setattr(runner, "ROLE_COUNTS", {role: 8})
    monkeypatch.setattr(runner, "ROLE_CELL_COUNTS", {role: 8 * 64 * 64})
    endpoint_ids = ["repeat", "repeat", "b", "c", "d", "e", "f", "g"]
    pairs = [
        {"scene_id": f"scene-{index}", "next_endpoint_sha256": endpoint}
        for index, endpoint in enumerate(endpoint_ids)
    ]
    loader = _RecordingLoader()
    model = _SemanticOnlyModel().eval().requires_grad_(False)
    logits, labels, receipt = runner._collect_role(
        model,
        loader,
        pairs,
        role=role,
        torch=torch,
        batch_size=3,
    )
    assert logits.shape == (8, 3, 64, 64)
    assert labels.shape == (8, 64, 64)
    assert [item for _, batch in loader.requests for item in batch] == endpoint_ids
    assert all(request_role == role for request_role, _ in loader.requests)
    assert receipt["batch_count"] == 3
    assert model.predictor_calls == 0


def _mock_science(*, passed: bool = True) -> dict[str, Any]:
    calibration = {
        "schema": "fixture_calibration",
        "content_sha256": "a" * 64,
        "id": "fixture-id",
    }
    checks = {"fixture_gate": passed}
    return {
        "calibration": calibration,
        "threshold_selection": {
            "candidate_count": 2_016,
            "passing_candidate_count": 1,
            "thresholds": {
                "free_probability_min": 0.9,
                "occupied_probability_max": 0.05,
                "unknown_probability_max": 0.05,
                "occupied_detection_min": 0.5,
            },
            "calibration_role_metrics": {},
        },
        "selection": {
            "calibration_metrics": {},
            "traversability": {},
            "physical_evidence": {},
        },
        "gate": {
            "status": (
                "PASS_DEVELOPMENT_PHYSICAL_EVIDENCE"
                if passed
                else "FAIL_DEVELOPMENT_PHYSICAL_EVIDENCE"
            ),
            "passed": passed,
            "checks": checks,
            "failed_checks": [] if passed else ["fixture_gate"],
        },
    }


class _MockInputs:
    def __init__(self) -> None:
        self.consumed = {
            "calibration-rgb": {
                "kind": "development_rgb",
                "roles": ["probability_calibration"],
            },
            "selection-rgb": {
                "kind": "development_rgb",
                "roles": ["checkpoint_selection"],
            },
        }

    def role_pairs(self, role: str) -> list[dict[str, str]]:
        return [{"role": role}]


class _MockAccessLoader:
    def receipt(self) -> dict[str, Any]:
        return {
            "raw_inputs_frame_attribute_invocation_count": 0,
            "forbidden_semantic_counters": {
                "general_raw_frame_loader_call_count": 0,
                "other_supervision_array_open_count": 0,
            },
        }

    def model_facing_access_counts(self) -> dict[str, int]:
        total = sum(runner.ROLE_COUNTS.values())
        return {
            "endpoint_rgb_row_request_count": total,
            "raster_label_row_request_count": total,
            "current_rgb_row_request_count": 0,
            "next_rgb_row_request_count": 0,
            "fixed_negative_rgb_row_request_count": 0,
        }


def _patch_successful_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    science: dict[str, Any],
) -> list[str]:
    monkeypatch.setattr(runner, "_validate_sources", lambda root: {"fixture": "hash"})
    monkeypatch.setattr(runner, "_load_candidate", lambda root, access: object())
    inputs = _MockInputs()
    loader = _MockAccessLoader()
    monkeypatch.setattr(
        runner,
        "_build_data_boundary",
        lambda root: (
            SimpleNamespace(torch=torch),
            inputs,
            loader,
            {"_raw_constructor_reads": {"fixture": {"read_success_count": 1}}},
        ),
    )
    roles: list[str] = []

    def collect(model: Any, loader: Any, pairs: Any, *, role: str, torch: Any):
        roles.append(role)
        receipt = {
            "role": role,
            "pair_count": runner.ROLE_COUNTS[role],
            "cell_count": runner.ROLE_CELL_COUNTS[role],
            "next_endpoint_order_sha256": role[0] * 64,
            "batch_count": 1,
            "model_state_mutated": False,
        }
        return torch.zeros((1, 3, 2, 2)), torch.zeros((1, 2, 2), dtype=torch.long), receipt

    monkeypatch.setattr(runner, "_collect_role", collect)
    def fit_select(*args: Any, **kwargs: Any) -> dict[str, Any]:
        counts = kwargs["operation_counts"]
        counts["calibration_fit_calls"] += 1
        counts["threshold_selection_calls"] += 1
        return science

    monkeypatch.setattr(runner, "_fit_select_score", fit_select)
    return roles


@pytest.mark.parametrize("passed", [True, False])
def test_execute_records_valid_science_without_selection_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed: bool,
) -> None:
    monkeypatch.setattr(runner, "OUTPUT_RELATIVE_PATH", "output/attempt_v1")
    roles = _patch_successful_boundaries(
        monkeypatch,
        science=_mock_science(passed=passed),
    )
    result = runner.execute(repository_root=tmp_path)
    output = tmp_path / "output/attempt_v1"
    assert roles == ["probability_calibration", "checkpoint_selection"]
    assert result["gate"]["passed"] is passed
    assert result["access"]["calibration_fit_calls"] == 1
    assert result["access"]["threshold_selection_calls"] == 1
    assert result["access"]["predictor_calls"] == 0
    assert (output / "calibration.json").is_file()
    assert (output / "result.json").is_file()
    assert not (output / "failure.json").exists()


def test_operational_candidate_failure_writes_only_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "OUTPUT_RELATIVE_PATH", "output/attempt_v1")
    monkeypatch.setattr(runner, "_validate_sources", lambda root: {})
    monkeypatch.setattr(
        runner,
        "_load_candidate",
        lambda root, access: (_ for _ in ()).throw(RuntimeError("candidate load failed")),
    )
    result = runner.execute(repository_root=tmp_path)
    output = tmp_path / "output/attempt_v1"
    assert result["status"] == "FAILED_OPERATIONALLY"
    assert result["stage"] == "loaded_candidate"
    assert (output / "failure.json").is_file()
    assert not (output / "calibration.json").exists()
    assert not (output / "result.json").exists()


def test_post_data_failure_preserves_raw_access_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "OUTPUT_RELATIVE_PATH", "output/attempt_v1")
    _patch_successful_boundaries(monkeypatch, science=_mock_science())
    monkeypatch.setattr(
        runner,
        "_fit_select_score",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    result = runner.execute(repository_root=tmp_path)
    raw_access = result["raw_access"]
    assert result["status"] == "FAILED_OPERATIONALLY"
    assert raw_access["loader_full_receipt"][
        "raw_inputs_frame_attribute_invocation_count"
    ] == 0
    assert raw_access["consumed_unique_file_count"] == 2
    assert len(raw_access["consumed_records_sha256"]) == 64
