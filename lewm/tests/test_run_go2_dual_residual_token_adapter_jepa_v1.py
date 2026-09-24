from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lewm.models.go2_dual_residual_token_adapter_jepa_v1 import (
    JointResidualTokenAdapterJEPAV1,
)
from scripts import run_go2_dual_residual_token_adapter_jepa_v1 as dual


ARMS = ("residual_joint_vjepa2_1", "residual_joint_dinov2")


def _metrics(
    *,
    ratio: float = 0.79,
    retrieval: float = 0.51,
    margin: float = 0.02,
    deterministic: bool = True,
    nonfinite_count: int = 0,
) -> dict[str, float | bool | int]:
    persistence = 0.1
    matched = persistence * ratio
    return {
        "matched_cosine_error": matched,
        "persistence_cosine_error": persistence,
        "error_to_persistence_ratio": ratio,
        "branch_retrieval_accuracy": retrieval,
        "cyclic_deranged_cosine_error": matched + margin,
        "action_intervention_margin": margin,
        "deterministic_repeat_passed": deterministic,
        "nonfinite_count": nonfinite_count,
    }


def _retention(
    *,
    cosine: float = 0.97,
    rank_ratio: float = 0.91,
    deterministic: bool = True,
) -> dict[str, float | bool]:
    return {
        "mean_online_to_frozen_token_cosine": cosine,
        "effective_rank_ratio": rank_ratio,
        "all_tokens_finite": True,
        "online_unit_norm_passed": True,
        "ema_unit_norm_passed": True,
        "deterministic_repeat_passed": deterministic,
        "retention_passed": (
            cosine >= 0.965 and rank_ratio >= 0.90 and deterministic
        ),
    }


def _index(*, count: int = 1_536, index_sha256: str = "0" * 64) -> object:
    contexts: list[tuple[int, int, int]] = []
    targets: list[tuple[int, ...]] = []
    histories: list[tuple[int, int]] = []
    for state in range(128):
        offset = state * 12
        contexts.append((offset, offset + 1, offset + 2))
        targets.append(tuple(range(offset + 3, offset + 12)))
        histories.append((state % 9, (state + 1) % 9))
    return dual.predecessor.ScreenIndexV1(
        state_ids=tuple(f"state-{state}" for state in range(128)),
        family_ids=tuple(f"family-{state % 8}" for state in range(128)),
        scene_ids=tuple(f"scene-{state // 8}" for state in range(128)),
        artifact_ids=tuple(f"artifact-{item}" for item in range(count)),
        context_indices=torch.tensor(contexts, dtype=torch.long),
        target_indices=torch.tensor(targets, dtype=torch.long),
        history_actions=torch.tensor(histories, dtype=torch.long),
        index_sha256=index_sha256,
    )


def test_config_freezes_both_arms_mechanism_optimizer_and_gates() -> None:
    config = dual.dual_config_v1()

    assert config["arms"] == list(ARMS)
    assert config["seed"] == 2_026_080_301
    assert config["action_count"] == 9
    assert config["batch_states"] == 8
    assert config["hidden_dim"] == 128
    assert config["adapter_blocks"] == 2
    assert config["adapter_bottleneck"] == 64
    assert config["adapter_residual_scale"] == 0.125
    assert config["ema_momentum"] == 0.996
    assert config["minimum_updates"] == 800
    assert config["maximum_updates"] == 1_600
    assert config["trace_updates"] == [0, 400, 800, 1_600]

    assert config["learning_rate"] == 3.0e-4
    assert config["weight_decay"] == 1.0e-4
    assert tuple(config["adamw_betas"]) == (0.9, 0.999)
    assert config["adamw_eps"] == 1.0e-8
    assert config["adamw_amsgrad"] is False
    assert config["adamw_maximize"] is False
    assert config["adamw_foreach"] is False
    assert config["adamw_capturable"] is False
    assert config["adamw_differentiable"] is False
    assert config["adamw_fused"] is False
    assert config["gradient_clip_norm"] == 1.0

    assert config["ema_target_coefficient"] == 0.5
    assert config["frozen_target_coefficient"] == 0.5
    assert config["identity_coefficient"] == 0.10
    assert config["relative_variance_coefficient"] == 0.10
    assert config["relative_variance_floor"] == 0.90
    assert config["cross_entropy_coefficient"] == 0.25
    assert config["cross_entropy_temperature"] == 0.1

    assert config["maximum_error_to_persistence_ratio"] == 0.80
    assert config["minimum_branch_retrieval_accuracy"] == 0.50
    assert config["minimum_retention_cosine"] == 0.965
    assert config["minimum_effective_rank_ratio"] == 0.90
    assert config["unit_norm_absolute_tolerance"] == 1.0e-5
    assert config["cache_compute_dtype"] == "float32"
    assert config["autocast_enabled"] is False
    assert config["midpoint_gates"] == {
        "residual_joint_vjepa2_1": {
            "maximum_error_to_persistence_ratio": 0.8582181769526677,
            "minimum_branch_retrieval_accuracy": 0.3901909722222222,
        },
        "residual_joint_dinov2": {
            "maximum_error_to_persistence_ratio": 0.8833446296789655,
            "minimum_branch_retrieval_accuracy": 0.3524305555555556,
        },
    }


def test_effective_rank_uses_centered_population_covariance_entropy() -> None:
    isotropic = torch.tensor(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    )
    collinear = torch.tensor(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))

    assert dual.effective_rank_v1(isotropic) == pytest.approx(2.0, abs=1.0e-12)
    assert dual.effective_rank_v1(7.0 * isotropic + 13.0) == pytest.approx(
        2.0, abs=1.0e-12
    )
    assert dual.effective_rank_v1(collinear) == pytest.approx(1.0, abs=1.0e-12)

    with pytest.raises(dual.DualScreenError, match="effective-rank|effective rank"):
        dual.effective_rank_v1(torch.ones(4, 3))


def test_initial_retention_is_identity_like_finite_and_repeatable(
) -> None:
    torch.manual_seed(41)
    features = F.normalize(torch.randn(16, 256, 8), dim=-1).to(torch.float16)
    model = JointResidualTokenAdapterJEPAV1(feature_dim=8)

    first = dual.retention_metrics_v1(
        model, features, device=torch.device("cpu"), batch_artifacts=4
    )
    second = dual.retention_metrics_v1(
        model, features, device=torch.device("cpu"), batch_artifacts=4
    )

    assert first == second
    assert first["mean_online_to_frozen_token_cosine"] == pytest.approx(
        1.0, abs=2.0e-6
    )
    assert first["effective_rank_ratio"] == pytest.approx(
        1.0, rel=2.0e-3
    )
    assert first["all_tokens_finite"] is True
    assert first["online_unit_norm_passed"] is True
    assert first["ema_unit_norm_passed"] is True


def test_finite_adapted_rank_collapse_is_arm_qualification_failure() -> None:
    class _CollapsedAdapter(torch.nn.Module):
        def eval(self) -> "_CollapsedAdapter":
            return self

        def adapt_online(self, tokens: torch.Tensor) -> torch.Tensor:
            collapsed = torch.zeros_like(tokens)
            collapsed[..., 0] = 1.0
            return collapsed

        def adapt_target(self, tokens: torch.Tensor) -> torch.Tensor:
            return F.normalize(tokens, dim=-1)

    features = F.normalize(torch.randn(16, 256, 8), dim=-1).to(torch.float16)
    retention = dual.retention_metrics_v1(
        _CollapsedAdapter(),
        features,
        device=torch.device("cpu"),
        batch_artifacts=4,
    )

    assert retention["adapted_effective_rank"] == 0.0
    assert retention["effective_rank_ratio"] == 0.0
    assert retention["retention_passed"] is False


def test_nonfinite_rank_spectrum_is_arm_local_numerical_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dual.torch.linalg,
        "eigvalsh",
        lambda covariance: torch.full(
            (covariance.shape[0],), float("nan"), dtype=torch.float64
        ),
    )
    with pytest.raises(dual.ArmNonfiniteError, match="nonfinite"):
        dual.effective_rank_v1(torch.eye(3, dtype=torch.float64))


def test_frozen_evaluator_uses_frozen_last_context_for_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dual.predecessor, "STATE_COUNT", 1)
    width = 4
    frozen = torch.zeros(12, 256, width, dtype=torch.float16)
    frozen[:, :, 0] = 1.0
    index = dual.predecessor.ScreenIndexV1(
        state_ids=("state",),
        family_ids=("family",),
        scene_ids=("scene",),
        artifact_ids=tuple(f"artifact-{item}" for item in range(12)),
        context_indices=torch.tensor(((0, 1, 2),), dtype=torch.long),
        target_indices=torch.tensor((tuple(range(3, 12)),), dtype=torch.long),
        history_actions=torch.tensor(((0, 1),), dtype=torch.long),
        index_sha256="2" * 64,
    )

    class _DeliberatelyRotatedContext(torch.nn.Module):
        def adapt_online(self, tokens: torch.Tensor) -> torch.Tensor:
            rotated = torch.zeros_like(tokens)
            rotated[..., 1] = 1.0
            return rotated

        def predict_from_adapted_context(
            self,
            adapted_context: torch.Tensor,
            history_actions: torch.Tensor,
            candidate_action: torch.Tensor,
        ) -> torch.Tensor:
            del history_actions, candidate_action
            prediction = torch.zeros(
                adapted_context.shape[0], 256, width, dtype=torch.float32
            )
            prediction[..., 0] = 1.0
            return prediction

    metrics = dual.evaluate_arm_v1(
        _DeliberatelyRotatedContext(),
        frozen,
        index,
        device=torch.device("cpu"),
        batch_states=1,
    )

    # Frozen last context and every frozen target are identical.  Using the
    # deliberately rotated adapted context would instead produce error 1.0.
    assert metrics["persistence_cosine_error"] == pytest.approx(0.0, abs=1.0e-7)


def test_bound_input_loader_reuses_both_caches_without_rgb_or_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dual.predecessor, "ARTIFACT_COUNT", 2)
    cache_bindings = {
        "vjepa2_1": {"token": "vjepa-cache"},
        "dinov2": {"token": "dino-cache"},
    }
    receipts = {
        encoder: {
            "binding": cache_bindings[encoder],
            "index_sha256": dual.INDEX_SHA256,
            "artifact_order_sha256": dual.ARTIFACT_ORDER_SHA256,
            "train_artifact_open_count": 2,
            "eval_artifact_open_count": 0,
        }
        for encoder in ("vjepa2_1", "dinov2")
    }
    symbolic = {
        label: {"token": label}
        for label in (
            "screen_result",
            "screen_terminal",
            "screen_terminal_review",
            "horizon_result",
            "horizon_terminal",
            "horizon_terminal_review",
            "vjepa2_1_feature_receipt",
            "dinov2_feature_receipt",
        )
    }
    symbolic["vjepa2_1_feature_cache"] = cache_bindings["vjepa2_1"]
    symbolic["dinov2_feature_cache"] = cache_bindings["dinov2"]
    documents = {
        "screen result": {
            "schema": dual.predecessor.SCHEMA,
            "status": "COMPLETE_ENGINEERING_SCREEN",
            "collection_justified": False,
            "navigation_usefulness_established": False,
            "screen_index": {
                "index_sha256": dual.INDEX_SHA256,
                "eval_rgb_leaf_open_count": 0,
            },
            "feature_caches": receipts,
        },
        "screen terminal": {
            "schema": dual.predecessor.TERMINAL_SCHEMA,
            "status": "COMPLETE_COLLECTION_NOT_JUSTIFIED",
            "result_binding": symbolic["screen_result"],
        },
        "screen terminal review": {
            "schema": "lewm_go2_matched_branch_successor_screen_terminal_review_v1",
            "result_binding": symbolic["screen_result"],
            "terminal_binding": symbolic["screen_terminal"],
            "protected_material_opened": False,
            "evaluation_rgb_opened": False,
            "findings": [],
        },
        "horizon result": {
            "schema": "lewm_go2_dense_vjepa2_1_horizon_diagnostic_result_v1",
            "status": "COMPLETE_FUTILITY_STOP",
            "training_set_capacity_established": False,
            "collection_justified": False,
        },
        "horizon terminal": {
            "schema": "lewm_go2_dense_vjepa2_1_horizon_diagnostic_terminal_v1",
            "status": "COMPLETE_FUTILITY_STOP",
            "result_binding": symbolic["horizon_result"],
        },
        "horizon terminal review": {
            "schema": "lewm_go2_dense_vjepa2_1_horizon_diagnostic_terminal_review_v1",
            "status": "PASS_COMPLETE_FUTILITY_STOP_TERMINAL_REVIEW",
            "result_binding": symbolic["horizon_result"],
            "terminal_binding": symbolic["horizon_terminal"],
            "protected_material_opened": False,
            "findings": [],
        },
        "vjepa2_1 receipt": receipts["vjepa2_1"],
        "dinov2 receipt": receipts["dinov2"],
    }
    monkeypatch.setattr(
        dual,
        "_json_from_bound_file",
        lambda _binding, *, label: documents[label],
    )
    monkeypatch.setattr(
        dual, "_validate_embedded_source_closures", lambda *_args: None
    )
    index = dual.predecessor.ScreenIndexV1(
        state_ids=(),
        family_ids=(),
        scene_ids=(),
        artifact_ids=("artifact-0", "artifact-1"),
        context_indices=torch.empty((0, 3), dtype=torch.long),
        target_indices=torch.empty((0, 9), dtype=torch.long),
        history_actions=torch.empty((0, 2), dtype=torch.long),
        index_sha256=dual.INDEX_SHA256,
    )
    bundle = SimpleNamespace(access_audit={"rgb_leaf_open_count": 0})
    monkeypatch.setattr(
        dual.predecessor.screen_data,
        "load_bound_posthoc_bundle_v1",
        lambda: bundle,
    )
    monkeypatch.setattr(
        dual.predecessor, "build_screen_index_v1", lambda _bundle: index
    )
    features = {
        "vjepa2_1": F.normalize(torch.ones(2, 256, 768), dim=-1).to(torch.float16),
        "dinov2": F.normalize(torch.ones(2, 256, 384), dim=-1).to(torch.float16),
    }
    cache_calls: list[str] = []

    def fake_cache(
        _receipt: object, *, expected_encoder: str, index: object
    ) -> torch.Tensor:
        assert index is not None
        cache_calls.append(expected_encoder)
        return features[expected_encoder]

    monkeypatch.setattr(dual.predecessor, "_load_feature_cache", fake_cache)
    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # noqa: E731
        AssertionError("RGB/extraction path called")
    )
    monkeypatch.setattr(dual.predecessor, "read_bound_rgb_bytes_v1", forbidden)
    monkeypatch.setattr(dual.predecessor, "extract_feature_cache_v1", forbidden)

    loaded, loaded_index, loaded_result = dual.load_bound_inputs_v1(
        {"predecessor_bindings": symbolic, "source_bindings": {}}
    )

    assert cache_calls == ["vjepa2_1", "dinov2"]
    assert loaded["vjepa2_1"] is features["vjepa2_1"]
    assert loaded["dinov2"] is features["dinov2"]
    assert loaded_index is index
    assert loaded_result is documents["screen result"]


@pytest.mark.parametrize(
    ("metrics", "retention", "movement"),
    (
        (_metrics(ratio=0.8000000000000001), _retention(), 0.1),
        (_metrics(retrieval=0.49999999999999994), _retention(), 0.1),
        (_metrics(margin=0.0), _retention(), 0.1),
        (_metrics(), _retention(cosine=0.9649999999999999), 0.1),
        (_metrics(), _retention(rank_ratio=0.8999999999999999), 0.1),
        (_metrics(), _retention(deterministic=False), 0.1),
        (_metrics(), _retention(), 0.0),
    ),
)
def test_capacity_gate_is_conjunctive_and_uses_closed_boundaries(
    metrics: dict[str, float | bool | int],
    retention: dict[str, float | bool],
    movement: float,
) -> None:
    assert not dual._capacity_passed_v1(metrics, retention, movement)  # noqa: SLF001

    boundary = _metrics(ratio=0.80, retrieval=0.50, margin=1.0e-12)
    assert dual._capacity_passed_v1(  # noqa: SLF001
        boundary, _retention(cosine=0.965, rank_ratio=0.90), 1.0e-12
    )


@pytest.mark.parametrize(
    ("arm", "ratio", "retrieval"),
    (
        ("residual_joint_vjepa2_1", 0.8582181769526677, 0.3901909722222222),
        ("residual_joint_dinov2", 0.8833446296789655, 0.3524305555555556),
    ),
)
def test_update_800_continuation_uses_exact_midpoints_and_strict_progress(
    arm: str, ratio: float, retrieval: float
) -> None:
    update_400 = _metrics(ratio=ratio + 0.01, retrieval=retrieval - 0.01)
    update_800 = _metrics(ratio=ratio, retrieval=retrieval)
    assert dual._may_continue_v1(  # noqa: SLF001
        arm, update_400, update_800, _retention(), 1.0e-12
    )

    no_ratio_progress = _metrics(ratio=ratio, retrieval=retrieval - 0.01)
    no_retrieval_progress = _metrics(ratio=ratio + 0.01, retrieval=retrieval)
    assert not dual._may_continue_v1(  # noqa: SLF001
        arm, no_ratio_progress, update_800, _retention(), 0.1
    )
    assert not dual._may_continue_v1(  # noqa: SLF001
        arm, no_retrieval_progress, update_800, _retention(), 0.1
    )
    assert not dual._may_continue_v1(  # noqa: SLF001
        arm, update_400, update_800, _retention(cosine=0.96), 0.1
    )
    assert not dual._may_continue_v1(  # noqa: SLF001
        arm, update_400, _metrics(ratio=ratio, retrieval=retrieval, margin=0.0),
        _retention(), 0.1
    )
    assert not dual._may_continue_v1(  # noqa: SLF001
        arm, update_400, update_800, _retention(), 0.0
    )


def test_authority_rejects_changed_caller_binding(tmp_path: Path) -> None:
    authority = tmp_path / "authority.json"
    authority.write_text("{}\n")
    with pytest.raises(dual.DualScreenError, match="caller binding"):
        dual._read_authority(  # noqa: SLF001
            authority,
            expected_sha256="0" * 64,
            expected_byte_count=3,
        )


def test_train_arm_converts_nonfinite_loss_to_arm_local_terminal_without_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = dual.dual_config_v1()
    config["maximum_updates"] = 1
    features = F.normalize(torch.randn(1_536, 256, 4), dim=-1).to(torch.float16)
    monkeypatch.setattr(
        dual,
        "evaluate_arm_v1",
        lambda *_args, **_kwargs: _metrics(
            ratio=1.0, retrieval=1.0 / 9.0, margin=0.0
        ),
    )

    def nonfinite_objective(
        *_args: object, **_kwargs: object
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        value = torch.tensor(float("nan"), requires_grad=True)
        return value, {"matched": value, "contrastive": value}

    monkeypatch.setattr(
        dual.predecessor, "common_objective_v1", nonfinite_objective
    )

    result = dual.train_arm_v1(
        ARMS[1],
        features,
        _index(),
        config=config,
        device=torch.device("cpu"),
        output_root=tmp_path,
    )

    assert result["status"] == "COMPLETE_NONFINITE_CAPACITY_NOT_ESTABLISHED"
    assert result["completed_updates"] == 0
    assert result["nonfinite_count"] == 1
    assert result["capacity_established"] is False
    assert result["checkpoint_bindings"] == {}
    assert result["deterministic_repeat_passed"] is None
    assert result["retention_repeat_passed"] is None
    assert result["execution_witness_passed"] is None


def test_zero_update_400_movement_still_reaches_update_800_before_qualification_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TinyJoint(torch.nn.Module):
        def __init__(self, *, feature_dim: int) -> None:
            super().__init__()
            self.feature_dim = feature_dim
            self.online_adapter = torch.nn.Linear(1, 1, bias=False)
            self.target_adapter = torch.nn.Linear(1, 1, bias=False)
            self.target_adapter.requires_grad_(False)

        def train(self, mode: bool = True) -> "_TinyJoint":
            super().train(mode)
            self.target_adapter.eval()
            return self

        def adapt_online(self, tokens: torch.Tensor) -> torch.Tensor:
            return F.normalize(tokens, dim=-1)

        @torch.no_grad()
        def adapt_target(self, tokens: torch.Tensor) -> torch.Tensor:
            return F.normalize(tokens, dim=-1)

        def predict_from_adapted_context(
            self,
            adapted_context: torch.Tensor,
            history_actions: torch.Tensor,
            candidate_action: torch.Tensor,
        ) -> torch.Tensor:
            del history_actions, candidate_action
            return adapted_context[:, -1] + self.online_adapter.weight.reshape(1, 1, 1)

        @torch.no_grad()
        def update_target_ema_(self, momentum: float) -> None:
            self.target_adapter.weight.mul_(momentum).add_(
                self.online_adapter.weight, alpha=1.0 - momentum
            )

    config = dual.dual_config_v1()
    config["maximum_updates"] = 800
    fixed_context = F.normalize(torch.randn(8, 3, 256, 2), dim=-1)
    fixed_targets = F.normalize(torch.randn(8, 9, 256, 2), dim=-1)
    fixed_history = torch.zeros(8, 2, dtype=torch.long)
    monkeypatch.setattr(dual, "JointResidualTokenAdapterJEPAV1", _TinyJoint)
    monkeypatch.setattr(
        dual,
        "_unique_batch_v1",
        lambda *_args, **_kwargs: (fixed_context, fixed_targets, fixed_history),
    )
    monkeypatch.setattr(
        dual,
        "evaluate_arm_v1",
        lambda *_args, **_kwargs: _metrics(
            ratio=0.90, retrieval=0.40, margin=0.02
        ),
    )
    monkeypatch.setattr(
        dual,
        "retention_metrics_v1",
        lambda *_args, **_kwargs: _retention(),
    )
    monkeypatch.setattr(dual, "_adapter_movement_v1", lambda *_args: 0.0)
    monkeypatch.setattr(
        dual,
        "_checkpoint_v1",
        lambda *_args, **_kwargs: {
            "path": "/synthetic-checkpoint",
            "sha256": "9" * 64,
            "byte_count": 1,
        },
    )

    def differentiable_objective(
        predictions: torch.Tensor,
        _targets: torch.Tensor,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        value = predictions.mean()
        return value, {"matched": value, "contrastive": value}

    monkeypatch.setattr(
        dual.predecessor, "common_objective_v1", differentiable_objective
    )
    result = dual.train_arm_v1(
        ARMS[1],
        torch.ones(1, 256, 2, dtype=torch.float16),
        _index(),
        config=config,
        device=torch.device("cpu"),
        output_root=tmp_path,
    )

    assert result["completed_updates"] == 800
    assert result["status"] == (
        "COMPLETE_QUALIFICATION_FAILURE_CAPACITY_NOT_ESTABLISHED"
    )
    assert result["execution_witness_passed"] is False
    assert set(result["checkpoint_bindings"]) == {"update_800"}


def test_checkpoint_roundtrip_rejects_finite_tensor_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = JointResidualTokenAdapterJEPAV1(feature_dim=4)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    initial = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.online_adapter.state_dict().items()
    }
    original_load = dual.torch.load

    def corrupted_load(*args: object, **kwargs: object) -> dict[str, object]:
        payload = original_load(*args, **kwargs)
        state = payload["model_state_dict"]
        assert isinstance(state, dict)
        tensor = next(
            value
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        )
        assert isinstance(tensor, torch.Tensor)
        tensor.reshape(-1)[0] += 1.0
        return payload

    monkeypatch.setattr(dual.torch, "load", corrupted_load)
    with pytest.raises(dual.DualScreenError, match="round-trip"):
        dual._checkpoint_v1(  # noqa: SLF001
            tmp_path / "corrupt.pt",
            arm=ARMS[1],
            model=model,
            optimizer=optimizer,
            initial_online_adapter_state=initial,
            movement=0.0,
            update=800,
            config=dual.dual_config_v1(),
        )


def _arm_result(
    arm: str,
    *,
    status: str,
    capacity: bool,
    checkpoint_updates: tuple[int, ...] = (800,),
) -> dict[str, object]:
    return {
        "arm": arm,
        "status": status,
        "completed_updates": checkpoint_updates[-1],
        "capacity_established": capacity,
        "nonfinite_count": int(status == "COMPLETE_NONFINITE_CAPACITY_NOT_ESTABLISHED"),
        "deterministic_repeat_passed": True,
        "traces": [],
        "checkpoint_bindings": {
            f"update_{update}": {
                "path": f"/{arm}-{update}.pt",
                "sha256": str(update).zfill(64),
                "byte_count": update,
            }
            for update in checkpoint_updates
        },
    }


@pytest.mark.parametrize(
    "first_status",
    (
        "COMPLETE_UPDATE_800_FUTILITY_STOP",
        "COMPLETE_NONFINITE_CAPACITY_NOT_ESTABLISHED",
        "COMPLETE_QUALIFICATION_FAILURE_CAPACITY_NOT_ESTABLISHED",
    ),
)
def test_execute_launches_second_arm_after_first_arm_scientific_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_status: str,
) -> None:
    output = tmp_path / "attempt"
    index = SimpleNamespace(
        index_sha256="1" * 64,
        state_ids=tuple(range(128)),
        scene_ids=tuple(f"scene-{item // 8}" for item in range(128)),
        family_ids=tuple(f"family-{item % 8}" for item in range(128)),
        artifact_ids=tuple(range(1_536)),
    )
    features = {"vjepa2_1": object(), "dinov2": object()}
    predecessor_result = {
        "feature_caches": {},
        "screen_index": {"eval_rgb_leaf_open_count": 0},
        "collection_justified": False,
    }
    calls: list[str] = []

    monkeypatch.setattr(
        dual,
        "load_bound_inputs_v1",
        lambda _authority: (features, index, predecessor_result),
    )

    def fake_train(
        arm: str,
        _features: object,
        _index: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append(arm)
        if arm == ARMS[0]:
            return _arm_result(arm, status=first_status, capacity=False)
        return _arm_result(
            arm,
            status="COMPLETE_TRAIN_SET_CAPACITY_ESTABLISHED",
            capacity=True,
            checkpoint_updates=(800, 1_600),
        )

    monkeypatch.setattr(dual, "train_arm_v1", fake_train)
    monkeypatch.setattr(dual.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dual.torch.cuda, "get_device_name", lambda *_args: "mock-gpu")
    monkeypatch.setattr(
        dual, "_source_bindings_unchanged", lambda *_args: None, raising=False
    )
    authority = {
        "output_root": str(output),
        "config": dual.dual_config_v1(),
        "source_bindings": {},
        "predecessor_bindings": {
            "screen_result": dual.PREDECESSOR_BINDINGS["screen_result"]
        },
    }

    report = dual.execute_v1(authority)

    assert calls == list(ARMS)
    assert report["status"] == "COMPLETE_BOTH_ATTEMPTED_AT_LEAST_ONE_CAPACITY_ESTABLISHED"
    assert set(report["arms"]) == set(ARMS)
    assert report["arms"][ARMS[0]]["status"] == first_status
    assert report["arms"][ARMS[1]]["capacity_established"] is True
    assert (output / "result.json").is_file()
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == report["status"]
    assert terminal["authorizes_rgb_access"] is False


def test_execute_preserves_both_update_checkpoints_in_joint_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "attempt"
    index = SimpleNamespace(
        index_sha256="1" * 64,
        state_ids=tuple(range(128)),
        scene_ids=tuple(f"scene-{item // 8}" for item in range(128)),
        family_ids=tuple(f"family-{item % 8}" for item in range(128)),
        artifact_ids=tuple(range(1_536)),
    )
    monkeypatch.setattr(
        dual,
        "load_bound_inputs_v1",
        lambda _authority: (
            {"vjepa2_1": object(), "dinov2": object()},
            index,
            {
                "feature_caches": {},
                "screen_index": {"eval_rgb_leaf_open_count": 0},
                "collection_justified": False,
            },
        ),
    )
    monkeypatch.setattr(
        dual,
        "train_arm_v1",
        lambda arm, *_args, **_kwargs: _arm_result(
            arm,
            status="COMPLETE_TRAIN_SET_CAPACITY_NOT_ESTABLISHED",
            capacity=False,
            checkpoint_updates=(800, 1_600),
        ),
    )
    monkeypatch.setattr(dual.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dual.torch.cuda, "get_device_name", lambda *_args: "mock-gpu")
    monkeypatch.setattr(
        dual, "_source_bindings_unchanged", lambda *_args: None, raising=False
    )

    report = dual.execute_v1(
        {
            "output_root": str(output),
            "config": dual.dual_config_v1(),
            "source_bindings": {},
            "predecessor_bindings": {
                "screen_result": dual.PREDECESSOR_BINDINGS["screen_result"]
            },
        }
    )

    assert report["status"] == "COMPLETE_BOTH_ATTEMPTED_NO_CAPACITY_ESTABLISHED"
    for arm in ARMS:
        assert set(report["arms"][arm]["checkpoint_bindings"]) == {
            "update_800",
            "update_1600",
        }


def test_new_output_gets_consumed_terminal_on_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "attempt"
    monkeypatch.setattr(
        dual,
        "_read_authority",
        lambda *args, **kwargs: {"output_root": str(output)},
    )

    def fail(_authority: object) -> object:
        output.mkdir()
        raise dual.DualScreenError("checkpoint round-trip failed")

    monkeypatch.setattr(dual, "execute_v1", fail)
    with pytest.raises(dual.DualScreenError, match="checkpoint round-trip"):
        dual.main(
            [
                "--authority",
                str(tmp_path / "unused.json"),
                "--expected-authority-sha256",
                "0" * 64,
                "--expected-authority-byte-count",
                "1",
            ]
        )

    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE"
    assert terminal["citable_as_scientific_evidence"] is False
    assert terminal["authorizes_rgb_access"] is False
    assert terminal["authorizes_evaluation"] is False
    assert terminal["authorizes_collection"] is False


def test_preexisting_output_is_not_contaminated_by_failure_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve\n")
    monkeypatch.setattr(
        dual,
        "_read_authority",
        lambda *args, **kwargs: {"output_root": str(output)},
    )
    monkeypatch.setattr(
        dual,
        "execute_v1",
        lambda _authority: (_ for _ in ()).throw(dual.DualScreenError("failure")),
    )

    with pytest.raises(dual.DualScreenError, match="failure"):
        dual.main(
            [
                "--authority",
                str(tmp_path / "unused.json"),
                "--expected-authority-sha256",
                "0" * 64,
                "--expected-authority-byte-count",
                "1",
            ]
        )

    assert sentinel.read_text() == "preserve\n"
    assert not (output / "terminal.json").exists()
