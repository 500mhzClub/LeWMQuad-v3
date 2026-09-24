from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from lewm.benchmarks import go2_shared_jepa_v5_protected_camera_adaptation_v4 as contract
from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import OUTPUT_SHAPE
from lewm.models import shared_observable_camera_ray_jepa_v5 as shared_v5
from lewm.models import shared_observable_camera_ray_jepa_v5_full_training_v4_loss as baseline_loss
from lewm.models import shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth as tail_loss
from lewm.models.observable_camera_ray_evidence_v4 import ordered_obstacle_first_hit_log_probabilities_v4
from lewm.models.observable_camera_ray_evidence_v4_training import ObservableCameraRayEvidenceV4Targets


ROOT = Path(__file__).resolve().parents[2]


def _runner():
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("_test_protected_camera_adaptation_v4_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _targets(mask: torch.Tensor) -> ObservableCameraRayEvidenceV4Targets:
    batch, height, width = mask.shape
    index = torch.zeros((batch, height, width), dtype=torch.long, device=mask.device)
    offset = torch.zeros_like(mask, dtype=torch.float32)
    ground = torch.zeros((batch, 1, 1, 5), dtype=torch.bool, device=mask.device)
    return ObservableCameraRayEvidenceV4Targets(
        pixel_in_range_hit_mask=mask,
        pixel_no_hit_mask=~mask,
        pixel_hit_bin_index=index,
        pixel_within_bin_offset_m=offset,
        ground_in_frustum=ground,
        ground_clear_to_target=ground,
    )


def _small_model() -> shared_v5.SharedObservableCameraRayJepaV5:
    config = shared_v5.SharedObservableCameraRayJepaV5Config(
        schema=shared_v5.SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
        encoder_depth=0,
        action_dim=3,
        bev_dim=8,
        bev_size=(4, 4),
        predictor_hidden_dim=12,
        target_ema_momentum=0.5,
        source_shape=(128, 128),
        pixel_ray_shape=(3, 4),
        query_chunk_size=5,
        v4_pixel_ray_chunk_size=32,
        observable_camera_ray_v4_weight=1.0,
    )
    return shared_v5.SharedObservableCameraRayJepaV5(config)


def _pair(model: shared_v5.SharedObservableCameraRayJepaV5, batch: int = 4):
    current = torch.randn(batch, 3, shared_v5.IMAGE_SIZE, shared_v5.IMAGE_SIZE)
    origin = torch.tensor((0.326, 0.02, 0.043))[None].expand(batch, -1).clone()
    basis = torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)))[None].expand(batch, -1, -1).clone()
    ground = torch.full((batch,), -0.35)
    action = torch.zeros(batch, model.action_dim)
    action[:, 0] = 1.0
    delta = torch.zeros(batch, 3)
    delta[:, 0] = 0.05
    return model.forward_training_pair(
        current, torch.randn_like(current), action, delta,
        commanded_delta_pose_current=delta,
        current_camera_origin_body_m=origin, current_camera_basis_body_fru=basis,
        current_ground_plane_z_body_m=ground, next_camera_origin_body_m=origin,
        next_camera_basis_body_fru=basis, next_ground_plane_z_body_m=ground,
        diagnostic_wrong_action=torch.roll(action, 1, 1),
        diagnostic_wrong_action_delta_pose_current=torch.roll(delta, 1, 1),
        diagnostic_wrong_commanded_delta_pose_current=torch.roll(delta, 1, 1),
    )


def _supervision(frame: shared_v5.SharedOnlineFrameV5):
    hazard = frame.evidence.pixel_first_hit_hazard_logits
    shape = (hazard.shape[0], hazard.shape[2], hazard.shape[3])
    hit = torch.ones(shape, dtype=torch.bool)
    distance = torch.full(shape, shared_v5.DEPTH_NEAR_EDGE_M + 1.25 * shared_v5.DEPTH_BIN_SIZE_M)
    visible = frame.evidence.ground_query_in_frustum.detach().clone()
    clear = visible & (torch.arange(visible.numel()).reshape(visible.shape) % 2 == 0)
    labels = torch.zeros((hazard.shape[0], *OUTPUT_SHAPE), dtype=torch.long)
    labels[:, 12:40, 16:48] = 1
    labels[:, 28:32, 30:34] = 2
    return shared_v5.ObservableCameraRayV4FrameSupervisionV5(
        pixel_hit_mask=hit,
        pixel_first_hit_distance_m=distance,
        ground_support_in_frustum=visible,
        ground_support_clear_to_target=clear,
        target_raster_labels=labels,
    )


def _progress(update: int, *, passed: int, shortfall: float, worst: float, all_nine: bool = False, loss: float = 999.0):
    return contract.control_decision_from_progress(
        update=update, passed_margin_count=passed, total_shortfall=shortfall,
        worst_margin=worst, aggregate_complete_v4_loss=loss,
        all_nine_physical_pass=all_nine,
    )


def test_tail_depth_math_conditional_mass_top_five_percent_and_gradients() -> None:
    torch.manual_seed(3)
    hazard = torch.randn(4, 64, 1, 5, requires_grad=True)
    offset = torch.randn(4, 64, 1, 5).mul_(0.01).requires_grad_()
    targets = _targets(torch.ones((4, 1, 5), dtype=torch.bool))
    result = tail_loss.tail_depth_p95_cvar_v4(hazard, offset, targets)
    log_hit = ordered_obstacle_first_hit_log_probabilities_v4(hazard).hit
    q = torch.softmax(log_hit, dim=1)
    centers = shared_v5.DEPTH_NEAR_EDGE_M + (torch.arange(64, dtype=offset.dtype) + 0.5) * shared_v5.DEPTH_BIN_SIZE_M
    error = (q * (centers[None, :, None, None] + offset - centers[0]).abs()).sum(1) / 0.25
    assert torch.equal(result, error.flatten().max())
    result.backward()
    for tensor in (hazard, offset):
        assert tensor.grad is not None and bool(torch.isfinite(tensor.grad).all())
        assert float(tensor.grad.abs().sum()) > 0.0


def test_tail_depth_no_hits_has_zero_graph_and_rejects_changed_contract() -> None:
    hazard = torch.zeros(1, 64, 2, 2, requires_grad=True)
    offset = torch.zeros_like(hazard, requires_grad=True)
    zero = tail_loss.tail_depth_p95_cvar_v4(hazard, offset, _targets(torch.zeros((1, 2, 2), dtype=torch.bool)))
    zero.backward()
    assert float(zero.detach()) == 0.0
    assert torch.equal(hazard.grad, torch.zeros_like(hazard))
    assert torch.equal(offset.grad, torch.zeros_like(offset))
    with pytest.raises(ValueError, match="share shape"):
        tail_loss.tail_depth_p95_cvar_v4(hazard[:, :63], offset[:, :63], _targets(torch.ones((1, 2, 2), dtype=torch.bool)))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        tail_loss.tail_depth_p95_cvar_v4(hazard.detach() + float("nan"), offset.detach(), _targets(torch.ones((1, 2, 2), dtype=torch.bool)))


def test_actual_adapter_changes_only_one_component_and_preserves_weights() -> None:
    torch.manual_seed(20260715)
    model = _small_model().train()
    pair = _pair(model)
    current, next_ = _supervision(pair.current), _supervision(pair.next)
    baseline = baseline_loss.observable_camera_ray_v4_loss_v4(model, pair, current, next_)
    adapted = tail_loss.observable_camera_ray_v4_tail_depth_loss_v4(model, pair, current, next_)
    for old, new in zip((baseline.current, baseline.next), (adapted.current, adapted.next), strict=True):
        assert torch.equal(old.hierarchical_first_hit_nll, new.hierarchical_first_hit_nll)
        assert torch.equal(old.ground_clear_distance_state_balanced_bce, new.ground_clear_distance_state_balanced_bce)
        assert torch.equal(old.derived_raster_hierarchical_bce.total, new.derived_raster_hierarchical_bce.total)
        assert torch.equal(old.derived_raster_cell_nll, new.derived_raster_cell_nll)
        expected = 0.25 * (
            new.hierarchical_first_hit_nll + new.tail_depth_p95_cvar
            + new.ground_clear_distance_state_balanced_bce
            + new.derived_raster_hierarchical_bce.total
        ) + 0.25 * new.derived_raster_cell_nll
        assert torch.equal(new.total, expected)
    assert torch.equal(adapted.total, 0.5 * adapted.current.total + 0.5 * adapted.next.total)
    adapted.total.backward()
    assert any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters() if parameter.requires_grad)


def test_controls_are_loss_free_exact_pareto_and_final_hard_stop() -> None:
    for update, base in contract.BASELINE_PROGRESS.items():
        equal = _progress(update, passed=base["passed_margin_count"], shortfall=base["total_shortfall"], worst=base["worst_margin"], loss=0.0)
        assert equal["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
        assert _progress(update, passed=base["passed_margin_count"] + 1, shortfall=base["total_shortfall"], worst=base["worst_margin"], loss=1e30)["action"] == contract.CONTROL_ACTION_CONTINUE
        assert _progress(update, passed=base["passed_margin_count"], shortfall=math.nextafter(base["total_shortfall"], -math.inf), worst=base["worst_margin"], loss=1e30)["action"] == contract.CONTROL_ACTION_CONTINUE
        assert _progress(update, passed=base["passed_margin_count"], shortfall=base["total_shortfall"], worst=math.nextafter(base["worst_margin"], math.inf), loss=1e30)["action"] == contract.CONTROL_ACTION_CONTINUE
        assert _progress(update, passed=base["passed_margin_count"] - 1, shortfall=base["total_shortfall"] - 1.0, worst=base["worst_margin"] + 1.0)["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    assert _progress(4_000, passed=188, shortfall=0.001, worst=-0.001)["action"] == contract.CONTROL_ACTION_STOP_MAXIMUM
    assert _progress(100, passed=0, shortfall=999.0, worst=-99.0, all_nine=True)["action"] == contract.CONTROL_ACTION_QUALIFY


def test_update100_integrity_precedes_apparent_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = {"update": 100, "passed_margin_count": 189, "total_shortfall": 0.0, "worst_margin": 0.1, "aggregate_complete_v4_loss": 1.0, "all_nine_physical_pass": True}
    monkeypatch.setattr(contract, "checkpoint_progress", lambda metric: progress)
    metric = {"state_sha256_before": contract.UPDATE0_STATE_SHA256, "state_sha256_after": contract.UPDATE0_STATE_SHA256, "frozen_state_sha256_before_and_after": "1" * 64, "state_mutation_count": 0}
    with pytest.raises(PermissionError, match="movement"):
        contract.checkpoint_control_decision(metric)
    moved = {**metric, "state_sha256_before": "2" * 64, "state_sha256_after": "2" * 64}
    assert contract.checkpoint_control_decision(moved)["action"] == contract.CONTROL_ACTION_QUALIFY


def test_integrity_failure_occurs_before_sidecar_publication(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    progress = {"update": 100, "passed_margin_count": 189, "total_shortfall": 0.0, "worst_margin": 0.1, "aggregate_complete_v4_loss": 1.0, "all_nine_physical_pass": True}
    monkeypatch.setattr(contract, "checkpoint_progress", lambda metric: progress)
    monkeypatch.setattr(runner._v3, "contract", contract)
    metric = {"state_sha256_before": contract.UPDATE0_STATE_SHA256, "state_sha256_after": contract.UPDATE0_STATE_SHA256, "frozen_state_sha256_before_and_after": "1" * 64, "state_mutation_count": 0}
    with pytest.raises(PermissionError, match="movement"):
        runner._v3._publish_metric_sidecar(tmp_path, update=100, checkpoint={}, metric=metric)
    assert not list(tmp_path.rglob("*"))
    moved = {**metric, "update": 100, "role": "checkpoint_selection", "state_sha256_before": "2" * 64, "state_sha256_after": "2" * 64}
    binding = runner._v3._publish_metric_sidecar(tmp_path, update=100, checkpoint={}, metric=moved)
    path = tmp_path / binding["path"]
    value = contract.parse_canonical_json(path.read_bytes(), name="moved u100 sidecar")
    assert path.read_bytes() == contract.canonical_json_bytes(value) + b"\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert value["continuation"]["action"] == contract.CONTROL_ACTION_QUALIFY
    assert value["inline_evaluation_count"] == 1 and value["state_mutation_count"] == 0


def test_hashed_v3_sidecars_are_the_exact_pareto_baselines() -> None:
    for update, expected in contract.BASELINE_PROGRESS.items():
        relative = f"{contract.V3_SIDECAR_ROOT}/update_{update}.metrics.json"
        path = ROOT / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == contract.FIXED_EVIDENCE_SHA256[relative]
        sidecar = contract.parse_canonical_json(path.read_bytes(), name=relative)
        progress = contract.checkpoint_progress(sidecar["metric"])
        assert {key: progress[key] for key in expected} == expected


def test_source_closure_science_delta_and_runner_import_are_exact() -> None:
    bindings = contract.current_source_bindings(ROOT)
    assert set(bindings) == set(contract.SOURCE_PATHS)
    assert set(contract.V3_SOURCE_SHA256.items()) <= set(bindings.items())
    assert set(contract.FIXED_EVIDENCE_SHA256.items()) <= set(bindings.items())
    assert contract.science_contract()["initial_state_sha256"] == contract.UPDATE0_STATE_SHA256
    assert contract.science_delta()["other_science_changes"] == []
    assert contract.science_delta()["scientific_change_count"] == 1
    assert len(contract.science_delta()["contract_leaf_changes_encoding_that_one_slot_replacement"]) == 3
    assert contract.science_contract()["camera_loss"]["terms"][1] == "tail_depth_p95_cvar"
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    code = f"import importlib.util,sys; p={str(path)!r}; s=importlib.util.spec_from_file_location('r',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert hashlib.sha256((ROOT / contract.V3_RUNNER_RELATIVE_PATH).read_bytes()).hexdigest() == contract.V3_SOURCE_SHA256[contract.V3_RUNNER_RELATIVE_PATH]


def test_persisted_review_and_authorization_round_trip_validate() -> None:
    sources = contract.current_source_bindings(ROOT)
    review = contract.with_content_sha256({
        "schema": contract.REVIEW_SCHEMA,
        "status": "PASS",
        "implementation_author": contract.IMPLEMENTATION_AUTHOR,
        "reviewer": "/root/camera_v4_roundtrip_reviewer",
        "reviewed_sources": sources,
        "predecessor": contract.predecessor_contract(),
        "science_contract": contract.science_contract(),
        "science_delta": contract.science_delta(),
        "evidence": contract.evidence_contract(),
        "reporting_contract": contract.reporting_contract(),
        "control_contract": contract.control_contract(),
        "source_only": True,
        "findings": [],
        "authority": dict(contract.REVIEW_AUTHORITY),
    })
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    parsed_review = contract.parse_canonical_json(review_raw, name="round-trip V4 review")
    assert contract.validate_review(parsed_review, expected_sources=sources) == parsed_review
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=parsed_review["content_sha256"],
    )
    authorization = contract.with_content_sha256({
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": "authorized_one_exact_protected_camera_adaptation_v4_tail_depth_attempt",
        "authorizer": "/root/camera_v4_roundtrip_authorizer",
        "independent_review": review_binding,
        "predecessor": contract.predecessor_contract(),
        "raw": contract.expected_raw_authority(),
        "camera": contract.expected_camera_authority(),
        "experiment": contract.science_contract(),
        "science_delta": contract.science_delta(),
        "evidence": contract.evidence_contract(),
        "reporting_contract": contract.reporting_contract(),
        "control_contract": contract.control_contract(),
        "authority": dict(contract.EXECUTION_AUTHORITY),
    })
    authorization_raw = contract.canonical_json_bytes(authorization) + b"\n"
    parsed_authorization = contract.parse_canonical_json(
        authorization_raw, name="round-trip V4 authorization"
    )
    assert contract.validate_authorization(
        parsed_authorization,
        review_binding=review_binding,
        reviewer=parsed_review["reviewer"],
    ) == parsed_authorization


def test_runner_restores_loss_trace_and_snapshot_hooks_on_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    runtime = SimpleNamespace(loss_adapter=baseline_loss, torch=torch)
    original_loss = baseline_loss.observable_camera_ray_v4_loss_v4
    original_components = runner._v1_runner._camera_components
    original_snapshot = runner._v1_runner._snapshot
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic train failure")
    monkeypatch.setattr(runner, "_BASE_V3_TRAIN", fail)
    with pytest.raises(RuntimeError, match="synthetic train failure"):
        runner._train(runtime, None, None, [], [], [], [], [], [], [], None, None, tmp_path)
    assert baseline_loss.observable_camera_ray_v4_loss_v4 is original_loss
    assert runner._v1_runner._camera_components is original_components
    assert runner._v1_runner._snapshot is original_snapshot


def test_runner_trace_names_tail_term_and_restores_after_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    runtime = SimpleNamespace(loss_adapter=baseline_loss, torch=torch)
    original_loss = baseline_loss.observable_camera_ray_v4_loss_v4
    row = {"losses": {"current_tail_depth_p95_cvar": 1.0, "next_tail_depth_p95_cvar": 2.0}}
    original_components = runner._v1_runner._camera_components
    original_snapshot = runner._v1_runner._snapshot
    def succeed(*args, **kwargs):
        assert runtime.loss_adapter.observable_camera_ray_v4_loss_v4 is tail_loss.observable_camera_ray_v4_tail_depth_loss_v4
        assert runner._v1_runner._camera_components is runner._camera_components
        assert runner._v1_runner._snapshot is not original_snapshot
        return [row], [], [], None, {}, {}
    monkeypatch.setattr(runner, "_BASE_V3_TRAIN", succeed)
    result = runner._train(runtime, None, None, [], [], [], [], [], [], [], None, None, tmp_path)
    assert result[0] == [row]
    assert baseline_loss.observable_camera_ray_v4_loss_v4 is original_loss
    assert runner._v1_runner._camera_components is original_components
    assert runner._v1_runner._snapshot is original_snapshot


def test_finite_snapshot_rejects_nonfinite_state_before_base_snapshot(tmp_path: Path) -> None:
    runner = _runner()
    runtime = SimpleNamespace(torch=torch)
    model = SimpleNamespace(state_dict=lambda: {"encoder.bad": torch.tensor(float("nan"))})
    called = False
    def base(*args, **kwargs):
        nonlocal called
        called = True
        return {}
    checked = runner._finite_snapshot(base, runtime)
    with pytest.raises(FloatingPointError, match="encoder.bad"):
        checked(runtime, model, tmp_path, update=100, frozen_sha="1" * 64)
    assert called is False
