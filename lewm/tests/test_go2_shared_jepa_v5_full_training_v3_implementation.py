"""Source-only and CPU-synthetic proof for Full Training V3."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pytest
import torch

from lewm.benchmarks import go2_shared_jepa_v5_full_training_v3_policy as policy
from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import OUTPUT_SHAPE
from lewm.models import shared_observable_camera_ray_jepa_v5 as shared_v5
from lewm.models import (
    shared_observable_camera_ray_jepa_v5_full_training_v3_loss as loss_adapter,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / policy.EXACT_EXECUTION_MANIFEST_RELATIVE_PATH
SOURCES = {
    "policy": ROOT / policy.POLICY_RELATIVE_PATH,
    "loss_adapter": ROOT / policy.LOSS_ADAPTER_RELATIVE_PATH,
    "preflight": ROOT / policy.PREFLIGHT_EXECUTOR_RELATIVE_PATH,
    "preflight_verifier": ROOT / policy.PREFLIGHT_VERIFIER_RELATIVE_PATH,
    "executor": ROOT / policy.EXACT_EXECUTOR_RELATIVE_PATH,
    "trainer": ROOT / policy.EXACT_TRAINER_RELATIVE_PATH,
    "verifier": ROOT / policy.EXACT_VERIFIER_RELATIVE_PATH,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(name: str) -> str:
    return SOURCES[name].read_text(encoding="ascii")


def _calibration(batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    origin = torch.tensor((0.326, 0.02, 0.043))[None].expand(batch, -1).clone()
    basis = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    )[None].expand(batch, -1, -1).clone()
    ground = torch.full((batch,), -0.35)
    return origin, basis, ground


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


def _pair(
    model: shared_v5.SharedObservableCameraRayJepaV5,
    *,
    batch: int,
) -> shared_v5.SharedTrainingPairV5:
    current = torch.randn(batch, 3, shared_v5.IMAGE_SIZE, shared_v5.IMAGE_SIZE)
    next_image = torch.randn_like(current)
    origin, basis, ground = _calibration(batch)
    action = torch.zeros(batch, model.action_dim)
    action[:, 0] = 1.0
    action[1::2, 0] = 0.0
    action[1::2, 1] = 1.0
    wrong_action = torch.roll(action, shifts=1, dims=1)
    realized = torch.zeros(batch, 3)
    realized[:, 0] = 0.05
    commanded = torch.zeros(batch, 3)
    commanded[:, 0] = 0.10
    wrong_delta = torch.zeros(batch, 3)
    wrong_delta[:, 1] = 0.20
    return model.forward_training_pair(
        current,
        next_image,
        action,
        realized,
        commanded_delta_pose_current=commanded,
        current_camera_origin_body_m=origin,
        current_camera_basis_body_fru=basis,
        current_ground_plane_z_body_m=ground,
        next_camera_origin_body_m=origin,
        next_camera_basis_body_fru=basis,
        next_ground_plane_z_body_m=ground,
        diagnostic_wrong_action=wrong_action,
        diagnostic_wrong_action_delta_pose_current=wrong_delta,
        diagnostic_wrong_commanded_delta_pose_current=wrong_delta,
    )


def _supervision(
    frame: shared_v5.SharedOnlineFrameV5,
) -> shared_v5.ObservableCameraRayV4FrameSupervisionV5:
    hazard = frame.evidence.pixel_first_hit_hazard_logits
    pixel_shape = (hazard.shape[0], hazard.shape[2], hazard.shape[3])
    hit = torch.zeros(pixel_shape, dtype=torch.bool)
    hit[:, 0, 0] = True
    hit[::2, -1, -1] = True
    distance = torch.zeros(pixel_shape, dtype=hazard.dtype)
    distance[hit] = (
        shared_v5.DEPTH_NEAR_EDGE_M
        + 1.25 * shared_v5.DEPTH_BIN_SIZE_M
    )
    in_frustum = frame.evidence.ground_query_in_frustum.detach().clone()
    parity = torch.arange(in_frustum.numel()).reshape(in_frustum.shape) % 2 == 0
    clear = in_frustum & parity
    labels = torch.zeros((hazard.shape[0], *OUTPUT_SHAPE), dtype=torch.long)
    labels[:, 12:40, 16:48] = 1
    labels[:, 28:32, 30:34] = 2
    return shared_v5.ObservableCameraRayV4FrameSupervisionV5(
        pixel_hit_mask=hit,
        pixel_first_hit_distance_m=distance,
        ground_support_in_frustum=in_frustum,
        ground_support_clear_to_target=clear,
        target_raster_labels=labels,
    )


def _group_balanced(rows: Iterable[tuple[str, float]]) -> float:
    groups: dict[str, list[float]] = {}
    for name, value in rows:
        groups.setdefault(name, []).append(value)
    return sum(sum(values) / len(values) for values in groups.values()) / len(groups)


def test_frozen_source_only_parent_and_raw_chain_hashes() -> None:
    assert _sha256(ROOT / policy.V3_AMENDMENT_RELATIVE_PATH) == (
        "93737e1556fc3b523408e0fd01ed632ec8571acb30978ae1f17e1dd653e40278"
    )
    assert _sha256(ROOT / policy.V3_TOPOLOGY_CORRECTION_RELATIVE_PATH) == (
        "49e06b84da81141e59a3a9c4623abc82901320804732c864c8ecd66c51c768a0"
    )
    bindings = policy.reviewed_source_bindings()
    assert all(not path.startswith(".generated/") for path in bindings)
    assert {path: _sha256(ROOT / path) for path in bindings} == bindings
    assert policy.RAW_CHAIN_SOURCE_BINDINGS[
        policy.RAW_SUPERVISION_BUILDER_RELATIVE_PATH
    ] == "2388c1138d9b03ea6e385cc0250c81a1869a40cab62507d02f709ef39197c664"
    assert policy.RAW_CHAIN_SOURCE_BINDINGS[
        policy.RAW_SUPERVISION_AUDITOR_RELATIVE_PATH
    ] == "fddc678187f082a0a245ff5868ca5d944cba4adc2703d3b97088d57451deb4b7"


def test_source_time_manifest_resolves_raw_only_and_blocks_every_camera_hash() -> None:
    manifest = policy.parse_canonical_json(
        MANIFEST.read_bytes(),
        name="blocked Full Training V3 manifest",
    )
    assert manifest == policy.content_value(policy.execution_manifest_core())
    assert manifest["status"] == "blocked_required_bindings_unset"
    assert manifest["exact_execution_authorized"] is False
    assert manifest["dataset_use_authorized_for_exact_attempt"] is False
    assert manifest["terminal_raw_v13_bindings"] == policy.FROZEN_RESOLVED_BINDINGS
    for name, expected in policy.FROZEN_RESOLVED_BINDINGS.items():
        assert manifest["required_exact_bindings"][name] == expected
        assert name not in manifest["unresolved_required_bindings"]
    for name in policy.CAMERA_V13_UNRESOLVED_BINDING_NAMES:
        assert manifest["required_exact_bindings"][name] is None
        assert name in manifest["unresolved_required_bindings"]
    topology = manifest["camera_ladder_topology"]
    assert topology["existing_seed_20260710_n5_attempt_count"] == 1
    assert topology["future_attempt_count"] == 7
    assert topology["aggregate_rung_count"] == 8
    assert topology["seed_20260710_n5_reexecution_authorized"] is False
    assert topology["future_attempts"] == [
        {"seed": 20260710, "fit_size": 16},
        {"seed": 20260710, "fit_size": 32},
        {"seed": 20260710, "fit_size": 320},
        {"seed": 20260711, "fit_size": 5},
        {"seed": 20260711, "fit_size": 16},
        {"seed": 20260711, "fit_size": 32},
        {"seed": 20260711, "fit_size": 320},
    ]
    with pytest.raises(PermissionError, match="blocked before reservation"):
        policy.validate_execution_manifest(manifest, require_ready=True)
    changed = dict(manifest["required_exact_bindings"])
    changed["development_raw_supervision_audit_file_sha256"] = "f" * 64
    with pytest.raises(PermissionError, match="terminal Raw V13"):
        policy.execution_manifest_core(required_bindings=changed)


def test_implementation_review_contract_binds_both_amendments_and_adapter() -> None:
    source_bindings = {path: "a" * 64 for path in policy.IMPLEMENTATION_SOURCE_PATHS}
    core = policy.expected_implementation_review_core(
        reviewer="/root/full_training_v3_independent_review",
        source_bindings=source_bindings,
    )
    assert core["frozen_design_bindings"][policy.V3_AMENDMENT_RELATIVE_PATH] == (
        policy.V3_AMENDMENT_SHA256
    )
    assert core["frozen_design_bindings"][
        policy.V3_TOPOLOGY_CORRECTION_RELATIVE_PATH
    ] == policy.V3_TOPOLOGY_CORRECTION_SHA256
    assert policy.LOSS_ADAPTER_RELATIVE_PATH in core["reviewed_sources"]
    assert core["raw_v13_dataset_use_grant"] == policy.RAW_DATASET_USE_GRANT
    assert core["camera_ladder_future_attempt_count"] == 7
    assert core["seed_20260710_n5_reexecution_authorized"] is False
    with pytest.raises(PermissionError):
        policy.expected_implementation_review_core(
            reviewer=policy.IMPLEMENTATION_AUTHOR,
            source_bindings=source_bindings,
        )


def test_loss_and_reduction_contract_are_exact() -> None:
    promoted = policy.JOINT_LOSS_CONTRACT["promoted_jepa"]
    assert promoted["v4_components"] == {
        "hierarchical_first_hit_nll": 0.25,
        "target_bin_offset_smooth_l1": 0.25,
        "ground_clear_distance_state_balanced_bce": 0.25,
        "derived_raster_hierarchical_bce": 0.25,
        "derived_raster_cell_nll": 0.25,
    }
    assert promoted["current_and_next_computed_separately_at_batch_size"] == 4
    assert promoted["current_next_scalar_average"] == [0.5, 0.5]
    assert promoted["microbatch_scalar_average"] == [0.25] * 4
    assert promoted["synthetic_b16_nonlinear_pooling_authorized"] is False
    assert policy.average_current_next_b4_scalars(2.0, 6.0) == 4.0

    microbatches = [
        [("A", 0.0)],
        [("A", 0.0)],
        [("A", 0.0), ("B", 100.0)],
        [("A", 0.0)],
    ]
    four_scalars = [_group_balanced(rows) for rows in microbatches]
    correct = policy.average_four_microbatch_scalars(four_scalars)
    pooled_b16_style = _group_balanced(
        row for microbatch in microbatches for row in microbatch
    )
    assert correct == 12.5
    assert pooled_b16_style == 50.0
    assert correct != pooled_b16_style
    with pytest.raises(ValueError, match="exactly four"):
        policy.average_four_microbatch_scalars([pooled_b16_style])

    leaves = [torch.tensor(float(index), requires_grad=True) for index in range(4)]
    mean = loss_adapter.average_four_microbatch_tensor_scalars_v3(leaves)
    mean.backward()
    assert float(mean.detach()) == 1.5
    assert [float(value.grad) for value in leaves] == [0.25] * 4


def test_cpu_synthetic_adapter_uses_five_terms_and_separate_current_next_b4() -> None:
    torch.manual_seed(20260714)
    model = _small_model().train()
    pair = _pair(model, batch=4)
    current_supervision = _supervision(pair.current)
    next_supervision = _supervision(pair.next)
    camera = loss_adapter.observable_camera_ray_v4_loss_v3(
        model,
        pair,
        current_supervision,
        next_supervision,
    )
    joint = loss_adapter.combine_joint_losses_v3(
        model,
        pair,
        current_supervision,
        next_supervision,
    )
    for frame in (camera.current, camera.next):
        expected_base = 0.25 * (
            frame.hierarchical_first_hit_nll
            + frame.target_bin_offset_smooth_l1
            + frame.ground_clear_distance_state_balanced_bce
            + frame.derived_raster_hierarchical_bce.total
        )
        assert torch.allclose(frame.retained_v11_base_total, expected_base)
        assert torch.allclose(
            frame.total,
            expected_base + 0.25 * frame.derived_raster_cell_nll,
        )
        assert frame.derived_raster_cell_nll.requires_grad
        assert frame.hierarchical_first_hit_nll.requires_grad
        assert not hasattr(frame, "ordered_first_hit_nll")
    assert torch.allclose(camera.total, 0.5 * camera.current.total + 0.5 * camera.next.total)
    assert torch.allclose(joint.total, pair.jepa.total + camera.total)
    raster_nll_gradient = torch.autograd.grad(
        camera.current.derived_raster_cell_nll,
        pair.current.evidence.pixel_first_hit_hazard_logits,
        retain_graph=True,
    )[0]
    hierarchical_gradient = torch.autograd.grad(
        camera.current.hierarchical_first_hit_nll,
        pair.current.evidence.pixel_first_hit_hazard_logits,
        retain_graph=True,
    )[0]
    assert float(raster_nll_gradient.abs().sum()) > 0.0
    assert float(hierarchical_gradient.abs().sum()) > 0.0
    joint.total.backward()
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    wrong_batch_pair = _pair(model, batch=2)
    with pytest.raises(ValueError, match="synthetic-B16 pooling is forbidden"):
        loss_adapter.observable_camera_ray_v4_loss_v3(
            model,
            wrong_batch_pair,
            _supervision(wrong_batch_pair.current),
            _supervision(wrong_batch_pair.next),
        )


def test_pre_g2_candidate_schema_is_strictly_distinct_from_post_g2() -> None:
    state_sha = "1" * 64
    core = policy.pre_g2_candidate_checkpoint_core(
        model_config={"model": "shared-v5"},
        deployment_state_sha256=state_sha,
        selection={"selected_update": 8000},
        calibration={"global_threshold": 0.9},
    )
    assert core == {
        "schema": policy.PRE_G2_CANDIDATE_CHECKPOINT_SCHEMA,
        "lifecycle_stage": (
            "development_selected_and_calibrated_pending_independent_"
            "exact_reconstruction_and_g2"
        ),
        "checkpoint_kind": "pre_g2_candidate",
        "model_config": {"model": "shared-v5"},
        "deployment_state_sha256": state_sha,
        "selection": {"selected_update": 8000},
        "calibration": {"global_threshold": 0.9},
        "development_only": True,
        "independent_exact_reconstruction_required": True,
        "g2_attempted": False,
        "g2_gate_receipt": None,
        "post_g2_qualified": False,
        "runtime_ready": False,
        "heldout_authorized": False,
        "runtime_navigation_hardware_authorized": False,
        "production_promotion_deployment_authorized": False,
    }
    assert core["schema"] != shared_v5.CHECKPOINT_V5_SCHEMA
    assert "pre_g2_candidate_checkpoint.pt" in policy.EXACT_INVENTORY
    assert "qualified_checkpoint.pt" not in policy.EXACT_INVENTORY


def test_production_sources_use_v3_adapter_and_preserve_reservation_boundaries() -> None:
    combined = "\n".join(_source(name) for name in SOURCES)
    assert "development_fit_v2" not in combined
    assert "qualified_checkpoint.pt" not in combined
    assert "CHECKPOINT_V5_SCHEMA" not in combined
    assert "ordered_first_hit_nll" not in combined
    assert "model.combine_joint_losses(" not in combined
    assert "model.observable_camera_ray_v4_loss(" not in combined
    assert "pre_g2_candidate_checkpoint.pt" in combined
    assert "combine_joint_losses_v3" in _source("preflight")
    assert "combine_joint_losses_v3" in _source("trainer")
    assert "observable_camera_ray_v4_loss_v3" in _source("verifier")
    assert "(backward / policy.ACCUMULATION_STEPS).backward()" in _source(
        "trainer"
    )
    assert "(joint.total / policy.ACCUMULATION_STEPS).backward()" in _source(
        "preflight"
    )
    assert "future_attempt_count\") != 7" in _source("trainer")
    assert "seed_20260710_n5_reexecuted" in _source("trainer")
    assert policy.PREFLIGHT_ROOT_RELATIVE_PATH.endswith("full_training_v3_preflight")
    assert policy.EXACT_ROOT_RELATIVE_PATH.endswith("full_training_v3")

    neural_loaders = {
        "preflight": {"_load_production_backend"},
        "trainer": {"_fixed_production_backend_loader"},
        "verifier": {"_load_fixed_backend"},
    }
    for name, allowed in neural_loaders.items():
        tree = ast.parse(_source(name), filename=str(SOURCES[name]))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        neural_import_count = 0
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            if not any(
                module == "torch"
                or module.startswith("torch.")
                or module.startswith("lewm.models")
                for module in modules
            ):
                continue
            neural_import_count += 1
            ancestors: list[str] = []
            current: ast.AST | None = node
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ancestors.append(current.name)
                current = parents.get(current)
            assert allowed.intersection(ancestors), (name, modules, ancestors)
        assert neural_import_count > 0
    for name in ("preflight_verifier", "executor"):
        assert "import torch" not in _source(name)
        assert "from lewm.models" not in _source(name)


def test_sources_are_ascii_parseable_and_do_not_expose_dynamic_backends() -> None:
    for path in SOURCES.values():
        raw = path.read_bytes()
        raw.decode("ascii")
        ast.parse(raw, filename=str(path))
    combined = "\n".join(_source(name) for name in SOURCES)
    for forbidden in (
        "--backend",
        "--module",
        "--callback",
        "--test-only",
        "--fixture",
        "autocast(",
        "cuda:1",
    ):
        assert forbidden not in combined
    assert math.isfinite(policy.learning_rate(8000))
    assert json.loads(MANIFEST.read_text(encoding="ascii"))["heldout_authorized"] is False
