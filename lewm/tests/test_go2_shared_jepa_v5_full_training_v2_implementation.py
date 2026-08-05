"""Source-only author checks for the frozen Shared JEPA V5 V2 implementation."""
from __future__ import annotations

import ast
import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import pytest

from lewm.benchmarks import go2_shared_jepa_v5_full_training_v2_policy as policy


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "policy": ROOT / policy.POLICY_RELATIVE_PATH,
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


def _function_from_source(name: str, function_name: str) -> tuple[Any, dict[str, Any]]:
    tree = ast.parse(_source(name), filename=str(SOURCES[name]))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    assert len(matches) == 1
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            matches[0],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, str(SOURCES[name]), "exec"), namespace)
    return namespace[function_name], namespace


def _eligible_scope() -> dict[str, Any]:
    return {
        "physical": {
            "pixel_first_hit_balanced_accuracy": 0.99,
            "depth_median_error_m": 0.05,
            "depth_p95_error_m": 0.15,
            "ground_clear_balanced_accuracy": 0.99,
            "distance_group_balanced_accuracy": [0.96, 0.97],
            "derived_raster_nll": 0.10,
            "derived_raster_balanced_accuracy": 0.99,
            "present_class_recall": {
                "UNKNOWN": 0.99,
                "FREE": 0.99,
                "OCCUPIED": 0.99,
            },
            "wrong_rgb_pixel_balanced_accuracy_drop": 0.20,
            "wrong_rgb_depth_median_error_increase_m": 0.20,
            "wrong_rgb_depth_p95_error_increase_m": 0.30,
            "wrong_rgb_ground_balanced_accuracy_drop": 0.20,
            "wrong_rgb_raster_nll_increase": 0.20,
            "wrong_rgb_raster_balanced_accuracy_drop": 0.20,
        },
        "jepa": {
            "prediction_valid_cell_count": 100,
            "target_cross_sample_std_mean": 0.10,
            "target_cross_sample_effective_rank": 8.0,
            "warped_persistence_target_change": 0.20,
            "prediction_to_warped_persistence_ratio": 0.50,
            "wrong_action_advantage_over_target_change": 0.20,
            "wrong_commanded_delta_advantage_over_target_change": 0.10,
            "wrong_action_prediction_sensitivity": 0.10,
            "wrong_commanded_delta_prediction_sensitivity": 0.10,
        },
    }


def _candidate(update: int, *, complete_v4_loss: float) -> dict[str, Any]:
    scopes = {scope: copy.deepcopy(_eligible_scope()) for scope in policy.SCOPES}
    return {
        "update": update,
        "scopes": scopes,
        "aggregate_complete_v4_loss": complete_v4_loss,
        "aggregate_prediction_to_persistence_ratio": 0.50,
    }


def _append_event(
    events: list[dict[str, Any]],
    *,
    stage: str,
    role: str,
    arm: str | None,
    path: str,
) -> None:
    digest = hashlib.sha256(path.encode("ascii")).hexdigest()
    events.append(
        policy.append_access_event(
            events,
            stage=stage,
            arm=arm,
            role=role,
            operation="author_test_rehash",
            relative_path=path,
            expected_sha256=digest,
            observed_sha256=digest,
            byte_count=len(path),
            process_identity="author-test",
        )
    )


def test_frozen_parent_closure_and_blocked_manifest_are_exact() -> None:
    assert {
        relative: _sha256(ROOT / relative)
        for relative in policy.reviewed_source_bindings()
    } == policy.reviewed_source_bindings()
    manifest_path = ROOT / policy.EXACT_EXECUTION_MANIFEST_RELATIVE_PATH
    manifest = policy.parse_canonical_json(
        manifest_path.read_bytes(),
        name="blocked V2 execution manifest",
    )
    assert manifest == policy.content_value(policy.execution_manifest_core())
    assert manifest["exact_execution_authorized"] is False
    assert manifest["unresolved_required_bindings"] == sorted(
        policy.REQUIRED_BINDING_NAMES
    )
    assert len(manifest["unresolved_required_bindings"]) == 19
    assert all(
        value is None for value in manifest["required_exact_bindings"].values()
    )
    assert manifest["live_navigation_readiness_hash_authoritative"] is False
    with pytest.raises(PermissionError, match="blocked before reservation"):
        policy.validate_execution_manifest(manifest, require_ready=True)


def test_frozen_science_counts_loss_device_and_inventory() -> None:
    assert (
        policy.UPDATE_COUNT,
        policy.PRESENTATION_COUNT,
        policy.EFFECTIVE_BATCH_SIZE,
        policy.MICROBATCH_SIZE,
        policy.ACCUMULATION_STEPS,
    ) == (8000, 128000, 16, 4, 4)
    assert policy.CHECKPOINT_UPDATES == tuple(range(1000, 8001, 1000))
    assert policy.ARMS == ("promoted_jepa", "matched_no_jepa")
    assert policy.ROLE_COUNTS == {
        "train": {
            "scenes": 72,
            "pairs": 4262,
            "endpoint_instances": 8524,
            "unique_endpoints": 7777,
        },
        "checkpoint_selection": {
            "scenes": 8,
            "pairs": 495,
            "endpoint_instances": 990,
            "unique_endpoints": 924,
        },
        "probability_calibration": {
            "scenes": 8,
            "pairs": 415,
            "endpoint_instances": 830,
            "unique_endpoints": 759,
        },
    }
    assert policy.OPTIMIZER_CONTRACT["name"] == "AdamW"
    assert policy.OPTIMIZER_CONTRACT["gradient_clip_norm"] == 1.0
    assert policy.OPTIMIZER_CONTRACT["ema_updates_per_optimizer_step"] == 1
    assert policy.JOINT_LOSS_CONTRACT["promoted_jepa"]["v4_components"] == {
        "ordered_first_hit_nll": 0.25,
        "target_bin_offset_smooth_l1": 0.25,
        "ground_clear_distance_state_balanced_bce": 0.25,
        "derived_raster_hierarchical_bce": 0.25,
    }
    assert policy.JOINT_LOSS_CONTRACT["matched_no_jepa"] == {
        "established_jepa_total_weight": 0.0,
        "current_v4_weight": 0.5,
        "next_v4_weight": 0.5,
        "same_forward_and_diagnostics": True,
    }
    assert policy.DEVICE_CONTRACT["device_name"] == "AMD Radeon AI PRO R9700"
    assert policy.DEVICE_CONTRACT["minimum_total_memory_bytes"] == 32 * 1024**3
    assert policy.MODEL_IMAGE_SIZE == 112
    assert policy.MODEL_SOURCE_SHAPE == (128, 128)
    assert policy.MODEL_PIXEL_RAY_SHAPE == (84, 112)
    assert policy.MODEL_BEV_SHAPE == (64, 64)
    assert policy.EXACT_INVENTORY[-1] == "completed.json"
    assert "preflight_receipt_binding.json" in policy.EXACT_INVENTORY
    assert "selection_role_ablation_diagnostic.json" in policy.EXACT_INVENTORY
    assert "ablation_comparison.json" not in policy.EXACT_INVENTORY


def test_learning_rate_and_schedule_contract_are_exact() -> None:
    assert policy.learning_rate(1) == 1e-6
    assert policy.learning_rate(400) == 1e-4
    assert policy.learning_rate(8000) == 1e-5
    assert policy.learning_rate(401) < policy.learning_rate(400)
    with pytest.raises(ValueError):
        policy.learning_rate(0)
    cycles, remainder = divmod(policy.PRESENTATION_COUNT, policy.TRAIN_PAIR_COUNT)
    indices = list(range(policy.TRAIN_PAIR_COUNT)) * cycles
    indices.extend(range(remainder))
    pair_ids = [f"pair-{index:04d}" for index in range(policy.TRAIN_PAIR_COUNT)]
    commitment = policy.schedule_commitment(indices, pair_ids)
    assert commitment["seed"] == 20260713
    assert commitment["presentation_count"] == 128000
    assert commitment["update_count"] == 8000
    broken = list(indices)
    broken[1] = broken[0]
    with pytest.raises(ValueError, match="not a train-role permutation"):
        policy.validate_exact_schedule_indices(broken)


def test_checkpoint_ranking_and_calibration_are_unrounded_and_role_fixed() -> None:
    candidates = [
        _candidate(update, complete_v4_loss=1.0 if update == 4000 else 2.0)
        for update in policy.CHECKPOINT_UPDATES
    ]
    selection = policy.select_promoted_checkpoint(candidates)
    assert selection["selected_update"] == 4000
    assert selection["eligible_updates"] == list(policy.CHECKPOINT_UPDATES)
    assert policy.selection_role_ablation_contract() == {
        "population_role": "checkpoint_selection",
        "interpretation": "matched_development_diagnostic_only",
        "causal_generalization_claim_authorized": False,
        "qualification_or_selection_effect": "none",
        "ablation_checkpoint_substitution_authorized": False,
        "retry_or_intervention_authorized": False,
    }
    reports = {
        policy.canonical_json_sha256(list(values)): {
            "admitted_free_count": 100,
            "admitted_free_true_free_count": 100,
            "useful_free_count": 100,
            "useful_free_admitted_count": 90,
            "obstacle_within_2m_count": 100,
            "obstacle_within_2m_excluded_count": 100,
            "obstacle_within_2m_detected_count": 100,
        }
        for values in policy.threshold_grid()
    }
    threshold = policy.select_calibration_threshold(reports)
    assert threshold["free_probability_minimum"] == 0.50
    assert threshold["occupied_detection_minimum"] == 0.50
    parameters = policy.centered_vector_scaling_parameters(
        (-4.0, 0.0, 4.0),
        (1.0, 2.0, 3.0),
    )
    assert parameters["log_scales"] == [-3.0, 0.0, 3.0]
    assert math.isclose(sum(parameters["centered_biases"]), 0.0, abs_tol=1e-12)


def test_exact_access_ledger_requires_complete_terminal_rehash() -> None:
    opened: list[dict[str, Any]] = []
    _append_event(
        opened,
        stage="exact_source_closure",
        role="source_closure",
        arm=None,
        path="docs/frozen.md",
    )
    _append_event(
        opened,
        stage="gradient",
        role="train",
        arm="promoted_jepa",
        path="development/train.bin",
    )
    assert policy.validate_access_ledger(opened)["completion_rehash_event_count"] == 0
    with pytest.raises(PermissionError, match="completion rehash closure"):
        policy.validate_access_ledger(opened, require_completion_rehash=True)
    completed = list(opened)
    for event in opened:
        completed.append(
            policy.append_access_event(
                completed,
                stage="completion_rehash",
                arm=event["arm"],
                role=event["role"],
                operation="rehash_before_completion",
                relative_path=event["relative_path"],
                expected_sha256=event["expected_sha256"],
                observed_sha256=event["observed_sha256"],
                byte_count=event["byte_count"],
                process_identity="author-test",
            )
        )
    summary = policy.validate_access_ledger(
        completed,
        require_completion_rehash=True,
    )
    assert summary["unique_input_count"] == 2
    assert summary["completion_rehash_event_count"] == 2
    duplicate = list(completed)
    event = opened[0]
    duplicate.append(
        policy.append_access_event(
            duplicate,
            stage="completion_rehash",
            arm=event["arm"],
            role=event["role"],
            operation="duplicate",
            relative_path=event["relative_path"],
            expected_sha256=event["expected_sha256"],
            observed_sha256=event["observed_sha256"],
            byte_count=event["byte_count"],
            process_identity="author-test",
        )
    )
    with pytest.raises(PermissionError, match="duplicated"):
        policy.validate_access_ledger(duplicate, require_completion_rehash=True)


def test_neural_imports_are_nested_behind_fixed_reservations() -> None:
    allowed_loaders = {
        "preflight": {"_load_production_backend"},
        "trainer": {"_fixed_production_backend_loader"},
        "verifier": {"_load_fixed_backend"},
    }
    for name, loaders in allowed_loaders.items():
        tree = ast.parse(_source(name), filename=str(SOURCES[name]))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        found = 0
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
            found += 1
            ancestors = []
            current: ast.AST | None = node
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ancestors.append(current.name)
                current = parents.get(current)
            assert loaders.intersection(ancestors), (name, modules, ancestors)
        assert found > 0
    for name in ("preflight_verifier", "executor"):
        source = _source(name)
        assert "import torch" not in source
        assert "from lewm.models" not in source


def test_fixed_orchestration_orders_reservation_processes_and_publication() -> None:
    preflight, preflight_ns = _function_from_source(
        "preflight", "_orchestrate_preflight"
    )
    calls: list[str] = []
    reservation = object()
    preflight_ns["_terminalize_failure"] = lambda _reservation, _error: calls.append(
        "failure"
    )
    preflight_ns["_close_reservation"] = lambda _reservation: calls.append("close")
    result = preflight(
        lambda: calls.append("reserve") or reservation,
        lambda item: calls.append("fresh-smoke-child") or {"measurement": item is reservation},
        lambda item, measurement: calls.append("publish") or {"published": measurement},
        lambda item: calls.append("fresh-verifier") or {"verified": item is reservation},
        lambda item, artifacts, verification: calls.append("complete") or (artifacts, verification),
    )
    assert result[0]["published"]["measurement"] is True
    assert calls == [
        "reserve",
        "fresh-smoke-child",
        "publish",
        "fresh-verifier",
        "complete",
        "close",
    ]

    exact, _exact_ns = _function_from_source("executor", "_orchestrate_exact")
    calls = []
    result = exact(
        lambda: calls.append("reserve") or reservation,
        lambda item: calls.append("fresh-trainer") or {"trainer": item is reservation},
        lambda item: calls.append("fresh-verifier") or {"verifier": item is reservation},
        lambda item, trainer, verifier: calls.append("complete") or (trainer, verifier),
        lambda _item, _error: calls.append("failure"),
        lambda _item: calls.append("close"),
    )
    assert result == ({"trainer": True}, {"verifier": True})
    assert calls == ["reserve", "fresh-trainer", "fresh-verifier", "complete", "close"]

    trainer, _trainer_ns = _function_from_source("trainer", "_orchestrate_trainer")
    calls = []
    result = trainer(
        lambda: calls.append("reservation") or reservation,
        lambda item: calls.append("preflight-first") or {"reservation": item},
        lambda item, preflight_value: calls.append("authority") or (item, preflight_value),
        lambda: calls.append("backend-import") or object(),
        lambda backend, authority: calls.append("backend-run") or (backend, authority),
    )
    assert result[1][0] is reservation
    assert calls == [
        "reservation",
        "preflight-first",
        "authority",
        "backend-import",
        "backend-run",
    ]


def test_fixed_sources_close_dynamic_backends_and_preserve_exact_semantics() -> None:
    combined = "\n".join(_source(name) for name in SOURCES)
    for forbidden in (
        "--backend",
        "--module",
        "--callback",
        "--test-only",
        "--fixture",
        "autocast(",
        "cuda:1",
        'HIP_VISIBLE_DEVICES"] = "1"',
        'ROCR_VISIBLE_DEVICES"] = "1"',
    ):
        assert forbidden not in combined
    preflight = _source("preflight")
    trainer = _source("trainer")
    verifier = _source("verifier")
    executor = _source("executor")
    assert '"-I"' in preflight and '"--fixed-smoke-child"' in preflight
    assert preflight.index("reservation = reserve_operation()") < preflight.index(
        "measurements = smoke_operation(reservation)"
    )
    assert "Image.Resampling.BILINEAR" in trainer
    assert "Image.Resampling.BILINEAR" in verifier
    assert trainer.index('fit.load_state_dict(checkpoint["state_dict"], strict=True)') < trainer.index(
        "torch.random.default_generator.manual_seed("
    ) < trainer.index("shared = model_module.SharedObservableCameraRayJepaV5(config)")
    assert verifier.index('fit.load_state_dict(checkpoint["state_dict"], strict=True)') < verifier.index(
        "torch.random.default_generator.manual_seed("
    ) < verifier.index("shared = model_module.SharedObservableCameraRayJepaV5()")
    assert "if parameter.requires_grad" in trainer
    assert "self._completion_rehash()" in trainer
    assert "target_parts.append(target.cpu())" in trainer
    assert "target_parts.append(target.cpu())" in verifier
    assert "float(value.cpu()) * len(batch_pairs)" not in trainer
    assert 'report["calibrated_nll"]' in trainer
    assert 'report["calibrated_nll"]' in verifier
    assert '"scene_id_by_family": scene_id_by_family' in trainer
    assert '"raw_metric_deltas": self._numeric_metric_delta(' in trainer
    assert '"raw_metric_deltas": self._metric_delta(' in verifier
    assert "require_completion_rehash=True" in verifier
    verifier_tree = ast.parse(verifier)
    matched_checkpoint_calls = [
        node
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_checkpoint"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "matched_no_jepa"
    ]
    assert matched_checkpoint_calls
    assert "migration_baseline_nonselectable" in verifier
    assert "from scripts.train_go2_shared_jepa_v5_full_training_v2" not in verifier
    assert "_inventory_files(reservation.directory_fd)" in executor
    assert "raw_inputs_and_checkpoints_reopened" in executor
    assert (
        "preflight source ledger did not exactly cover reviewed sources"
        in _source("preflight_verifier")
    )
