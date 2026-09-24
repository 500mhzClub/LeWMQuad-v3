from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = (
    ROOT
    / (
        "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v4_"
        "matched_no_persistence.py"
    )
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _TinyInitialState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        self.register_buffer("mask", torch.tensor([True, False], dtype=torch.bool))
        self.register_buffer(
            "target_hard_sync_count", torch.ones((), dtype=torch.long)
        )
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))


def test_bindings_schemas_and_frozen_reference_are_exact_and_detached() -> None:
    module = _load("_test_matched_no_persistence_bindings")
    for name in (
        "LABEL_ROOT_RELATIVE_PATH",
        "LABEL_MANIFEST_NAME",
        "LABEL_MANIFEST_CONTENT_SHA256",
        "LABEL_MANIFEST_FILE_SHA256",
        "LABEL_MANIFEST_BYTE_COUNT",
        "REQUIRED_GPU_NAME",
        "REQUIRED_GPU_MEMORY_BYTES",
        "ACTION_ORDER",
        "ROLE_FILES",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
        "CONSTRUCTOR_INITIALIZATION_SEED",
        "SEMANTIC_DECODER_INITIALIZATION_SEED",
        "EXPERIMENT_SEED",
        "BOOTSTRAP_SEED",
        "CONTROL_NAMES",
        "ALL_ARM_NAMES",
        "REGISTERED_FAMILIES",
        "GATE_THRESHOLDS",
    ):
        assert getattr(module, name) == getattr(module._v4, name)
    assert module.AUXILIARY_OBJECTIVE == module._v4.AUXILIARY_OBJECTIVE
    assert module.AUXILIARY_OBJECTIVE["coefficient"] == 0.5
    assert module.OUTPUT_RELATIVE_PATH == (
        ".generated/"
        "go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence/"
        "attempt_v1"
    )
    schemas = {
        module.CHECKPOINT_SCHEMA,
        module.TRACE_SCHEMA,
        module.RESULT_SCHEMA,
        module.FAILURE_SCHEMA,
    }
    assert len(schemas) == 4
    assert all("v4_matched_no_persistence" in value for value in schemas)

    receipt = module.full_v4_reference_family_utility_receipt_v1()
    assert receipt["canonical_json_sha256"] == (
        "8ba8d6126e922f6a36038304e3444d0d21ee69350fef4acd3828265754810e1e"
    )
    assert receipt["payload"]["schema"] == (
        "lewm_v4_full_reference_family_utility_v1"
    )
    assert receipt["payload"]["family_order"] == list(module.REGISTERED_FAMILIES)
    assert receipt["payload"]["normalized_chosen_prefix_utility"] == [
        0.8896189747752248,
        0.9384050589932943,
        0.8938629676334595,
        0.8772593292124542,
        0.8934829059829059,
        0.9430145611963794,
        0.922340425531915,
        0.9229020111832612,
    ]
    assert receipt["runtime_artifact_reopened"] is False
    assert receipt["reviewed_result_file_sha256"] == (
        "bf93c96cf020553be74d51847c6876e345cd6cc391b05cec186e36b20ca15aa4"
    )
    assert receipt["reviewed_result_content_sha256"] == (
        "27ecf4895dfea01a1e5bb4f6f13f3add6a182a8dfa4b9f8651204bd1e6222ad8"
    )
    receipt["payload"]["family_order"][0] = "changed"
    assert module.FULL_V4_REFERENCE_FAMILY_UTILITY["family_order"][0] == (
        "large_enclosed_maze"
    )


def test_persistence_receipt_freezes_the_only_backward_delta() -> None:
    module = _load("_test_matched_no_persistence_treatment")
    receipt = module.persistence_treatment_receipt_v1()
    assert receipt == {
        "schema": "lewm_v4_matched_no_persistence_backward_membership_v1",
        "full_v4_backward_scalar": "S + P + U + R + O",
        "control_backward_scalar": "S + U + R + O",
        "persistence_diagnostic_computed": True,
        "persistence_backward_coefficient": 0.0,
        "persistence_detached": False,
        "sole_treatment_delta": "P_absent_from_backward_membership",
    }
    receipt["persistence_backward_coefficient"] = 1.0
    assert module.PERSISTENCE_TREATMENT["persistence_backward_coefficient"] == 0.0
    assert module.TRACE_LOSS_KEYS == (
        "S",
        "P_diagnostic",
        "U",
        "R",
        "O",
        "L_full_diagnostic",
        "L_backward",
    )


def test_two_fresh_initial_states_receive_one_canonical_equal_witness() -> None:
    module = _load("_test_matched_no_persistence_initial_state")
    first = _TinyInitialState()
    second = _TinyInitialState()
    receipt = module._reconstructed_initialization_receipt_v1(
        first, second, torch=torch
    )
    assert receipt["reconstruction_count"] == 2
    assert receipt["selected_control_reconstruction"] == 1
    assert receipt["payloads_equal"] is True
    assert receipt["digests_equal"] is True
    assert receipt["reconstruction_digests"] == [
        receipt["canonical_state_entries_sha256"],
        receipt["canonical_state_entries_sha256"],
    ]
    assert [row["name"] for row in receipt["canonical_state_entries"]] == sorted(
        first.state_dict()
    )
    assert all(
        set(row) == {"name", "dtype", "shape", "tensor_byte_sha256"}
        and len(row["tensor_byte_sha256"]) == 64
        for row in receipt["canonical_state_entries"]
    )
    assert receipt["counters"] == [
        {
            "reconstruction": 1,
            "target_hard_sync_count": 1,
            "ema_update_count": 0,
        },
        {
            "reconstruction": 2,
            "target_hard_sync_count": 1,
            "ema_update_count": 0,
        },
    ]

    with torch.no_grad():
        second.weight[0, 0] = -1.0
    with pytest.raises(RuntimeError, match="not tensor-identical"):
        module._reconstructed_initialization_receipt_v1(
            first, second, torch=torch
        )


def test_optimizer_receipt_records_empty_state_membership_and_hyperparameters() -> None:
    module = _load("_test_matched_no_persistence_optimizer")
    encoder = torch.nn.Parameter(torch.ones((2, 2)))
    lift = torch.nn.Parameter(torch.ones((3,)))
    predictor = torch.nn.Parameter(torch.ones((4,)))
    target = torch.nn.Parameter(torch.ones((2,)), requires_grad=False)
    partition = SimpleNamespace(
        encoder=(encoder,),
        lift_semantic=(lift,),
        predictor=(predictor,),
        target=(target,),
        names={
            "encoder": ("encoder.weight",),
            "lift_semantic": ("semantic_head.weight",),
            "predictor": ("predictor.weight",),
            "target": ("target_encoder.weight",),
        },
    )
    optimizer = torch.optim.AdamW(
        [
            {"name": "encoder", "params": [encoder], "lr": 1e-4},
            {"name": "lift_semantic", "params": [lift], "lr": 3e-4},
            {"name": "predictor", "params": [predictor], "lr": 3e-4},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    receipt = module._optimizer_receipt_v1(optimizer, partition)
    assert receipt["optimizer_type"] == "AdamW"
    assert receipt["state_empty"] is True
    assert receipt["state_entry_count"] == 0
    assert receipt["target_parameter_names"] == ["target_encoder.weight"]
    assert receipt["target_excluded_from_optimizer"] is True
    assert [group["name"] for group in receipt["parameter_groups"]] == [
        "encoder",
        "lift_semantic",
        "predictor",
    ]
    assert [
        group["hyperparameters"]["lr"] for group in receipt["parameter_groups"]
    ] == [1e-4, 3e-4, 3e-4]
    assert all(
        group["hyperparameters"]["betas"] == [0.9, 0.999]
        and group["hyperparameters"]["eps"] == 1e-8
        and group["hyperparameters"]["weight_decay"] == 1e-4
        for group in receipt["parameter_groups"]
    )
    assert len(receipt["canonical_json_sha256"]) == 64

    optimizer.state[encoder]["step"] = torch.tensor(0.0)
    with pytest.raises(RuntimeError, match="not empty"):
        module._optimizer_receipt_v1(optimizer, partition)


def test_training_core_guard_binds_exact_public_api_and_first_update_values() -> None:
    module = _load("_test_matched_no_persistence_core_guard")
    training_v1 = importlib.import_module(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1"
    )
    training_v3 = importlib.import_module(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v3_"
        "half_occupied_safety_aux"
    )
    control = importlib.import_module(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v4_"
        "matched_no_persistence"
    )
    module._validate_training_core_v1(training_v1, training_v3, control)
    bad_control = SimpleNamespace(
        **{
            name: getattr(control, name)
            for name in (
                "ACTION_ORDER",
                "MICROBATCH_SIZE",
                "MICROBATCHES_PER_UPDATE",
                "PRESENTATIONS_PER_UPDATE",
                "MAXIMUM_UPDATES",
                "MAXIMUM_PRESENTATIONS",
                "run_fixed_training_v4_matched_no_persistence",
            )
        },
        FIRST_UPDATE_COMPONENT_MEANS={
            **dict(control.FIRST_UPDATE_COMPONENT_MEANS),
            "P_diagnostic": 0.0,
        },
    )
    with pytest.raises(PermissionError, match="first-update witness"):
        module._validate_training_core_v1(training_v1, training_v3, bad_control)


def test_terminal_receipt_requires_exact_order_accounting_and_witness() -> None:
    module = _load("_test_matched_no_persistence_terminal")
    accounting = {
        "updates": 1_000,
        "presentations": 16_000,
        "microbatch_graphs": 4_000,
        "backward_calls": 4_000,
        "optimizer_steps": 1_000,
        "ema_steps": 1_000,
        "predictor_forwards": 4_000,
        "predictor_objectives": 4_000,
    }
    components = dict(module.FIRST_UPDATE_COMPONENT_MEANS)
    losses = {
        **components,
        "L_full_diagnostic": sum(components.values()),
        "L_backward": sum(components[name] for name in ("S", "U", "R", "O")),
    }
    trace = tuple(
        {
            "update": update,
            "presentations": update * 16,
            "losses": dict(losses),
            "gradient_l2": {
                "encoder": 1.0,
                "lift_semantic": 1.0,
                "predictor": 1.0,
            },
        }
        for update in range(1, 1_001)
    )
    diagnostics = {
        "gradient_groups": ["encoder", "lift_semantic", "predictor"],
        "first_update_component_witness": {
            "expected": components,
            "observed": components,
            "exact_match": True,
            "checked_after_backward_calls": 4,
            "checked_before_optimizer_step": True,
        },
    }
    module._validate_terminal_training_receipt_v1(
        accounting, trace, diagnostics
    )
    bad = dict(diagnostics)
    bad["first_update_component_witness"] = {
        **diagnostics["first_update_component_witness"],
        "checked_before_optimizer_step": False,
    }
    with pytest.raises(RuntimeError, match="first-update component witness"):
        module._validate_terminal_training_receipt_v1(accounting, trace, bad)


def test_treatment_predicate_uses_exact_order_seed_index_and_three_checks() -> None:
    module = _load("_test_matched_no_persistence_bootstrap")
    reference = np.asarray(
        module.FULL_V4_REFERENCE_FAMILY_UTILITY[
            "normalized_chosen_prefix_utility"
        ],
        dtype=np.float64,
    )
    deltas = np.asarray([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, -0.01, 0.02])
    control = reference - deltas
    metrics = {
        "families": {
            family: {"normalized_chosen_prefix_utility": float(value)}
            for family, value in zip(
                module.REGISTERED_FAMILIES, control, strict=True
            )
        }
    }
    scene_ids = tuple(f"scene_{index}" for index in range(8) for _ in range(2))
    family_ids = tuple(
        family for family in module.REGISTERED_FAMILIES for _ in range(2)
    )
    result = module.v4_minus_control_treatment_comparison_v1(
        metrics, scene_ids, family_ids, np=np
    )
    observed_deltas = np.asarray(
        result["full_v4_minus_control_delta_vector"], dtype=np.float64
    )
    rng = np.random.default_rng(20_260_728)
    draws = rng.integers(0, 8, size=(10_000, 8))
    expected_lower = float(np.sort(observed_deltas[draws].mean(axis=1))[249])
    assert result["valid"] is True
    assert result["passed"] is True
    assert result["checks"] == {
        "strictly_positive_equal_scene_mean": True,
        "strictly_positive_bootstrap_lower_95": True,
        "at_least_six_positive_families": True,
    }
    assert result["family_order"] == list(module.REGISTERED_FAMILIES)
    assert result["full_v4_minus_control_delta_vector"] == pytest.approx(deltas)
    assert result["equal_scene_mean_delta"] == pytest.approx(float(deltas.mean()))
    assert result["bootstrap"] == {
        "algorithm": "paired_control_comparison_v1",
        "dtype": "float64",
        "seed": 20_260_728,
        "replicates": 10_000,
        "draw_shape": [10_000, 8],
        "lower_95_sorted_zero_based_index": 249,
        "lower_95": expected_lower,
    }
    assert result["positive_family_count"] == 7
    assert result["allowed_positive_conclusion"] == (
        "P improved development selection utility under this fixed "
        "deterministic training schedule."
    )

    negative_control = reference - np.asarray(
        [0.1, 0.1, 0.1, 0.1, 0.1, -0.2, -0.2, -0.2]
    )
    negative_metrics = {
        "families": {
            family: {"normalized_chosen_prefix_utility": float(value)}
            for family, value in zip(
                module.REGISTERED_FAMILIES, negative_control, strict=True
            )
        }
    }
    negative = module.v4_minus_control_treatment_comparison_v1(
        negative_metrics, scene_ids, family_ids, np=np
    )
    assert negative["passed"] is False
    assert negative["positive_family_count"] == 5
    assert negative["allowed_positive_conclusion"] is None


def test_output_is_write_once_and_source_has_one_training_call_no_artifact_load(
    tmp_path: Path,
) -> None:
    module = _load("_test_matched_no_persistence_source")
    output = module._fresh_output_root_v4_matched_no_persistence(tmp_path)
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="matched-no-persistence"):
        module._fresh_output_root_v4_matched_no_persistence(tmp_path)

    source = ENTRYPOINT.read_text()
    assert source.count(
        "training_control.run_fixed_training_v4_matched_no_persistence("
    ) == 1
    assert source.count(
        "model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4("
    ) == 2
    assert "torch.load(" not in source
    assert "execute_v4(" not in source
    assert "run_fixed_training_v3(" not in source
    for forbidden in (
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v1/",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux/",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v3_"
        "half_occupied_safety_aux/",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v4_"
        "residual_local_semantic_decoder/",
    ):
        assert forbidden not in source
    first_model = source.index(
        "model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4("
    )
    second_model = source.index(
        "model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(",
        first_model + 1,
    )
    state_receipt = source.index(
        "_reconstructed_initialization_receipt_v1(", second_model
    )
    device_move = source.index('model = model.to(context["device"])', state_receipt)
    optimizer = source.index("training_v1.build_frozen_optimizer_v1(", device_move)
    optimizer_receipt = source.index("_optimizer_receipt_v1(", optimizer)
    training = source.index(
        "training_control.run_fixed_training_v4_matched_no_persistence(",
        optimizer_receipt,
    )
    checkpoint_write = source.index('output / "checkpoint_update_1000.pt"', training)
    trace_write = source.index('output / "training_trace.json"', checkpoint_write)
    evaluation = source.index("_v1.score_role_v1(", trace_write)
    assert first_model < second_model < state_receipt < device_move
    assert device_move < optimizer < optimizer_receipt < training
    assert training < checkpoint_write < trace_write < evaluation
    assert '"predecessor_experiment_checkpoint_read": False' in source
    assert '"heldout_or_sealed_opened": False' in source
    assert '"retry_or_resume_authorized": False' in source
    assert '"replacement_or_warm_start_authorized": False' in source
