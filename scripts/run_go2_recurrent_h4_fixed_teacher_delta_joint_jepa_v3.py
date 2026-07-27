#!/usr/bin/env python3
"""Run the one fresh fixed-teacher latent-delta recurrent-H4 JEPA probe."""
from __future__ import annotations

import os
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_go2_recurrent_h4_joint_jepa_v1 as core  # noqa: E402


CORE_SOURCE = ROOT / "scripts/run_go2_recurrent_h4_joint_jepa_v1.py"
CORE_SOURCE_SHA256 = "fc35d535e1c07b56c667474e6e10c5c7587fa01e627567148c022331983616fc"
CORE_SOURCE_BYTES = 70_301
MODEL_MODULE = "lewm.models.go2_recurrent_h4_fixed_teacher_delta_joint_jepa_v3"
MODEL_SOURCE = (
    ROOT / "lewm/models/go2_recurrent_h4_fixed_teacher_delta_joint_jepa_v3.py"
)
MODEL_SOURCE_SHA256 = "16cd3f25e3cc6b81787b5699c63e9fd180a19a9fcfd9dcc6fd01b0ee810a015c"
MODEL_SOURCE_BYTES = 13_542
BASE_MODEL_SOURCE = ROOT / "lewm/models/go2_recurrent_h4_joint_jepa.py"
BASE_MODEL_SOURCE_SHA256 = "ddd84561aba5a36df1255ab942bb29db943cc1bf7b0e496ae41b3d1cdc218f55"
BASE_MODEL_SOURCE_BYTES = 21_166
ENCODER_SOURCE = ROOT / "lewm/models/encoders.py"
ENCODER_SOURCE_SHA256 = "5eed7bbe424d5ddd293ea67ed1596e74504c68dd8da93f8420795f216cb7599d"
ENCODER_SOURCE_BYTES = 7_028
OUTPUT_ROOT = (
    ROOT / ".generated/go2_recurrent_h4_fixed_teacher_delta_joint_jepa_v3/probe_v1"
)
SCHEMA = "lewm_go2_recurrent_h4_fixed_teacher_delta_joint_jepa_v3"
PASS_DECISION = "PASS_MAIN_POOL_RECURRENT_H4_FIXED_TEACHER_DELTA_JOINT_JEPA_V3_PROBE"
STOP_DECISION = "STOP_MAIN_POOL_RECURRENT_H4_FIXED_TEACHER_DELTA_JOINT_JEPA_V3_PROBE"
_GEOMETRY_TOLERANCE = 1e-6
_INITIAL_GAP_TOLERANCE = 1e-5


def _source_binding(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    core._read_regular_bound(
        path,
        expected_sha256=sha256,
        expected_bytes=byte_count,
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256,
        "byte_count": byte_count,
    }


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    wrapper_sha256 = os.environ.get("LEWM_V3_WRAPPER_SHA256", "")
    wrapper_bytes_text = os.environ.get("LEWM_V3_WRAPPER_BYTES", "")
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError("external V3 wrapper byte binding is required") from error
    return {
        "v3_wrapper": _source_binding(
            Path(__file__).resolve(), wrapper_sha256, wrapper_bytes
        ),
        "shared_runner": _source_binding(
            CORE_SOURCE, CORE_SOURCE_SHA256, CORE_SOURCE_BYTES
        ),
        "v3_model": _source_binding(
            MODEL_SOURCE, MODEL_SOURCE_SHA256, MODEL_SOURCE_BYTES
        ),
        "inherited_v1_model": _source_binding(
            BASE_MODEL_SOURCE, BASE_MODEL_SOURCE_SHA256, BASE_MODEL_SOURCE_BYTES
        ),
        "encoder_dependency": _source_binding(
            ENCODER_SOURCE, ENCODER_SOURCE_SHA256, ENCODER_SOURCE_BYTES
        ),
    }


def _install_bound_model_package_stubs() -> None:
    """Provide package paths without executing unbound package initializers."""

    if any(name == "lewm" or name.startswith("lewm.") for name in sys.modules):
        raise core.ContractError("a lewm package was imported before bound model loading")

    def package(name: str, path: Path) -> ModuleType:
        module = ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        spec = ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        module.__spec__ = spec
        return module

    lewm_package = package("lewm", ROOT / "lewm")
    models_package = package("lewm.models", ROOT / "lewm/models")
    lewm_package.models = models_package
    sys.modules["lewm"] = lewm_package
    sys.modules["lewm.models"] = models_package


def _configure_core(source_bindings: Mapping[str, Mapping[str, Any]]) -> None:
    core.MODEL_MODULE = MODEL_MODULE
    core.MODEL_SOURCE = MODEL_SOURCE
    core.MODEL_SOURCE_SHA256 = MODEL_SOURCE_SHA256
    core.MODEL_SOURCE_BYTES = MODEL_SOURCE_BYTES
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.SCHEMA = SCHEMA
    core.PASS_DECISION = PASS_DECISION
    core.STOP_DECISION = STOP_DECISION
    core.PREDICTION_WEIGHT = 0.0
    core.VARIANCE_WEIGHT = 0.0
    core.ACTION_RANKING_WEIGHT = 0.0
    core.TRAIN_WRONG_ACTION_CONTRAST = False
    core.UPDATE_TARGET_EMA = False
    core.TARGET_DESCRIPTION = "fixed_accepted_N320_teacher_stop_gradient_no_ema"
    core.OBJECTIVE_DESCRIPTION = (
        "1.0*raw_fixed_teacher_future_delta + "
        "1.0*three_frame_online_to_fixed_teacher_alignment; "
        "absolute_prediction_variance_and_synthetic_rankings_weight_0"
    )
    core.ADDITIONAL_SCIENCE = {
        "teacher": "accepted_N320_fixed_for_entire_probe",
        "online_components": "encoder+ordered_history+action_predictor_jointly_trained",
        "belief": "hard_online_e2_anchor_concat_ordered_recurrent_context",
        "future": "direct_nonrecursive_per_horizon_delta_from_e2",
        "delta_head_initialization": "sole_final_linear_weight_and_bias_zero",
        "training_losses": [
            "raw_fixed_teacher_future_minus_e2_delta",
            "three_frame_online_to_fixed_teacher_alignment",
        ],
        "evaluation_only_controls": [
            "wrong_action",
            "hold_action",
            "persistence",
            "reversed_or_reset_history",
        ],
        "model_import": "bound_namespace_stubs_no_package_initializer_execution",
        "v1_checkpoint_tensor_open_count": 0,
        "v2_checkpoint_tensor_open_count": 0,
    }
    core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER = 0
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _v3_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    baseline = next(item for item in observations if item["update"] == 0)
    baseline_noncollapse = baseline["noncollapse"]
    target_rank = baseline_noncollapse["target_effective_rank_ratio"]
    target_near_zero = baseline_noncollapse["target_near_zero_variance_fraction"]
    target_rank_drift = max(
        abs(item["noncollapse"]["target_effective_rank_ratio"] - target_rank)
        for item in observations
    )
    target_near_zero_drift = max(
        abs(
            item["noncollapse"]["target_near_zero_variance_fraction"]
            - target_near_zero
        )
        for item in observations
    )
    initial_gap_values = (
        baseline["aggregate"]["action_gap"]
        + baseline["aggregate"]["hold_gap"]
        + baseline["aggregate"]["persistence_gap"]
        + baseline["aggregate"]["history_gap"]
    )
    candidates = [
        item
        for item in observations
        if item["update"] > 0 and core._noncollapsed(item)
    ]
    selected = (
        min(
            candidates,
            key=lambda item: sum(item["aggregate"]["real_normalized_error"])
            / 4.0,
        )
        if candidates
        else None
    )
    gates: dict[str, bool] = {
        "completed_exact_cap": updates_completed == core.UPDATES,
        "all_observations_finite": all(
            bool(item["all_registered_values_finite"]) for item in observations
        ),
        "fixed_teacher_metric_geometry_unchanged": (
            target_rank_drift <= _GEOMETRY_TOLERANCE
            and target_near_zero_drift <= _GEOMETRY_TOLERANCE
        ),
        "fixed_teacher_rank_floor_all_observations": all(
            item["noncollapse"]["target_effective_rank_ratio"] >= 0.10
            and item["noncollapse"]["target_near_zero_variance_fraction"] <= 0.05
            for item in observations
        ),
        "update_zero_is_exact_persistence": (
            max(abs(value) for value in initial_gap_values)
            <= _INITIAL_GAP_TOLERANCE
        ),
        "eligible_noncollapsed_checkpoint_exists": selected is not None,
    }
    selected_gate_names = (
        "h4_real_error_improved_ten_percent",
        "h1_h3_real_errors_all_improved",
        "all_horizon_persistence_gaps_positive",
        "h4_persistence_gap_at_least_point10",
        "h4_persistence_gap_bootstrap_lower_positive",
        "persistence_positive_in_six_families",
        "no_family_persistence_gap_below_minus_point02",
        "h4_action_gap_at_least_point05",
        "h4_action_gap_bootstrap_lower_positive",
        "h1_h3_action_gaps_nonnegative",
        "action_positive_in_six_families",
        "no_family_action_gap_below_minus_point02",
        "h4_history_gap_at_least_point03",
        "h4_history_gap_bootstrap_lower_positive",
        "history_positive_in_six_families",
        "h4_hold_gap_positive",
    )
    diagnostics: dict[str, Any] = {
        "selected_update": None,
        "selected_presentations": None,
        "fixed_teacher_target_rank_drift": target_rank_drift,
        "fixed_teacher_target_near_zero_drift": target_near_zero_drift,
    }
    if selected is None:
        gates.update({name: False for name in selected_gate_names})
    else:
        base_real = baseline["aggregate"]["real_normalized_error"]
        real = selected["aggregate"]["real_normalized_error"]
        action = selected["aggregate"]["action_gap"]
        hold = selected["aggregate"]["hold_gap"]
        persistence = selected["aggregate"]["persistence_gap"]
        history = selected["aggregate"]["history_gap"]
        action_positive_families = sum(
            selected["family"][family]["action_gap"][3] > 0
            for family in core.FAMILIES
        )
        persistence_positive_families = sum(
            selected["family"][family]["persistence_gap"][3] > 0
            for family in core.FAMILIES
        )
        history_positive_families = sum(
            selected["family"][family]["history_gap"][3] > 0
            for family in core.FAMILIES
        )
        h4_improvement = (base_real[3] - real[3]) / max(abs(base_real[3]), 1e-8)
        gates.update(
            {
                "h4_real_error_improved_ten_percent": h4_improvement >= 0.10,
                "h1_h3_real_errors_all_improved": all(
                    real[index] < base_real[index] for index in range(3)
                ),
                "all_horizon_persistence_gaps_positive": all(
                    value > 0 for value in persistence
                ),
                "h4_persistence_gap_at_least_point10": persistence[3] >= 0.10,
                "h4_persistence_gap_bootstrap_lower_positive": selected[
                    "bootstrap_lower_95"
                ]["persistence_gap_h4"]
                > 0,
                "persistence_positive_in_six_families": (
                    persistence_positive_families >= 6
                ),
                "no_family_persistence_gap_below_minus_point02": min(
                    selected["family"][family]["persistence_gap"][3]
                    for family in core.FAMILIES
                )
                >= -0.02,
                "h4_action_gap_at_least_point05": action[3] >= 0.05,
                "h4_action_gap_bootstrap_lower_positive": selected[
                    "bootstrap_lower_95"
                ]["action_gap_h4"]
                > 0,
                "h1_h3_action_gaps_nonnegative": all(
                    value >= 0 for value in action[:3]
                ),
                "action_positive_in_six_families": action_positive_families >= 6,
                "no_family_action_gap_below_minus_point02": min(
                    selected["family"][family]["action_gap"][3]
                    for family in core.FAMILIES
                )
                >= -0.02,
                "h4_history_gap_at_least_point03": history[3] >= 0.03,
                "h4_history_gap_bootstrap_lower_positive": selected[
                    "bootstrap_lower_95"
                ]["history_gap_h4"]
                > 0,
                "history_positive_in_six_families": history_positive_families >= 6,
                "h4_hold_gap_positive": hold[3] > 0,
            }
        )
        diagnostics.update(
            {
                "selected_update": selected["update"],
                "selected_presentations": selected["presentations"],
                "h4_real_error_fractional_improvement": h4_improvement,
                "action_positive_family_count": action_positive_families,
                "history_positive_family_count": history_positive_families,
                "persistence_positive_family_count": persistence_positive_families,
            }
        )
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    return {
        "decision": PASS_DECISION if not failed_gates else STOP_DECISION,
        "gates": gates,
        "failed_gates": failed_gates,
        "diagnostics": diagnostics,
        "authority": (
            "A pass establishes bounded train/validation RGB fixed-teacher delta-JEPA "
            "substrate feasibility only; it does not authorize navigation, held-out "
            "access, promotion, or deployment. A stop ends this recurrent-H4 branch."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    source_bindings = _verify_source_closure()
    _install_bound_model_package_stubs()
    _configure_core(source_bindings)
    core._decision = _v3_decision
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
