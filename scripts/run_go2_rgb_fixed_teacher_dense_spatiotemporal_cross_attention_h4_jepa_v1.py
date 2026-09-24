#!/usr/bin/env python3
"""Run the one fresh RGB fixed-teacher dense cross-attention H4 JEPA probe."""
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
from scripts import (  # noqa: E402
    run_go2_recurrent_h4_fixed_teacher_delta_joint_jepa_v3 as v3_runner,
)


CORE_SOURCE = ROOT / "scripts/run_go2_recurrent_h4_joint_jepa_v1.py"
CORE_SOURCE_SHA256 = "fc35d535e1c07b56c667474e6e10c5c7587fa01e627567148c022331983616fc"
CORE_SOURCE_BYTES = 70_301
V3_GATE_SOURCE = (
    ROOT / "scripts/run_go2_recurrent_h4_fixed_teacher_delta_joint_jepa_v3.py"
)
V3_GATE_SOURCE_SHA256 = (
    "6347adaaae9f66236b3988960ac45f8923d39a0be232572adb6531d401c63368"
)
V3_GATE_SOURCE_BYTES = 13_691
MODEL_MODULE = (
    "lewm.models."
    "go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1"
)
MODEL_SOURCE = ROOT / (
    "lewm/models/"
    "go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1.py"
)
MODEL_SOURCE_SHA256 = "5c74675b93667e6035fc21c9fe497880ba4bff22641b3e735272e4cc1ede3d30"
MODEL_SOURCE_BYTES = 23_712
BASE_MODEL_SOURCE = ROOT / "lewm/models/go2_recurrent_h4_joint_jepa.py"
BASE_MODEL_SOURCE_SHA256 = "ddd84561aba5a36df1255ab942bb29db943cc1bf7b0e496ae41b3d1cdc218f55"
BASE_MODEL_SOURCE_BYTES = 21_166
ENCODER_SOURCE = ROOT / "lewm/models/encoders.py"
ENCODER_SOURCE_SHA256 = "5eed7bbe424d5ddd293ea67ed1596e74504c68dd8da93f8420795f216cb7599d"
ENCODER_SOURCE_BYTES = 7_028
OUTPUT_ROOT = ROOT / (
    ".generated/"
    "go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1/"
    "probe_v1"
)
SCHEMA = (
    "lewm_go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1"
)
PASS_DECISION = (
    "PASS_MAIN_POOL_RGB_FIXED_TEACHER_DENSE_SPATIOTEMPORAL_CROSS_ATTENTION_"
    "H4_JEPA_V1"
)
STOP_DECISION = (
    "STOP_MAIN_POOL_RGB_FIXED_TEACHER_DENSE_SPATIOTEMPORAL_CROSS_ATTENTION_"
    "H4_JEPA_V1"
)
_CORE_RUN = core._run


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
    wrapper_sha256 = os.environ.get("LEWM_DENSE_H4_WRAPPER_SHA256", "")
    wrapper_bytes_text = os.environ.get("LEWM_DENSE_H4_WRAPPER_BYTES", "")
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError(
            "external dense-H4 wrapper byte binding is required"
        ) from error
    return {
        "dense_h4_wrapper": _source_binding(
            Path(__file__).resolve(),
            wrapper_sha256,
            wrapper_bytes,
        ),
        "shared_runner": _source_binding(
            CORE_SOURCE,
            CORE_SOURCE_SHA256,
            CORE_SOURCE_BYTES,
        ),
        "v3_gate_source": _source_binding(
            V3_GATE_SOURCE,
            V3_GATE_SOURCE_SHA256,
            V3_GATE_SOURCE_BYTES,
        ),
        "dense_h4_model": _source_binding(
            MODEL_SOURCE,
            MODEL_SOURCE_SHA256,
            MODEL_SOURCE_BYTES,
        ),
        "inherited_v1_model": _source_binding(
            BASE_MODEL_SOURCE,
            BASE_MODEL_SOURCE_SHA256,
            BASE_MODEL_SOURCE_BYTES,
        ),
        "encoder_dependency": _source_binding(
            ENCODER_SOURCE,
            ENCODER_SOURCE_SHA256,
            ENCODER_SOURCE_BYTES,
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
        "all_other_training_terms_absent"
    )
    core.ADDITIONAL_SCIENCE = {
        "teacher": "accepted_N320_fixed_for_entire_probe",
        "online_components": (
            "encoder+dense_history+action_path+cross_attention_predictor_"
            "jointly_trained"
        ),
        "history": (
            "770_interleaved_tokens_from_normalized_e0_p0_e1_p1_e2_with_"
            "learned_spatial_time_and_transition_step_embeddings"
        ),
        "history_encoder": (
            "exactly_two_independently_initialized_prenorm_transformer_blocks"
        ),
        "belief": "raw_online_e2_persistence_anchor_concat_770_token_context",
        "future_action": (
            "ordered_fixed_four_slot_zero_suffix_prefix_mlp_per_horizon"
        ),
        "future_queries": (
            "normalized_e2_plus_shared_spatial_plus_horizon_plus_action_prefix"
        ),
        "future_decoder": (
            "four_independent_horizons_share_exactly_two_prenorm_decoder_blocks"
        ),
        "prediction": "direct_nonrecursive_per_horizon_delta_from_raw_e2",
        "delta_head_initialization": "sole_final_linear_weight_and_bias_zero",
        "training_losses": [
            "raw_fixed_teacher_future_minus_e2_delta",
            "three_frame_online_to_fixed_teacher_alignment",
        ],
        "evaluation_only_controls": [
            "wrong_action",
            "hold_action",
            "persistence",
            "reordered_or_reset_history",
        ],
        "recurrent_module_count": 0,
        "model_import": "bound_namespace_stubs_no_package_initializer_execution",
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
    }
    core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER = 0
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _dense_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    if v3_runner.core is not core:
        raise core.ContractError("V3 gate source and dense wrapper use different cores")
    result = dict(v3_runner._v3_decision(observations, updates_completed))
    prior_decision = result.get("decision")
    if prior_decision == v3_runner.PASS_DECISION:
        result["decision"] = PASS_DECISION
    elif prior_decision == v3_runner.STOP_DECISION:
        result["decision"] = STOP_DECISION
    else:
        raise core.ContractError("V3 gate source returned an unknown decision")
    result["authority"] = (
        "A pass establishes bounded development RGB/action dense-attention JEPA "
        "substrate feasibility only; it grants no navigation, held-out, promotion, "
        "or deployment authority. A stop closes this dense cross-attention "
        "mechanism and deterministic dense-H4 predictor variants."
    )
    return result


def _dense_run(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], ...]:
    metrics, artifact, decision = _CORE_RUN(*args, **kwargs)
    if artifact.get("fresh_recurrent_and_predictor_initialization") is not True:
        raise core.ContractError("shared runner initialization receipt changed")
    dense_artifact = dict(artifact)
    del dense_artifact["fresh_recurrent_and_predictor_initialization"]
    dense_artifact[
        "fresh_dense_attention_embeddings_action_path_and_delta_head_initialization"
    ] = True
    return metrics, dense_artifact, decision


def _install_dense_run_adapter() -> None:
    if core._run is _dense_run:
        return
    if core._run is not _CORE_RUN:
        raise core.ContractError("shared runner was modified before dense adapter install")
    core._run = _dense_run


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    if Path(v3_runner.__file__).resolve() != V3_GATE_SOURCE:
        raise core.ContractError("V3 gate source imported from an unexpected path")
    source_bindings = _verify_source_closure()
    _install_bound_model_package_stubs()
    _configure_core(source_bindings)
    core._decision = _dense_decision
    _install_dense_run_adapter()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
