"""Source-only contract for Action-Query Spatial Successor joint-JEPA V1.

The frozen geometry-anchored V3 contract remains authoritative for the data,
schedule, runtime interpreter, narrow loader, custody boundary, and receipt
helpers.  This module replaces only the model, objective, gates, accounting,
source closure, and one-shot authority identities preregistered for V1.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V3_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)


def _source_module(name: str, relative: str) -> Any:
    source = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_v3 = _source_module(
    "_lewm_action_query_spatial_successor_frozen_v3_contract",
    FROZEN_V3_CONTRACT_RELATIVE_PATH,
)
for _name in _v3.__all__:
    globals()[_name] = getattr(_v3, _name)

_read_regular_source = _v3._read_regular_source
_source_freeze_commit = _v3._source_freeze_commit
_validate_artifact_binding = _v3._validate_artifact_binding


IMPLEMENTATION_AUTHORS = (
    "/root",
    "/root/action_query_model_impl",
    "/root/action_query_runner_impl",
    "/root/action_query_contract_impl2",
)
SCHEMA_PREFIX = (
    "lewm_go2_rgb_geometry_anchored_action_query_spatial_successor_joint_jepa_v1"
)
EXPERIMENT_ID = "geometry_anchored_action_query_spatial_successor_joint_jepa_v1"

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_action_query_spatial_successor_joint_"
    "jepa_v1_preregistration_2026-07-27.md"
)
PREREGISTRATION_COMMIT = "af2b6513cc587c3257bc56a2ef840b36b101d78a"
PREREGISTRATION_FILE_SHA256 = (
    "a5b14ba2470b2e1a8311bc751d7e0ce76b92836e92c6bd82eb6b1d354a813bac"
)
PREREGISTRATION_BYTE_COUNT = 26_850

STANDING_SCOPE_AUTHORITY = {
    "path": (
        "docs/lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_v10r_"
        "integrity_replacement_preregistration_2026-07-26.json"
    ),
    "commit": "bdf30305645efbcde56c7e52711e2ded7bf728fb",
    "file_sha256": "38e3f4d9378d4974f77b4a10b069a704b6722caea31bd97f237f0eac00f2308a",
    "content_sha256": "4100001b5217091bea6b917057eb33cb9331b77c47dd24468c036d5535e8d97e",
    "byte_count": 16_613,
}

FROZEN_V3_SOURCE_COMMIT = "ebcde189628b1a7040ffaf95aafaf9fd8f404fc4"
FROZEN_V3_SOURCE_COUNT = 84
FROZEN_V3_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _v3.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)
FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH = _v3.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V3_SOURCE_MANIFEST_FILE_SHA256 = (
    "d7139af8e6de2e1beae935a9d0de814bad395cdbc0b6190131f143c85b985bf0"
)
FROZEN_V3_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "4b5c30274bba178eca50b0a04185cafd2efe2226149f084a104d0bbac89d8512"
)
FROZEN_V3_SOURCE_BINDINGS_SHA256 = (
    "5931f3083b9996d037aab46f815721a5cb98d1b9368c3d6d4c5fb67764967d3a"
)
FROZEN_V3_SOURCE_MANIFEST_BYTE_COUNT = 27_898

PRIOR_PUBLIC_TERMINAL_AUDITS = (
    {
        "name": "deformable_bev_joint_jepa_v3",
        "path": (
            "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_"
            "jepa_v3_scalar_tensor_state_hash_integrity_replacement_terminal_"
            "audit_2026-07-27.json"
        ),
        "commit": "6b48b528c53766276f4912626728611910837a92",
        "file_sha256": "bbb1d82faefc62c0358df531941ab07f2b3253d274eca2156df378ffb17a52c4",
        "content_sha256": "595ac5198edfcba196ced8213c3f83ff9a5fa2c8100231b028bb99690c8a5d2b",
        "byte_count": 10_661,
    },
    {
        "name": "global_action_indexed_rigid_bev_transport_v1",
        "path": (
            "docs/lewm_go2_rgb_geometry_anchored_global_action_indexed_rigid_"
            "bev_transport_joint_jepa_v1_terminal_audit_2026-07-27.json"
        ),
        "commit": "293a02e1cc99c3b5aac876efc68104093468506c",
        "file_sha256": "38d65b46bd4ff83ab67924233badb96c37bd079f0b85b36c69512c905557f25e",
        "content_sha256": "5a87a441560c8ab397a0cdfab5ca08fc1c7ad7b8294d323194f712ad834821ca",
        "byte_count": 19_570,
    },
    {
        "name": "fixed_two_mode_event_delta_v2",
        "path": (
            "docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_"
            "jepa_v2_runtime_delegation_integrity_replacement_terminal_audit_"
            "2026-07-27.json"
        ),
        "commit": "2a8561617ace1e47f769e8849238ad0b20bc32dd",
        "file_sha256": "4fd36445ebad3db5d568dab2444eeb4350ae698f0ec54f476e35242f175d2096",
        "content_sha256": "ba8a089e86dcc5ebad69d924b413f9476d0e3089d3321e7b22764946f66420a5",
        "byte_count": 21_027,
    },
    {
        "name": "patch_whitened_action_residual_v4",
        "path": (
            "docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v4_action_"
            "indexed_energy_nll_terminal_audit_2026-07-25.json"
        ),
        "commit": "20a5099f17a6da17bb2858d96724f9f8e88ae3f9",
        "file_sha256": "ddb3c784382f92161b82d7321c8ad3c70901cb8d5a813c3ecc7153083480d809",
        "content_sha256": "c3edbe1932c5647e576b25216cee38ad904f5b5fa581d39f70c1d8cef3e92f01",
        "byte_count": 15_366,
    },
    {
        "name": "local_correspondence_all_candidate_v8",
        "path": (
            "docs/lewm_go2_rgb_action_conditioned_local_correspondence_all_"
            "candidate_identification_jepa_v8_terminal_audit_2026-07-25.json"
        ),
        "commit": "9f3e2bc96a6e4ea419574f109c890299d0608659",
        "file_sha256": "3ea4a8cc4405b0880d2e05217e4b4acefc5b9df5fad9bcdd9a682db42e273173",
        "content_sha256": "ff8339aa6109933e85d60ad118dc912fd091dddf7dfd80b18d00453ce7c01367",
        "byte_count": 20_028,
    },
    {
        "name": "next_target_retrieval_v10r",
        "path": (
            "docs/lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
            "v10r_integrity_replacement_terminal_audit_2026-07-26.json"
        ),
        "commit": "79d6de74b795065f7a5a47b32f1a56fc4fd4580a",
        "file_sha256": "8cd27a7d21e9ce1875d322cad2ea5aae8a846a301247f774d4da86074ebd28a5",
        "content_sha256": "ab6b9d9ad3b6de1462fe142c42e18bf30751cc483aceaa9fabb632b2999cca73",
        "byte_count": 12_862,
    },
    {
        "name": "masked_current_next_pair_tubelet_v11",
        "path": (
            "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_"
            "terminal_audit_2026-07-26.json"
        ),
        "commit": "4d3e967f1d30bc3843626a9b5aaecd79e6f1dca0",
        "file_sha256": "89ac1155e7108118133d6eb0648437e3a337f03e31c6c93e6ca63cc590f27044",
        "content_sha256": "9641274f58e84b4a3c3603f7cf19714e006ec27d062d57a0f24f0bb38677aec9",
        "byte_count": 7_876,
    },
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_action_query_spatial_successor_"
    "joint_jepa_v1.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/geometry_anchored_action_query_spatial_successor_joint_jepa_v1.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_action_query_spatial_successor_joint_"
    "jepa_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_action_query_spatial_successor_joint_"
    "jepa_v1.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_action_query_spatial_successor_joint_"
    "jepa_v1_source_closure.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_geometry_anchored_action_query_spatial_successor_joint_jepa_v1.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_geometry_anchored_action_query_spatial_successor_joint_"
    "jepa_v1_contract_runner.py"
)
TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_action_query_spatial_successor_joint_"
    "jepa_v1_source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_action_query_spatial_successor_joint_"
    "jepa_v1_source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_action_query_spatial_successor_joint_"
    "jepa_v1_execution_authorization_2026-07-27.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_geometry_anchored_action_query_spatial_successor_joint_"
    "jepa_v1/attempt_v1"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(set(_v3.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != 84 or len(SOURCE_PATHS) != 91:
    raise PermissionError("Action-Query V1 must be frozen V3 84 plus seven files")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_metrics_v1"
ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_artifact_manifest_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_receipt_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

MODEL_CLASS_NAME = "GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1"
PREDICTOR_PARAMETER_COUNT = 504_384
PREDICTOR_PARAMETER_TENSOR_COUNT = 34
PREDICTOR_ORDERED_PARAMETER_NAMES = tuple(
    f"predictor.{name}" for name in (
        "future_queries",
        "current_downsampler.weight", "current_downsampler.bias",
        "action_embedding.weight",
        "blocks.0.query_norm.weight", "blocks.0.query_norm.bias",
        "blocks.0.memory_norm.weight", "blocks.0.memory_norm.bias",
        "blocks.0.attention.in_proj_weight", "blocks.0.attention.in_proj_bias",
        "blocks.0.attention.out_proj.weight", "blocks.0.attention.out_proj.bias",
        "blocks.0.ffn_norm.weight", "blocks.0.ffn_norm.bias",
        "blocks.0.linear1.weight", "blocks.0.linear1.bias",
        "blocks.0.linear2.weight", "blocks.0.linear2.bias",
        "blocks.1.query_norm.weight", "blocks.1.query_norm.bias",
        "blocks.1.memory_norm.weight", "blocks.1.memory_norm.bias",
        "blocks.1.attention.in_proj_weight", "blocks.1.attention.in_proj_bias",
        "blocks.1.attention.out_proj.weight", "blocks.1.attention.out_proj.bias",
        "blocks.1.ffn_norm.weight", "blocks.1.ffn_norm.bias",
        "blocks.1.linear1.weight", "blocks.1.linear1.bias",
        "blocks.1.linear2.weight", "blocks.1.linear2.bias",
        "output_head.weight", "output_head.bias",
    )
)

EXECUTION_AUTHORITY = {
    **{
        key: value for key, value in _v3.EXECUTION_AUTHORITY.items()
        if key not in {
            "output_root", "science_identical_runtime_import_integrity_replacement_only",
            "science_identical_scalar_tensor_state_hash_integrity_replacement_only",
            "v1_retry_authorized", "v2_retry_authorized",
            "v2_resume_or_state_reuse_authorized",
        }
    },
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "fresh_materially_different_action_query_mechanism_only": True,
    "predecessor_checkpoint_state_or_runtime_output_reuse_authorized": False,
}
SOURCE_ONLY_AUTHORITY = {
    **_v3.SOURCE_ONLY_AUTHORITY,
    "cpu_synthetic_action_query_model_and_joint_u1_route_tests_authorized": True,
}
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_RGB_GEOMETRY_ANCHORED_ACTION_QUERY_SPATIAL_"
    "SUCCESSOR_JOINT_JEPA_V1_ATTEMPT"
)
OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_RGB_GEOMETRY_ANCHORED_ACTION_QUERY_SPATIAL_SUCCESSOR_JOINT_"
    "JEPA_V1_OPERATIONAL_FAILURE_NO_RETRY"
)
CONTROL_FAIL_OPERATIONAL = OPERATIONAL_FAILURE_STATUS


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def frozen_v3_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V3_SOURCE_COMMIT,
        "file_sha256": FROZEN_V3_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V3_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V3_SOURCE_MANIFEST_BYTE_COUNT,
        "source_bindings_sha256": FROZEN_V3_SOURCE_BINDINGS_SHA256,
        "source_count": FROZEN_V3_SOURCE_COUNT,
    }


def prior_public_terminal_audit_bindings() -> list[dict[str, Any]]:
    return [dict(binding) for binding in PRIOR_PUBLIC_TERMINAL_AUDITS]


def model_config() -> dict[str, Any]:
    value = copy.deepcopy(_v3.model_config())
    value["predictor"] = {
        "name": "Action_Query_Spatial_Successor_V1",
        "input": "per_cell_channel_normalized_online_current_BEV_plus_action_index",
        "current_downsampler": "Conv2d_64_128_k4_s4_p0_bias",
        "token_shape": [256, 128],
        "fixed_position_encoding": "2D_sine_cosine_row_column_16x16_width128",
        "action_tokens": "Embedding_9_128_distinct_xavier_uniform",
        "future_queries": "learned_256x128_xavier_uniform",
        "future_query_blocks": 2,
        "block": "separate_pre_norm_cross_attention_4_heads_plus_128_256_128_GELU_MLP",
        "memory": "immutable_action_token_plus_current_tokens_and_position",
        "output": "bilinear_x4_align_corners_false_then_Conv2d_128_64_k3_p1_bias",
        "successor": "normalized_current_latent_plus_continuous_action_residual",
        "all_nine_actions_one_vectorized_call": True,
        "parameter_count": PREDICTOR_PARAMETER_COUNT,
        "parameter_tensor_count": PREDICTOR_PARAMETER_TENSOR_COUNT,
        "ordered_parameter_names": list(PREDICTOR_ORDERED_PARAMETER_NAMES),
        "warp_flow_transport_event_mode_inverse_classifier_or_future_input": False,
    }
    value["inference_inputs"] = ["single_current_RGB", "commanded_action"]
    return value


def objective_contract() -> dict[str, Any]:
    return {
        "latent_normalization": "per_cell_non_affine_channel_LayerNorm_64_eps1e-5",
        "local_energy": "avgpool4(mean_channel(smooth_L1_beta1))",
        "smooth_spatial_soft_min": "-0.25*(logsumexp(-v/0.25,token)-log(256))",
        "detached_scale_floor": 1e-3,
        "semantic_S": "(0.5*A_current+0.5*A_next)/log(3)",
        "P_successor": "mean(executed_positive_local_energy)",
        "R_local_action": "mean_b(SSM(local_action_CE/log(9)))",
        "C_deranged": "mean_b(SSM(local_correct_deranged_CE/log(2)))",
        "total": "S+P_successor+R_local_action+C_deranged",
        "component_weights": {
            "S": 1.0, "P_successor": 1.0,
            "R_local_action": 1.0, "C_deranged": 1.0,
        },
        "joint_from_update": 1,
        "perception_only_or_predictor_only_phase": False,
        "combined_backward_divisor": 4,
        "update_one_route_probes": {
            "semantic_autograd_grad_calls": 4,
            "combined_dynamics_autograd_grad_calls": 4,
            "abort_on_finite_ratio": False,
        },
    }


def optimizer_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v3.optimizer_contract())
    value["joint_from_update_one"] = True
    value["rebuilt_or_reinitialized_after_update_zero"] = False
    value["gradient_routes"] = {
        "representation_and_semantic_clip_l2": 1.0,
        "complete_predictor_clip_l2": 1.0,
        "combined_backward_divisor": 4,
    }
    return value


def build_schedule_identity() -> dict[str, Any]:
    value = copy.deepcopy(_v3.build_schedule_identity())
    value["preserve_rows_roles_order_labels_actions_endpoints_and_mappings"] = True
    return value


def runtime_authorization_template() -> dict[str, Any]:
    return copy.deepcopy(_v3.runtime_authorization_template())


GATE_THRESHOLDS = {
    100: {
        "semantic_balanced_accuracy_minimum": 0.60,
        "free_recall_minimum": 0.55,
        "occupied_recall_minimum": 0.30,
        "free_occupied_gap_maximum": 0.50,
        "paired_rgb_family_wins_minimum": 6,
        "action_nll_strict_maximum": math.log(9.0),
        "action_macro_balanced_accuracy_strict_minimum": 1.0 / 9.0,
        "hardest_wrong_family_wins_minimum": 1,
        "correct_deranged_nll_strict_maximum": math.log(2.0),
        "correct_deranged_win_rate_strict_minimum": 0.50,
    },
    400: {
        "semantic_balanced_accuracy_minimum": 0.80,
        "occupied_recall_minimum": 0.60,
        "rough_occupied_recall_minimum": 0.55,
        "paired_rgb_family_wins_minimum": 6,
        "action_nll_strict_maximum": 0.98 * math.log(9.0),
        "action_macro_balanced_accuracy_minimum": 0.18,
        "hardest_wrong_family_wins_minimum": 3,
        "correct_deranged_win_rate_minimum": 0.70,
        "anti_collapse_retention_minimum": 0.75,
    },
    1_000: {
        "raster_nll_maximum": 0.38,
        "semantic_balanced_accuracy_minimum": 0.80,
        "unknown_recall_minimum": 0.80,
        "free_recall_minimum": 0.75,
        "occupied_recall_minimum": 0.70,
        "free_occupied_gap_maximum": 0.25,
        "rough_balanced_accuracy_minimum": 0.772,
        "rough_occupied_recall_minimum": 0.65,
        "action_nll_strict_maximum": 0.95 * math.log(9.0),
        "action_macro_balanced_accuracy_strict_minimum": 2.0 / 9.0,
        "family_count_minimum": 6,
        "correct_deranged_nll_strict_maximum": 0.95 * math.log(2.0),
        "correct_deranged_win_rate_minimum": 0.70,
        "successor_persistence_factor_maximum": 0.90,
        "anti_collapse_retention_minimum": 0.75,
    },
}

GATE_CONTROLS = {
    0: (
        "FAIL_UPDATE_ZERO_ACTION_QUERY_SPATIAL_SUCCESSOR_JOINT_JEPA_V1_STRUCTURAL_GATE_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_ZERO_ACTION_QUERY_SPATIAL_SUCCESSOR_JOINT_JEPA_V1",
    ),
    100: (
        "FAIL_UPDATE_100_ACTION_QUERY_SPATIAL_SUCCESSOR_JOINT_JEPA_V1_GATE_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_100_ACTION_QUERY_SPATIAL_SUCCESSOR_JOINT_JEPA_V1",
    ),
    400: (
        "FAIL_UPDATE_400_ACTION_QUERY_SPATIAL_SUCCESSOR_JOINT_JEPA_V1_GATE_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_400_ACTION_QUERY_SPATIAL_SUCCESSOR_JOINT_JEPA_V1",
    ),
    1_000: (
        "FAIL_UPDATE_1000_ACTION_QUERY_SPATIAL_SUCCESSOR_JOINT_JEPA_V1_GATE_NO_RETRY",
        "PASS_ACTION_QUERY_SPATIAL_SUCCESSOR_JOINT_JEPA_V1_MECHANISM_ONLY",
    ),
}
CONTROL_UPDATE_ZERO_FAIL = GATE_CONTROLS[0][0]
CONTROL_UPDATE_100_FAIL = GATE_CONTROLS[100][0]
CONTROL_UPDATE_400_FAIL = GATE_CONTROLS[400][0]
CONTROL_UPDATE_1000_FAIL = GATE_CONTROLS[1_000][0]
CONTROL_PASS = GATE_CONTROLS[1_000][1]
FAILURE_CONTROLS = tuple(value[0] for value in GATE_CONTROLS.values())

WORK_CONTRACT = {
    "per_update": {
        "presentations": 16,
        "microbatches": 4,
        "combined_objectives": 4,
        "combined_backwards": 4,
        "all_nine_action_predictor_calls": 4,
        "candidate_row_successors": 144,
        "online_encoder_lift_calls": 8,
        "semantic_head_calls": 8,
        "target_encoder_lift_calls": 8,
        "online_optimizer_updates": 1,
        "predictor_optimizer_updates": 1,
        "ema_updates": 1,
    },
    "complete": {
        "updates": 1_000, "presentations": 16_000,
        "microbatches": 4_000, "combined_objectives": 4_000,
        "combined_backwards": 4_000, "scalar_component_evaluations": 16_000,
        "candidate_row_successors": 144_000,
        "update_one_route_probe_calls": 8,
        "perception_only_updates": 0, "predictor_only_updates": 0,
        "separately_trained_predictor_updates": 0,
    },
}
WARNING_CONTRACT = {
    "all_runtime_warnings_recorded": True,
    "warning_never_changes_a_gate_or_checkpoint_qualification": True,
    "nonfinite_warning_field_is_terminal": True,
    "post_return_warning_receipt_must_be_complete": True,
}
JOINT_U1_ROUTE_PREFLIGHT_REQUIREMENTS = {
    "device": "cpu_only_synthetic",
    "four_combined_backwards_each_divided_by_four": True,
    "semantic_route_autograd_grad_calls": 4,
    "combined_dynamics_route_autograd_grad_calls": 4,
    "shared_encoder_and_lift_semantic_gradient_finite_nonzero": True,
    "shared_encoder_and_lift_dynamics_gradient_finite_nonzero": True,
    "every_predictor_component_gradient_finite_nonzero": True,
    "target_gradients_absent": True,
    "target_optimizer_membership_absent": True,
    "finite_gradient_ratio_is_informational_not_an_abort": True,
    "presentations_optimizer_steps_and_ema_steps": 0,
}


def _metric(metrics: Mapping[str, Any], name: str, *aliases: str) -> Any:
    for container in (
        metrics,
        metrics.get("integrity", {}),
        metrics.get("joint_accounting", {}),
        metrics.get("work_accounting", {}),
        metrics.get("first_step_integrity", {}) or {},
    ):
        if isinstance(container, Mapping):
            for key in (name, *aliases):
                if key in container:
                    return container[key]
    raise ValueError(f"missing metric: {name}")


def _finite(metrics: Mapping[str, Any], name: str, *aliases: str) -> float:
    value = _metric(metrics, name, *aliases)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _integer(metrics: Mapping[str, Any], name: str, *aliases: str) -> int:
    value = _metric(metrics, name, *aliases)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _prior(
    prior_metrics: Mapping[int | str, Mapping[str, Any]] | None, update: int
) -> Mapping[str, Any]:
    if not isinstance(prior_metrics, Mapping):
        raise ValueError(f"update {update} prior metrics are required")
    value = prior_metrics.get(update, prior_metrics.get(str(update)))
    if not isinstance(value, Mapping):
        raise ValueError(f"update {update} prior metrics are required")
    return value


def _common_gate_conjuncts(metrics: Mapping[str, Any]) -> dict[str, bool]:
    fields = (
        "source_authority_exact", "runtime_input_bindings_exact",
        "schedule_prefix_exact", "role_and_mapping_bindings_exact",
        "model_parameter_inventory_exact", "optimizer_inventory_exact",
        "rgb_only_causal_call_graph_exact", "forbidden_input_and_bypass_counts_zero",
        "target_requires_grad_false", "all_forbidden_access_counts_zero",
        "all_registered_values_finite", "state_nonconstant",
        "paired_rgb_latents_nonidentical", "out_of_frustum_semantic_unknown_exact",
        "work_accounting_exact",
    )
    return {name: _metric(metrics, name) is True for name in fields}


def _work_conjuncts(update: int, metrics: Mapping[str, Any]) -> dict[str, bool]:
    microbatches = update * 4
    expected = {
        "training_microbatch_count": microbatches,
        "scheduled_pair_presentations_loaded": update * 16,
        "joint_combined_objective_evaluation_count": microbatches,
        "combined_backward_call_count": microbatches,
        "effective_batch_divided_backward_count": microbatches,
        "registered_scalar_term_evaluation_count": microbatches * 4,
        "all_action_predictor_training_forward_count": microbatches,
        "candidate_row_successor_count": microbatches * 36,
        "online_optimizer_update_count": update,
        "target_ema_update_count": update,
        "predictor_optimizer_update_count": update,
        "joint_optimizer_update_count": update,
        "perception_only_update_count": 0,
        "predictor_only_update_count": 0,
        "separately_trained_predictor_update_count": 0,
        "route_probe_call_count": 0 if update == 0 else 8,
    }
    return {
        f"{name}_equals_{value}": _integer(metrics, name) == value
        for name, value in expected.items()
    }


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    prior_metrics: Mapping[int | str, Mapping[str, Any]] | None = None,
    *,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate the four exact conjunctive scientific observations."""

    if update not in GATE_CONTROLS:
        raise ValueError("update must be one of 0, 100, 400, or 1000")
    conjuncts = {
        "prior_gates_passed": prior_gates_passed is True,
        **_common_gate_conjuncts(metrics),
        **_work_conjuncts(update, metrics),
        f"presentations_equals_{update * 16}": (
            _integer(metrics, "presentations") == update * 16
        ),
        "target_gradient_tensor_count_equals_0": (
            _integer(metrics, "target_gradient_tensor_count") == 0
        ),
        "target_optimizer_membership_count_equals_0": (
            _integer(metrics, "target_optimizer_membership_count") == 0
        ),
    }
    if update == 0:
        for field in (
            "online_target_representation_bitwise_equal",
            "predictor_parameter_group_present",
            "semantic_objective_formula_exact",
            "action_query_objective_formula_exact",
            "reviewed_model_source_synthetic_witness_sha256_non_null",
        ):
            conjuncts[field] = metrics.get(field) is True
        conjuncts["initial_target_hard_sync_count_equals_1"] = (
            _integer(metrics, "initial_target_hard_sync_count") == 1
        )
    elif update == 100:
        zero = _prior(prior_metrics, 0)
        conjuncts.update({
            "A_strictly_lower_than_update_0": _finite(metrics, "A") < _finite(zero, "A"),
            "raster_nll_strictly_lower_than_update_0": _finite(
                metrics, "aggregate_raster_nll"
            ) < _finite(zero, "aggregate_raster_nll"),
            "semantic_balanced_accuracy_at_least_point60": _finite(
                metrics, "aggregate_raster_balanced_accuracy"
            ) >= 0.60,
            "free_recall_at_least_point55": _finite(metrics, "aggregate_free_recall") >= 0.55,
            "occupied_recall_at_least_point30": _finite(
                metrics, "aggregate_occupied_recall"
            ) >= 0.30,
            "free_occupied_gap_at_most_point50": _finite(
                metrics, "free_occupied_recall_gap"
            ) <= 0.50,
            "rough_balanced_accuracy_strictly_improves": _finite(
                metrics, "rough_raster_balanced_accuracy"
            ) > _finite(zero, "rough_raster_balanced_accuracy"),
            "rough_occupied_recall_strictly_improves": _finite(
                metrics, "rough_raster_occupied_recall"
            ) > _finite(zero, "rough_raster_occupied_recall"),
            "paired_rgb_margin_strictly_improves": _finite(
                metrics, "paired_rgb_margin"
            ) > _finite(zero, "paired_rgb_margin"),
            "paired_rgb_scene_wins_at_least_6": _integer(
                metrics, "paired_rgb_scene_wins"
            ) >= 6,
            "action_nll_strictly_below_log9": _finite(
                metrics, "action_raw_nll"
            ) < math.log(9.0),
            "action_macro_balanced_accuracy_strictly_above_one_ninth": _finite(
                metrics, "action_macro_balanced_accuracy"
            ) > 1.0 / 9.0,
            "hardest_wrong_positive_in_at_least_1_family": _integer(
                metrics, "hardest_wrong_positive_margin_family_count"
            ) >= 1,
            "correct_deranged_nll_strictly_below_log2": _finite(
                metrics, "correct_next_deranged_raw_nll"
            ) < math.log(2.0),
            "correct_deranged_win_rate_strictly_above_point50": _finite(
                metrics, "correct_next_deranged_strict_win_rate"
            ) > 0.50,
            "encoder_displaced": _metric(metrics, "encoder_parameter_displaced") is True,
            "all_predictor_components_displaced": (
                _metric(metrics, "all_predictor_components_displaced") is True
            ),
        })
    elif update == 400:
        hundred = _prior(prior_metrics, 100)
        conjuncts.update({
            "semantic_balanced_accuracy_at_least_point80": _finite(
                metrics, "aggregate_raster_balanced_accuracy"
            ) >= 0.80,
            "occupied_recall_at_least_point60": _finite(
                metrics, "aggregate_occupied_recall"
            ) >= 0.60,
            "rough_occupied_recall_at_least_point55": _finite(
                metrics, "rough_raster_occupied_recall"
            ) >= 0.55,
            "paired_rgb_margin_strictly_positive": _finite(
                metrics, "paired_rgb_margin"
            ) > 0.0,
            "paired_rgb_scene_wins_at_least_6": _integer(
                metrics, "paired_rgb_scene_wins"
            ) >= 6,
            "action_nll_strictly_below_point98_log9": _finite(
                metrics, "action_raw_nll"
            ) < 0.98 * math.log(9.0),
            "action_macro_balanced_accuracy_at_least_point18": _finite(
                metrics, "action_macro_balanced_accuracy"
            ) >= 0.18,
            "hardest_wrong_positive_in_at_least_3_families": _integer(
                metrics, "hardest_wrong_positive_margin_family_count"
            ) >= 3,
            "correct_deranged_win_rate_at_least_point70": _finite(
                metrics, "correct_next_deranged_strict_win_rate"
            ) >= 0.70,
            "target_effective_rank_strictly_positive": _finite(
                metrics, "target_effective_rank"
            ) > 0.0,
            "target_channel_variance_strictly_positive": _finite(
                metrics, "target_channel_variance"
            ) > 0.0,
            "target_spatial_diversity_strictly_positive": _finite(
                metrics, "target_spatial_diversity"
            ) > 0.0,
            "target_effective_rank_retains_point75_update_100": _finite(
                metrics, "target_effective_rank"
            ) >= 0.75 * _finite(hundred, "target_effective_rank"),
            "target_channel_variance_retains_point75_update_100": _finite(
                metrics, "target_channel_variance"
            ) >= 0.75 * _finite(hundred, "target_channel_variance"),
            "target_spatial_diversity_retains_point75_update_100": _finite(
                metrics, "target_spatial_diversity"
            ) >= 0.75 * _finite(hundred, "target_spatial_diversity"),
        })
    else:
        four_hundred = _prior(prior_metrics, 400)
        conjuncts.update({
            "A_at_most_update_400": _finite(metrics, "A") <= _finite(four_hundred, "A"),
            "raster_nll_at_most_min_point38_or_update_400_plus_point01": _finite(
                metrics, "aggregate_raster_nll"
            ) <= min(0.38, _finite(four_hundred, "aggregate_raster_nll") + 0.01),
            "semantic_balanced_accuracy_at_least_relative_floor": _finite(
                metrics, "aggregate_raster_balanced_accuracy"
            ) >= max(0.80, _finite(four_hundred, "aggregate_raster_balanced_accuracy") - 0.01),
            "unknown_recall_at_least_point80": _finite(
                metrics, "aggregate_unknown_recall"
            ) >= 0.80,
            "free_recall_at_least_point75": _finite(metrics, "aggregate_free_recall") >= 0.75,
            "occupied_recall_at_least_relative_floor": _finite(
                metrics, "aggregate_occupied_recall"
            ) >= max(0.70, _finite(four_hundred, "aggregate_occupied_recall") - 0.03),
            "free_occupied_gap_at_most_point25": _finite(
                metrics, "free_occupied_recall_gap"
            ) <= 0.25,
            "rough_balanced_accuracy_at_least_relative_floor": _finite(
                metrics, "rough_raster_balanced_accuracy"
            ) >= max(0.772, _finite(four_hundred, "rough_raster_balanced_accuracy") - 0.01),
            "rough_occupied_recall_at_least_relative_floor": _finite(
                metrics, "rough_raster_occupied_recall"
            ) >= max(0.65, _finite(four_hundred, "rough_raster_occupied_recall") - 0.03),
            "paired_rgb_margin_strictly_positive": _finite(metrics, "paired_rgb_margin") > 0.0,
            "paired_rgb_scene_wins_equal_8": _integer(metrics, "paired_rgb_scene_wins") == 8,
            "target_effective_rank_strictly_positive": _finite(
                metrics, "target_effective_rank"
            ) > 0.0,
            "target_channel_variance_strictly_positive": _finite(
                metrics, "target_channel_variance"
            ) > 0.0,
            "target_spatial_diversity_strictly_positive": _finite(
                metrics, "target_spatial_diversity"
            ) > 0.0,
            "anti_collapse_effective_rank_retains_point75": _finite(
                metrics, "target_effective_rank"
            ) >= 0.75 * _finite(four_hundred, "target_effective_rank"),
            "anti_collapse_channel_variance_retains_point75": _finite(
                metrics, "target_channel_variance"
            ) >= 0.75 * _finite(four_hundred, "target_channel_variance"),
            "anti_collapse_spatial_diversity_retains_point75": _finite(
                metrics, "target_spatial_diversity"
            ) >= 0.75 * _finite(four_hundred, "target_spatial_diversity"),
            "action_nll_strictly_below_point95_log9": _finite(
                metrics, "action_raw_nll"
            ) < 0.95 * math.log(9.0),
            "action_macro_balanced_accuracy_strictly_above_two_ninths": _finite(
                metrics, "action_macro_balanced_accuracy"
            ) > 2.0 / 9.0,
            "hardest_wrong_positive_in_at_least_6_families": _integer(
                metrics, "hardest_wrong_positive_margin_family_count"
            ) >= 6,
            "mean_wrong_energy_strictly_above_executed": _finite(
                metrics, "mean_wrong_action_energy"
            ) > _finite(metrics, "mean_executed_action_energy"),
            "non_hold_hold_energy_strictly_above_executed": _finite(
                metrics, "mean_non_hold_hold_action_energy"
            ) > _finite(metrics, "mean_non_hold_executed_action_energy"),
            "correct_deranged_nll_strictly_below_point95_log2": _finite(
                metrics, "correct_next_deranged_raw_nll"
            ) < 0.95 * math.log(2.0),
            "correct_deranged_win_rate_at_least_point70": _finite(
                metrics, "correct_next_deranged_strict_win_rate"
            ) >= 0.70,
            "correct_deranged_positive_in_at_least_6_families": _integer(
                metrics, "correct_next_positive_margin_family_count"
            ) >= 6,
            "successor_energy_at_most_point90_persistence": _finite(
                metrics, "mean_successor_unscaled_local_energy"
            ) <= 0.90 * _finite(metrics, "mean_persistence_unscaled_local_energy"),
            "successor_over_persistence_in_at_least_6_families": _integer(
                metrics, "successor_over_persistence_strict_win_family_count"
            ) >= 6,
            "autoregressive_rollout_exact": all((
                _integer(metrics, "autoregressive_rollout_step_count") == 8,
                _integer(metrics, "autoregressive_rollout_action_count") == 9,
                _metric(metrics, "autoregressive_rollout_all_intermediate_and_final_finite") is True,
                _integer(metrics, "autoregressive_rollout_future_rgb_input_count") == 0,
                _integer(metrics, "autoregressive_rollout_objective_backward_step_ema_count") == 0,
                _integer(metrics, "autoregressive_rollout_renormalization_count") == 0,
            )),
            "encoder_still_displaced": _metric(metrics, "encoder_parameter_displaced") is True,
            "lift_still_displaced": _metric(metrics, "lift_parameter_displaced") is True,
            "all_predictor_components_still_displaced": (
                _metric(metrics, "all_predictor_components_displaced") is True
            ),
        })

    passed = all(conjuncts.values())
    fail_control, pass_control = GATE_CONTROLS[update]
    return {
        "update": update,
        "kind": {
            0: "structural_and_source_witness",
            100: "early_joint_learning_falsification",
            400: "intermediate_joint_learning_falsification",
            1_000: "complete_action_query_mechanism_qualification",
        }[update],
        "passed": passed,
        "control": pass_control if passed else fail_control,
        "conjuncts": conjuncts,
        "thresholds": {} if update == 0 else copy.deepcopy(GATE_THRESHOLDS[update]),
        "all_conjunctive": True,
        "scientific_gate_evidence": update != 0,
    }


def science_contract() -> dict[str, Any]:
    inherited = copy.deepcopy(_v3.science_contract())
    return {
        "schema": f"{SCHEMA_PREFIX}_science_contract_v1",
        "repository_goal": inherited["repository_goal"],
        "scientific_question": (
            "can_local_all_action_query_contrast_make_geometry_grounded_RGB_"
            "successor_latents_action_identifiable_without_a_privileged_bypass"
        ),
        "governing_documents": {
            "preregistration": preregistration_binding(),
            "standing_scope_authority": copy.deepcopy(STANDING_SCOPE_AUTHORITY),
            "frozen_v3_source_manifest": frozen_v3_source_manifest_binding(),
            "prior_public_terminal_audits": prior_public_terminal_audit_bindings(),
        },
        "runtime_inputs": copy.deepcopy(inherited["runtime_inputs"]),
        "data": copy.deepcopy(inherited["data"]),
        "model": model_config(),
        "call_graph": {
            "online_current_RGB": "trainable_representation_to_semantics_and_action_query_predictor",
            "online_next_RGB": "same_trainable_representation_to_semantic_term_only",
            "target_next_RGB": "stop_gradient_EMA_successor_target_only",
            "target_deranged_next_RGB": "stop_gradient_EMA_contrast_target_only",
            "predictor_input": "normalized_online_current_BEV_plus_action_index_only",
            "all_nine_actions_vectorized": True,
            "training_all_nine_actions_unchanged": True,
            "rollout_selected_action_path_uses_identical_learned_modules": True,
            "future_RGB_pose_odometry_map_goal_or_label_predictor_input": False,
        },
        "objective": objective_contract(),
        "optimizer": optimizer_contract(),
        "schedule": build_schedule_identity(),
        "gates": {
            "observations": [0, 100, 400, 1_000],
            "thresholds": {str(key): value for key, value in GATE_THRESHOLDS.items()},
            "all_conditions_conjunctive": True,
            "stop_at_first_applicable_failure": True,
        },
        "work": copy.deepcopy(WORK_CONTRACT),
        "warning_contract": copy.deepcopy(WARNING_CONTRACT),
        "joint_u1_route_preflight": copy.deepcopy(JOINT_U1_ROUTE_PREFLIGHT_REQUIREMENTS),
        "lifecycle": {
            "attempt_index": 1,
            "maximum_attempts": 1,
            "maximum_updates": MAXIMUM_UPDATES,
            "maximum_presentations": MAXIMUM_PRESENTATIONS,
            "gpu_active_minutes_maximum": GPU_ACTIVE_TIME_CAP_MINUTES,
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "output_root_must_be_absent_before_mode_0700_reservation": True,
            "checkpoint_and_training_trace_write_only": True,
            "retry_resume_repair_recovery_extension_or_second_attempt": False,
        },
        "authority": {
            "checkpoint_qualified": False,
            "checkpoint_read_authorized": False,
            "g2_navigation_heldout_sealed_production_or_deployment_authorized": False,
            "predecessor_runtime_state_reuse_authorized": False,
        },
    }


SCIENCE_COMPONENT_SHA256 = {
    "model": canonical_json_sha256(model_config()),
    "objective": canonical_json_sha256(objective_contract()),
    "optimizer": canonical_json_sha256(optimizer_contract()),
    "schedule": canonical_json_sha256(build_schedule_identity()),
    "gates": canonical_json_sha256(GATE_THRESHOLDS),
    "work": canonical_json_sha256(WORK_CONTRACT),
    "warning": canonical_json_sha256(WARNING_CONTRACT),
    "authority": canonical_json_sha256(EXECUTION_AUTHORITY),
}
SCIENCE_CONTRACT_SHA256 = canonical_json_sha256(science_contract())


def validate_frozen_v3_source_closure(root: Path = ROOT) -> dict[str, str]:
    raw = _read_regular_source(root / FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH)
    if (
        len(raw) != FROZEN_V3_SOURCE_MANIFEST_BYTE_COUNT
        or hashlib.sha256(raw).hexdigest() != FROZEN_V3_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V3 source manifest identity changed")
    value = _v3.validate_source_manifest(raw, root)
    if (
        value.get("content_sha256") != FROZEN_V3_SOURCE_MANIFEST_CONTENT_SHA256
        or value.get("source_bindings_sha256") != FROZEN_V3_SOURCE_BINDINGS_SHA256
        or value.get("source_count") != FROZEN_V3_SOURCE_COUNT
    ):
        raise PermissionError("frozen V3 source closure conclusion changed")
    return {
        binding["path"]: binding["file_sha256"]
        for binding in value["source_bindings"]
    }


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    result = validate_frozen_v3_source_closure(root)
    prereg = _read_regular_source(root / PREREGISTRATION_RELATIVE_PATH)
    if (
        len(prereg) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(prereg).hexdigest() != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("Action-Query preregistration changed")
    result[PREREGISTRATION_RELATIVE_PATH] = PREREGISTRATION_FILE_SHA256
    authority_raw = _read_regular_source(root / STANDING_SCOPE_AUTHORITY["path"])
    if (
        len(authority_raw) != STANDING_SCOPE_AUTHORITY["byte_count"]
        or hashlib.sha256(authority_raw).hexdigest()
        != STANDING_SCOPE_AUTHORITY["file_sha256"]
    ):
        raise PermissionError("standing scientific-scope authority changed")
    authority_value = json.loads(authority_raw)
    if (
        type(authority_value) is not dict
        or authority_value.get("content_sha256")
        != STANDING_SCOPE_AUTHORITY["content_sha256"]
    ):
        raise PermissionError("standing scientific-scope authority conclusion changed")
    result[STANDING_SCOPE_AUTHORITY["path"]] = STANDING_SCOPE_AUTHORITY[
        "file_sha256"
    ]
    for binding in PRIOR_PUBLIC_TERMINAL_AUDITS:
        raw = _read_regular_source(root / binding["path"])
        if (
            len(raw) != binding["byte_count"]
            or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
        ):
            raise PermissionError(f"prior public audit changed: {binding['name']}")
        value = json.loads(raw)
        if (
            type(value) is not dict
            or value.get("content_sha256") != binding["content_sha256"]
        ):
            raise PermissionError(f"prior public audit conclusion changed: {binding['name']}")
        result[binding["path"]] = binding["file_sha256"]
    result[FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH] = (
        FROZEN_V3_SOURCE_MANIFEST_FILE_SHA256
    )
    return result


def validate_source_manifest(raw: bytes, root: Path = ROOT) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="Action-Query V1 source manifest")
    fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
    }
    bindings = value.get("source_bindings")
    if (
        set(value) != fields
        or value.get("schema") != SOURCE_MANIFEST_SCHEMA
        or value.get("status") != "PASS_SOURCE_CLOSURE"
        or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value.get("forced_dynamic_sources") != list(SOURCE_PATHS)
        or value.get("excluded_runtime_categories") != list(PROHIBITED_RUNTIME_CATEGORIES)
        or value.get("source_paths") != list(SOURCE_PATHS)
        or type(bindings) is not list
        or len(bindings) != 91
        or value.get("source_count") != 91
        or value.get("source_bindings_sha256") != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
    ):
        raise PermissionError("Action-Query source manifest contract changed")
    for relative, binding in zip(SOURCE_PATHS, bindings, strict=True):
        if (
            type(binding) is not dict
            or set(binding) != {"path", "file_sha256", "byte_count"}
            or binding.get("path") != relative
            or safe_relative_source_path(relative) != relative
            or not is_sha256(binding.get("file_sha256"))
            or type(binding.get("byte_count")) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("Action-Query source binding changed")
        payload = _read_regular_source(root / relative)
        if (
            len(payload) != binding["byte_count"]
            or hashlib.sha256(payload).hexdigest() != binding["file_sha256"]
        ):
            raise PermissionError(f"manifest-bound source changed: {relative}")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    value = validate_source_manifest(raw, root)
    result = {
        binding["path"]: binding["file_sha256"]
        for binding in value["source_bindings"]
    }
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(raw).hexdigest()
    result.update(validate_governing_documents(root))
    return result


SCIENTIFIC_REVIEW_CHECKS = {
    "explicit_user_scope_reopening_acknowledged": True,
    "one_failure_closes_reopened_hypothesis_without_another_current_action_variant": True,
    "source_freeze_commit_contains_exact_manifest_and_91_reviewed_sources": True,
    "frozen_v3_data_schedule_runtime_and_custody_preserved": True,
    "fresh_model_and_no_predecessor_runtime_state_reuse": True,
    "predictor_inventory_504384_parameters_34_tensors_exact": True,
    "all_nine_action_successors_vectorized_and_locally_contrasted": True,
    "observation_only_selected_action_rollout_matches_vectorized_slice": True,
    "online_representation_and_predictor_joint_from_update_one": True,
    "no_perception_only_or_separately_trained_predictor_phase": True,
    "joint_u1_semantic_and_dynamics_routes_finite_nonzero": True,
    "finite_route_ratio_is_informational_only": True,
    "one_attempt_1000_update_16000_presentation_cap": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}
SOURCE_ONLY_CHECKS = {
    "generated_inputs_opened": [],
    "checkpoints_tensors_or_predecessor_outputs_opened": [],
    "runtime_outputs_or_traces_opened": [],
    "accelerators_queried_or_used": [],
    "navigation_g2_heldout_sealed_or_rejected_material_opened": [],
}


def validate_review(
    raw: bytes,
    manifest_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="Action-Query V1 source review")
    expected_manifest = _validate_artifact_binding(
        dict(manifest_binding), path=SOURCE_MANIFEST_RELATIVE_PATH
    )
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    if expected_manifest != artifact_binding(
        SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    ):
        raise PermissionError("review source-manifest binding changed")
    fields = {
        "schema", "status", "implementation_authors", "reviewer",
        "source_freeze_commit", "reviewed_sources", "source_manifest",
        "preregistration", "frozen_v3_source_manifest",
        "prior_public_terminal_audits", "science_contract",
        "joint_u1_route_preflight", "source_only_checks", "scientific_checks",
        "findings", "authority", "content_sha256",
    }
    reviewer = value.get("reviewer")
    _source_freeze_commit(
        value.get("source_freeze_commit"), name="review.source_freeze_commit"
    )
    if (
        set(value) != fields
        or value.get("schema") != REVIEW_SCHEMA
        or value.get("status") != "PASS_SOURCE_SCIENCE_AND_CUSTODY_REVIEW"
        or value.get("implementation_authors") != list(IMPLEMENTATION_AUTHORS)
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer in IMPLEMENTATION_AUTHORS
        or value.get("reviewed_sources") != current_source_bindings(root)
        or value.get("source_manifest") != expected_manifest
        or value.get("preregistration") != preregistration_binding()
        or value.get("frozen_v3_source_manifest") != frozen_v3_source_manifest_binding()
        or value.get("prior_public_terminal_audits") != prior_public_terminal_audit_bindings()
        or value.get("science_contract") != science_contract()
        or value.get("joint_u1_route_preflight") != JOINT_U1_ROUTE_PREFLIGHT_REQUIREMENTS
        or value.get("source_only_checks") != SOURCE_ONLY_CHECKS
        or value.get("scientific_checks") != SCIENTIFIC_REVIEW_CHECKS
        or value.get("findings") != []
        or value.get("authority") != REVIEW_AUTHORITY
    ):
        raise PermissionError("Action-Query independent source review changed")
    return dict(value)


def validate_authorization(
    raw: bytes,
    review_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="Action-Query V1 authorization")
    expected_review = _validate_artifact_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    review_raw = _read_regular_source(root / REVIEW_RELATIVE_PATH)
    review_content = parse_canonical_json(
        review_raw, name="Action-Query V1 source review"
    )["content_sha256"]
    if expected_review != artifact_binding(
        REVIEW_RELATIVE_PATH, review_raw, content_sha256=review_content
    ):
        raise PermissionError("authorization review binding changed")
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    review = validate_review(
        review_raw,
        artifact_binding(
            SOURCE_MANIFEST_RELATIVE_PATH,
            manifest_raw,
            content_sha256=manifest["content_sha256"],
        ),
        root=root,
    )
    fields = {
        "schema", "status", "authorizer", "source_freeze_commit",
        "independent_source_review", "preregistration",
        "prior_public_terminal_audits", "runtime_inputs", "experiment",
        "authority", "content_sha256",
    }
    authorizer = value.get("authorizer")
    source_commit = _source_freeze_commit(
        review.get("source_freeze_commit"), name="review.source_freeze_commit"
    )
    if (
        set(value) != fields
        or value.get("schema") != AUTHORIZATION_SCHEMA
        or value.get("status") != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {*IMPLEMENTATION_AUTHORS, review["reviewer"]}
        or value.get("source_freeze_commit") != source_commit
        or value.get("independent_source_review") != expected_review
        or value.get("preregistration") != preregistration_binding()
        or value.get("prior_public_terminal_audits") != prior_public_terminal_audit_bindings()
        or value.get("runtime_inputs") != runtime_authorization_template()
        or value.get("experiment") != science_contract()
        or value.get("authority") != EXECUTION_AUTHORITY
    ):
        raise PermissionError("Action-Query one-shot execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_v3.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity", "current_source_bindings", "evaluate_gate",
    "frozen_v3_source_manifest_binding", "model_config", "objective_contract",
    "optimizer_contract", "preregistration_binding",
    "prior_public_terminal_audit_bindings", "runtime_authorization_template",
    "science_contract", "validate_authorization",
    "validate_frozen_v3_source_closure", "validate_governing_documents",
    "validate_review", "validate_source_manifest",
})
