"""Source-only contract for the two-mode event-delta joint JEPA V1.

This module binds one fresh, capped, RGB-only experiment.  It reuses the
reviewed 91-source rigid attempt only as an operational/source-custody closure.
Scientifically it returns to the frozen V3 geometry encoder, deformable lift,
semantic head, EMA target, data, schedule, optimizer, and perception gates.
The sole new mechanism is the preregistered fixed-scale ZERO_EVENT versus
LEARNED_EVENT latent-delta predictor.  Importing this module opens no runtime
input, generated artifact, checkpoint, tensor, accelerator, held-out, sealed,
navigation, or rejected material.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_RIGID_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1.py"
)


def _source_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_rigid = _source_module(
    "_lewm_two_mode_event_delta_frozen_rigid_operational_contract",
    FROZEN_RIGID_CONTRACT_RELATIVE_PATH,
)
_v3 = _rigid._v3
_base = _v3._v2._v1

# Reuse non-scientific constants and helpers without exporting stale active
# transport identities, paths, predictor definitions, or gate controls.
_INHERITED_EXCLUSIONS = {
    "ROOT",
    "IMPLEMENTATION_AUTHORS",
    "SCHEMA_PREFIX",
    "EXPERIMENT_ID",
    "FROZEN_V3_SOURCE_COUNT",
    "FROZEN_V3_SOURCE_COMMIT",
    "FROZEN_V3_EXECUTION_AUTHORIZATION_COMMIT",
    "PREREGISTRATION_RELATIVE_PATH",
    "PREREGISTRATION_COMMIT",
    "PREREGISTRATION_FILE_SHA256",
    "PREREGISTRATION_CONTENT_SHA256",
    "PREREGISTRATION_BYTE_COUNT",
    "CONTRACT_RELATIVE_PATH",
    "MODEL_RELATIVE_PATH",
    "MODEL_CLASS_NAME",
    "RUNNER_RELATIVE_PATH",
    "LAUNCHER_RELATIVE_PATH",
    "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH",
    "MODEL_TEST_RELATIVE_PATH",
    "CONTRACT_RUNNER_TEST_RELATIVE_PATH",
    "CONTRACT_TEST_RELATIVE_PATH",
    "RUNNER_TEST_RELATIVE_PATH",
    "LAUNCHER_TEST_RELATIVE_PATH",
    "SOURCE_CLOSURE_TEST_RELATIVE_PATH",
    "TEST_RELATIVE_PATH",
    "SOURCE_MANIFEST_RELATIVE_PATH",
    "REVIEW_RELATIVE_PATH",
    "AUTHORIZATION_RELATIVE_PATH",
    "ADDITIVE_SOURCE_PATHS",
    "REUSED_SOURCE_PATHS",
    "SOURCE_PATHS",
    "SOURCE_MANIFEST_ENTRYPOINTS",
    "SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES",
    "SOURCE_MANIFEST_SCHEMA",
    "REVIEW_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "RESERVATION_SCHEMA",
    "METRICS_SCHEMA",
    "ARTIFACT_SCHEMA",
    "ACCESS_SCHEMA",
    "RESULT_SCHEMA",
    "COMPLETION_SCHEMA",
    "FAILURE_SCHEMA",
    "AUTHORIZATION_STATUS",
    "OPERATIONAL_FAILURE_STATUS",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "EXECUTION_AUTHORITY",
    "SOURCE_ONLY_AUTHORITY",
    "REVIEW_AUTHORITY",
    "GATE_CONTROLS",
    "CONTROL_UPDATE_ZERO_FAIL",
    "CONTROL_UPDATE_100_FAIL",
    "CONTROL_UPDATE_400_FAIL",
    "CONTROL_UPDATE_1000_FAIL",
    "CONTROL_PASS",
    "FAILURE_CONTROLS",
    "CONTROL_FAIL_JOINT_GRADIENT",
    "CONTROL_FAIL_OPERATIONAL",
    "PHASE_SWITCH_CONTROLS",
    "_GATE_CONTROL_REBIND",
    "_PHASE_CONTROL_REBIND",
    "PREDICTOR_PARAMETER_COUNT",
    "PREDICTOR_PARAMETER_TENSOR_COUNT",
    "PREDICTOR_ORDERED_PARAMETER_NAMES",
    "MODEL_CONFIG",
    "OBJECTIVE_CONTRACT",
    "OPTIMIZER_CONTRACT",
    "CALL_GRAPH_CONTRACT",
    "GATE_THRESHOLDS",
    "INTEGRITY_DELTA",
    "RIGID_TRANSPORT_PREFLIGHT_REQUIREMENTS",
    "SCIENTIFIC_REVIEW_CHECKS",
    "WARNING_POLICY",
    "model_config",
    "objective_contract",
    "optimizer_contract",
    "build_schedule_identity",
    "runtime_authorization_template",
    "evaluate_gate",
    "evaluate_update_401_phase_switch",
    "science_contract",
    "preregistration_binding",
    "current_source_bindings",
    "validate_governing_documents",
    "validate_source_manifest",
    "validate_review",
    "validate_authorization",
}
for _name in _rigid.__all__:
    if _name not in _INHERITED_EXCLUSIONS:
        globals()[_name] = getattr(_rigid, _name)

_read_regular_source = _rigid._read_regular_source
_source_freeze_commit = _rigid._source_freeze_commit
_validate_artifact_binding = _rigid._validate_artifact_binding
_finite = _base._finite
_integer = _base._integer
_boolean = _base._boolean
_metric = _base._metric


IMPLEMENTATION_AUTHORS = (
    "/root",
    "/root/v3_successor_assessment",
    "/root/event_delta_impl_assessment",
    "/root/event_delta_runner_impl",
)
SCHEMA_PREFIX = (
    "lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v1"
)
EXPERIMENT_ID = "geometry_anchored_two_mode_event_delta_joint_jepa_v1"

FROZEN_RIGID_SOURCE_COUNT = 91
FROZEN_RIGID_SOURCE_COMMIT = "9ee72d5b9fcd4c762c4538503bba38119db2ac9b"
FROZEN_RIGID_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _rigid.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)
FROZEN_V3_GEOMETRY_SOURCE_COUNT = 84
FROZEN_V3_GEOMETRY_SOURCE_COMMIT = "ebcde189628b1a7040ffaf95aafaf9fd8f404fc4"
FROZEN_V3_GEOMETRY_CONTRACT_RELATIVE_PATH = _rigid.FROZEN_V3_CONTRACT_RELATIVE_PATH

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v1_"
    "preregistration_2026-07-27.md"
)
PREREGISTRATION_COMMIT = "597f309382a56666a34c176cf92317cb11d3087a"
PREREGISTRATION_FILE_SHA256 = (
    "f67c6aaa9ae3de06f49d24d142b80322573452032179ce116a5177de8a2ad981"
)
PREREGISTRATION_BYTE_COUNT = 38_562

RIGID_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1_terminal_audit_2026-07-27.json"
)
RIGID_TERMINAL_AUDIT_COMMIT = "293a02e1cc99c3b5aac876efc68104093468506c"
RIGID_TERMINAL_AUDIT_FILE_SHA256 = (
    "38d65b46bd4ff83ab67924233badb96c37bd079f0b85b36c69512c905557f25e"
)
RIGID_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "5a87a441560c8ab397a0cdfab5ca08fc1c7ad7b8294d323194f712ad834821ca"
)
RIGID_TERMINAL_AUDIT_BYTE_COUNT = 19_570
RIGID_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_COMPLETE_BUDGET_SCIENTIFIC_FAILURE_RIGID_TRANSPORT_V1_"
    "CONSUMED_CLOSED_NO_RETRY_OR_TRANSPORT_VARIANT"
)
RIGID_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_COMPLETE_BUDGET_TERMINAL_SCIENTIFIC_QUALIFICATION_FAILURE_WITH_"
    "NO_OPERATIONAL_OR_INTEGRITY_DEFECT"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
MODEL_CLASS_NAME = "GeometryAnchoredTwoModeEventDeltaJointJepaV1"
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1_"
    "source_closure.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
CONTRACT_RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1_"
    "contract_runner.py"
)
CONTRACT_TEST_RELATIVE_PATH = CONTRACT_RUNNER_TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = CONTRACT_RUNNER_TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = CONTRACT_RUNNER_TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = CONTRACT_RUNNER_TEST_RELATIVE_PATH
TEST_RELATIVE_PATH = CONTRACT_RUNNER_TEST_RELATIVE_PATH

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v1_"
    "source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v1_"
    "source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v1_"
    "execution_authorization_2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    CONTRACT_RUNNER_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(set(_rigid.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if (
    len(REUSED_SOURCE_PATHS) != FROZEN_RIGID_SOURCE_COUNT
    or len(ADDITIVE_SOURCE_PATHS) != 7
    or len(SOURCE_PATHS) != 98
):
    raise PermissionError("event-delta closure must be rigid 91 plus seven files")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_metrics_v1"
ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_artifact_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_GEOMETRY_ANCHORED_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1"
)
OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_GEOMETRY_ANCHORED_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_"
    "OPERATIONAL_OR_INTEGRITY_FAILURE_NO_RETRY"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v1/"
    "attempt_v1"
)

# The successor keeps the proven canonical receipt machinery, while the
# preregistration fixes the exact terminal inventories below.
NORMAL_RECEIPT_PATHS = (
    "reservation.json", "metrics.json", "artifact.json", "access.json",
    "result.json", "completed.json",
)
OPERATIONAL_FAILURE_RECEIPT_PATHS = ("failure.json", "completed.json")
MANDATORY_OPERATIONAL_TERMINATORS = ("failure.json", "completed.json")

EXECUTION_AUTHORITY = {
    key: copy.deepcopy(value)
    for key, value in _rigid.EXECUTION_AUTHORITY.items()
    if key != "single_registered_rigid_transport_attempt_only"
}
EXECUTION_AUTHORITY.update({
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "scientific_successor_fresh_model_only": True,
    "single_registered_two_mode_event_delta_attempt_only": True,
    "v3_checkpoint_tensor_trace_optimizer_rng_or_runtime_state_reuse_authorized": False,
    "rigid_checkpoint_tensor_trace_optimizer_rng_receipt_or_runtime_state_reuse_authorized": False,
    "transport_flow_warp_inverse_or_scale_variant_authorized": False,
})
SOURCE_ONLY_AUTHORITY = {
    key: copy.deepcopy(value)
    for key, value in _rigid.SOURCE_ONLY_AUTHORITY.items()
    if key != "rigid_transport_cpu_synthetic_preflight_authorized"
}
SOURCE_ONLY_AUTHORITY.update({
    "two_mode_event_delta_cpu_synthetic_preflight_authorized": True,
    "runtime_generated_input_checkpoint_tensor_gpu_or_training_access_authorized": False,
})
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)

GATE_CONTROLS = {
    0: (
        "FAIL_UPDATE_ZERO_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_STRUCTURAL_GATE_"
        "TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_ZERO_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_"
        "STRUCTURAL_GATE",
    ),
    100: (
        "FAIL_UPDATE_100_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_PERCEPTION_GATE_"
        "TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_100_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_"
        "PERCEPTION_GATE",
    ),
    400: (
        "FAIL_UPDATE_400_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_CALIBRATION_GATE_"
        "TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_400_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_"
        "CALIBRATION_GATE",
    ),
    1_000: (
        "FAIL_UPDATE_1000_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_QUALIFICATION_"
        "GATE_TERMINAL_NO_RETRY",
        "PASS_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_MECHANISM_ONLY",
    ),
}
CONTROL_UPDATE_ZERO_FAIL = GATE_CONTROLS[0][0]
CONTROL_UPDATE_100_FAIL = GATE_CONTROLS[100][0]
CONTROL_UPDATE_400_FAIL = GATE_CONTROLS[400][0]
CONTROL_UPDATE_1000_FAIL = GATE_CONTROLS[1_000][0]
CONTROL_PASS = GATE_CONTROLS[1_000][1]
FAILURE_CONTROLS = tuple(row[0] for row in GATE_CONTROLS.values())
CONTROL_FAIL_JOINT_GRADIENT = (
    "FAIL_TWO_MODE_EVENT_DELTA_JOINT_SHARED_GRADIENT_CONTRIBUTION_GATE_"
    "TERMINAL_NO_RETRY"
)
CONTROL_FAIL_OPERATIONAL = OPERATIONAL_FAILURE_STATUS
PHASE_SWITCH_CONTROLS = (
    "FAIL_UPDATE_401_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_PHASE_SWITCH_"
    "TERMINAL_NO_RETRY",
    "CONTINUE_AFTER_UPDATE_401_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V1_PHASE_SWITCH",
)

PREDICTOR_PARAMETER_COUNT = 231_505
PREDICTOR_PARAMETER_TENSOR_COUNT = 15
PREDICTOR_ORDERED_PARAMETER_NAMES = (
    "predictor.action_embedding.weight",
    "predictor.input_projection.weight",
    "predictor.input_projection.bias",
    "predictor.residual_blocks.0.conv1.weight",
    "predictor.residual_blocks.0.conv1.bias",
    "predictor.residual_blocks.0.conv2.weight",
    "predictor.residual_blocks.0.conv2.bias",
    "predictor.residual_blocks.1.conv1.weight",
    "predictor.residual_blocks.1.conv1.bias",
    "predictor.residual_blocks.1.conv2.weight",
    "predictor.residual_blocks.1.conv2.bias",
    "predictor.event_mean_head.weight",
    "predictor.event_mean_head.bias",
    "predictor.event_logit_head.weight",
    "predictor.event_logit_head.bias",
)

MODEL_CONFIG = copy.deepcopy(_v3.model_config())
MODEL_CONFIG["predictor"] = {
    "action_count": 9,
    "action_vocabulary": list(ACTION_VOCABULARY),
    "input_current": "per_cell_channel_LayerNorm_online_current_BEV_(B,64,64,64)",
    "input_action": "exact_one_hot_(B,9)_through_zero_initialized_Embedding(9,16)",
    "trunk": (
        "concat_current64_broadcast_action16_then_Conv2d80to64_k3_GELU_"
        "then_two_shared_width64_local_residual_blocks"
    ),
    "event_mean_head": "Conv2d(64,64,k3,s1,p1,bias=True)",
    "event_logit_head": "Conv2d(64,1,k3,s1,p1,bias=True)",
    "one_action_outputs": {
        "mu_event": ["B", 64, 64, 64],
        "event_logit": ["B", 1, 64, 64],
    },
    "all_action_outputs": {
        "mu_event": ["B", 9, 64, 64, 64],
        "event_logit": ["B", 9, 1, 64, 64],
    },
    "fixed_modes": {
        "ZERO_EVENT": "exact_zeros_like_mu_event_no_trainable_mean",
        "LEARNED_EVENT": "mu_event",
    },
    "parameter_count": PREDICTOR_PARAMETER_COUNT,
    "parameter_tensor_count": PREDICTOR_PARAMETER_TENSOR_COUNT,
    "ordered_parameter_names": list(PREDICTOR_ORDERED_PARAMETER_NAMES),
    "predict_all_actions": True,
    "learned_scale_variance_temperature_codebook_pair_posterior_or_third_mode": False,
    "transport_warp_flow_correspondence_grid_sample_inverse_classifier_or_action_specific_convolution": False,
}
MODEL_CONFIG["event_delta_target"] = {
    "normalization": "per_cell_channel_LayerNorm_eps_1e-5_no_affine",
    "current": "stopgrad(N(EMA_RGB_current))",
    "next": "stopgrad(N(EMA_RGB_next))",
    "deranged": "stopgrad(N(EMA_RGB_fixed_negative))",
    "correct_delta": "next_minus_current_shape_(B,64,64,64)",
    "deranged_delta": "deranged_minus_current_shape_(B,64,64,64)",
    "predictor_future_or_target_input": False,
}

CALL_GRAPH_CONTRACT = {
    "online_current_RGB": "trainable_encoder_lift_to_semantics_and_normalized_predictor_current",
    "online_next_RGB": "same_trainable_encoder_lift_to_semantic_S_only",
    "online_fixed_negative_RGB": "no_grad_normalized_detached_context_then_autograd_enabled_predictor",
    "target_current_RGB": "stop_gradient_EMA_encoder_lift_to_delta_origin",
    "target_next_RGB": "stop_gradient_EMA_encoder_lift_to_correct_delta",
    "target_fixed_negative_RGB": "stop_gradient_EMA_encoder_lift_to_deranged_delta",
    "predictor_input": "normalized_online_current_BEV_plus_commanded_action_only",
    "action_logits": "mechanically_negative_all_action_forward_mixture_energies_only",
    "future_label_pose_depth_flow_odometry_map_goal_or_privileged_input": False,
    "inverse_classifier_pair_posterior_transport_warp_flow_or_correspondence_path": False,
}

OBJECTIVE_CONTRACT = {
    "semantic": copy.deepcopy(_v3.objective_contract()["semantic_A"]),
    "semantic_normalized": "S=A/log(3)",
    "warmup_updates_1_400": {
        "total": "L=S",
        "predictor_training_forward_objective_backward_optimizer_state_or_update": 0,
    },
    "target": "d=stopgrad(N(EMA_next)-N(EMA_current));d_neg=stopgrad(N(EMA_negative)-N(EMA_current))",
    "component_energy": "mean_channel_SmoothL1_beta1_to_zero_or_mu_event",
    "temperature": "T400=raw_selection_mean_cellwise_persistence_energy",
    "change_weight": "w=e_persist/(e_persist+T400)_detached_correct_target_only",
    "balanced_energy": "0.5*changed_weighted_mean+0.5*static_weighted_mean",
    "mixture": (
        "-T400*logaddexp(logsigmoid(-ell)-e0/T400,"
        "logsigmoid(ell)-e1/T400)"
    ),
    "B400": "selection_mean_balanced_persistence_energy_independent_of_predictor",
    "joint_updates_401_1000": {
        "P_event": "mean_executed_balanced_mixture_energy/B400",
        "R_action": "raw_nine_way_forward_energy_CE/log(9)",
        "C_target": "raw_correct_vs_deranged_forward_energy_CE/log(2)",
        "C_context": "raw_true_vs_same_command_swapped_context_forward_energy_CE/log(2)",
        "total": "L=S+P_event+R_action+C_target+C_context",
        "component_weights": {
            "S": 1.0,
            "P_event": 1.0,
            "R_action": 1.0,
            "C_target": 1.0,
            "C_context": 1.0,
        },
        "encoder_and_predictor_train_together": True,
        "inverse_or_separate_classifier_loss": False,
    },
}

OPTIMIZER_CONTRACT = copy.deepcopy(_v3.OPTIMIZER_CONTRACT)

OBSERVATION_ACCOUNTING_EXPECTATIONS = {
    0: {
        "observation_pair_population_count": 495,
        "observation_endpoint_population_count": 924,
        "observation_pair_pass_count": 1,
        "observation_endpoint_pass_count": 1,
        "observation_pair_microbatch_count": 124,
        "observation_endpoint_microbatch_count": 231,
        "observation_microbatch_count": 355,
        "observation_online_encoder_lift_forward_count": 603,
        "observation_target_encoder_lift_forward_count": 0,
        "observation_semantic_head_forward_count": 603,
        "observation_all_action_predictor_forward_count": 0,
        "observation_one_action_predictor_forward_count": 0,
        "observation_predictor_forward_count": 0,
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
        "runtime_update_zero_synthetic_accelerator_call_count": 0,
    },
    100: {
        "observation_pair_population_count": 495,
        "observation_endpoint_population_count": 924,
        "observation_pair_pass_count": 1,
        "observation_endpoint_pass_count": 1,
        "observation_pair_microbatch_count": 124,
        "observation_endpoint_microbatch_count": 231,
        "observation_microbatch_count": 355,
        "observation_online_encoder_lift_forward_count": 603,
        "observation_target_encoder_lift_forward_count": 0,
        "observation_semantic_head_forward_count": 603,
        "observation_all_action_predictor_forward_count": 0,
        "observation_one_action_predictor_forward_count": 0,
        "observation_predictor_forward_count": 0,
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
    },
    400: {
        "observation_pair_population_count": 495,
        "observation_endpoint_population_count": 924,
        "observation_pair_pass_count": 3,
        "observation_endpoint_pass_count": 2,
        "observation_pair_microbatch_count": 372,
        "observation_endpoint_microbatch_count": 462,
        "observation_microbatch_count": 834,
        "observation_online_encoder_lift_forward_count": 727,
        "observation_target_encoder_lift_forward_count": 727,
        "observation_semantic_head_forward_count": 603,
        "observation_all_action_predictor_forward_count": 124,
        "observation_one_action_predictor_forward_count": 0,
        "observation_predictor_forward_count": 124,
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
    },
    1_000: {
        "observation_pair_population_count": 495,
        "observation_endpoint_population_count": 924,
        "observation_pair_pass_count": 3,
        "observation_endpoint_pass_count": 2,
        "observation_pair_microbatch_count": 372,
        "observation_endpoint_microbatch_count": 462,
        "observation_microbatch_count": 834,
        "observation_online_encoder_lift_forward_count": 1_099,
        "observation_target_encoder_lift_forward_count": 975,
        "observation_semantic_head_forward_count": 603,
        "observation_all_action_predictor_forward_count": 248,
        "observation_one_action_predictor_forward_count": 372,
        "observation_predictor_forward_count": 620,
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
    },
}

WORK_ACCOUNTING_CONTRACT = {
    "online_optimizer_updates": 1_000,
    "target_EMA_updates": 1_000,
    "presentations": 16_000,
    "combined_objective_evaluations": 4_000,
    "combined_backward_calls": 4_000,
    "warmup_microbatch_objectives_and_backwards": 1_600,
    "joint_microbatch_objectives_and_backwards": 2_400,
    "semantic_term_evaluations": 4_000,
    "event_persistence_term_evaluations": 2_400,
    "action_term_evaluations": 2_400,
    "target_term_evaluations": 2_400,
    "context_term_evaluations": 2_400,
    "registered_scalar_term_evaluations": 13_600,
    "predictor_and_joint_optimizer_updates": 600,
    "all_action_predictor_training_forward_calls": 2_400,
    "context_swap_predictor_training_forward_calls": 2_400,
    "total_predictor_training_forward_calls": 4_800,
    "online_encoder_lift_training_forward_calls": 10_400,
    "semantic_head_training_forward_calls": 8_000,
    "EMA_target_encoder_lift_training_forward_calls": 7_200,
    "warmup_predictor_training_work": 0,
    "shared_gradient_gate_passes": 600,
    "gpu_active_minutes_maximum": 30,
    "observation_accounting_by_update": {
        str(update): copy.deepcopy(expected)
        for update, expected in OBSERVATION_ACCOUNTING_EXPECTATIONS.items()
    },
}

GATE_THRESHOLDS = copy.deepcopy(_v3.GATE_THRESHOLDS)
GATE_THRESHOLDS[1_000] = {
    **GATE_THRESHOLDS[1_000],
    "event_balanced_B400_factor_maximum": 0.90,
    "context_energy_ratio_maximum": 0.95,
    "context_nll_strict_maximum": 0.95 * LOG2,
    "context_win_rate_minimum": 0.65,
    "state_template_energy_ratio_maximum": 0.95,
    "state_only_energy_ratio_maximum": 0.95,
    "matched_single_energy_ratio_maximum": 0.98,
    "event_changed_to_zero_changed_factor_maximum": 0.90,
    "zero_static_to_event_static_factor_maximum": 0.95,
    "mu_event_changed_abs_strict_minimum": 1e-4,
    "prior_changed_minus_static_minimum": 0.05,
    "prior_mean_minimum": 0.10,
    "prior_mean_maximum": 0.90,
    "prior_spatial_population_variance_minimum": 1e-4,
    "prior_context_difference_minimum": 0.02,
    "posterior_changed_minus_static_minimum": 0.10,
    "posterior_mean_minimum": 0.10,
    "posterior_mean_maximum": 0.90,
    "per_mode_family_mean_responsibility_minimum": 0.05,
    "family_count_minimum": 6,
}

WARNING_POLICY = copy.deepcopy(_rigid.WARNING_POLICY)

EVENT_DELTA_PREFLIGHT_REQUIREMENTS = {
    "cpu_synthetic_torch_only": True,
    "predictor_parameter_count": PREDICTOR_PARAMETER_COUNT,
    "predictor_parameter_tensor_count": PREDICTOR_PARAMETER_TENSOR_COUNT,
    "ordered_parameter_names": list(PREDICTOR_ORDERED_PARAMETER_NAMES),
    "one_action_shapes": {
        "mu_event": ["B", 64, 64, 64],
        "event_logit": ["B", 1, 64, 64],
    },
    "all_action_shapes": {
        "mu_event": ["B", 9, 64, 64, 64],
        "event_logit": ["B", 9, 1, 64, 64],
    },
    "exact_zero_action_embedding_and_logit_head": True,
    "event_mean_head_normal_std": 1e-3,
    "update_zero_all_action_outputs_bitwise_equal": True,
    "fixed_zero_and_learned_event_mode_identities": True,
    "singleton_logit_squeeze_and_shape_failures_checked": True,
    "layer_norm_last_channel_only_eps": 1e-5,
    "smooth_l1_beta": 1.0,
    "stable_logaddexp_mixture_reference_checked": True,
    "zero_mean_persistence_identity_atol_rtol": [2e-6, 2e-6],
    "synthetic_positive_temperature_log9_and_one_ninth_checked": True,
    "T400_B400_change_weight_and_reductions_checked": True,
    "matched_single_context_template_and_state_only_ablations_checked": True,
    "all_action_permutation_equivariance_checked": True,
    "update_401_all_online_gradient_routes_checked": True,
    "detached_context_predictor_autograd_route_checked": True,
    "warmup_predictor_zero_training_activity_checked": True,
    "no_scale_inverse_pair_posterior_transport_warp_flow_or_future_bypass": True,
    "generated_inputs_or_runtime_rows_opened": [],
    "checkpoints_tensors_traces_or_predecessor_runtime_outputs_opened": [],
    "accelerators_queried_or_used": [],
    "navigation_g2_heldout_sealed_or_rejected_material_opened": [],
}

SCIENTIFIC_REVIEW_CHECKS = {
    "frozen_preregistration_V3_geometry_and_rigid_terminal_audit_exact": True,
    "all_91_rigid_operational_sources_and_seven_additive_sources_rehashed": True,
    "rigid_sources_reused_as_operational_closure_not_active_science": True,
    "sole_scientific_delta_is_fixed_scale_two_mode_event_delta_predictive_state": True,
    "encoder_lift_semantic_target_data_seed_schedule_and_initialization_preserved": True,
    "optimizer_clipping_EMA_caps_and_perception_thresholds_preserved": True,
    "all_20_event_conjuncts_and_raw_vs_normalized_NLL_metrics_exact": True,
    "exact_forward_objective_backward_and_gradient_work_accounting": True,
    "observation_populations_passes_microbatches_forwards_and_zero_training_counter_effect_exact": True,
    "warning_and_terminal_receipt_policy_exact": True,
    "fresh_absent_before_reservation_root_and_one_attempt_lifecycle_exact": True,
    "no_active_transport_scale_inverse_pair_posterior_or_separate_predictor_training": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def frozen_v3_geometry_binding() -> dict[str, Any]:
    """Bind the completed V3 geometry experiment, never its runtime state."""

    return {
        "source_commit": FROZEN_V3_GEOMETRY_SOURCE_COMMIT,
        "source_count": FROZEN_V3_GEOMETRY_SOURCE_COUNT,
        "contract_path": FROZEN_V3_GEOMETRY_CONTRACT_RELATIVE_PATH,
        "terminal_audit": copy.deepcopy(_rigid.v3_terminal_audit_binding()),
        "checkpoint_tensor_trace_optimizer_rng_or_runtime_state_reuse_authorized": False,
    }


def rigid_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": RIGID_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": RIGID_TERMINAL_AUDIT_COMMIT,
        "file_sha256": RIGID_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": RIGID_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": RIGID_TERMINAL_AUDIT_BYTE_COUNT,
        "status": RIGID_TERMINAL_AUDIT_STATUS,
        "classification": RIGID_TERMINAL_AUDIT_CLASSIFICATION,
    }


def model_config() -> dict[str, Any]:
    return copy.deepcopy(MODEL_CONFIG)


def objective_contract() -> dict[str, Any]:
    return copy.deepcopy(OBJECTIVE_CONTRACT)


def optimizer_contract() -> dict[str, Any]:
    return copy.deepcopy(OPTIMIZER_CONTRACT)


def build_schedule_identity() -> dict[str, Any]:
    return _v3.build_schedule_identity()


def runtime_authorization_template() -> dict[str, Any]:
    return _v3.runtime_authorization_template()


def _alias_metric(
    target: dict[str, Any], canonical: str, *aliases: str
) -> None:
    if canonical in target:
        return
    for alias in aliases:
        if alias in target:
            target[canonical] = target[alias]
            return


def _base_metrics(update: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Supply inherited V3 names without changing event metric meanings."""

    value = copy.deepcopy(dict(metrics))
    _alias_metric(
        value, "latent_prediction_objective_formula_exact",
        "event_objective_formula_exact",
    )
    _alias_metric(
        value, "same_action_contrast_formula_exact",
        "target_and_context_contrast_formula_exact",
        "target_contrast_objective_formula_exact",
    )
    _alias_metric(value, "latent_prediction_loss", "event_balanced_energy")
    _alias_metric(
        value, "executed_action_beats_hardest_wrong_family_count",
        "hardest_wrong_positive_family_count",
    )
    _alias_metric(
        value, "non_hold_mean_executed_action_energy",
        "mean_non_hold_executed_action_energy",
    )
    _alias_metric(
        value, "non_hold_mean_hold_or_zero_action_energy",
        "mean_non_hold_hold_action_energy",
    )
    _alias_metric(
        value, "same_action_correct_next_deranged_nll", "target_nll"
    )
    _alias_metric(
        value, "same_action_correct_next_strict_win_rate",
        "target_strict_win_rate",
    )
    _alias_metric(
        value, "same_action_correct_next_positive_family_count",
        "target_positive_family_count",
    )
    if update == 400:
        _alias_metric(
            value, "B400_content_sha256", "T400_B400_content_sha256"
        )
        _alias_metric(
            value, "B400_frozen_before_joint_phase",
            "T400_B400_frozen_before_joint_phase",
        )
    return value


def _bool_metric(metrics: Mapping[str, Any], name: str, *aliases: str) -> bool:
    for key in (name, *aliases):
        if key in metrics:
            return _boolean(metrics[key], key)
    raise ValueError(f"missing metric: {name}")


def _observation_accounting_conjuncts(
    update: int, metrics: Mapping[str, Any]
) -> dict[str, bool]:
    """Gate observation work separately from scheduled training work."""

    expected = OBSERVATION_ACCOUNTING_EXPECTATIONS[update]
    return {
        f"{field}_equals_{count}": (
            _integer(_metric(metrics, field), field) == count
        )
        for field, count in expected.items()
    }


def _final_event_conjuncts(
    metrics: Mapping[str, Any], B400: float
) -> dict[str, bool]:
    """Return the exact twenty preregistered event/action conjuncts."""

    event_balanced = _finite(
        _metric(metrics, "event_balanced_energy", "latent_prediction_loss"),
        "event_balanced_energy",
    )
    action_nll = _finite(_metric(metrics, "action_nll"), "action_nll")
    action_ba = _finite(
        _metric(metrics, "action_macro_balanced_accuracy"),
        "action_macro_balanced_accuracy",
    )
    mean_exec = _finite(
        _metric(metrics, "mean_executed_action_energy"),
        "mean_executed_action_energy",
    )
    mean_wrong = _finite(
        _metric(metrics, "mean_wrong_action_energy"),
        "mean_wrong_action_energy",
    )
    nonhold_exec = _finite(
        _metric(metrics, "mean_non_hold_executed_action_energy"),
        "mean_non_hold_executed_action_energy",
    )
    nonhold_hold = _finite(
        _metric(metrics, "mean_non_hold_hold_action_energy"),
        "mean_non_hold_hold_action_energy",
    )

    context_true = _finite(
        _metric(metrics, "context_true_energy"), "context_true_energy"
    )
    context_swap = _finite(
        _metric(metrics, "context_swap_energy"), "context_swap_energy"
    )
    context_nll = _finite(
        _metric(metrics, "context_nll"), "context_nll"
    )
    context_ratio = _finite(
        _metric(metrics, "context_true_to_swap_energy_ratio"),
        "context_true_to_swap_energy_ratio",
    )
    state_true = _finite(
        _metric(metrics, "state_true_energy"), "state_true_energy"
    )
    state_template = _finite(
        _metric(metrics, "state_template_energy"), "state_template_energy"
    )
    state_template_ratio = _finite(
        _metric(metrics, "state_true_to_template_energy_ratio"),
        "state_true_to_template_energy_ratio",
    )
    executed = _finite(
        _metric(metrics, "executed_action_energy"), "executed_action_energy"
    )
    state_only = _finite(
        _metric(metrics, "state_only_energy"), "state_only_energy"
    )
    state_only_ratio = _finite(
        _metric(metrics, "executed_to_state_only_energy_ratio"),
        "executed_to_state_only_energy_ratio",
    )
    two_mode = _finite(
        _metric(metrics, "two_mode_energy", "event_balanced_energy"),
        "two_mode_energy",
    )
    matched = _finite(
        _metric(metrics, "matched_single_mean_energy"),
        "matched_single_mean_energy",
    )
    matched_ratio = _finite(
        _metric(metrics, "two_mode_to_matched_single_ratio"),
        "two_mode_to_matched_single_ratio",
    )
    event_changed = _finite(
        _metric(metrics, "event_changed_energy"), "event_changed_energy"
    )
    zero_changed = _finite(
        _metric(metrics, "zero_changed_energy"), "zero_changed_energy"
    )
    zero_static = _finite(
        _metric(metrics, "zero_static_energy"), "zero_static_energy"
    )
    event_static = _finite(
        _metric(metrics, "event_static_energy"), "event_static_energy"
    )
    mixture_overall = _finite(
        _metric(metrics, "mixture_overall_energy", "two_mode_energy"),
        "mixture_overall_energy",
    )
    zero_overall = _finite(
        _metric(metrics, "zero_overall_energy"), "zero_overall_energy"
    )
    event_overall = _finite(
        _metric(metrics, "event_overall_energy"), "event_overall_energy"
    )
    prior_changed = _finite(
        _metric(metrics, "prior_changed_mean"), "prior_changed_mean"
    )
    prior_static = _finite(
        _metric(metrics, "prior_static_mean"), "prior_static_mean"
    )
    prior_mean = _finite(_metric(metrics, "prior_mean"), "prior_mean")
    posterior_changed = _finite(
        _metric(metrics, "posterior_changed_mean"), "posterior_changed_mean"
    )
    posterior_static = _finite(
        _metric(metrics, "posterior_static_mean"), "posterior_static_mean"
    )
    posterior_mean = _finite(
        _metric(metrics, "posterior_mean"), "posterior_mean"
    )

    accounting_exact = (
        _integer(
            _metric(
                metrics,
                "all_action_predictor_training_forward_count",
            ),
            "all_action_predictor_training_forward_count",
        ) == 2_400
        and _integer(
            _metric(metrics, "context_swap_predictor_training_forward_count"),
            "context_swap_predictor_training_forward_count",
        ) == 2_400
        and _integer(
            _metric(metrics, "semantic_term_evaluation_count"),
            "semantic_term_evaluation_count",
        ) == 4_000
        and _integer(
            _metric(metrics, "event_persistence_term_evaluation_count"),
            "event_persistence_term_evaluation_count",
        ) == 2_400
        and _integer(
            _metric(metrics, "action_term_evaluation_count"),
            "action_term_evaluation_count",
        ) == 2_400
        and _integer(
            _metric(metrics, "target_term_evaluation_count"),
            "target_term_evaluation_count",
        ) == 2_400
        and _integer(
            _metric(metrics, "context_term_evaluation_count"),
            "context_term_evaluation_count",
        ) == 2_400
        and _integer(
            _metric(metrics, "registered_scalar_term_evaluation_count"),
            "registered_scalar_term_evaluation_count",
        ) == 13_600
        and _integer(
            _metric(
                metrics,
                "combined_objective_evaluation_count",
                "objective_evaluations",
            ),
            "combined_objective_evaluation_count",
        ) == 4_000
        and _integer(
            _metric(metrics, "backward_call_count", "backward_calls"),
            "backward_call_count",
        ) == 4_000
        and _integer(
            _metric(metrics, "online_encoder_lift_training_forward_count"),
            "online_encoder_lift_training_forward_count",
        ) == 10_400
        and _integer(
            _metric(metrics, "semantic_head_training_forward_count"),
            "semantic_head_training_forward_count",
        ) == 8_000
        and _integer(
            _metric(metrics, "target_encoder_lift_training_forward_count"),
            "target_encoder_lift_training_forward_count",
        ) == 7_200
        and _bool_metric(metrics, "warning_policy_exact")
        and _bool_metric(metrics, "state_hash_accounting_exact")
        and _bool_metric(metrics, "receipt_schema_accounting_exact")
        and _bool_metric(
            metrics,
            "access_and_custody_accounting_exact",
            "all_forbidden_access_counts_zero",
        )
    )

    shared_gradients = (
        _integer(
            _metric(metrics, "shared_gradient_ratio_evaluation_count"),
            "shared_gradient_ratio_evaluation_count",
        ) == 600
        and _integer(
            _metric(metrics, "shared_gradient_ratio_pass_count"),
            "shared_gradient_ratio_pass_count",
        ) == 600
        and _integer(
            _metric(metrics, "shared_gradient_ratio_failure_count"),
            "shared_gradient_ratio_failure_count",
        ) == 0
        and _integer(
            _metric(metrics, "action_embedding_dynamics_gradient_update_count"),
            "action_embedding_dynamics_gradient_update_count",
        ) == 600
        and _integer(
            _metric(metrics, "predictor_trunk_dynamics_gradient_update_count"),
            "predictor_trunk_dynamics_gradient_update_count",
        ) == 600
        and _integer(
            _metric(metrics, "event_mean_head_dynamics_gradient_update_count"),
            "event_mean_head_dynamics_gradient_update_count",
        ) == 600
        and _integer(
            _metric(metrics, "event_logit_head_dynamics_gradient_update_count"),
            "event_logit_head_dynamics_gradient_update_count",
        ) == 600
    )

    return {
        "event_01_balanced_energy_at_most_point90_B400": (
            event_balanced <= 0.90 * B400
        ),
        "event_02_action_nll_strictly_below_point95_log9": (
            action_nll < 0.95 * LOG9
        ),
        "event_03_action_macro_balanced_accuracy_strictly_above_two_ninths": (
            action_ba > 2.0 / 9.0
        ),
        "event_04_executed_beats_hardest_wrong_in_at_least_6_families": (
            _integer(
                _metric(metrics, "hardest_wrong_positive_family_count"),
                "hardest_wrong_positive_family_count",
            ) >= 6
        ),
        "event_05_mean_wrong_energy_strictly_above_executed": (
            mean_wrong > mean_exec
        ),
        "event_06_nonhold_HOLD_energy_strictly_above_executed": (
            nonhold_hold > nonhold_exec
        ),
        "event_07_correct_deranged_target_all_thresholds": (
            _finite(_metric(metrics, "target_nll"), "target_nll")
            < 0.95 * LOG2
            and _finite(
                _metric(metrics, "target_strict_win_rate"),
                "target_strict_win_rate",
            ) >= 0.65
            and _integer(
                _metric(metrics, "target_positive_family_count"),
                "target_positive_family_count",
            ) >= 6
        ),
        "event_08_true_context_beats_same_command_swap": (
            context_nll < 0.95 * LOG2
            and context_true <= 0.95 * context_swap
            and context_ratio <= 0.95
            and _finite(
                _metric(metrics, "context_true_strict_win_rate"),
                "context_true_strict_win_rate",
            ) >= 0.65
            and _integer(
                _metric(metrics, "context_positive_family_count"),
                "context_positive_family_count",
            ) >= 6
        ),
        "event_09_true_state_beats_action_template": (
            state_true <= 0.95 * state_template
            and state_template_ratio <= 0.95
            and _integer(
                _metric(metrics, "state_positive_family_count"),
                "state_positive_family_count",
            ) >= 6
        ),
        "event_10_executed_action_beats_state_only_mixture": (
            executed <= 0.95 * state_only
            and state_only_ratio <= 0.95
            and _integer(
                _metric(metrics, "state_only_positive_family_count"),
                "state_only_positive_family_count",
            ) >= 6
        ),
        "event_11_two_mode_beats_matched_single_mean": (
            two_mode <= 0.98 * matched
            and matched_ratio <= 0.98
            and _integer(
                _metric(metrics, "matched_single_positive_family_count"),
                "matched_single_positive_family_count",
            ) >= 6
        ),
        "event_12_learned_event_specializes_changed_support": (
            event_changed <= 0.90 * zero_changed
            and _integer(
                _metric(
                    metrics, "event_over_zero_changed_positive_family_count"
                ),
                "event_over_zero_changed_positive_family_count",
            ) >= 6
        ),
        "event_13_zero_event_specializes_static_support": (
            zero_static <= 0.95 * event_static
            and _integer(
                _metric(
                    metrics, "zero_over_event_static_positive_family_count"
                ),
                "zero_over_event_static_positive_family_count",
            ) >= 6
        ),
        "event_14_mixture_beats_each_unmixed_component": (
            mixture_overall < zero_overall
            and mixture_overall < event_overall
            and _integer(
                _metric(metrics, "mixture_beats_zero_family_count"),
                "mixture_beats_zero_family_count",
            ) >= 6
            and _integer(
                _metric(metrics, "mixture_beats_event_family_count"),
                "mixture_beats_event_family_count",
            ) >= 6
        ),
        "event_15_changed_weighted_mu_event_nontrivial": (
            _finite(
                _metric(metrics, "mu_event_changed_abs"),
                "mu_event_changed_abs",
            ) > 1e-4
        ),
        "event_16_learned_prior_state_dependent_and_context_sensitive": (
            prior_changed - prior_static >= 0.05
            and 0.10 <= prior_mean <= 0.90
            and _finite(
                _metric(metrics, "prior_spatial_variance"),
                "prior_spatial_variance",
            ) >= 1e-4
            and _finite(
                _metric(metrics, "prior_context_difference"),
                "prior_context_difference",
            ) >= 0.02
            and _integer(
                _metric(
                    metrics,
                    "prior_context_difference_positive_family_count",
                ),
                "prior_context_difference_positive_family_count",
            ) >= 6
        ),
        "event_17_posterior_changed_static_separation": (
            posterior_changed - posterior_static >= 0.10
        ),
        "event_18_both_posterior_modes_materially_used": (
            0.10 <= posterior_mean <= 0.90
            and _integer(
                _metric(metrics, "posterior_event_and_zero_family_count"),
                "posterior_event_and_zero_family_count",
            ) >= 6
        ),
        "event_19_joint_gradient_routes_exact_for_all_600_updates": (
            shared_gradients
        ),
        "event_20_all_work_warning_state_access_and_receipts_exact": (
            accounting_exact
        ),
    }


_REPLACED_V3_PREDICTIVE_CONJUNCTS = {
    "latent_prediction_loss_at_most_point90_B400",
    "action_nll_strictly_below_point95_log9",
    "action_macro_balanced_accuracy_strictly_above_two_ninths",
    "executed_action_beats_hardest_wrong_in_at_least_6_families",
    "mean_wrong_action_energy_strictly_above_executed",
    "non_hold_mean_hold_energy_strictly_above_executed",
    "same_action_correct_next_deranged_nll_strictly_below_point95_log2",
    "same_action_correct_next_strict_win_rate_at_least_point65",
    "same_action_correct_next_positive_in_at_least_6_families",
}


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    prior_metrics: Mapping[int | str, Mapping[str, Any]] | None = None,
    *,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate inherited perception plus registered event-delta gates."""

    if update not in GATE_CONTROLS:
        raise ValueError("update must be one of 0, 100, 400, or 1000")
    normalized = _base_metrics(update, metrics)
    normalized_prior = None
    if prior_metrics is not None:
        normalized_prior = {
            int(key): _base_metrics(int(key), value)
            for key, value in prior_metrics.items()
        }
    # The rigid contract is an operational/source-custody predecessor only.
    # Scientific gate inheritance comes directly from frozen V3.
    value = copy.deepcopy(_v3.evaluate_gate(
        update,
        normalized,
        prior_metrics=normalized_prior,
        prior_gates_passed=prior_gates_passed,
    ))
    conjuncts = dict(value["conjuncts"])
    if update == 1_000:
        for field in _REPLACED_V3_PREDICTIVE_CONJUNCTS:
            conjuncts.pop(field, None)
    conjuncts.update(_observation_accounting_conjuncts(update, metrics))

    if update == 0:
        for field in (
            "event_tensor_shapes_exact",
            "fixed_mode_identities_exact",
            "output_parameter_action_symmetry_exact",
            "synthetic_positive_temperature_stable_energy_exact",
            "zero_mean_persistence_identity_exact",
            "synthetic_action_nll_log9_exact",
            "synthetic_action_macro_balanced_accuracy_one_ninth_exact",
            "event_initialization_and_gradient_witness_exact",
            "action_embedding_exact_zero_at_update_zero",
            "event_prior_bitwise_float32_half_at_update_zero",
            "mean_and_logit_head_initialization_and_rng_order_exact",
            "online_encoder_lift_and_each_predictor_submodule_gradient_witness_exact",
            "reviewed_model_source_synthetic_encoder_lift_and_each_residual_conv_witness_bound",
            "no_scale_inverse_pair_posterior_transport_or_future_bypass",
        ):
            conjuncts[field] = _boolean(metrics.get(field), field)
    elif update == 400:
        T400 = _finite(_metric(metrics, "T400"), "T400")
        B400 = _finite(_metric(metrics, "B400"), "B400")
        actual_nll = _finite(
            _metric(metrics, "actual_population_action_nll"),
            "actual_population_action_nll",
        )
        actual_ba = _finite(
            _metric(metrics, "actual_population_action_macro_balanced_accuracy"),
            "actual_population_action_macro_balanced_accuracy",
        )
        conjuncts.update({
            "T400_finite_strictly_positive": T400 > 0.0,
            "B400_finite_strictly_positive_event_reference": B400 > 0.0,
            "T400_B400_joint_content_hash_frozen": is_sha256(
                _metric(metrics, "T400_B400_content_sha256")
            ),
            "T400_B400_frozen_before_joint_phase": _bool_metric(
                metrics, "T400_B400_frozen_before_joint_phase"
            ),
            "calibration_model_state_preserved": _bool_metric(
                metrics, "calibration_model_state_preserved"
            ),
            "calibration_target_state_preserved": _bool_metric(
                metrics, "calibration_target_state_preserved"
            ),
            "calibration_optimizer_state_preserved": _bool_metric(
                metrics, "calibration_optimizer_state_preserved"
            ),
            "calibration_CPU_RNG_preserved": _bool_metric(
                metrics, "calibration_cpu_rng_preserved"
            ),
            "calibration_accelerator_RNG_preserved": _bool_metric(
                metrics, "calibration_accelerator_rng_preserved"
            ),
            "actual_population_action_energy_bitwise_symmetric": _bool_metric(
                metrics, "actual_population_action_energy_bitwise_symmetric"
            ),
            "actual_population_action_nll_matches_log9": math.isclose(
                actual_nll, LOG9, rel_tol=2e-6, abs_tol=2e-6
            ),
            "actual_population_action_macro_BA_matches_one_ninth": (
                abs(actual_ba - (1.0 / 9.0)) <= 1e-12
            ),
            "event_prior_bitwise_float32_half": _bool_metric(
                metrics, "event_prior_bitwise_half"
            ),
        })
    elif update == 1_000:
        if normalized_prior is None:
            raise ValueError("update 1000 prior metrics are required")
        prior_400 = normalized_prior.get(400)
        if not isinstance(prior_400, Mapping):
            raise ValueError("update 400 prior metrics are required")
        B400 = _finite(_metric(prior_400, "B400"), "update_400.B400")
        conjuncts.update(_final_event_conjuncts(metrics, B400))

    passed = all(conjuncts.values())
    fail_control, pass_control = GATE_CONTROLS[update]
    value.update({
        "kind": {
            0: "structural_only",
            100: "perception_learning_health",
            400: "perception_calibration_and_anti_collapse",
            1_000: "joint_perception_and_two_mode_event_delta_qualification",
        }[update],
        "passed": passed,
        "control": pass_control if passed else fail_control,
        "conjuncts": conjuncts,
        "thresholds": (
            {} if update == 0 else copy.deepcopy(GATE_THRESHOLDS[update])
        ),
        "all_conjunctive": True,
        "scientific_gate_evidence": True,
    })
    return value


def evaluate_update_401_phase_switch(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(metrics))
    _alias_metric(
        normalized, "joint_objective_formula_exact",
        "event_joint_objective_formula_exact",
    )
    value = copy.deepcopy(_v3.evaluate_update_401_phase_switch(normalized))
    conjuncts = dict(value["conjuncts"])
    for field in (
        "all_action_logits_derived_mechanically_from_forward_event_energies",
        "semantic_gradient_finite_nonzero",
        "dynamics_gradient_finite_nonzero",
        "action_embedding_gradient_finite_nonzero",
        "predictor_trunk_gradient_finite_nonzero",
        "event_mean_head_gradient_finite_nonzero",
        "event_logit_head_gradient_finite_nonzero",
        "target_gradient_and_optimizer_membership_zero",
        "unit_weighted_joint_objective_exact",
    ):
        conjuncts[field] = _boolean(metrics.get(field), field)
    passed = all(conjuncts.values())
    value.update({
        "kind": "non_scientific_joint_event_integrity_receipt",
        "passed": passed,
        "control": PHASE_SWITCH_CONTROLS[1 if passed else 0],
        "conjuncts": conjuncts,
        "scientific_gate_evidence": False,
    })
    return value


SOURCE_ONLY_REVIEW_CHECKS = {
    "generated_inputs_or_runtime_rows_opened": [],
    "checkpoints_tensors_traces_or_predecessor_runtime_outputs_opened": [],
    "accelerators_queried_or_used": [],
    "navigation_g2_heldout_sealed_or_rejected_material_opened": [],
}


def science_contract() -> dict[str, Any]:
    """Return the complete frozen science and one-attempt lifecycle."""

    frozen = _v3.science_contract()
    return {
        "schema": f"{SCHEMA_PREFIX}_science_contract_v1",
        "repository_goal": frozen["repository_goal"],
        "scientific_question": (
            "can_an_end_to_end_jointly_trained_RGB_encoder_and_fixed_scale_"
            "two_mode_event_delta_predictor_distinguish_executed_action_"
            "consequences_from_persistence_generic_temporal_matching_and_"
            "action_only_templates_while_retaining_geometry_grounded_perception"
        ),
        "governing_documents": {
            "event_delta_preregistration": preregistration_binding(),
            "frozen_v3_geometry": frozen_v3_geometry_binding(),
            "closed_rigid_predecessor": rigid_terminal_audit_binding(),
        },
        "frozen_scientific_base": {
            "experiment": _v3.EXPERIMENT_ID,
            "source_commit": FROZEN_V3_GEOMETRY_SOURCE_COMMIT,
            "source_count": FROZEN_V3_GEOMETRY_SOURCE_COUNT,
            "contract_path": FROZEN_V3_GEOMETRY_CONTRACT_RELATIVE_PATH,
            "preserved": [
                "RGB_encoder_and_N320_encoder_state_only_initialization",
                "geometry_anchored_deformable_BEV_lift_and_semantic_head",
                "EMA_target_and_momentum",
                "data_roles_rows_endpoints_families_negatives_loader_and_schedule",
                "seed_microbatch_effective_batch_optimizer_clips_and_caps",
                "perception_gates_and_target_anti_collapse_retention",
            ],
            "checkpoint_tensor_trace_optimizer_registry_observation_rng_receipt_or_runtime_output_reused": False,
        },
        "rigid_operational_source_closure": {
            "scientific_mechanism_reused": False,
            "source_commit": FROZEN_RIGID_SOURCE_COMMIT,
            "source_count": FROZEN_RIGID_SOURCE_COUNT,
            "source_closure_checker": (
                FROZEN_RIGID_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
            ),
            "terminal_audit": rigid_terminal_audit_binding(),
            "transport_family_closed": True,
        },
        "runtime_inputs": copy.deepcopy(frozen["runtime_inputs"]),
        "data": copy.deepcopy(frozen["data"]),
        "model": model_config(),
        "call_graph": copy.deepcopy(CALL_GRAPH_CONTRACT),
        "objective": objective_contract(),
        "optimizer": optimizer_contract(),
        "schedule": build_schedule_identity(),
        "registered_scientific_delta": {
            "count": 1,
            "name": "fixed_scale_two_mode_event_delta_predictive_state",
            "fixed_modes": ["ZERO_EVENT", "LEARNED_EVENT"],
            "learned_event_logit": True,
            "predictor_parameter_count": PREDICTOR_PARAMETER_COUNT,
            "predictor_parameter_tensor_count": (
                PREDICTOR_PARAMETER_TENSOR_COUNT
            ),
            "predictor_ordered_parameter_names": list(
                PREDICTOR_ORDERED_PARAMETER_NAMES
            ),
            "includes_registered_target_objective_and_ablation_observations": True,
            "learned_variance_scale_temperature_codebook_pair_posterior_inverse_classifier_transport_flow_warp_correspondence_or_third_mode": False,
            "encoder_and_predictor_jointly_trained_updates_401_1000": True,
            "separate_frozen_encoder_predictor_fit": False,
        },
        "gates": {
            "observations": [0, 100, 400, 1_000],
            "phase_switch_integrity_receipt": 401,
            "controls": {
                str(update): list(controls)
                for update, controls in GATE_CONTROLS.items()
            },
            "phase_switch_controls": list(PHASE_SWITCH_CONTROLS),
            "thresholds": {
                str(update): copy.deepcopy(thresholds)
                for update, thresholds in GATE_THRESHOLDS.items()
            },
            "frozen_perception_and_integrity_source": (
                FROZEN_V3_GEOMETRY_CONTRACT_RELATIVE_PATH
            ),
            "rigid_scientific_gate_inheritance": False,
            "update_1000_event_conjunct_count": 20,
            "raw_vs_normalized_metrics": {
                "action_gate": "raw_nine_way_cross_entropy",
                "target_gate": "raw_correct_vs_deranged_cross_entropy",
                "context_gate": "raw_true_vs_swapped_cross_entropy",
                "objective_action": "raw_action_cross_entropy_divided_by_log9",
                "objective_target": "raw_target_cross_entropy_divided_by_log2",
                "objective_context": "raw_context_cross_entropy_divided_by_log2",
            },
            "all_conditions_conjunctive": True,
            "stop_at_first_applicable_failure": True,
        },
        "work_accounting": copy.deepcopy(WORK_ACCOUNTING_CONTRACT),
        "warning_policy": copy.deepcopy(WARNING_POLICY),
        "receipts": {
            "schemas": {
                "reservation": RESERVATION_SCHEMA,
                "metrics": METRICS_SCHEMA,
                "artifact": ARTIFACT_SCHEMA,
                "access": ACCESS_SCHEMA,
                "result": RESULT_SCHEMA,
                "completion": COMPLETION_SCHEMA,
                "failure": FAILURE_SCHEMA,
            },
            "normal_terminal_inventory": list(NORMAL_RECEIPT_PATHS),
            "operational_exception_inventory": list(
                OPERATIONAL_FAILURE_RECEIPT_PATHS
            ),
            "operational_failure_receipt_must_record": [
                "first_failure",
                "all_work_counts",
                "active_gpu_seconds",
                "source_input_and_state_bindings",
                "warnings_and_access_counts",
                "checkpoint_bindings_and_read_after_write_counts",
                "downstream_authority",
            ],
            "canonical_ASCII_finite_duplicate_safe_JSON": True,
            "single_trailing_newline": True,
            "file_mode": "0444",
            "terminal_root_mode": "0555",
        },
        "lifecycle": {
            "attempt_index": ATTEMPT_INDEX,
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "maximum_updates": MAXIMUM_UPDATES,
            "maximum_presentations": MAXIMUM_PRESENTATIONS,
            "gpu_active_minutes_maximum": GPU_ACTIVE_TIME_CAP_MINUTES,
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "output_root_must_be_absent_before_mode_0700_reservation": True,
            "fresh_model_from_N320_encoder_state_only": True,
            "checkpoint_and_training_trace_write_only": True,
            "retry_resume_alternate_seed_extension_same_root_reuse_second_attempt_repair_or_integrity_replacement": False,
        },
        "terminal_family_closure": {
            "failure_after_any_scientific_presentation_closes_fixed_two_mode_event_delta_family": True,
            "zero_presentation_operational_failure_authorizes_no_retry": True,
            "temperature_changed_static_weighting_head_depth_coefficient_threshold_or_mode_count_variant_authorized": False,
            "larger_codebook_extra_mode_pair_posterior_or_distributional_successor_authorized": False,
            "complete_pass_authorizes_only_independent_terminal_audit_and_later_decision": True,
        },
        "authority": copy.deepcopy(DOWNSTREAM_DENIALS),
    }


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    """Validate only public, committed governing source documents."""

    result = _rigid.validate_governing_documents(root)
    prereg_raw = _read_regular_source(root / PREREGISTRATION_RELATIVE_PATH)
    if (
        len(prereg_raw) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(prereg_raw).hexdigest()
        != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("event-delta preregistration changed")

    audit_raw = _read_regular_source(root / RIGID_TERMINAL_AUDIT_RELATIVE_PATH)
    if (
        len(audit_raw) != RIGID_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit_raw).hexdigest()
        != RIGID_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("rigid terminal audit raw identity changed")
    audit = parse_canonical_json(audit_raw, name="rigid terminal audit")
    attempt = audit.get("attempt", {})
    runtime = audit.get("runtime_and_access", {})
    conclusion = audit.get("scientific_conclusion", {})
    closure = audit.get("closure", {})
    if (
        audit.get("content_sha256")
        != RIGID_TERMINAL_AUDIT_CONTENT_SHA256
        or audit.get("status") != RIGID_TERMINAL_AUDIT_STATUS
        or audit.get("classification")
        != RIGID_TERMINAL_AUDIT_CLASSIFICATION
        or attempt.get("attempt_index") != 1
        or attempt.get("maximum_attempts") != 1
        or attempt.get("retry_or_resume_authorized") is not False
        or runtime.get("updates") != 1_000
        or runtime.get("presentations") != 16_000
        or runtime.get("online_optimizer_updates") != 1_000
        or runtime.get("target_ema_updates") != 1_000
        or runtime.get("joint_optimizer_updates") != 600
        or runtime.get("objective_evaluations") != 4_000
        or runtime.get("backward_calls") != 4_000
        or runtime.get("shared_gradient_gate_passes") != 600
        or runtime.get("all_forbidden_semantic_counts_zero") is not True
        or runtime.get("g2_navigation_heldout_sealed_open_count") != 0
        or runtime.get("rejected_checkpoint_open_count") != 0
        or conclusion.get("scientific_evidence_produced") is not True
        or conclusion.get("mechanism_was_tested") is not True
        or conclusion.get("mechanism_passed") is not False
        or conclusion.get("final_gate_passed") is not False
        or conclusion.get("failed_conjunct_count") != 7
        or closure.get("attempt_consumed") is not True
        or closure.get("rigid_transport_v1_closed") is not True
        or closure.get("transport_family_closed") is not True
        or closure.get(
            "retry_resume_repair_second_attempt_alternate_seed_or_same_root_reuse_authorized"
        ) is not False
        or closure.get(
            "checkpoint_trace_state_receipt_or_runtime_output_reuse_authorized"
        ) is not False
    ):
        raise PermissionError("rigid terminal conclusion changed")
    result.update({
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
        RIGID_TERMINAL_AUDIT_RELATIVE_PATH:
            RIGID_TERMINAL_AUDIT_FILE_SHA256,
    })
    return result


def validate_source_manifest(raw: bytes, root: Path = ROOT) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="event-delta source manifest")
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
        or value.get("excluded_runtime_categories")
        != list(PROHIBITED_RUNTIME_CATEGORIES)
        or value.get("source_paths") != list(SOURCE_PATHS)
        or type(bindings) is not list
        or len(bindings) != len(SOURCE_PATHS)
        or value.get("source_bindings_sha256")
        != canonical_json_sha256(bindings)
        or value.get("source_count") != 98
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
    ):
        raise PermissionError("event-delta source manifest contract changed")
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
            raise PermissionError("event-delta source binding changed")
        payload = _read_regular_source(root / relative)
        if (
            len(payload) != binding["byte_count"]
            or hashlib.sha256(payload).hexdigest() != binding["file_sha256"]
        ):
            raise PermissionError(f"manifest-bound source changed: {relative}")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    result = {
        binding["path"]: binding["file_sha256"]
        for binding in manifest["source_bindings"]
    }
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def validate_review(
    raw: bytes,
    manifest_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="event-delta source review")
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
        raise PermissionError("event-delta review manifest binding changed")
    fields = {
        "schema", "status", "implementation_authors", "reviewer",
        "source_freeze_commit", "reviewed_sources", "source_manifest",
        "preregistration", "frozen_v3_geometry", "rigid_terminal_audit",
        "science_contract", "cpu_synthetic_preflight", "source_only_checks",
        "scientific_checks", "findings", "authority", "content_sha256",
    }
    reviewer = value.get("reviewer")
    _source_freeze_commit(
        value.get("source_freeze_commit"), name="review.source_freeze_commit"
    )
    if (
        set(value) != fields
        or value.get("schema") != REVIEW_SCHEMA
        or value.get("status") != "PASS_SOURCE_AND_SCIENCE"
        or value.get("implementation_authors") != list(IMPLEMENTATION_AUTHORS)
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer in IMPLEMENTATION_AUTHORS
        or value.get("reviewed_sources") != current_source_bindings(root)
        or value.get("source_manifest") != expected_manifest
        or value.get("preregistration") != preregistration_binding()
        or value.get("frozen_v3_geometry") != frozen_v3_geometry_binding()
        or value.get("rigid_terminal_audit") != rigid_terminal_audit_binding()
        or value.get("science_contract") != science_contract()
        or value.get("cpu_synthetic_preflight")
        != EVENT_DELTA_PREFLIGHT_REQUIREMENTS
        or value.get("source_only_checks") != SOURCE_ONLY_REVIEW_CHECKS
        or value.get("scientific_checks") != SCIENTIFIC_REVIEW_CHECKS
        or value.get("findings") != []
        or value.get("authority") != REVIEW_AUTHORITY
    ):
        raise PermissionError("event-delta source review changed")
    return dict(value)


def validate_authorization(
    raw: bytes,
    review_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="event-delta authorization")
    expected_review = _validate_artifact_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    review_raw = _read_regular_source(root / REVIEW_RELATIVE_PATH)
    parsed_review = parse_canonical_json(
        review_raw, name="event-delta source review"
    )
    if expected_review != artifact_binding(
        REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=parsed_review["content_sha256"],
    ):
        raise PermissionError("event-delta authorization review changed")
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    manifest_binding = artifact_binding(
        SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    )
    review = validate_review(review_raw, manifest_binding, root=root)
    fields = {
        "schema", "status", "authorizer", "source_freeze_commit",
        "independent_source_review", "preregistration",
        "frozen_v3_geometry", "rigid_terminal_audit", "runtime_inputs",
        "experiment", "authority", "content_sha256",
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
        or value.get("frozen_v3_geometry") != frozen_v3_geometry_binding()
        or value.get("rigid_terminal_audit") != rigid_terminal_audit_binding()
        or value.get("runtime_inputs") != runtime_authorization_template()
        or value.get("experiment") != science_contract()
        or value.get("authority") != EXECUTION_AUTHORITY
    ):
        raise PermissionError("event-delta execution authorization changed")
    return dict(value)


__all__ = sorted({
    *(name for name in _rigid.__all__ if name not in _INHERITED_EXCLUSIONS),
    *(name for name in globals() if name.isupper() and not name.startswith("_")),
    "build_schedule_identity", "current_source_bindings", "evaluate_gate",
    "evaluate_update_401_phase_switch", "frozen_v3_geometry_binding",
    "model_config", "objective_contract", "optimizer_contract",
    "preregistration_binding", "rigid_terminal_audit_binding",
    "runtime_authorization_template", "science_contract",
    "validate_authorization", "validate_governing_documents",
    "validate_review", "validate_source_manifest",
})
