"""Source-only contract for the global rigid-BEV transport joint JEPA V1.

The experiment preserves the completed geometry-anchored V3 data, schedule,
objectives, optimizer, gates, scalar-safe state hashing, and import-lifetime
fix.  Its sole scientific change is the predictor defined in the separately
bound model source: a learned nine-row rigid transform table followed by one
shared local corrector.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V3_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
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


_v3 = _source_module(
    "_lewm_global_rigid_bev_transport_frozen_v3_contract",
    FROZEN_V3_CONTRACT_RELATIVE_PATH,
)
for _name in _v3.__all__:
    globals()[_name] = getattr(_v3, _name)

_read_regular_source = _v3._read_regular_source
_source_freeze_commit = _v3._source_freeze_commit
_validate_artifact_binding = _v3._validate_artifact_binding


IMPLEMENTATION_AUTHORS = (
    "/root",
    "/root/transport_prereg_design",
    "/root/v3_successor_assessment",
)
SCHEMA_PREFIX = (
    "lewm_go2_rgb_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1"
)
EXPERIMENT_ID = (
    "geometry_anchored_global_action_indexed_rigid_bev_transport_joint_jepa_v1"
)

FROZEN_V3_SOURCE_COUNT = 84
FROZEN_V3_SOURCE_COMMIT = "ebcde189628b1a7040ffaf95aafaf9fd8f404fc4"
FROZEN_V3_EXECUTION_AUTHORIZATION_COMMIT = (
    "3681264a7365d48ad43cbb75e73dba290b8b0134"
)
FROZEN_V3_RUNNER_RELATIVE_PATH = _v3.RUNNER_RELATIVE_PATH
FROZEN_V3_LAUNCHER_RELATIVE_PATH = _v3.LAUNCHER_RELATIVE_PATH
FROZEN_V3_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _v3.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1_preregistration_2026-07-27.md"
)
PREREGISTRATION_COMMIT = "964aab0eb382bc99765bd074cdc999e2128a718e"
PREREGISTRATION_FILE_SHA256 = (
    "90bf02ecf88a8ae3d691ca56714556d6b7cbf903a4030e0b05c6806c485bf5bb"
)
PREREGISTRATION_BYTE_COUNT = 17_168

V3_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement_terminal_audit_"
    "2026-07-27.json"
)
V3_TERMINAL_AUDIT_COMMIT = "6b48b528c53766276f4912626728611910837a92"
V3_TERMINAL_AUDIT_FILE_SHA256 = (
    "bbb1d82faefc62c0358df531941ab07f2b3253d274eca2156df378ffb17a52c4"
)
V3_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "595ac5198edfcba196ced8213c3f83ff9a5fa2c8100231b028bb99690c8a5d2b"
)
V3_TERMINAL_AUDIT_BYTE_COUNT = 10_661
V3_TERMINAL_AUDIT_STATUS = (
    "PASS_COMPLETE_SCIENTIFIC_GATE_FAILURE_V3_CONSUMED_CLOSED_NO_RETRY"
)
V3_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_COMPLETE_BUDGET_SCIENTIFIC_QUALIFICATION_FAILURE_WITH_POST_"
    "RETURN_WARNING_RECEIPT_DEFECT"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1.py"
)
MODEL_CLASS_NAME = (
    "GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1_source_closure.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1.py"
)
CONTRACT_RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1_contract_runner.py"
)
CONTRACT_TEST_RELATIVE_PATH = CONTRACT_RUNNER_TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = CONTRACT_RUNNER_TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = CONTRACT_RUNNER_TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = CONTRACT_RUNNER_TEST_RELATIVE_PATH

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1_source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1_source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1_execution_authorization_2026-07-27.json"
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
REUSED_SOURCE_PATHS = tuple(sorted(set(_v3.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if (
    len(REUSED_SOURCE_PATHS) != FROZEN_V3_SOURCE_COUNT
    or len(ADDITIVE_SOURCE_PATHS) != 7
    or len(SOURCE_PATHS) != 91
):
    raise PermissionError("rigid transport closure must be V3 plus seven files")
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
    "AUTHORIZED_ONE_EXACT_GEOMETRY_ANCHORED_GLOBAL_ACTION_INDEXED_RIGID_BEV_"
    "TRANSPORT_JOINT_JEPA_V1"
)
OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_GEOMETRY_ANCHORED_GLOBAL_ACTION_INDEXED_RIGID_BEV_TRANSPORT_"
    "JOINT_JEPA_V1_OPERATIONAL_OR_INTEGRITY_FAILURE_NO_RETRY"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_geometry_anchored_global_action_indexed_rigid_bev_"
    "transport_joint_jepa_v1/attempt_v1"
)

EXECUTION_AUTHORITY = {
    key: copy.deepcopy(value)
    for key, value in _v3.EXECUTION_AUTHORITY.items()
    if key not in {
        "science_identical_runtime_import_integrity_replacement_only",
        "science_identical_scalar_tensor_state_hash_integrity_replacement_only",
        "v1_retry_authorized",
        "v2_retry_authorized",
        "v2_resume_or_state_reuse_authorized",
    }
}
EXECUTION_AUTHORITY.update({
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "scientific_successor_fresh_model_only": True,
    "v3_checkpoint_tensor_trace_optimizer_rng_or_runtime_state_reuse_authorized": False,
    "single_registered_rigid_transport_attempt_only": True,
})
SOURCE_ONLY_AUTHORITY = {
    **_v3.SOURCE_ONLY_AUTHORITY,
    "rigid_transport_cpu_synthetic_preflight_authorized": True,
    "warning_canonicalization_cpu_preflight_authorized": True,
}
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)

GATE_CONTROLS = {
    0: (
        "FAIL_UPDATE_ZERO_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_JEPA_V1_"
        "STRUCTURAL_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_ZERO_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_JEPA_"
        "V1_STRUCTURAL_GATE",
    ),
    100: (
        "FAIL_UPDATE_100_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_JEPA_V1_"
        "PERCEPTION_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_100_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_JEPA_"
        "V1_PERCEPTION_GATE",
    ),
    400: (
        "FAIL_UPDATE_400_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_JEPA_V1_ANTI_"
        "COLLAPSE_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_400_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_JEPA_"
        "V1_ANTI_COLLAPSE_GATE",
    ),
    1_000: (
        "FAIL_UPDATE_1000_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_JEPA_V1_"
        "QUALIFICATION_GATE_TERMINAL_NO_RETRY",
        "PASS_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_JEPA_V1_MECHANISM_ONLY",
    ),
}
CONTROL_UPDATE_ZERO_FAIL = GATE_CONTROLS[0][0]
CONTROL_UPDATE_100_FAIL = GATE_CONTROLS[100][0]
CONTROL_UPDATE_400_FAIL = GATE_CONTROLS[400][0]
CONTROL_UPDATE_1000_FAIL = GATE_CONTROLS[1_000][0]
CONTROL_PASS = GATE_CONTROLS[1_000][1]
FAILURE_CONTROLS = tuple(value[0] for value in GATE_CONTROLS.values())
CONTROL_FAIL_JOINT_GRADIENT = (
    "FAIL_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_SHARED_GRADIENT_CONTRIBUTION_"
    "GATE_TERMINAL_NO_RETRY"
)
CONTROL_FAIL_OPERATIONAL = OPERATIONAL_FAILURE_STATUS
PHASE_SWITCH_CONTROLS = (
    "FAIL_UPDATE_401_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_JEPA_V1_PHASE_SWITCH_"
    "TERMINAL_NO_RETRY",
    "CONTINUE_AFTER_UPDATE_401_GLOBAL_RIGID_BEV_TRANSPORT_JOINT_JEPA_V1_"
    "PHASE_SWITCH",
)
_GATE_CONTROL_REBIND = {
    old: new
    for update in GATE_CONTROLS
    for old, new in zip(_v3.GATE_CONTROLS[update], GATE_CONTROLS[update], strict=True)
}
_PHASE_CONTROL_REBIND = dict(zip(
    _v3.PHASE_SWITCH_CONTROLS, PHASE_SWITCH_CONTROLS, strict=True
))

PREDICTOR_PARAMETER_COUNT = 184_667
PREDICTOR_PARAMETER_TENSOR_COUNT = 11
PREDICTOR_ORDERED_PARAMETER_NAMES = (
    "predictor.raw_twist",
    "predictor.residual_blocks.0.conv1.weight",
    "predictor.residual_blocks.0.conv1.bias",
    "predictor.residual_blocks.0.conv2.weight",
    "predictor.residual_blocks.0.conv2.bias",
    "predictor.residual_blocks.1.conv1.weight",
    "predictor.residual_blocks.1.conv1.bias",
    "predictor.residual_blocks.1.conv2.weight",
    "predictor.residual_blocks.1.conv2.bias",
    "predictor.residual_head.weight",
    "predictor.residual_head.bias",
)

WARNING_POLICY = {
    "allowed_category": "UserWarning",
    "allowed_base_message": _v3._v2._v1.ROCM_GRID_SAMPLE_DETERMINISM_WARNING
        if hasattr(_v3._v2._v1, "ROCM_GRID_SAMPLE_DETERMINISM_WARNING")
        else (
            "grid_sampler_2d_backward_cuda does not have a deterministic "
            "implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file "
            "an issue at https://github.com/pytorch/pytorch/issues to help us "
            "prioritize adding deterministic support for this operation."
        ),
    "optional_suffix_regex": (
        r" \(Triggered internally at /pytorch/aten/src/ATen/"
        r"Context\.cpp:[0-9]+\.\)"
    ),
    "canonicalize_optional_suffix_only": True,
    "reject_every_other_warning": True,
    "post_callable_scientific_result_must_be_retained": True,
}

RIGID_TRANSPORT_PREFLIGHT_REQUIREMENTS = {
    "cpu_synthetic_torch_only": True,
    "predictor_parameter_count": PREDICTOR_PARAMETER_COUNT,
    "predictor_parameter_tensor_count": PREDICTOR_PARAMETER_TENSOR_COUNT,
    "ordered_parameter_names": list(PREDICTOR_ORDERED_PARAMETER_NAMES),
    "raw_twist_shape": [9, 3],
    "raw_twist_initially_exact_zero": True,
    "residual_head_initially_exact_zero": True,
    "translation_cell_bound_absolute": 8.0,
    "yaw_radian_bound_absolute": "pi/4",
    "reference_atol_rtol": [2e-5, 2e-5],
    "rotation_identity_and_determinant_atol_rtol": [2e-6, 2e-6],
    "bound_absolute_tolerance": 1e-6,
    "single_twist_row_locality_checked": True,
    "action_permutation_equivariance_checked": True,
    "global_spatial_impulse_transport_checked": True,
    "shared_corrector_identity_checked": True,
    "initial_gradient_routes_checked": True,
    "residual_block_reachability_witness_checked": True,
    "all_nine_action_objective_gradients_checked": True,
    "warmup_predictor_zero_activity_checked": True,
    "generated_inputs_or_runtime_rows_opened": [],
    "checkpoints_tensors_traces_or_v3_runtime_outputs_opened": [],
    "accelerators_queried_or_used": [],
    "navigation_g2_heldout_sealed_or_rejected_material_opened": [],
}
SCIENTIFIC_REVIEW_CHECKS = {
    "preregistration_and_completed_v3_terminal_audit_exact": True,
    "all_84_frozen_v3_sources_and_seven_additive_sources_rehashed": True,
    "sole_scientific_delta_is_registered_predictor_replacement": True,
    "encoder_lift_semantic_target_data_seed_schedule_and_initialization_preserved": True,
    "objectives_optimizer_clipping_ema_caps_and_gate_thresholds_preserved": True,
    "gate_controls_and_runtime_receipt_schemas_rebound_to_new_experiment": True,
    "scalar_safe_state_hash_and_runtime_import_root_lifetime_fix_preserved": True,
    "predictor_inventory_initialization_transform_and_gradient_routes_exact": True,
    "warning_policy_accepts_only_registered_base_and_optional_context_suffix": True,
    "warning_finalization_retains_an_already_computed_scientific_result": True,
    "fresh_absent_before_reservation_root_and_one_attempt_lifecycle_exact": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def v3_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V3_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V3_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V3_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V3_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V3_TERMINAL_AUDIT_BYTE_COUNT,
    }


def model_config() -> dict[str, Any]:
    value = copy.deepcopy(_v3.model_config())
    value["predictor"] = {
        "action_count": 9,
        "action_vocabulary": list(ACTION_VOCABULARY),
        "input": "current_online_RGB_derived_BEV_latent_plus_exact_one_hot_action",
        "operator": "learned_global_action_indexed_rigid_BEV_transport_plus_shared_local_corrector",
        "raw_twist_shape": [9, 3],
        "raw_twist_columns": ["forward_cells", "left_cells", "yaw_radians"],
        "forward_and_left_cell_bound_absolute": 8.0,
        "yaw_bound_absolute": "pi/4",
        "sampling": {
            "affine_grid_align_corners": False,
            "grid_sample_mode": "bilinear",
            "grid_sample_padding_mode": "zeros",
            "grid_sample_align_corners": False,
        },
        "corrector": "two_shared_width64_residual_Conv2d_k3_blocks_plus_shared_zero_residual_head",
        "parameter_count": PREDICTOR_PARAMETER_COUNT,
        "parameter_tensor_count": PREDICTOR_PARAMETER_TENSOR_COUNT,
        "predict_all_actions": True,
        "action_embedding_broadcast_film_action_specific_convolution_or_coordinate_channel": False,
        "pose_odometry_motion_table_goal_map_label_future_or_global_attention_bypass": False,
    }
    return value


def objective_contract() -> dict[str, Any]:
    return _v3.objective_contract()


def optimizer_contract() -> dict[str, Any]:
    return _v3.optimizer_contract()


def build_schedule_identity() -> dict[str, Any]:
    return _v3.build_schedule_identity()


def runtime_authorization_template() -> dict[str, Any]:
    return _v3.runtime_authorization_template()


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    prior_metrics: Mapping[int | str, Mapping[str, Any]] | None = None,
    *,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    value = _v3.evaluate_gate(
        update,
        metrics,
        prior_metrics=prior_metrics,
        prior_gates_passed=prior_gates_passed,
    )
    value = copy.deepcopy(value)
    value["control"] = _GATE_CONTROL_REBIND[str(value["control"])]
    return value


def evaluate_update_401_phase_switch(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(_v3.evaluate_update_401_phase_switch(metrics))
    value["control"] = _PHASE_CONTROL_REBIND[str(value["control"])]
    return value


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v3.science_contract())
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["scientific_question"] = (
        "can_a_learned_global_action_indexed_rigid_transform_make_the_"
        "geometry_grounded_latent_distinguish_the_executed_primitive_from_"
        "all_eight_wrong_primitives_while_genuine_joint_JEPA_retains_"
        "registered_RGB_perception_quality"
    )
    value["governing_documents"] = {
        **value["governing_documents"],
        "rigid_transport_preregistration": preregistration_binding(),
        "completed_v3_terminal_audit": v3_terminal_audit_binding(),
    }
    value["model"] = model_config()
    lifecycle = copy.deepcopy(value["lifecycle"])
    for stale_name in (
        "integrity_replacement_of",
        "v1_retry",
        "v2_retry_or_resume",
    ):
        lifecycle.pop(stale_name, None)
    value["lifecycle"] = {
        **lifecycle,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "scientific_successor_of": _v3.EXPERIMENT_ID,
        "predecessor_checkpoint_tensor_trace_optimizer_rng_or_runtime_state_reused": False,
        "one_final_spatial_transport_parameterization_attempt": True,
    }
    value.pop("integrity_replacement", None)
    value["registered_scientific_delta"] = {
        "count": 1,
        "name": "global_action_indexed_rigid_BEV_transport_predictor",
        "replaced_module": "_LocalActionConditionedPredictorV1",
        "all_other_science_preserved": True,
        "predictor_parameter_count": PREDICTOR_PARAMETER_COUNT,
        "predictor_ordered_parameter_names": list(PREDICTOR_ORDERED_PARAMETER_NAMES),
        "no_transport_parameterization_successor_after_failure": True,
    }
    value["inherited_execution_integrity"] = {
        "v3_scalar_safe_tensor_state_hash": "reshape(-1).view(torch.uint8)",
        "v2_runtime_import_root_lifetime_fix": True,
        "deterministic_warning_policy": dict(WARNING_POLICY),
    }
    return value


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    result = _v3.validate_governing_documents(root)
    prereg_raw = _read_regular_source(root / PREREGISTRATION_RELATIVE_PATH)
    if (
        len(prereg_raw) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(prereg_raw).hexdigest() != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("rigid transport preregistration changed")
    audit_raw = _read_regular_source(root / V3_TERMINAL_AUDIT_RELATIVE_PATH)
    if (
        len(audit_raw) != V3_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit_raw).hexdigest() != V3_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("completed V3 terminal audit raw identity changed")
    # The frozen, raw-bound audit was committed with one additional trailing
    # newline.  Remove exactly that formatting byte only after the exact raw
    # identity check; the remaining payload must still be canonical JSON.
    if not audit_raw.endswith(b"\n\n") or audit_raw.count(b"\n") != 2:
        raise PermissionError("completed V3 terminal audit framing changed")
    audit = parse_canonical_json(
        audit_raw[:-1], name="completed V3 terminal audit"
    )
    runtime = audit.get("runtime_and_access", {})
    conclusion = audit.get("scientific_conclusion", {})
    closure = audit.get("closure", {})
    if (
        audit.get("content_sha256") != V3_TERMINAL_AUDIT_CONTENT_SHA256
        or audit.get("status") != V3_TERMINAL_AUDIT_STATUS
        or audit.get("classification") != V3_TERMINAL_AUDIT_CLASSIFICATION
        or runtime.get("updates") != 1_000
        or runtime.get("presentations") != 16_000
        or runtime.get("joint_optimizer_updates") != 600
        or runtime.get("all_forbidden_access_counts_zero") is not True
        or conclusion.get("scientific_evidence_produced") is not True
        or conclusion.get("mechanism_was_tested") is not True
        or conclusion.get("final_gate_passed") is not False
        or conclusion.get("failed_conjunct_count") != 5
        or closure.get("v3_closed") is not True
        or closure.get("v3_retry_resume_repair_or_same_root_reuse_authorized")
        is not False
    ):
        raise PermissionError("completed V3 terminal conclusion changed")
    result.update({
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
        V3_TERMINAL_AUDIT_RELATIVE_PATH: V3_TERMINAL_AUDIT_FILE_SHA256,
    })
    return result


def validate_source_manifest(raw: bytes, root: Path = ROOT) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="rigid transport source manifest")
    fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
    }
    paths = value.get("source_paths")
    bindings = value.get("source_bindings")
    if (
        set(value) != fields
        or value.get("schema") != SOURCE_MANIFEST_SCHEMA
        or value.get("status") != "PASS_SOURCE_CLOSURE"
        or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value.get("forced_dynamic_sources") != list(SOURCE_PATHS)
        or value.get("excluded_runtime_categories")
        != list(PROHIBITED_RUNTIME_CATEGORIES)
        or paths != list(SOURCE_PATHS)
        or type(bindings) is not list
        or len(bindings) != len(SOURCE_PATHS)
        or value.get("source_count") != len(SOURCE_PATHS)
        or value.get("source_bindings_sha256") != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
    ):
        raise PermissionError("rigid transport source manifest contract changed")
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
            raise PermissionError("rigid transport source binding changed")
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
    value = parse_canonical_json(raw, name="rigid transport source review")
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
        raise PermissionError("rigid transport review manifest binding changed")
    fields = {
        "schema", "status", "implementation_authors", "reviewer",
        "source_freeze_commit", "reviewed_sources", "source_manifest",
        "preregistration", "v3_terminal_audit", "science_contract",
        "cpu_synthetic_preflight", "source_only_checks", "scientific_checks",
        "findings", "authority", "content_sha256",
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
        or value.get("v3_terminal_audit") != v3_terminal_audit_binding()
        or value.get("science_contract") != science_contract()
        or value.get("cpu_synthetic_preflight")
        != RIGID_TRANSPORT_PREFLIGHT_REQUIREMENTS
        or value.get("source_only_checks") != {
            "generated_inputs_or_runtime_rows_opened": [],
            "checkpoints_tensors_traces_or_v3_runtime_outputs_opened": [],
            "accelerators_queried_or_used": [],
            "navigation_g2_heldout_sealed_or_rejected_material_opened": [],
        }
        or value.get("scientific_checks") != SCIENTIFIC_REVIEW_CHECKS
        or value.get("findings") != []
        or value.get("authority") != REVIEW_AUTHORITY
    ):
        raise PermissionError("rigid transport source review changed")
    return dict(value)


def validate_authorization(
    raw: bytes,
    review_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="rigid transport authorization")
    expected_review = _validate_artifact_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    review_raw = _read_regular_source(root / REVIEW_RELATIVE_PATH)
    parsed_review = parse_canonical_json(review_raw, name="rigid transport review")
    if expected_review != artifact_binding(
        REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=parsed_review["content_sha256"],
    ):
        raise PermissionError("rigid transport authorization review changed")
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
        "independent_source_review", "preregistration", "v3_terminal_audit",
        "runtime_inputs", "experiment", "authority", "content_sha256",
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
        or value.get("v3_terminal_audit") != v3_terminal_audit_binding()
        or value.get("runtime_inputs") != runtime_authorization_template()
        or value.get("experiment") != science_contract()
        or value.get("authority") != EXECUTION_AUTHORITY
    ):
        raise PermissionError("rigid transport execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_v3.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity", "current_source_bindings", "evaluate_gate",
    "evaluate_update_401_phase_switch", "model_config", "objective_contract",
    "optimizer_contract", "preregistration_binding",
    "runtime_authorization_template", "science_contract",
    "v3_terminal_audit_binding", "validate_authorization",
    "validate_governing_documents", "validate_review",
    "validate_source_manifest",
})
