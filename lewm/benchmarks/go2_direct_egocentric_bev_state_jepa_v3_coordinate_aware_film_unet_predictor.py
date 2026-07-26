"""Source-only contract for the Direct BEV V3 predictor falsification."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V2_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v2_integrity.py"
)
_V2_SPEC = importlib.util.spec_from_file_location(
    "_lewm_direct_bev_v3_frozen_v2_contract",
    ROOT / FROZEN_V2_CONTRACT_RELATIVE_PATH,
)
if _V2_SPEC is None or _V2_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V2 source-only contract")
_v2 = importlib.util.module_from_spec(_V2_SPEC)
sys.modules[_V2_SPEC.name] = _v2
_V2_SPEC.loader.exec_module(_v2)

for _name in _v2.__all__:
    globals()[_name] = getattr(_v2, _name)
with_content_sha256 = _v2.with_content_sha256


IMPLEMENTATION_AUTHOR = "/root/plan_efficiency"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor"
)

FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v2_integrity_"
    "source_manifest_2026-07-26.json"
)
FROZEN_V2_SOURCE_MANIFEST_COMMIT = (
    "acf43bcf3e31729df7498d33539132fda205e27c"
)
FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256 = (
    "a52b99f13cdbb3e8841e9c87e451d4ab5aa09db3c943acb8b14e67a49ec2e510"
)
FROZEN_V2_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "158fb914ec24c5f893cfee92a02fc951a28c2c0bad9b31b1a3ad8bd27445a3f8"
)
FROZEN_V2_SOURCE_MANIFEST_BYTE_COUNT = 22_507
FROZEN_V2_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V2_SOURCE_COUNT = 73

V2_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v2_integrity_"
    "terminal_audit_2026-07-26.json"
)
V2_TERMINAL_AUDIT_COMMIT = "625a79dbfc85fbf32d0925b4668574828d433ca9"
V2_TERMINAL_AUDIT_FILE_SHA256 = (
    "93132058a0f94f652864e73e00cfb050c35f901e73d06277e13e3897825ef5a0"
)
V2_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "4ed33359075c9cda7ddac16854ccbfd902e0dfe900a38aa83cd94d5fb74f1340"
)
V2_TERMINAL_AUDIT_BYTE_COUNT = 11_802
V2_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_SCIENTIFIC_UPDATE_100_DIRECTIONAL_GATE_FAILURE_CLOSES_V2_"
    "NO_RETRY"
)
V2_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_SCIENTIFIC_GATE_FAILURE_STRONG_EARLY_DIRECT_BEV_PERCEPTION_"
    "LEARNING_ACTION_TRANSITION_REMAINS_AT_CHANCE_V2_PERMANENTLY_CLOSED"
)
V2_TERMINAL_ACCOUNTING = {
    "updates": 100,
    "presentations": 1_600,
    "objective_evaluations": 400,
    "backward_calls": 400,
    "optimizer_updates": 100,
    "ema_updates": 100,
    "registered_observations": 2,
}

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor_preregistration_2026-07-26.md"
)
PREREGISTRATION_COMMIT = "6b1566ba3786485b89e0a702a5e25418fcb0ed18"
PREREGISTRATION_FILE_SHA256 = (
    "be75f268816f422f1a40b7ee56dbf4bf544cd6893f9d3b296540ff4a98176c02"
)
PREREGISTRATION_BYTE_COUNT = 7_951

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor_contract.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor_runner.py"
)
LAUNCHER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_launch_go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor_source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor_source_closure.py"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor_source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor_source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor_execution_authorization_"
    "2026-07-26.json"
)

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    LAUNCHER_TEST_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(set(_v2.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH,
    V2_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v3_"
    "coordinate_aware_film_unet_predictor"
)

RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_metrics_v1"
ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_artifact_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
AUTHORIZATION_STATUS = "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V3_FILM_UNET_PROBE"

PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    **_v2.EXECUTION_AUTHORITY,
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v2_retry_authorized": False,
    "v2_checkpoint_or_runtime_output_reuse_authorized": False,
    "coordinate_aware_film_unet_predictor_only": True,
    "science_identical_rng_integrity_replacement_only": False,
}

PREDICTOR_ORDERED_PARAMETER_NAME_SHA256 = (
    "ebbd0bb384b09862c867338b39b4ffcfa4072e43730451f0eee337be3167fad2"
)
PREDICTOR_ORDERED_PARAMETER_INVENTORY_SHA256 = (
    "5c8cac4bb77b3669894b04a7def61fe8f35ee2f7cb84bb2e38c0efdb8ab35665"
)
PREDICTOR_FULLY_QUALIFIED_ORDERED_PARAMETER_NAME_SHA256 = (
    "0398031cb776c10a23b14c7935d2566f4a3087175213e87b49c2a05cadf6e1dd"
)
PREDICTOR_PARAMETER_COUNT = 317_107
PREDICTOR_TENSOR_COUNT = 79
TOTAL_PARAMETER_COUNT = 6_552_249
TOTAL_TENSOR_COUNT = 277
MODEL_PARAMETER_INVENTORY = {
    group: dict(binding)
    for group, binding in _v2.MODEL_PARAMETER_INVENTORY.items()
}
MODEL_PARAMETER_INVENTORY["predictor"] = {
    "parameter_count": PREDICTOR_PARAMETER_COUNT,
    "tensor_count": PREDICTOR_TENSOR_COUNT,
    "ordered_parameter_name_sha256": (
        PREDICTOR_FULLY_QUALIFIED_ORDERED_PARAMETER_NAME_SHA256
    ),
}
MODEL_PARAMETER_INVENTORY["total"] = {
    "parameter_count": TOTAL_PARAMETER_COUNT,
    "tensor_count": TOTAL_TENSOR_COUNT,
}

PREDICTOR_CONFIG = {
    "shape": "CoordinateAwareFilmUnetResidualPredictorV3",
    "inputs_in_order": [
        "current_three_logit_state",
        "normalized_row_index",
        "normalized_column_index",
    ],
    "coordinate_construction": {
        "row": "linspace(-1,1,H) expanded along columns",
        "column": "linspace(-1,1,W) expanded along rows",
        "dtype_and_device": "current_state_logits",
        "persistent_buffer": False,
        "metric_pose_or_geometry": False,
    },
    "action_vocabulary_in_order": [
        "arc_left", "arc_right", "backward", "forward_fast",
        "forward_medium", "forward_slow", "hold", "yaw_left", "yaw_right",
    ],
    "hold_action_index": 6,
    "action_embedding": {"count": 9, "dim": 64},
    "convolution": {"kernel": 3, "padding": 1, "bias": True},
    "normalization": {"type": "GroupNorm", "groups": 4, "affine": True},
    "activation": "GELU",
    "block": "Conv-GN4-GELU-Conv-GN4-GELU",
    "stages_in_forward_order": [
        "enc64:block:5-16-16:save16",
        "down32:stride2:16-32:GN4:GELU",
        "enc32:block:32-32-32:save32",
        "down16:stride2:32-48:GN4:GELU",
        "enc16:block:48-48-48:save48",
        "down8:stride2:48-64:GN4:GELU",
        "bottleneck8:block:64-64-64:FiLM64",
        "dec16:nearest2x+skip:block:112-48-48:FiLM48",
        "dec32:nearest2x+skip:block:80-32-32:FiLM32",
        "dec64:nearest2x+skip:block:48-16-16:FiLM16",
        "head:Conv3x3:16-3:zero_weight_and_bias",
    ],
    "film": {
        "embedding_to_channels_in_order": [[64, 128], [64, 96], [64, 64], [64, 32]],
        "channel_order": [64, 48, 32, 16],
        "formula": "x*(1+gamma)+beta",
        "placement": "after_each_listed_block",
    },
    "all_actions": "encode_once_decode_nine_film_conditions",
    "prediction": "current_state_logits+zero_head_residual",
    "forbidden": [
        "pose", "odometry", "depth", "flow_target", "metric_motion",
        "geometry_label", "hand_coded_transform", "analytical_warp",
        "grid_sample", "attention", "auxiliary_loss", "bypass",
    ],
    "construction_draw_order": [
        "unchanged_v2_perception", "action_embedding", "enc64", "down32",
        "enc32", "down16", "enc16", "down8", "bottleneck", "film64",
        "dec16", "film48", "dec32", "film32", "dec64", "film16",
        "residual_head",
    ],
    "rng": {
        "seed": BASE_INITIALIZATION_SEED,
        "seed_target": "torch.random.default_generator_only",
        "caller_cpu_rng_restored": True,
        "accelerator_seed_calls": 0,
    },
}

GATE_THRESHOLDS = {key: dict(value) for key, value in _v2.GATE_THRESHOLDS.items()}
GATE_THRESHOLDS[100] = {
    **GATE_THRESHOLDS[100],
    "v3_action_macro_balanced_accuracy_minimum": 0.13,
    "v3_action_nll_maximum": 2.187,
    "v3_hardest_wrong_positive_scene_count_minimum": 2,
    "v3_aggregate_raster_balanced_accuracy_minimum": 0.65,
    "v3_J_maximum": 0.60,
}
GATE_CONTROLS = {
    0: (
        "FAIL_UPDATE_ZERO_INTEGRITY_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_ZERO_GATE",
    ),
    100: (
        "FAIL_UPDATE_100_V3_PREDICTOR_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_100_V3_PREDICTOR_GATE",
    ),
    400: (
        "FAIL_UPDATE_400_MECHANISM_GATE_TERMINAL_NO_RETRY",
        "CONTINUE_AFTER_UPDATE_400_GATE",
    ),
    1000: (
        "FAIL_UPDATE_1000_PERCEPTION_GATE_TERMINAL_NO_RETRY",
        "PASS_DIRECT_BEV_V3_FILM_UNET_PERCEPTION_GATE_REQUALIFICATION_ONLY",
    ),
}
CONTROL_UPDATE_ZERO_FAIL = GATE_CONTROLS[0][0]
CONTROL_UPDATE_100_FAIL = GATE_CONTROLS[100][0]
CONTROL_CONTINUE_UPDATE_100 = GATE_CONTROLS[100][1]
CONTROL_UPDATE_400_FAIL = GATE_CONTROLS[400][0]
CONTROL_CONTINUE_UPDATE_400 = GATE_CONTROLS[400][1]
CONTROL_UPDATE_1000_FAIL = GATE_CONTROLS[1000][0]
CONTROL_PASS = GATE_CONTROLS[1000][1]
FAILURE_CONTROLS = tuple(pair[0] for pair in GATE_CONTROLS.values())

SCIENTIFIC_REVIEW_CHECKS = {
    "v2_source_manifest_exact_and_all_73_sources_rehashed": True,
    "v2_terminal_audit_exact_and_v2_permanently_closed": True,
    "v3_preregistration_exact": True,
    "shared_perception_data_objective_optimizer_schedule_ema_exact": True,
    "predictor_is_the_only_scientific_model_delta": True,
    "v2_checkpoint_tensor_trace_or_runtime_output_reuse": False,
    "u0_u400_u1000_numerically_unchanged": True,
    "u100_strengthened_conjuncts_exact": True,
    "caps_one_attempt_and_downstream_denials_exact": True,
    "no_generated_runtime_or_protected_material_opened": True,
}


def frozen_v2_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V2_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V2_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V2_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V2_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V2_SOURCE_COUNT,
    }


def v2_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V2_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V2_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V2_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V2_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V2_TERMINAL_AUDIT_BYTE_COUNT,
    }


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def validate_frozen_v2_source_closure(root: Path = ROOT) -> dict[str, str]:
    read = _v2._v1._read_regular_source
    raw = read(root / FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH)
    if (
        len(raw) != FROZEN_V2_SOURCE_MANIFEST_BYTE_COUNT
        or hashlib.sha256(raw).hexdigest() != FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V2 source manifest raw identity changed")
    manifest = _v2.validate_source_manifest(raw)
    if (
        manifest.get("content_sha256") != FROZEN_V2_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("status") != FROZEN_V2_SOURCE_MANIFEST_STATUS
        or manifest.get("source_count") != FROZEN_V2_SOURCE_COUNT
    ):
        raise PermissionError("frozen V2 source manifest conclusion changed")
    current = _v2.current_source_bindings(root)
    if current.get(FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH) != (
        FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("current V2 source manifest changed")
    for binding in manifest["source_bindings"]:
        if current.get(binding["path"]) != binding["file_sha256"]:
            raise PermissionError(f"current V2 source changed: {binding['path']}")
    return current


def model_config() -> dict[str, Any]:
    value = _v2.model_config()
    value["transition"] = dict(PREDICTOR_CONFIG)
    inventory = value["parameter_inventory"]
    inventory["predictor"] = {
        **MODEL_PARAMETER_INVENTORY["predictor"],
        "predictor_local_ordered_parameter_name_sha256": (
            PREDICTOR_ORDERED_PARAMETER_NAME_SHA256
        ),
        "ordered_parameter_inventory_sha256": (
            PREDICTOR_ORDERED_PARAMETER_INVENTORY_SHA256
        ),
    }
    inventory["total"] = dict(MODEL_PARAMETER_INVENTORY["total"])
    return value


def objective_contract() -> dict[str, Any]:
    return _v2.objective_contract()


def optimizer_contract() -> dict[str, Any]:
    return _v2.optimizer_contract()


def build_schedule_identity() -> dict[str, Any]:
    return _v2.build_schedule_identity()


def runtime_authorization_template() -> dict[str, Any]:
    return _v2.runtime_authorization_template()


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    *,
    update_zero: Mapping[str, Any] | None = None,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    result = _v2.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        prior_gates_passed=prior_gates_passed,
    )
    if update == 100:
        conjuncts = dict(result["conjuncts"])
        conjuncts.update({
            "v3_action_macro_balanced_accuracy_at_least_point13": (
                _v2._v1._finite_number(
                    metrics.get("action_macro_balanced_accuracy"),
                    name="action_macro_balanced_accuracy",
                ) >= 0.13
            ),
            "v3_action_nll_at_most_2point187": (
                _v2._v1._finite_number(metrics.get("action_nll"), name="action_nll")
                <= 2.187
            ),
            "v3_hardest_wrong_positive_scenes_at_least_two": (
                _v2._v1._finite_number(
                    metrics.get("hardest_wrong_positive_scene_count"),
                    name="hardest_wrong_positive_scene_count",
                ) >= 2
            ),
            "v3_aggregate_raster_balanced_accuracy_at_least_point65": (
                _v2._v1._finite_number(
                    metrics.get("aggregate_raster_balanced_accuracy"),
                    name="aggregate_raster_balanced_accuracy",
                ) >= 0.65
            ),
            "v3_J_at_most_point60": (
                _v2._v1._finite_number(metrics.get("J"), name="J") <= 0.60
            ),
        })
        result["conjuncts"] = conjuncts
        result["passed"] = all(conjuncts.values())
        result["thresholds"] = dict(GATE_THRESHOLDS[100])
    result["control"] = GATE_CONTROLS[update][1 if result["passed"] else 0]
    return result


def science_contract() -> dict[str, Any]:
    value = _v2.science_contract()
    v2_integrity_provenance = value.pop("integrity_replacement")
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["model"] = model_config()
    value["gates"]["thresholds"]["100"] = dict(GATE_THRESHOLDS[100])
    value["governing_documents"] = {
        **value["governing_documents"],
        "frozen_v2_source_manifest": frozen_v2_source_manifest_binding(),
        "v2_terminal_audit": v2_terminal_audit_binding(),
        "v3_preregistration": preregistration_binding(),
    }
    lifecycle = dict(value["lifecycle"])
    frozen_v2_integrity_replacement_of = lifecycle.pop(
        "integrity_replacement_of"
    )
    frozen_v2_retry_control = lifecycle.pop(
        "retry_resume_repair_recovery_replacement_second_seed_or_v2"
    )
    value["lifecycle"] = {
        **lifecycle,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "predictor_successor_of": _v2.EXPERIMENT_ID,
        "frozen_v2_integrity_replacement_of": (
            frozen_v2_integrity_replacement_of
        ),
        "frozen_v2_retry_resume_repair_recovery_replacement_second_seed": (
            frozen_v2_retry_control
        ),
        "v2_retry": False,
        "v2_checkpoint_reuse": False,
        "retry_resume_repair_recovery_replacement_second_seed_or_v3": False,
    }
    value["frozen_v2_integrity_provenance"] = {
        "scope": "historical_v1_to_v2_integrity_replacement_only",
        "not_a_v3_unchanged_architecture_claim": True,
        "v1_to_v2": v2_integrity_provenance,
    }
    value["predictor_successor"] = {
        "sole_scientific_delta": "coordinate_aware_film_unet_predictor",
        "all_actions_path_changed_only_to_share_state_encoding": True,
        "predictor_config": dict(PREDICTOR_CONFIG),
        "u100_thresholds": {
            key: value for key, value in GATE_THRESHOLDS[100].items()
            if key.startswith("v3_")
        },
    }
    value["authority"] = {
        **value["authority"],
        "v3_execution_authorized_by_source_contract": False,
        "v2_checkpoint_or_runtime_output_reuse_authorized": False,
    }
    value["scientific_checks"] = dict(SCIENTIFIC_REVIEW_CHECKS)
    return value


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    result = _v2.validate_governing_documents(root)
    validate_frozen_v2_source_closure(root)
    read = _v2._v1._read_regular_source
    audit = read(root / V2_TERMINAL_AUDIT_RELATIVE_PATH)
    preregistration = read(root / PREREGISTRATION_RELATIVE_PATH)
    if (
        len(audit) != V2_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit).hexdigest() != V2_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("V2 terminal audit raw identity changed")
    audit_value = _v2.parse_canonical_json(audit, name="V2 terminal audit")
    accounting = audit_value.get("execution_accounting", {})
    consequence = audit_value.get("scientific_consequence", {})
    if (
        audit_value.get("content_sha256") != V2_TERMINAL_AUDIT_CONTENT_SHA256
        or audit_value.get("status") != V2_TERMINAL_AUDIT_STATUS
        or audit_value.get("classification") != V2_TERMINAL_AUDIT_CLASSIFICATION
        or any(accounting.get(key) != expected for key, expected in V2_TERMINAL_ACCOUNTING.items())
        or consequence.get("valid_scientific_result_produced") is not True
        or consequence.get("direct_bev_perception_early_learning_supported") is not True
        or consequence.get("action_conditioned_transition_learning_supported") is not False
        or consequence.get("v2_permanently_closed") is not True
        or consequence.get("retry_resume_repair_or_checkpoint_reuse_authorized") is not False
    ):
        raise PermissionError("V2 terminal audit conclusion changed")
    if (
        len(preregistration) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(preregistration).hexdigest() != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("V3 preregistration changed")
    result[FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH] = FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256
    result[V2_TERMINAL_AUDIT_RELATIVE_PATH] = V2_TERMINAL_AUDIT_FILE_SHA256
    result[PREREGISTRATION_RELATIVE_PATH] = PREREGISTRATION_FILE_SHA256
    return result


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = _v2.parse_canonical_json(raw, name="V3 source manifest")
    expected_fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
    }
    paths = value.get("source_paths")
    bindings = value.get("source_bindings")
    if (
        set(value) != expected_fields
        or value.get("schema") != SOURCE_MANIFEST_SCHEMA
        or value.get("status") != "PASS_SOURCE_CLOSURE"
        or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value.get("forced_dynamic_sources") != list(SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
        or value.get("excluded_runtime_categories") != list(PROHIBITED_RUNTIME_CATEGORIES)
        or type(paths) is not list or paths != sorted(paths) or len(paths) != len(set(paths))
        or not set(SOURCE_PATHS).issubset(paths)
        or type(bindings) is not list or len(bindings) != len(paths)
        or value.get("source_count") != len(paths)
        or value.get("source_bindings_sha256") != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
    ):
        raise PermissionError("V3 source manifest contract changed")
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {"path", "file_sha256", "byte_count"}:
            raise PermissionError("V3 source binding fields changed")
        relative = _v2._v1.safe_relative_source_path(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V3 source binding identity changed")
        normalized.append(relative)
    if normalized != paths:
        raise PermissionError("V3 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = _v2._v1._read_regular_source
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(f"manifest-bound V3 source changed: {binding['path']}")
        result[binding["path"]] = digest
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(manifest_raw).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def _manifest_binding_or_read(
    source_manifest_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source_manifest_binding is None:
        raw = _v2._v1._read_regular_source(ROOT / SOURCE_MANIFEST_RELATIVE_PATH)
        value = validate_source_manifest(raw)
        source_manifest_binding = artifact_binding(
            SOURCE_MANIFEST_RELATIVE_PATH,
            raw,
            content_sha256=str(value["content_sha256"]),
        )
    return validate_binding(dict(source_manifest_binding), path=SOURCE_MANIFEST_RELATIVE_PATH)


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
    source_manifest_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "implementation_author", "reviewer",
        "reviewed_sources", "source_manifest", "frozen_v2_source_manifest",
        "v2_terminal_audit", "v3_preregistration", "science_contract",
        "source_only_checks", "scientific_checks", "findings", "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V3 source review fields changed")
    manifest_binding = _manifest_binding_or_read(source_manifest_binding)
    core = dict(value)
    declared = core.pop("content_sha256")
    reviewer = value["reviewer"]
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != "PASS_SOURCE_AND_PREDICTOR_ONLY_SCIENCE"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or value["reviewed_sources"] != dict(expected_sources)
        or value["source_manifest"] != manifest_binding
        or expected_sources.get(SOURCE_MANIFEST_RELATIVE_PATH) != manifest_binding["file_sha256"]
        or expected_sources.get(FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH) != FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256
        or expected_sources.get(V2_TERMINAL_AUDIT_RELATIVE_PATH) != V2_TERMINAL_AUDIT_FILE_SHA256
        or expected_sources.get(PREREGISTRATION_RELATIVE_PATH) != PREREGISTRATION_FILE_SHA256
        or value["frozen_v2_source_manifest"] != frozen_v2_source_manifest_binding()
        or value["v2_terminal_audit"] != v2_terminal_audit_binding()
        or value["v3_preregistration"] != preregistration_binding()
        or value["science_contract"] != science_contract()
        or value["source_only_checks"] != {
            "stdlib_only_contract_import": True,
            "cpu_synthetic_torch_tests_permitted": True,
            "generated_inputs_opened": [],
            "checkpoints_tensors_traces_or_runtime_outputs_opened": [],
            "gpu_state_opened": [],
            "sealed_or_heldout_opened": [],
        }
        or value["scientific_checks"] != SCIENTIFIC_REVIEW_CHECKS
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V3 source review did not pass exact scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "frozen_v2_source_manifest", "v2_terminal_audit", "v3_preregistration",
        "runtime_inputs", "experiment", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V3 execution authorization fields changed")
    expected_review = validate_binding(dict(review_binding), path=REVIEW_RELATIVE_PATH)
    core = dict(value)
    declared = core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != AUTHORIZATION_STATUS
        or type(authorizer) is not str or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_source_review"] != expected_review
        or value["frozen_v2_source_manifest"] != frozen_v2_source_manifest_binding()
        or value["v2_terminal_audit"] != v2_terminal_audit_binding()
        or value["v3_preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V3 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_v2.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity",
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v2_source_manifest_binding",
    "model_config",
    "objective_contract",
    "optimizer_contract",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "v2_terminal_audit_binding",
    "validate_authorization",
    "validate_frozen_v2_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
    "with_content_sha256",
})
