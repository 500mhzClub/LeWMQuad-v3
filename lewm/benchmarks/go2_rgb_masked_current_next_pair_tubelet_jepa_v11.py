"""Source-only contract for the masked pair-tubelet RGB JEPA V11 probe.

The frozen V10 contract supplies the unchanged data, mapping, custody, and
canonical-JSON validators.  It is executed into this module (rather than
imported and mutated) so every inherited helper resolves the V11 globals
below while the original V10 module remains untouched.  This import reads
tracked Python source only and imports no tensor library.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


_V10_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py"
)
_V10_CONTRACT_PATH = Path(__file__).resolve().parents[2] / _V10_CONTRACT_RELATIVE_PATH
_V10_SOURCE = _V10_CONTRACT_PATH.read_bytes()
exec(compile(_V10_SOURCE, str(_V10_CONTRACT_PATH), "exec"), globals())
del _V10_SOURCE
_FROZEN_V10_SOURCE_PATHS = tuple(SOURCE_PATHS)


# V11 identity and source closure.
IMPLEMENTATION_AUTHOR = "/root/plan_full_stack"
SCHEMA_PREFIX = "lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v11"
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_contract.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_runner.py"
)
V11_MODEL_RELATIVE_PATH = (
    "lewm/models/rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
V11_MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_"
    "source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_"
    "source_closure.py"
)
OBJECTIVE_MODEL_RELATIVE_PATH = V11_MODEL_RELATIVE_PATH
OBJECTIVE_TEST_RELATIVE_PATH = V11_MODEL_TEST_RELATIVE_PATH
FROZEN_V10_CONTRACT_RELATIVE_PATH = _V10_CONTRACT_RELATIVE_PATH
FROZEN_V10_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py"
)
TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_"
    "preregistration_2026-07-26.json"
)
PREREGISTRATION_COMMIT = "46de4c1b6a89dad43550b62a6e9327dec0a7b9da"
PREREGISTRATION_FILE_SHA256 = (
    "bbc4fa556788ce8df90c417aaa074bb7daf2aea47e4465434a4f119e18530dee"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "b7d4c6a59e2bad1e9dfa399a01a83c29658b88a0f11236a49265c8f7eccd15d9"
)
PREREGISTRATION_BYTE_COUNT = 27_808
PRIOR_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_v10r_"
    "integrity_replacement_terminal_audit_2026-07-26.json"
)
PRIOR_TERMINAL_AUDIT_COMMIT = "79d6de74b795065f7a5a47b32f1a56fc4fd4580a"
PRIOR_TERMINAL_AUDIT_FILE_SHA256 = (
    "8cd27a7d21e9ce1875d322cad2ea5aae8a846a301247f774d4da86074ebd28a5"
)
PRIOR_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "ab6b9d9ad3b6de1462fe142c42e18bf30751cc483aceaa9fabb632b2999cca73"
)
PRIOR_TERMINAL_AUDIT_BYTE_COUNT = 12_862

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_"
    "source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_"
    "source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_"
    "execution_authorization_2026-07-26.json"
)
SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"

ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    V11_MODEL_RELATIVE_PATH,
    V11_MODEL_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted((
    FROZEN_V10_CONTRACT_RELATIVE_PATH,
    FROZEN_V10_RUNNER_RELATIVE_PATH,
    *_FROZEN_V10_SOURCE_PATHS,
)))
SOURCE_PATHS = tuple(sorted((*ADDITIVE_SOURCE_PATHS, *REUSED_SOURCE_PATHS)))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    PRIOR_TERMINAL_AUDIT_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_masked_current_next_pair_tubelet_jepa_probe_v11"
)


# Exact V11 architecture, optimizer ownership, and EMA inventory.
TARGET_EMA_MOMENTUM = 0.996
V11_NEW_PARAMETER_DRAW_ORDER = (
    "online_future_mask_token_trunc_normal_std_0_02",
    "online_future_temporal_embedding_trunc_normal_std_0_02",
    "online_action_embedding.weight_exact_zeros",
    "online_future_projector.weight_xavier_uniform",
    "online_future_projector.bias_exact_zeros",
)
PHASE_A_ENCODER_PARAMETER_PREFIXES = ("encoder.",)
PHASE_A_AUXILIARY_PARAMETER_PREFIXES = (
    "online_future_mask_token",
    "online_future_temporal_embedding",
    "online_action_embedding.",
    "online_future_projector.",
)
PHASE_A_TRAINABLE_PARAMETER_PREFIXES = (
    *PHASE_A_ENCODER_PARAMETER_PREFIXES,
    *PHASE_A_AUXILIARY_PARAMETER_PREFIXES,
)
PHASE_A_EXACT_FROZEN_PARAMETER_NAMES = ("encoder.cls_token",)
PHASE_A_FROZEN_PARAMETER_PREFIXES = (
    "target_encoder.",
    "target_future_temporal_embedding",
    "target_future_projector.",
)


def _target_ema_parameter_pairs() -> tuple[tuple[str, str], ...]:
    encoder_names = [
        "cls_token",
        "pos_embed",
        "patch_embed.weight",
        "patch_embed.bias",
    ]
    for block in range(6):
        prefix = f"blocks.{block}."
        encoder_names.extend(
            prefix + suffix
            for suffix in (
                "norm1.weight",
                "norm1.bias",
                "attn.in_proj_weight",
                "attn.in_proj_bias",
                "attn.out_proj.weight",
                "attn.out_proj.bias",
                "norm2.weight",
                "norm2.bias",
                "mlp.0.weight",
                "mlp.0.bias",
                "mlp.3.weight",
                "mlp.3.bias",
            )
        )
    encoder_names.extend(("norm.weight", "norm.bias"))
    pairs = [
        (f"encoder.{name}", f"target_encoder.{name}")
        for name in encoder_names
    ]
    pairs.extend((
        (
            "online_future_temporal_embedding",
            "target_future_temporal_embedding",
        ),
        (
            "online_future_projector.weight",
            "target_future_projector.weight",
        ),
        (
            "online_future_projector.bias",
            "target_future_projector.bias",
        ),
    ))
    return tuple(pairs)


TARGET_EMA_PARAMETER_PAIRS = _target_ema_parameter_pairs()


def v11_model_config() -> dict[str, Any]:
    return {
        "image_size": 112,
        "patch_size": 7,
        "feature_dim": 192,
        "encoder_depth": 6,
        "encoder_heads": 6,
        "encoder_mlp_ratio": 4,
        "encoder_dropout": 0.0,
        "future_token_count": 256,
        "action_count": 9,
        "target_ema_momentum": TARGET_EMA_MOMENTUM,
        "normalization_epsilon": 1e-8,
        "whitening_epsilon": 1e-4,
        "whitening_variance_weight": 0.50,
        "whitening_covariance_weight": 0.02,
    }


phase_a_model_config = v11_model_config


# One capped phase only: 1,000 updates and 16,000 pair presentations.
PHASE_A_MAXIMUM_UPDATE = 1_000
PHASE_B_MAXIMUM_UPDATE = 0
MAXIMUM_UPDATE = PHASE_A_MAXIMUM_UPDATE
CUMULATIVE_MAXIMUM_UPDATE = PHASE_A_MAXIMUM_UPDATE
PHASE_A_MAXIMUM_PRESENTATIONS = 16_000
PHASE_B_MAXIMUM_PRESENTATIONS = 0
MAXIMUM_PRESENTATIONS = PHASE_A_MAXIMUM_PRESENTATIONS
CUMULATIVE_MAXIMUM_PRESENTATIONS = PHASE_A_MAXIMUM_PRESENTATIONS
PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES = 60
PHASE_B_GPU_ACTIVE_TIME_CAP_MINUTES = 0
CUMULATIVE_GPU_ACTIVE_TIME_CAP_MINUTES = 60

PHASE_A_PASS_THRESHOLDS = {
    "normalized_projected_future_effective_rank_minimum": 48.0,
    "action_equal_logit_reference_factor_maximum": 0.90,
    "action_macro_balanced_accuracy_minimum": 1.0 / 3.0,
    "energy_ratio_maximum": 0.95,
    "mean_target_ratio_maximum": 0.90,
    "positive_family_margin_count_minimum": 6,
}
PHASE_A_UPDATE_100_THRESHOLDS = {
    "normalized_projected_future_effective_rank_strictly_greater_than":
        17.426651000976562,
    "action_equal_logit_reference_factor_maximum": 0.99,
    "action_macro_balanced_accuracy_minimum": 0.15,
    "same_action_two_target_reference_factor_maximum": 0.95,
    "same_action_strict_win_rate_minimum": 0.60,
    "non_hold_correct_current_ratio_strictly_less_than": 1.0,
    "positive_family_margin_count_minimum": 6,
}
PHASE_A_UPDATE_400_THRESHOLDS = {
    "normalized_projected_future_effective_rank_minimum": 32.71332550048828,
    "action_equal_logit_reference_factor_maximum": 0.95,
    "action_macro_balanced_accuracy_strictly_greater_than": 2.0 / 9.0,
    "energy_ratio_strictly_less_than": 0.99,
    "mean_target_ratio_strictly_less_than": 1.0,
    "positive_family_margin_count_minimum": 6,
}

PHASE_A_METRIC_FIELDS = frozenset({
    "all_values_finite",
    "ema_target_gradient_free",
    "pair_count",
    "scene_family_count",
    "non_hold_pair_count",
    "masked_future_jepa_loss",
    "normalized_projected_future_effective_rank",
    "normalized_projected_future_cross_sample_variance",
    "normalized_projected_future_off_diagonal_covariance",
    "normalized_projected_future_spatial_diversity",
    "detached_target_future_effective_rank",
    "detached_target_future_cross_sample_variance",
    "detached_target_future_off_diagonal_covariance",
    "detached_target_future_spatial_diversity",
    "true_pair_mse",
    "shuffled_next_mse",
    "shuffled_current_mse",
    "mean_target_mse",
    "factorized_retrieval",
})
PHASE_A_UPDATE0_FIELDS = frozenset({
    *PHASE_A_METRIC_FIELDS,
    "all_action_predictions_bitwise_equal",
    "all_action_unordered_pair_count",
    "all_action_prediction_row_count",
})
PHASE_A_UPDATE0_HEALTH_FIELDS = frozenset({
    "normalized_projected_future_cross_sample_variance",
    "normalized_projected_future_spatial_diversity",
})
PHASE_A_OBSERVATION_INTEGRITY_FIELDS = frozenset({
    "rng_state_preserved",
    "state_mutation_count",
    "future_leakage_prohibition_passed",
    "target_path_nonvacuity_passed",
    "online_target_autograd_separation_passed",
    "ema_inventory_exact",
    "ema_update_count",
    "expected_ema_update_count",
    "normalized_population_exact",
    "all_nine_candidates_exact",
    "observation_row_count",
})

CONTROL_CONTINUE = "CONTINUE_INFORMATIONAL"
CONTROL_PHASE_A_PASS = (
    "PASS_MASKED_PAIR_TUBELET_PROXY_SEPARATE_REQUALIFICATION_ONLY"
)
CONTROL_PHASE_A_FAIL = "FAIL_PHASE_A_TERMINAL_NO_RETRY"
CONTROL_PHASE_A_UPDATE_ZERO_FAIL = (
    "FAIL_PHASE_A_UPDATE_ZERO_INVARIANT_GATE_TERMINAL"
)
CONTROL_PHASE_A_UPDATE_100_FAIL = (
    "FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL"
)
CONTROL_PHASE_A_UPDATE_400_FAIL = (
    "FAIL_PHASE_A_UPDATE_400_CONTINUATION_GATE_TERMINAL"
)
CONTROL_PASS = CONTROL_PHASE_A_PASS
CONTROL_FAIL = CONTROL_PHASE_A_FAIL
PHASE_A_TERMINAL_CONTROLS = (
    CONTROL_PHASE_A_UPDATE_ZERO_FAIL,
    CONTROL_PHASE_A_UPDATE_100_FAIL,
    CONTROL_PHASE_A_UPDATE_400_FAIL,
    CONTROL_PHASE_A_FAIL,
    CONTROL_PHASE_A_PASS,
)
PHASE_A_FAILURE_CONTROLS = PHASE_A_TERMINAL_CONTROLS[:-1]
RESULT_TERMINAL_STATUSES = (
    CONTROL_PHASE_A_PASS,
    *PHASE_A_FAILURE_CONTROLS,
)
COMPLETION_TERMINAL_STATUSES = (
    *PHASE_A_FAILURE_CONTROLS,
    "TERMINAL_PASS",
    "TERMINAL_FAIL",
)
UPDATE_ZERO_ACTION_NLL_ABSOLUTE_TOLERANCE = 8.0 * (2.0 ** -23)

RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
PHASE_A_METRICS_SCHEMA = f"{SCHEMA_PREFIX}_phase_a_metrics_v1"
PHASE_A_ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_phase_a_artifact_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

ACCESS_ZERO_COUNTER_FIELDS = (
    "probability_calibration_open_count",
    "prior_runtime_output_open_count",
    "rejected_checkpoint_open_count",
    "phase_a_camera_supervision_array_open_count",
    "phase_a_general_raw_loader_call_count",
    "g2_open_count",
    "navigation_open_count",
    "heldout_open_count",
    "sealed_open_count",
    "production_input_open_count",
    "deployment_input_open_count",
    "observer_rerun_count",
)

DOWNSTREAM_DENIALS = {
    "checkpoint_qualified": False,
    "perception_qualification_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "sealed_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "calibration_authorized": False,
    "retry_resume_replacement_authorized": False,
    "pass_authorizes": (
        "separate_perception_requalification_preregistration_and_"
        "mandatory_matched_no_jepa_development_arm_only"
    ),
}
SOURCE_ONLY_AUTHORITY = {
    "generated_input_access_authorized": False,
    "checkpoint_or_tensor_access_authorized": False,
    "gpu_training_or_evaluation_authorized": False,
    "sealed_or_heldout_access_authorized": False,
    "source_only_review_authorized": True,
}
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    "one_fresh_attempt_authorized": True,
    "maximum_updates": PHASE_A_MAXIMUM_UPDATE,
    "maximum_presentations": PHASE_A_MAXIMUM_PRESENTATIONS,
    "gpu_active_minutes_maximum": PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES,
    "phase_b_authorized": False,
    **DOWNSTREAM_DENIALS,
}
SCIENTIFIC_REVIEW_CHECKS = {
    "online_context_accepts_no_next_or_target_input": True,
    "target_accepts_no_action_and_is_detached": True,
    "all_six_inherited_blocks_mix_exact_512_token_tubelet": True,
    "all_nine_actions_reuse_one_current_patch_tensor": True,
    "fixed_current_target_candidates_exact": True,
    "normalized_energy_and_whitening_population_exact": True,
    "target_ema_inventory_exact": True,
    "one_phase_16000_presentation_cap": True,
    "no_phase_b_or_downstream_authority": True,
}


def runtime_authorization_template() -> dict[str, Any]:
    value = {
        "raw": {
            "root": RAW_ROOT_RELATIVE_PATH,
            "manifest": _runtime_leaf(RAW_MANIFEST_RELATIVE_PATH),
            "audit": _runtime_leaf(RAW_AUDIT_RELATIVE_PATH),
            "role_counts": {
                "train": dict(TRAIN_ROLE_COUNTS),
                "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            },
            "role_policy": {
                "metadata_only_roles": ["authority", "index"],
                "model_facing_roles": ["train", "checkpoint_selection"],
                "raw_manifest_audit_pairs_and_endpoints_may_be_opened_only_"
                "for_bound_authority_or_index_validation": True,
            },
            "phase_a_grant": {
                "allowed_inputs": [
                    "bound_pair_index",
                    "bound_endpoint_index",
                    "bound_current_rgb",
                    "bound_next_rgb",
                    "bound_fixed_same_scene_mapped_negative_next_rgb",
                    "requested_primitive",
                ],
                "target_mapping_bindings": {
                    role: dict(binding)
                    for role, binding in TARGET_MAPPING_BINDINGS.items()
                },
                "selection_action_permutation_binding":
                    dict(SELECTION_ACTION_PERMUTATION_BINDING),
                "selection_family_bindings": {
                    family: dict(binding)
                    for family, binding in SELECTION_FAMILY_BINDINGS.items()
                },
                "selection_family_bindings_sha256":
                    SELECTION_FAMILY_BINDINGS_SHA256,
                "camera_supervision_array_open_authorized": False,
                "general_raw_v13_frame_loader_authorized": False,
            },
            "phase_b_grant": None,
        },
        "camera": {
            "root": N320_ROOT_RELATIVE_PATH,
            "gate": _runtime_leaf(N320_GATE_RELATIVE_PATH),
            "checkpoint": _runtime_leaf(N320_CHECKPOINT_RELATIVE_PATH),
            "fit_seed": 20260710,
            "fit_size": 320,
            "updates": 40_000,
            "gate_must_pass_all_checks": 26,
        },
        "schedule": _runtime_leaf(SCHEDULE_RELATIVE_PATH),
    }
    return value


def build_schedule_identity(phase: str) -> dict[str, Any]:
    if phase == "phase_b":
        return {
            "phase": "phase_b",
            "authorized": False,
            "source": None,
            "seed": None,
            "updates": 0,
            "presentations": 0,
            "microbatch_size": 0,
            "microbatches_per_update": 0,
            "effective_batch_size": 0,
            "checkpoints": [],
            "prefix_sha256": {},
            "reuse_same_frozen_prefix_independently": False,
        }
    if phase != "phase_a":
        raise ValueError("V11 schedule phase must be phase_a or disabled phase_b")
    return {
        "phase": "phase_a",
        "source": _runtime_leaf(SCHEDULE_RELATIVE_PATH),
        "seed": SCHEDULE_SEED,
        "updates": PHASE_A_MAXIMUM_UPDATE,
        "presentations": PHASE_A_MAXIMUM_PRESENTATIONS,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "checkpoints": list(CHECKPOINT_UPDATES),
        "prefix_sha256": {
            str(update): digest
            for update, digest in PHASE_A_SCHEDULE_PREFIX_SHA256.items()
        },
        "reuse_same_frozen_prefix_independently": False,
    }


def science_contract() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_science_contract_v1",
        "scientific_question": (
            "can_a_masked_current_to_next_pair_tubelet_make_action_specific_"
            "future_structure_accessible_without_online_future_leakage"
        ),
        "repository_goal": (
            "fully_learned_perception_only_rgb_jepa_navigation_validated_"
            "once_on_externally_custodied_heldout_mazes"
        ),
        "model": v11_model_config(),
        "initialization": {
            "base_seed": BASE_INITIALIZATION_SEED,
            "n320_encoder_only_migration": True,
            "new_parameter_draw_order": list(V11_NEW_PARAMETER_DRAW_ORDER),
            "target_inventory_hard_synced_before_update_zero": True,
            "new_transformer_block_count": 0,
        },
        "data": {
            "roles": ["train", "checkpoint_selection"],
            "train": dict(TRAIN_ROLE_COUNTS),
            "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            "same_action_target_mapping": {
                role: dict(binding)
                for role, binding in TARGET_MAPPING_BINDINGS.items()
            },
            "selection_action_permutation":
                dict(SELECTION_ACTION_PERMUTATION_BINDING),
            "new_data_or_predecessor_lookup": False,
        },
        "schedule": build_schedule_identity("phase_a"),
        "objective": {
            "masked_future_jepa_weight": 1.0,
            "action_retrieval_weight": 1.0,
            "target_retrieval_weight": 1.0,
            "whitening_variance_weight": 0.50,
            "whitening_covariance_weight": 0.02,
            "temperature_scale_margin_or_class_weight": False,
        },
        "gates": {
            "updates": list(CHECKPOINT_UPDATES),
            "observation_updates": [0, *CHECKPOINT_UPDATES],
            "update_0": {
                "common_invariants_only": True,
                "failure_control": CONTROL_PHASE_A_UPDATE_ZERO_FAIL,
            },
            "update_100": dict(PHASE_A_UPDATE_100_THRESHOLDS),
            "update_400": dict(PHASE_A_UPDATE_400_THRESHOLDS),
            "terminal": dict(PHASE_A_PASS_THRESHOLDS),
        },
        "lifecycle": {
            "attempt_index": 1,
            "maximum_attempts": 1,
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "phase_b_authorized": False,
            "retry_resume_replacement_authorized": False,
        },
        "authority": dict(DOWNSTREAM_DENIALS),
    }


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    """Rehash the frozen source graph and pretty-JSON governing documents."""

    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        relative = binding["path"]
        raw = _read_regular_source(root / relative)
        digest = hashlib.sha256(raw).hexdigest()
        if (
            len(raw) != binding["byte_count"]
            or digest != binding["file_sha256"]
        ):
            raise PermissionError(f"manifest-bound source changed: {relative}")
        result[relative] = digest

    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    documents = (
        (
            PREREGISTRATION_RELATIVE_PATH,
            PREREGISTRATION_BYTE_COUNT,
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_CONTENT_SHA256,
            "V11 preregistration",
        ),
        (
            PRIOR_TERMINAL_AUDIT_RELATIVE_PATH,
            PRIOR_TERMINAL_AUDIT_BYTE_COUNT,
            PRIOR_TERMINAL_AUDIT_FILE_SHA256,
            PRIOR_TERMINAL_AUDIT_CONTENT_SHA256,
            "V10R terminal audit",
        ),
    )
    for relative, byte_count, file_sha256, content_sha256, name in documents:
        raw = _read_regular_source(root / relative)
        digest = hashlib.sha256(raw).hexdigest()
        try:
            value = json.loads(raw, object_pairs_hook=_reject_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise PermissionError(f"{name} changed") from error
        if (
            len(raw) != byte_count
            or digest != file_sha256
            or type(value) is not dict
            or value.get("content_sha256") != content_sha256
        ):
            raise PermissionError(f"{name} changed")
        result[relative] = digest
    return result


def _normalize_v11_phase_a_inputs(
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    if type(metrics) is not dict or set(metrics) != PHASE_A_METRIC_FIELDS:
        raise ValueError("V11 Phase-A metric fields changed")
    if (
        type(update0_metrics) is not dict
        or set(update0_metrics) != PHASE_A_UPDATE0_FIELDS
    ):
        raise ValueError("V11 update-zero metric fields changed")
    if (
        type(observation_integrity) is not dict
        or set(observation_integrity)
        != PHASE_A_OBSERVATION_INTEGRITY_FIELDS
    ):
        raise ValueError("V11 observation-integrity fields changed")
    boolean_integrity = (
        "rng_state_preserved",
        "future_leakage_prohibition_passed",
        "target_path_nonvacuity_passed",
        "online_target_autograd_separation_passed",
        "ema_inventory_exact",
        "normalized_population_exact",
        "all_nine_candidates_exact",
    )
    if any(
        type(observation_integrity[field]) is not bool
        for field in boolean_integrity
    ):
        raise TypeError("V11 observation-integrity Boolean changed")
    integer_integrity = (
        "state_mutation_count",
        "ema_update_count",
        "expected_ema_update_count",
        "observation_row_count",
    )
    if any(
        type(observation_integrity[field]) is not int
        or observation_integrity[field] < 0
        for field in integer_integrity
    ):
        raise TypeError("V11 observation-integrity count changed")
    if (
        observation_integrity["ema_update_count"]
        != observation_integrity["expected_ema_update_count"]
        or observation_integrity["observation_row_count"]
        != SELECTION_ROLE_COUNTS["pairs"]
    ):
        raise ValueError("V11 EMA or observation count changed")
    if (
        type(update0_metrics["all_action_predictions_bitwise_equal"])
        is not bool
        or not update0_metrics["all_action_predictions_bitwise_equal"]
        or update0_metrics["all_action_unordered_pair_count"] != 36
        or update0_metrics["all_action_prediction_row_count"]
        != SELECTION_ROLE_COUNTS["pairs"]
    ):
        raise ValueError("V11 update-zero action symmetry changed")

    retrieval = _validate_factorized_retrieval_observation(
        metrics["factorized_retrieval"],
        name="V11 Phase-A factorized retrieval",
    )
    update0_retrieval = _validate_factorized_retrieval_observation(
        update0_metrics["factorized_retrieval"],
        name="V11 update-zero factorized retrieval",
    )
    for field in ("all_values_finite", "ema_target_gradient_free"):
        if type(metrics[field]) is not bool:
            raise TypeError(f"{field} must be Boolean")
    if (
        metrics["pair_count"] != SELECTION_ROLE_COUNTS["pairs"]
        or type(metrics["pair_count"]) is not int
        or metrics["scene_family_count"] != len(SCENE_FAMILIES)
        or type(metrics["scene_family_count"]) is not int
        or metrics["non_hold_pair_count"] != SELECTION_NON_HOLD_PAIR_COUNT
        or type(metrics["non_hold_pair_count"]) is not int
    ):
        raise ValueError("V11 frozen observation population changed")

    nonnumeric = {
        "all_values_finite",
        "ema_target_gradient_free",
        "pair_count",
        "scene_family_count",
        "non_hold_pair_count",
        "factorized_retrieval",
    }
    values = {
        field: _finite_nonnegative(metrics[field], name=field)
        for field in sorted(PHASE_A_METRIC_FIELDS - nonnumeric)
    }
    update0_values = {
        field: _finite_nonnegative(
            update0_metrics[field], name=f"update0 {field}"
        )
        for field in sorted(PHASE_A_METRIC_FIELDS - nonnumeric)
    }
    for field in PHASE_A_UPDATE0_HEALTH_FIELDS:
        if update0_values[field] <= 0.0:
            raise ValueError(f"V11 update-zero {field} must be positive")
    ratios = {
        "normalized_projected_future_variance_to_update0": (
            values["normalized_projected_future_cross_sample_variance"]
            / update0_values[
                "normalized_projected_future_cross_sample_variance"
            ]
        ),
        "normalized_projected_future_spatial_diversity_to_update0": (
            values["normalized_projected_future_spatial_diversity"]
            / update0_values[
                "normalized_projected_future_spatial_diversity"
            ]
        ),
        "true_to_shuffled_next": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["shuffled_next_mse"],
            name="V11 shuffled-next ratio",
        ),
        "true_to_shuffled_current": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["shuffled_current_mse"],
            name="V11 shuffled-current ratio",
        ),
        "true_to_mean_target": _positive_denominator_ratio(
            values["true_pair_mse"],
            values["mean_target_mse"],
            name="V11 mean-target ratio",
        ),
    }

    immutable_fields = (
        "action_equal_logit_reference",
        "two_target_equal_logit_reference",
        "selection_target_mapping_sha256",
        "selection_action_permutation_sha256",
    )
    retrieval_health_fields = (
        "all_values_finite",
        "energy_values_within_closed_zero_four",
        "target_candidate_order_and_counts_exact",
        "same_action_target_mapping_exact",
        "selection_action_permutation_exact",
        "reference_values_immutable",
    )
    action_ratio_fields = (
        "executed_to_cyclic_ratio",
        "executed_to_hardest_wrong_ratio",
        "non_hold_executed_to_hold_ratio",
        "executed_to_permuted_ratio",
    )
    action_family_fields = (
        "cyclic_wrong_minus_executed_energy",
        "hardest_wrong_minus_executed_energy",
        "hold_minus_non_hold_executed_energy",
        "permuted_minus_executed_energy",
    )
    update0_action_chance_exact = (
        math.isclose(
            update0_retrieval["action_retrieval_nll"],
            update0_retrieval["action_equal_logit_reference"],
            rel_tol=0.0,
            abs_tol=UPDATE_ZERO_ACTION_NLL_ABSOLUTE_TOLERANCE,
        )
        and math.isclose(
            update0_retrieval[
                "action_retrieval_macro_balanced_accuracy"
            ],
            1.0 / len(ACTION_VOCABULARY),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and all(update0_retrieval[field] == 1.0 for field in action_ratio_fields)
        and all(
            family[field] == 0.0
            for family in update0_retrieval["per_family"].values()
            for field in action_family_fields
        )
    )
    family_populations_exact = all(
        row["scene_id"] == SELECTION_FAMILY_BINDINGS[family]["scene_id"]
        and row["row_count"]
        == SELECTION_FAMILY_BINDINGS[family]["row_count"]
        and row["same_action_row_count"]
        == SELECTION_FAMILY_BINDINGS[family]["same_action_row_count"]
        and row["non_hold_row_count"]
        == SELECTION_FAMILY_BINDINGS[family]["non_hold_row_count"]
        and row["hold_action_rows_match_non_hold_rows"]
        for family, row in retrieval["per_family"].items()
    )
    common = {
        "future_leakage_target_nonavuity_and_autograd_isolation": all(
            observation_integrity[field]
            for field in (
                "future_leakage_prohibition_passed",
                "target_path_nonvacuity_passed",
                "online_target_autograd_separation_passed",
            )
        ),
        "finite_and_ema_gradient_free": (
            metrics["all_values_finite"]
            and metrics["ema_target_gradient_free"]
        ),
        "ema_inventory_and_update_count_exact": (
            observation_integrity["ema_inventory_exact"]
            and observation_integrity["ema_update_count"]
            == observation_integrity["expected_ema_update_count"]
        ),
        "observation_rng_and_model_state_preserved": (
            observation_integrity["rng_state_preserved"]
            and observation_integrity["state_mutation_count"] == 0
        ),
        "normalized_population_and_all_nine_candidates_exact": (
            observation_integrity["normalized_population_exact"]
            and observation_integrity["all_nine_candidates_exact"]
            and observation_integrity["observation_row_count"]
            == SELECTION_ROLE_COUNTS["pairs"]
        ),
        "factorized_retrieval_health_exact": all(
            retrieval[field] for field in retrieval_health_fields
        ),
        "update_zero_factorized_retrieval_health_exact": all(
            update0_retrieval[field] for field in retrieval_health_fields
        ),
        "retrieval_references_and_mappings_immutable": all(
            retrieval[field] == update0_retrieval[field]
            for field in immutable_fields
        ),
        "update_zero_action_symmetry_and_chance_exact": (
            update0_metrics["all_action_predictions_bitwise_equal"]
            and update0_action_chance_exact
        ),
        "normalized_projected_future_variance_at_least_quarter_update0": (
            ratios[
                "normalized_projected_future_variance_to_update0"
            ] >= 0.25
        ),
        "normalized_projected_future_spatial_diversity_at_least_quarter_update0": (
            ratios[
                "normalized_projected_future_spatial_diversity_to_update0"
            ] >= 0.25
        ),
        "true_at_most_point90_shuffled_next": (
            ratios["true_to_shuffled_next"] <= 0.90
        ),
        "true_at_most_point95_shuffled_current": (
            ratios["true_to_shuffled_current"] <= 0.95
        ),
        "control_populations_and_one_scene_per_family_exact": (
            family_populations_exact
        ),
    }
    return {
        "metrics": metrics,
        "values": values,
        "update0_values": update0_values,
        "retrieval": retrieval,
        "update0_retrieval": update0_retrieval,
        "ratios": ratios,
        "common": common,
        "per_family": retrieval["per_family"],
    }


def _v11_update_100_conjuncts(
    normalized: Mapping[str, Any],
) -> dict[str, bool]:
    values = normalized["values"]
    update0 = normalized["update0_values"]
    retrieval = normalized["retrieval"]
    threshold = PHASE_A_UPDATE_100_THRESHOLDS
    return {
        **dict(normalized["common"]),
        "masked_future_jepa_strictly_below_update0": (
            values["masked_future_jepa_loss"]
            < update0["masked_future_jepa_loss"]
        ),
        "normalized_projected_future_rank_above_registered_update100_threshold": (
            values["normalized_projected_future_effective_rank"]
            > threshold[
                "normalized_projected_future_effective_rank_"
                "strictly_greater_than"
            ]
        ),
        "action_nll_at_most_point99_equal_logit_reference": (
            retrieval["action_retrieval_nll"]
            <= threshold["action_equal_logit_reference_factor_maximum"]
            * retrieval["action_equal_logit_reference"]
        ),
        "action_macro_balanced_accuracy_at_least_point15": (
            retrieval["action_retrieval_macro_balanced_accuracy"]
            >= threshold["action_macro_balanced_accuracy_minimum"]
        ),
        "same_action_two_target_nll_at_most_point95_reference": (
            retrieval["same_action_two_target_nll"]
            <= threshold[
                "same_action_two_target_reference_factor_maximum"
            ] * retrieval["two_target_equal_logit_reference"]
        ),
        "same_action_strict_win_rate_at_least_point60": (
            retrieval["same_action_strict_win_rate"]
            >= threshold["same_action_strict_win_rate_minimum"]
        ),
        "non_hold_correct_to_fixed_current_strictly_below_one": (
            retrieval["non_hold_correct_to_current_ratio"]
            < threshold[
                "non_hold_correct_current_ratio_strictly_less_than"
            ]
        ),
        "fixed_current_margin_positive_in_at_least_six_families": (
            retrieval["current_target_positive_family_margin_count"]
            >= threshold["positive_family_margin_count_minimum"]
        ),
        "hold_margin_positive_in_at_least_six_families": (
            retrieval["hold_positive_family_margin_count"]
            >= threshold["positive_family_margin_count_minimum"]
        ),
        "permuted_margin_positive_in_at_least_six_families": (
            retrieval["permuted_positive_family_margin_count"]
            >= threshold["positive_family_margin_count_minimum"]
        ),
    }


def evaluate_phase_a_update_zero(
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Enforce every registered common invariant before the first update."""

    normalized = _normalize_v11_phase_a_inputs(
        metrics, update0_metrics, observation_integrity
    )
    conjuncts = dict(normalized["common"])
    passed = all(conjuncts.values())
    return {
        "update": 0,
        "passed": passed,
        "control": (
            CONTROL_CONTINUE
            if passed
            else CONTROL_PHASE_A_UPDATE_ZERO_FAIL
        ),
        "conjuncts": conjuncts,
        "ratios": dict(normalized["ratios"]),
        "thresholds": {"common_invariants_only": True},
        "per_family": dict(normalized["per_family"]),
        "factorized_retrieval": dict(normalized["retrieval"]),
    }


def _v11_loss_progress_conjuncts(
    normalized: Mapping[str, Any],
    previous_metrics: Mapping[str, Any],
    *,
    previous_update: int,
) -> dict[str, bool]:
    if (
        type(previous_metrics) is not dict
        or set(previous_metrics) != PHASE_A_METRIC_FIELDS
    ):
        raise ValueError(f"V11 update-{previous_update} fields changed")
    previous_retrieval = _validate_factorized_retrieval_observation(
        previous_metrics["factorized_retrieval"],
        name=f"V11 update-{previous_update} factorized retrieval",
    )
    current = normalized["retrieval"]
    values = normalized["values"]
    suffix = f"update{previous_update}"
    immutable = (
        "action_equal_logit_reference",
        "two_target_equal_logit_reference",
        "selection_target_mapping_sha256",
        "selection_action_permutation_sha256",
    )
    return {
        f"references_and_mappings_match_{suffix}": all(
            current[field] == previous_retrieval[field]
            for field in immutable
        ),
        f"masked_future_jepa_strictly_lower_than_{suffix}": (
            values["masked_future_jepa_loss"]
            < _finite_nonnegative(
                previous_metrics["masked_future_jepa_loss"],
                name=f"{suffix} masked future JEPA",
            )
        ),
        f"action_retrieval_nll_strictly_lower_than_{suffix}": (
            current["action_retrieval_nll"]
            < previous_retrieval["action_retrieval_nll"]
        ),
        f"target_retrieval_nll_strictly_lower_than_{suffix}": (
            current["target_retrieval_nll"]
            < previous_retrieval["target_retrieval_nll"]
        ),
        f"same_action_target_retrieval_nll_strictly_lower_than_{suffix}": (
            current["same_action_target_retrieval_nll"]
            < previous_retrieval["same_action_target_retrieval_nll"]
        ),
        f"same_action_two_target_nll_strictly_lower_than_{suffix}": (
            current["same_action_two_target_nll"]
            < previous_retrieval["same_action_two_target_nll"]
        ),
        f"action_macro_balanced_accuracy_not_below_{suffix}": (
            current["action_retrieval_macro_balanced_accuracy"]
            >= previous_retrieval[
                "action_retrieval_macro_balanced_accuracy"
            ]
        ),
    }


def _v11_update_400_mechanism_conjuncts(
    normalized: Mapping[str, Any],
) -> dict[str, bool]:
    values = normalized["values"]
    ratios = normalized["ratios"]
    retrieval = normalized["retrieval"]
    threshold = PHASE_A_UPDATE_400_THRESHOLDS
    ratio_fields = (
        "same_action_correct_to_deranged_ratio",
        "non_hold_correct_to_current_ratio",
        "executed_to_cyclic_ratio",
        "executed_to_hardest_wrong_ratio",
        "non_hold_executed_to_hold_ratio",
        "executed_to_permuted_ratio",
    )
    return {
        "normalized_projected_future_rank_at_least_registered_update400_threshold": (
            values["normalized_projected_future_effective_rank"]
            >= threshold[
                "normalized_projected_future_effective_rank_minimum"
            ]
        ),
        "action_nll_at_most_point95_equal_logit_reference": (
            retrieval["action_retrieval_nll"]
            <= threshold["action_equal_logit_reference_factor_maximum"]
            * retrieval["action_equal_logit_reference"]
        ),
        "action_macro_balanced_accuracy_strictly_above_two_ninths": (
            retrieval["action_retrieval_macro_balanced_accuracy"]
            > threshold[
                "action_macro_balanced_accuracy_strictly_greater_than"
            ]
        ),
        **{
            f"{field}_strictly_below_point99": (
                retrieval[field]
                < threshold["energy_ratio_strictly_less_than"]
            )
            for field in ratio_fields
        },
        "cyclic_margin_positive_in_at_least_six_families": (
            retrieval["cyclic_positive_family_margin_count"]
            >= threshold["positive_family_margin_count_minimum"]
        ),
        "hold_margin_still_positive_in_at_least_six_families": (
            retrieval["hold_positive_family_margin_count"]
            >= threshold["positive_family_margin_count_minimum"]
        ),
        "permuted_margin_still_positive_in_at_least_six_families": (
            retrieval["permuted_positive_family_margin_count"]
            >= threshold["positive_family_margin_count_minimum"]
        ),
        "true_strictly_below_mean_target": (
            ratios["true_to_mean_target"]
            < threshold["mean_target_ratio_strictly_less_than"]
        ),
    }


def evaluate_phase_a_continuation(
    update: int,
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
    previous_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if update not in {100, 400}:
        raise ValueError("V11 continuation update must be 100 or 400")
    normalized = _normalize_v11_phase_a_inputs(
        metrics, update0_metrics, observation_integrity
    )
    conjuncts = _v11_update_100_conjuncts(normalized)
    if update == 100:
        if previous_metrics is not None:
            raise ValueError("V11 update-100 has no previous checkpoint")
        thresholds: Mapping[str, Any] = PHASE_A_UPDATE_100_THRESHOLDS
        failure = CONTROL_PHASE_A_UPDATE_100_FAIL
    else:
        if previous_metrics is None:
            raise ValueError("V11 update-400 requires update-100 metrics")
        conjuncts.update(_v11_update_400_mechanism_conjuncts(normalized))
        conjuncts.update(_v11_loss_progress_conjuncts(
            normalized, previous_metrics, previous_update=100
        ))
        thresholds = PHASE_A_UPDATE_400_THRESHOLDS
        failure = CONTROL_PHASE_A_UPDATE_400_FAIL
    passed = all(conjuncts.values())
    return {
        "update": update,
        "passed": passed,
        "control": CONTROL_CONTINUE if passed else failure,
        "conjuncts": conjuncts,
        "ratios": dict(normalized["ratios"]),
        "thresholds": dict(thresholds),
        "per_family": dict(normalized["per_family"]),
        "factorized_retrieval": dict(normalized["retrieval"]),
    }


def evaluate_phase_a(
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
    previous_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _normalize_v11_phase_a_inputs(
        metrics, update0_metrics, observation_integrity
    )
    values = normalized["values"]
    ratios = normalized["ratios"]
    retrieval = normalized["retrieval"]
    threshold = PHASE_A_PASS_THRESHOLDS
    conjuncts = {
        **_v11_update_100_conjuncts(normalized),
        **_v11_update_400_mechanism_conjuncts(normalized),
        **_v11_loss_progress_conjuncts(
            normalized, previous_metrics, previous_update=400
        ),
        "normalized_projected_future_rank_at_least_48": (
            values["normalized_projected_future_effective_rank"]
            >= threshold[
                "normalized_projected_future_effective_rank_minimum"
            ]
        ),
        "action_nll_at_most_point90_equal_logit_reference": (
            retrieval["action_retrieval_nll"]
            <= threshold["action_equal_logit_reference_factor_maximum"]
            * retrieval["action_equal_logit_reference"]
        ),
        "action_macro_balanced_accuracy_at_least_one_third": (
            retrieval["action_retrieval_macro_balanced_accuracy"]
            >= threshold["action_macro_balanced_accuracy_minimum"]
        ),
        "true_at_most_point90_mean_target": (
            ratios["true_to_mean_target"]
            <= threshold["mean_target_ratio_maximum"]
        ),
        **{
            f"{field}_at_most_point95": (
                retrieval[field] <= threshold["energy_ratio_maximum"]
            )
            for field in (
                "same_action_correct_to_deranged_ratio",
                "non_hold_correct_to_current_ratio",
                "executed_to_cyclic_ratio",
                "executed_to_hardest_wrong_ratio",
                "non_hold_executed_to_hold_ratio",
                "executed_to_permuted_ratio",
            )
        },
    }
    passed = all(conjuncts.values())
    return {
        "update": 1_000,
        "passed": passed,
        "control": CONTROL_PHASE_A_PASS if passed else CONTROL_PHASE_A_FAIL,
        "conjuncts": conjuncts,
        "ratios": dict(ratios),
        "thresholds": {
            "update_100": dict(PHASE_A_UPDATE_100_THRESHOLDS),
            "update_400": dict(PHASE_A_UPDATE_400_THRESHOLDS),
            "terminal": dict(PHASE_A_PASS_THRESHOLDS),
        },
        "per_family": dict(normalized["per_family"]),
        "factorized_retrieval": dict(retrieval),
    }


AUTHORIZATION_STATUS = "AUTHORIZED_ONE_EXACT_V11_MASKED_PAIR_TUBELET_PROBE"


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "authorizer",
        "independent_source_review",
        "preregistration",
        "runtime_inputs",
        "experiment",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V11 execution-authorization fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    authorizer = value["authorizer"]
    try:
        expected_review = validate_binding(
            dict(review_binding), path=REVIEW_RELATIVE_PATH
        )
    except (TypeError, ValueError) as error:
        raise PermissionError("V11 review binding changed") from error
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_source_review"] != expected_review
        or value["preregistration"] != preregistration_binding()
        or validate_runtime_inputs(value["runtime_inputs"])
        != value["runtime_inputs"]
        or value["experiment"] != science_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V11 execution authorization changed")
    return dict(value)


__all__ = sorted(set(__all__) | {
    name for name in globals()
    if name.isupper() or name in {
        "build_schedule_identity",
        "evaluate_phase_a",
        "evaluate_phase_a_continuation",
        "evaluate_phase_a_update_zero",
        "phase_a_model_config",
        "runtime_authorization_template",
        "science_contract",
        "v11_model_config",
        "validate_authorization",
    }
})
