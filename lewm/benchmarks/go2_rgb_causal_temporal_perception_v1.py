"""Source-only contract for the causal temporal perception V1 falsification.

Importing this module reads no generated input, dataset, RGB, checkpoint, or
tensor and imports no accelerator framework.  A separately generated recursive
manifest binds the complete Python closure without embedding the contract's own
hash.  Independent review and authorization remain external artifacts whose
exact hashes are supplied to the launcher, avoiding source/review/auth cycles.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_AUTHOR = "/root/runtime_reuse_audit"
SCHEMA_PREFIX = "lewm_go2_rgb_causal_temporal_perception_v1"

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5_multires_temporal_v1.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_causal_temporal_perception_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_causal_temporal_perception_v1.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_contract.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_shared_observable_camera_ray_jepa_v5_multires_temporal_v1.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_runner.py"
)
EVALUATOR_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_evaluator.py"
)
RECEIPT_BOUNDARY_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_receipt_boundary.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_causal_temporal_perception_v1_source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_source_closure.py"
)
SOURCE_CLOSURE_BASE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_multires_probe_source_closure_v3.py"
)
TEST_RELATIVE_PATH = RUNNER_TEST_RELATIVE_PATH

PREDECESSOR_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py"
)
SCHEDULE_ADAPTER_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py"
)
MATCHED_V1_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py"
)
MATCHED_V1_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py"
)
TAIL_DEPTH_LOSS_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5_"
    "protected_camera_adaptation_v4_tail_depth.py"
)
PREDECESSOR_MODEL_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py"
)

ADDITIVE_SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    EVALUATOR_TEST_RELATIVE_PATH,
    RECEIPT_BOUNDARY_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)
REUSED_SOURCE_PATHS = (
    PREDECESSOR_CONTRACT_RELATIVE_PATH,
    PREDECESSOR_MODEL_RELATIVE_PATH,
    SCHEDULE_ADAPTER_RELATIVE_PATH,
    MATCHED_V1_CONTRACT_RELATIVE_PATH,
    MATCHED_V1_RUNNER_RELATIVE_PATH,
    TAIL_DEPTH_LOSS_RELATIVE_PATH,
)
SOURCE_PATHS = tuple(dict.fromkeys((*ADDITIVE_SOURCE_PATHS, *REUSED_SOURCE_PATHS)))

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_causal_temporal_perception_v1_"
    "preregistration_2026-07-24.md"
)
PREREGISTRATION_COMMIT = "3e30b8ae9dbdfeafd0f62bfc4243cece7a885d95"
PREREGISTRATION_FILE_SHA256 = (
    "72377d64dcc70c15eaadac0130d2f9244ca93141d2948c01ebe2011102bca405"
)
PREREGISTRATION_BYTE_COUNT = 11_473
PREREGISTRATION_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_causal_temporal_perception_v1_"
    "preregistration_independent_review_2026-07-24.json"
)
PREREGISTRATION_REVIEW_FILE_SHA256 = (
    "ac54f078ed769a6aa5d06febacdb6b5e54ad860287d68def2ada772741ae8ee6"
)
PREREGISTRATION_REVIEW_CONTENT_SHA256 = (
    "dbe1018af69583e63282782d49012500909b26a6ec0af321674ce3b528c253df"
)
PREREGISTRATION_REVIEW_BYTE_COUNT = 3_790

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_causal_temporal_perception_v1_"
    "source_manifest_2026-07-24.json"
)
SOURCE_MANIFEST_SCHEMA = (
    "lewm_go2_rgb_causal_temporal_perception_v1_source_manifest"
)
SOURCE_MANIFEST_ENTRYPOINTS = (
    LAUNCHER_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = (
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    PREDECESSOR_CONTRACT_RELATIVE_PATH,
    PREDECESSOR_MODEL_RELATIVE_PATH,
    SCHEDULE_ADAPTER_RELATIVE_PATH,
    MATCHED_V1_CONTRACT_RELATIVE_PATH,
    MATCHED_V1_RUNNER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    EVALUATOR_TEST_RELATIVE_PATH,
    RECEIPT_BOUNDARY_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_BASE_CHECKER_RELATIVE_PATH,
)
SOURCE_MANIFEST_EXCLUDED_RUNTIME_CATEGORIES = (
    ".generated artifacts and attempt registries",
    "tensor checkpoints and metric sidecars",
    "raw RGB, scene shards, datasets, and role payloads",
    "configuration and custody roots",
    "sealed or held-out benchmark material",
    "review, authorization, result, and completion records",
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_causal_temporal_perception_v1_"
    "source_review_2026-07-24.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_causal_temporal_perception_v1_"
    "execution_authorization_2026-07-24.json"
)
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    PREREGISTRATION_REVIEW_RELATIVE_PATH,
)

# These are already frozen predecessor identities, not hashes of this additive
# source graph.  The external recursive manifest binds the final temporal
# contract, model, runner, launcher, tests, and their complete Python closure.
# Review and authorization file hashes are supplied at launch and are therefore
# deliberately absent here; embedding them would recreate a self-reference.
FROZEN_SOURCE_SHA256 = {
    PREDECESSOR_CONTRACT_RELATIVE_PATH:
        "3553810c79686f642a30fdfd0d2ff6ae047a97ea65c1366cae4cb3231e44e669",
    PREDECESSOR_MODEL_RELATIVE_PATH:
        "a63da1137539953b2f40d184def1652ae05f63d7b434084b1a91787e1fc83d0b",
    SCHEDULE_ADAPTER_RELATIVE_PATH:
        "a8efe19da92c9c2107f11be38db8ed80e66aedca3ef41af0428ab13d50f56bd1",
    MATCHED_V1_CONTRACT_RELATIVE_PATH:
        "53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a",
    MATCHED_V1_RUNNER_RELATIVE_PATH:
        "e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578",
    TAIL_DEPTH_LOSS_RELATIVE_PATH:
        "6fc0a114386ee2fb0ae98704a970d38a7194db192283b904138015498fb02384",
}

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_causal_temporal_perception_probe_v1"
)
PROHIBITED_RUNTIME_OUTPUT_ROOTS = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v1",
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v2",
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v3",
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v3_retry2",
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v1",
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v2",
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v3",
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v4",
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v5_native_schedule_completion",
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v5_native_schedule_completion_"
    "environment_recovery_v1",
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v6_final_fresh_update0_tail_depth_8k",
)


def _load_predecessor_contract() -> Any:
    """Load only the standard-library predecessor contract by exact source path."""

    source = ROOT / PREDECESSOR_CONTRACT_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location(
        "_lewm_temporal_v1_predecessor_contract",
        source,
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load predecessor perception contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PREDECESSOR = _load_predecessor_contract()

# Reuse only identity-free canonical serialization and path/binding helpers.
canonical_json_bytes = _PREDECESSOR.canonical_json_bytes
canonical_json_sha256 = _PREDECESSOR.canonical_json_sha256
is_sha256 = _PREDECESSOR.is_sha256
with_content_sha256 = _PREDECESSOR.with_content_sha256
parse_canonical_json = _PREDECESSOR.parse_canonical_json
safe_relative_path = _PREDECESSOR.safe_relative_path
artifact_binding = _PREDECESSOR.artifact_binding
validate_binding = _PREDECESSOR.validate_binding

# The new probe reuses the already bound raw, N320, and schedule inputs.  These
# assignments copy source constants only; importing this module opens none of
# the referenced generated paths.
RAW_ROOT_RELATIVE_PATH = _PREDECESSOR.RAW_ROOT_RELATIVE_PATH
RAW_MANIFEST_RELATIVE_PATH = _PREDECESSOR.RAW_MANIFEST_RELATIVE_PATH
RAW_AUDIT_RELATIVE_PATH = _PREDECESSOR.RAW_AUDIT_RELATIVE_PATH
RAW_PAIRS_RELATIVE_PATH = f"{RAW_ROOT_RELATIVE_PATH}/pairs.jsonl"
RAW_ENDPOINTS_RELATIVE_PATH = f"{RAW_ROOT_RELATIVE_PATH}/endpoints.jsonl"
N320_ROOT_RELATIVE_PATH = _PREDECESSOR.N320_ROOT_RELATIVE_PATH
N320_GATE_RELATIVE_PATH = _PREDECESSOR.N320_GATE_RELATIVE_PATH
N320_CHECKPOINT_RELATIVE_PATH = _PREDECESSOR.N320_CHECKPOINT_RELATIVE_PATH
SCHEDULE_RELATIVE_PATH = _PREDECESSOR.SCHEDULE_RELATIVE_PATH
RUNTIME_FILE_SHA256 = {
    **dict(_PREDECESSOR.RUNTIME_FILE_SHA256),
    RAW_PAIRS_RELATIVE_PATH:
        "5a6f7de405206aba855051bd9e14cab5262cfbfebc070ed02ef81d8cf62afc8d",
    RAW_ENDPOINTS_RELATIVE_PATH:
        "34e47ddcc40ad8c1f092c73193d16773cf4dedae05e7f4f684abb385cc2c0d01",
}
RUNTIME_CONTENT_SHA256 = dict(_PREDECESSOR.RUNTIME_CONTENT_SHA256)
RUNTIME_BYTE_COUNTS = {
    **dict(_PREDECESSOR.RUNTIME_BYTE_COUNTS),
    RAW_PAIRS_RELATIVE_PATH: 6_207_286,
    RAW_ENDPOINTS_RELATIVE_PATH: 9_108_028,
}

MODEL_FAMILY = "shared_observable_camera_ray_jepa_v5_multires_temporal_v1"
MODEL_RUNTIME_VERSION = (
    "lewm_go2_rgb_causal_temporal_perception_v1_model_runtime_v1"
)
BASE_INITIALIZATION_SEED = 20_260_712
DECODER_INITIALIZATION_SEED = 20_260_724
TEMPORAL_INITIALIZATION_SEED = 20_260_725
SCHEDULE_SEED = 20_260_713

TRAIN_ROLE_COUNTS = {
    "pairs": 4_262,
    "unique_endpoints": 7_777,
    "scenes": 72,
}
SELECTION_ROLE_COUNTS = {
    "pairs": 495,
    "unique_endpoints": 924,
    "warm_endpoints": 495,
    "cold_endpoints": 429,
    "both_roles": 66,
    "ambiguous_predecessors": 0,
    "scenes": 8,
}
CALIBRATION_ROLE_COUNTS = {
    "pairs": 415,
    "unique_endpoints": 759,
    "scenes": 8,
}
FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
SCOPES = ("aggregate", *FAMILIES)

HISTORY_LAG_S = 0.5
HISTORY_TICK_COUNT = 5
HISTORY_TICK_S = 0.10
TEMPORAL_CHANNEL_COUNT = 8
TEMPORAL_PARAMETER_COUNT = 3_160
TEMPORAL_PARAMETER_TENSOR_COUNT = 5
TEMPORAL_STATE_PREFIX = "evidence_head.temporal_residual."
PREDECESSOR_EVIDENCE_HEAD_PARAMETER_COUNT = 352_689
PREDECESSOR_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT = 26
EVIDENCE_HEAD_PARAMETER_CEILING = 357_993
EXPECTED_PARAMETER_COUNTS = {
    "evidence_head": 355_849,
    "encoder": 2_747_520,
}
EXPECTED_PARAMETER_TENSOR_COUNTS = {
    "evidence_head": 31,
    "encoder": 78,
}
EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT = 3_103_369
EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT = 109

TEMPORAL_MODEL_INPUTS = (
    "previous_raw_visual_tokens_at_fixed_lag",
    "current_raw_visual_tokens",
    "history_valid",
)
FORBIDDEN_TEMPORAL_MODEL_INPUTS = (
    "requested_primitive",
    "requested_primitive_proxy",
    "median_commanded_delta",
    "executed_command",
    "realized_simulator_se2",
    "exact_simulator_pose",
    "camera_attitude_delta",
    "depth_input",
    "ground_truth",
    "scene_geometry",
    "evaluator_feedback",
)
MODEL_INPUTS = TEMPORAL_MODEL_INPUTS
FORBIDDEN_MODEL_INPUTS = FORBIDDEN_TEMPORAL_MODEL_INPUTS

MAXIMUM_UPDATE = 1_000
CHECKPOINT_UPDATES = (100, 400, 1_000)
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
EFFECTIVE_BATCH_SIZE = 16
MAXIMUM_PRESENTATIONS = 16_000
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = {
    100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    400: "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
    1_000: "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
}

TRAINABLE_PARAMETER_PREFIXES = ("evidence_head.", "encoder.")
FROZEN_STATE_PREFIXES = (
    "bev_decoder.",
    "predictor.",
    "occupancy_head.",
    "target_encoder.",
    "target_bev_decoder.",
)
POST_CLIP_NORM_ASSERTION_TOLERANCE = 1e-5

MARGIN_COUNT = 189
PASS_THRESHOLDS = {
    "complete_physical_scope_count_minimum": 1,
    "passed_margin_count_minimum": 98,
    "total_shortfall_strictly_less_than": 41.01776266878769,
    "rough_pixel_balanced_accuracy_strictly_greater_than":
        0.8198594673963917,
    "rough_ground_balanced_accuracy_strictly_greater_than":
        0.647134926562893,
    "rough_depth_p95_m_strictly_less_than": 0.9777327477931971,
}
PHYSICAL_LOWER_THRESHOLDS = dict(_PREDECESSOR.PHYSICAL_LOWER_THRESHOLDS)
PHYSICAL_UPPER_THRESHOLDS = dict(_PREDECESSOR.PHYSICAL_UPPER_THRESHOLDS)

CONTROL_CONTINUE = "CONTINUE_INFORMATIONAL"
CONTROL_PASS = "PASS_BOUNDED_FALSIFICATION"
CONTROL_FAIL = "FAIL_TERMINAL_NO_RETRY"
CONTROL_INTEGRITY_FAIL = "INTEGRITY_FAILURE_TERMINAL_NO_RETRY"

REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_snapshot_v1"
METRIC_SIDECAR_SCHEMA = f"{SCHEMA_PREFIX}_metric_sidecar_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
PRE_LEDGER_FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_pre_ledger_failure_v1"
PARTIAL_ACCESS_RECORD_SCHEMA = f"{SCHEMA_PREFIX}_partial_access_record_v1"
PARTIAL_ACCESS_LEDGER_SCHEMA = f"{SCHEMA_PREFIX}_partial_access_ledger_v1"

DOWNSTREAM_DENIALS = {
    "probe_checkpoint_qualified": False,
    "perception_qualification_authorized": False,
    "probability_calibration_authorized": False,
    "jepa_training_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "prior_runtime_output_open_authorized": False,
    "retry_resume_repair_second_seed_extension_or_rerun_authorized": False,
}
SOURCE_ONLY_AUTHORITY = {
    "execution_authorized": False,
    "gpu_or_hardware_authorized": False,
    "generated_input_open_authorized": False,
    "dataset_or_rgb_open_authorized": False,
    "checkpoint_or_tensor_open_authorized": False,
    "output_root_inspection_or_reservation_authorized": False,
    **DOWNSTREAM_DENIALS,
}
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    "one_exact_probe_attempt_authorized": True,
    "one_discrete_r9700_authorized": True,
    "n320_initialization_only_authorized": True,
    "train_and_checkpoint_selection_roles_only_authorized": True,
    "generated_mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
    "output_root_must_be_absent": True,
    **DOWNSTREAM_DENIALS,
}

PARTIAL_OPERATION_INTEGER_FIELDS = (
    "optimizer_construction_attempt_count",
    "optimizer_construction_completion_count",
    "optimizer_update_attempt_count",
    "complete_optimizer_updates",
    "pair_index_presentations_attempted",
    "pair_index_presentations_materialized",
    "microbatch_attempt_count",
    "microbatch_completion_count",
    "camera_objective_attempt_count",
    "camera_objective_completion_count",
    "finite_camera_objective_count",
    "backward_attempt_count",
    "backward_completion_count",
    "head_clip_attempt_count",
    "head_clip_completion_count",
    "encoder_clip_attempt_count",
    "encoder_clip_completion_count",
    "optimizer_step_attempt_count",
    "optimizer_step_completion_count",
    "checkpoint_snapshot_completion_count",
    "checkpoint_selection_evaluation_attempt_count",
    "checkpoint_selection_evaluation_completion_count",
    "metric_sidecar_publication_count",
    "observer_evaluation_rerun_count",
    "jepa_objective_count",
    "jepa_backward_count",
    "ema_update_count_after_initial_hard_sync",
    "global_clip_invocation_count",
    "probability_calibration_open_count",
    "prior_runtime_output_open_count",
)


def preregistration_binding() -> dict[str, Any]:
    return {
        "commit": PREREGISTRATION_COMMIT,
        "path": PREREGISTRATION_RELATIVE_PATH,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
        "independent_review": {
            "path": PREREGISTRATION_REVIEW_RELATIVE_PATH,
            "file_sha256": PREREGISTRATION_REVIEW_FILE_SHA256,
            "content_sha256": PREREGISTRATION_REVIEW_CONTENT_SHA256,
            "byte_count": PREREGISTRATION_REVIEW_BYTE_COUNT,
            "verdict": "PASS",
        },
    }


def _read_regular_source(path: Path) -> bytes:
    """Read one source artifact with no-follow fingerprint continuity."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError("O_NOFOLLOW is required for temporal source custody")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"source is not regular: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise PermissionError(f"opened source is not regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)

    def fingerprint(value: os.stat_result) -> tuple[int, ...]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    if not (
        fingerprint(before)
        == fingerprint(opened_before)
        == fingerprint(opened_after)
        == fingerprint(after)
    ):
        raise RuntimeError(f"source changed while read: {path}")
    return b"".join(chunks)


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    """Validate the external recursive source closure without self-reference."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"source manifest contains nonfinite constant {value}")

    def reject_duplicates(
        pairs: Sequence[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"source manifest repeats key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PermissionError(
            "source manifest is not finite duplicate-safe JSON"
        ) from error
    fields = {
        "authority",
        "consumed_adaptation_runner_source_count",
        "content_sha256",
        "date",
        "entrypoints",
        "excluded_runtime_categories",
        "forced_dynamic_sources",
        "generated_input_open_count",
        "schema",
        "sealed_or_heldout_open_count",
        "source_bindings",
        "source_bindings_sha256",
        "source_count",
        "source_paths",
        "status",
        "tensor_checkpoint_open_count",
        "whole_tree_export_authorized",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("source manifest fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    bindings = value["source_bindings"]
    paths = value["source_paths"]
    if (
        value["schema"] != SOURCE_MANIFEST_SCHEMA
        or value["status"] != "SOURCE_ONLY_RECURSIVE_CLOSURE"
        or value["date"] != "2026-07-24"
        or value["authority"]
        != "source_closure_only_no_generated_input_checkpoint_training_gpu_"
        "qualification_g2_navigation_heldout_production_or_promotion_authority"
        or value["entrypoints"] != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value["forced_dynamic_sources"]
        != list(SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
        or value["excluded_runtime_categories"]
        != list(SOURCE_MANIFEST_EXCLUDED_RUNTIME_CATEGORIES)
        or value["consumed_adaptation_runner_source_count"] != 0
        or value["generated_input_open_count"] != 0
        or value["sealed_or_heldout_open_count"] != 0
        or value["tensor_checkpoint_open_count"] != 0
        or value["whole_tree_export_authorized"] is not False
        or type(value["source_count"]) is not int
        or value["source_count"] <= 0
        or type(bindings) is not list
        or len(bindings) != value["source_count"]
        or type(paths) is not list
        or len(paths) != value["source_count"]
        or not is_sha256(value["source_bindings_sha256"])
        or value["source_bindings_sha256"]
        != canonical_json_sha256(bindings)
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("source manifest contract changed")
    normalized: list[dict[str, Any]] = []
    for binding in bindings:
        if (
            type(binding) is not dict
            or set(binding) != {"path", "file_sha256", "byte_count"}
        ):
            raise PermissionError("source-manifest binding fields changed")
        path = safe_relative_path(binding["path"], name="source-manifest path")
        parts = PurePosixPath(path).parts
        if (
            not path.endswith(".py")
            or any(
                part in {
                    ".generated",
                    "config",
                    "configs",
                    "custody",
                    "data",
                    "datasets",
                    "sealed",
                }
                or part.startswith("sealed_")
                for part in parts
            )
            or path.endswith("sealed_test.json")
            or not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError(f"source-manifest path is forbidden: {path}")
        normalized.append(dict(binding))
    normalized_paths = [binding["path"] for binding in normalized]
    if (
        normalized_paths != paths
        or normalized_paths != sorted(normalized_paths)
        or len(set(normalized_paths)) != len(normalized_paths)
        or not (
            set(SOURCE_PATHS)
            | set(SOURCE_MANIFEST_ENTRYPOINTS)
            | set(SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
        ).issubset(normalized_paths)
    ):
        raise PermissionError("source-manifest path order or closure changed")
    return dict(value)


def source_freeze_status(root: Path = ROOT) -> dict[str, Any]:
    """Report whether the noncyclic external source manifest is usable."""

    manifest_path = root / SOURCE_MANIFEST_RELATIVE_PATH
    try:
        manifest_raw = _read_regular_source(manifest_path)
        manifest = validate_source_manifest(manifest_raw)
    except FileNotFoundError:
        return {
            "ready": False,
            "manifest_present": False,
            "manifest_valid": False,
            "expected_source_count": len(SOURCE_PATHS),
            "bound_source_count": 0,
            "missing_source_paths": list(SOURCE_PATHS),
            "extra_source_paths": [],
            "malformed_source_bindings": [],
            "inconsistent_freeze_fields": [],
            "unset_freeze_fields": ["external_recursive_source_manifest"],
            "manifest_error": "absent",
        }
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as error:
        return {
            "ready": False,
            "manifest_present": True,
            "manifest_valid": False,
            "expected_source_count": len(SOURCE_PATHS),
            "bound_source_count": 0,
            "missing_source_paths": list(SOURCE_PATHS),
            "extra_source_paths": [],
            "malformed_source_bindings": [],
            "inconsistent_freeze_fields": [],
            "unset_freeze_fields": [],
            "manifest_error": type(error).__name__,
        }
    paths = set(manifest["source_paths"])
    missing = sorted(set(SOURCE_PATHS) - paths)
    frozen_index = {
        binding["path"]: binding["file_sha256"]
        for binding in manifest["source_bindings"]
    }
    inconsistent = sorted(
        relative
        for relative, expected in FROZEN_SOURCE_SHA256.items()
        if frozen_index.get(relative) != expected
    )
    return {
        "ready": not missing and not inconsistent,
        "manifest_present": True,
        "manifest_valid": True,
        "expected_source_count": len(SOURCE_PATHS),
        "bound_source_count": int(manifest["source_count"]),
        "missing_source_paths": missing,
        "extra_source_paths": sorted(paths - set(SOURCE_PATHS)),
        "malformed_source_bindings": [],
        "inconsistent_freeze_fields": inconsistent,
        "unset_freeze_fields": [],
        "manifest_error": None,
    }


def require_source_freeze(root: Path = ROOT) -> None:
    status = source_freeze_status(root)
    if not status["ready"]:
        raise PermissionError(
            "temporal V1 external recursive source freeze is incomplete"
        )


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    """Rehash every external-manifest source and noncyclic review input."""

    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        relative = binding["path"]
        source_raw = _read_regular_source(root / relative)
        digest = hashlib.sha256(source_raw).hexdigest()
        if (
            digest != binding["file_sha256"]
            or len(source_raw) != binding["byte_count"]
        ):
            raise PermissionError(f"manifest-bound source changed: {relative}")
        result[relative] = digest
    for relative in SOURCE_REVIEW_ADDITIONAL_PATHS:
        if relative in result:
            continue
        source_raw = (
            manifest_raw
            if relative == SOURCE_MANIFEST_RELATIVE_PATH
            else _read_regular_source(root / relative)
        )
        result[relative] = hashlib.sha256(source_raw).hexdigest()
    if not set(SOURCE_PATHS).issubset(result):
        raise PermissionError("direct temporal source escaped recursive closure")
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        if result.get(relative) != expected:
            raise PermissionError(f"frozen predecessor source changed: {relative}")
    if (
        result.get(PREREGISTRATION_RELATIVE_PATH)
        != PREREGISTRATION_FILE_SHA256
        or result.get(PREREGISTRATION_REVIEW_RELATIVE_PATH)
        != PREREGISTRATION_REVIEW_FILE_SHA256
    ):
        raise PermissionError("temporal preregistration source changed")
    return result


def hardware_contract() -> dict[str, Any]:
    """Exact no-payload preflight required before attempt reservation."""

    return {
        "source_authority_validated_before_hardware_query": True,
        "isolated_python": True,
        "bytecode_disabled": True,
        "hip_visible_devices": "0",
        "conflicting_accelerator_selectors_absent": True,
        "native_thread_selectors_equal_one": True,
        "visible_device_count": 1,
        "normalized_visible_device_name_contains": "r9700",
        "minimum_total_memory_bytes": 32_000_000_000,
        "tensor_allocation_count": 0,
        "payload_open_count": 0,
        "preflight_then_immediate_exec_without_intervening_gpu_query": True,
    }


def _validate_runtime_leaf(value: object, path: str) -> dict[str, Any]:
    binding = validate_binding(value, path=path)
    if (
        binding["file_sha256"] != RUNTIME_FILE_SHA256[path]
        or binding["content_sha256"] != RUNTIME_CONTENT_SHA256[path]
        or (
            path in RUNTIME_BYTE_COUNTS
            and binding["byte_count"] != RUNTIME_BYTE_COUNTS[path]
        )
    ):
        raise PermissionError(f"runtime binding changed: {path}")
    return binding


def validate_runtime_inputs(value: object) -> dict[str, Any]:
    """Validate only the frozen raw, N320, and schedule authority leaves."""

    if type(value) is not dict or set(value) != {"raw", "camera", "schedule"}:
        raise PermissionError("runtime input groups changed")
    raw, camera, schedule = value["raw"], value["camera"], value["schedule"]
    if (
        type(raw) is not dict
        or set(raw) != {"root", "manifest", "audit", "role_counts", "grant"}
        or raw["root"] != RAW_ROOT_RELATIVE_PATH
        or raw["role_counts"] != {
            "train": TRAIN_ROLE_COUNTS,
            "checkpoint_selection": SELECTION_ROLE_COUNTS,
        }
        or raw["grant"] != {
            "allowed_roles": ["train", "checkpoint_selection"],
            "allowed_operations": [
                "development_rgb_decode",
                "causal_temporal_perception_training",
                "physical_checkpoint_selection",
            ],
            "calibration_g2_navigation_heldout_or_production_use": False,
        }
    ):
        raise PermissionError("raw runtime authority changed")
    _validate_runtime_leaf(raw["manifest"], RAW_MANIFEST_RELATIVE_PATH)
    _validate_runtime_leaf(raw["audit"], RAW_AUDIT_RELATIVE_PATH)
    if (
        type(camera) is not dict
        or set(camera) != {
            "root",
            "gate",
            "checkpoint",
            "seed",
            "fit_size",
            "updates",
            "gate_must_pass_all_checks",
        }
        or camera["root"] != N320_ROOT_RELATIVE_PATH
        or camera["seed"] != 20_260_710
        or camera["fit_size"] != 320
        or camera["updates"] != 40_000
        or camera["gate_must_pass_all_checks"] != 26
    ):
        raise PermissionError("N320 runtime authority changed")
    _validate_runtime_leaf(camera["gate"], N320_GATE_RELATIVE_PATH)
    _validate_runtime_leaf(
        camera["checkpoint"], N320_CHECKPOINT_RELATIVE_PATH
    )
    _validate_runtime_leaf(schedule, SCHEDULE_RELATIVE_PATH)
    return {
        "raw": dict(raw),
        "camera": dict(camera),
        "schedule": dict(schedule),
    }


def science_contract() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_science_contract_v1",
        "preregistration": preregistration_binding(),
        "model_family": MODEL_FAMILY,
        "model_runtime_version": MODEL_RUNTIME_VERSION,
        "one_science_delta":
            "pure_visual_fixed_lag_token_difference_residual_only",
        "temporal_mechanism": {
            "formula": [
                "delta=current_tokens-previous_tokens",
                "conv1x1_192_to_8_no_bias",
                "group_norm_4_groups_8_channels_eps_1e-5_affine",
                "gelu_approximate_none",
                "depthwise_conv3x3_8_groups_8_padding_1_no_bias",
                "gelu_approximate_none",
                "zero_initialized_conv1x1_8_to_192_no_bias",
                "fused=current_tokens+history_valid*residual",
            ],
            "raw_tokens_only_in_history": True,
            "persistent_history_buffer_state": False,
            "state_prefix": TEMPORAL_STATE_PREFIX,
            "recurrent_accumulation": False,
            "inputs": list(MODEL_INPUTS),
            "forbidden_inputs": list(FORBIDDEN_MODEL_INPUTS),
            "lag_s": HISTORY_LAG_S,
            "tick_count": HISTORY_TICK_COUNT,
            "tick_s": HISTORY_TICK_S,
            "same_environment_episode_and_reset_required": True,
            "missing_irregular_or_reset_history_fails_cold": True,
        },
        "initialization": {
            "base_seed": BASE_INITIALIZATION_SEED,
            "decoder_local_cpu_seed": DECODER_INITIALIZATION_SEED,
            "temporal_local_cpu_seed": TEMPORAL_INITIALIZATION_SEED,
            "n320_only_tensor_input": True,
            "permitted_copies": ["encoder", "pixel_head", "ground_head"],
            "temporal_entry_copy_count": 0,
            "predecessor_dense_decoder_copy_count": 0,
            "temporal_final_projection_zero": True,
            "caller_cpu_rng_restored": True,
            "hard_sync_count": 1,
            "rejected_checkpoint_open_count": 0,
        },
        "data": {
            "train": dict(TRAIN_ROLE_COUNTS),
            "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            "probability_calibration": dict(CALIBRATION_ROLE_COUNTS),
            "probability_calibration_open_count": 0,
            "role_rebuild_refinement_or_reordering": False,
        },
        "evaluation": {
            "primary_population": "924_unique_endpoints",
            "primary_margin_count": MARGIN_COUNT,
            "scope_count": len(SCOPES),
            "scopes": list(SCOPES),
            "warm_only_view": "informational_only",
            "warm_only_may_control_checkpoint": False,
            "wrong_rgb_mapping": "existing_cyclic_within_family_complete_history",
        },
        "schedule": {
            "seed": SCHEDULE_SEED,
            "maximum_updates": MAXIMUM_UPDATE,
            "maximum_presentations": MAXIMUM_PRESENTATIONS,
            "checkpoints": list(CHECKPOINT_UPDATES),
            "microbatch_size": MICROBATCH_SIZE,
            "microbatches_per_update": MICROBATCHES_PER_UPDATE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "prefix_sha256": {
                str(update): digest
                for update, digest in CHECKPOINT_SCHEDULE_PREFIX_SHA256.items()
            },
        },
        "optimizer": {
            "name": "AdamW",
            "group_order": ["evidence_head", "encoder"],
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 1e-4,
            "amsgrad": False,
            "precision": "float32",
            "autocast": False,
            "encoder_learning_rate_scale": 1.0,
            "learning_rate_horizon_updates": 8_000,
            "independent_group_clip_norm": 1.0,
            "microbatch_size": MICROBATCH_SIZE,
            "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        },
        "parameter_counts": {
            **dict(EXPECTED_PARAMETER_COUNTS),
            "temporal": TEMPORAL_PARAMETER_COUNT,
            "total_trainable": EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT,
        },
        "parameter_tensor_counts": {
            **dict(EXPECTED_PARAMETER_TENSOR_COUNTS),
            "temporal": TEMPORAL_PARAMETER_TENSOR_COUNT,
            "total_trainable":
                EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT,
        },
        "checkpoints": {
            "100": "integrity_and_informational_only",
            "400": "integrity_and_informational_only",
            "1000": "strict_terminal_conjunction",
        },
        "pass_thresholds": dict(PASS_THRESHOLDS),
        "threshold_equality_passes": False,
        "maximum_attempts": 1,
        "calibration_authorized": False,
        "prior_runtime_output_open_count": 0,
        "jepa_objective_count": 0,
        "jepa_backward_count": 0,
        "ema_update_count_after_initial_hard_sync": 0,
        "authority": dict(SOURCE_ONLY_AUTHORITY),
    }


def lifecycle_contract() -> dict[str, Any]:
    return {
        "source_freeze_complete": source_freeze_status()["ready"],
        "source_preparation_may_reserve_attempt": False,
        "execution_requires_committed_frozen_source": True,
        "execution_requires_different_agent_source_review_pass": True,
        "execution_requires_exact_runtime_authorization": True,
        "execution_requires_fail_closed_preflight": True,
        "reservation_consumes_attempt": True,
        "maximum_attempts": 1,
        "retry_resume_repair_second_seed_extension_or_rerun": False,
        "train_and_checkpoint_selection_roles_only": True,
        "probability_calibration_open_count": 0,
        "prior_runtime_output_open_count": 0,
        "prohibited_runtime_output_roots":
            list(PROHIBITED_RUNTIME_OUTPUT_ROOTS),
        "jepa_training_authorized": False,
        "downstream_authority": dict(DOWNSTREAM_DENIALS),
    }


def operation_counts(
    update: int,
    evaluated_updates: Sequence[int],
) -> dict[str, Any]:
    if type(update) is not int or not 0 <= update <= MAXIMUM_UPDATE:
        raise ValueError("operation-count update is invalid")
    observed = tuple(evaluated_updates)
    if observed != CHECKPOINT_UPDATES[: len(observed)]:
        raise ValueError("evaluation updates are not one fixed prefix")
    if observed and observed[-1] > update:
        raise ValueError("evaluation occurred beyond completed training")
    return {
        "maximum_optimizer_updates": MAXIMUM_UPDATE,
        "complete_optimizer_updates": update,
        "maximum_pair_index_presentations": MAXIMUM_PRESENTATIONS,
        "pair_index_presentations": update * EFFECTIVE_BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        "camera_objective_count": update * MICROBATCHES_PER_UPDATE,
        "backward_call_count": update * MICROBATCHES_PER_UPDATE,
        "head_clip_invocation_count": update,
        "encoder_clip_invocation_count": update,
        "global_clip_invocation_count": 0,
        "optimizer_construction_count": 1 if update else 0,
        "checkpoint_selection_evaluation_count": len(observed),
        "checkpoint_selection_evaluation_updates": list(observed),
        "observer_evaluation_rerun_count": 0,
        "jepa_objective_count": 0,
        "jepa_backward_count": 0,
        "ema_update_count_after_initial_hard_sync": 0,
        "probability_calibration_open_count": 0,
        "prior_runtime_output_open_count": 0,
    }


def empty_partial_operation_counts() -> dict[str, Any]:
    return {
        "training_entered": False,
        **{name: 0 for name in PARTIAL_OPERATION_INTEGER_FIELDS},
        "checkpoint_selection_evaluation_updates_attempted": [],
        "checkpoint_selection_evaluation_updates_completed": [],
    }


def validate_partial_operation_counts(value: object) -> dict[str, Any]:
    fields = {
        "training_entered",
        *PARTIAL_OPERATION_INTEGER_FIELDS,
        "checkpoint_selection_evaluation_updates_attempted",
        "checkpoint_selection_evaluation_updates_completed",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("partial operation-count fields changed")
    if type(value["training_entered"]) is not bool or any(
        type(value[name]) is not int or value[name] < 0
        for name in PARTIAL_OPERATION_INTEGER_FIELDS
    ):
        raise PermissionError("partial operation count is invalid")
    attempted = value["checkpoint_selection_evaluation_updates_attempted"]
    completed = value["checkpoint_selection_evaluation_updates_completed"]
    if (
        type(attempted) is not list
        or type(completed) is not list
        or attempted != list(CHECKPOINT_UPDATES[: len(attempted)])
        or completed != list(CHECKPOINT_UPDATES[: len(completed)])
        or completed != attempted[: len(completed)]
        or value["optimizer_construction_attempt_count"] > 1
        or value["optimizer_construction_completion_count"]
        > value["optimizer_construction_attempt_count"]
        or value["optimizer_update_attempt_count"] > MAXIMUM_UPDATE
        or value["complete_optimizer_updates"]
        > value["optimizer_update_attempt_count"]
        or value["pair_index_presentations_attempted"] > MAXIMUM_PRESENTATIONS
        or value["pair_index_presentations_materialized"]
        > value["pair_index_presentations_attempted"]
        or value["microbatch_attempt_count"]
        > MAXIMUM_UPDATE * MICROBATCHES_PER_UPDATE
        or value["microbatch_completion_count"]
        > value["microbatch_attempt_count"]
        or value["camera_objective_attempt_count"]
        > MAXIMUM_UPDATE * MICROBATCHES_PER_UPDATE
        or value["camera_objective_completion_count"]
        > value["camera_objective_attempt_count"]
        or value["finite_camera_objective_count"]
        > value["camera_objective_completion_count"]
        or value["backward_attempt_count"]
        > MAXIMUM_UPDATE * MICROBATCHES_PER_UPDATE
        or value["backward_completion_count"]
        > value["backward_attempt_count"]
        or value["head_clip_attempt_count"] > MAXIMUM_UPDATE
        or value["head_clip_completion_count"]
        > value["head_clip_attempt_count"]
        or value["encoder_clip_attempt_count"] > MAXIMUM_UPDATE
        or value["encoder_clip_completion_count"]
        > value["encoder_clip_attempt_count"]
        or value["optimizer_step_attempt_count"] > MAXIMUM_UPDATE
        or value["optimizer_step_completion_count"]
        > value["optimizer_step_attempt_count"]
        or value["complete_optimizer_updates"]
        != value["optimizer_step_completion_count"]
        or value["checkpoint_snapshot_completion_count"]
        > len(CHECKPOINT_UPDATES)
        or value["checkpoint_selection_evaluation_attempt_count"]
        != len(attempted)
        or value["checkpoint_selection_evaluation_completion_count"]
        != len(completed)
        or value["metric_sidecar_publication_count"]
        > value["checkpoint_selection_evaluation_completion_count"]
        or any(
            value[name] != 0
            for name in (
                "observer_evaluation_rerun_count",
                "jepa_objective_count",
                "jepa_backward_count",
                "ema_update_count_after_initial_hard_sync",
                "global_clip_invocation_count",
                "probability_calibration_open_count",
                "prior_runtime_output_open_count",
            )
        )
    ):
        raise PermissionError("partial operation-count relationship changed")
    return {
        name: list(item) if isinstance(item := value[name], list) else item
        for name in fields
    }


def learning_rates(update: int) -> tuple[float, float]:
    if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATE:
        raise ValueError("probe update must lie in [1,1000]")
    if update <= 400:
        head = 1e-6 + (1e-4 - 1e-6) * (update - 1) / 399
    else:
        head = 1e-5 + 0.5 * (1e-4 - 1e-5) * (
            1.0 + math.cos(math.pi * (update - 400) / 7600)
        )
    if not math.isfinite(head) or head <= 0.0:
        raise ValueError("probe learning rate is invalid")
    return head, head


def validate_checkpoint_prefix(updates: Sequence[int]) -> tuple[int, ...]:
    result = tuple(updates)
    if not result or result != CHECKPOINT_UPDATES[: len(result)]:
        raise ValueError("checkpoint updates must be one nonempty fixed prefix")
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _lower(value: object, threshold: float, *, name: str) -> float:
    return (_finite(value, name=name) - threshold) / max(threshold, 1e-12)


def _upper(value: object, threshold: float, *, name: str) -> float:
    return (threshold - _finite(value, name=name)) / max(threshold, 1e-12)


def physical_margins(metrics: Mapping[str, Any]) -> list[float]:
    if type(metrics) is not dict:
        raise TypeError("physical metrics must be a plain dict")
    margins = [
        _lower(metrics.get(name), threshold, name=name)
        for name, threshold in PHYSICAL_LOWER_THRESHOLDS.items()
    ]
    margins.extend(
        _upper(metrics.get(name), threshold, name=name)
        for name, threshold in PHYSICAL_UPPER_THRESHOLDS.items()
    )
    distance = metrics.get("distance_group_balanced_accuracy")
    recalls = metrics.get("present_class_recall")
    if (
        not isinstance(distance, Sequence)
        or isinstance(distance, (str, bytes))
        or not distance
        or type(recalls) is not dict
        or not recalls
    ):
        raise ValueError("physical metric groups are empty")
    margins.extend(
        _lower(value, 0.92, name="distance group") for value in distance
    )
    margins.extend(
        _lower(value, 0.95, name=f"{name} recall")
        for name, value in sorted(recalls.items())
    )
    return margins


def evaluate_physical_scopes(
    scopes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if type(scopes) is not dict or tuple(scopes) != SCOPES:
        raise ValueError("physical scope order changed")
    rows: dict[str, Any] = {}
    flat: list[float] = []
    complete = 0
    for scope in SCOPES:
        margins = physical_margins(scopes[scope])
        passed = all(value >= 0.0 for value in margins)
        rows[scope] = {"physical_margins": margins, "passes": passed}
        flat.extend(margins)
        complete += int(passed)
    if len(flat) != MARGIN_COUNT:
        raise PermissionError(
            f"physical evaluator produced {len(flat)} rather than 189 margins"
        )
    rough = scopes["rough_local_dynamics"]
    return {
        "scope_evaluations": rows,
        "complete_physical_scope_count": complete,
        "margin_count": len(flat),
        "passed_margin_count": sum(value >= 0.0 for value in flat),
        "total_shortfall": sum(max(0.0, -value) for value in flat),
        "worst_margin": min(flat),
        "rough_motion": {
            "pixel_balanced_accuracy": _finite(
                rough["pixel_first_hit_balanced_accuracy"],
                name="rough pixel balanced accuracy",
            ),
            "ground_balanced_accuracy": _finite(
                rough["ground_clear_balanced_accuracy"],
                name="rough ground balanced accuracy",
            ),
            "depth_p95_m": _finite(
                rough["depth_p95_error_m"],
                name="rough depth p95",
            ),
        },
    }


def checkpoint_control_decision(
    *,
    update: int,
    evaluation: Mapping[str, Any],
    integrity_pass: bool,
) -> dict[str, Any]:
    if type(update) is not int or update not in CHECKPOINT_UPDATES:
        raise ValueError("control update is not fixed")
    if type(integrity_pass) is not bool:
        raise TypeError("integrity decision must be Boolean")
    required = {
        "complete_physical_scope_count",
        "margin_count",
        "passed_margin_count",
        "total_shortfall",
        "worst_margin",
        "rough_motion",
    }
    if type(evaluation) is not dict or not required.issubset(evaluation):
        raise ValueError("checkpoint evaluation summary changed")
    if evaluation["margin_count"] != MARGIN_COUNT:
        raise ValueError("checkpoint margin count changed")
    rough = evaluation["rough_motion"]
    if type(rough) is not dict or set(rough) != {
        "pixel_balanced_accuracy",
        "ground_balanced_accuracy",
        "depth_p95_m",
    }:
        raise ValueError("rough-motion summary changed")
    complete = evaluation["complete_physical_scope_count"]
    passed = evaluation["passed_margin_count"]
    if (
        type(complete) is not int
        or not 0 <= complete <= len(SCOPES)
        or type(passed) is not int
        or not 0 <= passed <= MARGIN_COUNT
    ):
        raise ValueError("checkpoint counts are invalid")
    shortfall = _finite(evaluation["total_shortfall"], name="total shortfall")
    worst = _finite(evaluation["worst_margin"], name="worst margin")
    pixel = _finite(
        rough["pixel_balanced_accuracy"],
        name="rough pixel balanced accuracy",
    )
    ground = _finite(
        rough["ground_balanced_accuracy"],
        name="rough ground balanced accuracy",
    )
    depth = _finite(rough["depth_p95_m"], name="rough depth p95")
    if shortfall < 0.0:
        raise ValueError("total shortfall is negative")
    conjuncts = {
        "complete_physical_scope_count_at_least_1": complete >= 1,
        "passed_margin_count_at_least_98": passed >= 98,
        "total_shortfall_strictly_below_threshold":
            shortfall < PASS_THRESHOLDS[
                "total_shortfall_strictly_less_than"
            ],
        "rough_pixel_balanced_accuracy_strictly_above_threshold":
            pixel > PASS_THRESHOLDS[
                "rough_pixel_balanced_accuracy_strictly_greater_than"
            ],
        "rough_ground_balanced_accuracy_strictly_above_threshold":
            ground > PASS_THRESHOLDS[
                "rough_ground_balanced_accuracy_strictly_greater_than"
            ],
        "rough_depth_p95_strictly_below_threshold":
            depth < PASS_THRESHOLDS[
                "rough_depth_p95_m_strictly_less_than"
            ],
    }
    if not integrity_pass:
        action = CONTROL_INTEGRITY_FAIL
    elif update in (100, 400):
        action = CONTROL_CONTINUE
    elif all(conjuncts.values()):
        action = CONTROL_PASS
    else:
        action = CONTROL_FAIL
    return {
        "schema": f"{SCHEMA_PREFIX}_checkpoint_control_v1",
        "update": update,
        "integrity_pass": integrity_pass,
        "informational_only": update in (100, 400),
        "threshold_equality_passes": False,
        "conjuncts": conjuncts,
        "statistics": {
            "complete_physical_scope_count": complete,
            "margin_count": MARGIN_COUNT,
            "passed_margin_count": passed,
            "total_shortfall": shortfall,
            "worst_margin": worst,
            "rough_motion": {
                "pixel_balanced_accuracy": pixel,
                "ground_balanced_accuracy": ground,
                "depth_p95_m": depth,
            },
        },
        "action": action,
        "qualifies_probe": action == CONTROL_PASS,
        "terminal": action != CONTROL_CONTINUE,
        "next_update": {100: 400, 400: 1_000}.get(update)
        if action == CONTROL_CONTINUE
        else None,
        "retry_authorized": False,
        "downstream_authority": dict(DOWNSTREAM_DENIALS),
    }


def metric_sidecar_relative_path(update: int) -> str:
    if update not in CHECKPOINT_UPDATES:
        raise ValueError("metric-sidecar update is not fixed")
    return f"checkpoints/update_{update}.metrics.json"


def parameter_partition(name: str) -> str:
    prefixes = (*TRAINABLE_PARAMETER_PREFIXES, *FROZEN_STATE_PREFIXES)
    matches = [
        prefix.removesuffix(".") for prefix in prefixes
        if name.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"model state escaped the fixed partition: {name}")
    return matches[0]


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "implementation_author",
        "reviewer",
        "reviewed_sources",
        "preregistration",
        "frozen_source_bindings",
        "science_contract",
        "lifecycle_contract",
        "source_only",
        "deferred_runtime_inputs_opened",
        "findings",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("source-review fields changed")
    if type(expected_sources) is not dict:
        raise PermissionError("expected source bindings are not a plain dict")
    required = {*SOURCE_PATHS, *SOURCE_REVIEW_ADDITIONAL_PATHS}
    if not required.issubset(expected_sources):
        raise PermissionError("expected source bindings are incomplete")
    for relative, digest in expected_sources.items():
        path = safe_relative_path(relative, name="reviewed source path")
        parts = PurePosixPath(path).parts
        if (
            not is_sha256(digest)
            or any(
                part in {".generated", "sealed"}
                or part.startswith("sealed_")
                for part in parts
            )
            or path.endswith("sealed_test.json")
            or (
                path not in SOURCE_REVIEW_ADDITIONAL_PATHS
                and not path.endswith(".py")
            )
        ):
            raise PermissionError("reviewed source binding is unsafe")
    core = dict(value)
    declared = core.pop("content_sha256")
    reviewer = value["reviewer"]
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != "PASS_SOURCE_ONLY"
        or value["implementation_author"] != CONTRACT_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == CONTRACT_AUTHOR
        or value["reviewed_sources"] != dict(expected_sources)
        or value["preregistration"] != preregistration_binding()
        or value["frozen_source_bindings"] != FROZEN_SOURCE_SHA256
        or value["science_contract"] != science_contract()
        or value["lifecycle_contract"] != lifecycle_contract()
        or value["source_only"] is not True
        or value["deferred_runtime_inputs_opened"] != []
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("source review did not pass these exact sources")
    return dict(value)


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
        "hardware",
        "experiment",
        "lifecycle",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("execution-authorization fields changed")
    review = validate_binding(
        review_binding,
        path=REVIEW_RELATIVE_PATH,
    )
    core = dict(value)
    declared = core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == CONTRACT_AUTHOR
        or value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != "AUTHORIZED_ONE_EXACT_BOUNDED_PROBE"
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {CONTRACT_AUTHOR, reviewer}
        or value["independent_source_review"] != review
        or value["preregistration"] != preregistration_binding()
        or validate_runtime_inputs(value["runtime_inputs"])
        != value["runtime_inputs"]
        or value["hardware"] != hardware_contract()
        or value["experiment"] != science_contract()
        or value["lifecycle"] != lifecycle_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("execution authorization changed")
    return dict(value)


def _validated_error(
    value: object,
    *,
    allow_none: bool,
) -> dict[str, str] | None:
    if value is None and allow_none:
        return None
    if (
        type(value) is not dict
        or set(value) != {"type", "message", "message_sha256"}
        or type(value["type"]) is not str
        or not value["type"]
        or type(value["message"]) is not str
        or not is_sha256(value["message_sha256"])
        or hashlib.sha256(value["message"].encode("utf-8")).hexdigest()
        != value["message_sha256"]
    ):
        raise PermissionError("error evidence changed")
    return dict(value)


def _validated_progress_stage(value: object) -> dict[str, Any]:
    fields = {"name", "update", "microbatch", "checkpoint_update", "role"}
    if (
        type(value) is not dict
        or set(value) != fields
        or type(value["name"]) is not str
        or not value["name"]
        or any(
            item is not None
            and (type(item) is not int or item < 0)
            for item in (
                value["update"],
                value["microbatch"],
                value["checkpoint_update"],
            )
        )
        or (
            value["role"] is not None
            and (
                type(value["role"]) is not str
                or value["role"] not in {
                    "authority",
                    "index",
                    "train",
                    "checkpoint_selection",
                }
            )
        )
    ):
        raise PermissionError("failure stage changed")
    return dict(value)


def _validated_expected_open_binding(value: object) -> dict[str, Any]:
    if (
        type(value) is not dict
        or set(value)
        != {"path", "file_sha256", "content_sha256", "byte_count"}
        or not is_sha256(value["file_sha256"])
        or (
            value["content_sha256"] is not None
            and not is_sha256(value["content_sha256"])
        )
        or (
            value["byte_count"] is not None
            and (
                type(value["byte_count"]) is not int
                or value["byte_count"] <= 0
            )
        )
    ):
        raise PermissionError("attempted runtime binding changed")
    path = safe_relative_path(value["path"], name="attempted runtime input")
    fixed = {
        RAW_MANIFEST_RELATIVE_PATH,
        RAW_AUDIT_RELATIVE_PATH,
        N320_GATE_RELATIVE_PATH,
        N320_CHECKPOINT_RELATIVE_PATH,
        SCHEDULE_RELATIVE_PATH,
    }
    if (
        path not in fixed
        and not path.startswith(RAW_ROOT_RELATIVE_PATH + "/")
    ):
        raise PermissionError("attempted input escaped frozen runtime roots")
    if any(
        path == root or path.startswith(root + "/")
        for root in PROHIBITED_RUNTIME_OUTPUT_ROOTS
    ):
        raise PermissionError("prior runtime output was attempted")
    parts = PurePosixPath(path).parts
    if (
        path.endswith("sealed_test.json")
        or any(part == "sealed" or part.startswith("sealed_") for part in parts)
    ):
        raise PermissionError("protected input was attempted")
    return dict(value)


def _validate_open_outcome(
    *,
    expected: Mapping[str, Any],
    outcome: object,
    descriptor_opened: object,
    read_completed: object,
    binding_accepted: object,
    observed_binding: object,
    partial_byte_count: object,
    error: object,
) -> None:
    if (
        outcome not in {
            "ACCEPTED",
            "REJECTED_BINDING",
            "OPEN_FAILED",
            "READ_FAILED",
        }
        or type(descriptor_opened) is not bool
        or type(read_completed) is not bool
        or type(binding_accepted) is not bool
        or type(partial_byte_count) is not int
        or partial_byte_count < 0
        or (read_completed and not descriptor_opened)
    ):
        raise PermissionError("runtime open outcome changed")
    observed: dict[str, Any] | None
    if observed_binding is None:
        observed = None
    elif (
        type(observed_binding) is not dict
        or set(observed_binding) != {"path", "file_sha256", "byte_count"}
        or observed_binding["path"] != expected["path"]
        or not is_sha256(observed_binding["file_sha256"])
        or type(observed_binding["byte_count"]) is not int
        or observed_binding["byte_count"] < 0
    ):
        raise PermissionError("observed runtime binding changed")
    else:
        observed = dict(observed_binding)
    if read_completed and (
        observed is None
        or observed["byte_count"] != partial_byte_count
    ):
        raise PermissionError("completed runtime read evidence changed")
    if outcome == "ACCEPTED":
        if (
            not descriptor_opened
            or not read_completed
            or not binding_accepted
            or observed is None
            or error is not None
            or observed["file_sha256"] != expected["file_sha256"]
            or (
                expected["byte_count"] is not None
                and observed["byte_count"] != expected["byte_count"]
            )
        ):
            raise PermissionError("accepted runtime open evidence changed")
    else:
        _validated_error(error, allow_none=False)
        if binding_accepted:
            raise PermissionError("failed runtime open was accepted")
        if outcome == "OPEN_FAILED" and (
            descriptor_opened
            or read_completed
            or observed is not None
            or partial_byte_count != 0
        ):
            raise PermissionError("open-failure evidence changed")
        if outcome == "READ_FAILED" and (
            not descriptor_opened or read_completed or observed is not None
        ):
            raise PermissionError("read-failure evidence changed")
        if outcome == "REJECTED_BINDING" and (
            not descriptor_opened
            or not read_completed
            or observed is None
        ):
            raise PermissionError("binding-rejection evidence changed")


def parse_partial_access_ledger(raw: bytes) -> list[dict[str, Any]]:
    """Validate the complete canonical, self-hashed runtime-open ledger."""

    if type(raw) is not bytes or not raw or not raw.endswith(b"\n"):
        raise PermissionError("partial-access ledger is not newline terminated")
    records: list[dict[str, Any]] = []
    previous: str | None = None
    attempts: dict[int, dict[str, Any]] = {}
    outcomes: dict[int, dict[str, Any]] = {}
    pending_open_id: int | None = None
    terminal_seen = False
    common = {
        "schema",
        "sequence",
        "previous_record_content_sha256",
        "record_type",
        "content_sha256",
    }
    type_fields = {
        "LEDGER_OPENED": {"attempt_identity", "reservation"},
        "OPEN_ATTEMPTED": {
            "open_id",
            "stage",
            "kind",
            "role",
            "purpose",
            "expected_binding",
        },
        "OPEN_OUTCOME": {
            "open_id",
            "stage",
            "kind",
            "outcome",
            "descriptor_opened",
            "read_completed",
            "binding_accepted",
            "observed_binding",
            "partial_byte_count",
            "error",
        },
        "ATTEMPT_TERMINATING": {"stage", "operation_counts", "error"},
        "RUNTIME_INPUT_ACCESS_FINALIZED": {
            "stage",
            "operation_counts",
            "error",
        },
    }
    for sequence, line in enumerate(raw.splitlines()):
        line_raw = line + b"\n"
        record = parse_canonical_json(
            line_raw,
            name=f"partial-access ledger record {sequence}",
        )
        record_type = record.get("record_type")
        declared = record.get("content_sha256")
        core = dict(record)
        core.pop("content_sha256", None)
        if (
            canonical_json_bytes(record) + b"\n" != line_raw
            or record_type not in type_fields
            or set(record) != common | type_fields[record_type]
            or record["schema"] != PARTIAL_ACCESS_RECORD_SCHEMA
            or type(record["sequence"]) is not int
            or record["sequence"] != sequence
            or record["previous_record_content_sha256"] != previous
            or not is_sha256(declared)
            or canonical_json_sha256(core) != declared
            or terminal_seen
        ):
            raise PermissionError("partial-access ledger chain changed")
        if record_type == "LEDGER_OPENED":
            if (
                sequence != 0
                or not is_sha256(record["attempt_identity"])
                or pending_open_id is not None
            ):
                raise PermissionError("partial-access ledger header changed")
            validate_binding(record["reservation"], path="reservation.json")
        elif record_type == "OPEN_ATTEMPTED":
            open_id = record["open_id"]
            if (
                type(open_id) is not int
                or open_id != len(attempts) + 1
                or pending_open_id is not None
                or type(record["stage"]) is not str
                or not record["stage"]
                or type(record["kind"]) is not str
                or not record["kind"]
                or (
                    record["role"] is not None
                    and (
                        type(record["role"]) is not str
                        or record["role"] not in {
                            "authority",
                            "index",
                            "train",
                            "checkpoint_selection",
                        }
                    )
                )
                or type(record["purpose"]) is not str
                or record["purpose"] not in {"runtime_load", "terminal_rehash"}
            ):
                raise PermissionError("partial-access attempt changed")
            _validated_expected_open_binding(record["expected_binding"])
            attempts[open_id] = record
            pending_open_id = open_id
        elif record_type == "OPEN_OUTCOME":
            open_id = record["open_id"]
            if (
                type(open_id) is not int
                or open_id != pending_open_id
                or open_id not in attempts
                or open_id in outcomes
                or record["stage"] != attempts[open_id]["stage"]
                or record["kind"] != attempts[open_id]["kind"]
            ):
                raise PermissionError("partial-access outcome changed")
            _validate_open_outcome(
                expected=attempts[open_id]["expected_binding"],
                outcome=record["outcome"],
                descriptor_opened=record["descriptor_opened"],
                read_completed=record["read_completed"],
                binding_accepted=record["binding_accepted"],
                observed_binding=record["observed_binding"],
                partial_byte_count=record["partial_byte_count"],
                error=record["error"],
            )
            outcomes[open_id] = record
            pending_open_id = None
        else:
            if pending_open_id is not None:
                raise PermissionError("partial-access ledger has an unpaired open")
            _validated_progress_stage(record["stage"])
            validate_partial_operation_counts(record["operation_counts"])
            if record_type == "ATTEMPT_TERMINATING":
                _validated_error(record["error"], allow_none=False)
            elif record["error"] is not None or any(
                not item["binding_accepted"] for item in outcomes.values()
            ):
                raise PermissionError("finalized runtime-input ledger changed")
            terminal_seen = True
        previous = str(declared)
        records.append(record)
    if (
        not records
        or records[0]["record_type"] != "LEDGER_OPENED"
        or set(attempts) != set(outcomes)
        or pending_open_id is not None
        or not terminal_seen
        or records[-1]["record_type"]
        not in {"ATTEMPT_TERMINATING", "RUNTIME_INPUT_ACCESS_FINALIZED"}
    ):
        raise PermissionError("partial-access ledger is incomplete")
    return records


def validate_pre_ledger_header(
    raw: bytes,
    *,
    reservation_binding: Mapping[str, Any],
    attempt_identity: str,
) -> dict[str, Any]:
    """Validate one durable header without calling it a complete ledger."""

    reservation = validate_binding(
        reservation_binding,
        path="reservation.json",
    )
    if not is_sha256(attempt_identity):
        raise PermissionError("pre-ledger attempt identity changed")
    record = parse_canonical_json(raw, name="pre-ledger durable header")
    core = dict(record)
    declared = core.pop("content_sha256", None)
    if (
        canonical_json_bytes(record) + b"\n" != raw
        or set(record)
        != {
            "schema",
            "sequence",
            "previous_record_content_sha256",
            "record_type",
            "attempt_identity",
            "reservation",
            "content_sha256",
        }
        or record["schema"] != PARTIAL_ACCESS_RECORD_SCHEMA
        or type(record["sequence"]) is not int
        or record["sequence"] != 0
        or record["previous_record_content_sha256"] is not None
        or record["record_type"] != "LEDGER_OPENED"
        or record["attempt_identity"] != attempt_identity
        or record["reservation"] != reservation
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("pre-ledger durable header changed")
    return {
        "path": "partial_access.jsonl",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "record_content_sha256": declared,
        "record_count": 1,
        "header_canonical_and_self_hashed": True,
        "constructor_accepted": False,
        "complete_ledger": False,
    }


def validate_pre_ledger_failure_receipt(
    value: object,
    *,
    reservation_binding: Mapping[str, Any] | None = None,
    attempt_identity: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "attempt_identity",
        "reservation",
        "ledger_state",
        "failure_stage",
        "operation_counts",
        "published_prefix",
        "published_prefix_sha256",
        "directories_including_root",
        "error",
        "scientific_result",
        "scientific_result_status",
        "retry_authorized",
        "g2_navigation_or_heldout_attempted",
        "prior_runtime_output_open_count",
        "authority",
        "terminalization",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("temporal pre-ledger failure fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    reservation = validate_binding(value["reservation"], path="reservation.json")
    ledger_state = value["ledger_state"]
    stage = value["failure_stage"]
    prefix = value["published_prefix"]
    if type(ledger_state) is not dict or set(ledger_state) != {
        "status",
        "header",
        "header_prefix",
        "runtime_input_open_count",
        "standard_ledger_complete",
        "standard_failure_validator_applicable",
    }:
        raise PermissionError("temporal pre-ledger state changed")
    header_status = ledger_state["status"]
    header = ledger_state["header"]
    header_prefix = ledger_state["header_prefix"]
    header_fields = {
        "path",
        "file_sha256",
        "byte_count",
        "record_content_sha256",
        "record_count",
        "header_canonical_and_self_hashed",
        "constructor_accepted",
        "complete_ledger",
    }
    if header_status == "NOT_PUBLISHED":
        if header is not None or header_prefix is not None:
            raise PermissionError("absent pre-ledger header changed")
        expected_prefix_paths = ["reservation.json"]
        expected_boundary = "before_header_publication"
    elif header_status == "UNACCEPTED_HEADER_PREFIX":
        if (
            header is not None
            or type(header_prefix) is not dict
            or set(header_prefix)
            != {
                "path",
                "file_sha256",
                "byte_count",
                "matches_expected_header",
                "constructor_accepted",
                "complete_ledger",
            }
            or header_prefix["path"] != "partial_access.jsonl"
            or not is_sha256(header_prefix["file_sha256"])
            or type(header_prefix["byte_count"]) is not int
            or header_prefix["byte_count"] < 0
            or type(header_prefix["matches_expected_header"]) is not bool
            or header_prefix["constructor_accepted"] is not False
            or header_prefix["complete_ledger"] is not False
        ):
            raise PermissionError("unaccepted pre-ledger prefix changed")
        expected_prefix_paths = ["partial_access.jsonl", "reservation.json"]
        expected_boundary = "during_header_publication_unaccepted_prefix"
    elif header_status == "DURABLE_NOT_CONSTRUCTOR_ACCEPTED":
        if (
            header_prefix is not None
            or type(header) is not dict
            or set(header) != header_fields
            or header["path"] != "partial_access.jsonl"
            or not is_sha256(header["file_sha256"])
            or type(header["byte_count"]) is not int
            or header["byte_count"] <= 0
            or not is_sha256(header["record_content_sha256"])
            or header["record_count"] != 1
            or header["header_canonical_and_self_hashed"] is not True
            or header["constructor_accepted"] is not False
            or header["complete_ledger"] is not False
        ):
            raise PermissionError("durable pre-ledger header changed")
        expected_prefix_paths = ["partial_access.jsonl", "reservation.json"]
        expected_boundary = (
            "after_durable_header_before_constructor_acceptance"
        )
    else:
        raise PermissionError("temporal pre-ledger status changed")
    prefix_fields = {"path", "file_sha256", "byte_count", "mode"}
    if type(prefix) is not list or any(
        type(row) is not dict
        or set(row) != prefix_fields
        or not is_sha256(row["file_sha256"])
        or type(row["byte_count"]) is not int
        or row["byte_count"] < 0
        for row in prefix
    ):
        raise PermissionError("pre-ledger published prefix changed")
    prefix_index = {row["path"]: row for row in prefix}
    if (
        len(prefix_index) != len(prefix)
        or [row["path"] for row in prefix] != expected_prefix_paths
        or value["schema"] != PRE_LEDGER_FAILURE_SCHEMA
        or value["status"]
        != "TERMINAL_CAUSAL_TEMPORAL_V1_POST_RESERVATION_PRE_LEDGER_"
        "FAILURE_NO_RETRY"
        or not is_sha256(value["attempt_identity"])
        or (
            reservation_binding is not None
            and reservation != dict(reservation_binding)
        )
        or (
            attempt_identity is not None
            and value["attempt_identity"] != attempt_identity
        )
        or ledger_state["runtime_input_open_count"] != 0
        or ledger_state["standard_ledger_complete"] is not False
        or ledger_state["standard_failure_validator_applicable"] is not False
        or type(stage) is not dict
        or set(stage) != {"name", "boundary"}
        or stage["name"] != "partial_access_ledger_initialization"
        or stage["boundary"] != expected_boundary
        or validate_partial_operation_counts(value["operation_counts"])
        != empty_partial_operation_counts()
        or prefix_index["reservation.json"]["mode"] != "0644"
        or prefix_index["reservation.json"]["file_sha256"]
        != reservation["file_sha256"]
        or prefix_index["reservation.json"]["byte_count"]
        != reservation["byte_count"]
        or value["published_prefix_sha256"] != canonical_json_sha256(prefix)
        or value["directories_including_root"] != ["."]
        or _validated_error(value["error"], allow_none=False) is None
        or value["scientific_result"] is not None
        or value["scientific_result_status"]
        != "NOT_OBSERVED_TERMINAL_PRE_LEDGER_FAILURE"
        or value["retry_authorized"] is not False
        or value["g2_navigation_or_heldout_attempted"] is not False
        or value["prior_runtime_output_open_count"] != 0
        or value["authority"] != DOWNSTREAM_DENIALS
        or value["terminalization"]
        != {
            "failure_publication": "exclusive_atomic_fsync",
            "terminal_file_mode": "0444",
            "terminal_directory_mode": "0555",
            "seal_after_publication": True,
        }
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("temporal pre-ledger failure receipt changed")
    if header is not None and (
        prefix_index["partial_access.jsonl"]["file_sha256"]
        != header["file_sha256"]
        or prefix_index["partial_access.jsonl"]["byte_count"]
        != header["byte_count"]
        or prefix_index["partial_access.jsonl"]["mode"] != "0600"
    ):
        raise PermissionError("pre-ledger header prefix changed")
    if header_prefix is not None and (
        prefix_index["partial_access.jsonl"]["file_sha256"]
        != header_prefix["file_sha256"]
        or prefix_index["partial_access.jsonl"]["byte_count"]
        != header_prefix["byte_count"]
        or prefix_index["partial_access.jsonl"]["mode"] != "0600"
    ):
        raise PermissionError("unaccepted header-prefix binding changed")
    return dict(value)


def validate_failure_receipt(
    value: object,
    *,
    reservation_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "attempt_identity",
        "reservation",
        "partial_access_ledger",
        "runtime_opens",
        "runtime_opens_sha256",
        "failure_stage",
        "operation_counts",
        "published_prefix",
        "published_prefix_sha256",
        "directories_including_root",
        "error",
        "scientific_result",
        "scientific_result_status",
        "retry_authorized",
        "g2_navigation_or_heldout_attempted",
        "prior_runtime_output_open_count",
        "authority",
        "terminalization",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("temporal failure-receipt fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    reservation = validate_binding(value["reservation"], path="reservation.json")
    ledger = value["partial_access_ledger"]
    opens = value["runtime_opens"]
    prefix = value["published_prefix"]
    ledger_fields = {
        "path",
        "file_sha256",
        "byte_count",
        "records_content_sha256",
        "record_count",
        "last_record_content_sha256",
        "attempted_open_count",
        "descriptor_opened_count",
        "read_completed_count",
        "accepted_open_count",
        "rejected_or_failed_open_count",
    }
    open_fields = {
        "open_id",
        "stage",
        "kind",
        "role",
        "purpose",
        "expected_binding",
        "outcome",
        "descriptor_opened",
        "read_completed",
        "binding_accepted",
        "observed_binding",
        "partial_byte_count",
        "error",
    }
    if (
        type(ledger) is not dict
        or set(ledger) != ledger_fields
        or ledger["path"] != "partial_access.jsonl"
        or not all(
            is_sha256(ledger[name])
            for name in (
                "file_sha256",
                "records_content_sha256",
                "last_record_content_sha256",
            )
        )
        or type(ledger["byte_count"]) is not int
        or ledger["byte_count"] <= 0
        or type(ledger["record_count"]) is not int
        or ledger["record_count"] < 2
        or any(
            type(ledger[name]) is not int or ledger[name] < 0
            for name in (
                "attempted_open_count",
                "descriptor_opened_count",
                "read_completed_count",
                "accepted_open_count",
                "rejected_or_failed_open_count",
            )
        )
        or type(opens) is not list
        or ledger["attempted_open_count"] != len(opens)
    ):
        raise PermissionError("failure ledger summary changed")
    for index, row in enumerate(opens, start=1):
        if (
            type(row) is not dict
            or set(row) != open_fields
            or type(row["open_id"]) is not int
            or row["open_id"] != index
            or type(row["stage"]) is not str
            or not row["stage"]
            or type(row["kind"]) is not str
            or not row["kind"]
            or (
                row["role"] is not None
                and (
                    type(row["role"]) is not str
                    or row["role"] not in {
                        "authority",
                        "index",
                        "train",
                        "checkpoint_selection",
                    }
                )
            )
            or row["purpose"] not in {"runtime_load", "terminal_rehash"}
        ):
            raise PermissionError("failure runtime-open row changed")
        expected = _validated_expected_open_binding(row["expected_binding"])
        _validate_open_outcome(
            expected=expected,
            outcome=row["outcome"],
            descriptor_opened=row["descriptor_opened"],
            read_completed=row["read_completed"],
            binding_accepted=row["binding_accepted"],
            observed_binding=row["observed_binding"],
            partial_byte_count=row["partial_byte_count"],
            error=row["error"],
        )
    prefix_fields = {"path", "file_sha256", "byte_count", "mode"}
    if type(prefix) is not list or any(
        type(row) is not dict
        or set(row) != prefix_fields
        or safe_relative_path(row["path"], name="published prefix path")
        != row["path"]
        or not is_sha256(row["file_sha256"])
        or type(row["byte_count"]) is not int
        or row["byte_count"] < 0
        or row["mode"] not in {"0444", "0600", "0644"}
        for row in prefix
    ):
        raise PermissionError("failure published prefix changed")
    prefix_index = {row["path"]: row for row in prefix}
    directories = value["directories_including_root"]
    if (
        value["schema"] != FAILURE_SCHEMA
        or value["status"]
        != "TERMINAL_CAUSAL_TEMPORAL_V1_OPERATIONAL_OR_INTEGRITY_"
        "FAILURE_NO_RETRY"
        or not is_sha256(value["attempt_identity"])
        or (
            reservation_binding is not None
            and reservation != dict(reservation_binding)
        )
        or ledger["descriptor_opened_count"]
        != sum(row["descriptor_opened"] for row in opens)
        or ledger["read_completed_count"]
        != sum(row["read_completed"] for row in opens)
        or ledger["accepted_open_count"]
        != sum(row["binding_accepted"] for row in opens)
        or ledger["rejected_or_failed_open_count"]
        != sum(not row["binding_accepted"] for row in opens)
        or value["runtime_opens_sha256"] != canonical_json_sha256(opens)
        or _validated_progress_stage(value["failure_stage"])
        != value["failure_stage"]
        or validate_partial_operation_counts(value["operation_counts"])
        != value["operation_counts"]
        or len(prefix_index) != len(prefix)
        or [row["path"] for row in prefix] != sorted(prefix_index)
        or "reservation.json" not in prefix_index
        or "partial_access.jsonl" not in prefix_index
        or prefix_index["reservation.json"]["file_sha256"]
        != reservation["file_sha256"]
        or prefix_index["reservation.json"]["byte_count"]
        != reservation["byte_count"]
        or prefix_index["partial_access.jsonl"]["file_sha256"]
        != ledger["file_sha256"]
        or prefix_index["partial_access.jsonl"]["byte_count"]
        != ledger["byte_count"]
        or value["published_prefix_sha256"] != canonical_json_sha256(prefix)
        or type(directories) is not list
        or not directories
        or directories[0] != "."
        or directories[1:] != sorted(set(directories[1:]))
        or any(
            safe_relative_path(path, name="terminal directory") != path
            for path in directories[1:]
        )
        or _validated_error(value["error"], allow_none=False) is None
        or value["scientific_result"] is not None
        or value["scientific_result_status"]
        != "NOT_OBSERVED_TERMINAL_OPERATIONAL_OR_INTEGRITY_FAILURE"
        or value["retry_authorized"] is not False
        or value["g2_navigation_or_heldout_attempted"] is not False
        or value["prior_runtime_output_open_count"] != 0
        or value["authority"] != DOWNSTREAM_DENIALS
        or value["terminalization"]
        != {
            "failure_publication": "exclusive_atomic_fsync",
            "terminal_file_mode": "0444",
            "terminal_directory_mode": "0555",
            "seal_after_publication": True,
        }
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("temporal failure receipt changed")
    return dict(value)


def validate_metric_sidecar(
    value: object,
    *,
    update: int | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "update",
        "checkpoint",
        "metric",
        "inline_evaluation_count",
        "state_mutation_count",
        "publication_order",
        "continuation",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("metric-sidecar fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    observed_update = value["update"]
    checkpoint = value["checkpoint"]
    metric = value["metric"]
    checkpoint_fields = {
        "path",
        "file_sha256",
        "content_sha256",
        "byte_count",
        "state_sha256",
        "frozen_state_sha256",
    }
    metric_fields = {
        "update",
        "role",
        "pair_count",
        "unique_endpoint_count",
        "temporal_population",
        "scopes",
        "warm_scopes_informational_only",
        "aggregate_complete_v4_tail_depth_loss",
        "evaluation",
        "integrity_pass",
        "state_sha256_before",
        "state_sha256_after",
        "frozen_state_sha256_before_and_after",
        "state_mutation_count",
    }
    if (
        type(observed_update) is not int
        or observed_update not in CHECKPOINT_UPDATES
        or (update is not None and observed_update != update)
        or type(checkpoint) is not dict
        or set(checkpoint) != checkpoint_fields
        or checkpoint["path"] != f"checkpoints/update_{observed_update}.pt"
        or not all(
            is_sha256(checkpoint[name])
            for name in (
                "file_sha256",
                "content_sha256",
                "state_sha256",
                "frozen_state_sha256",
            )
        )
        or type(checkpoint["byte_count"]) is not int
        or checkpoint["byte_count"] <= 0
        or type(metric) is not dict
        or set(metric) != metric_fields
        or metric["update"] != observed_update
        or metric["role"] != "checkpoint_selection"
        or metric["pair_count"] != SELECTION_ROLE_COUNTS["pairs"]
        or metric["unique_endpoint_count"]
        != SELECTION_ROLE_COUNTS["unique_endpoints"]
        or type(metric["temporal_population"]) is not dict
        or type(metric["scopes"]) is not dict
        or tuple(metric["scopes"]) != SCOPES
        or type(metric["warm_scopes_informational_only"]) is not dict
        or tuple(metric["warm_scopes_informational_only"]) != SCOPES
        or not math.isfinite(
            _finite(
                metric["aggregate_complete_v4_tail_depth_loss"],
                name="metric camera loss",
            )
        )
        or metric["integrity_pass"] is not True
        or not all(
            is_sha256(metric[name])
            for name in (
                "state_sha256_before",
                "state_sha256_after",
                "frozen_state_sha256_before_and_after",
            )
        )
        or metric["state_sha256_before"] != metric["state_sha256_after"]
        or metric["state_mutation_count"] != 0
    ):
        raise PermissionError("metric-sidecar metric changed")
    continuation = checkpoint_control_decision(
        update=observed_update,
        evaluation=metric["evaluation"],
        integrity_pass=metric["integrity_pass"],
    )
    if (
        value["schema"] != METRIC_SIDECAR_SCHEMA
        or value["status"]
        != "PUBLISHED_0444_AFTER_INLINE_EVALUATION_BEFORE_CONTROL"
        or value["inline_evaluation_count"] != 1
        or value["state_mutation_count"] != 0
        or value["publication_order"]
        != [
            "cpu_snapshot",
            "inline_nonmutating_selection_evaluation",
            "atomic_mode_0444_sidecar",
            "control_branch",
        ]
        or value["continuation"] != continuation
        or value["authority"] != DOWNSTREAM_DENIALS
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("metric sidecar changed")
    return dict(value)


__all__ = [name for name in globals() if name.isupper()] + [
    "artifact_binding",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "checkpoint_control_decision",
    "current_source_bindings",
    "empty_partial_operation_counts",
    "evaluate_physical_scopes",
    "hardware_contract",
    "is_sha256",
    "learning_rates",
    "lifecycle_contract",
    "metric_sidecar_relative_path",
    "operation_counts",
    "parameter_partition",
    "parse_canonical_json",
    "parse_partial_access_ledger",
    "physical_margins",
    "preregistration_binding",
    "require_source_freeze",
    "safe_relative_path",
    "science_contract",
    "source_freeze_status",
    "validate_authorization",
    "validate_binding",
    "validate_checkpoint_prefix",
    "validate_failure_receipt",
    "validate_metric_sidecar",
    "validate_partial_operation_counts",
    "validate_pre_ledger_failure_receipt",
    "validate_pre_ledger_header",
    "validate_review",
    "validate_runtime_inputs",
    "validate_source_manifest",
    "with_content_sha256",
]
