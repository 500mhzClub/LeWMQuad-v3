"""Source-only contract for the science-identical V2 operational recovery.

This module intentionally imports only the Python standard library.  Importing
it does not inspect generated inputs, deserialize a tensor, or import Torch.
The separately reviewed runner owns the one fail-closed execution lifecycle.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/probe_lifecycle_impl"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_multires_probe_v2"

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_shared_jepa_v5_multires_probe_v2.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_shared_jepa_v5_multires_probe_v2.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_shared_jepa_v5_multires_probe_v2.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_shared_observable_camera_ray_jepa_v5_multires_v1.py"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "source_manifest_2026-07-24.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_multires_probe_source_closure_v2.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_multires_probe_source_closure_v2.py"
)
SCIENCE_IDENTITY_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_multires_probe_v2_science_identity.py"
)
SCIENCE_IDENTITY_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_multires_probe_v2_science_identity.py"
)
PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "preregistration_2026-07-24.json"
)
PREREGISTRATION_COMMIT = "5849dc497acd272d56026c00b821b3662b040752"
PREREGISTRATION_FILE_SHA256 = (
    "642897b82ccdee6ac6c23168754056335d7a3701a19ccfc682527872461f16cc"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "264a4e3d52dd0ec658afce8c4bc54f86e9c18bbfb43229c14521b5f683a6514a"
)
PREREGISTRATION_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "preregistration_independent_review_2026-07-24.json"
)
PREREGISTRATION_REVIEW_FILE_SHA256 = (
    "b8314774a707e1f8af8db214d0c12fe304352710b2ff4d569068b9c3d184bf84"
)
RECOVERY_DECISION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "operational_recovery_decision_2026-07-24.md"
)
RECOVERY_DECISION_FILE_SHA256 = (
    "9df833efb3949744e66cb5263d341baef69241d4b2b1653d90ca9bf87f8ec1fb"
)
SCHEDULE_ADAPTER_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py"
)
SCHEDULE_ADAPTER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_shared_jepa_v5_multires_probe_v2_schedule.py"
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

MODEL_FILE_SHA256 = (
    "a63da1137539953b2f40d184def1652ae05f63d7b434084b1a91787e1fc83d0b"
)
FROZEN_SOURCE_SHA256 = {
    PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
    PREREGISTRATION_REVIEW_RELATIVE_PATH:
        PREREGISTRATION_REVIEW_FILE_SHA256,
    RECOVERY_DECISION_RELATIVE_PATH: RECOVERY_DECISION_FILE_SHA256,
    SCHEDULE_ADAPTER_RELATIVE_PATH:
        "a8efe19da92c9c2107f11be38db8ed80e66aedca3ef41af0428ab13d50f56bd1",
    SCHEDULE_ADAPTER_TEST_RELATIVE_PATH:
        "340828cb55a03da575ccfb8242ff3e3db8b8f15527d43891b737cfad8a5b2204",
    MATCHED_V1_CONTRACT_RELATIVE_PATH:
        "53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a",
    MATCHED_V1_RUNNER_RELATIVE_PATH:
        "e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578",
    TAIL_DEPTH_LOSS_RELATIVE_PATH:
        "6fc0a114386ee2fb0ae98704a970d38a7194db192283b904138015498fb02384",
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py":
        "708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85",
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py":
        "6a0e40f9dcb496831553dc5bbc6d1efcdf6d82676d6f18aa20e417f8de4fa6a0",
    "lewm/models/observable_camera_ray_evidence_v4.py":
        "6238f7fb2b9c0c5201c9d7ebb5343ceef72fa97b423dddb466465b6c594cc882",
    "lewm/models/observable_camera_ray_evidence_v4_training.py":
        "c0f3f944883987950edb7579a9e108171486122a9a3ae9d84d2a1abb6ac015ed",
    "lewm/models/shared_observable_camera_ray_jepa_v5.py":
        "b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9",
    "lewm/models/shared_observable_camera_ray_jepa_v5_full_training_v4_loss.py":
        "8422c253c3eca3b34dd42b4f823dab4ac67f0e90fb2cff8eeaa67a1310b3c53a",
    "lewm/models/encoders.py":
        "5eed7bbe424d5ddd293ea67ed1596e74504c68dd8da93f8420795f216cb7599d",
    "lewm/models/egomotion_bev_jepa.py":
        "c4006e9804182b077399229d43bc8c9be64b5af12c81fff4076d5a78e6ef359b",
    "lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py":
        "52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd",
    "lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py":
        "735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662",
    MODEL_TEST_RELATIVE_PATH:
        "a241910c83bc44cf15b56270659becf1def66f358f3f2bb1a89d89a9bce30fae",
}
SOURCE_PATHS = tuple(dict.fromkeys((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    SCHEDULE_ADAPTER_RELATIVE_PATH,
    *FROZEN_SOURCE_SHA256,
)))
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    PREREGISTRATION_REVIEW_RELATIVE_PATH,
    RECOVERY_DECISION_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    SCHEDULE_ADAPTER_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
    SCIENCE_IDENTITY_CHECKER_RELATIVE_PATH,
    SCIENCE_IDENTITY_TEST_RELATIVE_PATH,
)

REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "source_review_2026-07-24.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "execution_authorization_2026-07-24.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v2"
)
V1_OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v1"
)

RAW_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
RAW_MANIFEST_RELATIVE_PATH = f"{RAW_ROOT_RELATIVE_PATH}/manifest.json"
RAW_AUDIT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1.audit_v13.json"
)
N320_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1"
)
N320_GATE_RELATIVE_PATH = f"{N320_ROOT_RELATIVE_PATH}/gate.json"
N320_CHECKPOINT_RELATIVE_PATH = f"{N320_ROOT_RELATIVE_PATH}/checkpoint.pt"
SCHEDULE_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "matched_training_v4/schedule.json"
)

RUNTIME_FILE_SHA256 = {
    RAW_MANIFEST_RELATIVE_PATH:
        "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360",
    RAW_AUDIT_RELATIVE_PATH:
        "0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76",
    N320_GATE_RELATIVE_PATH:
        "4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6",
    N320_CHECKPOINT_RELATIVE_PATH:
        "ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0",
    SCHEDULE_RELATIVE_PATH:
        "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270",
}
RUNTIME_CONTENT_SHA256 = {
    RAW_MANIFEST_RELATIVE_PATH:
        "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a",
    RAW_AUDIT_RELATIVE_PATH:
        "0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca",
    N320_GATE_RELATIVE_PATH:
        "76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b",
    N320_CHECKPOINT_RELATIVE_PATH:
        "9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b",
    SCHEDULE_RELATIVE_PATH:
        "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15",
}
RUNTIME_BYTE_COUNTS = {
    N320_GATE_RELATIVE_PATH: 7_960,
    N320_CHECKPOINT_RELATIVE_PATH: 13_777_100,
    SCHEDULE_RELATIVE_PATH: 607_373,
}

TRAIN_ROLE_COUNTS = {"pairs": 4_262, "unique_endpoints": 7_777, "scenes": 72}
SELECTION_ROLE_COUNTS = {
    "pairs": 495,
    "unique_endpoints": 924,
    "scenes": 8,
}
BASE_INITIALIZATION_SEED = 20260712
DECODER_INITIALIZATION_SEED = 20260724
SCHEDULE_SEED = 20260713
MAXIMUM_UPDATE = 1_000
CHECKPOINT_UPDATES = (100, 400, 1_000)
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
EFFECTIVE_BATCH_SIZE = 16
MAXIMUM_PRESENTATIONS = 16_000
MARGIN_COUNT = 189
MODEL_RUNTIME_VERSION = (
    "lewm_go2_rgb_multiresolution_perception_v1_model_runtime_v1"
)
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
EXPECTED_PARAMETER_COUNTS = {"evidence_head": 352_689, "encoder": 2_747_520}
EXPECTED_PARAMETER_TENSOR_COUNTS = {"evidence_head": 26, "encoder": 78}
POST_CLIP_NORM_ASSERTION_TOLERANCE = 1e-5

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

PHYSICAL_LOWER_THRESHOLDS = {
    "pixel_first_hit_balanced_accuracy": 0.95,
    "ground_clear_balanced_accuracy": 0.95,
    "derived_raster_balanced_accuracy": 0.95,
    "wrong_rgb_pixel_balanced_accuracy_drop": 0.12,
    "wrong_rgb_depth_median_error_increase_m": 0.12,
    "wrong_rgb_depth_p95_error_increase_m": 0.20,
    "wrong_rgb_ground_balanced_accuracy_drop": 0.12,
    "wrong_rgb_raster_nll_increase": 0.12,
    "wrong_rgb_raster_balanced_accuracy_drop": 0.12,
}
PHYSICAL_UPPER_THRESHOLDS = {
    "depth_median_error_m": 0.10,
    "depth_p95_error_m": 0.25,
    "derived_raster_nll": 0.15,
}
SCOPES = (
    "aggregate",
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)

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
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v2"
PARTIAL_ACCESS_RECORD_SCHEMA = f"{SCHEMA_PREFIX}_partial_access_record_v1"
PARTIAL_ACCESS_LEDGER_SCHEMA = f"{SCHEMA_PREFIX}_partial_access_ledger_v1"

DOWNSTREAM_DENIALS = {
    "probe_checkpoint_qualified": False,
    "perception_qualification_authorized": False,
    "jepa_training_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "retry_resume_recovery_second_seed_or_extension_authorized": False,
}
REVIEW_AUTHORITY = {
    "execution_authorized": False,
    "gpu_authorized": False,
    "generated_input_open_authorized": False,
    "checkpoint_open_authorized": False,
    "dataset_or_rgb_open_authorized": False,
    "output_mutation_authorized": False,
    **DOWNSTREAM_DENIALS,
}
EXECUTION_AUTHORITY = {
    "one_exact_probe_attempt_authorized": True,
    "one_discrete_r9700_authorized": True,
    "n320_initialization_only_authorized": True,
    "train_and_checkpoint_selection_roles_only_authorized": True,
    "generated_mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
    "output_root_must_be_absent": True,
    **DOWNSTREAM_DENIALS,
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    if type(core) is not dict or "content_sha256" in core:
        raise TypeError("self-hashed core must be a plain dict without its hash")
    return {**core, "content_sha256": canonical_json_sha256(core)}


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(f"{name} must be exactly one canonical JSON line")
    try:
        value = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not ASCII JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"{name} is not canonical")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} self hash changed")
    return value


def safe_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{name} is not a safe relative path")
    return value


def artifact_binding(
    path: str,
    raw: bytes,
    *,
    content_sha256: str,
) -> dict[str, Any]:
    safe_relative_path(path, name="artifact path")
    if not is_sha256(content_sha256):
        raise ValueError("artifact content hash is malformed")
    return {
        "path": path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256,
        "byte_count": len(raw),
    }


def validate_binding(
    value: object,
    *,
    path: str | None = None,
) -> dict[str, Any]:
    fields = {"path", "file_sha256", "content_sha256", "byte_count"}
    if type(value) is not dict or set(value) != fields:
        raise ValueError("artifact binding fields changed")
    safe_relative_path(value["path"], name="artifact path")
    if (
        (path is not None and value["path"] != path)
        or not is_sha256(value["file_sha256"])
        or not is_sha256(value["content_sha256"])
        or type(value["byte_count"]) is not int
        or value["byte_count"] <= 0
    ):
        raise ValueError("artifact binding changed")
    return dict(value)


def preregistration_binding() -> dict[str, str]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "content_sha256": PREREGISTRATION_CONTENT_SHA256,
    }


def _read_regular_source(path: Path) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError("O_NOFOLLOW is required for source custody")
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
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
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
        raise PermissionError("source manifest is not finite duplicate-safe JSON") from error
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
        value["schema"]
        != "lewm_go2_rgb_multiresolution_perception_v2_source_manifest"
        or value["status"] != "SOURCE_ONLY_RECURSIVE_CLOSURE"
        or value["date"] != "2026-07-24"
        or value["authority"]
        != "source_closure_only_no_generated_input_checkpoint_training_gpu_"
        "qualification_g2_navigation_heldout_production_or_promotion_authority"
        or value["entrypoints"] != [
            LAUNCHER_RELATIVE_PATH,
            RUNNER_RELATIVE_PATH,
        ]
        or value["forced_dynamic_sources"] != [
            CONTRACT_RELATIVE_PATH,
            MATCHED_V1_CONTRACT_RELATIVE_PATH,
            MATCHED_V1_RUNNER_RELATIVE_PATH,
            SCHEDULE_ADAPTER_RELATIVE_PATH,
        ]
        or not isinstance(value["excluded_runtime_categories"], list)
        or not value["excluded_runtime_categories"]
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
    ):
        raise PermissionError("source-manifest path order or uniqueness changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    if not is_sha256(MODEL_FILE_SHA256):
        raise PermissionError("multires model source is not frozen")
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    expected = {**FROZEN_SOURCE_SHA256, MODEL_RELATIVE_PATH: MODEL_FILE_SHA256}
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        relative = binding["path"]
        raw = _read_regular_source(root / relative)
        digest = hashlib.sha256(raw).hexdigest()
        if (
            digest != binding["file_sha256"]
            or len(raw) != binding["byte_count"]
        ):
            raise PermissionError(f"manifest-bound source changed: {relative}")
        result[relative] = digest
    for relative in SOURCE_REVIEW_ADDITIONAL_PATHS:
        if relative in result:
            continue
        raw = (
            manifest_raw
            if relative == SOURCE_MANIFEST_RELATIVE_PATH
            else _read_regular_source(root / relative)
        )
        result[relative] = hashlib.sha256(raw).hexdigest()
    if not set(SOURCE_PATHS).issubset(result):
        raise PermissionError("direct probe source escaped recursive closure")
    for relative, digest in expected.items():
        if result.get(relative) != digest:
            raise PermissionError(f"frozen source changed: {relative}")
    return result


def hardware_contract() -> dict[str, Any]:
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


def lifecycle_contract() -> dict[str, Any]:
    return {
        "immutable_order": [
            "exact_source_review_and_authorization_rehash",
            "validate_isolated_no_tensor_hardware_preflight",
            "reserve_unique_mode_0700_output_root",
            "create_fsynced_hash_chained_partial_access_ledger",
            "deferred_torch_stack_import",
            "ledgered_bound_schedule_owner_validation_first",
            "ledgered_n320_and_raw_runtime_input_load",
            "schedule_ordered_train_identity_finalization_without_reopen",
            "training_update",
            "cpu_snapshot",
            "one_inline_nonmutating_selection_evaluation",
            "atomic_mode_0444_metric_sidecar",
            "control_branch",
            "terminal_publication",
            "seal_all_terminal_files_read_only",
        ],
        "reservation_consumes_attempt": True,
        "retry_resume_recovery_second_seed_or_extension": False,
        "source_review_may_open_generated_inputs": False,
        "v1_runtime_output_open_authorized": False,
        "failure_receipt_binds_reservation_and_partial_access_ledger": True,
        "runtime_open_attempt_and_outcome_fsync_required": True,
        "whole_tree_export_authorized": False,
    }


def operation_counts(update: int, evaluated_updates: Sequence[int]) -> dict[str, Any]:
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
)


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
    if type(value["training_entered"]) is not bool:
        raise PermissionError("training-entered marker changed")
    if any(
        type(value[name]) is not int or value[name] < 0
        for name in PARTIAL_OPERATION_INTEGER_FIELDS
    ):
        raise PermissionError("partial operation count is invalid")
    attempted_evaluations = value[
        "checkpoint_selection_evaluation_updates_attempted"
    ]
    completed_evaluations = value[
        "checkpoint_selection_evaluation_updates_completed"
    ]
    if (
        type(attempted_evaluations) is not list
        or type(completed_evaluations) is not list
        or attempted_evaluations
        != list(CHECKPOINT_UPDATES[: len(attempted_evaluations)])
        or completed_evaluations
        != list(CHECKPOINT_UPDATES[: len(completed_evaluations)])
        or completed_evaluations
        != attempted_evaluations[: len(completed_evaluations)]
        or value["optimizer_construction_attempt_count"] > 1
        or value["optimizer_construction_completion_count"]
        > value["optimizer_construction_attempt_count"]
        or value["optimizer_update_attempt_count"] > MAXIMUM_UPDATE
        or value["complete_optimizer_updates"]
        > value["optimizer_update_attempt_count"]
        or value["pair_index_presentations_attempted"]
        > MAXIMUM_PRESENTATIONS
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
        != len(attempted_evaluations)
        or value["checkpoint_selection_evaluation_completion_count"]
        != len(completed_evaluations)
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
            )
        )
    ):
        raise PermissionError("partial operation-count relationship changed")
    return {
        name: (
            list(item)
            if isinstance(item := value[name], list)
            else item
        )
        for name in fields
    }


def parse_partial_access_ledger(raw: bytes) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n"):
        raise PermissionError("partial-access ledger is not newline terminated")
    records: list[dict[str, Any]] = []
    previous: str | None = None
    attempts: dict[int, dict[str, Any]] = {}
    outcomes: dict[int, dict[str, Any]] = {}
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
        record = parse_canonical_json(
            line + b"\n",
            name=f"partial-access ledger record {sequence}",
        )
        record_type = record.get("record_type")
        if (
            record_type not in type_fields
            or set(record) != common | type_fields[record_type]
            or record["schema"] != PARTIAL_ACCESS_RECORD_SCHEMA
            or record["sequence"] != sequence
            or record["previous_record_content_sha256"] != previous
        ):
            raise PermissionError("partial-access ledger chain changed")
        if record_type == "LEDGER_OPENED":
            if sequence != 0 or not is_sha256(record["attempt_identity"]):
                raise PermissionError("partial-access ledger header changed")
            validate_binding(record["reservation"], path="reservation.json")
        elif record_type == "OPEN_ATTEMPTED":
            open_id = record["open_id"]
            expected = record["expected_binding"]
            if (
                type(open_id) is not int
                or open_id != len(attempts) + 1
                or type(record["stage"]) is not str
                or not record["stage"]
                or type(record["kind"]) is not str
                or not record["kind"]
                or (
                    record["role"] is not None
                    and type(record["role"]) is not str
                )
                or type(record["purpose"]) is not str
                or type(expected) is not dict
                or set(expected) != {
                    "path",
                    "file_sha256",
                    "content_sha256",
                    "byte_count",
                }
                or not is_sha256(expected["file_sha256"])
                or (
                    expected["content_sha256"] is not None
                    and not is_sha256(expected["content_sha256"])
                )
                or (
                    expected["byte_count"] is not None
                    and (
                        type(expected["byte_count"]) is not int
                        or expected["byte_count"] <= 0
                    )
                )
            ):
                raise PermissionError("partial-access attempt changed")
            safe_relative_path(expected["path"], name="attempted input path")
            attempts[open_id] = record
        elif record_type == "OPEN_OUTCOME":
            open_id = record["open_id"]
            observed = record["observed_binding"]
            if (
                type(open_id) is not int
                or open_id not in attempts
                or open_id in outcomes
                or record["stage"] != attempts[open_id]["stage"]
                or record["kind"] != attempts[open_id]["kind"]
                or record["outcome"] not in {
                    "ACCEPTED",
                    "REJECTED_BINDING",
                    "OPEN_FAILED",
                    "READ_FAILED",
                }
                or type(record["descriptor_opened"]) is not bool
                or type(record["read_completed"]) is not bool
                or type(record["binding_accepted"]) is not bool
                or type(record["partial_byte_count"]) is not int
                or record["partial_byte_count"] < 0
                or (
                    record["read_completed"]
                    and not record["descriptor_opened"]
                )
                or (
                    observed is not None
                    and (
                        type(observed) is not dict
                        or set(observed) != {
                            "path",
                            "file_sha256",
                            "byte_count",
                        }
                        or not is_sha256(observed["file_sha256"])
                        or type(observed["byte_count"]) is not int
                        or observed["byte_count"] < 0
                        or observed["path"]
                        != attempts[open_id]["expected_binding"]["path"]
                    )
                )
                or (
                    record["read_completed"]
                    and observed is None
                )
                or (
                    record["binding_accepted"]
                    and (
                        record["outcome"] != "ACCEPTED"
                        or not record["descriptor_opened"]
                        or not record["read_completed"]
                        or observed is None
                        or record["error"] is not None
                    )
                )
                or (
                    not record["binding_accepted"]
                    and (
                        record["outcome"] == "ACCEPTED"
                        or type(record["error"]) is not dict
                        or set(record["error"])
                        != {"type", "message", "message_sha256"}
                        or type(record["error"]["message"]) is not str
                        or hashlib.sha256(
                            record["error"]["message"].encode("utf-8")
                        ).hexdigest()
                        != record["error"]["message_sha256"]
                    )
                )
            ):
                raise PermissionError("partial-access outcome changed")
            outcomes[open_id] = record
        else:
            validate_partial_operation_counts(record["operation_counts"])
        previous = record["content_sha256"]
        records.append(record)
    if (
        not records
        or records[0]["record_type"] != "LEDGER_OPENED"
        or set(attempts) != set(outcomes)
        or records[-1]["record_type"]
        not in {"ATTEMPT_TERMINATING", "RUNTIME_INPUT_ACCESS_FINALIZED"}
    ):
        raise PermissionError("partial-access ledger is incomplete")
    return records


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
        "v1_runtime_output_open_count",
        "authority",
        "terminalization",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V2 failure-receipt fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    reservation = validate_binding(value["reservation"], path="reservation.json")
    ledger = value["partial_access_ledger"]
    opens = value["runtime_opens"]
    stage = value["failure_stage"]
    error = value["error"]
    prefix = value["published_prefix"]
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
    prefix_fields = {"path", "file_sha256", "byte_count", "mode"}
    if (
        value["schema"] != FAILURE_SCHEMA
        or value["status"]
        != "TERMINAL_V2_OPERATIONAL_OR_INTEGRITY_FAILURE_NO_RETRY"
        or not is_sha256(value["attempt_identity"])
        or (
            reservation_binding is not None
            and reservation != dict(reservation_binding)
        )
        or type(ledger) is not dict
        or set(ledger) != {
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
        or ledger["path"] != "partial_access.jsonl"
        or not all(
            is_sha256(ledger[name])
            for name in (
                "file_sha256",
                "records_content_sha256",
                "last_record_content_sha256",
            )
        )
        or any(
            type(ledger[name]) is not int or ledger[name] < 0
            for name in (
                "byte_count",
                "record_count",
                "attempted_open_count",
                "descriptor_opened_count",
                "read_completed_count",
                "accepted_open_count",
                "rejected_or_failed_open_count",
            )
        )
        or type(opens) is not list
        or ledger["attempted_open_count"] != len(opens)
        or any(
            type(row) is not dict
            or set(row) != open_fields
            or row["open_id"] != index
            for index, row in enumerate(opens, start=1)
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
        or type(stage) is not dict
        or set(stage) != {
            "name",
            "update",
            "microbatch",
            "checkpoint_update",
            "role",
        }
        or type(stage["name"]) is not str
        or not stage["name"]
        or validate_partial_operation_counts(value["operation_counts"])
        != value["operation_counts"]
        or type(prefix) is not list
        or any(
            type(row) is not dict
            or set(row) != prefix_fields
            or not is_sha256(row["file_sha256"])
            or type(row["byte_count"]) is not int
            or row["byte_count"] < 0
            or row["mode"] not in {"0444", "0600", "0644"}
            for row in prefix
        )
        or [row["path"] for row in prefix]
        != sorted({row["path"] for row in prefix})
        or "reservation.json" not in {row["path"] for row in prefix}
        or "partial_access.jsonl" not in {row["path"] for row in prefix}
        or value["published_prefix_sha256"] != canonical_json_sha256(prefix)
        or type(value["directories_including_root"]) is not list
        or not value["directories_including_root"]
        or value["directories_including_root"][0] != "."
        or type(error) is not dict
        or set(error) != {"type", "message", "message_sha256"}
        or hashlib.sha256(error["message"].encode("utf-8")).hexdigest()
        != error["message_sha256"]
        or value["scientific_result"] is not None
        or value["scientific_result_status"]
        != "NOT_OBSERVED_TERMINAL_OPERATIONAL_OR_INTEGRITY_FAILURE"
        or value["retry_authorized"] is not False
        or value["g2_navigation_or_heldout_attempted"] is not False
        or value["v1_runtime_output_open_count"] != 0
        or value["authority"] != DOWNSTREAM_DENIALS
        or value["terminalization"] != {
            "failure_publication": "exclusive_atomic_fsync",
            "terminal_file_mode": "0444",
            "terminal_directory_mode": "0555",
            "seal_after_publication": True,
        }
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V2 failure receipt changed")
    return dict(value)


def science_contract() -> dict[str, Any]:
    return {
        "model_family": "shared_observable_camera_ray_jepa_v5_multires_v1",
        "model_runtime_version": MODEL_RUNTIME_VERSION,
        "one_science_delta":
            "progressive_multiresolution_spatial_decoder_only",
        "initialization": {
            "base_seed": BASE_INITIALIZATION_SEED,
            "decoder_local_cpu_seed": DECODER_INITIALIZATION_SEED,
            "n320_only_tensor_input": True,
            "permitted_copies": ["encoder", "pixel_head", "ground_head"],
            "predecessor_dense_decoder_copy_count": 0,
            "hard_sync_ema_count": 1,
            "rejected_adaptation_checkpoint_open_count": 0,
        },
        "data": {
            "train": dict(TRAIN_ROLE_COUNTS),
            "checkpoint_selection": dict(SELECTION_ROLE_COUNTS),
            "probability_calibration_open_count": 0,
            "role_rebuild_refinement_or_reordering": False,
        },
        "schedule": {
            "seed": SCHEDULE_SEED,
            "updates": MAXIMUM_UPDATE,
            "presentations": MAXIMUM_PRESENTATIONS,
            "checkpoints": list(CHECKPOINT_UPDATES),
            "prefix_sha256": {
                str(key): value
                for key, value in CHECKPOINT_SCHEDULE_PREFIX_SHA256.items()
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
            "post_clip_tolerance": POST_CLIP_NORM_ASSERTION_TOLERANCE,
            "microbatch_size": MICROBATCH_SIZE,
            "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        },
        "parameter_counts": dict(EXPECTED_PARAMETER_COUNTS),
        "parameter_tensor_counts": dict(EXPECTED_PARAMETER_TENSOR_COUNTS),
        "checkpoints": {
            "100": "integrity_and_informational_only",
            "400": "integrity_and_informational_only",
            "1000": "strict_terminal_conjunction",
        },
        "pass_thresholds": dict(PASS_THRESHOLDS),
        "threshold_equality_passes": False,
        "maximum_attempts": 1,
        "operation_cap": operation_counts(MAXIMUM_UPDATE, CHECKPOINT_UPDATES),
        "authority": dict(DOWNSTREAM_DENIALS),
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
                "multires_perception_training",
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
        or camera["seed"] != 20260710
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
    core = dict(value)
    declared = core.pop("content_sha256")
    reviewer = value["reviewer"]
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != "PASS_SOURCE_ONLY"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or value["reviewed_sources"] != dict(expected_sources)
        or value["preregistration"] != preregistration_binding()
        or value["frozen_source_bindings"] != {
            **FROZEN_SOURCE_SHA256,
            MODEL_RELATIVE_PATH: MODEL_FILE_SHA256,
        }
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
    core = dict(value)
    declared = core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != "AUTHORIZED_ONE_EXACT_BOUNDED_PROBE"
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_source_review"] != dict(review_binding)
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
    margins.extend(_lower(value, 0.92, name="distance group") for value in distance)
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
                rough["depth_p95_error_m"], name="rough depth p95"
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
        rough["pixel_balanced_accuracy"], name="rough pixel balanced accuracy"
    )
    ground = _finite(
        rough["ground_balanced_accuracy"], name="rough ground balanced accuracy"
    )
    depth = _finite(rough["depth_p95_m"], name="rough depth p95")
    if shortfall < 0.0:
        raise ValueError("total shortfall is negative")
    conjuncts = {
        "complete_physical_scope_count_at_least_1": complete >= 1,
        "passed_margin_count_at_least_98": passed >= 98,
        "total_shortfall_strictly_below_threshold":
            shortfall < PASS_THRESHOLDS["total_shortfall_strictly_less_than"],
        "rough_pixel_balanced_accuracy_strictly_above_threshold":
            pixel
            > PASS_THRESHOLDS[
                "rough_pixel_balanced_accuracy_strictly_greater_than"
            ],
        "rough_ground_balanced_accuracy_strictly_above_threshold":
            ground
            > PASS_THRESHOLDS[
                "rough_ground_balanced_accuracy_strictly_greater_than"
            ],
        "rough_depth_p95_strictly_below_threshold":
            depth
            < PASS_THRESHOLDS["rough_depth_p95_m_strictly_less_than"],
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
        raise ValueError("metric sidecar update is not fixed")
    return f"checkpoints/update_{update}.metrics.json"


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
    metric = value["metric"]
    if type(metric) is not dict:
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
        or observed_update not in CHECKPOINT_UPDATES
        or (update is not None and observed_update != update)
        or type(value["checkpoint"]) is not dict
        or metric.get("update") != observed_update
        or value["inline_evaluation_count"] != 1
        or value["state_mutation_count"] != 0
        or value["publication_order"] != [
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


def parameter_partition(name: str) -> str:
    prefixes = (*TRAINABLE_PARAMETER_PREFIXES, *FROZEN_STATE_PREFIXES)
    matches = [prefix.removesuffix(".") for prefix in prefixes if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"model state escaped the fixed partition: {name}")
    return matches[0]


__all__ = [name for name in globals() if name.isupper()] + [
    "artifact_binding",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "checkpoint_control_decision",
    "current_source_bindings",
    "evaluate_physical_scopes",
    "hardware_contract",
    "is_sha256",
    "learning_rates",
    "lifecycle_contract",
    "metric_sidecar_relative_path",
    "operation_counts",
    "parameter_partition",
    "parse_canonical_json",
    "physical_margins",
    "preregistration_binding",
    "safe_relative_path",
    "science_contract",
    "validate_authorization",
    "validate_binding",
    "validate_checkpoint_prefix",
    "validate_metric_sidecar",
    "validate_review",
    "validate_runtime_inputs",
    "with_content_sha256",
]
