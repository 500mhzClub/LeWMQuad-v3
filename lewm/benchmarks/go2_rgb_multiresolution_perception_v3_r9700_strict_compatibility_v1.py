"""Pure contracts for the V3 synthetic R9700 strict-kernel compatibility run.

This module is deliberately standard-library-only.  It defines one fixed,
synthetic compatibility attempt for the exact ``grid_sample`` and
``scatter_add`` invocation shapes used by the candidate.  It imports no
Torch, model, dataset, image, checkpoint, or generated-runtime module and
performs no file I/O at import time.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping


SCHEMA_PREFIX = (
    "lewm_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1"
)
IMPLEMENTATION_AUTHOR = "/root/schedule_mismatch_forensics"

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

DECISION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_multiresolution_perception_v3_"
    "operational_recovery_decision_2026-07-24.md"
)
DECISION_BINDING = {
    "path": DECISION_RELATIVE_PATH,
    "file_sha256":
        "94ab2ca50cdc5c33008a411aafc07461684d8564433a9fd787f68308db04b6a2",
    "byte_count": 12030,
}
PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_multiresolution_perception_v3_"
    "preregistration_2026-07-24.json"
)
PREREGISTRATION_BINDING = {
    "path": PREREGISTRATION_RELATIVE_PATH,
    "file_sha256":
        "a8a5d870382ad505edd907f96dfae8a6ed737caf7ff424d2b52f8e4bc020e5d5",
    "content_sha256":
        "64da13d6e38a8c1ee2a1bc87b9917611097023a36939ee4305be9a4e85f602b7",
    "byte_count": 12423,
    "commit": "7e6e539370c8f9d9d228da5ef4bc9ea4d10569a2",
}
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1_source_review_2026-07-24.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1_execution_authorization_2026-07-24.json"
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/"
    "go2_rgb_multiresolution_perception_r9700_strict_compatibility_v1"
)
V3_PROBE_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v3"
)

REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
PREFLIGHT_SCHEMA = f"{SCHEMA_PREFIX}_hardware_preflight_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
SUBPROBE_SCHEMA = f"{SCHEMA_PREFIX}_subprobe_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

REVIEW_STATUS = (
    "PASS_INDEPENDENT_SOURCE_LAUNCHER_OUTPUT_CONTRACT_AND_CLOSURE_REVIEW"
)
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_SYNTHETIC_R9700_STRICT_COMPATIBILITY_RUN"
)
PREFLIGHT_STATUS = "PASS_EXACTLY_ONE_VISIBLE_R9700_ZERO_TENSORS"
RESERVATION_STATUS = "RESERVED_SYNTHETIC_COMPATIBILITY_ATTEMPT"
RESULT_PASS = "PASS_STRICT_KERNEL_COMPATIBILITY"
RESULT_COMPATIBILITY_FAIL = "FAIL_STRICT_KERNEL_COMPATIBILITY_UNSUPPORTED"
COMPLETION_PASS = "TERMINAL_PASS_NO_DOWNSTREAM_AUTHORITY"
COMPLETION_COMPATIBILITY_FAIL = (
    "TERMINAL_COMPATIBILITY_FAIL_NO_RETRY_NO_FALLBACK"
)
FAILURE_STATUS = "TERMINAL_OPERATIONAL_OR_INTEGRITY_FAILURE_NO_RETRY"

EXIT_PASS = 0
EXIT_COMPATIBILITY_FAIL = 10
EXIT_OPERATIONAL_FAILURE = 20

MINIMUM_R9700_TOTAL_MEMORY_BYTES = 32_000_000_000
ATTEMPT_INDEX = 1
MAXIMUM_ATTEMPTS = 1

DECLARED_CANDIDATE_SOURCE_WITNESSES = {
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py":
        "708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85",
    "lewm/models/observable_camera_ray_evidence_v4.py":
        "6238f7fb2b9c0c5201c9d7ebb5343ceef72fa97b423dddb466465b6c594cc882",
    "lewm/models/observable_camera_ray_evidence_v4_training.py":
        "c0f3f944883987950edb7579a9e108171486122a9a3ae9d84d2a1abb6ac015ed",
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py":
        "e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578",
}
PRIOR_STRICT_FAILURE_AUDIT_WITNESS = {
    "path":
        "docs/lewm_go2_shared_jepa_v5_matched_training_v3_"
        "terminal_failure_audit_2026-07-15.json",
    "file_sha256":
        "2f94d6ddaf076bc011eaac46408261aea3b8ac030386c9d2185463fe87a08e4a",
    "content_sha256":
        "b93146f00c79a6b2d151a07fb33696c673a1d45677ee6b948e20acadef9c9899",
    "byte_count": 12883,
}

STRICT_ERROR_SUFFIX = (
    " does not have a deterministic implementation, but you set "
    "'torch.use_deterministic_algorithms(True)'. You can turn off "
    "determinism just for this operation, or you can use the "
    "'warn_only=True' option, if that's acceptable for your application. "
    "You can also file an issue at https://github.com/pytorch/pytorch/issues "
    "to help us prioritize adding deterministic support for this operation."
)
EXPECTED_GRID_STRICT_ERROR = "grid_sampler_2d_backward_cuda" + STRICT_ERROR_SUFFIX
EXPECTED_SCATTER_STRICT_ERROR = "scatter_add_cuda_kernel" + STRICT_ERROR_SUFFIX

GRID_OPERATION = {
    "operation": "grid_sample",
    "execution_order": 1,
    "batch_size": 4,
    "dense_feature_shape": [4, 36, 112, 112],
    "full_query_shape": [4, 128, 128, 5, 2],
    "query_count_per_batch": 81920,
    "query_chunk_size": 4096,
    "grid_call_shape": [4, 4096, 1, 2],
    "grid_call_output_shape": [4, 36, 4096, 1],
    "call_count": 20,
    "input_dtype": "torch.float32",
    "grid_dtype": "torch.float32",
    "input_requires_grad": True,
    "grid_requires_grad": False,
    "mode": "bilinear",
    "padding_mode": "zeros",
    "align_corners": False,
    "backward_call_count": 1,
    "cuda_synchronize_count": 1,
}
SCATTER_OPERATION = {
    "operation": "scatter_add",
    "execution_order": 2,
    "batch_size": 4,
    "depth_bin_count": 64,
    "pixel_ray_shape": [84, 112],
    "ray_count_per_batch": 9408,
    "output_shape": [64, 64],
    "cell_count": 4096,
    "dimension": 0,
    "candidate_count_per_chunk": 4,
    "full_chunk": {
        "local_ray_count": 256,
        "chunk_count": 36,
        "destination_shape": [4194304],
        "source_and_index_shape_before_mask": [4, 64, 256],
        "selected_source_and_index_count": 65536,
    },
    "tail_chunk": {
        "local_ray_count": 192,
        "chunk_count": 1,
        "destination_shape": [3145728],
        "source_and_index_shape_before_mask": [4, 64, 192],
        "selected_source_and_index_count": 49152,
    },
    "scatter_add_call_count": 148,
    "source_dtype": "torch.float32",
    "index_dtype": "torch.int64",
    "validity_dtype": "torch.bool",
    "source_requires_grad": True,
    "synthetic_validity": "all_true_maximal_selected_shape",
    "synthetic_indices": "deterministic_in_range_with_collisions",
    "backward_call_count": 1,
    "cuda_synchronize_count": 1,
}
DETERMINISM_CONTRACT = {
    "requested": "torch.use_deterministic_algorithms(True, warn_only=False)",
    "algorithms_enabled": True,
    "warn_only_enabled": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "warning_count_required": 0,
    "fallback_authorized": False,
    "state_change_after_enable_authorized": False,
}
OPERATION_CONTRACT = {
    "schema": f"{SCHEMA_PREFIX}_operation_contract_v1",
    "synthetic_only": True,
    "strict": DETERMINISM_CONTRACT,
    "operations": [GRID_OPERATION, SCATTER_OPERATION],
    "separate_isolated_children": True,
    "child_order": ["grid_sample", "scatter_add"],
    "model_instantiation_count": 0,
    "optimizer_step_count": 0,
    "learned_state_mutation_count": 0,
}

DOWNSTREAM_DENIALS = {
    "v3_probe_execution_authorized": False,
    "v3_probe_root_inspection_or_reservation_authorized": False,
    "probe_checkpoint_qualified": False,
    "perception_qualification_authorized": False,
    "jepa_training_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "warn_only_or_strict_disable_authorized": False,
    "alternate_device_operation_shape_or_dtype_authorized": False,
    "retry_resume_extension_replacement_or_fallback_authorized": False,
}
PROHIBITED_OPEN_COUNTS = {
    "generated_input_open_count": 0,
    "checkpoint_open_count": 0,
    "tensor_checkpoint_open_count": 0,
    "dataset_open_count": 0,
    "rgb_open_count": 0,
    "model_module_open_count": 0,
    "model_state_open_count": 0,
    "v1_runtime_output_open_count": 0,
    "v2_runtime_output_open_count": 0,
    "v3_probe_root_inspection_count": 0,
    "navigation_artifact_open_count": 0,
    "heldout_or_sealed_open_count": 0,
}
ZERO_TRAINING_COUNTS = {
    "model_instantiation_count": 0,
    "model_training_count": 0,
    "optimizer_construction_count": 0,
    "optimizer_step_count": 0,
    "checkpoint_write_count": 0,
    "learned_state_mutation_count": 0,
}
REVIEW_AUTHORITY = {
    "source_review_only": True,
    "compatibility_run_authorized": False,
    "output_mutation_authorized": False,
    **DOWNSTREAM_DENIALS,
}
EXECUTION_AUTHORITY = {
    "one_exact_synthetic_compatibility_run_authorized": True,
    "required_visible_device": "exactly_one_R9700",
    "synthetic_tensor_allocation_only": True,
    "generated_mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
    "output_root_must_be_absent": True,
    "attempt_index": ATTEMPT_INDEX,
    "maximum_attempts": MAXIMUM_ATTEMPTS,
    "retry_authorized": False,
    **DOWNSTREAM_DENIALS,
}
OUTPUT_CONTRACT = {
    "schema": f"{SCHEMA_PREFIX}_output_contract_v1",
    "root": OUTPUT_ROOT_RELATIVE_PATH,
    "attempt_index": ATTEMPT_INDEX,
    "maximum_attempts": MAXIMUM_ATTEMPTS,
    "root_mode_at_reservation": "0700",
    "terminal_file_mode": "0444",
    "terminal_root_mode": "0555",
    "pass_or_compatibility_fail_inventory": [
        "access.json",
        "completed.json",
        "reservation.json",
        "result.json",
    ],
    "operational_failure_inventory":
        "reservation.json plus optional durable access/result prefix plus failed.json",
    "result_contains_tensors": False,
    "result_contains_scientific_metrics": False,
    "one_attempt_no_retry": True,
    "v3_probe_root": V3_PROBE_ROOT_RELATIVE_PATH,
    "v3_probe_root_must_never_be_inspected_or_reserved": True,
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


OPERATION_CONTRACT_SHA256 = canonical_json_sha256(OPERATION_CONTRACT)
OUTPUT_CONTRACT_SHA256 = canonical_json_sha256(OUTPUT_CONTRACT)


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


def validate_self_hash(
    value: object,
    *,
    schema: str | None = None,
    name: str = "receipt",
) -> dict[str, Any]:
    if type(value) is not dict or not is_sha256(value.get("content_sha256")):
        raise ValueError(f"{name} is not a self-hashed object")
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if canonical_json_sha256(core) != value["content_sha256"]:
        raise ValueError(f"{name} self-hash changed")
    if schema is not None and value.get("schema") != schema:
        raise ValueError(f"{name} schema changed")
    return dict(value)


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(f"{name} must be exactly one canonical JSON line")
    try:
        value = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not ASCII JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"{name} is not canonical")
    return value


def artifact_binding(
    path: str,
    raw: bytes,
    *,
    content_sha256: str | None = None,
) -> dict[str, Any]:
    result = {
        "path": path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    if content_sha256 is not None:
        if not is_sha256(content_sha256):
            raise ValueError("artifact content SHA-256 is invalid")
        result["content_sha256"] = content_sha256
    return result


def validate_binding(value: object, *, path: str | None = None) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("artifact binding must be an object")
    required = {"path", "file_sha256", "byte_count"}
    allowed = required | {"content_sha256", "commit"}
    if set(value) < required or not set(value) <= allowed:
        raise ValueError("artifact binding fields changed")
    if path is not None and value["path"] != path:
        raise ValueError("artifact binding path changed")
    if not is_sha256(value["file_sha256"]):
        raise ValueError("artifact file SHA-256 is invalid")
    if "content_sha256" in value and not is_sha256(value["content_sha256"]):
        raise ValueError("artifact content SHA-256 is invalid")
    if type(value["byte_count"]) is not int or value["byte_count"] < 0:
        raise ValueError("artifact byte count is invalid")
    if "commit" in value and not (
        type(value["commit"]) is str
        and len(value["commit"]) == 40
        and all(character in "0123456789abcdef" for character in value["commit"])
    ):
        raise ValueError("artifact commit is invalid")
    return dict(value)


def validate_fixed_source_layout() -> None:
    output = PurePosixPath(OUTPUT_ROOT_RELATIVE_PATH)
    v3 = PurePosixPath(V3_PROBE_ROOT_RELATIVE_PATH)
    if not output.parts or output.parts[0] != ".generated":
        raise AssertionError("compatibility output root left .generated")
    if output == v3 or output in v3.parents or v3 in output.parents:
        raise AssertionError("compatibility and V3 probe roots overlap")
    if len(SOURCE_PATHS) != len(set(SOURCE_PATHS)) or set(SOURCE_PATHS) != {
        CONTRACT_RELATIVE_PATH,
        LAUNCHER_RELATIVE_PATH,
        RUNNER_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
    }:
        raise AssertionError("checker source closure changed")


validate_fixed_source_layout()


def current_source_bindings(root: Path) -> dict[str, str]:
    """Hash only the four checker sources; never traverse the repository."""

    bindings: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if path.is_symlink():
            raise PermissionError(f"checker source is not a regular file: {relative}")
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError(f"checker source is not a regular file: {relative}")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        fingerprints = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_mode),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        ), (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_mode),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
        )
        if fingerprints[0] != fingerprints[1]:
            raise PermissionError(f"checker source changed while read: {relative}")
        bindings[relative] = hashlib.sha256(b"".join(chunks)).hexdigest()
    return bindings


def source_bindings_sha256(bindings: Mapping[str, str]) -> str:
    if type(bindings) is not dict or set(bindings) != set(SOURCE_PATHS):
        raise ValueError("source binding closure changed")
    if any(not is_sha256(value) for value in bindings.values()):
        raise ValueError("source binding digest is invalid")
    return canonical_json_sha256(dict(bindings))


def validate_preregistration(value: object) -> dict[str, Any]:
    prereg = validate_self_hash(
        value,
        schema="lewm_go2_rgb_multiresolution_perception_v3_preregistration_v1",
        name="V3 preregistration",
    )
    if prereg["content_sha256"] != PREREGISTRATION_BINDING["content_sha256"]:
        raise PermissionError("V3 preregistration content changed")
    if prereg.get("decision_binding") != {
        key: DECISION_BINDING[key]
        for key in ("path", "file_sha256", "byte_count")
    }:
        raise PermissionError("V3 decision binding changed")
    if prereg.get("status") != (
        "PREREGISTERED_SOURCE_ONLY_SCIENCE_IDENTICAL_V3_PENDING_"
        "INDEPENDENT_REVIEW_NO_PROBE_EXECUTION_AUTHORITY"
    ):
        raise PermissionError("V3 preregistration status changed")
    run = prereg.get("synthetic_r9700_compatibility_run")
    if type(run) is not dict or (
        run.get("maximum_runs"),
        run.get("strict_mode_required"),
        run.get("retry_resume_extension_or_fallback_authorized"),
        run.get("v3_probe_root_inspection_or_reservation_count"),
    ) != (1, True, False, 0):
        raise PermissionError("synthetic compatibility authority changed")
    gate = prereg.get("strict_determinism_gate")
    if type(gate) is not dict or (
        gate.get("warn_only"),
        gate.get("disable_strict_mode_authorized"),
        gate.get("alternate_operation_authorized"),
    ) != (False, False, False):
        raise PermissionError("strict determinism gate changed")
    return prereg


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields changed")


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
    preregistration_binding: Mapping[str, Any],
    decision_binding: Mapping[str, Any],
) -> dict[str, Any]:
    review = validate_self_hash(value, schema=REVIEW_SCHEMA, name="source review")
    _exact_fields(
        review,
        {
            "schema",
            "status",
            "reviewer",
            "reviewed_source_commit",
            "source_paths",
            "source_bindings",
            "source_bindings_sha256",
            "preregistration",
            "decision",
            "declared_candidate_source_witnesses",
            "prior_strict_failure_audit_witness",
            "operation_contract_sha256",
            "output_contract_sha256",
            "findings",
            "authority",
            "content_sha256",
        },
        name="source review",
    )
    if review["status"] != REVIEW_STATUS:
        raise PermissionError("source review did not pass")
    if (
        type(review["reviewer"]) is not str
        or not review["reviewer"]
        or review["reviewer"] == IMPLEMENTATION_AUTHOR
    ):
        raise PermissionError("source review is not independent")
    commit = review["reviewed_source_commit"]
    if not (
        type(commit) is str
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("reviewed source commit is invalid")
    sources = dict(expected_sources)
    if (
        review["source_paths"] != list(SOURCE_PATHS)
        or review["source_bindings"] != sources
        or review["source_bindings_sha256"] != source_bindings_sha256(sources)
        or review["preregistration"] != dict(preregistration_binding)
        or review["decision"] != dict(decision_binding)
        or review["declared_candidate_source_witnesses"]
        != DECLARED_CANDIDATE_SOURCE_WITNESSES
        or review["prior_strict_failure_audit_witness"]
        != PRIOR_STRICT_FAILURE_AUDIT_WITNESS
        or review["operation_contract_sha256"] != OPERATION_CONTRACT_SHA256
        or review["output_contract_sha256"] != OUTPUT_CONTRACT_SHA256
        or review["findings"] != []
        or review["authority"] != REVIEW_AUTHORITY
    ):
        raise PermissionError("source review bindings changed")
    return review


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
    expected_source_bindings_sha256: str,
) -> dict[str, Any]:
    authorization = validate_self_hash(
        value,
        schema=AUTHORIZATION_SCHEMA,
        name="execution authorization",
    )
    _exact_fields(
        authorization,
        {
            "schema",
            "status",
            "authorizer",
            "reviewer",
            "source_review",
            "source_bindings_sha256",
            "operation_contract_sha256",
            "output_contract_sha256",
            "attempt_index",
            "maximum_attempts",
            "output_root",
            "authority",
            "content_sha256",
        },
        name="execution authorization",
    )
    authorizer = authorization["authorizer"]
    if (
        type(authorizer) is not str
        or not authorizer
        or authorizer in {reviewer, IMPLEMENTATION_AUTHOR}
    ):
        raise PermissionError("execution authorizer is not independent")
    if (
        authorization["status"] != AUTHORIZATION_STATUS
        or authorization["reviewer"] != reviewer
        or authorization["source_review"] != dict(review_binding)
        or authorization["source_bindings_sha256"]
        != expected_source_bindings_sha256
        or authorization["operation_contract_sha256"]
        != OPERATION_CONTRACT_SHA256
        or authorization["output_contract_sha256"] != OUTPUT_CONTRACT_SHA256
        or authorization["attempt_index"] != ATTEMPT_INDEX
        or authorization["maximum_attempts"] != MAXIMUM_ATTEMPTS
        or authorization["output_root"] != OUTPUT_ROOT_RELATIVE_PATH
        or authorization["authority"] != EXECUTION_AUTHORITY
    ):
        raise PermissionError("execution authorization bindings changed")
    return authorization


def validate_python_identity(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("Python identity must be an object")
    _exact_fields(
        value,
        {
            "implementation",
            "version",
            "cache_tag",
            "executable",
            "isolated",
            "dont_write_bytecode",
        },
        name="Python identity",
    )
    if (
        type(value["implementation"]) is not str
        or type(value["version"]) is not str
        or type(value["cache_tag"]) is not str
        or type(value["executable"]) is not str
        or value["isolated"] is not True
        or value["dont_write_bytecode"] is not True
    ):
        raise ValueError("Python identity values changed")
    return dict(value)


def validate_stack_identity(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("Torch/HIP identity must be an object")
    _exact_fields(
        value,
        {"torch_version", "torch_git_version", "hip_version"},
        name="Torch/HIP identity",
    )
    if not all(type(item) is str and item for item in value.values()):
        raise ValueError("Torch/HIP identity values changed")
    return dict(value)


def validate_device_identity(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("device identity must be an object")
    _exact_fields(
        value,
        {
            "visible_device_count",
            "visible_device_index",
            "visible_device_name",
            "total_memory_bytes",
        },
        name="device identity",
    )
    normalized = str(value["visible_device_name"]).casefold().replace(" ", "")
    if (
        value["visible_device_count"] != 1
        or value["visible_device_index"] != 0
        or "r9700" not in normalized
        or type(value["total_memory_bytes"]) is not int
        or value["total_memory_bytes"] < MINIMUM_R9700_TOTAL_MEMORY_BYTES
    ):
        raise PermissionError("device is not exactly one qualifying R9700")
    return dict(value)


def validate_source_authority_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("source authority receipt must be an object")
    _exact_fields(
        value,
        {
            "source_binding_count",
            "source_bindings_sha256",
            "preregistration",
            "decision",
            "source_review",
            "execution_authorization",
            "generated_runtime_input_open_count",
            "model_or_runtime_root_open_count",
            "torch_imported",
        },
        name="source authority receipt",
    )
    if (
        value["source_binding_count"] != len(SOURCE_PATHS)
        or not is_sha256(value["source_bindings_sha256"])
        or value["generated_runtime_input_open_count"] != 0
        or value["model_or_runtime_root_open_count"] != 0
        or value["torch_imported"] is not False
    ):
        raise PermissionError("source authority receipt changed")
    for key, path in (
        ("preregistration", PREREGISTRATION_RELATIVE_PATH),
        ("decision", DECISION_RELATIVE_PATH),
        ("source_review", REVIEW_RELATIVE_PATH),
        ("execution_authorization", AUTHORIZATION_RELATIVE_PATH),
    ):
        validate_binding(value[key], path=path)
    return dict(value)


def validate_preflight(
    value: object,
    *,
    expected_source_authority: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = validate_self_hash(
        value,
        schema=PREFLIGHT_SCHEMA,
        name="hardware preflight",
    )
    _exact_fields(
        preflight,
        {
            "schema",
            "status",
            "launcher_process_id",
            "preflight_child_process_id",
            "python",
            "stack",
            "device",
            "tensor_allocation_count",
            "memory_allocated_bytes",
            "memory_reserved_bytes",
            "payload_open_count",
            "model_or_runtime_root_open_count",
            "source_authority",
            "launcher_source_sha256",
            "immediate_exec_required",
            "intervening_gpu_query_count",
            "content_sha256",
        },
        name="hardware preflight",
    )
    if (
        preflight["status"] != PREFLIGHT_STATUS
        or type(preflight["launcher_process_id"]) is not int
        or type(preflight["preflight_child_process_id"]) is not int
        or preflight["tensor_allocation_count"] != 0
        or preflight["memory_allocated_bytes"] != 0
        or preflight["memory_reserved_bytes"] != 0
        or preflight["payload_open_count"] != 0
        or preflight["model_or_runtime_root_open_count"] != 0
        or preflight["source_authority"] != dict(expected_source_authority)
        or not is_sha256(preflight["launcher_source_sha256"])
        or preflight["immediate_exec_required"] is not True
        or preflight["intervening_gpu_query_count"] != 0
    ):
        raise PermissionError("hardware preflight contract changed")
    validate_python_identity(preflight["python"])
    validate_stack_identity(preflight["stack"])
    validate_device_identity(preflight["device"])
    return preflight


def attempt_identity_core(
    *,
    source_authority: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_attempt_identity_v1",
        "attempt_index": ATTEMPT_INDEX,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "source_authority": dict(source_authority),
        "preflight_content_sha256": preflight["content_sha256"],
        "operation_contract_sha256": OPERATION_CONTRACT_SHA256,
        "output_contract_sha256": OUTPUT_CONTRACT_SHA256,
    }


def make_attempt_identity(
    *,
    source_authority: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(
        attempt_identity_core(
            source_authority=source_authority,
            preflight=preflight,
        )
    )


def validate_determinism_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict or value != DETERMINISM_CONTRACT:
        raise PermissionError("strict deterministic state changed")
    return dict(value)


def _validate_warning_rows(value: object) -> list[dict[str, str]]:
    if type(value) is not list:
        raise ValueError("warning receipt must be a list")
    result: list[dict[str, str]] = []
    for row in value:
        if type(row) is not dict or set(row) != {
            "category",
            "message",
            "message_sha256",
        }:
            raise ValueError("warning row fields changed")
        if (
            type(row["category"]) is not str
            or type(row["message"]) is not str
            or row["message_sha256"]
            != hashlib.sha256(row["message"].encode("utf-8")).hexdigest()
        ):
            raise ValueError("warning row changed")
        result.append(dict(row))
    return result


def _expected_subprobe_state(
    operation: str,
    outcome: str,
) -> tuple[dict[str, bool], dict[str, int], str]:
    if operation == "grid_sample":
        pass_checks = {
            "strict_state_verified_before_allocation": True,
            "forward_completed": True,
            "output_finite": True,
            "backward_invoked": True,
            "backward_completed": True,
            "input_gradient_finite": True,
            "cuda_synchronize_completed": True,
            "exact_grid_call_count": True,
        }
        pass_counts = {
            "grid_sample_forward_invocation_count": 20,
            "grid_sample_forward_completion_count": 20,
            "backward_invocation_count": 1,
            "backward_completion_count": 1,
            "cuda_synchronize_count": 1,
            "synthetic_dense_tensor_count": 1,
            "synthetic_grid_tensor_count": 1,
            "model_instantiation_count": 0,
            "optimizer_step_count": 0,
            "payload_open_count": 0,
        }
        if outcome == "PASS":
            return pass_checks, pass_counts, "completed"
        fail_checks = dict(pass_checks)
        fail_checks.update({
            "backward_completed": False,
            "input_gradient_finite": False,
            "cuda_synchronize_completed": False,
        })
        fail_counts = dict(pass_counts)
        fail_counts.update({
            "backward_completion_count": 0,
            "cuda_synchronize_count": 0,
        })
        return fail_checks, fail_counts, "grid_backward"
    if operation != "scatter_add":
        raise ValueError("unknown compatibility operation")
    pass_checks = {
        "strict_state_verified_before_allocation": True,
        "all_chunks_completed": True,
        "output_finite": True,
        "backward_invoked": True,
        "backward_completed": True,
        "source_gradients_finite": True,
        "cuda_synchronize_completed": True,
        "exact_scatter_add_call_count": True,
    }
    pass_counts = {
        "full_chunk_invocation_count": 36,
        "full_chunk_completion_count": 36,
        "tail_chunk_invocation_count": 1,
        "tail_chunk_completion_count": 1,
        "scatter_add_invocation_count": 148,
        "scatter_add_completion_count": 148,
        "backward_invocation_count": 1,
        "backward_completion_count": 1,
        "cuda_synchronize_count": 1,
        "synthetic_source_tensor_count": 37,
        "model_instantiation_count": 0,
        "optimizer_step_count": 0,
        "payload_open_count": 0,
    }
    if outcome == "PASS":
        return pass_checks, pass_counts, "completed"
    fail_checks = {name: False for name in pass_checks}
    fail_checks["strict_state_verified_before_allocation"] = True
    fail_counts = {name: 0 for name in pass_counts}
    fail_counts.update({
        "full_chunk_invocation_count": 1,
        "scatter_add_invocation_count": 1,
        "synthetic_source_tensor_count": 1,
    })
    return fail_checks, fail_counts, "scatter_full_forward_candidate_0"


def validate_subprobe_receipt(
    value: object,
    *,
    expected_operation: str,
    expected_python: Mapping[str, Any],
    expected_stack: Mapping[str, Any],
    expected_device: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    receipt = validate_self_hash(
        value,
        schema=SUBPROBE_SCHEMA,
        name=f"{expected_operation} subprobe",
    )
    _exact_fields(
        receipt,
        {
            "schema",
            "operation",
            "execution_order",
            "status",
            "stage",
            "python",
            "stack",
            "device",
            "determinism",
            "operation_contract_sha256",
            "operation_spec",
            "warnings",
            "exception",
            "checks",
            "counts",
            "content_sha256",
        },
        name=f"{expected_operation} subprobe",
    )
    expected_spec = (
        GRID_OPERATION if expected_operation == "grid_sample" else SCATTER_OPERATION
    )
    if (
        receipt["operation"] != expected_operation
        or receipt["execution_order"] != expected_spec["execution_order"]
        or receipt["python"] != dict(expected_python)
        or receipt["stack"] != dict(expected_stack)
        or receipt["device"] != dict(expected_device)
        or canonical_json_bytes(receipt["determinism"])
        != canonical_json_bytes(DETERMINISM_CONTRACT)
        or receipt["operation_contract_sha256"] != OPERATION_CONTRACT_SHA256
        or canonical_json_bytes(receipt["operation_spec"])
        != canonical_json_bytes(expected_spec)
    ):
        raise PermissionError(f"{expected_operation} identity changed")
    validate_python_identity(receipt["python"])
    validate_stack_identity(receipt["stack"])
    validate_device_identity(receipt["device"])
    warnings = _validate_warning_rows(receipt["warnings"])
    if warnings:
        raise RuntimeError(
            f"{expected_operation} emitted a warning under strict mode"
        )
    if type(receipt["checks"]) is not dict or type(receipt["counts"]) is not dict:
        raise ValueError(f"{expected_operation} checks/counts changed")
    exception = receipt["exception"]
    if exception is None:
        expected_checks, expected_counts, expected_stage = (
            _expected_subprobe_state(expected_operation, "PASS")
        )
        if (
            receipt["status"] != "PASS"
            or receipt["stage"] != expected_stage
            or canonical_json_bytes(receipt["checks"])
            != canonical_json_bytes(expected_checks)
            or canonical_json_bytes(receipt["counts"])
            != canonical_json_bytes(expected_counts)
        ):
            raise RuntimeError(f"{expected_operation} did not complete cleanly")
        return receipt, "PASS"
    if type(exception) is not dict or set(exception) != {
        "type",
        "message",
        "message_sha256",
    }:
        raise ValueError(f"{expected_operation} exception fields changed")
    message = exception["message"]
    if (
        exception["type"] != "RuntimeError"
        or type(message) is not str
        or exception["message_sha256"]
        != hashlib.sha256(message.encode("utf-8")).hexdigest()
        or receipt["status"] != "EXCEPTION"
    ):
        raise RuntimeError(f"{expected_operation} raised an unexpected exception")
    if expected_operation == "grid_sample":
        expected_message = EXPECTED_GRID_STRICT_ERROR
    else:
        expected_message = EXPECTED_SCATTER_STRICT_ERROR
    expected_checks, expected_counts, expected_stage = (
        _expected_subprobe_state(expected_operation, "COMPATIBILITY_FAIL")
    )
    if (
        message != expected_message
        or receipt["stage"] != expected_stage
        or canonical_json_bytes(receipt["checks"])
        != canonical_json_bytes(expected_checks)
        or canonical_json_bytes(receipt["counts"])
        != canonical_json_bytes(expected_counts)
    ):
        raise RuntimeError(f"{expected_operation} strict error changed")
    return receipt, "COMPATIBILITY_FAIL"


def validate_access_receipt(value: object) -> dict[str, Any]:
    access = validate_self_hash(value, schema=ACCESS_SCHEMA, name="access receipt")
    if (
        access.get("prohibited_open_counts") != PROHIBITED_OPEN_COUNTS
        or access.get("training_counts") != ZERO_TRAINING_COUNTS
        or access.get("v3_probe_root") != {
            "path": V3_PROBE_ROOT_RELATIVE_PATH,
            "inspected": False,
            "reserved": False,
        }
        or access.get("downstream_denials") != DOWNSTREAM_DENIALS
    ):
        raise PermissionError("access receipt authority changed")
    return access


def validate_result_receipt(value: object) -> dict[str, Any]:
    result = validate_self_hash(value, schema=RESULT_SCHEMA, name="result receipt")
    if result.get("status") not in {RESULT_PASS, RESULT_COMPATIBILITY_FAIL}:
        raise ValueError("result status changed")
    if (
        result.get("operation_contract_sha256") != OPERATION_CONTRACT_SHA256
        or result.get("output_contract_sha256") != OUTPUT_CONTRACT_SHA256
        or result.get("prohibited_open_counts") != PROHIBITED_OPEN_COUNTS
        or result.get("training_counts") != ZERO_TRAINING_COUNTS
        or result.get("downstream_denials") != DOWNSTREAM_DENIALS
        or result.get("scientific_metric") is not None
        or result.get("checkpoint_qualified") is not False
    ):
        raise PermissionError("result authority changed")
    outcomes = result.get("subprobe_outcomes")
    if type(outcomes) is not dict or set(outcomes) != {
        "grid_sample",
        "scatter_add",
    }:
        raise ValueError("result subprobe outcomes changed")
    expected_status = (
        RESULT_PASS
        if set(outcomes.values()) == {"PASS"}
        else RESULT_COMPATIBILITY_FAIL
    )
    if result["status"] != expected_status or any(
        item not in {"PASS", "COMPATIBILITY_FAIL"} for item in outcomes.values()
    ):
        raise ValueError("result status does not match subprobes")
    return result


def validate_completion_receipt(value: object) -> dict[str, Any]:
    completion = validate_self_hash(
        value,
        schema=COMPLETION_SCHEMA,
        name="completion receipt",
    )
    if (
        completion.get("status")
        not in {COMPLETION_PASS, COMPLETION_COMPATIBILITY_FAIL}
        or completion.get("attempt_index") != ATTEMPT_INDEX
        or completion.get("maximum_attempts") != MAXIMUM_ATTEMPTS
        or completion.get("attempt_consumed") is not True
        or completion.get("retry_authorized") is not False
        or completion.get("downstream_denials") != DOWNSTREAM_DENIALS
        or completion.get("terminal_file_mode") != "0444"
        or completion.get("terminal_root_mode") != "0555"
    ):
        raise PermissionError("completion receipt changed")
    return completion


def validate_failure_receipt(value: object) -> dict[str, Any]:
    failure = validate_self_hash(
        value,
        schema=FAILURE_SCHEMA,
        name="failure receipt",
    )
    if (
        failure.get("status") != FAILURE_STATUS
        or failure.get("attempt_index") != ATTEMPT_INDEX
        or failure.get("maximum_attempts") != MAXIMUM_ATTEMPTS
        or failure.get("attempt_consumed") is not True
        or failure.get("retry_authorized") is not False
        or failure.get("compatibility_result") is not None
        or failure.get("prohibited_open_counts") != PROHIBITED_OPEN_COUNTS
        or failure.get("training_counts") != ZERO_TRAINING_COUNTS
        or failure.get("downstream_denials") != DOWNSTREAM_DENIALS
    ):
        raise PermissionError("failure receipt authority changed")
    return failure


__all__ = [
    "ACCESS_SCHEMA",
    "ATTEMPT_INDEX",
    "AUTHORIZATION_RELATIVE_PATH",
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZATION_STATUS",
    "COMPLETION_COMPATIBILITY_FAIL",
    "COMPLETION_PASS",
    "COMPLETION_SCHEMA",
    "CONTRACT_RELATIVE_PATH",
    "DECISION_BINDING",
    "DECISION_RELATIVE_PATH",
    "DETERMINISM_CONTRACT",
    "DOWNSTREAM_DENIALS",
    "EXECUTION_AUTHORITY",
    "EXIT_COMPATIBILITY_FAIL",
    "EXIT_OPERATIONAL_FAILURE",
    "EXIT_PASS",
    "EXPECTED_GRID_STRICT_ERROR",
    "EXPECTED_SCATTER_STRICT_ERROR",
    "FAILURE_SCHEMA",
    "FAILURE_STATUS",
    "GRID_OPERATION",
    "IMPLEMENTATION_AUTHOR",
    "LAUNCHER_RELATIVE_PATH",
    "MAXIMUM_ATTEMPTS",
    "MINIMUM_R9700_TOTAL_MEMORY_BYTES",
    "OPERATION_CONTRACT",
    "OPERATION_CONTRACT_SHA256",
    "OUTPUT_CONTRACT",
    "OUTPUT_CONTRACT_SHA256",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "PREFLIGHT_SCHEMA",
    "PREFLIGHT_STATUS",
    "PREREGISTRATION_BINDING",
    "PREREGISTRATION_RELATIVE_PATH",
    "PROHIBITED_OPEN_COUNTS",
    "RESERVATION_SCHEMA",
    "RESERVATION_STATUS",
    "RESULT_COMPATIBILITY_FAIL",
    "RESULT_PASS",
    "RESULT_SCHEMA",
    "REVIEW_AUTHORITY",
    "REVIEW_RELATIVE_PATH",
    "REVIEW_SCHEMA",
    "REVIEW_STATUS",
    "RUNNER_RELATIVE_PATH",
    "SCATTER_OPERATION",
    "SCHEMA_PREFIX",
    "SOURCE_PATHS",
    "SUBPROBE_SCHEMA",
    "TEST_RELATIVE_PATH",
    "V3_PROBE_ROOT_RELATIVE_PATH",
    "ZERO_TRAINING_COUNTS",
    "artifact_binding",
    "attempt_identity_core",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "current_source_bindings",
    "is_sha256",
    "make_attempt_identity",
    "parse_canonical_json",
    "source_bindings_sha256",
    "validate_access_receipt",
    "validate_authorization",
    "validate_binding",
    "validate_completion_receipt",
    "validate_device_identity",
    "validate_failure_receipt",
    "validate_fixed_source_layout",
    "validate_preflight",
    "validate_preregistration",
    "validate_result_receipt",
    "validate_review",
    "validate_self_hash",
    "validate_source_authority_receipt",
    "validate_stack_identity",
    "validate_subprobe_receipt",
    "with_content_sha256",
]
