#!/usr/bin/env python3
"""Externally supervise one exact existing-pool three-arm experiment.

This source grants no authority.  It accepts only a separately committed,
caller-bound authority, verifies its plan/review/source/runtime/input closure,
exclusively reserves the one fixed development attempt, and then launches the
bound worker once under a hard subprocess wall ceiling.  A receipt-only JSON
checker runs under the same ceiling.  Reservation consumes the attempt even
when the worker or checker fails; retry, resume, refill, and overwrite remain
false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_ROOT = REPO_ROOT / ".generated" / "dev"
ATTEMPT_ROOT = (
    DEVELOPMENT_ROOT
    / "world_model_existing_pool_three_arm_v1"
    / "attempt_v1"
)
ATTEMPT_ID = "world_model_existing_pool_three_arm_v1/attempt_v1"
WORKER_RELATIVE = Path(
    "scripts/execute_go2_world_model_existing_pool_three_arm_v1.py"
)
CHECKER_RELATIVE = Path(
    "scripts/check_go2_world_model_existing_pool_three_arm_v1.py"
)
SUPERVISOR_RELATIVE = Path(
    "scripts/run_go2_world_model_existing_pool_three_arm_authorized_v1.py"
)

AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_execution_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_EXISTING_POOL_THREE_ARM_ATTEMPT"
PLAN_SCHEMA = "lewm_go2_world_model_existing_pool_three_arm_plan_v1"
REVIEW_SCHEMA = "lewm_go2_world_model_follow_on_independent_source_review_v1"
REVIEW_STATUS = "PASS_SOURCE_ONLY_NOT_AUTHORITY"
RESERVATION_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_reservation_v1"
)
RESULT_SCHEMA = "lewm_go2_world_model_existing_pool_three_arm_result_v1"
RESULT_STATUS = "COMPLETE_PENDING_TERMINAL_REVIEW"
CHECK_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_receipt_check_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_supervision_terminal_v1"
)
ARM_ORDER = ["conditioned", "blind", "shuffled"]
WORKER_OUTPUT_PATHS = frozenset(
    {
        "pack/manifest.json",
        "pack/train_frames.u8",
        "pack/train_actions.npy",
        "pack/train_meta.json",
        "pack/val_frames.u8",
        "pack/val_actions.npy",
        "pack/val_meta.json",
        "overlap_audit.json",
        "shuffle_audit.json",
    }
    | {
        f"arms/{arm}/measurements/update_{update:06d}.json"
        for arm in ARM_ORDER
        for update in range(0, 701, 100)
    }
    | {
        f"arms/{arm}/snapshots/update_{update:06d}.pt"
        for arm in ARM_ORDER
        for update in range(0, 701, 100)
    }
)
EXACT_CHILD_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "HIP_VISIBLE_DEVICES": "0",
    "ROCR_VISIBLE_DEVICES": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONSAFEPATH": "1",
    "OMP_NUM_THREADS": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
}
GIT_EXECUTABLE = "/usr/bin/git"
GIT_ENVIRONMENT = {
    key: EXACT_CHILD_ENVIRONMENT[key]
    for key in (
        "PATH",
        "LANG",
        "LC_ALL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_TERMINAL_PROMPT",
        "GIT_OPTIONAL_LOCKS",
    )
}
MINIMUM_FREE_OUTPUT_BYTES = 16 * 1024**3
REQUIRED_SOURCE_PATHS = {
    "lewm_package": "lewm/__init__.py",
    "benchmarks_package": "lewm/benchmarks/__init__.py",
    "counterfactual_metrics": "lewm/benchmarks/counterfactual.py",
    "datasets_package": "lewm/datasets/__init__.py",
    "models_package": "lewm/models/__init__.py",
    "base_world_model": "lewm/models/lewm.py",
    "phase2d_spatial_model": "lewm/models/phase2d_spatial_lewm.py",
    "base_predictor": "lewm/models/predictor.py",
    "primitive_affordance": "lewm/models/primitive_affordance.py",
    "sigreg": "lewm/models/sigreg.py",
    "source_action_utility": "lewm/models/source_action_utility.py",
    "spatial_lewm": "lewm/models/spatial_lewm.py",
    "spatial_predictor": "lewm/models/spatial_predictor.py",
    "worker": "scripts/execute_go2_world_model_existing_pool_three_arm_v1.py",
    "checker": "scripts/check_go2_world_model_existing_pool_three_arm_v1.py",
    "external_supervisor": (
        "scripts/run_go2_world_model_existing_pool_three_arm_authorized_v1.py"
    ),
    "experiment_metrics": (
        "lewm/benchmarks/go2_world_model_existing_pool_three_arm_v1.py"
    ),
    "temporal_metrics": (
        "lewm/benchmarks/go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "h6_dataset": (
        "lewm/datasets/go2_explicit_plan_discounted_successor_state_v27.py"
    ),
    "h6_main_pool_census": (
        "lewm/benchmarks/go2_recurrent_jepa_main_pool_census.py"
    ),
    "h6_sequence_contract_v2": (
        "lewm/datasets/go2_recurrent_h4_rgb_sequences_v2.py"
    ),
    "h6_sequence_contract_v1": (
        "lewm/datasets/go2_recurrent_h4_rgb_sequences.py"
    ),
    "temporal_model": (
        "lewm/models/rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "spatial_model": (
        "lewm/models/rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "encoders": "lewm/models/encoders.py",
    "temporal_training_core": (
        "scripts/run_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "temporal_evaluator": (
        "scripts/evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "spatial_evaluator": (
        "scripts/evaluate_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "spatial_metrics": (
        "lewm/benchmarks/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "place_data": "lewm/datasets/go2_memory_role_place_triplets_v1.py",
    "packer": "scripts/dev_pack_h6_temporal_frames.py",
    "scaled_runtime": "scripts/dev_train_temporal_jepa_scaled.py",
}

_SHA256 = frozenset("0123456789abcdef")
_AUTHORITY_KEYS = frozenset(
    {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "authorizer",
        "issued_at",
        "scientific_claim_authorized",
        "network_access",
        "source_commit",
        "plan_binding",
        "review_binding",
        "source_bindings",
        "runtime",
        "input_bindings",
        "output_root",
        "attempt",
        "caps",
        "authorized_command",
        "execution",
        "external_supervisor",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "id",
        "root",
        "maximum_attempts",
        "must_be_absent",
        "reservation_consumes_attempt",
        "retry",
        "resume",
        "overwrite",
        "refill",
    }
)
class ThreeArmSupervisionError(RuntimeError):
    """Raised when authority or one-shot execution fails closed."""


def _fail(message: str) -> None:
    raise ThreeArmSupervisionError(message)


def _reject_protected_path(path: Path, *, label: str) -> None:
    lowered = tuple(part.lower() for part in Path(path).parts)
    if any(
        part == "sealed_test.json"
        or part == "sealed"
        or part.startswith("sealed_")
        or part in {"heldout", "held_out", "held-out"}
        or part.startswith("heldout_")
        or part.startswith("held_out_")
        or part.startswith("held-out-")
        for part in lowered
    ):
        _fail(f"{label} path is custody-protected")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ThreeArmSupervisionError(
                    f"non-finite JSON value in {label}: {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThreeArmSupervisionError(f"invalid JSON in {label}") from exc


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in _SHA256 for character in value)
    )


def _plain_dict(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be a plain JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], *, label: str) -> None:
    observed = set(value)
    required = set(expected)
    if observed != required:
        _fail(
            f"{label} keys changed: missing={sorted(required - observed)}, "
            f"unexpected={sorted(observed - required)}"
        )


def file_binding(path: Path) -> dict[str, Any]:
    """Hash one regular non-symlink file under a stable inode/size check."""

    selected = Path(path)
    _reject_protected_path(selected, label="bound file")
    if selected.is_symlink() or not selected.is_file():
        _fail(f"bound file is absent, non-regular, or a symlink: {selected}")
    before = selected.stat()
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = selected.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        _fail(f"bound file changed while being read: {selected}")
    return {
        "path": str(selected.resolve()),
        "file_sha256": digest.hexdigest(),
        "byte_count": int(after.st_size),
    }


def binding_shape(value: Any, *, label: str) -> dict[str, Any]:
    binding = _plain_dict(value, label=label)
    _exact_keys(
        binding,
        ("path", "file_sha256", "byte_count"),
        label=label,
    )
    if type(binding["path"]) is not str or not binding["path"]:
        _fail(f"{label}.path is invalid")
    _reject_protected_path(Path(binding["path"]), label=label)
    if not _is_sha256(binding["file_sha256"]):
        _fail(f"{label}.file_sha256 is invalid")
    if type(binding["byte_count"]) is not int or binding["byte_count"] < 1:
        _fail(f"{label}.byte_count is invalid")
    return dict(binding)


def _resolve_bound_path(path_text: str) -> Path:
    value = Path(path_text)
    return value if value.is_absolute() else REPO_ROOT / value


def verify_binding(value: Any, *, label: str) -> dict[str, Any]:
    expected = binding_shape(value, label=label)
    actual = file_binding(_resolve_bound_path(expected["path"]))
    if actual["byte_count"] != expected["byte_count"]:
        _fail(f"{label} byte count changed")
    if actual["file_sha256"] != expected["file_sha256"]:
        _fail(f"{label} SHA-256 changed")
    return actual


def _read_bound_json(value: Any, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = binding_shape(value, label=label)
    path = _resolve_bound_path(str(expected["path"]))
    actual = verify_binding(expected, label=label)
    raw = path.read_bytes()
    if file_binding(path) != actual:
        _fail(f"{label} changed while being parsed")
    document = strict_json_bytes(raw, label=label)
    return _plain_dict(document, label=label), actual


def _git_output(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        [GIT_EXECUTABLE, *args],
        cwd=REPO_ROOT,
        env=GIT_ENVIRONMENT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return result.stdout if binary else result.stdout.strip()


def _git_head() -> str:
    value = _git_output("rev-parse", "HEAD")
    assert isinstance(value, str)
    return value


def _require_commit_ancestor(commit: Any, *, label: str) -> str:
    if not _is_commit(commit):
        _fail(f"{label} commit must be full lowercase Git hex")
    result = subprocess.run(
        [GIT_EXECUTABLE, "merge-base", "--is-ancestor", str(commit), "HEAD"],
        cwd=REPO_ROOT,
        env=GIT_ENVIRONMENT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        _fail(f"{label} commit is not an ancestor of HEAD")
    return str(commit)


def _require_binding_at_commit(
    binding: Mapping[str, Any], *, commit: str, label: str
) -> None:
    try:
        path = _resolve_bound_path(str(binding["path"])).resolve(strict=True)
        relative = path.relative_to(REPO_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ThreeArmSupervisionError(
            f"{label} must be a tracked repository file"
        ) from exc
    try:
        raw = _git_output("show", f"{commit}:{relative.as_posix()}", binary=True)
    except subprocess.CalledProcessError as exc:
        raise ThreeArmSupervisionError(
            f"{label} is absent from commit {commit}"
        ) from exc
    assert isinstance(raw, bytes)
    if len(raw) != int(binding["byte_count"]):
        _fail(f"committed {label} byte count changed")
    if hashlib.sha256(raw).hexdigest() != str(binding["file_sha256"]):
        _fail(f"committed {label} SHA-256 changed")


def _validate_binding_map(
    value: Any, *, label: str, verify_files: bool
) -> dict[str, dict[str, Any]]:
    mapping = _plain_dict(value, label=label)
    if not mapping:
        _fail(f"{label} must not be empty")
    result: dict[str, dict[str, Any]] = {}
    for name, item in mapping.items():
        if type(name) is not str or not name:
            _fail(f"{label} has an invalid binding name")
        binding = binding_shape(item, label=f"{label}.{name}")
        if verify_files:
            verify_binding(binding, label=f"{label}.{name}")
        result[name] = binding
    return result


def _validate_attempt(value: Any, *, output_root: str) -> dict[str, Any]:
    attempt = _plain_dict(value, label="authority.attempt")
    _exact_keys(attempt, _ATTEMPT_KEYS, label="authority.attempt")
    expected_root = str(ATTEMPT_ROOT.resolve(strict=False))
    if (
        attempt.get("id") != ATTEMPT_ID
        or attempt.get("root") != expected_root
        or output_root != expected_root
        or attempt.get("maximum_attempts") != 1
        or attempt.get("must_be_absent") is not True
        or attempt.get("reservation_consumes_attempt") is not True
        or attempt.get("retry") is not False
        or attempt.get("resume") is not False
        or attempt.get("overwrite") is not False
        or attempt.get("refill") is not False
    ):
        _fail("authority attempt is not the exact fresh one-shot attempt")
    return dict(attempt)


def _validate_caps(value: Any) -> dict[str, Any]:
    caps = _plain_dict(value, label="authority.caps")
    wall = caps.get("maximum_wall_seconds")
    gpu = caps.get("maximum_gpu_seconds")
    updates = caps.get("maximum_training_updates")
    if (
        type(wall) not in (int, float)
        or not math.isfinite(float(wall))
        or float(wall) <= 0.0
        or float(wall) > 43_200.0
        or type(gpu) not in (int, float)
        or not math.isfinite(float(gpu))
        or float(gpu) <= 0.0
        or float(gpu) > 36_000.0
        or type(updates) is not int
        or updates != 700
    ):
        _fail(
            "authority caps must stay within 43,200 wall seconds, 36,000 GPU "
            "seconds, and exactly 700 updates"
        )
    return dict(caps)


def _validate_runtime(value: Any, *, verify_files: bool) -> dict[str, Any]:
    runtime = _plain_dict(value, label="authority.runtime")
    if set(runtime) != {"python_invocation_path", "environment", "bindings"}:
        _fail("authority runtime keys changed")
    invocation = runtime["python_invocation_path"]
    environment = runtime["environment"]
    if type(invocation) is not str or not Path(invocation).is_absolute():
        _fail("runtime Python invocation must be absolute")
    if type(environment) is not dict or any(
        type(key) is not str or type(item) is not str
        for key, item in environment.items()
    ):
        _fail("runtime environment must be a string map")
    if environment != EXACT_CHILD_ENVIRONMENT:
        _fail("runtime environment is not the exact allowlisted child environment")
    bindings = _validate_binding_map(
        runtime["bindings"], label="authority.runtime.bindings", verify_files=verify_files
    )
    for required in (
        "python_executable_target",
        "python_environment_config",
        "git_executable",
    ):
        if required not in bindings:
            _fail(f"runtime omits {required}")
    if verify_files:
        invocation_path = Path(invocation)
        target = _resolve_bound_path(bindings["python_executable_target"]["path"])
        config = _resolve_bound_path(bindings["python_environment_config"]["path"])
        git = _resolve_bound_path(bindings["git_executable"]["path"])
        if (
            invocation_path.resolve(strict=True) != target.resolve(strict=True)
            or config.name != "pyvenv.cfg"
            or invocation_path.parent.parent != config.parent
            or git.resolve(strict=True) != Path(GIT_EXECUTABLE).resolve(strict=True)
        ):
            _fail("runtime Python/Git executables differ from their bindings")
    return {
        "python_invocation_path": invocation,
        "environment": dict(environment),
        "bindings": bindings,
    }


def _validate_review(
    review: Mapping[str, Any],
    *,
    source_commit: str,
    source_bindings: list[dict[str, Any]],
    plan_binding: Mapping[str, Any],
) -> None:
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("status") != REVIEW_STATUS
        or review.get("authority_granted_by_this_document") is not False
        or review.get("reviewed_source_commit") != source_commit
        or review.get("reviewed_source_bindings") != source_bindings
        or review.get("reviewed_plan_binding") != plan_binding
        or review.get("remaining_findings") != []
    ):
        _fail("independent source review is not an exact non-authorizing PASS")


def load_and_validate_authority(
    authority_path: Path,
    *,
    expected_byte_count: int,
    expected_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    """Validate the complete launch closure before consuming the attempt."""

    authority_binding = file_binding(authority_path)
    if authority_binding["byte_count"] != expected_byte_count:
        _fail("authority byte count disagrees with caller")
    if authority_binding["file_sha256"] != expected_sha256:
        _fail("authority SHA-256 disagrees with caller")
    _require_binding_at_commit(
        authority_binding, commit="HEAD", label="execution authority"
    )
    authority_raw = authority_path.read_bytes()
    if file_binding(authority_path) != authority_binding:
        _fail("authority changed while being parsed")
    authority = _plain_dict(
        strict_json_bytes(authority_raw, label="authority"),
        label="authority",
    )
    _exact_keys(authority, _AUTHORITY_KEYS, label="authority")
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("authority_granted_by_this_document") is not True
        or authority.get("scientific_claim_authorized") is not False
        or authority.get("network_access") is not False
    ):
        _fail("authority semantic grant is invalid")
    authorizer = authority.get("authorizer")
    if (
        type(authorizer) is not dict
        or type(authorizer.get("identity")) is not str
        or not authorizer["identity"].strip()
        or type(authority.get("issued_at")) is not str
        or not authority["issued_at"].strip()
    ):
        _fail("durable authority authorizer/issue evidence is absent")
    source_commit = _require_commit_ancestor(
        authority.get("source_commit"), label="authorized source"
    )
    plan, plan_binding = _read_bound_json(
        authority["plan_binding"], label="plan"
    )
    review, review_binding = _read_bound_json(
        authority["review_binding"], label="independent source review"
    )
    _require_binding_at_commit(plan_binding, commit="HEAD", label="plan")
    _require_binding_at_commit(
        review_binding, commit="HEAD", label="independent source review"
    )

    raw_sources = authority["source_bindings"]
    if type(raw_sources) is not list or not raw_sources:
        _fail("authority source closure is absent")
    source_bindings: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for row_value in raw_sources:
        row = _plain_dict(row_value, label="source binding row")
        _exact_keys(row, ("name", "binding"), label="source binding row")
        name = row["name"]
        if type(name) is not str or not name or name in by_name:
            _fail("source binding names are invalid or duplicated")
        binding = binding_shape(row["binding"], label=f"source {name}")
        verify_binding(binding, label=f"source {name}")
        _require_binding_at_commit(
            binding, commit=source_commit, label=f"source {name}"
        )
        row_copy = {"name": name, "binding": binding}
        source_bindings.append(row_copy)
        by_name[name] = binding
    for required in REQUIRED_SOURCE_PATHS:
        if required not in by_name:
            _fail(f"source closure omits {required}")
    for name, relative_path in REQUIRED_SOURCE_PATHS.items():
        expected = REPO_ROOT / relative_path
        observed = _resolve_bound_path(by_name[name]["path"]).resolve(strict=True)
        if observed != expected.resolve(strict=True):
            _fail(f"authority binds a different {name} source")

    output_root = authority["output_root"]
    if type(output_root) is not str:
        _fail("authority output_root is invalid")
    attempt = _validate_attempt(authority["attempt"], output_root=output_root)
    caps = _validate_caps(authority["caps"])
    runtime = _validate_runtime(authority["runtime"], verify_files=True)
    inputs = _validate_binding_map(
        authority["input_bindings"],
        label="authority.input_bindings",
        verify_files=True,
    )

    execution = _plain_dict(authority["execution"], label="authority.execution")
    _exact_keys(execution, ("worker_path", "checker_path"), label="authority.execution")
    if execution != {
        "worker_path": str((REPO_ROOT / WORKER_RELATIVE).resolve()),
        "checker_path": str((REPO_ROOT / CHECKER_RELATIVE).resolve()),
    }:
        _fail("authority execution paths changed")

    command = _plain_dict(
        authority["authorized_command"], label="authority.authorized_command"
    )
    _exact_keys(command, ("argv_template",), label="authority.authorized_command")
    expected_command = [
        runtime["python_invocation_path"],
        str((REPO_ROOT / SUPERVISOR_RELATIVE).resolve()),
        "--authority",
        str(authority_path.resolve()),
        "--expected-authority-byte-count",
        "<CALLER_BOUND_AUTHORITY_BYTE_COUNT>",
        "--expected-authority-sha256",
        "<CALLER_BOUND_AUTHORITY_SHA256>",
    ]
    if command["argv_template"] != expected_command:
        _fail("authority does not bind the exact external supervisor invocation")

    external = _plain_dict(
        authority["external_supervisor"], label="authority.external_supervisor"
    )
    _exact_keys(
        external,
        ("source_binding", "terminal_reviewer"),
        label="authority.external_supervisor",
    )
    if (
        binding_shape(external["source_binding"], label="external supervisor")
        != by_name["external_supervisor"]
        or type(external["terminal_reviewer"]) is not str
        or not external["terminal_reviewer"].strip()
    ):
        _fail("external supervisor contract is invalid")
    if Path(verify_binding(external["source_binding"], label="external supervisor")["path"]) != Path(__file__).resolve():
        _fail("authority external supervisor source does not identify this file")

    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("purpose")
        != "existing_pool_three_arm_factual_learning_experiment"
        or plan.get("citable_as_scientific_evidence") is not False
        or plan.get("authorizes_retry_or_resume") is not False
        or plan.get("arm_order") != ARM_ORDER
        or plan.get("output_root") != output_root
        or plan.get("attempt") != attempt
        or plan.get("caps") != caps
        or plan.get("runtime") != runtime
        or plan.get("input_bindings") != inputs
        or plan.get("execution") != execution
    ):
        _fail("bound plan differs from the exact authorized experiment")
    _validate_review(
        review,
        source_commit=source_commit,
        source_bindings=source_bindings,
        plan_binding=plan_binding,
    )
    # Normalize values whose paths may have been repository-relative.
    authority = dict(authority)
    authority["attempt"] = attempt
    authority["caps"] = caps
    authority["runtime"] = runtime
    authority["input_bindings"] = inputs
    authority["source_bindings"] = source_bindings
    authority["review_binding"] = binding_shape(
        authority["review_binding"], label="authority.review_binding"
    )
    authority["plan_binding"] = binding_shape(
        authority["plan_binding"], label="authority.plan_binding"
    )
    return authority, authority_binding, plan, plan_binding, by_name


def _require_fresh_attempt_root(path_text: str) -> Path:
    expected = ATTEMPT_ROOT.resolve(strict=False)
    candidate = Path(path_text)
    if not candidate.is_absolute() or candidate.resolve(strict=False) != expected:
        _fail("attempt root is not the one exact authorized development root")
    development = DEVELOPMENT_ROOT.resolve(strict=True)
    try:
        relative = expected.relative_to(development)
    except ValueError as exc:
        raise ThreeArmSupervisionError("attempt root escapes .generated/dev") from exc
    cursor = development
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"attempt path contains a symlink: {cursor}")
        if not cursor.exists():
            break
    if candidate.exists() or candidate.is_symlink():
        _fail(f"attempt root is not fresh: {candidate}")
    free_bytes = int(shutil.disk_usage(DEVELOPMENT_ROOT).free)
    if free_bytes < MINIMUM_FREE_OUTPUT_BYTES:
        _fail(
            "development output volume lacks the 16 GiB preregistered free-space floor"
        )
    return candidate


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    raw = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_binding(path)


def _reserve_attempt(
    attempt_root: Path,
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    worker_binding: Mapping[str, Any],
    checker_binding: Mapping[str, Any],
    worker_command: Sequence[str],
    checker_command_template: Sequence[str],
    supervisor_nonce: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exclusively create the attempt and its attempt-consuming reservation."""

    parent = attempt_root.parent
    parent.mkdir(mode=0o755, parents=False, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        _fail("attempt parent is not a regular directory")
    os.mkdir(attempt_root, mode=0o755)
    parent_descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    reservation = {
        "schema": RESERVATION_SCHEMA,
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "supervisor_nonce": supervisor_nonce,
        "authority_binding": dict(authority_binding),
        "plan_binding": dict(plan_binding),
        "review_binding": dict(authority["review_binding"]),
        "source_commit": authority["source_commit"],
        "source_bindings": authority["source_bindings"],
        "runtime": authority["runtime"],
        "input_bindings": authority["input_bindings"],
        "attempt": authority["attempt"],
        "caps": authority["caps"],
        "worker_binding": dict(worker_binding),
        "checker_binding": dict(checker_binding),
        "output_root": authority["output_root"],
        "execution": authority["execution"],
        "worker_command": list(worker_command),
        "checker_command_template": list(checker_command_template),
        "maximum_attempts": 1,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    reservation_binding = _write_json_exclusive(
        attempt_root / "reservation.json", reservation
    )
    directory_descriptor = os.open(attempt_root, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return reservation, reservation_binding


def _child_environment(runtime: Mapping[str, Any]) -> dict[str, str]:
    environment = runtime.get("environment")
    if environment != EXACT_CHILD_ENVIRONMENT:
        _fail("refusing to construct an unbound child environment")
    return dict(EXACT_CHILD_ENVIRONMENT)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_once(
    argv: Sequence[str], *, timeout: float, env: Mapping[str, str]
) -> dict[str, Any]:
    if timeout <= 0.0:
        _fail("hard wall ceiling exhausted")
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=REPO_ROOT,
        env=dict(env),
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise ThreeArmSupervisionError(
            "supervised command exceeded hard wall ceiling"
        ) from exc
    except BaseException:
        _terminate_process_group(process)
        raise
    elapsed = time.monotonic() - started
    if returncode != 0:
        _terminate_process_group(process)
        _fail(f"supervised command exited with status {returncode}")
    return {"argv": list(argv), "elapsed_seconds": elapsed, "exit_code": 0}


def _remaining_wall(*, wall_started: float, wall_cap: float) -> float:
    remaining = wall_cap - (time.monotonic() - wall_started)
    if remaining <= 0.0:
        _fail("hard wall ceiling exhausted")
    return remaining


def _load_result_if_present(
    attempt_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = attempt_root / "result.json"
    if not path.exists():
        return None, None
    binding = file_binding(path)
    document = strict_json_bytes(path.read_bytes(), label="worker result")
    if type(document) is not dict:
        _fail("worker result must be a JSON object")
    if file_binding(path) != binding:
        _fail("worker result changed while being loaded")
    return document, binding


def _expected_result_attempt(
    authority_attempt: Mapping[str, Any],
    *,
    reservation_binding: Mapping[str, Any],
    supervisor_nonce: str,
) -> dict[str, Any]:
    return {
        **dict(authority_attempt),
        "reservation": {
            "binding": dict(reservation_binding),
            "supervisor_nonce": supervisor_nonce,
            "status": "RESERVED_ATTEMPT_CONSUMED",
            "maximum_attempts": 1,
            "retry": False,
            "resume": False,
            "overwrite": False,
            "refill": False,
        },
    }


def _validate_worker_result(
    result: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    supervisor_nonce: str,
) -> None:
    runtime = result.get("runtime")
    if (
        type(runtime) is not dict
        or set(runtime) != {"authorized", "observed"}
        or runtime.get("authorized") != authority["runtime"]
        or type(runtime.get("observed")) is not dict
        or not runtime["observed"]
    ):
        _fail("worker result runtime evidence is not exactly linked")
    observed = runtime["observed"]
    expected_runtime = {
        "device_name": "AMD Radeon AI PRO R9700",
        "device_arch": "gfx1201",
        "torch_version": "2.9.1+rocm7.2.1.gitff65f5bc",
        "torch_hip": "7.2.53211-e1a6bc5663",
        "numpy_version": "1.26.4",
        "pillow_version": "11.3.0",
    }
    if any(observed.get(key) != value for key, value in expected_runtime.items()):
        _fail("worker result observed runtime identity changed")
    gpu_elapsed = observed.get("gpu_phase_elapsed_seconds")
    wall_elapsed = observed.get("wall_elapsed_seconds")
    inventory = observed.get("output_inventory")
    if (
        type(gpu_elapsed) not in (int, float)
        or not math.isfinite(float(gpu_elapsed))
        or float(gpu_elapsed) < 0.0
        or float(gpu_elapsed) > float(authority["caps"]["maximum_gpu_seconds"])
        or type(wall_elapsed) not in (int, float)
        or not math.isfinite(float(wall_elapsed))
        or float(wall_elapsed) < 0.0
        or float(wall_elapsed) > float(authority["caps"]["maximum_wall_seconds"])
        or type(inventory) is not list
        or len(inventory) != len(WORKER_OUTPUT_PATHS)
        or any(type(item) is not str or not item for item in inventory)
        or len(set(inventory)) != len(inventory)
        or set(inventory) != WORKER_OUTPUT_PATHS
    ):
        _fail("worker result observed runtime/cap/output evidence is invalid")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != RESULT_STATUS
        or result.get("authority_binding") != authority_binding
        or result.get("plan_binding") != plan_binding
        or result.get("review_binding") != authority["review_binding"]
        or result.get("source_commit") != authority["source_commit"]
        or result.get("attempt")
        != _expected_result_attempt(
            authority["attempt"],
            reservation_binding=reservation_binding,
            supervisor_nonce=supervisor_nonce,
        )
        or result.get("caps") != authority["caps"]
        or result.get("input_bindings") != authority["input_bindings"]
    ):
        _fail("worker result is not an exact linked success")


def _reservation_unchanged(
    path: Path,
    *,
    reservation: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
) -> bool:
    try:
        if file_binding(path) != reservation_binding:
            return False
        observed = strict_json_bytes(path.read_bytes(), label="reservation")
        return observed == reservation and file_binding(path) == reservation_binding
    except BaseException:
        return False


def _reverify_contract(authority: Mapping[str, Any]) -> None:
    verify_binding(authority["plan_binding"], label="plan")
    verify_binding(authority["review_binding"], label="independent source review")
    for row in authority["source_bindings"]:
        verify_binding(row["binding"], label=f"source {row['name']}")
    _validate_runtime(authority["runtime"], verify_files=True)
    _validate_binding_map(
        authority["input_bindings"],
        label="authority.input_bindings",
        verify_files=True,
    )


def _write_terminal(
    attempt_root: Path, value: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        return None
    return _write_json_exclusive(attempt_root / "terminal_supervision.json", value)


def supervise(
    authority_path: Path,
    *,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    authority, authority_binding, _plan, plan_binding, sources = (
        load_and_validate_authority(
            authority_path,
            expected_byte_count=expected_authority_byte_count,
            expected_sha256=expected_authority_sha256,
        )
    )
    attempt_root = _require_fresh_attempt_root(authority["output_root"])
    invocation = str(authority["runtime"]["python_invocation_path"])
    child_env = _child_environment(authority["runtime"])
    wall_cap = float(authority["caps"]["maximum_wall_seconds"])
    gpu_cap = float(authority["caps"]["maximum_gpu_seconds"])
    wall_started = time.monotonic()
    supervisor_nonce = secrets.token_hex(32)
    worker_argv = [
        invocation,
        str((REPO_ROOT / WORKER_RELATIVE).resolve()),
        "--authority",
        str(authority_path.resolve()),
        "--expected-authority-byte-count",
        str(authority_binding["byte_count"]),
        "--expected-authority-sha256",
        str(authority_binding["file_sha256"]),
    ]
    checker_command_template = [
        invocation,
        str((REPO_ROOT / CHECKER_RELATIVE).resolve()),
        "--manifest",
        str((attempt_root / "result.json").resolve()),
        "--expected-file-sha256",
        "<WORKER_RESULT_SHA256>",
        "--expected-byte-count",
        "<WORKER_RESULT_BYTE_COUNT>",
        "--output",
        str((attempt_root / "receipt_check.json").resolve()),
    ]
    reservation, reservation_binding = _reserve_attempt(
        attempt_root,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        worker_binding=sources["worker"],
        checker_binding=sources["checker"],
        worker_command=worker_argv,
        checker_command_template=checker_command_template,
        supervisor_nonce=supervisor_nonce,
    )
    phases: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    result_binding: dict[str, Any] | None = None
    check_binding: dict[str, Any] | None = None
    failure: str | None = None
    try:
        phases.append(
            _run_once(
                worker_argv,
                timeout=_remaining_wall(
                    wall_started=wall_started,
                    wall_cap=min(wall_cap, gpu_cap),
                ),
                env=child_env,
            )
        )
        result, result_binding = _load_result_if_present(attempt_root)
        if result is None or result_binding is None:
            _fail("worker completed without result.json")
        _validate_worker_result(
            result,
            authority=authority,
            authority_binding=authority_binding,
            plan_binding=plan_binding,
            reservation_binding=reservation_binding,
            supervisor_nonce=supervisor_nonce,
        )
        if file_binding(authority_path) != authority_binding:
            _fail("authority changed during worker execution")
        _reverify_contract(authority)
        remaining = _remaining_wall(wall_started=wall_started, wall_cap=wall_cap)
        checker_argv = [
            invocation,
            str((REPO_ROOT / CHECKER_RELATIVE).resolve()),
            "--manifest",
            str((attempt_root / "result.json").resolve()),
            "--expected-file-sha256",
            str(result_binding["file_sha256"]),
            "--expected-byte-count",
            str(result_binding["byte_count"]),
            "--output",
            str((attempt_root / "receipt_check.json").resolve()),
        ]
        phases.append(_run_once(checker_argv, timeout=remaining, env=child_env))
        check_path = attempt_root / "receipt_check.json"
        check_binding = file_binding(check_path)
        check_raw = check_path.read_bytes()
        if file_binding(check_path) != check_binding:
            _fail("receipt check changed while being parsed")
        check = strict_json_bytes(check_raw, label="receipt check")
        if (
            type(check) is not dict
            or check.get("schema") != CHECK_SCHEMA
            or check.get("status") != "PASS"
            or check.get("manifest_binding") != result_binding
            or check.get("pack_payloads_opened") is not False
            or check.get("input_data_opened") is not False
            or check.get("runtime_payloads_opened") is not False
            or check.get("rgb_bytes_opened") is not False
            or check.get("checkpoints_opened") is not False
            or check.get("sealed_material_opened") is not False
        ):
            _fail("receipt-only checker did not exactly pass")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if result_binding is None:
            try:
                result, result_binding = _load_result_if_present(attempt_root)
            except BaseException as result_exc:
                failure += (
                    "; result receipt load failed: "
                    f"{type(result_exc).__name__}: {result_exc}"
                )

    if not _reservation_unchanged(
        attempt_root / "reservation.json",
        reservation=reservation,
        reservation_binding=reservation_binding,
    ):
        changed = "ThreeArmSupervisionError: reservation changed after consumption"
        failure = changed if failure is None else f"{failure}; {changed}"

    blocked = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        wall_elapsed = time.monotonic() - wall_started
        if failure is None and wall_elapsed > wall_cap:
            failure = (
                "ThreeArmSupervisionError: terminal validation exceeded hard wall "
                f"ceiling ({wall_elapsed:.6f} > {wall_cap:.6f} seconds)"
            )
        terminal = {
            "schema": TERMINAL_SCHEMA,
            "status": (
                RESULT_STATUS if failure is None else "CONSUMED_TERMINAL_FAILURE"
            ),
            "citable_as_scientific_evidence": False,
            "scientific_verdict_emitted": False,
            "authorizes_retry_or_resume": False,
            "authority_binding": authority_binding,
            "plan_binding": plan_binding,
            "review_binding": authority["review_binding"],
            "source_commit": authority["source_commit"],
            "execution_head": _git_head(),
            "attempt_root": str(attempt_root.resolve()),
            "reservation_binding": reservation_binding,
            "result_binding": result_binding,
            "receipt_check_binding": check_binding,
            "phase_receipts": phases,
            "wall_elapsed_seconds": wall_elapsed,
            "wall_ceiling_seconds": wall_cap,
            "gpu_ceiling_seconds": float(
                authority["caps"]["maximum_gpu_seconds"]
            ),
            "failure": failure,
            "terminal_reviewer": authority["external_supervisor"]["terminal_reviewer"],
            "automatic_checkpoint_selection_performed": False,
            "retry_authorized": False,
            "resume_authorized": False,
            "overwrite_authorized": False,
            "refill_authorized": False,
            "supervisor_nonce": supervisor_nonce,
        }
        terminal_binding = _write_terminal(attempt_root, terminal)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    return terminal, terminal_binding


def _raise_on_termination_signal(signum: int, _frame: Any) -> None:
    raise ThreeArmSupervisionError(f"supervisor received signal {signum}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_authority_byte_count < 1:
        parser.error("authority byte count must be positive")
    if not _is_sha256(args.expected_authority_sha256):
        parser.error("authority SHA-256 must be lowercase hexadecimal")
    signal.signal(signal.SIGINT, _raise_on_termination_signal)
    signal.signal(signal.SIGTERM, _raise_on_termination_signal)
    terminal, terminal_binding = supervise(
        args.authority,
        expected_authority_byte_count=args.expected_authority_byte_count,
        expected_authority_sha256=args.expected_authority_sha256,
    )
    if terminal_binding is None:
        print("pre-reservation failure; no attempt consumed", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "terminal_supervision": terminal_binding,
            },
            sort_keys=True,
        )
    )
    return 0 if terminal["status"] == RESULT_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
