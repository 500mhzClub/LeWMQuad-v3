#!/usr/bin/env python3
"""One-shot CPU-only extraction of V3 action-level diagnostic aggregates.

The command may open exactly one runtime snapshot and one validation metadata
index.  It hashes the snapshot before weights-only deserialization, performs no
model reconstruction or forward pass, follows no RGB path from the metadata,
and emits no row-level tensors, model state, optimizer state, or scene IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import traceback
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_world_model_v3_action_localization_v1 as localization_metrics,
)
from lewm.datasets import (  # noqa: E402
    go2_explicit_plan_discounted_successor_state_v27 as h6,
)


AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_action_localization_v1_"
    "execution_authority_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_action_localization_v1_"
    "result_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_action_localization_v1_"
    "reservation_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_action_localization_v1_"
    "failure_v1"
)
SNAPSHOT_SCHEMA = "lewm_go2_world_model_existing_pool_three_arm_snapshot_v1"

ATTEMPT_ID = "world_model_existing_pool_three_arm_v1_action_localization_v1/attempt_v1"
ATTEMPT_ROOT = (
    REPO_ROOT
    / ".generated/dev/world_model_existing_pool_three_arm_v1_action_localization_v1"
    / "attempt_v1"
)
AUTHORITY_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
    "action_localization_v1_execution_authority_2026-08-01.json"
)
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
    "action_localization_v1_preregistration_2026-08-01.md"
)
PLAN_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
    "action_localization_v1_plan_2026-08-01.json"
)
REVIEW_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
    "action_localization_v1_independent_source_review_2026-08-01.json"
)
SUPERVISOR_PATH = (
    REPO_ROOT
    / "scripts/run_go2_world_model_existing_pool_three_arm_v1_action_"
    "localization_authorized_v1.py"
)
SNAPSHOT_PATH = (
    REPO_ROOT
    / ".generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3"
    / "attempt_v1/arms/conditioned/snapshots/update_000700.pt"
)
SNAPSHOT_BINDING = {
    "path": str(SNAPSHOT_PATH),
    "file_sha256": "df961a98ad148d6ba14bcdb03ddf13f3ec6edf73350ca60e1806af04281abe09",
    "byte_count": 212_616_145,
}
VALIDATION_INDEX_PATH = REPO_ROOT / h6.VALIDATION_INDEX
VALIDATION_INDEX_BINDING = {
    "path": str(VALIDATION_INDEX_PATH),
    "file_sha256": h6.VALIDATION_INDEX_SHA256,
    "byte_count": h6.VALIDATION_INDEX_BYTES,
    "row_count": h6.VALIDATION_INDEX_ROWS,
}
V3_INTERNAL_AUTHORITY_BINDING = {
    "path": str(
        REPO_ROOT
        / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
        "integrity_replacement_v3_execution_authority_2026-08-01.json"
    ),
    "file_sha256": "d90aba6198dd3e73629106a37c6567505ade90206d58cd593f2315d637bfaaee",
    "byte_count": 16_075,
}
V3_INTERNAL_PLAN_BINDING = {
    "path": str(
        REPO_ROOT
        / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
        "integrity_replacement_v3_plan_2026-08-01.json"
    ),
    "file_sha256": "053036385837a243509e447ab03d3b177178833c755cdaea33bfbe9b4d60d6dc",
    "byte_count": 7_016,
}
V3_SUBSTRATE_SHA256 = "41160726b4f713f94dd2a4cf9d6f602033ddceab9cd0f06261feca5947230075"
PREDECESSOR_EVIDENCE_BINDINGS = {
    "v3_overlap_audit": {
        "path": str(
            REPO_ROOT
            / ".generated/dev/world_model_existing_pool_three_arm_v1_"
            "integrity_replacement_v3/attempt_v1/overlap_audit.json"
        ),
        "file_sha256": "ec2cfcd008059994d7803f1a14ede5d4ea3b76d50c36d0ca77532ae1deb8c2db",
        "byte_count": 13_368,
    },
    "v3_result": {
        "path": str(
            REPO_ROOT
            / ".generated/dev/world_model_existing_pool_three_arm_v1_"
            "integrity_replacement_v3/attempt_v1/result.json"
        ),
        "file_sha256": "764ee61b7bb8b7e1221f01fc34ba0554d0ca681fde21e99b1a9f5585b3360bd4",
        "byte_count": 26_054,
    },
    "v3_conditioned_update_700_measurement": {
        "path": str(
            REPO_ROOT
            / ".generated/dev/world_model_existing_pool_three_arm_v1_"
            "integrity_replacement_v3/attempt_v1/arms/conditioned/"
            "measurements/update_000700.json"
        ),
        "file_sha256": "87bad8bbfb15ce665a2b80477f4d11c3b4a997c4c0c7e7c15ba96c911345cf0b",
        "byte_count": 5_392,
    },
    "v3_terminal_review": {
        "path": str(
            REPO_ROOT
            / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
            "integrity_replacement_v3_terminal_review_2026-08-01.json"
        ),
        "file_sha256": "457ca867f406fb6cf4db48bbe9d70340be792b4ee79c38902de112c857b091d2",
        "byte_count": 24_635,
    },
}
PUBLIC_ANCHORS = {
    "balanced_accuracy": 0.2469343816883539,
    "balanced_accuracy_one_sided_95_lower_bound": 0.23014452836846072,
    "hardest_wrong_action_margin": -0.009453551490358742,
    "hardest_wrong_action_margin_one_sided_95_lower_bound": -0.01138311990101325,
    "persistence_log_energy_advantage": -0.14645548512800682,
    "persistence_one_sided_95_lower_bound": -0.1829122861354923,
    "wrong_history_log_energy_advantage": 0.12255093276460897,
    "wrong_history_one_sided_95_lower_bound": 0.11766703087321294,
}
CLAIM_BOUNDARY = [
    "post-hoc development localization only; not citable scientific or promotion evidence",
    "no training, optimizer, checkpoint restoration, data generation, retry, resume, refill, or V3 extension authority",
    "requested-action association only; no requested-versus-executed equivalence or untaken-action causal claim",
    "no architecture-sufficiency, planner-utility, navigation, WM-A, WM-D, G2-G8, deployment, or production claim",
]
EXPECTED_SNAPSHOT_KEYS = {
    "schema",
    "status",
    "citable_as_scientific_evidence",
    "authorizes_retry_or_resume",
    "arm",
    "update",
    "authority_binding",
    "plan_binding",
    "substrate",
    "schedule",
    "metric_vectors",
    "arm_state_dict",
    "optimizer_state_dict",
}
EXPECTED_METRIC_VECTOR_KEYS = {
    "validation_row_indices",
    "validation_factual_energy",
    "validation_persistence_energy",
    "validation_wrong_history_energy",
    "validation_candidate_energy",
    "prediction_tokens",
    "target_tokens",
    "training_row_indices",
    "training_factual_energy",
}
REQUIRED_SOURCE_PATHS = {
    "lewm_package": "lewm/__init__.py",
    "benchmarks_package": "lewm/benchmarks/__init__.py",
    "counterfactual_metrics": "lewm/benchmarks/counterfactual.py",
    "h6_main_pool_census": "lewm/benchmarks/go2_recurrent_jepa_main_pool_census.py",
    "localization_metrics": "lewm/benchmarks/go2_world_model_v3_action_localization_v1.py",
    "three_arm_metrics": "lewm/benchmarks/go2_world_model_existing_pool_three_arm_v1.py",
    "datasets_package": "lewm/datasets/__init__.py",
    "h6_dataset": "lewm/datasets/go2_explicit_plan_discounted_successor_state_v27.py",
    "h6_sequence_contract_v1": "lewm/datasets/go2_recurrent_h4_rgb_sequences.py",
    "h6_sequence_contract_v2": "lewm/datasets/go2_recurrent_h4_rgb_sequences_v2.py",
    "worker": "scripts/extract_go2_world_model_existing_pool_three_arm_v1_action_localization_v1.py",
    "checker": "scripts/check_go2_world_model_existing_pool_three_arm_v1_action_localization_v1.py",
    "external_supervisor": (
        "scripts/run_go2_world_model_existing_pool_three_arm_v1_action_"
        "localization_authorized_v1.py"
    ),
}
REQUIRED_TEST_PATHS = {
    "localization_metric_tests": (
        "lewm/tests/test_go2_world_model_v3_action_localization_v1.py"
    ),
    "worker_checker_tests": (
        "lewm/tests/test_extract_go2_world_model_existing_pool_three_arm_v1_"
        "action_localization_v1.py"
    ),
    "supervisor_tests": (
        "lewm/tests/test_run_go2_world_model_existing_pool_three_arm_v1_"
        "action_localization_authorized_v1.py"
    ),
}
EXACT_CHILD_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "",
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
MAXIMUM_WALL_SECONDS = 1_800
REVIEW_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_action_localization_v1_"
    "independent_source_review_v1"
)
REVIEW_STATUS = "PASS_SOURCE_ONLY_NOT_AUTHORITY"
_HEX = frozenset("0123456789abcdef")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


class LocalizationWorkerError(RuntimeError):
    """The one-shot extraction contract failed closed."""


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            text = json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        else:
            text = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
    except (TypeError, ValueError) as error:
        raise LocalizationWorkerError("document is not finite canonical JSON") from error
    return (text + "\n").encode("utf-8")


def strict_json_bytes(raw: bytes) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise LocalizationWorkerError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LocalizationWorkerError("JSON input is invalid") from error


def _binding_shape(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise LocalizationWorkerError(f"{label} binding fields changed")
    path = value["path"]
    digest = value["file_sha256"]
    count = value["byte_count"]
    if (
        type(path) is not str
        or not Path(path).is_absolute()
        or type(digest) is not str
        or len(digest) != 64
        or any(character not in _HEX for character in digest)
        or type(count) is not int
        or count <= 0
    ):
        raise LocalizationWorkerError(f"{label} binding is invalid")
    return value


def _read_absolute_regular_once(
    binding: Mapping[str, Any],
    *,
    label: str,
) -> bytes:
    selected = _binding_shape(dict(binding), label=label)
    path = Path(selected["path"])
    parts = path.parts
    if not parts or parts[0] != "/" or any(part in {"", ".", ".."} for part in parts[1:]):
        raise LocalizationWorkerError(f"{label} path is not canonical absolute")
    directory_fd = os.open("/", _DIR_FLAGS)
    file_fd: int | None = None
    try:
        for component in parts[1:-1]:
            child_fd = os.open(component, _DIR_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        file_fd = os.open(parts[-1], _READ_FLAGS, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != selected["byte_count"]:
            raise LocalizationWorkerError(f"{label} is not the bound regular file")
        raw = bytearray()
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(file_fd)
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or len(raw) != selected["byte_count"]
            or hashlib.sha256(raw).hexdigest() != selected["file_sha256"]
        ):
            raise LocalizationWorkerError(f"{label} identity changed while opening")
        return bytes(raw)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def file_binding(path: Path) -> dict[str, Any]:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise LocalizationWorkerError(f"binding target is not a regular file: {selected}")
    digest = hashlib.sha256()
    count = 0
    with selected.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
    return {
        "path": str(selected.resolve(strict=True)),
        "file_sha256": digest.hexdigest(),
        "byte_count": count,
    }


def write_immutable_json(path: Path, value: Any) -> dict[str, Any]:
    selected = Path(path)
    if selected.exists() or selected.is_symlink():
        raise LocalizationWorkerError(f"refusing to overwrite {selected}")
    payload = canonical_json_bytes(value, pretty=True)
    with selected.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return file_binding(selected)


def exact_root_inventory(expected: set[str]) -> list[str]:
    try:
        root_stat = ATTEMPT_ROOT.lstat()
    except FileNotFoundError as error:
        raise LocalizationWorkerError("localization attempt root is absent") from error
    if not stat.S_ISDIR(root_stat.st_mode) or ATTEMPT_ROOT.is_symlink():
        raise LocalizationWorkerError("localization attempt root is not a real directory")
    observed: list[str] = []
    with os.scandir(ATTEMPT_ROOT) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise LocalizationWorkerError("localization root contains a non-file")
            observed.append(entry.name)
    if set(observed) != expected or len(observed) != len(expected):
        raise LocalizationWorkerError(
            f"localization root inventory changed: {sorted(observed)}"
        )
    return sorted(observed)


def _git(*arguments: str) -> str:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise LocalizationWorkerError(f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise LocalizationWorkerError(f"git {' '.join(arguments)} failed")
    return completed.stdout


def _validate_commit(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 40 or any(character not in _HEX for character in value):
        raise LocalizationWorkerError(f"{label} is not a full commit identity")
    return value


def _require_strict_commit_ancestor(
    ancestor: str, descendant: str, *, label: str
) -> None:
    if ancestor == descendant:
        raise LocalizationWorkerError(f"{label} commits must be distinct")
    _git("merge-base", "--is-ancestor", ancestor, descendant)


def _require_binding_at_commit(
    binding: Mapping[str, Any], *, commit: str, label: str
) -> None:
    selected = _binding_shape(dict(binding), label=label)
    try:
        relative = Path(selected["path"]).resolve(strict=True).relative_to(
            REPO_ROOT.resolve(strict=True)
        )
    except (OSError, ValueError) as error:
        raise LocalizationWorkerError(
            f"{label} is not a tracked repository path"
        ) from error
    raw = _git_bytes("show", f"{commit}:{relative.as_posix()}")
    if (
        len(raw) != selected["byte_count"]
        or hashlib.sha256(raw).hexdigest() != selected["file_sha256"]
    ):
        raise LocalizationWorkerError(f"{label} differs from frozen commit")


def _validate_frozen_document_binding(
    value: Any,
    *,
    exact_path: Path,
    commit: str,
    label: str,
) -> dict[str, Any]:
    binding = _binding_shape(value, label=label)
    if Path(binding["path"]) != exact_path:
        raise LocalizationWorkerError(f"{label} path changed")
    if file_binding(exact_path) != binding:
        raise LocalizationWorkerError(f"live {label} binding changed")
    _require_binding_at_commit(binding, commit=commit, label=label)
    return binding


def expected_reservation(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    *,
    supervisor_nonce: str,
) -> dict[str, Any]:
    if (
        type(supervisor_nonce) is not str
        or len(supervisor_nonce) != 64
        or any(character not in _HEX for character in supervisor_nonce)
    ):
        raise LocalizationWorkerError("supervisor nonce is invalid")
    runtime = authority["runtime"]
    execution = authority["execution"]
    worker_template = [
        runtime["python_invocation_path"],
        execution["worker_path"],
        "--authority",
        str(AUTHORITY_PATH),
        "--expected-authority-sha256",
        authority_binding["file_sha256"],
        "--expected-authority-byte-count",
        str(authority_binding["byte_count"]),
        "--reservation-sha256",
        "<SUPERVISOR_BOUND_RESERVATION_SHA256>",
        "--reservation-byte-count",
        "<SUPERVISOR_BOUND_RESERVATION_BYTE_COUNT>",
    ]
    checker_template = [
        runtime["python_invocation_path"],
        execution["checker_path"],
        "--manifest",
        str(ATTEMPT_ROOT / "localization.json"),
        "--expected-file-sha256",
        "<WORKER_RESULT_SHA256>",
        "--expected-byte-count",
        "<WORKER_RESULT_BYTE_COUNT>",
        "--authority",
        str(AUTHORITY_PATH),
        "--expected-authority-sha256",
        authority_binding["file_sha256"],
        "--expected-authority-byte-count",
        str(authority_binding["byte_count"]),
    ]
    return {
        "schema": RESERVATION_SCHEMA,
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "attempt_id": ATTEMPT_ID,
        "attempt_root": str(ATTEMPT_ROOT),
        "authority_binding": dict(authority_binding),
        "source_commit": authority["source_commit"],
        "review_commit": authority["review_commit"],
        "execution_head": authority["execution_head"],
        "caps": authority["caps"],
        "supervisor_nonce": supervisor_nonce,
        "worker_command_template": worker_template,
        "checker_command_template": checker_template,
        "maximum_attempts": 1,
        "retry": False,
        "resume": False,
        "refill": False,
        "overwrite": False,
    }


def validate_reservation(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    *,
    reservation_sha256: str,
    reservation_byte_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = {
        "path": str(ATTEMPT_ROOT / "reservation.json"),
        "file_sha256": reservation_sha256,
        "byte_count": reservation_byte_count,
    }
    raw = _read_absolute_regular_once(binding, label="reservation")
    document = strict_json_bytes(raw)
    if not isinstance(document, dict):
        raise LocalizationWorkerError("reservation is not an object")
    expected = expected_reservation(
        authority,
        authority_binding,
        supervisor_nonce=document.get("supervisor_nonce", ""),
    )
    if document != expected:
        raise LocalizationWorkerError("reservation contract changed")
    exact_root_inventory({"reservation.json"})
    return document, binding


def validate_exact_child_environment() -> None:
    if dict(os.environ) != EXACT_CHILD_ENVIRONMENT:
        raise LocalizationWorkerError("child process environment changed")


def load_and_validate_authority(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if Path(path).is_symlink():
        raise LocalizationWorkerError("execution authority must not be a symlink")
    binding = {
        "path": str(Path(path).resolve(strict=True)),
        "file_sha256": expected_sha256,
        "byte_count": expected_byte_count,
    }
    raw = _read_absolute_regular_once(binding, label="execution authority")
    if Path(binding["path"]) != AUTHORITY_PATH:
        raise LocalizationWorkerError("execution authority path changed")
    authority = strict_json_bytes(raw)
    expected_keys = {
        "schema",
        "status",
        "authority_granted",
        "citable_as_scientific_evidence",
        "source_commit",
        "review_commit",
        "execution_head",
        "preregistration_binding",
        "plan_binding",
        "review_binding",
        "attempt",
        "input_bindings",
        "predecessor_evidence_bindings",
        "public_anchors",
        "source_bindings",
        "test_bindings",
        "runtime",
        "caps",
        "access_contract",
        "claim_boundary",
        "execution",
        "authorized_command",
        "external_supervisor",
    }
    if not isinstance(authority, dict) or set(authority) != expected_keys:
        raise LocalizationWorkerError("authority top-level fields changed")
    if (
        authority["schema"] != AUTHORITY_SCHEMA
        or authority["status"] != "AUTHORIZED_ONE_SHOT_READ_ONLY"
        or authority["authority_granted"] is not True
        or authority["citable_as_scientific_evidence"] is not False
    ):
        raise LocalizationWorkerError("authority status or schema changed")
    source_commit = _validate_commit(authority["source_commit"], label="source commit")
    review_commit = _validate_commit(authority["review_commit"], label="review commit")
    execution_head = _validate_commit(authority["execution_head"], label="execution head")
    if _git("rev-parse", "HEAD") != execution_head:
        raise LocalizationWorkerError("execution HEAD differs from authority")
    _require_strict_commit_ancestor(
        source_commit, review_commit, label="source-before-review"
    )
    _require_strict_commit_ancestor(
        review_commit, execution_head, label="review-before-authority"
    )
    _require_binding_at_commit(
        binding, commit=execution_head, label="execution authority"
    )
    preregistration_binding = _validate_frozen_document_binding(
        authority["preregistration_binding"],
        exact_path=PREREGISTRATION_PATH,
        commit=source_commit,
        label="preregistration",
    )
    plan_binding = _validate_frozen_document_binding(
        authority["plan_binding"],
        exact_path=PLAN_PATH,
        commit=source_commit,
        label="plan",
    )
    review_binding = _validate_frozen_document_binding(
        authority["review_binding"],
        exact_path=REVIEW_PATH,
        commit=review_commit,
        label="independent source review",
    )

    expected_attempt = {
        "id": ATTEMPT_ID,
        "root": str(ATTEMPT_ROOT),
        "maximum_attempts": 1,
        "must_be_absent": True,
        "reservation_consumes_attempt": True,
        "retry": False,
        "resume": False,
        "refill": False,
        "overwrite": False,
    }
    if authority["attempt"] != expected_attempt:
        raise LocalizationWorkerError("authority attempt envelope changed")
    if authority["input_bindings"] != {
        "conditioned_update_700_snapshot": SNAPSHOT_BINDING,
        "validation_index": VALIDATION_INDEX_BINDING,
    }:
        raise LocalizationWorkerError("authority input allowlist changed")
    if authority["public_anchors"] != PUBLIC_ANCHORS:
        raise LocalizationWorkerError("authority public anchors changed")
    if authority["predecessor_evidence_bindings"] != PREDECESSOR_EVIDENCE_BINDINGS:
        raise LocalizationWorkerError("authority predecessor evidence changed")

    source_bindings = authority["source_bindings"]
    if not isinstance(source_bindings, dict) or set(source_bindings) != set(REQUIRED_SOURCE_PATHS):
        raise LocalizationWorkerError("authority source closure changed")
    for name, relative in REQUIRED_SOURCE_PATHS.items():
        expected = _binding_shape(source_bindings[name], label=f"source {name}")
        actual = file_binding(REPO_ROOT / relative)
        if actual != expected:
            raise LocalizationWorkerError(f"live source binding changed: {name}")
        _require_binding_at_commit(
            expected, commit=source_commit, label=f"source {name}"
        )
    test_bindings = authority["test_bindings"]
    if not isinstance(test_bindings, dict) or set(test_bindings) != set(
        REQUIRED_TEST_PATHS
    ):
        raise LocalizationWorkerError("authority synthetic-test closure changed")
    for name, relative in REQUIRED_TEST_PATHS.items():
        expected = _binding_shape(test_bindings[name], label=f"test {name}")
        actual = file_binding(REPO_ROOT / relative)
        if actual != expected:
            raise LocalizationWorkerError(f"live test binding changed: {name}")
        _require_binding_at_commit(
            expected, commit=source_commit, label=f"test {name}"
        )

    review = strict_json_bytes(
        _read_absolute_regular_once(
            review_binding, label="independent source review"
        )
    )
    expected_review_keys = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "execution_authority_granted",
        "source_commit",
        "reviewers",
        "review_materialization",
        "preregistration_binding",
        "plan_binding",
        "source_bindings",
        "test_bindings",
        "verification",
        "custody",
        "findings",
        "claim_boundary",
    }
    if (
        not isinstance(review, dict)
        or set(review) != expected_review_keys
        or review["schema"] != REVIEW_SCHEMA
        or review["status"] != REVIEW_STATUS
        or review["citable_as_scientific_evidence"] is not False
        or review["execution_authority_granted"] is not False
        or review["source_commit"] != source_commit
        or review["reviewers"]
        != [
            {
                "kind": "independent_subagent",
                "task_name": "/root/localization_custody_final",
                "scope": "custody_execution_and_terminal_chain",
                "separate_from_source_author": True,
                "separate_from_execution_authorizer": True,
            },
            {
                "kind": "independent_subagent",
                "task_name": "/root/localization_science_final",
                "scope": "scientific_metrics_routing_and_preregistration",
                "separate_from_source_author": True,
                "separate_from_execution_authorizer": True,
            },
        ]
        or review["review_materialization"]
        != {
            "date": "2026-08-01",
            "method": "independent_live_source_only_audits_then_frozen_binding_review",
            "reviewed_after_focused_tests": True,
            "reviewed_before_execution_authority": True,
        }
        or review["preregistration_binding"] != preregistration_binding
        or review["plan_binding"] != plan_binding
        or review["source_bindings"] != source_bindings
        or review["test_bindings"] != test_bindings
        or review["findings"] != []
        or review["claim_boundary"] != CLAIM_BOUNDARY
    ):
        raise LocalizationWorkerError("independent source review contract changed")
    verification = review["verification"]
    if verification != {
        "focused_test_count": 14,
        "focused_tests_passed": 14,
        "strict_json_plan_passed": True,
        "python_compilation_passed": True,
        "git_diff_check_passed": True,
        "required_source_path_count": len(REQUIRED_SOURCE_PATHS),
        "required_test_path_count": len(REQUIRED_TEST_PATHS),
        "recursive_source_closure_complete": True,
        "source_plan_preregistration_consistent": True,
        "strict_commit_order_and_blob_binding_enforced": True,
        "supervised_worker_checker_terminal_chain_enforced": True,
        "attempt_root_absent_at_review": True,
    }:
        raise LocalizationWorkerError("independent source verification changed")
    if review["custody"] != {
        "real_snapshot_payload_opened": False,
        "real_validation_index_opened": False,
        "pack_payloads_opened": False,
        "rgb_payloads_opened": False,
        "other_checkpoint_or_snapshot_payloads_opened": False,
        "train_index_opened": False,
        "heldout_or_sealed_opened": False,
        "network_access_used": False,
        "synthetic_fixtures_only": True,
        "attempt_reserved": False,
        "execution_authority_granted": False,
    }:
        raise LocalizationWorkerError("independent source-review custody changed")

    runtime = authority["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "python_invocation_path",
        "python_version",
        "torch_version",
        "numpy_version",
        "cpu_only",
        "gpu_visibility_must_be_empty",
        "environment",
    }:
        raise LocalizationWorkerError("runtime contract fields changed")
    observed_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if (
        Path(sys.executable).absolute()
        != Path(runtime["python_invocation_path"]).absolute()
        or runtime["python_version"] != observed_python
        or runtime["torch_version"] != torch.__version__
        or runtime["numpy_version"] != np.__version__
        or runtime["cpu_only"] is not True
        or runtime["gpu_visibility_must_be_empty"] is not True
        or runtime["environment"] != EXACT_CHILD_ENVIRONMENT
    ):
        raise LocalizationWorkerError("runtime identity changed")
    if any(
        name not in os.environ or os.environ[name] != ""
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    ):
        raise LocalizationWorkerError("GPU visibility is not empty")

    if authority["caps"] != {
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "maximum_gpu_seconds": 0,
        "maximum_training_updates": 0,
        "maximum_optimizer_steps": 0,
    }:
        raise LocalizationWorkerError("authority caps changed")
    execution = authority["execution"]
    if execution != {
        "worker_path": str(Path(__file__).resolve()),
        "checker_path": str(
            REPO_ROOT
            / "scripts/check_go2_world_model_existing_pool_three_arm_v1_"
            "action_localization_v1.py"
        ),
        "supervisor_path": str(SUPERVISOR_PATH),
    }:
        raise LocalizationWorkerError("authority execution paths changed")
    expected_command = [
        runtime["python_invocation_path"],
        str(SUPERVISOR_PATH),
        "--authority",
        str(AUTHORITY_PATH),
        "--expected-authority-sha256",
        "<CALLER_BOUND_AUTHORITY_SHA256>",
        "--expected-authority-byte-count",
        "<CALLER_BOUND_AUTHORITY_BYTE_COUNT>",
    ]
    if authority["authorized_command"] != {"argv_template": expected_command}:
        raise LocalizationWorkerError("authorized supervisor command changed")
    external = authority["external_supervisor"]
    if (
        not isinstance(external, dict)
        or set(external) != {"source_binding", "terminal_reviewer"}
        or external["source_binding"] != source_bindings["external_supervisor"]
        or type(external["terminal_reviewer"]) is not str
        or not external["terminal_reviewer"].strip()
    ):
        raise LocalizationWorkerError("external supervisor contract changed")

    access = authority["access_contract"]
    expected_access = {
        "snapshot_content_open_count": 1,
        "validation_index_content_open_count": 1,
        "pack_payload_open_count": 0,
        "rgb_open_count": 0,
        "other_snapshot_or_checkpoint_open_count": 0,
        "train_index_open_count": 0,
        "model_forward_count": 0,
        "training_update_count": 0,
        "optimizer_step_count": 0,
        "gpu_seconds": 0,
        "network_access_count": 0,
        "write_beneath_v3_attempt_root": False,
    }
    if access != expected_access:
        raise LocalizationWorkerError("authority access contract changed")
    if authority["claim_boundary"] != CLAIM_BOUNDARY:
        raise LocalizationWorkerError("authority claim boundary changed")
    return authority, binding


def load_bound_snapshot_metric_vectors() -> tuple[dict[str, torch.Tensor | list[int]], dict[str, Any]]:
    raw = _read_absolute_regular_once(SNAPSHOT_BINDING, label="conditioned u700 snapshot")
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise LocalizationWorkerError("weights-only snapshot load failed") from error
    finally:
        del raw
    if not isinstance(payload, dict) or set(payload) != EXPECTED_SNAPSHOT_KEYS:
        raise LocalizationWorkerError("snapshot payload fields changed")
    if (
        payload["schema"] != SNAPSHOT_SCHEMA
        or payload["status"] != "INERT_AUDIT_SNAPSHOT"
        or payload["citable_as_scientific_evidence"] is not False
        or payload["authorizes_retry_or_resume"] is not False
        or payload["arm"] != "conditioned"
        or payload["update"] != 700
        or payload["authority_binding"] != V3_INTERNAL_AUTHORITY_BINDING
        or payload["plan_binding"] != V3_INTERNAL_PLAN_BINDING
    ):
        raise LocalizationWorkerError("snapshot envelope or V3 binding changed")
    substrate = payload["substrate"]
    if not isinstance(substrate, dict) or any(
        substrate.get(name) != V3_SUBSTRATE_SHA256
        for name in ("encoder_sha256", "target_sha256")
    ):
        raise LocalizationWorkerError("snapshot frozen-substrate identity changed")
    metric_vectors = payload["metric_vectors"]
    if not isinstance(metric_vectors, dict) or set(metric_vectors) != EXPECTED_METRIC_VECTOR_KEYS:
        raise LocalizationWorkerError("snapshot metric-vector fields changed")
    if metric_vectors["validation_row_indices"] != list(range(h6.VALIDATION_INDEX_ROWS)):
        raise LocalizationWorkerError("snapshot validation row order changed")
    expected_shapes = {
        "validation_factual_energy": (h6.VALIDATION_INDEX_ROWS,),
        "validation_persistence_energy": (h6.VALIDATION_INDEX_ROWS,),
        "validation_wrong_history_energy": (h6.VALIDATION_INDEX_ROWS,),
        "validation_candidate_energy": (
            h6.VALIDATION_INDEX_ROWS,
            len(localization_metrics.TRAIN_EXPOSURE_COUNTS),
        ),
    }
    selected: dict[str, torch.Tensor | list[int]] = {
        "validation_row_indices": list(metric_vectors["validation_row_indices"])
    }
    for name, shape in expected_shapes.items():
        value = metric_vectors[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float64
            or tuple(value.shape) != shape
            or not bool(torch.isfinite(value).all())
            or (
                name == "validation_candidate_energy"
                and bool((value < 0.0).any())
            )
            or (
                name != "validation_candidate_energy"
                and not bool((value > 0.0).all())
            )
        ):
            raise LocalizationWorkerError(f"snapshot vector contract changed: {name}")
        selected[name] = value.detach().clone()
    contract = {
        "schema": payload["schema"],
        "status": payload["status"],
        "arm": payload["arm"],
        "update": payload["update"],
        "metric_vector_keys": sorted(metric_vectors),
        "model_or_optimizer_state_consumed_computationally": False,
        "model_or_optimizer_state_restored_or_emitted": False,
    }
    del metric_vectors
    del payload
    return selected, contract


def _assert_public_anchor_reproduction(localization: Mapping[str, Any]) -> dict[str, float]:
    action = localization["action_identification"]
    controls = localization["registered_control_reproduction"]
    observed = {
        "balanced_accuracy": action["scene_family_balanced_accuracy"],
        "balanced_accuracy_one_sided_95_lower_bound": action[
            "balanced_accuracy_bootstrap_lower_95"
        ],
        "hardest_wrong_action_margin": action["hardest_action_margin"],
        "hardest_wrong_action_margin_one_sided_95_lower_bound": action[
            "hardest_margin_bootstrap_lower_95"
        ],
        "persistence_log_energy_advantage": controls["persistence"][
            "macro_log_advantage"
        ],
        "persistence_one_sided_95_lower_bound": controls["persistence"][
            "bootstrap_lower_95"
        ],
        "wrong_history_log_energy_advantage": controls["wrong_history"][
            "macro_log_advantage"
        ],
        "wrong_history_one_sided_95_lower_bound": controls["wrong_history"][
            "bootstrap_lower_95"
        ],
    }
    if set(observed) != set(PUBLIC_ANCHORS) or any(
        not math.isclose(float(observed[name]), expected, rel_tol=0.0, abs_tol=1.0e-15)
        for name, expected in PUBLIC_ANCHORS.items()
    ):
        raise LocalizationWorkerError("localization failed public-anchor reproduction")
    return {name: float(observed[name]) for name in sorted(observed)}


def execute(authority: Mapping[str, Any], authority_binding: Mapping[str, Any]) -> dict[str, Any]:
    vectors, snapshot_contract = load_bound_snapshot_metric_vectors()
    validation_rows, validation_audit = h6.load_bound_index(REPO_ROOT, role="val")
    if validation_audit["rgb_open_count"] != 0:
        raise LocalizationWorkerError("validation metadata loader followed RGB")
    localization = localization_metrics.localize_action_and_controls(
        candidate_energies=vectors["validation_candidate_energy"],
        factual_energy=vectors["validation_factual_energy"],
        persistence_energy=vectors["validation_persistence_energy"],
        wrong_history_energy=vectors["validation_wrong_history_energy"],
        validation_rows=validation_rows,
    )
    del vectors
    anchors = _assert_public_anchor_reproduction(localization)
    return {
        "schema": RESULT_SCHEMA,
        "status": "PASS_COMPLETE_READ_ONLY_LOCALIZATION",
        "citable_as_scientific_evidence": False,
        "authorizes_training_or_data_generation": False,
        "authorizes_retry_or_resume": False,
        "attempt": {
            "id": ATTEMPT_ID,
            "root": str(ATTEMPT_ROOT),
            "consumed": True,
            "retry": False,
            "resume": False,
            "refill": False,
            "overwrite": False,
        },
        "source_commit": authority["source_commit"],
        "review_commit": authority["review_commit"],
        "execution_head": authority["execution_head"],
        "authority_binding": dict(authority_binding),
        "input_bindings": authority["input_bindings"],
        "snapshot_contract": snapshot_contract,
        "validation_index_audit": validation_audit,
        "public_anchor_reproduction": anchors,
        "localization": localization,
        "access_accounting": authority["access_contract"],
        "custody": {
            "snapshot_bytes_opened": True,
            "validation_index_bytes_opened": True,
            "pack_payloads_opened": False,
            "rgb_paths_followed": False,
            "other_snapshots_or_checkpoints_opened": False,
            "train_index_opened": False,
            "model_forward_performed": False,
            "training_or_optimizer_step_performed": False,
            "model_or_optimizer_state_restored_or_emitted": False,
            "network_access_used": False,
            "write_beneath_v3_attempt_root": False,
        },
        "claim_boundary": list(authority["claim_boundary"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    parser.add_argument("--reservation-sha256", required=True)
    parser.add_argument("--reservation-byte-count", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    validate_exact_child_environment()
    authority, authority_binding = load_and_validate_authority(
        arguments.authority,
        expected_sha256=arguments.expected_authority_sha256,
        expected_byte_count=arguments.expected_authority_byte_count,
    )
    _reservation, reservation_binding = validate_reservation(
        authority,
        authority_binding,
        reservation_sha256=arguments.reservation_sha256,
        reservation_byte_count=arguments.reservation_byte_count,
    )
    try:
        result = execute(authority, authority_binding)
        result["reservation_binding"] = reservation_binding
        write_immutable_json(ATTEMPT_ROOT / "localization.json", result)
        exact_root_inventory({"reservation.json", "localization.json"})
    except BaseException as error:
        failure = {
            "schema": FAILURE_SCHEMA,
            "status": "TERMINAL_FAILURE",
            "attempt_id": ATTEMPT_ID,
            "attempt_root": str(ATTEMPT_ROOT),
            "authority_binding": authority_binding,
            "reservation_binding": reservation_binding,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "retry": False,
            "resume": False,
            "refill": False,
            "overwrite": False,
        }
        try:
            write_immutable_json(ATTEMPT_ROOT / "failure.json", failure)
        finally:
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
