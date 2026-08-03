#!/usr/bin/env python3
"""Run and fresh-process replay the one-shot dense V-JEPA ceiling V1."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
from io import BytesIO
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import time
from types import MappingProxyType
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image
from PIL import __version__ as PILLOW_VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_dense_vjepa2_1_physical_interface_ceiling_v1 as evaluator,
)
from lewm.benchmarks import (  # noqa: E402
    go2_dinov2_physical_readout_calibration_v1 as feature_plans,
)
from lewm.benchmarks import (  # noqa: E402
    go2_world_model_counterfactual_pilot_v1 as contract,
)
from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as consumer  # noqa: E402
from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (  # noqa: E402
    read_bound_rgb_bytes_v1,
)


SCHEMA = "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_result_v1"
TERMINAL_SCHEMA = "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_terminal_v1"
RESERVATION_SCHEMA = (
    "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_reservation_v1"
)
EVAL_CACHE_SCHEMA = (
    "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_eval_cache_v1"
)
EVAL_CACHE_RECEIPT_SCHEMA = (
    "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_eval_cache_receipt_v1"
)
REPLAY_SCHEMA = "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_replay_v1"
REPLAY_STATUS = "PASS_EXACT_FRESH_PROCESS_CACHE_ONLY_REPLAY"
AUTHORITY_SCHEMA = (
    "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_execution_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_DENSE_VJEPA2_1_PHYSICAL_INTERFACE_CEILING_ATTEMPT"
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_DENSE_VJEPA2_1_CEILING_SOURCE_REVIEW"

PASS_STATUS = evaluator.PASS_STATUS
STOP_STATUS = evaluator.STOP_STATUS
FAIL_STATUS = evaluator.INFRASTRUCTURE_FAILURE_STATUS
TERMINAL_STATUSES = frozenset((PASS_STATUS, STOP_STATUS, FAIL_STATUS))

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_"
    "preregistration_2026-08-03.md"
)
PREREGISTRATION_SHA256 = (
    "ef5c687d509929169280a456618e92e92f2a072a646bc292be3d16850f801ad0"
)
PREREGISTRATION_BYTE_COUNT = 20_816
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_"
    "source_review_2026-08-03.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_dense_vjepa2_1_physical_interface_ceiling_v1/attempt_v1"
)

POSTHOC_ROOT = REPO_ROOT / (
    ".generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1"
)
PHYSICS_ROOT = REPO_ROOT / (
    ".generated/dev/lewm-go2-wm-bounded-branch-experiment-integrity-replacement-v1"
)
TRAIN_CACHE = REPO_ROOT / (
    ".generated/dev/go2_matched_branch_successor_screen_v1/attempt_v1/"
    "features/vjepa2_1.pt"
)
TRAIN_CACHE_RECEIPT = TRAIN_CACHE.with_suffix(".json")
VJEPA_CHECKPOINT = Path(
    "/home/andrewknowles/.cache/vjepa2_1_vitb_dist_vitG_384.pt"
)
VJEPA_REPOSITORY = Path(
    "/home/andrewknowles/.cache/vjepa2-204698b45b3712590f06245fbfba32d3be539812"
)
VJEPA_REPOSITORY_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA_REPOSITORY_TREE = "dd6cfc1e792158510b983d827cb2e84f47fd5706"
POSTHOC_MANIFEST_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_posthoc_join_admission_manifest_v1"
)
POSTHOC_MANIFEST_STATUS = "COMPLETE_POSTHOC_METADATA_DERIVATION_PENDING_REVIEW"


def _vjepa_source_binding_v1(
    relative: str, sha256: str, byte_count: int
) -> dict[str, Any]:
    return {
        "path": str((VJEPA_REPOSITORY / relative).resolve()),
        "sha256": sha256,
        "byte_count": byte_count,
    }


VJEPA_TRANSITIVE_SOURCE_BINDINGS = {
    "hubconf.py": _vjepa_source_binding_v1(
        "hubconf.py", "6a61c46a80c82ed10331a19822d58e9a19f062e52845e1787fc810979ff03c7b", 543
    ),
    "evals/hub/preprocessor.py": _vjepa_source_binding_v1(
        "evals/hub/preprocessor.py", "1f0776605ef69689a99ac79527b391712680e96c0d95735035355ae0e0aea58b", 510
    ),
    "src/hub/backbones.py": _vjepa_source_binding_v1(
        "src/hub/backbones.py", "391cdde1e9a1da47cb8094bbea5fbbe8acac0135b27e82f1a6ab19c0b39cc692", 10_164
    ),
    "app/vjepa_2_1/models/vision_transformer.py": _vjepa_source_binding_v1(
        "app/vjepa_2_1/models/vision_transformer.py", "d2932eabeba684d8f558302a13cfd4be70a0170ee5112f5a794652d0a29089b9", 18_195
    ),
    "app/vjepa_2_1/models/predictor.py": _vjepa_source_binding_v1(
        "app/vjepa_2_1/models/predictor.py", "30111720eb90c6dcdde44521cea53cc736020f984e5573150fd3dc7b4acc05d8", 10_679
    ),
    "app/vjepa_2_1/models/utils/modules.py": _vjepa_source_binding_v1(
        "app/vjepa_2_1/models/utils/modules.py", "64be6a87bd9f18d385f4e44186db3347d1665e18a1f0511d51d3b305531562e2", 16_963
    ),
    "app/vjepa_2_1/models/utils/patch_embed.py": _vjepa_source_binding_v1(
        "app/vjepa_2_1/models/utils/patch_embed.py", "29e11ab97ab3ccdef107d6a7d0d7b374b58e712076cc3561f07b7e603c9b5165", 1_883
    ),
    "src/masks/utils.py": _vjepa_source_binding_v1(
        "src/masks/utils.py", "833f111a0fa5ffdbd3a6412e2dace2517c3c178f49c14f8bb631d9f6a070dfd0", 660
    ),
    "src/utils/tensors.py": _vjepa_source_binding_v1(
        "src/utils/tensors.py", "782b58bd2af456e184750e5318ab773105108383f61b280fe4c7a90f46add2c8", 1_832
    ),
}

EINOPS_SITE_PACKAGES = Path(
    "/home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/"
    "lib/python3.12/site-packages"
)
EINOPS_MODULE_ORIGIN = EINOPS_SITE_PACKAGES / "einops/__init__.py"
EINOPS_DISTRIBUTION_ROOT = EINOPS_SITE_PACKAGES / "einops-0.8.1.dist-info"


def _einops_binding_v1(relative: str, sha256: str, byte_count: int) -> dict[str, Any]:
    return {
        "path": str((EINOPS_SITE_PACKAGES / relative).resolve()),
        "sha256": sha256,
        "byte_count": byte_count,
    }


EINOPS_RUNTIME_SOURCE_BINDINGS = {
    "einops/__init__.py": _einops_binding_v1(
        "einops/__init__.py", "acacaf13ae1b60c38c5c01b811f30c4951b1805674c79d5db7cda946cc389471", 422
    ),
    "einops/einops.py": _einops_binding_v1(
        "einops/einops.py", "b17bc3f12585aae7f38b24092913e67c018754dd5c303bd838f36e67c2f55d05", 37_569
    ),
    "einops/_backends.py": _einops_binding_v1(
        "einops/_backends.py", "ec10222967412cbbc08abc9f8700dbc318251835bd0e1e4534c4f51114aecf19", 21_281
    ),
    "einops/parsing.py": _einops_binding_v1(
        "einops/parsing.py", "c5ba9cbf045e2e244e12e7287a0676d1642210794b834bb1a3fbd8a1ecca07fe", 6_746
    ),
    "einops/packing.py": _einops_binding_v1(
        "einops/packing.py", "d7037dbcc6be728ab5462391fcdbb6697aa0224ac37f78fe37543e60dd3ec56a", 7_650
    ),
    "einops/_torch_specific.py": _einops_binding_v1(
        "einops/_torch_specific.py", "c8c6907aa0198412d647543e26feae4483497f344e84b6feaf328a7afa5e7d45", 4_138
    ),
}
EINOPS_DISTRIBUTION_METADATA_BINDINGS = {
    "einops-0.8.1.dist-info/METADATA": _einops_binding_v1(
        "einops-0.8.1.dist-info/METADATA", "79529559045603ccd27713c844f2b57247c48e93135077bdb904bc53ed189403", 13_451
    ),
    "einops-0.8.1.dist-info/RECORD": _einops_binding_v1(
        "einops-0.8.1.dist-info/RECORD", "b4b231ccd46f7891c44ea965d03d5118d2c43042422a948f64ab0913e0c27583", 4_051
    ),
    "einops-0.8.1.dist-info/WHEEL": _einops_binding_v1(
        "einops-0.8.1.dist-info/WHEEL", "aad0b0a12256807936d52d4a6f88a1773236ae527564a688bab4e3fe780e8724", 87
    ),
    "einops-0.8.1.dist-info/INSTALLER": _einops_binding_v1(
        "einops-0.8.1.dist-info/INSTALLER", "ceebae7b8927a3227e5303cf5e0f1f7b34bb542ad7250ac03fbcde36ec2f1508", 4
    ),
}


def einops_dependency_v1() -> dict[str, Any]:
    return {
        "version": "0.8.1",
        "module_origin": str(EINOPS_MODULE_ORIGIN.resolve()),
        "distribution_root": str(EINOPS_DISTRIBUTION_ROOT.resolve()),
        "runtime_source_bindings": EINOPS_RUNTIME_SOURCE_BINDINGS,
        "distribution_metadata_bindings": EINOPS_DISTRIBUTION_METADATA_BINDINGS,
    }

ROLE_STATE_COUNT = 128
ROLE_SCENE_COUNT = 16
ROLE_ARTIFACT_COUNT = 1_536
TOKEN_SHAPE = (256, 768)
MAX_TOKEN_NORM_ERROR = 2.0e-3
EVAL_BATCH_SIZE = 4
STATE_RECEIPT_COUNT = 256
EVAL_CONTEXT_COUNT = 384
EVAL_SUCCESSOR_COUNT = 1_152

OUTPUT_NAMES = (
    "reservation.json",
    "vjepa2_1_eval.pt",
    "vjepa2_1_eval.json",
    "ceiling_checkpoint.pt",
    "evaluation.json",
    "replay.json",
    "result.json",
    "terminal.json",
)

SOURCE_PATHS = {
    "action_regret_evaluator": REPO_ROOT
    / "scripts/evaluate_go2_world_model_counterfactual_action_regret_v1.py",
    "counterfactual_benchmark_contract": REPO_ROOT
    / "lewm/benchmarks/go2_world_model_counterfactual_pilot_v1.py",
    "pilot_consumer": REPO_ROOT
    / "lewm/datasets/go2_world_model_counterfactual_pilot_v1.py",
    "feature_plan_evaluator": REPO_ROOT
    / "lewm/benchmarks/go2_dinov2_physical_readout_calibration_v1.py",
    "dense_shared_evaluator": REPO_ROOT
    / "lewm/benchmarks/go2_dinov2_dense_shared_spatial_readout_calibration_v1.py",
    "dense_shared_model": REPO_ROOT
    / "lewm/models/go2_dinov2_dense_shared_spatial_readout_calibration_v1.py",
    "physical_evaluator": REPO_ROOT
    / "lewm/benchmarks/go2_matched_branch_physical_outcome_screen_v1.py",
    "physical_model": REPO_ROOT
    / "lewm/models/go2_matched_branch_physical_outcome_screen_v1.py",
    "ceiling_evaluator": Path(evaluator.__file__).resolve(),
    "ceiling_runner": Path(__file__).resolve(),
    "ceiling_evaluator_test": REPO_ROOT
    / "lewm/tests/test_go2_dense_vjepa2_1_physical_interface_ceiling_v1.py",
    "ceiling_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_dense_vjepa2_1_physical_interface_ceiling_v1.py",
}

SOURCE_REVIEW_FIELDS = frozenset(
    {
        "schema",
        "status",
        "review_date",
        "reviewer",
        "protected_material_opened",
        "preregistration_binding",
        "source_bindings",
        "encoder_source",
        "checks",
        "findings",
    }
)
SOURCE_REVIEW_CHECKS = frozenset(
    {
        "frozen_preregistration_binding_exact",
        "source_closure_complete_exact_and_committed",
        "all_20_scientific_inputs_and_10_lineage_witnesses_complete_and_exact",
        "narrow_loader_opens_only_256_state_receipts_and_no_render_receipts",
        "train_eval_roles_scenes_artifacts_and_plan_identities_exact",
        "eval_rgb_closed_set_nofollow_hash_pixel_and_once_only",
        "vjepa_source_checkpoint_shim_preprocessing_and_output_exact",
        "fit_checkpoint_precedes_eval_rgb_access",
        "pca_readout_training_arms_controls_metrics_and_gates_exact",
        "retained_physical_uses_published_evaluation_without_checkpoint",
        "fresh_process_replay_retrains_from_both_caches_without_rgb_or_encoder",
        "exact_eight_file_output_and_failure_terminal_fail_closed",
        "no_collection_protected_retry_resume_or_successor_path",
        "focused_tests_passed",
        "compile_and_whitespace_checks_passed",
    }
)

PERMISSION_FIELDS = frozenset(
    {
        "primary_eval_rgb_access",
        "primary_vjepa2_1_encoder_execution",
        "primary_dense_readout_fitting",
        "one_eval_cache_creation",
        "required_eval_frames_decoded",
        "required_eval_frames_encoded",
        "required_eval_context_frames",
        "required_eval_successor_frames",
        "train_rgb_access",
        "other_encoder_execution",
        "second_or_repeated_extraction",
        "replay_rgb_access",
        "replay_extraction",
        "replay_encoder_execution",
        "collection",
        "protected_access",
        "heldout_access",
        "sealed_access",
        "retry",
        "resume",
        "extension",
        "replacement_attempt",
        "downstream_successor_execution",
    }
)


class DenseVJEPACeilingRunnerError(RuntimeError):
    """Raised when a one-shot execution or custody contract changes."""


def canonical_bytes_v1(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DenseVJEPACeilingRunnerError("document is not finite canonical JSON") from error


def _reject_protected(path: Path, *, label: str) -> None:
    for part in path.parts:
        lowered = part.lower()
        if (
            lowered == "sealed_test.json"
            or lowered == "sealed"
            or lowered.startswith("sealed_")
            or lowered in {"heldout", "held_out", "held-out"}
            or lowered.startswith(("heldout_", "held_out_", "held-out-"))
            or lowered == "protected"
            or lowered.startswith("protected_")
        ):
            raise DenseVJEPACeilingRunnerError(f"{label} names protected material")


def _safe_path(path: Path, *, label: str, must_exist: bool = True) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    _reject_protected(selected, label=label)
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise DenseVJEPACeilingRunnerError(f"{label} traverses a symlink")
        if not cursor.exists():
            if must_exist:
                raise DenseVJEPACeilingRunnerError(f"{label} is absent")
            break
    if must_exist and not selected.exists():
        raise DenseVJEPACeilingRunnerError(f"{label} is absent")
    return selected


def file_binding_v1(path: Path) -> dict[str, Any]:
    selected = _safe_path(path, label="bound file")
    if not selected.is_file():
        raise DenseVJEPACeilingRunnerError("bound path is not a regular file")
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(selected), "sha256": digest.hexdigest(), "byte_count": size}


def _binding(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256,
        "byte_count": byte_count,
    }


def _normalize_inert_binding(
    value: object, *, base: Path | None = None, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DenseVJEPACeilingRunnerError(f"{label} binding is malformed")
    if set(value) == {"path", "sha256", "byte_count"}:
        digest = value.get("sha256")
    elif set(value) == {"path", "file_sha256", "byte_count"}:
        digest = value.get("file_sha256")
    else:
        raise DenseVJEPACeilingRunnerError(f"{label} binding fields changed")
    raw_path = value.get("path")
    byte_count = value.get("byte_count")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(byte_count) is not int
        or byte_count <= 0
    ):
        raise DenseVJEPACeilingRunnerError(f"{label} binding is malformed")
    path = Path(raw_path)
    if not path.is_absolute():
        if base is None:
            raise DenseVJEPACeilingRunnerError(f"{label} relative binding has no root")
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise DenseVJEPACeilingRunnerError(f"{label} binding escapes its root")
        path = base.joinpath(*relative.parts)
    return {
        "path": str(Path(os.path.abspath(os.fspath(path)))),
        "sha256": digest,
        "byte_count": byte_count,
    }


def _require_binding(value: object, *, label: str) -> dict[str, Any]:
    expected = _normalize_inert_binding(value, label=label)
    if file_binding_v1(Path(expected["path"])) != expected:
        raise DenseVJEPACeilingRunnerError(f"{label} binding changed")
    return expected


def _read_bound_json(
    path: Path, *, expected_sha256: str, expected_byte_count: int, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, binding = _read_bound_bytes_once_v1(
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label=label,
    )
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DenseVJEPACeilingRunnerError(f"{label} is not valid JSON") from error
    if not isinstance(document, Mapping):
        raise DenseVJEPACeilingRunnerError(f"{label} is not a JSON object")
    canonical_bytes_v1(document)
    return dict(document), binding


def _read_bound_bytes_once_v1(
    path: Path, *, expected_sha256: str, expected_byte_count: int, label: str
) -> tuple[bytes, dict[str, Any]]:
    selected = _safe_path(path, label=label)
    descriptor = None
    try:
        descriptor = os.open(
            selected,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DenseVJEPACeilingRunnerError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise DenseVJEPACeilingRunnerError(f"cannot safely open {label}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or digest.hexdigest() != expected_sha256
        or byte_count != expected_byte_count
    ):
        raise DenseVJEPACeilingRunnerError(f"{label} caller binding changed")
    raw = b"".join(chunks)
    binding = {
        "path": str(selected),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }
    return raw, binding


def _load_bound_torch_once_v1(
    binding: object, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _normalize_inert_binding(binding, label=label)
    selected = _safe_path(Path(expected["path"]), label=label)
    descriptor = None
    try:
        descriptor = os.open(
            selected,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise DenseVJEPACeilingRunnerError(f"{label} is not a regular file")
            digest = hashlib.sha256()
            byte_count = 0
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            if (
                digest.hexdigest() != expected["sha256"]
                or byte_count != expected["byte_count"]
            ):
                raise DenseVJEPACeilingRunnerError(f"{label} binding changed")
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=True)
            after = os.fstat(handle.fileno())
    except DenseVJEPACeilingRunnerError:
        raise
    except Exception as error:
        raise DenseVJEPACeilingRunnerError(f"{label} is not a safe Torch payload") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or not isinstance(payload, Mapping)
    ):
        raise DenseVJEPACeilingRunnerError(f"{label} changed while loading")
    return dict(payload), expected


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_bytes_v1(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_torch_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fixed_input_bindings_v1() -> dict[str, dict[str, Any]]:
    values = {
        "posthoc_manifest": (POSTHOC_ROOT / "manifest.json", "87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e", 11_964),
        "posthoc_terminal": (POSTHOC_ROOT / "terminal.json", "a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56", 1_250),
        "posthoc_terminal_review": (REPO_ROOT / "docs/lewm_go2_world_model_bounded_branch_posthoc_join_admission_v1_terminal_review_2026-08-02.json", "bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669", 2_844),
        "posthoc_rgb_manifest": (POSTHOC_ROOT / "rgb_manifest.json", "5e03afa7665ffef54a1cab5e37135a18d42761bc844ecefacaa433f75a1b1f7e", 1_880_307),
        "posthoc_train_rows": (POSTHOC_ROOT / "train.jsonl", "edc6f88bb105c39575477fbfbb0224bf0312cf5ee3e90551f86a9c11c2ebb447", 30_432_624),
        "posthoc_eval_rows": (POSTHOC_ROOT / "eval.jsonl", "531debbc431f2f8afc83a491b491b8822134c831b16ca4d283fe1e7f4ba07768", 30_411_588),
        "stored_task_relevance_result": (REPO_ROOT / "docs/lewm_go2_world_model_visual_domain_parity_task_relevant_input_adequacy_result_v1_2026-08-02.json", "5094104ac29b4652cd577015c5fbf23b42f0768c78a205cbf07a77d992339ca7", 94_165),
        "stored_task_relevance_review": (REPO_ROOT / "docs/lewm_go2_world_model_visual_domain_parity_task_relevant_input_adequacy_independent_review_v1_2026-08-02.json", "29eb00a486604824effb56502194855553f87c81a9691d4075a5810273c92ca9", 2_080),
        "physics_result": (PHYSICS_ROOT / "physics_result.json", "25caf0a5d4c69e99559a663aa4cae96fb23ef191ccf34486804c3f2243553314", 183_320),
        "physics_receipt_check": (PHYSICS_ROOT / "physics_receipt_check.json", "faeb50293bc684e35b6d725b027983ad0110e739db2d7b1aca1926e89a547dc6", 892),
        "collection_terminal": (PHYSICS_ROOT / "terminal_supervision.json", "f7d2796139645892d22ad6bb99d26caffc2b5c3dcac2a655b1883b299d22bff4", 12_949),
        "collection_plan": (REPO_ROOT / "docs/lewm_go2_world_model_bounded_branch_integrity_replacement_v1_exact_plan_2026-08-02.json", "8fe34054bb9ae709b6a8ecfea5fdae55c742d1b2e22af3c289d27a77f11c66ef", 343_973),
        "calibration_receipt": (REPO_ROOT / ".generated/dev/lewm-go2-wm-counterfactual-calibration-v3-textured-v03-posthoc-analysis-v1/calibration_receipt.json", "58d1291ede7ee03a93d68eb7cec80c9322c47cd0b1d5fd1c41bf8f4b49ad484e", 72_475),
        "train_vjepa_cache": (TRAIN_CACHE, "3549855ea857906dfe3a4b55fc817633b5114b2457f8facaa4fa87f9eddd798b", 604_097_648),
        "train_vjepa_receipt": (TRAIN_CACHE_RECEIPT, "5d4f8a82d10a33c21b41f1543d6f56b3a230a38f67b02d3f8e7330a8d30180f5", 1_822),
        "vjepa_checkpoint": (VJEPA_CHECKPOINT, "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d", 1_664_223_428),
        "physical_evaluation": (REPO_ROOT / ".generated/dev/go2_matched_branch_physical_outcome_screen_v1/attempt_v2_integrity_replacement_v1/evaluation.json", "4320b80b20a1f347b1dbc6a7c026bb868820db21edbdcf1053470a400e19cec1", 1_755_424),
        "physical_result": (REPO_ROOT / ".generated/dev/go2_matched_branch_physical_outcome_screen_v1/attempt_v2_integrity_replacement_v1/result.json", "a2ba2c3ca7881af54b3553b342b36ea72e3f7ca9b858a5eef4102ae9f7b643ee", 1_769_042),
        "physical_terminal": (REPO_ROOT / ".generated/dev/go2_matched_branch_physical_outcome_screen_v1/attempt_v2_integrity_replacement_v1/terminal.json", "6eb16ea5fa3f9f1e6090eeddc47aace7dd5b9fee7807a56ed84bc7aa0fba2830", 642),
        "physical_terminal_review": (REPO_ROOT / "docs/lewm_go2_matched_branch_physical_outcome_screen_integrity_replacement_v1_terminal_review_2026-08-03.json", "d3f2d99c1a7f7d4e6d02215f04209732f326651e10bd06d040418cc7aafc5cbe", 22_378),
        "matched_successor_review": (REPO_ROOT / "docs/lewm_go2_matched_branch_successor_screen_v1_terminal_review_2026-08-03.json", "c450baab14b50caed3469fa88f5812c92c02b04676059568e8dae3dc2e5bad83", 4_991),
        "vjepa_horizon_review": (REPO_ROOT / "docs/lewm_go2_dense_vjepa2_1_horizon_diagnostic_v1_terminal_review_2026-08-03.json", "0751a9c2d6d2d7d7131ca32f3d3fdc5b4aa9740632fd9a84a51f5e87b82ee1cd", 4_913),
        "dinov2_physical_review": (REPO_ROOT / "docs/lewm_go2_dinov2_physical_readout_calibration_integrity_replacement_v1_terminal_review_2026-08-03.json", "7074779bdc506548d903c0319b74243f2b2934a1888325f813ee52f5a115c679", 14_382),
        "dinov2_dense_review": (REPO_ROOT / "docs/lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_terminal_review_2026-08-03.json", "f6ed2d09a407a4cf70097eaa4b2dcffd223e598e4eb59cf8e751997459384020", 27_120),
        "dual_adapter_review": (REPO_ROOT / "docs/lewm_go2_dual_residual_token_adapter_jepa_v1_terminal_review_2026-08-03.json", "365ab4057bfc51fe9d1b0bd3e7dd415bbddcde9adf89a3ac7674f34b2bc5f1fd", 9_116),
        "v20_result": (REPO_ROOT / "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_grounding_joint_jepa_v20_scientific_result_2026-07-30.json", "d76fd16732d15b7637bbe8f68df65ba23990046812f4ec3d85297f7f8ea64956", 17_166),
        "v21_result": (REPO_ROOT / "docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21_scientific_result_2026-07-30.json", "c9544055b11d162b5b5fc9b02d0a04f3961a61b4547411964812a9ae4c5da1e7", 15_724),
        "v22_result": (REPO_ROOT / "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22_scientific_result_2026-07-30.json", "1f4896e8f0ae8cadbf09e6f6f34417f3fa6362f9321cfd5abd0aeb09735453d0", 18_445),
        "recurrent_result": (REPO_ROOT / "docs/lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_scientific_result_2026-07-31.json", "180b348449ef16326cd797087a85251037f1fbd6f722b141f35f72aa3f57821c", 8_843),
        "recurrent_review": (REPO_ROOT / "docs/lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_scientific_result_independent_review_2026-07-31.json", "fe630b86a3ba2b07224e44f4734f0d187294ef616bcda9d8224e8c5fe41ff473", 3_099),
    }
    result = {
        label: _binding(path, sha256, byte_count)
        for label, (path, sha256, byte_count) in values.items()
    }
    if len(result) != 30:
        raise DenseVJEPACeilingRunnerError("fixed input/witness count changed")
    return result


SCIENTIFIC_INPUT_LABELS = frozenset(
    {
        "posthoc_manifest", "posthoc_terminal", "posthoc_terminal_review",
        "posthoc_rgb_manifest", "posthoc_train_rows", "posthoc_eval_rows",
        "stored_task_relevance_result", "stored_task_relevance_review",
        "physics_result", "physics_receipt_check", "collection_terminal",
        "collection_plan", "calibration_receipt", "train_vjepa_cache",
        "train_vjepa_receipt", "vjepa_checkpoint", "physical_evaluation",
        "physical_result", "physical_terminal", "physical_terminal_review",
    }
)
LINEAGE_WITNESS_LABELS = frozenset(
    {
        "matched_successor_review", "vjepa_horizon_review",
        "dinov2_physical_review", "dinov2_dense_review", "dual_adapter_review",
        "v20_result", "v21_result", "v22_result", "recurrent_result",
        "recurrent_review",
    }
)
if (
    len(SCIENTIFIC_INPUT_LABELS) != 20
    or len(LINEAGE_WITNESS_LABELS) != 10
    or SCIENTIFIC_INPUT_LABELS & LINEAGE_WITNESS_LABELS
    or SCIENTIFIC_INPUT_LABELS | LINEAGE_WITNESS_LABELS
    != frozenset(_fixed_input_bindings_v1())
):
    raise DenseVJEPACeilingRunnerError("input classification changed")


def permissions_v1() -> dict[str, Any]:
    result = {
        "primary_eval_rgb_access": True,
        "primary_vjepa2_1_encoder_execution": True,
        "primary_dense_readout_fitting": True,
        "one_eval_cache_creation": True,
        "required_eval_frames_decoded": ROLE_ARTIFACT_COUNT,
        "required_eval_frames_encoded": ROLE_ARTIFACT_COUNT,
        "required_eval_context_frames": EVAL_CONTEXT_COUNT,
        "required_eval_successor_frames": EVAL_SUCCESSOR_COUNT,
        "train_rgb_access": False,
        "other_encoder_execution": False,
        "second_or_repeated_extraction": False,
        "replay_rgb_access": False,
        "replay_extraction": False,
        "replay_encoder_execution": False,
        "collection": False,
        "protected_access": False,
        "heldout_access": False,
        "sealed_access": False,
        "retry": False,
        "resume": False,
        "extension": False,
        "replacement_attempt": False,
        "downstream_successor_execution": False,
    }
    if set(result) != PERMISSION_FIELDS:
        raise DenseVJEPACeilingRunnerError("permission inventory changed")
    return result


def config_v1() -> dict[str, Any]:
    return {
        "scientific": evaluator.config_v1(),
        "scientific_input_file_count": 20,
        "lineage_witness_file_count": 10,
        "total_fixed_input_file_count": 30,
        "state_receipt_file_count": STATE_RECEIPT_COUNT,
        "eval_rgb_file_count": ROLE_ARTIFACT_COUNT,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "feature_shape": [ROLE_ARTIFACT_COUNT, *TOKEN_SHAPE],
        "source_file_count": len(SOURCE_PATHS),
        "output_inventory": list(OUTPUT_NAMES),
        "replay_count": 1,
    }


def _bound_document_v1(authority: Mapping[str, Any], label: str) -> dict[str, Any]:
    binding = authority["input_bindings"][label]
    document, _ = _read_bound_json(
        Path(binding["path"]),
        expected_sha256=binding["sha256"],
        expected_byte_count=binding["byte_count"],
        label=label.replace("_", " "),
    )
    return document


def _state_receipt_bindings_from_physics_v1(
    authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    document = _bound_document_v1(authority, "physics_result")
    values = document.get("state_receipt_bindings")
    if (
        document.get("schema") != contract.PHYSICS_RESULT_SCHEMA
        or document.get("status") != "PHYSICS_COMPLETE"
        or document.get("failure") is not None
        or not isinstance(values, list)
        or len(values) != STATE_RECEIPT_COUNT
    ):
        raise DenseVJEPACeilingRunnerError("physics state-receipt closure changed")
    normalized = []
    for index, value in enumerate(values):
        raw_path = value.get("path") if isinstance(value, Mapping) else None
        relative = PurePosixPath(raw_path) if isinstance(raw_path, str) else None
        if (
            relative is None
            or relative.is_absolute()
            or not relative.parts
            or "." in relative.parts
            or ".." in relative.parts
            or relative.as_posix() != raw_path
        ):
            raise DenseVJEPACeilingRunnerError(
                f"state receipt {index} path is not canonical relative POSIX"
            )
        normalized.append(
            _normalize_inert_binding(
                value, base=PHYSICS_ROOT, label=f"state receipt {index}"
            )
        )
    if len({item["path"] for item in normalized}) != STATE_RECEIPT_COUNT:
        raise DenseVJEPACeilingRunnerError("state receipt path repeats")
    return normalized


def _read_jsonl_v1(authority: Mapping[str, Any], *, role: str) -> list[dict[str, Any]]:
    if role not in {"train", "eval"}:
        raise DenseVJEPACeilingRunnerError("JSONL role changed")
    label = f"posthoc_{role}_rows"
    binding = authority["input_bindings"][label]
    raw, observed = _read_bound_bytes_once_v1(
        Path(binding["path"]),
        expected_sha256=binding["sha256"],
        expected_byte_count=binding["byte_count"],
        label=f"{role} role rows",
    )
    if observed != binding:
        raise DenseVJEPACeilingRunnerError(f"{role} row binding changed")
    rows: list[dict[str, Any]] = []
    try:
        lines = raw.decode("utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise DenseVJEPACeilingRunnerError(
                    f"{role} row {line_number} is not an object"
                )
            rows.append(dict(value))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DenseVJEPACeilingRunnerError(f"{role} JSONL is invalid") from error
    if len(rows) != ROLE_STATE_COUNT:
        raise DenseVJEPACeilingRunnerError(f"{role} JSONL state count changed")
    return rows


def _consumer_compatible_sync_document_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only the reviewed redundant all-zero synchronization diagnostics."""

    result = copy.deepcopy(dict(value))
    sync = result.get("synchronization_audit")
    components = sync.get("components") if isinstance(sync, Mapping) else None
    if not isinstance(sync, dict) or not isinstance(components, Mapping):
        raise DenseVJEPACeilingRunnerError("synchronization components are absent")
    projected: dict[str, Any] = {}
    for name, component in components.items():
        if (
            not isinstance(component, Mapping)
            or set(component)
            != {
                "exact_equal", "max_abs_difference",
                "per_lane_max_abs_difference", "rms_difference",
                "shape_per_lane",
            }
            or component.get("exact_equal") is not True
            or float(component.get("max_abs_difference", -1.0)) != 0.0
            or float(component.get("rms_difference", -1.0)) != 0.0
            or not isinstance(component.get("per_lane_max_abs_difference"), list)
            or len(component["per_lane_max_abs_difference"]) != evaluator.ACTION_COUNT
            or any(float(item) != 0.0 for item in component["per_lane_max_abs_difference"])
        ):
            raise DenseVJEPACeilingRunnerError(
                "synchronization diagnostic is not redundant and exact"
            )
        projected[str(name)] = {
            "exact_equal": True,
            "max_abs_difference": 0.0,
            "shape_per_lane": list(component["shape_per_lane"]),
        }
    sync["components"] = projected
    return result


def _load_state_documents_v1(
    authority: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    expected = _state_receipt_bindings_from_physics_v1(authority)
    if authority.get("state_receipt_bindings") != expected:
        raise DenseVJEPACeilingRunnerError("authority state-receipt closure changed")
    documents: dict[str, dict[str, Any]] = {}
    for index, binding in enumerate(expected):
        receipt, _ = _read_bound_json(
            Path(binding["path"]),
            expected_sha256=binding["sha256"],
            expected_byte_count=binding["byte_count"],
            label=f"state receipt {index}",
        )
        compatible = _consumer_compatible_sync_document_v1(receipt)
        state = compatible.get("state")
        state_id = state.get("state_id") if isinstance(state, Mapping) else None
        if not isinstance(state_id, str) or not state_id or state_id in documents:
            raise DenseVJEPACeilingRunnerError("state receipt identity changed")
        documents[state_id] = compatible
    if len(documents) != STATE_RECEIPT_COUNT:
        raise DenseVJEPACeilingRunnerError("state receipt count changed")
    return documents, expected


def _artifact_metadata_v1(
    authority: Mapping[str, Any], source_root: Path
) -> dict[str, consumer.RGBArtifactV1]:
    document = _bound_document_v1(authority, "posthoc_rgb_manifest")
    values = document.get("artifacts")
    if (
        document.get("schema") != consumer.RGB_MANIFEST_SCHEMA
        or not isinstance(values, list)
        or len(values) != 2 * ROLE_ARTIFACT_COUNT
    ):
        raise DenseVJEPACeilingRunnerError("posthoc RGB manifest changed")
    artifacts: dict[str, consumer.RGBArtifactV1] = {}
    paths: set[str] = set()
    expected_fields = {
        "artifact_id", "frame_identity", "path", "file_sha256", "pixel_sha256",
        "byte_count", "width", "height", "mode", "format", "camera_valid",
        "low_information", "low_info_reasons",
    }
    for value in values:
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise DenseVJEPACeilingRunnerError("RGB artifact metadata fields changed")
        artifact_id = value.get("artifact_id")
        relative = PurePosixPath(str(value.get("path")))
        if (
            not isinstance(artifact_id, str)
            or not artifact_id
            or artifact_id in artifacts
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in paths
            or value.get("width") != 224
            or value.get("height") != 224
            or value.get("mode") != "RGB"
            or value.get("format") != "PNG"
            or value.get("camera_valid") is not True
        ):
            raise DenseVJEPACeilingRunnerError("RGB artifact metadata changed")
        _reject_protected(source_root.joinpath(*relative.parts), label="RGB artifact metadata")
        artifacts[artifact_id] = consumer.RGBArtifactV1(
            artifact_id=artifact_id,
            frame_identity=str(value["frame_identity"]),
            relative_path=relative.as_posix(),
            byte_count=int(value["byte_count"]),
            file_sha256=str(value["file_sha256"]),
            pixel_sha256=str(value["pixel_sha256"]),
            low_information=bool(value["low_information"]),
            low_info_reasons=tuple(str(item) for item in value["low_info_reasons"]),
        )
        paths.add(relative.as_posix())
    return artifacts


def _load_narrow_bundle_v1(
    authority: Mapping[str, Any],
) -> tuple[consumer.CounterfactualPilotBundleV1, dict[str, Any]]:
    """Load only the frozen direct metadata and the 256 admitted state receipts."""

    manifest = _bound_document_v1(authority, "posthoc_manifest")
    if (
        manifest.get("schema") != POSTHOC_MANIFEST_SCHEMA
        or manifest.get("status") != POSTHOC_MANIFEST_STATUS
        or manifest.get("derived_output_root") != str(POSTHOC_ROOT.resolve())
        or manifest.get("rgb_artifacts") != 2 * ROLE_ARTIFACT_COUNT
        or manifest.get("role_scene_counts") != {"train": 16, "eval": 16}
    ):
        raise DenseVJEPACeilingRunnerError("posthoc manifest contract changed")
    leaves = manifest.get("derived_leaf_bindings")
    if not isinstance(leaves, Mapping):
        raise DenseVJEPACeilingRunnerError("posthoc derived leaves are absent")
    for leaf, input_label in (
        ("rgb_manifest", "posthoc_rgb_manifest"),
        ("train", "posthoc_train_rows"),
        ("eval", "posthoc_eval_rows"),
    ):
        observed = _normalize_inert_binding(
            leaves.get(leaf), base=POSTHOC_ROOT, label=f"posthoc {leaf} leaf"
        )
        if observed != authority["input_bindings"][input_label]:
            raise DenseVJEPACeilingRunnerError(f"posthoc {leaf} leaf changed")
    source_root = _safe_path(
        Path(str(manifest.get("source_receipt_root"))), label="source receipt root"
    )
    if source_root != PHYSICS_ROOT.resolve() or not source_root.is_dir():
        raise DenseVJEPACeilingRunnerError("source receipt root changed")
    artifacts = _artifact_metadata_v1(authority, source_root)
    state_documents, state_bindings = _load_state_documents_v1(authority)
    try:
        _excluded, tolerances = consumer._validate_calibration_contract(  # noqa: SLF001
            manifest["calibration_contract"], textured_v03=True
        )
        requested_blocks = tuple(
            consumer._validate_tape(item["requested_block"], name="requested action")  # noqa: SLF001
            for item in manifest["action_catalog"]
        )
    except Exception as error:
        raise DenseVJEPACeilingRunnerError("posthoc calibration metadata changed") from error
    if len(requested_blocks) != evaluator.ACTION_COUNT:
        raise DenseVJEPACeilingRunnerError("posthoc action catalog changed")
    groups_by_role: dict[str, tuple[Any, ...]] = {}
    used_state_ids: set[str] = set()
    used_scenes: set[str] = set()
    used_artifacts: dict[str, set[str]] = {}
    role_bindings: dict[str, Mapping[str, Any]] = {}
    for role in ("train", "eval"):
        rows = _read_jsonl_v1(authority, role=role)
        groups = []
        for row in rows:
            state_id = row.get("state_id")
            if not isinstance(state_id, str) or state_id not in state_documents:
                raise DenseVJEPACeilingRunnerError(f"{role} row state receipt is absent")
            groups.append(
                consumer._parse_group(  # noqa: SLF001
                    _consumer_compatible_sync_document_v1(row),
                    role=role,
                    artifacts=artifacts,
                    tolerances=tolerances,
                    requested_blocks=requested_blocks,
                    collection_state=state_documents[state_id],
                )
            )
        if len(groups) != ROLE_STATE_COUNT:
            raise DenseVJEPACeilingRunnerError(f"{role} group count changed")
        state_ids = {group.state_id for group in groups}
        scenes = {group.scene_id for group in groups}
        artifact_ids = {
            artifact_id
            for group in groups
            for artifact_id in (
                *group.context_rgb_artifact_ids,
                *(branch.target_rgb_artifact_id for branch in group.branches),
            )
        }
        if (
            len(state_ids) != ROLE_STATE_COUNT
            or len(scenes) != ROLE_SCENE_COUNT
            or len(artifact_ids) != ROLE_ARTIFACT_COUNT
            or state_ids & used_state_ids
            or scenes & used_scenes
            or any(artifact_ids & prior for prior in used_artifacts.values())
        ):
            raise DenseVJEPACeilingRunnerError(f"{role} role closure changed")
        groups_by_role[role] = tuple(groups)
        used_state_ids.update(state_ids)
        used_scenes.update(scenes)
        used_artifacts[role] = artifact_ids
        role_bindings[role] = MappingProxyType(
            dict(authority["input_bindings"][f"posthoc_{role}_rows"])
        )
    if used_state_ids != set(state_documents) or set().union(*used_artifacts.values()) != set(artifacts):
        raise DenseVJEPACeilingRunnerError("rows do not exhaust the frozen closures")
    bundle = consumer.CounterfactualPilotBundleV1(
        root=source_root,
        manifest_binding=MappingProxyType(dict(authority["input_bindings"]["posthoc_manifest"])),
        manifest=MappingProxyType(dict(manifest)),
        rgb_manifest_binding=MappingProxyType(dict(authority["input_bindings"]["posthoc_rgb_manifest"])),
        artifacts=MappingProxyType(artifacts),
        groups_by_role=MappingProxyType(groups_by_role),
        role_bindings=MappingProxyType(role_bindings),
        calibration_receipt=MappingProxyType({}),
        calibration_tolerances=MappingProxyType(dict(tolerances)),
        access_audit=MappingProxyType(
            {
                "direct_scientific_document_parse_count": 5,
                "state_receipt_open_count": STATE_RECEIPT_COUNT,
                "render_receipt_open_count": 0,
                "calibration_collection_receipt_open_count": 0,
                "rgb_leaf_open_count": 0,
                "narrow_preregistered_closure": True,
            }
        ),
    )
    plans = feature_plans.build_calibration_feature_plans_v1(
        groups_by_role["train"], groups_by_role["eval"]
    )
    if (
        plans.train.identity_sha256 != evaluator.EXPECTED_TRAIN_PLAN_IDENTITY
        or plans.eval.identity_sha256 != evaluator.EXPECTED_EVAL_PLAN_IDENTITY
        or plans.identity_sha256 != evaluator.EXPECTED_COMBINED_PLAN_IDENTITY
    ):
        raise DenseVJEPACeilingRunnerError("frozen feature-plan identity changed")
    return bundle, {
        "state_receipt_bindings": state_bindings,
        "feature_plan": {
            "train": plans.train.identity_sha256,
            "eval": plans.eval.identity_sha256,
            "combined": plans.identity_sha256,
        },
    }


def _input_classification_v1() -> dict[str, list[str]]:
    return {
        "scientific_inputs": sorted(SCIENTIFIC_INPUT_LABELS),
        "lineage_witnesses": sorted(LINEAGE_WITNESS_LABELS),
    }


def _validate_eval_rgb_authority_bindings_v1(
    value: object,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != ROLE_ARTIFACT_COUNT:
        raise DenseVJEPACeilingRunnerError("authority evaluation RGB closure changed")
    result: list[dict[str, Any]] = []
    artifact_ids: set[str] = set()
    paths: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {
            "artifact_id", "path", "sha256", "pixel_sha256", "byte_count"
        }:
            raise DenseVJEPACeilingRunnerError(
                f"evaluation RGB binding {index} fields changed"
            )
        artifact_id = item.get("artifact_id")
        path = item.get("path")
        sha256 = item.get("sha256")
        pixel_sha256 = item.get("pixel_sha256")
        byte_count = item.get("byte_count")
        if (
            not isinstance(artifact_id, str)
            or not artifact_id
            or artifact_id in artifact_ids
            or not isinstance(path, str)
            or not Path(path).is_absolute()
            or path in paths
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(pixel_sha256, str)
            or len(pixel_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in pixel_sha256
            )
            or type(byte_count) is not int
            or byte_count <= 0
        ):
            raise DenseVJEPACeilingRunnerError(
                f"evaluation RGB binding {index} is malformed"
            )
        _reject_protected(Path(path), label=f"evaluation RGB binding {index}")
        result.append(dict(item))
        artifact_ids.add(artifact_id)
        paths.add(path)
    return result


def _validate_encoder_source_v1(
    value: object,
    *,
    source_bindings: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "repo_path", "repo_commit", "repo_tree", "hub_entrypoint",
        "checkpoint_binding", "transitive_source_bindings",
        "drop_path_shim_source_binding", "einops_dependency",
    }:
        raise DenseVJEPACeilingRunnerError("V-JEPA encoder-source fields changed")
    if (
        value.get("repo_path") != str(VJEPA_REPOSITORY.resolve())
        or value.get("repo_commit") != VJEPA_REPOSITORY_COMMIT
        or value.get("repo_tree") != VJEPA_REPOSITORY_TREE
        or value.get("hub_entrypoint") != "vjepa2_1_vit_base_384"
        or value.get("checkpoint_binding") != input_bindings["vjepa_checkpoint"]
        or value.get("drop_path_shim_source_binding")
        != source_bindings["ceiling_runner"]
        or value.get("einops_dependency") != einops_dependency_v1()
    ):
        raise DenseVJEPACeilingRunnerError("V-JEPA encoder source changed")
    transitive = value.get("transitive_source_bindings")
    if (
        not isinstance(transitive, Mapping)
        or dict(transitive) != VJEPA_TRANSITIVE_SOURCE_BINDINGS
    ):
        raise DenseVJEPACeilingRunnerError("V-JEPA transitive source closure is absent")
    repo = _safe_path(VJEPA_REPOSITORY, label="V-JEPA repository")
    if not repo.is_dir():
        raise DenseVJEPACeilingRunnerError("V-JEPA repository is not a directory")
    for label, binding in transitive.items():
        if not isinstance(label, str) or not label:
            raise DenseVJEPACeilingRunnerError("V-JEPA source label changed")
        observed = _require_binding(binding, label=f"V-JEPA source {label}")
        try:
            Path(observed["path"]).relative_to(repo)
        except ValueError as error:
            raise DenseVJEPACeilingRunnerError(
                "V-JEPA transitive source escapes the bound repository"
            ) from error
    dependency = value["einops_dependency"]
    for section in (
        "runtime_source_bindings", "distribution_metadata_bindings"
    ):
        for label, binding in dependency[section].items():
            _require_binding(binding, label=f"einops dependency {label}")
    specification = importlib.util.find_spec("einops")
    try:
        distribution = importlib.metadata.distribution("einops")
    except importlib.metadata.PackageNotFoundError as error:
        raise DenseVJEPACeilingRunnerError("einops distribution is absent") from error
    distribution_root = Path(distribution.locate_file("einops-0.8.1.dist-info")).resolve()
    if (
        specification is None
        or specification.origin != str(EINOPS_MODULE_ORIGIN.resolve())
        or distribution.version != "0.8.1"
        or distribution_root != EINOPS_DISTRIBUTION_ROOT.resolve()
    ):
        raise DenseVJEPACeilingRunnerError("einops runtime identity changed")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != VJEPA_REPOSITORY_COMMIT or tree != VJEPA_REPOSITORY_TREE or status:
        raise DenseVJEPACeilingRunnerError("V-JEPA repository identity changed")
    return dict(value)


def _validate_source_review_v1(
    review: Mapping[str, Any],
    *,
    preregistration_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    encoder_source: Mapping[str, Any],
) -> None:
    reviewer = review.get("reviewer")
    checks = review.get("checks")
    if (
        set(review) != SOURCE_REVIEW_FIELDS
        or review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != SOURCE_REVIEW_STATUS
        or review.get("review_date") != "2026-08-03"
        or review.get("protected_material_opened") is not False
        or review.get("preregistration_binding") != preregistration_binding
        or review.get("source_bindings") != source_bindings
        or review.get("encoder_source") != encoder_source
        or review.get("findings") != []
        or not isinstance(reviewer, Mapping)
        or set(reviewer) != {"identity", "independence_basis"}
        or any(
            not isinstance(item, str) or not item.strip()
            for item in reviewer.values()
        )
        or not isinstance(checks, Mapping)
        or set(checks) != SOURCE_REVIEW_CHECKS
        or any(item is not True for item in checks.values())
    ):
        raise DenseVJEPACeilingRunnerError(
            "independent source review did not pass exactly"
        )


def _read_authority(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    replay_mode: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority, authority_binding = _read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="execution authority",
    )
    required = {
        "schema", "status", "citable_as_scientific_evidence",
        "development_only", "preregistration_binding", "source_review_binding",
        "source_bindings", "input_bindings", "input_classification",
        "state_receipt_bindings", "eval_rgb_bindings", "encoder_source",
        "environment", "hardware", "config", "permissions", "output_root",
        "git_commit",
    }
    if (
        set(authority) != required
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("citable_as_scientific_evidence") is not False
        or authority.get("development_only") is not True
        or authority.get("input_classification") != _input_classification_v1()
        or authority.get("config") != config_v1()
        or authority.get("permissions") != permissions_v1()
        or authority.get("output_root") != str(DEFAULT_OUTPUT_ROOT.resolve())
    ):
        raise DenseVJEPACeilingRunnerError("execution authority contract changed")
    preregistration = _require_binding(
        authority["preregistration_binding"], label="preregistration"
    )
    if preregistration != _binding(
        PREREGISTRATION, PREREGISTRATION_SHA256, PREREGISTRATION_BYTE_COUNT
    ):
        raise DenseVJEPACeilingRunnerError(
            "authority does not bind the frozen preregistration"
        )
    inputs = authority.get("input_bindings")
    if not isinstance(inputs, Mapping) or dict(inputs) != _fixed_input_bindings_v1():
        raise DenseVJEPACeilingRunnerError("authority fixed-input closure changed")
    for label, binding in inputs.items():
        if replay_mode and label == "vjepa_checkpoint":
            if _normalize_inert_binding(binding, label="V-JEPA checkpoint") != binding:
                raise DenseVJEPACeilingRunnerError("V-JEPA checkpoint binding changed")
        else:
            _require_binding(binding, label=f"fixed input {label}")
    sources = authority.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_PATHS):
        raise DenseVJEPACeilingRunnerError("authority source closure changed")
    for label, expected_path in SOURCE_PATHS.items():
        observed = _require_binding(sources[label], label=f"source {label}")
        if observed["path"] != str(expected_path.resolve()):
            raise DenseVJEPACeilingRunnerError(f"source {label} path changed")
    encoder_source = _validate_encoder_source_v1(
        authority.get("encoder_source"),
        source_bindings=sources,
        input_bindings=inputs,
    )
    source_review_binding = _require_binding(
        authority["source_review_binding"], label="source review"
    )
    if source_review_binding["path"] != str(SOURCE_REVIEW.resolve()):
        raise DenseVJEPACeilingRunnerError("source review path changed")
    review, _ = _read_bound_json(
        Path(source_review_binding["path"]),
        expected_sha256=source_review_binding["sha256"],
        expected_byte_count=source_review_binding["byte_count"],
        label="source review",
    )
    _validate_source_review_v1(
        review,
        preregistration_binding=preregistration,
        source_bindings=sources,
        encoder_source=encoder_source,
    )
    expected_state_receipts = _state_receipt_bindings_from_physics_v1(authority)
    if authority.get("state_receipt_bindings") != expected_state_receipts:
        raise DenseVJEPACeilingRunnerError("authority state-receipt closure changed")
    _validate_eval_rgb_authority_bindings_v1(authority.get("eval_rgb_bindings"))
    environment = authority.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment)
        != {"python", "python_version", "torch", "hip_rocm", "numpy", "pillow", "einops", "timm"}
        or environment.get("python") != str(Path(sys.executable).resolve())
        or environment.get("python_version") != platform.python_version()
        or environment.get("torch") != torch.__version__
        or environment.get("hip_rocm") != torch.version.hip
        or environment.get("numpy") != np.__version__
        or environment.get("pillow") != PILLOW_VERSION
        or environment.get("einops") != "0.8.1"
        or importlib.metadata.version("einops") != "0.8.1"
        or environment.get("timm") != "NOT_INSTALLED_REVIEWED_DROP_PATH_SHIM"
        or importlib.util.find_spec("timm") is not None
    ):
        raise DenseVJEPACeilingRunnerError("execution environment changed")
    hardware = authority.get("hardware")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise DenseVJEPACeilingRunnerError("authorized ROCm device is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if (
        not isinstance(hardware, Mapping)
        or set(hardware)
        != {"device_type", "device_index", "device_name", "total_memory_bytes"}
        or hardware.get("device_type") != "cuda"
        or hardware.get("device_index") != 0
        or hardware.get("device_name") != torch.cuda.get_device_name(0)
        or hardware.get("total_memory_bytes") != int(properties.total_memory)
    ):
        raise DenseVJEPACeilingRunnerError("execution hardware changed")
    commit = authority.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        != 0
    ):
        raise DenseVJEPACeilingRunnerError(
            "reviewed source commit is not an execution ancestor"
        )
    return authority, authority_binding


def _feature_plans_v1(
    bundle: consumer.CounterfactualPilotBundleV1,
) -> tuple[tuple[Any, ...], tuple[Any, ...], Any, Any]:
    audit = bundle.access_audit
    if (
        audit.get("state_receipt_open_count") != STATE_RECEIPT_COUNT
        or audit.get("render_receipt_open_count") != 0
        or audit.get("calibration_collection_receipt_open_count") != 0
        or audit.get("rgb_leaf_open_count") != 0
        or audit.get("narrow_preregistered_closure") is not True
    ):
        raise DenseVJEPACeilingRunnerError("narrow-loader access audit changed")
    groups = bundle.groups_by_role
    if set(groups) != {"train", "eval"}:
        raise DenseVJEPACeilingRunnerError("bounded bundle roles changed")
    train_groups = tuple(groups["train"])
    eval_groups = tuple(groups["eval"])
    plans = feature_plans.build_calibration_feature_plans_v1(
        train_groups, eval_groups
    )
    if (
        len(train_groups) != ROLE_STATE_COUNT
        or len(eval_groups) != ROLE_STATE_COUNT
        or plans.train.identity_sha256 != evaluator.EXPECTED_TRAIN_PLAN_IDENTITY
        or plans.eval.identity_sha256 != evaluator.EXPECTED_EVAL_PLAN_IDENTITY
        or plans.identity_sha256 != evaluator.EXPECTED_COMBINED_PLAN_IDENTITY
        or set(plans.train.artifact_ids) & set(plans.eval.artifact_ids)
    ):
        raise DenseVJEPACeilingRunnerError("frozen feature plans changed")
    return train_groups, eval_groups, plans.train, plans.eval


def _validate_feature_tensor_v1(features: object, *, role: str) -> torch.Tensor:
    if (
        not isinstance(features, torch.Tensor)
        or tuple(features.shape) != (ROLE_ARTIFACT_COUNT, *TOKEN_SHAPE)
        or features.dtype != torch.float16
        or features.device.type != "cpu"
        or not bool(torch.isfinite(features).all())
    ):
        raise DenseVJEPACeilingRunnerError(f"{role} V-JEPA cache tensor changed")
    maximum_error = 0.0
    for start in range(0, ROLE_ARTIFACT_COUNT, 32):
        norms = torch.linalg.vector_norm(
            features[start : start + 32].float(), dim=-1
        )
        maximum_error = max(
            maximum_error, float(torch.max(torch.abs(norms - 1.0)).item())
        )
    if maximum_error > MAX_TOKEN_NORM_ERROR:
        raise DenseVJEPACeilingRunnerError(
            f"{role} V-JEPA token normalization changed"
        )
    return features


def feature_preprocessing_contract_v1() -> dict[str, Any]:
    return {
        "decoded_input": {"format": "PNG", "mode": "RGB", "size": [224, 224]},
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "token_conversion": {
            "output_grid": [16, 16],
            "compute_dtype": "float32",
            "per_token_l2_normalization": True,
        },
        "image_geometry": {
            "resize": [438, 438],
            "resize_kernel": "PIL_BILINEAR",
            "center_crop": [384, 384],
            "image_mode_frames": 1,
        },
        "encoder_output_grid": [24, 24],
        "spatial_conversion": "torch_area_24x24_to_16x16",
    }


def _decode_exact_rgb_png_v1(raw: bytes) -> Image.Image:
    if not isinstance(raw, bytes) or not raw:
        raise DenseVJEPACeilingRunnerError("RGB payload must be nonempty bytes")
    try:
        with Image.open(BytesIO(raw)) as probe:
            if (
                probe.format != "PNG"
                or probe.mode != "RGB"
                or probe.size != (224, 224)
                or getattr(probe, "n_frames", 1) != 1
            ):
                raise DenseVJEPACeilingRunnerError(
                    "RGB leaf must be one exact 224x224 RGB PNG"
                )
            probe.verify()
        with Image.open(BytesIO(raw)) as decoded:
            decoded.load()
            return decoded.copy()
    except DenseVJEPACeilingRunnerError:
        raise
    except Exception as error:
        raise DenseVJEPACeilingRunnerError("RGB PNG cannot be decoded") from error


def preprocess_vjepa2_1_png_bytes_v1(raw: bytes) -> torch.Tensor:
    image = _decode_exact_rgb_png_v1(raw)
    image = image.resize((438, 438), resample=Image.Resampling.BILINEAR)
    image = image.crop((27, 27, 411, 411))
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (384, 384, 3):
        raise DenseVJEPACeilingRunnerError("decoded RGB raster shape changed")
    tensor = (
        torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
    )
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    result = tensor.sub_(mean).div_(std).unsqueeze(1)
    if tuple(result.shape) != (3, 1, 384, 384):
        raise DenseVJEPACeilingRunnerError("V-JEPA preprocessing shape changed")
    return result


def normalize_vjepa_token_grid_v1(tokens: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 3
        or tokens.shape[0] < 1
        or tuple(tokens.shape[1:]) != (576, 768)
        or tokens.dtype
        not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        or not bool(torch.isfinite(tokens).all())
    ):
        raise DenseVJEPACeilingRunnerError("V-JEPA raw token grid changed")
    batch = tokens.shape[0]
    grid = tokens.float().transpose(1, 2).reshape(batch, 768, 24, 24)
    converted = F.interpolate(grid, size=(16, 16), mode="area")
    converted = converted.flatten(2).transpose(1, 2)
    norms = torch.linalg.vector_norm(converted, dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0.0).any()):
        raise DenseVJEPACeilingRunnerError("V-JEPA token grid is zero or nonfinite")
    result = F.normalize(converted, p=2.0, dim=-1)
    if tuple(result.shape) != (batch, *TOKEN_SHAPE) or not bool(
        torch.isfinite(result).all()
    ):
        raise DenseVJEPACeilingRunnerError("normalized V-JEPA grid changed")
    return result.contiguous()


def drop_path_compat_v1(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    probability = float(drop_prob)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("drop_prob must be in [0,1]")
    if probability == 0.0 or not training:
        return x
    keep_prob = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPathCompatV1(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path_compat_v1(
            x, self.drop_prob, self.training, self.scale_by_keep
        )


def _package_module_v1(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = []  # type: ignore[attr-defined]
    module.__spec__ = importlib.machinery.ModuleSpec(
        name=name, loader=None, is_package=True
    )
    return module


@contextmanager
def scoped_timm_drop_path_shim_v1() -> Iterator[None]:
    names = ("timm", "timm.models", "timm.models.layers")
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    timm_module = _package_module_v1("timm")
    models_module = _package_module_v1("timm.models")
    layers_module = _package_module_v1("timm.models.layers")
    layers_module.drop_path = drop_path_compat_v1  # type: ignore[attr-defined]
    layers_module.DropPath = DropPathCompatV1  # type: ignore[attr-defined]
    timm_module.models = models_module  # type: ignore[attr-defined]
    models_module.layers = layers_module  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "timm": timm_module,
            "timm.models": models_module,
            "timm.models.layers": layers_module,
        }
    )
    try:
        yield
    finally:
        for name in reversed(names):
            prior = previous[name]
            if prior is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior  # type: ignore[assignment]


def _load_train_cache_v1(
    authority: Mapping[str, Any],
    bundle: consumer.CounterfactualPilotBundleV1,
    train_plan: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    receipt = _bound_document_v1(authority, "train_vjepa_receipt")
    receipt_encoder = receipt.get("encoder_source")
    receipt_manifest = _normalize_inert_binding(
        receipt.get("source_bundle_manifest"), label="train cache source manifest"
    )
    if (
        not isinstance(receipt_encoder, Mapping)
        or set(receipt_encoder) != {"repo_path", "repo_commit", "checkpoint_binding"}
        or receipt_encoder.get("repo_path") != authority["encoder_source"]["repo_path"]
        or receipt_encoder.get("repo_commit")
        != authority["encoder_source"]["repo_commit"]
        or _normalize_inert_binding(
            receipt_encoder.get("checkpoint_binding"),
            label="train cache V-JEPA checkpoint",
        )
        != authority["input_bindings"]["vjepa_checkpoint"]
        or receipt_manifest != authority["input_bindings"]["posthoc_manifest"]
    ):
        raise DenseVJEPACeilingRunnerError("train V-JEPA provenance changed")
    if receipt.get("binding") != authority["input_bindings"]["train_vjepa_cache"]:
        raise DenseVJEPACeilingRunnerError("train V-JEPA receipt binding changed")
    artifact_ids = tuple(train_plan.artifact_ids)
    order_sha256 = hashlib.sha256(canonical_bytes_v1(list(artifact_ids))).hexdigest()
    if (
        receipt.get("schema")
        != "lewm_go2_matched_branch_successor_feature_cache_receipt_v1"
        or receipt.get("encoder") != "vjepa2_1"
        or receipt.get("preprocessing") != feature_preprocessing_contract_v1()
        or receipt.get("artifact_order_sha256") != order_sha256
        or receipt.get("artifact_count") != ROLE_ARTIFACT_COUNT
        or receipt.get("train_artifact_open_count") != ROLE_ARTIFACT_COUNT
        or receipt.get("eval_artifact_open_count") != 0
        or receipt.get("shape") != [ROLE_ARTIFACT_COUNT, *TOKEN_SHAPE]
        or receipt.get("storage_dtype") != "float16"
    ):
        raise DenseVJEPACeilingRunnerError("train V-JEPA receipt changed")
    payload, _binding_observed = _load_bound_torch_once_v1(
        receipt.get("binding"), label="train V-JEPA cache"
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema")
        != "lewm_go2_matched_branch_successor_feature_cache_v1"
        or payload.get("encoder") != "vjepa2_1"
        or payload.get("index_sha256") != receipt.get("index_sha256")
        or tuple(payload.get("artifact_ids", ())) != artifact_ids
    ):
        raise DenseVJEPACeilingRunnerError("train V-JEPA cache payload changed")
    return _validate_feature_tensor_v1(payload.get("features"), role="train"), receipt


def _eval_rgb_bindings_from_bundle_v1(
    authority: Mapping[str, Any],
    bundle: consumer.CounterfactualPilotBundleV1,
    eval_plan: Any,
) -> list[dict[str, Any]]:
    declared = _validate_eval_rgb_authority_bindings_v1(
        authority.get("eval_rgb_bindings")
    )
    expected: list[dict[str, Any]] = []
    for artifact_id in eval_plan.artifact_ids:
        artifact = bundle.artifacts.get(artifact_id)
        if artifact is None or artifact.pixel_sha256 is None:
            raise DenseVJEPACeilingRunnerError("evaluation RGB metadata is absent")
        relative = PurePosixPath(artifact.relative_path)
        if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
            raise DenseVJEPACeilingRunnerError("evaluation RGB path changed")
        expected.append(
            {
                "artifact_id": artifact_id,
                "path": str(bundle.root.joinpath(*relative.parts)),
                "sha256": artifact.file_sha256,
                "pixel_sha256": artifact.pixel_sha256,
                "byte_count": artifact.byte_count,
            }
        )
    if declared != expected:
        raise DenseVJEPACeilingRunnerError(
            "authority evaluation RGB closure or artifact order changed"
        )
    return declared


def _load_vjepa_encoder_v1(
    authority: Mapping[str, Any], device: torch.device
) -> torch.nn.Module:
    try:
        source = authority["encoder_source"]
        with scoped_timm_drop_path_shim_v1():
            encoder, predictor = torch.hub.load(
                str(source["repo_path"]),
                "vjepa2_1_vit_base_384",
                source="local",
                pretrained=False,
            )
        del predictor
        payload, _checkpoint_binding = _load_bound_torch_once_v1(
            source["checkpoint_binding"], label="V-JEPA checkpoint"
        )
        state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in payload["ema_encoder"].items()
        }
        encoder.load_state_dict(state, strict=True)
        del payload, state
        return encoder.to(device).eval().requires_grad_(False)
    except Exception as error:
        raise DenseVJEPACeilingRunnerError("frozen V-JEPA encoder load failed") from error


@torch.no_grad()
def extract_eval_feature_cache_v1(
    authority: Mapping[str, Any],
    bundle: consumer.CounterfactualPilotBundleV1,
    eval_plan: Any,
    *,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    declared = _eval_rgb_bindings_from_bundle_v1(
        authority, bundle, eval_plan
    )
    artifact_ids = tuple(eval_plan.artifact_ids)
    if len(artifact_ids) != ROLE_ARTIFACT_COUNT or EVAL_BATCH_SIZE < 1:
        raise DenseVJEPACeilingRunnerError("evaluation extraction geometry changed")
    context_indices = {
        index
        for state in eval_plan.states
        for index in state.context_artifact_indices
    }
    successor_indices = {
        index
        for state in eval_plan.states
        for index in state.target_artifact_indices
    }
    if (
        len(context_indices) != EVAL_CONTEXT_COUNT
        or len(successor_indices) != EVAL_SUCCESSOR_COUNT
        or context_indices & successor_indices
        or context_indices | successor_indices != set(range(ROLE_ARTIFACT_COUNT))
    ):
        raise DenseVJEPACeilingRunnerError("evaluation artifact slot classes changed")
    encoder = _load_vjepa_encoder_v1(authority, device).eval().requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batches: list[torch.Tensor] = []
    opened: list[str] = []
    started = time.perf_counter()
    for start in range(0, ROLE_ARTIFACT_COUNT, EVAL_BATCH_SIZE):
        selected = artifact_ids[start : start + EVAL_BATCH_SIZE]
        prepared = []
        for artifact_id in selected:
            raw = read_bound_rgb_bytes_v1(bundle, artifact_id)
            opened.append(artifact_id)
            prepared.append(preprocess_vjepa2_1_png_bytes_v1(raw))
        inputs = torch.stack(prepared).to(device)
        raw_tokens = encoder(inputs)
        normalized = normalize_vjepa_token_grid_v1(raw_tokens)
        batches.append(normalized.to(dtype=torch.float16, device="cpu"))
    elapsed = time.perf_counter() - started
    if tuple(opened) != artifact_ids or len(set(opened)) != ROLE_ARTIFACT_COUNT:
        raise DenseVJEPACeilingRunnerError(
            "evaluation RGB was not opened exactly once in artifact order"
        )
    features = _validate_feature_tensor_v1(torch.cat(batches), role="eval")
    order_sha256 = hashlib.sha256(canonical_bytes_v1(list(artifact_ids))).hexdigest()
    payload = {
        "schema": EVAL_CACHE_SCHEMA,
        "encoder": "vjepa2_1",
        "role": "eval",
        "eval_plan_identity": eval_plan.identity_sha256,
        "artifact_ids": artifact_ids,
        "artifact_order_sha256": order_sha256,
        "features": features,
    }
    _save_torch_exclusive(output_path, payload)
    receipt = {
        "schema": EVAL_CACHE_RECEIPT_SCHEMA,
        "encoder": "vjepa2_1",
        "role": "eval",
        "binding": file_binding_v1(output_path),
        "source_bundle_manifest": dict(bundle.manifest_binding),
        "encoder_source": dict(authority["encoder_source"]),
        "preprocessing": feature_preprocessing_contract_v1(),
        "eval_plan_identity": eval_plan.identity_sha256,
        "artifact_order_sha256": order_sha256,
        "artifact_count": ROLE_ARTIFACT_COUNT,
        "eval_artifact_open_count": ROLE_ARTIFACT_COUNT,
        "eval_context_open_count": len(context_indices),
        "eval_successor_open_count": len(successor_indices),
        "train_artifact_open_count": 0,
        "decoded_pixel_verification_count": ROLE_ARTIFACT_COUNT,
        "encoded_frame_count": ROLE_ARTIFACT_COUNT,
        "shape": list(features.shape),
        "storage_dtype": "float16",
        "elapsed_seconds": elapsed,
        "frames_per_second": ROLE_ARTIFACT_COUNT / elapsed,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "authority_eval_rgb_binding_order_sha256": hashlib.sha256(
            canonical_bytes_v1(declared)
        ).hexdigest(),
    }
    _write_json_exclusive(output_path.with_suffix(".json"), receipt)
    del encoder, batches, features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt


def _load_eval_cache_v1(
    receipt: Mapping[str, Any],
    eval_plan: Any,
    *,
    authority: Mapping[str, Any],
    bundle: consumer.CounterfactualPilotBundleV1,
) -> torch.Tensor:
    artifact_ids = tuple(eval_plan.artifact_ids)
    order_sha256 = hashlib.sha256(canonical_bytes_v1(list(artifact_ids))).hexdigest()
    if (
        receipt.get("schema") != EVAL_CACHE_RECEIPT_SCHEMA
        or receipt.get("encoder") != "vjepa2_1"
        or receipt.get("role") != "eval"
        or receipt.get("eval_plan_identity") != evaluator.EXPECTED_EVAL_PLAN_IDENTITY
        or receipt.get("artifact_order_sha256") != order_sha256
        or receipt.get("artifact_count") != ROLE_ARTIFACT_COUNT
        or receipt.get("eval_artifact_open_count") != ROLE_ARTIFACT_COUNT
        or receipt.get("eval_context_open_count") != EVAL_CONTEXT_COUNT
        or receipt.get("eval_successor_open_count") != EVAL_SUCCESSOR_COUNT
        or receipt.get("train_artifact_open_count") != 0
        or receipt.get("decoded_pixel_verification_count") != ROLE_ARTIFACT_COUNT
        or receipt.get("encoded_frame_count") != ROLE_ARTIFACT_COUNT
        or receipt.get("shape") != [ROLE_ARTIFACT_COUNT, *TOKEN_SHAPE]
        or receipt.get("storage_dtype") != "float16"
        or receipt.get("preprocessing")
        != feature_preprocessing_contract_v1()
        or receipt.get("source_bundle_manifest") != dict(bundle.manifest_binding)
        or receipt.get("encoder_source") != authority["encoder_source"]
        or receipt.get("authority_eval_rgb_binding_order_sha256")
        != hashlib.sha256(
            canonical_bytes_v1(authority["eval_rgb_bindings"])
        ).hexdigest()
    ):
        raise DenseVJEPACeilingRunnerError("evaluation cache receipt changed")
    inert_binding = _normalize_inert_binding(
        receipt.get("binding"), label="evaluation cache"
    )
    if inert_binding["path"] != str(
        (DEFAULT_OUTPUT_ROOT / "vjepa2_1_eval.pt").resolve()
    ):
        raise DenseVJEPACeilingRunnerError("evaluation cache path changed")
    payload, _binding_observed = _load_bound_torch_once_v1(
        inert_binding, label="evaluation cache"
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != EVAL_CACHE_SCHEMA
        or payload.get("encoder") != "vjepa2_1"
        or payload.get("role") != "eval"
        or payload.get("eval_plan_identity") != eval_plan.identity_sha256
        or tuple(payload.get("artifact_ids", ())) != artifact_ids
        or payload.get("artifact_order_sha256") != order_sha256
    ):
        raise DenseVJEPACeilingRunnerError("evaluation cache payload changed")
    return _validate_feature_tensor_v1(payload.get("features"), role="eval")


def _authorized_device_v1() -> torch.device:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise DenseVJEPACeilingRunnerError("authorized ROCm device is unavailable")
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    return torch.device("cuda", 0)


def _historical_dino_comparators_v1(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "frozen_dinov2_compact_physical_interface": {
            "terminal_review_binding": dict(
                authority["input_bindings"]["dinov2_physical_review"]
            ),
            "terminal_review": _bound_document_v1(
                authority, "dinov2_physical_review"
            ),
            "report_only": True,
            "gates_this_experiment": False,
        },
        "frozen_dinov2_dense_shared_spatial_interface": {
            "terminal_review_binding": dict(
                authority["input_bindings"]["dinov2_dense_review"]
            ),
            "terminal_review": _bound_document_v1(
                authority, "dinov2_dense_review"
            ),
            "report_only": True,
            "gates_this_experiment": False,
        },
    }


def _evaluate_v1(
    authority: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    train_groups: Sequence[Any],
    eval_groups: Sequence[Any],
    train_features: torch.Tensor,
    eval_features: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    retained = _bound_document_v1(authority, "physical_evaluation")
    result = evaluator.evaluate_primary_checkpoint_v1(
        checkpoint,
        train_groups,
        eval_groups,
        train_features,
        eval_features,
        retained,
        _historical_dino_comparators_v1(authority),
        device,
        implementation_source_binding=authority["source_bindings"][
            "ceiling_evaluator"
        ],
    )
    if not isinstance(result, Mapping):
        raise DenseVJEPACeilingRunnerError("evaluator returned a non-object")
    canonical_bytes_v1(result)
    return dict(result)


def _read_new_json_v1(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = file_binding_v1(path)
    return _read_bound_json(
        path,
        expected_sha256=binding["sha256"],
        expected_byte_count=binding["byte_count"],
        label=label,
    )


def _exact_tree_equal_v1(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_tree_equal_v1(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and type(left) is type(right)
            and len(left) == len(right)
            and all(
                _exact_tree_equal_v1(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


def _selected_actions_v1(evaluation: Mapping[str, Any]) -> dict[str, list[int]]:
    arms = evaluation.get("arms")
    if not isinstance(arms, Mapping):
        raise DenseVJEPACeilingRunnerError("evaluation arm inventory changed")
    result: dict[str, list[int]] = {}
    for name, report in arms.items():
        rows = report.get("group_results") if isinstance(report, Mapping) else None
        if isinstance(rows, list):
            result[str(name)] = [int(row["selected_action_id"]) for row in rows]
    return result


REPRODUCTION_FIELDS = frozenset(
    {
        "pca_identity_and_arrays",
        "train_action_mean_identity_and_values",
        "task_readout_identity_and_state",
        "model_state_identities_and_values",
        "optimizer_step_counts",
        "complete_checkpoint_tree",
        "per_seed_scores",
        "ensemble_scores",
        "selected_actions",
        "summaries",
        "bootstrap_intervals",
        "gates",
        "verdict",
        "complete_evaluation",
        "exactly_reproduced",
    }
)


def _reproduction_v1(
    primary_checkpoint: Mapping[str, Any],
    replay_checkpoint: Mapping[str, Any],
    primary_evaluation: Mapping[str, Any],
    replay_evaluation: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    primary_verdict = evaluator.verdict_v1(
        primary_evaluation,
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )
    replay_verdict = evaluator.verdict_v1(
        replay_evaluation,
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )
    pca_equal = _exact_tree_equal_v1(
        primary_checkpoint.get("pca"), replay_checkpoint.get("pca")
    )
    action_mean_equal = _exact_tree_equal_v1(
        primary_checkpoint.get("train_action_mean_innovation"),
        replay_checkpoint.get("train_action_mean_innovation"),
    )
    task_equal = _exact_tree_equal_v1(
        primary_checkpoint.get("task_action_only"),
        replay_checkpoint.get("task_action_only"),
    )
    primary_members = primary_checkpoint.get("members")
    replay_members = replay_checkpoint.get("members")
    model_states_equal = _exact_tree_equal_v1(primary_members, replay_members)
    step_counts_equal = False
    if isinstance(primary_members, list) and isinstance(replay_members, list):
        step_counts_equal = [
            (
                member["true_training"]["optimizer_steps"],
                member["current_training"]["optimizer_steps"],
            )
            for member in primary_members
        ] == [
            (
                member["true_training"]["optimizer_steps"],
                member["current_training"]["optimizer_steps"],
            )
            for member in replay_members
        ] == [(evaluator.OPTIMIZER_STEPS, evaluator.OPTIMIZER_STEPS)] * 3
    comparisons_equal = canonical_bytes_v1(
        primary_evaluation.get("paired_family_scene_cluster_comparisons")
    ) == canonical_bytes_v1(
        replay_evaluation.get("paired_family_scene_cluster_comparisons")
    )
    reproduction = {
        "pca_identity_and_arrays": pca_equal,
        "train_action_mean_identity_and_values": action_mean_equal,
        "task_readout_identity_and_state": task_equal,
        "model_state_identities_and_values": model_states_equal,
        "optimizer_step_counts": step_counts_equal,
        "complete_checkpoint_tree": _exact_tree_equal_v1(
            primary_checkpoint, replay_checkpoint
        ),
        "per_seed_scores": canonical_bytes_v1(
            primary_evaluation.get("prediction_diagnostics")
        )
        == canonical_bytes_v1(replay_evaluation.get("prediction_diagnostics")),
        "ensemble_scores": canonical_bytes_v1(primary_evaluation.get("score_evidence"))
        == canonical_bytes_v1(replay_evaluation.get("score_evidence")),
        "selected_actions": _selected_actions_v1(primary_evaluation)
        == _selected_actions_v1(replay_evaluation),
        "summaries": canonical_bytes_v1(
            {
                name: report.get("summary")
                for name, report in primary_evaluation["arms"].items()
            }
        )
        == canonical_bytes_v1(
            {
                name: report.get("summary")
                for name, report in replay_evaluation["arms"].items()
            }
        ),
        "bootstrap_intervals": comparisons_equal,
        "gates": canonical_bytes_v1(primary_evaluation.get("gates"))
        == canonical_bytes_v1(replay_evaluation.get("gates")),
        "verdict": canonical_bytes_v1(primary_verdict)
        == canonical_bytes_v1(replay_verdict),
        "complete_evaluation": canonical_bytes_v1(primary_evaluation)
        == canonical_bytes_v1(replay_evaluation),
        "exactly_reproduced": False,
    }
    reproduction["exactly_reproduced"] = all(
        value for key, value in reproduction.items() if key != "exactly_reproduced"
    )
    if set(reproduction) != REPRODUCTION_FIELDS:
        raise DenseVJEPACeilingRunnerError("reproduction inventory changed")
    return reproduction, dict(replay_verdict)


def _verdict_status_v1(
    verdict: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> str:
    expected = evaluator.verdict_v1(
        evaluation,
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )
    if dict(verdict) != expected or verdict.get("terminal_status") not in TERMINAL_STATUSES:
        raise DenseVJEPACeilingRunnerError("final verdict contract changed")
    return str(verdict["terminal_status"])


def _assert_inventory_v1(output_root: Path, expected: set[str]) -> None:
    observed: list[str] = []
    with os.scandir(output_root) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise DenseVJEPACeilingRunnerError(
                    "attempt root contains a non-regular file"
                )
            observed.append(entry.name)
    if set(observed) != expected or len(observed) != len(expected):
        raise DenseVJEPACeilingRunnerError(
            f"attempt inventory changed: {sorted(observed)}"
        )


def _execution_bindings_unchanged_v1(
    authority: Mapping[str, Any], authority_binding: Mapping[str, Any]
) -> None:
    closure: list[tuple[str, Mapping[str, Any]]] = [
        ("execution authority", authority_binding),
        ("preregistration", authority["preregistration_binding"]),
        ("source review", authority["source_review_binding"]),
    ]
    closure.extend(
        (f"source {label}", binding)
        for label, binding in authority["source_bindings"].items()
    )
    closure.extend(
        (f"fixed input {label}", binding)
        for label, binding in authority["input_bindings"].items()
    )
    for label, binding in closure:
        if file_binding_v1(Path(binding["path"])) != dict(binding):
            raise DenseVJEPACeilingRunnerError(f"{label} changed during execution")
    _validate_encoder_source_v1(
        authority["encoder_source"],
        source_bindings=authority["source_bindings"],
        input_bindings=authority["input_bindings"],
    )


def run_replay_v1(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
    eval_cache_receipt_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    evaluation_binding: Mapping[str, Any],
) -> dict[str, Any]:
    output_root = _safe_path(
        Path(str(authority["output_root"])), label="replay attempt root"
    )
    if output_root != DEFAULT_OUTPUT_ROOT.resolve() or not output_root.is_dir():
        raise DenseVJEPACeilingRunnerError("replay attempt root changed")
    _assert_inventory_v1(
        output_root,
        {
            "reservation.json", "vjepa2_1_eval.pt", "vjepa2_1_eval.json",
            "ceiling_checkpoint.pt", "evaluation.json",
        },
    )
    reservation, _ = _read_new_json_v1(
        output_root / "reservation.json", label="attempt reservation"
    )
    if (
        reservation.get("schema") != RESERVATION_SCHEMA
        or reservation.get("authority_binding") != authority_binding
        or reservation.get("attempt_root") != str(output_root)
        or reservation.get("consumes_attempt") is not True
    ):
        raise DenseVJEPACeilingRunnerError("attempt reservation changed")
    bundle, _bundle_audit = _load_narrow_bundle_v1(authority)
    train_groups, eval_groups, train_plan, eval_plan = _feature_plans_v1(bundle)
    train_features, train_receipt = _load_train_cache_v1(
        authority, bundle, train_plan
    )
    eval_receipt, observed_eval_receipt_binding = _read_bound_json(
        Path(eval_cache_receipt_binding["path"]),
        expected_sha256=eval_cache_receipt_binding["sha256"],
        expected_byte_count=eval_cache_receipt_binding["byte_count"],
        label="evaluation cache receipt",
    )
    if (
        observed_eval_receipt_binding != eval_cache_receipt_binding
        or observed_eval_receipt_binding["path"]
        != str((output_root / "vjepa2_1_eval.json").resolve())
    ):
        raise DenseVJEPACeilingRunnerError("evaluation cache receipt path changed")
    eval_features = _load_eval_cache_v1(
        eval_receipt, eval_plan, authority=authority, bundle=bundle
    )
    device = _authorized_device_v1()
    replay_checkpoint = evaluator.fit_primary_checkpoint_v1(
        train_groups,
        train_features,
        device,
        implementation_source_binding=authority["source_bindings"][
            "ceiling_evaluator"
        ],
    )
    replay_evaluation = _evaluate_v1(
        authority,
        replay_checkpoint,
        train_groups,
        eval_groups,
        train_features,
        eval_features,
        device,
    )
    # Primary outputs are loaded only after independent replay recomputation.
    primary_checkpoint, observed_checkpoint_binding = _load_bound_torch_once_v1(
        checkpoint_binding, label="primary ceiling checkpoint"
    )
    if observed_checkpoint_binding["path"] != str(
        (output_root / "ceiling_checkpoint.pt").resolve()
    ):
        raise DenseVJEPACeilingRunnerError("primary checkpoint path changed")
    primary_evaluation, observed_evaluation_binding = _read_bound_json(
        Path(evaluation_binding["path"]),
        expected_sha256=evaluation_binding["sha256"],
        expected_byte_count=evaluation_binding["byte_count"],
        label="primary evaluation",
    )
    if observed_evaluation_binding["path"] != str(
        (output_root / "evaluation.json").resolve()
    ):
        raise DenseVJEPACeilingRunnerError("primary evaluation path changed")
    reproduction, replay_verdict = _reproduction_v1(
        primary_checkpoint,
        replay_checkpoint,
        primary_evaluation,
        replay_evaluation,
    )
    if not reproduction["exactly_reproduced"]:
        raise DenseVJEPACeilingRunnerError(
            "fresh replay did not exactly reproduce the primary execution"
        )
    report = {
        "schema": REPLAY_SCHEMA,
        "status": REPLAY_STATUS,
        "citable_as_scientific_evidence": False,
        "authority_binding": dict(authority_binding),
        "checkpoint_binding": dict(checkpoint_binding),
        "primary_evaluation_binding": dict(evaluation_binding),
        "eval_cache_receipt_binding": dict(eval_cache_receipt_binding),
        "cache_only_feature_inputs": True,
        "fresh_process": True,
        "comparison_reference_loads": {
            "primary_checkpoint": 1,
            "primary_evaluation": 1,
        },
        "comparison_references_loaded_after_recomputation": True,
        "protected_material_opened": False,
        "rgb_access": {"train": 0, "eval": 0},
        "encoder_execution": {"vjepa2_1": 0, "other": 0},
        "cache_loads": {"train_vjepa2_1": 1, "eval_vjepa2_1": 1},
        "train_cache_receipt": train_receipt,
        "recomputed_evaluation": replay_evaluation,
        "recomputed_verdict": replay_verdict,
        "reproduction": reproduction,
    }
    _write_json_exclusive(output_root / "replay.json", report)
    _assert_inventory_v1(output_root, set(OUTPUT_NAMES[:6]))
    return report


def _launch_replay_v1(
    *,
    authority_binding: Mapping[str, Any],
    eval_cache_receipt_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    evaluation_binding: Mapping[str, Any],
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--replay",
        "--authority", str(authority_binding["path"]),
        "--expected-authority-sha256", str(authority_binding["sha256"]),
        "--expected-authority-byte-count", str(authority_binding["byte_count"]),
        "--eval-cache-receipt", str(eval_cache_receipt_binding["path"]),
        "--expected-eval-cache-receipt-sha256", str(eval_cache_receipt_binding["sha256"]),
        "--expected-eval-cache-receipt-byte-count", str(eval_cache_receipt_binding["byte_count"]),
        "--checkpoint", str(checkpoint_binding["path"]),
        "--expected-checkpoint-sha256", str(checkpoint_binding["sha256"]),
        "--expected-checkpoint-byte-count", str(checkpoint_binding["byte_count"]),
        "--evaluation", str(evaluation_binding["path"]),
        "--expected-evaluation-sha256", str(evaluation_binding["sha256"]),
        "--expected-evaluation-byte-count", str(evaluation_binding["byte_count"]),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DenseVJEPACeilingRunnerError(
            f"fresh replay process failed: {detail[-2_000:]}"
        )


def _validate_replay_v1(
    replay: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
    eval_cache_receipt_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    evaluation_binding: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema", "status", "citable_as_scientific_evidence",
        "authority_binding", "checkpoint_binding", "primary_evaluation_binding",
        "eval_cache_receipt_binding", "cache_only_feature_inputs", "fresh_process",
        "comparison_reference_loads",
        "comparison_references_loaded_after_recomputation",
        "protected_material_opened", "rgb_access", "encoder_execution",
        "cache_loads", "train_cache_receipt", "recomputed_evaluation",
        "recomputed_verdict", "reproduction",
    }
    reproduction = replay.get("reproduction")
    if (
        set(replay) != required
        or replay.get("schema") != REPLAY_SCHEMA
        or replay.get("status") != REPLAY_STATUS
        or replay.get("citable_as_scientific_evidence") is not False
        or replay.get("authority_binding") != authority_binding
        or replay.get("checkpoint_binding") != checkpoint_binding
        or replay.get("primary_evaluation_binding") != evaluation_binding
        or replay.get("eval_cache_receipt_binding") != eval_cache_receipt_binding
        or replay.get("cache_only_feature_inputs") is not True
        or replay.get("fresh_process") is not True
        or replay.get("comparison_reference_loads")
        != {"primary_checkpoint": 1, "primary_evaluation": 1}
        or replay.get("comparison_references_loaded_after_recomputation") is not True
        or replay.get("protected_material_opened") is not False
        or replay.get("rgb_access") != {"train": 0, "eval": 0}
        or replay.get("encoder_execution") != {"vjepa2_1": 0, "other": 0}
        or replay.get("cache_loads")
        != {"train_vjepa2_1": 1, "eval_vjepa2_1": 1}
        or not isinstance(reproduction, Mapping)
        or set(reproduction) != REPRODUCTION_FIELDS
        or any(value is not True for value in reproduction.values())
        or canonical_bytes_v1(replay.get("recomputed_evaluation"))
        != canonical_bytes_v1(evaluation)
    ):
        raise DenseVJEPACeilingRunnerError(
            "fresh replay report did not reproduce the primary execution"
        )
    verdict = replay.get("recomputed_verdict")
    if not isinstance(verdict, Mapping):
        raise DenseVJEPACeilingRunnerError("replay verdict is absent")
    _verdict_status_v1(verdict, evaluation)
    return dict(verdict)


def execute_v1(
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, Any]
) -> dict[str, Any]:
    output_root = _safe_path(
        Path(str(authority["output_root"])),
        label="dense V-JEPA ceiling output",
        must_exist=False,
    )
    if output_root != DEFAULT_OUTPUT_ROOT.resolve():
        raise DenseVJEPACeilingRunnerError("attempt output root changed")
    _safe_path(output_root.parent, label="attempt output parent", must_exist=False)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        output_root / "reservation.json",
        {
            "schema": RESERVATION_SCHEMA,
            "authority_binding": dict(authority_binding),
            "attempt_root": str(output_root),
            "owner_pid": os.getpid(),
            "consumes_attempt": True,
            "reserved_before_cache_deserialization_or_rgb_decode": True,
        },
    )
    bundle, bundle_audit = _load_narrow_bundle_v1(authority)
    train_groups, eval_groups, train_plan, eval_plan = _feature_plans_v1(bundle)
    _eval_rgb_bindings_from_bundle_v1(authority, bundle, eval_plan)
    train_features, train_receipt = _load_train_cache_v1(
        authority, bundle, train_plan
    )
    device = _authorized_device_v1()
    checkpoint = evaluator.fit_primary_checkpoint_v1(
        train_groups,
        train_features,
        device,
        implementation_source_binding=authority["source_bindings"][
            "ceiling_evaluator"
        ],
    )
    if not isinstance(checkpoint, Mapping):
        raise DenseVJEPACeilingRunnerError("evaluator returned a non-object checkpoint")
    checkpoint_path = output_root / "ceiling_checkpoint.pt"
    _save_torch_exclusive(checkpoint_path, checkpoint)
    checkpoint_binding = file_binding_v1(checkpoint_path)

    # The frozen fit and durable checkpoint precede encoder load and every eval RGB open.
    eval_receipt = extract_eval_feature_cache_v1(
        authority,
        bundle,
        eval_plan,
        device=device,
        output_path=output_root / "vjepa2_1_eval.pt",
    )
    eval_receipt_binding = file_binding_v1(output_root / "vjepa2_1_eval.json")
    eval_features = _load_eval_cache_v1(
        eval_receipt, eval_plan, authority=authority, bundle=bundle
    )
    evaluation = _evaluate_v1(
        authority,
        checkpoint,
        train_groups,
        eval_groups,
        train_features,
        eval_features,
        device,
    )
    evaluation_path = output_root / "evaluation.json"
    _write_json_exclusive(evaluation_path, evaluation)
    evaluation_binding = file_binding_v1(evaluation_path)
    del checkpoint, train_features, eval_features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _assert_inventory_v1(output_root, set(OUTPUT_NAMES[:5]))
    _execution_bindings_unchanged_v1(authority, authority_binding)
    _launch_replay_v1(
        authority_binding=authority_binding,
        eval_cache_receipt_binding=eval_receipt_binding,
        checkpoint_binding=checkpoint_binding,
        evaluation_binding=evaluation_binding,
    )
    replay, replay_binding = _read_new_json_v1(
        output_root / "replay.json", label="fresh replay"
    )
    verdict = _validate_replay_v1(
        replay,
        authority_binding=authority_binding,
        eval_cache_receipt_binding=eval_receipt_binding,
        checkpoint_binding=checkpoint_binding,
        evaluation_binding=evaluation_binding,
        evaluation=evaluation,
    )
    _execution_bindings_unchanged_v1(authority, authority_binding)
    status = _verdict_status_v1(verdict, evaluation)
    report = {
        "schema": SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "development_only": True,
        "claim_scope": (
            "PRIVILEGED_ACTUAL_FUTURE_VJEPA_REPRESENTATION_INTERFACE_CEILING"
        ),
        "navigation_usefulness_established": False,
        "authorizes_model_training": False,
        "authorizes_downstream_successor_execution": False,
        "authority_binding": dict(authority_binding),
        "preregistration_binding": dict(authority["preregistration_binding"]),
        "input_classification": _input_classification_v1(),
        "source_bundle_manifest": dict(bundle.manifest_binding),
        "bundle_access_audit": dict(bundle.access_audit),
        "bundle_validation": bundle_audit,
        "artifact_bindings": {
            "eval_cache": dict(eval_receipt["binding"]),
            "eval_cache_receipt": eval_receipt_binding,
            "ceiling_checkpoint": checkpoint_binding,
            "evaluation": evaluation_binding,
            "replay": replay_binding,
        },
        "cache_receipts": {"train": train_receipt, "eval": eval_receipt},
        "access_counts": {
            "primary": {
                "train_rgb": 0,
                "eval_rgb": ROLE_ARTIFACT_COUNT,
                "eval_context_rgb": EVAL_CONTEXT_COUNT,
                "eval_successor_rgb": EVAL_SUCCESSOR_COUNT,
                "vjepa_encoded_frames": ROLE_ARTIFACT_COUNT,
            },
            "replay": {
                "train_rgb": 0,
                "eval_rgb": 0,
                "encoder_execution": 0,
                "feature_cache_loads": {
                    "train_vjepa2_1": 1,
                    "eval_vjepa2_1": 1,
                },
                "comparison_reference_loads": {
                    "primary_checkpoint": 1,
                    "primary_evaluation": 1,
                },
            },
        },
        "evaluation": evaluation,
        "replay": {
            "binding": replay_binding,
            "exactly_reproduced": True,
            "fresh_process": True,
            "cache_only_feature_inputs": True,
            "feature_cache_loads": {"train_vjepa2_1": 1, "eval_vjepa2_1": 1},
            "comparison_reference_loads": {
                "primary_checkpoint": 1,
                "primary_evaluation": 1,
            },
            "comparison_references_loaded_after_recomputation": True,
        },
        "verdict": verdict,
    }
    _write_json_exclusive(output_root / "result.json", report)
    _assert_inventory_v1(output_root, set(OUTPUT_NAMES[:7]))
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "authorizes_model_training": False,
        "authorizes_downstream_successor_execution": False,
        "result_binding": file_binding_v1(output_root / "result.json"),
        "deterministic_replay_passed": True,
        "access_counts": report["access_counts"],
        "failure": None,
    }
    _write_json_exclusive(output_root / "terminal.json", terminal)
    _assert_inventory_v1(output_root, set(OUTPUT_NAMES))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--eval-cache-receipt", type=Path)
    parser.add_argument("--expected-eval-cache-receipt-sha256")
    parser.add_argument("--expected-eval-cache-receipt-byte-count", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-checkpoint-byte-count", type=int)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--expected-evaluation-sha256")
    parser.add_argument("--expected-evaluation-byte-count", type=int)
    args = parser.parse_args(argv)
    authority, authority_binding = _read_authority(
        args.authority,
        expected_sha256=args.expected_authority_sha256,
        expected_byte_count=args.expected_authority_byte_count,
        replay_mode=args.replay,
    )
    if args.replay:
        replay_values = (
            args.eval_cache_receipt,
            args.expected_eval_cache_receipt_sha256,
            args.expected_eval_cache_receipt_byte_count,
            args.checkpoint,
            args.expected_checkpoint_sha256,
            args.expected_checkpoint_byte_count,
            args.evaluation,
            args.expected_evaluation_sha256,
            args.expected_evaluation_byte_count,
        )
        if any(value is None for value in replay_values):
            raise DenseVJEPACeilingRunnerError(
                "fresh replay caller bindings are incomplete"
            )
        checkpoint_binding = _binding(
            args.checkpoint,
            args.expected_checkpoint_sha256,
            args.expected_checkpoint_byte_count,
        )
        evaluation_binding = _binding(
            args.evaluation,
            args.expected_evaluation_sha256,
            args.expected_evaluation_byte_count,
        )
        replay = run_replay_v1(
            authority,
            authority_binding=authority_binding,
            eval_cache_receipt_binding=_binding(
                args.eval_cache_receipt,
                args.expected_eval_cache_receipt_sha256,
                args.expected_eval_cache_receipt_byte_count,
            ),
            checkpoint_binding=checkpoint_binding,
            evaluation_binding=evaluation_binding,
        )
        print(json.dumps({"status": replay["status"]}, sort_keys=True))
        return 0
    if any(
        value is not None
        for value in (
            args.eval_cache_receipt,
            args.expected_eval_cache_receipt_sha256,
            args.expected_eval_cache_receipt_byte_count,
            args.checkpoint,
            args.expected_checkpoint_sha256,
            args.expected_checkpoint_byte_count,
            args.evaluation,
            args.expected_evaluation_sha256,
            args.expected_evaluation_byte_count,
        )
    ):
        raise DenseVJEPACeilingRunnerError(
            "primary execution received replay-only arguments"
        )
    output_root = Path(str(authority["output_root"]))
    existed = output_root.exists()
    try:
        report = execute_v1(authority, authority_binding=authority_binding)
    except Exception as error:
        if (
            not existed
            and output_root.is_dir()
            and not (output_root / "terminal.json").exists()
        ):
            _write_json_exclusive(
                output_root / "terminal.json",
                {
                    "schema": TERMINAL_SCHEMA,
                    "status": FAIL_STATUS,
                    "citable_as_scientific_evidence": False,
                    "authorizes_retry_or_resume": False,
                    "authorizes_model_training": False,
                    "authorizes_downstream_successor_execution": False,
                    "result_binding": None,
                    "deterministic_replay_passed": False,
                    "access_counts": None,
                    "failure": {
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                },
            )
        raise
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "DEFAULT_OUTPUT_ROOT",
    "DenseVJEPACeilingRunnerError",
    "EINOPS_DISTRIBUTION_METADATA_BINDINGS",
    "EINOPS_RUNTIME_SOURCE_BINDINGS",
    "EVAL_CACHE_RECEIPT_SCHEMA",
    "EVAL_CACHE_SCHEMA",
    "FAIL_STATUS",
    "LINEAGE_WITNESS_LABELS",
    "OUTPUT_NAMES",
    "PASS_STATUS",
    "PERMISSION_FIELDS",
    "PREREGISTRATION_BYTE_COUNT",
    "PREREGISTRATION_SHA256",
    "REPLAY_SCHEMA",
    "REPLAY_STATUS",
    "SCIENTIFIC_INPUT_LABELS",
    "SOURCE_PATHS",
    "SOURCE_REVIEW_CHECKS",
    "SOURCE_REVIEW_FIELDS",
    "SOURCE_REVIEW_SCHEMA",
    "SOURCE_REVIEW_STATUS",
    "STOP_STATUS",
    "VJEPA_TRANSITIVE_SOURCE_BINDINGS",
    "config_v1",
    "einops_dependency_v1",
    "execute_v1",
    "extract_eval_feature_cache_v1",
    "feature_preprocessing_contract_v1",
    "file_binding_v1",
    "main",
    "permissions_v1",
    "run_replay_v1",
)
