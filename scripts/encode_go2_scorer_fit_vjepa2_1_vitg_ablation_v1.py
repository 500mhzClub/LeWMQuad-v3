#!/usr/bin/env python3
"""Encode the frozen 1,440-row scorer view with V-JEPA 2.1 ViT-g.

This is a representation-scale ablation only.  It reuses the completed
oracle-v1.3 training view and its existing RGB frames, writes a separate
latent namespace, and never enters simulation, rendering, prediction, or a
final benchmark path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource as process_resource
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import encode_go2_scorer_fit_oracle_v1_3 as BASE  # noqa: E402


STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
SCHEMA = "go2_scorer_fit_vjepa2_1_vitg_ablation_v1_encoding_contract_v1"
LATENT_INDEX_SCHEMA = "go2_scorer_fit_vjepa2_1_vitg_ablation_v1_latent_index_v1"
ENCODING_RECEIPT_SCHEMA = (
    "go2_scorer_fit_vjepa2_1_vitg_ablation_v1_encoding_receipt_v1"
)
LATENT_INDEX_SELF_KEY = "latent_index_digest"
ENCODING_RECEIPT_SELF_KEY = "encoding_receipt_digest"
ENCODING_CONTRACT_SELF_KEY = "encoding_contract_digest"

GENERATED_ROOT = Path(".generated/go2_scorer_fit_vjepa2_1_vitg_ablation_v1")
REGISTERED_GENERATED_TARGET_ROOT = Path(
    "/home/andrewknowles/.local/share/lewm_go2_planning_utility_v1_2/active/"
    "go2_scorer_fit_vjepa2_1_vitg_ablation_v1"
)
RESOURCE_SMOKE_RECEIPT_NAME = "resource_smoke_receipt.json"
ENCODING_CONTRACT_NAME = "encoding_contract.json"
ENCODED_DIRECTORY_NAME = "encoded_training_view"
LATENT_INDEX_NAME = "latent_index.json"
ENCODING_RECEIPT_NAME = "encoding_receipt.json"

EXPECTED_TRAINING_VIEW_DIGEST = (
    "9eefff24953fdfc1eb7718ff6067a9bc06f5f8bd321f62769521234d6393291c"
)
EXPECTED_ROWS = 1_440
EXPECTED_FIT_ROWS = 1_152
EXPECTED_CALIBRATION_ROWS = 288
TOKENS = 768
TOKEN_DIM = 1_408
HORIZONS = 4
HORIZON_SHAPE = (HORIZONS, TOKENS, TOKEN_DIM)
SHARD_BYTES = int(np.prod(HORIZON_SHAPE)) * np.dtype(np.float16).itemsize
TOTAL_LATENT_BYTES = EXPECTED_ROWS * SHARD_BYTES
MIN_FREE_STORAGE_BYTES = 50 * (1 << 30)
MIN_DEVICE_TOTAL_MEMORY_BYTES = 30_000_000_000
MIN_DEVICE_FREE_MEMORY_BYTES = 26 * (1 << 30)
ENCODER_BATCH_FRAMES = 4
DEFAULT_LOADER_WORKERS = 4
MIN_LOADER_WORKERS = 4
MAX_LOADER_WORKERS = 8
EXPECTED_PARAMETER_COUNT = 1_013_267_968
SOURCE_DIGEST_KEYS = BASE.SOURCE_DIGEST_KEYS
RECORD_KEYS = frozenset((
    "training_view_row_digest", "state_id", "state_identity_digest",
    "candidate_index", "source_kind", "path", "sha256", "byte_count",
    "shape",
))
PROJECT_SOURCE_PATHS = (
    "scripts/encode_go2_scorer_fit_vjepa2_1_vitg_ablation_v1.py",
    "scripts/vjepa2_1_vitg_frozen_encoder_ablation_v1.py",
)


class VitGEncodingError(RuntimeError):
    """The scale-ablation input, output, or resource contract changed."""


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VitGEncodingError(message)


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _signed(value: Mapping[str, Any], self_key: str) -> dict[str, Any]:
    payload = dict(value)
    _require(self_key not in payload, f"{self_key} already present")
    payload[self_key] = canonical_digest(payload)
    return payload


def _validate_signed(value: Mapping[str, Any], self_key: str,
                     label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    payload = dict(value)
    recorded = payload.pop(self_key, None)
    _require(_is_digest(recorded) and recorded == canonical_digest(payload),
             f"{label} self digest does not verify")
    payload[self_key] = recorded
    return payload


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def generated_root(root: Path = ROOT) -> Path:
    return root / GENERATED_ROOT


def resource_smoke_receipt_path(root: Path = ROOT) -> Path:
    return generated_root(root) / RESOURCE_SMOKE_RECEIPT_NAME


def encoding_contract_path(root: Path = ROOT) -> Path:
    return generated_root(root) / ENCODING_CONTRACT_NAME


def encoded_root(root: Path = ROOT) -> Path:
    return generated_root(root) / ENCODED_DIRECTORY_NAME


def latent_index_path(root: Path = ROOT) -> Path:
    return encoded_root(root) / LATENT_INDEX_NAME


def encoding_receipt_path(root: Path = ROOT) -> Path:
    return encoded_root(root) / ENCODING_RECEIPT_NAME


def _managed_root(root: Path = ROOT, *, require_free_space: bool) -> Path:
    logical = generated_root(root).absolute()
    if root.resolve() == ROOT.resolve():
        _require(logical.is_symlink(), "registered ViT-g output alias is absent")
        raw = logical.readlink()
        target = raw if raw.is_absolute() else logical.parent / raw
        _require(target == REGISTERED_GENERATED_TARGET_ROOT,
                 "registered ViT-g output alias changed")
        physical = REGISTERED_GENERATED_TARGET_ROOT
        _require(physical.is_dir() and not physical.is_symlink(),
                 "registered ViT-g physical root is invalid")
    else:
        _require(logical.is_dir() and not logical.is_symlink(),
                 "synthetic ViT-g output root is invalid")
        physical = logical
    if require_free_space:
        free = int(shutil.disk_usage(physical).free)
        _require(free >= MIN_FREE_STORAGE_BYTES,
                 f"ViT-g output storage has only {free} free bytes")
    return logical


def _guarded_output(relative: str | Path, *, root: Path = ROOT) -> Path:
    relative = Path(relative)
    _require(not relative.is_absolute() and relative.parts
             and ".." not in relative.parts,
             "ViT-g output path must be a relative descendant")
    logical = _managed_root(root, require_free_space=False)
    candidate = (logical / relative).absolute()
    _require(logical in candidate.parents,
             "ViT-g output path escaped its dedicated root")
    cursor = candidate.parent
    while cursor != logical.parent:
        _require(cursor == logical or not cursor.is_symlink(),
                 f"ViT-g nested output ancestor is a symlink: {cursor}")
        if cursor == logical:
            break
        cursor = cursor.parent
    return candidate


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is absent or not a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise VitGEncodingError(f"{label} is invalid JSON") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _write_all(descriptor: int, raw: bytes) -> None:
    position = 0
    while position < len(raw):
        position += os.write(descriptor, raw[position:])


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_operational_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically advance the resumable index; never install a symlink."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.is_symlink(), "operational index path is a symlink")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.partial")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        _write_all(descriptor, _json_bytes(value))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _publish_json_once(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0), 0o444)
    except FileExistsError:
        _require(path.is_file() and not path.is_symlink()
                 and path.read_bytes() == raw,
                 f"{label} is already different")
        return
    try:
        _write_all(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def atomic_missing_f16(path: Path, array: np.ndarray) -> tuple[str, int]:
    """Publish one immutable FP16 shard with a no-overwrite hard link."""

    value = np.ascontiguousarray(array, dtype=np.float16)
    _require(tuple(value.shape) == HORIZON_SHAPE,
             f"ViT-g shard shape changed: {list(value.shape)}")
    raw = value.tobytes(order="C")
    _require(len(raw) == SHARD_BYTES, "ViT-g shard byte count changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists() and not path.is_symlink(),
             f"refusing to replace ViT-g shard {path}")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.partial")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        _write_all(descriptor, raw)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise VitGEncodingError(f"ViT-g shard appeared concurrently: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _validate_resource_smoke_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    from scripts import vjepa2_1_vitg_frozen_encoder_ablation_v1 as runtime

    checkpoint_value = value.get("checkpoint_binding")
    _require(isinstance(checkpoint_value, Mapping)
             and _is_digest(checkpoint_value.get("sha256")),
             "ViT-g resource smoke checkpoint binding is malformed")
    validated = runtime.validate_resource_smoke_receipt_v1(
        value, expected_checkpoint_sha256=checkpoint_value["sha256"])
    _require(isinstance(validated, Mapping),
             "ViT-g resource smoke validator returned no receipt")
    result = dict(validated)
    checkpoint = result.get("checkpoint_binding")
    probes = result.get("probes")
    selected_batch = result.get("maximum_passing_batch_size")
    execution_modes = {
        "torch.bfloat16": "bfloat16_autocast_fp32_weights",
        "torch.float32": "float32_no_autocast",
    }
    _require(result.get("status") == runtime.RECEIPT_STATUS_PASS
             and result.get("encoder_contract_digest")
             == runtime.ENCODER_CONTRACT_DIGEST
             and result.get("parameter_count") == EXPECTED_PARAMETER_COUNT
             and result.get("predictor_constructed") is False
             and result.get("predictor_checkpoint_state_access_count") == 0
             and result.get("scientific_labels_opened") == 0
             and result.get("corpus_frames_opened")
             == runtime.SMOKE_IMAGE_COUNT
             and isinstance(result.get("smoke_images"), list)
             and len(result["smoke_images"]) == runtime.SMOKE_IMAGE_COUNT
             and _is_digest(result.get("receipt_sha256"))
             and isinstance(checkpoint, Mapping)
             and _is_digest(checkpoint.get("sha256"))
             and isinstance(checkpoint.get("byte_count"), int)
             and checkpoint["byte_count"] > 0
             and result.get("parameter_dtype") == "torch.float32"
             and result.get("execution_mode")
             == execution_modes.get(result.get("inference_dtype"))
             and isinstance(probes, list)
             and selected_batch in {1, 2, ENCODER_BATCH_FRAMES}
             and any(probe.get("batch_size") == selected_batch
                     and probe.get("status") == "PASS"
                     and probe.get("output_shape")
                     == [selected_batch, TOKENS, TOKEN_DIM]
                     and probe.get("output_finite") is True
                     for probe in probes if isinstance(probe, Mapping)),
             "ViT-g resource smoke receipt changed")
    return result


def load_resource_smoke_receipt(*, root: Path = ROOT) -> dict[str, Any]:
    return _validate_resource_smoke_receipt(_read_json(
        resource_smoke_receipt_path(root), label="ViT-g resource smoke receipt"))


def selected_batch_frames(resource_receipt: Mapping[str, Any]) -> int:
    selected = resource_receipt.get("maximum_passing_batch_size")
    _require(selected in {1, 2, ENCODER_BATCH_FRAMES},
             "ViT-g smoke has no supported passing batch")
    return int(selected)


def _current_project_source_binding(*, root: Path = ROOT) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VitGEncodingError("could not bind the project source commit") from exc
    _require(len(commit) == 40 and all(character in "0123456789abcdef"
                                       for character in commit),
             "project source commit is malformed")
    _require(not status.strip(),
             "project source tree must be clean before freezing encoding")
    files: dict[str, dict[str, Any]] = {}
    for relative in PROJECT_SOURCE_PATHS:
        path = root / relative
        _require(path.is_file() and not path.is_symlink(),
                 f"encoding source file is absent: {relative}")
        files[relative] = {
            "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        }
    return {"source_commit": commit, "clean": True, "files": files}


def _validate_project_source_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, Mapping)
             and isinstance(value.get("source_commit"), str)
             and len(value["source_commit"]) == 40
             and value.get("clean") is True
             and isinstance(value.get("files"), Mapping)
             and set(value["files"]) == set(PROJECT_SOURCE_PATHS),
             "encoding project source binding is malformed")
    for relative in PROJECT_SOURCE_PATHS:
        binding = value["files"].get(relative)
        _require(isinstance(binding, Mapping)
                 and _is_digest(binding.get("sha256"))
                 and isinstance(binding.get("byte_count"), int)
                 and binding["byte_count"] > 0,
                 f"encoding source binding is malformed: {relative}")
    return dict(value)


def encoding_contract(
        resource_receipt: Mapping[str, Any], *,
        project_source_binding: Mapping[str, Any],
        ) -> dict[str, Any]:
    checkpoint = resource_receipt["checkpoint_binding"]
    encoder_contract = resource_receipt["encoder_contract"]
    source_binding = _validate_project_source_binding(project_source_binding)
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "ablation": "V-JEPA 2.1 ViT-L to ViT-g representation scale only",
        "training_view_digest": EXPECTED_TRAINING_VIEW_DIGEST,
        "row_count": EXPECTED_ROWS,
        "fit_rows": EXPECTED_FIT_ROWS,
        "calibration_rows": EXPECTED_CALIBRATION_ROWS,
        "encoder_contract_digest": resource_receipt["encoder_contract_digest"],
        "encoder_source_binding_digest": canonical_digest(
            resource_receipt["source_binding"]),
        "preprocess_contract_digest": canonical_digest(
            encoder_contract["preprocessing"]),
        "resource_smoke_receipt_digest": resource_receipt["receipt_sha256"],
        "project_source_binding": source_binding,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_byte_count": checkpoint["byte_count"],
        "compute_dtype": resource_receipt["inference_dtype"],
        "execution_mode": resource_receipt["execution_mode"],
        "parameter_dtype": resource_receipt["parameter_dtype"],
        "latent_shape": list(HORIZON_SHAPE),
        "latent_storage_dtype": "float16",
        "latent_shard_bytes": SHARD_BYTES,
        "total_latent_bytes": TOTAL_LATENT_BYTES,
        "output_root": str(GENERATED_ROOT),
        "registered_physical_root": str(REGISTERED_GENERATED_TARGET_ROOT),
        "minimum_free_storage_bytes": MIN_FREE_STORAGE_BYTES,
        "minimum_device_total_memory_bytes": MIN_DEVICE_TOTAL_MEMORY_BYTES,
        "minimum_device_free_memory_bytes": MIN_DEVICE_FREE_MEMORY_BYTES,
        "selected_batch_frames": selected_batch_frames(resource_receipt),
        "horizon_frames_per_row": HORIZONS,
        "loader_workers": {
            "default": DEFAULT_LOADER_WORKERS,
            "minimum": MIN_LOADER_WORKERS,
            "maximum": MAX_LOADER_WORKERS,
            "shuffle": False,
        },
        "write_policy": "atomic missing-only shards; monotonic resumable index",
        "simulator_runs": 0,
        "renders_generated": 0,
        "predictor_checkpoints_opened": 0,
        "final_200_state_corpus_generated": False,
    }


def _validate_encoding_contract(
        value: Mapping[str, Any], *, resource_receipt: Mapping[str, Any],
        ) -> dict[str, Any]:
    frozen = _validate_signed(
        value, ENCODING_CONTRACT_SELF_KEY, "ViT-g encoding contract")
    expected = encoding_contract(
        resource_receipt,
        project_source_binding=frozen.get("project_source_binding", {}))
    _require(frozen == _signed(expected, ENCODING_CONTRACT_SELF_KEY),
             "ViT-g frozen encoding contract changed")
    return frozen


def freeze_encoding_contract(
        *, resource_receipt: Mapping[str, Any], root: Path = ROOT,
        ) -> dict[str, Any]:
    source_binding = _current_project_source_binding(root=root)
    contract = _signed(encoding_contract(
        resource_receipt, project_source_binding=source_binding),
        ENCODING_CONTRACT_SELF_KEY)
    path = encoding_contract_path(root)
    _publish_json_once(path, contract, label="ViT-g encoding contract")
    return _validate_encoding_contract(
        _read_json(path, label="ViT-g encoding contract"),
        resource_receipt=resource_receipt)


def load_encoding_contract(
        *, resource_receipt: Mapping[str, Any], root: Path = ROOT,
        ) -> dict[str, Any]:
    return _validate_encoding_contract(
        _read_json(encoding_contract_path(root),
                   label="ViT-g encoding contract"),
        resource_receipt=resource_receipt)


def _load_preserved_training_view(root: Path) -> dict[str, Any]:
    from scripts import train_go2_utility_scorer_v1_3 as trainer

    return trainer.load_preserved_encoded_training_view_for_replacement(
        root=root, verify_encoder_checkpoint=False)["view"]


def load_training_view(*, root: Path = ROOT) -> dict[str, Any]:
    view = _load_preserved_training_view(root)
    view = BASE.validate_training_view_structure(view)
    _require(view.get(BASE.WORKFLOW.TRAINING_VIEW_SELF_KEY)
             == EXPECTED_TRAINING_VIEW_DIGEST,
             "ViT-g ablation received another training view")
    _require(len(view["rows"]) == EXPECTED_ROWS,
             "ViT-g ablation training view is not 1,440 rows")
    return view


def _ordered_rows(view: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = sorted(view["rows"], key=lambda row: (
        BASE._row_role(row) != "fit", str(row["state_id"]),
        int(row["candidate_index"])))
    _require(len(rows) == EXPECTED_ROWS
             and sum(BASE._row_role(row) == "fit" for row in rows)
             == EXPECTED_FIT_ROWS,
             "ViT-g row ordering changed")
    return rows


def _record_projection(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: record[key] for key in sorted(RECORD_KEYS)}
            for record in records]


def latent_content_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest(_record_projection(records))


def _record(path: Path, row: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    return {
        "training_view_row_digest": row["training_view_row_digest"],
        "state_id": row["state_id"],
        "state_identity_digest": row["state_identity_digest"],
        "candidate_index": int(row["candidate_index"]),
        "source_kind": row["source_kind"],
        "path": str(path.relative_to(encoded_root(root))),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
        "shape": list(HORIZON_SHAPE),
    }


def _adoptable_unindexed_shard(path: Path) -> bool:
    """Recognise only a completed atomic shard from the index crash window."""

    if not path.is_file() or path.is_symlink():
        return False
    metadata = path.stat()
    if (not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_size != SHARD_BYTES):
        return False
    try:
        values = np.memmap(
            path, mode="r", dtype=np.float16, shape=HORIZON_SHAPE)
        finite = bool(np.isfinite(values).all())
        del values
        return finite
    except (OSError, ValueError, TypeError):
        return False


def _valid_record(path: Path, record: Any, row: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(record, Mapping) and set(record) == RECORD_KEYS
        and path.is_file() and not path.is_symlink()
        and record.get("training_view_row_digest")
        == row.get("training_view_row_digest")
        and record.get("state_id") == row.get("state_id")
        and record.get("state_identity_digest")
        == row.get("state_identity_digest")
        and record.get("candidate_index") == row.get("candidate_index")
        and record.get("source_kind") == row.get("source_kind")
        and record.get("shape") == list(HORIZON_SHAPE)
        and record.get("byte_count") == SHARD_BYTES
        and path.stat().st_size == SHARD_BYTES
        and file_sha256(path) == record.get("sha256")
    )


def _execution_stats(*, started: float, resumed: int, adopted: int,
                     encoded: int, loader_workers: int,
                     device: Any | None) -> dict[str, Any]:
    elapsed = max(0.0, time.monotonic() - started)
    peak_vram = 0
    if device is not None:
        import torch
        peak_vram = int(torch.cuda.max_memory_allocated(device))
    self_rss = int(process_resource.getrusage(
        process_resource.RUSAGE_SELF).ru_maxrss) * 1024
    child_rss = int(process_resource.getrusage(
        process_resource.RUSAGE_CHILDREN).ru_maxrss) * 1024
    return {
        "loader_workers": loader_workers,
        "resumed_shard_count": resumed,
        "adopted_unindexed_shard_count": adopted,
        "new_shard_count": encoded,
        "invalid_existing_shard_count": 0,
        "encoded_frame_count": encoded * HORIZONS,
        "wall_seconds": elapsed,
        "new_frames_per_second": (
            encoded * HORIZONS / elapsed if encoded and elapsed > 0 else None),
        "peak_vram_bytes": peak_vram,
        "peak_process_rss_bytes": self_rss,
        "peak_child_worker_rss_bytes": child_rss,
    }


def _index_payload(view: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                   *, resource_receipt: Mapping[str, Any],
                   contract: Mapping[str, Any],
                   complete: bool, execution: Mapping[str, Any]) -> dict[str, Any]:
    contract_digest = contract[ENCODING_CONTRACT_SELF_KEY]
    return _signed({
        "schema": LATENT_INDEX_SCHEMA,
        "status": STATUS,
        "complete": bool(complete),
        "encoding_contract_digest": contract_digest,
        "resource_smoke_receipt_digest": resource_receipt["receipt_sha256"],
        "training_view_digest": EXPECTED_TRAINING_VIEW_DIGEST,
        "oracle_v1_3_digest": view["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest":
            view["scorer_fit_oracle_v1_3_contract_digest"],
        "authority_digest": view["authority_digest"],
        **{key: view[key] for key in SOURCE_DIGEST_KEYS},
        "row_count": EXPECTED_ROWS,
        "fit_rows": EXPECTED_FIT_ROWS,
        "calibration_rows": EXPECTED_CALIBRATION_ROWS,
        "horizons": HORIZONS,
        "selected_batch_frames": contract["selected_batch_frames"],
        "tokens": TOKENS,
        "token_dim": TOKEN_DIM,
        "horizon_shape": [len(records), *HORIZON_SHAPE],
        "encoder_contract_digest": resource_receipt["encoder_contract_digest"],
        "encoder_source_binding_digest":
            contract["encoder_source_binding_digest"],
        "preprocess_contract_digest": contract["preprocess_contract_digest"],
        "target_encoder_checkpoint_sha256":
            resource_receipt["checkpoint_binding"]["sha256"],
        "encoder_compute_dtype": resource_receipt["inference_dtype"],
        "latent_storage_dtype": "float16",
        "latent_content_digest": latent_content_digest(records),
        "execution": dict(execution),
        "horizon_records": list(records),
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "simulator_runs": 0,
        "renders_generated": 0,
        "final_200_state_corpus_generated": False,
    }, LATENT_INDEX_SELF_KEY)


def _validate_index(index: Mapping[str, Any], view: Mapping[str, Any],
                    *, resource_receipt: Mapping[str, Any],
                    contract: Mapping[str, Any], root: Path,
                    require_complete: bool) -> dict[str, Any]:
    value = _validate_signed(index, LATENT_INDEX_SELF_KEY, "ViT-g latent index")
    contract_digest = contract[ENCODING_CONTRACT_SELF_KEY]
    _require(value.get("schema") == LATENT_INDEX_SCHEMA
             and value.get("status") == STATUS
             and value.get("complete") is require_complete
             and value.get("encoding_contract_digest") == contract_digest
             and value.get("resource_smoke_receipt_digest")
             == resource_receipt["receipt_sha256"]
             and value.get("training_view_digest")
             == EXPECTED_TRAINING_VIEW_DIGEST,
             "ViT-g latent index contract changed")
    for key in ("oracle_v1_3_digest",
                "scorer_fit_oracle_v1_3_contract_digest", "authority_digest",
                *SOURCE_DIGEST_KEYS):
        _require(value.get(key) == view.get(key),
                 f"ViT-g latent index changed source binding {key}")
    records = value.get("horizon_records")
    execution = value.get("execution")
    _require(isinstance(records, list)
             and len(records) <= EXPECTED_ROWS
             and value.get("horizon_shape") == [len(records), *HORIZON_SHAPE]
             and value.get("row_count") == EXPECTED_ROWS
             and value.get("fit_rows") == EXPECTED_FIT_ROWS
             and value.get("calibration_rows") == EXPECTED_CALIBRATION_ROWS
             and value.get("horizons") == HORIZONS
             and value.get("selected_batch_frames")
             == contract["selected_batch_frames"]
             and value.get("tokens") == TOKENS
             and value.get("token_dim") == TOKEN_DIM
             and value.get("encoder_contract_digest")
             == resource_receipt["encoder_contract_digest"]
             and value.get("encoder_source_binding_digest")
             == contract["encoder_source_binding_digest"]
             and value.get("preprocess_contract_digest")
             == contract["preprocess_contract_digest"]
             and value.get("target_encoder_checkpoint_sha256")
             == resource_receipt["checkpoint_binding"]["sha256"]
             and value.get("encoder_compute_dtype")
             == resource_receipt["inference_dtype"]
             and value.get("latent_storage_dtype") == "float16"
             and value.get("latent_content_digest")
             == latent_content_digest(records)
             and isinstance(execution, Mapping)
             and isinstance(execution.get("loader_workers"), int)
             and MIN_LOADER_WORKERS <= execution["loader_workers"]
             <= MAX_LOADER_WORKERS
             and all(isinstance(execution.get(key), int)
                     and execution[key] >= 0 for key in (
                         "resumed_shard_count",
                         "adopted_unindexed_shard_count",
                         "new_shard_count", "invalid_existing_shard_count",
                         "encoded_frame_count", "peak_vram_bytes",
                         "peak_process_rss_bytes",
                         "peak_child_worker_rss_bytes"))
             and execution.get("invalid_existing_shard_count") == 0
             and execution.get("encoded_frame_count")
             == execution.get("new_shard_count") * HORIZONS
             and (execution.get("resumed_shard_count")
                  + execution.get("adopted_unindexed_shard_count")
                  + execution.get("new_shard_count")) == len(records)
             and isinstance(execution.get("wall_seconds"), (int, float))
             and execution["wall_seconds"] >= 0
             and (execution.get("new_frames_per_second") is None
                  or isinstance(execution["new_frames_per_second"],
                                (int, float))
                  and execution["new_frames_per_second"] >= 0)
             and value.get("predictor_checkpoints_opened") == 0
             and value.get("predictor_utility_shards_opened") == 0
             and value.get("simulator_runs") == 0
             and value.get("renders_generated") == 0
             and value.get("final_200_state_corpus_generated") is False,
             "ViT-g latent index scientific fields changed")
    by_digest = {str(row["training_view_row_digest"]): row
                 for row in _ordered_rows(view)}
    seen: set[str] = set()
    for record in records:
        _require(isinstance(record, Mapping), "ViT-g record is not an object")
        digest = str(record.get("training_view_row_digest"))
        row = by_digest.get(digest)
        _require(row is not None and digest not in seen,
                 "ViT-g record identity is absent or duplicated")
        relative = Path(str(record.get("path", "")))
        _require(relative == Path("latents/horizon") / f"{digest}.f16",
                 "ViT-g shard path is not canonical")
        _require(_valid_record(encoded_root(root) / relative, record, row),
                 f"ViT-g shard changed for {digest}")
        seen.add(digest)
    if require_complete:
        _require(len(records) == EXPECTED_ROWS and seen == set(by_digest),
                 "ViT-g complete index omits a training row")
    return value


def _require_monotonic(prior: Sequence[Mapping[str, Any]],
                       current: Sequence[Mapping[str, Any]]) -> None:
    prior_by_id = {str(record["training_view_row_digest"]): dict(record)
                   for record in prior}
    current_by_id = {str(record["training_view_row_digest"]): dict(record)
                     for record in current}
    _require(len(prior_by_id) == len(prior)
             and len(current_by_id) == len(current)
             and set(prior_by_id) <= set(current_by_id)
             and all(current_by_id[key] == value
                     for key, value in prior_by_id.items()),
             "ViT-g latent index did not advance monotonically")


def _receipt_payload(index: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    path = latent_index_path(root)
    return _signed({
        "schema": ENCODING_RECEIPT_SCHEMA,
        "status": STATUS,
        "complete": True,
        "encoding_contract_digest": index["encoding_contract_digest"],
        "resource_smoke_receipt_digest": index["resource_smoke_receipt_digest"],
        "training_view_digest": index["training_view_digest"],
        "latent_index_digest": index[LATENT_INDEX_SELF_KEY],
        "latent_index_path": str(path.relative_to(root)),
        "latent_index_sha256": file_sha256(path),
        "latent_index_byte_count": path.stat().st_size,
        "latent_content_digest": index["latent_content_digest"],
        "execution": index["execution"],
        "horizon_latent_count": EXPECTED_ROWS,
        "horizon_shape": [EXPECTED_ROWS, *HORIZON_SHAPE],
        "total_latent_bytes": TOTAL_LATENT_BYTES,
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "simulator_runs": 0,
        "renders_generated": 0,
        "final_200_state_corpus_generated": False,
    }, ENCODING_RECEIPT_SELF_KEY)


def load_and_validate_encoded_training_view_for_consumption(
        *, root: Path = ROOT, verify_encoder_checkpoint: bool = False,
        ) -> dict[str, Any]:
    _managed_root(root, require_free_space=False)
    view = load_training_view(root=root)
    resource = load_resource_smoke_receipt(root=root)
    contract = load_encoding_contract(
        resource_receipt=resource, root=root)
    index = _validate_index(
        _read_json(latent_index_path(root), label="ViT-g latent index"),
        view, resource_receipt=resource, contract=contract, root=root,
        require_complete=True)
    receipt = _validate_signed(
        _read_json(encoding_receipt_path(root), label="ViT-g encoding receipt"),
        ENCODING_RECEIPT_SELF_KEY, "ViT-g encoding receipt")
    _require(receipt == _receipt_payload(index, root=root),
             "ViT-g encoding receipt differs from the exact index")
    if verify_encoder_checkpoint:
        from scripts import vjepa2_1_vitg_frozen_encoder_ablation_v1 as runtime
        observed = runtime.file_binding_v1(runtime.VJEPA_CHECKPOINT)
        _require(observed["sha256"]
                 == resource["checkpoint_binding"]["sha256"],
                 "ViT-g checkpoint bytes changed")
    return {"view": view, "index": index, "receipt": receipt,
            "encoded_root": encoded_root(root)}


def _runtime_device(resource: Mapping[str, Any]):
    import torch

    _require(torch.cuda.is_available(), "ViT-g encoding requires ROCm/CUDA")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    _require(int(properties.total_memory) >= MIN_DEVICE_TOTAL_MEMORY_BYTES
             and int(total) >= MIN_DEVICE_TOTAL_MEMORY_BYTES
             and int(free) >= MIN_DEVICE_FREE_MEMORY_BYTES,
             "ViT-g device memory gate failed")
    dtype_by_name = {
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    dtype = dtype_by_name.get(resource.get("inference_dtype"))
    _require(dtype is not None, "ViT-g smoke compute dtype changed")
    return device, dtype


def _default_runtime_loader(device: Any, dtype: Any,
                            checkpoint_sha256: str) -> Any:
    from scripts import vjepa2_1_vitg_frozen_encoder_ablation_v1 as runtime

    return runtime.load_official_frozen_encoder_v1(
        device=device, dtype=dtype,
        expected_checkpoint_sha256=checkpoint_sha256)


class _HorizonFrameDataset:
    """Ordered four-frame groups preprocessed only by bounded workers."""

    def __init__(self, groups: Sequence[Sequence[str]]) -> None:
        self.groups = [tuple(group) for group in groups]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image
        from scripts import vjepa2_1_vitg_frozen_encoder_ablation_v1 as runtime

        pixels = []
        for path in self.groups[index]:
            with Image.open(path) as image:
                pixels.append(runtime.preprocess_v03_image_v1(image))
        return torch.stack(pixels)


def _loader_worker_init(_worker_id: int) -> None:
    import torch
    torch.set_num_threads(1)


def _default_preprocessed_batches(
        groups: Sequence[Sequence[str]], *, loader_workers: int):
    import torch

    loader = torch.utils.data.DataLoader(
        _HorizonFrameDataset(groups), batch_size=1, shuffle=False,
        num_workers=loader_workers, persistent_workers=False,
        prefetch_factor=2, drop_last=False,
        worker_init_fn=_loader_worker_init)
    for batch in loader:
        yield batch.squeeze(0)


def _default_encode_pixels(arm: Any, pixels: Any,
                           device: Any, dtype: Any,
                           selected_batch: int) -> np.ndarray:
    import torch
    from scripts import vjepa2_1_vitg_frozen_encoder_ablation_v1 as runtime

    del dtype  # The wrapper owns BF16 autocast; weights and inputs stay FP32.
    _require(tuple(pixels.shape[:1]) == (HORIZONS,),
             "ViT-g horizon frame grouping changed")
    batch = pixels.to(device=device, dtype=torch.float32)
    token_chunks = []
    for start, stop in _frame_chunk_ranges(selected_batch):
        chunk = runtime.extract_final_dense_tokens_v1(arm, batch[start:stop])
        _require(tuple(chunk.shape) == (stop - start, TOKENS, TOKEN_DIM)
                 and bool(torch.isfinite(chunk).all()),
                 "ViT-g dense-token output changed")
        token_chunks.append(chunk)
    tokens = torch.cat(token_chunks, dim=0)
    _require(tuple(tokens.shape) == HORIZON_SHAPE,
             "ViT-g concatenated horizon output changed")
    return tokens.detach().to(
        device="cpu", dtype=torch.float16).numpy()


def _frame_chunk_ranges(selected_batch: int) -> tuple[tuple[int, int], ...]:
    _require(selected_batch in {1, 2, ENCODER_BATCH_FRAMES},
             "ViT-g selected batch is unsupported")
    return tuple((start, min(start + selected_batch, HORIZONS))
                 for start in range(0, HORIZONS, selected_batch))


def _default_encode_paths(arm: Any, paths: Sequence[str],
                          device: Any, dtype: Any,
                          selected_batch: int = ENCODER_BATCH_FRAMES,
                          ) -> np.ndarray:
    """Synthetic-test adapter; production uses the bounded ordered loader."""

    import torch
    from PIL import Image
    from scripts import vjepa2_1_vitg_frozen_encoder_ablation_v1 as runtime

    pixels = []
    for path in paths:
        with Image.open(path) as image:
            pixels.append(runtime.preprocess_v03_image_v1(image))
    return _default_encode_pixels(
        arm, torch.stack(pixels), device, dtype, selected_batch)


def encode_training_view(
        *, root: Path = ROOT, batch_frames: int | None = None,
        loader_workers: int = DEFAULT_LOADER_WORKERS,
        runtime_loader: Callable[[Any, Any, str], Any] | None = None,
        encode_paths: Callable[[Any, Sequence[str], Any, Any], np.ndarray]
        | None = None,
        ) -> dict[str, Any]:
    _require(MIN_LOADER_WORKERS <= loader_workers <= MAX_LOADER_WORKERS,
             "ViT-g loader workers must be between four and eight")
    started = time.monotonic()
    _managed_root(root, require_free_space=True)
    view = load_training_view(root=root)
    horizon_paths = BASE.validate_frame_inputs(view, root=root)
    resource = load_resource_smoke_receipt(root=root)
    selected_batch = selected_batch_frames(resource)
    _require(batch_frames is None or batch_frames == selected_batch,
             "requested batch does not match the smoke-selected batch")
    contract = freeze_encoding_contract(
        resource_receipt=resource, root=root)
    rows = _ordered_rows(view)
    index_path = latent_index_path(root)
    receipt_path = encoding_receipt_path(root)
    if receipt_path.exists() or receipt_path.is_symlink():
        return load_and_validate_encoded_training_view_for_consumption(root=root)

    prior_records: list[Mapping[str, Any]] = []
    if index_path.exists() or index_path.is_symlink():
        raw_index = _read_json(index_path, label="partial ViT-g latent index")
        prior_complete = raw_index.get("complete") is True
        prior = _validate_index(
            raw_index,
            view, resource_receipt=resource, contract=contract, root=root,
            require_complete=prior_complete)
        if prior_complete:
            receipt = _receipt_payload(prior, root=root)
            _publish_json_once(
                receipt_path, receipt, label="ViT-g encoding receipt")
            return {"view": view, "index": prior, "receipt": receipt,
                    "encoded_root": encoded_root(root)}
        prior_records = list(prior["horizon_records"])
    current = {
        str(record["training_view_row_digest"]): dict(record)
        for record in prior_records
    }
    output = _guarded_output(
        Path(ENCODED_DIRECTORY_NAME) / "latents/horizon", root=root)
    output.mkdir(parents=True, exist_ok=True)
    registered_names = {f"{row['training_view_row_digest']}.f16" for row in rows}
    unexpected = [path.name for path in output.glob("*.f16")
                  if path.name not in registered_names]
    _require(not unexpected, "ViT-g latent directory contains another identity")
    _require(not list(output.glob("*.partial")),
             "ViT-g latent directory contains an interrupted temporary")
    missing: list[Mapping[str, Any]] = []
    adopted_count = 0
    for row in rows:
        digest = str(row["training_view_row_digest"])
        path = output / f"{digest}.f16"
        record = current.get(digest)
        if record is not None:
            _require(_valid_record(path, record, row),
                     f"registered ViT-g shard changed for {digest}")
        else:
            if path.exists() or path.is_symlink():
                _require(_adoptable_unindexed_shard(path),
                         f"invalid unindexed ViT-g shard exists for {digest}")
                current[digest] = _record(path, row, root=root)
                adopted_count += 1
            else:
                missing.append(row)

    resumed_count = len(prior_records)
    encoded_count = 0
    device = None
    if missing:
        device, dtype = _runtime_device(resource)
        import torch
        torch.cuda.reset_peak_memory_stats(device)
        loader = runtime_loader or _default_runtime_loader
        encoder = loader(
            device, dtype, resource["checkpoint_binding"]["sha256"])
        if encode_paths is None:
            groups = [horizon_paths[str(row["training_view_row_digest"])]
                      for row in missing]
            batches = _default_preprocessed_batches(
                groups, loader_workers=loader_workers)
            encoded_rows = (
                (row, _default_encode_pixels(
                    encoder, pixels, device, dtype, selected_batch))
                for row, pixels in zip(missing, batches, strict=True)
            )
        else:
            encoded_rows = (
                (row, encode_paths(
                    encoder,
                    horizon_paths[str(row["training_view_row_digest"])],
                    device, dtype))
                for row in missing
            )
        for row, arrays in encoded_rows:
            digest = str(row["training_view_row_digest"])
            array = np.asarray(arrays, dtype=np.float16).reshape(HORIZON_SHAPE)
            path = output / f"{digest}.f16"
            atomic_missing_f16(path, array)
            current[digest] = _record(path, row, root=root)
            encoded_count += 1
            ordered_current = [current[str(candidate["training_view_row_digest"])]
                               for candidate in rows
                               if str(candidate["training_view_row_digest"])
                               in current]
            _require_monotonic(prior_records, ordered_current)
            partial = _index_payload(
                view, ordered_current, resource_receipt=resource,
                contract=contract,
                complete=False, execution=_execution_stats(
                    started=started, resumed=resumed_count,
                    adopted=adopted_count, encoded=encoded_count,
                    loader_workers=loader_workers, device=device))
            _atomic_operational_json(index_path, partial)
            prior_records = ordered_current

    ordered = [current[str(row["training_view_row_digest"])] for row in rows]
    _require_monotonic(prior_records, ordered)
    final = _index_payload(
        view, ordered, resource_receipt=resource, contract=contract,
        complete=True,
        execution=_execution_stats(
            started=started, resumed=resumed_count, adopted=adopted_count,
            encoded=encoded_count, loader_workers=loader_workers,
            device=device))
    _atomic_operational_json(index_path, final)
    final = _validate_index(
        final, view, resource_receipt=resource, contract=contract, root=root,
        require_complete=True)
    receipt = _receipt_payload(final, root=root)
    _publish_json_once(receipt_path, receipt, label="ViT-g encoding receipt")
    return {"view": view, "index": final, "receipt": receipt,
            "encoded_root": encoded_root(root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-frames", type=int, choices=(1, 2, 4),
                        default=None)
    parser.add_argument("--loader-workers", type=int,
                        choices=range(MIN_LOADER_WORKERS,
                                      MAX_LOADER_WORKERS + 1),
                        default=DEFAULT_LOADER_WORKERS)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args(argv)
    result = encode_training_view(
        batch_frames=args.batch_frames, loader_workers=args.loader_workers)
    print(json.dumps({
        "status": "COMPLETE_VJEPA2_1_VITG_ABLATION_ENCODING",
        "training_view_digest": result["index"]["training_view_digest"],
        "latent_index_digest": result["index"][LATENT_INDEX_SELF_KEY],
        "encoding_receipt_digest":
            result["receipt"][ENCODING_RECEIPT_SELF_KEY],
        "horizon_latent_count": len(result["index"]["horizon_records"]),
        "selected_batch_frames": result["index"]["selected_batch_frames"],
        "predictor_checkpoints_opened": 0,
        "simulator_runs": 0,
        "renders_generated": 0,
        "final_200_state_corpus_generated": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
