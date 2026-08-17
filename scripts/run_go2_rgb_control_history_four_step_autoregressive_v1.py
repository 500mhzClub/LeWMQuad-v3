#!/usr/bin/env python3
"""Bounded four-step RGB + control-history autoregressive experiment.

This driver is intentionally additive.  It never mutates the frozen one-step or
two-step comparator packages, never opens utility-scorer material, and never
discovers or reads sealed benchmark paths.  Its staged interface is:

    issue -> manifest -> encode -> smoke -> preflight -> train-seed/train-all
          -> evaluate -> validate

Every mutating scientific stage consumes the immutable contract emitted by the
``issue`` stage and writes only below the contract's registered runtime root.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import (  # noqa: E402
    go2_rgb_control_history_four_step_autoregressive_v1_contract as C,
)
from scripts import build_dev_canonical_cache_map_v1 as MAP  # noqa: E402
from scripts import build_dev_factorial_manifest_v1 as FM  # noqa: E402
from scripts import dev_checkpoint_v1 as CK  # noqa: E402
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402
from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import build_dev_v03_horizon_sequences_v1 as HSEQ  # noqa: E402
from scripts import run_dev_proprio_factorial_driver_v1 as F  # noqa: E402
from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import run_dev_v03_two_step_rollout_v1 as R  # noqa: E402

STATUS = "EXPLORATORY_FOUR_STEP_AUTOREGRESSIVE_ROLLOUT_OBJECTIVE"
EPOCHS = 24
CHECKPOINT_EPOCH = 21
BATCH = 4
LR = 3.0e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WIDTH, DEPTH, HEADS = 384, 6, 6
HORIZONS = (1, 2, 3, 4)
EXPECTED_COMMON = {"train": 3_854, "checkpoint_selection": 466}
EXPECTED_COMMON_ROWS_DIGEST = C.TARGET_AVAILABILITY["common_manifest_preimage_digest"]
EXPECTED_STABLE_ID_DIGEST = C.TARGET_AVAILABILITY["stable_id_list_digest"]
EXPECTED_POSITION_DIGEST = C.TARGET_AVAILABILITY["factorial_position_list_digest"]
COMMON_MANIFEST = "common_h4_manifest.json"
ENCODE_RECEIPT = "target_cache_index.json"
TARGET_BLOBS = {3: "train_target_h3.f16", 4: "train_target_h4.f16"}
HORIZON_FAMILY_COUNTS = C.TARGET_AVAILABILITY["horizon_family_counts"]


def _die(message: str) -> None:
    raise RuntimeError(message)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        _die(f"stale atomic-write temporary exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_read_only(path: Path) -> None:
    path.chmod(0o444)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_once(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        _die(f"create-only output already exists: {path}")
    _write_json(path, value)
    _make_read_only(path)


def _write_jsonl_once(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        _die(f"create-only output already exists: {path}")
    _write_jsonl(path, rows)
    _make_read_only(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        _die(f"stale atomic-write temporary exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                _die(f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def _require_immutable_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        _die(f"{label} is absent, non-regular, or a symlink: {path}")
    if (path.stat().st_mode & 0o777) != 0o444:
        _die(f"{label} mode is not exactly 0444: {path}")


def _sha256_file(path: Path) -> str:
    return C.file_sha256(path)


def _digest(value: Any) -> str:
    return C.digest(value)


def runtime_root() -> Path:
    return Path(C.runtime_root())


def _canonical_compact(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _compact_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_compact(value)).hexdigest()


def _sequence_digest(value: Sequence[Any]) -> str:
    return hashlib.sha256(_canonical_compact(list(value))).hexdigest()


def _mem_available_bytes() -> int:
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    _die("/proc/meminfo carries no MemAvailable")


def _mem_total_bytes() -> int:
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    _die("/proc/meminfo carries no MemTotal")


def _model_state_digest(model: nn.Module) -> str:
    return F.state_digest(model.state_dict())


def _contract_digest(contract: dict[str, Any]) -> str:
    for name in ("contract_digest", "digest", "sha256"):
        value = contract.get(name)
        if isinstance(value, str) and len(value) == 64:
            return value
    return _digest(contract)


def environment_record(require_exact: bool = False) -> dict[str, Any]:
    expected = C.ENVIRONMENT_REFERENCE
    record = {
        "interpreter": str(Path(sys.executable)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_module": str(Path(torch.__file__).resolve()),
        "hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    exact = {
        "interpreter": "/home/andrewknowles/TinyQuadJEPA/bin/python",
        "python": str(expected["historical_python"]),
        "torch": str(expected["historical_torch"]),
        "torch_module": (
            "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/"
            "site-packages/torch/__init__.py"
        ),
        "hip": str(expected["historical_hip"]),
        "cuda_available": True,
        "device": str(expected["historical_device"]),
    }
    record["required"] = exact
    record["exact_match"] = all(record[key] == value for key, value in exact.items())
    if require_exact and not record["exact_match"]:
        _die(f"scientific runtime environment differs: {record}")
    return record


def require_contract() -> dict[str, Any]:
    path = Path(C.contract_path())
    if not path.is_file():
        _die(f"four-step contract is absent: {path}")
    return C.validate_installed_source(_read_json(path), ROOT)


def _common_manifest_path() -> Path:
    return runtime_root() / COMMON_MANIFEST


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            _die("registered R9700 is unavailable")
        index = 0 if device.index is None else int(device.index)
        observed = torch.cuda.get_device_name(index)
        expected = F.DEVICE_POLICY["expected_device_name"]
        if index != F.DEVICE_POLICY["device_index"] or observed != expected:
            _die(f"device policy differs: cuda:{index} is {observed!r}, expected {expected!r}")
        return torch.device(f"cuda:{index}")
    if device.type != "cpu":
        _die(f"unsupported device {device}")
    return device


def validate_common_manifest() -> dict[str, Any]:
    path = _common_manifest_path()
    _require_immutable_file(path, "common H4 manifest")
    value = _read_json(path)
    expected_manifest_keys = {
        "schema", "status", "complete", "four_step_contract_digest",
        "factorial_manifest_digest", "canonical_cache_map_digest",
        "proprio_manifest_rows_sha256", "normalisation_sha256",
        "historical_comparator_lineage",
        "verified_rgb_comparator_checkpoint_sha256", "order", "counts",
        "family_counts", "historical_control_train_rows",
        "historical_control_train_row_difference",
        "historical_controls_sample_matched",
        "historical_controls_retrained_or_reselected", "data_order_contract",
        "per_seed_epoch_order_digests", "common_rows_digest",
        "stable_id_list_digest", "factorial_position_list_digest",
        "partition_pair_order_digest", "exclusions", "exclusion_counts",
        "excluded_stable_id_list_digest", "excluded_pair_newline_digest",
        "rows", "manifest_digest",
    }
    if set(value) != expected_manifest_keys:
        _die("common manifest top-level schema differs")
    if (value.get("schema")
            != "go2_rgb_control_history_four_step_common_h4_manifest_v1"
            or value.get("status") != STATUS or value.get("complete") is not True):
        _die("common manifest completion/schema differs")
    if value["manifest_digest"] != _digest({
        key: item for key, item in value.items() if key != "manifest_digest"
    }):
        _die("common manifest self digest differs")
    contract = require_contract()
    if value["four_step_contract_digest"] != _contract_digest(contract):
        _die("common manifest contract binding differs")
    fixed_bindings = {
        "factorial_manifest_digest": C.FROZEN_FACTORIAL["factorial_manifest_digest"],
        "canonical_cache_map_digest": C.FROZEN_FACTORIAL[
            "canonical_cache_map_digest"
        ],
        "normalisation_sha256": C.FROZEN_FACTORIAL["normalisation_sha256"],
        "historical_control_train_rows": 3922,
        "historical_control_train_row_difference": 68,
        "historical_controls_sample_matched": False,
        "historical_controls_retrained_or_reselected": False,
        "data_order_contract": C.DATA_ORDER_CONTRACT,
    }
    for key, expected in fixed_bindings.items():
        if value.get(key) != expected:
            _die(f"common manifest {key} binding differs")
    rows = value.get("rows")
    if not isinstance(rows, list):
        _die("common manifest has no row list")
    observed = _compact_digest(rows)
    if observed != value.get("common_rows_digest"):
        _die("common manifest row digest is internally inconsistent")
    if observed != EXPECTED_COMMON_ROWS_DIGEST:
        _die(f"common manifest digest {observed} differs from frozen audit")
    counts = collections.Counter(str(row["split"]) for row in rows)
    if dict(counts) != EXPECTED_COMMON:
        _die(f"common H4 counts {dict(counts)} != {EXPECTED_COMMON}")
    if [int(row["position"]) for row in rows] != sorted(
        int(row["position"]) for row in rows
    ):
        _die("common manifest does not preserve factorial position order")
    if len({row["stable_row_id"] for row in rows}) != len(rows):
        _die("common manifest stable row identities are not unique")
    expected_keys = {
        "position", "stable_row_id", "pair_sha256", "split", "family", "scene",
        "env_index", "episode_id", "reset_count", "source_frame_index",
        "max_horizon", "action_blocks_available", "first_exclusion_reason",
        "horizon_frame_indices",
    }
    if any(set(row) != expected_keys for row in rows):
        _die("common manifest row key schema differs from frozen audit")
    if any(len(row["horizon_frame_indices"]) != 4 for row in rows):
        _die("common manifest must carry H1-H4 frame indices only")
    stable_digest = _sequence_digest([row["stable_row_id"] for row in rows])
    position_digest = _sequence_digest([row["position"] for row in rows])
    pair_digest = hashlib.sha256("".join(
        f"{row['pair_sha256']}\n" for row in rows
    ).encode("ascii")).hexdigest()
    if stable_digest != value["stable_id_list_digest"] \
            or stable_digest != EXPECTED_STABLE_ID_DIGEST:
        _die("common manifest stable-ID digest differs")
    if position_digest != value["factorial_position_list_digest"] \
            or position_digest != EXPECTED_POSITION_DIGEST:
        _die("common manifest factorial-position digest differs")
    if pair_digest != value["partition_pair_order_digest"] \
            or pair_digest != C.TARGET_AVAILABILITY["partition_pair_order_digest"]:
        _die("common manifest pair ordering digest differs")
    train_family = dict(collections.Counter(
        row["family"] for row in rows if row["split"] == "train"
    ))
    selection_family = dict(collections.Counter(
        row["family"] for row in rows
        if row["split"] == "checkpoint_selection"
    ))
    if value["family_counts"] != {
        "train": C.TARGET_AVAILABILITY["common_train_family_counts"],
        "checkpoint_selection":
            C.TARGET_AVAILABILITY["common_selection_family_counts"],
    } or train_family != C.TARGET_AVAILABILITY["common_train_family_counts"] \
            or selection_family != C.TARGET_AVAILABILITY[
                "common_selection_family_counts"
            ]:
        _die("common manifest family ledger differs")
    exclusions = value["exclusions"]
    if not isinstance(exclusions, list) or len(exclusions) != 77:
        _die("common manifest exclusion ledger length differs")
    exclusion_counts = dict(collections.Counter(
        row.get("first_exclusion_reason") for row in exclusions
    ))
    if exclusion_counts != {
        "missing_frame_metadata": 56, "reset_or_episode_boundary": 21,
    } or value["exclusion_counts"] != exclusion_counts:
        _die("common manifest exclusion counts differ")
    excluded_stable = _sequence_digest([
        str(row["stable_row_id"]) for row in exclusions
    ])
    excluded_pairs = hashlib.sha256("".join(
        f"{row['pair_sha256']}\n" for row in exclusions
    ).encode("ascii")).hexdigest()
    if excluded_stable != value["excluded_stable_id_list_digest"] \
            or excluded_stable != C.TARGET_AVAILABILITY[
                "runner_excluded_stable_compact_digest"
            ] or excluded_pairs != value["excluded_pair_newline_digest"] \
            or excluded_pairs != C.TARGET_AVAILABILITY[
                "runner_excluded_pair_newline_sha256"
            ]:
        _die("common manifest excluded identity ordering differs")
    expected_order_digests: dict[str, dict[str, dict[str, str]]] = {}
    for seed in C.FROZEN_SEEDS:
        expected_order_digests[str(seed)] = {}
        for epoch in range(EPOCHS):
            plan = common_plan_from_rows(int(seed), epoch, rows)
            sequence = [index for batch in plan for index in batch]
            expected_order_digests[str(seed)][str(epoch)] = {
                "sequence_digest": _sequence_digest(sequence),
                "batch_digest": _compact_digest(plan),
            }
    if value["per_seed_epoch_order_digests"] != expected_order_digests:
        _die("common manifest per-seed/epoch data-order digests differ")
    expected_checkpoints = {
        str(seed): {
            "rgb_one_step": C.COMPARATOR_CHECKPOINT_SHA256[int(seed)][
                "rgb_one_step"
            ],
            "rgb_two_step_rollout": C.COMPARATOR_CHECKPOINT_SHA256[int(seed)][
                "rgb_two_step_rollout"
            ],
        }
        for seed in C.FROZEN_SEEDS
    }
    if value["verified_rgb_comparator_checkpoint_sha256"] != expected_checkpoints:
        _die("common manifest frozen comparator bindings differ")
    lineage = value["historical_comparator_lineage"]
    expected_lineage = {
        "confirmatory_commit": C.FROZEN_FACTORIAL["confirmatory_commit"],
        "confirmatory_commit_ancestor_of_head": True,
        "confirmatory_report_digest": C.FROZEN_FACTORIAL["final_report_digest"],
        "run_package_digest": C.FROZEN_FACTORIAL["run_package_digest"],
        "run_package_sha256": C.FROZEN_FACTORIAL["run_package_file_sha256"],
        "run_package_independent_verifier":
            "scripts.freeze_dev_proprio_run_package_v1.verify",
        "initial_launch_receipt_digest": C.FROZEN_FACTORIAL[
            "initial_launch_receipt_digest"
        ],
        "continuation_launch_receipt_digest": C.FROZEN_FACTORIAL[
            "continuation_receipt_digest"
        ],
        "frozen_seed_prefix": [int(seed) for seed in C.FROZEN_SEEDS],
        "checkpoint_count": 32,
        "normalisation_sha256": C.FROZEN_FACTORIAL["normalisation_sha256"],
    }
    if not isinstance(lineage, dict) or any(
        lineage.get(key) != expected for key, expected in expected_lineage.items()
    ) or not isinstance(lineage.get("checkpoint_hash_verification_wall_time_s"),
                        (int, float)) \
            or not math.isfinite(float(lineage["checkpoint_hash_verification_wall_time_s"])) \
            or float(lineage["checkpoint_hash_verification_wall_time_s"]) < 0.0 \
            or not isinstance(lineage.get(
                "predictor_source_bindings_at_confirmatory_commit"
            ), dict) or not lineage["predictor_source_bindings_at_confirmatory_commit"]:
        _die("common manifest historical comparator lineage differs")
    for relative, binding in lineage[
        "predictor_source_bindings_at_confirmatory_commit"
    ].items():
        source_path = ROOT / relative
        if not source_path.is_file() or source_path.is_symlink() \
                or binding.get("sha256") != _sha256_file(source_path) \
                or not isinstance(binding.get("git_blob"), str) \
                or len(binding["git_blob"]) != 40:
            _die(f"historical comparator source binding differs: {relative}")
    proprio_binding = C.FROZEN_TRAINING_INPUT_FILES["proprio_control_manifest"]
    proprio_path = Path(proprio_binding["path"])
    if not proprio_path.is_file() or proprio_path.stat().st_size != int(
        proprio_binding["bytes"]
    ) or _sha256_file(proprio_path) != proprio_binding["sha256"] \
            or value["proprio_manifest_rows_sha256"] != _read_json(proprio_path).get(
                "rows_sha256"
            ):
        _die("common manifest proprio-row lineage differs")
    rows_path = runtime_root() / "common_h4_rows.jsonl"
    audit_path = runtime_root() / "target_availability.json"
    _require_immutable_file(rows_path, "common H4 row ledger")
    _require_immutable_file(audit_path, "target availability audit")
    if _read_jsonl(rows_path) != rows:
        _die("common H4 row ledger differs from manifest")
    audit = _read_json(audit_path)
    if audit.get("audit_digest") != _digest({
        key: item for key, item in audit.items() if key != "audit_digest"
    }) or audit.get("four_step_contract_digest") != _contract_digest(contract) \
            or audit.get("recomputed_common_manifest_digest") != observed \
            or audit.get("frozen_audit") != C.TARGET_AVAILABILITY \
            or audit.get("available_rows_H1_H4") != C.TARGET_AVAILABILITY[
                "horizon_counts"
            ] or audit.get("family_counts_H1_H4") != HORIZON_FAMILY_COUNTS \
            or audit.get("reset_and_boundary_exclusions") != C.TARGET_AVAILABILITY[
                "incremental_exclusions"
            ] \
            or audit.get("additional_encoding") != "train H3 and H4 only" \
            or audit.get("new_simulator_data_generated") is not False:
        _die("target-availability audit differs")
    return value


def existing_target_path_map() -> tuple[
    dict[str, tuple[str, int]], dict[str, tuple[Path, int]]
]:
    """Map frozen raster paths to already-cached raw f16 ViT-L tokens.

    The mapping covers all original train/selection context, H1 and H2 caches,
    plus the existing selection-only H3/H4 caches.  It never executes an
    encoder and lets the H3/H4 builder encode only genuinely uncached rasters.
    """
    temporal = _read_jsonl(Path(F.CACHE) / "temporal_rows.jsonl")
    train = [row for row in temporal if row["role"] == "train"]
    selection = [row for row in temporal if row["role"] == "checkpoint_selection"]
    sources: dict[str, tuple[Path, int]] = {
        "train_ctx0": (F.DIAG_CACHE / "frozen_train_ctx0.f16", len(train)),
        "train_ctx1": (F.DIAG_CACHE / "frozen_train_ctx1.f16", len(train)),
        "selection_ctx0": (F.EVAL_CACHE / "frozen_ctx0.f16", len(selection)),
        "selection_ctx1": (F.EVAL_CACHE / "frozen_ctx1.f16", len(selection)),
        "current": (F.EVAL_CACHE / "frozen_current.f16", len(train) + len(selection)),
        "train_h1": (F.EVAL_CACHE / "frozen_train_future.f16", len(train)),
        "selection_h1": (F.EVAL_CACHE / "frozen_sel_future.f16", len(selection)),
    }
    mapping: dict[str, tuple[str, int]] = {}

    def register(path: str | Path, source: str, row: int) -> None:
        key = str(Path(path).resolve())
        mapping.setdefault(key, (source, int(row)))

    for split, rows in (("train", train), ("selection", selection)):
        current_offset = 0 if split == "train" else len(train)
        for index, row in enumerate(rows):
            contexts = row["context_paths"]
            register(contexts[0], f"{split}_ctx0", index)
            register(contexts[1], f"{split}_ctx1", index)
            register(contexts[2], "current", current_offset + index)
            register(row["target_path"], f"{split}_h1", index)

    two_rows = _read_jsonl(Path(HSEQ.TWO) / "two_step_rows.jsonl")
    for split, role, blob_name in (
        ("train", "train", "frozen_train_step2.f16"),
        ("selection", "checkpoint_selection", "frozen_sel_step2.f16"),
    ):
        rows = [row for row in two_rows if row["role"] == role]
        source = f"{split}_h2"
        path = Path(HSEQ.TWO) / blob_name
        sources[source] = (path, path.stat().st_size // (P.TOKENS * P.TOKEN_DIM * 2))
        for index, row in enumerate(rows):
            register(row["step2_path"], source, index)

    horizon_rows = _read_jsonl(
        Path(HSEQ.OUT) / "FINAL" / "FINAL_horizon_rows_479.jsonl"
    )
    horizon_rows = [row for row in horizon_rows if int(row["max_horizon"]) >= 4]
    for horizon in (3, 4):
        source = f"selection_h{horizon}"
        path = Path(HSEQ.OUT) / f"target_h{horizon}.f16"
        sources[source] = (path, len(horizon_rows))
        for index, row in enumerate(horizon_rows):
            frames = {int(frame["h"]): frame["path"] for frame in row["horizon_frames"]}
            register(frames[horizon], source, index)

    expected_per_row = P.TOKENS * P.TOKEN_DIM * 2
    for source, (path, rows) in sources.items():
        if not path.is_file() or path.stat().st_size != rows * expected_per_row:
            _die(f"existing target cache {source} is absent or wrong-sized: {path}")
    return mapping, sources


def _frame_metadata_for_rows(
    two_step_rows: Sequence[dict[str, Any]], workers: int
) -> dict[str, dict[int, dict[str, Any]]]:
    """Read only the five registered positional records needed per candidate row."""
    paired = _read_json(Path(HSEQ.PAIRED))
    sources = {
        str(source["scene_id"]): str(source["paths"]["frames_jsonl"])
        for source in paired["sources"]
    }
    wanted: dict[str, set[int]] = collections.defaultdict(set)
    for row in two_step_rows:
        wanted[str(row["scene"])].update(
            int(row["t"]) + 240 * horizon for horizon in range(5)
        )
    tasks = []
    for scene in sorted(wanted):
        if scene not in sources:
            _die(f"paired manifest has no frames source for {scene}")
        tasks.append((scene, sources[scene], sorted(wanted[scene])))
    # The existing helper enforces positional frame_index == JSONL line number.
    # A local pool only accelerates independent read-only scene scans.
    if workers > 1:
        from multiprocessing import Pool

        with Pool(min(workers, len(tasks))) as pool:
            return dict(pool.map(HSEQ._blocks, tasks))
    return dict(HSEQ._blocks(task) for task in tasks)


def build_common_manifest_rows(
    factorial: dict[str, Any], map_record: dict[str, Any],
    proprio_rows: Sequence[dict[str, Any]],
    two_step_rows: Sequence[dict[str, Any]],
    frame_metadata: dict[str, dict[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pure H4 intersection reducer, in the frozen factorial order.

    The returned included rows have exactly the keys bound by the independent
    availability audit.  Exclusions are diagnostic and never enter the digest.
    """
    map_by_manifest = {
        int(entry["manifest_row_index"]): entry for entry in map_record["entries"]
    }
    factorial_by_position = {
        int(entry["position"]): entry for entry in factorial["rows"]
    }
    two_by_pair: dict[str, dict[str, Any]] = {}
    for row in two_step_rows:
        pair = str(row["pair_sha256"])
        if pair in two_by_pair:
            _die(f"duplicate two-step pair identity {pair}")
        two_by_pair[pair] = row

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    expected_positions = sorted(factorial_by_position)
    for position in expected_positions:
        entry = factorial_by_position[position]
        reason: str | None = None
        pair = str(entry["pair_sha256"])
        two = two_by_pair.get(pair)
        manifest_index = int(entry["manifest_row_index"])
        if two is None:
            reason = "missing_two_step_identity"
        elif manifest_index not in map_by_manifest or not (
            0 <= manifest_index < len(proprio_rows)
        ):
            reason = "missing_manifest_identity"
        if reason is not None:
            excluded.append({"position": position, "pair_sha256": pair,
                             "first_exclusion_reason": reason})
            continue

        proprio = proprio_rows[manifest_index]
        scene = str(two["scene"])
        source = int(two["t"])
        identity = (
            int(two["env_index"]), int(two["episode_id"]), int(two["reset_count"])
        )
        blocks = frame_metadata.get(scene, {})
        frames = [source + 240 * horizon for horizon in range(5)]
        metadata = [blocks.get(index) for index in frames]
        if any(item is None for item in metadata):
            reason = "missing_frame_metadata"
        elif any(
            (int(item["env"]), int(item["episode_id"]), int(item["reset_count"]))
            != identity for item in metadata if item is not None
        ):
            reason = "reset_or_episode_boundary"
        else:
            env_text = str(two.get("env", f"{identity[0]:02d}"))
            for horizon, frame_index in enumerate(frames[1:], 1):
                path = Path(HSEQ.V03) / scene / "rgb" / (
                    f"frame_{frame_index:06d}_env_{env_text}.png"
                )
                if not path.is_file():
                    reason = f"missing_rendered_h{horizon}"
                    break
        if reason is None:
            base_size = int(metadata[0]["block_size"])
            if base_size <= 0 or any(
                int(item["block_size"]) != base_size for item in metadata
            ):
                reason = "command_block_size_mismatch"
            elif any(
                int(metadata[index]["sequence_id"])
                == int(metadata[index + 1]["sequence_id"])
                for index in range(4)
            ):
                reason = "successive_command_sequence_not_distinct"
            elif str(metadata[0]["primitive"]) != str(two["action_step1"]):
                reason = "h1_action_disagrees_with_two_step"
            elif str(metadata[1]["primitive"]) != str(two["action_step2"]):
                reason = "h2_action_disagrees_with_two_step"
            elif len(proprio.get("action_blocks", [])) < 4:
                reason = "fewer_than_four_verified_post_slew_action_blocks"
            elif any(
                len(block) != P.ACTION_DIM
                for block in proprio.get("action_blocks", [])[:4]
            ):
                reason = "verified_action_block_dimension_differs"

        if reason is not None:
            excluded.append({
                "position": position, "stable_row_id": entry["stable_row_id"],
                "pair_sha256": pair, "split": entry["split"],
                "family": entry["family"], "first_exclusion_reason": reason,
            })
            continue
        included.append({
            "position": position,
            "stable_row_id": str(entry["stable_row_id"]),
            "pair_sha256": pair,
            "split": str(entry["split"]),
            "family": str(entry["family"]),
            "scene": scene,
            "env_index": identity[0],
            "episode_id": identity[1],
            "reset_count": identity[2],
            "source_frame_index": source,
            "max_horizon": 4,
            "action_blocks_available": len(proprio["action_blocks"]),
            "first_exclusion_reason": None,
            "horizon_frame_indices": frames[1:],
        })
    return included, excluded


def common_plan_from_rows(
    seed: int, epoch: int, common_rows: Sequence[dict[str, Any]]
) -> list[list[int]]:
    train_rows = [row for row in common_rows if row["split"] == "train"]
    historical_count = int(C.DATA_ORDER_CONTRACT["historical_train_rows"])
    historical = [position for batch in F.batch_plan(
        seed, epoch, historical_count, BATCH
    ) for position in batch]
    old_to_new = {
        int(row["position"]): new_position
        for new_position, row in enumerate(train_rows)
    }
    filtered = [old_to_new[position] for position in historical
                if position in old_to_new]
    if len(filtered) != len(train_rows) or len(set(filtered)) != len(train_rows):
        _die("common data-order plan is incomplete or duplicated")
    return [filtered[offset:offset + BATCH]
            for offset in range(0, len(filtered), BATCH)]


def issue_stage(args: argparse.Namespace) -> dict[str, Any]:
    environment_record(require_exact=True)
    source = C.source_closure(ROOT)
    storage = C.storage_binding(ROOT)
    contract = C.build_contract(source, storage)
    path = Path(C.contract_path())
    # storage_binding has already established that the whole one-shot namespace
    # did not exist.  Creating this parent is therefore itself the issue event.
    _write_json_once(path, contract)
    return C.validate_installed_source(_read_json(path), ROOT)


def manifest_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze the ordered intersection of factorial rows valid through H=4."""
    contract = require_contract()
    if _common_manifest_path().exists():
        return validate_common_manifest()

    # The frozen lineage verifier hashes every registered epoch-21 checkpoint
    # before returning; it never torch-loads the checkpoint payloads.
    from scripts import analyze_go2_counterfactual_predictor_qualification_v1_2 as Q

    checkpoints, lineage = Q.verify_frozen_predictor_lineage()
    observed_comparators: dict[str, dict[str, str]] = {}
    for seed in C.FROZEN_SEEDS:
        by_cell = {item.cell: item for item in checkpoints if item.seed == seed}
        expected = C.COMPARATOR_CHECKPOINT_SHA256[seed]
        observed_comparators[str(seed)] = {
            "rgb_one_step": by_cell["rgb_one_step"].sha256,
            "rgb_two_step_rollout": by_cell["rgb_rollout"].sha256,
        }
        if observed_comparators[str(seed)] != expected:
            _die(f"frozen RGB comparator digest mismatch for seed {seed}")

    factorial = FM.load()
    map_record = MAP.load()
    if factorial["digest"] != C.FROZEN_FACTORIAL["factorial_manifest_digest"]:
        _die("frozen factorial manifest digest changed")
    if map_record["digest"] != C.FROZEN_FACTORIAL["canonical_cache_map_digest"]:
        _die("frozen canonical map digest changed")
    proprio_rows, proprio_manifest, stats = F.load_rows()
    two_step_rows = _read_jsonl(Path(HSEQ.TWO) / "two_step_rows.jsonl")
    frame_metadata = _frame_metadata_for_rows(two_step_rows, args.workers)
    rows, exclusions = build_common_manifest_rows(
        factorial, map_record, proprio_rows, two_step_rows, frame_metadata
    )
    counts = collections.Counter(str(row["split"]) for row in rows)
    if dict(counts) != EXPECTED_COMMON:
        _die(f"recomputed common H4 counts {dict(counts)} != {EXPECTED_COMMON}")
    row_digest = _compact_digest(rows)
    stable_digest = _sequence_digest([row["stable_row_id"] for row in rows])
    position_digest = _sequence_digest([row["position"] for row in rows])
    if row_digest != EXPECTED_COMMON_ROWS_DIGEST:
        _die(f"recomputed common-row digest {row_digest} differs from audit")
    if stable_digest != EXPECTED_STABLE_ID_DIGEST:
        _die("recomputed stable-ID ordering differs from audit")
    if position_digest != EXPECTED_POSITION_DIGEST:
        _die("recomputed factorial-position ordering differs from audit")
    train_family = dict(collections.Counter(
        row["family"] for row in rows if row["split"] == "train"
    ))
    selection_family = dict(collections.Counter(
        row["family"] for row in rows
        if row["split"] == "checkpoint_selection"
    ))
    if train_family != C.TARGET_AVAILABILITY["common_train_family_counts"]:
        _die("recomputed train family counts differ from availability audit")
    if selection_family != C.TARGET_AVAILABILITY["common_selection_family_counts"]:
        _die("recomputed selection family counts differ from availability audit")

    pair_newline = hashlib.sha256("".join(
        f"{row['pair_sha256']}\n" for row in rows
    ).encode("ascii")).hexdigest()
    if pair_newline != C.TARGET_AVAILABILITY["partition_pair_order_digest"]:
        _die("recomputed pair ordering differs from availability audit")
    exclusion_counts = dict(collections.Counter(
        row["first_exclusion_reason"] for row in exclusions
    ))
    if len(exclusions) != 77:
        _die(f"common-manifest exclusion count {len(exclusions)} != 77")
    if exclusion_counts != {
        "missing_frame_metadata": 56, "reset_or_episode_boundary": 21,
    }:
        _die(f"common-manifest exclusion ledger differs: {exclusion_counts}")
    excluded_stable = [str(row["stable_row_id"]) for row in exclusions]
    excluded_pairs = [str(row["pair_sha256"]) for row in exclusions]
    excluded_stable_digest = _sequence_digest(excluded_stable)
    excluded_pair_newline_digest = hashlib.sha256(
        "".join(f"{pair}\n" for pair in excluded_pairs).encode("ascii")
    ).hexdigest()
    if excluded_stable_digest != (
        "7d2dafc31a8563293165d0d867d8c08fcf4488f8c5d0445cb121bf7ffb48a949"
    ) or excluded_pair_newline_digest != (
        "f5b7fdd2da598ffb5a123a685885ab6dd593e8b2fdd82e7212beb59973a86d1f"
    ):
        _die("excluded factorial identity/order differs from availability audit")
    order_digests: dict[str, dict[str, dict[str, str]]] = {}
    for seed in C.FROZEN_SEEDS:
        epochs: dict[str, dict[str, str]] = {}
        for epoch in range(EPOCHS):
            plan = common_plan_from_rows(int(seed), epoch, rows)
            sequence = [index for batch in plan for index in batch]
            epochs[str(epoch)] = {
                "sequence_digest": _sequence_digest(sequence),
                "batch_digest": _compact_digest(plan),
            }
        order_digests[str(seed)] = epochs
    manifest = {
        "schema": "go2_rgb_control_history_four_step_common_h4_manifest_v1",
        "status": STATUS,
        "complete": True,
        "four_step_contract_digest": _contract_digest(contract),
        "factorial_manifest_digest": factorial["digest"],
        "canonical_cache_map_digest": map_record["digest"],
        "proprio_manifest_rows_sha256": proprio_manifest["rows_sha256"],
        "normalisation_sha256": proprio_manifest["normalisation_sha256"],
        "historical_comparator_lineage": lineage,
        "verified_rgb_comparator_checkpoint_sha256": observed_comparators,
        "order": "frozen factorial position order filtered only by H4 validity",
        "counts": dict(counts),
        "family_counts": {
            "train": train_family, "checkpoint_selection": selection_family,
        },
        "historical_control_train_rows": 3_922,
        "historical_control_train_row_difference": 68,
        "historical_controls_sample_matched": False,
        "historical_controls_retrained_or_reselected": False,
        "data_order_contract": C.DATA_ORDER_CONTRACT,
        "per_seed_epoch_order_digests": order_digests,
        "common_rows_digest": row_digest,
        "stable_id_list_digest": stable_digest,
        "factorial_position_list_digest": position_digest,
        "partition_pair_order_digest": pair_newline,
        "exclusions": exclusions,
        "exclusion_counts": exclusion_counts,
        "excluded_stable_id_list_digest": excluded_stable_digest,
        "excluded_pair_newline_digest": excluded_pair_newline_digest,
        "rows": rows,
    }
    manifest["manifest_digest"] = _digest(manifest)
    audit = {
        "schema": "go2_rgb_control_history_four_step_target_availability_v1",
        "status": STATUS,
        "complete": True,
        "four_step_contract_digest": _contract_digest(contract),
        "frozen_audit": C.TARGET_AVAILABILITY,
        "available_rows_H1_H4": C.TARGET_AVAILABILITY["horizon_counts"],
        "family_counts_H1_H4": HORIZON_FAMILY_COUNTS,
        "reset_and_boundary_exclusions": C.TARGET_AVAILABILITY[
            "incremental_exclusions"
        ],
        "recomputed_common_manifest_digest": row_digest,
        "common_manifest_digest_matches": True,
        "new_simulator_data_generated": False,
        "raw_future_frames_present": True,
        "existing_H1_H2_target_caches_reused": True,
        "existing_selection_H3_H4_target_caches_reused": True,
        "additional_encoding": "train H3 and H4 only",
    }
    audit["audit_digest"] = _digest(audit)
    _write_jsonl_once(runtime_root() / "common_h4_rows.jsonl", rows)
    _write_json_once(runtime_root() / "target_availability.json", audit)
    _write_json_once(_common_manifest_path(), manifest)
    return validate_common_manifest()


def encode_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Encode only missing H3/H4 train targets with the frozen ViT-L target."""
    contract = require_contract()
    manifest = validate_common_manifest()
    receipt_path = runtime_root() / ENCODE_RECEIPT
    if receipt_path.is_file():
        return validate_target_cache_index()

    device = resolve_device(args.device)
    input_verification = validate_training_input_files(hash_files=True)
    encoder_checkpoint = Path(C.TARGET_CACHE_CONTRACT["target_encoder_checkpoint"])
    if _sha256_file(encoder_checkpoint) != C.TARGET_CACHE_CONTRACT[
        "target_encoder_checkpoint_sha256"
    ]:
        _die("frozen ViT-L target encoder checkpoint digest changed")
    train_rows = [row for row in manifest["rows"] if row["split"] == "train"]
    if len(train_rows) != EXPECTED_COMMON["train"]:
        _die("target encoding row count differs")

    # Each unique raster is executed once and scattered into all H3/H4 row slots.
    destinations: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for row_index, row in enumerate(train_rows):
        env_text = f"{int(row['env_index']):02d}"
        for horizon in (3, 4):
            frame_index = int(row["horizon_frame_indices"][horizon - 1])
            path = Path(HSEQ.V03) / row["scene"] / "rgb" / (
                f"frame_{frame_index:06d}_env_{env_text}.png"
            )
            if not path.is_file():
                _die(f"missing frozen H{horizon} training target raster: {path}")
            destinations[str(path)].append((horizon, row_index))
    all_paths = sorted(destinations)
    existing_map, existing_sources = existing_target_path_map()
    cached_paths = [path for path in all_paths if path in existing_map]
    paths = [path for path in all_paths if path not in existing_map]
    if len(paths) != C.TARGET_CACHE_CONTRACT[
        "unique_train_frames_requiring_encoder_execution"
    ]:
        _die(f"uncached unique target image count {len(paths)} differs from audit")
    if sum(map(len, destinations.values())) != 2 * len(train_rows):
        _die("H3/H4 output destination count differs")
    cache_misses = sum(len(destinations[path]) for path in paths)
    if cache_misses != C.TARGET_CACHE_CONTRACT["row_horizon_cache_misses"]:
        _die(f"row/horizon cache miss count {cache_misses} differs from audit")

    partial_paths = {
        horizon: runtime_root() / (TARGET_BLOBS[horizon] + ".partial")
        for horizon in (3, 4)
    }
    final_paths = {
        horizon: runtime_root() / TARGET_BLOBS[horizon] for horizon in (3, 4)
    }
    if any(path.exists() for path in final_paths.values()):
        _die("target cache exists without its complete receipt")
    progress_path = runtime_root() / "target_encoding_progress.json"
    start = 0
    if progress_path.is_file():
        progress = _read_json(progress_path)
        if progress.get("common_rows_digest") != manifest["common_rows_digest"] \
                or progress.get("unique_path_digest") != _sequence_digest(paths):
            _die("target encoding progress belongs to a different manifest")
        start = int(progress["unique_paths_completed"])
        if start < 0 or start > len(paths):
            _die("target encoding progress count is outside its frozen path list")
    shape = (len(train_rows), P.TOKENS, P.TOKEN_DIM)
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    modes = "r+" if start else "w+"
    memories: dict[int, np.memmap] = {}
    for horizon in (3, 4):
        path = partial_paths[horizon]
        if start and (not path.is_file() or path.stat().st_size != expected_bytes):
            _die("resumable target-cache partial has wrong size")
        memories[horizon] = np.memmap(
            path, dtype=np.float16, mode=modes, shape=shape
        )

    source_arrays: dict[str, np.memmap] = {}
    locator_counts: collections.Counter[str] = collections.Counter()
    for source, (path, rows) in existing_sources.items():
        source_arrays[source] = np.memmap(
            path, dtype=np.float16, mode="r", shape=(rows, P.TOKENS, P.TOKEN_DIM)
        )
    for path in cached_paths:
        source, source_row = existing_map[path]
        locator_counts[source] += len(destinations[path])
        value = source_arrays[source][source_row]
        for horizon, row_index in destinations[path]:
            memories[horizon][row_index] = value
    for memory in memories.values():
        memory.flush()
    del source_arrays

    arm = E.VJepa21CroppedV03Arm()
    arm_identity = arm.identity()
    preprocessing_digest = E.preprocessing_hash(arm)
    if preprocessing_digest != C.TARGET_CACHE_CONTRACT["preprocessing_digest"]:
        _die("frozen target preprocessing identity changed")
    identity_material = json.dumps(arm_identity, sort_keys=True)
    for required in (
        C.TARGET_CACHE_CONTRACT["target_encoder_checkpoint_sha256"],
        C.TARGET_CACHE_CONTRACT["target_encoder_constructor"],
        C.TARGET_CACHE_CONTRACT["target_encoder_repository_commit"],
    ):
        if str(required) not in identity_material:
            _die(f"target encoder identity omits frozen binding {required}")
    module = arm.build(device, torch.float32)
    module.eval()
    started = time.time()
    encode_batch = int(args.encode_batch)
    if encode_batch <= 0:
        _die("--encode-batch must be positive")
    with torch.no_grad():
        for offset in range(start, len(paths), encode_batch):
            selected = paths[offset:offset + encode_batch]
            pixels = torch.stack([arm.preprocess(path) for path in selected]).to(
                device, torch.float32
            )
            values = module(pixels.unsqueeze(2)).half().cpu().numpy()
            if values.shape != (len(selected), P.TOKENS, P.TOKEN_DIM):
                _die(f"target encoder output shape differs: {values.shape}")
            if not bool(np.isfinite(values).all()):
                _die("target encoder emitted non-finite values")
            for local, path in enumerate(selected):
                for horizon, row_index in destinations[path]:
                    memories[horizon][row_index] = values[local]
            complete = offset + len(selected)
            for memory in memories.values():
                memory.flush()
            _write_json(progress_path, {
                "common_rows_digest": manifest["common_rows_digest"],
                "unique_path_digest": _sequence_digest(paths),
                "unique_paths_completed": complete,
                "unique_paths_total": len(paths),
            })
            print(f"[targets] {complete}/{len(paths)} unique rasters", flush=True)
    del module, memories
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for horizon in (3, 4):
        partial_paths[horizon].replace(final_paths[horizon])
        _make_read_only(final_paths[horizon])

    caches = {}
    for horizon in (3, 4):
        path = final_paths[horizon]
        if path.stat().st_size != expected_bytes:
            _die(f"H{horizon} target cache byte count differs")
        caches[str(horizon)] = {
            "path": str(path), "shape": list(shape), "dtype": "float16",
            "bytes": path.stat().st_size, "sha256": _sha256_file(path),
        }
    # These existing selection caches are recorded read-only; none is rewritten.
    selection_reuse = {}
    for horizon in (3, 4):
        path = Path(HSEQ.OUT) / f"target_h{horizon}.f16"
        selection_reuse[str(horizon)] = {
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": _sha256_file(path), "rewritten": False,
        }
    receipt = {
        "schema": "go2_rgb_control_history_four_step_target_cache_index_v1",
        "status": STATUS,
        "complete": True,
        "four_step_contract_digest": _contract_digest(contract),
        "common_rows_digest": manifest["common_rows_digest"],
        "target_encoder_checkpoint_sha256": C.TARGET_CACHE_CONTRACT[
            "target_encoder_checkpoint_sha256"
        ],
        "target_encoder_digest": C.TARGET_CACHE_CONTRACT["target_encoder_digest"],
        "target_encoder_identity": arm_identity,
        "preprocessing_digest": preprocessing_digest,
        "encoder_constructor": C.TARGET_CACHE_CONTRACT[
            "target_encoder_constructor"
        ],
        "encoder_repository_commit": C.TARGET_CACHE_CONTRACT[
            "target_encoder_repository_commit"
        ],
        "target_encoder_executions": len(paths),
        "unique_raw_target_paths": len(all_paths),
        "unique_paths_reused_from_frozen_caches": len(cached_paths),
        "cached_row_horizon_outputs": 2 * len(train_rows) - cache_misses,
        "row_horizon_cache_misses": cache_misses,
        "row_horizon_outputs": 2 * len(train_rows),
        "encoded_path_list_digest": _sequence_digest(paths),
        "reused_path_list_digest": _sequence_digest(cached_paths),
        "source_cache_output_counts": {
            source: int(locator_counts[source]) for source in existing_sources
        },
        "source_caches": {
            source: {"path": str(path), "bytes": path.stat().st_size,
                     "rows": rows,
                     "sha256": next(
                         record["sha256"]
                         for record in input_verification["files"].values()
                         if record["path"] == str(path)
                     )}
            for source, (path, rows) in existing_sources.items()
        },
        "training_input_verification_digest": input_verification["verification_digest"],
        "only_missing_train_H3_H4_encoded": True,
        "H1_H2_train_reused": True,
        "selection_H3_H4_reused": selection_reuse,
        "caches": caches,
        "wall_seconds": time.time() - started,
        "new_simulator_corpus_generated": False,
    }
    receipt["target_cache_index_digest"] = _digest(receipt)
    _write_json_once(receipt_path, receipt)
    progress_path.unlink()
    return validate_target_cache_index()


def validate_target_cache_index() -> dict[str, Any]:
    receipt_path = runtime_root() / ENCODE_RECEIPT
    _require_immutable_file(receipt_path, "target cache index")
    receipt = _read_json(receipt_path)
    recorded = receipt.get("target_cache_index_digest")
    payload = {key: value for key, value in receipt.items()
               if key != "target_cache_index_digest"}
    if recorded != _digest(payload):
        _die("target cache index self digest differs")
    expected_keys = {
        "schema", "status", "complete", "four_step_contract_digest",
        "common_rows_digest", "target_encoder_checkpoint_sha256",
        "target_encoder_digest", "target_encoder_identity",
        "preprocessing_digest", "encoder_constructor",
        "encoder_repository_commit", "target_encoder_executions",
        "unique_raw_target_paths", "unique_paths_reused_from_frozen_caches",
        "cached_row_horizon_outputs", "row_horizon_cache_misses",
        "row_horizon_outputs", "encoded_path_list_digest",
        "reused_path_list_digest", "source_cache_output_counts",
        "source_caches", "training_input_verification_digest",
        "only_missing_train_H3_H4_encoded", "H1_H2_train_reused",
        "selection_H3_H4_reused", "caches", "wall_seconds",
        "new_simulator_corpus_generated", "target_cache_index_digest",
    }
    if set(receipt) != expected_keys or receipt.get("schema") != (
        "go2_rgb_control_history_four_step_target_cache_index_v1"
    ) or receipt.get("status") != STATUS or receipt.get("complete") is not True:
        _die("target cache index schema/completion differs")
    contract = require_contract()
    if receipt["four_step_contract_digest"] != _contract_digest(contract):
        _die("target cache index contract binding differs")
    manifest = validate_common_manifest()
    if receipt.get("common_rows_digest") != manifest["common_rows_digest"]:
        _die("target cache index binds a different common manifest")
    shape = tuple(C.TARGET_CACHE_CONTRACT["dense_cache_shape_each"])
    expected_bytes = int(np.prod(shape)) * 2
    arm = E.VJepa21CroppedV03Arm()
    expected_identity = arm.identity()
    identity_material = json.dumps(expected_identity, sort_keys=True)
    for required in (
        C.TARGET_CACHE_CONTRACT["target_encoder_checkpoint_sha256"],
        C.TARGET_CACHE_CONTRACT["target_encoder_constructor"],
        C.TARGET_CACHE_CONTRACT["target_encoder_repository_commit"],
    ):
        if str(required) not in identity_material:
            _die("live target encoder identity omits a frozen binding")
    frozen_scalar = {
        "target_encoder_checkpoint_sha256": C.TARGET_CACHE_CONTRACT[
            "target_encoder_checkpoint_sha256"
        ],
        "target_encoder_digest": C.TARGET_CACHE_CONTRACT["target_encoder_digest"],
        "target_encoder_identity": expected_identity,
        "preprocessing_digest": C.TARGET_CACHE_CONTRACT["preprocessing_digest"],
        "encoder_constructor": C.TARGET_CACHE_CONTRACT[
            "target_encoder_constructor"
        ],
        "encoder_repository_commit": C.TARGET_CACHE_CONTRACT[
            "target_encoder_repository_commit"
        ],
        "target_encoder_executions": C.TARGET_CACHE_CONTRACT[
            "unique_train_frames_requiring_encoder_execution"
        ],
        "unique_raw_target_paths": 7063,
        "unique_paths_reused_from_frozen_caches": 1665,
        "cached_row_horizon_outputs": 2018,
        "row_horizon_cache_misses": C.TARGET_CACHE_CONTRACT[
            "row_horizon_cache_misses"
        ],
        "row_horizon_outputs": C.TARGET_CACHE_CONTRACT["output_entries"],
        "only_missing_train_H3_H4_encoded": True,
        "H1_H2_train_reused": True,
        "new_simulator_corpus_generated": False,
    }
    for key, expected in frozen_scalar.items():
        if receipt.get(key) != expected:
            _die(f"target cache index {key} differs from frozen contract")
    if E.preprocessing_hash(arm) != receipt["preprocessing_digest"]:
        _die("target cache preprocessing implementation differs")
    for horizon in (3, 4):
        record = receipt.get("caches", {}).get(str(horizon), {})
        path = Path(str(record.get("path", "")))
        if path != runtime_root() / TARGET_BLOBS[horizon]:
            _die(f"H{horizon} target cache path differs")
        _require_immutable_file(path, f"H{horizon} target cache")
        if path.stat().st_size != expected_bytes:
            _die(f"H{horizon} target cache missing or wrong-sized")
        if record.get("sha256") != _sha256_file(path):
            _die(f"H{horizon} target cache digest differs")
        if record.get("shape") != list(shape) or record.get("dtype") != "float16" \
                or int(record.get("bytes", -1)) != expected_bytes:
            _die(f"H{horizon} target cache metadata differs")
    input_verification = validate_training_input_files(hash_files=False)
    if receipt["training_input_verification_digest"] != input_verification[
        "verification_digest"
    ]:
        _die("target cache index training-input binding differs")
    existing_map, sources = existing_target_path_map()
    if set(receipt["source_caches"]) != set(sources):
        _die("target cache source-cache inventory differs")
    for source, (path, rows) in sources.items():
        expected_input = next(
            value for value in input_verification["files"].values()
            if value["path"] == str(path)
        )
        observed = receipt["source_caches"][source]
        if observed != {
            "path": str(path), "bytes": path.stat().st_size, "rows": rows,
            "sha256": expected_input["sha256"],
        }:
            _die(f"target cache source binding differs for {source}")
    destinations: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    train_rows = [row for row in manifest["rows"] if row["split"] == "train"]
    for row_index, row in enumerate(train_rows):
        env_text = f"{int(row['env_index']):02d}"
        for horizon in (3, 4):
            frame_index = int(row["horizon_frame_indices"][horizon - 1])
            image = Path(HSEQ.V03) / row["scene"] / "rgb" / (
                f"frame_{frame_index:06d}_env_{env_text}.png"
            )
            destinations[str(image)].append((horizon, row_index))
    all_paths = sorted(destinations)
    reused_paths = [path for path in all_paths if path in existing_map]
    encoded_paths = [path for path in all_paths if path not in existing_map]
    locator_counts: collections.Counter[str] = collections.Counter()
    for path in reused_paths:
        source, _ = existing_map[path]
        locator_counts[source] += len(destinations[path])
    if len(all_paths) != 7063 or len(reused_paths) != 1665 \
            or len(encoded_paths) != 5398 \
            or sum(len(destinations[path]) for path in encoded_paths) != 5690 \
            or receipt["encoded_path_list_digest"] != _sequence_digest(encoded_paths) \
            or receipt["reused_path_list_digest"] != _sequence_digest(reused_paths) \
            or receipt["source_cache_output_counts"] != {
                source: int(locator_counts[source]) for source in sources
            }:
        _die("target cache path/reuse reduction differs")
    for horizon in (3, 4):
        path = Path(HSEQ.OUT) / f"target_h{horizon}.f16"
        expected_input = next(
            value for value in input_verification["files"].values()
            if value["path"] == str(path)
        )
        if receipt["selection_H3_H4_reused"].get(str(horizon)) != {
            "path": str(path), "bytes": int(expected_input["bytes"]),
            "sha256": expected_input["sha256"], "rewritten": False,
        }:
            _die(f"selection H{horizon} reuse binding differs")
    if (runtime_root() / "target_encoding_progress.json").exists() \
            or any((runtime_root() / f"{name}.partial").exists()
                   for name in TARGET_BLOBS.values()):
        _die("completed target cache retains a partial/progress artifact")
    return receipt


def validate_training_input_files(hash_files: bool = False) -> dict[str, Any]:
    """One durable hash verification of every frozen cache/manifest input."""
    receipt_path = runtime_root() / "training_input_verification.json"
    if receipt_path.is_file():
        _require_immutable_file(receipt_path, "training input verification")
        receipt = _read_json(receipt_path)
        recorded = receipt.get("verification_digest")
        if recorded != _digest({key: value for key, value in receipt.items()
                                if key != "verification_digest"}):
            _die("training-input verification receipt digest differs")
        if set(receipt.get("files", {})) != set(C.FROZEN_TRAINING_INPUT_FILES):
            _die("training-input verification file set differs")
        for name, expected in C.FROZEN_TRAINING_INPUT_FILES.items():
            observed = receipt["files"][name]
            if observed != {
                "path": str(Path(expected["path"])),
                "bytes": int(expected["bytes"]),
                "sha256": expected["sha256"],
            }:
                _die(f"training-input verification binding differs for {name}")
        return receipt
    if not hash_files:
        _die("frozen training inputs have not been hash-verified")
    observed: dict[str, dict[str, Any]] = {}
    for name, expected in C.FROZEN_TRAINING_INPUT_FILES.items():
        path = Path(expected["path"])
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
            _die(f"frozen input {name} is missing or wrong-sized: {path}")
        sha256 = _sha256_file(path)
        if sha256 != expected["sha256"]:
            _die(f"frozen input {name} digest differs")
        observed[name] = {
            "path": str(path), "bytes": path.stat().st_size, "sha256": sha256,
        }
        print(f"[input hash] {name}", flush=True)
    receipt = {
        "schema": "go2_rgb_control_history_four_step_training_input_verification_v1",
        "status": STATUS, "complete": True, "files": observed,
    }
    receipt["verification_digest"] = _digest(receipt)
    _write_json_once(receipt_path, receipt)
    return receipt


class FourStepLoader:
    """One ordered H4-valid training path across all four target horizons."""

    def __init__(self) -> None:
        self.manifest = validate_common_manifest()
        self.cache_index = validate_target_cache_index()
        self.factorial = FM.load()
        self.map_record = MAP.load()
        all_rows, _, self.stats = F.load_rows()
        self.entries = [
            row for row in self.manifest["rows"] if row["split"] == "train"
        ]
        if len(self.entries) != EXPECTED_COMMON["train"]:
            _die("four-step loader train count differs")
        factorial_by_position = {
            int(row["position"]): row for row in self.factorial["rows"]
        }
        self.factorial_rows = [factorial_by_position[int(row["position"])]
                               for row in self.entries]
        self.rows = [all_rows[int(row["manifest_row_index"])]
                     for row in self.factorial_rows]
        for common, factorial, row in zip(
            self.entries, self.factorial_rows, self.rows, strict=True
        ):
            if common["pair_sha256"] != factorial["pair_sha256"] \
                    or common["pair_sha256"] != row["pair_sha256"]:
                _die("four-step loader row alignment differs")
            if len(row["action_blocks"]) < 4 or any(
                len(block) != P.ACTION_DIM for block in row["action_blocks"][:4]
            ):
                _die("four-step loader action block contract differs")

        n_train = int(self.map_record["source_train"])
        n_selection = int(self.map_record["source_selection"])
        self.ctx0 = R.load_cache(F.DIAG_CACHE / "frozen_train_ctx0.f16", n_train)
        self.ctx1 = R.load_cache(F.DIAG_CACHE / "frozen_train_ctx1.f16", n_train)
        self.ctx2 = R.load_cache(
            F.EVAL_CACHE / "frozen_current.f16", n_train + n_selection
        )[:n_train]
        self.y1 = R.load_cache(F.EVAL_CACHE / "frozen_train_future.f16", n_train)
        step2_path = F.TWO_CACHE / "frozen_train_step2.f16"
        step2_rows = step2_path.stat().st_size // (P.TOKENS * P.TOKEN_DIM * 2)
        self.y2 = R.load_cache(step2_path, step2_rows)
        self.y3 = R.load_cache(runtime_root() / TARGET_BLOBS[3], len(self.entries))
        self.y4 = R.load_cache(runtime_root() / TARGET_BLOBS[4], len(self.entries))

    def __len__(self) -> int:
        return len(self.entries)

    def batch(self, positions: Sequence[int], device: torch.device) -> dict[str, Any]:
        entries = [self.factorial_rows[index] for index in positions]
        rows = [self.rows[index] for index in positions]
        cache = [int(entry["cache_index"]) for entry in entries]
        step2 = [int(entry["step2_cache_index"]) for entry in entries]
        context = torch.stack([
            T.normalise(self.ctx0[cache].float()),
            T.normalise(self.ctx1[cache].float()),
            T.normalise(self.ctx2[cache].float()),
        ], dim=1).to(device)
        targets = (
            T.normalise(self.y1[cache].float()).to(device),
            T.normalise(self.y2[step2].float()).to(device),
            T.normalise(self.y3[list(positions)].float()).to(device),
            T.normalise(self.y4[list(positions)].float()).to(device),
        )
        actions = tuple(
            torch.tensor([row["action_blocks"][horizon] for row in rows],
                         dtype=torch.float32, device=device)
            for horizon in range(4)
        )
        control = torch.tensor(
            [row["control"] for row in rows], dtype=torch.float32, device=device
        ).reshape(len(rows), 3, P.SAMPLES_PER_SLOT, P.CONTROL_DIM)
        control_mean = torch.tensor(
            self.stats["control_mean"], dtype=torch.float32, device=device
        )
        control_std = torch.tensor(
            self.stats["control_std"], dtype=torch.float32, device=device
        )
        control = (control - control_mean) / control_std
        return {
            "context": context,
            "targets": targets,
            "actions": actions,
            "control": control,
            "stable_row_id": [self.entries[index]["stable_row_id"]
                              for index in positions],
        }


def four_step_objective(
    outputs: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Frozen L1 latent loss at H1..H4, combined with exact equal weights."""
    if len(outputs) != 4 or len(targets) != 4:
        _die("four-step objective requires exactly four outputs and targets")
    component = tuple(
        (prediction - target).abs().mean()
        for prediction, target in zip(outputs, targets, strict=True)
    )
    return torch.stack(component).mean(), component


def forward_four_step(
    model: P.ProprioActionPredictor, batch: dict[str, Any]
) -> tuple[list[torch.Tensor], torch.Tensor, tuple[torch.Tensor, ...]]:
    outputs = P.unroll(
        model, batch["context"], batch["actions"], proprio=None,
        control=batch["control"], max_h=4,
    )
    loss, components = four_step_objective(outputs, batch["targets"])
    return outputs, loss, components


def objective_component_separation(
    outputs: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
) -> dict[str, Any]:
    """Prove each registered horizon perturbs only its own Li and has weight .25."""
    del targets
    with torch.no_grad():
        # Use the real output shapes/dtypes but an exact zero-residual fixture.
        # This avoids cancellation in mean absolute error when a real residual
        # happens to contain balanced signs.
        synthetic_targets = [output.detach() for output in outputs]
        _, baseline = four_step_objective(outputs, synthetic_targets)
        baseline_values = [float(value) for value in baseline]
        matrix: list[list[bool]] = []
        for changed in range(4):
            altered = list(outputs)
            altered[changed] = synthetic_targets[changed] + 0.25
            _, components = four_step_objective(altered, synthetic_targets)
            values = [float(value) for value in components]
            matrix.append([
                value != baseline_values[index]
                for index, value in enumerate(values)
            ])
    scalar = [torch.tensor(1.0 + index, requires_grad=True) for index in range(4)]
    torch.stack(scalar).mean().backward()
    derivatives = [float(value.grad) for value in scalar]
    expected = [[index == changed for index in range(4)] for changed in range(4)]
    return {
        "baseline_component_losses": baseline_values,
        "changed_component_matrix": matrix,
        "expected_changed_component_matrix": expected,
        "only_registered_component_changes": matrix == expected,
        "combined_loss_derivative_per_component": derivatives,
        "all_derivatives_exactly_one_quarter": derivatives == [0.25] * 4,
    }


def _optimizer_state_digest(optimizer: torch.optim.Optimizer) -> str:
    return _optimizer_state_dict_digest(optimizer.state_dict())


def _optimizer_state_dict_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_compact(state["param_groups"]))
    for parameter_id in sorted(state["state"]):
        digest.update(str(parameter_id).encode("ascii"))
        for name, value in sorted(state["state"][parameter_id].items()):
            digest.update(name.encode("utf-8"))
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(_canonical_compact(list(tensor.shape)))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(_canonical_compact(value))
    return digest.hexdigest()


def _all_finite_model_optimizer(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> bool:
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        return False
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                return False
    return True


def _gradient_report(model: nn.Module) -> dict[str, Any]:
    missing, nonfinite, zero = [], [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        elif float(parameter.grad.abs().sum()) == 0.0:
            zero.append(name)
    return {
        "parameter_count": len(list(model.parameters())),
        "missing": missing, "nonfinite": nonfinite, "zero": zero,
        "all_present_and_finite": not missing and not nonfinite,
    }


def _registered_base(seed: int) -> tuple[Path, dict[str, Any]]:
    if seed not in C.FROZEN_SEEDS:
        _die(f"seed {seed} is not in the frozen eight")
    seed_root = F.OUT / f"seed_{seed}"
    path = seed_root / f"seed_{seed}_base_weights.pt"
    expected = C.BASE_WEIGHT_SHA256[seed]
    if not path.is_file() or _sha256_file(path) != expected:
        _die(f"registered base weights changed for seed {seed}")
    run = _read_json(seed_root / "run_record.json")
    if run.get("base_weights_sha256") != expected:
        _die(f"historical run record binds other base weights for seed {seed}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("shared_state_dict")
    if not isinstance(state, dict) or payload.get("state_digest") != F.state_digest(state):
        _die(f"registered base state digest changed for seed {seed}")
    if payload["state_digest"] != C.BASE_STATE_DIGEST[seed]:
        _die(f"registered base state differs from frozen state digest for seed {seed}")
    return path, payload


def _fresh_model(seed: int, device: torch.device) -> tuple[
    P.ProprioActionPredictor, Path, dict[str, Any]
]:
    path, payload = _registered_base(seed)
    model = F.make_cell_model(
        "rgb_one_step", seed, path, WIDTH, DEPTH, HEADS
    ).to(device)
    if _model_state_digest(model) != payload["state_digest"]:
        _die(f"model is not the registered base state for seed {seed}")
    F.assert_no_active_dropout(model)
    return model, path, payload


def _start_once(name: str, success_path: Path) -> None:
    if success_path.is_file():
        return
    attempt = runtime_root() / "attempts" / f"{name}.json"
    if attempt.exists():
        _die(f"one-shot stage {name} was already attempted without success")
    contract = require_contract()
    _write_json_once(attempt, {
        "schema": "go2_rgb_control_history_four_step_attempt_v1",
        "stage": name,
        "four_step_contract_digest": _contract_digest(contract),
        "started_unix_ns": time.time_ns(),
        "process_id": os.getpid(),
        "environment": environment_record(require_exact=True),
    })


def _one_batch_step(
    model: nn.Module, optimizer: torch.optim.Optimizer,
    batch: dict[str, Any], device: torch.device,
) -> tuple[float, list[float], dict[str, Any]]:
    optimizer.zero_grad()
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        _, loss, components = forward_four_step(model, batch)
    if not bool(torch.isfinite(loss)):
        _die("four-step objective is non-finite")
    loss.backward()
    gradients = _gradient_report(model)
    if not gradients["all_present_and_finite"]:
        _die(f"four-step parameter gradient failure: {gradients}")
    clip_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not bool(torch.isfinite(clip_norm)):
        _die("pre-clip gradient norm is non-finite")
    optimizer.step()
    if not _all_finite_model_optimizer(model, optimizer):
        _die("model or optimizer became non-finite")
    return float(loss.detach()), [float(value.detach()) for value in components], gradients


def smoke_stage(args: argparse.Namespace) -> dict[str, Any]:
    """One tiny real-feature smoke; its warmup is discarded before training."""
    contract = require_contract()
    environment = environment_record(require_exact=True)
    receipt_path = runtime_root() / "smoke.json"
    if receipt_path.is_file():
        return validate_smoke_receipt()
    _start_once("smoke", receipt_path)
    validate_training_input_files()
    loader = FourStepLoader()
    device = resolve_device(args.device)
    seed = int(C.FROZEN_SEEDS[0])
    model, base_path, base = _fresh_model(seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False
    )
    plan = common_batch_plan(seed, 0, loader)
    batch = loader.batch(plan[0], device)
    model.train()
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        separation_outputs, _, _ = forward_four_step(model, batch)
    component_separation = objective_component_separation(
        separation_outputs, batch["targets"]
    )
    del separation_outputs
    if not component_separation["only_registered_component_changes"] \
            or not component_separation["all_derivatives_exactly_one_quarter"]:
        _die("H1-H4 objective component separation failed")
    initial_loss, initial_components, gradient = _one_batch_step(
        model, optimizer, batch, device
    )
    objective_separation = {
        "component_losses_H1_H4": initial_components,
        "combined": initial_loss,
        "arithmetic_mean": float(np.mean(initial_components)),
        # ``np.mean`` returns a NumPy scalar; receipts must contain only
        # ordinary JSON-native values.  This is a receipt-boundary conversion
        # only—the objective calculation and its truth value are unchanged.
        "exact_equal_weight_formula": bool(
            abs(initial_loss - np.mean(initial_components)) < 1e-7
        ),
    }
    if not objective_separation["exact_equal_weight_formula"]:
        _die("four-step objective is not the arithmetic mean of H1-H4")

    # Save/resume the updated smoke state exactly; this directory is temporary
    # and has no relationship to the later scientific epoch-21 checkpoints.
    updated_model_digest = _model_state_digest(model)
    updated_optimizer_digest = _optimizer_state_digest(optimizer)
    with tempfile.TemporaryDirectory(dir=runtime_root()) as temporary:
        checkpoint = Path(temporary) / "smoke_checkpoint.pt"
        save_receipt = CK.save(
            checkpoint, model=model, optimizer=optimizer, epoch=0, global_step=1,
            seed=seed,
            model_config={"cell": "rgb_four_step", "use_proprio": False,
                          "width": WIDTH, "depth": DEPTH, "heads": HEADS},
            scheduler=None,
            scheduler_absent_reason="fixed learning rate; no scheduler is constructed",
            data_order_generator=F.stream(seed, "data_order", 0),
            extra={"smoke_only": True},
        )
        resumed, _, _ = _fresh_model(seed, device)
        resumed_optimizer = torch.optim.AdamW(
            resumed.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False
        )
        CK.load_for_resume(
            checkpoint, model=resumed, optimizer=resumed_optimizer,
            data_order_generator=F.stream(seed, "data_order", 0),
        )
        checkpoint_exact = (
            _model_state_digest(resumed) == updated_model_digest
            and _optimizer_state_digest(resumed_optimizer) == updated_optimizer_digest
        )
        if not checkpoint_exact:
            _die("smoke checkpoint save/resume is not exact")
        next_batch = loader.batch(plan[1], device)
        original_next = _one_batch_step(model, optimizer, next_batch, device)
        resumed_next = _one_batch_step(resumed, resumed_optimizer, next_batch, device)
        next_exact = (
            original_next[:2] == resumed_next[:2]
            and _model_state_digest(model) == _model_state_digest(resumed)
            and _optimizer_state_digest(optimizer)
            == _optimizer_state_digest(resumed_optimizer)
        )
        checkpoint_record = {
            **save_receipt, "exact_resume": checkpoint_exact,
            "next_batch_loss_and_components_equal": original_next[:2] == resumed_next[:2],
            "next_batch_update_state_equal": next_exact,
        }
        if not next_exact:
            _die("resumed smoke next-batch update differs from uninterrupted update")
        model, optimizer = resumed, resumed_optimizer

    # AdaLN-Zero is context-independent at initialisation.  Warm exactly fifty
    # non-scientific steps, prove H3/H4 reach all preceding predictions, and then
    # discard this state by reloading the registered base artefact.
    warmup_steps = 50
    flat_plan = [indices for epoch in range(2)
                 for indices in common_batch_plan(seed, epoch, loader)]
    for step in range(2, warmup_steps):
        warm = loader.batch(flat_plan[step], device)
        _one_batch_step(model, optimizer, warm, device)

    def chain_probe(target_horizon: int) -> dict[str, Any]:
        model.zero_grad(set_to_none=True)
        probe_batch = loader.batch(plan[1], device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            outputs, _, _ = forward_four_step(model, probe_batch)
            for output in outputs[:target_horizon - 1]:
                output.retain_grad()
            loss = (outputs[target_horizon - 1]
                    - probe_batch["targets"][target_horizon - 1]).abs().mean()
        loss.backward()
        preceding = []
        for horizon, output in enumerate(outputs[:target_horizon - 1], 1):
            finite = output.grad is not None and bool(torch.isfinite(output.grad).all())
            magnitude = float(output.grad.abs().sum()) if finite else None
            preceding.append({"horizon": horizon, "finite": finite,
                              "gradient_abs_sum": magnitude,
                              "nonzero": bool(finite and magnitude and magnitude > 0.0)})
        report = _gradient_report(model)
        return {"loss_horizon": target_horizon, "preceding_predictions": preceding,
                "parameter_gradients": report,
                "all_preceding_finite_nonzero": all(item["nonzero"] for item in preceding),
                "all_parameter_gradients_present_finite": report["all_present_and_finite"]}

    h3 = chain_probe(3)
    h4 = chain_probe(4)
    if not h3["all_preceding_finite_nonzero"] or not h4["all_preceding_finite_nonzero"] \
            or not h3["all_parameter_gradients_present_finite"] \
            or not h4["all_parameter_gradients_present_finite"]:
        _die("H3/H4 loss does not backpropagate through the full autoregressive chain")
    discarded, _, _ = _fresh_model(seed, device)
    discarded_digest = _model_state_digest(discarded)
    if discarded_digest != base["state_digest"]:
        _die("registered base did not reload after smoke warmup")
    del model, optimizer, discarded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    receipt = {
        "schema": "go2_rgb_control_history_four_step_smoke_v1",
        "status": STATUS, "complete": True, "valid": True,
        "four_step_contract_digest": _contract_digest(contract),
        "environment": environment,
        "seed": seed, "base_weights_path": str(base_path),
        "base_weights_sha256": C.BASE_WEIGHT_SHA256[seed],
        "real_feature_rows": batch["stable_row_id"],
        "objective_separation": objective_separation,
        "component_perturbation_separation": component_separation,
        "initial_parameter_gradients": gradient,
        "checkpoint_save_resume": checkpoint_record,
        "warmup_steps": warmup_steps,
        "warmup_scientific_training": False,
        "H3_chain_probe": h3, "H4_chain_probe": h4,
        "warmup_state_discarded": True,
        "registered_base_reloaded_digest": discarded_digest,
        "scientific_optimizer_step_performed": False,
        "calibration_or_counterfactual_corpus_opened": False,
    }
    receipt["smoke_digest"] = _digest(receipt)
    _write_json_once(receipt_path, receipt)
    return validate_smoke_receipt()


def _train_epoch(
    model: nn.Module, optimizer: torch.optim.Optimizer, loader: FourStepLoader,
    seed: int, epoch: int, device: torch.device,
    monitor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.train()
    plan = common_batch_plan(seed, epoch, loader)
    total = np.zeros(5, dtype=np.float64)
    started = time.time()
    for batch_index, positions in enumerate(plan):
        batch = loader.batch(positions, device)
        loss, components, _ = _one_batch_step(model, optimizer, batch, device)
        total[:4] += np.asarray(components, dtype=np.float64)
        total[4] += loss
        if monitor is not None:
            monitor["minimum_mem_available_bytes"] = min(
                int(monitor["minimum_mem_available_bytes"]), _mem_available_bytes()
            )
            monitor["process_peak_rss_bytes"] = max(
                int(monitor["process_peak_rss_bytes"]),
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            )
        if batch_index % 100 == 0:
            print(f"[seed {seed} epoch {epoch}] {batch_index}/{len(plan)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    means = total / len(plan)
    return {
        "epoch": epoch, "batches": len(plan),
        "e1": float(means[0]), "e2": float(means[1]),
        "e3": float(means[2]), "e4": float(means[3]),
        "loss": float(means[4]),
        "batch_plan_digest": _digest(plan),
        "wall_seconds": time.time() - started,
    }


def common_batch_plan(seed: int, epoch: int, loader: FourStepLoader) -> list[list[int]]:
    """Historical 3,922-row order filtered to the 3,854 common rows, then rebatch."""
    plan = common_plan_from_rows(seed, epoch, loader.manifest["rows"])
    expected = loader.manifest["per_seed_epoch_order_digests"][str(seed)][str(epoch)]
    sequence = [index for batch in plan for index in batch]
    if _sequence_digest(sequence) != expected["sequence_digest"] \
            or _compact_digest(plan) != expected["batch_digest"]:
        _die("live common batch plan differs from frozen manifest receipt")
    return plan


def resource_gate(
    *, peak_vram_bytes: int, minimum_mem_available_bytes: int,
    filesystem_free_bytes: int, projected_remaining_bytes: int,
) -> dict[str, Any]:
    vram_limit = int(C.RESOURCE_GATES["peak_vram_strictly_below_bytes"])
    ram_floor = int(C.RESOURCE_GATES["free_system_ram_strictly_above_bytes"])
    checks = {
        "peak_vram_strictly_below_28_GiB": peak_vram_bytes < vram_limit,
        "minimum_free_system_ram_strictly_above_20_GiB":
            minimum_mem_available_bytes > ram_floor,
        "filesystem_covers_all_remaining_outputs_with_2_GiB_reserve":
            filesystem_free_bytes > projected_remaining_bytes + 2 * 2**30,
    }
    return {"checks": checks, "pass": all(checks.values()),
            "limits": {"peak_vram_bytes": vram_limit,
                       "minimum_free_system_ram_bytes": ram_floor,
                       "filesystem_reserve_bytes": 2 * 2**30}}


def preflight_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Measure exactly one complete four-step training epoch."""
    contract = require_contract()
    environment = environment_record(require_exact=True)
    receipt_path = runtime_root() / "resource_preflight.json"
    if receipt_path.is_file():
        return validate_preflight_receipt()
    _start_once("preflight", receipt_path)
    smoke = validate_smoke_receipt()
    if smoke.get("valid") is not True:
        _die("valid smoke is required before resource preflight")
    validate_training_input_files()
    loader = FourStepLoader()
    device = resolve_device(args.device)
    if device.type != "cuda":
        _die("resource preflight must run on the registered R9700")
    seed = int(C.FROZEN_SEEDS[0])
    model, _, base = _fresh_model(seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    monitor = {
        "minimum_mem_available_bytes": _mem_available_bytes(),
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ) * 1024,
    }
    epoch = _train_epoch(model, optimizer, loader, seed, 0, device, monitor)
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    target_cache_bytes = sum(
        (runtime_root() / TARGET_BLOBS[horizon]).stat().st_size for horizon in (3, 4)
    )
    comparator_sizes = []
    for comparator_seed in C.FROZEN_SEEDS:
        for cell in ("rgb_one_step", "rgb_rollout"):
            path = F.OUT / f"seed_{comparator_seed}" / (
                f"seed_{comparator_seed}_{cell}_epoch21.pt"
            )
            comparator_sizes.append(path.stat().st_size)
    checkpoint_projection = max(comparator_sizes) * len(C.FROZEN_SEEDS)
    evaluation_projection = 1 * 2**30  # ledgers, occupancy counts and receipts; no latent shards
    receipt_projection = 64 * 2**20
    projected_remaining = checkpoint_projection + evaluation_projection + receipt_projection
    disk = shutil.disk_usage(runtime_root())
    system_ram_total = _mem_total_bytes()
    gate = resource_gate(
        peak_vram_bytes=peak_reserved,
        minimum_mem_available_bytes=int(monitor["minimum_mem_available_bytes"]),
        filesystem_free_bytes=int(disk.free),
        projected_remaining_bytes=int(projected_remaining),
    )
    # The measured epoch is never a scientific epoch.  Prove that discarding it
    # and rebuilding returns the exact registered base state.
    del model, optimizer
    torch.cuda.empty_cache()
    reloaded, _, _ = _fresh_model(seed, device)
    base_reloaded = _model_state_digest(reloaded) == base["state_digest"]
    del reloaded
    torch.cuda.empty_cache()
    if not base_reloaded:
        _die("preflight epoch state was not discarded exactly")
    receipt = {
        "schema": "go2_rgb_control_history_four_step_resource_preflight_v1",
        "status": STATUS, "complete": True, "valid": bool(gate["pass"]),
        "four_step_contract_digest": _contract_digest(contract),
        "environment": environment,
        "full_epochs_measured": 1, "batch_size": BATCH,
        "epoch": epoch,
        "wall_seconds_per_epoch": epoch["wall_seconds"],
        "projected_eight_run_wall_seconds":
            epoch["wall_seconds"] * EPOCHS * len(C.FROZEN_SEEDS),
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "process_peak_rss_bytes": monitor["process_peak_rss_bytes"],
        "minimum_mem_available_bytes": monitor["minimum_mem_available_bytes"],
        "system_ram_total_bytes": system_ram_total,
        "peak_system_ram_used_bytes":
            system_ram_total - int(monitor["minimum_mem_available_bytes"]),
        "target_cache_storage_bytes": target_cache_bytes,
        "projected_eight_epoch21_checkpoint_bytes": checkpoint_projection,
        "projected_evaluation_and_receipt_bytes":
            evaluation_projection + receipt_projection,
        "evaluation_storage_projection_note": (
            "direct and H2-H4 occupancy metrics consume the same sole in-memory "
            "prediction; only metric/similarity/count ledgers persist, never latent shards"
        ),
        "projected_remaining_bytes": projected_remaining,
        "destination_filesystem_free_bytes": disk.free,
        "destination_filesystem_total_bytes": disk.total,
        "gate": gate,
        "preflight_weights_discarded": True,
        "registered_base_reloaded": base_reloaded,
        "scientific_epoch_completed": False,
    }
    receipt["preflight_digest"] = _digest(receipt)
    _write_json_once(receipt_path, receipt)
    return validate_preflight_receipt()


def _seed_directory(seed: int) -> Path:
    return runtime_root() / "training" / f"seed_{seed}"


def _seed_receipt_path(seed: int) -> Path:
    return _seed_directory(seed) / "training_receipt.json"


def _seed_checkpoint_path(seed: int) -> Path:
    return _seed_directory(seed) / f"seed_{seed}_rgb_four_step_epoch21.pt"


def validate_training_receipt(seed: int) -> dict[str, Any]:
    if seed not in C.FROZEN_SEEDS:
        _die(f"training receipt seed is not registered: {seed}")
    path = _seed_receipt_path(seed)
    _require_immutable_file(path, f"training receipt seed {seed}")
    receipt = _read_json(path)
    recorded = receipt.get("training_receipt_digest")
    if recorded != _digest({key: value for key, value in receipt.items()
                            if key != "training_receipt_digest"}):
        _die(f"training receipt digest differs for seed {seed}")
    contract = require_contract()
    if receipt.get("schema") != (
        "go2_rgb_control_history_four_step_training_receipt_v1"
    ) or receipt.get("status") != STATUS or receipt.get("complete") is not True \
            or receipt.get("valid") is not True \
            or receipt.get("four_step_contract_digest") != _contract_digest(contract) \
            or receipt.get("seed") != seed:
        _die(f"training receipt schema/completion differs for seed {seed}")
    checkpoint = _seed_checkpoint_path(seed)
    _require_immutable_file(checkpoint, f"epoch-21 checkpoint seed {seed}")
    if receipt.get("checkpoint_sha256") != _sha256_file(checkpoint):
        _die(f"epoch-21 checkpoint digest differs for seed {seed}")
    if receipt.get("checkpoint_bytes") != checkpoint.stat().st_size \
            or receipt.get("checkpoint_verified_reloadable") is not True:
        _die(f"epoch-21 checkpoint byte/reload receipt differs for seed {seed}")
    if receipt.get("epochs_trained") != EPOCHS \
            or receipt.get("checkpoint_epoch") != CHECKPOINT_EPOCH:
        _die(f"training budget differs for seed {seed}")
    if receipt.get("best_epoch_selected") is not False \
            or receipt.get("extension_or_retry") is not False \
            or receipt.get("finite_weak_run_retained") is not True:
        _die(f"training selection/retry policy differs for seed {seed}")
    if receipt.get("base_weights_sha256") != C.BASE_WEIGHT_SHA256[seed] \
            or receipt.get("base_state_digest") != C.BASE_STATE_DIGEST[seed] \
            or receipt.get("initial_state_digest") != C.BASE_STATE_DIGEST[seed]:
        _die(f"training base-state binding differs for seed {seed}")
    manifest = validate_common_manifest()
    cache_index = validate_target_cache_index()
    history = receipt.get("history")
    if not isinstance(history, list) or len(history) != EPOCHS \
            or [row.get("epoch") for row in history] != list(range(EPOCHS)):
        _die(f"training history/data order differs for seed {seed}")
    plans = [common_plan_from_rows(seed, epoch, manifest["rows"])
             for epoch in range(EPOCHS)]
    for epoch, (row, plan) in enumerate(zip(history, plans, strict=True)):
        losses = [row.get(key) for key in ("e1", "e2", "e3", "e4")]
        if row.get("batch_plan_digest") != _digest(plan) \
                or row.get("batches") != len(plan) or len(plan) != 964 \
                or not all(isinstance(value, (int, float))
                           and math.isfinite(float(value)) for value in losses) \
                or not isinstance(row.get("loss"), (int, float)) \
                or not math.isfinite(float(row["loss"])) \
                or not isinstance(row.get("wall_seconds"), (int, float)) \
                or not math.isfinite(float(row["wall_seconds"])) \
                or float(row["wall_seconds"]) < 0.0 \
                or not math.isclose(float(row["loss"]), float(np.mean(losses)),
                                    abs_tol=1e-6, rel_tol=1e-6):
            _die(f"training epoch/update accounting differs for seed {seed}/{epoch}")
    if receipt.get("terminal_window") != F.terminal_window(history):
        _die(f"training terminal-window diagnostic differs for seed {seed}")
    ledger = Path(str(receipt.get("retained_checkpoint_receipt_ledger", "")))
    if ledger != checkpoint.parent / "checkpoint_receipts.jsonl":
        _die(f"checkpoint receipt ledger path differs for seed {seed}")
    _require_immutable_file(ledger, f"checkpoint receipt ledger seed {seed}")
    if receipt.get("retained_checkpoint_receipt_ledger_sha256") != _sha256_file(ledger):
        _die(f"checkpoint receipt ledger digest differs for seed {seed}")
    ledger_rows = _read_jsonl(ledger)
    if len(ledger_rows) != 1:
        _die(f"checkpoint receipt ledger content differs for seed {seed}")
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_state.get("schema") != CK.SCHEMA \
            or checkpoint_state.get("epoch") != CHECKPOINT_EPOCH \
            or checkpoint_state.get("seed") != seed \
            or checkpoint_state.get("base_state_digest") != C.BASE_STATE_DIGEST[seed] \
            or checkpoint_state.get("common_rows_digest") != manifest[
                "common_rows_digest"
            ] or checkpoint_state.get("target_cache_index_digest") != cache_index[
                "target_cache_index_digest"
            ]:
        _die(f"checkpoint payload bindings differ for seed {seed}")
    expected_model_config = {
        "cell": "rgb_four_step", "use_proprio": False,
        "objective": "(L1+L2+L3+L4)/4", "width": WIDTH,
        "depth": DEPTH, "heads": HEADS,
    }
    if checkpoint_state.get("global_step") != sum(
        row["batches"] for row in history[:CHECKPOINT_EPOCH + 1]
    ) or checkpoint_state.get("history") != history[:CHECKPOINT_EPOCH + 1] \
            or checkpoint_state.get("model_config") != expected_model_config:
        _die(f"checkpoint update/history/model contract differs for seed {seed}")
    expected_global_step = sum(row["batches"] for row in history[:22])
    if ledger_rows[0] != {
        "path": str(checkpoint), "bytes": checkpoint.stat().st_size,
        "sha256": receipt["checkpoint_sha256"], "epoch": CHECKPOINT_EPOCH,
        "global_step": expected_global_step,
        "optimizer_state_entries": len(checkpoint_state[
            "optimizer_state_dict"
        ]["state"]),
        "verified_reloadable": True,
        "durable": "fsync(file) + atomic replace + fsync(dir)",
    }:
        _die(f"checkpoint durable receipt fields differ for seed {seed}")
    retained = F.state_digest(checkpoint_state["model_state_dict"])
    if retained != receipt.get("retained_epoch21_state_digest") \
            or retained != receipt.get("retained_epoch21_strict_reload_state_digest"):
        _die(f"checkpoint retained state digest differs for seed {seed}")
    optimizer_digest = _optimizer_state_dict_digest(
        checkpoint_state["optimizer_state_dict"]
    )
    if optimizer_digest != receipt.get("retained_epoch21_optimizer_digest"):
        _die(f"checkpoint retained optimizer digest differs for seed {seed}")
    if checkpoint_state.get("scheduler_state_dict") is not None \
            or checkpoint_state.get("scheduler_absent_reason") != (
                "fixed learning rate; no scheduler is constructed"
            ) or not torch.equal(
                checkpoint_state["data_order_generator_state"],
                F.stream(seed, "data_order", CHECKPOINT_EPOCH).get_state(),
            ):
        _die(f"checkpoint scheduler/data-order state differs for seed {seed}")
    param_groups = checkpoint_state["optimizer_state_dict"]["param_groups"]
    if len(param_groups) != 1 or float(param_groups[0].get("lr", math.nan)) != LR \
            or float(param_groups[0].get("weight_decay", math.nan)) != WEIGHT_DECAY \
            or param_groups[0].get("foreach") is not False:
        _die(f"checkpoint optimizer hyperparameters differ for seed {seed}")
    tensors = list(checkpoint_state["model_state_dict"].values()) + [
        value for state in checkpoint_state["optimizer_state_dict"]["state"].values()
        for value in state.values() if torch.is_tensor(value)
    ]
    if not tensors or not all(bool(torch.isfinite(value).all()) for value in tensors):
        _die(f"checkpoint model/optimizer tensor finiteness differs for seed {seed}")
    cpu_model, _, _ = _fresh_model(seed, torch.device("cpu"))
    cpu_optimizer = torch.optim.AdamW(
        cpu_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False
    )
    cpu_model.load_state_dict(checkpoint_state["model_state_dict"], strict=True)
    cpu_optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    if _optimizer_state_digest(cpu_optimizer) != receipt.get(
        "retained_epoch21_optimizer_digest"
    ):
        _die(f"strict checkpoint optimizer reload differs for seed {seed}")
    del cpu_model, cpu_optimizer
    if receipt.get("final_epoch23_digests_are_execution_receipt_only") is not True \
            or not isinstance(receipt.get("wall_seconds"), (int, float)) \
            or not math.isfinite(float(receipt["wall_seconds"])) \
            or float(receipt["wall_seconds"]) < 0.0:
        _die(f"training final/runtime diagnostic custody differs for seed {seed}")
    if receipt.get("seed_namespace_expected_files") != [
        checkpoint.name, ledger.name, path.name,
    ] or {item.name for item in checkpoint.parent.iterdir()} != set(
        receipt["seed_namespace_expected_files"]
    ):
        _die(f"training seed namespace inventory differs for seed {seed}")
    return receipt


def train_seed_stage(args: argparse.Namespace) -> dict[str, Any]:
    contract = require_contract()
    environment_record(require_exact=True)
    if args.seed is None or int(args.seed) not in C.FROZEN_SEEDS:
        _die("train-seed requires --seed from the frozen eight")
    seed = int(args.seed)
    receipt_path = _seed_receipt_path(seed)
    if receipt_path.is_file():
        return validate_training_receipt(seed)
    preflight = validate_preflight_receipt()
    if preflight.get("valid") is not True:
        _die("passing resource preflight is required before training")
    _start_once(f"train_seed_{seed}", receipt_path)
    validate_training_input_files()
    loader = FourStepLoader()
    device = resolve_device(args.device)
    if device.type != "cuda":
        _die("scientific training must run on the registered R9700")
    model, base_path, base = _fresh_model(seed, device)
    initial_digest = _model_state_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False
    )
    started = time.time()
    history: list[dict[str, Any]] = []
    checkpoint_receipt: dict[str, Any] | None = None
    for epoch in range(EPOCHS):
        result = _train_epoch(model, optimizer, loader, seed, epoch, device)
        history.append(result)
        print(json.dumps({"seed": seed, **result}, sort_keys=True), flush=True)
        if epoch == CHECKPOINT_EPOCH:
            checkpoint = _seed_checkpoint_path(seed)
            checkpoint_receipt = CK.save(
                checkpoint, model=model, optimizer=optimizer, epoch=epoch,
                global_step=sum(item["batches"] for item in history), seed=seed,
                model_config={
                    "cell": "rgb_four_step", "use_proprio": False,
                    "objective": "(L1+L2+L3+L4)/4", "width": WIDTH,
                    "depth": DEPTH, "heads": HEADS,
                },
                scheduler=None,
                scheduler_absent_reason="fixed learning rate; no scheduler is constructed",
                data_order_generator=F.stream(seed, "data_order", epoch),
                extra={
                    "history": history,
                    "common_rows_digest": validate_common_manifest()["common_rows_digest"],
                    "base_state_digest": base["state_digest"],
                    "target_cache_index_digest": validate_target_cache_index()[
                        "target_cache_index_digest"
                    ],
                },
            )
    if checkpoint_receipt is None:
        _die("fixed epoch-21 checkpoint was not written")
    if not _all_finite_model_optimizer(model, optimizer):
        _die("final model or optimizer state is non-finite")
    final_state_digest = _model_state_digest(model)
    final_optimizer_digest = _optimizer_state_digest(optimizer)
    terminal = F.terminal_window(history)
    checkpoint_path = _seed_checkpoint_path(seed)
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint_state.get("epoch") != CHECKPOINT_EPOCH \
            or checkpoint_state.get("seed") != seed:
        _die("checkpoint reload identity differs")
    if checkpoint_state.get("common_rows_digest") != validate_common_manifest()[
        "common_rows_digest"
    ] or checkpoint_state.get("target_cache_index_digest") != validate_target_cache_index()[
        "target_cache_index_digest"
    ] or checkpoint_state.get("base_state_digest") != C.BASE_STATE_DIGEST[seed]:
        _die("checkpoint scientific bindings differ")
    retained_state_digest = F.state_digest(checkpoint_state["model_state_dict"])
    retained_model, _, _ = _fresh_model(seed, torch.device("cpu"))
    retained_optimizer = torch.optim.AdamW(
        retained_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False
    )
    CK.load_for_resume(
        checkpoint_path, model=retained_model, optimizer=retained_optimizer,
        data_order_generator=F.stream(seed, "data_order", CHECKPOINT_EPOCH),
    )
    strict_reloaded_state_digest = _model_state_digest(retained_model)
    retained_optimizer_digest = _optimizer_state_digest(retained_optimizer)
    if strict_reloaded_state_digest != retained_state_digest:
        _die("strict epoch-21 model reload differs")
    checkpoint_receipts_path = checkpoint_path.parent / "checkpoint_receipts.jsonl"
    checkpoint_receipts = _read_jsonl(checkpoint_receipts_path)
    if len(checkpoint_receipts) != 1 \
            or checkpoint_receipts[0].get("sha256") != checkpoint_receipt["sha256"]:
        _die("checkpoint receipt ledger differs")
    _make_read_only(checkpoint_path)
    _make_read_only(checkpoint_receipts_path)
    receipt = {
        "schema": "go2_rgb_control_history_four_step_training_receipt_v1",
        "status": STATUS, "complete": True, "valid": True,
        "four_step_contract_digest": _contract_digest(contract),
        "seed": seed, "epochs_trained": EPOCHS,
        "checkpoint_epoch": CHECKPOINT_EPOCH, "best_epoch_selected": False,
        "base_weights_path": str(base_path),
        "base_weights_sha256": C.BASE_WEIGHT_SHA256[seed],
        "base_state_digest": C.BASE_STATE_DIGEST[seed],
        "initial_state_digest": initial_digest,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_receipt["sha256"],
        "checkpoint_bytes": checkpoint_receipt["bytes"],
        "checkpoint_verified_reloadable": True,
        "retained_epoch21_state_digest": retained_state_digest,
        "retained_epoch21_optimizer_digest": retained_optimizer_digest,
        "retained_epoch21_strict_reload_state_digest": strict_reloaded_state_digest,
        "retained_checkpoint_receipt_ledger": str(checkpoint_receipts_path),
        "retained_checkpoint_receipt_ledger_sha256":
            _sha256_file(checkpoint_receipts_path),
        "final_epoch23_state_digest": final_state_digest,
        "final_epoch23_optimizer_digest": final_optimizer_digest,
        "final_epoch23_digests_are_execution_receipt_only": True,
        "history": history, "terminal_window": terminal,
        "wall_seconds": time.time() - started,
        "finite_weak_run_retained": True,
        "extension_or_retry": False,
        "seed_namespace_expected_files": [
            checkpoint_path.name, checkpoint_receipts_path.name,
            _seed_receipt_path(seed).name,
        ],
    }
    receipt["training_receipt_digest"] = _digest(receipt)
    _write_json_once(receipt_path, receipt)
    del checkpoint_state, retained_model, retained_optimizer, model, optimizer
    torch.cuda.empty_cache()
    return validate_training_receipt(seed)


def train_all_stage(args: argparse.Namespace) -> dict[str, Any]:
    contract = require_contract()
    aggregate_path = runtime_root() / "training_receipts.json"
    if aggregate_path.is_file():
        return validate_training_receipt_set()
    receipts = []
    for seed in C.FROZEN_SEEDS:
        child = argparse.Namespace(**vars(args))
        child.seed = int(seed)
        receipts.append(train_seed_stage(child))
    aggregate = {
        "schema": "go2_rgb_control_history_four_step_training_receipt_set_v1",
        "status": STATUS, "complete": True, "valid": True,
        "four_step_contract_digest": _contract_digest(contract),
        "seed_order": list(C.FROZEN_SEEDS),
        "receipt_digests": [row["training_receipt_digest"] for row in receipts],
        "checkpoint_sha256": {str(row["seed"]): row["checkpoint_sha256"]
                              for row in receipts},
        "runs": len(receipts), "epochs_each": EPOCHS,
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "total_wall_seconds": sum(float(row["wall_seconds"]) for row in receipts),
    }
    aggregate["training_receipt_set_digest"] = _digest(aggregate)
    _write_json_once(aggregate_path, aggregate)
    return validate_training_receipt_set()


def validate_smoke_receipt() -> dict[str, Any]:
    path = runtime_root() / "smoke.json"
    _require_immutable_file(path, "four-step smoke receipt")
    receipt = _read_json(path)
    if receipt.get("smoke_digest") != _digest({
        key: value for key, value in receipt.items() if key != "smoke_digest"
    }):
        _die("smoke receipt self digest differs")
    if receipt.get("schema") != "go2_rgb_control_history_four_step_smoke_v1" \
            or receipt.get("status") != STATUS or receipt.get("complete") is not True \
            or receipt.get("valid") is not True \
            or receipt.get("four_step_contract_digest") != _contract_digest(
                require_contract()
            ):
        _die("smoke receipt schema/completion differs")
    separation = receipt.get("component_perturbation_separation", {})
    checkpoint = receipt.get("checkpoint_save_resume", {})
    objective = receipt.get("objective_separation", {})
    exact_matrix = [[row == column for column in range(4)] for row in range(4)]
    if separation.get("only_registered_component_changes") is not True \
            or separation.get("all_derivatives_exactly_one_quarter") is not True \
            or separation.get("changed_component_matrix") != exact_matrix \
            or separation.get("expected_changed_component_matrix") != exact_matrix \
            or separation.get("combined_loss_derivative_per_component") != [0.25] * 4 \
            or objective.get("exact_equal_weight_formula") is not True \
            or checkpoint.get("exact_resume") is not True \
            or checkpoint.get("next_batch_update_state_equal") is not True:
        _die("smoke objective/checkpoint gate differs")
    components = objective.get("component_losses_H1_H4", [])
    if len(components) != 4 or not all(math.isfinite(float(x)) for x in components) \
            or not math.isclose(float(objective.get("combined", math.nan)),
                                float(np.mean(components)), abs_tol=1e-7, rel_tol=0.0):
        _die("smoke mean objective reduction differs")
    for horizon in (3, 4):
        probe = receipt.get(f"H{horizon}_chain_probe", {})
        if probe.get("all_preceding_finite_nonzero") is not True \
                or probe.get("all_parameter_gradients_present_finite") is not True:
            _die(f"smoke H{horizon} autoregressive chain gate differs")
    seed = int(C.FROZEN_SEEDS[0])
    if receipt.get("warmup_state_discarded") is not True \
            or receipt.get("registered_base_reloaded_digest") != C.BASE_STATE_DIGEST[seed] \
            or receipt.get("scientific_optimizer_step_performed") is not False \
            or receipt.get("calibration_or_counterfactual_corpus_opened") is not False:
        _die("smoke discard/custody gate differs")
    return receipt


def validate_preflight_receipt() -> dict[str, Any]:
    path = runtime_root() / "resource_preflight.json"
    _require_immutable_file(path, "four-step resource preflight")
    receipt = _read_json(path)
    if receipt.get("preflight_digest") != _digest({
        key: value for key, value in receipt.items() if key != "preflight_digest"
    }):
        _die("resource preflight self digest differs")
    if receipt.get("schema") != (
        "go2_rgb_control_history_four_step_resource_preflight_v1"
    ) or receipt.get("status") != STATUS or receipt.get("complete") is not True \
            or receipt.get("valid") is not True \
            or receipt.get("full_epochs_measured") != 1 \
            or receipt.get("batch_size") != BATCH \
            or receipt.get("preflight_weights_discarded") is not True \
            or receipt.get("registered_base_reloaded") is not True \
            or receipt.get("scientific_epoch_completed") is not False:
        _die("resource preflight schema/gate differs")
    gate = resource_gate(
        peak_vram_bytes=int(receipt["peak_vram_reserved_bytes"]),
        minimum_mem_available_bytes=int(receipt["minimum_mem_available_bytes"]),
        filesystem_free_bytes=int(receipt["destination_filesystem_free_bytes"]),
        projected_remaining_bytes=int(receipt["projected_remaining_bytes"]),
    )
    if receipt.get("gate") != gate or gate["pass"] is not True:
        _die("resource preflight gate recomputation differs")
    manifest = validate_common_manifest()
    seed = int(C.FROZEN_SEEDS[0])
    plan = common_plan_from_rows(seed, 0, manifest["rows"])
    epoch = receipt.get("epoch", {})
    components = [epoch.get(key) for key in ("e1", "e2", "e3", "e4")]
    if epoch.get("epoch") != 0 or epoch.get("batches") != len(plan) \
            or len(plan) != 964 or epoch.get("batch_plan_digest") != _digest(plan) \
            or not all(isinstance(value, (int, float))
                       and math.isfinite(float(value)) for value in components) \
            or not math.isclose(float(epoch.get("loss", math.nan)),
                                float(np.mean(components)), abs_tol=1e-6, rel_tol=1e-6):
        _die("resource preflight measured-epoch accounting differs")
    wall = float(receipt["wall_seconds_per_epoch"])
    if wall != float(epoch["wall_seconds"]) \
            or float(receipt["projected_eight_run_wall_seconds"]) != (
                wall * EPOCHS * len(C.FROZEN_SEEDS)
            ) or receipt.get("target_cache_storage_bytes") != (
                C.TARGET_CACHE_CONTRACT["missing_dense_cache_bytes_total"]
            ):
        _die("resource preflight runtime/target-storage projection differs")
    comparator_sizes = [
        (F.OUT / f"seed_{seed_value}" / (
            f"seed_{seed_value}_{cell}_epoch21.pt"
        )).stat().st_size
        for seed_value in C.FROZEN_SEEDS
        for cell in ("rgb_one_step", "rgb_rollout")
    ]
    checkpoint_projection = max(comparator_sizes) * len(C.FROZEN_SEEDS)
    expected_remaining = checkpoint_projection + 1 * 2**30 + 64 * 2**20
    if receipt.get("projected_eight_epoch21_checkpoint_bytes") != checkpoint_projection \
            or receipt.get("projected_evaluation_and_receipt_bytes") != (
                1 * 2**30 + 64 * 2**20
            ) or receipt.get("projected_remaining_bytes") != expected_remaining:
        _die("resource preflight checkpoint/evaluation storage projection differs")
    return receipt


def validate_training_receipt_set() -> dict[str, Any]:
    path = runtime_root() / "training_receipts.json"
    _require_immutable_file(path, "eight-seed training receipt set")
    aggregate = _read_json(path)
    if aggregate.get("training_receipt_set_digest") != _digest({
        key: value for key, value in aggregate.items()
        if key != "training_receipt_set_digest"
    }):
        _die("training receipt-set self digest differs")
    receipts = [validate_training_receipt(int(seed)) for seed in C.FROZEN_SEEDS]
    expected = {
        "seed_order": [int(seed) for seed in C.FROZEN_SEEDS],
        "receipt_digests": [row["training_receipt_digest"] for row in receipts],
        "checkpoint_sha256": {
            str(row["seed"]): row["checkpoint_sha256"] for row in receipts
        },
        "runs": 8, "epochs_each": EPOCHS,
        "checkpoint_epoch": CHECKPOINT_EPOCH,
    }
    if aggregate.get("schema") != (
        "go2_rgb_control_history_four_step_training_receipt_set_v1"
    ) or aggregate.get("status") != STATUS or aggregate.get("complete") is not True \
            or aggregate.get("valid") is not True:
        _die("training receipt-set schema/completion differs")
    for key, value in expected.items():
        if aggregate.get(key) != value:
            _die(f"training receipt-set {key} differs")
    expected_wall = sum(float(row["wall_seconds"]) for row in receipts)
    if float(aggregate.get("total_wall_seconds", -1)) != expected_wall:
        _die("training receipt-set runtime aggregation differs")
    return aggregate


def t_interval(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 8 or not all(math.isfinite(float(value)) for value in values):
        _die("paired t interval requires eight finite seed values")
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    critical = 2.3646242510102993
    half = critical * sd / math.sqrt(8)
    return {
        "values": [float(value) for value in array], "n": 8,
        "mean": mean, "sample_standard_deviation": sd,
        "t_critical_df7": critical,
        "two_sided_95_t_interval": [mean - half, mean + half],
    }


def paired_effect_summary(
    one_step: Sequence[float], two_step: Sequence[float],
    four_step: Sequence[float], *, higher_is_better: bool,
) -> dict[str, Any]:
    if not (len(one_step) == len(two_step) == len(four_step) == 8):
        _die("paired effect needs one/two/four values for eight seeds")
    sign = 1.0 if higher_is_better else -1.0
    effect_4_2 = [sign * (float(four) - float(two))
                  for four, two in zip(four_step, two_step, strict=True)]
    effect_4_1 = [sign * (float(four) - float(one))
                  for four, one in zip(four_step, one_step, strict=True)]
    summary_4_2 = t_interval(effect_4_2)
    summary_4_1 = t_interval(effect_4_1)
    summary_4_2["eight_seed_effects"] = [
        {"seed": int(seed), "effect": float(effect)}
        for seed, effect in zip(C.FROZEN_SEEDS, effect_4_2, strict=True)
    ]
    summary_4_1["eight_seed_effects"] = [
        {"seed": int(seed), "effect": float(effect)}
        for seed, effect in zip(C.FROZEN_SEEDS, effect_4_1, strict=True)
    ]
    return {
        "benefit_orientation": (
            "four_step - comparator" if higher_is_better
            else "comparator - four_step"
        ),
        "cell_means": {
            "one_step": t_interval(one_step),
            "two_step": t_interval(two_step),
            "four_step": t_interval(four_step),
        },
        "four_step_minus_two_step_benefit": summary_4_2,
        "four_step_minus_one_step_benefit": summary_4_1,
    }


EFFECT_METRICS: dict[str, tuple[str, bool]] = {
    "changed_token_correct_future_cosine": ("changed_cosine", True),
    "normalized_error_reduction": ("normalised_error_vs_persistence", False),
    "advantage_over_persistence": ("advantage_over_persistence", True),
    "full_token_cosine": ("full_token_cosine", True),
    "correct_branch_top1_retrieval": ("retrieval_top1", True),
    "top3_retrieval": ("retrieval_top3", True),
    "mean_reciprocal_rank": ("retrieval_mean_reciprocal_rank", True),
    "mean_rank_reduction": ("retrieval_mean_rank", False),
    "pairwise_branch_discrimination": ("retrieval_pairwise_accuracy", True),
    "own_vs_best_other_margin": ("retrieval_mean_margin_over_best_wrong", True),
    "own_vs_mean_other_margin": ("retrieval_mean_margin_over_mean_wrong", True),
}


def _metric_value(
    aggregate: dict[str, Any], horizon: int, weighting: str,
    metric: str, family: str | None = None,
) -> float:
    block = aggregate["per_horizon"][str(horizon)]
    source = block["per_family"][family] if family is not None else block[weighting]
    if metric.startswith("retrieval_"):
        value = source["retrieval"][metric.removeprefix("retrieval_")]
    else:
        corpus_alias = {
            "changed_cosine": "token_pooled_changed_cosine",
            "normalised_error_vs_persistence":
                "token_pooled_normalised_error_vs_persistence",
            "advantage_over_persistence": "token_pooled_advantage_over_persistence",
        }
        key = corpus_alias.get(metric, metric) \
            if weighting == "corpus_weighted" and family is None else metric
        value = source["direct"][key]
    if value is None or not math.isfinite(float(value)):
        _die(f"metric {metric} is unavailable at H{horizon}/{weighting}/{family}")
    return float(value)


def _effect_table(
    cells: dict[int, dict[str, dict[str, Any]]], horizon: int,
    weighting: str, family: str | None = None,
) -> dict[str, Any]:
    result = {}
    for endpoint, (metric, higher) in EFFECT_METRICS.items():
        vectors = {
            cell: [_metric_value(cells[int(seed)][cell], horizon, weighting,
                                 metric, family)
                   for seed in C.FROZEN_SEEDS]
            for cell in ("one_step", "two_step", "four_step")
        }
        result[endpoint] = paired_effect_summary(
            vectors["one_step"], vectors["two_step"], vectors["four_step"],
            higher_is_better=higher,
        )
    return result


def occupancy_horizons(qualified: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in qualified)
    if values != (2, 3, 4):
        _die(f"frozen occupancy horizon gate differs: {values}")
    return values


def interpretation_from_effects(analysis: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen useful/trade-off rules without a post-result choice."""
    h4 = analysis["equal_family"]["H4"]

    def effect(endpoint: str, horizon: str = "H4") -> dict[str, Any]:
        return analysis["equal_family"][horizon][endpoint][
            "four_step_minus_two_step_benefit"
        ]

    cosine = effect("changed_token_correct_future_cosine")
    error_reduction = effect("normalized_error_reduction")
    top1 = effect("correct_branch_top1_retrieval")
    pairwise = effect("pairwise_branch_discrimination")
    direct_improved = cosine["mean"] > 0.0 and error_reduction["mean"] > 0.0
    retrieval_improved = top1["mean"] > 0.0 or pairwise["mean"] > 0.0
    useful = direct_improved and retrieval_improved
    material_regressions: list[dict[str, Any]] = []
    for horizon in ("H1", "H2"):
        for endpoint in (
            "changed_token_correct_future_cosine", "normalized_error_reduction",
        ):
            summary = effect(endpoint, horizon)
            upper = float(summary["two_sided_95_t_interval"][1])
            if upper < 0.0:
                material_regressions.append({
                    "horizon": horizon,
                    "endpoint": endpoint,
                    "criterion": "paired 95% t-interval wholly below zero",
                    "effect": summary,
                })
    if useful and material_regressions:
        classification = C.INTERPRETATION[
            "H4_improves_with_material_H1_H2_regression"
        ]
    elif useful:
        classification = "USEFUL_FOUR_STEP_PREDICTIVE_DYNAMICS_RESULT"
    elif direct_improved and not retrieval_improved:
        classification = C.INTERPRETATION["direct_improves_retrieval_does_not"]
    elif cosine["mean"] > 0.0 or error_reduction["mean"] > 0.0:
        classification = "DIRECT_FIDELITY_EVIDENCE_DISCORDANT_OR_MIXED"
    else:
        classification = "NO_FOUR_STEP_DIRECT_FIDELITY_IMPROVEMENT"
    return {
        "classification": classification,
        "useful": useful,
        "direct_fidelity_improved": direct_improved,
        "direct_fidelity_rule": (
            "both H4 equal-family changed-token cosine and normalized-error "
            "reduction paired mean benefits are positive"
        ),
        "retrieval_improved": retrieval_improved,
        "retrieval_rule": (
            "H4 equal-family top-1 or pairwise paired mean benefit is positive"
        ),
        "H4_changed_cosine_effect": cosine["mean"],
        "H4_normalized_error_reduction_effect": error_reduction["mean"],
        "H4_top1_effect": top1["mean"],
        "H4_pairwise_effect": pairwise["mean"],
        "H1_H2_material_regression_rule": (
            "paired equal-family 95% t-interval wholly below zero for either "
            "direct-fidelity endpoint"
        ),
        "H1_H2_material_regressions": material_regressions,
        "horizon_tradeoff": bool(material_regressions),
        "planning_or_utility_claim": False,
    }


def _retrieval_confusion(
    cells: dict[int, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        key = f"H{horizon}"
        result[key] = {}
        for cell in ("one_step", "two_step", "four_step"):
            first = cells[int(C.FROZEN_SEEDS[0])][cell]["per_horizon"][
                str(horizon)
            ]["corpus_weighted"]["retrieval"]
            overall = np.sum([
                np.asarray(
                    cells[int(seed)][cell]["per_horizon"][str(horizon)][
                        "corpus_weighted"
                    ]["retrieval"]["confusion"],
                    dtype=np.int64,
                )
                for seed in C.FROZEN_SEEDS
            ], axis=0)
            per_family = {
                family: np.sum([
                    np.asarray(
                        cells[int(seed)][cell]["per_horizon"][str(horizon)][
                            "per_family"
                        ][family]["retrieval"]["confusion"],
                        dtype=np.int64,
                    )
                    for seed in C.FROZEN_SEEDS
                ], axis=0).tolist()
                for family in C.FAMILIES
            }
            result[key][cell] = {
                "candidate_order": first["candidate_order"],
                "overall": overall.tolist(),
                "per_family": per_family,
                "seeds": [int(seed) for seed in C.FROZEN_SEEDS],
            }
    return result


def _occupancy_analysis(
    occupancy_by_seed: dict[int, dict[int, dict[str, Any]]],
    gate: dict[str, Any], provenance: dict[str, Any], probe_digest: str,
) -> dict[str, Any]:
    """Pair streamed four-step occupancy values with frozen RGB controls."""
    from scripts import run_go2_counterfactual_occupancy_assay_v1_2 as O

    frozen_path = Path(C.OCCUPANCY["result_path"])
    if (not frozen_path.is_file()
            or frozen_path.stat().st_size != C.OCCUPANCY["frozen_result_bytes"]
            or _sha256_file(frozen_path)
            != C.OCCUPANCY["frozen_result_file_sha256"]):
        _die("frozen occupancy result bytes differ")
    frozen = _read_json(frozen_path)
    O.verify_self_digest(frozen, "report_digest", "frozen occupancy result")
    if frozen["report_digest"] != C.OCCUPANCY["frozen_result_digest"]:
        _die("frozen occupancy result digest differs")
    if gate.get("true_target_gate_digest") != C.OCCUPANCY[
        "true_target_gate_digest"
    ]:
        _die("frozen occupancy true-target gate digest differs")
    if probe_digest != C.OCCUPANCY["probe_state_digest"]:
        _die("frozen occupancy probe state differs")
    if set(occupancy_by_seed) != {int(seed) for seed in C.FROZEN_SEEDS}:
        _die("new occupancy estimates do not contain all registered seeds")

    estimator_names = {
        "equal_family": "primary_equal_family",
        "corpus_weighted": "secondary_corpus_weighted",
        "whole_pilot_observable_occupied_iou":
            "whole_pilot_pooled_diagnostic",
    }
    horizons: dict[str, Any] = {
        "1": {
            "horizon": 1, "available": False,
            "predictor_latents_scored": False,
            "reason": "frozen true-target occupancy probe did not qualify at H1",
            "H1_not_reinterpreted": True,
        }
    }
    for horizon in occupancy_horizons(gate["qualified_horizons"]):
        frozen_h = frozen["horizons"][str(horizon)]
        reports: dict[str, Any] = {}
        for source_key, frozen_key in estimator_names.items():
            one = frozen_h[frozen_key]["four_cells"]["rgb_one_step"][
                "predicted"
            ]["values"]
            two = frozen_h[frozen_key]["four_cells"]["rgb_rollout"][
                "predicted"
            ]["values"]
            four = [
                float(occupancy_by_seed[int(seed)][horizon][source_key])
                for seed in C.FROZEN_SEEDS
            ]
            reports[frozen_key] = paired_effect_summary(
                one, two, four, higher_is_better=True
            )
            reports[frozen_key]["true_target"] = frozen_h[frozen_key][
                "true_target"
            ]
        per_family: dict[str, Any] = {}
        for family in C.FAMILIES:
            old = frozen_h["per_family"][family]
            one = old["four_cells"]["rgb_one_step"]["predicted"]["values"]
            two = old["four_cells"]["rgb_rollout"]["predicted"]["values"]
            four = [
                occupancy_by_seed[int(seed)][horizon]["per_family"][family]
                for seed in C.FROZEN_SEEDS
            ]
            if not all(value is not None and math.isfinite(float(value))
                       for value in (*one, *two, *four)):
                per_family[family] = {
                    "available": False,
                    "reason": "one or more frozen per-family seed estimates undefined",
                    "one_step": one, "two_step": two, "four_step": four,
                }
            else:
                per_family[family] = {
                    "available": True,
                    **paired_effect_summary(one, two, four, higher_is_better=True),
                    "true_target": old["true_target"],
                }
        horizons[str(horizon)] = {
            "horizon": horizon, "available": True,
            "predictor_latents_scored": True,
            **reports, "per_family": per_family,
        }
    result = {
        "schema": "go2_rgb_control_history_four_step_occupancy_co_outcome_v1",
        "status": STATUS, "complete": True, "valid": True,
        "four_step_contract_digest": _contract_digest(require_contract()),
        "claim_bearing": False,
        "co_outcome_not_formal_non_regression": True,
        "qualified_true_target_horizons": [2, 3, 4],
        "H1_unavailable_and_not_reinterpreted": True,
        "probe_refit": False,
        "probe_package_digest": C.OCCUPANCY["probe_package_digest"],
        "probe_state_digest": probe_digest,
        "probe_provenance": provenance,
        "true_target_gate_digest": gate["true_target_gate_digest"],
        "frozen_control_result_digest": frozen["report_digest"],
        "horizons": horizons,
    }
    result["occupancy_digest"] = _digest(result)
    return result


def _score_occupancy_state(
    prediction: np.ndarray, state_id: str, bundle: Any, labels: Any,
    probe: nn.Module, device: torch.device, module: Any,
) -> list[dict[str, Any]]:
    rows = sorted(
        (row for row in bundle.rows if str(row["state_id"]) == state_id),
        key=lambda row: int(row["candidate_index"]),
    )
    if len(rows) != 12:
        _die(f"occupancy row set for {state_id} is not twelve candidates")
    targets = np.stack([
        module._load_label_array(labels, module._row_key(row)) for row in rows
    ], axis=0)
    predictions: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for horizon in (2, 3, 4):
            tokens = torch.from_numpy(
                np.asarray(prediction[:, horizon - 1], dtype=np.float32)
            ).to(device)
            predictions[horizon] = (
                probe(tokens, module.TOKEN_GRID).argmax(1).cpu().numpy().astype(np.uint8)
            )
    branches = []
    for index, row in enumerate(rows):
        branches.append({
            "branch_identity_digest": module._row_key(row),
            "candidate": row["candidate"],
            "candidate_index": int(row["candidate_index"]),
            "horizons": [
                {"horizon": horizon, **module.occupied_counts(
                    predictions[horizon][index], targets[index, horizon - 1]
                )}
                for horizon in (2, 3, 4)
            ],
        })
    return branches


def _load_eval_prefix(
    path: Path, seed: int, checkpoint_sha256: str, states: Sequence[Any]
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_jsonl(path)
    if len(rows) > len(states):
        _die(f"evaluation ledger is too long for seed {seed}")
    for index, row in enumerate(rows):
        recorded = row.get("evaluation_state_digest")
        if recorded != _digest({key: value for key, value in row.items()
                                if key != "evaluation_state_digest"}):
            _die(f"evaluation ledger digest differs for seed {seed} state {index}")
        if row.get("seed") != seed or row.get("checkpoint_sha256") != checkpoint_sha256 \
                or int(row.get("state_index", -1)) != index \
                or row.get("state_id") != states[index].state_id:
            _die(f"evaluation ledger prefix binding differs for seed {seed}")
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_epoch21_model(seed: int, device: torch.device) -> nn.Module:
    receipt = validate_training_receipt(seed)
    model, _, _ = _fresh_model(seed, device)
    state = torch.load(_seed_checkpoint_path(seed), map_location="cpu", weights_only=False)
    if state.get("schema") != CK.SCHEMA or state.get("epoch") != CHECKPOINT_EPOCH \
            or state.get("seed") != seed:
        _die(f"new epoch-21 checkpoint identity differs for seed {seed}")
    model.load_state_dict(state["model_state_dict"], strict=True)
    if receipt["checkpoint_sha256"] != _sha256_file(_seed_checkpoint_path(seed)):
        _die(f"new checkpoint changed before inference for seed {seed}")
    model.to(device).eval()
    return model


def _require_finite_json(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _die(f"{label} contains a non-finite floating value")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, f"{label}[{index}]")


def validate_evaluation_result() -> dict[str, Any]:
    """Validate persisted metric evidence only; never load or execute a model."""
    contract = require_contract()
    from scripts import analyze_go2_counterfactual_predictor_qualification_v1_2 as Q

    frozen_result_path = Path(C.FROZEN_EVALUATION["predictor_result_path"])
    if (not frozen_result_path.is_file()
            or frozen_result_path.stat().st_size != C.FROZEN_EVALUATION[
                "frozen_predictor_result_bytes"
            ] or _sha256_file(frozen_result_path) != C.FROZEN_EVALUATION[
                "frozen_predictor_result_file_sha256"
            ]):
        _die("frozen predictor result bytes differ during validation")
    frozen_result = _read_json(frozen_result_path)
    Q.verify_embedded_digest(frozen_result, "report_digest", "frozen predictor result")
    if frozen_result["report_digest"] != C.FROZEN_EVALUATION[
        "frozen_predictor_result_digest"
    ]:
        _die("frozen predictor result digest differs during validation")
    stage_a = Q.validate_stage_a_metadata()
    result_path = runtime_root() / "evaluation" / "result.json"
    occupancy_path = runtime_root() / "evaluation" / "occupancy.json"
    _require_immutable_file(result_path, "four-step evaluation result")
    _require_immutable_file(occupancy_path, "four-step occupancy co-outcome")
    report = _read_json(result_path)
    occupancy = _read_json(occupancy_path)
    _require_finite_json(report, "evaluation result")
    _require_finite_json(occupancy, "occupancy result")
    if report.get("result_digest") != _digest({
        key: value for key, value in report.items() if key != "result_digest"
    }):
        _die("evaluation result self digest differs")
    if occupancy.get("occupancy_digest") != _digest({
        key: value for key, value in occupancy.items() if key != "occupancy_digest"
    }):
        _die("occupancy result self digest differs")
    if report.get("schema") != (
        "go2_rgb_control_history_four_step_evaluation_result_v1"
    ) or report.get("status") != STATUS or report.get("complete") is not True \
            or report.get("valid") is not True \
            or report.get("four_step_contract_digest") != _contract_digest(contract):
        _die("evaluation result schema/completion differs")
    if report.get("environment") != environment_record(require_exact=True):
        _die("evaluation environment receipt differs from the registered runtime")
    manifest = validate_common_manifest()
    cache_index = validate_target_cache_index()
    training_set_path = runtime_root() / "training_receipts.json"
    _require_immutable_file(training_set_path, "training receipt set")
    training_set = _read_json(training_set_path)
    if report.get("common_manifest_digest") != manifest["common_rows_digest"] \
            or report.get("target_cache_index_digest") != cache_index[
                "target_cache_index_digest"
            ] or report.get("training_receipt_set_digest") != training_set.get(
                "training_receipt_set_digest"
            ):
        _die("evaluation manifest/cache/training lineage differs")
    if occupancy.get("schema") != (
        "go2_rgb_control_history_four_step_occupancy_co_outcome_v1"
    ) or occupancy.get("complete") is not True or occupancy.get("valid") is not True \
            or occupancy.get("four_step_contract_digest") != _contract_digest(contract) \
            or occupancy.get("claim_bearing") is not False \
            or occupancy.get("co_outcome_not_formal_non_regression") is not True \
            or occupancy.get("H1_unavailable_and_not_reinterpreted") is not True:
        _die("occupancy co-outcome schema/custody differs")
    if report.get("occupancy_co_outcome") != occupancy:
        _die("evaluation embeds a different occupancy co-outcome")
    frozen = report.get("frozen_stage_a", {})
    if frozen.get("states") != 20 or frozen.get("branches") != 240 \
            or frozen.get("identity_manifest_digest") != C.FROZEN_EVALUATION[
                "stage_a_identity_manifest_digest"
            ] or frozen.get("identity_manifest_digest") != stage_a.identity_digest \
            or frozen.get("corpus_digest") != C.FROZEN_EVALUATION["corpus_digest"] \
            or frozen.get("corpus_digest") != stage_a.corpus_digest \
            or frozen.get("latent_index_digest") != C.FROZEN_EVALUATION[
                "latent_index_digest"
            ] or frozen.get("latent_index_digest") != stage_a.latent_index_digest \
            or frozen.get("verified_shard_set_digest") != C.FROZEN_EVALUATION[
                "verified_latent_shard_set_digest"
            ] or report.get("frozen_comparator_result_digest") != (
                C.FROZEN_EVALUATION["frozen_predictor_result_digest"]
            ) or report.get("historical_comparator_model_forwards") != 0 \
            or report.get("new_four_step_model_forward_states") != 160 \
            or report.get("new_four_step_autoregressive_horizon_forward_calls") != 640:
        _die("evaluation corpus/forward accounting differs")
    if report.get("no_predictor_utility_shards_opened") is not True \
            or report.get("no_branches_targets_labels_or_masks_regenerated") is not True:
        _die("evaluation custody assertion differs")
    if report.get("historical_control_comparability") != {
        "historical_control_train_rows": 3922,
        "new_four_step_train_rows": 3854,
        "row_difference": 68,
        "row_difference_fraction_of_historical": 68 / 3922,
        "historical_controls_sample_matched": False,
        "historical_controls_retrained_or_reselected": False,
        "disposition": "registered historical controls; sample mismatch retained",
    }:
        _die("evaluation historical-control comparability disclosure differs")

    cells_raw = report.get("cells_by_seed")
    if not isinstance(cells_raw, dict) or set(cells_raw) != {
        str(seed) for seed in C.FROZEN_SEEDS
    }:
        _die("evaluation cell matrix lacks the registered seeds")
    cells = {int(seed): value for seed, value in cells_raw.items()}
    for seed in C.FROZEN_SEEDS:
        if set(cells[int(seed)]) != {"one_step", "two_step", "four_step"}:
            _die(f"evaluation objective cells differ for seed {seed}")
        historical = frozen_result["cells_by_seed"][str(seed)]
        if cells[int(seed)]["one_step"] != historical["rgb_one_step"] \
                or cells[int(seed)]["two_step"] != historical["rgb_rollout"]:
            _die(f"evaluation frozen comparator cells differ for seed {seed}")

    # Recompute every direct/retrieval aggregate and paired endpoint from the
    # persisted cell trees.  This executes no model and makes result publication
    # fail closed on a stale or altered reducer.
    expected_analysis: dict[str, Any] = {
        "equal_family": {}, "corpus_weighted": {}, "per_family": {},
    }
    for horizon in HORIZONS:
        key = f"H{horizon}"
        expected_analysis["equal_family"][key] = _effect_table(
            cells, horizon, "equal_family"
        )
        expected_analysis["corpus_weighted"][key] = _effect_table(
            cells, horizon, "corpus_weighted"
        )
        expected_analysis["per_family"][key] = {
            family: _effect_table(cells, horizon, "per_family", family)
            for family in C.FAMILIES
        }
    expected_analysis["retrieval_confusion_across_seeds"] = _retrieval_confusion(cells)
    if report.get("paired_seed_analysis") != expected_analysis:
        _die("evaluation paired metric tree differs from evidence reduction")
    if report.get("primary_H4_equal_family") != expected_analysis[
        "equal_family"
    ]["H4"]:
        _die("evaluation primary H4 table differs")
    expected_interpretation = interpretation_from_effects(expected_analysis)
    if report.get("interpretation") != expected_interpretation:
        _die("evaluation interpretation differs from frozen rules")

    expected_slopes: dict[str, Any] = {}
    for endpoint, (metric, higher) in EFFECT_METRICS.items():
        values = {cell: [] for cell in ("one_step", "two_step", "four_step")}
        for seed in C.FROZEN_SEEDS:
            for cell in values:
                sequence = [
                    _metric_value(cells[int(seed)][cell], horizon,
                                  "equal_family", metric)
                    for horizon in HORIZONS
                ]
                values[cell].append(float(np.polyfit(HORIZONS, sequence, 1)[0]))
        expected_slopes[endpoint] = paired_effect_summary(
            values["one_step"], values["two_step"], values["four_step"],
            higher_is_better=higher,
        )
    if report.get("degradation_slopes_H1_H4") != expected_slopes:
        _die("evaluation degradation-slope reduction differs")

    receipts = report.get("evaluation_receipts")
    if not isinstance(receipts, list) or len(receipts) != 8:
        _die("evaluation receipt count differs")
    occupancy_by_seed: dict[int, dict[int, dict[str, Any]]] = {}
    from scripts import run_go2_counterfactual_occupancy_assay_v1_2 as O
    occupancy_bundle = O.load_stage_a()
    training_receipts = {
        int(seed): validate_training_receipt(int(seed)) for seed in C.FROZEN_SEEDS
    }
    if report.get("new_checkpoint_set_digest") != _sequence_digest([
        training_receipts[int(seed)]["checkpoint_sha256"] for seed in C.FROZEN_SEEDS
    ]) or report.get("terminal_window_stability") != {
        str(seed): training_receipts[int(seed)]["terminal_window"]
        for seed in C.FROZEN_SEEDS
    } or not isinstance(report.get("runtime_seconds"), (int, float)) \
            or not math.isfinite(float(report["runtime_seconds"])) \
            or float(report["runtime_seconds"]) <= 0.0:
        _die("evaluation checkpoint/terminal-window/runtime binding differs")
    for index, seed_value in enumerate(C.FROZEN_SEEDS):
        seed = int(seed_value)
        receipt = receipts[index]
        if receipt.get("receipt_digest") != _digest({
            key: value for key, value in receipt.items() if key != "receipt_digest"
        }) or receipt.get("schema") != (
            "go2_rgb_control_history_four_step_evaluation_seed_receipt_v1"
        ) or receipt.get("status") != STATUS or receipt.get("complete") is not True \
                or receipt.get("seed") != seed or receipt.get("states") != 20 \
                or receipt.get("branches") != 240 \
                or receipt.get("model_forward_states") != 20 \
                or receipt.get("autoregressive_horizon_forward_calls") != 80 \
                or receipt.get("historical_comparator_model_forwards") != 0 \
                or receipt.get("checkpoint_sha256") != training_receipts[seed][
                    "checkpoint_sha256"
                ]:
            _die(f"evaluation receipt differs for seed {seed}")
        ledger = Path(str(receipt.get("ledger", "")))
        expected_ledger = runtime_root() / "evaluation" / "prediction_ledgers" / (
            f"seed_{seed}.jsonl"
        )
        if ledger != expected_ledger:
            _die(f"evaluation ledger path differs for seed {seed}")
        _require_immutable_file(ledger, f"evaluation ledger seed {seed}")
        if receipt.get("ledger_sha256") != _sha256_file(ledger):
            _die(f"evaluation ledger digest differs for seed {seed}")
        rows = _read_jsonl(ledger)
        if len(rows) != 20 or [row.get("state_index") for row in rows] != list(range(20)) \
                or len({row.get("state_id") for row in rows}) != 20:
            _die(f"evaluation ledger state identities differ for seed {seed}")
        for row_index, row in enumerate(rows):
            state = stage_a.states[row_index]
            if row.get("evaluation_state_digest") != _digest({
                key: value for key, value in row.items()
                if key != "evaluation_state_digest"
            }) or row.get("schema") != (
                "go2_rgb_control_history_four_step_evaluation_state_v1"
            ) or row.get("status") != STATUS or row.get("seed_index") != index \
                    or row.get("seed") != seed \
                    or row.get("checkpoint_sha256") != training_receipts[seed][
                        "checkpoint_sha256"
                    ] or row.get("state_index") != row_index \
                    or row.get("state_id") != state.state_id \
                    or row.get("family") != state.family \
                    or row.get("scene_id") != state.scene_id \
                    or row.get("episode_cluster_id") != state.episode_cluster_id \
                    or row.get("candidate_names") != list(state.candidate_names) \
                    or set(row.get("per_horizon", {})) != {"1", "2", "3", "4"} \
                    or len(row.get("candidate_names", [])) != 12 \
                    or len(row.get("occupancy_branches_H2_H4", [])) != 12 \
                    or row.get("occupancy_H1_scored") is not False:
                _die(f"evaluation state evidence differs for seed {seed}/{row_index}")
            for horizon in HORIZONS:
                block = row["per_horizon"][str(horizon)]
                matrix = block.get("retrieval_similarity_matrix", [])
                if len(block.get("direct", [])) != 12 \
                        or len(matrix) != 12 or any(len(line) != 12 for line in matrix) \
                        or not isinstance(block.get("retrieval"), dict):
                    _die(f"evaluation H{horizon} evidence differs for {seed}/{row_index}")
            occupancy_branches = row["occupancy_branches_H2_H4"]
            if [branch.get("candidate_index") for branch in occupancy_branches] != list(
                range(12)
            ) or any([
                entry.get("horizon") for entry in branch.get("horizons", [])
            ] != [2, 3, 4] for branch in occupancy_branches):
                _die(f"occupancy branch ordering differs for {seed}/{row_index}")
        if receipt.get("state_digest_set") != _sequence_digest([
            row["evaluation_state_digest"] for row in rows
        ]):
            _die(f"evaluation state digest set differs for seed {seed}")
        if Q.aggregate_records(rows) != cells[seed]["four_step"]:
            _die(f"four-step metric cell differs from ledger evidence for seed {seed}")
        flat = [
            {"state_id": row["state_id"], "state_index": row["state_index"],
             "family": row["family"],
             "episode_cluster_id": row["episode_cluster_id"], **branch}
            for row in rows for branch in row["occupancy_branches_H2_H4"]
        ]
        occupancy_by_seed[seed] = {
            horizon: O.aggregate_prediction(flat, horizon, occupancy_bundle)
            for horizon in (2, 3, 4)
        }
    gate = O.read_json(O.RESULT_ROOT / "true_target_gate.json", "occupancy gate")
    O.verify_self_digest(gate, "true_target_gate_digest", "occupancy gate")
    expected_occupancy = _occupancy_analysis(
        occupancy_by_seed, gate, occupancy["probe_provenance"],
        occupancy["probe_state_digest"],
    )
    if occupancy != expected_occupancy:
        _die("occupancy co-outcome differs from persisted count evidence")
    return report


def evaluate_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Infer only the new arm, then pair it with frozen persisted controls."""
    contract = require_contract()
    environment = environment_record(require_exact=True)
    manifest = validate_common_manifest()
    cache_index = validate_target_cache_index()
    result_path = runtime_root() / "evaluation" / "result.json"
    if result_path.is_file():
        return validate_evaluation_result()
    aggregate_training_path = runtime_root() / "training_receipts.json"
    if not aggregate_training_path.is_file():
        _die("eight valid four-step training receipts are required")
    aggregate_training = _read_json(aggregate_training_path)
    if aggregate_training.get("training_receipt_set_digest") != _digest({
        key: value for key, value in aggregate_training.items()
        if key != "training_receipt_set_digest"
    }):
        _die("training receipt set self digest differs before evaluation attempt")
    attempt = runtime_root() / "attempts" / "evaluation.json"
    if attempt.exists() or attempt.is_symlink():
        _die("the sole evaluation attempt was already consumed; resume/retry forbidden")
    _write_json_once(attempt, {
        "schema": "go2_rgb_control_history_four_step_evaluation_attempt_v1",
        "four_step_contract_digest": _contract_digest(contract),
        "common_manifest_digest": manifest["common_rows_digest"],
        "target_cache_index_digest": cache_index["target_cache_index_digest"],
        "training_receipt_set_digest": aggregate_training[
            "training_receipt_set_digest"
        ],
        "started_unix_ns": time.time_ns(),
        "resumable": False, "retry_authorised": False,
        "scientific_unit": "one frozen 240-branch evaluation per checkpoint",
        "historical_comparator_forward_permitted": False,
    })

    from scripts import analyze_go2_counterfactual_predictor_qualification_v1_2 as Q
    from scripts import run_go2_counterfactual_occupancy_assay_v1_2 as O

    frozen_result_path = Path(C.FROZEN_EVALUATION["predictor_result_path"])
    if frozen_result_path.stat().st_size != C.FROZEN_EVALUATION[
        "frozen_predictor_result_bytes"
    ] or _sha256_file(frozen_result_path) != C.FROZEN_EVALUATION[
        "frozen_predictor_result_file_sha256"
    ]:
        _die("frozen one/two-step predictor result bytes differ")
    frozen_result = _read_json(frozen_result_path)
    Q.verify_embedded_digest(frozen_result, "report_digest", "frozen predictor result")
    if frozen_result["report_digest"] != C.FROZEN_EVALUATION[
        "frozen_predictor_result_digest"
    ]:
        _die("frozen one/two-step predictor result digest differs")

    bundle = Q.validate_stage_a_metadata()
    shards = Q.validate_stage_a_latent_shards(bundle)
    if shards["verified_shard_set_digest"] != C.FROZEN_EVALUATION[
        "verified_latent_shard_set_digest"
    ]:
        _die("frozen Stage-A shard set differs")
    _, lineage = Q.verify_frozen_predictor_lineage()
    normalisation = Q.load_frozen_normalisation(lineage["normalisation_sha256"])

    # Hash all eight new checkpoints before torch-loading the first one.
    for seed in C.FROZEN_SEEDS:
        receipt_path = _seed_receipt_path(int(seed))
        _require_immutable_file(receipt_path, f"training receipt seed {seed}")
        raw_receipt = _read_json(receipt_path)
        checkpoint_path = _seed_checkpoint_path(int(seed))
        _require_immutable_file(checkpoint_path, f"epoch-21 checkpoint seed {seed}")
        if _sha256_file(checkpoint_path) != raw_receipt.get("checkpoint_sha256"):
            _die(f"new checkpoint changed before pre-load hash barrier: {seed}")
    new_receipts = [validate_training_receipt(int(seed)) for seed in C.FROZEN_SEEDS]
    aggregate_training = validate_training_receipt_set()
    if aggregate_training.get("valid") is not True or aggregate_training.get("runs") != 8:
        _die("eight valid four-step training receipts are required")
    checkpoint_set_digest = _sequence_digest([
        row["checkpoint_sha256"] for row in new_receipts
    ])
    device = resolve_device(args.device)
    if device.type != "cuda":
        _die("counterfactual evaluation must run on the registered R9700")

    occupancy_bundle = O.load_stage_a()
    labels = O.load_labels(occupancy_bundle)
    gate_path = O.RESULT_ROOT / "true_target_gate.json"
    gate = O.read_json(gate_path, "frozen true-target occupancy gate")
    O.verify_self_digest(gate, "true_target_gate_digest", "occupancy gate")
    qualified = occupancy_horizons(gate["qualified_horizons"])
    provenance = O.validate_probe_package_metadata()
    probe, probe_digest = O.load_probe(device)
    if probe_digest != C.OCCUPANCY["probe_state_digest"]:
        _die("frozen occupancy probe state digest differs")

    cells: dict[int, dict[str, dict[str, Any]]] = {}
    occupancy_by_seed: dict[int, dict[int, dict[str, Any]]] = {}
    evaluation_receipts = []
    scoring_started = time.time()
    for seed_index, seed_value in enumerate(C.FROZEN_SEEDS):
        seed = int(seed_value)
        training = new_receipts[seed_index]
        ledger = runtime_root() / "evaluation" / "prediction_ledgers" / f"seed_{seed}.jsonl"
        if ledger.exists() or ledger.is_symlink():
            _die(f"pre-existing evaluation ledger is forbidden for seed {seed}")
        records: list[dict[str, Any]] = []
        model = _load_epoch21_model(seed, device)
        for state in bundle.states:
            prediction = Q.predict_state(
                model, state, bundle.context_records[state.context_key],
                normalisation, False, device,
            )
            per_horizon = Q.score_state_predictions(bundle, state, prediction, device)
            occupancy = _score_occupancy_state(
                prediction, state.state_id, occupancy_bundle, labels, probe, device, O
            )
            row = {
                "schema": "go2_rgb_control_history_four_step_evaluation_state_v1",
                "status": STATUS, "seed_index": seed_index, "seed": seed,
                "checkpoint_sha256": training["checkpoint_sha256"],
                "state_index": state.state_index, "state_id": state.state_id,
                "family": state.family, "scene_id": state.scene_id,
                "episode_cluster_id": state.episode_cluster_id,
                "candidate_names": list(state.candidate_names),
                "per_horizon": per_horizon,
                "occupancy_branches_H2_H4": occupancy,
                "occupancy_H1_scored": False,
            }
            row["evaluation_state_digest"] = _digest(row)
            _append_jsonl(ledger, row)
            records.append(row)
            print(f"[evaluation] seed {seed}: {len(records)}/20", flush=True)
        if model is not None:
            del model
            torch.cuda.empty_cache()
        if len(records) != 20:
            _die(f"evaluation is incomplete for seed {seed}")
        _make_read_only(ledger)
        four = Q.aggregate_records(records)
        frozen_cells = frozen_result["cells_by_seed"][str(seed)]
        cells[seed] = {
            "one_step": frozen_cells["rgb_one_step"],
            "two_step": frozen_cells["rgb_rollout"],
            "four_step": four,
        }
        flat_occupancy = [
            {"state_id": row["state_id"], "state_index": row["state_index"],
             "family": row["family"], "episode_cluster_id": row["episode_cluster_id"],
             **branch}
            for row in records for branch in row["occupancy_branches_H2_H4"]
        ]
        occupancy_by_seed[seed] = {
            horizon: O.aggregate_prediction(flat_occupancy, horizon, occupancy_bundle)
            for horizon in qualified
        }
        receipt = {
            "schema": "go2_rgb_control_history_four_step_evaluation_seed_receipt_v1",
            "status": STATUS, "complete": True,
            "seed": seed, "checkpoint_sha256": training["checkpoint_sha256"],
            "states": len(records), "branches": len(records) * 12,
            "model_forward_states": 20,
            "autoregressive_horizon_forward_calls": 20 * 4,
            "historical_comparator_model_forwards": 0,
            "ledger": str(ledger), "ledger_sha256": _sha256_file(ledger),
            "state_digest_set": _sequence_digest([
                row["evaluation_state_digest"] for row in records
            ]),
        }
        receipt["receipt_digest"] = _digest(receipt)
        evaluation_receipts.append(receipt)

    del probe
    torch.cuda.empty_cache()

    analysis = {"equal_family": {}, "corpus_weighted": {}, "per_family": {}}
    for horizon in HORIZONS:
        key = f"H{horizon}"
        analysis["equal_family"][key] = _effect_table(
            cells, horizon, "equal_family"
        )
        analysis["corpus_weighted"][key] = _effect_table(
            cells, horizon, "corpus_weighted"
        )
        analysis["per_family"][key] = {
            family: _effect_table(cells, horizon, "per_family", family)
            for family in C.FAMILIES
        }
    analysis["retrieval_confusion_across_seeds"] = _retrieval_confusion(cells)

    # Raw H1--H4 degradation slopes by seed, with benefit-oriented 4-vs-2 deltas.
    slopes: dict[str, Any] = {}
    for endpoint, (metric, higher) in EFFECT_METRICS.items():
        cell_slopes = {cell: [] for cell in ("one_step", "two_step", "four_step")}
        for seed in C.FROZEN_SEEDS:
            for cell in cell_slopes:
                y = [_metric_value(cells[int(seed)][cell], horizon,
                                   "equal_family", metric)
                     for horizon in HORIZONS]
                cell_slopes[cell].append(float(np.polyfit(HORIZONS, y, 1)[0]))
        slopes[endpoint] = paired_effect_summary(
            cell_slopes["one_step"], cell_slopes["two_step"],
            cell_slopes["four_step"], higher_is_better=higher,
        )

    occupancy_result = _occupancy_analysis(
        occupancy_by_seed, gate, provenance, probe_digest
    )
    h4 = analysis["equal_family"]["H4"]
    interpretation = interpretation_from_effects(analysis)
    report = {
        "schema": "go2_rgb_control_history_four_step_evaluation_result_v1",
        "status": STATUS, "complete": True, "valid": True,
        "four_step_contract_digest": _contract_digest(contract),
        "common_manifest_digest": manifest["common_rows_digest"],
        "target_cache_index_digest": cache_index["target_cache_index_digest"],
        "training_receipt_set_digest": aggregate_training[
            "training_receipt_set_digest"
        ],
        "environment": environment,
        "frozen_stage_a": {
            "identity_manifest_digest": bundle.identity_digest,
            "corpus_digest": bundle.corpus_digest,
            "latent_index_digest": bundle.latent_index_digest,
            "verified_shard_set_digest": shards["verified_shard_set_digest"],
            "states": 20, "branches": 240,
        },
        "frozen_comparator_result_digest": frozen_result["report_digest"],
        "historical_comparator_model_forwards": 0,
        "new_four_step_model_forward_states": 8 * 20,
        "new_four_step_autoregressive_horizon_forward_calls": 8 * 20 * 4,
        "new_checkpoint_set_digest": checkpoint_set_digest,
        "historical_control_comparability": {
            "historical_control_train_rows": 3922,
            "new_four_step_train_rows": 3854,
            "row_difference": 68,
            "row_difference_fraction_of_historical": 68 / 3922,
            "historical_controls_sample_matched": False,
            "historical_controls_retrained_or_reselected": False,
            "disposition": "registered historical controls; sample mismatch retained",
        },
        "evaluation_receipts": evaluation_receipts,
        "cells_by_seed": {str(seed): value for seed, value in cells.items()},
        "paired_seed_analysis": analysis,
        "primary_H4_equal_family": h4,
        "degradation_slopes_H1_H4": slopes,
        "terminal_window_stability": {
            str(row["seed"]): row["terminal_window"] for row in new_receipts
        },
        "occupancy_co_outcome": occupancy_result,
        "interpretation": interpretation,
        "runtime_seconds": time.time() - scoring_started,
        "no_predictor_utility_shards_opened": True,
        "no_branches_targets_labels_or_masks_regenerated": True,
    }
    report["result_digest"] = _digest(report)
    occupancy_path = runtime_root() / "evaluation" / "occupancy.json"
    _write_json_once(occupancy_path, occupancy_result)
    _write_json_once(result_path, report)
    return validate_evaluation_result()


def validate_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Close the one-shot namespace after validating every immutable receipt."""
    terminal_path = runtime_root() / "terminal.json"
    if terminal_path.exists() or terminal_path.is_symlink():
        _die("the four-step namespace is already terminal; second validation forbidden")
    contract = require_contract()
    environment = environment_record(require_exact=True)
    manifest = validate_common_manifest()
    input_verification = validate_training_input_files(hash_files=False)
    cache_index = validate_target_cache_index()
    smoke = validate_smoke_receipt()
    preflight = validate_preflight_receipt()
    training = validate_training_receipt_set()
    evaluation = validate_evaluation_result()
    occupancy = evaluation["occupancy_co_outcome"]

    expected_files = {
        "contract.json", "target_availability.json", "common_h4_rows.jsonl",
        "common_h4_manifest.json", "training_input_verification.json",
        TARGET_BLOBS[3], TARGET_BLOBS[4], "target_cache_index.json",
        "attempts/smoke.json", "smoke.json", "attempts/preflight.json",
        "resource_preflight.json", "training_receipts.json",
        "attempts/evaluation.json", "evaluation/result.json",
        "evaluation/occupancy.json",
    }
    for seed_value in C.FROZEN_SEEDS:
        seed = int(seed_value)
        expected_files.update({
            f"attempts/train_seed_{seed}.json",
            f"training/seed_{seed}/seed_{seed}_rgb_four_step_epoch21.pt",
            f"training/seed_{seed}/checkpoint_receipts.jsonl",
            f"training/seed_{seed}/training_receipt.json",
            f"evaluation/prediction_ledgers/seed_{seed}.jsonl",
        })
    observed_files: set[str] = set()
    storage_bytes = 0
    inventory: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(
        runtime_root(), topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            if child.is_symlink():
                _die(f"runtime namespace contains a directory symlink: {child}")
        for name in file_names:
            path = directory_path / name
            relative = str(path.relative_to(runtime_root()))
            observed_files.add(relative)
            _require_immutable_file(path, f"runtime artifact {relative}")
            size = path.stat().st_size
            storage_bytes += size
            inventory.append({
                "path": relative, "bytes": size, "mode": "0444",
            })
    if observed_files != expected_files:
        _die(json.dumps({
            "runtime_namespace_missing": sorted(expected_files - observed_files),
            "runtime_namespace_extra": sorted(observed_files - expected_files),
        }, sort_keys=True))
    inventory.sort(key=lambda row: row["path"])

    expected_attempts = {
        "smoke": "smoke", "preflight": "preflight",
        **{f"train_seed_{seed}": f"train_seed_{seed}" for seed in C.FROZEN_SEEDS},
    }
    for filename, stage_name in expected_attempts.items():
        attempt = _read_json(runtime_root() / "attempts" / f"{filename}.json")
        if attempt.get("schema") != "go2_rgb_control_history_four_step_attempt_v1" \
                or attempt.get("stage") != stage_name \
                or attempt.get("four_step_contract_digest") != _contract_digest(contract):
            _die(f"attempt receipt differs for {filename}")
    evaluation_attempt = _read_json(runtime_root() / "attempts" / "evaluation.json")
    if evaluation_attempt.get("schema") != (
        "go2_rgb_control_history_four_step_evaluation_attempt_v1"
    ) or evaluation_attempt.get("resumable") is not False \
            or evaluation_attempt.get("retry_authorised") is not False:
        _die("evaluation attempt custody differs")

    terminal = {
        "schema": "go2_rgb_control_history_four_step_terminal_v1",
        "status": STATUS, "complete": True, "valid": True,
        "classification": "COMPLETE_FOUR_STEP_ROLLOUT_OBJECTIVE_RESULT",
        "four_step_contract_digest": _contract_digest(contract),
        "source_commit": contract["source_closure"]["source_repository_commit"],
        "source_closure_digest": next(
            value for key, value in contract["source_closure"].items()
            if key.endswith("digest") and isinstance(value, str) and len(value) == 64
        ),
        "environment": environment,
        "training_target_availability": {
            "available_rows_H1_H4": C.TARGET_AVAILABILITY["horizon_counts"],
            "family_counts_H1_H4": HORIZON_FAMILY_COUNTS,
            "common_H4_counts": manifest["counts"],
            "common_H4_family_counts": manifest["family_counts"],
            "exclusion_counts": manifest["exclusion_counts"],
            "incremental_reset_and_boundary_exclusions": C.TARGET_AVAILABILITY[
                "incremental_exclusions"
            ],
            "additional_encoding": "train H3 and H4 only",
            "new_simulator_corpus_generated": False,
        },
        "common_manifest_digest": manifest["common_rows_digest"],
        "common_manifest_receipt_digest": manifest["manifest_digest"],
        "training_input_verification_digest": input_verification[
            "verification_digest"
        ],
        "target_cache_index_digest": cache_index["target_cache_index_digest"],
        "smoke_digest": smoke["smoke_digest"],
        "resource_preflight_digest": preflight["preflight_digest"],
        "training_receipt_set_digest": training["training_receipt_set_digest"],
        "eight_training_receipt_digests": training["receipt_digests"],
        "evaluation_result_digest": evaluation["result_digest"],
        "occupancy_co_outcome_digest": occupancy["occupancy_digest"],
        "interpretation": evaluation["interpretation"],
        "primary_H4_equal_family": evaluation["primary_H4_equal_family"],
        "complete_H1_H4_result_path": "evaluation/result.json",
        "occupancy_H2_H4_result_path": "evaluation/occupancy.json",
        "historical_control_comparability": {
            "historical_control_train_rows": 3922,
            "new_four_step_train_rows": 3854,
            "row_difference": 68,
            "historical_controls_sample_matched": False,
            "historical_controls_retrained_or_reselected": False,
        },
        "runtime": {
            "target_encoding_wall_seconds": cache_index["wall_seconds"],
            "preflight_wall_seconds": preflight["wall_seconds_per_epoch"],
            "eight_training_wall_seconds": training["total_wall_seconds"],
            "evaluation_wall_seconds": evaluation["runtime_seconds"],
        },
        "runtime_namespace_inventory": inventory,
        "runtime_storage_bytes_before_terminal": storage_bytes,
        "technical_invalidity": None,
        "closed_scientific_lines_preserved": C.CLOSED_SCIENTIFIC_LINES,
        "predictor_utility_scoring_or_shards_opened": False,
        "utility_readout_trained": False,
        "planning_or_selected_action_endpoint": False,
        "final_200_state_corpus_generated": False,
        "new_counterfactual_or_simulator_corpus_generated": False,
        "automatic_follow_on_experiment_started": False,
        "training_or_evaluation_processes_remaining": 0,
        "target_encoder_processes_remaining": 0,
        "all_started_child_processes_joined": True,
        "nothing_remains_running": True,
    }
    terminal["terminal_digest"] = _digest(terminal)
    _write_json_once(terminal_path, terminal)
    return terminal


def _failure_artifact_inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.is_dir() or root.is_symlink():
        return records
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        directory_names[:] = [
            name for name in directory_names
            if not (directory_path / name).is_symlink()
        ]
        for name in sorted(file_names):
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                records.append({
                    "path": str(path.relative_to(root)), "regular": False,
                    "symlink": path.is_symlink(),
                })
                continue
            size = path.stat().st_size
            record: dict[str, Any] = {
                "path": str(path.relative_to(root)), "regular": True,
                "bytes": size, "mode": oct(path.stat().st_mode & 0o777),
            }
            if size <= 128 * 2**20:
                record["sha256"] = _sha256_file(path)
            else:
                record["sha256_not_recomputed"] = (
                    "large artifact remains bound by its immutable receipt if complete"
                )
            records.append(record)
    return sorted(records, key=lambda row: row["path"])


def _freeze_failure_artifacts(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        return
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        directory_names[:] = [
            name for name in directory_names
            if not (directory_path / name).is_symlink()
        ]
        for name in file_names:
            path = directory_path / name
            if path.is_file() and not path.is_symlink():
                _make_read_only(path)


def record_failure_terminal(stage: str, exception: BaseException) -> dict[str, Any] | None:
    """Preserve a one-shot technical failure without authorising any retry."""
    root = runtime_root()
    terminal_path = root / "terminal.json"
    if terminal_path.exists() or terminal_path.is_symlink():
        return None
    if not root.is_dir() or root.is_symlink():
        return None
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    _freeze_failure_artifacts(root)
    raw_contract: dict[str, Any] = {}
    contract_path = Path(C.contract_path())
    if contract_path.is_file() and not contract_path.is_symlink():
        with contextlib.suppress(Exception):
            raw_contract = _read_json(contract_path)
    message = str(exception)
    classification_stage = stage
    if stage == "manifest":
        classification_stage = (
            "manifest_lineage" if any(
                token in message.lower() for token in ("comparator", "lineage")
            ) else "manifest_availability"
        )
    elif stage in ("evaluate", "validate") and "occupancy" in message.lower():
        classification_stage = "occupancy"
    classification = C.FAILURE_STAGE_CLASSIFICATION.get(
        classification_stage, "INVALID_CONTRACT_OR_SOURCE_LINEAGE"
    )
    completed_seeds: list[int] = []
    present_training_seed_receipts: list[int] = []
    for seed_value in C.FROZEN_SEEDS:
        seed = int(seed_value)
        receipt_path = _seed_receipt_path(seed)
        if not receipt_path.is_file() or receipt_path.is_symlink():
            continue
        present_training_seed_receipts.append(seed)
        with contextlib.suppress(Exception):
            receipt = _read_json(receipt_path)
            payload = {key: value for key, value in receipt.items()
                       if key != "training_receipt_digest"}
            checkpoint = _seed_checkpoint_path(seed)
            if receipt.get("training_receipt_digest") == _digest(payload) \
                    and receipt.get("schema") == (
                        "go2_rgb_control_history_four_step_training_receipt_v1"
                    ) and receipt.get("complete") is True \
                    and receipt.get("valid") is True \
                    and receipt.get("seed") == seed \
                    and receipt.get("epochs_trained") == EPOCHS \
                    and receipt.get("checkpoint_epoch") == CHECKPOINT_EPOCH \
                    and isinstance(receipt.get("history"), list) \
                    and [row.get("epoch") for row in receipt["history"]] == list(
                        range(EPOCHS)
                    ) and checkpoint.is_file() and not checkpoint.is_symlink() \
                    and receipt.get("checkpoint_sha256") == _sha256_file(checkpoint):
                completed_seeds.append(seed)
    completed_eval_seeds = []
    completed_eval_states: dict[str, int] = {}
    for seed in C.FROZEN_SEEDS:
        ledger = root / "evaluation" / "prediction_ledgers" / f"seed_{seed}.jsonl"
        if ledger.is_file():
            with contextlib.suppress(Exception):
                rows = _read_jsonl(ledger)
                count = len(rows)
                completed_eval_states[str(seed)] = count
                valid_rows = count == 20 and all(
                    row.get("seed") == int(seed)
                    and row.get("state_index") == index
                    and row.get("evaluation_state_digest") == _digest({
                        key: value for key, value in row.items()
                        if key != "evaluation_state_digest"
                    })
                    and set(row.get("per_horizon", {})) == {"1", "2", "3", "4"}
                    and len(row.get("occupancy_branches_H2_H4", [])) == 12
                    for index, row in enumerate(rows)
                )
                if valid_rows:
                    completed_eval_seeds.append(int(seed))
    targets_encoded: Any = False
    index_path = root / ENCODE_RECEIPT
    progress_path = root / "target_encoding_progress.json"
    if index_path.is_file():
        targets_encoded = {"receipt_present": True, "semantically_complete": False}
        with contextlib.suppress(Exception):
            index = _read_json(index_path)
            self_valid = index.get("target_cache_index_digest") == _digest({
                key: value for key, value in index.items()
                if key != "target_cache_index_digest"
            })
            cache_files_valid = all(
                Path(index["caches"][str(horizon)]["path"]).is_file()
                and Path(index["caches"][str(horizon)]["path"]).stat().st_size
                == int(index["caches"][str(horizon)]["bytes"])
                for horizon in (3, 4)
            )
            targets_encoded = {
                "receipt_present": True,
                "receipt_self_digest_valid": self_valid,
                "cache_files_size_bound": cache_files_valid,
                "semantically_complete": bool(
                    self_valid and cache_files_valid and index.get("complete") is True
                ),
                "independent_large_blob_rehash_after_failure": False,
            }
    elif progress_path.is_file():
        with contextlib.suppress(Exception):
            progress = _read_json(progress_path)
            targets_encoded = {
                "complete": False,
                "unique_paths_completed": progress.get("unique_paths_completed"),
                "unique_paths_total": progress.get("unique_paths_total"),
            }
    manifest_digest = None
    manifest_path = root / COMMON_MANIFEST
    if manifest_path.is_file():
        with contextlib.suppress(Exception):
            manifest_digest = _read_json(manifest_path).get("common_rows_digest")
    source = raw_contract.get("source_closure", {})
    terminal = {
        "schema": "go2_rgb_control_history_four_step_terminal_failure_v1",
        "status": STATUS, "complete": True, "valid": False,
        "classification": classification, "failed_stage": stage,
        "exception_type": type(exception).__name__, "exception": message,
        "contract_digest": _contract_digest(raw_contract) if raw_contract else None,
        "source_commit": source.get("source_repository_commit", C.BASE_SOURCE_COMMIT),
        "common_manifest_digest_if_issued": manifest_digest,
        "targets_encoded": targets_encoded,
        "completed_training_seed_count": len(completed_seeds),
        "completed_training_seeds": completed_seeds,
        "present_training_seed_receipts": present_training_seed_receipts,
        "completed_training_epochs_lower_bound": len(completed_seeds) * EPOCHS,
        "completed_training_updates_lower_bound": len(completed_seeds) * EPOCHS * 964,
        "completed_evaluation_seed_count": len(completed_eval_seeds),
        "completed_evaluation_seeds": completed_eval_seeds,
        "completed_evaluation_states_by_seed": completed_eval_states,
        "artifacts_present": _failure_artifact_inventory(root),
        "resource_counters": {
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ) * 1024,
            "system_ram_total_bytes": _mem_total_bytes(),
            "system_ram_available_bytes": _mem_available_bytes(),
            "cuda_memory_allocated_bytes": (
                int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None
            ),
            "cuda_memory_reserved_bytes": (
                int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else None
            ),
        },
        "retry_resume_or_replacement_authorised": False,
        "automatic_follow_on_experiment_started": False,
        "predictor_utility_or_final_corpus_access": False,
        "all_started_child_processes_joined": True,
        "training_or_evaluation_processes_remaining_after_driver_exit": 0,
        "target_encoder_processes_remaining_after_driver_exit": 0,
        "nothing_remains_running": True,
    }
    missing = set(C.FAILURE_RECEIPT_REQUIRED_FIELDS) - set(terminal)
    if missing:
        terminal["failure_receipt_internal_error"] = (
            f"missing required fields before publication: {sorted(missing)}"
        )
    terminal["terminal_digest"] = _digest(terminal)
    _write_json_once(terminal_path, terminal)
    return terminal


STAGES = {
    "issue": issue_stage,
    "manifest": manifest_stage,
    "encode": encode_stage,
    "smoke": smoke_stage,
    "preflight": preflight_stage,
    "train-seed": train_seed_stage,
    "train-all": train_all_stage,
    "evaluate": evaluate_stage,
    "validate": validate_stage,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--encode-batch", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    terminal_path = runtime_root() / "terminal.json"
    if terminal_path.exists() or terminal_path.is_symlink():
        print("four-step namespace is terminal; no retry or second invocation permitted",
              file=sys.stderr, flush=True)
        return 2
    try:
        result = STAGES[args.stage](args)
    except BaseException as exception:
        terminal = record_failure_terminal(args.stage, exception)
        payload = terminal if terminal is not None else {
            "failed_stage": args.stage, "exception_type": type(exception).__name__,
            "exception": str(exception), "failure_terminal_written": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
              file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
