#!/usr/bin/env python3
"""Audit production FP32 dynamic support against the stdlib FP64 reference.

This gate is metadata-only.  It opens the frozen micro-overfit panel and the
authorized train attitude sidecar, but never opens RGB, labels, checkpoints,
holdouts, calibration, G2, runtime, or sealed payloads.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_SCHEMA = "lewm_go2_dynamic_cartesian_fit_panel_parity_result_v1"
PANEL_PATH = ROOT / ".generated/go2_physical_micro_overfit/patch7_v1/panel.json"
SIDECAR_MANIFEST_PATH = (
    ROOT / ".generated/go2_attitude_sidecar/dynamic_cartesian_v1/manifest.json"
)
BINDING_PATH = ROOT / "docs/lewm_go2_dynamic_cartesian_n32_v1_binding_2026-07-11.md"
DYNAMIC_GEOMETRY_PATH = (
    ROOT / "lewm/benchmarks/go2_dynamic_cell_square_projection.py"
)
MODEL_PATH = ROOT / "lewm/models/egomotion_bev_jepa.py"
DEFAULT_OUTPUT_PATH = (
    ROOT / ".generated/go2_dynamic_cartesian_fit_panel_parity/v1/result.json"
)

PANEL_FILE_SHA256 = "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c"
PANEL_CONTENT_SHA256 = "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f"
FIT_ROWS_SHA256 = "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d"
SIDECAR_MANIFEST_FILE_SHA256 = (
    "6fafa417b4f724a0fdf32cfde5740025c3117e4c0b43231fe9ebe94bd9eff529"
)
SIDECAR_MANIFEST_CONTENT_SHA256 = (
    "6f1ef7d9ac0c55a42182c3e2c75909f00ab37fffa460aadb549d5cd60d278c1a"
)
SIDECAR_TRAIN_FILE_SHA256 = (
    "6cd47d0d679ace897f5b5d8e5c2f11eabab01930904666161eec3792fd9ab6d6"
)
BINDING_SHA256 = "42687e80a16fb424be47d49782699bbc3ed549d7826a0ce6e78e92aa37188e1e"
DYNAMIC_GEOMETRY_SHA256 = (
    "ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf"
)
MODEL_SHA256 = "c4006e9804182b077399229d43bc8c9be64b5af12c81fff4076d5a78e6ef359b"

EXPECTED_TRANSITIONS = 160
EXPECTED_FRAMES = 320
GRID_SHAPE = (64, 64)
CELLS_PER_FRAME = GRID_SHAPE[0] * GRID_SHAPE[1]
EXPECTED_CELL_DECISIONS = EXPECTED_FRAMES * CELLS_PER_FRAME
TORCH_BATCH_SIZE = 4
STDLIB_WORKERS = 6
TOKEN_SIDE = 16
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
SIDES = ("current", "next")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _strict_json_bytes(data: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains nonfinite JSON number {value}")

    try:
        value = json.loads(data, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_thread_caps() -> dict[str, str]:
    actual = {name: os.environ.get(name, "") for name in THREAD_ENV_NAMES}
    if actual != {name: "1" for name in THREAD_ENV_NAMES}:
        raise ValueError("all native CPU thread caps must be explicitly set to one")
    return actual


def _worker_initializer() -> None:
    for name in THREAD_ENV_NAMES:
        os.environ[name] = "1"


def _stdlib_mask(task: tuple[int, tuple[float, ...], float]) -> tuple[int, bytes]:
    frame_index, quaternion, yaw = task
    from lewm.benchmarks.go2_dynamic_cell_square_projection import (
        build_dynamic_cell_square_support_mask,
    )

    mask = build_dynamic_cell_square_support_mask(quaternion, yaw)
    payload = bytes(int(value) for row in mask for value in row)
    if len(payload) != CELLS_PER_FRAME:
        raise ValueError("stdlib reference returned the wrong support-mask shape")
    return frame_index, payload


def _frame_identity(frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": str(frame["family"]),
        "global_row": int(frame["global_row"]),
        "side": str(frame["side"]),
        "frame_index": int(frame["frame_index"]),
        "timestamp_ns": int(frame["timestamp_ns"]),
        "sidecar_row_identity_sha256": str(frame["sidecar_row_identity_sha256"]),
    }


def _load_joined_frames() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    panel_data = PANEL_PATH.read_bytes()
    if _sha256_bytes(panel_data) != PANEL_FILE_SHA256:
        raise ValueError("fit-panel file SHA-256 differs from the frozen value")
    panel = _strict_json_bytes(panel_data, name="fit panel")

    from lewm.benchmarks.go2_physical_micro_overfit import validate_panel_manifest
    from lewm.datasets.go2_attitude_sidecar import (
        FROZEN_BUILD_CONTRACT,
        load_attitude_sidecar_roles,
        row_identity_sha256,
    )

    panels = validate_panel_manifest(panel)
    if panel.get("content_sha256") != PANEL_CONTENT_SHA256:
        raise ValueError("fit-panel content SHA-256 differs from the frozen value")
    fit_record = panel.get("panels", {}).get("fit")
    if not isinstance(fit_record, Mapping):
        raise ValueError("fit-panel record is missing")
    fit_rows = list(panels["fit"])
    if (
        len(fit_rows) != EXPECTED_TRANSITIONS
        or int(fit_record.get("frame_count", -1)) != EXPECTED_FRAMES
        or str(fit_record.get("rows_sha256", "")) != FIT_ROWS_SHA256
        or _canonical_json_sha256(fit_rows) != FIT_ROWS_SHA256
    ):
        raise ValueError("fit-panel row identity/count differs from the frozen value")
    if any(str(row.get("dataset_role")) != "train" for row in fit_rows):
        raise ValueError("fit panel contains a non-train row")

    loaded = load_attitude_sidecar_roles(
        SIDECAR_MANIFEST_PATH,
        roles=("train",),
        expected_manifest_sha256=SIDECAR_MANIFEST_FILE_SHA256,
        contract=FROZEN_BUILD_CONTRACT,
    )
    train_rows = loaded.get("train")
    if train_rows is None or len(train_rows) != 4262:
        raise ValueError("authorized train attitude sidecar has the wrong row count")
    by_global_row = {int(row["global_row"]): row for row in train_rows}
    if len(by_global_row) != len(train_rows):
        raise ValueError("authorized train attitude sidecar has duplicate global rows")

    frames: list[dict[str, Any]] = []
    joined_global_rows: list[int] = []
    for row in fit_rows:
        global_row = int(row["global_row"])
        sidecar_row = by_global_row.get(global_row)
        if sidecar_row is None:
            raise ValueError(f"fit global row {global_row} lacks train attitude metadata")
        if str(sidecar_row["row_identity_sha256"]) != row_identity_sha256(row):
            raise ValueError(f"fit/attitude identity mismatch at global row {global_row}")
        exact_fields = (
            ("dataset_role", "dataset_role"),
            ("env_index", "env_index"),
            ("current_frame_index", "current_frame_index"),
            ("next_frame_index", "next_frame_index"),
            ("current_timestamp_ns", "current_timestamp_ns"),
            ("next_timestamp_ns", "next_timestamp_ns"),
        )
        for panel_field, sidecar_field in exact_fields:
            if sidecar_row[sidecar_field] != row[panel_field]:
                raise ValueError(
                    f"fit/attitude {panel_field} mismatch at global row {global_row}"
                )
        joined_global_rows.append(global_row)
        for side in SIDES:
            attitude = sidecar_row[side]
            frames.append(
                {
                    "family": str(row["family"]),
                    "global_row": global_row,
                    "side": side,
                    "frame_index": int(row[f"{side}_frame_index"]),
                    "timestamp_ns": int(row[f"{side}_timestamp_ns"]),
                    "sidecar_row_identity_sha256": str(
                        sidecar_row["row_identity_sha256"]
                    ),
                    "quaternion": tuple(
                        float(value)
                        for value in attitude["base_quat_world_xyzw"]
                    ),
                    "yaw": float(attitude["stored_base_yaw_rad"]),
                }
            )
    if (
        len(joined_global_rows) != EXPECTED_TRANSITIONS
        or len(set(joined_global_rows)) != EXPECTED_TRANSITIONS
        or joined_global_rows != sorted(joined_global_rows)
        or len(frames) != EXPECTED_FRAMES
    ):
        raise ValueError("fit/attitude join is incomplete, duplicate, or reordered")
    if Counter(frame["family"] for frame in frames) != Counter(
        {family: 64 for family in FAMILIES}
    ):
        raise ValueError("joined fit frames lost the frozen family balance")
    if Counter(frame["side"] for frame in frames) != Counter(
        {side: EXPECTED_TRANSITIONS for side in SIDES}
    ):
        raise ValueError("joined fit frames lost the endpoint balance")
    identities = [_frame_identity(frame) for frame in frames]
    return frames, {
        "ordered_frame_identity_sha256": _canonical_json_sha256(identities),
        "ordered_global_rows_sha256": _canonical_json_sha256(joined_global_rows),
    }


def _run_stdlib(frames: Sequence[Mapping[str, Any]]) -> list[bytes]:
    tasks = [
        (index, tuple(frame["quaternion"]), float(frame["yaw"]))
        for index, frame in enumerate(frames)
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=STDLIB_WORKERS,
        mp_context=context,
        initializer=_worker_initializer,
    ) as executor:
        outputs = list(executor.map(_stdlib_mask, tasks, chunksize=4))
    indices = [index for index, _payload in outputs]
    if indices != list(range(EXPECTED_FRAMES)):
        raise ValueError("stdlib worker results were incomplete or reordered")
    return [payload for _index, payload in outputs]


def _run_torch(frames: Sequence[Mapping[str, Any]]) -> list[bytes]:
    import torch

    from lewm.models.egomotion_bev_jepa import (
        _cell_square_horizontal_offsets,
        _dynamic_projective_cell_square_attention_geometry,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    forward = torch.linspace(-0.95, 5.35, 64, dtype=torch.float32, device=device)
    left = torch.linspace(-3.15, 3.15, 64, dtype=torch.float32, device=device)
    forward_grid, left_grid = torch.meshgrid(forward, left, indexing="ij")
    outputs: list[bytes] = []
    for start in range(0, len(frames), TORCH_BATCH_SIZE):
        batch = frames[start : start + TORCH_BATCH_SIZE]
        quaternions = torch.tensor(
            [frame["quaternion"] for frame in batch],
            dtype=torch.float64,
            device=device,
        )
        yaws = torch.tensor(
            [frame["yaw"] for frame in batch],
            dtype=torch.float64,
            device=device,
        )
        with torch.no_grad():
            bias, visibility = _dynamic_projective_cell_square_attention_geometry(
                metric_forward_grid=forward_grid,
                metric_left_grid=left_grid,
                token_side=TOKEN_SIDE,
                horizontal_fov_deg=78.323,
                vertical_fov_deg=62.8370386364,
                camera_xyz_body_m=(0.326, 0.0, 0.043),
                camera_rpy_body_rad=(0.0, 0.0, 0.0),
                near_m=0.05,
                vertical_anchor_z_body_m=(-0.333, -0.133, 0.067, 0.267, 0.467),
                horizontal_offsets_body_m=_cell_square_horizontal_offsets(0.1),
                sigma_tokens=2.0,
                bias_floor=-6.0,
                base_quat_world_xyzw=quaternions,
                stored_base_yaw_rad=yaws,
            )
        expected_bias_shape = (len(batch), CELLS_PER_FRAME, TOKEN_SIDE * TOKEN_SIDE)
        if tuple(bias.shape) != expected_bias_shape or not bool(torch.isfinite(bias).all()):
            raise ValueError("production Torch attention bias is malformed or nonfinite")
        if tuple(visibility.shape) != (len(batch), CELLS_PER_FRAME):
            raise ValueError("production Torch visibility has the wrong shape")
        outputs.extend(
            bytes(mask.to(dtype=torch.uint8).cpu().tolist()) for mask in visibility
        )
        del bias, visibility, quaternions, yaws
    if len(outputs) != EXPECTED_FRAMES:
        raise ValueError("production Torch output frame count mismatch")
    return outputs


def _compare(
    frames: Sequence[Mapping[str, Any]],
    stdlib_masks: Sequence[bytes],
    torch_masks: Sequence[bytes],
) -> dict[str, Any]:
    if not (
        len(frames) == len(stdlib_masks) == len(torch_masks) == EXPECTED_FRAMES
    ):
        raise ValueError("parity inputs have inconsistent frame counts")
    mismatch_count = 0
    mismatch_frames = 0
    first_mismatch: dict[str, Any] | None = None
    stdlib_visible = 0
    torch_visible = 0
    per_family = {
        family: {"frames": 0, "visible": 0, "cell_decisions": 0}
        for family in FAMILIES
    }
    per_side = {
        side: {"frames": 0, "visible": 0, "cell_decisions": 0}
        for side in SIDES
    }
    for frame_index, (frame, reference, production) in enumerate(
        zip(frames, stdlib_masks, torch_masks, strict=True)
    ):
        if len(reference) != CELLS_PER_FRAME or len(production) != CELLS_PER_FRAME:
            raise ValueError("one parity mask has the wrong byte count")
        reference_visible = sum(reference)
        production_visible = sum(production)
        stdlib_visible += reference_visible
        torch_visible += production_visible
        family = str(frame["family"])
        side = str(frame["side"])
        per_family[family]["frames"] += 1
        per_family[family]["visible"] += reference_visible
        per_family[family]["cell_decisions"] += CELLS_PER_FRAME
        per_side[side]["frames"] += 1
        per_side[side]["visible"] += reference_visible
        per_side[side]["cell_decisions"] += CELLS_PER_FRAME
        frame_mismatches = [
            cell for cell, (left, right) in enumerate(zip(reference, production))
            if left != right
        ]
        if frame_mismatches:
            mismatch_frames += 1
            mismatch_count += len(frame_mismatches)
            if first_mismatch is None:
                cell = frame_mismatches[0]
                first_mismatch = {
                    **_frame_identity(frame),
                    "ordered_frame_index": frame_index,
                    "cell_row": cell // GRID_SHAPE[1],
                    "cell_column": cell % GRID_SHAPE[1],
                    "stdlib_visible": bool(reference[cell]),
                    "torch_visible": bool(production[cell]),
                }
    result = {
        "frames": len(frames),
        "cell_decisions": len(frames) * CELLS_PER_FRAME,
        "stdlib_visible_cells": stdlib_visible,
        "torch_visible_cells": torch_visible,
        "mismatch_frames": mismatch_frames,
        "mismatch_cells": mismatch_count,
        "first_mismatch": first_mismatch,
        "stdlib_ordered_masks_sha256": _sha256_bytes(b"".join(stdlib_masks)),
        "torch_ordered_masks_sha256": _sha256_bytes(b"".join(torch_masks)),
        "per_family": per_family,
        "per_side": per_side,
    }
    if (
        result["frames"] != EXPECTED_FRAMES
        or result["cell_decisions"] != EXPECTED_CELL_DECISIONS
        or mismatch_count != 0
        or mismatch_frames != 0
        or first_mismatch is not None
        or stdlib_visible != torch_visible
        or result["stdlib_ordered_masks_sha256"]
        != result["torch_ordered_masks_sha256"]
    ):
        raise ValueError(f"FP32/FP64 support parity failed: {result}")
    return result


def _write_result_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload), sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_bytes(encoded)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.resolve()
    if output != DEFAULT_OUTPUT_PATH.resolve():
        raise ValueError("authoritative parity output path changed")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing result: {output}")
    thread_environment = _require_thread_caps()
    source_hashes = {
        "binding": _sha256_file(BINDING_PATH),
        "dynamic_geometry": _sha256_file(DYNAMIC_GEOMETRY_PATH),
        "model": _sha256_file(MODEL_PATH),
        "runner": _sha256_file(Path(__file__).resolve()),
    }
    expected_source_hashes = {
        "binding": BINDING_SHA256,
        "dynamic_geometry": DYNAMIC_GEOMETRY_SHA256,
        "model": MODEL_SHA256,
    }
    for role, expected in expected_source_hashes.items():
        if source_hashes[role] != expected:
            raise ValueError(f"{role} source differs from its frozen SHA-256")

    started_at = datetime.now(timezone.utc).isoformat()
    frames, join = _load_joined_frames()
    stdlib_masks = _run_stdlib(frames)
    torch_masks = _run_torch(frames)
    comparison = _compare(frames, stdlib_masks, torch_masks)
    completed_at = datetime.now(timezone.utc).isoformat()
    core = {
        "schema": RESULT_SCHEMA,
        "status": "pass",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "inputs": {
            "binding": {"path": str(BINDING_PATH), "sha256": BINDING_SHA256},
            "panel": {
                "path": str(PANEL_PATH),
                "file_sha256": PANEL_FILE_SHA256,
                "content_sha256": PANEL_CONTENT_SHA256,
                "fit_rows_sha256": FIT_ROWS_SHA256,
            },
            "attitude_sidecar": {
                "manifest_path": str(SIDECAR_MANIFEST_PATH),
                "manifest_file_sha256": SIDECAR_MANIFEST_FILE_SHA256,
                "manifest_content_sha256": SIDECAR_MANIFEST_CONTENT_SHA256,
                "authorized_roles": ["train"],
                "train_file_sha256": SIDECAR_TRAIN_FILE_SHA256,
            },
            "source_sha256": source_hashes,
        },
        "execution": {
            "device": "cpu",
            "gpu_used": False,
            "torch_dtype": "float32",
            "stdlib_reference_dtype": "float64",
            "torch_batch_size": TORCH_BATCH_SIZE,
            "stdlib_workers": STDLIB_WORKERS,
            "native_threads_per_process": 1,
            "thread_environment": thread_environment,
            "token_side": TOKEN_SIDE,
            "grid_shape": list(GRID_SHAPE),
        },
        "join": {
            "transitions": EXPECTED_TRANSITIONS,
            "frames": EXPECTED_FRAMES,
            "role": "train",
            "method": "exact_global_row_plus_row_identity_and_endpoint_metadata",
            **join,
        },
        "comparison": comparison,
        "access_ledger": {
            "panel_metadata_byte_opens": 1,
            "sidecar_manifest_byte_opens": 1,
            "train_sidecar_byte_opens": 1,
            "checkpoint_selection_sidecar_byte_opens": 0,
            "probability_calibration_sidecar_byte_opens": 0,
            "g2_sidecar_byte_opens": 0,
            "rgb_byte_opens": 0,
            "label_shard_byte_opens": 0,
            "depth_byte_opens": 0,
            "model_checkpoint_or_output_opens": 0,
            "runtime_or_sealed_payload_opens": 0,
        },
        "gates": {
            "exact_fit_transition_count": True,
            "exact_fit_frame_count": True,
            "train_only_attitude_join": True,
            "all_cell_decisions_scored": True,
            "zero_mismatch_frames": True,
            "zero_mismatch_cells": True,
            "ordered_mask_hashes_equal": True,
            "cpu_only": True,
            "forbidden_payload_opens_zero": True,
            "pass": True,
        },
    }
    result = {**core, "content_sha256": _canonical_json_sha256(core)}
    result_file_sha256 = _write_result_exclusive(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": result_file_sha256,
                "content_sha256": result["content_sha256"],
                "frames": comparison["frames"],
                "cell_decisions": comparison["cell_decisions"],
                "visible_cells": comparison["torch_visible_cells"],
                "mismatch_frames": comparison["mismatch_frames"],
                "mismatch_cells": comparison["mismatch_cells"],
                "ordered_masks_sha256": comparison[
                    "torch_ordered_masks_sha256"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
