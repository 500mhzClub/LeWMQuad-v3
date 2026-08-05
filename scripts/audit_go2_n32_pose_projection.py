#!/usr/bin/env python3
"""Audit fixed versus recorded camera projection using train metadata only."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.benchmarks.go2_n32_pose_projection_audit import (  # noqa: E402
    FAMILIES,
    QUERY_COUNT,
    RESULT_SCHEMA,
    ProjectionComparison,
    compare_projection,
    geometry_contract,
    ordering_decision,
    reconstruct_yaw_aligned_camera,
    summarize_frame_comparisons,
)
BINDING_SHA256 = "c959c45737b9242ef667772af4c7b72effcbb39ae687f5ee28226e38cd63854a"
AMENDMENT_SHA256 = "56f29c4f2eb05c726b0b4461352fe89da2639b86bf9341ec3072958720cf7c6d"
SUPERSEDED_SCOPE_AMENDMENT_SHA256 = (
    "35c0de28a795d6b5c246548f5d773326b3f137310c0ec9a840b3e7bf1d302e1d"
)
ROLE_NAMESPACE_AMENDMENT_SHA256 = (
    "ae17eb856c5329e8c5dfa5e4339306ef19e60c53c5f67d43746b268be9cc3370"
)
FIT_PANEL_FILE_SHA256 = (
    "77d84e242d75b81fd2b96f086e9cf5df72f0a907e1fe7ce24fc48bbc5d514037"
)
FIT_PANEL_CONTENT_SHA256 = (
    "8e44dd0238077120e97fd06b4550d6504627066c7e8ddfdfbd138fd7504ee7a8"
)
SOURCE_PANEL_FILE_SHA256 = (
    "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c"
)
SOURCE_PANEL_CONTENT_SHA256 = (
    "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f"
)
FIT_ROWS_SHA256 = "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d"
EXPECTED_FIT_ROWS = 160
EXPECTED_FRAME_RECORDS = 320
EXPECTED_SUMMARY_COUNT = 20
BINDING_PATH = ROOT / "docs/lewm_go2_n32_pose_projection_audit_binding_2026-07-11.md"
AMENDMENT_PATH = (
    ROOT / "docs/lewm_go2_n32_pose_projection_fit_panel_amendment_2026-07-11.md"
)
SUPERSEDED_SCOPE_AMENDMENT_PATH = (
    ROOT
    / "docs/lewm_go2_n32_pose_projection_train_source_scope_amendment_2026-07-11.md"
)
ROLE_NAMESPACE_AMENDMENT_PATH = (
    ROOT
    / "docs/lewm_go2_n32_pose_projection_role_namespace_amendment_2026-07-11.md"
)
FIT_PANEL_PATH = (
    ROOT / ".generated/go2_n32_pose_projection_audit/v1/fit_panel.json"
)
EXPECTED_SOURCE_PANEL_PATH = str(
    ROOT / ".generated/go2_physical_micro_overfit/patch7_v1/panel.json"
)
OUTPUT_PATH = ROOT / ".generated/go2_n32_pose_projection_audit/v1/result.json"
SUMMARY_ROOT = ROOT / ".generated/go2_render_selected_v04/scenes"
FRAMES_ROOT = ROOT / ".generated/datagen_full/rollout"
SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "lewm/benchmarks/go2_n32_pose_projection_audit.py",
)
EXPECTED_SUMMARY_SHA256 = {
    "scene_074f19f0608afca2/summary.json": "7a5d3b1e6ff5a8acb914ae5226326084c2b951517c110ffc19d7a99945fe0413",
    "scene_142dbd9b0428f16f/summary.json": "995e192cc1830f32bd2dc6d358da91f5bdaec48bd585ac2dadecc45517cbd2b0",
    "scene_4931dab75d2ceee8/summary.json": "7800d0d6a14ea54b9970d1dac36472446cd525af8c893736ebe1c4b4bf57cc23",
    "scene_49db95fc9ed0ce8f/summary.json": "80a035ceecf56f2c668fed3ab1dbabeeca181cb2886fedafa7116ec26bc0566d",
    "scene_4af4d0549179a705/summary.json": "bcb3866fe141c0c629368eefee8e228630ca8f3b30e1c2810b34e68fd61347b4",
    "scene_7239d51aced24ee3/summary.json": "5c6785479b9a302fcffb1d7532e450af10d2e2625a030eff872edf22b23aef6f",
    "scene_7f390beda8f5070f/summary.json": "2dc1f874130cb733be4f28eccae3359aac7bdc4e2947718391182ad651d027e7",
    "scene_9ff98ead4f1a2e96/summary.json": "203ffca9205f68dc74e6135718d3fec4bfb55e9c841bf7a4eb49964930309cc0",
    "scene_a81215e4d326a2a2/summary.json": "7b9c5dff08be0876327f8b625d225e4b1729320f98b9ccb1efcbd1c68cc2e3c1",
    "scene_b1355439db03d8f8/summary.json": "d21cd06b202422ecce81c009c08b13ab4e92be86bdc93f6571e69ac265f33fa9",
    "scene_b748962d390baeca/summary.json": "a3a90172486dc08f3e7a1728da71e43ae224aefddc22ba32e1de5b4fa6ab7f38",
    "scene_b75bb34744434970/summary.json": "64bcf8f57c55cb3456f6dd04be23bbdc417865b2ee8dbad914b5eaa387d61b6b",
    "scene_bc5a05ec9fce8d9c/summary.json": "41377a7619560162b7fd4453ca302321d2f5f22aee1a8c7397ff32626bbb1a92",
    "scene_c60650f53aaae4a6/summary.json": "be319a4b1a6e456367c3a6b4d9eee5059380ef83ebe720416b7f292a959c2d6e",
    "scene_cfcadb2bd44cce85/summary.json": "fa5a9049889a10700cd678fea78ecfb6f91545403ebfdfd304d1dc59a4b6d40a",
    "scene_d8b06cdfb1f739ed/summary.json": "6f06ee751ec3a26de741bdafcf39cb044e49734cb5a2ab1103ab2834e3edf3c2",
    "scene_ddc88df212918857/summary.json": "7b1deec174715696d4a3dd653610886e1244edfa993a8c0dc0e91176b728488f",
    "scene_df1c6b34503f2ae1/summary.json": "deed15024342195754b9022522c048624ab09a1d55e2727f615822d5b6f658e8",
    "scene_e0c2fe611e747d90/summary.json": "df2fde293612833f00f15a25a8c81c799e15e4674f5ad7f29a0d7ea06e9fd341",
    "scene_ebc33be3e6a87264/summary.json": "12b5825f4dc2388631190cc80dd42f9cea1bbbbf002f666f12ca53ddde704a35",
}
EXPECTED_LEGACY_SOURCE_SPLIT = {
    "scene_074f19f0608afca2/summary.json": "train",
    "scene_142dbd9b0428f16f/summary.json": "test_hard",
    "scene_4931dab75d2ceee8/summary.json": "train",
    "scene_49db95fc9ed0ce8f/summary.json": "train",
    "scene_4af4d0549179a705/summary.json": "train",
    "scene_7239d51aced24ee3/summary.json": "test_id",
    "scene_7f390beda8f5070f/summary.json": "train",
    "scene_9ff98ead4f1a2e96/summary.json": "train",
    "scene_a81215e4d326a2a2/summary.json": "train",
    "scene_b1355439db03d8f8/summary.json": "val",
    "scene_b748962d390baeca/summary.json": "train",
    "scene_b75bb34744434970/summary.json": "test_id",
    "scene_bc5a05ec9fce8d9c/summary.json": "val",
    "scene_c60650f53aaae4a6/summary.json": "train",
    "scene_cfcadb2bd44cce85/summary.json": "train",
    "scene_d8b06cdfb1f739ed/summary.json": "train",
    "scene_ddc88df212918857/summary.json": "train",
    "scene_df1c6b34503f2ae1/summary.json": "train",
    "scene_e0c2fe611e747d90/summary.json": "train",
    "scene_ebc33be3e6a87264/summary.json": "train",
}
EXPECTED_LEGACY_SOURCE_SPLIT_FRAME_COUNTS = {
    "train": 244,
    "test_hard": 14,
    "test_id": 32,
    "val": 30,
}


def canonical_json_sha256(value: object) -> str:
    """Hash strict canonical JSON without an unbound transitive dependency."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def new_access_ledger() -> dict[str, Any]:
    """Return the explicit zero-access state used before authorization."""

    return {
        "authorization_checked": False,
        "authorized": False,
        "metadata": {
            "binding": {"unique_files": 0, "hash_opens": 0, "parse_opens": 0},
            "fit_panel_amendment": {
                "unique_files": 0,
                "hash_opens": 0,
                "parse_opens": 0,
            },
            "superseded_scope_amendment": {
                "unique_files": 0,
                "hash_opens": 0,
                "parse_opens": 0,
            },
            "role_namespace_amendment": {
                "unique_files": 0,
                "hash_opens": 0,
                "parse_opens": 0,
            },
            "fit_panel": {"unique_files": 0, "hash_opens": 0, "parse_opens": 0},
            "scene_summaries": {
                "unique_files": 0,
                "hash_opens": 0,
                "parse_opens": 0,
            },
            "source_frames_jsonl": {
                "unique_files": 0,
                "hash_opens": 0,
                "parse_opens": 0,
                "json_records_scanned": 0,
                "requested_records": 0,
                "matched_records": 0,
            },
            "source_code": {"unique_files": 0, "hash_opens": 0, "parse_opens": 0},
        },
        "role_namespace": {
            "physical_dataset_role_train_frame_records": 0,
            "physical_dataset_nontrain_frame_records": 0,
            "legacy_source_split_frame_records": {},
            "legacy_source_split_used_for_inclusion": False,
        },
        "forbidden": {
            "original_monolithic_panel_byte_opens": 0,
            "rgb_byte_opens": 0,
            "label_shard_byte_opens": 0,
            "model_checkpoint_byte_opens": 0,
            "model_output_byte_opens": 0,
            "g2_payload_opens": 0,
            "sealed_manifest_or_payload_opens": 0,
            "physical_dataset_nontrain_role_metadata_records": 0,
            "non_fit_panel_rows_used": 0,
        },
    }


def _require_authorization(
    authorization: str, ledger: dict[str, Any]
) -> None:
    ledger["authorization_checked"] = True
    if str(authorization) != ROLE_NAMESPACE_AMENDMENT_SHA256:
        raise PermissionError(
            "audit is not authorized: pass the exact frozen role-namespace "
            "amendment SHA-256"
        )
    ledger["authorized"] = True


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _strict_existing_path(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    if not _is_within(resolved, root_resolved):
        raise PermissionError(f"{label} escapes its authorized metadata root")
    lowered = str(resolved).lower()
    forbidden_fragments = (
        "/sealed",
        "sealed_",
        "/g2/",
        "g2_payload",
        "checkpoint_selection",
    )
    if any(fragment in lowered for fragment in forbidden_fragments):
        raise PermissionError(f"{label} names a forbidden role or payload")
    if resolved.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".npz",
        ".npy",
        ".pt",
        ".pth",
        ".ckpt",
    }:
        raise PermissionError(f"{label} is not an authorized metadata file")
    return resolved


def _sha256_file(
    path: Path, *, ledger: dict[str, Any], bucket: str
) -> str:
    ledger["metadata"][bucket]["hash_opens"] += 1
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(
    path: Path, *, ledger: dict[str, Any], bucket: str
) -> dict[str, Any]:
    ledger["metadata"][bucket]["parse_opens"] += 1
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_hash_records(
    *, ledger: dict[str, Any]
) -> list[dict[str, str]]:
    return [
        {
            "path": str(path),
            "sha256": _sha256_file(
                path,
                ledger=ledger,
                bucket="source_code",
            ),
        }
        for path in SOURCE_PATHS
    ]


def _require_unchanged_hash(before: str, after: str, *, label: str) -> None:
    if str(after) != str(before):
        raise ValueError(f"{label} changed while the audit parsed metadata")


def _vector_from_xyz(value: Mapping[str, Any], *, name: str) -> tuple[float, ...]:
    if not isinstance(value, Mapping) or set(("x", "y", "z")) - set(value):
        raise ValueError(f"source frame lacks {name} xyz")
    vector = tuple(float(value[axis]) for axis in ("x", "y", "z"))
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"source frame has non-finite {name}")
    return vector


def _vector3(value: Sequence[Any], *, name: str) -> tuple[float, ...]:
    vector = tuple(float(component) for component in value)
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        raise ValueError(f"source frame has malformed {name}")
    return vector


def _record_key(row: Mapping[str, Any], *, side: str) -> dict[str, Any]:
    if side not in {"current", "next"}:
        raise ValueError("panel side must be current or next")
    return {
        "family": str(row["family"]),
        "physical_dataset_role": str(row["dataset_role"]),
        "scene_id": str(row["scene_id"]),
        "global_row": int(row["global_row"]),
        "side": side,
        "frame_index": int(row[f"{side}_frame_index"]),
        "env_index": int(row["env_index"]),
        "timestamp_ns": int(row[f"{side}_timestamp_ns"]),
        "episode_id": str(row["episode_id"]),
        "reset_count": int(row["reset_count"]),
        "episode_step": int(row[f"{side}_episode_step"]),
        "image_path_metadata_only": str(row[f"{side}_image_path"]),
        "image_sha256_commitment_only": str(row[f"{side}_image_sha256"]),
    }


def _frame_identity(key: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(key["scene_id"]),
        int(key["frame_index"]),
        int(key["env_index"]),
        int(key["timestamp_ns"]),
    )


def _validate_fit_panel(
    panel: Mapping[str, Any], *, panel_file_sha256: str
) -> list[dict[str, Any]]:
    if panel_file_sha256 != FIT_PANEL_FILE_SHA256:
        raise ValueError("frozen fit-only panel file SHA-256 mismatch")
    expected_keys = {
        "schema",
        "created_at_utc",
        "amendment_sha256",
        "source_panel",
        "family_order",
        "fit",
        "access_ledger",
        "interpretation_limits",
        "content_sha256",
    }
    if set(panel) != expected_keys:
        raise ValueError("frozen fit-only panel schema keys changed")
    if str(panel.get("schema", "")) != "lewm_go2_n32_pose_projection_fit_panel_v1":
        raise ValueError("frozen fit-only panel schema changed")
    if str(panel.get("amendment_sha256", "")) != AMENDMENT_SHA256:
        raise ValueError("frozen fit-only panel amendment binding mismatch")
    if str(panel.get("content_sha256", "")) != FIT_PANEL_CONTENT_SHA256:
        raise ValueError("frozen fit-only panel declared content SHA-256 mismatch")
    content = dict(panel)
    content.pop("content_sha256", None)
    if canonical_json_sha256(content) != FIT_PANEL_CONTENT_SHA256:
        raise ValueError("frozen fit-only panel canonical content SHA-256 mismatch")
    if panel.get("family_order") != list(FAMILIES):
        raise ValueError("frozen fit-only panel family order changed")

    source_panel = panel.get("source_panel")
    expected_source_panel = {
        "path": EXPECTED_SOURCE_PANEL_PATH,
        "file_sha256_before_parse": SOURCE_PANEL_FILE_SHA256,
        "file_sha256_after_parse": SOURCE_PANEL_FILE_SHA256,
        "content_sha256": SOURCE_PANEL_CONTENT_SHA256,
    }
    if source_panel != expected_source_panel:
        raise ValueError("frozen fit-only panel source lineage changed")
    expected_extraction_access = {
        "source_panel_byte_opens": 2,
        "source_panel_parse_count": 1,
        "fit_rows_copied": EXPECTED_FIT_ROWS,
        "fit_frames_represented": EXPECTED_FRAME_RECORDS,
        "non_fit_rows_copied": 0,
        "rgb_byte_opens": 0,
        "label_shard_byte_opens": 0,
        "model_checkpoint_or_output_opens": 0,
        "g2_payload_opens": 0,
        "sealed_manifest_or_payload_opens": 0,
    }
    if panel.get("access_ledger") != expected_extraction_access:
        raise ValueError("frozen fit-only panel extraction access does not reconcile")
    expected_limits = {
        "is_research_result": False,
        "can_pass_n32": False,
        "can_pass_g2": False,
        "can_license_runtime": False,
    }
    if panel.get("interpretation_limits") != expected_limits:
        raise ValueError("frozen fit-only panel interpretation limits changed")

    fit = panel.get("fit")
    if not isinstance(fit, Mapping):
        raise ValueError("frozen fit-only panel lacks fit metadata")
    if set(fit) != {"row_count", "frame_count", "rows_sha256", "rows"}:
        raise ValueError("frozen fit-only panel fit schema changed")
    rows = fit.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_FIT_ROWS:
        raise ValueError("fit panel must contain exactly 160 transition rows")
    if int(fit.get("row_count", -1)) != EXPECTED_FIT_ROWS:
        raise ValueError("fit row count commitment mismatch")
    if int(fit.get("frame_count", -1)) != EXPECTED_FRAME_RECORDS:
        raise ValueError("fit frame count commitment mismatch")
    if str(fit.get("rows_sha256", "")) != FIT_ROWS_SHA256:
        raise ValueError("fit rows declared SHA-256 mismatch")
    if canonical_json_sha256(rows) != FIT_ROWS_SHA256:
        raise ValueError("fit rows canonical SHA-256 mismatch")

    family_counts: Counter[str] = Counter()
    frame_records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("fit row is not an object")
        family = str(row.get("family", ""))
        if family not in FAMILIES:
            raise ValueError("fit row has an unregistered family")
        if str(row.get("dataset_role", "")) != "train":
            raise PermissionError("fit row is not train-role metadata")
        family_counts[family] += 1
        for side in ("current", "next"):
            key = _record_key(row, side=side)
            key["panel_row_index"] = row_index
            frame_records.append(key)
    if family_counts != Counter({family: 32 for family in FAMILIES}):
        raise ValueError("fit panel does not contain 32 transitions per family")
    identities = [_frame_identity(record) for record in frame_records]
    if len(frame_records) != EXPECTED_FRAME_RECORDS:
        raise ValueError("fit panel did not expand to exactly 320 frames")
    if len(set(identities)) != EXPECTED_FRAME_RECORDS:
        raise ValueError("fit panel contains duplicate frame records")
    family_rank = {family: index for index, family in enumerate(FAMILIES)}
    frame_records.sort(
        key=lambda record: (
            family_rank[record["family"]],
            record["scene_id"],
            record["global_row"],
            0 if record["side"] == "current" else 1,
        )
    )
    return frame_records


def _summary_path_for_record(record: Mapping[str, Any]) -> Path:
    image_path = Path(str(record["image_path_metadata_only"])).resolve(strict=False)
    if image_path.parent.name != "rgb":
        raise PermissionError("committed image metadata is not under a V04 rgb directory")
    summary = image_path.parent.parent / "summary.json"
    return _strict_existing_path(summary, SUMMARY_ROOT, label="V04 scene summary")


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[Path, str, str]:
    if not records:
        raise ValueError("summary validation received no requested records")
    expected_family = str(records[0]["family"])
    expected_scene = str(records[0]["scene_id"])
    if any(
        str(record["family"]) != expected_family
        or str(record["scene_id"]) != expected_scene
        for record in records
    ):
        raise ValueError("one summary was assigned records from multiple scenes")
    summary_root = SUMMARY_ROOT.resolve(strict=True)
    try:
        summary_key = str(summary_path.resolve(strict=False).relative_to(summary_root))
    except ValueError as exc:
        raise PermissionError("V04 summary escapes the frozen summary root") from exc
    expected_legacy_split = EXPECTED_LEGACY_SOURCE_SPLIT.get(summary_key)
    if expected_legacy_split is None:
        raise ValueError("V04 summary path lacks frozen legacy-split provenance")
    actual_legacy_split = str(summary.get("split", ""))
    if actual_legacy_split != expected_legacy_split:
        raise ValueError("V04 summary legacy source split changed")
    if str(summary.get("family", "")) != expected_family:
        raise ValueError("V04 summary family disagrees with panel")
    if str(summary.get("scene_id", "")) != expected_scene:
        raise ValueError("V04 summary scene disagrees with panel")
    if str(summary.get("render_status", "")) != "complete":
        raise ValueError("V04 summary is not complete")
    if bool(summary.get("g2_model_outputs_opened", False)):
        raise PermissionError("V04 summary declares G2 model-output access")
    source = summary.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("V04 summary lacks source metadata")
    source_frames = source.get("frames_jsonl")
    if not isinstance(source_frames, Mapping):
        raise ValueError("V04 summary lacks source frames commitment")
    source_sha = str(source_frames.get("sha256", ""))
    if len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("V04 source frames SHA-256 is malformed")
    source_path = _strict_existing_path(
        Path(str(source_frames.get("path", ""))),
        FRAMES_ROOT / actual_legacy_split,
        label="legacy source frames JSONL",
    )
    if source_path.name != "frames.jsonl" or source_path.suffix != ".jsonl":
        raise PermissionError("legacy source metadata must be named frames.jsonl")

    rendered = summary.get("rendered_frames")
    if not isinstance(rendered, list):
        raise ValueError("V04 summary lacks rendered-frame metadata")
    rendered_by_key: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for item in rendered:
        if not isinstance(item, Mapping):
            raise ValueError("V04 rendered-frame metadata is malformed")
        key = (int(item["frame_index"]), int(item["env_index"]))
        rendered_by_key.setdefault(key, []).append(item)
    for record in records:
        key = (int(record["frame_index"]), int(record["env_index"]))
        matches = rendered_by_key.get(key, [])
        if len(matches) != 1:
            raise ValueError("requested rendered-frame key did not match exactly once")
        match = matches[0]
        if int(match.get("timestamp_ns", -1)) != int(record["timestamp_ns"]):
            raise ValueError("rendered-frame timestamp disagrees with panel")
        if str(match.get("image_sha256", "")) != str(
            record["image_sha256_commitment_only"]
        ):
            raise ValueError("rendered-frame image commitment disagrees with panel")
        expected_parent = summary_path.parent / "rgb"
        image_parent = Path(str(record["image_path_metadata_only"])).parent.resolve(
            strict=False
        )
        if image_parent != expected_parent.resolve(strict=True):
            raise PermissionError("committed image metadata escapes its V04 scene")
    return source_path, source_sha, actual_legacy_split


def _validate_and_extract_source_frame(
    frame: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    expected_legacy_source_split: str = "train",
) -> dict[str, Any]:
    if int(frame.get("frame_index", -1)) != int(record["frame_index"]):
        raise ValueError("source-frame index disagrees with requested record")
    if int(frame.get("env_index", -1)) != int(record["env_index"]):
        raise ValueError("source-frame environment disagrees with requested record")
    if int(frame.get("timestamp_ns", -1)) != int(record["timestamp_ns"]):
        raise ValueError("source-frame timestamp disagrees with requested record")
    episode = frame.get("episode")
    if not isinstance(episode, Mapping):
        raise ValueError("source frame lacks legacy episode provenance")
    if str(episode.get("split", "")) != str(expected_legacy_source_split):
        raise ValueError("source-frame legacy source split changed")
    if str(episode.get("episode_id", "")) != str(record["episode_id"]):
        raise ValueError("source-frame episode ID disagrees with panel")
    if int(episode.get("reset_count", -1)) != int(record["reset_count"]):
        raise ValueError("source-frame reset count disagrees with panel")
    if int(episode.get("episode_step", -1)) != int(record["episode_step"]):
        raise ValueError("source-frame episode step disagrees with panel")
    base_pose = frame.get("base_pose_world")
    base_rpy = frame.get("base_rpy_rad")
    camera_pose = frame.get("camera_pose_world")
    if not isinstance(base_pose, Mapping) or not isinstance(base_rpy, Mapping):
        raise ValueError("source frame lacks base pose metadata")
    if not isinstance(camera_pose, Mapping):
        raise ValueError("source frame lacks camera pose metadata")
    base_position = _vector_from_xyz(base_pose.get("position"), name="base position")
    base_yaw = float(base_rpy.get("yaw"))
    if not math.isfinite(base_yaw):
        raise ValueError("source frame has non-finite base yaw")
    camera_position = _vector3(camera_pose.get("position"), name="camera position")
    camera_lookat = _vector3(camera_pose.get("lookat"), name="camera lookat")
    camera_up = _vector3(camera_pose.get("up"), name="camera up")
    return {
        "base_position_world": base_position,
        "base_yaw_rad": base_yaw,
        "camera_position_world": camera_position,
        "camera_lookat_world": camera_lookat,
        "camera_up_world": camera_up,
    }


def _scan_requested_source_frames(
    source_path: Path,
    records: Sequence[dict[str, Any]],
    *,
    ledger: dict[str, Any],
    expected_legacy_source_split: str = "train",
) -> dict[tuple[int, int], dict[str, Any]]:
    requested = {
        (int(record["frame_index"]), int(record["env_index"])): record
        for record in records
    }
    if len(requested) != len(records):
        raise ValueError("one source file received duplicate requested frame keys")
    matches: dict[tuple[int, int], list[dict[str, Any]]] = {
        key: [] for key in requested
    }
    ledger["metadata"]["source_frames_jsonl"]["parse_opens"] += 1
    with source_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            ledger["metadata"]["source_frames_jsonl"]["json_records_scanned"] += 1
            frame = json.loads(line)
            if not isinstance(frame, Mapping):
                raise ValueError(f"source frame {source_path}:{line_number} is not an object")
            key = (int(frame.get("frame_index", -1)), int(frame.get("env_index", -1)))
            if key in matches:
                matches[key].append(
                    _validate_and_extract_source_frame(
                        frame,
                        requested[key],
                        expected_legacy_source_split=expected_legacy_source_split,
                    )
                )
    duplicate_count = sum(max(0, len(values) - 1) for values in matches.values())
    missing_count = sum(not values for values in matches.values())
    if duplicate_count or missing_count:
        raise ValueError(
            "requested source-frame keys did not match exactly once: "
            f"missing={missing_count} duplicate={duplicate_count}"
        )
    ledger["metadata"]["source_frames_jsonl"]["requested_records"] += len(records)
    ledger["metadata"]["source_frames_jsonl"]["matched_records"] += len(records)
    return {key: values[0] for key, values in matches.items()}


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise ValueError("git HEAD is malformed")
    return value


def _exclusive_atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable audit output already exists: {path}")
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_access_ledger(
    ledger: Mapping[str, Any],
    *,
    source_frames_file_count: int,
) -> dict[str, Any]:
    expected_metadata = {
        "binding": {"unique_files": 1, "hash_opens": 2, "parse_opens": 0},
        "fit_panel_amendment": {
            "unique_files": 1,
            "hash_opens": 2,
            "parse_opens": 0,
        },
        "superseded_scope_amendment": {
            "unique_files": 1,
            "hash_opens": 2,
            "parse_opens": 0,
        },
        "role_namespace_amendment": {
            "unique_files": 1,
            "hash_opens": 2,
            "parse_opens": 0,
        },
        "fit_panel": {"unique_files": 1, "hash_opens": 2, "parse_opens": 1},
        "scene_summaries": {
            "unique_files": len(EXPECTED_SUMMARY_SHA256),
            "hash_opens": 2 * len(EXPECTED_SUMMARY_SHA256),
            "parse_opens": len(EXPECTED_SUMMARY_SHA256),
        },
        "source_code": {
            "unique_files": len(SOURCE_PATHS),
            "hash_opens": 2 * len(SOURCE_PATHS),
            "parse_opens": 0,
        },
    }
    if ledger.get("authorization_checked") is not True or ledger.get(
        "authorized"
    ) is not True:
        raise PermissionError("audit access ledger lacks authorization")
    metadata = ledger.get("metadata")
    role_namespace = ledger.get("role_namespace")
    forbidden = ledger.get("forbidden")
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(role_namespace, Mapping)
        or not isinstance(forbidden, Mapping)
    ):
        raise ValueError("audit access ledger structure changed")
    if set(metadata) != set(expected_metadata) | {"source_frames_jsonl"}:
        raise ValueError("audit metadata access bucket set changed")
    for bucket, expected in expected_metadata.items():
        if metadata.get(bucket) != expected:
            raise ValueError(f"audit metadata access does not reconcile: {bucket}")
    frames = metadata.get("source_frames_jsonl")
    expected_frames = {
        "unique_files": int(source_frames_file_count),
        "hash_opens": 2 * int(source_frames_file_count),
        "parse_opens": int(source_frames_file_count),
        "requested_records": EXPECTED_FRAME_RECORDS,
        "matched_records": EXPECTED_FRAME_RECORDS,
    }
    if not isinstance(frames, Mapping):
        raise ValueError("audit source-frame access ledger is missing")
    if set(frames) != set(expected_frames) | {"json_records_scanned"}:
        raise ValueError("audit source-frame access bucket schema changed")
    for name, expected in expected_frames.items():
        if int(frames.get(name, -1)) != expected:
            raise ValueError(f"audit source-frame access does not reconcile: {name}")
    if int(frames.get("json_records_scanned", 0)) < EXPECTED_FRAME_RECORDS:
        raise ValueError("audit scanned fewer source records than it matched")
    expected_role_namespace = {
        "physical_dataset_role_train_frame_records": EXPECTED_FRAME_RECORDS,
        "physical_dataset_nontrain_frame_records": 0,
        "legacy_source_split_frame_records": dict(
            EXPECTED_LEGACY_SOURCE_SPLIT_FRAME_COUNTS
        ),
        "legacy_source_split_used_for_inclusion": False,
    }
    if role_namespace != expected_role_namespace:
        raise ValueError("audit role-namespace ledger does not reconcile")
    expected_forbidden = {
        "original_monolithic_panel_byte_opens": 0,
        "rgb_byte_opens": 0,
        "label_shard_byte_opens": 0,
        "model_checkpoint_byte_opens": 0,
        "model_output_byte_opens": 0,
        "g2_payload_opens": 0,
        "sealed_manifest_or_payload_opens": 0,
        "physical_dataset_nontrain_role_metadata_records": 0,
        "non_fit_panel_rows_used": 0,
    }
    if forbidden != expected_forbidden:
        raise PermissionError("forbidden access ledger is nonzero")
    byte_open_count = sum(
        int(bucket["hash_opens"]) + int(bucket["parse_opens"])
        for bucket in metadata.values()
    )
    return {
        "schema": "lewm_go2_n32_pose_projection_access_reconciliation_v1",
        "metadata_bucket_count": len(metadata),
        "metadata_unique_file_count": sum(
            int(bucket["unique_files"]) for bucket in metadata.values()
        ),
        "metadata_byte_open_count": byte_open_count,
        "requested_source_record_count": int(frames["requested_records"]),
        "matched_source_record_count": int(frames["matched_records"]),
        "physical_dataset_role_train_frame_count": int(
            role_namespace["physical_dataset_role_train_frame_records"]
        ),
        "legacy_source_split_frame_counts": dict(
            role_namespace["legacy_source_split_frame_records"]
        ),
        "forbidden_counter_count": len(forbidden),
        "all_forbidden_counts_zero": True,
        "exact_bucket_and_count_reconciliation_passes": True,
    }


def run_authoritative_audit(
    *,
    authorization: str,
    ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run after exact authorization by the newest role-namespace amendment."""

    access = new_access_ledger() if ledger is None else ledger
    _require_authorization(authorization, access)
    if (
        len(EXPECTED_SUMMARY_SHA256) != EXPECTED_SUMMARY_COUNT
        or set(EXPECTED_SUMMARY_SHA256) != set(EXPECTED_LEGACY_SOURCE_SPLIT)
    ):
        raise ValueError("frozen summary hash and legacy-split maps disagree")
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"immutable audit output already exists: {OUTPUT_PATH}")

    access["metadata"]["source_code"]["unique_files"] = len(SOURCE_PATHS)
    source_hashes_pre = _source_hash_records(ledger=access)

    binding_path = BINDING_PATH.resolve(strict=True)
    amendment_path = AMENDMENT_PATH.resolve(strict=True)
    superseded_scope_amendment_path = (
        SUPERSEDED_SCOPE_AMENDMENT_PATH.resolve(strict=True)
    )
    role_namespace_amendment_path = (
        ROLE_NAMESPACE_AMENDMENT_PATH.resolve(strict=True)
    )
    fit_panel_path = FIT_PANEL_PATH.resolve(strict=True)
    access["metadata"]["binding"]["unique_files"] = 1
    access["metadata"]["fit_panel_amendment"]["unique_files"] = 1
    access["metadata"]["superseded_scope_amendment"]["unique_files"] = 1
    access["metadata"]["role_namespace_amendment"]["unique_files"] = 1
    access["metadata"]["fit_panel"]["unique_files"] = 1
    binding_pre = _sha256_file(binding_path, ledger=access, bucket="binding")
    if binding_pre != BINDING_SHA256:
        raise ValueError("camera-pose audit binding SHA-256 mismatch")
    amendment_pre = _sha256_file(
        amendment_path,
        ledger=access,
        bucket="fit_panel_amendment",
    )
    if amendment_pre != AMENDMENT_SHA256:
        raise ValueError("camera-pose audit fit-panel amendment SHA-256 mismatch")
    superseded_scope_amendment_pre = _sha256_file(
        superseded_scope_amendment_path,
        ledger=access,
        bucket="superseded_scope_amendment",
    )
    if superseded_scope_amendment_pre != SUPERSEDED_SCOPE_AMENDMENT_SHA256:
        raise ValueError("camera-pose audit superseded-scope SHA-256 mismatch")
    role_namespace_amendment_pre = _sha256_file(
        role_namespace_amendment_path,
        ledger=access,
        bucket="role_namespace_amendment",
    )
    if role_namespace_amendment_pre != ROLE_NAMESPACE_AMENDMENT_SHA256:
        raise ValueError("camera-pose audit role-namespace SHA-256 mismatch")
    fit_panel_pre = _sha256_file(
        fit_panel_path,
        ledger=access,
        bucket="fit_panel",
    )
    if fit_panel_pre != FIT_PANEL_FILE_SHA256:
        raise ValueError("camera-pose audit fit-only panel SHA-256 mismatch")
    fit_panel = _load_json(
        fit_panel_path,
        ledger=access,
        bucket="fit_panel",
    )
    records = _validate_fit_panel(
        fit_panel,
        panel_file_sha256=fit_panel_pre,
    )

    records_by_summary: dict[Path, list[dict[str, Any]]] = {}
    for record in records:
        records_by_summary.setdefault(_summary_path_for_record(record), []).append(record)
    access["metadata"]["scene_summaries"]["unique_files"] = len(records_by_summary)
    summary_root = SUMMARY_ROOT.resolve(strict=True)
    observed_summary_keys = {
        str(path.relative_to(summary_root)) for path in records_by_summary
    }
    if observed_summary_keys != set(EXPECTED_SUMMARY_SHA256):
        raise ValueError("fit panel summary set differs from the frozen allowlist")

    summary_sources: dict[Path, dict[str, Any]] = {}
    summary_provenance: list[dict[str, Any]] = []
    legacy_split_frame_counts: Counter[str] = Counter()
    for summary_path in sorted(records_by_summary):
        summary_pre = _sha256_file(
            summary_path, ledger=access, bucket="scene_summaries"
        )
        summary_key = str(summary_path.relative_to(summary_root))
        if summary_pre != EXPECTED_SUMMARY_SHA256[summary_key]:
            raise ValueError("V04 scene summary differs from its frozen SHA-256")
        summary = _load_json(
            summary_path, ledger=access, bucket="scene_summaries"
        )
        source_path, source_commitment, legacy_source_split = _validate_summary(
            summary,
            summary_path=summary_path,
            records=records_by_summary[summary_path],
        )
        summary_post = _sha256_file(
            summary_path, ledger=access, bucket="scene_summaries"
        )
        if summary_post != summary_pre:
            raise ValueError("V04 summary changed while the audit parsed it")
        for record in records_by_summary[summary_path]:
            record["legacy_source_split"] = legacy_source_split
        existing = summary_sources.get(source_path)
        if existing is not None and existing["commitment"] != source_commitment:
            raise ValueError("one source frames file has conflicting commitments")
        if (
            existing is not None
            and existing["legacy_source_split"] != legacy_source_split
        ):
            raise ValueError("one source frames file has conflicting legacy splits")
        entry = summary_sources.setdefault(
            source_path,
            {
                "commitment": source_commitment,
                "legacy_source_split": legacy_source_split,
                "records": [],
                "summaries": [],
            },
        )
        entry["records"].extend(records_by_summary[summary_path])
        entry["summaries"].append(str(summary_path))
        legacy_split_frame_counts[legacy_source_split] += len(
            records_by_summary[summary_path]
        )
        summary_provenance.append(
            {
                "path": str(summary_path),
                "sha256_before_parse": summary_pre,
                "sha256_after_parse": summary_post,
                "expected_sha256": EXPECTED_SUMMARY_SHA256[summary_key],
                "legacy_source_split": legacy_source_split,
                "requested_frame_count": len(records_by_summary[summary_path]),
                "source_frames_jsonl_path": str(source_path),
                "source_frames_jsonl_committed_sha256": source_commitment,
            }
        )

    if dict(sorted(legacy_split_frame_counts.items())) != dict(
        sorted(EXPECTED_LEGACY_SOURCE_SPLIT_FRAME_COUNTS.items())
    ):
        raise ValueError("legacy source-split frame counts changed")
    physical_train_frame_count = sum(
        record["physical_dataset_role"] == "train" for record in records
    )
    physical_nontrain_frame_count = len(records) - physical_train_frame_count
    access["role_namespace"] = {
        "physical_dataset_role_train_frame_records": physical_train_frame_count,
        "physical_dataset_nontrain_frame_records": physical_nontrain_frame_count,
        "legacy_source_split_frame_records": dict(
            EXPECTED_LEGACY_SOURCE_SPLIT_FRAME_COUNTS
        ),
        "legacy_source_split_used_for_inclusion": False,
    }

    access["metadata"]["source_frames_jsonl"]["unique_files"] = len(summary_sources)
    source_pose_by_record: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_provenance: list[dict[str, Any]] = []
    for source_path in sorted(summary_sources):
        entry = summary_sources[source_path]
        source_pre = _sha256_file(
            source_path, ledger=access, bucket="source_frames_jsonl"
        )
        if source_pre != entry["commitment"]:
            raise ValueError("source frames JSONL differs from its V04 commitment")
        extracted = _scan_requested_source_frames(
            source_path,
            entry["records"],
            ledger=access,
            expected_legacy_source_split=entry["legacy_source_split"],
        )
        source_post = _sha256_file(
            source_path, ledger=access, bucket="source_frames_jsonl"
        )
        if source_post != source_pre:
            raise ValueError("source frames JSONL changed while the audit parsed it")
        for record in entry["records"]:
            key = (int(record["frame_index"]), int(record["env_index"]))
            identity = _frame_identity(record)
            if identity in source_pose_by_record:
                raise ValueError("duplicate exact record identity after source matching")
            source_pose_by_record[identity] = extracted[key]
        source_provenance.append(
            {
                "path": str(source_path),
                "sha256_before_parse": source_pre,
                "sha256_after_parse": source_post,
                "requested_frame_count": len(entry["records"]),
                "legacy_source_split": entry["legacy_source_split"],
                "summary_paths": sorted(entry["summaries"]),
            }
        )

    if len(source_pose_by_record) != EXPECTED_FRAME_RECORDS:
        raise ValueError("source matching did not yield exactly 320 frame poses")
    comparisons_by_family: dict[str, list[ProjectionComparison]] = {
        family: [] for family in FAMILIES
    }
    frame_reports: list[dict[str, Any]] = []
    for record in records:
        pose = source_pose_by_record[_frame_identity(record)]
        camera = reconstruct_yaw_aligned_camera(**pose)
        comparison = compare_projection(camera)
        comparisons_by_family[record["family"]].append(comparison)
        frame_reports.append(
            {
                "record_key": dict(record),
                "geometry": comparison.metrics,
            }
        )
    if any(len(comparisons_by_family[family]) != 64 for family in FAMILIES):
        raise ValueError("audit did not produce 64 frame comparisons per family")

    binding_post = _sha256_file(binding_path, ledger=access, bucket="binding")
    amendment_post = _sha256_file(
        amendment_path,
        ledger=access,
        bucket="fit_panel_amendment",
    )
    superseded_scope_amendment_post = _sha256_file(
        superseded_scope_amendment_path,
        ledger=access,
        bucket="superseded_scope_amendment",
    )
    role_namespace_amendment_post = _sha256_file(
        role_namespace_amendment_path,
        ledger=access,
        bucket="role_namespace_amendment",
    )
    fit_panel_post = _sha256_file(
        fit_panel_path,
        ledger=access,
        bucket="fit_panel",
    )
    for label, before, after in (
        ("original audit binding", binding_pre, binding_post),
        ("fit-panel amendment", amendment_pre, amendment_post),
        (
            "superseded train-source scope amendment",
            superseded_scope_amendment_pre,
            superseded_scope_amendment_post,
        ),
        (
            "role-namespace amendment",
            role_namespace_amendment_pre,
            role_namespace_amendment_post,
        ),
        ("fit-only panel", fit_panel_pre, fit_panel_post),
    ):
        _require_unchanged_hash(before, after, label=label)
    source_hashes_post = _source_hash_records(ledger=access)
    if source_hashes_post != source_hashes_pre:
        raise ValueError("audit source changed while metadata was parsed")
    duplicate_exact_record_count = EXPECTED_FRAME_RECORDS - len(
        {_frame_identity(record) for record in records}
    )
    access_reconciliation = _validate_access_ledger(
        access,
        source_frames_file_count=len(summary_sources),
    )
    core: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "fit_train_metadata_only_no_model_evaluation",
        "bindings": {
            "original_audit_binding": {
                "path": str(binding_path),
                "sha256_before_parse": binding_pre,
                "sha256_after_parse": binding_post,
            },
            "fit_panel_amendment": {
                "path": str(amendment_path),
                "sha256_before_parse": amendment_pre,
                "sha256_after_parse": amendment_post,
            },
            "superseded_scope_amendment": {
                "path": str(superseded_scope_amendment_path),
                "sha256_before_parse": superseded_scope_amendment_pre,
                "sha256_after_parse": superseded_scope_amendment_post,
                "status": "superseded_before_execution",
            },
            "role_namespace_amendment": {
                "path": str(role_namespace_amendment_path),
                "sha256_before_parse": role_namespace_amendment_pre,
                "sha256_after_parse": role_namespace_amendment_post,
            },
        },
        "fit_panel": {
            "path": str(fit_panel_path),
            "file_sha256_before_parse": fit_panel_pre,
            "file_sha256_after_parse": fit_panel_post,
            "content_sha256": FIT_PANEL_CONTENT_SHA256,
            "amendment_sha256": AMENDMENT_SHA256,
            "fit_rows_sha256": FIT_ROWS_SHA256,
            "fit_transition_count": EXPECTED_FIT_ROWS,
            "fit_frame_record_count": EXPECTED_FRAME_RECORDS,
            "source_panel_identity_from_fit_artifact": fit_panel["source_panel"],
            "original_monolithic_panel_byte_opens_by_audit_runner": 0,
        },
        "role_namespace": {
            "governing_role_field": "physical_dataset_role",
            "physical_dataset_role_train_frame_records": physical_train_frame_count,
            "physical_dataset_nontrain_frame_records": physical_nontrain_frame_count,
            "legacy_source_split_by_summary_path": dict(
                EXPECTED_LEGACY_SOURCE_SPLIT
            ),
            "legacy_source_split_frame_records": dict(
                EXPECTED_LEGACY_SOURCE_SPLIT_FRAME_COUNTS
            ),
            "legacy_source_split_used_for_inclusion": False,
            "all_five_families_frame_records": 64,
        },
        "geometry_contract": geometry_contract(),
        "record_integrity": {
            "physical_dataset_train_record_count": len(records),
            "physical_dataset_nontrain_record_count": 0,
            "requested_record_count": EXPECTED_FRAME_RECORDS,
            "matched_record_count": len(source_pose_by_record),
            "missing_exact_record_count": 0,
            "duplicate_exact_record_count": duplicate_exact_record_count,
            "family_order": list(FAMILIES),
            "family_frame_counts": {family: 64 for family in FAMILIES},
        },
        "aggregate": summarize_frame_comparisons(
            [comparison for family in FAMILIES for comparison in comparisons_by_family[family]]
        ),
        "families": {
            family: summarize_frame_comparisons(comparisons_by_family[family])
            for family in FAMILIES
        },
        "ordering_decision": ordering_decision(comparisons_by_family),
        "frames": frame_reports,
        "input_provenance": {
            "scene_summaries": summary_provenance,
            "source_frames_jsonl": source_provenance,
        },
        "source_hashes": [
            {
                "path": before["path"],
                "sha256_before_parse": before["sha256"],
                "sha256_after_parse": after["sha256"],
            }
            for before, after in zip(source_hashes_pre, source_hashes_post, strict=True)
        ],
        "git": {"head": _git_head()},
        "artifact_access_ledger": access,
        "access_reconciliation": access_reconciliation,
        "interpretation_limits": {
            "is_model_evaluation": False,
            "can_pass_n32": False,
            "can_pass_g2": False,
            "can_pass_runtime_gate": False,
            "camera_centered_polar_factorization_remains_distinct": True,
        },
    }
    if duplicate_exact_record_count:
        raise ValueError("duplicate exact records reached result construction")
    if int(core["aggregate"]["query_count_per_frame"]) != QUERY_COUNT:
        raise AssertionError("aggregate query shape changed")
    result = dict(core)
    result["content_sha256"] = canonical_json_sha256(core)
    _exclusive_atomic_write_json(OUTPUT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        required=True,
        help="Exact SHA-256 of the frozen role-namespace amendment.",
    )
    args = parser.parse_args()
    result = run_authoritative_audit(authorization=args.authorization)
    output_sha256 = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "file_sha256": output_sha256,
                "content_sha256": result["content_sha256"],
                "frame_count": result["record_integrity"]["matched_record_count"],
                "material_dynamic_pose_mismatch": result["ordering_decision"][
                    "material_dynamic_pose_mismatch"
                ],
                "next_intervention": result["ordering_decision"]["next_intervention"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
