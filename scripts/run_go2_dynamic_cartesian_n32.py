#!/usr/bin/env python3
"""Run the frozen dynamic-Cartesian N32 occupancy qualification.

The runner is deliberately not a general trainer.  It consumes only the
registered train-role panel and attitude sidecar, runs the two fixed optimizer
branches, and publishes one immutable result.  Holdout payloads remain closed
unless a branch passes the complete terminal fit gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import secrets
import stat
import subprocess
import sys
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "lewm_worlds"))

from lewm.benchmarks import go2_dynamic_cartesian_n32 as contract  # noqa: E402
from lewm.benchmarks.go2_physical_micro_overfit import (  # noqa: E402
    DISTANCE_BINS_M,
    TRAINING_WEIGHTS,
    attach_role_global_shuffle,
    attach_same_scene_wrong_view,
    empty_raw_accumulator,
    finalize_raw_accumulator,
    frame_records,
    update_raw_accumulator,
    validate_panel_manifest,
)
from lewm.datasets.go2_attitude_sidecar import (  # noqa: E402
    FROZEN_BUILD_CONTRACT,
    canonical_json_sha256 as sidecar_json_sha256,
    load_attitude_sidecar_roles,
    row_identity_sha256,
)
from lewm.models.egomotion_bev_jepa import (  # noqa: E402
    DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    EgomotionBevJepa,
    build_projective_query_support_contract,
    validate_projective_query_support_binding,
)


PANEL_PATH = REPOSITORY_ROOT / ".generated/go2_physical_micro_overfit/patch7_v1/panel.json"
PANEL_FILE_SHA256 = "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c"
PANEL_CONTENT_SHA256 = "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f"
PANEL_ROWS_SHA256 = {
    "fit": "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d",
    "same_scene_holdout": "d32713086c042d20f94825aa362c27a07bef6fd0e0cce0aa5846bb67bf8dc465",
    "cross_scene_holdout": "3565f7f7844f3aeee28b0433aa6dc77d553a9ebb831cf9af20b6d392c5416817",
}
DATASET_MANIFEST_PATH = REPOSITORY_ROOT / ".generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/dataset_manifest.json"
DATASET_MANIFEST_FILE_SHA256 = "ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180"
SIDECAR_MANIFEST_PATH = REPOSITORY_ROOT / ".generated/go2_attitude_sidecar/dynamic_cartesian_v1/manifest.json"
SIDECAR_MANIFEST_FILE_SHA256 = "6fafa417b4f724a0fdf32cfde5740025c3117e4c0b43231fe9ebe94bd9eff529"
SIDECAR_MANIFEST_CONTENT_SHA256 = "6f1ef7d9ac0c55a42182c3e2c75909f00ab37fffa460aadb549d5cd60d278c1a"
SIDECAR_TRAIN_PATH = REPOSITORY_ROOT / ".generated/go2_attitude_sidecar/dynamic_cartesian_v1/train.jsonl"
SIDECAR_TRAIN_FILE_SHA256 = "6cd47d0d679ace897f5b5d8e5c2f11eabab01930904666161eec3792fd9ab6d6"
SIDECAR_TRAIN_CONTENT_SHA256 = "137f1286e85fbd3e4b45d1c9fb0337255ac735508d6ead57cd816e5134725fa2"
PATCH7_RESULT_PATH = REPOSITORY_ROOT / ".generated/go2_physical_micro_overfit/patch7_v1/seed_20260710_result.json"
PATCH7_RESULT_FILE_SHA256 = "6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c"
PATCH7_RESULT_CONTENT_SHA256 = "32d848d3df68e670ddb4cc24436981f62a1aa5562b89e6d6719ecb113f66b749"
PATCH7_RESULT_SCHEMA = "lewm_go2_physical_micro_overfit_result_v1"
PARITY_RESULT_PATH = REPOSITORY_ROOT / ".generated/go2_dynamic_cartesian_fit_panel_parity/v1/result.json"
PARITY_RESULT_FILE_SHA256 = "72d21aaf5e923126dd3a5022b0ea9775340877a00f40aa22845b244886fde70b"
PARITY_RESULT_CONTENT_SHA256 = "3729a3fcd61b523d744c476da89fb2f638593145055b52bc96035bb30c3f3cea"
PARITY_RESULT_SCHEMA = "lewm_go2_dynamic_cartesian_fit_panel_parity_result_v1"
BINDING_PATH = REPOSITORY_ROOT / "docs/lewm_go2_dynamic_cartesian_n32_v1_binding_2026-07-11.md"
AMENDMENT_PATH = REPOSITORY_ROOT / "docs/lewm_go2_dynamic_cartesian_n32_v1_preoutput_amendment_2026-07-11.md"
ATTEMPT_AMENDMENT_PATH = REPOSITORY_ROOT / "docs/lewm_go2_dynamic_cartesian_n32_v1_attempt_control_amendment_2026-07-11.md"
IMPLEMENTATION_MANIFEST_PATH = REPOSITORY_ROOT / "docs/lewm_go2_dynamic_cartesian_n32_v1_implementation_manifest_2026-07-11.json"

SOURCE_ROLES = (
    "attempt_control_amendment", "benchmarks_package", "binding",
    "categorical_n32_metrics", "counterfactual", "datasets_package",
    "dynamic_geometry", "encoder", "finalizer", "finalizer_test", "lewm_package",
    "manifest_preparer", "manifest_preparer_test", "model", "models_lewm",
    "models_package", "parity_report", "parity_runner",
    "phase2d_spatial_lewm", "physical_metrics", "physical_spatial_metrics",
    "predictor", "preoutput_amendment", "primitive_affordance", "pure_contract",
    "pure_contract_test", "runner", "runner_test", "sidecar_library", "sigreg",
    "source_action_utility", "spatial_lewm", "spatial_predictor",
)
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SOURCE_WORKERS = 6
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
SMOKE_UPDATES = 3
SMOKE_EVALUATION_INTERVAL = 1
EVENT_FIELDS = (
    "image_requests", "target_requests", "attitude_requests",
    "image_decode_events", "label_shard_npz_open_events", "model_calls",
    "model_output_frames", "model_attitude_frames",
)
DATA_EVENT_FIELDS = EVENT_FIELDS[:5]
EXPECTED_PANEL_ARTIFACT_COUNTS = {
    "fit": {"images": 320, "shards": 20},
    "same_scene_holdout": {"images": 320, "shards": 20},
    "cross_scene_holdout": {"images": 320, "shards": 25},
}


def _source_path_contract() -> dict[str, Path]:
    return {
        "attempt_control_amendment": ATTEMPT_AMENDMENT_PATH,
        "benchmarks_package": REPOSITORY_ROOT / "lewm/benchmarks/__init__.py",
        "binding": BINDING_PATH,
        "categorical_n32_metrics": REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32.py",
        "counterfactual": REPOSITORY_ROOT / "lewm/benchmarks/counterfactual.py",
        "datasets_package": REPOSITORY_ROOT / "lewm/datasets/__init__.py",
        "dynamic_geometry": REPOSITORY_ROOT / "lewm/benchmarks/go2_dynamic_cell_square_projection.py",
        "encoder": REPOSITORY_ROOT / "lewm/models/encoders.py",
        "finalizer": REPOSITORY_ROOT / "scripts/finalize_go2_dynamic_cartesian_n32.py",
        "finalizer_test": REPOSITORY_ROOT / "lewm/tests/test_finalize_go2_dynamic_cartesian_n32.py",
        "lewm_package": REPOSITORY_ROOT / "lewm/__init__.py",
        "manifest_preparer": REPOSITORY_ROOT / "scripts/prepare_go2_dynamic_cartesian_n32_implementation.py",
        "manifest_preparer_test": REPOSITORY_ROOT / "lewm/tests/test_prepare_go2_dynamic_cartesian_n32_implementation.py",
        "model": REPOSITORY_ROOT / "lewm/models/egomotion_bev_jepa.py",
        "models_lewm": REPOSITORY_ROOT / "lewm/models/lewm.py",
        "models_package": REPOSITORY_ROOT / "lewm/models/__init__.py",
        "parity_report": REPOSITORY_ROOT / "docs/lewm_go2_dynamic_cartesian_fit_panel_parity_result_2026-07-11.md",
        "parity_runner": REPOSITORY_ROOT / "scripts/audit_go2_dynamic_cartesian_fit_panel_parity.py",
        "phase2d_spatial_lewm": REPOSITORY_ROOT / "lewm/models/phase2d_spatial_lewm.py",
        "physical_metrics": REPOSITORY_ROOT / "lewm/benchmarks/go2_physical_micro_overfit.py",
        "physical_spatial_metrics": REPOSITORY_ROOT / "lewm/benchmarks/go2_physical_spatial_grounding.py",
        "predictor": REPOSITORY_ROOT / "lewm/models/predictor.py",
        "preoutput_amendment": AMENDMENT_PATH,
        "primitive_affordance": REPOSITORY_ROOT / "lewm/models/primitive_affordance.py",
        "pure_contract": REPOSITORY_ROOT / "lewm/benchmarks/go2_dynamic_cartesian_n32.py",
        "pure_contract_test": REPOSITORY_ROOT / "lewm/tests/test_go2_dynamic_cartesian_n32.py",
        "runner": Path(__file__).resolve(),
        "runner_test": REPOSITORY_ROOT / "lewm/tests/test_run_go2_dynamic_cartesian_n32.py",
        "sidecar_library": REPOSITORY_ROOT / "lewm/datasets/go2_attitude_sidecar.py",
        "sigreg": REPOSITORY_ROOT / "lewm/models/sigreg.py",
        "source_action_utility": REPOSITORY_ROOT / "lewm/models/source_action_utility.py",
        "spatial_lewm": REPOSITORY_ROOT / "lewm/models/spatial_lewm.py",
        "spatial_predictor": REPOSITORY_ROOT / "lewm/models/spatial_predictor.py",
    }


def _canonical_path(value: Path | str, *, name: str) -> Path:
    path = Path(value)
    absolute = Path(os.path.abspath(os.fspath(path)))
    if path != absolute:
        raise ValueError(f"{name} must be a canonical absolute path")
    return absolute


def _open_directory(path: Path, *, name: str) -> int:
    path = _canonical_path(path, name=name)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _open_regular(path: Path, *, name: str) -> Iterable[BinaryIO]:
    path = _canonical_path(path, name=name)
    parent = _open_directory(path.parent, name=f"{name} parent")
    try:
        descriptor = os.open(path.name, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
    finally:
        os.close(parent)
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        os.close(descriptor)
        raise ValueError(f"{name} must be a regular file")
    stream = os.fdopen(descriptor, "rb")
    try:
        yield stream
        after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
            raise RuntimeError(f"{name} changed while open")
    finally:
        stream.close()


def _regular_bytes(path: Path, *, name: str) -> bytes:
    with _open_regular(path, name=name) as stream:
        return stream.read()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _open_regular(path, name=str(path)) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(data: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains nonfinite JSON number {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            data,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _read_json(path: Path, *, expected_sha256: str, name: str) -> dict[str, Any]:
    data = _regular_bytes(path, name=name)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"{name} file SHA-256 mismatch")
    return _strict_json(data, name=name)


def _validate_content(value: Mapping[str, Any], *, schema: str, content_sha256: str, name: str) -> None:
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if value.get("schema") != schema or declared != content_sha256 or contract.canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} canonical content mismatch")


def _input_contract() -> dict[str, Any]:
    return json.loads(json.dumps(contract.INPUT_BINDINGS))


def _validate_implementation_manifest(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    path = _canonical_path(path, name="implementation manifest path")
    if path != IMPLEMENTATION_MANIFEST_PATH:
        raise ValueError("implementation manifest path is not canonical")
    value = _read_json(path, expected_sha256=expected_file_sha256, name="implementation manifest")
    value = contract.validate_implementation_manifest(value)
    paths = _source_path_contract()
    entries = value["sources"]["entries"]
    expected_paths = {name: str(paths[name]) for name in SOURCE_ROLES}
    committed = {str(entry["role"]): {"path": str(entry["path"]), "sha256": str(entry["sha256"])} for entry in entries}
    if set(committed) != set(SOURCE_ROLES) or {name: committed[name]["path"] for name in SOURCE_ROLES} != expected_paths:
        raise ValueError("implementation manifest source map mismatch")
    with ThreadPoolExecutor(max_workers=SOURCE_WORKERS) as pool:
        actual = dict(zip(SOURCE_ROLES, pool.map(lambda role: _sha256_file(paths[role]), SOURCE_ROLES), strict=True))
    if actual != {name: committed[name]["sha256"] for name in SOURCE_ROLES}:
        raise ValueError("implementation manifest source hash mismatch")
    return value


def _scene_sha256(scene_id: str) -> str:
    return hashlib.sha256(scene_id.encode("utf-8")).hexdigest()


def _join_panel_attitudes(panels: Mapping[str, Sequence[Mapping[str, Any]]], sidecar_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    sidecar_by_global = {int(row["global_row"]): row for row in sidecar_rows}
    if len(sidecar_by_global) != len(sidecar_rows):
        raise ValueError("train sidecar global rows are not injective")
    joined: dict[str, list[dict[str, Any]]] = {}
    all_globals: list[int] = []
    all_identities: list[str] = []
    for panel_name in ("fit", *contract.HOLDOUT_PANELS):
        rows = list(panels[panel_name])
        panel_globals: list[int] = []
        panel_identities: list[str] = []
        for row in rows:
            global_row = int(row["global_row"])
            sidecar_row = sidecar_by_global.get(global_row)
            if sidecar_row is None:
                raise ValueError(f"{panel_name} global row lacks train attitude: {global_row}")
            identity = row_identity_sha256(row)
            expected = {
                "dataset_role": "train",
                "row_identity_sha256": identity,
                "scene_id_sha256": _scene_sha256(str(row["scene_id"])),
                "env_index": int(row["env_index"]),
                "current_frame_index": int(row["current_frame_index"]),
                "next_frame_index": int(row["next_frame_index"]),
                "current_timestamp_ns": int(row["current_timestamp_ns"]),
                "next_timestamp_ns": int(row["next_timestamp_ns"]),
            }
            if any(sidecar_row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{panel_name} attitude join mismatch at global row {global_row}")
            panel_globals.append(global_row)
            panel_identities.append(identity)
        if len(panel_globals) != 160 or len(set(panel_globals)) != 160:
            raise ValueError(f"{panel_name} does not contain 160 unique transitions")
        records = frame_records(rows)
        for record in records:
            attitude = sidecar_by_global[int(record["global_row"])][str(record["side"])]
            record["base_quat_world_xyzw"] = list(attitude["base_quat_world_xyzw"])
            record["stored_base_yaw_rad"] = float(attitude["stored_base_yaw_rad"])
            record["row_identity_sha256"] = sidecar_by_global[int(record["global_row"])]["row_identity_sha256"]
        if [str(record["side"]) for record in records[::2]] != ["current"] * 160 or [str(record["side"]) for record in records[1::2]] != ["next"] * 160:
            raise ValueError(f"{panel_name} endpoint expansion is not current-then-next")
        joined[panel_name] = records
        all_globals.extend(panel_globals)
        all_identities.extend(panel_identities)
    if len(all_globals) != 480 or len(set(all_globals)) != 480:
        raise ValueError("N32 panels do not join exactly 480 unique train rows")
    audit = {"transition_count": 480, "frame_count": 960, "all_train_role": True, "global_rows_sha256": contract.canonical_json_sha256(all_globals), "row_identities_sha256": contract.canonical_json_sha256(all_identities), "panel_global_rows_sha256": {name: contract.canonical_json_sha256([int(row["global_row"]) for row in panels[name]]) for name in ("fit", *contract.HOLDOUT_PANELS)}}
    return joined, audit


def _decode_image(path: str, digest: str) -> torch.Tensor:
    data = _regular_bytes(Path(path), name="panel RGB")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("panel RGB changed during decode")
    with Image.open(BytesIO(data)) as image:
        image = image.convert("RGB").resize((112, 112), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = tensor.new_tensor(NORMALIZATION_MEAN)[:, None, None]
    std = tensor.new_tensor(NORMALIZATION_STD)[:, None, None]
    return (tensor - mean) / std


def _decode_shard(path: str, digest: str) -> dict[str, np.ndarray]:
    data = _regular_bytes(Path(path), name="panel label shard")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("panel label shard changed during decode")
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


class DynamicPanelDataset:
    """Panel-scoped cache with explicit logical and physical access events."""

    def __init__(self, records: Sequence[Mapping[str, Any]], panel: str) -> None:
        self.records = [dict(record) for record in records]
        self.panel = panel
        self._images: dict[str, torch.Tensor] = {}
        self._shards: dict[str, dict[str, np.ndarray]] = {}
        self._targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.events: Counter[str] = Counter()

    def snapshot(self) -> dict[str, int]:
        return {name: int(self.events[name]) for name in EVENT_FIELDS}

    def delta(self, before: Mapping[str, int]) -> dict[str, int]:
        return {name: int(self.events[name]) - int(before.get(name, 0)) for name in EVENT_FIELDS}

    def preload(self) -> None:
        images = sorted({(str(record["image_path"]), str(record["image_sha256"])) for record in self.records})
        shards = sorted({(str(record["label_shard_path"]), str(record["label_shard_sha256"])) for record in self.records})
        with ThreadPoolExecutor(max_workers=SOURCE_WORKERS) as pool:
            decoded_images = list(pool.map(lambda item: (item[0], _decode_image(*item)), images))
            decoded_shards = list(pool.map(lambda item: (item[0], _decode_shard(*item)), shards))
        self._images.update(decoded_images)
        self._shards.update(decoded_shards)
        self.events["image_decode_events"] += len(decoded_images)
        self.events["label_shard_npz_open_events"] += len(decoded_shards)

    def _target(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.events["target_requests"] += 1
        if index not in self._targets:
            record = self.records[index]
            shard = self._shards[str(record["label_shard_path"])]
            side, row = str(record["side"]), int(record["label_shard_row"])
            labels = np.asarray(shard[f"{side}_labels"][row], dtype=np.int64)
            mask = np.asarray(shard[f"{side}_supervision_mask"][row], dtype=bool)
            if labels.shape != (64, 64) or mask.shape != (64, 64):
                raise ValueError("N32 target grid shape changed")
            self._targets[index] = (torch.from_numpy(labels.copy()).long(), torch.from_numpy(mask.copy()).bool())
        return self._targets[index]

    def _image(self, path: str) -> torch.Tensor:
        self.events["image_requests"] += 1
        return self._images[path]

    def _attitude(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.events["attitude_requests"] += 1
        record = self.records[index]
        return torch.tensor(record["base_quat_world_xyzw"], dtype=torch.float32), torch.tensor(record["stored_base_yaw_rad"], dtype=torch.float32)

    def training_batch(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        images, labels, masks, quaternions, yaws = [], [], [], [], []
        for raw in indices:
            index = int(raw)
            label, mask = self._target(index)
            quaternion, yaw = self._attitude(index)
            images.append(self._image(str(self.records[index]["image_path"])))
            labels.append(label); masks.append(mask); quaternions.append(quaternion); yaws.append(yaw)
        return {"image": torch.stack(images), "labels": torch.stack(labels), "mask": torch.stack(masks), "base_quat_world_xyzw": torch.stack(quaternions), "stored_base_yaw_rad": torch.stack(yaws)}

    def evaluation_batch(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        output: dict[str, list[torch.Tensor]] = {condition: [] for condition in contract.CONDITIONS}
        labels, masks, quaternions, yaws = [], [], [], []
        image_keys = {"correct_rgb": "image_path", "role_global_shuffled_rgb": "control_image_path", "same_scene_wrong_view_rgb": "same_scene_control_image_path"}
        for raw in indices:
            index = int(raw); record = self.records[index]
            label, mask = self._target(index)
            quaternion, yaw = self._attitude(index)
            for condition, key in image_keys.items():
                output[condition].append(self._image(str(record[key])))
            labels.append(label); masks.append(mask); quaternions.append(quaternion); yaws.append(yaw)
        return {**{name: torch.stack(values) for name, values in output.items()}, "labels": torch.stack(labels), "mask": torch.stack(masks), "base_quat_world_xyzw": torch.stack(quaternions), "stored_base_yaw_rad": torch.stack(yaws)}


def _distance_grid() -> np.ndarray:
    forward = np.linspace(-0.95, 5.35, 64, dtype=np.float64)
    left = np.linspace(-3.15, 3.15, 64, dtype=np.float64)
    return np.sqrt(forward[:, None] ** 2 + left[None, :] ** 2)


def _canonical_records(records: Sequence[Mapping[str, Any]], *, seed: int, panel: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    copied = [dict(record) for record in records]
    if len(copied) != contract.FRAME_COUNT or len({str(record["image_sha256"]) for record in copied}) != contract.FRAME_COUNT:
        raise ValueError(f"{panel} must contain 320 unique endpoint images")
    copied, role_global = attach_role_global_shuffle(copied, seed=seed, namespace=panel)
    copied, same_scene = attach_same_scene_wrong_view(copied, seed=seed, namespace=panel)
    return copied, {"role_global_shuffle": role_global, "same_scene_wrong_view": same_scene, "wrong_rgb_uses_target_attitude": True}


def _artifact_contract(records: Sequence[Mapping[str, Any]], panel: str) -> tuple[dict[str, str], dict[str, str]]:
    images: dict[str, str] = {}
    shards: dict[str, str] = {}
    for record in records:
        for collection, path_key, hash_key in ((images, "image_path", "image_sha256"), (shards, "label_shard_path", "label_shard_sha256")):
            path, digest = str(record[path_key]), str(record[hash_key])
            if collection.setdefault(path, digest) != digest:
                raise ValueError(f"conflicting {panel} artifact hash: {path}")
    if {"images": len(images), "shards": len(shards)} != EXPECTED_PANEL_ARTIFACT_COUNTS[panel]:
        raise ValueError(f"{panel} artifact counts changed")
    return images, shards


def _verify_artifacts(images: Mapping[str, str], shards: Mapping[str, str]) -> None:
    items = list(sorted(images.items())) + list(sorted(shards.items()))
    with ThreadPoolExecutor(max_workers=SOURCE_WORKERS) as pool:
        actual = list(pool.map(lambda item: _sha256_file(Path(item[0])), items))
    for (path, expected), observed in zip(items, actual, strict=True):
        if observed != expected:
            raise ValueError(f"authorized panel artifact changed: {path}")


def _weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    applied = weights[labels] * mask.to(loss.dtype)
    return (loss * applied).sum() / applied.sum().clamp_min(torch.finfo(loss.dtype).tiny)


def direct_hierarchical_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[1:] != (3, 64, 64) or labels.shape != logits.shape[:1] + logits.shape[2:] or labels.dtype != torch.long or mask.shape != labels.shape or mask.dtype != torch.bool:
        raise ValueError("direct hierarchical loss tensor contract changed")
    unknown_known_weights = logits.new_tensor(TRAINING_WEIGHTS["unknown_known"])
    free_occupied_weights = logits.new_tensor(TRAINING_WEIGHTS["free_occupied"])
    known_logit = torch.logsumexp(logits[:, 1:], dim=1)
    unknown_known = torch.stack((logits[:, 0], known_logit), dim=1)
    unknown_known_loss = _weighted_cross_entropy(unknown_known, (labels != 0).long(), mask, unknown_known_weights)
    known_mask = mask & (labels != 0)
    free_occupied_loss = _weighted_cross_entropy(logits[:, 1:], (labels - 1).clamp_min(0), known_mask, free_occupied_weights)
    return 0.5 * unknown_known_loss + 0.5 * free_occupied_loss


def _normalized_events(value: Mapping[str, int]) -> dict[str, int]:
    return {name: int(value.get(name, 0)) for name in EVENT_FIELDS}


@torch.no_grad()
def evaluate_panel(model: torch.nn.Module, dataset: DynamicPanelDataset, records: Sequence[Mapping[str, Any]], *, device: torch.device, panel: str, controls: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    before = dataset.snapshot()
    aggregate = {condition: empty_raw_accumulator() for condition in contract.CONDITIONS}
    by_family = {family: {condition: empty_raw_accumulator() for condition in contract.CONDITIONS} for family in contract.FAMILIES}
    distances = _distance_grid()
    model.eval(); model_calls = 0; model_output_frames = 0
    for start in range(0, len(records), contract.BATCH_SIZE):
        indices = tuple(range(start, start + contract.BATCH_SIZE))
        batch = dataset.evaluation_batch(indices)
        images = torch.cat([batch[condition] for condition in contract.CONDITIONS], dim=0).to(device=device, dtype=torch.float32)
        target_quaternion = batch["base_quat_world_xyzw"].to(device=device, dtype=torch.float32)
        target_yaw = batch["stored_base_yaw_rad"].to(device=device, dtype=torch.float32)
        quaternions = torch.cat([target_quaternion] * len(contract.CONDITIONS), dim=0)
        yaws = torch.cat([target_yaw] * len(contract.CONDITIONS), dim=0)
        logits = model.occupancy_logits(images, quaternions, yaws).float().cpu().numpy()
        if logits.shape != (12, 3, 64, 64):
            raise RuntimeError("N32 evaluation did not use one combined batch of 12")
        model_calls += 1; model_output_frames += 12
        split = np.split(logits, len(contract.CONDITIONS), axis=0)
        labels, mask = batch["labels"].numpy(), batch["mask"].numpy()
        for condition, values in zip(contract.CONDITIONS, split, strict=True):
            update_raw_accumulator(aggregate[condition], values, labels, mask, distances)
            for offset, target_index in enumerate(indices):
                family = str(records[target_index]["family"])
                update_raw_accumulator(by_family[family][condition], values[offset:offset + 1], labels[offset:offset + 1], mask[offset:offset + 1], distances)
    report = {
        "schema": getattr(contract, "PANEL_REPORT_SCHEMA", "lewm_go2_dynamic_cartesian_n32_panel_report_v1"),
        "panel": panel,
        "frame_count": len(records),
        "target_batch_size": contract.BATCH_SIZE,
        "combined_model_batch_size": 12,
        "model_call_dtype": "float32",
        "metric_accumulator_dtype": "float64",
        "wrong_rgb_uses_target_attitude": True,
        "conditions": {name: finalize_raw_accumulator(aggregate[name]) for name in contract.CONDITIONS},
        "families": {family: {"conditions": {name: finalize_raw_accumulator(by_family[family][name]) for name in contract.CONDITIONS}} for family in contract.FAMILIES},
        "controls": dict(controls),
    }
    if panel == "fit":
        report["fit_gate"] = contract.fit_panel_gate_report(report)
    access = dataset.delta(before)
    access.update({"model_calls": model_calls, "model_output_frames": model_output_frames, "model_attitude_frames": model_output_frames})
    return report, _normalized_events(access)


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(contract.canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _state_contract(model: torch.nn.Module) -> tuple[dict[str, Any], str]:
    requires_grad = {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()}
    entries = [{"name": name, "dtype": str(value.dtype), "shape": list(value.shape), "requires_grad": requires_grad.get(name)} for name, value in sorted(model.state_dict().items())]
    record = {"entry_count": len(entries), "entries": entries}
    return record, contract.canonical_json_sha256(record)


def _model_kwargs() -> dict[str, Any]:
    return {key: (tuple(value) if key in {"bev_size", "forward_range_m", "left_range_m", "projective_camera_xyz_body_m", "projective_camera_rpy_body_rad", "projective_vertical_anchor_z_body_m"} else value) for key, value in contract.MODEL_CONFIG.items()}


def _new_model(device: torch.device) -> EgomotionBevJepa:
    with torch.device(device):
        model = EgomotionBevJepa(**_model_kwargs())
    if model.bev_lift_type != DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT or model.occupancy_weight != 2.0:
        raise RuntimeError("dynamic model construction drift")
    return model


def _derive_initial_state(
    device: torch.device, seed: int
) -> tuple[dict[str, torch.Tensor], str, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _new_model(device)
    state_contract, state_contract_sha = _state_contract(model)
    state = _clone_state(model.state_dict())
    state_sha = _state_dict_sha256(state)
    proof = {"initial_state_sha256": state_sha, "state_contract": state_contract, "state_contract_sha256": state_contract_sha, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
    del model
    torch.cuda.empty_cache()
    return state, state_sha, proof


def _build_initial_state(device: torch.device, seed: int, manifest: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], str, dict[str, Any]]:
    state, state_sha, proof = _derive_initial_state(device, seed)
    if state_sha != manifest["model_initial_state_sha256"][str(seed)] or proof[
        "state_contract_sha256"
    ] != manifest["model_state_contract_sha256"][str(seed)]:
        raise RuntimeError("registered N32 initialization changed")
    return state, state_sha, proof


def _stage_schedule(seed: int, branch: str, smoke: bool) -> list[list[int]]:
    updates = SMOKE_UPDATES if smoke else int(contract.BRANCH_CONFIGS[branch]["updates"])
    return contract.deterministic_minibatch_schedule(seed=seed, branch=branch, updates=updates)


def _run_stage(*, branch: str, smoke: bool, initial_state: Mapping[str, torch.Tensor], initial_state_sha256: str, dataset: DynamicPanelDataset, records: Sequence[Mapping[str, Any]], controls: Mapping[str, Any], device: torch.device, seed: int, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], torch.nn.Module]:
    config = dict(contract.BRANCH_CONFIGS[branch])
    updates = SMOKE_UPDATES if smoke else int(config["updates"])
    interval = SMOKE_EVALUATION_INTERVAL if smoke else contract.EVALUATION_INTERVAL
    schedule = _stage_schedule(seed, branch, smoke)
    schedule_sha = contract.canonical_json_sha256(schedule)
    commitment_key = f"{seed}:{branch}"
    schedule_group = "smoke_sha256" if smoke else "authoritative_sha256"
    expected_schedule = manifest["schedules"][schedule_group][commitment_key]
    if schedule_sha != expected_schedule:
        raise RuntimeError("registered minibatch schedule changed")
    if not smoke:
        contract.validate_minibatch_schedule(schedule, seed=seed, branch=branch)
    model = _new_model(device)
    model.load_state_dict(initial_state, strict=True)
    if _state_dict_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("optimizer branch did not restart the exact initial state")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]), betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    before = dataset.snapshot(); evaluation_access: Counter[str] = Counter(); curve: list[dict[str, Any]] = []
    for step, indices in enumerate(schedule, start=1):
        batch = dataset.training_batch(indices)
        model.train(); optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device=device, dtype=torch.float32)
        quaternions = batch["base_quat_world_xyzw"].to(device=device, dtype=torch.float32)
        yaws = batch["stored_base_yaw_rad"].to(device=device, dtype=torch.float32)
        logits = model.occupancy_logits(images, quaternions, yaws)
        loss = direct_hierarchical_loss(logits, batch["labels"].to(device), batch["mask"].to(device))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite {branch} loss at update {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, contract.GRADIENT_CLIP)
        optimizer.step()
        if step % interval == 0:
            report, access = evaluate_panel(model, dataset, records, device=device, panel="fit", controls=controls)
            evaluation_access.update(access)
            curve.append({"step": step, "batch_loss": float(loss.detach().item()), "gradient_norm_before_clip": float(gradient_norm), "fit_panel": report})
    terminal = contract.terminal_fit_gate_summary(curve, updates, interval)
    total = _normalized_events(dataset.delta(before)); evaluation = _normalized_events(evaluation_access)
    training = {name: total[name] - evaluation[name] for name in EVENT_FIELDS}
    training.update({"model_calls": updates, "model_output_frames": updates * contract.BATCH_SIZE, "model_attitude_frames": updates * contract.BATCH_SIZE})
    stage = {
        "schema": contract.STAGE_SCHEMA,
        "stage": branch,
        "config": config,
        "maximum_steps": updates,
        "completed_steps": updates,
        "batch_size": contract.BATCH_SIZE,
        "evaluation_interval": interval,
        "optimizer": {"name": "AdamW", "learning_rate": float(config["learning_rate"]), "weight_decay": float(config["weight_decay"]), "betas": [0.9, 0.999], "epsilon": 1e-8, "amsgrad": False, "gradient_clip": contract.GRADIENT_CLIP, "constant_learning_rate": True},
        "objective": {"entrypoint": "occupancy_logits", "loss": "direct_equal_capacity_hierarchical", "occupancy_weight_stored_but_not_multiplied": 2.0, "jepa_weight": 0.0, "equivariance_weight": 0.0, "action_contrast_weight": 0.0, "variance_weight": 0.0},
        "fixed_update_budget_consumed": True,
        "one_direct_forward_backward_per_update": True,
        "gradient_accumulation_or_microbatching": False,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": _state_dict_sha256(model.state_dict()),
        "exact_initial_state_restart_verified": True,
        "minibatch_indices": schedule,
        "minibatch_indices_sha256": schedule_sha,
        "learning_curve": curve,
        "terminal_fit_gate": terminal,
        "training_access": training,
        "fit_evaluation_access": evaluation,
        "holdouts_evaluated": False,
    }
    return stage, model


def _validate_resource_environment(device_value: str) -> tuple[torch.device, dict[str, Any]]:
    if device_value not in ("cuda", "cuda:0"):
        raise ValueError("dynamic N32 requires visible discrete device cuda:0")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise ValueError("HIP_VISIBLE_DEVICES must be exactly 0")
    if "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        raise ValueError("HSA_OVERRIDE_GFX_VERSION must be unset")
    if {name: os.environ.get(name) for name in THREAD_ENV} != {name: "1" for name in THREAD_ENV}:
        raise ValueError("native CPU thread limits must all be exactly one")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one HIP-visible CUDA device is required")
    device = torch.device("cuda:0")
    name = str(torch.cuda.get_device_name(device))
    properties = torch.cuda.get_device_properties(device)
    memory = int(properties.total_memory)
    normalized = name.lower().replace(" ", "")
    if "r9700" not in normalized or "raphael" in normalized or memory < 16 * 1024**3:
        raise ValueError("visible device is not the registered discrete R9700")
    return device, {"device": "cuda:0", "device_name": name, "total_memory_bytes": memory, "hip_visible_devices": "0", "hsa_override_gfx_version_unset": True, "raphael_rejected": True}


def _configure_determinism(seed: int) -> dict[str, Any]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    return {"seed": seed, "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True}


def _git_snapshot() -> dict[str, Any]:
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status_text = subprocess.run(("git", "status", "--short"), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True).stdout.rstrip()
    diff = subprocess.run(("git", "diff", "--binary", "--no-ext-diff"), cwd=REPOSITORY_ROOT, check=True, capture_output=True).stdout
    return {"head": head, "status_short": status_text, "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(), "tracked_dirty_diff_bytes": len(diff)}


def _load_bound_inputs() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_manifest = _read_json(DATASET_MANIFEST_PATH, expected_sha256=DATASET_MANIFEST_FILE_SHA256, name="physical-v3 dataset manifest")
    query_support = build_projective_query_support_contract(dataset_manifest, lift_type=DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT)
    bound_support = validate_projective_query_support_binding(
        model_config=contract.MODEL_CONFIG,
        projective_query_support=query_support,
        dataset_manifest=dataset_manifest,
        occupancy_output_contract={"projective_query_support_contract_sha256": query_support["contract_sha256"]},
    )
    if bound_support != query_support:
        raise ValueError("dynamic projective query support binding changed")
    panel = _read_json(PANEL_PATH, expected_sha256=PANEL_FILE_SHA256, name="N32 panel")
    if panel.get("content_sha256") != PANEL_CONTENT_SHA256:
        raise ValueError("N32 panel content hash mismatch")
    panels = validate_panel_manifest(panel)
    for name in ("fit", *contract.HOLDOUT_PANELS):
        if panel["panels"][name].get("rows_sha256") != PANEL_ROWS_SHA256[name]:
            raise ValueError(f"N32 {name} rows changed")
    comparator = _read_json(PATCH7_RESULT_PATH, expected_sha256=PATCH7_RESULT_FILE_SHA256, name="static patch7 comparator")
    _validate_content(comparator, schema=PATCH7_RESULT_SCHEMA, content_sha256=PATCH7_RESULT_CONTENT_SHA256, name="static patch7 comparator")
    reference = contract.extract_faithful_patch7_family_reference(comparator)
    parity = _read_json(PARITY_RESULT_PATH, expected_sha256=PARITY_RESULT_FILE_SHA256, name="fit projection parity result")
    _validate_content(parity, schema=PARITY_RESULT_SCHEMA, content_sha256=PARITY_RESULT_CONTENT_SHA256, name="fit projection parity result")
    sidecar_roles = load_attitude_sidecar_roles(SIDECAR_MANIFEST_PATH, roles=("train",), expected_manifest_sha256=SIDECAR_MANIFEST_FILE_SHA256, contract=FROZEN_BUILD_CONTRACT)
    sidecar_rows = sidecar_roles["train"]
    if len(sidecar_rows) != 4262 or sidecar_json_sha256(sidecar_rows) != SIDECAR_TRAIN_CONTENT_SHA256:
        raise ValueError("authorized train sidecar content changed")
    joined, join_audit = _join_panel_attitudes(panels, sidecar_rows)
    return panel, joined, reference, parity, join_audit, query_support


def _validate_primary_result(
    path: Path,
    expected_sha256: str,
    attempt_marker_path: Path,
    expected_attempt_marker_sha256: str,
    implementation_manifest: Mapping[str, Any],
    implementation_manifest_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _canonical_path(path, name="seed-20260710 result path")
    if path != _canonical_output(20260710):
        raise ValueError("seed-20260710 result path is not canonical")
    attempt_marker_path = _canonical_path(
        attempt_marker_path, name="seed-20260710 attempt-marker path"
    )
    if attempt_marker_path != _canonical_attempt_marker(20260710):
        raise ValueError("seed-20260710 attempt-marker path is not canonical")
    attempt_marker = _read_json(
        attempt_marker_path,
        expected_sha256=expected_attempt_marker_sha256,
        name="seed-20260710 attempt marker",
    )
    value = _read_json(path, expected_sha256=expected_sha256, name="seed-20260710 result")
    validated = contract.validate_authoritative_result(
        value,
        20260710,
        implementation_manifest,
        implementation_manifest_file_sha256,
        attempt_marker,
        expected_attempt_marker_sha256,
    )
    return validated, attempt_marker


def _reconcile_stage_access(stage: Mapping[str, Any]) -> None:
    updates = int(stage["completed_steps"]); evaluations = len(stage["learning_curve"])
    training = _normalized_events(stage["training_access"])
    expected_training = {"image_requests": updates * 4, "target_requests": updates * 4, "attitude_requests": updates * 4, "image_decode_events": 0, "label_shard_npz_open_events": 0, "model_calls": updates, "model_output_frames": updates * 4, "model_attitude_frames": updates * 4}
    expected_eval = {"image_requests": evaluations * 960, "target_requests": evaluations * 320, "attitude_requests": evaluations * 320, "image_decode_events": 0, "label_shard_npz_open_events": 0, "model_calls": evaluations * 80, "model_output_frames": evaluations * 960, "model_attitude_frames": evaluations * 960}
    if training != expected_training or _normalized_events(stage["fit_evaluation_access"]) != expected_eval:
        raise RuntimeError(f"{stage['stage']} access ledger does not reconcile")


def _reconcile_access(fit_dataset: DynamicPanelDataset, stages: Mapping[str, Any], panel_access: Mapping[str, Any], holdouts: Mapping[str, Any] | None) -> None:
    logical_total = Counter()
    for stage in stages.values():
        if stage is None:
            continue
        _reconcile_stage_access(stage)
        logical_total.update(stage["training_access"]); logical_total.update(stage["fit_evaluation_access"])
    totals = fit_dataset.snapshot()
    for name in ("image_requests", "target_requests", "attitude_requests"):
        if totals[name] != logical_total[name]:
            raise RuntimeError("fit logical access totals do not reconcile")
    if totals["image_decode_events"] != 320 or totals["label_shard_npz_open_events"] != 20:
        raise RuntimeError("fit cache preload chronology changed")
    for panel in contract.HOLDOUT_PANELS:
        record = panel_access[panel]
        if holdouts is None:
            if record["authorized"] is not False or any(record["dataset_access"].values()):
                raise RuntimeError("unauthorized holdout access was recorded")
        else:
            expected = {"image_requests": 960, "target_requests": 320, "attitude_requests": 320, "image_decode_events": 320, "label_shard_npz_open_events": EXPECTED_PANEL_ARTIFACT_COUNTS[panel]["shards"], "model_calls": 80, "model_output_frames": 960, "model_attitude_frames": 960}
            if _normalized_events(record["dataset_access"]) != expected:
                raise RuntimeError(f"{panel} one-shot access does not reconcile")


def _mkdir_parents(path: Path) -> None:
    path = _canonical_path(path, name="output parent")
    descriptor = os.open(path.anchor, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        for component in path.parts[1:]:
            try:
                next_descriptor = os.open(component, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, 0o755, dir_fd=descriptor)
                next_descriptor = os.open(component, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
            os.close(descriptor); descriptor = next_descriptor
    finally:
        os.close(descriptor)


def _publish_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = _canonical_path(path, name="result path")
    _mkdir_parents(path.parent)
    parent = _open_directory(path.parent, name="result parent")
    staging_name = f".{path.name}.staging-{secrets.token_hex(12)}"
    try:
        os.mkdir(staging_name, 0o700, dir_fd=parent)
        staging = os.open(staging_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
        try:
            descriptor = os.open("result.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o644, dir_fd=staging)
            data = _published_json_bytes(payload)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            try:
                os.link("result.json", path.name, src_dir_fd=staging, dst_dir_fd=parent, follow_symlinks=False)
            except FileExistsError as exc:
                raise FileExistsError(f"result already exists: {path}") from exc
            os.fsync(parent)
            os.unlink("result.json", dir_fd=staging)
        finally:
            os.close(staging)
        os.rmdir(staging_name, dir_fd=parent)
    except BaseException:
        try:
            staging = os.open(staging_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
            try:
                try: os.unlink("result.json", dir_fd=staging)
                except FileNotFoundError: pass
            finally: os.close(staging)
            os.rmdir(staging_name, dir_fd=parent)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(parent)


def _canonical_output(seed: int) -> Path:
    return Path(
        os.path.abspath(
            REPOSITORY_ROOT
            / f".generated/go2_dynamic_cartesian_n32/v1/seed_{seed}_result.json"
        )
    )


def _canonical_attempt_marker(seed: int) -> Path:
    contract.validate_seed(seed)
    return Path(os.path.abspath(contract.ATTEMPT_MARKER_PATHS[seed]))


def _published_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False).encode()
        + b"\n"
    )


def _claim_authoritative_attempt(
    *,
    seed: int,
    invocation: Sequence[str],
    started_at_utc: str,
    implementation_manifest: Mapping[str, Any],
    implementation_manifest_file_sha256: str,
    primary_result: Mapping[str, Any] | None = None,
    primary_file_sha256: str | None = None,
    primary_attempt_marker: Mapping[str, Any] | None = None,
    primary_attempt_marker_file_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Consume one authoritative seed before any experiment payload access."""

    seed = contract.validate_seed(seed)
    marker_path = _canonical_attempt_marker(seed)
    invocation_list = list(map(str, invocation))
    core = {
        "schema": contract.ATTEMPT_MARKER_SCHEMA,
        "authoritative": True,
        "seed": seed,
        "created_at_utc": str(started_at_utc),
        "invocation": invocation_list,
        "invocation_sha256": contract.canonical_json_sha256(invocation_list),
        "canonical_result_path": str(_canonical_output(seed)),
        "canonical_attempt_marker_path": str(marker_path),
        "contract": {
            "path": str(BINDING_PATH),
            "sha256": contract.EXECUTION_BINDING_SHA256,
        },
        "preoutput_amendment": {
            "path": str(AMENDMENT_PATH),
            "sha256": contract.PREOUTPUT_AMENDMENT_SHA256,
        },
        "attempt_control_amendment": {
            "path": str(ATTEMPT_AMENDMENT_PATH),
            "sha256": contract.ATTEMPT_CONTROL_AMENDMENT_SHA256,
        },
        "implementation_manifest": {
            "path": str(IMPLEMENTATION_MANIFEST_PATH),
            "sha256": str(implementation_manifest_file_sha256),
            "content_sha256": implementation_manifest["content_sha256"],
        },
        "seed_20260710_result": (
            None
            if primary_result is None
            else {
                "path": str(_canonical_output(contract.EXPECTED_SEEDS[0])),
                "sha256": str(primary_file_sha256),
                "content_sha256": primary_result["content_sha256"],
            }
        ),
        "seed_20260710_attempt_marker": (
            None
            if primary_attempt_marker is None
            else {
                "path": str(_canonical_attempt_marker(contract.EXPECTED_SEEDS[0])),
                "sha256": str(primary_attempt_marker_file_sha256),
                "content_sha256": primary_attempt_marker["content_sha256"],
            }
        ),
        "attempt_consumed": True,
        "retry_permitted": False,
        "payload_access_started": False,
    }
    marker = {**core, "content_sha256": contract.canonical_json_sha256(core)}
    contract.validate_attempt_marker(
        marker,
        seed,
        implementation_manifest,
        implementation_manifest_file_sha256,
        primary_result=primary_result,
        primary_file_sha256=primary_file_sha256,
        primary_attempt_marker=primary_attempt_marker,
        primary_attempt_marker_file_sha256=primary_attempt_marker_file_sha256,
    )
    expected_file_sha256 = hashlib.sha256(_published_json_bytes(marker)).hexdigest()
    _publish_json_exclusive(marker_path, marker)
    observed = _read_json(
        marker_path,
        expected_sha256=expected_file_sha256,
        name=f"seed-{seed} authoritative attempt marker",
    )
    contract.validate_attempt_marker(
        observed,
        seed,
        implementation_manifest,
        implementation_manifest_file_sha256,
        primary_result=primary_result,
        primary_file_sha256=primary_file_sha256,
        primary_attempt_marker=primary_attempt_marker,
        primary_attempt_marker_file_sha256=primary_attempt_marker_file_sha256,
    )
    return observed, expected_file_sha256


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-manifest", type=Path, required=True)
    parser.add_argument("--expected-implementation-manifest-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--seed-20260710-result", type=Path)
    parser.add_argument("--expected-seed-20260710-sha256")
    parser.add_argument("--seed-20260710-attempt-marker", type=Path)
    parser.add_argument("--expected-seed-20260710-attempt-marker-sha256")
    parser.add_argument("--non-authoritative-smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.output = _canonical_path(args.output, name="output path")
        args.implementation_manifest = _canonical_path(args.implementation_manifest, name="implementation manifest path")
        if args.seed_20260710_result is not None:
            args.seed_20260710_result = _canonical_path(args.seed_20260710_result, name="seed-20260710 result path")
        if args.seed_20260710_attempt_marker is not None:
            args.seed_20260710_attempt_marker = _canonical_path(
                args.seed_20260710_attempt_marker,
                name="seed-20260710 attempt-marker path",
            )
    except ValueError as exc:
        parser.error(str(exc))
    try: contract.validate_seed(args.seed)
    except ValueError as exc: parser.error(str(exc))
    if args.output.exists(): parser.error("output already exists; results are immutable")
    primary_args = (
        args.seed_20260710_result,
        args.expected_seed_20260710_sha256,
        args.seed_20260710_attempt_marker,
        args.expected_seed_20260710_attempt_marker_sha256,
    )
    if args.seed == 20260710 and any(value is not None for value in primary_args): parser.error("seed 20260710 rejects primary-result arguments")
    if args.seed == 20260711 and any(value is None for value in primary_args): parser.error("seed 20260711 requires the exact seed-20260710 result and hash")
    if args.non_authoritative_smoke and args.seed != 20260710: parser.error("smoke is seed-20260710-only")
    if args.non_authoritative_smoke and args.output in {_canonical_output(seed) for seed in contract.EXPECTED_SEEDS}: parser.error("smoke may not occupy a canonical result path")
    if not args.non_authoritative_smoke and args.output != _canonical_output(args.seed): parser.error("authoritative output path is not canonical")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    invocation = list(sys.argv) if argv is None else [str(Path(__file__).resolve()), *(str(value) for value in argv)]
    started = datetime.now(timezone.utc).isoformat()
    implementation = _validate_implementation_manifest(args.implementation_manifest, str(args.expected_implementation_manifest_sha256))
    source_start = {str(entry["role"]): str(entry["sha256"]) for entry in implementation["sources"]["entries"]}
    primary = None
    primary_attempt_marker = None
    if args.seed == 20260711:
        primary, primary_attempt_marker = _validate_primary_result(
            args.seed_20260710_result,
            str(args.expected_seed_20260710_sha256),
            args.seed_20260710_attempt_marker,
            str(args.expected_seed_20260710_attempt_marker_sha256),
            implementation,
            str(args.expected_implementation_manifest_sha256),
        )
    device, device_record = _validate_resource_environment(str(args.device))
    smoke = bool(args.non_authoritative_smoke)
    attempt_marker = None
    attempt_marker_file_sha256 = None
    if not smoke:
        attempt_marker, attempt_marker_file_sha256 = _claim_authoritative_attempt(
            seed=int(args.seed),
            invocation=invocation,
            started_at_utc=started,
            implementation_manifest=implementation,
            implementation_manifest_file_sha256=str(
                args.expected_implementation_manifest_sha256
            ),
            primary_result=primary,
            primary_file_sha256=(
                None if primary is None else str(args.expected_seed_20260710_sha256)
            ),
            primary_attempt_marker=primary_attempt_marker,
            primary_attempt_marker_file_sha256=(
                None
                if primary_attempt_marker is None
                else str(args.expected_seed_20260710_attempt_marker_sha256)
            ),
        )
    panel, joined_records, patch7_reference, parity_result, join_audit, query_support = _load_bound_inputs()
    git_start = _git_snapshot()
    determinism = _configure_determinism(int(args.seed))
    initial_state, initial_state_sha, initialization = _build_initial_state(device, int(args.seed), implementation)

    fit_records, fit_controls = _canonical_records(joined_records["fit"], seed=args.seed, panel="fit")
    fit_images, fit_shards = _artifact_contract(fit_records, "fit")
    _verify_artifacts(fit_images, fit_shards)
    fit_dataset = DynamicPanelDataset(fit_records, "fit"); fit_dataset.preload()
    faithful, faithful_model = _run_stage(branch="production_faithful", smoke=smoke, initial_state=initial_state, initial_state_sha256=initial_state_sha, dataset=fit_dataset, records=fit_records, controls=fit_controls, device=device, seed=args.seed, manifest=implementation)
    ceiling = None; qualifying_model: torch.nn.Module | None = None; qualifying_branch: str | None = None
    if bool(faithful["terminal_fit_gate"]["passes"]):
        if not smoke:
            qualifying_model, qualifying_branch = faithful_model, "production_faithful"
        else:
            del faithful_model
    else:
        del faithful_model; torch.cuda.empty_cache()
        ceiling, ceiling_model = _run_stage(branch="ceiling_optimizer", smoke=smoke, initial_state=initial_state, initial_state_sha256=initial_state_sha, dataset=fit_dataset, records=fit_records, controls=fit_controls, device=device, seed=args.seed, manifest=implementation)
        if ceiling["minibatch_indices"][:len(faithful["minibatch_indices"])] != faithful["minibatch_indices"]:
            raise RuntimeError("faithful and ceiling schedules do not share their exact prefix")
        if bool(ceiling["terminal_fit_gate"]["passes"]) and not smoke:
            qualifying_model, qualifying_branch = ceiling_model, "ceiling_optimizer"
        else:
            del ceiling_model

    holdouts: dict[str, Any] | None = None
    holdout_checks: dict[str, Any] | None = None
    panel_access: dict[str, Any] = {"fit": {"authorized": True, "artifact_hash_passes": 2, "image_hash_byte_open_events": 2 * len(fit_images), "shard_hash_byte_open_events": 2 * len(fit_shards), "dataset_access": fit_dataset.snapshot()}}
    if qualifying_model is not None:
        holdouts = {}; holdout_checks = {}
        for panel_name in contract.HOLDOUT_PANELS:
            records, controls = _canonical_records(joined_records[panel_name], seed=args.seed, panel=panel_name)
            images, shards = _artifact_contract(records, panel_name)
            _verify_artifacts(images, shards)
            dataset = DynamicPanelDataset(records, panel_name); dataset.preload()
            report, access = evaluate_panel(qualifying_model, dataset, records, device=device, panel=panel_name, controls=controls)
            _verify_artifacts(images, shards)
            holdouts[panel_name] = report
            holdout_checks[panel_name] = contract.strict_patch7_holdout_checks(report, patch7_reference["panels"][panel_name])
            complete_access = dict(access)
            complete_access["image_decode_events"] = dataset.snapshot()["image_decode_events"]
            complete_access["label_shard_npz_open_events"] = dataset.snapshot()["label_shard_npz_open_events"]
            panel_access[panel_name] = {"authorized": True, "authorized_by_branch": qualifying_branch, "artifact_hash_passes": 2, "image_hash_byte_open_events": 2 * len(images), "shard_hash_byte_open_events": 2 * len(shards), "dataset_access": complete_access, "one_shot_evaluation": True}
        (faithful if qualifying_branch == "production_faithful" else ceiling)["holdouts_evaluated"] = True
        del qualifying_model
    else:
        for panel_name in contract.HOLDOUT_PANELS:
            panel_access[panel_name] = {"authorized": False, "authorized_by_branch": None, "artifact_hash_passes": 0, "image_hash_byte_open_events": 0, "shard_hash_byte_open_events": 0, "dataset_access": {name: 0 for name in EVENT_FIELDS}, "one_shot_evaluation": False}
    _verify_artifacts(fit_images, fit_shards)
    stages = {"production_faithful": faithful, "ceiling_optimizer": ceiling}
    _reconcile_access(fit_dataset, stages, panel_access, holdouts)

    if smoke:
        decision = {"schema": contract.SEED_DECISION_SCHEMA, "ceiling_invoked": ceiling is not None, "qualifying_branch": None, "holdouts_authorized": False, "holdout_passes": None, "classification": "non_authoritative_smoke", "favorable": False, "aggregation_eligible": False, "shared_jepa_construction_licensed": False, "g2_licensed": False, "runtime_licensed": False}
    else:
        decision = contract.per_seed_decision(faithful, ceiling, holdout_checks)
    if primary is not None and _sha256_file(args.seed_20260710_result) != str(args.expected_seed_20260710_sha256):
        raise RuntimeError("seed-20260710 result changed during replication")
    if primary_attempt_marker is not None and _sha256_file(
        args.seed_20260710_attempt_marker
    ) != str(args.expected_seed_20260710_attempt_marker_sha256):
        raise RuntimeError("seed-20260710 attempt marker changed during replication")
    paths = _source_path_contract()
    with ThreadPoolExecutor(max_workers=SOURCE_WORKERS) as pool:
        source_end = dict(zip(SOURCE_ROLES, pool.map(lambda role: _sha256_file(paths[role]), SOURCE_ROLES), strict=True))
    if source_end != source_start:
        raise RuntimeError("N32 sources changed during execution")
    for record in _input_contract().values():
        if _sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("bound N32 input changed during execution")
    if _sha256_file(args.implementation_manifest) != str(
        args.expected_implementation_manifest_sha256
    ):
        raise RuntimeError("implementation manifest changed during execution")
    if attempt_marker is not None and _sha256_file(
        _canonical_attempt_marker(int(args.seed))
    ) != attempt_marker_file_sha256:
        raise RuntimeError("authoritative attempt marker changed during execution")
    git_end = _git_snapshot()
    authoritative = not smoke
    access_ledger = {
        "schema": "lewm_go2_dynamic_cartesian_n32_access_ledger_v1",
        "panels": panel_access,
        "fit_dataset_totals": fit_dataset.snapshot(),
        "sidecar": {"manifest_byte_opens": 2, "train_role_byte_opens": 2, "checkpoint_selection_role_byte_opens": 0, "probability_calibration_role_byte_opens": 0, "g2_evaluation_role_byte_opens": 0},
        "dataset_roles": {"train": {"panel_transition_rows_joined": 480, "model_outputs": sum(int(stage["training_access"]["model_output_frames"] + stage["fit_evaluation_access"]["model_output_frames"]) for stage in stages.values() if stage is not None) + (0 if holdouts is None else 1920)}, "checkpoint_selection": {"image_byte_opens": 0, "label_shard_byte_opens": 0, "model_outputs": 0}, "probability_calibration": {"image_byte_opens": 0, "label_shard_byte_opens": 0, "model_outputs": 0}, "g2_evaluation": {"image_byte_opens": 0, "label_shard_byte_opens": 0, "model_outputs": 0}},
        "holdout_payloads_opened_only_after_terminal_fit_pass": True,
        "wrong_rgb_target_attitude_frames": sum(int(stage["fit_evaluation_access"]["model_attitude_frames"] * 2 // 3) for stage in stages.values() if stage is not None) + (0 if holdouts is None else 1280),
        "non_train_image_opens": 0,
        "non_train_label_shard_opens": 0,
        "non_train_model_outputs": 0,
        "controlled_metadata_reads": {
            "implementation_manifest_byte_opens": 2,
            "source_byte_opens": {role: 2 for role in SOURCE_ROLES},
            "input_byte_opens": {name: 2 for name in _input_contract()},
            "seed_20260710_result_byte_opens": 0 if primary is None else 2,
            "authoritative_attempt_marker_byte_opens": 0 if smoke else 2,
            "seed_20260710_attempt_marker_byte_opens": (
                0 if primary_attempt_marker is None else 2
            ),
        },
    }
    core = {
        "schema": contract.RESULT_SCHEMA if authoritative else contract.SMOKE_RESULT_SCHEMA,
        "authoritative": authoritative,
        "aggregation_eligible": authoritative,
        "promotion_eligible": False,
        "seed": int(args.seed),
        "created_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation,
        "execution": {"device": device_record, "determinism": determinism, "batch_size_frames": contract.BATCH_SIZE, "evaluation_combined_model_batch_size": 12, "evaluation_interval": SMOKE_EVALUATION_INTERVAL if smoke else contract.EVALUATION_INTERVAL, "branches": {name: ({**config, "updates": SMOKE_UPDATES} if smoke else config) for name, config in contract.BRANCH_CONFIGS.items()}, "source_workers": SOURCE_WORKERS, "native_threads_per_worker": 1, "fp32_no_autocast_amp_compile_quantization_or_query_chunking": True},
        "contract": {"path": str(BINDING_PATH), "sha256": contract.EXECUTION_BINDING_SHA256},
        "preoutput_amendment": {"path": str(AMENDMENT_PATH), "sha256": contract.PREOUTPUT_AMENDMENT_SHA256},
        "attempt_control_amendment": {
            "path": str(ATTEMPT_AMENDMENT_PATH),
            "sha256": contract.ATTEMPT_CONTROL_AMENDMENT_SHA256,
        },
        "attempt_marker": (
            None
            if attempt_marker is None
            else {
                "path": str(_canonical_attempt_marker(int(args.seed))),
                "sha256": str(attempt_marker_file_sha256),
                "content_sha256": attempt_marker["content_sha256"],
            }
        ),
        "implementation_manifest": {"path": str(args.implementation_manifest), "sha256": str(args.expected_implementation_manifest_sha256), "content_sha256": implementation["content_sha256"]},
        "implementation_manifest_content_sha256": implementation["content_sha256"],
        "inputs": {**_input_contract(), "seed_20260710_result": None if primary is None else {"path": str(args.seed_20260710_result), "sha256": str(args.expected_seed_20260710_sha256), "content_sha256": primary["content_sha256"]}},
        "source_hashes": source_end,
        "git": {"start": git_start, "end": git_end},
        "model_config": contract.MODEL_CONFIG,
        "model": {"class": "EgomotionBevJepa", "entrypoint": "occupancy_logits", "initialization": initialization, "all_invoked_branches_restart_same_initial_state": all(stage is None or stage["initial_state_sha256"] == initial_state_sha for stage in stages.values()), "n32_weights_are_not_checkpointed_or_promotable": True},
        "preprocessing": contract.PREPROCESSING_CONTRACT,
        "objective": contract.OBJECTIVE_CONTRACT,
        "projective_query_support": query_support,
        "panel_join": join_audit,
        "projection_parity": {"content_sha256": parity_result["content_sha256"], "frame_count": 320, "mismatched_cells": 0},
        "stages": stages,
        "qualifying_branch": qualifying_branch,
        "patch7_reference": patch7_reference,
        "holdouts": holdouts,
        "holdout_checks": holdout_checks,
        "decision": decision,
        "artifact_verification": {"fit_verified_before_first_payload_access": True, "fit_verified_after_last_model_access": True, "holdouts_verified_only_after_terminal_fit_pass": True, "holdouts_evaluated_once": holdouts is not None},
        "access_ledger": access_ledger,
        "publication": {"mode": "private_staging_hardlink_noreplace", "canonical_output": str(_canonical_output(args.seed)) if authoritative else None},
        "shared_jepa_construction_licensed": False,
        "g2_licensed": False,
        "runtime_licensed": False,
    }
    payload = {**core, "content_sha256": contract.canonical_json_sha256(core)}
    if authoritative:
        assert attempt_marker is not None
        assert attempt_marker_file_sha256 is not None
        contract.validate_authoritative_result(
            payload,
            args.seed,
            implementation,
            str(args.expected_implementation_manifest_sha256),
            attempt_marker,
            attempt_marker_file_sha256,
            primary_result=primary,
            primary_file_sha256=(None if primary is None else str(args.expected_seed_20260710_sha256)),
            primary_attempt_marker=primary_attempt_marker,
            primary_attempt_marker_file_sha256=(
                None
                if primary_attempt_marker is None
                else str(args.expected_seed_20260710_attempt_marker_sha256)
            ),
        )
    _publish_json_exclusive(args.output, payload)
    print(json.dumps({"output": str(args.output), "content_sha256": payload["content_sha256"], "classification": decision["classification"], "favorable": decision["favorable"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
