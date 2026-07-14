#!/usr/bin/env python3
"""Run the sole lean Shared-V5 matched development training attempt.

The parent process reserves the output namespace before importing Torch or
opening Camera/Raw payloads.  It trains both matched arms, performs promoted-
only development selection and calibration, then asks an isolated CPU child
to reload the full pre-G2 checkpoint exactly once.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_CONTRACT_PATH = ROOT / "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py"
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_lewm_go2_shared_jepa_v5_matched_training_v1_contract",
    _CONTRACT_PATH,
)
if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
    raise ImportError("cannot load the matched-training contract directly")
contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
sys.modules[_CONTRACT_SPEC.name] = contract
_CONTRACT_SPEC.loader.exec_module(contract)


class DevelopmentGateFailure(RuntimeError):
    """The one attempt completed science but did not qualify for pre-G2."""


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_regular(path: Path, *, expected_sha256: str | None = None) -> bytes:
    if path.is_symlink():
        raise PermissionError(f"symlink input forbidden: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"input is not regular: {path}")
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
    if _fingerprint(before) != _fingerprint(after):
        raise RuntimeError(f"input changed while read: {path}")
    raw = b"".join(chunks)
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise PermissionError(f"input hash changed: {path}")
    return raw


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_json(path: Path, core: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    value = contract.with_content_sha256(dict(core))
    raw = contract.canonical_json_bytes(value) + b"\n"
    _write_exclusive(path, raw)
    return value, raw


def _output_binding(
    output_root: Path,
    relative: str,
    value: Mapping[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    del output_root
    return contract.artifact_binding(
        relative,
        raw,
        content_sha256=str(value["content_sha256"]),
    )


def _read_bound(path: Path, binding: Mapping[str, Any]) -> bytes:
    validated = contract.validate_binding(binding, path=path.relative_to(ROOT).as_posix())
    raw = _read_regular(path, expected_sha256=validated["file_sha256"])
    if len(raw) != validated["byte_count"]:
        raise PermissionError(f"bound byte count changed: {path}")
    return raw


def _parse_jsonl(raw: bytes, *, name: str) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n") or b"\n\n" in raw:
        raise ValueError(f"{name} is not canonical nonempty JSONL")
    result: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{name} row {index} is not ASCII JSON") from error
        if type(value) is not dict or contract.canonical_json_bytes(value) != line:
            raise ValueError(f"{name} row {index} is noncanonical")
        core = dict(value)
        declared = core.pop("content_sha256", None)
        if not contract.is_sha256(declared) or contract.canonical_json_sha256(core) != declared:
            raise ValueError(f"{name} row {index} self hash changed")
        result.append(value)
    return result


def _load_review_and_authorization(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, str],
]:
    sources = contract.current_source_bindings(ROOT)
    review_path = ROOT / contract.REVIEW_RELATIVE_PATH
    review_raw = _read_regular(review_path, expected_sha256=review_file_sha256)
    review = contract.parse_canonical_json(review_raw, name="independent review")
    contract.validate_review(review, expected_sources=sources)
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    authorization_path = ROOT / contract.AUTHORIZATION_RELATIVE_PATH
    authorization_raw = _read_regular(
        authorization_path,
        expected_sha256=authorization_file_sha256,
    )
    authorization = contract.parse_canonical_json(
        authorization_raw,
        name="execution authorization",
    )
    contract.validate_authorization(
        authorization,
        review_binding=review_binding,
    )
    return review, review_raw, authorization, authorization_raw, sources


def _validate_parent_environment() -> dict[str, Any]:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("exact run requires python -I -B")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise PermissionError("exact run requires HIP_VISIBLE_DEVICES=0")
    present = [
        name
        for name in contract.CONFLICTING_ACCELERATOR_ENVIRONMENT
        if name in os.environ
    ]
    if present:
        raise PermissionError("conflicting accelerator selectors are present: " + ", ".join(present))
    threads = {name: os.environ.get(name) for name in contract.THREAD_ENVIRONMENT}
    if any(value != "1" for value in threads.values()):
        raise PermissionError("all six native thread selectors must equal one")
    return {
        "hip_visible_devices": "0",
        "conflicting_selectors_absent": True,
        "native_thread_environment": threads,
        "isolated_python": True,
        "bytecode_disabled": True,
    }


def _reserve_output(
    output_root: Path,
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
    environment: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("matched-training attempt already has terminal or partial state")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output_root, mode=0o700)
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    authorization_binding = contract.artifact_binding(
        contract.AUTHORIZATION_RELATIVE_PATH,
        authorization_raw,
        content_sha256=str(authorization["content_sha256"]),
    )
    attempt_identity = contract.canonical_json_sha256(
        {
            "schema": f"{contract.SCHEMA_PREFIX}_attempt_identity_v1",
            "review": review_binding,
            "authorization": authorization_binding,
            "science_contract_sha256": contract.canonical_json_sha256(
                contract.science_contract()
            ),
        }
    )
    core = {
        "schema": contract.RESERVATION_SCHEMA,
        "status": "reserved_before_torch_camera_raw_or_rgb",
        "attempt_index": 1,
        "maximum_attempts": 1,
        "attempt_identity": attempt_identity,
        "independent_review": review_binding,
        "execution_authorization": authorization_binding,
        "reviewed_sources": dict(sources),
        "science_contract": contract.science_contract(),
        "raw": authorization["raw"],
        "camera": authorization["camera"],
        "environment": dict(environment),
        "torch_imported_before_reservation": False,
        "camera_or_raw_opened_before_reservation": False,
        "retry_authorized": False,
    }
    try:
        reservation, reservation_raw = _publish_json(
            output_root / "reservation.json",
            core,
        )
    except BaseException as error:
        failure_core = {
            "schema": contract.FAILURE_SCHEMA,
            "status": "failed_reservation_commit",
            "stage": "reservation_commit",
            "attempt_identity": attempt_identity,
            "error": {"type": type(error).__name__, "message": str(error)},
            "torch_imported": False,
            "camera_raw_or_rgb_opened": False,
            "g2_attempted": False,
            "heldout_open_count": 0,
            "retry_authorized": False,
        }
        try:
            _publish_json(output_root / "reservation_failed.json", failure_core)
        except BaseException as terminal_error:
            raise RuntimeError(
                "reservation commit and terminalization both failed"
            ) from terminal_error
        raise
    return reservation, reservation_raw


@dataclass(frozen=True)
class Runtime:
    np: Any
    Image: Any
    torch: Any
    F: Any
    model_module: Any
    loss_adapter: Any
    FitModel: Any
    MetricAccumulator: Any
    derive_targets: Any
    soft_rasterize: Any


def _load_runtime() -> Runtime:
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from lewm.benchmarks.go2_observable_camera_ray_fit_v4_metrics import (
        ObservableCameraRayFitV4MetricAccumulator,
    )
    from lewm.models.observable_camera_ray_evidence_v4 import (
        ObservableCameraRayEvidenceV4Model,
    )
    from lewm.models.observable_camera_ray_evidence_v4_training import (
        derive_observable_camera_ray_evidence_v4_targets,
        soft_rasterize_observable_camera_ray_evidence_v4,
    )
    from lewm.models import shared_observable_camera_ray_jepa_v5 as model_module
    from lewm.models import (
        shared_observable_camera_ray_jepa_v5_full_training_v4_loss as loss_adapter,
    )

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    return Runtime(
        np=np,
        Image=Image,
        torch=torch,
        F=F,
        model_module=model_module,
        loss_adapter=loss_adapter,
        FitModel=ObservableCameraRayEvidenceV4Model,
        MetricAccumulator=ObservableCameraRayFitV4MetricAccumulator,
        derive_targets=derive_observable_camera_ray_evidence_v4_targets,
        soft_rasterize=soft_rasterize_observable_camera_ray_evidence_v4,
    )


def _tensor_manifest(runtime: Runtime, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for name, value in sorted(state.items()):
        if type(name) is not str or not isinstance(value, runtime.torch.Tensor):
            raise TypeError("state entries must be named tensors")
        tensor = value.detach().to(device="cpu").contiguous()
        result.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(
                    tensor.view(runtime.torch.uint8).numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
    if not result:
        raise ValueError("state is empty")
    return result


def _normalize_camera_gate_checkpoint_binding(value: object) -> dict[str, Any]:
    try:
        leaf = contract.validate_binding(value, path="checkpoint.pt")
    except (TypeError, ValueError) as error:
        raise PermissionError("Camera gate checkpoint binding changed") from error
    return {**leaf, "path": contract.CAMERA_CHECKPOINT_RELATIVE_PATH}


def _camera_model_after_reservation(
    runtime: Runtime,
    authorization: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    camera = authorization["camera"]
    gate_raw = _read_bound(ROOT / camera["gate"]["path"], camera["gate"])
    gate = contract.parse_canonical_json(gate_raw, name="Camera N320 gate")
    numeric = gate.get("numeric_gate")
    if (
        gate.get("schema")
        != "lewm_go2_observable_camera_ray_fit_v4_n320_compute_scaled_v1_gate_v1"
        or gate.get("status") != "passed"
        or gate.get("row", {}).get("seed") != 20260710
        or gate.get("row", {}).get("fit_size") != 320
        or gate.get("row", {}).get("updates") != 40_000
        or gate.get("check_count") != 26
        or gate.get("failure_count") != 0
        or gate.get("passes") is not True
        or type(numeric) is not dict
        or numeric.get("passes") is not True
        or numeric.get("failure_count") != 0
        or gate.get("retry_authorized") is not False
        or type(gate.get("artifacts")) is not dict
        or _normalize_camera_gate_checkpoint_binding(
            gate["artifacts"].get("checkpoint")
        )
        != camera["checkpoint"]
        or gate.get("licenses", {}).get("shared_v5_development_use_authorized")
        is not True
        or any(
            gate.get("licenses", {}).get(name) is not False
            for name in (
                "g2_authorized",
                "navigation_authorized",
                "heldout_authorized",
                "production_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise PermissionError("Camera N320 did not terminally pass all 26 checks")
    checkpoint_raw = _read_bound(
        ROOT / camera["checkpoint"]["path"],
        camera["checkpoint"],
    )
    checkpoint = runtime.torch.load(
        io.BytesIO(checkpoint_raw),
        map_location="cpu",
        weights_only=True,
    )
    fields = {
        "schema",
        "model_class",
        "state_manifest",
        "metadata",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "state_dict",
        "content_sha256",
    }
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != fields
        or checkpoint["schema"]
        != "lewm_go2_observable_camera_ray_fit_v4_development_checkpoint_v2"
        or checkpoint["model_class"] != "ObservableCameraRayEvidenceV4Model"
        or checkpoint["authoritative"] is not False
        or checkpoint["aggregation_eligible"] is not False
        or checkpoint["promotion_eligible"] is not False
        or type(checkpoint["state_dict"]) is not dict
    ):
        raise PermissionError("Camera checkpoint schema or scope changed")
    manifest = _tensor_manifest(runtime, checkpoint["state_dict"])
    semantic = {
        name: checkpoint[name]
        for name in (
            "schema",
            "model_class",
            "state_manifest",
            "metadata",
            "authoritative",
            "aggregation_eligible",
            "promotion_eligible",
        )
    }
    if (
        checkpoint["state_manifest"] != manifest
        or checkpoint["content_sha256"] != contract.canonical_json_sha256(semantic)
        or checkpoint["content_sha256"] != camera["checkpoint"]["content_sha256"]
    ):
        raise PermissionError("Camera checkpoint tensor or semantic hash changed")
    fit = runtime.FitModel()
    fit.load_state_dict(checkpoint["state_dict"], strict=True)
    return fit, gate, dict(camera["checkpoint"])


class RawInputs:
    def __init__(
        self,
        runtime: Runtime,
        authorization: Mapping[str, Any],
    ) -> None:
        self.runtime = runtime
        self.consumed: dict[str, dict[str, Any]] = {}
        raw_auth = authorization["raw"]
        manifest_raw = _read_bound(
            ROOT / raw_auth["manifest"]["path"],
            raw_auth["manifest"],
        )
        audit_raw = _read_bound(
            ROOT / raw_auth["audit"]["path"],
            raw_auth["audit"],
        )
        self.manifest = contract.validate_raw_manifest(
            contract.parse_canonical_json(manifest_raw, name="Raw V13 manifest")
        )
        self.audit = contract.validate_raw_audit(
            contract.parse_canonical_json(audit_raw, name="Raw V13 audit")
        )
        for binding, kind in (
            (raw_auth["manifest"], "raw_manifest"),
            (raw_auth["audit"], "raw_audit"),
        ):
            self._record(
                path=str(binding["path"]),
                expected_sha256=str(binding["file_sha256"]),
                byte_count=int(binding["byte_count"]),
                role="authority",
                arm="shared",
                stage="input_validation",
                kind=kind,
            )
        self.root = ROOT / contract.RAW_ROOT_RELATIVE_PATH
        self.inventory = {
            str(item["path"]): dict(item) for item in self.manifest["files"]
        }
        if len(self.inventory) != len(self.manifest["files"]):
            raise PermissionError("Raw V13 file inventory contains duplicates")
        self.array_cache: dict[str, Any] = {}
        self.shard_cache: dict[str, dict[str, Any]] = {}
        self.frame_cache: dict[str, dict[str, Any]] = {}
        self.frame_sources: dict[str, tuple[str, ...]] = {}
        pair_raw = self.read_dataset(
            "pairs.jsonl", role="index", arm="shared", stage="input_validation"
        )
        endpoint_raw = self.read_dataset(
            "endpoints.jsonl", role="index", arm="shared", stage="input_validation"
        )
        if (
            hashlib.sha256(pair_raw).hexdigest()
            != self.manifest["pair_index"]["file_sha256"]
            or hashlib.sha256(endpoint_raw).hexdigest()
            != self.manifest["endpoint_index"]["file_sha256"]
        ):
            raise PermissionError("Raw V13 index binding changed")
        self.pairs = _parse_jsonl(pair_raw, name="Raw V13 pairs")
        endpoint_rows = _parse_jsonl(endpoint_raw, name="Raw V13 endpoints")
        self.endpoints = {
            str(item["endpoint_identity_sha256"]): item for item in endpoint_rows
        }
        self._validate_indexes(endpoint_rows)

    def _record(
        self,
        *,
        path: str,
        expected_sha256: str,
        byte_count: int,
        role: str,
        arm: str,
        stage: str,
        kind: str,
    ) -> None:
        record = self.consumed.get(path)
        if record is None:
            self.consumed[path] = {
                "path": path,
                "file_sha256": expected_sha256,
                "byte_count": byte_count,
                "kind": kind,
                "roles": [role],
                "arms": [arm],
                "stages": [stage],
            }
            return
        if (
            record["file_sha256"] != expected_sha256
            or record["byte_count"] != byte_count
            or record["kind"] != kind
        ):
            raise PermissionError("consumed source identity changed")
        for key, value in (("roles", role), ("arms", arm), ("stages", stage)):
            if value not in record[key]:
                record[key].append(value)

    def read_dataset(
        self,
        relative: str,
        *,
        role: str,
        arm: str,
        stage: str,
    ) -> bytes:
        contract.safe_relative_path(relative, name="Raw dataset relative path")
        record = self.inventory.get(relative)
        if type(record) is not dict:
            raise PermissionError(f"Raw V13 inventory does not bind {relative}")
        raw = _read_regular(
            self.root / relative,
            expected_sha256=str(record["file_sha256"]),
        )
        if len(raw) != record["byte_count"]:
            raise PermissionError("Raw V13 inventory byte count changed")
        self._record(
            path=f"{contract.RAW_ROOT_RELATIVE_PATH}/{relative}",
            expected_sha256=str(record["file_sha256"]),
            byte_count=len(raw),
            role=role,
            arm=arm,
            stage=stage,
            kind="raw_supervision",
        )
        return raw

    def read_rgb(
        self,
        relative: str,
        expected_sha256: str,
        *,
        role: str,
        arm: str,
        stage: str,
    ) -> bytes:
        contract.safe_relative_path(relative, name="development RGB path")
        if role not in contract.ROLES:
            raise PermissionError("RGB role escaped the three development roles")
        raw = _read_regular(ROOT / relative, expected_sha256=expected_sha256)
        self._record(
            path=relative,
            expected_sha256=expected_sha256,
            byte_count=len(raw),
            role=role,
            arm=arm,
            stage=stage,
            kind="development_rgb",
        )
        return raw

    def _validate_indexes(self, endpoint_rows: Sequence[Mapping[str, Any]]) -> None:
        if (
            len(self.pairs) != 5172
            or len(endpoint_rows) != 9460
            or len(self.endpoints) != 9460
            or contract.canonical_json_sha256(
                [item["content_sha256"] for item in self.pairs]
            )
            != contract.RAW_ORDERED_PAIR_SHA256
            or contract.canonical_json_sha256(
                [item["content_sha256"] for item in endpoint_rows]
            )
            != contract.RAW_ENDPOINT_INDEX_ORDER_SHA256
        ):
            raise PermissionError("Raw V13 index population or ordering changed")
        pair_ids = {str(item["content_sha256"]) for item in self.pairs}
        if len(pair_ids) != len(self.pairs):
            raise PermissionError("Raw V13 pair identities repeat")
        for role in contract.ROLES:
            if (
                sum(item.get("dataset_role") == role for item in self.pairs)
                != contract.ROLE_COUNTS[role]["pairs"]
                or sum(item.get("dataset_role") == role for item in endpoint_rows)
                != contract.ROLE_COUNTS[role]["unique_endpoints"]
            ):
                raise PermissionError(f"Raw V13 {role} population changed")
        for pair in self.pairs:
            role = pair.get("dataset_role")
            family = pair.get("family")
            current = self.endpoints.get(str(pair.get("current_endpoint_sha256")))
            next_ = self.endpoints.get(str(pair.get("next_endpoint_sha256")))
            if (
                role not in contract.ROLES
                or family not in contract.FAMILIES
                or type(current) is not dict
                or type(next_) is not dict
                or any(
                    item.get("dataset_role") != role
                    or item.get("family") != family
                    or item.get("scene_id") != pair.get("scene_id")
                    for item in (current, next_)
                )
            ):
                raise PermissionError("Raw V13 pair crossed role, family, scene or endpoint")

    def role_pairs(self, role: str) -> list[dict[str, Any]]:
        if role not in contract.ROLES:
            raise PermissionError("dataset role is not development-authorized")
        rows = [item for item in self.pairs if item["dataset_role"] == role]
        if len(rows) != contract.ROLE_COUNTS[role]["pairs"]:
            raise PermissionError("dataset role count changed")
        return rows

    def _shard(self, endpoint: Mapping[str, Any], *, arm: str, stage: str) -> dict[str, Any]:
        relative = str(endpoint["scene_shard"])
        cached = self.shard_cache.get(relative)
        if cached is not None:
            return cached
        raw = self.read_dataset(
            relative,
            role=str(endpoint["dataset_role"]),
            arm=arm,
            stage=stage,
        )
        value = contract.parse_canonical_json(raw, name="Raw V13 shard")
        manifest_rows = {
            item["path"]: item for item in self.manifest["shards"]
        }
        bound = manifest_rows.get(relative)
        if (
            type(bound) is not dict
            or value.get("content_sha256") != bound.get("content_sha256")
            or value.get("dataset_role") != endpoint["dataset_role"]
            or value.get("family") != endpoint["family"]
            or value.get("scene_id") != endpoint["scene_id"]
        ):
            raise PermissionError("Raw V13 shard binding changed")
        self.shard_cache[relative] = value
        return value

    def _row_array(
        self,
        endpoint: Mapping[str, Any],
        shard: Mapping[str, Any],
        filename: str,
        *,
        arm: str,
        stage: str,
    ) -> Any:
        shard_path = Path(str(endpoint["scene_shard"]))
        relative = (shard_path.parent / filename).as_posix()
        cache = self.array_cache.get(relative)
        records = {item["path"]: item for item in shard["files"]}
        record = records.get(filename)
        layout = {item["path"]: item for item in contract.RAW_ARRAY_LAYOUT}[filename]
        if (
            type(record) is not dict
            or record.get("dtype") != layout["dtype"]
            or record.get("shape", [])[1:] != layout["trailing_shape"]
            or self.inventory.get(relative, {}).get("file_sha256")
            != record.get("file_sha256")
        ):
            raise PermissionError("Raw V13 shard array contract changed")
        if cache is None:
            raw = self.read_dataset(
                relative,
                role=str(endpoint["dataset_role"]),
                arm=arm,
                stage=stage,
            )
            cache = self.runtime.np.frombuffer(
                bytearray(raw),
                dtype=self.runtime.np.dtype(record["dtype"]),
            ).reshape(tuple(record["shape"]))
            self.array_cache[relative] = cache
        row = int(endpoint["shard_row"])
        if not 0 <= row < cache.shape[0]:
            raise PermissionError("Raw V13 shard row escaped")
        return self.runtime.torch.from_numpy(cache[row])

    def frame(self, endpoint_id: str, *, role: str, arm: str, stage: str) -> dict[str, Any]:
        endpoint = self.endpoints.get(endpoint_id)
        if type(endpoint) is not dict or endpoint.get("dataset_role") != role:
            raise PermissionError("endpoint crossed its dataset role")
        cached = self.frame_cache.get(endpoint_id)
        if cached is not None:
            for path in self.frame_sources[endpoint_id]:
                record = self.consumed[path]
                self._record(
                    path=path,
                    expected_sha256=record["file_sha256"],
                    byte_count=record["byte_count"],
                    role=role,
                    arm=arm,
                    stage=stage,
                    kind=record["kind"],
                )
            return cached
        shard = self._shard(endpoint, arm=arm, stage=stage)
        image_raw = self.read_rgb(
            str(endpoint["image_path_metadata_only"]),
            str(endpoint["image_sha256_commitment_only"]),
            role=role,
            arm=arm,
            stage=stage,
        )
        with self.runtime.Image.open(io.BytesIO(image_raw)) as decoded:
            image = decoded.convert("RGB").resize(
                (112, 112),
                self.runtime.Image.Resampling.BILINEAR,
            )
            array = self.runtime.np.asarray(image, dtype=self.runtime.np.float32) / 255.0
        tensor = self.runtime.torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
        mean = tensor.new_tensor(self.runtime.model_module.NORMALIZATION_MEAN)[:, None, None]
        std = tensor.new_tensor(self.runtime.model_module.NORMALIZATION_STD)[:, None, None]
        names = {
            "camera_origin": "camera_origin_body_m.f4",
            "camera_basis": "camera_basis_body_fru.f4",
            "ground": "ground_plane_z_body_m.f4",
            "ground_in_frustum": "ground_support_in_frustum.u1",
            "ground_clear": "ground_support_clear_to_target.u1",
            "pixel_hit": "pixel_hit_mask.u1",
            "pixel_distance": "pixel_first_hit_distance_m.f4",
            "raster_labels": "raster_labels.u1",
        }
        result = {"image": (tensor - mean) / std, "family": endpoint["family"]}
        for key, filename in names.items():
            result[key] = self._row_array(
                endpoint,
                shard,
                filename,
                arm=arm,
                stage=stage,
            )
        self.frame_cache[endpoint_id] = result
        shard_path = Path(str(endpoint["scene_shard"]))
        self.frame_sources[endpoint_id] = (
            str(endpoint["image_path_metadata_only"]),
            f"{contract.RAW_ROOT_RELATIVE_PATH}/{shard_path.as_posix()}",
            *(
                f"{contract.RAW_ROOT_RELATIVE_PATH}/"
                f"{(shard_path.parent / filename).as_posix()}"
                for filename in names.values()
            ),
        )
        for path in self.frame_sources[endpoint_id]:
            record = self.consumed[path]
            self._record(
                path=path,
                expected_sha256=record["file_sha256"],
                byte_count=record["byte_count"],
                role=role,
                arm=arm,
                stage=stage,
                kind=record["kind"],
            )
        return result

    def rehash_consumed(self) -> dict[str, Any]:
        for path, record in sorted(self.consumed.items()):
            raw = _read_regular(ROOT / path, expected_sha256=record["file_sha256"])
            if len(raw) != record["byte_count"]:
                raise PermissionError("consumed payload changed before completion")
        return {
            "unique_file_count": len(self.consumed),
            "raw_supervision_file_count": sum(
                item["kind"] == "raw_supervision" for item in self.consumed.values()
            ),
            "development_rgb_file_count": sum(
                item["kind"] == "development_rgb" for item in self.consumed.values()
            ),
            "all_consumed_files_rehashed": True,
            "records_sha256": contract.canonical_json_sha256(
                [self.consumed[name] for name in sorted(self.consumed)]
            ),
            "records": [self.consumed[name] for name in sorted(self.consumed)],
        }


class Trainer:
    def __init__(
        self,
        runtime: Runtime,
        inputs: RawInputs,
        output_root: Path,
        reservation: Mapping[str, Any],
    ) -> None:
        self.r = runtime
        self.inputs = inputs
        self.output_root = output_root
        self.reservation = reservation

    def device(self) -> tuple[Any, dict[str, Any]]:
        torch = self.r.torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise PermissionError("matched training requires exactly one visible GPU")
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        if (
            "r9700" not in str(properties.name).casefold().replace(" ", "")
            or int(properties.total_memory)
            < contract.MINIMUM_R9700_TOTAL_MEMORY_BYTES
        ):
            raise PermissionError("matched training requires the discrete R9700")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return device, {
            "device": "cuda:0",
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "minimum_total_memory_bytes": contract.MINIMUM_R9700_TOTAL_MEMORY_BYTES,
            "visible_device_count": 1,
            "torch_version": str(torch.__version__),
            "hip_version": str(torch.version.hip),
        }

    def initialize(self, fit: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        torch = self.r.torch
        torch.manual_seed(contract.INITIALIZATION_SEED)
        model = self.r.model_module.SharedObservableCameraRayJepaV5()
        migration = model.migrate_from_fit_model(fit)
        state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in sorted(model.state_dict().items())
        }
        state_sha = self.r.model_module.tensor_state_dict_sha256(state)
        del model
        receipt = {
            "schema": f"{contract.SCHEMA_PREFIX}_initialization_v1",
            "seed": contract.INITIALIZATION_SEED,
            "device": "cpu",
            "precision": "float32",
            "fit_model_state_sha256": migration.fit_model_state_sha256,
            "shared_encoder_state_sha256": migration.shared_encoder_state_sha256,
            "evidence_head_state_sha256": migration.evidence_head_state_sha256,
            "migrated_head_key_count": migration.migrated_head_key_count,
            "hard_sync_count": 1,
            "complete_state_sha256": state_sha,
            "arm_initial_state_sha256": {arm: state_sha for arm in contract.ARMS},
            "identical_before_optimizer_construction": True,
        }
        return state, receipt

    def schedule(self, train_pairs: Sequence[Mapping[str, Any]]) -> tuple[list[int], dict[str, Any]]:
        generator = self.r.torch.Generator(device="cpu")
        generator.manual_seed(contract.SCHEDULE_SEED)
        indices: list[int] = []
        while len(indices) < contract.PRESENTATION_COUNT:
            indices.extend(
                self.r.torch.randperm(
                    contract.TRAIN_PAIR_COUNT,
                    generator=generator,
                ).tolist()
            )
        indices = indices[: contract.PRESENTATION_COUNT]
        pair_ids = [str(item["content_sha256"]) for item in train_pairs]
        core = contract.schedule_core(indices, pair_ids)
        value = contract.with_content_sha256(
            {**core, "presentation_indices": indices}
        )
        raw = contract.canonical_json_bytes(value) + b"\n"
        _write_exclusive(self.output_root / "schedule.json", raw)
        return indices, value

    def commanded_table(
        self,
        train_pairs: Sequence[Mapping[str, Any]],
    ) -> tuple[list[str], Any]:
        vocabulary = sorted({str(item["primitive"]) for item in train_pairs})
        if len(vocabulary) != 9:
            raise PermissionError("train primitive vocabulary changed")
        rows = []
        for primitive in vocabulary:
            values = self.r.torch.tensor(
                [
                    item["relative_se2_current_frame"]
                    for item in train_pairs
                    if item["primitive"] == primitive
                ],
                dtype=self.r.torch.float32,
            )
            if values.numel() == 0:
                raise PermissionError("train primitive has no rows")
            rows.append(self.r.torch.quantile(values, 0.5, dim=0))
        return vocabulary, self.r.torch.stack(rows)

    def supervision(self, frames: Sequence[Mapping[str, Any]], device: Any) -> Any:
        stack = lambda name: self.r.torch.stack([item[name] for item in frames]).to(device)
        return self.r.model_module.ObservableCameraRayV4FrameSupervisionV5(
            pixel_hit_mask=stack("pixel_hit").bool(),
            pixel_first_hit_distance_m=stack("pixel_distance").float(),
            ground_support_in_frustum=stack("ground_in_frustum").bool(),
            ground_support_clear_to_target=stack("ground_clear").bool(),
            target_raster_labels=stack("raster_labels").long(),
        )

    def batch(
        self,
        pairs: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        vocabulary: Sequence[str],
        commanded_table: Any,
        device: Any,
        *,
        role: str,
        arm: str,
        stage: str,
    ) -> dict[str, Any]:
        selected = [pairs[index] for index in indices]
        if any(item["dataset_role"] != role for item in selected):
            raise PermissionError("batch crossed dataset roles")
        current = [
            self.inputs.frame(
                str(item["current_endpoint_sha256"]),
                role=role,
                arm=arm,
                stage=stage,
            )
            for item in selected
        ]
        next_ = [
            self.inputs.frame(
                str(item["next_endpoint_sha256"]),
                role=role,
                arm=arm,
                stage=stage,
            )
            for item in selected
        ]
        action_indices = [vocabulary.index(str(item["primitive"])) for item in selected]
        action = self.r.torch.zeros(
            (len(selected), len(vocabulary)),
            dtype=self.r.torch.float32,
            device=device,
        )
        action[
            self.r.torch.arange(len(selected), device=device),
            self.r.torch.tensor(action_indices, device=device),
        ] = 1.0
        wrong_action = self.r.torch.roll(action, shifts=1, dims=1)
        realized = self.r.torch.tensor(
            [item["relative_se2_current_frame"] for item in selected],
            dtype=self.r.torch.float32,
            device=device,
        )
        commanded = action @ commanded_table
        wrong_commanded = wrong_action @ commanded_table
        stack = lambda frames, name: self.r.torch.stack(
            [item[name] for item in frames]
        ).to(device)
        return {
            "forward": {
                "current_image": stack(current, "image"),
                "next_image": stack(next_, "image"),
                "action": action,
                "realized_delta_pose_current": realized,
                "commanded_delta_pose_current": commanded,
                "current_camera_origin_body_m": stack(current, "camera_origin").float(),
                "current_camera_basis_body_fru": stack(current, "camera_basis").float(),
                "current_ground_plane_z_body_m": stack(current, "ground").float(),
                "next_camera_origin_body_m": stack(next_, "camera_origin").float(),
                "next_camera_basis_body_fru": stack(next_, "camera_basis").float(),
                "next_ground_plane_z_body_m": stack(next_, "ground").float(),
                "next_prediction_mask": self.r.torch.ones(
                    (len(selected), 64, 64),
                    dtype=self.r.torch.bool,
                    device=device,
                ),
                "diagnostic_wrong_action": wrong_action,
                "diagnostic_wrong_action_delta_pose_current": wrong_commanded,
                "diagnostic_wrong_commanded_delta_pose_current": -commanded,
            },
            "current_supervision": self.supervision(current, device),
            "next_supervision": self.supervision(next_, device),
            "families": [str(item["family"]) for item in selected],
        }

    @staticmethod
    def backward_for_arm(joint: Any, arm: str) -> Any:
        if arm == "promoted_jepa":
            return joint.total
        if arm == "matched_no_jepa":
            return joint.observable_camera_ray_v4.total
        raise ValueError("unknown matched-training arm")

    def _snapshot(
        self,
        model: Any,
        *,
        arm: str,
        update: int,
        initial_sha: str,
        schedule_sha: str,
    ) -> tuple[bytes, dict[str, Any]]:
        state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in sorted(model.state_dict().items())
        }
        manifest = _tensor_manifest(self.r, state)
        state_sha = self.r.model_module.tensor_state_dict_sha256(state)
        metadata = {
            "schema": contract.SNAPSHOT_SCHEMA,
            "arm": arm,
            "update": update,
            "model_config": model.model_config.to_dict(),
            "state_manifest": manifest,
            "state_sha256": state_sha,
            "initialization_state_sha256": initial_sha,
            "schedule_content_sha256": schedule_sha,
            "optimizer_contract": contract.OPTIMIZER_CONTRACT,
            "development_only": True,
            "resume_authorized": False,
            "runtime_ready": False,
        }
        content_sha = contract.canonical_json_sha256(metadata)
        buffer = io.BytesIO()
        self.r.torch.save(
            {**metadata, "content_sha256": content_sha, "model_state_dict": state},
            buffer,
        )
        return buffer.getvalue(), {**metadata, "content_sha256": content_sha}

    def train_arm(
        self,
        *,
        arm: str,
        initial_state: Mapping[str, Any],
        schedule: Sequence[int],
        schedule_sha: str,
        train_pairs: Sequence[Mapping[str, Any]],
        vocabulary: Sequence[str],
        commanded_table: Any,
        device: Any,
        publish_updates: Sequence[int],
    ) -> dict[int, dict[str, Any]]:
        torch = self.r.torch
        torch.manual_seed(contract.INITIALIZATION_SEED)
        torch.cuda.manual_seed_all(contract.INITIALIZATION_SEED)
        model = self.r.model_module.SharedObservableCameraRayJepaV5().to(device)
        model.load_state_dict(initial_state, strict=True)
        initial_sha = self.r.model_module.tensor_state_dict_sha256(
            {name: value.detach().cpu() for name, value in model.state_dict().items()}
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=contract.learning_rate(1),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
            amsgrad=False,
        )
        snapshots: dict[int, dict[str, Any]] = {}
        trace: list[dict[str, Any]] = []
        for update in range(1, contract.UPDATE_COUNT + 1):
            learning_rate = contract.learning_rate(update)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            start = (update - 1) * contract.EFFECTIVE_BATCH_SIZE
            update_indices = schedule[start : start + contract.EFFECTIVE_BATCH_SIZE]
            for microbatch in range(contract.ACCUMULATION_STEPS):
                low = microbatch * contract.MICROBATCH_SIZE
                batch = self.batch(
                    train_pairs,
                    update_indices[low : low + contract.MICROBATCH_SIZE],
                    vocabulary,
                    commanded_table,
                    device,
                    role="train",
                    arm=arm,
                    stage="gradient",
                )
                pair = model.forward_training_pair(**batch["forward"])
                joint = self.r.loss_adapter.combine_joint_losses_v4(
                    model,
                    pair,
                    batch["current_supervision"],
                    batch["next_supervision"],
                )
                backward = self.backward_for_arm(joint, arm)
                if not bool(torch.isfinite(backward).item()):
                    raise FloatingPointError("matched-training loss became nonfinite")
                (backward / contract.ACCUMULATION_STEPS).backward()
                current = joint.observable_camera_ray_v4.current
                next_ = joint.observable_camera_ray_v4.next
                values = {
                    "backward": backward,
                    "joint_total": joint.total,
                    "jepa_total": joint.established_jepa.total,
                    "camera_pair_total": joint.observable_camera_ray_v4.total,
                    "current_hierarchical_first_hit_nll": current.hierarchical_first_hit_nll,
                    "current_target_bin_offset_smooth_l1": current.target_bin_offset_smooth_l1,
                    "current_ground_clear_distance_state_balanced_bce": current.ground_clear_distance_state_balanced_bce,
                    "current_derived_raster_hierarchical_bce": current.derived_raster_hierarchical_bce.total,
                    "current_derived_raster_cell_nll": current.derived_raster_cell_nll,
                    "next_hierarchical_first_hit_nll": next_.hierarchical_first_hit_nll,
                    "next_target_bin_offset_smooth_l1": next_.target_bin_offset_smooth_l1,
                    "next_ground_clear_distance_state_balanced_bce": next_.ground_clear_distance_state_balanced_bce,
                    "next_derived_raster_hierarchical_bce": next_.derived_raster_hierarchical_bce.total,
                    "next_derived_raster_cell_nll": next_.derived_raster_cell_nll,
                }
                observed = self.r.torch.stack(tuple(values.values())).detach().cpu().tolist()
                for name, value in zip(values, observed, strict=True):
                    sums[name] = sums.get(name, 0.0) + float(value) / 4.0
            gradient_before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_before).item()):
                raise FloatingPointError("matched-training gradient became nonfinite")
            gradient_after = math.sqrt(
                sum(
                    float(parameter.grad.detach().float().square().sum().cpu())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
            optimizer.step()
            model.update_ema_target_after_optimizer_step()
            trace.append(
                {
                    "schema": f"{contract.SCHEMA_PREFIX}_trace_row_v1",
                    "arm": arm,
                    "update": update,
                    "learning_rate": learning_rate,
                    "microbatch_count": 4,
                    "optimizer_step_count": update,
                    "ema_step_count": update,
                    "gradient_norm_before_clip": float(gradient_before.detach().cpu()),
                    "gradient_norm_after_clip": gradient_after,
                    "losses": sums,
                }
            )
            if update in publish_updates:
                raw, metadata = self._snapshot(
                    model,
                    arm=arm,
                    update=update,
                    initial_sha=initial_sha,
                    schedule_sha=schedule_sha,
                )
                relative = f"arms/{arm}/checkpoints/update_{update}.pt"
                _write_exclusive(self.output_root / relative, raw)
                binding = contract.artifact_binding(
                    relative,
                    raw,
                    content_sha256=metadata["content_sha256"],
                )
                snapshots[update] = {"binding": binding, "metadata": metadata}
        trace_raw = b"".join(contract.canonical_json_bytes(item) + b"\n" for item in trace)
        _write_exclusive(self.output_root / f"arms/{arm}/training_trace.jsonl", trace_raw)
        snapshots[-1] = {
            "trace": {
                "path": f"arms/{arm}/training_trace.jsonl",
                "file_sha256": hashlib.sha256(trace_raw).hexdigest(),
                "byte_count": len(trace_raw),
                "row_count": len(trace),
                "content_sha256": contract.canonical_json_sha256(trace),
            }
        }
        model.to("cpu")
        del model, optimizer
        torch.cuda.empty_cache()
        return snapshots

    def load_snapshot(self, record: Mapping[str, Any], device: Any) -> Any:
        binding = record["binding"]
        raw = _read_regular(
            self.output_root / binding["path"],
            expected_sha256=binding["file_sha256"],
        )
        if len(raw) != binding["byte_count"]:
            raise PermissionError("training snapshot byte count changed")
        value = self.r.torch.load(
            io.BytesIO(raw), map_location="cpu", weights_only=True
        )
        if (
            type(value) is not dict
            or value.get("schema") != contract.SNAPSHOT_SCHEMA
            or value.get("content_sha256") != binding["content_sha256"]
            or value.get("state_manifest") != _tensor_manifest(
                self.r, value.get("model_state_dict", {})
            )
            or value.get("state_sha256")
            != self.r.model_module.tensor_state_dict_sha256(
                value["model_state_dict"]
            )
        ):
            raise PermissionError("training snapshot changed")
        model = self.r.model_module.SharedObservableCameraRayJepaV5().to(device).eval()
        model.load_state_dict(value["model_state_dict"], strict=True)
        return model

    def _single_frame_pair(self, frame: Any) -> Any:
        torch = self.r.torch
        return self.r.model_module.SharedTrainingPairV5(
            current=frame,
            next=frame,
            predicted_next_bev=frame.bev,
            stop_gradient_target_next_bev=frame.bev.detach(),
            commanded_warped_current_bev=frame.bev,
            commanded_overlap_mask=torch.ones_like(frame.bev[:, :1], dtype=torch.bool),
            realized_warped_current_bev=frame.bev,
            realized_overlap_mask=torch.ones_like(frame.bev[:, :1], dtype=torch.bool),
            jepa=None,
        )

    @staticmethod
    def _flatten_physical(
        correct: Mapping[str, Any],
        wrong: Mapping[str, Any],
    ) -> dict[str, Any]:
        c_depth, w_depth = correct["pixel_hit_depth"], wrong["pixel_hit_depth"]
        c_raster, w_raster = correct["derived_raster"], wrong["derived_raster"]
        return {
            "pixel_first_hit_balanced_accuracy": correct["pixel_hit_no_hit"]["balanced_accuracy"],
            "depth_median_error_m": c_depth["median_absolute_error_m"],
            "depth_p95_error_m": c_depth["p95_absolute_error_m"],
            "ground_clear_balanced_accuracy": correct["ground_clear"]["overall"]["balanced_accuracy"],
            "distance_group_balanced_accuracy": [
                item["balanced_accuracy"]
                for item in correct["ground_clear"]["by_distance_m"].values()
                if item["count"] > 0
            ],
            "derived_raster_nll": c_raster["nll"],
            "derived_raster_balanced_accuracy": c_raster["balanced_accuracy"],
            "present_class_recall": {
                name: value
                for name, value in c_raster["class_recalls"].items()
                if value is not None
            },
            "wrong_rgb_pixel_balanced_accuracy_drop": correct["pixel_hit_no_hit"]["balanced_accuracy"] - wrong["pixel_hit_no_hit"]["balanced_accuracy"],
            "wrong_rgb_depth_median_error_increase_m": w_depth["median_absolute_error_m"] - c_depth["median_absolute_error_m"],
            "wrong_rgb_depth_p95_error_increase_m": w_depth["p95_absolute_error_m"] - c_depth["p95_absolute_error_m"],
            "wrong_rgb_ground_balanced_accuracy_drop": correct["ground_clear"]["overall"]["balanced_accuracy"] - wrong["ground_clear"]["overall"]["balanced_accuracy"],
            "wrong_rgb_raster_nll_increase": w_raster["nll"] - c_raster["nll"],
            "wrong_rgb_raster_balanced_accuracy_drop": c_raster["balanced_accuracy"] - w_raster["balanced_accuracy"],
        }

    def physical_metrics(
        self,
        model: Any,
        pairs: Sequence[Mapping[str, Any]],
        device: Any,
        *,
        arm: str,
        stage: str,
    ) -> tuple[dict[str, Any], float]:
        torch = self.r.torch
        correct = {scope: self.r.MetricAccumulator() for scope in contract.SCOPES}
        wrong = {scope: self.r.MetricAccumulator() for scope in contract.SCOPES}
        loss_sum = 0.0
        frame_count = 0
        ids_by_family = {
            family: sorted(
                {
                    str(pair[f"{side}_endpoint_sha256"])
                    for pair in pairs
                    if pair["family"] == family
                    for side in ("current", "next")
                }
            )
            for family in contract.FAMILIES
        }
        with torch.no_grad():
            for family, ids in ids_by_family.items():
                if len(ids) < 2:
                    raise ValueError(f"selection family has insufficient endpoints: {family}")
                wrong_ids = ids[1:] + ids[:1]
                for start in range(0, len(ids), contract.MICROBATCH_SIZE):
                    target_ids = ids[start : start + 4]
                    mapped_ids = wrong_ids[start : start + 4]
                    target = [
                        self.inputs.frame(item, role="checkpoint_selection", arm=arm, stage=stage)
                        for item in target_ids
                    ]
                    mapped = [
                        self.inputs.frame(item, role="checkpoint_selection", arm=arm, stage=stage)
                        for item in mapped_ids
                    ]
                    origin = torch.stack([item["camera_origin"] for item in target]).to(device).float()
                    basis = torch.stack([item["camera_basis"] for item in target]).to(device).float()
                    ground = torch.stack([item["ground"] for item in target]).to(device).float()
                    supervision = self.supervision(target, device)
                    targets = self.r.derive_targets(
                        pixel_hit_mask=supervision.pixel_hit_mask,
                        pixel_first_hit_distance_m=supervision.pixel_first_hit_distance_m,
                        ground_support_in_frustum=supervision.ground_support_in_frustum,
                        ground_support_clear_to_target=supervision.ground_support_clear_to_target,
                    )
                    outputs = []
                    for frames in (target, mapped):
                        online = model.forward_frame(
                            torch.stack([item["image"] for item in frames]).to(device),
                            origin,
                            basis,
                            ground,
                        )
                        soft = self.r.soft_rasterize(
                            online.evidence,
                            camera_origin_body_m=origin,
                            camera_basis_body_fru=basis,
                            pixel_ray_chunk_size=model.model_config.v4_pixel_ray_chunk_size,
                        )
                        outputs.append((online, soft))
                    for accumulator_set, output in zip((correct, wrong), outputs, strict=True):
                        online, soft = output
                        for scope in ("aggregate", family):
                            accumulator_set[scope].update(
                                raw_output=online.evidence,
                                targets=targets,
                                soft_raster=soft,
                                target_raster_labels=supervision.target_raster_labels,
                                families=[family] * len(target),
                            )
                    camera = self.r.loss_adapter.observable_camera_ray_v4_loss_v4(
                        model,
                        self._single_frame_pair(outputs[0][0]),
                        supervision,
                        supervision,
                        require_b4=False,
                    )
                    loss_sum += float(camera.total.cpu()) * len(target)
                    frame_count += len(target)
        metrics = {
            scope: self._flatten_physical(
                correct[scope].finalize(), wrong[scope].finalize()
            )
            for scope in contract.SCOPES
        }
        return metrics, loss_sum / frame_count

    def jepa_metrics(
        self,
        model: Any,
        pairs: Sequence[Mapping[str, Any]],
        vocabulary: Sequence[str],
        commanded_table: Any,
        device: Any,
        *,
        arm: str,
        stage: str,
    ) -> dict[str, Any]:
        torch = self.r.torch
        names = (
            "prediction",
            "persistence",
            "wrong_action_real",
            "wrong_action_persistence",
            "wrong_action",
            "wrong_action_sensitivity",
            "wrong_delta_real",
            "wrong_delta_persistence",
            "wrong_delta",
            "wrong_delta_sensitivity",
        )
        sums = {scope: {name: 0.0 for name in names} for scope in contract.SCOPES}
        counts = {scope: {name: 0 for name in names} for scope in contract.SCOPES}
        targets = {scope: [] for scope in contract.SCOPES}

        def normalized(prediction: Any, target: Any) -> Any:
            return (
                self.r.F.normalize(prediction, dim=1)
                - self.r.F.normalize(target, dim=1)
            ).square().mean(dim=1)

        def add(scope: str, name: str, values: Any, mask: Any) -> None:
            weight = mask.to(values.dtype)
            sums[scope][name] += float((values * weight).sum().cpu())
            counts[scope][name] += int(mask.sum().cpu())

        with torch.no_grad():
            for family in contract.FAMILIES:
                family_pairs = [item for item in pairs if item["family"] == family]
                if not family_pairs:
                    raise ValueError(f"selection family is empty: {family}")
                for start in range(0, len(family_pairs), 4):
                    chunk = family_pairs[start : start + 4]
                    batch = self.batch(
                        family_pairs,
                        list(range(start, start + len(chunk))),
                        vocabulary,
                        commanded_table,
                        device,
                        role="checkpoint_selection",
                        arm=arm,
                        stage=stage,
                    )
                    pair = model.forward_training_pair(**batch["forward"])
                    target = pair.stop_gradient_target_next_bev.detach()
                    prediction = pair.predicted_next_bev.detach()
                    persistence = pair.commanded_warped_current_bev.detach()
                    mask = pair.commanded_overlap_mask[:, 0].bool()
                    prediction_error = normalized(prediction, target)
                    persistence_error = normalized(persistence, target)
                    wrong_action, _warp, wrong_action_overlap = model.predict_from_command(
                        pair.current.bev.detach(),
                        batch["forward"]["diagnostic_wrong_action"],
                        batch["forward"]["diagnostic_wrong_action_delta_pose_current"],
                    )
                    wrong_delta, _warp, wrong_delta_overlap = model.predict_from_command(
                        pair.current.bev.detach(),
                        batch["forward"]["action"],
                        batch["forward"]["diagnostic_wrong_commanded_delta_pose_current"],
                    )
                    action_mask = mask & wrong_action_overlap[:, 0]
                    delta_mask = mask & wrong_delta_overlap[:, 0]
                    values = {
                        "prediction": (prediction_error, mask),
                        "persistence": (persistence_error, mask),
                        "wrong_action_real": (prediction_error, action_mask),
                        "wrong_action_persistence": (persistence_error, action_mask),
                        "wrong_action": (normalized(wrong_action, target), action_mask),
                        "wrong_action_sensitivity": (normalized(wrong_action, prediction), action_mask),
                        "wrong_delta_real": (prediction_error, delta_mask),
                        "wrong_delta_persistence": (persistence_error, delta_mask),
                        "wrong_delta": (normalized(wrong_delta, target), delta_mask),
                        "wrong_delta_sensitivity": (normalized(wrong_delta, prediction), delta_mask),
                    }
                    target_cpu = target.cpu()
                    for scope in ("aggregate", family):
                        for name, (observed, observed_mask) in values.items():
                            add(scope, name, observed, observed_mask)
                        targets[scope].append(target_cpu)

        result = {}
        for scope in contract.SCOPES:
            def mean(name: str) -> float:
                return sums[scope][name] / max(1, counts[scope][name])

            target = torch.cat(targets[scope]).float()
            target_std = float(target.std(dim=0, unbiased=False).mean())
            centered = target - target.mean(dim=0, keepdim=True)
            samples = centered.permute(0, 2, 3, 1).reshape(-1, centered.shape[1])
            if samples.shape[0] > 65_536:
                samples = samples[:: math.ceil(samples.shape[0] / 65_536)]
            covariance = samples.T @ samples / max(1, samples.shape[0] - 1)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            total = eigenvalues.sum()
            target_rank = 0.0
            if bool((total > 0).item()):
                probabilities = eigenvalues / total
                target_rank = float(
                    torch.exp(
                        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                    )
                )
            persistence = mean("persistence")
            result[scope] = {
                "prediction_valid_cell_count": counts[scope]["prediction"],
                "target_cross_sample_std_mean": target_std,
                "target_cross_sample_effective_rank": target_rank,
                "warped_persistence_target_change": persistence,
                "prediction_to_warped_persistence_ratio": mean("prediction") / max(persistence, 1e-8),
                "wrong_action_advantage_over_target_change": (mean("wrong_action") - mean("wrong_action_real")) / max(mean("wrong_action_persistence"), 1e-8),
                "wrong_commanded_delta_advantage_over_target_change": (mean("wrong_delta") - mean("wrong_delta_real")) / max(mean("wrong_delta_persistence"), 1e-8),
                "wrong_action_prediction_sensitivity": mean("wrong_action_sensitivity"),
                "wrong_commanded_delta_prediction_sensitivity": mean("wrong_delta_sensitivity"),
            }
        return result

    def evaluate_snapshot(
        self,
        record: Mapping[str, Any],
        *,
        update: int,
        selection_pairs: Sequence[Mapping[str, Any]],
        vocabulary: Sequence[str],
        commanded_table: Any,
        device: Any,
        arm: str,
        stage: str,
    ) -> dict[str, Any]:
        model = self.load_snapshot(record, device)
        physical, camera_loss = self.physical_metrics(
            model, selection_pairs, device, arm=arm, stage=stage
        )
        jepa = self.jepa_metrics(
            model,
            selection_pairs,
            vocabulary,
            commanded_table,
            device,
            arm=arm,
            stage=stage,
        )
        scopes = {
            scope: {"physical": physical[scope], "jepa": jepa[scope]}
            for scope in contract.SCOPES
        }
        result = {
            "update": update,
            "scopes": scopes,
            "aggregate_complete_v4_loss": camera_loss,
            "aggregate_prediction_to_persistence_ratio": jepa["aggregate"][
                "prediction_to_warped_persistence_ratio"
            ],
        }
        model.to("cpu")
        del model
        self.r.torch.cuda.empty_cache()
        return result

    def calibration(
        self,
        record: Mapping[str, Any],
        calibration_pairs: Sequence[Mapping[str, Any]],
        device: Any,
    ) -> dict[str, Any]:
        torch, F = self.r.torch, self.r.F
        model = self.load_snapshot(record, device)
        ids_by_family = {
            family: sorted(
                {
                    str(pair[f"{side}_endpoint_sha256"])
                    for pair in calibration_pairs
                    if pair["family"] == family
                    for side in ("current", "next")
                }
            )
            for family in contract.FAMILIES
        }
        rows, columns = model.model_config.bev_size
        forward = torch.linspace(
            model.model_config.forward_range_m[0],
            model.model_config.forward_range_m[1],
            rows + 1,
        )
        left = torch.linspace(
            model.model_config.left_range_m[0],
            model.model_config.left_range_m[1],
            columns + 1,
        )
        grid_forward, grid_left = torch.meshgrid(
            0.5 * (forward[:-1] + forward[1:]),
            0.5 * (left[:-1] + left[1:]),
            indexing="ij",
        )
        within_frame = (grid_forward.square() + grid_left.square()).sqrt().reshape(-1) <= 2.0
        logits_parts, label_parts, within_parts, family_parts = [], [], [], []
        with torch.no_grad():
            for family, ids in ids_by_family.items():
                for endpoint_id in ids:
                    frame = self.inputs.frame(
                        endpoint_id,
                        role="probability_calibration",
                        arm="promoted_jepa",
                        stage="calibration",
                    )
                    online = model.forward_frame(
                        frame["image"][None].to(device),
                        frame["camera_origin"][None].to(device).float(),
                        frame["camera_basis"][None].to(device).float(),
                        frame["ground"][None].to(device).float(),
                    )
                    soft = self.r.soft_rasterize(
                        online.evidence,
                        camera_origin_body_m=frame["camera_origin"][None].to(device).float(),
                        camera_basis_body_fru=frame["camera_basis"][None].to(device).float(),
                        pixel_ray_chunk_size=model.model_config.v4_pixel_ray_chunk_size,
                    )
                    logits_parts.append(
                        soft.class_probabilities.clamp_min(torch.finfo(torch.float32).eps)
                        .log()
                        .permute(0, 2, 3, 1)
                        .reshape(-1, 3)
                        .cpu()
                    )
                    labels = frame["raster_labels"].reshape(-1).long()
                    label_parts.append(labels)
                    within_parts.append(within_frame)
                    family_parts.extend([family] * labels.numel())
        logits = torch.cat(logits_parts).float()
        labels = torch.cat(label_parts).long()
        within = torch.cat(within_parts).bool()
        counts = torch.bincount(labels, minlength=3)
        if bool((counts == 0).any().item()):
            raise DevelopmentGateFailure("calibration role is missing a raster class")
        raw_parameters = torch.zeros(6, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            (raw_parameters,), lr=0.5, max_iter=80, line_search_fn="strong_wolfe"
        )

        def scaled(values: Any, parameters: Any) -> Any:
            scales = parameters[:3].clamp(-3.0, 3.0).exp()
            biases = parameters[3:] - parameters[3:].mean()
            return values * scales[None] + biases[None]

        before = float(F.cross_entropy(logits, labels))

        def closure() -> Any:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(scaled(logits, raw_parameters), labels)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("calibration NLL became nonfinite")
            loss.backward()
            return loss

        optimizer.step(closure)
        parameters = raw_parameters.detach().clone()
        calibrated_logits = scaled(logits, parameters)
        after = float(F.cross_entropy(calibrated_logits, labels))
        if not math.isfinite(after) or after > before + 1e-6:
            raise DevelopmentGateFailure("vector calibration worsened NLL")
        probabilities = calibrated_logits.softmax(dim=1)
        aggregate_reports = self._threshold_grid_reports(
            probabilities, labels, within, device
        )
        threshold = contract.select_calibration_threshold(aggregate_reports)
        family_vector = family_parts
        scope_reports = {}
        for scope in contract.SCOPES:
            if scope == "aggregate":
                mask = torch.ones(labels.numel(), dtype=torch.bool)
            else:
                mask = torch.tensor([item == scope for item in family_vector], dtype=torch.bool)
            report = self._fixed_threshold_report(
                probabilities[mask], labels[mask], within[mask], threshold
            )
            report["uncalibrated_nll"] = float(F.cross_entropy(logits[mask], labels[mask]))
            report["calibrated_nll"] = float(F.cross_entropy(calibrated_logits[mask], labels[mask]))
            report["class_counts"] = torch.bincount(labels[mask], minlength=3).tolist()
            scope_reports[scope] = report
            if (
                any(item <= 0 for item in report["class_counts"])
                or report["admitted_free_precision"] < 0.99
                or report["useful_free_recall"] < 0.90
                or report["obstacle_exclusion_recall_within_2m"] < 0.95
                or report["obstacle_detection_recall_within_2m"] < 0.95
                or report["calibrated_nll"] > report["uncalibrated_nll"] + 1e-6
            ):
                raise DevelopmentGateFailure(f"calibration gate failed: {scope}")
        centered_bias = parameters[3:] - parameters[3:].mean()
        result = {
            "schema": contract.CALIBRATION_SCHEMA,
            "arm": "promoted_jepa",
            "role": "probability_calibration",
            "pair_count": 415,
            "unique_endpoint_count": 759,
            "parameters": {
                "log_scales": parameters[:3].clamp(-3.0, 3.0).tolist(),
                "scales": parameters[:3].clamp(-3.0, 3.0).exp().tolist(),
                "centered_biases": centered_bias.tolist(),
            },
            "uncalibrated_nll": before,
            "calibrated_nll": after,
            "class_counts": counts.tolist(),
            "threshold": threshold,
            "scope_reports": scope_reports,
            "matched_no_jepa_influenced_calibration": False,
        }
        model.to("cpu")
        del model, optimizer
        torch.cuda.empty_cache()
        return result

    def _threshold_grid_reports(
        self,
        probabilities: Any,
        labels: Any,
        within: Any,
        device: Any,
    ) -> dict[str, Any]:
        torch = self.r.torch
        p = probabilities.to(device)
        labels_gpu = labels.to(device)
        within_gpu = within.to(device)
        free = labels_gpu == 1
        obstacles = (labels_gpu == 2) & within_gpu
        triples = [
            (free_min, occupied_max, unknown_max)
            for free_min in contract.CALIBRATION_FREE_MIN_GRID
            for occupied_max in contract.CALIBRATION_OCCUPIED_MAX_GRID
            for unknown_max in contract.CALIBRATION_UNKNOWN_MAX_GRID
        ]
        admission_counts: dict[tuple[float, float, float], tuple[int, int, int, int]] = {}
        for start in range(0, len(triples), 16):
            chunk = triples[start : start + 16]
            free_min = p.new_tensor([item[0] for item in chunk])[None]
            occupied_max = p.new_tensor([item[1] for item in chunk])[None]
            unknown_max = p.new_tensor([item[2] for item in chunk])[None]
            admitted = (
                (p[:, 1:2] >= free_min)
                & (p[:, 2:3] <= occupied_max)
                & (p[:, 0:1] <= unknown_max)
            )
            totals = admitted.sum(dim=0).cpu().tolist()
            true_free = (admitted & free[:, None]).sum(dim=0).cpu().tolist()
            useful = (admitted & free[:, None]).sum(dim=0).cpu().tolist()
            excluded = (obstacles[:, None] & ~admitted).sum(dim=0).cpu().tolist()
            for index, key in enumerate(chunk):
                admission_counts[key] = (
                    int(totals[index]),
                    int(true_free[index]),
                    int(useful[index]),
                    int(excluded[index]),
                )
        detection_counts = {}
        for threshold in contract.CALIBRATION_DETECTION_GRID:
            detection_counts[threshold] = int(
                (obstacles & (p[:, 2] >= threshold)).sum().cpu()
            )
        useful_count = int(free.sum().cpu())
        obstacle_count = int(obstacles.sum().cpu())
        reports = {}
        for values in contract.threshold_grid():
            admitted, true_free, useful_admitted, excluded = admission_counts[values[:3]]
            reports[contract.canonical_json_sha256(list(values))] = {
                "admitted_free_count": admitted,
                "admitted_free_true_free_count": true_free,
                "useful_free_count": useful_count,
                "useful_free_admitted_count": useful_admitted,
                "obstacle_within_2m_count": obstacle_count,
                "obstacle_within_2m_excluded_count": excluded,
                "obstacle_within_2m_detected_count": detection_counts[values[3]],
            }
        return reports

    @staticmethod
    def _fixed_threshold_report(
        probabilities: Any,
        labels: Any,
        within: Any,
        threshold: Mapping[str, Any],
    ) -> dict[str, Any]:
        admitted = (
            (probabilities[:, 1] >= threshold["free_probability_minimum"])
            & (probabilities[:, 2] <= threshold["occupied_probability_maximum"])
            & (probabilities[:, 0] <= threshold["unknown_probability_maximum"])
        )
        free = labels == 1
        obstacles = (labels == 2) & within
        detected = probabilities[:, 2] >= threshold["occupied_detection_minimum"]
        admitted_count = int(admitted.sum())
        useful_count = int(free.sum())
        obstacle_count = int(obstacles.sum())
        if min(admitted_count, useful_count, obstacle_count) <= 0:
            raise DevelopmentGateFailure("calibration scope has an empty denominator")
        true_free = int((admitted & free).sum())
        useful_admitted = int((admitted & free).sum())
        excluded = int((obstacles & ~admitted).sum())
        detected_count = int((obstacles & detected).sum())
        return {
            "admitted_free_count": admitted_count,
            "admitted_free_true_free_count": true_free,
            "useful_free_count": useful_count,
            "useful_free_admitted_count": useful_admitted,
            "obstacle_within_2m_count": obstacle_count,
            "obstacle_within_2m_excluded_count": excluded,
            "obstacle_within_2m_detected_count": detected_count,
            "admitted_free_precision": true_free / admitted_count,
            "useful_free_recall": useful_admitted / useful_count,
            "obstacle_exclusion_recall_within_2m": excluded / obstacle_count,
            "obstacle_detection_recall_within_2m": detected_count / obstacle_count,
        }

    def candidate_bytes(
        self,
        record: Mapping[str, Any],
        selection: Mapping[str, Any],
        calibration: Mapping[str, Any],
        vocabulary: Sequence[str],
        commanded_table: Any,
    ) -> tuple[bytes, dict[str, Any]]:
        model = self.load_snapshot(record, self.r.torch.device("cpu"))
        evaluation_state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in sorted(model.state_dict().items())
        }
        deployment_state = model.deployment_state_dict()
        evaluation_manifest = _tensor_manifest(self.r, evaluation_state)
        deployment_manifest = _tensor_manifest(self.r, deployment_state)
        evaluation_sha = self.r.model_module.tensor_state_dict_sha256(evaluation_state)
        deployment_sha = self.r.model_module.tensor_state_dict_sha256(deployment_state)
        metadata = contract.pre_g2_candidate_metadata(
            model_config=model.model_config.to_dict(),
            evaluation_state_manifest=evaluation_manifest,
            evaluation_state_sha256=evaluation_sha,
            deployment_state_manifest=deployment_manifest,
            deployment_state_sha256=deployment_sha,
            selection=selection,
            calibration=calibration,
            primitive_vocabulary=vocabulary,
            commanded_delta_table=commanded_table.tolist(),
            training_snapshot=record["binding"],
        )
        content_sha = contract.canonical_json_sha256(metadata)
        buffer = io.BytesIO()
        self.r.torch.save(
            {
                **metadata,
                "content_sha256": content_sha,
                "evaluation_state_dict": evaluation_state,
                "deployment_state_dict": deployment_state,
            },
            buffer,
        )
        del model
        return buffer.getvalue(), {**metadata, "content_sha256": content_sha}


def _child_command() -> tuple[str, ...]:
    return (
        sys.executable,
        "-I",
        "-B",
        str(ROOT / contract.RUNNER_RELATIVE_PATH),
        "--internal-verify",
    )


def _invoke_internal_verifier() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["HIP_VISIBLE_DEVICES"] = ""
    for name in contract.CONFLICTING_ACCELERATOR_ENVIRONMENT:
        environment.pop(name, None)
    for name in contract.THREAD_ENVIRONMENT:
        environment[name] = "1"
    completed = subprocess.run(
        _child_command(),
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated checkpoint verifier failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-4000:]
        )
    if completed.stderr:
        raise RuntimeError("isolated checkpoint verifier wrote stderr")
    return contract.parse_canonical_json(
        completed.stdout,
        name="isolated checkpoint verification",
    )


def _terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    error: BaseException,
    *,
    stage: str,
) -> None:
    core = {
        "schema": contract.FAILURE_SCHEMA,
        "status": "failed_numeric_gate" if isinstance(error, DevelopmentGateFailure) else "failed_infrastructure",
        "stage": stage,
        "reservation": contract.artifact_binding(
            "reservation.json",
            reservation_raw,
            content_sha256=str(reservation["content_sha256"]),
        ),
        "error": {"type": type(error).__name__, "message": str(error)},
        "g2_attempted": False,
        "heldout_open_count": 0,
        "retry_authorized": False,
    }
    try:
        _publish_json(output_root / "failed.json", core)
    except FileExistsError:
        pass


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    environment = _validate_parent_environment()
    review, review_raw, authorization, authorization_raw, sources = (
        _load_review_and_authorization(
            review_file_sha256=review_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
        )
    )
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    reservation, reservation_raw = _reserve_output(
        output_root,
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
        environment=environment,
    )
    stage = "post_reservation_runtime_import"
    try:
        runtime = _load_runtime()
        stage = "camera_checkpoint_validation"
        fit, camera_gate, camera_checkpoint_binding = _camera_model_after_reservation(
            runtime, authorization
        )
        stage = "raw_v13_validation"
        inputs = RawInputs(runtime, authorization)
        trainer = Trainer(runtime, inputs, output_root, reservation)
        device, device_record = trainer.device()
        stage = "initialization"
        initial_state, initialization = trainer.initialize(fit)
        del fit
        train_pairs = inputs.role_pairs("train")
        selection_pairs = inputs.role_pairs("checkpoint_selection")
        calibration_pairs = inputs.role_pairs("probability_calibration")
        schedule, schedule_record = trainer.schedule(train_pairs)
        vocabulary, commanded_cpu = trainer.commanded_table(train_pairs)
        commanded = commanded_cpu.to(device)
        initialization = {
            **initialization,
            "primitive_vocabulary": vocabulary,
            "commanded_delta_table": commanded_cpu.tolist(),
            "commanded_delta_table_sha256": contract.canonical_json_sha256(
                commanded_cpu.tolist()
            ),
        }
        initialization_value, initialization_raw = _publish_json(
            output_root / "initialization.json", initialization
        )
        stage = "promoted_training"
        promoted = trainer.train_arm(
            arm="promoted_jepa",
            initial_state=initial_state,
            schedule=schedule,
            schedule_sha=str(schedule_record["content_sha256"]),
            train_pairs=train_pairs,
            vocabulary=vocabulary,
            commanded_table=commanded,
            device=device,
            publish_updates=contract.CHECKPOINT_UPDATES,
        )
        stage = "promoted_selection"
        candidates = [
            trainer.evaluate_snapshot(
                promoted[update],
                update=update,
                selection_pairs=selection_pairs,
                vocabulary=vocabulary,
                commanded_table=commanded,
                device=device,
                arm="promoted_jepa",
                stage="selection",
            )
            for update in contract.CHECKPOINT_UPDATES
        ]
        selection_metrics, selection_metrics_raw = _publish_json(
            output_root / "promoted_checkpoint_metrics.json",
            {
                "schema": f"{contract.SCHEMA_PREFIX}_checkpoint_metrics_v1",
                "role": "checkpoint_selection",
                "pair_count": 495,
                "unique_endpoint_count": 924,
                "scopes": list(contract.SCOPES),
                "candidates": candidates,
                "matched_no_jepa_influenced_selection": False,
            },
        )
        try:
            selection_core = contract.select_promoted_checkpoint(candidates)
        except ValueError as error:
            raise DevelopmentGateFailure(str(error)) from error
        selection, selection_raw = _publish_json(
            output_root / "selection.json",
            {
                **selection_core,
                "checkpoint_metrics": _output_binding(
                    output_root,
                    "promoted_checkpoint_metrics.json",
                    selection_metrics,
                    selection_metrics_raw,
                ),
            },
        )
        selected_update = int(selection["selected_update"])
        stage = "matched_diagnostic_training"
        matched = trainer.train_arm(
            arm="matched_no_jepa",
            initial_state=initial_state,
            schedule=schedule,
            schedule_sha=str(schedule_record["content_sha256"]),
            train_pairs=train_pairs,
            vocabulary=vocabulary,
            commanded_table=commanded,
            device=device,
            publish_updates=(selected_update,),
        )
        stage = "matched_selected_update_diagnostic"
        matched_metrics = trainer.evaluate_snapshot(
            matched[selected_update],
            update=selected_update,
            selection_pairs=selection_pairs,
            vocabulary=vocabulary,
            commanded_table=commanded,
            device=device,
            arm="matched_no_jepa",
            stage="diagnostic",
        )
        matched_value, matched_raw = _publish_json(
            output_root / "matched_selected_update_metrics.json",
            {
                "schema": f"{contract.SCHEMA_PREFIX}_matched_diagnostic_v1",
                "selected_promoted_update": selected_update,
                "matched_update": selected_update,
                "metrics": matched_metrics,
                "selection_effect": "none",
                "calibration_effect": "none",
            },
        )
        stage = "promoted_calibration"
        calibration_core = trainer.calibration(
            promoted[selected_update], calibration_pairs, device
        )
        calibration, calibration_raw = _publish_json(
            output_root / "calibration.json", calibration_core
        )
        stage = "pre_g2_candidate_publication"
        candidate_raw, candidate_metadata = trainer.candidate_bytes(
            promoted[selected_update],
            selection,
            calibration,
            vocabulary,
            commanded_cpu,
        )
        candidate_path = output_root / "pre_g2_candidate.pt"
        _write_exclusive(candidate_path, candidate_raw)
        candidate_binding = contract.artifact_binding(
            "pre_g2_candidate.pt",
            candidate_raw,
            content_sha256=str(candidate_metadata["content_sha256"]),
        )
        stage = "consumed_input_rehash"
        _read_bound(
            ROOT / authorization["camera"]["gate"]["path"],
            authorization["camera"]["gate"],
        )
        _read_bound(
            ROOT / authorization["camera"]["checkpoint"]["path"],
            authorization["camera"]["checkpoint"],
        )
        access = inputs.rehash_consumed()
        access["camera_gate_and_checkpoint_rehashed"] = True
        access_value, access_raw = _publish_json(
            output_root / "access_ledger.json",
            {
                "schema": f"{contract.SCHEMA_PREFIX}_access_ledger_v1",
                **access,
                "raw_roles_opened": list(contract.ROLES),
                "g2_open_count": 0,
                "heldout_open_count": 0,
                "navigation_open_count": 0,
                "production_open_count": 0,
                "gpu1_use_count": 0,
            },
        )
        stage = "result_publication"
        artifacts = {
            "reservation": contract.artifact_binding(
                "reservation.json",
                reservation_raw,
                content_sha256=str(reservation["content_sha256"]),
            ),
            "initialization": _output_binding(
                output_root, "initialization.json", initialization_value, initialization_raw
            ),
            "selection": _output_binding(
                output_root, "selection.json", selection, selection_raw
            ),
            "calibration": _output_binding(
                output_root, "calibration.json", calibration, calibration_raw
            ),
            "matched_diagnostic": _output_binding(
                output_root,
                "matched_selected_update_metrics.json",
                matched_value,
                matched_raw,
            ),
            "access_ledger": _output_binding(
                output_root, "access_ledger.json", access_value, access_raw
            ),
            "pre_g2_candidate": candidate_binding,
        }
        result, result_raw = _publish_json(
            output_root / "result.json",
            {
                "schema": contract.RESULT_SCHEMA,
                "status": "pending_isolated_checkpoint_reload",
                "attempt_identity": reservation["attempt_identity"],
                "camera_gate_content_sha256": camera_gate["content_sha256"],
                "camera_checkpoint": camera_checkpoint_binding,
                "raw_manifest_content_sha256": inputs.manifest["content_sha256"],
                "raw_audit_content_sha256": inputs.audit["content_sha256"],
                "selected_update": selected_update,
                "device": device_record,
                "artifacts": artifacts,
                **contract.PRE_G2_DENIALS,
            },
        )
        result_binding = _output_binding(
            output_root, "result.json", result, result_raw
        )
        runtime.torch.cuda.empty_cache()
        stage = "isolated_checkpoint_reload"
        verification = _invoke_internal_verifier()
        if (
            verification.get("schema") != contract.VERIFICATION_SCHEMA
            or verification.get("status") != "PASS"
            or verification.get("result") != result_binding
            or verification.get("candidate") != candidate_binding
            or verification.get("evaluation_state_sha256")
            != candidate_metadata["evaluation_state_sha256"]
            or verification.get("deployment_state_sha256")
            != candidate_metadata["deployment_state_sha256"]
            or verification.get("checkpoint_open_count") != 1
            or verification.get("strict_full_state_reload") is not True
            or verification.get("g2_attempted") is not False
            or verification.get("retry_authorized") is not False
        ):
            raise RuntimeError("isolated checkpoint verification changed")
        verification_raw = contract.canonical_json_bytes(verification) + b"\n"
        _write_exclusive(output_root / "verification.json", verification_raw)
        verification_binding = contract.artifact_binding(
            "verification.json",
            verification_raw,
            content_sha256=str(verification["content_sha256"]),
        )
        stage = "completion_publication"
        completion, completion_raw = _publish_json(
            output_root / "completed.json",
            {
                "schema": contract.COMPLETION_SCHEMA,
                "status": "pre_g2_candidate_verified",
                "attempt_identity": reservation["attempt_identity"],
                "result": result_binding,
                "verification": verification_binding,
                "pre_g2_candidate": candidate_binding,
                "selected_update": selected_update,
                "all_nine_selection_scopes_passed": True,
                "all_nine_calibration_scopes_passed": True,
                "matched_arm_influenced_selection_or_calibration": False,
                **contract.PRE_G2_DENIALS,
            },
        )
        summary = {
            "status": completion["status"],
            "selected_update": selected_update,
            "candidate_file_sha256": candidate_binding["file_sha256"],
            "candidate_content_sha256": candidate_binding["content_sha256"],
            "completion_file_sha256": hashlib.sha256(completion_raw).hexdigest(),
            "completion_content_sha256": completion["content_sha256"],
            "g2_attempted": False,
        }
        print(contract.canonical_json_bytes(summary).decode("ascii"), flush=True)
        return 0
    except BaseException as error:
        _terminal_failure(
            output_root,
            reservation,
            reservation_raw,
            error,
            stage=stage,
        )
        raise


def _read_binding_from_root(
    root: Path,
    binding: Mapping[str, Any],
    *,
    expected_path: str,
) -> tuple[dict[str, Any], bytes]:
    validated = contract.validate_binding(binding, path=expected_path)
    raw = _read_regular(
        root / expected_path,
        expected_sha256=validated["file_sha256"],
    )
    if len(raw) != validated["byte_count"]:
        raise PermissionError(f"bound byte count changed: {expected_path}")
    return validated, raw


def _revalidate_isolated_authority(
    output_root: Path,
    result: Mapping[str, Any],
    *,
    repository_root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    artifacts = result.get("artifacts")
    if type(artifacts) is not dict:
        raise PermissionError("training result artifacts changed")
    reservation_binding, reservation_raw = _read_binding_from_root(
        output_root,
        artifacts.get("reservation"),
        expected_path="reservation.json",
    )
    reservation = contract.parse_canonical_json(
        reservation_raw,
        name="training reservation",
    )
    expected_reservation_fields = {
        "schema",
        "status",
        "attempt_index",
        "maximum_attempts",
        "attempt_identity",
        "independent_review",
        "execution_authorization",
        "reviewed_sources",
        "science_contract",
        "raw",
        "camera",
        "environment",
        "torch_imported_before_reservation",
        "camera_or_raw_opened_before_reservation",
        "retry_authorized",
        "content_sha256",
    }
    recomputed_reservation_binding = contract.artifact_binding(
        "reservation.json",
        reservation_raw,
        content_sha256=str(reservation["content_sha256"]),
    )
    if (
        set(reservation) != expected_reservation_fields
        or recomputed_reservation_binding != reservation_binding
        or reservation.get("schema") != contract.RESERVATION_SCHEMA
        or reservation.get("status") != "reserved_before_torch_camera_raw_or_rgb"
        or reservation.get("attempt_index") != 1
        or reservation.get("maximum_attempts") != 1
        or reservation.get("attempt_identity") != result.get("attempt_identity")
        or reservation.get("science_contract") != contract.science_contract()
        or reservation.get("torch_imported_before_reservation") is not False
        or reservation.get("camera_or_raw_opened_before_reservation") is not False
        or reservation.get("retry_authorized") is not False
    ):
        raise PermissionError("training reservation changed")

    sources = contract.current_source_bindings(repository_root)
    if reservation.get("reviewed_sources") != sources:
        raise PermissionError("reviewed source bindings changed before verification")

    review_binding, review_raw = _read_binding_from_root(
        repository_root,
        reservation.get("independent_review"),
        expected_path=contract.REVIEW_RELATIVE_PATH,
    )
    review = contract.parse_canonical_json(review_raw, name="independent review")
    contract.validate_review(review, expected_sources=sources)
    if contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    ) != review_binding:
        raise PermissionError("independent review binding changed")

    authorization_binding, authorization_raw = _read_binding_from_root(
        repository_root,
        reservation.get("execution_authorization"),
        expected_path=contract.AUTHORIZATION_RELATIVE_PATH,
    )
    authorization = contract.parse_canonical_json(
        authorization_raw,
        name="execution authorization",
    )
    contract.validate_authorization(
        authorization,
        review_binding=review_binding,
    )
    if contract.artifact_binding(
        contract.AUTHORIZATION_RELATIVE_PATH,
        authorization_raw,
        content_sha256=str(authorization["content_sha256"]),
    ) != authorization_binding:
        raise PermissionError("execution authorization binding changed")
    expected_attempt_identity = contract.canonical_json_sha256(
        {
            "schema": f"{contract.SCHEMA_PREFIX}_attempt_identity_v1",
            "review": review_binding,
            "authorization": authorization_binding,
            "science_contract_sha256": contract.canonical_json_sha256(
                contract.science_contract()
            ),
        }
    )
    if (
        reservation["attempt_identity"] != expected_attempt_identity
        or reservation.get("raw") != authorization["raw"]
        or reservation.get("camera") != authorization["camera"]
    ):
        raise PermissionError("reserved execution authority changed")
    return reservation, review, authorization, sources


def run_internal_verifier() -> int:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("internal verifier requires python -I -B")
    if os.environ.get("HIP_VISIBLE_DEVICES") not in {"", None}:
        raise PermissionError("internal verifier must be accelerator-hidden")
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    result_raw = _read_regular(output_root / "result.json")
    result = contract.parse_canonical_json(result_raw, name="training result")
    if (
        result.get("schema") != contract.RESULT_SCHEMA
        or result.get("status") != "pending_isolated_checkpoint_reload"
        or result.get("g2_attempted") is not False
        or result.get("retry_authorized") is not False
    ):
        raise PermissionError("training result is not pre-verification")
    result_binding = contract.artifact_binding(
        "result.json", result_raw, content_sha256=str(result["content_sha256"])
    )
    _revalidate_isolated_authority(output_root, result)
    candidate_binding = contract.validate_binding(
        result["artifacts"]["pre_g2_candidate"], path="pre_g2_candidate.pt"
    )
    candidate_raw = _read_regular(
        output_root / "pre_g2_candidate.pt",
        expected_sha256=candidate_binding["file_sha256"],
    )
    if len(candidate_raw) != candidate_binding["byte_count"]:
        raise PermissionError("pre-G2 checkpoint byte count changed")
    runtime = _load_runtime()
    candidate = runtime.torch.load(
        io.BytesIO(candidate_raw), map_location="cpu", weights_only=True
    )
    tensor_fields = {"evaluation_state_dict", "deployment_state_dict"}
    if type(candidate) is not dict or not tensor_fields <= set(candidate):
        raise PermissionError("pre-G2 checkpoint tensor fields changed")
    metadata = {name: value for name, value in candidate.items() if name not in tensor_fields}
    content_sha = metadata.pop("content_sha256", None)
    if (
        content_sha != candidate_binding["content_sha256"]
        or contract.canonical_json_sha256(metadata) != content_sha
        or metadata.get("schema") != contract.PRE_G2_CHECKPOINT_SCHEMA
        or any(metadata.get(name) != value for name, value in contract.PRE_G2_DENIALS.items())
    ):
        raise PermissionError("pre-G2 checkpoint semantics changed")
    evaluation_state = candidate["evaluation_state_dict"]
    deployment_state = candidate["deployment_state_dict"]
    if (
        metadata["evaluation_state_manifest"] != _tensor_manifest(runtime, evaluation_state)
        or metadata["deployment_state_manifest"] != _tensor_manifest(runtime, deployment_state)
        or metadata["evaluation_state_sha256"]
        != runtime.model_module.tensor_state_dict_sha256(evaluation_state)
        or metadata["deployment_state_sha256"]
        != runtime.model_module.tensor_state_dict_sha256(deployment_state)
    ):
        raise PermissionError("pre-G2 checkpoint state hashes changed")
    model_config = runtime.model_module.SharedObservableCameraRayJepaV5Config.from_mapping(
        metadata["model_config"]
    )
    model = runtime.model_module.SharedObservableCameraRayJepaV5(model_config)
    model.load_state_dict(evaluation_state, strict=True)
    reexported = model.deployment_state_dict()
    if (
        runtime.model_module.tensor_state_dict_sha256(reexported)
        != metadata["deployment_state_sha256"]
        or set(reexported) != set(deployment_state)
        or any(
            not runtime.torch.equal(reexported[name], deployment_state[name])
            for name in reexported
        )
    ):
        raise PermissionError("deployment state is not the full-state model projection")
    required = tuple(metadata["required_evaluation_state_prefixes"])
    if any(not any(name.startswith(prefix) for name in evaluation_state) for prefix in required):
        raise PermissionError("predictor or EMA target state is missing")
    value = contract.with_content_sha256(
        {
            "schema": contract.VERIFICATION_SCHEMA,
            "status": "PASS",
            "result": result_binding,
            "candidate": candidate_binding,
            "evaluation_state_sha256": metadata["evaluation_state_sha256"],
            "deployment_state_sha256": metadata["deployment_state_sha256"],
            "required_evaluation_state_prefixes": list(required),
            "checkpoint_open_count": 1,
            "fresh_model_constructed": True,
            "strict_full_state_reload": True,
            "deployment_projection_recomputed": True,
            "result_metrics_reused_for_state_verification": False,
            "g2_attempted": False,
            "heldout_open_count": 0,
            "retry_authorized": False,
        }
    )
    sys.stdout.buffer.write(contract.canonical_json_bytes(value) + b"\n")
    sys.stdout.buffer.flush()
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--internal-verify", action="store_true")
    parser.add_argument("--review-sha256")
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args(argv)
    if args.run:
        if not contract.is_sha256(args.review_sha256) or not contract.is_sha256(
            args.authorization_sha256
        ):
            parser.error("--run requires both exact review and authorization SHA-256 values")
    elif args.review_sha256 is not None or args.authorization_sha256 is not None:
        parser.error("internal verifier accepts no credentials")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.internal_verify:
        return run_internal_verifier()
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
