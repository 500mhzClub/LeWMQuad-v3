"""Strict RGB-only runtime adapter for memory-role place triplets V1.

The generated index contains privileged cell/yaw values only as a selection
proof.  They are validated while decoding the index and then discarded.  The
only payload returned to a model is the anchor/positive/negative RGB tensor
triplet.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Callable, Mapping
import re


SCHEMA = "lewm_go2_memory_role_place_triplet_index_v1"
MANIFEST_SCHEMA = "lewm_go2_memory_role_place_triplet_index_manifest_v1"
RECEIPT_SCHEMA = "lewm_go2_memory_role_place_triplet_index_build_receipt_v1"
ALLOWED_ROLES = ("train", "checkpoint_selection")
SOURCE_IMAGE_SIZE = (224, 168)
MODEL_IMAGE_SIZE = (112, 112)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_ROW_FIELDS = frozenset(
    {
        "schema",
        "role",
        "family",
        "scene_id",
        "anchor",
        "positive",
        "negative",
        "selection_proof",
        "content_sha256",
    }
)
_RGB_REFERENCE_FIELDS = frozenset(
    {"endpoint_identity_sha256", "rgb_path", "image_sha256"}
)
_PROOF_FIELDS = frozenset(
    {"anchor", "positive", "negative", "positive_separation"}
)
_PROOF_ENDPOINT_FIELDS = frozenset(
    {"cell_id", "yaw_bin", "env_index", "episode_id", "timestamp_ns"}
)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class PlaceTripletContractError(RuntimeError):
    """A place-triplet index, image, or role contract failed."""


@dataclass(frozen=True, slots=True)
class RGBReference:
    endpoint_identity_sha256: str
    rgb_path: str
    image_sha256: str


@dataclass(frozen=True, slots=True)
class PlaceTripletRow:
    """Runtime-safe row; privileged cell/yaw selection labels are absent."""

    index: int
    role: str
    family: str
    scene_id: str
    anchor: RGBReference
    positive: RGBReference
    negative: RGBReference
    content_sha256: str


@dataclass(frozen=True, slots=True)
class RGBTriplet:
    """The complete model input emitted by :func:`load_rgb_triplet`."""

    anchor_rgb: Any
    positive_rgb: Any
    negative_rgb: Any


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PlaceTripletContractError("value is not canonical finite JSON") from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_loads(raw: bytes, *, name: str) -> Any:
    def unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PlaceTripletContractError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PlaceTripletContractError(
                    f"{name} contains non-finite JSON constant {value}"
                )
            ),
        )
    except PlaceTripletContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlaceTripletContractError(f"{name} is invalid UTF-8 JSON") from error


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise PlaceTripletContractError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _exact_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise PlaceTripletContractError(f"{name} must be a nonempty string")
    return str(value)


def _validate_relative_rgb_path(value: object, *, scene_id: str) -> str:
    del scene_id
    text = _exact_text(value, name="rgb_path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) != 6
        or path.parts[:3]
        != (".generated", "go2_render_selected_v04", "scenes")
        or re.fullmatch(r"scene_[0-9a-f]{16}", path.parts[3]) is None
        or path.parts[4] != "rgb"
        or re.fullmatch(r"frame_[0-9]{6}_env_[0-9]{2}\.png", path.parts[5])
        is None
    ):
        raise PlaceTripletContractError("RGB path left the exact source-layout allowlist")
    return text


def _decode_rgb_reference(value: object, *, scene_id: str) -> RGBReference:
    if type(value) is not dict or set(value) != _RGB_REFERENCE_FIELDS:
        raise PlaceTripletContractError("RGB reference fields changed")
    endpoint = value.get("endpoint_identity_sha256")
    image = value.get("image_sha256")
    if not _is_sha256(endpoint) or not _is_sha256(image):
        raise PlaceTripletContractError("RGB reference hash is invalid")
    return RGBReference(
        endpoint_identity_sha256=str(endpoint),
        rgb_path=_validate_relative_rgb_path(value.get("rgb_path"), scene_id=scene_id),
        image_sha256=str(image),
    )


def _decode_proof_endpoint(value: object, *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _PROOF_ENDPOINT_FIELDS:
        raise PlaceTripletContractError(f"{name} selection-proof fields changed")
    episode_id = _exact_text(value.get("episode_id"), name=f"{name}.episode_id")
    return {
        "cell_id": _exact_int(value.get("cell_id"), name=f"{name}.cell_id"),
        "yaw_bin": _exact_int(value.get("yaw_bin"), name=f"{name}.yaw_bin"),
        "env_index": _exact_int(value.get("env_index"), name=f"{name}.env_index"),
        "episode_id": episode_id,
        "timestamp_ns": _exact_int(
            value.get("timestamp_ns"), name=f"{name}.timestamp_ns"
        ),
    }


def _validate_selection_proof(value: object) -> None:
    if type(value) is not dict or set(value) != _PROOF_FIELDS:
        raise PlaceTripletContractError("selection-proof fields changed")
    anchor = _decode_proof_endpoint(value.get("anchor"), name="anchor")
    positive = _decode_proof_endpoint(value.get("positive"), name="positive")
    negative = _decode_proof_endpoint(value.get("negative"), name="negative")
    if (
        positive["cell_id"] != anchor["cell_id"]
        or positive["yaw_bin"] != anchor["yaw_bin"]
        or negative["cell_id"] == anchor["cell_id"]
        or negative["yaw_bin"] != anchor["yaw_bin"]
    ):
        raise PlaceTripletContractError("selection proof does not describe a place triplet")
    different_stream = (
        positive["env_index"], positive["episode_id"]
    ) != (anchor["env_index"], anchor["episode_id"])
    separated_in_time = (
        abs(positive["timestamp_ns"] - anchor["timestamp_ns"]) >= 4_000_000_000
    )
    expected = "different_stream" if different_stream else "timestamp_gap_ge_4s"
    if not (different_stream or separated_in_time) or value.get(
        "positive_separation"
    ) != expected:
        raise PlaceTripletContractError("positive endpoint lacks the required separation")


def decode_index_row(value: object, *, index: int, role: str) -> PlaceTripletRow:
    if role not in ALLOWED_ROLES:
        raise PlaceTripletContractError("only train/checkpoint_selection roles are allowed")
    if type(value) is not dict or set(value) != _ROW_FIELDS:
        raise PlaceTripletContractError("place-triplet row fields changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PlaceTripletContractError("place-triplet row content hash changed")
    if value.get("schema") != SCHEMA or value.get("role") != role:
        raise PlaceTripletContractError("place-triplet row schema or role changed")
    family = _exact_text(value.get("family"), name="family")
    scene_id = _exact_text(value.get("scene_id"), name="scene_id")
    anchor = _decode_rgb_reference(value.get("anchor"), scene_id=scene_id)
    positive = _decode_rgb_reference(value.get("positive"), scene_id=scene_id)
    negative = _decode_rgb_reference(value.get("negative"), scene_id=scene_id)
    if len(
        {
            anchor.endpoint_identity_sha256,
            positive.endpoint_identity_sha256,
            negative.endpoint_identity_sha256,
        }
    ) != 3 or len({anchor.rgb_path, positive.rgb_path, negative.rgb_path}) != 3:
        raise PlaceTripletContractError("triplet endpoint identities or RGB paths overlap")
    _validate_selection_proof(value.get("selection_proof"))
    return PlaceTripletRow(
        index=index,
        role=role,
        family=family,
        scene_id=scene_id,
        anchor=anchor,
        positive=positive,
        negative=negative,
        content_sha256=str(declared),
    )


def decode_index_bytes(raw: bytes, *, role: str) -> tuple[PlaceTripletRow, ...]:
    if type(raw) is not bytes or not raw or not raw.endswith(b"\n"):
        raise PlaceTripletContractError("place-triplet index must be newline-terminated JSONL")
    rows: list[PlaceTripletRow] = []
    for index, line in enumerate(raw.splitlines()):
        if not line:
            raise PlaceTripletContractError("place-triplet index contains a blank row")
        rows.append(
            decode_index_row(
                _strict_json_loads(line, name=f"{role} index row {index}"),
                index=index,
                role=role,
            )
        )
    if len({row.content_sha256 for row in rows}) != len(rows):
        raise PlaceTripletContractError("place-triplet index repeats a row")
    return tuple(rows)


def _read_regular_path(path: Path) -> bytes:
    descriptor = os.open(path, _READ_FLAGS)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PlaceTripletContractError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise PlaceTripletContractError(f"file changed while reading: {path}")
    return b"".join(chunks)


def _decode_manifest(raw: bytes) -> Mapping[str, Any]:
    value = _strict_json_loads(raw, name="place-triplet manifest")
    if type(value) is not dict or value.get("schema") != MANIFEST_SCHEMA:
        raise PlaceTripletContractError("place-triplet manifest schema changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PlaceTripletContractError("place-triplet manifest content hash changed")
    if value.get("status") != "PASS":
        raise PlaceTripletContractError("place-triplet manifest is not passing")
    return value


def load_index(
    repo_root: Path,
    index_root: Path,
    *,
    role: str,
    expected_manifest_sha256: str | None = None,
) -> tuple[tuple[PlaceTripletRow, ...], dict[str, Any]]:
    """Load one manifest-bound role index without opening any RGB leaves."""

    root = Path(repo_root).resolve(strict=True)
    candidate = Path(index_root)
    directory = candidate if candidate.is_absolute() else root / candidate
    manifest_raw = _read_regular_path(directory / "manifest.json")
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise PlaceTripletContractError("place-triplet manifest file hash changed")
    manifest = _decode_manifest(manifest_raw)
    artifacts = manifest.get("artifacts")
    name = f"{role}.jsonl"
    if role not in ALLOWED_ROLES or type(artifacts) is not dict:
        raise PlaceTripletContractError("requested place-triplet role is unavailable")
    binding = artifacts.get(name)
    if type(binding) is not dict or set(binding) != {
        "path",
        "row_count",
        "byte_count",
        "sha256",
    } or binding.get("path") != name:
        raise PlaceTripletContractError("place-triplet artifact binding changed")
    raw = _read_regular_path(directory / name)
    if (
        len(raw) != _exact_int(binding.get("byte_count"), name="artifact byte_count")
        or hashlib.sha256(raw).hexdigest() != binding.get("sha256")
    ):
        raise PlaceTripletContractError("place-triplet artifact bytes changed")
    rows = decode_index_bytes(raw, role=role)
    if len(rows) != _exact_int(binding.get("row_count"), name="artifact row_count"):
        raise PlaceTripletContractError("place-triplet artifact row count changed")
    return rows, {
        "schema": SCHEMA,
        "role": role,
        "row_count": len(rows),
        "manifest_file_sha256": manifest_sha256,
        "index_file_sha256": str(binding["sha256"]),
        "rgb_open_count": 0,
        "privileged_label_fields_emitted_to_model": 0,
    }


def _read_rgb_reference(repo_root: Path, reference: RGBReference) -> bytes:
    root = Path(repo_root).resolve(strict=True)
    relative = PurePosixPath(reference.rgb_path)
    descriptor = os.open(root, _DIR_FLAGS)
    file_descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(relative.parts[-1], _READ_FLAGS, dir_fd=descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PlaceTripletContractError("RGB leaf is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or hashlib.sha256(raw).hexdigest() != reference.image_sha256
    ):
        raise PlaceTripletContractError("RGB leaf identity or SHA-256 changed")
    return raw


def decode_rgb_bytes(raw: bytes) -> Any:
    """Decode one exact source RGB leaf to the shared 112-square model input."""

    if type(raw) is not bytes or not raw:
        raise PlaceTripletContractError("RGB payload must be nonempty bytes")
    try:
        import torch
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise PlaceTripletContractError("Pillow and torch are required") from error
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != SOURCE_IMAGE_SIZE:
                raise PlaceTripletContractError("image must be exact 224x168 RGB PNG")
            image.load()
            image = image.resize(MODEL_IMAGE_SIZE, Image.Resampling.BILINEAR)
            pixels = bytearray(image.tobytes())
    except PlaceTripletContractError:
        raise
    except Exception as error:
        raise PlaceTripletContractError("RGB PNG decode failed") from error
    tensor = torch.frombuffer(pixels, dtype=torch.uint8).reshape(112, 112, 3)
    tensor = tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    tensor.sub_(mean).div_(std)
    if tuple(tensor.shape) != (3, 112, 112) or not bool(torch.isfinite(tensor).all()):
        raise PlaceTripletContractError("decoded RGB tensor is invalid")
    return tensor


def load_rgb_triplet(
    repo_root: Path,
    row: PlaceTripletRow,
    *,
    record_reference_access: Callable[[str, str], None] | None = None,
) -> RGBTriplet:
    """Return RGB tensors only; no cell/yaw/pose value crosses this boundary."""

    if not isinstance(row, PlaceTripletRow):
        raise TypeError("row must be a PlaceTripletRow")

    def load_reference(role: str, reference: RGBReference) -> Any:
        if record_reference_access is not None:
            record_reference_access(role, "attempt")
        try:
            raw = _read_rgb_reference(repo_root, reference)
            if record_reference_access is not None:
                record_reference_access(role, "sha256_verified")
            tensor = decode_rgb_bytes(raw)
        except BaseException:
            if record_reference_access is not None:
                record_reference_access(role, "failure")
            raise
        if record_reference_access is not None:
            record_reference_access(role, "success")
        return tensor

    return RGBTriplet(
        anchor_rgb=load_reference("anchor", row.anchor),
        positive_rgb=load_reference("positive", row.positive),
        negative_rgb=load_reference("negative", row.negative),
    )


__all__ = [
    "ALLOWED_ROLES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MANIFEST_SCHEMA",
    "MODEL_IMAGE_SIZE",
    "PlaceTripletContractError",
    "PlaceTripletRow",
    "RECEIPT_SCHEMA",
    "RGBReference",
    "RGBTriplet",
    "SCHEMA",
    "SOURCE_IMAGE_SIZE",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "decode_index_bytes",
    "decode_index_row",
    "decode_rgb_bytes",
    "load_index",
    "load_rgb_triplet",
]
