"""Canonical semantic identity for graph-edge handoff snapshots.

Protocol-4 pickle is retained as a transport artifact, but it is not a state
identity: legacy PyTorch tensor serialization embeds allocation-specific
storage identifiers.  This module instead serializes the supported logical
snapshot graph into a domain-separated, length-framed binary language.

Object and shared-storage identity are represented by deterministic IDs from
one canonical first traversal.  No Python identity, address, data pointer, or
PyTorch storage key is emitted or hashed.  Importing this module does not open
an artifact, import torch, or construct a simulator.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import fields as dataclass_fields, is_dataclass
import hashlib
import math
import struct
from typing import Any, Iterable, Mapping, Sequence


SEMANTIC_SERIALIZER_SCHEMA = (
    "physical_graph_edge_handoff_snapshot_semantics_v1.canonical_binary.v1"
)
SEMANTIC_DIGEST_NAME = "snapshot_semantic_digest_v1"
SEMANTIC_BINARY_MAGIC = b"PGEHQ-SNAPSHOT-SEMANTICS\x00\x01"
REFERENCE_POLICY = "CANONICAL_FIRST_TRAVERSAL_REFERENCE_AND_STORAGE_GRAPH_V1"
TORCH_DEVICE_RULE = "CPU_OR_ACCELERATOR_CLASS_WITH_PATH_ASSOCIATION_V1"

STRUCTURED_TYPE_ALLOWLIST = (
    "scripts.run_go2_oracle_branch_pilot_v1.BranchSnapshot",
    "lewm_genesis.lewm_contract.EpisodeState",
)
STRUCTURED_FIELD_AUTHORITY: dict[str, tuple[str, ...]] = {
    "scripts.run_go2_oracle_branch_pilot_v1.BranchSnapshot": (
        "solver_state",
        "step_index",
        "last_actions",
        "harness",
        "rng",
        "counters",
        "goal",
        "identity",
        "boundary",
        "digest",
    ),
    "lewm_genesis.lewm_contract.EpisodeState": (
        "scene_id",
        "episode_id",
        "reset_count",
        "episode_step",
        "scene_family",
        "split",
        "manifest_sha256",
    ),
}

_NONFINITE_ARRAY_AUTHORITIES: dict[str, dict[str, Any]] = {
    "$root/solver_state/Scene._sim._coupler.rigid_solver.dofs_info.force_range": {
        "dtype_str": "<f4",
        "shape": [18, 2],
        "strides_bytes": [8, 4],
        "positive_infinity_count": 6,
        "negative_infinity_count": 6,
        "nan_count": 0,
    },
    "$root/solver_state/Scene._sim._coupler.rigid_solver.dofs_info.limit": {
        "dtype_str": "<f4",
        "shape": [18, 2],
        "strides_bytes": [8, 4],
        "positive_infinity_count": 6,
        "negative_infinity_count": 6,
        "nan_count": 0,
    },
}


class SemanticSnapshotError(ValueError):
    """A value is outside the frozen semantic snapshot language."""


def _fq_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _u64(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
        raise SemanticSnapshotError("value is outside canonical uint64")
    return value.to_bytes(8, "big")


def _frame(tag: bytes, payload: bytes) -> bytes:
    if not isinstance(tag, bytes) or not tag or len(tag) > 255:
        raise SemanticSnapshotError("invalid canonical tag")
    if not isinstance(payload, bytes):
        raise SemanticSnapshotError("canonical payload must be bytes")
    return bytes([len(tag)]) + tag + _u64(len(payload)) + payload


def _utf8(value: str) -> bytes:
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise SemanticSnapshotError("string is not strict UTF-8") from exc


def _integer_payload(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticSnapshotError("integer payload received a non-integer")
    if value == 0:
        return b"\x00"
    magnitude = abs(value).to_bytes((abs(value).bit_length() + 7) // 8, "big")
    return (b"\x01" if value > 0 else b"\x02") + magnitude


def _integer_vector(values: Iterable[int]) -> bytes:
    rows = tuple(int(value) for value in values)
    return _u64(len(rows)) + b"".join(
        _frame(b"int", _integer_payload(value)) for value in rows
    )


def _shape_payload(values: Iterable[int]) -> bytes:
    shape = tuple(int(value) for value in values)
    if any(value < 0 for value in shape):
        raise SemanticSnapshotError("negative shape dimension")
    return _u64(len(shape)) + b"".join(_u64(value) for value in shape)


def _path_component(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _is_torch_tensor(value: Any) -> bool:
    if type(value).__module__.split(".", 1)[0] != "torch":
        return False
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - physical runtime has torch
        raise SemanticSnapshotError("torch value encountered without torch") from exc
    return isinstance(value, torch.Tensor)


def _structured_fields(value: Any) -> tuple[tuple[str, Any], ...]:
    type_name = _fq_type(value)
    if type_name not in STRUCTURED_TYPE_ALLOWLIST:
        raise SemanticSnapshotError(f"unknown structured type: {type_name}")
    if is_dataclass(value) and not isinstance(value, type):
        names = [field.name for field in dataclass_fields(value)]
    else:
        slots = getattr(type(value), "__slots__", None)
        if isinstance(slots, str):
            names = [slots]
        elif isinstance(slots, (list, tuple)) and all(
            isinstance(name, str) for name in slots
        ):
            names = list(slots)
        else:
            raise SemanticSnapshotError(f"structured field authority absent: {type_name}")
    if (
        tuple(names) != STRUCTURED_FIELD_AUTHORITY[type_name]
        or len(names) != len(set(names))
        or any(not hasattr(value, name) for name in names)
    ):
        raise SemanticSnapshotError(f"structured field inventory drift: {type_name}")
    # Declared order, rather than alphabetical order, is semantic authority.
    return tuple((name, getattr(value, name)) for name in names)


def _encode_numpy_scalar(value: Any) -> bytes:
    import numpy as np

    scalar = np.asarray(value)
    if scalar.shape != () or scalar.dtype.hasobject:
        raise SemanticSnapshotError("invalid NumPy scalar")
    if scalar.dtype.fields is not None or scalar.dtype.subdtype is not None:
        raise SemanticSnapshotError("structured NumPy scalar is unsupported")
    if scalar.dtype.kind in "fc" and not bool(np.isfinite(scalar).all()):
        raise SemanticSnapshotError("nonfinite NumPy scalar")
    return _frame(
        b"numpy-scalar",
        _frame(b"dtype", _utf8(scalar.dtype.str))
        + _frame(b"data", np.ascontiguousarray(scalar).tobytes(order="C")),
    )


def _encode_scalar(value: Any, *, mapping_key: bool = False) -> bytes:
    import numpy as np

    if value is None:
        return _frame(b"none", b"")
    if type(value) is bool:
        return _frame(b"bool", b"\x01" if value else b"\x00")
    if type(value) is int:
        return _frame(b"int", _integer_payload(value))
    if type(value) is float:
        if not math.isfinite(value):
            raise SemanticSnapshotError("nonfinite Python float")
        return _frame(b"float64", struct.pack(">d", value))
    if type(value) is str:
        return _frame(b"utf8", _utf8(value))
    if type(value) is bytes:
        return _frame(b"bytes", value)
    if isinstance(value, np.generic) and not mapping_key:
        return _encode_numpy_scalar(value)
    role = "mapping key" if mapping_key else "scalar"
    raise SemanticSnapshotError(f"unsupported typed {role}: {_fq_type(value)}")


def _set_member_encoding(value: Any) -> bytes:
    # The production sets are empty.  Scalar-only support keeps isolated set
    # ordering independent of object/reference traversal.
    try:
        return _encode_scalar(value, mapping_key=True)
    except SemanticSnapshotError as exc:
        raise SemanticSnapshotError(
            f"set member is not a supported typed scalar: {_fq_type(value)}"
        ) from exc


def _referenceable(value: Any) -> bool:
    import numpy as np

    return bool(
        type(value) in (dict, OrderedDict, list, tuple, set, frozenset)
        or isinstance(value, np.ndarray)
        or _is_torch_tensor(value)
        or _fq_type(value) in STRUCTURED_TYPE_ALLOWLIST
    )


def _canonical_children(value: Any, path: str) -> tuple[tuple[str, Any], ...]:
    import numpy as np

    if type(value) in (dict, OrderedDict):
        rows = [(_encode_scalar(key, mapping_key=True), key, item) for key, item in value.items()]
        rows.sort(key=lambda row: row[0])
        encoded = [row[0] for row in rows]
        if len(encoded) != len(set(encoded)):
            raise SemanticSnapshotError("mapping has duplicate canonical keys")
        result: list[tuple[str, Any]] = []
        for _encoded, key, item in rows:
            if isinstance(key, str):
                child_path = f"{path}/{_path_component(key)}"
            else:
                child_path = f"{path}/@{hashlib.sha256(_encoded).hexdigest()}"
            # Keys are scalar and are encoded inline, not reference nodes.
            result.append((child_path, item))
        return tuple(result)
    if type(value) in (list, tuple):
        return tuple((f"{path}/{index}", item) for index, item in enumerate(value))
    if type(value) in (set, frozenset):
        rows = sorted((_set_member_encoding(item), item) for item in value)
        encoded = [row[0] for row in rows]
        if len(encoded) != len(set(encoded)):
            raise SemanticSnapshotError("set has duplicate canonical members")
        # Values are scalar-only, so no reference child is traversed.
        return ()
    if isinstance(value, np.ndarray) or _is_torch_tensor(value):
        return ()
    if _fq_type(value) in STRUCTURED_TYPE_ALLOWLIST:
        return tuple(
            (f"{path}/{_path_component(name)}", item)
            for name, item in _structured_fields(value)
        )
    return ()


class _GraphPlan:
    """First-traversal object IDs and shared-storage groups."""

    def __init__(self, root: Any) -> None:
        import numpy as np

        self.object_ids: dict[int, int] = {}
        self.object_paths: dict[int, str] = {}
        self.object_types: dict[int, str] = {}
        self.reference_edges: list[dict[str, Any]] = []
        self._active: set[int] = set()
        self.types: Counter[str] = Counter()
        self.structured: dict[str, tuple[str, ...]] = {}
        self.arrays: list[Any] = []
        self.tensors: list[Any] = []
        self.tensor_paths: dict[int, str] = {}
        self.nonfinite_sentinel_inventory: list[dict[str, Any]] = []
        self._walk(root, "$root")

        values = [*self.arrays, *self.tensors]
        parent = list(range(len(values)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)

        numpy_owners = [_numpy_storage_owner(item)[0] for item in self.arrays]
        for left_index, left in enumerate(self.arrays):
            for right_index, right in enumerate(self.arrays[:left_index]):
                try:
                    if numpy_owners[left_index] is numpy_owners[right_index] or bool(
                        np.shares_memory(left, right)
                    ):
                        union(left_index, right_index)
                except Exception as exc:
                    raise SemanticSnapshotError("cannot inspect NumPy storage alias") from exc

        if self.tensors:
            import torch

            alias_predicate = getattr(torch._C, "_is_alias_of", None)
            if not callable(alias_predicate):
                raise SemanticSnapshotError("torch alias predicate is unavailable")
            offset = len(self.arrays)
            for left_index, left in enumerate(self.tensors):
                for right_index, right in enumerate(self.tensors[:left_index]):
                    try:
                        if bool(alias_predicate(left, right)):
                            union(offset + left_index, offset + right_index)
                    except Exception as exc:
                        raise SemanticSnapshotError("cannot inspect torch storage alias") from exc

        grouped: dict[int, list[Any]] = {}
        for index, item in enumerate(values):
            grouped.setdefault(find(index), []).append(item)
        ordered = sorted(
            grouped.values(),
            key=lambda group: min(self.object_ids[id(item)] for item in group),
        )
        self.storage_ids: dict[int, int] = {}
        self.storage_manifests: list[dict[str, Any]] = []
        self.numpy_origins: dict[int, int] = {}
        for storage_id, group in enumerate(ordered):
            object_ids = sorted(self.object_ids[id(item)] for item in group)
            paths = [self.object_paths[object_id] for object_id in object_ids]
            kinds = {"torch" if _is_torch_tensor(item) else "numpy" for item in group}
            if len(kinds) != 1:
                raise SemanticSnapshotError("NumPy and torch cannot share one storage group")
            kind = next(iter(kinds))
            for item in group:
                self.storage_ids[id(item)] = storage_id
            if kind == "numpy":
                # Transient addresses locate view offsets only; they are never
                # emitted, counted, or hashed.  The stable offset is relative to
                # the lowest byte bound of the shared group.
                try:
                    for item in group:
                        _owner, origin = _numpy_storage_owner(item)
                        self.numpy_origins[id(item)] = origin
                except Exception as exc:
                    raise SemanticSnapshotError("cannot derive NumPy view offsets") from exc
            self.storage_manifests.append(
                {
                    "storage_id": storage_id,
                    "kind": kind,
                    "owner_object_id": object_ids[0],
                    "member_object_ids": object_ids,
                    "member_paths": paths,
                }
            )

    def _walk(self, value: Any, path: str) -> None:
        import numpy as np

        if not _referenceable(value):
            if not (
                value is None
                or type(value) in (bool, int, float, str, bytes)
                or isinstance(value, np.generic)
            ):
                raise SemanticSnapshotError(
                    f"unknown semantic type: {_fq_type(value)}"
                )
            _encode_scalar(value)
            self.types[_fq_type(value)] += 1
            return
        identity = id(value)
        if identity in self.object_ids:
            self.reference_edges.append(
                {
                    "path": path,
                    "reference_id": self.object_ids[identity],
                    "cycle": identity in self._active,
                }
            )
            return
        object_id = len(self.object_ids)
        self.object_ids[identity] = object_id
        self.object_paths[object_id] = path
        self.object_types[object_id] = _fq_type(value)
        self.types[_fq_type(value)] += 1
        self._active.add(identity)
        if isinstance(value, np.ndarray):
            self.arrays.append(value)
        elif _is_torch_tensor(value):
            self.tensors.append(value)
            self.tensor_paths[identity] = path
        elif _fq_type(value) in STRUCTURED_TYPE_ALLOWLIST:
            fields = tuple(name for name, _item in _structured_fields(value))
            prior = self.structured.setdefault(_fq_type(value), fields)
            if prior != fields:
                raise SemanticSnapshotError("structured field order changed within graph")
        if type(value) in (dict, OrderedDict):
            # Scalar mapping keys are inline semantic nodes even though they do
            # not participate in the object/reference graph.
            for key in value:
                _encode_scalar(key, mapping_key=True)
                self.types[_fq_type(key)] += 1
        for child_path, child in _canonical_children(value, path):
            self._walk(child, child_path)
        self._active.remove(identity)

    def evidence(self) -> dict[str, Any]:
        tensor_devices = []
        for tensor in self.tensors:
            device = tensor.device
            tensor_devices.append(
                {
                    "path": self.tensor_paths[id(tensor)],
                    "object_id": self.object_ids[id(tensor)],
                    "semantic_device_class": (
                        "cpu" if str(device.type) == "cpu" else "accelerator"
                    ),
                    "source_device_type": str(device.type),
                    "source_device_index": device.index,
                }
            )
        return {
            "reference_policy": REFERENCE_POLICY,
            "referenceable_object_count": len(self.object_ids),
            "reference_alias_edge_count": len(self.reference_edges),
            "reference_cycle_edge_count": sum(row["cycle"] for row in self.reference_edges),
            "reference_manifest": [
                {
                    "object_id": object_id,
                    "path": self.object_paths[object_id],
                    "type": self.object_types[object_id],
                }
                for object_id in range(len(self.object_paths))
            ],
            "reference_edge_manifest": list(self.reference_edges),
            "storage_manifest": list(self.storage_manifests),
            "structured_type_inventory": [
                {"type": name, "declared_fields": list(fields)}
                for name, fields in sorted(self.structured.items())
            ],
            "tensor_device_manifest": tensor_devices,
            "nonfinite_sentinel_inventory": sorted(
                self.nonfinite_sentinel_inventory, key=lambda row: row["path"]
            ),
            "type_inventory": [
                {"type": name, "count": int(count)}
                for name, count in sorted(self.types.items())
            ],
        }


def _numpy_storage_owner(array: Any) -> tuple[Any, int]:
    """Return the true backing owner and its transient lowest byte address."""

    import numpy as np

    byte_bounds = getattr(np, "byte_bounds", None)
    if not callable(byte_bounds):
        byte_bounds = np.lib.array_utils.byte_bounds
    current = array
    seen: set[int] = set()
    while isinstance(getattr(current, "base", None), np.ndarray):
        if id(current) in seen:
            raise SemanticSnapshotError("cycle in NumPy base chain")
        seen.add(id(current))
        current = current.base
    owner = getattr(current, "base", None)
    if owner is None:
        owner = current
        origin = int(byte_bounds(current)[0])
    elif isinstance(owner, np.ndarray):  # defensive; loop normally consumes it
        origin = int(byte_bounds(owner)[0])
    else:
        try:
            buffer = np.frombuffer(owner, dtype=np.uint8)
            origin = int(byte_bounds(buffer)[0])
        except Exception as exc:
            raise SemanticSnapshotError("unsupported NumPy backing storage") from exc
    return owner, origin


def _validate_array_nonfinite(path: str, array: Any) -> dict[str, Any] | None:
    import numpy as np

    if array.dtype.kind not in "fc":
        return None
    finite = np.isfinite(array)
    if bool(finite.all()):
        return None
    authority = _NONFINITE_ARRAY_AUTHORITIES.get(path)
    if authority is None:
        raise SemanticSnapshotError(f"nonfinite array outside authority: {path}")
    observed = {
        "dtype_str": array.dtype.str,
        "shape": [int(value) for value in array.shape],
        "strides_bytes": [int(value) for value in array.strides],
        "positive_infinity_count": int(np.isposinf(array).sum()),
        "negative_infinity_count": int(np.isneginf(array).sum()),
        "nan_count": int(np.isnan(array).sum()),
    }
    if observed != authority:
        raise SemanticSnapshotError(f"nonfinite array authority drift: {path}")
    return {"path": path, **observed}


class _Encoder:
    def __init__(self, plan: _GraphPlan) -> None:
        self.plan = plan
        self.emitted: set[int] = set()

    def encode(self, value: Any, path: str = "$root") -> bytes:
        if not _referenceable(value):
            return _encode_scalar(value)
        object_id = self.plan.object_ids[id(value)]
        if object_id in self.emitted:
            return _frame(b"REF", _u64(object_id))
        self.emitted.add(object_id)
        body = self._body(value, path)
        return _frame(b"DEF", _u64(object_id) + body)

    def _body(self, value: Any, path: str) -> bytes:
        import numpy as np

        if type(value) is list:
            return _frame(
                b"list",
                _u64(len(value))
                + b"".join(self.encode(item, f"{path}/{index}") for index, item in enumerate(value)),
            )
        if type(value) is tuple:
            return _frame(
                b"tuple",
                _u64(len(value))
                + b"".join(self.encode(item, f"{path}/{index}") for index, item in enumerate(value)),
            )
        if type(value) in (set, frozenset):
            rows = sorted(_set_member_encoding(item) for item in value)
            if len(rows) != len(set(rows)):
                raise SemanticSnapshotError("set has duplicate canonical members")
            return _frame(
                b"set" if type(value) is set else b"frozenset",
                _u64(len(rows)) + b"".join(rows),
            )
        if type(value) in (dict, OrderedDict):
            rows = [(_encode_scalar(key, mapping_key=True), key, item) for key, item in value.items()]
            rows.sort(key=lambda row: row[0])
            encoded_keys = [row[0] for row in rows]
            if len(encoded_keys) != len(set(encoded_keys)):
                raise SemanticSnapshotError("mapping has duplicate canonical keys")
            encoded_rows = []
            for key_bytes, key, item in rows:
                child_path = (
                    f"{path}/{_path_component(key)}"
                    if isinstance(key, str)
                    else f"{path}/@{hashlib.sha256(key_bytes).hexdigest()}"
                )
                encoded_rows.append(_frame(b"entry", key_bytes + self.encode(item, child_path)))
            return _frame(
                b"mapping",
                _frame(b"mapping-type", _utf8(_fq_type(value)))
                + _u64(len(rows))
                + b"".join(encoded_rows),
            )
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields is not None or value.dtype.subdtype is not None:
                raise SemanticSnapshotError("unsupported NumPy dtype")
            sentinel = _validate_array_nonfinite(path, value)
            if sentinel is not None:
                self.plan.nonfinite_sentinel_inventory.append(sentinel)
            contiguous = np.ascontiguousarray(value)
            try:
                data_pointer = int(value.__array_interface__["data"][0])
                offset_bytes = data_pointer - self.plan.numpy_origins[id(value)]
            except Exception as exc:
                raise SemanticSnapshotError("cannot derive NumPy storage offset") from exc
            if offset_bytes < 0:
                raise SemanticSnapshotError("negative NumPy storage offset")
            return _frame(
                b"numpy",
                _frame(b"storage-id", _u64(self.plan.storage_ids[id(value)]))
                + _frame(b"storage-offset-bytes", _u64(offset_bytes))
                + _frame(b"dtype", _utf8(value.dtype.str))
                + _frame(b"shape", _shape_payload(value.shape))
                + _frame(b"stride-bytes", _integer_vector(value.strides))
                + _frame(b"logical-c-bytes", contiguous.tobytes(order="C")),
            )
        if _is_torch_tensor(value):
            import torch

            tensor = value
            if tensor.layout != torch.strided or bool(tensor.is_sparse) or bool(tensor.is_quantized):
                raise SemanticSnapshotError("unsupported torch tensor layout")
            if tensor.device.type == "meta" or bool(tensor.requires_grad):
                raise SemanticSnapshotError("unsupported torch tensor state")
            if bool(tensor.is_conj()) or bool(tensor.is_neg()):
                raise SemanticSnapshotError("lazy torch view is unsupported")
            if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
                if not bool(torch.isfinite(tensor).all().item()):
                    raise SemanticSnapshotError(f"nonfinite torch tensor outside authority: {path}")
            device_class = "cpu" if str(tensor.device.type) == "cpu" else "accelerator"
            contiguous = tensor.detach().cpu().contiguous()
            try:
                data = contiguous.view(torch.uint8).numpy().tobytes(order="C")
            except Exception as exc:
                raise SemanticSnapshotError("cannot materialize torch logical bytes") from exc
            return _frame(
                b"torch",
                _frame(b"storage-id", _u64(self.plan.storage_ids[id(value)]))
                + _frame(
                    b"storage-offset-bytes",
                    _u64(int(tensor.storage_offset()) * int(tensor.element_size())),
                )
                + _frame(b"dtype", _utf8(str(tensor.dtype)))
                + _frame(b"shape", _shape_payload(tensor.shape))
                + _frame(b"logical-stride", _integer_vector(tensor.stride()))
                + _frame(b"device-rule", _utf8(TORCH_DEVICE_RULE))
                + _frame(b"device-class", _utf8(device_class))
                + _frame(b"logical-cpu-c-bytes", data),
            )
        type_name = _fq_type(value)
        if type_name in STRUCTURED_TYPE_ALLOWLIST:
            rows = _structured_fields(value)
            encoded = []
            for name, item in rows:
                encoded.append(
                    _frame(
                        b"field",
                        _frame(b"name", _utf8(name))
                        + self.encode(item, f"{path}/{_path_component(name)}"),
                    )
                )
            return _frame(
                b"structured",
                _frame(b"type", _utf8(type_name))
                + _u64(len(rows))
                + b"".join(encoded),
            )
        raise SemanticSnapshotError(f"unknown semantic type: {type_name}")


def _serialize(value: Any) -> tuple[bytes, _GraphPlan]:
    plan = _GraphPlan(value)
    payload = SEMANTIC_BINARY_MAGIC + _Encoder(plan).encode(value)
    return payload, plan


def canonical_semantic_snapshot(value: Any) -> bytes:
    """Return the exact canonical semantic byte stream."""

    return _serialize(value)[0]


def semantic_snapshot_sha256(value: Any) -> str:
    """Return ``snapshot_semantic_digest_v1`` for a supported snapshot."""

    return hashlib.sha256(canonical_semantic_snapshot(value)).hexdigest()


def semantic_snapshot_evidence(value: Any) -> dict[str, Any]:
    """Return the digest plus complete type/reference/storage/device manifests."""

    payload, plan = _serialize(value)
    return {
        "serializer_schema": SEMANTIC_SERIALIZER_SCHEMA,
        SEMANTIC_DIGEST_NAME: hashlib.sha256(payload).hexdigest(),
        "canonical_semantic_byte_count": len(payload),
        **plan.evidence(),
    }


__all__ = [
    "REFERENCE_POLICY",
    "SEMANTIC_BINARY_MAGIC",
    "SEMANTIC_DIGEST_NAME",
    "SEMANTIC_SERIALIZER_SCHEMA",
    "STRUCTURED_TYPE_ALLOWLIST",
    "STRUCTURED_FIELD_AUTHORITY",
    "TORCH_DEVICE_RULE",
    "SemanticSnapshotError",
    "canonical_semantic_snapshot",
    "semantic_snapshot_evidence",
    "semantic_snapshot_sha256",
]
