"""Development-only zero-inflation physical evidence adapter for G3.

The adapter has no privileged planning shortcut. It canonicalizes exact
physical labels, verifies their content address and zero-inflation semantics,
constructs the shared transaction type, and commits through
``RevisionedPhysicalMemory.apply_transaction``.
"""
from __future__ import annotations

import hashlib
import json
import math
import operator
from typing import Mapping, Sequence

import numpy as np

from lewm.planning.revisioned_physical_configuration_memory import (
    Cell,
    EvidenceAuthority,
    ObservationIdentity,
    PhysicalCellEvidence,
    PhysicalEvidenceTransaction,
    PhysicalLabel,
    PoseProvenance,
    RevisionedPhysicalMemory,
    TransactionReceipt,
    ZERO_INFLATION_EXACT_PHYSICAL_SEMANTICS,
)


EXACT_PHYSICAL_SEMANTICS = ZERO_INFLATION_EXACT_PHYSICAL_SEMANTICS


def _cell(value: Sequence[int]) -> Cell:
    if len(value) != 2:
        raise ValueError("cell must contain two values")
    if any(isinstance(item, bool) or int(item) != item for item in value):
        raise ValueError("cell coordinates must be integers")
    return (int(value[0]), int(value[1]))


def _physical_label(value: PhysicalLabel | int) -> PhysicalLabel:
    if isinstance(value, bool):
        raise ValueError("boolean is not a physical label")
    if not isinstance(value, PhysicalLabel):
        try:
            parsed = operator.index(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported physical label {value!r}") from exc
        value = parsed
    try:
        return value if isinstance(value, PhysicalLabel) else PhysicalLabel(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported physical label {value!r}") from exc


def _normalise_cells(
    labels: Mapping[Cell, PhysicalLabel | int],
) -> tuple[tuple[Cell, PhysicalLabel], ...]:
    if not isinstance(labels, Mapping):
        raise TypeError("labels must be a mapping")
    normalised: dict[Cell, PhysicalLabel] = {}
    for raw_cell, raw_label in labels.items():
        cell = _cell(raw_cell)
        if cell in normalised:
            raise ValueError("duplicate physical cell after normalization")
        normalised[cell] = _physical_label(raw_label)
    if not normalised:
        raise ValueError("exact physical label mapping cannot be empty")
    return tuple(sorted(normalised.items()))


def exact_physical_cells_content_sha256(
    labels: Mapping[Cell, PhysicalLabel | int],
) -> str:
    """Hash every exact cell, including UNKNOWN cells, in canonical order."""

    normalised = _normalise_cells(labels)
    payload = {
        "schema": "lewm_g3_zero_inflation_exact_physical_cells_v1",
        "labels": [
            {"cell": list(cell), "label": int(label)} for cell, label in normalised
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exact_physical_raster_cells(
    labels: np.ndarray,
    *,
    min_cell: Sequence[int] = (0, 0),
) -> dict[Cell, PhysicalLabel]:
    """Convert an ``[x,y]`` physical raster into an explicit cell mapping."""

    array = np.asarray(labels)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("exact physical raster must be a non-empty 2D array")
    if array.dtype.kind not in "iu":
        raise ValueError("exact physical raster must use an integer dtype")
    if not np.isin(array, tuple(int(label) for label in PhysicalLabel)).all():
        raise ValueError("exact physical raster contains an unsupported label")
    minimum = _cell(min_cell)
    result: dict[Cell, PhysicalLabel] = {}
    for x, y in np.ndindex(array.shape):
        result[(minimum[0] + x, minimum[1] + y)] = PhysicalLabel(
            int(array[x, y])
        )
    return result


class ZeroInflationExactPhysicalAdapterV1:
    """Feed exact physical labels through the shared transaction boundary."""

    def __init__(self, memory: RevisionedPhysicalMemory) -> None:
        if not isinstance(memory, RevisionedPhysicalMemory):
            raise TypeError("memory must be a RevisionedPhysicalMemory")
        if memory.config.promoted_runtime:
            raise PermissionError(
                "exact physical adapter is development-only and cannot bind a promoted runtime"
            )
        self._memory = memory

    def build_transaction_from_cells(
        self,
        labels: Mapping[Cell, PhysicalLabel | int],
        *,
        observation: ObservationIdentity,
        pose: PoseProvenance,
        label_inflation_radius_m: float,
        source_semantics: str = EXACT_PHYSICAL_SEMANTICS,
    ) -> PhysicalEvidenceTransaction:
        if not isinstance(observation, ObservationIdentity):
            raise TypeError("observation must be an ObservationIdentity")
        if observation.authority is not EvidenceAuthority.EXACT_PHYSICAL:
            raise ValueError("exact adapter requires EXACT_PHYSICAL authority")
        if not isinstance(pose, PoseProvenance):
            raise TypeError("pose must be a PoseProvenance")
        inflation = float(label_inflation_radius_m)
        if not math.isfinite(inflation) or inflation != 0.0:
            raise ValueError("exact physical adapter requires zero label inflation")
        if source_semantics != EXACT_PHYSICAL_SEMANTICS:
            raise ValueError("exact physical source semantics mismatch")
        normalised = _normalise_cells(labels)
        actual_payload_sha256 = exact_physical_cells_content_sha256(dict(normalised))
        if observation.payload_sha256 != actual_payload_sha256:
            raise ValueError("exact physical payload SHA-256 mismatch")
        evidence = tuple(
            PhysicalCellEvidence(cell=cell, label=label)
            for cell, label in normalised
            if label is not PhysicalLabel.UNKNOWN
        )
        unknown_cells = tuple(
            cell for cell, label in normalised if label is PhysicalLabel.UNKNOWN
        )
        return self._memory._build_exact_physical_transaction(
            observation=observation,
            pose=pose,
            physical_evidence=evidence,
            observed_unknown_cells=unknown_cells,
            source_semantics=source_semantics,
            label_inflation_radius_m=inflation,
        )

    def fuse_cells(
        self,
        labels: Mapping[Cell, PhysicalLabel | int],
        *,
        observation: ObservationIdentity,
        pose: PoseProvenance,
        label_inflation_radius_m: float,
        source_semantics: str = EXACT_PHYSICAL_SEMANTICS,
    ) -> TransactionReceipt:
        transaction = self.build_transaction_from_cells(
            labels,
            observation=observation,
            pose=pose,
            label_inflation_radius_m=label_inflation_radius_m,
            source_semantics=source_semantics,
        )
        return self._memory.apply_transaction(transaction)

    def build_transaction_from_raster(
        self,
        labels: np.ndarray,
        *,
        min_cell: Sequence[int],
        observation: ObservationIdentity,
        pose: PoseProvenance,
        label_inflation_radius_m: float,
        source_semantics: str = EXACT_PHYSICAL_SEMANTICS,
    ) -> PhysicalEvidenceTransaction:
        cells = exact_physical_raster_cells(labels, min_cell=min_cell)
        return self.build_transaction_from_cells(
            cells,
            observation=observation,
            pose=pose,
            label_inflation_radius_m=label_inflation_radius_m,
            source_semantics=source_semantics,
        )

    def fuse_raster(
        self,
        labels: np.ndarray,
        *,
        min_cell: Sequence[int],
        observation: ObservationIdentity,
        pose: PoseProvenance,
        label_inflation_radius_m: float,
        source_semantics: str = EXACT_PHYSICAL_SEMANTICS,
    ) -> TransactionReceipt:
        transaction = self.build_transaction_from_raster(
            labels,
            min_cell=min_cell,
            observation=observation,
            pose=pose,
            label_inflation_radius_m=label_inflation_radius_m,
            source_semantics=source_semantics,
        )
        return self._memory.apply_transaction(transaction)


__all__ = [
    "EXACT_PHYSICAL_SEMANTICS",
    "ZeroInflationExactPhysicalAdapterV1",
    "exact_physical_cells_content_sha256",
    "exact_physical_raster_cells",
]
