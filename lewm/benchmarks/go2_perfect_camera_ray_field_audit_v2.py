"""Additive V2 audit with a genuinely observable-only camera-ray arm.

V1 intentionally reported two useful reconstruction arms, but both retained
the privileged collision-geometry FREE-to-UNKNOWN veto.  V2 preserves those
arms byte-for-byte and adds ``observable_ray_only``:

* no zero-inflation physical-free prior; and
* no collision-geometry veto.

The new arm is therefore exactly the output raster mechanically determined by
the prescribed perfect camera rays.  V1 is loaded under a neutral module name
so importing this audit before the fit reader's source authorization does not
import the protected ``lewm`` package graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


V1_CORE_PATH = Path(__file__).with_name("go2_perfect_camera_ray_field_audit.py")
FRAME_AUDIT_SCHEMA = "lewm_go2_perfect_camera_ray_field_frame_audit_v2"
FIT_AUDIT_SCHEMA = "lewm_go2_perfect_camera_ray_field_fit_audit_v2"
EXPECTED_FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)


def _load_v1_core() -> Any:
    name = "go2_perfect_camera_ray_field_audit_v1_for_v2"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, V1_CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the V1 perfect-ray audit core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load_v1_core()

UNKNOWN_CLASS = v1.UNKNOWN_CLASS
FREE_CLASS = v1.FREE_CLASS
OCCUPIED_CLASS = v1.OCCUPIED_CLASS
CLASS_NAMES = v1.CLASS_NAMES
CameraRaySpec = v1.CameraRaySpec
OrientedBox = v1.OrientedBox
OutputGridSpec = v1.OutputGridSpec
PerfectCameraRayField = v1.PerfectCameraRayField
PhysicalWindow = v1.PhysicalWindow


def _readonly_labels(value: Any, *, name: str) -> np.ndarray:
    labels = np.array(value, dtype=np.uint8, order="C", copy=True)
    if labels.shape != (64, 64):
        raise ValueError(f"{name} must have shape [64, 64]")
    if not np.isin(labels, (UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS)).all():
        raise ValueError(f"{name} contains an unsupported class")
    labels.setflags(write=False)
    return labels


@dataclass(frozen=True)
class RayFieldRasterizationV2:
    v1_rasterization: Any
    observable_ray_only_labels: np.ndarray

    def __post_init__(self) -> None:
        labels = _readonly_labels(
            self.observable_ray_only_labels, name="observable-ray-only labels"
        )
        expected = np.asarray(self.v1_rasterization.ray_only_pre_veto_labels)
        if expected.shape != (64, 64) or not np.array_equal(labels, expected):
            raise ValueError(
                "observable-ray-only must equal the no-physical-prior pre-veto raster"
            )
        object.__setattr__(self, "observable_ray_only_labels", labels)

    @property
    def contract_labels(self) -> np.ndarray:
        return self.v1_rasterization.contract_labels

    @property
    def collision_vetoed_ray_only_labels(self) -> np.ndarray:
        return self.v1_rasterization.ray_only_labels

    @property
    def collision_overlap(self) -> np.ndarray:
        return self.v1_rasterization.collision_overlap

    @property
    def field_sha256(self) -> str:
        return str(self.v1_rasterization.field_sha256)


def reconstruct_frame_from_perfect_rays(**kwargs: Any) -> RayFieldRasterizationV2:
    base = v1.reconstruct_frame_from_perfect_rays(**kwargs)
    return RayFieldRasterizationV2(
        v1_rasterization=base,
        observable_ray_only_labels=base.ray_only_pre_veto_labels,
    )


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        name: int(np.count_nonzero(labels == class_index))
        for class_index, name in enumerate(CLASS_NAMES)
    }


def _confusion(reference: np.ndarray, prediction: np.ndarray) -> list[list[int]]:
    return [
        [
            int(np.count_nonzero((reference == expected) & (prediction == predicted)))
            for predicted in range(3)
        ]
        for expected in range(3)
    ]


def audit_frame_labels(
    *,
    authoritative_labels: np.ndarray,
    supervision_mask: np.ndarray,
    reconstruction: RayFieldRasterizationV2,
    frame_key: Mapping[str, Any],
) -> dict[str, Any]:
    legacy = v1.audit_frame_labels(
        authoritative_labels=authoritative_labels,
        supervision_mask=supervision_mask,
        reconstruction=reconstruction.v1_rasterization,
        frame_key=frame_key,
    )
    authoritative = np.asarray(authoritative_labels)
    observable = reconstruction.observable_ray_only_labels
    mismatch = authoritative != observable
    result = dict(legacy)
    result.update(
        {
            "schema": FRAME_AUDIT_SCHEMA,
            "observable_ray_only_labels_sha256": hashlib.sha256(
                observable.tobytes(order="C")
            ).hexdigest(),
            "observable_ray_only_class_counts": _class_counts(observable),
            "observable_ray_only_confusion_reference_rows": _confusion(
                authoritative, observable
            ),
            "observable_ray_only_mismatch_cell_count": int(
                np.count_nonzero(mismatch)
            ),
            "observable_ray_only_mismatch_sample": [
                [int(row), int(column)]
                for row, column in np.argwhere(mismatch)[:32]
            ],
            "collision_veto_effect_on_ray_only_cell_count": int(
                np.count_nonzero(
                    observable != reconstruction.collision_vetoed_ray_only_labels
                )
            ),
        }
    )
    return result


def _sum_confusion(
    reports: Sequence[Mapping[str, Any]], name: str
) -> list[list[int]]:
    total = np.zeros((3, 3), dtype=np.int64)
    for report in reports:
        value = np.asarray(report[name])
        if value.shape != (3, 3) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} is malformed")
        if np.any(value < 0):
            raise ValueError(f"{name} contains a negative count")
        total += value.astype(np.int64)
    return total.tolist()


def _mismatch_transitions(confusion: Sequence[Sequence[int]]) -> dict[str, int]:
    matrix = np.asarray(confusion, dtype=np.int64)
    return {
        f"{CLASS_NAMES[expected]}->{CLASS_NAMES[predicted]}": int(
            matrix[expected, predicted]
        )
        for expected in range(3)
        for predicted in range(3)
        if expected != predicted and int(matrix[expected, predicted]) > 0
    }


def _arm_scope(
    reports: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> dict[str, Any]:
    mismatch_field = f"{prefix}_mismatch_cell_count"
    confusion_field = f"{prefix}_confusion_reference_rows"
    confusion = _sum_confusion(reports, confusion_field)
    mismatches = sum(int(report[mismatch_field]) for report in reports)
    return {
        "exact": mismatches == 0,
        "frame_count": len(reports),
        "cell_count": len(reports) * 64 * 64,
        "mismatch_frame_count": sum(
            int(report[mismatch_field]) > 0 for report in reports
        ),
        "mismatch_cell_count": mismatches,
        "confusion_reference_rows": confusion,
        "mismatch_class_transitions": _mismatch_transitions(confusion),
    }


def summarize_exact_fit(frame_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reports = list(frame_reports)
    if len(reports) != 320:
        raise ValueError("the V2 perfect-ray fit audit requires exactly 320 frames")
    if any(report.get("schema") != FRAME_AUDIT_SCHEMA for report in reports):
        raise ValueError("one V2 frame report has the wrong schema")
    keys = [report.get("frame_key") for report in reports]
    encoded_keys = [
        repr(sorted(dict(key).items())) for key in keys if isinstance(key, Mapping)
    ]
    if len(encoded_keys) != 320 or len(set(encoded_keys)) != 320:
        raise ValueError("V2 fit frame keys must be exactly unique")
    family_groups: dict[str, list[Mapping[str, Any]]] = {
        family: [] for family in EXPECTED_FAMILIES
    }
    for report in reports:
        frame_key = report["frame_key"]
        family = str(frame_key.get("family", ""))
        if family not in family_groups:
            raise ValueError("one V2 frame has an unregistered family")
        family_groups[family].append(report)
    if any(not group for group in family_groups.values()):
        raise ValueError("V2 exact fit lacks one or more registered families")

    contract = _arm_scope(reports, prefix="contract")
    collision_vetoed = _arm_scope(reports, prefix="ray_only")
    observable = _arm_scope(reports, prefix="observable_ray_only")
    families = {
        family: {
            "contract_assisted": _arm_scope(group, prefix="contract"),
            "collision_vetoed_ray_only": _arm_scope(group, prefix="ray_only"),
            "observable_ray_only": _arm_scope(
                group, prefix="observable_ray_only"
            ),
        }
        for family, group in family_groups.items()
    }
    return {
        "schema": FIT_AUDIT_SCHEMA,
        "frame_count": 320,
        "cell_count": 320 * 64 * 64,
        "arm_semantics": {
            "contract_assisted": (
                "perfect prescribed camera rays plus zero-inflation physical-free "
                "prior plus privileged collision FREE-to-UNKNOWN veto"
            ),
            "collision_vetoed_ray_only": (
                "perfect prescribed camera rays without physical-free prior, but "
                "with privileged collision FREE-to-UNKNOWN veto; this is V1 ray_only"
            ),
            "observable_ray_only": (
                "perfect prescribed camera rays without physical-free prior and "
                "without collision-geometry veto"
            ),
        },
        "contract_assisted": contract,
        "collision_vetoed_ray_only": collision_vetoed,
        "observable_ray_only": observable,
        "families": families,
        "collision_veto_effect_on_ray_only_cell_count": sum(
            int(report["collision_veto_effect_on_ray_only_cell_count"])
            for report in reports
        ),
        "ordered_frame_keys_sha256": hashlib.sha256(
            "\n".join(encoded_keys).encode("utf-8")
        ).hexdigest(),
        "ordered_authoritative_label_hashes_sha256": hashlib.sha256(
            "\n".join(
                str(report["authoritative_labels_sha256"]) for report in reports
            ).encode("ascii")
        ).hexdigest(),
        "ordered_contract_label_hashes_sha256": hashlib.sha256(
            "\n".join(str(report["contract_labels_sha256"]) for report in reports).encode(
                "ascii"
            )
        ).hexdigest(),
        "ordered_collision_vetoed_ray_only_label_hashes_sha256": hashlib.sha256(
            "\n".join(str(report["ray_only_labels_sha256"]) for report in reports).encode(
                "ascii"
            )
        ).hexdigest(),
        "ordered_observable_ray_only_label_hashes_sha256": hashlib.sha256(
            "\n".join(
                str(report["observable_ray_only_labels_sha256"])
                for report in reports
            ).encode("ascii")
        ).hexdigest(),
    }


__all__ = [
    "CLASS_NAMES",
    "CameraRaySpec",
    "EXPECTED_FAMILIES",
    "FIT_AUDIT_SCHEMA",
    "FRAME_AUDIT_SCHEMA",
    "FREE_CLASS",
    "OCCUPIED_CLASS",
    "OrientedBox",
    "OutputGridSpec",
    "PerfectCameraRayField",
    "PhysicalWindow",
    "RayFieldRasterizationV2",
    "UNKNOWN_CLASS",
    "V1_CORE_PATH",
    "audit_frame_labels",
    "reconstruct_frame_from_perfect_rays",
    "summarize_exact_fit",
]
