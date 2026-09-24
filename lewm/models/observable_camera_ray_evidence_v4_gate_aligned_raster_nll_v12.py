"""Gate-aligned all-cell raster NLL for the Camera V12 successor.

The retained Camera V11 losses remain unchanged.  This module supplies the
single additive V12 term and aggregate-only diagnostics computed from the same
soft-raster class probabilities consumed by the retained metric accumulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


RASTER_CLASS_NAMES_V12 = ("UNKNOWN", "FREE", "OCCUPIED")
RASTER_NLL_WEIGHT_V12 = 0.25


def _validated_inputs(
    class_probabilities: Tensor,
    target_raster_labels: Tensor,
) -> tuple[Tensor, Tensor]:
    if not isinstance(class_probabilities, Tensor) or not isinstance(
        target_raster_labels, Tensor
    ):
        raise TypeError("V12 raster NLL inputs must be torch tensors")
    if class_probabilities.dtype != torch.float32:
        raise TypeError("V12 raster class probabilities must be float32")
    if target_raster_labels.dtype == torch.bool or target_raster_labels.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("V12 raster target labels must have an integer dtype")
    if class_probabilities.ndim != 4 or class_probabilities.shape[1] != 3:
        raise ValueError("V12 raster class probabilities must have shape (B,3,H,W)")
    if target_raster_labels.ndim != 3 or tuple(target_raster_labels.shape) != (
        class_probabilities.shape[0],
        class_probabilities.shape[2],
        class_probabilities.shape[3],
    ):
        raise ValueError("V12 raster target labels must have shape (B,H,W)")
    if class_probabilities.device != target_raster_labels.device:
        raise ValueError("V12 raster probabilities and labels must share a device")
    if class_probabilities.numel() == 0:
        raise ValueError("V12 raster NLL requires at least one target cell")
    if not bool(torch.isfinite(class_probabilities).all().item()):
        raise ValueError("V12 raster class probabilities must be finite")
    if bool((class_probabilities < 0.0).any().item()) or bool(
        (class_probabilities > 1.0).any().item()
    ):
        raise ValueError("V12 raster class probabilities must lie in [0,1]")
    probability_sums = class_probabilities.sum(dim=1)
    if not bool(
        torch.allclose(
            probability_sums,
            torch.ones_like(probability_sums),
            rtol=0.0,
            atol=8.0 * torch.finfo(torch.float32).eps,
        )
    ):
        raise ValueError("V12 raster class probabilities must be normalized")
    labels = target_raster_labels.to(dtype=torch.long)
    if bool((labels < 0).any().item()) or bool((labels >= 3).any().item()):
        raise ValueError("V12 raster target labels must lie in [0,2]")
    return class_probabilities, labels


def target_cell_nll_map_v12(
    class_probabilities: Tensor,
    target_raster_labels: Tensor,
) -> Tensor:
    """Return one differentiable target-class NLL value per raster cell."""

    probabilities, labels = _validated_inputs(
        class_probabilities, target_raster_labels
    )
    epsilon = torch.finfo(probabilities.dtype).eps
    target_probabilities = probabilities.gather(1, labels[:, None]).squeeze(1)
    return -target_probabilities.clamp_min(epsilon).log()


def derived_raster_cell_nll_v12(
    class_probabilities: Tensor,
    target_raster_labels: Tensor,
) -> Tensor:
    """Exact V12 ``gather -> clamp -> log -> all-cell mean`` objective."""

    return target_cell_nll_map_v12(
        class_probabilities, target_raster_labels
    ).mean()


@dataclass(frozen=True)
class GateAlignedObjectiveV12:
    v11_base_total: Tensor
    derived_raster_cell_nll: Tensor
    total: Tensor


def compose_gate_aligned_objective_v12(
    retained_v11_components: Mapping[str, Tensor],
    derived_raster_cell_nll: Tensor,
) -> GateAlignedObjectiveV12:
    """Add exactly ``0.25 * G`` to the four equally weighted V11 terms."""

    expected = (
        "hierarchical_first_hit_nll",
        "target_bin_offset_smooth_l1",
        "ground_clear_distance_state_balanced_bce",
        "derived_raster_hierarchical_bce",
    )
    if tuple(retained_v11_components) != expected:
        raise ValueError("V12 retained V11 component order or fields changed")
    values = tuple(retained_v11_components[name] for name in expected)
    if any(not isinstance(value, Tensor) or value.ndim != 0 for value in values):
        raise TypeError("V12 retained V11 components must be scalar tensors")
    if not isinstance(derived_raster_cell_nll, Tensor) or (
        derived_raster_cell_nll.ndim != 0
    ):
        raise TypeError("V12 derived raster cell NLL must be a scalar tensor")
    if any(value.device != derived_raster_cell_nll.device for value in values):
        raise ValueError("V12 loss components must share one device")
    v11_base_total = 0.25 * sum(values)
    total = v11_base_total + RASTER_NLL_WEIGHT_V12 * derived_raster_cell_nll
    return GateAlignedObjectiveV12(
        v11_base_total=v11_base_total,
        derived_raster_cell_nll=derived_raster_cell_nll,
        total=total,
    )


def _summary(count: int, nll_sum: float) -> dict[str, int | float | None]:
    if isinstance(count, bool) or count < 0:
        raise ValueError("V12 diagnostic count is invalid")
    if not math.isfinite(nll_sum) or nll_sum < 0.0:
        raise ValueError("V12 diagnostic NLL sum is invalid")
    return {
        "count": count,
        "nll_sum": nll_sum,
        "mean": None if count == 0 else nll_sum / count,
    }


def raster_nll_diagnostics_v12(
    class_probabilities: Tensor,
    target_raster_labels: Tensor,
    families: Sequence[str],
) -> dict[str, Any]:
    """Publish aggregate-only overall, target-class, and family NLL partitions."""

    probabilities, labels = _validated_inputs(
        class_probabilities, target_raster_labels
    )
    if len(families) != probabilities.shape[0] or any(
        not isinstance(family, str) or not family for family in families
    ):
        raise ValueError("V12 raster NLL families must name every batch row")
    nll = target_cell_nll_map_v12(probabilities, labels).detach().to("cpu")
    labels_cpu = labels.detach().to("cpu")
    overall_count = int(nll.numel())
    overall_sum = float(nll.to(dtype=torch.float64).sum().item())
    by_class: dict[str, dict[str, int | float | None]] = {}
    for class_index, class_name in enumerate(RASTER_CLASS_NAMES_V12):
        mask = labels_cpu == class_index
        count = int(mask.sum().item())
        nll_sum = float(nll[mask].to(dtype=torch.float64).sum().item())
        by_class[class_name] = _summary(count, nll_sum)
    by_family: dict[str, dict[str, int | float | None]] = {}
    for batch_index, family in enumerate(families):
        family_nll = nll[batch_index]
        row = by_family.setdefault(family, _summary(0, 0.0))
        count = int(row["count"]) + int(family_nll.numel())
        nll_sum = float(row["nll_sum"]) + float(
            family_nll.to(dtype=torch.float64).sum().item()
        )
        by_family[family] = _summary(count, nll_sum)
    result = {
        "overall": _summary(overall_count, overall_sum),
        "by_target_class": by_class,
        "by_family": {key: by_family[key] for key in sorted(by_family)},
    }
    validate_raster_nll_diagnostics_v12(result)
    return result


def merge_raster_nll_diagnostics_v12(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge batch-one diagnostic records without reusing a training scalar."""

    if not records:
        raise ValueError("V12 raster NLL diagnostic merge requires records")
    class_totals = {name: [0, 0.0] for name in RASTER_CLASS_NAMES_V12}
    family_totals: dict[str, list[int | float]] = {}
    overall_count = 0
    overall_sum = 0.0
    for record in records:
        checked = validate_raster_nll_diagnostics_v12(record)
        overall_count += int(checked["overall"]["count"])
        overall_sum += float(checked["overall"]["nll_sum"])
        for name in RASTER_CLASS_NAMES_V12:
            row = checked["by_target_class"][name]
            class_totals[name][0] = int(class_totals[name][0]) + int(row["count"])
            class_totals[name][1] = float(class_totals[name][1]) + float(
                row["nll_sum"]
            )
        for family, row in checked["by_family"].items():
            total = family_totals.setdefault(family, [0, 0.0])
            total[0] = int(total[0]) + int(row["count"])
            total[1] = float(total[1]) + float(row["nll_sum"])
    merged = {
        "overall": _summary(overall_count, overall_sum),
        "by_target_class": {
            name: _summary(int(value[0]), float(value[1]))
            for name, value in class_totals.items()
        },
        "by_family": {
            name: _summary(int(value[0]), float(value[1]))
            for name, value in sorted(family_totals.items())
        },
    }
    validate_raster_nll_diagnostics_v12(merged)
    return merged


def validate_raster_nll_diagnostics_v12(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on malformed or arithmetically inconsistent diagnostics."""

    if not isinstance(value, Mapping) or set(value) != {
        "overall",
        "by_target_class",
        "by_family",
    }:
        raise ValueError("V12 raster NLL diagnostic fields changed")

    def checked_row(row: object, *, name: str) -> dict[str, int | float | None]:
        if not isinstance(row, Mapping) or set(row) != {"count", "nll_sum", "mean"}:
            raise ValueError(f"V12 {name} diagnostic fields changed")
        count = row["count"]
        nll_sum = row["nll_sum"]
        mean = row["mean"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"V12 {name} count is invalid")
        if isinstance(nll_sum, bool) or not isinstance(nll_sum, (int, float)):
            raise ValueError(f"V12 {name} NLL sum is not numeric")
        normalized_sum = float(nll_sum)
        if not math.isfinite(normalized_sum) or normalized_sum < 0.0:
            raise ValueError(f"V12 {name} NLL sum is invalid")
        expected_mean = None if count == 0 else normalized_sum / count
        if expected_mean is None:
            if mean is not None:
                raise ValueError(f"V12 {name} empty mean must be null")
        elif (
            isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(float(mean))
            or not math.isclose(
                float(mean), expected_mean, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError(f"V12 {name} mean is inconsistent")
        return {"count": count, "nll_sum": normalized_sum, "mean": expected_mean}

    overall = checked_row(value["overall"], name="overall")
    by_class = value["by_target_class"]
    if not isinstance(by_class, Mapping) or set(by_class) != set(
        RASTER_CLASS_NAMES_V12
    ):
        raise ValueError("V12 raster NLL class partition changed")
    checked_classes = {
        name: checked_row(by_class[name], name=f"class {name}")
        for name in RASTER_CLASS_NAMES_V12
    }
    by_family = value["by_family"]
    if (
        not isinstance(by_family, Mapping)
        or not by_family
        or list(by_family) != sorted(by_family)
        or any(not isinstance(name, str) or not name for name in by_family)
    ):
        raise ValueError("V12 raster NLL family partition changed")
    checked_families = {
        name: checked_row(row, name=f"family {name}")
        for name, row in by_family.items()
    }
    for partition_name, rows in (
        ("class", checked_classes.values()),
        ("family", checked_families.values()),
    ):
        if sum(int(row["count"]) for row in rows) != overall["count"]:
            raise ValueError(f"V12 {partition_name} counts do not partition overall")
        if not math.isclose(
            sum(float(row["nll_sum"]) for row in rows),
            float(overall["nll_sum"]),
            rel_tol=0.0,
            abs_tol=max(1e-10, 1e-12 * max(1.0, float(overall["nll_sum"]))),
        ):
            raise ValueError(f"V12 {partition_name} NLL sums do not reconstruct overall")
    return {
        "overall": overall,
        "by_target_class": checked_classes,
        "by_family": checked_families,
    }


def branch_reduction_decomposition_v12(
    class_probabilities: Tensor,
    target_raster_labels: Tensor,
) -> dict[str, dict[str, int | float | None] | float]:
    """Expose the analytical R/O/U/F reduction for synthetic proof only."""

    probabilities, labels = _validated_inputs(
        class_probabilities, target_raster_labels
    )
    epsilon = torch.finfo(probabilities.dtype).eps
    non_occupied_probability = (
        probabilities[:, 0] + probabilities[:, 1]
    ).clamp_min(epsilon)
    branch_maps = {
        "R": -non_occupied_probability.log(),
        "O": -probabilities[:, 2].clamp_min(epsilon).log(),
        "U": -(probabilities[:, 0] / non_occupied_probability)
        .clamp_min(epsilon)
        .log(),
        "F": -(probabilities[:, 1] / non_occupied_probability)
        .clamp_min(epsilon)
        .log(),
    }
    masks = {
        "R": labels != 2,
        "O": labels == 2,
        "U": labels == 0,
        "F": labels == 1,
    }
    rows: dict[str, dict[str, int | float | None]] = {}
    for name in ("R", "O", "U", "F"):
        values = branch_maps[name][masks[name]].detach().to(dtype=torch.float64)
        rows[name] = _summary(int(values.numel()), float(values.sum().item()))
    count = int(labels.numel())
    weighted_sum = (
        float(rows["R"]["nll_sum"])
        + float(rows["O"]["nll_sum"])
        + float(rows["U"]["nll_sum"])
        + float(rows["F"]["nll_sum"])
    )
    result: dict[str, dict[str, int | float | None] | float] = {
        **rows,
        "cell_micro_mean": weighted_sum / count,
    }
    direct = float(
        derived_raster_cell_nll_v12(probabilities, labels).detach().cpu().item()
    )
    if not math.isclose(
        float(result["cell_micro_mean"]), direct, rel_tol=0.0, abs_tol=2e-6
    ):
        raise ValueError("V12 R/O/U/F decomposition does not reconstruct cell NLL")
    return result
