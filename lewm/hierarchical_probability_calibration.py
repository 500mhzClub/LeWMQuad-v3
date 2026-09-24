"""Hierarchy-consistent calibration for UNKNOWN/FREE/OCCUPIED logits.

The occupancy head is interpreted as two binary factors: UNKNOWN versus KNOWN,
and FREE versus OCCUPIED conditional on KNOWN.  This module calibrates those
factors with positive affine log-odds maps and reconstructs the three-class
simplex in log space.  It deliberately contains no dataset loader or runtime
policy so the same numerical implementation can be shared by training,
evaluation, and deployment code.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


UNKNOWN_CLASS = 0
FREE_CLASS = 1
OCCUPIED_CLASS = 2
CLASS_ORDER = ("unknown", "free", "occupied")
CALIBRATION_ROLE = "probability_calibration"

CALIBRATION_SCHEMA = "lewm_go2_hierarchical_probability_calibration_v1"
CALIBRATION_METHOD = "positive_affine_hierarchical_log_odds"
OUTPUT_TRANSFORM = (
    "hierarchical_binary_affine_calibration_then_product_simplex_v1"
)
FIT_OBJECTIVE = (
    "natural_prior_joint_multiclass_nll_equivalent_to_unknown_known_plus_"
    "known_free_occupied_binary_nll"
)

_ID_PREFIX = "go2-hier-cal-"
_LOG_SCALE_BOUND = 12.0
_BIAS_BOUND = 64.0
_DEFAULT_ECE_BINS = 15


@dataclass(frozen=True)
class HierarchicalCalibrationParameters:
    """Positive-affine parameters for the two occupancy factors."""

    unknown_known_log_scale: float = 0.0
    unknown_known_bias: float = 0.0
    free_occupied_log_scale: float = 0.0
    free_occupied_bias: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "unknown_known_log_scale": self.unknown_known_log_scale,
            "unknown_known_bias": self.unknown_known_bias,
            "free_occupied_log_scale": self.free_occupied_log_scale,
            "free_occupied_bias": self.free_occupied_bias,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if abs(float(self.unknown_known_log_scale)) > _LOG_SCALE_BOUND:
            raise ValueError("unknown_known_log_scale exceeds the fitted bound")
        if abs(float(self.free_occupied_log_scale)) > _LOG_SCALE_BOUND:
            raise ValueError("free_occupied_log_scale exceeds the fitted bound")
        if abs(float(self.unknown_known_bias)) > _BIAS_BOUND:
            raise ValueError("unknown_known_bias exceeds the fitted bound")
        if abs(float(self.free_occupied_bias)) > _BIAS_BOUND:
            raise ValueError("free_occupied_bias exceeds the fitted bound")

    @property
    def unknown_known_scale(self) -> float:
        return math.exp(float(self.unknown_known_log_scale))

    @property
    def free_occupied_scale(self) -> float:
        return math.exp(float(self.free_occupied_log_scale))


IDENTITY_PARAMETERS = HierarchicalCalibrationParameters()


def _normalize_class_dim(logits: torch.Tensor, class_dim: int) -> int:
    if not isinstance(logits, torch.Tensor) or not torch.is_floating_point(logits):
        raise TypeError("logits must be a floating-point torch.Tensor")
    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions")
    try:
        normalized = int(class_dim) % logits.ndim
    except (TypeError, ValueError) as error:
        raise TypeError("class_dim must be an integer") from error
    if logits.shape[normalized] != len(CLASS_ORDER):
        raise ValueError("occupancy class dimension must have size three")
    return normalized


def hierarchical_log_odds(
    logits: torch.Tensor,
    *,
    class_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return KNOWN and OCCUPIED-given-KNOWN log odds.

    The returned tensors have the class dimension removed.  Both factors are
    invariant to a common offset applied to the three input logits.
    """

    dim = _normalize_class_dim(logits, class_dim)
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite")
    unknown = logits.select(dim, UNKNOWN_CLASS)
    free = logits.select(dim, FREE_CLASS)
    occupied = logits.select(dim, OCCUPIED_CLASS)
    known = torch.logaddexp(free, occupied) - unknown
    occupied_given_known = occupied - free
    return known, occupied_given_known


def _calibrated_factor_logits(
    known_log_odds: torch.Tensor,
    occupied_log_odds: torch.Tensor,
    parameters: HierarchicalCalibrationParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    known = (
        known_log_odds * known_log_odds.new_tensor(parameters.unknown_known_scale)
        + known_log_odds.new_tensor(float(parameters.unknown_known_bias))
    )
    occupied = (
        occupied_log_odds
        * occupied_log_odds.new_tensor(parameters.free_occupied_scale)
        + occupied_log_odds.new_tensor(float(parameters.free_occupied_bias))
    )
    return known, occupied


def _reconstruct_log_probabilities(
    known_logit: torch.Tensor,
    occupied_given_known_logit: torch.Tensor,
    *,
    class_dim: int,
) -> torch.Tensor:
    log_unknown = F.logsigmoid(-known_logit)
    log_known = F.logsigmoid(known_logit)
    log_free = log_known + F.logsigmoid(-occupied_given_known_logit)
    log_occupied = log_known + F.logsigmoid(occupied_given_known_logit)
    result = torch.stack((log_unknown, log_free, log_occupied), dim=class_dim)
    # The product factorization already sums to one mathematically.  Explicit
    # normalization removes the last few ulps of drift and is stable for large
    # finite logits.
    return result - torch.logsumexp(result, dim=class_dim, keepdim=True)


def hierarchical_calibrated_log_probabilities(
    logits: torch.Tensor,
    parameters: HierarchicalCalibrationParameters = IDENTITY_PARAMETERS,
    *,
    class_dim: int = 1,
) -> torch.Tensor:
    """Calibrate occupancy logits and return normalized log probabilities."""

    if not isinstance(parameters, HierarchicalCalibrationParameters):
        raise TypeError("parameters must be HierarchicalCalibrationParameters")
    dim = _normalize_class_dim(logits, class_dim)
    known, occupied = hierarchical_log_odds(logits, class_dim=dim)
    known, occupied = _calibrated_factor_logits(known, occupied, parameters)
    return _reconstruct_log_probabilities(known, occupied, class_dim=dim)


def hierarchical_calibrated_probabilities(
    logits: torch.Tensor,
    parameters: HierarchicalCalibrationParameters = IDENTITY_PARAMETERS,
    *,
    class_dim: int = 1,
) -> torch.Tensor:
    """Calibrate occupancy logits and return UNKNOWN/FREE/OCCUPIED probabilities."""

    return hierarchical_calibrated_log_probabilities(
        logits,
        parameters,
        class_dim=class_dim,
    ).exp()


def apply_hierarchical_probability_calibration(
    logits: torch.Tensor,
    calibration: Mapping[str, Any],
    *,
    class_dim: int = 1,
    return_log_probabilities: bool = False,
) -> torch.Tensor:
    """Validate and apply a fitted calibration artifact.

    Integrations that validate once at checkpoint load time may instead retain
    the returned :class:`HierarchicalCalibrationParameters` and call the
    lower-level functions above on every inference.
    """

    parameters = validate_hierarchical_probability_calibration(calibration)
    log_probabilities = hierarchical_calibrated_log_probabilities(
        logits,
        parameters,
        class_dim=class_dim,
    )
    return log_probabilities if return_log_probabilities else log_probabilities.exp()


def _flatten_cells(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    class_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    dim = _normalize_class_dim(logits, class_dim)
    expected_shape = tuple(logits.shape[:dim]) + tuple(logits.shape[dim + 1 :])
    if not isinstance(labels, torch.Tensor) or tuple(labels.shape) != expected_shape:
        raise ValueError("labels must match logits with the class dimension removed")
    if labels.dtype == torch.bool or torch.is_floating_point(labels):
        raise TypeError("labels must be an integer tensor")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != expected_shape:
            raise ValueError("mask must have the same shape as labels")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")

    flat_logits = (
        logits.detach().movedim(dim, -1).reshape(-1, len(CLASS_ORDER)).to(
            device="cpu", dtype=torch.float64
        )
    )
    flat_labels = labels.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
    flat_mask = (
        torch.ones(flat_labels.shape, dtype=torch.bool)
        if mask is None
        else mask.detach().reshape(-1).to(device="cpu")
    )
    selected_logits = flat_logits[flat_mask].contiguous()
    selected_labels = flat_labels[flat_mask].contiguous()
    if selected_labels.numel() == 0:
        raise ValueError("calibration input contains no valid cells")
    if not bool(torch.isfinite(selected_logits).all().item()):
        raise ValueError("valid calibration logits must be finite")
    if bool(((selected_labels < 0) | (selected_labels >= len(CLASS_ORDER))).any().item()):
        raise ValueError("labels must use UNKNOWN=0, FREE=1, OCCUPIED=2")
    return selected_logits, selected_labels, {
        "source_logits_shape": list(logits.shape),
        "source_labels_shape": list(labels.shape),
        "source_logits_dtype": str(logits.dtype),
        "class_dim": int(dim),
        "source_cell_count": int(flat_labels.numel()),
        "masked_out_cell_count": int((~flat_mask).sum().item()),
        "valid_cell_count": int(selected_labels.numel()),
    }


def _binary_ece(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    bins: int,
) -> float:
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    total = float(probabilities.numel())
    error = 0.0
    for index in range(bins):
        selected = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if index == bins - 1
            else probabilities < edges[index + 1]
        )
        count = int(selected.sum().item())
        if count == 0:
            continue
        confidence = float(probabilities[selected].mean().item())
        frequency = float(targets[selected].mean().item())
        error += count / total * abs(confidence - frequency)
    return float(error)


def _factor_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bins: int,
    positive_class: str,
    negative_class: str,
) -> dict[str, Any]:
    targets = targets.to(dtype=torch.float64)
    probabilities = torch.sigmoid(logits)
    return {
        "positive_class": positive_class,
        "negative_class": negative_class,
        "support_count": int(targets.numel()),
        "positive_count": int(targets.sum().item()),
        "negative_count": int(targets.numel() - targets.sum().item()),
        "nll": float(F.binary_cross_entropy_with_logits(logits, targets).item()),
        "brier": float(torch.square(probabilities - targets).mean().item()),
        "ece": _binary_ece(probabilities, targets, bins=bins),
    }


def _metrics_from_flat_cells(
    logits: torch.Tensor,
    labels: torch.Tensor,
    parameters: HierarchicalCalibrationParameters,
    *,
    bins: int,
) -> dict[str, Any]:
    raw_known, raw_occupied = hierarchical_log_odds(logits, class_dim=1)
    known, occupied = _calibrated_factor_logits(
        raw_known,
        raw_occupied,
        parameters,
    )
    log_probabilities = _reconstruct_log_probabilities(
        known,
        occupied,
        class_dim=1,
    )
    probabilities = log_probabilities.exp()
    one_hot = F.one_hot(labels, num_classes=len(CLASS_ORDER)).to(torch.float64)
    row_indices = torch.arange(labels.numel(), dtype=torch.long)
    confidence, prediction = probabilities.max(dim=1)
    class_counts = torch.bincount(labels, minlength=len(CLASS_ORDER))[:3]
    known_target = labels != UNKNOWN_CLASS
    conditional_labels = labels[known_target]
    return {
        "joint": {
            "sample_count": int(labels.numel()),
            "class_counts": {
                name: int(class_counts[index].item())
                for index, name in enumerate(CLASS_ORDER)
            },
            "nll": float(
                (-log_probabilities[row_indices, labels]).mean().item()
            ),
            "multiclass_brier": float(
                torch.square(probabilities - one_hot).sum(dim=1).mean().item()
            ),
            "confidence_ece": _binary_ece(
                confidence,
                (prediction == labels).to(torch.float64),
                bins=bins,
            ),
            "accuracy": float((prediction == labels).to(torch.float64).mean().item()),
        },
        "unknown_vs_known": _factor_metrics(
            known,
            known_target,
            bins=bins,
            positive_class="known",
            negative_class="unknown",
        ),
        "free_vs_occupied_given_known": _factor_metrics(
            occupied[known_target],
            conditional_labels == OCCUPIED_CLASS,
            bins=bins,
            positive_class="occupied",
            negative_class="free",
        ),
    }


def evaluate_hierarchical_probability_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    calibration: Mapping[str, Any] | HierarchicalCalibrationParameters | None = None,
    *,
    mask: torch.Tensor | None = None,
    class_dim: int = 1,
    ece_bins: int = _DEFAULT_ECE_BINS,
) -> dict[str, Any]:
    """Evaluate joint and factor calibration metrics on natural-prior cells."""

    if isinstance(ece_bins, bool) or int(ece_bins) != ece_bins or ece_bins <= 0:
        raise ValueError("ece_bins must be a positive integer")
    if calibration is None:
        parameters = IDENTITY_PARAMETERS
    elif isinstance(calibration, HierarchicalCalibrationParameters):
        parameters = calibration
    elif isinstance(calibration, Mapping):
        parameters = validate_hierarchical_probability_calibration(calibration)
    else:
        raise TypeError("calibration must be an artifact or parameter object")
    flat_logits, flat_labels, _ = _flatten_cells(
        logits,
        labels,
        mask,
        class_dim=class_dim,
    )
    return _metrics_from_flat_cells(
        flat_logits,
        flat_labels,
        parameters,
        bins=int(ece_bins),
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("calibration provenance must be canonical-JSON serializable") from error


def _normalized_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    normalized = json.loads(_canonical_json(dict(provenance)))
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError("provenance must be a non-empty object")
    if normalized.get("role") != CALIBRATION_ROLE:
        raise ValueError(
            f"calibration provenance role must be {CALIBRATION_ROLE!r}"
        )
    return normalized


def _fit_sample_sha256(logits: torch.Tensor, labels: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema": "lewm_hierarchical_calibration_fit_cells_v1",
                "logits_dtype": "float64",
                "logits_shape": list(logits.shape),
                "labels_dtype": "int64",
                "labels_shape": list(labels.shape),
                "order": "input_row_major_after_mask",
            }
        ).encode("utf-8")
    )
    digest.update(logits.numpy().tobytes(order="C"))
    digest.update(labels.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _bounded_parameters(raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (
        _LOG_SCALE_BOUND * torch.tanh(raw[0] / _LOG_SCALE_BOUND),
        _BIAS_BOUND * torch.tanh(raw[1] / _BIAS_BOUND),
        _LOG_SCALE_BOUND * torch.tanh(raw[2] / _LOG_SCALE_BOUND),
        _BIAS_BOUND * torch.tanh(raw[3] / _BIAS_BOUND),
    )


def _log_probabilities_from_raw_fit_parameters(
    logits: torch.Tensor,
    raw: torch.Tensor,
) -> torch.Tensor:
    known, occupied = hierarchical_log_odds(logits, class_dim=1)
    known_log_scale, known_bias, occupied_log_scale, occupied_bias = (
        _bounded_parameters(raw)
    )
    known = known * torch.exp(known_log_scale) + known_bias
    occupied = occupied * torch.exp(occupied_log_scale) + occupied_bias
    return _reconstruct_log_probabilities(known, occupied, class_dim=1)


def _parameter_record(
    parameters: HierarchicalCalibrationParameters,
) -> dict[str, dict[str, float]]:
    return {
        "unknown_vs_known": {
            "log_scale": float(parameters.unknown_known_log_scale),
            "scale": float(parameters.unknown_known_scale),
            "bias": float(parameters.unknown_known_bias),
        },
        "free_vs_occupied_given_known": {
            "log_scale": float(parameters.free_occupied_log_scale),
            "scale": float(parameters.free_occupied_scale),
            "bias": float(parameters.free_occupied_bias),
        },
    }


def _improvements(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "joint_nll": float(before["joint"]["nll"] - after["joint"]["nll"]),
        "joint_multiclass_brier": float(
            before["joint"]["multiclass_brier"]
            - after["joint"]["multiclass_brier"]
        ),
        "unknown_known_nll": float(
            before["unknown_vs_known"]["nll"]
            - after["unknown_vs_known"]["nll"]
        ),
        "free_occupied_given_known_nll": float(
            before["free_vs_occupied_given_known"]["nll"]
            - after["free_vs_occupied_given_known"]["nll"]
        ),
    }


def _content_sha256(record_without_integrity_fields: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(record_without_integrity_fields).encode("utf-8")
    ).hexdigest()


def fit_hierarchical_probability_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    provenance: Mapping[str, Any],
    mask: torch.Tensor | None = None,
    class_dim: int = 1,
    maximum_iterations: int = 80,
    ece_bins: int = _DEFAULT_ECE_BINS,
) -> dict[str, Any]:
    """Fit hierarchy-consistent calibration on all valid natural-prior cells.

    There is intentionally no weighting, balancing, subsampling, or class
    backfill option.  Every mask-selected cell is optimized exactly once under
    empirical class priors, and absence of any occupancy class fails closed.
    The optimizer has no random operations and always runs on CPU in float64.
    """

    if (
        isinstance(maximum_iterations, bool)
        or int(maximum_iterations) != maximum_iterations
        or maximum_iterations <= 0
    ):
        raise ValueError("maximum_iterations must be a positive integer")
    if isinstance(ece_bins, bool) or int(ece_bins) != ece_bins or ece_bins <= 0:
        raise ValueError("ece_bins must be a positive integer")
    source_provenance = _normalized_provenance(provenance)
    flat_logits, flat_labels, input_record = _flatten_cells(
        logits,
        labels,
        mask,
        class_dim=class_dim,
    )
    counts = torch.bincount(flat_labels, minlength=len(CLASS_ORDER))[:3]
    missing = [
        name for index, name in enumerate(CLASS_ORDER) if int(counts[index].item()) == 0
    ]
    if missing:
        raise ValueError(
            "probability-calibration role lacks required factor support; "
            f"missing={missing}, class_counts="
            f"{dict(zip(CLASS_ORDER, counts.tolist(), strict=True))}; "
            "class backfill is forbidden"
        )

    before = _metrics_from_flat_cells(
        flat_logits,
        flat_labels,
        IDENTITY_PARAMETERS,
        bins=int(ece_bins),
    )
    raw = torch.zeros(4, dtype=torch.float64, device="cpu", requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (raw,),
        lr=1.0,
        max_iter=int(maximum_iterations),
        max_eval=max(1, int(maximum_iterations) * 5 // 4),
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        log_probabilities = _log_probabilities_from_raw_fit_parameters(
            flat_logits,
            raw,
        )
        loss = F.nll_loss(log_probabilities, flat_labels, reduction="mean")
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("hierarchical calibration produced non-finite NLL")
        loss.backward()
        evaluations += 1
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
    with torch.no_grad():
        fitted = _bounded_parameters(raw.detach())
    parameters = HierarchicalCalibrationParameters(
        unknown_known_log_scale=float(fitted[0].item()),
        unknown_known_bias=float(fitted[1].item()),
        free_occupied_log_scale=float(fitted[2].item()),
        free_occupied_bias=float(fitted[3].item()),
    )
    after = _metrics_from_flat_cells(
        flat_logits,
        flat_labels,
        parameters,
        bins=int(ece_bins),
    )
    if after["joint"]["nll"] > before["joint"]["nll"] + 1e-10:
        raise RuntimeError("hierarchical calibration worsened natural-prior joint NLL")

    class_counts = {
        name: int(counts[index].item()) for index, name in enumerate(CLASS_ORDER)
    }
    core: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "method": CALIBRATION_METHOD,
        "class_order": list(CLASS_ORDER),
        "factorization": {
            "unknown_vs_known_log_odds": (
                "logsumexp(free_logit,occupied_logit)-unknown_logit"
            ),
            "free_vs_occupied_given_known_log_odds": (
                "occupied_logit-free_logit"
            ),
        },
        "output_transform": OUTPUT_TRANSFORM,
        "parameters": _parameter_record(parameters),
        "fit": {
            "role": CALIBRATION_ROLE,
            "objective": FIT_OBJECTIVE,
            "prior": "natural_empirical_calibration_role_prior",
            "cell_selection": "all_mask_selected_cells_in_input_order",
            "class_weights": "none",
            "balancing": "none",
            "subsampling": "none",
            "class_backfill": "forbidden",
            "class_support": "unknown_free_and_occupied_each_required_fail_closed",
            "sample_count": int(flat_labels.numel()),
            "class_counts": class_counts,
            "optimizer": "torch.optim.LBFGS_strong_wolfe",
            "device": "cpu",
            "dtype": "float64",
            "randomness": "none",
            "deterministic_execution": True,
            "maximum_iterations": int(maximum_iterations),
            "function_evaluations": int(evaluations),
            "parameter_bounds": {
                "absolute_log_scale_max": _LOG_SCALE_BOUND,
                "absolute_bias_max": _BIAS_BOUND,
                "scale_is_strictly_positive": True,
            },
            "ece_bins": int(ece_bins),
            "torch_version": str(torch.__version__),
        },
        "metrics": {
            "before": before,
            "after": after,
            "improvement": _improvements(before, after),
        },
        "provenance": {
            "source": source_provenance,
            "fit_data": {
                **input_record,
                "fit_cell_count": int(flat_labels.numel()),
                "fit_class_counts": class_counts,
                "fit_sample_sha256": _fit_sample_sha256(
                    flat_logits,
                    flat_labels,
                ),
                "natural_prior_preserved": True,
                "backfilled_cell_count": 0,
                "duplicated_cell_count": 0,
                "dropped_valid_cell_count": 0,
            },
        },
    }
    content_sha256 = _content_sha256(core)
    artifact = {
        **core,
        "content_sha256": content_sha256,
        "id": _ID_PREFIX + content_sha256[:16],
    }
    validate_hierarchical_probability_calibration(artifact)
    return artifact


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _parameters_from_record(record: Mapping[str, Any]) -> HierarchicalCalibrationParameters:
    parameters = record.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != {
        "unknown_vs_known",
        "free_vs_occupied_given_known",
    }:
        raise ValueError("hierarchical calibration factor parameters do not match schema")
    known = parameters["unknown_vs_known"]
    occupied = parameters["free_vs_occupied_given_known"]
    if not isinstance(known, Mapping) or not isinstance(occupied, Mapping):
        raise ValueError("hierarchical calibration factor parameters must be objects")
    if set(known) != {"log_scale", "scale", "bias"} or set(occupied) != {
        "log_scale",
        "scale",
        "bias",
    }:
        raise ValueError("hierarchical calibration factor fields do not match schema")
    result = HierarchicalCalibrationParameters(
        unknown_known_log_scale=_finite_number(
            known["log_scale"], "unknown_vs_known.log_scale"
        ),
        unknown_known_bias=_finite_number(known["bias"], "unknown_vs_known.bias"),
        free_occupied_log_scale=_finite_number(
            occupied["log_scale"], "free_vs_occupied_given_known.log_scale"
        ),
        free_occupied_bias=_finite_number(
            occupied["bias"], "free_vs_occupied_given_known.bias"
        ),
    )
    if not math.isclose(
        _finite_number(known["scale"], "unknown_vs_known.scale"),
        result.unknown_known_scale,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError("unknown_vs_known scale does not equal exp(log_scale)")
    if not math.isclose(
        _finite_number(occupied["scale"], "free_vs_occupied_given_known.scale"),
        result.free_occupied_scale,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "free_vs_occupied_given_known scale does not equal exp(log_scale)"
        )
    return result


def validate_hierarchical_probability_calibration(
    calibration: Mapping[str, Any],
) -> HierarchicalCalibrationParameters:
    """Fail closed on schema, provenance, parameters, metrics, or ID tampering."""

    if not isinstance(calibration, Mapping):
        raise TypeError("calibration must be a mapping")
    required = {
        "schema",
        "method",
        "class_order",
        "factorization",
        "output_transform",
        "parameters",
        "fit",
        "metrics",
        "provenance",
        "content_sha256",
        "id",
    }
    if set(calibration) != required:
        raise ValueError("hierarchical probability calibration fields do not match v1")
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported hierarchical probability calibration schema")
    if calibration.get("method") != CALIBRATION_METHOD:
        raise ValueError("unsupported hierarchical probability calibration method")
    if calibration.get("class_order") != list(CLASS_ORDER):
        raise ValueError("hierarchical probability calibration class order changed")
    if calibration.get("output_transform") != OUTPUT_TRANSFORM:
        raise ValueError("unsupported hierarchical calibration output transform")
    parameters = _parameters_from_record(calibration)

    fit = calibration.get("fit")
    if not isinstance(fit, Mapping):
        raise ValueError("hierarchical calibration fit record must be an object")
    expected_fit_values = {
        "role": CALIBRATION_ROLE,
        "objective": FIT_OBJECTIVE,
        "prior": "natural_empirical_calibration_role_prior",
        "cell_selection": "all_mask_selected_cells_in_input_order",
        "class_weights": "none",
        "balancing": "none",
        "subsampling": "none",
        "class_backfill": "forbidden",
        "device": "cpu",
        "dtype": "float64",
        "randomness": "none",
        "deterministic_execution": True,
    }
    for name, expected in expected_fit_values.items():
        if fit.get(name) != expected:
            raise ValueError(f"hierarchical calibration fit contract changed: {name}")
    class_counts = fit.get("class_counts")
    if not isinstance(class_counts, Mapping) or set(class_counts) != set(CLASS_ORDER):
        raise ValueError("hierarchical calibration class counts do not match schema")
    parsed_counts: dict[str, int] = {}
    for name in CLASS_ORDER:
        value = class_counts[name]
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError("every hierarchical calibration class requires support")
        parsed_counts[name] = int(value)
    sample_count = fit.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or int(sample_count) != sample_count
        or int(sample_count) != sum(parsed_counts.values())
    ):
        raise ValueError("hierarchical calibration sample count is inconsistent")

    provenance = calibration.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {"source", "fit_data"}:
        raise ValueError("hierarchical calibration provenance does not match schema")
    _normalized_provenance(provenance["source"])
    fit_data = provenance["fit_data"]
    if not isinstance(fit_data, Mapping):
        raise ValueError("hierarchical calibration fit-data provenance is missing")
    required_fit_data = {
        "fit_cell_count": int(sample_count),
        "fit_class_counts": parsed_counts,
        "natural_prior_preserved": True,
        "backfilled_cell_count": 0,
        "duplicated_cell_count": 0,
        "dropped_valid_cell_count": 0,
    }
    for name, expected in required_fit_data.items():
        if fit_data.get(name) != expected:
            raise ValueError(f"hierarchical calibration provenance changed: {name}")
    sample_sha256 = str(fit_data.get("fit_sample_sha256", ""))
    if len(sample_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sample_sha256
    ):
        raise ValueError("hierarchical calibration fit sample digest is invalid")

    metrics = calibration.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "before",
        "after",
        "improvement",
    }:
        raise ValueError("hierarchical calibration metrics do not match schema")
    try:
        before_nll = _finite_number(metrics["before"]["joint"]["nll"], "before NLL")
        after_nll = _finite_number(metrics["after"]["joint"]["nll"], "after NLL")
    except (KeyError, TypeError) as error:
        raise ValueError("hierarchical calibration joint metrics are missing") from error
    if after_nll > before_nll + 1e-10:
        raise ValueError("hierarchical calibration worsens joint NLL")

    unhashed = dict(calibration)
    recorded_content_sha256 = str(unhashed.pop("content_sha256", ""))
    recorded_id = str(unhashed.pop("id", ""))
    expected_content_sha256 = _content_sha256(unhashed)
    if recorded_content_sha256 != expected_content_sha256:
        raise ValueError("hierarchical calibration content digest does not match")
    if recorded_id != _ID_PREFIX + expected_content_sha256[:16]:
        raise ValueError("hierarchical calibration ID does not match its content")
    return parameters


__all__ = [
    "CALIBRATION_METHOD",
    "CALIBRATION_ROLE",
    "CALIBRATION_SCHEMA",
    "CLASS_ORDER",
    "FIT_OBJECTIVE",
    "FREE_CLASS",
    "HierarchicalCalibrationParameters",
    "IDENTITY_PARAMETERS",
    "OCCUPIED_CLASS",
    "OUTPUT_TRANSFORM",
    "UNKNOWN_CLASS",
    "apply_hierarchical_probability_calibration",
    "evaluate_hierarchical_probability_calibration",
    "fit_hierarchical_probability_calibration",
    "hierarchical_calibrated_log_probabilities",
    "hierarchical_calibrated_probabilities",
    "hierarchical_log_odds",
    "validate_hierarchical_probability_calibration",
]
