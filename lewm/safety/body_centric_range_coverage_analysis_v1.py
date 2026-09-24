"""Pure analysis layer for body-centric range coverage qualification V1.

This module is deliberately downstream of sensor materialisation and upstream
of report persistence.  It performs no file I/O, corpus loading, ray casting,
simulation, model inference, or training.  Callers provide one normalized
record per frozen transition and, optionally, already-built frozen two-ply
state structures.

The minimum transition-record schema is::

    {
        "identity": "globally unique transition identity",
        "state_id": "...",
        "role": "calibration" | "heldout" | "training",
        "family": "...",
        "level": "current" | "successor",
        "current_action_index": -1 | 0..13,
        "action_index": 0..13,
        "controller": "route" | "lateral",
        "applied_action": [...],
        "oracle_contact": False,
        "clearance_m": 0.12,
        "unsupported": False,
    }

Current rows additionally carry the frozen H3 fields used by
``body_centric_range_coverage_metrics_v1``.  Per-link evidence may be supplied
as a ``per_link`` mapping.  The accepted per-link fields are documented by
``_normalise_link_evidence`` below; values may be scalars or physics-step
vectors, so materialisers do not need to pre-reduce coverage fractions.

Unsupported transition evidence is conservative: it is passed verbatim to
the frozen contact reducer, which always predicts unsupported or non-finite
clearance as contact risk.  This module never silently treats missing support
as free space.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import math
from typing import Any

import numpy as np

from lewm.safety import body_centric_range_coverage_metrics_v1 as METRICS


class AnalysisError(ValueError):
    """Raised when materialised evidence is incomplete or inconsistent."""


_CONTACT_ALIASES = (
    "oracle_contact",
    "frozen_contact",
    "exact_contact",
    "current_contact",
    "contact",
)
_UID_ALIASES = ("transition_uid", "identity")


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnalysisError(f"{location} must be a non-string sequence")
    return value


def _finite_float(value: Any, location: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{location} must be numeric") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{location} must be finite")
    return result


def _contact(row: Mapping[str, Any], location: str) -> bool:
    for key in _CONTACT_ALIASES:
        if key in row:
            value = row[key]
            if not isinstance(value, (bool, np.bool_)):
                raise AnalysisError(f"{location}.{key} must be bool")
            return bool(value)
    raise AnalysisError(f"{location} lacks an oracle contact field")


def transition_key(
    row: Mapping[str, Any],
    *,
    state_id: str | None = None,
    level: str | None = None,
    current_action_index: int | None = None,
) -> tuple[Any, ...]:
    """Return the immutable lookup key used to join evidence to state rows."""

    for name in _UID_ALIASES:
        if row.get(name) is not None:
            return ("uid", str(row[name]))
    resolved_state = str(row.get("state_id", state_id or ""))
    resolved_level = str(row.get("level", level or ""))
    if resolved_level not in ("current", "successor"):
        raise AnalysisError("transition key requires level current/successor")
    if not resolved_state:
        raise AnalysisError("transition key requires state_id")
    prefix = int(
        row.get(
            "current_action_index",
            -1 if resolved_level == "current" else current_action_index,
        )
    )
    if resolved_level == "successor" and prefix < 0:
        raise AnalysisError("successor transition key requires current_action_index")
    if "action_index" not in row:
        raise AnalysisError("transition key requires action_index")
    return ("tuple", resolved_state, resolved_level, prefix, int(row["action_index"]))


def _validate_record(row: Mapping[str, Any], index: int) -> None:
    location = f"records[{index}]"
    for key in ("state_id", "role", "family", "level", "action_index", "clearance_m"):
        if key not in row:
            raise AnalysisError(f"{location} lacks {key}")
    if str(row["level"]) not in ("current", "successor"):
        raise AnalysisError(f"{location}.level must be current/successor")
    try:
        float(row["clearance_m"])
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{location}.clearance_m must be numeric") from exc
    if "unsupported" in row and not isinstance(row["unsupported"], (bool, np.bool_)):
        raise AnalysisError(f"{location}.unsupported must be bool")
    _contact(row, location)
    transition_key(row)


def validate_transition_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Validate normalized records and reject duplicate immutable identities."""

    source = _sequence(records, "records")
    output: list[Mapping[str, Any]] = []
    observed: set[tuple[Any, ...]] = set()
    for index, row in enumerate(source):
        if not isinstance(row, Mapping):
            raise AnalysisError(f"records[{index}] must be a mapping")
        _validate_record(row, index)
        key = transition_key(row)
        if key in observed:
            raise AnalysisError(f"duplicate transition identity {key}")
        observed.add(key)
        output.append(row)
    if not output:
        raise AnalysisError("records must not be empty")
    return tuple(output)


def transition_records_from_arrays(
    identity_rows: Sequence[Mapping[str, Any]],
    *,
    clearance_m: Any,
    unsupported: Any,
    link_names: Sequence[str] | None = None,
    per_link_clearance_m: Any | None = None,
    per_link_support: Any | None = None,
    per_link_contact: Any | None = None,
    per_link_nominal_fov: Any | None = None,
    per_link_direct_visibility: Any | None = None,
    per_link_point_support_count: Any | None = None,
    body_region_by_link: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Join frozen identity rows to one condition/mode materialisation slice.

    ``clearance_m`` and ``unsupported`` must be explicit one-dimensional
    decision arrays.  Requiring the decision-level unsupported flag here is
    intentional: the analysis layer must not invent whether partial swept
    volume support is sufficient.

    Per-link arrays may have shape ``[transition, link]`` or
    ``[transition, physics_step, link]``.  Contact is reduced with ``any`` over
    physics steps; coverage fractions retain all physics steps.  This helper
    copies inputs and never mutates or persists materialised evidence.
    """

    identities = _sequence(identity_rows, "identity_rows")
    rows = len(identities)
    global_clearance = np.asarray(clearance_m, dtype=np.float64)
    global_unsupported = np.asarray(unsupported, dtype=bool)
    if global_clearance.shape != (rows,):
        raise AnalysisError(
            f"clearance_m must have shape ({rows},), got {global_clearance.shape}"
        )
    if global_unsupported.shape != (rows,):
        raise AnalysisError(
            f"unsupported must have shape ({rows},), got {global_unsupported.shape}"
        )

    link_arrays = (
        per_link_clearance_m,
        per_link_support,
        per_link_contact,
        per_link_nominal_fov,
        per_link_direct_visibility,
        per_link_point_support_count,
    )
    has_link_evidence = any(value is not None for value in link_arrays)
    if has_link_evidence and (link_names is None or per_link_clearance_m is None):
        raise AnalysisError(
            "per-link evidence requires link_names and per_link_clearance_m"
        )
    names = [] if link_names is None else [str(value) for value in link_names]
    if len(set(names)) != len(names):
        raise AnalysisError("link_names must be unique")

    def trajectory(value: Any, name: str, dtype: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=dtype)
        if array.ndim == 2:
            array = array[:, None, :]
        expected_prefix = (rows,)
        if array.ndim != 3 or array.shape[0:1] != expected_prefix or array.shape[2] != len(names):
            raise AnalysisError(
                f"{name} must have shape [transition,link] or "
                f"[transition,physics_step,link], got {array.shape}"
            )
        return array

    link_clearance = trajectory(per_link_clearance_m, "per_link_clearance_m", np.float64)
    link_support = trajectory(per_link_support, "per_link_support", bool)
    link_contact = trajectory(per_link_contact, "per_link_contact", bool)
    link_nominal = trajectory(per_link_nominal_fov, "per_link_nominal_fov", bool)
    link_direct = trajectory(
        per_link_direct_visibility, "per_link_direct_visibility", bool
    )
    link_point_count = trajectory(
        per_link_point_support_count,
        "per_link_point_support_count",
        np.float64,
    )
    step_counts = {
        array.shape[1]
        for array in (
            link_clearance,
            link_support,
            link_contact,
            link_nominal,
            link_direct,
            link_point_count,
        )
        if array is not None and array.shape[1] != 1
    }
    if len(step_counts) > 1:
        raise AnalysisError("per-link physics-step dimensions disagree")

    regions = {} if body_region_by_link is None else dict(body_region_by_link)
    output: list[dict[str, Any]] = []
    for index, identity in enumerate(identities):
        if not isinstance(identity, Mapping):
            raise AnalysisError(f"identity_rows[{index}] must be a mapping")
        record = copy.deepcopy(dict(identity))
        record["clearance_m"] = float(global_clearance[index])
        record["unsupported"] = bool(global_unsupported[index])
        if link_clearance is not None:
            per_link: dict[str, dict[str, Any]] = {}
            for link_index, link_name in enumerate(names):
                clearance_value = link_clearance[index, :, link_index]
                row: dict[str, Any] = {
                    "clearance_m": clearance_value.copy(),
                    "body_region": str(regions.get(link_name, "UNSPECIFIED")),
                }
                if link_support is not None:
                    row["support"] = link_support[index, :, link_index].copy()
                    row["unsupported_swept_volume_fraction"] = float(
                        1.0 - link_support[index, :, link_index].mean()
                    )
                if link_contact is not None:
                    row["oracle_contact"] = bool(
                        link_contact[index, :, link_index].any()
                    )
                if link_nominal is not None:
                    row["nominal_fov"] = link_nominal[index, :, link_index].copy()
                if link_direct is not None:
                    row["direct_visibility"] = link_direct[index, :, link_index].copy()
                if link_point_count is not None:
                    row["point_support_count"] = link_point_count[
                        index, :, link_index
                    ].copy()
                per_link[link_name] = row
            record["per_link"] = per_link
        output.append(record)
    validate_transition_records(output)
    return output


def _action_row(record: Mapping[str, Any]) -> dict[str, Any]:
    action_index = int(record["action_index"])
    row: dict[str, Any] = {
        "transition_uid": str(record.get("transition_uid", record.get("identity", ""))),
        "state_id": str(record["state_id"]),
        "level": str(record["level"]),
        "current_action_index": int(record.get("current_action_index", -1)),
        "action_index": action_index,
        "controller": str(
            record.get("controller", "route" if action_index < 12 else "lateral")
        ),
        "oracle_contact": _contact(record, "record"),
    }
    if not row["transition_uid"]:
        row.pop("transition_uid")
    if record.get("applied_action") is not None:
        row["applied_action"] = tuple(float(value) for value in record["applied_action"])
    elif record.get("action_identity") is not None:
        row["action_identity"] = copy.deepcopy(record["action_identity"])
    for field in (
        "h3_progress_m",
        "h3_heading_improvement_rad",
        "decision_progress_m",
    ):
        if field in record:
            row[field] = record[field]
    if str(record["level"]) == "current":
        if action_index < 12:
            for field in (
                "h3_progress_m",
                "h3_heading_improvement_rad",
                "decision_progress_m",
            ):
                if row.get(field) is None:
                    raise AnalysisError(f"current route action lacks {field}")
        elif row.get("decision_progress_m") is None:
            raise AnalysisError("current lateral action lacks decision_progress_m")
    return row


def build_two_ply_states(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build immutable-metadata state structures from transition records.

    Predictions are intentionally absent.  Use :func:`inject_contact_predictions`
    for a particular threshold.  The construction retains the frozen input
    order; physical action deduplication remains delegated to the exact metrics
    reducer and its applied-action identity contract.
    """

    source = validate_transition_records(records)
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    state_metadata: dict[str, tuple[str, str]] = {}
    for row in source:
        state_id = str(row["state_id"])
        metadata = (str(row["family"]), str(row["role"]))
        if state_id in state_metadata and state_metadata[state_id] != metadata:
            raise AnalysisError(f"state {state_id} has inconsistent family/role")
        state_metadata[state_id] = metadata
        by_state[state_id].append(row)

    output: list[dict[str, Any]] = []
    for state_id, rows in by_state.items():
        current = [row for row in rows if str(row["level"]) == "current"]
        successor: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            if str(row["level"]) == "successor":
                successor[int(row["current_action_index"])].append(row)
        if not current:
            raise AnalysisError(f"state {state_id} lacks current rows")
        actions: list[dict[str, Any]] = []
        for record in current:
            action = _action_row(record)
            prefix = int(record["action_index"])
            action["next_actions"] = [
                _action_row(next_record) for next_record in successor.get(prefix, ())
            ]
            actions.append(action)
        unused_prefixes = set(successor) - {int(row["action_index"]) for row in current}
        if unused_prefixes:
            raise AnalysisError(
                f"state {state_id} has successor prefixes without current rows: "
                f"{sorted(unused_prefixes)}"
            )
        family, role = state_metadata[state_id]
        output.append(
            {
                "state_id": state_id,
                "family": family,
                "role": role,
                "current_actions": actions,
            }
        )
    return output


def _prediction_lookup(
    records: Sequence[Mapping[str, Any]], threshold_m: float
) -> dict[tuple[Any, ...], bool]:
    source = validate_transition_records(records)
    clearance = np.asarray([float(row["clearance_m"]) for row in source], np.float64)
    unsupported = np.asarray([bool(row.get("unsupported", False)) for row in source], bool)
    predicted = METRICS.contact_predictions(clearance, threshold_m, unsupported)
    return {
        transition_key(row): bool(value)
        for row, value in zip(source, predicted, strict=True)
    }


def inject_contact_predictions(
    states: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    threshold_m: float,
) -> list[dict[str, Any]]:
    """Copy state templates and inject deterministic threshold predictions."""

    threshold = _finite_float(threshold_m, "threshold_m")
    predictions = _prediction_lookup(records, threshold)
    output: list[dict[str, Any]] = []
    used: set[tuple[Any, ...]] = set()
    for state_index, source_state in enumerate(_sequence(states, "states")):
        if not isinstance(source_state, Mapping):
            raise AnalysisError(f"states[{state_index}] must be a mapping")
        state = copy.deepcopy(dict(source_state))
        state_id = str(state.get("state_id", ""))
        if not state_id:
            raise AnalysisError(f"states[{state_index}] lacks state_id")
        current = state.get("current_actions", state.get("current_rows"))
        current = _sequence(current, f"states[{state_index}].current_actions")
        replaced: list[dict[str, Any]] = []
        for action_source in current:
            action = copy.deepcopy(dict(action_source))
            key = transition_key(action, state_id=state_id, level="current")
            if key not in predictions:
                raise AnalysisError(f"no materialised prediction for {key}")
            action["predicted_contact"] = predictions[key]
            used.add(key)
            next_source = action.get("next_actions")
            if next_source is not None:
                next_rows: list[dict[str, Any]] = []
                for next_item in _sequence(next_source, "next_actions"):
                    next_row = copy.deepcopy(dict(next_item))
                    next_key = transition_key(
                        next_row,
                        state_id=state_id,
                        level="successor",
                        current_action_index=int(action["action_index"]),
                    )
                    if next_key not in predictions:
                        raise AnalysisError(f"no materialised prediction for {next_key}")
                    next_row["predicted_contact"] = predictions[next_key]
                    used.add(next_key)
                    next_rows.append(next_row)
                action["next_actions"] = next_rows
            replaced.append(action)
        state["current_actions"] = replaced
        state.pop("current_rows", None)
        output.append(state)
    missing = set(predictions) - used
    if missing:
        raise AnalysisError(
            f"{len(missing)} transition predictions are not bound by supplied state structures"
        )
    return output


def _role_records(
    records: Sequence[Mapping[str, Any]], role: str
) -> tuple[Mapping[str, Any], ...]:
    aliases = {
        "internal_calibration": "calibration",
        "development_heldout": "heldout",
    }
    wanted = aliases.get(str(role), str(role))
    selected = tuple(
        row
        for row in validate_transition_records(records)
        if aliases.get(str(row["role"]), str(row["role"])) == wanted
    )
    if not selected:
        raise AnalysisError(f"no records for role {role}")
    return selected


def _states_for_records(
    states: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    state_ids = {str(row["state_id"]) for row in records}
    selected = [state for state in states if str(state.get("state_id", "")) in state_ids]
    if {str(state["state_id"]) for state in selected} != state_ids:
        raise AnalysisError("state structures do not cover every transition-record state")
    return selected


def calibrate_threshold(
    records: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]] | None = None,
    *,
    role: str = "calibration",
) -> dict[str, Any]:
    """Enumerate the exact frozen calibration frontier and select one threshold."""

    calibration = _role_records(records, role)
    templates = (
        build_two_ply_states(calibration)
        if states is None
        else _states_for_records(states, calibration)
    )
    labels = np.asarray([_contact(row, "calibration row") for row in calibration], bool)
    clearance = np.asarray([float(row["clearance_m"]) for row in calibration], np.float64)
    unsupported = np.asarray(
        [bool(row.get("unsupported", False)) for row in calibration], bool
    )

    def decision(threshold_m: float) -> Mapping[str, Any]:
        predicted = inject_contact_predictions(templates, calibration, threshold_m)
        return METRICS.reduce_two_ply_states(predicted)

    return METRICS.enumerate_threshold_frontier(
        labels,
        clearance,
        decision,
        unsupported=unsupported,
    )


def _contact_sections(
    records: Sequence[Mapping[str, Any]], threshold_m: float
) -> dict[str, Mapping[str, Any]]:
    current = [row for row in records if str(row["level"]) == "current"]
    successor = [row for row in records if str(row["level"]) == "successor"]
    if not current or not successor:
        raise AnalysisError("contact summary requires both current and successor rows")

    def values(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray([_contact(row, "contact row") for row in rows], bool),
            np.asarray([float(row["clearance_m"]) for row in rows], np.float64),
            np.asarray([bool(row.get("unsupported", False)) for row in rows], bool),
        )

    current_label, current_clearance, current_unsupported = values(current)
    successor_label, successor_clearance, successor_unsupported = values(successor)
    summary = METRICS.current_successor_contact_summaries(
        current_labels=current_label,
        current_clearance_m=current_clearance,
        successor_labels=successor_label,
        successor_clearance_m=successor_clearance,
        threshold_m=threshold_m,
        current_unsupported=current_unsupported,
        successor_unsupported=successor_unsupported,
    )
    return {
        "current_contact": summary["current"],
        "successor_contact": summary["successor"],
        "combined_contact": summary["combined"],
    }


def _as_vector(value: Any, location: str, *, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim > 1:
        raise AnalysisError(f"{location} must be scalar or one-dimensional")
    return array.reshape(-1)


def _mean_boolean(value: Any, location: str) -> float:
    array = _as_vector(value, location, dtype=bool)
    if not len(array):
        raise AnalysisError(f"{location} must not be empty")
    return float(array.mean())


def _normalise_link_evidence(
    record: Mapping[str, Any], record_index: int
) -> dict[str, dict[str, Any]]:
    """Return canonical per-link evidence from mapping or vector form.

    Mapping form accepts ``clearance_m`` or
    ``minimum_observed_environment_clearance_m``; ``unsupported`` or
    ``unsupported_swept_volume_fraction``; contact aliases; and the coverage
    fields ``support``/``observation_support``, ``nominal_fov``,
    ``direct_visibility``, and ``point_support_count``.

    Vector form uses ``link_names``, ``per_link_clearance_m`` and optional
    parallel ``per_link_unsupported``, ``per_link_contact``,
    ``per_link_support_fraction``, ``per_link_nominal_fov_fraction``, and
    ``per_link_direct_visibility_fraction`` arrays.
    """

    location = f"records[{record_index}]"
    if record.get("per_link") is not None:
        source = record["per_link"]
        if not isinstance(source, Mapping):
            raise AnalysisError(f"{location}.per_link must be a mapping")
        output: dict[str, dict[str, Any]] = {}
        for raw_name, raw_row in source.items():
            name = str(raw_name)
            if not isinstance(raw_row, Mapping):
                raise AnalysisError(f"{location}.per_link.{name} must be a mapping")
            row = dict(raw_row)
            clearance_source = row.get(
                "clearance_m", row.get("minimum_observed_environment_clearance_m")
            )
            if clearance_source is None:
                raise AnalysisError(f"{location}.per_link.{name} lacks clearance")
            clearance_vector = _as_vector(
                clearance_source, f"{location}.per_link.{name}.clearance_m"
            )
            finite = clearance_vector[np.isfinite(clearance_vector)]
            clearance = float(finite.min()) if len(finite) else math.inf
            support_source = row.get("support", row.get("observation_support"))
            support_fraction = (
                None
                if support_source is None
                else _mean_boolean(support_source, f"{location}.per_link.{name}.support")
            )
            unsupported_source = row.get("unsupported")
            if unsupported_source is not None:
                unsupported = bool(np.asarray(unsupported_source, dtype=bool).any())
            elif "unsupported_swept_volume_fraction" in row:
                unsupported = float(row["unsupported_swept_volume_fraction"]) > 0.0
            else:
                unsupported = not math.isfinite(clearance)
            if "unsupported_swept_volume_fraction" in row:
                unsupported_fraction = _finite_float(
                    row["unsupported_swept_volume_fraction"],
                    f"{location}.per_link.{name}.unsupported_swept_volume_fraction",
                )
            elif support_fraction is not None:
                unsupported_fraction = 1.0 - support_fraction
            else:
                unsupported_fraction = 1.0 if unsupported else 0.0
            if not 0.0 <= unsupported_fraction <= 1.0:
                raise AnalysisError(f"{location}.per_link.{name} unsupported fraction out of range")
            contact_value: bool | None = None
            for key in _CONTACT_ALIASES:
                if key in row:
                    contact_value = bool(row[key])
                    break
            if contact_value is None:
                oracle_link = record.get("oracle_contact_link")
                contact_value = bool(_contact(record, location) and oracle_link == name)
            nominal_source = row.get("nominal_fov", row.get("nominal_fov_inclusion"))
            direct_source = row.get(
                "direct_visibility", row.get("direct_visibility_after_self_occlusion")
            )
            point_count_source = row.get("point_support_count", 0)
            point_count = np.asarray(point_count_source, dtype=np.float64)
            output[name] = {
                "clearance_m": clearance,
                "unsupported": unsupported,
                "oracle_contact": contact_value,
                "body_region": str(row.get("body_region", row.get("collision_region", "UNSPECIFIED"))),
                "support_fraction": support_fraction,
                "unsupported_swept_volume_fraction": unsupported_fraction,
                "nominal_fov_fraction": (
                    None
                    if nominal_source is None
                    else _mean_boolean(nominal_source, f"{location}.per_link.{name}.nominal_fov")
                ),
                "direct_visibility_fraction": (
                    None
                    if direct_source is None
                    else _mean_boolean(direct_source, f"{location}.per_link.{name}.direct_visibility")
                ),
                "mean_point_support_count": float(point_count.mean()),
            }
        return output

    if record.get("link_names") is None or record.get("per_link_clearance_m") is None:
        return {}
    names = [str(value) for value in _sequence(record["link_names"], f"{location}.link_names")]
    clearance = _as_vector(record["per_link_clearance_m"], f"{location}.per_link_clearance_m")
    if len(clearance) != len(names):
        raise AnalysisError(f"{location} per-link clearance/name length mismatch")

    def optional(name: str, default: Any, dtype: Any) -> np.ndarray:
        if record.get(name) is None:
            return np.full(len(names), default, dtype=dtype)
        result = _as_vector(record[name], f"{location}.{name}", dtype=dtype)
        if len(result) != len(names):
            raise AnalysisError(f"{location}.{name} length mismatch")
        return result

    unsupported = optional("per_link_unsupported", False, bool)
    contacts = optional("per_link_contact", False, bool)
    support = optional("per_link_support_fraction", np.nan, np.float64)
    nominal = optional("per_link_nominal_fov_fraction", np.nan, np.float64)
    direct = optional("per_link_direct_visibility_fraction", np.nan, np.float64)
    regions = record.get("body_region_by_link", {})
    return {
        name: {
            "clearance_m": float(clearance[index]),
            "unsupported": bool(unsupported[index]),
            "oracle_contact": bool(contacts[index]),
            "body_region": str(regions.get(name, "UNSPECIFIED")),
            "support_fraction": None if np.isnan(support[index]) else float(support[index]),
            "unsupported_swept_volume_fraction": (
                1.0 - float(support[index])
                if not np.isnan(support[index])
                else float(bool(unsupported[index]))
            ),
            "nominal_fov_fraction": None if np.isnan(nominal[index]) else float(nominal[index]),
            "direct_visibility_fraction": None if np.isnan(direct[index]) else float(direct[index]),
            "mean_point_support_count": 0.0,
        }
        for index, name in enumerate(names)
    }


def _optional_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return None if not values else float(np.mean(values))


def _coverage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unsupported = np.asarray(
        [float(row["unsupported_swept_volume_fraction"]) for row in rows], np.float64
    )
    return {
        "transition_link_rows": len(rows),
        "support_fraction": _optional_mean(rows, "support_fraction"),
        "nominal_fov_fraction": _optional_mean(rows, "nominal_fov_fraction"),
        "direct_visibility_fraction": _optional_mean(rows, "direct_visibility_fraction"),
        "mean_unsupported_swept_volume_fraction": float(unsupported.mean()),
        "p95_unsupported_swept_volume_fraction": float(np.percentile(unsupported, 95)),
        "fully_unsupported_rows": int((unsupported >= 1.0).sum()),
        "mean_point_support_count": _optional_mean(rows, "mean_point_support_count"),
    }


def per_link_and_coverage_summaries(
    records: Sequence[Mapping[str, Any]], threshold_m: float
) -> dict[str, Any]:
    """Compute per-link contact metrics and per-link/body-region coverage."""

    source = validate_transition_records(records)
    normalised: list[tuple[Mapping[str, Any], str, Mapping[str, Any]]] = []
    observed_names: set[str] = set()
    for index, record in enumerate(source):
        links = _normalise_link_evidence(record, index)
        observed_names.update(links)
        normalised.extend((record, name, row) for name, row in links.items())
    if not normalised:
        return {"per_link": {}, "per_region": {}, "coverage": {"available": False}}
    expected_names = observed_names
    for index, record in enumerate(source):
        if set(_normalise_link_evidence(record, index)) != expected_names:
            raise AnalysisError("every transition must contain the same protected-link set")

    grouped_link: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    grouped_region: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for record, name, row in normalised:
        grouped_link[name].append((record, row))
        grouped_region[str(row["body_region"])].append((record, row))

    def summarize(group: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
        unresolved = [
            (record, row)
            for record, row in group
            if bool(_contact(record, "per-link attribution row"))
            and not bool(record.get("oracle_contact_link_resolved", True))
        ]
        resolved = [
            (record, row)
            for record, row in group
            if not (
                bool(_contact(record, "per-link attribution row"))
                and not bool(record.get("oracle_contact_link_resolved", True))
            )
        ]
        contact_records = [
            {
                **dict(record),
                "oracle_contact": bool(row["oracle_contact"]),
                "clearance_m": float(row["clearance_m"]),
                "unsupported": bool(row["unsupported"]),
            }
            for record, row in resolved
        ]
        result = _contact_sections(contact_records, threshold_m)
        result["coverage"] = _coverage_summary([row for _, row in group])
        result["unresolved_oracle_contact_rows_excluded"] = len(unresolved)
        return result

    per_link = {name: summarize(grouped_link[name]) for name in sorted(grouped_link)}
    per_region = {
        name: summarize(grouped_region[name]) for name in sorted(grouped_region)
    }
    return {
        "per_link": per_link,
        "per_region": per_region,
        "coverage": {
            "available": True,
            "protected_links": len(grouped_link),
            "body_regions": len(grouped_region),
            **_coverage_summary([row for _, _, row in normalised]),
            "unresolved_oracle_contact_rows": len(
                {
                    str(record.get("identity", record.get("transition_uid", "")))
                    for record, _, _ in normalised
                    if bool(_contact(record, "coverage attribution row"))
                    and not bool(record.get("oracle_contact_link_resolved", True))
                }
            ),
        },
    }


def summarize_condition_mode(
    records: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]] | None,
    threshold_m: float,
    *,
    role: str = "heldout",
) -> dict[str, Any]:
    """Compute every held-out scalar needed by gate and final reporting."""

    evaluation = _role_records(records, role)
    templates = (
        build_two_ply_states(evaluation)
        if states is None
        else _states_for_records(states, evaluation)
    )
    predicted_states = inject_contact_predictions(templates, evaluation, threshold_m)
    two_ply = METRICS.reduce_two_ply_states(predicted_states)
    contact = _contact_sections(evaluation, threshold_m)

    per_family: dict[str, Any] = {}
    for family in sorted({str(row["family"]) for row in evaluation}):
        family_records = [row for row in evaluation if str(row["family"]) == family]
        family_contact = _contact_sections(family_records, threshold_m)
        family_decision = dict(two_ply["per_family"][family])
        family_decision.pop("per_state", None)
        family_link = per_link_and_coverage_summaries(family_records, threshold_m)
        per_family[family] = {
            **family_contact,
            **family_decision,
            "coverage": family_link["coverage"],
            "per_link": family_link["per_link"],
        }

    link = per_link_and_coverage_summaries(evaluation, threshold_m)
    viability = {
        key: copy.deepcopy(value)
        for key, value in two_ply.items()
        if key not in ("per_state", "per_family", "safe_action_count")
    }
    viability["per_family"] = per_family
    return {
        "threshold_m": float(threshold_m),
        **contact,
        "safe_action_count": copy.deepcopy(two_ply["safe_action_count"]),
        "viability": viability,
        "per_family": per_family,
        "per_link": link["per_link"],
        "per_region": link["per_region"],
        "coverage": link["coverage"],
        "per_state_decisions": copy.deepcopy(two_ply["per_state"]),
    }


def analyze_condition_mode(
    records: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]] | None = None,
    *,
    calibration_role: str = "calibration",
    evaluation_role: str = "heldout",
) -> dict[str, Any]:
    """Calibrate on the frozen calibration role, then evaluate held-out once."""

    source = validate_transition_records(records)
    templates = build_two_ply_states(source) if states is None else list(states)
    frontier = calibrate_threshold(
        source,
        templates,
        role=calibration_role,
    )
    threshold = float(frontier["selected_threshold_m"])
    evaluation = summarize_condition_mode(
        source,
        templates,
        threshold,
        role=evaluation_role,
    )
    return {
        "selected_threshold_m": threshold,
        "calibration": frontier,
        "evaluation": evaluation,
    }


def analyze_condition_modes(
    records_by_condition_mode: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    states: Sequence[Mapping[str, Any]] | None = None,
    *,
    calibration_role: str = "calibration",
    evaluation_role: str = "heldout",
) -> dict[str, dict[str, Any]]:
    """Analyze an arbitrary complete condition/mode mapping without I/O.

    Keys are ``(condition_id, evidence_mode)`` pairs.  Completeness against the
    frozen four-by-two contract is intentionally checked by the caller that
    owns the executable contract; this pure helper is also useful in compact
    deterministic fixtures containing fewer conditions.
    """

    if not isinstance(records_by_condition_mode, Mapping) or not records_by_condition_mode:
        raise AnalysisError("records_by_condition_mode must be a nonempty mapping")
    output: dict[str, dict[str, Any]] = defaultdict(dict)
    for key in sorted(records_by_condition_mode):
        if not isinstance(key, tuple) or len(key) != 2:
            raise AnalysisError("condition/mode keys must be two-tuples")
        condition_id, mode_id = (str(key[0]), str(key[1]))
        output[condition_id][mode_id] = analyze_condition_mode(
            records_by_condition_mode[key],
            states,
            calibration_role=calibration_role,
            evaluation_role=evaluation_role,
        )
    return {condition: dict(modes) for condition, modes in output.items()}


__all__ = [
    "AnalysisError",
    "analyze_condition_mode",
    "analyze_condition_modes",
    "build_two_ply_states",
    "calibrate_threshold",
    "inject_contact_predictions",
    "per_link_and_coverage_summaries",
    "summarize_condition_mode",
    "transition_key",
    "transition_records_from_arrays",
    "validate_transition_records",
]
