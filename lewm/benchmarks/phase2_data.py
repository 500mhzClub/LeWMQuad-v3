"""Auditable data and action-control contracts for Phase 2 spatial JEPA work."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

RowMode = Literal["all", "any_transition", "complete"]

CONFIRMATORY_SPLIT_REQUIREMENTS = {
    "train": {"minimum_scenes": 32, "minimum_source_states_per_scene": 16},
    "validation": {"minimum_scenes": 16, "minimum_source_states_per_scene": 16},
    "test_id": {"minimum_scenes": 16, "minimum_source_states_per_scene": 16},
    "test_hard": {"minimum_scenes": 16, "minimum_source_states_per_scene": 16},
}


def source_key(row: dict) -> tuple[str, int]:
    """Return the split-independent matched-state identifier for one candidate row."""

    return (
        str(row["scene_id"]),
        int(row["source_index"]),
    )


def action_vector(row: dict, step: int) -> tuple[float, ...]:
    """Return one action block as a hashable exact vector."""

    blocks = row["active_blocks"]
    if step < 0 or step >= len(blocks):
        raise IndexError(f"action step {step} outside horizon {len(blocks)}")
    return tuple(float(value) for value in blocks[step])


def action_name(row: dict, step: int) -> str:
    """Return the registered primitive name, or a stable fallback label."""

    names = row.get("primitive_sequence") or ()
    return str(names[step]) if step < len(names) else f"unlabelled_step_{step}"


def is_zero_action(row: dict, step: int, *, atol: float = 1e-8) -> bool:
    """Return whether one action block is numerically zero."""

    return all(abs(value) <= atol for value in action_vector(row, step))


def future_observation_validity(row: dict) -> tuple[bool, ...]:
    """Return validity of each future observation slot from the dataset contract."""

    horizon = len(row["active_blocks"])
    observations = row.get("future_observations")
    if observations is not None:
        if len(observations) != horizon:
            raise ValueError(
                "future_observations and active_blocks horizons differ: "
                f"{len(observations)} != {horizon}"
            )
        return tuple(
            bool(observation.get("observation_valid", False))
            and observation.get("rgb_path") is not None
            for observation in observations
        )
    frames = row.get("future_frames")
    if frames is None or len(frames) != horizon:
        raise ValueError("future_frames must align with active_blocks")
    return tuple(frame is not None for frame in frames)


def transition_validity(row: dict) -> tuple[bool, ...]:
    """Return teacher-forced one-step eligibility for each transition.

    A later transition requires both its current and future observations. Once a
    future observation is invalid, the following teacher-forced transition
    cannot use it as a real current state even if the later target is valid.
    """

    current_valid = True
    transitions = []
    for future_valid in future_observation_validity(row):
        transitions.append(current_valid and future_valid)
        current_valid = future_valid
    return tuple(transitions)


def future_frame_paths(row: dict) -> tuple[str | None, ...]:
    """Return future RGB paths without requiring complete validity."""

    horizon = len(row["active_blocks"])
    observations = row.get("future_observations")
    if observations is not None:
        if len(observations) != horizon:
            raise ValueError("future_observations and active_blocks horizons differ")
        return tuple(
            None if observation.get("rgb_path") is None else str(observation["rgb_path"])
            for observation in observations
        )
    frames = row.get("future_frames")
    if frames is None or len(frames) != horizon:
        raise ValueError("future_frames must align with active_blocks")
    return tuple(None if frame is None else str(frame) for frame in frames)


def materialized_frame_paths(row: dict) -> tuple[tuple[Path, ...], tuple[bool, ...]]:
    """Return loadable frame paths and the aligned one-step transition mask.

    Invalid future slots are replaced with the most recent valid observation so
    tensor construction remains deterministic. The returned transition mask is
    the only authority for prediction loss; substituted frames are never valid
    targets.
    """

    start = Path(str(row["start_frame"]))
    paths = [start]
    previous = start
    for path, valid in zip(
        future_frame_paths(row),
        future_observation_validity(row),
        strict=True,
    ):
        if valid and path is not None:
            previous = Path(path)
        paths.append(previous)
    return tuple(paths), transition_validity(row)


def _row_matches_mode(row: dict, mode: RowMode) -> bool:
    validity = transition_validity(row)
    if mode == "all":
        return True
    if mode == "any_transition":
        return any(validity)
    if mode == "complete":
        return bool(validity) and all(validity)
    raise ValueError(f"unsupported row mode: {mode}")


def load_spatial_future_rows(
    path: Path,
    *,
    mode: RowMode = "all",
    max_rows: int = 0,
) -> tuple[list[dict], dict]:
    """Load Phase 2 rows with explicit row-selection provenance."""

    input_rows = []
    selected = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            try:
                future_observation_validity(row)
                transition_validity(row)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            input_rows.append(row)
            if _row_matches_mode(row, mode):
                selected.append(row)
    before_limit = len(selected)
    if max_rows > 0:
        selected = selected[:max_rows]
    if not selected:
        raise ValueError(f"spatial future dataset has no rows for mode={mode}: {path}")
    audit = {
        "schema": "jepa_phase2_row_load_audit_v0",
        "path": str(path.resolve()),
        "mode": mode,
        "input_candidate_sequences": len(input_rows),
        "selected_before_limit": before_limit,
        "selected_rows": len(selected),
        "excluded_by_mode": len(input_rows) - before_limit,
        "excluded_by_max_rows": before_limit - len(selected),
        "selected_fraction": len(selected) / len(input_rows),
    }
    return selected, audit


def source_grouped_batches(
    rows: Sequence[dict],
    *,
    source_states_per_batch: int,
    seed: int,
    shuffle: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic batches that never split a matched source state."""

    if source_states_per_batch < 1:
        raise ValueError("source_states_per_batch must be positive")
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[source_key(row)].append(index)
    keys = sorted(grouped)
    if shuffle:
        random.Random(seed).shuffle(keys)
    batches = []
    for offset in range(0, len(keys), source_states_per_batch):
        selected_keys = keys[offset : offset + source_states_per_batch]
        batches.append(
            tuple(index for key in selected_keys for index in grouped[key])
        )
    return tuple(batches)


@dataclass(frozen=True)
class HardNegativeIndex:
    """Eligible same-source, non-identical negatives for one action step."""

    step: int
    candidates: dict[int, tuple[int, ...]]
    audit: dict


def build_hard_negative_index(
    rows: Sequence[dict],
    *,
    step: int,
    require_non_hold_positive: bool = True,
    require_valid_positive_transition: bool = True,
) -> HardNegativeIndex:
    """Build exhaustive valid negatives without relying on batch order."""

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if step >= len(row["active_blocks"]):
            continue
        grouped[source_key(row)].append(index)

    candidates: dict[int, tuple[int, ...]] = {}
    exclusion_counts = Counter()
    action_counts = Counter()
    negative_counts = []
    potential_positive_rows = 0
    for positive_index, positive in enumerate(rows):
        if step >= len(positive["active_blocks"]):
            exclusion_counts["step_outside_horizon"] += 1
            continue
        if require_valid_positive_transition and not transition_validity(positive)[step]:
            exclusion_counts["invalid_positive_transition"] += 1
            continue
        if require_non_hold_positive and is_zero_action(positive, step):
            exclusion_counts["zero_positive_action"] += 1
            continue
        potential_positive_rows += 1
        positive_action = action_vector(positive, step)
        unique: dict[tuple[float, ...], int] = {}
        for negative_index in grouped[source_key(positive)]:
            if negative_index == positive_index:
                continue
            negative_action = action_vector(rows[negative_index], step)
            if negative_action == positive_action:
                continue
            unique.setdefault(negative_action, negative_index)
        if not unique:
            exclusion_counts["no_nonidentical_same_source_negative"] += 1
            continue
        selected = tuple(unique.values())
        candidates[positive_index] = selected
        action_counts[action_name(positive, step)] += 1
        negative_counts.append(len(selected))

    audit = {
        "schema": "jepa_phase2_hard_negative_audit_v0",
        "step": step + 1,
        "input_rows": len(rows),
        "source_states": len(grouped),
        "potential_non_hold_valid_positive_rows": potential_positive_rows,
        "eligible_positive_rows": len(candidates),
        "eligible_positive_fraction": len(candidates) / len(rows) if rows else 0.0,
        "eligible_non_hold_valid_coverage": (
            len(candidates) / potential_positive_rows if potential_positive_rows else 0.0
        ),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "eligible_positive_action_counts": dict(sorted(action_counts.items())),
        "total_unique_hard_negatives": sum(negative_counts),
        "minimum_negatives_per_eligible_positive": min(negative_counts, default=0),
        "mean_negatives_per_eligible_positive": (
            sum(negative_counts) / len(negative_counts) if negative_counts else 0.0
        ),
        "maximum_negatives_per_eligible_positive": max(negative_counts, default=0),
        "identical_negative_count": 0,
        "negative_contract": (
            "same source state; unique action vector; action differs at evaluated step"
        ),
    }
    return HardNegativeIndex(step=step, candidates=candidates, audit=audit)


def rolling_action_control_audit(rows: Sequence[dict], *, batch_size: int) -> dict:
    """Reproduce the legacy batch-roll shuffled-action control for auditing."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pairs = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        if len(batch) == 1:
            pairs.append((batch[0], None))
        else:
            pairs.extend((row, batch[(index - 1) % len(batch)]) for index, row in enumerate(batch))
    horizon = max((len(row["active_blocks"]) for row in rows), default=0)
    return {
        "schema": "jepa_phase2_legacy_rolling_control_audit_v0",
        "batch_size": batch_size,
        "pairs": len(pairs),
        "same_source_fraction": _fraction(
            left is not None and right is not None and source_key(left) == source_key(right)
            for left, right in pairs
        ),
        "same_full_sequence_fraction": _fraction(
            left is not None
            and right is not None
            and tuple(left.get("primitive_sequence", ()))
            == tuple(right.get("primitive_sequence", ()))
            for left, right in pairs
        ),
        "same_action_fraction_by_step": [
            _fraction(
                left is not None
                and right is not None
                and step < len(left["active_blocks"])
                and step < len(right["active_blocks"])
                and action_vector(left, step) == action_vector(right, step)
                for left, right in pairs
            )
            for step in range(horizon)
        ],
        "zero_positive_fraction_by_step": [
            _fraction(
                left is not None
                and step < len(left["active_blocks"])
                and is_zero_action(left, step)
                for left, _right in pairs
            )
            for step in range(horizon)
        ],
    }


def _fraction(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _counts_by_step(rows: Sequence[dict], *, valid_only: bool) -> list[dict[str, int]]:
    horizon = max((len(row["active_blocks"]) for row in rows), default=0)
    result = []
    for step in range(horizon):
        counts = Counter()
        for row in rows:
            if step >= len(row["active_blocks"]):
                continue
            if valid_only and not transition_validity(row)[step]:
                continue
            counts[action_name(row, step)] += 1
        result.append(dict(sorted(counts.items())))
    return result


def phase2_dataset_audit(rows: Sequence[dict], *, legacy_batch_size: int = 8) -> dict:
    """Return the registered Stage 0/1 data and control audit for one split."""

    horizon = max((len(row["active_blocks"]) for row in rows), default=0)
    complete = [row for row in rows if all(transition_validity(row))]
    source_counts = Counter(source_key(row) for row in rows)
    family_counts = Counter(str(row["family"]) for row in rows)
    complete_family_counts = Counter(str(row["family"]) for row in complete)
    slot_valid = [
        sum(
            step < len(row["active_blocks"]) and future_observation_validity(row)[step]
            for row in rows
        )
        for step in range(horizon)
    ]
    transition_valid = [
        sum(
            step < len(row["active_blocks"]) and transition_validity(row)[step]
            for row in rows
        )
        for step in range(horizon)
    ]
    source_candidate_counts = list(source_counts.values())
    hard_negatives = [
        build_hard_negative_index(rows, step=step).audit for step in range(horizon)
    ]
    return {
        "schema": "jepa_phase2_dataset_audit_v0",
        "rows": len(rows),
        "scenes": len({str(row["scene_id"]) for row in rows}),
        "scene_ids": sorted({str(row["scene_id"]) for row in rows}),
        "source_states": len(source_counts),
        "horizon": horizon,
        "complete_valid_rows": len(complete),
        "complete_valid_fraction": len(complete) / len(rows) if rows else 0.0,
        "rows_with_any_valid_transition": sum(any(transition_validity(row)) for row in rows),
        "future_slot_valid_counts": slot_valid,
        "transition_valid_counts": transition_valid,
        "family_counts": dict(sorted(family_counts.items())),
        "complete_valid_family_counts": dict(sorted(complete_family_counts.items())),
        "action_counts_by_step": _counts_by_step(rows, valid_only=False),
        "valid_transition_action_counts_by_step": _counts_by_step(rows, valid_only=True),
        "minimum_candidates_per_source": min(source_candidate_counts, default=0),
        "mean_candidates_per_source": (
            sum(source_candidate_counts) / len(source_candidate_counts)
            if source_candidate_counts
            else 0.0
        ),
        "maximum_candidates_per_source": max(source_candidate_counts, default=0),
        "full_81_candidate_sources": sum(count == 81 for count in source_candidate_counts),
        "hard_negative_by_step": hard_negatives,
        "legacy_complete_subset_rolling_control": rolling_action_control_audit(
            complete,
            batch_size=legacy_batch_size,
        ),
    }


def pairwise_split_overlap(named_rows: dict[str, Sequence[dict]]) -> dict:
    """Report scene and source-state overlap for every named split pair."""

    names = sorted(named_rows)
    result = {}
    for left_index, left_name in enumerate(names):
        left_rows = named_rows[left_name]
        left_scenes = {str(row["scene_id"]) for row in left_rows}
        left_sources = {source_key(row) for row in left_rows}
        for right_name in names[left_index + 1 :]:
            right_rows = named_rows[right_name]
            key = f"{left_name}__{right_name}"
            result[key] = {
                "scene_ids": sorted(
                    left_scenes & {str(row["scene_id"]) for row in right_rows}
                ),
                "source_keys": [
                    list(value)
                    for value in sorted(left_sources & {source_key(row) for row in right_rows})
                ],
            }
    return result


def confirmatory_data_gate(
    named_rows: dict[str, Sequence[dict]],
    *,
    lineage_verified: bool = False,
) -> dict:
    """Apply the registered Phase 2D split, balance, and lineage gate."""

    missing_splits = sorted(set(CONFIRMATORY_SPLIT_REQUIREMENTS) - set(named_rows))
    overlap = pairwise_split_overlap(named_rows)
    split_checks = {}
    for name, requirements in CONFIRMATORY_SPLIT_REQUIREMENTS.items():
        rows = named_rows.get(name, ())
        scenes = sorted({str(row["scene_id"]) for row in rows})
        sources_by_scene = Counter(source_key(row) for row in rows)
        source_count_by_scene = Counter(scene_id for scene_id, _source in sources_by_scene)
        grouped_rows: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for row in rows:
            grouped_rows[source_key(row)].append(row)
        full_sequence_counts = {
            key: len(
                {
                    tuple(
                        action_vector(row, step)
                        for step in range(len(row["active_blocks"]))
                    )
                    for row in source_rows
                }
            )
            for key, source_rows in grouped_rows.items()
        }
        first_action_counts = {
            key: len({action_vector(row, 0) for row in source_rows})
            for key, source_rows in grouped_rows.items()
        }
        hard_negative = (
            build_hard_negative_index(rows, step=0).audit if rows else None
        )
        eligible_action_counts = (
            hard_negative["eligible_positive_action_counts"] if hard_negative else {}
        )
        eligible_action_total = sum(eligible_action_counts.values())
        split_checks[name] = {
            "present": name in named_rows,
            "scenes": len(scenes),
            "minimum_scenes_required": requirements["minimum_scenes"],
            "minimum_scene_count_passed": (
                len(scenes) >= requirements["minimum_scenes"]
            ),
            "minimum_source_states_per_scene": min(
                source_count_by_scene.values(),
                default=0,
            ),
            "minimum_source_states_per_scene_required": requirements[
                "minimum_source_states_per_scene"
            ],
            "minimum_source_states_per_scene_passed": bool(source_count_by_scene)
            and min(source_count_by_scene.values())
            >= requirements["minimum_source_states_per_scene"],
            "all_sources_have_full_81_unique_two_block_sequences": bool(grouped_rows)
            and all(
                len(source_rows) == 81
                and full_sequence_counts[key] == 81
                and first_action_counts[key] == 9
                and all(len(row["active_blocks"]) == 2 for row in source_rows)
                for key, source_rows in grouped_rows.items()
            ),
            "eligible_non_hold_hard_negative_coverage": (
                hard_negative["eligible_non_hold_valid_coverage"]
                if hard_negative
                else 0.0
            ),
            "hard_negative_coverage_at_least_70pct": bool(hard_negative)
            and hard_negative["eligible_non_hold_valid_coverage"] >= 0.70,
            "eligible_first_action_minimum_share": (
                min(eligible_action_counts.values()) / eligible_action_total
                if eligible_action_total
                else 0.0
            ),
            "eligible_first_action_minimum_share_at_least_5pct": bool(
                eligible_action_counts
            )
            and min(eligible_action_counts.values()) / eligible_action_total >= 0.05,
        }
        split_checks[name]["passed"] = all(
            (
                split_checks[name]["present"],
                split_checks[name]["minimum_scene_count_passed"],
                split_checks[name]["minimum_source_states_per_scene_passed"],
                split_checks[name][
                    "all_sources_have_full_81_unique_two_block_sequences"
                ],
                split_checks[name]["hard_negative_coverage_at_least_70pct"],
                split_checks[name][
                    "eligible_first_action_minimum_share_at_least_5pct"
                ],
            )
        )
    disjoint = all(
        not value["scene_ids"] and not value["source_keys"]
        for value in overlap.values()
    )
    checks = {
        "all_required_splits_present": not missing_splits,
        "all_split_pairs_scene_and_source_disjoint": disjoint,
        "all_split_requirements_passed": all(
            split_check["passed"] for split_check in split_checks.values()
        ),
        "artifact_and_seed_lineage_verified": bool(lineage_verified),
    }
    return {
        "schema": "jepa_phase2d_confirmatory_data_gate_v0",
        "required_splits": sorted(CONFIRMATORY_SPLIT_REQUIREMENTS),
        "missing_splits": missing_splits,
        "split_checks": split_checks,
        "pairwise_split_overlap": overlap,
        "checks": checks,
        "passed": all(checks.values()),
    }
