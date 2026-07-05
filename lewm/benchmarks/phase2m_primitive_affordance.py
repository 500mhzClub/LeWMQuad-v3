"""Phase 2M source-local first-primitive affordance contracts."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .phase2_data import action_name, source_key
from .phase2d_generation import PHASE2D_PRIMITIVE_NAMES
from .phase2d_training import ACTION_UTILITY_TARGET_VERSION, action_utility_target, image_tensor

PRIMITIVE_AFFORDANCE_TARGET_VERSION = (
    "phase2m_source_local_first_primitive_max_utility_v0"
)


@dataclass(frozen=True)
class PrimitiveAffordanceExample:
    """One source state with a utility target for each first primitive."""

    scene_id: str
    source_index: int
    start_frame: str
    primitive_names: tuple[str, ...]
    utility_targets: tuple[float, ...]
    utility_mask: tuple[bool, ...]
    candidate_rows: int
    valid_utility_rows: int
    valid_primitive_count: int
    oracle_primitive: str | None
    oracle_utility: float | None
    oracle_row_index: int | None
    oracle_sequence: tuple[str, ...]


@dataclass
class Phase2MPrimitiveBatch:
    """Materialized source-image batch with primitive-level utility targets."""

    example_indices: tuple[int, ...]
    examples: tuple[PrimitiveAffordanceExample, ...]
    start_vision: torch.Tensor
    primitive_utility_targets: torch.Tensor
    primitive_utility_mask: torch.Tensor

    def to(self, device: torch.device) -> "Phase2MPrimitiveBatch":
        return Phase2MPrimitiveBatch(
            example_indices=self.example_indices,
            examples=self.examples,
            start_vision=self.start_vision.to(device),
            primitive_utility_targets=self.primitive_utility_targets.to(device),
            primitive_utility_mask=self.primitive_utility_mask.to(device),
        )


def primitive_vocabulary(rows: Sequence[Mapping]) -> tuple[str, ...]:
    """Return the registered primitive order followed by deterministic extras."""

    observed = {
        action_name(dict(row), 0)
        for row in rows
        if len(row.get("active_blocks", ())) >= 1
    }
    if not observed:
        raise ValueError("cannot build primitive vocabulary without action rows")
    ordered = [name for name in PHASE2D_PRIMITIVE_NAMES if name in observed]
    extras = sorted(observed - set(ordered))
    return tuple(ordered + extras)


def _primitive_sequence(row: Mapping) -> tuple[str, ...]:
    return tuple(str(value) for value in row.get("primitive_sequence", ()))


def build_primitive_affordance_examples(
    rows: Sequence[dict],
    *,
    primitive_names: Sequence[str] | None = None,
) -> tuple[PrimitiveAffordanceExample, ...]:
    """Aggregate two-block candidates to source-local first-primitive targets.

    For each source state and first primitive, the target is the maximum valid
    utility over all continuations beginning with that primitive. This converts
    the Phase 2D 81-way sequence target into a 9-way immediate affordance target
    without using validation/test outcomes outside the current split.
    """

    names = (
        primitive_vocabulary(rows)
        if primitive_names is None
        else tuple(str(name) for name in primitive_names)
    )
    if not names:
        raise ValueError("primitive_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("primitive_names must be unique")
    primitive_to_index = {name: index for index, name in enumerate(names)}
    grouped: dict[tuple[str, int], list[tuple[int, dict]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped[source_key(row)].append((row_index, row))

    examples = []
    for key, source_rows in sorted(grouped.items()):
        start_frames = {str(row["start_frame"]) for _row_index, row in source_rows}
        if len(start_frames) != 1:
            raise ValueError(
                "all rows for one source state must share one start_frame: "
                f"{key}"
            )
        best_by_primitive: dict[str, tuple[float, int, tuple[str, ...]]] = {}
        valid_utility_rows = 0
        for row_index, row in source_rows:
            first = action_name(row, 0)
            if first not in primitive_to_index:
                raise ValueError(
                    f"row contains first primitive outside vocabulary: {first}"
                )
            utility, valid = action_utility_target(row)
            if not valid:
                continue
            valid_utility_rows += 1
            previous = best_by_primitive.get(first)
            sequence = _primitive_sequence(row)
            if previous is None or float(utility) > previous[0]:
                best_by_primitive[first] = (float(utility), row_index, sequence)

        targets = [0.0 for _name in names]
        mask = [False for _name in names]
        for primitive, (utility, _row_index, _sequence) in best_by_primitive.items():
            primitive_index = primitive_to_index[primitive]
            targets[primitive_index] = utility
            mask[primitive_index] = True

        oracle_primitive = None
        oracle_utility = None
        oracle_row_index = None
        oracle_sequence: tuple[str, ...] = ()
        if any(mask):
            oracle_index = max(
                (index for index, valid in enumerate(mask) if valid),
                key=lambda index: targets[index],
            )
            oracle_primitive = names[oracle_index]
            oracle_utility, oracle_row_index, oracle_sequence = best_by_primitive[
                oracle_primitive
            ]

        examples.append(
            PrimitiveAffordanceExample(
                scene_id=key[0],
                source_index=int(key[1]),
                start_frame=next(iter(start_frames)),
                primitive_names=names,
                utility_targets=tuple(targets),
                utility_mask=tuple(mask),
                candidate_rows=len(source_rows),
                valid_utility_rows=valid_utility_rows,
                valid_primitive_count=sum(mask),
                oracle_primitive=oracle_primitive,
                oracle_utility=oracle_utility,
                oracle_row_index=oracle_row_index,
                oracle_sequence=oracle_sequence,
            )
        )
    return tuple(examples)


def primitive_affordance_batches(
    example_count: int,
    *,
    source_states_per_batch: int,
    seed: int,
    shuffle: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic source-state batches for Phase 2M examples."""

    if source_states_per_batch < 1:
        raise ValueError("source_states_per_batch must be positive")
    indices = list(range(int(example_count)))
    if shuffle:
        import random

        random.Random(seed).shuffle(indices)
    return tuple(
        tuple(indices[offset : offset + source_states_per_batch])
        for offset in range(0, len(indices), source_states_per_batch)
    )


def materialize_phase2m_primitive_batch(
    examples: Sequence[PrimitiveAffordanceExample],
    indices: Sequence[int],
    *,
    image_size: int = 224,
) -> Phase2MPrimitiveBatch:
    """Build one source-image primitive-affordance training batch."""

    example_indices = tuple(int(index) for index in indices)
    if not example_indices:
        raise ValueError("cannot materialize an empty Phase 2M batch")
    selected = tuple(examples[index] for index in example_indices)
    primitive_names = selected[0].primitive_names
    if any(example.primitive_names != primitive_names for example in selected):
        raise ValueError("all Phase 2M examples in a batch must share vocabulary")
    image_cache: dict[Path, torch.Tensor] = {}

    def cached_image(path: Path) -> torch.Tensor:
        cached = image_cache.get(path)
        if cached is None:
            cached = image_tensor(path, image_size=image_size)
            image_cache[path] = cached
        return cached

    return Phase2MPrimitiveBatch(
        example_indices=example_indices,
        examples=selected,
        start_vision=torch.stack(
            [cached_image(Path(example.start_frame)) for example in selected]
        ),
        primitive_utility_targets=torch.tensor(
            [example.utility_targets for example in selected],
            dtype=torch.float32,
        ),
        primitive_utility_mask=torch.tensor(
            [example.utility_mask for example in selected],
            dtype=torch.bool,
        ),
    )


def phase2m_batch_contract_audit(batch: Phase2MPrimitiveBatch) -> dict:
    """Return compact evidence for one Phase 2M materialized batch."""

    return {
        "schema": "jepa_phase2m_primitive_affordance_batch_contract_v0",
        "examples": len(batch.examples),
        "primitive_count": int(batch.primitive_utility_targets.shape[1]),
        "primitive_utility_targets": int(batch.primitive_utility_mask.sum()),
        "source_states_with_two_or_more_valid_primitives": sum(
            example.valid_primitive_count >= 2 for example in batch.examples
        ),
        "target_version": PRIMITIVE_AFFORDANCE_TARGET_VERSION,
        "source_target_version": ACTION_UTILITY_TARGET_VERSION,
        "all_start_frames_finite": bool(torch.isfinite(batch.start_vision).all()),
    }


def primitive_affordance_dataset_audit(
    examples: Sequence[PrimitiveAffordanceExample],
    *,
    split_name: str,
) -> dict:
    """Summarize Phase 2M source-state label coverage and oracle distribution."""

    valid_examples = [example for example in examples if example.valid_primitive_count >= 2]
    ranges = [
        max(
            value
            for value, valid in zip(
                example.utility_targets,
                example.utility_mask,
                strict=True,
            )
            if valid
        )
        - min(
            value
            for value, valid in zip(
                example.utility_targets,
                example.utility_mask,
                strict=True,
            )
            if valid
        )
        for example in valid_examples
    ]
    oracle_counts = Counter(
        example.oracle_primitive
        for example in valid_examples
        if example.oracle_primitive is not None
    )
    primitive_names = examples[0].primitive_names if examples else ()
    primitive_label_counts = {
        primitive: sum(
            bool(example.utility_mask[index]) for example in examples
        )
        for index, primitive in enumerate(primitive_names)
    }
    return {
        "schema": "jepa_phase2m_primitive_affordance_dataset_audit_v0",
        "split": split_name,
        "target_version": PRIMITIVE_AFFORDANCE_TARGET_VERSION,
        "source_target_version": ACTION_UTILITY_TARGET_VERSION,
        "source_states": len(examples),
        "primitive_names": list(primitive_names),
        "primitive_count": len(primitive_names),
        "valid_primitive_utility_targets": sum(
            example.valid_primitive_count for example in examples
        ),
        "source_states_with_two_or_more_valid_primitives": len(valid_examples),
        "mean_valid_primitives_per_source": (
            sum(example.valid_primitive_count for example in examples) / len(examples)
            if examples
            else 0.0
        ),
        "minimum_valid_primitives_per_source": min(
            (example.valid_primitive_count for example in examples),
            default=0,
        ),
        "maximum_valid_primitives_per_source": max(
            (example.valid_primitive_count for example in examples),
            default=0,
        ),
        "mean_utility_range_per_source": (
            sum(ranges) / len(ranges) if ranges else None
        ),
        "minimum_utility_range_per_source": min(ranges, default=None),
        "maximum_utility_range_per_source": max(ranges, default=None),
        "oracle_primitive_counts": dict(sorted(oracle_counts.items())),
        "primitive_label_counts": dict(sorted(primitive_label_counts.items())),
    }


def primitive_affordance_selection_records(
    examples: Sequence[PrimitiveAffordanceExample],
    primitive_scores: torch.Tensor,
    *,
    seed: int,
    split_name: str,
    scorer_name: str,
) -> list[dict]:
    """Return source-local first-primitive selection diagnostics."""

    scores = primitive_scores.detach().cpu().float()
    if scores.ndim != 2:
        raise ValueError("primitive_scores must have shape (B, primitive_count)")
    if scores.shape[0] != len(examples):
        raise ValueError("primitive_scores row count must match examples")
    records = []
    for index, example in enumerate(examples):
        valid_indices = [
            primitive_index
            for primitive_index, valid in enumerate(example.utility_mask)
            if valid
        ]
        if len(valid_indices) < 2:
            continue
        target_values = torch.tensor(
            [example.utility_targets[primitive_index] for primitive_index in valid_indices],
            dtype=torch.float32,
        )
        predicted_values = scores[index, valid_indices]
        selected_local = int(torch.argmax(predicted_values))
        oracle_local = int(torch.argmax(target_values))
        selected_index = valid_indices[selected_local]
        oracle_index = valid_indices[oracle_local]
        selected_utility = float(target_values[selected_local])
        oracle_utility = float(target_values[oracle_local])
        records.append(
            {
                "schema": "jepa_phase2m_primitive_affordance_selection_v0",
                "seed": int(seed),
                "split": split_name,
                "scorer_name": scorer_name,
                "scene_id": example.scene_id,
                "source_index": int(example.source_index),
                "candidate_rows": int(example.candidate_rows),
                "valid_primitive_count": int(example.valid_primitive_count),
                "selected_primitive": example.primitive_names[selected_index],
                "oracle_primitive": example.primitive_names[oracle_index],
                "oracle_row_index": example.oracle_row_index,
                "oracle_sequence": list(example.oracle_sequence),
                "selected_predicted_utility": float(predicted_values[selected_local]),
                "oracle_predicted_utility": float(predicted_values[oracle_local]),
                "selected_target_utility": selected_utility,
                "oracle_target_utility": oracle_utility,
                "target_utility_regret": oracle_utility - selected_utility,
                "primitive_match": selected_index == oracle_index,
                "uniform_random_primitive_match_rate": 1.0 / len(valid_indices),
            }
        )
    return records


def primitive_affordance_selection_summary(records: Sequence[dict]) -> dict | None:
    """Summarize Phase 2M primitive-affordance selection records."""

    if not records:
        return None
    selected_counts = Counter(str(record["selected_primitive"]) for record in records)
    oracle_counts = Counter(str(record["oracle_primitive"]) for record in records)

    def mean_key(key: str) -> float:
        return sum(float(record[key]) for record in records) / len(records)

    return {
        "schema": "jepa_phase2m_primitive_affordance_selection_summary_v0",
        "source_state_count": len(records),
        "mean_candidate_rows": mean_key("candidate_rows"),
        "mean_valid_primitive_count": mean_key("valid_primitive_count"),
        "primitive_match_rate": mean_key("primitive_match"),
        "mean_target_utility_regret": mean_key("target_utility_regret"),
        "mean_selected_target_utility": mean_key("selected_target_utility"),
        "mean_oracle_target_utility": mean_key("oracle_target_utility"),
        "uniform_random_expected_primitive_match_rate": mean_key(
            "uniform_random_primitive_match_rate"
        ),
        "selected_primitive_counts": dict(sorted(selected_counts.items())),
        "oracle_primitive_counts": dict(sorted(oracle_counts.items())),
        "selected_max_primitive_fraction": max(selected_counts.values()) / len(records),
        "oracle_max_primitive_fraction": max(oracle_counts.values()) / len(records),
    }


def fit_primitive_action_priors(
    examples: Sequence[PrimitiveAffordanceExample],
) -> dict:
    """Fit source-independent first-primitive utility priors."""

    if not examples:
        raise ValueError("cannot fit primitive priors without examples")
    primitive_names = examples[0].primitive_names
    by_primitive: dict[str, list[float]] = {name: [] for name in primitive_names}
    global_values = []
    for example in examples:
        if example.primitive_names != primitive_names:
            raise ValueError("all examples must share one primitive vocabulary")
        for primitive, value, valid in zip(
            primitive_names,
            example.utility_targets,
            example.utility_mask,
            strict=True,
        ):
            if not valid:
                continue
            by_primitive[primitive].append(float(value))
            global_values.append(float(value))
    if not global_values:
        raise ValueError("cannot fit primitive priors without valid utility targets")
    global_mean = sum(global_values) / len(global_values)
    return {
        "schema": "jepa_phase2m_primitive_action_priors_v0",
        "target_version": PRIMITIVE_AFFORDANCE_TARGET_VERSION,
        "source_target_version": ACTION_UTILITY_TARGET_VERSION,
        "primitive_names": list(primitive_names),
        "valid_training_targets": len(global_values),
        "global_mean_utility": global_mean,
        "primitive_mean_utility": {
            primitive: {
                "mean_utility": (
                    sum(values) / len(values) if values else global_mean
                ),
                "count": len(values),
                "seen_in_train": bool(values),
            }
            for primitive, values in by_primitive.items()
        },
    }


def oracle_primitive_class_weights(
    examples: Sequence[PrimitiveAffordanceExample],
    *,
    max_weight: float = 5.0,
) -> tuple[float, ...]:
    """Return mean-one inverse-frequency weights for oracle primitives."""

    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")
    if not examples:
        raise ValueError("cannot compute class weights without examples")
    primitive_names = examples[0].primitive_names
    counts = Counter(
        example.oracle_primitive
        for example in examples
        if example.oracle_primitive is not None
        and example.valid_primitive_count >= 2
    )
    total = sum(counts.values())
    if total == 0:
        raise ValueError("cannot compute class weights without oracle labels")
    seen_classes = sum(1 for count in counts.values() if count > 0)
    raw = []
    for primitive in primitive_names:
        count = counts.get(primitive, 0)
        raw.append(0.0 if count == 0 else min(total / (seen_classes * count), max_weight))
    positive_sum = sum(value for value in raw if value > 0.0)
    positive_count = sum(value > 0.0 for value in raw)
    scale = positive_count / positive_sum if positive_sum > 0.0 else 1.0
    return tuple(value * scale for value in raw)


def evaluate_primitive_action_only_baseline(
    examples: Sequence[PrimitiveAffordanceExample],
    priors: Mapping,
    *,
    split_name: str,
    seed: int,
) -> dict:
    """Evaluate source-independent primitive priors on source-local selection."""

    primitive_names = tuple(str(name) for name in priors["primitive_names"])
    scores = torch.tensor(
        [
            [
                float(
                    priors["primitive_mean_utility"]
                    .get(primitive, {})
                    .get("mean_utility", priors["global_mean_utility"])
                )
                for primitive in primitive_names
            ]
            for _example in examples
        ],
        dtype=torch.float32,
    )
    records = primitive_affordance_selection_records(
        examples,
        scores,
        seed=seed,
        split_name=split_name,
        scorer_name="primitive_action_only_mean_prior",
    )
    return {
        "schema": "jepa_phase2m_primitive_action_only_baseline_v0",
        "split": split_name,
        "target_version": PRIMITIVE_AFFORDANCE_TARGET_VERSION,
        "selection_records": records,
        "selection_summary": primitive_affordance_selection_summary(records),
    }
