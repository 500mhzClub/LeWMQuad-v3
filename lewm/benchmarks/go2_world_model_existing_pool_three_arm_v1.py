"""Pure metrics for the existing-pool Go2 world-model three-arm experiment.

This module accepts bound H6 metadata and already-computed scalar energies.  It
does not discover or open a corpus, image, checkpoint, or runtime artifact.
The runner is responsible for custody-safe access and for binding every value
passed here to its frozen experiment plan.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "lewm_go2_world_model_existing_pool_three_arm_metrics_v1"
OVERLAP_AUDIT_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_overlap_audit_v1"
)
CANDIDATE_ACTION_DERANGEMENT_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_candidate_action_derangement_v1"
)
ACTION_IDENTIFICATION_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_action_identification_v1"
)
DECISION_SCHEMA = "lewm_go2_world_model_existing_pool_three_arm_decision_v1"

ARMS = ("conditioned", "blind", "shuffled")
ROLES = ("train", "val")
REGISTERED_FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
ACTION_VOCABULARY = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
ACTION_COUNT = len(ACTION_VOCABULARY)
H6_LENGTH = 6
CANDIDATE_ACTION_POSITION = 2
TRAIN_PRIMARY_UPDATES = (500, 600, 700)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_LOWER_QUANTILE = 0.05
CONTROL_BOOTSTRAP_SEEDS = {
    "blind": 20_260_801,
    "shuffled": 20_260_802,
    "persistence": 20_260_804,
    "wrong_history": 20_260_805,
}
ACTION_IDENTIFICATION_BOOTSTRAP_SEED = 20_260_803
ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM = (
    "python_random_mt19937_getrandbits52_open01_neg_log1p_shared_family_scene_weights_v1"
)
ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION = (
    "bayesian_positive_weight_cluster_5th_percentile_not_frequentist_coverage"
)
ACTION_IDENTIFICATION_MIN_SUPPORTING_SCENES = 2
RANK_RATIO_MIN = 0.25
RANK_PASS_UPDATE_COUNT = 2


class ThreeArmMetricError(ValueError):
    """A pure metric input violates the frozen experiment contract."""


@dataclass(frozen=True)
class H6MetadataRow:
    """The metadata-only identity needed by this module."""

    index: int | str
    role: str
    family: str
    scene_id: str
    actions: tuple[int, ...]

    @property
    def candidate_action_id(self) -> int:
        return self.actions[CANDIDATE_ACTION_POSITION]


@dataclass(frozen=True)
class CandidateActionDerangement:
    """A row-position permutation used only to replace candidate action a2."""

    row_indices: tuple[int | str, ...]
    donor_positions: tuple[int, ...]
    donor_indices: tuple[int | str, ...]
    factual_candidate_action_ids: tuple[int, ...]
    deranged_candidate_action_ids: tuple[int, ...]
    mapping_sha256: str
    audit: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.audit)


@dataclass(frozen=True)
class PairedLogEnergyComparison:
    """Scene-then-family macro of log(control / conditioned) energy."""

    control_name: str
    row_count: int
    scene_count: int
    family_count: int
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_lower_index: int
    macro_log_advantage: float
    bootstrap_lower_95: float
    positive_family_count: int
    conditioned_energy_by_scene: Mapping[str, float]
    control_energy_by_scene: Mapping[str, float]
    log_advantage_by_scene: Mapping[str, float]
    log_advantage_by_family: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_name": self.control_name,
            "row_count": self.row_count,
            "scene_count": self.scene_count,
            "family_count": self.family_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_lower_index": self.bootstrap_lower_index,
            "advantage_definition": "mean_log_control_energy_over_conditioned_energy",
            "favorable_direction": "positive",
            "macro_log_advantage": self.macro_log_advantage,
            "bootstrap_lower_95": self.bootstrap_lower_95,
            "positive_family_count": self.positive_family_count,
            "conditioned_energy_by_scene": dict(self.conditioned_energy_by_scene),
            "control_energy_by_scene": dict(self.control_energy_by_scene),
            "log_advantage_by_scene": dict(self.log_advantage_by_scene),
            "log_advantage_by_family": dict(self.log_advantage_by_family),
        }


@dataclass(frozen=True)
class FamilyEqualLogEnergyAdvantage:
    """Full-train row-then-family macro of log(control / conditioned)."""

    control_name: str
    row_count: int
    family_count: int
    macro_log_advantage: float
    log_advantage_by_family: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_name": self.control_name,
            "row_count": self.row_count,
            "family_count": self.family_count,
            "advantage_definition": "family_equal_mean_row_log_control_energy_over_conditioned_energy",
            "favorable_direction": "positive",
            "macro_log_advantage": self.macro_log_advantage,
            "log_advantage_by_family": dict(self.log_advantage_by_family),
        }


@dataclass(frozen=True)
class ActionIdentificationSummary:
    """Nine-way factual-a2 identification and hardest-action margin metrics."""

    row_count: int
    scene_count: int
    family_count: int
    action_count: int
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_lower_index: int
    bootstrap_algorithm: str
    bootstrap_interpretation: str
    family_action_supporting_scene_counts: Mapping[str, tuple[int, ...]]
    minimum_family_action_supporting_scene_count: int
    confusion_matrix: tuple[tuple[int, ...], ...]
    factual_action_counts: tuple[int, ...]
    predicted_action_counts: tuple[int, ...]
    row_weighted_accuracy: float
    row_weighted_per_action_recall: tuple[float, ...]
    row_weighted_balanced_accuracy: float
    scene_family_per_action_recall: tuple[float, ...]
    scene_family_balanced_accuracy: float
    balanced_accuracy_bootstrap_lower_95: float
    scene_family_margin_by_action: tuple[float, ...]
    hardest_action_id: int
    hardest_action_margin: float
    hardest_margin_bootstrap_lower_95: float
    exact_tie_row_count: int
    exact_tie_rate: float
    unique_winner_count: int
    unique_winner_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ACTION_IDENTIFICATION_SCHEMA,
            "status": "PASS",
            "row_count": self.row_count,
            "scene_count": self.scene_count,
            "family_count": self.family_count,
            "action_count": self.action_count,
            "prediction_rule": "lowest_action_id_argmin_exact_ties",
            "margin_definition": "minimum_wrong_action_energy_minus_factual_action_energy",
            "favorable_direction": "positive",
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_lower_index": self.bootstrap_lower_index,
            "bootstrap_algorithm": self.bootstrap_algorithm,
            "bootstrap_interpretation": self.bootstrap_interpretation,
            "family_action_supporting_scene_counts": {
                family: list(counts)
                for family, counts in self.family_action_supporting_scene_counts.items()
            },
            "minimum_family_action_supporting_scene_count": (
                self.minimum_family_action_supporting_scene_count
            ),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "factual_action_counts": list(self.factual_action_counts),
            "predicted_action_counts": list(self.predicted_action_counts),
            "row_weighted_accuracy": self.row_weighted_accuracy,
            "row_weighted_per_action_recall": list(
                self.row_weighted_per_action_recall
            ),
            "row_weighted_balanced_accuracy": self.row_weighted_balanced_accuracy,
            "scene_family_per_action_recall": list(
                self.scene_family_per_action_recall
            ),
            "scene_family_balanced_accuracy": self.scene_family_balanced_accuracy,
            "balanced_accuracy_bootstrap_lower_95": (
                self.balanced_accuracy_bootstrap_lower_95
            ),
            "scene_family_margin_by_action": list(
                self.scene_family_margin_by_action
            ),
            "hardest_action_id": self.hardest_action_id,
            "hardest_action_margin": self.hardest_action_margin,
            "hardest_margin_bootstrap_lower_95": (
                self.hardest_margin_bootstrap_lower_95
            ),
            "exact_tie_row_count": self.exact_tie_row_count,
            "exact_tie_rate": self.exact_tie_rate,
            "unique_winner_count": self.unique_winner_count,
            "unique_winner_accuracy": self.unique_winner_accuracy,
        }


@dataclass(frozen=True)
class ThreeArmDecision:
    status: str
    passed: bool
    checks: Mapping[str, bool]
    failed_checks: tuple[str, ...]
    localization_stage: str
    observed: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DECISION_SCHEMA,
            "status": self.status,
            "passed": self.passed,
            "localization_stage": self.localization_stage,
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "observed": dict(self.observed),
        }


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ThreeArmMetricError("value is not canonical finite JSON") from error


def _field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        if name not in row:
            raise ThreeArmMetricError(f"H6 row is missing {name!r}")
        return row[name]
    if not hasattr(row, name):
        raise ThreeArmMetricError(f"H6 row is missing {name!r}")
    return getattr(row, name)


def _action_id(value: Any) -> int:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        result = int(value)
        if 0 <= result < ACTION_COUNT:
            return result
    if type(value) is str and value in ACTION_VOCABULARY:
        return ACTION_VOCABULARY.index(value)
    raise ThreeArmMetricError("every H6 action must be a registered name or ID")


def _rows_payload(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        try:
            value = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ThreeArmMetricError("JSON row input is invalid") from error
    if isinstance(value, Mapping):
        if "rows" not in value:
            raise ThreeArmMetricError("JSON row document must contain 'rows'")
        value = value["rows"]
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ThreeArmMetricError("H6 input must be a row sequence or JSON document")
    rows = tuple(value)
    if not rows:
        raise ThreeArmMetricError("H6 row population is empty")
    return rows


def normalize_h6_metadata_rows(value: Any) -> tuple[H6MetadataRow, ...]:
    """Normalize mapping/dataclass rows without performing filesystem access."""

    normalized: list[H6MetadataRow] = []
    identities: set[tuple[str, int | str]] = set()
    for raw in _rows_payload(value):
        index = _field(raw, "index")
        if not (
            (isinstance(index, (int, np.integer)) and not isinstance(index, bool) and index >= 0)
            or (type(index) is str and bool(index))
        ):
            raise ThreeArmMetricError("H6 row index must be nonnegative int or string")
        if isinstance(index, np.integer):
            index = int(index)
        role = _field(raw, "role")
        family = _field(raw, "family")
        scene = _field(raw, "scene_id")
        if role not in ROLES:
            raise ThreeArmMetricError("H6 role must be 'train' or 'val'")
        if family not in REGISTERED_FAMILIES:
            raise ThreeArmMetricError("H6 family is unregistered")
        if type(scene) is not str or not scene:
            raise ThreeArmMetricError("H6 scene_id must be a nonempty string")
        actions_raw = _field(raw, "actions")
        if not isinstance(actions_raw, Sequence) or isinstance(
            actions_raw, (str, bytes, bytearray)
        ) or len(actions_raw) != H6_LENGTH:
            raise ThreeArmMetricError("H6 actions must have exact length six")
        actions = tuple(_action_id(item) for item in actions_raw)
        identity = (role, index)
        if identity in identities:
            raise ThreeArmMetricError("H6 row identities must be unique")
        identities.add(identity)
        normalized.append(H6MetadataRow(index, role, family, scene, actions))
    return tuple(normalized)


def _entropy_bits(values: Sequence[Any]) -> float:
    counts = Counter(values)
    total = len(values)
    return -math.fsum(
        count / total * math.log2(count / total) for count in counts.values()
    )


def _mutual_information_bits(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) != len(right) or not left:
        raise ThreeArmMetricError("mutual-information inputs are invalid")
    joint = Counter(zip(left, right, strict=True))
    left_counts = Counter(left)
    right_counts = Counter(right)
    total = len(left)
    return math.fsum(
        count / total
        * math.log2(count * total / (left_counts[a] * right_counts[b]))
        for (a, b), count in joint.items()
    )


def _count_vector(values: Sequence[int]) -> list[int]:
    counts = Counter(values)
    return [counts[index] for index in range(ACTION_COUNT)]


def audit_h6_metadata_overlap(value: Any) -> dict[str, Any]:
    """Report metadata-only entropy, MI, support, and split-overlap facts."""

    rows = normalize_h6_metadata_rows(value)
    if set(row.role for row in rows) != set(ROLES):
        raise ThreeArmMetricError("overlap audit requires both train and val rows")
    scene_families: dict[tuple[str, str], str] = {}
    for row in rows:
        previous = scene_families.setdefault((row.role, row.scene_id), row.family)
        if previous != row.family:
            raise ThreeArmMetricError("one role/scene cannot belong to multiple families")
    by_role = {role: tuple(row for row in rows if row.role == role) for role in ROLES}
    scenes = {role: {row.scene_id for row in by_role[role]} for role in ROLES}
    role_overlap = sorted(scenes["train"] & scenes["val"])

    entropies: dict[str, Any] = {}
    mutual_information: dict[str, Any] = {}
    scene_diagnostics: dict[str, Any] = {}
    for role in ROLES:
        role_rows = by_role[role]
        position = {}
        for action_position in range(H6_LENGTH):
            values = [row.actions[action_position] for row in role_rows]
            position[f"a{action_position}"] = {
                "counts": _count_vector(values),
                "entropy_bits": _entropy_bits(values),
                "normalized_entropy": _entropy_bits(values) / math.log2(ACTION_COUNT),
            }
        entropies[role] = {
            "by_position": position,
            "candidate_a2_by_family": {
                family: {
                    "row_count": len(family_values),
                    "counts": _count_vector(family_values),
                    "entropy_bits": _entropy_bits(family_values),
                }
                for family in REGISTERED_FAMILIES
                if (
                    family_values := [
                        row.candidate_action_id
                        for row in role_rows
                        if row.family == family
                    ]
                )
            },
        }
        a2 = [row.candidate_action_id for row in role_rows]
        mutual_information[role] = {
            "candidate_a2_with_a1_bits": _mutual_information_bits(
                a2, [row.actions[1] for row in role_rows]
            ),
            "candidate_a2_with_history_a0_a1_bits": _mutual_information_bits(
                a2, [(row.actions[0], row.actions[1]) for row in role_rows]
            ),
            "candidate_a2_with_family_bits": _mutual_information_bits(
                a2, [row.family for row in role_rows]
            ),
            "candidate_a2_with_scene_bits": _mutual_information_bits(
                a2, [row.scene_id for row in role_rows]
            ),
        }
        per_scene_entropy = []
        full_support = 0
        for scene in sorted(scenes[role]):
            values = [
                row.candidate_action_id for row in role_rows if row.scene_id == scene
            ]
            per_scene_entropy.append(_entropy_bits(values))
            full_support += len(set(values)) == ACTION_COUNT
        scene_diagnostics[role] = {
            "scene_count": len(per_scene_entropy),
            "full_candidate_action_support_scene_count": full_support,
            "macro_candidate_a2_entropy_bits": math.fsum(per_scene_entropy)
            / len(per_scene_entropy),
            "minimum_candidate_a2_entropy_bits": min(per_scene_entropy),
        }

    train_rows = by_role["train"]
    actions_by_position = {
        position: {row.actions[position] for row in train_rows}
        for position in range(CANDIDATE_ACTION_POSITION + 1)
    }
    pairs_by_position = {
        position: {
            (row.actions[position], row.actions[position + 1]) for row in train_rows
        }
        for position in range(CANDIDATE_ACTION_POSITION)
    }
    triples = {tuple(row.actions[:3]) for row in train_rows}
    expected_actions = set(range(ACTION_COUNT))
    expected_pairs = {
        (left, right) for left in range(ACTION_COUNT) for right in range(ACTION_COUNT)
    }
    expected_triples = {
        (first, second, third)
        for first in range(ACTION_COUNT)
        for second in range(ACTION_COUNT)
        for third in range(ACTION_COUNT)
    }
    checks = {
        "role_scene_disjointness": not role_overlap,
        "train_all_actions_supported": all(
            values == expected_actions for values in actions_by_position.values()
        ),
        "train_all_ordered_pairs_supported": all(
            values == expected_pairs for values in pairs_by_position.values()
        ),
    }
    diagnostic_checks = {
        "train_all_ordered_triples_supported": triples == expected_triples,
    }
    passed = all(checks.values())
    return {
        "schema": OVERLAP_AUDIT_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "row_count": len(rows),
        "role_row_counts": {role: len(by_role[role]) for role in ROLES},
        "role_scene_counts": {role: len(scenes[role]) for role in ROLES},
        "checks": checks,
        "failed_checks": [name for name, result in checks.items() if not result],
        "diagnostic_checks": diagnostic_checks,
        "failed_diagnostic_checks": [
            name for name, result in diagnostic_checks.items() if not result
        ],
        "role_scene_overlap_count": len(role_overlap),
        "role_scene_overlap": role_overlap,
        "train_support": {
            "visible_action_positions": [0, 1, 2],
            "action_count": min(len(values) for values in actions_by_position.values()),
            "action_count_by_position": {
                f"a{position}": len(values)
                for position, values in actions_by_position.items()
            },
            "ordered_pair_count": min(
                len(values) for values in pairs_by_position.values()
            ),
            "ordered_pair_count_by_position": {
                f"a{position}_a{position + 1}": len(values)
                for position, values in pairs_by_position.items()
            },
            "ordered_triple_count": len(triples),
            "missing_action_ids_by_position": {
                f"a{position}": sorted(expected_actions - values)
                for position, values in actions_by_position.items()
            },
            "missing_action_ids": sorted(
                set().union(
                    *(expected_actions - values for values in actions_by_position.values())
                )
            ),
            "missing_ordered_pairs_by_position": {
                f"a{position}_a{position + 1}": [
                    list(item) for item in sorted(expected_pairs - values)
                ]
                for position, values in pairs_by_position.items()
            },
            "missing_ordered_pairs": [
                list(item)
                for item in sorted(
                    set().union(
                        *(expected_pairs - values for values in pairs_by_position.values())
                    )
                )
            ],
            "missing_ordered_triples": [
                list(item) for item in sorted(expected_triples - triples)
            ],
        },
        "entropy": entropies,
        "mutual_information_bits": mutual_information,
        "scene_diagnostics": scene_diagnostics,
        "gate_scope": (
            "role_scene_disjointness_and_visible_action_and_adjacent_pair_support_only; "
            "triple_support_entropy_and_mutual_information_are_diagnostic"
        ),
    }


def _identity_token(row: H6MetadataRow) -> str:
    return json.dumps(
        [row.role, row.family, row.scene_id, row.index],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _rank_digest(namespace: str, row: H6MetadataRow) -> str:
    return hashlib.sha256(
        f"{namespace}\0{_identity_token(row)}".encode("utf-8")
    ).hexdigest()


def _exact_compatible_donor_map(
    rows: tuple[H6MetadataRow, ...], positions: Sequence[int]
) -> dict[int, int]:
    """Return an exact deterministic matching for the dense complement graph."""

    scene_counts = Counter(rows[index].scene_id for index in positions)
    action_counts = Counter(rows[index].candidate_action_id for index in positions)
    recipients = sorted(
        positions,
        key=lambda index: (
            -(
                scene_counts[rows[index].scene_id]
                + action_counts[rows[index].candidate_action_id]
            ),
            _rank_digest("exact-recipient", rows[index]),
            _identity_token(rows[index]),
        ),
    )
    donors = sorted(
        positions,
        key=lambda index: (
            _rank_digest("exact-donor", rows[index]),
            _identity_token(rows[index]),
        ),
    )
    neighbor_cache: dict[tuple[str, int], tuple[int, ...]] = {}

    def neighbors(left: int) -> tuple[int, ...]:
        row = rows[recipients[left]]
        key = (row.scene_id, row.candidate_action_id)
        if key not in neighbor_cache:
            neighbor_cache[key] = tuple(
                right
                for right, donor in enumerate(donors)
                if rows[donor].scene_id != row.scene_id
                and rows[donor].candidate_action_id != row.candidate_action_id
            )
        return neighbor_cache[key]

    count = len(recipients)
    left_to_right = [-1] * count
    right_to_left = [-1] * count
    distance = [-1] * count

    def breadth_first() -> bool:
        queue: deque[int] = deque()
        for left in range(count):
            if left_to_right[left] < 0:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = -1
        found = False
        while queue:
            left = queue.popleft()
            for right in neighbors(left):
                mate = right_to_left[right]
                if mate < 0:
                    found = True
                elif distance[mate] < 0:
                    distance[mate] = distance[left] + 1
                    queue.append(mate)
        return found

    def depth_first(left: int) -> bool:
        for right in neighbors(left):
            mate = right_to_left[right]
            if mate < 0 or (
                distance[mate] == distance[left] + 1 and depth_first(mate)
            ):
                left_to_right[left] = right
                right_to_left[right] = left
                return True
        distance[left] = -1
        return False

    matched = 0
    while breadth_first():
        progress = 0
        for left in range(count):
            if left_to_right[left] < 0 and depth_first(left):
                progress += 1
        if progress == 0:
            break
        matched += progress
    if matched != count or any(right < 0 for right in left_to_right):
        raise ThreeArmMetricError(
            "role/family population has no different-scene, different-a2 donor bijection"
        )
    return {
        recipients[left]: donors[right]
        for left, right in enumerate(left_to_right)
    }


def build_candidate_action_derangement(value: Any) -> CandidateActionDerangement:
    """Build a deterministic role/family-local a2-only donor bijection.

    Donors always have a different scene and different factual a2.  Actions
    a0/a1 and a3..a5 are intentionally outside this return value and must not
    be changed by consumers.
    """

    rows = normalize_h6_metadata_rows(value)
    donor_positions = [-1] * len(rows)
    group_offsets: dict[str, int | None] = {}
    group_methods: dict[str, str] = {}
    groups = sorted({(row.role, row.family) for row in rows})
    for role, family in groups:
        positions = [
            index
            for index, row in enumerate(rows)
            if row.role == role and row.family == family
        ]
        if len(positions) < 2:
            raise ThreeArmMetricError("every role/family shuffle group needs two rows")
        recipients = sorted(
            positions,
            key=lambda index: (
                rows[index].candidate_action_id,
                _rank_digest("recipient", rows[index]),
                _identity_token(rows[index]),
            ),
        )
        donors = sorted(
            positions,
            key=lambda index: (
                rows[index].candidate_action_id,
                _rank_digest("donor", rows[index]),
                _identity_token(rows[index]),
            ),
        )
        selected: tuple[int, ...] | None = None
        selected_offset: int | None = None
        for offset in range(len(donors)):
            rotated = tuple(donors[(index + offset) % len(donors)] for index in range(len(donors)))
            if all(
                rows[recipient].scene_id != rows[donor].scene_id
                and rows[recipient].candidate_action_id
                != rows[donor].candidate_action_id
                for recipient, donor in zip(recipients, rotated, strict=True)
            ):
                selected = rotated
                selected_offset = offset
                break
        group_name = f"{role}:{family}"
        if selected is not None and selected_offset is not None:
            for recipient, donor in zip(recipients, selected, strict=True):
                donor_positions[recipient] = donor
            group_offsets[group_name] = selected_offset
            group_methods[group_name] = "dual_hash_ranked_cyclic_search"
        else:
            exact = _exact_compatible_donor_map(rows, positions)
            for recipient, donor in exact.items():
                donor_positions[recipient] = donor
            group_offsets[group_name] = None
            group_methods[group_name] = "exact_hopcroft_karp_dense_complement"

    if any(position < 0 for position in donor_positions):
        raise AssertionError("internal derangement accounting failed")
    donor_rows = tuple(rows[position] for position in donor_positions)
    factual = tuple(row.candidate_action_id for row in rows)
    deranged = tuple(row.candidate_action_id for row in donor_rows)
    checks = {
        "donor_map_is_global_bijection": sorted(donor_positions) == list(range(len(rows))),
        "donor_identity_zero_fixed_points": all(
            donor != index for index, donor in enumerate(donor_positions)
        ),
        "different_scene_donors": all(
            row.scene_id != donor.scene_id
            for row, donor in zip(rows, donor_rows, strict=True)
        ),
        "candidate_a2_zero_fixed_points": all(
            left != right for left, right in zip(factual, deranged, strict=True)
        ),
        "role_family_action_marginals_exact": all(
            Counter(
                rows[index].candidate_action_id
                for index in range(len(rows))
                if rows[index].role == role and rows[index].family == family
            )
            == Counter(
                donor_rows[index].candidate_action_id
                for index in range(len(rows))
                if rows[index].role == role and rows[index].family == family
            )
            for role, family in groups
        ),
    }
    if not all(checks.values()):
        raise AssertionError("internal derangement invariant failed")
    mapping_rows = [
        {
            "row_position": index,
            "row_index": row.index,
            "role": row.role,
            "family": row.family,
            "scene_id": row.scene_id,
            "factual_candidate_action_id": row.candidate_action_id,
            "donor_position": donor_positions[index],
            "donor_index": donor_rows[index].index,
            "donor_scene_id": donor_rows[index].scene_id,
            "deranged_candidate_action_id": donor_rows[index].candidate_action_id,
        }
        for index, row in enumerate(rows)
    ]
    mapping_sha256 = hashlib.sha256(_canonical_json_bytes(mapping_rows)).hexdigest()
    audit = {
        "schema": CANDIDATE_ACTION_DERANGEMENT_SCHEMA,
        "status": "PASS",
        "passed": True,
        "algorithm": "role_family_local_cyclic_then_exact_bipartite_derangement_v1",
        "candidate_action_position": CANDIDATE_ACTION_POSITION,
        "changed_action_positions": [CANDIDATE_ACTION_POSITION],
        "row_count": len(rows),
        "role_family_group_count": len(groups),
        "group_selected_offsets": group_offsets,
        "group_methods": group_methods,
        "mapping_sha256": mapping_sha256,
        "checks": checks,
        "fixed_donor_identity_count": 0,
        "same_scene_donor_count": 0,
        "fixed_candidate_action_count": 0,
        "mapping_rows": mapping_rows,
    }
    return CandidateActionDerangement(
        row_indices=tuple(row.index for row in rows),
        donor_positions=tuple(donor_positions),
        donor_indices=tuple(row.index for row in donor_rows),
        factual_candidate_action_ids=factual,
        deranged_candidate_action_ids=deranged,
        mapping_sha256=mapping_sha256,
        audit=audit,
    )


def _energy_vector(value: Any, *, name: str, count: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ThreeArmMetricError(f"{name} must be a numeric vector") from error
    if result.size < 1 or (count is not None and result.size != count):
        raise ThreeArmMetricError(f"{name} row count changed")
    if not np.isfinite(result).all() or bool((result <= 0.0).any()):
        raise ThreeArmMetricError(f"{name} must contain finite strictly positive energies")
    return result


def _metric_metadata(
    scene_ids: Sequence[str], family_ids: Sequence[str], row_count: int
) -> tuple[tuple[str, ...], dict[str, str]]:
    scenes = tuple(scene_ids)
    families = tuple(family_ids)
    if len(scenes) != row_count or len(families) != row_count:
        raise ThreeArmMetricError("scene/family metadata row count changed")
    if any(type(value) is not str or not value for value in scenes + families):
        raise ThreeArmMetricError("scene/family identifiers must be nonempty strings")
    if set(families) != set(REGISTERED_FAMILIES):
        raise ThreeArmMetricError("metrics require exactly the eight registered families")
    scene_family: dict[str, str] = {}
    for scene, family in zip(scenes, families, strict=True):
        if family not in REGISTERED_FAMILIES:
            raise ThreeArmMetricError("metric family is unregistered")
        previous = scene_family.setdefault(scene, family)
        if previous != family:
            raise ThreeArmMetricError("one scene belongs to multiple families")
    return scenes, scene_family


def _bootstrap_lower(values: list[float]) -> tuple[int, float]:
    index = math.floor(BOOTSTRAP_LOWER_QUANTILE * len(values))
    return index, sorted(values)[index]


def _strict_positive_exponential_weight_from_52_bits(bits: int) -> float:
    if type(bits) is not int or not 0 <= bits < 2**52:
        raise ThreeArmMetricError("Bayesian cluster weight bits left the 52-bit range")
    uniform_open = (bits + 1) / (2**52 + 1)
    weight = -math.log1p(-uniform_open)
    if not 0.0 < uniform_open < 1.0 or not math.isfinite(weight) or weight <= 0.0:
        raise ThreeArmMetricError("Bayesian cluster weight construction is invalid")
    return weight


def _paired_log_values(conditioned: np.ndarray, control: np.ndarray) -> np.ndarray:
    values = np.log(control) - np.log(conditioned)
    if not np.isfinite(values).all():
        raise ThreeArmMetricError("paired log-energy advantages are nonfinite")
    return values


def family_equal_paired_log_energy_advantage(
    conditioned_energy: Any,
    control_energy: Any,
    family_ids: Sequence[str],
    *,
    control_name: str,
) -> FamilyEqualLogEnergyAdvantage:
    """Compute the frozen full-train row-then-family point advantage."""

    if control_name not in {"blind", "shuffled"}:
        raise ThreeArmMetricError("full-train control must be blind or shuffled")
    conditioned = _energy_vector(conditioned_energy, name="conditioned_energy")
    control = _energy_vector(
        control_energy, name="control_energy", count=conditioned.size
    )
    families = tuple(family_ids)
    if len(families) != conditioned.size or any(
        type(family) is not str or family not in REGISTERED_FAMILIES
        for family in families
    ):
        raise ThreeArmMetricError("full-train family metadata changed")
    if set(families) != set(REGISTERED_FAMILIES):
        raise ThreeArmMetricError("full-train metric requires all registered families")
    row_log = _paired_log_values(conditioned, control)
    by_family: dict[str, float] = {}
    for family in REGISTERED_FAMILIES:
        indices = [index for index, value in enumerate(families) if value == family]
        by_family[family] = float(row_log[indices].mean())
    return FamilyEqualLogEnergyAdvantage(
        control_name=control_name,
        row_count=int(conditioned.size),
        family_count=len(REGISTERED_FAMILIES),
        macro_log_advantage=math.fsum(by_family.values())
        / len(REGISTERED_FAMILIES),
        log_advantage_by_family=by_family,
    )


def paired_log_energy_comparison(
    conditioned_energy: Any,
    control_energy: Any,
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    control_name: str,
) -> PairedLogEnergyComparison:
    """Compare paired rows using mean log(control / conditioned) energy."""

    if control_name not in CONTROL_BOOTSTRAP_SEEDS:
        raise ThreeArmMetricError("control_name is not registered")
    conditioned = _energy_vector(conditioned_energy, name="conditioned_energy")
    control = _energy_vector(
        control_energy, name="control_energy", count=conditioned.size
    )
    scenes, scene_family = _metric_metadata(scene_ids, family_ids, conditioned.size)
    row_log = _paired_log_values(conditioned, control)
    conditioned_scene: dict[str, float] = {}
    control_scene: dict[str, float] = {}
    advantage_scene: dict[str, float] = {}
    for scene in sorted(scene_family):
        indices = [index for index, name in enumerate(scenes) if name == scene]
        conditioned_scene[scene] = float(conditioned[indices].mean())
        control_scene[scene] = float(control[indices].mean())
        advantage_scene[scene] = float(row_log[indices].mean())
    family_scenes = {
        family: tuple(
            scene for scene in sorted(scene_family) if scene_family[scene] == family
        )
        for family in REGISTERED_FAMILIES
    }
    if any(not names for names in family_scenes.values()):
        raise ThreeArmMetricError("every family requires at least one scene")
    advantage_family = {
        family: math.fsum(advantage_scene[scene] for scene in names) / len(names)
        for family, names in family_scenes.items()
    }
    macro = math.fsum(advantage_family.values()) / len(REGISTERED_FAMILIES)
    rng = random.Random(CONTROL_BOOTSTRAP_SEEDS[control_name])
    draws: list[float] = []
    for _replicate in range(BOOTSTRAP_REPLICATES):
        family_values = []
        for family in REGISTERED_FAMILIES:
            names = family_scenes[family]
            family_values.append(
                math.fsum(
                    advantage_scene[names[rng.randrange(len(names))]]
                    for _ in range(len(names))
                )
                / len(names)
            )
        draws.append(math.fsum(family_values) / len(family_values))
    lower_index, lower = _bootstrap_lower(draws)
    return PairedLogEnergyComparison(
        control_name=control_name,
        row_count=int(conditioned.size),
        scene_count=len(scene_family),
        family_count=len(REGISTERED_FAMILIES),
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=CONTROL_BOOTSTRAP_SEEDS[control_name],
        bootstrap_lower_index=lower_index,
        macro_log_advantage=macro,
        bootstrap_lower_95=lower,
        positive_family_count=sum(value > 0.0 for value in advantage_family.values()),
        conditioned_energy_by_scene=conditioned_scene,
        control_energy_by_scene=control_scene,
        log_advantage_by_scene=advantage_scene,
        log_advantage_by_family=advantage_family,
    )


def summarize_nine_way_action_identification(
    candidate_energies: Any,
    factual_action_ids: Sequence[int | str],
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
) -> ActionIdentificationSummary:
    """Score all nine a2 candidates using deterministic lowest-ID argmin."""

    try:
        energies = np.asarray(candidate_energies, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ThreeArmMetricError("candidate_energies must be numeric") from error
    if energies.ndim != 2 or energies.shape[0] < 1 or energies.shape[1] != ACTION_COUNT:
        raise ThreeArmMetricError("candidate_energies must have shape (N,9)")
    if not np.isfinite(energies).all() or bool((energies < 0.0).any()):
        raise ThreeArmMetricError("candidate energies must be finite and nonnegative")
    factual = np.asarray([_action_id(value) for value in factual_action_ids], dtype=np.int64)
    if factual.shape != (energies.shape[0],):
        raise ThreeArmMetricError("factual action row count changed")
    scenes, scene_family = _metric_metadata(scene_ids, family_ids, energies.shape[0])
    predictions = np.argmin(energies, axis=1)
    minimum = energies.min(axis=1)
    tie_sizes = np.equal(energies, minimum[:, None]).sum(axis=1)
    correct = predictions == factual
    factual_energy = energies[np.arange(energies.shape[0]), factual]
    wrong = energies.copy()
    wrong[np.arange(energies.shape[0]), factual] = math.inf
    margin = wrong.min(axis=1) - factual_energy

    confusion = np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.int64)
    for truth, prediction in zip(factual, predictions, strict=True):
        confusion[truth, prediction] += 1
    factual_counts = confusion.sum(axis=1)
    if bool((factual_counts == 0).any()):
        raise ThreeArmMetricError("nine-way panel must support every factual action")
    row_recalls = np.diag(confusion) / factual_counts

    cell_correct: dict[tuple[str, int], float] = {}
    cell_margin: dict[tuple[str, int], float] = {}
    for scene in sorted(scene_family):
        for action in range(ACTION_COUNT):
            indices = [
                index
                for index, name in enumerate(scenes)
                if name == scene and factual[index] == action
            ]
            if indices:
                cell_correct[(scene, action)] = float(correct[indices].mean())
                cell_margin[(scene, action)] = float(margin[indices].mean())
    family_scenes = {
        family: tuple(
            scene for scene in sorted(scene_family) if scene_family[scene] == family
        )
        for family in REGISTERED_FAMILIES
    }
    family_action_supporting_scene_counts = {
        family: tuple(
            sum((scene, action) in cell_correct for scene in family_scenes[family])
            for action in range(ACTION_COUNT)
        )
        for family in REGISTERED_FAMILIES
    }
    missing_family_actions = [
        (family, action)
        for family in REGISTERED_FAMILIES
        for action in range(ACTION_COUNT)
        if family_action_supporting_scene_counts[family][action] == 0
    ]
    if missing_family_actions:
        raise ThreeArmMetricError(
            "nine-way panel must support all eight-by-nine family/action cells"
        )
    minimum_supporting_scene_count = min(
        min(counts) for counts in family_action_supporting_scene_counts.values()
    )
    if minimum_supporting_scene_count < ACTION_IDENTIFICATION_MIN_SUPPORTING_SCENES:
        raise ThreeArmMetricError(
            "every family/action cell requires at least two supporting scenes"
        )

    def action_macro(cells: Mapping[tuple[str, int], float], action: int) -> float:
        family_cells = []
        for family in REGISTERED_FAMILIES:
            values = [
                cells[(scene, action)]
                for scene in family_scenes[family]
                if (scene, action) in cells
            ]
            if not values:
                raise AssertionError("validated family/action cell disappeared")
            family_cells.append(math.fsum(values) / len(values))
        return math.fsum(family_cells) / len(family_cells)

    sf_recalls = tuple(action_macro(cell_correct, action) for action in range(ACTION_COUNT))
    sf_margins = tuple(action_macro(cell_margin, action) for action in range(ACTION_COUNT))
    sf_balanced = math.fsum(sf_recalls) / ACTION_COUNT

    rng = random.Random(ACTION_IDENTIFICATION_BOOTSTRAP_SEED)
    ba_draws: list[float] = []
    margin_draws: list[float] = []
    for _replicate in range(BOOTSTRAP_REPLICATES):
        scene_weights_by_family: dict[str, dict[str, float]] = {}
        for family in REGISTERED_FAMILIES:
            weights = {
                scene: _strict_positive_exponential_weight_from_52_bits(
                    rng.getrandbits(52)
                )
                for scene in family_scenes[family]
            }
            scene_weights_by_family[family] = weights
        replicate_recalls = []
        replicate_margins = []
        for action in range(ACTION_COUNT):
            correct_family = []
            margin_family = []
            for family in REGISTERED_FAMILIES:
                supporting = [
                    (scene, scene_weights_by_family[family][scene])
                    for scene in family_scenes[family]
                    if (scene, action) in cell_correct
                ]
                if not supporting:
                    raise AssertionError("validated family/action support disappeared")
                denominator = math.fsum(weight for _scene, weight in supporting)
                if not math.isfinite(denominator) or denominator <= 0.0:
                    raise ThreeArmMetricError(
                        "Bayesian cluster bootstrap normalization is invalid"
                    )
                correct_family.append(
                    math.fsum(
                        weight * cell_correct[(scene, action)]
                        for scene, weight in supporting
                    )
                    / denominator
                )
                margin_family.append(
                    math.fsum(
                        weight * cell_margin[(scene, action)]
                        for scene, weight in supporting
                    )
                    / denominator
                )
            replicate_recalls.append(math.fsum(correct_family) / len(correct_family))
            replicate_margins.append(math.fsum(margin_family) / len(margin_family))
        ba_draws.append(math.fsum(replicate_recalls) / ACTION_COUNT)
        margin_draws.append(min(replicate_margins))
    lower_index, ba_lower = _bootstrap_lower(ba_draws)
    margin_lower_index, margin_lower = _bootstrap_lower(margin_draws)
    if lower_index != margin_lower_index:
        raise AssertionError("bootstrap lower indices disagree")
    hardest = min(range(ACTION_COUNT), key=lambda action: (sf_margins[action], action))
    exact_ties = int((tie_sizes > 1).sum())
    unique = tie_sizes == 1
    return ActionIdentificationSummary(
        row_count=energies.shape[0],
        scene_count=len(scene_family),
        family_count=len(REGISTERED_FAMILIES),
        action_count=ACTION_COUNT,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=ACTION_IDENTIFICATION_BOOTSTRAP_SEED,
        bootstrap_lower_index=lower_index,
        bootstrap_algorithm=ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM,
        bootstrap_interpretation=ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION,
        family_action_supporting_scene_counts=family_action_supporting_scene_counts,
        minimum_family_action_supporting_scene_count=minimum_supporting_scene_count,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion),
        factual_action_counts=tuple(int(value) for value in factual_counts),
        predicted_action_counts=tuple(int(value) for value in confusion.sum(axis=0)),
        row_weighted_accuracy=float(correct.mean()),
        row_weighted_per_action_recall=tuple(float(value) for value in row_recalls),
        row_weighted_balanced_accuracy=float(row_recalls.mean()),
        scene_family_per_action_recall=sf_recalls,
        scene_family_balanced_accuracy=sf_balanced,
        balanced_accuracy_bootstrap_lower_95=ba_lower,
        scene_family_margin_by_action=sf_margins,
        hardest_action_id=hardest,
        hardest_action_margin=sf_margins[hardest],
        hardest_margin_bootstrap_lower_95=margin_lower,
        exact_tie_row_count=exact_ties,
        exact_tie_rate=exact_ties / energies.shape[0],
        unique_winner_count=int(unique.sum()),
        unique_winner_accuracy=(
            float(correct[unique].mean()) if bool(unique.any()) else 0.0
        ),
    )


def _metric_attr(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ThreeArmMetricError(f"metric mapping is missing {name!r}")
        return value[name]
    if not hasattr(value, name):
        raise ThreeArmMetricError(f"metric object is missing {name!r}")
    return getattr(value, name)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ThreeArmMetricError(f"{name} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ThreeArmMetricError(f"{name} must be finite numeric") from error
    if not math.isfinite(result):
        raise ThreeArmMetricError(f"{name} must be finite numeric")
    return result


def localize_three_arm_decision(
    *,
    train_point_advantages: Mapping[str, float],
    validation_tail_point_advantages: Mapping[int, Mapping[str, float]],
    validation_comparisons: Mapping[str, PairedLogEnergyComparison | Mapping[str, Any]],
    action_identification: ActionIdentificationSummary | Mapping[str, Any],
    persistence_comparison: PairedLogEnergyComparison | Mapping[str, Any],
    wrong_history_comparison: PairedLogEnergyComparison | Mapping[str, Any],
    rank_ratio_by_update: Mapping[int, float],
    encoder_identity_exact: bool,
    target_identity_exact: bool,
    contract_checks: Mapping[str, bool],
) -> ThreeArmDecision:
    """Apply the frozen gates and return the earliest scientific localization."""

    if set(validation_comparisons) != {"blind", "shuffled"}:
        raise ThreeArmMetricError("validation comparisons must be blind and shuffled")
    if set(train_point_advantages) != {"blind", "shuffled"}:
        raise ThreeArmMetricError("u700 training points must be blind and shuffled")
    if set(validation_tail_point_advantages) != set(TRAIN_PRIMARY_UPDATES):
        raise ThreeArmMetricError("validation tail must be exactly u500/u600/u700")
    if set(rank_ratio_by_update) != set(TRAIN_PRIMARY_UPDATES):
        raise ThreeArmMetricError("rank points must be exactly u500/u600/u700")
    if type(encoder_identity_exact) is not bool or type(target_identity_exact) is not bool:
        raise ThreeArmMetricError("identity checks must be booleans")
    if any(type(name) is not str or type(value) is not bool for name, value in contract_checks.items()):
        raise ThreeArmMetricError("contract_checks must map names to booleans")

    checks: dict[str, bool] = {
        f"contract:{name}": result for name, result in sorted(contract_checks.items())
    }
    checks["contract:encoder_identity_exact"] = encoder_identity_exact
    checks["contract:target_identity_exact"] = target_identity_exact
    observed_train: dict[str, float] = {}
    for control in ("blind", "shuffled"):
        value = _finite_float(
            train_point_advantages[control], name=f"u700 train {control} point advantage"
        )
        observed_train[control] = value
        checks[f"train:u700:conditioned_vs_{control}_point_positive"] = value > 0.0

    observed_validation_tail: dict[str, dict[str, float]] = {}
    for update in TRAIN_PRIMARY_UPDATES:
        controls = validation_tail_point_advantages[update]
        if set(controls) != {"blind", "shuffled"}:
            raise ThreeArmMetricError("each validation-tail point requires blind and shuffled")
        observed_validation_tail[str(update)] = {}
        for control in ("blind", "shuffled"):
            value = _finite_float(
                controls[control], name=f"u{update} validation {control} point advantage"
            )
            observed_validation_tail[str(update)][control] = value
            checks[
                f"validation:u{update}:conditioned_vs_{control}_point_positive"
            ] = value > 0.0

    validation_observed: dict[str, dict[str, float]] = {}
    for control in ("blind", "shuffled"):
        comparison = validation_comparisons[control]
        comparison_name = _metric_attr(comparison, "control_name")
        comparison_seed = _metric_attr(comparison, "bootstrap_seed")
        comparison_replicates = _metric_attr(comparison, "bootstrap_replicates")
        comparison_lower_index = _metric_attr(comparison, "bootstrap_lower_index")
        checks[f"contract:validation_{control}_comparison_identity"] = (
            comparison_name == control
            and comparison_seed == CONTROL_BOOTSTRAP_SEEDS[control]
            and comparison_replicates == BOOTSTRAP_REPLICATES
            and comparison_lower_index
            == math.floor(BOOTSTRAP_LOWER_QUANTILE * BOOTSTRAP_REPLICATES)
        )
        lower = _finite_float(
            _metric_attr(comparison, "bootstrap_lower_95"),
            name=f"validation {control} lower",
        )
        point = _finite_float(
            _metric_attr(comparison, "macro_log_advantage"),
            name=f"validation {control} point",
        )
        validation_observed[control] = {"point": point, "lower_95": lower}
        checks[f"contract:validation_u700_{control}_point_consistent"] = math.isclose(
            point,
            observed_validation_tail["700"][control],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        checks[f"validation:conditioned_vs_{control}_log_advantage_lower_positive"] = lower > 0.0

    action_seed = _metric_attr(action_identification, "bootstrap_seed")
    action_replicates = _metric_attr(action_identification, "bootstrap_replicates")
    action_lower_index = _metric_attr(action_identification, "bootstrap_lower_index")
    action_algorithm = _metric_attr(action_identification, "bootstrap_algorithm")
    action_interpretation = _metric_attr(
        action_identification, "bootstrap_interpretation"
    )
    action_support_counts = _metric_attr(
        action_identification, "family_action_supporting_scene_counts"
    )
    action_minimum_support = _metric_attr(
        action_identification, "minimum_family_action_supporting_scene_count"
    )
    support_counts_valid = (
        isinstance(action_support_counts, Mapping)
        and set(action_support_counts) == set(REGISTERED_FAMILIES)
        and all(
            isinstance(counts, (tuple, list))
            and len(counts) == ACTION_COUNT
            and all(
                type(count) is int
                and count >= ACTION_IDENTIFICATION_MIN_SUPPORTING_SCENES
                for count in counts
            )
            for counts in action_support_counts.values()
        )
    )
    observed_minimum_support = (
        min(min(counts) for counts in action_support_counts.values())
        if support_counts_valid
        else None
    )
    checks["contract:action_identification_bootstrap_identity"] = (
        action_seed == ACTION_IDENTIFICATION_BOOTSTRAP_SEED
        and action_replicates == BOOTSTRAP_REPLICATES
        and action_lower_index
        == math.floor(BOOTSTRAP_LOWER_QUANTILE * BOOTSTRAP_REPLICATES)
        and action_algorithm == ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM
        and action_interpretation == ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION
        and support_counts_valid
        and type(action_minimum_support) is int
        and action_minimum_support == observed_minimum_support
        and action_minimum_support >= ACTION_IDENTIFICATION_MIN_SUPPORTING_SCENES
    )
    ba_lower = _finite_float(
        _metric_attr(action_identification, "balanced_accuracy_bootstrap_lower_95"),
        name="balanced accuracy lower",
    )
    hardest_lower = _finite_float(
        _metric_attr(action_identification, "hardest_margin_bootstrap_lower_95"),
        name="hardest margin lower",
    )
    checks["action_alignment:balanced_accuracy_lower_above_chance"] = ba_lower > 1.0 / ACTION_COUNT
    checks["action_alignment:hardest_margin_lower_positive"] = hardest_lower > 0.0

    checks["contract:persistence_comparison_identity"] = (
        _metric_attr(persistence_comparison, "control_name") == "persistence"
        and _metric_attr(persistence_comparison, "bootstrap_seed")
        == CONTROL_BOOTSTRAP_SEEDS["persistence"]
        and _metric_attr(persistence_comparison, "bootstrap_replicates")
        == BOOTSTRAP_REPLICATES
        and _metric_attr(persistence_comparison, "bootstrap_lower_index")
        == math.floor(BOOTSTRAP_LOWER_QUANTILE * BOOTSTRAP_REPLICATES)
    )
    checks["contract:wrong_history_comparison_identity"] = (
        _metric_attr(wrong_history_comparison, "control_name") == "wrong_history"
        and _metric_attr(wrong_history_comparison, "bootstrap_seed")
        == CONTROL_BOOTSTRAP_SEEDS["wrong_history"]
        and _metric_attr(wrong_history_comparison, "bootstrap_replicates")
        == BOOTSTRAP_REPLICATES
        and _metric_attr(wrong_history_comparison, "bootstrap_lower_index")
        == math.floor(BOOTSTRAP_LOWER_QUANTILE * BOOTSTRAP_REPLICATES)
    )
    persistence_lower = _finite_float(
        _metric_attr(persistence_comparison, "bootstrap_lower_95"),
        name="persistence lower",
    )
    wrong_history_lower = _finite_float(
        _metric_attr(wrong_history_comparison, "bootstrap_lower_95"),
        name="wrong-history lower",
    )
    checks["predictor:conditioned_vs_persistence_lower_positive"] = persistence_lower > 0.0
    checks["predictor:conditioned_vs_wrong_history_lower_positive"] = wrong_history_lower > 0.0
    ranks = {
        str(update): _finite_float(rank_ratio_by_update[update], name=f"u{update} rank ratio")
        for update in TRAIN_PRIMARY_UPDATES
    }
    rank_pass_count = sum(value >= RANK_RATIO_MIN for value in ranks.values())
    checks["predictor:rank_ratio_at_least_0_25_at_two_updates"] = (
        rank_pass_count >= RANK_PASS_UPDATE_COUNT
    )

    contract_ok = all(value for name, value in checks.items() if name.startswith("contract:"))
    train_ok = all(value for name, value in checks.items() if name.startswith("train:"))
    validation_ok = all(value for name, value in checks.items() if name.startswith("validation:"))
    alignment_ok = all(value for name, value in checks.items() if name.startswith("action_alignment:"))
    predictor_ok = all(value for name, value in checks.items() if name.startswith("predictor:"))
    if not contract_ok:
        status, stage = "INCONCLUSIVE_CONTRACT_FAILURE", "contract"
    elif not train_ok:
        status, stage = "LOCALIZE_TRAIN_FIT_FAILURE", "training_primary"
    elif not validation_ok:
        status, stage = (
            "LOCALIZE_GENERALIZATION_OR_CONFOUNDING",
            "validation_cross_arm",
        )
    elif not alignment_ok:
        status, stage = "LOCALIZE_ACTION_ALIGNMENT_FAILURE", "action_alignment"
    elif not predictor_ok:
        status, stage = "LOCALIZE_PREDICTOR_NOT_USEFUL", "predictor_health"
    else:
        status, stage = (
            "PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY",
            "complete",
        )
    return ThreeArmDecision(
        status=status,
        passed=status == "PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY",
        checks=checks,
        failed_checks=tuple(name for name, result in checks.items() if not result),
        localization_stage=stage,
        observed={
            "train_u700_point_log_advantages": observed_train,
            "validation_tail_point_log_advantages": observed_validation_tail,
            "validation_log_advantages": validation_observed,
            "balanced_accuracy_bootstrap_lower_95": ba_lower,
            "chance_balanced_accuracy": 1.0 / ACTION_COUNT,
            "hardest_margin_bootstrap_lower_95": hardest_lower,
            "persistence_log_advantage_bootstrap_lower_95": persistence_lower,
            "wrong_history_log_advantage_bootstrap_lower_95": wrong_history_lower,
            "rank_ratio_by_update": ranks,
            "rank_pass_update_count": rank_pass_count,
        },
    )


__all__ = [
    "ACTION_COUNT",
    "ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM",
    "ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION",
    "ACTION_IDENTIFICATION_MIN_SUPPORTING_SCENES",
    "ACTION_IDENTIFICATION_BOOTSTRAP_SEED",
    "ACTION_IDENTIFICATION_SCHEMA",
    "ACTION_VOCABULARY",
    "ARMS",
    "BOOTSTRAP_REPLICATES",
    "CANDIDATE_ACTION_DERANGEMENT_SCHEMA",
    "CANDIDATE_ACTION_POSITION",
    "CONTROL_BOOTSTRAP_SEEDS",
    "DECISION_SCHEMA",
    "OVERLAP_AUDIT_SCHEMA",
    "REGISTERED_FAMILIES",
    "SCHEMA",
    "TRAIN_PRIMARY_UPDATES",
    "ActionIdentificationSummary",
    "CandidateActionDerangement",
    "FamilyEqualLogEnergyAdvantage",
    "H6MetadataRow",
    "PairedLogEnergyComparison",
    "ThreeArmDecision",
    "ThreeArmMetricError",
    "audit_h6_metadata_overlap",
    "build_candidate_action_derangement",
    "family_equal_paired_log_energy_advantage",
    "localize_three_arm_decision",
    "normalize_h6_metadata_rows",
    "paired_log_energy_comparison",
    "summarize_nine_way_action_identification",
]
