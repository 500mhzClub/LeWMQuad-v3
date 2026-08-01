from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random

import numpy as np
import pytest

from lewm.benchmarks import go2_world_model_existing_pool_three_arm_v1 as metrics_module
from lewm.benchmarks.go2_world_model_existing_pool_three_arm_v1 import (
    ACTION_COUNT,
    ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM,
    ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION,
    ACTION_IDENTIFICATION_MIN_SUPPORTING_SCENES,
    ACTION_IDENTIFICATION_BOOTSTRAP_SEED,
    BOOTSTRAP_REPLICATES,
    CANDIDATE_ACTION_POSITION,
    CONTROL_BOOTSTRAP_SEEDS,
    REGISTERED_FAMILIES,
    ThreeArmMetricError,
    audit_h6_metadata_overlap,
    build_candidate_action_derangement,
    family_equal_paired_log_energy_advantage,
    localize_three_arm_decision,
    normalize_h6_metadata_rows,
    paired_log_energy_comparison,
    summarize_nine_way_action_identification,
)


@dataclass(frozen=True)
class Row:
    index: int | str
    role: str
    family: str
    scene_id: str
    actions: tuple[int, ...]


def _full_support_rows() -> list[Row]:
    rows: list[Row] = []
    index = 0
    for first in range(ACTION_COUNT):
        for second in range(ACTION_COUNT):
            for third in range(ACTION_COUNT):
                family = REGISTERED_FAMILIES[index % len(REGISTERED_FAMILIES)]
                rows.append(
                    Row(
                        index=index,
                        role="train",
                        family=family,
                        scene_id=f"train-{family}-{index % 17}",
                        actions=(first, second, third, first, second, third),
                    )
                )
                index += 1
    for family_index, family in enumerate(REGISTERED_FAMILIES):
        for action in range(ACTION_COUNT):
            rows.append(
                Row(
                    index=f"val-{family_index}-{action}",
                    role="val",
                    family=family,
                    scene_id=f"val-{family}",
                    actions=(action,) * 6,
                )
            )
    return rows


def test_metadata_normalization_accepts_dataclass_mapping_and_json_without_io() -> None:
    row = Row(0, "train", REGISTERED_FAMILIES[0], "scene-a", (0, 1, 2, 3, 4, 5))
    assert normalize_h6_metadata_rows([row])[0].candidate_action_id == 2
    document = json.dumps(
        {
            "rows": [
                {
                    "index": "row-b",
                    "role": "val",
                    "family": REGISTERED_FAMILIES[1],
                    "scene_id": "scene-b",
                    "actions": [
                        "arc_left",
                        "arc_right",
                        "backward",
                        "forward_fast",
                        "forward_medium",
                        "forward_slow",
                    ],
                }
            ]
        }
    )
    normalized = normalize_h6_metadata_rows(document)
    assert normalized[0].actions == (0, 1, 2, 3, 4, 5)
    with pytest.raises(ThreeArmMetricError, match="exact length six"):
        normalize_h6_metadata_rows([{**json.loads(document)["rows"][0], "actions": [0]}])


def test_overlap_audit_has_only_support_and_role_scene_gates() -> None:
    rows = _full_support_rows()
    audit = audit_h6_metadata_overlap(rows)
    assert audit["status"] == "PASS"
    assert audit["passed"]
    assert audit["checks"] == {
        "role_scene_disjointness": True,
        "train_all_actions_supported": True,
        "train_all_ordered_pairs_supported": True,
    }
    assert audit["diagnostic_checks"]["train_all_ordered_triples_supported"]
    assert audit["train_support"]["action_count"] == 9
    assert audit["train_support"]["ordered_pair_count"] == 81
    assert audit["train_support"]["ordered_triple_count"] == 729
    assert not audit["train_support"]["missing_ordered_triples"]
    assert "candidate_a2_with_history_a0_a1_bits" in audit["mutual_information_bits"]["train"]
    assert "diagnostic" in audit["gate_scope"]

    hidden_only = [
        Row(
            row.index,
            row.role,
            row.family,
            row.scene_id,
            row.actions[:2] + (0,) + row.actions[3:],
        )
        if row.role == "train"
        else row
        for row in rows
    ]
    hidden_only_audit = audit_h6_metadata_overlap(hidden_only)
    assert not hidden_only_audit["checks"]["train_all_actions_supported"]
    assert not hidden_only_audit["checks"]["train_all_ordered_pairs_supported"]
    assert not hidden_only_audit["diagnostic_checks"][
        "train_all_ordered_triples_supported"
    ]

    missing_triples = {(0, 0, third) for third in range(7)}
    partial = [
        row
        for row in rows
        if row.role != "train" or tuple(row.actions[:3]) not in missing_triples
    ]
    partial_audit = audit_h6_metadata_overlap(partial)
    assert partial_audit["status"] == "PASS"
    assert partial_audit["passed"]
    assert partial_audit["train_support"]["ordered_triple_count"] == 722
    assert partial_audit["train_support"]["missing_ordered_triples"] == [
        [0, 0, third] for third in range(7)
    ]
    assert partial_audit["failed_checks"] == []
    assert partial_audit["failed_diagnostic_checks"] == [
        "train_all_ordered_triples_supported"
    ]

    overlapping = list(rows)
    first_val = next(index for index, row in enumerate(overlapping) if row.role == "val")
    overlapping[first_val] = Row(
        index=overlapping[first_val].index,
        role="val",
        family=overlapping[first_val].family,
        scene_id=rows[0].scene_id,
        actions=overlapping[first_val].actions,
    )
    failed = audit_h6_metadata_overlap(overlapping)
    assert failed["status"] == "FAIL"
    assert failed["failed_checks"] == ["role_scene_disjointness"]


def _derangement_rows() -> list[Row]:
    rows = []
    for family_index, family in enumerate(REGISTERED_FAMILIES[:2]):
        for local_index, action in enumerate((0, 0, 1, 1, 2, 2, 3, 3)):
            actions = [4, 5, action, 6, 7, 8]
            rows.append(
                Row(
                    index=family_index * 100 + local_index,
                    role="train",
                    family=family,
                    scene_id=f"scene-{family_index}-{local_index}",
                    actions=tuple(actions),
                )
            )
    return rows


def test_candidate_action_derangement_is_exact_deterministic_and_a2_only() -> None:
    rows = _derangement_rows()
    first = build_candidate_action_derangement(rows)
    second = build_candidate_action_derangement(rows)
    assert first == second
    assert first.mapping_sha256 == first.to_dict()["mapping_sha256"]
    assert sorted(first.donor_positions) == list(range(len(rows)))
    assert all(index != donor for index, donor in enumerate(first.donor_positions))
    assert all(
        rows[index].scene_id != rows[donor].scene_id
        for index, donor in enumerate(first.donor_positions)
    )
    assert all(
        factual != shuffled
        for factual, shuffled in zip(
            first.factual_candidate_action_ids,
            first.deranged_candidate_action_ids,
            strict=True,
        )
    )
    changed = []
    for row, candidate in zip(rows, first.deranged_candidate_action_ids, strict=True):
        actions = list(row.actions)
        actions[CANDIDATE_ACTION_POSITION] = candidate
        changed.append(tuple(actions))
    assert all(
        before[:2] == after[:2] and before[3:] == after[3:]
        for before, after in zip((row.actions for row in rows), changed, strict=True)
    )
    assert first.to_dict()["checks"]["role_family_action_marginals_exact"]

    fallback_rows = [
        Row(
            index,
            "train",
            REGISTERED_FAMILIES[0],
            f"fallback-scene-{index % 5}",
            (0, 1, (index * 7 + index // 3) % 4, 3, 4, 5),
        )
        for index in range(17)
    ]
    fallback = build_candidate_action_derangement(fallback_rows)
    assert set(fallback.audit["group_methods"].values()) == {
        "exact_hopcroft_karp_dense_complement"
    }
    assert fallback.audit["checks"]["different_scene_donors"]

    impossible = [
        Row(index, "train", REGISTERED_FAMILIES[0], f"s{index}", (0, 0, 0, 0, 0, 0))
        for index in range(3)
    ]
    with pytest.raises(ThreeArmMetricError, match="no different-scene"):
        build_candidate_action_derangement(impossible)


def _metric_panel() -> tuple[list[str], list[str], np.ndarray]:
    scenes: list[str] = []
    families: list[str] = []
    factual: list[int] = []
    for family in REGISTERED_FAMILIES:
        for scene_index in range(2):
            for action in range(ACTION_COUNT):
                scenes.append(f"scene-{family}-{scene_index}")
                families.append(family)
                factual.append(action)
    return scenes, families, np.asarray(factual, dtype=np.int64)


def test_paired_log_energy_is_row_paired_scene_family_macro_and_seeded() -> None:
    scenes, families, factual = _metric_panel()
    conditioned = np.ones(factual.size)
    control = np.full(factual.size, 2.0)
    summary = paired_log_energy_comparison(
        conditioned, control, scenes, families, control_name="blind"
    )
    assert summary.bootstrap_replicates == BOOTSTRAP_REPLICATES
    assert summary.bootstrap_seed == CONTROL_BOOTSTRAP_SEEDS["blind"]
    assert summary.bootstrap_lower_index == 500
    assert summary.macro_log_advantage == pytest.approx(math.log(2.0))
    assert summary.bootstrap_lower_95 == pytest.approx(math.log(2.0))
    assert summary.positive_family_count == len(REGISTERED_FAMILIES)
    assert summary == paired_log_energy_comparison(
        conditioned, control, scenes, families, control_name="blind"
    )
    with pytest.raises(ThreeArmMetricError, match="strictly positive"):
        paired_log_energy_comparison(
            conditioned, np.zeros_like(control), scenes, families, control_name="blind"
        )


def test_full_train_point_is_row_then_family_not_scene_equal() -> None:
    conditioned: list[float] = []
    control: list[float] = []
    scenes: list[str] = []
    families: list[str] = []
    for family in REGISTERED_FAMILIES:
        conditioned.extend((1.0, 1.0, 1.0))
        control.extend((1.0, 1.0, math.exp(3.0)))
        scenes.extend((f"{family}-large", f"{family}-large", f"{family}-small"))
        families.extend((family, family, family))
    training = family_equal_paired_log_energy_advantage(
        conditioned, control, families, control_name="blind"
    )
    validation = paired_log_energy_comparison(
        conditioned, control, scenes, families, control_name="blind"
    )
    assert training.macro_log_advantage == pytest.approx(1.0)
    assert validation.macro_log_advantage == pytest.approx(1.5)
    assert all(value == pytest.approx(1.0) for value in training.log_advantage_by_family.values())

    tiny = np.full(len(REGISTERED_FAMILIES), np.finfo(np.float64).tiny)
    huge = np.full(len(REGISTERED_FAMILIES), np.finfo(np.float64).max)
    extreme = family_equal_paired_log_energy_advantage(
        tiny, huge, REGISTERED_FAMILIES, control_name="shuffled"
    )
    assert math.isfinite(extreme.macro_log_advantage)


def test_nine_way_summary_uses_lowest_id_ties_and_reports_them() -> None:
    scenes, families, factual = _metric_panel()
    energies = np.full((factual.size, ACTION_COUNT), 2.0)
    energies[np.arange(factual.size), factual] = 1.0
    summary = summarize_nine_way_action_identification(
        energies, factual, scenes, families
    )
    assert summary.bootstrap_seed == ACTION_IDENTIFICATION_BOOTSTRAP_SEED
    assert summary.row_weighted_balanced_accuracy == 1.0
    assert summary.scene_family_balanced_accuracy == 1.0
    assert summary.balanced_accuracy_bootstrap_lower_95 == 1.0
    assert summary.hardest_action_id == 0
    assert summary.hardest_action_margin == 1.0
    assert summary.hardest_margin_bootstrap_lower_95 == 1.0
    assert summary.exact_tie_row_count == 0
    assert summary.unique_winner_count == summary.row_count
    assert summary.bootstrap_algorithm == ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM
    assert summary.bootstrap_interpretation == (
        ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION
    )
    assert summary.minimum_family_action_supporting_scene_count == (
        ACTION_IDENTIFICATION_MIN_SUPPORTING_SCENES
    )
    assert summary.family_action_supporting_scene_counts == {
        family: (2,) * ACTION_COUNT for family in REGISTERED_FAMILIES
    }

    tied = summarize_nine_way_action_identification(
        np.ones_like(energies), factual, scenes, families
    )
    assert tied.predicted_action_counts == (factual.size,) + (0,) * 8
    assert tied.row_weighted_balanced_accuracy == pytest.approx(1 / 9)
    assert tied.scene_family_balanced_accuracy == pytest.approx(1 / 9)
    assert tied.exact_tie_rate == 1.0
    assert tied.hardest_action_margin == 0.0
    assert tied.unique_winner_accuracy == 0.0


def test_nine_way_bootstrap_reuses_one_shared_scene_draw_across_actions() -> None:
    scenes: list[str] = []
    families: list[str] = []
    factual: list[int] = []
    energies: list[list[float]] = []
    for family in REGISTERED_FAMILIES:
        for scene_index in range(2):
            for action in range(ACTION_COUNT):
                row = [10.0] * ACTION_COUNT
                row[action] = 2.0
                wrong = (action + 1) % ACTION_COUNT
                if action == 0:
                    margin = -1.0 if scene_index == 0 else 3.0
                elif action == 1:
                    margin = 3.0 if scene_index == 0 else -1.0
                else:
                    margin = 3.0
                row[wrong] = 2.0 + margin
                scenes.append(f"{family}-scene-{scene_index}")
                families.append(family)
                factual.append(action)
                energies.append(row)

    summary = summarize_nine_way_action_identification(
        energies, factual, scenes, families
    )
    # Actions 0 and 1 are perfectly anti-correlated by scene.  One shared
    # cluster draw preserves exactly one correct outcome between them in every
    # replicate, while the other seven actions are always correct.
    assert summary.scene_family_balanced_accuracy == pytest.approx(8 / 9)
    assert summary.balanced_accuracy_bootstrap_lower_95 == pytest.approx(8 / 9)

    # Reproduce independent per-action positive weights to make this test
    # sensitive to accidentally breaking the shared-weight dependence.
    rng = random.Random(ACTION_IDENTIFICATION_BOOTSTRAP_SEED)
    independent_draws = []
    for _replicate in range(BOOTSTRAP_REPLICATES):
        action_recalls = []
        for action in range(ACTION_COUNT):
            family_recalls = []
            for _family in REGISTERED_FAMILIES:
                weights = [
                    metrics_module._strict_positive_exponential_weight_from_52_bits(
                        rng.getrandbits(52)
                    )
                    for _ in range(2)
                ]
                denominator = math.fsum(weights)
                if action == 0:
                    value = weights[1] / denominator
                elif action == 1:
                    value = weights[0] / denominator
                else:
                    value = 1.0
                family_recalls.append(value)
            action_recalls.append(math.fsum(family_recalls) / len(family_recalls))
        independent_draws.append(math.fsum(action_recalls) / ACTION_COUNT)
    independent_lower = sorted(independent_draws)[500]
    assert independent_lower < summary.balanced_accuracy_bootstrap_lower_95


def test_nine_way_positive_weights_handle_sparse_family_action_support() -> None:
    scenes: list[str] = []
    families: list[str] = []
    factual: list[int] = []
    energies: list[list[float]] = []
    for family_index, family in enumerate(REGISTERED_FAMILIES):
        for scene_index in range(3):
            for action in range(ACTION_COUNT):
                if family_index == 0 and scene_index == 2 and action == 0:
                    continue
                row = [2.0] * ACTION_COUNT
                row[action] = 1.0
                scenes.append(f"{family}-sparse-scene-{scene_index}")
                families.append(family)
                factual.append(action)
                energies.append(row)
    summary = summarize_nine_way_action_identification(
        energies, factual, scenes, families
    )
    assert summary.minimum_family_action_supporting_scene_count == 2
    assert summary.family_action_supporting_scene_counts[REGISTERED_FAMILIES[0]][0] == 2
    assert summary.bootstrap_algorithm == ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM
    assert summary.balanced_accuracy_bootstrap_lower_95 == 1.0
    assert summary.hardest_margin_bootstrap_lower_95 == 1.0


def test_positive_weight_sampler_is_deterministic_finite_at_both_bit_extremes() -> None:
    low = metrics_module._strict_positive_exponential_weight_from_52_bits(0)
    high = metrics_module._strict_positive_exponential_weight_from_52_bits(2**52 - 1)
    assert math.isfinite(low) and low > 0.0
    assert math.isfinite(high) and high > low
    assert low == metrics_module._strict_positive_exponential_weight_from_52_bits(0)
    assert high == metrics_module._strict_positive_exponential_weight_from_52_bits(
        2**52 - 1
    )
    with pytest.raises(ThreeArmMetricError, match="52-bit range"):
        metrics_module._strict_positive_exponential_weight_from_52_bits(-1)
    with pytest.raises(ThreeArmMetricError, match="52-bit range"):
        metrics_module._strict_positive_exponential_weight_from_52_bits(2**52)


def test_nine_way_requires_two_supporting_scenes_per_family_action() -> None:
    scenes: list[str] = []
    families: list[str] = []
    factual: list[int] = []
    energies: list[list[float]] = []
    for family in REGISTERED_FAMILIES:
        for action in range(ACTION_COUNT):
            row = [2.0] * ACTION_COUNT
            row[action] = 1.0
            scenes.append(f"{family}-only-scene")
            families.append(family)
            factual.append(action)
            energies.append(row)
    with pytest.raises(ThreeArmMetricError, match="at least two supporting scenes"):
        summarize_nine_way_action_identification(
            energies, factual, scenes, families
        )


def _decision(**overrides: object):
    def comparison(control: str, point: float = 0.1, lower: float = 0.01):
        return {
            "control_name": control,
            "bootstrap_seed": CONTROL_BOOTSTRAP_SEEDS[control],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_lower_index": 500,
            "macro_log_advantage": point,
            "bootstrap_lower_95": lower,
        }

    arguments: dict[str, object] = {
        "train_point_advantages": {"blind": 0.1, "shuffled": 0.2},
        "validation_tail_point_advantages": {
            500: {"blind": 0.1, "shuffled": 0.1},
            600: {"blind": 0.1, "shuffled": 0.1},
            700: {"blind": 0.1, "shuffled": 0.1},
        },
        "validation_comparisons": {
            "blind": comparison("blind"),
            "shuffled": comparison("shuffled"),
        },
        "action_identification": {
            "bootstrap_seed": ACTION_IDENTIFICATION_BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_lower_index": 500,
            "bootstrap_algorithm": ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM,
            "bootstrap_interpretation": (
                ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION
            ),
            "family_action_supporting_scene_counts": {
                family: [2] * ACTION_COUNT for family in REGISTERED_FAMILIES
            },
            "minimum_family_action_supporting_scene_count": 2,
            "balanced_accuracy_bootstrap_lower_95": 0.2,
            "hardest_margin_bootstrap_lower_95": 0.01,
        },
        "persistence_comparison": comparison("persistence"),
        "wrong_history_comparison": comparison("wrong_history"),
        "rank_ratio_by_update": {500: 0.25, 600: 0.3, 700: 0.2},
        "encoder_identity_exact": True,
        "target_identity_exact": True,
        "contract_checks": {"exact_accounting": True, "finite": True},
    }
    arguments.update(overrides)
    return localize_three_arm_decision(**arguments)  # type: ignore[arg-type]


def test_decision_pass_and_registered_localization_precedence() -> None:
    passed = _decision()
    assert passed.status == "PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY"
    assert passed.passed

    contract = _decision(encoder_identity_exact=False)
    assert contract.status == "INCONCLUSIVE_CONTRACT_FAILURE"

    train = _decision(train_point_advantages={"blind": 0.0, "shuffled": 0.2})
    assert train.status == "LOCALIZE_TRAIN_FIT_FAILURE"

    tail = {
        500: {"blind": 0.0, "shuffled": 0.1},
        600: {"blind": 0.1, "shuffled": 0.1},
        700: {"blind": 0.1, "shuffled": 0.1},
    }
    generalization = _decision(validation_tail_point_advantages=tail)
    assert generalization.status == "LOCALIZE_GENERALIZATION_OR_CONFOUNDING"

    alignment = _decision(
        action_identification={
            "bootstrap_seed": ACTION_IDENTIFICATION_BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_lower_index": 500,
            "bootstrap_algorithm": ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM,
            "bootstrap_interpretation": (
                ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION
            ),
            "family_action_supporting_scene_counts": {
                family: [2] * ACTION_COUNT for family in REGISTERED_FAMILIES
            },
            "minimum_family_action_supporting_scene_count": 2,
            "balanced_accuracy_bootstrap_lower_95": 1 / 9,
            "hardest_margin_bootstrap_lower_95": 1.0,
        }
    )
    assert alignment.status == "LOCALIZE_ACTION_ALIGNMENT_FAILURE"

    predictor = _decision(rank_ratio_by_update={500: 0.24, 600: 0.25, 700: 0.24})
    assert predictor.status == "LOCALIZE_PREDICTOR_NOT_USEFUL"
    assert predictor.to_dict()["localization_stage"] == "predictor_health"

    inconsistent = _decision(
        validation_comparisons={
            "blind": {
                "control_name": "blind",
                "bootstrap_seed": CONTROL_BOOTSTRAP_SEEDS["blind"],
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_lower_index": 500,
                "macro_log_advantage": 0.2,
                "bootstrap_lower_95": 0.01,
            },
            "shuffled": {
                "control_name": "shuffled",
                "bootstrap_seed": CONTROL_BOOTSTRAP_SEEDS["shuffled"],
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_lower_index": 500,
                "macro_log_advantage": 0.1,
                "bootstrap_lower_95": 0.01,
            },
        }
    )
    assert inconsistent.status == "INCONCLUSIVE_CONTRACT_FAILURE"
    assert "contract:validation_u700_blind_point_consistent" in inconsistent.failed_checks


def test_decision_rejects_missing_frozen_points_and_nonfinite_values() -> None:
    with pytest.raises(ThreeArmMetricError, match="validation tail"):
        _decision(validation_tail_point_advantages={500: {"blind": 1.0, "shuffled": 1.0}})
    with pytest.raises(ThreeArmMetricError, match="finite numeric"):
        _decision(train_point_advantages={"blind": math.nan, "shuffled": 1.0})
