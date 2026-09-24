from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from lewm.benchmarks.go2_physical_micro_overfit import (
    FAMILIES,
    PANEL_SCHEMA,
    RESULT_SCHEMA,
    ROWS_PER_FAMILY_PANEL,
    SCENE_POOL_POLICY,
    SELECTION_SEED,
    SELECTION_UNIT,
    aggregate_two_seed_decisions,
    aggregate_two_seed_result_artifacts,
    attach_role_global_shuffle,
    attach_same_scene_wrong_view,
    canonical_json_sha256,
    classify_cross_arm_decision,
    empty_raw_accumulator,
    finalize_raw_accumulator,
    fit_gate,
    frame_records,
    select_train_only_panels,
    update_raw_accumulator,
    validate_panel_manifest,
)
from scripts.run_go2_physical_micro_overfit import (
    _copy_shared_initial_state,
    _independent_query_visibility_report,
    _model_config,
    _reconciled_train_access,
    _terminal_fit_gate_summary,
    _validate_support_contract,
)


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(*, scene: str, family: str, role: str, global_row: int) -> dict:
    return {
        "scene_id": scene,
        "family": family,
        "dataset_role": role,
        "global_row": global_row,
        "env_index": global_row % 4,
        "episode_id": f"episode-{global_row}",
        "reset_count": global_row,
        "current_episode_step": 10,
        "next_episode_step": 11,
        "current_frame_index": 2 * global_row,
        "next_frame_index": 2 * global_row + 1,
        "current_timestamp_ns": 1_000_000 * global_row,
        "next_timestamp_ns": 1_000_000 * global_row + 100_000,
        "primitive": "forward_slow",
        "relative_se2_current_frame": [0.1, 0.0, 0.0],
        "label_shard_path": f"/{scene}.npz",
        "label_shard_sha256": _sha(f"shard:{scene}"),
        "label_shard_row": global_row,
        "current_image_path": f"/{scene}_{global_row}_current.png",
        "current_image_sha256": _sha(f"{scene}:{global_row}:current"),
        "next_image_path": f"/{scene}_{global_row}_next.png",
        "next_image_sha256": _sha(f"{scene}:{global_row}:next"),
    }


def _synthetic_rows() -> tuple[list[dict], dict[str, str]]:
    rows = []
    assignments = {}
    global_row = 0
    for family in FAMILIES:
        for scene_index in range(9):
            scene = f"{family}_train_{scene_index}"
            assignments[scene] = "train"
            for _ in range(32):
                rows.append(
                    _row(
                        scene=scene,
                        family=family,
                        role="train",
                        global_row=global_row,
                    )
                )
                global_row += 1
    forbidden = "forbidden_scene"
    assignments[forbidden] = "g2_evaluation"
    rows.append(
        {
            "scene_id": forbidden,
            "dataset_role": "g2_evaluation",
            "artifact_path": "MUST_NOT_BE_MATERIALIZED",
        }
    )
    return rows, assignments


def _manifest(selection: dict) -> dict:
    core = {
        "schema": PANEL_SCHEMA,
        "selection_seed": selection["selection_seed"],
        "families": selection["families"],
        "rows_per_family_panel": selection["rows_per_family_panel"],
        "selection_unit": selection["selection_unit"],
        "scene_pool_policy": selection["scene_pool_policy"],
        "train_scenes_per_family": selection["train_scenes_per_family"],
        "fit_same_pool_scene_count": selection["fit_same_pool_scene_count"],
        "cross_pool_scene_count": selection["cross_pool_scene_count"],
        "pool_contract": selection["pool_contract"],
        "selection_reports": selection["selection_reports"],
        "primitive_vocabulary": selection["primitive_vocabulary"],
        "panels": selection["panels"],
        "metadata_access": selection["metadata_access"],
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def test_panel_selection_is_train_only_balanced_and_image_disjoint() -> None:
    rows, assignments = _synthetic_rows()
    selection = select_train_only_panels(rows, assignments)
    manifest = _manifest(selection)
    panels = validate_panel_manifest(manifest)

    assert {name: len(value) for name, value in panels.items()} == {
        "fit": 5 * ROWS_PER_FAMILY_PANEL,
        "same_scene_holdout": 5 * ROWS_PER_FAMILY_PANEL,
        "cross_scene_holdout": 5 * ROWS_PER_FAMILY_PANEL,
    }
    encoded = json.dumps(manifest)
    assert "MUST_NOT_BE_MATERIALIZED" not in encoded
    assert all(row["dataset_role"] == "train" for values in panels.values() for row in values)
    hashes = [
        row[f"{side}_image_sha256"]
        for values in panels.values()
        for row in values
        for side in ("current", "next")
    ]
    assert len(hashes) == len(set(hashes)) == 30 * ROWS_PER_FAMILY_PANEL
    assert selection["selection_seed"] == SELECTION_SEED
    assert selection["selection_unit"] == SELECTION_UNIT
    assert selection["scene_pool_policy"] == SCENE_POOL_POLICY
    assert selection["metadata_access"]["role_row_counts"]["g2_evaluation"] == 1
    assert selection["metadata_access"][
        "full_row_objects_parsed_including_non_train_path_metadata"
    ] is True
    assert selection["metadata_access"][
        "non_train_artifact_paths_emitted_to_panel"
    ] is False
    assert selection["metadata_access"]["non_train_artifact_paths_dereferenced"] is False
    fit_streams = {
        (row["scene_id"], row["env_index"], row["episode_id"], row["reset_count"])
        for row in panels["fit"]
    }
    same_streams = {
        (row["scene_id"], row["env_index"], row["episode_id"], row["reset_count"])
        for row in panels["same_scene_holdout"]
    }
    assert fit_streams.isdisjoint(same_streams)


def test_panel_selection_is_label_independent() -> None:
    rows, assignments = _synthetic_rows()
    first = select_train_only_panels(rows, assignments)
    changed = copy.deepcopy(rows)
    for row in changed:
        row["unused_label_statistic"] = 0 if row.get("dataset_role") == "train" else 1
    second = select_train_only_panels(changed, assignments)
    assert first == second


def test_panel_selection_skips_metadata_unusable_stream_without_label_input() -> None:
    rows, assignments = _synthetic_rows()
    first = select_train_only_panels(rows, assignments)
    chosen_global_row = int(
        first["panels"]["fit"]["rows"][0]["global_row"]
    )
    changed = copy.deepcopy(rows)
    raw = next(row for row in changed if row.get("global_row") == chosen_global_row)
    raw["next_image_sha256"] = raw["current_image_sha256"]
    raw["next_image_path"] = raw["current_image_path"]

    second = select_train_only_panels(changed, assignments)
    assert all(
        record["row_count"] == 5 * ROWS_PER_FAMILY_PANEL
        for record in second["panels"].values()
    )
    assert sum(
        int(report["skipped_unusable_stream_count"])
        for panel_reports in second["selection_reports"].values()
        for report in panel_reports.values()
    ) >= 1
    selected_global_rows = {
        int(row["global_row"])
        for panel in second["panels"].values()
        for row in panel["rows"]
    }
    assert chosen_global_row not in selected_global_rows


def test_panel_validation_rejects_cross_panel_image_overlap() -> None:
    rows, assignments = _synthetic_rows()
    manifest = _manifest(select_train_only_panels(rows, assignments))
    changed = copy.deepcopy(manifest)
    changed["panels"]["cross_scene_holdout"]["rows"][0][
        "current_image_sha256"
    ] = changed["panels"]["fit"]["rows"][0]["current_image_sha256"]
    for panel in changed["panels"].values():
        panel["rows_sha256"] = canonical_json_sha256(panel["rows"])
    core = dict(changed)
    core.pop("content_sha256")
    changed["content_sha256"] = canonical_json_sha256(core)
    with pytest.raises(ValueError, match="image hashes overlap"):
        validate_panel_manifest(changed)


def test_role_global_shuffle_is_cross_scene_image_and_transition() -> None:
    rows, assignments = _synthetic_rows()
    fit = select_train_only_panels(rows, assignments)["panels"]["fit"]["rows"]
    records, report = attach_role_global_shuffle(
        frame_records(fit), seed=20260710, namespace="fit"
    )
    assert report["same_image_pairs"] == 0
    assert report["same_scene_pairs"] == 0
    assert report["same_transition_pairs"] == 0
    assert all(record["scene_id"] != record["control_scene_id"] for record in records)


def test_same_scene_wrong_view_is_deterministic_and_never_same_transition() -> None:
    rows, assignments = _synthetic_rows()
    fit = select_train_only_panels(rows, assignments)["panels"]["fit"]["rows"]
    first, report = attach_same_scene_wrong_view(
        frame_records(fit), seed=20260710, namespace="fit"
    )
    second, second_report = attach_same_scene_wrong_view(
        frame_records(fit), seed=20260710, namespace="fit"
    )
    assert first == second
    assert report == second_report
    assert report["same_image_pairs"] == 0
    assert report["same_transition_pairs"] == 0
    assert report["different_scene_pairs"] == 0


def test_raw_metrics_and_fit_gate_accept_perfect_spatial_predictions() -> None:
    labels = np.asarray(
        [
            [
                [0, 1, 2],
                [1, 1, 2],
                [0, 1, 2],
            ]
        ],
        dtype=np.int64,
    )
    mask = np.ones_like(labels, dtype=bool)
    distances = np.asarray(
        [
            [1.2, 1.2, 1.2],
            [2.2, 2.2, 2.2],
            [3.2, 3.2, 3.2],
        ]
    )
    logits = np.full((1, 3, 3, 3), -12.0)
    for class_index in range(3):
        logits[:, class_index][labels == class_index] = 12.0
    accumulator = empty_raw_accumulator()
    update_raw_accumulator(accumulator, logits, labels, mask, distances)
    metrics = finalize_raw_accumulator(accumulator)
    gate = fit_gate(
        metrics,
        cross_scene_shuffled_nll=(
            float(metrics["raw_hierarchical_balanced_nll"]) + 1.0
        ),
        same_scene_shuffled_nll=(
            float(metrics["raw_hierarchical_balanced_nll"]) + 1.0
        ),
    )
    assert gate["passes"] is True
    assert metrics["distance_free_recall"] == {
        "0.0_to_0.5": None,
        "0.5_to_1.0": None,
        "1.0_to_2.0": 1.0,
        "2.0_to_3.0": 1.0,
        "3.0_plus": 1.0,
    }
    assert metrics["class_precision"] == {
        "unknown": 1.0,
        "free": 1.0,
        "occupied": 1.0,
    }
    assert metrics["free_average_precision"] == pytest.approx(1.0)
    assert metrics["occupied_average_precision"] == pytest.approx(1.0)
    quantiles = metrics["posterior_quantiles_by_truth_class"]
    assert quantiles["free"]["free"]["p50"] > 0.999
    assert quantiles["occupied"]["free"]["p95"] < 0.001


class _TinyStateModule(torch.nn.Module):
    def __init__(self, *, width: int) -> None:
        super().__init__()
        self.shared = torch.nn.Linear(3, 3)
        self.variant = torch.nn.Parameter(torch.randn(width))


def test_shared_initialization_copies_only_identically_shaped_tensors() -> None:
    torch.manual_seed(1)
    baseline = _TinyStateModule(width=2)
    torch.manual_seed(2)
    variant = _TinyStateModule(width=4)
    variant_specific_before = variant.variant.detach().clone()
    report = _copy_shared_initial_state(baseline, variant)  # type: ignore[arg-type]

    assert torch.equal(variant.shared.weight, baseline.shared.weight)
    assert torch.equal(variant.shared.bias, baseline.shared.bias)
    assert torch.equal(variant.variant, variant_specific_before)
    assert "shared.weight" in report["copied_tensor_names"]
    assert "variant" in report["variant_specific_tensor_names"]


def test_query_visibility_is_compared_independently_before_copy() -> None:
    class _Decoder:
        def __init__(self, visibility: torch.Tensor) -> None:
            self.projective_query_visibility = visibility

    class _Model:
        def __init__(self, visibility: torch.Tensor) -> None:
            self.bev_decoder = _Decoder(visibility)

    baseline = _Model(torch.tensor([[True, False]]))
    variant = _Model(torch.tensor([[True, False]]))
    report = _independent_query_visibility_report(  # type: ignore[arg-type]
        baseline, variant
    )
    assert report["checked_before_shared_initialization_copy"] is True
    assert report["equal"] is True
    with pytest.raises(ValueError, match="independently constructed"):
        _independent_query_visibility_report(  # type: ignore[arg-type]
            baseline, _Model(torch.tensor([[False, True]]))
        )


def test_patch7_model_config_changes_only_patch_and_normalized_sigma() -> None:
    panel = {
        "local_grid": {
            "shape": [64, 64],
            "forward_center_range_m": [-0.95, 5.35],
            "left_center_range_m": [-3.15, 3.15],
        },
        "source_camera_projection": {
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": 62.837038636424516,
            "near_m": 0.05,
        },
    }
    patch14 = _model_config(panel, arm="patch14_8x8", action_dim=9)
    patch7 = _model_config(panel, arm="patch7_16x16", action_dim=9)
    changed = {
        key for key in patch14 if patch14[key] != patch7[key]
    }
    assert changed == {"patch_size", "projective_attention_sigma_tokens"}
    assert patch14["image_size"] == patch7["image_size"] == 112
    assert patch14["projective_attention_sigma_tokens"] / 8 == pytest.approx(
        patch7["projective_attention_sigma_tokens"] / 16
    )


def _decision_metrics(*, nll: float, far_recall: float, recall: float = 0.99) -> dict:
    return {
        "raw_hierarchical_balanced_nll": nll,
        "unknown_known_balanced_accuracy": recall,
        "free_occupied_balanced_accuracy": recall,
        "class_recall": {
            "unknown": recall,
            "free": recall,
            "occupied": recall,
        },
        "distance_free_recall": {
            "0.0_to_0.5": None,
            "0.5_to_1.0": None,
            "1.0_to_2.0": recall,
            "2.0_to_3.0": recall,
            "3.0_plus": far_recall,
        },
    }


def _decision_arm(
    *,
    fit_pass: bool,
    nll: float,
    far_recall: float,
    unfavorable_families: int = 0,
    family_recall: float = 0.99,
) -> dict:
    correct = _decision_metrics(nll=nll, far_recall=far_recall)
    shuffled = _decision_metrics(nll=nll + 0.5, far_recall=0.2)
    panels = {}
    for name in ("fit", "same_scene_holdout", "cross_scene_holdout"):
        families = {}
        for index, family in enumerate(FAMILIES):
            family_nll = nll
            family_far = far_recall
            if index < unfavorable_families:
                family_nll = 0.21
                family_far = 0.19
            family_correct = _decision_metrics(
                nll=family_nll,
                far_recall=family_far,
                recall=family_recall,
            )
            families[family] = {
                "conditions": {
                    "correct_rgb": family_correct,
                    "role_global_shuffled_rgb": copy.deepcopy(shuffled),
                    "same_scene_wrong_view_rgb": copy.deepcopy(shuffled),
                },
                "fit_gate": {"passes": fit_pass},
            }
        panels[name] = {
            "conditions": {
                "correct_rgb": copy.deepcopy(correct),
                "role_global_shuffled_rgb": copy.deepcopy(shuffled),
                "same_scene_wrong_view_rgb": copy.deepcopy(shuffled),
            },
            "families": families,
            "fit_gate": {"passes": fit_pass},
        }
    return {
        "fit_gate_passed_terminal_three_evaluations": fit_pass,
        "final_panels": panels,
    }


def _decision_stage(
    *,
    patch14_pass: bool,
    patch7_pass: bool,
    patch14_nll: float = 0.20,
    patch7_nll: float = 0.10,
    patch14_far: float = 0.20,
    patch7_far: float = 0.40,
    patch7_unfavorable_families: int = 0,
    patch7_family_recall: float = 0.99,
) -> dict:
    return {
        "patch14_8x8": _decision_arm(
            fit_pass=patch14_pass,
            nll=patch14_nll,
            far_recall=patch14_far,
        ),
        "patch7_16x16": _decision_arm(
            fit_pass=patch7_pass,
            nll=patch7_nll,
            far_recall=patch7_far,
            unfavorable_families=patch7_unfavorable_families,
            family_recall=patch7_family_recall,
        ),
    }


def test_cross_arm_decision_requires_ceiling_after_any_faithful_failure() -> None:
    with pytest.raises(ValueError, match="ceiling optimizer is mandatory"):
        classify_cross_arm_decision(
            _decision_stage(patch14_pass=False, patch7_pass=True), None
        )


def test_cross_arm_decision_reports_only_provisional_causal_support() -> None:
    faithful = _decision_stage(patch14_pass=False, patch7_pass=True)
    ceiling = _decision_stage(patch14_pass=False, patch7_pass=True)
    decision = classify_cross_arm_decision(
        faithful, ceiling
    )
    assert decision["classification"] == "patch7_tokenization_bundle_causal_support"
    assert decision["provisional_patch7_support"] is True
    assert decision["provisional_support_basis"] == "causal_fit"
    assert decision["patch7_full_train_candidate_licensed"] is False
    assert decision["second_seed"] == 20260711


def test_cross_arm_decision_records_both_arm_failure() -> None:
    stage = _decision_stage(patch14_pass=False, patch7_pass=False)
    decision = classify_cross_arm_decision(
        stage, stage
    )
    assert decision["classification"] == (
        "both_arms_fail_patch7_tokenization_bundle_insufficient"
    )
    assert decision["patch7_full_train_candidate_licensed"] is False


def test_cross_arm_decision_records_patch14_expressive_patch7_negative() -> None:
    stage = _decision_stage(patch14_pass=True, patch7_pass=False)
    decision = classify_cross_arm_decision(
        stage, stage
    )
    assert decision["classification"] == (
        "patch14_expressive_patch7_tokenization_bundle_negative"
    )
    assert decision["patch7_full_train_candidate_licensed"] is False


def test_cross_arm_decision_applies_family_macro_holdout_rules() -> None:
    decision = classify_cross_arm_decision(
        _decision_stage(patch14_pass=True, patch7_pass=True), None
    )
    assert decision["classification"] == "patch7_tokenization_bundle_holdout_support"
    assert decision["provisional_patch7_support"] is True
    assert decision["patch7_full_train_candidate_licensed"] is False
    assert all(
        record["passes"] for record in decision["holdout_patch7_checks"].values()
    )
    cross = decision["holdout_patch7_checks"]["cross_scene_holdout"]
    assert cross["strictly_favorable_family_count"] == 5
    assert cross["cross_scene_observed_one_sided_exact_sign_p"] == pytest.approx(
        1 / 32
    )


def test_cross_arm_decision_rejects_cross_scene_four_of_five_even_when_macro_passes() -> None:
    decision = classify_cross_arm_decision(
        _decision_stage(
            patch14_pass=True,
            patch7_pass=True,
            patch7_unfavorable_families=1,
        ),
        None,
    )
    assert decision["classification"] == (
        "both_expressive_no_patch7_tokenization_bundle_support"
    )
    cross = decision["holdout_patch7_checks"]["cross_scene_holdout"]
    same = decision["holdout_patch7_checks"]["same_scene_holdout"]
    assert cross["strictly_favorable_family_count"] == 4
    assert cross["passes"] is False
    assert same["strictly_favorable_family_count"] == 4
    assert same["checks"]["strict_family_nll_and_far_improvement_ge_4_of_5"] is True
    assert decision["patch7_full_train_candidate_licensed"] is False


def test_cross_arm_decision_rejects_one_family_class_recall_regression() -> None:
    stage = _decision_stage(patch14_pass=True, patch7_pass=True)
    for panel in ("same_scene_holdout", "cross_scene_holdout"):
        stage["patch7_16x16"]["final_panels"][panel]["families"][FAMILIES[0]][
            "conditions"
        ]["correct_rgb"]["class_recall"]["occupied"] = 0.97
    decision = classify_cross_arm_decision(stage, None)
    assert decision["provisional_patch7_support"] is False
    for record in decision["holdout_patch7_checks"].values():
        assert record["macro"]["patch7_minus_patch14_class_recall"][
            "occupied"
        ] == pytest.approx(-0.004)
        assert record["checks"][
            "no_family_class_recall_delta_lt_neg_0_01"
        ] is False


def test_cross_arm_decision_flags_second_seed_when_optimizer_stages_disagree() -> None:
    faithful = _decision_stage(patch14_pass=False, patch7_pass=False)
    ceiling = _decision_stage(patch14_pass=True, patch7_pass=True)
    decision = classify_cross_arm_decision(faithful, ceiling)
    assert decision["second_seed_needed"] is True
    assert decision["second_seed"] == 20260711
    assert "faithful_and_ceiling_fit_gates_disagree" in decision["second_seed_reasons"]


def test_cross_arm_decision_keeps_faithful_pass_and_requires_common_stage() -> None:
    faithful = _decision_stage(patch14_pass=True, patch7_pass=False)
    ceiling = _decision_stage(patch14_pass=False, patch7_pass=True)
    decision = classify_cross_arm_decision(faithful, ceiling)
    assert decision["per_arm_expressive_faithful_or_ceiling"] == {
        "patch14_8x8": True,
        "patch7_16x16": True,
    }
    assert decision["matched_holdout_stage"] is None
    assert decision["classification"] == "both_expressive_no_common_stage_comparison"


def test_two_seed_aggregation_is_only_full_training_license_path() -> None:
    faithful = _decision_stage(patch14_pass=False, patch7_pass=True)
    ceiling = _decision_stage(patch14_pass=False, patch7_pass=True)
    primary = classify_cross_arm_decision(faithful, ceiling, seed=20260710)
    replication = classify_cross_arm_decision(faithful, ceiling, seed=20260711)
    aggregate = aggregate_two_seed_decisions(primary, replication)
    assert primary["patch7_full_train_candidate_licensed"] is False
    assert replication["patch7_full_train_candidate_licensed"] is False
    assert aggregate["patch7_full_train_candidate_licensed"] is True
    assert aggregate["qualifying_stage"] == "production_faithful"


def test_two_seed_aggregation_rejects_optimizer_stage_discordance() -> None:
    primary = classify_cross_arm_decision(
        _decision_stage(patch14_pass=False, patch7_pass=True),
        _decision_stage(patch14_pass=False, patch7_pass=True),
        seed=20260710,
    )
    replication = classify_cross_arm_decision(
        _decision_stage(patch14_pass=False, patch7_pass=False),
        _decision_stage(patch14_pass=False, patch7_pass=True),
        seed=20260711,
    )
    aggregate = aggregate_two_seed_decisions(primary, replication)
    assert aggregate["patch7_full_train_candidate_licensed"] is False
    assert aggregate["checks"]["same_qualifying_optimizer_stage"] is False


def _result_artifact(decision: dict) -> dict:
    core = {
        "schema": RESULT_SCHEMA,
        "execution": {"determinism": {"seed": decision["seed"]}},
        "inputs": {
            "panel_manifest": {
                "sha256": "a" * 64,
                "content_sha256": "b" * 64,
            }
        },
        "contract": {"intervention": "patch_tokenization_bundle"},
        "source_hashes": {"protocol": {"sha256": "c" * 64}},
        "cross_arm_decision": decision,
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def test_two_result_artifact_finalizer_validates_provenance_before_license() -> None:
    faithful = _decision_stage(patch14_pass=False, patch7_pass=True)
    ceiling = _decision_stage(patch14_pass=False, patch7_pass=True)
    primary = _result_artifact(
        classify_cross_arm_decision(faithful, ceiling, seed=20260710)
    )
    replication = _result_artifact(
        classify_cross_arm_decision(faithful, ceiling, seed=20260711)
    )
    finalized = aggregate_two_seed_result_artifacts(primary, replication)
    assert finalized["patch7_full_train_candidate_licensed"] is True

    changed = copy.deepcopy(replication)
    changed["source_hashes"]["protocol"]["sha256"] = "d" * 64
    core = dict(changed)
    core.pop("content_sha256")
    changed["content_sha256"] = canonical_json_sha256(core)
    with pytest.raises(ValueError, match="disagree on source_hashes"):
        aggregate_two_seed_result_artifacts(primary, changed)


def test_terminal_fit_gate_requires_exact_last_three_passes() -> None:
    curve = [
        {"step": step, "all_family_and_aggregate_fit_gate_pass": passed}
        for step, passed in ((100, True), (200, False), (300, True), (400, True), (500, True))
    ]
    summary = _terminal_fit_gate_summary(
        curve, maximum_steps=500, evaluation_interval=100
    )
    assert summary["terminal_evaluation_steps"] == [300, 400, 500]
    assert summary["passes"] is True
    curve[-1]["all_family_and_aggregate_fit_gate_pass"] = False
    assert _terminal_fit_gate_summary(
        curve, maximum_steps=500, evaluation_interval=100
    )["passes"] is False


def test_support_contract_aborts_on_any_weak_family_distance_bin() -> None:
    class_counts = np.asarray([100, 100, 100])
    family_class_counts = {
        family: np.asarray([10, 10, 10]) for family in FAMILIES
    }
    aggregate = {name: 1000 for name in ("1.0_to_2.0", "2.0_to_3.0", "3.0_plus")}
    per_family = {
        family: {name: 100 for name in aggregate} for family in FAMILIES
    }
    _validate_support_contract(
        "fit", class_counts, aggregate, family_class_counts, per_family
    )
    per_family[FAMILIES[0]]["3.0_plus"] = 99
    with pytest.raises(ValueError, match="abort without reselection"):
        _validate_support_contract(
            "fit", class_counts, aggregate, family_class_counts, per_family
        )


def test_train_access_reconciliation_sums_support_training_and_evaluation() -> None:
    support = {
        "fit": {
            "label_shard_npz_open_events": 2,
            "label_frame_access_events": 80,
        }
    }
    evaluation_access = {
        "image_decode_events": 240,
        "label_access_events": 80,
        "label_shard_npz_open_events": 2,
    }
    arm = {
        "completed_steps": 3,
        "batch_size": 4,
        "transition_dataset_access": {
            "image_decode_events": 24,
            "label_shard_npz_open_events": 2,
        },
        "learning_curve": [{"fit": {"access": evaluation_access}}],
        "final_panels": {"fit": {"access": evaluation_access}},
    }
    result = _reconciled_train_access(
        support,
        {"patch14_8x8": arm},
        None,
        distinct_images_hashed=5,
        distinct_label_shards_hashed=3,
    )
    totals = result["all_train_role_totals"]
    assert totals["image_byte_open_events"] == 514
    assert totals["image_decode_events"] == 504
    assert totals["label_frame_access_events"] == 264
    assert totals["label_shard_npz_open_events"] == 8
    assert totals["label_shard_byte_open_events"] == 14
    assert totals["model_output_frames"] == 504
    assert result["events_reconciled"] is True
