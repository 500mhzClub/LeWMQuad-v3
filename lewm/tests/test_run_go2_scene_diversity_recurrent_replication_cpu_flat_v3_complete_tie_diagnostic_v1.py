from __future__ import annotations

import ast
from pathlib import Path

import pytest

from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as calibration
from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical
from lewm.benchmarks import go2_scene_diversity_recurrent_replication_v1 as benchmark
from scripts import run_go2_scene_diversity_recurrent_replication_cpu_flat_v3_complete_tie_diagnostic_v1 as diagnostic


def _groups(*, complete_tie_states: int) -> tuple[object, ...]:
    if complete_tie_states not in {0, 4}:
        raise ValueError("test fixture supports zero or four complete ties")
    rows = []
    group_index = 128
    for family_index, family in enumerate(calibration.FAMILIES):
        for scene_index in range(4):
            scene_id = f"{family}-scene-{scene_index}"
            for state_index in range(4):
                tie = complete_tie_states == 4 and family_index == 0 and scene_index == 0
                ranks = (0,) * 9 if tie else tuple(range(9))
                state_id = f"eval-state-{group_index}"
                labels = physical._PhysicalLabelsV1(  # noqa: SLF001
                    target_progress_m=0.1,
                    path_length_m=0.2,
                    fell=False,
                    tipped=False,
                    planar_clearance_proxy_min_m=None,
                    grid_recoverability_proxy=None,
                )
                branches = tuple(
                    physical._PhysicalBranchV1(  # noqa: SLF001
                        action_id=action_id,
                        target_rgb_artifact_id=f"{state_id}-target-{action_id}",
                        oracle_dense_rank=ranks[action_id],
                        labels=labels,
                    )
                    for action_id in range(9)
                )
                rows.append(
                    physical._PhysicalGroupV1(  # noqa: SLF001
                        role="eval",
                        state_id=state_id,
                        family=family,
                        scene_id=scene_id,
                        group_index=group_index,
                        state_index_in_scene=state_index,
                        relative_target_xy_body_m=(1.0, 0.0),
                        context_rgb_artifact_ids=tuple(
                            f"{state_id}-context-{index}" for index in range(3)
                        ),
                        branches=branches,
                    )
                )
                group_index += 1
    return tuple(rows)


def test_adapter_is_identical_to_frozen_builder_on_strict_rank_domain() -> None:
    groups = _groups(complete_tie_states=0)
    frozen_plan = benchmark.build_role_feature_plan_v1(groups, role="eval")
    diagnostic_plan = diagnostic.build_eval_role_feature_plan_complete_ties_v1(
        groups, role="eval"
    )

    assert diagnostic_plan == frozen_plan
    assert diagnostic_plan.identity_sha256 == frozen_plan.identity_sha256


def test_adapter_preserves_all_128_states_and_only_admits_four_complete_ties() -> None:
    groups = _groups(complete_tie_states=4)
    with pytest.raises(
        benchmark.SceneDiversityRecurrentReplicationError,
        match="dense ranks are invalid",
    ):
        benchmark.build_role_feature_plan_v1(groups, role="eval")

    plan = diagnostic.build_eval_role_feature_plan_complete_ties_v1(
        groups, role="eval"
    )
    complete_ties = [state for state in plan.states if max(state.dense_ranks) == 0]
    assert len(plan.states) == 128
    assert len(complete_ties) == 4
    assert {state.state_id for state in plan.states} == {
        group.state_id for group in groups
    }
    assert all(state.dense_ranks == (0,) * 9 for state in complete_ties)


def test_random_expected_is_unchanged_off_ties_and_total_on_ties() -> None:
    strict_plan = diagnostic.build_eval_role_feature_plan_complete_ties_v1(
        _groups(complete_tie_states=0), role="eval"
    )
    assert diagnostic.random_expected_report_complete_ties_v1(
        strict_plan
    ) == calibration._random_expected_report(strict_plan)  # noqa: SLF001

    tie_plan = diagnostic.build_eval_role_feature_plan_complete_ties_v1(
        _groups(complete_tie_states=4), role="eval"
    )
    report = diagnostic.random_expected_report_complete_ties_v1(tie_plan)
    tie_rows = [
        row
        for row in report["group_results"]
        if max(tie_plan.states[int(row["state_id"].rsplit("-", 1)[1]) - 128].dense_ranks)
        == 0
    ]
    assert len(tie_rows) == 4
    assert all(row["normalized_rank_regret"] == 0.0 for row in tie_rows)
    assert all(row["oracle_equivalent_selection_rate"] == 1.0 for row in tie_rows)


def test_evaluation_contract_freezes_every_non_domain_variable() -> None:
    contract = diagnostic._expected_evaluation_contract()  # noqa: SLF001
    assert contract["evaluation_only"] is True
    assert contract["training_authorized"] is False
    assert contract["rendering_authorized"] is False
    assert contract["collection_authorized"] is False
    assert contract["eval_state_count"] == 128
    assert contract["expected_eval_complete_tie_state_count"] == 4
    assert contract["eval_state_exclusion_authorized"] is False
    assert contract["random_expected_denominator"] == "max(1,max_dense_rank)"
    assert contract["train_context_rgb_open_count"] == 0
    assert contract["eval_context_rgb_open_count"] == 384
    assert contract["successor_rgb_open_count"] == 0
    assert contract["model_seeds"] == [2026080411, 2026080412, 2026080413]
    assert contract["sampler_seed"] == 2026080414
    assert contract["bootstrap_resamples"] == 10_000
    assert contract["bootstrap_seed"] == 2026080407
    assert contract["frozen_thresholds"] == {
        "maximum_regret": 0.13,
        "visual_minus_task_maximum": -0.02,
        "visual_minus_no_vision_maximum": -0.01,
        "paired_upper_95_must_be_below_zero": True,
        "must_beat_random": True,
    }


def test_evaluation_only_ledger_forbids_train_and_successor_rgb() -> None:
    ledger = diagnostic.EvaluationOnlyLedgerV1()
    ledger.load_receipts("train")
    with pytest.raises(diagnostic.CompleteTieDiagnosticError):
        ledger.open_rgb("train", "context", "forbidden")
    ledger.checkpoint()
    ledger.load_receipts("eval")
    with pytest.raises(diagnostic.CompleteTieDiagnosticError):
        ledger.open_rgb("eval", "successor", "forbidden")


def test_reservation_creates_exact_fresh_hierarchy_and_refuses_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_root = tmp_path / "fresh" / "diagnostic" / "attempt_v1"
    reservation_path = attempt_root / "reservation.json"
    monkeypatch.setattr(diagnostic, "ATTEMPT_ROOT", attempt_root)
    monkeypatch.setattr(diagnostic, "RESERVATION_PATH", reservation_path)
    plan_binding = {
        "path": str(tmp_path / "plan.json"),
        "sha256": "0" * 64,
        "byte_count": 1,
    }
    review_binding = {
        "path": str(tmp_path / "review.json"),
        "sha256": "1" * 64,
        "byte_count": 2,
    }

    binding = diagnostic._reserve_attempt(  # noqa: SLF001
        plan_binding=plan_binding, source_review_binding=review_binding
    )
    assert attempt_root.is_dir()
    assert reservation_path.is_file()
    assert binding == diagnostic.file_binding_v1(reservation_path)
    assert not diagnostic.ATTEMPT_ROOT.samefile(diagnostic.REPO_ROOT)

    with pytest.raises(
        diagnostic.CompleteTieDiagnosticError,
        match="fresh diagnostic root is not fresh",
    ):
        diagnostic._reserve_attempt(  # noqa: SLF001
            plan_binding=plan_binding, source_review_binding=review_binding
        )


def test_source_has_no_predecessor_mutation_training_rendering_or_monkeypatch() -> None:
    source_path = Path(diagnostic.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "fit_checkpoint_v1(" not in source
    assert "train_member_v1(" not in source
    assert "monkeypatch" not in source
    assert ".render(" not in source
    assert "subprocess" not in source
    assert source.count("evaluate_checkpoint_complete_ties_v1(") == 3
    assert "weights_only=True" in source
    assert "map_location=\"cpu\"" in source
    assert "PREDECESSOR_RESULT_PATH.exists()" in source

    imported_modules = {
        "benchmark",
        "calibration",
        "frozen",
        "frozen_runner",
        "grounded",
        "physical",
        "upstream",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    assert target.value.id not in imported_modules


def test_cli_requires_bound_plan_and_source_review() -> None:
    parser = diagnostic.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    parsed = parser.parse_args(
        [
            "--plan",
            "plan.json",
            "--expected-plan-sha256",
            "0" * 64,
            "--expected-plan-byte-count",
            "1",
            "--source-review",
            "review.json",
            "--expected-source-review-sha256",
            "1" * 64,
            "--expected-source-review-byte-count",
            "2",
        ]
    )
    assert parsed.plan == Path("plan.json")
    assert parsed.source_review == Path("review.json")
