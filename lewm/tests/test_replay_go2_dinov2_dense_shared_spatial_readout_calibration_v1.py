from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from lewm.benchmarks.go2_world_model_counterfactual_pilot_v1 import FAMILIES
from lewm.models.go2_dinov2_dense_shared_spatial_readout_calibration_v1 import (
    dense_shared_state_identity_v1,
    initialize_dense_shared_spatial_readout_v1,
)
from scripts import (
    replay_go2_dinov2_dense_shared_spatial_readout_calibration_v1 as subject,
)


def _binding(path: str, character: str = "a") -> dict[str, object]:
    return {"path": path, "sha256": character * 64, "byte_count": 1}


def _stored_document() -> dict[str, object]:
    return {
        "schema": subject.runner.compatibility.TASK_RELEVANCE_SCHEMA,
        "status": subject.runner.compatibility.TASK_RELEVANCE_PASS_STATUS,
        "thresholds": {
            "minimum_reference_candidate_rgb_ssim": 0.99,
            "required_paired_nearest_neighbour_retrieval_count": 32,
        },
        "measurements": {
            "pixels": {
                "minimum_reference_candidate_rgb_ssim": 0.999873849744854,
            },
            "frozen_predecessor_descriptor_retrieval": {
                "maximum_paired_descriptor_distance": 0.0014817728354341111,
                "paired_nearest_neighbour_retrieval_count": 32,
            },
        },
        "bindings": {
            "parity_result": {
                "path": "/development/parity-result.json",
                "file_sha256": "b" * 64,
                "byte_count": 2,
            },
            "terminal_failure": {
                "path": "/development/terminal-failure.json",
                "file_sha256": "c" * 64,
                "byte_count": 3,
            },
            "progression_analysis": {
                "path": "/development/progression-analysis.json",
                "file_sha256": "d" * 64,
                "byte_count": 4,
            },
        },
    }


def _authority(output_root: Path) -> dict[str, object]:
    return {
        "output_root": str(output_root),
        "preregistration_binding": _binding("/development/preregistration", "0"),
        "input_bindings": {
            "prior_terminal_review": _binding("/development/prior-review", "1"),
            "prior_compatibility_receipt": _binding(
                "/development/prior-compatibility.json", "7"
            ),
            "stored_task_relevance_result": _binding(
                "/development/task-result", "2"
            ),
            "stored_task_relevance_review": _binding(
                "/development/task-review", "3"
            ),
        },
        "source_bindings": {
            "task_relevance_evaluator": _binding(
                "/development/task-evaluator.py", "5"
            ),
            "dense_shared_evaluator": _binding(
                "/development/dense-evaluator.py", "a"
            ),
        },
        "environment": {
            "python": "/usr/bin/python3",
            "torch": "synthetic",
            "hip": "synthetic",
            "numpy": "synthetic",
            "pillow": "synthetic",
        },
    }


def _authority_binding() -> dict[str, object]:
    return _binding("/development/authority.json", "6")


def _prior_admission(
    stored: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    recomputed = deepcopy(dict(stored))
    recomputed["measurements"]["pixels"][  # type: ignore[index]
        "minimum_reference_candidate_rgb_ssim"
    ] = 0.9998738497448542
    return subject.runner.compatibility.admit_task_relevance_result_v1(
        stored=stored, recomputed=recomputed
    )


def _groups() -> tuple[SimpleNamespace, ...]:
    groups = []
    for family_index, family in enumerate(FAMILIES):
        branches = []
        for action in range(9):
            labels = SimpleNamespace(
                fell=False,
                tipped=False,
                target_progress_m=float(9 - action),
                path_length_m=float(action + 1),
                planar_clearance_proxy_min_m=0.5 + action,
                grid_recoverability_proxy=1.0 - action / 10.0,
            )
            branches.append(
                SimpleNamespace(
                    oracle_dense_rank=action,
                    labels=labels,
                )
            )
        groups.append(
            SimpleNamespace(
                state_id=f"state_{family_index}",
                scene_id=f"scene_{family_index}",
                family=family,
                branches=tuple(branches),
            )
        )
    return tuple(groups)


def test_replay_compatibility_receipt_precedes_loader_return_without_live_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _stored_document()
    admitted, admission = _prior_admission(stored)
    calls: list[str] = []

    def forbidden_live_evaluator(*_args: object, **_kwargs: object) -> object:
        calls.append("forbidden_live_evaluator")
        raise AssertionError("RGB/encoder evaluator must not run")

    def strict_loader() -> object:
        calls.append("strict_loader_enter")
        returned = subject.task_relevance.evaluate_task_relevance_v1(
            **subject.runner._task_relevance_call_bindings_v1(stored)  # noqa: SLF001
        )
        assert returned is stored
        assert (tmp_path / "replay_compatibility_receipt.json").is_file()
        calls.append("strict_loader_return")
        return SimpleNamespace(bundle=True)

    monkeypatch.setattr(
        subject.task_relevance,
        "evaluate_task_relevance_v1",
        forbidden_live_evaluator,
    )
    monkeypatch.setattr(
        subject.runner.prior_runner.screen_data,
        "load_bound_posthoc_bundle_v1",
        strict_loader,
    )
    monkeypatch.setattr(
        subject.runner,
        "_load_stored_task_relevance_v1",
        lambda _authority: stored,
    )
    monkeypatch.setattr(
        subject.runner,
        "_replay_prior_compatibility_admission_v1",
        lambda _authority, _stored: (admitted, admission),
    )
    authority = _authority(tmp_path)

    with subject.scoped_replay_compatibility_admission_v1(
        authority, authority_binding=_authority_binding()
    ) as state:
        bundle = (
            subject.runner.prior_runner.screen_data.load_bound_posthoc_bundle_v1()
        )
        assert bundle.bundle is True
        assert state["evaluator_calls"] == 1
        assert state["loader_calls"] == 1
        assert state["admission"] == admission

    receipt = json.loads(
        (tmp_path / "replay_compatibility_receipt.json").read_text()
    )
    assert receipt["phase"] == "replay"
    assert receipt["prior_compatibility_receipt_binding"] == authority[
        "input_bindings"
    ]["prior_compatibility_receipt"]
    assert receipt["publication_stage"] == (
        "inside_task_relevance_compatibility_replay_before_replay_"
        "strict_loader_return"
    )
    assert calls == ["strict_loader_enter", "strict_loader_return"]
    assert (
        subject.task_relevance.evaluate_task_relevance_v1
        is forbidden_live_evaluator
    )
    assert (
        subject.runner.prior_runner.screen_data.load_bound_posthoc_bundle_v1
        is strict_loader
    )


def test_independent_selection_uses_action_id_tie_break_and_exact_summaries() -> None:
    groups = _groups()
    scores = np.zeros((len(groups), 9), dtype=np.float64)
    scores[0, 3] = -1.0
    report = subject._selection_report_v1(groups, scores)  # noqa: SLF001
    assert report["group_results"][0]["selected_action_id"] == 3
    assert report["group_results"][0]["normalized_rank_regret"] == 3.0 / 8.0
    assert all(
        row["selected_action_id"] == 0 for row in report["group_results"][1:]
    )
    assert report["summary"]["chosen_action_histogram"] == {
        str(action): (7 if action == 0 else 1 if action == 3 else 0)
        for action in range(9)
    }
    assert report["per_family"][FAMILIES[0]]["physical_target_progress_m"] == 6.0


def test_independent_family_scene_bootstrap_is_deterministic_and_stratified() -> None:
    candidate = []
    baseline = []
    for family in FAMILIES:
        for scene_index, delta in enumerate((-0.2, -0.1)):
            state_id = f"{family}_{scene_index}"
            common = {
                "state_id": state_id,
                "scene_id": f"{family}_scene_{scene_index}",
                "family": family,
            }
            candidate.append({**common, "normalized_rank_regret": 0.5 + delta})
            baseline.append({**common, "normalized_rank_regret": 0.5})
    first = subject.paired_family_scene_cluster_comparison_replay_v1(
        candidate, baseline, resamples=512, seed=2_026_080_302
    )
    second = subject.paired_family_scene_cluster_comparison_replay_v1(
        candidate, baseline, resamples=512, seed=2_026_080_302
    )
    assert first == second
    assert first["paired_states"] == 16
    assert first["scene_clusters"] == 16
    assert first["scenes_per_family"] == {family: 2 for family in FAMILIES}
    assert first["mean_delta"] == pytest.approx(-0.15)
    assert first["upper_95"] < 0.0


def test_member_prediction_reports_per_seed_ensemble_and_dispersion_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject.mechanism, "STATE_COUNT", 2)
    monkeypatch.setattr(subject.mechanism, "BATCH_STATES", 1)
    members = []
    for seed in subject.mechanism.MODEL_SEEDS:
        state = {
            name: value.detach().clone()
            for name, value in initialize_dense_shared_spatial_readout_v1(
                seed
            ).state_dict().items()
        }
        identity = dense_shared_state_identity_v1(state)
        members.append(
            {
                "seed": seed,
                "true_state": state,
                "true_identity_sha256": identity,
                "true_training": {"state_identity_sha256": identity},
            }
        )
    checkpoint = {"members": members}
    relations = torch.linspace(
        -1.0, 1.0, steps=2 * 9 * 256 * 24, dtype=torch.float32
    ).reshape(2, 9, 256, 24)
    conditions = torch.zeros((2, 9, 4), dtype=torch.float32)
    stacked, ensemble, diagnostics = subject._predict_members_v1(  # noqa: SLF001
        checkpoint,
        relations,
        conditions,
        state_key="true_state",
        device=torch.device("cpu"),
    )
    assert stacked.shape == (3, 2, 9)
    assert ensemble.shape == (2, 9)
    assert diagnostics["score_stack_shape"] == [3, 2, 9]
    assert diagnostics["ensemble_score_sha256"] == subject._score_identity_v1(  # noqa: SLF001
        ensemble
    )
    assert [row["score_shape"] for row in diagnostics["members"]] == [
        [2, 9],
        [2, 9],
        [2, 9],
    ]
    assert diagnostics["seed_dispersion"]["definition"] == (
        "population_std_across_three_seed_scores_per_state_action"
    )


def test_checkpoint_comparison_requires_exact_pca_states_and_step_counts() -> None:
    def checkpoint() -> dict[str, Any]:
        members = []
        for seed in subject.mechanism.MODEL_SEEDS:
            members.append(
                {
                    "seed": seed,
                    "true_state": {"weight": torch.tensor([1.0])},
                    "true_identity_sha256": f"true_{seed}",
                    "true_training": {
                        "optimizer_steps": subject.mechanism.OPTIMIZER_STEPS
                    },
                    "current_state": {"weight": torch.tensor([2.0])},
                    "current_identity_sha256": f"current_{seed}",
                    "current_training": {
                        "optimizer_steps": subject.mechanism.OPTIMIZER_STEPS
                    },
                }
            )
        return {
            "pca": {"identity_sha256": "a" * 64, "mean": torch.tensor([0.0])},
            "members": members,
        }

    stored = checkpoint()
    exact = subject._checkpoint_reproduction_v1(  # noqa: SLF001
        deepcopy(stored), stored
    )
    assert all(exact.values())

    changed = deepcopy(stored)
    changed["members"][1]["true_state"]["weight"][0] = 3.0
    mismatch = subject._checkpoint_reproduction_v1(changed, stored)  # noqa: SLF001
    assert mismatch["state_dict_identities"] is False
    assert mismatch["checkpoint_exact"] is False


def test_independent_verdict_and_cli_contract_do_not_call_primary_aggregation() -> None:
    gates = {
        name: {"passed": True}
        for name in subject.mechanism.SCIENTIFIC_GATE_NAMES
    }
    evaluation = {
        "schema": subject.mechanism.SCHEMA,
        "gates": gates,
        "scientific_gates_2_to_6_passed": True,
    }
    verdict = subject._verdict_v1(evaluation)  # noqa: SLF001
    assert verdict["passed"] is True
    assert verdict["terminal_status"] == subject.runner.PASS_STATUS
    failed = deepcopy(evaluation)
    failed["gates"]["6_true_future_beats_random_expected"]["passed"] = False
    failed["scientific_gates_2_to_6_passed"] = False
    assert subject._verdict_v1(failed)["terminal_status"] == (  # noqa: SLF001
        subject.runner.STOP_STATUS
    )

    parsed = subject.build_parser().parse_args(
        [
            "--authority",
            "/development/authority.json",
            "--expected-authority-sha256",
            "a" * 64,
            "--expected-authority-byte-count",
            "1",
            "--checkpoint",
            "/development/checkpoint.pt",
            "--expected-checkpoint-sha256",
            "b" * 64,
            "--expected-checkpoint-byte-count",
            "2",
            "--evaluation",
            "/development/evaluation.json",
            "--expected-evaluation-sha256",
            "c" * 64,
            "--expected-evaluation-byte-count",
            "3",
        ]
    )
    assert parsed.expected_checkpoint_byte_count == 2

    source = Path(subject.__file__).read_text()
    for forbidden in (
        "mechanism.evaluate_primary_checkpoint_v1",
        "mechanism._report_arm",
        "mechanism._scientific_gates_v1",
        "mechanism.verdict_v1",
        "runner._verdict_status_v1",
        "runner.prior_evaluator.paired_family_scene_cluster_comparison_v1",
    ):
        assert forbidden not in source
