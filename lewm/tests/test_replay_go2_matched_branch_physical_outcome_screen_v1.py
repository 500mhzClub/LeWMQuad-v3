from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts import replay_go2_matched_branch_physical_outcome_screen_v1 as subject


def _gates(**overrides: bool) -> dict[str, dict[str, bool]]:
    values = {
        "2_privileged_physical_oracle": True,
        "3_odometry_beats_task_action_only": True,
        "4_visual_beats_task_action_only": True,
        "5_visual_beats_odometry": True,
        "6a_odometry_beats_random_expected": True,
        "6b_visual_beats_random_expected": True,
    }
    values.update(overrides)
    return {name: {"passed": value} for name, value in values.items()}


def test_exact_tree_comparison_covers_tensor_dtype_shape_value_and_containers() -> None:
    stored = {
        "tensor": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "nested": [{"value": (1, 2)}],
    }
    assert subject._exact_tree_equal(deepcopy(stored), stored)  # noqa: SLF001
    changed_value = deepcopy(stored)
    changed_value["tensor"][0] = 3.0
    assert not subject._exact_tree_equal(changed_value, stored)  # noqa: SLF001
    changed_dtype = deepcopy(stored)
    changed_dtype["tensor"] = changed_dtype["tensor"].to(torch.float64)
    assert not subject._exact_tree_equal(changed_dtype, stored)  # noqa: SLF001
    changed_container = deepcopy(stored)
    changed_container["nested"] = tuple(changed_container["nested"])
    assert not subject._exact_tree_equal(changed_container, stored)  # noqa: SLF001


def test_checkpoint_reproduction_requires_all_registered_identity_classes() -> None:
    stored = {
        "pca": {"identity_sha256": "a" * 64},
        "outcome_stats": {"identity_sha256": "b" * 64},
        "arms": {
            arm: {
                "input_stats": {"identity_sha256": ("c" if index == 0 else "d") * 64},
                "members": [
                    {
                        "state_identity_sha256": "e" * 64,
                        "training": {"optimizer_steps": 1_024},
                        "state_dict": {"weight": torch.tensor([1.0])},
                    }
                ],
            }
            for index, arm in enumerate(subject.mechanism.LEARNED_ARMS)
        },
    }
    exact = subject._checkpoint_reproduction_v1(  # noqa: SLF001
        deepcopy(stored), stored
    )
    assert all(exact.values())

    changed = deepcopy(stored)
    changed["arms"][subject.mechanism.LEARNED_ARMS[0]]["members"][0][
        "state_dict"
    ]["weight"][0] = 2.0
    mismatch = subject._checkpoint_reproduction_v1(changed, stored)  # noqa: SLF001
    assert mismatch["checkpoint_exact"] is False
    # The separately recorded identity class still compares equal, exposing
    # why both content equality and identity fields are required.
    assert mismatch["state_dict_identities"] is True


def test_evaluation_reproduction_requires_predictions_actions_and_bootstrap() -> None:
    stored = {
        "prediction_artifacts": {"visual": {"prediction_sha256": "a" * 64}},
        "arms": {
            "visual": {
                "summary": {"normalized_rank_regret": 0.1},
                "group_results": [{"selected_action_id": 3}],
            }
        },
        "paired_family_scene_cluster_comparisons": {
            "visual_minus_task": {"lower_95": -0.2, "upper_95": -0.1}
        },
        "gates": _gates(),
    }
    monkey_identity = lambda value: subject.runner.canonical_bytes_v1(value).hex()
    original = subject.mechanism.evaluation_identity_v1
    subject.mechanism.evaluation_identity_v1 = monkey_identity
    try:
        exact = subject._evaluation_reproduction_v1(  # noqa: SLF001
            deepcopy(stored), stored
        )
    finally:
        subject.mechanism.evaluation_identity_v1 = original
    assert all(exact.values())


def test_independent_verdict_implements_visual_odometry_and_stop_precedence() -> None:
    visual = subject._independent_verdict_v1({"gates": _gates()})  # noqa: SLF001
    assert visual["terminal_status"] == subject.runner.PASS_VISUAL_STATUS
    assert visual["passed"] is True
    assert visual["gates"]["1_infrastructure_and_custody"] == {"passed": True}
    assert visual["gates"]["7_deterministic_replay"] == {"passed": True}

    odometry = subject._independent_verdict_v1(  # noqa: SLF001
        {
            "gates": _gates(
                **{
                    "4_visual_beats_task_action_only": False,
                    "5_visual_beats_odometry": False,
                    "6b_visual_beats_random_expected": False,
                }
            )
        }
    )
    assert odometry["terminal_status"] == subject.runner.PASS_ODOMETRY_STATUS
    assert odometry["passed"] is True

    stop = subject._independent_verdict_v1(  # noqa: SLF001
        {
            "gates": _gates(
                **{
                    "3_odometry_beats_task_action_only": False,
                    "4_visual_beats_task_action_only": False,
                    "5_visual_beats_odometry": False,
                }
            )
        }
    )
    assert stop["terminal_status"] == subject.runner.STOP_STATUS
    assert stop["passed"] is False


def test_independent_verdict_rejects_missing_or_extra_gate() -> None:
    malformed = _gates()
    malformed.pop("2_privileged_physical_oracle")
    with pytest.raises(subject.PhysicalOutcomeReplayError, match="inventory"):
        subject._independent_verdict_v1({"gates": malformed})  # noqa: SLF001


def test_cli_contract_binds_authority_checkpoint_and_evaluation() -> None:
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
    assert parsed.expected_authority_byte_count == 1
    assert parsed.expected_checkpoint_byte_count == 2
    assert parsed.expected_evaluation_byte_count == 3


def test_replay_source_has_no_rgb_encoder_or_legacy_loader_execution_path() -> None:
    source = Path(subject.__file__).read_text()
    for forbidden in (
        "load_bound_posthoc_bundle_v1(",
        "derive_documents_v1(",
        "evaluate_task_relevance_v1(",
        "preprocess_dinov2_png_bytes_v1(",
        "forward_features(",
        "read_bound_rgb_bytes_v1(",
    ):
        assert forbidden not in source
