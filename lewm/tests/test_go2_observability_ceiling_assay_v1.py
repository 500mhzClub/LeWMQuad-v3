"""Focused tests for the observability-ceiling assay V1 contract."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lewm.benchmarks import go2_observability_ceiling_assay_v1 as assay
from lewm.models.go2_dinov2_dense_shared_spatial_readout_calibration_v1 import (
    PARAMETER_COUNT as FROZEN_PARAMETER_COUNT,
    initialize_dense_shared_spatial_readout_v1,
)
from lewm.models.go2_observability_ceiling_readout_v1 import (
    initialize_ceiling_readout_v1,
    initialize_privileged_mlp_v1,
    parameter_count_v1,
)


class _Labels:
    def __init__(self, progress: float, path: float, fell: bool, tipped: bool) -> None:
        self.target_progress_m = progress
        self.path_length_m = path
        self.fell = fell
        self.tipped = tipped


class _Branch:
    def __init__(self, action_id: int, rank: int, labels: _Labels) -> None:
        self.action_id = action_id
        self.oracle_dense_rank = rank
        self.labels = labels
        self.target_rgb_artifact_id = f"artifact:{action_id}"


class _Group:
    def __init__(self, state_id, scene_id, family, ranks, progress) -> None:
        self.state_id = state_id
        self.scene_id = scene_id
        self.family = family
        self.group_index = 0
        self.state_index_in_scene = 0
        self.role = "eval"
        self.relative_target_xy_body_m = (1.0, 2.0)
        self.context_rgb_artifact_ids = ("c0", "c1", "c2")
        self.branches = tuple(
            _Branch(action, ranks[action], _Labels(progress[action], 0.1, False, False))
            for action in range(assay.ACTION_COUNT)
        )


def _group(ranks, progress=None, state_id="s0", scene_id="scene0", family="f0"):
    if progress is None:
        progress = [-float(rank) for rank in ranks]
    return _Group(state_id, scene_id, family, ranks, progress)


# ---------------------------------------------------------------- readout ---


def test_rung0_reproduces_the_frozen_245_parameter_readout_exactly():
    """The smallest rung must be a byte-exact replication anchor."""

    assert parameter_count_v1(8, 4) == FROZEN_PARAMETER_COUNT == 245
    ceiling = initialize_ceiling_readout_v1(2_026_080_511, pca_width=8, hidden_width=4)
    frozen = initialize_dense_shared_spatial_readout_v1(2_026_080_511)
    ceiling_state, frozen_state = ceiling.state_dict(), frozen.state_dict()
    assert set(ceiling_state) == set(frozen_state)
    for name in ceiling_state:
        assert torch.equal(ceiling_state[name], frozen_state[name]), name
    panel = torch.randn(5, 256, 24)
    condition = torch.randn(5, 4)
    assert torch.equal(ceiling(panel, condition), frozen(panel, condition))


def test_capacity_ladder_parameter_counts_are_the_registered_values():
    assert [parameter_count_v1(r["pca_width"], r["hidden_width"]) for r in assay.RUNGS] == [
        245,
        6561,
        99969,
    ]


def test_broadcast_panel_degenerates_to_bilinear():
    """Amendment 1's premise: identical tokens make attention cancel.

    This is why the original privileged capacity control tested a degenerate
    code path and had to be replaced.
    """

    model = initialize_ceiling_readout_v1(7, pca_width=32, hidden_width=32)
    features = torch.randn(4, 6)
    panel = assay.broadcast_feature_panel_v1(features, relational_width=96)
    condition = torch.randn(4, 4)
    output = model.forward_with_attention(panel, condition)
    # Attention genuinely varies across patches via the positional term ...
    assert float(output.attention.std().detach()) > 1e-8
    # ... but pooled value is exactly W_v r, so the tanh is inert on this path.
    expected = panel[:, 0, :] @ model.W_v.T
    assert torch.allclose(output.pooled_value, expected, atol=1e-5)


def test_privileged_mlp_is_not_a_member_of_the_readout_family():
    model = initialize_privileged_mlp_v1(11, feature_width=6)
    scores = model(torch.randn(3, 6), torch.randn(3, 4))
    assert tuple(scores.shape) == (3,)
    assert bool(torch.isfinite(scores).all())


# ------------------------------------------------------------------ scorer ---


def test_complete_tie_state_uses_a_denominator_of_one_and_scores_zero():
    """Every action in a complete tie is oracle-equivalent, per the convention."""

    group = _group([0] * assay.ACTION_COUNT)
    for policy, scores in (
        ("argmin", np.zeros((1, assay.ACTION_COUNT))),
        ("random", None),
        ("oracle", None),
    ):
        rows = assay.regret_rows_v1([group], scores, policy=policy)
        assert rows[0]["normalized_rank_regret"] == 0.0
        assert rows[0]["oracle_equivalent_selection"] == 1.0


def test_normalized_targets_admit_a_complete_tie_row():
    targets = assay.normalized_rank_targets_v1([_group([0] * assay.ACTION_COUNT)])
    assert np.allclose(targets, 0.0)


def test_random_expectation_matches_the_closed_form():
    ranks = list(range(assay.ACTION_COUNT))
    rows = assay.regret_rows_v1([_group(ranks)], None, policy="random")
    assert rows[0]["normalized_rank_regret"] == pytest.approx(
        float(np.mean(ranks) / max(ranks))
    )


def test_argmin_selects_and_scores_the_chosen_branch():
    ranks = [4, 0, 8, 1, 2, 3, 5, 6, 7]
    scores = np.zeros((1, assay.ACTION_COUNT))
    scores[0, 2] = -1.0  # force the worst-ranked branch
    rows = assay.regret_rows_v1([_group(ranks)], scores, policy="argmin")
    assert rows[0]["selected_action_id"] == 2
    assert rows[0]["normalized_rank_regret"] == pytest.approx(8 / 8)
    assert rows[0]["oracle_equivalent_selection"] == 0.0


# ------------------------------------------------- closed-form control (V2) ---


def _privileged(dx, dy, dyaw=0.0, path=0.1, fell=False, tipped=False):
    return [dx, dy, dyaw, path, 1.0 if fell else 0.0, 1.0 if tipped else 0.0]


def test_closed_form_control_reconstructs_the_frozen_dense_rank():
    """The V2 corrected control must reproduce the rank rule exactly.

    Progress is reconstructed as |g| - |g - d|; the remaining key components are
    read straight from the privileged feature.
    """

    goal = (3.0, 4.0)  # |g| = 5
    # Nine branches with distinct displacements toward or away from the goal.
    displacements = [
        (0.30, 0.40), (0.15, 0.20), (0.00, 0.00), (-0.15, -0.20), (0.60, 0.80),
        (0.03, 0.04), (-0.30, -0.40), (0.45, 0.60), (0.09, 0.12),
    ]
    group = _group(list(range(assay.ACTION_COUNT)))
    group.relative_target_xy_body_m = goal
    # Rebuild branches so the collection's own rank matches the geometry.
    progress = []
    for dx, dy in displacements:
        progress.append(math.hypot(*goal) - math.hypot(goal[0] - dx, goal[1] - dy))
    ranks = assay.physical._dense_observed_ranks(  # noqa: SLF001
        [
            {
                "action_id": action,
                "physical_target_progress_m": progress[action],
                "physical_path_length_m": 0.1,
                "physical_fell": False,
                "physical_tipped": False,
            }
            for action in range(assay.ACTION_COUNT)
        ]
    )
    group.branches = tuple(
        _Branch(action, ranks[action], _Labels(progress[action], 0.1, False, False))
        for action in range(assay.ACTION_COUNT)
    )
    features = torch.tensor(
        [[_privileged(dx, dy) for dx, dy in displacements]], dtype=torch.float32
    )
    scores = assay.closed_form_identifiability_scores_v1([group], features)
    assert (scores == assay.dense_rank_matrix_v1([group])).all()
    report = assay.arm_report_v1([group], scores, policy="argmin")
    assert report["summary"]["normalized_rank_regret"] == 0.0


def test_closed_form_control_honours_fall_and_tip_precedence():
    """A fallen branch must never outrank an upright one, whatever its progress."""

    goal = (2.0, 0.0)
    displacements = [(0.5, 0.0)] + [(0.05 * i, 0.0) for i in range(1, 9)]
    flags = [True] + [False] * 8  # the best-progress branch fell
    progress = [
        math.hypot(*goal) - math.hypot(goal[0] - dx, goal[1] - dy)
        for dx, dy in displacements
    ]
    ranks = assay.physical._dense_observed_ranks(  # noqa: SLF001
        [
            {
                "action_id": action,
                "physical_target_progress_m": progress[action],
                "physical_path_length_m": 0.1,
                "physical_fell": flags[action],
                "physical_tipped": False,
            }
            for action in range(assay.ACTION_COUNT)
        ]
    )
    group = _group(list(range(assay.ACTION_COUNT)))
    group.relative_target_xy_body_m = goal
    group.branches = tuple(
        _Branch(a, ranks[a], _Labels(progress[a], 0.1, flags[a], False))
        for a in range(assay.ACTION_COUNT)
    )
    features = torch.tensor(
        [
            [
                _privileged(dx, dy, fell=flags[a])
                for a, (dx, dy) in enumerate(displacements)
            ]
        ],
        dtype=torch.float32,
    )
    scores = assay.closed_form_identifiability_scores_v1([group], features)
    assert (scores == assay.dense_rank_matrix_v1([group])).all()
    assert int(np.argmin(scores[0])) != 0  # the fallen branch is not selected


def test_closed_form_control_rejects_a_malformed_feature_shape():
    group = _group(list(range(assay.ACTION_COUNT)))
    with pytest.raises(assay.ObservabilityCeilingAssayError):
        assay.closed_form_identifiability_scores_v1(
            [group], torch.zeros(1, assay.ACTION_COUNT, 3)
        )


# -------------------------------------------------------------- inner split ---


def test_inner_split_is_stratified_one_validation_scene_per_family():
    groups = []
    for family_index in range(assay.FAMILY_COUNT):
        family = f"family{family_index}"
        for scene_index in range(assay.SCENES_PER_FAMILY):
            scene = f"{family}_scene{scene_index}"
            for state_index in range(assay.STATES_PER_SCENE):
                group = _group([0, 1, 2, 3, 4, 5, 6, 7, 8], state_id=f"{scene}-{state_index}", scene_id=scene, family=family)
                group.role = "train"
                groups.append(group)
    split = assay.inner_split_v1(groups)
    assert len(split["fit"]) == 24
    assert len(split["validation"]) == 8
    assert not set(split["fit"]) & set(split["validation"])
    families = {scene.rsplit("_scene", 1)[0] for scene in split["validation"]}
    assert len(families) == assay.FAMILY_COUNT
    # Deterministic in the registered split seed.
    assert assay.inner_split_v1(groups) == split


# ---------------------------------------------------------------- decision ---


def _reports(ceiling, task, privileged=0.2):
    return {
        assay.DINO_ARM: {"summary": {"normalized_rank_regret": ceiling}},
        assay.TASK_ARM: {"summary": {"normalized_rank_regret": task}},
        assay.PRIVILEGED_ARM: {"summary": {"normalized_rank_regret": privileged}},
    }


def _comparisons(dino_task=(-0.05, -0.09, -0.01), context_dino=(0.05, 0.01, 0.09)):
    def entry(values):
        point, lower, upper = values
        return {
            "point_delta": point,
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_half_width": (upper - lower) / 2.0,
        }

    return {
        "dinov2_true_successor_minus_task_action_only": entry(dino_task),
        "context_only_minus_dinov2_true_successor": entry(context_dino),
    }


def test_validity_control_failure_blocks_every_outcome():
    for identifiability, expressivity in ((0.30, 0.01), (0.01, 0.30), (0.30, 0.30)):
        decision = assay.decide_v1(
            _reports(0.05, 0.30),
            _comparisons(),
            identifiability_regret=identifiability,
            expressivity_regret=expressivity,
        )
        assert decision["terminal"] == assay.CAPACITY_FAILURE
        assert decision["assay_valid"] is False


def test_outcome_i_when_the_gate_is_achievable():
    decision = assay.decide_v1(
        _reports(0.10, 0.30), _comparisons(),
        identifiability_regret=0.01, expressivity_regret=0.01,
    )
    assert decision["terminal"] == assay.OUTCOME_I


def test_outcome_iv_takes_precedence_over_iii_and_ii():
    """A degenerate panel is diagnosed before any headroom claim."""

    decision = assay.decide_v1(
        _reports(0.20, 0.30),
        _comparisons(context_dino=(0.001, -0.01, 0.02)),
        identifiability_regret=0.01, expressivity_regret=0.01,
    )
    assert decision["terminal"] == assay.OUTCOME_IV


def test_outcome_iii_when_the_ceiling_does_not_beat_the_task_control():
    decision = assay.decide_v1(
        _reports(0.32, 0.30), _comparisons(),
        identifiability_regret=0.01, expressivity_regret=0.01,
    )
    assert decision["terminal"] == assay.OUTCOME_III


def test_outcome_ii_when_above_the_gate_but_securely_beating_task():
    decision = assay.decide_v1(
        _reports(0.20, 0.30), _comparisons(),
        identifiability_regret=0.01, expressivity_regret=0.01,
    )
    assert decision["terminal"] == assay.OUTCOME_II


def test_inconclusive_when_no_registered_condition_holds():
    decision = assay.decide_v1(
        _reports(0.20, 0.30),
        _comparisons(dino_task=(-0.05, -0.09, 0.02)),
        identifiability_regret=0.01, expressivity_regret=0.01,
    )
    assert decision["terminal"] == assay.INCONCLUSIVE


# ------------------------------------------------------------------- power ---


def test_scenes_to_resolve_scales_as_the_square_of_the_half_width():
    assert assay.scenes_to_resolve_effect_v1(0.02) == pytest.approx(assay.SCENE_COUNT)
    assert assay.scenes_to_resolve_effect_v1(0.04) == pytest.approx(
        4.0 * assay.SCENE_COUNT
    )


def test_bootstrap_is_deterministic_in_the_registered_seed():
    rows_a = assay.regret_rows_v1(
        [_group(list(range(9)), scene_id=f"scene{index}") for index in range(4)],
        np.tile(np.arange(9, dtype=float), (4, 1)),
        policy="argmin",
    )
    rows_b = assay.regret_rows_v1(
        [_group(list(range(9)), scene_id=f"scene{index}") for index in range(4)],
        np.tile(np.arange(9, dtype=float)[::-1], (4, 1)),
        policy="argmin",
    )
    first = assay.paired_family_scene_bootstrap_v1(rows_a, rows_b)
    second = assay.paired_family_scene_bootstrap_v1(rows_a, rows_b)
    assert first == second
    assert first["ci_lower"] <= first["point_delta"] <= first["ci_upper"]
