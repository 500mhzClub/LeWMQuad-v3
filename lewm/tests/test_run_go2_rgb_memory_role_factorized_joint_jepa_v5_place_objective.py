from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runner = importlib.import_module(
    "scripts.run_go2_rgb_memory_role_factorized_joint_jepa_v1"
)


def _operands(seed: int = 41):
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randn(8, 64, generator=generator, dtype=torch.float32)
    anchors.requires_grad_()
    prediction_offset = torch.randn(
        8, 64, generator=generator, dtype=torch.float32
    )
    predictions = F.normalize(anchors + 0.05 * prediction_offset, dim=1)
    positives = F.normalize(
        torch.randn(8, 64, generator=generator, dtype=torch.float32), dim=1
    )
    negatives = F.normalize(
        torch.randn(8, 64, generator=generator, dtype=torch.float32), dim=1
    )
    return anchors, predictions, positives, negatives


def test_v5_place_objective_matches_scene_local_hinge_formula() -> None:
    anchors, predictions, positives, negatives = _operands()

    terms = runner.place_objective_v5(
        torch, anchors, predictions, positives, negatives
    )

    positive_energy = 1.0 - F.cosine_similarity(
        predictions, positives, dim=1, eps=1.0e-6
    )
    negative_energy = 1.0 - F.cosine_similarity(
        predictions, negatives, dim=1, eps=1.0e-6
    )
    all_group_positive_energy = 1.0 - F.cosine_similarity(
        predictions.reshape(2, 4, 1, 64),
        positives.reshape(2, 1, 4, 64),
        dim=3,
        eps=1.0e-6,
    )
    other_positive_energy = all_group_positive_energy.masked_select(
        (~torch.eye(4, dtype=torch.bool)).unsqueeze(0)
    ).reshape(8, 3)
    competitor_energy = torch.cat(
        (negative_energy.unsqueeze(1), other_positive_energy), dim=1
    )
    expected_hinge = F.relu(
        0.05 + positive_energy.unsqueeze(1) - competitor_energy
    )

    centered = anchors - anchors.mean(dim=0, keepdim=True)
    covariance_matrix = centered.T @ centered / 7.0
    expected_variance = F.relu(
        0.05 - torch.sqrt(covariance_matrix.diagonal() + 1.0e-4)
    ).mean()
    off_diagonal = ~torch.eye(64, dtype=torch.bool)
    expected_covariance = (
        covariance_matrix.square().masked_select(off_diagonal).sum() / 64.0
    )
    expected_loss = (
        positive_energy.mean()
        + expected_hinge.mean()
        + expected_variance
        + 0.10 * expected_covariance
    )

    torch.testing.assert_close(terms.positive_energy, positive_energy)
    torch.testing.assert_close(terms.negative_energy, negative_energy)
    torch.testing.assert_close(terms.other_positive_energy, other_positive_energy)
    torch.testing.assert_close(terms.competitor_energy, competitor_energy)
    torch.testing.assert_close(terms.ranking_hinge, expected_hinge)
    torch.testing.assert_close(terms.alignment, positive_energy.mean())
    torch.testing.assert_close(terms.ranking, expected_hinge.mean())
    torch.testing.assert_close(terms.variance, expected_variance)
    torch.testing.assert_close(terms.covariance, expected_covariance)
    torch.testing.assert_close(terms.loss, expected_loss)
    torch.testing.assert_close(
        terms.hardest_competitor_energy, competitor_energy.min(dim=1).values
    )

    terms.loss.backward()
    assert anchors.grad is not None
    assert bool(torch.isfinite(anchors.grad).all())
    assert float(anchors.grad.abs().sum()) > 0.0


def test_v5_candidates_are_confined_to_each_contiguous_b4_group() -> None:
    anchors, predictions, positives, negatives = _operands(seed=83)
    original = runner.place_objective_v5(
        torch, anchors, predictions, positives, negatives
    )

    replacement_generator = torch.Generator().manual_seed(97)
    changed_positives = positives.clone()
    changed_positives[4:] = F.normalize(
        torch.randn(4, 64, generator=replacement_generator), dim=1
    )
    changed = runner.place_objective_v5(
        torch, anchors, predictions, changed_positives, negatives
    )

    torch.testing.assert_close(
        original.other_positive_energy[:4], changed.other_positive_energy[:4]
    )
    torch.testing.assert_close(
        original.competitor_energy[:4], changed.competitor_energy[:4]
    )
    torch.testing.assert_close(
        original.ranking_hinge[:4], changed.ranking_hinge[:4]
    )
    assert not torch.equal(
        original.other_positive_energy[4:], changed.other_positive_energy[4:]
    )

    for row in range(8):
        group_start = 0 if row < 4 else 4
        other_rows = tuple(
            index for index in range(group_start, group_start + 4) if index != row
        )
        expected = 1.0 - F.cosine_similarity(
            predictions[row].expand(3, -1), positives[list(other_rows)], dim=1
        )
        torch.testing.assert_close(original.other_positive_energy[row], expected)


def test_v5_dispatch_is_explicit_and_guards_gradient_topology() -> None:
    anchors, predictions, positives, negatives = _operands(seed=101)

    default_terms = runner.place_objective_by_version_v1(
        torch, anchors, predictions, positives, negatives
    )
    v5_terms = runner.place_objective_by_version_v1(
        torch,
        anchors,
        predictions,
        positives,
        negatives,
        place_objective_version=5,
    )
    assert isinstance(default_terms, runner.PlaceObjectiveTermsV3)
    assert isinstance(v5_terms, runner.PlaceObjectiveTermsV5)
    assert {
        "PLACE_SCENE_LOCAL_RANKING_MARGIN_V5",
        "PlaceObjectiveTermsV5",
        "place_objective_by_version_v1",
        "place_objective_v5",
    }.issubset(runner.__all__)
    assert (
        inspect.signature(runner.joint_training_update_v1)
        .parameters["place_objective_version"]
        .default
        == 3
    )

    for invalid in (4, True, 3.0):
        with pytest.raises(ValueError, match="integer 3 or 5|must be 3 or 5"):
            runner.place_objective_by_version_v1(
                torch,
                anchors,
                predictions,
                positives,
                negatives,
                place_objective_version=invalid,
            )

    with pytest.raises(RuntimeError, match="gradient topology changed"):
        runner.place_objective_v5(
            torch,
            anchors,
            predictions,
            positives.clone().requires_grad_(),
            negatives,
        )

    nonfinite_predictions = predictions.detach().clone().requires_grad_()
    with torch.no_grad():
        nonfinite_predictions[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="operands are invalid"):
        runner.place_objective_v5(
            torch, anchors, nonfinite_predictions, positives, negatives
        )
