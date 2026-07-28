from __future__ import annotations

from copy import deepcopy
import math

import pytest
import torch

from lewm.hierarchical_probability_calibration import (
    CALIBRATION_METHOD,
    CALIBRATION_ROLE,
    CALIBRATION_SCHEMA,
    OUTPUT_TRANSFORM,
    HierarchicalCalibrationParameters,
    apply_hierarchical_probability_calibration,
    evaluate_hierarchical_probability_calibration,
    fit_hierarchical_probability_calibration,
    hierarchical_calibrated_log_probabilities,
    hierarchical_calibrated_probabilities,
    hierarchical_log_odds,
    validate_hierarchical_probability_calibration,
)


def _provenance() -> dict[str, str]:
    return {
        "role": CALIBRATION_ROLE,
        "dataset_manifest_sha256": "a" * 64,
        "scene_roles_sha256": "b" * 64,
        "model_state_sha256": "c" * 64,
    }


def _overlapping_calibration_cells() -> tuple[torch.Tensor, torch.Tensor]:
    # Both factors have overlapping support, so the positive-affine optimum is
    # finite rather than relying on the parameter bound to solve separability.
    labels = torch.tensor(
        [0] * 8 + [1] * 8 + [2] * 8,
        dtype=torch.long,
    )
    known_log_odds = torch.tensor(
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        + [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        + [-0.8, -0.3, 0.2, 0.7, 1.2, 1.7, 2.2, 2.7],
        dtype=torch.float64,
    )
    occupied_log_odds = torch.tensor(
        [0.0] * 8
        + [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        + [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        dtype=torch.float64,
    )
    # z_F=0, z_O=s_O, z_U=logsumexp(z_F,z_O)-s_K realizes the
    # requested hierarchical log odds exactly.
    free = torch.zeros_like(known_log_odds)
    occupied = occupied_log_odds
    unknown = torch.logaddexp(free, occupied) - known_log_odds
    return torch.stack((unknown, free, occupied), dim=1), labels


def test_identity_reconstruction_matches_softmax_and_is_shift_invariant() -> None:
    logits = torch.tensor(
        [
            [[-10_000.0, 2.0], [10_000.0, -5.0]],
            [[0.0, -3.0], [-10_000.0, 4.0]],
            [[10_000.0, 1.0], [0.0, 8.0]],
        ],
        dtype=torch.float64,
    ).unsqueeze(0)
    log_probabilities = hierarchical_calibrated_log_probabilities(logits)
    probabilities = log_probabilities.exp()

    torch.testing.assert_close(
        log_probabilities,
        torch.log_softmax(logits, dim=1),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        rtol=0.0,
        atol=1e-15,
    )
    assert torch.isfinite(log_probabilities).all()

    shifted = logits + torch.tensor(7_500.0, dtype=logits.dtype)
    shifted_known, shifted_occupied = hierarchical_log_odds(shifted)
    known, occupied = hierarchical_log_odds(logits)
    torch.testing.assert_close(shifted_known, known, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(shifted_occupied, occupied, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        hierarchical_calibrated_probabilities(shifted),
        probabilities,
        rtol=1e-12,
        atol=1e-12,
    )


def test_factor_parameters_are_decoupled_and_keep_a_finite_simplex() -> None:
    logits = torch.tensor(
        [[0.5, -1.0, 2.0], [-0.2, 1.1, -0.4]],
        dtype=torch.float64,
    )
    baseline = hierarchical_calibrated_probabilities(logits)
    known_only = hierarchical_calibrated_probabilities(
        logits,
        HierarchicalCalibrationParameters(unknown_known_bias=1.7),
    )
    conditional_only = hierarchical_calibrated_probabilities(
        logits,
        HierarchicalCalibrationParameters(free_occupied_bias=-1.3),
    )

    # KNOWN calibration preserves the FREE:OCCUPIED conditional ratio.
    torch.testing.assert_close(
        known_only[:, 2] / known_only[:, 1],
        baseline[:, 2] / baseline[:, 1],
        rtol=1e-12,
        atol=1e-12,
    )
    # Conditional calibration preserves total KNOWN probability.
    torch.testing.assert_close(
        conditional_only[:, 1:].sum(dim=1),
        baseline[:, 1:].sum(dim=1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.isfinite(known_only).all()
    assert torch.isfinite(conditional_only).all()


def test_fit_is_deterministic_natural_prior_and_records_factor_joint_metrics() -> None:
    logits, labels = _overlapping_calibration_cells()
    first = fit_hierarchical_probability_calibration(
        logits.float(),
        labels,
        provenance=_provenance(),
        maximum_iterations=50,
    )
    second = fit_hierarchical_probability_calibration(
        logits.float(),
        labels,
        provenance=_provenance(),
        maximum_iterations=50,
    )

    assert first == second
    assert first["schema"] == CALIBRATION_SCHEMA
    assert first["method"] == CALIBRATION_METHOD
    assert first["output_transform"] == OUTPUT_TRANSFORM
    assert first["id"].startswith("go2-hier-cal-")
    assert len(first["content_sha256"]) == 64
    assert first["fit"]["device"] == "cpu"
    assert first["fit"]["dtype"] == "float64"
    assert first["fit"]["class_weights"] == "none"
    assert first["fit"]["balancing"] == "none"
    assert first["fit"]["subsampling"] == "none"
    assert first["fit"]["class_backfill"] == "forbidden"
    assert first["fit"]["class_counts"] == {
        "unknown": 8,
        "free": 8,
        "occupied": 8,
    }
    fit_data = first["provenance"]["fit_data"]
    assert fit_data["valid_cell_count"] == 24
    assert fit_data["fit_cell_count"] == 24
    assert fit_data["dropped_valid_cell_count"] == 0
    assert fit_data["backfilled_cell_count"] == 0
    assert fit_data["natural_prior_preserved"] is True

    before = first["metrics"]["before"]
    after = first["metrics"]["after"]
    assert after["joint"]["nll"] < before["joint"]["nll"]
    for factor in (
        "joint",
        "unknown_vs_known",
        "free_vs_occupied_given_known",
    ):
        assert factor in before
        assert factor in after
    assert after["unknown_vs_known"]["support_count"] == 24
    assert after["free_vs_occupied_given_known"]["support_count"] == 16
    assert all(
        math.isfinite(after[factor][metric])
        for factor, metric in (
            ("joint", "nll"),
            ("joint", "multiclass_brier"),
            ("joint", "confidence_ece"),
            ("unknown_vs_known", "nll"),
            ("unknown_vs_known", "brier"),
            ("unknown_vs_known", "ece"),
            ("free_vs_occupied_given_known", "nll"),
            ("free_vs_occupied_given_known", "brier"),
            ("free_vs_occupied_given_known", "ece"),
        )
    )

    parameters = validate_hierarchical_probability_calibration(first)
    expected = hierarchical_calibrated_probabilities(logits.float(), parameters)
    actual = apply_hierarchical_probability_calibration(logits.float(), first)
    torch.testing.assert_close(actual, expected)
    assert evaluate_hierarchical_probability_calibration(
        logits.float(),
        labels,
        first,
    ) == first["metrics"]["after"]


@pytest.mark.parametrize(
    "labels,missing",
    [
        (torch.tensor([1, 2, 1, 2]), "unknown"),
        (torch.tensor([0, 2, 0, 2]), "free"),
        (torch.tensor([0, 1, 0, 1]), "occupied"),
    ],
)
def test_fit_fails_closed_on_missing_factor_support_without_backfill(
    labels: torch.Tensor,
    missing: str,
) -> None:
    logits = torch.zeros((labels.numel(), 3), dtype=torch.float32)
    with pytest.raises(ValueError, match=rf"missing=.*{missing}.*backfill is forbidden"):
        fit_hierarchical_probability_calibration(
            logits,
            labels,
            provenance=_provenance(),
        )

    with pytest.raises(ValueError, match="provenance role"):
        fit_hierarchical_probability_calibration(
            torch.zeros((3, 3)),
            torch.tensor([0, 1, 2]),
            provenance={"role": "checkpoint_selection"},
        )


def test_mask_uses_every_valid_cell_and_artifact_tampering_fails_closed() -> None:
    logits, labels = _overlapping_calibration_cells()
    spatial_logits = logits.T.reshape(1, 3, 4, 6)
    spatial_labels = labels.reshape(1, 4, 6)
    mask = torch.ones_like(spatial_labels, dtype=torch.bool)
    mask[:, 0, 0] = False
    # Preserve support after excluding one UNKNOWN cell.
    artifact = fit_hierarchical_probability_calibration(
        spatial_logits,
        spatial_labels,
        mask=mask,
        provenance=_provenance(),
        maximum_iterations=40,
    )
    assert artifact["fit"]["sample_count"] == 23
    assert artifact["fit"]["class_counts"] == {
        "unknown": 7,
        "free": 8,
        "occupied": 8,
    }
    assert artifact["provenance"]["fit_data"]["masked_out_cell_count"] == 1

    changed_parameters = deepcopy(artifact)
    changed_parameters["parameters"]["unknown_vs_known"]["bias"] += 0.1
    with pytest.raises(ValueError, match="content digest"):
        validate_hierarchical_probability_calibration(changed_parameters)

    changed_provenance = deepcopy(artifact)
    changed_provenance["provenance"]["fit_data"]["backfilled_cell_count"] = 1
    with pytest.raises(ValueError, match="backfilled_cell_count"):
        validate_hierarchical_probability_calibration(changed_provenance)

    changed_id = deepcopy(artifact)
    changed_id["id"] = "go2-hier-cal-0000000000000000"
    with pytest.raises(ValueError, match="ID does not match"):
        validate_hierarchical_probability_calibration(changed_id)
