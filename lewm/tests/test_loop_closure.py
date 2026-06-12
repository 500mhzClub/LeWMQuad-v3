"""Unit tests for lewm.models.loop_closure (Stage 3a consumer gate)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lewm.models.loop_closure import (  # noqa: E402
    LoopClosureHead,
    expected_calibration_error,
    fit_platt,
    pair_features,
    precision_recall_at,
    threshold_at_precision,
)


def test_pair_features_symmetric_and_shaped() -> None:
    torch.manual_seed(0)
    a = F.normalize(torch.randn(32, 16), dim=-1)
    b = F.normalize(torch.randn(32, 16), dim=-1)
    fab, fba = pair_features(a, b), pair_features(b, a)
    assert fab.shape == (32, 32)
    assert torch.allclose(fab, fba)
    print("PASS pair_features symmetric + shaped")


def test_head_separates_synthetic_clusters() -> None:
    torch.manual_seed(1)
    centers = F.normalize(torch.randn(8, 16), dim=-1)
    cells = torch.randint(0, 8, (256,))
    emb = F.normalize(centers[cells] + 0.05 * torch.randn(256, 16), dim=-1)
    i, j = torch.randint(0, 256, (2048,)), torch.randint(0, 256, (2048,))
    labels = (cells[i] == cells[j]).float()
    head = LoopClosureHead(16, hidden=32, dropout=0.0)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(emb[i], emb[j]), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probs = torch.sigmoid(head(emb[i], emb[j]))
    accuracy = ((probs > 0.5).float() == labels).float().mean()
    assert accuracy > 0.95, f"head failed to fit separable clusters (acc={accuracy:.3f})"
    print(f"PASS head separates synthetic clusters (acc={accuracy:.3f})")


def test_threshold_at_precision() -> None:
    probs = torch.tensor([0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    # Prefixes: precision 1,1,1 then drops at 4th (3/4) and never recovers to 1.0.
    threshold = threshold_at_precision(probs, labels, 1.0)
    assert threshold is not None and abs(threshold - 0.8) < 1e-6
    precision, recall = precision_recall_at(probs, labels, threshold)
    assert precision == 1.0 and abs(recall - 0.75) < 1e-9
    assert threshold_at_precision(torch.tensor([0.9, 0.8]), torch.tensor([0.0, 0.0]), 0.99) is None
    print("PASS threshold_at_precision + precision_recall_at")


def test_platt_improves_calibration() -> None:
    torch.manual_seed(2)
    # Overconfident scores: true log-odds scaled 4x.
    true_logits = torch.randn(4000)
    labels = torch.bernoulli(torch.sigmoid(true_logits))
    overconfident = 4.0 * true_logits
    ece_before = expected_calibration_error(torch.sigmoid(overconfident), labels)
    a, b = fit_platt(overconfident, labels)
    ece_after = expected_calibration_error(torch.sigmoid(a * overconfident + b), labels)
    assert ece_after < ece_before, f"Platt did not improve ECE ({ece_before:.3f} -> {ece_after:.3f})"
    assert ece_after < 0.05, f"Platt-calibrated ECE too high ({ece_after:.3f})"
    assert 0.2 < a < 0.4  # recovers ~1/4 scale
    print(f"PASS Platt improves ECE ({ece_before:.3f} -> {ece_after:.3f}, a={a:.3f})")


def test_ece_zero_for_perfect_calibration() -> None:
    torch.manual_seed(3)
    probs = torch.rand(20000)
    labels = torch.bernoulli(probs)
    ece = expected_calibration_error(probs, labels)
    assert ece < 0.02, f"ECE should be near 0 for sampled labels (got {ece:.3f})"
    print(f"PASS ECE near zero when perfectly calibrated ({ece:.3f})")


if __name__ == "__main__":
    test_pair_features_symmetric_and_shaped()
    test_head_separates_synthetic_clusters()
    test_threshold_at_precision()
    test_platt_improves_calibration()
    test_ece_zero_for_perfect_calibration()
    print("ALL PASS")
