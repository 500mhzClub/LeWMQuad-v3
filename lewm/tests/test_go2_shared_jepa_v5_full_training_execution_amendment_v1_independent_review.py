"""Independent, data-free review of the V5 full-training amendment V1."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_2026-07-13.md"
)
READINESS = ROOT / "docs/lewm_go2_navigation_work_readiness_goal_2026-07-13.md"
AMENDMENT_SHA256 = (
    "b21d01d062543cc7b7f3f5281f66ac40df76726c678a9364f7a4e451b035a4a7"
)
BOUND_READINESS_SHA256 = (
    "1095252d67f2b450861e97a6083c2866ee3158382f339049e1766b3369dd8a12"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text() -> str:
    return AMENDMENT.read_text(encoding="ascii")


def test_v1_exact_candidate_is_frozen() -> None:
    assert _sha256(AMENDMENT) == AMENDMENT_SHA256


def test_v1_binds_a_mutable_parent_whose_current_bytes_have_changed() -> None:
    text = _text()
    assert BOUND_READINESS_SHA256 in text
    assert _sha256(READINESS) != BOUND_READINESS_SHA256
    assert "Navigation-work readiness goal" in text


def test_v1_gpu_smoke_and_attempt_reservation_order_are_contradictory() -> None:
    text = _text()
    smoke = "Before the exact reservation, the reviewed implementation must pass a"
    reservation = (
        "must reserve the canonical attempt before GPU,\n"
        "model, RGB, label, or role-payload access"
    )
    assert smoke in text
    assert reservation in text
    assert text.index(smoke) < text.index(reservation)


def test_v1_causal_claim_uses_the_promoted_selection_population() -> None:
    text = _text()
    one_line = " ".join(text.split())
    assert "The eight saved promoted checkpoints are evaluated" in text
    assert "on all 495 checkpoint-selection pairs" in text
    assert "the ablation is evaluated once at that exact update on" in text
    assert "the same ordered checkpoint-selection pairs and controls" in text
    assert "A causal claim that JEPA improved development generalization" in one_line
    assert "does not reduce planner-admitted FREE precision" in one_line
