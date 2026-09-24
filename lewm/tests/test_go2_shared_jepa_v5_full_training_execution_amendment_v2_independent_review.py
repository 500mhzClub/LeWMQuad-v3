"""Independent, payload-free review of the V5 training amendment V2."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_2026-07-13.md"
V2 = ROOT / "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_v2_2026-07-13.md"
FROZEN = {
    V1: "b21d01d062543cc7b7f3f5281f66ac40df76726c678a9364f7a4e451b035a4a7",
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_author_handoff_2026-07-13.md": "fa0a497fad2f17a5d0919e1160b6040cbe13740315cfc180418d99dbf494d6bc",
    ROOT
    / "lewm/tests/test_go2_shared_jepa_v5_full_training_execution_amendment_v1_independent_review.py": "b2959ea11cff80091a9f94c61dde14750726332001326c0fa30bd186418c6b38",
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_v1_independent_review_2026-07-13.md": "2cd1bf56edd213041496c67238dcf540f2f4a1b72e9abae529e327b4e22c125c",
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_v1_independent_review_block_2026-07-13.json": "c3debd1ee4394e8916b8bfeb7d9237c44f3152e0fd36c27cdf84819c3e356273",
    V2: "b521d2885b5dca1a72838282fbb8e193a21ec0f2db0e0a5950074506fba1f66d",
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_v2_author_handoff_2026-07-13.md": "13102b0a21a71b5c6554ecce380d1ef12f3f3bb582b7175dee6decd17e5cdbfa",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_line(path: Path) -> str:
    return " ".join(path.read_text(encoding="ascii").split())


def test_v2_freezes_candidate_and_complete_v1_block_evidence() -> None:
    assert {path: _sha256(path) for path in FROZEN} == FROZEN


def test_v2_removes_live_readiness_bytes_from_authority() -> None:
    text = _one_line(V2)
    assert "is a live status record. It is informational only" in text
    assert "deleted from the authoritative parent closure" in text
    assert "non_authoritative_status_context" in text
    assert "1095252d67f2b450861e97a6083c2866ee3158382f339049e1766b3369dd8a12" not in text


def test_v2_separates_preflight_and_exact_namespaces_and_processes() -> None:
    text = _one_line(V2)
    assert "full_training_v2_preflight" in text
    assert "full_training_v2`" in text
    assert "different namespaces, reservations, process lifetimes, ledgers, and receipts" in text
    assert "cannot share the preflight process" in text
    assert "cannot silently rerun the preflight" in text


def test_v2_preflight_is_reserved_before_gpu_and_forbids_payloads() -> None:
    text = _one_line(V2)
    assert "retain its directory descriptor before the first preflight GPU-runtime access" in text
    assert "must not open or derive from any repository dataset" in text
    assert "V4 fit checkpoint, V5 checkpoint, learned tensor state" in text
    assert "Synthetic values test shape, memory, finiteness, backward, and device support only" in text


def test_v2_exact_reservation_precedes_all_exact_neural_and_payload_access() -> None:
    text = _one_line(V2)
    reservation = "write/fsync `reservation.json` while holding its directory descriptor"
    neural = "Torch import or GPU-runtime/device initialization"
    payload = "train, checkpoint-selection, probability-calibration, RGB, label, source"
    assert reservation in text and neural in text and payload in text
    assert text.index(reservation) < text.index(neural) < text.index(payload)
    assert "The exact process must be newly spawned after reservation" in text


def test_v2_selection_role_ablation_is_diagnostic_only() -> None:
    text = _one_line(V2)
    assert 'population_role = "checkpoint_selection"' in text
    assert 'interpretation = "matched_development_diagnostic_only"' in text
    assert "causal_generalization_claim_authorized = false" in text
    assert 'qualification_or_selection_effect = "none"' in text
    assert "sentence permitting a causal development-generalization claim" in text
    assert "is deleted" in text


def test_v2_future_untouched_claim_is_separately_gated_and_exact() -> None:
    text = _one_line(V2)
    assert "V2 authorizes no untouched two-arm evaluation" in text
    assert "separate dated preregistration and different-agent PASS before the first byte" in text
    assert "delta_M > 0.0" in text
    assert "count_f(delta_M_f > 0.0) >= 5 of the same 8 frozen families" in text
    assert "delta_P >= 0.0" in text
    assert "P_promoted >= P_ablation" in text
    assert "G2 remains promoted-arm only and closed to the ablation" in text


def test_v2_retains_v1_science_and_all_downstream_authority_false() -> None:
    text = _one_line(V2)
    for required in (
        "20260710`, `N=320` may migrate",
        "128,000 pair presentations, 8,000 updates",
        "complete established JEPA package plus complete current and next V4 supervision",
        "415 pairs, and 759 unique endpoints",
        "role-global one-shot G2 boundary remain mandatory",
        "all six production-stage authority identities remain unset",
    ):
        assert required in text
    assert "It does not license preflight execution, dataset use, V4 execution" in text

