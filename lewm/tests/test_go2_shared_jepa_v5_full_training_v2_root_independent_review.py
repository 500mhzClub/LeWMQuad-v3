"""Independent source review checks for Shared JEPA V5 full-training V2."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from lewm.benchmarks import go2_shared_jepa_v5_full_training_v2_policy as policy


ROOT = Path(__file__).resolve().parents[2]
REVIEWER = "/root/full_training_v2_independent_review"
AUTHOR_HANDOFF = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_full_training_v2_implementation_author_"
    "handoff_2026-07-13.md"
)
EXPECTED_SOURCE_HASHES = {
    policy.POLICY_RELATIVE_PATH: (
        "e0c3409ce104d954e40aa73ae5bd5b79ec3daa77564e90c6be183c2fbc19f680"
    ),
    policy.PREFLIGHT_EXECUTOR_RELATIVE_PATH: (
        "fbc6d63394625d2c3ccc79821d9a07b507fdfb95e02ee1768ed6325857531eff"
    ),
    policy.PREFLIGHT_VERIFIER_RELATIVE_PATH: (
        "1453a6a6134c25cad21d41f44628e4cc8e1e041ae8994d570413ebb1101e09e3"
    ),
    policy.EXACT_EXECUTOR_RELATIVE_PATH: (
        "698fb92f2f854365f2d0bfbf6f034b1c3f04704a8d6227fceff7c3ed275fc271"
    ),
    policy.EXACT_TRAINER_RELATIVE_PATH: (
        "bdd8e4b1c24e855f3e3ff535a195f2c370c4ffdadc48eb9e83b214b53362f23b"
    ),
    policy.EXACT_VERIFIER_RELATIVE_PATH: (
        "d8950c8bf23b0bd5494c7c864f2f2543d533b0bc07af3f70287291227c872543"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="ascii")


def test_frozen_candidate_and_handoff_hashes_are_exact() -> None:
    assert set(EXPECTED_SOURCE_HASHES) == set(policy.IMPLEMENTATION_SOURCE_PATHS)
    assert {
        relative: _sha256(ROOT / relative)
        for relative in policy.IMPLEMENTATION_SOURCE_PATHS
    } == EXPECTED_SOURCE_HASHES
    assert _sha256(AUTHOR_HANDOFF) == (
        "10f08adf660e06f0290d394d5e7d7b9796fb3640b12eebc1cbb8ac5c0d99a0da"
    )


def test_review_core_is_narrow_and_cannot_authorize_exact_work() -> None:
    core = policy.expected_implementation_review_core(
        reviewer=REVIEWER,
        source_bindings=EXPECTED_SOURCE_HASHES,
    )
    assert core["implementation_author"] == "/root/coordinator_v2_qa"
    assert core["payload_free_preflight_authorized"] is True
    assert core["exact_execution_authorized"] is False
    assert core["dataset_or_checkpoint_access_authorized"] is False
    assert core["g2_or_heldout_authorized"] is False
    assert core["production_or_promotion_authorized"] is False
    with pytest.raises(PermissionError):
        policy.expected_implementation_review_core(
            reviewer=policy.IMPLEMENTATION_AUTHOR,
            source_bindings=EXPECTED_SOURCE_HASHES,
        )


def test_manifest_remains_blocked_on_every_required_binding() -> None:
    raw = (ROOT / policy.EXACT_EXECUTION_MANIFEST_RELATIVE_PATH).read_bytes()
    manifest = policy.parse_canonical_json(raw, name="blocked exact manifest")
    assert manifest == policy.content_value(policy.execution_manifest_core())
    assert manifest["exact_execution_authorized"] is False
    assert manifest["unresolved_required_bindings"] == sorted(
        policy.REQUIRED_BINDING_NAMES
    )
    assert len(manifest["unresolved_required_bindings"]) == 19
    assert all(
        value is None for value in manifest["required_exact_bindings"].values()
    )
    with pytest.raises(PermissionError, match="blocked before reservation"):
        policy.validate_execution_manifest(manifest, require_ready=True)


def test_reservation_precedes_neural_imports_and_payload_openers() -> None:
    preflight = _source(policy.PREFLIGHT_EXECUTOR_RELATIVE_PATH)
    exact = _source(policy.EXACT_EXECUTOR_RELATIVE_PATH)
    trainer = _source(policy.EXACT_TRAINER_RELATIVE_PATH)
    assert preflight.index("reservation = reserve_operation()") < preflight.index(
        "measurements = smoke_operation(reservation)"
    )
    assert exact.index("reservation = reserve_operation()") < exact.index(
        "trainer_summary = trainer_operation(reservation)"
    )
    assert trainer.index("preflight, preflight_raw = _load_preflight_first") < (
        trainer.index("backend = backend_loader()")
    )
    tree = ast.parse(trainer)
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    for node in imports:
        names = (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
        if any(name == "torch" or name.startswith("torch.") for name in names):
            assert node.lineno >= 600


def test_independent_verifier_reconstructs_science_and_inventory() -> None:
    verifier = _source(policy.EXACT_VERIFIER_RELATIVE_PATH)
    required_fragments = (
        "expected_table.append(torch.quantile(rows, 0.5, dim=0))",
        "migration_baseline = self._candidate(",
        'self._checkpoint("promoted_jepa", update)',
        'self._checkpoint(\n                        "matched_no_jepa", update',
        "calibrations = {",
        "require_completion_rehash=True",
        'trainer_metrics_trusted": False',
        'raw_inputs_and_checkpoints_reopened": True',
    )
    for fragment in required_fragments:
        assert fragment in verifier
    assert "from scripts.train_go2_shared_jepa_v5_full_training_v2" not in verifier


def test_known_successor_boundaries_are_explicit_not_silent() -> None:
    assert policy.RAW_SUPERVISION_BUILDER_RELATIVE_PATH.endswith(
        "raw_supervision_builder_v1.py"
    )
    assert policy.RAW_SUPERVISION_AUDITOR_RELATIVE_PATH.endswith(
        "raw_supervision_auditor_v1.py"
    )
    assert policy.JOINT_LOSS_CONTRACT["promoted_jepa"]["v4_components"] == {
        "ordered_first_hit_nll": 0.25,
        "target_bin_offset_smooth_l1": 0.25,
        "ground_clear_distance_state_balanced_bce": 0.25,
        "derived_raster_hierarchical_bce": 0.25,
    }
    handoff = AUTHOR_HANDOFF.read_text(encoding="ascii")
    assert "Builder V7 approval" in handoff
    assert "before G2" in handoff
    assert "Camera V9" in handoff
    assert "must not execute" in handoff
