from __future__ import annotations

from pathlib import Path

import pytest

from lewm.oracle import (
    go2_attentive_readout_layernorm_affine_scientific_successor_v1_contract
    as C,
)


def test_authority_is_one_exploratory_attempt_only() -> None:
    assert C.BASE_SOURCE_COMMIT == "ff1841507366da0c3b9734c532c23240a2385248"
    assert C.AUTHORITY == {
        "scientific_training_authorised": True,
        "authorisation_condition": "validated predecessor SUCCESS_CLASSIFICATION",
        "attempts": 1,
        "exploratory_only": True,
        "calibration_qualification_claim": False,
        "predictor_checkpoints_or_utility_shards_authorised": False,
        "world_model_retraining_authorised": False,
        "final_200_state_corpus_authorised": False,
        "another_probe_seed_architecture_or_retry_authorised": False,
    }
    assert C.TRAINING["total_updates"] == 1_080
    assert C.TRAINING["selection"] == "final_epoch_only_no_selection"
    assert C.EVALUATION["model_sweeps"] == 1


def test_predecessor_success_and_implementation_are_exact() -> None:
    assert C.PREDECESSOR_BINDING["terminal_digest"] == \
        "f8429157e30e4cce8dd902b0c062704c77fa4eba65a9c5757be266664ecd2448"
    assert C.PREDECESSOR_BINDING["local_cases_digest"] == \
        "ce7493274546911478ae7c49177a285957db3bf18ba9a9d8b94b300764dce50b"
    assert C.PREDECESSOR_BINDING["conditional_smoke_digest"] == \
        "cde52904f02e07a7f1c70bf03ea0a3bef8ff2295c22bc54df46e42d255cdbaf1"
    assert C.PREDECESSOR_BINDING["conditional_smoke_checkpoint_sha256"] == \
        "369ce42f854817421e1b83ed695933fabe277e9c1d4c820aebf2f97c315897e1"
    assert C.PREDECESSOR_BINDING["implementation_digest"] == \
        C.LN.IMPLEMENTATION_DIGEST
    assert C.static_contract()["implementation_contract"] == \
        C.LN.IMPLEMENTATION_CONTRACT


def test_fit_only_ledger_is_closed_and_unique() -> None:
    rows = C.fit_only_ledger()
    assert len(rows) == 1_152
    assert C.digest(rows) == C.FIT_ONLY_LEDGER_DIGEST
    assert len({row["branch_identity_digest"] for row in rows}) == 1_152
    assert len({row["training_view_row_digest"] for row in rows}) == 1_152
    assert {row["latent_byte_count"] for row in rows} == {6_291_456}


def test_frozen_training_order_and_gates_are_complete() -> None:
    assert C.DATA_ORDER["base_training_view_row_digest_sequence_digest"] == \
        "c862d0814efb0cbac179eedf9835d869a4dd3588e66c2df668feb44e469e1296"
    assert C.DATA_ORDER["permutation_plan_digest"] == \
        "8e0f2c195f57fa3b883bb8830a4067f95e7965716c851be31b369d5e997c255d"
    assert C.DATA_ORDER["row_presentation_plan_digest"] == \
        "85b1b96ad3aab1442c71a90e6afdbb3e3dc87e8115cb0f9c127953531f7efefb"
    assert len(C.ORIGINAL_GATES) == 8
    assert C.ORIGINAL_GATES["no_latent_pairwise_margin_min"] == 0.05
    assert C.DECISION_ROUTE["classifications"] == [
        "STRONG_READOUT_SIGNAL", "MIXED_READOUT_SIGNAL", "NO_READOUT_SIGNAL"]
    assert C.DECISION_ROUTE["per_family_consistency"] == \
        "reported_only_not_a_gate"


def test_no_latent_and_comparison_bindings_are_frozen() -> None:
    assert C.NO_LATENT_BASELINE["checkpoint_sha256"] == \
        "cfd07d2ad739ef884f3d8ebc3faa01a0b807ef6f19049874eb7fc6ecc9c418ca"
    assert C.NO_LATENT_BASELINE["retrained"] is False
    assert C.NO_LATENT_BASELINE["reevaluated"] is False
    trees = C.static_contract()["frozen_metric_tree_digests"]
    assert set(trees) == {"vitl", "vitg", "no_latent"}
    assert all(set(value) == {"overall", "per_family", "per_stratum"}
               for value in trees.values())


def test_source_closure_binds_fit_loader_and_science_dependencies() -> None:
    assert set(C.NEW_SOURCE_PATHS) == {
        "lewm/oracle/go2_attentive_readout_layernorm_affine_scientific_successor_v1_contract.py",
        "lewm/tests/test_go2_attentive_readout_layernorm_affine_scientific_successor_v1_contract.py",
        "scripts/train_go2_utility_scorer_v1_3_attentive_readout_layernorm_affine_successor_v1.py",
        "lewm/tests/test_train_go2_utility_scorer_v1_3_attentive_readout_layernorm_affine_successor_v1.py",
    }
    for path in (
        "scripts/build_go2_branch_corpus_v1_2.py",
        "scripts/run_go2_scorer_fit_oracle_v1_3.py",
        "scripts/encode_go2_scorer_fit_oracle_v1_3.py",
        "scripts/train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1.py",
    ):
        assert path in C.FROZEN_DEPENDENCY_FILES


def test_predecessor_semantic_preflight_works_from_successor_head() -> None:
    before = C.runtime_root(C.ROOT)
    assert not before.exists() and not before.is_symlink()
    assert C.validate_predecessor_success(C.ROOT) == C.PREDECESSOR_BINDING
    assert not before.exists() and not before.is_symlink()


class _BridgeRunner:
    def __init__(self, *, requested_root: Path | None = None,
                 terminal_error: BaseException | None = None) -> None:
        self.requested_root = requested_root
        self.terminal_error = terminal_error
        self.observed_contract: dict[str, object] | None = None

    def load_contract(self, root: Path) -> dict[str, object]:
        raise AssertionError("historical live loader must be scoped out")

    def validate_terminal(self, root: Path) -> dict[str, object]:
        self.observed_contract = self.load_contract(
            self.requested_root or root)
        if self.terminal_error is not None:
            raise self.terminal_error
        return {"terminal": "validated"}


def test_installed_contract_bridge_is_root_scoped_and_always_restored(
        tmp_path: Path) -> None:
    installed = {"contract": "immutable"}
    success = _BridgeRunner()
    success_loader = success.load_contract
    assert C._validate_predecessor_terminal_with_installed_contract(
        runner=success, installed_contract=installed, root=tmp_path,
    ) == {"terminal": "validated"}
    assert success.observed_contract == installed
    assert success.load_contract == success_loader

    wrong_root = _BridgeRunner(requested_root=tmp_path.parent)
    wrong_loader = wrong_root.load_contract
    with pytest.raises(C.ScientificSuccessorContractError,
                       match="bridge root changed"):
        C._validate_predecessor_terminal_with_installed_contract(
            runner=wrong_root, installed_contract=installed, root=tmp_path)
    assert wrong_root.load_contract == wrong_loader

    sentinel = RuntimeError("semantic validator failed")
    failed = _BridgeRunner(terminal_error=sentinel)
    failed_loader = failed.load_contract
    with pytest.raises(RuntimeError, match="semantic validator failed"):
        C._validate_predecessor_terminal_with_installed_contract(
            runner=failed, installed_contract=installed, root=tmp_path)
    assert failed.load_contract == failed_loader
