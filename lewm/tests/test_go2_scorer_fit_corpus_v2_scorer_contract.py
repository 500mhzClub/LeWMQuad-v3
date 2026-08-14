"""Source-only tests for the full-bank scorer-fit V2 successor contract."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from lewm.oracle import go2_scorer_fit_corpus_v2_scorer_contract as C


def _binding(key: str, digit: str, *, authority: bool = False) -> dict:
    value = {
        "path": f".generated/fixture/{key}.json",
        "self_digest_key": key,
        "self_digest": digit * 64,
        "raw_sha256": digit * 64,
        "byte_count": 101,
    }
    if authority:
        value.update({
            "schema": f"fixture_{key}",
            "source_repository_commit": digit * 40,
        })
    return value


def _inputs() -> dict:
    return {
        "source_binding": {
            "schema": "fixture_clean_source",
            "source_repository_commit": "a" * 40,
            "source_repository_clean": True,
            "bound_implementations_digest": "b" * 64,
        },
        "design_binding": _binding(
            C.DESIGN.DESIGN_SELF_KEY, "1", authority=True),
        "source_correction_binding": _binding(
            C.DESIGN.SOURCE_CORRECTION_SELF_KEY, "7", authority=True),
        "mask_classification_binding": _binding(
            C.DESIGN.MASK_CLASSIFICATION_SELF_KEY, "2", authority=True),
        "selection_binding": _binding(
            "full_bank_small_completion_selection_digest", "3"),
        "revalidation_binding": _binding(
            "full_bank_preoutcome_state_revalidation_digest", "4"),
        "state_manifest_binding": _binding("state_manifest_digest", "5"),
        "assignment_manifest_binding": _binding(
            "full_bank_assignment_manifest_digest", "6"),
    }


def test_successor_preserves_science_and_interprets_epoch_budget_exactly():
    contract = C.build_contract(**_inputs())
    assert C.validate_contract(contract) == contract
    assert contract["protected_predecessor_scientific_contract"]["scorer"] \
        == C.PREDECESSOR.SCORER
    assert contract["corpus_counts"]["branches"] == 1_440
    assert contract["corpus_counts"]["fit_branches"] == 1_152
    assert contract["corpus_counts"]["calibration_branches"] == 288
    budget = contract["training_budget_interpretation"]
    assert budget["epochs"] == 60
    assert budget["optimizer_updates_per_model"] == 1_080
    assert budget["example_presentations_per_model"] == 69_120
    assert budget["step_budget_also_retained"] is False
    lineage = contract["preoutcome_lineage"]
    assert lineage["scorer_fit_corpus_v2_source_correction_digest"] \
        == "7" * 64
    assert lineage["v1_parallel_failure_receipt_digest"] \
        == C.V1_FAILURE_RECEIPT_DIGEST
    assert lineage["exact_infeasibility_digest"] == C.EXACT_INFEASIBILITY_DIGEST


def test_contract_and_artifact_reject_any_resigned_scientific_change():
    artifact = C.build_contract_artifact(**_inputs())
    assert C.validate_contract_artifact(artifact) == artifact

    changed = copy.deepcopy(artifact)
    changed["contract"]["protected_predecessor_scientific_contract"][
        "scorer"]["training"]["epochs"] = 61
    changed["contract"][C.CONTRACT_SELF_KEY] = C.canonical_digest({
        key: value for key, value in changed["contract"].items()
        if key != C.CONTRACT_SELF_KEY
    })
    changed[C.CONTRACT_SELF_KEY] = changed["contract"][C.CONTRACT_SELF_KEY]
    changed[C.ARTIFACT_SELF_KEY] = C.canonical_digest({
        key: value for key, value in changed.items()
        if key != C.ARTIFACT_SELF_KEY
    })
    with pytest.raises(C.ScorerFitCorpusV2ContractError,
                       match="protected predecessor scorer field"):
        C.validate_contract_artifact(changed)


def test_exclusive_writer_is_read_only_durable_and_never_overwrites(tmp_path):
    path = tmp_path / "contract.json"
    payload = {"schema": "fixture", "value": 1}
    C._write_exclusive_json(path, payload)
    assert json.loads(path.read_text()) == payload
    assert path.stat().st_mode & 0o222 == 0
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        C._write_exclusive_json(path, {"schema": "replacement"})
    assert path.read_bytes() == original


def test_clean_source_projection_rejects_dirty_or_wrong_root(tmp_path):
    with pytest.raises(C.ScorerFitCorpusV2ContractError,
                       match="not clean"):
        C._validated_repository_state(
            root=tmp_path, commit="a" * 40, status=" M source.py",
            top_level=str(tmp_path))
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(C.ScorerFitCorpusV2ContractError,
                       match="custody root"):
        C._validated_repository_state(
            root=tmp_path, commit="a" * 40, status="",
            top_level=str(other))


def test_active_binding_projection_uses_producer_supplied_raw_bindings(monkeypatch):
    inputs = _inputs()
    authority = {
        "design_amendment": {"payload": "design"},
        "source_correction": {"payload": "source-correction"},
        "rotation_mask_classification": {"payload": "masks"},
        "design_amendment_binding": inputs["design_binding"],
        "source_correction_binding": inputs["source_correction_binding"],
        "rotation_mask_classification_binding":
            inputs["mask_classification_binding"],
    }
    manifests = {
        "selection_binding": inputs["selection_binding"],
        "revalidation_binding": inputs["revalidation_binding"],
        "state_manifest_binding": inputs["state_manifest_binding"],
        "assignment_manifest_binding": inputs["assignment_manifest_binding"],
    }
    projected = C._bindings_from_active_inputs(authority, manifests)
    assert projected == {
        key: inputs[key] for key in (
            "design_binding", "source_correction_binding",
            "mask_classification_binding",
            "selection_binding", "revalidation_binding",
            "state_manifest_binding", "assignment_manifest_binding",
        )
    }


def test_active_inputs_accept_exact_correction_aware_manifest_bundle(
        monkeypatch, tmp_path):
    from scripts import build_go2_branch_corpus_v1_2 as builder

    authority = {"authority": "fixture"}
    manifests = {
        key: ({"authority": "fixture"}
              if key == "design_authority" else f"fixture-{key}")
        for key in C._MANIFEST_BUNDLE_KEYS
    }
    monkeypatch.setattr(
        C.DESIGN, "load_active_design_authority",
        lambda **_kwargs: authority)
    monkeypatch.setattr(
        builder, "load_and_validate_full_bank_v2_manifests_for_consumption",
        lambda **_kwargs: manifests)
    loaded_authority, loaded_manifests = C._active_inputs(root=tmp_path)
    assert loaded_authority == authority
    assert loaded_manifests == manifests
