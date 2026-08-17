from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lewm.oracle import go2_attentive_metric_reconciliation_v1_contract as C


def _source() -> dict:
    payload = {
        "schema": C.SOURCE_SCHEMA,
        "source_repository_commit": "1" * 40,
        "source_repository_clean": True,
        "base_source_commit": C.BASE_SOURCE_COMMIT,
        "exact_committed_additive_path_diff": list(C.NEW_SOURCE_PATHS),
        "frozen_source_files": {},
        "additive_files": {},
    }
    return {**payload, C.SOURCE_SELF_KEY: C.digest(payload)}


def _lineage() -> dict:
    artifacts = {name: {"sha256": row["sha256"],
                        "byte_count": row["byte_count"]}
                 for name, row in C.ORIGINAL_ARTIFACTS.items()}
    return {
        "original_scientific_source_commit":
            C.SCIENTIFIC_ATTEMPT_LINEAGE["original_scientific_source_commit"],
        "installed_scientific_source_commit":
            C.SCIENTIFIC_ATTEMPT_LINEAGE["installed_scientific_source_commit"],
        "artifacts": artifacts,
        "artifact_set_digest": C.digest(artifacts),
        "failure": dict(C.FROZEN_FAILURE),
        "scientific_attempt_lineage": dict(C.SCIENTIFIC_ATTEMPT_LINEAGE),
    }


def _storage() -> dict:
    return {
        "logical_parent": str(C.GENERATED_PARENT),
        "registered_parent": str(C.REGISTERED_PARENT),
        "resolved_parent": str(C.REGISTERED_PARENT),
        "runtime_relative": str(C.RUNTIME_RELATIVE),
        "runtime_namespace_absent_before_issue": True,
    }


def test_contract_uses_only_the_two_authorised_classifications() -> None:
    assert C.TERMINAL_KINDS == (
        "INVALID_TECHNICAL_UNRECOVERABLE_METRIC_EVIDENCE",
        "POST_EVALUATION_METRIC_CONSUMER_DEFECT_RECOVERABLE",
    )
    assert C.REPAIRED_RESULT_LABEL == "POST_EVALUATION_CONSUMER_REPAIR"


def test_contract_binds_named_scientific_lineage_and_hash_only_tensors() -> None:
    lineage = C.SCIENTIFIC_ATTEMPT_LINEAGE
    assert lineage["original_scientific_source_commit"].startswith("89dde156")
    assert lineage["installed_scientific_source_commit"] == C.BASE_SOURCE_COMMIT
    assert lineage["scientific_source_closure_digest"].startswith("33a497e3")
    assert lineage["attentive_architecture_digest"].startswith("0c5edc71")
    assert lineage["optimizer_state_digest"].startswith("d079a489")
    assert "NOT_INDEPENDENTLY_REVERIFIED" in (
        lineage["optimizer_state_digest_provenance"])
    assert C.ORIGINAL_ARTIFACTS["initialisation.pt"]["content_access"] \
        == "HASH_BYTES_ONLY_NEVER_DESERIALISE"
    assert C.ORIGINAL_ARTIFACTS["final_checkpoint.pt"]["content_access"] \
        == "HASH_BYTES_ONLY_NEVER_DESERIALISE"


def test_static_contract_freezes_completeness_and_metric_contract() -> None:
    static = C.static_contract()
    assert static["expected_evidence_inventory"]["rows_missing_action_blocks"] == 288
    assert static["expected_evidence_inventory"][
        "frozen_calibration_manifest_or_state_family_mapping_retained"] is False
    assert static["metric_suite"]["grouping"] == "per_family"
    assert static["consumer_float_tolerances"] == {
        "absolute": 1e-10, "relative": 1e-9}
    assert static["original_gates"][
        "decimal_latent_over_baseline_pairwise_gain_min"] == "0.05"
    assert static["authority"]["torch_import_or_torch_load"] is False
    assert static["authority"]["publish_attentive_metrics_when_evidence_incomplete"] is False


def test_source_reconstruction_binds_first_row_and_tie_witnesses() -> None:
    source = C.SOURCE_RECONSTRUCTION
    assert source["first_divergent_row"]["row_index"] == 0
    assert source["first_divergent_row"]["direct_absolute_error"] \
        == 0.36313265562057495
    assert source["first_overall_metric_leaf"]["path"] == "progress.mae"
    assert source["first_tie_induced_state"]["state_id"] \
        == "oracle_v1_3-calibration-local_composite_motifs-general"
    assert source["first_tie_induced_state"]["direct_pairs_considered"] == 62
    assert source["first_tie_induced_state"]["replay_pairs_considered"] == 63


def test_contract_round_trip_and_mutation_rejection() -> None:
    contract = C.build_contract(_source(), _lineage(), _storage())
    assert C.validate_contract(contract) == contract
    changed = copy.deepcopy(contract)
    changed["authority"]["model_construction_or_forward"] = True
    body = {key: value for key, value in changed.items()
            if key != C.CONTRACT_SELF_KEY}
    changed[C.CONTRACT_SELF_KEY] = C.digest(body)
    with pytest.raises(C.MetricReconciliationContractError):
        C.validate_contract(changed)


def test_additive_paths_are_exact_and_runtime_namespace_is_new_sibling() -> None:
    assert len(C.NEW_SOURCE_PATHS) == 4
    assert len(set(C.NEW_SOURCE_PATHS)) == 4
    assert C.RUNTIME_RELATIVE.name == "attentive_metric_reconciliation_v1"
    assert C.RUNTIME_RELATIVE.parent == C.GENERATED_PARENT

