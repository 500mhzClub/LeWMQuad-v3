"""Pure provenance-order tests for the scorer-fit latent encoder."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import encode_go2_branch_corpus_v1_2 as encoder


class BranchCorpusEncoderProvenanceTests(unittest.TestCase):
    def test_branch_row_remains_bound_to_manifest_scientific_contract(self):
        historical = "1" * 64
        manifest = {
            "state_manifest_digest": "2" * 64,
            **{key: "3" * 64 for key in encoder.CORPUS_BINDING_KEYS},
            "scorer_contract_v1_2_digest": historical,
        }
        row = {
            "state_id": "fixture-state",
            "candidate": "fixture-candidate",
            "state_manifest_digest": manifest["state_manifest_digest"],
            **{key: manifest[key] for key in encoder.CORPUS_BINDING_KEYS},
        }
        row["branch_row_digest"] = encoder.canonical_digest(row)
        encoder._validate_row(row, manifest, historical)
        with self.assertRaisesRegex(RuntimeError, "scorer_contract_v1_2_digest"):
            encoder._validate_row(row, manifest, encoder.contract_digest())

    def test_global_exact_manifest_uses_successor_operational_contract(self):
        historical_contract = "0" * 64
        current_contract = encoder.contract_digest()
        predecessor = {
            "clean_source_launch_receipt_digest": "1" * 64,
            "source_repository_commit": "2" * 40,
            "clean_source_binding_digest": "3" * 64,
            "bound_implementations_digest": "4" * 64,
            "scorer_contract_artifact_digest": "5" * 64,
        }
        successor = {
            "clean_source_launch_receipt_digest": "a" * 64,
            "source_repository_commit": "b" * 40,
            "clean_source_binding_digest": "c" * 64,
            "bound_implementations_digest": "d" * 64,
            "scorer_contract_artifact_digest": "e" * 64,
            "clean_source_launch_receipt_sha256": "6" * 64,
            "scorer_contract_artifact_sha256": "7" * 64,
            "launch_state_selector_feasibility_receipt_digest": "8" * 64,
            "mixed_precontract_disposition_receipt_digest": "9" * 64,
            "global_exact_execution_amendment_digest": "a" * 64,
            "global_exact_successor_scorer_contract_digest": "b" * 64,
            "current_scorer_contract_v1_2_digest": current_contract,
            "scientific_predecessor_launch_bindings": predecessor,
        }
        manifest = {
            "small_completion_global_exact_execution": {},
            **predecessor,
            "mixed_precontract_disposition_receipt_digest": "9" * 64,
            "scorer_contract_v1_2_digest": historical_contract,
        }
        with mock.patch.object(
                encoder.CORPUS_BUILDER,
                "load_global_exact_successor_scorer_contract_for_consumption",
                return_value=successor, create=True) as load_successor, \
                mock.patch.object(
                    encoder, "_load_clean_source_launch_receipt") as legacy:
            operational, scientific, selector = (
                encoder._load_manifest_launch_lineage(manifest))
        load_successor.assert_called_once_with(manifest)
        legacy.assert_not_called()
        self.assertEqual(
            operational["current_scorer_contract_v1_2_digest"],
            current_contract)
        self.assertEqual(
            operational["global_exact_scorer_contract_lineage"], {
                "schema":
                    encoder.GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_SCHEMA,
                "scientific_predecessor_scorer_contract_v1_2_digest":
                    historical_contract,
                "current_scorer_contract_v1_2_digest": current_contract,
                "global_exact_successor_scorer_contract_digest": "b" * 64,
            })
        self.assertEqual(
            {key: scientific[key]
             for key in encoder.SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS},
            predecessor)
        self.assertEqual(
            selector["launch_state_selector_feasibility_receipt_digest"],
            "8" * 64)

        malformed = dict(
            operational["global_exact_scorer_contract_lineage"])
        malformed["unexpected"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "schema is not closed"):
            encoder._validate_global_exact_scorer_contract_lineage(malformed)

    def test_live_selection_replay_failure_precedes_rows_and_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "scorer_fit"
            out.mkdir()
            manifest = {"pool": "scorer_fit", "states": []}
            (out / "state_manifest.json").write_text(json.dumps(manifest))
            with mock.patch.object(
                    encoder.CORPUS_BUILDER,
                    "load_active_state_manifest_for_consumption",
                    side_effect=RuntimeError(
                        "later replacement capture prefix is not canonical"
                    )) as replay, \
                    mock.patch.object(
                        encoder, "_load_selector_successor_receipts"
                    ) as selector, \
                    mock.patch.object(encoder, "_verify_frames") as frames, \
                    mock.patch.object(encoder.json, "loads") as raw_json:
                with self.assertRaisesRegex(
                        RuntimeError, "later replacement capture prefix"):
                    encoder._load_inputs(
                        out, allow_partial=False, pool="scorer_fit")
            replay.assert_called_once_with(
                out / "state_manifest.json", pool="scorer_fit")
            selector.assert_not_called()
            frames.assert_not_called()
            raw_json.assert_not_called()

    def test_global_selector_replay_never_calls_legacy_allocation_validator(self):
        feasibility = {
            "state_selector_feasibility_receipt_digest": "f" * 64}
        disposition = {
            "mixed_precontract_disposition_receipt_digest": "d" * 64}
        revalidation = {
            "preserved_state_revalidation_receipt_digest": "e" * 64}
        allocation = {"allocation_manifest_digest": "1" * 64}
        manifest = {"small_completion_global_exact_execution": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out_root = root / ".generated/go2_branch_corpus_v1_2"
            scorer_fit = out_root / "scorer_fit"
            scorer_fit.mkdir(parents=True)
            (root / encoder.STATE_SELECTOR.
             STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH).parent.mkdir(
                 parents=True, exist_ok=True)
            (root / encoder.STATE_SELECTOR.
             STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH).write_text(
                 json.dumps(feasibility))
            (root / encoder.STATE_SELECTOR.
             PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH
             ).write_text(json.dumps(disposition))
            (root / encoder.STATE_SELECTOR.
             PRESERVED_STATE_REVALIDATION_RECEIPT_PATH).write_text(
                 json.dumps(revalidation))
            (scorer_fit / "candidate_allocation_manifest.json").write_text(
                json.dumps(allocation))
            with mock.patch.object(encoder, "ROOT", root), \
                    mock.patch.object(encoder, "OUT_ROOT", out_root), \
                    mock.patch.object(
                        encoder.STATE_SELECTOR,
                        "validate_authority_artifacts"), \
                    mock.patch.object(
                        encoder.STATE_SELECTOR,
                        "validate_frozen_reachability_feasibility_pass",
                        return_value=feasibility), \
                    mock.patch.object(
                        encoder.STATE_SELECTOR,
                        "validate_preserved_state_mixed_precontract_disposition_receipt"), \
                    mock.patch.object(
                        encoder.STATE_SELECTOR,
                        "validate_preserved_state_revalidation_receipt") as legacy, \
                    mock.patch.object(
                        encoder.CORPUS_BUILDER,
                        "validate_global_exact_allocation_for_consumption",
                        return_value={
                            "preserved_state_revalidation_receipt_digest":
                                "e" * 64,
                        }) as certify:
                bindings = encoder._load_selector_successor_receipts(
                    source_commit="c" * 40,
                    selection_digest="b" * 64,
                    active_states=[],
                    expected_feasibility_receipt_digest="f" * 64,
                    expected_mixed_precontract_disposition_receipt_digest=
                        "d" * 64,
                    expected_clean_source_binding_digest="a" * 64,
                    expected_bound_implementations_digest="9" * 64,
                    global_exact_manifest=manifest)
            certify.assert_called_once_with(manifest, allocation)
            legacy.assert_not_called()
            self.assertEqual(
                bindings["preserved_state_revalidation_receipt_digest"],
                "e" * 64)


if __name__ == "__main__":
    unittest.main()
