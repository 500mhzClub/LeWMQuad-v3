"""Pure provenance-order tests for the scorer-fit latent encoder."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import encode_go2_branch_corpus_v1_2 as encoder


class BranchCorpusEncoderProvenanceTests(unittest.TestCase):
    def test_target_encoder_compute_dtype_is_fp32_on_cpu_and_cuda(self):
        self.assertEqual(
            encoder.TARGET_ENCODER_COMPUTE_DTYPE_NAME,
            "float32",
        )
        self.assertIs(
            encoder.target_encoder_compute_dtype(
                encoder.torch.device("cpu"), full_bank_v2=True),
            encoder.torch.float32,
        )
        self.assertIs(
            encoder.target_encoder_compute_dtype(
                encoder.torch.device("cuda:0"), full_bank_v2=True),
            encoder.torch.float32,
        )

    def test_legacy_compute_dtype_policy_is_unchanged_on_cpu_and_cuda(self):
        self.assertIs(
            encoder.target_encoder_compute_dtype(
                encoder.torch.device("cpu"), full_bank_v2=False),
            encoder.torch.float32,
        )
        self.assertIs(
            encoder.target_encoder_compute_dtype(
                encoder.torch.device("cuda:0"), full_bank_v2=False),
            encoder.torch.bfloat16,
        )

    def test_full_bank_dtype_binding_rejects_missing_or_bf16_lineage(self):
        encoder._validate_target_encoder_compute_dtype(  # noqa: SLF001
            "float32", label="fixture")
        for value in (None, "bfloat16"):
            with self.assertRaisesRegex(
                    RuntimeError, "target-encoder compute dtype changed"):
                encoder._validate_target_encoder_compute_dtype(  # noqa: SLF001
                    value, label="fixture")

    def test_dtype_correction_digest_binding_is_exact(self):
        digest = "e" * 64
        self.assertEqual(
            encoder._validate_encoder_compute_dtype_correction_digest(  # noqa: SLF001
                digest, label="fixture"),
            digest,
        )
        for value in (None, "e" * 63, "g" * 64):
            with self.assertRaisesRegex(
                    RuntimeError, "correction digest changed"):
                encoder._validate_encoder_compute_dtype_correction_digest(  # noqa: SLF001
                    value, label="fixture")

    def test_full_bank_v2_input_route_uses_only_exact_v2_producers(self):
        bindings = {key: "a" * 64
                    for key in encoder.FULL_BANK_V2_BINDING_KEYS}
        bindings["scorer_fit_corpus_v2_scorer_contract_digest"] = "c" * 64
        bindings[
            "scorer_fit_corpus_v2_scorer_contract_artifact_digest"] = "d" * 64
        bindings["target_encoder_checkpoint_sha256"] = (
            encoder.contract()["target_encoder"]["checkpoint_sha256"])
        manifest = {
            "schema":
                encoder.CORPUS_BUILDER.SCORER_FIT_V2_STATE_MANIFEST_SCHEMA,
            "pool": "scorer_fit_v2",
            "state_manifest_digest": "b" * 64,
            "attempted_branch_count_registered": 1_440,
            "states": [
                {"candidate_indices": list(range(12))}
                for _ in range(120)
            ],
            **bindings,
        }
        rows = []
        for candidate_index in range(12):
            row = {
                "state_id": "smoke-state",
                "candidate": f"candidate-{candidate_index}",
                "candidate_index": candidate_index,
                "valid": True,
                "state_manifest_digest": manifest["state_manifest_digest"],
                **bindings,
            }
            row["branch_row_digest"] = encoder.canonical_digest(row)
            rows.append(row)
        successor = {
            "preoutcome_lineage": {
                "scorer_fit_corpus_v2_source_correction_digest": bindings[
                    "scorer_fit_corpus_v2_source_correction_digest"],
            },
            "state_selector_binding": {
                "state_manifest_digest": manifest["state_manifest_digest"],
                "assignment_manifest_digest": manifest[
                    "full_bank_assignment_manifest_digest"],
            },
            "protected_predecessor_scientific_contract": {
                "target_encoder": encoder.contract()["target_encoder"],
            },
            encoder.V2_CONTRACT.CONTRACT_SELF_KEY: "c" * 64,
        }
        artifact = {
            "contract": successor,
            encoder.V2_CONTRACT.CONTRACT_SELF_KEY: "c" * 64,
            encoder.V2_CONTRACT.ARTIFACT_SELF_KEY: "d" * 64,
        }
        dtype_correction = {
            encoder.V2_DESIGN.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY:
                "e" * 64,
        }
        bundle = {
            "manifest": manifest,
            "receipt": {"complete": False},
            "rows": rows,
            "scorer_contract": artifact,
        }
        with mock.patch.object(
                encoder.V2_DESIGN, "load_active_design_authority",
                return_value={
                    "source_correction_digest": bindings[
                        "scorer_fit_corpus_v2_source_correction_digest"],
                    "encoder_compute_dtype_correction_digest": "e" * 64,
                    "encoder_compute_dtype_correction": dtype_correction,
                }), \
                mock.patch.object(
                encoder.CORPUS_BUILDER,
                "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
                return_value=bundle) as producer, \
                mock.patch.object(
                    encoder.V2_CONTRACT, "load_contract_for_consumption",
                    return_value=artifact) as contract_loader, \
                mock.patch.object(
                    encoder.V2_CONTRACT, "validate_contract_artifact",
                    return_value=artifact), \
                mock.patch.object(
                    encoder.ALLOC, "allocation_contract_digest",
                    side_effect=AssertionError("legacy ALLOC opened")) as alloc, \
                mock.patch.object(
                    encoder, "_load_inputs",
                    side_effect=AssertionError("legacy route opened")) as legacy:
            observed = encoder._load_full_bank_v2_inputs(
                encoder.OUT_ROOT / "scorer_fit", allow_partial=True)
        producer.assert_called_once_with(
            out=encoder.OUT_ROOT / "scorer_fit", allow_partial=True)
        contract_loader.assert_called_once_with(
            root=encoder.ROOT,
            encoder_compute_dtype_correction=dtype_correction)
        alloc.assert_not_called()
        legacy.assert_not_called()
        self.assertEqual(observed[0], manifest)
        self.assertEqual(len(observed[2]), 12)
        self.assertEqual(observed[4], "e" * 64)

    def test_full_bank_v2_source_correction_mismatch_precedes_branch_producer(
            self):
        artifact = {
            "contract": {
                "preoutcome_lineage": {
                    "scorer_fit_corpus_v2_source_correction_digest": "b" * 64,
                },
            },
        }
        dtype_correction = {
            encoder.V2_DESIGN.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY:
                "e" * 64,
        }
        with mock.patch.object(
                encoder.V2_DESIGN, "load_active_design_authority",
                return_value={
                    "source_correction_digest": "a" * 64,
                    "encoder_compute_dtype_correction_digest": "e" * 64,
                    "encoder_compute_dtype_correction": dtype_correction,
                }), \
                mock.patch.object(
                    encoder.V2_CONTRACT, "load_contract_for_consumption",
                    return_value=artifact), \
                mock.patch.object(
                    encoder.V2_CONTRACT, "validate_contract_artifact",
                    return_value=artifact), \
                mock.patch.object(
                    encoder.CORPUS_BUILDER,
                    "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
                    side_effect=AssertionError("branch producer opened")) as producer:
            with self.assertRaisesRegex(
                    RuntimeError, "source-correction lineage changed"):
                encoder._load_full_bank_v2_inputs(
                    encoder.OUT_ROOT / "scorer_fit", allow_partial=True)
        producer.assert_not_called()

    def test_missing_dtype_correction_precedes_contract_and_branch_producer(
            self):
        with mock.patch.object(
                encoder.V2_DESIGN, "load_active_design_authority",
                return_value={"source_correction_digest": "a" * 64}), \
                mock.patch.object(
                    encoder.V2_CONTRACT, "load_contract_for_consumption",
                    side_effect=AssertionError("contract opened")) as contract, \
                mock.patch.object(
                    encoder.CORPUS_BUILDER,
                    "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
                    side_effect=AssertionError("branch producer opened")) as producer:
            with self.assertRaisesRegex(
                    RuntimeError,
                    "active encoder-compute-dtype correction is unavailable"):
                encoder._load_full_bank_v2_inputs(  # noqa: SLF001
                    encoder.OUT_ROOT / "scorer_fit", allow_partial=True)
        contract.assert_not_called()
        producer.assert_not_called()

    def test_full_bank_v2_output_registry_is_versioned_and_disjoint(self):
        self.assertEqual(encoder.FULL_BANK_V2_INDEX_NAME,
                         "latents_index_v2.json")
        self.assertEqual(encoder.FULL_BANK_V2_SMOKE_NAME,
                         "smoke_encoding_receipt_v2.json")
        self.assertEqual(encoder.FULL_BANK_V2_LATENTS_NAME, "latents_v2")
        self.assertNotIn("candidate_allocator_contract_digest",
                         encoder.FULL_BANK_V2_BINDING_KEYS)
        self.assertIn("scorer_fit_corpus_v2_source_correction_digest",
                      encoder.FULL_BANK_V2_BINDING_KEYS)

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
