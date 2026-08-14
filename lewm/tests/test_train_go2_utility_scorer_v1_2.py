"""Focused non-sealed tests for the v1.2 utility-scorer trainer."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from scripts import train_go2_utility_scorer_v1_2 as scorer


class UtilityScorerTrainerTests(unittest.TestCase):
    def test_global_exact_launch_separates_scientific_and_operational_source(self):
        historical_contract = "0" * 64
        current_contract = scorer.contract_digest()
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
        pre_identity = {"pre_identity_validation_digest": "f" * 64}
        manifest = {
            "small_completion_global_exact_execution": {},
            **predecessor,
            "mixed_precontract_disposition_receipt_digest": "9" * 64,
            "pre_identity_allocation_validation_digest": "f" * 64,
            "scorer_contract_v1_2_digest": historical_contract,
        }
        with mock.patch.object(
                scorer.CORPUS_BUILDER,
                "load_global_exact_successor_scorer_contract_for_consumption",
                return_value=successor, create=True) as load_successor, \
                mock.patch.object(
                    scorer, "_validate_clean_source_launch") as legacy:
            operational, scientific, selector = (
                scorer._load_manifest_launch_lineage(
                    manifest, Path("/unused"), pre_identity))
        load_successor.assert_called_once_with(manifest)
        legacy.assert_not_called()
        self.assertEqual(operational["source_repository_commit"], "b" * 40)
        self.assertEqual(scientific["source_repository_commit"], "2" * 40)
        self.assertEqual(selector["source_repository_commit"], "2" * 40)
        self.assertEqual(
            operational["global_exact_scorer_contract_lineage"]
            ["scientific_predecessor_scorer_contract_v1_2_digest"],
            historical_contract)
        self.assertEqual(
            scorer.operational_scorer_contract_digest(operational),
            current_contract)
        self.assertEqual(
            set(scorer.scorer_provenance_binding_keys(operational)),
            set(scorer.SCORER_PROVENANCE_BINDING_KEYS)
            | set(scorer.GLOBAL_EXACT_PROVENANCE_BINDING_KEYS))

        changed = dict(manifest)
        changed["source_repository_commit"] = "0" * 40
        with mock.patch.object(
                scorer.CORPUS_BUILDER,
                "load_global_exact_successor_scorer_contract_for_consumption",
                return_value=successor, create=True):
            with self.assertRaisesRegex(
                    scorer.CorpusValidationError,
                    "manifest scientific launch differs"):
                scorer._load_manifest_launch_lineage(
                    changed, Path("/unused"), pre_identity)

        malformed = dict(
            operational["global_exact_scorer_contract_lineage"])
        malformed["unexpected"] = "f" * 64
        with self.assertRaisesRegex(
                scorer.CorpusValidationError, "schema is not closed"):
            scorer.validate_global_exact_scorer_contract_lineage(malformed)

    def test_live_selection_replay_failure_precedes_rows_latents_and_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = root / scorer.EXPECTED_POOL
            pool.mkdir()
            for name in (
                "state_manifest.json",
                "pre_identity_allocation_validation.json",
                "candidate_allocation_manifest.json",
                "branch_rows.jsonl",
                "corpus_receipt.json",
                "latents_index.json",
            ):
                (pool / name).write_text("{}")
            with mock.patch.object(scorer, "OUT_ROOT", root), \
                    mock.patch.object(
                        scorer.CORPUS_BUILDER,
                        "load_active_state_manifest_for_consumption",
                        side_effect=RuntimeError(
                            "later small-family passing combination"
                        )) as replay, \
                    mock.patch.object(scorer, "_parse_rows") as parse_rows, \
                    mock.patch.object(scorer, "_validate_latent_index") as latents, \
                    mock.patch.object(scorer, "UtilityScorer") as model:
                with self.assertRaisesRegex(
                        scorer.CorpusValidationError,
                        "later small-family passing combination"):
                    scorer.validate_scorer_fit_corpus(
                        verify_encoder_checkpoint=False,
                        verify_frame_paths=False,
                    )
            replay.assert_called_once_with(
                pool / "state_manifest.json", pool=scorer.EXPECTED_POOL)
            parse_rows.assert_not_called()
            latents.assert_not_called()
            model.assert_not_called()

    def test_clean_source_launch_revalidated_at_training_boundary(self):
        source = {
            "source_repository_commit": "c" * 40,
            "bound_implementations_digest": "b" * 64,
        }
        pre_identity = {"pre_identity_validation_digest": "i" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = root / "scorer_fit"
            pool.mkdir()
            artifact_path = root / "scorer_contract_v1_2.json"
            artifact = {
                "schema": "go2_utility_scorer_contract_v1_2_artifact",
                "complete": True,
                "source_repository_clean": True,
                "state_selector_amendment_verified": True,
                "state_selector_feasibility_verified": True,
                "preserved_state_mixed_precontract_disposition_verified": True,
                "state_selector_feasibility_receipt_digest": "f" * 64,
                "mixed_precontract_disposition_receipt_digest": "d" * 64,
                "mixed_state_post_allocation_revalidation": {
                    "status": "PENDING_POST_IDENTITY_PRE_OUTCOME",
                    "required_before_active_identity_manifest": True,
                    "schema": scorer.STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_SCHEMA,
                    "path":
                        scorer.STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH,
                    "realized_receipt_digest_bound_at_contract_issue": False,
                },
                "scorer_contract_v1_2_digest": scorer.contract_digest(),
                "contract": scorer.contract(),
                "clean_source_binding": source,
                "clean_source_binding_digest": scorer.canonical_digest(source),
            }
            artifact["contract_artifact_digest"] = scorer.canonical_digest(artifact)
            artifact_path.write_text(json.dumps(artifact))
            launch = {
                "schema": "go2_utility_scorer_v1_2_clean_source_launch_receipt",
                "complete": True,
                "source_repository_clean": True,
                "source_repository_commit": source["source_repository_commit"],
                "clean_source_binding_digest": scorer.canonical_digest(source),
                "bound_implementations_digest":
                    source["bound_implementations_digest"],
                "scorer_contract_v1_2_digest": scorer.contract_digest(),
                "scorer_contract_artifact_digest":
                    artifact["contract_artifact_digest"],
                "scorer_contract_artifact_sha256":
                    scorer.sha256_file(artifact_path),
                "candidate_allocation_amendment_digest":
                    scorer.allocation_amendment_digest(),
                "invalid_scorer_identity_exclusion_digest":
                    scorer.invalid_identity_exclusion_digest(),
                "state_selector_amendment_digest":
                    scorer.STATE_SELECTOR.state_selector_amendment_digest(),
                "state_selector_feasibility_receipt_digest": "f" * 64,
                "mixed_precontract_disposition_receipt_digest": "d" * 64,
                "pre_identity_allocation_validation_digest":
                    pre_identity["pre_identity_validation_digest"],
            }
            launch["clean_source_launch_receipt_digest"] = (
                scorer.canonical_digest(launch))
            (pool / "clean_source_launch_receipt.json").write_text(
                json.dumps(launch))
            with mock.patch.object(
                    scorer, "SCORER_CONTRACT_ARTIFACT_PATH", artifact_path), \
                    mock.patch.object(
                        scorer, "clean_source_binding", return_value=source):
                bindings = scorer._validate_clean_source_launch(
                    pool, pre_identity)
            self.assertEqual(bindings["source_repository_commit"], "c" * 40)
            self.assertEqual(
                bindings["scorer_contract_artifact_digest"],
                artifact["contract_artifact_digest"])

    def test_both_current_smoke_receipts_are_required_and_bound(self):
        manifest = {
            "state_manifest_digest": "d" * 64,
            "candidate_allocation_post_identity_validation_digest": "p" * 64,
            "pre_identity_allocation_validation_digest": "i" * 64,
            "clean_source_launch_receipt_digest": "l" * 64,
            "source_repository_commit": "c" * 40,
            "clean_source_binding_digest": "s" * 64,
            "bound_implementations_digest": "b" * 64,
            "scorer_contract_artifact_digest": "a" * 64,
            "mixed_precontract_disposition_receipt_digest": "m" * 64,
            "state_selector_amendment_digest": "a" * 64,
            "state_selector_feasibility_receipt_digest": "f" * 64,
            "preserved_state_revalidation_receipt_digest": "e" * 64,
        }
        frozen = scorer.contract()
        common = {
            "pass": True,
            "state_manifest_digest": manifest["state_manifest_digest"],
            "scorer_contract_v1_2_digest": scorer.contract_digest(),
            "candidate_allocator_contract_digest":
                scorer.allocation_contract_digest(),
            "candidate_allocation_amendment_digest":
                scorer.allocation_amendment_digest(),
            "candidate_allocation_post_identity_validation_digest": "p" * 64,
            "pre_identity_allocation_validation_digest": "i" * 64,
            "invalid_scorer_identity_exclusion_digest":
                scorer.invalid_identity_exclusion_digest(),
            **{key: manifest[key] for key in scorer.SELECTOR_BINDING_KEYS},
            **{key: manifest[key] for key in scorer.LAUNCH_BINDING_KEYS},
            "render_contract_digest": scorer.canonical_digest(
                frozen["render_contract"]),
            "textured_v03_renderer_contract_digest":
                scorer.textured_v03_renderer_contract_digest(),
            "preprocess_contract_digest": scorer.canonical_digest(
                frozen["preprocess_contract"]),
            "preprocessing_digest": scorer.FROZEN_PREPROCESSING_DIGEST,
            "target_encoder_digest": scorer.canonical_digest(
                frozen["target_encoder"]),
            "target_encoder_checkpoint_sha256":
                frozen["target_encoder"]["checkpoint_sha256"],
        }
        specifications = (
            ("smoke_branch_receipt.json", "smoke_branch_receipt_digest",
             "go2_scorer_fit_branch_smoke_receipt_v1_2"),
            ("smoke_encoding_receipt.json", "smoke_receipt_digest",
             "go2_scorer_fit_end_to_end_smoke_receipt_v1"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, digest_key, schema in specifications:
                payload = {"schema": schema, **common}
                if filename == "smoke_encoding_receipt.json":
                    payload["corpus_bound_digests"] = {
                        key: payload.pop(key) for key in scorer.LAUNCH_BINDING_KEYS
                    }
                payload[digest_key] = scorer.canonical_digest(payload)
                (root / filename).write_text(json.dumps(payload))
            verified = scorer._validate_smoke_receipts(root, manifest)
            self.assertEqual(set(verified), {item[0] for item in specifications})
            (root / "smoke_encoding_receipt.json").unlink()
            with self.assertRaisesRegex(scorer.CorpusValidationError,
                                        "missing required end-to-end smoke receipt"):
                scorer._validate_smoke_receipts(root, manifest)

    def test_global_smoke_separates_historical_branch_from_current_encoding(self):
        historical = "0" * 64
        current = scorer.contract_digest()
        successor = "9" * 64
        lineage = {
            "schema": scorer.GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_SCHEMA,
            "scientific_predecessor_scorer_contract_v1_2_digest": historical,
            "current_scorer_contract_v1_2_digest": current,
            "global_exact_successor_scorer_contract_digest": successor,
        }
        manifest = {
            "small_completion_global_exact_execution": {},
            "state_manifest_digest": "d" * 64,
            "scorer_contract_v1_2_digest": historical,
            "candidate_allocation_post_identity_validation_digest": "1" * 64,
            "pre_identity_allocation_validation_digest": "2" * 64,
            "clean_source_launch_receipt_digest": "3" * 64,
            "source_repository_commit": "4" * 40,
            "clean_source_binding_digest": "5" * 64,
            "bound_implementations_digest": "6" * 64,
            "scorer_contract_artifact_digest": "7" * 64,
            "mixed_precontract_disposition_receipt_digest": "8" * 64,
            "state_selector_amendment_digest": "a" * 64,
            "state_selector_feasibility_receipt_digest": "b" * 64,
            "preserved_state_revalidation_receipt_digest": "c" * 64,
        }
        frozen = scorer.contract()
        common = {
            "pass": True,
            "state_manifest_digest": manifest["state_manifest_digest"],
            "candidate_allocator_contract_digest":
                scorer.allocation_contract_digest(),
            "candidate_allocation_amendment_digest":
                scorer.allocation_amendment_digest(),
            "candidate_allocation_post_identity_validation_digest": "1" * 64,
            "pre_identity_allocation_validation_digest": "2" * 64,
            "invalid_scorer_identity_exclusion_digest":
                scorer.invalid_identity_exclusion_digest(),
            **{key: manifest[key] for key in scorer.SELECTOR_BINDING_KEYS},
            "render_contract_digest": scorer.canonical_digest(
                frozen["render_contract"]),
            "textured_v03_renderer_contract_digest":
                scorer.textured_v03_renderer_contract_digest(),
            "preprocess_contract_digest": scorer.canonical_digest(
                frozen["preprocess_contract"]),
            "preprocessing_digest": scorer.FROZEN_PREPROCESSING_DIGEST,
            "target_encoder_digest": scorer.canonical_digest(
                frozen["target_encoder"]),
            "target_encoder_checkpoint_sha256":
                frozen["target_encoder"]["checkpoint_sha256"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            branch = {
                "schema": "go2_scorer_fit_branch_smoke_receipt_v1_2",
                **common,
                **{key: manifest[key] for key in scorer.LAUNCH_BINDING_KEYS},
                "scorer_contract_v1_2_digest": historical,
            }
            branch["smoke_branch_receipt_digest"] = scorer.canonical_digest(
                branch)
            encoding = {
                "schema": "go2_scorer_fit_end_to_end_smoke_receipt_v1",
                **common,
                "scorer_contract_v1_2_digest": current,
                "global_exact_scorer_contract_lineage": lineage,
                "corpus_bound_digests": {
                    key: manifest[key] for key in scorer.LAUNCH_BINDING_KEYS
                },
            }
            encoding["smoke_receipt_digest"] = scorer.canonical_digest(encoding)
            (root / "smoke_branch_receipt.json").write_text(json.dumps(branch))
            (root / "smoke_encoding_receipt.json").write_text(json.dumps(encoding))
            scorer._validate_smoke_receipts(
                root, manifest, contract_lineage=lineage)

            encoding["scorer_contract_v1_2_digest"] = historical
            encoding["smoke_receipt_digest"] = scorer.canonical_digest({
                key: value for key, value in encoding.items()
                if key != "smoke_receipt_digest"
            })
            (root / "smoke_encoding_receipt.json").write_text(json.dumps(encoding))
            with self.assertRaisesRegex(
                    scorer.CorpusValidationError, "scorer-contract role"):
                scorer._validate_smoke_receipts(
                    root, manifest, contract_lineage=lineage)

    def test_selector_successor_requires_complete_two_phase_outcome_free_chain(self):
        feasibility = {
            "state_selector_feasibility_receipt_digest": "f" * 64,
        }
        disposition = {
            "mixed_precontract_disposition_receipt_digest": "d" * 64,
        }
        revalidation = {
            "preserved_state_revalidation_receipt_digest": "e" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / scorer.STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME
             ).write_text(json.dumps(feasibility))
            disposition_path = (
                root
                / scorer.STATE_SELECTOR.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_NAME)
            disposition_path.write_text(json.dumps(disposition))
            revalidation_path = (
                root
                / scorer.STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_NAME)
            revalidation_path.write_text(json.dumps(revalidation))
            with mock.patch.object(
                    scorer.STATE_SELECTOR, "validate_authority_artifacts"), \
                    mock.patch.object(
                        scorer.STATE_SELECTOR,
                        "validate_frozen_reachability_feasibility_pass",
                        return_value=feasibility), \
                    mock.patch.object(
                        scorer.STATE_SELECTOR,
                        "validate_preserved_state_mixed_precontract_disposition_receipt"
                    ) as validate_mixed_disposition, \
                    mock.patch.object(
                        scorer.STATE_SELECTOR,
                        "validate_preserved_state_revalidation_receipt"
                    ) as validate_final_revalidation, \
                    mock.patch.object(
                        scorer.STATE_SELECTOR, "state_selector_amendment_digest",
                        return_value="a" * 64):
                bindings = scorer._validate_selector_successor(
                    root, {
                        "source_repository_commit": "c" * 40,
                        "clean_source_binding_digest": "b" * 64,
                        "bound_implementations_digest": "a" * 64,
                        "launch_state_selector_feasibility_receipt_digest":
                            "f" * 64,
                        "mixed_precontract_disposition_receipt_digest":
                            "d" * 64,
                    }, {}, [])
                self.assertEqual(bindings, {
                    "state_selector_amendment_digest": "a" * 64,
                    "state_selector_feasibility_receipt_digest": "f" * 64,
                    "preserved_state_revalidation_receipt_digest": "e" * 64,
                })
                validate_mixed_disposition.assert_called_once()
                validate_final_revalidation.assert_called_once()
                final_kwargs = validate_final_revalidation.call_args.kwargs
                self.assertEqual(final_kwargs["allocation_manifest"], {})
                self.assertEqual(
                    final_kwargs[
                        "expected_mixed_precontract_disposition_receipt_digest"],
                    "d" * 64)
                self.assertEqual(final_kwargs["active_states"], [])
                revalidation_path.unlink()
                with self.assertRaisesRegex(
                        scorer.CorpusValidationError,
                        "missing preserved-state revalidation receipt"):
                    scorer._validate_selector_successor(
                        root, {
                            "source_repository_commit": "c" * 40,
                            "launch_state_selector_feasibility_receipt_digest":
                                "f" * 64,
                            "mixed_precontract_disposition_receipt_digest":
                                "d" * 64,
                        }, {}, [])
                disposition_path.write_text(json.dumps(disposition))
                revalidation_path.write_text(json.dumps(revalidation))
                disposition_path.unlink()
                with self.assertRaisesRegex(
                        scorer.CorpusValidationError,
                        "missing preserved-state mixed precontract disposition"):
                    scorer._validate_selector_successor(
                        root, {
                            "source_repository_commit": "c" * 40,
                            "launch_state_selector_feasibility_receipt_digest":
                                "f" * 64,
                            "mixed_precontract_disposition_receipt_digest":
                                "d" * 64,
                        }, {}, [])

    def test_global_selector_successor_uses_solve_free_model_certificate(self):
        feasibility = {
            "state_selector_feasibility_receipt_digest": "f" * 64}
        disposition = {
            "mixed_precontract_disposition_receipt_digest": "d" * 64}
        revalidation = {
            "preserved_state_revalidation_receipt_digest": "e" * 64}
        manifest = {"small_completion_global_exact_execution": {}}
        allocation = {"allocation_manifest_digest": "1" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / scorer.STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME
             ).write_text(json.dumps(feasibility))
            (root / scorer.STATE_SELECTOR.
             PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_NAME
             ).write_text(json.dumps(disposition))
            (root / scorer.STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_NAME
             ).write_text(json.dumps(revalidation))
            with mock.patch.object(
                    scorer.STATE_SELECTOR, "validate_authority_artifacts"), \
                    mock.patch.object(
                        scorer.STATE_SELECTOR,
                        "validate_frozen_reachability_feasibility_pass",
                        return_value=feasibility), \
                    mock.patch.object(
                        scorer.STATE_SELECTOR,
                        "validate_preserved_state_mixed_precontract_disposition_receipt"), \
                    mock.patch.object(
                        scorer.STATE_SELECTOR,
                        "validate_preserved_state_revalidation_receipt") as legacy, \
                    mock.patch.object(
                        scorer.STATE_SELECTOR, "state_selector_amendment_digest",
                        return_value="a" * 64), \
                    mock.patch.object(
                        scorer.CORPUS_BUILDER,
                        "validate_global_exact_allocation_for_consumption",
                        return_value={
                            "preserved_state_revalidation_receipt_digest":
                                "e" * 64,
                        }) as certify:
                bindings = scorer._validate_selector_successor(
                    root, {
                        "source_repository_commit": "c" * 40,
                        "clean_source_binding_digest": "b" * 64,
                        "bound_implementations_digest": "a" * 64,
                        "launch_state_selector_feasibility_receipt_digest":
                            "f" * 64,
                        "mixed_precontract_disposition_receipt_digest":
                            "d" * 64,
                    }, allocation, [], global_exact_manifest=manifest)
            certify.assert_called_once_with(manifest, allocation)
            legacy.assert_not_called()
            self.assertEqual(
                bindings["preserved_state_revalidation_receipt_digest"],
                "e" * 64)

    def test_registered_initialisation_is_reproducible_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(scorer, "PACKAGE_DIR", root):
                first, first_receipt = scorer.register_initialisation(
                    "latent", use_latent=True, seed=20260811,
                    binding_digest="a" * 64)
                second, second_receipt = scorer.register_initialisation(
                    "latent", use_latent=True, seed=20260811,
                    binding_digest="a" * 64)
            self.assertEqual(first_receipt["initial_state_digest"],
                             second_receipt["initial_state_digest"])
            self.assertEqual(scorer.state_dict_digest(first.state_dict()),
                             scorer.state_dict_digest(second.state_dict()))
            self.assertEqual(second_receipt["recovery_decision"],
                             "reused_verified_registered_initialisation")

    def test_checkpoint_requires_exact_rng_optimizer_and_order_state(self):
        scorer.configure_determinism(20260811)
        model = scorer.UtilityScorer(use_latent=False, hidden=8)
        state = scorer._cpu_state(model)
        execution = scorer._execution_fingerprint(torch.device("cpu"))
        generator = torch.Generator(device="cpu").manual_seed(20260811)
        order = torch.tensor([2, 0, 1], dtype=torch.int64)
        optimizer_state = torch.optim.AdamW(model.parameters()).state_dict()
        rng_state = scorer._capture_rng_state()
        generator_state = generator.get_state()
        payload = {
            "schema": "go2_utility_scorer_epoch_checkpoint_v1",
            "model_name": "no_latent", "use_latent": False,
            "training_run_digest": "b" * 64,
            "initial_state_digest": "c" * 64,
            "execution_fingerprint": execution,
            "fixed_final_epoch": 60,
            "epoch_selection": "final_epoch_only_no_selection",
            "learning_rate_schedule": "constant",
            "training_budget_digest": scorer.canonical_digest(
                dict(scorer.SCORER["training"])),
            "completed_epoch": 1,
            "model_state_dict": state,
            "model_state_digest": scorer.state_dict_digest(state),
            "optimizer_state_dict": optimizer_state,
            "optimizer_state_digest": scorer.structured_digest(optimizer_state),
            "rng_state": rng_state,
            "rng_state_digest": scorer.structured_digest(rng_state),
            "order_generator_state": generator_state,
            "order_generator_state_sha256": scorer.tensor_digest(generator_state),
            "last_epoch_order": order,
            "last_epoch_order_sha256": scorer.tensor_digest(order),
        }
        self.assertEqual(scorer._validate_checkpoint(
            payload, name="no_latent", use_latent=False,
            training_run_digest="b" * 64, initial_state_digest="c" * 64,
            execution=execution, training_rows=3, epochs=60,
            path=Path("epoch_001.pt")), 1)
        broken = dict(payload)
        broken.pop("order_generator_state")
        with self.assertRaisesRegex(ValueError, "shuffle/order state absent"):
            scorer._validate_checkpoint(
                broken, name="no_latent", use_latent=False,
                training_run_digest="b" * 64, initial_state_digest="c" * 64,
                execution=execution, training_rows=3, epochs=60,
                path=Path("epoch_001.pt"))

    def test_composite_metrics_report_recovery_regret_ties_and_spread(self):
        rows = [
            {"state_id": state, "utility": utility}
            for state, values in (("a", (3.0, 2.0, 1.0)),
                                  ("b", (1.0, 2.0, 3.0)))
            for utility in values
        ]
        truth = np.asarray([row["utility"] for row in rows])
        result = scorer.composite_metrics(rows, truth, truth.copy())
        self.assertEqual(result["pairwise_ordering_accuracy"], 1.0)
        self.assertEqual(result["normalised_rank_regret"], 0.0)
        self.assertEqual(result["top1_recovery"], 1.0)
        self.assertEqual(result["top3_recovery"], 1.0)
        self.assertEqual(result["tie_rate"], 0.0)
        self.assertEqual(result["candidate_score_spread"]["mean"], 2.0)

    def test_every_frozen_qualification_threshold_is_conjunctive(self):
        latent = {
            "progress": {"spearman": 0.50},
            "safety": {"auc_any_hazard": 0.75, "calibration_error": 0.10},
            "completion": {"auc": 0.75, "calibration_error": 0.10},
            "composite": {"pairwise_ordering_accuracy": 0.70},
        }
        baseline = {"composite": {"pairwise_ordering_accuracy": 0.65}}
        distribution = {"completion_prevalence": 0.5}
        criteria, _details, dominance = scorer.qualification_criteria(
            latent, baseline, distribution, distribution)
        self.assertAlmostEqual(dominance, 0.05)
        self.assertTrue(all(criteria.values()))
        latent["completion"]["calibration_error"] = 0.100001
        criteria, _details, _dominance = scorer.qualification_criteria(
            latent, baseline, distribution, distribution)
        self.assertFalse(criteria["completion_calibration_le_0.10"])
        self.assertFalse(all(criteria.values()))

    def test_label_distributions_include_quartiles_family_and_stratum(self):
        rows = []
        for index, value in enumerate((0.0, 1.0, 2.0, 3.0)):
            rows.append({
                "state_id": f"state-{index // 2}",
                "family": "family_a" if index < 2 else "family_b",
                "stratum": "general" if index < 2 else "safety_enriched",
                "progress": value, "safety": value / 3,
                "completion": float(index % 2), "utility": value - index / 4,
            })
        distribution = scorer.grouped_label_distributions(rows)
        self.assertEqual(distribution["overall"]["progress"]["quartile_1"], 0.75)
        self.assertEqual(distribution["overall"]["progress"]["quartile_3"], 2.25)
        self.assertEqual(set(distribution["by_family"]), {"family_a", "family_b"})
        self.assertEqual(distribution["by_stratum"]["completion_enriched"]["rows"], 0)

    def test_component_and_composite_diagnostics_include_required_ranking_fields(self):
        rows = [
            {"state_id": "a", "utility": value}
            for value in (0.0, 1.0, 2.0)
        ]
        truth = np.asarray([0.0, 1.0, 2.0])
        ranking = scorer._component_ranking(rows, truth, truth)
        self.assertEqual(ranking["within_state_pairwise_ordering_accuracy"], 1.0)
        self.assertEqual(ranking["highest_target_top1_recovery"], 1.0)
        composite = scorer.composite_metrics(rows, truth, truth)
        self.assertEqual(composite["realised_selected_utility"], 2.0)
        self.assertEqual(composite["oracle_best_utility"], 2.0)
        self.assertEqual(len(composite["per_state"]), 1)


if __name__ == "__main__":
    unittest.main()
