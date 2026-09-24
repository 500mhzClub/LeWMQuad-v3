from __future__ import annotations

import copy
import unittest

from lewm.safety import occluded_goal_topological_belief_v1_contract as V1
from lewm.safety import occluded_goal_topological_belief_v2_contract as C


class OccludedGoalTopologicalBeliefV2ContractTest(unittest.TestCase):
    def test_v2_identity_subjects_and_output_tree_are_exact(self) -> None:
        self.assertEqual(C.EXPERIMENT_ID, "OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V2")
        self.assertEqual(
            C.CONTRACT_FREEZE_COMMIT_SUBJECT,
            "Freeze canonical occluded-goal topological belief replacement",
        )
        self.assertEqual(
            C.RESULT_COMMIT_SUBJECT,
            "Evaluate canonical occluded-goal topological belief replacement",
        )
        self.assertEqual(
            str(C.OUTPUT_ROOT),
            "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/occluded_goal_topological_belief_v2",
        )
        self.assertEqual(len(C.UNCONDITIONAL_OUTPUT_LEAVES), 22)
        self.assertEqual(len(C.RUNTIME_OUTPUT_PATHS), 24)
        self.assertNotIn("latents.npz", C.RUNTIME_OUTPUT_PATHS.values())
        self.assertNotIn("latent_index.json", C.RUNTIME_OUTPUT_PATHS.values())
        self.assertEqual(
            {
                "pixel_index.json", "template_to_pixel_index.json",
                "occurrence_index.json", "canonical_tokens.npz",
                "canonical_descriptors.npz", "canonical_latent_index.json",
                "canonical_descriptor_index.json",
                "encoding_determinism_receipt.json", "cache_integrity_receipt.json",
            },
            {
                value for key, value in C.RUNTIME_OUTPUT_PATHS.items()
                if key in {
                    "pixel_index", "template_to_pixel_index", "occurrence_index",
                    "canonical_tokens", "canonical_descriptors",
                    "canonical_latent_index", "canonical_descriptor_index",
                    "encoding_determinism_receipt", "cache_integrity_receipt",
                }
            },
        )

    def test_v1_scientific_design_is_exactly_preserved(self) -> None:
        names = (
            "FAMILY_IDS", "SPLIT_EPISODE_COUNTS", "SPLIT_FAMILY_EPISODE_COUNTS",
            "CONDITION_IDS", "PRIMARY_CONDITION_IDS", "ABLATION_CONDITION_IDS",
            "CALIBRATION_GRID", "CALIBRATION_GRID_ORDER", "CALIBRATION_SELECTION",
            "FILTER_AUTHORITY", "ALIASING_AUTHORITY", "METRIC_IDS",
            "METRIC_FORMULAS", "ABSOLUTE_GATE", "INCREMENTAL_OVER_CURRENT_GATE",
            "INCREMENTAL_OVER_MAP_GATE", "SHORT_HISTORY_MATCH_GATE",
            "STAGE_A_CLASSIFICATIONS", "STAGE_A_PRECEDENCE",
            "STAGE_B_CONDITION_IDS", "STAGE_B_EXECUTION_POLICY", "STAGE_B_GATE",
            "STAGE_B_METRIC_FORMULAS", "STAGE_B_CLASSIFICATIONS",
            "NEXT_DECISION_BY_CLASSIFICATION", "STAGE_B_AUTHORITY",
        )
        for name in names:
            self.assertEqual(getattr(C, name), getattr(V1, name), name)
        self.assertEqual(C.SCIENTIFIC_PROHIBITIONS, V1.PROHIBITIONS)
        for key, value in V1.OBSERVATION_LIKELIHOOD.items():
            if key != "persisted_arrays":
                self.assertEqual(C.OBSERVATION_LIKELIHOOD[key], value)
        self.assertEqual(
            C.SCIENTIFIC_OBSERVATION_LIKELIHOOD, V1.OBSERVATION_LIKELIHOOD
        )
        for key, value in V1.PROHIBITIONS.items():
            self.assertEqual(C.PROHIBITIONS[key], value)
        self.assertTrue(all(value == 0 for value in C.PROHIBITIONS.values()))

    def test_v1_failed_root_binding_is_exact_and_partitioned(self) -> None:
        authority = C.v1_retained_root_authority()
        self.assertEqual(authority["disposition"], "BATCH_SLOT_DEPENDENT_DUPLICATE_IMAGE_ENCODING")
        self.assertEqual(authority["leaf_count"], 12)
        self.assertEqual(authority["total_bytes"], 1_744_065_052)
        self.assertEqual(len(authority["leaves"]), 12)
        self.assertEqual(
            sum(row["bytes"] for row in authority["leaves"]),
            authority["total_bytes"],
        )
        self.assertEqual(authority["leaves"], authority["inventory"])
        self.assertEqual(len(authority["reusable_leaves"]), 6)
        self.assertEqual(len(authority["invalid_nonreusable_leaves"]), 5)
        self.assertEqual(authority["custody_only_leaves"], ["contract.json"])
        self.assertEqual(len(authority["nonreusable_leaves"]), 6)
        self.assertEqual(
            set(authority["reusable_leaves"])
            | set(authority["invalid_nonreusable_leaves"])
            | set(authority["custody_only_leaves"]),
            {row["path"] for row in authority["leaves"]},
        )
        defect = authority["defect_evidence"]
        self.assertEqual(
            (defect["duplicate_exact_pixel_groups"], defect["conflicting_duplicate_exact_pixel_groups"], defect["affected_template_rows"]),
            (76, 75, 292),
        )
        self.assertEqual(defect["slot_identity"], "row_index % 8")
        science = authority["scientific_disposition"]
        self.assertTrue(science["stage_b_ranker_initialized"])
        self.assertFalse(science["stage_b_ranker_score_evaluated"])
        self.assertEqual(science["stage_b_ranker_call_count"], 0)
        self.assertIsNone(science["stage_b_result"])
        C.validate_v1_retained_root_binding(authority)
        authority["leaves"][0]["bytes"] += 1
        with self.assertRaises(C.OccludedGoalV2ContractError):
            C.validate_v1_retained_root_binding(authority)

    def test_outcome_observed_and_confirmation_boundaries_are_explicit(self) -> None:
        boundary = C.OUTCOME_OBSERVED_CLAIMS_BOUNDARY
        self.assertTrue(boundary["v1_stage_a_outcome_observed"])
        self.assertFalse(boundary["v1_stage_a_outcome_scientifically_usable"])
        self.assertFalse(boundary["v2_is_fresh_untouched_or_confirmatory"])
        self.assertIn("fresh scene-disjoint panel", boundary["positive_v2_confirmation_requirement"])
        self.assertIn("unchanged descriptor and belief method", boundary["positive_v2_confirmation_requirement"])

    def test_canonical_encoding_authority_is_singleton_two_pass(self) -> None:
        self.assertEqual(C.TEMPLATE_ROW_COUNT, 375)
        self.assertEqual(C.UNIQUE_PIXEL_COUNT, 157)
        self.assertEqual(C.REUSED_TEMPLATE_ROW_COUNT, 218)
        self.assertEqual(C.OCCURRENCE_COUNT, 33_384)
        authority = C.CANONICAL_ENCODING_AUTHORITY
        self.assertEqual(authority["passes"], 2)
        self.assertTrue(authority["fresh_encoder_load_per_pass"])
        self.assertEqual(authority["batch_size"], 1)
        self.assertEqual(authority["encoder_invocations_per_pass"], 157)
        self.assertEqual(authority["tolerance"], 0)
        self.assertEqual(
            authority["terminal_failure"]["classification"],
            "CANONICAL_SINGLETON_ENCODER_NONDETERMINISM",
        )
        self.assertEqual(authority["terminal_failure"]["boundary"], "stop before calibration")
        self.assertIsNone(authority["terminal_failure"]["scientific_result"])
        self.assertEqual(
            authority["pre_outcome_boundary"], C.PRE_OUTCOME_BOUNDARY_AUTHORITY
        )
        self.assertEqual(
            tuple(authority["required_cache_gates"]), C.CANONICAL_CACHE_GATE_IDS
        )
        self.assertTrue(
            {
                "no_calibration_or_heldout_outcome_opened",
                "no_occurrence_mapped_by_gpu_slot",
                "no_template_specific_token_copy_differs",
                "canonical_regeneration_byte_identical",
            }.issubset(C.CANONICAL_CACHE_GATE_IDS)
        )

    def test_contract_digest_round_trip_and_receipt_policy(self) -> None:
        value = C.build_contract()
        self.assertEqual(C.validate_contract(value), value)
        self.assertFalse(value["output"]["receipt_self_digests"])
        self.assertEqual(len(C.TRACKED_SOURCE_PATHS), 13)
        self.assertEqual(len(set(C.TRACKED_SOURCE_PATHS)), 13)
        tampered = copy.deepcopy(value)
        tampered["claims"]["outcome_observed"] = False
        with self.assertRaises(C.OccludedGoalV2ContractError):
            C.validate_contract(tampered)


if __name__ == "__main__":
    unittest.main()
