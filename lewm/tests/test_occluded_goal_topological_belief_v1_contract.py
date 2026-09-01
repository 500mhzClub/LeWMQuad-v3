from __future__ import annotations

import copy
import unittest

from lewm.safety import occluded_goal_topological_belief_v1_contract as C


class OccludedGoalTopologicalBeliefContractTest(unittest.TestCase):
    def test_frozen_scientific_authority(self) -> None:
        contract = C.validate_contract(C.build_contract())
        self.assertEqual(contract["panel"]["episode_count"], 96)
        self.assertEqual(contract["panel"]["split_episode_counts"], {
            "FIT": 64, "CALIBRATION": 16, "DEVELOPMENT_HELDOUT": 16,
        })
        self.assertEqual(C.QUERY_INDEX_VALUES, tuple(range(8)))
        self.assertEqual(C.DEPTH_SINCE_LAST_UNAMBIGUOUS_VALUES, tuple(range(4, 12)))
        self.assertEqual(len(C.CONDITION_IDS), 9)
        self.assertEqual(C.ABLATION_CONDITION_IDS, (
            "NO_ACTION_CONSISTENCY",
            "SHUFFLED_ACTION_HISTORY",
            "NO_OBSERVATION_LIKELIHOOD",
        ))
        self.assertEqual(C.CALIBRATION_SELECTION["grid_results_persisted"], 144)
        self.assertEqual(C.CALIBRATION_SELECTION["grid_query_results_persisted"], 18432)
        self.assertEqual(C.IDENTITY_DOMAIN, "ogtb-v1")
        self.assertEqual(C.PROCEDURAL_SEED_VALUES, tuple(range(5712627296233390080, 5712627296233390176)))
        self.assertEqual(len(C.PRIOR_PANEL_EXCLUSION_AUTHORITY["authorities"]), 10)
        self.assertEqual(
            C.PRIOR_PANEL_EXCLUSION_AUTHORITY["prior_projection"]["scene_identity_count"],
            1295,
        )
        self.assertEqual(
            C.PRIOR_PANEL_EXCLUSION_AUTHORITY["prior_projection"]["scene_identity_sha256_count"],
            1487,
        )
        self.assertIn("TOPOLOGICAL_PERSISTENT_GATE_UNRESOLVED", C.STAGE_A_CLASSIFICATIONS)
        self.assertEqual(C.STAGE_B_EXECUTION_POLICY["decision_budget"], 16)
        self.assertEqual(C.STAGE_B_EXECUTION_POLICY["registered_choice_points"], 8)
        self.assertIn("observed goal node", C.STAGE_B_EXECUTION_POLICY["initial_state"])
        self.assertEqual(len(C.STAGE_B_LOCAL_CANDIDATE_IDS), 12)
        self.assertEqual(len(C.TRACKED_SOURCE_PATHS), 13)

    def test_output_tree_and_external_bindings_are_exact(self) -> None:
        self.assertEqual(set(C.RUNTIME_OUTPUT_PATHS.values()), {
            "contract.json", "panel_manifest.json", "split_manifest.json",
            "graph_manifest.json", "keyframe_index.json", "query_ledger.jsonl",
            "observations.npz", "latent_index.json", "latents.npz",
            "calibration.json", "stage_a_beliefs.jsonl", "stage_a_metrics.json",
            "stage_b_trace.jsonl", "stage_b_metrics.json", "result.json",
            "result.md", "file_hashes.json",
        })
        self.assertEqual(C.VJEPA_ENCODER_BINDING["checkpoint_size_bytes"], 5151198524)
        self.assertEqual(
            C.VJEPA_ENCODER_BINDING["checkpoint_sha256"],
            "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6",
        )
        self.assertEqual(C.VJEPA_ENCODER_BINDING["constructor"], "vjepa2_1_vit_large_384")
        self.assertEqual(C.STAGE_B_CURRENT_VISUAL_BINDING["checkpoint_size_bytes"], 2470669)
        self.assertEqual(C.STAGE_B_CURRENT_VISUAL_BINDING["parameter_count"], 204289)
        self.assertFalse(C.PREDECESSOR_RESULT_BINDING["true_future_jepa_incremental_route_value_supported"])
        self.assertFalse(C.PREDECESSOR_RESULT_BINDING["stage_b_ran"])

    def test_digest_and_contract_tamper_fail(self) -> None:
        contract = C.build_contract()
        tampered = copy.deepcopy(contract)
        tampered["panel"]["episode_count"] = 95
        with self.assertRaises(C.OccludedGoalContractError):
            C.validate_contract(tampered)
        repaired_digest = C.attach_content_digest(tampered)
        with self.assertRaises(C.OccludedGoalContractError):
            C.validate_contract(repaired_digest)

    def test_pure_module_contains_no_execution_framework(self) -> None:
        forbidden = {"torch", "sys.addaudithook", "subprocess", "Genesis"}
        with open(C.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for token in forbidden:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
