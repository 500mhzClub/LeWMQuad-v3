from __future__ import annotations

import copy
import hashlib
import unittest

from lewm.safety import occluded_goal_topological_belief_metrics_v1 as M1
from lewm.safety import occluded_goal_topological_belief_metrics_v2 as M
from lewm.safety import occluded_goal_topological_belief_v2_contract as C
from lewm.tests.test_occluded_goal_topological_belief_metrics_v1 import (
    _beliefs as _v1_beliefs,
    _calibration as _v1_calibration,
    _fixture as _v1_fixture,
    _stage_b_trace as _v1_stage_b_trace,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file(path: str, marker: str) -> dict[str, object]:
    return {"path": path, "bytes": 100 + len(marker), "sha256": _sha(marker)}


def _cache_digest(records: list[dict[str, object]]) -> str:
    projection = [
        {
            "pixel_index": row["pixel_index"],
            "pixel_sha256": row["pixel_sha256"],
            "canonical_template_id": row["canonical_template_id"],
            "preprocessed_tensor_sha256": row["preprocessed_tensor_sha256"],
            "raw_token_sha256": row["raw_token_sha256"],
            "spatial_descriptor_sha256": row["spatial_descriptor_sha256"],
        }
        for row in records
    ]
    return hashlib.sha256(C.canonical_json_bytes(projection)[:-1]).hexdigest()


def _evidence() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    group_sizes = [4] * 66 + [3] * 10 + [1] * 81
    assert len(group_sizes) == C.UNIQUE_PIXEL_COUNT and sum(group_sizes) == C.TEMPLATE_ROW_COUNT
    pixel_records: list[dict] = []
    template_records: list[dict] = []
    template_row = 0
    for pixel_index, size in enumerate(group_sizes):
        pixel_sha = f"{pixel_index + 1:064x}"
        members = [f"template-{template_row + offset:04d}" for offset in range(size)]
        rows = list(range(template_row, template_row + size))
        canonical = min(members)
        pixel_records.append({
            "pixel_index": pixel_index,
            "pixel_sha256": pixel_sha,
            "canonical_template_id": canonical,
            "member_template_ids": members,
            "observation_row_indices": rows,
        })
        for member, row in zip(members, rows):
            template_records.append({
                "template_row_index": row,
                "pixel_template_id": member,
                "pixel_sha256": pixel_sha,
                "pixel_index": pixel_index,
                "canonical_template_id": canonical,
            })
        template_row += size
    observations_binding = _file("observations.npz", "observations")
    pixel_index_doc = C.attach_content_digest({
        "schema": "occluded_goal_topological_belief_v2.pixel_index.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "observations_binding": observations_binding,
        "identity_rule": C.CANONICAL_HASH_DOMAINS["rgb_pixel_sha256"],
        "records": pixel_records,
    })
    template_doc = C.attach_content_digest({
        "schema": "occluded_goal_topological_belief_v2.template_to_pixel_index.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "pixel_index_binding": pixel_index_doc["content_digest"],
        "records": template_records,
    })
    occurrence_records: list[dict] = []
    for index in range(C.OCCURRENCE_COUNT):
        template = template_records[index % len(template_records)]
        occurrence_records.append({
            "occurrence_index": index,
            "capture_id": f"capture-{index:05d}",
            "episode_id": f"episode-{index // C.QUERIES_PER_EPISODE:05d}",
            "node_id": f"node-{index % 257:03d}",
            "phase": "PHASE_A_KEYFRAME" if index % 2 == 0 else "PHASE_C_QUERY_TRACE",
            "timestamp_s": float(index),
            "pixel_template_id": template["pixel_template_id"],
            "pixel_sha256": template["pixel_sha256"],
            "pixel_index": template["pixel_index"],
            "query_ids": [] if index % 8 else [f"query-{index:05d}"],
        })
    occurrence_doc = C.attach_content_digest({
        "schema": "occluded_goal_topological_belief_v2.occurrence_index.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "keyframe_index_binding": _file("keyframe_index.json", "keyframes"),
        "template_index_binding": template_doc["content_digest"],
        "records": occurrence_records,
    })
    pass_records = [
        {
            "pixel_index": row["pixel_index"],
            "pixel_sha256": row["pixel_sha256"],
            "canonical_template_id": row["canonical_template_id"],
            "preprocessed_tensor_sha256": _sha(f"pre-{row['pixel_index']}"),
            "raw_token_sha256": _sha(f"raw-{row['pixel_index']}"),
            "spatial_descriptor_sha256": _sha(f"descriptor-{row['pixel_index']}"),
        }
        for row in pixel_records
    ]
    cache_digest = _cache_digest(pass_records)
    pass_rows = [
        {
            "pass_index": index,
            "fresh_encoder_instance_id": f"fresh-encoder-pass-{index}",
            "ordered_pixel_sha256s": [row["pixel_sha256"] for row in pixel_records],
            "records": copy.deepcopy(pass_records),
            "canonical_cache_content_digest": cache_digest,
        }
        for index in (1, 2)
    ]
    encoder_binding = {
        key: C.VJEPA_ENCODER_BINDING[key]
        for key in (
            "constructor", "checkpoint_sha256", "checkpoint_size_bytes",
            "helper_path", "helper_sha256", "external_repository_commit",
        )
    }
    encoder_binding["preprocessing_digest"] = C.PREPROCESSING_DIGEST
    encoding_receipt = {
        "schema": "occluded_goal_topological_belief_v2.encoding_determinism_receipt.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "source_freeze_commit": "a" * 40,
        "observations_binding": observations_binding,
        "pixel_index_content_digest": pixel_index_doc["content_digest"],
        "encoder_binding": encoder_binding,
        "hash_domains": copy.deepcopy(C.CANONICAL_HASH_DOMAINS),
        "counts": {
            "template_rows": C.TEMPLATE_ROW_COUNT,
            "unique_pixel_rows": C.UNIQUE_PIXEL_COUNT,
            "reused_template_rows": C.REUSED_TEMPLATE_ROW_COUNT,
            "pass_count": 2,
            "encoder_invocations_per_pass": C.UNIQUE_PIXEL_COUNT,
            "singleton_batch_size": 1,
        },
        "pre_outcome_boundary": copy.deepcopy(C.PRE_OUTCOME_BOUNDARY_AUTHORITY),
        "passes": pass_rows,
        "comparisons": {
            "pixel_order_exact": True,
            "preprocessed_tensors_exact": True,
            "raw_tokens_exact": True,
            "spatial_descriptors_exact": True,
            "canonical_cache_content_digest_exact": True,
            "pass": True,
        },
    }
    encoding_file = {
        "path": "encoding_determinism_receipt.json",
        "bytes": len(C.canonical_json_bytes(encoding_receipt)),
        "sha256": hashlib.sha256(C.canonical_json_bytes(encoding_receipt)).hexdigest(),
    }
    latent_doc = C.attach_content_digest({
        "schema": "occluded_goal_topological_belief_v2.canonical_latent_index.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "source_freeze_commit": "a" * 40,
        "pixel_index_binding": pixel_index_doc["content_digest"],
        "encoding_determinism_receipt_binding": encoding_file,
        "tokens_file": _file("canonical_tokens.npz", "tokens"),
        "records": [
            {
                "pixel_index": row["pixel_index"],
                "pixel_sha256": row["pixel_sha256"],
                "canonical_template_id": row["canonical_template_id"],
                "raw_token_row_index": row["pixel_index"],
                "raw_token_sha256": pass_records[row["pixel_index"]]["raw_token_sha256"],
            }
            for row in pixel_records
        ],
    })
    descriptor_doc = C.attach_content_digest({
        "schema": "occluded_goal_topological_belief_v2.canonical_descriptor_index.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "source_freeze_commit": "a" * 40,
        "pixel_index_binding": pixel_index_doc["content_digest"],
        "encoding_determinism_receipt_binding": encoding_file,
        "descriptors_file": _file("canonical_descriptors.npz", "descriptors"),
        "records": [
            {
                "pixel_index": row["pixel_index"],
                "pixel_sha256": row["pixel_sha256"],
                "canonical_template_id": row["canonical_template_id"],
                "spatial_descriptor_row_index": row["pixel_index"],
                "spatial_descriptor_sha256": pass_records[row["pixel_index"]]["spatial_descriptor_sha256"],
            }
            for row in pixel_records
        ],
    })
    reusable = [
        copy.deepcopy(row) for row in C.V1_RETAINED_LEAVES
        if row["path"] in C.V1_REUSABLE_LEAVES
    ]
    cache_receipt = {
        "schema": "occluded_goal_topological_belief_v2.cache_integrity_receipt.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "source_freeze_commit": "a" * 40,
        "v1_retained_root_binding": C.v1_retained_root_authority(),
        "copied_v1_input_bindings": reusable,
        "pixel_index_content_digest": pixel_index_doc["content_digest"],
        "template_index_content_digest": template_doc["content_digest"],
        "occurrence_index_content_digest": occurrence_doc["content_digest"],
        "encoding_determinism_receipt_binding": encoding_file,
        "canonical_latent_index_content_digest": latent_doc["content_digest"],
        "canonical_descriptor_index_content_digest": descriptor_doc["content_digest"],
        "counts": {
            "template_rows": C.TEMPLATE_ROW_COUNT,
            "unique_pixel_rows": C.UNIQUE_PIXEL_COUNT,
            "reused_template_rows": C.REUSED_TEMPLATE_ROW_COUNT,
            "occurrence_rows": C.OCCURRENCE_COUNT,
            "canonical_token_rows": C.UNIQUE_PIXEL_COUNT,
            "canonical_descriptor_rows": C.UNIQUE_PIXEL_COUNT,
            "encoder_invocations_per_pass": C.UNIQUE_PIXEL_COUNT,
            "multi_template_pixel_groups": C.MULTI_TEMPLATE_PIXEL_GROUP_COUNT,
            "singleton_pixel_groups": C.UNIQUE_PIXEL_COUNT - C.MULTI_TEMPLATE_PIXEL_GROUP_COUNT,
            "templates_in_multi_template_groups": 294,
        },
        "gates": {field: True for field in M.CACHE_GATE_FIELDS},
    }
    return (
        pixel_index_doc, template_doc, occurrence_doc, latent_doc,
        descriptor_doc, encoding_receipt, cache_receipt,
    )


class OccludedGoalTopologicalBeliefMetricsV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = _evidence()

    def test_public_scientific_authority_is_frozen_v1(self) -> None:
        self.assertEqual(M.stage_a_condition_ids(), M1.stage_a_condition_ids())
        for query_id, count in (("query-a", 4), ("query-b", 9)):
            self.assertEqual(
                M.deterministic_action_history_position_mapping(query_id, count),
                M1.deterministic_action_history_position_mapping(query_id, count),
            )
        authority = M.reducer_authority()
        C.validate_content_digest(authority)
        self.assertEqual(authority["experiment_id"], C.EXPERIMENT_ID)
        self.assertEqual(authority["cache_gate"]["counts"]["unique_pixel_rows"], 157)
        self.assertEqual(
            (
                authority["cache_gate"]["counts"]["multi_template_pixel_groups"],
                authority["cache_gate"]["counts"]["singleton_pixel_groups"],
                authority["cache_gate"]["counts"][
                    "templates_in_multi_template_groups"
                ],
            ),
            (76, 81, 294),
        )

    def test_complete_cache_gate_passes_and_receipts_have_no_self_digest(self) -> None:
        result = M.validate_cache_gate(*self.evidence)
        self.assertTrue(result["pass"])
        self.assertEqual(result["counts"]["template_rows"], 375)
        self.assertEqual(result["counts"]["occurrence_rows"], 33_384)
        self.assertNotIn("content_digest", self.evidence[5])
        self.assertNotIn("content_digest", self.evidence[6])

    def test_explicit_cache_gates_cover_outcome_slot_copy_and_regeneration(self) -> None:
        required = {
            "no_calibration_or_heldout_outcome_opened",
            "no_occurrence_mapped_by_gpu_slot",
            "no_template_specific_token_copy_differs",
            "canonical_regeneration_byte_identical",
        }
        self.assertTrue(required.issubset(M.CACHE_GATE_FIELDS))
        self.assertEqual(set(C.CANONICAL_CACHE_GATE_IDS), M.CACHE_GATE_FIELDS)
        authority = M.cache_gate_authority()
        self.assertTrue(required.issubset(authority["fields"]["cache_gates"]))

    def test_cache_gate_rejects_pre_outcome_boundary_tamper(self) -> None:
        evidence = list(copy.deepcopy(self.evidence))
        evidence[5]["pre_outcome_boundary"][
            "heldout_outcome_documents_opened"
        ] = 1
        with self.assertRaisesRegex(M.OccludedGoalV2MetricsError, "pre-outcome"):
            M.validate_cache_gate(*evidence)

    def test_canonical_cache_digest_excludes_trailing_lf(self) -> None:
        records = self.evidence[5]["passes"][0]["records"]
        projection = [
            {key: row[key] for key in (
                "pixel_index", "pixel_sha256", "canonical_template_id",
                "preprocessed_tensor_sha256", "raw_token_sha256",
                "spatial_descriptor_sha256",
            )}
            for row in records
        ]
        without_lf = hashlib.sha256(C.canonical_json_bytes(projection)[:-1]).hexdigest()
        with_lf = hashlib.sha256(C.canonical_json_bytes(projection)).hexdigest()
        self.assertEqual(self.evidence[5]["passes"][0]["canonical_cache_content_digest"], without_lf)
        self.assertNotEqual(without_lf, with_lf)

    def test_cache_gate_rejects_noncanonical_template(self) -> None:
        evidence = list(copy.deepcopy(self.evidence))
        evidence[0]["records"][0]["canonical_template_id"] = "template-z"
        evidence[0].pop("content_digest")
        evidence[0] = C.attach_content_digest(evidence[0])
        with self.assertRaisesRegex(M.OccludedGoalV2MetricsError, "canonical template"):
            M.validate_cache_gate(*evidence)

    def test_cache_gate_rejects_two_pass_drift_even_if_claimed_pass(self) -> None:
        evidence = list(copy.deepcopy(self.evidence))
        evidence[5]["passes"][1]["records"][0]["raw_token_sha256"] = _sha("different")
        evidence[5]["passes"][1]["canonical_cache_content_digest"] = _cache_digest(
            evidence[5]["passes"][1]["records"]
        )
        with self.assertRaisesRegex(M.OccludedGoalV2MetricsError, "two-pass"):
            M.validate_cache_gate(*evidence)

    def test_cache_gate_rejects_receipt_self_digest(self) -> None:
        evidence = list(copy.deepcopy(self.evidence))
        evidence[5]["content_digest"] = "a" * 64
        with self.assertRaisesRegex(M.OccludedGoalV2MetricsError, "field set"):
            M.validate_cache_gate(*evidence)

    def test_stage_a_authorization_requires_v2_identity(self) -> None:
        v2 = C.attach_content_digest({
            "schema": M.V2_STAGE_A_METRICS_SCHEMA,
            "experiment_id": C.EXPERIMENT_ID,
            "decision": {"stage_b_authorized": True},
        })
        self.assertTrue(M.stage_a_authorizes_stage_b(v2))
        v1 = copy.deepcopy(v2)
        v1.pop("content_digest")
        v1["schema"] = "occluded_goal_topological_belief_v1.stage_a_metrics.v1"
        v1["experiment_id"] = C.V1_EXPERIMENT_ID
        v1 = C.attach_content_digest(v1)
        with self.assertRaises(M.OccludedGoalV2MetricsError):
            M.stage_a_authorizes_stage_b(v1)

    def test_stage_a_wrapper_changes_identity_not_formula_or_decision(self) -> None:
        manifest, queries = _v1_fixture()
        calibration_v1 = _v1_calibration(manifest, queries)
        beliefs = _v1_beliefs(manifest, queries, "map")
        expected = M1.recompute_stage_a_metrics(
            manifest, queries, beliefs, calibration_v1
        )
        calibration_v2 = copy.deepcopy(calibration_v1)
        calibration_v2.pop("content_digest")
        calibration_v2["schema"] = M.V2_CALIBRATION_SCHEMA
        calibration_v2["experiment_id"] = C.EXPERIMENT_ID
        calibration_v2 = C.attach_content_digest(calibration_v2)
        observed = M.recompute_stage_a_metrics(
            manifest, queries, beliefs, calibration_v2
        )
        self.assertEqual(observed["schema"], M.V2_STAGE_A_METRICS_SCHEMA)
        self.assertEqual(observed["experiment_id"], C.EXPERIMENT_ID)
        self.assertEqual(observed["conditions"], expected["conditions"])
        self.assertEqual(observed["gate"], expected["gate"])
        self.assertEqual(observed["decision"], expected["decision"])
        self.assertEqual(
            observed["calibration_binding"]["selected_parameters"],
            expected["calibration_binding"]["selected_parameters"],
        )

    def test_stage_b_wrapper_changes_identity_not_formula_or_decision(self) -> None:
        manifest, queries = _v1_fixture()
        calibration_v1 = _v1_calibration(manifest, queries)
        beliefs = _v1_beliefs(manifest, queries, "map")
        stage_a_v1 = M1.recompute_stage_a_metrics(
            manifest, queries, beliefs, calibration_v1
        )
        trace = _v1_stage_b_trace(manifest, queries, stage_a_v1)
        expected = M1.recompute_stage_b_metrics(
            manifest, queries, trace, stage_a_v1
        )
        calibration_v2 = copy.deepcopy(calibration_v1)
        calibration_v2.pop("content_digest")
        calibration_v2["schema"] = M.V2_CALIBRATION_SCHEMA
        calibration_v2["experiment_id"] = C.EXPERIMENT_ID
        calibration_v2 = C.attach_content_digest(calibration_v2)
        stage_a_v2 = M.recompute_stage_a_metrics(
            manifest, queries, beliefs, calibration_v2
        )
        observed = M.recompute_stage_b_metrics(
            manifest, queries, trace, stage_a_v2
        )
        self.assertEqual(observed["schema"], M.V2_STAGE_B_METRICS_SCHEMA)
        self.assertEqual(observed["experiment_id"], C.EXPERIMENT_ID)
        self.assertEqual(observed["conditions"], expected["conditions"])
        self.assertEqual(observed["gate"], expected["gate"])
        self.assertEqual(observed["decision"], expected["decision"])


if __name__ == "__main__":
    unittest.main()
