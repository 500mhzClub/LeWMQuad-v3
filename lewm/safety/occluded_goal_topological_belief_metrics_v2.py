"""Pure V2 cache gates and exact wrappers around frozen V1 reducers.

The canonical-cache replacement changes evidence identity and document
schemas, not any scientific metric formula, threshold, gate, classification,
or next-decision rule.  This module performs no file I/O or model execution.
"""
from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from lewm.safety import occluded_goal_topological_belief_metrics_v1 as M1
from lewm.safety import occluded_goal_topological_belief_v2_contract as C


class OccludedGoalV2MetricsError(ValueError):
    """Raised when V2 evidence or canonical-cache custody drifts."""


# Scientific row schemas remain exactly V1.  Graph/query inputs are the six
# byte-identical, explicitly reusable V1 leaves; belief and trace rows have no
# root schema field.
GRAPH_ROOT_FIELDS = set(M1.GRAPH_ROOT_FIELDS)
GRAPH_FIELDS = set(M1.GRAPH_FIELDS)
IDENTITY_DISJOINTNESS_FIELDS = set(M1.IDENTITY_DISJOINTNESS_FIELDS)
CURRENT_IDENTITY_PROJECTION_FIELDS = set(M1.CURRENT_IDENTITY_PROJECTION_FIELDS)
IDENTITY_COMPARISON_FIELDS = set(M1.IDENTITY_COMPARISON_FIELDS)
NODE_FIELDS = set(M1.NODE_FIELDS)
EDGE_FIELDS = set(M1.EDGE_FIELDS)
QUERY_FIELDS = set(M1.QUERY_FIELDS)
BELIEF_FIELDS = set(M1.BELIEF_FIELDS)
CALIBRATION_FIELDS = set(M1.CALIBRATION_FIELDS)
STAGE_B_TRACE_FIELDS = set(M1.STAGE_B_TRACE_FIELDS)

PIXEL_INDEX_FIELDS = {
    "schema", "experiment_id", "observations_binding", "identity_rule",
    "records", "content_digest",
}
PIXEL_RECORD_FIELDS = {
    "pixel_index", "pixel_sha256", "canonical_template_id",
    "member_template_ids", "observation_row_indices",
}
TEMPLATE_INDEX_FIELDS = {
    "schema", "experiment_id", "pixel_index_binding", "records",
    "content_digest",
}
TEMPLATE_RECORD_FIELDS = {
    "template_row_index", "pixel_template_id", "pixel_sha256", "pixel_index",
    "canonical_template_id",
}
OCCURRENCE_INDEX_FIELDS = {
    "schema", "experiment_id", "keyframe_index_binding",
    "template_index_binding", "records", "content_digest",
}
OCCURRENCE_RECORD_FIELDS = {
    "occurrence_index", "capture_id", "episode_id", "node_id", "phase",
    "timestamp_s", "pixel_template_id", "pixel_sha256", "pixel_index",
    "query_ids",
}
FILE_BINDING_FIELDS = {"path", "bytes", "sha256"}
ENCODER_BINDING_FIELDS = {
    "constructor", "checkpoint_sha256", "checkpoint_size_bytes",
    "helper_path", "helper_sha256", "external_repository_commit",
    "preprocessing_digest",
}
ENCODING_RECEIPT_FIELDS = {
    "schema", "experiment_id", "source_freeze_commit", "observations_binding",
    "pixel_index_content_digest", "encoder_binding", "hash_domains", "counts",
    "pre_outcome_boundary", "passes", "comparisons",
}
PRE_OUTCOME_BOUNDARY_FIELDS = {
    "boundary", "present_v2_leaf_names", "forbidden_outcome_leaf_names",
    "observed_forbidden_outcome_leaf_names",
    "calibration_outcome_documents_opened",
    "heldout_outcome_documents_opened",
    "external_regeneration_receipt_present",
}
ENCODING_COUNT_FIELDS = {
    "template_rows", "unique_pixel_rows", "reused_template_rows", "pass_count",
    "encoder_invocations_per_pass", "singleton_batch_size",
}
ENCODING_PASS_FIELDS = {
    "pass_index", "fresh_encoder_instance_id", "ordered_pixel_sha256s",
    "records", "canonical_cache_content_digest",
}
ENCODING_PASS_RECORD_FIELDS = {
    "pixel_index", "pixel_sha256", "canonical_template_id",
    "preprocessed_tensor_sha256", "raw_token_sha256",
    "spatial_descriptor_sha256",
}
ENCODING_COMPARISON_FIELDS = {
    "pixel_order_exact", "preprocessed_tensors_exact", "raw_tokens_exact",
    "spatial_descriptors_exact", "canonical_cache_content_digest_exact", "pass",
}
CANONICAL_LATENT_INDEX_FIELDS = {
    "schema", "experiment_id", "source_freeze_commit", "pixel_index_binding",
    "encoding_determinism_receipt_binding", "tokens_file", "records",
    "content_digest",
}
CANONICAL_LATENT_RECORD_FIELDS = {
    "pixel_index", "pixel_sha256", "canonical_template_id",
    "raw_token_row_index", "raw_token_sha256",
}
CANONICAL_DESCRIPTOR_INDEX_FIELDS = {
    "schema", "experiment_id", "source_freeze_commit", "pixel_index_binding",
    "encoding_determinism_receipt_binding", "descriptors_file", "records",
    "content_digest",
}
CANONICAL_DESCRIPTOR_RECORD_FIELDS = {
    "pixel_index", "pixel_sha256", "canonical_template_id",
    "spatial_descriptor_row_index", "spatial_descriptor_sha256",
}
CACHE_RECEIPT_FIELDS = {
    "schema", "experiment_id", "source_freeze_commit",
    "v1_retained_root_binding", "copied_v1_input_bindings",
    "pixel_index_content_digest", "template_index_content_digest",
    "occurrence_index_content_digest", "encoding_determinism_receipt_binding",
    "canonical_latent_index_content_digest",
    "canonical_descriptor_index_content_digest", "counts", "gates",
}
CACHE_COUNT_FIELDS = {
    "template_rows", "unique_pixel_rows", "reused_template_rows",
    "occurrence_rows", "canonical_token_rows", "canonical_descriptor_rows",
    "encoder_invocations_per_pass",
    "multi_template_pixel_groups", "singleton_pixel_groups",
    "templates_in_multi_template_groups",
}
CACHE_GATE_FIELDS = set(C.CANONICAL_CACHE_GATE_IDS)

V2_CALIBRATION_SCHEMA = "occluded_goal_topological_belief_v2.calibration.v1"
V2_STAGE_A_METRICS_SCHEMA = "occluded_goal_topological_belief_v2.stage_a_metrics.v1"
V2_STAGE_B_METRICS_SCHEMA = "occluded_goal_topological_belief_v2.stage_b_metrics.v1"


def _mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OccludedGoalV2MetricsError(f"{label} field set drift")
    return dict(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OccludedGoalV2MetricsError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OccludedGoalV2MetricsError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OccludedGoalV2MetricsError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OccludedGoalV2MetricsError(f"{label} must be finite numeric")
    return result


def _sha(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise OccludedGoalV2MetricsError(f"{label} must be lowercase SHA-256 hex")
    return text


def _commit(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise OccludedGoalV2MetricsError(f"{label} must be lowercase Git SHA-1 hex")
    return text


def _validate_content_document(
    value: Mapping[str, Any], *, fields: set[str], schema: str, label: str
) -> dict[str, Any]:
    row = _mapping(value, fields, label)
    try:
        C.validate_content_digest(row)
    except C.OccludedGoalV2ContractError as exc:
        raise OccludedGoalV2MetricsError(str(exc)) from exc
    if row["schema"] != schema or row["experiment_id"] != C.EXPERIMENT_ID:
        raise OccludedGoalV2MetricsError(f"{label} identity drift")
    return row


def _validate_file_binding(value: Any, label: str, expected_path: str | None = None) -> dict[str, Any]:
    row = _mapping(value, FILE_BINDING_FIELDS, label)
    if expected_path is not None and row["path"] != expected_path:
        raise OccludedGoalV2MetricsError(f"{label} path drift")
    _integer(row["bytes"], f"{label}.bytes", minimum=1)
    _sha(row["sha256"], f"{label}.sha256")
    return row


def _ordinary_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(C.canonical_json_bytes(dict(value))).hexdigest()


def _cache_content_digest(records: Sequence[Mapping[str, Any]]) -> str:
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


def stage_a_condition_ids() -> tuple[str, ...]:
    return tuple(C.CONDITION_IDS)


def deterministic_action_history_position_mapping(
    query_id: str, action_count: int
) -> tuple[int, ...]:
    return M1.deterministic_action_history_position_mapping(query_id, action_count)


def v1_retained_root_authority() -> dict[str, Any]:
    return C.v1_retained_root_authority()


def graph_manifest_authority() -> dict[str, Any]:
    authority = M1.graph_manifest_authority()
    authority["inherited_schema"] = "occluded_goal_topological_belief_v1.graph_manifest.v1"
    authority["v2_copy_policy"] = "byte-identical reusable V1 input"
    return authority


def query_row_authority() -> dict[str, Any]:
    return M1.query_row_authority()


def belief_row_authority() -> dict[str, Any]:
    return M1.belief_row_authority()


def calibration_authority() -> dict[str, Any]:
    authority = M1.calibration_authority()
    authority["schema"] = V2_CALIBRATION_SCHEMA
    authority["experiment_id"] = C.EXPERIMENT_ID
    return authority


def stage_b_trace_authority() -> dict[str, Any]:
    return M1.stage_b_trace_authority()


def cache_gate_authority() -> dict[str, Any]:
    return C.attach_content_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.cache_gate_authority.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "counts": {
                "template_rows": C.TEMPLATE_ROW_COUNT,
                "unique_pixel_rows": C.UNIQUE_PIXEL_COUNT,
                "reused_template_rows": C.REUSED_TEMPLATE_ROW_COUNT,
                "multi_template_pixel_groups": C.MULTI_TEMPLATE_PIXEL_GROUP_COUNT,
                "singleton_pixel_groups": (
                    C.UNIQUE_PIXEL_COUNT - C.MULTI_TEMPLATE_PIXEL_GROUP_COUNT
                ),
                "templates_in_multi_template_groups": (
                    C.TEMPLATE_ROW_COUNT
                    - (C.UNIQUE_PIXEL_COUNT - C.MULTI_TEMPLATE_PIXEL_GROUP_COUNT)
                ),
                "occurrence_rows": C.OCCURRENCE_COUNT,
                "passes": C.ENCODING_PASS_COUNT,
                "encoder_invocations_per_pass": C.UNIQUE_PIXEL_COUNT,
                "singleton_batch_size": C.SINGLETON_ENCODER_BATCH_SIZE,
            },
            "schemas": {
                "pixel_index": "occluded_goal_topological_belief_v2.pixel_index.v1",
                "template_index": "occluded_goal_topological_belief_v2.template_to_pixel_index.v1",
                "occurrence_index": "occluded_goal_topological_belief_v2.occurrence_index.v1",
                "canonical_latent_index": "occluded_goal_topological_belief_v2.canonical_latent_index.v1",
                "canonical_descriptor_index": "occluded_goal_topological_belief_v2.canonical_descriptor_index.v1",
                "encoding_receipt": "occluded_goal_topological_belief_v2.encoding_determinism_receipt.v1",
                "cache_receipt": "occluded_goal_topological_belief_v2.cache_integrity_receipt.v1",
                "canonical_tokens_npz": "occluded_goal_topological_belief_v2.canonical_tokens.v1",
                "canonical_descriptors_npz": "occluded_goal_topological_belief_v2.canonical_descriptors.v1",
            },
            "files": {
                "pixel_index": "pixel_index.json",
                "template_index": "template_to_pixel_index.json",
                "occurrence_index": "occurrence_index.json",
                "canonical_tokens": "canonical_tokens.npz",
                "canonical_descriptors": "canonical_descriptors.npz",
                "canonical_latent_index": "canonical_latent_index.json",
                "canonical_descriptor_index": "canonical_descriptor_index.json",
                "encoding_receipt": "encoding_determinism_receipt.json",
                "cache_receipt": "cache_integrity_receipt.json",
            },
            "npz": {
                "canonical_tokens": {
                    "members": ["schema", "pixel_sha256", "raw_tokens"],
                    "rows": C.UNIQUE_PIXEL_COUNT,
                    "raw_tokens_shape": [C.UNIQUE_PIXEL_COUNT, 768, 1024],
                    "raw_tokens_dtype": "float16",
                },
                "canonical_descriptors": {
                    "members": ["schema", "pixel_sha256", "spatial_descriptors"],
                    "rows": C.UNIQUE_PIXEL_COUNT,
                    "spatial_descriptors_shape": [C.UNIQUE_PIXEL_COUNT, 768, 1024],
                    "spatial_descriptors_dtype": "float32",
                },
            },
            "fields": {
                "pixel_index": sorted(PIXEL_INDEX_FIELDS),
                "pixel_record": sorted(PIXEL_RECORD_FIELDS),
                "template_index": sorted(TEMPLATE_INDEX_FIELDS),
                "template_record": sorted(TEMPLATE_RECORD_FIELDS),
                "occurrence_index": sorted(OCCURRENCE_INDEX_FIELDS),
                "occurrence_record": sorted(OCCURRENCE_RECORD_FIELDS),
                "encoding_receipt": sorted(ENCODING_RECEIPT_FIELDS),
                "pre_outcome_boundary": sorted(PRE_OUTCOME_BOUNDARY_FIELDS),
                "encoding_counts": sorted(ENCODING_COUNT_FIELDS),
                "encoding_pass": sorted(ENCODING_PASS_FIELDS),
                "encoding_pass_record": sorted(ENCODING_PASS_RECORD_FIELDS),
                "encoding_comparisons": sorted(ENCODING_COMPARISON_FIELDS),
                "encoder_binding": sorted(ENCODER_BINDING_FIELDS),
                "canonical_latent_index": sorted(CANONICAL_LATENT_INDEX_FIELDS),
                "canonical_latent_record": sorted(CANONICAL_LATENT_RECORD_FIELDS),
                "canonical_descriptor_index": sorted(CANONICAL_DESCRIPTOR_INDEX_FIELDS),
                "canonical_descriptor_record": sorted(CANONICAL_DESCRIPTOR_RECORD_FIELDS),
                "cache_receipt": sorted(CACHE_RECEIPT_FIELDS),
                "cache_counts": sorted(CACHE_COUNT_FIELDS),
                "cache_gates": sorted(CACHE_GATE_FIELDS),
            },
            "hash_domains": copy.deepcopy(C.CANONICAL_HASH_DOMAINS),
            "receipt_self_digests": False,
        }
    )


def reducer_authority() -> dict[str, Any]:
    return C.attach_content_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.reducer_authority.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "v1_retained_root": v1_retained_root_authority(),
            "graph_manifest": graph_manifest_authority(),
            "query_rows": query_row_authority(),
            "belief_rows": belief_row_authority(),
            "calibration": calibration_authority(),
            "stage_b_trace": stage_b_trace_authority(),
            "cache_gate": cache_gate_authority(),
            "stage_a_authorization_path": "decision.stage_b_authorized",
        }
    )


def validate_graph_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return M1.validate_graph_manifest(value)
    except M1.OccludedGoalMetricsError as exc:
        raise OccludedGoalV2MetricsError(str(exc)) from exc


def validate_query_rows(
    rows: Sequence[Mapping[str, Any]], graph_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    try:
        return M1.validate_query_rows(rows, graph_manifest)
    except M1.OccludedGoalMetricsError as exc:
        raise OccludedGoalV2MetricsError(str(exc)) from exc


def validate_belief_rows(
    rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    graph_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        return M1.validate_belief_rows(rows, query_rows, graph_manifest)
    except M1.OccludedGoalMetricsError as exc:
        raise OccludedGoalV2MetricsError(str(exc)) from exc


def _translate_document(
    value: Mapping[str, Any], *, expected_schema: str, target_schema: str,
    expected_experiment: str, target_experiment: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OccludedGoalV2MetricsError("document must be a mapping")
    try:
        C.validate_content_digest(value)
    except C.OccludedGoalV2ContractError as exc:
        raise OccludedGoalV2MetricsError(str(exc)) from exc
    if value.get("schema") != expected_schema or value.get("experiment_id") != expected_experiment:
        raise OccludedGoalV2MetricsError("document identity drift")
    translated = copy.deepcopy(dict(value))
    translated.pop("content_digest", None)
    translated["schema"] = target_schema
    translated["experiment_id"] = target_experiment
    return C.attach_content_digest(translated)


def _calibration_to_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    return _translate_document(
        value, expected_schema=V2_CALIBRATION_SCHEMA,
        target_schema="occluded_goal_topological_belief_v1.calibration.v1",
        expected_experiment=C.EXPERIMENT_ID, target_experiment=C.V1_EXPERIMENT_ID,
    )


def _stage_a_to_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    return _translate_document(
        value, expected_schema=V2_STAGE_A_METRICS_SCHEMA,
        target_schema="occluded_goal_topological_belief_v1.stage_a_metrics.v1",
        expected_experiment=C.EXPERIMENT_ID, target_experiment=C.V1_EXPERIMENT_ID,
    )


def validate_calibration(
    value: Mapping[str, Any], graph_manifest: Mapping[str, Any],
    query_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    translated = _calibration_to_v1(value)
    try:
        M1.validate_calibration(translated, graph_manifest, query_rows)
    except M1.OccludedGoalMetricsError as exc:
        raise OccludedGoalV2MetricsError(str(exc)) from exc
    return copy.deepcopy(dict(value))


def recompute_stage_a_metrics(
    graph_manifest: Mapping[str, Any], query_rows: Sequence[Mapping[str, Any]],
    belief_rows: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any],
) -> dict[str, Any]:
    calibration_v1 = _calibration_to_v1(calibration)
    try:
        result = M1.recompute_stage_a_metrics(
            graph_manifest, query_rows, belief_rows, calibration_v1
        )
    except M1.OccludedGoalMetricsError as exc:
        raise OccludedGoalV2MetricsError(str(exc)) from exc
    result = copy.deepcopy(result)
    result.pop("content_digest", None)
    result["schema"] = V2_STAGE_A_METRICS_SCHEMA
    result["experiment_id"] = C.EXPERIMENT_ID
    result["calibration_binding"]["content_digest"] = calibration["content_digest"]
    return C.attach_content_digest(result)


def stage_a_authorizes_stage_b(stage_a_metrics: Mapping[str, Any]) -> bool:
    translated = _stage_a_to_v1(stage_a_metrics)
    try:
        return M1.stage_a_authorizes_stage_b(translated)
    except M1.OccludedGoalMetricsError as exc:
        raise OccludedGoalV2MetricsError(str(exc)) from exc


def validate_stage_b_trace_rows(
    graph_manifest: Mapping[str, Any], query_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]], stage_a_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    translated = _stage_a_to_v1(stage_a_metrics)
    try:
        return M1.validate_stage_b_trace_rows(
            graph_manifest, query_rows, rows, translated
        )
    except M1.OccludedGoalMetricsError as exc:
        raise OccludedGoalV2MetricsError(str(exc)) from exc


def recompute_stage_b_metrics(
    graph_manifest: Mapping[str, Any], query_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]], stage_a_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    translated = _stage_a_to_v1(stage_a_metrics)
    try:
        result = M1.recompute_stage_b_metrics(
            graph_manifest, query_rows, trace_rows, translated
        )
    except M1.OccludedGoalMetricsError as exc:
        raise OccludedGoalV2MetricsError(str(exc)) from exc
    result = copy.deepcopy(result)
    result.pop("content_digest", None)
    result["schema"] = V2_STAGE_B_METRICS_SCHEMA
    result["experiment_id"] = C.EXPERIMENT_ID
    result["stage_a_binding"] = stage_a_metrics["content_digest"]
    return C.attach_content_digest(result)


def validate_cache_gate(
    pixel_index: Mapping[str, Any], template_index: Mapping[str, Any],
    occurrence_index: Mapping[str, Any], canonical_latent_index: Mapping[str, Any],
    canonical_descriptor_index: Mapping[str, Any],
    encoding_receipt: Mapping[str, Any], cache_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all canonical pixel, two-pass, fanout, and cache gates."""

    pixels = _validate_content_document(
        pixel_index, fields=PIXEL_INDEX_FIELDS,
        schema="occluded_goal_topological_belief_v2.pixel_index.v1",
        label="pixel_index",
    )
    _validate_file_binding(pixels["observations_binding"], "observations_binding", "observations.npz")
    if pixels["identity_rule"] != C.CANONICAL_HASH_DOMAINS["rgb_pixel_sha256"]:
        raise OccludedGoalV2MetricsError("pixel identity rule drift")
    pixel_records = pixels["records"]
    if not isinstance(pixel_records, list) or len(pixel_records) != C.UNIQUE_PIXEL_COUNT:
        raise OccludedGoalV2MetricsError("pixel index cardinality drift")
    normalized_pixels: list[dict[str, Any]] = []
    all_templates: set[str] = set(); all_observation_rows: set[int] = set()
    prior_sha = ""
    for index, original in enumerate(pixel_records):
        row = _mapping(original, PIXEL_RECORD_FIELDS, f"pixel[{index}]")
        if _integer(row["pixel_index"], "pixel_index") != index:
            raise OccludedGoalV2MetricsError("pixel indices must be contiguous")
        pixel_sha = _sha(row["pixel_sha256"], "pixel_sha256")
        if pixel_sha <= prior_sha:
            raise OccludedGoalV2MetricsError("pixel hashes must be unique lexicographic order")
        prior_sha = pixel_sha
        members = row["member_template_ids"]
        observation_rows = row["observation_row_indices"]
        if (
            not isinstance(members, list) or not members
            or members != sorted(set(members))
            or not all(isinstance(value, str) and value for value in members)
            or row["canonical_template_id"] != members[0]
        ):
            raise OccludedGoalV2MetricsError("canonical template membership drift")
        if (
            not isinstance(observation_rows, list)
            or len(observation_rows) != len(members)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in observation_rows)
            or len(set(observation_rows)) != len(observation_rows)
        ):
            raise OccludedGoalV2MetricsError("observation row membership drift")
        if all_templates.intersection(members) or all_observation_rows.intersection(observation_rows):
            raise OccludedGoalV2MetricsError("pixel groups overlap")
        all_templates.update(members); all_observation_rows.update(observation_rows)
        normalized_pixels.append(row)
    if len(all_templates) != C.TEMPLATE_ROW_COUNT or all_observation_rows != set(range(C.TEMPLATE_ROW_COUNT)):
        raise OccludedGoalV2MetricsError("pixel groups do not cover all template rows")

    templates = _validate_content_document(
        template_index, fields=TEMPLATE_INDEX_FIELDS,
        schema="occluded_goal_topological_belief_v2.template_to_pixel_index.v1",
        label="template_index",
    )
    if templates["pixel_index_binding"] != pixels["content_digest"]:
        raise OccludedGoalV2MetricsError("template/pixel index binding drift")
    template_records = templates["records"]
    if not isinstance(template_records, list) or len(template_records) != C.TEMPLATE_ROW_COUNT:
        raise OccludedGoalV2MetricsError("template index cardinality drift")
    template_by_id: dict[str, dict[str, Any]] = {}
    expected_projection: dict[int, tuple[str, int, str, str]] = {}
    for pixel in normalized_pixels:
        for template_id, row_index in zip(pixel["member_template_ids"], pixel["observation_row_indices"]):
            expected_projection[row_index] = (
                template_id, pixel["pixel_index"], pixel["pixel_sha256"],
                pixel["canonical_template_id"],
            )
    for index, original in enumerate(template_records):
        row = _mapping(original, TEMPLATE_RECORD_FIELDS, f"template[{index}]")
        if row["template_row_index"] != index or (
            row["pixel_template_id"], row["pixel_index"], row["pixel_sha256"],
            row["canonical_template_id"],
        ) != expected_projection[index]:
            raise OccludedGoalV2MetricsError("template fanout projection drift")
        template_by_id[row["pixel_template_id"]] = row
    if len(template_by_id) != C.TEMPLATE_ROW_COUNT:
        raise OccludedGoalV2MetricsError("template identities are not unique")

    occurrences = _validate_content_document(
        occurrence_index, fields=OCCURRENCE_INDEX_FIELDS,
        schema="occluded_goal_topological_belief_v2.occurrence_index.v1",
        label="occurrence_index",
    )
    _validate_file_binding(occurrences["keyframe_index_binding"], "keyframe_index_binding", "keyframe_index.json")
    if occurrences["template_index_binding"] != templates["content_digest"]:
        raise OccludedGoalV2MetricsError("occurrence/template index binding drift")
    occurrence_records = occurrences["records"]
    if not isinstance(occurrence_records, list) or len(occurrence_records) != C.OCCURRENCE_COUNT:
        raise OccludedGoalV2MetricsError("occurrence index cardinality drift")
    capture_ids: set[str] = set()
    for index, original in enumerate(occurrence_records):
        row = _mapping(original, OCCURRENCE_RECORD_FIELDS, f"occurrence[{index}]")
        if row["occurrence_index"] != index:
            raise OccludedGoalV2MetricsError("occurrence indices must be contiguous")
        capture_id = _string(row["capture_id"], "capture_id")
        if capture_id in capture_ids:
            raise OccludedGoalV2MetricsError("duplicate occurrence identity")
        capture_ids.add(capture_id)
        _string(row["episode_id"], "episode_id"); _string(row["node_id"], "node_id")
        _string(row["phase"], "phase"); _finite(row["timestamp_s"], "timestamp_s")
        if not isinstance(row["query_ids"], list) or len(row["query_ids"]) != len(set(row["query_ids"])):
            raise OccludedGoalV2MetricsError("occurrence query identities drift")
        template = template_by_id.get(row["pixel_template_id"])
        if template is None or row["pixel_sha256"] != template["pixel_sha256"] or row["pixel_index"] != template["pixel_index"]:
            raise OccludedGoalV2MetricsError("occurrence pixel mapping drift")

    receipt = _mapping(encoding_receipt, ENCODING_RECEIPT_FIELDS, "encoding_receipt")
    if receipt["schema"] != "occluded_goal_topological_belief_v2.encoding_determinism_receipt.v1" or receipt["experiment_id"] != C.EXPERIMENT_ID:
        raise OccludedGoalV2MetricsError("encoding receipt identity drift")
    _commit(receipt["source_freeze_commit"], "encoding source freeze")
    encoding_observations = _validate_file_binding(
        receipt["observations_binding"], "encoding observations", "observations.npz"
    )
    if encoding_observations != pixels["observations_binding"]:
        raise OccludedGoalV2MetricsError("encoding/pixel observation binding drift")
    if receipt["pixel_index_content_digest"] != pixels["content_digest"] or receipt["hash_domains"] != C.CANONICAL_HASH_DOMAINS:
        raise OccludedGoalV2MetricsError("encoding receipt input/hash binding drift")
    encoder = _mapping(receipt["encoder_binding"], ENCODER_BINDING_FIELDS, "encoder_binding")
    for field in ("checkpoint_sha256", "helper_sha256", "preprocessing_digest"):
        _sha(encoder[field], f"encoder.{field}")
    expected_encoder = C.VJEPA_ENCODER_BINDING
    for field in ("constructor", "checkpoint_sha256", "checkpoint_size_bytes", "helper_path", "helper_sha256", "external_repository_commit"):
        if encoder[field] != expected_encoder[field]:
            raise OccludedGoalV2MetricsError("frozen encoder binding drift")
    if encoder["preprocessing_digest"] != C.PREPROCESSING_DIGEST:
        raise OccludedGoalV2MetricsError("frozen preprocessing binding drift")
    counts = _mapping(receipt["counts"], ENCODING_COUNT_FIELDS, "encoding counts")
    expected_counts = {
        "template_rows": C.TEMPLATE_ROW_COUNT,
        "unique_pixel_rows": C.UNIQUE_PIXEL_COUNT,
        "reused_template_rows": C.REUSED_TEMPLATE_ROW_COUNT,
        "pass_count": C.ENCODING_PASS_COUNT,
        "encoder_invocations_per_pass": C.UNIQUE_PIXEL_COUNT,
        "singleton_batch_size": C.SINGLETON_ENCODER_BATCH_SIZE,
    }
    if counts != expected_counts:
        raise OccludedGoalV2MetricsError("encoding counts drift")
    pre_outcome = _mapping(
        receipt["pre_outcome_boundary"],
        PRE_OUTCOME_BOUNDARY_FIELDS,
        "pre-outcome boundary",
    )
    if pre_outcome != C.PRE_OUTCOME_BOUNDARY_AUTHORITY:
        raise OccludedGoalV2MetricsError("pre-outcome boundary evidence drift")
    passes = receipt["passes"]
    if not isinstance(passes, list) or len(passes) != 2:
        raise OccludedGoalV2MetricsError("encoding requires exactly two passes")
    normalized_passes: list[dict[str, Any]] = []
    expected_shas = [row["pixel_sha256"] for row in normalized_pixels]
    for pass_index, original in enumerate(passes, start=1):
        row = _mapping(original, ENCODING_PASS_FIELDS, f"pass[{pass_index}]")
        if row["pass_index"] != pass_index or row["ordered_pixel_sha256s"] != expected_shas:
            raise OccludedGoalV2MetricsError("encoding pass order drift")
        _string(row["fresh_encoder_instance_id"], "fresh_encoder_instance_id")
        records = row["records"]
        if not isinstance(records, list) or len(records) != C.UNIQUE_PIXEL_COUNT:
            raise OccludedGoalV2MetricsError("encoding pass record count drift")
        normalized_records: list[dict[str, Any]] = []
        for index, original_record in enumerate(records):
            record = _mapping(original_record, ENCODING_PASS_RECORD_FIELDS, f"pass record[{index}]")
            pixel = normalized_pixels[index]
            if record["pixel_index"] != index or record["pixel_sha256"] != pixel["pixel_sha256"] or record["canonical_template_id"] != pixel["canonical_template_id"]:
                raise OccludedGoalV2MetricsError("encoding pass pixel identity drift")
            for field in ("preprocessed_tensor_sha256", "raw_token_sha256", "spatial_descriptor_sha256"):
                _sha(record[field], field)
            normalized_records.append(record)
        if row["canonical_cache_content_digest"] != _cache_content_digest(normalized_records):
            raise OccludedGoalV2MetricsError("canonical cache content digest drift")
        normalized_passes.append({**row, "records": normalized_records})
    if normalized_passes[0]["fresh_encoder_instance_id"] == normalized_passes[1]["fresh_encoder_instance_id"]:
        raise OccludedGoalV2MetricsError("encoding passes did not use fresh encoder instances")
    if normalized_passes[0]["records"] != normalized_passes[1]["records"] or normalized_passes[0]["canonical_cache_content_digest"] != normalized_passes[1]["canonical_cache_content_digest"]:
        raise OccludedGoalV2MetricsError("two-pass encoder evidence differs")
    comparisons = _mapping(receipt["comparisons"], ENCODING_COMPARISON_FIELDS, "encoding comparisons")
    if any(value is not True for value in comparisons.values()):
        raise OccludedGoalV2MetricsError("two-pass comparison gate failed")

    latent = _validate_content_document(
        canonical_latent_index, fields=CANONICAL_LATENT_INDEX_FIELDS,
        schema="occluded_goal_topological_belief_v2.canonical_latent_index.v1",
        label="canonical_latent_index",
    )
    descriptor = _validate_content_document(
        canonical_descriptor_index, fields=CANONICAL_DESCRIPTOR_INDEX_FIELDS,
        schema="occluded_goal_topological_belief_v2.canonical_descriptor_index.v1",
        label="canonical_descriptor_index",
    )
    receipt_sha = _ordinary_json_sha256(receipt)
    for row, file_field, path in (
        (latent, "tokens_file", "canonical_tokens.npz"),
        (descriptor, "descriptors_file", "canonical_descriptors.npz"),
    ):
        _commit(row["source_freeze_commit"], "canonical index source freeze")
        if row["source_freeze_commit"] != receipt["source_freeze_commit"]:
            raise OccludedGoalV2MetricsError("canonical index source-freeze drift")
        if row["pixel_index_binding"] != pixels["content_digest"]:
            raise OccludedGoalV2MetricsError("canonical index pixel binding drift")
        binding = _validate_file_binding(row["encoding_determinism_receipt_binding"], "encoding receipt binding", "encoding_determinism_receipt.json")
        if binding["sha256"] != receipt_sha or binding["bytes"] != len(
            C.canonical_json_bytes(receipt)
        ):
            raise OccludedGoalV2MetricsError("canonical index encoding receipt binding drift")
        _validate_file_binding(row[file_field], file_field, path)
    latent_records = latent["records"]; descriptor_records = descriptor["records"]
    if not isinstance(latent_records, list) or len(latent_records) != C.UNIQUE_PIXEL_COUNT or not isinstance(descriptor_records, list) or len(descriptor_records) != C.UNIQUE_PIXEL_COUNT:
        raise OccludedGoalV2MetricsError("canonical index cardinality drift")
    pass_records = normalized_passes[0]["records"]
    for index in range(C.UNIQUE_PIXEL_COUNT):
        raw = _mapping(latent_records[index], CANONICAL_LATENT_RECORD_FIELDS, f"latent[{index}]")
        desc = _mapping(descriptor_records[index], CANONICAL_DESCRIPTOR_RECORD_FIELDS, f"descriptor[{index}]")
        pixel = normalized_pixels[index]; encoded = pass_records[index]
        if raw != {
            "pixel_index": index, "pixel_sha256": pixel["pixel_sha256"],
            "canonical_template_id": pixel["canonical_template_id"],
            "raw_token_row_index": index, "raw_token_sha256": encoded["raw_token_sha256"],
        } or desc != {
            "pixel_index": index, "pixel_sha256": pixel["pixel_sha256"],
            "canonical_template_id": pixel["canonical_template_id"],
            "spatial_descriptor_row_index": index,
            "spatial_descriptor_sha256": encoded["spatial_descriptor_sha256"],
        }:
            raise OccludedGoalV2MetricsError("canonical cache index projection drift")

    cache = _mapping(cache_receipt, CACHE_RECEIPT_FIELDS, "cache_receipt")
    if cache["schema"] != "occluded_goal_topological_belief_v2.cache_integrity_receipt.v1" or cache["experiment_id"] != C.EXPERIMENT_ID:
        raise OccludedGoalV2MetricsError("cache receipt identity drift")
    _commit(cache["source_freeze_commit"], "cache source freeze")
    if cache["source_freeze_commit"] != receipt["source_freeze_commit"]:
        raise OccludedGoalV2MetricsError("cache/encoding source-freeze drift")
    C.validate_v1_retained_root_binding(cache["v1_retained_root_binding"])
    copied = cache["copied_v1_input_bindings"]
    expected_reusable = {
        row["path"]: row for row in C.V1_RETAINED_LEAVES
        if row["path"] in C.V1_REUSABLE_LEAVES
    }
    if not isinstance(copied, list) or len(copied) != len(expected_reusable):
        raise OccludedGoalV2MetricsError("copied V1 input binding count drift")
    observed_copies: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(copied):
        binding = _validate_file_binding(value, f"copied input[{index}]")
        observed_copies[binding["path"]] = binding
    if observed_copies != expected_reusable:
        raise OccludedGoalV2MetricsError("copied V1 input binding drift")
    encoding_binding = _validate_file_binding(cache["encoding_determinism_receipt_binding"], "cache encoding receipt", "encoding_determinism_receipt.json")
    if encoding_binding["sha256"] != receipt_sha or encoding_binding["bytes"] != len(
        C.canonical_json_bytes(receipt)
    ):
        raise OccludedGoalV2MetricsError("cache/encoding receipt SHA drift")
    if (
        cache["pixel_index_content_digest"] != pixels["content_digest"]
        or cache["template_index_content_digest"] != templates["content_digest"]
        or cache["occurrence_index_content_digest"] != occurrences["content_digest"]
        or cache["canonical_latent_index_content_digest"] != latent["content_digest"]
        or cache["canonical_descriptor_index_content_digest"] != descriptor["content_digest"]
    ):
        raise OccludedGoalV2MetricsError("cache receipt index binding drift")
    cache_counts = _mapping(cache["counts"], CACHE_COUNT_FIELDS, "cache counts")
    if cache_counts != {
        "template_rows": C.TEMPLATE_ROW_COUNT,
        "unique_pixel_rows": C.UNIQUE_PIXEL_COUNT,
        "reused_template_rows": C.REUSED_TEMPLATE_ROW_COUNT,
        "occurrence_rows": C.OCCURRENCE_COUNT,
        "canonical_token_rows": C.UNIQUE_PIXEL_COUNT,
        "canonical_descriptor_rows": C.UNIQUE_PIXEL_COUNT,
        "encoder_invocations_per_pass": C.UNIQUE_PIXEL_COUNT,
        "multi_template_pixel_groups": C.MULTI_TEMPLATE_PIXEL_GROUP_COUNT,
        "singleton_pixel_groups": C.UNIQUE_PIXEL_COUNT - C.MULTI_TEMPLATE_PIXEL_GROUP_COUNT,
        "templates_in_multi_template_groups": C.TEMPLATE_ROW_COUNT - (
            C.UNIQUE_PIXEL_COUNT - C.MULTI_TEMPLATE_PIXEL_GROUP_COUNT
        ),
    }:
        raise OccludedGoalV2MetricsError("cache counts drift")
    gates = _mapping(cache["gates"], CACHE_GATE_FIELDS, "cache gates")
    if any(value is not True for value in gates.values()):
        raise OccludedGoalV2MetricsError("cache integrity gate failed")
    return {
        "pass": True,
        "pixel_index_content_digest": pixels["content_digest"],
        "template_index_content_digest": templates["content_digest"],
        "occurrence_index_content_digest": occurrences["content_digest"],
        "encoding_determinism_receipt_sha256": receipt_sha,
        "cache_integrity_receipt_sha256": _ordinary_json_sha256(cache),
        "canonical_latent_index_content_digest": latent["content_digest"],
        "canonical_descriptor_index_content_digest": descriptor["content_digest"],
        "counts": copy.deepcopy(cache_counts),
    }


# Public spelling used by the independent persisted-evidence reducer.  Keep a
# single implementation so the runner and reducer cannot diverge.
validate_cache_gate_evidence = validate_cache_gate


__all__ = [
    "BELIEF_FIELDS", "CACHE_RECEIPT_FIELDS", "CALIBRATION_FIELDS",
    "CANONICAL_DESCRIPTOR_INDEX_FIELDS", "CANONICAL_LATENT_INDEX_FIELDS",
    "CURRENT_IDENTITY_PROJECTION_FIELDS", "EDGE_FIELDS", "ENCODING_RECEIPT_FIELDS",
    "GRAPH_FIELDS", "GRAPH_ROOT_FIELDS", "IDENTITY_COMPARISON_FIELDS",
    "IDENTITY_DISJOINTNESS_FIELDS", "NODE_FIELDS", "OCCURRENCE_INDEX_FIELDS",
    "PIXEL_INDEX_FIELDS", "QUERY_FIELDS", "STAGE_B_TRACE_FIELDS",
    "TEMPLATE_INDEX_FIELDS", "OccludedGoalV2MetricsError", "belief_row_authority",
    "cache_gate_authority", "calibration_authority",
    "deterministic_action_history_position_mapping", "graph_manifest_authority",
    "query_row_authority", "recompute_stage_a_metrics", "recompute_stage_b_metrics",
    "reducer_authority", "stage_a_authorizes_stage_b", "stage_a_condition_ids",
    "stage_b_trace_authority", "v1_retained_root_authority",
    "validate_belief_rows", "validate_cache_gate", "validate_cache_gate_evidence",
    "validate_calibration",
    "validate_graph_manifest", "validate_query_rows", "validate_stage_b_trace_rows",
]
