"""Frozen contract for one protected Camera-only broad adaptation attempt."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/camera_adapt_design"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_protected_camera_adaptation_v1"
CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v1.py"
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v1.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v1.py"
DIAGNOSTIC_CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1.py"
DIAGNOSTIC_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1.py"
DIAGNOSTIC_TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1.py"
DIAGNOSTIC_SOURCE_SHA256 = {
    DIAGNOSTIC_CONTRACT_RELATIVE_PATH: "4eb86e6808017baf54381678881b823f295f792f0b39d522e796c84f33e95e2e",
    DIAGNOSTIC_RUNNER_RELATIVE_PATH: "cb268c73a5bad166b49ec6137a6680a697864fcf8475d6d997586570bddf7a23",
    DIAGNOSTIC_TEST_RELATIVE_PATH: "4d9896d2acbaed6c6d115bb2739ce4136d34d9e2974c88867281ee58a73bad49",
}


def _load_exact(path: str, name: str, digest: str) -> ModuleType:
    source = ROOT / path
    raw = source.read_bytes()
    if source.is_symlink() or not source.is_file() or hashlib.sha256(raw).hexdigest() != digest:
        raise PermissionError(f"frozen source changed: {path}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_diagnostic = _load_exact(
    DIAGNOSTIC_CONTRACT_RELATIVE_PATH,
    "_lewm_protected_camera_adaptation_update0_contract",
    DIAGNOSTIC_SOURCE_SHA256[DIAGNOSTIC_CONTRACT_RELATIVE_PATH],
)
_v1 = _diagnostic._v1

UPDATE0_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1_terminal_audit_2026-07-15.json"
UPDATE0_AUDIT_BINDING = {
    "path": UPDATE0_AUDIT_RELATIVE_PATH,
    "file_sha256": "52d6ac4a7287b9cb9bd33fbdd3eadbb939f9368643953fe61866777f620914bf",
    "content_sha256": "86276a89a6cc637aefdee798c916eeaafe1d30ca2c69038b6debdaa8332f8fe0",
    "byte_count": 13_422,
}
UPDATE0_ROOT_RELATIVE_PATH = _diagnostic.OUTPUT_ROOT_RELATIVE_PATH
UPDATE0_STATE_SHA256 = _diagnostic.UPDATE0_STATE_SHA256
UPDATE0_TERMINAL_ARTIFACTS = {
    "access": {"path": "access.json", "file_sha256": "b5bb878b7aa58f943dd74950d467517cb59c2087e6cdc24d9dab2ca53d0fb621", "content_sha256": "18472539607fce69361c1d95e0281b657941736174c86c965130606a0689f2ea", "byte_count": 404_716, "schema": _diagnostic.ACCESS_SCHEMA},
    "completed": {"path": "completed.json", "file_sha256": "f33c2fc4426d2b7ce85206a9c54cf59db1e21d4a3c7a88ec782e437e6424493e", "content_sha256": "32c17f68b580f62507c6580093806d85d24b9ff214638c754233cbe16c360ee5", "byte_count": 1_629, "schema": _diagnostic.COMPLETION_SCHEMA},
    "reservation": {"path": "reservation.json", "file_sha256": "92b9be478a7b3708d7be9256a30f62dd5f2bc288f529aa02c9dd3dd469271265", "content_sha256": "3d9d28d6cad0271315510934001abcfb3dc8656e9007a25d9e0c1fbcca5a9051", "byte_count": 10_693, "schema": _diagnostic.RESERVATION_SCHEMA},
    "result": {"path": "result.json", "file_sha256": "b6dae1489b5ad42f0b763912089731de95d6147aebda434d9ca998fcfbe37af4", "content_sha256": "753dc6c1812c04f0ed30b4c7a2ffe5e3de0c644d72795e5700dcb66dea0b68af", "byte_count": 32_767, "schema": _diagnostic.RESULT_SCHEMA},
}

OUTPUT_ROOT_RELATIVE_PATH = ".generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v1"
REVIEW_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v1_independent_review_2026-07-15.json"
AUTHORIZATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v1_execution_authorization_2026-07-15.json"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_independent_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_camera_snapshot_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_checkpoint_metrics_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
SOURCE_PATHS = tuple(dict.fromkeys((CONTRACT_RELATIVE_PATH, RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH, UPDATE0_AUDIT_RELATIVE_PATH, *_diagnostic.SOURCE_PATHS)))

MAXIMUM_UPDATE = 4_000
CHECKPOINT_UPDATES = (100, 400, 1_000, 2_000, 4_000)
SCHEDULE_PREFIX_INDICES_SHA256 = "14e83952c758c2ee4118d38c116625feb351813bc24b017d7b47f53426df47ab"
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = {
    100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    400: "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
    1_000: "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
    2_000: "caa943d8c98ad960c561ee8d736a7265a45ad12006a54fe135a0e3c2b2cad434",
    4_000: SCHEDULE_PREFIX_INDICES_SHA256,
}
ENCODER_LR_SCALE = 0.01
POST_CLIP_NORM_ASSERTION_TOLERANCE = 1e-5
TRAINABLE_PARAMETER_PREFIXES = ("encoder.", "evidence_head.")
FROZEN_STATE_PREFIXES = ("bev_decoder.", "predictor.", "occupancy_head.", "target_encoder.", "target_bev_decoder.")
EXPECTED_PARAMETER_COUNTS = {"encoder": 2_747_520, "evidence_head": 357_993}
EXPECTED_PARAMETER_TENSOR_COUNTS = {"encoder": 78, "evidence_head": 14}
OPTIMIZER_CONTRACT = {
    "name": "AdamW", "group_order": ["evidence_head", "encoder"],
    "head_learning_rate": "exact_v1_learning_rate(update)", "encoder_learning_rate_scale": ENCODER_LR_SCALE,
    "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 1e-4, "amsgrad": False,
    "precision": "float32", "autocast": False, "independent_group_clip_norm": 1.0,
    "microbatch_size": 4, "accumulation_steps": 4, "effective_batch_size": 16,
    "maximum_updates": MAXIMUM_UPDATE, "ema_update_count": 0,
    "post_clip_norm_assertion_tolerance": POST_CLIP_NORM_ASSERTION_TOLERANCE,
}
DOWNSTREAM_DENIALS = {
    "automatic_successor_authorized": False, "calibration_authorized": False,
    "probability_calibration_authorized": False, "jepa_training_authorized": False,
    "jepa_objective_authorized": False, "jepa_backward_authorized": False,
    "ema_update_authorized": False, "architecture_change_authorized": False,
    "architecture_mutation_authorized": False, "data_mutation_authorized": False,
    "data_refinement_authorized": False, "schedule_extension_authorized": False,
    "training_extension_authorized": False, "closest_checkpoint_promotion_authorized": False,
    "soft_promotion_authorized": False, "no_pass_promotion_authorized": False,
    "runtime_candidate_use_authorized": False, "g2_authorized": False,
    "navigation_authorized": False, "heldout_authorized": False,
    "hardware_authorized": False, "production_authorized": False,
    "promotion_authorized": False, "deployment_authorized": False, "retry_authorized": False,
}
REVIEW_AUTHORITY = {
    "execution_authorized": False, "training_authorized": False, "gpu0_authorized": False,
    "development_payload_read_authorized": False, "checkpoint_selection_authorized": False,
    "generated_mutation_authorized": False, "output_root_observed_absent_at_authorization": False,
    **DOWNSTREAM_DENIALS,
}
EXECUTION_AUTHORITY = {
    "one_exact_development_attempt_authorized": True, "protected_camera_training_authorized": True,
    "gpu0_training_authorized": True, "development_payload_read_authorized": True,
    "development_rgb_decode_authorized": True, "checkpoint_selection_authorized": True,
    "camera_checkpoint_migration_authorized": True, "generated_mutation_authorized": True,
    "mutation_scope": OUTPUT_ROOT_RELATIVE_PATH, "requires_absent_output_root_before_reservation": True,
    "output_root_observed_absent_at_authorization": True,
    **DOWNSTREAM_DENIALS,
}
_JEPA_SENTINEL = {
    "prediction_valid_cell_count": 1, "target_cross_sample_std_mean": 1.0,
    "target_cross_sample_effective_rank": 8.0, "warped_persistence_target_change": 1.0,
    "prediction_to_warped_persistence_ratio": 0.0,
    "wrong_action_advantage_over_target_change": 1.0,
    "wrong_commanded_delta_advantage_over_target_change": 1.0,
    "wrong_action_prediction_sensitivity": 1.0,
    "wrong_commanded_delta_prediction_sensitivity": 1.0,
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(core), "content_sha256": canonical_json_sha256(core)}


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    value = _diagnostic.parse_canonical_json(raw, name=name)
    return dict(value)


def artifact_binding(path: str, raw: bytes, *, content_sha256: str) -> dict[str, Any]:
    return _diagnostic.artifact_binding(path, raw, content_sha256=content_sha256)


def validate_binding(value: object, *, path: str | None = None) -> dict[str, Any]:
    return _diagnostic.validate_binding(value, path=path)


def expected_raw_authority() -> dict[str, Any]:
    value = _diagnostic.expected_raw_authority()
    value["grant"] = {
        "source_raw_grant_remains_false": True, "narrow_grant_created_by_this_authorization": True,
        "allowed_roles": ["train", "checkpoint_selection"],
        "allowed_operations": ["development_rgb_decode", "protected_camera_training", "physical_checkpoint_selection"],
        "g2_navigation_heldout_or_production_use": False,
    }
    return value


def expected_camera_authority() -> dict[str, Any]:
    return _diagnostic.expected_camera_authority()


def predecessor_contract() -> dict[str, Any]:
    return {"terminal_audit": dict(UPDATE0_AUDIT_BINDING), "root": UPDATE0_ROOT_RELATIVE_PATH, "terminal_artifacts": copy.deepcopy(UPDATE0_TERMINAL_ARTIFACTS), "update0_state_sha256": UPDATE0_STATE_SHA256}


def science_contract() -> dict[str, Any]:
    return {
        "purpose": "adapt_only_the_migrated_camera_encoder_and_evidence_head_before_separate_jepa_training",
        "predecessor": predecessor_contract(),
        "data": {"train_role": copy.deepcopy(_v1.ROLE_COUNTS["train"]), "selection_role": copy.deepcopy(_v1.ROLE_COUNTS["checkpoint_selection"]), "probability_calibration_open_count": 0, "new_data_or_refinement": False},
        "initial_state_sha256": UPDATE0_STATE_SHA256,
        "trainable_parameter_prefixes": list(TRAINABLE_PARAMETER_PREFIXES),
        "frozen_state_prefixes": list(FROZEN_STATE_PREFIXES),
        "expected_parameter_counts": dict(EXPECTED_PARAMETER_COUNTS),
        "expected_parameter_tensor_counts": dict(EXPECTED_PARAMETER_TENSOR_COUNTS),
        "camera_loss": {"source": _v1.LOSS_RELATIVE_PATH, "backward_scalar": "observable_camera_ray_v4.total", "terms": list(_v1.CAMERA_TERMS), "current_next_weights": [0.5, 0.5], "microbatch_scalar_weights": [0.25] * 4, "jepa_objective_count": 0, "jepa_backward_count": 0, "ema_update_count": 0},
        "schedule": {"source": _diagnostic.V4_SCHEDULE_BINDING, "use_exact_prefix_updates": MAXIMUM_UPDATE, "presentation_count": MAXIMUM_UPDATE * 16, "presentation_indices_sha256": SCHEDULE_PREFIX_INDICES_SHA256, "checkpoint_updates": list(CHECKPOINT_UPDATES), "checkpoint_prefix_sha256": {str(key): value for key, value in CHECKPOINT_SCHEDULE_PREFIX_SHA256.items()}},
        "optimizer": dict(OPTIMIZER_CONTRACT),
        "selection": {"role": "checkpoint_selection", "pair_count": 495, "unique_endpoint_count": 924, "scopes": list(_v1.SCOPES), "metrics": "exact_existing_physical_metrics", "physical_margins": "exact_v1_evaluate_checkpoint_scope", "wrong_rgb_mapping": "cyclic_plus_one_within_family", "rule": "stop_and_select_earliest_all_nine_physical_pass", "soft_promotion": False},
        "maximum_attempts": 1, "retry_authorized": False, "authority": dict(DOWNSTREAM_DENIALS),
    }


def validate_update0_audit(raw: bytes) -> dict[str, Any]:
    if len(raw) != UPDATE0_AUDIT_BINDING["byte_count"] or hashlib.sha256(raw).hexdigest() != UPDATE0_AUDIT_BINDING["file_sha256"]:
        raise PermissionError("update0 terminal audit byte binding changed")
    value = parse_canonical_json(raw, name="update0 terminal audit")
    decision, operation, inventory, authority = value.get("successor_decision", {}), value.get("operation_boundary", {}), value.get("terminal_inventory", {}), value.get("authority", {})
    if value.get("content_sha256") != UPDATE0_AUDIT_BINDING["content_sha256"] or value.get("verdict") != "PASS_CONFIRMED_READ_ONLY_UPDATE0_DIAGNOSTIC_AND_SEPARATED_SUCCESSOR_DECISION" or value.get("bindings", {}).get("terminal_artifacts") != UPDATE0_TERMINAL_ARTIFACTS or inventory.get("exact_paths") != ["access.json", "completed.json", "reservation.json", "result.json"] or inventory.get("exact_file_count") != 4 or inventory.get("exact_directory_count_including_root") != 1 or operation.get("optimizer_step_count") != 0 or operation.get("persistent_learned_state_mutation_count") != 0 or operation.get("state_sha256_after_all_three_clones") != UPDATE0_STATE_SHA256 or decision.get("decision") != "SEPARATE_CAMERA_ADAPTATION_BEFORE_FROZEN_CAMERA_JEPA_TRAINING" or decision.get("no_new_architecture_required_now") is not True or decision.get("no_new_data_or_data_refinement") is not True or authority.get("automatic_successor_authorized") is not False or authority.get("heldout_authorized") is not False:
        raise PermissionError("update0 terminal conclusion changed")
    return value


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    _diagnostic.current_source_bindings(root)
    audit_raw = (root / UPDATE0_AUDIT_RELATIVE_PATH).read_bytes()
    validate_update0_audit(audit_raw)
    result = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}
    if any(result.get(path) != digest for path, digest in DIAGNOSTIC_SOURCE_SHA256.items()) or result[UPDATE0_AUDIT_RELATIVE_PATH] != UPDATE0_AUDIT_BINDING["file_sha256"]:
        raise PermissionError("frozen predecessor source changed")
    return result


def validate_review(value: object, *, expected_sources: Mapping[str, str]) -> dict[str, Any]:
    fields = {"schema", "status", "implementation_author", "reviewer", "reviewed_sources", "science_contract", "source_only", "findings", "authority", "content_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("independent review fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    reviewer = value["reviewer"]
    if value["schema"] != REVIEW_SCHEMA or value["status"] != "PASS" or value["implementation_author"] != IMPLEMENTATION_AUTHOR or type(reviewer) is not str or not reviewer.startswith("/root/") or reviewer == IMPLEMENTATION_AUTHOR or value["reviewed_sources"] != dict(expected_sources) or value["science_contract"] != science_contract() or value["source_only"] is not True or value["findings"] != [] or value["authority"] != REVIEW_AUTHORITY or not _v1.is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PermissionError("independent review did not pass exact sources")
    return dict(value)


def validate_authorization(value: object, *, review_binding: Mapping[str, Any], reviewer: str) -> dict[str, Any]:
    fields = {"schema", "status", "authorizer", "independent_review", "predecessor", "raw", "camera", "experiment", "authority", "content_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("execution authorization fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    authorizer = value["authorizer"]
    if value["schema"] != AUTHORIZATION_SCHEMA or value["status"] != "authorized_one_exact_protected_camera_adaptation_attempt" or type(authorizer) is not str or not authorizer.startswith("/root/") or authorizer in {IMPLEMENTATION_AUTHOR, reviewer} or value["independent_review"] != dict(review_binding) or value["predecessor"] != predecessor_contract() or value["raw"] != expected_raw_authority() or value["camera"] != expected_camera_authority() or value["experiment"] != science_contract() or value["authority"] != EXECUTION_AUTHORITY or not _v1.is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PermissionError("execution authorization changed")
    return dict(value)


def learning_rates(update: int) -> tuple[float, float]:
    if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATE:
        raise ValueError("protected Camera update must lie in [1,4000]")
    head = _v1.learning_rate(update)
    encoder = ENCODER_LR_SCALE * head
    if not math.isfinite(encoder) or encoder <= 0.0:
        raise ValueError("protected encoder learning rate is invalid")
    return head, encoder


def parameter_partition(name: str) -> str:
    matches = [prefix.removesuffix(".") for prefix in (*TRAINABLE_PARAMETER_PREFIXES, *FROZEN_STATE_PREFIXES) if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"model state escaped the protected partition: {name}")
    return matches[0]


def validate_checkpoint_prefix(updates: Sequence[int]) -> tuple[int, ...]:
    result = tuple(updates)
    if not result or result != CHECKPOINT_UPDATES[: len(result)]:
        raise ValueError("checkpoint updates must be one nonempty fixed prefix")
    return result


def evaluate_physical_scopes(scopes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if type(scopes) is not dict or tuple(scopes) != _v1.SCOPES:
        raise ValueError("physical scope order changed")
    rows, passed = {}, {}
    for scope in _v1.SCOPES:
        evaluation = _v1.evaluate_checkpoint_scope({"physical": dict(scopes[scope]), "jepa": dict(_JEPA_SENTINEL)})
        rows[scope] = {"physical_margins": evaluation["physical_margins"], "passes": all(value >= 0.0 for value in evaluation["physical_margins"])}
        passed[scope] = rows[scope]["passes"]
    return {"scope_evaluations": rows, "physical_pass_by_scope": passed, "physical_pass_count": sum(passed.values()), "all_nine_physical_pass": all(passed.values())}


__all__ = [name for name in globals() if name.isupper()] + ["artifact_binding", "canonical_json_bytes", "canonical_json_sha256", "current_source_bindings", "evaluate_physical_scopes", "expected_camera_authority", "expected_raw_authority", "learning_rates", "parameter_partition", "parse_canonical_json", "predecessor_contract", "science_contract", "validate_authorization", "validate_binding", "validate_checkpoint_prefix", "validate_review", "validate_update0_audit", "with_content_sha256"]
