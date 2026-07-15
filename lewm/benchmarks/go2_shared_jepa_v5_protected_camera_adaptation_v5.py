"""Frozen contract for one Camera V5 native-schedule completion attempt."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_native_schedule_completion"
CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
PREREGISTRATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_native_schedule_completion_preregistration_2026-07-16.md"
PREREGISTRATION_COMMIT = "eb662d518dabe4989a711d4545d469b930dbe79f"

V3_CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v3.py"
V3_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v3.py"
V3_TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v3.py"
V3_SOURCE_SHA256 = {
    V3_CONTRACT_RELATIVE_PATH: "9fd912538d94944881bd8a2789023470345208abb55a94466fbecc9d82afa0be",
    V3_RUNNER_RELATIVE_PATH: "921f88a149940adb9df79684bdc810f87a97fe657b4d321a4df76c69d3f4af61",
    V3_TEST_RELATIVE_PATH: "06f0cf6d1fe74b59f9de880d9ee9adcc27d8863537c3170f3163946c1eb9d656",
}


def _load_exact(path: str, name: str, digest: str) -> ModuleType:
    source = ROOT / path
    raw = source.read_bytes()
    if source.is_symlink() or not source.is_file() or hashlib.sha256(raw).hexdigest() != digest:
        raise PermissionError(f"frozen Camera V3 source changed: {path}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen Camera V3 source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v3_contract = _load_exact(
    V3_CONTRACT_RELATIVE_PATH,
    "_lewm_protected_camera_adaptation_v3_contract_for_v5",
    V3_SOURCE_SHA256[V3_CONTRACT_RELATIVE_PATH],
)
_v1 = _v3_contract._v1

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v5_native_schedule_completion"
)
REVIEW_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_independent_review_2026-07-16.json"
AUTHORIZATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_execution_authorization_2026-07-16.json"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_independent_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_camera_snapshot_v1"
METRIC_SIDECAR_SCHEMA = f"{SCHEMA_PREFIX}_checkpoint_metric_sidecar_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_checkpoint_metrics_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

V4_TERMINAL_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_terminal_audit_2026-07-16.json"
V3_TERMINAL_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v3_terminal_audit_2026-07-15.json"
V3_WARMSTART_BLOCK_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v3_warmstart_science_review_2026-07-15.json"
MATCHED_V4_TERMINAL_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_matched_training_v4_terminal_numeric_failure_audit_2026-07-15.json"
SCHEDULE_RELATIVE_PATH = ".generated/go2_shared_observable_camera_ray_jepa_v5/matched_training_v4/schedule.json"
N320_RESULT_RELATIVE_PATH = ".generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1/result.json"
N320_GATE_RELATIVE_PATH = ".generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1/gate.json"
N320_CHECKPOINT_RELATIVE_PATH = ".generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1/checkpoint.pt"
V3_UPDATE1000_SIDECAR_RELATIVE_PATH = ".generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v3/checkpoints/update_1000.metrics.json"
V3_UPDATE4000_SIDECAR_RELATIVE_PATH = ".generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v3/checkpoints/update_4000.metrics.json"

FIXED_EVIDENCE_SHA256 = {
    PREREGISTRATION_RELATIVE_PATH: "e45c981e7def6d87d2b3ebc83b9a8ecb65710b0677360377779c34c98281c70d",
    V4_TERMINAL_AUDIT_RELATIVE_PATH: "5d0d4a1cf966e5f612e15da9cacbc705ace4f629183038c6743f0e2fac1b355f",
    V3_TERMINAL_AUDIT_RELATIVE_PATH: "3eb77a83ede536680e03363521f73f41205ac17d845a0e28251a40dcf82f77ab",
    V3_WARMSTART_BLOCK_RELATIVE_PATH: "b37829a2c311533240f6191c099d79411d453adbde43cd0304f1e5c74bd676d7",
    MATCHED_V4_TERMINAL_AUDIT_RELATIVE_PATH: "70371a2cd09e912e05ba0b5efdf75ee2de38cc89347e8111fff303e2a55c485b",
    SCHEDULE_RELATIVE_PATH: "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270",
    N320_RESULT_RELATIVE_PATH: "9fb603566002cd57797895fe27cb2ccabf0e39484c2a8e705c99982933aa3a44",
    N320_GATE_RELATIVE_PATH: "4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6",
    N320_CHECKPOINT_RELATIVE_PATH: "ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0",
    V3_UPDATE1000_SIDECAR_RELATIVE_PATH: "26f5e06d141b974b335d7f056b5392bd308342082bf832acfbd83f70b451e926",
    V3_UPDATE4000_SIDECAR_RELATIVE_PATH: "5b83a880d13983c398083525fb05d939673cad2a86ec38596a7f279670cf1a05",
}
FIXED_EVIDENCE_CONTENT_SHA256 = {
    V4_TERMINAL_AUDIT_RELATIVE_PATH: "246e50b986316f7dc8c806960e8661cf83417fd34c0baa269d83b221cf98d5e2",
    V3_TERMINAL_AUDIT_RELATIVE_PATH: "a5a86d5260c519003f7a5efeb1d21c535afeb65ef7596a627174a41c633be2ac",
    V3_WARMSTART_BLOCK_RELATIVE_PATH: "f317c80e527706faf267ba0be3ab8a19187aeeabba49896f4f7d0722aac98168",
    MATCHED_V4_TERMINAL_AUDIT_RELATIVE_PATH: "ae86d1479fc3016eb96302304e079b7bf9647e26b24b3d860e7d32013bf9c6f4",
    SCHEDULE_RELATIVE_PATH: "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15",
    N320_RESULT_RELATIVE_PATH: "8be838e6b558b396d926f24432d95e1ba9f691d12752cde088e061f13d97d768",
    N320_GATE_RELATIVE_PATH: "76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b",
    V3_UPDATE1000_SIDECAR_RELATIVE_PATH: "26e369149c30afdaf676a6ad111f0914bc410896ec6c0ac145db2a221a7e394a",
    V3_UPDATE4000_SIDECAR_RELATIVE_PATH: "55dda1394ecb201c37ade773c76e1b30c3238e3c33d5d68fcc09d90266141f1a",
}
SCHEDULE_FILE_SHA256 = FIXED_EVIDENCE_SHA256[SCHEDULE_RELATIVE_PATH]
SCHEDULE_CONTENT_SHA256 = FIXED_EVIDENCE_CONTENT_SHA256[SCHEDULE_RELATIVE_PATH]
FIXED_EVIDENCE_BYTE_COUNT = {
    PREREGISTRATION_RELATIVE_PATH: 7_846,
    V4_TERMINAL_AUDIT_RELATIVE_PATH: 20_077,
    V3_TERMINAL_AUDIT_RELATIVE_PATH: 15_957,
    V3_WARMSTART_BLOCK_RELATIVE_PATH: 3_287,
    MATCHED_V4_TERMINAL_AUDIT_RELATIVE_PATH: 21_517,
    SCHEDULE_RELATIVE_PATH: 607_373,
    N320_RESULT_RELATIVE_PATH: 203_833,
    N320_GATE_RELATIVE_PATH: 7_960,
    N320_CHECKPOINT_RELATIVE_PATH: 13_777_100,
    V3_UPDATE1000_SIDECAR_RELATIVE_PATH: 15_204,
    V3_UPDATE4000_SIDECAR_RELATIVE_PATH: 15_224,
}
N320_CHECKPOINT_CONTENT_SHA256 = "9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b"

UPDATE0_STATE_SHA256 = "e03613bf5da2d93910630a0e2b98799a907f9a2b4767a0c2c36b1fa942cd2a87"
MAXIMUM_UPDATE = 8_000
CHECKPOINT_UPDATES = (100, 400, 1_000, 4_000, 6_000, 8_000)
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = {
    100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    400: "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
    1_000: "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
    4_000: "14e83952c758c2ee4118d38c116625feb351813bc24b017d7b47f53426df47ab",
    6_000: "5ba218ed5335c357b60d5f8c2f2d0a3f9e1171631cc299e5d0747ae858e92c50",
    8_000: "a6f4fda5eb570336fb360631af3629832cccbe4cba21bdbb325dcb8a21963663",
}
SCHEDULE_PREFIX_INDICES_SHA256 = CHECKPOINT_SCHEDULE_PREFIX_SHA256[MAXIMUM_UPDATE]
PRESENTATION_COUNT = 128_000
MARGIN_COUNT = 189
ENCODER_LR_SCALE = 1.0
TRAINABLE_PARAMETER_PREFIXES = tuple(_v3_contract.TRAINABLE_PARAMETER_PREFIXES)
FROZEN_STATE_PREFIXES = tuple(_v3_contract.FROZEN_STATE_PREFIXES)
EXPECTED_PARAMETER_COUNTS = dict(_v3_contract.EXPECTED_PARAMETER_COUNTS)
EXPECTED_PARAMETER_TENSOR_COUNTS = dict(_v3_contract.EXPECTED_PARAMETER_TENSOR_COUNTS)
OPTIMIZER_CONTRACT = copy.deepcopy(_v3_contract.OPTIMIZER_CONTRACT)
OPTIMIZER_CONTRACT["maximum_updates"] = MAXIMUM_UPDATE
POST_CLIP_NORM_ASSERTION_TOLERANCE = _v3_contract.POST_CLIP_NORM_ASSERTION_TOLERANCE
DOWNSTREAM_DENIALS = dict(_v3_contract.DOWNSTREAM_DENIALS)
REVIEW_AUTHORITY = {**dict(_v3_contract.REVIEW_AUTHORITY), "native_schedule_completion_authorized": False}
EXECUTION_AUTHORITY = {
    **dict(_v3_contract.EXECUTION_AUTHORITY),
    "mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
    "native_schedule_completion_authorized": True,
}
CONTROL_ACTION_CONTINUE = _v3_contract.CONTROL_ACTION_CONTINUE
CONTROL_ACTION_QUALIFY = _v3_contract.CONTROL_ACTION_QUALIFY
CONTROL_ACTION_STOP_PROGRESS = _v3_contract.CONTROL_ACTION_STOP_PROGRESS
CONTROL_ACTION_STOP_MAXIMUM = _v3_contract.CONTROL_ACTION_STOP_MAXIMUM
METRIC_SIDECAR_DIRECTORY = "checkpoints"
TERMINAL_DIRECTORIES_INCLUDING_ROOT = (".", "checkpoints")

V3_PROGRESS_BASELINES = {
    1_000: {"passed_margin_count": 106, "total_shortfall": 49.09939462151839, "worst_margin": -7.944758415222166},
    4_000: {"passed_margin_count": 134, "total_shortfall": 19.869159033399846, "worst_margin": -4.920835733413693},
}
SOURCE_PATHS = tuple(dict.fromkeys((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    *_v3_contract.SOURCE_PATHS,
    *FIXED_EVIDENCE_SHA256,
)))


def canonical_json_bytes(value: object) -> bytes:
    return _v3_contract.canonical_json_bytes(value)


def canonical_json_sha256(value: object) -> str:
    return _v3_contract.canonical_json_sha256(value)


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return _v3_contract.with_content_sha256(core)


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    return _v3_contract.parse_canonical_json(raw, name=name)


def artifact_binding(path: str, raw: bytes, *, content_sha256: str) -> dict[str, Any]:
    return _v3_contract.artifact_binding(path, raw, content_sha256=content_sha256)


def validate_binding(value: object, *, path: str | None = None) -> dict[str, Any]:
    return _v3_contract.validate_binding(value, path=path)


def _fixed_evidence_bindings() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path, file_sha256 in FIXED_EVIDENCE_SHA256.items():
        value: dict[str, Any] = {
            "path": path,
            "file_sha256": file_sha256,
            "byte_count": FIXED_EVIDENCE_BYTE_COUNT[path],
        }
        if path in FIXED_EVIDENCE_CONTENT_SHA256:
            value["content_sha256"] = FIXED_EVIDENCE_CONTENT_SHA256[path]
        if path == N320_CHECKPOINT_RELATIVE_PATH:
            value["content_sha256"] = N320_CHECKPOINT_CONTENT_SHA256
            value["content_sha256_validated_through_bound_n320_result_not_deserialization"] = True
        result[path] = value
    return result


def evidence_contract() -> dict[str, Any]:
    return {
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "fixed_artifacts": _fixed_evidence_bindings(),
        "v3_progress_baselines": {str(update): dict(value) for update, value in V3_PROGRESS_BASELINES.items()},
        "v3_warm_start_authorized": False,
        "v4_checkpoint_qualified": False,
        "matched_v4_outcome_reused": False,
        "matched_v4_schedule_only_reused": True,
        "checkpoint_deserialization_during_source_review_authorized": False,
    }


def predecessor_contract() -> dict[str, Any]:
    return {
        "fresh_update0_initialization_not_v3_or_v4_continuation": True,
        "update0_state_sha256": UPDATE0_STATE_SHA256,
        "v3_predecessor": _v3_contract.predecessor_contract(),
        "v3_qualified_checkpoint_exists": False,
        "v4_qualified_checkpoint_exists": False,
        "v3_warm_start_authorized": False,
        "retry_resume_or_optimizer_reconstruction_authorized": False,
        "evidence": evidence_contract(),
    }


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v3_contract.science_contract())
    value["optimizer"]["maximum_updates"] = MAXIMUM_UPDATE
    value["schedule"] = {
        **value["schedule"],
        "checkpoint_prefix_sha256": {str(update): digest for update, digest in CHECKPOINT_SCHEDULE_PREFIX_SHA256.items()},
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "presentation_count": PRESENTATION_COUNT,
        "presentation_indices_sha256": SCHEDULE_PREFIX_INDICES_SHA256,
        "use_exact_prefix_updates": MAXIMUM_UPDATE,
    }
    return value


def science_delta() -> dict[str, Any]:
    base = _v3_contract.science_contract()
    return {
        "base_science_contract_sha256": canonical_json_sha256(base),
        "training_science_change_count": 1,
        "training_science_change": "complete_the_already_frozen_native_schedule_from_update_4000_to_update_8000",
        "contract_leaf_changes_encoding_that_one_training_change": [
            {"path": "optimizer.maximum_updates", "before": 4_000, "after": MAXIMUM_UPDATE},
            {"path": "schedule.use_exact_prefix_updates", "before": 4_000, "after": MAXIMUM_UPDATE},
            {"path": "schedule.presentation_count", "before": 64_000, "after": PRESENTATION_COUNT},
            {"path": "schedule.presentation_indices_sha256", "before": base["schedule"]["presentation_indices_sha256"], "after": SCHEDULE_PREFIX_INDICES_SHA256},
        ],
        "nonmutating_control_and_reporting_delta": {
            "checkpoint_updates_before": list(_v3_contract.CHECKPOINT_UPDATES),
            "checkpoint_updates_after": list(CHECKPOINT_UPDATES),
            "update_2000_omitted": True,
            "update_1000_and_4000_reproduction_floors": True,
            "update_6000_same_run_immutable_update_4000_pareto_gate": True,
            "update_8000_exact_terminal_physical_gate": True,
        },
        "architecture_loss_data_sampling_optimizer_or_initialization_changes": [],
        "other_training_science_changes": [],
    }


def control_contract() -> dict[str, Any]:
    return {
        "precedence": [
            "integrity_failure_is_terminal",
            "earliest_all_nine_physical_pass_qualifies",
            "fixed_checkpoint_control",
        ],
        "margin_statistics": {"count": MARGIN_COUNT, "P": "count(m>=0)", "S": "sum(max(0,-m))", "W": "min(m)"},
        "loss_policy": "finite_integrity_value_only_never_compared_for_continuation",
        "update_100": "finite_state_metrics_92_gradients_frozen_hash_trainable_movement_and_189_margins_required_then_qualify_or_continue",
        "update_400": "informational_then_qualify_or_continue",
        "update_1000_continue_if": {**V3_PROGRESS_BASELINES[1_000], "strict_improvement_required": False},
        "update_4000_continue_if": {**V3_PROGRESS_BASELINES[4_000], "strict_improvement_required": False},
        "update_6000_continue_if": "P_S_W_weakly_pareto_dominate_same_run_immutable_update_4000_sidecar_with_at_least_one_strict",
        "update_8000": "only_exact_all_nine_and_189_of_189_nonnegative_margins_qualifies_otherwise_stop",
        "observer_policy": "read_only_completed_mode_0444_sidecars_only_no_checkpoint_load_or_evaluation_rerun",
        "retry_resume_extension_threshold_relaxation_or_soft_promotion_authorized": False,
    }


def metric_sidecar_path(update: int) -> str:
    if type(update) is not int or update not in CHECKPOINT_UPDATES:
        raise ValueError("metric sidecar update is not a fixed V5 checkpoint")
    return f"{METRIC_SIDECAR_DIRECTORY}/update_{update}.metrics.json"


def validate_checkpoint_prefix(updates: Sequence[int]) -> tuple[int, ...]:
    if isinstance(updates, (str, bytes)):
        raise TypeError("checkpoint prefix is not an update sequence")
    result = tuple(updates)
    if any(type(update) is not int for update in result) or result != CHECKPOINT_UPDATES[: len(result)]:
        raise ValueError("checkpoint updates are not an exact V5 fixed prefix")
    return result


def expected_metric_sidecar_paths(updates: Sequence[int]) -> tuple[str, ...]:
    return tuple(metric_sidecar_path(update) for update in validate_checkpoint_prefix(updates))


def reporting_contract() -> dict[str, Any]:
    return {
        "fixed_checkpoint_updates": list(CHECKPOINT_UPDATES),
        "sidecar_paths": list(expected_metric_sidecar_paths(CHECKPOINT_UPDATES)),
        "one_inline_physical_evaluation_per_published_sidecar": True,
        "publication_order": "snapshot_then_inline_nonmutating_evaluation_then_atomic_exclusive_canonical_sidecar_then_predeclared_control_branch",
        "publication_mechanism": "same_directory_fsynced_temporary_regular_file_then_chmod_0444_then_atomic_exclusive_hard_link_then_directory_fsync",
        "sidecar_files_read_only_after_publication": True,
        "sidecar_is_only_live_checkpoint_readiness_marker": True,
        "checkpoint_file_existence_is_not_a_readiness_marker": True,
        "read_only_observers_must_not_load_checkpoints_or_rerun_evaluation": True,
        "numeric_continuation_rule": "V5_P_S_W_only_control_bound_inside_each_immutable_sidecar",
        "numeric_progress_cutoff_updates": [1_000, 4_000, 6_000],
        "terminal_physical_gate_update": 8_000,
        "integrity_nonfinite_or_frozen_mutation_remains_terminal": True,
        "final_metrics_collate_only_already_computed_sidecar_rows": True,
        "all_nine_gate_unchanged": True,
    }


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    inherited = _v3_contract.current_source_bindings(root)
    if any(inherited.get(path) != digest for path, digest in V3_SOURCE_SHA256.items()):
        raise PermissionError("protected Camera V3 source binding changed")
    result = dict(inherited)
    for path in SOURCE_PATHS:
        if path in result:
            continue
        source = root / path
        if source.is_symlink() or not source.is_file():
            raise PermissionError(f"reviewed V5 input is not one regular file: {path}")
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path in FIXED_EVIDENCE_SHA256:
            if digest != FIXED_EVIDENCE_SHA256[path] or len(raw) != FIXED_EVIDENCE_BYTE_COUNT[path]:
                raise PermissionError(f"preregistered V5 evidence changed: {path}")
            expected_content = FIXED_EVIDENCE_CONTENT_SHA256.get(path)
            if expected_content is not None:
                parsed = parse_canonical_json(raw, name=f"fixed V5 evidence {path}")
                if parsed.get("content_sha256") != expected_content:
                    raise PermissionError(f"preregistered V5 evidence content changed: {path}")
        result[path] = digest
    if set(result) != set(SOURCE_PATHS):
        raise PermissionError("protected Camera V5 source closure changed")
    return result


def validate_review(value: object, *, expected_sources: Mapping[str, str]) -> dict[str, Any]:
    fields = {
        "schema", "status", "implementation_author", "reviewer", "reviewed_sources",
        "predecessor", "science_contract", "science_delta", "evidence", "reporting_contract",
        "control_contract", "source_only", "findings", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("independent review fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    reviewer = value["reviewer"]
    if (
        value["schema"] != REVIEW_SCHEMA or value["status"] != "PASS"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str or not reviewer.startswith("/root/") or reviewer == IMPLEMENTATION_AUTHOR
        or value["reviewed_sources"] != dict(expected_sources)
        or value["predecessor"] != predecessor_contract()
        or value["science_contract"] != science_contract()
        or value["science_delta"] != science_delta()
        or value["evidence"] != evidence_contract()
        or value["reporting_contract"] != reporting_contract()
        or value["control_contract"] != control_contract()
        or value["source_only"] is not True or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not _v1.is_sha256(declared) or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("independent review did not pass exact V5 sources")
    return dict(value)


def validate_authorization(value: object, *, review_binding: Mapping[str, Any], reviewer: str) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_review", "predecessor", "raw",
        "camera", "experiment", "science_delta", "evidence", "reporting_contract",
        "control_contract", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("execution authorization fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != "authorized_one_exact_protected_camera_adaptation_v5_native_schedule_completion_attempt"
        or type(authorizer) is not str or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_review"] != dict(review_binding)
        or value["predecessor"] != predecessor_contract()
        or value["raw"] != expected_raw_authority()
        or value["camera"] != expected_camera_authority()
        or value["experiment"] != science_contract()
        or value["science_delta"] != science_delta()
        or value["evidence"] != evidence_contract()
        or value["reporting_contract"] != reporting_contract()
        or value["control_contract"] != control_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not _v1.is_sha256(declared) or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("execution authorization changed")
    return dict(value)


def learning_rates(update: int) -> tuple[float, float]:
    if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATE:
        raise ValueError("protected Camera V5 update must lie in [1,8000]")
    head = _v1.learning_rate(update)
    encoder = ENCODER_LR_SCALE * head
    if not math.isfinite(head) or not math.isfinite(encoder) or head <= 0.0 or encoder <= 0.0:
        raise ValueError("protected Camera V5 learning rate is invalid")
    return head, encoder


def parameter_partition(name: str) -> str:
    return _v3_contract.parameter_partition(name)


def evaluate_physical_scopes(scopes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return _v3_contract.evaluate_physical_scopes(scopes)


def _validate_update_4000_control_baseline(value: object) -> dict[str, Any]:
    fields = {
        "update", "path", "file_sha256", "content_sha256",
        "passed_margin_count", "total_shortfall", "worst_margin",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("update 6000 control baseline fields changed")
    if (
        value["update"] != 4_000 or value["path"] != metric_sidecar_path(4_000)
        or not _v1.is_sha256(value["file_sha256"]) or not _v1.is_sha256(value["content_sha256"])
        or type(value["passed_margin_count"]) is not int
        or not 0 <= value["passed_margin_count"] <= MARGIN_COUNT
        or any(type(value[name]) not in (int, float) or not math.isfinite(float(value[name])) for name in ("total_shortfall", "worst_margin"))
        or float(value["total_shortfall"]) < 0.0
    ):
        raise PermissionError("update 6000 control baseline changed")
    return {
        **dict(value),
        "total_shortfall": float(value["total_shortfall"]),
        "worst_margin": float(value["worst_margin"]),
    }


def control_decision_from_progress(
    *,
    update: int,
    passed_margin_count: int,
    total_shortfall: float,
    worst_margin: float,
    aggregate_complete_v4_loss: float,
    all_nine_physical_pass: bool,
    update_4000_control_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(update) is not int or update not in CHECKPOINT_UPDATES:
        raise ValueError("checkpoint control update is not fixed")
    if type(passed_margin_count) is not int or not 0 <= passed_margin_count <= MARGIN_COUNT:
        raise ValueError("passed physical margin count is invalid")
    if type(all_nine_physical_pass) is not bool:
        raise TypeError("all-nine physical decision is not Boolean")
    scalars = (total_shortfall, worst_margin, aggregate_complete_v4_loss)
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in scalars):
        raise FloatingPointError("checkpoint progress scalar became nonfinite")
    shortfall, worst, loss = map(float, scalars)
    if shortfall < 0.0:
        raise ValueError("checkpoint total shortfall became negative")
    baseline = None
    if update == 6_000:
        baseline = _validate_update_4000_control_baseline(update_4000_control_baseline)
    elif update_4000_control_baseline is not None:
        raise PermissionError("update 4000 control baseline appeared at the wrong checkpoint")
    statistics = {
        "margin_count": MARGIN_COUNT,
        "passed_margin_count": passed_margin_count,
        "total_shortfall": shortfall,
        "worst_margin": worst,
        "aggregate_complete_v4_loss": loss,
        "all_nine_physical_pass": all_nine_physical_pass,
        "update_4000_control_baseline": baseline,
    }
    if all_nine_physical_pass:
        action = CONTROL_ACTION_QUALIFY
        reason = "earliest fixed checkpoint passed every physical margin in all nine scopes"
        terminal_stage = "earliest_all_nine_physical_pass"
        next_update = None
    elif update in (100, 400):
        action = CONTROL_ACTION_CONTINUE
        reason = "fixed spotcheck continues because it did not yet qualify"
        terminal_stage = None
        next_update = CHECKPOINT_UPDATES[CHECKPOINT_UPDATES.index(update) + 1]
    elif update in V3_PROGRESS_BASELINES:
        floor = V3_PROGRESS_BASELINES[update]
        keep = (
            passed_margin_count >= floor["passed_margin_count"]
            and shortfall <= floor["total_shortfall"]
            and worst >= floor["worst_margin"]
        )
        action = CONTROL_ACTION_CONTINUE if keep else CONTROL_ACTION_STOP_PROGRESS
        reason = f"update {update} reproduced the frozen V3 P/S/W floor" if keep else f"update {update} missed the frozen V3 P/S/W reproduction floor"
        terminal_stage = None if keep else f"predeclared_numeric_progress_cutoff_at_update_{update}"
        next_update = CHECKPOINT_UPDATES[CHECKPOINT_UPDATES.index(update) + 1] if keep else None
    elif update == 6_000:
        assert baseline is not None
        weak = (
            passed_margin_count >= baseline["passed_margin_count"]
            and shortfall <= baseline["total_shortfall"]
            and worst >= baseline["worst_margin"]
        )
        strict = (
            passed_margin_count > baseline["passed_margin_count"]
            or shortfall < baseline["total_shortfall"]
            or worst > baseline["worst_margin"]
        )
        keep = weak and strict
        action = CONTROL_ACTION_CONTINUE if keep else CONTROL_ACTION_STOP_PROGRESS
        reason = "update 6000 improved on the bound same-run update 4000 P/S/W baseline" if keep else "update 6000 missed strict weak-Pareto improvement over the bound same-run update 4000 baseline"
        terminal_stage = None if keep else "predeclared_numeric_progress_cutoff_at_update_6000"
        next_update = 8_000 if keep else None
    else:
        action = CONTROL_ACTION_STOP_MAXIMUM
        reason = "update 8000 did not pass the unchanged all-nine physical gate"
        terminal_stage = "scientific_numeric_physical_gate_at_update_8000"
        next_update = None
    return {
        "schema": f"{SCHEMA_PREFIX}_checkpoint_control_decision_v1",
        "update": update,
        "statistics": statistics,
        "action": action,
        "reason": reason,
        "qualifies": action == CONTROL_ACTION_QUALIFY,
        "terminal_stage": terminal_stage,
        "next_checkpoint_update": next_update,
        "control_contract_sha256": canonical_json_sha256(control_contract()),
    }


def checkpoint_progress(metric: Mapping[str, Any]) -> dict[str, Any]:
    progress = _v3_contract.checkpoint_progress(metric)
    progress["update_4000_control_baseline"] = metric.get("update_4000_control_baseline")
    return progress


def checkpoint_control_decision(metric: Mapping[str, Any]) -> dict[str, Any]:
    progress = checkpoint_progress(metric)
    if progress["update"] == 100:
        before, after = metric.get("state_sha256_before"), metric.get("state_sha256_after")
        frozen = metric.get("frozen_state_sha256_before_and_after")
        if (
            not _v1.is_sha256(before) or after != before or before == UPDATE0_STATE_SHA256
            or not _v1.is_sha256(frozen) or metric.get("state_mutation_count") != 0
        ):
            raise PermissionError("update 100 integrity or trainable-state movement failed")
    return control_decision_from_progress(**progress)


def validate_metric_sidecar(
    value: object,
    *,
    update: int | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    metric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "update", "checkpoint", "metric", "inline_evaluation_count",
        "state_mutation_count", "publication", "continuation", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("checkpoint metric sidecar fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    observed_update, observed_metric = value["update"], value["metric"]
    if (
        type(observed_update) is not int or observed_update not in CHECKPOINT_UPDATES
        or value["schema"] != METRIC_SIDECAR_SCHEMA
        or value["status"] != "published_after_inline_nonmutating_physical_evaluation_before_control_branch"
        or type(value["checkpoint"]) is not dict or type(observed_metric) is not dict
        or observed_metric.get("update") != observed_update or observed_metric.get("role") != "checkpoint_selection"
        or observed_metric.get("state_mutation_count") != 0
        or value["inline_evaluation_count"] != 1 or value["state_mutation_count"] != 0
        or value["publication"] != reporting_contract()["publication_order"]
        or value["continuation"] != checkpoint_control_decision(observed_metric)
        or value["authority"] != {
            "read_only_observation_authorized": True,
            "observer_evaluation_rerun_authorized": False,
            "only_predeclared_metric_control_authorized": True,
            "g2_navigation_or_heldout_use_authorized": False,
        }
        or (update is not None and observed_update != update)
        or (checkpoint is not None and value["checkpoint"] != dict(checkpoint))
        or (metric is not None and observed_metric != dict(metric))
        or not _v1.is_sha256(declared) or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("checkpoint metric sidecar changed")
    return dict(value)


def expected_raw_authority() -> dict[str, Any]:
    return _v3_contract.expected_raw_authority()


def expected_camera_authority() -> dict[str, Any]:
    value = _v3_contract.expected_camera_authority()
    checkpoint = value.get("checkpoint")
    if type(checkpoint) is not dict or checkpoint.get("content_sha256") != N320_CHECKPOINT_CONTENT_SHA256:
        raise PermissionError("N320 checkpoint content binding changed")
    return value


def __getattr__(name: str) -> Any:
    """Delegate unchanged V3/V2/V1 lifecycle constants to the exact frozen contract."""
    return getattr(_v3_contract, name)


__all__ = [name for name in globals() if name.isupper()] + [
    "artifact_binding", "canonical_json_bytes", "canonical_json_sha256", "checkpoint_control_decision",
    "checkpoint_progress", "control_contract", "control_decision_from_progress", "current_source_bindings",
    "evidence_contract", "evaluate_physical_scopes", "expected_camera_authority", "expected_metric_sidecar_paths",
    "expected_raw_authority", "learning_rates", "metric_sidecar_path", "parameter_partition",
    "parse_canonical_json", "predecessor_contract", "reporting_contract", "science_contract", "science_delta",
    "validate_authorization", "validate_binding", "validate_checkpoint_prefix", "validate_metric_sidecar",
    "validate_review", "with_content_sha256",
]
