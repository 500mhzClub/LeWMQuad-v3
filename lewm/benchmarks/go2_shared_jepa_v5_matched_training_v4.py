"""Warn-only runtime successor over the exact matched-training V3 stack."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping
import warnings


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/matched_v4_warnonly_design"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_matched_training_v4"
CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v4.py"
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_matched_training_v4.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_matched_training_v4.py"
V3_CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v3.py"
V3_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_matched_training_v3.py"
V3_TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_matched_training_v3.py"
V3_SOURCE_SHA256 = {
    V3_CONTRACT_RELATIVE_PATH: "0a62268f096632b1fe4d7ddd14411e640297d60a81b018ffbe5025f2d711bd7f",
    V3_RUNNER_RELATIVE_PATH: "55217518ff38eb6ddef4973e9dccfcbb5510d400ede877c16a2a69fabdd1fafc",
    V3_TEST_RELATIVE_PATH: "28678f25e79eb44c22be0b889774c211a9e21e66d985bcbd5bc12b8e3c0fd8e2",
}


def _load_exact_v3() -> ModuleType:
    path = ROOT / V3_CONTRACT_RELATIVE_PATH
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != V3_SOURCE_SHA256[V3_CONTRACT_RELATIVE_PATH]:
        raise PermissionError("the frozen V3 contract changed")
    spec = importlib.util.spec_from_file_location("_lewm_matched_training_v4_readonly_v3", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the frozen V3 contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v3 = _load_exact_v3()
_V3_SCIENCE_CONTRACT = _v3.science_contract()
REVIEW_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_matched_training_v4_independent_review_2026-07-15.json"
AUTHORIZATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_matched_training_v4_execution_authorization_2026-07-15.json"
OUTPUT_ROOT_RELATIVE_PATH = ".generated/go2_shared_observable_camera_ray_jepa_v5/matched_training_v4"
PREDECESSOR_ROOT_RELATIVE_PATH = _v3.OUTPUT_ROOT_RELATIVE_PATH
TERMINAL_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_matched_training_v3_terminal_failure_audit_2026-07-15.json"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_independent_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
SCHEDULE_SCHEMA = f"{SCHEMA_PREFIX}_schedule_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_training_snapshot_v1"
SELECTION_SCHEMA = f"{SCHEMA_PREFIX}_selection_v1"
CALIBRATION_SCHEMA = f"{SCHEMA_PREFIX}_calibration_v1"
PRE_G2_CHECKPOINT_SCHEMA = f"{SCHEMA_PREFIX}_pre_g2_checkpoint_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
VERIFICATION_SCHEMA = f"{SCHEMA_PREFIX}_verification_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
INITIALIZATION_SCHEMA = f"{SCHEMA_PREFIX}_initialization_v1"
SOURCE_PATHS = (CONTRACT_RELATIVE_PATH, RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH, TERMINAL_AUDIT_RELATIVE_PATH, *_v3.SOURCE_PATHS)
CONTRACT_PATCH_WHITELIST = _v3.CONTRACT_PATCH_WHITELIST
RUNNER_PATCH_WHITELIST = _v3.RUNNER_PATCH_WHITELIST
INSTALLATION_SENTINEL = "_lewm_matched_training_v4_installed"
MAXIMUM_ATTEMPTS, RETRY_AUTHORIZED, AUTOMATIC_V5_AUTHORIZED = 1, False, False

_binding = _v3._binding
PREDECESSOR_ARTIFACT_BINDINGS = {
    "reservation.json": _binding("reservation.json", "a57c65c20f57aca25573e0f0db655b7189541cb46229badc7475b83b0446b0b0", "a75582362ab47df13cbfdb9d27086f5f904d552bb23664d8ef9be580a2294368", 9_067),
    "initialization.json": _binding("initialization.json", "8880dcbda7da11e455069c32b73d623704048ae98ffa6bc26d22413140afac22", "3ada3faad51ca4c6aa5357d2238d8e684a53b8cceab6de6eff59984c7fb727de", 1_724),
    "schedule.json": _binding("schedule.json", "a94f8eecea35320fdf96035da62e42630b6c61c245164c0dfab26337a27d8782", "98d2da3f0c5639f842cc5e775951a16fda771a065cfa80d98cdb7a1dcb4d9c4d", 607_373),
    "failed.json": _binding("failed.json", "227e9932eca4c93892a9a6378c56fff5a8993c38a6c0598f7d0b6819533b421a", "9c0bbe2d8a00d77720dc4e3cfc26732b0f271129cdbc06f03b0d4e2a663c366c", 967),
}
V3_REVIEW_BINDING = _binding(_v3.REVIEW_RELATIVE_PATH, "d996bb0c290250c95037dcd12f036855da06d5136af38373f7a647490f812cc3", "0749cee5897a8abb8f3f61e20105af2fd112cb13a817217126ddef0696f99d02", 6_410)
V3_AUTHORIZATION_BINDING = _binding(_v3.AUTHORIZATION_RELATIVE_PATH, "0e2149d5b22bcf646f8a9f638d7938bb728b2a4df7e8a8a16ee80054d06da61b", "18b5135e08313c21981269af841b378c7f145e19d71d1f69ee12e730bd086457", 6_199)
V3_TERMINAL_AUDIT_BINDING = _binding(TERMINAL_AUDIT_RELATIVE_PATH, "2f94d6ddaf076bc011eaac46408261aea3b8ac030386c9d2185463fe87a08e4a", "b93146f00c79a6b2d151a07fb33696c673a1d45677ee6b948e20acadef9c9899", 12_883)
PREDECESSOR_ATTEMPT_IDENTITY = "bab49ded947a367100ead194cf1c29168c7cc7a371dc915f3a60b6e2f229126c"
GRID_WARNING = "grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation."
_CONTEXT_PREFIX, _CONTEXT_SUFFIX = " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:", ".)"
RUNTIME_DETERMINISM = {
    "requested": "strict_deterministic_algorithms",
    "effective": "strict_where_supported_warn_only_exact_grid_sampler_2d_backward_cuda",
    "known_nondeterministic_kernel": "grid_sampler_2d_backward_cuda",
    "bitwise_repeatability_guaranteed": False,
    "unexpected_warning_policy": "fail_closed",
}


def __getattr__(name: str) -> Any:
    return getattr(_v3, name)


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_V3_SCIENCE_CONTRACT)
    value["candidate"]["schema"] = PRE_G2_CHECKPOINT_SCHEMA
    return value


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    _v3.current_source_bindings(root)
    values = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}
    if any(values.get(path) != digest for path, digest in V3_SOURCE_SHA256.items()) or values.get(TERMINAL_AUDIT_RELATIVE_PATH) != V3_TERMINAL_AUDIT_BINDING["file_sha256"]:
        raise PermissionError("the frozen V3 source or terminal audit changed")
    return values


def validate_predecessor(read_regular: Callable[..., bytes], *, root: Path = ROOT) -> dict[str, Any]:
    predecessor_root = root / PREDECESSOR_ROOT_RELATIVE_PATH
    entries = list(predecessor_root.iterdir()) if predecessor_root.is_dir() and not predecessor_root.is_symlink() else []
    if {item.name for item in entries} != set(PREDECESSOR_ARTIFACT_BINDINGS) or any(item.is_symlink() or not item.is_file() for item in entries):
        raise PermissionError("the V3 terminal inventory changed")
    artifacts = {name: _v3._read_bound_json(predecessor_root, binding, read_regular, name=f"V3 {name}") for name, binding in PREDECESSOR_ARTIFACT_BINDINGS.items()}
    reservation, failed = artifacts["reservation.json"], artifacts["failed.json"]
    expected_error = "grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation."
    if reservation.get("schema") != _v3.RESERVATION_SCHEMA or reservation.get("attempt_identity") != PREDECESSOR_ATTEMPT_IDENTITY or reservation.get("status") != "reserved_before_torch_camera_raw_or_rgb" or reservation.get("retry_authorized") is not False:
        raise PermissionError("the V3 reservation changed")
    if failed.get("schema") != _v3.FAILURE_SCHEMA or (failed.get("status"), failed.get("stage"), failed.get("retry_authorized"), failed.get("g2_attempted"), failed.get("heldout_open_count")) != ("failed_infrastructure", "promoted_training", False, False, 0) or failed.get("error") != {"type": "RuntimeError", "message": expected_error}:
        raise PermissionError("the V3 terminal failure changed")
    review = _v3._read_bound_json(root, V3_REVIEW_BINDING, read_regular, name="V3 review")
    authorization = _v3._read_bound_json(root, V3_AUTHORIZATION_BINDING, read_regular, name="V3 authorization")
    audit = _v3._read_bound_json(root, V3_TERMINAL_AUDIT_BINDING, read_regular, name="V3 terminal audit")
    boundary = audit.get("training_boundary", {})
    if review.get("schema") != _v3.REVIEW_SCHEMA or review.get("status") != "PASS" or authorization.get("schema") != _v3.AUTHORIZATION_SCHEMA or authorization.get("status") != "authorized_one_exact_development_attempt" or audit.get("verdict") != "PASS_CONFIRMED_FIRST_B4_FORWARD_LOSS_THEN_ROCM_DETERMINISM_INFRASTRUCTURE_FAILURE_ZERO_LEARNED_UPDATE" or audit.get("terminal_inventory", {}).get("exact_entry_count") != 4 or audit.get("authority", {}).get("automatic_successor_authorized") is not False or {name: boundary.get(name) for name in ("model_forward_count", "joint_loss_combination_count", "backward_invocation_count", "backward_completion_count", "optimizer_step_count", "ema_update_count", "persistent_learned_state_mutation_count")} != {"model_forward_count": 1, "joint_loss_combination_count": 1, "backward_invocation_count": 1, "backward_completion_count": 0, "optimizer_step_count": 0, "ema_update_count": 0, "persistent_learned_state_mutation_count": 0}:
        raise PermissionError("the independently audited V3 failure changed")
    return {"artifacts": artifacts, "v3_review": review, "v3_authorization": authorization, "terminal_audit": audit}


def normalize_lifecycle_artifact_to_v3(value: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    schemas = {"initialization": (INITIALIZATION_SCHEMA, _v3.INITIALIZATION_SCHEMA), "schedule": (SCHEDULE_SCHEMA, _v3.SCHEDULE_SCHEMA)}
    if kind not in schemas or type(value) is not dict:
        raise TypeError("only plain V4 initialization or schedule artifacts normalize")
    core, declared = dict(value), value.get("content_sha256")
    core.pop("content_sha256", None)
    if value.get("schema") != schemas[kind][0] or not _v3.is_sha256(declared) or _v3.canonical_json_sha256(core) != declared:
        raise PermissionError(f"V4 {kind} artifact changed")
    core["schema"] = schemas[kind][1]
    return _v3.with_content_sha256(core)


def require_v3_initialization_and_schedule(output_root: Path, predecessor: Mapping[str, Any], read_regular: Callable[..., bytes]) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts, current = predecessor.get("artifacts"), {}
    if type(artifacts) is not dict:
        raise PermissionError("validated V3 predecessor state is absent")
    for kind, filename in (("initialization", "initialization.json"), ("schedule", "schedule.json")):
        value = _v3.parse_canonical_json(read_regular(output_root / filename), name=f"V4 {kind}")
        if normalize_lifecycle_artifact_to_v3(value, kind=kind) != artifacts[filename]:
            raise PermissionError(f"V4 {kind} is not the exact V3 identity")
        current[kind] = value
    return current["initialization"], current["schedule"]


def normalize_determinism_warning(message: object, category: type[Warning]) -> dict[str, Any]:
    if category is not UserWarning:
        raise RuntimeError("unexpected training warning category")
    raw, source_line = str(message), None
    if raw != GRID_WARNING:
        if not raw.startswith(GRID_WARNING + _CONTEXT_PREFIX) or not raw.endswith(_CONTEXT_SUFFIX):
            raise RuntimeError(f"unexpected training warning: {raw}")
        digits = raw[len(GRID_WARNING + _CONTEXT_PREFIX) : -len(_CONTEXT_SUFFIX)]
        if not digits or digits[0] == "0" or not digits.isascii() or not digits.isdigit():
            raise RuntimeError(f"unexpected training warning: {raw}")
        source_line = int(digits)
    return {"normalized": GRID_WARNING, "context_source_line": source_line}


class CompactDeterminismWarnings:
    def __init__(self) -> None:
        self.warning_count = self.context_trailer_count = 0

    def __call__(self, message: object, category: type[Warning], *args: Any, **kwargs: Any) -> None:
        normalized = normalize_determinism_warning(message, category)
        self.warning_count += 1
        self.context_trailer_count += int(normalized["context_source_line"] is not None)

    def receipt(self) -> dict[str, Any]:
        histogram = [{"message": GRID_WARNING, "count": self.warning_count}]
        return {"warning_count": self.warning_count, "context_trailer_count": self.context_trailer_count, "kernel_counts": {"grid_sampler_2d_backward_cuda": self.warning_count}, "normalized_histogram": histogram, "normalized_histogram_sha256": _v3.canonical_json_sha256(histogram)}


def run_with_expected_grid_warning(operation: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    collector = CompactDeterminismWarnings()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        original_showwarning = warnings.showwarning
        warnings.showwarning = collector
        try:
            result = operation()
        finally:
            warnings.showwarning = original_showwarning
    receipt = collector.receipt()
    if receipt["warning_count"] <= 0:
        raise RuntimeError("completed training arm emitted no expected grid-sampler warning")
    return result, receipt


def install_successor(base_runner: object, baseline_namespace: Mapping[str, object], v3_contract: object) -> ModuleType:
    current_source_bindings(ROOT)
    v3_module = _v3._v2._exact_module(v3_contract, V3_CONTRACT_RELATIVE_PATH, V3_SOURCE_SHA256[V3_CONTRACT_RELATIVE_PATH])
    base = _v3._v2._exact_module(base_runner, _v3._v2.V1_RUNNER_RELATIVE_PATH, _v3._v2.V1_SOURCE_SHA256[_v3._v2.V1_RUNNER_RELATIVE_PATH])
    if INSTALLATION_SENTINEL in vars(base):
        raise RuntimeError("the V4 successor was already installed")
    if _v3.INSTALLATION_SENTINEL not in vars(base) or set(vars(base)) != set(baseline_namespace) or any(vars(base)[name] is not value for name, value in baseline_namespace.items()):
        raise PermissionError("the installed V3 runner namespace drifted")
    base_contract = _v3._v2._exact_module(base.contract, _v3._v2.V1_CONTRACT_RELATIVE_PATH, _v3._v2.V1_SOURCE_SHA256[_v3._v2.V1_CONTRACT_RELATIVE_PATH])
    if any(getattr(base_contract, name, object()) != getattr(v3_module, name) for name in CONTRACT_PATCH_WHITELIST):
        raise PermissionError("the installed V3 lifecycle contract drifted")
    original_raw, original_trainer = base.RawInputs, base.Trainer
    if original_raw.__name__ != "RawInputsV3" or original_trainer.__name__ != "TrainerV3" or [item.__name__ for item in original_trainer.__mro__[:3]] != ["TrainerV3", "TrainerV2", "Trainer"]:
        raise PermissionError("the installed V3 overlay classes drifted")
    contract_snapshot, runner_snapshot = dict(vars(base_contract)), dict(vars(base))
    original_run_parent, original_run_internal_verifier = base.run_parent, base.run_internal_verifier
    predecessor_state: dict[str, Any] = {}
    for name in CONTRACT_PATCH_WHITELIST:
        setattr(base_contract, name, globals()[name])
    frozen_v1_train_arm = original_trainer.__mro__[2].train_arm

    class TrainerV4(original_trainer):
        def device(self) -> tuple[Any, dict[str, Any]]:
            device, resource = super().device()
            torch = self.r.torch
            torch.use_deterministic_algorithms(True, warn_only=True)
            if not torch.are_deterministic_algorithms_enabled() or not torch.is_deterministic_algorithms_warn_only_enabled():
                raise RuntimeError("V4 warn-only deterministic runtime was not enabled")
            record = {**RUNTIME_DETERMINISM, "torch_deterministic_algorithms": True, "warn_only": True, "arms": {}}
            resource["determinism"] = self._v4_determinism_record = record
            return device, resource

        def train_arm(self, **kwargs: Any) -> dict[int, dict[str, Any]]:
            predecessor = predecessor_state.get("value")
            if type(predecessor) is not dict:
                raise PermissionError("V3 predecessor was not validated before training")
            initialization, schedule = require_v3_initialization_and_schedule(self.output_root, predecessor, base._read_regular)
            if type(kwargs.get("initial_state")) is not dict or self.r.model_module.tensor_state_dict_sha256(kwargs["initial_state"]) != initialization["complete_state_sha256"] or list(kwargs.get("schedule", ())) != schedule["presentation_indices"] or list(kwargs.get("vocabulary", ())) != initialization["primitive_vocabulary"] or getattr(kwargs.get("commanded_table"), "tolist", lambda: None)() != initialization["commanded_delta_table"]:
                raise PermissionError("in-memory V4 initialization or schedule changed")
            arm = str(kwargs.get("arm"))
            record = getattr(self, "_v4_determinism_record", None)
            if type(record) is not dict or arm in record["arms"]:
                raise RuntimeError("V4 determinism receipt lifecycle changed")
            result, record["arms"][arm] = run_with_expected_grid_warning(lambda: frozen_v1_train_arm(self, **kwargs))
            return result

    TrainerV4.__name__ = "TrainerV4"

    def run_parent(*, review_file_sha256: str, authorization_file_sha256: str) -> int:
        predecessor_state["value"] = validate_predecessor(base._read_regular, root=ROOT)
        return original_run_parent(review_file_sha256=review_file_sha256, authorization_file_sha256=authorization_file_sha256)

    def run_internal_verifier() -> int:
        predecessor_state["value"] = validate_predecessor(base._read_regular, root=ROOT)
        return original_run_internal_verifier()

    base.Trainer, base.run_parent, base.run_internal_verifier = TrainerV4, run_parent, run_internal_verifier
    setattr(base, INSTALLATION_SENTINEL, True)
    if set(vars(base_contract)) != set(contract_snapshot) or any(vars(base_contract)[name] is not value for name, value in contract_snapshot.items() if name not in CONTRACT_PATCH_WHITELIST):
        raise RuntimeError("V4 changed an undeclared private-contract attribute")
    if set(vars(base)) != set(runner_snapshot) | {INSTALLATION_SENTINEL} or any(vars(base)[name] is not value for name, value in runner_snapshot.items() if name not in RUNNER_PATCH_WHITELIST):
        raise RuntimeError("V4 changed an undeclared private-runner attribute")
    return base
