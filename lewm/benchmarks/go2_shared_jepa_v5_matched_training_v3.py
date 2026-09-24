"""One narrow scalar-tensorization successor over exact matched-training V2."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/matched_v3_scalar_successor_design"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_matched_training_v3"
CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v3.py"
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_matched_training_v3.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_matched_training_v3.py"
V2_CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v2.py"
V2_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_matched_training_v2.py"
V2_TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_matched_training_v2.py"
V2_SOURCE_SHA256 = {
    V2_CONTRACT_RELATIVE_PATH: "e2e8fcfe3294909dd94eae67fd5c69ea87fc7f1a70cb650a925ff1074c8ad772",
    V2_RUNNER_RELATIVE_PATH: "68cde7d4a961786e62ccdd471d341b2b5299be75747b0e6e060a1a89b425049e",
    V2_TEST_RELATIVE_PATH: "c105250b826fcdecc655408cf6d9f11e5da9a22872775d83e58a35e50be23b8b",
}


def _load_exact_v2_contract() -> ModuleType:
    path = ROOT / V2_CONTRACT_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise PermissionError("the frozen V2 contract is not a regular file")
    if hashlib.sha256(path.read_bytes()).hexdigest() != V2_SOURCE_SHA256[V2_CONTRACT_RELATIVE_PATH]:
        raise PermissionError("the frozen V2 contract changed")
    spec = importlib.util.spec_from_file_location(
        "_lewm_go2_shared_jepa_v5_matched_training_v3_readonly_v2_contract", path
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the frozen V2 contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v2 = _load_exact_v2_contract()
_V2_SCIENCE_CONTRACT = _v2.science_contract()
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_matched_training_v3_independent_review_2026-07-15.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_matched_training_v3_execution_authorization_2026-07-15.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/matched_training_v3"
)
PREDECESSOR_ROOT_RELATIVE_PATH = _v2.OUTPUT_ROOT_RELATIVE_PATH
TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_matched_training_v2_terminal_failure_audit_2026-07-15.json"
)

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
SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    TERMINAL_AUDIT_RELATIVE_PATH,
    *_v2.SOURCE_PATHS,
)
CONTRACT_PATCH_WHITELIST = _v2.CONTRACT_PATCH_WHITELIST
RUNNER_PATCH_WHITELIST = _v2.RUNNER_PATCH_WHITELIST
INSTALLATION_SENTINEL = "_lewm_matched_training_v3_installed"
MAXIMUM_ATTEMPTS = 1
RETRY_AUTHORIZED = False
AUTOMATIC_V4_AUTHORIZED = False


def _binding(path: str, file_sha: str, content_sha: str, size: int) -> dict[str, Any]:
    return _v2._binding(path, file_sha, content_sha, size)


PREDECESSOR_ARTIFACT_BINDINGS = {
    "reservation.json": _binding("reservation.json", "e9b278648e451905c00c27ca2d071e1d323f02de8d33d6b607929443e5a0dee8", "d1a9c514bce52eb433a1d7a88ef7b9adb77130441dae20d23b44ca91bf40a7ff", 8_533),
    "initialization.json": _binding("initialization.json", "db4f4f226bef30b0b1d9f60cd72bd12975744e48aac60aacfc0dd16d5d1dab26", "17925e26f795b394a6d631defad3de4d0c9c60f7587877604a934d6cb1ac4dee", 1_724),
    "schedule.json": _binding("schedule.json", "184603f4039700dbff77732abc365403acc9c034df4ba0389f20b1f816577868", "131680c6438bd631e6049558af5d987a0e8b7215d4cf412b2360b5154acb0dba", 607_373),
    "failed.json": _binding("failed.json", "da125d297165c04e1837499aac95495ed617a53a9720392ec8f74a7aafecf9a9", "c57e8b221ebbeb31dfdc6aad05f227f6608436032207fbf1556e86e329e4b8c3", 588),
}
V2_REVIEW_BINDING = _binding(_v2.REVIEW_RELATIVE_PATH, "3325e84a4134f4599be2d18f10c3459bf76a7eef311bf5bc4f9ecf389f4afe33", "5aa4c72cfea7d31da9abe61057447cb897177d4f6b49802564a4269f80b46571", 5_857)
V2_AUTHORIZATION_BINDING = _binding(_v2.AUTHORIZATION_RELATIVE_PATH, "1bddbedf818c31fb88b7ec576539d48e8c4eb72e05974f85236e39f1d8195fe4", "98c1c1129b2e3875a6eec420f42a88230594a0c4d51c23b9c96c10ad64c3c8b8", 6_199)
V2_TERMINAL_AUDIT_BINDING = _binding(TERMINAL_AUDIT_RELATIVE_PATH, "0fa0708013202a33df16ec1212ebbcc8c6add9980b23f89f4e2267152f54c871", "00dc98ce651132d51861b7cff96132e8177c18845b0032b159e9e9236771aaf4", 18_632)
PREDECESSOR_ATTEMPT_IDENTITY = "fa2aad4f675c1eabdeab2b64861a3a8157b156de733b9fc8cae6aa5c27fcfab6"


def __getattr__(name: str) -> Any:
    return getattr(_v2, name)


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_V2_SCIENCE_CONTRACT)
    value["candidate"]["schema"] = PRE_G2_CHECKPOINT_SCHEMA
    return value


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    _v2.current_source_bindings(root)
    bindings = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}
    if any(bindings.get(path) != digest for path, digest in V2_SOURCE_SHA256.items()):
        raise PermissionError("the three frozen V2 implementation files changed")
    return bindings


def predecessor_contract() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_predecessor_contract_v1",
        "root": PREDECESSOR_ROOT_RELATIVE_PATH,
        "terminal_artifacts": copy.deepcopy(PREDECESSOR_ARTIFACT_BINDINGS),
        "v1_sources": dict(_v2.V1_SOURCE_SHA256),
        "v1_review": dict(_v2.V1_REVIEW_BINDING),
        "v1_authorization": dict(_v2.V1_AUTHORIZATION_BINDING),
        "v2_sources": dict(V2_SOURCE_SHA256),
        "v2_review": dict(V2_REVIEW_BINDING),
        "v2_authorization": dict(V2_AUTHORIZATION_BINDING),
        "v2_terminal_audit": dict(V2_TERMINAL_AUDIT_BINDING),
        "attempt_identity": PREDECESSOR_ATTEMPT_IDENTITY,
        "failure": {
            "status": "failed_infrastructure",
            "stage": "promoted_training",
            "error": {"type": "TypeError", "message": "expected np.ndarray (got numpy.float32)"},
            "g2_attempted": False,
            "heldout_open_count": 0,
            "retry_authorized": False,
        },
        "v2_rerun_authorized": False,
        "v3_maximum_attempts": MAXIMUM_ATTEMPTS,
        "v3_retry_authorized": RETRY_AUTHORIZED,
        "automatic_v4_authorized": AUTOMATIC_V4_AUTHORIZED,
    }


def _read_bound_json(root: Path, binding: Mapping[str, Any], read_regular: Callable[..., bytes], *, name: str) -> dict[str, Any]:
    return _v2._read_bound_json(root, binding, read_regular, name=name)


def validate_predecessor(read_regular: Callable[..., bytes], *, root: Path = ROOT) -> dict[str, Any]:
    predecessor_root = root / PREDECESSOR_ROOT_RELATIVE_PATH
    if predecessor_root.is_symlink() or not predecessor_root.is_dir():
        raise PermissionError("the V2 terminal root changed")
    entries = list(predecessor_root.iterdir())
    if {item.name for item in entries} != set(PREDECESSOR_ARTIFACT_BINDINGS) or any(
        item.is_symlink() or not item.is_file() for item in entries
    ):
        raise PermissionError("the V2 terminal inventory changed")
    artifacts = {
        name: _read_bound_json(predecessor_root, binding, read_regular, name=f"V2 {name}")
        for name, binding in PREDECESSOR_ARTIFACT_BINDINGS.items()
    }
    reservation, failed = artifacts["reservation.json"], artifacts["failed.json"]
    if (
        reservation.get("schema") != _v2.RESERVATION_SCHEMA
        or reservation.get("attempt_identity") != PREDECESSOR_ATTEMPT_IDENTITY
        or reservation.get("status") != "reserved_before_torch_camera_raw_or_rgb"
        or reservation.get("retry_authorized") is not False
        or {key: failed.get(key) for key in predecessor_contract()["failure"]}
        != predecessor_contract()["failure"]
        or failed.get("schema") != _v2.FAILURE_SCHEMA
    ):
        raise PermissionError("the V2 terminal attempt changed")
    review = _read_bound_json(root, V2_REVIEW_BINDING, read_regular, name="V2 review")
    authorization = _read_bound_json(root, V2_AUTHORIZATION_BINDING, read_regular, name="V2 authorization")
    audit = _read_bound_json(root, V2_TERMINAL_AUDIT_BINDING, read_regular, name="V2 terminal audit")
    counts = audit.get("zero_training_proof", {}).get("counts", {})
    if (
        review.get("schema") != _v2.REVIEW_SCHEMA
        or review.get("status") != "PASS"
        or authorization.get("schema") != _v2.AUTHORIZATION_SCHEMA
        or authorization.get("status") != "authorized_one_exact_development_attempt"
        or audit.get("schema") != "lewm_go2_shared_jepa_v5_matched_training_v2_terminal_failure_audit_v1"
        or audit.get("verdict") != "PASS_CONFIRMED_ZERO_FORWARD_INFRASTRUCTURE_SCALAR_TENSORIZATION_FAILURE"
        or audit.get("terminal_inventory", {}).get("exact_entry_count") != 4
        or audit.get("authority", {}).get("automatic_successor_authorized") is not False
        or any(counts.get(name) != 0 for name in ("model_forward_count", "loss_combination_count", "backward_count", "optimizer_step_count", "ema_update_count"))
    ):
        raise PermissionError("the independently audited V2 failure changed")
    return {"artifacts": artifacts, "v2_review": review, "v2_authorization": authorization, "terminal_audit": audit}


def normalize_lifecycle_artifact_to_v2(value: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    schemas = {
        "initialization": (INITIALIZATION_SCHEMA, _v2.INITIALIZATION_SCHEMA),
        "schedule": (SCHEDULE_SCHEMA, _v2.SCHEDULE_SCHEMA),
    }
    if kind not in schemas or type(value) is not dict:
        raise TypeError("only plain V3 initialization or schedule artifacts normalize")
    current_schema, predecessor_schema = schemas[kind]
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if value.get("schema") != current_schema or not _v2.is_sha256(declared) or _v2.canonical_json_sha256(core) != declared:
        raise PermissionError(f"V3 {kind} artifact changed")
    core["schema"] = predecessor_schema
    return _v2.with_content_sha256(core)


def require_v2_initialization_and_schedule(output_root: Path, predecessor: Mapping[str, Any], read_regular: Callable[..., bytes]) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = predecessor.get("artifacts")
    if type(artifacts) is not dict:
        raise PermissionError("validated V2 predecessor state is absent")
    current = {}
    for kind, filename in (("initialization", "initialization.json"), ("schedule", "schedule.json")):
        value = _v2.parse_canonical_json(read_regular(output_root / filename), name=f"V3 {kind}")
        if normalize_lifecycle_artifact_to_v2(value, kind=kind) != artifacts[filename]:
            raise PermissionError(f"V3 {kind} is not the exact V2 identity")
        current[kind] = value
    return current["initialization"], current["schedule"]


def install_successor(base_runner: object, baseline_namespace: Mapping[str, object], v2_contract: object) -> ModuleType:
    if any(
        hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest
        for path, digest in {**_v2.V1_SOURCE_SHA256, **V2_SOURCE_SHA256}.items()
    ):
        raise PermissionError("the frozen V1/V2 implementation closure changed")
    v2_module = _v2._exact_module(v2_contract, V2_CONTRACT_RELATIVE_PATH, V2_SOURCE_SHA256[V2_CONTRACT_RELATIVE_PATH])
    base = _v2._exact_module(base_runner, _v2.V1_RUNNER_RELATIVE_PATH, _v2.V1_SOURCE_SHA256[_v2.V1_RUNNER_RELATIVE_PATH])
    if INSTALLATION_SENTINEL in vars(base):
        raise RuntimeError("the V3 successor was already installed")
    if _v2.INSTALLATION_SENTINEL not in vars(base) or set(vars(base)) != set(baseline_namespace) or any(
        vars(base)[name] is not value for name, value in baseline_namespace.items()
    ):
        raise PermissionError("the installed V2 runner namespace drifted")
    base_contract = _v2._exact_module(getattr(base, "contract", None), _v2.V1_CONTRACT_RELATIVE_PATH, _v2.V1_SOURCE_SHA256[_v2.V1_CONTRACT_RELATIVE_PATH])
    if any(getattr(base_contract, name, object()) != getattr(v2_module, name) for name in CONTRACT_PATCH_WHITELIST):
        raise PermissionError("the installed V2 lifecycle contract drifted")
    original_raw, original_trainer = base.RawInputs, base.Trainer
    if (
        original_raw.__name__ != "RawInputsV2"
        or original_trainer.__name__ != "TrainerV2"
        or original_raw.__mro__[1].__name__ != "RawInputs"
        or original_trainer.__mro__[1].__name__ != "Trainer"
    ):
        raise PermissionError("the installed V2 overlay classes drifted")
    contract_snapshot, runner_snapshot = dict(vars(base_contract)), dict(vars(base))
    original_run_parent, original_run_internal_verifier = base.run_parent, base.run_internal_verifier
    predecessor_state: dict[str, Any] = {}
    for name in CONTRACT_PATCH_WHITELIST:
        setattr(base_contract, name, globals()[name])

    class RawInputsV3(original_raw):
        def _row_array(self, endpoint: Mapping[str, Any], shard: Mapping[str, Any], filename: str, *, arm: str, stage: str) -> Any:
            try:
                return super()._row_array(endpoint, shard, filename, arm=arm, stage=stage)
            except TypeError as error:
                if filename != "ground_plane_z_body_m.f4" or str(error) != "expected np.ndarray (got numpy.float32)":
                    raise
                relative = (Path(str(endpoint["scene_shard"])).parent / filename).as_posix()
                cache, row = self.array_cache.get(relative), int(endpoint["shard_row"])
                if (
                    type(cache) is not self.runtime.np.ndarray
                    or cache.dtype != self.runtime.np.dtype("<f4")
                    or cache.shape != (int(shard["endpoint_count"]),)
                    or not 0 <= row < cache.shape[0]
                ):
                    raise PermissionError("the ground scalar cache contract changed") from error
                scalar = cache[row]
                if type(scalar) is not self.runtime.np.float32 or scalar.shape != ():
                    raise PermissionError("the ground scalar row contract changed") from error
                tensor = self.runtime.torch.as_tensor(scalar)
                if tensor.shape != self.runtime.torch.Size([]) or tensor.dtype != self.runtime.torch.float32 or tensor.device.type != "cpu":
                    raise PermissionError("the ground scalar tensor contract changed") from error
                return tensor

    frozen_v1_train_arm = original_trainer.__mro__[1].train_arm

    class TrainerV3(original_trainer):
        def train_arm(self, **kwargs: Any) -> dict[int, dict[str, Any]]:
            predecessor = predecessor_state.get("value")
            if type(predecessor) is not dict:
                raise PermissionError("V2 predecessor was not validated before training")
            initialization, schedule_record = require_v2_initialization_and_schedule(self.output_root, predecessor, base._read_regular)
            initial_state, schedule = kwargs.get("initial_state"), kwargs.get("schedule")
            if (
                type(initial_state) is not dict
                or self.r.model_module.tensor_state_dict_sha256(initial_state) != initialization["complete_state_sha256"]
                or list(schedule) != schedule_record["presentation_indices"]
                or list(kwargs.get("vocabulary", ())) != initialization["primitive_vocabulary"]
                or getattr(kwargs.get("commanded_table"), "tolist", lambda: None)() != initialization["commanded_delta_table"]
            ):
                raise PermissionError("in-memory V3 initialization or schedule changed")
            return frozen_v1_train_arm(self, **kwargs)

    RawInputsV3.__name__, TrainerV3.__name__ = "RawInputsV3", "TrainerV3"

    def run_parent(*, review_file_sha256: str, authorization_file_sha256: str) -> int:
        predecessor_state["value"] = validate_predecessor(base._read_regular, root=ROOT)
        return original_run_parent(review_file_sha256=review_file_sha256, authorization_file_sha256=authorization_file_sha256)

    def run_internal_verifier() -> int:
        predecessor_state["value"] = validate_predecessor(base._read_regular, root=ROOT)
        return original_run_internal_verifier()

    base.RawInputs, base.Trainer = RawInputsV3, TrainerV3
    base.run_parent, base.run_internal_verifier = run_parent, run_internal_verifier
    setattr(base, INSTALLATION_SENTINEL, True)
    if set(vars(base_contract)) != set(contract_snapshot) or any(
        vars(base_contract)[name] is not value for name, value in contract_snapshot.items() if name not in CONTRACT_PATCH_WHITELIST
    ):
        raise RuntimeError("V3 changed an undeclared private-contract attribute")
    if set(vars(base)) != set(runner_snapshot) | {INSTALLATION_SENTINEL} or any(
        vars(base)[name] is not value for name, value in runner_snapshot.items() if name not in RUNNER_PATCH_WHITELIST
    ):
        raise RuntimeError("V3 changed an undeclared private-runner attribute")
    return base
