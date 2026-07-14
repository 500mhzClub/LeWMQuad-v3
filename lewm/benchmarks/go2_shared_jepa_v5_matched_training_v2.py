"""Narrow V2 successor for the frozen matched-training V1 implementation.

Importing this module is read-only.  ``install_successor`` explicitly verifies
and patches one exact private V1 runner instance.  The sole execution change is
normalizing the already-bound endpoint RGB metadata paths before V1 opens them.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path, PurePosixPath
import re
import stat
from types import FunctionType, ModuleType
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/lean_shared_v5_impl"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_matched_training_v2"

CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v2.py"
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_matched_training_v2.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_matched_training_v2.py"
V1_CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py"
V1_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_matched_training_v1.py"
V1_TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_matched_training_v1.py"
V1_SOURCE_SHA256 = {
    V1_CONTRACT_RELATIVE_PATH: "53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a",
    V1_RUNNER_RELATIVE_PATH: "e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578",
    V1_TEST_RELATIVE_PATH: "d6a56361d2a409d6270492d173843850c2c5a22344981a64ce63c43ab3c8307c",
}


def _load_exact_v1_contract() -> ModuleType:
    path = ROOT / V1_CONTRACT_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise PermissionError("the frozen V1 contract is not a regular file")
    if hashlib.sha256(path.read_bytes()).hexdigest() != V1_SOURCE_SHA256[
        V1_CONTRACT_RELATIVE_PATH
    ]:
        raise PermissionError("the frozen V1 contract changed")
    spec = importlib.util.spec_from_file_location(
        "_lewm_go2_shared_jepa_v5_matched_training_v2_readonly_v1_contract", path
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the frozen V1 contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v1 = _load_exact_v1_contract()
_V1_SCIENCE_CONTRACT = _v1.science_contract()
V1_REVIEW_RELATIVE_PATH = _v1.REVIEW_RELATIVE_PATH
V1_AUTHORIZATION_RELATIVE_PATH = _v1.AUTHORIZATION_RELATIVE_PATH
V1_RESERVATION_SCHEMA = _v1.RESERVATION_SCHEMA

REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_matched_training_v2_"
    "independent_review_2026-07-14.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_matched_training_v2_"
    "execution_authorization_2026-07-14.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/matched_training_v2"
)
PREDECESSOR_ROOT_RELATIVE_PATH = _v1.OUTPUT_ROOT_RELATIVE_PATH
TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_matched_training_v1_"
    "terminal_failure_audit_2026-07-14.json"
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
V1_INITIALIZATION_SCHEMA = "lewm_go2_shared_jepa_v5_matched_training_v1_initialization_v1"
V1_SCHEDULE_SCHEMA = _v1.SCHEDULE_SCHEMA

SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    TERMINAL_AUDIT_RELATIVE_PATH,
    *_v1.SOURCE_PATHS,
)

CONTRACT_PATCH_WHITELIST = (
    "IMPLEMENTATION_AUTHOR",
    "SCHEMA_PREFIX",
    "CONTRACT_RELATIVE_PATH",
    "RUNNER_RELATIVE_PATH",
    "TEST_RELATIVE_PATH",
    "SOURCE_PATHS",
    "REVIEW_RELATIVE_PATH",
    "AUTHORIZATION_RELATIVE_PATH",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "REVIEW_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "RESERVATION_SCHEMA",
    "SCHEDULE_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "SELECTION_SCHEMA",
    "CALIBRATION_SCHEMA",
    "PRE_G2_CHECKPOINT_SCHEMA",
    "RESULT_SCHEMA",
    "VERIFICATION_SCHEMA",
    "COMPLETION_SCHEMA",
    "FAILURE_SCHEMA",
)
RUNNER_PATCH_WHITELIST = (
    "RawInputs",
    "Trainer",
    "run_parent",
    "run_internal_verifier",
)
INSTALLATION_SENTINEL = "_lewm_matched_training_v2_installed"


def _binding(path: str, file_sha: str, content_sha: str, size: int) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": file_sha,
        "content_sha256": content_sha,
        "byte_count": size,
    }


PREDECESSOR_ARTIFACT_BINDINGS = {
    "reservation.json": _binding("reservation.json", "c9249c364b348954ba0ca93eec8c18f2ab9b863b0cdb3247a628d8ac13de0c15", "5aab40263525c36607bcd9b374d2028d0e28c889eaa0980548bb910f52efcafc", 7_998),
    "initialization.json": _binding("initialization.json", "985ff8ec9a9ee745ea982c4ea3112fb9c7efb2c2f18a6dd85437a1178ef7803d", "fdd831a452720ae33af5694c18bd7ac46a0429496ef4e9b15e3c53736ea00e98", 1_724),
    "schedule.json": _binding("schedule.json", "ea1f04d245ad28ef1d741f16f27a21bdfb17c338781cb72b2344ed171fa4cf3c", "893c48b2c2c591dbc90469e5a19a74e70bd54f96689b63881c216605255c0e5d", 607_373),
    "failed.json": _binding("failed.json", "76180f5a5e963c871fb87713ae03fe37db77e401b329aef8e00eecdfc098669f", "5a3a0c4769e2bcce9813eb23e36f05bfd222a64b211285070ab9649f19849665", 587),
}
V1_REVIEW_BINDING = _binding(V1_REVIEW_RELATIVE_PATH, "42ae2889688c491a39faf006365d195801ca57144e8afadaf824f45184562385", "7aae696caabcb5f6ddbf8d536a0b685e32c22b176643ca95b568f6fc39df9ddc", 5_327)
V1_AUTHORIZATION_BINDING = _binding(V1_AUTHORIZATION_RELATIVE_PATH, "39d3013de0e51d7672d9d4cfef421b70cd0fca7d3317a971d9cc407b92d6f0ff", "79d346ea27a8ec8a3b71380ea1a9444d1cd09aaf043eff56e9175f0bfc9c7e5f", 6_204)
V1_TERMINAL_AUDIT_BINDING = _binding(TERMINAL_AUDIT_RELATIVE_PATH, "844859737da59dc46de59390500cc7a6f8370680021d10cf2bc2244a22ae1330", "a1617d3b890046730b3859a924dcf41eec182f01e8510e66494d1c474e5df97f", 12_108)
PREDECESSOR_ATTEMPT_IDENTITY = "74269852a9921f2cf110ba08e3a93f2c89458f4a1c51ec6fc491320e6e9ee741"
PREDECESSOR_INITIALIZATION_IDENTITY = {
    "complete_state_sha256": "e03613bf5da2d93910630a0e2b98799a907f9a2b4767a0c2c36b1fa942cd2a87",
    "fit_model_state_sha256": "37196e82eb75b6ac7f2a9e0fc60949f5c5708f193337dd99b102761bbe4a149c",
    "shared_encoder_state_sha256": "28d365fed900bd5be481d413b5a8c9060e1183672063f083cde128bb65fc64e9",
    "evidence_head_state_sha256": "4469eb84189ca4213d5f14709227e79ceeb25e14f260dabfd6b7f6f5e7f1f1cc",
    "commanded_delta_table_sha256": "9feee2cc12a9b1f74e53b8300a0f065d731cff0353ba5b1ba951c8aaff7bbec5",
}
PREDECESSOR_SCHEDULE_IDENTITY = {
    "indices_sha256": "a6f4fda5eb570336fb360631af3629832cccbe4cba21bdbb325dcb8a21963663",
    "ordered_pair_ids_sha256": "74b90f10347a89d2151c4f65f76d6fc3c6a94fb3e8caa350d2a92e934e80840a",
    "presentation_pair_ids_sha256": "1534dcdd85feb8421639a0dc433473913f6674556e22e0fa9f515be455b7b79a",
    "per_update_pair_ids_sha256": "fe4aab82bd05b5e3438e8623319211ae75220f8bf3143223f6b6e375d91d46f0",
}
MAXIMUM_ATTEMPTS = 1
RETRY_AUTHORIZED = False
AUTOMATIC_V3_AUTHORIZED = False


def __getattr__(name: str) -> Any:
    return getattr(_v1, name)


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_V1_SCIENCE_CONTRACT)
    value["candidate"]["schema"] = PRE_G2_CHECKPOINT_SCHEMA
    return value


def v1_science_contract() -> dict[str, Any]:
    return copy.deepcopy(_V1_SCIENCE_CONTRACT)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    _v1.current_source_bindings(root)
    bindings = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }
    if any(bindings.get(path) != digest for path, digest in V1_SOURCE_SHA256.items()):
        raise PermissionError("the three frozen V1 implementation files changed")
    return bindings


def predecessor_contract() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_predecessor_contract_v1",
        "root": PREDECESSOR_ROOT_RELATIVE_PATH,
        "terminal_artifacts": {
            name: dict(value) for name, value in PREDECESSOR_ARTIFACT_BINDINGS.items()
        },
        "v1_sources": dict(V1_SOURCE_SHA256),
        "v1_review": dict(V1_REVIEW_BINDING),
        "v1_authorization": dict(V1_AUTHORIZATION_BINDING),
        "v1_terminal_audit": dict(V1_TERMINAL_AUDIT_BINDING),
        "attempt_identity": PREDECESSOR_ATTEMPT_IDENTITY,
        "failure": {
            "status": "failed_infrastructure",
            "stage": "promoted_training",
            "error": {
                "type": "ValueError",
                "message": "development RGB path escaped its root",
            },
            "g2_attempted": False,
            "heldout_open_count": 0,
            "retry_authorized": False,
        },
        "terminal_inventory_only": sorted(PREDECESSOR_ARTIFACT_BINDINGS),
        "v1_rerun_authorized": False,
        "v2_maximum_attempts": MAXIMUM_ATTEMPTS,
        "v2_retry_authorized": RETRY_AUTHORIZED,
        "automatic_v3_authorized": AUTOMATIC_V3_AUTHORIZED,
    }


def _read_bound_json(
    root: Path,
    binding: Mapping[str, Any],
    read_regular: Callable[..., bytes],
    *,
    name: str,
) -> dict[str, Any]:
    validated = _v1.validate_binding(binding)
    raw = read_regular(root / validated["path"], expected_sha256=validated["file_sha256"])
    if len(raw) != validated["byte_count"]:
        raise PermissionError(f"{name} byte count changed")
    value = _v1.parse_canonical_json(raw, name=name)
    if value["content_sha256"] != validated["content_sha256"]:
        raise PermissionError(f"{name} content binding changed")
    return value


def validate_predecessor(
    read_regular: Callable[..., bytes], *, root: Path = ROOT
) -> dict[str, Any]:
    predecessor_root = root / PREDECESSOR_ROOT_RELATIVE_PATH
    if predecessor_root.is_symlink() or not predecessor_root.is_dir():
        raise PermissionError("the V1 terminal root changed")
    entries = list(predecessor_root.iterdir())
    if {item.name for item in entries} != set(PREDECESSOR_ARTIFACT_BINDINGS):
        raise PermissionError("the V1 terminal inventory changed")
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise PermissionError("the V1 terminal inventory is not four regular files")
    artifacts = {
        name: _read_bound_json(predecessor_root, binding, read_regular, name=f"V1 {name}")
        for name, binding in PREDECESSOR_ARTIFACT_BINDINGS.items()
    }
    reservation = artifacts["reservation.json"]
    failed = artifacts["failed.json"]
    if (
        reservation.get("schema") != V1_RESERVATION_SCHEMA
        or reservation.get("status") != "reserved_before_torch_camera_raw_or_rgb"
        or reservation.get("attempt_identity") != PREDECESSOR_ATTEMPT_IDENTITY
        or reservation.get("retry_authorized") is not False
    ):
        raise PermissionError("the V1 reservation semantics changed")
    if (
        failed.get("schema")
        != "lewm_go2_shared_jepa_v5_matched_training_v1_failure_v1"
        or {
            "status": failed.get("status"),
            "stage": failed.get("stage"),
            "error": failed.get("error"),
            "g2_attempted": failed.get("g2_attempted"),
            "heldout_open_count": failed.get("heldout_open_count"),
            "retry_authorized": failed.get("retry_authorized"),
        }
        != predecessor_contract()["failure"]
        or failed.get("reservation") != PREDECESSOR_ARTIFACT_BINDINGS["reservation.json"]
    ):
        raise PermissionError("the V1 terminal failure changed")
    review = _read_bound_json(root, V1_REVIEW_BINDING, read_regular, name="V1 review")
    authorization = _read_bound_json(
        root, V1_AUTHORIZATION_BINDING, read_regular, name="V1 authorization"
    )
    audit = _read_bound_json(
        root, V1_TERMINAL_AUDIT_BINDING, read_regular, name="V1 terminal audit"
    )
    if (
        audit.get("schema")
        != "lewm_go2_shared_jepa_v5_matched_training_v1_terminal_failure_audit_v1"
        or audit.get("verdict")
        != "PASS_CONFIRMED_ZERO_TRAINING_INFRASTRUCTURE_PATH_CONTRACT_MISMATCH"
        or audit.get("terminal_inventory", {}).get("exact_entry_count") != 4
        or any(audit.get("zero_training_proof", {}).get("counts", {}).values())
    ):
        raise PermissionError("the independent V1 terminal audit changed")
    return {
        "artifacts": artifacts,
        "v1_review": review,
        "v1_authorization": authorization,
        "terminal_audit": audit,
    }


def normalize_lifecycle_artifact_to_v1(
    value: Mapping[str, Any], *, kind: str
) -> dict[str, Any]:
    schemas = {
        "initialization": (INITIALIZATION_SCHEMA, V1_INITIALIZATION_SCHEMA),
        "schedule": (SCHEDULE_SCHEMA, V1_SCHEDULE_SCHEMA),
    }
    if kind not in schemas or type(value) is not dict:
        raise TypeError("only plain V2 initialization or schedule artifacts normalize")
    current_schema, predecessor_schema = schemas[kind]
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        value.get("schema") != current_schema
        or not _v1.is_sha256(declared)
        or _v1.canonical_json_sha256(core) != declared
    ):
        raise PermissionError(f"V2 {kind} artifact changed")
    core["schema"] = predecessor_schema
    return _v1.with_content_sha256(core)


def require_v1_initialization_and_schedule(
    output_root: Path,
    predecessor: Mapping[str, Any],
    read_regular: Callable[..., bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = predecessor.get("artifacts")
    if type(artifacts) is not dict:
        raise PermissionError("validated V1 predecessor state is absent")
    current: dict[str, dict[str, Any]] = {}
    for kind, filename in (("initialization", "initialization.json"), ("schedule", "schedule.json")):
        value = _v1.parse_canonical_json(
            read_regular(output_root / filename), name=f"V2 {kind}"
        )
        if normalize_lifecycle_artifact_to_v1(value, kind=kind) != artifacts[filename]:
            raise PermissionError(f"V2 {kind} is not the exact V1 identity")
        current[kind] = value
    return current["initialization"], current["schedule"]


_SCENE_PATTERN = re.compile(r"scene_[0-9a-f]{16}")
_LEAF_PATTERN = re.compile(r"frame_[0-9]{6}_env_[0-9]{2}\.png")


def normalize_endpoint_rgb_path(
    value: object, *, repository_root: Path = ROOT
) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise PermissionError("endpoint RGB path is not canonical POSIX text")
    root = PurePosixPath(repository_root.as_posix())
    path = PurePosixPath(value)
    if (
        not repository_root.is_absolute()
        or root.as_posix() != repository_root.as_posix()
        or not path.is_absolute()
        or path.as_posix() != value
        or "." in path.parts
        or ".." in path.parts
    ):
        raise PermissionError("endpoint RGB path is not canonical absolute metadata")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise PermissionError("endpoint RGB path escaped the repository") from error
    parts = relative.parts
    if (
        len(parts) != 6
        or parts[:3] != (".generated", "go2_render_selected_v04", "scenes")
        or _SCENE_PATTERN.fullmatch(parts[3]) is None
        or parts[4] != "rgb"
        or _LEAF_PATTERN.fullmatch(parts[5]) is None
    ):
        raise PermissionError("endpoint RGB path escaped the selected-render layout")
    normalized = relative.as_posix()
    _v1.safe_relative_path(normalized, name="normalized endpoint RGB path")
    return normalized


def build_rgb_allowlist(
    endpoints: Mapping[str, Mapping[str, Any]], *, repository_root: Path = ROOT
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    allowlist: dict[str, dict[str, str]] = {}
    normalized_endpoints: dict[str, dict[str, Any]] = {}
    for endpoint_identity, row in endpoints.items():
        if type(endpoint_identity) is not str or type(row) is not dict:
            raise PermissionError("endpoint index mapping changed")
        source_path = row.get("image_path_metadata_only")
        image_sha = row.get("image_sha256_commitment_only")
        role = row.get("dataset_role")
        content_sha = row.get("content_sha256")
        if (
            row.get("endpoint_identity_sha256") != endpoint_identity
            or not _v1.is_sha256(endpoint_identity)
            or not _v1.is_sha256(image_sha)
            or not _v1.is_sha256(content_sha)
            or role not in _v1.ROLES
        ):
            raise PermissionError("endpoint RGB authority fields changed")
        relative = normalize_endpoint_rgb_path(source_path, repository_root=repository_root)
        if relative in allowlist:
            raise PermissionError("endpoint RGB allowlist path repeats")
        allowlist[relative] = {
            "file_sha256": image_sha,
            "dataset_role": role,
            "endpoint_identity_sha256": endpoint_identity,
            "endpoint_content_sha256": content_sha,
            "source_absolute_path": source_path,
        }
        normalized_endpoints[endpoint_identity] = {
            **row,
            "image_path_metadata_only": relative,
        }
    return allowlist, normalized_endpoints


def _reject_symlink_components(relative: str, *, repository_root: Path) -> None:
    _v1.safe_relative_path(relative, name="internal endpoint RGB path")
    cursor = repository_root
    for part in ("", *PurePosixPath(relative).parts):
        if part:
            cursor = cursor / part
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError("endpoint RGB path contains a symlink component")


def _exact_module(module: object, relative: str, expected_sha256: str) -> ModuleType:
    if type(module) is not ModuleType or type(getattr(module, "__file__", None)) is not str:
        raise PermissionError("frozen V1 module identity changed")
    path = Path(module.__file__)
    expected = ROOT / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != expected.resolve()
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256
    ):
        raise PermissionError("frozen V1 module bytes changed")
    return module


def _patch_values() -> dict[str, Any]:
    return {name: globals()[name] for name in CONTRACT_PATCH_WHITELIST}


def _private_contract_matches_frozen_baseline(value: ModuleType) -> bool:
    expected_names = {name for name in vars(_v1) if not name.startswith("__")}
    if {name for name in vars(value) if not name.startswith("__")} != expected_names:
        return False
    for name in expected_names:
        expected = vars(_v1)[name]
        observed = vars(value)[name]
        if type(expected) is FunctionType:
            if type(observed) is not FunctionType or observed.__code__ != expected.__code__:
                return False
        elif type(expected) is ModuleType:
            if type(observed) is not ModuleType or observed.__name__ != expected.__name__:
                return False
        elif observed != expected:
            return False
    return True


def install_successor(
    base_runner: object,
    baseline_namespace: Mapping[str, object],
) -> ModuleType:
    """Install V2 once into one exact, otherwise untouched private V1 runner."""
    base = _exact_module(
        base_runner,
        V1_RUNNER_RELATIVE_PATH,
        V1_SOURCE_SHA256[V1_RUNNER_RELATIVE_PATH],
    )
    if INSTALLATION_SENTINEL in vars(base):
        raise RuntimeError("the V2 successor was already installed")
    if set(vars(base)) != set(baseline_namespace) or any(
        vars(base)[name] is not value for name, value in baseline_namespace.items()
    ):
        raise PermissionError("the private V1 runner namespace drifted")
    base_contract = _exact_module(
        getattr(base, "contract", None),
        V1_CONTRACT_RELATIVE_PATH,
        V1_SOURCE_SHA256[V1_CONTRACT_RELATIVE_PATH],
    )
    patch_values = _patch_values()
    if not _private_contract_matches_frozen_baseline(base_contract):
        raise PermissionError("the private V1 contract baseline drifted")
    for name, expected_name in (
        ("RawInputs", "RawInputs"),
        ("Trainer", "Trainer"),
        ("run_parent", "run_parent"),
        ("run_internal_verifier", "run_internal_verifier"),
    ):
        value = vars(base).get(name)
        if getattr(value, "__name__", None) != expected_name or getattr(
            value, "__module__", None
        ) != base.__name__:
            raise PermissionError("the private V1 runner baseline drifted")

    contract_snapshot = dict(vars(base_contract))
    runner_snapshot = dict(vars(base))
    original_raw_inputs = base.RawInputs
    original_trainer = base.Trainer
    original_run_parent = base.run_parent
    original_run_internal_verifier = base.run_internal_verifier
    predecessor_state: dict[str, Any] = {}

    for name, value in patch_values.items():
        setattr(base_contract, name, value)

    class RawInputsV2(original_raw_inputs):
        def __init__(self, runtime: Any, authorization: Mapping[str, Any]) -> None:
            super().__init__(runtime, authorization)
            self.bound_endpoints = dict(self.endpoints)
            allowlist, normalized = build_rgb_allowlist(self.bound_endpoints)
            if len(allowlist) != 9_460 or len(normalized) != 9_460:
                raise PermissionError("endpoint RGB allowlist population changed")
            self.rgb_allowlist = allowlist
            self.endpoints = normalized
            self._active_rgb_endpoint: str | None = None

        def frame(
            self, endpoint_id: str, *, role: str, arm: str, stage: str
        ) -> dict[str, Any]:
            if self._active_rgb_endpoint is not None:
                raise RuntimeError("nested endpoint RGB authority is forbidden")
            self._active_rgb_endpoint = endpoint_id
            try:
                return super().frame(endpoint_id, role=role, arm=arm, stage=stage)
            finally:
                self._active_rgb_endpoint = None

        def read_rgb(
            self,
            relative: str,
            expected_sha256: str,
            *,
            role: str,
            arm: str,
            stage: str,
        ) -> bytes:
            base_contract.safe_relative_path(relative, name="internal endpoint RGB path")
            authority = self.rgb_allowlist.get(relative)
            if (
                type(authority) is not dict
                or authority.get("file_sha256") != expected_sha256
                or authority.get("dataset_role") != role
                or authority.get("endpoint_identity_sha256")
                != self._active_rgb_endpoint
            ):
                raise PermissionError("endpoint RGB read escaped its exact allowlist row")
            _reject_symlink_components(relative, repository_root=ROOT)
            return super().read_rgb(
                relative,
                expected_sha256,
                role=role,
                arm=arm,
                stage=stage,
            )

    class TrainerV2(original_trainer):
        def train_arm(self, **kwargs: Any) -> dict[int, dict[str, Any]]:
            predecessor = predecessor_state.get("value")
            if type(predecessor) is not dict:
                raise PermissionError("V1 predecessor was not validated before training")
            initialization, schedule_record = require_v1_initialization_and_schedule(
                self.output_root, predecessor, base._read_regular
            )
            initial_state = kwargs.get("initial_state")
            schedule = kwargs.get("schedule")
            if (
                type(initial_state) is not dict
                or self.r.model_module.tensor_state_dict_sha256(initial_state)
                != initialization["complete_state_sha256"]
                or list(schedule) != schedule_record["presentation_indices"]
                or list(kwargs.get("vocabulary", ()))
                != initialization["primitive_vocabulary"]
                or getattr(kwargs.get("commanded_table"), "tolist", lambda: None)()
                != initialization["commanded_delta_table"]
            ):
                raise PermissionError("in-memory V2 initialization or schedule changed")
            return super().train_arm(**kwargs)

    RawInputsV2.__name__ = "RawInputsV2"
    TrainerV2.__name__ = "TrainerV2"

    def run_parent(*, review_file_sha256: str, authorization_file_sha256: str) -> int:
        predecessor_state["value"] = validate_predecessor(base._read_regular, root=ROOT)
        return original_run_parent(
            review_file_sha256=review_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
        )

    def run_internal_verifier() -> int:
        predecessor_state["value"] = validate_predecessor(base._read_regular, root=ROOT)
        return original_run_internal_verifier()

    base.RawInputs = RawInputsV2
    base.Trainer = TrainerV2
    base.run_parent = run_parent
    base.run_internal_verifier = run_internal_verifier
    setattr(base, INSTALLATION_SENTINEL, True)
    if set(vars(base_contract)) != set(contract_snapshot) or any(
        vars(base_contract)[name] is not value
        for name, value in contract_snapshot.items()
        if name not in CONTRACT_PATCH_WHITELIST
    ):
        raise RuntimeError("V2 changed an undeclared private-contract attribute")
    if set(vars(base)) != set(runner_snapshot) | {INSTALLATION_SENTINEL} or any(
        vars(base)[name] is not value
        for name, value in runner_snapshot.items()
        if name not in RUNNER_PATCH_WHITELIST
    ):
        raise RuntimeError("V2 changed an undeclared private-runner attribute")
    return base
