#!/usr/bin/env python3
"""Fail-closed, source-only V1 -> V2 science-identity and delta guard.

The guard reads only the exact source and authority documents enumerated
below.  It never imports project modules, walks the repository, opens a
generated path, deserializes a checkpoint, imports Torch, or queries a device.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCIENCE_SHA256 = (
    "e181381c00585fa5df41a71fff918b5599acc955d59283ce397ba6dd530dc23f"
)
V1_ROOT = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v1"
)
V2_ROOT = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v2"
)
MODEL_PATH = "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py"
MODEL_SHA256 = (
    "a63da1137539953b2f40d184def1652ae05f63d7b434084b1a91787e1fc83d0b"
)
MODEL_FAMILY = "shared_observable_camera_ray_jepa_v5_multires_v1"
MODEL_RUNTIME_VERSION = (
    "lewm_go2_rgb_multiresolution_perception_v1_model_runtime_v1"
)

V1_CONTRACT = "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v1.py"
V2_CONTRACT = "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2.py"
V1_RUNNER = "scripts/run_go2_shared_jepa_v5_multires_probe_v1.py"
V2_RUNNER = "scripts/run_go2_shared_jepa_v5_multires_probe_v2.py"
V1_LAUNCHER = "scripts/launch_go2_shared_jepa_v5_multires_probe_v1.py"
V2_LAUNCHER = "scripts/launch_go2_shared_jepa_v5_multires_probe_v2.py"
V1_TEST = "lewm/tests/test_go2_shared_jepa_v5_multires_probe_v1.py"
V2_TEST = "lewm/tests/test_go2_shared_jepa_v5_multires_probe_v2.py"
V1_CLOSURE_CHECKER = "scripts/check_go2_multires_probe_source_closure.py"
V2_CLOSURE_CHECKER = "scripts/check_go2_multires_probe_source_closure_v2.py"
V1_CLOSURE_TEST = "lewm/tests/test_go2_multires_probe_source_closure.py"
V2_CLOSURE_TEST = "lewm/tests/test_go2_multires_probe_source_closure_v2.py"
V2_SCHEDULE_ADAPTER = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py"
)
V2_SCHEDULE_ADAPTER_SHA256 = (
    "a8efe19da92c9c2107f11be38db8ed80e66aedca3ef41af0428ab13d50f56bd1"
)

V1_AUTHORIZATION = (
    "docs/lewm_go2_rgb_multiresolution_perception_v1_"
    "execution_authorization_2026-07-24.json"
)
V1_TERMINAL_AUDIT = (
    "docs/lewm_go2_rgb_multiresolution_perception_v1_"
    "terminal_lifecycle_failure_audit_2026-07-24.json"
)
V2_DECISION = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "operational_recovery_decision_2026-07-24.md"
)
V2_PREREGISTRATION = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "preregistration_2026-07-24.json"
)
V2_PREREGISTRATION_REVIEW = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "preregistration_independent_review_2026-07-24.json"
)

# These are the seven V1 implementation/verification files explicitly frozen
# before V2 source work.  V2 never replaces or mutates any of them.
V1_FROZEN_SOURCE_SHA256 = {
    V1_CONTRACT:
        "ffdeb2b6b3a03a1b1b65e2fe3961a8561717c8ced4d800c640f03710af40fa3b",
    V1_RUNNER:
        "c84604df4933a04939c297fa68e765ec6c00e68d360da0c6ed8de5a56ba87e41",
    V1_LAUNCHER:
        "adf97ed861c2f37960db1fbc171c91913847d2f4a98e553ea903d9371419f42e",
    V1_TEST:
        "dba0954f9eed9d700bfe808b6911466cce8728cef247788fbcfe00b65798de0b",
    V1_CLOSURE_CHECKER:
        "ac9fcaa9107ad43201b5082581c0743ebb46653ff8b51a6f09c33fc992142911",
    V1_CLOSURE_TEST:
        "fb09c98b0f008eb863622dab1b4204535b719734eaf9293adb6eaefd3417f846",
    MODEL_PATH: MODEL_SHA256,
}

AUTHORITY_FILE_BINDINGS = {
    V1_AUTHORIZATION: {
        "byte_count": 7_834,
        "content_sha256":
            "cb06d8642484e95030fc9ce26b57f2efe60b7977ebb99ae1373321b97d9551ed",
        "file_sha256":
            "522cba9cefed795cfd03b9db3949881a65fe24620821bc463a96a7920326c542",
    },
    V1_TERMINAL_AUDIT: {
        "byte_count": 7_363,
        "content_sha256":
            "ccfc14731e569aed773d4380865395a60e00d8354ba9903757b1f23675a7b3d3",
        "file_sha256":
            "6adaaaea3ec1d63438f63e5282b832c27c34348075f57317070acd04b615b541",
    },
    V2_DECISION: {
        "byte_count": 7_565,
        "file_sha256":
            "9df833efb3949744e66cb5263d341baef69241d4b2b1653d90ca9bf87f8ec1fb",
    },
    V2_PREREGISTRATION: {
        "byte_count": 8_576,
        "content_sha256":
            "264a4e3d52dd0ec658afce8c4bc54f86e9c18bbfb43229c14521b5f683a6514a",
        "file_sha256":
            "642897b82ccdee6ac6c23168754056335d7a3701a19ccfc682527872461f16cc",
    },
    V2_PREREGISTRATION_REVIEW: {
        "byte_count": 4_961,
        "content_sha256":
            "6abd1b01aa7e4df68b1fe05b0ff854124971d5b1f2f4eccd34aa42320987e04c",
        "file_sha256":
            "b8314774a707e1f8af8db214d0c12fe304352710b2ff4d569068b9c3d184bf84",
    },
}
V2_PREREGISTRATION_COMMIT = "5849dc497acd272d56026c00b821b3662b040752"

READABLE_SOURCE_PATHS = frozenset({
    *V1_FROZEN_SOURCE_SHA256,
    V2_CONTRACT,
    V2_RUNNER,
    V2_LAUNCHER,
    V2_TEST,
    V2_CLOSURE_CHECKER,
    V2_CLOSURE_TEST,
    V2_SCHEDULE_ADAPTER,
    *AUTHORITY_FILE_BINDINGS,
})

_SCIENCE_CONSTANTS = {
    "BASE_INITIALIZATION_SEED",
    "CHECKPOINT_SCHEDULE_PREFIX_SHA256",
    "CHECKPOINT_UPDATES",
    "DECODER_INITIALIZATION_SEED",
    "DOWNSTREAM_DENIALS",
    "EFFECTIVE_BATCH_SIZE",
    "EXPECTED_PARAMETER_COUNTS",
    "EXPECTED_PARAMETER_TENSOR_COUNTS",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATE",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "MODEL_RUNTIME_VERSION",
    "PASS_THRESHOLDS",
    "POST_CLIP_NORM_ASSERTION_TOLERANCE",
    "SCHEDULE_SEED",
    "SELECTION_ROLE_COUNTS",
    "TRAIN_ROLE_COUNTS",
}


class GuardFailure(RuntimeError):
    """A source identity, authority, or operational-delta invariant failed."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_relative_source(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        relative not in READABLE_SOURCE_PATHS
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or any(part == ".generated" for part in path.parts)
        or any(part == "checkpoints" for part in path.parts)
        or any(part == "sealed" or part.startswith("sealed_") for part in path.parts)
        or path.name == "sealed_test.json"
        or path.suffix not in {".py", ".json", ".md"}
    ):
        raise PermissionError(f"path is outside the source-only allowlist: {relative}")


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_regular_source(root: Path, relative: str) -> bytes:
    _safe_relative_source(relative)
    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError("O_NOFOLLOW is required for source custody")
    path = root / relative
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"source is not a regular file: {relative}")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise PermissionError(f"opened source is not regular: {relative}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    if not (
        _fingerprint(before)
        == _fingerprint(opened_before)
        == _fingerprint(opened_after)
        == _fingerprint(after)
    ):
        raise GuardFailure(f"source changed while read: {relative}")
    return b"".join(chunks)


def _parse_source(root: Path, relative: str) -> tuple[bytes, ast.Module]:
    raw = _read_regular_source(root, relative)
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GuardFailure(f"source is not UTF-8: {relative}") from error
    return raw, ast.parse(source, filename=relative)


def _parse_canonical_json(
    root: Path,
    relative: str,
    *,
    expected: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    raw = _read_regular_source(root, relative)
    if (
        len(raw) != expected["byte_count"]
        or _sha256(raw) != expected["file_sha256"]
        or not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
    ):
        raise GuardFailure(f"canonical authority file binding changed: {relative}")
    try:
        value = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GuardFailure(f"authority file is not canonical ASCII JSON: {relative}") from error
    if type(value) is not dict or _canonical_bytes(value) + b"\n" != raw:
        raise GuardFailure(f"authority file canonicalization changed: {relative}")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        declared != expected.get("content_sha256")
        or _sha256(_canonical_bytes(core)) != declared
    ):
        raise GuardFailure(f"authority content self-hash changed: {relative}")
    return raw, value


def _literal_assignments(tree: ast.Module) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for statement in tree.body:
        name: str | None = None
        value: ast.expr | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            name = statement.targets[0].id
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            name = statement.target.id
            value = statement.value
        if name is None or value is None:
            continue
        try:
            result[name] = ast.literal_eval(value)
        except (TypeError, ValueError):
            continue
    return result


class _StripAnnotations(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = copy.deepcopy(node)
        node.returns = None
        node.decorator_list = []
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            argument.annotation = None
        if node.args.vararg is not None:
            node.args.vararg.annotation = None
        if node.args.kwarg is not None:
            node.args.kwarg.annotation = None
        return self.generic_visit(node)


def _science_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == name
    ]
    if len(matches) != 1:
        raise GuardFailure(f"expected exactly one {name} definition")
    return matches[0]


def _assert_science_ast_is_pure(function: ast.FunctionDef) -> None:
    forbidden = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.Delete,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.Try,
        ast.While,
        ast.With,
        ast.Yield,
        ast.YieldFrom,
    )
    allowed_named_calls = {
        "ValueError",
        "dict",
        "len",
        "list",
        "operation_counts",
        "str",
        "tuple",
        "type",
    }
    for node in ast.walk(function):
        if isinstance(node, forbidden):
            raise GuardFailure(f"impure node in {function.name}: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_named_calls:
                    raise GuardFailure(
                        f"unauthorized call in {function.name}: {node.func.id}"
                    )
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "items"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "CHECKPOINT_SCHEDULE_PREFIX_SHA256"
            ):
                continue
            else:
                raise GuardFailure(f"unauthorized attribute call in {function.name}")
        if isinstance(node, ast.Attribute) and not (
            isinstance(node.ctx, ast.Load)
            and node.attr == "items"
            and isinstance(node.value, ast.Name)
            and node.value.id == "CHECKPOINT_SCHEDULE_PREFIX_SHA256"
        ):
            raise GuardFailure(f"unauthorized attribute in {function.name}")


def _evaluate_science_contract(tree: ast.Module) -> dict[str, Any]:
    literals = _literal_assignments(tree)
    missing = _SCIENCE_CONSTANTS - literals.keys()
    if missing:
        raise GuardFailure(f"science constants are not literal-bound: {sorted(missing)}")
    operation = _science_function(tree, "operation_counts")
    science = _science_function(tree, "science_contract")
    _assert_science_ast_is_pure(operation)
    _assert_science_ast_is_pure(science)
    stripped = _StripAnnotations()
    module = ast.Module(
        body=[
            stripped.visit(operation),
            stripped.visit(science),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    safe_builtins = {
        "ValueError": ValueError,
        "dict": dict,
        "int": int,
        "len": len,
        "list": list,
        "str": str,
        "tuple": tuple,
        "type": type,
    }
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        **{name: literals[name] for name in _SCIENCE_CONSTANTS},
    }
    exec(compile(module, "<source-only-science-contract>", "exec"), namespace)
    value = namespace["science_contract"]()
    if type(value) is not dict:
        raise GuardFailure("science_contract() did not return a plain dict")
    # A canonical JSON round trip also rejects non-JSON values and non-finite
    # numbers before the object participates in an identity decision.
    return json.loads(_canonical_bytes(value).decode("ascii"))


def verify_v1_frozen_sources(root: Path = ROOT) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in V1_FROZEN_SOURCE_SHA256.items():
        digest = _sha256(_read_regular_source(root, relative))
        if digest != expected:
            raise GuardFailure(f"frozen V1 source changed: {relative}")
        observed[relative] = digest
    if len(observed) != 7:
        raise GuardFailure("the exact seven-file V1 freeze changed")
    return observed


def verify_science_identity(root: Path = ROOT) -> dict[str, Any]:
    _, authorization = _parse_canonical_json(
        root,
        V1_AUTHORIZATION,
        expected=AUTHORITY_FILE_BINDINGS[V1_AUTHORIZATION],
    )
    _, preregistration = _parse_canonical_json(
        root,
        V2_PREREGISTRATION,
        expected=AUTHORITY_FILE_BINDINGS[V2_PREREGISTRATION],
    )
    _, v1_tree = _parse_source(root, V1_CONTRACT)
    _, v2_tree = _parse_source(root, V2_CONTRACT)
    v1_source_science = _evaluate_science_contract(v1_tree)
    v2_source_science = _evaluate_science_contract(v2_tree)
    v1_authorized_science = authorization.get("experiment")
    embedded = preregistration.get("science_identity", {}).get("science_contract")
    values = (
        v1_source_science,
        v2_source_science,
        v1_authorized_science,
        embedded,
    )
    if any(type(value) is not dict for value in values):
        raise GuardFailure("one science identity source is not a plain object")
    if not all(value == v1_source_science for value in values[1:]):
        raise GuardFailure("V2 science_contract() is not deeply equal to V1")
    digests = tuple(_sha256(_canonical_bytes(value)) for value in values)
    if digests != (SCIENCE_SHA256,) * len(values):
        raise GuardFailure(f"science identity digest changed: {digests}")
    return {
        "deep_equal_source_count": len(values),
        "science_contract_sha256": SCIENCE_SHA256,
    }


def _binding_without_path(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: binding[key]
        for key in ("byte_count", "content_sha256", "file_sha256")
        if key in binding
    }


def verify_preregistration_authority(root: Path = ROOT) -> dict[str, Any]:
    decision_raw = _read_regular_source(root, V2_DECISION)
    decision_expected = AUTHORITY_FILE_BINDINGS[V2_DECISION]
    if (
        len(decision_raw) != decision_expected["byte_count"]
        or _sha256(decision_raw) != decision_expected["file_sha256"]
    ):
        raise GuardFailure("V2 recovery decision binding changed")
    _, preregistration = _parse_canonical_json(
        root,
        V2_PREREGISTRATION,
        expected=AUTHORITY_FILE_BINDINGS[V2_PREREGISTRATION],
    )
    _, review = _parse_canonical_json(
        root,
        V2_PREREGISTRATION_REVIEW,
        expected=AUTHORITY_FILE_BINDINGS[V2_PREREGISTRATION_REVIEW],
    )
    decision_binding = preregistration.get("decision_binding")
    if decision_binding != {
        "byte_count": decision_expected["byte_count"],
        "file_sha256": decision_expected["file_sha256"],
        "path": V2_DECISION,
    }:
        raise GuardFailure("preregistration decision binding changed")
    reviewed = review.get("reviewed_files")
    if type(reviewed) is not dict:
        raise GuardFailure("independent preregistration review bindings are absent")
    expected_reviewed = {
        V1_AUTHORIZATION:
            _binding_without_path(AUTHORITY_FILE_BINDINGS[V1_AUTHORIZATION]),
        V1_TERMINAL_AUDIT: {
            **_binding_without_path(AUTHORITY_FILE_BINDINGS[V1_TERMINAL_AUDIT]),
            "commit": "e3e0cc50877c9dc5cbd7d269e4b169f19857e897",
        },
        V2_DECISION: {
            "byte_count": decision_expected["byte_count"],
            "file_sha256": decision_expected["file_sha256"],
        },
        V2_PREREGISTRATION:
            _binding_without_path(AUTHORITY_FILE_BINDINGS[V2_PREREGISTRATION]),
    }
    if reviewed != expected_reviewed:
        raise GuardFailure("independent preregistration reviewed-file bindings changed")
    if (
        preregistration.get("schema")
        != "lewm_go2_rgb_multiresolution_perception_v2_preregistration_v1"
        or preregistration.get("status")
        != "PREREGISTERED_SOURCE_ONLY_PENDING_INDEPENDENT_RECOVERY_AND_SOURCE_REVIEW"
        or preregistration.get("fresh_attempt", {}).get("output_root") != V2_ROOT
        or preregistration.get("fresh_attempt", {}).get("v1_root_must_remain_sealed")
        is not True
        or preregistration.get("fresh_attempt", {}).get(
            "v1_runtime_output_open_authorized"
        )
        is not False
        or review.get("status")
        != "PASS_SOURCE_FREE_SCIENCE_IDENTICAL_OPERATIONAL_RECOVERY_PREREGISTRATION"
        or review.get("checks", {}).get("science_identity", {}).get(
            "v1_and_v2_canonical_sha256"
        )
        != SCIENCE_SHA256
        or review.get("authority", {}).get("execution_or_gpu_authorized") is not False
    ):
        raise GuardFailure("V2 preregistration or independent-review authority changed")
    _, v2_tree = _parse_source(root, V2_CONTRACT)
    constants = _literal_assignments(v2_tree)
    expected_constants = {
        "PREREGISTRATION_COMMIT": V2_PREREGISTRATION_COMMIT,
        "PREREGISTRATION_CONTENT_SHA256":
            AUTHORITY_FILE_BINDINGS[V2_PREREGISTRATION]["content_sha256"],
        "PREREGISTRATION_FILE_SHA256":
            AUTHORITY_FILE_BINDINGS[V2_PREREGISTRATION]["file_sha256"],
        "PREREGISTRATION_RELATIVE_PATH": V2_PREREGISTRATION,
        "PREREGISTRATION_REVIEW_FILE_SHA256":
            AUTHORITY_FILE_BINDINGS[V2_PREREGISTRATION_REVIEW]["file_sha256"],
        "PREREGISTRATION_REVIEW_RELATIVE_PATH": V2_PREREGISTRATION_REVIEW,
        "RECOVERY_DECISION_FILE_SHA256": decision_expected["file_sha256"],
        "RECOVERY_DECISION_RELATIVE_PATH": V2_DECISION,
    }
    for name, expected in expected_constants.items():
        if constants.get(name) != expected:
            raise GuardFailure(f"V2 contract authority constant changed: {name}")
    return {
        "decision_file_sha256": decision_expected["file_sha256"],
        "preregistration_content_sha256":
            AUTHORITY_FILE_BINDINGS[V2_PREREGISTRATION]["content_sha256"],
        "preregistration_review_content_sha256":
            AUTHORITY_FILE_BINDINGS[V2_PREREGISTRATION_REVIEW]["content_sha256"],
    }


def verify_model_and_roots(root: Path = ROOT) -> dict[str, Any]:
    _, v1_tree = _parse_source(root, V1_CONTRACT)
    _, v2_tree = _parse_source(root, V2_CONTRACT)
    _, model_tree = _parse_source(root, MODEL_PATH)
    v1 = _literal_assignments(v1_tree)
    v2 = _literal_assignments(v2_tree)
    model = _literal_assignments(model_tree)
    for name, expected in {
        "MODEL_FILE_SHA256": MODEL_SHA256,
        "MODEL_RELATIVE_PATH": MODEL_PATH,
        "MODEL_RUNTIME_VERSION": MODEL_RUNTIME_VERSION,
    }.items():
        if v1.get(name) != expected or v2.get(name) != expected:
            raise GuardFailure(f"model/runtime identity changed: {name}")
    if model.get("MODEL_FAMILY") != MODEL_FAMILY:
        raise GuardFailure("model-family source identity changed")
    if v1.get("OUTPUT_ROOT_RELATIVE_PATH") != V1_ROOT:
        raise GuardFailure("V1 output-root identity changed")
    if v2.get("OUTPUT_ROOT_RELATIVE_PATH") != V2_ROOT or V2_ROOT == V1_ROOT:
        raise GuardFailure("V2 output root is not exactly distinct")
    versioned_envelope_literals = {
        "AUTHORIZATION_RELATIVE_PATH",
        "CONTRACT_RELATIVE_PATH",
        "LAUNCHER_RELATIVE_PATH",
        "OUTPUT_ROOT_RELATIVE_PATH",
        "PREREGISTRATION_COMMIT",
        "PREREGISTRATION_CONTENT_SHA256",
        "PREREGISTRATION_FILE_SHA256",
        "PREREGISTRATION_RELATIVE_PATH",
        "REVIEW_RELATIVE_PATH",
        "RUNNER_RELATIVE_PATH",
        "SCHEMA_PREFIX",
        "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH",
        "SOURCE_CLOSURE_TEST_RELATIVE_PATH",
        "SOURCE_MANIFEST_RELATIVE_PATH",
        "TEST_RELATIVE_PATH",
    }
    changed_common_literals = {
        name
        for name in set(v1) & set(v2)
        if v1[name] != v2[name]
    }
    if changed_common_literals != versioned_envelope_literals:
        raise GuardFailure(
            "literal runtime/science configuration changed outside the "
            f"versioned envelope: {sorted(changed_common_literals)}"
        )
    return {
        "unchanged_common_literal_binding_count":
            len((set(v1) & set(v2)) - changed_common_literals),
        "model_file_sha256": MODEL_SHA256,
        "model_family": MODEL_FAMILY,
        "model_runtime_version": MODEL_RUNTIME_VERSION,
        "v1_output_root": V1_ROOT,
        "v2_output_root": V2_ROOT,
    }


def _definition_map(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if statement.name in result:
                raise GuardFailure(f"duplicate top-level definition: {statement.name}")
            result[statement.name] = statement
    return result


def _assignment_map(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for statement in tree.body:
        targets: list[ast.expr] = []
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            result[targets[0].id] = statement
    return result


class _VersionStringNormalizer(ast.NodeTransformer):
    _REPLACEMENTS = (
        ("go2_shared_jepa_v5_multires_probe_v2", "go2_shared_jepa_v5_multires_probe_v1"),
        ("rgb_multiresolution_perception_probe_v2", "rgb_multiresolution_perception_probe_v1"),
        ("rgb_multiresolution_perception_v2", "rgb_multiresolution_perception_v1"),
        ("source_closure_v2.py", "source_closure.py"),
    )

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not isinstance(node.value, str):
            return node
        value = node.value
        for old, new in self._REPLACEMENTS:
            value = value.replace(old, new)
        return ast.copy_location(ast.Constant(value=value), node)


def _normalized_dump(node: ast.AST) -> str:
    normalized = _VersionStringNormalizer().visit(copy.deepcopy(node))
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def _references_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Name) and item.id == name
        for item in ast.walk(node)
    )


class _TrainingProgressStripper(ast.NodeTransformer):
    """Remove receipt-only V2 instrumentation from a copied `_train` AST."""

    def __init__(self) -> None:
        self.microbatch_indices: ast.expr | None = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = copy.deepcopy(node)
        assignments = [
            item
            for item in ast.walk(node)
            if (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
                and item.targets[0].id == "microbatch_indices"
            )
        ]
        if len(assignments) > 1:
            raise GuardFailure("training introduced multiple schedule-slice aliases")
        if assignments:
            self.microbatch_indices = assignments[0].value
        node.args.args = [
            argument for argument in node.args.args if argument.arg != "progress"
        ]
        node.args.kwonlyargs = [
            argument
            for argument in node.args.kwonlyargs
            if argument.arg != "progress"
        ]
        return self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        if _references_name(node, "progress") or (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_failure_boundary"
        ):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if any(
            isinstance(target, ast.Name) and target.id == "microbatch_indices"
            for target in node.targets
        ):
            return None
        if any(_references_name(target, "progress") for target in node.targets):
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if _references_name(node.target, "progress"):
            return None
        return self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> ast.AST:
        kept = [
            (key, value)
            for key, value in zip(node.keys, node.values, strict=True)
            if not (
                isinstance(key, ast.Constant)
                and key.value == "partial_operation_counts"
            )
        ]
        node.keys = [self.visit(key) if key is not None else None for key, _ in kept]
        node.values = [self.visit(value) for _, value in kept]
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            node.id == "microbatch_indices"
            and isinstance(node.ctx, ast.Load)
            and self.microbatch_indices is not None
        ):
            return ast.copy_location(copy.deepcopy(self.microbatch_indices), node)
        return node


def _normalized_training_dump(node: ast.AST) -> str:
    stripped = _TrainingProgressStripper().visit(copy.deepcopy(node))
    if stripped is None:
        raise GuardFailure("training definition disappeared during normalization")
    return _normalized_dump(stripped)


def _definition_delta(
    v1_tree: ast.Module,
    v2_tree: ast.Module,
) -> tuple[set[str], set[str], set[str]]:
    before = _definition_map(v1_tree)
    after = _definition_map(v2_tree)
    added = set(after) - set(before)
    removed = set(before) - set(after)
    changed = {
        name
        for name in set(before) & set(after)
        if _normalized_dump(before[name]) != _normalized_dump(after[name])
    }
    return added, removed, changed


def _function_call_names(function: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _definition(
    definitions: Mapping[str, ast.AST],
    name: str,
    expected_type: type[ast.AST],
) -> ast.AST:
    value = definitions.get(name)
    if not isinstance(value, expected_type):
        raise GuardFailure(f"required definition changed: {name}")
    return value


def _method_map(class_node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        statement.name: statement
        for statement in class_node.body
        if isinstance(statement, ast.FunctionDef)
    }


def _dict_constant_keys(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Dict):
            result.update(
                str(key.value)
                for key in item.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
    return result


def _literal_function_return(
    definitions: Mapping[str, ast.AST],
    name: str,
) -> Any:
    function = _definition(definitions, name, ast.FunctionDef)
    assert isinstance(function, ast.FunctionDef)
    returns = [
        statement
        for statement in function.body
        if isinstance(statement, ast.Return)
    ]
    if len(returns) != 1 or returns[0].value is None:
        raise GuardFailure(f"{name} is not one literal return")
    try:
        return ast.literal_eval(returns[0].value)
    except (TypeError, ValueError) as error:
        raise GuardFailure(f"{name} return is not literal custody data") from error


def verify_operational_mechanisms(root: Path = ROOT) -> dict[str, Any]:
    _, contract_tree = _parse_source(root, V2_CONTRACT)
    _, runner_tree = _parse_source(root, V2_RUNNER)
    contract_definitions = _definition_map(contract_tree)
    runner_definitions = _definition_map(runner_tree)

    lifecycle = _literal_function_return(contract_definitions, "lifecycle_contract")
    expected_order = [
        "exact_source_review_and_authorization_rehash",
        "validate_isolated_no_tensor_hardware_preflight",
        "reserve_unique_mode_0700_output_root",
        "create_fsynced_hash_chained_partial_access_ledger",
        "deferred_torch_stack_import",
        "ledgered_bound_schedule_owner_validation_first",
        "ledgered_n320_and_raw_runtime_input_load",
        "schedule_ordered_train_identity_finalization_without_reopen",
        "training_update",
        "cpu_snapshot",
        "one_inline_nonmutating_selection_evaluation",
        "atomic_mode_0444_metric_sidecar",
        "control_branch",
        "terminal_publication",
        "seal_all_terminal_files_read_only",
    ]
    if lifecycle != {
        "immutable_order": expected_order,
        "reservation_consumes_attempt": True,
        "retry_resume_recovery_second_seed_or_extension": False,
        "source_review_may_open_generated_inputs": False,
        "v1_runtime_output_open_authorized": False,
        "failure_receipt_binds_reservation_and_partial_access_ledger": True,
        "runtime_open_attempt_and_outcome_fsync_required": True,
        "whole_tree_export_authorized": False,
    }:
        raise GuardFailure("V2 lifecycle is not exactly the two authorized mechanisms")

    execute = _definition(
        runner_definitions, "_execute_after_reservation", ast.FunctionDef
    )
    assert isinstance(execute, ast.FunctionDef)
    call_lines: dict[str, list[int]] = {}
    for node in ast.walk(execute):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name is not None:
                call_lines.setdefault(name, []).append(node.lineno)
    for name in (
        "_load_schedule_phase_a",
        "_camera_model_after_reservation",
        "RawInputs",
        "_finalize_schedule_train_identity",
    ):
        if name not in call_lines:
            raise GuardFailure(f"V2 execution omits required custody call: {name}")
    if not (
        min(call_lines["_load_schedule_phase_a"])
        < min(call_lines["_camera_model_after_reservation"])
        < min(call_lines["RawInputs"])
        < min(call_lines["_finalize_schedule_train_identity"])
    ):
        raise GuardFailure("schedule-first runtime custody order changed")

    ledger_class = _definition(
        runner_definitions, "PartialAccessLedger", ast.ClassDef
    )
    progress_class = _definition(
        runner_definitions, "OperationProgress", ast.ClassDef
    )
    assert isinstance(ledger_class, ast.ClassDef)
    assert isinstance(progress_class, ast.ClassDef)
    ledger_methods = _method_map(ledger_class)
    progress_methods = _method_map(progress_class)
    if not {
        "__init__",
        "_append",
        "_error",
        "_record_value",
        "append_terminal",
        "binding",
        "close",
        "read_regular",
        "runtime_opens",
    }.issubset(ledger_methods) or not {
        "enter",
        "increment",
        "location",
        "snapshot",
    }.issubset(progress_methods):
        raise GuardFailure("failure-receipt class surface changed")

    read_regular = ledger_methods["read_regular"]
    constants = [
        (node.value, node.lineno)
        for node in ast.walk(read_regular)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    attempted_lines = [line for value, line in constants if value == "OPEN_ATTEMPTED"]
    outcome_lines = [line for value, line in constants if value == "OPEN_OUTCOME"]
    runtime_os_open_lines = [
        node.lineno
        for node in ast.walk(read_regular)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "open"
        )
    ]
    if (
        len(attempted_lines) != 1
        or len(outcome_lines) != 2
        or len(runtime_os_open_lines) != 1
        or attempted_lines[0] >= runtime_os_open_lines[0]
        or min(outcome_lines) <= runtime_os_open_lines[0]
        or "V1_OUTPUT_ROOT_RELATIVE_PATH"
        not in {
            node.attr
            for node in ast.walk(read_regular)
            if isinstance(node, ast.Attribute)
        }
    ):
        raise GuardFailure("ledger does not receipt every open attempt and outcome")
    append_calls = _function_call_names(ledger_methods["_append"])
    if "fsync" not in append_calls or "write" not in append_calls:
        raise GuardFailure("partial-access records are not durably appended")

    terminal_failure = _definition(
        runner_definitions, "_terminal_failure", ast.FunctionDef
    )
    assert isinstance(terminal_failure, ast.FunctionDef)
    required_failure_keys = {
        "attempt_identity",
        "authority",
        "directories_including_root",
        "error",
        "failure_stage",
        "g2_navigation_or_heldout_attempted",
        "operation_counts",
        "partial_access_ledger",
        "published_prefix",
        "published_prefix_sha256",
        "reservation",
        "retry_authorized",
        "runtime_opens",
        "runtime_opens_sha256",
        "schema",
        "scientific_result",
        "scientific_result_status",
        "status",
    }
    if not required_failure_keys.issubset(_dict_constant_keys(terminal_failure)):
        raise GuardFailure("terminal failure receipt fields are incomplete")
    calls = _function_call_names(terminal_failure)
    if not {
        "_binding",
        "_seal_terminal",
        "binding",
        "runtime_opens",
        "snapshot",
    }.issubset(calls):
        raise GuardFailure("terminal failure omits a required immutable binding")
    reservation_bindings = [
        node
        for node in ast.walk(terminal_failure)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_binding"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "reservation.json"
        )
    ]
    if not reservation_bindings:
        raise GuardFailure("terminal failure does not directly bind reservation.json")

    contract_constants = _literal_assignments(contract_tree)
    contract_assignments = _assignment_map(contract_tree)
    expected_failure_schema = ast.parse(
        'FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v2"\n'
    ).body[0]
    if (
        contract_constants.get("SCHEMA_PREFIX")
        != "lewm_go2_shared_jepa_v5_multires_probe_v2"
        or _normalized_dump(contract_assignments.get("FAILURE_SCHEMA", ast.Pass()))
        != _normalized_dump(expected_failure_schema)
        or contract_constants.get("V1_OUTPUT_ROOT_RELATIVE_PATH") != V1_ROOT
        or "parse_partial_access_ledger" not in contract_definitions
        or "validate_failure_receipt" not in contract_definitions
    ):
        raise GuardFailure("V2 failure-receipt validation contract changed")
    return {
        "failure_receipt_direct_reservation_binding": True,
        "fsynced_open_attempt_and_outcome_ledger": True,
        "schedule_validation_precedes_n320_and_raw": True,
        "v1_runtime_output_open_authorized": False,
    }


def _function_argument_shape(function: ast.FunctionDef) -> dict[str, list[str]]:
    return {
        "positional": [
            argument.arg
            for argument in (*function.args.posonlyargs, *function.args.args)
        ],
        "keyword_only": [argument.arg for argument in function.args.kwonlyargs],
    }


def verify_schedule_adapter(root: Path = ROOT) -> dict[str, Any]:
    raw, tree = _parse_source(root, V2_SCHEDULE_ADAPTER)
    if _sha256(raw) != V2_SCHEDULE_ADAPTER_SHA256:
        raise GuardFailure("reviewed schedule adapter bytes changed")
    imported_modules: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imported_modules.update(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            imported_modules.add(statement.module or "")
    if not imported_modules.issubset({
        "__future__",
        "dataclasses",
        "hashlib",
        "typing",
        "lewm.benchmarks",
    }):
        raise GuardFailure(
            f"schedule adapter import surface changed: {sorted(imported_modules)}"
        )
    definitions = _definition_map(tree)
    required = {
        "validate_bound_schedule_phase_a",
        "finalize_train_identity",
    }
    if not required.issubset(definitions):
        raise GuardFailure("the two-stage schedule adapter API is incomplete")
    phase_a = definitions["validate_bound_schedule_phase_a"]
    phase_b = definitions["finalize_train_identity"]
    if not isinstance(phase_a, ast.FunctionDef) or not isinstance(
        phase_b, ast.FunctionDef
    ):
        raise GuardFailure("schedule adapter API must use ordinary functions")
    if _function_argument_shape(phase_a) != {
        "positional": [],
        "keyword_only": ["raw", "binding"],
    } or _function_argument_shape(phase_b) != {
        "positional": [],
        "keyword_only": ["state", "ordered_train_pair_ids"],
    }:
        raise GuardFailure("schedule adapter signature changed")
    all_calls = _function_call_names(tree)
    forbidden_calls = {
        "__import__",
        "exec",
        "eval",
        "load",
        "manual_seed",
        "open",
        "permutation",
        "random",
        "randperm",
        "read_bytes",
        "read_text",
        "run",
        "seed",
        "shuffle",
        "system",
    }
    if all_calls & forbidden_calls:
        raise GuardFailure(f"schedule adapter performs I/O: {sorted(all_calls & forbidden_calls)}")
    constants = _literal_assignments(tree)
    if (
        constants.get("BOUND_SCHEDULE_PATH")
        != (
            ".generated/go2_shared_observable_camera_ray_jepa_v5/"
            "matched_training_v4/schedule.json"
        )
        or constants.get("BOUND_SCHEDULE_FILE_SHA256")
        != "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270"
        or constants.get("BOUND_SCHEDULE_CONTENT_SHA256")
        != "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15"
        or constants.get("BOUND_SCHEDULE_BYTE_COUNT") != 607_373
        or constants.get("NORMALIZED_V1_SCHEDULE_CONTENT_SHA256")
        != "893c48b2c2c591dbc90469e5a19a74e70bd54f96689b63881c216605255c0e5d"
        or constants.get("V1_SCIENCE_CONTRACT_SHA256") != SCIENCE_SHA256
        or constants.get("USED_PRESENTATIONS") != 16_000
    ):
        raise GuardFailure("normalized schedule adapter identity changed")
    return {
        "adapter_file_sha256": _sha256(raw),
        "functions": sorted(required),
        "normalized_schedule_content_sha256":
            constants["NORMALIZED_V1_SCHEDULE_CONTENT_SHA256"],
    }


# Only these existing definitions may differ after version-string
# normalization.  Every name is tied to schedule-adapter integration or a
# durable failure/access receipt.  The final implementation test freezes this
# set and fails on any third changed definition.
AUTHORIZED_CHANGED_DEFINITIONS = {
    V2_CONTRACT: frozenset({
        "lifecycle_contract",
        "safe_relative_path",
        "validate_source_manifest",
    }),
    V2_RUNNER: frozenset({
        "_execute_after_reservation",
        "_load_post_reservation_stack",
        "_rehash_deferred_runtime_and_authority",
        "_terminal_failure",
        "_train",
    }),
    V2_LAUNCHER: frozenset(),
    V2_TEST: frozenset(),
    V2_CLOSURE_CHECKER: frozenset(),
    V2_CLOSURE_TEST: frozenset({
        "test_recursive_manifest_matches_every_discovered_source_byte",
    }),
}
AUTHORIZED_ADDED_DEFINITIONS = {
    V2_CONTRACT: frozenset({
        "empty_partial_operation_counts",
        "parse_partial_access_ledger",
        "validate_failure_receipt",
        "validate_partial_operation_counts",
    }),
    V2_RUNNER: frozenset({
        "OperationProgress",
        "PartialAccessLedger",
        "_failure_boundary",
        "_finalize_schedule_train_identity",
        "_install_ledgered_matched_reader",
        "_load_schedule_phase_a",
        "_runtime_binding_index",
        "_runtime_kind",
        "_terminal_file_bindings",
    }),
    V2_LAUNCHER: frozenset(),
    V2_TEST: frozenset({
        "_first_backward_counts",
        "_first_evaluation_counts",
        "_full_counts",
        "_ledger",
        "_read",
        "_reservation",
        "_write_input",
        "test_every_required_boundary_publishes_complete_sealed_failure",
        "test_injected_open_failure_is_durably_paired_before_rethrow",
        "test_launcher_still_orders_authority_preflight_then_immediate_exec",
        "test_ledger_denies_any_v1_attempt_output_access",
        "test_ledger_rejects_symlinked_parent_components",
        "test_partial_access_ledger_is_hash_chained_and_pairs_every_open",
        "test_partial_operation_counts_are_exact_and_capped",
        "test_partial_read_failure_records_exact_bytes_before_rethrow",
        "test_safe_relative_path_rejects_noncanonical_spellings",
        "test_schedule_adapter_calls_are_in_the_required_custody_order",
        "test_v1_implementation_bytes_remain_frozen",
        "test_v2_identity_and_fresh_operational_envelope",
        "test_v2_source_imports_do_not_import_torch",
    }),
    V2_CLOSURE_CHECKER: frozenset(),
    V2_CLOSURE_TEST: frozenset(),
}
AUTHORIZED_REMOVED_DEFINITIONS = {
    V2_CONTRACT: frozenset(),
    V2_RUNNER: frozenset({"_load_schedule"}),
    V2_LAUNCHER: frozenset(),
    V2_TEST: frozenset({
        "_binding",
        "_evaluation",
        "_physical_metrics",
        "_runtime_inputs",
        "test_contract_binds_prereg_sources_and_deferred_runtime",
        "test_exact_schedule_and_operation_caps",
        "test_launcher_preflight_is_immediately_followed_by_exec",
        "test_physical_evaluator_retains_nine_scopes_and_189_margins",
        "test_post_mkdir_reservation_failure_is_terminal_and_sealed",
        "test_readonly_sidecar_and_terminal_sealing",
        "test_recursive_local_import_closure_is_inside_exact_source_set",
        "test_reservation_is_mode_0700_and_consumes_attempt",
        "test_review_and_authorization_are_exact_and_independent",
        "test_run_parent_orders_authority_preflight_reservation_then_execution",
        "test_runner_requires_complete_production_migration_receipt",
        "test_snapshot_evaluation_sidecar_control_source_order",
        "test_source_imports_do_not_import_torch_or_open_payloads",
        "test_update_1000_requires_every_strict_conjunct",
        "test_updates_100_and_400_are_integrity_only",
    }),
    V2_CLOSURE_CHECKER: frozenset(),
    V2_CLOSURE_TEST: frozenset(),
}
PAIRED_SOURCES = {
    V2_CONTRACT: V1_CONTRACT,
    V2_RUNNER: V1_RUNNER,
    V2_LAUNCHER: V1_LAUNCHER,
    V2_TEST: V1_TEST,
    V2_CLOSURE_CHECKER: V1_CLOSURE_CHECKER,
    V2_CLOSURE_TEST: V1_CLOSURE_TEST,
}


def verify_delta_surface(root: Path = ROOT) -> dict[str, Any]:
    observed: dict[str, dict[str, list[str]]] = {}
    for v2_relative, v1_relative in PAIRED_SOURCES.items():
        _, before = _parse_source(root, v1_relative)
        _, after = _parse_source(root, v2_relative)
        added, removed, changed = _definition_delta(before, after)
        expected = (
            set(AUTHORIZED_ADDED_DEFINITIONS[v2_relative]),
            set(AUTHORIZED_REMOVED_DEFINITIONS[v2_relative]),
            set(AUTHORIZED_CHANGED_DEFINITIONS[v2_relative]),
        )
        if (added, removed, changed) != expected:
            raise GuardFailure(
                f"unauthorized V1->V2 definition delta in {v2_relative}: "
                f"added={sorted(added)}, removed={sorted(removed)}, "
                f"changed={sorted(changed)}"
            )
        observed[v2_relative] = {
            "added": sorted(added),
            "changed": sorted(changed),
            "removed": sorted(removed),
        }
    # Science-bearing runner definitions must remain byte-semantic AST
    # identities after version-only string normalization.
    _, v1_runner = _parse_source(root, V1_RUNNER)
    _, v2_runner = _parse_source(root, V2_RUNNER)
    before = _definition_map(v1_runner)
    after = _definition_map(v2_runner)
    science_bearing = {
        "_evaluate",
        "_prepare_model",
        "_publish_metric_sidecar",
        "_publish_training_records",
        "_snapshot",
        "_validate_migration_receipt",
    }
    for name in science_bearing:
        if (
            name not in before
            or name not in after
            or _normalized_dump(before[name]) != _normalized_dump(after[name])
        ):
            raise GuardFailure(f"science-bearing runner definition changed: {name}")
    if (
        "_train" not in before
        or "_train" not in after
        or _normalized_training_dump(before["_train"])
        != _normalized_training_dump(after["_train"])
    ):
        raise GuardFailure(
            "training computation changed beyond exact operation-progress receipts"
        )
    return {
        "authorized_operational_delta_ids": [
            "bound_schedule_schema_adapter",
            "complete_failure_receipts",
        ],
        "paired_source_deltas": observed,
        "science_bearing_runner_definition_count": len(science_bearing) + 1,
    }


def verify_all(root: Path = ROOT) -> dict[str, Any]:
    return {
        "authority": verify_preregistration_authority(root),
        "delta_surface": verify_delta_surface(root),
        "generated_runtime_checkpoint_or_gpu_open_count": 0,
        "model_and_roots": verify_model_and_roots(root),
        "operational_mechanisms": verify_operational_mechanisms(root),
        "schedule_adapter": verify_schedule_adapter(root),
        "science_identity": verify_science_identity(root),
        "v1_frozen_sources": verify_v1_frozen_sources(root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the source-only verification report as canonical JSON",
    )
    arguments = parser.parse_args(argv)
    report = verify_all()
    if arguments.json:
        print(_canonical_bytes(report).decode("ascii"))
    else:
        print(
            "PASS V2 science identity and two-delta source guard "
            f"(science={SCIENCE_SHA256}, frozen_v1=7, generated_opens=0)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
