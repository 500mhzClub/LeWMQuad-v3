#!/usr/bin/env python3
"""Fail-closed, source-only V2 -> V3 science-identity and delta guard.

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
V2_ROOT = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v2"
)
V3_ROOT = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_multiresolution_perception_probe_v3"
)
MODEL_PATH = "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py"
MODEL_SHA256 = (
    "a63da1137539953b2f40d184def1652ae05f63d7b434084b1a91787e1fc83d0b"
)
MODEL_FAMILY = "shared_observable_camera_ray_jepa_v5_multires_v1"
MODEL_RUNTIME_VERSION = (
    "lewm_go2_rgb_multiresolution_perception_v1_model_runtime_v1"
)

V2_CONTRACT = "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2.py"
V3_CONTRACT = "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py"
V2_RUNNER = "scripts/run_go2_shared_jepa_v5_multires_probe_v2.py"
V3_RUNNER = "scripts/run_go2_shared_jepa_v5_multires_probe_v3.py"
V2_LAUNCHER = "scripts/launch_go2_shared_jepa_v5_multires_probe_v2.py"
V3_LAUNCHER = "scripts/launch_go2_shared_jepa_v5_multires_probe_v3.py"
V2_TEST = "lewm/tests/test_go2_shared_jepa_v5_multires_probe_v2.py"
V3_TEST = "lewm/tests/test_go2_shared_jepa_v5_multires_probe_v3.py"
V3_BOUNDARY_TEST = (
    "lewm/tests/test_go2_shared_jepa_v5_multires_probe_v3_receipt_boundary.py"
)
V2_CLOSURE_CHECKER = "scripts/check_go2_multires_probe_source_closure_v2.py"
V3_CLOSURE_CHECKER = "scripts/check_go2_multires_probe_source_closure_v3.py"
V2_CLOSURE_TEST = "lewm/tests/test_go2_multires_probe_source_closure_v2.py"
V3_CLOSURE_TEST = "lewm/tests/test_go2_multires_probe_source_closure_v3.py"
V2_IDENTITY_CHECKER = "scripts/check_go2_multires_probe_v2_science_identity.py"
V2_IDENTITY_TEST = "lewm/tests/test_go2_multires_probe_v2_science_identity.py"
V2_SCHEDULE_TEST = (
    "lewm/tests/test_go2_shared_jepa_v5_multires_probe_v2_schedule.py"
)
MODEL_TEST = "lewm/tests/test_shared_observable_camera_ray_jepa_v5_multires_v1.py"
V2_SCHEDULE_ADAPTER = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py"
)
V2_SCHEDULE_ADAPTER_SHA256 = (
    "a8efe19da92c9c2107f11be38db8ed80e66aedca3ef41af0428ab13d50f56bd1"
)

V2_AUTHORIZATION = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "execution_authorization_2026-07-24.json"
)
V2_TERMINAL_AUDIT = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "terminal_audit_2026-07-24.json"
)
V2_TERMINAL_ROOT_CAUSE = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "terminal_root_cause_2026-07-24.json"
)
V2_TERMINAL_REVIEW = (
    "docs/lewm_go2_rgb_multiresolution_perception_v2_"
    "terminal_independent_review_2026-07-24.json"
)
PRIOR_STRICT_AUDIT = (
    "docs/lewm_go2_shared_jepa_v5_matched_training_v3_"
    "terminal_failure_audit_2026-07-15.json"
)
V3_DECISION = (
    "docs/lewm_go2_rgb_multiresolution_perception_v3_"
    "operational_recovery_decision_2026-07-24.md"
)
V3_PREREGISTRATION = (
    "docs/lewm_go2_rgb_multiresolution_perception_v3_"
    "preregistration_2026-07-24.json"
)
V3_PREREGISTRATION_REVIEW = (
    "docs/lewm_go2_rgb_multiresolution_perception_v3_"
    "preregistration_independent_review_2026-07-24.json"
)
COMPAT_CONTRACT = (
    "lewm/benchmarks/go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
COMPAT_RUNNER = (
    "scripts/run_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
COMPAT_LAUNCHER = (
    "scripts/launch_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
COMPAT_TEST = (
    "lewm/tests/test_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)

# These V2 implementation and verification files are frozen before V3 work.
V2_FROZEN_SOURCE_SHA256 = {
    V2_CONTRACT:
        "53e045a208a39705e12537a698c20d6d1c4508cc13145ebdb04cd66f494ad1fd",
    V2_RUNNER:
        "5fdec79263e904b41b279eb1560b60ab2f9a89384fd032b31330d68b9d003c45",
    V2_LAUNCHER:
        "d721334113a9c580dc2db4a3444c80ab3f9b08d268b56090a95236f33a947296",
    V2_TEST:
        "a49050ffe3f46ff12c6901894fede47c4e5159c84f06b66fc8dce6ae75d8000c",
    V2_CLOSURE_CHECKER:
        "c5010ba4dec12c1d23d1c158ccdd35f20c0dc6e3fab0b39916912f2790866b79",
    V2_CLOSURE_TEST:
        "720f6c42f41bc350a0854c7276a875499c7516b705fcc179f535a690fa66a431",
    V2_IDENTITY_CHECKER:
        "e87402f57abffa70340161fc54c2285d624747933d5a12d4fbed1b4422acab6e",
    V2_IDENTITY_TEST:
        "5211f7bdd77a018a42ad920aa47ebfc9ac63c0b0036665e5e93c80489a5792d8",
    V2_SCHEDULE_ADAPTER: V2_SCHEDULE_ADAPTER_SHA256,
    V2_SCHEDULE_TEST:
        "340828cb55a03da575ccfb8242ff3e3db8b8f15527d43891b737cfad8a5b2204",
    MODEL_PATH: MODEL_SHA256,
    MODEL_TEST:
        "a241910c83bc44cf15b56270659becf1def66f358f3f2bb1a89d89a9bce30fae",
}

AUTHORITY_FILE_BINDINGS = {
    V2_AUTHORIZATION: {
        "byte_count": 8_194,
        "content_sha256":
            "fe1f75a9f184d37450d39c9d5f20e97acc23cb8fefcba639f84a5f4c30f7455c",
        "file_sha256":
            "5e1ac702b7a17e0dc05a40a67b4c7700870abe8f5f772d01b609f0388607f0fb",
    },
    V2_TERMINAL_AUDIT: {
        "byte_count": 8_841,
        "content_sha256":
            "872a942e4a16b112d934ddc5063289d963ee1873e43e8adb4242d66065da11f7",
        "file_sha256":
            "de43da71bf92526d4d8314bbdd1c186af608aa73eb03b4246dffa16cdeace90b",
    },
    V2_TERMINAL_ROOT_CAUSE: {
        "byte_count": 9_414,
        "content_sha256":
            "58fe292f810f38549f10e7c46fdf83715cdc66b64b4b8c82977a2161e77af434",
        "file_sha256":
            "feb88c8fe42ca02da4a10669755539f7a1aa1560b3bc54355b1e7b9cce2c1883",
    },
    V2_TERMINAL_REVIEW: {
        "byte_count": 7_956,
        "content_sha256":
            "d3c342167f9ae73bae20f9a9f049b09c6cb79777dd649d123afb0221dd15798d",
        "file_sha256":
            "7c19df3af038a80d510dc6247dea69d6a32091352dcf056b24ceb903c8bd0d7c",
    },
    PRIOR_STRICT_AUDIT: {
        "byte_count": 12_883,
        "content_sha256":
            "b93146f00c79a6b2d151a07fb33696c673a1d45677ee6b948e20acadef9c9899",
        "file_sha256":
            "2f94d6ddaf076bc011eaac46408261aea3b8ac030386c9d2185463fe87a08e4a",
    },
    V3_DECISION: {
        "byte_count": 12_030,
        "file_sha256":
            "94ab2ca50cdc5c33008a411aafc07461684d8564433a9fd787f68308db04b6a2",
    },
    V3_PREREGISTRATION: {
        "byte_count": 12_423,
        "content_sha256":
            "64da13d6e38a8c1ee2a1bc87b9917611097023a36939ee4305be9a4e85f602b7",
        "file_sha256":
            "a8a5d870382ad505edd907f96dfae8a6ed737caf7ff424d2b52f8e4bc020e5d5",
    },
    V3_PREREGISTRATION_REVIEW: {
        "byte_count": 9_850,
        "content_sha256":
            "ca42b49c9360946dd5ab5ad29e488a7354ea55f788bc566f528520256bf8aa23",
        "file_sha256":
            "0214656acf8bc60e2c355e21824b1764bcddd9bb1643a8b8efee3c8ddbb8e1bf",
    },
}
V3_PREREGISTRATION_COMMIT = "7e6e539370c8f9d9d228da5ef4bc9ea4d10569a2"

READABLE_SOURCE_PATHS = frozenset({
    *V2_FROZEN_SOURCE_SHA256,
    V3_CONTRACT,
    V3_RUNNER,
    V3_LAUNCHER,
    V3_TEST,
    V3_CLOSURE_CHECKER,
    V3_CLOSURE_TEST,
    V3_BOUNDARY_TEST,
    COMPAT_CONTRACT,
    COMPAT_RUNNER,
    COMPAT_LAUNCHER,
    COMPAT_TEST,
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


def verify_v2_frozen_sources(root: Path = ROOT) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in V2_FROZEN_SOURCE_SHA256.items():
        digest = _sha256(_read_regular_source(root, relative))
        if digest != expected:
            raise GuardFailure(f"frozen V2 source changed: {relative}")
        observed[relative] = digest
    if len(observed) != 12:
        raise GuardFailure("the exact twelve-file V2 freeze changed")
    return observed


def verify_science_identity(root: Path = ROOT) -> dict[str, Any]:
    _, authorization = _parse_canonical_json(
        root,
        V2_AUTHORIZATION,
        expected=AUTHORITY_FILE_BINDINGS[V2_AUTHORIZATION],
    )
    _, preregistration = _parse_canonical_json(
        root,
        V3_PREREGISTRATION,
        expected=AUTHORITY_FILE_BINDINGS[V3_PREREGISTRATION],
    )
    _, v2_tree = _parse_source(root, V2_CONTRACT)
    _, v3_tree = _parse_source(root, V3_CONTRACT)
    v2_source_science = _evaluate_science_contract(v2_tree)
    v3_source_science = _evaluate_science_contract(v3_tree)
    v2_authorized_science = authorization.get("experiment")
    values = (
        v2_source_science,
        v3_source_science,
        v2_authorized_science,
    )
    if any(type(value) is not dict for value in values):
        raise GuardFailure("one science identity source is not a plain object")
    if not all(value == v2_source_science for value in values[1:]):
        raise GuardFailure("V3 science_contract() is not deeply equal to V2")
    digests = tuple(_sha256(_canonical_bytes(value)) for value in values)
    if digests != (SCIENCE_SHA256,) * len(values):
        raise GuardFailure(f"science identity digest changed: {digests}")
    science_identity = preregistration.get("science_identity")
    if (
        type(science_identity) is not dict
        or science_identity.get("deep_equal_to_v2_experiment_required") is not True
        or science_identity.get("exact_science_contract_sha256")
        != SCIENCE_SHA256
        or science_identity.get("science_delta_count") != 0
    ):
        raise GuardFailure("V3 preregistered science identity changed")
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
    decision_raw = _read_regular_source(root, V3_DECISION)
    decision_expected = AUTHORITY_FILE_BINDINGS[V3_DECISION]
    if (
        len(decision_raw) != decision_expected["byte_count"]
        or _sha256(decision_raw) != decision_expected["file_sha256"]
    ):
        raise GuardFailure("V3 recovery decision binding changed")
    _, preregistration = _parse_canonical_json(
        root,
        V3_PREREGISTRATION,
        expected=AUTHORITY_FILE_BINDINGS[V3_PREREGISTRATION],
    )
    _, review = _parse_canonical_json(
        root,
        V3_PREREGISTRATION_REVIEW,
        expected=AUTHORITY_FILE_BINDINGS[V3_PREREGISTRATION_REVIEW],
    )
    historical: dict[str, dict[str, Any]] = {}
    for relative in (
        V2_TERMINAL_AUDIT,
        V2_TERMINAL_ROOT_CAUSE,
        V2_TERMINAL_REVIEW,
        PRIOR_STRICT_AUDIT,
    ):
        _, historical[relative] = _parse_canonical_json(
            root,
            relative,
            expected=AUTHORITY_FILE_BINDINGS[relative],
        )
    decision_binding = preregistration.get("decision_binding")
    if decision_binding != {
        "byte_count": decision_expected["byte_count"],
        "file_sha256": decision_expected["file_sha256"],
        "path": V3_DECISION,
    }:
        raise GuardFailure("preregistration decision binding changed")
    if (
        preregistration.get("schema")
        != "lewm_go2_rgb_multiresolution_perception_v3_preregistration_v1"
        or preregistration.get("status")
        != (
            "PREREGISTERED_SOURCE_ONLY_SCIENCE_IDENTICAL_V3_PENDING_"
            "INDEPENDENT_REVIEW_NO_PROBE_EXECUTION_AUTHORITY"
        )
        or preregistration.get("fresh_attempt", {}).get("output_root") != V3_ROOT
        or preregistration.get("fresh_attempt", {}).get(
            "v1_and_v2_runtime_outputs_must_remain_frozen"
        )
        is not True
        or preregistration.get("fresh_attempt", {}).get("execution_authorized")
        is not False
        or preregistration.get("fresh_attempt", {}).get("reservation_authorized")
        is not False
        or review.get("status")
        != (
            "PASS_SOURCE_ONLY_SCIENCE_IDENTICAL_V3_RECOVERY_"
            "PREREGISTRATION_WITH_CONDITIONAL_SYNTHETIC_GATE_AUTHORITY"
        )
        or review.get("recovery_decision_binding") != {
            "byte_count": decision_expected["byte_count"],
            "file_sha256": decision_expected["file_sha256"],
            "path": V3_DECISION,
        }
        or review.get("preregistration_binding") != {
            **_binding_without_path(
                AUTHORITY_FILE_BINDINGS[V3_PREREGISTRATION]
            ),
            "path": V3_PREREGISTRATION,
        }
        or review.get("frozen_science_identity", {}).get(
            "exact_science_contract_sha256"
        )
        != SCIENCE_SHA256
        or review.get("frozen_science_identity", {}).get(
            "science_delta_count"
        )
        != 0
        or review.get("delta_surface", {}).get(
            "authorized_lifecycle_delta_count"
        )
        != 2
        or review.get("delta_surface", {}).get(
            "authorized_synthetic_checker_delta_count"
        )
        != 1
        or review.get("delta_surface", {}).get("no_fourth_delta_authorized")
        is not True
        or review.get("authority", {}).get("v3_probe_execution_authorized")
        is not False
        or historical[V2_TERMINAL_AUDIT].get("status")
        != (
            "PASS_COMPLETE_SEALED_TERMINAL_FAILURE_NO_SCIENTIFIC_RESULT_"
            "NO_RETRY"
        )
        or historical[V2_TERMINAL_ROOT_CAUSE].get("status")
        != (
            "CONFIRMED_SOURCE_ONLY_TUPLE_LIST_RECEIPT_NORMALIZATION_"
            "ROOT_CAUSE_TERMINAL_NO_RETRY"
        )
        or historical[V2_TERMINAL_REVIEW].get("status")
        != (
            "PASS_INDEPENDENT_SOURCE_AND_CUSTODY_REVIEW_CONFIRMS_"
            "LIFECYCLE_ONLY_TUPLE_LIST_FAILURE_ZERO_TRAINING_ATTEMPT_"
            "CONSUMED_NO_RETRY"
        )
        or historical[PRIOR_STRICT_AUDIT].get("verdict")
        != (
            "PASS_CONFIRMED_FIRST_B4_FORWARD_LOSS_THEN_ROCM_DETERMINISM_"
            "INFRASTRUCTURE_FAILURE_ZERO_LEARNED_UPDATE"
        )
    ):
        raise GuardFailure("V3 preregistration or independent-review authority changed")
    _, v3_tree = _parse_source(root, V3_CONTRACT)
    constants = _literal_assignments(v3_tree)
    expected_constants = {
        "PREREGISTRATION_COMMIT": V3_PREREGISTRATION_COMMIT,
        "PREREGISTRATION_CONTENT_SHA256":
            AUTHORITY_FILE_BINDINGS[V3_PREREGISTRATION]["content_sha256"],
        "PREREGISTRATION_FILE_SHA256":
            AUTHORITY_FILE_BINDINGS[V3_PREREGISTRATION]["file_sha256"],
        "PREREGISTRATION_RELATIVE_PATH": V3_PREREGISTRATION,
        "PREREGISTRATION_REVIEW_FILE_SHA256":
            AUTHORITY_FILE_BINDINGS[V3_PREREGISTRATION_REVIEW]["file_sha256"],
        "PREREGISTRATION_REVIEW_RELATIVE_PATH": V3_PREREGISTRATION_REVIEW,
        "RECOVERY_DECISION_FILE_SHA256": decision_expected["file_sha256"],
        "RECOVERY_DECISION_RELATIVE_PATH": V3_DECISION,
    }
    for name, expected in expected_constants.items():
        if constants.get(name) != expected:
            raise GuardFailure(f"V3 contract authority constant changed: {name}")
    return {
        "decision_file_sha256": decision_expected["file_sha256"],
        "preregistration_content_sha256":
            AUTHORITY_FILE_BINDINGS[V3_PREREGISTRATION]["content_sha256"],
        "preregistration_review_content_sha256":
            AUTHORITY_FILE_BINDINGS[V3_PREREGISTRATION_REVIEW]["content_sha256"],
    }


def verify_model_and_roots(root: Path = ROOT) -> dict[str, Any]:
    _, v2_tree = _parse_source(root, V2_CONTRACT)
    _, v3_tree = _parse_source(root, V3_CONTRACT)
    _, model_tree = _parse_source(root, MODEL_PATH)
    v2 = _literal_assignments(v2_tree)
    v3 = _literal_assignments(v3_tree)
    model = _literal_assignments(model_tree)
    for name, expected in {
        "MODEL_FILE_SHA256": MODEL_SHA256,
        "MODEL_RELATIVE_PATH": MODEL_PATH,
        "MODEL_RUNTIME_VERSION": MODEL_RUNTIME_VERSION,
    }.items():
        if v2.get(name) != expected or v3.get(name) != expected:
            raise GuardFailure(f"model/runtime identity changed: {name}")
    if model.get("MODEL_FAMILY") != MODEL_FAMILY:
        raise GuardFailure("model-family source identity changed")
    if v2.get("OUTPUT_ROOT_RELATIVE_PATH") != V2_ROOT:
        raise GuardFailure("V2 output-root identity changed")
    if v3.get("OUTPUT_ROOT_RELATIVE_PATH") != V3_ROOT or V3_ROOT == V2_ROOT:
        raise GuardFailure("V3 output root is not exactly distinct")
    versioned_envelope_literals = {
        "AUTHORIZATION_RELATIVE_PATH",
        "CONTRACT_RELATIVE_PATH",
        "IMPLEMENTATION_AUTHOR",
        "LAUNCHER_RELATIVE_PATH",
        "OUTPUT_ROOT_RELATIVE_PATH",
        "PREREGISTRATION_COMMIT",
        "PREREGISTRATION_CONTENT_SHA256",
        "PREREGISTRATION_FILE_SHA256",
        "PREREGISTRATION_RELATIVE_PATH",
        "PREREGISTRATION_REVIEW_FILE_SHA256",
        "PREREGISTRATION_REVIEW_RELATIVE_PATH",
        "RECOVERY_DECISION_FILE_SHA256",
        "RECOVERY_DECISION_RELATIVE_PATH",
        "REVIEW_RELATIVE_PATH",
        "RUNNER_RELATIVE_PATH",
        "SCHEMA_PREFIX",
        "SCIENCE_IDENTITY_CHECKER_RELATIVE_PATH",
        "SCIENCE_IDENTITY_TEST_RELATIVE_PATH",
        "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH",
        "SOURCE_CLOSURE_TEST_RELATIVE_PATH",
        "SOURCE_MANIFEST_RELATIVE_PATH",
        "TEST_RELATIVE_PATH",
    }
    changed_common_literals = {
        name
        for name in set(v2) & set(v3)
        if v2[name] != v3[name]
    }
    if changed_common_literals != versioned_envelope_literals:
        raise GuardFailure(
            "literal runtime/science configuration changed outside the "
            f"versioned envelope: {sorted(changed_common_literals)}"
        )
    return {
        "unchanged_common_literal_binding_count":
            len((set(v2) & set(v3)) - changed_common_literals),
        "model_file_sha256": MODEL_SHA256,
        "model_family": MODEL_FAMILY,
        "model_runtime_version": MODEL_RUNTIME_VERSION,
        "v2_output_root": V2_ROOT,
        "v3_output_root": V3_ROOT,
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
        ("go2_shared_jepa_v5_multires_probe_v3", "go2_shared_jepa_v5_multires_probe_v2"),
        ("rgb_multiresolution_perception_probe_v3", "rgb_multiresolution_perception_probe_v2"),
        ("rgb_multiresolution_perception_v3", "rgb_multiresolution_perception_v2"),
        ("source_closure_v3.py", "source_closure_v2.py"),
        ("V3", "V2"),
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
    _, contract_tree = _parse_source(root, V3_CONTRACT)
    _, runner_tree = _parse_source(root, V3_RUNNER)
    contract_definitions = _definition_map(contract_tree)
    runner_definitions = _definition_map(runner_tree)

    lifecycle = _literal_function_return(contract_definitions, "lifecycle_contract")
    expected_order = [
        "exact_source_review_and_authorization_rehash",
        "validate_isolated_no_tensor_hardware_preflight",
        "reserve_unique_mode_0700_output_root",
        "protect_post_reservation_pre_ledger_terminalization",
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
        "v2_runtime_output_open_authorized": False,
        "pre_ledger_failure_receipt_binds_reservation_and_exact_prefix": True,
        "pre_ledger_failure_receipt_never_claims_complete_ledger": True,
        "failure_receipt_binds_reservation_and_partial_access_ledger": True,
        "runtime_open_attempt_and_outcome_fsync_required": True,
        "whole_tree_export_authorized": False,
    }:
        raise GuardFailure("V3 lifecycle is not exactly the authorized recovery")

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
        "_initialize_partial_access_ledger",
        "_load_post_reservation_stack",
    ):
        if name not in call_lines:
            raise GuardFailure(f"V3 execution omits required custody call: {name}")
    if not (
        min(call_lines["_initialize_partial_access_ledger"])
        < min(call_lines["_load_post_reservation_stack"])
        < min(call_lines["_load_schedule_phase_a"])
        and
        min(call_lines["_load_schedule_phase_a"])
        < min(call_lines["_camera_model_after_reservation"])
        < min(call_lines["RawInputs"])
        < min(call_lines["_finalize_schedule_train_identity"])
    ):
        raise GuardFailure("schedule-first runtime custody order changed")

    pre_ledger_handlers = [
        node
        for node in ast.walk(execute)
        if (
            isinstance(node, ast.ExceptHandler)
            and isinstance(node.type, ast.Name)
            and node.type.id == "_PreLedgerInitializationError"
        )
    ]
    if len(pre_ledger_handlers) != 1:
        raise GuardFailure("pre-ledger initialization is not caught exactly once")
    pre_ledger_handler = pre_ledger_handlers[0]
    if (
        "_terminal_pre_ledger_failure"
        not in _function_call_names(pre_ledger_handler)
        or not any(
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Attribute)
            and isinstance(node.exc.value, ast.Name)
            and node.exc.value.id == "failure"
            and node.exc.attr == "error"
            for node in ast.walk(pre_ledger_handler)
        )
    ):
        raise GuardFailure("pre-ledger failure is not terminalized then re-raised")

    receipt_dict = _definition(
        runner_definitions, "_receipt_dict", ast.FunctionDef
    )
    assert isinstance(receipt_dict, ast.FunctionDef)
    receipt_calls: dict[str, list[int]] = {}
    for node in ast.walk(receipt_dict):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name is not None:
                receipt_calls.setdefault(name, []).append(node.lineno)
    if not all(
        name in receipt_calls
        for name in ("getattr", "callable", "is_dataclass", "asdict")
    ) or not (
        min(receipt_calls["getattr"])
        < min(receipt_calls["callable"])
        < min(receipt_calls["is_dataclass"])
        < min(receipt_calls["asdict"])
    ) or "to_dict" not in {
        node.value
        for node in ast.walk(receipt_dict)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }:
        raise GuardFailure("receipt normalization does not prefer canonical to_dict")

    initialize = _definition(
        runner_definitions, "_initialize_partial_access_ledger", ast.FunctionDef
    )
    assert isinstance(initialize, ast.FunctionDef)
    initialize_calls: dict[str, list[int]] = {}
    for node in ast.walk(initialize):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name is not None:
                initialize_calls.setdefault(name, []).append(node.lineno)
    boundaries = {
        node.value
        for node in ast.walk(initialize)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if (
        not {
            "ledger_before_header",
            "ledger_after_durable_header",
            "before_header_publication",
            "after_durable_header_before_constructor_acceptance",
        }.issubset(boundaries)
        or len(initialize_calls.get("_failure_boundary", [])) != 2
        or len(initialize_calls.get("_write_exclusive", [])) != 1
        or len(initialize_calls.get("PartialAccessLedger", [])) != 1
        or not (
            min(initialize_calls["_failure_boundary"])
            < min(initialize_calls["_write_exclusive"])
            < max(initialize_calls["_failure_boundary"])
            < min(initialize_calls["PartialAccessLedger"])
        )
    ):
        raise GuardFailure("pre-ledger boundary protection changed")

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
        or not {
            "V1_OUTPUT_ROOT_RELATIVE_PATH",
            "V2_OUTPUT_ROOT_RELATIVE_PATH",
        }.issubset({
            node.attr
            for node in ast.walk(read_regular)
            if isinstance(node, ast.Attribute)
        })
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
        "v1_runtime_output_open_count",
        "v2_runtime_output_open_count",
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

    pre_ledger_failure = _definition(
        runner_definitions, "_terminal_pre_ledger_failure", ast.FunctionDef
    )
    assert isinstance(pre_ledger_failure, ast.FunctionDef)
    required_pre_ledger_keys = {
        "attempt_identity",
        "authority",
        "directories_including_root",
        "error",
        "failure_stage",
        "g2_navigation_or_heldout_attempted",
        "ledger_state",
        "operation_counts",
        "published_prefix",
        "published_prefix_sha256",
        "reservation",
        "retry_authorized",
        "schema",
        "scientific_result",
        "scientific_result_status",
        "status",
        "v1_runtime_output_open_count",
        "v2_runtime_output_open_count",
    }
    if not required_pre_ledger_keys.issubset(
        _dict_constant_keys(pre_ledger_failure)
    ) or not {
        "_binding",
        "_publish_readonly_atomic",
        "_seal_terminal",
        "validate_pre_ledger_failure_receipt",
        "validate_pre_ledger_header",
    }.issubset(_function_call_names(pre_ledger_failure)):
        raise GuardFailure("pre-ledger terminal receipt is incomplete")

    contract_constants = _literal_assignments(contract_tree)
    contract_assignments = _assignment_map(contract_tree)
    expected_failure_schema = ast.parse(
        'FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v3"\n'
    ).body[0]
    expected_pre_ledger_schema = ast.parse(
        'PRE_LEDGER_FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_pre_ledger_failure_v1"\n'
    ).body[0]
    if (
        contract_constants.get("SCHEMA_PREFIX")
        != "lewm_go2_shared_jepa_v5_multires_probe_v3"
        or _normalized_dump(contract_assignments.get("FAILURE_SCHEMA", ast.Pass()))
        != _normalized_dump(expected_failure_schema)
        or _normalized_dump(
            contract_assignments.get("PRE_LEDGER_FAILURE_SCHEMA", ast.Pass())
        )
        != _normalized_dump(expected_pre_ledger_schema)
        or contract_constants.get("OUTPUT_ROOT_RELATIVE_PATH") != V3_ROOT
        or "parse_partial_access_ledger" not in contract_definitions
        or "validate_failure_receipt" not in contract_definitions
        or "validate_pre_ledger_failure_receipt" not in contract_definitions
        or "validate_pre_ledger_header" not in contract_definitions
    ):
        raise GuardFailure("V3 failure-receipt validation contract changed")
    standard_validator = _definition(
        contract_definitions, "validate_failure_receipt", ast.FunctionDef
    )
    pre_validator = _definition(
        contract_definitions,
        "validate_pre_ledger_failure_receipt",
        ast.FunctionDef,
    )
    assert isinstance(standard_validator, ast.FunctionDef)
    assert isinstance(pre_validator, ast.FunctionDef)
    standard_strings = {
        node.value
        for node in ast.walk(standard_validator)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    pre_strings = {
        node.value
        for node in ast.walk(pre_validator)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if (
        "validate_partial_operation_counts"
        not in _function_call_names(standard_validator)
        or "validate_partial_operation_counts"
        not in _function_call_names(pre_validator)
        or "empty_partial_operation_counts"
        not in _function_call_names(pre_validator)
        or "partial_access_ledger" not in standard_strings
        or "ledger_state" in standard_strings
        or "ledger_state" not in pre_strings
        or "partial_access_ledger" in pre_strings
    ):
        raise GuardFailure("standard and pre-ledger validators are not distinct")
    return {
        "canonical_to_dict_precedes_dataclass_normalization": True,
        "failure_receipt_direct_reservation_binding": True,
        "fsynced_open_attempt_and_outcome_ledger": True,
        "post_reservation_pre_ledger_terminalization": True,
        "prior_v1_and_v2_runtime_output_open_authorized": False,
        "schedule_validation_precedes_n320_and_raw": True,
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


def verify_compatibility_checker_static(root: Path = ROOT) -> dict[str, Any]:
    """Prove the one-shot checker is synthetic, strict, and source-bounded."""
    parsed: dict[str, ast.Module] = {}
    raw_sources: dict[str, bytes] = {}
    for relative in (COMPAT_CONTRACT, COMPAT_RUNNER, COMPAT_LAUNCHER):
        raw, tree = _parse_source(root, relative)
        raw_sources[relative] = raw
        parsed[relative] = tree
        imported: set[str] = set()
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                imported.update(alias.name for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom):
                imported.add(statement.module or "")
        if any(
            name == "torch"
            or name.startswith("torch.")
            or name == "lewm"
            or name.startswith("lewm.")
            for name in imported
        ):
            raise GuardFailure(
                f"compatibility parent imports payload code: {relative}"
            )

    contract_literals = _literal_assignments(parsed[COMPAT_CONTRACT])
    expected_compatibility_root = (
        ".generated/"
        "go2_rgb_multiresolution_perception_r9700_strict_compatibility_v1"
    )
    if (
        contract_literals.get("OUTPUT_ROOT_RELATIVE_PATH")
        != expected_compatibility_root
        or contract_literals.get("V3_PROBE_ROOT_RELATIVE_PATH") != V3_ROOT
        or expected_compatibility_root == V3_ROOT
        or expected_compatibility_root.startswith(V3_ROOT + "/")
        or V3_ROOT.startswith(expected_compatibility_root + "/")
        or contract_literals.get("ATTEMPT_INDEX") != 1
        or contract_literals.get("MAXIMUM_ATTEMPTS") != 1
        or contract_literals.get("EXIT_PASS") != 0
        or contract_literals.get("EXIT_COMPATIBILITY_FAIL") != 10
        or contract_literals.get("EXIT_OPERATIONAL_FAILURE") != 20
    ):
        raise GuardFailure("compatibility output or one-shot exit contract changed")

    expected_grid = {
        "operation": "grid_sample",
        "execution_order": 1,
        "batch_size": 4,
        "dense_feature_shape": [4, 36, 112, 112],
        "full_query_shape": [4, 128, 128, 5, 2],
        "query_count_per_batch": 81_920,
        "query_chunk_size": 4_096,
        "grid_call_shape": [4, 4_096, 1, 2],
        "grid_call_output_shape": [4, 36, 4_096, 1],
        "call_count": 20,
        "input_dtype": "torch.float32",
        "grid_dtype": "torch.float32",
        "input_requires_grad": True,
        "grid_requires_grad": False,
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False,
        "backward_call_count": 1,
        "cuda_synchronize_count": 1,
    }
    expected_scatter = {
        "operation": "scatter_add",
        "execution_order": 2,
        "batch_size": 4,
        "depth_bin_count": 64,
        "pixel_ray_shape": [84, 112],
        "ray_count_per_batch": 9_408,
        "output_shape": [64, 64],
        "cell_count": 4_096,
        "dimension": 0,
        "candidate_count_per_chunk": 4,
        "full_chunk": {
            "local_ray_count": 256,
            "chunk_count": 36,
            "destination_shape": [4_194_304],
            "source_and_index_shape_before_mask": [4, 64, 256],
            "selected_source_and_index_count": 65_536,
        },
        "tail_chunk": {
            "local_ray_count": 192,
            "chunk_count": 1,
            "destination_shape": [3_145_728],
            "source_and_index_shape_before_mask": [4, 64, 192],
            "selected_source_and_index_count": 49_152,
        },
        "scatter_add_call_count": 148,
        "source_dtype": "torch.float32",
        "index_dtype": "torch.int64",
        "validity_dtype": "torch.bool",
        "source_requires_grad": True,
        "synthetic_validity": "all_true_maximal_selected_shape",
        "synthetic_indices": "deterministic_in_range_with_collisions",
        "backward_call_count": 1,
        "cuda_synchronize_count": 1,
    }
    expected_determinism = {
        "requested":
            "torch.use_deterministic_algorithms(True, warn_only=False)",
        "algorithms_enabled": True,
        "warn_only_enabled": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "warning_count_required": 0,
        "fallback_authorized": False,
        "state_change_after_enable_authorized": False,
    }
    if (
        contract_literals.get("GRID_OPERATION") != expected_grid
        or contract_literals.get("SCATTER_OPERATION") != expected_scatter
        or contract_literals.get("DETERMINISM_CONTRACT")
        != expected_determinism
    ):
        raise GuardFailure("strict synthetic operation contract changed")

    prohibited = contract_literals.get("PROHIBITED_OPEN_COUNTS")
    zero_training = contract_literals.get("ZERO_TRAINING_COUNTS")
    denials = contract_literals.get("DOWNSTREAM_DENIALS")
    if (
        type(prohibited) is not dict
        or not prohibited
        or set(prohibited.values()) != {0}
        or type(zero_training) is not dict
        or not zero_training
        or set(zero_training.values()) != {0}
        or type(denials) is not dict
        or not denials
        or set(denials.values()) != {False}
        or denials.get("warn_only_or_strict_disable_authorized") is not False
        or denials.get(
            "retry_resume_extension_replacement_or_fallback_authorized"
        )
        is not False
    ):
        raise GuardFailure("compatibility denial or zero-use contract changed")

    runner_assignments = _assignment_map(parsed[COMPAT_RUNNER])

    def literal_stripped_template(name: str) -> str:
        assignment = runner_assignments.get(name)
        value: ast.expr | None = None
        if isinstance(assignment, ast.Assign):
            value = assignment.value
        elif isinstance(assignment, ast.AnnAssign):
            value = assignment.value
        if not (
            isinstance(value, ast.Call)
            and not value.args
            and not value.keywords
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "strip"
            and isinstance(value.func.value, ast.Constant)
            and isinstance(value.func.value.value, str)
        ):
            raise GuardFailure(f"{name} is not one literal stripped template")
        return value.func.value.value.strip()

    templates = {
        "grid_sample": literal_stripped_template("_GRID_TEMPLATE"),
        "scatter_add": literal_stripped_template("_SCATTER_TEMPLATE"),
    }
    for operation, source in templates.items():
        if type(source) is not str:
            raise GuardFailure(f"{operation} child template is not literal")
        child = ast.parse(source, filename=f"<{operation}-compatibility-child>")
        imported: set[str] = set()
        for statement in child.body:
            if isinstance(statement, ast.Import):
                imported.update(alias.name for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom):
                imported.add(statement.module or "")
        if (
            "torch" not in imported
            or any(
                name == "lewm" or name.startswith("lewm.")
                for name in imported
            )
            or "open" in _function_call_names(child)
            or "load" in _function_call_names(child)
        ):
            raise GuardFailure(f"{operation} child escaped synthetic-only scope")
        strict_calls = [
            node
            for node in ast.walk(child)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "use_deterministic_algorithms"
            )
        ]
        if (
            len(strict_calls) != 1
            or len(strict_calls[0].args) != 1
            or not isinstance(strict_calls[0].args[0], ast.Constant)
            or strict_calls[0].args[0].value is not True
            or len(strict_calls[0].keywords) != 1
            or strict_calls[0].keywords[0].arg != "warn_only"
            or not isinstance(strict_calls[0].keywords[0].value, ast.Constant)
            or strict_calls[0].keywords[0].value.value is not False
        ):
            raise GuardFailure(f"{operation} child is not strict warn_only=False")

    grid_source = templates["grid_sample"]
    scatter_source = templates["scatter_add"]
    assert isinstance(grid_source, str)
    assert isinstance(scatter_source, str)
    if not all(fragment in grid_source for fragment in {
        "reshape(4, 36, 112, 112)",
        "range(0, 81920, 4096)",
        "flat_grid[:, start:start + 4096, None, :]",
        "scalar.backward()",
        "torch.cuda.synchronize(0)",
    }) or not all(fragment in scatter_source for fragment in {
        'chunk_plan = [(256, "full", chunk) for chunk in range(36)]',
        'chunk_plan.append((192, "tail", 0))',
        "for candidate in range(4):",
        "destination = destination.scatter_add(",
        "scalar.backward()",
        "torch.cuda.synchronize(0)",
    }):
        raise GuardFailure("embedded compatibility multiplicity changed")

    runner_definitions = _definition_map(parsed[COMPAT_RUNNER])
    run_parent = _definition(
        runner_definitions, "run_parent", ast.FunctionDef
    )
    reserve = _definition(
        runner_definitions, "_reserve_output_root", ast.FunctionDef
    )
    assert isinstance(run_parent, ast.FunctionDef)
    assert isinstance(reserve, ast.FunctionDef)
    subprobe_calls = [
        node
        for node in ast.walk(run_parent)
        if isinstance(node, ast.Call) and _call_name(node) == "_run_subprobe"
    ]
    programs = [
        keyword.value.id
        for call in sorted(subprobe_calls, key=lambda node: node.lineno)
        for keyword in call.keywords
        if (
            keyword.arg == "program"
            and isinstance(keyword.value, ast.Name)
        )
    ]
    reserve_attributes = {
        node.attr
        for node in ast.walk(reserve)
        if isinstance(node, ast.Attribute)
    }
    if (
        programs != ["GRID_CHILD_PROGRAM", "SCATTER_CHILD_PROGRAM"]
        or "OUTPUT_ROOT_RELATIVE_PATH" not in reserve_attributes
        or "V3_PROBE_ROOT_RELATIVE_PATH" in reserve_attributes
    ):
        raise GuardFailure("compatibility child order or root reservation changed")

    combined_parent = b"\n".join(raw_sources.values())
    for forbidden in (
        b"lewm.models",
        b"lewm.datasets",
        b"torch.load",
    ):
        if forbidden in combined_parent:
            raise GuardFailure(
                f"compatibility source contains forbidden path: {forbidden!r}"
            )
    return {
        "child_order": ["grid_sample", "scatter_add"],
        "compatibility_output_root": expected_compatibility_root,
        "generated_dataset_checkpoint_model_open_count": 0,
        "grid_sample_call_count": 20,
        "scatter_add_call_count": 148,
        "strict_warn_only": False,
        "synthetic_only": True,
        "v3_probe_root_inspection_or_reservation_authorized": False,
    }


# Only these existing definitions may differ after V3-to-V2 envelope-string
# normalization. Every name is tied to the two lifecycle-only deltas.
AUTHORIZED_CHANGED_DEFINITIONS = {
    V3_CONTRACT: frozenset({
        "lifecycle_contract",
        "validate_failure_receipt",
        "validate_source_manifest",
    }),
    V3_RUNNER: frozenset({
        "PartialAccessLedger",
        "_execute_after_reservation",
        "_receipt_dict",
        "_terminal_failure",
        "_terminal_file_bindings",
    }),
    V3_LAUNCHER: frozenset(),
    V3_TEST: frozenset({
        "_ledger",
        "test_every_required_boundary_publishes_complete_sealed_failure",
        "test_schedule_adapter_calls_are_in_the_required_custody_order",
        "test_v1_implementation_bytes_remain_frozen",
    }),
    V3_CLOSURE_CHECKER: frozenset(),
    V3_CLOSURE_TEST: frozenset({
        "test_dynamic_runtime_edges_and_new_mechanism_are_inside_closure",
    }),
}
AUTHORIZED_ADDED_DEFINITIONS = {
    V3_CONTRACT: frozenset({
        "validate_pre_ledger_failure_receipt",
        "validate_pre_ledger_header",
    }),
    V3_RUNNER: frozenset({
        "_PreLedgerInitializationError",
        "_initialize_partial_access_ledger",
        "_read_pre_ledger_prefix",
        "_terminal_pre_ledger_failure",
    }),
    V3_LAUNCHER: frozenset(),
    V3_TEST: frozenset({
        "test_ledger_denies_any_prior_attempt_output_access",
        "test_v3_identity_and_fresh_operational_envelope",
        "test_v3_source_imports_do_not_import_torch",
    }),
    V3_CLOSURE_CHECKER: frozenset(),
    V3_CLOSURE_TEST: frozenset(),
}
AUTHORIZED_REMOVED_DEFINITIONS = {
    V3_CONTRACT: frozenset(),
    V3_RUNNER: frozenset(),
    V3_LAUNCHER: frozenset(),
    V3_TEST: frozenset({
        "test_ledger_denies_any_v1_attempt_output_access",
        "test_v2_identity_and_fresh_operational_envelope",
        "test_v2_source_imports_do_not_import_torch",
    }),
    V3_CLOSURE_CHECKER: frozenset(),
    V3_CLOSURE_TEST: frozenset(),
}
PAIRED_SOURCES = {
    V3_CONTRACT: V2_CONTRACT,
    V3_RUNNER: V2_RUNNER,
    V3_LAUNCHER: V2_LAUNCHER,
    V3_TEST: V2_TEST,
    V3_CLOSURE_CHECKER: V2_CLOSURE_CHECKER,
    V3_CLOSURE_TEST: V2_CLOSURE_TEST,
}


def verify_delta_surface(root: Path = ROOT) -> dict[str, Any]:
    observed: dict[str, dict[str, list[str]]] = {}
    for v3_relative, v2_relative in PAIRED_SOURCES.items():
        _, before = _parse_source(root, v2_relative)
        _, after = _parse_source(root, v3_relative)
        added, removed, changed = _definition_delta(before, after)
        expected = (
            set(AUTHORIZED_ADDED_DEFINITIONS[v3_relative]),
            set(AUTHORIZED_REMOVED_DEFINITIONS[v3_relative]),
            set(AUTHORIZED_CHANGED_DEFINITIONS[v3_relative]),
        )
        if (added, removed, changed) != expected:
            raise GuardFailure(
                f"unauthorized V2->V3 definition delta in {v3_relative}: "
                f"added={sorted(added)}, removed={sorted(removed)}, "
                f"changed={sorted(changed)}"
            )
        observed[v3_relative] = {
            "added": sorted(added),
            "changed": sorted(changed),
            "removed": sorted(removed),
        }
    # Science-bearing runner definitions must remain byte-semantic AST
    # identities after version-only string normalization.
    _, v2_runner = _parse_source(root, V2_RUNNER)
    _, v3_runner = _parse_source(root, V3_RUNNER)
    before = _definition_map(v2_runner)
    after = _definition_map(v3_runner)
    science_bearing = {
        "_camera_components",
        "_camera_pair",
        "_evaluate",
        "_gradient_group_norm",
        "_prepare_model",
        "_publish_metric_sidecar",
        "_publish_training_records",
        "_snapshot",
        "_state_sha",
        "_subset_sha",
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
        or _normalized_dump(before["_train"])
        != _normalized_dump(after["_train"])
    ):
        raise GuardFailure("V3 training computation changed")

    _, v2_contract = _parse_source(root, V2_CONTRACT)
    _, v3_contract = _parse_source(root, V3_CONTRACT)
    before_contract = _definition_map(v2_contract)
    after_contract = _definition_map(v3_contract)
    science_contract_functions = {
        "checkpoint_control_decision",
        "evaluate_physical_scopes",
        "learning_rates",
        "operation_counts",
        "parameter_partition",
        "physical_margins",
        "science_contract",
    }
    for name in science_contract_functions:
        if (
            name not in before_contract
            or name not in after_contract
            or _normalized_dump(before_contract[name])
            != _normalized_dump(after_contract[name])
        ):
            raise GuardFailure(f"science-bearing contract definition changed: {name}")
    return {
        "authorized_operational_delta_ids": [
            "canonical_initialization_receipt_normalization",
            "post_reservation_pre_ledger_terminalization",
            "synthetic_r9700_strict_determinism_compatibility_checker",
        ],
        "paired_source_deltas": observed,
        "science_bearing_contract_definition_count":
            len(science_contract_functions),
        "science_bearing_runner_definition_count": len(science_bearing) + 1,
    }


def verify_all(root: Path = ROOT) -> dict[str, Any]:
    return {
        "authority": verify_preregistration_authority(root),
        "compatibility_checker": verify_compatibility_checker_static(root),
        "delta_surface": verify_delta_surface(root),
        "generated_runtime_checkpoint_or_gpu_open_count": 0,
        "model_and_roots": verify_model_and_roots(root),
        "operational_mechanisms": verify_operational_mechanisms(root),
        "schedule_adapter": verify_schedule_adapter(root),
        "science_identity": verify_science_identity(root),
        "v2_frozen_sources": verify_v2_frozen_sources(root),
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
            "PASS V3 science identity and exact recovery-delta source guard "
            f"(science={SCIENCE_SHA256}, frozen_v2=12, generated_opens=0)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
