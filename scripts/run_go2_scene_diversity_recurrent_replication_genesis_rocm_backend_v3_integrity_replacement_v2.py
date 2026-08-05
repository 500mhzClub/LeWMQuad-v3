#!/usr/bin/env python3
"""Run replacement V2 science only after exact replacement qualification."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import threading
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as collector  # noqa: E402
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as qualifier  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1 as predecessor  # noqa: E402


predecessor_runner = predecessor.predecessor_runner
v2_runner = predecessor.v2_runner
v1_replacement_runner = predecessor.v1_replacement_runner
frozen_runner = predecessor.frozen_runner

AUTHORITY_SCHEMA = collector.AUTHORITY_SCHEMA
AUTHORITY_STATUS = collector.AUTHORITY_STATUS
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"
RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_result_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_terminal_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_attempt_reservation_v1"
)
PREREGISTRATION = qualifier.PREREGISTRATION
SOURCE_REVIEW = qualifier.SOURCE_REVIEW
SCENE_PANEL = predecessor.SCENE_PANEL
SCENE_PANEL_SHA256 = predecessor.SCENE_PANEL_SHA256
SCENE_PANEL_BYTE_COUNT = predecessor.SCENE_PANEL_BYTE_COUNT
DEFAULT_ATTEMPT_ROOT = plan_builder.DEFAULT_ATTEMPT_ROOT
DEFAULT_COLLECTION_ROOT = plan_builder.DEFAULT_OUTPUT_ROOT

CPU_TERMINAL_REVIEW = plan_builder.CPU_TERMINAL_REVIEW
CPU_TERMINAL_REVIEW_SHA256 = plan_builder.CPU_TERMINAL_REVIEW_SHA256
CPU_TERMINAL_REVIEW_BYTE_COUNT = plan_builder.CPU_TERMINAL_REVIEW_BYTE_COUNT

DINO_REPOSITORY = predecessor.DINO_REPOSITORY
DINO_CHECKPOINT = predecessor.DINO_CHECKPOINT
DINO_REPOSITORY_COMMIT = predecessor.DINO_REPOSITORY_COMMIT
DINO_CHECKPOINT_SHA256 = predecessor.DINO_CHECKPOINT_SHA256
DINO_CHECKPOINT_BYTE_COUNT = predecessor.DINO_CHECKPOINT_BYTE_COUNT
PROCESS_RESET_DEPENDENCY_PATHS = predecessor.PROCESS_RESET_DEPENDENCY_PATHS
ROCM_BACKEND_DEPENDENCY_PATHS = predecessor.ROCM_BACKEND_DEPENDENCY_PATHS

V2_PREREGISTRATION_SOURCE = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v2_preregistration_2026-08-04.md"
)
V2_PREREGISTRATION_SOURCE_SHA256 = (
    "f4d2b46ddb7a0ac97f95160f55c8aadd58f22ae1e63b7ab85e500c083a86b334"
)
V2_PREREGISTRATION_SOURCE_BYTE_COUNT = 4_007

_replacement_v1_source_paths = dict(predecessor.SOURCE_PATHS)
_replacement_v1_source_paths["v2_rocm_preregistration_source"] = (
    V2_PREREGISTRATION_SOURCE
)
if len(_replacement_v1_source_paths) != 236:
    raise RuntimeError("reviewed replacement V1 source closure key count changed")
FROZEN_REPLACEMENT_V1_SOURCE_PATHS = MappingProxyType(
    _replacement_v1_source_paths
)

SOURCE_PATHS = MappingProxyType({
    **FROZEN_REPLACEMENT_V1_SOURCE_PATHS,
    "rocm_backend_v3_integrity_replacement_v2_plan_builder": Path(
        plan_builder.__file__
    ).resolve(),
    "rocm_backend_v3_integrity_replacement_v2_collector": Path(
        collector.__file__
    ).resolve(),
    "rocm_backend_v3_integrity_replacement_v2_qualifier": Path(
        qualifier.__file__
    ).resolve(),
    "rocm_backend_v3_integrity_replacement_v2_runner": Path(__file__).resolve(),
    "rocm_backend_v3_integrity_replacement_v2_qualification_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_qualification_authority.py",
    "rocm_backend_v3_integrity_replacement_v2_scientific_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_authority.py",
    "rocm_backend_v3_integrity_replacement_v2_plan_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_plan.py",
    "rocm_backend_v3_integrity_replacement_v2_collector_test": REPO_ROOT
    / "lewm/tests/test_collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2.py",
    "rocm_backend_v3_integrity_replacement_v2_qualifier_test": REPO_ROOT
    / "lewm/tests/test_qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2.py",
    "rocm_backend_v3_integrity_replacement_v2_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2.py",
    "rocm_backend_v3_integrity_replacement_v2_qualification_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_qualification_authority.py",
    "rocm_backend_v3_integrity_replacement_v2_scientific_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_authority.py",
    "rocm_backend_v3_integrity_replacement_v2_preregistration": PREREGISTRATION,
    "rocm_backend_v3_integrity_replacement_v2_scientific_exact_plan": (
        plan_builder.DEFAULT_PLAN_OUTPUT
    ),
    "rocm_backend_v3_integrity_replacement_v2_qualification_exact_plan": (
        plan_builder.QUALIFICATION_PLAN_OUTPUT
    ),
    "predecessor_replacement_v1_qualification_terminal_review": (
        plan_builder.REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW
    ),
})

SceneDiversityRunnerError = predecessor.SceneDiversityRunnerError
ContextOnlyLedgerV1 = predecessor.ContextOnlyLedgerV1
RoleRuntimeDataV1 = predecessor.RoleRuntimeDataV1
benchmark = predecessor.benchmark
torch = predecessor.torch
canonical_bytes_v1 = predecessor.canonical_bytes_v1
file_binding_v1 = predecessor.file_binding_v1
expected_dino_v1 = predecessor.expected_dino_v1
assert_role_disjointness_v1 = predecessor.assert_role_disjointness_v1


def v2_preregistration_source_binding() -> dict[str, object]:
    expected = {
        "path": str(V2_PREREGISTRATION_SOURCE.resolve(strict=True)),
        "sha256": V2_PREREGISTRATION_SOURCE_SHA256,
        "byte_count": V2_PREREGISTRATION_SOURCE_BYTE_COUNT,
    }
    if (
        FROZEN_REPLACEMENT_V1_SOURCE_PATHS.get("v2_rocm_preregistration_source")
        != V2_PREREGISTRATION_SOURCE
        or file_binding_v1(V2_PREREGISTRATION_SOURCE) != expected
    ):
        raise SceneDiversityRunnerError(
            "literal V2 preregistration source binding changed"
        )
    return expected

ROCM_EGL_PREFLIGHT_FIELDS = frozenset(
    set(predecessor.ROCM_EGL_PREFLIGHT_FIELDS)
    | {
        "path_ld_lld_driver",
        "rocm_path_ld_lld_driver",
        "lld_driver_entrypoint",
        "lld_driver_link_text",
        "lld_resolved_target",
        "lld_invocation_argv",
        "lld_version_prefix_passed",
    }
)
ROCM_IDENTITY_FIELDS = frozenset(
    set(predecessor.ROCM_IDENTITY_FIELDS) | {"home"}
)
QUALIFICATION_SCENE_RESULT_FIELDS = predecessor.QUALIFICATION_SCENE_RESULT_FIELDS
QUALIFICATION_PROBE_FIELDS = predecessor.QUALIFICATION_PROBE_FIELDS
QUALIFICATION_WORKER_FIELDS = predecessor.QUALIFICATION_WORKER_FIELDS
QUALIFICATION_RELEASE_BARRIER_FIELDS = (
    predecessor.QUALIFICATION_RELEASE_BARRIER_FIELDS
)
QUALIFICATION_KERNEL_AUDIT_FIELDS = predecessor.QUALIFICATION_KERNEL_AUDIT_FIELDS
QUALIFICATION_TIMING_GATE_FIELDS = predecessor.QUALIFICATION_TIMING_GATE_FIELDS

_ORIGINAL_VALIDATE_QUALIFICATION_RESULT = (
    predecessor.validate_qualification_result_binding
)
_ORIGINAL_VALIDATE_AUTHORITY_ROCM = predecessor._validate_authority_rocm  # noqa: SLF001
_CONFIGURATION_LOCK = threading.RLock()


class _V3QualifierReplayFacade:
    """Expose V3 qualification APIs with the inherited scene implementation.

    The inherited runner treats ``qualifier.predecessor`` as the scene
    collector implementation.  V3 preserves V2's explicit replay seam.
    """

    predecessor = qualifier.scene_predecessor

    def __getattr__(self, name: str) -> object:
        return getattr(qualifier, name)


_QUALIFIER_REPLAY_FACADE = _V3QualifierReplayFacade()


def _standard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "sha256": str(value.get("sha256", value.get("file_sha256"))),
        "byte_count": int(value["byte_count"]),
    }


def _immutable_binding(
    relative_path: str, sha256: str, byte_count: int
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "path": str((REPO_ROOT / relative_path).resolve()),
            "sha256": sha256,
            "byte_count": byte_count,
        }
    )


# This is deliberately literal and flat.  The replacement never calls an
# inherited failure-evidence helper after import, so temporary plan-builder
# overlays cannot redirect an older helper through an API that the fresh
# builder does not expose.  The final replacement-V1 review is the sole new
# predecessor evidence; no replacement-V1 authority, receipt, or runtime
# payload is eligible.
FROZEN_PREDECESSOR_FAILURE_BINDINGS = MappingProxyType(
    {
        "predecessor_failure_terminal": _immutable_binding(
            ".generated/dev/go2_scene_diversity_recurrent_replication_v1/attempt_v1/terminal.json",
            "df4cecb5edc45f25a98f4753e82e95334a6b8c4e9e0d719bb13150b9be690bfa",
            451,
        ),
        "predecessor_terminal_review": _immutable_binding(
            "docs/lewm_go2_scene_diversity_recurrent_replication_v1_terminal_review_2026-08-04.json",
            "7f2ac7eb3f9fa16fd91a5311009cdbcd7d4777e8d9c3f7746666ce1afbc6da59",
            15_809,
        ),
        "predecessor_replacement_v1_failure_terminal": _immutable_binding(
            ".generated/dev/go2_scene_diversity_recurrent_replication_integrity_replacement_v1/attempt_v1/terminal.json",
            "c2650529ff6b1aa1322738e7d4f748fbab03ea9ef33a623ff3e182a98e1cb77d",
            473,
        ),
        "predecessor_replacement_v1_terminal_review": _immutable_binding(
            "docs/lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_terminal_review_2026-08-04.json",
            "e25f9281449147b6937edfafa022b5fbeb18ca3e844438c70a5babb84c2eb0cb",
            21_621,
        ),
        "predecessor_replacement_v2_failure_terminal": _immutable_binding(
            ".generated/dev/go2_scene_diversity_recurrent_replication_integrity_replacement_v2/attempt_v1/terminal.json",
            "ebb520c596ae69c19e3be255c0f661fe55286883e03994a43a0e506936120465",
            442,
        ),
        "predecessor_replacement_v2_terminal_review": _immutable_binding(
            "docs/lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_terminal_review_2026-08-04.json",
            "7878dbedf21ed3bdb13927e0404925edc376e512c00a5ba4bf56e0091e3204c6",
            20_561,
        ),
        "predecessor_cpu_qualification_terminal_review": _immutable_binding(
            "docs/lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_qualification_terminal_review_2026-08-04.json",
            "9b0c31c05b4fb6064c67116a456d34a6f7e49cfe85ec55ed081599acb18502f0",
            20_536,
        ),
        "predecessor_v1_qualification_terminal_review": _immutable_binding(
            "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_qualification_terminal_review_2026-08-04.json",
            "3e35cdb459c18d862e21df676b0a630a0496d1a26f8a97874095c71ab2facb5b",
            14_742,
        ),
        "predecessor_v2_qualification_terminal_review": _immutable_binding(
            "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_qualification_terminal_review_2026-08-04.json",
            "166aec87b6e61d62116069a12472b768c3ff462c09cf1e6088af62ab7397dd0e",
            16_198,
        ),
        "predecessor_v3_qualification_terminal_review": _immutable_binding(
            "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_qualification_terminal_review_2026-08-04.json",
            "139c7365236eb76b56c8354de9100d51dce56964f15541980f102c9f33804cd5",
            8_551,
        ),
        "v2_rocm_preregistration_source": _immutable_binding(
            "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_preregistration_2026-08-04.md",
            V2_PREREGISTRATION_SOURCE_SHA256,
            V2_PREREGISTRATION_SOURCE_BYTE_COUNT,
        ),
        "predecessor_replacement_v1_qualification_terminal_review": (
            MappingProxyType(
                _standard_binding(
                    plan_builder.replacement_v1_qualification_terminal_review_binding()
                )
            )
        ),
    }
)


def predecessor_failure_bindings_rocm() -> dict[str, dict[str, object]]:
    """Rehash the complete frozen predecessor evidence without delegation."""

    evidence: dict[str, dict[str, object]] = {}
    for name, frozen in FROZEN_PREDECESSOR_FAILURE_BINDINGS.items():
        expected = dict(frozen)
        actual = _standard_binding(
            file_binding_v1(Path(str(expected["path"])))
        )
        if actual != expected:
            raise SceneDiversityRunnerError(
                f"frozen predecessor failure evidence changed at {name}"
            )
        evidence[name] = expected
    return evidence


def _validate_v3_preflight_evidence(result: Mapping[str, Any]) -> None:
    preflight = result.get("rocm_egl_preflight")
    if not isinstance(preflight, Mapping):
        raise SceneDiversityRunnerError("V3 linker evidence is absent")
    driver = str(plan_builder.ROCM_LD_LLD_DRIVER_ENTRYPOINT)
    target = str(
        plan_builder.ROCM_RUNTIME_PATHS["rocm_lld_executable"].resolve(
            strict=True
        )
    )
    if (
        set(preflight) != ROCM_EGL_PREFLIGHT_FIELDS
        or preflight.get("path_ld_lld") != target
        or preflight.get("rocm_path_ld_lld") != target
        or preflight.get("path_ld_lld_driver") != driver
        or preflight.get("rocm_path_ld_lld_driver") != driver
        or preflight.get("lld_driver_entrypoint") != driver
        or preflight.get("lld_driver_link_text")
        != plan_builder.ROCM_LD_LLD_DRIVER_LINK_TEXT
        or preflight.get("lld_resolved_target") != target
        or preflight.get("lld_invocation_argv") != [driver, "--version"]
        or preflight.get("lld_version_prefix_passed") is not True
        or not isinstance(preflight.get("identity"), Mapping)
        or preflight["identity"].get("home")
        != plan_builder.REQUIRED_HOST_HOME
        or preflight.get("expectation")
        != plan_builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION
    ):
        raise SceneDiversityRunnerError(
            "V3 exact ld.lld driver evidence changed"
        )


def _configuration_overrides_v3() -> dict[str, object]:
    return {
        "plan_builder": plan_builder,
        "collector": collector,
        "qualifier": _QUALIFIER_REPLAY_FACADE,
        "AUTHORITY_SCHEMA": AUTHORITY_SCHEMA,
        "AUTHORITY_STATUS": AUTHORITY_STATUS,
        "SOURCE_REVIEW_SCHEMA": SOURCE_REVIEW_SCHEMA,
        "SOURCE_REVIEW_STATUS": SOURCE_REVIEW_STATUS,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "TERMINAL_SCHEMA": TERMINAL_SCHEMA,
        "RESERVATION_SCHEMA": RESERVATION_SCHEMA,
        "PREREGISTRATION": PREREGISTRATION,
        "SOURCE_REVIEW": SOURCE_REVIEW,
        "DEFAULT_ATTEMPT_ROOT": DEFAULT_ATTEMPT_ROOT,
        "DEFAULT_COLLECTION_ROOT": DEFAULT_COLLECTION_ROOT,
        "SOURCE_PATHS": SOURCE_PATHS,
        "ROCM_EGL_PREFLIGHT_FIELDS": ROCM_EGL_PREFLIGHT_FIELDS,
        "ROCM_IDENTITY_FIELDS": ROCM_IDENTITY_FIELDS,
        "predecessor_failure_bindings_rocm": predecessor_failure_bindings_rocm,
        "validate_qualification_result_binding": (
            validate_qualification_result_binding
        ),
        "_validate_plan_rocm": _validate_plan_rocm,
    }


_TOP_RUNNER_OVERRIDE_KEYS = frozenset(
    {
        "plan_builder",
        "collector",
        "qualifier",
        "AUTHORITY_SCHEMA",
        "AUTHORITY_STATUS",
        "SOURCE_REVIEW_SCHEMA",
        "SOURCE_REVIEW_STATUS",
        "RESULT_SCHEMA",
        "TERMINAL_SCHEMA",
        "RESERVATION_SCHEMA",
        "PREREGISTRATION",
        "SOURCE_REVIEW",
        "DEFAULT_ATTEMPT_ROOT",
        "DEFAULT_COLLECTION_ROOT",
        "SOURCE_PATHS",
        "ROCM_EGL_PREFLIGHT_FIELDS",
        "ROCM_IDENTITY_FIELDS",
        "predecessor_failure_bindings_rocm",
        "validate_qualification_result_binding",
        "_validate_plan_rocm",
    }
)
_V2_TO_V1_RUNNER_OVERRIDE_KEYS = frozenset(
    _TOP_RUNNER_OVERRIDE_KEYS
    - {"ROCM_IDENTITY_FIELDS"}
)
_V1_TO_RECURRENT_RUNNER_OVERRIDE_KEYS = frozenset(
    {
        "collector",
        "plan_builder",
        "AUTHORITY_SCHEMA",
        "AUTHORITY_STATUS",
        "SOURCE_REVIEW_SCHEMA",
        "SOURCE_REVIEW_STATUS",
        "RESULT_SCHEMA",
        "TERMINAL_SCHEMA",
        "RESERVATION_SCHEMA",
        "PREREGISTRATION",
        "SCENE_PANEL",
        "SCENE_PANEL_SHA256",
        "SCENE_PANEL_BYTE_COUNT",
        "SOURCE_REVIEW",
        "DEFAULT_ATTEMPT_ROOT",
        "DEFAULT_COLLECTION_ROOT",
        "SOURCE_PATHS",
        "predecessor_failure_bindings_v3",
        "_load_replacement_physics_index_v3",
        "_validate_plan_v3",
    }
)
RUNNER_COMPATIBILITY_MATRIX = MappingProxyType(
    {
        "replacement_v2_to_replacement_v1": tuple(
            sorted(_TOP_RUNNER_OVERRIDE_KEYS)
        ),
        "replacement_v1_to_v3": tuple(sorted(_TOP_RUNNER_OVERRIDE_KEYS)),
        "v3_to_v2": tuple(sorted(_TOP_RUNNER_OVERRIDE_KEYS)),
        "v2_to_v1": tuple(sorted(_V2_TO_V1_RUNNER_OVERRIDE_KEYS)),
        "v1_to_recurrent": tuple(
            sorted(_V1_TO_RECURRENT_RUNNER_OVERRIDE_KEYS)
        ),
        "runtime_entrypoints": (
            "benchmark",
            "collector",
            "qualifier",
            "plan_builder",
            "file_binding_v1",
            "predecessor_failure_bindings_rocm",
            "validate_qualification_result_binding",
            "_validate_plan_rocm",
            "_validate_authority_rocm",
            "execute_rocm",
            "build_parser",
            "main",
        ),
    }
)


def validate_runner_compatibility_facade() -> dict[str, Any]:
    collector_audit = collector.validate_collector_compatibility_facade()
    qualifier_audit = qualifier.validate_qualifier_compatibility_facade()
    missing = [
        name
        for name in RUNNER_COMPATIBILITY_MATRIX["runtime_entrypoints"]
        if name not in globals()
    ]
    if missing:
        raise SceneDiversityRunnerError(
            f"replacement V2 runner compatibility facade is incomplete: {missing}"
        )

    r1 = predecessor
    r3 = r1.predecessor
    r2 = r3.predecessor
    r1_base = r2.predecessor
    observed = {
        "replacement_v2_to_replacement_v1": set(
            _configuration_overrides_v3()
        ),
        "replacement_v1_to_v3": set(r1._configuration_overrides_v3()),  # noqa: SLF001
        "v3_to_v2": set(r3._configuration_overrides_v3()),  # noqa: SLF001
        "v2_to_v1": set(r2._configuration_overrides_v2()),  # noqa: SLF001
        "v1_to_recurrent": set(r1_base._configuration_overrides_rocm()),  # noqa: SLF001
    }
    expected = {
        "replacement_v2_to_replacement_v1": _TOP_RUNNER_OVERRIDE_KEYS,
        "replacement_v1_to_v3": _TOP_RUNNER_OVERRIDE_KEYS,
        "v3_to_v2": _TOP_RUNNER_OVERRIDE_KEYS,
        "v2_to_v1": _V2_TO_V1_RUNNER_OVERRIDE_KEYS,
        "v1_to_recurrent": _V1_TO_RECURRENT_RUNNER_OVERRIDE_KEYS,
    }
    mismatches = {
        layer: {
            "missing": sorted(set(expected[layer]) - values),
            "unexpected": sorted(values - set(expected[layer])),
        }
        for layer, values in observed.items()
        if values != set(expected[layer])
    }
    replay_missing = [
        name
        for layer in (
            "v1_runner_and_replay",
            "v2_v3_and_replacement_runners",
            "runtime_entrypoints",
        )
        for name in qualifier.QUALIFIER_COMPATIBILITY_MATRIX[layer]
        if not hasattr(_QUALIFIER_REPLAY_FACADE, name)
    ]
    if _QUALIFIER_REPLAY_FACADE.predecessor is not qualifier.scene_predecessor:
        replay_missing.append("predecessor_scene_identity")
    if mismatches or replay_missing:
        raise SceneDiversityRunnerError(
            "replacement V2 runner layered interface changed: "
            f"overrides={mismatches}, replay_missing={sorted(set(replay_missing))}"
        )
    return {
        "schema": (
            "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
            "backend_v3_integrity_replacement_v2_runner_compatibility_v1"
        ),
        "collector": collector_audit,
        "qualifier": qualifier_audit,
        "layers": {
            layer: list(names)
            for layer, names in RUNNER_COMPATIBILITY_MATRIX.items()
        },
        "passed": True,
    }


@contextmanager
def _configured_predecessor_runner_rocm() -> Iterator[None]:
    validate_runner_compatibility_facade()
    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_v3()
        originals = {
            name: getattr(predecessor, name) for name in overrides
        }
        lower = predecessor.predecessor
        original_identity_fields = lower.ROCM_IDENTITY_FIELDS
        try:
            for name, value in overrides.items():
                setattr(predecessor, name, value)
            lower.ROCM_IDENTITY_FIELDS = ROCM_IDENTITY_FIELDS
            with predecessor._configured_predecessor_runner_rocm():  # noqa: SLF001
                yield
        finally:
            lower.ROCM_IDENTITY_FIELDS = original_identity_fields
            for name, value in originals.items():
                setattr(predecessor, name, value)


def validate_qualification_result_binding(value: object):
    validate_runner_compatibility_facade()
    with _configured_predecessor_runner_rocm():
        result, binding = _ORIGINAL_VALIDATE_QUALIFICATION_RESULT(
            value
        )
    _validate_v3_preflight_evidence(result)
    try:
        authority = json.loads(qualifier.QUALIFICATION_AUTHORITY.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "V3 qualification authority is not strict JSON"
        ) from exc
    collector._require_replacement_v1_review_binding(authority)  # noqa: SLF001
    return result, binding


def _validate_plan_rocm(
    plan: Mapping[str, Any], authority: Mapping[str, Any]
) -> None:
    if authority.get("attempt_id") != plan_builder.DEFAULT_ATTEMPT_ID:
        raise SceneDiversityRunnerError(
            "V3 scientific authority identity changed"
        )
    try:
        plan_builder.validate_rocm_plan(
            plan,
            expected_attempt_id=plan_builder.DEFAULT_ATTEMPT_ID,
            expected_output_root=plan_builder.DEFAULT_OUTPUT_ROOT,
            plan_role="scientific",
        )
    except plan_builder.SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError as exc:
        raise SceneDiversityRunnerError(str(exc)) from exc


def _validate_authority_rocm(
    authority_path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
):
    validate_runner_compatibility_facade()
    predecessor_failure_bindings_rocm()
    with _configured_predecessor_runner_rocm():
        validated = _ORIGINAL_VALIDATE_AUTHORITY_ROCM(
            authority_path,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
        )
    collector._require_replacement_v1_review_binding(validated[0])  # noqa: SLF001
    validate_qualification_result_binding(
        validated[0].get("qualification_result_binding")
    )
    _validate_plan_rocm(validated[2], validated[0])
    return validated


def execute_rocm(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    validate_runner_compatibility_facade()
    collector.require_exact_orchestrator_python()
    collector.require_exact_orchestrator_environment("scientific")
    _validate_plan_rocm(plan, authority)
    predecessor_failure_bindings_rocm()
    collector._require_replacement_v1_review_binding(authority)  # noqa: SLF001
    validate_qualification_result_binding(
        authority.get("qualification_result_binding")
    )
    with _configured_predecessor_runner_rocm():
        return predecessor_runner.execute_v3(
            authority,
            authority_binding=authority_binding,
            plan=plan,
        )


execute_v3 = execute_rocm
execute_v2 = execute_rocm
execute_v1 = execute_rocm
_validate_plan_v3 = _validate_plan_rocm
_validate_authority_v3 = _validate_authority_rocm


def build_parser():
    validate_runner_compatibility_facade()
    parser = predecessor_runner.build_parser()
    parser.description = (
        "Run fresh Genesis ROCm V3 science after exact V3 qualification."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    validate_runner_compatibility_facade()
    collector.require_exact_orchestrator_python()
    collector.require_exact_orchestrator_environment("scientific")
    args = build_parser().parse_args(argv)
    authority: Mapping[str, Any] | None = None
    try:
        authority, binding, plan = _validate_authority_rocm(
            args.authority,
            expected_sha256=args.expected_authority_sha256,
            expected_byte_count=args.expected_authority_byte_count,
        )
        result = execute_rocm(
            authority,
            authority_binding=binding,
            plan=plan,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "attempt_root": authority["attempt_root"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if authority is not None:
            attempt = DEFAULT_ATTEMPT_ROOT
            terminal = attempt / "terminal.json"
            if attempt.is_dir() and not terminal.exists():
                try:
                    collector.pilot.write_json_exclusive(
                        terminal,
                        {
                            "schema": TERMINAL_SCHEMA,
                            "status": (
                                "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
                            ),
                            "authorizes_retry_or_resume": False,
                            "authorizes_navigation_claim": False,
                            "authorizes_blind_rollout_preregistration": False,
                            "result_binding": None,
                            "failure": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        },
                    )
                except Exception:
                    pass
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "ContextOnlyLedgerV1",
    "DEFAULT_ATTEMPT_ROOT",
    "DEFAULT_COLLECTION_ROOT",
    "RESULT_SCHEMA",
    "ROCM_BACKEND_DEPENDENCY_PATHS",
    "SOURCE_PATHS",
    "SceneDiversityRunnerError",
    "TERMINAL_SCHEMA",
    "execute_rocm",
    "predecessor_failure_bindings_rocm",
    "validate_qualification_result_binding",
]
