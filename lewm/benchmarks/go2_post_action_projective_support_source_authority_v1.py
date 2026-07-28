"""Source-only custody for the post-action projective-support JEPA probe.

The functions in this module bind explicitly named source and runtime artifacts.
Importing it reads Python source only: it does not import tensor libraries or open
generated data, labels, checkpoints, traces, or benchmark material.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Final, Iterable, Mapping, Sequence


ROOT: Final = Path(__file__).resolve().parents[2]


def _source_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_GEOMETRY_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
_CORRIDOR_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_post_action_projective_support_corridor_contract_v1.py"
)
_geometry = _source_module(
    "_lewm_post_action_projective_support_geometry_authority_base",
    _GEOMETRY_CONTRACT_RELATIVE_PATH,
)
contract = _source_module(
    "_lewm_post_action_projective_support_corridor_authority_contract",
    _CORRIDOR_CONTRACT_RELATIVE_PATH,
)


AUTHORITY_RELATIVE_PATH: Final = (
    "lewm/benchmarks/go2_post_action_projective_support_source_authority_v1.py"
)
LABELS_RELATIVE_PATH: Final = (
    "lewm/benchmarks/go2_post_action_projective_support_labels_v1.py"
)
SCORING_RELATIVE_PATH: Final = (
    "lewm/benchmarks/go2_post_action_projective_support_joint_jepa_v1.py"
)
METRICS_RELATIVE_PATH: Final = (
    "lewm/benchmarks/go2_post_action_projective_support_metrics_v1.py"
)
CORE_RUNNER_RELATIVE_PATH: Final = (
    "scripts/run_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1.py"
)
LABEL_BUILDER_RELATIVE_PATH: Final = (
    "scripts/build_go2_post_action_projective_support_labels_v1.py"
)
PREFLIGHT_RELATIVE_PATH: Final = (
    "scripts/preflight_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1.py"
)
EXECUTE_RELATIVE_PATH: Final = (
    "scripts/execute_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1.py"
)

AUTHORITY_TEST_RELATIVE_PATH: Final = (
    "lewm/tests/test_go2_post_action_projective_support_source_authority_v1.py"
)
CONTRACT_TEST_RELATIVE_PATH: Final = (
    "lewm/tests/test_go2_post_action_projective_support_corridor_contract_v1.py"
)
LABELS_TEST_RELATIVE_PATH: Final = (
    "lewm/tests/test_go2_post_action_projective_support_labels_v1.py"
)
SCORING_TEST_RELATIVE_PATH: Final = (
    "lewm/tests/test_go2_post_action_projective_support_joint_jepa_v1.py"
)
METRICS_TEST_RELATIVE_PATH: Final = (
    "lewm/tests/test_go2_post_action_projective_support_metrics_v1.py"
)
CORE_RUNNER_TEST_RELATIVE_PATH: Final = (
    "lewm/tests/test_run_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1.py"
)
PREFLIGHT_TEST_RELATIVE_PATH: Final = (
    "lewm/tests/test_preflight_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1.py"
)
EXECUTE_TEST_RELATIVE_PATH: Final = (
    "lewm/tests/test_execute_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1.py"
)

INHERITED_GEOMETRY_SOURCE_PATHS: Final = tuple(_geometry.SOURCE_PATHS)
ADDITIVE_SOURCE_PATHS: Final = tuple(sorted({
    AUTHORITY_RELATIVE_PATH,
    _CORRIDOR_CONTRACT_RELATIVE_PATH,
    LABELS_RELATIVE_PATH,
    SCORING_RELATIVE_PATH,
    METRICS_RELATIVE_PATH,
    CORE_RUNNER_RELATIVE_PATH,
    LABEL_BUILDER_RELATIVE_PATH,
    PREFLIGHT_RELATIVE_PATH,
    EXECUTE_RELATIVE_PATH,
    AUTHORITY_TEST_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    LABELS_TEST_RELATIVE_PATH,
    SCORING_TEST_RELATIVE_PATH,
    METRICS_TEST_RELATIVE_PATH,
    CORE_RUNNER_TEST_RELATIVE_PATH,
    PREFLIGHT_TEST_RELATIVE_PATH,
    EXECUTE_TEST_RELATIVE_PATH,
    contract.PREREGISTRATION_RELATIVE_PATH,
    contract.INTEGRITY_ADAPTER_AMENDMENT_RELATIVE_PATH,
}))
SOURCE_MANIFEST_ENTRYPOINTS: Final = (
    LABEL_BUILDER_RELATIVE_PATH,
    PREFLIGHT_RELATIVE_PATH,
    EXECUTE_RELATIVE_PATH,
    CORE_RUNNER_RELATIVE_PATH,
)

# The four entrypoints contain several reviewed ``importlib`` seams.  Every
# inherited geometry source and every additive Python source is therefore an
# explicit dynamic root, after which an AST walk must add every ordinary local
# import (including package ``__init__.py`` files).  This is deliberately not a
# hand-maintained allow-list masquerading as a recursive closure.
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES: Final = tuple(sorted({
    *INHERITED_GEOMETRY_SOURCE_PATHS,
    *(path for path in ADDITIVE_SOURCE_PATHS if path.endswith(".py")),
}))
_LOCAL_PACKAGE_ROOTS: Final = (
    ("lewm", "lewm"),
    ("scripts", "scripts"),
    ("lewm_worlds", "lewm_worlds/lewm_worlds"),
)

SOURCE_MANIFEST_RELATIVE_PATH: Final = (
    "docs/lewm_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1_"
    "source_manifest_v2_2026-07-28.json"
)
SOURCE_REVIEW_RELATIVE_PATH: Final = (
    "docs/lewm_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1_"
    "source_review_v2_2026-07-28.json"
)
EXECUTION_BINDING_RELATIVE_PATH: Final = (
    "docs/lewm_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1_"
    "execution_binding_v2_2026-07-28.json"
)
LABEL_BUILDER_EXECUTION_BINDING_RELATIVE_PATH: Final = (
    "docs/lewm_go2_post_action_projective_support_labels_v2_"
    "execution_binding_2026-07-28.json"
)
LABEL_RESERVATION_RELATIVE_PATH: Final = (
    f"{contract.LABEL_ROOT_RELATIVE_PATH}/reservation.json"
)
LABEL_BUILDER_CLAIM_RELATIVE_PATH: Final = (
    f"{contract.LABEL_ROOT_RELATIVE_PATH}/builder_claim.json"
)
LABEL_PREFLIGHT_RECEIPT_RELATIVE_PATH: Final = (
    ".generated/"
    "go2_post_action_projective_support_labels_v2_preflight_receipt.json"
)
LABEL_PREFLIGHT_RECEIPT_SCHEMA: Final = (
    f"{contract.SCHEMA_PREFIX}_label_preflight_receipt_v1"
)
LABEL_BUILDER_EXECUTION_BINDING_SCHEMA: Final = (
    "lewm_go2_post_action_projective_support_labels_v1_execution_binding_v1"
)
LABEL_RESERVATION_SCHEMA: Final = (
    "lewm_go2_post_action_projective_support_labels_v1_reservation_v1"
)
LABEL_BUILDER_CLAIM_SCHEMA: Final = (
    "lewm_go2_post_action_projective_support_labels_v1_builder_claim_v1"
)

LABEL_MASK_RELATIVE_PATHS: Final = {
    "predicted_next_corridor_masks": (
        f"{contract.LABEL_ROOT_RELATIVE_PATH}/predicted_next_corridor_masks.u1"
    ),
    "persistence_corridor_masks": (
        f"{contract.LABEL_ROOT_RELATIVE_PATH}/persistence_corridor_masks.u1"
    ),
    "projective_support_mask": (
        f"{contract.LABEL_ROOT_RELATIVE_PATH}/projective_support_mask.u1"
    ),
}
LABEL_FILE_PATHS: Final = tuple(sorted({
    *contract.LABEL_ROLE_RELATIVE_PATHS.values(),
    *LABEL_MASK_RELATIVE_PATHS.values(),
}))

IMPLEMENTATION_AUTHORS: Final = (
    "/root",
    "/root/counterfactual_label_mapping",
    "/root/joint_jepa_integration",
    "/root/probe_gate_review",
    "/root/label_boundary_fix",
    "/root/execution_authority_fix",
    "/root/attempt_runner",
    "/root/authority_source_review",
    "/root/authority_v2_adapter",
)
SOURCE_ONLY_AUTHORITY: Final = {
    "source_implementation_authorized": True,
    "synthetic_cpu_tests_authorized": True,
    "source_manifest_authorized": True,
    "independent_source_review_authorized": True,
    "generated_input_label_checkpoint_tensor_gpu_or_training_access_authorized": False,
    "navigation_g2_heldout_sealed_production_or_promotion_authorized": False,
}
REVIEW_CHECKS: Final = {
    "source_manifest_exact_and_regular_files_hash_bound": True,
    "geometry_source_closure_inherited_without_omission": True,
    "preregistration_identity_exact": True,
    "integrity_adapter_amendment_identity_exact": True,
    "label_v1_terminal_predecessor_bindings_literal_without_runtime_open": True,
    "labels_scores_metrics_and_runner_match_preregistration": True,
    "actual_execution_and_preflight_entrypoints_in_closure": True,
    "one_attempt_caps_inputs_output_and_denials_exact": True,
    "no_generated_checkpoint_tensor_or_protected_material_opened": True,
}
EXECUTION_AUTHORITY: Final = {
    "one_exact_fresh_attempt_authorized": True,
    "attempt_index": 1,
    "maximum_attempts": contract.MAXIMUM_ATTEMPTS,
    "maximum_updates": contract.MAXIMUM_UPDATES,
    "maximum_presentations": contract.MAXIMUM_PRESENTATIONS,
    "retry_or_resume_authorized": False,
    "second_seed_or_second_attempt_authorized": False,
    "output_root": contract.OUTPUT_ROOT_RELATIVE_PATH,
    "output_root_must_be_absent_before_reservation": True,
    **contract.DOWNSTREAM_DENIALS,
}
AUTHORIZATION_STATUS: Final = (
    "AUTHORIZED_ONE_EXACT_RGB_POST_ACTION_PROJECTIVE_SUPPORT_CORRIDOR_"
    "JOINT_JEPA_V1_ATTEMPT"
)

WRONG_RGB_MAPPING_ALGORITHM: Final = (
    "role_scene_local_lexicographic_cyclic_derangement_v1"
)
STATIC_MASK_EXPECTATIONS: Final = {
    "predicted_next_corridor_masks.u1": {
        "shape": [11, 64, 64],
        "byte_count": 11 * 64 * 64,
        "set_cell_count": 659,
        "file_sha256": contract.FULL_MASK_SHA256,
    },
    "persistence_corridor_masks.u1": {
        "shape": [9, 11, 64, 64],
        "byte_count": 9 * 11 * 64 * 64,
        "set_cell_count": 6_040,
        "file_sha256": contract.PERSISTENCE_MASK_STACK_SHA256,
    },
    "projective_support_mask.u1": {
        "shape": [64, 64],
        "byte_count": 64 * 64,
        "set_cell_count": 1_964,
        "file_sha256": (
            "cbcdb7d6fda08626522732ff092d90a87f5b5f2cd2534baf2bb4aa556d832753"
        ),
    },
}
ORACLE_METRIC_PREFLIGHT_CHECKS: Final = frozenset({
    "calibration_threshold_eligible",
    "calibration_threshold_exact_one",
    "calibration_precision_exact_one",
    "calibration_safe_recall_exact_one",
    "calibration_unsafe_recall_exact_one",
    "calibration_has_exact_registered_families",
    "selection_has_eight_scenes",
    "selection_has_exact_registered_families",
    "selection_precision_exact_one",
    "selection_safe_recall_exact_one",
    "selection_unsafe_recall_exact_one",
    "selection_utility_exact_one",
    "bootstrap_oracle_delta_exact_one",
    *(
        f"family:{family}:{suffix}"
        for family in contract.SCENE_FAMILIES
        for suffix in (
            "nonempty_admission",
            "nonempty_prefix_exact_one",
        )
    ),
})
LABEL_PREFLIGHT_ACCESS_LEDGER: Final = {
    "rgb_opens": 0,
    "checkpoint_opens": 0,
    "tensor_opens": 0,
    "gpu_opens": 0,
    "training_opens": 0,
    "runtime_output_opens": 0,
    "g2_opens": 0,
    "navigation_opens": 0,
    "heldout_opens": 0,
    "sealed_opens": 0,
    "production_opens": 0,
}
LABEL_PREFLIGHT_AUTHORITY: Final = {
    "training_authorized": False,
    **contract.DOWNSTREAM_DENIALS,
}
_LABEL_RESERVATION_ACCESS_LEDGER: Final = {
    name: 0
    for name in (
        "execution_binding_opens",
        "source_manifest_opens",
        "independent_source_review_opens",
        "source_authority_validation_calls",
        "raw_manifest_opens",
        "raw_pairs_opens",
        "raw_endpoints_opens",
        "raw_audit_opens",
        "geometry_contract_opens",
        "geometry_contract_validation_calls",
        "directional_policy_opens",
        "primitive_registry_opens",
        "schedule_opens",
        "scene_join_calls_started",
        "render_summary_opens",
        "source_frames_jsonl_opens",
        "scene_manifest_opens",
        "rgb_opens",
        "checkpoint_opens",
        "runtime_output_opens",
        "g2_opens",
        "navigation_opens",
        "heldout_opens",
        "sealed_opens",
        "production_opens",
    )
}
_LABEL_RESERVATION_AUTHORITY: Final = {
    "development_label_preflight_authorized": True,
    "training_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "sealed_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
}

_RUNTIME_INPUT_NAMES: Final = {
    "raw_manifest": contract.RAW_MANIFEST_RELATIVE_PATH,
    "raw_audit": contract.RAW_AUDIT_RELATIVE_PATH,
    "raw_pairs": contract.RAW_PAIRS_RELATIVE_PATH,
    "raw_endpoints": contract.RAW_ENDPOINTS_RELATIVE_PATH,
    "n320_gate": contract.N320_GATE_RELATIVE_PATH,
    "n320_encoder_checkpoint": contract.N320_CHECKPOINT_RELATIVE_PATH,
    "schedule": contract.SCHEDULE_RELATIVE_PATH,
}


def _require_hex(value: object, length: int, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or value != value.casefold()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PermissionError(f"{name} must be lowercase {length}-hex")
    return value


def _safe_source_path(value: object) -> str:
    path = _geometry.safe_relative_path(value)
    if path in {
        contract.PREREGISTRATION_RELATIVE_PATH,
        contract.INTEGRITY_ADAPTER_AMENDMENT_RELATIVE_PATH,
    }:
        return path
    return _geometry.safe_relative_source_path(path)


def _candidate_python_sources_v1(root: Path) -> tuple[str, ...]:
    """List only ordinary Python source names, honoring repository ignores."""

    package_paths = [
        relative for _, relative in _LOCAL_PACKAGE_ROOTS
        if (Path(root) / relative).is_dir()
    ]
    if not package_paths:
        raise RuntimeError("recursive source package roots are absent")
    completed = subprocess.run(
        [
            "rg",
            "--files",
            "--glob",
            "*.py",
            "--glob",
            "!**/sealed_test.json",
            "--glob",
            "!**/sealed/**",
            "--glob",
            "!**/sealed_*/**",
            "--glob",
            "!**/heldout/**",
            "--glob",
            "!**/heldout_*/**",
            "--glob",
            "!**/.generated/**",
            *package_paths,
        ],
        cwd=Path(root),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ignore-honoring recursive source discovery failed: "
            + completed.stderr.strip()
        )
    result = tuple(sorted({
        _geometry.safe_relative_source_path(line)
        for line in completed.stdout.splitlines()
        if line
    }))
    if not result:
        raise RuntimeError("recursive Python source discovery was empty")
    return result


def _module_index_v1(
    root: Path,
    *,
    candidate_paths: Sequence[str] | None = None,
) -> tuple[dict[str, Path], dict[Path, str]]:
    root = Path(root).resolve()
    candidates = (
        _candidate_python_sources_v1(root)
        if candidate_paths is None
        else tuple(candidate_paths)
    )
    by_module: dict[str, Path] = {}
    by_path: dict[Path, str] = {}
    for prefix, package_relative in _LOCAL_PACKAGE_ROOTS:
        package_root = (root / package_relative).resolve()
        for relative in candidates:
            relative = _geometry.safe_relative_source_path(relative)
            path = (root / relative).resolve()
            if not path.is_relative_to(package_root):
                continue
            module_parts = list(path.relative_to(package_root).with_suffix("").parts)
            if not module_parts:
                continue
            if module_parts[-1] == "__init__":
                module_parts.pop()
            module = ".".join((prefix, *module_parts)) if module_parts else prefix
            existing = by_module.get(module)
            if existing is not None and existing != path:
                raise RuntimeError(f"duplicate local source module: {module}")
            by_module[module] = path
            by_path[path] = module
    return by_module, by_path


def _absolute_import_base_v1(
    *,
    current_module: str,
    current_path: Path,
    node: ast.ImportFrom,
) -> str:
    if node.level == 0:
        return node.module or ""
    package = (
        current_module
        if current_path.name == "__init__.py"
        else current_module.rpartition(".")[0]
    )
    parts = package.split(".") if package else []
    if node.level > len(parts):
        return ""
    kept = parts[: len(parts) - node.level + 1]
    if node.module:
        kept.extend(node.module.split("."))
    return ".".join(kept)


def _import_candidates_v1(
    tree: ast.AST,
    *,
    current_module: str,
    current_path: Path,
) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import_base_v1(
                current_module=current_module,
                current_path=current_path,
                node=node,
            )
            if not base:
                continue
            yield base
            for alias in node.names:
                if alias.name != "*":
                    yield f"{base}.{alias.name}"


def _parent_package_paths_v1(
    module: str,
    by_module: Mapping[str, Path],
) -> Iterable[Path]:
    parts = module.split(".")
    for length in range(1, len(parts)):
        candidate = by_module.get(".".join(parts[:length]))
        if candidate is not None and candidate.name == "__init__.py":
            yield candidate


def discover_recursive_source_closure_v1(
    *,
    root: Path = ROOT,
    entrypoints: Sequence[str] = SOURCE_MANIFEST_ENTRYPOINTS,
    forced_dynamic_sources: Sequence[str] = SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES,
    candidate_paths: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the exact local recursive import closure without importing it."""

    root = Path(root).resolve()
    by_module, by_path = _module_index_v1(
        root, candidate_paths=candidate_paths
    )
    queue = [
        (root / _geometry.safe_relative_source_path(relative)).resolve()
        for relative in (*entrypoints, *forced_dynamic_sources)
    ]
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        current_module = by_path.get(path)
        if current_module is None:
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as error:
                raise PermissionError("recursive source escaped repository root") from error
            raise PermissionError(
                f"recursive source is outside fixed local modules: {relative}"
            )
        relative = path.relative_to(root).as_posix()
        _geometry.safe_relative_source_path(relative)
        try:
            source = _geometry._read_regular_source(path).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"recursive source is not UTF-8: {relative}") from error
        visited.add(path)
        tree = ast.parse(source, filename=relative)
        for module in _import_candidates_v1(
            tree,
            current_module=current_module,
            current_path=path,
        ):
            dependency = by_module.get(module)
            if dependency is None:
                continue
            queue.append(dependency)
            queue.extend(_parent_package_paths_v1(module, by_module))
    return tuple(sorted(path.relative_to(root).as_posix() for path in visited))


def validate_recursive_source_paths_v1(
    proposed: Sequence[str],
    discovered: Sequence[str],
) -> tuple[str, ...]:
    proposed_tuple = tuple(proposed)
    discovered_tuple = tuple(discovered)
    if (
        proposed_tuple != tuple(sorted(set(proposed_tuple)))
        or discovered_tuple != tuple(sorted(set(discovered_tuple)))
        or proposed_tuple != discovered_tuple
    ):
        missing = sorted(set(discovered_tuple) - set(proposed_tuple))
        stale = sorted(set(proposed_tuple) - set(discovered_tuple))
        raise PermissionError(
            f"recursive source closure changed: missing={missing}, stale={stale}"
        )
    return proposed_tuple


RECURSIVE_PYTHON_SOURCE_PATHS: Final = discover_recursive_source_closure_v1()
SOURCE_PATHS: Final = tuple(sorted({
    *RECURSIVE_PYTHON_SOURCE_PATHS,
    contract.PREREGISTRATION_RELATIVE_PATH,
    contract.INTEGRITY_ADAPTER_AMENDMENT_RELATIVE_PATH,
}))


def _regular_file_binding(
    path: str,
    *,
    root: Path,
    content_sha256: str | None = None,
) -> dict[str, Any]:
    _geometry.safe_relative_path(path)
    raw = _geometry._read_regular_source(Path(root) / path)
    return _geometry.artifact_binding(
        path,
        raw,
        content_sha256=content_sha256,
    )


def _artifact_binding(
    value: object,
    *,
    path: str,
    require_content: bool = False,
) -> dict[str, Any]:
    _geometry.safe_relative_path(path)
    required = {"path", "file_sha256", "byte_count"}
    allowed = {*required, "content_sha256"}
    if (
        type(value) is not dict
        or not required.issubset(value)
        or not set(value).issubset(allowed)
        or value.get("path") != path
        or not _geometry.is_sha256(value.get("file_sha256"))
        or type(value.get("byte_count")) is not int
        or value["byte_count"] <= 0
        or (require_content and "content_sha256" not in value)
        or (
            "content_sha256" in value
            and not _geometry.is_sha256(value.get("content_sha256"))
        )
    ):
        raise PermissionError(f"artifact binding changed: {path}")
    return dict(value)


def canonical_document_bytes(value: Mapping[str, Any]) -> bytes:
    return contract.canonical_json_bytes(dict(value)) + b"\n"


def preregistration_binding() -> dict[str, Any]:
    return contract.preregistration_binding()


def runtime_input_bindings() -> dict[str, dict[str, Any]]:
    return {
        name: copy.deepcopy(contract.RUNTIME_BINDINGS[path])
        for name, path in _RUNTIME_INPUT_NAMES.items()
    }


def geometry_input_bindings() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(contract.GEOMETRY_BINDINGS)


def build_source_manifest(*, root: Path = ROOT) -> dict[str, Any]:
    discovered = discover_recursive_source_closure_v1(
        root=root,
        entrypoints=SOURCE_MANIFEST_ENTRYPOINTS,
        forced_dynamic_sources=SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES,
    )
    validate_recursive_source_paths_v1(
        RECURSIVE_PYTHON_SOURCE_PATHS,
        discovered,
    )
    expected_paths = tuple(sorted({
        *discovered,
        contract.PREREGISTRATION_RELATIVE_PATH,
        contract.INTEGRITY_ADAPTER_AMENDMENT_RELATIVE_PATH,
    }))
    if (
        set(INHERITED_GEOMETRY_SOURCE_PATHS) != set(_geometry.SOURCE_PATHS)
        or SOURCE_PATHS != expected_paths
        or not set(SOURCE_MANIFEST_ENTRYPOINTS).issubset(
            SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
        )
        or not set(INHERITED_GEOMETRY_SOURCE_PATHS).issubset(
            SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
        )
    ):
        raise PermissionError("source closure layout changed")
    bindings = [
        _regular_file_binding(_safe_source_path(path), root=root)
        for path in SOURCE_PATHS
    ]
    preregistration = preregistration_binding()
    prereg_source = next(
        (row for row in bindings if row["path"] == preregistration["path"]),
        None,
    )
    if prereg_source != {
        "path": preregistration["path"],
        "file_sha256": preregistration["file_sha256"],
        "byte_count": preregistration["byte_count"],
    }:
        raise PermissionError("governing preregistration identity changed")
    amendment = contract.integrity_adapter_amendment_binding()
    amendment_source = next(
        (row for row in bindings if row["path"] == amendment["path"]),
        None,
    )
    if amendment_source != amendment:
        raise PermissionError("integrity-adapter amendment identity changed")
    predecessor_bindings = copy.deepcopy(
        contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS
    )
    return contract.with_content_sha256({
        "schema": contract.SOURCE_MANIFEST_SCHEMA,
        "status": "PASS_SOURCE_CLOSURE",
        "entrypoints": list(SOURCE_MANIFEST_ENTRYPOINTS),
        "forced_dynamic_sources": list(
            SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
        ),
        "inherited_geometry_source_paths": list(INHERITED_GEOMETRY_SOURCE_PATHS),
        "inherited_geometry_source_paths_sha256": contract.canonical_json_sha256(
            list(INHERITED_GEOMETRY_SOURCE_PATHS)
        ),
        "additive_source_paths": list(ADDITIVE_SOURCE_PATHS),
        "recursive_python_source_paths": list(
            RECURSIVE_PYTHON_SOURCE_PATHS
        ),
        "recursive_python_source_paths_sha256": (
            contract.canonical_json_sha256(
                list(RECURSIVE_PYTHON_SOURCE_PATHS)
            )
        ),
        "source_paths": list(SOURCE_PATHS),
        "source_bindings": bindings,
        "source_bindings_sha256": contract.canonical_json_sha256(bindings),
        "source_count": len(bindings),
        "preregistration": preregistration,
        "integrity_adapter_amendment": amendment,
        "label_v1_terminal_predecessor_bindings": predecessor_bindings,
        "generated_input_open_count": 0,
        "checkpoint_or_tensor_open_count": 0,
        "heldout_or_sealed_open_count": 0,
        "whole_tree_discovery_or_export_authorized": False,
        "authority": dict(SOURCE_ONLY_AUTHORITY),
    })


def validate_source_manifest(
    raw: bytes,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = contract.parse_canonical_json(raw, name="source manifest")
    if value != build_source_manifest(root=root):
        raise PermissionError("source manifest is not the exact current closure")
    return value


def source_manifest_binding(raw: bytes) -> dict[str, Any]:
    parsed = contract.parse_canonical_json(raw, name="source manifest")
    return _geometry.artifact_binding(
        SOURCE_MANIFEST_RELATIVE_PATH,
        raw,
        content_sha256=parsed["content_sha256"],
    )


def _review_identity(value: object, *, field: str) -> str:
    if type(value) is not str or not value.startswith("/root/"):
        raise PermissionError(f"{field} must identify one /root/ agent")
    return value


def build_source_review_receipt(
    source_manifest_raw: bytes,
    *,
    reviewer: str,
    source_freeze_commit: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    manifest = validate_source_manifest(source_manifest_raw, root=root)
    reviewer = _review_identity(reviewer, field="reviewer")
    if reviewer in IMPLEMENTATION_AUTHORS:
        raise PermissionError("source reviewer is not independent")
    source_freeze_commit = _require_hex(
        source_freeze_commit, 40, name="source_freeze_commit"
    )
    return contract.with_content_sha256({
        "schema": contract.SOURCE_REVIEW_SCHEMA,
        "status": "PASS_SOURCE_AND_SCIENCE",
        "implementation_authors": list(IMPLEMENTATION_AUTHORS),
        "reviewer": reviewer,
        "source_freeze_commit": source_freeze_commit,
        "source_manifest": source_manifest_binding(source_manifest_raw),
        "reviewed_source_count": manifest["source_count"],
        "reviewed_source_bindings_sha256": manifest["source_bindings_sha256"],
        "preregistration": preregistration_binding(),
        "integrity_adapter_amendment": copy.deepcopy(
            manifest["integrity_adapter_amendment"]
        ),
        "label_v1_terminal_predecessor_bindings": copy.deepcopy(
            manifest["label_v1_terminal_predecessor_bindings"]
        ),
        "science_contract": contract.science_contract(),
        "source_only_checks": {
            "generated_inputs_opened": [],
            "labels_opened": [],
            "checkpoints_or_tensors_opened": [],
            "runtime_outputs_or_traces_opened": [],
            "heldout_or_sealed_opened": [],
        },
        "scientific_checks": dict(REVIEW_CHECKS),
        "findings": [],
        "authority": dict(SOURCE_ONLY_AUTHORITY),
    })


def validate_source_review_receipt(
    raw: bytes,
    source_manifest_raw: bytes,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = contract.parse_canonical_json(raw, name="source review")
    expected = build_source_review_receipt(
        source_manifest_raw,
        reviewer=value.get("reviewer"),
        source_freeze_commit=value.get("source_freeze_commit"),
        root=root,
    )
    if value != expected:
        raise PermissionError("source review receipt changed")
    return value


def source_review_binding(raw: bytes) -> dict[str, Any]:
    parsed = contract.parse_canonical_json(raw, name="source review")
    return _geometry.artifact_binding(
        SOURCE_REVIEW_RELATIVE_PATH,
        raw,
        content_sha256=parsed["content_sha256"],
    )


def _label_builder_execution_binding(raw: bytes) -> dict[str, Any]:
    value = contract.parse_canonical_json(
        raw, name="label-builder execution binding"
    )
    authority = value.get("authority")
    source_records = value.get("source_records")
    if (
        value.get("schema") != LABEL_BUILDER_EXECUTION_BINDING_SCHEMA
        or value.get("status")
        != "AUTHORIZED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT"
        or value.get("preregistration_commit") != contract.PREREGISTRATION_COMMIT
        or value.get("integrity_adapter_amendment")
        != contract.integrity_adapter_amendment_binding()
        or value.get("label_v1_terminal_predecessor_bindings")
        != contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS
        or value.get("output_directory") != contract.LABEL_ROOT_RELATIVE_PATH
        or value.get("schedule_prefix_sha256") != contract.SCHEDULE_PREFIX_SHA256
        or type(source_records) is not list
        or len(source_records) != 264
        or type(authority) is not dict
        or authority.get("development_label_preflight_authorized") is not True
        or any(
            flag is True
            for name, flag in authority.items()
            if name != "development_label_preflight_authorized"
            and str(name).endswith("_authorized")
        )
    ):
        raise PermissionError("label-builder execution binding changed")
    _artifact_binding(
        value.get("source_manifest"),
        path=SOURCE_MANIFEST_RELATIVE_PATH,
        require_content=True,
    )
    _artifact_binding(
        value.get("independent_source_review"),
        path=SOURCE_REVIEW_RELATIVE_PATH,
        require_content=True,
    )
    return value


def label_builder_execution_binding_binding(raw: bytes) -> dict[str, Any]:
    parsed = _label_builder_execution_binding(raw)
    return _geometry.artifact_binding(
        LABEL_BUILDER_EXECUTION_BINDING_RELATIVE_PATH,
        raw,
        content_sha256=parsed["content_sha256"],
    )


def _exact_integer_map(
    value: object,
    keys: set[str],
    *,
    name: str,
    minimum: int,
) -> dict[str, int]:
    if type(value) is not dict or set(value) != keys:
        raise PermissionError(f"{name} key set changed")
    result: dict[str, int] = {}
    for key in sorted(keys):
        item = value[key]
        if type(item) is not int or item < minimum:
            raise PermissionError(f"{name}.{key} changed")
        result[key] = item
    return result


def _structural_preflight(value: object) -> dict[str, Any]:
    expected_keys = {
        "exact_state_count",
        "exact_action_row_count",
        "exact_station_label_count",
        "informative_state_counts",
        "train_action_ranking_participation_counts",
        "selection_family_informative_counts",
        "role_scene_and_endpoint_disjoint",
        "role_scene_counts",
        "minimum_states_per_role_scene",
        "safe_unsafe_support",
        "every_non_hold_action_station_has_safe_and_unsafe_support",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise PermissionError("structural label-preflight check set changed")
    if (
        value["exact_state_count"] != contract.TOTAL_STATES
        or value["exact_action_row_count"] != contract.TOTAL_ACTION_ROWS
        or value["exact_station_label_count"] != contract.TOTAL_STATION_LABELS
        or value["role_scene_and_endpoint_disjoint"] is not True
        or value["every_non_hold_action_station_has_safe_and_unsafe_support"]
        is not True
    ):
        raise PermissionError("structural label-preflight population did not pass")

    role_keys = set(contract.ROLE_ORDER)
    informative = _exact_integer_map(
        value["informative_state_counts"],
        role_keys,
        name="informative_state_counts",
        minimum=0,
    )
    if (
        informative["train"] < 512
        or informative["probability_calibration"] < 128
        or informative["checkpoint_selection"] < 128
    ):
        raise PermissionError("informative-state label-preflight gate did not pass")
    non_hold = set(contract.ACTION_VOCABULARY) - {"hold"}
    _exact_integer_map(
        value["train_action_ranking_participation_counts"],
        non_hold,
        name="train_action_ranking_participation_counts",
        minimum=1,
    )
    _exact_integer_map(
        value["selection_family_informative_counts"],
        set(contract.SCENE_FAMILIES),
        name="selection_family_informative_counts",
        minimum=8,
    )
    role_scene_counts = _exact_integer_map(
        value["role_scene_counts"],
        role_keys,
        name="role_scene_counts",
        minimum=1,
    )
    if role_scene_counts != {
        role: contract.ROLE_COUNTS[role]["scenes"] for role in contract.ROLE_ORDER
    }:
        raise PermissionError("role scene-count label-preflight gate did not pass")
    _exact_integer_map(
        value["minimum_states_per_role_scene"],
        role_keys,
        name="minimum_states_per_role_scene",
        minimum=2,
    )

    support = value["safe_unsafe_support"]
    if type(support) is not dict or set(support) != {
        "train",
        "calibration_plus_selection",
    }:
        raise PermissionError("safe/unsafe support population changed")
    for population_name, population in support.items():
        if type(population) is not dict or set(population) != non_hold:
            raise PermissionError(
                f"safe/unsafe support action set changed: {population_name}"
            )
        for action, stations in population.items():
            if type(stations) is not list or len(stations) != contract.STATION_COUNT:
                raise PermissionError(
                    f"safe/unsafe support station set changed: {population_name}/{action}"
                )
            for station in stations:
                if (
                    type(station) is not dict
                    or set(station) != {"safe", "unsafe"}
                    or type(station["safe"]) is not int
                    or type(station["unsafe"]) is not int
                    or station["safe"] <= 0
                    or station["unsafe"] <= 0
                ):
                    raise PermissionError(
                        "safe/unsafe support label-preflight gate did not pass"
                    )
    return copy.deepcopy(value)


def _schedule_preflight(value: object) -> dict[str, Any]:
    expected_keys = {
        "presentation_count",
        "presentation_indices_sha256",
        "informative_presentation_count",
        "ranking_participation_presentations_by_action",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise PermissionError("schedule label-preflight check set changed")
    if (
        value["presentation_count"] != contract.MAXIMUM_PRESENTATIONS
        or value["presentation_indices_sha256"]
        != contract.SCHEDULE_PREFIX_SHA256
        or type(value["informative_presentation_count"]) is not int
        or value["informative_presentation_count"] < 512
    ):
        raise PermissionError("16k schedule label-preflight gate did not pass")
    _exact_integer_map(
        value["ranking_participation_presentations_by_action"],
        set(contract.ACTION_VOCABULARY) - {"hold"},
        name="ranking_participation_presentations_by_action",
        minimum=32,
    )
    return copy.deepcopy(value)


def _oracle_metric_pipeline(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "status",
        "passed",
        "failed_checks",
        "checks",
    }:
        raise PermissionError("oracle metric-pipeline receipt shape changed")
    checks = value.get("checks")
    if (
        value.get("status") != "PASS"
        or value.get("passed") is not True
        or value.get("failed_checks") != []
        or type(checks) is not dict
        or set(checks) != ORACLE_METRIC_PREFLIGHT_CHECKS
        or any(item is not True for item in checks.values())
    ):
        raise PermissionError("oracle metric-pipeline preflight did not pass exactly")
    return copy.deepcopy(value)


def _wrong_rgb_mapping(value: object) -> dict[str, Any]:
    expected_keys = {
        "algorithm",
        "roles",
        "row_count",
        "mapping_sha256",
        "per_role",
        "paired_next_collision_count",
        "paired_next_collision_rows_sha256",
        "mapped_endpoint_is_never_paired_next",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise PermissionError("wrong-RGB mapping summary shape changed")
    per_role = value.get("per_role")
    if (
        value.get("algorithm") != WRONG_RGB_MAPPING_ALGORITHM
        or value.get("roles") != list(contract.ROLE_ORDER)
        or value.get("row_count") != contract.TOTAL_STATES
        or not _geometry.is_sha256(value.get("mapping_sha256"))
        or type(per_role) is not dict
        or set(per_role) != set(contract.ROLE_ORDER)
        or value.get("paired_next_collision_count") != 0
        or value.get("paired_next_collision_rows_sha256")
        != contract.canonical_json_sha256([])
        or value.get("mapped_endpoint_is_never_paired_next") is not True
    ):
        raise PermissionError("wrong-RGB mapping summary changed")
    normalized_roles: dict[str, dict[str, Any]] = {}
    for role in contract.ROLE_ORDER:
        record = per_role[role]
        if (
            type(record) is not dict
            or set(record) != {"row_count", "mapping_sha256"}
            or record.get("row_count") != contract.ROLE_COUNTS[role]["states"]
            or not _geometry.is_sha256(record.get("mapping_sha256"))
        ):
            raise PermissionError(f"wrong-RGB role summary changed: {role}")
        normalized_roles[role] = dict(record)
    return {
        "algorithm": WRONG_RGB_MAPPING_ALGORITHM,
        "roles": list(contract.ROLE_ORDER),
        "row_count": contract.TOTAL_STATES,
        "mapping_sha256": value["mapping_sha256"],
        "per_role": normalized_roles,
        "paired_next_collision_count": 0,
        "paired_next_collision_rows_sha256": contract.canonical_json_sha256([]),
        "mapped_endpoint_is_never_paired_next": True,
    }


def _action_prior(value: object) -> dict[str, Any]:
    expected_keys = {
        "source_role",
        "source_roles",
        "source_state_count",
        "action_order",
        "station_count",
        "shape",
        "probabilities",
        "probabilities_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise PermissionError("action-prior summary shape changed")
    probabilities = value.get("probabilities")
    if (
        value.get("source_role") != "train"
        or value.get("source_roles") != ["train"]
        or value.get("source_state_count")
        != contract.ROLE_COUNTS["train"]["states"]
        or value.get("action_order") != list(contract.ACTION_VOCABULARY)
        or value.get("station_count") != contract.STATION_COUNT
        or value.get("shape") != [len(contract.ACTION_VOCABULARY), contract.STATION_COUNT]
        or type(probabilities) is not list
        or len(probabilities) != len(contract.ACTION_VOCABULARY)
    ):
        raise PermissionError("train-only action-prior identity changed")
    normalized: list[list[float]] = []
    for action_index, row in enumerate(probabilities):
        if type(row) is not list or len(row) != contract.STATION_COUNT:
            raise PermissionError("action-prior probability shape changed")
        normalized_row: list[float] = []
        for probability in row:
            if (
                type(probability) is not float
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
                or (
                    action_index in contract.NON_HOLD_ACTION_INDICES
                    and not 0.0 < probability < 1.0
                )
            ):
                raise PermissionError("action-prior probability changed")
            normalized_row.append(probability)
        normalized.append(normalized_row)
    if value.get("probabilities_sha256") != contract.canonical_json_sha256(
        normalized
    ):
        raise PermissionError("action-prior probability hash changed")
    result = dict(value)
    result["probabilities"] = normalized
    return result


def _label_materialization_chain(
    manifest: Mapping[str, Any],
    label_builder: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate reservation -> claim -> builder -> manifest provenance exactly."""

    provenance = manifest.get("input_bindings")
    if type(provenance) is not dict:
        raise PermissionError("label manifest input provenance is absent")
    required = {
        "label_reservation",
        "label_builder_claim",
        "integrity_adapter_amendment",
        "label_v1_terminal_predecessor_bindings",
        "source_manifest",
        "independent_source_review",
        "execution_binding_content_sha256",
        "source_records_sha256",
        "schedule_prefix_sha256",
    }
    if not required.issubset(provenance):
        raise PermissionError("label manifest provenance fields are incomplete")
    if (
        manifest.get("preregistration_commit")
        != contract.PREREGISTRATION_COMMIT
        or provenance.get("integrity_adapter_amendment")
        != contract.integrity_adapter_amendment_binding()
        or provenance.get("integrity_adapter_amendment")
        != label_builder.get("integrity_adapter_amendment")
        or provenance.get("label_v1_terminal_predecessor_bindings")
        != contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS
        or provenance.get("label_v1_terminal_predecessor_bindings")
        != label_builder.get("label_v1_terminal_predecessor_bindings")
        or provenance.get("source_manifest")
        != label_builder.get("source_manifest")
        or provenance.get("independent_source_review")
        != label_builder.get("independent_source_review")
        or provenance.get("execution_binding_content_sha256")
        != label_builder.get("content_sha256")
        or provenance.get("source_records_sha256")
        != contract.canonical_json_sha256(label_builder.get("source_records"))
        or provenance.get("schedule_prefix_sha256")
        != contract.SCHEDULE_PREFIX_SHA256
        or label_builder.get("schedule_prefix_sha256")
        != contract.SCHEDULE_PREFIX_SHA256
    ):
        raise PermissionError("label manifest escaped its builder provenance")

    expected_reservation = contract.with_content_sha256({
        "schema": LABEL_RESERVATION_SCHEMA,
        "status": "RESERVED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT",
        "preregistration": preregistration_binding(),
        "execution_binding_path": LABEL_BUILDER_EXECUTION_BINDING_RELATIVE_PATH,
        "source_manifest": copy.deepcopy(label_builder["source_manifest"]),
        "independent_source_review": copy.deepcopy(
            label_builder["independent_source_review"]
        ),
        "output_directory": contract.LABEL_ROOT_RELATIVE_PATH,
        "attempt": {
            "index": 1,
            "maximum_attempts": 1,
            "retry_authorized": False,
            "resume_authorized": False,
            "second_invocation_authorized": False,
        },
        "access_ledger": dict(_LABEL_RESERVATION_ACCESS_LEDGER),
        "authority": dict(_LABEL_RESERVATION_AUTHORITY),
    })
    if provenance.get("label_reservation") != expected_reservation:
        raise PermissionError("label reservation escaped the label manifest")
    expected_claim = contract.with_content_sha256({
        "schema": LABEL_BUILDER_CLAIM_SCHEMA,
        "status": "CLAIMED_ONE_EXACT_LABEL_BUILDER_INVOCATION",
        "reservation_content_sha256": expected_reservation["content_sha256"],
        "execution_binding_content_sha256": label_builder["content_sha256"],
        "retry_authorized": False,
        "resume_authorized": False,
        "second_invocation_authorized": False,
    })
    if provenance.get("label_builder_claim") != expected_claim:
        raise PermissionError("label builder claim escaped the label manifest")

    reservation_raw = canonical_document_bytes(expected_reservation)
    claim_raw = canonical_document_bytes(expected_claim)
    return {
        "preregistration": preregistration_binding(),
        "integrity_adapter_amendment": copy.deepcopy(
            provenance["integrity_adapter_amendment"]
        ),
        "label_v1_terminal_predecessor_bindings": copy.deepcopy(
            provenance["label_v1_terminal_predecessor_bindings"]
        ),
        "source_manifest": copy.deepcopy(label_builder["source_manifest"]),
        "independent_source_review": copy.deepcopy(
            label_builder["independent_source_review"]
        ),
        "label_builder_execution_binding_content_sha256": label_builder[
            "content_sha256"
        ],
        "source_records_sha256": provenance["source_records_sha256"],
        "schedule_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256,
        "label_reservation": _geometry.artifact_binding(
            LABEL_RESERVATION_RELATIVE_PATH,
            reservation_raw,
            content_sha256=expected_reservation["content_sha256"],
        ),
        "label_builder_claim": _geometry.artifact_binding(
            LABEL_BUILDER_CLAIM_RELATIVE_PATH,
            claim_raw,
            content_sha256=expected_claim["content_sha256"],
        ),
    }


def _label_bundle(
    label_manifest_raw: bytes,
    label_file_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = contract.parse_canonical_json(label_manifest_raw, name="label manifest")
    if (
        manifest.get("schema") != contract.LABEL_MANIFEST_SCHEMA
        or manifest.get("status") != "complete_pre_gpu_development_labels"
        or manifest.get("preregistration_commit")
        != contract.PREREGISTRATION_COMMIT
        or manifest.get("roles") != list(contract.ROLE_ORDER)
        or manifest.get("action_order") != list(contract.ACTION_VOCABULARY)
        or manifest.get("state_count") != contract.TOTAL_STATES
        or manifest.get("action_row_count") != contract.TOTAL_ACTION_ROWS
        or manifest.get("station_label_count") != contract.TOTAL_STATION_LABELS
    ):
        raise PermissionError("label manifest population or schema changed")
    records = manifest.get("files")
    if type(records) is not list:
        raise PermissionError("label manifest file records are absent")
    by_name = {
        record.get("path"): record
        for record in records
        if type(record) is dict and type(record.get("path")) is str
    }
    if len(by_name) != len(records):
        raise PermissionError("label manifest repeats or malforms a file record")
    expected_names = {PurePosixPath(path).name for path in LABEL_FILE_PATHS}
    if set(by_name) != expected_names or set(label_file_bindings) != set(LABEL_FILE_PATHS):
        raise PermissionError("label bundle file set changed")
    normalized: dict[str, dict[str, Any]] = {}
    for path in LABEL_FILE_PATHS:
        record = by_name[PurePosixPath(path).name]
        binding = _artifact_binding(label_file_bindings[path], path=path)
        if (
            record.get("file_sha256") != binding["file_sha256"]
            or record.get("byte_count") != binding["byte_count"]
        ):
            raise PermissionError(f"label file escaped its manifest: {path}")
        normalized[path] = binding
    for role in contract.ROLE_ORDER:
        path = contract.LABEL_ROLE_RELATIVE_PATHS[role]
        record = by_name[PurePosixPath(path).name]
        counts = contract.ROLE_COUNTS[role]
        if (
            record.get("schema") != contract.LABEL_ROW_SCHEMA
            or record.get("dataset_role") != role
            or record.get("state_count") != counts["states"]
            or record.get("action_row_count") != counts["action_rows"]
        ):
            raise PermissionError(f"label role record changed: {role}")
    for name, expected in STATIC_MASK_EXPECTATIONS.items():
        record = by_name[name]
        path = next(
            path for path in LABEL_MASK_RELATIVE_PATHS.values()
            if PurePosixPath(path).name == name
        )
        if (
            record.get("dtype") != "|u1"
            or record.get("shape") != expected["shape"]
            or record.get("byte_count") != expected["byte_count"]
            or record.get("set_cell_count") != expected["set_cell_count"]
            or record.get("file_sha256") != expected["file_sha256"]
            or normalized[path]["file_sha256"] != expected["file_sha256"]
            or normalized[path]["byte_count"] != expected["byte_count"]
        ):
            raise PermissionError(f"label mask record changed: {name}")
    return {
        "manifest": _geometry.artifact_binding(
            contract.LABEL_MANIFEST_RELATIVE_PATH,
            label_manifest_raw,
            content_sha256=manifest["content_sha256"],
        ),
        "files": normalized,
    }


def build_label_preflight_receipt(
    label_builder_execution_binding_raw: bytes,
    label_manifest_raw: bytes,
    label_file_bindings: Mapping[str, Mapping[str, Any]],
    *,
    oracle_metric_pipeline: Mapping[str, Any],
    wrong_rgb_mapping: Mapping[str, Any],
    action_prior: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the sole canonical, data-only receipt that can unlock execution."""

    label_builder = _label_builder_execution_binding(
        label_builder_execution_binding_raw
    )
    label_bundle = _label_bundle(label_manifest_raw, label_file_bindings)
    manifest = contract.parse_canonical_json(
        label_manifest_raw, name="label manifest"
    )
    materialization_chain = _label_materialization_chain(
        manifest,
        label_builder,
    )
    manifest_preflight = manifest.get("preflight")
    if type(manifest_preflight) is not dict or "frozen_schedule" not in manifest_preflight:
        raise PermissionError("label manifest lacks the frozen preflight checks")
    structural = {
        key: copy.deepcopy(value)
        for key, value in manifest_preflight.items()
        if key != "frozen_schedule"
    }
    return contract.with_content_sha256({
        "schema": LABEL_PREFLIGHT_RECEIPT_SCHEMA,
        "status": "PASS_LABEL_PREFLIGHT",
        "label_builder_execution_binding": (
            label_builder_execution_binding_binding(
                label_builder_execution_binding_raw
            )
        ),
        "label_materialization_chain": materialization_chain,
        "label_bundle": label_bundle,
        "structural_preflight": {
            "status": "PASS",
            "checks": _structural_preflight(structural),
        },
        "schedule_preflight": {
            "status": "PASS",
            "checks": _schedule_preflight(
                manifest_preflight["frozen_schedule"]
            ),
        },
        "oracle_metric_pipeline": _oracle_metric_pipeline(
            oracle_metric_pipeline
        ),
        "wrong_rgb_mapping": _wrong_rgb_mapping(wrong_rgb_mapping),
        "action_prior": _action_prior(action_prior),
        "access_ledger": dict(LABEL_PREFLIGHT_ACCESS_LEDGER),
        "authority": dict(LABEL_PREFLIGHT_AUTHORITY),
    })


def validate_label_preflight_receipt(
    raw: bytes,
    label_builder_execution_binding_raw: bytes,
    label_manifest_raw: bytes,
    label_file_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = contract.parse_canonical_json(raw, name="label-preflight receipt")
    expected = build_label_preflight_receipt(
        label_builder_execution_binding_raw,
        label_manifest_raw,
        label_file_bindings,
        oracle_metric_pipeline=value.get("oracle_metric_pipeline"),
        wrong_rgb_mapping=value.get("wrong_rgb_mapping"),
        action_prior=value.get("action_prior"),
    )
    if value != expected:
        raise PermissionError("label-preflight receipt changed")
    return value


def label_preflight_receipt_binding(raw: bytes) -> dict[str, Any]:
    parsed = contract.parse_canonical_json(raw, name="label-preflight receipt")
    if (
        parsed.get("schema") != LABEL_PREFLIGHT_RECEIPT_SCHEMA
        or parsed.get("status") != "PASS_LABEL_PREFLIGHT"
    ):
        raise PermissionError("label-preflight receipt identity changed")
    return _geometry.artifact_binding(
        LABEL_PREFLIGHT_RECEIPT_RELATIVE_PATH,
        raw,
        content_sha256=parsed["content_sha256"],
    )


def build_execution_binding(
    source_manifest_raw: bytes,
    source_review_raw: bytes,
    label_manifest_raw: bytes,
    label_file_bindings: Mapping[str, Mapping[str, Any]],
    *,
    label_builder_execution_binding_raw: bytes,
    label_preflight_receipt_raw: bytes,
    authorizer: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    review = validate_source_review_receipt(
        source_review_raw, source_manifest_raw, root=root
    )
    authorizer = _review_identity(authorizer, field="authorizer")
    if authorizer in {*IMPLEMENTATION_AUTHORS, review["reviewer"]}:
        raise PermissionError("execution authorizer is not independent")
    label_builder = _label_builder_execution_binding(
        label_builder_execution_binding_raw
    )
    if (
        label_builder.get("source_manifest")
        != source_manifest_binding(source_manifest_raw)
        or label_builder.get("independent_source_review")
        != source_review_binding(source_review_raw)
    ):
        raise PermissionError(
            "label-builder authority is not rooted in the reviewed source closure"
        )
    receipt = validate_label_preflight_receipt(
        label_preflight_receipt_raw,
        label_builder_execution_binding_raw,
        label_manifest_raw,
        label_file_bindings,
    )
    return contract.with_content_sha256({
        "schema": contract.EXECUTION_BINDING_SCHEMA,
        "status": AUTHORIZATION_STATUS,
        "authorizer": authorizer,
        "source_freeze_commit": review["source_freeze_commit"],
        "source_manifest": source_manifest_binding(source_manifest_raw),
        "independent_source_review": source_review_binding(source_review_raw),
        "preregistration": preregistration_binding(),
        "integrity_adapter_amendment": copy.deepcopy(
            review["integrity_adapter_amendment"]
        ),
        "label_v1_terminal_predecessor_bindings": copy.deepcopy(
            review["label_v1_terminal_predecessor_bindings"]
        ),
        "label_bundle": _label_bundle(label_manifest_raw, label_file_bindings),
        "label_preflight_receipt": label_preflight_receipt_binding(
            label_preflight_receipt_raw
        ),
        "wrong_rgb_mapping": copy.deepcopy(receipt["wrong_rgb_mapping"]),
        "runtime_inputs": runtime_input_bindings(),
        "geometry_inputs": geometry_input_bindings(),
        "runtime": {
            "interpreter_path": contract.RUNTIME_INTERPRETER_PATH,
            "sys_prefix": contract.RUNTIME_SYS_PREFIX,
        },
        "seeds": {
            "initialization": contract.INITIALIZATION_SEED,
            "schedule": contract.SCHEDULE_SEED,
            "experiment": contract.EXPERIMENT_SEED,
            "bootstrap": contract.BOOTSTRAP_SEED,
        },
        "schedule_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256,
        "caps": {
            "attempts": contract.MAXIMUM_ATTEMPTS,
            "updates": contract.MAXIMUM_UPDATES,
            "presentations": contract.MAXIMUM_PRESENTATIONS,
            "microbatch_size": contract.MICROBATCH_SIZE,
            "microbatches_per_update": contract.MICROBATCHES_PER_UPDATE,
            "effective_batch_size": contract.EFFECTIVE_BATCH_SIZE,
            "target_ema_momentum": contract.TARGET_EMA_MOMENTUM,
        },
        "output_root": contract.OUTPUT_ROOT_RELATIVE_PATH,
        "attempt": {
            "index": 1,
            "maximum_attempts": 1,
            "fresh": True,
            "retry": False,
            "resume": False,
        },
        "science_contract": contract.science_contract(),
        "authority": dict(EXECUTION_AUTHORITY),
        "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
    })


def validate_execution_binding(
    raw: bytes,
    source_manifest_raw: bytes,
    source_review_raw: bytes,
    label_manifest_raw: bytes,
    label_file_bindings: Mapping[str, Mapping[str, Any]],
    *,
    label_builder_execution_binding_raw: bytes,
    label_preflight_receipt_raw: bytes,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = contract.parse_canonical_json(raw, name="execution binding")
    expected = build_execution_binding(
        source_manifest_raw,
        source_review_raw,
        label_manifest_raw,
        label_file_bindings,
        label_builder_execution_binding_raw=label_builder_execution_binding_raw,
        label_preflight_receipt_raw=label_preflight_receipt_raw,
        authorizer=value.get("authorizer"),
        root=root,
    )
    if value != expected:
        raise PermissionError("execution binding changed")
    return value


__all__ = [name for name in globals() if name.isupper()] + [
    "build_execution_binding",
    "build_label_preflight_receipt",
    "build_source_manifest",
    "build_source_review_receipt",
    "canonical_document_bytes",
    "discover_recursive_source_closure_v1",
    "geometry_input_bindings",
    "label_builder_execution_binding_binding",
    "label_preflight_receipt_binding",
    "preregistration_binding",
    "runtime_input_bindings",
    "source_manifest_binding",
    "source_review_binding",
    "validate_execution_binding",
    "validate_label_preflight_receipt",
    "validate_recursive_source_paths_v1",
    "validate_source_manifest",
    "validate_source_review_receipt",
]
