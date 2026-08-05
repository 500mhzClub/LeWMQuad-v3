#!/usr/bin/env python3
"""Build, but never execute, the Genesis ROCm qualification authority."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_plan as plan_builder  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_authority as predecessor_authority  # noqa: E402
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1 as qualifier  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1 as runner  # noqa: E402


AUTHORITY_OUTPUT = qualifier.QUALIFICATION_AUTHORITY
ROCM_BACKEND_SOURCE_AUDIT = {
    "audit_passed": True,
    "material_backend_successor_not_vulkan_v4_or_cpu_retry": True,
    "scientific_plan_changes_limited_to_identity_runtime_backend_and_successor_contract": True,
    "genesis_version_exact_0_4_6": True,
    "genesis_backend_exact_amdgpu": True,
    "hip_device_exact_r9700_gfx1201": True,
    "rocm_7_1_1_lld20_with_torch_rocm7_2_exactly_bound": True,
    "host_rocm_ld_library_path_forbidden": True,
    "hsa_override_gfx_version_forbidden": True,
    "role_local_fresh_quadrants_cache_required": True,
    "one_scene_fresh_process_policy_unchanged_for_science": True,
    "data_panel_model_arms_seeds_updates_evaluation_and_gates_unchanged": True,
    "physics_numerics_may_differ_from_genesis_0_3_14_vulkan": True,
    "qualification_is_separate_non_scientific_two_probe_attempt": True,
    "qualification_outputs_forbidden_from_scientific_reuse": True,
    "consumed_cpu_qualification_terminal_review_exactly_bound": True,
    "direct_successor_and_focused_runtime_dependency_files_bound": True,
}


class GenesisRocmBackendAuthorityError(RuntimeError):
    """Raised before an incomplete ROCm authority can be emitted."""


def file_binding(path: Path) -> dict[str, object]:
    try:
        return runner.file_binding_v1(path)
    except (OSError, RuntimeError) as exc:
        raise GenesisRocmBackendAuthorityError(str(exc)) from exc


def _require_binding(
    value: object, *, path: Path, label: str
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or value.get("path") != str(path.resolve())
        or file_binding(path) != dict(value)
    ):
        raise GenesisRocmBackendAuthorityError(f"{label} binding changed")
    return dict(value)


def _cpu_terminal_review_binding() -> dict[str, Any]:
    return runner._standard_binding(  # noqa: SLF001
        plan_builder.CPU_TERMINAL_REVIEW_BINDING
    )


def source_bindings() -> dict[str, dict[str, object]]:
    evidence = runner.predecessor_failure_bindings_rocm()
    bindings = {
        name: file_binding(path)
        for name, path in sorted(runner.SOURCE_PATHS.items())
    }
    if any(bindings.get(name) != value for name, value in evidence.items()):
        raise GenesisRocmBackendAuthorityError(
            "predecessor failure evidence is not exact"
        )
    return bindings


def _load_json(
    binding: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    try:
        value = json.loads(Path(str(binding["path"])).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GenesisRocmBackendAuthorityError(
            f"{label} is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise GenesisRocmBackendAuthorityError(
            f"{label} must be a JSON object"
        )
    return value


def _validate_review(
    review: Mapping[str, Any],
    *,
    preregistration_binding: Mapping[str, Any],
    scene_panel_binding: Mapping[str, Any],
    scientific_plan_binding: Mapping[str, Any],
    qualification_plan_binding: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> None:
    if (
        review.get("schema") != runner.SOURCE_REVIEW_SCHEMA
        or review.get("status") != runner.SOURCE_REVIEW_STATUS
        or review.get("protected_material_opened") is not False
        or review.get("findings") != []
        or review.get("preregistration_binding")
        != dict(preregistration_binding)
        or review.get("scene_panel_binding") != dict(scene_panel_binding)
        or review.get("plan_binding") != dict(scientific_plan_binding)
        or review.get("scientific_plan_binding")
        != dict(scientific_plan_binding)
        or review.get("qualification_plan_binding")
        != dict(qualification_plan_binding)
        or review.get("source_bindings") != dict(sources)
        or review.get("rocm_backend_source_audit")
        != ROCM_BACKEND_SOURCE_AUDIT
        or review.get("process_reset_equivalence_audit")
        != runner.collector.PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM
        or review.get("qualification_contract_audit")
        != qualifier.QUALIFICATION_CONTRACT
        or review.get("predecessor_cpu_terminal_review_binding")
        != _cpu_terminal_review_binding()
        or review.get("per_scene_process_evidence_audit")
        != predecessor_authority.REQUIRED_PROCESS_EVIDENCE_AUDIT
    ):
        raise GenesisRocmBackendAuthorityError(
            "independent ROCm source review changed"
        )


def build_qualification_authority(
    *,
    preregistration_binding: Mapping[str, Any],
    scene_panel_binding: Mapping[str, Any],
    scientific_plan: Mapping[str, Any],
    scientific_plan_binding: Mapping[str, Any],
    qualification_plan: Mapping[str, Any],
    qualification_plan_binding: Mapping[str, Any],
    source_review: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    prereg = _require_binding(
        preregistration_binding,
        path=runner.PREREGISTRATION,
        label="preregistration",
    )
    panel = _require_binding(
        scene_panel_binding,
        path=runner.SCENE_PANEL,
        label="scene panel",
    )
    science_binding = _require_binding(
        scientific_plan_binding,
        path=plan_builder.DEFAULT_PLAN_OUTPUT,
        label="scientific plan",
    )
    qualification_binding = _require_binding(
        qualification_plan_binding,
        path=plan_builder.QUALIFICATION_PLAN_OUTPUT,
        label="qualification plan",
    )
    review_binding = _require_binding(
        source_review_binding,
        path=runner.SOURCE_REVIEW,
        label="source review",
    )
    if _load_json(science_binding, label="scientific plan") != dict(
        scientific_plan
    ):
        raise GenesisRocmBackendAuthorityError(
            "scientific plan document changed"
        )
    if _load_json(
        qualification_binding, label="qualification plan"
    ) != dict(qualification_plan):
        raise GenesisRocmBackendAuthorityError(
            "qualification plan document changed"
        )
    plan_builder.validate_rocm_plan(
        scientific_plan,
        expected_attempt_id=plan_builder.DEFAULT_ATTEMPT_ID,
        expected_output_root=plan_builder.DEFAULT_OUTPUT_ROOT,
        plan_role="scientific",
    )
    plan_builder.validate_rocm_plan(
        qualification_plan,
        expected_attempt_id=plan_builder.QUALIFICATION_ATTEMPT_ID,
        expected_output_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
        plan_role="qualification",
    )
    sources = source_bindings()
    _validate_review(
        source_review,
        preregistration_binding=prereg,
        scene_panel_binding=panel,
        scientific_plan_binding=science_binding,
        qualification_plan_binding=qualification_binding,
        sources=sources,
    )
    if _load_json(review_binding, label="source review") != dict(
        source_review
    ):
        raise GenesisRocmBackendAuthorityError(
            "source review document changed"
        )
    if (
        plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()
        or plan_builder.QUALIFICATION_ATTEMPT_ROOT.is_symlink()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.is_symlink()
    ):
        raise GenesisRocmBackendAuthorityError(
            "ROCm qualification/scientific roots are not fresh"
        )
    authority = {
        "schema": qualifier.QUALIFICATION_AUTHORITY_SCHEMA,
        "status": qualifier.QUALIFICATION_AUTHORITY_STATUS,
        "attempt_id": plan_builder.QUALIFICATION_ATTEMPT_ID,
        "attempt_root": str(
            plan_builder.QUALIFICATION_ATTEMPT_ROOT.resolve()
        ),
        "collection_root": str(
            plan_builder.QUALIFICATION_OUTPUT_ROOT.resolve()
        ),
        "plan_binding": qualification_binding,
        "preregistration_binding": prereg,
        "source_review_binding": review_binding,
        "source_bindings": sources,
        "dino": predecessor_authority.dino_declaration_v2(),
        "config": runner.benchmark.config_v1(),
        "caps": copy.deepcopy(runner.collector.EXPECTED_CAPS),
        "permissions": copy.deepcopy(runner.collector.EXPECTED_PERMISSIONS),
        "qualification_contract": copy.deepcopy(
            qualifier.QUALIFICATION_CONTRACT
        ),
        "predecessor_cpu_terminal_review_binding": (
            _cpu_terminal_review_binding()
        ),
    }
    if set(authority) != qualifier.QUALIFICATION_AUTHORITY_FIELDS:
        raise GenesisRocmBackendAuthorityError(
            "qualification authority fields changed"
        )
    return authority


def _write_json_exclusive(
    path: Path, value: Mapping[str, Any]
) -> dict[str, object]:
    raw = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise GenesisRocmBackendAuthorityError(
            "qualification authority output is not fresh"
        ) from exc
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_binding(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AUTHORITY_OUTPUT)
    args = parser.parse_args(argv)
    prereg = file_binding(runner.PREREGISTRATION)
    panel = file_binding(runner.SCENE_PANEL)
    science_binding = file_binding(plan_builder.DEFAULT_PLAN_OUTPUT)
    qualification_binding = file_binding(plan_builder.QUALIFICATION_PLAN_OUTPUT)
    review_binding = file_binding(runner.SOURCE_REVIEW)
    authority = build_qualification_authority(
        preregistration_binding=prereg,
        scene_panel_binding=panel,
        scientific_plan=_load_json(
            science_binding, label="scientific plan"
        ),
        scientific_plan_binding=science_binding,
        qualification_plan=_load_json(
            qualification_binding, label="qualification plan"
        ),
        qualification_plan_binding=qualification_binding,
        source_review=_load_json(review_binding, label="source review"),
        source_review_binding=review_binding,
    )
    print(
        json.dumps(
            {"authority": _write_json_exclusive(args.output, authority)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_OUTPUT",
    "GenesisRocmBackendAuthorityError",
    "ROCM_BACKEND_SOURCE_AUDIT",
    "build_qualification_authority",
    "file_binding",
    "source_bindings",
]
