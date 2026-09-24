#!/usr/bin/env python3
"""Build, but never execute, the fresh V3 qualification authority."""
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

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_plan as plan_builder  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_authority as predecessor_authority  # noqa: E402
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as qualifier  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as runner  # noqa: E402


AUTHORITY_OUTPUT = qualifier.QUALIFICATION_AUTHORITY
ROCM_BACKEND_SOURCE_AUDIT = {
    "audit_passed": True,
    "science_identical_v3_integrity_replacement_not_retry": True,
    "complete_v3_source_closure_frozen_before_predecessor_overlay": True,
    "v2_preregistration_source_is_replacement_owned_literal": True,
    "v2_preregistration_source_sha256": (
        runner.V2_PREREGISTRATION_SOURCE_SHA256
    ),
    "qualifier_first_isolated_process_regression_passed": True,
    "poisoned_predecessor_cache_regression_passed": True,
    "scientific_content_data_model_evaluation_and_gates_unchanged": True,
    "required_host_home_literal": plan_builder.REQUIRED_HOST_HOME,
    "required_host_home_is_not_ambient_derived": True,
    "required_host_home_checked_before_reservation": True,
    "required_host_home_overwrites_child_ambient_value": True,
    "user_logname_and_lang_remain_absent": True,
    "exact_lexical_ld_lld_driver_in_plan_and_source": True,
    "driver_symlink_link_text_lld_required": True,
    "resolved_regular_lld_target_separately_bound": True,
    "direct_regular_lld_target_production_invocation_forbidden": True,
    "actual_plan_interpreter_venv_regression_passed": True,
    "actual_driver_and_generic_target_regression_passed": True,
    "genesis_version_exact_0_4_6": True,
    "genesis_backend_exact_amdgpu": True,
    "hip_device_exact_r9700_gfx1201": True,
    "host_rocm_ld_library_path_forbidden": True,
    "hsa_override_gfx_version_forbidden": True,
    "fresh_replacement_role_local_quadrants_cache_required": True,
    "qualification_is_separate_non_scientific_two_probe_attempt": True,
    "qualification_outputs_forbidden_from_scientific_reuse": True,
    "v2_terminal_review_document_exactly_bound": True,
    "v2_terminal_reservation_runtime_and_payload_reuse_forbidden": True,
    "v3_terminal_review_document_exactly_bound": True,
    "v3_authority_command_runtime_and_payload_reuse_forbidden": True,
    "replacement_v1_terminal_review_document_exactly_bound": True,
    "replacement_v1_authority_command_runtime_and_payload_reuse_forbidden": True,
    "closed_collector_qualifier_runner_interface_matrix_passed": True,
    "direct_successor_and_focused_runtime_dependency_files_bound": True,
}


class GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(RuntimeError):
    """Raised before an incomplete V3 authority can be emitted."""


GenesisRocmBackendAuthorityError = GenesisRocmBackendV3IntegrityReplacementV2AuthorityError


def file_binding(path: Path) -> dict[str, object]:
    try:
        return runner.file_binding_v1(path)
    except (OSError, RuntimeError) as exc:
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(str(exc)) from exc


def _require_binding(
    value: object, *, path: Path, label: str
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or value.get("path") != str(path.resolve())
        or file_binding(path) != dict(value)
    ):
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(
            f"{label} binding changed"
        )
    return dict(value)


def _cpu_terminal_review_binding() -> dict[str, Any]:
    return runner._standard_binding(  # noqa: SLF001
        plan_builder.CPU_TERMINAL_REVIEW_BINDING
    )


def _v1_terminal_review_binding() -> dict[str, Any]:
    return dict(runner.collector._EXACT_V1_REVIEW_BINDING)  # noqa: SLF001


def _v2_terminal_review_binding() -> dict[str, Any]:
    return dict(runner.collector._EXACT_V2_REVIEW_BINDING)  # noqa: SLF001


def _v3_terminal_review_binding() -> dict[str, Any]:
    return dict(runner.collector._EXACT_V3_REVIEW_BINDING)  # noqa: SLF001


def _replacement_v1_terminal_review_binding() -> dict[str, Any]:
    return dict(  # noqa: SLF001
        runner.collector._EXACT_REPLACEMENT_V1_REVIEW_BINDING
    )


def source_bindings() -> dict[str, dict[str, object]]:
    evidence = runner.predecessor_failure_bindings_rocm()
    bindings = {
        name: file_binding(path)
        for name, path in sorted(runner.SOURCE_PATHS.items())
    }
    if any(bindings.get(name) != value for name, value in evidence.items()):
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(
            "predecessor failure evidence is not exact"
        )
    return bindings


def _load_json(
    binding: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    try:
        value = json.loads(Path(str(binding["path"])).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(
            f"{label} is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(
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
        or review.get(
            "predecessor_v1_qualification_terminal_review_binding"
        )
        != _v1_terminal_review_binding()
        or review.get(
            "predecessor_v2_qualification_terminal_review_binding"
        )
        != _v2_terminal_review_binding()
        or review.get(
            "predecessor_v3_qualification_terminal_review_binding"
        )
        != _v3_terminal_review_binding()
        or review.get(
            "predecessor_replacement_v1_qualification_terminal_review_binding"
        )
        != _replacement_v1_terminal_review_binding()
        or review.get("per_scene_process_evidence_audit")
        != predecessor_authority.REQUIRED_PROCESS_EVIDENCE_AUDIT
        or review.get("v1_runtime_payload_opened") is not False
        or review.get("v1_terminal_payload_opened") is not False
        or review.get("v2_runtime_payload_opened") is not False
        or review.get("v2_terminal_payload_opened") is not False
        or review.get("v2_reservation_payload_opened") is not False
        or review.get("v3_runtime_payload_opened") is not False
        or review.get("v3_terminal_payload_opened") is not False
        or review.get("v3_reservation_payload_opened") is not False
        or review.get("v3_authority_or_command_payload_opened") is not False
        or review.get("replacement_v1_runtime_payload_opened") is not False
        or review.get("replacement_v1_terminal_payload_opened") is not False
        or review.get("replacement_v1_reservation_payload_opened") is not False
        or review.get("replacement_v1_authority_or_command_payload_opened")
        is not False
    ):
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(
            "independent V3 source review changed"
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
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(
            "scientific plan document changed"
        )
    if _load_json(
        qualification_binding, label="qualification plan"
    ) != dict(qualification_plan):
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(
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
    if _load_json(review_binding, label="source review") != dict(source_review):
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(
            "source review document changed"
        )
    if (
        plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()
        or plan_builder.QUALIFICATION_ATTEMPT_ROOT.is_symlink()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.is_symlink()
    ):
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(
            "fresh V3 qualification/scientific roots changed"
        )
    authority = {
        "schema": qualifier.QUALIFICATION_AUTHORITY_SCHEMA,
        "status": qualifier.QUALIFICATION_AUTHORITY_STATUS,
        "attempt_id": plan_builder.QUALIFICATION_ATTEMPT_ID,
        "attempt_root": str(plan_builder.QUALIFICATION_ATTEMPT_ROOT.resolve()),
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
        "predecessor_v1_qualification_terminal_review_binding": (
            _v1_terminal_review_binding()
        ),
        "predecessor_v2_qualification_terminal_review_binding": (
            _v2_terminal_review_binding()
        ),
        "predecessor_v3_qualification_terminal_review_binding": (
            _v3_terminal_review_binding()
        ),
        "predecessor_replacement_v1_qualification_terminal_review_binding": (
            _replacement_v1_terminal_review_binding()
        ),
    }
    if set(authority) != qualifier.QUALIFICATION_AUTHORITY_FIELDS:
        raise GenesisRocmBackendV3IntegrityReplacementV2AuthorityError(
            "V3 qualification authority fields changed"
        )
    return authority


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, object]:
    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        selected,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        payload = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_binding(selected)


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
        scientific_plan=_load_json(science_binding, label="scientific plan"),
        scientific_plan_binding=science_binding,
        qualification_plan=_load_json(
            qualification_binding, label="qualification plan"
        ),
        qualification_plan_binding=qualification_binding,
        source_review=_load_json(review_binding, label="source review"),
        source_review_binding=review_binding,
    )
    print(json.dumps({"authority": _write_json_exclusive(args.output, authority)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_OUTPUT",
    "GenesisRocmBackendAuthorityError",
    "GenesisRocmBackendV3IntegrityReplacementV2AuthorityError",
    "ROCM_BACKEND_SOURCE_AUDIT",
    "build_qualification_authority",
    "source_bindings",
]
