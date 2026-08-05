#!/usr/bin/env python3
"""Build V3 scientific authority only after exact V3 qualification PASS."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_plan as plan_builder  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_qualification_authority as qualification_authority  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_authority as predecessor_authority  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3 as runner  # noqa: E402


AUTHORITY_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_execution_authority_2026-08-04.json"
)
GenesisRocmBackendV3AuthorityError = (
    qualification_authority.GenesisRocmBackendV3AuthorityError
)
GenesisRocmBackendAuthorityError = GenesisRocmBackendV3AuthorityError
file_binding = qualification_authority.file_binding


def build_scientific_authority(
    *,
    preregistration_binding: Mapping[str, Any],
    scene_panel_binding: Mapping[str, Any],
    scientific_plan: Mapping[str, Any],
    scientific_plan_binding: Mapping[str, Any],
    qualification_plan_binding: Mapping[str, Any],
    source_review: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
    qualification_result_binding: Mapping[str, Any],
) -> dict[str, Any]:
    prereg = qualification_authority._require_binding(  # noqa: SLF001
        preregistration_binding,
        path=runner.PREREGISTRATION,
        label="preregistration",
    )
    panel = qualification_authority._require_binding(  # noqa: SLF001
        scene_panel_binding,
        path=runner.SCENE_PANEL,
        label="scene panel",
    )
    science_binding = qualification_authority._require_binding(  # noqa: SLF001
        scientific_plan_binding,
        path=plan_builder.DEFAULT_PLAN_OUTPUT,
        label="scientific plan",
    )
    qualification_binding = qualification_authority._require_binding(  # noqa: SLF001
        qualification_plan_binding,
        path=plan_builder.QUALIFICATION_PLAN_OUTPUT,
        label="qualification plan",
    )
    review_binding = qualification_authority._require_binding(  # noqa: SLF001
        source_review_binding,
        path=runner.SOURCE_REVIEW,
        label="source review",
    )
    if qualification_authority._load_json(  # noqa: SLF001
        science_binding, label="scientific plan"
    ) != dict(scientific_plan):
        raise GenesisRocmBackendV3AuthorityError(
            "scientific plan document changed"
        )
    plan_builder.validate_rocm_plan(
        scientific_plan,
        expected_attempt_id=plan_builder.DEFAULT_ATTEMPT_ID,
        expected_output_root=plan_builder.DEFAULT_OUTPUT_ROOT,
        plan_role="scientific",
    )
    sources = qualification_authority.source_bindings()
    qualification_authority._validate_review(  # noqa: SLF001
        source_review,
        preregistration_binding=prereg,
        scene_panel_binding=panel,
        scientific_plan_binding=science_binding,
        qualification_plan_binding=qualification_binding,
        sources=sources,
    )
    if qualification_authority._load_json(  # noqa: SLF001
        review_binding, label="source review"
    ) != dict(source_review):
        raise GenesisRocmBackendV3AuthorityError(
            "source review document changed"
        )
    _result, exact_result_binding = (
        runner.validate_qualification_result_binding(
            qualification_result_binding
        )
    )
    if (
        plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.is_symlink()
    ):
        raise GenesisRocmBackendV3AuthorityError(
            "fresh V3 scientific attempt root changed"
        )
    authority = {
        "schema": runner.collector.AUTHORITY_SCHEMA,
        "status": runner.collector.AUTHORITY_STATUS,
        "attempt_id": plan_builder.DEFAULT_ATTEMPT_ID,
        "attempt_root": str(plan_builder.DEFAULT_ATTEMPT_ROOT.resolve()),
        "collection_root": str(plan_builder.DEFAULT_OUTPUT_ROOT.resolve()),
        "plan_binding": science_binding,
        "preregistration_binding": prereg,
        "source_review_binding": review_binding,
        "source_bindings": sources,
        "dino": predecessor_authority.dino_declaration_v2(),
        "config": runner.benchmark.config_v1(),
        "caps": copy.deepcopy(runner.collector.EXPECTED_CAPS),
        "permissions": copy.deepcopy(runner.collector.EXPECTED_PERMISSIONS),
        "qualification_result_binding": exact_result_binding,
        "predecessor_cpu_terminal_review_binding": (
            qualification_authority._cpu_terminal_review_binding()  # noqa: SLF001
        ),
        "predecessor_v1_qualification_terminal_review_binding": (
            qualification_authority._v1_terminal_review_binding()  # noqa: SLF001
        ),
        "predecessor_v2_qualification_terminal_review_binding": (
            qualification_authority._v2_terminal_review_binding()  # noqa: SLF001
        ),
    }
    if set(authority) != runner.collector.AUTHORITY_FIELDS:
        raise GenesisRocmBackendV3AuthorityError(
            "V3 scientific authority fields changed"
        )
    runner.collector._require_v2_review_binding(authority)  # noqa: SLF001
    return authority


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AUTHORITY_OUTPUT)
    parser.add_argument(
        "--qualification-result",
        type=Path,
        default=runner.qualifier.QUALIFICATION_RESULT_PATH,
    )
    args = parser.parse_args(argv)
    prereg = file_binding(runner.PREREGISTRATION)
    panel = file_binding(runner.SCENE_PANEL)
    science_binding = file_binding(plan_builder.DEFAULT_PLAN_OUTPUT)
    qualification_binding = file_binding(plan_builder.QUALIFICATION_PLAN_OUTPUT)
    review_binding = file_binding(runner.SOURCE_REVIEW)
    result_binding = file_binding(args.qualification_result)
    authority = build_scientific_authority(
        preregistration_binding=prereg,
        scene_panel_binding=panel,
        scientific_plan=qualification_authority._load_json(  # noqa: SLF001
            science_binding, label="scientific plan"
        ),
        scientific_plan_binding=science_binding,
        qualification_plan_binding=qualification_binding,
        source_review=qualification_authority._load_json(  # noqa: SLF001
            review_binding, label="source review"
        ),
        source_review_binding=review_binding,
        qualification_result_binding=result_binding,
    )
    print(
        json.dumps(
            {
                "authority": qualification_authority._write_json_exclusive(  # noqa: SLF001
                    args.output, authority
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_OUTPUT",
    "GenesisRocmBackendAuthorityError",
    "GenesisRocmBackendV3AuthorityError",
    "build_scientific_authority",
]
