#!/usr/bin/env python3
"""Run the science-identical DINO calibration integrity replacement V1."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import __version__ as PILLOW_VERSION
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_dinov2_physical_readout_calibration_integrity_replacement_v1
    as compatibility,
)
from scripts import (  # noqa: E402
    evaluate_go2_world_model_visual_domain_parity_task_relevance_v1
    as task_relevance,
)
from scripts import (  # noqa: E402
    run_go2_dinov2_physical_readout_calibration_v1 as original,
)


AUTHORITY_SCHEMA = (
    "lewm_go2_dinov2_physical_readout_calibration_"
    "integrity_replacement_v1_execution_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_SCIENCE_IDENTICAL_INTEGRITY_REPLACEMENT"
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_dinov2_physical_readout_calibration_"
    "integrity_replacement_v1_source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_INTEGRITY_REPLACEMENT_SOURCE_REVIEW"
COMPATIBILITY_RECEIPT_SCHEMA = (
    "lewm_go2_dinov2_physical_readout_calibration_"
    "integrity_replacement_v1_compatibility_receipt_v1"
)
COMPATIBILITY_RECEIPT_STATUS = "PASS_PUBLISHED_BEFORE_CALIBRATION_EVAL_ACCESS"

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_dinov2_physical_readout_calibration_"
    "integrity_replacement_v1_preregistration_2026-08-03.md"
)
PREREGISTRATION_SHA256 = (
    "034e7b29803394d7da229273945deafa771a8104568e6b9a40ce492458f00c52"
)
PREREGISTRATION_BYTE_COUNT = 8_511
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_dinov2_physical_readout_calibration_"
    "integrity_replacement_v1_source_review_2026-08-03.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_dinov2_physical_readout_calibration_v1/"
    "attempt_v2_integrity_replacement_v1"
)
ORIGINAL_OUTPUT_ROOT = original.DEFAULT_OUTPUT_ROOT

ORIGINAL_PREREGISTRATION = original.PREREGISTRATION
ORIGINAL_SOURCE_REVIEW = original.SOURCE_REVIEW
ORIGINAL_AUTHORITY = REPO_ROOT / (
    "docs/lewm_go2_dinov2_physical_readout_calibration_v1_"
    "execution_authority_2026-08-03.json"
)
ORIGINAL_RESERVATION = ORIGINAL_OUTPUT_ROOT / "reservation.json"
ORIGINAL_TERMINAL = ORIGINAL_OUTPUT_ROOT / "terminal.json"
ORIGINAL_TERMINAL_FAILURE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_dinov2_physical_readout_calibration_v1_"
    "terminal_failure_review_2026-08-03.json"
)
STORED_TASK_RELEVANCE_RESULT = REPO_ROOT / (
    "docs/lewm_go2_world_model_visual_domain_parity_"
    "task_relevant_input_adequacy_result_v1_2026-08-02.json"
)
STORED_TASK_RELEVANCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_world_model_visual_domain_parity_"
    "task_relevant_input_adequacy_independent_review_v1_2026-08-02.json"
)

SOURCE_REVIEW_FIELDS = frozenset(
    {
        "schema",
        "status",
        "review_date",
        "reviewer",
        "protected_material_opened",
        "preregistration_binding",
        "original_failure_review_binding",
        "source_bindings",
        "checks",
        "audit_history",
        "findings",
    }
)
SOURCE_REVIEW_CHECKS = frozenset(
    {
        "replacement_preregistration_exact",
        "original_science_contract_unchanged",
        "original_attempt_failure_and_consumption_exact",
        "source_and_input_closure_complete_and_exact",
        "singleton_ssim_admission_matches_preregistration",
        "stored_and_recomputed_results_independently_pass",
        "all_non_ssim_fields_remain_canonical_exact",
        "compatibility_scope_restores_shared_runtime_functions",
        "compatibility_receipt_precedes_eval_rgb_and_dino_work",
        "frozen_posthoc_loader_remains_in_force",
        "original_executor_and_scientific_verdict_remain_in_force",
        "replacement_authority_and_fresh_root_fail_closed",
        "no_training_collection_or_protected_access_authorized",
        "focused_tests_passed",
        "compile_and_whitespace_checks_passed",
    }
)

SOURCE_PATHS = {
    **{
        f"original_{label}": path
        for label, path in original.SOURCE_PATHS.items()
    },
    "compatibility_module": Path(compatibility.__file__).resolve(),
    "compatibility_test": REPO_ROOT
    / "lewm/tests/test_go2_dinov2_physical_readout_calibration_integrity_replacement_v1.py",
    "replacement_runner": Path(__file__).resolve(),
    "replacement_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_dinov2_physical_readout_calibration_integrity_replacement_v1.py",
    "task_relevance_evaluator": Path(task_relevance.__file__).resolve(),
    "task_relevance_collector": REPO_ROOT
    / "scripts/collect_go2_world_model_counterfactual_pilot_v1.py",
    "task_relevance_h6_dataset": REPO_ROOT
    / "lewm/datasets/go2_explicit_plan_discounted_successor_state_v27.py",
    "task_relevance_mask_benchmark": REPO_ROOT
    / "lewm/benchmarks/go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py",
    "task_relevance_parity_authority_builder": REPO_ROOT
    / "scripts/build_go2_world_model_visual_domain_parity_authority_v1.py",
    "task_relevance_parity_evaluator": REPO_ROOT
    / "scripts/evaluate_go2_world_model_visual_domain_parity_v1.py",
    "task_relevance_parity_plan_builder": REPO_ROOT
    / "scripts/build_go2_world_model_visual_domain_parity_plan_v1.py",
    "task_relevance_parity_supervisor": REPO_ROOT
    / "scripts/run_go2_world_model_visual_domain_parity_authorized_v1.py",
    "task_relevance_probe": REPO_ROOT
    / "scripts/dev_probe_counterfactual_action_fidelity.py",
    "task_relevance_probe_evaluator": REPO_ROOT
    / "scripts/evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py",
    "task_relevance_probe_model": REPO_ROOT
    / "lewm/models/rgb_recurrent_patch_memory_temporal_jepa_v1.py",
    "task_relevance_probe_trainer": REPO_ROOT
    / "scripts/dev_train_temporal_jepa_scaled.py",
    "task_relevance_reference_renderer": REPO_ROOT / "scripts/render_replay_v03.py",
    "task_relevance_graphics_supervisor": REPO_ROOT
    / "scripts/run_go2_world_model_counterfactual_calibration_authorized_v1.py",
}


class IntegrityReplacementRunnerError(RuntimeError):
    """Raised when replacement authority or compatibility scope changes."""


def _binding(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256,
        "byte_count": byte_count,
    }


def replacement_config_v1() -> dict[str, Any]:
    return {
        **original.calibration_config_v1(),
        "compatibility_absolute_tolerance": (
            compatibility.SSIM_ABSOLUTE_TOLERANCE
        ),
        "compatibility_relative_tolerance": (
            compatibility.SSIM_RELATIVE_TOLERANCE
        ),
        "compatibility_allowed_differing_paths": [
            compatibility.SSIM_DOTTED_PATH
        ],
        "integrity_replacement": "v1",
    }


def _fixed_input_bindings_v1() -> dict[str, dict[str, Any]]:
    return {
        **original._fixed_input_bindings_v1(),  # noqa: SLF001
        "original_preregistration": _binding(
            ORIGINAL_PREREGISTRATION,
            original.PREREGISTRATION_SHA256,
            original.PREREGISTRATION_BYTE_COUNT,
        ),
        "original_source_review": _binding(
            ORIGINAL_SOURCE_REVIEW,
            "2e0305154674da1f39a621d4ac90e58652721403dde7e0cb1d782b8f79944174",
            6_927,
        ),
        "original_authority": _binding(
            ORIGINAL_AUTHORITY,
            "3a403377b071a9f916882cc5315b6b5be4d097a6c215562a565245962e6e2cc2",
            6_407,
        ),
        "original_reservation": _binding(
            ORIGINAL_RESERVATION,
            "963b915597c69e55e459db0018e9e9acb773a75f6d1da3ee3b3fd706a641f6fd",
            501,
        ),
        "original_terminal": _binding(
            ORIGINAL_TERMINAL,
            "066ad891811f7f5a7d7969b8d584ae12ca896a51b7702716cfc50ce096addb48",
            403,
        ),
        "original_terminal_failure_review": _binding(
            ORIGINAL_TERMINAL_FAILURE_REVIEW,
            "7f99e3136857a5149acfa74daed5f2ba54be5110942544c2df5aa230e0dd7ea9",
            6_201,
        ),
        "stored_task_relevance_result": _binding(
            STORED_TASK_RELEVANCE_RESULT,
            "5094104ac29b4652cd577015c5fbf23b42f0768c78a205cbf07a77d992339ca7",
            94_165,
        ),
        "stored_task_relevance_review": _binding(
            STORED_TASK_RELEVANCE_REVIEW,
            "29eb00a486604824effb56502194855553f87c81a9691d4075a5810273c92ca9",
            2_080,
        ),
    }


def _validate_source_review_v1(
    review: Mapping[str, Any],
    *,
    preregistration_binding: Mapping[str, Any],
    original_failure_review_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
) -> None:
    reviewer = review.get("reviewer")
    history = review.get("audit_history")
    if (
        set(review) != SOURCE_REVIEW_FIELDS
        or review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != SOURCE_REVIEW_STATUS
        or review.get("review_date") != "2026-08-03"
        or review.get("preregistration_binding") != preregistration_binding
        or review.get("original_failure_review_binding")
        != original_failure_review_binding
        or review.get("source_bindings") != source_bindings
        or review.get("protected_material_opened") is not False
        or review.get("findings") != []
        or not isinstance(reviewer, Mapping)
        or set(reviewer) != {"identity", "independence_basis"}
        or any(
            not isinstance(value, str) or not value.strip()
            for value in reviewer.values()
        )
        or not isinstance(history, list)
        or not history
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"finding", "resolution", "evidence"}
            or any(
                not isinstance(value, str) or not value.strip()
                for value in item.values()
            )
            for item in history
        )
        or not isinstance(review.get("checks"), Mapping)
        or set(review["checks"]) != SOURCE_REVIEW_CHECKS
        or any(value is not True for value in review["checks"].values())
    ):
        raise IntegrityReplacementRunnerError(
            "independent replacement source review did not pass exactly"
        )


def _validate_output_root_v1(path: object) -> Path:
    if not isinstance(path, str) or path != str(DEFAULT_OUTPUT_ROOT.resolve()):
        raise IntegrityReplacementRunnerError("replacement output root changed")
    selected = Path(path)
    if selected == ORIGINAL_OUTPUT_ROOT.resolve():
        raise IntegrityReplacementRunnerError("original attempt root is immutable")
    return selected


def _check_dino_repository_v1(encoder: Mapping[str, Any]) -> None:
    repo = original._safe_path(  # noqa: SLF001
        Path(str(encoder["repo_path"])), label="DINO repository"
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if commit != original.DINO_REPOSITORY_COMMIT or status:
        raise IntegrityReplacementRunnerError(
            "DINO repository is not clean at the frozen commit"
        )


def _read_authority(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority, authority_binding = original._read_bound_json(  # noqa: SLF001
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="replacement execution authority",
    )
    required = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "science_identical_to_original",
        "original_attempt_remains_consumed",
        "authorizes_compatibility_admission",
        "authorizes_collection",
        "authorizes_eval_rgb_access",
        "authorizes_model_training",
        "authorizes_retry_or_resume",
        "authorizes_train_rgb_access",
        "preregistration_binding",
        "source_review_binding",
        "source_bindings",
        "input_bindings",
        "encoder_source",
        "output_root",
        "environment",
        "config",
        "git_commit",
    }
    if (
        set(authority) != required
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("citable_as_scientific_evidence") is not False
        or authority.get("science_identical_to_original") is not True
        or authority.get("original_attempt_remains_consumed") is not True
        or authority.get("authorizes_compatibility_admission") is not True
        or authority.get("authorizes_collection") is not False
        or authority.get("authorizes_eval_rgb_access") is not True
        or authority.get("authorizes_model_training") is not False
        or authority.get("authorizes_retry_or_resume") is not False
        or authority.get("authorizes_train_rgb_access") is not False
        or authority.get("config") != replacement_config_v1()
    ):
        raise IntegrityReplacementRunnerError(
            "replacement execution authority contract changed"
        )
    _validate_output_root_v1(authority.get("output_root"))
    preregistration = original._require_binding(  # noqa: SLF001
        authority.get("preregistration_binding"), label="replacement preregistration"
    )
    if preregistration != _binding(
        PREREGISTRATION, PREREGISTRATION_SHA256, PREREGISTRATION_BYTE_COUNT
    ):
        raise IntegrityReplacementRunnerError(
            "replacement preregistration binding changed"
        )
    inputs = authority.get("input_bindings")
    expected_inputs = _fixed_input_bindings_v1()
    if not isinstance(inputs, Mapping) or dict(inputs) != expected_inputs:
        raise IntegrityReplacementRunnerError("replacement input closure changed")
    for label, item in inputs.items():
        original._require_binding(item, label=f"replacement input {label}")  # noqa: SLF001
    sources = authority.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_PATHS):
        raise IntegrityReplacementRunnerError("replacement source closure changed")
    for label, expected_path in SOURCE_PATHS.items():
        item = original._require_binding(  # noqa: SLF001
            sources[label], label=f"replacement source {label}"
        )
        if item["path"] != str(expected_path.resolve()):
            raise IntegrityReplacementRunnerError(
                f"replacement source {label} path changed"
            )
    review_binding = original._require_binding(  # noqa: SLF001
        authority.get("source_review_binding"), label="replacement source review"
    )
    if review_binding["path"] != str(SOURCE_REVIEW.resolve()):
        raise IntegrityReplacementRunnerError("replacement source review path changed")
    review, _ = original._read_bound_json(  # noqa: SLF001
        Path(review_binding["path"]),
        expected_sha256=review_binding["sha256"],
        expected_byte_count=review_binding["byte_count"],
        label="replacement source review",
    )
    _validate_source_review_v1(
        review,
        preregistration_binding=preregistration,
        original_failure_review_binding=inputs[
            "original_terminal_failure_review"
        ],
        source_bindings=sources,
    )
    encoder = authority.get("encoder_source")
    if (
        not isinstance(encoder, Mapping)
        or set(encoder) != {"repo_path", "repo_commit", "checkpoint_binding"}
        or encoder.get("repo_commit") != original.DINO_REPOSITORY_COMMIT
    ):
        raise IntegrityReplacementRunnerError("DINO encoder source changed")
    checkpoint = original._require_binding(  # noqa: SLF001
        encoder.get("checkpoint_binding"), label="DINO checkpoint"
    )
    if (
        checkpoint["sha256"] != original.DINO_CHECKPOINT_SHA256
        or checkpoint["byte_count"] != original.DINO_CHECKPOINT_BYTE_COUNT
    ):
        raise IntegrityReplacementRunnerError("DINO checkpoint changed")
    _check_dino_repository_v1(encoder)
    environment = authority.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != {"python", "torch", "hip", "numpy", "pillow"}
        or environment.get("python") != str(Path(sys.executable).resolve())
        or environment.get("torch") != torch.__version__
        or environment.get("hip") != torch.version.hip
        or environment.get("numpy") != np.__version__
        or environment.get("pillow") != PILLOW_VERSION
    ):
        raise IntegrityReplacementRunnerError("replacement environment changed")
    commit = authority.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=False,
        ).returncode
        != 0
    ):
        raise IntegrityReplacementRunnerError(
            "reviewed replacement source commit is not an execution ancestor"
        )
    return authority, authority_binding


def _load_stored_task_relevance_v1(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = authority["input_bindings"]
    result_binding = inputs["stored_task_relevance_result"]
    result, _ = original._read_bound_json(  # noqa: SLF001
        Path(result_binding["path"]),
        expected_sha256=result_binding["sha256"],
        expected_byte_count=result_binding["byte_count"],
        label="stored task-relevance result",
    )
    review_binding = inputs["stored_task_relevance_review"]
    review, _ = original._read_bound_json(  # noqa: SLF001
        Path(review_binding["path"]),
        expected_sha256=review_binding["sha256"],
        expected_byte_count=review_binding["byte_count"],
        label="stored task-relevance review",
    )
    adequacy = review.get("adequacy_result_binding")
    expected_adequacy = {
        "path": result_binding["path"],
        "file_sha256": result_binding["sha256"],
        "byte_count": result_binding["byte_count"],
    }
    if (
        review.get("schema")
        != "lewm_go2_world_model_visual_domain_parity_task_relevant_input_adequacy_independent_review_v1"
        or review.get("status")
        != "PASS_INDEPENDENTLY_REVIEWED_TASK_RELEVANT_INPUT_ADEQUACY_DEVELOPMENT_ONLY"
        or review.get("authority_granted_by_this_document") is not False
        or review.get("scientific_claim_granted_by_this_document") is not False
        or review.get("remaining_findings") != []
        or adequacy != expected_adequacy
    ):
        raise IntegrityReplacementRunnerError(
            "stored task-relevance review changed"
        )
    return result


@contextmanager
def scoped_compatibility_admission_v1(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Scope the singleton admission to one frozen posthoc-loader call."""

    output_root = Path(str(authority["output_root"]))
    receipt_path = output_root / "compatibility_receipt.json"
    stored = _load_stored_task_relevance_v1(authority)
    original_evaluator = task_relevance.evaluate_task_relevance_v1
    original_loader = original.screen_data.load_bound_posthoc_bundle_v1
    state: dict[str, Any] = {
        "evaluator_calls": 0,
        "loader_calls": 0,
        "receipt_binding": None,
    }

    def admitted_evaluator(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        state["evaluator_calls"] += 1
        if state["evaluator_calls"] != 1:
            raise IntegrityReplacementRunnerError(
                "task-relevance evaluator call count changed"
            )
        recomputed = original_evaluator(*args, **kwargs)
        admitted, evidence = compatibility.admit_task_relevance_result_v1(
            stored=stored,
            recomputed=recomputed,
        )
        receipt = {
            "schema": COMPATIBILITY_RECEIPT_SCHEMA,
            "status": COMPATIBILITY_RECEIPT_STATUS,
            "citable_as_scientific_evidence": False,
            "publication_stage": (
                "inside_task_relevance_evaluator_before_outer_loader_acceptance"
            ),
            "authority_binding": dict(authority_binding),
            "preregistration_binding": dict(
                authority["preregistration_binding"]
            ),
            "original_failure_review_binding": dict(
                authority["input_bindings"][
                    "original_terminal_failure_review"
                ]
            ),
            "stored_task_relevance_result_binding": dict(
                authority["input_bindings"]["stored_task_relevance_result"]
            ),
            "stored_task_relevance_review_binding": dict(
                authority["input_bindings"]["stored_task_relevance_review"]
            ),
            "task_relevance_evaluator_source_binding": dict(
                authority["source_bindings"]["task_relevance_evaluator"]
            ),
            "environment": dict(authority["environment"]),
            "admission": evidence,
        }
        original._write_json_exclusive(receipt_path, receipt)  # noqa: SLF001
        state["receipt_binding"] = original.file_binding_v1(receipt_path)
        return admitted

    def admitted_loader() -> object:
        state["loader_calls"] += 1
        if state["loader_calls"] != 1:
            raise IntegrityReplacementRunnerError(
                "strict posthoc loader call count changed"
            )
        bundle = original_loader()
        if state["evaluator_calls"] != 1 or state["receipt_binding"] is None:
            raise IntegrityReplacementRunnerError(
                "compatibility receipt was not published before loader return"
            )
        return bundle

    task_relevance.evaluate_task_relevance_v1 = admitted_evaluator
    original.screen_data.load_bound_posthoc_bundle_v1 = admitted_loader
    try:
        yield state
    finally:
        task_relevance.evaluate_task_relevance_v1 = original_evaluator
        original.screen_data.load_bound_posthoc_bundle_v1 = original_loader


def execute_replacement_v1(
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_output_root_v1(authority.get("output_root"))
    with scoped_compatibility_admission_v1(
        authority, authority_binding=authority_binding
    ):
        return original.execute_v1(
            authority,
            authority_binding=authority_binding,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    args = parser.parse_args(argv)
    authority, authority_binding = _read_authority(
        args.authority,
        expected_sha256=args.expected_authority_sha256,
        expected_byte_count=args.expected_authority_byte_count,
    )
    output_root = Path(str(authority["output_root"]))
    existed = output_root.exists()
    try:
        report = execute_replacement_v1(
            authority, authority_binding=authority_binding
        )
    except Exception as error:
        if (
            not existed
            and output_root.is_dir()
            and not (output_root / "terminal.json").exists()
        ):
            original._write_json_exclusive(  # noqa: SLF001
                output_root / "terminal.json",
                {
                    "schema": original.TERMINAL_SCHEMA,
                    "status": original.FAIL_STATUS,
                    "citable_as_scientific_evidence": False,
                    "authorizes_retry_or_resume": False,
                    "authorizes_model_training": False,
                    "result_binding": None,
                    "deterministic_replay_passed": False,
                    "failure": {
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                },
            )
        raise
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "COMPATIBILITY_RECEIPT_SCHEMA",
    "COMPATIBILITY_RECEIPT_STATUS",
    "DEFAULT_OUTPUT_ROOT",
    "IntegrityReplacementRunnerError",
    "SOURCE_PATHS",
    "SOURCE_REVIEW_CHECKS",
    "SOURCE_REVIEW_FIELDS",
    "SOURCE_REVIEW_SCHEMA",
    "SOURCE_REVIEW_STATUS",
    "execute_replacement_v1",
    "replacement_config_v1",
    "scoped_compatibility_admission_v1",
]
