#!/usr/bin/env python3
"""Run the exact one-shot 12-member bounded WM-A evaluation panel.

This supervisor has one fixed development output root.  It validates the
caller-bound pilot, independent pilot terminal review, progression result,
training-scene separation, and the complete fixed-terminal checkpoint panel
without opening an RGB leaf.  It then consumes the output root before the
first model evaluation.  Every member file is measurement-only; only the
complete 12-member aggregate may contain a global usefulness conclusion.

There is deliberately no retry, resume, overwrite, member-selection, or
partial-panel aggregation path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as consumer  # noqa: E402
from scripts import analyze_go2_world_model_progression_v1 as analyzer  # noqa: E402
from scripts import evaluate_go2_world_model_bounded_branch_experiment_v1 as evaluator  # noqa: E402
from scripts import dev_probe_counterfactual_action_fidelity as probe  # noqa: E402


PANEL_RESERVATION_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_evaluation_panel_reservation_v1"
)
PANEL_TERMINAL_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_evaluation_panel_terminal_v1"
)
DEV_ROOT = REPO_ROOT / ".generated/dev"
DEFAULT_OUTPUT_ROOT = (
    DEV_ROOT / "go2_world_model_bounded_branch_evaluation_panel_v1"
)
DEFAULT_PROGRESSION_ANALYSIS = (
    DEV_ROOT
    / "world_model_progression_v1"
    / "comparison_20260802_v1"
    / "analysis.json"
)
MEMBER_DIRECTORY = "members"
PANEL_RESULT_NAME = "panel_result.json"
TERMINAL_NAME = "terminal.json"
RESERVATION_NAME = "reservation.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPERVISOR_NONCE = re.compile(r"^[0-9a-f]{64}$")
_PANEL_RESERVATION_KEYS = {
    "schema",
    "status",
    "output_root",
    "supervisor_nonce",
    "supervisor_pid",
    "input_bindings",
    "pilot_terminal_gate",
    "training_scene_separation",
    "checkpoint_panel_bindings",
    "runtime_source_bindings",
    "evaluation_contract",
    "fixed_panel",
    "member_reports_are_measurement_only",
    "aggregate_requires_all_fixed_members",
    "retry_authorized",
    "resume_authorized",
    "overwrite_authorized",
    "member_selection_authorized",
    "citable_as_scientific_evidence",
}


class EvaluationPanelRunnerError(RuntimeError):
    """Raised when the consumed one-shot panel fails closed."""


def _reject_protected_path(path: Path, *, label: str) -> None:
    for part in Path(path).parts:
        lowered = part.lower()
        if (
            lowered == "sealed_test.json"
            or lowered == "sealed"
            or lowered.startswith("sealed_")
            or lowered in {"heldout", "held_out", "held-out"}
            or lowered.startswith("heldout_")
            or lowered.startswith("held_out_")
            or lowered.startswith("held-out-")
        ):
            raise EvaluationPanelRunnerError(f"{label} names protected material")


def _absolute_without_symlink(path: Path, *, label: str, must_exist: bool) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    _reject_protected_path(selected, label=label)
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise EvaluationPanelRunnerError(f"{label} traverses a symlink")
        if not cursor.exists():
            if must_exist:
                raise EvaluationPanelRunnerError(f"{label} is absent")
            break
    if must_exist and (not selected.exists() or selected.is_symlink()):
        raise EvaluationPanelRunnerError(f"{label} is absent or a symlink")
    return selected


def _require_sha256(value: str, *, label: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise EvaluationPanelRunnerError(f"{label} is not lowercase SHA-256")
    return value


def _pilot_binding(path: Path, *, label: str) -> dict[str, Any]:
    selected = _absolute_without_symlink(path, label=label, must_exist=True)
    return pilot.file_binding(selected)


def _probe_binding(path: Path, *, label: str) -> dict[str, Any]:
    selected = _absolute_without_symlink(path, label=label, must_exist=True)
    value = probe.file_binding(selected)
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "byte_count", "sha256"}
        or _SHA256.fullmatch(str(value.get("sha256"))) is None
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise EvaluationPanelRunnerError(f"{label} binding is malformed")
    return dict(value)


def _caller_binding(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    label: str,
) -> dict[str, Any]:
    _require_sha256(expected_sha256, label=f"expected {label} SHA-256")
    if type(expected_byte_count) is not int or expected_byte_count <= 0:
        raise EvaluationPanelRunnerError(f"expected {label} byte count is invalid")
    actual = _pilot_binding(path, label=label)
    if (
        actual["file_sha256"] != expected_sha256
        or actual["byte_count"] != expected_byte_count
    ):
        raise EvaluationPanelRunnerError(f"{label} caller binding changed")
    return actual


def _expected_member_keys() -> tuple[str, ...]:
    return tuple(
        f"{arm}/seed_{seed}"
        for arm in evaluator.MODEL_ARMS
        for seed in evaluator.TRAINING_SEEDS
    )


def _member_filename(arm: str, seed: int) -> str:
    return (
        f"{arm}_seed_{seed}_"
        f"update_{evaluator.EXPECTED_TERMINAL_UPDATE:06d}.json"
    )


def _validate_checkpoint_panel_bindings(
    value: object,
    *,
    training_result: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(_expected_member_keys()):
        raise EvaluationPanelRunnerError("training result did not bind the exact panel")
    output_root = training_result.resolve().parent
    result: dict[str, dict[str, Any]] = {}
    for arm in evaluator.MODEL_ARMS:
        for seed in evaluator.TRAINING_SEEDS:
            key = f"{arm}/seed_{seed}"
            binding = value[key]
            expected_path = (
                output_root
                / f"seed_{seed}"
                / (
                    f"{arm}_update_"
                    f"{evaluator.EXPECTED_TERMINAL_UPDATE:06d}.pt"
                )
            )
            if (
                not isinstance(binding, Mapping)
                or set(binding) != {"path", "byte_count", "sha256"}
                or Path(str(binding.get("path"))).resolve() != expected_path.resolve()
                or type(binding.get("byte_count")) is not int
                or int(binding["byte_count"]) <= 0
                or _SHA256.fullmatch(str(binding.get("sha256"))) is None
            ):
                raise EvaluationPanelRunnerError(
                    f"fixed-terminal checkpoint binding changed for {key}"
                )
            actual = _probe_binding(expected_path, label=f"checkpoint {key}")
            if actual != dict(binding):
                raise EvaluationPanelRunnerError(
                    f"fixed-terminal checkpoint bytes changed for {key}"
                )
            result[key] = actual
    return result


def _load_analyzer_checkpoint_panel_v1(
    analysis_path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Consume the immutable analyzer receipt; never request fresh training."""

    document, actual_binding = pilot.read_bound_json(
        analysis_path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="fixed progression analysis",
    )
    required = {
        "schema",
        "status",
        "development_only",
        "citable_as_world_model_usefulness_evidence",
        "input_result",
        "configuration",
        "decoder_anchor_by_seed",
        "contrasts",
        "proxy_routing",
        "terminal_snapshot_bindings",
        "uncertainty_limit",
    }
    routing = document.get("proxy_routing") if isinstance(document, Mapping) else None
    input_result = document.get("input_result") if isinstance(document, Mapping) else None
    if (
        not isinstance(document, Mapping)
        or set(document) != required
        or document.get("schema") != analyzer.SCHEMA
        or document.get("status") != "PASS_COMPLETE_FIXED_COMPARISON_ANALYSIS"
        or document.get("development_only") is not True
        or document.get("citable_as_world_model_usefulness_evidence") is not False
        or document.get("configuration") != analyzer.EXPECTED_CONFIGURATION
        or not isinstance(input_result, Mapping)
        or set(input_result) != {"path", "byte_count", "sha256"}
        or not isinstance(input_result.get("path"), str)
        or type(input_result.get("byte_count")) is not int
        or int(input_result["byte_count"]) <= 0
        or _SHA256.fullmatch(str(input_result.get("sha256"))) is None
        or not isinstance(routing, Mapping)
        or routing.get("causal_branch_evaluation_still_required") is not True
        or routing.get("bulk_training_scale_authorized") is not False
        or routing.get("world_model_usefulness_claim_authorized") is not False
    ):
        raise EvaluationPanelRunnerError("fixed progression analysis identity changed")
    result_path = _absolute_without_symlink(
        Path(str(input_result["path"])),
        label="analysis-bound training result",
        must_exist=True,
    )
    if result_path.name != "result.json" or result_path.parent != analysis_path.parent:
        raise EvaluationPanelRunnerError(
            "progression analysis does not bind its adjacent result.json"
        )
    actual_result = _probe_binding(
        result_path, label="analysis-bound training result"
    )
    if actual_result != dict(input_result):
        raise EvaluationPanelRunnerError(
            "progression analysis input result bytes changed"
        )
    nested = document.get("terminal_snapshot_bindings")
    if not isinstance(nested, Mapping) or set(nested) != {
        str(seed) for seed in evaluator.TRAINING_SEEDS
    }:
        raise EvaluationPanelRunnerError("analyzer terminal snapshot panel changed")
    flattened: dict[str, dict[str, Any]] = {}
    for seed in evaluator.TRAINING_SEEDS:
        seed_panel = nested[str(seed)]
        if not isinstance(seed_panel, Mapping) or set(seed_panel) != set(
            evaluator.MODEL_ARMS
        ):
            raise EvaluationPanelRunnerError(
                f"analyzer seed {seed} snapshot panel changed"
            )
        for arm in evaluator.MODEL_ARMS:
            flattened[f"{arm}/seed_{seed}"] = dict(seed_panel[arm])
    return {
        "analysis_binding": actual_binding,
        "analysis_document": dict(document),
        "training_result_binding": {
            "path": actual_result["path"],
            "byte_count": actual_result["byte_count"],
            "file_sha256": actual_result["sha256"],
        },
    }, flattened


def _preflight_v1(
    *,
    pilot_root: Path,
    manifest_byte_count: int,
    manifest_sha256: str,
    progression_analysis: Path,
    progression_analysis_sha256: str,
    progression_analysis_byte_count: int,
    pilot_terminal: Path,
    pilot_terminal_sha256: str,
    pilot_terminal_byte_count: int,
    pilot_terminal_review: Path,
    pilot_terminal_review_sha256: str,
    pilot_terminal_review_byte_count: int,
    evaluation_authority: Path,
    evaluation_authority_sha256: str,
    evaluation_authority_byte_count: int,
) -> dict[str, Any]:
    """Validate the complete experiment identity without opening an RGB leaf."""

    for path, label in (
        (pilot_root, "pilot root"),
        (progression_analysis, "progression analysis"),
        (pilot_terminal, "pilot terminal"),
        (pilot_terminal_review, "pilot terminal review"),
    ):
        _reject_protected_path(path, label=label)
    pilot_root = _absolute_without_symlink(
        pilot_root, label="pilot root", must_exist=True
    )
    progression_analysis = _absolute_without_symlink(
        progression_analysis, label="progression analysis", must_exist=True
    )
    if progression_analysis != Path(
        os.path.abspath(os.fspath(DEFAULT_PROGRESSION_ANALYSIS))
    ):
        raise EvaluationPanelRunnerError(
            "progression analysis is not comparison_20260802_v1/analysis.json"
        )
    pilot_terminal = _absolute_without_symlink(
        pilot_terminal, label="pilot terminal", must_exist=True
    )
    pilot_terminal_review = _absolute_without_symlink(
        pilot_terminal_review,
        label="pilot terminal review",
        must_exist=True,
    )
    selected_evaluation_authority = _absolute_without_symlink(
        evaluation_authority,
        label="posthoc evaluation authority",
        must_exist=True,
    )
    input_bindings = {
        "pilot_manifest": _caller_binding(
            pilot_root / "manifest.json",
            expected_sha256=manifest_sha256,
            expected_byte_count=manifest_byte_count,
            label="pilot manifest",
        ),
        "progression_analysis": _caller_binding(
            progression_analysis,
            expected_sha256=progression_analysis_sha256,
            expected_byte_count=progression_analysis_byte_count,
            label="progression analysis",
        ),
        "pilot_terminal": _caller_binding(
            pilot_terminal,
            expected_sha256=pilot_terminal_sha256,
            expected_byte_count=pilot_terminal_byte_count,
            label="pilot terminal",
        ),
        "pilot_terminal_review": _caller_binding(
            pilot_terminal_review,
            expected_sha256=pilot_terminal_review_sha256,
            expected_byte_count=pilot_terminal_review_byte_count,
            label="pilot terminal review",
        ),
    }
    input_bindings["evaluation_authority"] = _caller_binding(
        selected_evaluation_authority,
        expected_sha256=evaluation_authority_sha256,
        expected_byte_count=evaluation_authority_byte_count,
        label="posthoc evaluation authority",
    )
    analysis_receipt, analyzer_checkpoint_bindings = (
        _load_analyzer_checkpoint_panel_v1(
            progression_analysis,
            expected_sha256=progression_analysis_sha256,
            expected_byte_count=progression_analysis_byte_count,
        )
    )
    training_result_binding = analysis_receipt["training_result_binding"]
    input_bindings["training_result"] = dict(training_result_binding)
    training_result = Path(str(training_result_binding["path"]))
    bundle, terminal_gate = (
        evaluator.load_and_validate_posthoc_pilot_for_evaluation_v1(
            pilot_root=pilot_root,
            manifest_byte_count=manifest_byte_count,
            manifest_sha256=manifest_sha256,
            pilot_terminal=pilot_terminal,
            pilot_terminal_sha256=pilot_terminal_sha256,
            pilot_terminal_byte_count=pilot_terminal_byte_count,
            pilot_terminal_review=pilot_terminal_review,
            pilot_terminal_review_sha256=pilot_terminal_review_sha256,
            pilot_terminal_review_byte_count=pilot_terminal_review_byte_count,
            progression_analysis=progression_analysis,
            progression_analysis_sha256=progression_analysis_sha256,
            progression_analysis_byte_count=progression_analysis_byte_count,
            evaluation_authority=selected_evaluation_authority,
            evaluation_authority_sha256=evaluation_authority_sha256,
            evaluation_authority_byte_count=evaluation_authority_byte_count,
        )
    )
    if (
        len(bundle.groups_by_role.get("train", ())) != 128
        or len(bundle.groups_by_role.get("eval", ())) != 128
    ):
        raise EvaluationPanelRunnerError("pilot is not the exact 128/128 experiment")
    pilot_scene_ids = {
        group.scene_id
        for role in ("train", "eval")
        for group in bundle.groups_by_role[role]
    }
    first_arm = evaluator.MODEL_ARMS[0]
    first_seed = evaluator.TRAINING_SEEDS[0]
    first_checkpoint = Path(
        str(analyzer_checkpoint_bindings[f"{first_arm}/seed_{first_seed}"]["path"])
    )
    _analysis_document, training_separation = (
        evaluator.load_and_validate_progression_analysis_v1(
            progression_analysis,
            expected_sha256=progression_analysis_sha256,
            expected_byte_count=progression_analysis_byte_count,
            selected_checkpoint=first_checkpoint,
            expected_arm=first_arm,
            expected_seed=first_seed,
            pilot_scene_ids=pilot_scene_ids,
        )
    )
    if not isinstance(training_separation, Mapping):
        raise EvaluationPanelRunnerError("training separation receipt is malformed")
    if (
        training_separation.get("progression_analysis_binding")
        != input_bindings["progression_analysis"]
        or training_separation.get("training_result_binding")
        != input_bindings["training_result"]
    ):
        raise EvaluationPanelRunnerError(
            "evaluator progression-analysis ancestry disagrees with preflight"
        )
    checkpoint_bindings = _validate_checkpoint_panel_bindings(
        analyzer_checkpoint_bindings,
        training_result=training_result,
    )
    if training_separation.get("checkpoint_panel_bindings") != checkpoint_bindings:
        raise EvaluationPanelRunnerError(
            "analyzer snapshot bindings disagree with the validated training result"
        )
    evaluator._require_model_panel_lineage_match_v1(  # noqa: SLF001
        terminal_gate.get("model_panel_freeze"), training_separation
    )
    source_bindings = [
        _probe_binding(Path(path), label="panel runtime source")
        for path in (
            __file__,
            evaluator.__file__,
            analyzer.__file__,
            consumer.__file__,
            pilot.__file__,
            evaluator.posthoc_admission.__file__,
        )
    ]
    return {
        "pilot_root": pilot_root,
        "training_result": training_result,
        "progression_analysis": progression_analysis,
        "pilot_terminal": pilot_terminal,
        "pilot_terminal_review": pilot_terminal_review,
        "evaluation_authority": selected_evaluation_authority,
        "evaluation_authority_sha256": evaluation_authority_sha256,
        "evaluation_authority_byte_count": evaluation_authority_byte_count,
        "bundle_manifest_binding": dict(bundle.manifest_binding),
        "terminal_gate": dict(terminal_gate),
        "progression_analysis_receipt": analysis_receipt,
        "training_separation": dict(training_separation),
        "checkpoint_bindings": checkpoint_bindings,
        "input_bindings": input_bindings,
        "source_bindings": source_bindings,
    }


def _reserve_panel_v1(
    *,
    output_root: Path,
    preflight: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Consume the sole exact panel root before RGB/model evaluation."""

    dev_root = _absolute_without_symlink(
        DEV_ROOT, label="development output root", must_exist=True
    )
    selected = Path(os.path.abspath(os.fspath(output_root)))
    expected = Path(os.path.abspath(os.fspath(DEFAULT_OUTPUT_ROOT)))
    _reject_protected_path(selected, label="panel output root")
    if selected != expected or selected.parent != dev_root:
        raise EvaluationPanelRunnerError(
            f"panel output root must equal the exact fixed root {expected}"
        )
    if selected.exists() or selected.is_symlink():
        raise EvaluationPanelRunnerError(
            "fixed panel output root is already consumed; retry/resume is forbidden"
        )
    try:
        selected.mkdir(mode=0o755, parents=False, exist_ok=False)
        reservation_payload = {
            "schema": PANEL_RESERVATION_SCHEMA,
            "status": "RESERVED_PANEL_ATTEMPT_CONSUMED",
            "output_root": str(selected),
            "supervisor_nonce": secrets.token_hex(32),
            "supervisor_pid": os.getpid(),
            "input_bindings": preflight["input_bindings"],
            "pilot_terminal_gate": preflight["terminal_gate"],
            "training_scene_separation": preflight["training_separation"],
            "checkpoint_panel_bindings": preflight["checkpoint_bindings"],
            "runtime_source_bindings": preflight["source_bindings"],
            "evaluation_contract": evaluator.evaluation_contract_v1(),
            "fixed_panel": {
                "arms": list(evaluator.MODEL_ARMS),
                "training_seeds": list(evaluator.TRAINING_SEEDS),
                "terminal_update": evaluator.EXPECTED_TERMINAL_UPDATE,
                "member_count": len(_expected_member_keys()),
                "device": "cuda",
            },
            "member_reports_are_measurement_only": True,
            "aggregate_requires_all_fixed_members": True,
            "retry_authorized": False,
            "resume_authorized": False,
            "overwrite_authorized": False,
            "member_selection_authorized": False,
            "citable_as_scientific_evidence": False,
        }
        # Preserve the exact JSON-domain attempt document independently of the
        # stored receipt.  Terminal review checks both this in-memory identity
        # and the original byte binding; neither can substitute for semantic
        # validation of the reservation contract.
        expected_reservation = json.loads(
            json.dumps(reservation_payload, sort_keys=True, allow_nan=False)
        )
        reservation_binding = pilot.write_json_exclusive(
            selected / RESERVATION_NAME, expected_reservation
        )
    except Exception as exc:
        raise EvaluationPanelRunnerError(
            f"could not consume the fixed panel reservation: {exc}"
        ) from exc
    return selected, reservation_binding, expected_reservation


def _validate_terminal_reservation_v1(
    *,
    output_root: Path,
    reservation_binding: Mapping[str, Any],
    expected_reservation: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-open and semantically validate this supervisor's consumed attempt."""

    dev_root = _absolute_without_symlink(
        DEV_ROOT, label="development output root", must_exist=True
    )
    output = _absolute_without_symlink(
        output_root, label="panel output root", must_exist=True
    )
    expected_output = Path(os.path.abspath(os.fspath(DEFAULT_OUTPUT_ROOT)))
    if output != expected_output or output.parent != dev_root:
        raise EvaluationPanelRunnerError(
            "reservation output root is not the exact fixed panel attempt"
        )
    reservation_path = _absolute_without_symlink(
        output / RESERVATION_NAME,
        label="panel reservation",
        must_exist=True,
    )
    if reservation_path != output / RESERVATION_NAME:
        raise EvaluationPanelRunnerError(
            "reservation is not directly contained by the fixed panel attempt"
        )
    if (
        not isinstance(reservation_binding, Mapping)
        or set(reservation_binding) != {"path", "byte_count", "file_sha256"}
        or Path(str(reservation_binding.get("path"))) != reservation_path
        or type(reservation_binding.get("byte_count")) is not int
        or int(reservation_binding["byte_count"]) <= 0
        or _SHA256.fullmatch(str(reservation_binding.get("file_sha256"))) is None
    ):
        raise EvaluationPanelRunnerError("reservation binding is malformed")
    document, actual_binding = pilot.read_bound_json(
        reservation_path,
        expected_sha256=str(reservation_binding["file_sha256"]),
        expected_byte_count=int(reservation_binding["byte_count"]),
        label="panel reservation",
    )
    if actual_binding != dict(reservation_binding):
        raise EvaluationPanelRunnerError("reservation byte binding changed")
    if not isinstance(expected_reservation, Mapping):
        raise EvaluationPanelRunnerError("expected reservation identity is malformed")
    if document != dict(expected_reservation):
        raise EvaluationPanelRunnerError(
            "reservation differs from the supervisor-created attempt"
        )
    expected_fixed_panel = {
        "arms": list(evaluator.MODEL_ARMS),
        "training_seeds": list(evaluator.TRAINING_SEEDS),
        "terminal_update": evaluator.EXPECTED_TERMINAL_UPDATE,
        "member_count": len(_expected_member_keys()),
        "device": "cuda",
    }
    if (
        set(document) != _PANEL_RESERVATION_KEYS
        or document.get("schema") != PANEL_RESERVATION_SCHEMA
        or document.get("status") != "RESERVED_PANEL_ATTEMPT_CONSUMED"
        or document.get("output_root") != str(output)
        or not isinstance(document.get("supervisor_nonce"), str)
        or _SUPERVISOR_NONCE.fullmatch(document["supervisor_nonce"]) is None
        or type(document.get("supervisor_pid")) is not int
        or int(document["supervisor_pid"]) <= 0
        or document.get("supervisor_pid") != os.getpid()
        or document.get("input_bindings") != preflight["input_bindings"]
        or document.get("pilot_terminal_gate") != preflight["terminal_gate"]
        or document.get("training_scene_separation")
        != preflight["training_separation"]
        or document.get("checkpoint_panel_bindings")
        != preflight["checkpoint_bindings"]
        or document.get("runtime_source_bindings") != preflight["source_bindings"]
        or document.get("evaluation_contract") != evaluator.evaluation_contract_v1()
        or document.get("fixed_panel") != expected_fixed_panel
        or document.get("member_reports_are_measurement_only") is not True
        or document.get("aggregate_requires_all_fixed_members") is not True
        or document.get("retry_authorized") is not False
        or document.get("resume_authorized") is not False
        or document.get("overwrite_authorized") is not False
        or document.get("member_selection_authorized") is not False
        or document.get("citable_as_scientific_evidence") is not False
    ):
        raise EvaluationPanelRunnerError(
            "reservation does not describe the exact consumed no-retry attempt"
        )
    return {
        "binding": actual_binding,
        "schema": document["schema"],
        "status": document["status"],
        "output_root": document["output_root"],
        "supervisor_nonce": document["supervisor_nonce"],
        "supervisor_pid": document["supervisor_pid"],
    }


def _has_global_member_claim(value: object) -> bool:
    forbidden = {
        "global_verdict",
        "global_usefulness",
        "global_usefulness_verdict",
        "all_fixed_panel_members_reported",
    }
    if isinstance(value, Mapping):
        return any(
            str(key) in forbidden or _has_global_member_claim(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_global_member_claim(item) for item in value)
    return False


def _validate_member_measurement_v1(
    report: object,
    *,
    arm: str,
    seed: int,
    checkpoint_binding: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise EvaluationPanelRunnerError("member evaluator returned a non-object")
    identity = report.get("checkpoint_panel_identity")
    training = report.get("training_scene_separation")
    if (
        report.get("schema") != evaluator.REPORT_SCHEMA
        or report.get("status") != "COMPLETE_PENDING_INDEPENDENT_REVIEW"
        or report.get("citable_as_scientific_evidence") is not False
        or report.get("authorizes_retry_or_resume") is not False
        or report.get("scientific_verdict_emitted") is not False
        or report.get("pilot_manifest_binding")
        != preflight["bundle_manifest_binding"]
        or report.get("pilot_terminal_gate") != preflight["terminal_gate"]
        or report.get("checkpoint_binding") != dict(checkpoint_binding)
        or report.get("evaluation_contract") != evaluator.evaluation_contract_v1()
        or not isinstance(identity, Mapping)
        or identity.get("arm") != arm
        or identity.get("seed") != seed
        or identity.get("update") != evaluator.EXPECTED_TERMINAL_UPDATE
        or not isinstance(training, Mapping)
        or training.get("progression_analysis_binding")
        != preflight["training_separation"].get("progression_analysis_binding")
        or training.get("training_result_binding")
        != preflight["training_separation"].get("training_result_binding")
        or training.get("checkpoint_panel_bindings")
        != preflight["checkpoint_bindings"]
        or not isinstance(report.get("source_bindings"), list)
        or _has_global_member_claim(report)
    ):
        raise EvaluationPanelRunnerError(
            f"member {arm}/seed_{seed} is not a measurement-only fixed member"
        )
    return dict(report)


def _evaluate_member_v1(
    *,
    preflight: Mapping[str, Any],
    arm: str,
    seed: int,
    checkpoint_binding: Mapping[str, Any],
    manifest_byte_count: int,
    manifest_sha256: str,
    progression_analysis_sha256: str,
    progression_analysis_byte_count: int,
    pilot_terminal_sha256: str,
    pilot_terminal_byte_count: int,
    pilot_terminal_review_sha256: str,
    pilot_terminal_review_byte_count: int,
) -> dict[str, Any]:
    """Single adapter for the concurrently evolving member evaluator API."""

    return evaluator.evaluate_bound_model_v1(
        pilot_root=preflight["pilot_root"],
        manifest_byte_count=manifest_byte_count,
        manifest_sha256=manifest_sha256,
        checkpoint=Path(str(checkpoint_binding["path"])),
        checkpoint_sha256=str(checkpoint_binding["sha256"]),
        progression_analysis=preflight["progression_analysis"],
        progression_analysis_sha256=progression_analysis_sha256,
        progression_analysis_byte_count=progression_analysis_byte_count,
        pilot_terminal=preflight["pilot_terminal"],
        pilot_terminal_sha256=pilot_terminal_sha256,
        pilot_terminal_byte_count=pilot_terminal_byte_count,
        pilot_terminal_review=preflight["pilot_terminal_review"],
        pilot_terminal_review_sha256=pilot_terminal_review_sha256,
        pilot_terminal_review_byte_count=pilot_terminal_review_byte_count,
        expected_arm=arm,
        expected_training_seed=seed,
        device_name="cuda",
        evaluation_authority=preflight.get("evaluation_authority"),
        evaluation_authority_sha256=preflight.get(
            "evaluation_authority_sha256"
        ),
        evaluation_authority_byte_count=preflight.get(
            "evaluation_authority_byte_count"
        ),
    )


def _merge_member_sources_v1(
    expected: list[dict[str, Any]], report: Mapping[str, Any]
) -> None:
    by_path = {str(binding["path"]): dict(binding) for binding in expected}
    for value in report["source_bindings"]:
        if not isinstance(value, Mapping):
            raise EvaluationPanelRunnerError("member source binding is malformed")
        binding = dict(value)
        path = str(binding.get("path"))
        if path in by_path and by_path[path] != binding:
            raise EvaluationPanelRunnerError(
                "evaluator source changed between panel members"
            )
        if path not in by_path:
            if _probe_binding(Path(path), label="member evaluator source") != binding:
                raise EvaluationPanelRunnerError("member source binding changed")
            expected.append(binding)
            by_path[path] = binding


def _rehash_terminal_v1(
    *,
    output_root: Path,
    reservation_binding: Mapping[str, Any],
    expected_reservation: Mapping[str, Any],
    preflight: Mapping[str, Any],
    source_bindings: Sequence[Mapping[str, Any]],
    member_bindings: Mapping[str, Mapping[str, Any]],
    panel_result_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    failures: list[str] = []
    checked: dict[str, Any] = {
        "reservation": None,
        "inputs": {},
        "checkpoints": {},
        "runtime_sources": {},
        "member_reports": {},
        "panel_result": None,
    }
    try:
        checked["reservation"] = _validate_terminal_reservation_v1(
            output_root=output_root,
            reservation_binding=reservation_binding,
            expected_reservation=expected_reservation,
            preflight=preflight,
        )
    except Exception as exc:
        failures.append(f"reservation: {type(exc).__name__}: {exc}")
    for name, expected in preflight["input_bindings"].items():
        try:
            actual = _pilot_binding(Path(str(expected["path"])), label=f"input {name}")
            if actual != dict(expected):
                raise EvaluationPanelRunnerError("binding mismatch")
            checked["inputs"][name] = actual
        except Exception as exc:  # terminal audit must retain every finding
            failures.append(f"input {name}: {type(exc).__name__}: {exc}")
    for name, expected in preflight["checkpoint_bindings"].items():
        try:
            actual = _probe_binding(
                Path(str(expected["path"])), label=f"checkpoint {name}"
            )
            if actual != dict(expected):
                raise EvaluationPanelRunnerError("binding mismatch")
            checked["checkpoints"][name] = actual
        except Exception as exc:
            failures.append(f"checkpoint {name}: {type(exc).__name__}: {exc}")
    for expected in source_bindings:
        name = str(expected.get("path"))
        try:
            actual = _probe_binding(Path(name), label="runtime source")
            if actual != dict(expected):
                raise EvaluationPanelRunnerError("binding mismatch")
            checked["runtime_sources"][name] = actual
        except Exception as exc:
            failures.append(f"runtime source {name}: {type(exc).__name__}: {exc}")
    for name, expected in member_bindings.items():
        try:
            actual = _pilot_binding(
                Path(str(expected["path"])), label=f"member report {name}"
            )
            if actual != dict(expected):
                raise EvaluationPanelRunnerError("binding mismatch")
            checked["member_reports"][name] = actual
        except Exception as exc:
            failures.append(f"member report {name}: {type(exc).__name__}: {exc}")
    if panel_result_binding is not None:
        try:
            actual = _pilot_binding(
                Path(str(panel_result_binding["path"])), label="panel result"
            )
            if actual != dict(panel_result_binding):
                raise EvaluationPanelRunnerError("binding mismatch")
            checked["panel_result"] = actual
        except Exception as exc:
            failures.append(f"panel result: {type(exc).__name__}: {exc}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "checked_bindings": checked,
        "failures": failures,
    }


def _write_terminal_v1(
    output_root: Path,
    *,
    status: str,
    reservation_binding: Mapping[str, Any],
    member_bindings: Mapping[str, Mapping[str, Any]],
    panel_result_binding: Mapping[str, Any] | None,
    terminal_rehash: Mapping[str, Any],
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return pilot.write_json_exclusive(
        output_root / TERMINAL_NAME,
        {
            "schema": PANEL_TERMINAL_SCHEMA,
            "status": status,
            "citable_as_scientific_evidence": False,
            "scientific_verdict_emitted_by_terminal": False,
            "authorizes_retry_or_resume": False,
            "authorizes_overwrite": False,
            "all_fixed_panel_members_reported": (
                len(member_bindings) == len(_expected_member_keys())
            ),
            "reservation_binding": dict(reservation_binding),
            "member_report_bindings": dict(member_bindings),
            "panel_result_binding": (
                None if panel_result_binding is None else dict(panel_result_binding)
            ),
            "terminal_rehash": dict(terminal_rehash),
            "failure": None if failure is None else dict(failure),
            "independent_review_required": True,
        },
    )


def run_panel_v1(
    *,
    pilot_root: Path,
    manifest_byte_count: int,
    manifest_sha256: str,
    progression_analysis: Path,
    progression_analysis_sha256: str,
    progression_analysis_byte_count: int,
    pilot_terminal: Path,
    pilot_terminal_sha256: str,
    pilot_terminal_byte_count: int,
    pilot_terminal_review: Path,
    pilot_terminal_review_sha256: str,
    pilot_terminal_review_byte_count: int,
    evaluation_authority: Path,
    evaluation_authority_sha256: str,
    evaluation_authority_byte_count: int,
    output_root: Path | None = None,
) -> dict[str, Any]:
    selected_output = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    expected_output = Path(os.path.abspath(os.fspath(DEFAULT_OUTPUT_ROOT)))
    if Path(os.path.abspath(os.fspath(selected_output))) != expected_output:
        raise EvaluationPanelRunnerError(
            f"panel output root must equal the exact fixed root {expected_output}"
        )
    preflight = _preflight_v1(
        pilot_root=pilot_root,
        manifest_byte_count=manifest_byte_count,
        manifest_sha256=manifest_sha256,
        progression_analysis=progression_analysis,
        progression_analysis_sha256=progression_analysis_sha256,
        progression_analysis_byte_count=progression_analysis_byte_count,
        pilot_terminal=pilot_terminal,
        pilot_terminal_sha256=pilot_terminal_sha256,
        pilot_terminal_byte_count=pilot_terminal_byte_count,
        pilot_terminal_review=pilot_terminal_review,
        pilot_terminal_review_sha256=pilot_terminal_review_sha256,
        pilot_terminal_review_byte_count=pilot_terminal_review_byte_count,
        evaluation_authority=evaluation_authority,
        evaluation_authority_sha256=evaluation_authority_sha256,
        evaluation_authority_byte_count=evaluation_authority_byte_count,
    )
    output, reservation_binding, expected_reservation = _reserve_panel_v1(
        output_root=selected_output, preflight=preflight
    )
    member_bindings: dict[str, dict[str, Any]] = {}
    member_reports: list[dict[str, Any]] = []
    panel_result_binding: dict[str, Any] | None = None
    source_bindings = [dict(value) for value in preflight["source_bindings"]]
    try:
        for arm in evaluator.MODEL_ARMS:
            for seed in evaluator.TRAINING_SEEDS:
                key = f"{arm}/seed_{seed}"
                checkpoint_binding = preflight["checkpoint_bindings"][key]
                report = _evaluate_member_v1(
                    preflight=preflight,
                    arm=arm,
                    seed=seed,
                    checkpoint_binding=checkpoint_binding,
                    manifest_byte_count=manifest_byte_count,
                    manifest_sha256=manifest_sha256,
                    progression_analysis_sha256=progression_analysis_sha256,
                    progression_analysis_byte_count=progression_analysis_byte_count,
                    pilot_terminal_sha256=pilot_terminal_sha256,
                    pilot_terminal_byte_count=pilot_terminal_byte_count,
                    pilot_terminal_review_sha256=pilot_terminal_review_sha256,
                    pilot_terminal_review_byte_count=pilot_terminal_review_byte_count,
                )
                validated = _validate_member_measurement_v1(
                    report,
                    arm=arm,
                    seed=seed,
                    checkpoint_binding=checkpoint_binding,
                    preflight=preflight,
                )
                _merge_member_sources_v1(source_bindings, validated)
                report_path = output / MEMBER_DIRECTORY / _member_filename(arm, seed)
                report_binding = pilot.write_json_exclusive(report_path, validated)
                member_bindings[key] = report_binding
                stored, actual_binding = pilot.read_bound_json(
                    report_path,
                    expected_sha256=str(report_binding["file_sha256"]),
                    expected_byte_count=int(report_binding["byte_count"]),
                    label=f"stored model-panel member {key}",
                )
                if actual_binding != report_binding:
                    raise EvaluationPanelRunnerError(
                        f"stored member binding changed for {key}"
                    )
                member_reports.append(stored)
        if set(member_bindings) != set(_expected_member_keys()):
            raise EvaluationPanelRunnerError("fixed panel is incomplete")
        panel_result = evaluator.aggregate_model_panel_v1(member_reports)
        if (
            not isinstance(panel_result, Mapping)
            or panel_result.get("schema") != evaluator.PANEL_REPORT_SCHEMA
            or panel_result.get("status") != "COMPLETE_PENDING_INDEPENDENT_REVIEW"
            or panel_result.get("citable_as_scientific_evidence") is not False
            or panel_result.get("authorizes_retry_or_resume") is not False
            or panel_result.get("all_fixed_panel_members_reported") is not True
            or "global_verdict" not in panel_result
        ):
            raise EvaluationPanelRunnerError("complete-panel aggregate is malformed")
        panel_result_binding = pilot.write_json_exclusive(
            output / PANEL_RESULT_NAME, dict(panel_result)
        )
        terminal_rehash = _rehash_terminal_v1(
            output_root=output,
            reservation_binding=reservation_binding,
            expected_reservation=expected_reservation,
            preflight=preflight,
            source_bindings=source_bindings,
            member_bindings=member_bindings,
            panel_result_binding=panel_result_binding,
        )
        if terminal_rehash["status"] != "PASS":
            raise EvaluationPanelRunnerError("terminal input/source rehash failed")
        terminal_binding = _write_terminal_v1(
            output,
            status="COMPLETE_PENDING_INDEPENDENT_REVIEW",
            reservation_binding=reservation_binding,
            member_bindings=member_bindings,
            panel_result_binding=panel_result_binding,
            terminal_rehash=terminal_rehash,
            failure=None,
        )
        return {
            "output_root": str(output),
            "reservation_binding": reservation_binding,
            "member_report_bindings": member_bindings,
            "panel_result_binding": panel_result_binding,
            "terminal_binding": terminal_binding,
        }
    except BaseException as exc:
        terminal_rehash = _rehash_terminal_v1(
            output_root=output,
            reservation_binding=reservation_binding,
            expected_reservation=expected_reservation,
            preflight=preflight,
            source_bindings=source_bindings,
            member_bindings=member_bindings,
            panel_result_binding=panel_result_binding,
        )
        failure = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "completed_members": len(member_bindings),
            "aggregate_written": panel_result_binding is not None,
        }
        try:
            terminal_binding = _write_terminal_v1(
                output,
                status="FAILED_TERMINAL_NO_RETRY",
                reservation_binding=reservation_binding,
                member_bindings=member_bindings,
                panel_result_binding=panel_result_binding,
                terminal_rehash=terminal_rehash,
                failure=failure,
            )
        except Exception as terminal_exc:
            raise EvaluationPanelRunnerError(
                "one-shot panel failed and its terminal receipt could not be written: "
                f"{type(terminal_exc).__name__}: {terminal_exc}"
            ) from exc
        raise EvaluationPanelRunnerError(
            "one-shot panel failed; retry/resume is forbidden; "
            f"terminal={terminal_binding['path']}"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument(
        "--expected-pilot-manifest-byte-count", required=True, type=int
    )
    parser.add_argument("--expected-pilot-manifest-sha256", required=True)
    parser.add_argument("--progression-analysis", required=True, type=Path)
    parser.add_argument("--expected-progression-analysis-sha256", required=True)
    parser.add_argument(
        "--expected-progression-analysis-byte-count", required=True, type=int
    )
    parser.add_argument("--pilot-terminal", required=True, type=Path)
    parser.add_argument("--expected-pilot-terminal-sha256", required=True)
    parser.add_argument(
        "--expected-pilot-terminal-byte-count", required=True, type=int
    )
    parser.add_argument("--pilot-terminal-review", required=True, type=Path)
    parser.add_argument(
        "--expected-pilot-terminal-review-sha256", required=True
    )
    parser.add_argument(
        "--expected-pilot-terminal-review-byte-count", required=True, type=int
    )
    parser.add_argument("--evaluation-authority", required=True, type=Path)
    parser.add_argument(
        "--expected-evaluation-authority-sha256", required=True
    )
    parser.add_argument(
        "--expected-evaluation-authority-byte-count", required=True, type=int
    )
    args = parser.parse_args(argv)
    result = run_panel_v1(
        pilot_root=args.pilot_root,
        manifest_byte_count=args.expected_pilot_manifest_byte_count,
        manifest_sha256=args.expected_pilot_manifest_sha256,
        progression_analysis=args.progression_analysis,
        progression_analysis_sha256=(
            args.expected_progression_analysis_sha256
        ),
        progression_analysis_byte_count=(
            args.expected_progression_analysis_byte_count
        ),
        pilot_terminal=args.pilot_terminal,
        pilot_terminal_sha256=args.expected_pilot_terminal_sha256,
        pilot_terminal_byte_count=args.expected_pilot_terminal_byte_count,
        pilot_terminal_review=args.pilot_terminal_review,
        pilot_terminal_review_sha256=(
            args.expected_pilot_terminal_review_sha256
        ),
        pilot_terminal_review_byte_count=(
            args.expected_pilot_terminal_review_byte_count
        ),
        evaluation_authority=args.evaluation_authority,
        evaluation_authority_sha256=(
            args.expected_evaluation_authority_sha256
        ),
        evaluation_authority_byte_count=(
            args.expected_evaluation_authority_byte_count
        ),
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
                "output_root": result["output_root"],
                "terminal_binding": result["terminal_binding"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PROGRESSION_ANALYSIS",
    "EvaluationPanelRunnerError",
    "PANEL_RESERVATION_SCHEMA",
    "PANEL_TERMINAL_SCHEMA",
    "run_panel_v1",
]
