#!/usr/bin/env python3
"""Fresh-process cache-only replay for the physical-outcome screen V1."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_matched_branch_physical_outcome_screen_v1 as mechanism,
)
from scripts import (  # noqa: E402
    run_go2_matched_branch_physical_outcome_screen_v1 as runner,
)


class PhysicalOutcomeReplayError(RuntimeError):
    """Raised when fresh replay differs from the bound primary execution."""


def _canonical_equal(left: object, right: object) -> bool:
    return runner.canonical_bytes_v1(left) == runner.canonical_bytes_v1(right)


def _exact_tree_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and type(left) is type(right)
            and len(left) == len(right)
            and all(
                _exact_tree_equal(left_value, right_value)
                for left_value, right_value in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


def _read_checkpoint_v1(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = runner.file_binding_v1(path)
    if (
        binding["sha256"] != expected_sha256
        or binding["byte_count"] != expected_byte_count
    ):
        raise PhysicalOutcomeReplayError("checkpoint caller binding changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PhysicalOutcomeReplayError(
            "checkpoint is not a safe Torch payload"
        ) from error
    if not isinstance(payload, Mapping):
        raise PhysicalOutcomeReplayError("checkpoint payload is not a mapping")
    return dict(payload), binding


def _projection_v1(value: object, *, fragments: tuple[str, ...]) -> dict[str, object]:
    result: dict[str, object] = {}

    def visit(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                child = f"{path}.{key}" if path else str(key)
                if any(fragment in str(key).lower() for fragment in fragments):
                    result[child] = item[key]
                else:
                    visit(item[key], child)
        elif isinstance(item, (list, tuple)):
            for index, child_item in enumerate(item):
                visit(child_item, f"{path}[{index}]")

    visit(value, "")
    return result


def _nonempty_projection_equal_v1(
    left: object, right: object, *, fragments: tuple[str, ...]
) -> bool:
    left_projection = _projection_v1(left, fragments=fragments)
    right_projection = _projection_v1(right, fragments=fragments)
    return bool(left_projection) and _exact_tree_equal(left_projection, right_projection)


def _checkpoint_reproduction_v1(
    fresh: Mapping[str, Any], stored: Mapping[str, Any]
) -> dict[str, bool]:
    exact = _exact_tree_equal(fresh, stored)
    try:
        fresh_pca = fresh["pca"]["identity_sha256"]
        stored_pca = stored["pca"]["identity_sha256"]
        fresh_normalizers = {
            "outcome_stats": fresh["outcome_stats"]["identity_sha256"],
            **{
                arm: fresh["arms"][arm]["input_stats"]["identity_sha256"]
                for arm in mechanism.LEARNED_ARMS
            },
        }
        stored_normalizers = {
            "outcome_stats": stored["outcome_stats"]["identity_sha256"],
            **{
                arm: stored["arms"][arm]["input_stats"]["identity_sha256"]
                for arm in mechanism.LEARNED_ARMS
            },
        }
        fresh_states = {
            arm: [
                member["state_identity_sha256"]
                for member in fresh["arms"][arm]["members"]
            ]
            for arm in mechanism.LEARNED_ARMS
        }
        stored_states = {
            arm: [
                member["state_identity_sha256"]
                for member in stored["arms"][arm]["members"]
            ]
            for arm in mechanism.LEARNED_ARMS
        }
        fresh_steps = {
            arm: [
                member["training"]["optimizer_steps"]
                for member in fresh["arms"][arm]["members"]
            ]
            for arm in mechanism.LEARNED_ARMS
        }
        stored_steps = {
            arm: [
                member["training"]["optimizer_steps"]
                for member in stored["arms"][arm]["members"]
            ]
            for arm in mechanism.LEARNED_ARMS
        }
    except (KeyError, TypeError):
        return {
            "checkpoint_exact": exact,
            "pca_identity": False,
            "normalizer_identities": False,
            "state_dict_identities": False,
            "step_counts": False,
        }
    return {
        "checkpoint_exact": exact,
        "pca_identity": fresh_pca == stored_pca,
        "normalizer_identities": fresh_normalizers == stored_normalizers,
        "state_dict_identities": fresh_states == stored_states,
        "step_counts": fresh_steps == stored_steps,
    }


def _evaluation_reproduction_v1(
    fresh: Mapping[str, Any], stored: Mapping[str, Any]
) -> dict[str, bool]:
    exact = _canonical_equal(fresh, stored)
    try:
        prediction_equal = _canonical_equal(
            fresh["prediction_artifacts"], stored["prediction_artifacts"]
        )
        fresh_actions = {
            arm: [
                row.get("selected_action_id")
                for row in report["group_results"]
            ]
            for arm, report in fresh["arms"].items()
            if isinstance(report, Mapping) and "group_results" in report
        }
        stored_actions = {
            arm: [
                row.get("selected_action_id")
                for row in report["group_results"]
            ]
            for arm, report in stored["arms"].items()
            if isinstance(report, Mapping) and "group_results" in report
        }
        fresh_summaries = {
            arm: report["summary"] for arm, report in fresh["arms"].items()
        }
        stored_summaries = {
            arm: report["summary"] for arm, report in stored["arms"].items()
        }
        bootstrap_equal = _canonical_equal(
            fresh["paired_family_scene_cluster_comparisons"],
            stored["paired_family_scene_cluster_comparisons"],
        )
        gates_equal = _canonical_equal(fresh["gates"], stored["gates"])
    except (KeyError, TypeError, AttributeError):
        prediction_equal = False
        fresh_actions = None
        stored_actions = None
        fresh_summaries = None
        stored_summaries = None
        bootstrap_equal = False
        gates_equal = False
    return {
        "predictions": exact and prediction_equal,
        "selected_actions": exact and fresh_actions == stored_actions,
        "summaries": exact and _canonical_equal(fresh_summaries, stored_summaries),
        "bootstrap_intervals": exact and bootstrap_equal,
        "gates": exact and gates_equal,
        "evaluation_identity": (
            exact
            and mechanism.evaluation_identity_v1(fresh)
            == mechanism.evaluation_identity_v1(stored)
        ),
    }


def _independent_verdict_v1(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the preregistered branching decision without primary verdict code."""

    gates = evaluation.get("gates")
    if not isinstance(gates, Mapping):
        raise PhysicalOutcomeReplayError("evaluation gates are malformed")

    def passed(name: str) -> bool:
        value = gates.get(name)
        if not isinstance(value, Mapping) or type(value.get("passed")) is not bool:
            raise PhysicalOutcomeReplayError(f"evaluation gate {name} is malformed")
        return bool(value["passed"])

    names = set(gates)
    required_names = {
        "2_privileged_physical_oracle",
        "3_odometry_beats_task_action_only",
        "4_visual_beats_task_action_only",
        "5_visual_beats_odometry",
        "6a_odometry_beats_random_expected",
        "6b_visual_beats_random_expected",
    }
    if names != required_names:
        raise PhysicalOutcomeReplayError("evaluation gate inventory changed")
    visual_pass = all(
        passed(name)
        for name in (
            "2_privileged_physical_oracle",
            "4_visual_beats_task_action_only",
            "5_visual_beats_odometry",
            "6b_visual_beats_random_expected",
        )
    )
    odometry_pass = all(
        passed(name)
        for name in (
            "2_privileged_physical_oracle",
            "3_odometry_beats_task_action_only",
            "6a_odometry_beats_random_expected",
        )
    )
    status = (
        runner.PASS_VISUAL_STATUS
        if visual_pass
        else runner.PASS_ODOMETRY_STATUS
        if odometry_pass
        else runner.STOP_STATUS
    )
    combined = {
        "1_infrastructure_and_custody": {"passed": True},
        **dict(gates),
        "7_deterministic_replay": {"passed": True},
    }
    return {
        "gates": combined,
        "passed": status in {runner.PASS_VISUAL_STATUS, runner.PASS_ODOMETRY_STATUS},
        "terminal_status": status,
    }


def execute_replay_v1(
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    expected_checkpoint_byte_count: int,
    evaluation_path: Path,
    expected_evaluation_sha256: str,
    expected_evaluation_byte_count: int,
) -> dict[str, Any]:
    output_root = Path(str(authority["output_root"]))
    replay_path = output_root / "replay.json"
    if replay_path.exists() or replay_path.is_symlink():
        raise PhysicalOutcomeReplayError("replay output already exists")
    if checkpoint_path != output_root / "physical_outcome_checkpoint.pt":
        raise PhysicalOutcomeReplayError("checkpoint path changed")
    if evaluation_path != output_root / "evaluation.json":
        raise PhysicalOutcomeReplayError("evaluation path changed")

    # Fit from bound train inputs before opening either primary scientific output.
    runner._validate_upstream_route_v1(authority)  # noqa: SLF001
    train_receipts, eval_receipts, _receipt_audit = (
        runner._load_state_receipts_v1(authority)  # noqa: SLF001
    )
    train_cache, _train_receipt = runner._load_feature_cache_v1(  # noqa: SLF001
        authority, train_receipts, role="train"
    )
    train_dataset = mechanism.build_physical_dataset_v1(
        train_receipts=train_receipts,
        eval_receipts=None,
        train_cache=train_cache,
        eval_cache=None,
    )
    runner._configure_deterministic_cpu_training_v1()  # noqa: SLF001
    implementation_source_binding = authority["source_bindings"][
        "physical_outcome_evaluator"
    ]
    fresh_checkpoint = mechanism.fit_primary_checkpoint_v1(
        train_dataset,
        implementation_source_binding=implementation_source_binding,
    )
    mechanism.validate_checkpoint_v1(
        fresh_checkpoint,
        implementation_source_binding=implementation_source_binding,
    )
    stored_checkpoint, checkpoint_binding = _read_checkpoint_v1(
        checkpoint_path,
        expected_sha256=expected_checkpoint_sha256,
        expected_byte_count=expected_checkpoint_byte_count,
    )
    mechanism.validate_checkpoint_v1(
        stored_checkpoint,
        implementation_source_binding=implementation_source_binding,
    )
    checkpoint_reproduction = _checkpoint_reproduction_v1(
        fresh_checkpoint, stored_checkpoint
    )

    eval_cache, _eval_receipt = runner._load_feature_cache_v1(  # noqa: SLF001
        authority, eval_receipts, role="eval"
    )
    dataset = mechanism.build_physical_dataset_v1(
        train_receipts=train_receipts,
        eval_receipts=eval_receipts,
        train_cache=train_cache,
        eval_cache=eval_cache,
    )
    fresh_evaluation = mechanism.evaluate_primary_checkpoint_v1(
        fresh_checkpoint,
        dataset,
        implementation_source_binding=implementation_source_binding,
    )
    stored_evaluation, evaluation_binding = runner._read_bound_json(  # noqa: SLF001
        evaluation_path,
        expected_sha256=expected_evaluation_sha256,
        expected_byte_count=expected_evaluation_byte_count,
        label="primary evaluation",
    )
    evaluation_reproduction = _evaluation_reproduction_v1(
        fresh_evaluation, stored_evaluation
    )
    replay_verdict = _independent_verdict_v1(fresh_evaluation)
    verdict_reproduced = (
        replay_verdict.get("terminal_status") in mechanism.TERMINAL_STATUSES
        and replay_verdict.get("gates", {}).get("7_deterministic_replay")
        == {"passed": True}
    )
    reproduction = {
        **checkpoint_reproduction,
        **evaluation_reproduction,
        "verdict": verdict_reproduced,
    }
    reproduction["exactly_reproduced"] = all(reproduction.values())
    if set(reproduction) != runner.REPLAY_REPRODUCTION_FIELDS or not all(
        reproduction.values()
    ):
        raise PhysicalOutcomeReplayError(
            "fresh process did not exactly reproduce all registered outputs"
        )
    report = {
        "schema": runner.REPLAY_SCHEMA,
        "status": runner.REPLAY_STATUS,
        "citable_as_scientific_evidence": False,
        "authority_binding": dict(authority_binding),
        "checkpoint_binding": checkpoint_binding,
        "primary_evaluation_binding": evaluation_binding,
        "recomputed_evaluation": dict(fresh_evaluation),
        "recomputed_verdict": replay_verdict,
        "reproduction": reproduction,
        "protected_material_opened": False,
        "rgb_access": {"train": 0, "eval": 0},
        "encoder_execution_count": 0,
    }
    runner._execution_bindings_unchanged(  # noqa: SLF001
        authority, authority_binding=authority_binding
    )
    runner._write_json_exclusive(replay_path, report)  # noqa: SLF001
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-byte-count", type=int, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--expected-evaluation-sha256", required=True)
    parser.add_argument("--expected-evaluation-byte-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authority, authority_binding = runner._read_authority(  # noqa: SLF001
        args.authority,
        expected_sha256=args.expected_authority_sha256,
        expected_byte_count=args.expected_authority_byte_count,
    )
    report = execute_replay_v1(
        authority=authority,
        authority_binding=authority_binding,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_checkpoint_byte_count=args.expected_checkpoint_byte_count,
        evaluation_path=args.evaluation,
        expected_evaluation_sha256=args.expected_evaluation_sha256,
        expected_evaluation_byte_count=args.expected_evaluation_byte_count,
    )
    print(runner.canonical_bytes_v1({"status": report["status"]}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PhysicalOutcomeReplayError",
    "build_parser",
    "execute_replay_v1",
]
