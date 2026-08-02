#!/usr/bin/env python3
"""Validate and summarize the fixed Go2 world-model 2x2 progression screen.

This is an offline, payload-free analyzer.  It reads only the runner's JSON
receipt and hashes the four fixed-terminal snapshot files for each registered
seed.  It never opens a snapshot tensor payload and it does not select a
checkpoint from validation performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


SCHEMA = "go2_world_model_progression_v1_analysis_v1"
RUNNER_SCHEMA = "dev_go2_world_model_progression_v1"
RUNNER_STATUS = "COMPLETE_DEVELOPMENT_COMPARISON"
ARMS = ("masked_plain", "masked_delta", "full_plain", "full_delta")
SEEDS = (2026080201, 2026080202, 2026080203)
CHANCE = 1.0 / 9.0
STUDENT_T_975_DF2 = 4.302652729911275
GAP_CLOSURE_FRACTION = 0.25

EXPECTED_CONFIGURATION = {
    "arms": list(ARMS),
    "seeds": list(SEEDS),
    "updates": 700,
    "batch_size": 256,
    "microbatch_size": 16,
    "action_auxiliary_weight": 0.1,
    "decoder_pretrain_updates": 300,
    "minimum_decoder_anchor_lower_bound": CHANCE,
    "decoder_anchor_bootstrap_resamples": 2_000,
    "decoder_frozen_during_predictor_training": True,
    "strict_deterministic_algorithms": True,
    "checkpoint_selection": "fixed_terminal_update_only",
    "evaluation_rows": 2_048,
    "snapshots_written": True,
}

EXPECTED_SOURCE_BINDINGS = (
    {
        "path": "scripts/dev_train_go2_world_model_progression_v1.py",
        "byte_count": 44_968,
        "sha256": "0cb15c6414d7deeda6c206981457c72a45558905ea695cdae924a844702d49e0",
    },
    {
        "path": "lewm/models/go2_world_model_progression_v1.py",
        "byte_count": 12_224,
        "sha256": "b7582059034a1af475595b33f1a369a61aafa933564ec9e8d25317022680c0fb",
    },
    {
        "path": "scripts/execute_go2_world_model_existing_pool_three_arm_v1.py",
        "byte_count": 103_001,
        "sha256": "b0ca02d706b0108885e51b32353b8af6e440259a0108fa934ad8bd9e70366d7d",
    },
    {
        "path": "scripts/dev_train_temporal_jepa_scaled.py",
        "byte_count": 35_478,
        "sha256": "97154b693d3ca2b96e7e0d88c378c07b349c1d77e0db05548e48823834a35037",
    },
    {
        "path": "lewm/models/rgb_recurrent_patch_memory_temporal_jepa_v1.py",
        "byte_count": 17_480,
        "sha256": "324bd76eb0f8285dac01ccba0741e38546fe93506f5afd51facfaa71826346c3",
    },
    {
        "path": "lewm/benchmarks/go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py",
        "byte_count": 44_605,
        "sha256": "cec018dade02e4c8217d74792f8fdc6afba84f414d094f4879e353d78fee4f84",
    },
)

EXPECTED_PREDECESSOR = {
    "byte_count": 52_282_877,
    "sha256": "f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873",
}
EXPECTED_PACK = {
    "manifest_sha256": "22364f911ab5d3e2956ea9a3fc2d92e2869830cd858ef2d2269379dfc6041bae",
    "train": {
        "row_identity_sha256": "9bd2b1bb89d7290b4dcae8490e3188f14d0072b73e1ce0e67de503fe976b6809",
        "frames_sha256": "df9a5982370f4ba7c5d1c492f080d44f9900d889877ddb73f08454ba151a5a74",
        "actions_sha256": "11bfcd0724397be8fc84969a32c01b71d41fdedb34c75bbc7a9e4d481a934a78",
        "metadata_sha256": "2f265eaa57979f2e9c49956ab7bf83df29bcbc75d6b2f274f4d9b7b5d9635265",
    },
    "val": {
        "row_identity_sha256": "2d1859118824a99b52027d97ef2a406f3571cdf349325ca3b6b7f646f7554963",
        "frames_sha256": "e457d244c07516947ffb8005e2477d9a7f48c5e6a03b8701cf994debb06f6d66",
        "actions_sha256": "ad1b33d6ff4839736e27d37114bb1c01ca1cae693b5317c055dc9e776a8be6a1",
        "metadata_sha256": "6ef0d194c45a60d9cc28806dd8158360ae4ea6da55caf8685bdcdda9cfeff2a4",
    },
}

# Every contrast is oriented so that a positive value favors the named factor.
METRICS = {
    "hardest_wrong_action_margin_mean": 1.0,
    "hardest_wrong_action_margin_q05": 1.0,
    "persistence_advantage_mean": 1.0,
    "nine_way_action_balanced_accuracy": 1.0,
    "factual_energy_mean": -1.0,
}


class AnalysisError(ValueError):
    """Raised when the fixed comparison receipt is not admissible."""


def _reject_protected_path(path: Path) -> None:
    selected = path.resolve(strict=False)
    if selected.name == "sealed_test.json" or any(
        part == "sealed" or part.startswith("sealed_") for part in selected.parts
    ):
        raise AnalysisError("protected path rejected before access")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{name} is not numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise AnalysisError(f"{name} is nonfinite")
    return selected


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS) or any(not math.isfinite(value) for value in values):
        raise AnalysisError("factorial contrast must contain three finite seeds")
    center = mean(values)
    spread = stdev(values)
    half_width = STUDENT_T_975_DF2 * spread / math.sqrt(len(values))
    return {
        "values_by_seed": {str(seed): value for seed, value in zip(SEEDS, values, strict=True)},
        "mean": center,
        "sample_standard_deviation": spread,
        "minimum": min(values),
        "maximum": max(values),
        "positive_seed_count": sum(value > 0.0 for value in values),
        "two_sided_student_t_95": {
            "degrees_of_freedom": 2,
            "critical_value": STUDENT_T_975_DF2,
            "lower": center - half_width,
            "upper": center + half_width,
        },
    }


def _validate_pack(inputs: Mapping[str, Any]) -> None:
    predecessor = inputs.get("predecessor")
    if not isinstance(predecessor, Mapping):
        raise AnalysisError("predecessor binding is absent")
    for key, expected in EXPECTED_PREDECESSOR.items():
        if predecessor.get(key) != expected:
            raise AnalysisError(f"predecessor {key} binding changed")
    for role in ("train", "val"):
        binding = inputs.get(role)
        if not isinstance(binding, Mapping):
            raise AnalysisError(f"{role} binding is absent")
        expected = EXPECTED_PACK[role]
        if binding.get("manifest_sha256") != EXPECTED_PACK["manifest_sha256"]:
            raise AnalysisError(f"{role} manifest binding changed")
        if binding.get("row_identity_sha256") != expected["row_identity_sha256"]:
            raise AnalysisError(f"{role} row identity changed")
        for field in ("frames", "actions", "metadata"):
            nested = binding.get(field)
            if not isinstance(nested, Mapping) or nested.get("sha256") != expected[f"{field}_sha256"]:
                raise AnalysisError(f"{role} {field} binding changed")


def _validate_anchor(anchor: Mapping[str, Any], *, seed: int) -> None:
    if set(anchor) != {"masked", "full"}:
        raise AnalysisError(f"seed {seed} decoder anchor panels changed")
    for panel in ("masked", "full"):
        interval = anchor[panel]
        if not isinstance(interval, Mapping):
            raise AnalysisError(f"seed {seed} {panel} decoder anchor is absent")
        for key in ("point", "lower_95", "upper_95"):
            _finite_number(interval.get(key), name=f"seed {seed} {panel} anchor {key}")
        if int(interval.get("requested_resamples", -1)) != 2_000:
            raise AnalysisError(f"seed {seed} {panel} anchor resamples changed")
        if int(interval.get("seed", -1)) != 20260802:
            raise AnalysisError(f"seed {seed} {panel} anchor bootstrap seed changed")
        if int(interval.get("scene_clusters", -1)) != 150:
            raise AnalysisError(f"seed {seed} {panel} anchor scene count changed")
        if _finite_number(interval["lower_95"], name="anchor lower") <= CHANCE:
            raise AnalysisError(f"seed {seed} {panel} decoder anchor did not clear chance")


def _validate_terminal_metrics(metrics: Mapping[str, Any], *, seed: int, arm: str) -> None:
    if int(metrics.get("row_count", -1)) != 2_048:
        raise AnalysisError(f"seed {seed} {arm} terminal row count changed")
    for metric in METRICS:
        _finite_number(metrics.get(metric), name=f"seed {seed} {arm} {metric}")
    per_action = metrics.get("per_action")
    if not isinstance(per_action, Mapping) or set(per_action) != {str(value) for value in range(9)}:
        raise AnalysisError(f"seed {seed} {arm} per-action panel changed")
    rows = []
    for action in range(9):
        action_metrics = per_action[str(action)]
        if not isinstance(action_metrics, Mapping):
            raise AnalysisError(f"seed {seed} {arm} action {action} panel changed")
        rows.append(int(action_metrics.get("rows", -1)))
    if min(rows) < 1 or sum(rows) != 2_048:
        raise AnalysisError(f"seed {seed} {arm} per-action rows changed")


def _snapshot_bindings(result_path: Path) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        seed_bindings: dict[str, Any] = {}
        for arm in ARMS:
            path = result_path.parent / f"seed_{seed}" / f"{arm}_update_000700.pt"
            if path.is_symlink() or not path.is_file():
                raise AnalysisError(f"terminal snapshot is absent or a symlink: {path}")
            resolved = path.resolve(strict=True)
            protected_parts = {
                part for part in resolved.parts if part == "sealed" or part.startswith("sealed_")
            }
            if protected_parts:
                raise AnalysisError("protected snapshot path rejected")
            seed_bindings[arm] = {
                "path": str(resolved),
                "byte_count": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
        bindings[str(seed)] = seed_bindings
    return bindings


def analyze(payload: Mapping[str, Any], *, result_path: Path) -> dict[str, Any]:
    _reject_protected_path(result_path)
    if payload.get("schema") != RUNNER_SCHEMA or payload.get("status") != RUNNER_STATUS:
        raise AnalysisError("runner schema or terminal status changed")
    if payload.get("citable_as_scientific_evidence") is not False:
        raise AnalysisError("runner result crossed its development-only boundary")
    if payload.get("protected_material_opened") is not False:
        raise AnalysisError("runner reported protected-material access")
    if payload.get("configuration") != EXPECTED_CONFIGURATION:
        raise AnalysisError("fixed comparison configuration changed")
    if payload.get("source_bindings") != list(EXPECTED_SOURCE_BINDINGS):
        raise AnalysisError("fixed runner source closure changed")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise AnalysisError("runner input bindings are absent")
    _validate_pack(inputs)

    seed_results = payload.get("seed_results")
    if not isinstance(seed_results, Mapping) or set(seed_results) != {str(seed) for seed in SEEDS}:
        raise AnalysisError("fixed seed panel changed")

    terminal_by_seed: dict[int, Mapping[str, Any]] = {}
    anchor_by_seed: dict[str, Any] = {}
    for seed in SEEDS:
        seed_result = seed_results[str(seed)]
        if not isinstance(seed_result, Mapping):
            raise AnalysisError(f"seed {seed} result is absent")
        anchor = seed_result.get("decoder_anchor_balanced_accuracy")
        if not isinstance(anchor, Mapping):
            raise AnalysisError(f"seed {seed} anchor is absent")
        _validate_anchor(anchor, seed=seed)
        anchor_by_seed[str(seed)] = anchor
        build = seed_result.get("build")
        if not isinstance(build, Mapping):
            raise AnalysisError(f"seed {seed} build receipt is absent")
        core_initial = build.get("core_initial_sha256")
        if not isinstance(core_initial, Mapping) or set(core_initial) != set(ARMS):
            raise AnalysisError(f"seed {seed} arm initialization receipt changed")
        if len(set(core_initial.values())) != 1:
            raise AnalysisError(f"seed {seed} arm initializations differ")
        if _finite_number(
            build.get("dynamic_registered_parity_max_abs_error"),
            name=f"seed {seed} dynamic parity",
        ) != 0.0:
            raise AnalysisError(f"seed {seed} registered-route parity failed")
        if seed_result.get("terminal_decoder_sha256") != build.get("decoder_frozen_sha256"):
            raise AnalysisError(f"seed {seed} true-delta decoder changed during training")
        terminal = seed_result.get("terminal")
        if not isinstance(terminal, Mapping) or set(terminal) != set(ARMS):
            raise AnalysisError(f"seed {seed} terminal arm panel changed")
        for arm in ARMS:
            metrics = terminal[arm]
            if not isinstance(metrics, Mapping):
                raise AnalysisError(f"seed {seed} {arm} terminal metrics are absent")
            _validate_terminal_metrics(metrics, seed=seed, arm=arm)
        terminal_by_seed[seed] = terminal

    contrasts: dict[str, Any] = {}
    for metric, orientation in METRICS.items():
        per_seed: dict[str, Any] = {}
        effects = {"delta_main": [], "spatial_main": [], "interaction": []}
        for seed in SEEDS:
            terminal = terminal_by_seed[seed]
            raw = {arm: float(terminal[arm][metric]) for arm in ARMS}
            oriented = {arm: orientation * value for arm, value in raw.items()}
            masked_delta = oriented["masked_delta"] - oriented["masked_plain"]
            full_delta = oriented["full_delta"] - oriented["full_plain"]
            plain_spatial = oriented["full_plain"] - oriented["masked_plain"]
            delta_spatial = oriented["full_delta"] - oriented["masked_delta"]
            selected_effects = {
                "delta_main": 0.5 * (masked_delta + full_delta),
                "spatial_main": 0.5 * (plain_spatial + delta_spatial),
                "interaction": full_delta - masked_delta,
            }
            for name, value in selected_effects.items():
                effects[name].append(value)
            per_seed[str(seed)] = {
                "raw_cells": raw,
                "oriented_cells": oriented,
                "simple_effects": {
                    "delta_within_masked": masked_delta,
                    "delta_within_full": full_delta,
                    "full_within_plain": plain_spatial,
                    "full_within_delta": delta_spatial,
                },
                "factorial_effects": selected_effects,
            }
        contrasts[metric] = {
            "orientation": "higher_is_better" if orientation > 0 else "lower_is_better",
            "per_seed": per_seed,
            "across_seed": {name: _summary(values) for name, values in effects.items()},
        }

    primary = contrasts["hardest_wrong_action_margin_mean"]
    persistence = contrasts["persistence_advantage_mean"]
    primary_delta = primary["across_seed"]["delta_main"]
    persistence_delta = persistence["across_seed"]["delta_main"]
    plain_reference = mean(
        0.5
        * (
            primary["per_seed"][str(seed)]["raw_cells"]["masked_plain"]
            + primary["per_seed"][str(seed)]["raw_cells"]["full_plain"]
        )
        for seed in SEEDS
    )
    required_gap_closure = GAP_CLOSURE_FRACTION * max(0.0, -plain_reference)
    delta_proxy_supported = (
        primary_delta["positive_seed_count"] == len(SEEDS)
        and primary_delta["mean"] >= required_gap_closure
        and persistence_delta["mean"] >= 0.0
    )

    snapshots = _snapshot_bindings(result_path)
    result_raw = result_path.read_bytes()
    return {
        "schema": SCHEMA,
        "status": "PASS_COMPLETE_FIXED_COMPARISON_ANALYSIS",
        "development_only": True,
        "citable_as_world_model_usefulness_evidence": False,
        "input_result": {
            "path": str(result_path.resolve(strict=True)),
            "byte_count": len(result_raw),
            "sha256": hashlib.sha256(result_raw).hexdigest(),
        },
        "configuration": EXPECTED_CONFIGURATION,
        "decoder_anchor_by_seed": anchor_by_seed,
        "contrasts": contrasts,
        "proxy_routing": {
            "rule": (
                "delta hardest-margin main effect positive in all three seeds; mean effect "
                "closes at least 25% of the concurrent plain-arm mean deficit to zero; "
                "and delta persistence main effect is nonnegative"
            ),
            "plain_reference_hardest_margin_mean": plain_reference,
            "required_delta_main_effect": required_gap_closure,
            "delta_auxiliary_supported": delta_proxy_supported,
            "decision": (
                "DELTA_PROXY_MEANINGFUL"
                if delta_proxy_supported
                else "DELTA_PROXY_NOT_MEANINGFUL"
            ),
            "causal_branch_evaluation_still_required": True,
            "bulk_training_scale_authorized": False,
            "world_model_usefulness_claim_authorized": False,
        },
        "terminal_snapshot_bindings": snapshots,
        "uncertainty_limit": (
            "Student-t intervals describe three training seeds over one fixed validation "
            "panel; they are not fresh-scene, causal-branch, or deployment uncertainty."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _reject_protected_path(args.result)
    _reject_protected_path(args.output)
    if args.result.is_symlink() or not args.result.is_file():
        raise AnalysisError("runner result must be a regular non-symlink file")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = json.loads(args.result.read_bytes())
    analysis = analyze(payload, result_path=args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": analysis["status"], "decision": analysis["proxy_routing"]["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
