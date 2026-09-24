#!/usr/bin/env python3
"""Run the preregistered cosine-decay categorical-radial ladder v2."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "lewm_worlds"))

from scripts import run_go2_categorical_radial_ladder as v1  # noqa: E402


RESULT_SCHEMA = "lewm_go2_categorical_radial_ladder_result_v2"
SMOKE_RESULT_SCHEMA = "lewm_go2_categorical_radial_ladder_smoke_result_v2"
STAGE_SCHEMA = "lewm_go2_categorical_radial_ladder_stage_v2"
EVALUATION_SCHEMA = "lewm_go2_categorical_radial_ladder_evaluation_v2"
SCHEDULE_SCHEMA = "lewm_go2_categorical_radial_cosine_schedule_v2"
AMENDMENT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_ladder_v2_optimizer_amendment_2026-07-10.md"
)
EXPECTED_AMENDMENT_SHA256 = (
    "58f994a639c8e5a733d92c6da1fad63fa654e1f57aa7be0a8373e3eaa47b3f46"
)
FROZEN_LADDER_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_micro_overfit/v1/ladder_manifest.json"
)
EXPECTED_LADDER_FILE_SHA256 = (
    "967812399045b29e8be316f2f87bc16f02d681b0ea01884513c6b4f29bbe4b12"
)
V1_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_micro_overfit/v1/"
    "seed_20260710_ladder_result.json"
)
EXPECTED_V1_RESULT_FILE_SHA256 = (
    "72e4ecbe6b9e9024bb910e5231deb42e2d73f3187babd2a9af518251cbb7c2a2"
)
EXPECTED_V1_RESULT_CONTENT_SHA256 = (
    "02c627eb01e42a5b7e8ea57e5bd4bde3d1fc2ca0667abdd9dd1cf8162beacd52"
)
LEARNING_RATE_START = 2e-4
LEARNING_RATE_END = 1e-5

AUTHORITATIVE_STAGES = v1.AUTHORITATIVE_STAGES
SMOKE_STAGES = v1.SMOKE_STAGES
AUTHORITATIVE_EVALUATION_INTERVAL = v1.AUTHORITATIVE_EVALUATION_INTERVAL
SMOKE_EVALUATION_INTERVAL = v1.SMOKE_EVALUATION_INTERVAL
WEIGHT_DECAY = v1.WEIGHT_DECAY
GRADIENT_CLIP = v1.GRADIENT_CLIP
TRAINING_WEIGHTS = v1.TRAINING_WEIGHTS
LADDER_PREFIX_SIZES = v1.LADDER_PREFIX_SIZES
CategoricalRadialPerception = v1.CategoricalRadialPerception
LadderFrameDataset = v1.LadderFrameDataset
canonical_json_sha256 = v1.canonical_json_sha256


def cosine_learning_rate(update: int, total_updates: int) -> float:
    """Return the preregistered one-indexed stage-local cosine rate."""

    update = int(update)
    total_updates = int(total_updates)
    if total_updates <= 1:
        raise ValueError("cosine schedule requires at least two updates")
    if not 1 <= update <= total_updates:
        raise ValueError("cosine schedule update is outside the stage budget")
    if update == 1:
        return LEARNING_RATE_START
    if update == total_updates:
        return LEARNING_RATE_END
    progress = (update - 1) / (total_updates - 1)
    return LEARNING_RATE_END + 0.5 * (
        LEARNING_RATE_START - LEARNING_RATE_END
    ) * (1.0 + math.cos(math.pi * progress))


def learning_rate_schedule_contract(total_updates: int) -> dict[str, Any]:
    total_updates = int(total_updates)
    first = cosine_learning_rate(1, total_updates)
    final = cosine_learning_rate(total_updates, total_updates)
    return {
        "schema": SCHEDULE_SCHEMA,
        "name": "deterministic_stage_local_cosine_no_warmup_v2",
        "formula": (
            "1e-5 + 0.5 * (2e-4 - 1e-5) * "
            "(1 + cos(pi * (u - 1) / (U - 1)))"
        ),
        "update_indexing": "one_indexed_u_in_1_through_U",
        "total_updates": total_updates,
        "start_learning_rate": LEARNING_RATE_START,
        "end_learning_rate": LEARNING_RATE_END,
        "first_update_learning_rate": first,
        "final_update_learning_rate": final,
        "warmup_updates": 0,
        "stage_local_restart": True,
        "assignment_timing": "immediately_before_optimizer_step",
        "library_scheduler_used": False,
        "early_stopping": False,
        "ema": False,
        "checkpoint_averaging": False,
        "retry": False,
        "best_step_selection": False,
        "scope": "n1_n4_n16_train_only_ladder",
        "must_not_apply_to_n32_or_full_dataset": True,
    }


def _source_paths() -> dict[str, Path]:
    shared = {
        name: path
        for name, path in v1.SOURCE_PATHS.items()
        if name != "runner"
    }
    return {
        **shared,
        "amendment": AMENDMENT_PATH,
        "v1_result": V1_RESULT_PATH,
        "v1_runner": v1.SOURCE_PATHS["runner"],
        "runner": Path(__file__).resolve(),
    }


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": v1._sha256_file(path),
        }
        for name, path in sorted(_source_paths().items())
    }


def validate_bound_v1_result() -> dict[str, Any]:
    """Validate the exact immutable V1 evidence before any V2 model output."""

    path = V1_RESULT_PATH.resolve()
    if v1._sha256_file(path) != EXPECTED_V1_RESULT_FILE_SHA256:
        raise ValueError("immutable V1 result file SHA-256 mismatch")
    result = v1._read_json(path)
    core = dict(result)
    declared_content_sha256 = str(core.pop("content_sha256", ""))
    if (
        result.get("schema") != v1.RESULT_SCHEMA
        or declared_content_sha256 != EXPECTED_V1_RESULT_CONTENT_SHA256
        or canonical_json_sha256(core) != declared_content_sha256
    ):
        raise ValueError("immutable V1 result content SHA-256 mismatch")
    ladder_input = result.get("inputs", {}).get("ladder_manifest", {})
    if (
        result.get("authoritative") is not True
        or result.get("promotion_eligible") is not False
        or str(ladder_input.get("sha256", ""))
        != EXPECTED_LADDER_FILE_SHA256
        or result.get("source_hashes") != v1._source_hashes()
    ):
        raise ValueError("immutable V1 result provenance contract mismatch")
    decision = result.get("decision", {})
    if (
        decision.get("attempted_frame_counts") != [1, 4]
        or decision.get("stopped_on_first_failed_stage") is not True
        or decision.get("all_n1_n4_n16_gates_pass") is not False
    ):
        raise ValueError("immutable V1 result decision contract mismatch")
    return result


def validate_v2_preregistration() -> dict[str, Any]:
    if v1._sha256_file(AMENDMENT_PATH) != EXPECTED_AMENDMENT_SHA256:
        raise ValueError("V2 optimizer amendment SHA-256 mismatch")
    return validate_bound_v1_result()


@torch.no_grad()
def evaluate_ladder_model(
    model: torch.nn.Module,
    dataset: LadderFrameDataset,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    result = v1.evaluate_ladder_model(
        model,
        dataset,
        records,
        device=device,
        batch_size=batch_size,
    )
    return {**result, "schema": EVALUATION_SCHEMA}


def _assign_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    total_updates: int,
) -> float:
    learning_rate = cosine_learning_rate(update, total_updates)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    return learning_rate


def _train_stage(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_state: Mapping[str, torch.Tensor],
    initial_state_sha256: str,
    device: torch.device,
    seed: int,
    updates: int,
    batch_size: int,
    evaluation_interval: int,
) -> dict[str, Any]:
    frame_count = len(records)
    if frame_count not in LADDER_PREFIX_SIZES:
        raise ValueError("training frame count is not a registered ladder size")
    if updates <= 1 or evaluation_interval <= 0 or updates % evaluation_interval:
        raise ValueError(
            "stage updates must exceed one and be divisible by evaluation interval"
        )
    if batch_size <= 0 or frame_count % batch_size:
        raise ValueError("stage batch size must divide its frame count")
    model = CategoricalRadialPerception().to(device)
    model.load_state_dict(initial_state, strict=True)
    if v1._state_dict_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("stage did not restart from the frozen initial state")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_START,
        weight_decay=WEIGHT_DECAY,
    )
    dataset = LadderFrameDataset(records)
    generator = torch.Generator().manual_seed(int(seed))
    remaining_order: list[int] = []
    curve = []
    step = 0
    while step < updates:
        indices, remaining_order = v1._next_batch_indices(
            frame_count=frame_count,
            batch_size=batch_size,
            generator=generator,
            order=remaining_order,
        )
        raw_batch = dataset.batch(indices)
        image = raw_batch["image"].to(device)
        labels = raw_batch["labels"].to(device)
        mask = raw_batch["mask"].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = v1.hierarchical_occupancy_loss(logits, labels, mask)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite ladder loss at step {step + 1}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP
        )
        update = step + 1
        learning_rate = _assign_learning_rate(
            optimizer,
            update=update,
            total_updates=updates,
        )
        optimizer.step()
        step = update
        if step % evaluation_interval == 0:
            evaluation = evaluate_ladder_model(
                model,
                dataset,
                records,
                device=device,
                batch_size=batch_size,
            )
            curve.append(
                {
                    "step": step,
                    "learning_rate_for_update": learning_rate,
                    "batch_loss": float(loss.detach().item()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "evaluation": evaluation,
                }
            )
    if not curve or int(curve[-1]["step"]) != updates:
        raise RuntimeError("ladder stage lacks its exact final evaluation")
    schedule = learning_rate_schedule_contract(updates)
    final_evaluation = curve[-1]["evaluation"]
    result = {
        "schema": STAGE_SCHEMA,
        "frame_count": frame_count,
        "updates": updates,
        "completed_updates": step,
        "batch_size": batch_size,
        "evaluation_interval": evaluation_interval,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "learning_rate_schedule": schedule,
        },
        "fixed_budget_consumed": step == updates,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": v1._state_dict_sha256(model.state_dict()),
        "curve": curve,
        "final_evaluation": final_evaluation,
        "final_fit_gate_passes": bool(final_evaluation["fit_gate"]["passes"]),
        "access_ledger": dataset.access_ledger(),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-manifest", type=Path, required=True)
    parser.add_argument("--expected-ladder-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--non-authoritative-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; result artifacts are immutable")
    if str(args.expected_ladder_sha256) != EXPECTED_LADDER_FILE_SHA256:
        parser.error("expected-ladder-sha256 differs from the frozen V1 ladder")
    if args.seed not in (20260710, 20260711):
        parser.error("seed must be 20260710 or 20260711")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    invocation = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *(str(value) for value in argv)]
    )
    started_at = datetime.now(timezone.utc).isoformat()
    manifest_path = args.ladder_manifest.resolve()
    output_path = args.output.resolve()
    if manifest_path != FROZEN_LADDER_MANIFEST_PATH.resolve():
        raise ValueError("V2 requires the frozen V1 ladder manifest path")
    expected_manifest_sha256 = str(args.expected_ladder_sha256)
    manifest_file_sha256 = v1._sha256_file(manifest_path)
    if manifest_file_sha256 != EXPECTED_LADDER_FILE_SHA256:
        raise ValueError("frozen V1 ladder manifest SHA-256 mismatch")

    v1_result = validate_v2_preregistration()
    source_start = _source_hashes()
    git_start = v1._git_snapshot()
    manifest = v1._read_json(manifest_path)
    selected = v1.validate_ladder_manifest(manifest)
    images, shards = v1._artifact_contract(selected)
    v1._verify_artifacts(images, shards)
    device = v1._resolve_device(str(args.device))
    deterministic = v1._configure_determinism(int(args.seed))

    initial_model = CategoricalRadialPerception()
    initial_state = v1._clone_state(initial_model.state_dict())
    initial_state_sha256 = v1._state_dict_sha256(initial_state)
    v1_initial_state_sha256 = str(
        v1_result.get("model", {}).get("initial_state_sha256", "")
    )
    if int(args.seed) == 20260710 and (
        initial_state_sha256 != v1_initial_state_sha256
    ):
        raise RuntimeError("V2 seed-20260710 initialization differs from V1")
    model_parameter_count = sum(
        parameter.numel() for parameter in initial_model.parameters()
    )
    del initial_model

    smoke = bool(args.non_authoritative_smoke)
    stage_configs = SMOKE_STAGES if smoke else AUTHORITATIVE_STAGES
    evaluation_interval = (
        SMOKE_EVALUATION_INTERVAL
        if smoke
        else AUTHORITATIVE_EVALUATION_INTERVAL
    )
    stages = []
    for frame_count in LADDER_PREFIX_SIZES:
        config = stage_configs[frame_count]
        stage = _train_stage(
            selected[:frame_count],
            initial_state=initial_state,
            initial_state_sha256=initial_state_sha256,
            device=device,
            seed=int(args.seed),
            updates=int(config["updates"]),
            batch_size=int(config["batch_size"]),
            evaluation_interval=evaluation_interval,
        )
        stages.append(stage)
        if not smoke and not stage["final_fit_gate_passes"]:
            break

    v1._verify_artifacts(images, shards)
    if v1._sha256_file(manifest_path) != manifest_file_sha256:
        raise RuntimeError("ladder manifest changed during execution")
    panel_path = Path(str(manifest["inputs"]["panel_manifest"]["path"])).resolve()
    panel_sha256 = str(manifest["inputs"]["panel_manifest"]["sha256"])
    if v1._sha256_file(panel_path) != panel_sha256:
        raise RuntimeError("frozen parent panel changed during execution")
    if v1._sha256_file(V1_RESULT_PATH) != EXPECTED_V1_RESULT_FILE_SHA256:
        raise RuntimeError("immutable V1 result changed during V2 execution")
    if v1._sha256_file(AMENDMENT_PATH) != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("V2 optimizer amendment changed during execution")
    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("categorical-radial V2 sources changed during execution")
    git_end = v1._git_snapshot()

    completed_sizes = [int(stage["frame_count"]) for stage in stages]
    all_passed = completed_sizes == list(LADDER_PREFIX_SIZES) and all(
        bool(stage["final_fit_gate_passes"]) for stage in stages
    )
    access_totals: Counter[str] = Counter()
    for stage in stages:
        for key, value in stage["access_ledger"].items():
            if isinstance(value, int):
                access_totals[key] += value
    execution = {
        "authoritative": not smoke,
        "non_authoritative_smoke": smoke,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "stage_configs": {
            str(size): dict(config) for size, config in stage_configs.items()
        },
        "evaluation_interval": evaluation_interval,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "learning_rate_schedules": {
                str(size): learning_rate_schedule_contract(int(config["updates"]))
                for size, config in stage_configs.items()
            },
        },
        "determinism": deterministic,
    }
    core = {
        "schema": SMOKE_RESULT_SCHEMA if smoke else RESULT_SCHEMA,
        "created_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation,
        "authoritative": not smoke,
        "promotion_eligible": False,
        "train_only_implementation_diagnostic": True,
        "g2_evaluated": False,
        "v2_sole_intervention": "stage_local_cosine_learning_rate_no_warmup",
        "v2_scope": {
            "adaptive_post_hoc_train_only_schedule": True,
            "schedule_sweep_permitted": False,
            "applies_only_to_n1_n4_n16": True,
            "n32_optimizer_branches_unchanged": True,
            "full_dataset_optimizer_unchanged": True,
            "rocm_grid_sample_backward_warn_only_nondeterminism": True,
        },
        "inputs": {
            "ladder_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_file_sha256,
                "expected_sha256": expected_manifest_sha256,
                "content_sha256": str(manifest["content_sha256"]),
                "hash_stable_through_execution": True,
            },
            "parent_panel": {
                "path": str(panel_path),
                "sha256": panel_sha256,
                "hash_stable_through_execution": True,
            },
            "optimizer_amendment": {
                "path": str(AMENDMENT_PATH.resolve()),
                "sha256": EXPECTED_AMENDMENT_SHA256,
                "hash_stable_through_execution": True,
            },
            "immutable_v1_result": {
                "path": str(V1_RESULT_PATH.resolve()),
                "sha256": EXPECTED_V1_RESULT_FILE_SHA256,
                "content_sha256": EXPECTED_V1_RESULT_CONTENT_SHA256,
                "hash_stable_through_execution": True,
            },
        },
        "source_hashes": source_end,
        "git": {"start": git_start, "end": git_end},
        "execution": execution,
        "model": {
            "class": "CategoricalRadialPerception",
            "parameter_count": model_parameter_count,
            "initial_state_sha256": initial_state_sha256,
            "v1_seed_20260710_initial_state_sha256": v1_initial_state_sha256,
            "matches_v1_initial_state": (
                initial_state_sha256 == v1_initial_state_sha256
                if int(args.seed) == 20260710
                else None
            ),
            "stage_restart_initial_hashes_equal": all(
                stage["initial_state_sha256"] == initial_state_sha256
                for stage in stages
            ),
        },
        "training_weights": {
            name: list(map(float, values))
            for name, values in TRAINING_WEIGHTS.items()
        },
        "stages": stages,
        "decision": {
            "attempted_frame_counts": completed_sizes,
            "stopped_on_first_failed_stage": (
                not smoke
                and not all_passed
                and bool(stages)
                and not stages[-1]["final_fit_gate_passes"]
            ),
            "authoritative_first_failure_stop_policy_enforced": not smoke,
            "smoke_exercised_all_stage_paths": smoke
            and completed_sizes == list(LADDER_PREFIX_SIZES),
            "all_n1_n4_n16_gates_pass": all_passed,
            "n32_fit_panel_diagnostic_licensed": all_passed and not smoke,
            "n32_attempted": False,
            "promotion_licensed": False,
        },
        "artifact_access_ledger": {
            "selected_train_images_hashed_per_pass": len(images),
            "selected_train_label_shards_hashed_per_pass": len(shards),
            "integrity_hash_passes": 2,
            "selected_train_image_hash_byte_open_events": 2 * len(images),
            "selected_train_label_shard_hash_byte_open_events": 2 * len(shards),
            "immutable_v1_result_integrity_hash_passes": 4,
            "immutable_v1_result_json_deserializations": 1,
            "optimizer_amendment_integrity_hash_passes": 4,
            "stage_totals": dict(sorted(access_totals.items())),
            "checkpoint_selection": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "probability_calibration": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "g2_evaluation": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "non_train_image_opens": 0,
            "non_train_label_shard_opens": 0,
            "non_train_model_outputs": 0,
        },
    }
    payload = {**core, "content_sha256": canonical_json_sha256(core)}
    v1._atomic_write_json_exclusive(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "file_sha256": v1._sha256_file(output_path),
                "content_sha256": payload["content_sha256"],
                "schema": payload["schema"],
                "attempted_frame_counts": completed_sizes,
                "all_gates_pass": all_passed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
