#!/usr/bin/env python3
"""Run the preregistered full-ray categorical-radial ladder v3."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "lewm_worlds"))

from lewm.models.categorical_radial_perception_full_ray import (  # noqa: E402
    CategoricalRadialPerceptionFullRay,
    RADIAL_DILATIONS,
    REGISTERED_PARAMETER_COUNT,
    REGISTERED_RADIAL_BIN_COUNT,
    direct_radial_reachability,
)
from scripts import run_go2_categorical_radial_ladder_v2 as v2  # noqa: E402


RESULT_SCHEMA = "lewm_go2_categorical_radial_ladder_result_v3"
SMOKE_RESULT_SCHEMA = "lewm_go2_categorical_radial_ladder_smoke_result_v3"
STAGE_SCHEMA = "lewm_go2_categorical_radial_ladder_stage_v3"
EVALUATION_SCHEMA = "lewm_go2_categorical_radial_ladder_evaluation_v3"
AMENDMENT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_ladder_v3_full_ray_amendment_2026-07-10.md"
)
EXPECTED_AMENDMENT_SHA256 = (
    "921fc48cf2a41924c720654c2d08fbd09ca6ce3ccc7c94ccb6600096a434fcbf"
)
V2_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_micro_overfit/v2/"
    "seed_20260710_ladder_result.json"
)
EXPECTED_V2_RESULT_FILE_SHA256 = (
    "06517e2c6641495a6262aa9f8a5cb45648912c575f1c3663df899c50a2867daa"
)
EXPECTED_V2_RESULT_CONTENT_SHA256 = (
    "8528ae02d6faaf25eb666d591e15180e82f74c9cf4d798c8322f9d5c50c910bc"
)
EXPECTED_V2_INITIAL_STATE_SHA256 = (
    "ad120b467aeabb60f20b9fd663d0438451298895ee87717a6795810bbb5b8f75"
)

FROZEN_LADDER_MANIFEST_PATH = v2.FROZEN_LADDER_MANIFEST_PATH
EXPECTED_LADDER_FILE_SHA256 = v2.EXPECTED_LADDER_FILE_SHA256
AUTHORITATIVE_STAGES = v2.AUTHORITATIVE_STAGES
SMOKE_STAGES = v2.SMOKE_STAGES
AUTHORITATIVE_EVALUATION_INTERVAL = v2.AUTHORITATIVE_EVALUATION_INTERVAL
SMOKE_EVALUATION_INTERVAL = v2.SMOKE_EVALUATION_INTERVAL
WEIGHT_DECAY = v2.WEIGHT_DECAY
GRADIENT_CLIP = v2.GRADIENT_CLIP
LEARNING_RATE_START = v2.LEARNING_RATE_START
TRAINING_WEIGHTS = v2.TRAINING_WEIGHTS
LADDER_PREFIX_SIZES = v2.LADDER_PREFIX_SIZES
LadderFrameDataset = v2.LadderFrameDataset
V2CategoricalRadialPerception = v2.CategoricalRadialPerception
canonical_json_sha256 = v2.canonical_json_sha256


def _source_paths() -> dict[str, Path]:
    shared = {}
    for name, path in v2._source_paths().items():
        if name == "amendment":
            name = "v2_amendment"
        elif name == "runner":
            name = "v2_runner"
        shared[name] = path
    return {
        **shared,
        "model_full_ray": (
            REPOSITORY_ROOT
            / "lewm/models/categorical_radial_perception_full_ray.py"
        ),
        "v2_result": V2_RESULT_PATH,
        "v3_amendment": AMENDMENT_PATH,
        "runner": Path(__file__).resolve(),
    }


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": v2.v1._sha256_file(path),
        }
        for name, path in sorted(_source_paths().items())
    }


def validate_bound_v2_result() -> dict[str, Any]:
    """Validate the exact immutable V2 evidence before any V3 model output."""

    path = V2_RESULT_PATH.resolve()
    if v2.v1._sha256_file(path) != EXPECTED_V2_RESULT_FILE_SHA256:
        raise ValueError("immutable V2 result file SHA-256 mismatch")
    result = v2.v1._read_json(path)
    core = dict(result)
    declared_content_sha256 = str(core.pop("content_sha256", ""))
    if (
        result.get("schema") != v2.RESULT_SCHEMA
        or declared_content_sha256 != EXPECTED_V2_RESULT_CONTENT_SHA256
        or canonical_json_sha256(core) != declared_content_sha256
    ):
        raise ValueError("immutable V2 result content SHA-256 mismatch")
    ladder_input = result.get("inputs", {}).get("ladder_manifest", {})
    optimizer_amendment = result.get("inputs", {}).get(
        "optimizer_amendment", {}
    )
    if (
        result.get("authoritative") is not True
        or result.get("promotion_eligible") is not False
        or str(ladder_input.get("sha256", ""))
        != EXPECTED_LADDER_FILE_SHA256
        or str(optimizer_amendment.get("sha256", ""))
        != v2.EXPECTED_AMENDMENT_SHA256
        or result.get("source_hashes") != v2._source_hashes()
    ):
        raise ValueError("immutable V2 result provenance contract mismatch")
    if (
        str(result.get("model", {}).get("initial_state_sha256", ""))
        != EXPECTED_V2_INITIAL_STATE_SHA256
    ):
        raise ValueError("immutable V2 initial-state SHA-256 mismatch")
    decision = result.get("decision", {})
    if (
        decision.get("attempted_frame_counts") != [1, 4, 16]
        or decision.get("stopped_on_first_failed_stage") is not True
        or decision.get("all_n1_n4_n16_gates_pass") is not False
        or decision.get("n32_attempted") is not False
    ):
        raise ValueError("immutable V2 result decision contract mismatch")
    return result


def validate_v3_preregistration() -> dict[str, Any]:
    if v2.v1._sha256_file(AMENDMENT_PATH) != EXPECTED_AMENDMENT_SHA256:
        raise ValueError("V3 full-ray amendment SHA-256 mismatch")
    return validate_bound_v2_result()


def common_initialization_report(
    v2_model: torch.nn.Module,
    v3_model: torch.nn.Module,
) -> dict[str, Any]:
    """Prove equality of all common state outside the replaced radial block."""

    excluded_prefix = "radial_context."
    v2_state = {
        name: tensor
        for name, tensor in v2_model.state_dict().items()
        if not name.startswith(excluded_prefix)
    }
    v3_state = {
        name: tensor
        for name, tensor in v3_model.state_dict().items()
        if not name.startswith(excluded_prefix)
    }
    if set(v2_state) != set(v3_state):
        raise RuntimeError("V3 changed common state names outside radial_context")
    mismatches = [
        name
        for name in sorted(v2_state)
        if not torch.equal(v2_state[name], v3_state[name])
    ]
    if mismatches:
        raise RuntimeError(
            "V3 changed common initialization outside radial_context: "
            f"{mismatches[:5]}"
        )
    v2_hash = v2.v1._state_dict_sha256(v2_state)
    v3_hash = v2.v1._state_dict_sha256(v3_state)
    return {
        "schema": "lewm_go2_categorical_radial_common_init_audit_v3",
        "excluded_prefix": excluded_prefix,
        "common_tensor_count": len(v2_state),
        "v2_common_state_sha256": v2_hash,
        "v3_common_state_sha256": v3_hash,
        "all_common_tensors_exactly_equal": v2_hash == v3_hash,
    }


def full_ray_architecture_report(model: torch.nn.Module) -> dict[str, Any]:
    blocks = tuple(model.radial_context)
    dilations = tuple(int(block.dilation) for block in blocks)
    if dilations != RADIAL_DILATIONS:
        raise RuntimeError("V3 radial dilation sequence changed")
    radial_parameter_count = sum(
        parameter.numel() for parameter in model.radial_context.parameters()
    )
    if radial_parameter_count != 99_840:
        raise RuntimeError("V3 radial-context parameter count changed")
    adjacencies, reachability = direct_radial_reachability()
    if reachability.shape != (
        REGISTERED_RADIAL_BIN_COUNT,
        REGISTERED_RADIAL_BIN_COUNT,
    ) or not bool(reachability.all()):
        raise RuntimeError("V3 direct radial reachability is incomplete")
    adjacency_hashes = [
        canonical_json_sha256(adjacency.tolist())
        for adjacency in adjacencies
    ]
    return {
        "schema": "lewm_go2_categorical_radial_full_ray_audit_v3",
        "dilations": list(dilations),
        "block_count": len(blocks),
        "parameters_per_block": [
            sum(parameter.numel() for parameter in block.parameters())
            for block in blocks
        ],
        "radial_context_parameter_count": radial_parameter_count,
        "nominal_receptive_field_bins": 127,
        "layer_clipped_adjacency_true_counts": [
            int(adjacency.sum()) for adjacency in adjacencies
        ],
        "layer_clipped_adjacency_sha256": adjacency_hashes,
        "direct_reachability_shape": list(reachability.shape),
        "direct_reachability_true_count": int(reachability.sum()),
        "direct_reachability_all_true": True,
        "direct_reachability_sha256": canonical_json_sha256(
            reachability.tolist()
        ),
        "circular_range_wrap": False,
    }


@torch.no_grad()
def evaluate_ladder_model(
    model: torch.nn.Module,
    dataset: LadderFrameDataset,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    result = v2.v1.evaluate_ladder_model(
        model,
        dataset,
        records,
        device=device,
        batch_size=batch_size,
    )
    return {**result, "schema": EVALUATION_SCHEMA}


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
    model = CategoricalRadialPerceptionFullRay().to(device)
    model.load_state_dict(initial_state, strict=True)
    if v2.v1._state_dict_sha256(model.state_dict()) != initial_state_sha256:
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
        indices, remaining_order = v2.v1._next_batch_indices(
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
        loss = v2.v1.hierarchical_occupancy_loss(logits, labels, mask)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite ladder loss at step {step + 1}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP
        )
        update = step + 1
        learning_rate = v2._assign_learning_rate(
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
    schedule = v2.learning_rate_schedule_contract(updates)
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
        "final_state_sha256": v2.v1._state_dict_sha256(model.state_dict()),
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
    if args.seed != 20260710:
        parser.error("V3 is preregistered only for seed 20260710")
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
        raise ValueError("V3 requires the frozen V1 ladder manifest path")
    manifest_file_sha256 = v2.v1._sha256_file(manifest_path)
    if manifest_file_sha256 != EXPECTED_LADDER_FILE_SHA256:
        raise ValueError("frozen V1 ladder manifest SHA-256 mismatch")

    v2_result = validate_v3_preregistration()
    source_start = _source_hashes()
    git_start = v2.v1._git_snapshot()
    manifest = v2.v1._read_json(manifest_path)
    selected = v2.v1.validate_ladder_manifest(manifest)
    images, shards = v2.v1._artifact_contract(selected)
    v2.v1._verify_artifacts(images, shards)
    device = v2.v1._resolve_device(str(args.device))

    deterministic = v2.v1._configure_determinism(int(args.seed))
    v2_model = V2CategoricalRadialPerception()
    v2_initial_state_sha256 = v2.v1._state_dict_sha256(v2_model.state_dict())
    expected_v2_initial = str(
        v2_result.get("model", {}).get("initial_state_sha256", "")
    )
    if (
        expected_v2_initial != EXPECTED_V2_INITIAL_STATE_SHA256
        or v2_initial_state_sha256 != EXPECTED_V2_INITIAL_STATE_SHA256
    ):
        raise RuntimeError("V3 runtime cannot reproduce the V2 initialization")
    deterministic = v2.v1._configure_determinism(int(args.seed))
    initial_model = CategoricalRadialPerceptionFullRay()
    initialization_audit = common_initialization_report(v2_model, initial_model)
    architecture_audit = full_ray_architecture_report(initial_model)
    del v2_model
    model_parameter_count = sum(
        parameter.numel() for parameter in initial_model.parameters()
    )
    if model_parameter_count != REGISTERED_PARAMETER_COUNT:
        raise RuntimeError("V3 full-ray model parameter count changed")
    initial_state = v2.v1._clone_state(initial_model.state_dict())
    initial_state_sha256 = v2.v1._state_dict_sha256(initial_state)
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

    v2.v1._verify_artifacts(images, shards)
    if v2.v1._sha256_file(manifest_path) != manifest_file_sha256:
        raise RuntimeError("ladder manifest changed during execution")
    panel_path = Path(str(manifest["inputs"]["panel_manifest"]["path"])).resolve()
    panel_sha256 = str(manifest["inputs"]["panel_manifest"]["sha256"])
    if v2.v1._sha256_file(panel_path) != panel_sha256:
        raise RuntimeError("frozen parent panel changed during execution")
    if v2.v1._sha256_file(V2_RESULT_PATH) != EXPECTED_V2_RESULT_FILE_SHA256:
        raise RuntimeError("immutable V2 result changed during V3 execution")
    if v2.v1._sha256_file(AMENDMENT_PATH) != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("V3 full-ray amendment changed during execution")
    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("categorical-radial V3 sources changed during execution")
    git_end = v2.v1._git_snapshot()

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
                str(size): v2.learning_rate_schedule_contract(
                    int(config["updates"])
                )
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
        "v3_sole_intervention": "six_block_dilated_full_ray_radial_context",
        "v3_scope": {
            "changes_only_radial_context": True,
            "bundled_effect_not_receptive_field_only": True,
            "bundle_components": [
                "full_ray_direct_convolutional_path",
                "five_additional_groupnorm_gelu_stages",
                "nonlinear_depth",
                "decoder_parameters",
                "decoder_compute",
            ],
            "longer_run_permitted": False,
            "schedule_change_permitted": False,
            "second_seed_permitted": False,
            "non_train_access_permitted": False,
            "n32_optimizer_branches_unchanged": True,
            "n32_or_full_dataset_attempted": False,
        },
        "inputs": {
            "ladder_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_file_sha256,
                "expected_sha256": EXPECTED_LADDER_FILE_SHA256,
                "content_sha256": str(manifest["content_sha256"]),
                "hash_stable_through_execution": True,
            },
            "parent_panel": {
                "path": str(panel_path),
                "sha256": panel_sha256,
                "hash_stable_through_execution": True,
            },
            "full_ray_amendment": {
                "path": str(AMENDMENT_PATH.resolve()),
                "sha256": EXPECTED_AMENDMENT_SHA256,
                "hash_stable_through_execution": True,
            },
            "immutable_v2_result": {
                "path": str(V2_RESULT_PATH.resolve()),
                "sha256": EXPECTED_V2_RESULT_FILE_SHA256,
                "content_sha256": EXPECTED_V2_RESULT_CONTENT_SHA256,
                "hash_stable_through_execution": True,
            },
        },
        "source_hashes": source_end,
        "git": {"start": git_start, "end": git_end},
        "execution": execution,
        "model": {
            "class": "CategoricalRadialPerceptionFullRay",
            "parameter_count": model_parameter_count,
            "registered_parameter_count": REGISTERED_PARAMETER_COUNT,
            "initial_state_sha256": initial_state_sha256,
            "v2_initial_state_sha256": v2_initial_state_sha256,
            "common_initialization_audit": initialization_audit,
            "full_ray_architecture_audit": architecture_audit,
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
            "n32_diagnostic_construction_licensed": all_passed and not smoke,
            "n32_attempted": False,
            "promotion_licensed": False,
        },
        "artifact_access_ledger": {
            "selected_train_images_hashed_per_pass": len(images),
            "selected_train_label_shards_hashed_per_pass": len(shards),
            "integrity_hash_passes": 2,
            "selected_train_image_hash_byte_open_events": 2 * len(images),
            "selected_train_label_shard_hash_byte_open_events": 2 * len(shards),
            "immutable_v2_result_integrity_hash_passes": 4,
            "immutable_v2_result_json_deserializations": 1,
            "full_ray_amendment_integrity_hash_passes": 4,
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
    v2.v1._atomic_write_json_exclusive(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "file_sha256": v2.v1._sha256_file(output_path),
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
