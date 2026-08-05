#!/usr/bin/env python3
"""Run the frozen exposure-matched categorical-radial N32 V2 diagnostic."""
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

from lewm.benchmarks.go2_categorical_radial_n32 import (  # noqa: E402
    HOLDOUT_PANELS,
    categorical_holdout_checks,
    terminal_fit_gate_summary,
)
from lewm.benchmarks.go2_categorical_radial_n32_v2 import (  # noqa: E402
    EXECUTION_BINDING_SHA256,
    RESULT_SCHEMA,
    SMOKE_RESULT_SCHEMA,
    STAGE_NAME,
    STAGE_SCHEMA,
    per_seed_decision,
)
from lewm.models.categorical_radial_perception_full_ray import (  # noqa: E402
    CategoricalRadialPerceptionFullRay,
    REGISTERED_PARAMETER_COUNT,
)
from scripts import run_go2_categorical_radial_n32 as v1  # noqa: E402


PANEL_PATH = v1.PANEL_PATH
PANEL_FILE_SHA256 = v1.PANEL_FILE_SHA256
PANEL_CONTENT_SHA256 = v1.PANEL_CONTENT_SHA256
PANEL_ROWS_SHA256 = v1.PANEL_ROWS_SHA256
LADDER_PATH = v1.LADDER_PATH
LADDER_FILE_SHA256 = v1.LADDER_FILE_SHA256
LADDER_CONTENT_SHA256 = v1.LADDER_CONTENT_SHA256
V3_RESULT_PATH = v1.V3_RESULT_PATH
V3_RESULT_FILE_SHA256 = v1.V3_RESULT_FILE_SHA256
V3_RESULT_CONTENT_SHA256 = v1.V3_RESULT_CONTENT_SHA256
PATCH7_RESULT_PATH = v1.PATCH7_RESULT_PATH
PATCH7_RESULT_FILE_SHA256 = v1.PATCH7_RESULT_FILE_SHA256
PATCH7_RESULT_CONTENT_SHA256 = v1.PATCH7_RESULT_CONTENT_SHA256
PROTOCOL_PATH = v1.PROTOCOL_PATH
PROTOCOL_SHA256 = v1.PROTOCOL_SHA256
V1_CONTRACT_PATH = v1.CONTRACT_PATH
V1_CONTRACT_SHA256 = v1.EXECUTION_BINDING_SHA256
V1_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_n32/v1/seed_20260710_result.json"
)
V1_RESULT_FILE_SHA256 = (
    "2f079925000ebbcd06843c413f4dcfd07fce93358482dd05512735af69cbc946"
)
V1_RESULT_CONTENT_SHA256 = (
    "ef023faff0e49888ca673cfab5fca0c1110852e49312ce339ecb7f03ab3a8d5b"
)
V1_RESULT_NOTE_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_categorical_radial_n32_v1_result_2026-07-11.md"
)
V1_RESULT_NOTE_SHA256 = (
    "4848c61e72be3b81bb4fe4ad0e545f9c3e6031df353c9d25f15a9dcd5109ddfd"
)
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_n32_v2_exposure_binding_2026-07-11.md"
)
EXPECTED_INITIAL_STATE_SHA256 = {
    20260710: "8b149b57ae4bb305a2306a4dde2cab5f57a46f1c3760837593ed4d9862491278",
    20260711: "989e2db491d199bc544fabe2df40443a39f3ffc6e936f0d28c24625e7bd0ce13",
}
EXPECTED_PANEL_ARTIFACT_COUNTS = v1.EXPECTED_PANEL_ARTIFACT_COUNTS
AUTHORITATIVE_CONFIG = {
    "updates": 2000,
    "batch_size": 80,
    "learning_rate_start": 2e-4,
    "learning_rate_end": 1e-5,
    "weight_decay": 1e-4,
}
SMOKE_CONFIG = {**AUTHORITATIVE_CONFIG, "updates": 3}
AUTHORITATIVE_EVALUATION_INTERVAL = 100
SMOKE_EVALUATION_INTERVAL = 1
BATCH_SIZE = 80
EVALUATION_TARGET_BATCH_SIZE = 4
GRADIENT_CLIP = 1.0
EVENT_FIELDS = (
    "image_requests",
    "target_requests",
    "image_decode_events",
    "label_shard_npz_open_events",
    "model_calls",
    "model_output_frames",
)
DATA_EVENT_FIELDS = EVENT_FIELDS[:4]

PanelFrameDataset = v1.PanelFrameDataset
_canonical_panel_records = v1._canonical_panel_records
_artifact_contract = v1._artifact_contract
_verify_artifacts = v1._verify_artifacts
evaluate_panel = v1.evaluate_panel
direct_hierarchical_loss = v1.direct_hierarchical_loss


def _sha256_file(path: Path) -> str:
    return v1._sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    return v1._read_json(path)


def _source_paths() -> dict[str, Path]:
    shared = {
        ("v1_n32_runner" if name == "runner" else name): path
        for name, path in v1._source_paths().items()
    }
    return {
        **shared,
        "n32_v2_binding": CONTRACT_PATH,
        "n32_v2_pure": (
            REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32_v2.py"
        ),
        "v2_finalizer": (
            REPOSITORY_ROOT / "scripts/finalize_go2_categorical_radial_n32_v2.py"
        ),
        "v1_n32_finalizer": (
            REPOSITORY_ROOT / "scripts/finalize_go2_categorical_radial_n32.py"
        ),
        "runner": Path(__file__).resolve(),
    }


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in sorted(_source_paths().items())
    }


def _validate_content(
    payload: Mapping[str, Any],
    *,
    schema: str,
    content_sha256: str,
    name: str,
) -> None:
    core = dict(payload)
    declared = core.pop("content_sha256", None)
    if (
        payload.get("schema") != schema
        or declared != content_sha256
        or v1.v3.canonical_json_sha256(core) != declared
    ):
        raise ValueError(f"{name} content contract mismatch")


def _load_bound_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate every frozen input before deserializing the V1 result."""

    expected = {
        CONTRACT_PATH: EXECUTION_BINDING_SHA256,
        V1_RESULT_PATH: V1_RESULT_FILE_SHA256,
        V1_RESULT_NOTE_PATH: V1_RESULT_NOTE_SHA256,
    }
    for path, digest in expected.items():
        if _sha256_file(path) != digest:
            raise ValueError(f"bound N32 V2 evidence SHA-256 mismatch: {path}")
    panel, panels, ladder, v3_result, reference = v1._load_bound_inputs()
    v1_result = _read_json(V1_RESULT_PATH)
    _validate_content(
        v1_result,
        schema=v1.RESULT_SCHEMA,
        content_sha256=V1_RESULT_CONTENT_SHA256,
        name="N32 V1 result",
    )
    if (
        v1_result.get("authoritative") is not True
        or v1_result.get("aggregation_eligible") is not True
        or v1_result.get("decision", {}).get("classification")
        != "fit_gate_failed"
        or v1_result.get("decision", {}).get("favorable") is not False
        or v1_result.get("categorical_radial_full_train_candidate_licensed")
        is not False
        or v1_result.get("contract")
        != {
            "path": str(V1_CONTRACT_PATH.resolve()),
            "sha256": V1_CONTRACT_SHA256,
        }
    ):
        raise ValueError("bound N32 V1 failure identity changed")
    return panel, panels, ladder, v3_result, reference, v1_result


def cosine_learning_rate(update: int, total_updates: int) -> float:
    """Return the exact one-indexed, stage-local V3 cosine learning rate."""

    update = int(update)
    total_updates = int(total_updates)
    if total_updates < 2 or update < 1 or update > total_updates:
        raise ValueError("V2 cosine update is outside a valid fixed budget")
    low, high = 1e-5, 2e-4
    return low + 0.5 * (high - low) * (
        1.0 + math.cos(math.pi * (update - 1) / (total_updates - 1))
    )


def learning_rate_schedule_contract(total_updates: int) -> dict[str, Any]:
    return {
        "name": "deterministic_stage_local_cosine_no_warmup_v3",
        "formula": (
            "1e-5 + 0.5 * (2e-4 - 1e-5) * "
            "(1 + cos(pi * (u - 1) / (U - 1)))"
        ),
        "one_indexed": True,
        "total_updates": int(total_updates),
        "first_learning_rate": cosine_learning_rate(1, total_updates),
        "final_learning_rate": cosine_learning_rate(total_updates, total_updates),
        "warmup_updates": 0,
        "assignment_timing": "immediately_before_optimizer_step",
    }


def frozen_minibatch_schedule(
    frame_count: int,
    updates: int,
    seed: int,
) -> list[list[int]]:
    """Generate complete deterministic epochs split into ordered batches of 80."""

    frame_count = int(frame_count)
    updates = int(updates)
    if frame_count != 320 or updates <= 0 or frame_count % BATCH_SIZE:
        raise ValueError("N32 V2 requires 320 frames and positive batch-80 updates")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    batches: list[list[int]] = []
    while len(batches) < updates:
        epoch = torch.randperm(frame_count, generator=generator).tolist()
        batches.extend(
            [epoch[start : start + BATCH_SIZE] for start in range(0, 320, 80)]
        )
    return batches[:updates]


def _normalized_events(value: Mapping[str, int]) -> dict[str, int]:
    return {name: int(value.get(name, 0)) for name in EVENT_FIELDS}


def _normalized_data_events(value: Mapping[str, int]) -> dict[str, int]:
    return {name: int(value.get(name, 0)) for name in DATA_EVENT_FIELDS}


def _run_stage(
    *,
    config: Mapping[str, Any],
    initial_state: Mapping[str, torch.Tensor],
    initial_state_sha256: str,
    dataset: PanelFrameDataset,
    records: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    device: torch.device,
    seed: int,
    evaluation_interval: int,
) -> tuple[dict[str, Any], torch.nn.Module]:
    model = CategoricalRadialPerceptionFullRay().to(device)
    model.load_state_dict(initial_state, strict=True)
    if v1.v3.v2.v1._state_dict_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("N32 V2 did not start from the registered initial state")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate_start"]),
        weight_decay=float(config["weight_decay"]),
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )
    updates = int(config["updates"])
    schedule = frozen_minibatch_schedule(len(records), updates, seed)
    before_training = dataset.snapshot()
    evaluation_access: Counter[str] = Counter()
    curve = []
    for step, indices in enumerate(schedule, start=1):
        if len(indices) != BATCH_SIZE:
            raise RuntimeError("N32 V2 direct training batch was not 80 frames")
        batch = dataset.training_batch(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device=device, dtype=torch.float32)
        if int(images.shape[0]) != BATCH_SIZE:
            raise RuntimeError("N32 V2 training tensor was not a direct batch 80")
        logits = model(images)
        loss = direct_hierarchical_loss(
            logits,
            batch["labels"].to(device),
            batch["mask"].to(device),
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite N32 V2 loss at update {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP
        )
        learning_rate = cosine_learning_rate(step, updates)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.step()
        if step % int(evaluation_interval) == 0:
            fit_report, access = evaluate_panel(
                model,
                dataset,
                records,
                device=device,
                panel="fit",
                controls=controls,
            )
            evaluation_access.update(access)
            curve.append(
                {
                    "step": step,
                    "learning_rate": learning_rate,
                    "batch_loss": float(loss.detach().item()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "fit_panel": fit_report,
                }
            )
    terminal = terminal_fit_gate_summary(curve, updates, evaluation_interval)
    total_delta = _normalized_events(dataset.delta(before_training))
    evaluation = _normalized_events(evaluation_access)
    training = {
        name: total_delta[name] - evaluation[name]
        for name in EVENT_FIELDS
    }
    training["model_calls"] = updates
    training["model_output_frames"] = updates * BATCH_SIZE
    return {
        "schema": STAGE_SCHEMA,
        "stage": STAGE_NAME,
        "maximum_steps": updates,
        "completed_steps": updates,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": 4,
        "effective_epochs": updates / 4,
        "frame_presentations": updates * BATCH_SIZE,
        "presentations_per_fit_frame": updates / 4,
        "evaluation_interval": int(evaluation_interval),
        "optimizer": {
            "name": "AdamW",
            "weight_decay": float(config["weight_decay"]),
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
            "gradient_clip": GRADIENT_CLIP,
            "gradient_clip_applications": updates,
            "learning_rate_schedule": learning_rate_schedule_contract(updates),
        },
        "one_direct_forward_backward_per_update": True,
        "gradient_accumulation_or_microbatching": False,
        "fixed_update_budget_consumed": True,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": v1.v3.v2.v1._state_dict_sha256(model.state_dict()),
        "minibatch_indices": schedule,
        "minibatch_indices_sha256": v1.v3.canonical_json_sha256(schedule),
        "learning_curve": curve,
        "terminal_fit_gate": terminal,
        "training_access": training,
        "fit_evaluation_access": evaluation,
        "holdouts_evaluated": False,
    }, model


def _canonical_output(seed: int) -> Path:
    return (
        REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_n32/v2/"
        f"seed_{int(seed)}_result.json"
    ).resolve()


def _validate_primary_authorization(
    path: Path,
    expected_sha256: str,
    current_sources: Mapping[str, Any],
    patch7_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the pure strict validator before any seed-11 device construction."""

    from scripts import finalize_go2_categorical_radial_n32_v2 as finalizer

    return finalizer.validate_seed10_authorization(
        path,
        expected_sha256,
        expected_runner_sources=current_sources,
        expected_patch7_reference=patch7_reference,
    )


def _reconcile_access(
    fit_dataset: PanelFrameDataset,
    stage: Mapping[str, Any],
    panel_access: Mapping[str, Any],
    holdouts: Mapping[str, Any] | None,
) -> None:
    training = _normalized_events(stage["training_access"])
    evaluation = _normalized_events(stage["fit_evaluation_access"])
    updates = int(stage["completed_steps"])
    curve_count = len(stage["learning_curve"])
    expected_training = {
        "image_requests": updates * BATCH_SIZE,
        "target_requests": updates * BATCH_SIZE,
        "model_calls": updates,
        "model_output_frames": updates * BATCH_SIZE,
    }
    expected_evaluation = {
        "image_requests": curve_count * 960,
        "target_requests": curve_count * 320,
        "model_calls": curve_count * 80,
        "model_output_frames": curve_count * 960,
    }
    if any(training[name] != value for name, value in expected_training.items()):
        raise RuntimeError("N32 V2 direct-training access does not reconcile")
    if any(evaluation[name] != value for name, value in expected_evaluation.items()):
        raise RuntimeError("N32 V2 fit-evaluation access does not reconcile")
    totals = _normalized_data_events(fit_dataset.snapshot())
    if any(
        totals[name] != training[name] + evaluation[name]
        for name in DATA_EVENT_FIELDS
    ):
        raise RuntimeError("N32 V2 fit access totals do not reconcile")
    if totals["image_decode_events"] != 320 or totals[
        "label_shard_npz_open_events"
    ] != 20:
        raise RuntimeError("N32 V2 fit cache chronology changed")
    if holdouts is None:
        for panel in HOLDOUT_PANELS:
            record = panel_access[panel]
            if record.get("authorized") is not False or any(
                int(record.get("dataset_access", {}).get(name, 0)) != 0
                for name in EVENT_FIELDS
            ):
                raise RuntimeError("unauthorized N32 V2 holdout access was recorded")
        return
    for panel in HOLDOUT_PANELS:
        access = _normalized_events(panel_access[panel]["dataset_access"])
        expected = {
            "image_requests": 960,
            "target_requests": 320,
            "image_decode_events": 320,
            "label_shard_npz_open_events": EXPECTED_PANEL_ARTIFACT_COUNTS[panel][
                "shards"
            ],
            "model_calls": 80,
            "model_output_frames": 960,
        }
        if access != expected:
            raise RuntimeError(f"N32 V2 {panel} access does not reconcile")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--seed-20260710-result", type=Path)
    parser.add_argument("--expected-seed-20260710-sha256")
    parser.add_argument("--non-authoritative-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; N32 V2 results are immutable")
    if args.seed not in EXPECTED_INITIAL_STATE_SHA256:
        parser.error("N32 V2 seed must be 20260710 or 20260711")
    authorization = (
        args.seed_20260710_result,
        args.expected_seed_20260710_sha256,
    )
    if args.seed == 20260710 and any(value is not None for value in authorization):
        parser.error("seed 20260710 rejects seed-authorization arguments")
    if args.seed == 20260711 and any(value is None for value in authorization):
        parser.error("seed 20260711 requires both seed-authorization arguments")
    if args.non_authoritative_smoke and args.seed != 20260710:
        parser.error("N32 V2 smoke is seed-20260710-only")
    if not args.non_authoritative_smoke and args.output.resolve() != _canonical_output(
        args.seed
    ):
        parser.error("authoritative N32 V2 output must use its canonical path")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    invocation = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *(str(value) for value in argv)]
    )
    started = datetime.now(timezone.utc).isoformat()
    source_start = _source_hashes()
    panel, panels, ladder, v3_result, patch7_reference, v1_result = (
        _load_bound_inputs()
    )
    git_start = v1.v3.v2.v1._git_snapshot()
    primary = None
    if args.seed == 20260711:
        primary = _validate_primary_authorization(
            args.seed_20260710_result.resolve(),
            str(args.expected_seed_20260710_sha256),
            source_start,
            patch7_reference,
        )

    device = v1.v3.v2.v1._resolve_device(str(args.device))
    determinism = v1.v3.v2.v1._configure_determinism(int(args.seed))
    initial_model = CategoricalRadialPerceptionFullRay()
    parameter_count = sum(parameter.numel() for parameter in initial_model.parameters())
    if parameter_count != REGISTERED_PARAMETER_COUNT:
        raise RuntimeError("N32 V2 model parameter count changed")
    initial_state = v1.v3.v2.v1._clone_state(initial_model.state_dict())
    initial_state_sha256 = v1.v3.v2.v1._state_dict_sha256(initial_state)
    if initial_state_sha256 != EXPECTED_INITIAL_STATE_SHA256[int(args.seed)]:
        raise RuntimeError("N32 V2 registered initialization changed")
    del initial_model

    fit_records, fit_controls = _canonical_panel_records(
        panels["fit"], seed=args.seed, panel="fit"
    )
    fit_images, fit_shards = _artifact_contract(fit_records, "fit")
    _verify_artifacts(fit_images, fit_shards)
    fit_dataset = PanelFrameDataset(fit_records, "fit")
    smoke = bool(args.non_authoritative_smoke)
    config = SMOKE_CONFIG if smoke else AUTHORITATIVE_CONFIG
    interval = SMOKE_EVALUATION_INTERVAL if smoke else AUTHORITATIVE_EVALUATION_INTERVAL
    stage, model = _run_stage(
        config=config,
        initial_state=initial_state,
        initial_state_sha256=initial_state_sha256,
        dataset=fit_dataset,
        records=fit_records,
        controls=fit_controls,
        device=device,
        seed=args.seed,
        evaluation_interval=interval,
    )

    holdouts = None
    holdout_checks = None
    panel_access: dict[str, Any] = {
        "fit": {
            "authorized": True,
            "artifact_hash_passes": 2,
            "image_hash_byte_open_events": 2 * len(fit_images),
            "shard_hash_byte_open_events": 2 * len(fit_shards),
        }
    }
    if stage["terminal_fit_gate"]["passes"]:
        holdouts, holdout_checks = {}, {}
        for panel_name in HOLDOUT_PANELS:
            records, controls = _canonical_panel_records(
                panels[panel_name], seed=args.seed, panel=panel_name
            )
            images, shards = _artifact_contract(records, panel_name)
            _verify_artifacts(images, shards)
            dataset = PanelFrameDataset(records, panel_name)
            report, access = evaluate_panel(
                model,
                dataset,
                records,
                device=device,
                panel=panel_name,
                controls=controls,
            )
            _verify_artifacts(images, shards)
            holdouts[panel_name] = report
            holdout_checks[panel_name] = categorical_holdout_checks(
                report, patch7_reference["panels"][panel_name]
            )
            panel_access[panel_name] = {
                "authorized": True,
                "artifact_hash_passes": 2,
                "image_hash_byte_open_events": 2 * len(images),
                "shard_hash_byte_open_events": 2 * len(shards),
                "dataset_access": _normalized_events(access),
            }
        stage["holdouts_evaluated"] = True
    else:
        for panel_name in HOLDOUT_PANELS:
            panel_access[panel_name] = {
                "authorized": False,
                "artifact_hash_passes": 0,
                "image_hash_byte_open_events": 0,
                "shard_hash_byte_open_events": 0,
                "dataset_access": {name: 0 for name in EVENT_FIELDS},
            }
    del model
    _verify_artifacts(fit_images, fit_shards)
    decision = per_seed_decision(stage, holdout_checks)
    decision["aggregation_eligible"] = not smoke

    if primary is not None and _sha256_file(args.seed_20260710_result.resolve()) != str(
        args.expected_seed_20260710_sha256
    ):
        raise RuntimeError("seed-20260710 V2 authorization changed during execution")
    evidence_hashes = {
        str(path.resolve()): digest
        for path, digest in {
            PANEL_PATH: PANEL_FILE_SHA256,
            LADDER_PATH: LADDER_FILE_SHA256,
            V3_RESULT_PATH: V3_RESULT_FILE_SHA256,
            PATCH7_RESULT_PATH: PATCH7_RESULT_FILE_SHA256,
            PROTOCOL_PATH: PROTOCOL_SHA256,
            V1_CONTRACT_PATH: V1_CONTRACT_SHA256,
            V1_RESULT_PATH: V1_RESULT_FILE_SHA256,
            V1_RESULT_NOTE_PATH: V1_RESULT_NOTE_SHA256,
            CONTRACT_PATH: EXECUTION_BINDING_SHA256,
        }.items()
    }
    for path, digest in evidence_hashes.items():
        if _sha256_file(Path(path)) != digest:
            raise RuntimeError(f"bound N32 V2 evidence changed: {path}")
    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("N32 V2 sources changed during execution")
    git_end = v1.v3.v2.v1._git_snapshot()
    _reconcile_access(fit_dataset, stage, panel_access, holdouts)
    access_ledger = {
        "panels": panel_access,
        "fit_dataset_totals": _normalized_data_events(fit_dataset.snapshot()),
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
    }
    authoritative = not smoke
    core = {
        "schema": RESULT_SCHEMA if authoritative else SMOKE_RESULT_SCHEMA,
        "authoritative": authoritative,
        "aggregation_eligible": authoritative,
        "promotion_eligible": False,
        "seed": int(args.seed),
        "created_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation,
        "execution": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "determinism": determinism,
            "batch_size_frames": BATCH_SIZE,
            "evaluation_target_batch_size": EVALUATION_TARGET_BATCH_SIZE,
            "evaluation_combined_model_batch_size": 12,
            "evaluation_interval": interval,
            "stage_config": config,
            "effective_epochs": int(config["updates"]) / 4,
            "fp32_no_autocast_amp_compile_or_quantization": True,
        },
        "contract": {
            "path": str(CONTRACT_PATH.resolve()),
            "sha256": EXECUTION_BINDING_SHA256,
        },
        "inputs": {
            "panel": {
                "path": str(PANEL_PATH.resolve()),
                "sha256": PANEL_FILE_SHA256,
                "content_sha256": PANEL_CONTENT_SHA256,
            },
            "ladder_manifest": {
                "path": str(LADDER_PATH.resolve()),
                "sha256": LADDER_FILE_SHA256,
                "content_sha256": LADDER_CONTENT_SHA256,
            },
            "v3_result": {
                "path": str(V3_RESULT_PATH.resolve()),
                "sha256": V3_RESULT_FILE_SHA256,
                "content_sha256": V3_RESULT_CONTENT_SHA256,
            },
            "patch7_reference_result": {
                "path": str(PATCH7_RESULT_PATH.resolve()),
                "sha256": PATCH7_RESULT_FILE_SHA256,
                "content_sha256": PATCH7_RESULT_CONTENT_SHA256,
            },
            "failed_v1_n32_result": {
                "path": str(V1_RESULT_PATH.resolve()),
                "sha256": V1_RESULT_FILE_SHA256,
                "content_sha256": V1_RESULT_CONTENT_SHA256,
            },
            "failed_v1_n32_result_note": {
                "path": str(V1_RESULT_NOTE_PATH.resolve()),
                "sha256": V1_RESULT_NOTE_SHA256,
            },
            "seed_20260710_authorization": (
                None
                if primary is None
                else {
                    "path": str(args.seed_20260710_result.resolve()),
                    "sha256": str(args.expected_seed_20260710_sha256),
                }
            ),
        },
        "source_hashes": source_end,
        "git": {"start": git_start, "end": git_end},
        "model": {
            "class": "CategoricalRadialPerceptionFullRay",
            "parameter_count": parameter_count,
            "initial_state_sha256": initial_state_sha256,
        },
        "stages": {STAGE_NAME: stage},
        "patch7_reference": patch7_reference,
        "holdouts": holdouts,
        "holdout_checks": holdout_checks,
        "decision": decision,
        "artifact_verification": {
            "fit_verified_before_access": True,
            "holdouts_verified_only_after_terminal_fit_pass": True,
            "evidence_hashes": evidence_hashes,
        },
        "access_ledger": access_ledger,
        "categorical_radial_full_train_candidate_licensed": False,
    }
    payload = {**core, "content_sha256": v1.v3.canonical_json_sha256(core)}
    v1.v3.v2.v1._atomic_write_json_exclusive(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "content_sha256": payload["content_sha256"],
                "decision": decision["classification"],
                "favorable": decision["favorable"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
