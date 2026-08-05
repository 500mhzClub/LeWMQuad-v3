#!/usr/bin/env python3
"""Run a development-only fit diagnostic for the dynamic full-ray radial head.

This path is intentionally incapable of opening holdout or G2 payloads.  It
uses the registered 320-frame fit panel, train-role attitude sidecar, metrics,
controls, optimizer budgets, and all-family fit gate only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "lewm_worlds"))

from lewm.benchmarks import go2_dynamic_cartesian_n32 as contract  # noqa: E402
from lewm.benchmarks.go2_physical_micro_overfit import (  # noqa: E402
    attach_role_global_shuffle,
    attach_same_scene_wrong_view,
    empty_raw_accumulator,
    finalize_raw_accumulator,
    frame_records,
    update_raw_accumulator,
    validate_panel_manifest,
)
from lewm.benchmarks.go2_categorical_radial_factorization import (  # noqa: E402
    build_radial_factorization,
)
from lewm.datasets.go2_attitude_sidecar import (  # noqa: E402
    FROZEN_BUILD_CONTRACT,
    canonical_json_sha256 as sidecar_json_sha256,
    load_attitude_sidecar_roles,
    row_identity_sha256,
)
from lewm.models.categorical_radial_perception_full_ray import (  # noqa: E402
    REGISTERED_PARAMETER_COUNT,
)
from lewm.models.dynamic_categorical_radial_perception_full_ray import (  # noqa: E402
    DynamicCategoricalRadialPerceptionFullRay,
)
from scripts import run_go2_dynamic_cartesian_n32 as backend  # noqa: E402


RESULT_SCHEMA = "lewm_go2_dynamic_categorical_radial_fit_dev_result_v1"
STAGE_SCHEMA = "lewm_go2_dynamic_categorical_radial_fit_dev_stage_v1"
PANEL_PATH = backend.PANEL_PATH
PANEL_FILE_SHA256 = backend.PANEL_FILE_SHA256
PANEL_CONTENT_SHA256 = backend.PANEL_CONTENT_SHA256
FIT_ROWS_SHA256 = backend.PANEL_ROWS_SHA256["fit"]
SIDECAR_MANIFEST_PATH = backend.SIDECAR_MANIFEST_PATH
SIDECAR_MANIFEST_FILE_SHA256 = backend.SIDECAR_MANIFEST_FILE_SHA256
SIDECAR_TRAIN_CONTENT_SHA256 = backend.SIDECAR_TRAIN_CONTENT_SHA256
SOURCE_WORKERS = backend.SOURCE_WORKERS
EVENT_FIELDS = backend.EVENT_FIELDS
SMOKE_UPDATES = 3
SMOKE_EVALUATION_INTERVAL = 1
CANONICAL_DIRECTORY = REPOSITORY_ROOT / ".generated/go2_dynamic_categorical_radial_fit/v1"
FAITHFUL_CONFIG = {
    "updates": 2000,
    "batch_size": 80,
    "learning_rate_start": 2e-4,
    "learning_rate_end": 1e-5,
    "weight_decay": 1e-4,
    "schedule": "deterministic_stage_local_cosine_no_warmup_v3",
}
CEILING_CONFIG = {
    "updates": 5000,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "schedule": "constant",
}


def _source_paths() -> dict[str, Path]:
    return {
        "backend_runner": REPOSITORY_ROOT / "scripts/run_go2_dynamic_cartesian_n32.py",
        "dynamic_model": REPOSITORY_ROOT / "lewm/models/dynamic_categorical_radial_perception_full_ray.py",
        "egomotion_geometry": REPOSITORY_ROOT / "lewm/models/egomotion_bev_jepa.py",
        "encoder": REPOSITORY_ROOT / "lewm/models/encoders.py",
        "factorization": REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_factorization.py",
        "metrics": REPOSITORY_ROOT / "lewm/benchmarks/go2_dynamic_cartesian_n32.py",
        "physical_metrics": REPOSITORY_ROOT / "lewm/benchmarks/go2_physical_micro_overfit.py",
        "runner": Path(__file__).resolve(),
        "runner_test": REPOSITORY_ROOT / "lewm/tests/test_run_go2_dynamic_categorical_radial_fit.py",
        "sidecar_library": REPOSITORY_ROOT / "lewm/datasets/go2_attitude_sidecar.py",
        "static_model": REPOSITORY_ROOT / "lewm/models/categorical_radial_perception_full_ray.py",
    }


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": backend._sha256_file(path)}
        for name, path in sorted(_source_paths().items())
    }


def _scene_sha256(scene_id: str) -> str:
    import hashlib

    return hashlib.sha256(scene_id.encode("utf-8")).hexdigest()


def _load_fit_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open panel metadata and train sidecar, but join only the fit rows."""

    panel = backend._read_json(
        PANEL_PATH, expected_sha256=PANEL_FILE_SHA256, name="N32 fit panel"
    )
    if panel.get("content_sha256") != PANEL_CONTENT_SHA256:
        raise ValueError("N32 panel content hash changed")
    panels = validate_panel_manifest(panel)
    if panel["panels"]["fit"].get("rows_sha256") != FIT_ROWS_SHA256:
        raise ValueError("registered fit rows changed")
    sidecar = load_attitude_sidecar_roles(
        SIDECAR_MANIFEST_PATH,
        roles=("train",),
        expected_manifest_sha256=SIDECAR_MANIFEST_FILE_SHA256,
        contract=FROZEN_BUILD_CONTRACT,
    )["train"]
    if len(sidecar) != 4262 or sidecar_json_sha256(sidecar) != SIDECAR_TRAIN_CONTENT_SHA256:
        raise ValueError("train attitude sidecar changed")
    sidecar_by_global = {int(row["global_row"]): row for row in sidecar}
    if len(sidecar_by_global) != len(sidecar):
        raise ValueError("train sidecar global rows are not injective")
    fit_rows = list(panels["fit"])
    if len(fit_rows) != 160:
        raise ValueError("fit panel must contain exactly 160 transitions")
    identities = []
    globals_seen = []
    for row in fit_rows:
        global_row = int(row["global_row"])
        attitude = sidecar_by_global.get(global_row)
        identity = row_identity_sha256(row)
        expected = {
            "dataset_role": "train",
            "row_identity_sha256": identity,
            "scene_id_sha256": _scene_sha256(str(row["scene_id"])),
            "env_index": int(row["env_index"]),
            "current_frame_index": int(row["current_frame_index"]),
            "next_frame_index": int(row["next_frame_index"]),
            "current_timestamp_ns": int(row["current_timestamp_ns"]),
            "next_timestamp_ns": int(row["next_timestamp_ns"]),
        }
        if attitude is None or any(attitude.get(key) != value for key, value in expected.items()):
            raise ValueError(f"fit attitude join mismatch at global row {global_row}")
        globals_seen.append(global_row)
        identities.append(identity)
    if len(set(globals_seen)) != 160:
        raise ValueError("fit transition rows are not injective")
    records = frame_records(fit_rows)
    for record in records:
        sidecar_row = sidecar_by_global[int(record["global_row"])]
        pose = sidecar_row[str(record["side"])]
        record["base_quat_world_xyzw"] = list(pose["base_quat_world_xyzw"])
        record["stored_base_yaw_rad"] = float(pose["stored_base_yaw_rad"])
        record["row_identity_sha256"] = sidecar_row["row_identity_sha256"]
    if len(records) != 320 or any(
        records[index]["side"] != ("current" if index % 2 == 0 else "next")
        for index in range(320)
    ):
        raise ValueError("fit endpoints are not exact current-then-next pairs")
    audit = {
        "transition_count": 160,
        "frame_count": 320,
        "dataset_role": "train",
        "global_rows_sha256": contract.canonical_json_sha256(globals_seen),
        "row_identities_sha256": contract.canonical_json_sha256(identities),
    }
    return records, audit


def _canonical_records(
    records: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    copied, role_global = attach_role_global_shuffle(
        records, seed=seed, namespace="fit"
    )
    copied, same_scene = attach_same_scene_wrong_view(
        copied, seed=seed, namespace="fit"
    )
    return copied, {
        "role_global_shuffle": role_global,
        "same_scene_wrong_view": same_scene,
        "wrong_rgb_uses_target_attitude": True,
    }


def audit_fit_target_support(
    records: Sequence[Mapping[str, Any]],
    shards: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Prove that the loss never asks the radial head for impossible KNOWN."""

    support = np.asarray(
        build_radial_factorization().representable_mask, dtype=bool
    )
    supervised_cells = 0
    supervised_outside = 0
    known_outside = 0
    for record in records:
        shard = shards[str(record["label_shard_path"])]
        side = str(record["side"])
        row = int(record["label_shard_row"])
        labels = np.asarray(shard[f"{side}_labels"][row], dtype=np.int64)
        mask = np.asarray(shard[f"{side}_supervision_mask"][row], dtype=bool)
        if labels.shape != (64, 64) or mask.shape != (64, 64):
            raise ValueError("fit support audit target shape changed")
        supervised_cells += int(np.count_nonzero(mask))
        supervised_outside += int(np.count_nonzero(mask & ~support))
        known_outside += int(np.count_nonzero(mask & ~support & (labels != 0)))
    report = {
        "frame_count": len(records),
        "representable_cell_count": int(np.count_nonzero(support)),
        "supervised_cell_occurrences": supervised_cells,
        "supervised_outside_support_occurrences": supervised_outside,
        "known_outside_support_occurrences": known_outside,
        "all_supervised_known_cells_representable": known_outside == 0,
    }
    if known_outside:
        raise ValueError(
            "fit labels supervise KNOWN outside categorical radial support"
        )
    return report


@torch.no_grad()
def evaluate_fit(
    model: torch.nn.Module,
    dataset: backend.DynamicPanelDataset,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    controls: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    before = dataset.snapshot()
    aggregate = {
        condition: empty_raw_accumulator() for condition in contract.CONDITIONS
    }
    families = {
        family: {
            condition: empty_raw_accumulator()
            for condition in contract.CONDITIONS
        }
        for family in contract.FAMILIES
    }
    distances = backend._distance_grid()
    model.eval()
    model_calls = 0
    for start in range(0, len(records), contract.BATCH_SIZE):
        indices = tuple(range(start, start + contract.BATCH_SIZE))
        batch = dataset.evaluation_batch(indices)
        images = torch.cat(
            [batch[condition] for condition in contract.CONDITIONS], dim=0
        ).to(device=device, dtype=torch.float32)
        target_quaternion = batch["base_quat_world_xyzw"].to(
            device=device, dtype=torch.float32
        )
        target_yaw = batch["stored_base_yaw_rad"].to(
            device=device, dtype=torch.float32
        )
        quaternions = torch.cat([target_quaternion] * 3, dim=0)
        yaws = torch.cat([target_yaw] * 3, dim=0)
        logits = model(images, quaternions, yaws).float().cpu().numpy()
        if logits.shape != (12, 3, 64, 64):
            raise RuntimeError("dynamic radial evaluation requires batch 12")
        model_calls += 1
        labels = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        for condition, values in zip(
            contract.CONDITIONS, np.split(logits, 3, axis=0), strict=True
        ):
            update_raw_accumulator(
                aggregate[condition], values, labels, mask, distances
            )
            for offset, index in enumerate(indices):
                family = str(records[index]["family"])
                update_raw_accumulator(
                    families[family][condition],
                    values[offset : offset + 1],
                    labels[offset : offset + 1],
                    mask[offset : offset + 1],
                    distances,
                )
    report = {
        "schema": contract.PANEL_REPORT_SCHEMA,
        "panel": "fit",
        "frame_count": 320,
        "target_batch_size": 4,
        "combined_model_batch_size": 12,
        "model_call_dtype": "float32",
        "metric_accumulator_dtype": "float64",
        "wrong_rgb_uses_target_attitude": True,
        "conditions": {
            condition: finalize_raw_accumulator(aggregate[condition])
            for condition in contract.CONDITIONS
        },
        "families": {
            family: {
                "conditions": {
                    condition: finalize_raw_accumulator(
                        families[family][condition]
                    )
                    for condition in contract.CONDITIONS
                }
            }
            for family in contract.FAMILIES
        },
        "controls": dict(controls),
    }
    report["fit_gate"] = contract.fit_panel_gate_report(report)
    contract.validate_panel_report(
        report, seed=controls["role_global_shuffle"]["seed"], panel="fit",
        require_fit_gate=True,
    )
    access = dataset.delta(before)
    access.update(
        {
            "model_calls": model_calls,
            "model_output_frames": model_calls * 12,
            "model_attitude_frames": model_calls * 12,
        }
    )
    return report, backend._normalized_events(access)


def _new_model(device: torch.device) -> DynamicCategoricalRadialPerceptionFullRay:
    with torch.device(device):
        model = DynamicCategoricalRadialPerceptionFullRay()
    # NumPy-backed registered buffers do not honor torch's default-device
    # context, so explicitly move the complete module before first use.
    model = model.to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != REGISTERED_PARAMETER_COUNT:
        raise RuntimeError("dynamic full-ray parameter contract changed")
    return model


def _configure_development_determinism(seed: int) -> dict[str, Any]:
    """Record the one ROCm kernel exception used by this dev-only fit path."""

    record = backend._configure_determinism(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        **record,
        "warn_only": True,
        "known_kernel_exception": "ROCm grid_sampler_2d_backward_cuda",
        "replication_required_before_any_promotion": True,
    }


def _faithful_learning_rate(update: int, total_updates: int) -> float:
    if total_updates < 2 or update < 1 or update > total_updates:
        raise ValueError("faithful cosine update is outside the fixed budget")
    low, high = 1e-5, 2e-4
    return low + 0.5 * (high - low) * (
        1.0 + math.cos(math.pi * (update - 1) / (total_updates - 1))
    )


def _faithful_schedule(seed: int, updates: int) -> list[list[int]]:
    if updates <= 0:
        raise ValueError("faithful updates must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches: list[list[int]] = []
    while len(batches) < updates:
        epoch = torch.randperm(320, generator=generator).tolist()
        batches.extend(
            [epoch[start : start + 80] for start in range(0, 320, 80)]
        )
    return batches[:updates]


def _schedule(seed: int, branch: str, smoke: bool) -> list[list[int]]:
    updates = SMOKE_UPDATES if smoke else (
        FAITHFUL_CONFIG["updates"]
        if branch == "production_faithful"
        else CEILING_CONFIG["updates"]
    )
    if branch == "production_faithful":
        return _faithful_schedule(seed, int(updates))
    if branch != "ceiling_optimizer":
        raise ValueError(f"unknown optimizer branch: {branch}")
    return contract.deterministic_minibatch_schedule(
        seed=seed, branch=branch, updates=int(updates)
    )


def _run_stage(
    *,
    branch: str,
    smoke: bool,
    initial_state: Mapping[str, torch.Tensor],
    initial_state_sha256: str,
    dataset: backend.DynamicPanelDataset,
    records: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], torch.nn.Module]:
    config = dict(
        FAITHFUL_CONFIG if branch == "production_faithful" else CEILING_CONFIG
    )
    updates = SMOKE_UPDATES if smoke else int(config["updates"])
    batch_size = int(config["batch_size"])
    interval = SMOKE_EVALUATION_INTERVAL if smoke else contract.EVALUATION_INTERVAL
    schedule = _schedule(seed, branch, smoke)
    if not smoke and branch == "ceiling_optimizer":
        contract.validate_minibatch_schedule(schedule, seed=seed, branch=branch)
    model = _new_model(device)
    model.load_state_dict(initial_state, strict=True)
    if backend._state_dict_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("dynamic radial branch did not restart initial state")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            config.get("learning_rate", config.get("learning_rate_start"))
        ),
        weight_decay=float(config["weight_decay"]),
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )
    before = dataset.snapshot()
    evaluation_access: Counter[str] = Counter()
    curve = []
    for step, indices in enumerate(schedule, start=1):
        batch = dataset.training_batch(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["image"].to(device=device, dtype=torch.float32),
            batch["base_quat_world_xyzw"].to(device=device, dtype=torch.float32),
            batch["stored_base_yaw_rad"].to(device=device, dtype=torch.float32),
        )
        loss = backend.direct_hierarchical_loss(
            logits,
            batch["labels"].to(device),
            batch["mask"].to(device),
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite {branch} loss at {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), contract.GRADIENT_CLIP
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).item()):
            raise FloatingPointError(
                f"non-finite {branch} gradient norm at {step}"
            )
        learning_rate = (
            _faithful_learning_rate(step, updates)
            if branch == "production_faithful"
            else float(config["learning_rate"])
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.step()
        if step % interval == 0:
            fit, access = evaluate_fit(
                model, dataset, records, device=device, controls=controls
            )
            evaluation_access.update(access)
            curve.append(
                {
                    "step": step,
                    "learning_rate": learning_rate,
                    "batch_loss": float(loss.detach().item()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "fit_panel": fit,
                }
            )
    terminal = contract.terminal_fit_gate_summary(curve, updates, interval)
    total = backend._normalized_events(dataset.delta(before))
    evaluation = backend._normalized_events(evaluation_access)
    training = {
        field: total[field] - evaluation[field] for field in EVENT_FIELDS
    }
    training.update(
        {
            "model_calls": updates,
            "model_output_frames": updates * batch_size,
            "model_attitude_frames": updates * batch_size,
        }
    )
    return {
        "schema": STAGE_SCHEMA,
        "branch": branch,
        "config": config,
        "completed_steps": updates,
        "evaluation_interval": interval,
        "batch_size": batch_size,
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
            "weight_decay": float(config["weight_decay"]),
            "gradient_clip": 1.0,
            "learning_rate_schedule": str(config["schedule"]),
        },
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": backend._state_dict_sha256(model.state_dict()),
        "exact_initial_state_restart_verified": True,
        "minibatch_indices": schedule,
        "minibatch_indices_sha256": contract.canonical_json_sha256(schedule),
        "learning_curve": curve,
        "terminal_fit_gate": terminal,
        "training_access": training,
        "fit_evaluation_access": evaluation,
    }, model


def _canonical_output(seed: int, smoke: bool) -> Path:
    name = f"smoke_seed_{seed}_result.json" if smoke else f"seed_{seed}_result.json"
    return Path(os.path.abspath(CANONICAL_DIRECTORY / name))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.output = backend._canonical_path(args.output, name="output path")
        contract.validate_seed(args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output != _canonical_output(args.seed, args.smoke):
        parser.error("development fit output path is not canonical")
    if args.output.exists():
        parser.error("development fit result already exists")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    invocation = list(sys.argv) if argv is None else [str(Path(__file__).resolve()), *map(str, argv)]
    started = datetime.now(timezone.utc).isoformat()
    sources_start = _source_hashes()
    device, device_record = backend._validate_resource_environment(args.device)
    determinism = _configure_development_determinism(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    initial_model = _new_model(device)
    initial_state = backend._clone_state(initial_model.state_dict())
    initial_state_sha = backend._state_dict_sha256(initial_state)
    parameter_count = sum(parameter.numel() for parameter in initial_model.parameters())
    del initial_model
    torch.cuda.empty_cache()

    joined, join_audit = _load_fit_records()
    records, controls = _canonical_records(joined, seed=args.seed)
    images, shards = backend._artifact_contract(records, "fit")
    backend._verify_artifacts(images, shards)
    dataset = backend.DynamicPanelDataset(records, "fit")
    dataset.preload()
    target_support_audit = audit_fit_target_support(records, dataset._shards)
    faithful, faithful_model = _run_stage(
        branch="production_faithful", smoke=args.smoke,
        initial_state=initial_state, initial_state_sha256=initial_state_sha,
        dataset=dataset, records=records, controls=controls, device=device,
        seed=args.seed,
    )
    ceiling = None
    if faithful["terminal_fit_gate"]["passes"]:
        del faithful_model
    else:
        del faithful_model
        torch.cuda.empty_cache()
        ceiling, ceiling_model = _run_stage(
            branch="ceiling_optimizer", smoke=args.smoke,
            initial_state=initial_state, initial_state_sha256=initial_state_sha,
            dataset=dataset, records=records, controls=controls, device=device,
            seed=args.seed,
        )
        del ceiling_model
    backend._verify_artifacts(images, shards)
    sources_end = _source_hashes()
    if sources_end != sources_start:
        raise RuntimeError("development fit sources changed during execution")
    stages = {"production_faithful": faithful, "ceiling_optimizer": ceiling}
    qualifying = (
        "production_faithful"
        if faithful["terminal_fit_gate"]["passes"]
        else "ceiling_optimizer"
        if ceiling is not None and ceiling["terminal_fit_gate"]["passes"]
        else None
    )
    core = {
        "schema": RESULT_SCHEMA,
        "authoritative": False,
        "development_only": True,
        "fit_only": True,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "created_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation,
        "execution": {
            "device": device_record,
            "determinism": determinism,
            "training_branches": {
                "production_faithful": FAITHFUL_CONFIG,
                "ceiling_optimizer": CEILING_CONFIG,
            },
            "combined_evaluation_batch_size": 12,
            "source_workers": SOURCE_WORKERS,
        },
        "inputs": {
            "panel": {
                "path": str(PANEL_PATH), "sha256": PANEL_FILE_SHA256,
                "content_sha256": PANEL_CONTENT_SHA256,
                "fit_rows_sha256": FIT_ROWS_SHA256,
            },
            "attitude_sidecar": {
                "manifest_path": str(SIDECAR_MANIFEST_PATH),
                "manifest_sha256": SIDECAR_MANIFEST_FILE_SHA256,
                "opened_roles": ["train"],
            },
        },
        "source_hashes": sources_end,
        "model": {
            "class": "DynamicCategoricalRadialPerceptionFullRay",
            "parameter_count": parameter_count,
            "initial_state_sha256": initial_state_sha,
            "attitude_required": True,
        },
        "objective": contract.OBJECTIVE_CONTRACT,
        "preprocessing": contract.PREPROCESSING_CONTRACT,
        "controls": contract.CONTROL_CONTRACT,
        "fit_join": join_audit,
        "fit_target_support_audit": target_support_audit,
        "stages": stages,
        "qualifying_branch": qualifying,
        "fit_gate_passes": qualifying is not None,
        "access_ledger": {
            "fit_image_hash_byte_opens": 640,
            "fit_label_shard_hash_byte_opens": 40,
            "fit_image_decode_byte_opens": 320,
            "fit_label_shard_decode_byte_opens": 20,
            "train_sidecar_role_byte_opens": 1,
            "checkpoint_selection_sidecar_role_byte_opens": 0,
            "probability_calibration_sidecar_role_byte_opens": 0,
            "g2_sidecar_role_byte_opens": 0,
            "non_fit_image_payload_byte_opens": 0,
            "non_fit_label_payload_byte_opens": 0,
            "non_fit_model_output_frames": 0,
            "g2_payload_byte_opens": 0,
            "g2_model_output_frames": 0,
        },
        "holdouts": None,
        "g2": None,
        "licenses": {
            "heldout_claim": False,
            "g2": False,
            "shared_jepa": False,
            "runtime": False,
        },
    }
    payload = {**core, "content_sha256": contract.canonical_json_sha256(core)}
    backend._publish_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_sha256": payload["content_sha256"],
                "fit_gate_passes": payload["fit_gate_passes"],
                "qualifying_branch": qualifying,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
