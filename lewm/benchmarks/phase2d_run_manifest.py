"""Frozen Phase 2D training-run manifest construction."""
from __future__ import annotations

import math
import json
import shlex
from pathlib import Path
from typing import Mapping, Sequence

from .experiment_manifest import build_experiment_manifest, write_json
from .phase2d_training import ACTION_UTILITY_TARGET_VERSION, CONSEQUENCE_TARGET_DIM
from .phase2d_readiness import phase2d_training_start_readiness

PRIMARY_CELLS = ("C0", "C1", "C2")
REGISTERED_OPTIMIZATION_SEEDS = (20260614, 20260615, 20260616)
REGISTERED_CHECKPOINT_RULE = "registered_phase2d_validation_v0"
DEFAULT_SOURCE_STATES_PER_BATCH = 2
DEFAULT_EPOCHS = 3


def phase2d_epoch_schedule(
    *,
    train_source_states: int,
    source_states_per_batch: int = DEFAULT_SOURCE_STATES_PER_BATCH,
    epochs: int = DEFAULT_EPOCHS,
) -> dict:
    """Return the registered source-grouped epoch schedule."""

    if train_source_states < 1:
        raise ValueError("train_source_states must be positive")
    if source_states_per_batch < 1:
        raise ValueError("source_states_per_batch must be positive")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    steps_per_epoch = math.ceil(train_source_states / source_states_per_batch)
    return {
        "train_source_states": int(train_source_states),
        "source_states_per_batch": int(source_states_per_batch),
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "optimization_steps": int(steps_per_epoch * epochs),
        "evaluation_interval": int(steps_per_epoch),
    }


def _train_source_states_from_split_manifest(split_manifest: Mapping) -> int:
    try:
        return int(
            split_manifest["splits"]["train"]["dataset_audit"]["source_states"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "split manifest must contain splits.train.dataset_audit.source_states"
        ) from error


def build_phase2d_training_run_manifest(
    *,
    repository_root: Path,
    split_manifest_path: Path,
    split_manifest: Mapping,
    train_data_path: Path,
    validation_data_path: Path,
    output_checkpoint_path: Path,
    cell: str,
    seed: int,
    python_executable: str,
    device: str,
    schedule: Mapping,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 0.0,
    detach_action_control_state: bool = False,
    target_geometry: str = "patch",
    num_target_slots: int = 16,
    consequence_loss_lambda: float = 0.0,
    action_utility_loss_lambda: float = 0.0,
    action_utility_regression_weight: float = 0.1,
) -> dict:
    """Build one immutable Phase 2D training launch manifest."""

    if cell not in PRIMARY_CELLS:
        raise ValueError(f"cell must be one of {PRIMARY_CELLS}, got {cell}")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if max_grad_norm < 0.0:
        raise ValueError("max_grad_norm must be non-negative")
    if target_geometry not in ("patch", "slot"):
        raise ValueError("target_geometry must be 'patch' or 'slot'")
    if num_target_slots < 1:
        raise ValueError("num_target_slots must be positive")
    if consequence_loss_lambda < 0.0:
        raise ValueError("consequence_loss_lambda must be non-negative")
    if action_utility_loss_lambda < 0.0:
        raise ValueError("action_utility_loss_lambda must be non-negative")
    if action_utility_regression_weight < 0.0:
        raise ValueError("action_utility_regression_weight must be non-negative")
    readiness = phase2d_training_start_readiness(
        split_manifest_path=split_manifest_path,
        cell=cell,
        requested_run_class="confirmatory",
        train_data_path=train_data_path,
        validation_data_path=validation_data_path,
    )
    if not readiness["passed"]:
        raise ValueError(f"training-start readiness failed for {cell}: {readiness}")

    command_parts = [
        python_executable,
        "scripts/train_jepa_phase2d.py",
        "--train-data",
        str(train_data_path),
        "--validation-data",
        str(validation_data_path),
        "--output",
        str(output_checkpoint_path),
        "--cell",
        cell,
        "--run-class",
        "confirmatory",
        "--split-manifest",
        str(split_manifest_path),
        "--optimization-steps",
        str(int(schedule["optimization_steps"])),
        "--evaluation-interval",
        str(int(schedule["evaluation_interval"])),
        "--source-states-per-batch",
        str(int(schedule["source_states_per_batch"])),
        "--lr",
        f"{float(learning_rate):g}",
        "--weight-decay",
        f"{float(weight_decay):g}",
        "--seed",
        str(int(seed)),
        "--device",
        device,
    ]
    if max_grad_norm > 0.0:
        command_parts.extend(["--max-grad-norm", f"{float(max_grad_norm):g}"])
    if detach_action_control_state:
        command_parts.append("--detach-action-control-state")
    if target_geometry != "patch":
        command_parts.extend(
            [
                "--target-geometry",
                target_geometry,
                "--num-target-slots",
                str(int(num_target_slots)),
            ]
        )
    if consequence_loss_lambda > 0.0:
        command_parts.extend(
            [
                "--consequence-loss-lambda",
                f"{float(consequence_loss_lambda):g}",
            ]
        )
    if action_utility_loss_lambda > 0.0:
        command_parts.extend(
            [
                "--action-utility-loss-lambda",
                f"{float(action_utility_loss_lambda):g}",
                "--action-utility-regression-weight",
                f"{float(action_utility_regression_weight):g}",
            ]
        )
    run_command = " ".join(shlex.quote(part) for part in command_parts)
    config = {
        "phase": "Phase 2D",
        "run_class": "confirmatory",
        "cell": cell,
        "seed": int(seed),
        "checkpoint_rule": REGISTERED_CHECKPOINT_RULE,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "max_grad_norm": float(max_grad_norm),
        },
        "model_amendments": {
            "detach_action_control_state": bool(detach_action_control_state),
            "target_geometry": target_geometry,
            "num_target_slots": int(num_target_slots),
            "consequence_dim": (
                CONSEQUENCE_TARGET_DIM
                if consequence_loss_lambda > 0.0
                else 0
            ),
            "consequence_loss_lambda": float(consequence_loss_lambda),
            "action_utility_loss_lambda": float(action_utility_loss_lambda),
            "action_utility_regression_weight": float(
                action_utility_regression_weight
            ),
            "action_utility_target_version": (
                ACTION_UTILITY_TARGET_VERSION
                if action_utility_loss_lambda > 0.0
                else None
            ),
        },
        "schedule": dict(schedule),
        "device": device,
        "access_scope": (
            "train_and_validation_only; no test_id_or_test_hard_result_access"
        ),
        "split_manifest_schema": split_manifest.get("schema"),
        "expected_checkpoint_path": str(output_checkpoint_path),
    }
    return build_experiment_manifest(
        experiment_id=f"phase2d_{cell}_seed_{int(seed)}_confirmatory_train",
        repository_root=repository_root,
        inputs={
            "split_manifest": split_manifest_path,
            "train_data": train_data_path,
            "validation_data": validation_data_path,
            "trainer_script": repository_root / "scripts/train_jepa_phase2d.py",
            "phase2_data_module": repository_root / "lewm/benchmarks/phase2_data.py",
            "phase2d_training_module": (
                repository_root / "lewm/benchmarks/phase2d_training.py"
            ),
            "rollout_diagnostics_module": (
                repository_root / "lewm/benchmarks/rollout_diagnostics.py"
            ),
            "phase2d_model_module": (
                repository_root / "lewm/models/phase2d_spatial_lewm.py"
            ),
            "spatial_predictor_module": (
                repository_root / "lewm/models/spatial_predictor.py"
            ),
        },
        artifacts={},
        config=config,
        seeds=[int(seed)],
        run_command=run_command,
    )


def create_phase2d_training_run_manifests(
    *,
    repository_root: Path,
    split_manifest_path: Path,
    train_data_path: Path,
    validation_data_path: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    cells: Sequence[str] = PRIMARY_CELLS,
    seeds: Sequence[int] = REGISTERED_OPTIMIZATION_SEEDS,
    python_executable: str = ".generated/venvs/genesis_render_vulkan/bin/python",
    device: str = "auto",
    source_states_per_batch: int = DEFAULT_SOURCE_STATES_PER_BATCH,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 0.0,
    detach_action_control_state: bool = False,
    target_geometry: str = "patch",
    num_target_slots: int = 16,
    consequence_loss_lambda: float = 0.0,
    action_utility_loss_lambda: float = 0.0,
    action_utility_regression_weight: float = 0.1,
) -> dict:
    """Write the full registered Phase 2D primary-cell training manifest matrix."""

    split_manifest = json.loads(split_manifest_path.read_text())
    schedule = phase2d_epoch_schedule(
        train_source_states=_train_source_states_from_split_manifest(split_manifest),
        source_states_per_batch=source_states_per_batch,
        epochs=epochs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = {}
    for cell in cells:
        if cell not in PRIMARY_CELLS:
            raise ValueError(f"unsupported primary Phase 2D cell: {cell}")
        for seed in seeds:
            checkpoint = checkpoint_dir / f"{cell}_seed_{int(seed)}.pt"
            manifest = build_phase2d_training_run_manifest(
                repository_root=repository_root,
                split_manifest_path=split_manifest_path,
                split_manifest=split_manifest,
                train_data_path=train_data_path,
                validation_data_path=validation_data_path,
                output_checkpoint_path=checkpoint,
                cell=cell,
                seed=int(seed),
                python_executable=python_executable,
                device=device,
                schedule=schedule,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                detach_action_control_state=detach_action_control_state,
                target_geometry=target_geometry,
                num_target_slots=num_target_slots,
                consequence_loss_lambda=consequence_loss_lambda,
                action_utility_loss_lambda=action_utility_loss_lambda,
                action_utility_regression_weight=action_utility_regression_weight,
            )
            manifest_path = output_dir / f"{cell}_seed_{int(seed)}_manifest.json"
            write_json(manifest_path, manifest)
            manifest_records[f"{cell}_seed_{int(seed)}"] = {
                "manifest_path": str(manifest_path.resolve()),
                "checkpoint_path": str(checkpoint.resolve()),
                "run_command": manifest["run_command"],
            }
    summary = {
        "schema": "jepa_phase2d_training_run_manifest_matrix_v0",
        "cells": list(cells),
        "seeds": [int(seed) for seed in seeds],
        "checkpoint_rule": REGISTERED_CHECKPOINT_RULE,
        "schedule": schedule,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "max_grad_norm": float(max_grad_norm),
        },
        "model_amendments": {
            "detach_action_control_state": bool(detach_action_control_state),
            "target_geometry": target_geometry,
            "num_target_slots": int(num_target_slots),
            "consequence_dim": (
                CONSEQUENCE_TARGET_DIM
                if consequence_loss_lambda > 0.0
                else 0
            ),
            "consequence_loss_lambda": float(consequence_loss_lambda),
            "action_utility_loss_lambda": float(action_utility_loss_lambda),
            "action_utility_regression_weight": float(
                action_utility_regression_weight
            ),
            "action_utility_target_version": (
                ACTION_UTILITY_TARGET_VERSION
                if action_utility_loss_lambda > 0.0
                else None
            ),
        },
        "manifests": manifest_records,
        "access_scope": (
            "train_and_validation_only; no test_id_or_test_hard_result_access"
        ),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
