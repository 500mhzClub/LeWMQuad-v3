#!/usr/bin/env python3
"""Run the frozen categorical-radial N32 train-role diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "lewm_worlds"))

from lewm.benchmarks.go2_categorical_radial_n32 import (  # noqa: E402
    CONDITIONS,
    EXECUTION_BINDING_SHA256,
    FAMILIES,
    HOLDOUT_PANELS,
    RESULT_SCHEMA,
    SMOKE_RESULT_SCHEMA,
    categorical_holdout_checks,
    extract_faithful_patch7_family_reference,
    fit_panel_gate_report,
    per_seed_decision,
    terminal_fit_gate_summary,
)
from lewm.benchmarks.go2_physical_micro_overfit import (  # noqa: E402
    attach_role_global_shuffle,
    attach_same_scene_wrong_view,
    empty_raw_accumulator,
    finalize_raw_accumulator,
    frame_records,
    update_raw_accumulator,
    validate_panel_manifest,
)
from lewm.models.categorical_radial_perception import IMAGE_SIZE  # noqa: E402
from lewm.models.categorical_radial_perception_full_ray import (  # noqa: E402
    CategoricalRadialPerceptionFullRay,
    REGISTERED_PARAMETER_COUNT,
)
from scripts import run_go2_categorical_radial_ladder_v3 as v3  # noqa: E402


PANEL_PATH = (
    REPOSITORY_ROOT / ".generated/go2_physical_micro_overfit/patch7_v1/panel.json"
)
PANEL_FILE_SHA256 = (
    "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c"
)
PANEL_CONTENT_SHA256 = (
    "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f"
)
PANEL_ROWS_SHA256 = {
    "fit": "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d",
    "same_scene_holdout": (
        "d32713086c042d20f94825aa362c27a07bef6fd0e0cce0aa5846bb67bf8dc465"
    ),
    "cross_scene_holdout": (
        "3565f7f7844f3aeee28b0433aa6dc77d553a9ebb831cf9af20b6d392c5416817"
    ),
}
LADDER_PATH = v3.FROZEN_LADDER_MANIFEST_PATH
LADDER_FILE_SHA256 = v3.EXPECTED_LADDER_FILE_SHA256
LADDER_CONTENT_SHA256 = (
    "00a3ad1263af16e3b858f7e7522df7b108a49301d25fa805148e82b36cb52f8e"
)
V3_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_micro_overfit/v3/"
    "seed_20260710_ladder_result.json"
)
V3_RESULT_FILE_SHA256 = (
    "7a5f67bacb2e3df67421bcff13b15d1fa3e00d99f3b2af52c52b0b6ce14617a8"
)
V3_RESULT_CONTENT_SHA256 = (
    "517313139077027176c471f829f57148684d3df0def6096ce7702d3bbba46ce1"
)
PATCH7_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_physical_micro_overfit/patch7_v1/"
    "seed_20260710_result.json"
)
PATCH7_RESULT_FILE_SHA256 = (
    "6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c"
)
PATCH7_RESULT_CONTENT_SHA256 = (
    "32d848d3df68e670ddb4cc24436981f62a1aa5562b89e6d6719ecb113f66b749"
)
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_microfit_protocol_2026-07-10.md"
)
PROTOCOL_SHA256 = (
    "ef23ee607d0976d67adf33591f5af78652da4305811a563d94bd8539abc9d404"
)
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_n32_execution_binding_2026-07-10.md"
)
EXPECTED_SEED10_INITIAL_STATE_SHA256 = (
    "8b149b57ae4bb305a2306a4dde2cab5f57a46f1c3760837593ed4d9862491278"
)
PATCH7_RESULT_SCHEMA = "lewm_go2_physical_micro_overfit_result_v1"
EXPECTED_PANEL_ARTIFACT_COUNTS = {
    "fit": {"images": 320, "shards": 20},
    "same_scene_holdout": {"images": 320, "shards": 20},
    "cross_scene_holdout": {"images": 320, "shards": 25},
}
AUTHORITATIVE_BRANCHES = {
    "production_faithful": {
        "updates": 2000,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
    },
    "ceiling_optimizer": {
        "updates": 5000,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
    },
}
SMOKE_BRANCHES = {
    name: {**config, "updates": 3}
    for name, config in AUTHORITATIVE_BRANCHES.items()
}
BATCH_SIZE = 4
AUTHORITATIVE_EVALUATION_INTERVAL = 100
SMOKE_EVALUATION_INTERVAL = 1
GRADIENT_CLIP = 1.0
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
direct_hierarchical_loss = v3.v2.v1.hierarchical_occupancy_loss


def _sha256_file(path: Path) -> str:
    return v3.v2.v1._sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    return v3.v2.v1._read_json(path)


def _source_paths() -> dict[str, Path]:
    shared = {}
    for name, path in v3._source_paths().items():
        shared["v3_runner" if name == "runner" else name] = path
    return {
        **shared,
        "n32_contract": CONTRACT_PATH,
        "n32_pure": (
            REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32.py"
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
    declared = str(core.pop("content_sha256", ""))
    if (
        payload.get("schema") != schema
        or declared != content_sha256
        or v3.canonical_json_sha256(core) != declared
    ):
        raise ValueError(f"{name} content contract mismatch")


def _load_bound_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    expected = {
        PANEL_PATH: PANEL_FILE_SHA256,
        LADDER_PATH: LADDER_FILE_SHA256,
        V3_RESULT_PATH: V3_RESULT_FILE_SHA256,
        PATCH7_RESULT_PATH: PATCH7_RESULT_FILE_SHA256,
        PROTOCOL_PATH: PROTOCOL_SHA256,
        CONTRACT_PATH: EXECUTION_BINDING_SHA256,
    }
    for path, digest in expected.items():
        if _sha256_file(path) != digest:
            raise ValueError(f"bound N32 evidence SHA-256 mismatch: {path}")
    panel = _read_json(PANEL_PATH)
    ladder = _read_json(LADDER_PATH)
    v3_result = _read_json(V3_RESULT_PATH)
    patch7 = _read_json(PATCH7_RESULT_PATH)
    if str(panel.get("content_sha256", "")) != PANEL_CONTENT_SHA256:
        raise ValueError("bound panel content SHA-256 mismatch")
    panels = validate_panel_manifest(panel)
    for name in ("fit", *HOLDOUT_PANELS):
        if str(panel["panels"][name].get("rows_sha256", "")) != (
            PANEL_ROWS_SHA256[name]
        ):
            raise ValueError(f"bound panel rows changed: {name}")
    if str(ladder.get("content_sha256", "")) != LADDER_CONTENT_SHA256:
        raise ValueError("bound ladder content SHA-256 mismatch")
    v3.v2.v1.validate_ladder_manifest(ladder)
    _validate_content(
        v3_result,
        schema=v3.RESULT_SCHEMA,
        content_sha256=V3_RESULT_CONTENT_SHA256,
        name="V3 result",
    )
    if (
        v3_result.get("source_hashes") != v3._source_hashes()
        or v3_result.get("decision", {}).get(
            "n32_diagnostic_construction_licensed"
        )
        is not True
        or str(v3_result.get("model", {}).get("initial_state_sha256", ""))
        != EXPECTED_SEED10_INITIAL_STATE_SHA256
    ):
        raise ValueError("bound V3 result provenance or decision mismatch")
    _validate_content(
        patch7,
        schema=PATCH7_RESULT_SCHEMA,
        content_sha256=PATCH7_RESULT_CONTENT_SHA256,
        name="patch7 result",
    )
    reference = extract_faithful_patch7_family_reference(patch7)
    return panel, panels, ladder, v3_result, reference


def _artifact_contract(
    records: Sequence[Mapping[str, Any]],
    panel: str,
) -> tuple[dict[str, str], dict[str, str]]:
    images: dict[str, str] = {}
    shards: dict[str, str] = {}
    for record in records:
        for collection, path_key, sha_key in (
            (images, "image_path", "image_sha256"),
            (shards, "label_shard_path", "label_shard_sha256"),
        ):
            path = str(Path(str(record[path_key])).resolve())
            digest = str(record[sha_key])
            previous = collection.setdefault(path, digest)
            if previous != digest:
                raise ValueError(f"conflicting {panel} artifact digest: {path}")
    expected = EXPECTED_PANEL_ARTIFACT_COUNTS[panel]
    if len(images) != expected["images"] or len(shards) != expected["shards"]:
        raise ValueError(f"{panel} artifact counts differ from the frozen panel")
    return images, shards


def _verify_artifacts(images: Mapping[str, str], shards: Mapping[str, str]) -> None:
    for path, digest in (*sorted(images.items()), *sorted(shards.items())):
        if _sha256_file(Path(path)) != digest:
            raise ValueError(f"authorized train artifact SHA-256 mismatch: {path}")


class PanelFrameDataset:
    """Cached, panel-authorized RGB/label reader with explicit access events."""

    def __init__(self, records: Sequence[Mapping[str, Any]], panel: str) -> None:
        self.records = [dict(record) for record in records]
        self.panel = str(panel)
        self._images: dict[str, torch.Tensor] = {}
        self._targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._shards: dict[str, dict[str, np.ndarray]] = {}
        self.events: Counter[str] = Counter()

    def snapshot(self) -> dict[str, int]:
        return dict(self.events)

    def delta(self, before: Mapping[str, int]) -> dict[str, int]:
        keys = set(before) | set(self.events)
        return {
            key: int(self.events[key]) - int(before.get(key, 0))
            for key in sorted(keys)
        }

    def _image(self, path: str) -> torch.Tensor:
        self.events["image_requests"] += 1
        if path in self._images:
            return self._images[path]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = tensor.new_tensor(NORMALIZATION_MEAN)[:, None, None]
        std = tensor.new_tensor(NORMALIZATION_STD)[:, None, None]
        self._images[path] = (tensor - mean) / std
        self.events["image_decode_events"] += 1
        return self._images[path]

    def _target(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.events["target_requests"] += 1
        if index in self._targets:
            return self._targets[index]
        record = self.records[index]
        path = str(record["label_shard_path"])
        if path not in self._shards:
            with np.load(path, allow_pickle=False) as archive:
                self._shards[path] = {
                    name: np.asarray(archive[name]) for name in archive.files
                }
            self.events["label_shard_npz_open_events"] += 1
        shard = self._shards[path]
        side = str(record["side"])
        row = int(record["label_shard_row"])
        labels = np.asarray(shard[f"{side}_labels"][row], dtype=np.int64)
        mask = np.asarray(shard[f"{side}_supervision_mask"][row], dtype=bool)
        if labels.shape != (64, 64) or mask.shape != labels.shape:
            raise ValueError("N32 label grid shape changed")
        self._targets[index] = (
            torch.from_numpy(labels.copy()).long(),
            torch.from_numpy(mask.copy()).bool(),
        )
        return self._targets[index]

    def training_batch(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        images, labels, masks = [], [], []
        for raw_index in indices:
            index = int(raw_index)
            target, mask = self._target(index)
            images.append(self._image(str(self.records[index]["image_path"])))
            labels.append(target)
            masks.append(mask)
        return {
            "image": torch.stack(images),
            "labels": torch.stack(labels),
            "mask": torch.stack(masks),
        }

    def evaluation_batch(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        correct, role_global, same_scene, labels, masks = [], [], [], [], []
        for raw_index in indices:
            index = int(raw_index)
            record = self.records[index]
            target, mask = self._target(index)
            correct.append(self._image(str(record["image_path"])))
            role_global.append(self._image(str(record["control_image_path"])))
            same_scene.append(
                self._image(str(record["same_scene_control_image_path"]))
            )
            labels.append(target)
            masks.append(mask)
        return {
            "correct_rgb": torch.stack(correct),
            "role_global_shuffled_rgb": torch.stack(role_global),
            "same_scene_wrong_view_rgb": torch.stack(same_scene),
            "labels": torch.stack(labels),
            "mask": torch.stack(masks),
        }


def _distance_grid() -> np.ndarray:
    forward = np.linspace(-0.95, 5.35, 64, dtype=np.float64)
    left = np.linspace(-3.15, 3.15, 64, dtype=np.float64)
    return np.sqrt(forward[:, None] ** 2 + left[None, :] ** 2)


def _canonical_panel_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    panel: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = frame_records(rows)
    if len(records) != 320:
        raise ValueError(f"{panel} must expand to exactly 320 endpoint frames")
    if len({str(record["image_sha256"]) for record in records}) != 320:
        raise ValueError(f"{panel} endpoint images are not unique")
    records, role_global = attach_role_global_shuffle(
        records,
        seed=seed,
        namespace=panel,
    )
    records, same_scene = attach_same_scene_wrong_view(
        records,
        seed=seed,
        namespace=panel,
    )
    return records, {
        "role_global_shuffle": role_global,
        "same_scene_wrong_view": same_scene,
    }


def frozen_minibatch_schedule(
    frame_count: int,
    updates: int,
    seed: int,
) -> list[list[int]]:
    if frame_count <= 0 or frame_count % BATCH_SIZE:
        raise ValueError("N32 frame count must be positive and divisible by four")
    generator = torch.Generator().manual_seed(int(seed))
    remaining: list[int] = []
    batches = []
    for _ in range(int(updates)):
        indices, remaining = v3.v2.v1._next_batch_indices(
            frame_count=frame_count,
            batch_size=BATCH_SIZE,
            generator=generator,
            order=remaining,
        )
        batches.append(list(indices))
    return batches


@torch.no_grad()
def evaluate_panel(
    model: torch.nn.Module,
    dataset: PanelFrameDataset,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    panel: str,
    controls: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    before = dataset.snapshot()
    aggregate = {condition: empty_raw_accumulator() for condition in CONDITIONS}
    by_family = {
        family: {condition: empty_raw_accumulator() for condition in CONDITIONS}
        for family in FAMILIES
    }
    distances = _distance_grid()
    model.eval()
    model_calls = 0
    model_output_frames = 0
    for start in range(0, len(records), BATCH_SIZE):
        indices = tuple(range(start, start + BATCH_SIZE))
        batch = dataset.evaluation_batch(indices)
        images = torch.cat(
            [batch[condition] for condition in CONDITIONS],
            dim=0,
        ).to(device=device, dtype=torch.float32)
        logits = model(images).float().cpu().numpy()
        model_calls += 1
        model_output_frames += int(logits.shape[0])
        split = np.split(logits, len(CONDITIONS), axis=0)
        labels = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        for condition, values in zip(CONDITIONS, split):
            update_raw_accumulator(
                aggregate[condition], values, labels, mask, distances
            )
            for offset, target_index in enumerate(indices):
                family = str(records[target_index]["family"])
                update_raw_accumulator(
                    by_family[family][condition],
                    values[offset : offset + 1],
                    labels[offset : offset + 1],
                    mask[offset : offset + 1],
                    distances,
                )
    report = {
        "schema": "lewm_go2_categorical_radial_n32_panel_report_v1",
        "panel": panel,
        "frame_count": len(records),
        "target_batch_size": BATCH_SIZE,
        "combined_model_batch_size": BATCH_SIZE * len(CONDITIONS),
        "model_call_dtype": "float32",
        "metric_accumulator_dtype": "float64",
        "conditions": {
            condition: finalize_raw_accumulator(aggregate[condition])
            for condition in CONDITIONS
        },
        "families": {
            family: {
                "conditions": {
                    condition: finalize_raw_accumulator(
                        by_family[family][condition]
                    )
                    for condition in CONDITIONS
                }
            }
            for family in FAMILIES
        },
        "controls": dict(controls),
    }
    if panel == "fit":
        report["fit_gate"] = fit_panel_gate_report(report)
    access = dataset.delta(before)
    access.update(
        {
            "model_calls": model_calls,
            "model_output_frames": model_output_frames,
        }
    )
    return report, access


def _run_stage(
    *,
    stage_name: str,
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
    if v3.v2.v1._state_dict_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("N32 branch did not restart from the initial state")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    updates = int(config["updates"])
    schedule = frozen_minibatch_schedule(len(records), updates, seed)
    before_training = dataset.snapshot()
    curve = []
    evaluation_access = Counter()
    for step, indices in enumerate(schedule, start=1):
        batch = dataset.training_batch(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["image"].to(device=device, dtype=torch.float32))
        loss = direct_hierarchical_loss(
            logits,
            batch["labels"].to(device),
            batch["mask"].to(device),
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite N32 loss at update {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP
        )
        optimizer.step()
        if step % evaluation_interval == 0:
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
                    "batch_loss": float(loss.detach().item()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "fit_panel": fit_report,
                }
            )
    terminal = terminal_fit_gate_summary(curve, updates, evaluation_interval)
    total_delta = dataset.delta(before_training)
    training_access = {
        key: int(total_delta.get(key, 0)) - int(evaluation_access.get(key, 0))
        for key in total_delta
    }
    training_access["model_calls"] = updates
    training_access["model_output_frames"] = updates * BATCH_SIZE
    result = {
        "schema": "lewm_go2_categorical_radial_n32_stage_v1",
        "stage": stage_name,
        "maximum_steps": updates,
        "completed_steps": updates,
        "batch_size": BATCH_SIZE,
        "evaluation_interval": evaluation_interval,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
            "gradient_clip": GRADIENT_CLIP,
            "constant_learning_rate": True,
        },
        "fixed_update_budget_consumed": True,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": v3.v2.v1._state_dict_sha256(model.state_dict()),
        "minibatch_indices": schedule,
        "minibatch_indices_sha256": v3.canonical_json_sha256(schedule),
        "learning_curve": curve,
        "terminal_fit_gate": terminal,
        "training_access": training_access,
        "fit_evaluation_access": dict(sorted(evaluation_access.items())),
        "holdouts_evaluated": False,
    }
    return result, model


def _canonical_output(seed: int) -> Path:
    return (
        REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_n32/v1/"
        f"seed_{int(seed)}_result.json"
    ).resolve()


def _validate_stored_stage(
    stage: Mapping[str, Any],
    *,
    stage_name: str,
    seed: int,
) -> dict[str, Any]:
    config = AUTHORITATIVE_BRANCHES[stage_name]
    updates = int(config["updates"])
    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "amsgrad": False,
        "gradient_clip": GRADIENT_CLIP,
        "constant_learning_rate": True,
    }
    if (
        stage.get("stage") != stage_name
        or int(stage.get("maximum_steps", -1)) != updates
        or int(stage.get("completed_steps", -1)) != updates
        or int(stage.get("batch_size", -1)) != BATCH_SIZE
        or int(stage.get("evaluation_interval", -1))
        != AUTHORITATIVE_EVALUATION_INTERVAL
        or stage.get("optimizer") != expected_optimizer
        or stage.get("fixed_update_budget_consumed") is not True
        or str(stage.get("initial_state_sha256", ""))
        != EXPECTED_SEED10_INITIAL_STATE_SHA256
    ):
        raise ValueError(f"seed-20260710 stage contract drift: {stage_name}")
    schedule = frozen_minibatch_schedule(320, updates, seed)
    if (
        stage.get("minibatch_indices") != schedule
        or str(stage.get("minibatch_indices_sha256", ""))
        != v3.canonical_json_sha256(schedule)
    ):
        raise ValueError(f"seed-20260710 minibatch drift: {stage_name}")
    curve = stage.get("learning_curve")
    if not isinstance(curve, list):
        raise ValueError(f"seed-20260710 curve missing: {stage_name}")
    summary = terminal_fit_gate_summary(
        curve,
        updates,
        AUTHORITATIVE_EVALUATION_INTERVAL,
    )
    if summary != stage.get("terminal_fit_gate"):
        raise ValueError(f"seed-20260710 terminal drift: {stage_name}")
    training = stage.get("training_access", {})
    evaluation = stage.get("fit_evaluation_access", {})
    curve_count = len(curve)
    if (
        int(training.get("model_calls", -1)) != updates
        or int(training.get("model_output_frames", -1))
        != updates * BATCH_SIZE
        or int(evaluation.get("model_calls", -1)) != curve_count * 80
        or int(evaluation.get("model_output_frames", -1))
        != curve_count * 80 * 12
    ):
        raise ValueError(f"seed-20260710 stage access drift: {stage_name}")
    return summary


def _validate_primary_authorization(
    path: Path,
    expected_sha256: str,
    current_sources: Mapping[str, Any],
    patch7_reference: Mapping[str, Any],
) -> dict[str, Any]:
    if path.resolve() != _canonical_output(20260710):
        raise ValueError("seed-20260710 authorization path is not canonical")
    if _sha256_file(path) != expected_sha256:
        raise ValueError("seed-20260710 authorization SHA-256 mismatch")
    result = _read_json(path)
    if result.get("schema") != RESULT_SCHEMA or result.get("seed") != 20260710:
        raise ValueError("seed-20260710 authorization schema/seed mismatch")
    core = dict(result)
    declared = str(core.pop("content_sha256", ""))
    if v3.canonical_json_sha256(core) != declared:
        raise ValueError("seed-20260710 authorization content mismatch")
    if (
        result.get("authoritative") is not True
        or result.get("aggregation_eligible") is not True
        or result.get("source_hashes") != current_sources
        or result.get("decision", {}).get("favorable") is not True
        or result.get("categorical_radial_full_train_candidate_licensed")
        is not False
    ):
        raise ValueError("seed-20260710 result does not authorize seed 20260711")
    execution = result.get("execution", {})
    if (
        execution.get("batch_size_frames") != BATCH_SIZE
        or execution.get("evaluation_interval")
        != AUTHORITATIVE_EVALUATION_INTERVAL
        or execution.get("branches") != AUTHORITATIVE_BRANCHES
        or execution.get("fp32_no_autocast_amp_compile_or_quantization") is not True
    ):
        raise ValueError("seed-20260710 execution contract drift")
    inputs = result.get("inputs", {})
    expected_inputs = {
        "panel": (PANEL_FILE_SHA256, PANEL_CONTENT_SHA256),
        "ladder_manifest": (LADDER_FILE_SHA256, LADDER_CONTENT_SHA256),
        "v3_result": (V3_RESULT_FILE_SHA256, V3_RESULT_CONTENT_SHA256),
        "patch7_reference_result": (
            PATCH7_RESULT_FILE_SHA256,
            PATCH7_RESULT_CONTENT_SHA256,
        ),
    }
    for name, (file_digest, content_digest) in expected_inputs.items():
        record = inputs.get(name)
        if (
            not isinstance(record, Mapping)
            or str(record.get("sha256", "")) != file_digest
            or str(record.get("content_sha256", "")) != content_digest
        ):
            raise ValueError(f"seed-20260710 authorization input drift: {name}")
    if result.get("contract") != {
        "path": str(CONTRACT_PATH.resolve()),
        "sha256": EXECUTION_BINDING_SHA256,
    }:
        raise ValueError("seed-20260710 authorization contract drift")
    if result.get("patch7_reference") != patch7_reference:
        raise ValueError("seed-20260710 patch7 reference drift")
    model = result.get("model", {})
    if (
        model.get("class") != "CategoricalRadialPerceptionFullRay"
        or int(model.get("parameter_count", -1)) != REGISTERED_PARAMETER_COUNT
        or str(model.get("initial_state_sha256", ""))
        != EXPECTED_SEED10_INITIAL_STATE_SHA256
    ):
        raise ValueError("seed-20260710 authorization model drift")
    stages = result.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != {
        "production_faithful",
        "ceiling_optimizer",
    }:
        raise ValueError("seed-20260710 stage structure drift")
    faithful = stages["production_faithful"]
    if not isinstance(faithful, Mapping):
        raise ValueError("seed-20260710 faithful stage is missing")
    faithful_summary = _validate_stored_stage(
        faithful,
        stage_name="production_faithful",
        seed=20260710,
    )
    ceiling = stages["ceiling_optimizer"]
    if faithful_summary["passes"]:
        if ceiling is not None:
            raise ValueError("seed-20260710 ceiling ran after faithful fit pass")
        qualifying = faithful
    else:
        if not isinstance(ceiling, Mapping):
            raise ValueError("seed-20260710 ceiling stage is mandatory")
        ceiling_summary = _validate_stored_stage(
            ceiling,
            stage_name="ceiling_optimizer",
            seed=20260710,
        )
        if ceiling["minibatch_indices"][:2000] != faithful["minibatch_indices"]:
            raise ValueError("seed-20260710 branch minibatch prefix drift")
        qualifying = ceiling if ceiling_summary["passes"] else None
    holdouts = result.get("holdouts")
    stored_checks = result.get("holdout_checks")
    access = result.get("access_ledger", {})
    panel_access = access.get("panels", {}) if isinstance(access, Mapping) else {}
    if qualifying is None:
        if holdouts is not None or stored_checks is not None:
            raise ValueError("seed-20260710 unauthorized holdout payload exists")
        if any(
            bool(panel_access.get(panel, {}).get("authorized", True))
            for panel in HOLDOUT_PANELS
        ):
            raise ValueError("seed-20260710 unauthorized holdout access exists")
    else:
        if (
            not isinstance(holdouts, Mapping)
            or set(holdouts) != set(HOLDOUT_PANELS)
            or not isinstance(stored_checks, Mapping)
            or set(stored_checks) != set(HOLDOUT_PANELS)
        ):
            raise ValueError("seed-20260710 authorized holdouts are incomplete")
        recomputed_checks = {
            panel: categorical_holdout_checks(
                holdouts[panel],
                patch7_reference["panels"][panel],
            )
            for panel in HOLDOUT_PANELS
        }
        if recomputed_checks != stored_checks:
            raise ValueError("seed-20260710 holdout decision drift")
        if any(
            panel_access.get(panel, {}).get("authorized") is not True
            for panel in HOLDOUT_PANELS
        ):
            raise ValueError("seed-20260710 holdout access ledger drift")
    for stage in (faithful, ceiling):
        if stage is None:
            continue
        expected_holdouts = stage is qualifying
        if bool(stage.get("holdouts_evaluated")) is not expected_holdouts:
            raise ValueError("seed-20260710 stage holdout flag drift")
    recomputed = per_seed_decision(faithful, ceiling, stored_checks)
    if recomputed != result.get("decision"):
        raise ValueError("seed-20260710 stored decision does not recompute")
    if any(int(access.get(name, -1)) != 0 for name in (
        "non_train_image_opens",
        "non_train_label_shard_opens",
        "non_train_model_outputs",
    )):
        raise ValueError("seed-20260710 authorization has forbidden access")
    fit_totals = access.get("fit_dataset_totals", {})
    if (
        int(fit_totals.get("image_decode_events", -1)) != 320
        or int(fit_totals.get("label_shard_npz_open_events", -1)) != 20
    ):
        raise ValueError("seed-20260710 fit access totals drift")
    for panel in ("fit", *HOLDOUT_PANELS):
        record = panel_access.get(panel)
        expected = EXPECTED_PANEL_ARTIFACT_COUNTS[panel]
        if (
            not isinstance(record, Mapping)
            or record.get("authorized") is not True
            or int(record.get("artifact_hash_passes", -1)) != 2
            or int(record.get("image_hash_byte_open_events", -1))
            != 2 * expected["images"]
            or int(record.get("shard_hash_byte_open_events", -1))
            != 2 * expected["shards"]
        ):
            raise ValueError(f"seed-20260710 panel access drift: {panel}")
        if panel in HOLDOUT_PANELS:
            dataset_access = record.get("dataset_access", {})
            if (
                int(dataset_access.get("image_decode_events", -1)) != 320
                or int(dataset_access.get("label_shard_npz_open_events", -1))
                != expected["shards"]
                or int(dataset_access.get("target_requests", -1)) != 320
                or int(dataset_access.get("model_calls", -1)) != 80
                or int(dataset_access.get("model_output_frames", -1)) != 960
            ):
                raise ValueError(
                    f"seed-20260710 holdout event drift: {panel}"
                )
    verification = result.get("artifact_verification", {})
    if (
        verification.get("fit_verified_before_access") is not True
        or verification.get("holdouts_verified_only_after_terminal_fit_pass")
        is not True
    ):
        raise ValueError("seed-20260710 artifact-verification drift")
    return result


def _reconcile_access(
    fit_dataset: PanelFrameDataset,
    stages: Mapping[str, Mapping[str, Any] | None],
    panel_access: Mapping[str, Any],
    holdouts: Mapping[str, Any] | None,
) -> None:
    totals = fit_dataset.snapshot()
    if (
        int(totals.get("image_decode_events", -1)) != 320
        or int(totals.get("label_shard_npz_open_events", -1)) != 20
    ):
        raise RuntimeError("fit cached-access events do not reconcile")
    for stage in stages.values():
        if stage is None:
            continue
        updates = int(stage["completed_steps"])
        curve_count = len(stage["learning_curve"])
        training = stage["training_access"]
        evaluation = stage["fit_evaluation_access"]
        if (
            int(training.get("model_calls", -1)) != updates
            or int(training.get("model_output_frames", -1))
            != updates * BATCH_SIZE
            or int(evaluation.get("model_calls", -1)) != curve_count * 80
            or int(evaluation.get("model_output_frames", -1))
            != curve_count * 80 * 12
        ):
            raise RuntimeError("stage model-output events do not reconcile")
    if holdouts is None:
        if any(panel_access[panel]["authorized"] for panel in HOLDOUT_PANELS):
            raise RuntimeError("unauthorized holdout access was recorded")
        return
    for panel in HOLDOUT_PANELS:
        access = panel_access[panel]["dataset_access"]
        expected_shards = EXPECTED_PANEL_ARTIFACT_COUNTS[panel]["shards"]
        if (
            int(access.get("image_decode_events", -1)) != 320
            or int(access.get("label_shard_npz_open_events", -1))
            != expected_shards
            or int(access.get("target_requests", -1)) != 320
            or int(access.get("model_calls", -1)) != 80
            or int(access.get("model_output_frames", -1)) != 960
        ):
            raise RuntimeError(f"{panel} access events do not reconcile")


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
        parser.error("output already exists; N32 results are immutable")
    if args.seed not in (20260710, 20260711):
        parser.error("N32 seed must be 20260710 or 20260711")
    authorization = (
        args.seed_20260710_result,
        args.expected_seed_20260710_sha256,
    )
    if args.seed == 20260710 and any(value is not None for value in authorization):
        parser.error("seed 20260710 rejects seed-authorization arguments")
    if args.seed == 20260711 and any(value is None for value in authorization):
        parser.error("seed 20260711 requires both seed-authorization arguments")
    if args.non_authoritative_smoke and args.seed != 20260710:
        parser.error("N32 smoke is seed-20260710-only")
    if not args.non_authoritative_smoke and args.output.resolve() != _canonical_output(
        args.seed
    ):
        parser.error("authoritative N32 output must use its canonical path")
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
    panel, panels, ladder, v3_result, patch7_reference = _load_bound_inputs()
    git_start = v3.v2.v1._git_snapshot()
    primary = None
    if args.seed == 20260711:
        primary = _validate_primary_authorization(
            args.seed_20260710_result.resolve(),
            str(args.expected_seed_20260710_sha256),
            source_start,
            patch7_reference,
        )

    device = v3.v2.v1._resolve_device(str(args.device))
    determinism = v3.v2.v1._configure_determinism(int(args.seed))
    initial_model = CategoricalRadialPerceptionFullRay()
    parameter_count = sum(parameter.numel() for parameter in initial_model.parameters())
    if parameter_count != REGISTERED_PARAMETER_COUNT:
        raise RuntimeError("N32 model parameter count changed")
    initial_state = v3.v2.v1._clone_state(initial_model.state_dict())
    initial_state_sha256 = v3.v2.v1._state_dict_sha256(initial_state)
    if args.seed == 20260710 and (
        initial_state_sha256 != EXPECTED_SEED10_INITIAL_STATE_SHA256
    ):
        raise RuntimeError("N32 seed-20260710 initialization changed")
    del initial_model

    fit_records, fit_controls = _canonical_panel_records(
        panels["fit"], seed=args.seed, panel="fit"
    )
    fit_images, fit_shards = _artifact_contract(fit_records, "fit")
    _verify_artifacts(fit_images, fit_shards)
    fit_dataset = PanelFrameDataset(fit_records, "fit")
    smoke = bool(args.non_authoritative_smoke)
    configs = SMOKE_BRANCHES if smoke else AUTHORITATIVE_BRANCHES
    interval = (
        SMOKE_EVALUATION_INTERVAL
        if smoke
        else AUTHORITATIVE_EVALUATION_INTERVAL
    )
    faithful, faithful_model = _run_stage(
        stage_name="production_faithful",
        config=configs["production_faithful"],
        initial_state=initial_state,
        initial_state_sha256=initial_state_sha256,
        dataset=fit_dataset,
        records=fit_records,
        controls=fit_controls,
        device=device,
        seed=args.seed,
        evaluation_interval=interval,
    )
    ceiling = None
    qualifying_model = None
    if faithful["terminal_fit_gate"]["passes"]:
        qualifying_model = faithful_model
    else:
        del faithful_model
        ceiling, ceiling_model = _run_stage(
            stage_name="ceiling_optimizer",
            config=configs["ceiling_optimizer"],
            initial_state=initial_state,
            initial_state_sha256=initial_state_sha256,
            dataset=fit_dataset,
            records=fit_records,
            controls=fit_controls,
            device=device,
            seed=args.seed,
            evaluation_interval=interval,
        )
        prefix = len(faithful["minibatch_indices"])
        if ceiling["minibatch_indices"][:prefix] != faithful["minibatch_indices"]:
            raise RuntimeError("faithful/ceiling minibatch prefixes differ")
        if ceiling["terminal_fit_gate"]["passes"]:
            qualifying_model = ceiling_model
        else:
            del ceiling_model

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
    if qualifying_model is not None:
        holdouts = {}
        holdout_checks = {}
        for panel_name in HOLDOUT_PANELS:
            records, controls = _canonical_panel_records(
                panels[panel_name], seed=args.seed, panel=panel_name
            )
            images, shards = _artifact_contract(records, panel_name)
            _verify_artifacts(images, shards)
            dataset = PanelFrameDataset(records, panel_name)
            report, access = evaluate_panel(
                qualifying_model,
                dataset,
                records,
                device=device,
                panel=panel_name,
                controls=controls,
            )
            _verify_artifacts(images, shards)
            holdouts[panel_name] = report
            holdout_checks[panel_name] = categorical_holdout_checks(
                report,
                patch7_reference["panels"][panel_name],
            )
            panel_access[panel_name] = {
                "authorized": True,
                "artifact_hash_passes": 2,
                "image_hash_byte_open_events": 2 * len(images),
                "shard_hash_byte_open_events": 2 * len(shards),
                "dataset_access": access,
            }
        if faithful["terminal_fit_gate"]["passes"]:
            faithful["holdouts_evaluated"] = True
        else:
            ceiling["holdouts_evaluated"] = True
        del qualifying_model
    else:
        for panel_name in HOLDOUT_PANELS:
            panel_access[panel_name] = {
                "authorized": False,
                "artifact_hash_passes": 0,
                "image_hash_byte_open_events": 0,
                "shard_hash_byte_open_events": 0,
                "model_output_frames": 0,
            }
    _verify_artifacts(fit_images, fit_shards)
    decision = per_seed_decision(faithful, ceiling, holdout_checks)
    decision["aggregation_eligible"] = not smoke
    decision["categorical_radial_full_train_candidate_licensed"] = False
    decision["promotion_licensed"] = False

    if primary is not None and _sha256_file(
        args.seed_20260710_result.resolve()
    ) != str(args.expected_seed_20260710_sha256):
        raise RuntimeError("seed-20260710 authorization changed during execution")
    evidence_hashes = {
        str(path.resolve()): digest
        for path, digest in {
            PANEL_PATH: PANEL_FILE_SHA256,
            LADDER_PATH: LADDER_FILE_SHA256,
            V3_RESULT_PATH: V3_RESULT_FILE_SHA256,
            PATCH7_RESULT_PATH: PATCH7_RESULT_FILE_SHA256,
            PROTOCOL_PATH: PROTOCOL_SHA256,
            CONTRACT_PATH: EXECUTION_BINDING_SHA256,
        }.items()
    }
    for path, digest in evidence_hashes.items():
        if _sha256_file(Path(path)) != digest:
            raise RuntimeError(f"bound N32 evidence changed during execution: {path}")
    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("N32 sources changed during execution")
    git_end = v3.v2.v1._git_snapshot()
    authoritative = not smoke
    stages = {
        "production_faithful": faithful,
        "ceiling_optimizer": ceiling,
    }
    _reconcile_access(fit_dataset, stages, panel_access, holdouts)
    access_ledger = {
        "panels": panel_access,
        "fit_dataset_totals": fit_dataset.snapshot(),
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
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "determinism": determinism,
            "batch_size_frames": BATCH_SIZE,
            "evaluation_interval": interval,
            "branches": configs,
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
            "all_invoked_branches_restart_same_initial_state": all(
                stage is None
                or stage["initial_state_sha256"] == initial_state_sha256
                for stage in stages.values()
            ),
        },
        "stages": stages,
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
    payload = {**core, "content_sha256": v3.canonical_json_sha256(core)}
    v3.v2.v1._atomic_write_json_exclusive(args.output.resolve(), payload)
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
