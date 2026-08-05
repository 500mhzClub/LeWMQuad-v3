#!/usr/bin/env python3
"""Run the train-only categorical-radial N=1/4/16 overfit ladder."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "lewm_worlds"))

from lewm.benchmarks.go2_categorical_radial_factorization import (  # noqa: E402
    audit_mapping_injectivity,
    geometry_metadata,
)
from lewm.benchmarks.go2_categorical_radial_micro_overfit import (  # noqa: E402
    CLASS_NAMES,
    LADDER_NAMESPACE,
    LADDER_PREFIX_SIZES,
    LADDER_SCHEMA,
    canonical_json_sha256,
    ladder_fit_gate,
)
from lewm.benchmarks.go2_physical_micro_overfit import (  # noqa: E402
    TRAINING_WEIGHTS,
    empty_raw_accumulator,
    finalize_raw_accumulator,
    update_raw_accumulator,
)
from lewm.models.categorical_radial_perception import (  # noqa: E402
    CategoricalRadialPerception,
    IMAGE_SIZE,
)


MANIFEST_SCHEMA = "lewm_go2_categorical_radial_ladder_manifest_v1"
RESULT_SCHEMA = "lewm_go2_categorical_radial_ladder_result_v1"
SMOKE_RESULT_SCHEMA = "lewm_go2_categorical_radial_ladder_smoke_result_v1"
STAGE_SCHEMA = "lewm_go2_categorical_radial_ladder_stage_v1"
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
AUTHORITATIVE_STAGES = {
    1: {"updates": 1000, "batch_size": 1},
    4: {"updates": 1500, "batch_size": 4},
    16: {"updates": 2000, "batch_size": 4},
}
SMOKE_STAGES = {
    size: {"updates": 3, "batch_size": config["batch_size"]}
    for size, config in AUTHORITATIVE_STAGES.items()
}
AUTHORITATIVE_EVALUATION_INTERVAL = 100
SMOKE_EVALUATION_INTERVAL = 1
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
SOURCE_PATHS = {
    "encoder": REPOSITORY_ROOT / "lewm/models/encoders.py",
    "factorization": (
        REPOSITORY_ROOT
        / "lewm/benchmarks/go2_categorical_radial_factorization.py"
    ),
    "ladder_contract": (
        REPOSITORY_ROOT
        / "lewm/benchmarks/go2_categorical_radial_micro_overfit.py"
    ),
    "model": REPOSITORY_ROOT / "lewm/models/categorical_radial_perception.py",
    "panel_contract": (
        REPOSITORY_ROOT / "lewm/benchmarks/go2_physical_micro_overfit.py"
    ),
    "preparer": REPOSITORY_ROOT / "scripts/prepare_go2_categorical_radial_ladder.py",
    "protocol": (
        REPOSITORY_ROOT
        / "docs/lewm_go2_categorical_radial_microfit_protocol_2026-07-10.md"
    ),
    "runner": Path(__file__).resolve(),
}
MANIFEST_SOURCE_BINDINGS = frozenset(
    {"factorization", "ladder_contract", "panel_contract", "preparer", "protocol"}
)
FRAME_FIELDS = frozenset(
    {
        "scene_id",
        "family",
        "global_row",
        "side",
        "image_path",
        "image_sha256",
        "label_shard_path",
        "label_shard_sha256",
        "label_shard_row",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in sorted(SOURCE_PATHS.items())
    }


def _git_snapshot() -> dict[str, Any]:
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip()
    return {"head": head, "status_short": status}


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _clone_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _frame_prefix_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": str(record["scene_id"]),
        "global_row": int(record["global_row"]),
        "side": str(record["side"]),
        "image_sha256": str(record["image_sha256"]),
    }


def _validated_frame(record: Mapping[str, Any]) -> dict[str, Any]:
    if set(record) != FRAME_FIELDS:
        raise ValueError(
            "ladder selected frame fields differ from the frozen contract: "
            f"{sorted(set(record) ^ FRAME_FIELDS)}"
        )
    normalized = {
        "scene_id": str(record["scene_id"]),
        "family": str(record["family"]),
        "global_row": int(record["global_row"]),
        "side": str(record["side"]),
        "image_path": str(Path(str(record["image_path"])).resolve()),
        "image_sha256": str(record["image_sha256"]),
        "label_shard_path": str(Path(str(record["label_shard_path"])).resolve()),
        "label_shard_sha256": str(record["label_shard_sha256"]),
        "label_shard_row": int(record["label_shard_row"]),
    }
    if not normalized["scene_id"] or not normalized["family"]:
        raise ValueError("ladder selected frame scene and family must be nonempty")
    if normalized["side"] not in {"current", "next"}:
        raise ValueError("ladder selected frame side must be current or next")
    if normalized["global_row"] < 0 or normalized["label_shard_row"] < 0:
        raise ValueError("ladder selected frame indices must be nonnegative")
    for name in ("image_sha256", "label_shard_sha256"):
        if not _is_sha256(normalized[name]):
            raise ValueError(f"ladder selected frame has invalid {name}")
    return normalized


def validate_ladder_manifest(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the immutable preparer output without opening selected artifacts."""

    core = dict(manifest)
    declared_content_sha256 = str(core.pop("content_sha256", ""))
    if manifest.get("schema") != MANIFEST_SCHEMA or (
        canonical_json_sha256(core) != declared_content_sha256
    ):
        raise ValueError("invalid categorical-radial ladder manifest content")

    embedded_sources = manifest.get("source_hashes")
    if not isinstance(embedded_sources, Mapping) or set(embedded_sources) != set(
        MANIFEST_SOURCE_BINDINGS
    ):
        raise ValueError("ladder manifest source bindings are incomplete")
    current_sources = _source_hashes()
    for name in MANIFEST_SOURCE_BINDINGS:
        embedded = embedded_sources[name]
        if (
            not isinstance(embedded, Mapping)
            or Path(str(embedded.get("path", ""))).resolve()
            != SOURCE_PATHS[name].resolve()
            or str(embedded.get("sha256", "")) != current_sources[name]["sha256"]
        ):
            raise ValueError(f"ladder manifest was prepared under different {name}")

    if manifest.get("factorization") != geometry_metadata():
        raise ValueError("ladder manifest factorization metadata changed")
    if manifest.get("mapping_audit") != audit_mapping_injectivity():
        raise ValueError("ladder manifest mapping audit changed")
    roundtrip = manifest.get("roundtrip_audit")
    if (
        not isinstance(roundtrip, Mapping)
        or roundtrip.get("all_960_frames_exact") is not True
    ):
        raise ValueError("ladder manifest lacks its exact 960-frame roundtrip")
    panels = roundtrip.get("panels")
    if not isinstance(panels, Mapping) or set(panels) != {
        "fit",
        "same_scene_holdout",
        "cross_scene_holdout",
    }:
        raise ValueError("ladder roundtrip panels are incomplete")
    for name, report in panels.items():
        if (
            not isinstance(report, Mapping)
            or int(report.get("frame_count", -1)) != 320
            or int(report.get("outside_support_known_count", -1)) != 0
            or int(report.get("roundtrip_mismatch_count", -1)) != 0
            or report.get("exact_roundtrip") is not True
        ):
            raise ValueError(f"ladder roundtrip report is invalid: {name}")

    ledger = manifest.get("artifact_access_ledger")
    if not isinstance(ledger, Mapping) or ledger.get(
        "runner_input_contains_only_train_rows"
    ) is not True:
        raise ValueError("ladder manifest does not certify train-only runner input")
    if int(ledger.get("train_image_byte_opens", -1)) != 0:
        raise ValueError("ladder preparer opened train image bytes")
    for role in ("checkpoint_selection", "probability_calibration", "g2_evaluation"):
        record = ledger.get(role)
        if not isinstance(record, Mapping) or any(
            int(record.get(key, -1)) != 0
            for key in ("image_byte_opens", "label_shard_byte_opens", "model_outputs")
        ):
            raise ValueError(f"ladder manifest records forbidden {role} contact")

    ladder = manifest.get("ladder")
    if not isinstance(ladder, Mapping):
        raise ValueError("ladder manifest lacks its selected ladder")
    ladder_core = {
        key: value
        for key, value in ladder.items()
        if key not in {"selected_frames", "content_sha256"}
    }
    if (
        ladder.get("schema") != LADDER_SCHEMA
        or ladder.get("namespace") != LADDER_NAMESPACE
        or ladder.get("class_names") != list(CLASS_NAMES)
        or canonical_json_sha256(ladder_core) != str(ladder.get("content_sha256", ""))
    ):
        raise ValueError("ladder selection content is invalid")
    selected_raw = ladder.get("selected_frames")
    if (
        not isinstance(selected_raw, list)
        or len(selected_raw) != max(LADDER_PREFIX_SIZES)
    ):
        raise ValueError("ladder must contain exactly 16 selected frames")
    selected = [_validated_frame(record) for record in selected_raw]
    if len({record["scene_id"] for record in selected}) != len(selected):
        raise ValueError("ladder selected frames are not scene-disjoint")
    if len({record["image_sha256"] for record in selected}) != len(selected):
        raise ValueError("ladder selected image hashes are not unique")
    identities = {(record["global_row"], record["side"]) for record in selected}
    if len(identities) != len(selected):
        raise ValueError("ladder selected frame identities are not unique")

    prefixes = ladder.get("prefixes")
    if not isinstance(prefixes, Mapping) or set(prefixes) != {
        str(size) for size in LADDER_PREFIX_SIZES
    }:
        raise ValueError("ladder prefixes are incomplete")
    for size in LADDER_PREFIX_SIZES:
        record = prefixes[str(size)]
        expected_frames = [
            _frame_prefix_identity(frame) for frame in selected[:size]
        ]
        if (
            not isinstance(record, Mapping)
            or int(record.get("frame_count", -1)) != size
            or record.get("frames") != expected_frames
            or str(record.get("frames_sha256", ""))
            != canonical_json_sha256(expected_frames)
        ):
            raise ValueError(f"ladder prefix is invalid: N={size}")
    if ladder.get("anchor") != _frame_prefix_identity(selected[0]):
        raise ValueError("ladder anchor differs from its first selected frame")

    inputs = manifest.get("inputs")
    panel = inputs.get("panel_manifest") if isinstance(inputs, Mapping) else None
    if not isinstance(panel, Mapping):
        raise ValueError("ladder manifest lacks its frozen parent panel")
    panel_path = Path(str(panel.get("path", ""))).resolve()
    panel_sha256 = str(panel.get("sha256", ""))
    if (
        not _is_sha256(panel_sha256)
        or str(panel.get("expected_sha256", "")) != panel_sha256
        or panel.get("pre_deserialization_hash_match") is not True
        or not panel_path.is_file()
        or _sha256_file(panel_path) != panel_sha256
    ):
        raise ValueError("frozen parent panel hash validation failed")
    return selected


def deterministic_cyclic_wrong_view(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Return the frozen one-step cyclic, zero-image/scene-match control."""

    count = len(records)
    if count < 2:
        raise ValueError("wrong-view control requires at least two frames")
    scenes = tuple(str(record["scene_id"]) for record in records)
    images = tuple(str(record["image_sha256"]) for record in records)
    if len(set(scenes)) != count or len(set(images)) != count:
        raise ValueError("wrong-view inputs must be scene- and image-disjoint")
    permutation = tuple((*range(1, count), 0))
    same_scene = sum(
        scenes[index] == scenes[source]
        for index, source in enumerate(permutation)
    )
    same_image = sum(
        images[index] == images[source]
        for index, source in enumerate(permutation)
    )
    if same_scene or same_image:
        raise AssertionError("cyclic wrong-view permutation contains a match")
    return permutation, {
        "schema": "lewm_go2_categorical_radial_cyclic_wrong_view_v1",
        "record_count": count,
        "offset": 1,
        "permutation": list(permutation),
        "permutation_sha256": canonical_json_sha256(list(permutation)),
        "same_scene_pairs": 0,
        "same_image_pairs": 0,
    }


def _weighted_cross_entropy_mean(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    applied = weights[labels] * mask.to(dtype=loss.dtype)
    return (loss * applied).sum() / applied.sum().clamp_min(
        torch.finfo(loss.dtype).tiny
    )


def hierarchical_occupancy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Production-aligned equal-capacity UNKNOWN/KNOWN and FREE/OCC loss."""

    if logits.ndim != 4 or logits.shape[1:] != (3, 64, 64):
        raise ValueError("logits must have shape (B, 3, 64, 64)")
    if (
        labels.shape != logits.shape[:1] + logits.shape[2:]
        or labels.dtype != torch.long
    ):
        raise ValueError("labels must be int64 with shape (B, 64, 64)")
    if mask.shape != labels.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be bool with shape (B, 64, 64)")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) > 2):
        raise ValueError("labels must be UNKNOWN/FREE/OCCUPIED")
    uk_weights = logits.new_tensor(TRAINING_WEIGHTS["unknown_known"])
    fo_weights = logits.new_tensor(TRAINING_WEIGHTS["free_occupied"])
    known_logit = torch.logsumexp(logits[:, 1:], dim=1)
    uk_logits = torch.stack((logits[:, 0], known_logit), dim=1)
    uk_labels = (labels != 0).long()
    uk_loss = _weighted_cross_entropy_mean(uk_logits, uk_labels, mask, uk_weights)
    known_mask = mask & (labels != 0)
    fo_labels = (labels - 1).clamp_min(0)
    fo_loss = _weighted_cross_entropy_mean(
        logits[:, 1:], fo_labels, known_mask, fo_weights
    )
    return 0.5 * uk_loss + 0.5 * fo_loss


class LadderFrameDataset:
    """Exact, cached reader for the 16 precommitted train-only ladder frames."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = [dict(record) for record in records]
        self._images: dict[int, torch.Tensor] = {}
        self._targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._shards: dict[str, dict[str, np.ndarray]] = {}
        self.image_requests = 0
        self.target_requests = 0
        self.image_decode_events = 0
        self.label_shard_npz_open_events = 0
        self.opened_image_paths: set[str] = set()
        self.opened_shard_paths: set[str] = set()

    def __len__(self) -> int:
        return len(self.records)

    def _image(self, index: int) -> torch.Tensor:
        self.image_requests += 1
        if index in self._images:
            return self._images[index]
        path = str(self.records[index]["image_path"])
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = tensor.new_tensor(NORMALIZATION_MEAN)[:, None, None]
        std = tensor.new_tensor(NORMALIZATION_STD)[:, None, None]
        self._images[index] = (tensor - mean) / std
        self.image_decode_events += 1
        self.opened_image_paths.add(path)
        return self._images[index]

    def _target(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.target_requests += 1
        if index in self._targets:
            return self._targets[index]
        record = self.records[index]
        path = str(record["label_shard_path"])
        if path not in self._shards:
            with np.load(path, allow_pickle=False) as archive:
                self._shards[path] = {
                    name: np.asarray(archive[name]) for name in archive.files
                }
            self.label_shard_npz_open_events += 1
            self.opened_shard_paths.add(path)
        shard = self._shards[path]
        side = str(record["side"])
        row = int(record["label_shard_row"])
        labels = np.asarray(shard[f"{side}_labels"][row], dtype=np.int64)
        mask = np.asarray(shard[f"{side}_supervision_mask"][row], dtype=bool)
        if labels.shape != (64, 64) or mask.shape != labels.shape:
            raise ValueError("ladder label shard uses an unexpected grid shape")
        if not np.isin(labels, (0, 1, 2)).all():
            raise ValueError("ladder label shard contains an invalid class")
        self._targets[index] = (
            torch.from_numpy(labels.copy()).long(),
            torch.from_numpy(mask.copy()).bool(),
        )
        return self._targets[index]

    def batch(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        images = []
        labels = []
        masks = []
        for raw_index in indices:
            index = int(raw_index)
            if not 0 <= index < len(self.records):
                raise IndexError("ladder batch index is out of range")
            target, mask = self._target(index)
            images.append(self._image(index))
            labels.append(target)
            masks.append(mask)
        return {
            "image": torch.stack(images),
            "labels": torch.stack(labels),
            "mask": torch.stack(masks),
        }

    def image_batch(self, indices: Sequence[int]) -> torch.Tensor:
        images = []
        for raw_index in indices:
            index = int(raw_index)
            if not 0 <= index < len(self.records):
                raise IndexError("ladder image index is out of range")
            images.append(self._image(index))
        return torch.stack(images)

    def access_ledger(self) -> dict[str, Any]:
        return {
            "image_requests": self.image_requests,
            "target_requests": self.target_requests,
            "image_decode_events": self.image_decode_events,
            "label_shard_npz_open_events": self.label_shard_npz_open_events,
            "distinct_image_paths_opened": len(self.opened_image_paths),
            "distinct_label_shards_opened": len(self.opened_shard_paths),
            "non_train_image_opens": 0,
            "non_train_label_shard_opens": 0,
        }


def _distance_grid() -> np.ndarray:
    forward = np.linspace(-0.95, 5.35, 64, dtype=np.float64)
    left = np.linspace(-3.15, 3.15, 64, dtype=np.float64)
    return np.sqrt(forward[:, None] ** 2 + left[None, :] ** 2)


@torch.no_grad()
def evaluate_ladder_model(
    model: CategoricalRadialPerception,
    dataset: LadderFrameDataset,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    frame_count = len(records)
    if frame_count not in LADDER_PREFIX_SIZES:
        raise ValueError("evaluation frame count is not a registered ladder size")
    wrong_indices: tuple[int, ...] | None = None
    wrong_contract = None
    if frame_count > 1:
        wrong_indices, wrong_contract = deterministic_cyclic_wrong_view(records)
    correct_accumulator = empty_raw_accumulator()
    wrong_accumulator = empty_raw_accumulator() if wrong_indices is not None else None
    distances = _distance_grid()
    model.eval()
    for start in range(0, frame_count, batch_size):
        target_indices = tuple(range(start, min(start + batch_size, frame_count)))
        correct_batch = dataset.batch(target_indices)
        correct_images = correct_batch["image"].to(device)
        if wrong_indices is None:
            correct_logits = model(correct_images).cpu().numpy()
            wrong_logits = None
        else:
            control_indices = tuple(wrong_indices[index] for index in target_indices)
            control_images = dataset.image_batch(control_indices).to(device)
            combined = model(torch.cat((correct_images, control_images), dim=0))
            correct_logits, wrong_tensor = combined.chunk(2, dim=0)
            correct_logits = correct_logits.cpu().numpy()
            wrong_logits = wrong_tensor.cpu().numpy()
        labels = correct_batch["labels"].numpy()
        mask = correct_batch["mask"].numpy()
        update_raw_accumulator(
            correct_accumulator,
            correct_logits,
            labels,
            mask,
            distances,
        )
        if wrong_accumulator is not None and wrong_logits is not None:
            update_raw_accumulator(
                wrong_accumulator,
                wrong_logits,
                labels,
                mask,
                distances,
            )
    correct = finalize_raw_accumulator(correct_accumulator)
    wrong = (
        None
        if wrong_accumulator is None
        else finalize_raw_accumulator(wrong_accumulator)
    )
    wrong_nll = None if wrong is None else wrong["raw_hierarchical_balanced_nll"]
    gate = ladder_fit_gate(
        correct,
        frame_count=frame_count,
        wrong_view_nll=None if wrong_nll is None else float(wrong_nll),
    )
    return {
        "schema": "lewm_go2_categorical_radial_ladder_evaluation_v1",
        "frame_count": frame_count,
        "correct_rgb": correct,
        "wrong_view_rgb": wrong,
        "wrong_view_control": wrong_contract,
        "fit_gate": gate,
    }


def _next_batch_indices(
    *,
    frame_count: int,
    batch_size: int,
    generator: torch.Generator,
    order: list[int],
) -> tuple[tuple[int, ...], list[int]]:
    if len(order) < batch_size:
        order.extend(torch.randperm(frame_count, generator=generator).tolist())
    batch = tuple(order[:batch_size])
    return batch, order[batch_size:]


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
    if updates <= 0 or evaluation_interval <= 0 or updates % evaluation_interval:
        raise ValueError(
            "stage updates must be positive and divisible by evaluation interval"
        )
    if batch_size <= 0 or frame_count % batch_size:
        raise ValueError("stage batch size must divide its frame count")
    model = CategoricalRadialPerception().to(device)
    model.load_state_dict(initial_state, strict=True)
    if _state_dict_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("stage did not restart from the frozen initial state")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    dataset = LadderFrameDataset(records)
    generator = torch.Generator().manual_seed(int(seed))
    remaining_order: list[int] = []
    curve = []
    step = 0
    while step < updates:
        indices, remaining_order = _next_batch_indices(
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
        loss = hierarchical_occupancy_loss(logits, labels, mask)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite ladder loss at step {step + 1}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP
        )
        optimizer.step()
        step += 1
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
                    "batch_loss": float(loss.detach().item()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "evaluation": evaluation,
                }
            )
    if not curve or int(curve[-1]["step"]) != updates:
        raise RuntimeError("ladder stage lacks its exact final evaluation")
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
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "fixed_budget_consumed": step == updates,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": _state_dict_sha256(model.state_dict()),
        "curve": curve,
        "final_evaluation": final_evaluation,
        "final_fit_gate_passes": bool(final_evaluation["fit_gate"]["passes"]),
        "access_ledger": dataset.access_ledger(),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _configure_determinism(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ROCm grid_sampler_2d_backward has no deterministic implementation. Keep
    # every supported kernel strict while surfacing unsupported kernels loudly.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "requested": "strict_deterministic_algorithms",
        "effective": "strict_where_supported_warn_on_unsupported",
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    return device


def _artifact_contract(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    images: dict[str, str] = {}
    shards: dict[str, str] = {}
    for record in records:
        for collection, path_key, sha_key in (
            (images, "image_path", "image_sha256"),
            (shards, "label_shard_path", "label_shard_sha256"),
        ):
            path = str(Path(str(record[path_key])).resolve())
            expected = str(record[sha_key])
            previous = collection.setdefault(path, expected)
            if previous != expected:
                raise ValueError(f"conflicting selected artifact hash: {path}")
    return images, shards


def _verify_artifacts(
    images: Mapping[str, str],
    shards: Mapping[str, str],
) -> None:
    for path, expected in (*sorted(images.items()), *sorted(shards.items())):
        if _sha256_file(Path(path)) != expected:
            raise ValueError(f"selected train artifact SHA-256 mismatch: {path}")


def _atomic_write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish complete JSON atomically without replacing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


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
    if not _is_sha256(args.expected_ladder_sha256):
        parser.error("expected-ladder-sha256 must be a lowercase SHA-256")
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
    expected_manifest_sha256 = str(args.expected_ladder_sha256)
    manifest_file_sha256 = _sha256_file(manifest_path)
    if manifest_file_sha256 != expected_manifest_sha256:
        raise ValueError("ladder manifest differs from its precommitted SHA-256")
    source_start = _source_hashes()
    git_start = _git_snapshot()
    manifest = _read_json(manifest_path)
    selected = validate_ladder_manifest(manifest)
    images, shards = _artifact_contract(selected)
    _verify_artifacts(images, shards)
    device = _resolve_device(str(args.device))
    deterministic = _configure_determinism(int(args.seed))

    initial_model = CategoricalRadialPerception()
    initial_state = _clone_state(initial_model.state_dict())
    initial_state_sha256 = _state_dict_sha256(initial_state)
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

    _verify_artifacts(images, shards)
    if _sha256_file(manifest_path) != manifest_file_sha256:
        raise RuntimeError("ladder manifest changed during execution")
    panel_path = Path(str(manifest["inputs"]["panel_manifest"]["path"])).resolve()
    panel_sha256 = str(manifest["inputs"]["panel_manifest"]["sha256"])
    if _sha256_file(panel_path) != panel_sha256:
        raise RuntimeError("frozen parent panel changed during execution")
    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("categorical-radial ladder sources changed during execution")
    git_end = _git_snapshot()

    completed_sizes = [int(stage["frame_count"]) for stage in stages]
    all_passed = completed_sizes == list(LADDER_PREFIX_SIZES) and all(
        bool(stage["final_fit_gate_passes"]) for stage in stages
    )
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
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "determinism": deterministic,
    }
    access_totals: Counter[str] = Counter()
    for stage in stages:
        for key, value in stage["access_ledger"].items():
            if isinstance(value, int):
                access_totals[key] += value
    core = {
        "schema": SMOKE_RESULT_SCHEMA if smoke else RESULT_SCHEMA,
        "created_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation,
        "authoritative": not smoke,
        "promotion_eligible": False,
        "train_only_implementation_diagnostic": True,
        "g2_evaluated": False,
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
        },
        "source_hashes": source_end,
        "git": {"start": git_start, "end": git_end},
        "execution": execution,
        "model": {
            "class": "CategoricalRadialPerception",
            "parameter_count": model_parameter_count,
            "initial_state_sha256": initial_state_sha256,
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
            "n32_attempted": False,
            "promotion_licensed": False,
        },
        "artifact_access_ledger": {
            "selected_train_images_hashed_per_pass": len(images),
            "selected_train_label_shards_hashed_per_pass": len(shards),
            "integrity_hash_passes": 2,
            "selected_train_image_hash_byte_open_events": 2 * len(images),
            "selected_train_label_shard_hash_byte_open_events": 2 * len(shards),
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
    _atomic_write_json_exclusive(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "file_sha256": _sha256_file(output_path),
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
