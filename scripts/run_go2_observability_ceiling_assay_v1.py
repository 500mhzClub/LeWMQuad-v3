#!/usr/bin/env python3
"""Run the registered observability-ceiling assay V1.

This runner consumes the immutable CPU-flat V3 collection, encodes its frames
once with the frozen DINOv2 and V-JEPA 2.1 encoders, fits the registered
capacity ladder on the train role only, and evaluates the registered arms on
the scene-disjoint evaluation role.

It performs no Genesis work, no rendering, no recollection, no scene filtering,
and no mutation of the consumed collection.  It *does* open the evaluation
successor RGB for the first time; that one-way custody cost is declared in
section 4 of the preregistration.

The registered decision rule lives in
``lewm/benchmarks/go2_observability_ceiling_assay_v1.decide_v1`` and is not
reachable from the command line.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
for _package_root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical  # noqa: E402
from lewm.benchmarks import go2_observability_ceiling_assay_v1 as assay  # noqa: E402


STEM = "go2_observability_ceiling_assay_v1"
ATTEMPT_LABEL = "attempt_v1"
ATTEMPT_ID = f"{STEM}_{ATTEMPT_LABEL}"
ATTEMPT_ROOT = REPO_ROOT / ".generated" / "dev" / STEM / ATTEMPT_LABEL
RESULT_PATH = ATTEMPT_ROOT / "result.json"
TERMINAL_PATH = ATTEMPT_ROOT / "terminal.json"
FEATURE_ROOT = ATTEMPT_ROOT / "features"


def _bind_attempt(label: str) -> None:
    """Point the immutable output paths at a named attempt.

    A new label is the only way to run again after an infrastructure failure;
    an existing attempt is never overwritten, resumed, or repaired in place.
    """

    global ATTEMPT_LABEL, ATTEMPT_ID, ATTEMPT_ROOT, RESULT_PATH, TERMINAL_PATH
    global FEATURE_ROOT
    ATTEMPT_LABEL = label
    ATTEMPT_ID = f"{STEM}_{label}"
    ATTEMPT_ROOT = REPO_ROOT / ".generated" / "dev" / STEM / label
    RESULT_PATH = ATTEMPT_ROOT / "result.json"
    TERMINAL_PATH = ATTEMPT_ROOT / "terminal.json"
    FEATURE_ROOT = ATTEMPT_ROOT / "features"

PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / f"lewm_{STEM}_preregistration_2026-08-05.md"
)
AMENDMENT_1_PATH = REPO_ROOT / "docs" / f"lewm_{STEM}_amendment_1_2026-08-05.md"

COLLECTION_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3/attempt_v1/collection"
)
COLLECTION_RESULT_PATH = COLLECTION_ROOT / "physics_result.json"
COLLECTION_RESULT_SHA256 = (
    "711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0"
)

DINO_REPOSITORY = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
DINO_CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
VJEPA_REPOSITORY = Path.home() / ".cache/vjepa2-204698b45b3712590f06245fbfba32d3be539812"
VJEPA_CHECKPOINT = Path.home() / ".cache/vjepa2_1_vitb_dist_vitG_384.pt"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINO_FEATURE_DIM = 384
VJEPA_FEATURE_DIM = 768

EXPECTED_RGB_OPENS = {
    "train_context": 384,
    "train_successor": 1152,
    "eval_context": 384,
    "eval_successor": 1152,
}


class CeilingAssayRunnerError(RuntimeError):
    """Raised when the runner contract is violated."""


# --------------------------------------------------------------------------
# Custody
# --------------------------------------------------------------------------


@dataclass
class AccessLedgerV1:
    """Account for every RGB frame this assay opens, by role and kind."""

    rgb_opens: dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in EXPECTED_RGB_OPENS}
    )
    opened: set[tuple[str, str, str]] = field(default_factory=set)
    receipt_opens: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "eval": 0}
    )
    encoder_executions: dict[str, int] = field(
        default_factory=lambda: {"dinov2": 0, "vjepa2_1": 0}
    )

    def open_rgb(self, role: str, kind: str, artifact_id: str) -> None:
        key = f"{role}_{kind}"
        if key not in self.rgb_opens:
            raise CeilingAssayRunnerError(f"unexpected RGB class {key}")
        token = (role, kind, artifact_id)
        if token in self.opened:
            raise CeilingAssayRunnerError(f"RGB artifact reopened: {artifact_id}")
        self.opened.add(token)
        self.rgb_opens[key] += 1

    def open_receipt(self, role: str) -> None:
        if role not in self.receipt_opens:
            raise CeilingAssayRunnerError("unexpected receipt role")
        self.receipt_opens[role] += 1

    def finalize(self) -> dict[str, object]:
        if self.rgb_opens != EXPECTED_RGB_OPENS:
            raise CeilingAssayRunnerError(
                f"RGB access ledger mismatch: {self.rgb_opens} != {EXPECTED_RGB_OPENS}"
            )
        if self.receipt_opens != {"train": 128, "eval": 128}:
            raise CeilingAssayRunnerError("state receipt ledger mismatch")
        return {
            "rgb_opens": dict(self.rgb_opens),
            "expected_rgb_opens": dict(EXPECTED_RGB_OPENS),
            "state_receipt_opens": dict(self.receipt_opens),
            "encoder_executions": dict(self.encoder_executions),
            "successor_custody_note": (
                "evaluation successor RGB opened for the first time; the V3 panel "
                "is now spent for privileged-successor purposes"
            ),
        }


def file_binding_v1(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def canonical_bytes_v1(value: object) -> bytes:
    return assay.canonical_bytes_v1(value)


def write_json_exclusive_v1(path: Path, value: Mapping[str, Any]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, raw)
    finally:
        os.close(descriptor)
    return file_binding_v1(path)


# --------------------------------------------------------------------------
# Role loading
# --------------------------------------------------------------------------


def load_role_v1(
    role: str, *, ledger: AccessLedgerV1
) -> tuple[tuple[Any, ...], Mapping[str, Mapping[str, Any]]]:
    """Read the 128 bound state receipts for one role and build its groups."""

    scene_root = COLLECTION_ROOT / "scenes" / role
    if not scene_root.is_dir():
        raise CeilingAssayRunnerError(f"{role} scene root is missing")
    receipt_paths = sorted(scene_root.glob("*/state_receipts/*.json"))
    if len(receipt_paths) != assay.STATE_COUNT:
        raise CeilingAssayRunnerError(f"{role} receipt count changed")
    receipts = []
    for path in receipt_paths:
        document = json.loads(path.read_text())
        if document.get("status") != "PHYSICS_COMPLETE":
            raise CeilingAssayRunnerError(f"{role} receipt is not physics-complete")
        ledger.open_receipt(role)
        receipts.append(document)
    groups, receipt_by_id = physical._groups_from_receipts(  # noqa: SLF001
        receipts, role=role
    )
    assay.validate_role_geometry_v1(groups, role=role)
    return groups, receipt_by_id


def rgb_path_v1(role: str, artifact_id: str) -> Path:
    """Resolve an artifact identity to its bound PNG path.

    Identities have the form ``<state_id>:<kind>:<index>`` and state IDs embed
    the scene identity, so the path is fully determined by the identity.
    """

    parts = artifact_id.split(":")
    if len(parts) != 3 or parts[1] not in {"context", "candidate"}:
        raise CeilingAssayRunnerError(f"unexpected artifact identity {artifact_id}")
    state_id, kind, index = parts
    scene_id = state_id.rsplit("-state-", 1)[0].split(f"scene-diversity-{role}-", 1)[-1]
    path = (
        COLLECTION_ROOT
        / "scenes"
        / role
        / scene_id
        / "rgb"
        / f"{state_id}.{kind}.{index}.png"
    )
    if not path.is_file():
        raise CeilingAssayRunnerError(f"bound RGB artifact is missing: {path}")
    return path


def role_artifact_slots_v1(
    groups: Sequence[Any],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Return per-state context and successor artifact identities."""

    contexts = []
    successors = []
    for group in groups:
        context_ids = tuple(group.context_rgb_artifact_ids)
        successor_ids = tuple(
            branch.target_rgb_artifact_id
            for branch in sorted(group.branches, key=lambda item: int(item.action_id))
        )
        if (
            len(context_ids) != assay.CONTEXT_FRAME_COUNT
            or len(successor_ids) != assay.ACTION_COUNT
        ):
            raise CeilingAssayRunnerError("artifact slot geometry changed")
        contexts.append(context_ids)
        successors.append(successor_ids)
    return tuple(contexts), tuple(successors)


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------


def _decode_png_224_v1(raw: bytes) -> np.ndarray:
    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG":
            raise CeilingAssayRunnerError("bound RGB artifact is not PNG")
        rgb = image.convert("RGB")
        if rgb.size != (224, 224):
            raise CeilingAssayRunnerError("bound RGB artifact is not 224x224")
        return np.asarray(rgb, dtype=np.uint8).copy()


def preprocess_dino_v1(raw: bytes) -> torch.Tensor:
    array = _decode_png_224_v1(raw)
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


class FrozenDINOEncoderV1:
    """Full frozen DINOv2 ViT-S/14: 12 blocks, final norm, patch tokens only."""

    def __init__(self, device: torch.device) -> None:
        model = torch.hub.load(
            str(DINO_REPOSITORY), "dinov2_vits14", source="local", pretrained=False
        )
        payload = torch.load(DINO_CHECKPOINT, map_location="cpu", weights_only=True)
        if isinstance(payload, Mapping) and "state_dict" in payload:
            payload = payload["state_dict"]
        model.load_state_dict(payload, strict=True)
        self.model = model.to(device).eval().requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.model.prepare_tokens_with_masks(
            images.to(self.device, torch.float32)
        )
        for block in self.model.blocks:
            hidden = block(hidden)
        hidden = self.model.norm(hidden)
        patches = hidden[:, 1:, :]
        if tuple(patches.shape[1:]) != (assay.TOKEN_COUNT, DINO_FEATURE_DIM):
            raise CeilingAssayRunnerError("DINO patch grid changed")
        normalized = F.normalize(patches.float(), p=2.0, dim=-1)
        if not bool(torch.isfinite(normalized).all()):
            raise CeilingAssayRunnerError("DINO tokens are nonfinite")
        return normalized.detach().cpu()


class FrozenVJEPAEncoderV1:
    """Frozen V-JEPA 2.1 ViT-B/384 reduced to the shared 16x16 token grid."""

    def __init__(self, device: torch.device) -> None:
        from scripts import (  # noqa: PLC0415
            run_go2_dense_vjepa2_1_physical_interface_ceiling_v1 as ceiling,
        )

        self._ceiling = ceiling
        with ceiling.scoped_timm_drop_path_shim_v1():
            encoder, predictor = torch.hub.load(
                str(VJEPA_REPOSITORY),
                "vjepa2_1_vit_base_384",
                source="local",
                pretrained=False,
            )
        del predictor
        payload = torch.load(VJEPA_CHECKPOINT, map_location="cpu", weights_only=True)
        state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in payload["ema_encoder"].items()
        }
        encoder.load_state_dict(state, strict=True)
        del payload, state
        self.model = encoder.to(device).eval().requires_grad_(False)
        self.device = device

    def preprocess(self, raw: bytes) -> torch.Tensor:
        return self._ceiling.preprocess_vjepa2_1_png_bytes_v1(raw)

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model(images.to(self.device, torch.float32))
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        normalized = self._ceiling.normalize_vjepa_token_grid_v1(tokens)
        return normalized.detach().cpu()


def encode_role_v1(
    role: str,
    contexts: Sequence[Sequence[str]],
    successors: Sequence[Sequence[str]],
    *,
    encoder: Any,
    preprocess: Callable[[bytes], torch.Tensor],
    feature_dim: int,
    ledger: AccessLedgerV1,
    encoder_name: str,
    batch_size: int = 16,
    count_access: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one role's context and successor frames exactly once each."""

    def run(identities: Sequence[str], kind: str) -> torch.Tensor:
        rows = []
        buffer: list[torch.Tensor] = []
        for artifact_id in identities:
            path = rgb_path_v1(role, artifact_id)
            raw = path.read_bytes()
            if count_access:
                ledger.open_rgb(
                    role, "context" if kind == "context" else "successor", artifact_id
                )
            buffer.append(preprocess(raw))
            if len(buffer) == batch_size:
                rows.append(encoder.encode(torch.stack(buffer)))
                ledger.encoder_executions[encoder_name] += len(buffer)
                buffer = []
        if buffer:
            rows.append(encoder.encode(torch.stack(buffer)))
            ledger.encoder_executions[encoder_name] += len(buffer)
        return torch.cat(rows, dim=0)

    context_ids = [item for state in contexts for item in state]
    successor_ids = [item for state in successors for item in state]
    context_tokens = run(context_ids, "context").reshape(
        len(contexts), assay.CONTEXT_FRAME_COUNT, assay.TOKEN_COUNT, feature_dim
    )
    successor_tokens = run(successor_ids, "successor").reshape(
        len(successors), assay.ACTION_COUNT, assay.TOKEN_COUNT, feature_dim
    )
    return context_tokens, successor_tokens


# --------------------------------------------------------------------------
# Panel builders
# --------------------------------------------------------------------------


def make_visual_panel_builder_v1(
    projected_context: torch.Tensor,
    projected_successor: torch.Tensor | None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build ``[z_c, z_s, z_s - z_c]`` batches, or the current-state panel."""

    current = projected_context[:, -1]

    def build(selected: torch.Tensor) -> torch.Tensor:
        base = current[selected]
        repeated = base.unsqueeze(1).expand(-1, assay.ACTION_COUNT, -1, -1)
        if projected_successor is None:
            successor = repeated
        else:
            successor = projected_successor[selected]
        panel = assay.relational_panel_v1(
            repeated.reshape(-1, assay.TOKEN_COUNT, repeated.shape[-1]),
            successor.reshape(-1, assay.TOKEN_COUNT, successor.shape[-1]),
        )
        return panel.contiguous()

    return build


def make_privileged_panel_builder_v1(
    features: torch.Tensor, *, relational_width: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    def build(selected: torch.Tensor) -> torch.Tensor:
        chosen = features[selected].reshape(-1, features.shape[-1])
        return assay.broadcast_feature_panel_v1(
            chosen, relational_width=relational_width
        )

    return build


# --------------------------------------------------------------------------
# Assay execution
# --------------------------------------------------------------------------


def _fit_and_score_arm(
    *,
    arm: str,
    rung: Mapping[str, int],
    train_groups: Sequence[Any],
    eval_groups: Sequence[Any],
    train_builder_factory: Callable[[int], Callable[[torch.Tensor], torch.Tensor]],
    eval_builder_factory: Callable[[int], Callable[[torch.Tensor], torch.Tensor]],
    fit_indices: Sequence[int],
    train_conditions: torch.Tensor,
    eval_conditions: torch.Tensor,
    train_residual: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    pca_width = int(rung["pca_width"])
    hidden_width = int(rung["hidden_width"])
    train_builder = train_builder_factory(pca_width)
    eval_builder = eval_builder_factory(pca_width)
    states = []
    diagnostics = []
    for seed in assay.MODEL_SEEDS:
        state, diagnostic = assay.train_readout_v1(
            train_builder,
            train_conditions,
            train_residual,
            fit_indices,
            seed=seed,
            pca_width=pca_width,
            hidden_width=hidden_width,
            device=device,
        )
        states.append(state)
        diagnostics.append(diagnostic)
    train_residual_scores = assay.predict_scores_v1(
        states,
        train_builder,
        train_conditions,
        pca_width=pca_width,
        hidden_width=hidden_width,
        device=device,
    )
    eval_residual_scores = assay.predict_scores_v1(
        states,
        eval_builder,
        eval_conditions,
        pca_width=pca_width,
        hidden_width=hidden_width,
        device=device,
    )
    return {
        "arm": arm,
        "rung": rung["name"],
        "members": diagnostics,
        "train_residual_scores": train_residual_scores,
        "eval_residual_scores": eval_residual_scores,
    }


def execute_assay_v1(device: torch.device, *, encode_vjepa: bool) -> dict[str, Any]:
    started = time.time()
    ledger = AccessLedgerV1()

    collection_binding = file_binding_v1(COLLECTION_RESULT_PATH)
    if collection_binding["sha256"] != COLLECTION_RESULT_SHA256:
        raise CeilingAssayRunnerError(
            "consumed collection result does not rehash to its registered SHA-256"
        )

    train_groups, train_receipts = load_role_v1("train", ledger=ledger)
    eval_groups, eval_receipts = load_role_v1("eval", ledger=ledger)
    disjointness = assay.require_role_disjointness_v1(train_groups, eval_groups)

    train_contexts, train_successors = role_artifact_slots_v1(train_groups)
    eval_contexts, eval_successors = role_artifact_slots_v1(eval_groups)

    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)

    dino = FrozenDINOEncoderV1(device)
    train_dino_context, train_dino_successor = encode_role_v1(
        "train",
        train_contexts,
        train_successors,
        encoder=dino,
        preprocess=preprocess_dino_v1,
        feature_dim=DINO_FEATURE_DIM,
        ledger=ledger,
        encoder_name="dinov2",
        count_access=True,
    )
    eval_dino_context, eval_dino_successor = encode_role_v1(
        "eval",
        eval_contexts,
        eval_successors,
        encoder=dino,
        preprocess=preprocess_dino_v1,
        feature_dim=DINO_FEATURE_DIM,
        ledger=ledger,
        encoder_name="dinov2",
        count_access=True,
    )
    del dino
    torch.cuda.empty_cache() if device.type == "cuda" else None

    vjepa_caches: dict[str, torch.Tensor] = {}
    if encode_vjepa:
        vjepa = FrozenVJEPAEncoderV1(device)
        train_vjepa_context, train_vjepa_successor = encode_role_v1(
            "train",
            train_contexts,
            train_successors,
            encoder=vjepa,
            preprocess=vjepa.preprocess,
            feature_dim=VJEPA_FEATURE_DIM,
            ledger=ledger,
            encoder_name="vjepa2_1",
            count_access=False,
        )
        eval_vjepa_context, eval_vjepa_successor = encode_role_v1(
            "eval",
            eval_contexts,
            eval_successors,
            encoder=vjepa,
            preprocess=vjepa.preprocess,
            feature_dim=VJEPA_FEATURE_DIM,
            ledger=ledger,
            encoder_name="vjepa2_1",
            count_access=False,
        )
        del vjepa
        torch.cuda.empty_cache() if device.type == "cuda" else None
        vjepa_caches = {
            "train_context": train_vjepa_context,
            "train_successor": train_vjepa_successor,
            "eval_context": eval_vjepa_context,
            "eval_successor": eval_vjepa_successor,
        }

    access = ledger.finalize()

    # Registered targets, conditions, and controls.
    train_conditions = assay.conditions_v1(train_groups)
    eval_conditions = assay.conditions_v1(eval_groups)
    train_targets = assay.normalized_rank_targets_v1(train_groups)

    split = assay.inner_split_v1(train_groups)
    fit_indices = assay.state_indices_for_scenes_v1(train_groups, split["fit"])
    validation_indices = assay.state_indices_for_scenes_v1(
        train_groups, split["validation"]
    )
    all_train_indices = tuple(range(len(train_groups)))

    train_privileged = assay.privileged_physical_features_v1(
        train_groups, train_receipts
    )
    eval_privileged = assay.privileged_physical_features_v1(eval_groups, eval_receipts)

    def build_arm_panels(
        arm: str, pca_states: Sequence[int]
    ) -> tuple[
        Callable[[int], Callable[[torch.Tensor], torch.Tensor]],
        Callable[[int], Callable[[torch.Tensor], torch.Tensor]],
        dict[str, Any],
    ]:
        """Return train/eval panel-builder factories for one arm."""

        pca_records: dict[str, Any] = {}

        if arm == assay.PRIVILEGED_ARM:

            def train_factory(width: int):
                return make_privileged_panel_builder_v1(
                    train_privileged, relational_width=3 * width
                )

            def eval_factory(width: int):
                return make_privileged_panel_builder_v1(
                    eval_privileged, relational_width=3 * width
                )

            return train_factory, eval_factory, pca_records

        if arm == assay.DINO_ARM:
            train_context, train_successor = train_dino_context, train_dino_successor
            eval_context, eval_successor = eval_dino_context, eval_dino_successor
        elif arm == assay.VJEPA_ARM:
            train_context = vjepa_caches["train_context"]
            train_successor = vjepa_caches["train_successor"]
            eval_context = vjepa_caches["eval_context"]
            eval_successor = vjepa_caches["eval_successor"]
        elif arm == assay.CONTEXT_ARM:
            train_context, train_successor = train_dino_context, None
            eval_context, eval_successor = eval_dino_context, None
        else:
            raise CeilingAssayRunnerError(f"arm {arm} has no dense panel")

        cache: dict[int, dict[str, Any]] = {}

        def prepare(width: int) -> dict[str, Any]:
            if width in cache:
                return cache[width]
            pca = assay.fit_pca_v1(
                train_context, width=width, state_indices=pca_states
            )
            entry = {
                "pca": pca,
                "train_context": assay.project_tokens_v1(train_context, pca),
                "eval_context": assay.project_tokens_v1(eval_context, pca),
                "train_successor": (
                    assay.project_tokens_v1(train_successor, pca)
                    if train_successor is not None
                    else None
                ),
                "eval_successor": (
                    assay.project_tokens_v1(eval_successor, pca)
                    if eval_successor is not None
                    else None
                ),
            }
            pca_records[f"width_{width}"] = {
                "explained_variance_ratio": pca["explained_variance_ratio"],
                "width": pca["width"],
                "fit_states": len(pca_states),
            }
            cache[width] = entry
            return entry

        def train_factory(width: int):
            entry = prepare(width)
            return make_visual_panel_builder_v1(
                entry["train_context"], entry["train_successor"]
            )

        def eval_factory(width: int):
            entry = prepare(width)
            return make_visual_panel_builder_v1(
                entry["eval_context"], entry["eval_successor"]
            )

        return train_factory, eval_factory, pca_records

    dense_arms = [assay.PRIVILEGED_ARM, assay.DINO_ARM, assay.CONTEXT_ARM]
    if encode_vjepa:
        dense_arms.insert(2, assay.VJEPA_ARM)

    # ---- Stage 1: inner rung selection, fit on the 24 fit scenes only ----
    inner_ridge = assay.fit_task_ridge_v1(train_groups, fit_indices)
    inner_task_scores = assay.score_task_ridge_v1(inner_ridge)
    inner_residual = torch.from_numpy(
        (train_targets - inner_task_scores).astype(np.float32)
    )
    validation_groups = [train_groups[index] for index in validation_indices]

    inner: dict[str, dict[str, Any]] = {}
    for arm in dense_arms:
        train_factory, eval_factory, _records = build_arm_panels(arm, fit_indices)
        inner[arm] = {}
        for rung in assay.RUNGS:
            fitted = _fit_and_score_arm(
                arm=arm,
                rung=rung,
                train_groups=train_groups,
                eval_groups=eval_groups,
                train_builder_factory=train_factory,
                eval_builder_factory=train_factory,
                fit_indices=fit_indices,
                train_conditions=train_conditions,
                eval_conditions=train_conditions,
                train_residual=inner_residual,
                device=device,
            )
            combined = inner_task_scores + fitted["train_residual_scores"]
            validation_scores = combined[list(validation_indices)]
            report = assay.arm_report_v1(
                validation_groups, validation_scores, policy="argmin"
            )
            fit_report = assay.arm_report_v1(
                [train_groups[index] for index in fit_indices],
                combined[list(fit_indices)],
                policy="argmin",
            )
            inner[arm][rung["name"]] = {
                "inner_validation": report["summary"],
                "fit_split": fit_report["summary"],
                "members": fitted["members"],
            }

    selected_rung_name = min(
        (rung["name"] for rung in assay.RUNGS),
        key=lambda name: inner[assay.DINO_ARM][name]["inner_validation"][
            "normalized_rank_regret"
        ],
    )
    selected_rung = next(
        rung for rung in assay.RUNGS if rung["name"] == selected_rung_name
    )

    # ---- Stage 2: refit every rung on all 32 train scenes, score on eval ----
    full_ridge = assay.fit_task_ridge_v1(train_groups, all_train_indices)
    full_task_train = assay.score_task_ridge_v1(full_ridge)
    full_residual = torch.from_numpy(
        (train_targets - full_task_train).astype(np.float32)
    )
    eval_task_features = np.stack(
        [
            np.asarray(
                (
                    1.0,
                    group.relative_target_xy_body_m[0],
                    group.relative_target_xy_body_m[1],
                    float(np.hypot(*group.relative_target_xy_body_m)),
                ),
                dtype=np.float64,
            )
            for group in eval_groups
        ]
    )
    eval_task_scores = eval_task_features @ full_ridge["coefficients"].T

    rung_sensitivity: dict[str, dict[str, float]] = {}
    selected_reports: dict[str, dict[str, Any]] = {}
    pca_report: dict[str, Any] = {}
    dino_train_regret: dict[str, float] = {}
    for arm in dense_arms:
        train_factory, eval_factory, records = build_arm_panels(arm, all_train_indices)
        pca_report[arm] = records
        rung_sensitivity[arm] = {}
        for rung in assay.RUNGS:
            fitted = _fit_and_score_arm(
                arm=arm,
                rung=rung,
                train_groups=train_groups,
                eval_groups=eval_groups,
                train_builder_factory=train_factory,
                eval_builder_factory=eval_factory,
                fit_indices=all_train_indices,
                train_conditions=train_conditions,
                eval_conditions=eval_conditions,
                train_residual=full_residual,
                device=device,
            )
            eval_scores = eval_task_scores + fitted["eval_residual_scores"]
            report = assay.arm_report_v1(eval_groups, eval_scores, policy="argmin")
            rung_sensitivity[arm][rung["name"]] = report["summary"][
                "normalized_rank_regret"
            ]
            train_report = assay.arm_report_v1(
                train_groups,
                full_task_train + fitted["train_residual_scores"],
                policy="argmin",
            )
            if arm == assay.DINO_ARM:
                dino_train_regret[rung["name"]] = train_report["summary"][
                    "normalized_rank_regret"
                ]
            if rung["name"] == selected_rung_name:
                report["train_summary"] = train_report["summary"]
                report["members"] = fitted["members"]
                selected_reports[arm] = report

    # ---- Non-learned arms ----
    selected_reports[assay.ORACLE_ARM] = assay.arm_report_v1(
        eval_groups, None, policy="oracle"
    )
    selected_reports[assay.RANDOM_ARM] = assay.arm_report_v1(
        eval_groups, None, policy="random"
    )
    selected_reports[assay.TASK_ARM] = assay.arm_report_v1(
        eval_groups, eval_task_scores, policy="argmin"
    )
    if not encode_vjepa:
        selected_reports[assay.VJEPA_ARM] = {
            "selection_policy": "NOT_RUN",
            "summary": {"normalized_rank_regret": None},
            "state_results": [],
        }

    # ---- Paired comparisons ----
    def rows(arm: str) -> Sequence[Mapping[str, object]]:
        return selected_reports[arm]["state_results"]

    comparison_pairs = [
        (assay.DINO_ARM, assay.TASK_ARM),
        (assay.DINO_ARM, assay.CONTEXT_ARM),
        (assay.DINO_ARM, assay.PRIVILEGED_ARM),
        (assay.CONTEXT_ARM, assay.DINO_ARM),
        (assay.CONTEXT_ARM, assay.TASK_ARM),
        (assay.TASK_ARM, assay.RANDOM_ARM),
    ]
    if encode_vjepa:
        comparison_pairs.extend(
            [(assay.VJEPA_ARM, assay.DINO_ARM), (assay.VJEPA_ARM, assay.TASK_ARM)]
        )
    comparisons: dict[str, Any] = {}
    for candidate, baseline in comparison_pairs:
        comparisons[f"{candidate}_minus_{baseline}"] = (
            assay.paired_family_scene_bootstrap_v1(rows(candidate), rows(baseline))
        )

    # ---- Diagnostics ----
    bins = assay.displacement_spread_bins_v1(eval_groups)
    spread = {
        arm: assay.spread_conditioned_regret_v1(eval_groups, rows(arm), bins)
        for arm in (assay.DINO_ARM, assay.CONTEXT_ARM)
    }
    power = {
        name: {
            "ci_half_width": value["ci_half_width"],
            "scenes_to_resolve_0.02": assay.scenes_to_resolve_effect_v1(
                value["ci_half_width"], 0.02
            ),
        }
        for name, value in comparisons.items()
    }

    # ---- Amendment 1 validity controls ----
    identifiability_states = [
        assay.train_privileged_mlp_v1(
            train_privileged,
            train_conditions,
            full_residual,
            all_train_indices,
            seed=seed,
            device=device,
        )
        for seed in assay.MODEL_SEEDS
    ]
    identifiability_scores = assay.predict_privileged_mlp_v1(
        identifiability_states, eval_privileged, eval_conditions, device=device
    )
    identifiability_report = assay.arm_report_v1(
        eval_groups, eval_task_scores + identifiability_scores, policy="argmin"
    )
    identifiability_regret = identifiability_report["summary"][
        "normalized_rank_regret"
    ]
    top_rung_name = assay.RUNGS[-1]["name"]
    expressivity_regret = dino_train_regret[top_rung_name]

    decision = assay.decide_v1(
        selected_reports,
        comparisons,
        identifiability_regret=identifiability_regret,
        expressivity_regret=expressivity_regret,
    )

    integrity = {
        "collection_binding": collection_binding,
        "collection_sha256_matches_registered": True,
        "role_disjointness": disjointness,
        "oracle_regret_is_zero": (
            selected_reports[assay.ORACLE_ARM]["summary"]["normalized_rank_regret"]
            == 0.0
        ),
        "ceiling_beats_random": (
            selected_reports[assay.DINO_ARM]["summary"]["normalized_rank_regret"]
            < selected_reports[assay.RANDOM_ARM]["summary"]["normalized_rank_regret"]
        ),
        "access_ledger": access,
    }
    if not integrity["oracle_regret_is_zero"] or not integrity["ceiling_beats_random"]:
        raise CeilingAssayRunnerError("registered integrity gate failed")

    result: dict[str, Any] = {
        "schema": assay.SCHEMA,
        "attempt_id": ATTEMPT_ID,
        "development_only": True,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "preregistration_binding": file_binding_v1(PREREGISTRATION_PATH),
        "amendment_bindings": [file_binding_v1(AMENDMENT_1_PATH)],
        "config": assay.config_v1(),
        "integrity": integrity,
        "inner_split": {
            "fit_scenes": list(split["fit"]),
            "validation_scenes": list(split["validation"]),
            "selected_rung": selected_rung_name,
            "selected_rung_parameters": selected_rung,
            "inner_validation": {
                arm: {
                    name: inner[arm][name]["inner_validation"] for name in inner[arm]
                }
                for arm in inner
            },
            "fit_split": {
                arm: {name: inner[arm][name]["fit_split"] for name in inner[arm]}
                for arm in inner
            },
        },
        "pca": pca_report,
        "arms": {
            arm: {
                key: value
                for key, value in report.items()
                if key != "state_results"
            }
            for arm, report in selected_reports.items()
        },
        # Compact per-state selection record.  The metric-validity study needs
        # each arm's chosen branch to compute geometric regret in metres on the
        # identical states; the full per-state rows are not carried here.
        "per_state_selection": {
            arm: [
                {
                    "state_id": row["state_id"],
                    "scene_id": row["scene_id"],
                    "family": row["family"],
                    "selected_action_id": row["selected_action_id"],
                    "normalized_rank_regret": row["normalized_rank_regret"],
                }
                for row in report.get("state_results", [])
            ]
            for arm, report in selected_reports.items()
        },
        "rung_sensitivity": rung_sensitivity,
        "validity_controls": {
            "amendment": "lewm_go2_observability_ceiling_assay_v1_amendment_1_2026-08-05",
            "identifiability_2a": {
                "summary": identifiability_report["summary"],
                "threshold": assay.CAPACITY_CONTROL_MAX_REGRET,
            },
            "expressivity_2b": {
                "dino_train_regret_by_rung": dino_train_regret,
                "top_rung": top_rung_name,
                "value": expressivity_regret,
                "threshold": assay.CAPACITY_CONTROL_MAX_REGRET,
            },
        },
        "comparisons": comparisons,
        "power": power,
        "displacement_spread": {
            "quartile_edges_m": bins["quartile_edges_m"],
            "per_arm": spread,
        },
        "decision": decision,
        "wall_seconds": time.time() - started,
    }
    result["identity_sha256"] = assay.result_identity_v1(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda", help="torch device for encoding and fitting"
    )
    parser.add_argument(
        "--skip-vjepa",
        action="store_true",
        help="omit the V-JEPA comparator arm (arm 4) and its comparisons",
    )
    parser.add_argument(
        "--attempt",
        default="attempt_v1",
        help=(
            "attempt label; a fresh label is required after an infrastructure "
            "failure and never overwrites an existing attempt"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    _bind_attempt(arguments.attempt)
    device = torch.device(arguments.device)
    if RESULT_PATH.exists():
        raise CeilingAssayRunnerError(
            "an immutable result already exists; this attempt is not resumable"
        )
    try:
        result = execute_assay_v1(device, encode_vjepa=not arguments.skip_vjepa)
    except Exception as error:  # noqa: BLE001
        ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
        if not TERMINAL_PATH.exists():
            write_json_exclusive_v1(
                TERMINAL_PATH,
                {
                    "schema": f"{assay.SCHEMA}_terminal",
                    "attempt_id": ATTEMPT_ID,
                    "status": "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION",
                    "error": f"{type(error).__name__}: {error}",
                    "citable_as_scientific_evidence": False,
                },
            )
        raise
    binding = write_json_exclusive_v1(RESULT_PATH, result)
    write_json_exclusive_v1(
        TERMINAL_PATH,
        {
            "schema": f"{assay.SCHEMA}_terminal",
            "attempt_id": ATTEMPT_ID,
            "status": result["decision"]["terminal"],
            "assay_valid": result["decision"]["assay_valid"],
            "ceiling_regret": result["decision"]["ceiling_regret"],
            "result_binding": binding,
            "development_only": True,
            "citable_as_scientific_evidence": False,
            "authorizes_retry_or_resume": False,
        },
    )
    print(json.dumps(result["decision"], indent=2))
    print(json.dumps({arm: result["arms"][arm]["summary"] for arm in result["arms"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
