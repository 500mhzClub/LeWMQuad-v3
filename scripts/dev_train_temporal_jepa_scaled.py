#!/usr/bin/env python3
"""DEVELOPMENT-TIER scaled trainer for the temporal patch-memory JEPA.

Not a scientific attempt. Emits no sealed artifact, opens no held-out or sealed
material, writes only under `.generated/dev/`. It is a development diagnostic
for the registered model and control semantics; its results cannot qualify or
promote a model.

Differences from the registered V1 run, all deliberate:

- no 400-update cap (the frozen `training_update_v1` hard-raises there);
- batch 256 sequences instead of 10;
- warmup + cosine LR instead of a constant LR;
- all 16,000 bound train rows form the shuffled sampling pool, with permutation
  tails carried into the next batch rather than discarded;
- GPU-resident pre-decoded frames instead of per-step PNG decode;
- a fixed, operator-requested update budget; diagnostic metrics never alter it;
- an independently declared cosine-schedule horizon, so a bounded observation
  run can stop before the schedule endpoint without silently changing the
  learning-rate trajectory it is intended to reproduce.

The model, loss, mask schedule, EMA target, and control semantics are taken
unchanged from the reviewed modules. The loss is asserted equal to the frozen
`0.5 * (prediction - target).square().sum(-1).mean()` on every step.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

model_module = importlib.import_module(
    "lewm.models.rgb_recurrent_patch_memory_temporal_jepa_v1")
spatial_model_module = importlib.import_module(
    "lewm.models.rgb_single_frame_multiblock_masked_spatial_jepa_v1")
encoder_module = importlib.import_module("lewm.models.encoders")
training = importlib.import_module(
    "scripts.run_go2_rgb_recurrent_patch_memory_temporal_jepa_v1")
evaluation = importlib.import_module(
    "scripts.evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1")
metrics = importlib.import_module(
    "lewm.benchmarks.go2_rgb_recurrent_patch_memory_temporal_jepa_v1")
h6 = importlib.import_module(
    "lewm.datasets.go2_explicit_plan_discounted_successor_state_v27")
h6_census = importlib.import_module(
    "lewm.benchmarks.go2_recurrent_jepa_main_pool_census")
h4_v2 = importlib.import_module("lewm.datasets.go2_recurrent_h4_rgb_sequences_v2")
h4_v1 = importlib.import_module("lewm.datasets.go2_recurrent_h4_rgb_sequences")
packer = importlib.import_module("scripts.dev_pack_h6_temporal_frames")

PACK_ROOT = REPO_ROOT / ".generated/dev/h6_temporal_pack"
DEV_OUTPUT_ROOT = REPO_ROOT / ".generated/dev"
PREDECESSOR = (REPO_ROOT
               / ".generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
               / "attempt_v1/snapshots/update_1000.pt")
PREDECESSOR_BYTE_COUNT = 52_282_877
PREDECESSOR_SHA256 = (
    "f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873"
)
OUTPUT_ROOT = REPO_ROOT / ".generated/dev/temporal_jepa_scaled"
TRACE_SCHEMA = "dev_temporal_jepa_scaled_v4"
SNAPSHOT_SCHEMA = "dev_temporal_jepa_scaled_snapshot_v4"
ACTION_COUNT = int(evaluation.ACTION_COUNT_V1)
HOLD_ACTION = int(evaluation.HOLD_ACTION_INDEX_V1)
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)

if HOLD_ACTION != int(metrics.HOLD_ACTION_ID):
    raise RuntimeError("canonical temporal HOLD action identity changed")


def _artifact_path(pack_root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("pack artifact path must be a string")
    relative = Path(value)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != value
        or value in {"", ".", ".."}
    ):
        raise ValueError(f"pack artifact path is not one local filename: {value!r}")
    return Path(pack_root) / relative


def validate_pack_role(pack_root: Path, role: str) -> dict:
    """Validate a V2 pack role completely before constructing a memmap."""
    if role not in {"train", "val"}:
        raise ValueError("pack role must be train or val")
    root = Path(pack_root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("pack manifest must be a regular non-symlink file")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != packer.PACK_SCHEMA:
        raise ValueError("unsupported or legacy H6 temporal pack schema")
    if manifest.get("citable_as_scientific_evidence") is not False:
        raise ValueError("pack must remain explicitly non-citable")
    expected_sources = packer.pack_source_bindings()
    if manifest.get("sources") != expected_sources:
        raise ValueError("pack source bindings disagree with the current sources")
    if manifest.get("source_layout") != {
        "rgb_root": packer.RGB_ROOT.relative_to(REPO_ROOT).as_posix(),
        "visible_positions": list(packer.POSITIONS),
        "forbidden_positions_opened": [],
    }:
        raise ValueError("pack source layout or visible-position contract changed")
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, dict)
        or not isinstance(runtime.get("numpy_version"), str)
        or not runtime["numpy_version"]
        or not isinstance(runtime.get("pillow_version"), str)
        or not runtime["pillow_version"]
    ):
        raise ValueError("pack runtime provenance is absent")
    roles = manifest.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"train", "val"}:
        raise ValueError("pack must bind exactly train and val roles")
    selected = roles.get(role)
    if not isinstance(selected, dict):
        raise ValueError(f"pack role {role!r} is absent")

    binding = h6.INDEX_BINDINGS[role]
    expected_binding = {
        "role": binding.role,
        "path": binding.path.as_posix(),
        "byte_count": int(binding.byte_count),
        "file_sha256": binding.sha256,
    }
    if selected.get("index_binding") != expected_binding:
        raise ValueError(f"pack {role} index binding is stale or incomplete")
    rows = selected.get("rows")
    if type(rows) is not int or rows != binding.row_count:
        raise ValueError(f"pack {role} row count changed")
    if selected.get("positions") != list(packer.POSITIONS):
        raise ValueError(f"pack {role} visible positions changed")
    row_digest = selected.get("row_identity_sha256")
    if (
        not isinstance(row_digest, str)
        or len(row_digest) != 64
        or any(character not in "0123456789abcdef" for character in row_digest)
    ):
        raise ValueError(f"pack {role} row identity is not SHA-256-bound")

    bound_rows, index_audit = h6.load_bound_index(REPO_ROOT, role=role)
    # The reviewed loader may add non-identity audit fields. Bind every field
    # that determines which ordered index was opened without constraining those
    # additional diagnostics.
    for key, value in (
        ("role", expected_binding["role"]),
        ("path", expected_binding["path"]),
        ("byte_count", expected_binding["byte_count"]),
        ("file_sha256", expected_binding["file_sha256"]),
        ("row_count", rows),
    ):
        if index_audit.get(key) != value:
            raise ValueError(f"bound {role} index audit changed at {key}")
    if len(bound_rows) != rows or packer.row_identity_sha256(bound_rows) != row_digest:
        raise ValueError(f"pack {role} row identities do not match the bound index")
    source_rgb = selected.get("source_rgb")
    source_rgb_digest = (
        source_rgb.get("ordered_identity_sha256")
        if isinstance(source_rgb, dict)
        else None
    )
    if (
        not isinstance(source_rgb, dict)
        or source_rgb.get("leaf_count") != rows * len(packer.POSITIONS)
        or not isinstance(source_rgb_digest, str)
        or len(source_rgb_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_rgb_digest
        )
    ):
        raise ValueError(f"pack {role} source RGB identity is absent")

    expected = {
        "frames": {
            "dtype": "uint8",
            "shape": [rows, len(packer.POSITIONS), 112, 112, 3],
            "bytes": rows * len(packer.POSITIONS) * packer.FRAME_BYTES,
        },
        "actions": {"dtype": "int64", "shape": [rows, 3]},
    }
    paths = {}
    for name in ("frames", "actions", "metadata"):
        artifact = selected.get(name)
        if not isinstance(artifact, dict):
            raise ValueError(f"pack {role} {name} binding is absent")
        path = _artifact_path(root, artifact.get("path"))
        size = artifact.get("byte_count")
        digest = artifact.get("sha256")
        if (
            type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
            or packer.sha256_file(path) != digest
        ):
            raise ValueError(f"pack {role} {name} bytes or identity changed")
        paths[name] = path
    if selected["frames"].get("dtype") != expected["frames"]["dtype"]:
        raise ValueError(f"pack {role} frame dtype changed")
    if selected["frames"].get("shape") != expected["frames"]["shape"]:
        raise ValueError(f"pack {role} frame shape changed")
    if selected["frames"].get("byte_count") != expected["frames"]["bytes"]:
        raise ValueError(f"pack {role} frame byte count changed")
    if selected["actions"].get("dtype") != expected["actions"]["dtype"]:
        raise ValueError(f"pack {role} action dtype changed")
    if selected["actions"].get("shape") != expected["actions"]["shape"]:
        raise ValueError(f"pack {role} action shape changed")
    metadata = json.loads(paths["metadata"].read_text())
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("scene_ids"), list)
        or not isinstance(metadata.get("families"), list)
        or len(metadata["scene_ids"]) != rows
        or len(metadata["families"]) != rows
    ):
        raise ValueError(f"pack {role} metadata rows changed")
    if metadata["scene_ids"] != [row.scene_id for row in bound_rows]:
        raise ValueError(f"pack {role} scene metadata changed")
    if metadata["families"] != [row.family for row in bound_rows]:
        raise ValueError(f"pack {role} family metadata changed")
    verification = selected.get("verification")
    if (
        not isinstance(verification, dict)
        or verification.get("seed") != 20260731
        or type(verification.get("requested_rows")) is not int
        or verification["requested_rows"] < 1
        or type(verification.get("sampled_rows")) is not int
        or verification["sampled_rows"] < 1
        or verification["sampled_rows"] != min(rows, verification["requested_rows"])
        or verification.get("max_abs_deviation") != 0.0
    ):
        raise ValueError(f"pack {role} lacks a passing decoder verification")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": packer.sha256_file(manifest_path),
        "role": selected,
        "paths": paths,
        "bound_rows": bound_rows,
    }


def load_pack(pack_root: Path, role: str, device: torch.device):
    validated = validate_pack_role(pack_root, role)
    meta = validated["role"]
    rows = meta["rows"]
    frames = np.memmap(
        validated["paths"]["frames"], dtype=np.uint8, mode="c",
        shape=(rows, len(packer.POSITIONS), 112, 112, 3),
    )
    gpu = torch.from_numpy(frames).to(device)
    actions = torch.from_numpy(
        np.load(validated["paths"]["actions"], allow_pickle=False))
    if actions.dtype != torch.int64 or tuple(actions.shape) != (rows, 3):
        raise ValueError(f"pack {role} action tensor schema changed")
    if bool((actions < 0).any()) or bool((actions >= ACTION_COUNT).any()):
        raise ValueError(f"pack {role} action IDs left the canonical vocabulary")
    bound_rows = validated["bound_rows"]
    expected_actions = torch.tensor(
        [[int(row.actions[index]) for index in range(3)] for row in bound_rows],
        dtype=torch.int64,
    )
    if not torch.equal(actions, expected_actions):
        raise ValueError(f"pack {role} actions do not match the bound index")
    for name, path in validated["paths"].items():
        if packer.sha256_file(path) != meta[name]["sha256"]:
            raise ValueError(f"pack {role} {name} changed while it was loaded")
    return gpu, actions.to(device), {
        "manifest_path": str(validated["manifest_path"]),
        "manifest_sha256": validated["manifest_sha256"],
        "role": role,
        "row_identity_sha256": meta["row_identity_sha256"],
        "source_rgb": meta["source_rgb"],
        "index_binding": meta["index_binding"],
        "frames": meta["frames"],
        "actions": meta["actions"],
        "metadata": meta["metadata"],
    }


def _assert_pack_bindings_unchanged(
    pack_root: Path,
    expected_by_role: dict[str, dict],
) -> None:
    """Re-hash both packed roles before publishing a complete trace."""
    for role in ("train", "val"):
        expected = expected_by_role[role]
        current = validate_pack_role(pack_root, role)
        if current["manifest_sha256"] != expected["manifest_sha256"]:
            raise RuntimeError("H6 temporal pack manifest changed during training")
        current_role = current["role"]
        for field in (
            "row_identity_sha256",
            "source_rgb",
            "index_binding",
            "frames",
            "actions",
            "metadata",
        ):
            if current_role[field] != expected[field]:
                raise RuntimeError(
                    f"H6 temporal pack {role} {field} changed during training"
                )


def build_wrong_action_control(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse the registered `(action + 1) mod 9`, excluding canonical HOLD=6."""
    if actions.ndim != 2 or actions.shape[1] != 3 or actions.dtype != torch.long:
        raise TypeError("actions must be long with shape (B,3)")
    wrong_actions = actions.clone()
    wrong_actions[:, 2] = (wrong_actions[:, 2] + 1).remainder(ACTION_COUNT)
    eligible = actions[:, 2].ne(HOLD_ACTION)
    return wrong_actions, eligible


def carry_permutation_tail(
    order: torch.Tensor,
    cursor: int,
    batch_size: int,
    *,
    fresh_order: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Take one batch without discarding the end of the current permutation."""
    if (
        order.ndim != 1
        or order.dtype != torch.long
        or type(cursor) is not int
        or type(batch_size) is not int
        or not 0 <= cursor <= order.numel()
        or not 1 <= batch_size <= order.numel()
    ):
        raise ValueError("invalid permutation batch state")
    remaining = order.numel() - cursor
    if batch_size <= remaining:
        return order[cursor:cursor + batch_size], order, cursor + batch_size
    if (
        fresh_order is None
        or fresh_order.shape != order.shape
        or fresh_order.dtype != order.dtype
        or fresh_order.device != order.device
    ):
        raise ValueError("a matching fresh permutation is required at the boundary")
    tail_count = batch_size - remaining
    batch_rows = torch.cat((order[cursor:], fresh_order[:tail_count]))
    return batch_rows, fresh_order, tail_count


def to_float(packed_u8: torch.Tensor) -> torch.Tensor:
    """(B,P,112,112,3) uint8 -> (B,P,3,112,112) normalized float32, on device."""
    x = packed_u8.permute(0, 1, 4, 2, 3).to(torch.float32).div_(255.0)
    return x.sub_(IMAGENET_MEAN.to(x.device)).div_(IMAGENET_STD.to(x.device))


def effective_rank(tokens: torch.Tensor) -> tuple[float, float]:
    v = tokens.detach().to("cpu", torch.float64)
    rows, tok, dim = map(int, v.shape)
    centered = v - v.mean(dim=0, keepdim=True)
    flat = centered.reshape(-1, dim)
    cov = flat.T.mm(flat) / (rows * tok - 1)
    ev = torch.linalg.eigvalsh(0.5 * (cov + cov.T)).clamp_min(0.0)
    if float(ev.sum()) <= 0.0:
        return 0.0, 0.0
    p = ev / ev.sum()
    er = float((-(p * p.clamp_min(1e-12).log()).sum()).exp())
    return er, float(centered.square().sum() / (rows * tok * dim))


def _source_binding(path: Path) -> dict:
    selected = Path(path)
    return {
        "path": selected.resolve().relative_to(REPO_ROOT).as_posix(),
        "byte_count": selected.stat().st_size,
        "sha256": packer.sha256_file(selected),
    }


def training_source_paths() -> tuple[Path, ...]:
    """Return the complete local executable closure for pack validation/training."""

    return tuple(
        Path(path)
        for path in (
            __file__,
            model_module.__file__,
            spatial_model_module.__file__,
            encoder_module.__file__,
            training.__file__,
            evaluation.__file__,
            evaluation.spatial_evaluation.__file__,
            evaluation.spatial_evaluation.metrics.__file__,
            evaluation.spatial_evaluation.place_data.__file__,
            metrics.__file__,
            h6.__file__,
            h6_census.__file__,
            h4_v2.__file__,
            h4_v1.__file__,
            packer.__file__,
        )
    )


def device_receipt(requested: str, resolved: torch.device) -> dict[str, str | None]:
    return {
        "requested": str(requested),
        "resolved": str(resolved),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    }


def _assert_source_bindings_unchanged(bindings: list[dict]) -> None:
    for expected in bindings:
        path = REPO_ROOT / expected["path"]
        if _source_binding(path) != expected:
            raise RuntimeError(f"source changed during diagnostic: {expected['path']}")


def _input_binding(path: Path) -> dict:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"input is not a regular non-symlink file: {selected}")
    try:
        reported_path = selected.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        reported_path = str(selected.resolve())
    return {
        "path": reported_path,
        "byte_count": selected.stat().st_size,
        "sha256": packer.sha256_file(selected),
    }


def _write_immutable_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    if (
        path.exists()
        or path.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError(f"refusing to overwrite diagnostic JSON: {path}")
    with temporary.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, path)
    temporary.unlink()


def _save_immutable_checkpoint(path: Path, payload: dict) -> dict:
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
    os.link(temporary, path)
    temporary.unlink()
    try:
        reported_path = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        reported_path = str(path.resolve())
    return {
        "path": reported_path,
        "byte_count": path.stat().st_size,
        "sha256": packer.sha256_file(path),
    }


def _snapshot_payload(
    *, model, update: int, args, pack_bindings: dict,
    predecessor_binding: dict, source_bindings: list[dict],
) -> dict:
    return {
        "schema": SNAPSHOT_SCHEMA,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "update": int(update),
        "config": vars(args),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "pack_bindings": pack_bindings,
        "predecessor_binding": predecessor_binding,
        "source_bindings": source_bindings,
    }


def _validate_main_args(args) -> tuple[Path, Path]:
    if args.schedule_updates is None:
        args.schedule_updates = args.updates
    for name in ("updates", "batch", "microbatch", "eval_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least one")
    if args.schedule_updates < args.updates:
        raise ValueError("--schedule-updates cannot be smaller than --updates")
    if args.microbatch > args.batch:
        raise ValueError("--microbatch cannot exceed --batch")
    if args.lr_scale <= 0.0:
        raise ValueError("--lr-scale must be positive")
    if not 0 <= args.warmup <= args.updates:
        raise ValueError("--warmup must be in [0, updates]")
    if (
        not args.tag
        or Path(args.tag).is_absolute()
        or len(Path(args.tag).parts) != 1
        or args.tag in {".", ".."}
    ):
        raise ValueError("--tag must be one local directory name")

    pack_root = Path(args.pack_root).resolve()
    output_root = Path(args.output_root).resolve()
    dev_root = DEV_OUTPUT_ROOT.resolve()
    try:
        pack_root.relative_to(dev_root)
        output_root.relative_to(dev_root)
    except ValueError as exc:
        raise ValueError(
            "--pack-root and --output-root must stay under .generated/dev"
        ) from exc
    return pack_root, output_root


def learning_rate_fraction(
    update: int,
    *,
    warmup_updates: int,
    schedule_updates: int,
) -> float:
    """Return the warmup/cosine multiplier for one declared schedule step."""

    if not 1 <= update <= schedule_updates:
        raise ValueError("update must lie in [1, schedule_updates]")
    if not 0 <= warmup_updates <= schedule_updates:
        raise ValueError("warmup_updates must lie in [0, schedule_updates]")
    if warmup_updates and update <= warmup_updates:
        return update / warmup_updates
    if schedule_updates == warmup_updates:
        return 1.0
    return 0.5 * (
        1.0
        + math.cos(
            math.pi
            * (update - warmup_updates)
            / (schedule_updates - warmup_updates)
        )
    )


@torch.no_grad()
def evaluate(model, val_frames, val_actions, donors, sentinel, batch=64):
    """Registered control panel, computed on packed val frames."""
    model.eval()
    acc = {k: [] for k in ("real", "persistence", "current_only",
                           "wrong_history", "wrong_action")}
    pred_c, tgt_c, elig_c = [], [], []
    for start in range(0, len(sentinel), batch):
        idx = sentinel[start:start + batch]
        rows = torch.tensor(idx, device=val_frames.device)
        packed = to_float(val_frames[rows])
        context, future = packed[:, :3], packed[:, 3]
        actions = val_actions[rows]
        tgt_idx, _ = metrics.batched_mask_indices("val", idx,
                                                  device=val_frames.device)
        real = evaluation._predict_future(model, context, actions, tgt_idx)
        target = evaluation._target_tokens(model, future, tgt_idx)
        persistence = evaluation._target_tokens(model, context[:, 2], tgt_idx)
        current_only = evaluation._predict_current_only(
            model, context[:, 2], actions[:, 2], tgt_idx)

        donor_rows = torch.tensor([donors[i] for i in idx],
                                  device=val_frames.device)
        donor_packed = to_float(val_frames[donor_rows])
        wh_context = torch.cat((donor_packed[:, :2], context[:, 2:3]), dim=1)
        wh_actions = torch.cat((val_actions[donor_rows][:, :2],
                                actions[:, 2:3]), dim=1)
        wrong_history = evaluation._predict_future(
            model, wh_context, wh_actions, tgt_idx)
        wrong_actions, eligible = build_wrong_action_control(actions)
        wrong_action = evaluation._predict_future(
            model, context, wrong_actions, tgt_idx)

        for name, pred in (("real", real.prediction),
                           ("persistence", persistence),
                           ("current_only", current_only.prediction),
                           ("wrong_history", wrong_history.prediction),
                           ("wrong_action", wrong_action.prediction)):
            acc[name].append(evaluation._energy(pred, target).cpu())
        pred_c.append(real.prediction.cpu())
        tgt_c.append(target.cpu())
        elig_c.append(eligible.cpu())
    model.train()

    energy = {k: torch.cat(v) for k, v in acc.items()}
    elig = torch.cat(elig_c)
    p_er, p_var = effective_rank(torch.cat(pred_c))
    t_er, t_var = effective_rank(torch.cat(tgt_c))
    real_mean = float(energy["real"].mean())
    out = {
        "loss_energy_real": real_mean,
        "persistence_ratio": real_mean / float(energy["persistence"].mean()),
        "current_only_ratio": real_mean / float(energy["current_only"].mean()),
        "wrong_history_ratio": real_mean / float(energy["wrong_history"].mean()),
        "wrong_action_ratio": (
            float(energy["real"][elig].mean())
            / float(energy["wrong_action"][elig].mean())) if bool(elig.any()) else 1.0,
        "prediction_effective_rank": p_er,
        "prediction_variance": p_var,
        "target_effective_rank": t_er,
        "prediction_to_target_rank_ratio": p_er / t_er if t_er else 0.0,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--microbatch", type=int, default=32)
    ap.add_argument("--lr-scale", type=float, default=4.0)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument(
        "--schedule-updates",
        type=int,
        help=(
            "cosine-schedule endpoint; defaults to --updates and must be at "
            "least the authorized run length"
        ),
    )
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="rung1")
    ap.add_argument("--pack-root", default=str(PACK_ROOT))
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = ap.parse_args()

    pack_root, output_root = _validate_main_args(args)
    out_dir = output_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = out_dir / "checkpoints"
    measurements_dir = out_dir / "measurements"
    checkpoints_dir.mkdir()
    measurements_dir.mkdir()

    device = torch.device(args.device)
    seed = 20260731
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    predecessor_binding = _input_binding(PREDECESSOR)
    if (
        predecessor_binding["byte_count"] != PREDECESSOR_BYTE_COUNT
        or predecessor_binding["sha256"] != PREDECESSOR_SHA256
    ):
        raise ValueError("predecessor checkpoint disagrees with its frozen binding")
    ckpt = torch.load(PREDECESSOR, map_location="cpu", weights_only=True)
    if _input_binding(PREDECESSOR) != predecessor_binding:
        raise RuntimeError("predecessor checkpoint changed while it was loaded")
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model_state_dict"), dict):
        raise ValueError("predecessor checkpoint schema changed")
    state = {k: v.detach() for k, v in ckpt["model_state_dict"].items()}
    model = model_module.RGBRecurrentPatchMemoryTemporalJepaV1(state).to(device)
    partition = training.partition_parameters_v1(model)
    optimizer = training.build_optimizer_v1(model)
    base_lrs = [g["lr"] * args.lr_scale for g in optimizer.param_groups]

    train_frames, train_actions, train_binding = load_pack(
        pack_root, "train", device
    )
    val_frames, val_actions, val_binding = load_pack(pack_root, "val", device)
    pack_bindings = {"train": train_binding, "val": val_binding}
    source_bindings = [_source_binding(path) for path in training_source_paths()]
    provenance = {
        "seed": seed,
        "device": device_receipt(args.device, device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "predecessor": predecessor_binding,
        "packs": pack_bindings,
        "sources": source_bindings,
        "action_contract": {
            "action_count": ACTION_COUNT,
            "hold_action_index": HOLD_ACTION,
            "wrong_action_rule": "(factual_action + 1) mod action_count",
            "wrong_action_eligibility": "factual_action != hold_action_index",
        },
    }

    val_rows, _ = h6.load_bound_index(REPO_ROOT, role="val")
    donors = list(metrics.build_wrong_history_donor_indices(
        evaluation._metadata_rows(val_rows)))
    sentinel = list(metrics.build_sentinel_indices(
        evaluation._metadata_rows(val_rows)))
    n_train = train_frames.shape[0]
    print(f"train rows {n_train}, val rows {val_frames.shape[0]}, "
          f"sentinel {len(sentinel)}", flush=True)
    if args.batch > n_train:
        raise ValueError("--batch cannot exceed the bound training row count")

    records, started = [], time.time()
    order, cursor = torch.randperm(n_train, device=device), 0

    panel = evaluate(model, val_frames, val_actions, donors, sentinel)
    _assert_source_bindings_unchanged(source_bindings)
    panel.update({"update": 0, "train_loss": None, "elapsed_s": 0.0})
    initial_checkpoint = _save_immutable_checkpoint(
        checkpoints_dir / "update_000000.pt",
        _snapshot_payload(
            model=model,
            update=0,
            args=args,
            pack_bindings=pack_bindings,
            predecessor_binding=predecessor_binding,
            source_bindings=source_bindings,
        ),
    )
    panel["snapshot"] = initial_checkpoint
    records.append(panel)
    best = {
        "persistence_ratio": panel["persistence_ratio"],
        "wrong_action_ratio": panel["wrong_action_ratio"],
    }
    _write_immutable_json(
        measurements_dir / "update_000000.json",
        {
            "schema": f"{TRACE_SCHEMA}_measurement_v1",
            "status": "INTERMEDIATE",
            "citable_as_scientific_evidence": False,
            "authorizes_retry_or_resume": False,
            "config": vars(args),
            "provenance": provenance,
            "record": panel,
        },
    )
    print(json.dumps({k: round(v, 4) for k, v in panel.items()
                      if isinstance(v, float)}), flush=True)

    for update in range(1, args.updates + 1):
        frac = learning_rate_fraction(
            update,
            warmup_updates=args.warmup,
            schedule_updates=args.schedule_updates,
        )
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * frac

        fresh_order = (
            torch.randperm(n_train, device=device)
            if args.batch > n_train - cursor
            else None
        )
        batch_rows, order, cursor = carry_permutation_tail(
            order, cursor, args.batch, fresh_order=fresh_order
        )

        optimizer.zero_grad(set_to_none=True)
        total, n_micro = 0.0, 0
        for s in range(0, args.batch, args.microbatch):
            rows = batch_rows[s:s + args.microbatch]
            packed = to_float(train_frames[rows])
            context, future = packed[:, :3], packed[:, 3]
            actions = train_actions[rows]
            tgt_idx, _ = metrics.batched_mask_indices(
                "train", rows.cpu().tolist(), device=device)
            output = model(context, actions, future, tgt_idx)
            prediction = output.prediction.normalized_predicted_target_tokens
            target = output.target.normalized_target_tokens
            registered = 0.5 * (prediction - target).square().sum(-1).mean()
            loss = output.loss
            assert torch.allclose(loss, registered, rtol=1e-6, atol=1e-7), \
                "sole future JEPA objective changed"
            (loss * (rows.numel() / args.batch)).backward()
            total += float(loss.detach()) * rows.numel()
            n_micro += rows.numel()
        torch.nn.utils.clip_grad_norm_(partition.online, 1.0)
        optimizer.step()
        model.update_target_ema()
        train_loss = total / n_micro

        if update % args.eval_every == 0 or update == args.updates:
            panel = evaluate(model, val_frames, val_actions, donors, sentinel)
            _assert_source_bindings_unchanged(source_bindings)
            panel.update({"update": update, "train_loss": train_loss,
                          "lr_frac": frac,
                          "elapsed_s": round(time.time() - started, 1)})
            records.append(panel)
            for k in ("persistence_ratio", "wrong_action_ratio"):
                best[k] = min(best[k], panel[k])
            checkpoint_binding = _save_immutable_checkpoint(
                checkpoints_dir / f"update_{update:06d}.pt",
                _snapshot_payload(
                    model=model,
                    update=update,
                    args=args,
                    pack_bindings=pack_bindings,
                    predecessor_binding=predecessor_binding,
                    source_bindings=source_bindings,
                ),
            )
            panel["snapshot"] = checkpoint_binding
            _write_immutable_json(
                measurements_dir / f"update_{update:06d}.json",
                {
                    "schema": f"{TRACE_SCHEMA}_measurement_v1",
                    "status": (
                        "COMPLETE" if update == args.updates else "INTERMEDIATE"
                    ),
                    "citable_as_scientific_evidence": False,
                    "authorizes_retry_or_resume": False,
                    "config": vars(args),
                    "provenance": provenance,
                    "record": panel,
                },
            )
            print(json.dumps({
                "u": update, "loss": round(train_loss, 4),
                "persist": round(panel["persistence_ratio"], 4),
                "wrong_act": round(panel["wrong_action_ratio"], 5),
                "wrong_hist": round(panel["wrong_history_ratio"], 4),
                "rank_ratio": round(panel["prediction_to_target_rank_ratio"], 4),
                "min": round(panel["elapsed_s"] / 60, 1)}), flush=True)

    _assert_source_bindings_unchanged(source_bindings)
    _assert_pack_bindings_unchanged(pack_root, pack_bindings)
    _write_immutable_json(
        out_dir / "final_trace.json",
        {
            "schema": TRACE_SCHEMA,
            "status": "COMPLETE",
            "citable_as_scientific_evidence": False,
            "authorizes_retry_or_resume": False,
            "config": vars(args),
            "provenance": provenance,
            "best": best,
            "records": records,
            "final_snapshot": records[-1]["snapshot"],
        },
    )

    print(f"done: {json.dumps(best)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
