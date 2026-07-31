#!/usr/bin/env python3
"""DEVELOPMENT-TIER: spatial-retention + composability diagnostic.

Closes two gaps that the scaled trainer's own panel does not cover, and that
decide whether a good one-step result is actually a usable world model:

1. **Spatial retention.** The temporal objective could improve future prediction
   while destroying the place code it was initialized from -- the pattern
   already recorded for spatial-grid V4 and scene-local V5. The registered
   predecessor panel (`evaluate_predecessor_retention_panel_v1`) was scheduled
   at temporal updates 0/200/400 and never ran, because V1 terminated at 50.
   This runs it against a saved dev checkpoint.

2. **Structural composability.** This script does not run or score a rollout.
   It records whether the predictor output satisfies its own input contract and
   whether the bound H6 slice provides a multi-step scoring horizon.

Reads dev checkpoints; writes only under `.generated/dev/`. Not citable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

model_module = importlib.import_module(
    "lewm.models.rgb_recurrent_patch_memory_temporal_jepa_v1")
evaluation = importlib.import_module(
    "scripts.evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1")
metrics = importlib.import_module(
    "lewm.benchmarks.go2_rgb_recurrent_patch_memory_temporal_jepa_v1")
h6 = importlib.import_module(
    "lewm.datasets.go2_explicit_plan_discounted_successor_state_v27")
trainer = importlib.import_module("scripts.dev_train_temporal_jepa_scaled")

PREDECESSOR = (REPO_ROOT
               / ".generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
               / "attempt_v1/snapshots/update_1000.pt")
PREDECESSOR_BYTE_COUNT = 52_282_877
PREDECESSOR_SHA256 = (
    "f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873"
)
OUTPUT_SCHEMA = "dev_temporal_retention_composability_v4"
DEV_OUTPUT_ROOT = (REPO_ROOT / ".generated/dev").resolve()


def retention_source_paths() -> tuple[Path, ...]:
    """Return every repository source module that directly computes the panel."""

    spatial = evaluation.spatial_evaluation
    ordered = (
        Path(__file__),
        *trainer.training_source_paths(),
        Path(spatial.__file__),
        Path(spatial.metrics.__file__),
        Path(spatial.place_data.__file__),
    )
    return tuple(dict.fromkeys(path.resolve() for path in ordered))


def device_receipt(requested: str, resolved: torch.device) -> dict[str, str | None]:
    """Record both logical torch selection and the physical visibility selector."""

    return {
        "requested": str(requested),
        "resolved": str(resolved),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    }


def sha256_file(path: Path) -> str:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"input is not a regular non-symlink file: {selected}")
    digest = hashlib.sha256()
    with selected.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"input is not a regular non-symlink file: {selected}")
    try:
        reported = selected.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        reported = str(selected.resolve())
    return {
        "path": reported,
        "byte_count": selected.stat().st_size,
        "sha256": sha256_file(selected),
    }


def source_binding(path: Path) -> dict:
    selected = Path(path)
    return {
        "path": selected.resolve().relative_to(REPO_ROOT).as_posix(),
        "byte_count": selected.stat().st_size,
        "sha256": sha256_file(selected),
    }


def assert_source_bindings_unchanged(bindings: list[dict]) -> None:
    for expected in bindings:
        path = REPO_ROOT / expected["path"]
        if source_binding(path) != expected:
            raise RuntimeError(f"source changed during evaluation: {expected['path']}")


def write_immutable_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    if (
        path.exists()
        or path.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError(f"refusing to overwrite evaluation output: {path}")
    with temporary.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, path)
    temporary.unlink()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint")
    selection.add_argument("--migrated-predecessor-baseline", action="store_true")
    parser.add_argument("--expected-update", type=int)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-rollout", action="store_true",
        help="omit even the source-level structural composability report",
    )
    parser.add_argument("--out")
    return parser


def validate_selection(args) -> tuple[Path | None, int]:
    if args.checkpoint:
        if type(args.expected_update) is not int or args.expected_update < 0:
            raise ValueError("--checkpoint requires a non-negative --expected-update")
        digest = args.expected_checkpoint_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "--checkpoint requires --expected-checkpoint-sha256 as lowercase hex"
            )
        return Path(args.checkpoint), args.expected_update
    if args.expected_update is not None or args.expected_checkpoint_sha256 is not None:
        raise ValueError(
            "checkpoint expectations cannot accompany the migrated baseline"
        )
    return None, 0


def require_development_checkpoint(path: Path) -> Path:
    selected = Path(path)
    if selected.is_symlink():
        raise ValueError(
            f"development checkpoint must be a non-symlink file: {selected}"
        )
    resolved = selected.resolve()
    if not resolved.is_relative_to(DEV_OUTPUT_ROOT):
        raise ValueError(
            f"development checkpoint must remain under {DEV_OUTPUT_ROOT}"
        )
    return resolved


def build_model(
    checkpoint: Path | None,
    expected_update: int,
    expected_checkpoint_sha256: str | None,
    device: torch.device,
):
    predecessor_binding = file_binding(PREDECESSOR)
    if (
        predecessor_binding["byte_count"] != PREDECESSOR_BYTE_COUNT
        or predecessor_binding["sha256"] != PREDECESSOR_SHA256
    ):
        raise ValueError("migrated predecessor disagrees with its frozen binding")
    base = torch.load(PREDECESSOR, map_location="cpu", weights_only=True)
    if file_binding(PREDECESSOR) != predecessor_binding:
        raise RuntimeError("migrated predecessor changed while it was loaded")
    if not isinstance(base, dict) or not isinstance(base.get("model_state_dict"), dict):
        raise ValueError("migrated predecessor checkpoint schema changed")
    state = {k: v.detach() for k, v in base["model_state_dict"].items()}
    model = model_module.RGBRecurrentPatchMemoryTemporalJepaV1(state)
    if checkpoint is None:
        identity = {
            "kind": "migrated_predecessor_initialization",
            "selected_model_update": 0,
            "migrated_predecessor": predecessor_binding,
            "scaled_snapshot": None,
        }
        label = f"migrated_predecessor_init_{predecessor_binding['sha256'][:12]}"
        return model.to(device).eval(), label, identity

    checkpoint_binding = file_binding(checkpoint)
    if checkpoint_binding["sha256"] != expected_checkpoint_sha256:
        raise ValueError("selected snapshot SHA-256 disagrees with the expectation")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if file_binding(checkpoint) != checkpoint_binding:
        raise RuntimeError("selected snapshot changed while it was loaded")
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != trainer.SNAPSHOT_SCHEMA
        or payload.get("citable_as_scientific_evidence") is not False
        or payload.get("authorizes_retry_or_resume") is not False
        or type(payload.get("update")) is not int
        or payload["update"] != expected_update
        or not isinstance(payload.get("model_state_dict"), dict)
        or not isinstance(payload.get("pack_bindings"), dict)
        or not isinstance(payload.get("source_bindings"), list)
        or not isinstance(payload.get("config"), dict)
        or payload.get("predecessor_binding") != predecessor_binding
    ):
        raise ValueError("selected scaled snapshot schema or provenance changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    identity = {
        "kind": "scaled_temporal_snapshot",
        "selected_model_update": expected_update,
        "migrated_predecessor": predecessor_binding,
        "scaled_snapshot": checkpoint_binding,
        "snapshot_declared_pack_bindings": payload["pack_bindings"],
        "snapshot_declared_source_bindings": payload["source_bindings"],
        "snapshot_declarations_independently_revalidated_here": False,
        "snapshot_config": payload.get("config"),
    }
    label = f"scaled_update_{expected_update:06d}_{checkpoint_binding['sha256'][:12]}"
    return model.to(device).eval(), label, identity


@torch.no_grad()
def spatial_retention(model, device, selected_model_update: int) -> dict:
    """Run the frozen spatial evaluator with its accepted observation token.

    The adapter's temporal-update argument is only an observation token and is
    restricted to the original V1 schedule. The selected dev snapshot identity
    is therefore reported separately and never replaced by that token.
    """
    adapter = evaluation.evaluate_predecessor_retention_panel_v1(
        model, REPO_ROOT, 0, device
    )
    receipt = {key: value for key, value in adapter.items() if key != "evaluation"}
    return {
        "selected_model_update": selected_model_update,
        "adapter_temporal_update_token": 0,
        "adapter_token_is_not_selected_model_update": selected_model_update != 0,
        "adapter_receipt": receipt,
        "evaluation": adapter.get("evaluation", {}),
    }


@torch.no_grad()
def structural_composability(model) -> dict:
    """Report direct predictor-output-to-input plug compatibility.

    Free-running rollout requires prediction output to be re-usable as
    predictor input. In this architecture it is not:

    - `predict_from_encoded_history` requires `(B, S, 256, 192)` -- the full
      spatial token lattice enforced by `_validate_encoded_history`;
    - the prediction is `(B, 64, 192)` -- only the masked target subset
      selected by `target_indices`.

    Output space != input space, so direct re-feeding is unavailable. This does
    not rule out a separately specified mask-completion, lattice-assembly, or
    adapter mechanism; none is evaluated here. There is also no K-step ground
    truth available: the H6 contract exposes `rgb[0:4]` and marks positions
    4/5/6 forbidden, so such a mechanism could not be scored beyond one step on
    this slice.

    This is a structural property of the configuration, not a measured rollout.
    """
    return {
        "direct_output_to_input_plug_compatible": False,
        "overall_composability": "UNDETERMINED",
        "adapter_or_completion_path_evaluated": False,
        "predictor_input_tokens": int(model.config.spatial_token_count),
        "prediction_output_tokens": int(metrics.TARGET_TOKEN_COUNT),
        "reason": ("prediction emits the 64 masked target tokens; "
                   "predict_from_encoded_history requires all 256 spatial "
                   "tokens, so predictions cannot be re-fed as history"),
        "ground_truth_horizon_available": 1,
        "ground_truth_reason": "H6 exposes rgb[0:4]; positions 4/5/6 forbidden",
        "diagnostic_kind": "source_and_shape_contract_only",
        "rollout_was_executed": False,
        "implication": (
            "the masked prediction cannot be directly re-fed; broader "
            "composability requires a separately specified mechanism and test"
        ),
    }


def main() -> int:
    args = build_argument_parser().parse_args()
    checkpoint, selected_update = validate_selection(args)
    if checkpoint is not None:
        checkpoint = require_development_checkpoint(checkpoint)

    device = torch.device(args.device)
    model, label, identity = build_model(
        checkpoint,
        selected_update,
        args.expected_checkpoint_sha256,
        device,
    )
    print(f"evaluating {label}", flush=True)

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = (
            REPO_ROOT
            / ".generated/dev/temporal_jepa_scaled/evaluations"
            / f"retention_{label}.json"
        ).resolve()
    try:
        out_path.relative_to(DEV_OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError("--out must stay under .generated/dev") from exc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_name(out_path.name + ".partial")
    if (
        out_path.exists()
        or out_path.is_symlink()
        or partial_path.exists()
        or partial_path.is_symlink()
    ):
        raise FileExistsError(f"refusing to reuse evaluation output: {out_path}")

    sources = [source_binding(path) for path in retention_source_paths()]
    report = {
        "schema": OUTPUT_SCHEMA,
        "status": "COMPLETE",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "device": device_receipt(args.device, device),
        "label": label,
        "model_identity": identity,
        "source_bindings": sources,
    }

    retention = spatial_retention(model, device, selected_update)
    ev = retention["evaluation"]
    report["spatial_retention"] = retention
    flat = {k: v for k, v in ev.items() if isinstance(v, (int, float))}
    print("spatial retention:", json.dumps(flat)[:600], flush=True)

    if args.skip_rollout:
        report["structural_composability"] = {
            "status": "SKIPPED_BY_REQUEST",
            "rollout_was_executed": False,
        }
    else:
        report["structural_composability"] = structural_composability(model)
    print(
        "structural composability:",
        json.dumps(report["structural_composability"]),
        flush=True,
    )

    assert_source_bindings_unchanged(sources)
    write_immutable_json(out_path, report)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
