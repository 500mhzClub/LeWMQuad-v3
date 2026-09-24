#!/usr/bin/env python3
"""DEVELOPMENT-TIER diagnostic: temporal V1 finite rank trajectory.

This is NOT a scientific attempt and produces NO citable qualification.  It does
not touch the consumed attempt root, emits no checkpoint, and opens no held-out
or sealed material.  It exists to answer one measurement question that the
one-shot V1 run could not answer because it stopped at update 50:

  1. How does prediction effective rank evolve after update 50 through the
     remainder of the single reviewed 400-update schedule?
  2. What effective rank does a restricted linear least-squares reference have?
     -> fit current-token features, with an optional action main effect, and
        measure its rank on the second half of the same sentinel panel.
  3. What is the effective rank of the persistence baseline's prediction?
     -> the encoded current frame, which the gate never measured.

(2) is not an oracle, a generalization estimate, or a bound on nonlinear model
capacity. It is only a precisely named diagnostic reference.

Reuses the frozen, reviewed V1 modules unchanged for model, data, schedule,
masks and the training step.  The only deviations from the registered run are
deliberate and dev-tier: the continuation gate is not enforced. The reviewed
4,000-row schedule is used at most once and is never cycled by this code. This
script grants no retry or resume authority; execution authority is external.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

model_module = importlib.import_module(
    "lewm.models.rgb_recurrent_patch_memory_temporal_jepa_v1"
)
training = importlib.import_module(
    "scripts.run_go2_rgb_recurrent_patch_memory_temporal_jepa_v1"
)
evaluation = importlib.import_module(
    "scripts.evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1"
)
metrics = importlib.import_module(
    "lewm.benchmarks.go2_rgb_recurrent_patch_memory_temporal_jepa_v1"
)

PREDECESSOR_CHECKPOINT = (
    REPO_ROOT
    / ".generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
    / "attempt_v1/snapshots/update_1000.pt"
)
PREDECESSOR_BYTE_COUNT = 52_282_877
PREDECESSOR_SHA256 = (
    "f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873"
)
OUTPUT_ROOT = REPO_ROOT / ".generated/dev/temporal_v1_rank_trajectory"
SCHEDULE_UPDATES = 400  # registered one-pass schedule length
OUTPUT_SCHEMA = "dev_temporal_v1_rank_trajectory_probe_v2"


def sha256_file(path: Path) -> str:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"input is not a regular non-symlink file: {selected}")
    digest = hashlib.sha256()
    with selected.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            raise RuntimeError(f"source changed during diagnostic: {expected['path']}")


def input_binding(path: Path) -> dict:
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


def validate_requested_updates(updates: int) -> int:
    maximum = int(training.MAXIMUM_UPDATES_V1)
    if maximum != SCHEDULE_UPDATES:
        raise RuntimeError("reviewed training cap and one-pass schedule diverged")
    if type(updates) is not int or not 1 <= updates <= maximum:
        raise ValueError(f"--updates must be in [1,{maximum}]; cycling is forbidden")
    return updates


def write_immutable_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    if (
        path.exists()
        or path.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError(f"refusing to overwrite diagnostic output: {path}")
    with temporary.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, path)
    temporary.unlink()


def effective_rank_and_variance(tokens: torch.Tensor) -> tuple[float, float]:
    """Position-centered covariance health, identical to the frozen metric.

    Mirrors lewm/benchmarks/...:representation_health so the numbers here are
    directly comparable to the registered update-0/update-50 sentinels.
    """
    value = tokens.detach().to(device="cpu", dtype=torch.float64)
    rows, token_count, dim = map(int, value.shape)
    centered = value - value.mean(dim=0, keepdim=True)
    flat = centered.reshape(-1, dim)
    cov = flat.T.mm(flat) / (rows * token_count - 1)
    cov = 0.5 * (cov + cov.T)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = float(ev.sum())
    if total <= 0.0:
        return 0.0, 0.0
    p = ev / ev.sum()
    erank = float((-(p * p.clamp_min(1e-12).log()).sum()).exp())
    variance = float(centered.square().sum() / (rows * token_count * dim))
    return erank, variance


def restricted_linear_reference_rank(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    action_onehot: torch.Tensor | None = None,
) -> dict[str, float | int | str | bool]:
    """Measure a restricted linear least-squares reference on one panel.

    The design flattens tokens and uses the corresponding current token, an
    intercept, and (when supplied) a reference-coded row-level action main
    effect. It uses the existing row order: the first half is fit and the second
    half is evaluated, without randomization or stratification. This does not
    estimate generalization beyond the panel and is neither an oracle nor a
    capacity bound for the nonlinear temporal model.
    """
    if (
        source.ndim != 3
        or target.ndim != 3
        or source.shape[:2] != target.shape[:2]
        or source.shape[0] < 4
    ):
        raise ValueError(
            "source and target must be compatible (rows,tokens,dim) panels "
            "with at least four rows"
        )
    if action_onehot is not None and (
        action_onehot.ndim != 2
        or action_onehot.shape[0] != source.shape[0]
        or action_onehot.shape[1] < 2
    ):
        raise ValueError(
            "action_onehot must have shape (rows, at_least_two_action_features)"
        )
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("source and target must be finite")
    if action_onehot is not None and (
        not bool(torch.logical_or(action_onehot == 0, action_onehot == 1).all())
        or not bool(action_onehot.sum(dim=1).eq(1).all())
    ):
        raise ValueError("action_onehot must contain exactly one active class per row")
    rows = source.shape[0]
    half = rows // 2
    src = source.to(device="cpu", dtype=torch.float64)
    tgt = target.to(device="cpu", dtype=torch.float64)

    def design(chunk: torch.Tensor, act: torch.Tensor | None) -> torch.Tensor:
        b, t, d = chunk.shape
        flat = chunk.reshape(b * t, d)
        parts = [flat, torch.ones(b * t, 1, dtype=torch.float64)]
        if act is not None:
            parts.append(
                act[:, 1:]
                .to(device="cpu", dtype=torch.float64)
                .unsqueeze(1)
                .expand(b, t, act.shape[-1] - 1)
                .reshape(b * t, -1)
            )
        return torch.cat(parts, dim=1)

    act_fit = None if action_onehot is None else action_onehot[:half]
    act_evl = None if action_onehot is None else action_onehot[half:]
    x_fit = design(src[:half], act_fit)
    y_fit = tgt[:half].reshape(-1, tgt.shape[-1])
    beta = torch.linalg.lstsq(x_fit, y_fit).solution
    x_evl = design(src[half:], act_evl)
    pred = (x_evl @ beta).reshape(rows - half, tgt.shape[1], tgt.shape[-1])
    erank, variance = effective_rank_and_variance(pred)
    held_tgt = tgt[half:]
    resid = float((pred - held_tgt).square().mean())
    return {
        "effective_rank": erank,
        "cross_sample_variance": variance,
        "second_half_panel_mse": resid,
        "fit_rows": half,
        "evaluation_rows": rows - half,
        "split_rule": "ordered_first_half_fit_second_half_evaluation",
        "stratified": False,
    }


@torch.no_grad()
def diagnostic_panel(model, runtime, indices) -> dict:
    """Sentinel panel with persistence and restricted-linear references."""
    was_training = model.training
    model.eval()
    pred_chunks, tgt_chunks, persist_chunks, mem_chunks = [], [], [], []
    action_chunks = []
    energy_real, energy_persist = [], []
    try:
        for start in range(0, len(indices), evaluation.VALIDATION_BATCH_SIZE_V1):
            batch_indices = tuple(
                indices[start : start + evaluation.VALIDATION_BATCH_SIZE_V1]
            )
            controls = runtime.validation_control_batch(batch_indices)
            factual = controls.factual
            target_indices, _ = metrics.batched_mask_indices(
                "val", factual.row_indices, device=runtime.device
            )
            real = evaluation._predict_future(
                model, factual.context_rgb, factual.action_sequence, target_indices
            )
            future_target = evaluation._target_tokens(
                model, factual.target_rgb, target_indices
            )
            current_target = evaluation._target_tokens(
                model, factual.context_rgb[:, 2], target_indices
            )
            pred_chunks.append(real.prediction.detach().cpu())
            mem_chunks.append(real.memory.detach().cpu())
            tgt_chunks.append(future_target.detach().cpu())
            persist_chunks.append(current_target.detach().cpu())
            action_chunks.append(factual.action_sequence[:, 2].detach().cpu())
            energy_real.append(
                evaluation._energy(real.prediction, future_target).cpu()
            )
            energy_persist.append(
                evaluation._energy(current_target, future_target).cpu()
            )
    finally:
        model.train(was_training)

    prediction = torch.cat(pred_chunks)
    target = torch.cat(tgt_chunks)
    persistence = torch.cat(persist_chunks)
    memory = torch.cat(mem_chunks)
    actions = torch.cat(action_chunks)
    onehot = torch.nn.functional.one_hot(
        actions.long(), num_classes=int(evaluation.ACTION_COUNT_V1)
    ).float()

    p_rank, p_var = effective_rank_and_variance(prediction)
    t_rank, t_var = effective_rank_and_variance(target)
    s_rank, s_var = effective_rank_and_variance(persistence)
    m_rank, m_var = effective_rank_and_variance(memory)
    return {
        "prediction": {"effective_rank": p_rank, "cross_sample_variance": p_var},
        "target": {"effective_rank": t_rank, "cross_sample_variance": t_var},
        "persistence": {"effective_rank": s_rank, "cross_sample_variance": s_var},
        "recurrent": {"effective_rank": m_rank, "cross_sample_variance": m_var},
        "prediction_to_target_rank_ratio": p_rank / t_rank if t_rank else 0.0,
        "persistence_to_target_rank_ratio": s_rank / t_rank if t_rank else 0.0,
        "energy_real_mean": float(torch.cat(energy_real).mean()),
        "energy_persistence_mean": float(torch.cat(energy_persist).mean()),
        "persistence_ratio": float(
            torch.cat(energy_real).mean() / torch.cat(energy_persist).mean()
        ),
        "restricted_linear_reference_current_to_future": restricted_linear_reference_rank(
            persistence, target
        ),
        "restricted_linear_reference_current_action_to_future": restricted_linear_reference_rank(
            persistence, target, action_onehot=onehot
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=SCHEDULE_UPDATES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(OUTPUT_ROOT / "rank_trajectory.json"))
    args = parser.parse_args()

    validate_requested_updates(args.updates)
    out_path = Path(args.out).resolve()
    try:
        out_path.relative_to((REPO_ROOT / ".generated/dev").resolve())
    except ValueError as exc:
        raise ValueError("--out must stay under .generated/dev") from exc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        out_path.exists()
        or out_path.is_symlink()
        or out_path.with_name(out_path.name + ".partial").exists()
        or out_path.with_name(out_path.name + ".partial").is_symlink()
    ):
        raise FileExistsError(f"refusing to reuse diagnostic output: {out_path}")

    device = torch.device(args.device)
    seed = 20260731
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    predecessor = input_binding(PREDECESSOR_CHECKPOINT)
    if (
        predecessor["byte_count"] != PREDECESSOR_BYTE_COUNT
        or predecessor["sha256"] != PREDECESSOR_SHA256
    ):
        raise ValueError("predecessor checkpoint disagrees with its frozen binding")
    checkpoint = torch.load(
        PREDECESSOR_CHECKPOINT, map_location="cpu", weights_only=True
    )
    if input_binding(PREDECESSOR_CHECKPOINT) != predecessor:
        raise RuntimeError("predecessor checkpoint changed while it was loaded")
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("model_state_dict"), dict)
    ):
        raise ValueError("predecessor checkpoint schema changed")
    predecessor_state = {
        name: value.detach()
        for name, value in checkpoint["model_state_dict"].items()
    }
    model = model_module.RGBRecurrentPatchMemoryTemporalJepaV1(predecessor_state)
    model = model.to(device)
    optimizer = training.build_optimizer_v1(model)
    sources = [
        source_binding(Path(path))
        for path in (
            __file__, model_module.__file__, training.__file__, evaluation.__file__,
            metrics.__file__,
        )
    ]

    runtime, runtime_preflight_audit = evaluation.open_bound_runtime_v1(
        REPO_ROOT, device=device
    )
    sentinel = list(runtime.sentinel_indices)

    panel_updates = sorted(
        {
            10, 25, 50, 75, 100, 150, 200, 300, 400, args.updates,
        }
        & set(range(1, args.updates + 1))
    )
    records = []
    started = time.time()

    try:
        panel = diagnostic_panel(model, runtime, sentinel)
        assert_source_bindings_unchanged(sources)
        panel.update({"update": 0, "loss": None, "elapsed_s": 0.0})
        records.append(panel)
        print(json.dumps({key: panel[key] for key in (
            "update", "prediction_to_target_rank_ratio", "persistence_ratio"
        )}), flush=True)
    except BaseException:
        runtime.close()
        raise

    runtime_access_audit = None
    try:
        state = training.accounting_for_completed_updates_v1(0)
        for update in range(1, args.updates + 1):
            batches = runtime.train_microbatches_for_update(update)
            context = [batch.context_rgb for batch in batches]
            actions = [batch.action_sequence for batch in batches]
            future = [batch.target_rgb for batch in batches]
            rows = [list(batch.row_indices) for batch in batches]
            expected = [index for batch in batches for index in batch.row_indices]
            result = training.training_update_v1(
                model,
                optimizer,
                context,
                actions,
                future,
                rows,
                expected_row_indices=expected,
                schedule_offset=state.sequence_rows,
                accounting=state,
            )
            state = result.accounting
            if update in panel_updates:
                panel = diagnostic_panel(model, runtime, sentinel)
                assert_source_bindings_unchanged(sources)
                panel.update({
                    "update": update,
                    "loss": float(result.mean_jepa_loss),
                    "elapsed_s": round(time.time() - started, 1),
                })
                records.append(panel)
                print(json.dumps({key: panel[key] for key in (
                    "update", "loss", "prediction_to_target_rank_ratio",
                    "persistence_ratio",
                )}), flush=True)
    finally:
        try:
            runtime_access_audit = runtime.access_audit()
        finally:
            runtime.close()

    assert_source_bindings_unchanged(sources)
    write_immutable_json(
        out_path,
        {
            "schema": OUTPUT_SCHEMA,
            "status": "COMPLETE",
            "citable_as_scientific_evidence": False,
            "authorizes_retry_or_resume": False,
            "config": vars(args),
            "provenance": {
                "seed": seed,
                "torch_version": torch.__version__,
                "predecessor": predecessor,
                "sources": sources,
                "runtime_preflight_audit": runtime_preflight_audit,
                "runtime_access_audit": runtime_access_audit,
                "schedule": {
                    "first_update": 1,
                    "last_update": args.updates,
                    "maximum_updates": SCHEDULE_UPDATES,
                    "cycled": False,
                },
            },
            "registered_gate_context": {
                "threshold": float(metrics.PREDICTION_HEALTH_RETENTION_MINIMUM),
                "diagnostic_references_are_not_oracles": True,
                "no_historical_result_is_reasserted_by_this_probe": True,
            },
            "records": records,
        },
    )
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
