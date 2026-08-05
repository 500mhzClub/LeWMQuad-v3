#!/usr/bin/env python3
"""Run the frozen token-width-32 categorical-radial N32 V3 diagnostic."""
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

from lewm.benchmarks.go2_categorical_radial_n32 import (  # noqa: E402
    HOLDOUT_PANELS,
    categorical_holdout_checks,
    terminal_fit_gate_summary,
)
from lewm.benchmarks.go2_categorical_radial_n32_v3 import (  # noqa: E402
    EXECUTION_BINDING_SHA256,
    RESULT_SCHEMA,
    SMOKE_RESULT_SCHEMA,
    STAGE_NAME,
    STAGE_SCHEMA,
    per_seed_decision,
)
from lewm.models.categorical_radial_perception_full_ray_token32 import (  # noqa: E402
    REGISTERED_CONTEXT_DIM,
    REGISTERED_PARAMETER_COUNT,
    REGISTERED_SHAPE_CHANGED_STATE_KEYS,
    REGISTERED_TOKEN_FEATURE_DIM,
    CategoricalRadialPerceptionFullRayToken32,
    build_comparable_width24_and_token32_models,
)
from scripts import run_go2_categorical_radial_n32_v2 as v2  # noqa: E402


PANEL_PATH = v2.PANEL_PATH
PANEL_FILE_SHA256 = v2.PANEL_FILE_SHA256
PANEL_CONTENT_SHA256 = v2.PANEL_CONTENT_SHA256
PANEL_ROWS_SHA256 = v2.PANEL_ROWS_SHA256
LADDER_PATH = v2.LADDER_PATH
LADDER_FILE_SHA256 = v2.LADDER_FILE_SHA256
LADDER_CONTENT_SHA256 = v2.LADDER_CONTENT_SHA256
V3_RESULT_PATH = v2.V3_RESULT_PATH
V3_RESULT_FILE_SHA256 = v2.V3_RESULT_FILE_SHA256
V3_RESULT_CONTENT_SHA256 = v2.V3_RESULT_CONTENT_SHA256
PATCH7_RESULT_PATH = v2.PATCH7_RESULT_PATH
PATCH7_RESULT_FILE_SHA256 = v2.PATCH7_RESULT_FILE_SHA256
PATCH7_RESULT_CONTENT_SHA256 = v2.PATCH7_RESULT_CONTENT_SHA256
PROTOCOL_PATH = v2.PROTOCOL_PATH
PROTOCOL_SHA256 = v2.PROTOCOL_SHA256
V1_CONTRACT_PATH = v2.V1_CONTRACT_PATH
V1_CONTRACT_SHA256 = v2.V1_CONTRACT_SHA256
V1_RESULT_PATH = v2.V1_RESULT_PATH
V1_RESULT_FILE_SHA256 = v2.V1_RESULT_FILE_SHA256
V1_RESULT_CONTENT_SHA256 = v2.V1_RESULT_CONTENT_SHA256
V1_RESULT_NOTE_PATH = v2.V1_RESULT_NOTE_PATH
V1_RESULT_NOTE_SHA256 = v2.V1_RESULT_NOTE_SHA256
V2_CONTRACT_PATH = v2.CONTRACT_PATH
V2_CONTRACT_SHA256 = v2.EXECUTION_BINDING_SHA256
V2_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_n32/v2/seed_20260710_result.json"
)
V2_RESULT_FILE_SHA256 = (
    "0a5f8a822d7fec8287a30103125fca1a4927f0413e2f0906db431cef54ec2265"
)
V2_RESULT_CONTENT_SHA256 = (
    "e070cc96d69b76e1f85f533fa1d94221225963a2b66a491f0c2a867c008b97ef"
)
V2_RESULT_NOTE_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_categorical_radial_n32_v2_result_2026-07-11.md"
)
V2_RESULT_NOTE_SHA256 = (
    "d5e5748db8177d925990b5c31e23c45d43e16c62e0aac4f389ab47b1fa6547e0"
)
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_n32_v3_token_width_binding_2026-07-11.md"
)

EXPECTED_INITIAL_STATE_SHA256 = {
    20260710: "ddb8f6dbfa54a7445c2b4363d9978b0a99a86e6d88a28f480840c5d8d128804b",
    20260711: "fa9601fb5f658b640c43b50c28587c5129c6f42f8fd4fb09866983130e4954ee",
}
EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256 = dict(v2.EXPECTED_INITIAL_STATE_SHA256)
EXPECTED_PANEL_ARTIFACT_COUNTS = v2.EXPECTED_PANEL_ARTIFACT_COUNTS
AUTHORITATIVE_CONFIG = dict(v2.AUTHORITATIVE_CONFIG)
SMOKE_CONFIG = dict(v2.SMOKE_CONFIG)
AUTHORITATIVE_EVALUATION_INTERVAL = v2.AUTHORITATIVE_EVALUATION_INTERVAL
SMOKE_EVALUATION_INTERVAL = v2.SMOKE_EVALUATION_INTERVAL
BATCH_SIZE = v2.BATCH_SIZE
EVALUATION_TARGET_BATCH_SIZE = v2.EVALUATION_TARGET_BATCH_SIZE
GRADIENT_CLIP = v2.GRADIENT_CLIP
EVENT_FIELDS = v2.EVENT_FIELDS
DATA_EVENT_FIELDS = v2.DATA_EVENT_FIELDS

PanelFrameDataset = v2.PanelFrameDataset
_canonical_panel_records = v2._canonical_panel_records
_artifact_contract = v2._artifact_contract
_verify_artifacts = v2._verify_artifacts
evaluate_panel = v2.evaluate_panel
direct_hierarchical_loss = v2.direct_hierarchical_loss


def _sha256_file(path: Path) -> str:
    return v2._sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    return v2._read_json(path)


def _source_paths() -> dict[str, Path]:
    shared = {
        ("v2_n32_runner" if name == "runner" else name): path
        for name, path in v2._source_paths().items()
    }
    return {
        **shared,
        "n32_v3_binding": CONTRACT_PATH,
        "n32_v3_pure": (
            REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32_v3.py"
        ),
        "n32_v3_token32_model": (
            REPOSITORY_ROOT
            / "lewm/models/categorical_radial_perception_full_ray_token32.py"
        ),
        "n32_v3_model_test": (
            REPOSITORY_ROOT
            / "lewm/tests/test_categorical_radial_perception_full_ray_token32.py"
        ),
        "n32_v3_pure_test": (
            REPOSITORY_ROOT / "lewm/tests/test_go2_categorical_radial_n32_v3.py"
        ),
        "n32_v3_runner_test": (
            REPOSITORY_ROOT / "lewm/tests/test_run_go2_categorical_radial_n32_v3.py"
        ),
        "n32_v3_finalizer_test": (
            REPOSITORY_ROOT
            / "lewm/tests/test_finalize_go2_categorical_radial_n32_v3.py"
        ),
        "v3_finalizer": (
            REPOSITORY_ROOT / "scripts/finalize_go2_categorical_radial_n32_v3.py"
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
    v2._validate_content(
        payload,
        schema=schema,
        content_sha256=content_sha256,
        name=name,
    )


def _load_bound_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate V3 evidence before deserializing the failed V2 result."""

    expected = {
        CONTRACT_PATH: EXECUTION_BINDING_SHA256,
        V2_RESULT_PATH: V2_RESULT_FILE_SHA256,
        V2_RESULT_NOTE_PATH: V2_RESULT_NOTE_SHA256,
    }
    pre_hashes = {path: _sha256_file(path) for path in expected}
    if pre_hashes != expected:
        raise ValueError("bound N32 V3 evidence SHA-256 mismatch")
    panel, panels, ladder, ladder_v3_result, reference, v1_result = (
        v2._load_bound_inputs()
    )
    from scripts import finalize_go2_categorical_radial_n32_v2 as v2_finalizer

    v2_bound = v2_finalizer._load_bound_evidence()
    v2_result = _read_json(V2_RESULT_PATH)
    validated = v2_finalizer._validate_authoritative_result(
        v2_result,
        expected_seed=20260710,
        expected_controls=v2_bound["controls"][20260710],
        expected_patch7_reference=v2_bound["patch7_reference"],
    )
    if (
        v2_result.get("content_sha256") != V2_RESULT_CONTENT_SHA256
        or validated["decision"].get("classification") != "fit_gate_failed"
        or validated["decision"].get("favorable") is not False
        or validated["decision"].get(
            "exposure_matched_v3_cosine_fit_passes"
        ) is not False
    ):
        raise ValueError("bound failed N32 V2 result identity drift")
    post_hashes = {path: _sha256_file(path) for path in expected}
    if post_hashes != pre_hashes:
        raise RuntimeError("bound N32 V3 evidence changed during parsing")
    return (
        panel,
        panels,
        ladder,
        ladder_v3_result,
        reference,
        v1_result,
        v2_result,
    )


def cosine_learning_rate(update: int, total_updates: int) -> float:
    return v2.cosine_learning_rate(update, total_updates)


def learning_rate_schedule_contract(total_updates: int) -> dict[str, Any]:
    return v2.learning_rate_schedule_contract(total_updates)


def frozen_minibatch_schedule(
    frame_count: int,
    updates: int,
    seed: int,
) -> list[list[int]]:
    return v2.frozen_minibatch_schedule(frame_count, updates, seed)


def _normalized_events(value: Mapping[str, int]) -> dict[str, int]:
    return {name: int(value.get(name, 0)) for name in EVENT_FIELDS}


def _normalized_data_events(value: Mapping[str, int]) -> dict[str, int]:
    return {name: int(value.get(name, 0)) for name in DATA_EVENT_FIELDS}


def _tensor_shape(value: torch.Tensor) -> list[int]:
    return [int(dimension) for dimension in value.shape]


def _build_comparable_initialization(
    *, seed: int, cpu_rng_state: torch.Tensor
) -> tuple[dict[str, torch.Tensor], str, dict[str, Any]]:
    """Build and verify the registered width-24/width-32 initialization pair."""

    reference, candidate = build_comparable_width24_and_token32_models(cpu_rng_state)
    reference_state = v2.v1.v3.v2.v1._clone_state(reference.state_dict())
    candidate_state = v2.v1.v3.v2.v1._clone_state(candidate.state_dict())
    reference_hash = v2.v1.v3.v2.v1._state_dict_sha256(reference_state)
    candidate_hash = v2.v1.v3.v2.v1._state_dict_sha256(candidate_state)
    changed = sorted(
        name
        for name in reference_state
        if tuple(reference_state[name].shape) != tuple(candidate_state[name].shape)
    )
    registered = sorted(REGISTERED_SHAPE_CHANGED_STATE_KEYS)
    if set(reference_state) != set(candidate_state) or changed != registered:
        raise RuntimeError("N32 V3 initialization state-key/shape contract changed")
    same_shape = sorted(set(reference_state) - set(changed))
    if any(
        reference_state[name].dtype != candidate_state[name].dtype
        or not torch.equal(reference_state[name], candidate_state[name])
        for name in same_shape
    ):
        raise RuntimeError("N32 V3 same-shape initialization is not bit-identical")
    if reference_hash != EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256[int(seed)]:
        raise RuntimeError("N32 V3 width-24 reference initialization changed")
    if candidate_hash != EXPECTED_INITIAL_STATE_SHA256[int(seed)]:
        raise RuntimeError("N32 V3 registered initialization changed")
    proof = {
        "construction": (
            "save_cpu_rng_after_determinism_then_replay_for_width24_and_width32_"
            "and_copy_every_same_shape_entry"
        ),
        "v2_reference_initial_state_sha256": reference_hash,
        "candidate_initial_state_sha256": candidate_hash,
        "v2_reference_parameter_count": sum(
            int(parameter.numel()) for parameter in reference.parameters()
        ),
        "candidate_parameter_count": sum(
            int(parameter.numel()) for parameter in candidate.parameters()
        ),
        "state_key_sets_identical": True,
        "same_shape_entry_count": len(same_shape),
        "same_shape_entries_bit_identical": True,
        "only_shape_changed_state_keys": changed,
        "shape_changes": {
            name: {
                "v2_shape": _tensor_shape(reference_state[name]),
                "v3_shape": _tensor_shape(candidate_state[name]),
                "v2_dtype": str(reference_state[name].dtype),
                "v3_dtype": str(candidate_state[name].dtype),
            }
            for name in changed
        },
        "trained_v2_weight_loaded": False,
    }
    del reference, candidate
    return candidate_state, candidate_hash, proof


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
    model = CategoricalRadialPerceptionFullRayToken32().to(device)
    model.load_state_dict(initial_state, strict=True)
    if v2.v1.v3.v2.v1._state_dict_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("N32 V3 did not start from the registered initial state")
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
            raise RuntimeError("N32 V3 direct training batch was not 80 frames")
        batch = dataset.training_batch(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device=device, dtype=torch.float32)
        if int(images.shape[0]) != BATCH_SIZE:
            raise RuntimeError("N32 V3 training tensor was not a direct batch 80")
        logits = model(images)
        loss = direct_hierarchical_loss(
            logits,
            batch["labels"].to(device),
            batch["mask"].to(device),
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite N32 V3 loss at update {step}")
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
        "final_state_sha256": v2.v1.v3.v2.v1._state_dict_sha256(model.state_dict()),
        "minibatch_indices": schedule,
        "minibatch_indices_sha256": v2.v1.v3.canonical_json_sha256(schedule),
        "learning_curve": curve,
        "terminal_fit_gate": terminal,
        "training_access": training,
        "fit_evaluation_access": evaluation,
        "holdouts_evaluated": False,
    }, model


def _canonical_output(seed: int) -> Path:
    return (
        REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_n32/v3/"
        f"seed_{int(seed)}_result.json"
    ).resolve()


def _validate_primary_authorization(
    path: Path,
    expected_sha256: str,
    current_sources: Mapping[str, Any],
    patch7_reference: Mapping[str, Any],
) -> dict[str, Any]:
    from scripts import finalize_go2_categorical_radial_n32_v3 as finalizer

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
    v2._reconcile_access(fit_dataset, stage, panel_access, holdouts)


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
        parser.error("output already exists; N32 V3 results are immutable")
    if args.seed not in EXPECTED_INITIAL_STATE_SHA256:
        parser.error("N32 V3 seed must be 20260710 or 20260711")
    authorization = (
        args.seed_20260710_result,
        args.expected_seed_20260710_sha256,
    )
    if args.seed == 20260710 and any(value is not None for value in authorization):
        parser.error("seed 20260710 rejects seed-authorization arguments")
    if args.seed == 20260711 and any(value is None for value in authorization):
        parser.error("seed 20260711 requires both seed-authorization arguments")
    if args.non_authoritative_smoke and args.seed != 20260710:
        parser.error("N32 V3 smoke is seed-20260710-only")
    if args.non_authoritative_smoke and args.output.resolve() in {
        _canonical_output(seed) for seed in EXPECTED_INITIAL_STATE_SHA256
    }:
        parser.error("N32 V3 smoke output must not occupy a canonical result path")
    if not args.non_authoritative_smoke and args.output.resolve() != _canonical_output(
        args.seed
    ):
        parser.error("authoritative N32 V3 output must use its canonical path")
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
    panel, panels, ladder, ladder_v3_result, patch7_reference, v1_result, v2_result = (
        _load_bound_inputs()
    )
    git_start = v2.v1.v3.v2.v1._git_snapshot()
    primary = None
    if args.seed == 20260711:
        primary = _validate_primary_authorization(
            args.seed_20260710_result.resolve(),
            str(args.expected_seed_20260710_sha256),
            source_start,
            patch7_reference,
        )

    device = v2.v1.v3.v2.v1._resolve_device(str(args.device))
    determinism = v2.v1.v3.v2.v1._configure_determinism(int(args.seed))
    cpu_rng_state = torch.get_rng_state().clone()
    initial_state, initial_state_sha256, initialization_proof = (
        _build_comparable_initialization(
            seed=int(args.seed), cpu_rng_state=cpu_rng_state
        )
    )
    parameter_count = int(initialization_proof["candidate_parameter_count"])
    if parameter_count != REGISTERED_PARAMETER_COUNT:
        raise RuntimeError("N32 V3 model parameter count changed")

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
    holdouts_authorized = bool(
        not smoke and stage["terminal_fit_gate"]["passes"]
    )
    if holdouts_authorized:
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
    decision = per_seed_decision(
        stage,
        holdout_checks,
        authoritative=not smoke,
    )

    if primary is not None and _sha256_file(args.seed_20260710_result.resolve()) != str(
        args.expected_seed_20260710_sha256
    ):
        raise RuntimeError("seed-20260710 V3 authorization changed during execution")
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
            V2_CONTRACT_PATH: V2_CONTRACT_SHA256,
            V2_RESULT_PATH: V2_RESULT_FILE_SHA256,
            V2_RESULT_NOTE_PATH: V2_RESULT_NOTE_SHA256,
            CONTRACT_PATH: EXECUTION_BINDING_SHA256,
        }.items()
    }
    for path, digest in evidence_hashes.items():
        if _sha256_file(Path(path)) != digest:
            raise RuntimeError(f"bound N32 V3 evidence changed: {path}")
    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("N32 V3 sources changed during execution")
    git_end = v2.v1.v3.v2.v1._git_snapshot()
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
            "cpu_rng_state_captured_immediately_after_determinism": True,
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
            "failed_v2_n32_result": {
                "path": str(V2_RESULT_PATH.resolve()),
                "sha256": V2_RESULT_FILE_SHA256,
                "content_sha256": V2_RESULT_CONTENT_SHA256,
            },
            "failed_v2_n32_result_note": {
                "path": str(V2_RESULT_NOTE_PATH.resolve()),
                "sha256": V2_RESULT_NOTE_SHA256,
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
            "class": "CategoricalRadialPerceptionFullRayToken32",
            "parameter_count": parameter_count,
            "token_feature_dim": REGISTERED_TOKEN_FEATURE_DIM,
            "context_dim": REGISTERED_CONTEXT_DIM,
            "parameter_delta_from_v2": (
                REGISTERED_PARAMETER_COUNT - v2.REGISTERED_PARAMETER_COUNT
            ),
            "initial_state_sha256": initial_state_sha256,
            "initialization_comparability": initialization_proof,
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
        "shared_jepa_full_train_candidate_licensed": False,
        "runtime_ready": False,
        "g2_licensed": False,
        "g3_licensed": False,
    }
    payload = {**core, "content_sha256": v2.v1.v3.canonical_json_sha256(core)}
    v2.v1.v3.v2.v1._atomic_write_json_exclusive(args.output.resolve(), payload)
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
