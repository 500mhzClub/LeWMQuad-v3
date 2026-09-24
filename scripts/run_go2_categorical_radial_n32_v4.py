#!/usr/bin/env python3
"""Run the frozen explicit-hierarchy categorical-radial N32 V4 diagnostic."""
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
from lewm.benchmarks.go2_categorical_radial_n32_v4 import (  # noqa: E402
    EXECUTION_BINDING_SHA256,
    FACTOR_OUTPUT_CONTRACT,
    FACTOR_OUTPUT_CONTRACT_SHA256,
    RESULT_SCHEMA,
    SMOKE_RESULT_SCHEMA,
    STAGE_NAME,
    STAGE_SCHEMA,
    per_seed_decision,
)
from lewm.benchmarks.go2_n32_pose_projection_audit import (  # noqa: E402
    RESULT_SCHEMA as POSE_AUDIT_RESULT_SCHEMA,
)
from lewm.models.categorical_radial_perception_full_ray_hierarchical import (  # noqa: E402
    CategoricalRadialPerceptionFullRayHierarchical,
    REGISTERED_CONTEXT_DIM,
    REGISTERED_FACTOR_NAMES,
    REGISTERED_PARAMETER_COUNT,
    REGISTERED_SAME_SHAPE_STATE_ENTRY_COUNT,
    REGISTERED_SHAPE_CHANGED_STATE_KEYS,
    REGISTERED_STATE_ENTRY_COUNT,
    REGISTERED_TOKEN_FEATURE_DIM,
    build_comparable_width24_and_hierarchical_models,
)
from scripts import run_go2_categorical_radial_n32_v3 as v3  # noqa: E402


v2 = v3.v2
_backend = v2.v1.v3.v2.v1
_canonical_json_sha256 = v2.v1.v3.canonical_json_sha256

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
V2_CONTRACT_PATH = v3.V2_CONTRACT_PATH
V2_CONTRACT_SHA256 = v3.V2_CONTRACT_SHA256
V2_RESULT_PATH = v3.V2_RESULT_PATH
V2_RESULT_FILE_SHA256 = v3.V2_RESULT_FILE_SHA256
V2_RESULT_CONTENT_SHA256 = v3.V2_RESULT_CONTENT_SHA256
V2_RESULT_NOTE_PATH = v3.V2_RESULT_NOTE_PATH
V2_RESULT_NOTE_SHA256 = v3.V2_RESULT_NOTE_SHA256
V3_CONTRACT_PATH = v3.CONTRACT_PATH
V3_CONTRACT_SHA256 = v3.EXECUTION_BINDING_SHA256

N32_V3_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_n32/v3/seed_20260710_result.json"
)
N32_V3_RESULT_FILE_SHA256 = (
    "0f3eb212afe54a38d7a81a1fc51ca544dfab667a94a836be742d3ea3e2298d85"
)
N32_V3_RESULT_CONTENT_SHA256 = (
    "ec8dd8450fb34bee3a5ba1c5a5b532339d281241560c8ed9ac07a48d2c2bea4e"
)
N32_V3_RESULT_NOTE_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_categorical_radial_n32_v3_result_2026-07-11.md"
)
N32_V3_RESULT_NOTE_SHA256 = (
    "a346ecb9b909d897f839067e409bccd906f61223cec8e746da6b54c531f44fca"
)
KNOWN_BIAS_PROOF_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_n32_known_bias_impossibility_2026-07-11.md"
)
KNOWN_BIAS_PROOF_SHA256 = (
    "e214bb80bcccf9ae5051231d90f7a5d8c2bfa33ca799e7db3eb969698fa2108a"
)
POSE_AUDIT_RESULT_PATH = (
    REPOSITORY_ROOT / ".generated/go2_n32_pose_projection_audit/v1/result.json"
)
POSE_AUDIT_RESULT_FILE_SHA256 = (
    "2c7efba897054ea0067db58f020e70dc5f3c5804785c74cbda4a8b76e0210b9d"
)
POSE_AUDIT_RESULT_CONTENT_SHA256 = (
    "6a9d05a0fb92289334cf39bb6947a2022a05a7c1892e8bb1c5a7156f9ca227f4"
)
POSE_AUDIT_REPORT_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_n32_pose_projection_audit_result_2026-07-11.md"
)
POSE_AUDIT_REPORT_SHA256 = (
    "e1a0c7e8c161827c5d8a1e2088135d8d986cbce9f9f7c02aa43d78d37a0be5e8"
)
POSE_ROLE_NAMESPACE_AMENDMENT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_n32_pose_projection_role_namespace_amendment_2026-07-11.md"
)
POSE_ROLE_NAMESPACE_AMENDMENT_SHA256 = (
    "ae17eb856c5329e8c5dfa5e4339306ef19e60c53c5f67d43746b268be9cc3370"
)
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_n32_v4_hierarchical_binding_2026-07-11.md"
)

EXPECTED_INITIAL_STATE_SHA256 = {
    20260710: "0e82e8832eb2c27dc9ef2ea4c6ff35a83dcca181cb1d4172830fb6b2811a9c5e",
    20260711: "55ae2bbeecbe3913c7e886c11a3a14a5c4c435673a6067df45a2cca6d12fbc99",
}
EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256 = dict(v2.EXPECTED_INITIAL_STATE_SHA256)
EXPECTED_SCHEDULE_SHA256 = {
    20260710: "79b6e66d4e90246f9eb045675f2a06eb25ae28d26f0997392b6780518e668156",
    20260711: "f621b85716607b7e7b8e1ba931d19cf552eb944feca48d099a2c1a3b8ef801c6",
}
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
        ("v3_n32_runner" if name == "runner" else name): path
        for name, path in v3._source_paths().items()
    }
    return {
        **shared,
        "n32_v4_binding": CONTRACT_PATH,
        "n32_v4_pure": (
            REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32_v4.py"
        ),
        "n32_v4_hierarchical_model": (
            REPOSITORY_ROOT
            / "lewm/models/categorical_radial_perception_full_ray_hierarchical.py"
        ),
        "n32_v4_model_test": (
            REPOSITORY_ROOT
            / "lewm/tests/test_categorical_radial_perception_full_ray_hierarchical.py"
        ),
        "n32_v4_pure_test": (
            REPOSITORY_ROOT / "lewm/tests/test_go2_categorical_radial_n32_v4.py"
        ),
        "n32_v4_runner_test": (
            REPOSITORY_ROOT / "lewm/tests/test_run_go2_categorical_radial_n32_v4.py"
        ),
        "n32_v4_finalizer_test": (
            REPOSITORY_ROOT / "lewm/tests/test_finalize_go2_categorical_radial_n32_v4.py"
        ),
        "v4_finalizer": (
            REPOSITORY_ROOT / "scripts/finalize_go2_categorical_radial_n32_v4.py"
        ),
        "runner": Path(__file__).resolve(),
    }


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in sorted(_source_paths().items())
    }


def _evidence_files() -> dict[Path, str]:
    return {
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
        V3_CONTRACT_PATH: V3_CONTRACT_SHA256,
        N32_V3_RESULT_PATH: N32_V3_RESULT_FILE_SHA256,
        N32_V3_RESULT_NOTE_PATH: N32_V3_RESULT_NOTE_SHA256,
        KNOWN_BIAS_PROOF_PATH: KNOWN_BIAS_PROOF_SHA256,
        POSE_AUDIT_RESULT_PATH: POSE_AUDIT_RESULT_FILE_SHA256,
        POSE_AUDIT_REPORT_PATH: POSE_AUDIT_REPORT_SHA256,
        POSE_ROLE_NAMESPACE_AMENDMENT_PATH: POSE_ROLE_NAMESPACE_AMENDMENT_SHA256,
        CONTRACT_PATH: EXECUTION_BINDING_SHA256,
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
    dict[str, Any],
    dict[str, Any],
]:
    """Validate all V4 ordering evidence before deserializing new evidence."""

    expected = _evidence_files()
    pre_hashes = {path: _sha256_file(path) for path in expected}
    if pre_hashes != expected:
        raise ValueError("bound N32 V4 evidence SHA-256 mismatch")

    inherited = v3._load_bound_inputs()
    v3_result = _read_json(N32_V3_RESULT_PATH)
    _validate_content(
        v3_result,
        schema=v3.RESULT_SCHEMA,
        content_sha256=N32_V3_RESULT_CONTENT_SHA256,
        name="N32 V3 result",
    )
    v3_decision = v3_result.get("decision", {})
    if (
        v3_result.get("authoritative") is not True
        or v3_result.get("aggregation_eligible") is not True
        or v3_decision.get("classification") != "fit_gate_failed"
        or v3_decision.get("favorable") is not False
        or v3_decision.get("token_width_32_fit_passes") is not False
        or v3_result.get("shared_jepa_full_train_candidate_licensed") is not False
        or v3_result.get("contract")
        != {
            "path": str(V3_CONTRACT_PATH.resolve()),
            "sha256": V3_CONTRACT_SHA256,
        }
    ):
        raise ValueError("bound failed N32 V3 result identity drift")

    pose_result = _read_json(POSE_AUDIT_RESULT_PATH)
    _validate_content(
        pose_result,
        schema=POSE_AUDIT_RESULT_SCHEMA,
        content_sha256=POSE_AUDIT_RESULT_CONTENT_SHA256,
        name="N32 pose-projection audit result",
    )
    pose_decision = pose_result.get("ordering_decision", {})
    if (
        pose_decision.get("material_dynamic_pose_mismatch") is not False
        or pose_decision.get("rough_threshold_passes") is not False
        or pose_decision.get("contrast_threshold_passes") is not False
        or pose_decision.get("next_intervention")
        != "explicit_hierarchical_output"
    ):
        raise ValueError("bound N32 pose-projection ordering identity drift")

    post_hashes = {path: _sha256_file(path) for path in expected}
    if post_hashes != pre_hashes:
        raise RuntimeError("bound N32 V4 evidence changed during parsing")
    return (*inherited, v3_result, pose_result)


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
    """Build and verify the registered V2/V4 initialization pair."""

    reference, candidate = build_comparable_width24_and_hierarchical_models(
        cpu_rng_state
    )
    reference_state = _backend._clone_state(reference.state_dict())
    candidate_state = _backend._clone_state(candidate.state_dict())
    reference_hash = _backend._state_dict_sha256(reference_state)
    candidate_hash = _backend._state_dict_sha256(candidate_state)
    if len(reference_state) != REGISTERED_STATE_ENTRY_COUNT:
        raise RuntimeError("N32 V4 registered state-entry count changed")
    changed = sorted(
        name
        for name in reference_state
        if tuple(reference_state[name].shape) != tuple(candidate_state[name].shape)
    )
    registered = sorted(REGISTERED_SHAPE_CHANGED_STATE_KEYS)
    if set(reference_state) != set(candidate_state) or changed != registered:
        raise RuntimeError("N32 V4 initialization state-key/shape contract changed")
    same_shape = sorted(set(reference_state) - set(changed))
    if len(same_shape) != REGISTERED_SAME_SHAPE_STATE_ENTRY_COUNT or any(
        reference_state[name].dtype != candidate_state[name].dtype
        or not torch.equal(reference_state[name], candidate_state[name])
        for name in same_shape
    ):
        raise RuntimeError("N32 V4 same-shape initialization is not bit-identical")
    if reference_hash != EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256[int(seed)]:
        raise RuntimeError("N32 V4 V2 reference initialization changed")
    if candidate_hash != EXPECTED_INITIAL_STATE_SHA256[int(seed)]:
        raise RuntimeError("N32 V4 registered initialization changed")
    proof = {
        "construction": (
            "save_cpu_rng_after_determinism_then_replay_for_v2_and_v4_copy_131_"
            "same_shape_entries_leave_two_changed_head_tensors_at_pytorch_defaults"
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
                "v4_shape": _tensor_shape(candidate_state[name]),
                "v2_dtype": str(reference_state[name].dtype),
                "v4_dtype": str(candidate_state[name].dtype),
            }
            for name in changed
        },
        "shape_changed_head_tensors_left_at_deterministic_pytorch_default": True,
        "class_prior_bias_matching_applied": False,
        "analytic_v2_head_transform_applied": False,
        "zero_initialization_applied": False,
        "trained_v2_weight_loaded": False,
        "trained_v3_weight_loaded": False,
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
    model = CategoricalRadialPerceptionFullRayHierarchical().to(device)
    model.load_state_dict(initial_state, strict=True)
    if _backend._state_dict_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("N32 V4 did not start from the registered initial state")
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
    schedule_sha256 = _canonical_json_sha256(schedule)
    if updates == int(AUTHORITATIVE_CONFIG["updates"]) and (
        schedule_sha256 != EXPECTED_SCHEDULE_SHA256[int(seed)]
    ):
        raise RuntimeError("N32 V4 registered minibatch schedule changed")
    before_training = dataset.snapshot()
    evaluation_access: Counter[str] = Counter()
    curve = []
    for step, indices in enumerate(schedule, start=1):
        if len(indices) != BATCH_SIZE:
            raise RuntimeError("N32 V4 direct training batch was not 80 frames")
        batch = dataset.training_batch(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device=device, dtype=torch.float32)
        if int(images.shape[0]) != BATCH_SIZE:
            raise RuntimeError("N32 V4 training tensor was not a direct batch 80")
        logits = model(images)
        loss = direct_hierarchical_loss(
            logits,
            batch["labels"].to(device),
            batch["mask"].to(device),
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite N32 V4 loss at update {step}")
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
        "final_state_sha256": _backend._state_dict_sha256(model.state_dict()),
        "minibatch_indices": schedule,
        "minibatch_indices_sha256": schedule_sha256,
        "learning_curve": curve,
        "terminal_fit_gate": terminal,
        "training_access": training,
        "fit_evaluation_access": evaluation,
        "holdouts_evaluated": False,
    }, model


def _canonical_output(seed: int) -> Path:
    return (
        REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_n32/v4/"
        f"seed_{int(seed)}_result.json"
    ).resolve()


def _validate_primary_authorization(
    path: Path,
    expected_sha256: str,
    current_sources: Mapping[str, Any],
    patch7_reference: Mapping[str, Any],
) -> dict[str, Any]:
    from scripts import finalize_go2_categorical_radial_n32_v4 as finalizer

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
        parser.error("output already exists; N32 V4 results are immutable")
    if args.seed not in EXPECTED_INITIAL_STATE_SHA256:
        parser.error("N32 V4 seed must be 20260710 or 20260711")
    authorization = (
        args.seed_20260710_result,
        args.expected_seed_20260710_sha256,
    )
    if args.seed == 20260710 and any(value is not None for value in authorization):
        parser.error("seed 20260710 rejects seed-authorization arguments")
    if args.seed == 20260711 and any(value is None for value in authorization):
        parser.error("seed 20260711 requires both seed-authorization arguments")
    if args.non_authoritative_smoke and args.seed != 20260710:
        parser.error("N32 V4 smoke is seed-20260710-only")
    if args.non_authoritative_smoke and args.output.resolve() in {
        _canonical_output(seed) for seed in EXPECTED_INITIAL_STATE_SHA256
    }:
        parser.error("N32 V4 smoke output must not occupy a canonical result path")
    if not args.non_authoritative_smoke and args.output.resolve() != _canonical_output(
        args.seed
    ):
        parser.error("authoritative N32 V4 output must use its canonical path")
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
    (
        panel,
        panels,
        ladder,
        ladder_v3_result,
        patch7_reference,
        v1_result,
        v2_result,
        n32_v3_result,
        pose_audit_result,
    ) = _load_bound_inputs()
    git_start = _backend._git_snapshot()
    primary = None
    if args.seed == 20260711:
        primary = _validate_primary_authorization(
            args.seed_20260710_result.resolve(),
            str(args.expected_seed_20260710_sha256),
            source_start,
            patch7_reference,
        )

    device = _backend._resolve_device(str(args.device))
    determinism = _backend._configure_determinism(int(args.seed))
    cpu_rng_state = torch.get_rng_state().clone()
    initial_state, initial_state_sha256, initialization_proof = (
        _build_comparable_initialization(
            seed=int(args.seed), cpu_rng_state=cpu_rng_state
        )
    )
    parameter_count = int(initialization_proof["candidate_parameter_count"])
    if parameter_count != REGISTERED_PARAMETER_COUNT:
        raise RuntimeError("N32 V4 model parameter count changed")

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
        raise RuntimeError("seed-20260710 V4 authorization changed during execution")
    evidence_hashes = {
        str(path.resolve()): digest for path, digest in _evidence_files().items()
    }
    for path, digest in evidence_hashes.items():
        if _sha256_file(Path(path)) != digest:
            raise RuntimeError(f"bound N32 V4 evidence changed: {path}")
    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("N32 V4 sources changed during execution")
    git_end = _backend._git_snapshot()
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
        "dataset_role_policy": {
            "current_physical_dataset_role": "train",
            "current_physical_dataset_role_governs_access": True,
            "legacy_rollout_split_is_provenance_only": True,
            "legacy_rollout_split_used_to_filter_rank_calibrate_or_select": False,
        },
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
            "failed_v3_n32_result": {
                "path": str(N32_V3_RESULT_PATH.resolve()),
                "sha256": N32_V3_RESULT_FILE_SHA256,
                "content_sha256": N32_V3_RESULT_CONTENT_SHA256,
            },
            "failed_v3_n32_result_note": {
                "path": str(N32_V3_RESULT_NOTE_PATH.resolve()),
                "sha256": N32_V3_RESULT_NOTE_SHA256,
            },
            "known_bias_impossibility_proof": {
                "path": str(KNOWN_BIAS_PROOF_PATH.resolve()),
                "sha256": KNOWN_BIAS_PROOF_SHA256,
            },
            "pose_projection_audit_result": {
                "path": str(POSE_AUDIT_RESULT_PATH.resolve()),
                "sha256": POSE_AUDIT_RESULT_FILE_SHA256,
                "content_sha256": POSE_AUDIT_RESULT_CONTENT_SHA256,
            },
            "pose_projection_audit_report": {
                "path": str(POSE_AUDIT_REPORT_PATH.resolve()),
                "sha256": POSE_AUDIT_REPORT_SHA256,
            },
            "pose_projection_role_namespace_amendment": {
                "path": str(POSE_ROLE_NAMESPACE_AMENDMENT_PATH.resolve()),
                "sha256": POSE_ROLE_NAMESPACE_AMENDMENT_SHA256,
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
            "class": "CategoricalRadialPerceptionFullRayHierarchical",
            "parameter_count": parameter_count,
            "token_feature_dim": REGISTERED_TOKEN_FEATURE_DIM,
            "context_dim": REGISTERED_CONTEXT_DIM,
            "parameter_delta_from_v2": (
                REGISTERED_PARAMETER_COUNT - v2.REGISTERED_PARAMETER_COUNT
            ),
            "factor_order": list(FACTOR_OUTPUT_CONTRACT["raw_factor_order"]),
            "output_semantics": (
                "normalized_joint_log_probabilities_before_unchanged_cartesian_gather"
            ),
            "factor_output_contract": FACTOR_OUTPUT_CONTRACT,
            "factor_output_contract_sha256": FACTOR_OUTPUT_CONTRACT_SHA256,
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
        "categorical_radial_full_train_candidate_licensed": False,
        "runtime_ready": False,
        "g2_licensed": False,
        "g3_licensed": False,
    }
    payload = {**core, "content_sha256": _canonical_json_sha256(core)}
    _backend._atomic_write_json_exclusive(args.output.resolve(), payload)
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
