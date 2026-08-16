#!/usr/bin/env python3
"""Run the sole exploratory LayerNorm-affine attentive scorer successor.

The only model change is the already validated externalisation of affine
multiply/add at seven LayerNorm paths.  Fit data are materialised through a
1,152-row allowlist.  Calibration remains unopened until a final-checkpoint
evaluation-authorisation receipt exists, then is forwarded exactly once.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import (  # noqa: E402
    go2_attentive_readout_layernorm_affine_scientific_successor_v1_contract
    as CONTRACT,
)
from scripts import build_go2_branch_corpus_v1_2 as BUILDER  # noqa: E402
from scripts import encode_go2_scorer_fit_oracle_v1_3 as ENCODER  # noqa: E402
from scripts import run_go2_scorer_fit_oracle_v1_3 as ORACLE  # noqa: E402
from scripts import train_go2_utility_scorer_v1_2 as FROZEN  # noqa: E402
from scripts import train_go2_utility_scorer_v1_3 as V13  # noqa: E402
from scripts import (  # noqa: E402
    train_go2_utility_scorer_v1_3_attentive_readout_v1 as BASE,
)
from scripts import (  # noqa: E402
    diagnose_go2_attentive_readout_layernorm_affine_externalisation_v1
    as LN_DIAGNOSTIC,
)
from scripts import (  # noqa: E402
    train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1 as PREVIOUS,
)


STATUS = CONTRACT.STATUS
ATTEMPT_SCHEMA = "go2_layernorm_affine_scientific_successor_v1_attempt_v1"
INITIALISATION_SCHEMA = (
    "go2_layernorm_affine_scientific_successor_v1_initialisation_v1")
CHECKPOINT_SCHEMA = "go2_layernorm_affine_scientific_successor_v1_checkpoint_v1"
EVALUATION_SCHEMA = (
    "go2_layernorm_affine_scientific_successor_v1_evaluation_authorisation_v1")
RESULT_SCHEMA = "go2_layernorm_affine_scientific_successor_v1_result_v1"
FAILURE_SCHEMA = "go2_layernorm_affine_scientific_successor_v1_technical_failure_v1"
ATTEMPT_KEY = "layernorm_affine_successor_attempt_digest"
EVALUATION_KEY = "layernorm_affine_successor_evaluation_authorisation_digest"
RESULT_KEY = "layernorm_affine_successor_result_digest"
FAILURE_KEY = "layernorm_affine_successor_technical_failure_digest"
EVIDENCE_KEY = PREVIOUS.EVIDENCE_SELF_KEY

ENCODED_ROOT_RELATIVE = Path(
    ".generated/go2_scorer_fit_oracle_v1_3/encoded_training_view")
ROW_ROOT_RELATIVE = Path(
    ".generated/go2_branch_corpus_v1_2/scorer_fit/row_records_v2")
FRAME_ROOT_RELATIVE = Path(
    ".generated/go2_branch_corpus_v1_2/scorer_fit")
OVERLAY_ROOT_RELATIVE = Path(
    ".generated/go2_scorer_fit_oracle_v1_3/replay_overlays")
LATENT_INDEX_DIGEST = (
    "25bbd7731fc2e3026063544c64d31abff2c0ded43991504eab4d11938401b758")
TRAINING_VIEW_DIGEST = (
    "9eefff24953fdfc1eb7718ff6067a9bc06f5f8bd321f62769521234d6393291c")
FIT_STATE_SET_DIGEST = (
    "858ad55b14d0079ea11c49a1c79b2245c7adb71846493c449e7eb3cf1d16900a")
FIT_SCENE_SET_DIGEST = (
    "a7ef974169522a270f407de1b1b6023583816f82f76a9b8b9cc0b896bfa67373")
FIT_BRANCH_SET_DIGEST = (
    "0f7a4bd74a21b0f11ee258985983ab910e869055e4bfc7a804c19196284ffbb3")
CALIBRATION_FORWARD_BATCHES = CONTRACT.CALIBRATION_ROWS // CONTRACT.MICROBATCH


class ScientificSuccessorError(RuntimeError):
    """The one-shot successor execution or frozen evidence changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScientificSuccessorError(message)


def signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    require(key not in result, f"{key} already exists")
    result[key] = CONTRACT.digest(result)
    return result


def validate_signed(value: Mapping[str, Any], key: str,
                    label: str) -> dict[str, Any]:
    result = dict(value)
    recorded = result.pop(key, None)
    require(isinstance(recorded, str) and recorded == CONTRACT.digest(result),
            f"{label} self digest changed")
    result[key] = recorded
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    return CONTRACT.read_json(path, label)


def publish_json_once(path: Path, value: Mapping[str, Any], label: str) -> None:
    require(not path.exists() and not path.is_symlink(),
            f"refusing to overwrite {label}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    require(not temporary.exists() and not temporary.is_symlink(),
            f"stale temporary {label}")
    try:
        with temporary.open("x", encoding="ascii") as target:
            target.write(json.dumps(value, sort_keys=True, indent=2,
                                    ensure_ascii=True, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def runtime_root(root: Path = ROOT) -> Path:
    return CONTRACT.runtime_root(root)


def path(root: Path, name: str) -> Path:
    return runtime_root(root) / name


def attempt_path(root: Path = ROOT) -> Path:
    return path(root, "attempt.json")


def initialisation_path(root: Path = ROOT) -> Path:
    return path(root, "initialisation.pt")


def checkpoint_path(root: Path = ROOT) -> Path:
    return path(root, "final_checkpoint.pt")


def evaluation_path(root: Path = ROOT) -> Path:
    return path(root, "evaluation_authorisation.json")


def evidence_path(root: Path = ROOT) -> Path:
    return path(root, "calibration_evidence.json")


def result_path(root: Path = ROOT) -> Path:
    return path(root, "result.json")


def failure_path(root: Path = ROOT) -> Path:
    return path(root, "technical_failure.json")


RUNTIME_FILES = {
    "contract": "contract.json", "initialisation": "initialisation.pt",
    "attempt": "attempt.json", "checkpoint": "final_checkpoint.pt",
    "evaluation": "evaluation_authorisation.json",
    "evidence": "calibration_evidence.json", "result": "result.json",
    "failure": "technical_failure.json",
}


def runtime_inventory(root: Path = ROOT) -> dict[str, Any]:
    directory = runtime_root(root)
    require(directory.is_dir() and not directory.is_symlink(),
            "scientific runtime namespace changed")
    entries = []
    for member in sorted(directory.iterdir(), key=lambda value: value.name):
        require(member.is_file() and not member.is_symlink(),
                "scientific runtime contains a non-regular entry")
        require(member.name in set(RUNTIME_FILES.values()),
                "scientific runtime contains an unexpected artifact")
        entries.append({"name": member.name, "byte_count": member.stat().st_size,
                        "sha256": CONTRACT.file_sha256(member),
                        "mode": member.stat().st_mode & 0o777})
    return {"files": entries, "file_count": len(entries),
            "total_bytes": sum(row["byte_count"] for row in entries)}


def _module_at(model: nn.Module, dotted: str) -> nn.Module:
    module: nn.Module = model
    for member in dotted.split("."):
        module = (module[int(member)] if member.isdigit()
                  else getattr(module, member))
    return module


def implementation_inventory(model: nn.Module) -> dict[str, Any]:
    """Prove native modules support the exact validated context manager."""

    compatible = []
    for dotted in CONTRACT.LN.LAYER_NORM_PATHS:
        module = _module_at(model, dotted)
        require(isinstance(module, nn.LayerNorm)
                and tuple(module.normalized_shape) == (512,)
                and module.elementwise_affine
                and module.weight is not None and module.bias is not None
                and float(module.eps) == 1e-5,
                f"LayerNorm implementation boundary changed: {dotted}")
        compatible.append(dotted)
    negative = _module_at(model, CONTRACT.LN.NEGATIVE_CONTROL_PATH)
    require(isinstance(negative, nn.LayerNorm),
            "negative-control LayerNorm changed")
    return {
        "implementation_name": CONTRACT.LN.IMPLEMENTATION_NAME,
        "implementation_digest": CONTRACT.LN.IMPLEMENTATION_DIGEST,
        "context_manager": (
            "diagnose_go2_attentive_readout_layernorm_affine_"
            "externalisation_v1.externalised_layernorms"),
        "compatible_paths": compatible,
        "native_modules_preserved_outside_forward_context": True,
        "negative_control_path": CONTRACT.LN.NEGATIVE_CONTROL_PATH,
        "state_dict_keys": list(model.state_dict()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_tensor_count": sum(
            1 for _ in model.parameters()),
    }


def model_factory(*, configure_seed: bool = True) -> nn.Module:
    if configure_seed:
        FROZEN.configure_determinism(CONTRACT.SCORER_SEED)
    model = BASE.FinalLayerAttentiveUtilityScorer()
    inventory = implementation_inventory(model)
    require(inventory["compatible_paths"] == list(CONTRACT.LN.LAYER_NORM_PATHS)
            and inventory["trainable_parameter_count"]
            == CONTRACT.TRAINABLE_PARAMETERS
            and inventory["trainable_parameter_tensor_count"]
            == CONTRACT.TRAINABLE_TENSORS,
            "implementation factory inventory changed")
    return model


def issue_contract(root: Path = ROOT) -> dict[str, Any]:
    source = CONTRACT.source_closure(root)
    predecessor = CONTRACT.validate_predecessor_success(root)
    storage = CONTRACT.storage_binding(root)
    value = CONTRACT.build_contract(source, predecessor, storage)
    directory = runtime_root(root)
    directory.mkdir(parents=True, exist_ok=False)
    publish_json_once(CONTRACT.contract_path(root), value,
                      "scientific successor contract")
    return value


def load_contract(root: Path = ROOT) -> dict[str, Any]:
    value = CONTRACT.validate_contract(read_json(
        CONTRACT.contract_path(root), "scientific successor contract"))
    require(value["source_closure"] == CONTRACT.source_closure(root)
            and value["predecessor"]
            == CONTRACT.validate_predecessor_success(root),
            "live successor source or predecessor changed")
    return value


def device_preflight() -> tuple[torch.device, dict[str, Any]]:
    return PREVIOUS.device_preflight()


def _validate_raw_row(value: Mapping[str, Any], identity: str) -> None:
    body = dict(value)
    recorded = body.pop("branch_row_digest", None)
    require(isinstance(recorded, str)
            and recorded == BUILDER.canonical_digest(body),
            "fit row self digest changed")
    require(value.get("branch_identity_digest") == identity
            and value.get("split_role") == "fit"
            and value.get("record_complete") is True
            and value.get("candidate_index") in range(12)
            and len(value.get("action_blocks", [])) == 4
            and len(value.get("goal_binding_input", [])) == 3,
            "fit row identity, role, or input changed")


def load_fit_only_training_corpus(root: Path = ROOT) -> dict[str, Any]:
    """Open exactly 1,152 fit rows/six fit overlays and no calibration labels."""

    ledger = CONTRACT.fit_only_ledger()
    ledger_by_branch = {row["branch_identity_digest"]: row for row in ledger}
    identities = [row["branch_identity_digest"] for row in ledger]
    require(CONTRACT.digest(sorted(identities)) == FIT_BRANCH_SET_DIGEST,
            "fit-only embedded branch identity set changed")
    rows = []
    replay_count = 0
    for identity in identities:
        raw_path = root / ROW_ROOT_RELATIVE / f"{identity}.json"
        raw = read_json(raw_path, f"fit row {identity}")
        _validate_raw_row(raw, identity)
        overlay_path = root / OVERLAY_ROOT_RELATIVE / f"{identity}.json"
        if overlay_path.exists() or overlay_path.is_symlink():
            overlay = read_json(overlay_path, f"fit replay overlay {identity}")
            try:
                ORACLE._validate_self_digest(overlay, "replay_overlay_digest")
            except Exception as exc:
                raise ScientificSuccessorError(
                    "fit replay overlay self digest changed") from exc
            require(overlay.get("source_branch_identity_digest") == identity
                    and overlay.get("split_role") == "fit"
                    and overlay.get("oracle_v1_3_digest")
                    == SCIENCE_ORACLE_DIGEST,
                    "fit replay overlay identity changed")
            source_kind = ORACLE.SOURCE_KIND_REPLAY
            label_path, label_source = overlay_path, overlay
            label_digest = str(overlay["replay_overlay_digest"])
            projected = ORACLE._label_projection(overlay["labels"])
            require(raw.get("valid") is False
                    and all(math.isfinite(float(projected[key]))
                            for key in ("progress", "safety", "completion",
                                        "utility")),
                    "fit replay source/labels changed")
            replay_count += 1
        else:
            source_kind = ORACLE.SOURCE_KIND_V2_VALID
            label_path, label_source = raw_path, raw
            label_digest = str(raw["branch_row_digest"])
            projected = ORACLE._label_projection(raw)
            require(raw.get("valid") is True
                    and all(math.isfinite(float(projected[key]))
                            for key in ("progress", "safety", "completion",
                                        "utility")),
                    "fit V2 adoption source/labels changed")
        reference = ORACLE._training_view_row(
            role="fit", source_kind=source_kind, state=raw,
            input_path=raw_path, input_row=raw,
            frame_root=root / FRAME_ROOT_RELATIVE,
            label_path=label_path, label_source=label_source,
            label_self_digest=label_digest)
        expected = ledger_by_branch[identity]
        require(reference["training_view_row_digest"]
                == expected["training_view_row_digest"],
                "fit training-view row projection changed")
        labels = reference["label_projection"]
        rows.append({
            "split_role": "fit", "role": "fit",
            "source_kind": source_kind,
            "state_id": raw["state_id"],
            "state_identity_digest": raw["state_identity_digest"],
            "scene_id": raw["scene_id"], "family": raw["family"],
            "stratum": raw["stratum"],
            "candidate_index": int(raw["candidate_index"]),
            "branch_identity_digest": identity,
            "training_view_row_digest": reference["training_view_row_digest"],
            "action_blocks": raw["action_blocks"],
            "goal_binding_input": raw["goal_binding_input"],
            **labels,
        })
    rows.sort(key=lambda row: (str(row["state_id"]),
                               int(row["candidate_index"])))
    state_digests = sorted({str(row["state_identity_digest"]) for row in rows})
    scene_ids = sorted({str(row["scene_id"]) for row in rows})
    require(len(state_digests) == CONTRACT.FIT_STATES
            and CONTRACT.digest(state_digests) == FIT_STATE_SET_DIGEST
            and CONTRACT.digest(scene_ids) == FIT_SCENE_SET_DIGEST
            and replay_count == 6
            and CONTRACT.digest([row["training_view_row_digest"]
                                 for row in rows])
            == CONTRACT.DATA_ORDER[
                "base_training_view_row_digest_sequence_digest"],
            "fit-only row closure changed")
    selected = []
    encoded_root = root / ENCODED_ROOT_RELATIVE
    for position, row in enumerate(rows):
        expected = ledger_by_branch[row["branch_identity_digest"]]
        relative = (Path("latents/horizon") /
                    f"{row['training_view_row_digest']}.f16")
        record = {
            "training_view_row_digest": row["training_view_row_digest"],
            "state_id": row["state_id"],
            "state_identity_digest": row["state_identity_digest"],
            "candidate_index": row["candidate_index"],
            "source_kind": row["source_kind"],
            "path": str(relative),
            "sha256": expected["latent_sha256"],
            "byte_count": expected["latent_byte_count"],
            "shape": [4, 768, 1024],
        }
        require(ENCODER._valid_latent_record(
                    encoded_root / relative, record, row),
                "fit latent shard changed")
        selected.append(dict(record))
        row["_latent_index"] = position
    return {
        "fit_rows": rows,
        "horizon": FROZEN.HorizonShardStore(selected, encoded_root),
        "binding": {
            "fit_rows": CONTRACT.FIT_ROWS, "fit_states": CONTRACT.FIT_STATES,
            "v2_adoptions": 1_146, "fit_replay_overlays": 6,
            "fit_only_ledger_digest": CONTRACT.FIT_ONLY_LEDGER_DIGEST,
            "latent_index_digest": LATENT_INDEX_DIGEST,
            "training_view_digest": TRAINING_VIEW_DIGEST,
            "global_training_view_opened": False,
            "global_latent_index_bytes_read": False,
            "global_encoding_receipt_bytes_read": False,
            "calibration_row_records_opened": 0,
            "calibration_overlay_records_opened": 0,
            "calibration_latent_shards_opened": 0,
        },
    }


SCIENCE_ORACLE_DIGEST = (
    "0592876e7768a627198f1154da64b4ed492237fe68196e011fcbfcfef7706e63")


def fresh_initialisation(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    model = model_factory()
    state = FROZEN._cpu_state(model)
    state_digest = FROZEN.state_dict_digest(state)
    require(state_digest == CONTRACT.INITIAL_STATE_DIGEST,
            "fresh successor initial state changed")
    payload = {
        "schema": INITIALISATION_SCHEMA, "status": STATUS,
        "contract_digest": contract[CONTRACT.CONTRACT_SELF_KEY],
        "implementation_digest": CONTRACT.LN.IMPLEMENTATION_DIGEST,
        "registered_seed": CONTRACT.SCORER_SEED,
        "initial_state_digest": state_digest,
        "model_state_dict": state,
        "factory_inventory": implementation_inventory(model),
        "diagnostic_checkpoint_state_reused": False,
    }
    target = initialisation_path(root)
    require(not target.exists() and not target.is_symlink(),
            "scientific initialisation already exists")
    FROZEN.atomic_torch_save(payload, target)
    target.chmod(0o444)
    return payload


def execution_bindings(contract: Mapping[str, Any],
                       fit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_digest": contract[CONTRACT.CONTRACT_SELF_KEY],
        "predecessor_terminal_digest":
            CONTRACT.PREDECESSOR_BINDING["terminal_digest"],
        "predecessor_local_cases_digest":
            CONTRACT.PREDECESSOR_BINDING["local_cases_digest"],
        "predecessor_conditional_smoke_digest":
            CONTRACT.PREDECESSOR_BINDING["conditional_smoke_digest"],
        "implementation_digest": CONTRACT.LN.IMPLEMENTATION_DIGEST,
        "fit_only_binding": dict(fit["binding"]),
    }


def _loss(outputs: Sequence[torch.Tensor], targets: Mapping[str, torch.Tensor],
          indices: torch.Tensor) -> torch.Tensor:
    progress, safety, completion = outputs
    return (
        F.mse_loss(progress, targets["progress"][indices], reduction="sum")
        + F.binary_cross_entropy_with_logits(
            safety, targets["safety"][indices], reduction="sum")
        + F.binary_cross_entropy_with_logits(
            completion, targets["completion"][indices], reduction="sum")
    ) / CONTRACT.EFFECTIVE_BATCH


def expected_technical_trace() -> list[dict[str, Any]]:
    """Return the complete frozen technical-only 60-epoch trace."""

    return [{
        "epoch": epoch,
        "completed_optimizer_updates": epoch * CONTRACT.UPDATES_PER_EPOCH,
        "technical_finite": True,
        "performance_metric_inspected": False,
        "calibration_opened": False,
    } for epoch in range(1, CONTRACT.EPOCHS + 1)]


def train_once(*, fit: Mapping[str, Any], initialisation: Mapping[str, Any],
               bindings: Mapping[str, Any], device: torch.device,
               root: Path, custody: dict[str, Any]
               ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rows = fit["fit_rows"]
    rows, order_witness = BASE.registered_fit_rows_and_data_order(rows)
    require(order_witness["base_training_view_row_digest_sequence_digest"]
            == CONTRACT.DATA_ORDER[
                "base_training_view_row_digest_sequence_digest"]
            and order_witness["permutation_plan_digest"]
            == CONTRACT.DATA_ORDER["permutation_plan_digest"]
            and order_witness["row_presentation_plan_digest"]
            == CONTRACT.DATA_ORDER["row_presentation_plan_digest"]
            and order_witness["final_generator_state_digest"]
            == CONTRACT.DATA_ORDER["final_generator_state_digest"],
            "registered fit data order changed")
    attempt = signed({
        "schema": ATTEMPT_SCHEMA, "status": STATUS,
        "attempt_number": 1, "maximum_attempts": 1,
        "execution_bindings": dict(bindings),
        "initial_state_digest": initialisation["initial_state_digest"],
        "registered_seed": CONTRACT.SCORER_SEED,
        "data_order_seed": CONTRACT.DATA_ORDER_SEED,
        "data_order_witness": {key: order_witness[key] for key in (
            "base_training_view_row_digest_sequence_digest",
            "permutation_plan_digest", "row_presentation_plan_digest",
            "final_generator_state_digest")},
        "effective_batch": CONTRACT.EFFECTIVE_BATCH,
        "microbatch": CONTRACT.MICROBATCH,
        "gradient_accumulation_steps": CONTRACT.GRADIENT_ACCUMULATION_STEPS,
        "fixed_final_epoch": CONTRACT.EPOCHS,
        "epoch_selection": "final_epoch_only_no_selection",
        "optimizer": dict(CONTRACT.TRAINING["optimizer"]),
        "learning_rate_schedule": CONTRACT.TRAINING["learning_rate_schedule"],
        "calibration_metadata_labels_latents_opened": 0,
        "resume_retry_or_replacement_authorised": False,
    }, ATTEMPT_KEY)
    publish_json_once(attempt_path(root), attempt, "successor attempt")
    model = model_factory()
    model.load_state_dict(initialisation["model_state_dict"], strict=True)
    require(implementation_inventory(model)["compatible_paths"]
            == list(CONTRACT.LN.LAYER_NORM_PATHS),
            "training model lost the implementation")
    model.to(device)
    budget = BASE.frozen_budget()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(budget["lr"]),
        weight_decay=float(budget["weight_decay"]))
    action_goal, targets = BASE._small_features(rows, device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(CONTRACT.DATA_ORDER_SEED)
    updates = 0
    trace: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, CONTRACT.EPOCHS + 1):
        model.train()
        order = torch.randperm(CONTRACT.FIT_ROWS, generator=generator)
        epoch_updates = 0
        for start in range(0, CONTRACT.FIT_ROWS, CONTRACT.EFFECTIVE_BATCH):
            batch_cpu = order[start:start + CONTRACT.EFFECTIVE_BATCH]
            require(len(batch_cpu) == CONTRACT.EFFECTIVE_BATCH,
                    "effective batch changed")
            optimiser.zero_grad(set_to_none=True)
            for micro_start in range(
                    0, CONTRACT.EFFECTIVE_BATCH, CONTRACT.MICROBATCH):
                micro_cpu = batch_cpu[
                    micro_start:micro_start + CONTRACT.MICROBATCH]
                micro = micro_cpu.to(device)
                tokens = BASE._token_batch(
                    rows, fit["horizon"], micro_cpu.tolist(), device)
                with LN_DIAGNOSTIC.externalised_layernorms(model):
                    outputs = model(tokens, action_goal[micro])
                    loss = _loss(outputs, targets, micro)
                    require(bool(torch.isfinite(loss).item()),
                            "successor training loss is non-finite")
                    # Non-reentrant checkpoint recomputation occurs here;
                    # the exact validated patch must remain active through it.
                    loss.backward()
                del tokens
            require(all(parameter.grad is None or bool(
                        torch.isfinite(parameter.grad).all().item())
                        for parameter in model.parameters()),
                    "successor gradient is non-finite")
            nn.utils.clip_grad_norm_(model.parameters(),
                                     float(budget["grad_clip"]))
            optimiser.step()
            updates += 1
            custody["completed_updates"] = updates
            require(all(bool(torch.isfinite(parameter).all().item())
                        for parameter in model.parameters()),
                    "successor parameter is non-finite")
            epoch_updates += 1
        require(epoch_updates == CONTRACT.UPDATES_PER_EPOCH,
                "updates per epoch changed")
        trace.append({"epoch": epoch, "completed_optimizer_updates": updates,
                      "technical_finite": True,
                      "performance_metric_inspected": False,
                      "calibration_opened": False})
        custody["completed_epochs"] = epoch
        print(f"[layernorm-affine-successor] technical epoch {epoch:02d}/60",
              flush=True)
    require(updates == CONTRACT.TOTAL_UPDATES
            and trace == expected_technical_trace()
            and FROZEN.tensor_digest(order.to(torch.int64))
            == CONTRACT.DATA_ORDER["last_epoch_order_digest"]
            and FROZEN.tensor_digest(generator.get_state())
            == CONTRACT.DATA_ORDER["final_generator_state_digest"],
            "training budget or order changed")
    state = FROZEN._cpu_state(model)
    optimiser_state = optimiser.state_dict()
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA, "status": STATUS,
        "attempt_digest": attempt[ATTEMPT_KEY],
        "execution_bindings": dict(bindings),
        "implementation_digest": CONTRACT.LN.IMPLEMENTATION_DIGEST,
        "initial_state_digest": initialisation["initial_state_digest"],
        "final_state_digest": FROZEN.state_dict_digest(state),
        "model_state_dict": state,
        "optimizer_state_dict": optimiser_state,
        "optimizer_state_digest": FROZEN.structured_digest(optimiser_state),
        "registered_seed": CONTRACT.SCORER_SEED,
        "data_order_seed": CONTRACT.DATA_ORDER_SEED,
        "data_order_witness": attempt["data_order_witness"],
        "completed_epoch": CONTRACT.EPOCHS,
        "completed_optimizer_updates": updates,
        "example_presentations": CONTRACT.PRESENTATIONS,
        "effective_batch": CONTRACT.EFFECTIVE_BATCH,
        "microbatch": CONTRACT.MICROBATCH,
        "gradient_accumulation_steps": CONTRACT.GRADIENT_ACCUMULATION_STEPS,
        "epoch_selection": "final_epoch_only_no_selection",
        "optimizer": dict(CONTRACT.TRAINING["optimizer"]),
        "learning_rate_schedule": CONTRACT.TRAINING["learning_rate_schedule"],
        "last_epoch_order_digest": FROZEN.tensor_digest(order.to(torch.int64)),
        "final_order_generator_state_digest": FROZEN.tensor_digest(
            generator.get_state()),
        "technical_trace": trace,
        "training_wall_time_s": round(time.time() - started, 3),
        "calibration_metadata_labels_latents_opened": 0,
        "diagnostic_smoke_state_used": False,
    }
    require(not checkpoint_path(root).exists()
            and not checkpoint_path(root).is_symlink(),
            "final checkpoint already exists")
    FROZEN.atomic_torch_save(checkpoint, checkpoint_path(root))
    checkpoint_path(root).chmod(0o444)
    return state, {
        "path": str(checkpoint_path(root)),
        "sha256": CONTRACT.file_sha256(checkpoint_path(root)),
        "byte_count": checkpoint_path(root).stat().st_size,
        "final_state_digest": checkpoint["final_state_digest"],
        "optimizer_state_digest": checkpoint["optimizer_state_digest"],
        "completed_epoch": CONTRACT.EPOCHS,
        "completed_optimizer_updates": updates,
        "example_presentations": CONTRACT.PRESENTATIONS,
        "attempt_digest": attempt[ATTEMPT_KEY],
        "registered_seed": checkpoint["registered_seed"],
        "data_order_seed": checkpoint["data_order_seed"],
        "data_order_witness": checkpoint["data_order_witness"],
        "effective_batch": checkpoint["effective_batch"],
        "microbatch": checkpoint["microbatch"],
        "gradient_accumulation_steps":
            checkpoint["gradient_accumulation_steps"],
        "optimizer": checkpoint["optimizer"],
        "learning_rate_schedule": checkpoint["learning_rate_schedule"],
        "epoch_selection": checkpoint["epoch_selection"],
        "technical_trace_digest": CONTRACT.digest(checkpoint["technical_trace"]),
        "training_wall_time_s": checkpoint["training_wall_time_s"],
        "technical_validity": True,
    }


def strict_checkpoint_reload(state: Mapping[str, torch.Tensor],
                             checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    model = model_factory()
    model.load_state_dict(state, strict=True)
    budget = BASE.frozen_budget()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(budget["lr"]),
        weight_decay=float(budget["weight_decay"]))
    optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
    require(FROZEN.state_dict_digest(FROZEN._cpu_state(model))
            == checkpoint["final_state_digest"]
            and FROZEN.structured_digest(optimiser.state_dict())
            == checkpoint["optimizer_state_digest"],
            "strict final checkpoint reload changed state")
    return implementation_inventory(model)


def decision(criteria: Mapping[str, bool], safety_auc: float,
             pairwise_gain: float) -> dict[str, Any]:
    require(len(criteria) == 8
            and all(type(value) is bool for value in criteria.values()),
            "eight frozen criteria changed")
    safety_gate = safety_auc >= 0.75
    gain_gate = Decimal(str(pairwise_gain)) >= Decimal("0.05")
    delta_auc = safety_auc - 0.7043234198736978
    delta_gain = pairwise_gain - 0.0317880794701987
    all_original = all(criteria.values())
    if all_original and delta_auc > 0.0 and delta_gain > 0.0:
        classification = "STRONG_READOUT_SIGNAL"
    elif ((not safety_gate and not gain_gate)
          or (delta_auc < 0.0 and delta_gain < 0.0)):
        classification = "NO_READOUT_SIGNAL"
    else:
        classification = "MIXED_READOUT_SIGNAL"
    return {
        "classification": classification,
        "all_original_scorer_criteria_met": all_original,
        "safety_auc_gate_met": safety_gate,
        "latent_over_baseline_pairwise_gain_gate_met": gain_gate,
        "delta_attentive_minus_existing_vitl_safety_auc": delta_auc,
        "delta_attentive_minus_existing_vitl_latent_gain": delta_gain,
        "both_primary_quantities_strictly_improve":
            delta_auc > 0.0 and delta_gain > 0.0,
        "per_family_consistency_is_report_only": True,
        "exploratory_not_qualification": True,
    }


def _artifact(path_value: Path, digest_key: str) -> dict[str, Any]:
    value = validate_signed(read_json(path_value, path_value.name),
                            digest_key, path_value.name)
    return {"path": str(path_value), "sha256": CONTRACT.file_sha256(path_value),
            "byte_count": path_value.stat().st_size,
            "digest": value[digest_key]}


def _record_failure(root: Path, stage: str, error: BaseException,
                    custody: Mapping[str, Any]) -> None:
    target = failure_path(root)
    if target.exists() or target.is_symlink():
        return
    before = runtime_inventory(root)
    payload = signed({
        "schema": FAILURE_SCHEMA,
        "status": "INVALID_TECHNICAL_SCIENTIFIC_SUCCESSOR_ATTEMPT",
        "contract_digest": custody.get("contract_digest"),
        "stage": stage, "exception_type": type(error).__name__,
        "exception": str(error), "traceback": traceback.format_exc(),
        "completed_epochs": int(custody.get("completed_epochs", 0)),
        "completed_optimizer_updates": int(custody.get("completed_updates", 0)),
        "calibration_evaluation_session_consumed": bool(
            custody.get("calibration_session_consumed", False)),
        "calibration_evaluation_completed": bool(
            custody.get("calibration_completed", False)),
        "closed_evidence_rows": int(custody.get("closed_evidence_rows", 0)),
        "artifact_presence_before_failure": {
            name: any(row["name"] == filename for row in before["files"])
            for name, filename in RUNTIME_FILES.items()
            if name != "failure"},
        "runtime_inventory_before_failure": before,
        "runtime": {
            "wall_time_seconds": (
                time.monotonic() - float(custody["session_started"])
                if custody.get("session_started") is not None else 0.0),
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated(torch.device("cuda:0")))
                if torch.cuda.is_available() else 0),
        },
        "retry_resume_or_replacement_authorised": False,
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, FAILURE_KEY)
    publish_json_once(target, payload, "scientific successor failure")


def execute_once(root: Path, custody: dict[str, Any]) -> dict[str, Any]:
    contract = load_contract(root)
    custody["contract_digest"] = contract[CONTRACT.CONTRACT_SELF_KEY]
    initial_inventory = runtime_inventory(root)
    require([row["name"] for row in initial_inventory["files"]]
            == ["contract.json"],
            "scientific namespace was consumed before the sole attempt")
    device, device_receipt = device_preflight()
    torch.cuda.reset_peak_memory_stats(device)
    session_started = time.monotonic()
    custody["session_started"] = session_started
    custody["stage"] = "strict_fit_only_materialisation"
    fit = load_fit_only_training_corpus(root)
    custody["stage"] = "fresh_initialisation"
    initialisation = fresh_initialisation(root, contract)
    bindings = execution_bindings(contract, fit)
    custody["stage"] = "scientific_training"
    state, training = train_once(
        fit=fit, initialisation=initialisation, bindings=bindings,
        device=device, root=root, custody=custody)
    custody["completed_epochs"] = CONTRACT.EPOCHS
    custody["completed_updates"] = CONTRACT.TOTAL_UPDATES
    custody["stage"] = "final_checkpoint_pre_authorisation_reload"
    checkpoint = torch.load(
        checkpoint_path(root), map_location="cpu", weights_only=False)
    preauth_reload = strict_checkpoint_reload(
        checkpoint["model_state_dict"], checkpoint)
    evaluation = signed({
        "schema": EVALUATION_SCHEMA, "status": STATUS,
        "contract_digest": contract[CONTRACT.CONTRACT_SELF_KEY],
        "execution_bindings": bindings,
        "implementation_digest": CONTRACT.LN.IMPLEMENTATION_DIGEST,
        "final_checkpoint_sha256": training["sha256"],
        "final_state_digest": training["final_state_digest"],
        "preauthorisation_strict_reload_factory_inventory": preauth_reload,
        "evaluation_number": 1, "maximum_evaluations": 1,
        "calibration_states": CONTRACT.CALIBRATION_STATES,
        "calibration_rows": CONTRACT.CALIBRATION_ROWS,
        "calibration_metadata_labels_latents_opened_before_authorisation": 0,
        "calibration_model_forwards_before_authorisation": 0,
        "calibration_predictions_before_authorisation": 0,
        "calibration_metrics_before_authorisation": 0,
        "persist_closed_prediction_target_evidence": True,
    }, EVALUATION_KEY)
    publish_json_once(evaluation_path(root), evaluation,
                      "scientific successor evaluation authority")
    custody["stage"] = "authorised_calibration_materialisation"
    corpus = PREVIOUS._load_corpus(root)
    baseline, vitg, frozen_trees = PREVIOUS._load_frozen_comparisons(corpus, root)
    require([row["training_view_row_digest"] for row in fit["fit_rows"]]
            == [row["training_view_row_digest"] for row in sorted(
                corpus["fit_rows"], key=lambda row: (
                    str(row["state_id"]), int(row["candidate_index"])))],
            "strict fit-only and authorised full views disagree")
    custody["stage"] = "sole_calibration_forward"
    custody["calibration_session_consumed"] = True
    evaluation_model = model_factory()
    evaluation_model.load_state_dict(
        checkpoint["model_state_dict"], strict=True)
    evaluation_model.to(device)
    with LN_DIAGNOSTIC.externalised_layernorms(evaluation_model):
        direct_metrics, predictions, _ = BASE._evaluate_streaming(
            evaluation_model, rows=corpus["calibration_rows"],
            store=corpus["horizon"], device=device)
    custody["calibration_completed"] = True
    evidence = PREVIOUS._evidence_payload(
        rows=corpus["calibration_rows"], predictions=predictions,
        bindings=bindings, evaluation_digest=evaluation[EVALUATION_KEY],
        checkpoint_sha256=training["sha256"],
        final_state_digest=training["final_state_digest"])
    publish_json_once(evidence_path(root), evidence,
                      "closed calibration evidence")
    custody["closed_evidence_rows"] = CONTRACT.CALIBRATION_ROWS
    custody["stage"] = "evidence_metric_replay"
    metrics, per_family, per_stratum = PREVIOUS.metrics_from_evidence(
        corpus_rows=corpus["calibration_rows"], evidence=evidence,
        bindings=bindings, evaluation_digest=evaluation[EVALUATION_KEY],
        checkpoint_sha256=training["sha256"],
        final_state_digest=training["final_state_digest"])
    require(FROZEN._safe_json(direct_metrics) == metrics,
            "sole forward metrics and evidence replay differ")
    fit_distribution = FROZEN.label_distribution(corpus["fit_rows"])
    calibration_distribution = FROZEN.label_distribution(
        corpus["calibration_rows"])
    criteria, details, pairwise_gain = V13.qualification_criteria(
        metrics, baseline["metrics"], fit_distribution,
        calibration_distribution)
    route = decision(criteria, float(metrics["safety"]["auc_any_hazard"]),
                     float(pairwise_gain))
    family_report = BASE.per_family_primary_consistency(
        attentive=per_family,
        existing_vitl=baseline["vitl_per_family_metrics"],
        baseline=baseline["per_family_metrics"])
    comparisons = PREVIOUS._comparison_payload(
        attentive=metrics, attentive_family=per_family,
        baseline=baseline, vitg=vitg)
    # Complete the seven-construction scientific-session ledger before
    # terminal publication.  These consumers perform no data/model forward.
    validation_initial_model = model_factory()
    require(FROZEN.state_dict_digest(FROZEN._cpu_state(validation_initial_model))
            == CONTRACT.INITIAL_STATE_DIGEST,
            "prepublication initial-state consumer changed")
    del validation_initial_model
    validation_reload = strict_checkpoint_reload(
        checkpoint["model_state_dict"], checkpoint)
    validation_consumer_model = model_factory()
    validation_consumer_model.load_state_dict(
        checkpoint["model_state_dict"], strict=True)
    require(FROZEN.state_dict_digest(FROZEN._cpu_state(
                validation_consumer_model)) == training["final_state_digest"],
            "prepublication final-state consumer changed")
    del validation_consumer_model
    pre_result_storage = runtime_inventory(root)
    result = signed(FROZEN._safe_json({
        "schema": RESULT_SCHEMA, "status": STATUS,
        "label": CONTRACT.RESULT_LABEL,
        "complete": True, "scientific_result_valid": True,
        "exploratory_not_qualification": True,
        "contract_digest": contract[CONTRACT.CONTRACT_SELF_KEY],
        "execution_bindings": bindings,
        "device_preflight": device_receipt,
        "implementation_name": CONTRACT.LN.IMPLEMENTATION_NAME,
        "implementation_digest": CONTRACT.LN.IMPLEMENTATION_DIGEST,
        "initialisation": {
            "path": str(initialisation_path(root)),
            "sha256": CONTRACT.file_sha256(initialisation_path(root)),
            "byte_count": initialisation_path(root).stat().st_size,
            "initial_state_digest": initialisation["initial_state_digest"],
            "registered_seed": initialisation["registered_seed"],
            "factory_inventory": initialisation["factory_inventory"],
        },
        "training": training,
        "evaluation_authorisation_digest": evaluation[EVALUATION_KEY],
        "calibration_evidence": _artifact(evidence_path(root), EVIDENCE_KEY),
        "training_execution_count": 1,
        "calibration_evaluation_count": 1,
        "calibration_model_forward_batch_count": CALIBRATION_FORWARD_BATCHES,
        "calibration_metric_recomputations_from_closed_evidence": 1,
        "model_factory_construction_ledger": {
            "scientific_session_constructions": 7,
            "phases": [
                "fresh_initialisation", "training", "preauth_strict_reload",
                "sole_calibration_evaluation", "prepublication_initial_consumer",
                "prepublication_checkpoint_reload",
                "prepublication_implementation_consumer"],
            "all_constructions_use_same_factory": True,
            "all_forward_backward_calls_use_exact_externalisation_context": True,
            "prepublication_checkpoint_reload_inventory": validation_reload,
        },
        "results": {
            "attentive": {"calibration": metrics,
                          "per_family_calibration": per_family,
                          "per_stratum_calibration": per_stratum},
            "existing_vitl_frozen": {
                "calibration": frozen_trees["vitl"]["overall"],
                "per_family_calibration": frozen_trees["vitl"]["per_family"],
                "per_stratum_calibration": frozen_trees["vitl"]["per_stratum"],
                "terminal_digest": baseline["vitl_terminal_digest"],
                "safety_auc": 0.7043234198736978,
                "latent_over_baseline_pairwise_gain": 0.0317880794701987},
            "vitg_frozen": {
                "result_digest": vitg["exploratory_result_digest"],
                "calibration": frozen_trees["vitg"]["overall"],
                "per_family_calibration": frozen_trees["vitg"]["per_family"],
                "per_stratum_calibration": frozen_trees["vitg"]["per_stratum"],
                "latent_over_baseline_pairwise_gain":
                    vitg["latent_over_baseline_pairwise_gain"],
                "conclusion": vitg["exploratory_decision"]["classification"]},
            "no_latent_reused": {
                "calibration": frozen_trees["no_latent"]["overall"],
                "per_family_calibration": frozen_trees["no_latent"]["per_family"],
                "per_stratum_calibration": frozen_trees["no_latent"]["per_stratum"],
                **CONTRACT.NO_LATENT_BASELINE},
        },
        "frozen_metric_tree_digests":
            contract["frozen_metric_tree_digests"],
        "latent_over_baseline_pairwise_gain": pairwise_gain,
        "metric_comparisons": comparisons,
        "per_family_primary_consistency": family_report,
        "per_family_consistency_is_report_only": True,
        "frozen_original_gate_replay": {
            "criteria": criteria, "details": details},
        "would_meet_all_original_gates": all(criteria.values()),
        "exploratory_decision": route,
        "qualified_scorer_package_published": False,
        "predictor_retrained": False,
        "predictor_checkpoints_opened_for_utility": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
        "runtime": {
            "wall_time_seconds": time.monotonic() - session_started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "storage_before_result_publication": pre_result_storage,
        },
        "nothing_left_running_by_this_process_after_exit": True,
    }), RESULT_KEY)
    publish_json_once(result_path(root), result, "scientific successor result")
    return result


def validate_result(root: Path = ROOT) -> dict[str, Any]:
    """Replay the terminal solely from closed evidence; never forward again."""

    require(result_path(root).is_file() and not result_path(root).is_symlink()
            and not failure_path(root).exists()
            and not failure_path(root).is_symlink(),
            "result/failure terminal custody changed")
    contract = load_contract(root)
    result = validate_signed(read_json(result_path(root), "successor result"),
                             RESULT_KEY, "successor result")
    require(result.get("schema") == RESULT_SCHEMA
            and result.get("status") == STATUS
            and result.get("label") == CONTRACT.RESULT_LABEL
            and result.get("complete") is True
            and result.get("scientific_result_valid") is True
            and result.get("exploratory_not_qualification") is True
            and result.get("contract_digest")
            == contract[CONTRACT.CONTRACT_SELF_KEY]
            and result.get("implementation_digest")
            == CONTRACT.LN.IMPLEMENTATION_DIGEST
            and result.get("implementation_name")
            == CONTRACT.LN.IMPLEMENTATION_NAME
            and result.get("training_execution_count") == 1
            and result.get("calibration_evaluation_count") == 1
            and result.get("calibration_model_forward_batch_count")
            == CALIBRATION_FORWARD_BATCHES
            and result.get(
                "calibration_metric_recomputations_from_closed_evidence") == 1
            and result.get("qualified_scorer_package_published") is False
            and result.get("predictor_retrained") is False
            and result.get("predictor_checkpoints_opened_for_utility") == 0
            and result.get("predictor_utility_shards_opened") == 0
            and result.get("final_200_state_corpus_generated") is False
            and result.get(
                "nothing_left_running_by_this_process_after_exit") is True,
            "successor result envelope changed")
    bindings = result.get("execution_bindings", {})
    fit_binding = bindings.get("fit_only_binding", {})
    require(bindings.get("contract_digest")
            == contract[CONTRACT.CONTRACT_SELF_KEY]
            and bindings.get("predecessor_terminal_digest")
            == CONTRACT.PREDECESSOR_BINDING["terminal_digest"]
            and bindings.get("predecessor_local_cases_digest")
            == CONTRACT.PREDECESSOR_BINDING["local_cases_digest"]
            and bindings.get("predecessor_conditional_smoke_digest")
            == CONTRACT.PREDECESSOR_BINDING["conditional_smoke_digest"]
            and bindings.get("implementation_digest")
            == CONTRACT.LN.IMPLEMENTATION_DIGEST
            and fit_binding == {
                "fit_rows": 1_152, "fit_states": 96,
                "v2_adoptions": 1_146, "fit_replay_overlays": 6,
                "fit_only_ledger_digest": CONTRACT.FIT_ONLY_LEDGER_DIGEST,
                "latent_index_digest": LATENT_INDEX_DIGEST,
                "training_view_digest": TRAINING_VIEW_DIGEST,
                "global_training_view_opened": False,
                "global_latent_index_bytes_read": False,
                "global_encoding_receipt_bytes_read": False,
                "calibration_row_records_opened": 0,
                "calibration_overlay_records_opened": 0,
                "calibration_latent_shards_opened": 0,
            }, "result execution bindings changed")
    initial_file = initialisation_path(root)
    initial_summary = result.get("initialisation", {})
    require(initial_file.is_file() and not initial_file.is_symlink()
            and initial_summary.get("path") == str(initial_file)
            and initial_summary.get("sha256")
            == CONTRACT.file_sha256(initial_file)
            and initial_summary.get("byte_count") == initial_file.stat().st_size,
            "initialisation byte binding changed")
    initialisation = torch.load(
        initial_file, map_location="cpu", weights_only=False)
    initial_model = model_factory()
    expected_initial = FROZEN._cpu_state(initial_model)
    require(initialisation.get("schema") == INITIALISATION_SCHEMA
            and initialisation.get("contract_digest")
            == contract[CONTRACT.CONTRACT_SELF_KEY]
            and initialisation.get("implementation_digest")
            == CONTRACT.LN.IMPLEMENTATION_DIGEST
            and initialisation.get("registered_seed") == CONTRACT.SCORER_SEED
            and initialisation.get("initial_state_digest")
            == CONTRACT.INITIAL_STATE_DIGEST
            and set(initialisation.get("model_state_dict", {}))
            == set(expected_initial)
            and FROZEN.state_dict_digest(initialisation["model_state_dict"])
            == CONTRACT.INITIAL_STATE_DIGEST
            and all(torch.equal(initialisation["model_state_dict"][key], value)
                    for key, value in expected_initial.items())
            and initialisation.get("diagnostic_checkpoint_state_reused") is False,
            "fresh initialisation artifact changed")
    require(initial_summary == {
        "path": str(initial_file),
        "sha256": CONTRACT.file_sha256(initial_file),
        "byte_count": initial_file.stat().st_size,
        "initial_state_digest": initialisation["initial_state_digest"],
        "registered_seed": initialisation["registered_seed"],
        "factory_inventory": initialisation["factory_inventory"],
    }, "result initialisation summary changed")
    del initial_model, expected_initial
    checkpoint_file = checkpoint_path(root)
    training = result["training"]
    require(checkpoint_file.is_file() and not checkpoint_file.is_symlink()
            and CONTRACT.file_sha256(checkpoint_file) == training["sha256"]
            and checkpoint_file.stat().st_size == training["byte_count"],
            "final checkpoint bytes changed")
    checkpoint = torch.load(
        checkpoint_file, map_location="cpu", weights_only=False)
    require(checkpoint.get("schema") == CHECKPOINT_SCHEMA
            and checkpoint.get("status") == STATUS
            and checkpoint.get("implementation_digest")
            == CONTRACT.LN.IMPLEMENTATION_DIGEST
            and checkpoint.get("execution_bindings") == bindings
            and checkpoint.get("initial_state_digest")
            == CONTRACT.INITIAL_STATE_DIGEST
            and checkpoint.get("registered_seed") == CONTRACT.SCORER_SEED
            and checkpoint.get("data_order_seed") == CONTRACT.DATA_ORDER_SEED
            and checkpoint.get("completed_epoch") == CONTRACT.EPOCHS
            and checkpoint.get("completed_optimizer_updates")
            == CONTRACT.TOTAL_UPDATES
            and checkpoint.get("example_presentations")
            == CONTRACT.PRESENTATIONS
            and checkpoint.get("epoch_selection")
            == "final_epoch_only_no_selection"
            and checkpoint.get("effective_batch") == CONTRACT.EFFECTIVE_BATCH
            and checkpoint.get("microbatch") == CONTRACT.MICROBATCH
            and checkpoint.get("gradient_accumulation_steps")
            == CONTRACT.GRADIENT_ACCUMULATION_STEPS
            and checkpoint.get("optimizer") == CONTRACT.TRAINING["optimizer"]
            and checkpoint.get("learning_rate_schedule")
            == CONTRACT.TRAINING["learning_rate_schedule"]
            and checkpoint.get("data_order_witness")
            == training["data_order_witness"] == {
                key: CONTRACT.DATA_ORDER[key] for key in (
                    "base_training_view_row_digest_sequence_digest",
                    "permutation_plan_digest", "row_presentation_plan_digest",
                    "final_generator_state_digest")}
            and checkpoint.get("last_epoch_order_digest")
            == CONTRACT.DATA_ORDER["last_epoch_order_digest"]
            and checkpoint.get("final_order_generator_state_digest")
            == CONTRACT.DATA_ORDER["final_generator_state_digest"]
            and checkpoint.get("calibration_metadata_labels_latents_opened") == 0
            and checkpoint.get("diagnostic_smoke_state_used") is False
            and checkpoint.get("technical_trace") == expected_technical_trace()
            and FROZEN.state_dict_digest(checkpoint["model_state_dict"])
            == checkpoint["final_state_digest"] == training["final_state_digest"]
            and FROZEN.structured_digest(checkpoint["optimizer_state_dict"])
            == checkpoint["optimizer_state_digest"]
            == training["optimizer_state_digest"],
            "final checkpoint content changed")
    require(training == {
        "path": str(checkpoint_file),
        "sha256": CONTRACT.file_sha256(checkpoint_file),
        "byte_count": checkpoint_file.stat().st_size,
        "final_state_digest": checkpoint["final_state_digest"],
        "optimizer_state_digest": checkpoint["optimizer_state_digest"],
        "completed_epoch": CONTRACT.EPOCHS,
        "completed_optimizer_updates": CONTRACT.TOTAL_UPDATES,
        "example_presentations": CONTRACT.PRESENTATIONS,
        "attempt_digest": checkpoint["attempt_digest"],
        "registered_seed": CONTRACT.SCORER_SEED,
        "data_order_seed": CONTRACT.DATA_ORDER_SEED,
        "data_order_witness": checkpoint["data_order_witness"],
        "effective_batch": CONTRACT.EFFECTIVE_BATCH,
        "microbatch": CONTRACT.MICROBATCH,
        "gradient_accumulation_steps":
            CONTRACT.GRADIENT_ACCUMULATION_STEPS,
        "optimizer": CONTRACT.TRAINING["optimizer"],
        "learning_rate_schedule": CONTRACT.TRAINING["learning_rate_schedule"],
        "epoch_selection": "final_epoch_only_no_selection",
        "technical_trace_digest": CONTRACT.digest(expected_technical_trace()),
        "training_wall_time_s": checkpoint["training_wall_time_s"],
        "technical_validity": True,
    }, "result training summary changed")
    reload_inventory = strict_checkpoint_reload(
        checkpoint["model_state_dict"], checkpoint)
    require(reload_inventory["compatible_paths"]
            == list(CONTRACT.LN.LAYER_NORM_PATHS),
            "checkpoint consumer factory lost implementation")
    # Seventh bound factory construction: independent implementation/state
    # inventory validation, with no data access or model forward.
    consumer_model = model_factory()
    consumer_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    require(FROZEN.state_dict_digest(FROZEN._cpu_state(consumer_model))
            == checkpoint["final_state_digest"]
            and implementation_inventory(consumer_model)["compatible_paths"]
            == list(CONTRACT.LN.LAYER_NORM_PATHS),
            "independent checkpoint implementation consumer changed")
    del consumer_model
    attempt = validate_signed(read_json(attempt_path(root), "successor attempt"),
                              ATTEMPT_KEY, "successor attempt")
    require(attempt.get("schema") == ATTEMPT_SCHEMA
            and attempt.get("status") == STATUS
            and attempt.get("attempt_number") == 1
            and attempt.get("maximum_attempts") == 1
            and attempt.get("execution_bindings")
            == result["execution_bindings"]
            and attempt.get("initial_state_digest")
            == CONTRACT.INITIAL_STATE_DIGEST
            and attempt.get("registered_seed") == CONTRACT.SCORER_SEED
            and attempt.get("data_order_seed") == CONTRACT.DATA_ORDER_SEED
            and attempt.get("data_order_witness") == {
                key: CONTRACT.DATA_ORDER[key] for key in (
                    "base_training_view_row_digest_sequence_digest",
                    "permutation_plan_digest", "row_presentation_plan_digest",
                    "final_generator_state_digest")}
            and attempt.get("effective_batch") == CONTRACT.EFFECTIVE_BATCH
            and attempt.get("microbatch") == CONTRACT.MICROBATCH
            and attempt.get("gradient_accumulation_steps")
            == CONTRACT.GRADIENT_ACCUMULATION_STEPS
            and attempt.get("fixed_final_epoch") == CONTRACT.EPOCHS
            and attempt.get("epoch_selection")
            == "final_epoch_only_no_selection"
            and attempt.get("optimizer") == CONTRACT.TRAINING["optimizer"]
            and attempt.get("learning_rate_schedule")
            == CONTRACT.TRAINING["learning_rate_schedule"]
            and attempt.get("calibration_metadata_labels_latents_opened") == 0
            and attempt.get("resume_retry_or_replacement_authorised") is False
            and checkpoint.get("attempt_digest") == attempt[ATTEMPT_KEY],
            "successor attempt changed")
    evaluation = validate_signed(read_json(
        evaluation_path(root), "evaluation authority"), EVALUATION_KEY,
        "evaluation authority")
    require(evaluation.get("schema") == EVALUATION_SCHEMA
            and evaluation.get("status") == STATUS
            and evaluation.get("contract_digest")
            == contract[CONTRACT.CONTRACT_SELF_KEY]
            and evaluation.get("execution_bindings")
            == bindings
            and evaluation.get("implementation_digest")
            == CONTRACT.LN.IMPLEMENTATION_DIGEST
            and evaluation.get("final_checkpoint_sha256") == training["sha256"]
            and evaluation.get("final_state_digest") == training["final_state_digest"]
            and evaluation.get("evaluation_number") == 1
            and evaluation.get("maximum_evaluations") == 1
            and evaluation.get("calibration_states")
            == CONTRACT.CALIBRATION_STATES
            and evaluation.get("calibration_rows") == CONTRACT.CALIBRATION_ROWS
            and evaluation.get(
                "calibration_metadata_labels_latents_opened_before_authorisation") == 0
            and evaluation.get(
                "calibration_model_forwards_before_authorisation") == 0
            and evaluation.get(
                "calibration_predictions_before_authorisation") == 0
            and evaluation.get(
                "calibration_metrics_before_authorisation") == 0
            and evaluation.get(
                "persist_closed_prediction_target_evidence") is True
            and evaluation.get(
                "preauthorisation_strict_reload_factory_inventory")
            == reload_inventory
            and result.get("evaluation_authorisation_digest")
            == evaluation[EVALUATION_KEY],
            "evaluation authority changed")
    evidence = validate_signed(read_json(
        evidence_path(root), "calibration evidence"), EVIDENCE_KEY,
        "calibration evidence")
    evidence_binding = result["calibration_evidence"]
    require(evidence_binding == _artifact(evidence_path(root), EVIDENCE_KEY),
            "calibration evidence byte binding changed")
    corpus = PREVIOUS._load_corpus(root)
    baseline, vitg, frozen_trees = PREVIOUS._load_frozen_comparisons(corpus, root)
    metrics, per_family, per_stratum = PREVIOUS.metrics_from_evidence(
        corpus_rows=corpus["calibration_rows"], evidence=evidence,
        bindings=result["execution_bindings"],
        evaluation_digest=evaluation[EVALUATION_KEY],
        checkpoint_sha256=training["sha256"],
        final_state_digest=training["final_state_digest"])
    require(result["results"]["attentive"] == {
        "calibration": metrics, "per_family_calibration": per_family,
        "per_stratum_calibration": per_stratum},
        "closed evidence metrics changed")
    fit_distribution = FROZEN.label_distribution(corpus["fit_rows"])
    calibration_distribution = FROZEN.label_distribution(
        corpus["calibration_rows"])
    criteria, details, pairwise_gain = V13.qualification_criteria(
        metrics, baseline["metrics"], fit_distribution,
        calibration_distribution)
    route = decision(criteria, float(metrics["safety"]["auc_any_hazard"]),
                     float(pairwise_gain))
    comparisons = PREVIOUS._comparison_payload(
        attentive=metrics, attentive_family=per_family,
        baseline=baseline, vitg=vitg)
    family_report = BASE.per_family_primary_consistency(
        attentive=per_family,
        existing_vitl=baseline["vitl_per_family_metrics"],
        baseline=baseline["per_family_metrics"])
    require(result.get("latent_over_baseline_pairwise_gain") == pairwise_gain
            and result.get("metric_comparisons") == comparisons
            and result.get("per_family_primary_consistency") == family_report
            and result.get("frozen_original_gate_replay")
            == {"criteria": criteria, "details": details}
            and result.get("would_meet_all_original_gates")
            is all(criteria.values())
            and result.get("exploratory_decision") == route,
            "metric gates or interpretation changed")
    require(result.get("frozen_metric_tree_digests")
            == contract["frozen_metric_tree_digests"]
            and result["results"]["existing_vitl_frozen"]["calibration"]
            == frozen_trees["vitl"]["overall"]
            and result["results"]["existing_vitl_frozen"][
                "per_family_calibration"] == frozen_trees["vitl"]["per_family"]
            and result["results"]["existing_vitl_frozen"][
                "per_stratum_calibration"] == frozen_trees["vitl"]["per_stratum"]
            and result["results"]["existing_vitl_frozen"]["terminal_digest"]
            == baseline["vitl_terminal_digest"]
            and result["results"]["existing_vitl_frozen"]["safety_auc"]
            == 0.7043234198736978
            and result["results"]["existing_vitl_frozen"][
                "latent_over_baseline_pairwise_gain"] == 0.0317880794701987
            and result["results"]["vitg_frozen"]["calibration"]
            == frozen_trees["vitg"]["overall"]
            and result["results"]["vitg_frozen"]["per_family_calibration"]
            == frozen_trees["vitg"]["per_family"]
            and result["results"]["vitg_frozen"]["per_stratum_calibration"]
            == frozen_trees["vitg"]["per_stratum"]
            and result["results"]["vitg_frozen"]["result_digest"]
            == vitg["exploratory_result_digest"]
            and result["results"]["vitg_frozen"][
                "latent_over_baseline_pairwise_gain"]
            == vitg["latent_over_baseline_pairwise_gain"]
            and result["results"]["vitg_frozen"]["conclusion"]
            == vitg["exploratory_decision"]["classification"]
            and result["results"]["no_latent_reused"]["calibration"]
            == frozen_trees["no_latent"]["overall"]
            and result["results"]["no_latent_reused"][
                "per_family_calibration"] == frozen_trees["no_latent"]["per_family"]
            and result["results"]["no_latent_reused"][
                "per_stratum_calibration"] == frozen_trees["no_latent"]["per_stratum"]
            and all(result["results"]["no_latent_reused"][key] == value
                    for key, value in CONTRACT.NO_LATENT_BASELINE.items()),
            "frozen comparison lineage changed")
    construction = result.get("model_factory_construction_ledger", {})
    require(construction.get("scientific_session_constructions") == 7
            and construction.get("phases") == [
                "fresh_initialisation", "training", "preauth_strict_reload",
                "sole_calibration_evaluation", "prepublication_initial_consumer",
                "prepublication_checkpoint_reload",
                "prepublication_implementation_consumer"]
            and construction.get("all_constructions_use_same_factory") is True
            and construction.get(
                "all_forward_backward_calls_use_exact_externalisation_context")
            is True
            and construction.get(
                "prepublication_checkpoint_reload_inventory")
            == reload_inventory,
            "seven-construction implementation ledger changed")
    resources = result.get("runtime", {})
    require(isinstance(resources.get("wall_time_seconds"), (int, float))
            and math.isfinite(float(resources["wall_time_seconds"]))
            and float(resources["wall_time_seconds"]) > 0.0
            and isinstance(resources.get("peak_vram_bytes"), int)
            and resources["peak_vram_bytes"] > 0,
            "runtime resource receipt changed")
    inventory = runtime_inventory(root)
    require([row["name"] for row in inventory["files"]] == sorted([
        "attempt.json", "calibration_evidence.json", "contract.json",
        "evaluation_authorisation.json", "final_checkpoint.pt",
        "initialisation.pt", "result.json"])
            and all(row["mode"] == 0o444 for row in inventory["files"]),
            "completed runtime namespace inventory changed")
    before_result = {
        "files": [row for row in inventory["files"]
                  if row["name"] != "result.json"],
        "file_count": inventory["file_count"] - 1,
        "total_bytes": inventory["total_bytes"] - next(
            row["byte_count"] for row in inventory["files"]
            if row["name"] == "result.json"),
    }
    require(resources.get("storage_before_result_publication") == before_result,
            "prepublication storage receipt changed")
    return result


def validate_failure(root: Path = ROOT) -> dict[str, Any]:
    """Validate the immutable technical terminal; never resume it."""

    require(failure_path(root).is_file() and not failure_path(root).is_symlink(),
            "failure terminal custody changed")
    contract = load_contract(root)
    failure = validate_signed(read_json(failure_path(root), "successor failure"),
                              FAILURE_KEY, "successor failure")
    require(failure.get("schema") == FAILURE_SCHEMA
            and failure.get("status")
            == "INVALID_TECHNICAL_SCIENTIFIC_SUCCESSOR_ATTEMPT"
            and failure.get("contract_digest")
            == contract[CONTRACT.CONTRACT_SELF_KEY]
            and isinstance(failure.get("stage"), str)
            and isinstance(failure.get("exception_type"), str)
            and isinstance(failure.get("exception"), str)
            and isinstance(failure.get("traceback"), str)
            and failure.get("retry_resume_or_replacement_authorised") is False
            and failure.get("predictor_checkpoints_opened") == 0
            and failure.get("predictor_utility_shards_opened") == 0
            and failure.get("final_200_state_corpus_generated") is False,
            "technical failure envelope changed")
    preserved_invalid_result = (result_path(root).exists()
                                or result_path(root).is_symlink())
    require((not preserved_invalid_result)
            or (failure.get("stage") == "result_validation"
                and result_path(root).is_file()
                and not result_path(root).is_symlink()),
            "result may coexist only as a preserved validation-failed artifact")
    current = runtime_inventory(root)
    require(all(row["mode"] == 0o444 for row in current["files"]),
            "failure namespace contains mutable artifacts")
    before = {
        "files": [row for row in current["files"]
                  if row["name"] != "technical_failure.json"],
        "file_count": current["file_count"] - 1,
        "total_bytes": current["total_bytes"] - next(
            row["byte_count"] for row in current["files"]
            if row["name"] == "technical_failure.json"),
    }
    require(failure.get("runtime_inventory_before_failure") == before,
            "failure prepublication inventory changed")
    presence = failure.get("artifact_presence_before_failure", {})
    require(presence == {
        name: any(row["name"] == filename for row in before["files"])
        for name, filename in RUNTIME_FILES.items() if name != "failure"},
        "failure artifact-presence receipt changed")
    require(presence.get("contract") is True
            and presence.get("result") is preserved_invalid_result,
            "failure stage/result presence changed")
    if presence.get("attempt"):
        attempt = validate_signed(read_json(
            attempt_path(root), "failed successor attempt"), ATTEMPT_KEY,
            "failed successor attempt")
        require(attempt.get("execution_bindings", {}).get("contract_digest")
                == failure["contract_digest"]
                and attempt.get("resume_retry_or_replacement_authorised") is False,
                "failed attempt lineage changed")
    if presence.get("checkpoint"):
        checkpoint = torch.load(
            checkpoint_path(root), map_location="cpu", weights_only=False)
        require(checkpoint.get("schema") == CHECKPOINT_SCHEMA
                and checkpoint.get("implementation_digest")
                == CONTRACT.LN.IMPLEMENTATION_DIGEST
                and checkpoint.get("completed_optimizer_updates")
                == CONTRACT.TOTAL_UPDATES,
                "failure checkpoint content changed")
    require(not presence.get("evaluation") or presence.get("checkpoint"),
            "evaluation authority lacks final checkpoint")
    require(not presence.get("evidence") or presence.get("evaluation"),
            "evidence lacks evaluation authority")
    runtime = failure.get("runtime", {})
    require(isinstance(runtime.get("wall_time_seconds"), (int, float))
            and math.isfinite(float(runtime["wall_time_seconds"]))
            and float(runtime["wall_time_seconds"]) >= 0.0
            and isinstance(runtime.get("peak_vram_bytes"), int)
            and runtime["peak_vram_bytes"] >= 0,
            "failure resource receipt changed")
    return failure


def validate_outcome(root: Path = ROOT) -> dict[str, Any]:
    has_result = result_path(root).exists() or result_path(root).is_symlink()
    has_failure = failure_path(root).exists() or failure_path(root).is_symlink()
    require(has_result or has_failure,
            "a scientific successor terminal is required")
    return validate_failure(root) if has_failure else validate_result(root)


def run_once(root: Path = ROOT) -> dict[str, Any]:
    custody: dict[str, Any] = {
        "stage": "result_validation" if result_path(root).exists()
        else "scientific_preflight",
        "completed_epochs": CONTRACT.EPOCHS if checkpoint_path(root).exists()
        else 0,
        "completed_updates": CONTRACT.TOTAL_UPDATES
        if checkpoint_path(root).exists() else 0,
        "calibration_session_consumed": evidence_path(root).exists(),
        "calibration_completed": evidence_path(root).exists(),
        "closed_evidence_rows": CONTRACT.CALIBRATION_ROWS
        if evidence_path(root).exists() else 0,
    }
    try:
        if failure_path(root).exists() or failure_path(root).is_symlink():
            validate_failure(root)
            raise ScientificSuccessorError(
                "the sole scientific successor attempt is terminally failed")
        if result_path(root).exists() or result_path(root).is_symlink():
            return validate_result(root)
        result = execute_once(root, custody)
        custody["stage"] = "result_validation"
        return validate_result(root)
    except BaseException as exc:
        _record_failure(root, str(custody["stage"]), exc, custody)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("issue-contract", "run", "validate"))
    args = parser.parse_args(argv)
    if args.stage == "issue-contract":
        value = issue_contract(ROOT)
        summary = {"status": value["status"],
                   "contract_digest": value[CONTRACT.CONTRACT_SELF_KEY]}
    elif args.stage == "run":
        value = run_once(ROOT)
        summary = {"status": value["status"],
                   "result_digest": value[RESULT_KEY],
                   "classification": value["exploratory_decision"][
                       "classification"]}
    else:
        value = validate_outcome(ROOT)
        summary = {"status": value["status"],
                   "terminal_digest": value.get(RESULT_KEY,
                                                value.get(FAILURE_KEY)),
                   "validated": True}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
