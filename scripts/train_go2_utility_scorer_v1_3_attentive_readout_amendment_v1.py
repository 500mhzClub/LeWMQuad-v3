#!/usr/bin/env python3
"""Execute the amended sole final-layer attentive-readout experiment.

The prerequisite amendment closes two technically interrupted diagnostics but
does not alter the frozen attentive architecture, data, optimiser, budget,
metrics, or primary thresholds.  A fit-only production smoke is discarded;
the scientific attempt then starts from a freshly reconstructed registered
initial state, trains once, and forwards calibration exactly once.
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
    go2_scorer_failure_attribution_v1_contract as CONTRACT,
)
from lewm.oracle import (  # noqa: E402
    go2_scorer_failure_attribution_v1_prerequisite_amendment as AMENDMENT,
)
from scripts import train_go2_utility_scorer_v1_2 as FROZEN  # noqa: E402
from scripts import train_go2_utility_scorer_v1_3 as V13  # noqa: E402
from scripts import train_go2_utility_scorer_v1_3_attentive_readout_v1 as BASE  # noqa: E402
from scripts import train_go2_utility_scorer_vjepa2_1_vitg_ablation_v1 as VITG  # noqa: E402


STATUS = BASE.STATUS
SCHEMA = "go2_v1_3_final_layer_attentive_readout_amendment_v1_result_v1"
SMOKE_SCHEMA = "go2_v1_3_attentive_readout_fit_only_production_smoke_v1"
SMOKE_CHECKPOINT_SCHEMA = (
    "go2_v1_3_attentive_readout_fit_only_production_smoke_checkpoint_v1")
INITIALISATION_SCHEMA = (
    "go2_v1_3_final_layer_attentive_readout_amendment_v1_initialisation_v1")
ATTEMPT_SCHEMA = (
    "go2_v1_3_final_layer_attentive_readout_amendment_v1_attempt_v1")
CHECKPOINT_SCHEMA = (
    "go2_v1_3_final_layer_attentive_readout_amendment_v1_checkpoint_v1")
EVALUATION_SCHEMA = (
    "go2_v1_3_final_layer_attentive_readout_amendment_v1_evaluation_authorisation_v1")
EVIDENCE_SCHEMA = (
    "go2_v1_3_final_layer_attentive_readout_amendment_v1_calibration_evidence_v1")
FAILURE_SCHEMA = (
    "go2_v1_3_final_layer_attentive_readout_amendment_v1_technical_failure_v1")

SMOKE_SELF_KEY = "production_smoke_digest"
RESULT_SELF_KEY = "attentive_result_digest"
EVIDENCE_SELF_KEY = "calibration_evidence_digest"
EVALUATION_SELF_KEY = "evaluation_authorisation_digest"
ATTEMPT_SELF_KEY = "attempt_digest"
FAILURE_SELF_KEY = "technical_failure_digest"

HORIZONS = BASE.HORIZONS
TOKENS = BASE.TOKENS
TOKEN_DIM = BASE.TOKEN_DIM
EPOCHS = BASE.EPOCHS
FIT_ROWS = BASE.FIT_ROWS
CALIBRATION_ROWS = BASE.CALIBRATION_ROWS
EFFECTIVE_BATCH = BASE.EFFECTIVE_BATCH
MICROBATCH = BASE.MICROBATCH
UPDATES_PER_EPOCH = BASE.UPDATES_PER_EPOCH
TOTAL_UPDATES = BASE.TOTAL_UPDATES
PRESENTATIONS = BASE.PRESENTATIONS
DATA_ORDER_SEED = BASE.DATA_ORDER_SEED
ORIGINAL_SAFETY_AUC = 0.7043234198736978
ORIGINAL_PAIRWISE_GAIN = 0.0317880794701987
SAFETY_AUC_GATE = 0.75
PAIRWISE_GAIN_GATE = Decimal("0.05")
CALIBRATION_FORWARD_BATCHES = CALIBRATION_ROWS // MICROBATCH
FROZEN_METRIC_TREE_DIGESTS = {
    "vitl": {
        "overall": "b6ea983a038fc4013fa928abea137820fcc4caed1da21b267be1e89bb0717e73",
        "per_family": "109cbfe94f5405aed5f8df60c3fb763436e8594b61c228e8a5eeea4f63fb510e",
        "per_stratum": "bdd131a839a655e52e36f535e83e1129721f49559cfe7974bac8acd589803866",
    },
    "vitg": {
        "overall": "af4988c33a9a5248f58dbb105395652acdc04998de40ab6b7da68eb65ef3dc35",
        "per_family": "1fdf284d317db5d7fb768be622a5069b7a578b622a4ba1fa29e81b18538cf59e",
        "per_stratum": "1ac70f9046ced2f8105432fb21fcc47fbc780bceddc14e1aed370d58b3ffbe15",
    },
    "no_latent": {
        "overall": "b880ea86950c8d1f1c6aba522cdaf219cbf745e1d63c102edc750f0a881a5ad5",
        "per_family": "c6c6bb9085e13de8d9c7bfbd2d75fd78252fed451ebf63429f1dc997b8966f1d",
        "per_stratum": "83bd9d67f911a9f778d0c4ca3f08adc673cd9ae6427244bad2f542c8eb1011ce",
    },
}

# Four immutable fit rows are sufficient for the technical production smoke.
# Binding the exact row-record and latent bytes here avoids materialising the
# monolithic 1,440-row view or index and, critically, opens no calibration row.
SMOKE_FIT_FIXTURE = (
    {
        "training_view_row_digest":
            "58363af8146aa1fca52419c3f4b4b337ef169334a5bd968b1f34911a56550e49",
        "branch_identity_digest":
            "1ebcde263734fc6f2e144af65840cc0f1285518ff31d0d2d37b170a0430b8c34",
        "state_id": "scorer_fit-loop_alias_stress-safety_enriched-04",
        "state_identity_digest":
            "05e6f56a4df52a9eab07ff47d4e91f9e610c5d3c580c11d1752d67b03642c64d",
        "candidate_index": 0,
        "source_kind": "V2_VALID_ADOPTION",
        "input": {
            "path": ".generated/go2_branch_corpus_v1_2/scorer_fit/row_records_v2/1ebcde263734fc6f2e144af65840cc0f1285518ff31d0d2d37b170a0430b8c34.json",
            "self_digest":
                "354b85e958952208a2908348a82eb28fb5aaab02d98e3abc7f562da5a88c6d76",
            "sha256":
                "c4c761bbca093ba76b142cb359c7ff7d3fef1223fd7a5c4ee04ea4487cb06fe7",
        },
        "latent": {
            "path": "latents/horizon/58363af8146aa1fca52419c3f4b4b337ef169334a5bd968b1f34911a56550e49.f16",
            "sha256":
                "824c705f17b0c8a5b1fb74eba2ef7c35d0048c22d943ef5da15ca0a7c0ec5829",
        },
    },
    {
        "training_view_row_digest":
            "18137ed8fb1b51741e08d9fed767cb4500eb3592935cf915d3561e3b9a573bef",
        "branch_identity_digest":
            "61e19aa6e3d9251512d2a54de5d64234cfff1554f18977aa750afe179b59ffc3",
        "state_id": "scorer_fit-loop_alias_stress-safety_enriched-04",
        "state_identity_digest":
            "05e6f56a4df52a9eab07ff47d4e91f9e610c5d3c580c11d1752d67b03642c64d",
        "candidate_index": 1,
        "source_kind": "V2_VALID_ADOPTION",
        "input": {
            "path": ".generated/go2_branch_corpus_v1_2/scorer_fit/row_records_v2/61e19aa6e3d9251512d2a54de5d64234cfff1554f18977aa750afe179b59ffc3.json",
            "self_digest":
                "c75ea0fcfad52fde687ec50a8450abdf3a1ee14b91966b6aa3d491de35ba8f30",
            "sha256":
                "4e3da9f84083d91d7a8db9c4dfd11e7f2631dcb93ce633cb130ae65140c34243",
        },
        "latent": {
            "path": "latents/horizon/18137ed8fb1b51741e08d9fed767cb4500eb3592935cf915d3561e3b9a573bef.f16",
            "sha256":
                "658cf4b1b0fcabeb9800245f167a4d3c72a25fed82571a677ab0cc3a5540cea2",
        },
    },
    {
        "training_view_row_digest":
            "154abe9d3c23eece05958846c7cd1682b9a72610d3e638938a8db48f80444f3d",
        "branch_identity_digest":
            "43a604ceac0ca3368f0487e7d7e199a7fa8a69c6ef0e0ec63441a1455f0fe22a",
        "state_id": "scorer_fit-loop_alias_stress-safety_enriched-04",
        "state_identity_digest":
            "05e6f56a4df52a9eab07ff47d4e91f9e610c5d3c580c11d1752d67b03642c64d",
        "candidate_index": 2,
        "source_kind": "V2_VALID_ADOPTION",
        "input": {
            "path": ".generated/go2_branch_corpus_v1_2/scorer_fit/row_records_v2/43a604ceac0ca3368f0487e7d7e199a7fa8a69c6ef0e0ec63441a1455f0fe22a.json",
            "self_digest":
                "0fb671fe1a23a1b7c6672706b3148f2ca152e8413d398837c98266d783626904",
            "sha256":
                "7560208c0a725b9eff84e4941e13bc5da15a09465b41b1b2fc8eaa8bd35a5c11",
        },
        "latent": {
            "path": "latents/horizon/154abe9d3c23eece05958846c7cd1682b9a72610d3e638938a8db48f80444f3d.f16",
            "sha256":
                "cc37f26c577bcd6c96450c14d9b261406f2c14058c9fc24403f64157f5da0d2e",
        },
    },
    {
        "training_view_row_digest":
            "eaff2d60b7001e8e3d1573a6a7603071b655752f3d643cf35853e320ad4e48bb",
        "branch_identity_digest":
            "faa4259c48557590b2e1e86f2a88ac3202f926ceeea5d1ca0c011e2b129b3a1a",
        "state_id": "scorer_fit-loop_alias_stress-safety_enriched-04",
        "state_identity_digest":
            "05e6f56a4df52a9eab07ff47d4e91f9e610c5d3c580c11d1752d67b03642c64d",
        "candidate_index": 3,
        "source_kind": "V2_VALID_ADOPTION",
        "input": {
            "path": ".generated/go2_branch_corpus_v1_2/scorer_fit/row_records_v2/faa4259c48557590b2e1e86f2a88ac3202f926ceeea5d1ca0c011e2b129b3a1a.json",
            "self_digest":
                "4f211f59f41c45b677caa3d1761d1a1c4dceff20d955131a09ce34ffc661cc0e",
            "sha256":
                "4808f93ef4fbf4d88ec10aaaeb6f81b345fc89976d4e0ef590b5359657eb2dda",
        },
        "latent": {
            "path": "latents/horizon/eaff2d60b7001e8e3d1573a6a7603071b655752f3d643cf35853e320ad4e48bb.f16",
            "sha256":
                "13a0a2235904d11885baf2b23545cf00569d1e0503d081a4fc1205813ebdf23e",
        },
    },
)
SMOKE_FIT_FIXTURE_DIGEST = (
    "017e14d40a291380f54cd94e36f99d03970161425fb82c1efeeac1db34536888"
)


class AttentiveAmendmentError(RuntimeError):
    """The amended one-shot execution or frozen input changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttentiveAmendmentError(message)


def signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    require(key not in result, f"{key} already exists")
    result[key] = AMENDMENT.digest(result)
    return result


def validate_signed(value: Mapping[str, Any], key: str,
                    label: str) -> dict[str, Any]:
    result = dict(value)
    recorded = result.pop(key, None)
    require(isinstance(recorded, str) and len(recorded) == 64
            and recorded == AMENDMENT.digest(result),
            f"{label} self digest changed")
    result[key] = recorded
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"{label} is absent or not a regular file")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def publish_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    V13.publish_json_once(path, value, label=label)


def runtime_root(root: Path = ROOT) -> Path:
    return AMENDMENT.amendment_root(root)


def smoke_root(root: Path = ROOT) -> Path:
    return runtime_root(root) / "production_smoke"


def smoke_checkpoint_path(root: Path = ROOT) -> Path:
    return smoke_root(root) / "checkpoint.pt"


def smoke_path(root: Path = ROOT) -> Path:
    return runtime_root(root) / "production_smoke.json"


def smoke_failure_path(root: Path = ROOT) -> Path:
    return runtime_root(root) / "production_smoke_failure.json"


def initialisation_path(root: Path = ROOT) -> Path:
    return runtime_root(root) / "initialisation.pt"


def attempt_root(root: Path = ROOT) -> Path:
    return runtime_root(root) / "training/attempt_000"


def final_checkpoint_path(root: Path = ROOT) -> Path:
    return attempt_root(root) / "final_epoch_060.pt"


def evaluation_path(root: Path = ROOT) -> Path:
    return runtime_root(root) / "evaluation_authorisation.json"


def evidence_path(root: Path = ROOT) -> Path:
    return runtime_root(root) / "calibration_evidence.json"


def result_path(root: Path = ROOT) -> Path:
    return runtime_root(root) / "exploratory_result.json"


def technical_failure_path(root: Path = ROOT) -> Path:
    return runtime_root(root) / "technical_failure.json"


def device_preflight() -> tuple[torch.device, dict[str, Any]]:
    require(torch.cuda.is_available(), "ROCm CUDA compatibility is unavailable")
    count = torch.cuda.device_count()
    require(count == 2, "the frozen workstation must expose exactly two HIP devices")
    properties = torch.cuda.get_device_properties(0)
    name = str(properties.name)
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    require(name == "AMD Radeon AI PRO R9700" and architecture == "gfx1201",
            "cuda:0 is not the frozen R9700 gfx1201 device")
    free, total = torch.cuda.mem_get_info(0)
    return torch.device("cuda:0"), {
        "selected_device": "cuda:0",
        "selected_name": name,
        "selected_architecture": architecture,
        "visible_hip_device_count": count,
        "visible_device_names": [
            str(torch.cuda.get_device_properties(index).name)
            for index in range(count)
        ],
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "free_vram_bytes_before_stage": int(free),
        "total_vram_bytes": int(total),
    }


def _load_corpus(root: Path) -> dict[str, Any]:
    bundle = V13.load_preserved_encoded_training_view_for_replacement(
        root=root, verify_encoder_checkpoint=False)
    return V13.corpus_from_encoded_bundle({**bundle, "root": root})


def _load_fit_training_corpus(root: Path) -> dict[str, Any]:
    """Validate/materialise fit rows while never reading calibration latents."""

    V13.ENCODER._require_registered_generated_root(root)
    authority, _correction, _corpus, manifest = (
        V13._validate_preserved_workflow_inputs_for_replacement(root=root))
    view = V13._materialise_preserved_training_view_for_replacement(
        root=root, authority=authority, manifest=manifest)
    index_path = V13.ENCODER.latent_index_path(root)
    receipt_path = V13.ENCODER.encoding_receipt_path(root)
    raw = V13.CONTRACT.SCORER_TRAINING_INTEGRITY_REPLACEMENT[
        "preserved_raw_bindings"]
    index_binding = raw["encoded_training_view/latent_index.json"]
    receipt_binding = raw["encoded_training_view/encoding_receipt.json"]
    require(index_path.is_file() and not index_path.is_symlink()
            and index_path.stat().st_size == index_binding["byte_count"]
            and BASE.file_sha256(index_path) == index_binding["sha256"]
            and receipt_path.is_file() and not receipt_path.is_symlink()
            and receipt_path.stat().st_size == receipt_binding["byte_count"]
            and BASE.file_sha256(receipt_path) == receipt_binding["sha256"],
            "frozen encoded-view global receipts changed")
    index = V13.ENCODER._validate_signed(
        json.loads(index_path.read_text()), V13.ENCODER.LATENT_INDEX_SELF_KEY,
        "fit-only scientific latent index")
    receipt = V13.ENCODER._validate_signed(
        json.loads(receipt_path.read_text()),
        V13.ENCODER.ENCODING_RECEIPT_SELF_KEY,
        "fit-only scientific encoding receipt")
    view_self_key = getattr(
        V13.ENCODER.WORKFLOW, "TRAINING_VIEW_SELF_KEY", "training_view_digest")
    require(index[V13.ENCODER.LATENT_INDEX_SELF_KEY]
            == CONTRACT.FROZEN_LATENT_INDEX_DIGEST
            and index.get("training_view_digest")
            == view[view_self_key] == CONTRACT.FROZEN_TRAINING_VIEW_DIGEST
            and index.get("row_count") == 1_440
            and index.get("fit_rows") == FIT_ROWS
            and index.get("calibration_rows") == CALIBRATION_ROWS
            and index.get("horizon_shape")
            == [1_440, HORIZONS, TOKENS, TOKEN_DIM]
            and receipt.get("latent_index_digest")
            == CONTRACT.FROZEN_LATENT_INDEX_DIGEST
            and receipt.get(V13.ENCODER.ENCODING_RECEIPT_SELF_KEY)
            == V13.CONTRACT.SCORER_TRAINING_INTEGRITY_REPLACEMENT[
                "frozen_scientific_inputs"]["encoding_receipt_digest"],
            "fit-only scientific global latent binding changed")
    records = index.get("horizon_records")
    require(isinstance(records, list) and len(records) == 1_440,
            "fit-only scientific latent ledger changed")
    positions = {
        str(record.get("training_view_row_digest")): position
        for position, record in enumerate(records)
    }
    require(len(positions) == 1_440,
            "fit-only scientific latent identities are duplicated")
    fit_rows = []
    encoded_root = V13.ENCODER.encoded_root(root)
    for row in view["rows"]:
        if V13.ENCODER._row_role(row) != "fit":
            continue
        row_digest = str(row["training_view_row_digest"])
        require(row_digest in positions,
                "fit row lacks its frozen latent record")
        position = positions[row_digest]
        record = records[position]
        relative = V13.ENCODER._safe_relative(
            record.get("path"), label="fit latent")
        require(V13.ENCODER._valid_latent_record(
                    encoded_root / relative, record, row),
                f"fit latent shard changed for {row_digest}")
        fit_rows.append(V13._normalised_row(row, position))
    fit_rows.sort(key=lambda row: (
        str(row["state_id"]), int(row["candidate_index"])))
    require(len(fit_rows) == FIT_ROWS
            and len({row["state_id"] for row in fit_rows}) == 96,
            "fit-only scientific corpus changed")
    return {
        "view": view, "index": index, "receipt": receipt,
        "fit_rows": fit_rows,
        "horizon": FROZEN.HorizonShardStore(records, encoded_root),
        "calibration_latent_shards_read": 0,
    }


def _load_frozen_comparisons(
        corpus: Mapping[str, Any], root: Path,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline = VITG.validate_reused_no_latent_baseline(corpus, root=root)
    terminal = V13._validate_signed(
        V13._read_json(V13.qualification_path(root), label="frozen ViT-L terminal"),
        V13.QUALIFICATION_SELF_KEY, "frozen ViT-L terminal")
    vitg = BASE._load_frozen_vitg_result(root=root)
    trees = {
        "vitl": {
            "overall": terminal["results"]["latent"]["calibration"],
            "per_family": terminal["results"]["latent"][
                "per_family_calibration"],
            "per_stratum": terminal["results"]["latent"][
                "per_stratum_calibration"],
        },
        "vitg": {
            "overall": vitg["results"]["vitg"]["calibration"],
            "per_family": vitg["results"]["vitg"][
                "per_family_calibration"],
            "per_stratum": vitg["results"]["vitg"][
                "per_stratum_calibration"],
        },
        "no_latent": {
            "overall": terminal["results"]["no_latent"]["calibration"],
            "per_family": terminal["results"]["no_latent"][
                "per_family_calibration"],
            "per_stratum": terminal["results"]["no_latent"][
                "per_stratum_calibration"],
        },
    }
    observed = {
        model: {scope: AMENDMENT.digest(value)
                for scope, value in scopes.items()}
        for model, scopes in trees.items()
    }
    require(observed == FROZEN_METRIC_TREE_DIGESTS,
            "frozen comparison metric trees changed")
    require(trees["vitl"]["overall"] == baseline["vitl_metrics"]
            and trees["vitl"]["per_family"]
            == baseline["vitl_per_family_metrics"]
            and trees["no_latent"]["overall"] == baseline["metrics"]
            and trees["no_latent"]["per_family"]
            == baseline["per_family_metrics"],
            "frozen comparison validators disagree")
    return baseline, vitg, trees


def _fit_only_smoke_fixture(root: Path) -> tuple[list[dict[str, Any]], Any,
                                                  dict[str, Any]]:
    """Open exactly four frozen fit row records and four fit latent shards."""

    require(AMENDMENT.digest(SMOKE_FIT_FIXTURE) == SMOKE_FIT_FIXTURE_DIGEST,
            "fit-only smoke fixture changed")
    workflow = V13.ENCODER.WORKFLOW
    encoded_root = V13.ENCODER.encoded_root(root)
    rows: list[dict[str, Any]] = []
    latent_records = []
    bindings = []
    for position, frozen in enumerate(SMOKE_FIT_FIXTURE):
        input_row, resolved = workflow._resolve_bound_input(frozen["input"])
        require(resolved.is_file() and not resolved.is_symlink()
                and BASE.file_sha256(resolved) == frozen["input"]["sha256"]
                and input_row.get("branch_row_digest")
                == frozen["input"]["self_digest"]
                and input_row.get("split_role") == "fit"
                and input_row.get("state_id") == frozen["state_id"]
                and input_row.get("state_identity_digest")
                == frozen["state_identity_digest"]
                and input_row.get("branch_identity_digest")
                == frozen["branch_identity_digest"]
                and input_row.get("candidate_index")
                == frozen["candidate_index"],
                "fit-only smoke row binding changed")
        record = {
            "training_view_row_digest": frozen["training_view_row_digest"],
            "state_id": frozen["state_id"],
            "state_identity_digest": frozen["state_identity_digest"],
            "candidate_index": frozen["candidate_index"],
            "source_kind": frozen["source_kind"],
            "path": frozen["latent"]["path"],
            "sha256": frozen["latent"]["sha256"],
            "byte_count": HORIZONS * TOKENS * TOKEN_DIM * 2,
            "shape": [HORIZONS, TOKENS, TOKEN_DIM],
        }
        latent_path = encoded_root / record["path"]
        require(V13.ENCODER._valid_latent_record(
                    latent_path, record, {**frozen, "source_kind":
                                          frozen["source_kind"]}),
                "fit-only smoke latent binding changed")
        rows.append({
            "training_view_row_digest": frozen["training_view_row_digest"],
            "branch_identity_digest": frozen["branch_identity_digest"],
            "state_id": frozen["state_id"],
            "state_identity_digest": frozen["state_identity_digest"],
            "candidate_index": frozen["candidate_index"],
            "source_kind": frozen["source_kind"],
            "action_blocks": input_row["action_blocks"],
            "goal_binding_input": input_row["goal_binding_input"],
            "progress": input_row["progress"],
            "safety": input_row["safety"],
            "completion": input_row["completion"],
            "_latent_index": position,
        })
        latent_records.append(record)
        bindings.append({
            "training_view_row_digest": frozen["training_view_row_digest"],
            "branch_identity_digest": frozen["branch_identity_digest"],
            "row_record_path": frozen["input"]["path"],
            "row_record_sha256": frozen["input"]["sha256"],
            "row_record_self_digest": frozen["input"]["self_digest"],
            "latent_path": frozen["latent"]["path"],
            "latent_sha256": frozen["latent"]["sha256"],
        })
    require(len(rows) == MICROBATCH
            and len({row["training_view_row_digest"] for row in rows})
            == MICROBATCH,
            "fit-only smoke fixture cardinality changed")
    store = FROZEN.HorizonShardStore(latent_records, encoded_root)
    return rows, store, {
        "fixture_digest": SMOKE_FIT_FIXTURE_DIGEST,
        "row_count": MICROBATCH,
        "row_record_files_opened": MICROBATCH,
        "fit_latent_shards_opened": MICROBATCH,
        "calibration_rows_materialized": 0,
        "calibration_label_rows_opened": 0,
        "calibration_latent_shards_opened": 0,
        "global_training_view_digest": CONTRACT.FROZEN_TRAINING_VIEW_DIGEST,
        "global_latent_index_digest": CONTRACT.FROZEN_LATENT_INDEX_DIGEST,
        "registered_data_order_contract_digest": AMENDMENT.digest(
            CONTRACT.DATA_ORDER_CONTRACT),
        "files": bindings,
    }


def _source_initialisation(root: Path) -> dict[str, Any]:
    terminal = V13._validate_signed(
        V13._read_json(V13.qualification_path(root),
                       label="frozen ViT-L terminal"),
        V13.QUALIFICATION_SELF_KEY, "frozen ViT-L terminal")
    require(terminal[V13.QUALIFICATION_SELF_KEY]
            == BASE.FROZEN_VITL_TERMINAL_DIGEST
            and terminal.get("terminal_kind") == "QUALIFICATION_FAILURE"
            and terminal.get("qualified") is False,
            "frozen ViT-L terminal changed")
    receipt = terminal["initialisations"]["latent"]
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = root / path
    require(path.is_file() and not path.is_symlink()
            and BASE.file_sha256(path) == receipt["sha256"],
            "frozen ViT-L initialisation bytes changed")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    require(FROZEN.state_dict_digest(artifact["model_state_dict"])
            == receipt["initial_state_digest"],
            "frozen ViT-L initial state changed")
    return {
        "path": path,
        "sha256": receipt["sha256"],
        "state_digest": receipt["initial_state_digest"],
    }


def _fresh_model_state() -> tuple[BASE.FinalLayerAttentiveUtilityScorer,
                                  dict[str, torch.Tensor], str]:
    FROZEN.configure_determinism(CONTRACT.ATTENTIVE_SEED)
    model = BASE.FinalLayerAttentiveUtilityScorer()
    state = FROZEN._cpu_state(model)
    require(sum(parameter.numel() for parameter in model.parameters())
            == CONTRACT.ATTENTIVE_READOUT_ARCHITECTURE[
                "trainable_parameter_count"],
            "attentive parameter count changed")
    return model, state, FROZEN.state_dict_digest(state)


def build_initialisation(*, amendment: Mapping[str, Any],
                         root: Path = ROOT) -> dict[str, Any]:
    source = _source_initialisation(root)
    _model, state, state_digest = _fresh_model_state()
    payload = {
        "schema": INITIALISATION_SCHEMA,
        "status": STATUS,
        "prerequisite_amendment_digest": amendment[AMENDMENT.SELF_KEY],
        "registered_seed": CONTRACT.ATTENTIVE_SEED,
        "architecture_seed_digest": CONTRACT.ATTENTIVE_SEED_KEY_DIGEST,
        "source_vitl_initialisation_sha256": source["sha256"],
        "source_vitl_initial_state_digest": source["state_digest"],
        "model_state_dict": state,
        "initial_state_digest": state_digest,
        "parameter_initialisation": {
            "algorithm": (
                "construct the complete frozen attentive architecture once "
                "after configure_determinism(architecture_seed)"),
            "all_trainable_parameters_use_architecture_seed": True,
            "copied_predecessor_parameter_count": 0,
            "source_vitl_initialisation_is_lineage_only": True,
            "nontrainable_horizon_embedding_digest": FROZEN.tensor_digest(
                state["horizon_embeddings"]),
        },
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in _model.parameters()),
        "all_trainable_parameters_use_architecture_seed": True,
        "copied_predecessor_parameter_count": 0,
        "smoke_state_reused": False,
    }
    path = initialisation_path(root)
    if path.exists() or path.is_symlink():
        require(path.is_file() and not path.is_symlink(),
                "scientific initialisation path changed")
        installed = torch.load(path, map_location="cpu", weights_only=False)
        require(installed.keys() == payload.keys()
                and all(key == "model_state_dict" or installed[key] == value
                        for key, value in payload.items())
                and all(torch.equal(installed["model_state_dict"][key], value)
                        for key, value in state.items()),
                "scientific initialisation changed")
        return installed
    path.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.atomic_torch_save(payload, path)
    return payload


def _loss(progress: torch.Tensor, safety: torch.Tensor,
          completion: torch.Tensor, targets: Mapping[str, torch.Tensor],
          indices: torch.Tensor) -> torch.Tensor:
    return (
        F.mse_loss(progress, targets["progress"][indices], reduction="sum")
        + F.binary_cross_entropy_with_logits(
            safety, targets["safety"][indices], reduction="sum")
        + F.binary_cross_entropy_with_logits(
            completion, targets["completion"][indices], reduction="sum")
    ) / EFFECTIVE_BATCH


def _smoke_gradient_evidence(
        model: BASE.FinalLayerAttentiveUtilityScorer,
        outputs: Sequence[torch.Tensor]) -> dict[str, Any]:
    names = ("progress", "safety", "completion")
    require(len(outputs) == len(names)
            and all(tuple(value.shape) == (MICROBATCH,)
                    and bool(torch.isfinite(value).all().item())
                    for value in outputs),
            "fit-only smoke output shape or finiteness changed")

    projection = model.token_projection.weight.grad
    require(projection is not None
            and bool(torch.isfinite(projection).all().item()),
            "token-projection smoke gradient is absent or non-finite")
    projection_norm = float(torch.linalg.vector_norm(
        projection.detach().to(torch.float64)).cpu())
    require(math.isfinite(projection_norm) and projection_norm > 0.0,
            "token-projection smoke gradient is zero")

    attention_receipts = []
    attention_squared = torch.zeros(
        (), dtype=torch.float64, device=projection.device)
    for name, parameter in model.named_parameters():
        selected = (
            name.startswith("pooler.cross_attention_block.xattn.")
            or (name.startswith("pooler.blocks.") and ".attn." in name)
        )
        if not selected:
            continue
        gradient = parameter.grad
        require(gradient is not None
                and bool(torch.isfinite(gradient).all().item()),
                f"attention smoke gradient is absent or non-finite: {name}")
        gradient64 = gradient.detach().to(torch.float64)
        attention_squared += gradient64.square().sum()
        attention_receipts.append({
            "path": name,
            "shape": list(gradient.shape),
            "gradient_digest": FROZEN.tensor_digest(gradient.detach().cpu()),
            "gradient_norm": float(torch.linalg.vector_norm(
                gradient64).cpu()),
        })
    attention_norm = float(torch.sqrt(attention_squared).cpu())
    require(attention_receipts and math.isfinite(attention_norm)
            and attention_norm > 0.0,
            "attention smoke gradient is zero")

    query = model.pooler.query_tokens.grad
    require(query is not None and tuple(query.shape) == (1, 3, 512)
            and bool(torch.isfinite(query).all().item()),
            "component-query smoke gradients changed")
    query_rows = query.detach()[0]
    query_receipts = []
    for index, name in enumerate(names):
        norm = float(torch.linalg.vector_norm(
            query_rows[index].to(torch.float64)).cpu())
        require(math.isfinite(norm) and norm > 0.0,
                f"{name} query smoke gradient is zero")
        query_receipts.append({
            "component": name,
            "shape": [512],
            "gradient_norm": norm,
            "gradient_digest": FROZEN.tensor_digest(
                query_rows[index].cpu()),
        })
    pairwise_distinct = all(
        not torch.equal(query_rows[left], query_rows[right])
        for left in range(3) for right in range(left + 1, 3)
    )
    require(pairwise_distinct, "component-query smoke gradients are not distinct")
    return {
        "output_shapes": {name: list(value.shape)
                          for name, value in zip(names, outputs, strict=True)},
        "all_outputs_finite": True,
        "token_projection_weight": {
            "shape": list(projection.shape),
            "gradient_norm": projection_norm,
            "gradient_digest": FROZEN.tensor_digest(projection.detach().cpu()),
        },
        "attention": {
            "aggregate_gradient_norm": attention_norm,
            "parameter_count": len(attention_receipts),
            "parameters": attention_receipts,
        },
        "component_queries": {
            "gradient_shape": [3, 512],
            "all_nonzero_finite": True,
            "all_pairwise_distinct": True,
            "queries": query_receipts,
        },
    }


def _record_failure(*, path: Path, stage: str, error: BaseException,
                    epochs: int, updates: int,
                    calibration_evaluations: int = 0,
                    calibration_evaluation_completed: bool = False,
                    closed_evidence_rows: int = 0) -> None:
    if path.exists() or path.is_symlink():
        return
    payload = signed({
        "schema": FAILURE_SCHEMA,
        "status": "INVALID_TECHNICAL_ATTENTIVE_AMENDMENT_EXECUTION",
        "complete": True,
        "stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "completed_epochs": epochs,
        "completed_optimizer_updates": updates,
        "retry_resume_or_replacement_authorised": False,
        "calibration_evaluations": calibration_evaluations,
        "calibration_evaluation_completed": calibration_evaluation_completed,
        "closed_calibration_evidence_rows": closed_evidence_rows,
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, FAILURE_SELF_KEY)
    publish_json(path, payload, "attentive amendment technical failure")


def run_production_smoke(root: Path = ROOT) -> dict[str, Any]:
    if smoke_path(root).exists() or smoke_path(root).is_symlink():
        return validate_production_smoke(root)
    require(not smoke_failure_path(root).exists()
            and not smoke_failure_path(root).is_symlink()
            and not smoke_root(root).exists()
            and not smoke_root(root).is_symlink(),
            "the sole production smoke was consumed")
    amendment = AMENDMENT.validate_amendment(root)
    stage = "fit_only_smoke_preflight"
    smoke_updates = 0
    try:
        device, preflight = device_preflight()
        fit_rows, fit_store, fixture_binding = _fit_only_smoke_fixture(root)
        source = _source_initialisation(root)
        smoke_root(root).mkdir(parents=True, exist_ok=False)
        stage = "fit_only_smoke_update"
        model, initial_state, initial_digest = _fresh_model_state()
        model.to(device)
        budget = BASE.frozen_budget()
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=float(budget["lr"]),
            weight_decay=float(budget["weight_decay"]))
        action_goal, targets = BASE._small_features(fit_rows, device)
        batch_cpu = torch.arange(MICROBATCH, dtype=torch.int64)
        batch = batch_cpu.to(device)
        tokens = BASE._token_batch(
            fit_rows, fit_store, batch_cpu.tolist(), device)
        optimiser.zero_grad(set_to_none=True)
        progress, safety, completion = model(tokens, action_goal[batch])
        loss = _loss(progress, safety, completion, targets, batch)
        require(bool(torch.isfinite(loss).item()), "smoke loss is non-finite")
        loss.backward()
        gradient_evidence = _smoke_gradient_evidence(
            model, (progress, safety, completion))
        gradient_squared = torch.zeros((), dtype=torch.float64, device=device)
        for parameter in model.parameters():
            if parameter.grad is not None:
                require(bool(torch.isfinite(parameter.grad).all().item()),
                        "smoke gradient is non-finite")
                gradient_squared += parameter.grad.detach().to(
                    torch.float64).square().sum()
        gradient_norm = float(torch.sqrt(gradient_squared).detach().cpu())
        require(math.isfinite(gradient_norm) and gradient_norm > 0.0,
                "smoke gradient is zero or non-finite")
        nn.utils.clip_grad_norm_(model.parameters(), float(budget["grad_clip"]))
        optimiser.step()
        smoke_updates = 1
        updated_state = FROZEN._cpu_state(model)
        updated_digest = FROZEN.state_dict_digest(updated_state)
        optimiser_state = optimiser.state_dict()
        optimiser_digest = FROZEN.structured_digest(optimiser_state)
        require(updated_digest != initial_digest,
                "fit-only smoke did not update model state")
        checkpoint = {
            "schema": SMOKE_CHECKPOINT_SCHEMA,
            "status": "NON_SCIENTIFIC_FIT_ONLY_TECHNICAL_SMOKE",
            "prerequisite_amendment_digest": amendment[AMENDMENT.SELF_KEY],
            "registered_seed": CONTRACT.ATTENTIVE_SEED,
            "source_vitl_initialisation_sha256": source["sha256"],
            "initial_state_digest": initial_digest,
            "updated_state_digest": updated_digest,
            "model_state_dict": updated_state,
            "optimizer_state_dict": optimiser_state,
            "optimizer_state_digest": optimiser_digest,
            "fit_row_digests": [
                fit_rows[int(index)]["training_view_row_digest"]
                for index in batch_cpu.tolist()
            ],
            "fit_only_fixture_binding": fixture_binding,
            "gradient_evidence": gradient_evidence,
            "loss": float(loss.detach().cpu()),
            "preclip_gradient_norm": gradient_norm,
            "optimizer_updates": 1,
            "scientific_attempt": False,
        }
        FROZEN.atomic_torch_save(checkpoint, smoke_checkpoint_path(root))
        reloaded = torch.load(
            smoke_checkpoint_path(root), map_location="cpu", weights_only=False)
        reload_model, _, _ = _fresh_model_state()
        reload_optimiser = torch.optim.AdamW(
            reload_model.parameters(), lr=float(budget["lr"]),
            weight_decay=float(budget["weight_decay"]))
        reload_model.load_state_dict(reloaded["model_state_dict"], strict=True)
        reload_optimiser.load_state_dict(reloaded["optimizer_state_dict"])
        require(FROZEN.state_dict_digest(FROZEN._cpu_state(reload_model))
                == updated_digest
                and FROZEN.structured_digest(reload_optimiser.state_dict())
                == optimiser_digest,
                "fit-only smoke checkpoint reload changed state")
        receipt = signed(FROZEN._safe_json({
            "schema": SMOKE_SCHEMA,
            "status": "PASS_NON_SCIENTIFIC_FIT_ONLY_PRODUCTION_SMOKE",
            "complete": True,
            "prerequisite_amendment_digest": amendment[AMENDMENT.SELF_KEY],
            "device_preflight": preflight,
            "actual_scorer_architecture_digest": AMENDMENT.digest(
                CONTRACT.ATTENTIVE_READOUT_ARCHITECTURE),
            "official_pooler_binding_digest":
                BASE.OFFICIAL_POOLER_BINDING_DIGEST,
            "registered_seed": CONTRACT.ATTENTIVE_SEED,
            "initial_state_digest": initial_digest,
            "updated_state_digest": updated_digest,
            "optimizer_state_digest": optimiser_digest,
            "optimizer_updates": 1,
            "preclip_gradient_norm": gradient_norm,
            "fit_only_fixture_binding": fixture_binding,
            "gradient_evidence": gradient_evidence,
            "real_fit_batch_rows": MICROBATCH,
            "fit_row_digests": checkpoint["fit_row_digests"],
            "checkpoint": {
                "path": str(smoke_checkpoint_path(root)),
                "sha256": BASE.file_sha256(smoke_checkpoint_path(root)),
                "byte_count": smoke_checkpoint_path(root).stat().st_size,
                "reload_exact": True,
            },
            "calibration_latent_rows_opened": 0,
            "calibration_rows_materialized": 0,
            "calibration_label_rows_opened": 0,
            "calibration_latent_shards_opened": 0,
            "calibration_evaluations": 0,
            "scientific_training_attempts": 0,
            "smoke_model_and_optimizer_discarded": True,
            "smoke_state_reuse_permitted": False,
            "fresh_seed_initialisation_required_for_scientific_attempt": True,
            "predictor_checkpoints_opened": 0,
        }), SMOKE_SELF_KEY)
        publish_json(smoke_path(root), receipt, "production smoke receipt")
        del tokens, model, optimiser, reload_model, reload_optimiser
        torch.cuda.empty_cache()
        return validate_production_smoke(root)
    except BaseException as exc:
        _record_failure(path=smoke_failure_path(root), stage=stage,
                        error=exc, epochs=0, updates=smoke_updates)
        raise


def validate_production_smoke(root: Path = ROOT) -> dict[str, Any]:
    amendment = AMENDMENT.validate_amendment(root)
    receipt = validate_signed(
        read_json(smoke_path(root), "production smoke"),
        SMOKE_SELF_KEY, "production smoke")
    checkpoint_path = smoke_checkpoint_path(root)
    gradient = receipt.get("gradient_evidence", {})
    fixture = receipt.get("fit_only_fixture_binding", {})
    query_receipts = gradient.get("component_queries", {}).get("queries", [])
    require(receipt.get("schema") == SMOKE_SCHEMA
            and receipt.get("status")
            == "PASS_NON_SCIENTIFIC_FIT_ONLY_PRODUCTION_SMOKE"
            and receipt.get("complete") is True
            and receipt.get("prerequisite_amendment_digest")
            == amendment[AMENDMENT.SELF_KEY]
            and receipt.get("official_pooler_binding_digest")
            == BASE.OFFICIAL_POOLER_BINDING_DIGEST
            and receipt.get("registered_seed") == CONTRACT.ATTENTIVE_SEED
            and receipt.get("optimizer_updates") == 1
            and float(receipt.get("preclip_gradient_norm", 0.0)) > 0.0
            and receipt.get("real_fit_batch_rows") == MICROBATCH
            and receipt.get("calibration_latent_rows_opened") == 0
            and receipt.get("calibration_rows_materialized") == 0
            and receipt.get("calibration_label_rows_opened") == 0
            and receipt.get("calibration_latent_shards_opened") == 0
            and receipt.get("calibration_evaluations") == 0
            and receipt.get("scientific_training_attempts") == 0
            and receipt.get("smoke_model_and_optimizer_discarded") is True
            and receipt.get("smoke_state_reuse_permitted") is False
            and fixture.get("fixture_digest") == SMOKE_FIT_FIXTURE_DIGEST
            and fixture.get("row_count") == MICROBATCH
            and fixture.get("row_record_files_opened") == MICROBATCH
            and fixture.get("fit_latent_shards_opened") == MICROBATCH
            and fixture.get("calibration_rows_materialized") == 0
            and fixture.get("calibration_label_rows_opened") == 0
            and fixture.get("calibration_latent_shards_opened") == 0
            and gradient.get("output_shapes") == {
                "progress": [MICROBATCH], "safety": [MICROBATCH],
                "completion": [MICROBATCH]}
            and gradient.get("all_outputs_finite") is True
            and float(gradient.get("token_projection_weight", {}).get(
                "gradient_norm", 0.0)) > 0.0
            and float(gradient.get("attention", {}).get(
                "aggregate_gradient_norm", 0.0)) > 0.0
            and gradient.get("attention", {}).get("parameter_count", 0) > 0
            and gradient.get("component_queries", {}).get(
                "gradient_shape") == [3, 512]
            and gradient.get("component_queries", {}).get(
                "all_nonzero_finite") is True
            and gradient.get("component_queries", {}).get(
                "all_pairwise_distinct") is True
            and len(query_receipts) == 3
            and len({row.get("gradient_digest")
                     for row in query_receipts}) == 3
            and checkpoint_path.is_file() and not checkpoint_path.is_symlink()
            and BASE.file_sha256(checkpoint_path)
            == receipt["checkpoint"]["sha256"],
            "production smoke receipt changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint.get("schema") == SMOKE_CHECKPOINT_SCHEMA
            and checkpoint.get("prerequisite_amendment_digest")
            == amendment[AMENDMENT.SELF_KEY]
            and checkpoint.get("optimizer_updates") == 1
            and checkpoint.get("scientific_attempt") is False
            and checkpoint.get("fit_only_fixture_binding") == fixture
            and checkpoint.get("gradient_evidence") == gradient
            and FROZEN.state_dict_digest(checkpoint["model_state_dict"])
            == checkpoint.get("updated_state_digest")
            == receipt["updated_state_digest"]
            and FROZEN.structured_digest(checkpoint["optimizer_state_dict"])
            == checkpoint.get("optimizer_state_digest")
            == receipt["optimizer_state_digest"],
            "production smoke checkpoint changed")
    return receipt


def _execution_bindings(amendment: Mapping[str, Any],
                        smoke: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prerequisite_amendment_digest": amendment[AMENDMENT.SELF_KEY],
        "production_smoke_digest": smoke[SMOKE_SELF_KEY],
        "frozen_diagnostic_contract_digest":
            AMENDMENT.DIAGNOSTIC_CONTRACT_DIGEST,
        "safety_marker_receipt_set_digest":
            AMENDMENT.SAFETY_MARKER_RECEIPT_SET_DIGEST,
        "safety_trace_row_receipt_set_digest":
            AMENDMENT.SAFETY_ROW_RECEIPT_SET_DIGEST,
        "latent_failure_digest": AMENDMENT.LATENT_FAILURE_DIGEST,
        "latent_exception_binding_sha256":
            AMENDMENT.LATENT_EXCEPTION_BINDING_SHA256,
    }


def train_once(model: BASE.FinalLayerAttentiveUtilityScorer, *,
               rows: list[dict[str, Any]], store: Any,
               initialisation: Mapping[str, Any],
               bindings: Mapping[str, Any],
               data_order_witness: Mapping[str, Any],
               device: torch.device, root: Path) -> tuple[dict[str, torch.Tensor],
                                                          dict[str, Any]]:
    directory = attempt_root(root)
    require(not directory.exists() and not directory.is_symlink(),
            "the sole scientific attentive attempt was consumed")
    directory.mkdir(parents=True, exist_ok=False)
    attempt = signed({
        "schema": ATTEMPT_SCHEMA,
        "status": STATUS,
        "attempt_number": 1,
        "maximum_attempts": 1,
        "execution_bindings": dict(bindings),
        "initial_state_digest": initialisation["initial_state_digest"],
        "registered_seed": CONTRACT.ATTENTIVE_SEED,
        "data_order_seed": DATA_ORDER_SEED,
        "data_order_witness": {key: data_order_witness[key] for key in (
            "base_training_view_row_digest_sequence_digest",
            "permutation_plan_digest", "row_presentation_plan_digest",
            "final_generator_state_digest")},
        "effective_batch": EFFECTIVE_BATCH,
        "microbatch": MICROBATCH,
        "gradient_accumulation_steps": EFFECTIVE_BATCH // MICROBATCH,
        "fixed_final_epoch": EPOCHS,
        "resume_source": None,
        "retry_or_replacement_authorised": False,
        "calibration_opened": False,
        "smoke_checkpoint_used": False,
    }, ATTEMPT_SELF_KEY)
    publish_json(directory / "attempt.json", attempt,
                 "attentive amendment attempt")
    completed_epochs = completed_updates = 0
    started = time.time()
    try:
        budget = BASE.frozen_budget()
        FROZEN.configure_determinism(CONTRACT.ATTENTIVE_SEED)
        model.load_state_dict(initialisation["model_state_dict"], strict=True)
        model.to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=float(budget["lr"]),
            weight_decay=float(budget["weight_decay"]))
        action_goal, targets = BASE._small_features(rows, device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(DATA_ORDER_SEED)
        technical_trace = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            order = torch.randperm(FIT_ROWS, generator=generator)
            epoch_updates = 0
            for start in range(0, FIT_ROWS, EFFECTIVE_BATCH):
                batch_cpu = order[start:start + EFFECTIVE_BATCH]
                require(len(batch_cpu) == EFFECTIVE_BATCH,
                        "effective training batch changed")
                optimiser.zero_grad(set_to_none=True)
                for micro_start in range(0, EFFECTIVE_BATCH, MICROBATCH):
                    micro_cpu = batch_cpu[micro_start:micro_start + MICROBATCH]
                    micro = micro_cpu.to(device)
                    tokens = BASE._token_batch(
                        rows, store, micro_cpu.tolist(), device)
                    progress, safety, completion = model(
                        tokens, action_goal[micro])
                    loss = _loss(
                        progress, safety, completion, targets, micro)
                    require(bool(torch.isfinite(loss).item()),
                            "attentive training loss is non-finite")
                    loss.backward()
                    del tokens
                require(all(parameter.grad is None
                            or bool(torch.isfinite(parameter.grad).all().item())
                            for parameter in model.parameters()),
                        "attentive gradient is non-finite")
                nn.utils.clip_grad_norm_(model.parameters(),
                                         float(budget["grad_clip"]))
                optimiser.step()
                require(all(bool(torch.isfinite(parameter).all().item())
                            for parameter in model.parameters()),
                        "attentive parameter is non-finite")
                completed_updates += 1
                epoch_updates += 1
            require(epoch_updates == UPDATES_PER_EPOCH,
                    "optimizer updates per epoch changed")
            completed_epochs = epoch
            technical_trace.append({
                "epoch": epoch,
                "completed_optimizer_updates": completed_updates,
                "technical_finite": True,
                "performance_metric_inspected": False,
                "calibration_opened": False,
            })
            print(f"[attentive-amendment] technical epoch {epoch:02d}/60",
                  flush=True)
        require(completed_updates == TOTAL_UPDATES,
                "fixed optimizer update budget changed")
        require(FROZEN.tensor_digest(order.to(torch.int64))
                == data_order_witness["permutations"][-1][
                    "permutation_tensor_digest"]
                and FROZEN.tensor_digest(generator.get_state())
                == data_order_witness["final_generator_state_digest"],
                "executed training order changed")
        state = FROZEN._cpu_state(model)
        optimiser_state = optimiser.state_dict()
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "status": STATUS,
            "attempt_number": 1,
            "attempt_digest": attempt[ATTEMPT_SELF_KEY],
            "execution_bindings": dict(bindings),
            "initial_state_digest": initialisation["initial_state_digest"],
            "final_state_digest": FROZEN.state_dict_digest(state),
            "model_state_dict": state,
            "optimizer_state_dict": optimiser_state,
            "optimizer_state_digest": FROZEN.structured_digest(optimiser_state),
            "registered_seed": CONTRACT.ATTENTIVE_SEED,
            "completed_epoch": EPOCHS,
            "completed_optimizer_updates": completed_updates,
            "example_presentations": PRESENTATIONS,
            "data_order_seed": DATA_ORDER_SEED,
            "data_order_witness": attempt["data_order_witness"],
            "effective_batch": EFFECTIVE_BATCH,
            "microbatch": MICROBATCH,
            "epoch_selection": "final_epoch_only_no_selection",
            "learning_rate_schedule": "constant",
            "last_epoch_order_digest": FROZEN.tensor_digest(order.to(torch.int64)),
            "final_order_generator_state_digest": FROZEN.tensor_digest(
                generator.get_state()),
            "technical_trace": technical_trace,
            "training_wall_time_s": round(time.time() - started, 3),
            "smoke_state_used": False,
        }
        FROZEN.atomic_torch_save(checkpoint, final_checkpoint_path(root))
        return state, {
            "path": str(final_checkpoint_path(root)),
            "sha256": BASE.file_sha256(final_checkpoint_path(root)),
            "byte_count": final_checkpoint_path(root).stat().st_size,
            "final_state_digest": checkpoint["final_state_digest"],
            "optimizer_state_digest": checkpoint["optimizer_state_digest"],
            "completed_epoch": EPOCHS,
            "completed_optimizer_updates": completed_updates,
            "example_presentations": PRESENTATIONS,
            "training_wall_time_s": checkpoint["training_wall_time_s"],
            "technical_validity": True,
            "attempt_digest": attempt[ATTEMPT_SELF_KEY],
            "data_order_witness": checkpoint["data_order_witness"],
        }
    except BaseException as exc:
        _record_failure(path=technical_failure_path(root),
                        stage="attentive_training", error=exc,
                        epochs=completed_epochs, updates=completed_updates)
        raise


def _evidence_payload(*, rows: Sequence[Mapping[str, Any]],
                      predictions: Mapping[str, np.ndarray],
                      bindings: Mapping[str, Any], evaluation_digest: str,
                      checkpoint_sha256: str,
                      final_state_digest: str) -> dict[str, Any]:
    evidence_rows = []
    for index, row in enumerate(rows):
        evidence_rows.append({
            "training_view_row_digest": row["training_view_row_digest"],
            "branch_identity_digest": row["branch_identity_digest"],
            "state_id": row["state_id"],
            "family": row["family"],
            "stratum": row["stratum"],
            "candidate_index": int(row["candidate_index"]),
            "target": {key: float(row[key]) for key in (
                "progress", "safety", "completion", "utility")},
            "prediction": {key: float(predictions[key][index]) for key in (
                "progress", "safety", "completion", "utility")},
        })
    return signed(FROZEN._safe_json({
        "schema": EVIDENCE_SCHEMA,
        "status": STATUS,
        "complete": True,
        "execution_bindings": dict(bindings),
        "evaluation_authorisation_digest": evaluation_digest,
        "final_checkpoint_sha256": checkpoint_sha256,
        "final_state_digest": final_state_digest,
        "row_count": CALIBRATION_ROWS,
        "training_view_row_order_digest": AMENDMENT.digest([
            row["training_view_row_digest"] for row in evidence_rows]),
        "training_view_row_identity_set_digest": AMENDMENT.digest(sorted(
            row["training_view_row_digest"] for row in evidence_rows)),
        "branch_identity_set_digest": AMENDMENT.digest(sorted(
            row["branch_identity_digest"] for row in evidence_rows)),
        "rows": evidence_rows,
        "calibration_evaluation_session_count": 1,
        "model_forward_batch_count": CALIBRATION_FORWARD_BATCHES,
        "raw_latent_persisted": False,
        "predictor_material_accessed": False,
    }), EVIDENCE_SELF_KEY)


def metrics_from_evidence(*, corpus_rows: list[dict[str, Any]],
                          evidence: Mapping[str, Any],
                          bindings: Mapping[str, Any],
                          evaluation_digest: str,
                          checkpoint_sha256: str,
                          final_state_digest: str,
                          ) -> tuple[dict[str, Any], dict[str, Any],
                                     dict[str, Any]]:
    value = validate_signed(evidence, EVIDENCE_SELF_KEY,
                            "calibration evidence")
    require(value.get("schema") == EVIDENCE_SCHEMA
            and value.get("status") == STATUS
            and value.get("complete") is True
            and value.get("execution_bindings") == bindings
            and value.get("evaluation_authorisation_digest") == evaluation_digest
            and value.get("final_checkpoint_sha256") == checkpoint_sha256
            and value.get("final_state_digest") == final_state_digest
            and value.get("row_count") == CALIBRATION_ROWS
            and isinstance(value.get("rows"), list)
            and len(value["rows"]) == CALIBRATION_ROWS
            and len({row.get("training_view_row_digest")
                     for row in value["rows"]}) == CALIBRATION_ROWS
            and len({row.get("branch_identity_digest")
                     for row in value["rows"]}) == CALIBRATION_ROWS
            and value.get("training_view_row_order_digest")
            == AMENDMENT.digest([
                row["training_view_row_digest"] for row in value["rows"]])
            and value.get("training_view_row_identity_set_digest")
            == AMENDMENT.digest(sorted(
                row["training_view_row_digest"] for row in value["rows"]))
            and value.get("branch_identity_set_digest")
            == AMENDMENT.digest(sorted(
                row["branch_identity_digest"] for row in value["rows"]))
            and value.get("calibration_evaluation_session_count") == 1
            and value.get("model_forward_batch_count")
            == CALIBRATION_FORWARD_BATCHES
            and value.get("raw_latent_persisted") is False
            and value.get("predictor_material_accessed") is False,
            "calibration evidence binding changed")
    predicted = {key: [] for key in (
        "progress", "safety", "completion", "utility")}
    true = {key: [] for key in ("progress", "safety", "completion")}
    for frozen, row in zip(corpus_rows, value["rows"], strict=True):
        require(all(row.get(key) == frozen[key] for key in (
                    "training_view_row_digest", "branch_identity_digest",
                    "state_id", "family", "stratum", "candidate_index"))
                and all(float(row["target"][key]) == float(frozen[key])
                        for key in ("progress", "safety", "completion", "utility")),
                "calibration evidence row changed from frozen corpus")
        for key in predicted:
            number = float(row["prediction"][key])
            require(math.isfinite(number), "calibration prediction is non-finite")
            predicted[key].append(number)
        for key in true:
            true[key].append(float(row["target"][key]))
    predicted_arrays = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in predicted.items()
    }
    true_arrays = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in true.items()
    }
    overall = FROZEN._evaluate_arrays(
        corpus_rows, true_arrays, predicted_arrays)
    targets = {key: torch.from_numpy(value) for key, value in true_arrays.items()}
    per_family = FROZEN._grouped_calibration(
        corpus_rows, targets, predicted_arrays, "family")
    per_stratum = FROZEN._grouped_calibration(
        corpus_rows, targets, predicted_arrays, "stratum")
    return (FROZEN._safe_json(overall), FROZEN._safe_json(per_family),
            FROZEN._safe_json(per_stratum))


def exploratory_decision(*, criteria: Mapping[str, bool], safety_auc: float,
                         pairwise_gain: float) -> dict[str, Any]:
    require(len(criteria) == 8
            and all(type(value) is bool for value in criteria.values()),
            "the eight frozen scorer criteria changed")
    safety_gate = float(safety_auc) >= SAFETY_AUC_GATE
    gain_decimal = Decimal(str(pairwise_gain))
    gain_gate = gain_decimal >= PAIRWISE_GAIN_GATE
    delta_auc = float(safety_auc) - ORIGINAL_SAFETY_AUC
    delta_gain = float(pairwise_gain) - ORIGINAL_PAIRWISE_GAIN
    all_original = all(value is True for value in criteria.values())
    neither_primary = not safety_gate and not gain_gate
    material_degradation = delta_auc < 0.0 and delta_gain < 0.0
    if all_original and delta_auc > 0.0 and delta_gain > 0.0:
        classification = "STRONG_READOUT_SIGNAL"
        conclusion = (
            "final-layer ViT-L features contain more usable planning information; "
            "fresh independent qualification remains required")
    elif neither_primary or material_degradation:
        classification = "NO_READOUT_SIGNAL"
        conclusion = (
            "close learned utility scoring for the current final-layer H1-H4 "
            "latent contract; do not train another probe architecture")
    else:
        classification = "MIXED_READOUT_SIGNAL"
        conclusion = (
            "the readout hypothesis is not established; do not apply the scorer "
            "to predictor outputs")
    return {
        "classification": classification,
        "all_original_scorer_criteria_met": all_original,
        "safety_auc_gate_met": safety_gate,
        "latent_over_baseline_pairwise_gain_gate_met": gain_gate,
        "pairwise_gain_gate_decimal_comparison": {
            "observed_decimal_string": str(pairwise_gain),
            "threshold_decimal_string": str(PAIRWISE_GAIN_GATE),
        },
        "delta_attentive_minus_existing_vitl_safety_auc": delta_auc,
        "delta_attentive_minus_existing_vitl_latent_gain": delta_gain,
        "both_primary_quantities_strictly_improve":
            delta_auc > 0.0 and delta_gain > 0.0,
        "neither_primary_threshold_met": neither_primary,
        "material_primary_degradation": material_degradation,
        "per_family_consistency_is_report_only": True,
        "conclusion": conclusion,
    }


def _comparison_payload(*, attentive: Mapping[str, Any],
                        attentive_family: Mapping[str, Any],
                        baseline: Mapping[str, Any],
                        vitg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attentive_minus_existing_vitl": {
            "overall": BASE._metric_delta(attentive, baseline["vitl_metrics"]),
            "per_family": BASE._metric_delta(
                attentive_family, baseline["vitl_per_family_metrics"]),
        },
        "attentive_minus_vitg": {
            "overall": BASE._metric_delta(
                attentive, vitg["results"]["vitg"]["calibration"]),
            "per_family": BASE._metric_delta(
                attentive_family,
                vitg["results"]["vitg"]["per_family_calibration"]),
        },
        "attentive_minus_no_latent": {
            "overall": BASE._metric_delta(attentive, baseline["metrics"]),
            "per_family": BASE._metric_delta(
                attentive_family, baseline["per_family_metrics"]),
        },
    }


def _execute_once(root: Path, custody: dict[str, Any]) -> dict[str, Any]:
    require(not attempt_root(root).exists()
            and not attempt_root(root).is_symlink()
            and not evaluation_path(root).exists()
            and not evidence_path(root).exists()
            and not technical_failure_path(root).exists(),
            "the sole scientific attentive attempt was consumed")
    custody["stage"] = "scientific_preflight"
    amendment = AMENDMENT.validate_amendment(root)
    smoke = validate_production_smoke(root)
    device, preflight = device_preflight()
    fit_corpus = _load_fit_training_corpus(root)
    fit_rows, order_witness = BASE.registered_fit_rows_and_data_order(
        fit_corpus["fit_rows"])
    baseline, vitg, frozen_trees = _load_frozen_comparisons(fit_corpus, root)
    initialisation = build_initialisation(amendment=amendment, root=root)
    bindings = _execution_bindings(amendment, smoke)
    model = BASE.FinalLayerAttentiveUtilityScorer()
    started = time.time()
    custody["stage"] = "attentive_training"
    final_state, training = train_once(
        model, rows=fit_rows, store=fit_corpus["horizon"],
        initialisation=initialisation, bindings=bindings,
        data_order_witness=order_witness, device=device, root=root)
    custody["completed_epochs"] = EPOCHS
    custody["completed_updates"] = TOTAL_UPDATES
    custody["stage"] = "final_checkpoint_reload"
    model.load_state_dict(final_state, strict=True)
    model.to(device)
    evaluation = signed({
        "schema": EVALUATION_SCHEMA,
        "status": STATUS,
        "complete": True,
        "execution_bindings": bindings,
        "final_checkpoint_sha256": training["sha256"],
        "final_state_digest": training["final_state_digest"],
        "calibration_states": 24,
        "calibration_rows": CALIBRATION_ROWS,
        "maximum_evaluations": 1,
        "evaluation_number": 1,
        "calibration_metadata_may_be_preflight_validated": True,
        "calibration_latent_shards_read_before_authorisation": 0,
        "calibration_model_forwards_before_authorisation": 0,
        "calibration_predictions_before_authorisation": 0,
        "calibration_metrics_computed_before_authorisation": 0,
        "persist_closed_prediction_target_evidence": True,
    }, EVALUATION_SELF_KEY)
    publish_json(evaluation_path(root), evaluation,
                 "attentive evaluation authorisation")
    custody["stage"] = "authorised_calibration_materialisation"
    corpus = _load_corpus(root)
    require([row["training_view_row_digest"] for row in fit_rows]
            == [row["training_view_row_digest"]
                for row in sorted(corpus["fit_rows"], key=lambda row: (
                    str(row["state_id"]), int(row["candidate_index"])))],
            "fit-only and full frozen corpus views disagree")
    custody["stage"] = "calibration_evaluation"
    custody["calibration_evaluations"] = 1
    direct_metrics, predictions, _targets = BASE._evaluate_streaming(
        model, rows=corpus["calibration_rows"], store=corpus["horizon"],
        device=device)
    custody["calibration_evaluation_completed"] = True
    evidence = _evidence_payload(
        rows=corpus["calibration_rows"], predictions=predictions,
        bindings=bindings,
        evaluation_digest=evaluation[EVALUATION_SELF_KEY],
        checkpoint_sha256=training["sha256"],
        final_state_digest=training["final_state_digest"])
    publish_json(evidence_path(root), evidence,
                 "closed calibration prediction-target evidence")
    custody["closed_evidence_rows"] = CALIBRATION_ROWS
    custody["stage"] = "metric_replay_and_terminalisation"
    metrics, per_family, per_stratum = metrics_from_evidence(
        corpus_rows=corpus["calibration_rows"], evidence=evidence,
        bindings=bindings,
        evaluation_digest=evaluation[EVALUATION_SELF_KEY],
        checkpoint_sha256=training["sha256"],
        final_state_digest=training["final_state_digest"])
    require(FROZEN._safe_json(direct_metrics) == metrics,
            "sole calibration forward and evidence metrics differ")
    fit_distribution = FROZEN.label_distribution(corpus["fit_rows"])
    calibration_distribution = FROZEN.label_distribution(
        corpus["calibration_rows"])
    criteria, details, pairwise_gain = V13.qualification_criteria(
        metrics, baseline["metrics"], fit_distribution,
        calibration_distribution)
    family_report = BASE.per_family_primary_consistency(
        attentive=per_family,
        existing_vitl=baseline["vitl_per_family_metrics"],
        baseline=baseline["per_family_metrics"])
    decision = exploratory_decision(
        criteria=criteria,
        safety_auc=float(metrics["safety"]["auc_any_hazard"]),
        pairwise_gain=float(pairwise_gain))
    comparisons = _comparison_payload(
        attentive=metrics, attentive_family=per_family,
        baseline=baseline, vitg=vitg)
    result = signed(FROZEN._safe_json({
        "schema": SCHEMA,
        "status": STATUS,
        "complete": True,
        "scientific_result_valid": True,
        "label": "EXPLORATORY_FINAL_LAYER_ATTENTIVE_READOUT",
        "exploratory_not_qualification": True,
        "execution_bindings": bindings,
        "device_preflight": preflight,
        "official_pooler_binding_digest": BASE.OFFICIAL_POOLER_BINDING_DIGEST,
        "initialisation": {
            key: value for key, value in initialisation.items()
            if key != "model_state_dict"
        },
        "training": training,
        "evaluation_authorisation_digest": evaluation[EVALUATION_SELF_KEY],
        "calibration_evidence": {
            "path": str(evidence_path(root)),
            "sha256": BASE.file_sha256(evidence_path(root)),
            "byte_count": evidence_path(root).stat().st_size,
            "digest": evidence[EVIDENCE_SELF_KEY],
        },
        "training_execution_count": 1,
        "calibration_evaluation_count": 1,
        "calibration_model_forward_batch_count": CALIBRATION_FORWARD_BATCHES,
        "calibration_metric_recomputations_from_closed_evidence": 1,
        "results": {
            "attentive": {
                "calibration": metrics,
                "per_family_calibration": per_family,
                "per_stratum_calibration": per_stratum,
            },
            "existing_vitl_frozen": {
                "calibration": frozen_trees["vitl"]["overall"],
                "per_family_calibration": frozen_trees["vitl"]["per_family"],
                "per_stratum_calibration": frozen_trees["vitl"]["per_stratum"],
                "terminal_digest": baseline["vitl_terminal_digest"],
                "safety_auc": ORIGINAL_SAFETY_AUC,
                "latent_over_baseline_pairwise_gain": ORIGINAL_PAIRWISE_GAIN,
            },
            "vitg_frozen": {
                "result_digest": vitg["exploratory_result_digest"],
                "calibration": vitg["results"]["vitg"]["calibration"],
                "per_family_calibration": vitg["results"]["vitg"][
                    "per_family_calibration"],
                "per_stratum_calibration": vitg["results"]["vitg"][
                    "per_stratum_calibration"],
                "latent_over_baseline_pairwise_gain": vitg[
                    "latent_over_baseline_pairwise_gain"],
                "conclusion": vitg["exploratory_decision"]["classification"],
            },
            "no_latent_reused": {
                "calibration": frozen_trees["no_latent"]["overall"],
                "per_family_calibration": frozen_trees["no_latent"]["per_family"],
                "per_stratum_calibration": frozen_trees["no_latent"]["per_stratum"],
                "checkpoint_sha256": baseline["checkpoint_sha256"],
                "state_digest": baseline["state_digest"],
                "receipt_digest": baseline["receipt_digest"],
                "retrained": False,
                "reevaluated": False,
            },
        },
        "frozen_metric_tree_digests": FROZEN_METRIC_TREE_DIGESTS,
        "latent_over_baseline_pairwise_gain": pairwise_gain,
        "metric_comparisons": comparisons,
        "per_family_primary_consistency": family_report,
        "per_family_consistency_is_report_only": True,
        "frozen_original_gate_replay": {
            "criteria": criteria, "details": details,
        },
        "would_meet_all_original_gates": all(criteria.values()),
        "exploratory_decision": decision,
        "closed_prerequisite_interruptions": {
            "safety": amendment["closed_interruptions"][
                "safety_observability"]["terminal_kind"],
            "latent": amendment["closed_interruptions"][
                "latent_dependence"]["terminal_kind"],
        },
        "production_smoke_digest": smoke[SMOKE_SELF_KEY],
        "technical_failures_in_scientific_attempt": [],
        "invalid_scientific_attempts": [],
        "qualified_scorer_package_published": False,
        "predictor_retrained": False,
        "predictor_checkpoints_opened_for_utility": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
        "wall_time_s": round(time.time() - started, 3),
        "nothing_left_running_by_this_process_after_exit": True,
    }), RESULT_SELF_KEY)
    publish_json(result_path(root), result, "attentive amended result")
    custody["stage"] = "terminal_validation"
    return validate_result_for_consumption(root)


def run_once(root: Path = ROOT) -> dict[str, Any]:
    custody = {
        "stage": "terminal_validation" if result_path(root).exists()
        else "scientific_preflight",
        "completed_epochs": EPOCHS if final_checkpoint_path(root).exists() else 0,
        "completed_updates": TOTAL_UPDATES if final_checkpoint_path(root).exists()
        else 0,
        "calibration_evaluations": 1 if evidence_path(root).exists() else 0,
        "calibration_evaluation_completed": evidence_path(root).exists(),
        "closed_evidence_rows": CALIBRATION_ROWS if evidence_path(root).exists()
        else 0,
    }
    try:
        if result_path(root).exists() or result_path(root).is_symlink():
            return validate_result_for_consumption(root)
        return _execute_once(root, custody)
    except BaseException as exc:
        _record_failure(
            path=technical_failure_path(root), stage=str(custody["stage"]),
            error=exc, epochs=int(custody["completed_epochs"]),
            updates=int(custody["completed_updates"]),
            calibration_evaluations=int(custody["calibration_evaluations"]),
            calibration_evaluation_completed=bool(
                custody["calibration_evaluation_completed"]),
            closed_evidence_rows=int(custody["closed_evidence_rows"]))
        raise


def validate_result_for_consumption(root: Path = ROOT) -> dict[str, Any]:
    """Recompute all metrics/gates from evidence without another forward."""

    require(not technical_failure_path(root).exists()
            and not technical_failure_path(root).is_symlink(),
            "a technical failure terminal conflicts with the result")
    amendment = AMENDMENT.validate_amendment(root)
    smoke = validate_production_smoke(root)
    bindings = _execution_bindings(amendment, smoke)
    result = validate_signed(
        read_json(result_path(root), "attentive amended result"),
        RESULT_SELF_KEY, "attentive amended result")
    device = result.get("device_preflight", {})
    require(result.get("schema") == SCHEMA
            and result.get("status") == STATUS
            and result.get("complete") is True
            and result.get("scientific_result_valid") is True
            and result.get("exploratory_not_qualification") is True
            and result.get("execution_bindings") == bindings
            and result.get("official_pooler_binding_digest")
            == BASE.OFFICIAL_POOLER_BINDING_DIGEST
            and device.get("selected_device") == "cuda:0"
            and device.get("selected_name") == "AMD Radeon AI PRO R9700"
            and device.get("selected_architecture") == "gfx1201"
            and device.get("visible_hip_device_count") == 2
            and result.get("training_execution_count") == 1
            and result.get("calibration_evaluation_count") == 1
            and result.get("calibration_model_forward_batch_count")
            == CALIBRATION_FORWARD_BATCHES
            and result.get("calibration_metric_recomputations_from_closed_evidence")
            == 1
            and result.get("production_smoke_digest") == smoke[SMOKE_SELF_KEY]
            and result.get("closed_prerequisite_interruptions") == {
                "safety": amendment["closed_interruptions"][
                    "safety_observability"]["terminal_kind"],
                "latent": amendment["closed_interruptions"][
                    "latent_dependence"]["terminal_kind"],
            }
            and result.get("technical_failures_in_scientific_attempt") == []
            and result.get("invalid_scientific_attempts") == []
            and result.get("qualified_scorer_package_published") is False
            and result.get("predictor_retrained") is False
            and result.get("predictor_checkpoints_opened_for_utility") == 0
            and result.get("predictor_utility_shards_opened") == 0
            and result.get("final_200_state_corpus_generated") is False
            and result.get("nothing_left_running_by_this_process_after_exit")
            is True,
            "attentive amended result binding changed")
    initial_path = initialisation_path(root)
    require(initial_path.is_file() and not initial_path.is_symlink(),
            "scientific initialisation artifact is absent or non-regular")
    initialisation = torch.load(
        initial_path, map_location="cpu", weights_only=False)
    _model, expected_state, expected_digest = _fresh_model_state()
    require(initialisation.get("schema") == INITIALISATION_SCHEMA
            and initialisation.get("status") == STATUS
            and initialisation.get("prerequisite_amendment_digest")
            == amendment[AMENDMENT.SELF_KEY]
            and initialisation.get("initial_state_digest") == expected_digest
            and initialisation.get("registered_seed")
            == CONTRACT.ATTENTIVE_SEED
            and initialisation.get("architecture_seed_digest")
            == CONTRACT.ATTENTIVE_SEED_KEY_DIGEST
            and initialisation.get("trainable_parameter_count")
            == CONTRACT.ATTENTIVE_READOUT_ARCHITECTURE[
                "trainable_parameter_count"]
            and initialisation.get("parameter_initialisation", {}).get(
                "all_trainable_parameters_use_architecture_seed") is True
            and initialisation.get("parameter_initialisation", {}).get(
                "copied_predecessor_parameter_count") == 0
            and initialisation.get("smoke_state_reused") is False
            and all(torch.equal(
                initialisation["model_state_dict"][key], value)
                    for key, value in expected_state.items())
            and result.get("initialisation") == {
                key: value for key, value in initialisation.items()
                if key != "model_state_dict"},
            "fresh scientific initialisation changed")
    checkpoint_path = final_checkpoint_path(root)
    training = result["training"]
    require(Path(str(training.get("path"))).absolute()
            == checkpoint_path.absolute()
            and checkpoint_path.is_file() and not checkpoint_path.is_symlink()
            and BASE.file_sha256(checkpoint_path) == training["sha256"]
            and training.get("byte_count") == checkpoint_path.stat().st_size
            and training.get("completed_epoch") == EPOCHS
            and training.get("completed_optimizer_updates") == TOTAL_UPDATES
            and training.get("example_presentations") == PRESENTATIONS
            and training.get("technical_validity") is True,
            "final attentive checkpoint bytes changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint.get("schema") == CHECKPOINT_SCHEMA
            and checkpoint.get("status") == STATUS
            and checkpoint.get("attempt_number") == 1
            and checkpoint.get("execution_bindings") == bindings
            and checkpoint.get("initial_state_digest") == expected_digest
            and checkpoint.get("registered_seed") == CONTRACT.ATTENTIVE_SEED
            and checkpoint.get("data_order_seed") == DATA_ORDER_SEED
            and checkpoint.get("completed_epoch") == EPOCHS
            and checkpoint.get("completed_optimizer_updates") == TOTAL_UPDATES
            and checkpoint.get("example_presentations") == PRESENTATIONS
            and checkpoint.get("epoch_selection")
            == "final_epoch_only_no_selection"
            and checkpoint.get("learning_rate_schedule") == "constant"
            and checkpoint.get("effective_batch") == EFFECTIVE_BATCH
            and checkpoint.get("microbatch") == MICROBATCH
            and checkpoint.get("data_order_witness")
            == training.get("data_order_witness")
            and all(checkpoint["data_order_witness"].get(key)
                    == CONTRACT.DATA_ORDER_CONTRACT[key] for key in (
                        "base_training_view_row_digest_sequence_digest",
                        "permutation_plan_digest",
                        "row_presentation_plan_digest"))
            and checkpoint["data_order_witness"].get(
                "final_generator_state_digest")
            == "f1826a6a0c7f2cde2dcd028393e1229f2a6931099a22b8c31f97b968dbc77cb2"
            and checkpoint.get("last_epoch_order_digest")
            == "e71b4cb6ea9bf0854e603894457e265204cec4978256bb5e6d08a00e6026735a"
            and checkpoint.get("final_order_generator_state_digest")
            == "f1826a6a0c7f2cde2dcd028393e1229f2a6931099a22b8c31f97b968dbc77cb2"
            and checkpoint.get("smoke_state_used") is False
            and training.get("attempt_digest") == checkpoint.get("attempt_digest")
            and FROZEN.state_dict_digest(checkpoint["model_state_dict"])
            == checkpoint.get("final_state_digest")
            == training["final_state_digest"]
            and FROZEN.structured_digest(checkpoint["optimizer_state_dict"])
            == checkpoint.get("optimizer_state_digest")
            == training["optimizer_state_digest"],
            "final attentive checkpoint content changed")
    attempt = validate_signed(
        read_json(attempt_root(root) / "attempt.json", "attentive attempt"),
        ATTEMPT_SELF_KEY, "attentive attempt")
    require(attempt.get("schema") == ATTEMPT_SCHEMA
            and attempt.get("attempt_number") == 1
            and attempt.get("maximum_attempts") == 1
            and attempt.get("execution_bindings") == bindings
            and attempt.get("initial_state_digest") == expected_digest
            and attempt.get("registered_seed") == CONTRACT.ATTENTIVE_SEED
            and attempt.get("data_order_seed") == DATA_ORDER_SEED
            and attempt.get("data_order_witness")
            == checkpoint.get("data_order_witness")
            and attempt.get("effective_batch") == EFFECTIVE_BATCH
            and attempt.get("microbatch") == MICROBATCH
            and attempt.get("gradient_accumulation_steps")
            == EFFECTIVE_BATCH // MICROBATCH
            and attempt.get("fixed_final_epoch") == EPOCHS
            and attempt.get("resume_source") is None
            and attempt.get("retry_or_replacement_authorised") is False
            and attempt.get("calibration_opened") is False
            and attempt.get("smoke_checkpoint_used") is False
            and checkpoint.get("attempt_digest") == attempt[ATTEMPT_SELF_KEY],
            "attentive attempt receipt changed")
    evaluation = validate_signed(
        read_json(evaluation_path(root), "evaluation authorisation"),
        EVALUATION_SELF_KEY, "evaluation authorisation")
    require(evaluation.get("schema") == EVALUATION_SCHEMA
            and evaluation.get("execution_bindings") == bindings
            and evaluation.get("final_checkpoint_sha256") == training["sha256"]
            and evaluation.get("final_state_digest")
            == training["final_state_digest"]
            and evaluation.get("maximum_evaluations") == 1
            and evaluation.get("evaluation_number") == 1
            and evaluation.get("calibration_metadata_may_be_preflight_validated")
            is True
            and evaluation.get("calibration_latent_shards_read_before_authorisation")
            == 0
            and evaluation.get("calibration_model_forwards_before_authorisation")
            == 0
            and evaluation.get("calibration_predictions_before_authorisation")
            == 0
            and evaluation.get("calibration_metrics_computed_before_authorisation")
            == 0
            and result.get("evaluation_authorisation_digest")
            == evaluation[EVALUATION_SELF_KEY],
            "evaluation authorisation changed")
    evidence = validate_signed(
        read_json(evidence_path(root), "calibration evidence"),
        EVIDENCE_SELF_KEY, "calibration evidence")
    require(result.get("calibration_evidence", {}).get("path")
            == str(evidence_path(root))
            and result.get("calibration_evidence", {}).get("byte_count")
            == evidence_path(root).stat().st_size
            and BASE.file_sha256(evidence_path(root))
            == result["calibration_evidence"]["sha256"]
            and evidence[EVIDENCE_SELF_KEY]
            == result["calibration_evidence"]["digest"],
            "calibration evidence bytes changed")
    corpus = _load_corpus(root)
    baseline, vitg, frozen_trees = _load_frozen_comparisons(corpus, root)
    metrics, per_family, per_stratum = metrics_from_evidence(
        corpus_rows=corpus["calibration_rows"], evidence=evidence,
        bindings=bindings,
        evaluation_digest=evaluation[EVALUATION_SELF_KEY],
        checkpoint_sha256=training["sha256"],
        final_state_digest=training["final_state_digest"])
    require(result["results"]["attentive"] == {
                "calibration": metrics,
                "per_family_calibration": per_family,
                "per_stratum_calibration": per_stratum,
            }, "attentive metrics do not replay closed evidence")
    fit_distribution = FROZEN.label_distribution(corpus["fit_rows"])
    calibration_distribution = FROZEN.label_distribution(
        corpus["calibration_rows"])
    criteria, details, pairwise_gain = V13.qualification_criteria(
        metrics, baseline["metrics"], fit_distribution,
        calibration_distribution)
    family_report = BASE.per_family_primary_consistency(
        attentive=per_family,
        existing_vitl=baseline["vitl_per_family_metrics"],
        baseline=baseline["per_family_metrics"])
    decision = exploratory_decision(
        criteria=criteria,
        safety_auc=float(metrics["safety"]["auc_any_hazard"]),
        pairwise_gain=float(pairwise_gain))
    comparisons = _comparison_payload(
        attentive=metrics, attentive_family=per_family,
        baseline=baseline, vitg=vitg)
    require(result.get("latent_over_baseline_pairwise_gain") == pairwise_gain
            and result.get("metric_comparisons") == comparisons
            and result.get("per_family_primary_consistency") == family_report
            and result.get("per_family_consistency_is_report_only") is True
            and result.get("frozen_original_gate_replay")
            == {"criteria": criteria, "details": details}
            and result.get("would_meet_all_original_gates")
            is all(criteria.values())
            and result.get("exploratory_decision") == decision,
            "attentive comparisons or decision do not replay evidence")
    require(result.get("frozen_metric_tree_digests")
            == FROZEN_METRIC_TREE_DIGESTS
            and result["results"]["existing_vitl_frozen"]["calibration"]
            == frozen_trees["vitl"]["overall"]
            and result["results"]["existing_vitl_frozen"][
                "per_family_calibration"] == frozen_trees["vitl"]["per_family"]
            and result["results"]["existing_vitl_frozen"][
                "per_stratum_calibration"] == frozen_trees["vitl"]["per_stratum"]
            and result["results"]["vitg_frozen"]["calibration"]
            == frozen_trees["vitg"]["overall"]
            and result["results"]["vitg_frozen"]["per_family_calibration"]
            == frozen_trees["vitg"]["per_family"]
            and result["results"]["vitg_frozen"]["per_stratum_calibration"]
            == frozen_trees["vitg"]["per_stratum"]
            and result["results"]["no_latent_reused"]["calibration"]
            == frozen_trees["no_latent"]["overall"]
            and result["results"]["no_latent_reused"][
                "per_family_calibration"]
            == frozen_trees["no_latent"]["per_family"]
            and result["results"]["no_latent_reused"][
                "per_stratum_calibration"]
            == frozen_trees["no_latent"]["per_stratum"]
            and result["results"]["existing_vitl_frozen"]["safety_auc"]
            == ORIGINAL_SAFETY_AUC
            and result["results"]["existing_vitl_frozen"]["terminal_digest"]
            == baseline["vitl_terminal_digest"]
            and result["results"]["existing_vitl_frozen"][
                "latent_over_baseline_pairwise_gain"]
            == ORIGINAL_PAIRWISE_GAIN
            and result["results"]["vitg_frozen"]["result_digest"]
            == CONTRACT.FROZEN_VITG_RESULT_DIGEST
            and result["results"]["vitg_frozen"][
                "latent_over_baseline_pairwise_gain"]
            == vitg["latent_over_baseline_pairwise_gain"]
            and result["results"]["vitg_frozen"]["conclusion"]
            == vitg["exploratory_decision"]["classification"]
            and result["results"]["no_latent_reused"]["checkpoint_sha256"]
            == CONTRACT.FROZEN_BASELINE_CHECKPOINT_SHA256
            and result["results"]["no_latent_reused"]["state_digest"]
            == CONTRACT.FROZEN_BASELINE_STATE_DIGEST
            and result["results"]["no_latent_reused"]["receipt_digest"]
            == CONTRACT.FROZEN_BASELINE_RECEIPT_DIGEST
            and result["results"]["no_latent_reused"]["retrained"] is False
            and result["results"]["no_latent_reused"]["reevaluated"] is False,
            "frozen comparison lineage changed")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("issue-amendment", "smoke", "run"))
    arguments = parser.parse_args(argv)
    if arguments.stage == "issue-amendment":
        result = AMENDMENT.issue_amendment(ROOT)
        summary = {"status": result["status"],
                   "prerequisite_amendment_digest":
                       result[AMENDMENT.SELF_KEY]}
    elif arguments.stage == "smoke":
        result = run_production_smoke(ROOT)
        summary = {"status": result["status"],
                   "production_smoke_digest": result[SMOKE_SELF_KEY]}
    else:
        result = run_once(ROOT)
        summary = {
            "status": result["status"],
            "classification": result["exploratory_decision"]["classification"],
            "safety_auc": result["results"]["attentive"]["calibration"][
                "safety"]["auc_any_hazard"],
            "latent_over_baseline_pairwise_gain":
                result["latent_over_baseline_pairwise_gain"],
            "attentive_result_digest": result[RESULT_SELF_KEY],
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
