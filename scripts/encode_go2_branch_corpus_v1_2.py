#!/usr/bin/env python3
"""Encode a v1.2 branch corpus with the frozen factorial target encoder.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  No predictor checkpoint is opened.

The scientific encoder path is the unchanged V03 centre-crop path used by the
factorial: a native 224x224 textured-v03 PNG is cropped to rows 28:196, resized
bicubically to 512x384, ImageNet-normalised, and encoded by the frozen V-JEPA
2.1 ViT-L/384 EMA encoder.  The 24x32 tokens are layer-normalised over their
1024-dimensional feature axis.

Each state context and each valid branch horizon is written as an atomic f16
shard.  This is deliberately not one monolithic blob: after an interruption a
verified shard is retained and only missing exact registered items are encoded.
``latents_index.json`` binds every shard and is complete only when the corpus
receipt and every required shard validate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOC  # noqa: E402
from lewm.oracle import go2_invalid_scorer_identity_exclusion_v1_2 as INVALID_IDS  # noqa: E402
from lewm.oracle import go2_scorer_state_selector_amendment_v2 as STATE_SELECTOR  # noqa: E402
from lewm.oracle import go2_scorer_fit_corpus_v2_scorer_contract as V2_CONTRACT  # noqa: E402
from lewm.oracle.go2_scorer_contract_v1_2 import (  # noqa: E402
    clean_source_binding,
    contract,
    contract_digest,
    _managed_scorer_contract_output_path,
    preprocess_contract_digest,
    render_contract_digest,
    target_encoder_digest,
)
from lewm.oracle.go2_textured_v03_renderer import (  # noqa: E402
    renderer_contract_digest as textured_v03_renderer_contract_digest,
)
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402
from scripts import build_go2_branch_corpus_v1_2 as CORPUS_BUILDER  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_ROOT = ROOT / ".generated/go2_branch_corpus_v1_2"
TOKENS = 768
TOKEN_DIM = 1024
HORIZONS = 4
CONTEXT_SLOTS = 3
CONTEXT_SHAPE = (CONTEXT_SLOTS, TOKENS, TOKEN_DIM)
HORIZON_SHAPE = (HORIZONS, TOKENS, TOKEN_DIM)
PREPROCESSING_SHA256 = (
    "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"
)
CORPUS_BINDING_KEYS = (
    "candidate_allocation_manifest_digest",
    "candidate_allocator_contract_digest",
    "candidate_allocation_amendment_digest",
    "candidate_allocation_post_identity_validation_digest",
    "pre_identity_allocation_validation_digest",
    "invalid_scorer_identity_exclusion_digest",
    "state_selector_amendment_digest",
    "state_selector_feasibility_receipt_digest",
    "preserved_state_revalidation_receipt_digest",
    "clean_source_launch_receipt_digest",
    "source_repository_commit",
    "clean_source_binding_digest",
    "bound_implementations_digest",
    "scorer_contract_artifact_digest",
    "mixed_precontract_disposition_receipt_digest",
    "candidate_bank_digest",
    "progress_contract_digest",
    "safety_contract_digest",
    "oracle_v1_2_digest",
    "scorer_contract_v1_2_digest",
    "selection_digest",
    "boundary_digest",
    "render_contract_digest",
    "textured_v03_renderer_contract_digest",
    "preprocess_contract_digest",
    "preprocessing_digest",
    "target_encoder_digest",
    "target_encoder_checkpoint_sha256",
)
FULL_BANK_V2_BINDING_KEYS = (
    "scorer_fit_corpus_v2_design_digest",
    "rotation_mask_classification_digest",
    "full_bank_small_completion_selection_digest",
    "full_bank_preoutcome_state_revalidation_digest",
    "state_identity_projection_digest",
    "full_bank_assignment_manifest_digest",
    "scorer_fit_corpus_v2_scorer_contract_digest",
    "scorer_fit_corpus_v2_scorer_contract_artifact_digest",
    "invalid_scorer_identity_exclusion_digest",
    "state_selector_amendment_digest",
    "state_selector_feasibility_receipt_digest",
    "candidate_bank_digest",
    "progress_contract_digest",
    "safety_contract_digest",
    "oracle_v1_2_digest",
    "scorer_contract_v1_2_digest",
    "selection_digest",
    "boundary_digest",
    "render_contract_digest",
    "textured_v03_renderer_contract_digest",
    "preprocess_contract_digest",
    "preprocessing_digest",
    "target_encoder_digest",
    "target_encoder_checkpoint_sha256",
)
FULL_BANK_V2_CONTRACT_LINEAGE_SCHEMA = (
    "go2_scorer_fit_corpus_v2_operational_contract_lineage_v1"
)
FULL_BANK_V2_SMOKE_SCHEMA = (
    "go2_scorer_fit_corpus_v2_end_to_end_smoke_receipt_v1"
)
FULL_BANK_V2_LATENT_INDEX_SCHEMA = (
    "go2_scorer_fit_corpus_v2_latents_index_v1"
)
FULL_BANK_V2_ENCODING_INVOCATION_SCHEMA = (
    "go2_scorer_fit_corpus_v2_encoding_invocation_summary_v1"
)
FULL_BANK_V2_EXPECTED_STATES = 120
FULL_BANK_V2_EXPECTED_BRANCHES = 1_440
FULL_BANK_V2_EXPECTED_CANDIDATES_PER_STATE = 12
FULL_BANK_V2_INDEX_NAME = "latents_index_v2.json"
FULL_BANK_V2_ENCODING_SUMMARY_NAME = "encoding_invocation_summary_v2.json"
FULL_BANK_V2_SMOKE_NAME = "smoke_encoding_receipt_v2.json"
FULL_BANK_V2_LATENTS_NAME = "latents_v2"
FULL_BANK_V2_INVALID_ATTEMPTS_NAME = "invalid_attempts_v2"
FULL_BANK_V2_SUPERSEDED_RECEIPTS_NAME = "superseded_receipts_v2"
SELECTOR_BINDING_KEYS = tuple(STATE_SELECTOR.ACTIVE_SELECTOR_BINDING_KEYS)
LAUNCH_BINDING_KEYS = (
    "clean_source_launch_receipt_digest",
    "source_repository_commit",
    "clean_source_binding_digest",
    "bound_implementations_digest",
    "scorer_contract_artifact_digest",
    "mixed_precontract_disposition_receipt_digest",
)
SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS = LAUNCH_BINDING_KEYS[:-1]
GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_SCHEMA = (
    "go2_utility_scorer_v1_2_global_exact_contract_lineage_v1"
)
GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_KEYS = frozenset((
    "schema",
    "scientific_predecessor_scorer_contract_v1_2_digest",
    "current_scorer_contract_v1_2_digest",
    "global_exact_successor_scorer_contract_digest",
))
SCORER_CONTRACT_ARTIFACT_PATH = (
    ROOT / ".generated/go2_utility_scorer_v1_2/scorer_contract_v1_2.json"
)


def _is_full_bank_v2_manifest(value: Mapping[str, Any]) -> bool:
    return (value.get("schema")
            == CORPUS_BUILDER.SCORER_FIT_V2_STATE_MANIFEST_SCHEMA
            and value.get("pool") == "scorer_fit_v2")


def _corpus_binding_keys(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    return FULL_BANK_V2_BINDING_KEYS if _is_full_bank_v2_manifest(
        manifest) else CORPUS_BINDING_KEYS


def _output_name(*, full_bank_v2: bool, legacy: str, v2: str) -> str:
    return v2 if full_bank_v2 else legacy


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_index_if_changed(path: Path, payload: dict[str, Any],
                            prior: dict[str, Any]) -> bool:
    """Write a latent index only when its scientific content has changed."""

    if prior == payload and path.is_file():
        return False
    atomic_json(path, payload)
    return True


def atomic_f16(path: Path, array: np.ndarray) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    np.asarray(array, dtype=np.float16).tofile(temporary)
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    digest = file_sha256(temporary)
    byte_count = temporary.stat().st_size
    os.replace(temporary, path)
    return digest, byte_count


def _without_digest(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def _verify_self_digest(payload: dict[str, Any], key: str, label: str) -> None:
    observed = str(payload.get(key) or "")
    expected = canonical_digest(_without_digest(payload, key))
    if observed != expected:
        raise RuntimeError(f"{label} self digest mismatch: {observed} != {expected}")


def _require_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise RuntimeError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _validate_global_exact_scorer_contract_lineage(
        value: Any, *, expected: Mapping[str, Any] | None = None,
        ) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise RuntimeError("global exact scorer-contract lineage is not an object")
    if set(value) != GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_KEYS:
        raise RuntimeError(
            "global exact scorer-contract lineage schema is not closed")
    lineage = dict(value)
    if lineage.get("schema") != GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_SCHEMA:
        raise RuntimeError("global exact scorer-contract lineage schema changed")
    for key in GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_KEYS - {"schema"}:
        _require_sha256(lineage.get(key), f"global exact lineage {key}")
    if lineage["current_scorer_contract_v1_2_digest"] != contract_digest():
        raise RuntimeError(
            "global exact operational scorer contract differs from current source")
    if expected is not None and dict(expected) != lineage:
        raise RuntimeError("global exact scorer-contract lineage differs")
    return lineage


def _resolve_frame(out: Path, value: str) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else out / path
    resolved = resolved.resolve()
    if out.resolve() not in resolved.parents:
        raise RuntimeError(f"frame escapes corpus root: {value}")
    return resolved


def _frame_records(row: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    records = row.get(f"{kind}_frames")
    if isinstance(records, list):
        return records
    # A hard failure is preferable to silently accepting the interrupted v0
    # schema, which did not bind render bytes.
    raise RuntimeError(f"row lacks bound {kind}_frames records")


def _verify_frames(out: Path, records: list[dict[str, Any]], expected: int,
                   label: str) -> list[str]:
    if len(records) != expected:
        raise RuntimeError(f"{label}: expected {expected} frames, found {len(records)}")
    paths: list[str] = []
    for index, record in enumerate(records):
        if int(record.get("slot", record.get("horizon", -1))) != index + (
                1 if "horizon" in record else 0):
            # Context slots are 0..2 and horizons are 1..4.
            raise RuntimeError(f"{label}: frame order/index mismatch at {index}")
        if (record.get("shape") != [224, 224, 3]
                or record.get("dtype") != "uint8"):
            raise RuntimeError(f"{label}: frame render shape/dtype mismatch at {index}")
        path = _resolve_frame(out, str(record["path"]))
        if not path.is_file():
            raise RuntimeError(f"{label}: missing frame {path}")
        if path.stat().st_size != int(record["byte_count"]):
            raise RuntimeError(f"{label}: frame byte count mismatch for {path}")
        if file_sha256(path) != str(record["sha256"]):
            raise RuntimeError(f"{label}: frame digest mismatch for {path}")
        paths.append(str(path))
    return paths


def _validate_row(row: dict[str, Any], manifest: dict[str, Any],
                  expected_contract: str) -> None:
    _verify_self_digest(row, "branch_row_digest",
                        f"branch row {row.get('state_id')}|{row.get('candidate')}")
    bindings = {
        "state_manifest_digest": manifest["state_manifest_digest"],
        **{key: manifest[key] for key in _corpus_binding_keys(manifest)},
    }
    if not _is_full_bank_v2_manifest(manifest):
        bindings["scorer_contract_v1_2_digest"] = expected_contract
    else:
        bindings["scorer_fit_corpus_v2_scorer_contract_digest"] = \
            expected_contract
    for key, expected in bindings.items():
        if str(row.get(key)) != str(expected):
            raise RuntimeError(
                f"row {row.get('state_id')}|{row.get('candidate')} {key} mismatch"
            )


def _load_clean_source_launch_receipt() -> dict[str, Any]:
    raw_path = OUT_ROOT / "scorer_fit/clean_source_launch_receipt.json"
    path = CORPUS_BUILDER.pin_active_scorer_fit_artifact_for_consumption(
        raw_path, "clean_source_launch_receipt.json")
    scorer_contract_path = _managed_scorer_contract_output_path(
        SCORER_CONTRACT_ARTIFACT_PATH
    )
    if not path.is_file() or not scorer_contract_path.is_file():
        raise RuntimeError("latent encoding requires the clean-source launch artifacts")
    receipt = json.loads(path.read_text())
    _verify_self_digest(
        receipt, "clean_source_launch_receipt_digest", "clean-source launch receipt")
    scorer_artifact = json.loads(scorer_contract_path.read_text())
    _verify_self_digest(
        scorer_artifact, "contract_artifact_digest", "scorer contract artifact")
    current = clean_source_binding()
    pending_phase_2 = scorer_artifact.get(
        "mixed_state_post_allocation_revalidation")
    expected = {
        "clean_source_launch_receipt_digest":
            receipt["clean_source_launch_receipt_digest"],
        "source_repository_commit": current["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(current),
        "bound_implementations_digest": current["bound_implementations_digest"],
        "scorer_contract_artifact_digest":
            scorer_artifact["contract_artifact_digest"],
    }
    if (receipt.get("source_repository_clean") is not True
            or receipt.get("source_repository_commit")
            != current["source_repository_commit"]
            or scorer_artifact.get("clean_source_binding") != current
            or scorer_artifact.get("clean_source_binding_digest")
            != canonical_digest(current)
            or receipt.get("scorer_contract_artifact_digest")
            != scorer_artifact["contract_artifact_digest"]
            or receipt.get("scorer_contract_artifact_sha256")
            != file_sha256(scorer_contract_path)
            or scorer_artifact.get("state_selector_amendment_verified") is not True
            or scorer_artifact.get("state_selector_feasibility_verified") is not True
            or scorer_artifact.get(
                "preserved_state_mixed_precontract_disposition_verified") is not True
            or not isinstance(pending_phase_2, dict)
            or pending_phase_2.get("status")
            != "PENDING_POST_IDENTITY_PRE_OUTCOME"
            or pending_phase_2.get("required_before_active_identity_manifest")
            is not True
            or pending_phase_2.get("schema")
            != STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_SCHEMA
            or pending_phase_2.get("path")
            != STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH
            or pending_phase_2.get(
                "realized_receipt_digest_bound_at_contract_issue") is not False
            or receipt.get("state_selector_amendment_digest")
            != STATE_SELECTOR.state_selector_amendment_digest()
            or receipt.get("state_selector_feasibility_receipt_digest")
            != scorer_artifact.get(
                "state_selector_feasibility_receipt_digest")
            or receipt.get("mixed_precontract_disposition_receipt_digest")
            != scorer_artifact.get(
                "mixed_precontract_disposition_receipt_digest")):
        raise RuntimeError("clean-source launch artifacts differ from current clean HEAD")
    for key, value in expected.items():
        if key != "clean_source_launch_receipt_digest" and receipt.get(key) != value:
            raise RuntimeError(f"clean-source launch receipt {key} mismatch")
    feasibility_digest = receipt.get(
        "state_selector_feasibility_receipt_digest")
    disposition_digest = scorer_artifact.get(
        "mixed_precontract_disposition_receipt_digest")
    for value, label in (
        (feasibility_digest, "state-selector feasibility receipt"),
        (disposition_digest, "preserved-state mixed precontract disposition"),
    ):
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise RuntimeError(f"clean-source {label} digest is invalid")
    return {
        **expected,
        "launch_state_selector_feasibility_receipt_digest":
            feasibility_digest,
        "mixed_precontract_disposition_receipt_digest": disposition_digest,
    }


def _load_manifest_launch_lineage(
        manifest: Mapping[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve current operational and preserved scientific launch identities."""

    if "small_completion_global_exact_execution" not in manifest:
        legacy = _load_clean_source_launch_receipt()
        scientific = dict(legacy)
        return legacy, scientific, legacy
    try:
        successor = (
            CORPUS_BUILDER
            .load_global_exact_successor_scorer_contract_for_consumption(
                manifest))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise RuntimeError(
            f"global-exact successor scorer contract does not verify: {exc}"
        ) from exc
    if not isinstance(successor, Mapping):
        raise RuntimeError("global-exact successor scorer contract is not an object")
    predecessor = successor.get("scientific_predecessor_launch_bindings")
    if not isinstance(predecessor, Mapping):
        raise RuntimeError(
            "global-exact successor lacks scientific predecessor launch bindings")
    if set(predecessor) != set(SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS):
        raise RuntimeError(
            "global-exact scientific predecessor launch schema is not closed")
    digest_keys = (
        "clean_source_launch_receipt_digest",
        "clean_source_binding_digest",
        "bound_implementations_digest",
        "scorer_contract_artifact_digest",
    )
    for key in SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS:
        value = predecessor.get(key)
        expected_length = 40 if key == "source_repository_commit" else 64
        if (not isinstance(value, str) or len(value) != expected_length
                or any(character not in "0123456789abcdef" for character in value)):
            raise RuntimeError(f"scientific predecessor {key} is malformed")
    for key in (
            *SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS,
            "clean_source_launch_receipt_sha256",
            "scorer_contract_artifact_sha256",
            "launch_state_selector_feasibility_receipt_digest",
            "mixed_precontract_disposition_receipt_digest",
            "global_exact_execution_amendment_digest",
            "global_exact_successor_scorer_contract_digest",
            "current_scorer_contract_v1_2_digest",
            ):
        value = successor.get(key)
        expected_length = 40 if key == "source_repository_commit" else 64
        if (not isinstance(value, str) or len(value) != expected_length
                or any(character not in "0123456789abcdef" for character in value)):
            raise RuntimeError(f"operational successor {key} is malformed")
    # Keep this explicit to make accidental removal of one of the digest fields
    # visible to static review.
    if any(key not in successor for key in digest_keys):
        raise RuntimeError("global-exact operational launch bindings are incomplete")
    historical_scorer_digest = _require_sha256(
        manifest.get("scorer_contract_v1_2_digest"),
        "global-exact scientific predecessor scorer_contract_v1_2_digest",
    )
    contract_lineage = _validate_global_exact_scorer_contract_lineage({
        "schema": GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_SCHEMA,
        "scientific_predecessor_scorer_contract_v1_2_digest":
            historical_scorer_digest,
        "current_scorer_contract_v1_2_digest": successor[
            "current_scorer_contract_v1_2_digest"],
        "global_exact_successor_scorer_contract_digest": successor[
            "global_exact_successor_scorer_contract_digest"],
    })
    operational = {
        **dict(successor),
        "global_exact_scorer_contract_lineage": contract_lineage,
    }
    scientific = {
        **{key: predecessor[key]
           for key in SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS},
        "mixed_precontract_disposition_receipt_digest": successor[
            "mixed_precontract_disposition_receipt_digest"],
    }
    selector_launch = {
        **scientific,
        "launch_state_selector_feasibility_receipt_digest": successor[
            "launch_state_selector_feasibility_receipt_digest"],
    }
    return operational, scientific, selector_launch


def _load_selector_successor_receipts(
        *, source_commit: str, selection_digest: str,
        active_states: list[dict[str, Any]],
        expected_feasibility_receipt_digest: str,
        expected_mixed_precontract_disposition_receipt_digest: str,
        expected_clean_source_binding_digest: str | None = None,
        expected_bound_implementations_digest: str | None = None,
        enforce_managed_paths: bool = False,
        global_exact_manifest: Mapping[str, Any] | None = None,
        ) -> dict[str, str]:
    """Verify the outcome-free selector successor before opening branch rows."""

    try:
        STATE_SELECTOR.validate_authority_artifacts()
        raw_revalidation_path = (
            ROOT / STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH)
        raw_allocation_path = (
            OUT_ROOT / "scorer_fit/candidate_allocation_manifest.json")
        if enforce_managed_paths:
            revalidation_path = (
                CORPUS_BUILDER.pin_active_scorer_fit_artifact_for_consumption(
                    raw_revalidation_path,
                    STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_NAME))
            allocation_path = (
                CORPUS_BUILDER.pin_active_scorer_fit_artifact_for_consumption(
                    raw_allocation_path, "candidate_allocation_manifest.json"))
        else:
            revalidation_path = raw_revalidation_path
            allocation_path = raw_allocation_path
        raw_feasibility_path = (
            ROOT / STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH)
        raw_disposition_path = (
            ROOT
            / STATE_SELECTOR
            .PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH)
        if (not revalidation_path.is_file() or not allocation_path.is_file()
                or (not enforce_managed_paths
                    and (not raw_feasibility_path.is_file()
                         or not raw_disposition_path.is_file()))):
            raise RuntimeError(
                "scorer-fit selector successor artifacts are missing")
        feasibility = (
            STATE_SELECTOR.validate_frozen_reachability_feasibility_pass(root=ROOT)
            if enforce_managed_paths
            else json.loads(raw_feasibility_path.read_text())
        )
        if not enforce_managed_paths and feasibility != (
                STATE_SELECTOR.validate_frozen_reachability_feasibility_pass(
                    root=ROOT)):
            raise RuntimeError("selector feasibility differs from frozen pass")
        feasibility_digest = str(
            feasibility["state_selector_feasibility_receipt_digest"])
        if feasibility_digest != expected_feasibility_receipt_digest:
            raise RuntimeError(
                "selector feasibility differs from clean-source launch")
        if enforce_managed_paths:
            disposition = (
                STATE_SELECTOR
                .load_and_validate_preserved_state_mixed_precontract_disposition_receipt(
                    expected_source_commit=source_commit,
                    expected_successor_selection_digest=selection_digest,
                    expected_clean_source_binding_digest=
                        expected_clean_source_binding_digest,
                    expected_bound_implementations_digest=
                        expected_bound_implementations_digest,
                    root=ROOT,
                ))
        else:
            disposition = json.loads(raw_disposition_path.read_text())
            STATE_SELECTOR.validate_preserved_state_mixed_precontract_disposition_receipt(
                disposition,
                expected_source_commit=source_commit,
                expected_successor_selection_digest=selection_digest,
                expected_clean_source_binding_digest=
                    expected_clean_source_binding_digest,
                expected_bound_implementations_digest=
                    expected_bound_implementations_digest,
                root=ROOT,
            )
        if (disposition.get("mixed_precontract_disposition_receipt_digest")
                != expected_mixed_precontract_disposition_receipt_digest):
            raise RuntimeError(
                "mixed precontract disposition differs from clean-source launch")
        revalidation = json.loads(revalidation_path.read_text())
        allocation = json.loads(allocation_path.read_text())
        if global_exact_manifest is not None:
            certified = (
                CORPUS_BUILDER
                .validate_global_exact_allocation_for_consumption(
                    global_exact_manifest, allocation))
            if (certified["preserved_state_revalidation_receipt_digest"]
                    != revalidation.get(
                        "preserved_state_revalidation_receipt_digest")):
                raise RuntimeError(
                    "global exact phase-2 selector receipt changed")
        else:
            STATE_SELECTOR.validate_preserved_state_revalidation_receipt(
                revalidation, allocation_manifest=allocation,
                active_states=active_states,
                expected_source_commit=source_commit,
                expected_successor_selection_digest=selection_digest,
                expected_feasibility_receipt_digest=feasibility_digest,
                expected_mixed_precontract_disposition_receipt_digest=
                    expected_mixed_precontract_disposition_receipt_digest)
    except (OSError, json.JSONDecodeError,
            STATE_SELECTOR.StateSelectorAmendmentError) as exc:
        raise RuntimeError(
            f"scorer-fit selector successor does not verify: {exc}") from exc
    return {
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest": feasibility_digest,
        "preserved_state_revalidation_receipt_digest": str(
            revalidation["preserved_state_revalidation_receipt_digest"]),
    }


def _load_inputs(out: Path, *, allow_partial: bool,
                 pool: str | None = None) -> tuple[
        dict[str, Any], dict[str, Any], list[dict[str, Any]],
        dict[str, str] | None]:
    manifest_path = out / "state_manifest.json"
    # Reopen and replay the canonical pre-outcome selection provenance before
    # any branch row or frame is opened.  A self-signed manifest plus phase-2
    # mask receipt is insufficient: the builder validator also proves the
    # exact capture prefix and first passing small-family combination.
    if pool == "scorer_fit":
        manifest = CORPUS_BUILDER.load_active_state_manifest_for_consumption(
            manifest_path, pool="scorer_fit"
        )
    elif pool is not None:
        pinned_manifest = CORPUS_BUILDER._frozen_generated_artifact_path(
            manifest_path
        )
        manifest = json.loads(pinned_manifest.read_text())
        _verify_self_digest(manifest, "state_manifest_digest", "state manifest")
    else:
        # Source-only synthetic fixtures pass no production pool.  The CLI
        # always supplies one, so this cannot bypass the managed-root gate.
        manifest = json.loads(manifest_path.read_text())
        _verify_self_digest(manifest, "state_manifest_digest", "state manifest")
    frozen = contract()
    (operational_launch, manifest_launch,
     selector_launch) = _load_manifest_launch_lineage(manifest)
    contract_lineage = (
        _validate_global_exact_scorer_contract_lineage(
            operational_launch.get("global_exact_scorer_contract_lineage"))
        if "global_exact_successor_scorer_contract_digest" in operational_launch
        else None
    )
    scientific_contract_digest = (
        contract_lineage[
            "scientific_predecessor_scorer_contract_v1_2_digest"]
        if contract_lineage is not None else contract_digest()
    )
    selector = _load_selector_successor_receipts(
        source_commit=selector_launch["source_repository_commit"],
        selection_digest=frozen["corpus_selection_digest"],
        active_states=manifest["states"],
        expected_feasibility_receipt_digest=selector_launch[
            "launch_state_selector_feasibility_receipt_digest"],
        expected_mixed_precontract_disposition_receipt_digest=selector_launch[
            "mixed_precontract_disposition_receipt_digest"],
        expected_clean_source_binding_digest=selector_launch[
            "clean_source_binding_digest"],
        expected_bound_implementations_digest=selector_launch[
            "bound_implementations_digest"],
        enforce_managed_paths=True,
        global_exact_manifest=(
            manifest if isinstance(
                manifest.get("small_completion_global_exact_execution"),
                Mapping) else None))
    if (manifest.get("scorer_contract_v1_2_digest")
            != scientific_contract_digest):
        raise RuntimeError(
            "state manifest is bound to a different scientific scorer contract")
    target_encoder = frozen["target_encoder"]
    if (target_encoder.get("token_grid") != [24, 32]
            or target_encoder.get("tokens") != TOKENS
            or target_encoder.get("token_dim") != TOKEN_DIM):
        raise RuntimeError("scorer contract target-token layout changed")
    expected_manifest_bindings = {
        "selection_digest": frozen["corpus_selection_digest"],
        "candidate_allocator_contract_digest": ALLOC.allocation_contract_digest(),
        "candidate_allocation_amendment_digest":
            ALLOC.allocation_amendment_digest(),
        "invalid_scorer_identity_exclusion_digest":
            INVALID_IDS.invalid_identity_exclusion_digest(),
        **selector,
        "candidate_bank_digest": frozen["candidate_bank_digest"],
        "progress_contract_digest": frozen["progress_target_digest"],
        "safety_contract_digest": frozen["safety_target_digest"],
        "oracle_v1_2_digest": frozen["oracle_v1_2_digest"],
        "scorer_contract_v1_2_digest": scientific_contract_digest,
        "render_contract_digest": render_contract_digest(),
        "textured_v03_renderer_contract_digest":
            textured_v03_renderer_contract_digest(),
        "preprocess_contract_digest": preprocess_contract_digest(),
        "preprocessing_digest": PREPROCESSING_SHA256,
        "target_encoder_digest": target_encoder_digest(),
        "target_encoder_checkpoint_sha256": target_encoder["checkpoint_sha256"],
    }
    for key, expected in expected_manifest_bindings.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"state manifest frozen binding mismatch: {key}")
    for key in LAUNCH_BINDING_KEYS:
        if manifest.get(key) != manifest_launch[key]:
            raise RuntimeError(
                f"state manifest scientific launch binding mismatch: {key}")

    invalid_index = INVALID_IDS.load_invalid_identity_index()
    if manifest.get("exclusion_binding", {}).get(
            "invalid_scorer_identity_attempt") != invalid_index.binding():
        raise RuntimeError("state manifest invalid45 exclusion binding mismatch")
    INVALID_IDS.assert_disjoint(
        manifest.get("states", []), label="state manifest", index=invalid_index)

    if manifest.get("pool") == "scorer_fit":
        raw_preflight_path = out / "pre_identity_allocation_validation.json"
        raw_allocation_path = out / "candidate_allocation_manifest.json"
        preflight_path = (
            CORPUS_BUILDER.pin_active_scorer_fit_artifact_for_consumption(
                raw_preflight_path, "pre_identity_allocation_validation.json"))
        allocation_path = (
            CORPUS_BUILDER.pin_active_scorer_fit_artifact_for_consumption(
                raw_allocation_path, "candidate_allocation_manifest.json"))
        if not preflight_path.is_file() or not allocation_path.is_file():
            raise RuntimeError("scorer-fit allocation validation artifacts are missing")
        preflight = json.loads(preflight_path.read_text())
        ALLOC.validate_pre_identity_structural_validation(preflight)
        if (manifest.get("pre_identity_allocation_validation_digest")
                != preflight.get("pre_identity_validation_digest")):
            raise RuntimeError("pre-identity allocation validation binding mismatch")
        allocation = json.loads(allocation_path.read_text())
        if isinstance(
                manifest.get("small_completion_global_exact_execution"),
                Mapping):
            CORPUS_BUILDER.validate_global_exact_allocation_for_consumption(
                manifest, allocation)
        else:
            ALLOC.validate_allocation_manifest(
                allocation,
                expected_source_identity_manifest_digest=
                    manifest["pre_allocation_identity_manifest_digest"],
            )
        if (manifest.get("candidate_allocation_manifest_digest")
                != allocation.get("allocation_manifest_digest")
                or manifest.get(
                    "candidate_allocation_post_identity_validation_digest")
                != allocation.get("post_identity_pre_outcome_validation", {}).get(
                    "post_identity_validation_digest")):
            raise RuntimeError("post-identity allocation validation binding mismatch")

    rows_path = out / "branch_rows.jsonl"
    if not rows_path.is_file():
        raise RuntimeError("branch_rows.jsonl is missing")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()
            if line.strip()]
    receipt = json.loads((out / "corpus_receipt.json").read_text())
    payload = receipt.get("corpus_digest_payload")
    if not isinstance(payload, dict) or canonical_digest(payload) != receipt.get(
            "corpus_digest"):
        raise RuntimeError("corpus receipt digest is not independently reproducible")
    if receipt.get("branch_rows_sha256") != file_sha256(rows_path):
        raise RuntimeError("branch row ledger digest disagrees with corpus receipt")
    if receipt.get("state_manifest_digest") != manifest["state_manifest_digest"]:
        raise RuntimeError("corpus receipt is bound to another state manifest")
    expected_count = int(manifest["attempted_branch_count_registered"])
    expected_states = len(manifest["states"])
    valid_count = sum(bool(row.get("valid")) for row in rows)
    expected_payload_bindings = {
        key: manifest[key] for key in CORPUS_BINDING_KEYS
    }
    reconciliations = {
        "payload_pool": payload.get("pool") == manifest["pool"],
        "payload_manifest": payload.get("state_manifest_digest")
                            == manifest["state_manifest_digest"],
        "payload_allocation": payload.get("candidate_allocation_manifest_digest")
                              == manifest["candidate_allocation_manifest_digest"],
        "payload_branch_set": payload.get("branch_identity_set_digest")
                              == manifest["branch_identity_set_digest"],
        "payload_rows_sha": payload.get("branch_rows_sha256")
                            == receipt.get("branch_rows_sha256"),
        "payload_row_digests": payload.get("branch_row_digests")
                               == [row.get("branch_row_digest") for row in rows],
        "payload_state_count": int(payload.get("state_count", -1)) == expected_states,
        "payload_attempted": int(payload.get("attempted_branch_count", -1)) == len(rows),
        "payload_valid": int(payload.get("valid_branch_count", -1)) == valid_count,
        "payload_invalid": int(payload.get("invalid_branch_count", -1))
                           == len(rows) - valid_count,
        "payload_complete": bool(payload.get("complete"))
                            == bool(receipt.get("complete")),
        "payload_bindings": payload.get("bound_digests") == expected_payload_bindings,
        "top_state_count": int(receipt.get("state_count", -1)) == expected_states,
        "top_attempted": int(receipt.get("attempted_branches", -1)) == len(rows),
        "top_valid": int(receipt.get("valid_branches", -1)) == valid_count,
        "top_invalid": int(receipt.get("invalid_branches", -1))
                       == len(rows) - valid_count,
        "top_allocation": receipt.get("candidate_allocation_manifest_digest")
                          == manifest["candidate_allocation_manifest_digest"],
    }
    failed = [name for name, passed in reconciliations.items() if not passed]
    if failed:
        raise RuntimeError(f"corpus receipt reconciliation failed: {failed}")
    if receipt.get("complete") and len(rows) != expected_count:
        raise RuntimeError("completed receipt does not contain every registered branch")
    if not allow_partial and (not receipt.get("complete") or len(rows) != expected_count):
        raise RuntimeError("full encoding requires the exact complete registered corpus")
    for row in rows:
        _validate_row(row, manifest, scientific_contract_digest)
    INVALID_IDS.assert_disjoint(rows, label="branch row ledger", index=invalid_index)
    registered = {
        identity["branch_identity_digest"]: (state, identity)
        for state in manifest["states"] for identity in state["branch_identities"]
    }
    observed = [str(row.get("branch_identity_digest")) for row in rows]
    if len(set(observed)) != len(observed) or not set(observed).issubset(set(registered)):
        raise RuntimeError("branch row identities are duplicated or unregistered")
    for row in rows:
        state, identity = registered[str(row["branch_identity_digest"])]
        if any(row.get(key) != expected for key, expected in (
            ("state_id", state["state_id"]),
            ("state_identity_digest", state["state_identity_digest"]),
            ("candidate", identity["candidate"]),
            ("candidate_index", int(identity["candidate_index"])),
        )):
            raise RuntimeError("branch row relabels a registered branch identity")
    if receipt.get("complete") and set(observed) != set(registered):
        raise RuntimeError("completed corpus omits a registered branch identity")
    return manifest, receipt, rows, contract_lineage


def _load_full_bank_v2_inputs(
        out: Path, *, allow_partial: bool,
        ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]],
                   dict[str, Any]]:
    """Consume the exact V2 producers without opening legacy allocation data."""

    expected_out = OUT_ROOT / "scorer_fit"
    if out != expected_out:
        raise RuntimeError(
            "full-bank V2 encoding is registered only for scorer_fit")
    try:
        bundle = (
            CORPUS_BUILDER
            .load_and_validate_full_bank_v2_branch_outputs_for_consumption(
                out=out, allow_partial=allow_partial))
        artifact = V2_CONTRACT.load_contract_for_consumption(root=ROOT)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise RuntimeError(
            f"full-bank V2 branch producer rejected inputs: {exc}") from exc
    if not isinstance(bundle, Mapping):
        raise RuntimeError("full-bank V2 branch producer returned no bundle")
    if bundle.get("scorer_contract") != artifact:
        raise RuntimeError(
            "full-bank V2 branch producer binds another successor contract")
    manifest = bundle.get("manifest")
    receipt = bundle.get("receipt")
    rows_value = bundle.get("rows")
    if (not isinstance(manifest, Mapping)
            or not _is_full_bank_v2_manifest(manifest)
            or not isinstance(receipt, Mapping)
            or not isinstance(rows_value, list)):
        raise RuntimeError("full-bank V2 branch bundle is malformed")
    manifest = dict(manifest)
    receipt = dict(receipt)
    rows = [dict(row) for row in rows_value]
    successor = V2_CONTRACT.validate_contract_artifact(artifact)["contract"]
    if (manifest.get("state_manifest_digest")
            != successor["state_selector_binding"]["state_manifest_digest"]
            or manifest.get("full_bank_assignment_manifest_digest")
            != successor["state_selector_binding"][
                "assignment_manifest_digest"]
            or manifest.get(
                "scorer_fit_corpus_v2_scorer_contract_digest")
            != successor[V2_CONTRACT.CONTRACT_SELF_KEY]
            or manifest.get(
                "scorer_fit_corpus_v2_scorer_contract_artifact_digest")
            != artifact[V2_CONTRACT.ARTIFACT_SELF_KEY]):
        raise RuntimeError(
            "full-bank V2 manifest/successor contract lineage changed")
    if (len(manifest.get("states", [])) != FULL_BANK_V2_EXPECTED_STATES
            or manifest.get("attempted_branch_count_registered")
            != FULL_BANK_V2_EXPECTED_BRANCHES):
        raise RuntimeError("full-bank V2 registered cardinality changed")
    expected_candidates = list(range(FULL_BANK_V2_EXPECTED_CANDIDATES_PER_STATE))
    if any(state.get("candidate_indices") != expected_candidates
           for state in manifest["states"]):
        raise RuntimeError("full-bank V2 state does not bind candidates 0..11")
    if any(row.get("valid") is not True for row in rows):
        raise RuntimeError("full-bank V2 encoding refuses invalid branches")
    if not allow_partial:
        if (receipt.get("complete") is not True
                or len(rows) != FULL_BANK_V2_EXPECTED_BRANCHES):
            raise RuntimeError(
                "full-bank V2 encoding requires all 1,440 valid branches")
    elif len(rows) not in (FULL_BANK_V2_EXPECTED_CANDIDATES_PER_STATE,
                           FULL_BANK_V2_EXPECTED_BRANCHES):
        raise RuntimeError(
            "full-bank V2 smoke requires one complete twelve-branch state")
    for row in rows:
        _validate_row(
            row, manifest, successor[V2_CONTRACT.CONTRACT_SELF_KEY])
    target = successor["protected_predecessor_scientific_contract"][
        "target_encoder"]
    if (target.get("token_grid") != [24, 32]
            or target.get("tokens") != TOKENS
            or target.get("token_dim") != TOKEN_DIM
            or manifest.get("target_encoder_checkpoint_sha256")
            != target.get("checkpoint_sha256")):
        raise RuntimeError("full-bank V2 target-encoder contract changed")
    return manifest, receipt, rows, dict(artifact)


def normalise(tokens: torch.Tensor) -> torch.Tensor:
    """The frozen factorial target normalisation."""

    return F.layer_norm(tokens, (tokens.shape[-1],))


def encode_paths(arm: Any, encoder: Any, paths: list[str], device: torch.device,
                 dtype: torch.dtype) -> np.ndarray:
    pixels = torch.stack([arm.preprocess(path) for path in paths]).to(
        device=device, dtype=dtype)
    with torch.no_grad():
        tokens = encoder(pixels.unsqueeze(2))
    tokens = normalise(tokens.float())
    if tuple(tokens.shape[1:]) != (TOKENS, TOKEN_DIM):
        raise RuntimeError(f"unexpected token shape {tuple(tokens.shape)}")
    return tokens.cpu().numpy().astype(np.float16)


def _preserve_bad(path: Path, invalid_root: Path, reason: str) -> None:
    if not path.exists():
        return
    invalid_root.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(path) if path.is_file() else "not-a-file"
    target = invalid_root / f"{path.name}.{digest[:16]}.{reason}.invalid"
    suffix = 0
    while target.exists():
        suffix += 1
        target = invalid_root / f"{path.name}.{digest[:16]}.{reason}.{suffix}.invalid"
    path.rename(target)


def _valid_existing(path: Path, record: dict[str, Any] | None,
                    shape: tuple[int, ...]) -> bool:
    if not path.is_file() or not isinstance(record, dict):
        return False
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    return (record.get("shape") == list(shape)
            and int(record.get("byte_count", -1)) == expected_bytes
            and path.stat().st_size == expected_bytes
            and file_sha256(path) == record.get("sha256"))


def _batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _read_regular_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is missing or not a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _validate_full_bank_v2_latent_index(
        index: Mapping[str, Any], *, out: Path,
        manifest: Mapping[str, Any], receipt: Mapping[str, Any],
        rows: list[dict[str, Any]], contract_artifact: Mapping[str, Any],
        require_complete: bool, verify_encoder_checkpoint: bool,
        ) -> dict[str, Any]:
    """Validate the exact atomic-shard index produced by this module."""

    index = dict(index)
    _verify_self_digest(index, "latents_index_digest", "full-bank V2 latent index")
    artifact = V2_CONTRACT.validate_contract_artifact(contract_artifact)
    successor = artifact["contract"]
    expected_bindings = {
        key: manifest[key] for key in FULL_BANK_V2_BINDING_KEYS
    }
    if (index.get("schema") != FULL_BANK_V2_LATENT_INDEX_SCHEMA
            or index.get("status") != STATUS
            or index.get("pool") != "scorer_fit_v2"
            or index.get("corpus_design") != "full-bank-v2"
            or index.get("state_manifest_digest")
            != manifest["state_manifest_digest"]
            or index.get("full_bank_assignment_manifest_digest")
            != manifest["full_bank_assignment_manifest_digest"]
            or index.get("scorer_fit_corpus_v2_scorer_contract_digest")
            != successor[V2_CONTRACT.CONTRACT_SELF_KEY]
            or index.get(
                "scorer_fit_corpus_v2_scorer_contract_artifact_digest")
            != artifact[V2_CONTRACT.ARTIFACT_SELF_KEY]
            or index.get("corpus_digest") != receipt.get("corpus_digest")
            or index.get("branch_rows_sha256")
            != receipt.get("branch_rows_sha256")
            or index.get("corpus_bound_digests") != expected_bindings
            or index.get("tokens") != TOKENS
            or index.get("token_dim") != TOKEN_DIM
            or index.get("horizons") != HORIZONS
            or index.get("context_slots") != CONTEXT_SLOTS
            or index.get("dtype") != "float16"
            or index.get("preprocessing_digest") != PREPROCESSING_SHA256):
        raise RuntimeError("full-bank V2 latent-index lineage changed")
    contexts_value = index.get("context_records")
    horizons_value = index.get("horizon_records")
    if not isinstance(contexts_value, list) or not isinstance(horizons_value, list):
        raise RuntimeError("full-bank V2 latent-index records are malformed")
    contexts = [dict(record) for record in contexts_value]
    horizons = [dict(record) for record in horizons_value]
    if (index.get("context_shape")
            != [len(contexts), *CONTEXT_SHAPE]
            or index.get("horizon_shape")
            != [len(horizons), *HORIZON_SHAPE]
            or index.get("storage_bytes")
            != sum(int(record.get("byte_count", -1))
                   for record in contexts + horizons)):
        raise RuntimeError("full-bank V2 latent-index shape/accounting changed")
    states_by_id = {
        str(state["state_id"]): dict(state) for state in manifest["states"]
    }
    rows_by_key = {
        f"{row['state_id']}|{row['candidate']}": row for row in rows
    }
    if (len(rows_by_key) != len(rows)
            or [record.get("state_id") for record in contexts]
            != sorted(str(record.get("state_id")) for record in contexts)
            or [record.get("key") for record in horizons]
            != sorted(str(record.get("key")) for record in horizons)):
        raise RuntimeError("full-bank V2 latent record order is not canonical")
    if len({str(record.get("state_id")) for record in contexts}) != len(contexts):
        raise RuntimeError("full-bank V2 context latent is duplicated")
    for record in contexts:
        state = states_by_id.get(str(record.get("state_id")))
        if (state is None or record.get("state_identity_digest")
                != state.get("state_identity_digest")):
            raise RuntimeError("full-bank V2 context latent relabels a state")
        path = _resolve_frame(out, str(record.get("path", "")))
        if (not str(record.get("path", "")).startswith(
                f"{FULL_BANK_V2_LATENTS_NAME}/context/")
                or path.is_symlink()
                or not _valid_existing(path, record, CONTEXT_SHAPE)):
            raise RuntimeError("full-bank V2 context latent shard is invalid")
    for record in horizons:
        key = str(record.get("key"))
        row = rows_by_key.get(key)
        if (row is None
                or record.get("state_id") != row.get("state_id")
                or record.get("candidate") != row.get("candidate")
                or record.get("candidate_index") != row.get("candidate_index")
                or record.get("branch_identity_digest")
                != row.get("branch_identity_digest")
                or record.get("assignment_identity_digest")
                != row.get("assignment_identity_digest")):
            raise RuntimeError("full-bank V2 horizon latent relabels a branch")
        path = _resolve_frame(out, str(record.get("path", "")))
        if (not str(record.get("path", "")).startswith(
                f"{FULL_BANK_V2_LATENTS_NAME}/horizon/")
                or path.is_symlink()
                or not _valid_existing(path, record, HORIZON_SHAPE)):
            raise RuntimeError("full-bank V2 horizon latent shard is invalid")
    if require_complete:
        if (index.get("complete") is not True
                or receipt.get("complete") is not True
                or len(contexts) != FULL_BANK_V2_EXPECTED_STATES
                or len(horizons) != FULL_BANK_V2_EXPECTED_BRANCHES
                or set(rows_by_key) != {
                    str(record["key"]) for record in horizons}):
            raise RuntimeError("full-bank V2 latent corpus is incomplete")
    elif (len(contexts) not in (1, FULL_BANK_V2_EXPECTED_STATES)
          or len(horizons) not in (
              FULL_BANK_V2_EXPECTED_CANDIDATES_PER_STATE,
              FULL_BANK_V2_EXPECTED_BRANCHES)):
        raise RuntimeError("full-bank V2 smoke latent cardinality changed")
    if verify_encoder_checkpoint:
        arm = E.VJepa21CroppedV03Arm()
        identity = arm.identity()
        if (E.preprocessing_hash(arm) != PREPROCESSING_SHA256
                or index.get("encoder") != identity
                or identity.get("checkpoint_sha256")
                != manifest["target_encoder_checkpoint_sha256"]):
            raise RuntimeError("full-bank V2 target encoder identity changed")
    return index


def load_and_validate_full_bank_v2_encoding_smoke_for_consumption(
        *, out: Path | None = None, require_protocol_complete: bool = True,
        verify_encoder_checkpoint: bool = False,
        ) -> dict[str, Any]:
    """Validate the one registered 12-candidate V2 smoke and its shards."""

    scorer_fit = OUT_ROOT / "scorer_fit" if out is None else Path(out)
    manifest, receipt, rows, artifact = _load_full_bank_v2_inputs(
        scorer_fit, allow_partial=True)
    branch_bundle = (
        CORPUS_BUILDER
        .load_and_validate_full_bank_v2_branch_outputs_for_consumption(
            out=scorer_fit, allow_partial=True))
    branch_smoke = branch_bundle.get("branch_smoke")
    if not isinstance(branch_smoke, Mapping):
        raise RuntimeError("full-bank V2 branch smoke receipt is absent")
    index = _read_regular_json(
        scorer_fit / FULL_BANK_V2_INDEX_NAME,
        label="full-bank V2 latent index")
    index = _validate_full_bank_v2_latent_index(
        index, out=scorer_fit, manifest=manifest, receipt=receipt, rows=rows,
        contract_artifact=artifact, require_complete=False,
        verify_encoder_checkpoint=verify_encoder_checkpoint)
    smoke = _read_regular_json(
        scorer_fit / FULL_BANK_V2_SMOKE_NAME,
        label="full-bank V2 encoding smoke receipt")
    _verify_self_digest(
        smoke, "smoke_receipt_digest", "full-bank V2 encoding smoke receipt")
    smoke_state_id = str(branch_smoke["state_id"])
    smoke_rows = [row for row in rows if row["state_id"] == smoke_state_id]
    smoke_horizons = [
        record for record in index["horizon_records"]
        if record["state_id"] == smoke_state_id]
    smoke_contexts = [
        record for record in index["context_records"]
        if record["state_id"] == smoke_state_id]
    expected_candidates = list(range(12))
    if (smoke.get("schema") != FULL_BANK_V2_SMOKE_SCHEMA
            or smoke.get("status") != STATUS
            or smoke.get("base_end_to_end_pass") is not True
            or smoke.get("candidate_indices") != expected_candidates
            or smoke.get("branch_count") != 12
            or smoke.get("rendered_horizon_frame_count") != 48
            or smoke.get("true_latent_trajectory_count") != 12
            or smoke.get("true_latent_trajectory_shape")
            != [4, TOKENS, TOKEN_DIM]
            or smoke.get("state_id") != smoke_state_id
            or smoke.get("state_manifest_digest")
            != manifest["state_manifest_digest"]
            or smoke.get("full_bank_assignment_manifest_digest")
            != manifest["full_bank_assignment_manifest_digest"]
            or smoke.get("scorer_fit_corpus_v2_scorer_contract_digest")
            != artifact[V2_CONTRACT.CONTRACT_SELF_KEY]
            or smoke.get(
                "scorer_fit_corpus_v2_scorer_contract_artifact_digest")
            != artifact[V2_CONTRACT.ARTIFACT_SELF_KEY]
            or smoke.get("branch_smoke_receipt_digest")
            != branch_smoke.get("smoke_branch_receipt_digest")
            or smoke.get("corpus_digest") != receipt.get("corpus_digest")
            or smoke.get("latent_index_digest")
            != index["latents_index_digest"]
            or smoke.get("corpus_bound_digests")
            != {key: manifest[key] for key in FULL_BANK_V2_BINDING_KEYS}
            or len(smoke_rows) != 12
            or sorted(int(row["candidate_index"]) for row in smoke_rows)
            != expected_candidates
            or smoke.get("branch_identity_digests")
            != sorted(row["branch_identity_digest"] for row in smoke_rows)
            or smoke.get("branch_row_digests")
            != sorted(row["branch_row_digest"] for row in smoke_rows)
            or len(smoke_contexts) != 1
            or len(smoke_horizons) != 12):
        raise RuntimeError("full-bank V2 encoding smoke evidence changed")
    if require_protocol_complete and (
            smoke.get("pass") is not True
            or smoke.get("zero_new_resume_verified") is not True
            or smoke.get("single_shard_deletion_regeneration_verified")
            is not True
            or smoke.get("smoke_protocol_complete") is not True):
        raise RuntimeError("full-bank V2 smoke durability protocol is incomplete")
    target_record = min(
        smoke_horizons, key=lambda record: int(record["candidate_index"]))
    target_path = _resolve_frame(scorer_fit, str(target_record["path"]))
    latent_inventory = [
        {
            "path": str((_resolve_frame(
                scorer_fit, str(record["path"]))).relative_to(ROOT)),
            "sha256": record["sha256"],
            "byte_count": record["byte_count"],
            "shape": record["shape"],
        }
        for record in sorted(
            smoke_contexts + smoke_horizons,
            key=lambda value: str(value["path"]))
    ]
    registered_paths: set[Path] = {
        scorer_fit / CORPUS_BUILDER.SCORER_FIT_V2_BRANCH_ROWS_NAME,
        scorer_fit / CORPUS_BUILDER.SCORER_FIT_V2_CORPUS_RECEIPT_NAME,
        scorer_fit / CORPUS_BUILDER.SCORER_FIT_V2_BRANCH_SMOKE_RECEIPT_NAME,
        scorer_fit / FULL_BANK_V2_INDEX_NAME,
        scorer_fit / FULL_BANK_V2_SMOKE_NAME,
        scorer_fit / FULL_BANK_V2_ENCODING_SUMMARY_NAME,
    }
    for row in smoke_rows:
        registered_paths.add(
            scorer_fit / CORPUS_BUILDER.SCORER_FIT_V2_ROW_RECORDS_NAME
            / f"{row['branch_identity_digest']}.json")
        for frame in (
                list(row.get("context_frames", []))
                + list(row.get("horizon_frames", []))):
            registered_paths.add(
                _resolve_frame(scorer_fit, str(frame["path"])))
    for record in smoke_contexts + smoke_horizons:
        registered_paths.add(
            _resolve_frame(scorer_fit, str(record["path"])))
    inventory: list[dict[str, Any]] = []
    for path in sorted(registered_paths, key=lambda value: str(value)):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                "full-bank V2 smoke artifact inventory is incomplete")
        inventory.append({
            "path": str(path.relative_to(scorer_fit)),
            "raw_sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        })
    return {
        "manifest": manifest,
        "rows": rows,
        "corpus_receipt": receipt,
        "scorer_contract_artifact": artifact,
        "latent_index": index,
        "branch_smoke_receipt": dict(branch_smoke),
        "encoding_smoke_receipt": smoke,
        "single_shard_regeneration_target": {
            "path": str(target_path.relative_to(ROOT)),
            "sha256": target_record["sha256"],
            "byte_count": target_record["byte_count"],
            "shape": target_record["shape"],
        },
        "registered_smoke_artifact_inventory": inventory,
        "registered_smoke_shard_inventory": latent_inventory,
        "registered_smoke_shard_inventory_digest": canonical_digest(
            latent_inventory),
        "state_count": 1,
        "horizon_latent_count": 12,
        "horizon_shape": [HORIZONS, TOKENS, TOKEN_DIM],
        "invocation_new_context_shards": smoke[
            "invocation_new_context_shards"],
        "invocation_new_horizon_shards": smoke[
            "invocation_new_horizon_shards"],
        "zero_new_resume_verified": smoke["zero_new_resume_verified"],
        "single_registered_shard_regenerated": bool(
            smoke["invocation_new_context_shards"] == 0
            and smoke["invocation_new_horizon_shards"] == 1
            and smoke.get("single_shard_deletion_regeneration_verified")
            is True),
        "only_registered_missing_shard_changed": bool(
            smoke["invocation_new_context_shards"] == 0
            and smoke["invocation_new_horizon_shards"] == 1
            and smoke.get("single_shard_deletion_regeneration_verified")
            is True),
    }


def load_and_validate_full_bank_v2_encoded_corpus_for_consumption(
        *, out: Path | None = None, verify_frame_paths: bool = True,
        verify_encoder_checkpoint: bool = True,
        ) -> dict[str, Any]:
    """Strict producer replay for full-bank V2 scorer training."""

    # The branch producer always verifies every referenced frame.  The flag is
    # retained only for a source-only test seam; it cannot weaken production.
    _ = verify_frame_paths
    scorer_fit = OUT_ROOT / "scorer_fit" if out is None else Path(out)
    manifest, receipt, rows, artifact = _load_full_bank_v2_inputs(
        scorer_fit, allow_partial=False)
    branch_bundle = (
        CORPUS_BUILDER
        .load_and_validate_full_bank_v2_branch_outputs_for_consumption(
            out=scorer_fit, allow_partial=False))
    manifests = branch_bundle.get("manifests")
    if not isinstance(manifests, Mapping):
        raise RuntimeError("full-bank V2 manifest producer bundle is absent")
    assignment = manifests.get("assignment_manifest")
    if not isinstance(assignment, Mapping):
        raise RuntimeError("full-bank V2 assignment manifest is absent")
    index = _read_regular_json(
        scorer_fit / FULL_BANK_V2_INDEX_NAME,
        label="full-bank V2 latent index")
    index = _validate_full_bank_v2_latent_index(
        index, out=scorer_fit, manifest=manifest, receipt=receipt, rows=rows,
        contract_artifact=artifact, require_complete=True,
        verify_encoder_checkpoint=verify_encoder_checkpoint)
    smoke_bundle = (
        load_and_validate_full_bank_v2_encoding_smoke_for_consumption(
            out=scorer_fit, require_protocol_complete=True,
            verify_encoder_checkpoint=False))
    bindings = {
        **{key: manifest[key] for key in FULL_BANK_V2_BINDING_KEYS},
        "state_manifest_digest": manifest["state_manifest_digest"],
        "full_bank_assignment_manifest_digest": assignment[
            "full_bank_assignment_manifest_digest"],
        "corpus_digest": receipt["corpus_digest"],
        "branch_rows_sha256": receipt["branch_rows_sha256"],
        "latent_index_digest": index["latents_index_digest"],
        "encoder_checkpoint_sha256": manifest[
            "target_encoder_checkpoint_sha256"],
    }
    return {
        "state_manifest": manifest,
        "assignment_manifest": dict(assignment),
        "rows": rows,
        "corpus_receipt": receipt,
        "scorer_contract_artifact": artifact,
        "latent_index": index,
        "smoke": smoke_bundle,
        "bindings": bindings,
        "state_count": FULL_BANK_V2_EXPECTED_STATES,
        "horizon_latent_count": FULL_BANK_V2_EXPECTED_BRANCHES,
        "horizon_shape": [HORIZONS, TOKENS, TOKEN_DIM],
        "registered_smoke_shard_inventory_digest": smoke_bundle[
            "registered_smoke_shard_inventory_digest"],
        "invocation_new_context_shards": smoke_bundle[
            "invocation_new_context_shards"],
        "invocation_new_horizon_shards": smoke_bundle[
            "invocation_new_horizon_shards"],
        "zero_new_resume_verified": smoke_bundle[
            "zero_new_resume_verified"],
        "single_registered_shard_regenerated": smoke_bundle[
            "single_registered_shard_regenerated"],
        "only_registered_missing_shard_changed": smoke_bundle[
            "only_registered_missing_shard_changed"],
        "single_shard_regeneration_target": smoke_bundle[
            "single_shard_regeneration_target"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=("scorer_fit", "final_eval"), required=True)
    parser.add_argument(
        "--corpus-design", choices=("legacy", "full-bank-v2"),
        default="legacy",
        help="select the prospective full-bank V2 route explicitly")
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--smoke", action="store_true",
                        help="encode and verify only the registered first state")
    args = parser.parse_args()
    if args.batch_frames < HORIZONS:
        raise SystemExit("--batch-frames must be at least four")

    out = OUT_ROOT / args.pool
    full_bank_v2 = args.corpus_design == "full-bank-v2"
    if full_bank_v2 and args.pool != "scorer_fit":
        raise RuntimeError(
            "full-bank V2 does not authorise final-evaluation encoding")
    if full_bank_v2:
        (manifest, corpus_receipt, all_rows,
         v2_contract_artifact) = _load_full_bank_v2_inputs(
             out, allow_partial=args.smoke)
        contract_lineage = None
        operational_contract_digest = v2_contract_artifact[
            V2_CONTRACT.CONTRACT_SELF_KEY]
    else:
        manifest, corpus_receipt, all_rows, contract_lineage = _load_inputs(
            out, allow_partial=args.smoke, pool=args.pool)
        v2_contract_artifact = None
        operational_contract_digest = (
            contract_lineage["current_scorer_contract_v1_2_digest"]
            if contract_lineage is not None else contract_digest()
        )
    candidates_per_state = (
        FULL_BANK_V2_EXPECTED_CANDIDATES_PER_STATE if full_bank_v2 else 6)
    corpus_binding_keys = _corpus_binding_keys(manifest)
    index_path = out / _output_name(
        full_bank_v2=full_bank_v2, legacy="latents_index.json",
        v2=FULL_BANK_V2_INDEX_NAME)
    encoding_summary_path = out / _output_name(
        full_bank_v2=full_bank_v2, legacy="encoding_invocation_summary.json",
        v2=FULL_BANK_V2_ENCODING_SUMMARY_NAME)
    encoding_smoke_path = out / _output_name(
        full_bank_v2=full_bank_v2, legacy="smoke_encoding_receipt.json",
        v2=FULL_BANK_V2_SMOKE_NAME)
    branch_smoke_path = out / _output_name(
        full_bank_v2=full_bank_v2, legacy="smoke_branch_receipt.json",
        v2=CORPUS_BUILDER.SCORER_FIT_V2_BRANCH_SMOKE_RECEIPT_NAME)
    latent_root_name = _output_name(
        full_bank_v2=full_bank_v2, legacy="latents",
        v2=FULL_BANK_V2_LATENTS_NAME)
    invalid_root_name = _output_name(
        full_bank_v2=full_bank_v2, legacy="invalid_attempts",
        v2=FULL_BANK_V2_INVALID_ATTEMPTS_NAME)
    superseded_root_name = _output_name(
        full_bank_v2=full_bank_v2, legacy="superseded_receipts",
        v2=FULL_BANK_V2_SUPERSEDED_RECEIPTS_NAME)
    expected_states = manifest["states"]
    if args.smoke:
        if args.pool != "scorer_fit":
            raise RuntimeError("the end-to-end smoke is defined only for scorer_fit")
        branch_smoke = json.loads(branch_smoke_path.read_text())
        _verify_self_digest(
            branch_smoke, "smoke_branch_receipt_digest", "branch smoke receipt")
        if not branch_smoke.get("pass"):
            raise RuntimeError("branch smoke has not passed")
        smoke_state_id = str(branch_smoke["state_id"])
        states = [state for state in expected_states
                  if state["state_id"] == smoke_state_id]
        rows = [row for row in all_rows if row["state_id"] == smoke_state_id]
        if (len(states) != 1 or len(rows) != candidates_per_state
                or not all(r.get("valid") for r in rows)):
            raise RuntimeError(
                f"smoke requires one complete {candidates_per_state}-candidate state")
        scientific_contract_matches = (
            branch_smoke.get("scorer_fit_corpus_v2_scorer_contract_digest")
            == operational_contract_digest if full_bank_v2 else
            branch_smoke.get("scorer_contract_v1_2_digest")
            == manifest["scorer_contract_v1_2_digest"])
        if (branch_smoke.get("state_manifest_digest")
                != manifest["state_manifest_digest"]
                or not scientific_contract_matches
                or branch_smoke.get("state_identity_digest")
                != states[0]["state_identity_digest"]
                or branch_smoke.get("branch_identity_digests")
                != sorted(row["branch_identity_digest"] for row in rows)
                or branch_smoke.get("branch_row_digests")
                != sorted(row["branch_row_digest"] for row in rows)
                or branch_smoke.get("corpus_digest")
                != corpus_receipt["corpus_digest"]
                or branch_smoke.get("corpus_bound_digests")
                != {key: manifest[key] for key in corpus_binding_keys}):
            raise RuntimeError("branch smoke receipt is not bound to current exact rows")
    else:
        states = list(expected_states)
        rows = list(all_rows)

    valid_rows = [row for row in rows if row.get("valid")]
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in valid_rows:
        by_state.setdefault(str(row["state_id"]), []).append(row)
    if any(state["state_id"] not in by_state for state in states):
        raise RuntimeError("a selected state has no valid rendered branch")

    # Validate frame identities before loading the 5.1 GB checkpoint.
    context_paths: dict[str, list[str]] = {}
    for state in states:
        sid = str(state["state_id"])
        first = sorted(by_state[sid], key=lambda row: int(row["candidate_index"]))[0]
        records = _frame_records(first, "context")
        paths = _verify_frames(out, records, CONTEXT_SLOTS, f"{sid} context")
        canonical = canonical_digest(records)
        for other in by_state[sid][1:]:
            if canonical_digest(_frame_records(other, "context")) != canonical:
                raise RuntimeError(f"{sid}: branch rows disagree on context frames")
        context_paths[sid] = paths
    horizon_paths: dict[str, list[str]] = {}
    for row in valid_rows:
        key = f"{row['state_id']}|{row['candidate']}"
        horizon_paths[key] = _verify_frames(
            out, _frame_records(row, "horizon"), HORIZONS, f"{key} horizon")

    prior: dict[str, Any] = {}
    if index_path.is_file():
        try:
            prior = json.loads(index_path.read_text())
            _verify_self_digest(prior, "latents_index_digest", "latents index")
            expected_index_schema = (
                FULL_BANK_V2_LATENT_INDEX_SCHEMA if full_bank_v2
                else "go2_branch_corpus_v1_2_latents_index_v2")
            if prior.get("schema") != expected_index_schema:
                raise RuntimeError("prior latent-index schema changed")
            contract_matches = (
                prior.get("scorer_fit_corpus_v2_scorer_contract_digest")
                == operational_contract_digest if full_bank_v2 else
                prior.get("scorer_contract_v1_2_digest")
                == operational_contract_digest)
            allocation_matches = (
                prior.get("full_bank_assignment_manifest_digest")
                == manifest["full_bank_assignment_manifest_digest"]
                if full_bank_v2 else
                prior.get("candidate_allocation_amendment_digest")
                == manifest["candidate_allocation_amendment_digest"])
            if (prior.get("state_manifest_digest")
                    != manifest["state_manifest_digest"]
                    or not contract_matches
                    or not allocation_matches
                    or prior.get("invalid_scorer_identity_exclusion_digest")
                    != manifest["invalid_scorer_identity_exclusion_digest"]
                    or prior.get("corpus_bound_digests")
                    != {key: manifest[key] for key in corpus_binding_keys}):
                raise RuntimeError("prior latent index binds a different frozen corpus")
            if contract_lineage is not None:
                _validate_global_exact_scorer_contract_lineage(
                    prior.get("global_exact_scorer_contract_lineage"),
                    expected=contract_lineage,
                )
            if (prior.get("complete")
                    and prior.get("corpus_digest") != corpus_receipt["corpus_digest"]):
                raise RuntimeError("completed latent index binds a different corpus receipt")
        except Exception as exc:
            if args.smoke:
                raise RuntimeError(
                    "smoke refuses to replace or narrow an invalid existing latent index; "
                    "resume the registered full encoding"
                ) from exc
            _preserve_bad(
                index_path, out / invalid_root_name / "latents", "bad-index")
            prior = {}
    prior_context = {record["state_id"]: record
                     for record in prior.get("context_records", [])}
    prior_horizon = {record["key"]: record
                     for record in prior.get("horizon_records", [])}

    selected_state_ids_pre = {str(state["state_id"]) for state in states}
    selected_keys_pre = {
        f"{row['state_id']}|{row['candidate']}" for row in valid_rows
    }
    prior_complete_full = False
    if args.smoke and prior:
        prior_state_ids = set(prior_context)
        prior_keys = set(prior_horizon)
        if prior.get("complete"):
            expected_full_state_ids = {
                str(state["state_id"]) for state in expected_states
            }
            expected_full_rows = [row for row in all_rows if row.get("valid")]
            expected_full_keys = {
                f"{row['state_id']}|{row['candidate']}" for row in expected_full_rows
            }
            if (prior_state_ids != expected_full_state_ids
                    or prior_keys != expected_full_keys):
                raise RuntimeError("completed latent index does not cover the full corpus")
            state_by_id = {str(state["state_id"]): state for state in expected_states}
            row_by_full_key = {
                f"{row['state_id']}|{row['candidate']}": row
                for row in expected_full_rows
            }
            for sid, record in prior_context.items():
                state = state_by_id[sid]
                latent_path = out / str(record["path"])
                if (not _valid_existing(latent_path, record, CONTEXT_SHAPE)
                        or record.get("state_identity_digest")
                        != state["state_identity_digest"]):
                    raise RuntimeError(
                        "completed latent index has an invalid context shard; "
                        "full resume is required"
                    )
            for key, record in prior_horizon.items():
                row = row_by_full_key[key]
                latent_path = out / str(record["path"])
                if (not _valid_existing(latent_path, record, HORIZON_SHAPE)
                        or record.get("branch_identity_digest")
                        != row["branch_identity_digest"]):
                    raise RuntimeError(
                        "completed latent index has an invalid horizon shard; "
                        "full resume is required"
                    )
            prior_complete_full = True
        elif (not prior_state_ids.issubset(selected_state_ids_pre)
              or not prior_keys.issubset(selected_keys_pre)):
            raise RuntimeError(
                "smoke refuses to narrow or modify a mid-full latent encoding; "
                "resume the registered full encoding"
            )

    context_dir = out / latent_root_name / "context"
    horizon_dir = out / latent_root_name / "horizon"
    invalid_root = out / invalid_root_name / "latents"
    context_records: dict[str, dict[str, Any]] = {}
    horizon_records: dict[str, dict[str, Any]] = {}
    missing_context: list[dict[str, Any]] = []
    missing_horizon: list[dict[str, Any]] = []

    for state in states:
        sid = str(state["state_id"])
        identity = str(state["state_identity_digest"])
        path = context_dir / f"{identity}.f16"
        record = prior_context.get(sid)
        if (_valid_existing(path, record, CONTEXT_SHAPE)
                and record.get("state_identity_digest") == identity):
            context_records[sid] = record
        else:
            if path.exists():
                _preserve_bad(path, invalid_root, "bad-context-shard")
            missing_context.append({"state": state, "path": path})

    row_by_key = {f"{row['state_id']}|{row['candidate']}": row for row in valid_rows}
    for key, row in sorted(row_by_key.items()):
        identity = str(row["branch_identity_digest"])
        path = horizon_dir / f"{identity}.f16"
        record = prior_horizon.get(key)
        if (_valid_existing(path, record, HORIZON_SHAPE)
                and record.get("branch_identity_digest") == identity):
            horizon_records[key] = record
        else:
            if path.exists():
                _preserve_bad(path, invalid_root, "bad-horizon-shard")
            missing_horizon.append({"key": key, "row": row, "path": path})

    new_context_shards = len(missing_context)
    new_horizon_shards = len(missing_horizon)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    arm = E.VJepa21CroppedV03Arm()
    if E.preprocessing_hash(arm) != PREPROCESSING_SHA256:
        raise RuntimeError("frozen V03 preprocessing identity changed")
    encoder_identity = arm.identity()
    target_contract = (
        v2_contract_artifact["contract"]
        ["protected_predecessor_scientific_contract"]["target_encoder"]
        if full_bank_v2 else contract()["target_encoder"])
    if encoder_identity.get("checkpoint_sha256") != target_contract["checkpoint_sha256"]:
        raise RuntimeError("target checkpoint digest disagrees with scorer contract")
    if not missing_context and not missing_horizon:
        encoder = None
    else:
        print(f"loading {arm.name} on {device}; missing contexts={len(missing_context)}, "
              f"horizons={len(missing_horizon)}", flush=True)
        encoder = arm.build(device, dtype)

    started = time.time()
    context_batch_states = max(1, args.batch_frames // CONTEXT_SLOTS)
    for batch in _batches(missing_context, context_batch_states):
        paths = [path for item in batch
                 for path in context_paths[str(item["state"]["state_id"])] ]
        encoded = encode_paths(arm, encoder, paths, device, dtype).reshape(
            len(batch), *CONTEXT_SHAPE)
        for item, array in zip(batch, encoded):
            state = item["state"]
            digest, byte_count = atomic_f16(item["path"], array)
            sid = str(state["state_id"])
            context_records[sid] = {
                "state_id": sid,
                "state_identity_digest": state["state_identity_digest"],
                "path": str(item["path"].relative_to(out)),
                "sha256": digest,
                "byte_count": byte_count,
                "shape": list(CONTEXT_SHAPE),
            }
        print(f"encoded contexts {len(context_records)}/{len(states)}", flush=True)

    horizon_batch_rows = max(1, args.batch_frames // HORIZONS)
    for batch in _batches(missing_horizon, horizon_batch_rows):
        paths = [path for item in batch for path in horizon_paths[item["key"]]]
        encoded = encode_paths(arm, encoder, paths, device, dtype).reshape(
            len(batch), *HORIZON_SHAPE)
        for item, array in zip(batch, encoded):
            row = item["row"]
            digest, byte_count = atomic_f16(item["path"], array)
            horizon_records[item["key"]] = {
                "key": item["key"],
                "state_id": row["state_id"],
                "candidate": row["candidate"],
                "candidate_index": int(row["candidate_index"]),
                "branch_identity_digest": row["branch_identity_digest"],
                **({"assignment_identity_digest":
                        row["assignment_identity_digest"]}
                   if full_bank_v2 else {}),
                "path": str(item["path"].relative_to(out)),
                "sha256": digest,
                "byte_count": byte_count,
                "shape": list(HORIZON_SHAPE),
            }
        print(f"encoded horizons {len(horizon_records)}/{len(valid_rows)}", flush=True)

    selected_state_ids = {str(state["state_id"]) for state in states}
    selected_keys = set(row_by_key)
    ordered_context = [context_records[sid] for sid in sorted(selected_state_ids)]
    ordered_horizon = [horizon_records[key] for key in sorted(selected_keys)]
    full_expected_states = {str(state["state_id"]) for state in expected_states}
    all_valid_rows = [row for row in all_rows if row.get("valid")]
    full_expected_keys = {f"{row['state_id']}|{row['candidate']}" for row in all_valid_rows}
    complete = bool(
        corpus_receipt.get("complete")
        and selected_state_ids == full_expected_states
        and selected_keys == full_expected_keys
        and len(ordered_context) == len(full_expected_states)
        and len(ordered_horizon) == len(full_expected_keys)
    )
    candidate_index = {
        "schema": (FULL_BANK_V2_LATENT_INDEX_SCHEMA if full_bank_v2
                   else "go2_branch_corpus_v1_2_latents_index_v2"),
        "status": STATUS,
        "pool": manifest["pool"] if full_bank_v2 else args.pool,
        **({"corpus_design": "full-bank-v2"} if full_bank_v2 else {}),
        "complete": complete,
        "encoder": encoder_identity,
        "target_encoder_digest": manifest["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            manifest["target_encoder_checkpoint_sha256"],
        "preprocess_contract_digest": manifest["preprocess_contract_digest"],
        "preprocessing_digest": PREPROCESSING_SHA256,
        **({
            "scorer_fit_corpus_v2_scorer_contract_digest":
                operational_contract_digest,
            "scorer_fit_corpus_v2_scorer_contract_artifact_digest":
                v2_contract_artifact[V2_CONTRACT.ARTIFACT_SELF_KEY],
            "scorer_fit_corpus_v2_design_digest": manifest[
                "scorer_fit_corpus_v2_design_digest"],
            "rotation_mask_classification_digest": manifest[
                "rotation_mask_classification_digest"],
            "full_bank_small_completion_selection_digest": manifest[
                "full_bank_small_completion_selection_digest"],
            "full_bank_preoutcome_state_revalidation_digest": manifest[
                "full_bank_preoutcome_state_revalidation_digest"],
            "state_identity_projection_digest": manifest[
                "state_identity_projection_digest"],
            "full_bank_assignment_manifest_digest": manifest[
                "full_bank_assignment_manifest_digest"],
        } if full_bank_v2 else {
            "candidate_allocator_contract_digest":
                manifest["candidate_allocator_contract_digest"],
            "candidate_allocation_amendment_digest":
                manifest["candidate_allocation_amendment_digest"],
            "candidate_allocation_post_identity_validation_digest":
                manifest["candidate_allocation_post_identity_validation_digest"],
            "pre_identity_allocation_validation_digest":
                manifest["pre_identity_allocation_validation_digest"],
            "scorer_contract_v1_2_digest": operational_contract_digest,
        }),
        "invalid_scorer_identity_exclusion_digest":
            manifest["invalid_scorer_identity_exclusion_digest"],
        **({
            "global_exact_scorer_contract_lineage": contract_lineage,
        } if not full_bank_v2 and contract_lineage is not None else {}),
        "corpus_bound_digests": {
            key: manifest[key] for key in corpus_binding_keys
        },
        "state_manifest_digest": manifest["state_manifest_digest"],
        "corpus_digest": corpus_receipt["corpus_digest"],
        "branch_rows_sha256": corpus_receipt["branch_rows_sha256"],
        "tokens": TOKENS,
        "token_dim": TOKEN_DIM,
        "horizons": HORIZONS,
        "context_slots": CONTEXT_SLOTS,
        "dtype": "float16",
        "target_normalisation": "F.layer_norm over the token dimension",
        "preprocess": (
            "dev_frozen_dense_representation_encoders_v1."
            "preprocess_vjepa_v03_crop"
        ),
        "context_shape": [len(ordered_context), *CONTEXT_SHAPE],
        "horizon_shape": [len(ordered_horizon), *HORIZON_SHAPE],
        "context_records": ordered_context,
        "horizon_records": ordered_horizon,
        "storage_bytes": sum(r["byte_count"] for r in ordered_context + ordered_horizon),
    }
    candidate_index["latents_index_digest"] = canonical_digest(candidate_index)
    index = prior if prior_complete_full else candidate_index
    index_rewritten = _write_index_if_changed(index_path, index, prior)

    invocation_summary = {
        "schema": (FULL_BANK_V2_ENCODING_INVOCATION_SCHEMA if full_bank_v2
                   else "go2_branch_corpus_v1_2_encoding_invocation_summary"),
        "status": STATUS,
        "pool": manifest["pool"] if full_bank_v2 else args.pool,
        **({"corpus_design": "full-bank-v2"} if full_bank_v2 else {}),
        "smoke": bool(args.smoke),
        "new_context_shards": new_context_shards,
        "new_horizon_shards": new_horizon_shards,
        "new_shards": new_context_shards + new_horizon_shards,
        "resume_only_verified":
            new_context_shards == 0 and new_horizon_shards == 0,
        "retained_complete_full_index": prior_complete_full,
        "latents_index_rewritten": index_rewritten,
        "latents_index_digest": index["latents_index_digest"],
        "wall_time_s_this_invocation": round(time.time() - started, 3),
    }
    atomic_json(encoding_summary_path, invocation_summary)

    if args.smoke:
        state = states[0]
        smoke_context_shape_ok = index["context_shape"] in (
            [1, *CONTEXT_SHAPE], [len(expected_states), *CONTEXT_SHAPE])
        smoke_horizon_shape_ok = index["horizon_shape"] in (
            [candidates_per_state, *HORIZON_SHAPE],
            [len([row for row in all_rows if row.get("valid")]), *HORIZON_SHAPE],
        )
        smoke_context_records = [
            record for record in index["context_records"]
            if record.get("state_id") == state["state_id"]
        ]
        smoke_horizon_records = [
            record for record in index["horizon_records"]
            if record.get("state_id") == state["state_id"]
        ]
        resume_only_verified = (
            len(smoke_context_records) == 1
            and len(smoke_horizon_records) == candidates_per_state
            and all(_valid_existing(out / record["path"], record, CONTEXT_SHAPE)
                    for record in smoke_context_records)
            and all(_valid_existing(out / record["path"], record, HORIZON_SHAPE)
                    for record in smoke_horizon_records)
        )
        prior_smoke: dict[str, Any] = {}
        if full_bank_v2 and encoding_smoke_path.is_file():
            prior_smoke = json.loads(encoding_smoke_path.read_text())
            _verify_self_digest(
                prior_smoke, "smoke_receipt_digest",
                "full-bank V2 encoding smoke receipt")
            if (prior_smoke.get("schema") != FULL_BANK_V2_SMOKE_SCHEMA
                    or prior_smoke.get("state_identity_digest")
                    != state["state_identity_digest"]
                    or prior_smoke.get(
                        "scorer_fit_corpus_v2_scorer_contract_digest")
                    != operational_contract_digest):
                raise RuntimeError(
                    "existing full-bank V2 smoke receipt binds another run")
        zero_new_resume_verified = bool(
            prior_smoke.get("zero_new_resume_verified")
            or (prior_smoke and new_context_shards == 0
                and new_horizon_shards == 0))
        single_shard_regeneration_verified = bool(
            prior_smoke.get("single_shard_deletion_regeneration_verified")
            or (prior_smoke and prior and new_context_shards == 0
                and new_horizon_shards == 1
                and int(missing_horizon[0]["row"]["candidate_index"]) == 0))
        base_smoke_pass = bool(
            len(rows) == candidates_per_state
            and all(row.get("valid") for row in rows)
            and smoke_context_shape_ok
            and smoke_horizon_shape_ok
            and index["preprocessing_digest"] == PREPROCESSING_SHA256
            and all(row.get("utility") is not None for row in rows)
            and resume_only_verified)
        smoke = {
            "schema": (FULL_BANK_V2_SMOKE_SCHEMA if full_bank_v2
                       else "go2_scorer_fit_end_to_end_smoke_receipt_v1"),
            "status": STATUS,
            "pass": bool(base_smoke_pass and (
                not full_bank_v2 or (
                    zero_new_resume_verified
                    and single_shard_regeneration_verified))),
            **({
                "base_end_to_end_pass": base_smoke_pass,
                "zero_new_resume_verified": zero_new_resume_verified,
                "single_shard_deletion_regeneration_verified":
                    single_shard_regeneration_verified,
                "smoke_protocol_complete": bool(
                    zero_new_resume_verified
                    and single_shard_regeneration_verified),
                "candidate_indices": list(range(12)),
                "branch_count": 12,
                "rendered_horizon_frame_count": 48,
                "true_latent_trajectory_count": 12,
                "true_latent_trajectory_shape": [4, TOKENS, TOKEN_DIM],
            } if full_bank_v2 else {}),
            "state_id": state["state_id"],
            "state_identity_digest": state["state_identity_digest"],
            "branch_identity_digests": sorted(row["branch_identity_digest"] for row in rows),
            "branch_row_digests": sorted(row["branch_row_digest"] for row in rows),
            "state_manifest_digest": manifest["state_manifest_digest"],
            "corpus_digest": corpus_receipt["corpus_digest"],
            "branch_smoke_receipt_digest":
                branch_smoke["smoke_branch_receipt_digest"],
            **({
                "scorer_fit_corpus_v2_scorer_contract_digest":
                    operational_contract_digest,
                "scorer_fit_corpus_v2_scorer_contract_artifact_digest":
                    v2_contract_artifact[V2_CONTRACT.ARTIFACT_SELF_KEY],
                "full_bank_assignment_manifest_digest": manifest[
                    "full_bank_assignment_manifest_digest"],
            } if full_bank_v2 else {
                "scorer_contract_v1_2_digest": operational_contract_digest,
            }),
            **({
                "global_exact_scorer_contract_lineage": contract_lineage,
            } if not full_bank_v2 and contract_lineage is not None else {}),
            **({} if full_bank_v2 else {
                "candidate_allocator_contract_digest":
                    manifest["candidate_allocator_contract_digest"],
                "candidate_allocation_amendment_digest":
                    manifest["candidate_allocation_amendment_digest"],
                "candidate_allocation_post_identity_validation_digest":
                    manifest[
                        "candidate_allocation_post_identity_validation_digest"],
                "pre_identity_allocation_validation_digest":
                    manifest["pre_identity_allocation_validation_digest"],
            }),
            "invalid_scorer_identity_exclusion_digest":
                manifest["invalid_scorer_identity_exclusion_digest"],
            "corpus_bound_digests": {
                key: manifest[key] for key in corpus_binding_keys
            },
            "target_encoder_digest": manifest["target_encoder_digest"],
            "target_encoder_checkpoint_sha256":
                manifest["target_encoder_checkpoint_sha256"],
            "render_contract_digest": manifest["render_contract_digest"],
            "textured_v03_renderer_contract_digest":
                manifest["textured_v03_renderer_contract_digest"],
            "preprocess_contract_digest": manifest["preprocess_contract_digest"],
            "preprocessing_digest": PREPROCESSING_SHA256,
            "context_shape": index["context_shape"],
            "horizon_shape": index["horizon_shape"],
            "index_scope": ("complete_full_corpus" if index.get("complete")
                            else "registered_smoke_state"),
            "materialized_context_shard_count": len(index["context_records"]),
            "materialized_horizon_shard_count": len(index["horizon_records"]),
            "smoke_context_shards_verified": 1,
            "smoke_horizon_shards_verified": candidates_per_state,
            "invocation_new_shard_counts_receipt":
                encoding_summary_path.name,
            "invocation_new_context_shards": new_context_shards,
            "invocation_new_horizon_shards": new_horizon_shards,
            "resume_only_verified": resume_only_verified,
            "latent_index_digest": index["latents_index_digest"],
            "checks": {
                "render_shape_and_preprocessing": True,
                "target_encoder_shape_and_token_order": True,
                "goal_binding": all("goal" in row for row in rows),
                "scorer_row_schema": True,
                "oracle_labels": all(row.get("utility") is not None for row in rows),
                "exact_branch_and_identity_digests": True,
            },
        }
        smoke["smoke_receipt_digest"] = canonical_digest(smoke)
        smoke_path = encoding_smoke_path
        if smoke_path.is_file():
            existing_smoke = json.loads(smoke_path.read_text())
            if existing_smoke != smoke:
                archive = out / superseded_root_name / (
                    f"{smoke_path.stem}."
                    f"{file_sha256(smoke_path)[:16]}.json"
                )
                archive.parent.mkdir(parents=True, exist_ok=True)
                if archive.exists():
                    if archive.read_bytes() != smoke_path.read_bytes():
                        raise RuntimeError("encoding smoke receipt archive collision")
                    smoke_path.unlink()
                else:
                    os.replace(smoke_path, archive)
        if not smoke_path.is_file():
            atomic_json(smoke_path, smoke)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        # V2 smoke is a three-invocation durability protocol.  Intermediate
        # base and zero-new invocations succeed operationally, while only the
        # final receipt is marked PASS and can open full branch generation.
        return 0 if (base_smoke_pass if full_bank_v2 else smoke["pass"]) else 1

    if index.get("complete"):
        smoke_path = encoding_smoke_path
        if not smoke_path.is_file() or not branch_smoke_path.is_file():
            raise RuntimeError("complete scorer-fit encoding lacks required smoke receipts")
        smoke = json.loads(smoke_path.read_text())
        _verify_self_digest(smoke, "smoke_receipt_digest", "encoding smoke receipt")
        branch_smoke = json.loads(branch_smoke_path.read_text())
        _verify_self_digest(
            branch_smoke, "smoke_branch_receipt_digest", "branch smoke receipt")
        smoke_contract_matches = (
            smoke.get("scorer_fit_corpus_v2_scorer_contract_digest")
            == operational_contract_digest if full_bank_v2 else
            smoke.get("scorer_contract_v1_2_digest")
            == operational_contract_digest)
        branch_contract_matches = (
            branch_smoke.get("scorer_fit_corpus_v2_scorer_contract_digest")
            == operational_contract_digest if full_bank_v2 else
            branch_smoke.get("scorer_contract_v1_2_digest")
            == manifest["scorer_contract_v1_2_digest"])
        if not smoke_contract_matches or not branch_contract_matches:
            raise RuntimeError(
                "completed smoke receipts conflate scientific and operational "
                "scorer contracts")
        if not full_bank_v2 and contract_lineage is not None:
            _validate_global_exact_scorer_contract_lineage(
                smoke.get("global_exact_scorer_contract_lineage"),
                expected=contract_lineage,
            )
        smoke_state_id = str(branch_smoke["state_id"])
        smoke_rows = [row for row in all_rows if row["state_id"] == smoke_state_id]
        if (len(smoke_rows) != candidates_per_state
                or not all(row.get("valid") for row in smoke_rows)
                or smoke.get("branch_identity_digests")
                != sorted(row["branch_identity_digest"] for row in smoke_rows)
                or smoke.get("branch_row_digests")
                != sorted(row["branch_row_digest"] for row in smoke_rows)):
            raise RuntimeError("completed encoding smoke evidence does not match exact rows")
        smoke_context = [record for record in index["context_records"]
                         if record["state_id"] == smoke_state_id]
        smoke_horizons = [record for record in index["horizon_records"]
                          if record["state_id"] == smoke_state_id]
        resume_only_verified = (
            len(smoke_context) == 1
            and len(smoke_horizons) == candidates_per_state
            and all(_valid_existing(out / record["path"], record, CONTEXT_SHAPE)
                    for record in smoke_context)
            and all(_valid_existing(out / record["path"], record, HORIZON_SHAPE)
                    for record in smoke_horizons)
        )
        refreshed = {
            **{key: value for key, value in smoke.items()
               if key != "smoke_receipt_digest"},
            "pass": bool(smoke.get("pass") and resume_only_verified),
            "corpus_digest": corpus_receipt["corpus_digest"],
            "branch_smoke_receipt_digest":
                branch_smoke["smoke_branch_receipt_digest"],
            "corpus_bound_digests": {
                key: manifest[key] for key in corpus_binding_keys
            },
            "context_shape": index["context_shape"],
            "horizon_shape": index["horizon_shape"],
            "index_scope": "complete_full_corpus",
            "materialized_context_shard_count": len(index["context_records"]),
            "materialized_horizon_shard_count": len(index["horizon_records"]),
            "resume_only_verified": resume_only_verified,
            "latent_index_digest": index["latents_index_digest"],
        }
        refreshed["smoke_receipt_digest"] = canonical_digest(refreshed)
        if refreshed != smoke:
            archive = out / superseded_root_name / (
                f"{smoke_path.stem}."
                f"{file_sha256(smoke_path)[:16]}.json"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                if archive.read_bytes() != smoke_path.read_bytes():
                    raise RuntimeError("encoding smoke receipt archive collision")
                smoke_path.unlink()
            else:
                os.replace(smoke_path, archive)
            atomic_json(smoke_path, refreshed)

    print(json.dumps({
        "complete": complete,
        "context_shape": index["context_shape"],
        "horizon_shape": index["horizon_shape"],
        "latents_index_digest": index["latents_index_digest"],
        "storage_bytes": index["storage_bytes"],
    }, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
