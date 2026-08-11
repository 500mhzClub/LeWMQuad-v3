#!/usr/bin/env python3
"""Encode corrected Stage-A V03 frames with the frozen factorial target encoder.

The input corpus is ``.generated/go2_counterfactual_fidelity_v1_2``.  Native
224-square PNGs follow the unchanged V03 crop (rows 28:196), bicubic resize to
512x384, ImageNet normalisation, frozen V-JEPA 2.1 ViT-L/384 EMA encoder, and
layer normalisation over each 1024-D token.  Atomic float16 shards make an
interrupted run exact-resumable at the state/branch level.

``--smoke`` encodes the six registered smoke branches.  Run it twice: the
second pass must reuse all seven verified shards and issues the resume-qualified
``smoke_encoding_receipt.json`` required by the full Stage-A branch run.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle.go2_scorer_contract_v1_2 import TARGET_ENCODER  # noqa: E402
from scripts import build_go2_counterfactual_fidelity_stage_a_v1_2 as A  # noqa: E402
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402


STATUS = A.STATUS
OUT_ROOT = A.OUT_ROOT
TOKENS = 768
TOKEN_DIM = 1024
CONTEXT_SHAPE = (A.CONTEXT_SLOTS, TOKENS, TOKEN_DIM)
HORIZON_SHAPE = (A.HORIZONS, TOKENS, TOKEN_DIM)
TARGET_NORMALISATION = (
    "raw final-block tokens rounded to float16; consumers reload float16 as "
    "float32 and apply F.layer_norm over the 1024-D token dimension"
)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_f16(path: Path, array: np.ndarray) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as sink:
        np.asarray(array, dtype=np.float16).tofile(sink)
        sink.flush()
        os.fsync(sink.fileno())
    digest, byte_count = A.file_sha256(temporary), temporary.stat().st_size
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return digest, byte_count


def _preserve_bad(path: Path, reason: str) -> None:
    if not path.exists():
        return
    root = OUT_ROOT / "invalid_attempts/latents"
    root.mkdir(parents=True, exist_ok=True)
    digest = A.file_sha256(path) if path.is_file() else "not-a-file"
    target = root / f"{path.name}.{digest[:16]}.{reason}.invalid"
    suffix = 0
    while target.exists():
        suffix += 1
        target = root / f"{path.name}.{digest[:16]}.{reason}.{suffix}.invalid"
    path.rename(target)
    _fsync_directory(path.parent)
    _fsync_directory(root)


def _valid_shard(path: Path, record: Mapping[str, Any] | None,
                 shape: tuple[int, ...]) -> bool:
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    return bool(
        path.is_file() and isinstance(record, Mapping)
        and record.get("shape") == list(shape)
        and record.get("dtype") == "float16"
        and int(record.get("byte_count", -1)) == expected_bytes
        and path.stat().st_size == expected_bytes
        and A.file_sha256(path) == record.get("sha256")
    )


def encode_paths(arm: Any, encoder: Any, paths: list[str], device: torch.device,
                 dtype: torch.dtype) -> np.ndarray:
    pixels = torch.stack([arm.preprocess(path) for path in paths]).to(
        device=device, dtype=dtype)
    with torch.no_grad():
        tokens = encoder(pixels.unsqueeze(2))
    if tuple(tokens.shape[1:]) != (TOKENS, TOKEN_DIM):
        raise RuntimeError(f"frozen target encoder returned {tuple(tokens.shape)}")
    # Exact factorial cache order: raw encoder output -> float16 on disk.  The
    # scientific consumers reload f16 as f32 and only then layer-normalise.
    return tokens.cpu().numpy().astype(np.float16)


def _sidecar_path(shard: Path) -> Path:
    return shard.with_name(f"{shard.name}.receipt.json")


def _sidecar_common(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage_a_identity_manifest_digest":
            manifest["stage_a_identity_manifest_digest"],
        "assay_spec_digest": manifest["assay_spec_digest"],
        "candidate_bank_digest": manifest["candidate_bank_digest"],
        "oracle_v1_2_digest": manifest["oracle_v1_2_digest"],
        "render_contract_digest": manifest["render_contract_digest"],
        "textured_v03_renderer_contract_digest":
            manifest["textured_v03_renderer_contract_digest"],
        "preprocess_contract_digest": manifest["preprocess_contract_digest"],
        "preprocessing_digest": manifest["preprocessing_digest"],
        "target_encoder_digest": manifest["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            manifest["target_encoder_checkpoint_sha256"],
        "encoder_compute_dtype": "float32",
        "target_normalisation": TARGET_NORMALISATION,
    }


def _load_sidecar(shard: Path, *, expected_kind: str,
                  expected_identity: Mapping[str, Any],
                  expected_frames_digest: str, expected_shape: tuple[int, ...],
                  manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    receipt_path = _sidecar_path(shard)
    if not receipt_path.is_file():
        return None
    try:
        record = json.loads(receipt_path.read_text())
        A._verify_self_digest(record, "latent_shard_receipt_digest",
                              f"{expected_kind} latent shard receipt")
        if (record.get("schema")
                != "go2_counterfactual_fidelity_stage_a_latent_shard_receipt_v1_2"
                or record.get("record_complete") is not True
                or record.get("kind") != expected_kind
                or record.get("source_frame_set_digest") != expected_frames_digest
                or any(record.get(key) != value
                       for key, value in expected_identity.items())
                or any(record.get(key) != value
                       for key, value in _sidecar_common(manifest).items())
                or not _valid_shard(shard, record, expected_shape)
                or record.get("path") != str(shard.relative_to(OUT_ROOT))):
            raise RuntimeError("latent sidecar binding mismatch")
        return record
    except Exception:
        _preserve_bad(receipt_path, "bad-shard-sidecar")
        return None


def _write_sidecar(shard: Path, *, kind: str, identity: Mapping[str, Any],
                   frame_records: list[Mapping[str, Any]], shape: tuple[int, ...],
                   sha256: str, byte_count: int,
                   manifest: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "schema": "go2_counterfactual_fidelity_stage_a_latent_shard_receipt_v1_2",
        "status": STATUS,
        "record_complete": True,
        "kind": kind,
        **identity,
        "source_frame_set_digest": A.canonical_digest(frame_records),
        "source_frame_sha256": [frame["sha256"] for frame in frame_records],
        "path": str(shard.relative_to(OUT_ROOT)),
        "sha256": sha256,
        "byte_count": byte_count,
        "shape": list(shape),
        "dtype": "float16",
        **_sidecar_common(manifest),
    }
    record["latent_shard_receipt_digest"] = A.canonical_digest(record)
    A.atomic_json(_sidecar_path(shard), record)
    return record


def _batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _validate_receipt(receipt: Mapping[str, Any], manifest: Mapping[str, Any],
                      rows: list[dict[str, Any]], *, allow_partial: bool) -> None:
    A._verify_self_digest(receipt, "completion_receipt_digest",
                          "Stage-A completion receipt")
    payload = receipt.get("corpus_digest_payload")
    if not isinstance(payload, Mapping) or A.canonical_digest(payload) != receipt.get(
            "corpus_digest"):
        raise RuntimeError("Stage-A corpus identity payload mismatch")
    ledger = OUT_ROOT / "branch_rows.jsonl"
    if (receipt.get("schema")
            != "go2_counterfactual_fidelity_stage_a_completion_receipt_v1_2"
            or receipt.get("stage_a_identity_manifest_digest")
            != manifest["stage_a_identity_manifest_digest"]
            or receipt.get("branch_rows_sha256") != A.file_sha256(ledger)
            or payload.get("branch_rows_sha256") != receipt["branch_rows_sha256"]
            or payload.get("branch_row_digests")
            != [row["branch_row_digest"] for row in rows]
            or int(receipt.get("attempted_branch_count", -1)) != len(rows)
            or int(receipt.get("valid_branch_count", -1))
            != sum(bool(row.get("valid")) for row in rows)
            or int(receipt.get("oracle_equal_branch_count", -1))
            != sum(bool(row.get("oracle_outcome_equal")) for row in rows)):
        raise RuntimeError("Stage-A completion receipt reconciliation failed")
    if not allow_partial and (receipt.get("complete") is not True
                              or len(rows) != A.EXPECTED_BRANCHES):
        raise RuntimeError("full encoding requires the complete 240-branch corpus")


def load_inputs(*, smoke: bool) -> tuple[dict[str, Any], dict[str, Any],
                                         list[dict[str, Any]], list[dict[str, Any]]]:
    source = A.load_source_evidence()
    manifest = json.loads((OUT_ROOT / "stage_a_identity_manifest.json").read_text())
    A.validate_identity_manifest(manifest, source)
    rows = [json.loads(line) for line in
            (OUT_ROOT / "branch_rows.jsonl").read_text().splitlines() if line.strip()]
    receipt = json.loads((OUT_ROOT / "corpus_receipt.json").read_text())
    _validate_receipt(receipt, manifest, rows, allow_partial=smoke)
    registered = {
        identity["branch_identity_digest"]: (state, identity)
        for state in manifest["states"] for identity in state["branch_identities"]
    }
    seen: set[str] = set()
    for row in rows:
        digest = str(row.get("branch_identity_digest"))
        if digest in seen or digest not in registered:
            raise RuntimeError("Stage-A ledger duplicates or invents a branch identity")
        seen.add(digest)
        state, identity = registered[digest]
        A._validate_branch_row(row, state, identity, manifest, source, OUT_ROOT)
    if smoke:
        smoke_receipt = json.loads((OUT_ROOT / "smoke_receipt.json").read_text())
        A._verify_self_digest(smoke_receipt, "smoke_receipt_digest",
                              "Stage-A branch smoke receipt")
        if smoke_receipt.get("pass") is not True:
            raise RuntimeError("Stage-A six-branch smoke has not passed")
        state_id = str(smoke_receipt["state_id"])
        states = [state for state in manifest["states"] if state["state_id"] == state_id]
        selected = [row for row in rows if row["state_id"] == state_id
                    and int(row["candidate_index"]) < 6]
        if len(states) != 1 or len(selected) != 6:
            raise RuntimeError("Stage-A smoke ledger is not exactly six branches")
        return manifest, receipt, states, selected
    return manifest, receipt, list(manifest["states"]), rows


def _frame_paths(records: list[Mapping[str, Any]], *, kind: str) -> list[str]:
    expected = A.CONTEXT_SLOTS if kind == "context" else A.HORIZONS
    index_key = "slot" if kind == "context" else "horizon"
    start = 0 if kind == "context" else 1
    if len(records) != expected:
        raise RuntimeError(f"{kind} frame count changed")
    paths = []
    for offset, frame in enumerate(records):
        path = A._resolve_output_path(OUT_ROOT, str(frame["path"]))
        A._validate_frame(OUT_ROOT, frame, index_key=index_key,
                          index_value=start + offset,
                          raw_digest=str(frame["raw_manifest_digest"]))
        paths.append(str(path))
    return paths


def _resolve_encoding_scope(
        *, smoke: bool, manifest: dict[str, Any], receipt: dict[str, Any],
        states: list[dict[str, Any]], rows: list[dict[str, Any]],
        prior_index: Mapping[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]],
                   list[dict[str, Any]], bool]:
    """A post-completion smoke validates the complete scope, never a 1/6 downgrade."""

    if not smoke or receipt.get("complete") is not True:
        return manifest, receipt, states, rows, False
    if prior_index.get("complete") is not True:
        raise RuntimeError(
            "complete branch corpus with a partial/missing latent index requires "
            "full encoding without --smoke")
    full_manifest, full_receipt, full_states, full_rows = load_inputs(smoke=False)
    if full_manifest != manifest or full_receipt != receipt:
        raise RuntimeError("complete smoke scope resolved another Stage-A corpus")
    return full_manifest, full_receipt, full_states, full_rows, True


def _assert_smoke_encoding_scope(smoke: bool, receipt: Mapping[str, Any]) -> None:
    if (smoke and receipt.get("complete") is not True
            and int(receipt.get("attempted_branch_count", -1)) != 6):
        raise RuntimeError(
            "Stage-A encoding smoke is unavailable after partial full-branch "
            "progress; resume full branches and then run full encoding")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-frames", type=int, default=8)
    args = parser.parse_args()
    if args.batch_frames < A.HORIZONS:
        raise SystemExit("--batch-frames must be at least four")

    manifest, receipt, states, rows = load_inputs(smoke=args.smoke)
    _assert_smoke_encoding_scope(args.smoke, receipt)
    smoke_states = list(states)
    smoke_rows = list(rows)
    post_complete_smoke = False

    index_path = OUT_ROOT / "latents_index.json"
    prior: dict[str, Any] = {}
    if index_path.is_file():
        try:
            prior = json.loads(index_path.read_text())
            A._verify_self_digest(prior, "latents_index_digest", "Stage-A latent index")
            if (prior.get("schema")
                    != "go2_counterfactual_fidelity_stage_a_latents_index_v1_2"
                    or prior.get("stage_a_identity_manifest_digest")
                    != manifest["stage_a_identity_manifest_digest"]
                    or prior.get("assay_spec_digest") != manifest["assay_spec_digest"]
                    or (prior.get("target_encoder_digest")
                        != manifest["target_encoder_digest"])):
                raise RuntimeError("prior latent index binds another Stage-A identity")
            if prior.get("complete") and prior.get("corpus_digest") != receipt["corpus_digest"]:
                raise RuntimeError("completed latent index binds another corpus")
        except Exception:
            _preserve_bad(index_path, "bad-index")
            prior = {}
    manifest, receipt, states, rows, post_complete_smoke = _resolve_encoding_scope(
        smoke=args.smoke, manifest=manifest, receipt=receipt,
        states=states, rows=rows, prior_index=prior)
    index_corpus_digest = receipt["corpus_digest"]
    index_branch_rows_sha256 = receipt["branch_rows_sha256"]
    if args.smoke and not post_complete_smoke:
        smoke_text = "".join(
            json.dumps(A.V1._jsonable(row), sort_keys=True) + "\n"
            for row in smoke_rows)
        index_branch_rows_sha256 = hashlib.sha256(
            smoke_text.encode("utf-8")).hexdigest()
        index_corpus_digest = A.canonical_digest(A._corpus_identity_payload(
            manifest, smoke_rows, index_branch_rows_sha256))
    index_corpus_binding = {
        "corpus_digest": index_corpus_digest,
        "branch_rows_sha256": index_branch_rows_sha256,
    }
    if prior:
        try:
            if prior.get("complete") is True:
                A._validate_latents_index(
                    prior, manifest, receipt, states, rows, complete=True)
            elif args.smoke and not post_complete_smoke:
                A._validate_latents_index(
                    prior, manifest, index_corpus_binding, states, rows,
                    complete=False)
        except Exception:
            _preserve_bad(index_path, "bad-index-reconciliation")
            prior = {}

    context_paths: dict[str, list[str]] = {}
    for state in states:
        record = json.loads(A._state_record_path(OUT_ROOT, state).read_text())
        A._validate_state_record(record, state, manifest, OUT_ROOT)
        context_paths[state["state_id"]] = _frame_paths(
            record["context_frames"], kind="context")
    horizon_paths = {
        f"{row['state_id']}|{row['candidate']}": _frame_paths(
            row["horizon_frames"], kind="horizon")
        for row in rows
    }
    prior_context = {record["state_id"]: record
                     for record in prior.get("context_records", [])}
    prior_horizon = {record["branch_key"]: record
                     for record in prior.get("horizon_records", [])}

    context_records: dict[str, dict[str, Any]] = {}
    horizon_records: dict[str, dict[str, Any]] = {}
    missing_context: list[dict[str, Any]] = []
    missing_horizon: list[dict[str, Any]] = []
    for state in states:
        state_id = str(state["state_id"])
        path = OUT_ROOT / "latents/context" / f"{state['state_identity_digest']}.f16"
        state_record = json.loads(A._state_record_path(OUT_ROOT, state).read_text())
        identity = {
            "state_id": state_id,
            "state_identity_digest": state["state_identity_digest"],
            "state_record_digest": state_record["state_record_digest"],
        }
        frame_digest = A.canonical_digest(state_record["context_frames"])
        record = _load_sidecar(
            path, expected_kind="context", expected_identity=identity,
            expected_frames_digest=frame_digest, expected_shape=CONTEXT_SHAPE,
            manifest=manifest)
        if record is not None:
            context_records[state_id] = record
        else:
            if path.exists():
                _preserve_bad(path, "bad-context-shard")
            missing_context.append({"state": state, "state_record": state_record,
                                    "identity": identity, "path": path})
    for row in rows:
        key = f"{row['state_id']}|{row['candidate']}"
        path = OUT_ROOT / "latents/horizon" / f"{row['branch_identity_digest']}.f16"
        identity = {
            "branch_key": key,
            "state_id": row["state_id"],
            "candidate": row["candidate"],
            "candidate_index": int(row["candidate_index"]),
            "branch_identity_digest": row["branch_identity_digest"],
            "branch_row_digest": row["branch_row_digest"],
        }
        record = _load_sidecar(
            path, expected_kind="horizon", expected_identity=identity,
            expected_frames_digest=A.canonical_digest(row["horizon_frames"]),
            expected_shape=HORIZON_SHAPE, manifest=manifest)
        if record is not None:
            horizon_records[key] = record
        else:
            if path.exists():
                _preserve_bad(path, "bad-horizon-shard")
            missing_horizon.append({"row": row, "key": key, "identity": identity,
                                    "path": path})

    arm = E.VJepa21CroppedV03Arm()
    if E.preprocessing_hash(arm) != A.PREPROCESSING_DIGEST:
        raise RuntimeError("frozen V03 crop preprocessing identity changed")
    encoder_identity = dict(arm.identity())
    checkpoint_path = Path(str(encoder_identity["checkpoint_path"]))
    checkpoint_byte_count = checkpoint_path.stat().st_size
    if (encoder_identity.get("checkpoint_sha256") != TARGET_ENCODER["checkpoint_sha256"]
            or checkpoint_byte_count != int(TARGET_ENCODER["checkpoint_byte_count"])):
        raise RuntimeError("frozen target encoder checkpoint identity changed")
    encoder_identity["checkpoint_byte_count"] = checkpoint_byte_count
    encoder_identity["compute_dtype"] = "float32"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    encoder = None if not missing_context and not missing_horizon else arm.build(device, dtype)
    started = time.time()

    for batch in _batches(missing_context,
                          max(1, args.batch_frames // A.CONTEXT_SLOTS)):
        paths = [path for item in batch
                 for path in context_paths[str(item["state"]["state_id"])] ]
        encoded = encode_paths(arm, encoder, paths, device, dtype).reshape(
            len(batch), *CONTEXT_SHAPE)
        for item, array in zip(batch, encoded):
            state = item["state"]
            digest, byte_count = atomic_f16(item["path"], array)
            context_records[state["state_id"]] = _write_sidecar(
                item["path"], kind="context", identity=item["identity"],
                frame_records=item["state_record"]["context_frames"],
                shape=CONTEXT_SHAPE, sha256=digest, byte_count=byte_count,
                manifest=manifest)
    for batch in _batches(missing_horizon,
                          max(1, args.batch_frames // A.HORIZONS)):
        paths = [path for item in batch for path in horizon_paths[item["key"]]]
        encoded = encode_paths(arm, encoder, paths, device, dtype).reshape(
            len(batch), *HORIZON_SHAPE)
        for item, array in zip(batch, encoded):
            row = item["row"]
            digest, byte_count = atomic_f16(item["path"], array)
            horizon_records[item["key"]] = _write_sidecar(
                item["path"], kind="horizon", identity=item["identity"],
                frame_records=row["horizon_frames"], shape=HORIZON_SHAPE,
                sha256=digest, byte_count=byte_count, manifest=manifest)

    selected_state_ids = {str(state["state_id"]) for state in states}
    selected_keys = {f"{row['state_id']}|{row['candidate']}" for row in rows}
    ordered_context = [context_records[key] for key in sorted(selected_state_ids)]
    ordered_horizon = [horizon_records[key] for key in sorted(selected_keys)]
    complete = bool(
        receipt.get("complete")
        and len(ordered_context) == A.EXPECTED_STATES
        and len(ordered_horizon) == A.EXPECTED_BRANCHES
    )
    index = {
        "schema": "go2_counterfactual_fidelity_stage_a_latents_index_v1_2",
        "status": STATUS,
        "complete": complete,
        "state_count": len(ordered_context),
        "branch_count": len(ordered_horizon),
        "encoder": encoder_identity,
        "target_encoder_digest": manifest["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            manifest["target_encoder_checkpoint_sha256"],
        "preprocess_contract_digest": manifest["preprocess_contract_digest"],
        "preprocessing_digest": manifest["preprocessing_digest"],
        "render_contract_digest": manifest["render_contract_digest"],
        "textured_v03_renderer_contract_digest":
            manifest["textured_v03_renderer_contract_digest"],
        "stage_a_identity_manifest_digest":
            manifest["stage_a_identity_manifest_digest"],
        "assay_spec_digest": manifest["assay_spec_digest"],
        "candidate_bank_digest": manifest["candidate_bank_digest"],
        "oracle_v1_2_digest": manifest["oracle_v1_2_digest"],
        "source_state_manifest_digest": manifest["source_state_manifest_digest"],
        "source_pilot_branch_ledger_sha256":
            manifest["source_pilot_branch_ledger_sha256"],
        "corpus_digest": index_corpus_digest,
        "branch_rows_sha256": index_branch_rows_sha256,
        "tokens": TOKENS, "token_dim": TOKEN_DIM,
        "context_slots": A.CONTEXT_SLOTS, "horizons": A.HORIZONS,
        "dtype": "float16",
        "encoder_compute_dtype": "float32",
        "target_normalisation": TARGET_NORMALISATION,
        "preprocess": "preprocess_vjepa_v03_crop",
        "context_shape": [len(ordered_context), *CONTEXT_SHAPE],
        "horizon_shape": [len(ordered_horizon), *HORIZON_SHAPE],
        "context_records": ordered_context,
        "horizon_records": ordered_horizon,
        "storage_bytes": sum(record["byte_count"]
                             for record in ordered_context + ordered_horizon),
    }
    index["latents_index_digest"] = A.canonical_digest(index)
    A.atomic_json(index_path, index)

    summary = {
        "schema": "go2_counterfactual_fidelity_stage_a_encoding_summary_v1_2",
        "complete": complete,
        "new_context_shards_this_invocation": len(missing_context),
        "new_horizon_shards_this_invocation": len(missing_horizon),
        "runtime_s_this_invocation": round(time.time() - started, 6),
        "latents_index_digest": index["latents_index_digest"],
    }
    A.atomic_json(OUT_ROOT / "encoding_summary.json", summary)
    if args.smoke:
        branch_smoke = json.loads((OUT_ROOT / "smoke_receipt.json").read_text())
        raw_identity = branch_smoke.get("raw_frame_identity")
        expected_context_count = (A.EXPECTED_STATES if post_complete_smoke else 1)
        expected_horizon_count = (A.EXPECTED_BRANCHES if post_complete_smoke else 6)
        smoke_receipt = {
            "schema": "go2_counterfactual_fidelity_stage_a_smoke_encoding_receipt_v1_2",
            "status": STATUS,
            "pass": bool(
                len(ordered_context) == expected_context_count
                and len(ordered_horizon) == expected_horizon_count
                and index["context_shape"]
                == [expected_context_count, *CONTEXT_SHAPE]
                and index["horizon_shape"]
                == [expected_horizon_count, *HORIZON_SHAPE]
                and isinstance(raw_identity, Mapping) and raw_identity.get("identical") is True
            ),
            "resume_only_verified": bool(not missing_context and not missing_horizon),
            "new_context_shards_this_invocation": len(missing_context),
            "new_horizon_shards_this_invocation": len(missing_horizon),
            "raw_frame_identity": raw_identity,
            "state_id": smoke_states[0]["state_id"],
            "branch_identity_digests": [
                row["branch_identity_digest"] for row in smoke_rows],
            "branch_row_digests": [row["branch_row_digest"] for row in smoke_rows],
            "stage_a_identity_manifest_digest":
                manifest["stage_a_identity_manifest_digest"],
            "assay_spec_digest": manifest["assay_spec_digest"],
            "partial_corpus_digest": index_corpus_digest,
            "render_contract_digest": manifest["render_contract_digest"],
            "textured_v03_renderer_contract_digest":
                manifest["textured_v03_renderer_contract_digest"],
            "preprocess_contract_digest": manifest["preprocess_contract_digest"],
            "preprocessing_digest": manifest["preprocessing_digest"],
            "target_encoder_digest": manifest["target_encoder_digest"],
            "target_encoder_checkpoint_sha256":
                manifest["target_encoder_checkpoint_sha256"],
            "context_shape": index["context_shape"],
            "horizon_shape": index["horizon_shape"],
            "latents_index_digest": index["latents_index_digest"],
        }
        smoke_receipt["smoke_encoding_receipt_digest"] = A.canonical_digest(
            smoke_receipt)
        A.atomic_json(OUT_ROOT / "smoke_encoding_receipt.json", smoke_receipt)
        print(json.dumps(smoke_receipt, indent=2, sort_keys=True))
        return 0 if smoke_receipt["pass"] else 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
