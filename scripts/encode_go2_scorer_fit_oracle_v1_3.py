#!/usr/bin/env python3
"""Encode only the frozen oracle-v1.3 scorer training view.

This successor is intentionally narrow.  It consumes the workflow producer's
validated 96-fit/24-fresh-calibration view, verifies every label and rendered
frame before opening the frozen target encoder, and writes one true H=1..4
latent trajectory per training row.  It has no predictor or final-evaluation
route.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import go2_branch_oracle_v1_3 as ORACLE  # noqa: E402
from lewm.oracle import go2_scorer_fit_oracle_v1_3_contract as CONTRACT  # noqa: E402
from lewm.oracle import go2_scorer_fit_corpus_v2_design as V2_DESIGN  # noqa: E402
from lewm.oracle.go2_scorer_contract_v1_2 import (  # noqa: E402
    TARGET_ENCODER, preprocess_contract_digest, target_encoder_digest,
)
from scripts import run_go2_scorer_fit_oracle_v1_3 as WORKFLOW  # noqa: E402


STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
LATENT_INDEX_SCHEMA = CONTRACT.LATENT_INDEX_SCHEMA
ENCODING_RECEIPT_SCHEMA = CONTRACT.ENCODING_RECEIPT_SCHEMA
LATENT_INDEX_SELF_KEY = "latent_index_digest"
ENCODING_RECEIPT_SELF_KEY = "encoding_receipt_digest"

EXPECTED_ROWS = 1_440
EXPECTED_FIT_STATES = 96
EXPECTED_FIT_ROWS = 1_152
EXPECTED_CALIBRATION_STATES = 24
EXPECTED_CALIBRATION_ROWS = 288
EXPECTED_CANDIDATES = tuple(range(12))
EXPECTED_FAMILIES = tuple(V2_DESIGN.FAMILIES)
EXPECTED_STRATA = tuple(V2_DESIGN.STRATA)
EXPECTED_SOURCE_KINDS = {
    "V2_VALID_ADOPTION": 1_146,
    "V13_REPLAY_OVERLAY": 6,
    "V13_FRESH_CALIBRATION": 288,
}
SOURCE_DIGEST_KEYS = (
    "v2_corpus_digest",
    "equivalence_receipt_digest",
    "replay_overlay_manifest_digest",
    "fresh_calibration_state_manifest_digest",
    "fresh_calibration_corpus_digest",
)
TOKENS = 768
TOKEN_DIM = 1_024
HORIZONS = 4
HORIZON_SHAPE = (HORIZONS, TOKENS, TOKEN_DIM)
PREPROCESSING_SHA256 = (
    "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"
)


class V13EncodingError(RuntimeError):
    """The frozen training view or target-encoding lineage is invalid."""


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13EncodingError(message)


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _generated_root(root: Path = ROOT) -> Path:
    configured = Path(CONTRACT.GENERATED_ROOT)
    return configured if configured.is_absolute() else root / configured


def _require_registered_generated_root(root: Path = ROOT) -> Path:
    logical = _generated_root(root)
    if root.resolve() == ROOT.resolve():
        WORKFLOW.guarded_output_path(".storage-probe", out_root=logical)
    return logical


def encoded_root(root: Path = ROOT) -> Path:
    return _generated_root(root) / "encoded_training_view"


def latent_index_path(root: Path = ROOT) -> Path:
    return encoded_root(root) / "latent_index.json"


def encoding_receipt_path(root: Path = ROOT) -> Path:
    return encoded_root(root) / "encoding_receipt.json"


def _signed(value: Mapping[str, Any], self_key: str) -> dict[str, Any]:
    payload = dict(value)
    _require(self_key not in payload, f"{self_key} already present")
    payload[self_key] = canonical_digest(payload)
    return payload


def _validate_signed(value: Mapping[str, Any], self_key: str,
                     label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    payload = dict(value)
    recorded = payload.pop(self_key, None)
    _require(_is_digest(recorded), f"{label} self digest is malformed")
    _require(recorded == canonical_digest(payload),
             f"{label} self digest does not verify")
    payload[self_key] = recorded
    return payload


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _atomic_operational_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically refresh resumable operational metadata, never a terminal."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        raw = _json_bytes(value)
        position = 0
        while position < len(raw):
            position += os.write(descriptor, raw[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_json_once(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        _require(path.is_file() and not path.is_symlink(),
                 f"{label} path is not a regular file")
        _require(path.read_bytes() == raw, f"{label} is already different")
        return
    try:
        position = 0
        while position < len(raw):
            position += os.write(descriptor, raw[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _row_role(row: Mapping[str, Any]) -> str:
    role = row.get("role", row.get("split_role"))
    if "role" in row and "split_role" in row:
        _require(row["role"] == row["split_role"],
                 "training row role aliases disagree")
    _require(role in {"fit", "calibration"}, "training row role is invalid")
    return str(role)


def _label(row: Mapping[str, Any], key: str) -> float:
    projection = row.get("label_projection")
    value = (projection.get(key) if isinstance(projection, Mapping)
             else row.get(key))
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(float(value)),
             f"training row {key} label is missing or nonfinite")
    return float(value)


def _old_calibration_disposition(view: Mapping[str, Any]) -> dict[str, Any]:
    value = view.get("historical_calibration_disposition")
    _require(isinstance(value, Mapping),
             "historical calibration disposition is absent")
    disposition = dict(value)
    self_keys = [key for key in disposition
                 if key.endswith("disposition_digest")]
    _require(len(self_keys) == 1, "historical disposition self key changed")
    _validate_signed(disposition, self_keys[0], "historical disposition")
    _require(disposition.get("state_count") == 24
             and disposition.get("branch_count") == 288
             and disposition.get("status") == "DEVELOPMENT_ONLY"
             and disposition.get("qualification_eligible") is False
             and disposition.get("discarded") is False,
             "historical calibration disposition changed")
    state_digests = disposition.get("state_identity_digests")
    scene_ids = disposition.get("scene_ids")
    _require(isinstance(state_digests, list) and len(state_digests) == 24
             and len(set(state_digests)) == 24
             and all(_is_digest(value) for value in state_digests),
             "historical calibration state identities changed")
    _require(isinstance(scene_ids, list) and len(scene_ids) == 24
             and len(set(scene_ids)) == 24
             and all(isinstance(value, str) and value for value in scene_ids),
             "historical calibration scene identities changed")
    expected_states = set(CONTRACT.OLD_CALIBRATION_STATES)
    _require(set(state_digests) == {
                 row.state_identity_digest for row in expected_states
             }
             and set(scene_ids) == {row.scene_id for row in expected_states},
             "historical calibration disposition is not the frozen old 24")
    return disposition


def validate_training_view_structure(view: Mapping[str, Any]) -> dict[str, Any]:
    """Independently validate the materialized producer view before I/O."""

    _require(isinstance(view, Mapping), "training view is not an object")
    value = dict(view)
    expected_schema = getattr(
        WORKFLOW, "TRAINING_VIEW_SCHEMA",
        "go2_scorer_fit_oracle_v1_3_training_view_v1")
    self_key = getattr(WORKFLOW, "TRAINING_VIEW_SELF_KEY", "training_view_digest")
    _require(value.get("schema") == expected_schema,
             "training-view schema changed")
    _require(value.get("status") == WORKFLOW.STATUS
             and value.get("complete") is True
             and value.get("missing_label_count") == 0,
             "training view is not complete with zero missing labels")
    _require(_is_digest(value.get(self_key)),
             "training-view digest is malformed")
    _require(value.get("oracle_v1_3_digest") == ORACLE.oracle_digest(),
             "training view binds another oracle")
    _require(value.get("scorer_fit_oracle_v1_3_contract_digest")
             == CONTRACT.contract_digest(),
             "training view binds another scorer-fit contract")
    for key in ("authority_digest", *SOURCE_DIGEST_KEYS):
        _require(_is_digest(value.get(key)),
                 f"training view {key} is malformed")
    _require(value["v2_corpus_digest"] == CONTRACT.FROZEN_CORPUS_DIGEST,
             "training view binds another protected V2 corpus")
    _require(value.get("source_kind_counts") == EXPECTED_SOURCE_KINDS,
             "training-view declared source-kind counts changed")
    expected_counts = {
        "fit_state_count": EXPECTED_FIT_STATES,
        "fit_branch_count": EXPECTED_FIT_ROWS,
        "calibration_state_count": EXPECTED_CALIBRATION_STATES,
        "calibration_branch_count": EXPECTED_CALIBRATION_ROWS,
        "row_count": EXPECTED_ROWS,
    }
    _require(all(value.get(key) == count
                 for key, count in expected_counts.items()),
             "training-view cardinalities changed")
    disposition = _old_calibration_disposition(value)
    old_state_digests = set(disposition["state_identity_digests"])
    old_scene_ids = set(disposition["scene_ids"])

    rows = value.get("rows")
    _require(isinstance(rows, list) and len(rows) == EXPECTED_ROWS,
             "training view does not contain exactly 1,440 rows")
    state_rows: dict[str, list[Mapping[str, Any]]] = {}
    row_digests: set[str] = set()
    source_kinds: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    family_role: Counter[tuple[str, str]] = Counter()
    family_stratum_role: Counter[tuple[str, str, str]] = Counter()
    state_scene: dict[str, str] = {}
    state_role: dict[str, str] = {}
    state_identity: dict[str, str] = {}
    state_design: dict[str, tuple[str, str]] = {}
    branch_digests: set[str] = set()
    for row in rows:
        _require(isinstance(row, Mapping), "training row is not an object")
        role = _row_role(row)
        state_id = row.get("state_id")
        scene_id = row.get("scene_id")
        family = row.get("family")
        stratum = row.get("stratum")
        state_digest = row.get("state_identity_digest")
        row_digest = row.get("training_view_row_digest")
        branch_digest = row.get("branch_identity_digest")
        source_kind = row.get("source_kind")
        candidate = row.get("candidate_index")
        _require(isinstance(state_id, str) and state_id,
                 "training row state identity is absent")
        _require(isinstance(scene_id, str) and scene_id,
                 "training row scene identity is absent")
        _require(family in EXPECTED_FAMILIES and stratum in EXPECTED_STRATA,
                 "training row family/stratum changed")
        _require(_is_digest(state_digest),
                 "training row state digest is malformed")
        _require(_is_digest(row_digest) and row_digest not in row_digests,
                 "training-view row digest is malformed or duplicated")
        _require(_is_digest(branch_digest)
                 and branch_digest not in branch_digests,
                 "training branch identity is malformed or duplicated")
        _require(source_kind in EXPECTED_SOURCE_KINDS,
                 "training row source kind changed")
        _require(type(candidate) is int and candidate in EXPECTED_CANDIDATES,
                 "training row candidate index changed")
        row_digests.add(str(row_digest))
        branch_digests.add(str(branch_digest))
        source_kinds[str(source_kind)] += 1
        roles[role] += 1
        state_rows.setdefault(str(state_id), []).append(row)
        prior_scene = state_scene.setdefault(str(state_id), str(scene_id))
        prior_role = state_role.setdefault(str(state_id), role)
        prior_identity = state_identity.setdefault(str(state_id), str(state_digest))
        prior_design = state_design.setdefault(
            str(state_id), (str(family), str(stratum)))
        _require((prior_scene, prior_role, prior_identity)
                 == (scene_id, role, state_digest),
                 "one state maps to multiple identities, scenes, or roles")
        _require(prior_design == (family, stratum),
                 "one state maps to multiple family/stratum cells")

        progress = _label(row, "progress")
        safety = _label(row, "safety")
        completion = _label(row, "completion")
        utility = _label(row, "utility")
        _require(0.0 <= safety <= 1.0,
                 "training row safety is outside [0,1]")
        _require(completion in {0.0, 1.0},
                 "training row completion is not binary")
        expected_utility = float(progress - 2.0 * safety + 0.5 * completion)
        _require(math.isclose(utility, expected_utility,
                              rel_tol=0.0, abs_tol=1e-12),
                 "training row composite utility changed")
        action_blocks = row.get("action_blocks")
        goal = row.get("goal_binding_input")
        _require(isinstance(action_blocks, list) and len(action_blocks) == 4
                 and all(isinstance(block, list) and len(block) == 10
                         and all(not isinstance(item, bool)
                                 and isinstance(item, (int, float))
                                 and math.isfinite(float(item))
                                 for item in block)
                         for block in action_blocks),
                 "training row action input is not frozen 4x10 post-slew")
        _require(isinstance(goal, list) and len(goal) == 3
                 and all(not isinstance(item, bool)
                         and isinstance(item, (int, float))
                         and math.isfinite(float(item)) for item in goal),
                 "training row goal input is not a finite 3-vector")
        if role == "fit":
            _require(source_kind in {"V2_VALID_ADOPTION", "V13_REPLAY_OVERLAY"},
                     "fit row uses a fresh-calibration source")
        else:
            _require(source_kind == "V13_FRESH_CALIBRATION",
                     "calibration row is not fresh")
        _require(state_digest not in old_state_digests
                 and scene_id not in old_scene_ids,
                 "historical calibration identity entered the training view")

    _require(dict(source_kinds) == EXPECTED_SOURCE_KINDS,
             "training-view source-kind counts changed")
    expected_replays = {
        row.branch_identity_digest: row
        for row in CONTRACT.FAILED_BRANCH_IDENTITIES
        if row.split_role == "fit"
    }
    actual_replays = {
        str(row["branch_identity_digest"]): row for row in rows
        if row["source_kind"] == "V13_REPLAY_OVERLAY"
    }
    _require(set(actual_replays) == set(expected_replays)
             and all(
                 (actual_replays[digest]["state_id"],
                  actual_replays[digest]["state_identity_digest"],
                  actual_replays[digest]["scene_id"],
                  actual_replays[digest]["candidate_index"],
                  _row_role(actual_replays[digest]))
                 == (expected.state_id, expected.state_identity_digest,
                     expected.scene_id, expected.candidate_index, "fit")
                 for digest, expected in expected_replays.items()),
             "training view does not use the exact six fit replay overlays")
    _require(dict(roles) == {
        "fit": EXPECTED_FIT_ROWS,
        "calibration": EXPECTED_CALIBRATION_ROWS,
    }, "training-view split row counts changed")
    _require(len(state_rows) == EXPECTED_FIT_STATES + EXPECTED_CALIBRATION_STATES,
             "training view does not contain exactly 120 states")
    _require(len(set(state_scene.values())) == len(state_rows),
             "training-view states are not scene-disjoint")
    for state_id, selected in state_rows.items():
        _require(len(selected) == 12
                 and sorted(int(row["candidate_index"]) for row in selected)
                 == list(EXPECTED_CANDIDATES),
                 f"state {state_id} lacks the exact twelve-candidate bank")
        exemplar = selected[0]
        key = (str(exemplar["family"]), state_role[state_id])
        family_role[key] += 1
        family_stratum_role[(str(exemplar["family"]),
                             str(exemplar["stratum"]),
                             state_role[state_id])] += 1
    for family in EXPECTED_FAMILIES:
        _require(family_role[(family, "fit")] == 12
                 and family_role[(family, "calibration")] == 3,
                 f"{family} is not 12 fit / 3 fresh calibration states")
        for stratum in EXPECTED_STRATA:
            _require(family_stratum_role[(family, stratum, "fit")] == 4
                     and family_stratum_role[
                         (family, stratum, "calibration")] == 1,
                     f"{family}/{stratum} split quota changed")
    return value


def load_training_view(*, root: Path = ROOT) -> dict[str, Any]:
    _require(root.resolve() == ROOT.resolve(),
             "the registered training-view repository root changed")
    _require_registered_generated_root(root)
    view = WORKFLOW.load_training_view_for_consumption(
        path=WORKFLOW.TRAINING_VIEW_PATH, materialize=True)
    return validate_training_view_structure(view)


def _safe_relative(value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} path is absent")
    path = Path(value)
    _require(not path.is_absolute() and path.parts
             and all(part not in {"", ".", ".."} for part in path.parts),
             f"{label} path is not a safe repository-relative path")
    lowered = {part.lower() for part in path.parts}
    _require("sealed" not in lowered
             and not any(part.startswith("sealed_") for part in lowered)
             and "sealed_test.json" not in lowered,
             f"{label} path enters sealed custody")
    return path


def _frame_path(root: Path, row: Mapping[str, Any],
                record: Mapping[str, Any]) -> Path:
    relative = _safe_relative(record.get("path"), label="frame")
    if relative.parts[0] == ".generated":
        return root / relative
    frame_root = _safe_relative(row.get("frame_root"), label="frame root")
    return root / frame_root / relative


def validate_frame_inputs(view: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, list[str]]:
    """Verify all bound RGB bytes once before opening the target checkpoint."""

    rows = view["rows"]
    cache: dict[Path, tuple[int, str]] = {}
    horizon_paths: dict[str, list[str]] = {}
    context_by_state: dict[str, str] = {}
    for row in rows:
        for kind, expected in (("context_frames", 3), ("horizon_frames", 4)):
            records = row.get(kind)
            _require(isinstance(records, list) and len(records) == expected,
                     f"training row lacks {expected} {kind}")
            expected_ordinals = range(0, 3) if kind == "context_frames" else range(1, 5)
            paths: list[str] = []
            for expected_ordinal, record in zip(expected_ordinals, records):
                _require(isinstance(record, Mapping), "frame record is not an object")
                observed_ordinal = record.get(
                    "slot" if kind == "context_frames" else "horizon")
                _require(observed_ordinal == expected_ordinal,
                         f"{kind} ordinal changed")
                _require(record.get("shape") == [224, 224, 3]
                         and record.get("dtype") == "uint8",
                         f"{kind} shape/dtype changed")
                path = _frame_path(root, row, record)
                _require(path.is_file() and not path.is_symlink(),
                         f"bound frame is missing or symlinked: {path}")
                observed = cache.get(path)
                if observed is None:
                    observed = (path.stat().st_size, file_sha256(path))
                    cache[path] = observed
                _require(observed == (record.get("byte_count"), record.get("sha256")),
                         f"bound frame bytes changed: {path}")
                paths.append(str(path))
            if kind == "context_frames":
                digest = canonical_digest(records)
                prior = context_by_state.setdefault(str(row["state_id"]), digest)
                _require(prior == digest,
                         "rows in one state disagree on context frames")
            else:
                key = str(row["training_view_row_digest"])
                horizon_paths[key] = paths
    _require(len(horizon_paths) == EXPECTED_ROWS,
             "horizon-frame inputs are incomplete")
    return horizon_paths


def _latent_record(path: Path, row: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    return {
        "training_view_row_digest": row["training_view_row_digest"],
        "state_id": row["state_id"],
        "state_identity_digest": row["state_identity_digest"],
        "candidate_index": int(row["candidate_index"]),
        "source_kind": row["source_kind"],
        "path": str(path.relative_to(encoded_root(root))),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
        "shape": list(HORIZON_SHAPE),
    }


def _valid_latent_record(path: Path, record: Any,
                         row: Mapping[str, Any]) -> bool:
    if (not path.is_file() or path.is_symlink()
            or not isinstance(record, Mapping)):
        return False
    expected_bytes = int(np.prod(HORIZON_SHAPE)) * np.dtype(np.float16).itemsize
    return bool(
        record.get("training_view_row_digest")
        == row.get("training_view_row_digest")
        and record.get("state_id") == row.get("state_id")
        and record.get("state_identity_digest") == row.get("state_identity_digest")
        and record.get("candidate_index") == row.get("candidate_index")
        and record.get("source_kind") == row.get("source_kind")
        and record.get("shape") == list(HORIZON_SHAPE)
        and record.get("byte_count") == expected_bytes
        and path.stat().st_size == expected_bytes
        and file_sha256(path) == record.get("sha256")
    )


def _index_payload(view: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                   *, encoder_identity: Mapping[str, Any], complete: bool) -> dict[str, Any]:
    return _signed({
        "schema": LATENT_INDEX_SCHEMA,
        "status": STATUS,
        "complete": bool(complete),
        "oracle_v1_3_digest": view["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest":
            view["scorer_fit_oracle_v1_3_contract_digest"],
        "authority_digest": view["authority_digest"],
        **{key: view[key] for key in SOURCE_DIGEST_KEYS},
        "training_view_digest": view[getattr(
            WORKFLOW, "TRAINING_VIEW_SELF_KEY", "training_view_digest")],
        "row_count": EXPECTED_ROWS,
        "fit_rows": EXPECTED_FIT_ROWS,
        "calibration_rows": EXPECTED_CALIBRATION_ROWS,
        "tokens": TOKENS,
        "token_dim": TOKEN_DIM,
        "horizons": HORIZONS,
        "horizon_shape": [len(records), *HORIZON_SHAPE],
        "target_encoder_digest": target_encoder_digest(),
        "target_encoder_checkpoint_sha256": TARGET_ENCODER["checkpoint_sha256"],
        "preprocess_contract_digest": preprocess_contract_digest(),
        "preprocessing_digest": PREPROCESSING_SHA256,
        "encoder_compute_dtype": "float32",
        "latent_storage_dtype": "float16",
        "encoder": dict(encoder_identity),
        "horizon_records": list(records),
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, LATENT_INDEX_SELF_KEY)


def validate_latent_index(index: Mapping[str, Any], view: Mapping[str, Any],
                          *, root: Path = ROOT,
                          verify_encoder_checkpoint: bool = False) -> dict[str, Any]:
    value = _validate_signed(index, LATENT_INDEX_SELF_KEY, "v1.3 latent index")
    _require(value.get("schema") == LATENT_INDEX_SCHEMA
             and value.get("status") == STATUS
             and value.get("complete") is True,
             "v1.3 latent index is not a complete terminal index")
    for key in ("oracle_v1_3_digest",
                "scorer_fit_oracle_v1_3_contract_digest", "authority_digest",
                *SOURCE_DIGEST_KEYS):
        _require(value.get(key) == view.get(key),
                 f"latent index differs from training view at {key}")
    self_key = getattr(WORKFLOW, "TRAINING_VIEW_SELF_KEY", "training_view_digest")
    _require(value.get("training_view_digest") == view.get(self_key),
             "latent index binds another training view")
    _require(value.get("row_count") == EXPECTED_ROWS
             and value.get("fit_rows") == EXPECTED_FIT_ROWS
             and value.get("calibration_rows") == EXPECTED_CALIBRATION_ROWS
             and value.get("tokens") == TOKENS
             and value.get("token_dim") == TOKEN_DIM
             and value.get("horizons") == HORIZONS
             and value.get("horizon_shape") == [EXPECTED_ROWS, *HORIZON_SHAPE]
             and value.get("target_encoder_digest") == target_encoder_digest()
             and value.get("target_encoder_checkpoint_sha256")
             == TARGET_ENCODER["checkpoint_sha256"]
             and value.get("preprocess_contract_digest")
             == preprocess_contract_digest()
             and value.get("preprocessing_digest") == PREPROCESSING_SHA256
             and value.get("encoder_compute_dtype") == "float32"
             and value.get("latent_storage_dtype") == "float16"
             and value.get("predictor_checkpoints_opened") == 0
             and value.get("predictor_utility_shards_opened") == 0
             and value.get("final_200_state_corpus_generated") is False,
             "v1.3 latent-index scientific contract changed")
    records = value.get("horizon_records")
    _require(isinstance(records, list) and len(records) == EXPECTED_ROWS,
             "v1.3 latent index has incomplete horizon records")
    by_digest = {str(row["training_view_row_digest"]): row
                 for row in view["rows"]}
    _require(len(by_digest) == EXPECTED_ROWS,
             "training-view row identities are not unique")
    seen: set[str] = set()
    for record in records:
        _require(isinstance(record, Mapping), "latent record is not an object")
        digest = str(record.get("training_view_row_digest"))
        row = by_digest.get(digest)
        _require(row is not None and digest not in seen,
                 "latent record identity is absent or duplicated")
        relative = _safe_relative(record.get("path"), label="latent")
        _require(relative == Path("latents/horizon") / f"{digest}.f16",
                 "latent shard path is not the canonical row-digest path")
        path = encoded_root(root) / relative
        _require(_valid_latent_record(path, record, row),
                 f"latent shard changed for {digest}")
        seen.add(digest)
    _require(seen == set(by_digest), "latent index omits training rows")
    identity = value.get("encoder")
    _require(isinstance(identity, Mapping)
             and identity.get("checkpoint_sha256")
             == TARGET_ENCODER["checkpoint_sha256"],
             "latent index encoder identity changed")
    if verify_encoder_checkpoint:
        from scripts import dev_frozen_dense_representation_encoders_v1 \
            as frozen_encoders

        arm = frozen_encoders.VJepa21CroppedV03Arm()
        _require(frozen_encoders.preprocessing_hash(arm) == PREPROCESSING_SHA256
                 and arm.identity() == dict(identity),
                 "live frozen target encoder identity changed")
    return value


def _receipt_payload(index: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    path = latent_index_path(root)
    return _signed({
        "schema": ENCODING_RECEIPT_SCHEMA,
        "status": STATUS,
        "complete": True,
        "latent_index_digest": index[LATENT_INDEX_SELF_KEY],
        "latent_index_path": str(path.relative_to(root)),
        "latent_index_sha256": file_sha256(path),
        "latent_index_byte_count": path.stat().st_size,
        "training_view_digest": index["training_view_digest"],
        "oracle_v1_3_digest": index["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest":
            index["scorer_fit_oracle_v1_3_contract_digest"],
        "authority_digest": index["authority_digest"],
        **{key: index[key] for key in SOURCE_DIGEST_KEYS},
        "horizon_latent_count": EXPECTED_ROWS,
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, ENCODING_RECEIPT_SELF_KEY)


def load_and_validate_encoded_training_view_for_consumption(
        *, root: Path = ROOT,
        verify_encoder_checkpoint: bool = False) -> dict[str, Any]:
    view = load_training_view(root=root)
    index_path = latent_index_path(root)
    receipt_path = encoding_receipt_path(root)
    _require(index_path.is_file() and not index_path.is_symlink(),
             "v1.3 latent index is absent")
    _require(receipt_path.is_file() and not receipt_path.is_symlink(),
             "v1.3 encoding receipt is absent")
    index = validate_latent_index(
        json.loads(index_path.read_text()), view, root=root,
        verify_encoder_checkpoint=verify_encoder_checkpoint)
    receipt = _validate_signed(
        json.loads(receipt_path.read_text()), ENCODING_RECEIPT_SELF_KEY,
        "v1.3 encoding receipt")
    _require(receipt == _receipt_payload(index, root=root),
             "v1.3 encoding receipt differs from exact index bytes")
    return {"view": view, "index": index, "receipt": receipt,
            "encoded_root": encoded_root(root)}


def encode_training_view(
        *, root: Path = ROOT, batch_frames: int = 8,
        arm_factory: Callable[[], Any] | None = None,
        encode_fn: Callable[..., np.ndarray] | None = None,
        ) -> dict[str, Any]:
    _require(batch_frames >= HORIZONS,
             "batch-frames must be at least four")
    view = load_training_view(root=root)
    horizon_paths = validate_frame_inputs(view, root=root)
    # Heavy model dependencies are intentionally imported only after all
    # rows, labels, lineage, and RGB bytes have passed validation.
    import torch
    from scripts import dev_frozen_dense_representation_encoders_v1 \
        as frozen_encoders
    from scripts import encode_go2_branch_corpus_v1_2 as frozen_encoding

    if arm_factory is None:
        arm_factory = frozen_encoders.VJepa21CroppedV03Arm
    if encode_fn is None:
        encode_fn = frozen_encoding.encode_paths
    rows = sorted(view["rows"], key=lambda row: (
        _row_role(row) != "fit", str(row["state_id"]),
        int(row["candidate_index"])))
    index_path = latent_index_path(root)
    receipt_path = encoding_receipt_path(root)
    if receipt_path.exists() or receipt_path.is_symlink():
        return load_and_validate_encoded_training_view_for_consumption(
            root=root, verify_encoder_checkpoint=False)

    prior_records: dict[str, Mapping[str, Any]] = {}
    if index_path.is_file() and not index_path.is_symlink():
        prior = _validate_signed(
            json.loads(index_path.read_text()), LATENT_INDEX_SELF_KEY,
            "partial v1.3 latent index")
        _require(prior.get("schema") == LATENT_INDEX_SCHEMA
                 and prior.get("complete") is False
                 and prior.get("training_view_digest")
                 == view[getattr(WORKFLOW, "TRAINING_VIEW_SELF_KEY",
                                  "training_view_digest")]
                 and prior.get("oracle_v1_3_digest") == view["oracle_v1_3_digest"]
                 and prior.get("scorer_fit_oracle_v1_3_contract_digest")
                 == view["scorer_fit_oracle_v1_3_contract_digest"]
                 and prior.get("authority_digest") == view["authority_digest"],
                 "partial latent index binds another run")
        prior_records = {
            str(record["training_view_row_digest"]): record
            for record in prior.get("horizon_records", [])
            if isinstance(record, Mapping)
        }

    output = encoded_root(root) / "latents" / "horizon"
    output.mkdir(parents=True, exist_ok=True)
    current: dict[str, dict[str, Any]] = {}
    missing: list[Mapping[str, Any]] = []
    for row in rows:
        digest = str(row["training_view_row_digest"])
        path = output / f"{digest}.f16"
        record = prior_records.get(digest)
        if _valid_latent_record(path, record, row):
            current[digest] = dict(record)
        else:
            _require(not path.exists() and not path.is_symlink(),
                     "unregistered or invalid v1.3 latent shard already exists")
            missing.append(row)

    arm = arm_factory()
    _require(frozen_encoders.preprocessing_hash(arm) == PREPROCESSING_SHA256,
             "frozen target preprocessing identity changed")
    encoder_identity = arm.identity()
    _require(encoder_identity.get("checkpoint_sha256")
             == TARGET_ENCODER["checkpoint_sha256"],
             "target encoder checkpoint identity changed")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    encoder = None
    if missing:
        encoder = arm.build(device, dtype)
    batch_rows = max(1, batch_frames // HORIZONS)
    for start in range(0, len(missing), batch_rows):
        batch = missing[start:start + batch_rows]
        paths = [path for row in batch
                 for path in horizon_paths[str(row["training_view_row_digest"])] ]
        arrays = encode_fn(arm, encoder, paths, device, dtype).reshape(
            len(batch), *HORIZON_SHAPE)
        for row, array in zip(batch, arrays):
            digest = str(row["training_view_row_digest"])
            path = output / f"{digest}.f16"
            frozen_encoding.atomic_f16(path, array)
            current[digest] = _latent_record(path, row, root=root)
        partial = _index_payload(
            view, [current[key] for key in sorted(current)],
            encoder_identity=encoder_identity, complete=False)
        _atomic_operational_json(index_path, partial)

    ordered = [current[str(row["training_view_row_digest"])] for row in rows]
    final_index = _index_payload(
        view, ordered, encoder_identity=encoder_identity, complete=True)
    _atomic_operational_json(index_path, final_index)
    final_index = validate_latent_index(
        final_index, view, root=root, verify_encoder_checkpoint=False)
    receipt = _receipt_payload(final_index, root=root)
    _publish_json_once(receipt_path, receipt, label="v1.3 encoding receipt")
    return {"view": view, "index": final_index, "receipt": receipt,
            "encoded_root": encoded_root(root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-frames", type=int, default=8)
    args = parser.parse_args(argv)
    result = encode_training_view(batch_frames=args.batch_frames)
    print(json.dumps({
        "status": "COMPLETE_ORACLE_V1_3_TARGET_ENCODING",
        "training_view_digest": result["index"]["training_view_digest"],
        "latent_index_digest": result["index"][LATENT_INDEX_SELF_KEY],
        "encoding_receipt_digest":
            result["receipt"][ENCODING_RECEIPT_SELF_KEY],
        "horizon_latent_count": len(result["index"]["horizon_records"]),
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
