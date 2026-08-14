#!/usr/bin/env python3
"""Train and qualify the shared utility scorer against oracle v1.2.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  **No predictor checkpoint is opened.**
Training and qualification use only the scorer-fit corpus's TRUE latent
trajectories.

The architecture, optimiser, fixed 60-epoch budget, final-epoch rule and
qualification thresholds come from the prospective scorer contract.  This
implementation is deliberately fail closed: the complete 120-state/720-row
corpus, its identity manifest and completion receipt, and both latent blobs are
digest-verified before either scorer is constructed.

Training is restartable only at a verified completed-epoch boundary.  Every
checkpoint contains the model, optimiser, global RNGs, shuffle-generator state
and the exact last-epoch row order.  If no checkpoint can be resumed exactly,
the old attempt is retained and the same run starts again from its immutable,
registered initial state.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

# Required by CUDA deterministic matrix multiplication.  It must be set before
# the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle.go2_scorer_contract_v1_2 import (  # noqa: E402
    SCORER, TARGET_ENCODER, clean_source_binding, contract, contract_digest,
    _managed_scorer_contract_output_path,
)
from lewm.oracle.go2_candidate_allocation_v1_2 import (  # noqa: E402
    CandidateAllocationError, allocation_contract_digest,
    allocation_amendment_digest, allocation_manifest_digest,
    validate_allocation_manifest, validate_pre_identity_structural_validation,
)
from lewm.oracle.go2_invalid_scorer_identity_exclusion_v1_2 import (  # noqa: E402
    invalid_identity_exclusion_digest,
)
from lewm.oracle import go2_scorer_state_selector_amendment_v2 as STATE_SELECTOR  # noqa: E402
from lewm.oracle.go2_textured_v03_renderer import (  # noqa: E402
    renderer_contract_digest as textured_v03_renderer_contract_digest,
)
from lewm.oracle.go2_branch_oracle_v1_2 import PROGRESS_NORMALISER_M  # noqa: E402
from scripts import build_go2_branch_corpus_v1_2 as CORPUS_BUILDER  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_ROOT = ROOT / ".generated/go2_branch_corpus_v1_2"
PACKAGE_DIR = ROOT / ".generated/go2_utility_scorer_v1_2"
SCORER_CONTRACT_ARTIFACT_PATH = PACKAGE_DIR / "scorer_contract_v1_2.json"

TOKENS, TOKEN_DIM, HORIZONS = 768, 1024, 4
CONTEXT_SLOTS = 3
ACTION_DIM, GOAL_DIM = 40, 3
HIDDEN_DIM = 512
TIE_TOLERANCE = 0.02
WEIGHTS = SCORER["weights"]

EXPECTED_POOL = "scorer_fit"
EXPECTED_STATES = 120
EXPECTED_BRANCHES = 720
EXPECTED_FAMILIES = 8
EXPECTED_FIT_STATES = 96
EXPECTED_CALIBRATION_STATES = 24
EXPECTED_FIT_ROWS = 576
EXPECTED_CALIBRATION_ROWS = 144
EXPECTED_CANDIDATES_PER_STATE = 6
EXPECTED_STATES_PER_FAMILY = 15
EXPECTED_STRATA = ("general", "safety_enriched", "completion_enriched")
FROZEN_CANDIDATES = (
    "straight_fast", "straight_medium", "straight_slow",
    "arc_left_sustained", "arc_right_sustained",
    "turn_left_sustained", "turn_right_sustained",
    "turn_left_then_go", "turn_right_then_go", "go_then_turn_left",
    "reverse_then_turn", "hold_all",
)

FROZEN_SCORER_FIT_ALLOCATION_DESIGN_DIGEST = (
    "a587b1de264dfb54176aa231e5183ae4b7b4229bbf65c02d62438f86af5e7116"
)
FROZEN_BRANCH_BOUNDARY_DIGEST = (
    "1faae05f843e6f02f0f354c63ab3bcad9404111140146b1355d025da3d0c7a92"
)
EXPECTED_PREPROCESS = (
    "dev_frozen_dense_representation_encoders_v1.preprocess_vjepa_v03_crop"
)
EXPECTED_TARGET_NORMALISATION = "F.layer_norm over the token dimension"
FROZEN_PREPROCESSING_DIGEST = (
    "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
LAUNCH_BINDING_KEYS = (
    "clean_source_launch_receipt_digest",
    "source_repository_commit",
    "clean_source_binding_digest",
    "bound_implementations_digest",
    "scorer_contract_artifact_digest",
    "mixed_precontract_disposition_receipt_digest",
)
# The first five fields are the source/contract launch identity carried by the
# original (d9d) scientific manifest.  The mixed-precontract receipt is
# selector lineage and is intentionally handled alongside the selector
# feasibility receipt below.
SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS = LAUNCH_BINDING_KEYS[:-1]
SELECTOR_BINDING_KEYS = tuple(STATE_SELECTOR.ACTIVE_SELECTOR_BINDING_KEYS)
SCORER_PROVENANCE_BINDING_KEYS = SELECTOR_BINDING_KEYS + LAUNCH_BINDING_KEYS
GLOBAL_EXACT_PROVENANCE_BINDING_KEYS = (
    "clean_source_launch_receipt_sha256",
    "scorer_contract_artifact_sha256",
    "global_exact_execution_amendment_digest",
    "global_exact_successor_scorer_contract_digest",
    "current_scorer_contract_v1_2_digest",
    "scientific_predecessor_launch_bindings",
    "global_exact_scorer_contract_lineage",
)
GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_SCHEMA = (
    "go2_utility_scorer_v1_2_global_exact_contract_lineage_v1"
)
GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_KEYS = frozenset((
    "schema",
    "scientific_predecessor_scorer_contract_v1_2_digest",
    "current_scorer_contract_v1_2_digest",
    "global_exact_successor_scorer_contract_digest",
))

# There are no learned or outcome-derived scaler values.  Recording every
# identity transform explicitly prevents a downstream consumer from silently
# introducing model-specific calibration.
NORMALISATION = {
    "latent_tokens": EXPECTED_TARGET_NORMALISATION,
    "spatial_aggregation": "arithmetic mean over all 768 tokens independently at each horizon",
    "progress_target": "identity (oracle-v1.2 continuous metric-geodesic progress)",
    "safety_target": "identity in [0,1] (logistic head trained by soft-label BCE)",
    "completion_target": "identity in {0,1} (logistic head trained by BCE)",
    "candidate_action": "identity, 4 blocks x 10 post-slew values",
    "goal_bearing": "[sin(bearing_body_rad), cos(bearing_body_rad)]",
    "goal_range_m": "identity",
    "model_specific_calibration": None,
}


class CorpusValidationError(RuntimeError):
    """The registered scorer-fit corpus is incomplete or incorrectly bound."""


# ---------------------------------------------------------------- utilities --
def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def sorted_json_digest(value: Any) -> str:
    """Digest convention used by the pre-existing manifest generators."""

    return canonical_digest(value)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Stable digest independent of torch.save container metadata."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def structured_digest(value: Any) -> str:
    """Stable digest for nested optimiser/RNG state containing tensors/arrays."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"torch:")
            digest.update(tensor_digest(item).encode("ascii"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy:")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, tuple):
            digest.update(b"tuple[")
            for member in item:
                update(member)
            digest.update(b"]")
        elif isinstance(item, list):
            digest.update(b"list[")
            for member in item:
                update(member)
            digest.update(b"]")
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b":")
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.partial"
    )
    with temporary.open("wb") as sink:
        torch.save(payload, sink)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.partial"
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode()
    with temporary.open("wb") as sink:
        sink.write(encoded)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusValidationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusValidationError(f"{path} is not a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusValidationError(message)


def _require_digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
             f"{label} is not a lowercase SHA-256 digest")
    return str(value)


def scorer_provenance_binding_keys(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the provenance fields applicable to one scorer run.

    Legacy d9d manifests retain their byte-for-byte field set.  A global-exact
    successor adds its operational contract and historical-science bridge only
    when the successor digest is present.
    """

    if "global_exact_successor_scorer_contract_digest" not in value:
        return SCORER_PROVENANCE_BINDING_KEYS
    return SCORER_PROVENANCE_BINDING_KEYS + GLOBAL_EXACT_PROVENANCE_BINDING_KEYS


def validate_global_exact_scorer_contract_lineage(
        value: Any, *, expected: Mapping[str, Any] | None = None,
        ) -> dict[str, str]:
    """Validate the closed historical/current scorer-contract bridge."""

    _require(isinstance(value, Mapping),
             "global exact scorer-contract lineage is not an object")
    _require(set(value) == GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_KEYS,
             "global exact scorer-contract lineage schema is not closed")
    lineage = dict(value)
    _require(
        lineage.get("schema") == GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_SCHEMA,
        "global exact scorer-contract lineage schema changed",
    )
    for key in GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_KEYS - {"schema"}:
        _require_digest(lineage.get(key), f"global exact lineage {key}")
    _require(
        lineage["current_scorer_contract_v1_2_digest"] == contract_digest(),
        "global exact operational scorer contract differs from current source",
    )
    if expected is not None:
        _require(dict(expected) == lineage,
                 "global exact scorer-contract lineage differs")
    return lineage


def scorer_provenance_bindings(value: Mapping[str, Any]) -> dict[str, Any]:
    bindings = {
        key: value[key] for key in scorer_provenance_binding_keys(value)
    }
    if "global_exact_successor_scorer_contract_digest" in value:
        lineage = validate_global_exact_scorer_contract_lineage(
            bindings["global_exact_scorer_contract_lineage"])
        _require(
            lineage["current_scorer_contract_v1_2_digest"]
            == bindings["current_scorer_contract_v1_2_digest"],
            "global exact operational scorer digest is internally inconsistent",
        )
        _require(
            lineage["global_exact_successor_scorer_contract_digest"]
            == bindings["global_exact_successor_scorer_contract_digest"],
            "global exact successor scorer digest is internally inconsistent",
        )
    return bindings


def operational_scorer_contract_digest(value: Mapping[str, Any]) -> str:
    """Return the exact signed operational digest for a scorer artefact."""

    if "global_exact_successor_scorer_contract_digest" not in value:
        return contract_digest()
    lineage = validate_global_exact_scorer_contract_lineage(
        value.get("global_exact_scorer_contract_lineage"))
    _require(
        value.get("current_scorer_contract_v1_2_digest")
        == lineage["current_scorer_contract_v1_2_digest"],
        "global exact current scorer digest differs from its closed lineage",
    )
    _require(
        value.get("global_exact_successor_scorer_contract_digest")
        == lineage["global_exact_successor_scorer_contract_digest"],
        "global exact successor scorer digest differs from its closed lineage",
    )
    return lineage["current_scorer_contract_v1_2_digest"]


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _receipt_value(receipt: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    containers = [receipt]
    for key in ("counts", "bindings", "bound_digests"):
        nested = receipt.get(key)
        if isinstance(nested, Mapping):
            containers.append(nested)
    for container in containers:
        for alias in aliases:
            if alias in container:
                return container[alias]
    return None


def _bound_value(value: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    """Find one named binding at top level or in a conventional binding map."""

    containers: list[Mapping[str, Any]] = [value]
    for key in ("bindings", "bound_digests", "corpus_bound_digests", "digests"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            containers.append(nested)
    for container in containers:
        for alias in aliases:
            if alias in container:
                return container[alias]
    return None


def _validate_clean_source_launch(
        pool_dir: Path, pre_identity_validation: Mapping[str, Any], *,
        enforce_managed_paths: bool = False,
        ) -> dict[str, Any]:
    """Bind training to the same clean committed source that generated rows.

    The branch builder issues both artefacts before it is permitted to select a
    state identity.  Recomputing the clean-source binding here prevents a later
    dirty or differently committed trainer from accepting otherwise internally
    consistent corpus receipts.
    """

    launch_path = pool_dir / "clean_source_launch_receipt.json"
    artifact_path = SCORER_CONTRACT_ARTIFACT_PATH
    if enforce_managed_paths:
        launch_path = (
            CORPUS_BUILDER.pin_active_scorer_fit_artifact_for_consumption(
                launch_path, "clean_source_launch_receipt.json"))
        artifact_path = _managed_scorer_contract_output_path(artifact_path)
    _require(launch_path.is_file(),
             "missing required clean-source launch receipt")
    _require(artifact_path.is_file(),
             "missing required issued scorer-contract artifact")
    launch = _read_json(launch_path)
    artifact = _read_json(artifact_path)
    launch_digest = _require_digest(
        launch.get("clean_source_launch_receipt_digest"),
        "clean_source_launch_receipt_digest")
    artifact_digest = _require_digest(
        artifact.get("contract_artifact_digest"),
        "contract_artifact_digest")
    _require(canonical_digest({
        key: value for key, value in launch.items()
        if key != "clean_source_launch_receipt_digest"
    }) == launch_digest, "clean-source launch receipt self digest does not verify")
    _require(canonical_digest({
        key: value for key, value in artifact.items()
        if key != "contract_artifact_digest"
    }) == artifact_digest, "issued scorer-contract artifact self digest does not verify")
    _require(launch.get("schema")
             == "go2_utility_scorer_v1_2_clean_source_launch_receipt"
             and launch.get("complete") is True
             and launch.get("source_repository_clean") is True,
             "clean-source launch receipt is incomplete")
    _require(artifact.get("schema")
             == "go2_utility_scorer_contract_v1_2_artifact"
             and artifact.get("complete") is True
             and artifact.get("source_repository_clean") is True
             and artifact.get("state_selector_amendment_verified") is True
             and artifact.get("state_selector_feasibility_verified") is True
             and artifact.get(
                 "preserved_state_mixed_precontract_disposition_verified") is True
             and artifact.get("scorer_contract_v1_2_digest") == contract_digest()
             and artifact.get("contract") == contract(),
             "issued scorer-contract artifact is incomplete or differently bound")
    pending_phase_2 = artifact.get(
        "mixed_state_post_allocation_revalidation")
    _require(isinstance(pending_phase_2, Mapping)
             and pending_phase_2.get("status")
             == "PENDING_POST_IDENTITY_PRE_OUTCOME"
             and pending_phase_2.get("required_before_active_identity_manifest")
             is True
             and pending_phase_2.get("schema")
             == STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_SCHEMA
             and pending_phase_2.get("path")
             == STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH
             and pending_phase_2.get(
                 "realized_receipt_digest_bound_at_contract_issue") is False,
             "scorer contract artifact does not keep phase-2 revalidation pending")
    try:
        current = clean_source_binding()
    except RuntimeError as exc:
        raise CorpusValidationError(
            f"cannot validate clean committed scorer source: {exc}") from exc
    current_digest = canonical_digest(current)
    _require(artifact.get("clean_source_binding") == current
             and artifact.get("clean_source_binding_digest") == current_digest,
             "issued scorer contract differs from the current clean HEAD")
    expected = {
        "clean_source_launch_receipt_digest": launch_digest,
        "source_repository_commit": current["source_repository_commit"],
        "clean_source_binding_digest": current_digest,
        "bound_implementations_digest": current["bound_implementations_digest"],
        "scorer_contract_artifact_digest": artifact_digest,
    }
    for key, value in expected.items():
        if key != "clean_source_launch_receipt_digest":
            _require(launch.get(key) == value,
                     f"clean-source launch receipt {key} does not verify")
    _require(launch.get("scorer_contract_v1_2_digest") == contract_digest(),
             "clean-source launch receipt scorer contract differs")
    _require(launch.get("scorer_contract_artifact_sha256")
             == sha256_file(artifact_path),
             "clean-source launch receipt contract-artifact bytes differ")
    _require(launch.get("candidate_allocation_amendment_digest")
             == allocation_amendment_digest(),
             "clean-source launch receipt allocation amendment differs")
    _require(launch.get("invalid_scorer_identity_exclusion_digest")
             == invalid_identity_exclusion_digest(),
             "clean-source launch receipt invalid-identity exclusion differs")
    _require(launch.get("state_selector_amendment_digest")
             == STATE_SELECTOR.state_selector_amendment_digest(),
             "clean-source launch receipt selector amendment differs")
    feasibility_digest = _require_digest(
        launch.get("state_selector_feasibility_receipt_digest"),
        "clean-source launch state_selector_feasibility_receipt_digest")
    _require(artifact.get("state_selector_feasibility_receipt_digest")
             == feasibility_digest,
             "clean-source launch selector feasibility differs from contract artifact")
    disposition_digest = _require_digest(
        artifact.get("mixed_precontract_disposition_receipt_digest"),
        "mixed_precontract_disposition_receipt_digest")
    _require(
        launch.get("mixed_precontract_disposition_receipt_digest")
        == disposition_digest,
        "clean-source launch mixed disposition differs from contract artifact",
    )
    _require(launch.get("pre_identity_allocation_validation_digest")
             == pre_identity_validation.get("pre_identity_validation_digest"),
             "clean-source launch receipt pre-identity validation differs")
    return {
        **expected,
        "clean_source_launch_receipt_sha256": sha256_file(launch_path),
        "scorer_contract_artifact_sha256": sha256_file(artifact_path),
        "launch_state_selector_feasibility_receipt_digest": feasibility_digest,
        "mixed_precontract_disposition_receipt_digest": disposition_digest,
    }


def _load_manifest_launch_lineage(
        manifest: Mapping[str, Any], pool_dir: Path,
        pre_identity_validation: Mapping[str, Any], *,
        enforce_managed_paths: bool = False,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve operational, manifest-scientific, and selector launch bindings.

    A legacy manifest has one launch identity, so all three views are the
    existing clean-source launch receipt.  A global-exact manifest deliberately
    keeps the original d9d launch fields as scientific lineage while the
    post-manifest successor scorer contract binds the current clean executable
    source.  The builder helper validates that bridge before returning it.
    """

    if "small_completion_global_exact_execution" not in manifest:
        legacy = _validate_clean_source_launch(
            pool_dir, pre_identity_validation,
            enforce_managed_paths=enforce_managed_paths)
        scientific = dict(legacy)
        return legacy, scientific, legacy

    try:
        successor = (
            CORPUS_BUILDER
            .load_global_exact_successor_scorer_contract_for_consumption(
                manifest))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise CorpusValidationError(
            f"global-exact successor scorer contract does not verify: {exc}"
        ) from exc
    _require(isinstance(successor, Mapping),
             "global-exact successor scorer contract is not an object")
    predecessor = successor.get("scientific_predecessor_launch_bindings")
    _require(isinstance(predecessor, Mapping),
             "global-exact successor lacks scientific predecessor launch bindings")
    _require(set(predecessor) == set(SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS),
             "global-exact scientific predecessor launch schema is not closed")
    for key in SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS:
        value = predecessor.get(key)
        if key == "source_repository_commit":
            _require(isinstance(value, str)
                     and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
                     f"scientific predecessor {key} is not a commit digest")
        else:
            _require_digest(value, f"scientific predecessor {key}")
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
        if key == "source_repository_commit":
            _require(isinstance(value, str)
                     and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
                     f"operational successor {key} is not a commit digest")
        else:
            _require_digest(value, f"operational successor {key}")

    historical_scorer_digest = _require_digest(
        manifest.get("scorer_contract_v1_2_digest"),
        "global-exact scientific predecessor scorer_contract_v1_2_digest",
    )
    contract_lineage = validate_global_exact_scorer_contract_lineage({
        "schema": GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_SCHEMA,
        "scientific_predecessor_scorer_contract_v1_2_digest":
            historical_scorer_digest,
        "current_scorer_contract_v1_2_digest": successor[
            "current_scorer_contract_v1_2_digest"],
        "global_exact_successor_scorer_contract_digest": successor[
            "global_exact_successor_scorer_contract_digest"],
    })
    operational = {
        **{key: successor[key] for key in LAUNCH_BINDING_KEYS},
        **{key: successor[key] for key in GLOBAL_EXACT_PROVENANCE_BINDING_KEYS
           if key not in {
               "scientific_predecessor_launch_bindings",
               "global_exact_scorer_contract_lineage",
           }},
        "scientific_predecessor_launch_bindings": {
            key: predecessor[key]
            for key in SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS
        },
        "global_exact_scorer_contract_lineage": contract_lineage,
    }
    scientific = {
        **operational["scientific_predecessor_launch_bindings"],
        "mixed_precontract_disposition_receipt_digest": successor[
            "mixed_precontract_disposition_receipt_digest"],
    }
    for key, expected in scientific.items():
        _require(manifest.get(key) == expected,
                 f"global-exact manifest scientific launch differs at {key}")
    _require(
        manifest.get("pre_identity_allocation_validation_digest")
        == pre_identity_validation.get("pre_identity_validation_digest"),
        "global-exact manifest pre-identity validation binding differs",
    )
    selector_launch = {
        **scientific,
        "launch_state_selector_feasibility_receipt_digest": successor[
            "launch_state_selector_feasibility_receipt_digest"],
    }
    return operational, scientific, selector_launch


def _validate_selector_successor(
        pool_dir: Path, launch_bindings: Mapping[str, Any],
        allocation_manifest: Mapping[str, Any],
        active_states: Sequence[Mapping[str, Any]],
        *, enforce_managed_paths: bool = False,
        global_exact_manifest: Mapping[str, Any] | None = None,
        ) -> dict[str, str]:
    """Validate frozen feasibility, mixed disposition, and final phase 2."""

    selection_digest = contract()["corpus_selection_digest"]
    feasibility_path = (
        pool_dir / STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME)
    disposition_path = (
        pool_dir
        / STATE_SELECTOR.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_NAME)
    revalidation_path = (
        pool_dir / STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_NAME)
    if enforce_managed_paths:
        revalidation_path = (
            CORPUS_BUILDER.pin_active_scorer_fit_artifact_for_consumption(
                revalidation_path,
                STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_NAME))
    _require(revalidation_path.is_file(),
             "missing preserved-state revalidation receipt")
    try:
        STATE_SELECTOR.validate_authority_artifacts()
        if enforce_managed_paths:
            feasibility = (
                STATE_SELECTOR.validate_frozen_reachability_feasibility_pass(
                    root=ROOT))
        else:
            _require(feasibility_path.is_file(),
                     "missing all-family selector-feasibility receipt")
            feasibility = _read_json(feasibility_path)
            _require(
                feasibility ==
                STATE_SELECTOR.validate_frozen_reachability_feasibility_pass(
                    root=ROOT),
                "selector feasibility differs from frozen pass",
            )
        feasibility_digest = _require_digest(
            feasibility.get("state_selector_feasibility_receipt_digest"),
            "state_selector_feasibility_receipt_digest")
        _require(feasibility_digest == launch_bindings[
                    "launch_state_selector_feasibility_receipt_digest"],
                 "selector feasibility receipt differs from clean-source launch")
        if enforce_managed_paths:
            disposition = (
                STATE_SELECTOR
                .load_and_validate_preserved_state_mixed_precontract_disposition_receipt(
                    expected_source_commit=
                        launch_bindings["source_repository_commit"],
                    expected_successor_selection_digest=selection_digest,
                    expected_clean_source_binding_digest=str(
                        launch_bindings["clean_source_binding_digest"]),
                    expected_bound_implementations_digest=str(
                        launch_bindings["bound_implementations_digest"]),
                    root=ROOT,
                ))
        else:
            _require(disposition_path.is_file(),
                     "missing preserved-state mixed precontract disposition")
            disposition = _read_json(disposition_path)
            STATE_SELECTOR.validate_preserved_state_mixed_precontract_disposition_receipt(
                disposition,
                expected_source_commit=launch_bindings["source_repository_commit"],
                expected_successor_selection_digest=selection_digest,
                expected_clean_source_binding_digest=str(
                    launch_bindings["clean_source_binding_digest"]),
                expected_bound_implementations_digest=str(
                    launch_bindings["bound_implementations_digest"]),
                root=ROOT,
            )
        _require(
            disposition.get("mixed_precontract_disposition_receipt_digest")
            == launch_bindings[
                "mixed_precontract_disposition_receipt_digest"],
            "mixed precontract disposition differs from clean-source launch",
        )
        revalidation = _read_json(revalidation_path)
        if global_exact_manifest is not None:
            certified = (
                CORPUS_BUILDER
                .validate_global_exact_allocation_for_consumption(
                    global_exact_manifest, allocation_manifest))
            _require(
                certified["preserved_state_revalidation_receipt_digest"]
                == revalidation.get(
                    "preserved_state_revalidation_receipt_digest"),
                "global exact phase-2 selector receipt changed")
        else:
            STATE_SELECTOR.validate_preserved_state_revalidation_receipt(
                revalidation,
                allocation_manifest=allocation_manifest,
                active_states=active_states,
                expected_source_commit=launch_bindings[
                    "source_repository_commit"],
                expected_successor_selection_digest=selection_digest,
                expected_feasibility_receipt_digest=feasibility_digest,
                expected_mixed_precontract_disposition_receipt_digest=
                    launch_bindings[
                        "mixed_precontract_disposition_receipt_digest"])
        revalidation_digest = _require_digest(
            revalidation.get("preserved_state_revalidation_receipt_digest"),
            "preserved_state_revalidation_receipt_digest")
    except (OSError, ValueError, KeyError,
            STATE_SELECTOR.StateSelectorAmendmentError) as exc:
        raise CorpusValidationError(
            f"scorer-fit selector successor does not verify: {exc}") from exc
    return {
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest": feasibility_digest,
        "preserved_state_revalidation_receipt_digest": revalidation_digest,
    }


def _resolve_pool_artifact(raw: Any, pool_dir: Path) -> Path | None:
    if not isinstance(raw, str):
        return None
    supplied = Path(raw).expanduser()
    candidates = ([supplied] if supplied.is_absolute()
                  else [pool_dir / supplied, ROOT / supplied])
    try:
        pool_root = pool_dir.resolve(strict=True)
    except OSError:
        return None
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved == pool_root or pool_root in resolved.parents:
            return resolved
    return None


class HorizonShardStore:
    """Lazy, index-addressable collection of one float16 H=1..4 shard per row."""

    def __init__(self, records: Sequence[Mapping[str, Any]], pool_dir: Path) -> None:
        self.records = list(records)
        self.pool_dir = pool_dir
        self.shape = (len(self.records), HORIZONS, TOKENS, TOKEN_DIM)

    def __getitem__(self, item):
        if isinstance(item, slice):
            positions = list(range(*item.indices(len(self.records))))
        elif np.isscalar(item):
            positions = [int(item)]
        else:
            positions = [int(value) for value in np.asarray(item).reshape(-1)]
        arrays = []
        for position in positions:
            record = self.records[position]
            path = _resolve_pool_artifact(record.get("path"), self.pool_dir)
            if path is None:
                raise CorpusValidationError(
                    f"latent shard disappeared after validation: {record.get('path')}")
            arrays.append(np.memmap(path, mode="r", dtype=np.float16,
                                    shape=(HORIZONS, TOKENS, TOKEN_DIM)))
        result = np.stack(arrays, axis=0)
        if np.isscalar(item):
            return result[0]
        return result


# ----------------------------------------------------------------- the model --
class UtilityScorer(nn.Module):
    """Per-horizon shared trunk, attention pool over h, three separate heads."""

    def __init__(self, *, use_latent: bool, hidden: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.use_latent = use_latent
        if use_latent:
            self.per_horizon = nn.Sequential(
                nn.Linear(TOKEN_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.attention = nn.Linear(hidden, 1)
        self.context = nn.Sequential(
            nn.Linear(ACTION_DIM + GOAL_DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        fuse_in = hidden * (2 if use_latent else 1)
        self.fuse = nn.Sequential(nn.Linear(fuse_in, hidden), nn.SiLU())
        self.progress = nn.Linear(hidden, 1)
        self.safety = nn.Linear(hidden, 1)
        self.completion = nn.Linear(hidden, 1)

    def forward(self, latent, action_goal):
        parts = [self.context(action_goal)]
        if self.use_latent:
            per_h = self.per_horizon(latent)
            attention = torch.softmax(self.attention(per_h), dim=1)
            parts.insert(0, (per_h * attention).sum(dim=1))
        fused = self.fuse(torch.cat(parts, dim=-1))
        return (self.progress(fused).squeeze(-1),
                self.safety(fused).squeeze(-1),
                self.completion(fused).squeeze(-1))


def composite(progress, safety_logit, completion_logit):
    return (WEIGHTS["progress"] * progress
            + WEIGHTS["safety"] * torch.sigmoid(safety_logit)
            + WEIGHTS["completion"] * torch.sigmoid(completion_logit))


# ------------------------------------------------------------------- metrics --
def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denominator) if denominator > 0 else float("nan")


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels > 0.5
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores) + 1.0
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def expected_calibration_error(target: np.ndarray, predicted: np.ndarray,
                               bins: int = 10) -> float:
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, error = 0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (predicted >= lo) & (predicted < hi if hi < 1.0 else predicted <= hi)
        count = int(mask.sum())
        if count:
            error += count * abs(float(predicted[mask].mean())
                                 - float(target[mask].mean()))
            total += count
    return float(error / total) if total else float("nan")


def _state_groups(states: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        groups.setdefault(state, []).append(index)
    return groups


def pairwise_ordering(states: list[str], true_u: np.ndarray, pred_u: np.ndarray,
                      tolerance: float = TIE_TOLERANCE) -> tuple[float, int]:
    correct = considered = 0
    for indices in _state_groups(states).values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                gap = float(true_u[a] - true_u[b])
                if abs(gap) <= tolerance:
                    continue
                considered += 1
                if float(pred_u[a] - pred_u[b]) * gap > 0:
                    correct += 1
    return (correct / considered if considered else float("nan")), considered


def normalised_rank_regret(states: list[str], true_u: np.ndarray,
                           pred_u: np.ndarray) -> tuple[float, list[float]]:
    values: list[float] = []
    for indices in _state_groups(states).values():
        if len(indices) < 2:
            continue
        truth = true_u[indices]
        chosen = indices[int(np.argmax(pred_u[indices]))]
        spread = float(truth.max() - truth.min())
        values.append(0.0 if spread <= 0 else
                      float((truth.max() - true_u[chosen]) / spread))
    return (float(np.mean(values)) if values else float("nan")), values


def composite_metrics(rows: list[dict[str, Any]], true_u: np.ndarray,
                      predicted_u: np.ndarray) -> dict[str, Any]:
    states = [str(row["state_id"]) for row in rows]
    groups = _state_groups(states)
    accuracy, pairs = pairwise_ordering(states, true_u, predicted_u)
    regret, per_state_regret = normalised_rank_regret(states, true_u, predicted_u)
    absolute_regret: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    top_ties: list[float] = []
    pair_ties = pair_total = 0
    spreads: list[float] = []
    within_state_rank: list[float] = []
    selected_utilities: list[float] = []
    best_utilities: list[float] = []
    per_state: list[dict[str, Any]] = []
    for state_id, indices in groups.items():
        truth = true_u[indices]
        scores = predicted_u[indices]
        predicted_order = np.argsort(-scores, kind="mergesort")
        chosen_local = int(predicted_order[0])
        best_mask = truth == truth.max()
        absolute_regret.append(float(truth.max() - truth[chosen_local]))
        selected_utilities.append(float(truth[chosen_local]))
        best_utilities.append(float(truth.max()))
        top1.append(float(best_mask[chosen_local]))
        top3.append(float(np.any(best_mask[predicted_order[:3]])))
        spreads.append(float(scores.max() - scores.min()))
        top_ties.append(float(np.sum(np.abs(scores - scores.max()) <= TIE_TOLERANCE) > 1))
        within_state_rank.append(spearman(truth, scores))
        state_correct = state_considered = state_score_ties = state_pairs = 0
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                pair_total += 1
                pair_ties += int(abs(float(scores[i] - scores[j])) <= TIE_TOLERANCE)
                state_pairs += 1
                state_score_ties += int(
                    abs(float(scores[i] - scores[j])) <= TIE_TOLERANCE)
                true_gap = float(truth[i] - truth[j])
                if abs(true_gap) > TIE_TOLERANCE:
                    state_considered += 1
                    state_correct += int(float(scores[i] - scores[j]) * true_gap > 0)
        spread = float(truth.max() - truth.min())
        per_state.append({
            "state_id": state_id,
            "normalised_rank_regret": (0.0 if spread <= 0 else
                                        float((truth.max() - truth[chosen_local]) / spread)),
            "absolute_rank_regret": float(truth.max() - truth[chosen_local]),
            "realised_selected_utility": float(truth[chosen_local]),
            "oracle_best_utility": float(truth.max()),
            "pairwise_ordering_accuracy": (
                state_correct / state_considered if state_considered else float("nan")),
            "pairs_considered": state_considered,
            "spearman_within_state": within_state_rank[-1],
            "top1_recovery": top1[-1], "top3_recovery": top3[-1],
            "candidate_score_spread": spreads[-1],
            "top_score_tie": top_ties[-1],
            "all_pair_tie_rate": (
                state_score_ties / state_pairs if state_pairs else float("nan")),
        })
    rank_array = np.asarray(within_state_rank, dtype=np.float64)
    finite_rank = rank_array[np.isfinite(rank_array)]
    return {
        "pairwise_ordering_accuracy": accuracy,
        "pairs_considered": pairs,
        "normalised_rank_regret": regret,
        "normalised_rank_regret_median": (float(np.median(per_state_regret))
                                           if per_state_regret else float("nan")),
        "absolute_rank_regret": (float(np.mean(absolute_regret))
                                  if absolute_regret else float("nan")),
        "realised_selected_utility": (float(np.mean(selected_utilities))
                                       if selected_utilities else float("nan")),
        "oracle_best_utility": (float(np.mean(best_utilities))
                                if best_utilities else float("nan")),
        "top1_recovery": float(np.mean(top1)) if top1 else float("nan"),
        "top3_recovery": float(np.mean(top3)) if top3 else float("nan"),
        "spearman_within_state": (float(finite_rank.mean()) if len(finite_rank)
                                   else float("nan")),
        "tie_rate": float(np.mean(top_ties)) if top_ties else float("nan"),
        "tie_tolerance": TIE_TOLERANCE,
        "all_pair_tie_rate": pair_ties / pair_total if pair_total else float("nan"),
        "candidate_score_spread": {
            "mean": float(np.mean(spreads)) if spreads else float("nan"),
            "median": float(np.median(spreads)) if spreads else float("nan"),
            "min": float(np.min(spreads)) if spreads else float("nan"),
            "max": float(np.max(spreads)) if spreads else float("nan"),
        },
        "states": len(groups),
        "per_state": per_state,
    }


def _component_ranking(rows: Sequence[Mapping[str, Any]], target: np.ndarray,
                       predicted: np.ndarray) -> dict[str, Any]:
    """Descriptive, pre-outcome within-state component-head diagnostics.

    Component ties are omitted using exact float equality.  The frozen 0.02
    tolerance remains specific to the composite utility contract and is not
    silently imposed on progress, graded safety, or binary completion.
    """

    groups = _state_groups([str(row["state_id"]) for row in rows])
    correlations: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    correct = considered = 0
    eligible_top_states = 0
    for indices in groups.values():
        truth = np.asarray(target[indices], dtype=np.float64)
        scores = np.asarray(predicted[indices], dtype=np.float64)
        correlation = spearman(truth, scores)
        if math.isfinite(correlation):
            correlations.append(correlation)
        # Recovery is defined only where the target has a strict range.  This
        # keeps all-negative completion states from being called "recovered".
        if float(truth.max()) > float(truth.min()):
            eligible_top_states += 1
            order = np.argsort(-scores, kind="mergesort")
            best = truth == truth.max()
            top1.append(float(best[int(order[0])]))
            top3.append(float(np.any(best[order[:3]])))
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                gap = float(truth[left] - truth[right])
                if gap == 0.0:
                    continue
                considered += 1
                correct += int(float(scores[left] - scores[right]) * gap > 0)
    return {
        "mean_within_state_spearman": (
            float(np.mean(correlations)) if correlations else float("nan")),
        "states_with_defined_spearman": len(correlations),
        "within_state_pairwise_ordering_accuracy": (
            correct / considered if considered else float("nan")),
        "within_state_pairs_considered": considered,
        "highest_target_top1_recovery": (
            float(np.mean(top1)) if top1 else float("nan")),
        "highest_target_top3_recovery": (
            float(np.mean(top3)) if top3 else float("nan")),
        "states_eligible_for_top_recovery": eligible_top_states,
    }


def _regression_calibration(target: np.ndarray,
                            predicted: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    bias = float(np.mean(predicted - target))
    variance = float(np.sum((predicted - predicted.mean()) ** 2))
    slope = (float(np.sum((predicted - predicted.mean())
                          * (target - target.mean())) / variance)
             if variance > 0 else float("nan"))
    intercept = (float(target.mean() - slope * predicted.mean())
                 if math.isfinite(slope) else float("nan"))
    return {
        "target_mean": float(target.mean()),
        "prediction_mean": float(predicted.mean()),
        "mean_error_prediction_minus_target": bias,
        "calibration_intercept_target_on_prediction": intercept,
        "calibration_slope_target_on_prediction": slope,
    }


def _evaluate_arrays(rows: list[dict[str, Any]], true: Mapping[str, np.ndarray],
                     predicted: Mapping[str, np.ndarray]) -> dict[str, Any]:
    true_u = np.asarray([row["utility"] for row in rows], dtype=np.float64)
    p_true, p_pred = true["progress"], predicted["progress"]
    s_true, s_pred = true["safety"], predicted["safety"]
    c_true, c_pred = true["completion"], predicted["completion"]
    return {
        "rows": len(rows),
        "progress": {
            "spearman": spearman(p_true, p_pred),
            "mae": float(np.mean(np.abs(p_true - p_pred))),
            "rmse": float(np.sqrt(np.mean((p_true - p_pred) ** 2))),
            "calibration": _regression_calibration(p_true, p_pred),
            "within_state_ranking": _component_ranking(rows, p_true, p_pred),
        },
        "safety": {
            "auc_any_hazard": roc_auc((s_true > 0).astype(float), s_pred),
            "calibration_error": expected_calibration_error(s_true, s_pred),
            "mae": float(np.mean(np.abs(s_true - s_pred))),
            "rmse": float(np.sqrt(np.mean((s_true - s_pred) ** 2))),
            "any_hazard_prevalence": float(np.mean(s_true > 0)),
            "within_state_ranking": _component_ranking(rows, s_true, s_pred),
        },
        "completion": {
            "prevalence": float(np.mean(c_true)),
            "auc": roc_auc(c_true, c_pred),
            "calibration_error": expected_calibration_error(c_true, c_pred),
            "mae": float(np.mean(np.abs(c_true - c_pred))),
            "brier": float(np.mean((c_true - c_pred) ** 2)),
            "within_state_ranking": _component_ranking(rows, c_true, c_pred),
        },
        "composite": composite_metrics(rows, true_u, predicted["utility"]),
    }


def evaluate_model(model: nn.Module, latent: torch.Tensor,
                   action_goal: torch.Tensor, rows: list[dict[str, Any]],
                   targets: Mapping[str, torch.Tensor]
                   ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    with torch.no_grad():
        progress_logit, safety_logit, completion_logit = model(latent, action_goal)
        utility = composite(progress_logit, safety_logit, completion_logit)
    predicted = {
        "progress": progress_logit.detach().cpu().numpy().astype(np.float64),
        "safety": torch.sigmoid(safety_logit).detach().cpu().numpy().astype(np.float64),
        "completion": torch.sigmoid(completion_logit).detach().cpu().numpy().astype(np.float64),
        "utility": utility.detach().cpu().numpy().astype(np.float64),
    }
    true = {key: value.detach().cpu().numpy().astype(np.float64)
            for key, value in targets.items()}
    return _evaluate_arrays(rows, true, predicted), predicted


def evaluate(model: nn.Module, latent: torch.Tensor, action_goal: torch.Tensor,
             rows: list[dict[str, Any]], targets: Mapping[str, torch.Tensor]
             ) -> dict[str, Any]:
    """Compatibility wrapper used by focused tests and downstream analysis."""

    return evaluate_model(model, latent, action_goal, rows, targets)[0]


def label_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}

    def summary(key: str) -> dict[str, Any]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return {
            "min": float(values.min()),
            "quartile_1": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "quartile_3": float(np.quantile(values, 0.75)),
            "max": float(values.max()), "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "distinct": int(len(np.unique(np.round(values, 6)))),
        }

    completion = np.asarray([row["completion"] for row in rows], dtype=np.float64)
    return {
        "rows": len(rows),
        "states": len({row["state_id"] for row in rows}),
        "progress": summary("progress"),
        "safety": summary("safety"),
        "completion": summary("completion"),
        "utility": summary("utility"),
        "completion_positive": int(completion.sum()),
        "completion_negative": int(len(completion) - completion.sum()),
        "completion_prevalence": float(completion.mean()),
        "any_hazard_prevalence": float(np.mean([row["safety"] > 0 for row in rows])),
    }


def grouped_label_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Frozen descriptive label summaries overall, by family and by stratum."""

    return {
        "overall": label_distribution(rows),
        "by_family": {
            family: label_distribution([
                row for row in rows if row["family"] == family])
            for family in sorted({str(row["family"]) for row in rows})
        },
        "by_stratum": {
            stratum: label_distribution([
                row for row in rows if row["stratum"] == stratum])
            for stratum in EXPECTED_STRATA
        },
    }


def completion_by_split_family(fit_rows: list[dict[str, Any]],
                               calibration_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, rows in (("fit", fit_rows), ("calibration", calibration_rows)):
        per: dict[str, Any] = {}
        groups = {"overall": rows}
        for family in sorted({row["family"] for row in rows}):
            groups[family] = [row for row in rows if row["family"] == family]
        for family, selected in groups.items():
            positive = int(sum(int(row["completion"]) for row in selected))
            per[family] = {
                "rows": len(selected), "states": len({row["state_id"] for row in selected}),
                "positive": positive, "negative": len(selected) - positive,
                "prevalence": positive / len(selected) if selected else float("nan"),
            }
        result[split] = per
    return result


# --------------------------------------------------------------- corpus gate --
def _parse_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise CorpusValidationError(f"cannot parse {path}: {exc}") from exc
    return rows


def _validate_manifest(manifest: dict[str, Any],
                       allocation: Mapping[str, Any],
                       pre_identity_validation: Mapping[str, Any],
                       manifest_launch_bindings: Mapping[str, Any],
                       selector_bindings: Mapping[str, Any],
                       contract_lineage: Mapping[str, Any] | None,
                       ) -> dict[str, dict[str, Any]]:
    expected = contract()
    if isinstance(
            manifest.get("small_completion_global_exact_execution"), Mapping):
        scientific_contract_digest = _require_digest(
            manifest.get("scorer_contract_v1_2_digest"),
            "manifest scientific predecessor scorer_contract_v1_2_digest",
        )
        lineage = validate_global_exact_scorer_contract_lineage(
            contract_lineage)
        _require(
            lineage[
                "scientific_predecessor_scorer_contract_v1_2_digest"]
            == scientific_contract_digest,
            "manifest scientific scorer digest differs from successor lineage",
        )
    else:
        _require(contract_lineage is None,
                 "legacy manifest unexpectedly has global scorer lineage")
        scientific_contract_digest = contract_digest()
    _require(expected.get("scorer_fit_allocation_design_digest")
             == FROZEN_SCORER_FIT_ALLOCATION_DESIGN_DIGEST,
             "scorer contract allocation-design binding changed")
    _require(expected.get("candidate_allocator_contract_digest")
             == allocation_contract_digest(),
             "scorer contract predecessor allocator binding changed")
    _require(expected.get("candidate_allocation_amendment_digest")
             == allocation_amendment_digest(),
             "scorer contract allocation amendment binding changed")
    _require(expected.get("invalid_scorer_identity_exclusion_digest")
             == invalid_identity_exclusion_digest(),
             "scorer contract invalid-identity exclusion binding changed")
    _require(expected.get("state_selector_amendment_digest")
             == STATE_SELECTOR.state_selector_amendment_digest(),
             "scorer contract state-selector amendment binding changed")
    target_encoder_digest = canonical_digest(expected["target_encoder"])
    try:
        validate_pre_identity_structural_validation(pre_identity_validation)
    except CandidateAllocationError as exc:
        raise CorpusValidationError(
            f"pre-identity allocation validation is invalid: {exc}") from exc
    pre_identity_digest = _require_digest(
        pre_identity_validation.get("pre_identity_validation_digest"),
        "pre_identity_validation_digest")
    post_identity = allocation.get("post_identity_pre_outcome_validation")
    _require(isinstance(post_identity, Mapping),
             "candidate allocation has no post-identity/pre-outcome validation")
    post_identity_digest = _require_digest(
        post_identity.get("post_identity_validation_digest"),
        "post_identity_validation_digest")
    _require(manifest.get("schema") == "go2_branch_corpus_v1_2_state_manifest",
             "unexpected scorer-fit manifest schema")
    _require(manifest.get("pool") == EXPECTED_POOL, "manifest is not scorer_fit")
    _require(manifest.get("complete") is True, "state manifest is not complete")
    _require(manifest.get("genesis_backend") == "cpu", "scorer-fit backend is not CPU")
    _require(manifest.get("spec") == {
        "states_per_family": EXPECTED_STATES_PER_FAMILY,
        "candidates_per_state": EXPECTED_CANDIDATES_PER_STATE,
        "strata": {stratum: 5 for stratum in EXPECTED_STRATA},
        "calibration_per_stratum_per_family": 1,
    }, "state manifest scorer-fit specification changed")
    bindings = {
        "selection_digest": expected["corpus_selection_digest"],
        "scorer_fit_allocation_design_digest":
            FROZEN_SCORER_FIT_ALLOCATION_DESIGN_DIGEST,
        "candidate_allocator_contract_digest": allocation_contract_digest(),
        "candidate_allocation_amendment_digest": allocation_amendment_digest(),
        "candidate_allocation_post_identity_validation_digest": post_identity_digest,
        "pre_identity_allocation_validation_digest": pre_identity_digest,
        "invalid_scorer_identity_exclusion_digest":
            invalid_identity_exclusion_digest(),
        **{key: selector_bindings[key] for key in SELECTOR_BINDING_KEYS},
        **{key: manifest_launch_bindings[key] for key in LAUNCH_BINDING_KEYS},
        "candidate_bank_digest": expected["candidate_bank_digest"],
        "progress_contract_digest": expected["progress_target_digest"],
        "safety_contract_digest": expected["safety_target_digest"],
        "oracle_v1_2_digest": expected["oracle_v1_2_digest"],
        "scorer_contract_v1_2_digest": scientific_contract_digest,
        "boundary": FROZEN_BRANCH_BOUNDARY_DIGEST,
        "render_contract_digest": canonical_digest(expected["render_contract"]),
        "textured_v03_renderer_contract_digest":
            textured_v03_renderer_contract_digest(),
        "preprocess_contract_digest": canonical_digest(expected["preprocess_contract"]),
        "preprocessing_digest": FROZEN_PREPROCESSING_DIGEST,
        "target_encoder_digest": target_encoder_digest,
        "target_encoder_checkpoint_sha256":
            expected["target_encoder"]["checkpoint_sha256"],
    }
    for key, value in bindings.items():
        observed = (manifest.get("boundary_digest") if key == "boundary"
                    and "boundary_digest" in manifest else manifest.get(key))
        _require(observed == value,
                 f"manifest {key} does not match the frozen binding")
    source_identity_digest = _require_digest(
        manifest.get("pre_allocation_identity_manifest_digest"),
        "pre_allocation_identity_manifest_digest")
    _require(allocation.get("source_identity_manifest_digest") == source_identity_digest,
             "candidate allocation binds a different pre-allocation identity manifest")
    if isinstance(
            manifest.get("small_completion_global_exact_execution"), Mapping):
        try:
            CORPUS_BUILDER.validate_global_exact_allocation_for_consumption(
                manifest, allocation)
        except RuntimeError as exc:
            raise CorpusValidationError(
                f"global exact allocation certificate is invalid: {exc}") from exc
    else:
        try:
            validate_allocation_manifest(
                allocation,
                expected_source_identity_manifest_digest=
                    source_identity_digest)
        except CandidateAllocationError as exc:
            raise CorpusValidationError(
                f"candidate allocation manifest is invalid: {exc}") from exc
    allocation_digest = allocation_manifest_digest(allocation)
    _require(allocation.get("allocation_manifest_digest") == allocation_digest,
             "candidate allocation manifest self digest does not verify")
    _require(manifest.get("candidate_allocation_manifest_digest") == allocation_digest,
             "state manifest does not bind the verified candidate allocation")
    assignments = {
        str(assignment["state_id"]): assignment
        for assignment in allocation["assignments"]
    }
    recorded_manifest_digest = _require_digest(
        manifest.get("state_manifest_digest"), "state_manifest_digest")
    computed_manifest_digest = hashlib.sha256(json.dumps(
        {key: value for key, value in manifest.items()
         if key != "state_manifest_digest"}, sort_keys=True).encode()).hexdigest()
    _require(recorded_manifest_digest == computed_manifest_digest,
             "state_manifest_digest does not verify")

    states = manifest.get("states")
    _require(isinstance(states, list) and len(states) == EXPECTED_STATES,
             f"manifest must contain exactly {EXPECTED_STATES} states")
    by_id: dict[str, dict[str, Any]] = {}
    families: Counter[str] = Counter()
    family_stratum: Counter[tuple[str, str]] = Counter()
    family_split: Counter[tuple[str, str]] = Counter()
    family_stratum_split: Counter[tuple[str, str, str]] = Counter()
    candidate_total: Counter[int] = Counter()
    candidate_split: Counter[tuple[int, str]] = Counter()
    candidate_family: Counter[tuple[int, str]] = Counter()
    candidate_stratum: Counter[tuple[int, str]] = Counter()
    scenes: set[str] = set()
    episode_clusters: set[str] = set()
    state_identity_digests: set[str] = set()
    branch_identity_digests: set[str] = set()
    for expected_index, state in enumerate(states):
        _require(isinstance(state, dict), f"manifest state {expected_index} is not an object")
        state_id = state.get("state_id")
        _require(isinstance(state_id, str) and state_id not in by_id,
                 f"duplicate or missing state_id at index {expected_index}")
        _require(state.get("state_index") == expected_index,
                 f"state_index is not contiguous at {state_id}")
        identity_payload = {
            key: value for key, value in state.items()
            if key not in {"state_identity_digest", "state_index",
                           "candidate_indices", "candidate_rotation_index",
                           "branch_identities"}
        }
        expected_state_identity = canonical_digest({
            "schema": "go2_branch_state_identity_v1_2",
            "selection_digest": expected["corpus_selection_digest"],
            "scorer_contract_v1_2_digest": scientific_contract_digest,
            "state": identity_payload,
        })
        _require(state.get("state_identity_digest") == expected_state_identity,
                 f"state identity digest does not verify for {state_id}")
        _require(expected_state_identity not in state_identity_digests,
                 f"state identity digest is reused at {state_id}")
        state_identity_digests.add(expected_state_identity)
        family, stratum, split = (state.get("family"), state.get("stratum"),
                                  state.get("split_role"))
        _require(isinstance(family, str), f"missing family for {state_id}")
        _require(stratum in EXPECTED_STRATA, f"invalid stratum for {state_id}")
        _require(split in {"fit", "calibration"}, f"invalid split for {state_id}")
        scene = state.get("scene_id")
        _require(isinstance(scene, str) and scene not in scenes,
                 f"scene is missing or reused at {state_id}")
        scenes.add(scene)
        episode_cluster = state.get("episode_cluster_id")
        _require(isinstance(episode_cluster, str)
                 and episode_cluster not in episode_clusters,
                 f"episode cluster is missing or reused at {state_id}")
        episode_clusters.add(episode_cluster)
        goal = state.get("goal")
        _require(isinstance(goal, dict), f"missing snapshot-time goal for {state_id}")
        for key in ("landmark_id", "landmark_cell", "bearing_body_rad", "range_m"):
            _require(key in goal, f"goal.{key} missing for {state_id}")
        _require(_finite_number(goal["bearing_body_rad"])
                 and _finite_number(goal["range_m"]), f"invalid goal binding for {state_id}")
        assignment = assignments.get(str(state_id))
        _require(assignment is not None,
                 f"candidate allocation has no assignment for {state_id}")
        for key in ("state_identity_digest", "family", "stratum", "split_role",
                    "goal_type"):
            _require(state.get(key) == assignment.get(key),
                     f"candidate allocation {key} differs for {state_id}")
        candidates = state.get("candidate_indices")
        _require(candidates == assignment.get("candidate_indices"),
                 f"candidate allocation changed for {state_id}")
        branch_identities = state.get("branch_identities")
        _require(isinstance(branch_identities, list)
                 and len(branch_identities) == EXPECTED_CANDIDATES_PER_STATE,
                 f"pre-outcome branch identities are incomplete for {state_id}")
        branch_candidates: set[int] = set()
        for branch_identity in branch_identities:
            _require(isinstance(branch_identity, Mapping),
                     f"invalid branch identity record for {state_id}")
            candidate_index = branch_identity.get("candidate_index")
            _require(candidate_index in candidates and candidate_index not in branch_candidates,
                     f"branch identity candidate mismatch for {state_id}")
            branch_candidates.add(candidate_index)
            _require(branch_identity.get("candidate")
                     == FROZEN_CANDIDATES[candidate_index],
                     f"branch identity candidate name mismatch for {state_id}")
            recorded_branch_digest = _require_digest(
                branch_identity.get("branch_identity_digest"),
                f"{state_id} branch_identity_digest")
            branch_payload = {
                key: value for key, value in branch_identity.items()
                if key != "branch_identity_digest"
            }
            _require(canonical_digest(branch_payload) == recorded_branch_digest,
                     f"branch identity digest does not verify for {state_id}")
            _require(recorded_branch_digest not in branch_identity_digests,
                     f"branch identity digest is reused for {state_id}")
            branch_identity_digests.add(recorded_branch_digest)
            for identity_key in ("state_id", "state_identity_digest", "scene_id",
                                 "episode_cluster_id", "source_step", "goal"):
                _require(branch_identity.get(identity_key) == state.get(identity_key),
                         f"branch identity {identity_key} differs for {state_id}")
            for identity_key, expected_value in (
                ("schema", "go2_branch_identity_v1_2"),
                ("pool", EXPECTED_POOL),
                ("candidate_bank_digest", expected["candidate_bank_digest"]),
                ("oracle_v1_2_digest", expected["oracle_v1_2_digest"]),
                ("scorer_contract_v1_2_digest", scientific_contract_digest),
                ("render_contract_digest",
                 canonical_digest(expected["render_contract"])),
                ("textured_v03_renderer_contract_digest",
                 textured_v03_renderer_contract_digest()),
                ("preprocess_contract_digest",
                 canonical_digest(expected["preprocess_contract"])),
                ("target_encoder_digest", target_encoder_digest),
                ("candidate_allocation_amendment_digest",
                 allocation_amendment_digest()),
                ("candidate_allocation_post_identity_validation_digest",
                 post_identity_digest),
                ("pre_identity_allocation_validation_digest", pre_identity_digest),
                ("invalid_scorer_identity_exclusion_digest",
                 invalid_identity_exclusion_digest()),
                *((key, selector_bindings[key]) for key in SELECTOR_BINDING_KEYS),
                *((key, manifest_launch_bindings[key])
                  for key in LAUNCH_BINDING_KEYS),
            ):
                _require(branch_identity.get(identity_key) == expected_value,
                         f"branch identity {identity_key} differs for {state_id}")
        _require(branch_candidates == set(candidates),
                 f"pre-outcome branch identities do not cover {state_id}")
        by_id[state_id] = state
        families[family] += 1
        family_stratum[(family, stratum)] += 1
        family_split[(family, split)] += 1
        family_stratum_split[(family, stratum, split)] += 1
        for candidate in candidates:
            candidate_total[candidate] += 1
            candidate_split[(candidate, split)] += 1
            candidate_family[(candidate, family)] += 1
            candidate_stratum[(candidate, stratum)] += 1

    _require(len(families) == EXPECTED_FAMILIES
             and set(families.values()) == {EXPECTED_STATES_PER_FAMILY},
             "family allocation is not 8 x 15")
    for family in families:
        for stratum in EXPECTED_STRATA:
            _require(family_stratum[(family, stratum)] == 5,
                     f"{family}/{stratum} does not contain five states")
            _require(family_stratum_split[(family, stratum, "fit")] == 4,
                     f"{family}/{stratum} does not contain four fit states")
            _require(family_stratum_split[(family, stratum, "calibration")] == 1,
                     f"{family}/{stratum} does not contain one calibration state")
        _require(family_split[(family, "fit")] == 12
                 and family_split[(family, "calibration")] == 3,
                 f"{family} is not split 12 fit / 3 calibration")
    _require(all(candidate_total[index] == 60 for index in range(12)),
             "each candidate must occur exactly 60 times")
    _require(all(candidate_split[(index, "fit")] == 48
                 and candidate_split[(index, "calibration")] == 12
                 for index in range(12)), "candidate is confounded with fit/calibration")
    _require(all(7 <= candidate_family[(index, family)] <= 8
                 for index in range(12) for family in families),
             "candidate is confounded with family")
    _require(all(candidate_stratum[(index, stratum)] == 20
                 for index in range(12) for stratum in EXPECTED_STRATA),
             "candidate is confounded with state stratum")
    _require(manifest.get("attempted_branch_count_registered") == EXPECTED_BRANCHES,
             "manifest does not register exactly 720 branch identities")
    _require(manifest.get("branch_identity_set_digest")
             == canonical_digest(sorted(branch_identity_digests)),
             "manifest branch identity set digest does not verify")
    disjointness = manifest.get("disjointness")
    _require(isinstance(disjointness, Mapping),
             "manifest has no disjointness completion evidence")
    for key, expected_count in (
        ("state_count", EXPECTED_STATES),
        ("unique_scene_count", EXPECTED_STATES),
        ("unique_episode_cluster_count", EXPECTED_STATES),
        ("unique_state_identity_count", EXPECTED_STATES),
        ("unique_branch_identity_count", EXPECTED_BRANCHES),
    ):
        _require(disjointness.get(key) == expected_count,
                 f"manifest disjointness {key} does not verify")
    _require(disjointness.get("scene_episode_state_branch_disjoint") is True,
             "manifest disjointness gate did not pass")
    recorded_appearances = manifest.get("candidate_appearances")
    _require(recorded_appearances == {name: 60 for name in FROZEN_CANDIDATES},
             "manifest candidate_appearances does not verify")
    return by_id


def _path_in_pool(raw: Any, pool_dir: Path) -> bool:
    return _resolve_pool_artifact(raw, pool_dir) is not None


def _frame_records(row: Mapping[str, Any], kind: str) -> list[Mapping[str, Any]] | None:
    direct = row.get(f"{kind}_frames")
    if isinstance(direct, list) and all(isinstance(value, Mapping) for value in direct):
        return list(direct)
    direct = row.get(f"{kind}_records")
    if isinstance(direct, list) and all(isinstance(value, Mapping) for value in direct):
        return list(direct)
    combined = row.get("frame_records")
    if isinstance(combined, Mapping):
        direct = combined.get(kind)
        if isinstance(direct, list) and all(isinstance(value, Mapping) for value in direct):
            return list(direct)
    return None


def _verify_frame_records(records: Sequence[Mapping[str, Any]], *, expected: int,
                          pool_dir: Path, label: str,
                          cache: dict[str, tuple[int, str]],
                          first_index: int) -> None:
    _require(len(records) == expected, f"{label} has the wrong frame count")
    for index, record in enumerate(records):
        recorded_index = record.get("slot", record.get("horizon"))
        _require(recorded_index == index + first_index,
                 f"{label}[{index}] order/index binding changed")
        _require(record.get("shape") == [224, 224, 3]
                 and record.get("dtype") == "uint8",
                 f"{label}[{index}] render shape/dtype changed")
        path = _resolve_pool_artifact(record.get("path"), pool_dir)
        _require(path is not None, f"{label}[{index}] path is missing or outside its pool")
        expected_digest = _require_digest(record.get("sha256"),
                                          f"{label}[{index}].sha256")
        expected_bytes = record.get("byte_count")
        _require(isinstance(expected_bytes, int) and expected_bytes > 0,
                 f"{label}[{index}].byte_count is invalid")
        cache_key = str(path)
        if cache_key not in cache:
            assert path is not None
            cache[cache_key] = (path.stat().st_size, sha256_file(path))
        actual_bytes, actual_digest = cache[cache_key]
        _require(actual_bytes == expected_bytes and actual_digest == expected_digest,
                 f"{label}[{index}] frame binding does not verify")


def _validate_rows(rows: list[dict[str, Any]], states: dict[str, dict[str, Any]],
                   manifest: dict[str, Any], pool_dir: Path,
                   *, verify_frame_paths: bool,
                   scientific_contract_digest: str,
                   ) -> list[dict[str, Any]]:
    _require(len(rows) == EXPECTED_BRANCHES,
             f"branch_rows.jsonl must contain exactly {EXPECTED_BRANCHES} rows")
    expected_contract = contract()
    seen: set[tuple[str, int]] = set()
    sorted_rows: list[dict[str, Any]] = []
    frame_cache: dict[str, tuple[int, str]] = {}
    render_digest = canonical_digest(expected_contract["render_contract"])
    target_encoder_digest = canonical_digest(expected_contract["target_encoder"])
    expected_encoder_checkpoint = expected_contract["target_encoder"].get(
        "checkpoint_sha256")
    _require_digest(expected_encoder_checkpoint,
                    "scorer contract target encoder checkpoint_sha256")
    for row_number, row in enumerate(rows, 1):
        _require(row.get("schema") == "go2_branch_corpus_v1_2_branch_row",
                 f"row {row_number} has an unexpected schema")
        _require(row.get("record_complete") is True,
                 f"row {row_number} is not durably complete")
        recorded_row_digest = _require_digest(row.get("branch_row_digest"),
                                              f"row {row_number} branch_row_digest")
        row_payload = {key: value for key, value in row.items()
                       if key != "branch_row_digest"}
        _require(recorded_row_digest in {
            canonical_digest(row_payload), sorted_json_digest(row_payload)},
            f"row {row_number} branch_row_digest does not verify")
        state_id = row.get("state_id")
        _require(state_id in states, f"row {row_number} has unknown state_id")
        state = states[str(state_id)]
        _require(row.get("state_identity_digest") == state["state_identity_digest"],
                 f"row {row_number} has the wrong state identity binding")
        candidate_index = row.get("candidate_index")
        _require(isinstance(candidate_index, int)
                 and candidate_index in state["candidate_indices"],
                 f"row {row_number} has an unallocated candidate")
        key = (str(state_id), candidate_index)
        _require(key not in seen, f"duplicate branch row {key}")
        seen.add(key)
        _require(row.get("candidate") == FROZEN_CANDIDATES[candidate_index],
                 f"candidate name/index mismatch in row {row_number}")
        for field in ("state_index", "scene_id", "family", "stratum", "split_role"):
            _require(row.get(field) == state.get(field),
                     f"row {row_number} {field} differs from the identity manifest")
        _require(row.get("pool") == EXPECTED_POOL,
                 f"row {row_number} is not assigned to scorer_fit")
        _require(row.get("valid") is True and row.get("invalid_reason") in (None, ""),
                 f"row {row_number} is not a valid completed branch")
        _require(row.get("state_manifest_digest") == manifest["state_manifest_digest"],
                 f"row {row_number} has the wrong state manifest binding")
        _require(row.get("oracle_v1_2_digest") == expected_contract["oracle_v1_2_digest"],
                 f"row {row_number} has the wrong oracle binding")
        _require(row.get("scorer_contract_v1_2_digest")
                 == scientific_contract_digest,
                 f"row {row_number} has the wrong scorer-contract binding")
        for aliases, expected_value in (
            (("candidate_allocation_manifest_digest",),
             manifest["candidate_allocation_manifest_digest"]),
            (("candidate_allocator_contract_digest",), allocation_contract_digest()),
            (("candidate_allocation_amendment_digest",), allocation_amendment_digest()),
            (("candidate_allocation_post_identity_validation_digest",),
             manifest["candidate_allocation_post_identity_validation_digest"]),
            (("pre_identity_allocation_validation_digest",),
             manifest["pre_identity_allocation_validation_digest"]),
            (("invalid_scorer_identity_exclusion_digest",),
             invalid_identity_exclusion_digest()),
            *(((key,), manifest[key]) for key in SELECTOR_BINDING_KEYS),
            *(((key,), manifest[key]) for key in LAUNCH_BINDING_KEYS),
            (("candidate_bank_digest",), expected_contract["candidate_bank_digest"]),
            (("progress_contract_digest", "progress_target_digest"),
             expected_contract["progress_target_digest"]),
            (("safety_contract_digest", "safety_target_digest"),
             expected_contract["safety_target_digest"]),
            (("selection_digest",), expected_contract["corpus_selection_digest"]),
            (("boundary_digest", "boundary"), FROZEN_BRANCH_BOUNDARY_DIGEST),
            (("render_contract_digest", "rendering_contract_digest"), render_digest),
            (("textured_v03_renderer_contract_digest",),
             textured_v03_renderer_contract_digest()),
            (("preprocess_contract_digest", "preprocessing_contract_digest"),
             canonical_digest(expected_contract["preprocess_contract"])),
            (("preprocessing_digest",), FROZEN_PREPROCESSING_DIGEST),
            (("target_encoder_digest",), target_encoder_digest),
        ):
            observed = _bound_value(row, aliases)
            _require(observed == expected_value,
                     f"row {row_number} has the wrong or missing {aliases[0]}")
        _require(_bound_value(row, ("target_encoder_checkpoint_sha256",))
                 == expected_encoder_checkpoint,
                 f"row {row_number} target encoder checkpoint binding changed")
        if "preprocessing_digest" in row:
            _require(row["preprocessing_digest"] == FROZEN_PREPROCESSING_DIGEST,
                     f"row {row_number} preprocessing_digest changed")
        _require_digest(row.get("branch_identity_digest"),
                        f"row {row_number} branch_identity_digest")
        registered_branch = next(
            identity for identity in state["branch_identities"]
            if identity["candidate_index"] == candidate_index)
        _require(row["branch_identity_digest"]
                 == registered_branch["branch_identity_digest"],
                 f"row {row_number} differs from its pre-outcome branch identity")
        _require_digest(row.get("snapshot_digest"), f"row {row_number} snapshot_digest")
        _require(canonical_digest(row.get("goal")) == canonical_digest(state.get("goal")),
                 f"row {row_number} goal was not bound at snapshot time")
        expected_goal_input = [
            math.sin(float(state["goal"]["bearing_body_rad"])),
            math.cos(float(state["goal"]["bearing_body_rad"])),
            float(state["goal"]["range_m"]),
        ]
        observed_goal_input = row.get("goal_binding_input")
        _require(isinstance(observed_goal_input, list)
                 and len(observed_goal_input) == GOAL_DIM
                 and all(_finite_number(value) for value in observed_goal_input)
                 and np.allclose(observed_goal_input, expected_goal_input,
                                 rtol=0.0, atol=1e-12),
                 f"row {row_number} numeric goal binding changed")

        def valid_tick_blocks(value: Any) -> bool:
            return (isinstance(value, list) and len(value) == HORIZONS
                    and all(isinstance(block, list) and len(block) == 5
                            and all(isinstance(tick, list) and len(tick) == 2
                                    and all(_finite_number(component)
                                            for component in tick)
                                    for tick in block)
                            for block in value))

        requested = row.get("requested")
        realised_requested = row.get("realised_requested_prefix")
        post_slew = row.get("post_slew")
        registered_post_slew = row.get("candidate_post_slew_plan")
        _require(valid_tick_blocks(requested)
                 and valid_tick_blocks(realised_requested)
                 and valid_tick_blocks(post_slew)
                 and valid_tick_blocks(registered_post_slew),
                 f"row {row_number} requested/realised action trajectories are incomplete")
        _require(row.get("blocks_completed") == HORIZONS
                 and row.get("truncated_at_block") is None,
                 f"row {row_number} did not complete all four action blocks")
        _require(np.allclose(post_slew, registered_post_slew,
                             rtol=0.0, atol=1e-6),
                 f"row {row_number} realised post-slew action differs from its candidate")
        action_blocks = row.get("action_blocks")
        _require(isinstance(action_blocks, list) and len(action_blocks) == HORIZONS
                 and all(isinstance(block, list) and len(block) == 10
                         and all(_finite_number(value) for value in block)
                         for block in action_blocks),
                 f"row {row_number} does not have a 4 x 10 post-slew action")
        flattened_post_slew = [[component for tick in block for component in tick]
                               for block in post_slew]
        _require(np.allclose(action_blocks, flattened_post_slew,
                             rtol=0.0, atol=1e-6),
                 f"row {row_number} scorer action input differs from realised post-slew")
        for field in (
            "start_geodesic_m", "final_geodesic_m", "progress",
            "contact_fraction", "clearance_cost", "stuck_fraction", "fall",
            "safety", "completion", "utility", "min_clearance_m",
        ):
            _require(_finite_number(row.get(field)), f"row {row_number} has invalid {field}")
        _require(row.get("evaluation_points") == 20,
                 f"row {row_number} does not contain all 20 oracle evaluation points")
        expected_progress = max(-1.0, min(
            1.0, (float(row["start_geodesic_m"])
                  - float(row["final_geodesic_m"])) / PROGRESS_NORMALISER_M))
        _require(math.isclose(float(row["progress"]), expected_progress,
                              rel_tol=0.0, abs_tol=1e-12),
                 f"row {row_number} progress component does not verify")
        for field in ("contact_fraction", "clearance_cost", "stuck_fraction"):
            _require(0.0 <= float(row[field]) <= 1.0,
                     f"row {row_number} {field} is outside [0,1]")
        _require(float(row["fall"]) in (0.0, 1.0),
                 f"row {row_number} fall component is not binary")
        expected_safety = max(
            float(row["fall"]),
            (float(row["contact_fraction"]) + float(row["clearance_cost"])
             + float(row["stuck_fraction"])) / 3.0,
        )
        _require(0.0 <= float(row["safety"]) <= 1.0,
                 f"row {row_number} safety is outside [0,1]")
        _require(math.isclose(float(row["safety"]), expected_safety,
                              rel_tol=0.0, abs_tol=1e-12),
                 f"row {row_number} graded safety components do not verify")
        _require(float(row["completion"]) in (0.0, 1.0),
                 f"row {row_number} completion is not binary")
        expected_utility = (float(WEIGHTS["progress"]) * float(row["progress"])
                            + float(WEIGHTS["safety"]) * float(row["safety"])
                            + float(WEIGHTS["completion"]) * float(row["completion"]))
        _require(math.isclose(float(row["utility"]), expected_utility,
                              rel_tol=0.0, abs_tol=1e-12),
                 f"row {row_number} utility does not equal the frozen composite")
        context_records = _frame_records(row, "context")
        horizon_records = _frame_records(row, "horizon")
        _require(context_records is not None and len(context_records) == CONTEXT_SLOTS,
                 f"row {row_number} does not bind three context renders")
        _require(horizon_records is not None and len(horizon_records) == HORIZONS,
                 f"row {row_number} does not bind all H=1..4 renders")
        if verify_frame_paths:
            assert context_records is not None and horizon_records is not None
            _verify_frame_records(context_records, expected=CONTEXT_SLOTS,
                                  pool_dir=pool_dir,
                                  label=f"row {row_number} context", cache=frame_cache,
                                  first_index=0)
            _verify_frame_records(horizon_records, expected=HORIZONS,
                                  pool_dir=pool_dir,
                                  label=f"row {row_number} horizon", cache=frame_cache,
                                  first_index=1)
        sorted_rows.append(row)
    expected_keys = {
        (state_id, candidate) for state_id, state in states.items()
        for candidate in state["candidate_indices"]
    }
    _require(seen == expected_keys, "branch ledger does not cover the frozen allocation")
    sorted_rows.sort(key=lambda row: (int(row["state_index"]), int(row["candidate_index"])))
    return sorted_rows


def _validate_receipt(receipt: dict[str, Any], manifest: dict[str, Any],
                      branch_rows_sha256: str,
                      rows: Sequence[Mapping[str, Any]], *,
                      scientific_contract_digest: str,
                      ) -> str:
    _require(receipt.get("schema") == "go2_branch_corpus_v1_2_completion_receipt",
             "unexpected corpus receipt schema")
    _require(receipt.get("pool") == EXPECTED_POOL,
             "corpus receipt is not scorer_fit")
    _require(receipt.get("complete") is True, "corpus receipt is not complete")
    counts = (
        (_receipt_value(receipt, ("states", "state_count")), EXPECTED_STATES, "state"),
        (_receipt_value(receipt, ("attempted_branches", "attempted_branch_count",
                                  "attempt_count", "attempted")),
         EXPECTED_BRANCHES, "attempt"),
        (_receipt_value(receipt, ("valid_branches", "valid_branch_count",
                                  "valid_count", "valid")),
         EXPECTED_BRANCHES, "valid"),
    )
    for observed, expected, label in counts:
        _require(observed == expected, f"corpus receipt {label} count is not {expected}")
    _require(_receipt_value(receipt, ("branch_rows_sha256",)) == branch_rows_sha256,
             "corpus receipt branch_rows_sha256 does not verify")
    expected = contract()
    required = (
        (("state_manifest_digest",), manifest["state_manifest_digest"]),
        (("candidate_bank_digest",), expected["candidate_bank_digest"]),
        (("progress_contract_digest", "progress_target_digest"),
         expected["progress_target_digest"]),
        (("safety_contract_digest", "safety_target_digest"),
         expected["safety_target_digest"]),
        (("oracle_v1_2_digest",), expected["oracle_v1_2_digest"]),
        (("scorer_contract_v1_2_digest", "contract_digest"),
         scientific_contract_digest),
        (("selection_digest",), expected["corpus_selection_digest"]),
        (("candidate_allocation_manifest_digest",),
         manifest["candidate_allocation_manifest_digest"]),
        (("candidate_allocator_contract_digest",), allocation_contract_digest()),
        (("candidate_allocation_amendment_digest",), allocation_amendment_digest()),
        (("candidate_allocation_post_identity_validation_digest",),
         manifest["candidate_allocation_post_identity_validation_digest"]),
        (("pre_identity_allocation_validation_digest",),
         manifest["pre_identity_allocation_validation_digest"]),
        (("invalid_scorer_identity_exclusion_digest",),
         invalid_identity_exclusion_digest()),
        *(((key,), manifest[key]) for key in SELECTOR_BINDING_KEYS),
        *(((key,), manifest[key]) for key in LAUNCH_BINDING_KEYS),
        (("boundary_digest", "boundary"), FROZEN_BRANCH_BOUNDARY_DIGEST),
        (("render_contract_digest",), canonical_digest(expected["render_contract"])),
        (("textured_v03_renderer_contract_digest",),
         textured_v03_renderer_contract_digest()),
        (("preprocess_contract_digest",),
         canonical_digest(expected["preprocess_contract"])),
        (("preprocessing_digest",), FROZEN_PREPROCESSING_DIGEST),
        (("target_encoder_digest",), canonical_digest(expected["target_encoder"])),
        (("target_encoder_checkpoint_sha256",),
         expected["target_encoder"]["checkpoint_sha256"]),
    )
    for aliases, value in required:
        _require(_receipt_value(receipt, aliases) == value,
                 f"corpus receipt binding {aliases[0]} does not verify")
    corpus_digest = _require_digest(receipt.get("corpus_digest"), "corpus_digest")
    payload = receipt.get("corpus_digest_payload")
    _require(isinstance(payload, dict), "corpus receipt has no canonical digest payload")
    _require(canonical_digest(payload) == corpus_digest,
             "corpus_digest_payload does not verify")
    assert isinstance(payload, dict)
    for key, expected_value in (
        ("schema", "go2_branch_corpus_v1_2_corpus_identity"),
        ("pool", EXPECTED_POOL),
        ("state_manifest_digest", manifest["state_manifest_digest"]),
        ("candidate_allocation_manifest_digest",
         manifest["candidate_allocation_manifest_digest"]),
        ("branch_identity_set_digest", manifest["branch_identity_set_digest"]),
        ("branch_rows_sha256", branch_rows_sha256),
        ("state_count", EXPECTED_STATES),
        ("attempted_branch_count", EXPECTED_BRANCHES),
        ("valid_branch_count", EXPECTED_BRANCHES),
        ("invalid_branch_count", 0),
        ("complete", True),
    ):
        _require(payload.get(key) == expected_value,
                 f"corpus digest payload {key} does not verify")
    row_by_key = {
        (str(row["state_id"]), int(row["candidate_index"])): row for row in rows
    }
    expected_row_digests = [
        row_by_key[(str(state["state_id"]), int(candidate))]["branch_row_digest"]
        for state in manifest["states"] for candidate in state["candidate_indices"]
    ]
    _require(payload.get("branch_row_digests") == expected_row_digests,
             "corpus digest payload does not bind the ordered branch rows")
    for aliases, value in required:
        _require(_receipt_value(payload, aliases) == value,
                 f"corpus digest payload binding {aliases[0]} does not verify")
    return corpus_digest


def _validate_smoke_receipts(
        pool_dir: Path, manifest: Mapping[str, Any], *,
        contract_lineage: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    expected = contract()
    lineage = (
        validate_global_exact_scorer_contract_lineage(contract_lineage)
        if contract_lineage is not None else None
    )
    scientific_contract_digest = (
        _require_digest(
            manifest.get("scorer_contract_v1_2_digest"),
            "smoke scientific predecessor scorer_contract_v1_2_digest",
        )
        if lineage is not None else contract_digest()
    )
    operational_contract_digest = (
        lineage["current_scorer_contract_v1_2_digest"]
        if lineage is not None else contract_digest()
    )
    paths = (
        (pool_dir / "smoke_branch_receipt.json", "smoke_branch_receipt_digest",
         "go2_scorer_fit_branch_smoke_receipt_v1_2"),
        (pool_dir / "smoke_encoding_receipt.json", "smoke_receipt_digest",
         "go2_scorer_fit_end_to_end_smoke_receipt_v1"),
    )
    verified: dict[str, Any] = {}
    required = (
        (("state_manifest_digest",), manifest["state_manifest_digest"]),
        (("candidate_allocator_contract_digest",), allocation_contract_digest()),
        (("candidate_allocation_amendment_digest",), allocation_amendment_digest()),
        (("candidate_allocation_post_identity_validation_digest",),
         manifest["candidate_allocation_post_identity_validation_digest"]),
        (("pre_identity_allocation_validation_digest",),
         manifest["pre_identity_allocation_validation_digest"]),
        (("invalid_scorer_identity_exclusion_digest",),
         invalid_identity_exclusion_digest()),
        *(((key,), manifest[key]) for key in SELECTOR_BINDING_KEYS),
        *(((key,), manifest[key]) for key in LAUNCH_BINDING_KEYS),
        (("render_contract_digest",), canonical_digest(expected["render_contract"])),
        (("textured_v03_renderer_contract_digest",),
         textured_v03_renderer_contract_digest()),
        (("preprocess_contract_digest",),
         canonical_digest(expected["preprocess_contract"])),
        (("preprocessing_digest",), FROZEN_PREPROCESSING_DIGEST),
        (("target_encoder_digest",), canonical_digest(expected["target_encoder"])),
        (("target_encoder_checkpoint_sha256",),
         expected["target_encoder"]["checkpoint_sha256"]),
    )
    for path, self_key, schema in paths:
        _require(path.is_file(), f"missing required end-to-end smoke receipt {path}")
        receipt = _read_json(path)
        _require(receipt.get("schema") == schema,
                 f"{path.name} has an unexpected schema")
        _require(receipt.get("pass") is True, f"smoke receipt did not pass: {path}")
        observed_digest = _require_digest(receipt.get(self_key), f"{path.name} {self_key}")
        payload = {key: value for key, value in receipt.items() if key != self_key}
        _require(canonical_digest(payload) == observed_digest,
                 f"{path.name} self digest does not verify")
        expected_scorer_digest = (
            scientific_contract_digest
            if path.name == "smoke_branch_receipt.json"
            else operational_contract_digest
        )
        _require(
            receipt.get("scorer_contract_v1_2_digest")
            == expected_scorer_digest,
            f"{path.name} scorer-contract role does not verify",
        )
        if lineage is not None and path.name == "smoke_encoding_receipt.json":
            validate_global_exact_scorer_contract_lineage(
                receipt.get("global_exact_scorer_contract_lineage"),
                expected=lineage,
            )
        for aliases, expected_value in required:
            _require(_bound_value(receipt, aliases) == expected_value,
                     f"{path.name} binding {aliases[0]} does not verify")
        verified[path.name] = {
            "receipt_digest": observed_digest,
            "file_sha256": sha256_file(path),
        }
    return verified


def _validate_latent_index(index: dict[str, Any], pool_dir: Path,
                           states: dict[str, dict[str, Any]],
                           rows: list[dict[str, Any]], manifest: dict[str, Any],
                           corpus_digest: str, *, verify_encoder_checkpoint: bool,
                           contract_lineage: Mapping[str, Any] | None,
                           ) -> tuple[dict[str, Any], Any, list[str], list[str]]:
    lineage = (
        validate_global_exact_scorer_contract_lineage(contract_lineage)
        if contract_lineage is not None else None
    )
    operational_contract_digest = (
        lineage["current_scorer_contract_v1_2_digest"]
        if lineage is not None else contract_digest()
    )
    _require(index.get("complete") is True, "latent index is not complete")
    _require(index.get("schema") == "go2_branch_corpus_v1_2_latents_index_v2",
             "unexpected latent index schema")
    _require(index.get("pool") == EXPECTED_POOL, "latent index is not scorer_fit")
    self_key = ("latents_index_digest" if "latents_index_digest" in index
                else "index_digest" if "index_digest" in index else None)
    _require(self_key is not None, "latent index has no self digest")
    assert self_key is not None
    recorded_index_digest = _require_digest(index.get(self_key), self_key)
    index_payload = {key: value for key, value in index.items() if key != self_key}
    _require(recorded_index_digest in {
        sorted_json_digest(index_payload), canonical_digest(index_payload)},
        "latent index self digest does not verify")
    _require(index.get("tokens") == TOKENS and index.get("token_dim") == TOKEN_DIM,
             "latent token layout changed")
    _require(index.get("horizons") == HORIZONS
             and index.get("context_slots") == CONTEXT_SLOTS
             and index.get("dtype") == "float16",
             "latent horizon/context/dtype contract changed")
    expected_context_shape = [EXPECTED_STATES, CONTEXT_SLOTS, TOKENS, TOKEN_DIM]
    expected_horizon_shape = [EXPECTED_BRANCHES, HORIZONS, TOKENS, TOKEN_DIM]
    _require(index.get("context_shape") == expected_context_shape,
             f"unexpected context latent shape {index.get('context_shape')}")
    _require(index.get("horizon_shape") == expected_horizon_shape,
             f"unexpected horizon latent shape {index.get('horizon_shape')}")
    context_records = index.get("context_records")
    horizon_records = index.get("horizon_records")
    sharded = (isinstance(context_records, list)
               and isinstance(horizon_records, list))
    context_states = ([record.get("state_id") for record in context_records]
                      if sharded else index.get("context_states"))
    horizon_keys = ([record.get("key") for record in horizon_records]
                    if sharded else index.get("horizon_keys"))
    _require(isinstance(context_states, list) and len(context_states) == EXPECTED_STATES
             and len(set(context_states)) == EXPECTED_STATES
             and set(context_states) == set(states),
             "context latent index does not cover exactly the registered states")
    expected_horizon_keys = [f"{row['state_id']}|{row['candidate']}" for row in rows]
    _require(isinstance(horizon_keys, list) and len(horizon_keys) == EXPECTED_BRANCHES
             and len(set(horizon_keys)) == EXPECTED_BRANCHES
             and set(horizon_keys) == set(expected_horizon_keys),
             "horizon latent index does not cover exactly the registered branches")

    for key, expected_value in (
        ("state_manifest_digest", manifest["state_manifest_digest"]),
        ("corpus_digest", corpus_digest),
        ("scorer_contract_v1_2_digest", operational_contract_digest),
        ("candidate_allocator_contract_digest", allocation_contract_digest()),
        ("candidate_allocation_amendment_digest", allocation_amendment_digest()),
        ("candidate_allocation_post_identity_validation_digest",
         manifest["candidate_allocation_post_identity_validation_digest"]),
        ("pre_identity_allocation_validation_digest",
         manifest["pre_identity_allocation_validation_digest"]),
        ("invalid_scorer_identity_exclusion_digest",
         invalid_identity_exclusion_digest()),
        *((key, manifest[key]) for key in SELECTOR_BINDING_KEYS),
        *((key, manifest[key]) for key in LAUNCH_BINDING_KEYS),
        ("target_encoder_digest", canonical_digest(contract()["target_encoder"])),
        ("target_encoder_checkpoint_sha256",
         contract()["target_encoder"]["checkpoint_sha256"]),
        ("preprocess_contract_digest",
         canonical_digest(contract()["preprocess_contract"])),
        ("preprocessing_digest", FROZEN_PREPROCESSING_DIGEST),
        ("branch_rows_sha256", sha256_file(pool_dir / "branch_rows.jsonl")),
    ):
        _require(_bound_value(index, (key,)) == expected_value,
                 f"latent index {key} does not verify")
    if lineage is not None:
        validate_global_exact_scorer_contract_lineage(
            index.get("global_exact_scorer_contract_lineage"),
            expected=lineage,
        )
    _require(index.get("preprocess") == EXPECTED_PREPROCESS,
             "latent preprocessing implementation changed")
    _require(index.get("target_normalisation") == EXPECTED_TARGET_NORMALISATION,
             "latent target normalisation changed")

    encoder = index.get("encoder")
    _require(isinstance(encoder, dict), "latent index has no target encoder identity")
    checkpoint_digest = _require_digest(
        encoder.get("checkpoint_sha256"), "target encoder checkpoint_sha256")
    _require(checkpoint_digest == TARGET_ENCODER["checkpoint_sha256"],
             "latent index target encoder weights differ from the scorer contract")
    checkpoint_path_raw = encoder.get("checkpoint_path")
    _require(isinstance(checkpoint_path_raw, str), "target encoder checkpoint_path is missing")
    checkpoint_path = Path(checkpoint_path_raw).expanduser().resolve()
    expected_checkpoint = Path(str(TARGET_ENCODER["checkpoint"])).expanduser().resolve()
    _require(checkpoint_path == expected_checkpoint,
             "latent index references a different target encoder checkpoint")
    model_id = str(encoder.get("model_id", ""))
    _require("vjepa2_1_vit_large_384" in model_id,
             "latent index references a different target encoder architecture")
    if verify_encoder_checkpoint:
        _require(checkpoint_path.is_file(), "bound target encoder checkpoint is missing")
        _require(checkpoint_path.stat().st_size == TARGET_ENCODER["checkpoint_byte_count"],
                 "bound target encoder checkpoint byte count changed")
        _require(sha256_file(checkpoint_path) == checkpoint_digest,
                 "bound target encoder checkpoint digest does not verify")
    for row_number, row in enumerate(rows, 1):
        _require(_bound_value(
            row, ("target_encoder_checkpoint_sha256",))
            == checkpoint_digest,
            f"row {row_number} target encoder digest differs from latent index")

    if sharded:
        assert isinstance(context_records, list) and isinstance(horizon_records, list)
        _require(len(context_records) == EXPECTED_STATES,
                 "context shard ledger must contain exactly 120 records")
        _require(len(horizon_records) == EXPECTED_BRANCHES,
                 "horizon shard ledger must contain exactly 720 records")
        for record in context_records:
            state = states.get(str(record.get("state_id")))
            _require(state is not None
                     and record.get("state_identity_digest")
                     == state.get("state_identity_digest"),
                     "context shard identity differs from the state manifest")
        rows_by_key = {
            f"{row['state_id']}|{row['candidate']}": row for row in rows
        }
        for record in horizon_records:
            row = rows_by_key.get(str(record.get("key")))
            _require(row is not None
                     and record.get("state_id") == row.get("state_id")
                     and record.get("candidate") == row.get("candidate")
                     and record.get("candidate_index") == row.get("candidate_index")
                     and record.get("branch_identity_digest")
                     == row.get("branch_identity_digest"),
                     "horizon shard identity differs from its branch row")

        def verify_shards(records: Sequence[Mapping[str, Any]], expected_shape: list[int],
                          label: str) -> list[dict[str, Any]]:
            verified: list[dict[str, Any]] = []
            expected_bytes = (int(np.prod(expected_shape))
                              * np.dtype(np.float16).itemsize)
            for position, record in enumerate(records):
                _require(record.get("shape") == expected_shape,
                         f"{label} shard {position} shape changed")
                _require(record.get("byte_count") == expected_bytes,
                         f"{label} shard {position} byte_count changed")
                recorded_digest = _require_digest(
                    record.get("sha256"), f"{label} shard {position} sha256")
                path = _resolve_pool_artifact(record.get("path"), pool_dir)
                _require(path is not None,
                         f"{label} shard {position} is missing or outside its pool")
                assert path is not None
                _require(path.stat().st_size == expected_bytes,
                         f"{label} shard {position} actual byte count changed")
                _require(sha256_file(path) == recorded_digest,
                         f"{label} shard {position} digest does not verify")
                verified.append({
                    "position": position,
                    "identity": (record.get("state_id") if label == "context"
                                 else record.get("key")),
                    "sha256": recorded_digest, "byte_count": expected_bytes,
                    "shape": expected_shape,
                })
            return verified

        verified_context = verify_shards(
            context_records, [CONTEXT_SLOTS, TOKENS, TOKEN_DIM], "context")
        verified_horizon = verify_shards(
            horizon_records, [HORIZONS, TOKENS, TOKEN_DIM], "horizon")
        _require(index.get("storage_bytes") == sum(
            int(record["byte_count"]) for record in context_records + horizon_records),
            "latent index storage_bytes does not match its shards")
        context_binding_digest = canonical_digest(verified_context)
        horizon_binding_digest = canonical_digest(verified_horizon)
        horizon_source: Any = HorizonShardStore(horizon_records, pool_dir)
        storage = "per-item-float16-shards"
    else:
        expected_bytes = {
            "context.f16": int(np.prod(expected_context_shape))
                           * np.dtype(np.float16).itemsize,
            "horizon.f16": int(np.prod(expected_horizon_shape))
                           * np.dtype(np.float16).itemsize,
        }
        actual_hashes: dict[str, str] = {}
        for filename, size in expected_bytes.items():
            path = pool_dir / filename
            _require(path.is_file(), f"missing {path}")
            _require(path.stat().st_size == size,
                     f"{filename} byte count does not match its frozen shape")
            actual_hashes[filename] = sha256_file(path)
            index_key = filename.replace(".f16", "_sha256")
            _require(index.get(index_key) == actual_hashes[filename],
                     f"{index_key} does not verify")
        context_binding_digest = actual_hashes["context.f16"]
        horizon_binding_digest = actual_hashes["horizon.f16"]
        horizon_source = np.memmap(
            pool_dir / "horizon.f16", mode="r", dtype=np.float16,
            shape=tuple(index["horizon_shape"]))
        storage = "contiguous-float16-blobs"

    return ({
        "latent_storage": storage,
        "context_latent_binding_digest": context_binding_digest,
        "horizon_latent_binding_digest": horizon_binding_digest,
        "latents_index_digest": recorded_index_digest,
        "encoder": encoder,
        "encoder_checkpoint_sha256": checkpoint_digest,
    }, horizon_source, list(context_states), list(horizon_keys))


def validate_scorer_fit_corpus(pool: str = EXPECTED_POOL, *,
                               verify_encoder_checkpoint: bool = True,
                               verify_frame_paths: bool = True) -> dict[str, Any]:
    """Validate every identity, row and latent binding before model construction."""

    if pool != EXPECTED_POOL:
        raise CorpusValidationError("training is registered only for pool=scorer_fit")
    started = time.time()
    pool_dir = OUT_ROOT / pool
    manifest_path = pool_dir / "state_manifest.json"
    pre_identity_path = pool_dir / "pre_identity_allocation_validation.json"
    allocation_path = pool_dir / "candidate_allocation_manifest.json"
    rows_path = pool_dir / "branch_rows.jsonl"
    receipt_path = pool_dir / "corpus_receipt.json"
    index_path = pool_dir / "latents_index.json"
    # This pure identity-chain replay is deliberately before branch rows,
    # latents, encoder weights, or scorer construction.  It validates the
    # durable per-scene capture prefix and the first passing small-family joint
    # search, which cannot be established by a self-signed manifest alone.
    try:
        manifest = CORPUS_BUILDER.load_active_state_manifest_for_consumption(
            manifest_path, pool=EXPECTED_POOL
        )
    except RuntimeError as exc:
        raise CorpusValidationError(
            f"active scorer-fit state selection provenance is invalid: {exc}"
        ) from exc
    manifest_path = (
        CORPUS_BUILDER.pin_active_scorer_fit_artifact_for_consumption(
            manifest_path, "state_manifest.json"))
    pre_identity_path = (
        CORPUS_BUILDER.pin_active_scorer_fit_artifact_for_consumption(
            pre_identity_path, "pre_identity_allocation_validation.json"))
    allocation_path = (
        CORPUS_BUILDER.pin_active_scorer_fit_artifact_for_consumption(
            allocation_path, "candidate_allocation_manifest.json"))
    for path in (pre_identity_path, allocation_path, rows_path,
                 receipt_path, index_path):
        _require(path.is_file(), f"missing required scorer-fit artefact {path}")
    pre_identity_validation = _read_json(pre_identity_path)
    allocation = _read_json(allocation_path)
    (launch_bindings, manifest_launch_bindings,
     selector_launch_bindings) = _load_manifest_launch_lineage(
        manifest, pool_dir, pre_identity_validation,
        enforce_managed_paths=True)
    contract_lineage = (
        validate_global_exact_scorer_contract_lineage(
            launch_bindings.get("global_exact_scorer_contract_lineage"))
        if "global_exact_successor_scorer_contract_digest" in launch_bindings
        else None
    )
    scientific_contract_digest = (
        contract_lineage[
            "scientific_predecessor_scorer_contract_v1_2_digest"]
        if contract_lineage is not None else contract_digest()
    )
    selector_bindings = _validate_selector_successor(
        pool_dir, selector_launch_bindings, allocation,
        manifest.get("states", []),
        enforce_managed_paths=True,
        global_exact_manifest=(
            manifest if isinstance(
                manifest.get("small_completion_global_exact_execution"),
                Mapping) else None))
    states = _validate_manifest(
        manifest, allocation, pre_identity_validation,
        manifest_launch_bindings,
        selector_bindings, contract_lineage)
    raw_rows = _parse_rows(rows_path)
    rows = _validate_rows(raw_rows, states, manifest, pool_dir,
                          verify_frame_paths=verify_frame_paths,
                          scientific_contract_digest=
                              scientific_contract_digest)
    rows_digest = sha256_file(rows_path)
    receipt = _read_json(receipt_path)
    corpus_digest = _validate_receipt(
        receipt, manifest, rows_digest, rows,
        scientific_contract_digest=scientific_contract_digest)
    smoke_receipts = _validate_smoke_receipts(
        pool_dir, manifest, contract_lineage=contract_lineage)
    index = _read_json(index_path)
    latent, horizon, _context_states, horizon_keys = _validate_latent_index(
        index, pool_dir, states, rows, manifest, corpus_digest,
        verify_encoder_checkpoint=verify_encoder_checkpoint,
        contract_lineage=contract_lineage)
    positions = {key: position for position, key in enumerate(horizon_keys)}
    for row in rows:
        row["_latent_index"] = positions[f"{row['state_id']}|{row['candidate']}"]
    fit_rows = [row for row in rows if row["split_role"] == "fit"]
    calibration_rows = [row for row in rows if row["split_role"] == "calibration"]
    _require(len({row["state_id"] for row in fit_rows}) == EXPECTED_FIT_STATES
             and len(fit_rows) == EXPECTED_FIT_ROWS,
             "fit split is not 96 states / 576 rows")
    _require(len({row["state_id"] for row in calibration_rows})
             == EXPECTED_CALIBRATION_STATES
             and len(calibration_rows) == EXPECTED_CALIBRATION_ROWS,
             "calibration split is not 24 states / 144 rows")
    fit_scenes = {row["scene_id"] for row in fit_rows}
    calibration_scenes = {row["scene_id"] for row in calibration_rows}
    _require(not fit_scenes & calibration_scenes,
             "fit/calibration split is not scene-disjoint")
    training_order_digest = canonical_digest([
        [row["state_id"], row["candidate"], row["_latent_index"]] for row in fit_rows
    ])
    return {
        "rows": rows, "fit_rows": fit_rows, "calibration_rows": calibration_rows,
        "horizon": horizon, "index": index, "manifest": manifest,
        "receipt": receipt,
        "bindings": {
            "frozen_allocation_design_digest": FROZEN_SCORER_FIT_ALLOCATION_DESIGN_DIGEST,
            "corpus_selection_digest": contract()["corpus_selection_digest"],
            "candidate_allocation_manifest_digest":
                allocation["allocation_manifest_digest"],
            "candidate_allocation_manifest_sha256": sha256_file(allocation_path),
            "pre_allocation_identity_manifest_digest":
                manifest["pre_allocation_identity_manifest_digest"],
            "candidate_allocator_contract_digest": allocation_contract_digest(),
            "candidate_allocation_amendment_digest": allocation_amendment_digest(),
            "candidate_allocation_post_identity_validation_digest":
                manifest["candidate_allocation_post_identity_validation_digest"],
            "pre_identity_allocation_validation_digest":
                manifest["pre_identity_allocation_validation_digest"],
            "pre_identity_allocation_validation_sha256":
                sha256_file(pre_identity_path),
            "invalid_scorer_identity_exclusion_digest":
                invalid_identity_exclusion_digest(),
            **{key: selector_bindings[key] for key in SELECTOR_BINDING_KEYS},
            **{key: launch_bindings[key] for key in LAUNCH_BINDING_KEYS},
            "clean_source_launch_receipt_sha256":
                launch_bindings["clean_source_launch_receipt_sha256"],
            "scorer_contract_artifact_sha256":
                launch_bindings["scorer_contract_artifact_sha256"],
            **({
                key: launch_bindings[key]
                for key in GLOBAL_EXACT_PROVENANCE_BINDING_KEYS
            } if "global_exact_successor_scorer_contract_digest"
               in launch_bindings else {}),
            "state_manifest_digest": manifest["state_manifest_digest"],
            "state_manifest_file_sha256": sha256_file(manifest_path),
            "branch_rows_sha256": rows_digest,
            "corpus_receipt_sha256": sha256_file(receipt_path),
            "corpus_digest": corpus_digest,
            "smoke_receipts": smoke_receipts,
            "latent_index_sha256": sha256_file(index_path),
            **latent,
            "training_row_order_digest": training_order_digest,
        },
        "validation_wall_time_s": round(time.time() - started, 3),
    }


def load_corpus(pool: str) -> dict[str, Any]:
    """Lightweight loader retained for the downstream, separately gated scorer."""

    out = OUT_ROOT / pool
    rows = _parse_rows(out / "branch_rows.jsonl")
    index = _read_json(out / "latents_index.json")
    records = index.get("horizon_records")
    if isinstance(records, list):
        horizon = HorizonShardStore(records, out)
        horizon_keys = [record.get("key") for record in records]
    else:
        shape = tuple(index["horizon_shape"])
        expected_size = int(np.prod(shape)) * np.dtype(np.float16).itemsize
        horizon_path = out / "horizon.f16"
        if not horizon_path.is_file() or horizon_path.stat().st_size != expected_size:
            raise CorpusValidationError("horizon latent blob is absent or has the wrong size")
        horizon = np.memmap(horizon_path, mode="r", dtype=np.float16, shape=shape)
        horizon_keys = index["horizon_keys"]
    keys = {key: position for position, key in enumerate(horizon_keys)}
    usable: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row.get('state_id')}|{row.get('candidate')}"
        if row.get("valid") is True and key in keys:
            row["_latent_index"] = keys[key]
            usable.append(row)
    usable.sort(key=lambda row: (int(row.get("state_index", 0)),
                                 int(row.get("candidate_index", 0))))
    return {"rows": usable, "all_rows": rows, "horizon": horizon, "index": index}


def features(rows: list[dict[str, Any]], horizon: np.ndarray, device,
             *, latent_chunk: int = 8):
    """Materialise only spatial means, never a multi-gigabyte float32 token copy."""

    positions = np.asarray([row["_latent_index"] for row in rows], dtype=np.int64)
    latent_mean = np.empty((len(rows), HORIZONS, TOKEN_DIM), dtype=np.float32)
    for start in range(0, len(rows), latent_chunk):
        selected = np.asarray(horizon[positions[start:start + latent_chunk]],
                              dtype=np.float32)
        latent_mean[start:start + len(selected)] = selected.mean(axis=2, dtype=np.float32)
    action = np.zeros((len(rows), ACTION_DIM), dtype=np.float32)
    for index, row in enumerate(rows):
        flattened = [value for block in row["action_blocks"] for value in block]
        if len(flattened) != ACTION_DIM:
            raise CorpusValidationError("action block stopped matching frozen 40-D input")
        action[index] = np.asarray(flattened, dtype=np.float32)
    # The ledger's tensor was prospectively bound and validated against the
    # snapshot-time goal above; consume those exact bytes rather than silently
    # reimplementing the trigonometric binding at training time.
    goal = np.asarray([row["goal_binding_input"] for row in rows], dtype=np.float32)
    action_goal = np.concatenate([action, goal], axis=-1)
    targets = {
        "progress": torch.tensor([row["progress"] for row in rows], dtype=torch.float32),
        "safety": torch.tensor([row["safety"] for row in rows], dtype=torch.float32),
        "completion": torch.tensor([row["completion"] for row in rows], dtype=torch.float32),
    }
    return (torch.from_numpy(latent_mean).to(device),
            torch.from_numpy(action_goal).to(device),
            {key: value.to(device) for key, value in targets.items()})


# ------------------------------------------------------ registered training --
def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def register_initialisation(name: str, *, use_latent: bool, seed: int,
                            binding_digest: str) -> tuple[UtilityScorer, dict[str, Any]]:
    """Set the registered seed before construction and immutably register it."""

    configure_determinism(seed)
    model = UtilityScorer(use_latent=use_latent)
    initial_state = _cpu_state(model)
    initial_digest = state_dict_digest(initial_state)
    directory = PACKAGE_DIR / "registered_initialisations"
    directory.mkdir(parents=True, exist_ok=True)
    canonical = directory / f"{name}.pt"
    candidates = [canonical] + sorted(directory.glob(f"{name}_recovered_*.pt"))
    rejected: list[dict[str, str]] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (payload.get("schema") != "go2_scorer_registered_initialisation_v1"
                    or payload.get("model_name") != name
                    or payload.get("use_latent") is not use_latent
                    or payload.get("registered_seed") != seed
                    or payload.get("binding_digest") != binding_digest
                    or payload.get("initial_state_digest") != initial_digest
                    or state_dict_digest(payload["model_state_dict"]) != initial_digest):
                raise ValueError("registration metadata or state digest mismatch")
            model.load_state_dict(payload["model_state_dict"], strict=True)
            return model, {
                "path": str(path), "sha256": sha256_file(path),
                "initial_state_digest": initial_digest,
                "registered_seed": seed, "rejected_registrations": rejected,
                "recovery_decision": "reused_verified_registered_initialisation",
            }
        except Exception as exc:  # retained and recorded, never overwritten
            rejected.append({"path": str(path), "reason": str(exc)})
    payload = {
        "schema": "go2_scorer_registered_initialisation_v1", "status": STATUS,
        "model_name": name, "use_latent": use_latent, "registered_seed": seed,
        "binding_digest": binding_digest, "initial_state_digest": initial_digest,
        "model_state_dict": initial_state,
    }
    target = canonical
    if canonical.exists():
        target = directory / f"{name}_recovered_{initial_digest[:12]}_{time.time_ns()}.pt"
    atomic_torch_save(payload, target)
    return model, {
        "path": str(target), "sha256": sha256_file(target),
        "initial_state_digest": initial_digest, "registered_seed": seed,
        "rejected_registrations": rejected,
        "recovery_decision": ("registered_frozen_initialisation" if not rejected
                              else "preserved_invalid_registration_and_reregistered_same_initialisation"),
    }


def _execution_fingerprint(device: torch.device) -> dict[str, Any]:
    fingerprint: dict[str, Any] = {
        "torch": torch.__version__, "numpy": np.__version__, "device_type": device.type,
        "cuda_runtime": torch.version.cuda, "hip_runtime": getattr(torch.version, "hip", None),
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if device.type == "cuda":
        fingerprint["device_name"] = torch.cuda.get_device_name(device)
        fingerprint["device_capability"] = list(torch.cuda.get_device_capability(device))
    else:
        fingerprint["device_name"] = "cpu"
    return fingerprint


def _validate_budget(budget: Mapping[str, Any]) -> None:
    expected = {
        "epochs": 60, "batch": 64, "lr": 3e-4, "weight_decay": 0.01,
        "grad_clip": 1.0, "optimiser": "AdamW", "seed": 20260811,
    }
    for key, value in expected.items():
        if budget.get(key) != value:
            raise RuntimeError(f"frozen training budget changed at {key}")
    text = str(budget.get("budget", ""))
    if "FINAL-epoch" not in text or "no best-epoch selection" not in text:
        raise RuntimeError("frozen final-epoch/no-selection rule is absent")


def _new_optimiser(model: nn.Module, budget: Mapping[str, Any]):
    return torch.optim.AdamW(model.parameters(), lr=float(budget["lr"]),
                             weight_decay=float(budget["weight_decay"]))


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint_candidates(model_root: Path) -> list[Path]:
    candidates: list[tuple[int, int, Path]] = []
    for attempt in model_root.glob("attempt_*"):
        if not attempt.is_dir() or not attempt.name[8:].isdigit():
            continue
        attempt_number = int(attempt.name[8:])
        for path in attempt.glob("epoch_*.pt"):
            match = re.fullmatch(r"epoch_(\d{3})\.pt", path.name)
            if match:
                candidates.append((int(match.group(1)), attempt_number, path))
    candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
    return [path for _epoch, _attempt, path in candidates]


def _next_attempt(model_root: Path) -> tuple[int, Path]:
    existing = [int(path.name[8:]) for path in model_root.glob("attempt_*")
                if path.is_dir() and path.name[8:].isdigit()]
    number = max(existing, default=-1) + 1
    path = model_root / f"attempt_{number:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return number, path


def _validate_checkpoint(payload: Mapping[str, Any], *, name: str,
                         use_latent: bool, training_run_digest: str,
                         initial_state_digest: str, execution: Mapping[str, Any],
                         training_rows: int, epochs: int, path: Path) -> int:
    if payload.get("schema") != "go2_utility_scorer_epoch_checkpoint_v1":
        raise ValueError("wrong checkpoint schema")
    checks = {
        "model_name": name, "use_latent": use_latent,
        "training_run_digest": training_run_digest,
        "initial_state_digest": initial_state_digest,
        "execution_fingerprint": dict(execution), "fixed_final_epoch": epochs,
        "epoch_selection": "final_epoch_only_no_selection",
        "learning_rate_schedule": "constant",
        "training_budget_digest": canonical_digest(dict(SCORER["training"])),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint {key} mismatch")
    epoch = payload.get("completed_epoch")
    if not isinstance(epoch, int) or not 1 <= epoch <= epochs:
        raise ValueError("invalid completed_epoch")
    match = re.fullmatch(r"epoch_(\d{3})\.pt", path.name)
    if match is None or int(match.group(1)) != epoch:
        raise ValueError("filename/epoch mismatch")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("model state absent")
    if state_dict_digest(state) != payload.get("model_state_digest"):
        raise ValueError("model state digest mismatch")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("optimizer state absent")
    if structured_digest(optimizer_state) != payload.get("optimizer_state_digest"):
        raise ValueError("optimizer state digest mismatch")
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise ValueError("RNG state absent")
    if structured_digest(rng_state) != payload.get("rng_state_digest"):
        raise ValueError("RNG state digest mismatch")
    generator_state = payload.get("order_generator_state")
    order = payload.get("last_epoch_order")
    if not isinstance(generator_state, torch.Tensor) or not isinstance(order, torch.Tensor):
        raise ValueError("shuffle/order state absent")
    if tensor_digest(generator_state) != payload.get("order_generator_state_sha256"):
        raise ValueError("shuffle generator state digest mismatch")
    order_cpu = order.detach().cpu().to(torch.int64)
    if (order_cpu.numel() != training_rows
            or sorted(order_cpu.tolist()) != list(range(training_rows))):
        raise ValueError("last epoch order is not the registered-row permutation")
    if tensor_digest(order_cpu) != payload.get("last_epoch_order_sha256"):
        raise ValueError("last epoch order digest mismatch")
    return epoch


def train_registered_model(name: str, model: UtilityScorer, *, use_latent: bool,
                           latent: torch.Tensor, action_goal: torch.Tensor,
                           targets: Mapping[str, torch.Tensor], device: torch.device,
                           budget: Mapping[str, Any], training_run_digest: str,
                           initialisation: Mapping[str, Any]
                           ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Resume a verified epoch or execute the same fixed run from its init."""

    _validate_budget(budget)
    started = time.time()
    seed, epochs = int(budget["seed"]), int(budget["epochs"])
    training_rows = int(latent.shape[0])
    execution = _execution_fingerprint(device)
    model_root = PACKAGE_DIR / "training" / name
    model_root.mkdir(parents=True, exist_ok=True)
    initial_payload = torch.load(initialisation["path"], map_location="cpu",
                                 weights_only=False)
    initial_state = initial_payload["model_state_dict"]
    rejected: list[dict[str, str]] = []
    source_path: Path | None = None
    source_digest: str | None = None
    completed_epoch = 0
    optimiser = None
    order_generator = torch.Generator(device="cpu")

    for checkpoint_path in _checkpoint_candidates(model_root):
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            epoch = _validate_checkpoint(
                payload, name=name, use_latent=use_latent,
                training_run_digest=training_run_digest,
                initial_state_digest=initialisation["initial_state_digest"],
                execution=execution, training_rows=training_rows, epochs=epochs,
                path=checkpoint_path)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.to(device)
            candidate_optimiser = _new_optimiser(model, budget)
            candidate_optimiser.load_state_dict(payload["optimizer_state_dict"])
            _restore_rng_state(payload["rng_state"])
            order_generator.set_state(payload["order_generator_state"])
            optimiser = candidate_optimiser
            completed_epoch = epoch
            source_path = checkpoint_path
            source_digest = sha256_file(checkpoint_path)
            break
        except Exception as exc:
            rejected.append({"path": str(checkpoint_path), "reason": str(exc)})

    if source_path is None:
        configure_determinism(seed)
        model.load_state_dict(initial_state, strict=True)
        model.to(device)
        optimiser = _new_optimiser(model, budget)
        order_generator.manual_seed(seed)
        recovery_decision = (
            "started_from_registered_initialisation" if not rejected else
            "preserved_nonresumable_attempt_and_restarted_from_registered_initialisation"
        )
    elif completed_epoch == epochs:
        return _cpu_state(model), {
            "model_name": name, "initial_state_digest": initialisation["initial_state_digest"],
            "final_state_digest": state_dict_digest(model.state_dict()),
            "final_epoch": completed_epoch, "final_checkpoint": str(source_path),
            "final_checkpoint_sha256": source_digest,
            "recovery_decision": "reused_verified_final_epoch_checkpoint",
            "resume_source": str(source_path), "rejected_checkpoints": rejected,
            "training_wall_time_s": round(time.time() - started, 3),
            "epoch_selection": "final_epoch_only_no_selection",
        }
    else:
        recovery_decision = "resumed_from_latest_verified_epoch_checkpoint"

    assert optimiser is not None
    attempt_number, attempt_dir = _next_attempt(model_root)
    atomic_json_save({
        "schema": "go2_utility_scorer_training_attempt_v1", "status": STATUS,
        "model_name": name, "attempt": attempt_number,
        "training_run_digest": training_run_digest,
        "initial_state_digest": initialisation["initial_state_digest"],
        "fixed_final_epoch": epochs, "start_after_completed_epoch": completed_epoch,
        "resume_source": str(source_path) if source_path else None,
        "resume_source_sha256": source_digest,
        "rejected_checkpoints": rejected, "recovery_decision": recovery_decision,
        "execution_fingerprint": execution,
    }, attempt_dir / "attempt.json")

    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    last_checkpoint: Path | None = source_path
    for epoch_zero in range(completed_epoch, epochs):
        model.train()
        order = torch.randperm(training_rows, generator=order_generator)
        loss_sum = 0.0
        examples = 0
        for start in range(0, training_rows, int(budget["batch"])):
            index_cpu = order[start:start + int(budget["batch"])]
            index = index_cpu.to(device)
            progress, safety, completion = model(latent[index], action_goal[index])
            loss = (mse(progress, targets["progress"][index])
                    + bce(safety, targets["safety"][index])
                    + bce(completion, targets["completion"][index]))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(budget["grad_clip"]))
            optimiser.step()
            count = int(index.numel())
            loss_sum += float(loss.detach().cpu()) * count
            examples += count
        completed = epoch_zero + 1
        model_state = _cpu_state(model)
        optimizer_state = optimiser.state_dict()
        rng_state = _capture_rng_state()
        generator_state = order_generator.get_state()
        checkpoint = {
            "schema": "go2_utility_scorer_epoch_checkpoint_v1", "status": STATUS,
            "model_name": name, "use_latent": use_latent,
            "training_run_digest": training_run_digest,
            "initial_state_digest": initialisation["initial_state_digest"],
            "model_state_dict": model_state,
            "model_state_digest": state_dict_digest(model_state),
            "optimizer_state_dict": optimizer_state,
            "optimizer_state_digest": structured_digest(optimizer_state),
            "completed_epoch": completed, "fixed_final_epoch": epochs,
            "epoch_selection": "final_epoch_only_no_selection",
            "learning_rate_schedule": "constant",
            "training_budget_digest": canonical_digest(dict(SCORER["training"])),
            "rng_state": rng_state,
            "rng_state_digest": structured_digest(rng_state),
            "order_generator_state": generator_state,
            "order_generator_state_sha256": tensor_digest(generator_state),
            "last_epoch_order": order.cpu(),
            "last_epoch_order_sha256": tensor_digest(order.cpu().to(torch.int64)),
            "last_epoch_mean_loss": loss_sum / examples,
            "execution_fingerprint": execution,
            "resume_source": str(source_path) if source_path else None,
        }
        last_checkpoint = attempt_dir / f"epoch_{completed:03d}.pt"
        if last_checkpoint.exists():
            raise RuntimeError(f"refusing to overwrite epoch checkpoint {last_checkpoint}")
        atomic_torch_save(checkpoint, last_checkpoint)
        print(f"[{name}] completed fixed epoch {completed:02d}/{epochs} "
              f"loss={loss_sum / examples:.6f}", flush=True)

    assert last_checkpoint is not None
    final_state = _cpu_state(model)
    return final_state, {
        "model_name": name, "initial_state_digest": initialisation["initial_state_digest"],
        "final_state_digest": state_dict_digest(final_state),
        "final_epoch": epochs, "final_checkpoint": str(last_checkpoint),
        "final_checkpoint_sha256": sha256_file(last_checkpoint),
        "recovery_decision": recovery_decision,
        "resume_source": str(source_path) if source_path else None,
        "rejected_checkpoints": rejected,
        "training_wall_time_s": round(time.time() - started, 3),
        "epoch_selection": "final_epoch_only_no_selection",
    }


def _grouped_calibration(rows: list[dict[str, Any]],
                         targets: Mapping[str, torch.Tensor],
                         predicted: Mapping[str, np.ndarray],
                         group_key: str) -> dict[str, Any]:
    true = {key: value.detach().cpu().numpy().astype(np.float64)
            for key, value in targets.items()}
    values: dict[str, Any] = {}
    for group in sorted({str(row[group_key]) for row in rows}):
        indices = np.asarray([index for index, row in enumerate(rows)
                              if row[group_key] == group], dtype=np.int64)
        selected_rows = [rows[int(index)] for index in indices]
        selected_true = {key: value[indices] for key, value in true.items()}
        selected_predicted = {key: value[indices] for key, value in predicted.items()}
        values[group] = _evaluate_arrays(selected_rows, selected_true,
                                         selected_predicted)
    return values


def _paired_baseline_diagnostics(rows: list[dict[str, Any]],
                                 latent_scores: np.ndarray,
                                 baseline_scores: np.ndarray) -> dict[str, Any]:
    """State-paired descriptive differences; the frozen gate remains global."""

    true_u = np.asarray([row["utility"] for row in rows], dtype=np.float64)
    latent = composite_metrics(rows, true_u, latent_scores)
    baseline = composite_metrics(rows, true_u, baseline_scores)
    latent_by_state = {row["state_id"]: row for row in latent["per_state"]}
    baseline_by_state = {row["state_id"]: row for row in baseline["per_state"]}
    paired: list[dict[str, Any]] = []
    for state_id in sorted(latent_by_state):
        left, right = latent_by_state[state_id], baseline_by_state[state_id]
        paired.append({
            "state_id": state_id,
            "pairwise_accuracy_latent_minus_no_latent": (
                float(left["pairwise_ordering_accuracy"])
                - float(right["pairwise_ordering_accuracy"])),
            "normalised_rank_regret_no_latent_minus_latent": (
                float(right["normalised_rank_regret"])
                - float(left["normalised_rank_regret"])),
            "realised_selected_utility_latent_minus_no_latent": (
                float(left["realised_selected_utility"])
                - float(right["realised_selected_utility"])),
        })
    return {
        "states": len(paired),
        "per_state": paired,
        "mean_pairwise_accuracy_latent_minus_no_latent": float(np.mean([
            row["pairwise_accuracy_latent_minus_no_latent"] for row in paired])),
        "mean_normalised_rank_regret_no_latent_minus_latent": float(np.mean([
            row["normalised_rank_regret_no_latent_minus_latent"] for row in paired])),
        "mean_realised_selected_utility_latent_minus_no_latent": float(np.mean([
            row["realised_selected_utility_latent_minus_no_latent"] for row in paired])),
        "inferential_interval": None,
        "note": "descriptive state-paired calibration diagnostic; not a replacement gate",
    }


def _criterion(value: float, *, minimum: float | None = None,
               maximum: float | None = None) -> bool:
    if not math.isfinite(float(value)):
        return False
    return ((minimum is None or value >= minimum)
            and (maximum is None or value <= maximum))


def qualification_criteria(latent_calibration: Mapping[str, Any],
                           baseline_calibration: Mapping[str, Any],
                           fit_distribution: Mapping[str, Any],
                           calibration_distribution: Mapping[str, Any]
                           ) -> tuple[dict[str, bool], dict[str, Any], float]:
    # Subtract their human-readable decimal values so a mathematically exact
    # 0.70 - 0.65 does not fail the frozen 0.05 boundary through binary-float
    # representation alone.  This does not add a tolerance or relax the gate.
    dominance = float(
        Decimal(str(latent_calibration["composite"]["pairwise_ordering_accuracy"]))
        - Decimal(str(baseline_calibration["composite"]["pairwise_ordering_accuracy"])))
    fit_prevalence = float(fit_distribution["completion_prevalence"])
    calibration_prevalence = float(calibration_distribution["completion_prevalence"])
    completion_degenerate = (fit_prevalence in (0.0, 1.0)
                             or calibration_prevalence in (0.0, 1.0))
    observed = {
        "progress_spearman": float(latent_calibration["progress"]["spearman"]),
        "safety_auc_any_hazard": float(latent_calibration["safety"]["auc_any_hazard"]),
        "safety_calibration_error": float(latent_calibration["safety"]["calibration_error"]),
        "completion_auc": float(latent_calibration["completion"]["auc"]),
        "completion_calibration_error": float(
            latent_calibration["completion"]["calibration_error"]),
        "composite_pairwise_ordering_accuracy": float(
            latent_calibration["composite"]["pairwise_ordering_accuracy"]),
        "latent_minus_no_latent_pairwise": dominance,
        "fit_completion_prevalence": fit_prevalence,
        "calibration_completion_prevalence": calibration_prevalence,
    }
    criteria = {
        "progress_spearman_ge_0.50": _criterion(observed["progress_spearman"], minimum=0.50),
        "safety_auc_ge_0.75": _criterion(observed["safety_auc_any_hazard"], minimum=0.75),
        "safety_calibration_le_0.10": _criterion(
            observed["safety_calibration_error"], maximum=0.10),
        "completion_auc_ge_0.75": _criterion(observed["completion_auc"], minimum=0.75),
        "completion_calibration_le_0.10": _criterion(
            observed["completion_calibration_error"], maximum=0.10),
        "composite_pairwise_ge_0.65": _criterion(
            observed["composite_pairwise_ordering_accuracy"], minimum=0.65),
        "beats_no_latent_baseline_by_0.05": _criterion(dominance, minimum=0.05),
        "completion_labels_not_degenerate": not completion_degenerate,
    }
    details = {
        "observed": observed,
        "thresholds": {
            "progress_spearman_min": 0.50, "safety_auc_min": 0.75,
            "safety_ece_max": 0.10, "completion_auc_min": 0.75,
            "completion_ece_max": 0.10, "composite_pairwise_min": 0.65,
            "latent_over_no_latent_pairwise_min": 0.05,
            "completion_nondegenerate_in_each_split": True,
        },
    }
    return criteria, details, dominance


def _write_once_torch(payload: dict[str, Any], path: Path,
                      identity: Mapping[str, Any]) -> str:
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        if any(existing.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"refusing to overwrite differently bound artefact {path}")
        return sha256_file(path)
    atomic_torch_save(payload, path)
    return sha256_file(path)


def _training_storage_bytes(name: str) -> int:
    root = PACKAGE_DIR / "training" / name
    paths: list[Path] = []
    for attempt in root.glob("attempt_*"):
        if attempt.is_dir():
            paths.extend(path for path in attempt.glob("*") if path.is_file())
    return sum(path.stat().st_size for path in paths)


def _safe_json(value: Any) -> Any:
    """Represent undefined descriptive subgroup metrics as JSON null."""

    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------- main --
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=EXPECTED_POOL, choices=[EXPECTED_POOL])
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, or cuda; changing device forces a safe init restart")
    args = parser.parse_args()

    main_started = time.time()
    corpus = validate_scorer_fit_corpus(args.pool)
    operational_contract_digest = operational_scorer_contract_digest(
        corpus["bindings"])
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable")
    budget = SCORER["training"]
    _validate_budget(budget)
    fit_rows = corpus["fit_rows"]
    calibration_rows = corpus["calibration_rows"]
    fit_distribution = label_distribution(fit_rows)
    calibration_distribution = label_distribution(calibration_rows)
    grouped_distributions = {
        "fit": grouped_label_distributions(fit_rows),
        "calibration": grouped_label_distributions(calibration_rows),
    }

    feature_started = time.time()
    fit_features = features(fit_rows, corpus["horizon"], device)
    calibration_features = features(calibration_rows, corpus["horizon"], device)
    feature_wall_time = time.time() - feature_started
    binding_payload = {
        "schema": "go2_utility_scorer_training_binding_v1_2",
        "scorer_contract_v1_2_digest": operational_contract_digest,
        "corpus_bindings": corpus["bindings"],
        "normalisation": NORMALISATION,
        "architecture": {
            "tokens": TOKENS, "token_dim": TOKEN_DIM, "horizons": HORIZONS,
            "hidden": HIDDEN_DIM, "action_dim": ACTION_DIM, "goal_dim": GOAL_DIM,
            "separate_component_heads": True,
        },
        "training": dict(budget), "learning_rate_schedule": "constant",
        "final_epoch_only": True, "epoch_selection_permitted": False,
    }
    binding_digest = canonical_digest(binding_payload)
    models: dict[str, UtilityScorer] = {}
    initialisations: dict[str, dict[str, Any]] = {}
    for name, use_latent in (("latent", True), ("no_latent", False)):
        model, registration = register_initialisation(
            name, use_latent=use_latent, seed=int(budget["seed"]),
            binding_digest=binding_digest)
        models[name] = model
        initialisations[name] = registration
    training_run_digest = canonical_digest({
        "binding_digest": binding_digest,
        "initial_state_digests": {
            name: value["initial_state_digest"]
            for name, value in initialisations.items()
        },
    })

    packages: dict[str, dict[str, torch.Tensor]] = {}
    training_receipts: dict[str, dict[str, Any]] = {}
    for name, use_latent in (("latent", True), ("no_latent", False)):
        packages[name], training_receipts[name] = train_registered_model(
            name, models[name], use_latent=use_latent,
            latent=fit_features[0], action_goal=fit_features[1],
            targets=fit_features[2], device=device, budget=budget,
            training_run_digest=training_run_digest,
            initialisation=initialisations[name])
        models[name].load_state_dict(packages[name], strict=True)
        models[name].to(device)

    # A completed one-shot report is an immutable terminal result.  Validate
    # and reuse it before touching calibration tensors again; idempotent
    # recovery must not silently perform a second qualification evaluation.
    existing_report_path = PACKAGE_DIR / "qualification.json"
    current_final_state_digests = {
        name: state_dict_digest(state) for name, state in packages.items()
    }
    if existing_report_path.is_file():
        prior = _read_json(existing_report_path)
        prior_digest = prior.get("qualification_report_digest")
        prior_payload = {key: value for key, value in prior.items()
                         if key != "qualification_report_digest"}
        prior_baseline = prior.get("no_latent_baseline_package")
        prior_baseline_path = (Path(str(prior_baseline.get("path")))
                               if isinstance(prior_baseline, Mapping) else Path())
        baseline_valid = bool(
            isinstance(prior_baseline, Mapping)
            and prior_baseline_path.is_file()
            and prior_baseline.get("receipt_digest") == canonical_digest({
                key: value for key, value in prior_baseline.items()
                if key != "receipt_digest"})
            and prior_baseline.get("sha256") == sha256_file(prior_baseline_path))
        qualified_package_valid = bool(
            prior.get("qualified") is not True
            or ((PACKAGE_DIR / "scorer_package.pt").is_file()
                and prior.get("scorer_package_sha256")
                == sha256_file(PACKAGE_DIR / "scorer_package.pt")))
        if (prior.get("training_run_digest") == training_run_digest
                and prior.get("scorer_contract_v1_2_digest")
                == operational_contract_digest
                and prior.get("qualification_evaluations") == 1
                and prior.get("final_state_digests") == current_final_state_digests
                and prior_digest == canonical_digest(prior_payload)
                and isinstance(prior.get("criteria"), Mapping)
                and prior.get("qualified") is all(prior["criteria"].values())
                and baseline_valid and qualified_package_valid):
            print(json.dumps(prior, indent=2, default=str))
            return 0 if prior["qualified"] else 1

    results: dict[str, Any] = {}
    calibration_prediction_sets: dict[str, dict[str, np.ndarray]] = {}
    for name in ("latent", "no_latent"):
        fit_result, _fit_predictions = evaluate_model(
            models[name], fit_features[0], fit_features[1], fit_rows, fit_features[2])
        calibration_result, calibration_predictions = evaluate_model(
            models[name], calibration_features[0], calibration_features[1],
            calibration_rows, calibration_features[2])
        results[name] = {
            "fit": fit_result,
            "calibration": calibration_result,
            "per_family_calibration": _grouped_calibration(
                calibration_rows, calibration_features[2], calibration_predictions,
                "family"),
            "per_stratum_calibration": _grouped_calibration(
                calibration_rows, calibration_features[2], calibration_predictions,
                "stratum"),
        }
        calibration_prediction_sets[name] = calibration_predictions

    criteria, criterion_details, dominance = qualification_criteria(
        results["latent"]["calibration"], results["no_latent"]["calibration"],
        fit_distribution, calibration_distribution)
    qualified = all(criteria.values())
    paired_baseline = _paired_baseline_diagnostics(
        calibration_rows,
        calibration_prediction_sets["latent"]["utility"],
        calibration_prediction_sets["no_latent"]["utility"],
    )
    final_state_digests = current_final_state_digests
    common_artifact = {
        "schema": "go2_utility_scorer_package_v1_2", "status": STATUS,
        "training_run_digest": training_run_digest,
        "binding_digest": binding_digest,
        "contract_digest": operational_contract_digest,
        "scorer_contract_v1_2_digest": operational_contract_digest,
        "bindings": binding_payload,
        "candidate_allocator_contract_digest": allocation_contract_digest(),
        "candidate_allocation_amendment_digest": allocation_amendment_digest(),
        "candidate_allocation_post_identity_validation_digest":
            corpus["bindings"]["candidate_allocation_post_identity_validation_digest"],
        "pre_identity_allocation_validation_digest":
            corpus["bindings"]["pre_identity_allocation_validation_digest"],
        "invalid_scorer_identity_exclusion_digest":
            invalid_identity_exclusion_digest(),
        **scorer_provenance_bindings(corpus["bindings"]),
        "latent": packages["latent"], "no_latent": packages["no_latent"],
        "initial_state_digests": {
            name: value["initial_state_digest"] for name, value in initialisations.items()
        },
        "final_state_digests": final_state_digests,
        "final_epoch": int(budget["epochs"]),
        "epoch_selection": "final_epoch_only_no_selection",
        "architecture": binding_payload["architecture"],
        "spatial_aggregation": NORMALISATION["spatial_aggregation"],
        "goal_binding": "[sin(bearing_body_rad), cos(bearing_body_rad), range_m]",
        "normalisation": NORMALISATION,
        "weights": WEIGHTS,
        "target_encoder": {
            **corpus["index"]["encoder"],
            "checkpoint_sha256": corpus["bindings"]["encoder_checkpoint_sha256"],
        },
        "target_encoder_digest": corpus["index"]["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            corpus["bindings"]["encoder_checkpoint_sha256"],
        "render_contract_digest": corpus["manifest"]["render_contract_digest"],
        "textured_v03_renderer_contract_digest":
            corpus["manifest"]["textured_v03_renderer_contract_digest"],
        "preprocess_contract_digest":
            corpus["manifest"]["preprocess_contract_digest"],
        "preprocessing_digest": corpus["index"]["preprocessing_digest"],
        "preprocess": corpus["index"]["preprocess"],
        "target_normalisation": corpus["index"]["target_normalisation"],
        "qualification_criteria": criteria,
        "qualified": qualified,
    }
    identity = {
        "schema": common_artifact["schema"],
        "training_run_digest": training_run_digest,
        "qualified": qualified,
        "final_state_digests": final_state_digests,
    }
    scorer_package_digest = None
    failed_scorer_digest = None
    baseline_artifact = {
        "schema": "go2_utility_no_latent_baseline_package_v1_2",
        "status": STATUS,
        "training_run_digest": training_run_digest,
        "binding_digest": binding_digest,
        "scorer_contract_v1_2_digest": operational_contract_digest,
        **scorer_provenance_bindings(corpus["bindings"]),
        "model_state_dict": packages["no_latent"],
        "initial_state_digest": initialisations["no_latent"]["initial_state_digest"],
        "final_state_digest": final_state_digests["no_latent"],
        "final_epoch": int(budget["epochs"]),
        "epoch_selection": "final_epoch_only_no_selection",
        "architecture": {**binding_payload["architecture"], "use_latent": False},
        "goal_binding": "[sin(bearing_body_rad), cos(bearing_body_rad), range_m]",
        "normalisation": NORMALISATION,
        "weights": WEIGHTS,
        "qualified_shared_scorer": qualified,
    }
    baseline_path = (PACKAGE_DIR /
                     f"no_latent_baseline_{training_run_digest[:16]}.pt")
    baseline_package_digest = _write_once_torch(
        baseline_artifact, baseline_path, {
            "schema": baseline_artifact["schema"],
            "training_run_digest": training_run_digest,
            "final_state_digest": final_state_digests["no_latent"],
        })
    baseline_receipt = {
        "schema": "go2_utility_no_latent_baseline_receipt_v1_2",
        "status": STATUS, "complete": True,
        "training_run_digest": training_run_digest,
        "scorer_contract_v1_2_digest": operational_contract_digest,
        **scorer_provenance_bindings(corpus["bindings"]),
        "path": str(baseline_path),
        "sha256": baseline_package_digest,
        "byte_count": baseline_path.stat().st_size,
        "final_state_digest": final_state_digests["no_latent"],
        "final_epoch": int(budget["epochs"]),
        "epoch_selection": "final_epoch_only_no_selection",
    }
    baseline_receipt["receipt_digest"] = canonical_digest(baseline_receipt)
    atomic_json_save(
        baseline_receipt,
        PACKAGE_DIR / f"no_latent_baseline_{training_run_digest[:16]}.receipt.json")
    if qualified:
        scorer_package_digest = _write_once_torch(
            common_artifact, PACKAGE_DIR / "scorer_package.pt", identity)
        package_receipt = {
            "schema": "go2_utility_scorer_package_receipt_v1_2", "status": STATUS,
            "complete": True, "qualified": True,
            "training_run_digest": training_run_digest,
            "scorer_package_sha256": scorer_package_digest,
            "final_state_digests": final_state_digests,
            "bindings_digest": binding_digest,
            "scorer_contract_v1_2_digest": operational_contract_digest,
            "state_manifest_digest": corpus["bindings"]["state_manifest_digest"],
            "corpus_digest": corpus["bindings"]["corpus_digest"],
            "candidate_allocator_contract_digest": allocation_contract_digest(),
            "candidate_allocation_amendment_digest": allocation_amendment_digest(),
            "candidate_allocation_post_identity_validation_digest":
                corpus["bindings"]["candidate_allocation_post_identity_validation_digest"],
            "pre_identity_allocation_validation_digest":
                corpus["bindings"]["pre_identity_allocation_validation_digest"],
            "invalid_scorer_identity_exclusion_digest":
                invalid_identity_exclusion_digest(),
            **scorer_provenance_bindings(corpus["bindings"]),
            "target_encoder_digest": corpus["index"]["target_encoder_digest"],
            "target_encoder_checkpoint_sha256":
                corpus["bindings"]["encoder_checkpoint_sha256"],
            "render_contract_digest": corpus["manifest"]["render_contract_digest"],
            "textured_v03_renderer_contract_digest":
                corpus["manifest"]["textured_v03_renderer_contract_digest"],
            "preprocess_contract_digest":
                corpus["manifest"]["preprocess_contract_digest"],
            "preprocessing_digest": corpus["index"]["preprocessing_digest"],
            "preprocess": corpus["index"]["preprocess"],
            "target_normalisation": corpus["index"]["target_normalisation"],
        }
        package_receipt["scorer_package_receipt_digest"] = canonical_digest(
            package_receipt)
        atomic_json_save(package_receipt, PACKAGE_DIR / "scorer_package_receipt.json")
    else:
        failed_path = PACKAGE_DIR / f"failed_scorer_{training_run_digest[:16]}.pt"
        failed_scorer_digest = _write_once_torch(common_artifact, failed_path, identity)

    report = {
        "schema": "go2_utility_scorer_v1_2_qualification", "status": STATUS,
        "training_run_digest": training_run_digest,
        "scorer_contract_v1_2_digest": operational_contract_digest,
        "frozen_scorer_fit_allocation_design_digest":
            FROZEN_SCORER_FIT_ALLOCATION_DESIGN_DIGEST,
        "corpus_bindings": corpus["bindings"],
        "candidate_allocator_contract_digest": allocation_contract_digest(),
        "candidate_allocation_amendment_digest": allocation_amendment_digest(),
        "candidate_allocation_post_identity_validation_digest":
            corpus["bindings"]["candidate_allocation_post_identity_validation_digest"],
        "pre_identity_allocation_validation_digest":
            corpus["bindings"]["pre_identity_allocation_validation_digest"],
        "invalid_scorer_identity_exclusion_digest":
            invalid_identity_exclusion_digest(),
        **scorer_provenance_bindings(corpus["bindings"]),
        "target_encoder_digest": corpus["index"]["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            corpus["bindings"]["encoder_checkpoint_sha256"],
        "render_contract_digest": corpus["manifest"]["render_contract_digest"],
        "textured_v03_renderer_contract_digest":
            corpus["manifest"]["textured_v03_renderer_contract_digest"],
        "preprocess_contract_digest":
            corpus["manifest"]["preprocess_contract_digest"],
        "preprocessing_digest": corpus["index"]["preprocessing_digest"],
        "preprocess": corpus["index"]["preprocess"],
        "target_normalisation": corpus["index"]["target_normalisation"],
        "fit_states": EXPECTED_FIT_STATES,
        "calibration_states": EXPECTED_CALIBRATION_STATES,
        "fit_rows": EXPECTED_FIT_ROWS, "calibration_rows": EXPECTED_CALIBRATION_ROWS,
        "scene_disjoint": True,
        "label_distributions": grouped_distributions,
        "completion_prevalence_by_split_and_family": completion_by_split_family(
            fit_rows, calibration_rows),
        "latent_scorer": results["latent"],
        "no_latent_baseline": results["no_latent"],
        "baseline_dominance_pairwise": dominance,
        "paired_latent_vs_no_latent_calibration": paired_baseline,
        "no_latent_baseline_package": baseline_receipt,
        "criterion_details": criterion_details,
        "criteria": criteria, "qualified": qualified,
        "qualification_evaluations": 1,
        "qualification_input": "scene-disjoint true H=1..4 target latent trajectories",
        "epoch_selection_permitted": False,
        "initialisations": initialisations,
        "training_receipts": training_receipts,
        "final_state_digests": final_state_digests,
        "scorer_package_sha256": scorer_package_digest,
        "failed_scorer_sha256": failed_scorer_digest,
        "runtime": {
            "corpus_validation_s": corpus["validation_wall_time_s"],
            "feature_materialisation_s": round(feature_wall_time, 3),
            "latent_training_s": training_receipts["latent"]["training_wall_time_s"],
            "no_latent_training_s": training_receipts["no_latent"]["training_wall_time_s"],
            "total_wall_time_s": round(time.time() - main_started, 3),
        },
        "storage": {
            "scorer_fit_latent_shards_bytes": corpus["index"].get("storage_bytes"),
            "latent_training_attempts_bytes": _training_storage_bytes("latent"),
            "no_latent_training_attempts_bytes": _training_storage_bytes("no_latent"),
            "registered_initialisations_bytes": sum(
                Path(value["path"]).stat().st_size for value in initialisations.values()),
            "scorer_package_bytes": (
                (PACKAGE_DIR / "scorer_package.pt").stat().st_size
                if scorer_package_digest is not None else None),
            "no_latent_baseline_package_bytes": baseline_path.stat().st_size,
        },
        "predictor_checkpoints_loaded": 0,
    }
    report = _safe_json(report)
    report["qualification_report_digest"] = canonical_digest(report)
    report_path = PACKAGE_DIR / "qualification.json"
    if report_path.exists():
        # Any valid terminal report for this run returned before calibration.
        # Reaching this point means the canonical path is stale, incomplete, or
        # differently bound. Preserve its exact bytes before restoring the
        # canonical one-shot path for the current registered run.
        invalid_dir = PACKAGE_DIR / "invalid_attempts"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        prior_sha = sha256_file(report_path)
        preserved = invalid_dir / (
            f"qualification.{prior_sha[:16]}.{time.time_ns()}.invalid.json")
        os.replace(report_path, preserved)
        _fsync_directory(invalid_dir)
    atomic_json_save(report, report_path)
    print(json.dumps(report, indent=2, default=str))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
