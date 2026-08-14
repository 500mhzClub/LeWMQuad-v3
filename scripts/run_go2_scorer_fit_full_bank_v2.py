#!/usr/bin/env python3
"""Issue and execute the prospective full-bank scorer-fit corpus V2.

This runner is deliberately narrow.  It never invokes a MILP, CP-SAT model,
candidate-subset allocator, or performance benchmark.  Its public stages are:

* ``issue-design``: issue the rotation-mask classification, then the design;
* ``issue-source-correction``: bind the final preselection structural repair;
* ``freeze-manifests``: deterministically freeze the five pre-outcome files;
* ``issue-scorer-contract``: issue the successor contract before a branch;
* ``run``: execute the registered smoke/recovery/corpus/training pipeline; and
* ``status``: assemble a read-only metadata report.

Importing this module opens no generated artifact and starts no simulator,
encoder, trainer, predictor, or solver.  Heavy-runtime validation is delegated
to closed JSON subprocess stages in the already frozen Genesis and ROCm
interpreters.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import build_go2_branch_corpus_v1_2 as BUILDER  # noqa: E402
from lewm.oracle import (  # noqa: E402
    go2_scorer_fit_corpus_v2_design as DESIGN,
)
from lewm.oracle import (  # noqa: E402
    go2_scorer_fit_corpus_v2_scorer_contract as SCORER_CONTRACT,
)
from lewm.oracle import (  # noqa: E402
    go2_small_completion_global_execution_amendment_v1 as RUNTIME_AUTHORITY,
)


STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
RUNNER_RELATIVE_PATH = Path("scripts/run_go2_scorer_fit_full_bank_v2.py")
SCORER_FIT_RELATIVE_PATH = Path(
    ".generated/go2_branch_corpus_v1_2/scorer_fit")
UTILITY_V2_RELATIVE_PATH = Path(
    ".generated/go2_utility_scorer_fit_corpus_v2")
FEASIBILITY_FAILURE_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    BUILDER.SCORER_FIT_V2_FEASIBILITY_FAILURE_NAME)

FEASIBILITY_FAILURE_SCHEMA = (
    "go2_scorer_fit_corpus_v2_full_bank_preoutcome_feasibility_failure_v1")
FEASIBILITY_FAILURE_STATUS = (
    "FAIL_PRE_OUTCOME_FULL_BANK_FOUR_FIT_ONE_CALIBRATION_UNAVAILABLE")
FEASIBILITY_FAILURE_SELF_KEY = "preoutcome_feasibility_failure_digest"

RUNTIME_PROBE_SCHEMA = "go2_scorer_fit_corpus_v2_runtime_probe_v1"
RUNTIME_PROBE_STATUS = "PASS_EXACT_FROZEN_RUNTIME"
RUNTIME_PROBE_SELF_KEY = "runtime_probe_digest"
RUN_REPORT_SCHEMA = "go2_scorer_fit_corpus_v2_orchestrator_report_v1"

_RUNTIME_ROLES = ("genesis", "rocm")
_MANIFEST_SPECS = (
    ("selection", BUILDER.SCORER_FIT_V2_SELECTION_NAME,
     "full_bank_small_completion_selection_digest"),
    ("revalidation", BUILDER.SCORER_FIT_V2_REVALIDATION_NAME,
     "full_bank_preoutcome_state_revalidation_digest"),
    ("small_shard", BUILDER.SCORER_FIT_V2_SMALL_SHARD_NAME,
     "state_shard_digest"),
    ("assignment_manifest", BUILDER.SCORER_FIT_V2_ASSIGNMENT_MANIFEST_NAME,
     "full_bank_assignment_manifest_digest"),
    # Install the complete state manifest last.  Its presence is the terminal
    # marker for the resumable five-file pre-outcome transaction.
    ("state_manifest", BUILDER.SCORER_FIT_V2_STATE_MANIFEST_NAME,
     "state_manifest_digest"),
)

_V2_RUNTIME_STAGE_ROLES = {
    "branch_smoke": "genesis",
    "branch_smoke_zero_new": "genesis",
    "smoke_encoding": "rocm",
    "smoke_encoding_zero_new": "rocm",
    "smoke_single_shard_regeneration": "rocm",
    "full_branch_corpus": "genesis",
    "full_latent_encoding": "rocm",
    "scorer_training_and_qualification": "rocm",
    "development_transfer": "rocm",
}


class FullBankV2RunnerError(RuntimeError):
    """An exact-path, stage-order, runtime, or terminal gate failed."""


CommandRunner = Callable[[Sequence[str], Path], int]
RuntimeProbeInvoker = Callable[[str, Path, Path, Any], Mapping[str, Any]]
ValidationInvoker = Callable[[str, Path, Path], Mapping[str, Any]]
DeleteRegisteredShard = Callable[[Mapping[str, Any], Path], None]


class _DesignAuthority(Protocol):
    DESIGN_SELF_KEY: str
    MASK_CLASSIFICATION_SELF_KEY: str
    SOURCE_CORRECTION_SCHEMA: str
    SOURCE_CORRECTION_SELF_KEY: str
    IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST: str
    IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST: str

    def issue_rotation_mask_classification(
            self, *, root: Path) -> Mapping[str, Any]: ...

    def issue_design_amendment(
            self, *, root: Path) -> Mapping[str, Any]: ...

    def issue_preselection_source_correction(
            self, *, root: Path) -> Mapping[str, Any]: ...


def _require_final_source_correction_authority(
        authority: Mapping[str, Any], *, design: Any) -> dict[str, Any]:
    """Reject both historical repairs at every selection-entry boundary."""

    if not isinstance(authority, Mapping):
        raise FullBankV2RunnerError("active V2 design authority is malformed")
    correction = authority.get("source_correction")
    if (not isinstance(correction, Mapping)
            or correction.get("schema") != design.SOURCE_CORRECTION_SCHEMA
            or correction.get("structural_validation_correction_version") != 1
            or correction.get(
                "immutable_preselection_source_correction_v2_digest")
            != design.IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST
            or correction.get(
                "transitive_immutable_preselection_source_correction_v1_digest")
            != design.IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST
            or authority.get("source_correction_digest")
            != correction.get(design.SOURCE_CORRECTION_SELF_KEY)
            or authority.get("candidate_outcomes_consumed") is not False):
        raise FullBankV2RunnerError(
            "final preselection structural-validation correction is required")
    return dict(correction)


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True, "ensure_ascii": True, "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + ("\n" if pretty else "")).encode(
        "utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _without(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _signed(value: Mapping[str, Any], self_key: str) -> dict[str, Any]:
    payload = dict(value)
    if self_key in payload:
        raise FullBankV2RunnerError("self-digest key already exists")
    payload[self_key] = canonical_digest(payload)
    return payload


def _is_hex(value: Any, length: int = 64) -> bool:
    return bool(isinstance(value, str) and len(value) == length
                and all(character in "0123456789abcdef" for character in value))


def _forbidden_component(value: str) -> bool:
    return (value == "sealed" or value == "sealed_test.json"
            or value.startswith("sealed_"))


def _pin_relative(root: Path, relative: str | Path, *, label: str) -> Path:
    """Pin one named path; permit only the repository's managed aliases."""

    repository = Path(root).resolve(strict=True)
    rel = Path(relative)
    if (rel.is_absolute() or not rel.parts or any(
            part in {"", ".", ".."} or _forbidden_component(part)
            for part in rel.parts)):
        raise FullBankV2RunnerError(f"{label} is not a safe relative path")
    managed = []
    for generated_root in DESIGN.MANAGED_GENERATED_ROOTS:
        try:
            rel.relative_to(generated_root)
            managed.append(generated_root)
        except ValueError:
            pass
    if managed:
        if len(managed) != 1:
            raise FullBankV2RunnerError(
                f"{label} matches multiple managed generated roots")
        try:
            return DESIGN._pin_generated(repository, rel, label=label)
        except Exception as exc:
            raise FullBankV2RunnerError(
                f"{label} managed generated path is invalid") from exc
    cursor = repository
    for component in rel.parts[:-1]:
        cursor /= component
        if cursor.exists() and cursor.is_symlink():
            raise FullBankV2RunnerError(f"{label} parent is symlinked")
    return repository / rel


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise FullBankV2RunnerError(f"{label} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullBankV2RunnerError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise FullBankV2RunnerError(f"{label} is not a JSON object")
    return value, raw


def _install_or_require_exact_json(
        path: Path, payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Install one read-only JSON object without overwrite; replay is exact."""

    expected = dict(payload)
    encoded = _json_bytes(expected, pretty=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise FullBankV2RunnerError(f"{label} existing path is not regular")
        if path.read_bytes() != encoded or json.loads(encoded) != expected:
            raise FullBankV2RunnerError(f"{label} differs from deterministic replay")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise FullBankV2RunnerError(f"{label} is not read-only")
        return expected
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise FullBankV2RunnerError(f"{label} parent is unavailable")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except OSError as exc:
        raise FullBankV2RunnerError(f"cannot exclusively create {label}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as sink:
            descriptor = -1
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        os.chmod(path, 0o444, follow_symlinks=False)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    reopened, raw = _load_json(path, label=label)
    if (reopened != expected or raw != encoded
            or stat.S_IMODE(path.stat().st_mode) & 0o222):
        raise FullBankV2RunnerError(f"{label} durable reopen changed")
    return reopened


def issue_design(*, root: Path = ROOT, design: Any = DESIGN) -> dict[str, Any]:
    """Issue exactly the source classification followed by the V2 design."""

    classification = design.issue_rotation_mask_classification(root=root)
    amendment = design.issue_design_amendment(root=root)
    if (not isinstance(classification, Mapping)
            or not isinstance(amendment, Mapping)
            or amendment.get("rotation_mask_classification", {}).get(
                "self_digest")
            != classification.get(design.MASK_CLASSIFICATION_SELF_KEY)):
        raise FullBankV2RunnerError(
            "issued design does not bind the exact mask classification")
    return {
        "stage": "issue-design",
        "status": "PASS_CLASSIFICATION_THEN_DESIGN_ISSUED",
        "rotation_mask_classification_digest": classification[
            design.MASK_CLASSIFICATION_SELF_KEY],
        "scorer_fit_corpus_v2_design_digest": amendment[
            design.DESIGN_SELF_KEY],
        "classification": "ALL_OLD_ROTATION_CONDITIONS_"
                          "PARTIAL_SUBSET_ALLOCATION_ONLY",
        "old_rotation_related_condition_count": classification["counts"][
            "old_rotation_related_condition_count"],
        "true_branch_execution_requirement_count": classification["counts"][
            "true_branch_execution_requirement_count"],
        "candidate_outcomes_consumed": False,
        "solver_or_optimisation_used": False,
    }


def issue_source_correction(
        *, root: Path = ROOT, design: Any = DESIGN) -> dict[str, Any]:
    """Issue the final source-only bridge from immutable design to selection."""

    correction = design.issue_preselection_source_correction(root=root)
    active = design.load_active_design_authority(root=root)
    active_correction = _require_final_source_correction_authority(
        active, design=design)
    if (not isinstance(correction, Mapping)
            or active_correction != dict(correction)):
        raise FullBankV2RunnerError(
            "issued preselection source correction changed on active replay")
    return {
        "stage": "issue-source-correction",
        "status": (
            "PASS_PRESELECTION_STRUCTURAL_VALIDATION_CORRECTION_V1_ISSUED"),
        "scorer_fit_corpus_v2_design_digest": active["design_amendment"][
            design.DESIGN_SELF_KEY],
        "immutable_preselection_source_correction_v2_digest": correction[
            "immutable_preselection_source_correction_v2_digest"],
        "transitive_immutable_preselection_source_correction_v1_digest":
            correction[
                "transitive_immutable_preselection_source_correction_v1_digest"],
        "scorer_fit_corpus_v2_source_correction_digest": correction[
            design.SOURCE_CORRECTION_SELF_KEY],
        "candidate_outcomes_consumed": False,
        "selection_started": False,
        "solver_or_optimisation_used": False,
    }


def _manifest_paths(root: Path) -> dict[str, Path]:
    return {
        key: _pin_relative(
            root, SCORER_FIT_RELATIVE_PATH / name,
            label=f"full-bank V2 {key}")
        for key, name, _self_key in _MANIFEST_SPECS
    }


def _build_feasibility_failure(
        failure: BUILDER.FullBankV2FeasibilityFailure,
        *, authority: Mapping[str, Any]) -> dict[str, Any]:
    design = authority.get("design_amendment")
    classification = authority.get("rotation_mask_classification")
    if not isinstance(design, Mapping) or not isinstance(classification, Mapping):
        raise FullBankV2RunnerError("full-bank V2 design authority is incomplete")
    ordered = list(failure.ordered_scene_ids)
    if (len(ordered) != 17 or len(set(ordered)) != 17
            or failure.fit_count < 0 or failure.calibration_count < 0):
        raise FullBankV2RunnerError(
            "pre-outcome feasibility failure details are malformed")
    return _signed({
        "schema": FEASIBILITY_FAILURE_SCHEMA,
        "status": FEASIBILITY_FAILURE_STATUS,
        "complete": True,
        "source_repository_commit": authority[
            "active_source_repository_commit"],
        "scorer_fit_corpus_v2_design_digest": design[DESIGN.DESIGN_SELF_KEY],
        "scorer_fit_corpus_v2_source_correction_digest": authority[
            "source_correction_digest"],
        "rotation_mask_classification_digest": classification[
            DESIGN.MASK_CLASSIFICATION_SELF_KEY],
        "active_global_exact_amendment_digest":
            DESIGN.ACTIVE_GLOBAL_AMENDMENT_DIGEST,
        "global_exact_model_digest": DESIGN.GLOBAL_EXACT_MODEL_DIGEST,
        "exact_six_of_twelve_infeasibility_digest":
            DESIGN.EXACT_INFEASIBILITY_DIGEST,
        "terminal_six_of_twelve_infeasibility_receipt_digest":
            DESIGN.TERMINAL_RECEIPT_DIGEST,
        "failure_reason": failure.reason,
        "passing_fit_scene_count": failure.fit_count,
        "passing_calibration_scene_count": failure.calibration_count,
        "required_fit_scene_count": 4,
        "required_calibration_scene_count": 1,
        "ordered_eligible_scene_count": 17,
        "ordered_scene_ids": ordered,
        "old_rotation_condition_classification": {
            "partial_subset_allocation_only": 18,
            "true_branch_execution_requirement": 0,
        },
        "selected_state_manifest_issued": False,
        "assignment_manifest_issued": False,
        "branch_execution_started": False,
        "candidate_outcomes_consumed": False,
        "frames_or_latents_generated": False,
        "scorer_training_started": False,
        "predictor_checkpoint_opened": False,
        "final_200_state_corpus_generated": False,
        "milp_cp_sat_or_optimisation_used": False,
        "six_of_twelve_model_retried_or_reinterpreted": False,
        "nothing_running": True,
    }, FEASIBILITY_FAILURE_SELF_KEY)


def _validate_feasibility_failure(
        value: Mapping[str, Any], *, authority: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullBankV2RunnerError("feasibility failure is not a mapping")
    payload = dict(value)
    if (payload.get("schema") != FEASIBILITY_FAILURE_SCHEMA
            or payload.get("status") != FEASIBILITY_FAILURE_STATUS
            or payload.get("complete") is not True
            or payload.get(FEASIBILITY_FAILURE_SELF_KEY)
            != canonical_digest(_without(payload, FEASIBILITY_FAILURE_SELF_KEY))
            or payload.get("scorer_fit_corpus_v2_design_digest")
            != authority["design_amendment"][DESIGN.DESIGN_SELF_KEY]
            or payload.get("source_repository_commit")
            != authority["active_source_repository_commit"]
            or payload.get("scorer_fit_corpus_v2_source_correction_digest")
            != authority["source_correction_digest"]
            or payload.get("rotation_mask_classification_digest")
            != authority["rotation_mask_classification"][
                DESIGN.MASK_CLASSIFICATION_SELF_KEY]
            or payload.get("ordered_eligible_scene_count") != 17
            or len(payload.get("ordered_scene_ids", [])) != 17
            or len(set(payload.get("ordered_scene_ids", []))) != 17
            or payload.get("required_fit_scene_count") != 4
            or payload.get("required_calibration_scene_count") != 1
            or payload.get("candidate_outcomes_consumed") is not False
            or payload.get("branch_execution_started") is not False
            or payload.get("milp_cp_sat_or_optimisation_used") is not False
            or payload.get("nothing_running") is not True):
        raise FullBankV2RunnerError(
            "immutable full-bank V2 feasibility failure changed")
    return payload


def freeze_manifests(
        *, root: Path = ROOT, builder: Any = BUILDER,
        design_authority: Any = DESIGN,
        ) -> tuple[int, dict[str, Any]]:
    """Freeze five deterministic pre-outcome artifacts or one exact failure."""

    authority = design_authority.load_active_design_authority(root=root)
    _require_final_source_correction_authority(
        authority, design=design_authority)
    paths = _manifest_paths(root)
    failure_path = _pin_relative(
        root, FEASIBILITY_FAILURE_RELATIVE_PATH,
        label="full-bank V2 pre-outcome feasibility failure")
    if failure_path.exists() or failure_path.is_symlink():
        if any(path.exists() or path.is_symlink() for path in paths.values()):
            raise FullBankV2RunnerError(
                "terminal feasibility failure conflicts with success manifests")
        payload, _raw = _load_json(
            failure_path, label="full-bank V2 pre-outcome feasibility failure")
        terminal = _validate_feasibility_failure(payload, authority=authority)
        return 2, {
            "stage": "freeze-manifests", "status": terminal["status"],
            "terminal_failure_digest": terminal[FEASIBILITY_FAILURE_SELF_KEY],
            "candidate_outcomes_consumed": False, "nothing_running": True,
        }

    absence_before = design_authority.audit_v2_runtime_outputs_absent(
        root=root, phase="successor_contract")
    loaded = builder.load_scorer_fit_v2_preoutcome_inputs(
        out=root / SCORER_FIT_RELATIVE_PATH)
    if (loaded.get("design_authority") != authority
            or loaded.get("candidate_outcomes_consumed") is not False
            or loaded.get("solver_or_optimisation_used") is not False):
        raise FullBankV2RunnerError(
            "full-bank V2 pre-outcome producer boundary changed")
    try:
        bundle = builder.build_scorer_fit_v2_full_bank_bundle(
            design=authority["design_amendment"],
            classification=authority["rotation_mask_classification"],
            source_correction=authority["source_correction"],
            source_correction_binding=authority[
                "source_correction_binding"],
            source_correction_digest=authority["source_correction_digest"],
            predecessor_inputs=loaded["predecessor_inputs"],
            allowed_scene_ids_by_family=loaded[
                "allowed_scene_ids_by_family"],
            exclusion_authority=loaded["exclusion_authority"],
            preserved_vectors=loaded["preserved_vectors"],
            exclusion_binding=loaded["exclusion_authority"][
                "predecessor_exclusion_binding"],
            verify_scene_files=True,
        )
    except builder.FullBankV2FeasibilityFailure as failure:
        if any(path.exists() or path.is_symlink() for path in paths.values()):
            raise FullBankV2RunnerError(
                "cannot issue failure after any success artifact exists") from failure
        terminal = _build_feasibility_failure(failure, authority=authority)
        if (absence_before
                != design_authority.audit_v2_runtime_outputs_absent(
                    root=root, phase="successor_contract")
                or authority
                != design_authority.load_active_design_authority(root=root)):
            raise FullBankV2RunnerError(
                "authority or runtime absence changed before failure install")
        _install_or_require_exact_json(
            failure_path, terminal,
            label="full-bank V2 pre-outcome feasibility failure")
        return 2, {
            "stage": "freeze-manifests", "status": terminal["status"],
            "terminal_failure_digest": terminal[FEASIBILITY_FAILURE_SELF_KEY],
            "passing_fit_scene_count": terminal["passing_fit_scene_count"],
            "passing_calibration_scene_count": terminal[
                "passing_calibration_scene_count"],
            "candidate_outcomes_consumed": False, "nothing_running": True,
        }

    builder.validate_scorer_fit_v2_full_bank_bundle(
        bundle,
        predecessor_inputs=loaded["predecessor_inputs"],
        allowed_scene_ids_by_family=loaded["allowed_scene_ids_by_family"],
        exclusion_authority=loaded["exclusion_authority"],
        preserved_vectors=loaded["preserved_vectors"],
        exclusion_binding=loaded["exclusion_authority"][
            "predecessor_exclusion_binding"],
        verify_scene_files=True,
    )
    if (bundle.get("candidate_outcomes_consumed") is not False
            or bundle.get("solver_or_optimisation_used") is not False
            or failure_path.exists() or failure_path.is_symlink()
            or absence_before
            != design_authority.audit_v2_runtime_outputs_absent(
                root=root, phase="successor_contract")
            or authority
            != design_authority.load_active_design_authority(root=root)):
        raise FullBankV2RunnerError(
            "full-bank authority changed before manifest installation")

    bindings: dict[str, dict[str, Any]] = {}
    for key, _name, self_key in _MANIFEST_SPECS:
        payload = bundle.get(key)
        if (not isinstance(payload, Mapping)
                or not _is_hex(payload.get(self_key))):
            raise FullBankV2RunnerError(
                f"full-bank V2 {key} producer payload is malformed")
        installed = _install_or_require_exact_json(
            paths[key], payload, label=f"full-bank V2 {key}")
        bindings[key] = {
            "path": str(SCORER_FIT_RELATIVE_PATH / dict(
                (row_key, name) for row_key, name, _ in _MANIFEST_SPECS)[key]),
            "self_digest_key": self_key,
            "self_digest": installed[self_key],
            "raw_sha256": file_sha256(paths[key]),
            "byte_count": paths[key].stat().st_size,
        }

    replay = builder.load_and_validate_full_bank_v2_manifests_for_consumption(
        out=root / SCORER_FIT_RELATIVE_PATH)
    if any(replay.get(key) != bundle[key] for key, _name, _self in _MANIFEST_SPECS):
        raise FullBankV2RunnerError(
            "installed full-bank V2 manifests differ from producer replay")
    state_manifest = replay["state_manifest"]
    return 0, {
        "stage": "freeze-manifests",
        "status": "PASS_FULL_BANK_V2_MANIFESTS_FROZEN",
        "scorer_fit_corpus_v2_source_correction_digest": authority[
            "source_correction_digest"],
        "selected_small_completion_scene_ids": list(
            replay["selection"]["selected_scene_ids"]),
        "state_count": len(state_manifest["states"]),
        "assignment_count": replay["assignment_manifest"]["assignment_count"],
        "state_manifest_digest": state_manifest["state_manifest_digest"],
        "assignment_manifest_digest": replay["assignment_manifest"][
            "full_bank_assignment_manifest_digest"],
        "artifact_bindings": bindings,
        "candidate_outcomes_consumed": False,
        "solver_or_optimisation_used": False,
    }


def issue_scorer_contract(
        *, root: Path = ROOT, contract_authority: Any = SCORER_CONTRACT,
        ) -> dict[str, Any]:
    artifact = contract_authority.issue_contract(root=root)
    validated = contract_authority.load_contract_for_consumption(root=root)
    if artifact != validated:
        raise FullBankV2RunnerError(
            "issued successor scorer contract changed on exact replay")
    return {
        "stage": "issue-scorer-contract",
        "status": "PASS_SUCCESSOR_SCORER_CONTRACT_ISSUED",
        "scorer_fit_corpus_v2_scorer_contract_digest": artifact[
            contract_authority.CONTRACT_SELF_KEY],
        "contract_artifact_digest": artifact[
            contract_authority.ARTIFACT_SELF_KEY],
        "branch_execution_started": False,
        "candidate_outcomes_consumed": False,
    }


def _runtime_contract(authority: Any, role: str) -> dict[str, Any]:
    contracts = getattr(authority, "DOWNSTREAM_RUNTIME_CONTRACTS", None)
    if (not isinstance(contracts, Mapping)
            or set(contracts) != set(_RUNTIME_ROLES)
            or role not in contracts
            or not isinstance(contracts[role], Mapping)):
        raise FullBankV2RunnerError("downstream runtime contract surface changed")
    return json.loads(_json_bytes(contracts[role]))


def _runtime_observation_from_contract(
        contract: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "python_version", "torch_version", "torch_cuda_runtime",
        "torch_hip_runtime", "accelerator_available",
        "accelerator_device_count", "accelerator_devices",
    }
    if contract.get("role") == "genesis_branch_generation":
        keys.add("genesis_version")
    if not keys.issubset(contract):
        raise FullBankV2RunnerError("runtime observation contract is incomplete")
    return {key: json.loads(_json_bytes(contract[key])) for key in sorted(keys)}


def _bound_interpreters(
        *, root: Path, authority: Any = RUNTIME_AUTHORITY) -> dict[str, Path]:
    roles = getattr(authority, "DOWNSTREAM_STAGE_RUNTIME_ROLES", None)
    expected_projection = {
        "six_branch_smoke": "genesis",
        "smoke_encoding": "rocm",
        "full_720_branch_corpus": "genesis",
        "full_latent_encoding": "rocm",
        "scorer_training_and_qualification": "rocm",
        "development_transfer": "rocm",
        "qualification_validation": "rocm",
        "development_validation": "rocm",
    }
    if roles != expected_projection:
        raise FullBankV2RunnerError(
            "frozen downstream stage/runtime routing changed")
    result: dict[str, Path] = {}
    for role in _RUNTIME_ROLES:
        contract = _runtime_contract(authority, role)
        interpreter = _pin_relative(
            root, contract["interpreter_relative_path"],
            label=f"bound {role} interpreter")
        config = _pin_relative(
            root, contract["pyvenv_config_relative_path"],
            label=f"bound {role} pyvenv config")
        if (not interpreter.exists() or interpreter.is_dir()
                or not os.access(interpreter, os.X_OK)
                or not config.is_file() or config.is_symlink()
                or config.stat().st_size != contract["pyvenv_config_byte_count"]
                or file_sha256(config) != contract["pyvenv_config_sha256"]):
            raise FullBankV2RunnerError(
                f"bound {role} runtime custody changed")
        result[role] = interpreter
    return result


def build_runtime_probe_receipt(
        *, runtime_role: str, observation: Mapping[str, Any], authority: Any,
        ) -> dict[str, Any]:
    contract = _runtime_contract(authority, runtime_role)
    if dict(observation) != _runtime_observation_from_contract(contract):
        raise FullBankV2RunnerError(
            f"{runtime_role} runtime identity differs from frozen contract")
    return _signed({
        "schema": RUNTIME_PROBE_SCHEMA,
        "status": RUNTIME_PROBE_STATUS,
        "runtime_role": runtime_role,
        "runtime_contract_digest": canonical_digest(contract),
        "interpreter_relative_path": contract["interpreter_relative_path"],
        "pyvenv_config_sha256": contract["pyvenv_config_sha256"],
        "observed_runtime_identity": dict(observation),
    }, RUNTIME_PROBE_SELF_KEY)


def validate_runtime_probe_receipt(
        receipt: Mapping[str, Any], *, runtime_role: str, authority: Any,
        ) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise FullBankV2RunnerError("runtime probe is not a mapping")
    payload = dict(receipt)
    expected = build_runtime_probe_receipt(
        runtime_role=runtime_role,
        observation=payload.get("observed_runtime_identity", {}),
        authority=authority)
    if payload != expected:
        raise FullBankV2RunnerError("runtime probe receipt changed")
    return expected


def _observe_current_runtime(runtime_role: str) -> dict[str, Any]:
    import torch

    devices: list[dict[str, Any]] = []
    count = int(torch.cuda.device_count())
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        gcn = getattr(properties, "gcnArchName", None)
        if gcn is None:
            gcn = getattr(properties, "gcn_arch_name", None)
        devices.append({
            "index": index,
            "name": str(torch.cuda.get_device_name(index)),
            "capability": list(torch.cuda.get_device_capability(index)),
            "gcn_arch_name": gcn,
            "multi_processor_count": int(properties.multi_processor_count),
        })
    observation: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "torch_hip_runtime": torch.version.hip,
        "accelerator_available": bool(torch.cuda.is_available()),
        "accelerator_device_count": count,
        "accelerator_devices": devices,
    }
    if runtime_role == "genesis":
        observation["genesis_version"] = str(metadata.version("genesis-world"))
    elif runtime_role != "rocm":
        raise FullBankV2RunnerError("unknown runtime probe role")
    return observation


def _emit_runtime_probe(
        runtime_role: str, *, root: Path = ROOT,
        authority: Any = RUNTIME_AUTHORITY) -> int:
    contract = _runtime_contract(authority, runtime_role)
    expected = _pin_relative(
        root, contract["interpreter_relative_path"],
        label=f"bound {runtime_role} interpreter")
    config = _pin_relative(
        root, contract["pyvenv_config_relative_path"],
        label=f"bound {runtime_role} pyvenv config")
    if (Path(sys.executable).absolute() != expected.absolute()
            or not config.is_file() or config.is_symlink()
            or config.stat().st_size != contract["pyvenv_config_byte_count"]
            or file_sha256(config) != contract["pyvenv_config_sha256"]):
        raise FullBankV2RunnerError(
            f"runtime probe is not using the bound {runtime_role} interpreter")
    print(json.dumps(build_runtime_probe_receipt(
        runtime_role=runtime_role,
        observation=_observe_current_runtime(runtime_role),
        authority=authority), sort_keys=True), flush=True)
    return 0


def _default_runtime_probe_invoker(
        runtime_role: str, root: Path, interpreter: Path,
        authority: Any) -> Mapping[str, Any]:
    completed = subprocess.run(
        [str(interpreter), str(root / RUNNER_RELATIVE_PATH),
         "--stage", f"internal-probe-{runtime_role}"],
        cwd=root, check=False, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        raise FullBankV2RunnerError(
            f"bound {runtime_role} runtime probe failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FullBankV2RunnerError(
            f"bound {runtime_role} runtime probe output is not closed JSON") from exc
    return validate_runtime_probe_receipt(
        payload, runtime_role=runtime_role, authority=authority)


def downstream_command_sequence(
        *, root: Path = ROOT,
        interpreters: Mapping[str, Path] | None = None,
        authority: Any = RUNTIME_AUTHORITY,
        ) -> dict[str, list[str]]:
    """Return the exact V2 commands, with no solver or final-eval route."""

    selected = (_bound_interpreters(root=root, authority=authority)
                if interpreters is None else {
                    role: Path(interpreters[role]) for role in _RUNTIME_ROLES})
    if set(selected) != set(_RUNTIME_ROLES):
        raise FullBankV2RunnerError("downstream interpreter set changed")
    build = str(root / "scripts/build_go2_branch_corpus_v1_2.py")
    encode = str(root / "scripts/encode_go2_branch_corpus_v1_2.py")
    train = str(root / "scripts/train_go2_utility_scorer_v1_2.py")
    apply = str(root /
                "scripts/apply_go2_utility_scorer_to_counterfactual_"
                "development_v1_2.py")
    commands = {
        "branch_smoke": [str(selected["genesis"]), build, "--pool",
                         "scorer_fit", "--stage", "smoke", "--backend", "cpu"],
        "branch_smoke_zero_new": [str(selected["genesis"]), build, "--pool",
                                  "scorer_fit", "--stage", "smoke",
                                  "--backend", "cpu"],
        "smoke_encoding": [str(selected["rocm"]), encode, "--pool",
                           "scorer_fit", "--corpus-design", "full-bank-v2",
                           "--smoke"],
        "smoke_encoding_zero_new": [str(selected["rocm"]), encode, "--pool",
                                    "scorer_fit", "--corpus-design",
                                    "full-bank-v2", "--smoke"],
        "smoke_single_shard_regeneration": [
            str(selected["rocm"]), encode, "--pool", "scorer_fit",
            "--corpus-design", "full-bank-v2", "--smoke"],
        "full_branch_corpus": [str(selected["genesis"]), build, "--pool",
                               "scorer_fit", "--stage", "branches",
                               "--backend", "cpu"],
        "full_latent_encoding": [str(selected["rocm"]), encode, "--pool",
                                 "scorer_fit", "--corpus-design",
                                 "full-bank-v2"],
        "scorer_training_and_qualification": [
            str(selected["rocm"]), train, "--pool", "scorer_fit",
            "--corpus-design", "full-bank-v2"],
        "development_transfer": [str(selected["rocm"]), apply,
                                 "--scorer-corpus-design", "full-bank-v2"],
    }
    if set(commands) != set(_V2_RUNTIME_STAGE_ROLES):
        raise FullBankV2RunnerError("full-bank V2 command surface changed")
    for stage, command in commands.items():
        if Path(command[0]) != selected[_V2_RUNTIME_STAGE_ROLES[stage]]:
            raise FullBankV2RunnerError(f"{stage} escaped its bound runtime")
        lowered = " ".join(command).lower()
        if ("final_eval" in lowered or "final-eval" in lowered
                or "milp" in lowered or "cp-sat" in lowered
                or "small-completion-search" in lowered):
            raise FullBankV2RunnerError(
                f"forbidden execution route appeared in {stage}")
    return commands


def _default_command_runner(command: Sequence[str], root: Path) -> int:
    return int(subprocess.run(
        [str(part) for part in command], cwd=root, check=False).returncode)


def _ensure_final_eval_absent(*, root: Path = ROOT) -> None:
    for relative in DESIGN.V2_ALWAYS_ABSENT_PATHS:
        path = _pin_relative(root, relative, label="future final-evaluation absence")
        if path.exists() or path.is_symlink():
            raise FullBankV2RunnerError(
                "final 200-state evaluation corpus is not authorised in this pass")


def _run_command(
        stage: str, commands: Mapping[str, Sequence[str]], *, root: Path,
        command_runner: CommandRunner) -> int:
    result = command_runner(commands[stage], root)
    if type(result) is not int:
        raise FullBankV2RunnerError(f"{stage} returned a non-integer status")
    _ensure_final_eval_absent(root=root)
    return result


def _require_projection(
        value: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullBankV2RunnerError(f"{kind} validator did not return a mapping")
    projection = dict(value)
    if (projection.get("validation_kind") != kind
            or projection.get("pass") is not True
            or projection.get("candidate_outcomes_used_for_selection") is not False
            or projection.get("final_200_state_corpus_generated") is not False):
        raise FullBankV2RunnerError(f"{kind} validation projection changed")
    return projection


def _validate_zero_new_encoding(
        first: Mapping[str, Any], replay: Mapping[str, Any]) -> None:
    if (first.get("registered_smoke_shard_inventory_digest")
            != replay.get("registered_smoke_shard_inventory_digest")
            or first.get("registered_smoke_stable_artifact_inventory_digest")
            != replay.get(
                "registered_smoke_stable_artifact_inventory_digest")
            or replay.get("invocation_new_context_shards") != 0
            or replay.get("invocation_new_horizon_shards") != 0
            or replay.get("zero_new_resume_verified") is not True):
        raise FullBankV2RunnerError(
            "zero-new smoke encoding changed a completed artifact")


def _validate_single_shard_recovery(
        before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    target_before = before.get("single_shard_regeneration_target")
    target_after = after.get("single_shard_regeneration_target")
    if (not isinstance(target_before, Mapping)
            or target_after != target_before
            or before.get("registered_smoke_shard_inventory_digest")
            != after.get("registered_smoke_shard_inventory_digest")
            or before.get(
                "registered_smoke_stable_artifact_inventory_digest")
            != after.get(
                "registered_smoke_stable_artifact_inventory_digest")
            or after.get("invocation_new_context_shards") != 0
            or after.get("invocation_new_horizon_shards") != 1
            or after.get("single_registered_shard_regenerated") is not True
            or after.get("only_registered_missing_shard_changed") is not True):
        raise FullBankV2RunnerError(
            "single-shard smoke regeneration proof failed")


def _training_stop_report(
        terminal: Mapping[str, Any], *, completed: Sequence[str],
        runtime_probe_digests: Mapping[str, str]) -> tuple[int, dict[str, Any]]:
    terminal_kind = terminal.get("terminal_kind")
    if terminal_kind == "COMPLETION_DEGENERACY_FAILURE":
        status = "STOP_FROZEN_COMPLETION_DEGENERACY_FAILURE"
    elif terminal_kind == "QUALIFICATION_FAILURE":
        status = "STOP_FROZEN_SCORER_QUALIFICATION_FAILURE"
    else:
        raise FullBankV2RunnerError("requested stop is not a failure terminal")
    return 2, {
        "schema": RUN_REPORT_SCHEMA,
        "status": status,
        "completed_stages": list(completed),
        "qualified": False,
        "development_transfer_started": False,
        "predictor_access_before_qualification": False,
        "final_200_state_corpus_generated": False,
        "runtime_probe_digests": dict(runtime_probe_digests),
        "terminal_digest": terminal["terminal_digest"],
        "nothing_running": True,
    }


def _development_complete_report(
        *, terminal: Mapping[str, Any], development: Mapping[str, Any],
        completed: Sequence[str], runtime_probe_digests: Mapping[str, str],
        ) -> tuple[int, dict[str, Any]]:
    if (development.get("qualified_scorer_bound") is not True
            or development.get("development_state_count") != 20
            or development.get("development_branch_count") != 240):
        raise FullBankV2RunnerError("development transfer terminal changed")
    return 0, {
        "schema": RUN_REPORT_SCHEMA,
        "status": "COMPLETE_AUTHORISED_EXPLORATORY_DEVELOPMENT_TRANSFER",
        "completed_stages": list(completed),
        "qualified": True,
        "qualification_report_digest": terminal["terminal_digest"],
        "development_transfer_result_digest": development["terminal_digest"],
        "predictor_access_before_qualification": False,
        "final_200_state_corpus_generated": False,
        "runtime_probe_digests": dict(runtime_probe_digests),
        "nothing_running": True,
    }


def _delete_registered_smoke_shard(
        target: Mapping[str, Any], root: Path) -> None:
    expected_keys = {"path", "sha256", "byte_count", "shape"}
    if (not isinstance(target, Mapping) or set(target) != expected_keys
            or not _is_hex(target.get("sha256"))
            or isinstance(target.get("byte_count"), bool)
            or not isinstance(target.get("byte_count"), int)
            or target["byte_count"] <= 0
            or target.get("shape") != [4, 768, 1024]):
        raise FullBankV2RunnerError(
            "registered smoke regeneration target is malformed")
    relative = Path(str(target["path"]))
    expected_parent = SCORER_FIT_RELATIVE_PATH / "latents_v2/horizon"
    if relative.parent != expected_parent:
        raise FullBankV2RunnerError(
            "registered smoke regeneration target escaped horizon shards")
    path = _pin_relative(root, relative, label="registered smoke latent shard")
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size != target["byte_count"]
            or file_sha256(path) != target["sha256"]):
        raise FullBankV2RunnerError(
            "registered smoke regeneration target bytes changed")
    path.unlink()
    directory = os.open(
        path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if path.exists() or path.is_symlink():
        raise FullBankV2RunnerError("registered smoke shard deletion was not durable")


def _run_smoke_protocol(
        *, root: Path, commands: Mapping[str, Sequence[str]],
        interpreters: Mapping[str, Path], command_runner: CommandRunner,
        validation_invoker: ValidationInvoker,
        delete_registered_shard: DeleteRegisteredShard,
        ) -> list[str]:
    """Run the one authorised smoke and its two durability checks exactly once."""

    completed: list[str] = []
    if _run_command("branch_smoke", commands, root=root,
                    command_runner=command_runner) != 0:
        raise FullBankV2RunnerError("twelve-branch smoke failed")
    first_branch = _require_projection(
        validation_invoker("branch-smoke", root, interpreters["genesis"]),
        kind="branch-smoke")
    if (first_branch.get("branch_count") != 12
            or first_branch.get("candidate_indices") != list(range(12))
            or first_branch.get("rendered_horizon_frame_count") != 48):
        raise FullBankV2RunnerError("full-bank branch smoke cardinality changed")
    completed.append("branch_smoke")

    if _run_command("branch_smoke_zero_new", commands, root=root,
                    command_runner=command_runner) != 0:
        raise FullBankV2RunnerError("zero-new branch smoke failed")
    replay_branch = _require_projection(
        validation_invoker("branch-smoke", root, interpreters["genesis"]),
        kind="branch-smoke")
    if (first_branch.get("registered_smoke_artifact_inventory_digest")
            != replay_branch.get("registered_smoke_artifact_inventory_digest")):
        raise FullBankV2RunnerError(
            "zero-new branch replay changed a completed artifact")
    completed.append("branch_smoke_zero_new")

    if _run_command("smoke_encoding", commands, root=root,
                    command_runner=command_runner) != 0:
        raise FullBankV2RunnerError("full-bank smoke encoding failed")
    first_encoding = _require_projection(
        validation_invoker("encoding-smoke", root, interpreters["rocm"]),
        kind="encoding-smoke")
    if (first_encoding.get("horizon_latent_count") != 12
            or first_encoding.get("horizon_shape") != [4, 768, 1024]):
        raise FullBankV2RunnerError("full-bank smoke latent shape/count changed")
    completed.append("smoke_encoding")

    if _run_command("smoke_encoding_zero_new", commands, root=root,
                    command_runner=command_runner) != 0:
        raise FullBankV2RunnerError("zero-new smoke encoding failed")
    replay_encoding = _require_projection(
        validation_invoker("encoding-smoke", root, interpreters["rocm"]),
        kind="encoding-smoke")
    _validate_zero_new_encoding(first_encoding, replay_encoding)
    completed.append("smoke_encoding_zero_new")

    target = replay_encoding.get("single_shard_regeneration_target")
    if not isinstance(target, Mapping):
        raise FullBankV2RunnerError("smoke regeneration target is absent")
    delete_registered_shard(target, root)
    if _run_command("smoke_single_shard_regeneration", commands, root=root,
                    command_runner=command_runner) != 0:
        raise FullBankV2RunnerError("single-shard smoke regeneration failed")
    recovered_encoding = _require_projection(
        validation_invoker("encoding-smoke", root, interpreters["rocm"]),
        kind="encoding-smoke")
    _validate_single_shard_recovery(replay_encoding, recovered_encoding)
    completed.append("smoke_single_shard_regeneration")
    return completed


def run_pipeline(
        *, root: Path = ROOT, command_runner: CommandRunner,
        runtime_probe_invoker: RuntimeProbeInvoker,
        validation_invoker: ValidationInvoker,
        delete_registered_shard: DeleteRegisteredShard,
        authority: Any = RUNTIME_AUTHORITY,
        resume: bool = False,
        ) -> tuple[int, dict[str, Any]]:
    """Execute the fail-closed post-contract sequence using injected effects."""

    SCORER_CONTRACT.load_contract_for_consumption(root=root)
    BUILDER.load_and_validate_full_bank_v2_manifests_for_consumption(
        out=root / SCORER_FIT_RELATIVE_PATH)
    if not resume:
        DESIGN.audit_v2_runtime_outputs_absent(
            root=root, phase="post_contract_pre_branch")
    _ensure_final_eval_absent(root=root)

    interpreters = _bound_interpreters(root=root, authority=authority)
    probes = {
        role: validate_runtime_probe_receipt(
            runtime_probe_invoker(role, root, interpreters[role], authority),
            runtime_role=role, authority=authority)
        for role in _RUNTIME_ROLES
    }
    probe_digests = {
        role: probes[role][RUNTIME_PROBE_SELF_KEY] for role in _RUNTIME_ROLES}
    commands = downstream_command_sequence(
        root=root, interpreters=interpreters, authority=authority)
    completed: list[str] = []
    smoke_protocol_complete = False

    if resume:
        # A frozen scorer terminal dominates every upstream resumable stage.
        # Its producer validator replays the complete corpus/encoder lineage,
        # so opening it here neither skips validation nor accesses a predictor.
        retained_terminal = _require_projection(
            validation_invoker(
                "training-terminal-optional", root, interpreters["rocm"]),
            kind="training-terminal-optional")
        if retained_terminal.get("terminal_present") is True:
            completed.append("retained_existing_scorer_training_terminal")
            if retained_terminal.get("terminal_kind") in {
                    "COMPLETION_DEGENERACY_FAILURE",
                    "QUALIFICATION_FAILURE"}:
                if retained_terminal.get("qualified") is not False:
                    raise FullBankV2RunnerError(
                        "retained failure terminal has a passing verdict")
                return _training_stop_report(
                    retained_terminal, completed=completed,
                    runtime_probe_digests=probe_digests)
            if (retained_terminal.get("terminal_kind") != "QUALIFICATION_PASS"
                    or retained_terminal.get("qualified") is not True):
                raise FullBankV2RunnerError(
                    "retained training terminal has no exact verdict")
            retained_development = _require_projection(
                validation_invoker(
                    "development-terminal-optional", root,
                    interpreters["rocm"]),
                kind="development-terminal-optional")
            if retained_development.get("terminal_present") is True:
                completed.append(
                    "retained_existing_development_transfer_terminal")
                return _development_complete_report(
                    terminal=retained_terminal,
                    development=retained_development,
                    completed=completed, runtime_probe_digests=probe_digests)
            if retained_development.get("terminal_present") is not False:
                raise FullBankV2RunnerError(
                    "optional development-terminal presence verdict is missing")
            if _run_command(
                    "development_transfer", commands, root=root,
                    command_runner=command_runner) != 0:
                raise FullBankV2RunnerError("development transfer failed")
            development = _require_projection(
                validation_invoker(
                    "development-terminal", root, interpreters["rocm"]),
                kind="development-terminal")
            completed.append("development_transfer")
            return _development_complete_report(
                terminal=retained_terminal, development=development,
                completed=completed, runtime_probe_digests=probe_digests)
        if retained_terminal.get("terminal_present") is not False:
            raise FullBankV2RunnerError(
                "optional training-terminal presence verdict is missing")

        retained_smoke = _require_projection(
            validation_invoker(
                "encoding-smoke-optional", root, interpreters["rocm"]),
            kind="encoding-smoke-optional")
        if retained_smoke.get("terminal_present") is True:
            smoke_protocol_complete = (
                retained_smoke.get("smoke_protocol_complete") is True
                and retained_smoke.get("zero_new_resume_verified") is True
                and retained_smoke.get(
                    "single_registered_shard_regenerated") is True)
            if not smoke_protocol_complete:
                # The receipt proves that the base smoke ran.  Run the smoke
                # encoder once before strict shard replay: this repairs the
                # exact deliberate-deletion window without deleting a second
                # valid shard.  A repaired ordinary invalid shard is followed
                # by the still-required zero-new replay and designated
                # candidate-0 regeneration proof.
                if _run_command(
                        "smoke_single_shard_regeneration", commands,
                        root=root, command_runner=command_runner) != 0:
                    raise FullBankV2RunnerError(
                        "interrupted smoke recovery invocation failed")
                resumed_encoding = _require_projection(
                    validation_invoker(
                        "encoding-smoke", root, interpreters["rocm"]),
                    kind="encoding-smoke")
                if (resumed_encoding.get("horizon_latent_count") != 12
                        or resumed_encoding.get("horizon_shape")
                        != [4, 768, 1024]):
                    raise FullBankV2RunnerError(
                        "resumed smoke latent shape/count changed")
                if (resumed_encoding.get(
                        "single_registered_shard_regenerated") is True
                        and resumed_encoding.get(
                            "only_registered_missing_shard_changed") is True
                        and resumed_encoding.get(
                            "zero_new_resume_verified") is True):
                    completed.append(
                        "resumed_interrupted_smoke_shard_regeneration")
                    smoke_protocol_complete = True
                else:
                    replay_encoding = resumed_encoding
                    if (resumed_encoding.get(
                            "invocation_new_context_shards") != 0
                            or resumed_encoding.get(
                                "invocation_new_horizon_shards") != 0
                            or resumed_encoding.get(
                                "zero_new_resume_verified") is not True):
                        if _run_command(
                                "smoke_encoding_zero_new", commands,
                                root=root,
                                command_runner=command_runner) != 0:
                            raise FullBankV2RunnerError(
                                "resumed zero-new smoke encoding failed")
                        replay_encoding = _require_projection(
                            validation_invoker(
                                "encoding-smoke", root,
                                interpreters["rocm"]),
                            kind="encoding-smoke")
                        _validate_zero_new_encoding(
                            resumed_encoding, replay_encoding)
                        completed.append("resumed_smoke_encoding_zero_new")
                    target = replay_encoding.get(
                        "single_shard_regeneration_target")
                    if not isinstance(target, Mapping):
                        raise FullBankV2RunnerError(
                            "resumed smoke regeneration target is absent")
                    delete_registered_shard(target, root)
                    if _run_command(
                            "smoke_single_shard_regeneration", commands,
                            root=root, command_runner=command_runner) != 0:
                        raise FullBankV2RunnerError(
                            "resumed single-shard regeneration failed")
                    recovered_encoding = _require_projection(
                        validation_invoker(
                            "encoding-smoke", root,
                            interpreters["rocm"]),
                        kind="encoding-smoke")
                    _validate_single_shard_recovery(
                        replay_encoding, recovered_encoding)
                    completed.append(
                        "resumed_smoke_single_shard_regeneration")
                    smoke_protocol_complete = True
        elif retained_smoke.get("terminal_present") is not False:
            raise FullBankV2RunnerError(
                "optional encoding-smoke presence verdict is missing")

    if smoke_protocol_complete:
        completed.append("retained_completed_smoke_protocol")
    else:
        completed.extend(_run_smoke_protocol(
            root=root, commands=commands, interpreters=interpreters,
            command_runner=command_runner,
            validation_invoker=validation_invoker,
            delete_registered_shard=delete_registered_shard))

    if _run_command("full_branch_corpus", commands, root=root,
                    command_runner=command_runner) != 0:
        raise FullBankV2RunnerError("1,440-branch corpus generation failed")
    branch_corpus = _require_projection(
        validation_invoker("branch-corpus", root, interpreters["genesis"]),
        kind="branch-corpus")
    if (branch_corpus.get("state_count") != 120
            or branch_corpus.get("branch_count") != 1_440):
        raise FullBankV2RunnerError("full branch corpus cardinality changed")
    completed.append("full_branch_corpus")

    if _run_command("full_latent_encoding", commands, root=root,
                    command_runner=command_runner) != 0:
        raise FullBankV2RunnerError("full target-latent encoding failed")
    encoded = _require_projection(
        validation_invoker("encoded-corpus", root, interpreters["rocm"]),
        kind="encoded-corpus")
    if (encoded.get("state_count") != 120
            or encoded.get("horizon_latent_count") != 1_440):
        raise FullBankV2RunnerError("encoded corpus cardinality changed")
    completed.append("full_latent_encoding")

    # Reuse an exact immutable terminal before invoking the trainer.  This is
    # stricter than relying on the trainer's own reuse path, which necessarily
    # materialises the corpus and features before it reaches that check.
    prior_terminal = _require_projection(
        validation_invoker(
            "training-terminal-optional", root, interpreters["rocm"]),
        kind="training-terminal-optional")
    if prior_terminal.get("terminal_present") is True:
        terminal = prior_terminal
        completed.append("retained_existing_scorer_training_terminal")
        if terminal.get("terminal_kind") in {
                "COMPLETION_DEGENERACY_FAILURE", "QUALIFICATION_FAILURE"}:
            if terminal.get("qualified") is not False:
                raise FullBankV2RunnerError(
                    "retained failure terminal has a passing verdict")
            return _training_stop_report(
                terminal, completed=completed,
                runtime_probe_digests=probe_digests)
        if (terminal.get("terminal_kind") != "QUALIFICATION_PASS"
                or terminal.get("qualified") is not True):
            raise FullBankV2RunnerError(
                "retained training terminal has no exact verdict")
    elif prior_terminal.get("terminal_present") is False:
        training_return = _run_command(
            "scorer_training_and_qualification", commands, root=root,
            command_runner=command_runner)
        terminal = _require_projection(
            validation_invoker(
                "training-terminal", root, interpreters["rocm"]),
            kind="training-terminal")
        terminal_kind = terminal.get("terminal_kind")
        completed.append("scorer_training_and_qualification")
        if terminal_kind in {
                "COMPLETION_DEGENERACY_FAILURE", "QUALIFICATION_FAILURE"}:
            if training_return != 1 or terminal.get("qualified") is not False:
                raise FullBankV2RunnerError(
                    "training failure did not produce its exact terminal status")
            return _training_stop_report(
                terminal, completed=completed,
                runtime_probe_digests=probe_digests)
        if (terminal_kind != "QUALIFICATION_PASS" or training_return != 0
                or terminal.get("qualified") is not True):
            raise FullBankV2RunnerError(
                "scorer training produced no exact frozen terminal verdict")
    else:
        raise FullBankV2RunnerError(
            "optional training-terminal presence verdict is missing")

    # This is the first point at which any predictor package may be opened.
    prior_development = _require_projection(
        validation_invoker(
            "development-terminal-optional", root, interpreters["rocm"]),
        kind="development-terminal-optional")
    if prior_development.get("terminal_present") is True:
        completed.append("retained_existing_development_transfer_terminal")
        return _development_complete_report(
            terminal=terminal, development=prior_development,
            completed=completed, runtime_probe_digests=probe_digests)
    if prior_development.get("terminal_present") is not False:
        raise FullBankV2RunnerError(
            "optional development-terminal presence verdict is missing")
    if _run_command("development_transfer", commands, root=root,
                    command_runner=command_runner) != 0:
        raise FullBankV2RunnerError("development transfer failed")
    development = _require_projection(
        validation_invoker("development-terminal", root, interpreters["rocm"]),
        kind="development-terminal")
    completed.append("development_transfer")
    return _development_complete_report(
        terminal=terminal, development=development, completed=completed,
        runtime_probe_digests=probe_digests)


def _default_validation_invoker(
        validation_kind: str, root: Path, interpreter: Path) -> Mapping[str, Any]:
    command = [
        str(interpreter), str(root / RUNNER_RELATIVE_PATH), "--stage",
        f"internal-validate-{validation_kind}",
    ]
    completed = subprocess.run(
        command, cwd=root, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise FullBankV2RunnerError(
            f"bound {validation_kind} validator failed")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FullBankV2RunnerError(
            f"bound {validation_kind} validator output is not closed JSON") from exc
    if not isinstance(value, Mapping):
        raise FullBankV2RunnerError(
            f"bound {validation_kind} validator returned no object")
    return dict(value)


def run_authorised(
        *, root: Path = ROOT, resume: bool = False,
        command_runner: CommandRunner = _default_command_runner,
        runtime_probe_invoker: RuntimeProbeInvoker =
            _default_runtime_probe_invoker,
        validation_invoker: ValidationInvoker = _default_validation_invoker,
        delete_registered_shard: DeleteRegisteredShard =
            _delete_registered_smoke_shard,
        authority: Any = RUNTIME_AUTHORITY,
        ) -> tuple[int, dict[str, Any]]:
    return run_pipeline(
        root=root, command_runner=command_runner,
        runtime_probe_invoker=runtime_probe_invoker,
        validation_invoker=validation_invoker,
        delete_registered_shard=delete_registered_shard,
        authority=authority, resume=resume)


def _binding_if_present(
        root: Path, relative: Path, *, self_key: str | None = None,
        ) -> dict[str, Any] | None:
    path = _pin_relative(root, relative, label=f"status {relative.name}")
    if not path.exists() and not path.is_symlink():
        return None
    if not path.is_file() or path.is_symlink():
        raise FullBankV2RunnerError(f"status path is not regular: {relative}")
    row: dict[str, Any] = {
        "path": str(relative), "raw_sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }
    if self_key is not None:
        payload, _raw = _load_json(path, label=f"status {relative.name}")
        digest = payload.get(self_key)
        if not _is_hex(digest):
            raise FullBankV2RunnerError(
                f"status artifact lacks {self_key}: {relative}")
        row["self_digest_key"] = self_key
        row["self_digest"] = digest
    return row


def assemble_status_report(*, root: Path = ROOT) -> dict[str, Any]:
    """Return metadata only; never read a row, frame, latent, weight or shard."""

    artifacts: dict[str, Any] = {
        "rotation_mask_classification": _binding_if_present(
            root, DESIGN.MASK_CLASSIFICATION_RELATIVE_PATH,
            self_key=DESIGN.MASK_CLASSIFICATION_SELF_KEY),
        "design_amendment": _binding_if_present(
            root, DESIGN.DESIGN_RELATIVE_PATH, self_key=DESIGN.DESIGN_SELF_KEY),
        "preselection_source_correction": _binding_if_present(
            root, DESIGN.SOURCE_CORRECTION_RELATIVE_PATH,
            self_key=DESIGN.SOURCE_CORRECTION_SELF_KEY),
        "feasibility_failure": _binding_if_present(
            root, FEASIBILITY_FAILURE_RELATIVE_PATH,
            self_key=FEASIBILITY_FAILURE_SELF_KEY),
        "successor_scorer_contract": _binding_if_present(
            root, SCORER_CONTRACT.ARTIFACT_RELATIVE_PATH,
            self_key=SCORER_CONTRACT.ARTIFACT_SELF_KEY),
        "branch_smoke": _binding_if_present(
            root, SCORER_FIT_RELATIVE_PATH /
            BUILDER.SCORER_FIT_V2_BRANCH_SMOKE_RECEIPT_NAME,
            self_key="smoke_branch_receipt_digest"),
        "encoding_smoke": _binding_if_present(
            root, SCORER_FIT_RELATIVE_PATH /
            BUILDER.SCORER_FIT_V2_ENCODING_SMOKE_RECEIPT_NAME,
            self_key="smoke_receipt_digest"),
        "corpus_receipt": _binding_if_present(
            root, SCORER_FIT_RELATIVE_PATH /
            BUILDER.SCORER_FIT_V2_CORPUS_RECEIPT_NAME),
        "qualification": _binding_if_present(
            root, UTILITY_V2_RELATIVE_PATH / "qualification_v2.json",
            self_key="qualification_report_digest"),
        "development_transfer": _binding_if_present(
            root, UTILITY_V2_RELATIVE_PATH /
            "counterfactual_development_transfer_v2/result_v2.json",
            self_key="development_transfer_result_digest"),
    }
    for key, name, self_key in _MANIFEST_SPECS:
        artifacts[key] = _binding_if_present(
            root, SCORER_FIT_RELATIVE_PATH / name, self_key=self_key)
    final_eval = _pin_relative(
        root, DESIGN.V2_ALWAYS_ABSENT_PATHS[0],
        label="status future final evaluation")
    return {
        "schema": "go2_scorer_fit_corpus_v2_read_only_status_v1",
        "status": STATUS,
        "artifacts": artifacts,
        "final_200_state_corpus_absent": not (
            final_eval.exists() or final_eval.is_symlink()),
        "scientific_payload_rows_read": 0,
        "frame_latent_weight_or_predictor_shards_read": 0,
        "state_changed": False,
    }


# Heavy-runtime validation emitters are defined below the pure orchestrator so
# importing this module never imports the encoder, trainer, or development
# consumer.  Each emitter produces a deliberately small closed projection.
def _emit_validation(validation_kind: str, *, root: Path = ROOT) -> int:
    if validation_kind in {"branch-smoke", "branch-corpus"}:
        projection = _branch_validation_projection(
            validation_kind, root=root)
    elif validation_kind in {"encoding-smoke", "encoded-corpus"}:
        projection = _encoding_validation_projection(
            validation_kind, root=root)
    elif validation_kind == "encoding-smoke-optional":
        projection = _optional_encoding_smoke_projection(root=root)
    elif validation_kind == "training-terminal":
        projection = _training_terminal_projection(root=root)
    elif validation_kind == "training-terminal-optional":
        projection = _optional_training_terminal_projection(root=root)
    elif validation_kind == "development-terminal":
        projection = _development_terminal_projection(root=root)
    elif validation_kind == "development-terminal-optional":
        projection = _optional_development_terminal_projection(root=root)
    else:
        raise FullBankV2RunnerError("unknown internal validation kind")
    print(json.dumps(projection, sort_keys=True), flush=True)
    return 0


def _registered_file_binding(path: Path, *, root: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FullBankV2RunnerError(f"registered file is unavailable: {path}")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise FullBankV2RunnerError("registered file escaped repository") from exc
    return {
        "path": str(relative), "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _branch_validation_projection(
        validation_kind: str, *, root: Path) -> dict[str, Any]:
    out = root / SCORER_FIT_RELATIVE_PATH
    full = validation_kind == "branch-corpus"
    value = BUILDER.load_and_validate_full_bank_v2_branch_outputs_for_consumption(
        out=out, allow_partial=not full)
    rows = value["rows"]
    smoke = value.get("branch_smoke")
    expected_count = 1_440 if full else 12
    if (len(rows) != expected_count or not isinstance(smoke, Mapping)
            or smoke.get("pass") is not True):
        raise FullBankV2RunnerError(
            f"{validation_kind} exact branch output count changed")
    if not full:
        smoke_state = str(smoke["state_id"])
        rows = [row for row in rows if str(row["state_id"]) == smoke_state]
        if len(rows) != 12:
            raise FullBankV2RunnerError("branch smoke does not contain twelve rows")
    registered: list[dict[str, Any]] = []
    for row in rows:
        row_path = out / BUILDER.SCORER_FIT_V2_ROW_RECORDS_NAME / (
            f"{row['branch_identity_digest']}.json")
        registered.append(_registered_file_binding(row_path, root=root))
        for frame in [*row["context_frames"], *row["horizon_frames"]]:
            frame_path = out / str(frame["path"])
            binding = _registered_file_binding(frame_path, root=root)
            if (binding["sha256"] != frame["sha256"]
                    or binding["byte_count"] != frame["byte_count"]):
                raise FullBankV2RunnerError("registered frame binding changed")
            registered.append(binding)
    for path in (
            out / BUILDER.SCORER_FIT_V2_BRANCH_ROWS_NAME,
            out / BUILDER.SCORER_FIT_V2_CORPUS_RECEIPT_NAME,
            out / BUILDER.SCORER_FIT_V2_BRANCH_SMOKE_RECEIPT_NAME):
        registered.append(_registered_file_binding(path, root=root))
    # Context frames are intentionally shared by twelve branches.  Deduplicate
    # by exact path before hashing the zero-new inventory.
    unique = {row["path"]: row for row in registered}
    ordered = [unique[key] for key in sorted(unique)]
    return {
        "validation_kind": validation_kind,
        "pass": True,
        "state_count": 120 if full else 1,
        "branch_count": expected_count,
        "candidate_indices": list(range(12)),
        "rendered_horizon_frame_count": 48 if not full else 5_760,
        "registered_smoke_artifact_inventory_digest": canonical_digest(ordered),
        "candidate_outcomes_used_for_selection": False,
        "final_200_state_corpus_generated": False,
    }


def _encoding_validation_projection(
        validation_kind: str, *, root: Path) -> dict[str, Any]:
    from scripts import encode_go2_branch_corpus_v1_2 as encoder

    out = root / SCORER_FIT_RELATIVE_PATH
    if validation_kind == "encoding-smoke":
        value = encoder.load_and_validate_full_bank_v2_encoding_smoke_for_consumption(
            out=out, require_protocol_complete=False)
        return _normalise_encoding_projection(value, kind=validation_kind)
    value = encoder.load_and_validate_full_bank_v2_encoded_corpus_for_consumption(
        out=out)
    return _normalise_encoding_projection(value, kind=validation_kind)


def _optional_encoding_smoke_projection(*, root: Path) -> dict[str, Any]:
    path = root / SCORER_FIT_RELATIVE_PATH / (
        BUILDER.SCORER_FIT_V2_ENCODING_SMOKE_RECEIPT_NAME)
    if not path.exists() and not path.is_symlink():
        return {
            "validation_kind": "encoding-smoke-optional", "pass": True,
            "terminal_present": False,
            "candidate_outcomes_used_for_selection": False,
            "final_200_state_corpus_generated": False,
        }
    # This is intentionally receipt-only.  A resume may arrive in the narrow
    # window after the registered candidate-0 shard was durably removed and
    # before its regeneration, or after an unrelated full-index shard became
    # invalid.  The strict producer replay would reject both before the
    # encoder got its authorised chance to repair missing/invalid shards.
    # Exact shard validation remains mandatory immediately after recovery and
    # again for the complete corpus.
    smoke, _raw = _load_json(path, label="optional V2 encoding smoke receipt")
    manifests = BUILDER.load_and_validate_full_bank_v2_manifests_for_consumption(
        out=root / SCORER_FIT_RELATIVE_PATH)
    contract = SCORER_CONTRACT.load_contract_for_consumption(root=root)
    branch_smoke_path = root / SCORER_FIT_RELATIVE_PATH / (
        BUILDER.SCORER_FIT_V2_BRANCH_SMOKE_RECEIPT_NAME)
    branch_smoke, _branch_raw = _load_json(
        branch_smoke_path, label="optional V2 branch smoke receipt")
    state_manifest = manifests.get("state_manifest")
    assignment_manifest = manifests.get("assignment_manifest")
    if (not isinstance(state_manifest, Mapping)
            or not isinstance(assignment_manifest, Mapping)
            or not isinstance(contract, Mapping)
            or branch_smoke.get("schema")
            != BUILDER.SCORER_FIT_V2_BRANCH_SMOKE_SCHEMA
            or branch_smoke.get("status") != STATUS
            or branch_smoke.get("state_manifest_digest")
            != state_manifest.get("state_manifest_digest")
            or branch_smoke.get("full_bank_assignment_manifest_digest")
            != assignment_manifest.get(
                "full_bank_assignment_manifest_digest")
            or branch_smoke.get(
                "scorer_fit_corpus_v2_scorer_contract_digest")
            != contract.get(SCORER_CONTRACT.CONTRACT_SELF_KEY)
            or branch_smoke.get(
                "scorer_fit_corpus_v2_scorer_contract_artifact_digest")
            != contract.get(SCORER_CONTRACT.ARTIFACT_SELF_KEY)):
        raise FullBankV2RunnerError(
            "optional V2 smoke authority projection changed")
    branch_receipt_matches = (
        smoke.get("branch_smoke_receipt_digest")
        == branch_smoke.get("smoke_branch_receipt_digest"))
    branch_receipt_lag_after_complete_corpus = False
    if not branch_receipt_matches:
        corpus_path = root / SCORER_FIT_RELATIVE_PATH / (
            BUILDER.SCORER_FIT_V2_CORPUS_RECEIPT_NAME)
        corpus, _corpus_raw = _load_json(
            corpus_path, label="optional V2 complete branch corpus receipt")
        corpus_payload = corpus.get("corpus_digest_payload")
        branch_receipt_lag_after_complete_corpus = bool(
            isinstance(corpus_payload, Mapping)
            and corpus.get("corpus_digest") == canonical_digest(corpus_payload)
            and corpus.get("status") == STATUS
            and corpus.get("complete") is True
            and corpus.get("states") == 120
            and corpus.get("state_count") == 120
            and corpus.get("completed_states") == 120
            and corpus.get("expected_branches") == 1_440
            and corpus.get("attempted_branches") == 1_440
            and corpus.get("attempted_count") == 1_440
            and corpus.get("rows") == 1_440
            and corpus.get("valid_branches") == 1_440
            and corpus.get("valid_count") == 1_440
            and corpus.get("invalid_branches") == 0
            and corpus.get("invalid_count") == 0
            and corpus.get("state_manifest_digest")
            == state_manifest.get("state_manifest_digest")
            and corpus.get("full_bank_assignment_manifest_digest")
            == assignment_manifest.get(
                "full_bank_assignment_manifest_digest")
            and corpus_payload.get("state_count") == 120
            and corpus_payload.get("attempted_branch_count") == 1_440
            and corpus_payload.get("valid_branch_count") == 1_440
            and corpus_payload.get("invalid_branch_count") == 0
            and corpus_payload.get("complete") is True
            and corpus_payload.get("state_manifest_digest")
            == state_manifest.get("state_manifest_digest")
            and corpus_payload.get("full_bank_assignment_manifest_digest")
            == assignment_manifest.get(
                "full_bank_assignment_manifest_digest")
            and smoke.get("state_id") == branch_smoke.get("state_id")
            and smoke.get("branch_identity_digests")
            == branch_smoke.get("branch_identity_digests")
            and smoke.get("branch_row_digests")
            == branch_smoke.get("branch_row_digests"))
    if (smoke.get("smoke_receipt_digest")
            != canonical_digest(_without(smoke, "smoke_receipt_digest"))
            or smoke.get("schema")
            != "go2_scorer_fit_corpus_v2_end_to_end_smoke_receipt_v1"
            or smoke.get("status") != STATUS
            or smoke.get("base_end_to_end_pass") is not True
            or branch_smoke.get("smoke_branch_receipt_digest")
            != canonical_digest(_without(
                branch_smoke, "smoke_branch_receipt_digest"))
            or branch_smoke.get("pass") is not True
            or not (branch_receipt_matches
                    or branch_receipt_lag_after_complete_corpus)
            or smoke.get("candidate_indices") != list(range(12))
            or smoke.get("branch_count") != 12
            or smoke.get("rendered_horizon_frame_count") != 48
            or smoke.get("true_latent_trajectory_count") != 12
            or smoke.get("true_latent_trajectory_shape") != [4, 768, 1024]
            or smoke.get("state_manifest_digest")
            != state_manifest.get("state_manifest_digest")
            or smoke.get("full_bank_assignment_manifest_digest")
            != assignment_manifest.get(
                "full_bank_assignment_manifest_digest")
            or smoke.get("scorer_fit_corpus_v2_scorer_contract_digest")
            != contract.get(SCORER_CONTRACT.CONTRACT_SELF_KEY)
            or smoke.get(
                "scorer_fit_corpus_v2_scorer_contract_artifact_digest")
            != contract.get(SCORER_CONTRACT.ARTIFACT_SELF_KEY)
            or not _is_hex(smoke.get("latent_index_digest"))
            or not isinstance(smoke.get("zero_new_resume_verified"), bool)
            or not isinstance(smoke.get(
                "single_shard_deletion_regeneration_verified"), bool)
            or not isinstance(smoke.get("smoke_protocol_complete"), bool)):
        raise FullBankV2RunnerError(
            "optional V2 encoding smoke receipt changed")
    complete = bool(
        smoke["zero_new_resume_verified"]
        and smoke["single_shard_deletion_regeneration_verified"]
        and smoke["smoke_protocol_complete"])
    return {
        "validation_kind": "encoding-smoke-optional",
        "pass": True,
        "terminal_present": True,
        "smoke_protocol_complete": complete,
        "zero_new_resume_verified": smoke["zero_new_resume_verified"],
        "single_registered_shard_regenerated": smoke[
            "single_shard_deletion_regeneration_verified"],
        "only_registered_missing_shard_changed": smoke[
            "single_shard_deletion_regeneration_verified"],
        "requires_full_encoder_refresh":
            branch_receipt_lag_after_complete_corpus,
        "candidate_outcomes_used_for_selection": False,
        "final_200_state_corpus_generated": False,
    }


def _normalise_encoding_projection(
        value: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    """Normalize the encoder's closed producer projection for orchestration."""

    if not isinstance(value, Mapping):
        raise FullBankV2RunnerError("encoder validation result is not a mapping")
    fields = {
        "state_count", "horizon_latent_count", "horizon_shape",
        "registered_smoke_shard_inventory_digest",
        "invocation_new_context_shards", "invocation_new_horizon_shards",
        "zero_new_resume_verified", "single_registered_shard_regenerated",
        "only_registered_missing_shard_changed",
        "single_shard_regeneration_target",
        "registered_smoke_artifact_inventory",
    }
    if not fields.issubset(value):
        raise FullBankV2RunnerError(
            "encoder validation projection lacks runner protocol fields")
    target = value.get("single_shard_regeneration_target")
    if (not isinstance(target, Mapping)
            or set(target) != {"path", "sha256", "byte_count", "shape"}
            or target.get("shape") != [4, 768, 1024]):
        raise FullBankV2RunnerError("encoder regeneration target changed")
    inventory = value.get("registered_smoke_artifact_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise FullBankV2RunnerError(
            "encoder registered smoke artifact inventory changed")
    advancing_names = {
        "smoke_encoding_receipt_v2.json",
        "encoding_invocation_summary_v2.json",
    }
    stable_inventory: list[dict[str, Any]] = []
    observed_advancing: set[str] = set()
    for item in inventory:
        if (not isinstance(item, Mapping)
                or set(item) != {"path", "raw_sha256", "byte_count"}
                or not isinstance(item.get("path"), str)
                or not _is_hex(item.get("raw_sha256"))
                or isinstance(item.get("byte_count"), bool)
                or not isinstance(item.get("byte_count"), int)
                or item["byte_count"] <= 0):
            raise FullBankV2RunnerError(
                "encoder registered smoke artifact binding changed")
        name = Path(item["path"]).name
        if name in advancing_names:
            observed_advancing.add(name)
        else:
            stable_inventory.append(dict(item))
    if observed_advancing != advancing_names or not stable_inventory:
        raise FullBankV2RunnerError(
            "encoder advancing smoke metadata inventory changed")
    stable_inventory.sort(key=lambda item: str(item["path"]))
    return {
        "validation_kind": kind, "pass": True,
        **{key: value[key] for key in fields
           if key != "registered_smoke_artifact_inventory"},
        "registered_smoke_stable_artifact_inventory_digest":
            canonical_digest(stable_inventory),
        "candidate_outcomes_used_for_selection": False,
        "final_200_state_corpus_generated": False,
    }


def _training_terminal_projection(*, root: Path) -> dict[str, Any]:
    from scripts import train_go2_utility_scorer_v1_2 as trainer

    value = trainer.load_and_validate_full_bank_v2_training_terminal_for_consumption(
        verify_encoder_checkpoint=True)
    if not isinstance(value, Mapping):
        raise FullBankV2RunnerError("training terminal validator returned no mapping")
    terminal = value.get("terminal")
    terminal_kind = value.get("terminal_kind")
    if (not isinstance(terminal, Mapping)
            or terminal_kind not in {
                "COMPLETION_DEGENERACY_FAILURE", "QUALIFICATION_FAILURE",
                "QUALIFICATION_PASS"}
            or not _is_hex(value.get("terminal_digest"))):
        raise FullBankV2RunnerError("training terminal projection changed")
    return {
        "validation_kind": "training-terminal", "pass": True,
        "terminal_kind": terminal_kind,
        "qualified": value["qualified"],
        "terminal_digest": value["terminal_digest"],
        "candidate_outcomes_used_for_selection": False,
        "final_200_state_corpus_generated": False,
    }


def _optional_training_terminal_projection(*, root: Path) -> dict[str, Any]:
    paths = (
        root / UTILITY_V2_RELATIVE_PATH /
        "completion_degeneracy_failure_v2.json",
        root / UTILITY_V2_RELATIVE_PATH / "qualification_v2.json",
    )
    present = [path for path in paths if path.exists() or path.is_symlink()]
    if not present:
        return {
            "validation_kind": "training-terminal-optional", "pass": True,
            "terminal_present": False,
            "candidate_outcomes_used_for_selection": False,
            "final_200_state_corpus_generated": False,
        }
    projection = _training_terminal_projection(root=root)
    return {
        **projection,
        "validation_kind": "training-terminal-optional",
        "terminal_present": True,
    }


def _development_terminal_projection(*, root: Path) -> dict[str, Any]:
    from scripts import (
        apply_go2_utility_scorer_to_counterfactual_development_v1_2 as apply,
    )

    value = apply.load_and_validate_full_bank_v2_development_terminal_for_consumption(
        root=root)
    if not isinstance(value, Mapping):
        raise FullBankV2RunnerError(
            "development terminal validator returned no mapping")
    if (not _is_hex(value.get("terminal_digest"))
            or value.get("qualified_scorer_bound") is not True
            or value.get("development_state_count") != 20
            or value.get("development_branch_count") != 240):
        raise FullBankV2RunnerError("development terminal binding changed")
    return {
        "validation_kind": "development-terminal", "pass": True,
        "terminal_digest": value["terminal_digest"],
        "qualified_scorer_bound": value["qualified_scorer_bound"],
        "development_state_count": value["development_state_count"],
        "development_branch_count": value["development_branch_count"],
        "candidate_outcomes_used_for_selection": False,
        "final_200_state_corpus_generated": False,
    }


def _optional_development_terminal_projection(*, root: Path) -> dict[str, Any]:
    path = root / UTILITY_V2_RELATIVE_PATH / (
        "counterfactual_development_transfer_v2/result_v2.json")
    if not path.exists() and not path.is_symlink():
        return {
            "validation_kind": "development-terminal-optional", "pass": True,
            "terminal_present": False,
            "candidate_outcomes_used_for_selection": False,
            "final_200_state_corpus_generated": False,
        }
    projection = _development_terminal_projection(root=root)
    return {
        **projection,
        "validation_kind": "development-terminal-optional",
        "terminal_present": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(
        "issue-design", "issue-source-correction", "freeze-manifests",
        "issue-scorer-contract", "run",
        "status", "internal-probe-genesis", "internal-probe-rocm",
        "internal-validate-branch-smoke", "internal-validate-encoding-smoke",
        "internal-validate-encoding-smoke-optional",
        "internal-validate-branch-corpus", "internal-validate-encoded-corpus",
        "internal-validate-training-terminal",
        "internal-validate-training-terminal-optional",
        "internal-validate-development-terminal",
        "internal-validate-development-terminal-optional",
    ))
    parser.add_argument(
        "--resume", action="store_true",
        help="resume registered V2 outputs after an infrastructure interruption")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume and args.stage != "run":
        raise SystemExit("--resume is valid only with --stage run")
    if args.stage == "internal-probe-genesis":
        return _emit_runtime_probe("genesis")
    if args.stage == "internal-probe-rocm":
        return _emit_runtime_probe("rocm")
    if args.stage.startswith("internal-validate-"):
        return _emit_validation(args.stage.removeprefix("internal-validate-"))
    if args.stage == "issue-design":
        report = issue_design()
        code = 0
    elif args.stage == "issue-source-correction":
        report = issue_source_correction()
        code = 0
    elif args.stage == "freeze-manifests":
        code, report = freeze_manifests()
    elif args.stage == "issue-scorer-contract":
        report = issue_scorer_contract()
        code = 0
    elif args.stage == "run":
        code, report = run_authorised(resume=args.resume)
    else:
        report = assemble_status_report()
        code = 0
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
