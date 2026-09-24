"""Custody and lineage for the outcome-free scorer projection interruption.

The first mixed 37-retained/8-replacement implementation was interrupted
before it issued a replacement identity.  Fifteen scene captures contained no
selected state and the sixteenth worker failed before a durable capture because
the consumer compared the full snapshot task-status structure with its strict
four-boolean selector projection.  This module does not change selector
semantics.  It preserves the exact interrupted authority artefacts, binds the
31 outcome-free request/capture records, and makes them ineligible for resume
under the corrected clean source.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "go2_scorer_projection_fix_preoutcome_interruption_v1"
STATUS = "SUPERSEDED_PRE_OUTCOME_IMPLEMENTATION_INTERRUPTION"
RECEIPT_RELATIVE_PATH = Path(
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    "preoutcome_projection_fix_interruption_receipt_v1.json"
)
CORPUS_ROOT_RELATIVE = Path(".generated/go2_branch_corpus_v1_2")
SCORER_ROOT_RELATIVE = Path(".generated/go2_utility_scorer_v1_2")

INTERRUPTED_SOURCE_REPOSITORY_COMMIT = (
    "77380c3ec85c2c973219403f72717f404a8b9999"
)
INTERRUPTED_SELECTOR_AMENDMENT_DIGEST = (
    "8c1d9f5ff1430fda6d9d80512afdba3070c78301befa57604aafcad9cb5c880b"
)
ATTEMPT_ROW_SET_DIGEST = (
    "69040d869803606142db92c472b9c273b55da3ff6c863948633a91f373c7795a"
)
ATTEMPT_ROW_SET_CANONICALIZATION = (
    "UTF-8 json.dumps(sort_keys=True) with default separators"
)
ATTEMPT_ROW_SET_COMPACT_COUNTERFACTUAL_DIGEST = (
    "5b2c31adca903b86c58026166b6bc560ae2f3d39fbba888dfa9e4752bc116171"
)
ATTEMPT_REQUEST_COUNT = 16
ATTEMPT_CAPTURE_COUNT = 15
ATTEMPT_ROW_COUNT = 31
ATTEMPT_ROW_BYTE_COUNT = 391_057

INTERRUPTED_ARTIFACTS: dict[str, dict[str, Any]] = {
    "mixed_precontract_disposition": {
        "managed_root": str(CORPUS_ROOT_RELATIVE),
        "active_path": (
            ".generated/go2_branch_corpus_v1_2/scorer_fit/"
            "preserved_state_mixed_precontract_disposition_reachability_v2.json"
        ),
        "archive_path": (
            ".generated/go2_branch_corpus_v1_2/scorer_fit/"
            "superseded_preoutcome_projection_fix_v1/"
            "preserved_state_mixed_precontract_disposition_reachability_v2."
            "07b34f7ab2c1ac9b22a5c816b91fa902d2f48a2abf8ae34bcc8aa13e3c58b26d.json"
        ),
        "self_digest_key": "mixed_precontract_disposition_receipt_digest",
        "self_digest": (
            "07b34f7ab2c1ac9b22a5c816b91fa902d2f48a2abf8ae34bcc8aa13e3c58b26d"
        ),
        "raw_sha256": (
            "68b81f893293dc0543c6a328deea751f595a27ada4cca16c8f9aab716af2b047"
        ),
        "byte_count": 29_403,
    },
    "scorer_contract": {
        "managed_root": str(SCORER_ROOT_RELATIVE),
        "active_path": (
            ".generated/go2_utility_scorer_v1_2/scorer_contract_v1_2.json"
        ),
        "archive_path": (
            ".generated/go2_utility_scorer_v1_2/"
            "superseded_preoutcome_projection_fix_v1/"
            "scorer_contract_v1_2."
            "d80699dc8a299f48dd55a7c1b8f7906083af395864739a9bb017a6550d52ada2.json"
        ),
        "self_digest_key": "contract_artifact_digest",
        "self_digest": (
            "d80699dc8a299f48dd55a7c1b8f7906083af395864739a9bb017a6550d52ada2"
        ),
        "raw_sha256": (
            "8717b0bd50cb6044b264d2c5bf53145f2b459750ce24664732a4ba1d15f0ee06"
        ),
        "byte_count": 68_440,
        "scorer_contract_v1_2_digest": (
            "87689cd28811bd0e0b3167f7ebf51962f1121c0455c87304ae55e7cc73f122cf"
        ),
    },
    "clean_source_launch": {
        "managed_root": str(CORPUS_ROOT_RELATIVE),
        "active_path": (
            ".generated/go2_branch_corpus_v1_2/scorer_fit/"
            "clean_source_launch_receipt.json"
        ),
        "archive_path": (
            ".generated/go2_branch_corpus_v1_2/scorer_fit/"
            "superseded_preoutcome_projection_fix_v1/"
            "clean_source_launch_receipt."
            "aacb791bbee2e688f5e1b22054c18a5aed041d55d9ed9765eec58860b8bfcdc3.json"
        ),
        "self_digest_key": "clean_source_launch_receipt_digest",
        "self_digest": (
            "aacb791bbee2e688f5e1b22054c18a5aed041d55d9ed9765eec58860b8bfcdc3"
        ),
        "raw_sha256": (
            "90cc7ae322d45f7169344c4ab48dbadf633a955c3f9ac60187d72eda373d9182"
        ),
        "byte_count": 1_549,
    },
}

ATTEMPT_ROOTS = {
    "request": (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/"
        "mixed_preoutcome_replacement_scene_requests_v2/large_enclosed_maze"
    ),
    "capture": (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/"
        "mixed_preoutcome_replacement_scene_captures_v2/large_enclosed_maze"
    ),
}


class InterruptionLineageError(RuntimeError):
    """The exact outcome-free interruption lineage could not be established."""


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _forbidden(path: Path) -> bool:
    return any(
        part == ".." or part == "sealed" or part == "sealed_test.json"
        or part.startswith("sealed_") for part in path.parts
    )


def _assert_no_symlink(path: Path) -> None:
    if _forbidden(path):
        raise InterruptionLineageError("lineage path crosses inaccessible custody")
    absolute = path if path.is_absolute() else Path.cwd() / path
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise InterruptionLineageError("lineage path contains a symlink")


def _pin_managed(
    relative_path: str | Path, *, root: Path, managed_root_relative: str | Path,
) -> Path:
    """Pin one exact generated path, permitting only its registered root alias."""

    repository = Path(root)
    if not repository.is_absolute():
        repository = Path.cwd() / repository
    managed = repository / Path(managed_root_relative)
    logical = repository / Path(relative_path)
    if _forbidden(managed) or _forbidden(logical):
        raise InterruptionLineageError("lineage path crosses inaccessible custody")
    try:
        suffix = logical.relative_to(managed)
    except ValueError as exc:
        raise InterruptionLineageError("lineage path escaped its managed root") from exc
    if not suffix.parts:
        raise InterruptionLineageError("lineage path names only its managed root")
    _assert_no_symlink(managed.parent)
    if managed.is_symlink():
        raw_target = managed.readlink()
        target = raw_target if raw_target.is_absolute() else managed.parent / raw_target
        if target.name != managed.name or _forbidden(target):
            raise InterruptionLineageError("managed lineage alias identity changed")
        _assert_no_symlink(target)
        try:
            canonical_root = target.resolve(strict=True)
        except OSError as exc:
            raise InterruptionLineageError("managed lineage root is missing") from exc
    else:
        if not managed.is_dir():
            raise InterruptionLineageError("managed lineage root is missing")
        canonical_root = managed.resolve(strict=True)
    if not canonical_root.is_dir() or canonical_root.name != managed.name:
        raise InterruptionLineageError("managed lineage root identity changed")
    _assert_no_symlink(canonical_root)
    pinned = canonical_root.joinpath(*suffix.parts)
    _assert_no_symlink(pinned)
    return pinned


def _artifact_paths(binding: Mapping[str, Any], *, root: Path) -> tuple[Path, Path]:
    managed = str(binding["managed_root"])
    return (
        _pin_managed(binding["active_path"], root=root,
                     managed_root_relative=managed),
        _pin_managed(binding["archive_path"], root=root,
                     managed_root_relative=managed),
    )


def _load_exact(path: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise InterruptionLineageError(f"{label} is missing")
    raw = path.read_bytes()
    if (len(raw) != int(binding["byte_count"])
            or _raw_sha256(raw) != binding["raw_sha256"]):
        raise InterruptionLineageError(f"{label} raw binding changed")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InterruptionLineageError(f"{label} JSON is invalid") from exc
    self_key = str(binding["self_digest_key"])
    if (payload.get(self_key) != binding["self_digest"]
            or payload.get(self_key) != _digest({
                key: value for key, value in payload.items() if key != self_key
            })):
        raise InterruptionLineageError(f"{label} self binding changed")
    if ("scorer_contract_v1_2_digest" in binding
            and payload.get("scorer_contract_v1_2_digest")
            != binding["scorer_contract_v1_2_digest"]):
        raise InterruptionLineageError(f"{label} scorer-contract binding changed")
    return payload


def _locate_interrupted_artifact(
    label: str, *, root: Path, require_archived: bool,
) -> tuple[dict[str, Any], Path]:
    binding = INTERRUPTED_ARTIFACTS[label]
    active, archive = _artifact_paths(binding, root=root)
    active_exact = False
    archive_exact = False
    if active.is_file() and not active.is_symlink():
        raw = active.read_bytes()
        active_exact = (len(raw) == binding["byte_count"]
                        and _raw_sha256(raw) == binding["raw_sha256"])
    if archive.is_file() and not archive.is_symlink():
        raw = archive.read_bytes()
        archive_exact = (len(raw) == binding["byte_count"]
                         and _raw_sha256(raw) == binding["raw_sha256"])
    if active_exact and archive_exact:
        # Recover the sole possible hard-link interruption between archive
        # installation and active-name removal.  Independent duplicate copies
        # remain an ambiguous collision and fail closed.
        try:
            same_inode = os.path.samefile(active, archive)
        except OSError:
            same_inode = False
        if not same_inode:
            raise InterruptionLineageError(
                f"{label} exists at both active and archive paths")
        active.unlink()
        directory = os.open(active.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        active_exact = False
    if require_archived:
        if not archive_exact:
            raise InterruptionLineageError(f"{label} archive binding is missing")
        return _load_exact(archive, binding, label), archive
    if archive_exact:
        return _load_exact(archive, binding, label), archive
    if active_exact:
        return _load_exact(active, binding, label), active
    raise InterruptionLineageError(f"exact interrupted {label} is unavailable")


def _attempt_row(kind: str, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "request":
        self_key = "mixed_replacement_scene_request_digest"
        scene_id = payload.get("scene", {}).get("scene_id")
        ordinal = payload.get("scene_ordinal")
        selected = None
        failure = None
    else:
        self_key = "mixed_replacement_scene_capture_digest"
        scene_id = payload.get("scene_id")
        ordinal = payload.get("request", {}).get("scene_ordinal")
        selected = payload.get("chosen_state") is not None
        failure = payload.get("worker_failure")
    raw = path.read_bytes()
    return {
        "kind": kind,
        "name": path.name,
        "raw_sha256": _raw_sha256(raw),
        "byte_count": len(raw),
        "self_digest": payload.get(self_key),
        "scene_id": scene_id,
        "scene_ordinal": ordinal,
        "selected": selected,
        "worker_failure": failure,
    }


def _validate_zero_outcome_surface(payload: Mapping[str, Any], label: str) -> None:
    if any(payload.get(key) not in (False, 0) for key in (
        "candidate_outcomes_loaded", "branch_identities_created",
        "branches_attempted", "frames_rendered", "target_latents_encoded",
        "scorer_training_started", "predictor_checkpoints_opened",
    )):
        raise InterruptionLineageError(f"{label} is not outcome-free")


def _collect_attempt_rows(*, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    loaded_requests: dict[str, dict[str, Any]] = {}
    expected = {"request": ATTEMPT_REQUEST_COUNT, "capture": ATTEMPT_CAPTURE_COUNT}
    for kind in ("request", "capture"):
        directory = _pin_managed(
            ATTEMPT_ROOTS[kind], root=root,
            managed_root_relative=CORPUS_ROOT_RELATIVE,
        )
        if not directory.is_dir() or directory.is_symlink():
            raise InterruptionLineageError(f"interrupted {kind} root is missing")
        paths = sorted(directory.glob("*.json"), key=lambda value: value.name)
        if len(paths) != expected[kind] or any(path.is_symlink() for path in paths):
            raise InterruptionLineageError(f"interrupted {kind} inventory changed")
        for path in paths:
            _assert_no_symlink(path)
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise InterruptionLineageError(
                    f"interrupted {kind} JSON is invalid") from exc
            self_key = ("mixed_replacement_scene_request_digest" if kind == "request"
                        else "mixed_replacement_scene_capture_digest")
            self_digest = payload.get(self_key)
            if (not isinstance(self_digest, str) or path.stem != (
                    payload.get("mixed_replacement_scene_request_digest")
                    if kind == "request" else
                    payload.get("mixed_replacement_scene_request_digest"))):
                raise InterruptionLineageError(
                    f"interrupted {kind} filename binding changed")
            if self_digest != _digest({
                    key: value for key, value in payload.items() if key != self_key}):
                raise InterruptionLineageError(
                    f"interrupted {kind} self binding changed")
            _validate_zero_outcome_surface(payload, f"interrupted {kind}")
            if kind == "request":
                if payload.get("binding_receipt") is not False:
                    raise InterruptionLineageError("request binding surface changed")
                loaded_requests[path.name] = payload
            else:
                request = loaded_requests.get(path.name)
                if (request is None or payload.get("request") != request
                        or payload.get("chosen_state") is not None
                        or payload.get("worker_failure") is not None):
                    raise InterruptionLineageError(
                        "capture is not one exact outcome-free negative attempt")
            rows.append(_attempt_row(kind, path, payload))
    if (len(rows) != ATTEMPT_ROW_COUNT
            or sum(int(row["byte_count"]) for row in rows)
            != ATTEMPT_ROW_BYTE_COUNT
            or _digest(rows) != ATTEMPT_ROW_SET_DIGEST):
        raise InterruptionLineageError("interrupted request/capture row set changed")
    return rows


def _rows_from_receipt(receipt: Mapping[str, Any], *, root: Path) -> list[dict[str, Any]]:
    rows = receipt.get("attempt_rows")
    if not isinstance(rows, list) or len(rows) != ATTEMPT_ROW_COUNT:
        raise InterruptionLineageError("interruption receipt row inventory changed")
    observed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "kind", "name", "raw_sha256", "byte_count", "self_digest",
            "scene_id", "scene_ordinal", "selected", "worker_failure",
        }:
            raise InterruptionLineageError("interruption receipt row changed")
        kind = row.get("kind")
        name = row.get("name")
        if kind not in ATTEMPT_ROOTS or not isinstance(name, str) \
                or Path(name).name != name or not name.endswith(".json"):
            raise InterruptionLineageError("interruption receipt row path changed")
        path = _pin_managed(
            Path(ATTEMPT_ROOTS[kind]) / name, root=root,
            managed_root_relative=CORPUS_ROOT_RELATIVE,
        )
        if not path.is_file() or path.is_symlink():
            raise InterruptionLineageError("interruption attempt row is missing")
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise InterruptionLineageError("interruption attempt row is invalid") from exc
        _validate_zero_outcome_surface(payload, "interruption attempt row")
        observed.append(_attempt_row(str(kind), path, payload))
    if observed != rows or _digest(observed) != ATTEMPT_ROW_SET_DIGEST:
        raise InterruptionLineageError("interruption attempt bytes changed")
    return observed


def _receipt_payload(
    *, source_repository_commit: str, clean_source_binding_digest: str,
    bound_implementations_digest: str, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "record_complete": True,
        "attempt_complete": False,
        "binding_receipt": False,
        "scientific_gate_input": False,
        "may_satisfy_selector_gate": False,
        "cryptographically_bound_by_successor_contract": True,
        "reason": (
            "strict snapshot-status projection was not applied at the mixed "
            "replacement capture validation boundary"
        ),
        "disposition": (
            "invalid implementation attempt; preserve exact bytes; never resume "
            "its request/capture rows under successor source"
        ),
        "interrupted_source_repository_commit":
            INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
        "superseding_source_repository_commit": source_repository_commit,
        "superseding_clean_source_binding_digest": clean_source_binding_digest,
        "superseding_bound_implementations_digest": bound_implementations_digest,
        "state_selector_amendment_digest": INTERRUPTED_SELECTOR_AMENDMENT_DIGEST,
        "selector_semantics_changed": False,
        "interrupted_artifacts": INTERRUPTED_ARTIFACTS,
        "attempt_roots": ATTEMPT_ROOTS,
        "attempt_rows": rows,
        "attempt_row_count": ATTEMPT_ROW_COUNT,
        "attempt_request_count": ATTEMPT_REQUEST_COUNT,
        "attempt_capture_count": ATTEMPT_CAPTURE_COUNT,
        "attempt_row_byte_count": ATTEMPT_ROW_BYTE_COUNT,
        "attempt_row_set_digest": ATTEMPT_ROW_SET_DIGEST,
        "attempt_row_set_canonicalization": ATTEMPT_ROW_SET_CANONICALIZATION,
        "compact_separator_counterfactual_digest":
            ATTEMPT_ROW_SET_COMPACT_COUNTERFACTUAL_DIGEST,
        "positive_candidate_observed_transiently_in_worker_memory": True,
        "durable_validated_selected_state_artifact_count": 0,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "scorer_qualification_started": False,
        "predictor_checkpoints_opened": 0,
        "request_capture_rows_active_for_resume": False,
        "old_mixed_contract_and_launch_archived_nonoverwriting": True,
    }
    payload["preoutcome_projection_fix_interruption_receipt_digest"] = _digest(payload)
    return payload


def receipt_binding(receipt: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    path = _pin_managed(
        RECEIPT_RELATIVE_PATH, root=root,
        managed_root_relative=CORPUS_ROOT_RELATIVE,
    )
    raw = path.read_bytes()
    return {
        "path": str(RECEIPT_RELATIVE_PATH),
        "receipt_digest": receipt[
            "preoutcome_projection_fix_interruption_receipt_digest"],
        "raw_sha256": _raw_sha256(raw),
        "byte_count": len(raw),
        "status": STATUS,
    }


def validate_interruption_receipt(
    receipt: Mapping[str, Any], *, expected_source_repository_commit: str,
    expected_clean_source_binding_digest: str,
    expected_bound_implementations_digest: str, root: Path = ROOT,
    require_archived: bool = True,
) -> dict[str, Any]:
    expected_keys = {
        "schema", "status", "record_complete", "attempt_complete",
        "binding_receipt", "scientific_gate_input", "may_satisfy_selector_gate",
        "cryptographically_bound_by_successor_contract",
        "reason",
        "disposition", "interrupted_source_repository_commit",
        "superseding_source_repository_commit",
        "superseding_clean_source_binding_digest",
        "superseding_bound_implementations_digest",
        "state_selector_amendment_digest", "selector_semantics_changed",
        "interrupted_artifacts", "attempt_roots", "attempt_rows",
        "attempt_row_count", "attempt_request_count", "attempt_capture_count",
        "attempt_row_byte_count", "attempt_row_set_digest",
        "attempt_row_set_canonicalization",
        "compact_separator_counterfactual_digest",
        "positive_candidate_observed_transiently_in_worker_memory",
        "durable_validated_selected_state_artifact_count",
        "candidate_outcomes_loaded",
        "branch_identities_created", "branches_attempted", "frames_rendered",
        "target_latents_encoded", "scorer_training_started",
        "scorer_qualification_started", "predictor_checkpoints_opened",
        "request_capture_rows_active_for_resume",
        "old_mixed_contract_and_launch_archived_nonoverwriting",
        "preoutcome_projection_fix_interruption_receipt_digest",
    }
    if (set(receipt) != expected_keys or receipt.get("schema") != SCHEMA
            or receipt.get("status") != STATUS
            or receipt.get("record_complete") is not True
            or receipt.get("attempt_complete") is not False
            or receipt.get("binding_receipt") is not False
            or receipt.get("scientific_gate_input") is not False
            or receipt.get("may_satisfy_selector_gate") is not False
            or receipt.get("cryptographically_bound_by_successor_contract")
            is not True
            or receipt.get("interrupted_source_repository_commit")
            != INTERRUPTED_SOURCE_REPOSITORY_COMMIT
            or receipt.get("superseding_source_repository_commit")
            != expected_source_repository_commit
            or receipt.get("superseding_clean_source_binding_digest")
            != expected_clean_source_binding_digest
            or receipt.get("superseding_bound_implementations_digest")
            != expected_bound_implementations_digest
            or receipt.get("state_selector_amendment_digest")
            != INTERRUPTED_SELECTOR_AMENDMENT_DIGEST
            or receipt.get("selector_semantics_changed") is not False
            or receipt.get("interrupted_artifacts") != INTERRUPTED_ARTIFACTS
            or receipt.get("attempt_roots") != ATTEMPT_ROOTS
            or receipt.get("attempt_row_count") != ATTEMPT_ROW_COUNT
            or receipt.get("attempt_request_count") != ATTEMPT_REQUEST_COUNT
            or receipt.get("attempt_capture_count") != ATTEMPT_CAPTURE_COUNT
            or receipt.get("attempt_row_byte_count") != ATTEMPT_ROW_BYTE_COUNT
            or receipt.get("attempt_row_set_digest") != ATTEMPT_ROW_SET_DIGEST
            or receipt.get("attempt_row_set_canonicalization")
            != ATTEMPT_ROW_SET_CANONICALIZATION
            or receipt.get("compact_separator_counterfactual_digest")
            != ATTEMPT_ROW_SET_COMPACT_COUNTERFACTUAL_DIGEST
            or receipt.get(
                "positive_candidate_observed_transiently_in_worker_memory") is not True
            or receipt.get("durable_validated_selected_state_artifact_count") != 0
            or any(receipt.get(key) not in (False, 0) for key in (
                "candidate_outcomes_loaded", "branch_identities_created",
                "branches_attempted", "frames_rendered", "target_latents_encoded",
                "scorer_training_started", "scorer_qualification_started",
                "predictor_checkpoints_opened",
            ))
            or receipt.get("request_capture_rows_active_for_resume") is not False
            or receipt.get("old_mixed_contract_and_launch_archived_nonoverwriting")
            is not True):
        raise InterruptionLineageError("interruption receipt contract changed")
    self_key = "preoutcome_projection_fix_interruption_receipt_digest"
    if receipt.get(self_key) != _digest({
            key: value for key, value in receipt.items() if key != self_key}):
        raise InterruptionLineageError("interruption receipt self binding changed")
    rows = _rows_from_receipt(receipt, root=root)
    if _digest(rows) != ATTEMPT_ROW_SET_DIGEST:
        raise InterruptionLineageError("interruption row-set digest changed")
    expected_receipt = _receipt_payload(
        source_repository_commit=expected_source_repository_commit,
        clean_source_binding_digest=expected_clean_source_binding_digest,
        bound_implementations_digest=expected_bound_implementations_digest,
        rows=rows,
    )
    if dict(receipt) != expected_receipt:
        raise InterruptionLineageError(
            "interruption receipt differs from exact reconstruction")
    loaded = {
        label: _locate_interrupted_artifact(
            label, root=root, require_archived=require_archived)[0]
        for label in INTERRUPTED_ARTIFACTS
    }
    if (loaded["scorer_contract"].get(
            "mixed_precontract_disposition_receipt_digest")
            != INTERRUPTED_ARTIFACTS["mixed_precontract_disposition"]["self_digest"]
            or loaded["clean_source_launch"].get("scorer_contract_artifact_digest")
            != INTERRUPTED_ARTIFACTS["scorer_contract"]["self_digest"]
            or loaded["clean_source_launch"].get(
                "mixed_precontract_disposition_receipt_digest")
            != INTERRUPTED_ARTIFACTS["mixed_precontract_disposition"]["self_digest"]):
        raise InterruptionLineageError("interrupted artifact cross-binding changed")
    return dict(receipt)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    _assert_no_symlink(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _assert_no_symlink(temporary)
    if temporary.exists() or temporary.is_symlink():
        raise InterruptionLineageError("interruption receipt temporary exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    installed = False
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            sink.flush()
            os.fsync(sink.fileno())
        # Linking the already-fsynced inode is an atomic no-overwrite install;
        # unlike os.replace it cannot silently replace a competing receipt.
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        installed = True
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if not installed:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def issue_and_archive_interruption_receipt(
    *, source_repository_commit: str, clean_source_binding_digest: str,
    bound_implementations_digest: str, root: Path = ROOT,
) -> dict[str, Any]:
    """Archive the three exact authorities and issue one immutable receipt."""

    receipt_path = _pin_managed(
        RECEIPT_RELATIVE_PATH, root=root,
        managed_root_relative=CORPUS_ROOT_RELATIVE,
    )
    if receipt_path.exists() or receipt_path.is_symlink():
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise InterruptionLineageError("interruption receipt path is invalid")
        existing = json.loads(receipt_path.read_text())
        return validate_interruption_receipt(
            existing,
            expected_source_repository_commit=source_repository_commit,
            expected_clean_source_binding_digest=clean_source_binding_digest,
            expected_bound_implementations_digest=bound_implementations_digest,
            root=root, require_archived=True,
        )

    # Only first issuance requires the exact 16/15 inventory.  Once the receipt
    # exists, its explicit 31 rows are reopened by name so later successor-source
    # requests/captures can coexist without making lineage validation non-idempotent.
    rows = _collect_attempt_rows(root=root)
    expected = _receipt_payload(
        source_repository_commit=source_repository_commit,
        clean_source_binding_digest=clean_source_binding_digest,
        bound_implementations_digest=bound_implementations_digest,
        rows=rows,
    )

    # Locate and fully validate all old bytes before moving any of them.
    locations = {
        label: _locate_interrupted_artifact(
            label, root=root, require_archived=False)[1]
        for label in INTERRUPTED_ARTIFACTS
    }
    for label, source in locations.items():
        binding = INTERRUPTED_ARTIFACTS[label]
        active, archive = _artifact_paths(binding, root=root)
        if source == archive:
            continue
        if source != active:
            raise InterruptionLineageError("interrupted artifact location changed")
        _assert_no_symlink(archive.parent)
        archive.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_symlink(archive.parent)
        if archive.exists() or archive.is_symlink():
            raise InterruptionLineageError("interruption archive collision")
        # Preserve without an overwrite window.  A crash between link and
        # unlink leaves two names for the same inode; the locator recognizes
        # precisely that state, removes the active name, and resumes.  Two
        # independent exact copies remain ambiguous and fail closed.
        os.link(active, archive, follow_symlinks=False)
        active.unlink()
    _atomic_write(receipt_path, expected)
    return validate_interruption_receipt(
        expected,
        expected_source_repository_commit=source_repository_commit,
        expected_clean_source_binding_digest=clean_source_binding_digest,
        expected_bound_implementations_digest=bound_implementations_digest,
        root=root, require_archived=True,
    )


def load_and_validate_interruption_receipt(
    *, expected_source_repository_commit: str,
    expected_clean_source_binding_digest: str,
    expected_bound_implementations_digest: str, root: Path = ROOT,
) -> dict[str, Any]:
    path = _pin_managed(
        RECEIPT_RELATIVE_PATH, root=root,
        managed_root_relative=CORPUS_ROOT_RELATIVE,
    )
    if not path.is_file() or path.is_symlink():
        raise InterruptionLineageError("projection-fix interruption receipt is missing")
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InterruptionLineageError("projection-fix interruption receipt is invalid") \
            from exc
    return validate_interruption_receipt(
        receipt,
        expected_source_repository_commit=expected_source_repository_commit,
        expected_clean_source_binding_digest=expected_clean_source_binding_digest,
        expected_bound_implementations_digest=expected_bound_implementations_digest,
        root=root, require_archived=True,
    )


def lineage_contract() -> dict[str, Any]:
    """Static contract bound into the prospective scorer contract digest."""

    return {
        "schema": SCHEMA,
        "status": STATUS,
        "receipt_path": str(RECEIPT_RELATIVE_PATH),
        "interrupted_source_repository_commit":
            INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
        "interrupted_artifacts": INTERRUPTED_ARTIFACTS,
        "attempt_roots": ATTEMPT_ROOTS,
        "attempt_request_count": ATTEMPT_REQUEST_COUNT,
        "attempt_capture_count": ATTEMPT_CAPTURE_COUNT,
        "attempt_row_count": ATTEMPT_ROW_COUNT,
        "attempt_row_byte_count": ATTEMPT_ROW_BYTE_COUNT,
        "attempt_row_set_digest": ATTEMPT_ROW_SET_DIGEST,
        "attempt_row_set_canonicalization": ATTEMPT_ROW_SET_CANONICALIZATION,
        "compact_separator_counterfactual_digest":
            ATTEMPT_ROW_SET_COMPACT_COUNTERFACTUAL_DIGEST,
        "state_selector_amendment_digest": INTERRUPTED_SELECTOR_AMENDMENT_DIGEST,
        "selector_semantics_changed": False,
        "scientific_outcome_existed": False,
        "scientific_gate_input": False,
        "may_satisfy_selector_gate": False,
        "cryptographically_bound_by_successor_contract": True,
        "resume_under_successor_source": False,
    }
