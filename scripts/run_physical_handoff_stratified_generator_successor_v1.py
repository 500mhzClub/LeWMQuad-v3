#!/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
"""Stratified physical-handoff generator successor V1.

This is a conventional, direct foreground runner.  Each ``qualify-stream``
invocation owns exactly one family/stratum stream and visits its candidates in
attempt order until the fourth qualified state or the frozen 64-attempt cap.
Importing this module creates no output, simulator, encoder, or ranker.
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from lewm.safety import physical_handoff_stratified_generator_successor_v1_contract as CONTRACT
from lewm.safety import physical_handoff_stratified_generator_successor_v1_metrics as METRICS
from lewm_genesis.scene_builder import shutdown_genesis
from scripts import run_physical_graph_edge_handoff_qualification_v1 as V1
from scripts import run_physical_graph_edge_handoff_qualification_v4 as V4


class ExperimentError(RuntimeError):
    """A successor source, persistence, generator, or scientific gate failed."""


class GenesisSuccessorPhysicalBackend(V4.GenesisGo2PhysicalBackend):
    """Real V4 physics with successor-native candidate registration."""

    def __init__(self, *, backend: str = "cpu") -> None:
        super().__init__(backend=backend)
        self._owned_qualification_sessions: list[Any] | None = None

    def _session(self, spec: Mapping[str, Any]) -> Any:
        session = super()._session(spec)
        if self._owned_qualification_sessions is not None:
            self._owned_qualification_sessions.append(session)
        return session

    def _release_owned_qualification_sessions(
        self,
    ) -> list[tuple[str, BaseException]]:
        sessions = self._owned_qualification_sessions
        if sessions is None:
            raise ExperimentError("qualification session ownership is inactive")
        self._owned_qualification_sessions = None
        failures: list[tuple[str, BaseException]] = []
        while sessions:
            session = sessions.pop()
            try:
                session.ctx.build.scene.destroy()
            except BaseException as exc:  # pragma: no cover - native hard failure
                exc.__traceback__ = None
                failures.append(("scene teardown", exc))
            finally:
                del session
        del sessions
        try:
            shutdown_genesis()
        except BaseException as exc:  # pragma: no cover - native hard failure
            exc.__traceback__ = None
            failures.append(("Genesis process-global shutdown", exc))
        try:
            gc.collect()
        except BaseException as exc:  # pragma: no cover - native hard failure
            exc.__traceback__ = None
            failures.append(("post-shutdown garbage collection", exc))
        return failures

    @staticmethod
    def _cleanup_failure_text(
        failures: Sequence[tuple[str, BaseException]],
    ) -> str:
        return "; ".join(
            f"{phase}: {type(exc).__name__}: {exc}" for phase, exc in failures
        )

    @staticmethod
    def _clear_exception_frame_references(primary: BaseException) -> None:
        pending = [primary]
        seen: set[int] = set()
        while pending:
            error = pending.pop()
            identity = id(error)
            if identity in seen:
                continue
            seen.add(identity)
            if error.__traceback__ is not None:
                traceback.clear_frames(error.__traceback__)
            if error.__cause__ is not None:
                pending.append(error.__cause__)
            if error.__context__ is not None:
                pending.append(error.__context__)

    def qualify_successor_candidate(
        self, candidate_spec: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        spec = CONTRACT.validate_candidate_spec(candidate_spec)
        if self._owned_qualification_sessions is not None:
            raise ExperimentError("successor qualification cannot be re-entered")
        self._owned_qualification_sessions = []
        try:
            result = self._qualify(  # noqa: SLF001
                spec, require_registered=False
            )
        except BaseException as primary:
            self._clear_exception_frame_references(primary)
            cleanup_failures = self._release_owned_qualification_sessions()
            if cleanup_failures:
                primary.add_note(
                    "successor qualification lifecycle cleanup also failed: "
                    f"{self._cleanup_failure_text(cleanup_failures)}"
                )
            raise
        cleanup_failures = self._release_owned_qualification_sessions()
        if cleanup_failures:
            phase, first_failure = cleanup_failures[0]
            error = ExperimentError(f"successor qualification {phase} failed")
            if len(cleanup_failures) > 1:
                error.add_note(
                    "additional successor qualification cleanup failures: "
                    f"{self._cleanup_failure_text(cleanup_failures[1:])}"
                )
            raise error from first_failure
        return result


EXPERIMENT_ID = CONTRACT.EXPERIMENT_ID
PARENT_COMMIT = CONTRACT.SOURCE_PARENT_COMMIT
FREEZE_SUBJECT = CONTRACT.CONTRACT_FREEZE_COMMIT_SUBJECT
RESULT_SUBJECT = CONTRACT.RESULT_COMMIT_SUBJECT
OUTPUT_ROOT = Path(CONTRACT.OUTPUT_ROOT)
MATERIAL_ROOT = Path(CONTRACT.MATERIAL_ROOT)
V4_OUTPUT_ROOT = Path(CONTRACT.V4.OUTPUT_ROOT)
V4_MATERIAL_ROOT = Path(CONTRACT.V4.MATERIAL_ROOT)
V4_REGENERATION_RECEIPT = Path(CONTRACT.V4.EXTERNAL_REGENERATION_RECEIPT)

FAMILIES = tuple(CONTRACT.FAMILY_IDS)
STRATA_PER_FAMILY = int(CONTRACT.STRATA_PER_FAMILY)
ATTEMPT_CAP = int(CONTRACT.MAX_ATTEMPTS_PER_STREAM)
QUALIFIED_TARGET = int(CONTRACT.TARGET_QUALIFIED_PER_STREAM)
STREAM_COUNT = len(FAMILIES) * STRATA_PER_FAMILY
POTENTIAL_CANDIDATE_COUNT = STREAM_COUNT * ATTEMPT_CAP

DOC_PATHS = {
    "contract": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_contract_2026-09-04.json",
    "fixture": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_fixture_2026-09-04.json",
    "output_schema": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_output_schema_2026-09-04.json",
    "preregistration": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_preregistration_2026-09-04.md",
    "source_closure": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_source_closure_2026-09-04.json",
    "scientific_invariance": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_scientific_invariance_2026-09-04.json",
    "v4_context_binding": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_v4_context_binding_2026-09-04.json",
}


def canonical_bytes(value: Any) -> bytes:
    return CONTRACT.canonical_json_bytes(value).rstrip(b"\n") + b"\n"


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ExperimentError(f"required file is absent: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ExperimentError(f"path is not an ordinary regular file: {path}")
    _require_no_symlink_ancestors(path)
    return info


def _require_no_symlink_ancestors(path: Path) -> None:
    cursor = path.parent
    while True:
        if cursor.is_symlink():
            raise ExperimentError(f"path has a symlink ancestor: {path}")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent


def _is_successor_path(path: Path) -> bool:
    absolute = path.absolute()
    return any(
        absolute == root.absolute() or root.absolute() in absolute.parents
        for root in (OUTPUT_ROOT, MATERIAL_ROOT)
    )


_V4_EVIDENCE_INODE_CACHE: set[tuple[int, int]] | None = None
_V4_PHYSICAL_SHARD_SHA256_CACHE: set[str] | None = None


def _bound_v4_evidence_inodes() -> set[tuple[int, int]]:
    """Bind the exact nine-file/514-file V4 terminal without discovery."""

    global _V4_EVIDENCE_INODE_CACHE, _V4_PHYSICAL_SHARD_SHA256_CACHE
    if _V4_EVIDENCE_INODE_CACHE is not None:
        return set(_V4_EVIDENCE_INODE_CACHE)
    root = Path(CONTRACT.V4.OUTPUT_ROOT)
    material = Path(CONTRACT.V4.MATERIAL_ROOT)
    hashes_path = root / "file_hashes.json"
    hashes_binding = CONTRACT.V4_CONTEXT_AUTHORITY["bindings"]["file_hashes"]
    info = _require_regular(hashes_path)
    if (
        int(info.st_size) != int(hashes_binding["bytes"])
        or sha256_file(hashes_path) != str(hashes_binding["sha256"])
    ):
        raise ExperimentError("bound V4 file-hash authority drift")
    hashes = _ordinary_json(hashes_path)
    rows = hashes.get("files")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ExperimentError("bound V4 official inventory drift")
    expected_official = {
        "contract.json",
        "metrics.json",
        "panel_adequacy.json",
        "qualification_state_dispositions.jsonl",
        "result.json",
        "result.md",
        "scientific_invariance_receipt.json",
        "v1_v2_v3_custody_and_nonreuse.json",
    }
    if {row.get("path") for row in rows if isinstance(row, Mapping)} != expected_official:
        raise ExperimentError("bound V4 official file names drift")
    paths = [hashes_path]
    for row in rows:
        path = root / str(row["path"])
        file_info = _require_regular(path)
        if (
            int(file_info.st_size) != int(row["bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            raise ExperimentError("bound V4 official file drift")
        paths.append(path)
    ledger_path = root / "qualification_state_dispositions.jsonl"
    raw = ledger_path.read_bytes()
    material_paths = [
        material / "material_contract.json",
        material / "prospective_pool.json",
    ]
    physical_shard_sha256s: list[str] = []
    material_contract = _ordinary_json(material_paths[0])
    prospective_pool = _ordinary_json(material_paths[1])
    CONTRACT.V4.validate_content_digest(material_contract)
    CONTRACT.V4.validate_content_digest(prospective_pool)
    if (
        prospective_pool.get("experiment_id") != CONTRACT.V4.EXPERIMENT_ID
        or prospective_pool.get("source_freeze_commit")
        != CONTRACT.V4_SOURCE_FREEZE_COMMIT
        or prospective_pool.get("specs") != CONTRACT.V4.build_candidate_specs()
        or material_contract.get("experiment_id") != CONTRACT.V4.EXPERIMENT_ID
        or material_contract.get("source_freeze_commit")
        != CONTRACT.V4_SOURCE_FREEZE_COMMIT
        or material_contract.get("prospective_pool_sha256")
        != sha256_file(material_paths[1])
        or material_contract.get("official_contract_sha256")
        != sha256_file(root / "contract.json")
    ):
        raise ExperimentError("bound V4 material initialization drift")
    try:
        ledger_rows = [json.loads(line) for line in raw.splitlines()]
    except json.JSONDecodeError as exc:
        raise ExperimentError("bound V4 qualification ledger invalid") from exc
    if len(ledger_rows) != 256:
        raise ExperimentError("bound V4 qualification ledger cardinality drift")
    for index, row in enumerate(ledger_rows):
        if row.get("pool_index") != index:
            raise ExperimentError("bound V4 qualification ledger order drift")
        for field in ("material_metadata_binding", "material_payload_binding"):
            binding = row.get(field)
            if not isinstance(binding, Mapping):
                raise ExperimentError("bound V4 material binding absent")
            path = material / str(binding.get("path"))
            file_info = _require_regular(path)
            if (
                int(file_info.st_size) != int(binding.get("bytes", -1))
                or sha256_file(path) != str(binding.get("sha256"))
            ):
                raise ExperimentError("bound V4 material shard drift")
            material_paths.append(path)
            physical_shard_sha256s.append(str(binding["sha256"]))
    if len(physical_shard_sha256s) != 512:
        raise ExperimentError("bound V4 physical-shard inventory drift")
    if len({str(path) for path in material_paths}) != 514:
        raise ExperimentError("bound V4 material inventory drift")
    receipt = Path(CONTRACT.V4.EXTERNAL_REGENERATION_RECEIPT)
    receipt_binding = CONTRACT.V4_CONTEXT_AUTHORITY["bindings"][
        "independent_reducer_receipt"
    ]
    receipt_info = _require_regular(receipt)
    if (
        int(receipt_info.st_size) != int(receipt_binding["bytes"])
        or sha256_file(receipt) != str(receipt_binding["sha256"])
    ):
        raise ExperimentError("bound V4 independent receipt drift")
    paths.extend(material_paths)
    paths.append(receipt)
    inodes = {
        (int(_require_regular(path).st_dev), int(_require_regular(path).st_ino))
        for path in paths
    }
    if len(inodes) != len(paths):
        raise ExperimentError("bound V4 evidence contains an inode alias")
    _V4_EVIDENCE_INODE_CACHE = set(inodes)
    _V4_PHYSICAL_SHARD_SHA256_CACHE = set(physical_shard_sha256s)
    return set(inodes)


def _bound_v4_physical_shard_sha256s() -> set[str]:
    if _V4_PHYSICAL_SHARD_SHA256_CACHE is None:
        _bound_v4_evidence_inodes()
    if _V4_PHYSICAL_SHARD_SHA256_CACHE is None:  # pragma: no cover - invariant
        raise ExperimentError("bound V4 physical-shard SHA authority is absent")
    return set(_V4_PHYSICAL_SHARD_SHA256_CACHE)


def _require_successor_inode_nonreuse(path: Path) -> None:
    if not _is_successor_path(path):
        return
    info = _require_regular(path)
    if (int(info.st_dev), int(info.st_ino)) in _bound_v4_evidence_inodes():
        raise ExperimentError("successor output is hard-linked to bound V4 context")
    absolute = path.absolute()
    if (
        MATERIAL_ROOT.absolute() in absolute.parents
        and path.name in {"metadata.json", "payload.npz"}
        and sha256_file(path) in _bound_v4_physical_shard_sha256s()
    ):
        raise ExperimentError("successor material copies a bound V4 physical shard")


def _ordinary_json(path: Path) -> dict[str, Any]:
    _require_regular(path)
    _require_successor_inode_nonreuse(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ExperimentError(f"noncanonical ordinary JSON: {path}")
    return copy.deepcopy(value)


def _ordinary_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    _require_regular(path)
    _require_successor_inode_nonreuse(path)
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ExperimentError(f"invalid canonical JSONL framing: {path}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"invalid JSONL row {index}: {path}") from exc
        if not isinstance(value, dict) or line != canonical_bytes(value):
            raise ExperimentError(f"noncanonical JSONL row {index}: {path}")
        rows.append(copy.deepcopy(value))
    return raw, rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    _require_no_symlink_ancestors(path)
    V1.atomic_bytes(path, payload)
    _require_successor_inode_nonreuse(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, canonical_bytes(dict(value)))


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    supplied = [copy.deepcopy(dict(row)) for row in rows]
    if not supplied:
        raise ExperimentError("ordinary JSONL cannot be empty")
    _atomic_bytes(path, b"".join(canonical_bytes(row) for row in supplied))


def _file_binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    info = _require_regular(path)
    _require_successor_inode_nonreuse(path)
    root = path.parent if relative_to is None else relative_to
    return {
        "path": str(path if relative_to is None else path.relative_to(root)),
        "bytes": int(info.st_size),
        "sha256": sha256_file(path),
        "nlink": int(info.st_nlink),
        "ordinary_regular_file": True,
        "resolved_path_ancestor_symlink_count": 0,
    }


def _payload_binding(path: Path) -> dict[str, Any]:
    return _file_binding(path, relative_to=MATERIAL_ROOT)


def _write_material_shard(
    directory: Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> None:
    """Atomically persist one fresh successor terminal shard and reopen it."""

    import numpy as np

    if directory.exists() or directory.is_symlink():
        raise ExperimentError(f"material shard is not fresh: {directory}")
    _require_no_symlink_ancestors(directory)
    if directory.parent.is_symlink() or not directory.parent.is_dir():
        raise ExperimentError("material shard parent is not an ordinary directory")
    prepared = {
        str(name): np.ascontiguousarray(np.asarray(value))
        for name, value in arrays.items()
    }
    if not prepared or len(prepared) != len(arrays):
        raise ExperimentError("material array inventory is empty or ambiguous")
    directory.mkdir(mode=0o700)
    payload_path = directory / "payload.npz"
    V1.atomic_npz(payload_path, **prepared)
    with zipfile.ZipFile(payload_path, mode="a") as archive:
        archive.comment = CONTRACT.NPZ_ARCHIVE_COMMENT.encode("utf-8")
    with payload_path.open("rb+") as stream:
        os.fsync(stream.fileno())
    _require_successor_inode_nonreuse(payload_path)
    with zipfile.ZipFile(payload_path, mode="r") as archive:
        _call(METRICS, "validate_npz_archive_comment", archive.comment)
    with np.load(payload_path, allow_pickle=False) as archive:
        reopened = {
            name: np.ascontiguousarray(archive[name]) for name in archive.files
        }
    try:
        evidence = _call(
            METRICS,
            "build_persisted_array_evidence",
            shard_kind=str(metadata.get("schema", "unknown")).rsplit(".", 2)[-2],
            shard_id=str(directory.relative_to(MATERIAL_ROOT)),
            payload_file=_payload_binding(payload_path),
            arrays=prepared,
            reopened_arrays=reopened,
        )
    except Exception as exc:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: persisted-array evidence failed"
        ) from exc
    provisional = copy.deepcopy(dict(metadata))
    provisional["persisted_array_evidence"] = evidence
    provisional["payload"] = {
        "role": "material_shard_payload",
        **evidence["payload_file"],
        "kind": "npz",
    }
    _atomic_json(directory / "metadata.json", CONTRACT.attach_content_digest(provisional))
    reopened_metadata, reopened_arrays = _load_material_shard(directory)
    if sorted(reopened_arrays) != sorted(prepared):
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: reopened member inventory drift"
        )
    for name, source in prepared.items():
        target = reopened_arrays[name]
        if (
            source.dtype.str != target.dtype.str
            or source.shape != target.shape
            or source.tobytes(order="C") != target.tobytes(order="C")
        ):
            raise ExperimentError(
                f"STATE_MATERIALISATION_CORRUPT: reopened array drift: {name}"
            )
    if reopened_metadata.get("content_digest") is None:
        raise ExperimentError("STATE_MATERIALISATION_CORRUPT: metadata digest absent")


def _load_material_shard(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    if directory.is_symlink() or not directory.is_dir():
        raise ExperimentError(f"material shard directory is invalid: {directory}")
    metadata = _ordinary_json(directory / "metadata.json")
    _call(CONTRACT, "validate_content_digest", metadata)
    payload_path = directory / "payload.npz"
    binding = _payload_binding(payload_path)
    payload = metadata.get("payload")
    if (
        not isinstance(payload, Mapping)
        or payload.get("role") != "material_shard_payload"
        or payload.get("kind") != "npz"
        or {
            key: payload.get(key)
            for key in CONTRACT.MATERIAL_FILE_BINDING_FIELDS
        }
        != binding
    ):
        raise ExperimentError("STATE_MATERIALISATION_CORRUPT: payload binding drift")
    with zipfile.ZipFile(payload_path, mode="r") as archive:
        _call(METRICS, "validate_npz_archive_comment", archive.comment)
    evidence = metadata.get("persisted_array_evidence")
    if not isinstance(evidence, Mapping):
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: persisted-array evidence absent"
        )
    with np.load(payload_path, allow_pickle=False) as archive:
        arrays = {
            name: np.ascontiguousarray(archive[name]) for name in archive.files
        }
    _call(
        METRICS,
        "validate_persisted_array_evidence",
        evidence,
        reopened_arrays=arrays,
    )
    snapshot = metadata.get("snapshot")
    if isinstance(snapshot, Mapping) and "snapshot__previous_applied_command" in arrays:
        _call(
            METRICS,
            "validate_snapshot_previous_applied_command_binding",
            snapshot,
            evidence,
            reopened_arrays=arrays,
        )
    return metadata, arrays


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ExperimentError(
            f"git {' '.join(arguments)} failed: {exc.output}"
        ) from exc


def _call(module: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        raise ExperimentError(f"required successor API is absent: {name}")
    try:
        return function(*args, **kwargs)
    except ExperimentError:
        raise
    except Exception as exc:
        raise ExperimentError(f"successor API failed: {name}") from exc


def _require_no_external_successor_publication() -> None:
    """Enforce the frozen absence of every successor receipt/bundle sibling."""

    authority = _call(
        CONTRACT,
        "validate_content_digest",
        CONTRACT.PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY,
    )
    paths = [Path(value) for value in CONTRACT.PROHIBITED_EXTERNAL_PUBLICATION_PATHS]
    if (
        authority.get("paths") != [str(path) for path in paths]
        or len(paths) != 3
        or any(
            path.parent != OUTPUT_ROOT.parent
            or not path.name.startswith(f"{OUTPUT_ROOT.name}_")
            for path in paths
        )
    ):
        raise ExperimentError("prohibited external-publication path drift")
    opened = [path for path in paths if path.exists() or path.is_symlink()]
    if opened:
        raise ExperimentError(
            "successor external regeneration/custody publication is prohibited: "
            + ", ".join(str(path) for path in opened)
        )


def stream_index(family: str, stratum_index: int) -> int:
    if family not in FAMILIES:
        raise ExperimentError(f"unknown route family: {family}")
    stratum = int(stratum_index)
    if not 0 <= stratum < STRATA_PER_FAMILY:
        raise ExperimentError("stratum index outside frozen range")
    return FAMILIES.index(family) * STRATA_PER_FAMILY + stratum


def candidate_index(family: str, stratum_index: int, attempt_index: int) -> int:
    attempt = int(attempt_index)
    if not 0 <= attempt < ATTEMPT_CAP:
        raise ExperimentError("attempt index outside frozen stream cap")
    # Round-major registration is frozen so every attempt round covers the
    # complete 64-stream population before the next attempt number.
    return attempt * STREAM_COUNT + stream_index(family, stratum_index)


def _stream_id(family: str, stratum_index: int) -> str:
    return str(
        CONTRACT.build_candidate_spec(family, int(stratum_index), 0)["stream_id"]
    )


def _stream_directory(family: str, stratum_index: int) -> Path:
    return MATERIAL_ROOT / "qualification" / _stream_id(family, stratum_index)


def _candidate_directory(
    family: str, stratum_index: int, attempt_index: int
) -> Path:
    return _stream_directory(family, stratum_index) / f"attempt-{int(attempt_index):02d}"


def _selected_directory(state_id: str) -> Path:
    return MATERIAL_ROOT / "selected" / str(state_id)


def _fanout_directory(state_id: str) -> Path:
    return MATERIAL_ROOT / "fanout" / str(state_id)


def _repeatability_directory(state_id: str) -> Path:
    return MATERIAL_ROOT / "repeatability" / str(state_id)


def _validate_runtime_source_closure() -> dict[str, Any]:
    value = _ordinary_json(DOC_PATHS["source_closure"])
    rows = value.get("rows")
    expected = list(CONTRACT.SOURCE_CLOSURE_PATHS)
    if not isinstance(rows, list) or [row.get("path") for row in rows] != expected:
        raise ExperimentError("successor source-closure row order drift")
    for row in rows:
        path = REPO_ROOT / str(row["path"])
        info = _require_regular(path)
        if (
            int(row.get("bytes", -1)) != int(info.st_size)
            or row.get("sha256") != sha256_file(path)
        ):
            raise ExperimentError(f"successor source-closure drift: {path}")
    _call(CONTRACT, "validate_content_digest", value)
    return value


def require_runtime_source_freeze() -> str:
    head = _git("rev-parse", "HEAD")
    if _git("show", "-s", "--format=%s", head) != FREEZE_SUBJECT:
        raise ExperimentError("runtime HEAD is not the successor source freeze")
    if _git("rev-list", "--parents", "-n", "1", head).split() != [
        head,
        PARENT_COMMIT,
    ]:
        raise ExperimentError("successor freeze is not a direct child of V4 result")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ExperimentError("runtime worktree is not clean at successor freeze")
    for relative in CONTRACT.TRACKED_SOURCE_PATHS:
        path = REPO_ROOT / relative
        _require_regular(path)
        committed = subprocess.check_output(
            ["git", "show", f"{head}:{relative}"], cwd=REPO_ROOT
        )
        if path.read_bytes() != committed:
            raise ExperimentError(f"runtime source differs from HEAD: {relative}")
    _validate_runtime_source_closure()
    return head


def _load_evaluator(module: Any | None = None) -> Any:
    if module is not None:
        return module
    from scripts import evaluate_physical_handoff_stratified_generator_successor_v1 as evaluator

    return evaluator


def _validate_v4_context_files() -> dict[str, Any]:
    """Reopen only the exact named V4 context bindings."""

    context = CONTRACT.validate_v4_context(CONTRACT.build_v4_context())
    for name, binding in context["bindings"].items():
        path = Path(str(binding["path"]))
        info = _require_regular(path)
        if (
            int(info.st_size) != int(binding["bytes"])
            or sha256_file(path) != str(binding["sha256"])
        ):
            raise ExperimentError(f"bound V4 {name} bytes/hash drift")
        expected_digest = binding.get("content_digest")
        if expected_digest is not None:
            value = _ordinary_json(path)
            CONTRACT.V4.validate_content_digest(value)
            if value.get("content_digest") != expected_digest:
                raise ExperimentError(f"bound V4 {name} content digest drift")
    if (
        _git("show", "-s", "--format=%s", CONTRACT.V4_RESULT_COMMIT)
        != CONTRACT.V4.RESULT_COMMIT_SUBJECT
        or _git(
            "rev-list", "--parents", "-n", "1", CONTRACT.V4_RESULT_COMMIT
        ).split()
        != [CONTRACT.V4_RESULT_COMMIT, CONTRACT.V4_SOURCE_FREEZE_COMMIT]
        or _git("show", "-s", "--format=%s", CONTRACT.V4_SOURCE_FREEZE_COMMIT)
        != CONTRACT.V4.CONTRACT_FREEZE_COMMIT_SUBJECT
        or _git(
            "rev-list", "--parents", "-n", "1", CONTRACT.V4_SOURCE_FREEZE_COMMIT
        ).split()
        != [CONTRACT.V4_SOURCE_FREEZE_COMMIT, CONTRACT.V4_SOURCE_PARENT_COMMIT]
        or _git("rev-parse", f"{CONTRACT.V4_RESULT_COMMIT}^{{tree}}")
        != context["v4_result_and_freeze_tree_oid"]
        or _git("rev-parse", f"{CONTRACT.V4_SOURCE_FREEZE_COMMIT}^{{tree}}")
        != context["v4_result_and_freeze_tree_oid"]
    ):
        raise ExperimentError("V4 freeze/result Git binding drift")
    return context


def _sorted_projection_sha256(values: Sequence[str | int]) -> str:
    return hashlib.sha256(
        CONTRACT.canonical_json_bytes(sorted(set(values)))[:-1]
    ).hexdigest()


def _predecessor_identity_registries() -> tuple[
    dict[str, list[int]], dict[str, set[Any]], dict[str, Any]
]:
    """Reconstruct every bound predecessor identity class before simulation."""

    from scripts import run_occluded_goal_topological_belief_v1 as OGTB

    documents = OGTB._load_bound_prior_authorities()  # noqa: SLF001
    prior = OGTB._project_prior_identities(documents)  # noqa: SLF001
    inherited = sorted(int(value) for value in prior["numeric_seed"])
    v2_binding = CONTRACT.V4.PRIOR_SCENE_EXCLUSION_AUTHORITY["v2_panel_binding"]
    v2_path = Path(str(v2_binding["path"]))
    info = _require_regular(v2_path)
    if (
        int(info.st_size) != int(v2_binding["bytes"])
        or sha256_file(v2_path) != str(v2_binding["sha256"])
    ):
        raise ExperimentError("bound predecessor V2 panel drift")
    try:
        v2_panel = json.loads(v2_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise ExperimentError("bound predecessor V2 panel is invalid JSON") from exc
    stack: list[Any] = [v2_panel]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            scene = value.get("scene_id")
            if isinstance(scene, str) and scene:
                prior["scene_identity"].add(scene)
                prior["scene_identity_sha256"].add(
                    hashlib.sha256(scene.encode("utf-8")).hexdigest()
                )
            for field in ("state_id", "episode_id", "graph_id"):
                identity = value.get(field)
                if isinstance(identity, str) and identity:
                    prior["episode_or_state_identity"].add(identity)
            seed = value.get("procedural_seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                prior["numeric_seed"].add(seed)
            for field in (
                "episode_path_id",
                "geometry_digest",
                "geometry_sha256",
                "scene_dir",
                "source_path",
            ):
                identity = value.get(field)
                if isinstance(identity, str) and identity:
                    prior["textual_path_geometry_or_source_identity"].add(identity)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    broad = CONTRACT.V4.PRIOR_SCENE_EXCLUSION_AUTHORITY[
        "broad_prior_plus_v2_projection"
    ]
    projection_checks = (
        (
            "scene_identity",
            "scene_identity_count",
            "scene_identity_canonical_json_sha256",
            False,
        ),
        (
            "scene_identity_sha256",
            "scene_identity_sha256_count",
            "scene_identity_sha256_canonical_json_sha256",
            False,
        ),
        (
            "episode_or_state_identity",
            "episode_state_or_graph_identity_count",
            "episode_state_or_graph_identity_canonical_json_sha256",
            False,
        ),
        (
            "numeric_seed",
            "numeric_seed_count",
            "numeric_seed_canonical_json_sha256",
            True,
        ),
        (
            "textual_path_geometry_or_source_identity",
            "textual_path_geometry_or_source_identity_count",
            "textual_path_geometry_or_source_identity_canonical_json_sha256",
            False,
        ),
    )
    for key, count_key, digest_key, numeric in projection_checks:
        values = list(prior[key])
        if (
            len(values) != int(broad[count_key])
            or OGTB._identity_projection_digest(  # noqa: SLF001
                values, numeric=numeric
            )
            != str(broad[digest_key])
        ):
            raise ExperimentError(f"broad predecessor identity drift: {key}")
    structured = list(prior["structured_waypoint_or_sequence_path"])
    if (
        len(structured)
        != int(broad["structured_waypoint_or_sequence_path_count"])
        or OGTB._structured_identity_projection_digest(structured)  # noqa: SLF001
        != str(
            broad[
                "structured_waypoint_or_sequence_path_canonical_json_sha256"
            ]
        )
    ):
        raise ExperimentError("broad predecessor structured identity drift")
    fixtures = V4._nonregistered_family_fixture_specs()  # noqa: SLF001
    registries = {
        "v4_inherited_broad_numeric_seed_registry": inherited,
        "v4_registered_candidate_seeds": sorted(
            int(value) for value in CONTRACT.V4.PROCEDURAL_SEED_VALUES
        ),
        "v4_nonregistered_family_fixture_seeds": sorted(
            int(row["procedural_seed"]) for row in fixtures
        ),
        "v4_broad_predecessor_numeric_seed_registry": sorted(
            int(value) for value in prior["numeric_seed"]
        ),
    }
    projection = {
        key: {
            "count": len(prior[key]),
            "canonical_sorted_unique_no_lf_sha256": _sorted_projection_sha256(
                list(prior[key])
            ),
        }
        for key in (
            "scene_identity",
            "scene_identity_sha256",
            "episode_or_state_identity",
            "numeric_seed",
            "textual_path_geometry_or_source_identity",
        )
    }
    projection["structured_waypoint_or_sequence_path"] = {
        "count": len(structured),
        "canonical_sorted_unique_no_lf_sha256": hashlib.sha256(
            CONTRACT.canonical_json_bytes(
                sorted(
                    {
                        CONTRACT.canonical_json_bytes(list(value))[:-1].decode(
                            "utf-8"
                        )
                        for value in structured
                    }
                )
            )[:-1]
        ).hexdigest(),
        "intersection_not_applied": (
            "structured family and stratum semantics are scientifically invariant"
        ),
    }
    return registries, prior, projection


def _validate_fresh_identity_domain() -> dict[str, Any]:
    registries, broad_prior, broad_projection = _predecessor_identity_registries()
    seed_result = CONTRACT.validate_seed_nonoverlap(registries)
    candidates = CONTRACT.build_candidate_identity_manifest()
    fields = ("candidate_spec_id", "scene_id", "state_id", "episode_id", "graph_id")
    current_by_field = {
        field: {str(row[field]) for row in candidates} for field in fields
    }
    current = set().union(*current_by_field.values())
    if any(
        not identity.startswith(f"{CONTRACT.IDENTITY_NAMESPACE}-")
        or identity.startswith(("pgehq-", "ogtb-"))
        for identity in current
    ):
        raise ExperimentError("successor identity escaped the phsgs-v1 namespace")
    v4_registered = CONTRACT.V4.build_candidate_specs()
    v4_fixtures = V4._nonregistered_family_fixture_specs()  # noqa: SLF001
    prior_specs = [*v4_registered, *v4_fixtures]
    v4_prior = {
        str(row[field])
        for row in prior_specs
        for field in fields
        if field in row
    }
    broad_prior_strings = set().union(
        set(broad_prior["scene_identity"]),
        set(broad_prior["episode_or_state_identity"]),
        set(broad_prior["textual_path_geometry_or_source_identity"]),
    )
    broad_overlap = sorted(current & broad_prior_strings)
    v4_overlap = sorted(current & v4_prior)
    scene_hash_overlap = sorted(
        {
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in current_by_field["scene_id"]
        }
        & set(broad_prior["scene_identity_sha256"])
    )
    if broad_overlap:
        raise ExperimentError("successor identity overlaps a broad predecessor ID")
    if scene_hash_overlap:
        raise ExperimentError("successor scene hash overlaps a predecessor scene")
    if v4_overlap:
        raise ExperimentError("successor identity overlaps a V4 registered/fixture ID")
    fixture_seeds = {
        int(row["procedural_seed"])
        for row in v4_fixtures
    }
    if fixture_seeds & set(CONTRACT.PROCEDURAL_SEED_VALUES):
        raise ExperimentError("successor seed overlaps a V4 family fixture seed")
    projection = {
        "seed_nonoverlap": copy.deepcopy(seed_result),
        "broad_predecessor_identity_projection": broad_projection,
        "v4_registered_and_fixture_identity_count": len(v4_prior),
        "v4_registered_and_fixture_identity_projection_sha256": (
            _sorted_projection_sha256(list(v4_prior))
        ),
        "successor_identity_count": len(current),
        "successor_identity_projection_sha256": _sorted_projection_sha256(
            list(current)
        ),
        "successor_namespace": CONTRACT.IDENTITY_NAMESPACE,
        "successor_namespace_required_prefix": (
            f"{CONTRACT.IDENTITY_NAMESPACE}-"
        ),
        "broad_predecessor_identity_overlap_count": 0,
        "broad_predecessor_scene_hash_overlap_count": 0,
        "v4_registered_and_fixture_identity_overlap_count": 0,
        "v4_fixture_seed_overlap_count": 0,
        "structured_semantic_overlap_is_not_an_identity_gate": True,
        "all_identity_and_seed_overlap_counts_zero": True,
    }
    return copy.deepcopy(
        _call(CONTRACT, "validate_identity_and_seed_nonoverlap", projection)
    )


def _runtime_contract(freeze: str) -> dict[str, Any]:
    runtime = CONTRACT.build_runtime_contract(freeze)
    return CONTRACT.validate_runtime_contract(runtime, source_freeze_commit=freeze)


def _require_initialized() -> tuple[dict[str, Any], dict[str, Any]]:
    _require_no_external_successor_publication()
    if OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir():
        raise ExperimentError("successor official root is absent or invalid")
    if MATERIAL_ROOT.is_symlink() or not MATERIAL_ROOT.is_dir():
        raise ExperimentError("successor material root is absent or invalid")
    current_freeze = require_runtime_source_freeze()
    runtime = CONTRACT.validate_runtime_contract(
        _ordinary_json(OUTPUT_ROOT / "contract.json"),
        source_freeze_commit=current_freeze,
    )
    if runtime["source_freeze_commit"] != current_freeze:
        raise ExperimentError("runtime contract differs from current source freeze")
    context = CONTRACT.validate_v4_context(
        _ordinary_json(OUTPUT_ROOT / "V4_context.json")
    )
    if context != runtime["v4_context"]:
        raise ExperimentError("runtime/V4 context cross-link drift")
    manifest = CONTRACT.validate_generator_stream_manifest(
        _ordinary_json(OUTPUT_ROOT / "generator_stream_manifest.json")
    )
    material = _ordinary_json(MATERIAL_ROOT / "material_contract.json")
    material = _call(CONTRACT, "validate_material_contract", material)
    rebuilt_nonoverlap = _validate_fresh_identity_domain()
    if (
        material.get("experiment_id") != EXPERIMENT_ID
        or material.get("source_freeze_commit") != runtime["source_freeze_commit"]
        or material.get("runtime_contract_content_digest") != runtime["content_digest"]
        or material.get("stream_manifest_content_digest") != manifest["content_digest"]
        or material.get("predecessor_identity_and_seed_nonoverlap")
        != rebuilt_nonoverlap
    ):
        raise ExperimentError("successor material/runtime contract binding drift")
    return copy.deepcopy(runtime), copy.deepcopy(manifest)


def initialize_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    """Create only fresh successor roots after context and identity preflight."""

    if fake_runtime:
        # Test-only injection is available through lower-level stage functions;
        # initialization itself never mislabels a synthetic namespace.
        raise ExperimentError("initialization has no fake-runtime publication mode")
    _require_no_external_successor_publication()
    if any(path.exists() or path.is_symlink() for path in (OUTPUT_ROOT, MATERIAL_ROOT)):
        raise ExperimentError("successor roots must both be absent before initialization")
    context = _validate_v4_context_files()
    nonoverlap = _validate_fresh_identity_domain()
    # Validate the complete bound V4 9-file official, 514-file material, and
    # independent-reducer receipt inventory before creating either successor
    # namespace.  Later successor writes/reopens reuse this exact inode/SHA
    # authority to reject hard-link adoption and physical-shard byte copying.
    _bound_v4_evidence_inodes()
    freeze = require_runtime_source_freeze()
    runtime = _runtime_contract(freeze)
    manifest = CONTRACT.build_generator_stream_manifest()
    OUTPUT_ROOT.mkdir(mode=0o700)
    MATERIAL_ROOT.mkdir(mode=0o700)
    for relative in (
        "qualification",
        "selected",
        "fanout",
        "repeatability",
        "downstream_workspace",
    ):
        (MATERIAL_ROOT / relative).mkdir(mode=0o700)
    _atomic_json(OUTPUT_ROOT / "contract.json", runtime)
    _atomic_json(OUTPUT_ROOT / "V4_context.json", context)
    _atomic_json(OUTPUT_ROOT / "generator_stream_manifest.json", manifest)
    material = _call(
        CONTRACT,
        "build_material_contract",
        freeze,
        runtime["content_digest"],
        nonoverlap,
    )
    _atomic_json(MATERIAL_ROOT / "material_contract.json", material)
    return runtime


def _terminal_record(
    metadata: Mapping[str, Any], *, directory: Path
) -> dict[str, Any]:
    evidence = metadata.get("persisted_array_evidence")
    if not isinstance(evidence, Mapping):
        raise ExperimentError("terminal persisted-array evidence is absent")
    return _call(
        METRICS,
        "build_generator_terminal_record",
        metadata["state_disposition"],
        source_freeze_commit=metadata["source_freeze_commit"],
        runtime_contract_content_digest=metadata["runtime_contract_content_digest"],
        material_metadata_binding=_file_binding(
            directory / "metadata.json", relative_to=MATERIAL_ROOT
        ),
        material_payload_binding=_file_binding(
            directory / "payload.npz", relative_to=MATERIAL_ROOT
        ),
        persisted_array_evidence_sha256=hashlib.sha256(
            CONTRACT.canonical_json_bytes(evidence)[:-1]
        ).hexdigest(),
    )


def _preregistration_text() -> str:
    return f"""# {EXPERIMENT_ID}

Development-only, outcome-observed generator successor to the completed V4
panel-adequacy result at `{CONTRACT.V4_RESULT_COMMIT}`.

Each of the frozen four-family by sixteen-stratum streams receives a fresh,
deterministic sequence of at most {ATTEMPT_CAP} physical candidates.  A stream
stops only immediately after its fourth `QUALIFIED` state or after attempt 63.
It never stops after the first qualified state.  Candidate allocation is
teacher-only and cannot depend on fanout, ranker, or held-out outcomes.

The successor changes only candidate identity, seed allocation, and stream
depth.  Initial-boundary checks, semantic and behavioural snapshot contracts,
teacher routes and qualification, physical contact, deterministic panel
selection, 48/16 role assignment, exact reset, canonical ports, three target
contracts, the twelve-action bank, frozen encoder and frozen current-visual
ranker, metrics, gates, and handoff classifications remain frozen from V4.

Separately, the qualification-candidate implementation lifecycle destroys all
owned Scenes exactly once in reverse creation order, calls the public
`lewm_genesis.scene_builder.shutdown_genesis()` boundary, and lets the next
candidate build reinitialize Genesis from that candidate's frozen seed and
backend.  This lifecycle-only engineering correction changes no physical
value, render, step, evidence, formula, threshold, gate, disposition, or
model.  Its authority content digest is
`{CONTRACT.QUALIFICATION_CANDIDATE_LIFECYCLE_AUTHORITY["content_digest"]}`.

All 64 streams must reach four qualified states before any panel, encoder,
ranker, fanout, or held-out outcome is opened.  Otherwise the generator
terminal is published and downstream evidence remains absent.  No learned
model training is authorized.

`{CONTRACT.ENGINEERING_LOG_PATH}` is an external operator-owned concise log
used only after an adjudicated pre-panel implementation or persistence fault
requires full namespace deletion and deterministic restart.  This direct
runner never creates or appends it on generic exceptions; normal execution
leaves it absent and uses ordinary process stdout/stderr.
"""


def build_freeze_documents() -> dict[str, Any]:
    """Build the seven narrow prospective source-freeze documents."""

    _require_no_external_successor_publication()
    if _git("rev-parse", "HEAD") != PARENT_COMMIT:
        raise ExperimentError("freeze documents require the bound V4 result commit")
    if any(path.exists() or path.is_symlink() for path in (OUTPUT_ROOT, MATERIAL_ROOT)):
        raise ExperimentError("successor runtime roots exist before source freeze")
    context = _validate_v4_context_files()
    scientific = CONTRACT.build_contract()
    contract_document = CONTRACT.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "contract_document.v1"
            ),
            "status": "FROZEN_BEFORE_SUCCESSOR_PHYSICAL_COLLECTION",
            "parent_commit": PARENT_COMMIT,
            "source_baseline_commit": CONTRACT.SOURCE_BASELINE_COMMIT,
            "required_freeze_subject": FREEZE_SUBJECT,
            "required_result_subject": RESULT_SUBJECT,
            "scientific_contract": scientific,
        }
    )
    fixture = CONTRACT.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "fixture.v1"
            ),
            "registered_candidate_outcomes_opened": 0,
            "maximum_candidate_count": POTENTIAL_CANDIDATE_COUNT,
            "stream_count": STREAM_COUNT,
            "target_qualified_per_stream": QUALIFIED_TARGET,
            "stream_stops_after_four_qualified_or_attempt_63": True,
            "first_qualified_does_not_stop_stream": True,
            "all_generator_classification_fixtures_passed": True,
            "material_roundtrip_fixture_passed": True,
            "conditional_downstream_gate_fixture_passed": True,
            "independent_reducer_fixture_passed": True,
            "models_trained": 0,
        }
    )
    output_schema = CONTRACT.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "output_schema.v1"
            ),
            "root": str(OUTPUT_ROOT),
            "material_root": str(MATERIAL_ROOT),
            "success_leaves": list(CONTRACT.SUCCESS_OUTPUT_LEAVES),
            "generator_terminal_leaves": list(
                CONTRACT.GENERATOR_TERMINAL_OUTPUT_LEAVES
            ),
            "downstream_leaves": list(CONTRACT.DOWNSTREAM_OUTPUT_LEAVES),
            "external_regeneration_or_custody_receipt": None,
            "external_operator_engineering_log": {
                "path": str(CONTRACT.ENGINEERING_LOG_PATH),
                "runner_auto_append": False,
                "normal_execution_present": False,
                "ordinary_process_stdout_stderr_remain_authoritative": True,
            },
            "terminal_material_path": (
                "qualification/<stream-id>/attempt-NN/{metadata.json,payload.npz}"
            ),
            "maximum_terminal_material_records": POTENTIAL_CANDIDATE_COUNT,
        }
    )
    invariance = copy.deepcopy(CONTRACT.V4_SCIENTIFIC_INVARIANCE_AUTHORITY)
    documents = (
        (DOC_PATHS["contract"], contract_document),
        (DOC_PATHS["fixture"], fixture),
        (DOC_PATHS["output_schema"], output_schema),
        (DOC_PATHS["scientific_invariance"], invariance),
        (DOC_PATHS["v4_context_binding"], context),
    )
    for path, value in documents:
        _atomic_json(path, value)
    _atomic_bytes(DOC_PATHS["preregistration"], _preregistration_text().encode("utf-8"))
    closure_rows = []
    for relative in CONTRACT.SOURCE_CLOSURE_PATHS:
        path = REPO_ROOT / relative
        info = _require_regular(path)
        closure_rows.append(
            {
                "path": relative,
                "bytes": int(info.st_size),
                "sha256": sha256_file(path),
            }
        )
    closure = CONTRACT.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "source_closure.v1"
            ),
            "parent_commit": PARENT_COMMIT,
            "row_count": len(closure_rows),
            "rows": closure_rows,
        }
    )
    _atomic_json(DOC_PATHS["source_closure"], closure)
    return contract_document


def _state_disposition(
    spec: Mapping[str, Any],
    *,
    stage_reached: str,
    executable_snapshot_exists: bool,
    teacher_executed: bool,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return _call(
        METRICS,
        "build_state_disposition_record",
        spec,
        stage_reached=stage_reached,
        initial_termination_flags=evidence["initial_termination_flags"],
        probe_trial_termination_flags=evidence.get(
            "probe_trial_termination_flags"
        ),
        teacher_termination_flags=evidence.get("teacher_termination_flags"),
        probe_tip_sample_indices=evidence.get("probe_tip_sample_indices"),
        teacher_criteria=evidence.get("teacher_criteria"),
        executable_snapshot_exists=bool(executable_snapshot_exists),
        teacher_executed=bool(teacher_executed),
        snapshot_identity=evidence.get("snapshot_identity"),
        diagnostics_inventory=evidence.get("diagnostics_inventory", ()),
        payload_member_inventory=evidence.get("payload_member_inventory", ()),
        unresolved_state_failure=bool(evidence.get("unresolved_state_failure", False)),
        reset_or_candidate_outcome_opened=False,
    )


def _physical_stage_runtime(
    backend_runtime: Mapping[str, Any], *, fake_runtime: bool
) -> dict[str, Any]:
    try:
        core = V1.require_stage_runtime("physical", fake=fake_runtime)
        return V1._bind_physical_backend_runtime(core, backend_runtime)  # noqa: SLF001
    except V1.ExperimentError as exc:
        raise ExperimentError(str(exc)) from exc


def _persist_terminal(
    spec: Mapping[str, Any],
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    directory = _candidate_directory(
        str(spec["family"]), int(spec["stratum_index"]), int(spec["attempt_index"])
    )
    _write_material_shard(directory, metadata, arrays)
    reopened, reopened_arrays = _load_material_shard(directory)
    try:
        _call(
            METRICS,
            "validate_qualification_material_shard",
            reopened,
            reopened_arrays=reopened_arrays,
            expected_pool_index=int(spec["candidate_index"]),
        )
    except Exception as exc:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: successor terminal validation failed"
        ) from exc
    if (
        reopened.get("candidate_spec") != dict(spec)
        or reopened.get("pool_index") != int(spec["candidate_index"])
    ):
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: successor terminal identity drift"
        )
    return reopened


def _qualify_candidate(
    spec: Mapping[str, Any],
    *,
    runtime_contract: Mapping[str, Any],
    collector: Any,
    fake_runtime: bool,
) -> dict[str, Any]:
    """Execute and persist one fresh candidate under unchanged V4 physics."""

    import numpy as np

    candidate = CONTRACT.validate_candidate_spec(
        spec, expected_candidate_index=int(spec["candidate_index"])
    )
    qualifier = getattr(collector, "qualify_successor_candidate", None)
    if not callable(qualifier):
        raise ExperimentError(
            "physical backend has no successor-candidate qualification method"
        )
    raw = qualifier(copy.deepcopy(candidate))
    if (
        not isinstance(raw, Mapping)
        or raw.get("candidate_spec_id") != candidate["candidate_spec_id"]
    ):
        raise ExperimentError(
            "SNAPSHOT_IDENTITY_REPRODUCTION_FAILURE: backend identity drift"
        )
    mode = raw.get("mode")
    if mode not in {"INITIAL_REJECTION", "PROBE_REJECTION", "TEACHER"}:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: backend terminal mode drift"
        )
    if not isinstance(raw.get("runtime_evidence"), Mapping):
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: backend runtime evidence absent"
        )
    backend_runtime = copy.deepcopy(dict(raw["runtime_evidence"]))
    stage_runtime = _physical_stage_runtime(
        backend_runtime, fake_runtime=fake_runtime
    )
    index = int(candidate["candidate_index"])

    if mode == "INITIAL_REJECTION":
        disposition = str(raw["disposition"])
        if disposition not in {
            "INITIAL_BOUNDARY_TIPPED",
            "UNRESOLVED_STATE_FAILURE",
        }:
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: initial disposition drift"
            )
        arrays = {
            str(name): np.ascontiguousarray(np.asarray(value))
            for name, value in dict(raw["arrays"]).items()
        }
        flags = {
            name: bool(arrays["termination_flags"][offset])
            for offset, name in enumerate(V4.TERMINATION_FLAG_ORDER)
        }
        evidence = {
            "initial_termination_flags": flags,
            "probe_trial_termination_flags": None,
            "teacher_termination_flags": None,
            "probe_tip_sample_indices": None,
            "teacher_criteria": None,
            "snapshot_identity": None,
            "diagnostics_inventory": sorted(arrays),
            "payload_member_inventory": sorted(arrays),
            "unresolved_state_failure": disposition == "UNRESOLVED_STATE_FAILURE",
        }
        state = _state_disposition(
            candidate,
            stage_reached="INITIAL_BOUNDARY",
            executable_snapshot_exists=False,
            teacher_executed=False,
            evidence=evidence,
        )
        metadata = {
            "schema": CONTRACT.QUALIFICATION_MATERIAL_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "source_freeze_commit": runtime_contract["source_freeze_commit"],
            "runtime_contract_content_digest": runtime_contract["content_digest"],
            "pool_index": index,
            "candidate_spec": candidate,
            "candidate_spec_sha256": candidate["canonical_spec_sha256"],
            "disposition": state["disposition"],
            "qualified": False,
            "rejection_reason": state["disposition"],
            "rejection_components": [state["disposition"]],
            "stage_reached": "INITIAL_BOUNDARY",
            "executable_snapshot_exists": False,
            "teacher_executed": False,
            "boundary_evidence": copy.deepcopy(dict(raw["diagnostics"])),
            "snapshot": None,
            "graph": None,
            "teacher": None,
            "state_disposition": state,
            "stage_runtime": stage_runtime,
            "backend_runtime": backend_runtime,
            "reset_or_candidate_outcome_opened": False,
        }
        return _persist_terminal(candidate, metadata, arrays)

    if mode == "PROBE_REJECTION":
        snapshot_metadata = copy.deepcopy(dict(raw["snapshot"]))
        snapshot_arrays = {
            str(name): np.ascontiguousarray(np.asarray(value))
            for name, value in dict(raw["arrays"]).items()
            if name == "snapshot_payload_bytes" or name.startswith("snapshot__")
        }
    else:
        snapshot_arrays, snapshot_metadata = V4._normalise_snapshot(  # noqa: SLF001
            dict(raw["snapshot"])
        )
    initial_sha = str(snapshot_metadata["snapshot_payload_sha256"])
    rgb_source = raw["arrays"]["rgb"] if mode == "PROBE_REJECTION" else raw["rgb"]
    rgb = np.ascontiguousarray(np.asarray(rgb_source, dtype=np.uint8))
    if rgb.shape != (168, 224, 3):
        raise ExperimentError("STATE_MATERIALISATION_CORRUPT: RGB shape drift")
    probe_arrays = (
        dict(raw["arrays"])
        if mode == "PROBE_REJECTION"
        else dict(raw["probe_arrays"])
    )
    if mode == "PROBE_REJECTION":
        probe_arrays.pop("rgb", None)
        for name in snapshot_arrays:
            probe_arrays.pop(name, None)
    probe_arrays = {
        str(name): np.ascontiguousarray(np.asarray(value))
        for name, value in probe_arrays.items()
    }
    probe_metadata = copy.deepcopy(dict(raw["probe_metadata"]))

    if mode == "PROBE_REJECTION":
        disposition = str(raw["disposition"])
        if disposition not in {
            "RESTORATION_PROBE_TIPPED",
            "UNRESOLVED_STATE_FAILURE",
        }:
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: probe disposition drift"
            )
        evidence = {
            "initial_termination_flags": {
                name: False for name in V4.TERMINATION_FLAG_ORDER
            },
            "probe_trial_termination_flags": [
                copy.deepcopy(trial["termination_flags"])
                for trial in probe_metadata["behavioural_probe"]["trials"]
            ],
            "teacher_termination_flags": None,
            "probe_tip_sample_indices": [
                int(trial["tip_sample_index"])
                for trial in probe_metadata["behavioural_probe"]["trials"]
            ],
            "teacher_criteria": None,
            "snapshot_identity": copy.deepcopy(
                probe_metadata["snapshot_identity"]
            ),
            "diagnostics_inventory": [],
            "payload_member_inventory": sorted(
                {"rgb", *snapshot_arrays, *probe_arrays}
            ),
            "unresolved_state_failure": disposition == "UNRESOLVED_STATE_FAILURE",
        }
        state = _state_disposition(
            candidate,
            stage_reached="RESTORATION_PROBE",
            executable_snapshot_exists=True,
            teacher_executed=False,
            evidence=evidence,
        )
        metadata = {
            "schema": CONTRACT.QUALIFICATION_MATERIAL_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "source_freeze_commit": runtime_contract["source_freeze_commit"],
            "runtime_contract_content_digest": runtime_contract["content_digest"],
            "pool_index": index,
            "candidate_spec": candidate,
            "candidate_spec_sha256": candidate["canonical_spec_sha256"],
            "initial_decision_state_sha256": initial_sha,
            "snapshot": snapshot_metadata,
            **probe_metadata,
            "current_rgb_sha256": V1.canonical_array_sha256(rgb),
            "disposition": state["disposition"],
            "qualified": False,
            "rejection_reason": state["disposition"],
            "rejection_components": [state["disposition"]],
            "stage_reached": "RESTORATION_PROBE",
            "executable_snapshot_exists": True,
            "teacher_executed": False,
            "graph": None,
            "teacher": None,
            "state_disposition": state,
            "stage_runtime": stage_runtime,
            "backend_runtime": backend_runtime,
            "reset_or_candidate_outcome_opened": False,
        }
        return _persist_terminal(
            candidate,
            metadata,
            {"rgb": rgb, **snapshot_arrays, **probe_arrays},
        )

    teacher_flags = {
        name: bool(raw["teacher_termination_flags"][name])
        for name in V4.TERMINATION_FLAG_ORDER
    }
    if raw.get("teacher_completed_without_termination") is False:
        teacher = V4._normalise_partial_teacher_trace(  # noqa: SLF001
            dict(raw["teacher_trace"]), termination_flags=teacher_flags
        )
    else:
        teacher = V1._normalise_trace(  # noqa: SLF001
            dict(raw["teacher_trace"]), kind="teacher", exact_samples=None
        )
    if raw["initial_decision_state_sha256"] != initial_sha:
        raise ExperimentError("teacher initial state differs from captured snapshot")
    science_teacher = V4._teacher_science_trace(  # noqa: SLF001
        teacher,
        spec=candidate,
        snapshot_base_pose_world=snapshot_arrays["snapshot__base_pose_world"],
        termination_flags=teacher_flags,
    )
    science_poses = science_teacher["base_pose_world"]
    graph = copy.deepcopy(dict(raw["graph"]))
    selected_edge = copy.deepcopy(
        dict(candidate["geometry"]["selected_directed_edge"])
    )
    competing_edges = [
        copy.deepcopy(dict(value))
        for value in candidate["geometry"]["competing_directed_edges"]
    ]
    crossing_error: str | None = None
    if len(science_poses) < 2:
        crossing = None
        crossing_error = "teacher trace never crosses the canonical directed port"
    else:
        try:
            crossing = V1.canonical_port_crossing(
                science_poses,
                science_teacher["target_region_member"],
                selected_edge["opening_segment_world"],
                selected_edge["opening_normal_world"],
                competing_edges,
            )
            crossing = V4._canonicalize_v4_crossing_lateral_coordinate(  # noqa: SLF001
                crossing, selected_edge
            )
        except V1.ExperimentError as exc:
            crossing = None
            crossing_error = str(exc)
    contact_free = not bool(teacher["physics_contact"].any())
    left_source = bool((science_teacher["source_region_member"] == 0).any())
    competing_crossing = V1._first_competing_crossing(  # noqa: SLF001
        science_poses, competing_edges
    )
    competing_crossing = V4._canonicalize_v4_competing_crossing_lateral_coordinate(  # noqa: SLF001
        competing_crossing, competing_edges
    )
    competing = bool(
        competing_crossing is not None
        and (
            crossing is None
            or (
                int(competing_crossing["sample_after"]),
                float(competing_crossing["fraction"]),
                str(competing_crossing["edge_id"]),
            )
            <= (
                int(crossing["sample_after"]),
                float(crossing["fraction"]),
                str(selected_edge["edge_id"]),
            )
        )
    )
    route_progress = V1._teacher_route_progress_m(  # noqa: SLF001
        science_poses, selected_edge["opening_segment_world"]
    )
    positive = route_progress > 0.0
    goal_reachable = bool(graph.get("goal_reachable", False))
    physically_executable = bool(
        graph.get("graph_edge_physically_executable", False)
    )
    disposition, components = V4._teacher_disposition(  # noqa: SLF001
        contact_free=contact_free,
        crossing=crossing,
        left_source=left_source,
        positive_progress=positive,
        competing_port_entered=competing,
        goal_reachable=goal_reachable,
        physically_executable=physically_executable,
        termination_flags=teacher_flags,
    )
    contact = copy.deepcopy(dict(raw["contact_instrumentation"]))
    if (
        contact.get("api") != "robot.get_contacts"
        or contact.get("sample_period_s") != V4.TRACE_DT_S
        or contact.get("forbidden_net_force_api_used") is not False
        or contact.get("ontology_sha256")
        != CONTRACT.V4.CONTACT_AUTHORITY["ontology_sha256"]
    ):
        raise ExperimentError("teacher contact instrumentation drift")
    if (
        backend_runtime.get("snapshot_captured_before_teacher") is not True
        or backend_runtime.get("teacher_restored_from_serialized_snapshot") is not True
        or backend_runtime.get("teacher_snapshot_sha256") != initial_sha
    ):
        raise ExperimentError("teacher did not execute from the bound snapshot")
    teacher_valid = bool(
        crossing is not None
        and contact_free
        and left_source
        and positive
        and not competing
        and physically_executable
    )
    teacher_metadata = {
        "sample_count": int(len(teacher["timestamp_s"])),
        "trace_digests": V1._trace_digest_projection(teacher),  # noqa: SLF001
        "contact_free": contact_free,
        "left_source_region": left_source,
        "positive_route_progress": positive,
        "route_progress_m": route_progress,
        "competing_port_entered": competing,
        "competing_crossing": competing_crossing,
        "reached_target_node": bool(science_teacher["target_region_member"].any()),
        "crossing": crossing,
        "crossing_error": crossing_error,
        "termination_flags": teacher_flags,
        "terminated_unsafe": any(teacher_flags.values()),
        "teacher_valid": teacher_valid,
    }
    criteria = {
        "goal_reachable": goal_reachable,
        "teacher_trace_contact_free": contact_free,
        "teacher_left_source_region": left_source,
        "teacher_crossed_directed_port": crossing is not None,
        "teacher_positive_route_progress": positive,
        "teacher_no_competing_port": not competing,
        "teacher_normal_positive": bool(
            crossing and crossing["normal_dot_displacement_m"] > 0
        ),
        "teacher_within_lateral_bounds": bool(
            crossing
            and abs(crossing["lateral_coordinate_m"])
            <= float(candidate["passage_width_m"]) / 2.0 + 1e-9
        ),
        "teacher_dwell_satisfied": bool(
            crossing and crossing["sustained_or_target_reached"]
        ),
        "directed_port_defined": crossing is not None,
        "current_rgb_valid": True,
        "graph_edge_physically_executable": physically_executable,
    }
    evidence = {
        "initial_termination_flags": {
            name: False for name in V4.TERMINATION_FLAG_ORDER
        },
        "probe_trial_termination_flags": [
            {name: False for name in V4.TERMINATION_FLAG_ORDER} for _ in range(2)
        ],
        "teacher_termination_flags": teacher_flags,
        "probe_tip_sample_indices": [None, None],
        "teacher_criteria": criteria,
        "snapshot_identity": copy.deepcopy(probe_metadata["snapshot_identity"]),
        "diagnostics_inventory": [],
        "payload_member_inventory": sorted(
            {
                "rgb",
                *snapshot_arrays,
                *probe_arrays,
                *(f"teacher__{name}" for name in teacher),
            }
        ),
    }
    terminal_stage = "COMPLETE" if disposition == "QUALIFIED" else "TEACHER_EXECUTION"
    state = _state_disposition(
        candidate,
        stage_reached=terminal_stage,
        executable_snapshot_exists=True,
        teacher_executed=True,
        evidence=evidence,
    )
    arrays = {
        "rgb": rgb,
        **snapshot_arrays,
        **probe_arrays,
        **{f"teacher__{name}": value for name, value in teacher.items()},
    }
    metadata = {
        "schema": CONTRACT.QUALIFICATION_MATERIAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "source_freeze_commit": runtime_contract["source_freeze_commit"],
        "runtime_contract_content_digest": runtime_contract["content_digest"],
        "pool_index": index,
        "candidate_spec": candidate,
        "candidate_spec_sha256": candidate["canonical_spec_sha256"],
        "initial_decision_state_sha256": initial_sha,
        "snapshot": snapshot_metadata,
        **probe_metadata,
        "graph": graph,
        "teacher": teacher_metadata,
        "current_rgb_sha256": V1.canonical_array_sha256(rgb),
        "goal_reachable": goal_reachable,
        "graph_edge_physically_executable": physically_executable,
        "disposition": state["disposition"],
        "qualified": state["qualified"],
        "rejection_reason": None if state["qualified"] else state["disposition"],
        "rejection_components": components,
        "stage_reached": terminal_stage,
        "executable_snapshot_exists": True,
        "teacher_executed": True,
        "contact_instrumentation": contact,
        "state_disposition": state,
        "stage_runtime": stage_runtime,
        "backend_runtime": backend_runtime,
        "reset_or_candidate_outcome_opened": False,
    }
    return _persist_terminal(candidate, metadata, arrays)


def _stream_completion_document(
    runtime: Mapping[str, Any],
    family: str,
    stratum_index: int,
    terminal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts = len(terminal_rows)
    expected_indices = [
        candidate_index(family, stratum_index, attempt)
        for attempt in range(attempts)
    ]
    observed_indices = [int(row["candidate_index"]) for row in terminal_rows]
    qualified = sum(bool(row["qualified"]) for row in terminal_rows)
    if (
        not 1 <= attempts <= ATTEMPT_CAP
        or observed_indices != expected_indices
        or qualified > QUALIFIED_TARGET
        or any(
            row.get("hard_stop") is not False
            or row.get("continuation_authorized") is not True
            for row in terminal_rows
        )
        or (
            qualified == QUALIFIED_TARGET
            and (
                not bool(terminal_rows[-1]["qualified"])
                or sum(bool(row["qualified"]) for row in terminal_rows[:-1])
                != QUALIFIED_TARGET - 1
            )
        )
        or (qualified < QUALIFIED_TARGET and attempts != ATTEMPT_CAP)
    ):
        raise ExperimentError("stream terminal prefix/stop authority drift")
    return CONTRACT.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "stream_completion.v1"
            ),
            "experiment_id": EXPERIMENT_ID,
            "stream_index": stream_index(family, stratum_index),
            "stream_id": _stream_id(family, stratum_index),
            "family": family,
            "stratum_index": int(stratum_index),
            "attempt_count": attempts,
            "qualified_count": qualified,
            "terminal_candidate_indices": observed_indices,
            "termination_reason": (
                "TARGET_QUALIFIED_REACHED"
                if qualified == QUALIFIED_TARGET
                else "ATTEMPT_LIMIT_REACHED"
            ),
            "source_freeze_commit": runtime["source_freeze_commit"],
            "runtime_contract_content_digest": runtime["content_digest"],
        }
    )


def _validate_stream_completion_document(
    value: Mapping[str, Any],
    runtime: Mapping[str, Any],
    family: str,
    stratum_index: int,
    terminal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    supplied = CONTRACT.validate_content_digest(value)
    expected = _stream_completion_document(
        runtime, family, stratum_index, terminal_rows
    )
    if canonical_bytes(supplied) != canonical_bytes(expected):
        raise ExperimentError(
            "stream completion differs from reopened terminal prefix"
        )
    return copy.deepcopy(supplied)


def qualify_stream_stage(
    family: str,
    stratum_index: int,
    *,
    backend: Any | None = None,
    fake_runtime: bool = False,
) -> dict[str, Any]:
    """Execute one stream sequentially; never resume a partial stream."""

    runtime, _manifest = _require_initialized()
    index = stream_index(family, stratum_index)
    directory = _stream_directory(family, stratum_index)
    if directory.exists() or directory.is_symlink():
        raise ExperimentError("qualification stream is not fresh")
    directory.mkdir(mode=0o700)
    collector = GenesisSuccessorPhysicalBackend() if backend is None else backend
    terminal_rows: list[dict[str, Any]] = []
    qualified = 0
    for attempt in range(ATTEMPT_CAP):
        spec = CONTRACT.build_candidate_spec(family, int(stratum_index), attempt)
        global_index = candidate_index(family, int(stratum_index), attempt)
        if spec["candidate_index"] != global_index or spec["stream_index"] != index:
            raise ExperimentError("contract candidate coordinate drift")
        metadata = _qualify_candidate(
            spec,
            runtime_contract=runtime,
            collector=collector,
            fake_runtime=fake_runtime,
        )
        terminal_rows.append(
            _terminal_record(
                metadata,
                directory=_candidate_directory(family, stratum_index, attempt),
            )
        )
        disposition = metadata.get("state_disposition")
        if (
            not isinstance(disposition, Mapping)
            or not isinstance(disposition.get("hard_stop"), bool)
            or not isinstance(disposition.get("continuation_authorized"), bool)
        ):
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: terminal continuation authority absent"
            )
        if disposition["hard_stop"] or not disposition["continuation_authorized"]:
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: hard terminal disposition stops "
                "qualification stream"
            )
        qualified += int(metadata["qualified"])
        if qualified == QUALIFIED_TARGET:
            break
    summary = _stream_completion_document(
        runtime, family, int(stratum_index), terminal_rows
    )
    _atomic_json(directory / "stream_completion.json", summary)
    return summary


def reduce_generator_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    """Reopen all 64 completed streams and publish the generator gate."""

    runtime, manifest = _require_initialized()
    terminal_path = OUTPUT_ROOT / "generator_terminal_records.jsonl"
    metrics_path = OUTPUT_ROOT / "generator_metrics.json"
    if any(path.exists() or path.is_symlink() for path in (terminal_path, metrics_path)):
        raise ExperimentError("generator population is already frozen")
    records: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for stratum in range(STRATA_PER_FAMILY):
            stream_dir = _stream_directory(family, stratum)
            completion = _ordinary_json(stream_dir / "stream_completion.json")
            if (
                completion.get("stream_index") != stream_index(family, stratum)
                or completion.get("stream_id") != _stream_id(family, stratum)
                or completion.get("source_freeze_commit")
                != runtime["source_freeze_commit"]
                or completion.get("runtime_contract_content_digest")
                != runtime["content_digest"]
            ):
                raise ExperimentError("stream completion identity drift")
            CONTRACT.validate_content_digest(completion)
            attempts = int(completion["attempt_count"])
            if not 1 <= attempts <= ATTEMPT_CAP:
                raise ExperimentError("stream completion attempt count drift")
            stream_records: list[dict[str, Any]] = []
            for attempt in range(attempts):
                expected_index = candidate_index(family, stratum, attempt)
                candidate_dir = _candidate_directory(family, stratum, attempt)
                metadata, arrays = _load_material_shard(candidate_dir)
                try:
                    _call(
                        METRICS,
                        "validate_qualification_material_shard",
                        metadata,
                        reopened_arrays=arrays,
                        expected_pool_index=expected_index,
                    )
                except Exception as exc:
                    raise ExperimentError(
                        "STATE_MATERIALISATION_CORRUPT: generator terminal shard invalid"
                    ) from exc
                record = _terminal_record(metadata, directory=candidate_dir)
                records.append(record)
                stream_records.append(record)
                metadata_rows.append(metadata)
            _validate_stream_completion_document(
                completion, runtime, family, stratum, stream_records
            )
            if attempts < ATTEMPT_CAP and _candidate_directory(
                family, stratum, attempts
            ).exists():
                raise ExperimentError("stream contains evidence after its stop boundary")
    records.sort(key=lambda row: int(row["candidate_index"]))
    metadata_rows.sort(key=lambda row: int(row["pool_index"]))
    try:
        generator_runtime = _call(
            METRICS,
            "build_generator_runtime_environment",
            metadata_rows,
            runtime,
            allow_fake_runtime=fake_runtime,
        )
        _call(
            METRICS,
            "validate_generator_runtime_environment",
            generator_runtime,
            runtime,
            metadata_rows=metadata_rows,
            allow_fake_runtime=fake_runtime,
        )
        ledger = _call(METRICS, "build_generator_terminal_records_jsonl", records)
        metrics = _call(METRICS, "build_generator_metrics", records)
        _call(METRICS, "validate_generator_metrics", metrics, records)
    except Exception as exc:
        raise ExperimentError("generator population reduction failed") from exc
    if manifest["content_digest"] != metrics["stream_manifest_content_digest"]:
        raise ExperimentError("generator metric/manifest cross-link drift")
    _atomic_json(MATERIAL_ROOT / "generator_runtime_environment.json", generator_runtime)
    _atomic_bytes(terminal_path, ledger)
    _atomic_json(metrics_path, metrics)
    return copy.deepcopy(metrics)


def _load_generator_gate() -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]
]:
    runtime, _manifest = _require_initialized()
    raw, supplied = _ordinary_jsonl(
        OUTPUT_ROOT / "generator_terminal_records.jsonl"
    )
    records = _call(METRICS, "validate_generator_terminal_records_jsonl", raw)
    if not supplied:  # pragma: no cover - framing guard above
        raise ExperimentError("generator terminal population is empty")
    generator = _call(
        METRICS,
        "validate_generator_metrics",
        _ordinary_json(OUTPUT_ROOT / "generator_metrics.json"),
        records,
    )
    handoff = _call(METRICS, "build_panel_handoff", records, generator)
    _call(METRICS, "validate_panel_handoff", handoff, records, generator)
    return runtime, records, generator, handoff


def _panel_ordered_selected_specs(
    handoff: Mapping[str, Any]
) -> list[dict[str, Any]]:
    specs = [dict(value) for value in handoff["selected_candidate_specs"]]
    if len(specs) != CONTRACT.PANEL_STATE_COUNT:
        raise ExperimentError("selected panel cardinality drift")
    specs.sort(key=lambda value: str(value["state_id"]))
    if len({str(value["state_id"]) for value in specs}) != len(specs):
        raise ExperimentError("selected panel state identity collision")
    return specs


def _load_qualification_material_validations(
    records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    for record in records:
        spec = CONTRACT.build_candidate_spec(
            str(record["family"]),
            int(record["stratum_index"]),
            int(record["attempt_index"]),
        )
        directory = _candidate_directory(
            str(spec["family"]), int(spec["stratum_index"]), int(spec["attempt_index"])
        )
        metadata, arrays = _load_material_shard(directory)
        validation = _call(
            METRICS,
            "validate_qualification_material_shard",
            metadata,
            reopened_arrays=arrays,
            expected_pool_index=int(record["candidate_index"]),
        )
        if (
            record["material_metadata_binding"]
            != _file_binding(directory / "metadata.json", relative_to=MATERIAL_ROOT)
            or record["material_payload_binding"]
            != _file_binding(directory / "payload.npz", relative_to=MATERIAL_ROOT)
        ):
            raise ExperimentError("qualification terminal/material binding drift")
        validations.append(validation)
    return validations


def capture_selected_state_stage(
    state_id: str,
    *,
    backend: Any | None = None,
    fake_runtime: bool = False,
) -> dict[str, Any]:
    """Persist two exact restores for one selected successor snapshot."""

    import numpy as np

    runtime, records, generator, handoff = _load_generator_gate()
    if generator["status"] != CONTRACT.GENERATOR_PANEL_AVAILABLE:
        raise ExperimentError("selected reset is forbidden for generator terminal")
    selected_specs = _panel_ordered_selected_specs(handoff)
    by_state = {str(spec["state_id"]): (index, spec) for index, spec in enumerate(selected_specs)}
    if str(state_id) not in by_state:
        raise ExperimentError("selected reset requested for a non-panel state")
    panel_index, spec = by_state[str(state_id)]
    source_directory = _candidate_directory(
        str(spec["family"]), int(spec["stratum_index"]), int(spec["attempt_index"])
    )
    qualification_metadata, qualification_arrays = _load_material_shard(
        source_directory
    )
    _call(
        METRICS,
        "validate_qualification_material_shard",
        qualification_metadata,
        reopened_arrays=qualification_arrays,
        expected_pool_index=int(spec["candidate_index"]),
    )
    source_records = [
        row
        for row in records
        if int(row["candidate_index"]) == int(spec["candidate_index"])
    ]
    if (
        len(source_records) != 1
        or source_records[0]["material_metadata_binding"]
        != _file_binding(
            source_directory / "metadata.json", relative_to=MATERIAL_ROOT
        )
        or source_records[0]["material_payload_binding"]
        != _file_binding(
            source_directory / "payload.npz", relative_to=MATERIAL_ROOT
        )
    ):
        raise ExperimentError("selected source terminal/ledger binding drift")
    if (
        qualification_metadata.get("qualified") is not True
        or qualification_metadata.get("candidate_spec") != spec
    ):
        raise ExperimentError("selected snapshot is not qualified source evidence")
    snapshot_payload = bytes(
        np.asarray(
            qualification_arrays["snapshot_payload_bytes"], dtype=np.uint8
        ).tobytes(order="C")
    )
    snapshot_sha = hashlib.sha256(snapshot_payload).hexdigest()
    snapshot_identity = qualification_metadata.get("snapshot_identity")
    if (
        snapshot_sha != qualification_metadata.get("initial_decision_state_sha256")
        or not isinstance(snapshot_identity, Mapping)
        or snapshot_sha != snapshot_identity.get("artifact_file_sha256")
    ):
        raise ExperimentError("selected reset snapshot artifact binding drift")
    collector = GenesisSuccessorPhysicalBackend() if backend is None else backend
    raw = collector.reset_fixture(copy.deepcopy(spec), snapshot_payload)
    if not isinstance(raw, Mapping) or set(raw) != {
        "candidate_spec_id",
        "snapshot_payload_sha256",
        "reset_trials",
        "runtime_evidence",
    }:
        raise ExperimentError("selected reset backend result field drift")
    if (
        raw["candidate_spec_id"] != spec["candidate_spec_id"]
        or raw["snapshot_payload_sha256"] != snapshot_sha
    ):
        raise ExperimentError("selected reset did not restore the source snapshot")
    backend_runtime = copy.deepcopy(dict(raw["runtime_evidence"]))
    stage_runtime = _physical_stage_runtime(
        backend_runtime, fake_runtime=fake_runtime
    )
    arrays: dict[str, Any] = {
        "snapshot_payload_bytes": np.ascontiguousarray(
            qualification_arrays["snapshot_payload_bytes"]
        ),
        "rgb": np.ascontiguousarray(qualification_arrays["rgb"]),
    }
    for field in V1.SNAPSHOT_NUMERIC_FIELDS:
        arrays[f"snapshot__{field}"] = np.ascontiguousarray(
            qualification_arrays[f"snapshot__{field}"]
        )
    trials = [dict(value) for value in raw["reset_trials"]]
    if len(trials) != 2:
        raise ExperimentError("selected reset requires exactly two trials")
    trace_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for trial_index, trial in enumerate(trials):
        if set(trial) != {"metadata", "trace"}:
            raise ExperimentError("selected reset trial wrapper drift")
        trace = V1._normalise_trace(  # noqa: SLF001
            dict(trial["trace"]),
            kind="candidate",
            exact_samples=V1.RESET_FIXTURE_PHYSICS_STEPS,
        )
        metadata = copy.deepcopy(dict(trial["metadata"]))
        if (
            metadata.get("serialized_restore_used") is not True
            or metadata.get("clone_equivalence_used") is not False
            or metadata.get("restored_snapshot_sha256") != snapshot_sha
        ):
            raise ExperimentError("selected reset trial restoration drift")
        metadata.pop("exact_match", None)
        metadata["trace_digests"] = V1._trace_digest_projection(trace)  # noqa: SLF001
        trial_rows.append(metadata)
        trace_rows.append(trace)
        for member, value in trace.items():
            arrays[f"fixture_{trial_index}__{member}"] = value
    pair_matches, maxima = V1._reset_trace_pair_matches(  # noqa: SLF001
        trace_rows[0], trace_rows[1]
    )
    pair = V1._reset_pair_comparison(  # noqa: SLF001
        str(state_id),
        trace_rows[0],
        trace_rows[1],
        first_termination_reason=str(trial_rows[0]["termination_reason"]),
        second_termination_reason=str(trial_rows[1]["termination_reason"]),
        first_stuck=bool(trial_rows[0]["stuck"]),
        second_stuck=bool(trial_rows[1]["stuck"]),
    )
    if not pair_matches or pair.get("passed") is not True:
        raise ExperimentError(f"selected exact-reset mismatch: {maxima}")
    metadata = {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "selected_state_material.v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "source_freeze_commit": runtime["source_freeze_commit"],
        "runtime_contract_content_digest": runtime["content_digest"],
        "panel_index": panel_index,
        "qualification_candidate_index": int(spec["candidate_index"]),
        "state_id": str(state_id),
        "candidate_spec": spec,
        "source_terminal_metadata_binding": _file_binding(
            source_directory / "metadata.json", relative_to=MATERIAL_ROOT
        ),
        "source_terminal_payload_binding": _file_binding(
            source_directory / "payload.npz", relative_to=MATERIAL_ROOT
        ),
        "snapshot": copy.deepcopy(dict(qualification_metadata["snapshot"])),
        "snapshot_identity": copy.deepcopy(dict(snapshot_identity)),
        "snapshot_payload_sha256": snapshot_sha,
        "reset_trials": trial_rows,
        "reset_pair_comparison": pair,
        "reset_fixture_passed": True,
        "rgb_sha256": V1.canonical_array_sha256(arrays["rgb"]),
        "stage_runtime": stage_runtime,
        "backend_runtime": backend_runtime,
        "fanout_or_ranker_outcome_opened": False,
        "models_trained": 0,
    }
    directory = _selected_directory(str(state_id))
    _write_material_shard(directory, metadata, arrays)
    reopened, reopened_arrays = _load_material_shard(directory)
    _call(
        METRICS,
        "validate_selected_reset_material_shard",
        reopened,
        reopened_arrays=reopened_arrays,
        selected_candidate_spec=spec,
        expected_panel_index=panel_index,
        material_metadata_binding=_file_binding(
            directory / "metadata.json", relative_to=MATERIAL_ROOT
        ),
    )
    return reopened


def freeze_panel_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    """Freeze the successor panel after all 64 reset pairs pass."""

    runtime, records, generator, handoff = _load_generator_gate()
    if generator["status"] != CONTRACT.GENERATOR_PANEL_AVAILABLE:
        raise ExperimentError("panel freeze is forbidden for generator terminal")
    selected_specs = _panel_ordered_selected_specs(handoff)
    selected_material: list[dict[str, Any]] = []
    for panel_index, spec in enumerate(selected_specs):
        metadata, arrays = _load_material_shard(
            _selected_directory(str(spec["state_id"]))
        )
        validation = _call(
            METRICS,
            "validate_selected_reset_material_shard",
            metadata,
            reopened_arrays=arrays,
            selected_candidate_spec=spec,
            expected_panel_index=panel_index,
            material_metadata_binding=_file_binding(
                _selected_directory(str(spec["state_id"])) / "metadata.json",
                relative_to=MATERIAL_ROOT,
            ),
        )
        selected_material.append(validation)
    qualification_material = _load_qualification_material_validations(records)
    documents = _call(
        METRICS,
        "build_frozen_panel_documents",
        records,
        generator,
        handoff,
        qualification_material,
        selected_material,
        runtime,
        allow_fake_runtime=fake_runtime,
    )
    _call(
        METRICS,
        "validate_frozen_panel_documents",
        documents,
        records,
        generator,
        handoff,
        qualification_material,
        selected_material,
        runtime,
        allow_fake_runtime=fake_runtime,
    )
    expected = {
        "panel_manifest.json",
        "split_manifest.json",
        "state_snapshot_index.json",
        "teacher_trace_index.json",
        "edge_port_index.json",
        "target_contracts.json",
    }
    if set(documents) != expected:
        raise ExperimentError("frozen panel document inventory drift")
    for leaf in CONTRACT.DOWNSTREAM_OUTPUT_LEAVES:
        if (OUTPUT_ROOT / leaf).exists():
            raise ExperimentError("downstream output exists before panel freeze")
    for leaf in (
        "panel_manifest.json",
        "split_manifest.json",
        "state_snapshot_index.json",
        "teacher_trace_index.json",
        "edge_port_index.json",
        "target_contracts.json",
    ):
        _atomic_json(OUTPUT_ROOT / leaf, documents[leaf])
    return copy.deepcopy(documents["panel_manifest.json"])


def _load_frozen_panel_evidence(
    *, fake_runtime: bool
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    runtime, records, generator, handoff = _load_generator_gate()
    if generator["status"] != CONTRACT.GENERATOR_PANEL_AVAILABLE:
        raise ExperimentError("downstream stage is forbidden for generator terminal")
    selected_specs = _panel_ordered_selected_specs(handoff)
    selected: list[dict[str, Any]] = []
    for panel_index, spec in enumerate(selected_specs):
        metadata, arrays = _load_material_shard(
            _selected_directory(str(spec["state_id"]))
        )
        selected.append(
            _call(
                METRICS,
                "validate_selected_reset_material_shard",
                metadata,
                reopened_arrays=arrays,
                selected_candidate_spec=spec,
                expected_panel_index=panel_index,
                material_metadata_binding=_file_binding(
                    _selected_directory(str(spec["state_id"])) / "metadata.json",
                    relative_to=MATERIAL_ROOT,
                ),
            )
        )
    documents = {
        leaf: _ordinary_json(OUTPUT_ROOT / leaf)
        for leaf in (
            "panel_manifest.json",
            "split_manifest.json",
            "state_snapshot_index.json",
            "teacher_trace_index.json",
            "edge_port_index.json",
            "target_contracts.json",
        )
    }
    qualification_material = _load_qualification_material_validations(records)
    _call(
        METRICS,
        "validate_frozen_panel_documents",
        documents,
        records,
        generator,
        handoff,
        qualification_material,
        selected,
        runtime,
        allow_fake_runtime=fake_runtime,
    )
    return runtime, records, generator, handoff, documents, selected


def _visual_stage_runtime(*, role: str, fake_runtime: bool) -> dict[str, Any]:
    try:
        return copy.deepcopy(
            V1.require_stage_runtime("visual", fake=fake_runtime, visual_role=role)
        )
    except V1.ExperimentError as exc:
        raise ExperimentError(str(exc)) from exc


def _physical_candidate_outcome_exists() -> bool:
    outcome_roots = (
        MATERIAL_ROOT / "fanout",
        MATERIAL_ROOT / "repeatability",
    )
    if any(not path.is_dir() or path.is_symlink() for path in outcome_roots):
        raise ExperimentError("physical candidate material root drift")
    return any(any(path.iterdir()) for path in outcome_roots)


def encode_canonical_pixels_stage(
    *, encoder: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    """Encode each unique selected RGB exactly once in sorted hash order."""

    import numpy as np

    runtime, _records, _generator, _handoff, documents, _selected = (
        _load_frozen_panel_evidence(fake_runtime=fake_runtime)
    )
    if _physical_candidate_outcome_exists():
        raise ExperimentError("physical candidate outcome exists before encoding")
    panel_states = [dict(value) for value in documents["panel_manifest.json"]["states"]]
    if len(panel_states) != CONTRACT.PANEL_STATE_COUNT:
        raise ExperimentError("encoding panel cardinality drift")
    by_hash: dict[str, Any] = {}
    state_rows: list[dict[str, Any]] = []
    for state in panel_states:
        state_id = str(state["state_id"])
        metadata, arrays = _load_material_shard(_selected_directory(state_id))
        image = np.ascontiguousarray(np.asarray(arrays["rgb"], dtype=np.uint8))
        if image.shape != (168, 224, 3):
            raise ExperimentError("selected encoding RGB shape drift")
        digest = V1.canonical_array_sha256(image)
        if digest != metadata.get("rgb_sha256"):
            raise ExperimentError("selected encoding RGB digest drift")
        if digest in by_hash and not np.array_equal(by_hash[digest], image):
            raise ExperimentError("selected encoding pixel SHA collision")
        by_hash.setdefault(digest, image)
        state_rows.append(
            {
                "panel_index": int(state["panel_index"]),
                "state_id": state_id,
                "qualification_candidate_index": int(
                    state["qualification_candidate_index"]
                ),
                "pixel_sha256": digest,
            }
        )
    order = sorted(by_hash)
    canonical_index = {digest: index for index, digest in enumerate(order)}
    encoder_runtime = _visual_stage_runtime(
        role="encoder", fake_runtime=fake_runtime
    )
    runtime_encoder = V1._FrozenSingletonEncoder() if encoder is None else encoder  # noqa: SLF001
    source_authority = CONTRACT.V4.CANONICAL_ENCODING_AUTHORITY[
        "external_encoder_source"
    ]
    expected_source = {
        "repository_path": str(source_authority["repository_path"]),
        "commit": str(source_authority["commit"]),
        "worktree_clean": True,
    }
    external_source = (
        copy.deepcopy(runtime_encoder.source_binding)
        if encoder is None
        else copy.deepcopy(expected_source)
        if fake_runtime
        else copy.deepcopy(getattr(runtime_encoder, "source_binding", None))
    )
    if external_source != expected_source:
        raise ExperimentError("frozen encoder source binding drift")
    expected_checkpoint = CONTRACT.V4.VJEPA_ENCODER_BINDING["checkpoint_sha256"]
    if expected_checkpoint != (
        "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
    ):
        raise ExperimentError("frozen encoder checkpoint authority drift")
    raw_rows: list[Any] = []
    descriptor_rows: list[Any] = []
    receipt_rows: list[dict[str, Any]] = []
    for index, pixel_sha in enumerate(order):
        encoded = dict(runtime_encoder.encode_singleton(by_hash[pixel_sha]))
        if set(encoded) != {
            "raw_tokens",
            "spatial_descriptor",
            "preprocessed_tensor_sha256",
            "checkpoint_sha256",
            "batch_size",
        }:
            raise ExperimentError("singleton encoder result field drift")
        raw_tokens = np.ascontiguousarray(
            np.asarray(encoded["raw_tokens"], dtype=np.float16)
        )
        descriptors = np.ascontiguousarray(
            np.asarray(encoded["spatial_descriptor"], dtype=np.float32)
        )
        if (
            raw_tokens.shape != (768, 1024)
            or descriptors.shape != (768, 1024)
            or not bool(np.isfinite(raw_tokens).all())
            or not bool(np.isfinite(descriptors).all())
            or encoded["checkpoint_sha256"] != expected_checkpoint
            or int(encoded["batch_size"]) != 1
        ):
            raise ExperimentError("singleton frozen encoder contract drift")
        raw_rows.append(raw_tokens)
        descriptor_rows.append(descriptors)
        receipt_rows.append(
            {
                "canonical_pixel_index": index,
                "pixel_sha256": pixel_sha,
                "preprocessed_tensor_sha256": str(
                    encoded["preprocessed_tensor_sha256"]
                ),
                "raw_token_sha256": V1.canonical_array_sha256(raw_tokens),
                "spatial_descriptor_sha256": V1.canonical_array_sha256(
                    descriptors
                ),
            }
        )
    for row in state_rows:
        row["canonical_pixel_index"] = canonical_index[row["pixel_sha256"]]
    arrays = {
        "raw_tokens": np.ascontiguousarray(np.stack(raw_rows)).astype(np.float16),
        "spatial_descriptors": np.ascontiguousarray(
            np.stack(descriptor_rows)
        ).astype(np.float32),
        "pixel_sha256_bytes": np.ascontiguousarray(
            np.stack(
                [np.frombuffer(bytes.fromhex(value), dtype=np.uint8) for value in order]
            )
        ),
    }
    metadata = {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "canonical_encoding_material.v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "source_freeze_commit": runtime["source_freeze_commit"],
        "runtime_contract_content_digest": runtime["content_digest"],
        "panel_manifest_binding": _file_binding(
            OUTPUT_ROOT / "panel_manifest.json", relative_to=OUTPUT_ROOT
        ),
        "split_manifest_binding": _file_binding(
            OUTPUT_ROOT / "split_manifest.json", relative_to=OUTPUT_ROOT
        ),
        "state_rows": state_rows,
        "unique_pixel_count": len(order),
        "pixel_order": order,
        "encoding_rows": receipt_rows,
        "encoder_binding": copy.deepcopy(CONTRACT.V4.VJEPA_ENCODER_BINDING),
        "preprocessing_authority": copy.deepcopy(
            CONTRACT.V4.CANONICAL_ENCODING_AUTHORITY["preprocessing"]
        ),
        "external_encoder_source": external_source,
        "encoder_runtime_environment": encoder_runtime,
        "batch_size": 1,
        "fanout_outcomes_opened": 0,
        "models_trained": 0,
    }
    directory = MATERIAL_ROOT / "downstream_workspace" / "encoding"
    _write_material_shard(directory, metadata, arrays)
    reopened, reopened_arrays = _load_material_shard(directory)
    return copy.deepcopy(
        _call(
            METRICS,
            "validate_encoding_material_shard",
            reopened,
            reopened_arrays=reopened_arrays,
            panel_documents=documents,
            runtime_contract=runtime,
            allow_fake_runtime=fake_runtime,
            material_metadata_binding=_file_binding(
                directory / "metadata.json", relative_to=MATERIAL_ROOT
            ),
        )
    )


def _panel_state_and_spec(
    documents: Mapping[str, Mapping[str, Any]], handoff: Mapping[str, Any], state_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    states = [dict(value) for value in documents["panel_manifest.json"]["states"]]
    state = next((value for value in states if value["state_id"] == state_id), None)
    specs = {
        str(value["state_id"]): dict(value)
        for value in handoff["selected_candidate_specs"]
    }
    if state is None or state_id not in specs:
        raise ExperimentError("state is outside the frozen successor panel")
    spec = specs[state_id]
    if (
        int(state["qualification_candidate_index"]) != int(spec["candidate_index"])
        or state["candidate_spec_id"] != spec["candidate_spec_id"]
    ):
        raise ExperimentError("panel state/sparse qualification identity drift")
    return state, spec


def _load_selected_material_validation(
    state: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    state_id = str(state["state_id"])
    directory = _selected_directory(state_id)
    metadata, arrays = _load_material_shard(directory)
    return copy.deepcopy(
        _call(
            METRICS,
            "validate_selected_reset_material_shard",
            metadata,
            reopened_arrays=arrays,
            selected_candidate_spec=spec,
            expected_panel_index=int(state["panel_index"]),
            material_metadata_binding=_file_binding(
                directory / "metadata.json", relative_to=MATERIAL_ROOT
            ),
        )
    )


def _derive_candidate_outcome(
    spec: Mapping[str, Any],
    trace: Mapping[str, Any],
    reset_pose_world: Sequence[float],
    branch_candidate_index: int,
    directed_port_world: Sequence[float],
) -> dict[str, Any]:
    inherited = V1.derive_candidate_outcome(
        spec, trace, reset_pose_world, branch_candidate_index
    )
    aligned = _call(
        METRICS,
        "derive_candidate_port_metrics",
        reset_pose_world,
        trace["base_pose_world"],
        directed_port_world,
    )
    if set(aligned) != {
        "port_progress_m",
        "lateral_error_m",
        "positive_port_progress",
    }:
        raise ExperimentError("candidate port metric projection drift")
    result = copy.deepcopy(inherited)
    result.update(copy.deepcopy(dict(aligned)))
    return result


def fanout_state_stage(
    state_id: str,
    *,
    backend: Any | None = None,
    fake_runtime: bool = False,
) -> dict[str, Any]:
    """Execute the unchanged twelve-action bank from one selected snapshot."""

    import numpy as np

    runtime, _records, _generator, handoff, documents, _selected = (
        _load_frozen_panel_evidence(fake_runtime=fake_runtime)
    )
    state, spec = _panel_state_and_spec(documents, handoff, str(state_id))
    selection_path = OUTPUT_ROOT / "development_target_selection.json"
    target_selection_binding: dict[str, Any] | None = None
    target_selection_opened = False
    if state["role"] == "DEVELOPMENT_HELDOUT":
        if not selection_path.is_file():
            raise ExperimentError("held-out fanout opened before target selection")
        selection = _ordinary_json(selection_path)
        if selection.get("selection_frozen") is not True:
            raise ExperimentError("held-out fanout lacks frozen target selection")
        target_selection_binding = _file_binding(
            selection_path, relative_to=OUTPUT_ROOT
        )
        target_selection_opened = True
        encoding_validation = _load_encoding_material_validation(
            runtime, documents, fake_runtime=fake_runtime
        )
        specs_by_state = {
            str(value["state_id"]): dict(value)
            for value in handoff["selected_candidate_specs"]
        }
        development_fanout = [
            _load_fanout_material_validation(
                panel_state,
                specs_by_state[str(panel_state["state_id"])],
                runtime,
                documents,
                fake_runtime=fake_runtime,
            )
            for panel_state in documents["panel_manifest.json"]["states"]
            if panel_state["role"] == "DEVELOPMENT"
        ]
        _call(
            METRICS,
            "validate_development_target_selection",
            selection,
            panel_documents=documents,
            encoding_material=encoding_validation,
            fanout_material=development_fanout,
            runtime_contract=runtime,
            allow_fake_runtime=fake_runtime,
        )
    elif selection_path.exists():
        raise ExperimentError("development fanout reopened after target selection")
    if state["role"] == "DEVELOPMENT":
        _load_encoding_material_validation(
            runtime, documents, fake_runtime=fake_runtime
        )
    selected_validation = _load_selected_material_validation(state, spec)
    selected_metadata = selected_validation["metadata"]
    selected_arrays = selected_validation["arrays"]
    snapshot_payload = bytes(
        np.asarray(selected_arrays["snapshot_payload_bytes"], dtype=np.uint8).tobytes(
            order="C"
        )
    )
    collector = GenesisSuccessorPhysicalBackend() if backend is None else backend
    raw_rows = list(collector.fanout(copy.deepcopy(spec), snapshot_payload))
    if len(raw_rows) != len(CONTRACT.CANDIDATE_IDS) or len(raw_rows) != 12:
        raise ExperimentError("candidate fanout branch cardinality drift")
    port_rows = [
        row
        for row in documents["edge_port_index.json"]["records"]
        if row["state_id"] == str(state_id)
    ]
    if len(port_rows) != 1:
        raise ExperimentError("fanout directed port is absent or ambiguous")
    directed_port = port_rows[0]["directed_port_world"]
    arrays: dict[str, Any] = {}
    outcomes: list[dict[str, Any]] = []
    backend_runtime: dict[str, Any] | None = None
    for branch_index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "candidate_index",
            "snapshot_payload_sha256",
            "trace",
            "runtime_evidence",
        }:
            raise ExperimentError("candidate fanout backend row field drift")
        if (
            int(raw["candidate_index"]) != branch_index
            or raw["snapshot_payload_sha256"]
            != selected_metadata["snapshot_payload_sha256"]
        ):
            raise ExperimentError("candidate fanout snapshot/index binding drift")
        trace = V1._normalise_trace(  # noqa: SLF001
            dict(raw["trace"]),
            kind="candidate",
            exact_samples=V1.PHYSICS_STEPS_PER_BRANCH,
        )
        for member, value in trace.items():
            arrays[f"candidate_{branch_index:02d}__{member}"] = value
        outcome = _derive_candidate_outcome(
            spec,
            trace,
            selected_arrays["snapshot__base_pose_world"],
            branch_index,
            directed_port,
        )
        outcome["branch_candidate_id"] = CONTRACT.CANDIDATE_IDS[branch_index]
        outcome["branch_candidate_index"] = branch_index
        outcomes.append(outcome)
        runtime_row = copy.deepcopy(dict(raw["runtime_evidence"]))
        if backend_runtime is None:
            backend_runtime = runtime_row
        elif backend_runtime != runtime_row:
            raise ExperimentError("backend runtime changed within fanout")
    if backend_runtime is None:  # pragma: no cover - cardinality checked above
        raise ExperimentError("candidate fanout backend runtime absent")
    stage_runtime = _physical_stage_runtime(
        backend_runtime, fake_runtime=fake_runtime
    )
    metadata = {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "candidate_fanout_material.v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "source_freeze_commit": runtime["source_freeze_commit"],
        "runtime_contract_content_digest": runtime["content_digest"],
        "panel_index": int(state["panel_index"]),
        "qualification_candidate_index": int(spec["candidate_index"]),
        "state_id": str(state_id),
        "family": str(state["family"]),
        "role": str(state["role"]),
        "candidate_spec_id": str(spec["candidate_spec_id"]),
        "canonical_spec_sha256": str(spec["canonical_spec_sha256"]),
        "source_selected_metadata_binding": _file_binding(
            _selected_directory(str(state_id)) / "metadata.json",
            relative_to=MATERIAL_ROOT,
        ),
        "source_selected_payload_binding": _file_binding(
            _selected_directory(str(state_id)) / "payload.npz",
            relative_to=MATERIAL_ROOT,
        ),
        "development_target_selection_binding": target_selection_binding,
        "development_target_selection_opened": target_selection_opened,
        "snapshot_payload_sha256": selected_metadata["snapshot_payload_sha256"],
        "directed_port_world": copy.deepcopy(directed_port),
        "branch_count": len(outcomes),
        "outcome_rows": outcomes,
        "stage_runtime": stage_runtime,
        "backend_runtime": backend_runtime,
        "models_trained": 0,
    }
    directory = _fanout_directory(str(state_id))
    _write_material_shard(directory, metadata, arrays)
    reopened, reopened_arrays = _load_material_shard(directory)
    return copy.deepcopy(
        _call(
            METRICS,
            "validate_fanout_material_shard",
            reopened,
            reopened_arrays=reopened_arrays,
            panel_state=state,
            candidate_spec=spec,
            selected_material=selected_validation,
            directed_port_record=port_rows[0],
            runtime_contract=runtime,
            allow_fake_runtime=fake_runtime,
            material_metadata_binding=_file_binding(
                directory / "metadata.json", relative_to=MATERIAL_ROOT
            ),
        )
    )


def _load_encoding_material_validation(
    runtime: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    *,
    fake_runtime: bool,
) -> dict[str, Any]:
    metadata, arrays = _load_material_shard(
        MATERIAL_ROOT / "downstream_workspace" / "encoding"
    )
    return copy.deepcopy(
        _call(
            METRICS,
            "validate_encoding_material_shard",
            metadata,
            reopened_arrays=arrays,
            panel_documents=documents,
            runtime_contract=runtime,
            allow_fake_runtime=fake_runtime,
            material_metadata_binding=_file_binding(
                MATERIAL_ROOT / "downstream_workspace" / "encoding" / "metadata.json",
                relative_to=MATERIAL_ROOT,
            ),
        )
    )


def _load_fanout_material_validation(
    state: Mapping[str, Any],
    spec: Mapping[str, Any],
    runtime: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    *,
    fake_runtime: bool,
) -> dict[str, Any]:
    state_id = str(state["state_id"])
    metadata, arrays = _load_material_shard(_fanout_directory(state_id))
    selected_validation = _load_selected_material_validation(state, spec)
    ports = [
        row
        for row in documents["edge_port_index.json"]["records"]
        if row["state_id"] == state_id
    ]
    if len(ports) != 1:
        raise ExperimentError("fanout directed port is absent or ambiguous")
    return copy.deepcopy(
        _call(
            METRICS,
            "validate_fanout_material_shard",
            metadata,
            reopened_arrays=arrays,
            panel_state=state,
            candidate_spec=spec,
            selected_material=selected_validation,
            directed_port_record=ports[0],
            runtime_contract=runtime,
            allow_fake_runtime=fake_runtime,
            material_metadata_binding=_file_binding(
                _fanout_directory(state_id) / "metadata.json",
                relative_to=MATERIAL_ROOT,
            ),
        )
    )


def _ranker_inputs(
    state: Mapping[str, Any], encoding: Mapping[str, Any]
) -> tuple[Any, list[float], list[list[float]]]:
    import numpy as np

    state_id = str(state["state_id"])
    encoding_metadata = encoding["metadata"]
    encoding_arrays = encoding["arrays"]
    matches = [
        row
        for row in encoding_metadata["state_rows"]
        if row["state_id"] == state_id
    ]
    if len(matches) != 1:
        raise ExperimentError("state lacks a unique canonical encoding row")
    canonical_index = int(matches[0]["canonical_pixel_index"])
    tokens = np.ascontiguousarray(
        encoding_arrays["raw_tokens"][canonical_index]
    ).copy()
    selected_metadata, selected_arrays = _load_material_shard(
        _selected_directory(state_id)
    )
    if selected_metadata["qualification_candidate_index"] != state[
        "qualification_candidate_index"
    ]:
        raise ExperimentError("ranker selected/sparse qualification binding drift")
    previous = [
        float(value)
        for value in selected_arrays["snapshot__previous_applied_command"]
    ]
    history = [
        [float(value) for value in row]
        for row in selected_arrays["snapshot__control_history"]
    ]
    return tokens, previous, history


def _state_target_row(
    state: Mapping[str, Any],
    target: Mapping[str, Any],
    fanout: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
) -> dict[str, Any]:
    row = V1._selection_metrics(  # noqa: SLF001
        fanout, V1._rank_scores(scores), scores  # noqa: SLF001
    )
    row.update(
        {
            "panel_index": int(state["panel_index"]),
            "qualification_candidate_index": int(
                state["qualification_candidate_index"]
            ),
            "state_id": str(state["state_id"]),
            "target_id": str(target["target_id"]),
            "target_transform_error_m": max(
                float(target["position_transform_error_m"]),
                float(target["heading_transform_error_rad"]),
            ),
        }
    )
    if set(row) != set(CONTRACT.STATE_TARGET_ROW_FIELDS):
        raise ExperimentError("development state/target row field drift")
    return copy.deepcopy(row)


def development_target_selection_stage(
    *, ranker: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    """Use only the 48 development states to freeze one target contract."""

    runtime, _records, _generator, handoff, documents, _selected = (
        _load_frozen_panel_evidence(fake_runtime=fake_runtime)
    )
    output_path = OUTPUT_ROOT / "development_target_selection.json"
    if output_path.exists() or output_path.is_symlink():
        raise ExperimentError("development target selection is already frozen")
    if (OUTPUT_ROOT / "heldout_scores.jsonl").exists():
        raise ExperimentError("held-out scores exist before target selection")
    states = [dict(row) for row in documents["panel_manifest.json"]["states"]]
    specs = {
        str(spec["state_id"]): dict(spec)
        for spec in handoff["selected_candidate_specs"]
    }
    heldout = [row for row in states if row["role"] == "DEVELOPMENT_HELDOUT"]
    if any(_fanout_directory(str(row["state_id"])).exists() for row in heldout):
        raise ExperimentError("held-out fanout exists before target selection")
    development = [row for row in states if row["role"] == "DEVELOPMENT"]
    if len(development) != 48:
        raise ExperimentError("development panel cardinality drift")
    encoding = _load_encoding_material_validation(
        runtime, documents, fake_runtime=fake_runtime
    )
    target_rows = [dict(row) for row in documents["target_contracts.json"]["rows"]]
    target_by_identity = {
        (str(row["state_id"]), str(row["target_id"])): row
        for row in target_rows
    }
    runtime_ranker = V1._FrozenPhysicalRanker() if ranker is None else ranker  # noqa: SLF001
    ranker_runtime = _visual_stage_runtime(
        role="ranker", fake_runtime=fake_runtime
    )
    if (
        CONTRACT.FROZEN_RANKER_CHECKPOINT_SHA256
        != "e1a2a58ff527b4d2bc210f6b1c8fd1d00d87873691db64e8c3ccb2257fa6c127"
    ):
        raise ExperimentError("frozen ranker checkpoint authority drift")
    fanout_material: list[dict[str, Any]] = []
    state_target_rows: list[dict[str, Any]] = []
    for state in development:
        state_id = str(state["state_id"])
        fanout_validation = _load_fanout_material_validation(
            state,
            specs[state_id],
            runtime,
            documents,
            fake_runtime=fake_runtime,
        )
        fanout_material.append(fanout_validation)
        outcomes = [
            dict(row) for row in fanout_validation["metadata"]["outcome_rows"]
        ]
        tokens, previous, history = _ranker_inputs(state, encoding)
        for target_id in CONTRACT.TARGET_IDS:
            target = target_by_identity[(state_id, target_id)]
            scores = [
                float(value)
                for value in runtime_ranker.score(
                    tokens, target, previous, history
                )
            ]
            state_target_rows.append(
                _state_target_row(state, target, outcomes, scores)
            )
    summaries = [
        V1._target_summary(state_target_rows, target_id)  # noqa: SLF001
        for target_id in CONTRACT.TARGET_IDS
    ]
    document = _call(
        METRICS,
        "build_development_target_selection",
        documents,
        encoding,
        fanout_material,
        state_target_rows,
        summaries,
        ranker_runtime,
        runtime,
        allow_fake_runtime=fake_runtime,
    )
    _call(
        METRICS,
        "validate_development_target_selection",
        document,
        panel_documents=documents,
        encoding_material=encoding,
        fanout_material=fanout_material,
        runtime_contract=runtime,
        allow_fake_runtime=fake_runtime,
    )
    _atomic_json(output_path, document)
    return copy.deepcopy(document)


def _heldout_candidate_score_row(
    state: Mapping[str, Any],
    condition_id: str,
    target_id: str,
    fanout: Sequence[Mapping[str, Any]],
    scores: Sequence[float] | None,
    ranking: Sequence[int],
    ranker_runtime_sha256: str,
) -> dict[str, Any]:
    row = V1._selection_metrics(fanout, ranking, scores)  # noqa: SLF001
    row.update(
        {
            "score_row_id": f"{state['state_id']}::{condition_id}",
            "panel_index": int(state["panel_index"]),
            "qualification_candidate_index": int(
                state["qualification_candidate_index"]
            ),
            "state_id": str(state["state_id"]),
            "family": str(state["family"]),
            "condition_id": condition_id,
            "target_id": target_id,
            "teacher_trace_index": None,
            "teacher_trace_id": None,
            "teacher_correct_execution": None,
            "ranker_checkpoint_sha256": (
                CONTRACT.FROZEN_RANKER_CHECKPOINT_SHA256
                if condition_id == "FROZEN_CURRENT_VISUAL_RANKER"
                else None
            ),
            "ranker_runtime_environment_sha256": ranker_runtime_sha256,
        }
    )
    if set(row) != set(CONTRACT.HELDOUT_SCORE_ROW_FIELDS):
        raise ExperimentError("held-out candidate score row field drift")
    return copy.deepcopy(row)


def heldout_scores_stage(
    *, ranker: Any | None = None, fake_runtime: bool = False
) -> list[dict[str, Any]]:
    """Evaluate four frozen comparators on the sixteen held-out states."""

    runtime, _records, _generator, handoff, documents, _selected = (
        _load_frozen_panel_evidence(fake_runtime=fake_runtime)
    )
    output_path = OUTPUT_ROOT / "heldout_scores.jsonl"
    if output_path.exists() or output_path.is_symlink():
        raise ExperimentError("held-out score ledger already exists")
    selection = _ordinary_json(
        OUTPUT_ROOT / "development_target_selection.json"
    )
    if selection.get("selection_frozen") is not True:
        raise ExperimentError("held-out scoring lacks frozen target selection")
    encoding = _load_encoding_material_validation(
        runtime, documents, fake_runtime=fake_runtime
    )
    states = [dict(row) for row in documents["panel_manifest.json"]["states"]]
    heldout = [row for row in states if row["role"] == "DEVELOPMENT_HELDOUT"]
    if len(heldout) != 16:
        raise ExperimentError("held-out panel cardinality drift")
    specs = {
        str(spec["state_id"]): dict(spec)
        for spec in handoff["selected_candidate_specs"]
    }
    all_fanout_material = [
        _load_fanout_material_validation(
            state,
            specs[str(state["state_id"])],
            runtime,
            documents,
            fake_runtime=fake_runtime,
        )
        for state in states
    ]
    _call(
        METRICS,
        "validate_development_target_selection",
        selection,
        panel_documents=documents,
        encoding_material=encoding,
        fanout_material=[
            value
            for value in all_fanout_material
            if value["metadata"]["role"] == "DEVELOPMENT"
        ],
        runtime_contract=runtime,
        allow_fake_runtime=fake_runtime,
    )
    fanout_by_state = {
        str(value["metadata"]["state_id"]): value
        for value in all_fanout_material
    }
    target_id = str(selection["selected_target_id"])
    targets = {
        str(row["state_id"]): row
        for row in documents["target_contracts.json"]["rows"]
        if row["target_id"] == target_id
    }
    teachers = {
        str(row["state_id"]): row
        for row in documents["teacher_trace_index.json"]["records"]
        if row["selected"]
    }
    ranker_runtime = _visual_stage_runtime(
        role="ranker", fake_runtime=fake_runtime
    )
    if selection.get("ranker_runtime_environment") != ranker_runtime:
        raise ExperimentError("ranker runtime changed after target selection")
    ranker_runtime_sha256 = METRICS.runtime_environment_sha256(ranker_runtime)
    runtime_ranker = V1._FrozenPhysicalRanker() if ranker is None else ranker  # noqa: SLF001
    rows: list[dict[str, Any]] = []
    for state in heldout:
        state_id = str(state["state_id"])
        fanout_validation = fanout_by_state[state_id]
        outcomes = [
            dict(row) for row in fanout_validation["metadata"]["outcome_rows"]
        ]
        target = targets[state_id]
        tokens, previous, history = _ranker_inputs(state, encoding)
        ranker_scores = [
            float(value)
            for value in runtime_ranker.score(tokens, target, previous, history)
        ]
        for condition_id in CONTRACT.HELDOUT_CONDITION_IDS:
            if condition_id == "DETERMINISTIC_KINEMATICS":
                scores = V1._kinematic_scores(  # noqa: SLF001
                    outcomes, target["target_body_pose"]
                )
                rows.append(
                    _heldout_candidate_score_row(
                        state,
                        condition_id,
                        target_id,
                        outcomes,
                        scores,
                        V1._rank_scores(scores),  # noqa: SLF001
                        ranker_runtime_sha256,
                    )
                )
            elif condition_id == "FROZEN_CURRENT_VISUAL_RANKER":
                rows.append(
                    _heldout_candidate_score_row(
                        state,
                        condition_id,
                        target_id,
                        outcomes,
                        ranker_scores,
                        V1._rank_scores(ranker_scores),  # noqa: SLF001
                        ranker_runtime_sha256,
                    )
                )
            elif condition_id == "ORACLE_BEST_ADMISSIBLE_CANDIDATE":
                rows.append(
                    _heldout_candidate_score_row(
                        state,
                        condition_id,
                        target_id,
                        outcomes,
                        None,
                        V1._oracle_ranking(outcomes),  # noqa: SLF001
                        ranker_runtime_sha256,
                    )
                )
            elif condition_id == "TEACHER_TRACE":
                teacher = teachers[state_id]
                projection = teacher["raw_projection"]
                teacher_row = {
                    "score_row_id": f"{state_id}::{condition_id}",
                    "panel_index": int(state["panel_index"]),
                    "qualification_candidate_index": int(
                        state["qualification_candidate_index"]
                    ),
                    "state_id": state_id,
                    "family": str(state["family"]),
                    "condition_id": condition_id,
                    "target_id": target_id,
                    "candidate_ids": [],
                    "scores": None,
                    "ranking": None,
                    "eligible_correct_edge_candidate_indices": [],
                    "selected_candidate_index": None,
                    "correct_edge_top1": None,
                    "correct_edge_top3": None,
                    "correct_edge_mrr": None,
                    "selected_correct_edge_execution": None,
                    "selected_port_progress_m": float(
                        projection["route_progress_m"]
                    ),
                    "oracle_best_port_progress_m": None,
                    "minimum_admissible_port_progress_m": None,
                    "normalized_port_regret": None,
                    "teacher_trace_index": int(teacher["trace_index"]),
                    "teacher_trace_id": str(teacher["teacher_trace_id"]),
                    "teacher_correct_execution": bool(
                        projection["teacher_valid"]
                    ),
                    "ranker_checkpoint_sha256": None,
                    "pairwise_correct_edge_ordering": None,
                    "selected_wrong_edge": bool(
                        projection["competing_port_entered"]
                    ),
                    "selected_no_edge": bool(
                        not projection["crossed_directed_port"]
                        and not projection["competing_port_entered"]
                    ),
                    "selected_lateral_error_m": float(
                        projection["endpoint_lateral_error_m"]
                    ),
                    "selected_angular_error_rad": float(
                        projection["endpoint_angular_error_rad"]
                    ),
                    "selected_contact": not bool(projection["contact_free"]),
                    "selected_stuck": bool(projection["stuck"]),
                    "selected_successor_viable": bool(
                        projection["successor_viable"]
                    ),
                    "ranker_runtime_environment_sha256": ranker_runtime_sha256,
                }
                if set(teacher_row) != set(CONTRACT.HELDOUT_SCORE_ROW_FIELDS):
                    raise ExperimentError("held-out teacher row field drift")
                rows.append(teacher_row)
            else:  # pragma: no cover - frozen authority exhausts cases
                raise ExperimentError("unknown held-out condition")
    validated = _call(
        METRICS,
        "build_heldout_scores",
        documents,
        selection,
        encoding,
        all_fanout_material,
        rows,
        ranker_runtime,
        runtime,
        allow_fake_runtime=fake_runtime,
    )
    _call(
        METRICS,
        "validate_heldout_scores",
        validated,
        panel_documents=documents,
        development_target_selection=selection,
        encoding_material=encoding,
        fanout_material=all_fanout_material,
        runtime_contract=runtime,
        allow_fake_runtime=fake_runtime,
    )
    _atomic_jsonl(output_path, validated)
    return [copy.deepcopy(row) for row in validated]


def repeat_state_stage(
    state_id: str,
    *,
    backend: Any | None = None,
    fake_runtime: bool = False,
) -> dict[str, Any]:
    """Repeat the frozen ranker and oracle branches twice for one held-out state."""

    import numpy as np

    runtime, _records, _generator, handoff, documents, _selected = (
        _load_frozen_panel_evidence(fake_runtime=fake_runtime)
    )
    state, spec = _panel_state_and_spec(documents, handoff, str(state_id))
    if state["role"] != "DEVELOPMENT_HELDOUT":
        raise ExperimentError("repeatability requested outside held-out panel")
    heldout_path = OUTPUT_ROOT / "heldout_scores.jsonl"
    heldout_raw, heldout_rows = _ordinary_jsonl(heldout_path)
    del heldout_raw
    if len(heldout_rows) != 16 * len(CONTRACT.HELDOUT_CONDITION_IDS):
        raise ExperimentError("repeatability lacks complete held-out scores")
    selection = _ordinary_json(
        OUTPUT_ROOT / "development_target_selection.json"
    )
    encoding_validation = _load_encoding_material_validation(
        runtime, documents, fake_runtime=fake_runtime
    )
    specs_by_state = {
        str(value["state_id"]): dict(value)
        for value in handoff["selected_candidate_specs"]
    }
    all_fanout = [
        _load_fanout_material_validation(
            panel_state,
            specs_by_state[str(panel_state["state_id"])],
            runtime,
            documents,
            fake_runtime=fake_runtime,
        )
        for panel_state in documents["panel_manifest.json"]["states"]
    ]
    _call(
        METRICS,
        "validate_development_target_selection",
        selection,
        panel_documents=documents,
        encoding_material=encoding_validation,
        fanout_material=[
            value
            for value in all_fanout
            if value["metadata"]["role"] == "DEVELOPMENT"
        ],
        runtime_contract=runtime,
        allow_fake_runtime=fake_runtime,
    )
    heldout_rows = _call(
        METRICS,
        "validate_heldout_scores",
        heldout_rows,
        panel_documents=documents,
        development_target_selection=selection,
        encoding_material=encoding_validation,
        fanout_material=all_fanout,
        runtime_contract=runtime,
        allow_fake_runtime=fake_runtime,
    )
    score_by_condition = {
        str(row["condition_id"]): row
        for row in heldout_rows
        if row.get("state_id") == str(state_id)
    }
    required = {
        "FROZEN_CURRENT_VISUAL_RANKER",
        "ORACLE_BEST_ADMISSIBLE_CANDIDATE",
    }
    if not required.issubset(score_by_condition):
        raise ExperimentError("repeatability selectors are absent")
    ranker_index = score_by_condition["FROZEN_CURRENT_VISUAL_RANKER"].get(
        "selected_candidate_index"
    )
    oracle_index = score_by_condition["ORACLE_BEST_ADMISSIBLE_CANDIDATE"].get(
        "selected_candidate_index"
    )
    if (
        not isinstance(ranker_index, int)
        or isinstance(ranker_index, bool)
        or not isinstance(oracle_index, int)
        or isinstance(oracle_index, bool)
        or not 0 <= ranker_index < len(CONTRACT.CANDIDATE_IDS)
        or not 0 <= oracle_index < len(CONTRACT.CANDIDATE_IDS)
    ):
        raise ExperimentError("repeatability selector index drift")
    selected_indices = [ranker_index, ranker_index, oracle_index, oracle_index]
    fanout_directory = _fanout_directory(str(state_id))
    fanout_validation = next(
        value
        for value in all_fanout
        if value["metadata"]["state_id"] == str(state_id)
    )
    selected_validation = _load_selected_material_validation(state, spec)
    selected_metadata = selected_validation["metadata"]
    selected_arrays = selected_validation["arrays"]
    snapshot_payload = bytes(
        np.asarray(
            selected_arrays["snapshot_payload_bytes"], dtype=np.uint8
        ).tobytes(order="C")
    )
    collector = GenesisSuccessorPhysicalBackend() if backend is None else backend
    raw_rows = list(
        collector.repeat(copy.deepcopy(spec), snapshot_payload, selected_indices)
    )
    if len(raw_rows) != 4:
        raise ExperimentError("repeatability backend row cardinality drift")
    arrays: dict[str, Any] = {}
    outcomes: list[dict[str, Any]] = []
    backend_runtime: dict[str, Any] | None = None
    port_rows = [
        row
        for row in documents["edge_port_index.json"]["records"]
        if row["state_id"] == str(state_id)
    ]
    if len(port_rows) != 1:
        raise ExperimentError("repeatability directed port is absent or ambiguous")
    directed_port = port_rows[0]["directed_port_world"]
    for selector_index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "selector_index",
            "candidate_index",
            "snapshot_payload_sha256",
            "trace",
            "runtime_evidence",
        }:
            raise ExperimentError("repeatability backend row field drift")
        if (
            int(raw["selector_index"]) != selector_index
            or int(raw["candidate_index"]) != selected_indices[selector_index]
            or raw["snapshot_payload_sha256"]
            != selected_metadata["snapshot_payload_sha256"]
        ):
            raise ExperimentError("repeatability selector/snapshot binding drift")
        trace = V1._normalise_trace(  # noqa: SLF001
            dict(raw["trace"]),
            kind="candidate",
            exact_samples=V1.PHYSICS_STEPS_PER_BRANCH,
        )
        for member, value in trace.items():
            arrays[f"repeat_{selector_index:02d}__{member}"] = value
        rebuilt_outcome = _derive_candidate_outcome(
            spec,
            trace,
            selected_arrays["snapshot__base_pose_world"],
            selected_indices[selector_index],
            directed_port,
        )
        outcomes.append(
            {
                field: rebuilt_outcome[field]
                for field in CONTRACT.REPEAT_OUTCOME_FIELDS
            }
        )
        runtime_row = copy.deepcopy(dict(raw["runtime_evidence"]))
        if backend_runtime is None:
            backend_runtime = runtime_row
        elif backend_runtime != runtime_row:
            raise ExperimentError("backend runtime changed within repeatability")
    if backend_runtime is None:  # pragma: no cover - cardinality checked above
        raise ExperimentError("repeatability backend runtime absent")
    stage_runtime = _physical_stage_runtime(
        backend_runtime, fake_runtime=fake_runtime
    )
    metadata = {
        "schema": CONTRACT.REPEATABILITY_MATERIAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "source_freeze_commit": runtime["source_freeze_commit"],
        "runtime_contract_content_digest": runtime["content_digest"],
        "panel_index": int(state["panel_index"]),
        "qualification_candidate_index": int(spec["candidate_index"]),
        "state_id": str(state_id),
        "family": str(state["family"]),
        "role": str(state["role"]),
        "source_fanout_metadata_binding": _file_binding(
            fanout_directory / "metadata.json", relative_to=MATERIAL_ROOT
        ),
        "source_fanout_payload_binding": _file_binding(
            fanout_directory / "payload.npz", relative_to=MATERIAL_ROOT
        ),
        "heldout_scores_binding": _file_binding(
            heldout_path, relative_to=OUTPUT_ROOT
        ),
        "repeat_count": 4,
        "outcome_rows": outcomes,
        "stage_runtime": stage_runtime,
        "backend_runtime": backend_runtime,
        "models_trained": 0,
    }
    directory = _repeatability_directory(str(state_id))
    _write_material_shard(directory, metadata, arrays)
    reopened, reopened_arrays = _load_material_shard(directory)
    return copy.deepcopy(
        _call(
            METRICS,
            "validate_repeatability_material_shard",
            reopened,
            reopened_arrays=reopened_arrays,
            panel_state=state,
            candidate_spec=spec,
            selected_material=selected_validation,
            source_fanout_material=fanout_validation,
            heldout_score_rows=heldout_rows,
            directed_port_record=port_rows[0],
            runtime_contract=runtime,
            allow_fake_runtime=fake_runtime,
            material_metadata_binding=_file_binding(
                directory / "metadata.json", relative_to=MATERIAL_ROOT
            ),
        )
    )


def assemble_row_evidence_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    """Build native official fanout/repeat ledgers from exact material shards."""

    runtime, _records, _generator, handoff, documents, _selected = (
        _load_frozen_panel_evidence(fake_runtime=fake_runtime)
    )
    fanout_path = OUTPUT_ROOT / "candidate_fanout.jsonl"
    repeat_path = OUTPUT_ROOT / "repeatability.jsonl"
    if any(path.exists() or path.is_symlink() for path in (fanout_path, repeat_path)):
        raise ExperimentError("official row evidence already exists")
    selection = _ordinary_json(
        OUTPUT_ROOT / "development_target_selection.json"
    )
    _heldout_raw, heldout_rows = _ordinary_jsonl(
        OUTPUT_ROOT / "heldout_scores.jsonl"
    )
    states = [dict(row) for row in documents["panel_manifest.json"]["states"]]
    specs = {
        str(spec["state_id"]): dict(spec)
        for spec in handoff["selected_candidate_specs"]
    }
    encoding = _load_encoding_material_validation(
        runtime, documents, fake_runtime=fake_runtime
    )
    generator_runtime = _ordinary_json(
        MATERIAL_ROOT / "generator_runtime_environment.json"
    )
    CONTRACT.validate_content_digest(generator_runtime)
    if (
        generator_runtime.get("source_freeze_commit")
        != runtime["source_freeze_commit"]
        or generator_runtime.get("runtime_contract_content_digest")
        != runtime["content_digest"]
    ):
        raise ExperimentError("generator physical runtime binding drift")
    physical_runtime_core_sha256 = str(
        generator_runtime["physical_runtime_core_sha256"]
    )
    fanout_material: list[dict[str, Any]] = []
    fanout_rows: list[dict[str, Any]] = []
    for state in states:
        state_id = str(state["state_id"])
        validation = _load_fanout_material_validation(
            state,
            specs[state_id],
            runtime,
            documents,
            fake_runtime=fake_runtime,
        )
        fanout_material.append(validation)
        metadata = validation["metadata"]
        arrays = validation["arrays"]
        metadata_binding = _file_binding(
            _fanout_directory(state_id) / "metadata.json",
            relative_to=MATERIAL_ROOT,
        )
        payload_binding = _file_binding(
            _fanout_directory(state_id) / "payload.npz",
            relative_to=MATERIAL_ROOT,
        )
        for branch_index, outcome in enumerate(metadata["outcome_rows"]):
            trace = {
                member: arrays[f"candidate_{branch_index:02d}__{member}"]
                for member in CONTRACT.CANDIDATE_TRACE_MEMBER_ORDER
            }
            row = {
                "branch_id": (
                    f"{state_id}::{CONTRACT.CANDIDATE_IDS[branch_index]}"
                ),
                "panel_index": int(state["panel_index"]),
                "qualification_candidate_index": int(
                    state["qualification_candidate_index"]
                ),
                "state_id": state_id,
                "family": str(state["family"]),
                "role": str(state["role"]),
                "candidate_spec_id": str(state["candidate_spec_id"]),
                "branch_candidate_index": branch_index,
                "branch_candidate_id": CONTRACT.CANDIDATE_IDS[branch_index],
                "snapshot_payload_sha256": metadata["snapshot_payload_sha256"],
                "material_metadata_binding": metadata_binding,
                "material_payload_binding": payload_binding,
                "outcome_row_index": branch_index,
                "trace_digests": V1._trace_digest_projection(trace),  # noqa: SLF001
                **copy.deepcopy(dict(outcome)),
            }
            if set(row) != set(CONTRACT.CANDIDATE_FANOUT_ROW_FIELDS):
                raise ExperimentError("candidate fanout row field drift")
            fanout_rows.append(row)
    if len(fanout_rows) != CONTRACT.PANEL_STATE_COUNT * len(CONTRACT.CANDIDATE_IDS):
        raise ExperimentError("candidate fanout ledger cardinality drift")
    selection = _call(
        METRICS,
        "validate_development_target_selection",
        selection,
        panel_documents=documents,
        encoding_material=encoding,
        fanout_material=[
            value
            for value in fanout_material
            if value["metadata"]["role"] == "DEVELOPMENT"
        ],
        runtime_contract=runtime,
        allow_fake_runtime=fake_runtime,
    )
    fanout_rows = _call(
        METRICS,
        "validate_candidate_fanout_rows",
        fanout_rows,
        panel_documents=documents,
        fanout_material=fanout_material,
        development_target_selection=selection,
        runtime_contract=runtime,
        allow_fake_runtime=fake_runtime,
    )
    heldout_rows = _call(
        METRICS,
        "validate_heldout_scores",
        heldout_rows,
        panel_documents=documents,
        development_target_selection=selection,
        encoding_material=encoding,
        fanout_material=fanout_material,
        runtime_contract=runtime,
        allow_fake_runtime=fake_runtime,
    )
    fanout_by_identity = {
        (str(row["state_id"]), int(row["branch_candidate_index"])): row
        for row in fanout_rows
    }
    scores_by_identity = {
        (str(row["state_id"]), str(row["condition_id"])): row
        for row in heldout_rows
    }
    repeat_material: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    heldout_states = [
        state for state in states if state["role"] == "DEVELOPMENT_HELDOUT"
    ]
    for state in heldout_states:
        state_id = str(state["state_id"])
        directory = _repeatability_directory(state_id)
        metadata, arrays = _load_material_shard(directory)
        selected_validation = _load_selected_material_validation(
            state, specs[state_id]
        )
        selected_metadata = selected_validation["metadata"]
        fanout_validation = next(
            item
            for item in fanout_material
            if item["metadata"]["state_id"] == state_id
        )
        ports = [
            row
            for row in documents["edge_port_index.json"]["records"]
            if row["state_id"] == state_id
        ]
        validation = _call(
            METRICS,
            "validate_repeatability_material_shard",
            metadata,
            reopened_arrays=arrays,
            panel_state=state,
            candidate_spec=specs[state_id],
            selected_material=selected_validation,
            source_fanout_material=fanout_validation,
            heldout_score_rows=heldout_rows,
            directed_port_record=ports[0],
            runtime_contract=runtime,
            allow_fake_runtime=fake_runtime,
            material_metadata_binding=_file_binding(
                directory / "metadata.json", relative_to=MATERIAL_ROOT
            ),
        )
        repeat_material.append(validation)
        metadata_binding = _file_binding(
            directory / "metadata.json", relative_to=MATERIAL_ROOT
        )
        payload_binding = _file_binding(
            directory / "payload.npz", relative_to=MATERIAL_ROOT
        )
        selected_indices = [
            int(
                scores_by_identity[
                    (state_id, "FROZEN_CURRENT_VISUAL_RANKER")
                ]["selected_candidate_index"]
            ),
            int(
                scores_by_identity[
                    (state_id, "ORACLE_BEST_ADMISSIBLE_CANDIDATE")
                ]["selected_candidate_index"]
            ),
        ]
        for selector_index, selector_id in enumerate(CONTRACT.REPEAT_BRANCH_IDS):
            source_index = selected_indices[selector_index]
            source = fanout_by_identity[(state_id, source_index)]
            for repeat_index in range(2):
                outcome_index = selector_index * 2 + repeat_index
                outcome = dict(metadata["outcome_rows"][outcome_index])
                trace = {
                    member: arrays[f"repeat_{outcome_index:02d}__{member}"]
                    for member in CONTRACT.CANDIDATE_TRACE_MEMBER_ORDER
                }
                trace_digests = V1._trace_digest_projection(trace)  # noqa: SLF001
                source_endpoint = [
                    float(value) for value in source["h3_endpoint_body"]
                ]
                repeat_endpoint = [
                    float(value) for value in outcome["h3_endpoint_body"]
                ]
                position_error = math.hypot(
                    source_endpoint[0] - repeat_endpoint[0],
                    source_endpoint[1] - repeat_endpoint[1],
                )
                heading_error = abs(
                    V1._wrap_angle(  # noqa: SLF001
                        source_endpoint[2] - repeat_endpoint[2]
                    )
                )
                source_applied = str(
                    source["trace_digests"]["post_slew_applied_command"]
                )
                repeat_applied = str(
                    trace_digests["post_slew_applied_command"]
                )
                repeat_success = bool(
                    source_applied == repeat_applied
                    and bool(source["entered_correct_edge"])
                    is bool(outcome["entered_correct_edge"])
                    and source["endpoint_edge_id"] == outcome["endpoint_edge_id"]
                    and bool(source["physics_contact"])
                    is bool(outcome["physics_contact"])
                    and bool(source["stuck"]) is bool(outcome["stuck"])
                    and position_error
                    <= float(
                        CONTRACT.NUMERICAL_TOLERANCES[
                            "repeat_endpoint_position_m"
                        ]
                    )
                    and heading_error
                    <= float(
                        CONTRACT.NUMERICAL_TOLERANCES[
                            "repeat_endpoint_heading_rad"
                        ]
                    )
                )
                row = {
                    "repeat_id": f"{state_id}::{selector_id}::{repeat_index}",
                    "panel_index": int(state["panel_index"]),
                    "qualification_candidate_index": int(
                        state["qualification_candidate_index"]
                    ),
                    "state_id": state_id,
                    "family": str(state["family"]),
                    "branch_selector_id": selector_id,
                    "repeat_index": repeat_index,
                    "source_candidate_index": source_index,
                    "source_branch_id": str(source["branch_id"]),
                    "snapshot_payload_sha256": selected_metadata[
                        "snapshot_payload_sha256"
                    ],
                    "material_metadata_binding": metadata_binding,
                    "material_payload_binding": payload_binding,
                    "outcome_row_index": outcome_index,
                    "trace_digests": trace_digests,
                    "source_correct_edge_execution": bool(
                        source["entered_correct_edge"]
                    ),
                    "repeat_correct_edge_execution": bool(
                        outcome["entered_correct_edge"]
                    ),
                    "source_endpoint_body": source_endpoint,
                    "repeat_endpoint_body": repeat_endpoint,
                    "endpoint_position_error_m": position_error,
                    "endpoint_heading_error_rad": heading_error,
                    "physics_contact": bool(outcome["physics_contact"]),
                    "source_applied_command_sequence_sha256": source_applied,
                    "repeat_applied_command_sequence_sha256": repeat_applied,
                    "source_endpoint_edge_id": source["endpoint_edge_id"],
                    "repeat_endpoint_edge_id": outcome["endpoint_edge_id"],
                    "source_physics_contact": bool(source["physics_contact"]),
                    "source_stuck": bool(source["stuck"]),
                    "stuck": bool(outcome["stuck"]),
                    "repeat_success": repeat_success,
                    "physical_runtime_core_sha256": physical_runtime_core_sha256,
                }
                if set(row) != set(CONTRACT.REPEATABILITY_ROW_FIELDS):
                    raise ExperimentError("repeatability row field drift")
                repeat_rows.append(row)
    if len(repeat_rows) != 64:
        raise ExperimentError("repeatability ledger cardinality drift")
    repeat_rows = _call(
        METRICS,
        "validate_repeatability_rows",
        repeat_rows,
        panel_documents=documents,
        candidate_fanout_rows=fanout_rows,
        heldout_score_rows=heldout_rows,
        repeatability_material=repeat_material,
        runtime_contract=runtime,
        allow_fake_runtime=fake_runtime,
    )
    _atomic_jsonl(fanout_path, fanout_rows)
    _atomic_jsonl(repeat_path, repeat_rows)
    return {
        "candidate_fanout_count": len(fanout_rows),
        "heldout_score_count": len(heldout_rows),
        "repeatability_count": len(repeat_rows),
        "models_trained": 0,
    }


def report_stage(*, evaluator_module: Any | None = None) -> dict[str, Any]:
    evaluator = _load_evaluator(evaluator_module)
    entrypoint = (
        "validate_existing_publication"
        if (OUTPUT_ROOT / "result.json").exists()
        else "reduce_publish_and_validate"
    )
    return copy.deepcopy(
        _call(
            evaluator,
            entrypoint,
            OUTPUT_ROOT,
            material_root=MATERIAL_ROOT,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    stages.add_parser("freeze-docs")
    stages.add_parser("initialize")
    qualify = stages.add_parser("qualify-stream")
    qualify.add_argument("--family", choices=FAMILIES, required=True)
    qualify.add_argument("--stratum-index", type=int, required=True)
    stages.add_parser("reduce-generator")
    stages.add_parser("freeze-panel")
    reset = stages.add_parser("qualify-selected-reset")
    reset.add_argument("--state-id", required=True)
    stages.add_parser("encode-canonical-pixels")
    fanout = stages.add_parser("fanout-state")
    fanout.add_argument("--state-id", required=True)
    stages.add_parser("select-development-target")
    stages.add_parser("score-heldout")
    repeat = stages.add_parser("repeat-state")
    repeat.add_argument("--state-id", required=True)
    stages.add_parser("assemble-row-evidence")
    stages.add_parser("report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    arguments = build_parser().parse_args(argv)
    if arguments.stage == "freeze-docs":
        result: Any = build_freeze_documents()
    elif arguments.stage == "initialize":
        result = initialize_stage()
    elif arguments.stage == "qualify-stream":
        result = qualify_stream_stage(arguments.family, arguments.stratum_index)
    elif arguments.stage == "reduce-generator":
        result = reduce_generator_stage()
    elif arguments.stage == "freeze-panel":
        result = freeze_panel_stage()
    elif arguments.stage == "qualify-selected-reset":
        result = capture_selected_state_stage(arguments.state_id)
    elif arguments.stage == "encode-canonical-pixels":
        result = encode_canonical_pixels_stage()
    elif arguments.stage == "fanout-state":
        result = fanout_state_stage(arguments.state_id)
    elif arguments.stage == "select-development-target":
        result = development_target_selection_stage()
    elif arguments.stage == "score-heldout":
        result = heldout_scores_stage()
    elif arguments.stage == "repeat-state":
        result = repeat_state_stage(arguments.state_id)
    elif arguments.stage == "assemble-row-evidence":
        result = assemble_row_evidence_stage()
    elif arguments.stage == "report":
        result = report_stage()
    else:  # pragma: no cover
        raise ExperimentError(f"unknown stage: {arguments.stage}")
    print(canonical_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as exc:
        print(f"{EXPERIMENT_ID}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
