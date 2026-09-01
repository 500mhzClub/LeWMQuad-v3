#!/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
"""Corrected direct runner for physical graph-edge handoff qualification V2.

V2 deliberately delegates every scientific operation to the frozen V1
runner.  It owns only the fresh namespace, immutable-V1 custody gates, the
exact persisted-array byte contract, and the mandatory first-eight physical
reproduction gate.  Importing this module neither opens either result root nor
constructs a simulator or model.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from lewm.safety import physical_graph_edge_handoff_qualification_v2_contract as CONTRACT
from lewm.safety import physical_graph_edge_handoff_qualification_v2_metrics as METRICS
from scripts import run_physical_graph_edge_handoff_qualification_v1 as V1


class ExperimentError(RuntimeError):
    """A V2 source, custody, persistence, or reproduction invariant failed."""


EXPERIMENT_ID = CONTRACT.EXPERIMENT_ID
PARENT_COMMIT = CONTRACT.SOURCE_PARENT_COMMIT
SOURCE_BASELINE_COMMIT = CONTRACT.SOURCE_BASELINE_COMMIT
FREEZE_SUBJECT = CONTRACT.CONTRACT_FREEZE_COMMIT_SUBJECT
RESULT_SUBJECT = CONTRACT.RESULT_COMMIT_SUBJECT
OUTPUT_ROOT = Path(CONTRACT.OUTPUT_ROOT)
MATERIAL_ROOT = Path(CONTRACT.MATERIAL_ROOT)
EXTERNAL_REGENERATION_RECEIPT = Path(CONTRACT.EXTERNAL_REGENERATION_RECEIPT)
V1_CUSTODY_RECEIPT = Path(CONTRACT.V1_CUSTODY_RECEIPT_PATH)
V1_OFFICIAL_ROOT = Path(CONTRACT.V1_OFFICIAL_ROOT)
V1_MATERIAL_ROOT = Path(CONTRACT.V1_MATERIAL_ROOT)

# These are scientific constants, not V2 redefinitions.
STATE_COUNT = V1.STATE_COUNT
PROSPECTIVE_POOL_COUNT = V1.PROSPECTIVE_POOL_COUNT
DEVELOPMENT_COUNT = V1.DEVELOPMENT_COUNT
HELDOUT_COUNT = V1.HELDOUT_COUNT
CANDIDATE_COUNT = V1.CANDIDATE_COUNT
PHYSICS_STEPS_PER_BRANCH = V1.PHYSICS_STEPS_PER_BRANCH
TRACE_DT_S = V1.TRACE_DT_S
SCIENTIFIC_LEAVES = (
    *V1.SCIENTIFIC_LEAVES,
    CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["v1_custody_and_nonreuse"],
    CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["scientific_invariance_receipt"],
    CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["v1_v2_first_eight_reproduction"],
)
PUBLICATION_LEAVES = tuple(V1.PUBLICATION_LEAVES)
ALL_OUTPUT_LEAVES = SCIENTIFIC_LEAVES + PUBLICATION_LEAVES

DOC_PATHS = {
    "contract": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_contract_2026-09-01.json",
    "fixture": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_fixture_2026-09-01.json",
    "output_schema": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_output_schema_2026-09-01.json",
    "preregistration": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_preregistration_2026-09-01.md",
    "source_closure": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_source_closure_2026-09-01.json",
    "scientific_invariance": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_scientific_invariance_2026-09-01.json",
    "v1_custody_binding": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_v1_custody_binding_2026-09-01.json",
}

_ORIGINAL_V1_WRITER = V1._write_material_shard
_ORIGINAL_V1_LOADER = V1._load_material_shard
_ORIGINAL_V1_NORMALISE_SNAPSHOT = V1._normalise_snapshot
_ORIGINAL_V1_ATOMIC_JSON = V1.atomic_json
_ORIGINAL_V1_ATOMIC_JSONL = V1.atomic_jsonl
_ORIGINAL_V1_WRITE_PUBLICATION = V1._write_publication
_ORIGINAL_V1_EDGE_PORT_RECORD = V1._edge_port_record
_ORIGINAL_V1_DERIVE_CANDIDATE_OUTCOME = V1.derive_candidate_outcome


def canonical_bytes(value: Any) -> bytes:
    return CONTRACT.canonical_json_bytes(value).rstrip(b"\n") + b"\n"


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persisted_array_sha256(value: Any) -> str:
    """Hash only exact C-order bytes, with no dtype conversion or header."""

    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _array_row(member: str, value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return {
        "member": str(member),
        "dtype_str": array.dtype.str,
        "shape": [int(item) for item in array.shape],
        "c_contiguous": True,
        "array_bytes_sha256": persisted_array_sha256(array),
    }


def _require_regular(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ExperimentError(f"required file is absent: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ExperimentError(f"path is not an ordinary regular file: {path}")
    return info


def _ordinary_json(path: Path) -> dict[str, Any]:
    _require_regular(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ExperimentError(f"noncanonical ordinary JSON: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    V1.atomic_bytes(path, canonical_bytes(dict(value)))


def _v2_identity(value: Any) -> Any:
    """Change only V1 schema/experiment identity and rebuild content digests."""
    if isinstance(value, Mapping):
        schema = value.get("schema")
        if isinstance(schema, str) and (
            "physical_graph_edge_handoff_qualification_v2" in schema
        ):
            # A native V2 authority may intentionally embed immutable V1
            # projections and custody bindings.  Re-projecting it would mutate
            # those frozen nested values, so the adapter is root-identity gated.
            return copy.deepcopy(dict(value))
    return METRICS.project_v1_evidence_to_v2(value)


def _v2_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_json(path, _v2_identity(value))


def _v2_atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    V1.atomic_bytes(
        path,
        b"".join(canonical_bytes(_v2_identity(dict(row))) for row in rows),
    )


def _payload_binding(path: Path, root: Path) -> dict[str, Any]:
    info = _require_regular(path)
    try:
        relative = str(path.relative_to(root))
    except ValueError as exc:
        raise ExperimentError("material payload escapes its bound root") from exc
    return {"path": relative, "bytes": int(info.st_size), "sha256": sha256_file(path)}


def validate_persisted_array_payload(
    directory: Path,
    metadata: Mapping[str, Any],
    *,
    root: Path = MATERIAL_ROOT,
) -> dict[str, Any]:
    """Reopen a material NPZ and prove its logical arrays against metadata."""

    import numpy as np

    evidence_value = metadata.get("persisted_array_evidence")
    if not isinstance(evidence_value, Mapping):
        raise ExperimentError("persisted-array evidence is absent")
    evidence = dict(evidence_value)
    if set(evidence) != set(CONTRACT.PERSISTED_ARRAY_EVIDENCE_FIELDS):
        raise ExperimentError("persisted-array evidence field drift")
    if set(evidence["payload_file"]) != set(CONTRACT.PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS):
        raise ExperimentError("persisted-array payload binding field drift")
    path = directory / "payload.npz"
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            METRICS.validate_npz_archive_comment(archive.comment)
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        raise ExperimentError("persisted-array NPZ provenance comment drift") from exc
    if evidence["payload_file"] != _payload_binding(path, root):
        raise ExperimentError("persisted-array payload binding drift")
    rows = evidence.get("arrays")
    if not isinstance(rows, list) or int(evidence.get("array_count", -1)) != len(rows):
        raise ExperimentError("persisted-array inventory cardinality drift")
    if hashlib.sha256(canonical_bytes(rows)[:-1]).hexdigest() != evidence.get(
        "array_inventory_sha256"
    ):
        raise ExperimentError("persisted-array inventory digest drift")
    expected: dict[str, dict[str, Any]] = {}
    for index, row_value in enumerate(rows):
        if not isinstance(row_value, Mapping) or set(row_value) != set(
            CONTRACT.PERSISTED_ARRAY_ROW_FIELDS
        ):
            raise ExperimentError(f"persisted-array row field drift: {index}")
        row = dict(row_value)
        member = str(row["member"])
        if member in expected or row["c_contiguous"] is not True:
            raise ExperimentError("persisted-array member/order drift")
        expected[member] = row
    if list(expected) != sorted(expected):
        raise ExperimentError("persisted-array members are not sorted")
    with np.load(path, allow_pickle=False) as payload:
        if sorted(payload.files) != list(expected):
            raise ExperimentError("persisted-array NPZ member inventory drift")
        arrays: dict[str, Any] = {}
        for member in sorted(payload.files):
            observed = np.ascontiguousarray(payload[member])
            row = _array_row(member, observed)
            if row != expected[member]:
                raise ExperimentError(f"persisted-array bytes/dtype/shape drift: {member}")
            arrays[member] = observed
    try:
        METRICS.validate_persisted_array_evidence(evidence, reopened_arrays=arrays)
        snapshot = metadata.get("snapshot")
        if (
            isinstance(snapshot, Mapping)
            and "snapshot__previous_applied_command" in arrays
        ):
            METRICS.validate_snapshot_previous_applied_command_binding(
                snapshot,
                evidence,
                reopened_arrays=arrays,
            )
    except Exception as exc:
        raise ExperimentError("pure persisted-array evidence validation failed") from exc
    return arrays


def _write_material_shard_impl(
    directory: Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    import numpy as np

    if directory.exists() or directory.is_symlink():
        raise ExperimentError(f"material shard is not fresh: {directory}")
    directory.mkdir(parents=False, exist_ok=False)
    prepared = {
        str(name): np.ascontiguousarray(np.asarray(value))
        for name, value in arrays.items()
    }
    if not prepared or len(prepared) != len(arrays):
        raise ExperimentError("material array inventory is empty or ambiguous")
    V1.atomic_npz(directory / "payload.npz", **prepared)
    # A deterministic V2-only ZIP comment makes container provenance explicit
    # without changing any persisted array member, dtype, shape, or value.  It
    # also makes a whole-file copy of an immutable V1 NPZ mechanically
    # impossible while the independently regenerated logical arrays remain
    # byte-comparable.
    payload_path = directory / "payload.npz"
    with zipfile.ZipFile(payload_path, mode="a") as archive:
        archive.comment = CONTRACT.NPZ_ARCHIVE_COMMENT.encode("utf-8")
    with payload_path.open("rb+") as stream:
        os.fsync(stream.fileno())
    with zipfile.ZipFile(payload_path, mode="r") as archive:
        METRICS.validate_npz_archive_comment(archive.comment)
    provisional = _v2_identity(dict(metadata))
    import numpy as np

    with np.load(directory / "payload.npz", allow_pickle=False) as archive:
        reopened_before_metadata = {
            name: np.ascontiguousarray(archive[name]) for name in archive.files
        }
    shard_kind = str(provisional.get("schema", "unknown")).rsplit(".", 2)[-2]
    try:
        evidence = METRICS.build_persisted_array_evidence(
            shard_kind=shard_kind,
            shard_id=directory.name,
            payload_file=_payload_binding(directory / "payload.npz", root),
            arrays=prepared,
            reopened_arrays=reopened_before_metadata,
        )
    except Exception as exc:
        raise ExperimentError("persisted-array save/reopen evidence failed") from exc
    provisional["persisted_array_evidence"] = evidence
    provisional["payload"] = {
        "role": "material_shard_payload",
        **evidence["payload_file"],
        "kind": "npz",
    }
    payload = V1.attach_digest(provisional)
    _atomic_json(directory / "metadata.json", payload)
    reopened = validate_persisted_array_payload(directory, payload, root=root)
    if list(reopened) != sorted(prepared):
        raise ExperimentError("post-save member order drift")
    for member, source in prepared.items():
        target = reopened[member]
        if source.dtype.str != target.dtype.str or source.shape != target.shape:
            raise ExperimentError(f"post-save dtype/shape drift: {member}")
        if source.tobytes(order="C") != target.tobytes(order="C"):
            raise ExperimentError(f"post-save logical array drift: {member}")


def _write_material_shard(
    directory: Path, metadata: Mapping[str, Any], arrays: Mapping[str, Any]
) -> None:
    _write_material_shard_impl(directory, metadata, arrays, root=MATERIAL_ROOT)


def _load_material_shard(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _ordinary_json(directory / "metadata.json")
    if "content_digest" not in metadata or V1.content_digest(metadata) != metadata["content_digest"]:
        raise ExperimentError(f"material metadata content digest drift: {directory}")
    arrays = validate_persisted_array_payload(directory, metadata, root=MATERIAL_ROOT)
    expected_payload = {
        "role": "material_shard_payload",
        **dict(metadata["persisted_array_evidence"])["payload_file"],
        "kind": "npz",
    }
    if metadata.get("payload") != expected_payload:
        raise ExperimentError(f"material payload/evidence cross-binding drift: {directory}")
    return metadata, arrays


def _normalise_snapshot(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use V1's exact normalisation, then bind hashes to the persisted arrays."""

    arrays, metadata = _ORIGINAL_V1_NORMALISE_SNAPSHOT(value)
    # V2 corrects the named defective field in the exact user-authorized hash
    # domain: only reopened C-order bytes.  dtype.str and shape are separate
    # members of persisted_array_evidence and are checked on immediate reopen.
    metadata["previous_applied_command_sha256"] = persisted_array_sha256(
        arrays["snapshot__previous_applied_command"]
    )
    return arrays, metadata


def _edge_port_record_authority_alignment(
    spec: Mapping[str, Any], teacher_row: Mapping[str, Any],
    teacher_trace: Mapping[str, Any], graph_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Align V1's inherited port pose with its already-frozen definition.

    The teacher trajectory owns the first-crossing position and remains the
    separate authority for crossing velocity and velocity heading.  The
    canonical directed-port *pose* has always been frozen to the directed
    opening normal; V1 accidentally copied the robot yaw at the interpolated
    crossing and never reached this assembly stage before its technical stop.
    """

    import math

    row = _ORIGINAL_V1_EDGE_PORT_RECORD(
        spec, teacher_row, teacher_trace, graph_record
    )
    inherited = copy.deepcopy(row)
    geometry = spec.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ExperimentError("canonical port geometry is absent")
    edge = geometry.get("selected_directed_edge")
    if not isinstance(edge, Mapping):
        raise ExperimentError("selected directed-edge geometry is absent")
    normal = edge.get("opening_normal_world")
    if (
        not isinstance(normal, Sequence)
        or isinstance(normal, (str, bytes))
        or len(normal) != 2
    ):
        raise ExperimentError("selected opening normal is malformed")
    nx, ny = float(normal[0]), float(normal[1])
    if not math.isfinite(nx) or not math.isfinite(ny) or math.hypot(nx, ny) <= 0.0:
        raise ExperimentError("selected opening normal is nonfinite or degenerate")
    port = row.get("directed_port_world")
    if (
        not isinstance(port, list)
        or len(port) != 3
        or not all(math.isfinite(float(value)) for value in port)
    ):
        raise ExperimentError("inherited directed-port pose is malformed")
    # Mutate exactly the one frozen-authority field.  The inherited crossing
    # position and every other record field remain byte-equivalent values.
    port[2] = math.atan2(ny, nx)
    preserved = copy.deepcopy(row)
    preserved["directed_port_world"][2] = inherited["directed_port_world"][2]
    if preserved != inherited:
        raise ExperimentError("port heading alignment changed an unauthorized field")
    return row


def _canonical_directed_port(state_id: str) -> list[float]:
    document = _ordinary_json(OUTPUT_ROOT / "edge_port_index.json")
    if (
        document.get("schema")
        != "physical_graph_edge_handoff_qualification_v2.edge_port_index.v1"
        or document.get("experiment_id") != EXPERIMENT_ID
        or "content_digest" not in document
        or V1.content_digest(document) != document["content_digest"]
    ):
        raise ExperimentError("canonical edge-port index identity/digest drift")
    records = document.get("records")
    if not isinstance(records, list):
        raise ExperimentError("canonical edge-port record inventory is absent")
    matches = [
        row
        for row in records
        if isinstance(row, Mapping) and row.get("state_id") == str(state_id)
    ]
    if len(matches) != 1:
        raise ExperimentError("canonical directed port is absent or ambiguous")
    port = matches[0].get("directed_port_world")
    if (
        not isinstance(port, list)
        or len(port) != 3
        or not all(isinstance(value, (int, float)) for value in port)
    ):
        raise ExperimentError("canonical directed-port pose is malformed")
    result = [float(value) for value in port]
    if not all(__import__("math").isfinite(value) for value in result):
        raise ExperimentError("canonical directed-port pose is nonfinite")
    return result


def _derive_candidate_outcome_authority_alignment(
    spec: Mapping[str, Any], trace: Mapping[str, Any],
    reset_pose_world: Sequence[float], candidate_index: int,
) -> dict[str, Any]:
    """Align only the three inherited candidate-port metric fields."""

    inherited = _ORIGINAL_V1_DERIVE_CANDIDATE_OUTCOME(
        spec, trace, reset_pose_world, candidate_index
    )
    port = _canonical_directed_port(str(spec["state_id"]))
    poses = trace.get("base_pose_world")
    if poses is None or len(poses) != PHYSICS_STEPS_PER_BRANCH:
        raise ExperimentError("candidate trace pose authority drift")
    try:
        aligned_fields = METRICS.derive_candidate_port_metrics(
            reset_pose_world, poses, port
        )
    except Exception as exc:
        raise ExperimentError("candidate directed-port metric alignment failed") from exc
    authorized_fields = {
        "port_progress_m", "lateral_error_m", "positive_port_progress"
    }
    if set(aligned_fields) != authorized_fields:
        raise ExperimentError("candidate port metric alignment field drift")
    aligned = copy.deepcopy(inherited)
    aligned.update(aligned_fields)
    preserved = copy.deepcopy(aligned)
    for field in authorized_fields:
        preserved[field] = copy.deepcopy(inherited[field])
    if preserved != inherited:
        raise ExperimentError("candidate port alignment changed an unauthorized field")
    return aligned


def _write_publication(metrics: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Delegate presentation while retaining the unchanged scientific predecessor."""

    context = CONTRACT.build_contract().get("v2_context_binding")
    if not isinstance(context, Mapping) or not isinstance(context.get("result_commit"), str):
        raise ExperimentError("V2 scientific predecessor context is absent")
    source_parent = V1.PARENT_COMMIT
    scientific_experiment = V1.EXPERIMENT_ID
    try:
        # V1 used one constant both for source-parent custody and the scientific
        # predecessor.  V2's source parent is the failed V1 commit, while its
        # unchanged scientific predecessor remains OGTB V2.  Scope the latter
        # only to the inherited publication builder.
        V1.PARENT_COMMIT = str(context["result_commit"])
        V1.EXPERIMENT_ID = EXPERIMENT_ID
        return _ORIGINAL_V1_WRITE_PUBLICATION(metrics, receipt)
    finally:
        V1.PARENT_COMMIT = source_parent
        V1.EXPERIMENT_ID = scientific_experiment


@contextlib.contextmanager
def _v2_namespace() -> Iterable[None]:
    """Temporarily bind the frozen V1 implementation to the isolated V2 namespace."""

    replacements = {
        "CONTRACT": CONTRACT,
        "METRICS": METRICS,
        # Deliberately leave V1.EXPERIMENT_ID unchanged during every scientific
        # operation.  It participates in the physical runner identity and the
        # frozen family-stratified split hash.  Storage adapters rewrite only
        # the emitted schema/experiment identity to V2.
        "PARENT_COMMIT": PARENT_COMMIT,
        "SOURCE_BASELINE_COMMIT": SOURCE_BASELINE_COMMIT,
        "FREEZE_SUBJECT": FREEZE_SUBJECT,
        "RESULT_SUBJECT": RESULT_SUBJECT,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "MATERIAL_ROOT": MATERIAL_ROOT,
        "EXTERNAL_REGENERATION_RECEIPT": EXTERNAL_REGENERATION_RECEIPT,
        "SCIENTIFIC_LEAVES": SCIENTIFIC_LEAVES,
        "PUBLICATION_LEAVES": PUBLICATION_LEAVES,
        "ALL_OUTPUT_LEAVES": ALL_OUTPUT_LEAVES,
        "DOC_PATHS": DOC_PATHS,
        "_write_material_shard": _write_material_shard,
        "_load_material_shard": _load_material_shard,
        "_normalise_snapshot": _normalise_snapshot,
        "_edge_port_record": _edge_port_record_authority_alignment,
        "derive_candidate_outcome": _derive_candidate_outcome_authority_alignment,
        "_write_publication": _write_publication,
        "atomic_json": _v2_atomic_json,
        "atomic_jsonl": _v2_atomic_jsonl,
        "require_runtime_source_freeze": require_runtime_source_freeze,
        "_validate_runtime_source_closure": _validate_runtime_source_closure,
    }
    saved = {name: getattr(V1, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(V1, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(V1, name, value)


def _delegate(name: str, *args: Any, **kwargs: Any) -> Any:
    with _v2_namespace():
        try:
            return _v2_identity(getattr(V1, name)(*args, **kwargs))
        except V1.ExperimentError as exc:
            raise ExperimentError(str(exc)) from exc


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ExperimentError(f"git {' '.join(arguments)} failed: {exc.output}") from exc


def _validate_runtime_source_closure() -> dict[str, Any]:
    path = DOC_PATHS["source_closure"]
    info = _require_regular(path)
    if info.st_nlink != 1:
        raise ExperimentError("source-closure document is not single-linked")
    value = _ordinary_json(path)
    if "content_digest" not in value or V1.content_digest(value) != value["content_digest"]:
        raise ExperimentError("source-closure content digest drift")
    expected_fields = {"schema", "parent_commit", "row_count", "rows", "content_digest"}
    if (
        set(value) != expected_fields
        or value["schema"] != f"{EXPERIMENT_ID.lower()}.source_closure.v1"
        or value["parent_commit"] != PARENT_COMMIT
    ):
        raise ExperimentError("source-closure identity drift")
    expected_paths = list(CONTRACT.SOURCE_CLOSURE_PATHS)
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != len(expected_paths) or value["row_count"] != len(rows):
        raise ExperimentError("source-closure cardinality drift")
    repo = REPO_ROOT.resolve(strict=True)
    for index, (row_value, relative) in enumerate(zip(rows, expected_paths)):
        if not isinstance(row_value, Mapping) or set(row_value) != {"path", "bytes", "sha256"}:
            raise ExperimentError(f"source-closure row field drift: {index}")
        candidate = Path(relative)
        path_value = REPO_ROOT / candidate
        file_info = _require_regular(path_value)
        observed = {
            "path": relative,
            "bytes": int(file_info.st_size),
            "sha256": sha256_file(path_value),
        }
        if (
            dict(row_value) != observed
            or file_info.st_nlink != 1
            or path_value.resolve(strict=True) != repo / candidate
        ):
            raise ExperimentError(f"source-closure live binding drift: {relative}")
    return {"path": str(path), "row_count": len(rows), "validated": True}


def require_runtime_source_freeze() -> str:
    head = _git("rev-parse", "HEAD")
    if _git("show", "-s", "--format=%s", head) != FREEZE_SUBJECT:
        raise ExperimentError("runtime HEAD is not the required V2 contract-freeze subject")
    if _git("rev-parse", f"{head}^") != PARENT_COMMIT:
        raise ExperimentError("V2 contract freeze is not a direct child of frozen V1")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ExperimentError("runtime worktree is not clean at the V2 source freeze")
    for relative in CONTRACT.TRACKED_SOURCE_PATHS:
        path = REPO_ROOT / relative
        _require_regular(path)
        frozen = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=REPO_ROOT)
        if path.read_bytes() != frozen:
            raise ExperimentError(f"runtime source differs from frozen V2 HEAD: {relative}")
    _validate_runtime_source_closure()
    return head


def _receipt_binding(path: Path) -> dict[str, Any]:
    info = _require_regular(path)
    return {"path": str(path), "bytes": int(info.st_size), "sha256": sha256_file(path)}


def _load_custody_evaluator(module: Any | None = None) -> Any:
    if module is not None:
        return module
    try:
        from scripts import evaluate_physical_graph_edge_handoff_qualification_v2 as evaluator
    except ImportError as exc:
        raise ExperimentError("V2 custody evaluator is unavailable") from exc
    return evaluator


def validate_v1_custody_before_creation(*, evaluator_module: Any | None = None) -> dict[str, Any]:
    """Validate the frozen external receipt and both live V1 trees read-only."""

    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink() or MATERIAL_ROOT.exists() or MATERIAL_ROOT.is_symlink():
        raise ExperimentError("V2 roots must be absent during V1 custody preflight")
    expected = dict(CONTRACT.V1_CUSTODY_RECEIPT_BINDING)
    observed = _receipt_binding(V1_CUSTODY_RECEIPT)
    if expected != observed:
        raise ExperimentError("external V1 custody receipt binding drift")
    evaluator = _load_custody_evaluator(evaluator_module)
    validator = getattr(evaluator, "validate_existing_v1_custody_receipt", None)
    if not callable(validator):
        raise ExperimentError("external V1 custody validator API is absent")
    receipt = validator(
        V1_CUSTODY_RECEIPT,
        official_root=V1_OFFICIAL_ROOT,
        material_root=V1_MATERIAL_ROOT,
    )
    if not isinstance(receipt, Mapping):
        raise ExperimentError("external V1 custody validator returned no receipt")
    try:
        validated = METRICS.validate_external_v1_custody_receipt(receipt)
    except Exception as exc:
        raise ExperimentError("external V1 custody scientific projection drift") from exc
    return copy.deepcopy(dict(validated))


def _regression_results() -> list[dict[str, Any]]:
    """Run the ten storage/invariance regressions before simulator creation."""

    import numpy as np

    fixture = np.asarray(CONTRACT.REGRESSION_FIXTURE["values"], dtype=np.float64)
    raw_digest = persisted_array_sha256(fixture)
    cast_digest = persisted_array_sha256(fixture.astype(np.float32))
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    noncontiguous = base[:, ::2]
    results: dict[str, tuple[bool, Any]] = {
        "FLOAT64_EXACT_PERSISTED_BYTES_VALID": (
            raw_digest == hashlib.sha256(fixture.tobytes(order="C")).hexdigest(), raw_digest
        ),
        "FLOAT32_CAST_DIGEST_REJECTED": (cast_digest != raw_digest, cast_digest),
        "PERSISTED_DTYPE_MISMATCH_REJECTED": (
            _array_row("x", fixture)["dtype_str"] != _array_row("x", fixture.astype(np.float32))["dtype_str"],
            [fixture.dtype.str, fixture.astype(np.float32).dtype.str],
        ),
        "PERSISTED_SHAPE_MISMATCH_REJECTED": (
            _array_row("x", fixture)["shape"] != _array_row("x", fixture.reshape(3, 2))["shape"],
            [list(fixture.shape), [3, 2]],
        ),
        "PERSISTED_VALUE_MUTATION_REJECTED": (
            persisted_array_sha256(fixture + np.finfo(np.float64).eps) != raw_digest,
            "one-ulp mutation",
        ),
        "NONCONTIGUOUS_VIEW_C_CONTIGUOUS_NO_DTYPE_CAST": (
            (not noncontiguous.flags.c_contiguous)
            and np.ascontiguousarray(noncontiguous).dtype == noncontiguous.dtype,
            noncontiguous.dtype.str,
        ),
        "OTHER_METADATA_BINDINGS_UNCHANGED": (
            set(V1.SNAPSHOT_METADATA_FIELDS) == set(V1.SNAPSHOT_METADATA_FIELDS),
            "normalizer changes only exact array-derived hashes",
        ),
        "V1_SCIENTIFIC_CONSTANTS_AND_SOURCE_PATHS_UNCHANGED": (
            CONTRACT.scientific_invariance_projection(CONTRACT.build_contract())
            == CONTRACT.V1_SCIENTIFIC_PROJECTION
            and CONTRACT.build_candidate_specs() == V1.CONTRACT.build_candidate_specs()
            and tuple(CONTRACT.SOURCE_DEPENDENCY_PATHS) == tuple(V1.CONTRACT.SOURCE_DEPENDENCY_PATHS),
            CONTRACT.V1_SCIENTIFIC_PROJECTION_SHA256,
        ),
    }
    with tempfile.TemporaryDirectory(prefix="pgehq-v2-writer-fixture-") as raw:
        root = Path(raw)
        shard = root / "pool-000"
        fixture_arrays = {"float64_fixture": fixture, "noncontiguous_fixture": noncontiguous}
        _write_material_shard_impl(
            shard,
            {"schema": f"{EXPERIMENT_ID.lower()}.writer_fixture.v1", "experiment_id": EXPERIMENT_ID},
            fixture_arrays,
            root=root,
        )
        metadata = _ordinary_json(shard / "metadata.json")
        reopened = validate_persisted_array_payload(shard, metadata, root=root)
        results["SAVE_RELOAD_DIGEST_IDENTICAL"] = (
            persisted_array_sha256(reopened["float64_fixture"]) == raw_digest,
            persisted_array_sha256(reopened["float64_fixture"]),
        )
        results["FIRST_POOL_PRODUCTION_WRITER_VALIDATOR_PASS"] = (
            set(reopened) == set(fixture_arrays), "production writer/reopen/validator"
        )
    numeric = {
        field: np.zeros(shape, dtype=(np.float32 if field == "previous_applied_command" else np.float64))
        for field, shape in V1.SNAPSHOT_NUMERIC_FIELDS.items()
    }
    numeric["camera_world_transform"] = np.eye(4, dtype=np.float64)
    metadata = {
        "serialized_solver_state_sha256": "1" * 64,
        "serialized_controller_state_sha256": "2" * 64,
        "serialized_rng_state_sha256": "3" * 64,
        "policy_last_action_sha256": V1.canonical_array_sha256(numeric["policy_last_action"]),
        "capture_timestamp_s": 0.0,
        "controller_observation_sha256": V1.canonical_array_sha256(numeric["controller_observation"]),
        "previous_policy_action_sha256": V1.canonical_array_sha256(numeric["previous_policy_action"]),
        "torch_cpu_rng_state_sha256": "4" * 64,
        "torch_device_rng_state_sha256s": [],
        "torch_device_count": 0,
        "previous_applied_command_sha256": V1.canonical_array_sha256(numeric["previous_applied_command"]),
        "command_history_sha256": V1.canonical_array_sha256(numeric["command_history"]),
        "control_history_sha256": V1.canonical_array_sha256(numeric["control_history"]),
        "low_level_policy_state_sha256": V1.canonical_array_sha256(numeric["low_level_policy_state"]),
        "solver_field_inventory": ["fixture.solver"],
        "controller_field_inventory": ["fixture.controller"],
        "rng_field_inventory": ["fixture.rng"],
    }
    _normalised_arrays, corrected = _normalise_snapshot(
        {"payload_bytes": b"fixture", **numeric, **metadata}
    )
    unchanged_fields = set(metadata) - {
        "previous_applied_command_sha256"
    }
    other_metadata_unchanged = bool(
        all(corrected[name] == metadata[name] for name in unchanged_fields)
        and corrected["previous_applied_command_sha256"]
        == persisted_array_sha256(
            _normalised_arrays["snapshot__previous_applied_command"]
        )
        and corrected["previous_applied_command_sha256"]
        != metadata["previous_applied_command_sha256"]
    )
    # The pure authority owns the exact evidence strings and negative cases;
    # the runner contributes only the fact that its actual production writer
    # and immediate reopen validator passed above.
    rows = METRICS.build_regression_results(
        first_pool_production_writer_validator_passed=bool(
            results["FIRST_POOL_PRODUCTION_WRITER_VALIDATOR_PASS"][0]
        ),
        other_metadata_bindings_unchanged=other_metadata_unchanged,
    )
    if len(rows) != 10 or not all(row["passed"] for row in rows):
        raise ExperimentError("pre-simulator persisted-array regression gate failed")
    return rows


def build_scientific_invariance_receipt() -> dict[str, Any]:
    contract = CONTRACT.build_contract()
    regressions = _regression_results()
    value = METRICS.build_scientific_invariance_receipt(contract, regressions)
    METRICS.validate_scientific_invariance_receipt(value)
    if set(value) != set(CONTRACT.SCIENTIFIC_INVARIANCE_RECEIPT_FIELDS):
        raise ExperimentError("scientific invariance receipt field drift")
    return value


def _custody_nonreuse_receipt(source_freeze: str, external: Mapping[str, Any]) -> dict[str, Any]:
    # The external value was exact-validated immediately before this builder;
    # the pure builder owns the canonical projection and strict count types.
    METRICS.validate_external_v1_custody_receipt(external)
    return METRICS.build_v1_custody_and_nonreuse(
        source_freeze_commit=source_freeze
    )


def _tree_file_inventory(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ExperimentError(f"custody root is not an ordinary directory: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        info = _require_regular(path)
        if info.st_nlink != 1:
            raise ExperimentError(f"custody file is not single-linked: {path}")
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "device": int(info.st_dev),
                "inode": int(info.st_ino),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _assert_v1_v2_storage_isolation() -> None:
    v1_rows = _tree_file_inventory(V1_OFFICIAL_ROOT) + _tree_file_inventory(V1_MATERIAL_ROOT)
    v2_rows = _tree_file_inventory(OUTPUT_ROOT) + _tree_file_inventory(MATERIAL_ROOT)
    v1_inodes = {(row["device"], row["inode"]) for row in v1_rows}
    if any((row["device"], row["inode"]) in v1_inodes for row in v2_rows):
        raise ExperimentError("V2 contains an inode shared with immutable V1")
    v1_shard_hashes = {
        row["sha256"]
        for row in v1_rows
        if row["path"].endswith(("/metadata.json", "/payload.npz"))
    }
    if any(row["sha256"] in v1_shard_hashes for row in v2_rows):
        raise ExperimentError("V2 contains an exact V1 shard-file copy")


def initialize_stage(
    *, fake_runtime: bool = False, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Validate V1 custody and regression gates before creating fresh V2 roots."""

    external = validate_v1_custody_before_creation(evaluator_module=evaluator_module)
    invariance = build_scientific_invariance_receipt()
    source_freeze = require_runtime_source_freeze()
    result = _delegate("initialize_stage", fake_runtime=fake_runtime)
    # Rebuild the external receipt after V2 creation to prove that initialization
    # did not mutate V1, then directly prove no V1 file/inode entered V2.
    evaluator = _load_custody_evaluator(evaluator_module)
    validator = getattr(evaluator, "validate_existing_v1_custody_receipt", None)
    if not callable(validator):
        raise ExperimentError("external V1 custody validator API is absent")
    after = validator(
        V1_CUSTODY_RECEIPT,
        official_root=V1_OFFICIAL_ROOT,
        material_root=V1_MATERIAL_ROOT,
    )
    if not isinstance(after, Mapping) or canonical_bytes(after) != canonical_bytes(external):
        raise ExperimentError("immutable V1 custody changed during V2 initialization")
    _assert_v1_v2_storage_isolation()
    custody = _custody_nonreuse_receipt(source_freeze, external)
    _atomic_json(OUTPUT_ROOT / CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["v1_custody_and_nonreuse"], custody)
    _atomic_json(OUTPUT_ROOT / CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["scientific_invariance_receipt"], invariance)
    return result


def _reproduction_path() -> Path:
    return OUTPUT_ROOT / CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["v1_v2_first_eight_reproduction"]


def _require_reproduction_pass() -> dict[str, Any]:
    value = _ordinary_json(_reproduction_path())
    if set(value) != set(CONTRACT.FIRST_EIGHT_REPRODUCTION_FIELDS):
        raise ExperimentError("first-eight reproduction receipt field drift")
    if value.get("pass") is not True or value.get("full_collection_authorized") is not True:
        raise ExperimentError(CONTRACT.REPRODUCTION_MISMATCH_DISPOSITION)
    return value


def qualify_pool_state_stage(
    pool_index: int, *, backend: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    index = int(pool_index)
    if index >= int(CONTRACT.FIRST_EIGHT_REPRODUCTION_AUTHORITY["required_before_pool_index"]):
        _require_reproduction_pass()
    result = _delegate(
        "qualify_pool_state_stage", index, backend=backend, fake_runtime=fake_runtime
    )
    # Eligibility is exposed only after the just-written shard has reopened and
    # passed the exact array validator.
    metadata, _arrays = _load_material_shard(MATERIAL_ROOT / "qualification" / f"pool-{index:03d}")
    if bool(metadata.get("qualified")) != bool(result.get("qualified")):
        raise ExperimentError("post-reopen qualification value drift")
    return metadata


def _teacher_stuck(arrays: Mapping[str, Any]) -> bool:
    import math
    import numpy as np

    pose = np.asarray(arrays["teacher__base_pose_world"], dtype=np.float64)
    requested = np.asarray(arrays["teacher__requested_command"], dtype=np.float64)
    start = pose[0]
    end = pose[-1]
    start_yaw = V1._pose_yaw_xyzw(start)
    body = V1.body_from_world(
        [float(end[0]), float(end[1]), V1._pose_yaw_xyzw(end)],
        [float(start[0]), float(start[1]), start_yaw],
    )
    rule = CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]
    return bool(
        float(np.max(np.abs(requested))) > float(rule["command_activity_threshold"])
        and math.hypot(body[0], body[1]) < float(rule["h3_translation_threshold_m"])
        and abs(V1._wrap_angle(body[2])) < float(rule["h3_heading_threshold_rad"])
    )


def _first_eight_row(index: int) -> dict[str, Any]:
    v1_directory = V1_MATERIAL_ROOT / "qualification" / f"pool-{index:03d}"
    v2_directory = MATERIAL_ROOT / "qualification" / f"pool-{index:03d}"
    v1_metadata, v1_arrays = _ORIGINAL_V1_LOADER(v1_directory)
    v2_metadata, v2_arrays = _load_material_shard(v2_directory)
    expected_spec = dict(CONTRACT.build_prospective_pool_specs()[index])
    v1_spec = (
        dict(v1_metadata["candidate_spec"])
        if isinstance(v1_metadata.get("candidate_spec"), Mapping)
        else {}
    )
    v2_spec = (
        dict(v2_metadata["candidate_spec"])
        if isinstance(v2_metadata.get("candidate_spec"), Mapping)
        else {}
    )
    identities = ("candidate_spec_id", "state_id", "scene_id", "episode_id", "graph_id")
    identity_equal = all(
        v1_spec.get(name) == v2_spec.get(name) == expected_spec.get(name)
        for name in identities
    )
    # Every non-storage, non-source-freeze metadata value must also reproduce.
    # The sole permitted scientific-metadata exception is the already-bound V1
    # previous-command hash, whose underlying reopened arrays remain exact.
    def comparable_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
        projected = _v2_identity(value)
        for name in ("content_digest", "payload", "persisted_array_evidence"):
            projected.pop(name, None)
        snapshot_value = projected.get("snapshot")
        if not isinstance(snapshot_value, Mapping):
            return {"invalid_snapshot_metadata": True, **projected}
        snapshot = dict(snapshot_value)
        snapshot.pop("previous_applied_command_sha256", None)
        projected["snapshot"] = snapshot
        return projected

    metadata_equal = comparable_metadata(v1_metadata) == comparable_metadata(v2_metadata)
    v1_payload_info = _require_regular(v1_directory / "payload.npz")
    v2_payload_info = _require_regular(v2_directory / "payload.npz")
    storage_isolated = bool(
        v1_payload_info.st_nlink == 1
        and v2_payload_info.st_nlink == 1
        and (v1_payload_info.st_dev, v1_payload_info.st_ino)
        != (v2_payload_info.st_dev, v2_payload_info.st_ino)
    )
    v1_members = sorted(v1_arrays)
    v2_members = sorted(v2_arrays)
    shared = sorted(set(v1_members) & set(v2_members))
    teacher_v1 = sorted(name for name in v1_members if name.startswith("teacher__"))
    teacher_v2 = sorted(name for name in v2_members if name.startswith("teacher__"))
    dtypes_equal = all(v1_arrays[name].dtype.str == v2_arrays[name].dtype.str for name in shared)
    shapes_equal = all(v1_arrays[name].shape == v2_arrays[name].shape for name in shared)
    logical_equal = all(
        v1_arrays[name].tobytes(order="C") == v2_arrays[name].tobytes(order="C")
        for name in shared
    )
    teacher_inventory_equal = teacher_v1 == teacher_v2
    teacher_logical = (
        teacher_inventory_equal
        and all(
            v1_arrays[name].tobytes(order="C") == v2_arrays[name].tobytes(order="C")
            for name in teacher_v1
        )
    )
    known = ["snapshot.previous_applied_command_sha256"]
    previous_member = "snapshot__previous_applied_command"
    expected_v2_previous = (
        persisted_array_sha256(v2_arrays[previous_member])
        if previous_member in v2_arrays
        else None
    )
    v1_snapshot = v1_metadata.get("snapshot")
    v2_snapshot = v2_metadata.get("snapshot")
    known_defect_exact = (
        expected_v2_previous is not None
        and isinstance(v1_snapshot, Mapping)
        and isinstance(v2_snapshot, Mapping)
        and v1_snapshot.get("previous_applied_command_sha256") != expected_v2_previous
        and v2_snapshot.get("previous_applied_command_sha256") == expected_v2_previous
    )
    snapshot_payload_member = "snapshot_payload_bytes"
    contact_member = "teacher__physics_contact"
    stuck_members = {"teacher__base_pose_world", "teacher__requested_command"}
    snapshot_payload_equal = bool(
        snapshot_payload_member in v1_arrays
        and snapshot_payload_member in v2_arrays
        and isinstance(v1_snapshot, Mapping)
        and isinstance(v2_snapshot, Mapping)
        and v1_snapshot.get("snapshot_payload_sha256")
        == v2_snapshot.get("snapshot_payload_sha256")
        and v1_arrays[snapshot_payload_member].tobytes()
        == v2_arrays[snapshot_payload_member].tobytes()
    )
    contact_equal = bool(
        contact_member in v1_arrays
        and contact_member in v2_arrays
        and v1_arrays[contact_member].tobytes()
        == v2_arrays[contact_member].tobytes()
    )
    stuck_equal = bool(
        stuck_members.issubset(v1_arrays)
        and stuck_members.issubset(v2_arrays)
        and _teacher_stuck(v1_arrays) == _teacher_stuck(v2_arrays)
    )
    booleans = {
        "identity_equal": identity_equal and v1_spec == v2_spec == expected_spec and metadata_equal and storage_isolated,
        "snapshot_payload_sha256_equal": snapshot_payload_equal,
        "teacher_trace_member_inventory_equal": teacher_inventory_equal,
        "teacher_trace_dtypes_equal": teacher_inventory_equal and all(
            v1_arrays[name].dtype.str == v2_arrays[name].dtype.str for name in teacher_v1
        ),
        "teacher_trace_shapes_equal": teacher_inventory_equal and all(
            v1_arrays[name].shape == v2_arrays[name].shape for name in teacher_v1
        ),
        "teacher_trace_logical_arrays_equal": teacher_logical,
        "contact_sequence_equal": contact_equal,
        "stuck_equal": stuck_equal,
        "qualified_equal": v1_metadata["qualified"] is v2_metadata["qualified"],
        "rejection_reason_equal": v1_metadata["rejection_reason"] == v2_metadata["rejection_reason"],
        "all_payload_member_dtypes_equal": v1_members == v2_members and dtypes_equal,
        "all_payload_member_shapes_equal": v1_members == v2_members and shapes_equal,
        "shared_logical_array_members_equal": v1_members == v2_members and logical_equal,
    }
    passed = all(booleans.values()) and known_defect_exact
    row = {
        "pool_index": index,
        "candidate_spec_id": str(expected_spec["candidate_spec_id"]),
        "state_id": str(expected_spec["state_id"]),
        "scene_id": str(expected_spec["scene_id"]),
        "episode_id": str(expected_spec["episode_id"]),
        "graph_id": str(expected_spec["graph_id"]),
        **booleans,
        "noncomparable_v1_known_bad_hash_fields": known,
        "pass": passed,
    }
    if set(row) != set(CONTRACT.FIRST_EIGHT_REPRODUCTION_ROW_FIELDS):
        raise ExperimentError("first-eight reproduction row field drift")
    return row


def compare_v1_first_eight_stage(
    *, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Compare fresh V2 pool 0..7 to immutable V1 before opening pool 8."""

    if _reproduction_path().exists():
        raise ExperimentError("first-eight reproduction receipt is not fresh")
    # Revalidate external custody and live roots, now permitting the fresh V2
    # roots but never passing V1 arrays into a V2 execution function.
    expected = dict(CONTRACT.V1_CUSTODY_RECEIPT_BINDING)
    if _receipt_binding(V1_CUSTODY_RECEIPT) != expected:
        raise ExperimentError("external V1 custody receipt binding drift")
    evaluator = _load_custody_evaluator(evaluator_module)
    validator = getattr(evaluator, "validate_existing_v1_custody_receipt", None)
    if not callable(validator):
        raise ExperimentError("external V1 custody validator API is absent")
    after = validator(
        V1_CUSTODY_RECEIPT,
        official_root=V1_OFFICIAL_ROOT,
        material_root=V1_MATERIAL_ROOT,
    )
    if not isinstance(after, Mapping):
        raise ExperimentError("external V1 custody validator returned no receipt")
    rows = [_first_eight_row(index) for index in range(8)]
    _assert_v1_v2_storage_isolation()
    final_custody = validator(
        V1_CUSTODY_RECEIPT,
        official_root=V1_OFFICIAL_ROOT,
        material_root=V1_MATERIAL_ROOT,
    )
    if not isinstance(final_custody, Mapping) or canonical_bytes(final_custody) != canonical_bytes(after):
        raise ExperimentError("immutable V1 custody changed during first-eight comparison")
    passed = all(row["pass"] for row in rows)
    value = METRICS.build_first_eight_reproduction(rows)
    if value["pass"] is not passed:
        raise ExperimentError("first-eight reproduction builder disposition drift")
    try:
        METRICS.validate_first_eight_reproduction(value)
    except Exception as exc:
        raise ExperimentError("first-eight reproduction authority validation failed") from exc
    _atomic_json(_reproduction_path(), value)
    if not passed:
        observed = sorted(path.name for path in OUTPUT_ROOT.iterdir())
        if observed != sorted(CONTRACT.REPRODUCTION_MISMATCH_LEAVES):
            raise ExperimentError("technical mismatch output inventory drift")
    return value


def _gated_delegate(name: str, *args: Any, **kwargs: Any) -> Any:
    _require_reproduction_pass()
    return _delegate(name, *args, **kwargs)


def select_teacher_pool_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    return _gated_delegate("select_teacher_pool_stage", fake_runtime=fake_runtime)


def capture_selected_state_stage(
    state_id: str, *, backend: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    return _gated_delegate(
        "capture_selected_state_stage", state_id, backend=backend, fake_runtime=fake_runtime
    )


def freeze_panel_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    return _gated_delegate("freeze_panel_stage", fake_runtime=fake_runtime)


def encode_canonical_pixels_stage(
    *, encoder: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    return _gated_delegate(
        "encode_canonical_pixels_stage", encoder=encoder, fake_runtime=fake_runtime
    )


def fanout_state_stage(
    state_id: str, *, backend: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    return _gated_delegate(
        "fanout_state_stage", state_id, backend=backend, fake_runtime=fake_runtime
    )


def development_target_selection_stage(
    *, ranker: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    return _gated_delegate(
        "development_target_selection_stage", ranker=ranker, fake_runtime=fake_runtime
    )


def heldout_ranker_scores_stage(
    *, ranker: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    return _gated_delegate(
        "heldout_ranker_scores_stage", ranker=ranker, fake_runtime=fake_runtime
    )


def repeat_state_stage(
    state_id: str, *, backend: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    return _gated_delegate(
        "repeat_state_stage", state_id, backend=backend, fake_runtime=fake_runtime
    )


def assemble_row_evidence_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    return _gated_delegate("assemble_row_evidence_stage", fake_runtime=fake_runtime)


def report_stage(*, fake_runtime: bool = False, evaluator_module: Any | None = None) -> dict[str, Any]:
    if evaluator_module is None:
        evaluator_module = _load_custody_evaluator()
    return _gated_delegate(
        "report_stage", fake_runtime=fake_runtime, evaluator_module=evaluator_module
    )


def physical_backend_smoke_stage() -> dict[str, Any]:
    return _delegate("physical_backend_smoke_stage")


def _preregistration_text() -> str:
    return (
        "# PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V2\n\n"
        "Development-only exact scientific rerun of V1 with corrected persisted-array "
        "byte custody. V1 artifacts are immutable and are not execution inputs.\n\n"
        "## Frozen-authority implementation alignments\n\n"
        f"- `{CONTRACT.PORT_HEADING_ALIGNMENT_DISPOSITION}` changes only "
        "`directed_port_world[2]`, setting it to the already-frozen selected "
        "opening-normal heading. Crossing position, velocity evidence, indices, "
        "fraction, lookahead, remaining length, and every other edge-port field "
        "remain unchanged.\n"
        f"- `{CONTRACT.CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION}` changes only "
        "`port_progress_m`, `lateral_error_m`, and dependent "
        "`positive_port_progress`, evaluating the already-frozen formulas against "
        "the canonical actual-teacher-crossing port. Every other candidate outcome "
        "field remains unchanged.\n\n"
        "These are bounded inherited-implementation corrections to the prospectively "
        "frozen V1 authority. They change no frozen gate, formula, threshold, "
        "classification precedence, tuning choice, model, checkpoint, candidate bank, "
        "panel identity, split role, or outcome-access order.\n"
    )


def build_freeze_documents() -> dict[str, Any]:
    """Generate prospective V2 documents only after custody bytes are frozen."""

    if _git("rev-parse", "HEAD") != PARENT_COMMIT:
        raise ExperimentError("V2 freeze documents require the frozen V1 result commit")
    if int(CONTRACT.V1_CUSTODY_RECEIPT_BINDING["bytes"]) <= 0:
        raise ExperimentError("external V1 custody receipt binding is not frozen")
    scientific = CONTRACT.build_contract()
    contract_document = V1.attach_digest(
        {
            "schema": f"{EXPERIMENT_ID.lower()}.contract_document.v1",
            "status": "FROZEN_BEFORE_PHYSICAL_COLLECTION",
            "parent_commit": PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "required_freeze_subject": FREEZE_SUBJECT,
            "required_result_subject": RESULT_SUBJECT,
            "scientific_contract": scientific,
        }
    )
    fixture = V1.attach_digest(
        {
            "schema": f"{EXPERIMENT_ID.lower()}.fixture.v1",
            "prospective_pool_states": PROSPECTIVE_POOL_COUNT,
            "selected_states": STATE_COUNT,
            "first_eight_gate": list(range(8)),
            "persisted_array_hash_authority": copy.deepcopy(CONTRACT.PERSISTED_ARRAY_HASH_AUTHORITY),
            "regression_gate": copy.deepcopy(CONTRACT.REGRESSION_GATE_AUTHORITY),
        }
    )
    output_schema = V1.attach_digest(
        {
            "schema": f"{EXPERIMENT_ID.lower()}.output_schema.v1",
            "root": str(OUTPUT_ROOT),
            "material_root": str(MATERIAL_ROOT),
            "success_leaves": list(CONTRACT.SUCCESS_OUTPUT_LEAVES),
            "mismatch_leaves": list(CONTRACT.REPRODUCTION_MISMATCH_LEAVES),
            "external_regeneration_receipt": str(EXTERNAL_REGENERATION_RECEIPT),
            "receipt_self_digests": False,
        }
    )
    preregistration = _preregistration_text()
    invariance = V1.attach_digest(copy.deepcopy(CONTRACT.SCIENTIFIC_INVARIANCE_AUTHORITY))
    custody = V1.attach_digest(copy.deepcopy(CONTRACT.V1_CUSTODY_AND_NONREUSE_AUTHORITY))
    _atomic_json(DOC_PATHS["contract"], contract_document)
    _atomic_json(DOC_PATHS["fixture"], fixture)
    _atomic_json(DOC_PATHS["output_schema"], output_schema)
    V1.atomic_bytes(DOC_PATHS["preregistration"], preregistration.encode("utf-8"))
    _atomic_json(DOC_PATHS["scientific_invariance"], invariance)
    _atomic_json(DOC_PATHS["v1_custody_binding"], custody)
    rows = []
    for relative in CONTRACT.SOURCE_CLOSURE_PATHS:
        path = REPO_ROOT / relative
        info = _require_regular(path)
        rows.append({"path": relative, "bytes": int(info.st_size), "sha256": sha256_file(path)})
    _atomic_json(
        DOC_PATHS["source_closure"],
        V1.attach_digest(
            {
                "schema": f"{EXPERIMENT_ID.lower()}.source_closure.v1",
                "parent_commit": PARENT_COMMIT,
                "row_count": len(rows),
                "rows": rows,
            }
        ),
    )
    return contract_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("freeze-docs")
    subparsers.add_parser("initialize")
    qualify = subparsers.add_parser("qualify-pool-state")
    qualify.add_argument("--pool-index", type=int, required=True)
    subparsers.add_parser("compare-v1-first-eight")
    subparsers.add_parser("select-teacher-pool")
    reset = subparsers.add_parser("qualify-selected-reset")
    reset.add_argument("--state-id", required=True)
    subparsers.add_parser("freeze-panel")
    subparsers.add_parser("encode-canonical-pixels")
    fanout = subparsers.add_parser("fanout-state")
    fanout.add_argument("--state-id", required=True)
    subparsers.add_parser("select-development-target")
    subparsers.add_parser("score-heldout")
    repeat = subparsers.add_parser("repeat-state")
    repeat.add_argument("--state-id", required=True)
    subparsers.add_parser("assemble-row-evidence")
    subparsers.add_parser("report")
    subparsers.add_parser("physical-smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    arguments = build_parser().parse_args(argv)
    stage = arguments.stage
    if stage == "freeze-docs": result: Any = build_freeze_documents()
    elif stage == "initialize": result = initialize_stage()
    elif stage == "qualify-pool-state": result = qualify_pool_state_stage(arguments.pool_index)
    elif stage == "compare-v1-first-eight": result = compare_v1_first_eight_stage()
    elif stage == "select-teacher-pool": result = select_teacher_pool_stage()
    elif stage == "qualify-selected-reset": result = capture_selected_state_stage(arguments.state_id)
    elif stage == "freeze-panel": result = freeze_panel_stage()
    elif stage == "encode-canonical-pixels": result = encode_canonical_pixels_stage()
    elif stage == "fanout-state": result = fanout_state_stage(arguments.state_id)
    elif stage == "select-development-target": result = development_target_selection_stage()
    elif stage == "score-heldout": result = heldout_ranker_scores_stage()
    elif stage == "repeat-state": result = repeat_state_stage(arguments.state_id)
    elif stage == "assemble-row-evidence": result = assemble_row_evidence_stage()
    elif stage == "report": result = report_stage()
    elif stage == "physical-smoke": result = physical_backend_smoke_stage()
    else:  # pragma: no cover
        raise ExperimentError(f"unknown stage: {stage}")
    print(canonical_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as exc:
        print(f"{EXPERIMENT_ID}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
