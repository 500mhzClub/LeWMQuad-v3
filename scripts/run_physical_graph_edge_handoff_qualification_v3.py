#!/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
"""Semantic physical graph-edge handoff qualification V3.

V3 delegates the frozen physical panel, ranker, target-selection, reduction,
and publication operations to V2/V1.  Its additional authority is limited to
canonical snapshot semantics, a frozen two-restore behavioural probe, immutable
V1/V2 custody, and the mandatory first-eight semantic/behavioural gate.

Importing this module opens no result root and constructs no simulator/model.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import faulthandler
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import pickletools
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from lewm.safety import physical_graph_edge_handoff_qualification_v3_contract as CONTRACT
from lewm.safety import physical_graph_edge_handoff_qualification_v3_metrics as METRICS
from lewm.safety import physical_graph_edge_handoff_snapshot_semantics_v1 as SEMANTICS
from scripts import run_physical_graph_edge_handoff_qualification_v1 as V1
from scripts import run_physical_graph_edge_handoff_qualification_v2 as V2


class ExperimentError(RuntimeError):
    """A V3 source, custody, semantic, probe, or scientific invariant failed."""


EXPERIMENT_ID = CONTRACT.EXPERIMENT_ID
PARENT_COMMIT = CONTRACT.SOURCE_PARENT_COMMIT
SOURCE_BASELINE_COMMIT = CONTRACT.SOURCE_BASELINE_COMMIT
FREEZE_SUBJECT = CONTRACT.CONTRACT_FREEZE_COMMIT_SUBJECT
RESULT_SUBJECT = CONTRACT.RESULT_COMMIT_SUBJECT
OUTPUT_ROOT = Path(CONTRACT.OUTPUT_ROOT)
MATERIAL_ROOT = Path(CONTRACT.MATERIAL_ROOT)
EXTERNAL_REGENERATION_RECEIPT = Path(CONTRACT.EXTERNAL_REGENERATION_RECEIPT)
HISTORICAL_CUSTODY_RECEIPT = Path(CONTRACT.HISTORICAL_CUSTODY_RECEIPT_PATH)
V1_OFFICIAL_ROOT = Path(CONTRACT.V1_OFFICIAL_ROOT)
V1_MATERIAL_ROOT = Path(CONTRACT.V1_MATERIAL_ROOT)
V2_OFFICIAL_ROOT = Path(CONTRACT.V2_OFFICIAL_ROOT)
V2_MATERIAL_ROOT = Path(CONTRACT.V2_MATERIAL_ROOT)

STATE_COUNT = V1.STATE_COUNT
PROSPECTIVE_POOL_COUNT = V1.PROSPECTIVE_POOL_COUNT
DEVELOPMENT_COUNT = V1.DEVELOPMENT_COUNT
HELDOUT_COUNT = V1.HELDOUT_COUNT
CANDIDATE_COUNT = V1.CANDIDATE_COUNT
PHYSICS_STEPS_PER_BRANCH = V1.PHYSICS_STEPS_PER_BRANCH
TRACE_DT_S = V1.TRACE_DT_S

SCIENTIFIC_LEAVES = (
    *V1.SCIENTIFIC_LEAVES,
    CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["v1_v2_custody_and_nonreuse"],
    CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["scientific_invariance_receipt"],
    CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["v1_v2_v3_first_eight_reproduction"],
    CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["snapshot_equivalence_index"],
    CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["snapshot_behavioural_probes"],
)
PUBLICATION_LEAVES = tuple(V1.PUBLICATION_LEAVES)
ALL_OUTPUT_LEAVES = SCIENTIFIC_LEAVES + PUBLICATION_LEAVES

DOC_PATHS = {
    "contract": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_contract_2026-09-02.json",
    "fixture": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_fixture_2026-09-02.json",
    "output_schema": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_output_schema_2026-09-02.json",
    "preregistration": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_preregistration_2026-09-02.md",
    "source_closure": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_source_closure_2026-09-02.json",
    "scientific_invariance": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_scientific_invariance_2026-09-02.json",
    "historical_custody_binding": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_v1_v2_custody_binding_2026-09-02.json",
}

_ORIGINAL_V1_RUNTIME_CONTRACT = V1._runtime_contract
_ORIGINAL_V2_NORMALISE_SNAPSHOT = V2._normalise_snapshot
_ORIGINAL_V2_EDGE_PORT_ALIGNMENT = V2._edge_port_record_authority_alignment
_ORIGINAL_V2_CANDIDATE_ALIGNMENT = V2._derive_candidate_outcome_authority_alignment
_ORIGINAL_V2_CANONICAL_DIRECTED_PORT = V2._canonical_directed_port
_ORIGINAL_V2_EXPERIMENT_ID = V2.EXPERIMENT_ID
_ORIGINAL_V2_PUBLICATION = V2._write_publication
_ORIGINAL_V2_PROJECT_V1_TO_V2 = V2.METRICS.project_v1_evidence_to_v2


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


def _file_binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    info = _require_regular(path)
    bound_path = str(path if relative_to is None else path.relative_to(relative_to))
    return {"path": bound_path, "bytes": int(info.st_size), "sha256": sha256_file(path)}


def _v3_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        schema = value.get("schema")
        if isinstance(schema, str) and "physical_graph_edge_handoff_qualification_v3" in schema:
            return copy.deepcopy(dict(value))
    # Delegated stage builders are frozen V1 functions and therefore emit V1
    # document identity.  V3's public adapter is intentionally V2->V3 only;
    # compose the two frozen identity-only adapters mechanically.  This does
    # not rewrite lowercase pgehq-v1 scientific IDs or any numeric evidence.
    return METRICS.project_v2_evidence_to_v3(
        _ORIGINAL_V2_PROJECT_V1_TO_V2(value)
    )


def _v3_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_json(path, _v3_identity(value))


def _v3_atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    V1.atomic_bytes(
        path,
        b"".join(canonical_bytes(_v3_identity(dict(row))) for row in rows),
    )


@contextlib.contextmanager
def _patch_module(module: Any, replacements: Mapping[str, Any]) -> Iterable[None]:
    sentinel = object()
    saved = {name: getattr(module, name, sentinel) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in saved.items():
            if value is sentinel:
                delattr(module, name)
            else:
                setattr(module, name, value)


@contextlib.contextmanager
def _v2_storage_namespace() -> Iterable[None]:
    replacements = {
        "CONTRACT": CONTRACT,
        "METRICS": METRICS,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "MATERIAL_ROOT": MATERIAL_ROOT,
        "_v2_identity": _v3_identity,
    }
    with _patch_module(V2, replacements):
        yield


def persisted_array_sha256(value: Any) -> str:
    return V2.persisted_array_sha256(value)


def _write_material_shard_impl(
    directory: Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    with _v2_storage_namespace():
        V2._write_material_shard_impl(
            directory, _v3_identity(metadata), arrays, root=root
        )


def _write_material_shard(
    directory: Path, metadata: Mapping[str, Any], arrays: Mapping[str, Any]
) -> None:
    _write_material_shard_impl(directory, metadata, arrays, root=MATERIAL_ROOT)


def validate_persisted_array_payload(
    directory: Path, metadata: Mapping[str, Any], *, root: Path = MATERIAL_ROOT
) -> dict[str, Any]:
    with _v2_storage_namespace():
        return V2.validate_persisted_array_payload(directory, metadata, root=root)


def _load_material_shard(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with _v2_storage_namespace():
        return V2._load_material_shard(directory)


def _normalise_snapshot(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return _ORIGINAL_V2_NORMALISE_SNAPSHOT(value)


def _edge_port_record_authority_alignment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _v2_storage_namespace():
        return _ORIGINAL_V2_EDGE_PORT_ALIGNMENT(*args, **kwargs)


def _canonical_directed_port_for_inherited_v2(state_id: str) -> list[float]:
    """Read V3 custody, then reuse the exact frozen V2 port lookup.

    Candidate-port formulas remain V2's frozen implementation.  This adapter
    changes only the enclosing document identity presented to that lookup; it
    never rewrites the lower-case V1 scientific state identifier or the port
    values.
    """

    if (
        not isinstance(state_id, str)
        or state_id != state_id.lower()
        or not state_id.startswith("pgehq-v1-state-")
    ):
        raise ExperimentError("canonical directed-port state identity drift")
    document = _ordinary_json(OUTPUT_ROOT / "edge_port_index.json")
    if (
        document.get("schema")
        != "physical_graph_edge_handoff_qualification_v3.edge_port_index.v1"
        or document.get("experiment_id") != EXPERIMENT_ID
        or "content_digest" not in document
    ):
        raise ExperimentError("canonical V3 edge-port index identity drift")
    try:
        CONTRACT.validate_content_digest(document)
    except Exception as exc:
        raise ExperimentError("canonical V3 edge-port index digest drift") from exc
    records = document.get("records")
    if not isinstance(records, list):
        raise ExperimentError("canonical V3 edge-port record inventory is absent")
    matches = [
        row
        for row in records
        if isinstance(row, Mapping) and row.get("state_id") == state_id
    ]
    if len(matches) != 1:
        raise ExperimentError("canonical V3 directed port is absent or ambiguous")
    if matches[0].get("state_id") != state_id:
        raise ExperimentError("canonical V3 directed-port state identity changed")
    port = matches[0].get("directed_port_world")
    if (
        not isinstance(port, list)
        or len(port) != 3
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in port
        )
    ):
        raise ExperimentError("canonical V3 directed-port pose is malformed")

    projected = METRICS.project_v3_evidence_to_v2(document)
    with _patch_module(
        V2,
        {
            "_ordinary_json": lambda _path: copy.deepcopy(projected),
            "EXPERIMENT_ID": _ORIGINAL_V2_EXPERIMENT_ID,
        },
    ):
        result = _ORIGINAL_V2_CANONICAL_DIRECTED_PORT(state_id)
    expected = [float(value) for value in port]
    if result != expected:
        raise ExperimentError("V3-to-V2 canonical directed-port projection drift")
    return result


def _derive_candidate_outcome_authority_alignment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _v2_storage_namespace():
        with _patch_module(
            V2,
            {"_canonical_directed_port": _canonical_directed_port_for_inherited_v2},
        ):
            return _ORIGINAL_V2_CANDIDATE_ALIGNMENT(*args, **kwargs)


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ExperimentError(f"git {' '.join(arguments)} failed: {exc.output}") from exc


def _validate_runtime_source_closure() -> dict[str, Any]:
    path = DOC_PATHS["source_closure"]
    value = _ordinary_json(path)
    if value.get("content_digest") != V1.content_digest(value):
        raise ExperimentError("V3 source closure content digest drift")
    expected = list(CONTRACT.SOURCE_CLOSURE_PATHS)
    rows = value.get("rows")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != expected:
        raise ExperimentError("V3 source closure row order drift")
    for row in rows:
        source = REPO_ROOT / str(row["path"])
        info = _require_regular(source)
        if (
            int(row.get("bytes", -1)) != int(info.st_size)
            or row.get("sha256") != sha256_file(source)
        ):
            raise ExperimentError(f"V3 source closure live-byte drift: {source}")
    return value


def require_runtime_source_freeze() -> str:
    head = _git("rev-parse", "HEAD")
    if _git("show", "-s", "--format=%s", head) != FREEZE_SUBJECT:
        raise ExperimentError("runtime HEAD is not the V3 contract-freeze subject")
    if _git("rev-parse", f"{head}^") != PARENT_COMMIT:
        raise ExperimentError("V3 contract freeze is not a direct child of V2")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ExperimentError("runtime worktree is not clean at V3 source freeze")
    for relative in CONTRACT.TRACKED_SOURCE_PATHS:
        path = REPO_ROOT / relative
        _require_regular(path)
        frozen = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=REPO_ROOT)
        if path.read_bytes() != frozen:
            raise ExperimentError(f"runtime source differs from V3 HEAD: {relative}")
    _validate_runtime_source_closure()
    return head


def _load_custody_evaluator(module: Any | None = None) -> Any:
    if module is not None:
        return module
    try:
        from scripts import evaluate_physical_graph_edge_handoff_qualification_v3 as evaluator
    except ImportError as exc:
        raise ExperimentError("V3 independent evaluator is unavailable") from exc
    return evaluator


def _historical_custody_binding() -> dict[str, Any]:
    return _file_binding(HISTORICAL_CUSTODY_RECEIPT)


def validate_historical_custody_before_creation(
    *, evaluator_module: Any | None = None
) -> dict[str, Any]:
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink() or MATERIAL_ROOT.exists() or MATERIAL_ROOT.is_symlink():
        raise ExperimentError("V3 roots must be absent during historical custody preflight")
    evaluator = _load_custody_evaluator(evaluator_module)
    validator = getattr(evaluator, "validate_existing_historical_custody_receipt", None)
    if not callable(validator):
        raise ExperimentError("historical custody validator API is absent")
    value = validator(HISTORICAL_CUSTODY_RECEIPT)
    if not isinstance(value, Mapping):
        raise ExperimentError("historical custody validator returned no receipt")
    try:
        return METRICS.validate_external_v1_v2_custody_receipt(
            value, expected_binding=_historical_custody_binding()
        )
    except Exception as exc:
        raise ExperimentError("historical custody receipt authority drift") from exc


def _runtime_contract(freeze: str) -> dict[str, Any]:
    binding = _historical_custody_binding()
    runtime = CONTRACT.build_runtime_contract(freeze, binding)
    for row in runtime["external_artifact_bindings"]:
        path = Path(str(row["path"]))
        info = _require_regular(path)
        if int(row["bytes"]) != int(info.st_size) or row["sha256"] != sha256_file(path):
            raise ExperimentError(f"frozen external artifact drift: {row['role']}")
    predecessor = dict(runtime["predecessor_result_binding"])
    path = Path(str(predecessor["path"]))
    info = _require_regular(path)
    if int(predecessor["bytes"]) != int(info.st_size) or predecessor["sha256"] != sha256_file(path):
        raise ExperimentError("bound predecessor result changed")
    CONTRACT.validate_runtime_contract(
        runtime,
        source_freeze_commit=freeze,
        historical_custody_receipt_binding=binding,
    )
    return runtime


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ExperimentError(f"custody root is not an ordinary directory: {root}")
    rows = []
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


def _assert_historical_storage_isolation() -> None:
    historical = []
    for root in (V1_OFFICIAL_ROOT, V1_MATERIAL_ROOT, V2_OFFICIAL_ROOT, V2_MATERIAL_ROOT):
        historical.extend(_tree_inventory(root))
    current = _tree_inventory(OUTPUT_ROOT) + _tree_inventory(MATERIAL_ROOT)
    historical_inodes = {(row["device"], row["inode"]) for row in historical}
    if any((row["device"], row["inode"]) in historical_inodes for row in current):
        raise ExperimentError("V3 contains an inode shared with historical roots")
    historical_shards = {
        row["sha256"]
        for row in historical
        if row["path"].endswith(("/metadata.json", "/payload.npz"))
    }
    copied = [
        row["path"]
        for row in current
        if row["path"].endswith(("/metadata.json", "/payload.npz"))
        and row["sha256"] in historical_shards
    ]
    if copied:
        raise ExperimentError(
            f"V3 contains an exact historical shard-file copy: {copied[0]}"
        )


@contextlib.contextmanager
def _v3_namespace(
    *, writer: Any | None = None
) -> Iterable[None]:
    replacements = {
        "CONTRACT": CONTRACT,
        "METRICS": METRICS,
        # Preserve V1's internal experiment salt/IDs for the unchanged panel.
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
        "_write_material_shard": _write_material_shard if writer is None else writer,
        "_load_material_shard": _load_material_shard,
        "_normalise_snapshot": _normalise_snapshot,
        "_edge_port_record": _edge_port_record_authority_alignment,
        "derive_candidate_outcome": _derive_candidate_outcome_authority_alignment,
        "_write_publication": _write_publication,
        "atomic_json": _v3_atomic_json,
        "atomic_jsonl": _v3_atomic_jsonl,
        "require_runtime_source_freeze": require_runtime_source_freeze,
        "_validate_runtime_source_closure": _validate_runtime_source_closure,
        "_runtime_contract": _runtime_contract,
    }
    with _patch_module(V1, replacements):
        yield


def _delegate(name: str, *args: Any, writer: Any | None = None, **kwargs: Any) -> Any:
    with _v3_namespace(writer=writer):
        try:
            return _v3_identity(getattr(V1, name)(*args, **kwargs))
        except V1.ExperimentError as exc:
            raise ExperimentError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Snapshot semantic identity and the frozen behavioural probe
# ---------------------------------------------------------------------------


def _fresh_snapshot_semantics(payload: bytes) -> dict[str, Any]:
    """Canonicalize a snapshot freshly created by this trusted process.

    Historical bytes never enter this path.  They are reconstructed only by
    the independent evaluator's restricted worker API below.
    """

    if not isinstance(payload, bytes) or not payload:
        raise ExperimentError("fresh snapshot payload is empty")
    try:
        value = pickle.loads(payload)
        canonical = SEMANTICS.canonical_semantic_snapshot(value)
        evidence = SEMANTICS.semantic_snapshot_evidence(value)
        validated = METRICS.validate_snapshot_semantic_evidence(evidence)
        METRICS.validate_snapshot_semantic_bytes(canonical, validated)
    except Exception as exc:
        raise ExperimentError("fresh snapshot semantic reconstruction failed") from exc
    digest = hashlib.sha256(canonical).hexdigest()
    if validated["snapshot_semantic_digest_v1"] != digest:
        raise ExperimentError("fresh snapshot semantic digest drift")
    return {
        "artifact_file_sha256": hashlib.sha256(payload).hexdigest(),
        "artifact_bytes": len(payload),
        "snapshot_semantic_digest_v1": digest,
        "semantic_payload_bytes": canonical,
        "semantic_evidence": copy.deepcopy(validated),
    }


def _historical_snapshot_semantics(
    payload_path: Path, *, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Use only the evaluator's public isolated restricted-worker API."""

    evaluator = _load_custody_evaluator(evaluator_module)
    reconstruct = getattr(
        evaluator, "reconstruct_historical_snapshot_semantics", None
    )
    if not callable(reconstruct):
        raise ExperimentError("public historical semantic reconstruction API is absent")
    try:
        result = reconstruct(payload_path, metrics=METRICS)
    except Exception as exc:
        raise ExperimentError("restricted historical semantic reconstruction failed") from exc
    required = {
        "artifact_file_sha256",
        "artifact_bytes",
        "snapshot_semantic_digest_v1",
        "semantic_payload_bytes",
        "semantic_payload_sha256",
        "semantic_evidence",
    }
    if not isinstance(result, Mapping) or set(result) != required:
        raise ExperimentError("historical semantic reconstruction field drift")
    semantic_bytes = result["semantic_payload_bytes"]
    if (
        not isinstance(semantic_bytes, int)
        or isinstance(semantic_bytes, bool)
        or semantic_bytes <= len(SEMANTICS.SEMANTIC_BINARY_MAGIC)
        or result["semantic_payload_sha256"]
        != result["snapshot_semantic_digest_v1"]
    ):
        raise ExperimentError("historical semantic payload digest drift")
    try:
        evidence = METRICS.validate_snapshot_semantic_evidence(
            result["semantic_evidence"]
        )
    except Exception as exc:
        raise ExperimentError("historical semantic evidence validation failed") from exc
    if (
        evidence["canonical_semantic_byte_count"] != semantic_bytes
        or evidence["snapshot_semantic_digest_v1"]
        != result["snapshot_semantic_digest_v1"]
    ):
        raise ExperimentError("historical semantic evidence identity drift")
    copied = copy.deepcopy(dict(result))
    copied["semantic_evidence"] = evidence
    return copied


def _trace_stuck(trace: Mapping[str, Any]) -> bool:
    import numpy as np

    requested = np.asarray(trace["requested_command"], dtype=np.float64)
    poses = np.asarray(trace["base_pose_world"], dtype=np.float64)
    start_yaw = V1._pose_yaw_xyzw(poses[0])
    end_yaw = V1._pose_yaw_xyzw(poses[-1])
    authority = CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]
    return bool(
        float(np.max(np.abs(requested)))
        > float(authority["command_activity_threshold"])
        and float(np.linalg.norm(poses[-1, :2] - poses[0, :2]))
        < float(authority["h3_translation_threshold_m"])
        and abs(V1._wrap_angle(end_yaw - start_yaw))
        < float(authority["h3_heading_threshold_rad"])
    )


def _normalise_probe_trace(
    trace: Mapping[str, Any], *, final_snapshot_semantic_digest_v1: str
) -> dict[str, Any]:
    import numpy as np

    if set(trace) != set(CONTRACT.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY):
        raise ExperimentError("behavioural probe trace member drift")
    result: dict[str, Any] = {}
    for member, authority in CONTRACT.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY.items():
        array = np.ascontiguousarray(np.asarray(trace[member]))
        expected_shape = [
            CONTRACT.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES,
            *authority["shape"][1:],
        ]
        if array.dtype.str != authority["descr"] or list(array.shape) != expected_shape:
            raise ExperimentError(
                f"behavioural probe {member} dtype/shape drift: "
                f"{array.dtype.str}/{list(array.shape)}"
            )
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            raise ExperimentError(f"behavioural probe {member} is nonfinite")
        result[member] = array
    if (
        not isinstance(final_snapshot_semantic_digest_v1, str)
        or len(final_snapshot_semantic_digest_v1) != 64
    ):
        raise ExperimentError("behavioural final semantic digest drift")
    result["termination_reason"] = "H3_COMPLETE"
    result["stuck"] = _trace_stuck(result)
    result["final_snapshot_semantic_digest_v1"] = (
        final_snapshot_semantic_digest_v1
    )
    # The pure authority is the final shape/dtype/value boundary.
    METRICS.snapshot_behavioural_digest(result)
    return result


def _trace_member_manifests(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    for member in sorted(CONTRACT.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY):
        array = np.ascontiguousarray(np.asarray(trace[member]))
        rows.append(
            {
                "member": member,
                "dtype_str": array.dtype.str,
                "shape": [int(value) for value in array.shape],
                "array_bytes_sha256": hashlib.sha256(
                    array.tobytes(order="C")
                ).hexdigest(),
            }
        )
    return rows


class _V3GenesisPhysicalSession(V1._GenesisPhysicalSession):
    """V1 physical session with probe-only controller/policy sampling."""

    def _sample(
        self,
        requested: Sequence[float],
        applied: Sequence[float],
        timestamp_s: float,
    ) -> dict[str, Any]:
        import numpy as np

        row = super()._sample(requested, applied, timestamp_s)
        observation = np.asarray(
            self._last_controller_observation, dtype=np.float64
        ).reshape(45)
        # Policy output is deliberately the raw action immediately after act,
        # not the scaled/latency-applied target returned by policy.act.
        policy_output = np.asarray(
            self.ctx.policy._last_actions, dtype=np.float64
        ).reshape(1, 12)[0]
        row["controller_observation"] = observation.copy()
        row["policy_output"] = policy_output.copy()
        return row

    def execute_behavioural_probe(self) -> dict[str, Any]:
        members = tuple(CONTRACT.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY)
        commands = [
            list(CONTRACT.BEHAVIOURAL_PROBE_COMMAND)
            for _ in range(CONTRACT.BEHAVIOURAL_PROBE_COMMAND_TICKS)
        ]
        with _patch_module(V1, {"CANDIDATE_TRACE_MEMBERS": members}):
            trace = self.execute_requested_ticks(commands)
        if trace is None:
            raise ExperimentError("behavioural probe returned no trace")
        return dict(trace)


def _execute_probe_trials(
    session_factory: Callable[[], Any], snapshot_payload: bytes
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for trial_index in range(2):
        # A restoration trial is defined by a newly constructed simulator
        # instance.  A second restore into the same session is deliberately
        # not accepted as repeatability authority.
        session = session_factory()
        session.restore_snapshot(snapshot_payload)
        raw_trace = session.execute_behavioural_probe()
        final_payload, _snapshot, _auxiliary = session.capture_snapshot()
        final_semantics = _fresh_snapshot_semantics(final_payload)
        trace = _normalise_probe_trace(
            raw_trace,
            final_snapshot_semantic_digest_v1=final_semantics[
                "snapshot_semantic_digest_v1"
            ],
        )
        trials.append(
            {
                "trial_index": trial_index,
                "trace": trace,
                "snapshot_behavioural_digest_v1": (
                    METRICS.snapshot_behavioural_digest(trace)
                ),
                "trace_member_manifests": _trace_member_manifests(trace),
                "final_snapshot_semantic_evidence": final_semantics[
                    "semantic_evidence"
                ],
            }
        )
    return trials


def _probe_version_evidence(
    identity: Mapping[str, Any], trials: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(trials) != 2:
        raise ExperimentError("behavioural probe trial cardinality drift")
    comparison = METRICS.compare_behavioural_probe_traces(
        trials[0]["trace"], trials[1]["trace"]
    )
    return {
        "snapshot_identity": copy.deepcopy(dict(identity)),
        "trial_1_behavioural_digest_v1": str(
            trials[1]["snapshot_behavioural_digest_v1"]
        ),
        "final_snapshot_semantic_digests": [
            str(trial["trace"]["final_snapshot_semantic_digest_v1"])
            for trial in trials
        ],
        "trial_stuck": [bool(trial["trace"]["stuck"]) for trial in trials],
        "trial_termination_reasons": [
            str(trial["trace"]["termination_reason"]) for trial in trials
        ],
        "trial_pair_comparison": comparison,
        "trace_member_manifests": [
            copy.deepcopy(trial["trace_member_manifests"]) for trial in trials
        ],
    }


class GenesisGo2PhysicalBackend(V1.GenesisGo2PhysicalBackend):
    """Real V3 qualification backend; all inherited science stays delegated."""

    def __init__(self, *, backend: str = "cpu") -> None:
        super().__init__(backend=backend)
        self.last_v3_augmentation: dict[str, Any] | None = None
        self._pool_index_by_id = {
            str(row["candidate_spec_id"]): index
            for index, row in enumerate(CONTRACT.build_prospective_pool_specs())
        }

    def _session(self, spec: Mapping[str, Any]) -> _V3GenesisPhysicalSession:
        return _V3GenesisPhysicalSession(spec, backend=self.backend)

    def qualify(self, candidate_spec: Mapping[str, Any]) -> Mapping[str, Any]:
        spec = copy.deepcopy(dict(candidate_spec))
        if self._spec_sha_by_id.get(str(spec["candidate_spec_id"])) != str(
            spec["canonical_spec_sha256"]
        ):
            raise ExperimentError("physical candidate spec is outside frozen authority")
        session = self._session(spec)
        session.begin_and_settle()
        payload, snapshot, auxiliary = session.capture_snapshot()
        semantics = _fresh_snapshot_semantics(payload)
        # The frozen probe must run in a fresh simulator instance.  The
        # teacher-generation session is never reused as probe authority.
        trials = _execute_probe_trials(lambda: self._session(spec), payload)
        identity = {
            "artifact_file_sha256": semantics["artifact_file_sha256"],
            "snapshot_semantic_digest_v1": semantics[
                "snapshot_semantic_digest_v1"
            ],
            "snapshot_behavioural_digest_v1": trials[0][
                "snapshot_behavioural_digest_v1"
            ],
        }
        self.last_v3_augmentation = {
            "snapshot_semantic_evidence": semantics["semantic_evidence"],
            "snapshot_semantic_bytes": semantics["semantic_payload_bytes"],
            "snapshot_identity": identity,
            "behavioural_probe_trials": {"V3": trials},
            "semantic_evidence_by_version": {
                "V3": semantics["semantic_evidence"]
            },
        }
        # Restore once more so the teacher starts at the exact frozen state;
        # probe outcomes cannot leak into qualification.
        session.restore_snapshot(payload)
        teacher = session.execute_teacher()
        graph = session.graph()
        graph["teacher_positive_route_progress"] = V1._teacher_route_progress_m(
            teacher["base_pose_world"],
            spec["geometry"]["selected_directed_edge"]["opening_segment_world"],
        ) > 0.0
        graph["teacher_competing_port_entered"] = V1._first_competing_crossing(
            teacher["base_pose_world"],
            spec["geometry"]["competing_directed_edges"],
        ) is not None
        snapshot_sha = hashlib.sha256(payload).hexdigest()
        result = {
            "candidate_spec_id": spec["candidate_spec_id"],
            "initial_decision_state_sha256": snapshot_sha,
            "graph": graph,
            "teacher_trace": teacher,
            "rgb": auxiliary["rgb"],
            "snapshot": snapshot,
            "contact_instrumentation": {
                "api": "robot.get_contacts",
                "sample_period_s": TRACE_DT_S,
                "forbidden_net_force_api_used": False,
                "ontology_sha256": CONTRACT.CONTACT_AUTHORITY["ontology_sha256"],
            },
            "runtime_evidence": {
                **session._runtime,
                "snapshot_captured_before_teacher": True,
                "teacher_restored_from_serialized_snapshot": True,
                "teacher_snapshot_sha256": snapshot_sha,
            },
        }
        pool_index = self._pool_index_by_id.get(str(spec["candidate_spec_id"]))
        if pool_index is None:
            raise ExperimentError("physical candidate has no prospective pool identity")
        if pool_index < 8:
            self.add_historical_probe_versions(spec, pool_index)
        return result

    def add_historical_probe_versions(
        self,
        candidate_spec: Mapping[str, Any],
        pool_index: int,
        *,
        evaluator_module: Any | None = None,
    ) -> None:
        """Restore V1 and V2 pool snapshots read-only in fresh sessions.

        These traces answer only the technical first-eight reproduction gate;
        they are never used as teacher, candidate, panel, or ranker evidence.
        """

        if self.last_v3_augmentation is None:
            raise ExperimentError("V3 snapshot/probe augmentation is absent")
        if not 0 <= int(pool_index) < 8:
            raise ExperimentError("historical probe is restricted to pool indices 0..7")
        import numpy as np

        spec = copy.deepcopy(dict(candidate_spec))
        for version, root in (("V1", V1_MATERIAL_ROOT), ("V2", V2_MATERIAL_ROOT)):
            payload_path = root / "qualification" / f"pool-{pool_index:03d}" / "payload.npz"
            semantic = _historical_snapshot_semantics(
                payload_path, evaluator_module=evaluator_module
            )
            with np.load(payload_path, allow_pickle=False) as archive:
                if "snapshot_payload_bytes" not in archive.files:
                    raise ExperimentError(
                        f"historical {version} snapshot payload member is absent"
                    )
                member = np.ascontiguousarray(archive["snapshot_payload_bytes"])
            if member.dtype.str != "|u1" or member.ndim != 1:
                raise ExperimentError(
                    f"historical {version} snapshot payload extent drift"
                )
            payload = member.tobytes(order="C")
            if hashlib.sha256(payload).hexdigest() != semantic["artifact_file_sha256"]:
                raise ExperimentError(
                    f"historical {version} artifact binding drift"
                )
            # Each restoration trial receives its own newly constructed
            # session. No V1/V2 session, shard state, or outcome is reused.
            trials = _execute_probe_trials(lambda: self._session(spec), payload)
            identity = {
                "artifact_file_sha256": semantic["artifact_file_sha256"],
                "snapshot_semantic_digest_v1": semantic[
                    "snapshot_semantic_digest_v1"
                ],
                "snapshot_behavioural_digest_v1": trials[0][
                    "snapshot_behavioural_digest_v1"
                ],
            }
            self.last_v3_augmentation["behavioural_probe_trials"][version] = trials
            self.last_v3_augmentation["semantic_evidence_by_version"][version] = (
                semantic["semantic_evidence"]
            )
            self.last_v3_augmentation.setdefault("snapshot_identities", {})[
                version
            ] = identity


def _qualification_backend_default() -> Any:
    return GenesisGo2PhysicalBackend()


def _augmentation_arrays_and_metadata(
    augmentation: Mapping[str, Any],
    *,
    pool_index: int,
    snapshot_payload_bytes: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = augmentation.get("snapshot_semantic_bytes")
    evidence = augmentation.get("snapshot_semantic_evidence")
    identity = augmentation.get("snapshot_identity")
    trials_by_version = augmentation.get("behavioural_probe_trials")
    evidence_by_version = augmentation.get("semantic_evidence_by_version")
    if (
        not isinstance(canonical, bytes)
        or not isinstance(evidence, Mapping)
        or not isinstance(identity, Mapping)
        or not isinstance(trials_by_version, Mapping)
        or not isinstance(evidence_by_version, Mapping)
    ):
        raise ExperimentError("qualification semantic/probe augmentation is incomplete")
    expected_versions = {"V1", "V2", "V3"} if pool_index < 8 else {"V3"}
    if set(trials_by_version) != expected_versions or set(evidence_by_version) != expected_versions:
        raise ExperimentError("qualification probe version inventory drift")
    identities = {"V3": copy.deepcopy(dict(identity))}
    supplied_identities = augmentation.get("snapshot_identities", {})
    if pool_index < 8:
        if not isinstance(supplied_identities, Mapping) or set(supplied_identities) != {
            "V1",
            "V2",
        }:
            raise ExperimentError("historical snapshot identity inventory drift")
        identities.update(
            {
                version: copy.deepcopy(dict(supplied_identities[version]))
                for version in ("V1", "V2")
            }
        )
    traces = {
        version: [trial["trace"] for trial in trials_by_version[version]]
        for version in expected_versions
    }
    try:
        built = METRICS.build_qualification_shard_augmentation(
            pool_index=pool_index,
            snapshot_payload_bytes=snapshot_payload_bytes,
            canonical_semantic_bytes=canonical,
            snapshot_semantic_evidence=evidence,
            version_snapshot_identities=identities,
            behavioural_probe_traces=traces,
        )
    except Exception as exc:
        raise ExperimentError("pure qualification augmentation builder failed") from exc
    if not isinstance(built, Mapping) or set(built) != {"metadata", "arrays"}:
        raise ExperimentError("qualification augmentation builder field drift")
    return dict(built["arrays"]), dict(built["metadata"])


def _qualification_writer(
    backend: Any,
    pool_index: int,
) -> Any:
    def writer(
        directory: Path, metadata: Mapping[str, Any], arrays: Mapping[str, Any]
    ) -> None:
        augmentation = getattr(backend, "last_v3_augmentation", None)
        if not isinstance(augmentation, Mapping):
            raise ExperimentError(
                "V3 qualification backend did not provide semantic/probe evidence"
            )
        added_arrays, added_metadata = _augmentation_arrays_and_metadata(
            augmentation,
            pool_index=pool_index,
            snapshot_payload_bytes=arrays.get("snapshot_payload_bytes"),
        )
        overlap = set(arrays) & set(added_arrays)
        if overlap:
            raise ExperimentError(f"qualification augmentation member collision: {overlap}")
        combined_metadata = {**dict(metadata), **added_metadata}
        combined_arrays = {**dict(arrays), **added_arrays}
        _write_material_shard(directory, combined_metadata, combined_arrays)

    return writer


# ---------------------------------------------------------------------------
# Runtime receipts and technical first-eight gate
# ---------------------------------------------------------------------------


def _snapshot_regression_payload() -> bytes:
    from scripts.run_go2_oracle_branch_pilot_v1 import BranchSnapshot
    import numpy as np

    value = BranchSnapshot(
        solver_state={},
        step_index=0,
        last_actions=np.zeros((1, 12), dtype=np.float32),
        harness={},
        rng={},
        counters={},
        goal={},
        identity={},
        boundary={},
        digest="semantic-regression",
    )
    return pickle.dumps(value, protocol=4)


def _semantic_regression_gate(
    historical_custody: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Execute bounded pure regressions before any simulator is constructed."""

    import numpy as np

    passed: dict[str, bool] = {}
    ids = list(CONTRACT.SEMANTIC_SERIALIZER_REGRESSION_IDS)
    try:
        passed[ids[0]] = bool(
            SEMANTICS.semantic_snapshot_sha256({True: "x"})
            != SEMANTICS.semantic_snapshot_sha256({1: "x"})
            and SEMANTICS.canonical_semantic_snapshot({"b": 2, "a": 1})
            == SEMANTICS.canonical_semantic_snapshot({"a": 1, "b": 2})
        )
        passed[ids[1]] = bool(
            SEMANTICS.semantic_snapshot_sha256([1, 2])
            != SEMANTICS.semantic_snapshot_sha256((1, 2))
            and SEMANTICS.semantic_snapshot_sha256([1, 2])
            != SEMANTICS.semantic_snapshot_sha256([2, 1])
            and SEMANTICS.semantic_snapshot_sha256("café")
            != SEMANTICS.semantic_snapshot_sha256(b"caf\xc3\xa9")
        )
        passed[ids[2]] = bool(
            SEMANTICS.canonical_semantic_snapshot({"b", "a"})
            == SEMANTICS.canonical_semantic_snapshot({"a", "b"})
            and SEMANTICS.semantic_snapshot_sha256({"a"})
            != SEMANTICS.semantic_snapshot_sha256(frozenset({"a"}))
        )
        base = np.arange(12, dtype=np.float64).reshape(3, 4)
        view = base[:, ::2]
        mutation = np.ascontiguousarray(view).copy()
        mutation[0, 0] += 1.0
        passed[ids[3]] = bool(
            SEMANTICS.semantic_snapshot_sha256(view)
            != SEMANTICS.semantic_snapshot_sha256(np.ascontiguousarray(view))
            and SEMANTICS.semantic_snapshot_sha256(np.ascontiguousarray(view))
            != SEMANTICS.semantic_snapshot_sha256(mutation)
        )
        child: list[Any] = ["x"]
        alias = [child, child]
        alias_evidence = SEMANTICS.semantic_snapshot_evidence(alias)
        passed[ids[4]] = bool(
            alias_evidence["reference_alias_edge_count"] == 1
            and SEMANTICS.semantic_snapshot_sha256(alias)
            == SEMANTICS.semantic_snapshot_sha256(
                pickle.loads(pickle.dumps(alias, protocol=4))
            )
        )
        cycle: list[Any] = []
        cycle.append(cycle)
        cycle_evidence = SEMANTICS.semantic_snapshot_evidence(cycle)
        passed[ids[5]] = bool(
            cycle_evidence["reference_alias_edge_count"] == 1
            and cycle_evidence["reference_cycle_edge_count"] == 1
        )
        storage = np.arange(16, dtype=np.int64)
        storage_evidence = SEMANTICS.semantic_snapshot_evidence(
            [storage, storage[2:10:2]]
        )
        passed[ids[6]] = any(
            len(row["member_object_ids"]) == 2
            for row in storage_evidence["storage_manifest"]
        )
        import torch

        tensor = torch.arange(12, dtype=torch.float32)
        tensor_evidence = SEMANTICS.semantic_snapshot_evidence(
            [tensor, tensor[1:9:2]]
        )
        passed[ids[7]] = bool(
            any(
                len(row["member_object_ids"]) == 2
                for row in tensor_evidence["storage_manifest"]
            )
            and all(
                row["semantic_device_class"] == "cpu"
                for row in tensor_evidence["tensor_device_manifest"]
            )
        )
        snapshot = pickle.loads(_snapshot_regression_payload())
        structured = SEMANTICS.semantic_snapshot_evidence(snapshot)
        passed[ids[8]] = bool(
            structured["structured_type_inventory"]
            and structured["structured_type_inventory"][0]["declared_fields"]
            == list(
                CONTRACT.STRUCTURED_TYPE_FIELD_AUTHORITY[
                    "scripts.run_go2_oracle_branch_pilot_v1.BranchSnapshot"
                ]
            )
        )
        sentinels = np.zeros((18, 2), dtype=np.float32)
        sentinels[:3] = math.inf
        sentinels[3:6] = -math.inf
        snapshot.solver_state = {
            "Scene._sim._coupler.rigid_solver.dofs_info.force_range": sentinels
        }
        passed[ids[9]] = bool(
            len(
                SEMANTICS.semantic_snapshot_evidence(snapshot)[
                    "nonfinite_sentinel_inventory"
                ]
            )
            == 1
        )

        def rejected(value: Any) -> bool:
            try:
                SEMANTICS.canonical_semantic_snapshot(value)
            except SEMANTICS.SemanticSnapshotError:
                return True
            return False

        passed[ids[10]] = bool(
            rejected(float("inf"))
            and rejected(np.asarray([math.inf], dtype=np.float32))
        )

        class Unknown:
            pass

        passed[ids[11]] = bool(
            rejected(Unknown())
            and rejected(np.asarray([Unknown()], dtype=object))
        )
        # The independently rebuilt immutable historical receipt is the
        # production two-process and all-eight authority.
        pairs = historical_custody.get("first_eight_pairs")
        pair_rows = list(pairs) if isinstance(pairs, list) else []
        historical_all = bool(
            len(pair_rows) == 8
            and all(
                row.get("snapshot_semantic_digest_v1_equal") is True
                and row.get("semantic_evidence_equal") is True
                and row.get("pass") is True
                for row in pair_rows
            )
        )
        passed[ids[12]] = bool(
            historical_all
            and all(
                row.get("artifact_file_sha256_equal") is False
                and row.get("snapshot_semantic_digest_v1_equal") is True
                for row in pair_rows
            )
        )
        passed[ids[13]] = historical_all
    except Exception as exc:
        raise ExperimentError("pre-simulator semantic regression gate failed") from exc
    rows = METRICS.build_semantic_regression_results(passed)
    if len(rows) != len(ids) or not all(row["passed"] for row in rows):
        raise ExperimentError("pre-simulator semantic regression requirement failed")
    return rows


def build_scientific_invariance_receipt(
    historical_custody: Mapping[str, Any]
) -> dict[str, Any]:
    regressions = _semantic_regression_gate(historical_custody)
    try:
        return METRICS.validate_scientific_invariance_receipt(
            METRICS.build_scientific_invariance_receipt(
                CONTRACT.build_contract(), regressions
            )
        )
    except Exception as exc:
        raise ExperimentError("scientific invariance receipt construction failed") from exc


def _custody_nonreuse_receipt(
    source_freeze: str, external: Mapping[str, Any]
) -> dict[str, Any]:
    projection = METRICS.historical_custody_projection(external)
    projection_sha = hashlib.sha256(
        CONTRACT.canonical_json_bytes(projection)[:-1]
    ).hexdigest()
    value = METRICS.build_v1_v2_custody_and_nonreuse(
        source_freeze_commit=source_freeze,
        external_binding=_historical_custody_binding(),
        external_projection_sha256=projection_sha,
    )
    return METRICS.validate_v1_v2_custody_and_nonreuse(
        value, source_freeze_commit=source_freeze
    )


def initialize_stage(
    *, fake_runtime: bool = False, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Bind immutable history and semantic regressions before V3 creation."""

    external = validate_historical_custody_before_creation(
        evaluator_module=evaluator_module
    )
    invariance = build_scientific_invariance_receipt(external)
    source_freeze = require_runtime_source_freeze()
    result = _delegate("initialize_stage", fake_runtime=fake_runtime)
    reproduction = MATERIAL_ROOT / "reproduction"
    reproduction.mkdir(parents=False, exist_ok=False)
    # Rebuild custody after V3 creation and require exact canonical identity.
    evaluator = _load_custody_evaluator(evaluator_module)
    after = evaluator.validate_existing_historical_custody_receipt(
        HISTORICAL_CUSTODY_RECEIPT
    )
    if canonical_bytes(after) != canonical_bytes(external):
        raise ExperimentError("historical custody changed during V3 initialization")
    _assert_historical_storage_isolation()
    custody = _custody_nonreuse_receipt(source_freeze, external)
    _atomic_json(
        OUTPUT_ROOT
        / CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["v1_v2_custody_and_nonreuse"],
        custody,
    )
    _atomic_json(
        OUTPUT_ROOT / CONTRACT.NEW_RUNTIME_OUTPUT_PATHS["scientific_invariance_receipt"],
        invariance,
    )
    return result


def _reproduction_path() -> Path:
    return OUTPUT_ROOT / CONTRACT.NEW_RUNTIME_OUTPUT_PATHS[
        "v1_v2_v3_first_eight_reproduction"
    ]


def _require_reproduction_pass() -> dict[str, Any]:
    value = _ordinary_json(_reproduction_path())
    try:
        validated = METRICS.validate_first_eight_reproduction(value)
    except Exception as exc:
        raise ExperimentError("first-eight reproduction receipt drift") from exc
    if not METRICS.authorizes_full_v3_collection(validated):
        raise ExperimentError(CONTRACT.REPRODUCTION_MISMATCH_DISPOSITION)
    probe = MATERIAL_ROOT / str(
        validated["first_eight_behavioural_probe_material_binding"]["path"]
    )
    if _file_binding(probe, relative_to=MATERIAL_ROOT) != validated[
        "first_eight_behavioural_probe_material_binding"
    ]:
        raise ExperimentError("first-eight immutable probe material changed")
    return validated


def qualify_pool_state_stage(
    pool_index: int,
    *,
    backend: Any | None = None,
    fake_runtime: bool = False,
) -> dict[str, Any]:
    index = int(pool_index)
    if not 0 <= index < PROSPECTIVE_POOL_COUNT:
        raise ExperimentError("pool index outside frozen V3 population")
    if index >= 8:
        _require_reproduction_pass()
    collector = _qualification_backend_default() if backend is None else backend
    result = _delegate(
        "qualify_pool_state_stage",
        index,
        backend=collector,
        fake_runtime=fake_runtime,
        writer=_qualification_writer(collector, index),
    )
    metadata, arrays = _load_material_shard(
        MATERIAL_ROOT / "qualification" / f"pool-{index:03d}"
    )
    if bool(metadata.get("qualified")) != bool(result.get("qualified")):
        raise ExperimentError("post-reopen V3 qualification disposition drift")
    # Save/reopen validation must include every semantic/probe member before
    # eligibility is exposed to the caller.
    if "snapshot_semantic_bytes" not in arrays:
        raise ExperimentError("post-reopen V3 semantic bytes are absent")
    try:
        METRICS.validate_qualification_shard_augmentation(
            {
                field: metadata[field]
                for field in CONTRACT.QUALIFICATION_SHARD_AUGMENTATION_FIELDS
            },
            pool_index=index,
            reopened_arrays=arrays,
        )
    except Exception as exc:
        raise ExperimentError(
            "post-reopen V3 semantic/probe augmentation validation failed"
        ) from exc
    return metadata


def _probe_trials_from_shard(
    metadata: Mapping[str, Any], arrays: Mapping[str, Any], version: str
) -> list[dict[str, Any]]:
    import numpy as np

    probes = metadata.get("behavioural_probes")
    if not isinstance(probes, Mapping) or version not in probes:
        raise ExperimentError(f"qualification {version} probe metadata is absent")
    evidence = probes[version]
    if not isinstance(evidence, Mapping):
        raise ExperimentError(f"qualification {version} probe evidence drift")
    result = []
    for trial_index in (0, 1):
        trace: dict[str, Any] = {}
        for member in CONTRACT.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY:
            name = f"probe__{version}__{trial_index}__{member}"
            if name not in arrays:
                raise ExperimentError(f"qualification probe member absent: {name}")
            trace[member] = arrays[name]
        digest_name = (
            f"probe__{version}__{trial_index}__"
            "final_snapshot_semantic_digest_bytes"
        )
        if digest_name not in arrays:
            raise ExperimentError(
                f"qualification final semantic member absent: {digest_name}"
            )
        raw_digest = np.asarray(arrays[digest_name])
        if raw_digest.dtype.str != "|u1" or list(raw_digest.shape) != [32]:
            raise ExperimentError("qualification final semantic digest extent drift")
        normalized = _normalise_probe_trace(
            trace,
            final_snapshot_semantic_digest_v1=raw_digest.tobytes(order="C").hex(),
        )
        result.append(
            {
                "trial_index": trial_index,
                "trace": normalized,
                "snapshot_behavioural_digest_v1": (
                    METRICS.snapshot_behavioural_digest(normalized)
                ),
                "trace_member_manifests": _trace_member_manifests(normalized),
            }
        )
    identity = evidence.get("snapshot_identity")
    if not isinstance(identity, Mapping) or set(identity) != set(
        CONTRACT.SNAPSHOT_IDENTITY_FIELDS
    ):
        raise ExperimentError("qualification probe snapshot identity drift")
    if (
        identity["snapshot_behavioural_digest_v1"]
        != result[0]["snapshot_behavioural_digest_v1"]
        or evidence.get("trial_1_behavioural_digest_v1")
        != result[1]["snapshot_behavioural_digest_v1"]
    ):
        raise ExperimentError("qualification probe behavioural digest drift")
    return result


def _probe_npz_arrays(
    trace_rows: Sequence[tuple[int, str, int, Mapping[str, Any]]],
    *,
    first_eight_only: bool,
) -> dict[str, Any]:
    import numpy as np

    expected_count = 48 if first_eight_only else 544
    if len(trace_rows) != expected_count:
        raise ExperimentError("behavioural probe row cardinality drift")
    arrays: dict[str, Any] = {
        "trace_offsets": np.arange(
            0,
            expected_count * CONTRACT.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES + 1,
            CONTRACT.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES,
            dtype=np.int64,
        ),
        "version_code": np.asarray(
            [
                CONTRACT.BEHAVIOURAL_PROBE_AUTHORITY["version_codes"][version]
                for _pool, version, _trial, _trace in trace_rows
            ],
            dtype=np.int64,
        ),
        "pool_index": np.asarray(
            [pool for pool, _version, _trial, _trace in trace_rows],
            dtype=np.int64,
        ),
        "trial_index": np.asarray(
            [trial for _pool, _version, trial, _trace in trace_rows],
            dtype=np.int64,
        ),
        "stuck": np.asarray(
            [trace["stuck"] for _pool, _version, _trial, trace in trace_rows],
            dtype=np.uint8,
        ),
        "termination_code": np.ones(expected_count, dtype=np.int64),
        "final_snapshot_semantic_digest_bytes": np.stack(
            [
                np.frombuffer(
                    bytes.fromhex(trace["final_snapshot_semantic_digest_v1"]),
                    dtype=np.uint8,
                )
                for _pool, _version, _trial, trace in trace_rows
            ],
            axis=0,
        ),
    }
    for member in CONTRACT.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY:
        arrays[member] = np.ascontiguousarray(
            np.concatenate(
                [
                    np.asarray(trace[member])
                    for _pool, _version, _trial, trace in trace_rows
                ],
                axis=0,
            )
        )
    try:
        return (
            METRICS.validate_first_eight_behavioural_probe_npz_arrays(arrays)
            if first_eight_only
            else METRICS.validate_behavioural_probe_npz_arrays(arrays)
        )
    except Exception as exc:
        raise ExperimentError("behavioural probe NPZ authority validation failed") from exc


def _qualification_probe_rows(
    *, first_eight_only: bool
) -> tuple[
    list[tuple[int, str, int, Mapping[str, Any]]],
    list[tuple[dict[str, Any], dict[str, Any]]],
]:
    stop = 8 if first_eight_only else PROSPECTIVE_POOL_COUNT
    trace_rows: list[tuple[int, str, int, Mapping[str, Any]]] = []
    shards: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pool_index in range(stop):
        metadata, arrays = _load_material_shard(
            MATERIAL_ROOT / "qualification" / f"pool-{pool_index:03d}"
        )
        try:
            METRICS.validate_qualification_shard_augmentation(
                {
                    field: metadata[field]
                    for field in CONTRACT.QUALIFICATION_SHARD_AUGMENTATION_FIELDS
                },
                pool_index=pool_index,
                reopened_arrays=arrays,
            )
        except Exception as exc:
            raise ExperimentError(
                f"qualification semantic/probe evidence drift: {pool_index}"
            ) from exc
        shards.append((metadata, arrays))
        versions = CONTRACT.BEHAVIOURAL_PROBE_VERSION_ORDER if pool_index < 8 else ("V3",)
        for version in versions:
            trials = _probe_trials_from_shard(metadata, arrays, version)
            for trial_index, trial in enumerate(trials):
                trace_rows.append(
                    (pool_index, version, trial_index, trial["trace"])
                )
    return trace_rows, shards


def _external_historical_pair(
    historical_custody: Mapping[str, Any], pool_index: int
) -> dict[str, Any]:
    pairs = historical_custody.get("first_eight_pairs")
    if not isinstance(pairs, list) or len(pairs) != 8:
        raise ExperimentError("historical semantic pair inventory drift")
    row = pairs[pool_index]
    if not isinstance(row, Mapping) or int(row.get("pool_index", -1)) != pool_index:
        raise ExperimentError("historical semantic pair order drift")
    return copy.deepcopy(dict(row))


def _all_cross_probe_equal(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> bool:
    return bool(
        len(left) == 2
        and len(right) == 2
        and all(
            METRICS.compare_behavioural_probe_traces(
                left[trial]["trace"], right[trial]["trace"]
            )["pass"]
            for trial in (0, 1)
        )
    )


def _probe_comparison_failure_projection(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose the frozen comparison result without changing its decision rule."""

    row = copy.deepcopy(dict(comparison))
    exact = row.get("exact_member_equal")
    samplewise = row.get("samplewise_pass")
    if not isinstance(exact, Mapping) or not isinstance(samplewise, Mapping):
        raise ExperimentError("behavioural comparison diagnostic field drift")
    row["failing_exact_members"] = sorted(
        str(member) for member, passed in exact.items() if passed is not True
    )
    row["failing_samplewise_members"] = sorted(
        str(member) for member, passed in samplewise.items() if passed is not True
    )
    return row


def _production_fixture_diagnostics(
    semantics: Mapping[str, Mapping[str, Any]],
    probes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build immutable, already-computed fixture failure observability evidence."""

    versions = ("V1", "V2", "V3")
    if set(semantics) != set(versions) or set(probes) != set(versions):
        raise ExperimentError("production fixture diagnostic version drift")
    semantic_equal = len(
        {
            str(semantics[version]["snapshot_semantic_digest_v1"])
            for version in versions
        }
    ) == 1
    within_comparisons: dict[str, dict[str, Any]] = {}
    for version in versions:
        if len(probes[version]) != 2:
            raise ExperimentError("production fixture diagnostic trial drift")
        within_comparisons[version] = _probe_comparison_failure_projection(
            METRICS.compare_behavioural_probe_traces(
                probes[version][0]["trace"], probes[version][1]["trace"]
            )
        )
    pair_order = (("V1", "V2"), ("V1", "V3"), ("V2", "V3"))
    cross_comparisons: dict[str, list[dict[str, Any]]] = {}
    for left, right in pair_order:
        key = f"{left}_{right}"
        cross_comparisons[key] = [
            _probe_comparison_failure_projection(
                METRICS.compare_behavioural_probe_traces(
                    probes[left][trial]["trace"], probes[right][trial]["trace"]
                )
            )
            for trial in (0, 1)
        ]
    within = {
        version: bool(within_comparisons[version]["pass"])
        for version in versions
    }
    cross = {
        key: bool(all(comparison["pass"] for comparison in comparisons))
        for key, comparisons in cross_comparisons.items()
    }
    return {
        "semantic_equal": bool(semantic_equal),
        "within_version_probe_equal": within,
        "cross_version_probe_equal": cross,
        "within_version_comparisons": within_comparisons,
        "cross_version_comparisons": cross_comparisons,
    }


def _production_fixture_failure_message(
    diagnostics: Mapping[str, Any],
) -> str:
    encoded = canonical_bytes(dict(diagnostics)).decode("utf-8").rstrip("\n")
    return f"production snapshot semantic/probe fixture failed: {encoded}"


def _equivalence_record(
    pool_index: int,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    historical_custody: Mapping[str, Any] | None,
) -> dict[str, Any]:
    spec = CONTRACT.build_prospective_pool_specs()[pool_index]
    historical = pool_index < 8
    expected_versions = ("V1", "V2", "V3") if historical else ("V3",)
    probes = metadata.get("behavioural_probes")
    if not isinstance(probes, Mapping) or set(probes) != set(expected_versions):
        raise ExperimentError("equivalence probe version inventory drift")
    versions = {
        version: copy.deepcopy(dict(probes[version]["snapshot_identity"]))
        for version in expected_versions
    }
    trials = {
        version: _probe_trials_from_shard(metadata, arrays, version)
        for version in expected_versions
    }
    v3_evidence = METRICS.validate_snapshot_semantic_evidence(
        metadata["snapshot_semantic_evidence"]
    )
    semantic_manifests_equal = True
    if historical:
        if historical_custody is None:
            raise ExperimentError("historical custody is absent from first-eight gate")
        pair = _external_historical_pair(historical_custody, pool_index)
        for version in ("V1", "V2"):
            historical_version = pair.get(version.lower())
            if not isinstance(historical_version, Mapping):
                raise ExperimentError("historical semantic version evidence drift")
            if (
                historical_version.get("artifact_file_sha256")
                != versions[version]["artifact_file_sha256"]
                or historical_version.get("snapshot_semantic_digest_v1")
                != versions[version]["snapshot_semantic_digest_v1"]
            ):
                raise ExperimentError("historical semantic identity cross-link drift")
            semantic_manifests_equal = bool(
                semantic_manifests_equal
                and historical_version.get("semantic_evidence") == v3_evidence
            )
    trial_pair = {
        version: bool(
            METRICS.compare_behavioural_probe_traces(
                trials[version][0]["trace"], trials[version][1]["trace"]
            )["pass"]
        )
        for version in expected_versions
    }
    trial_one = {
        version: trials[version][1]["snapshot_behavioural_digest_v1"]
        for version in expected_versions
    }
    finals = {
        version: [
            item["trace"]["final_snapshot_semantic_digest_v1"]
            for item in trials[version]
        ]
        for version in expected_versions
    }
    pair_fields: dict[str, Any]
    if historical:
        pair_fields = {
            "v1_v2_artifact_file_sha256_equal": versions["V1"]["artifact_file_sha256"] == versions["V2"]["artifact_file_sha256"],
            "v1_v3_artifact_file_sha256_equal": versions["V1"]["artifact_file_sha256"] == versions["V3"]["artifact_file_sha256"],
            "v2_v3_artifact_file_sha256_equal": versions["V2"]["artifact_file_sha256"] == versions["V3"]["artifact_file_sha256"],
            "v1_v2_semantic_equal": versions["V1"]["snapshot_semantic_digest_v1"] == versions["V2"]["snapshot_semantic_digest_v1"],
            "v1_v3_semantic_equal": versions["V1"]["snapshot_semantic_digest_v1"] == versions["V3"]["snapshot_semantic_digest_v1"],
            "v2_v3_semantic_equal": versions["V2"]["snapshot_semantic_digest_v1"] == versions["V3"]["snapshot_semantic_digest_v1"],
            "v1_v2_behavioural_equal": _all_cross_probe_equal(trials["V1"], trials["V2"]),
            "v1_v3_behavioural_equal": _all_cross_probe_equal(trials["V1"], trials["V3"]),
            "v2_v3_behavioural_equal": _all_cross_probe_equal(trials["V2"], trials["V3"]),
            "v1_restore_trials_equal": trial_pair["V1"],
            "v2_restore_trials_equal": trial_pair["V2"],
            "historical_behavioural_probe_evidence_present": True,
        }
    else:
        pair_fields = {
            field: None
            for field in (
                "v1_v2_artifact_file_sha256_equal",
                "v1_v3_artifact_file_sha256_equal",
                "v2_v3_artifact_file_sha256_equal",
                "v1_v2_semantic_equal",
                "v1_v3_semantic_equal",
                "v2_v3_semantic_equal",
                "v1_v2_behavioural_equal",
                "v1_v3_behavioural_equal",
                "v2_v3_behavioural_equal",
                "v1_restore_trials_equal",
                "v2_restore_trials_equal",
            )
        }
        pair_fields["historical_behavioural_probe_evidence_present"] = False
    passed = bool(
        trial_pair["V3"]
        and semantic_manifests_equal
        and (
            not historical
            or all(
                pair_fields[field]
                for field in (
                    "v1_v2_semantic_equal",
                    "v1_v3_semantic_equal",
                    "v2_v3_semantic_equal",
                    "v1_v2_behavioural_equal",
                    "v1_v3_behavioural_equal",
                    "v2_v3_behavioural_equal",
                    "v1_restore_trials_equal",
                    "v2_restore_trials_equal",
                    "historical_behavioural_probe_evidence_present",
                )
            )
        )
    )
    row = {
        "pool_index": pool_index,
        "candidate_spec_id": spec["candidate_spec_id"],
        "state_id": spec["state_id"],
        "scene_id": spec["scene_id"],
        "episode_id": spec["episode_id"],
        "graph_id": spec["graph_id"],
        "versions": versions,
        **pair_fields,
        "v3_restore_trials_equal": trial_pair["V3"],
        "semantic_manifests_equal": semantic_manifests_equal,
        "historical_comparison_applicable": historical,
        "trial_1_behavioural_digests": trial_one,
        "final_snapshot_semantic_digests": finals,
        "behavioural_probe_trace_indices": {
            version: (
                [
                    pool_index * 6
                    + CONTRACT.BEHAVIOURAL_PROBE_VERSION_ORDER.index(version) * 2,
                    pool_index * 6
                    + CONTRACT.BEHAVIOURAL_PROBE_VERSION_ORDER.index(version) * 2
                    + 1,
                ]
                if historical
                else [
                    48 + (pool_index - 8) * 2,
                    48 + (pool_index - 8) * 2 + 1,
                ]
            )
            for version in expected_versions
        },
        "pass": passed,
    }
    if set(row) != set(CONTRACT.SNAPSHOT_EQUIVALENCE_RECORD_FIELDS):
        raise ExperimentError("snapshot equivalence row field drift")
    return row


def _write_probe_npz_once(path: Path, arrays: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ExperimentError(f"behavioural probe NPZ is not fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    V1.atomic_npz(path, **dict(arrays))
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        reopened = {
            member: np.ascontiguousarray(archive[member])
            for member in archive.files
        }
    if set(reopened) != set(arrays):
        raise ExperimentError("behavioural probe NPZ reopen member drift")
    for member, source_value in arrays.items():
        source = np.ascontiguousarray(np.asarray(source_value))
        target = reopened[member]
        if (
            source.dtype.str != target.dtype.str
            or source.shape != target.shape
            or source.tobytes(order="C") != target.tobytes(order="C")
        ):
            raise ExperimentError(
                f"behavioural probe NPZ reopen logical drift: {member}"
            )


def compare_v1_v2_v3_first_eight_stage(
    *, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Persist the immutable 48-trace gate before pool index 8 opens."""

    if _reproduction_path().exists() or _reproduction_path().is_symlink():
        raise ExperimentError("first-eight reproduction receipt is not fresh")
    if any(
        (MATERIAL_ROOT / "qualification" / f"pool-{index:03d}").exists()
        for index in range(8, PROSPECTIVE_POOL_COUNT)
    ):
        raise ExperimentError("pool index 8 was opened before first-eight gate")
    evaluator = _load_custody_evaluator(evaluator_module)
    historical = evaluator.validate_existing_historical_custody_receipt(
        HISTORICAL_CUSTODY_RECEIPT
    )
    METRICS.validate_external_v1_v2_custody_receipt(
        historical, expected_binding=_historical_custody_binding()
    )
    trace_rows, shards = _qualification_probe_rows(first_eight_only=True)
    arrays = _probe_npz_arrays(trace_rows, first_eight_only=True)
    material_path = MATERIAL_ROOT / "reproduction" / "first_eight_behavioural_probes.npz"
    _write_probe_npz_once(material_path, arrays)
    binding = _file_binding(material_path, relative_to=MATERIAL_ROOT)
    projection = METRICS.behavioural_probe_npz_projection_sha256(
        arrays, first_eight_only=True
    )
    rows = [
        _equivalence_record(
            index,
            shards[index][0],
            shards[index][1],
            historical_custody=historical,
        )
        for index in range(8)
    ]
    value = METRICS.build_first_eight_reproduction(
        rows,
        historical_custody_receipt_binding=_historical_custody_binding(),
        first_eight_behavioural_probe_material_binding=binding,
        first_eight_behavioural_probe_projection_sha256=projection,
    )
    METRICS.validate_first_eight_reproduction(value)
    _atomic_json(_reproduction_path(), value)
    # The first-eight comparison is read-only with respect to all historical
    # roots and creates no panel/candidate/ranker/held-out evidence.
    after = evaluator.validate_existing_historical_custody_receipt(
        HISTORICAL_CUSTODY_RECEIPT
    )
    if canonical_bytes(after) != canonical_bytes(historical):
        raise ExperimentError("historical custody changed during first-eight gate")
    _assert_historical_storage_isolation()
    if value["pass"] is not True:
        observed = sorted(path.name for path in OUTPUT_ROOT.iterdir())
        if observed != sorted(CONTRACT.REPRODUCTION_MISMATCH_LEAVES):
            raise ExperimentError("technical mismatch official inventory drift")
    return value


def assemble_snapshot_equivalence_stage(
    *, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Write the complete 544-trace NPZ and 256-row semantic index once."""

    _require_reproduction_pass()
    probe_path = OUTPUT_ROOT / CONTRACT.NEW_RUNTIME_OUTPUT_PATHS[
        "snapshot_behavioural_probes"
    ]
    index_path = OUTPUT_ROOT / CONTRACT.NEW_RUNTIME_OUTPUT_PATHS[
        "snapshot_equivalence_index"
    ]
    if probe_path.exists() or index_path.exists():
        raise ExperimentError("final snapshot equivalence evidence is not fresh")
    evaluator = _load_custody_evaluator(evaluator_module)
    historical = evaluator.validate_existing_historical_custody_receipt(
        HISTORICAL_CUSTODY_RECEIPT
    )
    trace_rows, shards = _qualification_probe_rows(first_eight_only=False)
    arrays = _probe_npz_arrays(trace_rows, first_eight_only=False)
    material_path = MATERIAL_ROOT / "reproduction" / "first_eight_behavioural_probes.npz"
    import numpy as np

    with np.load(material_path, allow_pickle=False) as archive:
        first_arrays = {
            member: np.ascontiguousarray(archive[member])
            for member in archive.files
        }
    METRICS.validate_first_eight_behavioural_probe_npz_arrays(first_arrays)
    # Metadata vectors use their first 48 rows; sample vectors use 36,000.
    per_trace = {
        "version_code",
        "pool_index",
        "trial_index",
        "stuck",
        "termination_code",
        "final_snapshot_semantic_digest_bytes",
    }
    for member, expected in first_arrays.items():
        if member == "trace_offsets":
            observed = arrays[member][:49]
        elif member in per_trace:
            observed = arrays[member][:48]
        else:
            observed = arrays[member][:36000]
        if (
            observed.dtype.str != expected.dtype.str
            or observed.shape != expected.shape
            or observed.tobytes(order="C") != expected.tobytes(order="C")
        ):
            raise ExperimentError(
                f"final probe NPZ first-eight projection drift: {member}"
            )
    rows = [
        _equivalence_record(
            index,
            shards[index][0],
            shards[index][1],
            historical_custody=historical if index < 8 else None,
        )
        for index in range(PROSPECTIVE_POOL_COUNT)
    ]
    first_binding = _file_binding(material_path, relative_to=MATERIAL_ROOT)
    first_projection = METRICS.behavioural_probe_npz_projection_sha256(
        first_arrays, first_eight_only=True
    )
    final_projection = METRICS.behavioural_probe_npz_projection_sha256(
        arrays, first_eight_only=False
    )
    # Stage the exact final bytes under an unregistered same-directory name,
    # build and validate every binding, then atomically publish the NPZ once.
    pending_path = OUTPUT_ROOT / ".snapshot_behavioural_probes.pending.npz"
    _write_probe_npz_once(pending_path, arrays)
    pending_info = _require_regular(pending_path)
    final_binding = {
        "path": probe_path.name,
        "bytes": int(pending_info.st_size),
        "sha256": sha256_file(pending_path),
    }
    index = METRICS.build_snapshot_equivalence_index(
        rows,
        behavioural_probe_npz_binding=final_binding,
        behavioural_probe_npz_projection_sha256=final_projection,
        first_eight_material_probe_binding=first_binding,
        first_eight_material_probe_projection_sha256=first_projection,
        first_eight_prefix_exact=True,
    )
    METRICS.validate_snapshot_equivalence_evidence(
        index, arrays, first_arrays
    )
    os.replace(pending_path, probe_path)
    directory_fd = os.open(OUTPUT_ROOT, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if _file_binding(probe_path, relative_to=OUTPUT_ROOT) != final_binding:
        raise ExperimentError("published behavioural probe NPZ binding drift")
    _v3_atomic_json(index_path, index)
    METRICS.validate_snapshot_equivalence_evidence(index, arrays, first_arrays)
    return index


def _require_snapshot_equivalence_pass() -> dict[str, Any]:
    path = OUTPUT_ROOT / CONTRACT.NEW_RUNTIME_OUTPUT_PATHS[
        "snapshot_equivalence_index"
    ]
    value = _ordinary_json(path)
    try:
        return METRICS.validate_snapshot_equivalence_index(value)
    except Exception as exc:
        raise ExperimentError("snapshot equivalence index drift") from exc


def _gated_delegate(name: str, *args: Any, **kwargs: Any) -> Any:
    _require_reproduction_pass()
    _require_snapshot_equivalence_pass()
    return _delegate(name, *args, **kwargs)


def select_teacher_pool_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    return _gated_delegate("select_teacher_pool_stage", fake_runtime=fake_runtime)


def capture_selected_state_stage(
    state_id: str, *, backend: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    return _gated_delegate(
        "capture_selected_state_stage",
        state_id,
        backend=backend,
        fake_runtime=fake_runtime,
    )


def freeze_panel_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    return _gated_delegate("freeze_panel_stage", fake_runtime=fake_runtime)


def encode_canonical_pixels_stage(
    *, encoder: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    return _gated_delegate(
        "encode_canonical_pixels_stage",
        encoder=encoder,
        fake_runtime=fake_runtime,
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
        "development_target_selection_stage",
        ranker=ranker,
        fake_runtime=fake_runtime,
    )


def heldout_ranker_scores_stage(
    *, ranker: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    return _gated_delegate(
        "heldout_ranker_scores_stage",
        ranker=ranker,
        fake_runtime=fake_runtime,
    )


def repeat_state_stage(
    state_id: str, *, backend: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    return _gated_delegate(
        "repeat_state_stage", state_id, backend=backend, fake_runtime=fake_runtime
    )


def assemble_row_evidence_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    return _gated_delegate("assemble_row_evidence_stage", fake_runtime=fake_runtime)


def _load_behavioural_probe_arrays(
    path: Path, *, first_eight_only: bool
) -> dict[str, Any]:
    """Reopen one probe NPZ into C-contiguous logical arrays."""

    import numpy as np

    _require_regular(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            member: np.ascontiguousarray(archive[member])
            for member in archive.files
        }
    try:
        return (
            METRICS.validate_first_eight_behavioural_probe_npz_arrays(arrays)
            if first_eight_only
            else METRICS.validate_behavioural_probe_npz_arrays(arrays)
        )
    except Exception as exc:
        raise ExperimentError("behavioural probe NPZ reopen validation failed") from exc


def recompute_and_persist_metrics_stage(
    *, fake_runtime: bool = False, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Recompute V3 metrics from only persisted ledgers and logical arrays."""

    _require_reproduction_pass()
    _require_snapshot_equivalence_pass()
    with _v3_namespace():
        V1.require_stage_runtime("ordinary", fake=fake_runtime)
    evaluator = _load_custody_evaluator(evaluator_module)
    external = evaluator.validate_existing_historical_custody_receipt(
        HISTORICAL_CUSTODY_RECEIPT
    )
    METRICS.validate_external_v1_v2_custody_receipt(
        external, expected_binding=_historical_custody_binding()
    )
    document_names = (
        "panel_manifest", "split_manifest", "graph_manifest",
        "state_snapshot_index", "teacher_trace_index", "edge_port_index",
        "waypoint_contracts", "pixel_index", "latent_index",
        "development_target_selection",
    )
    documents = {
        name: _ordinary_json(OUTPUT_ROOT / f"{name}.json")
        for name in document_names
    }
    ledgers = {
        "candidate_fanout": V1.load_jsonl(OUTPUT_ROOT / "candidate_fanout.jsonl"),
        "heldout_ranker_scores": V1.load_jsonl(
            OUTPUT_ROOT / "heldout_ranker_scores.jsonl"
        ),
        "repeated_execution": V1.load_jsonl(
            OUTPUT_ROOT / "repeated_execution.jsonl"
        ),
    }
    inspections = [
        V1._inspect_npz_for_pure_metrics(
            OUTPUT_ROOT / leaf, CONTRACT.NPZ_PAYLOAD_AUTHORITY[leaf]
        )
        for leaf in (
            "state_snapshots.npz", "teacher_traces.npz",
            "rgb_observations.npz", "canonical_latents.npz",
            "candidate_traces.npz",
        )
    ]
    inspections.append(
        V1._inspect_npz_for_pure_metrics(
            OUTPUT_ROOT / "snapshot_behavioural_probes.npz",
            CONTRACT.BEHAVIOURAL_PROBE_NPZ_AUTHORITY,
        )
    )
    first_arrays = _load_behavioural_probe_arrays(
        MATERIAL_ROOT / "reproduction" / "first_eight_behavioural_probes.npz",
        first_eight_only=True,
    )
    final_arrays = _load_behavioural_probe_arrays(
        OUTPUT_ROOT / "snapshot_behavioural_probes.npz",
        first_eight_only=False,
    )
    evidence = {
        **documents,
        **ledgers,
        "npz_inspections": inspections,
        "external_historical_custody_receipt": external,
        "v1_v2_custody_and_nonreuse": _ordinary_json(
            OUTPUT_ROOT / "v1_v2_custody_and_nonreuse.json"
        ),
        "scientific_invariance_receipt": _ordinary_json(
            OUTPUT_ROOT / "scientific_invariance_receipt.json"
        ),
        "v1_v2_v3_first_eight_reproduction": _ordinary_json(
            OUTPUT_ROOT / "v1_v2_v3_first_eight_reproduction.json"
        ),
        "snapshot_equivalence_index": _ordinary_json(
            OUTPUT_ROOT / "snapshot_equivalence_index.json"
        ),
        "snapshot_behavioural_probe_arrays": final_arrays,
        "first_eight_behavioural_probe_arrays": first_arrays,
    }
    try:
        metrics = METRICS.recompute_metrics(evidence)
    except Exception as exc:
        raise ExperimentError("pure V3 metric recomputation failed") from exc
    _atomic_json(OUTPUT_ROOT / "metrics.json", metrics)
    return metrics


def _scientific_bindings() -> dict[str, dict[str, Any]]:
    return {
        leaf: _file_binding(OUTPUT_ROOT / leaf, relative_to=OUTPUT_ROOT)
        for leaf in sorted(SCIENTIFIC_LEAVES)
    }


def _snapshot_equivalence_publication_projection() -> dict[str, Any]:
    index = _ordinary_json(OUTPUT_ROOT / "snapshot_equivalence_index.json")
    try:
        validated = METRICS.validate_snapshot_equivalence_index(index)
    except Exception as exc:
        raise ExperimentError("snapshot equivalence publication index drift") from exc
    first_binding = validated["first_eight_material_probe_binding"]
    return {
        "terminal": False,
        "first_eight_row_count": 8,
        "complete_row_count": PROSPECTIVE_POOL_COUNT,
        "historical_row_count": 8,
        "v3_only_row_count": PROSPECTIVE_POOL_COUNT - 8,
        "first_eight_receipt_exact_rebuild": True,
        "snapshot_equivalence_index_binding": _file_binding(
            OUTPUT_ROOT / "snapshot_equivalence_index.json",
            relative_to=OUTPUT_ROOT,
        ),
        "snapshot_behavioural_probe_binding": _file_binding(
            OUTPUT_ROOT / "snapshot_behavioural_probes.npz",
            relative_to=OUTPUT_ROOT,
        ),
        "snapshot_behavioural_probe_projection_sha256": validated[
            "behavioural_probe_npz_projection_sha256"
        ],
        "first_eight_material_binding": copy.deepcopy(first_binding),
        "first_eight_projection_sha256": validated[
            "first_eight_material_probe_projection_sha256"
        ],
        "all_semantic_and_behavioural_rows_pass": validated["pass"] is True,
        "scientific_result_authorized": validated["pass"] is True,
    }


def _write_publication(
    metrics: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish only pure-builder result/report bytes after independent reduction."""

    runtime = _ordinary_json(OUTPUT_ROOT / "contract.json")
    material = _ordinary_json(MATERIAL_ROOT / "material_contract.json")
    started = material.get("started_at_unix_s")
    if (
        isinstance(started, bool)
        or not isinstance(started, (int, float))
        or not math.isfinite(float(started))
    ):
        raise ExperimentError("material runtime start timestamp drift")
    if not EXTERNAL_REGENERATION_RECEIPT.exists():
        raise ExperimentError("independent regeneration receipt is absent")
    receipt_raw = EXTERNAL_REGENERATION_RECEIPT.read_bytes()
    if receipt_raw != canonical_bytes(receipt):
        raise ExperimentError("independent regeneration receipt byte drift")
    elapsed_seconds = max(0.0, float(time.time()) - float(started))
    scientific_bindings = _scientific_bindings()
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    snapshot_equivalence = _snapshot_equivalence_publication_projection()
    historical_binding = _historical_custody_binding()
    try:
        projection = METRICS.build_result_publication_projection(
            metrics,
            scientific_bindings,
            runtime,
            independent_reducer_receipt_sha256=receipt_sha256,
            snapshot_equivalence=snapshot_equivalence,
            historical_custody_receipt_binding=historical_binding,
            runtime_seconds=elapsed_seconds,
        )
        result = projection["result_document"]
        rebuilt = METRICS.validate_result_publication_projection(
            result,
            recomputed_metrics=metrics,
            scientific_bindings=scientific_bindings,
            runtime_contract=runtime,
            independent_reducer_receipt_sha256=receipt_sha256,
            snapshot_equivalence=snapshot_equivalence,
            historical_custody_receipt_binding=historical_binding,
        )
        if canonical_bytes(rebuilt) != canonical_bytes(projection):
            raise ExperimentError("V3 publication projection rebuild drift")
        report_bytes = METRICS.build_result_report_bytes(projection)
    except Exception as exc:
        raise ExperimentError("pure V3 result publication construction failed") from exc
    _atomic_json(OUTPUT_ROOT / "result.json", result)
    V1.atomic_bytes(OUTPUT_ROOT / "result.md", report_bytes)
    files = [
        _file_binding(OUTPUT_ROOT / leaf, relative_to=OUTPUT_ROOT)
        for leaf in sorted(set(ALL_OUTPUT_LEAVES) - {"file_hashes.json"})
    ]
    manifest = CONTRACT.attach_content_digest(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v3."
                "file_hashes.v1"
            ),
            "root": str(OUTPUT_ROOT),
            "files": files,
            "file_count_excluding_self": len(files),
            "bytes_excluding_self": sum(int(row["bytes"]) for row in files),
            "file_hashes_self_sha256_excluded": True,
        }
    )
    _atomic_json(OUTPUT_ROOT / "file_hashes.json", manifest)
    return result


def report_stage(
    *, fake_runtime: bool = False, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Persist metrics, reduce independently, then publish exact presentation."""

    evaluator = _load_custody_evaluator(evaluator_module)
    metrics = recompute_and_persist_metrics_stage(
        fake_runtime=fake_runtime, evaluator_module=evaluator
    )
    if set(path.name for path in OUTPUT_ROOT.iterdir()) != set(SCIENTIFIC_LEAVES):
        raise ExperimentError(
            "scientific root inventory drift before independent V3 reduction"
        )
    receipt = evaluator.verify_and_emit(
        OUTPUT_ROOT,
        EXTERNAL_REGENERATION_RECEIPT,
        metrics_module=METRICS,
        material_root=MATERIAL_ROOT,
        historical_custody_receipt=HISTORICAL_CUSTODY_RECEIPT,
    )
    result = _write_publication(metrics, receipt)
    evaluator.validate_existing_regeneration_receipt(
        OUTPUT_ROOT,
        EXTERNAL_REGENERATION_RECEIPT,
        metrics_module=METRICS,
        material_root=MATERIAL_ROOT,
        historical_custody_receipt=HISTORICAL_CUSTODY_RECEIPT,
    )
    if set(path.name for path in OUTPUT_ROOT.iterdir()) != set(ALL_OUTPUT_LEAVES):
        raise ExperimentError("final V3 official root inventory drift")
    return result


def physical_backend_smoke_stage() -> dict[str, Any]:
    return _delegate("physical_backend_smoke_stage")


def production_snapshot_fixture_stage(
    *, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Bounded pre-freeze retained-snapshot semantic/probe fixture.

    This opens no teacher controller and creates no official/material file.  It
    exercises the required seven-session lifecycle using the already-observed
    pool-000 state, then revalidates both historical trees byte/inode exactly.
    """

    if any(
        path.exists() or path.is_symlink()
        for path in (OUTPUT_ROOT, MATERIAL_ROOT, EXTERNAL_REGENERATION_RECEIPT)
    ):
        raise ExperimentError("production snapshot fixture requires V3 roots absent")
    V1.require_stage_runtime("physical")
    evaluator = _load_custody_evaluator(evaluator_module)
    historical_before = evaluator.validate_existing_historical_custody_receipt(
        HISTORICAL_CUSTODY_RECEIPT
    )
    import numpy as np

    spec = CONTRACT.build_prospective_pool_specs()[0]
    paths = {
        "V1": V1_MATERIAL_ROOT / "qualification/pool-000/payload.npz",
        "V2": V2_MATERIAL_ROOT / "qualification/pool-000/payload.npz",
    }
    payloads: dict[str, bytes] = {}
    semantics: dict[str, dict[str, Any]] = {}
    for version, path in paths.items():
        semantics[version] = _historical_snapshot_semantics(
            path, evaluator_module=evaluator
        )
        with np.load(path, allow_pickle=False) as archive:
            payloads[version] = np.ascontiguousarray(
                archive["snapshot_payload_bytes"]
            ).tobytes(order="C")
    # Session 1: restore retained V1 state and freshly serialize its V3
    # semantic-equivalent snapshot.  No command/physics/teacher is executed.
    capture_session = _V3GenesisPhysicalSession(spec, backend="cpu")
    capture_session.restore_snapshot(payloads["V1"])
    v3_payload, _snapshot, _auxiliary = capture_session.capture_snapshot()
    semantics["V3"] = _fresh_snapshot_semantics(v3_payload)
    # Sessions 2..7: one fresh simulator instance per restoration trial.
    probes = {
        "V3": _execute_probe_trials(
            lambda: _V3GenesisPhysicalSession(spec, backend="cpu"), v3_payload
        ),
        "V1": _execute_probe_trials(
            lambda: _V3GenesisPhysicalSession(spec, backend="cpu"), payloads["V1"]
        ),
        "V2": _execute_probe_trials(
            lambda: _V3GenesisPhysicalSession(spec, backend="cpu"), payloads["V2"]
        ),
    }
    diagnostics = _production_fixture_diagnostics(semantics, probes)
    semantic_equal = diagnostics["semantic_equal"]
    within = diagnostics["within_version_probe_equal"]
    cross = diagnostics["cross_version_probe_equal"]
    after = evaluator.validate_existing_historical_custody_receipt(
        HISTORICAL_CUSTODY_RECEIPT
    )
    if canonical_bytes(after) != canonical_bytes(historical_before):
        raise ExperimentError("production fixture changed historical custody")
    if OUTPUT_ROOT.exists() or MATERIAL_ROOT.exists():
        raise ExperimentError("production fixture created a V3 result root")
    passed = bool(semantic_equal and all(within.values()) and all(cross.values()))
    if not passed:
        raise ExperimentError(_production_fixture_failure_message(diagnostics))
    return {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "production_snapshot_fixture.v1"
        ),
        "pool_index": 0,
        "teacher_controller_executions": 0,
        "simulator_session_count": 7,
        "restore_probe_trials": 6,
        "physics_samples": 4500,
        "semantic_equal": semantic_equal,
        "within_version_probe_equal": within,
        "cross_version_probe_equal": cross,
        "historical_roots_unchanged": True,
        "official_or_material_root_created": False,
        "pass": True,
    }


# ---------------------------------------------------------------------------
# Prospective documents and direct CLI
# ---------------------------------------------------------------------------


def _preregistration_text() -> str:
    return (
        "# PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V3\n\n"
        "Development-only semantic replacement of V2. V3 preserves the exact V2/V1 "
        "physical panel, teacher, candidate bank, target selection, frozen encoder, "
        "frozen ranker, metrics, gates, classifications, and next-decision tree.\n\n"
        "V3 adds three descriptive snapshot identities: exact artifact-file SHA-256, "
        "canonical snapshot_semantic_digest_v1, and the designated trial-0 "
        "snapshot_behavioural_digest_v1. Raw artifact equality is descriptive only.\n\n"
        "Every freshly captured V3 pool snapshot receives two 750-sample restores, each "
        "in its own fresh simulator instance under command [0.2,0,0]. The exact 45-D controller "
        "observation before each policy act and raw 12-D policy._last_actions after that "
        "act are repeated over its ten 2 ms samples. Pools 0..7 also restore immutable "
        "historical V1/V2 snapshots read-only in separate fresh instances.\n\n"
        "Before pool 8, the immutable 48-trace material subset and eight-row semantic/"
        "behavioural comparison must pass. Failure is the technical terminal "
        f"`{CONTRACT.REPRODUCTION_MISMATCH_DISPOSITION}` with exactly four official "
        "leaves and no scientific result. Success authorizes the unchanged inherited "
        "pipeline and one final 544-trace official probe NPZ.\n\n"
        "No model, threshold, scientific formula, gate, class, target, candidate, "
        "split, or tuning rule changes. Historical shards are never copied, hardlinked, "
        "or used as V3 teacher/candidate/ranker inputs.\n"
    )


def build_freeze_documents(
    *, evaluator_module: Any | None = None
) -> dict[str, Any]:
    if _git("rev-parse", "HEAD") != PARENT_COMMIT:
        raise ExperimentError("V3 freeze documents require the frozen V2 parent")
    external = validate_historical_custody_before_creation(
        evaluator_module=evaluator_module
    )
    invariance_receipt = build_scientific_invariance_receipt(external)
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
            "semantic_serializer_regressions": list(
                CONTRACT.SEMANTIC_SERIALIZER_REGRESSION_IDS
            ),
            "production_snapshot_fixture": {
                "pool_index": 0,
                "teacher_controller_executions": 0,
                "simulator_sessions": CONTRACT.BEHAVIOURAL_PROBE_AUTHORITY[
                    "probe_session_topology"
                ]["production_fixture_sessions"],
                "probe_trials": 6,
            },
            "first_eight_gate": list(range(8)),
            "first_eight_probe_traces": 48,
            "complete_probe_traces": CONTRACT.BEHAVIOURAL_PROBE_TRACE_COUNT,
            "probe_physics_samples_per_trace": (
                CONTRACT.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES
            ),
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
            "historical_custody_receipt": _historical_custody_binding(),
            "receipt_self_digests": False,
        }
    )
    _atomic_json(DOC_PATHS["contract"], contract_document)
    _atomic_json(DOC_PATHS["fixture"], fixture)
    _atomic_json(DOC_PATHS["output_schema"], output_schema)
    V1.atomic_bytes(
        DOC_PATHS["preregistration"], _preregistration_text().encode("utf-8")
    )
    _atomic_json(
        DOC_PATHS["scientific_invariance"],
        V1.attach_digest(copy.deepcopy(CONTRACT.SCIENTIFIC_INVARIANCE_AUTHORITY)),
    )
    _atomic_json(
        DOC_PATHS["historical_custody_binding"],
        V1.attach_digest(
            {
                **copy.deepcopy(CONTRACT.V1_V2_CUSTODY_AND_NONREUSE_AUTHORITY),
                "bound_external_receipt": _historical_custody_binding(),
            }
        ),
    )
    rows = []
    for relative in CONTRACT.SOURCE_CLOSURE_PATHS:
        path = REPO_ROOT / relative
        info = _require_regular(path)
        rows.append(
            {
                "path": relative,
                "bytes": int(info.st_size),
                "sha256": sha256_file(path),
            }
        )
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
    # The ordinary scientific receipt is runtime evidence; the prospective
    # document above freezes only its authority, not a result claim.
    if invariance_receipt["pass"] is not True:
        raise ExperimentError("V3 semantic invariance preregistration failed")
    return contract_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("freeze-docs")
    subparsers.add_parser("production-snapshot-fixture")
    subparsers.add_parser("initialize")
    qualify = subparsers.add_parser("qualify-pool-state")
    qualify.add_argument("--pool-index", type=int, required=True)
    subparsers.add_parser("compare-v1-v2-v3-first-eight")
    subparsers.add_parser("assemble-snapshot-equivalence")
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
    if stage == "freeze-docs":
        result: Any = build_freeze_documents()
    elif stage == "production-snapshot-fixture":
        result = production_snapshot_fixture_stage()
    elif stage == "initialize":
        result = initialize_stage()
    elif stage == "qualify-pool-state":
        result = qualify_pool_state_stage(arguments.pool_index)
    elif stage == "compare-v1-v2-v3-first-eight":
        result = compare_v1_v2_v3_first_eight_stage()
    elif stage == "assemble-snapshot-equivalence":
        result = assemble_snapshot_equivalence_stage()
    elif stage == "select-teacher-pool":
        result = select_teacher_pool_stage()
    elif stage == "qualify-selected-reset":
        result = capture_selected_state_stage(arguments.state_id)
    elif stage == "freeze-panel":
        result = freeze_panel_stage()
    elif stage == "encode-canonical-pixels":
        result = encode_canonical_pixels_stage()
    elif stage == "fanout-state":
        result = fanout_state_stage(arguments.state_id)
    elif stage == "select-development-target":
        result = development_target_selection_stage()
    elif stage == "score-heldout":
        result = heldout_ranker_scores_stage()
    elif stage == "repeat-state":
        result = repeat_state_stage(arguments.state_id)
    elif stage == "assemble-row-evidence":
        result = assemble_row_evidence_stage()
    elif stage == "report":
        result = report_stage()
    elif stage == "physical-smoke":
        result = physical_backend_smoke_stage()
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
