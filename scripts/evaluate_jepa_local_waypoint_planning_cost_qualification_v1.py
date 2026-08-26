#!/usr/bin/env python3
"""Execute JEPA local-waypoint planning-cost qualification V1.

The public coordinator is deliberately import-safe: importing it opens no
scientific ledger, checkpoint, tensor, simulator, or GPU.  Genesis work is
performed only by the internal CPU state worker after the prospective freeze;
encoder and predictor work is delegated to the separately source-closed
TinyQuadJEPA entrypoint.

The evaluator is development-only.  It trains nothing and does not implement
closed-loop navigation, memory, routing, beacon capture, or a learned safety
decision.  Oracle contact/viability values are evaluation strata only.
"""

from __future__ import annotations

import argparse
import base64
import copy
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


# Package RECORD closure includes the exact set of presently materialised
# bytecode rows.  Prevent this evaluator and every forked worker from changing
# that set while it validates or executes the frozen environment.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import (  # noqa: E402
    jepa_local_waypoint_planning_cost_metrics_v1 as METRICS,
)
from lewm.safety import (  # noqa: E402
    jepa_local_waypoint_planning_cost_qualification_v1_contract as CONTRACT,
)


GPU_ENTRYPOINT = ROOT / "scripts/run_jepa_local_waypoint_planning_cost_inference_v1.py"
CPU_INTERPRETER = ROOT / ".generated/venvs/genesis_render_vulkan/bin/python"
GPU_INTERPRETER = Path("/home/andrewknowles/TinyQuadJEPA/bin/python")
OUTPUT_ROOT = CONTRACT.OUTPUT_ROOT

V1_ROOT = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
V2_ROOT = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
DENSE_ROOT = ROOT / ".generated/dense_temporal_true_future_safety_observability_v1"
CONTACT_ROOT = ROOT / ".generated/contact_hazard_ontology_and_instrumentation_v1"

STATE_MANIFEST = V1_ROOT / "state_manifest.json"
SPLIT = V1_ROOT / "split.json"
BRANCH_LEDGER = V1_ROOT / "branch_labels.jsonl"
ROUTE_LABELS = V2_ROOT / "route_intent_labels.jsonl"
TARGET_LATENT_INDEX = V2_ROOT / "target_latent_index.json"
DENSE_TOKEN_INDEX = DENSE_ROOT / "token_index.json"
CONTACT_EVENT_INDEX = CONTACT_ROOT / "raw_contact_event_index.json"
DENSE_EVIDENCE_RECEIPT = DENSE_ROOT / "evidence_receipt.json"
CONTROL_NORMALISATION = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1/proprio_norm_stats.json"
)

CONTEXT_INDEX_REL = Path("materialization/context_reconstruction_index.json")
FANOUT_INDEX_REL = Path("materialization/oracle_admissibility_fanout_index.json")
DENSE_REPLAY_INPUT_INDEX_REL = Path("materialization/dense_route_replay_input_index.json")
LATENT_INDEX_REL = Path("latents/tensor_index.json")
GOAL_INDEX_REL = Path("goal_views/index.json")
PREEXEC_REL = Path("receipts/preexecution.json")
ENVIRONMENT_REL = Path("receipts/environment.json")
PERSISTENCE_REL = Path("receipts/persistence.json")
CPU_RUNTIME_INPUT_INVENTORY_REL = Path(
    "materialization/cpu_runtime_input_inventory.json"
)
GPU_ENVIRONMENT_REL = Path("receipts/gpu_environment.json")
GPU_INFERENCE_REL = Path("receipts/gpu_inference.json")
CANDIDATE_LEDGER_REL = Path("evidence/candidate_evidence.jsonl.gz")
SELECTION_LEDGER_REL = Path("evidence/selection_evidence.jsonl.gz")
PAIRED_LEDGER_REL = Path("evidence/paired_effect_evidence.jsonl.gz")
AGGREGATE_REL = Path("aggregates/metrics.json")
RESULT_REL = Path("result.json")
REPORT_REL = Path("report.md")
RUNNING_REL = Path("receipts/RUNNING.json")

TENSOR_SHAPE = (768, 1024)
TENSOR_DTYPE = np.dtype(np.float16)
TENSOR_RAW_BYTES = int(np.prod(TENSOR_SHAPE) * TENSOR_DTYPE.itemsize)
PHYSICS_DT_S = 0.002
PHYSICS_STEPS_PER_BLOCK = 250
STATE_COUNT = 48
CANDIDATES_PER_STATE = 12
PRIMITIVE_COUNT = 9

INPUT_BINDINGS = {
    "state_manifest": {
        "path": STATE_MANIFEST,
        "sha256": "da67309c073f60d74e4b85427237b19691552a542136e6ddb95939f14b4c5c37",
    },
    "split": {
        "path": SPLIT,
        "sha256": "ebef7db828a4c754432375818fd6b1eff0731cc3bc546ff2b69667b03abe56a8",
    },
    "branch_ledger": {
        "path": BRANCH_LEDGER,
        "sha256": "9b25b227c3e4de11e68e4abee454c4251399fafb468458a4e0d65f89bc6cdf7c",
    },
    "route_labels": {
        "path": ROUTE_LABELS,
        "sha256": "e8d33671502f717426836ec9a1039d445558b81e636ae105a81e2113151a8b69",
    },
    "target_latent_index": {
        "path": TARGET_LATENT_INDEX,
        "sha256": "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874",
    },
    "dense_token_index": {
        "path": DENSE_TOKEN_INDEX,
        "sha256": "3cf2d42f52525ce8291f76ee5af0bd58ef0928d62a8f879efb40a9ac6530cd15",
    },
    "contact_event_index": {
        "path": CONTACT_EVENT_INDEX,
        "sha256": "1eac5be90b48e88cac7aa8db7f3ce3bd6655e1404f7ba6591ac90afa9e1f0d4d",
    },
    "dense_evidence_receipt": {
        "path": DENSE_EVIDENCE_RECEIPT,
        "sha256": "a547ac544a869a6ef75a4798b22875291e55604f9e53ceaea24a790db09df7e1",
    },
    "control_normalisation": {
        "path": CONTROL_NORMALISATION,
        "sha256": "9380b4c6d9b59099e43bba9898e1417c273f88075d1ed122401cbb3272e18f94",
    },
}


class QualificationError(RuntimeError):
    """Fail-closed execution or custody error."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Finite, canonical compact JSON with one terminal LF."""

    def ready(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): ready(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, np.ndarray):
            return ready(item.tolist())
        if isinstance(item, np.generic):
            return ready(item.item())
        if isinstance(item, (list, tuple)):
            return [ready(child) for child in item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise QualificationError("canonical JSON forbids non-finite floats")
            return item
        if item is None or isinstance(item, (str, int, bool)):
            return item
        raise QualificationError(f"unsupported JSON value {type(item).__name__}")

    return (
        json.dumps(
            ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        + "\n"
    ).encode("utf-8")


def attach_digest(value: Mapping[str, Any], key: str = "content_digest") -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop(key, None)
    payload[key] = hashlib.sha256(canonical_json_bytes(payload)[:-1]).hexdigest()
    return payload


def validate_digest(value: Mapping[str, Any], key: str = "content_digest") -> None:
    payload = copy.deepcopy(dict(value))
    declared = payload.pop(key, None)
    if not isinstance(declared, str) or len(declared) != 64:
        raise QualificationError(f"missing or malformed {key}")
    observed = hashlib.sha256(canonical_json_bytes(payload)[:-1]).hexdigest()
    if observed != declared:
        raise QualificationError(f"{key} mismatch")


def atomic_bytes(path: str | Path, payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return atomic_bytes(path, canonical_json_bytes(value))


def atomic_npy(path: str | Path, value: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def _artifact_reference(path: str | Path, output_root: Path) -> str:
    """Return an output-root-relative path stable across atomic publication."""

    target = Path(path).resolve()
    root = Path(output_root).resolve()
    try:
        return str(target.relative_to(root))
    except ValueError as exc:
        raise QualificationError(f"artifact is outside output root: {target}") from exc


def _artifact_path(reference: str | Path, output_root: Path) -> Path:
    value = Path(reference)
    return value if value.is_absolute() else Path(output_root) / value


def deterministic_gzip_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    sink = io.BytesIO()
    with gzip.GzipFile(fileobj=sink, mode="wb", filename="", mtime=0, compresslevel=6) as zipped:
        for row in rows:
            zipped.write(canonical_json_bytes(row))
    return sink.getvalue()


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"cannot load JSON {path}") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QualificationError(
                        f"JSONL row {line_number} is not an object: {path}"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"cannot load JSONL {path}") from exc
    return rows


def load_gzip_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QualificationError(
                        f"gzip JSONL row {line_number} is not an object: {path}"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"cannot load gzip JSONL {path}") from exc
    return rows


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def validate_source_freeze_commit(
    source_freeze_commit: str, *, require_live_head: bool
) -> None:
    if not isinstance(source_freeze_commit, str) or len(source_freeze_commit) != 40:
        raise QualificationError("source-freeze commit must be a full 40-hex commit")
    try:
        int(source_freeze_commit, 16)
    except ValueError as exc:
        raise QualificationError("source-freeze commit is not hexadecimal") from exc
    commit = git_output("rev-parse", f"{source_freeze_commit}^{{commit}}")
    if commit != source_freeze_commit:
        raise QualificationError("source-freeze commit does not resolve exactly")
    if require_live_head and git_output("rev-parse", "HEAD") != source_freeze_commit:
        raise QualificationError("live HEAD differs from source-freeze commit")


def _receipt_binding(path: Path, *, digest_key: str | None = None) -> dict[str, Any]:
    value = load_json(path)
    if digest_key is not None:
        validate_digest(value, digest_key)
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "content_digest": None if digest_key is None else value[digest_key],
    }


def validate_frozen_receipts() -> dict[str, Any]:
    contract = CONTRACT.load_and_validate_contract()
    schema = CONTRACT.load_and_validate_output_schema()
    fixture = CONTRACT.load_and_validate_fixture_receipt()
    checks = fixture.get("executed_checks")
    if fixture.get("pass") is not True or not isinstance(checks, Mapping) or not checks:
        raise QualificationError("frozen fixture receipt did not execute and pass checks")
    if any(value is not True for value in checks.values()):
        raise QualificationError("one or more frozen fixture checks did not pass")
    closure = load_json(ROOT / CONTRACT.TRACKED_SOURCE_CLOSURE_PATH)
    CONTRACT.validate_content_digest(closure)
    if closure.get("complete") is not True or closure.get("missing_paths") != []:
        raise QualificationError("frozen source closure is incomplete")
    try:
        regenerated_closure = CONTRACT.build_source_closure(
            ROOT, require_complete=True
        )
    except CONTRACT.ContractError as exc:
        raise QualificationError("source-closure domain regeneration failed") from exc
    if canonical_json_bytes(regenerated_closure) != canonical_json_bytes(closure):
        raise QualificationError(
            "stored source closure differs from the complete frozen path domain"
        )
    for row in closure.get("rows", []):
        path = ROOT / str(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise QualificationError(f"source-closure drift: {path}")
    return {
        "contract": contract,
        "output_schema": schema,
        "fixture": fixture,
        "source_closure": closure,
    }


def _validated_interpreter_binary_binding(
    executable: str | Path,
) -> dict[str, Any]:
    """Bind the resolved system interpreter behind a frozen venv entrypoint."""

    resolved = Path(executable).resolve()
    if not resolved.is_file():
        raise QualificationError("resolved interpreter binary is missing")
    observed = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if observed != CONTRACT.INTERPRETER_BINARY_BINDING:
        raise QualificationError("resolved interpreter binary binding drift")
    return observed


def _validate_cpu_interpreter_entrypoint() -> dict[str, Any]:
    entrypoint = Path(sys.executable).absolute()
    if entrypoint != CPU_INTERPRETER.absolute():
        raise QualificationError("process is not using the frozen CPU venv entrypoint")
    return _validated_interpreter_binary_binding(entrypoint)


def validate_input_hashes() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, binding in INPUT_BINDINGS.items():
        path = Path(binding["path"])
        if not path.is_file():
            raise QualificationError(f"missing frozen input {name}: {path}")
        digest = sha256_file(path)
        if digest != binding["sha256"]:
            raise QualificationError(
                f"frozen input hash drift for {name}: {digest} != {binding['sha256']}"
            )
        logical_stats_digest = None
        if name == "control_normalisation":
            logical_stats_digest = load_json(path).get("sha256")
            if logical_stats_digest != (
                "f5ea58b29d79362d4d814ff1b4225b54a5c97fb95442c866def80b0c2c4c2fab"
            ):
                raise QualificationError("control-normalisation logical digest drift")
        output[name] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
            **(
                {"stats_sha256": logical_stats_digest}
                if logical_stats_digest is not None
                else {}
            ),
        }
    return output


def _validated_file_binding(
    binding: Mapping[str, Any], *, relative_to_root: bool
) -> dict[str, Any]:
    path = ROOT / str(binding["path"]) if relative_to_root else Path(
        str(binding["resolved_path"])
    )
    path = path.resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(binding["bytes"])
        or sha256_file(path) != binding["sha256"]
    ):
        raise QualificationError(f"CPU runtime input binding drift: {path}")
    return {
        "path": str(path),
        "sha256": binding["sha256"],
        "bytes": int(binding["bytes"]),
    }


def _validate_prefreeze_byte_inventory(
    records: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    *,
    states: int | None,
) -> dict[str, Any]:
    """Validate a prospectively frozen byte inventory before payload parsing."""

    rows = [dict(row) for row in records]
    required_fields = list(binding["record_fields"])
    if any(set(row) != set(required_fields) for row in rows):
        raise QualificationError("prefreeze byte-inventory record field/order drift")
    for row in rows:
        path = Path(str(row["path"]))
        if path.is_absolute() or ".." in path.parts:
            raise QualificationError("prefreeze byte-inventory path is not repo-relative")
        source = ROOT / path
        if (
            not source.is_file()
            or source.stat().st_size != int(row["bytes"])
            or sha256_file(source) != row["sha256"]
        ):
            raise QualificationError(f"prefreeze byte-inventory file drift: {path}")
    canonical = canonical_json_bytes(rows)[:-1]
    observed = {
        "record_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "canonical_records_bytes": len(canonical),
        "canonical_sorted_path_sha_bytes_aggregate_sha256": hashlib.sha256(
            canonical
        ).hexdigest(),
    }
    expected = {key: binding[key] for key in observed}
    if observed != expected:
        raise QualificationError(
            f"prefreeze byte-inventory aggregate drift: {observed!r} != {expected!r}"
        )
    if states is not None and int(binding.get("states", -1)) != states:
        raise QualificationError("prefreeze byte-inventory state-count drift")
    return {
        "record_count": observed["record_count"],
        **({"states": states} if states is not None else {}),
        "total_bytes": observed["total_bytes"],
        "canonical_sorted_path_sha_bytes_aggregate_sha256": observed[
            "canonical_sorted_path_sha_bytes_aggregate_sha256"
        ],
        "validated_before_json_parse": True,
        "outcome_fields_parsed_before_validation": [],
        "pass": True,
    }


def _file_digests(path: Path, algorithms: Iterable[str]) -> dict[str, bytes]:
    names = sorted(set(algorithms) | {"sha256"})
    try:
        hashers = {name: hashlib.new(name) for name in names}
    except ValueError as exc:
        raise QualificationError(f"unsupported RECORD hash algorithm: {exc}") from exc
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {name: hasher.digest() for name, hasher in hashers.items()}


def _validate_distribution_record_closure(
    package_id: str, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate every installed file named by one frozen dist-info RECORD."""

    record_expected = expected["record_closure"]
    record_path = Path(str(record_expected["record_path"])).resolve()
    if (
        not record_path.is_file()
        or record_path.stat().st_size != int(record_expected["record_bytes"])
        or sha256_file(record_path) != record_expected["record_sha256"]
    ):
        raise QualificationError(f"CPU package RECORD binding drift: {package_id}")
    base = record_path.parent.parent
    present: list[dict[str, Any]] = []
    absent_unhashed: list[str] = []
    declared_hash_entries = 0
    with record_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if any(len(row) != 3 for row in rows):
        raise QualificationError(f"malformed CPU package RECORD: {package_id}")
    for relative, declared, declared_size in rows:
        path = (base / relative).resolve()
        if not path.is_file():
            if declared:
                raise QualificationError(
                    f"hash-bearing CPU package RECORD path is absent: {package_id}:{relative}"
                )
            absent_unhashed.append(relative)
            continue
        algorithm = "sha256"
        encoded = None
        if declared:
            declared_hash_entries += 1
            if "=" not in declared:
                raise QualificationError(
                    f"malformed CPU package RECORD digest: {package_id}:{relative}"
                )
            algorithm, encoded = declared.split("=", 1)
        digests = _file_digests(path, (algorithm,))
        actual_bytes = path.stat().st_size
        if declared_size and actual_bytes != int(declared_size):
            raise QualificationError(
                f"CPU package RECORD size drift: {package_id}:{relative}"
            )
        if encoded is not None:
            observed = base64.urlsafe_b64encode(digests[algorithm]).decode(
                "ascii"
            ).rstrip("=")
            if observed != encoded:
                raise QualificationError(
                    f"CPU package RECORD digest drift: {package_id}:{relative}"
                )
        present.append(
            {
                "path": relative,
                "sha256": digests["sha256"].hex(),
                "bytes": actual_bytes,
            }
        )
    present.sort(key=lambda row: str(row["path"]))
    absent_unhashed.sort()
    observed = {
        "record_path": str(record_path),
        "record_sha256": sha256_file(record_path),
        "record_bytes": record_path.stat().st_size,
        "record_entries": len(rows),
        "declared_hash_entries": declared_hash_entries,
        "present_files": len(present),
        "absent_unhashed_files": len(absent_unhashed),
        "absent_unhashed_path_list_sha256": hashlib.sha256(
            canonical_json_bytes(absent_unhashed)[:-1]
        ).hexdigest(),
        "present_file_bytes": sum(int(row["bytes"]) for row in present),
        "present_file_aggregate_sha256": hashlib.sha256(
            canonical_json_bytes(present)[:-1]
        ).hexdigest(),
    }
    if observed != dict(record_expected):
        differing = sorted(
            key
            for key in set(observed) | set(record_expected)
            if observed.get(key) != record_expected.get(key)
        )
        raise QualificationError(
            f"CPU package RECORD closure drift: {package_id}: {differing}"
        )
    return {
        "distribution": expected["distribution"],
        "version": expected["version"],
        "package_root": str(Path(str(expected["package_root"])).resolve()),
        "record_closure": observed,
        "all_declared_hashes_and_sizes_valid": True,
        "all_present_unhashed_rows_directly_hashed": True,
        "absent_rows_all_unhashed_and_exact": True,
        "pass": True,
    }


def _runtime_import_resolution(import_name: str, expected_root: Path) -> dict[str, Any]:
    expected_root = expected_root.resolve()
    spec = importlib.util.find_spec(import_name)
    if spec is None or spec.origin is None:
        raise QualificationError(f"CPU runtime import is unresolved: {import_name}")
    origin = Path(spec.origin).resolve()
    locations = sorted(
        str(Path(value).resolve()) for value in (spec.submodule_search_locations or ())
    )
    if not origin.is_relative_to(expected_root) or any(
        not Path(value).is_relative_to(expected_root) for value in locations
    ):
        raise QualificationError(f"CPU runtime import shadowing detected: {import_name}")
    module = importlib.import_module(import_name)
    module_file_raw = getattr(module, "__file__", None)
    if module_file_raw is None:
        raise QualificationError(f"CPU runtime import lacks __file__: {import_name}")
    module_file = Path(str(module_file_raw)).resolve()
    if not module_file.is_relative_to(expected_root):
        raise QualificationError(f"CPU live module root drift: {import_name}")
    return {
        "import_name": import_name,
        "find_spec_origin": str(origin),
        "submodule_search_locations": locations,
        "live_module_file": str(module_file),
        "expected_package_root": str(expected_root),
        "resolved_inside_frozen_package_root": True,
        "pass": True,
    }


def build_cpu_runtime_input_inventory(source_freeze_commit: str) -> dict[str, Any]:
    """Recompute every simulator/controller/render input without outcome access."""

    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or sys.dont_write_bytecode is not True
    ):
        raise QualificationError("CPU package closure requires bytecode writes disabled")
    if os.environ.get("LEWM_TEXTURE_ROOT") is not None:
        raise QualificationError("LEWM_TEXTURE_ROOT must be absent for frozen rendering")
    bindings = CONTRACT.CPU_RUNTIME_INPUT_BINDINGS
    platform = _validated_file_binding(
        bindings["platform_manifest"], relative_to_root=True
    )
    primitive_registry = _validated_file_binding(
        bindings["primitive_registry"], relative_to_root=True
    )
    policy_artifacts = {
        name: _validated_file_binding(binding, relative_to_root=True)
        for name, binding in bindings["policy_artifacts"].items()
    }
    urdf_binding = bindings["genesis_builtin_urdf"]
    validated_urdf = _validated_file_binding(urdf_binding, relative_to_root=False)
    urdf = {
        "interpreter_relative_path": urdf_binding["interpreter_relative_path"],
        "resolved_path": validated_urdf["path"],
        "sha256": validated_urdf["sha256"],
        "bytes": validated_urdf["bytes"],
    }
    # Every visual mesh reference in the frozen URDF is ``../dae/<name>``
    # relative to ``go2/urdf/go2.urdf``.  Resolve the actual URDF-relative
    # location; an invented ``meshes`` directory would silently validate the
    # wrong asset namespace (or fail only after the long replay starts).
    urdf_root = Path(urdf["resolved_path"]).parent
    try:
        urdf_tree = ET.parse(urdf["resolved_path"])
    except ET.ParseError as exc:
        raise QualificationError("frozen Go2 URDF is not parseable XML") from exc
    observed_mesh_references = sorted(
        {
            str(element.attrib["filename"])
            for element in urdf_tree.iter()
            if element.tag.rsplit("}", 1)[-1] == "mesh"
            and "filename" in element.attrib
        }
    )
    expected_mesh_references = sorted(
        str(row["relative_path"]) for row in urdf_binding["referenced_meshes"]
    )
    if observed_mesh_references != expected_mesh_references:
        raise QualificationError("Go2 URDF referenced-mesh identity drift")
    referenced_meshes = []
    for expected in urdf_binding["referenced_meshes"]:
        path = (urdf_root / str(expected["relative_path"])).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(expected["bytes"])
            or sha256_file(path) != expected["sha256"]
        ):
            raise QualificationError(f"Go2 URDF visual mesh binding drift: {path}")
        referenced_meshes.append(
            {
                "name": expected["name"],
                "relative_path": expected["relative_path"],
                "path": str(path),
                "sha256": expected["sha256"],
                "bytes": int(expected["bytes"]),
            }
        )
    urdf["referenced_meshes"] = referenced_meshes
    urdf["mesh_reference_validation"] = {
        "observed_unique_relative_paths": observed_mesh_references,
        "expected_unique_relative_paths": expected_mesh_references,
        "exact_set_match": True,
        "pass": True,
    }

    package_roots: dict[str, Any] = {}
    for name, expected in bindings["cpu_packages"].items():
        version = importlib.metadata.version(str(expected["distribution"]))
        root = Path(str(expected["package_root"])).resolve()
        if version != expected["version"] or not root.is_dir() or root != Path(
            str(expected["package_root"])
        ).resolve():
            raise QualificationError(f"CPU package root/version drift: {name}")
        package_roots[name] = _validate_distribution_record_closure(name, expected)
    for name, expected in bindings["cpu_packages"].items():
        package_roots[name]["import_resolution"] = _runtime_import_resolution(
            name, Path(str(expected["package_root"]))
        )
    foundational_package_roots: dict[str, Any] = {}
    for name, expected in bindings["foundational_packages"].items():
        version = importlib.metadata.version(str(expected["distribution"]))
        root = Path(str(expected["package_root"])).resolve()
        if version != expected["version"] or not root.is_dir():
            raise QualificationError(f"CPU foundational package drift: {name}")
        foundational_package_roots[name] = {
            **copy.deepcopy(expected),
            "package_root": str(root),
            "import_resolution": _runtime_import_resolution(
                str(expected["import_name"]), root
            ),
        }

    texture_root = (ROOT / bindings["textures"]["resolved_root"]).resolve()
    texture_records = []
    texture_paths: set[str] = set()
    for expected in bindings["textures"]["records"]:
        record = _validated_file_binding(expected, relative_to_root=True)
        texture_records.append(record)
        texture_paths.add(record["path"])
    discovered_texture_paths = {
        str(path.resolve())
        for category in ("floor", "wall", "obstacle")
        for path in (texture_root / category).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    if discovered_texture_paths != texture_paths:
        raise QualificationError(
            "CPU texture inventory has extra, missing, or renamed image files"
        )

    for package_root in (ROOT / "lewm_worlds", ROOT / "lewm_genesis"):
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
    from lewm_genesis.textures import (  # noqa: PLC0415 - source-closed pure helper.
        select_scene_textures,
    )
    from lewm_worlds.manifest import (  # noqa: PLC0415 - source-closed pure helper.
        manifest_sha256,
        parse_scene_manifest_dict,
    )

    state_manifest = load_json(STATE_MANIFEST)
    states = state_manifest.get("state_candidates")
    if not isinstance(states, list) or len(states) != STATE_COUNT:
        raise QualificationError("CPU inventory state manifest cardinality drift")
    prefreeze_scene_records: list[dict[str, Any]] = []
    scene_descriptors: list[tuple[Mapping[str, Any], str, str, Path, Path, Path]] = []
    seen_states: set[str] = set()
    seen_scene_dirs: set[str] = set()
    for state in states:
        state_id = str(state["state_id"])
        scene_id = str(state["scene_id"])
        scene_dir = Path(str(state["scene_dir"])).resolve()
        if state_id in seen_states or str(scene_dir) in seen_scene_dirs:
            raise QualificationError("CPU inventory repeats a state or scene directory")
        seen_states.add(state_id)
        seen_scene_dirs.add(str(scene_dir))
        manifest_path = scene_dir / "manifest.json"
        genesis_path = scene_dir / "genesis_scene.json"
        if not manifest_path.is_file() or not genesis_path.is_file():
            raise QualificationError(f"CPU inventory scene files missing: {scene_dir}")
        for kind, path in (
            ("manifest", manifest_path),
            ("genesis_scene", genesis_path),
        ):
            try:
                relative = path.relative_to(ROOT)
            except ValueError as exc:
                raise QualificationError(
                    f"CPU scene input is outside the frozen repository: {path}"
                ) from exc
            prefreeze_scene_records.append(
                {
                    "state_id": state_id,
                    "scene_id": scene_id,
                    "kind": kind,
                    "path": str(relative),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        scene_descriptors.append(
            (state, state_id, scene_id, scene_dir, manifest_path, genesis_path)
        )
    prefreeze_scene_validation = _validate_prefreeze_byte_inventory(
        prefreeze_scene_records,
        CONTRACT.SCENE_INPUT_BYTE_INVENTORY_BINDING,
        states=STATE_COUNT,
    )

    scene_records = []
    mesh_contract = bindings["box_obj_cache"]
    structural_objects_described = 0
    structural_builder_keys_present = 0
    texture_selection_records = 0
    for (
        state,
        state_id,
        scene_id,
        scene_dir,
        manifest_path,
        genesis_path,
    ) in scene_descriptors:
        manifest_payload = load_json(manifest_path)
        genesis_payload = load_json(genesis_path)
        declared = str(manifest_payload.get("manifest_sha256"))
        parsed = parse_scene_manifest_dict(manifest_payload)
        recomputed = manifest_sha256(parsed)
        genesis_declared = str(genesis_payload.get("manifest_sha256"))
        identity_ok = (
            parsed.scene_id == scene_id
            and str(genesis_payload.get("scene_id")) == scene_id
            and parsed.family == str(state["family"])
            and str(genesis_payload.get("family")) == str(state["family"])
            and declared == recomputed == genesis_declared
        )
        if not identity_ok:
            raise QualificationError(f"CPU inventory scene identity/digest drift: {state_id}")
        # Frozen compatibility is intentionally the historical renderer bug:
        # the caller supplies ``genesis_scene.json`` (geometry under
        # ``objects``), while build_scene consumes only the absent top-level
        # walls/obstacles/landmarks lists.  Prove that mismatch for every
        # scene.  Do not reinterpret ``objects`` or touch the OBJ cache.
        builder_keys = ("walls", "obstacles", "landmarks")
        present_builder_keys = [key for key in builder_keys if key in genesis_payload]
        if present_builder_keys or not isinstance(genesis_payload.get("objects"), list):
            raise QualificationError(
                f"historical floor-only renderer input structure drift: {state_id}"
            )
        structural_objects_described += len(genesis_payload["objects"])
        structural_builder_keys_present += len(present_builder_keys)
        selected = select_scene_textures(
            visual_seed=int(genesis_payload.get("visual_seed") or 0),
            scene_id=scene_id,
        )
        if any(
            selected.get(category) is None
            or str(Path(str(selected[category])).resolve()) not in texture_paths
            for category in ("floor", "wall", "obstacle")
        ):
            raise QualificationError(f"CPU inventory selected an unfrozen texture: {state_id}")
        selected = {key: str(Path(str(value)).resolve()) for key, value in selected.items()}
        texture_selection_records += 1
        scene_records.append(
            {
                "state_id": state_id,
                "scene_id": scene_id,
                "scene_dir": str(scene_dir),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "manifest_bytes": manifest_path.stat().st_size,
                "declared_manifest_sha256": declared,
                "recomputed_manifest_sha256": recomputed,
                "genesis_scene_path": str(genesis_path),
                "genesis_scene_sha256": sha256_file(genesis_path),
                "genesis_scene_bytes": genesis_path.stat().st_size,
                "genesis_scene_declared_manifest_sha256": genesis_declared,
                "identity_and_internal_digest_consistent": True,
                "genesis_scene_top_level_keys": sorted(genesis_payload),
                "historical_builder_structural_keys_present": [],
                "genesis_scene_object_records": len(genesis_payload["objects"]),
                "historical_floor_only_renderer": True,
                "texture_selection_by_category": selected,
                "texture_categories_reached_by_renderer": ["floor"],
            }
        )
    scene_records.sort(key=lambda row: int(str(row["state_id"]).split("-")[-1]))
    if structural_builder_keys_present != 0:
        raise QualificationError("historical renderer unexpectedly sees structural keys")

    return attach_digest(
        {
            **_phase_core(
                "jepa_local_waypoint_cpu_runtime_input_inventory_v1",
                source_freeze_commit,
            ),
            "platform_manifest": platform,
            "primitive_registry": primitive_registry,
            "policy_artifacts": policy_artifacts,
            "genesis_builtin_urdf": urdf,
            "prefreeze_scene_byte_inventory_binding": copy.deepcopy(
                CONTRACT.SCENE_INPUT_BYTE_INVENTORY_BINDING
            ),
            "prefreeze_scene_byte_inventory_records": prefreeze_scene_records,
            "prefreeze_scene_byte_inventory_validation": (
                prefreeze_scene_validation
            ),
            "scene_records": scene_records,
            "texture_root": {
                "path": str(texture_root),
                "environment_variable": "LEWM_TEXTURE_ROOT",
                "environment_value": None,
                "override_absent": True,
            },
            "texture_records": texture_records,
            "box_obj_cache": {
                **copy.deepcopy(mesh_contract),
                "historical_scene_derived_nonexecuted_records": int(
                    mesh_contract["historical_scene_derived_nonexecuted_records"]
                ),
                "regenerated_during_scientific_materialisation": False,
            },
            "package_roots": package_roots,
            "foundational_package_roots": foundational_package_roots,
            "foundational_package_closure_policy": copy.deepcopy(
                bindings["foundational_package_closure_policy"]
            ),
            "package_record_closure_algorithm": copy.deepcopy(
                bindings["package_record_closure_algorithm"]
            ),
            "counts": {
                "states": len(states),
                "scene_records": len(scene_records),
                "manifest_files": len(scene_records),
                "genesis_scene_files": len(scene_records),
                "textures": len(texture_records),
                "genesis_referenced_meshes": len(referenced_meshes),
            },
            "historical_renderer_limitations": copy.deepcopy(
                CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
            ),
            "renderer_structure_schema_audit": {
                "scene_records": len(scene_records),
                "genesis_scene_records_with_objects": len(scene_records),
                "genesis_scene_records_with_nonempty_objects": sum(
                    int(row["genesis_scene_object_records"] > 0)
                    for row in scene_records
                ),
                "total_genesis_scene_objects": structural_objects_described,
                "genesis_scene_records_with_walls": 0,
                "genesis_scene_records_with_obstacles": 0,
                "genesis_scene_records_with_landmarks": 0,
                "effective_scene_geometry": "FLOOR_PLANE_ONLY",
            },
            "texture_selection_validation": {
                "scene_records": texture_selection_records,
                "selected_paths_in_frozen_12_file_inventory": True,
                "reached_categories": ["floor"],
                "nonexecuted_categories": ["wall", "obstacle"],
                "pass": texture_selection_records == STATE_COUNT,
            },
            "historical_renderer_limitation_validation": {
                **copy.deepcopy(CONTRACT.HISTORICAL_RENDERER_LIMITATIONS),
                "scenes_checked": len(scene_records),
                "structural_objects_described_but_not_rendered": (
                    structural_objects_described
                ),
                "builder_structural_keys_present": structural_builder_keys_present,
                "box_obj_files_opened_or_used": 0,
                "pass": True,
            },
            "outcome_fields_read": [],
            "pass": True,
        }
    )


def validate_cpu_runtime_input_inventory(
    source_freeze_commit: str,
    *,
    output_root: Path,
    persist_if_absent: bool,
) -> dict[str, Any]:
    regenerated = build_cpu_runtime_input_inventory(source_freeze_commit)
    path = output_root / CPU_RUNTIME_INPUT_INVENTORY_REL
    if not path.exists():
        if not persist_if_absent:
            raise QualificationError("CPU runtime input inventory is absent")
        atomic_json(path, regenerated)
    persisted = load_json(path)
    validate_phase_binding(persisted, source_freeze_commit)
    _validate_schema_value("cpu_runtime_input_inventory", persisted)
    if canonical_json_bytes(persisted) != canonical_json_bytes(regenerated):
        raise QualificationError("CPU runtime input inventory regeneration drift")
    return persisted


def filesystem_receipt(path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    result = subprocess.run(
        ["df", "-PT", str(path)], check=True, text=True, capture_output=True
    )
    lines = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 2 or len(lines[1]) < 7:
        raise QualificationError(f"unexpected df output for {path}")
    row = lines[1]
    return {
        "path": str(path.resolve()),
        "device": row[0],
        "filesystem_type": row[1],
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
    }


def prohibition_counters() -> dict[str, int]:
    return {identifier: 0 for identifier in CONTRACT.PROHIBITION_COUNTER_IDS}


def _successful_execution_watchdog_status(
    fanout: Mapping[str, Any], gpu: Mapping[str, Any]
) -> dict[str, Any]:
    if fanout.get("cpu_watchdog_status") != (
        CONTRACT.CPU_WATCHDOG_STATUS_SUCCESS
    ) or gpu.get("gpu_watchdog_status") != CONTRACT.GPU_WATCHDOG_STATUS_SUCCESS:
        raise QualificationError("successful execution watchdog custody drift")
    return copy.deepcopy(CONTRACT.EXECUTION_WATCHDOG_STATUS_SUCCESS)


def _phase_core(schema: str, source_freeze_commit: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "experiment_id": CONTRACT.EXPERIMENT_ID,
        "source_freeze_commit": source_freeze_commit,
        "contract_sha256": CONTRACT.CONTRACT_SHA256,
        "output_schema_sha256": CONTRACT.OUTPUT_SCHEMA_SHA256,
    }


def validate_phase_binding(
    value: Mapping[str, Any], source_freeze_commit: str, *, digest_key: str = "content_digest"
) -> None:
    validate_digest(value, digest_key)
    if value.get("experiment_id") != CONTRACT.EXPERIMENT_ID:
        raise QualificationError("phase experiment identity mismatch")
    if value.get("source_freeze_commit") != source_freeze_commit:
        raise QualificationError("phase source-freeze commit mismatch")
    if value.get("contract_sha256") != CONTRACT.CONTRACT_SHA256:
        raise QualificationError("phase contract digest mismatch")
    if value.get("output_schema_sha256") != CONTRACT.OUTPUT_SCHEMA_SHA256:
        raise QualificationError("phase output-schema digest mismatch")


def freeze(repo_root: Path = ROOT) -> dict[str, Any]:
    """Write only prospective, payload-free tracked receipts."""

    if repo_root.resolve() != ROOT.resolve():
        raise QualificationError("freeze repository root differs from entrypoint root")
    CONTRACT.write_contract(ROOT / CONTRACT.TRACKED_CONTRACT_RECEIPT_PATH)
    CONTRACT.write_output_schema(ROOT / CONTRACT.TRACKED_OUTPUT_SCHEMA_PATH)
    CONTRACT.write_fixture_receipt(ROOT / CONTRACT.TRACKED_FIXTURE_PATH)
    closure = CONTRACT.build_source_closure(ROOT, require_complete=True)
    CONTRACT.write_source_closure(closure, ROOT / CONTRACT.TRACKED_SOURCE_CLOSURE_PATH)
    frozen = validate_frozen_receipts()
    return {
        "experiment_id": CONTRACT.EXPERIMENT_ID,
        "contract_sha256": CONTRACT.CONTRACT_SHA256,
        "output_schema_sha256": CONTRACT.OUTPUT_SCHEMA_SHA256,
        "fixture_content_digest": frozen["fixture"]["content_digest"],
        "source_closure_content_digest": frozen["source_closure"]["content_digest"],
        "outcome_rows_read": 0,
        "checkpoint_tensors_opened": 0,
        "predictor_inference_calls": 0,
    }


def _run_gpu_child(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    if not arguments:
        raise QualificationError("GPU child command is absent")
    timeout_by_phase = {
        "preflight": int(CONTRACT.EXECUTION_WATCHDOGS["gpu_preflight_timeout_s"]),
        "materialize": int(
            CONTRACT.EXECUTION_WATCHDOGS["gpu_materialization_timeout_s"]
        ),
        "check": int(CONTRACT.EXECUTION_WATCHDOGS["gpu_check_timeout_s"]),
    }
    try:
        timeout_s = timeout_by_phase[str(arguments[0])]
    except KeyError as exc:
        raise QualificationError(f"unknown GPU child phase: {arguments[0]!r}") from exc
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    try:
        return subprocess.run(
            [str(GPU_INTERPRETER), str(GPU_ENTRYPOINT), *arguments],
            cwd=ROOT,
            env=environment,
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise QualificationError(
            f"GPU {arguments[0]} watchdog exceeded {timeout_s} s"
        ) from exc


def preflight(
    source_freeze_commit: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    canonical_output_root: Path | None = None,
    archive_on_failure: bool = True,
) -> dict[str, Any]:
    """Validate frozen closure, imports, hashes, storage, and process separation."""

    started = time.time()
    validate_source_freeze_commit(source_freeze_commit, require_live_head=True)
    canonical_root = OUTPUT_ROOT if canonical_output_root is None else canonical_output_root
    if canonical_root.exists():
        raise QualificationError("canonical output root is not fresh/absent")
    if output_root.resolve() == canonical_root.resolve():
        raise QualificationError(
            "direct canonical preflight is forbidden; use hidden-attempt execute"
        )
    if git_output("status", "--porcelain"):
        raise QualificationError("preflight requires a clean source-freeze worktree")
    cpu_entrypoint = Path(sys.executable).absolute()
    cpu_interpreter_binding = _validate_cpu_interpreter_entrypoint()
    frozen = validate_frozen_receipts()
    inputs = validate_input_hashes()
    if output_root.exists() and any(output_root.iterdir()):
        existing = output_root / PREEXEC_REL
        if not existing.is_file():
            raise QualificationError("canonical output root is nonempty without preexecution")
        value = load_json(existing)
        validate_phase_binding(value, source_freeze_commit)
        if value.get("pass") is not True:
            raise QualificationError("existing preexecution receipt did not pass")
        return value

    workspace = filesystem_receipt(ROOT)
    output_parent = filesystem_receipt(output_root.parent)
    if workspace["free_bytes"] < 20 * 10**9:
        raise QualificationError("workspace has less than 20 GB free")
    if output_parent["free_bytes"] < 50 * 10**9:
        raise QualificationError("output filesystem has less than 50 GB free")
    if output_parent["filesystem_type"] != "ext4":
        raise QualificationError("output filesystem is not the frozen ext4 target")
    estimated_final = 9_500_000_000
    estimated_temporary = 11_500_000_000
    if estimated_final > 12 * 10**9 or estimated_temporary > 20 * 10**9:
        raise QualificationError("prospective storage estimate exceeds frozen ceiling")

    output_root.mkdir(parents=True, exist_ok=False)
    gpu_receipt = output_root / GPU_ENVIRONMENT_REL
    try:
        cpu_runtime_inventory = validate_cpu_runtime_input_inventory(
            source_freeze_commit,
            output_root=output_root,
            persist_if_absent=True,
        )
        cpu_runtime_binding = _binding(
            CPU_RUNTIME_INPUT_INVENTORY_REL, output_root
        )
        inputs["cpu_runtime_input_inventory"] = cpu_runtime_binding
        _run_gpu_child(
            [
                "preflight",
                "--source-freeze-commit",
                source_freeze_commit,
                "--output-root",
                str(output_root),
                "--receipt",
                str(gpu_receipt),
            ]
        )
        gpu = load_json(gpu_receipt)
        validate_phase_binding(gpu, source_freeze_commit)
        _validate_schema_value("gpu_environment_receipt", gpu)
        if gpu.get("checkpoint_tensor_open_count") != 0:
            raise QualificationError("GPU preflight opened checkpoint tensors")
        import genesis as gs
        import scipy
        import torch
        import yaml
        from PIL import __version__ as pillow_version

        cpu_expected = CONTRACT.build_contract()["execution"][
            "environments"
        ]["cpu_replay_render_oracle"]
        cpu_environment = {
            "python": sys.version.split()[0],
            "executable": str(cpu_entrypoint),
            "interpreter": cpu_interpreter_binding,
            "genesis": str(gs.__version__),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "pillow": pillow_version,
            "pyyaml": yaml.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_count": os.cpu_count(),
            "inner_threads": 1,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "TI_NUM_THREADS": "1",
            },
            "genesis_import_smoke_only_pre_materialization": True,
            "checkpoint_tensor_open_count": 0,
            "foundational_package_roots": copy.deepcopy(
                cpu_runtime_inventory["foundational_package_roots"]
            ),
            "foundational_package_closure_policy": copy.deepcopy(
                CONTRACT.FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
            ),
        }
        observed_cpu = {
            key: cpu_environment[key]
            for key in (
                "python",
                "genesis",
                "numpy",
                "scipy",
                "torch",
                "pillow",
                "pyyaml",
                "cuda_available",
            )
        }
        expected_cpu = {
            key: cpu_expected[key]
            for key in (
                "python",
                "genesis",
                "numpy",
                "scipy",
                "torch",
                "pillow",
                "pyyaml",
                "cuda_available",
            )
        }
        if observed_cpu != expected_cpu:
            raise QualificationError(
                f"CPU environment drift: {observed_cpu!r} != {expected_cpu!r}"
            )
        if _validated_interpreter_binary_binding(cpu_entrypoint) != (
            cpu_interpreter_binding
        ):
            raise QualificationError("CPU interpreter binary changed during preflight")
        if os.cpu_count() != 32:
            raise QualificationError("frozen execution requires exactly os.cpu_count()==32")
        validate_cpu_runtime_input_inventory(
            source_freeze_commit,
            output_root=output_root,
            persist_if_absent=False,
        )
        environment = attach_digest(
            {
                **_phase_core(
                    "jepa_local_waypoint_planning_cost_environment_v1",
                    source_freeze_commit,
                ),
                "python": cpu_environment["python"],
                "torch": gpu.get("torch"),
                "device": gpu.get("device"),
                "package_inventory": {"cpu": cpu_environment, "gpu": gpu.get("packages")},
                "import_closure": {
                    "cpu": [
                        "numpy",
                        "scipy.stats",
                        "PIL (materialization worker)",
                        "yaml",
                        "genesis (materialization worker only)",
                        "lewm.safety.jepa_local_waypoint_planning_cost_metrics_v1",
                        "lewm.safety.jepa_local_waypoint_planning_cost_qualification_v1_contract",
                    ],
                    "gpu": gpu.get("import_closure"),
                    "genesis_in_gpu_process": False,
                },
                "interpreter_bindings": {
                    "cpu": cpu_environment["interpreter"],
                    "gpu": gpu["interpreter_binding"],
                },
                "process_separation": {
                    "genesis_cpu_import_smoke": True,
                    "genesis_imported_in_gpu": gpu.get("genesis_imported"),
                    "gpu_inference_in_cpu_process": False,
                    "pass": gpu.get("genesis_imported") is False,
                },
                "source_hashes": {
                    row["path"]: row["sha256"] for row in frozen["source_closure"]["rows"]
                },
                "checkpoint_file_hashes": gpu.get("checkpoint_file_hashes"),
                "checkpoint_tensor_open_count": 0,
                "foundational_package_closure_policy": copy.deepcopy(
                    CONTRACT.FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
                ),
                "gpu_environment_receipt_binding": _binding(
                    GPU_ENVIRONMENT_REL, output_root
                ),
                "cpu_runtime_input_inventory_binding": cpu_runtime_binding,
                "historical_renderer_limitations": copy.deepcopy(
                    CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
                ),
                "smoke": {"cpu": True, "gpu": gpu.get("pass") is True},
            }
        )
        atomic_json(output_root / ENVIRONMENT_REL, environment)
        receipt = attach_digest(
            {
                **_phase_core(
                    "jepa_local_waypoint_planning_cost_preexecution_v1",
                    source_freeze_commit,
                ),
                "head": source_freeze_commit,
                "contract": _receipt_binding(ROOT / CONTRACT.TRACKED_CONTRACT_RECEIPT_PATH),
                "output_schema": _receipt_binding(ROOT / CONTRACT.TRACKED_OUTPUT_SCHEMA_PATH),
                "fixture": _receipt_binding(
                    ROOT / CONTRACT.TRACKED_FIXTURE_PATH, digest_key="content_digest"
                ),
                "source_closure": _receipt_binding(
                    ROOT / CONTRACT.TRACKED_SOURCE_CLOSURE_PATH,
                    digest_key="content_digest",
                ),
                "preexecution_custody": {
                    "contract_disclosure": copy.deepcopy(
                        CONTRACT.build_contract()["preexecution_custody"]
                    ),
                    "live_validation": {
                        "outcome_barrier_lifted_only_after_freeze_commit": True,
                        "outcome_rows_read": 0,
                        "checkpoint_tensors_opened": 0,
                        "predictor_inference_calls": 0,
                        "fixture_checks_passed": len(
                            frozen["fixture"]["executed_checks"]
                        ),
                        "fixture_pass": frozen["fixture"]["pass"],
                    },
                },
                "panel_bindings": inputs,
                "cpu_runtime_input_inventory_binding": cpu_runtime_binding,
                "checkpoint_file_bindings_without_tensor_open": gpu.get(
                    "checkpoint_file_hashes"
                ),
                "environment": {
                    "path": str(ENVIRONMENT_REL),
                    "sha256": sha256_file(output_root / ENVIRONMENT_REL),
                    "content_digest": environment["content_digest"],
                },
                "storage": {
                    "workspace": workspace,
                    "output": output_parent,
                    "estimated_final_bytes": estimated_final,
                    "estimated_temporary_bytes": estimated_temporary,
                    "final_ceiling_bytes": 12 * 10**9,
                    "temporary_ceiling_bytes": 20 * 10**9,
                },
                "canonical_output_fresh": True,
                "canonical_output_root": str(canonical_root),
                "hidden_attempt_root": str(output_root),
                "execution_watchdog_config": copy.deepcopy(
                    CONTRACT.EXECUTION_WATCHDOGS
                ),
                "prohibition_counters": prohibition_counters(),
                "runtime_s": time.time() - started,
                "pass": True,
            }
        )
        atomic_json(output_root / PREEXEC_REL, receipt)
        return receipt
    except BaseException:
        # A failed preflight has no scientific materialisation and is safe to
        # archive whole.  It is never silently reused as canonical evidence.
        if archive_on_failure and output_root.exists():
            archive = output_root.with_name(
                f"{output_root.name}.failed-preflight-{int(time.time())}-{os.getpid()}"
            )
            os.replace(output_root, archive)
        raise


def _state_role_map(split: Mapping[str, Any]) -> dict[str, str]:
    aliases = {
        "fit": "fit",
        "training": "fit",
        "train": "fit",
        "calibration": "calibration",
        "internal_calibration": "calibration",
        "heldout": "heldout",
        "development_heldout": "heldout",
    }
    output: dict[str, str] = {}
    for key, value in split.items():
        if key not in aliases or not isinstance(value, list):
            continue
        for state_id in value:
            sid = str(state_id)
            if sid in output:
                raise QualificationError(f"state appears in multiple roles: {sid}")
            output[sid] = aliases[key]
    if len(output) != STATE_COUNT:
        raise QualificationError(f"split resolves {len(output)} states, expected 48")
    return output


def _control_history_from_replay_blocks(
    executed_by_block: Mapping[int, Any],
) -> tuple[np.ndarray, list[int]]:
    """Return applied[k-1] block37-tick5 through block40-tick4 exactly."""

    try:
        command_rows = np.concatenate(
            [np.asarray(executed_by_block[index], np.float32) for index in (37, 38, 39, 40)],
            axis=0,
        )
    except KeyError as exc:
        raise QualificationError("control history lacks replay block 37..40") from exc
    if command_rows.shape != (20, 3):
        raise QualificationError("warmup control trace shape mismatch")
    values = command_rows[4:19]
    indices = list(range(184, 199))
    if values.shape != (15, 3) or len(indices) != 15:
        raise QualificationError("contiguous control history derivation failed")
    return values, indices


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return value.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(value)


def _robot_pose(ctx: Any) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(_as_numpy(ctx.build.robot.get_pos()), dtype=np.float64)
    quaternion = np.asarray(_as_numpy(ctx.build.robot.get_quat()), dtype=np.float64)
    if position.ndim > 1:
        position = position[0]
    if quaternion.ndim > 1:
        quaternion = quaternion[0]
    if position.shape != (3,) or quaternion.shape != (4,):
        raise QualificationError("unexpected robot base pose shape")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise QualificationError("robot base pose is non-finite")
    return position, quaternion


def _quat_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _atomic_png(path: Path, image: np.ndarray) -> str:
    from PIL import Image

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp-{os.getpid()}.png")
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(temporary)
    os.replace(temporary, target)
    return sha256_file(target)


def _contact_links_at_step(ctx: Any, topology: Mapping[str, Any]) -> list[str]:
    """Return sorted disallowed robot-link names at the current physics step."""

    from lewm.safety.contact_hazard_ontology_v1 import is_disallowed_contact

    robot = ctx.build.robot
    contacts = robot.get_contacts(exclude_self_contact=False)
    if not contacts:
        return []
    arrays = {key: _as_numpy(value) for key, value in contacts.items()}
    link_a = np.asarray(arrays.get("link_a", ()), dtype=np.int64).reshape(-1)
    link_b = np.asarray(arrays.get("link_b", ()), dtype=np.int64).reshape(-1)
    if link_a.shape != link_b.shape:
        raise QualificationError("contact link arrays have different shapes")
    force_a = arrays.get("force_a")
    magnitudes = None
    if force_a is not None and np.asarray(force_a).size:
        magnitudes = np.linalg.norm(np.asarray(force_a).reshape(-1, 3), axis=-1)
    low, high = topology["robot_link_range"]
    names = {int(link.idx): str(link.name) for link in robot.links}
    found: set[str] = set()
    for index, (left, right) in enumerate(zip(link_a, link_b, strict=True)):
        a, b = int(left), int(right)
        a_robot, b_robot = low <= a < high, low <= b < high
        if a_robot == b_robot:
            continue
        robot_link = a if a_robot else b
        environment_link = b if a_robot else a
        magnitude = None if magnitudes is None else float(magnitudes[index])
        if is_disallowed_contact(
            robot_link_id=robot_link,
            environment_link_id=environment_link,
            foot_link_ids=topology["foot_link_indices"],
            ground_link_ids=topology["ground_link_indices"],
            self_contact=False,
            force_magnitude_n=magnitude,
        ):
            found.add(names.get(robot_link, f"robot-link-{robot_link}"))
    return sorted(found)


def _execute_contact_block(
    ctx: Any, requested: np.ndarray, topology: Mapping[str, Any]
) -> tuple[Any, np.ndarray, list[list[str]]]:
    """Execute one 500 ms block and sample disallowed contact at all 250 steps."""

    runner = ctx.runner
    contact_links: list[list[str]] = []
    original = runner._step_policy_step

    def instrumented(_runner: Any, target_cmd: np.ndarray) -> None:
        observation = _runner._build_observation(target_cmd)
        joint_targets = _runner.policy.act(observation)
        _runner._apply_joint_targets(joint_targets)
        for _ in range(int(_runner._physics_steps_per_policy)):
            _runner.build.scene.step()
            contact_links.append(_contact_links_at_step(ctx, topology))
        _runner._sim_time_ns += _runner._policy_dt_ns

    runner._step_policy_step = types.MethodType(instrumented, runner)
    try:
        block = runner.execute_requested_block(np.asarray(requested, dtype=np.float32))
    finally:
        runner._step_policy_step = original
    if len(contact_links) != PHYSICS_STEPS_PER_BLOCK:
        raise QualificationError(
            f"contact block yielded {len(contact_links)} physics frames, expected 250"
        )
    executed = np.asarray(block.executed, dtype=np.float32)
    # Mirror the production BranchContext block-boundary counters exactly.
    # Reset checks are intentionally not invoked inside unconditional fanout:
    # every current and successor block remains the raw committed transition.
    for _ in range(int(runner._block_size)):
        for episode_state in runner.episode_states:
            episode_state.step()
    runner._blocks_in_episode += 1
    ctx.ticks_executed += runner._block_size
    ctx.episode_ticks += runner._block_size
    ctx.policy_steps += runner._block_size * runner._policy_steps_per_command_tick
    ctx.last_block_executed = executed.copy()
    return block, executed, contact_links


def _raw_continuation_boundary(ctx: Any, v1: Any) -> dict[str, Any]:
    """Validate a post-H1 block boundary without requiring nontermination."""

    runner, policy = ctx.runner, ctx.policy
    failures: list[str] = []
    if int(runner.n_envs) != 1:
        failures.append("raw continuation requires one simulator environment")
    block_size = int(runner._block_size)
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    command_phase = int(ctx.ticks_executed) % block_size
    decimation_phase = int(ctx.policy_steps) % steps_per_tick
    if command_phase != 0:
        failures.append(f"command-block phase {command_phase} is not zero")
    if decimation_phase != 0:
        failures.append(f"policy decimation phase {decimation_phase} is not zero")
    expected_ns = int(ctx.policy_steps) * int(runner._policy_dt_ns)
    if int(runner._sim_time_ns) != expected_ns:
        failures.append("simulation clock disagrees with policy-step counter")
    emission_phase = int(runner._sim_time_ns) % int(runner._command_dt_ns)
    if emission_phase != 0:
        failures.append("observation-emission phase is not zero")
    if bool(ctx.reset_in_last_block):
        failures.append("a production reset fired before raw continuation capture")
    if int(runner.episode_states[0].reset_count) != int(
        ctx.episode_start_reset_count
    ):
        failures.append("episode reset count moved before raw continuation capture")
    episode_step = int(runner.episode_states[0].episode_step)
    if episode_step != int(ctx.episode_ticks):
        failures.append("episode-state and BranchContext tick counters disagree")
    source_step = int(ctx.episode_ticks) + 1
    if (source_step - 1) % int(v1.TICKS) != 0:
        failures.append("raw continuation is not on a command-block boundary")
    last_actions = getattr(policy, "_last_actions", None)
    if last_actions is None:
        failures.append("policy last-action state is absent")
    else:
        action_array = np.asarray(last_actions)
        if action_array.shape != (1, len(policy.policy_joint_names)) or not np.all(
            np.isfinite(action_array)
        ):
            failures.append("policy last-action state is invalid")
    last_executed = np.asarray(runner._last_executed, dtype=np.float64)
    if last_executed.shape != (1, 3) or not np.all(np.isfinite(last_executed)):
        failures.append("runner last-executed command state is invalid")
    else:
        vx, vy, yaw_rate = (float(value) for value in last_executed[0])
        limits = runner.safety
        if not (
            limits.min_vx_mps - 1e-6 <= vx <= limits.max_vx_mps + 1e-6
            and limits.min_vy_mps - 1e-6 <= vy <= limits.max_vy_mps + 1e-6
            and abs(yaw_rate) <= limits.max_yaw_rate_radps + 1e-6
        ):
            failures.append("runner last-executed command is outside frozen limits")
    last_block = None if ctx.last_block_executed is None else np.asarray(
        ctx.last_block_executed, dtype=np.float64
    )
    if last_block is None or last_block.shape != (1, block_size, 3):
        failures.append("raw continuation lacks one complete executed block")
    elif last_executed.shape == (1, 3) and not np.array_equal(
        last_block[0, -1], last_executed[0]
    ):
        failures.append("executed-block tail disagrees with runner command state")
    terminal_flags = {
        str(key): bool(value) for key, value in v1._termination_flags(ctx).items()
    }
    if terminal_flags.get("nan", False):
        failures.append("NaN state cannot seed successor continuation")
    if failures:
        raise QualificationError("raw continuation boundary refused: " + "; ".join(failures))
    tipped = int(np.asarray(runner._consecutive_tipped_blocks).reshape(-1)[0])
    blocks = int(np.asarray(runner._blocks_in_episode).reshape(-1)[0])
    return {
        "capture_mode": "RAW_CONTINUATION_AFTER_CURRENT_H1",
        "evaluation_only": True,
        "production_reset_checks_suppressed": True,
        "command_block_tick": command_phase,
        "decimation_phase": decimation_phase,
        "observation_emission_phase_ns": emission_phase,
        "source_step": source_step,
        "episode_step": episode_step,
        "sim_time_ns": int(runner._sim_time_ns),
        "blocks_in_episode": blocks,
        "consecutive_tipped_blocks": tipped,
        "terminal_flags": terminal_flags,
    }


def _capture_raw_continuation_state(
    ctx: Any, v1: Any, *, goal: Mapping[str, Any], identity: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Use the exact V1 snapshot machinery with a declared raw boundary."""

    boundary = _raw_continuation_boundary(ctx, v1)
    original = v1.assert_canonical_boundary

    def raw_assert(observed: Any) -> dict[str, Any]:
        if observed is not ctx:
            raise QualificationError("raw continuation captured a different context")
        regenerated = _raw_continuation_boundary(observed, v1)
        if regenerated != boundary:
            raise QualificationError("raw continuation boundary changed during capture")
        return regenerated

    v1.assert_canonical_boundary = raw_assert
    try:
        snapshot = v1.capture_branch_state(
            ctx,
            goal=dict(goal),
            identity=dict(identity),
        )
    finally:
        v1.assert_canonical_boundary = original
    if snapshot.boundary != boundary or not isinstance(snapshot.digest, str) or len(
        snapshot.digest
    ) != 64:
        raise QualificationError("raw continuation snapshot custody is invalid")
    return snapshot, boundary


def _restore_raw_continuation_state(ctx: Any, v1: Any, snapshot: Any) -> None:
    """Restore an evaluation-only raw continuation without eligibility checks."""

    original = v1.assert_canonical_boundary

    def raw_assert(observed: Any) -> dict[str, Any]:
        regenerated = _raw_continuation_boundary(observed, v1)
        if regenerated != snapshot.boundary:
            raise QualificationError("raw continuation restore changed boundary state")
        return regenerated

    v1.assert_canonical_boundary = raw_assert
    try:
        v1.restore_branch_state(ctx, snapshot)
    finally:
        v1.assert_canonical_boundary = original
    if _raw_continuation_boundary(ctx, v1) != snapshot.boundary:
        raise QualificationError("raw continuation post-restore validation drift")


def _first_contact(contact_links: Sequence[Sequence[str]]) -> tuple[int | None, str | None]:
    for index, links in enumerate(contact_links):
        if links:
            return index + 1, sorted(str(value) for value in links)[0]
    return None, None


def _save_fanout_shard(
    path: Path,
    current: np.ndarray,
    successor: np.ndarray,
    *,
    output_root: Path,
) -> dict[str, Any]:
    if current.shape != (9, 250) or successor.shape != (9, 9, 250):
        raise QualificationError("fanout bitset shape mismatch")
    if current.dtype != np.bool_ or successor.dtype != np.bool_:
        raise QualificationError("fanout bitsets must be Boolean")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                current_contact_bitset=current,
                successor_contact_bitset=successor,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "path": _artifact_reference(target, output_root),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def _materialize_state(
    state_index: int,
    *,
    output_root: Path,
    source_freeze_commit: str,
) -> dict[str, Any]:
    """Real Genesis replay/render and unconditional 9x9 one-block fanout."""

    _validate_cpu_interpreter_entrypoint()
    # All imports that can initialize Genesis are deliberately below the
    # outcome barrier and inside the CPU worker.
    import genesis as gs
    from lewm.oracle.go2_textured_v03_renderer import BasePose, TexturedV03Renderer
    from scripts import run_go2_oracle_branch_pilot_v1_2 as V

    if not 0 <= state_index < STATE_COUNT:
        raise QualificationError("state index outside frozen 48-state panel")
    runtime_inventory = load_json(output_root / CPU_RUNTIME_INPUT_INVENTORY_REL)
    validate_phase_binding(runtime_inventory, source_freeze_commit)
    _validate_schema_value("cpu_runtime_input_inventory", runtime_inventory)
    runtime_inventory_binding = _binding(
        CPU_RUNTIME_INPUT_INVENTORY_REL, output_root
    )
    manifest = load_json(STATE_MANIFEST)
    states = manifest.get("state_candidates")
    if not isinstance(states, list) or len(states) != STATE_COUNT:
        raise QualificationError("frozen state manifest cardinality mismatch")
    state = states[state_index]
    state_id = str(state["state_id"])
    role = _state_role_map(load_json(SPLIT))[state_id]
    ledger = [row for row in load_jsonl(BRANCH_LEDGER) if str(row["state_id"]) == state_id]
    if len(ledger) != CANDIDATES_PER_STATE:
        raise QualificationError(f"{state_id}: branch-ledger cardinality mismatch")
    ledger.sort(key=lambda row: int(row["candidate_index"]))
    if [int(row["candidate_index"]) for row in ledger] != list(range(12)):
        raise QualificationError(f"{state_id}: candidate identities are not 0..11")
    snapshot_digests = {str(row["snapshot_digest"]) for row in ledger}
    if len(snapshot_digests) != 1:
        raise QualificationError(f"{state_id}: branch ledger lacks one snapshot authority")
    expected_snapshot = next(iter(snapshot_digests))

    shared = V.V1._load_shared("cpu")
    ctx = V.V1.build_context(
        Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu", shared=shared
    )
    ctx.begin_episode()
    start_sim_time_ns = int(ctx.runner._sim_time_ns)
    executed_by_block: dict[int, np.ndarray] = {}
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for block_index in range(1, 41):
        ctx.drive_one_block()
        executed = np.asarray(ctx.last_block_executed, dtype=np.float32)
        if executed.shape != (1, 5, 3):
            raise QualificationError(f"{state_id}: warmup block shape mismatch")
        executed_by_block[block_index] = executed[0].copy()
        if block_index in (38, 39, 40):
            poses[block_index] = _robot_pose(ctx)
    topology = V.link_topology(ctx)
    eligible = V.eligible_here(ctx, topology)
    if isinstance(eligible, str):
        raise QualificationError(f"{state_id}: frozen boundary is no longer eligible: {eligible}")
    goal_record, _field = eligible
    snapshot = V.V1.capture_branch_state(
        ctx,
        goal=dict(goal_record["goal"]),
        identity={
            "state_id": state_id,
            "scene_id": str(state["scene_id"]),
            "family": str(state["family"]),
        },
    )
    if snapshot.digest != expected_snapshot:
        raise QualificationError(
            f"{state_id}: replay snapshot {snapshot.digest} != branch authority {expected_snapshot}"
        )

    raw_manifest = load_json(Path(state["scene_dir"]) / "genesis_scene.json")
    if any(key in raw_manifest for key in ("walls", "obstacles", "landmarks")):
        raise QualificationError(
            f"{state_id}: frozen historical renderer input structure changed"
        )
    if not isinstance(raw_manifest.get("objects"), list):
        raise QualificationError(
            f"{state_id}: genesis_scene objects list is absent"
        )
    renderer = TexturedV03Renderer(ctx, gs=gs, raw_manifest=raw_manifest)
    state_root = output_root / "materialization/states" / state_id
    context_paths: list[str] = []
    context_hashes: list[str] = []
    for block_index in (38, 39, 40):
        position, quaternion = poses[block_index]
        render = renderer.render_pose(
            BasePose(tuple(float(v) for v in position), tuple(float(v) for v in quaternion))
        )
        path = state_root / "rgb" / f"context_block_{block_index:02d}.png"
        context_hashes.append(_atomic_png(path, render.image))
        context_paths.append(_artifact_reference(path, output_root))

    path_cells = [int(value) for value in state.get("waypoint_path_cells", [])]
    if len(path_cells) < 3:
        raise QualificationError(f"{state_id}: waypoint path has fewer than three cells")
    centres = [
        [float(v) for v in ctx.scene_graph.cell_center(cell)] for cell in path_cells[:3]
    ]
    blocked_cells = frozenset(
        int(value) for value in getattr(ctx.scene_graph, "nav_blocked_cells", ())
    )
    goal_cell = path_cells[2]
    goal_cell_free = goal_cell not in blocked_cells
    goal_cell_hops = ctx.scene_graph.bfs_distance(
        path_cells[0], goal_cell, transit_blocked=blocked_cells
    )
    goal_cell_reachable = goal_cell_hops is not None
    if not goal_cell_free or not goal_cell_reachable:
        raise QualificationError(
            f"{state_id}: frozen path[2] goal cell is not free and reachable"
        )
    waypoint_xy = centres[2]
    if state.get("waypoint_xy") is not None and list(state["waypoint_xy"]) != waypoint_xy:
        raise QualificationError(f"{state_id}: persisted waypoint_xy disagrees exactly")
    yaw = math.atan2(centres[1][1] - centres[0][1], centres[1][0] - centres[0][0])
    current_position, current_quaternion = poses[40]
    goal_position = (waypoint_xy[0], waypoint_xy[1], float(current_position[2]))
    goal_quaternion = _quat_wxyz_from_yaw(yaw)
    goal_render = renderer.render_pose(BasePose(goal_position, goal_quaternion))
    goal_path = state_root / "rgb/goal.png"
    goal_sha = _atomic_png(goal_path, goal_render.image)

    # Exact contiguous applied[k-1] history for the three observed contexts.
    raw_history_3d, command_indices = _control_history_from_replay_blocks(
        executed_by_block
    )
    raw_history = raw_history_3d[:, (0, 2)].reshape(3, 5, 2)
    stats_path = Path(
        CONTRACT.build_contract()["predictor_bindings"]["action_and_control_semantics"]
        ["normalisation_stats"]["path"]
    )
    stats_binding = CONTRACT.build_contract()["predictor_bindings"][
        "action_and_control_semantics"
    ]["normalisation_stats"]
    if sha256_file(stats_path) != stats_binding["file_sha256"]:
        raise QualificationError("frozen control-normalisation file SHA drift")
    stats = load_json(stats_path)
    if stats.get("sha256") != stats_binding["stats_sha256"]:
        raise QualificationError("frozen control-normalisation logical digest drift")
    mean = np.asarray(stats["control_mean"], dtype=np.float32)
    std = np.asarray(stats["control_std"], dtype=np.float32)
    if mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0):
        raise QualificationError("frozen control normalization statistics are invalid")
    normalized_history = (raw_history - mean) / std
    action_3x5x2: list[list[list[list[float]]]] = []
    action_3x10: list[list[list[float]]] = []
    requested_3x5x3: list[list[list[list[float]]]] = []
    applied_3x5x3: list[list[list[list[float]]]] = []
    requested_applied_rows: list[dict[str, Any]] = []
    for candidate_index, row in enumerate(ledger):
        requested_blocks = np.asarray(row["requested"][:3], dtype=np.float32)
        applied_blocks = np.asarray(row["post_slew"][:3], dtype=np.float32)
        if requested_blocks.shape != (3, 5, 3) or applied_blocks.shape != (
            3,
            5,
            3,
        ):
            raise QualificationError(f"{state_id}: post-slew action shape mismatch")
        _candidate_name, primitive_sequence = V.V1.CANDIDATE_BANK[candidate_index]
        if len(primitive_sequence) < 3:
            raise QualificationError(
                f"{state_id}:{candidate_index}: candidate has fewer than 3 blocks"
            )
        expected_requested = np.stack(
            [
                np.asarray(V.V1.block_for(primitive), dtype=np.float32)
                for primitive in primitive_sequence[:3]
            ],
            axis=0,
        )
        if expected_requested.shape != (3, 5, 3) or not np.array_equal(
            requested_blocks, expected_requested
        ):
            raise QualificationError(
                f"{state_id}:{candidate_index}: requested action tape drift"
            )
        active = applied_blocks[:, :, (0, 2)]
        action_3x5x2.append(active.astype(float).tolist())
        action_3x10.append(active.reshape(3, 10).astype(float).tolist())
        requested_3x5x3.append(requested_blocks.astype(float).tolist())
        applied_3x5x3.append(applied_blocks.astype(float).tolist())
        requested_applied_rows.append(
            {
                "candidate_index": candidate_index,
                "requested_matches_candidate_bank": True,
                "applied_matches_frozen_branch_ledger": True,
                "active_projection_matches_applied": True,
                "requested_and_applied_persisted_separately": True,
                "pass": True,
            }
        )

    primitives = list(
        CONTRACT.build_contract()["candidate_and_viability_semantics"]
        ["unique_first_block_primitives"]
    )
    if len(primitives) != PRIMITIVE_COUNT or len(set(primitives)) != PRIMITIVE_COUNT:
        raise QualificationError("frozen primitive identity contract is invalid")
    current_bitset = np.zeros((9, 250), dtype=np.bool_)
    successor_bitset = np.zeros((9, 9, 250), dtype=np.bool_)
    current_first_step: list[int | None] = []
    current_first_link: list[str | None] = []
    successor_first_step: list[list[int | None]] = []
    successor_first_link: list[list[str | None]] = []
    current_executed: list[Any] = []
    successor_executed: list[Any] = []
    raw_continuation_snapshots: list[dict[str, Any]] = []
    for current_index, primitive in enumerate(primitives):
        V.V1.restore_branch_state(ctx, snapshot)
        requested = V.V1.block_for(primitive)[None, ...]
        _block, executed, links = _execute_contact_block(ctx, requested, topology)
        current_bitset[current_index] = np.asarray([bool(value) for value in links])
        step, link = _first_contact(links)
        current_first_step.append(step)
        current_first_link.append(link)
        current_executed.append(executed[0].astype(float).tolist())
        successor_snapshot, raw_boundary = _capture_raw_continuation_state(
            ctx,
            V.V1,
            goal=dict(goal_record["goal"]),
            identity={"state_id": state_id, "current_primitive": primitive},
        )
        next_steps: list[int | None] = []
        next_links: list[str | None] = []
        next_executed: list[Any] = []
        for next_index, next_primitive in enumerate(primitives):
            _restore_raw_continuation_state(ctx, V.V1, successor_snapshot)
            requested_next = V.V1.block_for(next_primitive)[None, ...]
            _next_block, executed_next, links_next = _execute_contact_block(
                ctx, requested_next, topology
            )
            successor_bitset[current_index, next_index] = np.asarray(
                [bool(value) for value in links_next]
            )
            next_step, next_link = _first_contact(links_next)
            next_steps.append(next_step)
            next_links.append(next_link)
            next_executed.append(executed_next[0].astype(float).tolist())
        raw_continuation_snapshots.append(
            {
                "current_primitive_index": current_index,
                "current_primitive_id": primitive,
                "capture_mode": "RAW_CONTINUATION_AFTER_CURRENT_H1",
                "snapshot_digest": successor_snapshot.digest,
                "terminal_flags": raw_boundary["terminal_flags"],
                "consecutive_tipped_blocks": raw_boundary[
                    "consecutive_tipped_blocks"
                ],
                "production_reset_checks_suppressed": True,
                "evaluation_only": True,
                "restored_for_successors": len(next_executed),
            }
        )
        successor_first_step.append(next_steps)
        successor_first_link.append(next_links)
        successor_executed.append(next_executed)

    macro_to_primitive: dict[str, int] = {}
    for candidate_index, (_name, sequence) in enumerate(V.V1.CANDIDATE_BANK):
        primitive = str(sequence[0])
        if primitive not in primitives:
            raise QualificationError(f"unknown first primitive {primitive}")
        macro_to_primitive[str(candidate_index)] = primitives.index(primitive)
        ledger_first = np.asarray(ledger[candidate_index]["post_slew"][0], np.float32)
        if not np.array_equal(ledger_first, np.asarray(current_executed[primitives.index(primitive)], np.float32)):
            raise QualificationError(f"{state_id}:{candidate_index}: H1 executed action drift")
    safe_next = np.sum(~np.any(successor_bitset, axis=2), axis=1, dtype=np.int64)
    shard = _save_fanout_shard(
        state_root / "oracle_contact_fanout.npz",
        current_bitset,
        successor_bitset,
        output_root=output_root,
    )

    start_xy = np.asarray(current_position[:2], dtype=np.float64)
    # Use the frozen current yaw from the replayed start quaternion via ctx.pose
    # captured at the same boundary; after fanout restore before querying.
    V.V1.restore_branch_state(ctx, snapshot)
    _start_xy_pose, start_yaw, _start_z = ctx.pose()
    dx, dy = waypoint_xy[0] - float(start_xy[0]), waypoint_xy[1] - float(start_xy[1])
    body_dx = math.cos(start_yaw) * dx + math.sin(start_yaw) * dy
    body_dy = -math.sin(start_yaw) * dx + math.cos(start_yaw) * dy
    d_yaw = math.atan2(math.sin(yaw - start_yaw), math.cos(yaw - start_yaw))
    manifest_body_xy = state.get("waypoint_body_xy")
    body_xy = [body_dx, body_dy]
    if manifest_body_xy is not None and list(manifest_body_xy) != body_xy:
        raise QualificationError(
            f"{state_id}: persisted waypoint_body_xy disagrees exactly"
        )

    manifest_snapshot = state.get("snapshot_digest")
    command_dt_ns = int(ctx.runner._command_dt_ns)
    command_timestamps = [
        start_sim_time_ns + (index + 1) * command_dt_ns for index in command_indices
    ]
    record = attach_digest(
        {
            **_phase_core("jepa_local_waypoint_cpu_state_materialization_v1", source_freeze_commit),
            "state_index": state_index,
            "state_id": state_id,
            "family": str(state["family"]),
            "role": role,
            "cpu_runtime_input_inventory_binding": runtime_inventory_binding,
            "historical_renderer_limitations": copy.deepcopy(
                CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
            ),
            "branch_snapshot_digest": expected_snapshot,
            "replay_snapshot_digest": snapshot.digest,
            "manifest_snapshot_digest_or_null": manifest_snapshot,
            "manifest_snapshot_digest_match_or_null": (
                None if manifest_snapshot is None else str(manifest_snapshot) == snapshot.digest
            ),
            "context_rgb_paths": context_paths,
            "context_rgb_sha256s": context_hashes,
            "context_source_frame_indices": [-480, -240, 0],
            "context_command_tick_indices": [-10, -5, 0],
            "context_elapsed_s": [-1.0, -0.5, 0.0],
            "context_replay_boundaries": [38, 39, 40],
            "control_command_indices": command_indices,
            "control_timestamps": command_timestamps,
            "control_history_raw_3x5x2": raw_history.astype(float).tolist(),
            "control_history_normalized_3x5x2": normalized_history.astype(float).tolist(),
            "control_normalisation_binding": {
                "path": str(stats_path),
                "file_sha256": stats_binding["file_sha256"],
                "stats_sha256": stats_binding["stats_sha256"],
                "fields": list(stats_binding["fields"]),
            },
            "action_blocks_raw_3x5x2_by_candidate": action_3x5x2,
            "action_blocks_raw_3x10_by_candidate": action_3x10,
            "requested_action_blocks_raw_3x5x3_by_candidate": requested_3x5x3,
            "applied_action_blocks_raw_3x5x3_by_candidate": applied_3x5x3,
            "requested_applied_action_authority_validation": {
                "rows": requested_applied_rows,
                "candidates": 12,
                "requested_candidate_bank_exact": 12,
                "applied_branch_ledger_exact": 12,
                "active_projection_exact": 12,
                "requested_and_applied_never_conflated": True,
                "pass": True,
            },
            "goal": {
                "waypoint_world_xy": waypoint_xy,
                "goal_body_dx_dy_sin_dyaw_cos_dyaw": [
                    body_dx,
                    body_dy,
                    math.sin(d_yaw),
                    math.cos(d_yaw),
                ],
                "snapshot_base_z": float(current_position[2]),
                "waypoint_path_cells": path_cells,
                "path_cell_centers_world_xy": centres,
                "goal_cell_preconditions": {
                    "goal_cell": goal_cell,
                    "free": goal_cell_free,
                    "reachable_from_path_start": goal_cell_reachable,
                    "bfs_hops": int(goal_cell_hops),
                    "transit_blocked_cell_count": len(blocked_cells),
                    "pass": True,
                },
                "goal_pose_world_xyz_rpy": [
                    waypoint_xy[0],
                    waypoint_xy[1],
                    float(current_position[2]),
                    0.0,
                    0.0,
                    yaw,
                ],
                "manifest_waypoint_xy_or_null": state.get("waypoint_xy"),
                "manifest_waypoint_body_xy_or_null": state.get("waypoint_body_xy"),
                "manifest_field_equality_audit": {
                    "waypoint_xy": state.get("waypoint_xy") is None
                    or list(state["waypoint_xy"]) == waypoint_xy,
                    "waypoint_body_xy": manifest_body_xy is None
                    or list(manifest_body_xy) == body_xy,
                    "absence_allowed": True,
                },
                "rgb_path": _artifact_reference(goal_path, output_root),
                "rgb_sha256": goal_sha,
            },
            "fanout": {
                "shard_path": shard["path"],
                "shard_sha256": shard["sha256"],
                "shard_bytes": shard["bytes"],
                "array_manifest": {
                    "current_contact_bitset": {"shape": [9, 250], "dtype": "bool"},
                    "successor_contact_bitset": {"shape": [9, 9, 250], "dtype": "bool"},
                },
                "current_contact_bitset_shape": [9, 250],
                "successor_contact_bitset_shape": [9, 9, 250],
                "first_contact_step": {
                    "current": current_first_step,
                    "successor": successor_first_step,
                },
                "first_contact_link": {
                    "current": current_first_link,
                    "successor": successor_first_link,
                },
                "current_action_mapping": {
                    "primitive_ids": primitives,
                    "executed_blocks": current_executed,
                    "macro_candidate_to_primitive_index": macro_to_primitive,
                },
                "successor_action_mapping": {
                    "primitive_ids": primitives,
                    "executed_blocks_by_current_and_next": successor_executed,
                },
                "raw_continuation_snapshots": raw_continuation_snapshots,
                "safe_next_action_count": safe_next.astype(int).tolist(),
            },
            "pass": True,
        }
    )
    atomic_json(state_root / "state_receipt.json", record)
    return record


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("LEWM_TEXTURE_ROOT", None)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "TI_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _run_cpu_workers(
    source_freeze_commit: str, *, output_root: Path, workers: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if workers != os.cpu_count() or workers != 32:
        raise QualificationError("scientific materialization requires exactly all 32 CPU cores")
    pending = list(range(STATE_COUNT))
    active: dict[subprocess.Popen[bytes], tuple[int, Any, Path]] = {}
    started_monotonic = time.monotonic()
    last_completion_monotonic = started_monotonic
    completed_workers = 0
    no_progress_timeout_s = int(
        CONTRACT.EXECUTION_WATCHDOGS["cpu_worker_no_progress_timeout_s"]
    )
    global_timeout_s = int(
        CONTRACT.EXECUTION_WATCHDOGS["cpu_materialization_global_timeout_s"]
    )
    log_root = output_root / "materialization/logs"
    log_root.mkdir(parents=True, exist_ok=True)

    def stop_active_workers() -> tuple[int, int]:
        terminate_signals = 0
        kill_signals = 0
        for process, (_index, _handle, _path) in active.items():
            if process.poll() is None:
                process.terminate()
                terminate_signals += 1
        for process, (_index, _handle, _path) in active.items():
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    kill_signals += 1
                    process.wait(timeout=10)
        return terminate_signals, kill_signals

    try:
        while pending or active:
            while pending and len(active) < workers:
                index = pending.pop(0)
                log_path = log_root / f"state_{index:02d}.log"
                log_handle = log_path.open("wb")
                process = subprocess.Popen(
                    [
                        str(CPU_INTERPRETER),
                        str(Path(__file__).resolve()),
                        "cpu-state-worker",
                        "--state-index",
                        str(index),
                        "--source-freeze-commit",
                        source_freeze_commit,
                        "--output-root",
                        str(output_root),
                    ],
                    cwd=ROOT,
                    env=_worker_environment(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                active[process] = (index, log_handle, log_path)
            finished: list[subprocess.Popen[bytes]] = []
            for process, (index, log_handle, log_path) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log_handle.close()
                if status != 0:
                    tail = log_path.read_bytes()[-32_768:].decode("utf-8", errors="replace")
                    stop_active_workers()
                    raise QualificationError(
                        f"state worker {index} failed ({status}); log tail:\n{tail}"
                    )
                finished.append(process)
            for process in finished:
                del active[process]
            if finished:
                completed_workers += len(finished)
                last_completion_monotonic = time.monotonic()
            now = time.monotonic()
            if now - started_monotonic > global_timeout_s:
                terminate_count, kill_count = stop_active_workers()
                raise QualificationError(
                    "CPU materialization global watchdog exceeded "
                    f"{global_timeout_s} s; terminate={terminate_count}; kill={kill_count}"
                )
            if active and now - last_completion_monotonic > no_progress_timeout_s:
                terminate_count, kill_count = stop_active_workers()
                raise QualificationError(
                    "CPU worker no-progress watchdog exceeded "
                    f"{no_progress_timeout_s} s; terminate={terminate_count}; "
                    f"kill={kill_count}"
                )
            if active and not finished:
                time.sleep(0.1)
    finally:
        stop_active_workers()
        for process, (_index, handle, _path) in active.items():
            if not handle.closed:
                handle.close()
    records = []
    for index in range(STATE_COUNT):
        state_paths = sorted(
            (output_root / "materialization/states").glob(
                f"*/state_receipt.json"
            )
        )
        if len(state_paths) != STATE_COUNT:
            raise QualificationError(
                f"materialization produced {len(state_paths)} state receipts, expected 48"
            )
        records = [load_json(path) for path in state_paths]
        break
    records.sort(key=lambda row: int(row["state_index"]))
    for expected, row in enumerate(records):
        validate_phase_binding(row, source_freeze_commit)
        if int(row["state_index"]) != expected or row.get("pass") is not True:
            raise QualificationError("state receipt identity/status mismatch")
    return records, {
        "no_progress_timeout_s": no_progress_timeout_s,
        "global_timeout_s": global_timeout_s,
        "workers": workers,
        "completed_workers": completed_workers,
        "no_progress_timeout_breaches": 0,
        "global_timeout_breaches": 0,
        "terminate_signals": 0,
        "kill_signals": 0,
        "automatic_retries": 0,
        "resume_used": False,
        "pass": True,
    }


@contextmanager
def _cpu_materialization_global_deadline(
    timeout_s: int,
) -> Iterable[None]:
    """Interrupt the complete coordinator CPU phase at the frozen deadline."""

    if timeout_s <= 0:
        raise QualificationError("CPU materialization timeout must be positive")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise QualificationError("an unrelated process alarm is already active")

    def expire(_signum: int, _frame: Any) -> None:
        raise QualificationError(
            f"CPU materialization global watchdog exceeded {timeout_s} s"
        )

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _existing_h1_by_branch() -> dict[tuple[str, int], bool]:
    index = load_json(CONTACT_EVENT_INDEX)
    if index.get("schema") != "lewm_contact_hazard_raw_contact_event_index_v1":
        raise QualificationError("existing contact-event index schema mismatch")
    if int(index.get("states", -1)) != 48 or int(index.get("branches", -1)) != 576:
        raise QualificationError("existing contact-event index cardinality mismatch")
    output: dict[tuple[str, int], bool] = {}
    for state in index.get("state_records", []):
        state_id = str(state["state_id"])
        raw_path = Path(state["raw_points_path"])
        if not raw_path.is_file() or sha256_file(raw_path) != state["raw_points_sha256"]:
            raise QualificationError(f"existing contact evidence drift: {state_id}")
        # Physics-step rows are small (~10 MB total) and are existing structured
        # evidence; no raycasting or simulator replay is involved.
        per_branch_h1: dict[int, bool] = {index: False for index in range(12)}
        with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                candidate = int(row["candidate_index"])
                if int(row["physics_step"]) <= PHYSICS_STEPS_PER_BLOCK:
                    per_branch_h1[candidate] = True
        for candidate, value in per_branch_h1.items():
            output[(state_id, candidate)] = value
    if len(output) != 576:
        raise QualificationError("existing H1 contact map is incomplete")
    return output


def _build_materialization_indices(
    records: Sequence[Mapping[str, Any]], source_freeze_commit: str, output_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if len(records) != STATE_COUNT:
        raise QualificationError("materialization index requires exactly 48 states")
    cpu_runtime_binding = _binding(CPU_RUNTIME_INPUT_INVENTORY_REL, output_root)
    if any(
        row.get("cpu_runtime_input_inventory_binding") != cpu_runtime_binding
        for row in records
    ):
        raise QualificationError("CPU state receipt runtime-input binding drift")
    if any(
        row.get("historical_renderer_limitations")
        != CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
        for row in records
    ):
        raise QualificationError("CPU state receipt renderer-limitation drift")
    h1 = _existing_h1_by_branch()
    context_records: list[dict[str, Any]] = []
    fanout_records: list[dict[str, Any]] = []
    goal_records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    macro_mapping: dict[str, int] | None = None
    raw_continuation_capture_count = 0
    raw_continuation_restore_count = 0
    raw_terminal_flag_counts = {
        "fall": 0,
        "out_of_bounds": 0,
        "tipped": 0,
        "nan": 0,
    }
    for row in records:
        state_id = str(row["state_id"])
        raw_continuations = row["fanout"].get("raw_continuation_snapshots")
        if not isinstance(raw_continuations, list) or len(raw_continuations) != 9:
            raise QualificationError(
                f"{state_id}: raw-continuation snapshot cardinality is not 9"
            )
        for expected_index, raw in enumerate(raw_continuations):
            if (
                raw.get("current_primitive_index") != expected_index
                or raw.get("capture_mode")
                != "RAW_CONTINUATION_AFTER_CURRENT_H1"
                or raw.get("production_reset_checks_suppressed") is not True
                or raw.get("evaluation_only") is not True
                or raw.get("restored_for_successors") != 9
                or not isinstance(raw.get("snapshot_digest"), str)
                or len(raw["snapshot_digest"]) != 64
                or set(raw.get("terminal_flags", {})) != set(raw_terminal_flag_counts)
            ):
                raise QualificationError(
                    f"{state_id}: raw-continuation snapshot receipt drift"
                )
            if raw["terminal_flags"]["nan"] is not False:
                raise QualificationError(
                    f"{state_id}: NaN raw continuation was not rejected"
                )
            raw_continuation_capture_count += 1
            raw_continuation_restore_count += int(raw["restored_for_successors"])
            for flag, active in raw["terminal_flags"].items():
                raw_terminal_flag_counts[flag] += int(bool(active))
        context_records.append(
            {
                key: row[key]
                for key in (
                    "state_id",
                    "family",
                    "role",
                    "branch_snapshot_digest",
                    "replay_snapshot_digest",
                    "manifest_snapshot_digest_or_null",
                    "manifest_snapshot_digest_match_or_null",
                    "context_rgb_paths",
                    "context_rgb_sha256s",
                    "context_source_frame_indices",
                    "context_command_tick_indices",
                    "context_elapsed_s",
                    "control_command_indices",
                    "control_timestamps",
                    "control_history_raw_3x5x2",
                    "control_history_normalized_3x5x2",
                    "action_blocks_raw_3x5x2_by_candidate",
                    "action_blocks_raw_3x10_by_candidate",
                    "requested_action_blocks_raw_3x5x3_by_candidate",
                    "applied_action_blocks_raw_3x5x3_by_candidate",
                    "requested_applied_action_authority_validation",
                    "pass",
                )
            }
        )
        fanout = copy.deepcopy(row["fanout"])
        mapping = fanout["current_action_mapping"]["macro_candidate_to_primitive_index"]
        if macro_mapping is None:
            macro_mapping = mapping
        elif mapping != macro_mapping:
            raise QualificationError("macro-to-primitive mapping varies by state")
        with np.load(
            _artifact_path(fanout["shard_path"], output_root), allow_pickle=False
        ) as arrays:
            current = np.asarray(arrays["current_contact_bitset"], dtype=np.bool_)
            successor = np.asarray(arrays["successor_contact_bitset"], dtype=np.bool_)
        primitive_safe_counts = np.sum(
            ~np.any(successor, axis=2), axis=1, dtype=np.int64
        )
        candidate_safe_counts = [
            int(primitive_safe_counts[int(mapping[str(candidate)])])
            for candidate in range(12)
        ]
        candidate_successor_viable = [value > 0 for value in candidate_safe_counts]
        candidate_immediate_contact = [
            bool(np.any(current[int(mapping[str(candidate)])]))
            for candidate in range(12)
        ]
        candidate_oracle_admissible = [
            (not contact) and successor_viable
            for contact, successor_viable in zip(
                candidate_immediate_contact, candidate_successor_viable, strict=True
            )
        ]
        cross_rows = []
        for candidate in range(12):
            observed = bool(np.any(current[int(mapping[str(candidate)])]))
            expected = h1[(state_id, candidate)]
            match = observed == expected
            cross_rows.append(
                {
                    "candidate_index": candidate,
                    "rerun_h1_contact": observed,
                    "existing_h1_contact": expected,
                    "match": match,
                }
            )
            if not match:
                mismatches.append(f"{state_id}:{candidate:02d}")
        fanout_records.append(
            {
                "state_id": state_id,
                "family": row["family"],
                "role": row["role"],
                "branch_snapshot_digest": row["branch_snapshot_digest"],
                "shard_path": fanout["shard_path"],
                "shard_sha256": fanout["shard_sha256"],
                "shard_bytes": fanout["shard_bytes"],
                "array_manifest": fanout["array_manifest"],
                "current_contact_bitset_shape": fanout["current_contact_bitset_shape"],
                "successor_contact_bitset_shape": fanout["successor_contact_bitset_shape"],
                "first_contact_step": fanout["first_contact_step"],
                "first_contact_link": fanout["first_contact_link"],
                "current_action_mapping": fanout["current_action_mapping"],
                "successor_action_mapping": fanout["successor_action_mapping"],
                "raw_continuation_snapshots": raw_continuations,
                "successor_safe_action_count": candidate_safe_counts,
                "successor_viable": candidate_successor_viable,
                "oracle_viability_admissible": candidate_oracle_admissible,
                "h1_contact_cross_validation": cross_rows,
                "pass": all(item["match"] for item in cross_rows),
            }
        )
        goal = row["goal"]
        goal_records.append(
            {
                "state_id": state_id,
                "family": row["family"],
                "role": row["role"],
                "waypoint_world_xy": goal["waypoint_world_xy"],
                "goal_body_dx_dy_sin_dyaw_cos_dyaw": goal[
                    "goal_body_dx_dy_sin_dyaw_cos_dyaw"
                ],
                "snapshot_base_z": goal["snapshot_base_z"],
                "waypoint_path_cells": goal["waypoint_path_cells"],
                "path_cell_centers_world_xy": goal["path_cell_centers_world_xy"],
                "goal_cell_preconditions": goal["goal_cell_preconditions"],
                "goal_pose_world_xyz_rpy": goal["goal_pose_world_xyz_rpy"],
                "branch_snapshot_digest_expected": row["branch_snapshot_digest"],
                "snapshot_digest_observed": row["replay_snapshot_digest"],
                "manifest_waypoint_xy_or_null": goal["manifest_waypoint_xy_or_null"],
                "manifest_waypoint_body_xy_or_null": goal[
                    "manifest_waypoint_body_xy_or_null"
                ],
                "manifest_field_equality_audit": goal["manifest_field_equality_audit"],
                "rgb_path": goal["rgb_path"],
                "rgb_sha256": goal["rgb_sha256"],
                "token_path": None,
                "token_sha256": None,
                "token_shape": None,
                "pass": True,
            }
        )
    if mismatches:
        raise QualificationError(f"H1 physics-rate cross-validation mismatches: {mismatches}")
    context_index = attach_digest(
        {
            **_phase_core(
                "jepa_local_waypoint_context_reconstruction_index_v1",
                source_freeze_commit,
            ),
            "cpu_runtime_input_inventory_binding": cpu_runtime_binding,
            "states": 48,
            "records": context_records,
            "source_frame_offsets": [-480, -240, 0],
            "command_tick_offsets": [-10, -5, 0],
            "elapsed_offsets_s": [-1.0, -0.5, 0.0],
            "warmup_block_boundaries": [38, 39, 40],
            "reconstruction_prefix_custody": copy.deepcopy(
                CONTRACT.RECONSTRUCTION_PREFIX_CUSTODY
            ),
            "branch_snapshot_authority_validation": {
                "states": 48,
                "passed": 48,
                "failed": 0,
            },
            "manifest_snapshot_descriptive_audit": {
                "authority": False,
                "absence_allowed": True,
            },
            "current_rgb_authority_reproduction": {
                "status": "PENDING_GPU_BOUND_AUTHORITY_VALIDATION",
                "states_expected": 48,
                "states_exact": 0,
            },
            "current_token_authority_reproduction": {
                "status": "PENDING_GPU_BOUND_AUTHORITY_VALIDATION",
                "states_expected": 48,
                "states_exact": 0,
            },
            "control_history_index_timestamp_validation": {
                "states": 48,
                "contiguous_15_commands": True,
            },
            "requested_applied_action_authority_validation": {
                "states": 48,
                "branch_ledger_rows": 576,
                "requested_candidate_bank_exact": 576,
                "applied_branch_ledger_exact": 576,
                "active_projection_exact": 576,
                "requested_and_applied_never_conflated": True,
                "pass": True,
            },
            "control_normalisation_binding": copy.deepcopy(
                records[0]["control_normalisation_binding"]
            ),
            "failed_state_ids": [],
        }
    )
    if any(
        row["control_normalisation_binding"]
        != context_index["control_normalisation_binding"]
        for row in records
    ):
        raise QualificationError("control-normalisation binding varies by state")
    fanout_index = attach_digest(
        {
            **_phase_core(
                "jepa_local_waypoint_oracle_admissibility_fanout_index_v1",
                source_freeze_commit,
            ),
            "cpu_runtime_input_inventory_binding": cpu_runtime_binding,
            "states": 48,
            "current_blocks": 48 * 9,
            "successor_blocks": 48 * 81,
            "physics_frames": 48 * 90 * 250,
            "records": fanout_records,
            "macro_candidate_to_first_primitive": macro_mapping,
            "existing_h1_contact_cross_validation": {
                "rows": 576,
                "matches": 576,
                "mismatches": 0,
                "authority_sha256": INPUT_BINDINGS["contact_event_index"]["sha256"],
            },
            "raw_continuation_policy_validation": {
                "captures": raw_continuation_capture_count,
                "expected_captures": 48 * 9,
                "successor_restores": raw_continuation_restore_count,
                "expected_successor_restores": 48 * 9 * 9,
                "capture_mode": "RAW_CONTINUATION_AFTER_CURRENT_H1",
                "fall_tip_out_of_bounds_permitted_and_persisted": True,
                "terminal_flag_counts": raw_terminal_flag_counts,
                "nan_rejected": raw_terminal_flag_counts["nan"] == 0,
                "production_reset_checks_suppressed": True,
                "evaluation_only": True,
                "pass": raw_continuation_capture_count == 48 * 9
                and raw_continuation_restore_count == 48 * 9 * 9
                and raw_terminal_flag_counts["nan"] == 0,
            },
            "failed_state_ids": [],
        }
    )
    goal_index = attach_digest(
        {
            **_phase_core("jepa_local_waypoint_goal_view_index_v1", source_freeze_commit),
            "cpu_runtime_input_inventory_binding": cpu_runtime_binding,
            "historical_renderer_limitations": copy.deepcopy(
                CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
            ),
            "states": 48,
            "records": goal_records,
            "renderer_sha256": CONTRACT.STATIC_FILE_BINDINGS["renderer"]["sha256"],
            "encoder_checkpoint_sha256": CONTRACT.ENCODER_SHA256,
            "candidate_independence_validation": {
                "views": 48,
                "candidate_dependent_views": 0,
                "pass": True,
            },
            "failed_state_ids": [],
        }
    )
    atomic_json(output_root / CONTEXT_INDEX_REL, context_index)
    atomic_json(output_root / FANOUT_INDEX_REL, fanout_index)
    atomic_json(output_root / GOAL_INDEX_REL, goal_index)
    return context_index, fanout_index, goal_index


def materialize(
    source_freeze_commit: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run CPU reconstruction/fanout, then isolated GPU encoding/inference."""

    _validate_cpu_interpreter_entrypoint()
    if output_root.resolve() == OUTPUT_ROOT.resolve():
        raise QualificationError("direct canonical materialization is forbidden")
    validate_source_freeze_commit(source_freeze_commit, require_live_head=True)
    validate_frozen_receipts()
    validate_input_hashes()
    pre = load_json(output_root / PREEXEC_REL)
    validate_phase_binding(pre, source_freeze_commit)
    if pre.get("pass") is not True:
        raise QualificationError("preexecution did not pass")
    validate_cpu_runtime_input_inventory(
        source_freeze_commit,
        output_root=output_root,
        persist_if_absent=False,
    )
    cpu_runtime_binding = _binding(CPU_RUNTIME_INPUT_INVENTORY_REL, output_root)
    if pre.get("cpu_runtime_input_inventory_binding") != cpu_runtime_binding:
        raise QualificationError("preexecution CPU runtime-input binding drift")
    running = attach_digest(
        {
            **_phase_core(
                "jepa_local_waypoint_planning_cost_running_marker_v1",
                source_freeze_commit,
            ),
            "pid": os.getpid(),
            "phase": "CPU_MATERIALIZATION",
            "started_unix_s": time.time(),
        }
    )
    atomic_json(output_root / RUNNING_REL, running)
    started = time.time()
    worker_count = os.cpu_count() if workers is None else int(workers)
    global_timeout_s = int(
        CONTRACT.EXECUTION_WATCHDOGS["cpu_materialization_global_timeout_s"]
    )
    with _cpu_materialization_global_deadline(global_timeout_s):
        records, cpu_watchdog_status = _run_cpu_workers(
            source_freeze_commit, output_root=output_root, workers=worker_count
        )
        # The simulator/controller/renderer inputs are rehashed immediately after
        # all 32 workers and again immediately before the isolated GPU phase.
        validate_input_hashes()
        validate_cpu_runtime_input_inventory(
            source_freeze_commit,
            output_root=output_root,
            persist_if_absent=False,
        )
        context, fanout, goal = _build_materialization_indices(
            records, source_freeze_commit, output_root
        )
        cpu_runtime_s = time.time() - started
        fanout = attach_digest(
            {
                **fanout,
                "cpu_materialization_runtime_s": cpu_runtime_s,
                "cpu_workers": worker_count,
                "cpu_watchdog_status": cpu_watchdog_status,
                "cpu_peak_rss_bytes": max(
                    int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                    int(
                        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                        * 1024
                    ),
                ),
            }
        )
        atomic_json(output_root / FANOUT_INDEX_REL, fanout)
    running["phase"] = "GPU_ENCODING_AND_INFERENCE"
    running = attach_digest(running)
    atomic_json(output_root / RUNNING_REL, running)
    validate_source_freeze_commit(source_freeze_commit, require_live_head=True)
    validate_frozen_receipts()
    validate_input_hashes()
    validate_cpu_runtime_input_inventory(
        source_freeze_commit,
        output_root=output_root,
        persist_if_absent=False,
    )
    result = _run_gpu_child(
        [
            "materialize",
            "--source-freeze-commit",
            source_freeze_commit,
            "--output-root",
            str(output_root),
        ]
    )
    latent = load_json(output_root / LATENT_INDEX_REL)
    validate_phase_binding(latent, source_freeze_commit)
    validate_latent_tensor_index(latent, output_root=output_root, deep=True)
    validate_cpu_runtime_input_inventory(
        source_freeze_commit,
        output_root=output_root,
        persist_if_absent=False,
    )
    goal_after = load_json(output_root / GOAL_INDEX_REL)
    validate_phase_binding(goal_after, source_freeze_commit)
    return {
        "source_freeze_commit": source_freeze_commit,
        "workers": worker_count,
        "runtime_s": time.time() - started,
        "context_content_digest": context["content_digest"],
        "fanout_content_digest": fanout["content_digest"],
        "goal_content_digest": goal_after["content_digest"],
        "latent_content_digest": latent["content_digest"],
        "gpu_stdout": result.stdout.strip(),
    }


def validate_latent_tensor_index(
    value: Mapping[str, Any], *, output_root: Path, deep: bool
) -> None:
    schema = CONTRACT.build_output_schema()["files"]["latent_tensor_index"]
    expected = schema["expected_counts"]
    expected_counts = {key: expected[key] for key in expected if key != "total"}
    if value.get("counts_by_kind") != expected_counts:
        raise QualificationError("latent tensor counts-by-kind mismatch")
    if int(value.get("total_records", -1)) != int(expected["total"]):
        raise QualificationError("latent tensor total record count mismatch")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != int(expected["total"]):
        raise QualificationError("latent tensor record list cardinality mismatch")
    def state_key(state_id: str) -> tuple[str, int]:
        prefix, separator, suffix = state_id.rpartition("-")
        if not separator or not suffix.isdigit():
            raise QualificationError(f"latent state ID lacks numeric suffix: {state_id}")
        return prefix, int(suffix)

    state_ids = sorted(
        {str(row.get("state_id")) for row in records if row.get("kind") == "CURRENT"},
        key=state_key,
    )
    if len(state_ids) != STATE_COUNT:
        raise QualificationError("latent CURRENT records do not identify 48 states")
    expected_identities: list[tuple[Any, ...]] = []
    for kind in sorted(expected_counts):
        for state_id in state_ids:
            if kind == "CONTEXT":
                domain = ((None, horizon) for horizon in (-2, -1, 0))
            elif kind in {"CURRENT", "GOAL"}:
                domain = ((None, None),)
            else:
                domain = (
                    (candidate, horizon)
                    for candidate in range(CANDIDATES_PER_STATE)
                    for horizon in (1, 2, 3)
                )
            expected_identities.extend(
                (kind, state_id, candidate, horizon)
                for candidate, horizon in domain
            )

    observed_identities: list[tuple[Any, ...]] = []
    identities: set[tuple[Any, ...]] = set()
    unique_payloads: dict[tuple[str, str], int] = {}
    path_to_sha: dict[str, str] = {}
    by_identity: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in records:
        identity = (
            row.get("kind"),
            str(row.get("state_id")),
            row.get("candidate_index_or_null"),
            row.get("horizon_or_null"),
        )
        if identity in identities:
            raise QualificationError(f"duplicate latent tensor identity {identity}")
        identities.add(identity)
        observed_identities.append(identity)
        by_identity[identity] = row
        if row.get("shape") != [768, 1024] or row.get("dtype") != "float16":
            raise QualificationError(f"latent tensor shape/dtype mismatch {identity}")
        reference = Path(str(row["path"]))
        if reference.is_absolute() or ".." in reference.parts:
            raise QualificationError(f"latent tensor path is not output-relative {identity}")
        path = output_root / reference
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise QualificationError(f"latent tensor missing/byte mismatch {identity}")
        payload_identity = (str(reference), str(row["sha256"]))
        if payload_identity in unique_payloads and unique_payloads[payload_identity] != int(
            row["bytes"]
        ):
            raise QualificationError(f"latent shared-payload byte drift {identity}")
        if str(reference) in path_to_sha and path_to_sha[str(reference)] != str(
            row["sha256"]
        ):
            raise QualificationError(f"latent path aliases conflicting payloads {identity}")
        unique_payloads[payload_identity] = int(row["bytes"])
        path_to_sha[str(reference)] = str(row["sha256"])
        if deep:
            if sha256_file(path) != row["sha256"]:
                raise QualificationError(f"latent tensor SHA mismatch {identity}")
            array = np.load(path, allow_pickle=False)
            if array.shape != TENSOR_SHAPE or array.dtype != TENSOR_DTYPE:
                raise QualificationError(f"latent tensor payload mismatch {identity}")
            if not np.isfinite(array).all():
                raise QualificationError(f"latent tensor non-finite {identity}")

    if observed_identities != expected_identities:
        raise QualificationError("latent tensor identity domain or canonical order drift")
    recomputed_counts = {
        kind: sum(identity[0] == kind for identity in observed_identities)
        for kind in expected_counts
    }
    if recomputed_counts != expected_counts:
        raise QualificationError("latent tensor record-derived kind counts drift")
    if int(value.get("unique_tensor_payload_count", -1)) != len(unique_payloads):
        raise QualificationError("latent unique-payload count drift")
    if int(value.get("total_bytes", -1)) != sum(unique_payloads.values()):
        raise QualificationError("latent unique-payload byte total drift")
    if value.get("shape_dtype_validation") != {
        "records": int(expected["total"]),
        "passed": int(expected["total"]),
        "shape": [768, 1024],
        "dtype": "float16",
    } or value.get("failed_records") != []:
        raise QualificationError("latent shape/dtype validation receipt drift")

    for state_id in state_ids:
        current = by_identity[("CURRENT", state_id, None, None)]
        context = by_identity[("CONTEXT", state_id, None, 0)]
        for key in ("path", "sha256", "bytes", "shape", "dtype"):
            if current[key] != context[key]:
                raise QualificationError(f"CURRENT/context alias drift: {state_id}:{key}")
        external = current.get("external_existing_artifact")
        if not isinstance(external, Mapping) or any(
            external.get(key) is not True
            for key in (
                "byte_exact_array_equality",
                "raw_payload_byte_exact",
            )
        ) or external.get("raw_payload_sha256_expected") != external.get(
            "raw_payload_sha256_observed"
        ):
            raise QualificationError(f"CURRENT external authority drift: {state_id}")
    for identity, row in by_identity.items():
        kind = str(identity[0])
        external = row.get("external_existing_artifact")
        if kind == "TRUE_FUTURE":
            if not isinstance(external, Mapping) or external.get("array_equality") is not True:
                raise QualificationError(f"true-future external authority drift: {identity}")
        elif kind != "CURRENT" and external is not False:
            raise QualificationError(f"unexpected external tensor authority: {identity}")

    latent_contract = CONTRACT.build_contract()["latent_bindings"]
    expected_true = latent_contract["true_future_target_index"]
    if value.get("external_true_future_index_binding") != {
        "path": str(ROOT / expected_true["path"]),
        "sha256": expected_true["sha256"],
        "count": expected_true["entries"],
    }:
        raise QualificationError("latent true-future input-index binding drift")
    expected_current = latent_contract["dense_true_future_index"]
    if value.get("external_current_index_binding") != {
        "path": str(ROOT / expected_current["path"]),
        "sha256": expected_current["sha256"],
        "states": expected_current["current_view_authority"]["unique_state_ids"],
        "rgb_and_token_array_exact_equality": True,
    }:
        raise QualificationError("latent current input-index binding drift")

    predictor = CONTRACT.build_contract()["predictor_bindings"]
    expected_predictors = {
        name: {
            "path": predictor[name]["path"],
            "sha256": predictor[name]["sha256"],
            "bytes": predictor[name]["bytes"],
        }
        for name in ("one_step", "two_step")
    }
    if value.get("checkpoint_bindings") != expected_predictors:
        raise QualificationError("latent predictor checkpoint binding drift")
    encoder = value.get("encoder_binding")
    if not isinstance(encoder, Mapping) or encoder.get("sha256") != CONTRACT.ENCODER_SHA256:
        raise QualificationError("latent encoder checkpoint binding drift")
    if Path(str(encoder.get("path"))) != Path.home() / ".cache/vjepa2_1_vitl_dist_vitG_384.pt":
        raise QualificationError("latent encoder checkpoint path drift")
    if not isinstance(encoder.get("bytes"), int) or int(encoder["bytes"]) <= 0:
        raise QualificationError("latent encoder checkpoint byte receipt is invalid")

    batch_binding = value.get("batch_manifest")
    expected_batch_path = Path("latents/batch_manifest.json")
    if not isinstance(batch_binding, Mapping) or batch_binding.get("path") != str(
        expected_batch_path
    ):
        raise QualificationError("latent batch-manifest path drift")
    batch_path = output_root / expected_batch_path
    batch = load_json(batch_path)
    validate_phase_binding(batch, str(value["source_freeze_commit"]))
    if sha256_file(batch_path) != batch_binding.get("sha256") or batch.get(
        "content_digest"
    ) != batch_binding.get("content_digest"):
        raise QualificationError("latent batch-manifest binding drift")
    encoder_batch = batch.get("encoder")
    if not isinstance(encoder_batch, Mapping) or {
        key: encoder_batch.get(key)
        for key in ("order", "batch_size", "padding", "dynamic_fallback", "calls")
    } != {
        "order": "ascending rgb_sha256, kind, numeric state, slot",
        "batch_size": 16,
        "padding": False,
        "dynamic_fallback": False,
        "calls": 12,
    }:
        raise QualificationError("encoder batch contract drift")
    encoder_rows: list[Mapping[str, Any]] = []
    for batch_index, manifest_batch in enumerate(encoder_batch.get("batches", [])):
        if manifest_batch.get("batch_index") != batch_index or manifest_batch.get(
            "size"
        ) != 16 or len(manifest_batch.get("records", [])) != 16:
            raise QualificationError("encoder batch cardinality/order drift")
        encoder_rows.extend(manifest_batch["records"])
    if len(encoder_rows) != 192:
        raise QualificationError("encoder batch manifest does not cover 192 frames")
    expected_frame_domain = {
        ("CONTEXT", state_id, slot)
        for state_id in state_ids
        for slot in (0, 1, 2)
    } | {("GOAL", state_id, None) for state_id in state_ids}
    observed_frame_domain = {
        (str(row.get("kind")), str(row.get("state_id")), row.get("slot"))
        for row in encoder_rows
    }
    if observed_frame_domain != expected_frame_domain:
        raise QualificationError("encoder batch frame identity domain drift")
    frame_sort_key = lambda row: (  # noqa: E731 - mirrors frozen GPU ordering.
        str(row["rgb_sha256"]),
        str(row["kind"]),
        state_key(str(row["state_id"])),
        int(row.get("slot", -1)) if row.get("slot") is not None else -1,
    )
    if encoder_rows != sorted(encoder_rows, key=frame_sort_key):
        raise QualificationError("encoder batch frame order drift")

    predictor_batch = batch.get("predictor")
    if not isinstance(predictor_batch, Mapping) or {
        key: predictor_batch.get(key)
        for key in (
            "order",
            "state_order",
            "batch_size",
            "horizons",
            "dynamic_fallback",
            "one_step_calls",
            "two_step_calls",
        )
    } != {
        "order": [
            "one_step checkpoint all 48 states",
            "two_step checkpoint all 48 states",
        ],
        "state_order": "numeric frozen state order",
        "batch_size": 12,
        "horizons": [1, 2, 3],
        "dynamic_fallback": False,
        "one_step_calls": 48,
        "two_step_calls": 48,
    }:
        raise QualificationError("predictor batch contract drift")
    predictor_calls = predictor_batch.get("calls")
    if not isinstance(predictor_calls, list) or len(predictor_calls) != 96:
        raise QualificationError("predictor batch call cardinality drift")
    expected_calls = []
    for checkpoint, source in (
        ("one_step", "ONE_STEP_PREDICTED"),
        ("two_step", "TWO_STEP_PREDICTED"),
    ):
        expected_calls.extend(
            {
                "call_index": index,
                "state_id": state_id,
                "batch_size": 12,
                "horizons": [1, 2, 3],
                "checkpoint": checkpoint,
                "source": source,
            }
            for index, state_id in enumerate(state_ids)
        )
    if predictor_calls != expected_calls:
        raise QualificationError("predictor checkpoint/state call order drift")
    if batch.get("checkpoint_tensor_open_count") != 3 or batch.get(
        "training_steps"
    ) != 0:
        raise QualificationError("batch-manifest checkpoint/training custody drift")
    for key, expected_value in (
        ("checkpoint_tensor_open_count", 3),
        ("predictor_inference_calls", 96),
        ("predictor_model_forward_calls", 288),
        ("encoder_inference_calls", 12),
        ("training_steps", 0),
    ):
        if value.get(key) != expected_value:
            raise QualificationError(f"latent inference count drift: {key}")


def _load_fanout_arrays(
    record: Mapping[str, Any], *, output_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    path = _artifact_path(record["shard_path"], output_root)
    if not path.is_file() or sha256_file(path) != record["shard_sha256"]:
        raise QualificationError(f"fanout shard drift: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        current = np.asarray(arrays["current_contact_bitset"], dtype=np.bool_)
        successor = np.asarray(arrays["successor_contact_bitset"], dtype=np.bool_)
    if current.shape != (9, 250) or successor.shape != (9, 9, 250):
        raise QualificationError("fanout shard payload shape mismatch")
    return current, successor


def _tensor_map(index: Mapping[str, Any]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in index["records"]:
        key = (
            str(row["kind"]),
            str(row["state_id"]),
            row["candidate_index_or_null"],
            row["horizon_or_null"],
        )
        if key in output:
            raise QualificationError(f"duplicate tensor record {key}")
        output[key] = row
    return output


def _tensor_array(row: Mapping[str, Any], output_root: Path) -> np.ndarray:
    path = Path(row["path"])
    if not path.is_absolute():
        path = output_root / path
    value = np.load(path, allow_pickle=False)
    if value.shape != TENSOR_SHAPE or value.dtype != TENSOR_DTYPE:
        raise QualificationError(f"tensor payload contract mismatch: {path}")
    return value


def _build_dense_replay_input_index(
    source_freeze_commit: str, *, output_root: Path, persist: bool = True
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Bind all 48 existing dense replay records before descriptive use."""

    expected_receipt_sha = (
        "a547ac544a869a6ef75a4798b22875291e55604f9e53ceaea24a790db09df7e1"
    )
    if (
        not DENSE_EVIDENCE_RECEIPT.is_file()
        or sha256_file(DENSE_EVIDENCE_RECEIPT) != expected_receipt_sha
    ):
        raise QualificationError("dense replay evidence receipt binding drift")
    states = load_json(STATE_MANIFEST).get("state_candidates")
    if not isinstance(states, list) or len(states) != 48:
        raise QualificationError("dense replay custody lacks frozen state identities")
    prefreeze_records: list[dict[str, Any]] = []
    dense_paths: list[Path] = []
    for state in states:
        state_id = str(state["state_id"])
        path = DENSE_ROOT / "dense_replay" / f"{state_id}.json"
        if not path.is_file():
            raise QualificationError(f"dense replay input is missing: {state_id}")
        try:
            relative = path.relative_to(ROOT)
        except ValueError as exc:
            raise QualificationError(
                f"dense replay input is outside the frozen repository: {path}"
            ) from exc
        prefreeze_records.append(
            {
                "state_id": state_id,
                "path": str(relative),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
        dense_paths.append(path)
    prefreeze_validation = _validate_prefreeze_byte_inventory(
        prefreeze_records,
        CONTRACT.DENSE_ROUTE_REPLAY_INPUT_BINDINGS,
        states=None,
    )
    records: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for state, path, prefreeze_record in zip(
        states, dense_paths, prefreeze_records, strict=True
    ):
        state_id = str(state["state_id"])
        payload = load_json(path)
        declared = payload.get("content_digest")
        core = copy.deepcopy(payload)
        core.pop("content_digest", None)
        observed = hashlib.sha256(canonical_json_bytes(core)[:-1]).hexdigest()
        self_valid = isinstance(declared, str) and declared == observed
        boundaries = payload.get("horizon_tick_boundaries")
        branches = payload.get("branches")
        passed = bool(
            payload.get("schema") == "dense_route_intent_true_future_state_v1"
            and payload.get("status") == "PASS"
            and str(payload.get("state_id")) == state_id
            and self_valid
            and isinstance(branches, list)
            and len(branches) == 12
            and int(payload.get("h3_tick_count", -1)) == 15
            and boundaries == [5, 10, 15]
            and all(
                len(branch.get("ticks", [])) == 15
                and branch.get("horizon_tick_boundaries") == [5, 10, 15]
                for branch in branches or []
            )
        )
        records.append(
            {
                "state_id": state_id,
                "path": str(path),
                "sha256": prefreeze_record["sha256"],
                "bytes": prefreeze_record["bytes"],
                "schema": payload.get("schema"),
                "status": payload.get("status"),
                "content_digest": declared,
                "self_digest_valid": self_valid,
                "branches": 0 if not isinstance(branches, list) else len(branches),
                "h3_tick_count": payload.get("h3_tick_count"),
                "horizon_tick_boundaries": boundaries,
                "pass": passed,
            }
        )
        if not passed:
            raise QualificationError(f"dense replay input validation failed: {state_id}")
        payloads[state_id] = payload
    index = attach_digest(
        {
            **_phase_core(
                "jepa_local_waypoint_dense_route_replay_input_index_v1",
                source_freeze_commit,
            ),
            "evidence_receipt_binding": {
                "path": str(DENSE_EVIDENCE_RECEIPT),
                "sha256": expected_receipt_sha,
                "bytes": DENSE_EVIDENCE_RECEIPT.stat().st_size,
            },
            "prefreeze_byte_inventory_binding": copy.deepcopy(
                CONTRACT.DENSE_ROUTE_REPLAY_INPUT_BINDINGS
            ),
            "prefreeze_byte_inventory_records": prefreeze_records,
            "prefreeze_byte_inventory_validation": prefreeze_validation,
            "states": 48,
            "records": records,
            "cardinality_validation": {
                "states": 48,
                "branches": 576,
                "ticks_per_branch": 15,
                "horizon_tick_boundaries": [5, 10, 15],
            },
            "self_digest_validation": {"passed": 48, "failed": 0},
            "failed_state_ids": [],
            "pass": True,
        }
    )
    if persist:
        atomic_json(output_root / DENSE_REPLAY_INPUT_INDEX_REL, index)
    return index, payloads


def _realised_route_fields_by_horizon(
    route_row: Mapping[str, Any],
    dense_branch: Mapping[str, Any],
    contact_horizons: tuple[bool, bool, bool],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    ticks = dense_branch.get("ticks")
    if not isinstance(ticks, list) or len(ticks) != 15:
        raise QualificationError("dense route branch lacks exact 15 ticks")
    for horizon, boundary, contact in zip(
        (1, 2, 3), (5, 10, 15), contact_horizons, strict=True
    ):
        source = route_row.get("horizons", {}).get(str(horizon))
        if not isinstance(source, Mapping):
            raise QualificationError(f"route row lacks H{horizon} fields")
        p_d = source.get("p_d", source.get("progress", source.get("distance_progress_m")))
        p_theta = source.get(
            "p_theta_rad", source.get("p_theta", source.get("heading_improvement_rad"))
        )
        if p_d is None or p_theta is None:
            raise QualificationError(f"route row lacks H{horizon} route progress")
        output[f"H{horizon}"] = {
            "p_d_m": float(p_d),
            "p_theta_rad": float(p_theta),
            "completed": bool(source.get("completed", source.get("completion", False))),
            "descriptive_contact": bool(contact),
            "stuck": bool(ticks[boundary - 1]["cumulative_stuck"]),
        }
    return output


def _descriptive_contact_horizons() -> dict[tuple[str, int], tuple[bool, bool, bool]]:
    index = load_json(CONTACT_EVENT_INDEX)
    output: dict[tuple[str, int], tuple[bool, bool, bool]] = {}
    for state in index["state_records"]:
        state_id = str(state["state_id"])
        by_candidate = {candidate: [False, False, False] for candidate in range(12)}
        raw_path = Path(state["raw_points_path"])
        if (
            not raw_path.is_file()
            or sha256_file(raw_path) != state["raw_points_sha256"]
        ):
            raise QualificationError(
                f"descriptive contact-event evidence drift: {state_id}"
            )
        with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                candidate = int(row["candidate_index"])
                step = int(row["physics_step"])
                if step <= 250:
                    by_candidate[candidate][0] = True
                if step <= 500:
                    by_candidate[candidate][1] = True
                if step <= 750:
                    by_candidate[candidate][2] = True
        output.update(
            {(state_id, candidate): tuple(value) for candidate, value in by_candidate.items()}
        )
    if len(output) != 576:
        raise QualificationError("descriptive H1/H2/H3 contact map is incomplete")
    return output


def _write_evidence_and_metrics(
    source_freeze_commit: str,
    output_root: Path,
    *,
    persist_auxiliary_indices: bool = True,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Recompute costs from persisted FP16 and reduce all frozen populations."""

    context_index = load_json(output_root / CONTEXT_INDEX_REL)
    fanout_index = load_json(output_root / FANOUT_INDEX_REL)
    latent_index = load_json(output_root / LATENT_INDEX_REL)
    goal_index = load_json(output_root / GOAL_INDEX_REL)
    for index in (context_index, fanout_index, latent_index, goal_index):
        validate_phase_binding(index, source_freeze_commit)
    validate_latent_tensor_index(latent_index, output_root=output_root, deep=True)
    tensor_by = _tensor_map(latent_index)
    contexts = {str(row["state_id"]): row for row in context_index["records"]}
    fanouts = {str(row["state_id"]): row for row in fanout_index["records"]}
    goals = {str(row["state_id"]): row for row in goal_index["records"]}
    route_rows = load_jsonl(ROUTE_LABELS)
    routes = {(str(row["state_id"]), int(row["candidate_index"])): row for row in route_rows}
    ledger_rows = load_jsonl(BRANCH_LEDGER)
    ledger = {(str(row["state_id"]), int(row["candidate_index"])): row for row in ledger_rows}
    if len(routes) != 576 or len(ledger) != 576:
        raise QualificationError("frozen route/branch row cardinality mismatch")
    dense_replay_index, dense_payloads = _build_dense_replay_input_index(
        source_freeze_commit,
        output_root=output_root,
        persist=persist_auxiliary_indices,
    )
    dense_record_refs = {
        str(row["state_id"]): {
            "path": row["path"],
            "sha256": row["sha256"],
            "content_digest": row["content_digest"],
        }
        for row in dense_replay_index["records"]
    }
    descriptive_contacts = _descriptive_contact_horizons()
    source_ids = list(CONTRACT.SOURCE_IDS)
    candidate_rows: list[dict[str, Any]] = []
    for state_id in sorted(contexts, key=lambda value: int(value.split("-")[-1])):
        context = contexts[state_id]
        fanout = fanouts[state_id]
        goal = goals[state_id]
        current_bits, successor_bits = _load_fanout_arrays(
            fanout, output_root=output_root
        )
        macro = fanout["current_action_mapping"]["macro_candidate_to_primitive_index"]
        goal_tokens = _tensor_array(tensor_by[("GOAL", state_id, None, None)], output_root)
        current_tokens = _tensor_array(
            tensor_by[("CURRENT", state_id, None, None)], output_root
        )
        current_cost = float(
            METRICS.tokenwise_normalized_cosine_mean_cost(current_tokens, goal_tokens)
        )
        dense_branches = {
            int(row["candidate_index"]): row
            for row in dense_payloads[state_id]["branches"]
        }
        if set(dense_branches) != set(range(12)):
            raise QualificationError(f"{state_id}: dense branch identities are incomplete")
        for candidate_index in range(12):
            contact_horizons = descriptive_contacts[(state_id, candidate_index)]
            route_by_horizon = _realised_route_fields_by_horizon(
                routes[(state_id, candidate_index)],
                dense_branches[candidate_index],
                contact_horizons,
            )
            h3 = route_by_horizon["H3"]
            route = {
                "p_d": h3["p_d_m"],
                "p_theta": h3["p_theta_rad"],
                "completed": h3["completed"],
                "stuck": h3["stuck"],
            }
            primitive_index = int(macro[str(candidate_index)])
            immediate_contact = bool(np.any(current_bits[primitive_index]))
            safe_count = int(np.sum(~np.any(successor_bits[primitive_index], axis=1)))
            successor_viable = safe_count > 0
            oracle_admissible = not immediate_contact and successor_viable
            h1_contact, h2_contact, h3_contact = contact_horizons
            if h1_contact != immediate_contact:
                raise QualificationError(
                    f"{state_id}:{candidate_index}: structured H1 contact mismatch"
                )
            if safe_count != int(fanout["successor_safe_action_count"][candidate_index]):
                raise QualificationError(
                    f"{state_id}:{candidate_index}: successor safe-count mismatch"
                )
            candidate_identity = str(
                ledger[(state_id, candidate_index)].get(
                    "candidate", candidate_index
                )
            )
            requested_tape = context[
                "requested_action_blocks_raw_3x5x3_by_candidate"
            ][candidate_index]
            applied_tape = context[
                "applied_action_blocks_raw_3x5x3_by_candidate"
            ][candidate_index]
            ledger_row = ledger[(state_id, candidate_index)]
            if not np.array_equal(
                np.asarray(requested_tape, dtype=np.float32),
                np.asarray(ledger_row["requested"][:3], dtype=np.float32),
            ) or not np.array_equal(
                np.asarray(applied_tape, dtype=np.float32),
                np.asarray(ledger_row["post_slew"][:3], dtype=np.float32),
            ):
                raise QualificationError(
                    f"{state_id}:{candidate_index}: requested/applied ledger custody drift"
                )
            action_authority = context[
                "requested_applied_action_authority_validation"
            ]["rows"][candidate_index]
            if (
                action_authority.get("candidate_index") != candidate_index
                or action_authority.get("pass") is not True
            ):
                raise QualificationError(
                    f"{state_id}:{candidate_index}: action-authority receipt drift"
                )
            for source in source_ids:
                costs = []
                hashes: dict[str, str] = {}
                for horizon in (1, 2, 3):
                    tensor_row = tensor_by[(source, state_id, candidate_index, horizon)]
                    tensor = _tensor_array(tensor_row, output_root)
                    costs.append(
                        float(
                            METRICS.tokenwise_normalized_cosine_mean_cost(
                                tensor, goal_tokens
                            )
                        )
                    )
                    hashes[f"H{horizon}"] = str(tensor_row["sha256"])
                monotonic = METRICS.trajectory_cost_monotonicity(
                    [current_cost, *costs]
                )
                candidate_rows.append(
                    {
                        "schema": "jepa_local_waypoint_candidate_evidence_v1",
                        "state_id": state_id,
                        "family": context["family"],
                        "role": context["role"],
                        "candidate_index": candidate_index,
                        "candidate_identity": candidate_identity,
                        "source": source,
                        "population_membership": {
                            "ALL_CANDIDATES": True,
                            "ORACLE_CONTACT_FREE": not immediate_contact,
                            "ORACLE_VIABILITY_ADMISSIBLE": oracle_admissible,
                        },
                        "context_frame_identities": ["block38", "block39", "block40"],
                        "context_frame_sha256s": context["context_rgb_sha256s"],
                        "context_offsets_source_frames": [-480, -240, 0],
                        "context_offsets_command_ticks": [-10, -5, 0],
                        "context_offsets_elapsed_s": [-1.0, -0.5, 0.0],
                        "context_control_history_raw": context[
                            "control_history_raw_3x5x2"
                        ],
                        "context_control_history_normalized": context[
                            "control_history_normalized_3x5x2"
                        ],
                        "action_blocks_raw_3x5x2": context[
                            "action_blocks_raw_3x5x2_by_candidate"
                        ][candidate_index],
                        "action_blocks_raw_3x10": context[
                            "action_blocks_raw_3x10_by_candidate"
                        ][candidate_index],
                        "requested_action_blocks_raw_3x5x3": requested_tape,
                        "applied_action_blocks_raw_3x5x3": applied_tape,
                        "requested_applied_action_authority_validation": (
                            action_authority
                        ),
                        "snapshot_digest": context["branch_snapshot_digest"],
                        "goal_view_sha256": goal["rgb_sha256"],
                        "goal_token_sha256": tensor_by[("GOAL", state_id, None, None)][
                            "sha256"
                        ],
                        "candidate_token_sha256_by_horizon": hashes,
                        "cost_current": current_cost,
                        "cost_h1": costs[0],
                        "cost_h2": costs[1],
                        "cost_h3": costs[2],
                        "monotonic_diagnostics": monotonic,
                        "realised_route_fields_by_horizon": route_by_horizon,
                        **{
                            f"p_d_h{horizon}": route_by_horizon[f"H{horizon}"][
                                "p_d_m"
                            ]
                            for horizon in (1, 2, 3)
                        },
                        **{
                            f"p_theta_h{horizon}": route_by_horizon[f"H{horizon}"][
                                "p_theta_rad"
                            ]
                            for horizon in (1, 2, 3)
                        },
                        **{
                            f"completed_h{horizon}": route_by_horizon[f"H{horizon}"][
                                "completed"
                            ]
                            for horizon in (1, 2, 3)
                        },
                        "oracle_route_fields_h3_primary": route,
                        "dense_route_replay_state_record_ref": dense_record_refs[
                            state_id
                        ],
                        "immediate_contact_h1": immediate_contact,
                        "descriptive_contact_h2": h2_contact,
                        "descriptive_contact_h3": h3_contact,
                        "contact_free_h1": not immediate_contact,
                        "successor_safe_action_count": safe_count,
                        "successor_viable": successor_viable,
                        "oracle_viability_admissible": oracle_admissible,
                        "successor_nonviable": not successor_viable,
                        "stuck": route["stuck"],
                        "completed": route["completed"],
                        "current_contact_bitset_shard_ref": fanout["shard_sha256"],
                        "successor_contact_bitset_shard_ref": fanout["shard_sha256"],
                        "input_contract_validation": True,
                    }
                )
    if len(candidate_rows) != 1728:
        raise QualificationError("candidate evidence cardinality is not 1728")
    # Every subsequent selection, paired effect, aggregate, gate and
    # classification is reduced only from the persisted scalar-row schema and
    # non-tensor goal metadata.  This is the explicit no-tensor second stage.
    return _reduce_candidate_rows_without_tensors(
        source_freeze_commit,
        candidate_rows,
        goal_index,
    )


_CROSS_SOURCE_CANDIDATE_INVARIANT_FIELDS = (
    "family",
    "role",
    "candidate_identity",
    "population_membership",
    "context_frame_identities",
    "context_frame_sha256s",
    "context_offsets_source_frames",
    "context_offsets_command_ticks",
    "context_offsets_elapsed_s",
    "context_control_history_raw",
    "context_control_history_normalized",
    "action_blocks_raw_3x5x2",
    "action_blocks_raw_3x10",
    "requested_action_blocks_raw_3x5x3",
    "applied_action_blocks_raw_3x5x3",
    "requested_applied_action_authority_validation",
    "snapshot_digest",
    "goal_view_sha256",
    "goal_token_sha256",
    "cost_current",
    "realised_route_fields_by_horizon",
    "p_d_h1",
    "p_d_h2",
    "p_d_h3",
    "p_theta_h1",
    "p_theta_h2",
    "p_theta_h3",
    "completed_h1",
    "completed_h2",
    "completed_h3",
    "oracle_route_fields_h3_primary",
    "dense_route_replay_state_record_ref",
    "immediate_contact_h1",
    "descriptive_contact_h2",
    "descriptive_contact_h3",
    "contact_free_h1",
    "successor_safe_action_count",
    "successor_viable",
    "oracle_viability_admissible",
    "successor_nonviable",
    "stuck",
    "completed",
    "current_contact_bitset_shard_ref",
    "successor_contact_bitset_shard_ref",
    "input_contract_validation",
)


def _validate_cross_source_candidate_invariants(
    by_identity: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> None:
    state_candidates = sorted({(state, candidate) for state, candidate, _ in by_identity})
    for state_id, candidate_index in state_candidates:
        try:
            authority = by_identity[(state_id, candidate_index, "TRUE_FUTURE")]
            predicted = [
                (
                    source,
                    by_identity[(state_id, candidate_index, source)],
                )
                for source in ("ONE_STEP_PREDICTED", "TWO_STEP_PREDICTED")
            ]
        except KeyError as exc:
            raise QualificationError(
                f"cross-source candidate identity missing: {state_id}:{candidate_index}"
            ) from exc
        for source, observed in predicted:
            for field in _CROSS_SOURCE_CANDIDATE_INVARIANT_FIELDS:
                if field not in authority or field not in observed:
                    raise QualificationError(
                        f"cross-source candidate invariant field missing: "
                        f"{state_id}:{candidate_index}:{source}:{field}"
                    )
                if observed[field] != authority[field]:
                    raise QualificationError(
                        f"cross-source candidate invariant drift: "
                        f"{state_id}:{candidate_index}:{source}:{field}"
                    )


def _validated_applied_action_tape(
    authority: Mapping[str, Any],
) -> np.ndarray:
    """Return the full applied H1-H3 tape without fabricating lateral velocity."""

    active = np.asarray(authority["action_blocks_raw_3x5x2"], np.float64)
    applied = np.asarray(
        authority["applied_action_blocks_raw_3x5x3"], np.float64
    )
    if active.shape != (3, 5, 2) or applied.shape != (3, 5, 3):
        raise QualificationError("row-only applied action block shape drift")
    if not np.isfinite(active).all() or not np.isfinite(applied).all():
        raise QualificationError("row-only applied action block is non-finite")
    if not np.array_equal(active, applied[:, :, (0, 2)]):
        raise QualificationError(
            "row-only active-channel projection differs from the full applied tape"
        )
    return applied


def _reduce_candidate_rows_without_tensors(
    source_freeze_commit: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    goal_index: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Pure scalar-row → selection/aggregate reducer; opens no tensor payload."""

    rows = [copy.deepcopy(dict(row)) for row in candidate_rows]
    goals = {
        str(row["state_id"]): row["goal_body_dx_dy_sin_dyaw_cos_dyaw"]
        for row in goal_index["records"]
    }
    by_identity = {
        (str(row["state_id"]), int(row["candidate_index"]), str(row["source"])): row
        for row in rows
    }
    if len(by_identity) != 1728 or set(row["source"] for row in rows) != set(
        CONTRACT.SOURCE_IDS
    ):
        raise QualificationError("row-only reducer candidate identities are incomplete")
    _validate_cross_source_candidate_invariants(by_identity)
    state_order: list[str] = []
    for row in rows:
        state_id = str(row["state_id"])
        if state_id not in state_order:
            state_order.append(state_id)
    if len(state_order) != 48:
        raise QualificationError("row-only reducer state identities are incomplete")
    candidates_by_state: dict[str, list[dict[str, Any]]] = {}
    costs_by_source: dict[str, dict[str, np.ndarray]] = {
        source: {} for source in CONTRACT.SOURCE_IDS
    }
    for state_id in state_order:
        candidates: list[dict[str, Any]] = []
        for candidate_index in range(12):
            authority = by_identity[(state_id, candidate_index, "TRUE_FUTURE")]
            post_slew = _validated_applied_action_tape(authority)
            h3 = authority["oracle_route_fields_h3_primary"]
            candidates.append(
                {
                    "state_id": state_id,
                    "family": authority["family"],
                    "role": authority["role"],
                    "candidate_index": candidate_index,
                    "candidate_identity": authority["candidate_identity"],
                    "p_d": h3["p_d"],
                    "p_theta": h3["p_theta"],
                    "completed": h3["completed"],
                    "stuck": h3["stuck"],
                    "oracle_contact": authority["immediate_contact_h1"],
                    "successor_viable": authority["successor_viable"],
                    "oracle_viability_admissible": authority[
                        "oracle_viability_admissible"
                    ],
                    "successor_safe_action_count": authority[
                        "successor_safe_action_count"
                    ],
                    "safe_next_action_count": authority[
                        "successor_safe_action_count"
                    ],
                    "descriptive_contact_h2": authority[
                        "descriptive_contact_h2"
                    ],
                    "descriptive_contact_h3": authority[
                        "descriptive_contact_h3"
                    ],
                    "post_slew": post_slew.tolist(),
                    "goal_body": goals[state_id],
                }
            )
        candidates_by_state[state_id] = candidates
        for source in CONTRACT.SOURCE_IDS:
            costs_by_source[source][state_id] = np.asarray(
                [
                    float(by_identity[(state_id, candidate, source)]["cost_h3"])
                    for candidate in range(12)
                ],
                np.float64,
            )
    comparator_costs: dict[str, dict[str, np.ndarray]] = {
        "KINEMATIC_ROUTE_BASELINE": {},
        "RANDOM": {},
    }
    for state_id in state_order:
        candidates = candidates_by_state[state_id]
        nominal = []
        for candidate in candidates:
            goal = candidate["goal_body"]
            outcome = METRICS.kinematic_nominal_outcome(
                candidate["post_slew"],
                goal[:2],
                route_heading_rad=math.atan2(float(goal[2]), float(goal[3])),
            )
            nominal.append({**candidate, **outcome})
        comparator_costs["KINEMATIC_ROUTE_BASELINE"][state_id] = (
            METRICS.kinematic_rank_costs(nominal)
        )
        random_order = METRICS.deterministic_random_order(
            state_id, list(range(12))
        )
        rank = {candidate: position for position, candidate in enumerate(random_order)}
        comparator_costs["RANDOM"][state_id] = np.asarray(
            [rank[candidate] for candidate in range(12)], np.float64
        )
    all_cost_maps = {
        "TRUE_FUTURE_LATENT_COST": costs_by_source["TRUE_FUTURE"],
        "ONE_STEP_PREDICTED_LATENT_COST": costs_by_source["ONE_STEP_PREDICTED"],
        "TWO_STEP_PREDICTED_LATENT_COST": costs_by_source["TWO_STEP_PREDICTED"],
        **comparator_costs,
    }
    source_summaries: dict[str, Any] = {}
    role_summaries: dict[str, Any] = {}
    for source, costs in all_cost_maps.items():
        source_summaries[source] = METRICS.summarize_source(
            candidates_by_state, costs, source_id=source
        )
        role_summaries[source] = {}
        for role in ("fit", "calibration", "heldout"):
            ids = [
                state_id
                for state_id in state_order
                if candidates_by_state[state_id][0]["role"] == role
            ]
            role_summaries[source][role] = METRICS.summarize_source(
                {state_id: candidates_by_state[state_id] for state_id in ids},
                {state_id: costs[state_id] for state_id in ids},
                source_id=source,
            )
    held = {source: value["heldout"] for source, value in role_summaries.items()}
    true_gate = METRICS.evaluate_true_future_gate(held["TRUE_FUTURE_LATENT_COST"])
    predicted_gate = METRICS.evaluate_predicted_gate(
        true_future_source=held["TRUE_FUTURE_LATENT_COST"],
        one_step_source=held["ONE_STEP_PREDICTED_LATENT_COST"],
        two_step_source=held["TWO_STEP_PREDICTED_LATENT_COST"],
        true_future_gate=true_gate,
    )
    paired_inputs = {
        "TWO_STEP_MINUS_ONE_STEP": (
            "TWO_STEP_PREDICTED_LATENT_COST",
            "ONE_STEP_PREDICTED_LATENT_COST",
        ),
        "TWO_STEP_MINUS_KINEMATIC": (
            "TWO_STEP_PREDICTED_LATENT_COST",
            "KINEMATIC_ROUTE_BASELINE",
        ),
        "TWO_STEP_MINUS_TRUE_FUTURE": (
            "TWO_STEP_PREDICTED_LATENT_COST",
            "TRUE_FUTURE_LATENT_COST",
        ),
        "TRUE_FUTURE_MINUS_KINEMATIC": (
            "TRUE_FUTURE_LATENT_COST",
            "KINEMATIC_ROUTE_BASELINE",
        ),
    }
    paired = {
        identity: METRICS.paired_source_comparison(held[left], held[right])
        for identity, (left, right) in paired_inputs.items()
    }
    descriptive_one = METRICS.paired_source_comparison(
        held["ONE_STEP_PREDICTED_LATENT_COST"], held["KINEMATIC_ROUTE_BASELINE"]
    )
    reverse = METRICS.paired_source_comparison(
        held["KINEMATIC_ROUTE_BASELINE"], held["TWO_STEP_PREDICTED_LATENT_COST"]
    )
    classification = METRICS.classify_qualification(
        true_future_gate=true_gate,
        predicted_gate=predicted_gate,
        jepa_vs_kinematic=paired["TWO_STEP_MINUS_KINEMATIC"],
        kinematic_vs_jepa=reverse,
    )
    selection_rows: list[dict[str, Any]] = []
    selection_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source, summary in source_summaries.items():
        for population in CONTRACT.POPULATION_IDS:
            for state in summary["populations"][population]["per_state"]:
                selection = {
                    "schema": "jepa_local_waypoint_selection_evidence_v1",
                    "state_id": state["state_id"],
                    "family": state["family"],
                    "role": state["role"],
                    "source_or_comparator": source,
                    "population": population,
                    "eligible_candidate_indices": state.get(
                        "eligible_candidate_indices", state["cost_order"]
                    ),
                    "ranked_candidate_indices": state["cost_order"],
                    "selected_candidate_index": state["selected_candidate_index"],
                    "oracle_best_candidate_index": state[
                        "oracle_best_candidate_index"
                    ],
                    "abstained": state["abstained"],
                    "selected_immediate_contact_h1": state[
                        "selected_immediate_contact_h1"
                    ],
                    "selected_descriptive_contact_h2": state[
                        "selected_descriptive_contact_h2"
                    ],
                    "selected_descriptive_contact_h3": state[
                        "selected_descriptive_contact_h3"
                    ],
                    "selected_nonviable_successor": state[
                        "selected_nonviable_successor"
                    ],
                    "selected_stuck": state["selected_stuck"],
                    "selected_progress_m": state["selected_progress_m"],
                    "selected_heading_improvement_rad": state[
                        "selected_heading_improvement_rad"
                    ],
                    "selected_combined_utility": state["selected_combined_utility"],
                    "oracle_best_progress_m": state["oracle_best_progress_m"],
                    "normalised_regret": state["normalised_regret"],
                    "best_route_rank": state["best_route_rank"],
                    "completed": state["completed"],
                    "cost_spread": state["cost_spread"],
                    "cost_tie_pair_count": state["cost_tie_pair_count"],
                    "cost_tie_pair_rate": state["cost_tie_pair_rate"],
                    "ordered_pair_count": state["ordered_pair_count"],
                    "pairwise_denominator_ordered_pairs": state[
                        "pairwise_denominator_ordered_pairs"
                    ],
                    "pairwise_correct_credit": state["pairwise_correct_credit"],
                    "best_route_reciprocal_rank": state[
                        "best_route_reciprocal_rank"
                    ],
                }
                identity = (source, population, str(state["state_id"]))
                selection_map[identity] = selection
                selection_rows.append(selection)
    paired_rows: list[dict[str, Any]] = []
    for comparison_id in CONTRACT.PAIRED_COMPARISON_IDS:
        left, right = paired_inputs[comparison_id]
        for row in paired[comparison_id]["per_state"]:
            state_id = str(row["state_id"])
            left_selection = selection_map[
                (left, "ORACLE_VIABILITY_ADMISSIBLE", state_id)
            ]
            right_selection = selection_map[
                (right, "ORACLE_VIABILITY_ADMISSIBLE", state_id)
            ]
            paired_rows.append(
                {
                    "schema": "jepa_local_waypoint_paired_effect_evidence_v1",
                    "comparison_id": comparison_id,
                    "state_id": state_id,
                    "family": row["family"],
                    "role": left_selection["role"],
                    "population": "ORACLE_VIABILITY_ADMISSIBLE",
                    "left_source_or_comparator": left,
                    "right_source_or_comparator": right,
                    "left_selected_candidate_index": row[
                        "candidate_selected_candidate_index"
                    ],
                    "right_selected_candidate_index": row[
                        "comparator_selected_candidate_index"
                    ],
                    "left_selected_progress_m": row["candidate_selected_progress_m"],
                    "right_selected_progress_m": row[
                        "comparator_selected_progress_m"
                    ],
                    "selected_progress_effect_m": row["selected_progress_delta_m"],
                    "left_normalised_regret": row["candidate_normalized_regret"],
                    "right_normalised_regret": row["comparator_normalized_regret"],
                    "normalised_regret_improvement": row[
                        "normalized_regret_reduction"
                    ],
                    "left_pairwise_accuracy": row["candidate_pairwise_accuracy"],
                    "right_pairwise_accuracy": row["comparator_pairwise_accuracy"],
                    "pairwise_accuracy_effect": row["pairwise_accuracy_delta"],
                    "left_best_route_rank": row["candidate_best_route_rank"],
                    "right_best_route_rank": row["comparator_best_route_rank"],
                    "best_route_rank_improvement": row[
                        "best_route_rank_improvement"
                    ],
                    "left_selected_immediate_contact_h1": left_selection[
                        "selected_immediate_contact_h1"
                    ],
                    "right_selected_immediate_contact_h1": right_selection[
                        "selected_immediate_contact_h1"
                    ],
                    "contact_selection_delta": int(
                        left_selection["selected_immediate_contact_h1"]
                    )
                    - int(right_selection["selected_immediate_contact_h1"]),
                    "left_selected_nonviable_successor": left_selection[
                        "selected_nonviable_successor"
                    ],
                    "right_selected_nonviable_successor": right_selection[
                        "selected_nonviable_successor"
                    ],
                    "nonviable_selection_delta": int(
                        left_selection["selected_nonviable_successor"]
                    )
                    - int(right_selection["selected_nonviable_successor"]),
                }
            )
    latent_progress = {
        source: METRICS.latent_progress_diagnostics(
            [row for row in rows if row["source"] == source], source_id=source
        )
        for source in CONTRACT.SOURCE_IDS
    }
    tendencies = {
        source: METRICS.all_candidate_tendency_diagnostics(
            candidates_by_state, costs_by_source[source], source_id=source
        )
        for source in CONTRACT.SOURCE_IDS
    }
    aggregates = attach_digest(
        {
            **_phase_core(
                "jepa_local_waypoint_planning_cost_metrics_v1", source_freeze_commit
            ),
            "by_source_population": {
                source: value["populations"] for source, value in source_summaries.items()
            },
            "by_source_population_family": {
                source: {
                    population: value["populations"][population]["per_family"]
                    for population in CONTRACT.POPULATION_IDS
                }
                for source, value in source_summaries.items()
            },
            "per_state": {
                source: {
                    population: value["populations"][population]["per_state"]
                    for population in CONTRACT.POPULATION_IDS
                }
                for source, value in source_summaries.items()
            },
            "per_role": role_summaries,
            "latent_progress_diagnostics": latent_progress,
            "all_candidate_tendency_diagnostics": tendencies,
            "paired_comparisons": {
                **paired,
                "ONE_STEP_MINUS_KINEMATIC_DESCRIPTIVE": descriptive_one,
                "KINEMATIC_MINUS_TWO_STEP_DESCRIPTIVE": reverse,
            },
            "bootstrap": {
                "replicates": CONTRACT.BOOTSTRAP_REPLICATES,
                "seed": CONTRACT.SEED,
                "descriptive_only": True,
            },
            "family_collapse": {
                source: held[source]["populations"]["ORACLE_VIABILITY_ADMISSIBLE"][
                    "collapsed_families"
                ]
                for source in held
            },
            "gates": {"true_future": true_gate, "two_step_predicted": predicted_gate},
            "classification": classification,
            "row_reproduction": {
                "candidate_rows": 1728,
                "selection_rows": 720,
                "paired_effect_rows": 32,
                "from_persisted_tensors_without_inference": True,
                "cost_row_to_aggregate_without_tensor_open": True,
            },
        }
    )
    _validate_evidence_rows(rows, selection_rows, paired_rows)
    _validate_schema_value("aggregate_metrics", aggregates)
    return rows, selection_rows, paired_rows, aggregates, classification


def _actual_storage(output_root: Path) -> dict[str, Any]:
    files = [path for path in output_root.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    peak_rss_bytes = max(
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024),
    )
    return {
        "files": len(files),
        "bytes": total,
        "gb_decimal": total / 1e9,
        "final_ceiling_bytes": 12 * 10**9,
        "within_final_ceiling": total <= 12 * 10**9,
        "peak_rss_bytes": peak_rss_bytes,
    }


def _validate_schema_value(file_id: str, value: Mapping[str, Any]) -> None:
    spec = CONTRACT.build_output_schema()["files"][file_id]
    missing = sorted(set(spec.get("required_keys", ())) - set(value))
    if missing:
        raise QualificationError(f"{file_id} lacks required keys {missing}")
    if spec.get("schema") is not None and value.get("schema") != spec["schema"]:
        raise QualificationError(f"{file_id} schema mismatch")
    validate_digest(
        value,
        "result_content_sha256" if file_id == "result" else "content_digest",
    )
    if file_id == "preexecution_receipt":
        custody = value.get("preexecution_custody")
        if not isinstance(custody, Mapping) or set(custody) != set(
            spec["preexecution_custody_required_keys"]
        ):
            raise QualificationError("preexecution custody schema drift")
        if custody.get("contract_disclosure") != spec[
            "preexecution_contract_disclosure_exact"
        ]:
            raise QualificationError("preexecution contract disclosure drift")
        live = custody.get("live_validation")
        if not isinstance(live, Mapping) or set(live) != set(
            spec["preexecution_live_validation_required_keys"]
        ):
            raise QualificationError("preexecution live-validation schema drift")
        if any(
            live.get(key) != expected
            for key, expected in spec[
                "preexecution_live_validation_exact"
            ].items()
        ) or live.get("fixture_checks_passed") != len(
            CONTRACT.build_fixture_receipt()["executed_checks"]
        ):
            raise QualificationError("preexecution live-validation custody drift")
    if file_id == "environment_receipt":
        if value.get("interpreter_bindings") != spec[
            "interpreter_bindings_exact"
        ]:
            raise QualificationError("combined environment interpreter binding drift")
        cpu = value.get("package_inventory", {}).get("cpu", {})
        expected_cpu_entrypoint = str(CPU_INTERPRETER.absolute())
        if cpu.get("executable") != expected_cpu_entrypoint:
            raise QualificationError("CPU venv interpreter entrypoint drift")
    required_record = set(spec.get("record_required_keys", ()))
    if required_record:
        records = value.get("records")
        if not isinstance(records, list):
            raise QualificationError(f"{file_id} records are absent")
        for index, row in enumerate(records):
            absent = sorted(required_record - set(row))
            if absent:
                raise QualificationError(
                    f"{file_id} record {index} lacks required keys {absent}"
                )
    if file_id == "cpu_runtime_input_inventory":
        if value.get("prefreeze_scene_byte_inventory_binding") != (
            CONTRACT.SCENE_INPUT_BYTE_INVENTORY_BINDING
        ):
            raise QualificationError("CPU prefreeze scene binding drift")
        prefreeze_scenes = value.get("prefreeze_scene_byte_inventory_records")
        prefreeze_required = set(
            spec["prefreeze_scene_byte_inventory_record_required_keys"]
        )
        if (
            not isinstance(prefreeze_scenes, list)
            or len(prefreeze_scenes) != 96
            or any(set(row) != prefreeze_required for row in prefreeze_scenes)
        ):
            raise QualificationError("CPU prefreeze scene inventory schema drift")
        regenerated_scene_validation = _validate_prefreeze_byte_inventory(
            prefreeze_scenes,
            CONTRACT.SCENE_INPUT_BYTE_INVENTORY_BINDING,
            states=STATE_COUNT,
        )
        if value.get("prefreeze_scene_byte_inventory_validation") != (
            regenerated_scene_validation
        ):
            raise QualificationError("CPU prefreeze scene validation drift")
        scene_required = set(spec["scene_record_required_keys"])
        scenes = value.get("scene_records")
        if not isinstance(scenes, list) or len(scenes) != 48:
            raise QualificationError("CPU runtime scene inventory cardinality drift")
        if len({str(row.get("state_id")) for row in scenes}) != 48 or len(
            {str(row.get("scene_dir")) for row in scenes}
        ) != 48:
            raise QualificationError("CPU runtime scene identities are not unique")
        for index, row in enumerate(scenes):
            absent = sorted(scene_required - set(row))
            if absent:
                raise QualificationError(
                    f"CPU runtime scene record {index} lacks keys {absent}"
                )
            if (
                row.get("identity_and_internal_digest_consistent") is not True
                or row.get("historical_floor_only_renderer") is not True
                or row.get("historical_builder_structural_keys_present") != []
            ):
                raise QualificationError(
                    f"CPU runtime scene record {index} renderer custody drift"
                )
        textures = value.get("texture_records")
        if not isinstance(textures, list) or len(textures) != 12:
            raise QualificationError("CPU runtime texture inventory cardinality drift")
        texture_required = set(spec["texture_record_required_keys"])
        if any(not texture_required.issubset(row) for row in textures):
            raise QualificationError("CPU runtime texture record schema drift")
        urdf = value.get("genesis_builtin_urdf")
        if not isinstance(urdf, Mapping) or not set(
            spec["genesis_urdf_required_keys"]
        ).issubset(urdf):
            raise QualificationError("CPU runtime URDF binding schema drift")
        meshes = urdf["referenced_meshes"]
        mesh_required = set(spec["genesis_mesh_record_required_keys"])
        if (
            not isinstance(meshes, list)
            or len(meshes) != 7
            or any(not mesh_required.issubset(row) for row in meshes)
        ):
            raise QualificationError("CPU runtime URDF mesh inventory drift")
        if value.get("counts") != spec["counts_exact"]:
            raise QualificationError("CPU runtime exact counts drift")
        box = value.get("box_obj_cache")
        expected_box = spec["box_obj_cache_exact"]
        if not isinstance(box, Mapping) or {
            key: box.get(key) for key in expected_box
        } != expected_box:
            raise QualificationError("CPU runtime nonexecuted OBJ-cache custody drift")
        if value.get("outcome_fields_read") != spec["outcome_fields_read_exact"]:
            raise QualificationError("CPU runtime inventory accessed outcome fields")
        if value.get("package_record_closure_algorithm") != (
            CONTRACT.CPU_RUNTIME_INPUT_BINDINGS["package_record_closure_algorithm"]
        ):
            raise QualificationError("CPU package RECORD algorithm custody drift")
        package_roots = value.get("package_roots")
        expected_packages = CONTRACT.CPU_RUNTIME_INPUT_BINDINGS["cpu_packages"]
        if not isinstance(package_roots, Mapping) or set(package_roots) != set(
            expected_packages
        ):
            raise QualificationError("CPU package RECORD closure domain drift")
        for package_id, expected_package in expected_packages.items():
            package = package_roots[package_id]
            if (
                package.get("distribution") != expected_package["distribution"]
                or package.get("version") != expected_package["version"]
                or package.get("package_root")
                != str(Path(expected_package["package_root"]).resolve())
                or package.get("record_closure")
                != expected_package["record_closure"]
                or package.get("all_declared_hashes_and_sizes_valid") is not True
                or package.get("all_present_unhashed_rows_directly_hashed") is not True
                or package.get("absent_rows_all_unhashed_and_exact") is not True
                or package.get("pass") is not True
            ):
                raise QualificationError(
                    f"CPU package RECORD closure receipt drift: {package_id}"
                )
            resolution = package.get("import_resolution")
            expected_root = Path(expected_package["package_root"]).resolve()
            if (
                not isinstance(resolution, Mapping)
                or resolution.get("import_name") != package_id
                or Path(str(resolution.get("find_spec_origin"))).resolve()
                .is_relative_to(expected_root)
                is not True
                or Path(str(resolution.get("live_module_file"))).resolve()
                .is_relative_to(expected_root)
                is not True
                or any(
                    not Path(str(path)).resolve().is_relative_to(expected_root)
                    for path in resolution.get("submodule_search_locations", [])
                )
                or resolution.get("resolved_inside_frozen_package_root") is not True
                or resolution.get("pass") is not True
            ):
                raise QualificationError(
                    f"CPU package runtime import-resolution drift: {package_id}"
                )
        foundational = value.get("foundational_package_roots")
        expected_foundational = CONTRACT.CPU_FOUNDATIONAL_PACKAGE_BINDINGS
        if not isinstance(foundational, Mapping) or set(foundational) != set(
            expected_foundational
        ):
            raise QualificationError("CPU foundational package domain drift")
        for package_id, expected_package in expected_foundational.items():
            package = foundational[package_id]
            expected_root = Path(expected_package["package_root"]).resolve()
            resolution = package.get("import_resolution")
            if (
                {key: package.get(key) for key in expected_package}
                != {
                    **expected_package,
                    "package_root": str(expected_root),
                }
                or not isinstance(resolution, Mapping)
                or resolution.get("import_name") != expected_package["import_name"]
                or not Path(str(resolution.get("find_spec_origin"))).resolve()
                .is_relative_to(expected_root)
                or not Path(str(resolution.get("live_module_file"))).resolve()
                .is_relative_to(expected_root)
                or any(
                    not Path(str(path)).resolve().is_relative_to(expected_root)
                    for path in resolution.get("submodule_search_locations", [])
                )
                or resolution.get("expected_package_root") != str(expected_root)
                or resolution.get("resolved_inside_frozen_package_root") is not True
                or resolution.get("pass") is not True
            ):
                raise QualificationError(
                    f"CPU foundational package import-resolution drift: {package_id}"
                )
        if value.get("foundational_package_closure_policy") != (
            CONTRACT.FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
        ):
            raise QualificationError("CPU foundational package closure-policy drift")
        if value.get("historical_renderer_limitations") != (
            CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
        ):
            raise QualificationError("CPU historical renderer limitation drift")
        if value.get("renderer_structure_schema_audit") != spec[
            "renderer_structure_schema_audit_exact"
        ]:
            raise QualificationError("CPU renderer structure-schema audit drift")
        selection = value.get("texture_selection_validation")
        if selection != {
            "scene_records": 48,
            "selected_paths_in_frozen_12_file_inventory": True,
            "reached_categories": ["floor"],
            "nonexecuted_categories": ["wall", "obstacle"],
            "pass": True,
        }:
            raise QualificationError("CPU texture-selection custody drift")
        limitation = value.get("historical_renderer_limitation_validation")
        if (
            not isinstance(limitation, Mapping)
            or any(
                limitation.get(key) != expected
                for key, expected in CONTRACT.HISTORICAL_RENDERER_LIMITATIONS.items()
            )
            or limitation.get("scenes_checked") != 48
            or limitation.get("builder_structural_keys_present") != 0
            or limitation.get("box_obj_files_opened_or_used") != 0
            or limitation.get("pass") is not True
        ):
            raise QualificationError("CPU historical renderer limitation custody drift")
    if file_id == "dense_route_replay_input_index":
        if value.get("prefreeze_byte_inventory_binding") != (
            CONTRACT.DENSE_ROUTE_REPLAY_INPUT_BINDINGS
        ):
            raise QualificationError("dense replay prefreeze byte binding drift")
        prefreeze_records = value.get("prefreeze_byte_inventory_records")
        required = set(spec["prefreeze_byte_inventory_record_required_keys"])
        if (
            not isinstance(prefreeze_records, list)
            or len(prefreeze_records) != STATE_COUNT
            or any(set(row) != required for row in prefreeze_records)
        ):
            raise QualificationError("dense replay prefreeze inventory schema drift")
        regenerated = _validate_prefreeze_byte_inventory(
            prefreeze_records,
            CONTRACT.DENSE_ROUTE_REPLAY_INPUT_BINDINGS,
            states=None,
        )
        if value.get("prefreeze_byte_inventory_validation") != regenerated:
            raise QualificationError("dense replay prefreeze validation drift")
    if file_id == "gpu_environment_receipt":
        interpreter = value.get("interpreter_binding")
        device = value.get("device")
        packages = value.get("packages")
        if (
            not isinstance(interpreter, Mapping)
            or not set(spec["interpreter_binding_required_keys"]).issubset(
                interpreter
            )
            or not isinstance(device, Mapping)
            or not set(spec["device_required_keys"]).issubset(device)
            or not isinstance(packages, Mapping)
            or set(packages) != set(spec["package_ids"])
            or value.get("import_closure") != spec["import_closure_exact"]
        ):
            raise QualificationError("GPU environment nested schema drift")
        if interpreter != spec["interpreter_binding_exact"]:
            raise QualificationError("GPU resolved interpreter binding drift")
        expected_gpu_entrypoint = CONTRACT.build_contract()["execution"][
            "environments"
        ]["encoder_predictor"]["interpreter"]
        if value.get("executable") != expected_gpu_entrypoint:
            raise QualificationError("GPU venv interpreter entrypoint drift")
        for key, expected in spec["preinference_exact"].items():
            if value.get(key) != expected:
                raise QualificationError(
                    f"GPU preinference environment custody drift: {key}"
                )
        expected_gpu_environment = CONTRACT.build_contract()["execution"][
            "environments"
        ]["encoder_predictor"]
        if value.get("torch") != expected_gpu_environment["torch"] or any(
            packages.get(package_id) != expected_gpu_environment[package_id]
            for package_id in spec["package_ids"]
        ):
            raise QualificationError("GPU foundational package version drift")
        foundational = value.get("foundational_package_roots")
        expected_foundational = CONTRACT.GPU_FOUNDATIONAL_PACKAGE_BINDINGS
        if not isinstance(foundational, Mapping) or set(foundational) != set(
            expected_foundational
        ):
            raise QualificationError("GPU foundational package domain drift")
        for package_id, expected_package in expected_foundational.items():
            package = foundational[package_id]
            expected_root = Path(expected_package["package_root"]).resolve()
            resolution = package.get("import_resolution")
            if (
                {key: package.get(key) for key in expected_package}
                != {**expected_package, "package_root": str(expected_root)}
                or not isinstance(resolution, Mapping)
                or resolution.get("import_name") != expected_package["import_name"]
                or not Path(str(resolution.get("find_spec_origin"))).resolve()
                .is_relative_to(expected_root)
                or not Path(str(resolution.get("live_module_file"))).resolve()
                .is_relative_to(expected_root)
                or any(
                    not Path(str(path)).resolve().is_relative_to(expected_root)
                    for path in resolution.get("submodule_search_locations", [])
                )
                or resolution.get("expected_package_root") != str(expected_root)
                or resolution.get("resolved_inside_frozen_package_root") is not True
                or resolution.get("pass") is not True
            ):
                raise QualificationError(
                    f"GPU foundational package import-resolution drift: {package_id}"
                )
        if value.get("foundational_package_closure_policy") != (
            CONTRACT.FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
        ):
            raise QualificationError("GPU foundational package closure-policy drift")
    if file_id == "gpu_inference_receipt":
        expected_gpu_entrypoint = CONTRACT.build_contract()["execution"][
            "environments"
        ]["encoder_predictor"]["interpreter"]
        if (
            value.get("interpreter") != CONTRACT.INTERPRETER_BINARY_BINDING
            or value.get("interpreter_entrypoint") != expected_gpu_entrypoint
        ):
            raise QualificationError("GPU inference interpreter custody drift")
        if value.get("gpu_environment_revalidation") != spec[
            "gpu_environment_revalidation_exact"
        ]:
            raise QualificationError("GPU inference environment-revalidation drift")
        if value.get("gpu_watchdog_status") != spec[
            "gpu_watchdog_status_exact"
        ]:
            raise QualificationError("GPU inference watchdog-status drift")
    if file_id == "aggregate_metrics":
        source_ids = set(spec["source_or_comparator_ids"])
        population_ids = set(spec["population_ids"])
        role_ids = set(spec["role_ids"])
        family_ids = set(spec["family_ids"])
        if set(value["by_source_population"]) != source_ids or set(
            value["by_source_population_family"]
        ) != source_ids or set(value["per_state"]) != source_ids or set(
            value["per_role"]
        ) != source_ids:
            raise QualificationError("aggregate source/comparator axis drift")
        classification = value["classification"]
        missing_classification = sorted(
            set(spec["classification_required_keys"]) - set(classification)
        )
        if missing_classification:
            raise QualificationError(
                f"aggregate classification lacks {missing_classification}"
            )
        if set(classification["predicted_base_screens"]) != set(
            spec["classification_predicted_base_screen_keys"]
        ) or set(classification["diagnostic_flags"]) != set(
            spec["classification_diagnostic_flag_keys"]
        ):
            raise QualificationError("aggregate classification diagnostic schema drift")
        group_required = set(spec["metric_group_required_keys"])
        for source, populations in value["by_source_population"].items():
            if set(populations) != population_ids:
                raise QualificationError(
                    f"aggregate population axis drift for {source}"
                )
            for population, summary in populations.items():
                absent = sorted(group_required - set(summary["aggregate"]))
                if absent:
                    raise QualificationError(
                        f"aggregate metric group {source}/{population} lacks {absent}"
                    )
        for source in source_ids:
            if set(value["per_state"][source]) != population_ids or set(
                value["by_source_population_family"][source]
            ) != population_ids or set(value["per_role"][source]) != role_ids:
                raise QualificationError(
                    f"aggregate population/role axis drift for {source}"
                )
            for population in population_ids:
                if set(
                    value["by_source_population_family"][source][population]
                ) != family_ids:
                    raise QualificationError(
                        f"aggregate family axis drift for {source}/{population}"
                    )
    if file_id == "context_reconstruction_index":
        if value.get("reconstruction_prefix_custody") != (
            CONTRACT.RECONSTRUCTION_PREFIX_CUSTODY
        ):
            raise QualificationError("context reconstruction-prefix custody drift")
        action_validation = value.get(
            "requested_applied_action_authority_validation"
        )
        if action_validation != {
            "states": 48,
            "branch_ledger_rows": 576,
            "requested_candidate_bank_exact": 576,
            "applied_branch_ledger_exact": 576,
            "active_projection_exact": 576,
            "requested_and_applied_never_conflated": True,
            "pass": True,
        }:
            raise QualificationError("context action-authority aggregate drift")
        for row in value["records"]:
            requested = np.asarray(
                row["requested_action_blocks_raw_3x5x3_by_candidate"],
                dtype=np.float32,
            )
            applied = np.asarray(
                row["applied_action_blocks_raw_3x5x3_by_candidate"],
                dtype=np.float32,
            )
            active = np.asarray(
                row["action_blocks_raw_3x5x2_by_candidate"], dtype=np.float32
            )
            flat = np.asarray(
                row["action_blocks_raw_3x10_by_candidate"], dtype=np.float32
            )
            authority = row["requested_applied_action_authority_validation"]
            if (
                requested.shape != (12, 3, 5, 3)
                or applied.shape != (12, 3, 5, 3)
                or active.shape != (12, 3, 5, 2)
                or flat.shape != (12, 3, 10)
                or not np.array_equal(active, applied[:, :, :, (0, 2)])
                or not np.array_equal(flat, active.reshape(12, 3, 10))
                or authority.get("candidates") != 12
                or authority.get("pass") is not True
            ):
                raise QualificationError("context requested/applied action custody drift")
    if file_id == "oracle_admissibility_fanout_index":
        if value.get("cpu_watchdog_status") != spec["cpu_watchdog_status_exact"]:
            raise QualificationError("CPU fanout watchdog-status drift")
    if file_id == "result":
        nested = (
            ("materialisation_counts", "materialisation_count_required_keys"),
            ("goal_view_counts", "goal_view_count_required_keys"),
            ("runtime_s", "runtime_required_keys"),
            ("storage", "storage_required_keys"),
        )
        for value_key, specification_key in nested:
            child = value.get(value_key)
            if not isinstance(child, Mapping):
                raise QualificationError(f"result {value_key} is not an object")
            absent = sorted(set(spec[specification_key]) - set(child))
            if absent:
                raise QualificationError(
                    f"result {value_key} lacks required keys {absent}"
                )
        limitation = value.get("historical_renderer_limitations")
        limitation_required = set(
            spec["historical_renderer_limitation_required_keys"]
        )
        if (
            not isinstance(limitation, Mapping)
            or not limitation_required.issubset(limitation)
            or dict(limitation) != CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
        ):
            raise QualificationError("result historical renderer limitation drift")
        if value.get("reconstruction_prefix_custody") != (
            CONTRACT.RECONSTRUCTION_PREFIX_CUSTODY
        ):
            raise QualificationError("result reconstruction-prefix custody drift")
        if value.get("controller_execution_custody") != (
            CONTRACT.CONTROLLER_EXECUTION_CUSTODY
        ):
            raise QualificationError("result controller-execution custody drift")
        if value.get("execution_watchdog_status") != spec[
            "execution_watchdog_status_exact"
        ]:
            raise QualificationError("result execution-watchdog status drift")
    if file_id == "persistence_receipt":
        if value.get("execution_watchdog_status") != spec[
            "execution_watchdog_status_exact"
        ]:
            raise QualificationError("persistence execution-watchdog status drift")


def _validate_evidence_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> None:
    schema = CONTRACT.build_output_schema()["files"]
    groups = (
        ("candidate_evidence", candidate_rows, 1728),
        ("selection_evidence", selection_rows, 720),
        ("paired_effect_evidence", paired_rows, 32),
    )
    for file_id, rows, expected in groups:
        spec = schema[file_id]
        if len(rows) != expected:
            raise QualificationError(f"{file_id} row count {len(rows)} != {expected}")
        required = set(spec["required_keys"])
        for position, row in enumerate(rows):
            missing = sorted(required - set(row))
            if missing or row.get("schema") != spec["schema"]:
                raise QualificationError(
                    f"{file_id} row {position} schema/keys invalid: {missing}"
                )
    if len(
        {(row["state_id"], row["candidate_index"], row["source"]) for row in candidate_rows}
    ) != 1728:
        raise QualificationError("candidate evidence identities are not unique")
    if len(
        {
            (row["state_id"], row["source_or_comparator"], row["population"])
            for row in selection_rows
        }
    ) != 720:
        raise QualificationError("selection evidence identities are not unique")
    paired_ids = {(row["comparison_id"], row["state_id"]) for row in paired_rows}
    if len(paired_ids) != 32 or {
        row["comparison_id"] for row in paired_rows
    } != set(CONTRACT.PAIRED_COMPARISON_IDS):
        raise QualificationError("paired-effect identities are incomplete")
    horizon_required = set(
        schema["candidate_evidence"]["realised_route_horizon_required_keys"]
    )
    for row in candidate_rows:
        horizons = row["realised_route_fields_by_horizon"]
        if set(horizons) != {"H1", "H2", "H3"} or any(
            set(value) != horizon_required for value in horizons.values()
        ):
            raise QualificationError("candidate realised-route horizon schema drift")
        requested = np.asarray(
            row["requested_action_blocks_raw_3x5x3"], dtype=np.float32
        )
        applied = np.asarray(
            row["applied_action_blocks_raw_3x5x3"], dtype=np.float32
        )
        active = np.asarray(row["action_blocks_raw_3x5x2"], dtype=np.float32)
        flat = np.asarray(row["action_blocks_raw_3x10"], dtype=np.float32)
        authority = row["requested_applied_action_authority_validation"]
        if (
            requested.shape != (3, 5, 3)
            or applied.shape != (3, 5, 3)
            or active.shape != (3, 5, 2)
            or flat.shape != (3, 10)
            or not np.array_equal(active, applied[:, :, (0, 2)])
            or not np.array_equal(flat, active.reshape(3, 10))
            or authority.get("candidate_index") != int(row["candidate_index"])
            or authority.get("pass") is not True
        ):
            raise QualificationError("candidate requested/applied action custody drift")


def _binding(relative: Path, output_root: Path) -> dict[str, Any]:
    value = load_json(output_root / relative)
    return {
        "path": str(relative),
        "sha256": sha256_file(output_root / relative),
        "bytes": (output_root / relative).stat().st_size,
        "content_digest": value["content_digest"],
    }


EXPERIMENT_SCRIPT_NAMES = {
        "evaluate_jepa_local_waypoint_planning_cost_qualification_v1.py",
        "run_jepa_local_waypoint_planning_cost_inference_v1.py",
}


def _argv_is_experiment(arguments: Sequence[str]) -> bool:
    return any(Path(argument).name in EXPERIMENT_SCRIPT_NAMES for argument in arguments)


def _active_experiment_processes() -> list[int]:
    active: list[int] = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        pid = int(cmdline.parent.name)
        if pid == os.getpid():
            continue
        try:
            arguments = [
                value.decode("utf-8", errors="surrogateescape")
                for value in cmdline.read_bytes().split(b"\x00")
                if value
            ]
        except OSError:
            continue
        if _argv_is_experiment(arguments):
            active.append(pid)
    return sorted(active)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _markdown_report(result: Mapping[str, Any], aggregates: Mapping[str, Any]) -> str:
    lines = [
        "# JEPA Local Waypoint Planning Cost Qualification V1",
        "",
        "## Bindings and custody",
        "",
        f"Source-freeze commit: `{result['source_freeze_commit']}`; seed `{result['seed']}`. Development-only and non-claim-bearing.",
        "",
        "Genesis, rsl_rl, and tensordict are closed by exact dist-info RECORD plus every listed present-file digest. Torch, NumPy, SciPy, Pillow, and PyYAML are bound to the exact interpreter, version, package root, and live import resolution; residual same-version mutation in those foundational packages is explicitly not byte-closed.",
        "",
        "## Goal-view and predictor input reconstruction",
        "",
        f"`{json.dumps(result['materialisation_counts'], sort_keys=True)}`",
        "",
        "All 48 states use replay boundaries 38/39/40, source-frame offsets -480/-240/0, command-tick offsets -10/-5/0, and one candidate-independent path[2] goal view. Current RGB and raw FP16 token authority reproduce exactly.",
        "",
        "The `FROZEN_STATE_RECONSTRUCTION_REPLAY` invokes the production collector scheduler/RouteTeacher and frozen PPO for exactly 48×40 prefix blocks solely to reproduce the committed post-block-40 state identities. It performs no experimental candidate selection, executes zero JEPA-cost actions, and is not a navigation qualification.",
        "",
        "## Historical renderer limitation",
        "",
        "The frozen renderer receives `genesis_scene.json`, where structural geometry is under `objects`, while its historical builder reads only top-level `walls`, `obstacles`, and `landmarks`. Those keys are absent, so the effective rendered scene geometry is the textured floor plane only. This is preserved to require byte-identical current/true-future token compatibility; the experiment makes no explicit wall or landmark visual-reasoning claim.",
        "",
        f"Renderer limitation receipt: `{json.dumps(result['historical_renderer_limitations'], sort_keys=True)}`.",
        "",
        "## Population and materialisation counts",
        "",
        f"States/candidates/tensors/rows: `{json.dumps(result['materialisation_counts'], sort_keys=True)}`.",
        "",
        "## Controller execution custody",
        "",
        "The frozen PPO/controller executed 6,240 blocks (1,920 deterministic prefix-reconstruction blocks plus 4,320 fixed open-loop candidate/fanout blocks), totaling 1,560,000 physics frames. It was not trained or qualified here, made no experimental candidate selection, and no JEPA/MPC/navigation planner executed actions.",
        "",
        f"Controller custody receipt: `{json.dumps(result['controller_execution_custody'], sort_keys=True)}`.",
        "",
        "## True-future and predicted ranking metrics",
        "",
        "| source | population | n | pairwise | Spearman | regret | top-3 | progress ratio | contact | nonviable | stuck |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    per_role = aggregates["per_role"]
    sources = (
        "TRUE_FUTURE_LATENT_COST",
        "ONE_STEP_PREDICTED_LATENT_COST",
        "TWO_STEP_PREDICTED_LATENT_COST",
        "KINEMATIC_ROUTE_BASELINE",
        "RANDOM",
    )
    for source in sources:
        for population in CONTRACT.POPULATION_IDS:
            row = per_role[source]["heldout"]["populations"][population]["aggregate"]
            values = (
                source,
                population,
                row["states"],
                row["pairwise_accuracy"],
                row["spearman_rho"],
                row["normalized_regret"],
                row["best_route_top3_rate"],
                row["selected_progress_ratio"],
                row["selected_immediate_contacts_h1"],
                row["selected_nonviable_successors"],
                row["selected_stuck"],
            )
            lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Per-family results and collapse audit",
            "",
            "| source | family | n | pairwise | regret | top-3 | progress m | collapsed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source in sources:
        family_rows = per_role[source]["heldout"]["populations"][
            "ORACLE_VIABILITY_ADMISSIBLE"
        ]["per_family"]
        for family in CONTRACT.FAMILY_IDS:
            row = family_rows[family]
            values = (
                source,
                family,
                row["states"],
                row["pairwise_accuracy"],
                row["normalized_regret"],
                row["best_route_top3_rate"],
                row["selected_route_progress_m_sum"],
                row["complete_family_collapse"],
            )
            lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Selected route outcomes",
            "",
            "The held-out table reports selected immediate-contact, successor-nonviable and stuck decisions for every source/comparator and population; all 720 selections are persisted row-wise.",
            "",
            "## Paired comparisons and descriptive bootstrap",
            "",
        ]
    )
    for comparison_id in CONTRACT.PAIRED_COMPARISON_IDS:
        comparison = aggregates["paired_comparisons"][comparison_id]
        progress = comparison["mean_route_progress_gain_m"]
        regret = comparison["normalized_regret_reduction"]
        lines.append(
            f"- `{comparison_id}`: progress {_fmt(progress['point'])} m "
            f"(95% {_fmt(progress['bootstrap_lower_95'])}..{_fmt(progress['bootstrap_upper_95'])}); "
            f"regret improvement {_fmt(regret['point'])} "
            f"(95% {_fmt(regret['bootstrap_lower_95'])}..{_fmt(regret['bootstrap_upper_95'])}); "
            f"material={_fmt(comparison['material_improvement'])}."
        )
    lines.extend(["", "## Latent progression and all-candidate tendencies", ""])
    for source in CONTRACT.SOURCE_IDS:
        monotonic = aggregates["latent_progress_diagnostics"][source]["monotonicity"]["all"]
        tendency = aggregates["all_candidate_tendency_diagnostics"][source]["all"]
        lines.append(
            f"- `{source}`: endpoint current→H3 nonincrease={_fmt(monotonic['current_to_h3_nonincrease_fraction'])}; "
            f"all-adjacent-step monotonic={_fmt(monotonic['overall_monotonic_trajectory_fraction'])}; "
            f"contact down-ranking={_fmt(tendency['immediate_contact_h1']['downranking_accuracy'])}; "
            f"successor-nonviable={_fmt(tendency['successor_nonviable']['downranking_accuracy'])}; "
            f"stuck={_fmt(tendency['stuck']['downranking_accuracy'])}; "
            f"no-progress={_fmt(tendency['no_progress']['downranking_accuracy'])}."
        )
    lines.extend(
        [
            "",
            "## Gates and classifications",
            "",
            f"True-future gate `{_fmt(result['gates']['true_future']['pass'])}`; complete two-step gate `{_fmt(result['two_step_gate_passed'])}`.",
            f"Primary `{result['primary_classification']}`; secondary `{json.dumps(result['secondary_classifications'])}`; diagnostics `{json.dumps(result['diagnostic_flags'])}`.",
            f"Next experiment: `{result['next_experiment']}`.",
            "",
            "## Requirements boundary and next decision",
            "",
            CONTRACT.REQUIREMENTS_STATEMENT,
            "",
            "## Runtime, storage and prohibitions",
            "",
            f"Runtime `{json.dumps(result['runtime_s'], sort_keys=True)}`; storage `{json.dumps(result['storage'], sort_keys=True)}`.",
            "",
            "The frozen PPO/controller executed 1,920 deterministic prefix-reconstruction blocks and 4,320 fixed open-loop candidate/fanout physics blocks. No experimental candidate-selecting JEPA, MPC, or navigation planner was executed or evaluated. No training, optimizer, fresh panel, G2, memory, novelty, routing, or beacon capture was executed. Future targets and oracle viability are evaluation-only and unavailable before action execution.",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate(source_freeze_commit: str, *, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Persist scalar rows, aggregates, result, report, and noncircular receipt."""

    _validate_cpu_interpreter_entrypoint()
    if output_root.resolve() == OUTPUT_ROOT.resolve():
        raise QualificationError("direct canonical evaluation is forbidden")
    validate_source_freeze_commit(source_freeze_commit, require_live_head=True)
    validate_frozen_receipts()
    validate_input_hashes()
    validate_cpu_runtime_input_inventory(
        source_freeze_commit,
        output_root=output_root,
        persist_if_absent=False,
    )
    pre = load_json(output_root / PREEXEC_REL)
    validate_phase_binding(pre, source_freeze_commit)
    started = time.time()
    (
        candidate_rows,
        selection_rows,
        paired_rows,
        aggregates,
        classification,
    ) = _write_evidence_and_metrics(source_freeze_commit, output_root)
    _validate_evidence_rows(candidate_rows, selection_rows, paired_rows)
    atomic_bytes(
        output_root / CANDIDATE_LEDGER_REL,
        deterministic_gzip_jsonl_bytes(candidate_rows),
    )
    atomic_bytes(
        output_root / SELECTION_LEDGER_REL,
        deterministic_gzip_jsonl_bytes(selection_rows),
    )
    atomic_bytes(
        output_root / PAIRED_LEDGER_REL,
        deterministic_gzip_jsonl_bytes(paired_rows),
    )
    atomic_json(output_root / AGGREGATE_REL, aggregates)
    _validate_schema_value("aggregate_metrics", aggregates)
    fanout = load_json(output_root / FANOUT_INDEX_REL)
    latent = load_json(output_root / LATENT_INDEX_REL)
    goal = load_json(output_root / GOAL_INDEX_REL)
    gpu = load_json(output_root / GPU_INFERENCE_REL)
    dense = load_json(output_root / DENSE_REPLAY_INPUT_INDEX_REL)
    cpu_runtime_inventory = load_json(output_root / CPU_RUNTIME_INPUT_INVENTORY_REL)
    for file_id, value in (
        ("cpu_runtime_input_inventory", cpu_runtime_inventory),
        ("oracle_admissibility_fanout_index", fanout),
        ("latent_tensor_index", latent),
        ("goal_view_index", goal),
        ("gpu_inference_receipt", gpu),
        ("dense_route_replay_input_index", dense),
    ):
        _validate_schema_value(file_id, value)
    execution_watchdog_status = _successful_execution_watchdog_status(fanout, gpu)
    try:
        (output_root / RUNNING_REL).unlink()
    except FileNotFoundError:
        pass
    active = _active_experiment_processes()
    if active:
        raise QualificationError(f"experiment subprocesses remain active: {active}")
    row_reproduction = {
        "tensor_to_cost_rows_without_inference": True,
        "cost_rows_to_aggregates_without_inference": True,
        "candidate_rows": 1728,
        "selection_rows": 720,
        "paired_effect_rows": 32,
    }
    phase_runtime = {
        "preflight": float(pre.get("runtime_s", 0.0)),
        "cpu_materialization": float(fanout["cpu_materialization_runtime_s"]),
        "gpu_materialization": float(latent["runtime_s"]),
        "evaluation_reduction": time.time() - started,
    }
    phase_runtime["total"] = sum(phase_runtime.values())
    phase_runtime["terminal_validation_excluded_from_embedded_total"] = True
    result_core: dict[str, Any] = {
        "schema": "jepa_local_waypoint_planning_cost_result_v1",
        "experiment_id": CONTRACT.EXPERIMENT_ID,
        "head": source_freeze_commit,
        "source_freeze_commit": source_freeze_commit,
        "contract_sha256": CONTRACT.CONTRACT_SHA256,
        "output_schema_sha256": CONTRACT.OUTPUT_SCHEMA_SHA256,
        "fixture_sha256": sha256_file(ROOT / CONTRACT.TRACKED_FIXTURE_PATH),
        "source_closure_sha256": sha256_file(ROOT / CONTRACT.TRACKED_SOURCE_CLOSURE_PATH),
        "seed": CONTRACT.SEED,
        "materialisation_counts": {
            "states": 48,
            "candidates": 576,
            "current_blocks": 432,
            "successor_blocks": 3888,
            "total_oracle_blocks": 4320,
            "physics_frames": 1080000,
            "reconstruction_prefix_blocks": 1920,
            "reconstruction_physics_frames": 480000,
            "oracle_fanout_blocks": 4320,
            "oracle_fanout_physics_frames": 1080000,
            "total_simulator_blocks": 6240,
            "total_simulator_physics_frames": 1560000,
            "snapshot_reproductions": 48,
            "latent_tensors": latent["total_records"],
            "predicted_tensors": 3456,
            "candidate_evidence_rows": 1728,
            "selection_evidence_rows": 720,
            "paired_effect_evidence_rows": 32,
        },
        "goal_view_counts": {"states": 48, "views": 48, "failed": 0},
        "gpu_environment_receipt_binding": _binding(
            GPU_ENVIRONMENT_REL, output_root
        ),
        "gpu_inference_custody": _binding(GPU_INFERENCE_REL, output_root),
        "cpu_runtime_input_inventory_binding": _binding(
            CPU_RUNTIME_INPUT_INVENTORY_REL, output_root
        ),
        "dense_route_replay_input_index_binding": _binding(
            DENSE_REPLAY_INPUT_INDEX_REL, output_root
        ),
        "metrics": _binding(AGGREGATE_REL, output_root),
        "gates": aggregates["gates"],
        "true_future_gate_classification": classification[
            "true_future_gate_classification"
        ],
        "two_step_gate_passed": classification["two_step_gate_passed"],
        "two_step_gate_signal_or_null": classification[
            "two_step_gate_signal_or_null"
        ],
        "predicted_base_screens": classification["predicted_base_screens"],
        "diagnostic_flags": classification["diagnostic_flags"],
        "paired_materiality": aggregates["paired_comparisons"],
        "primary_classification": classification["primary_classification"],
        "secondary_classifications": classification["secondary_classifications"],
        "next_experiment": classification["next_experiment"],
        "requirements_custody": CONTRACT.build_contract()["requirements_custody"],
        "historical_renderer_limitations": copy.deepcopy(
            CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
        ),
        "reconstruction_prefix_custody": copy.deepcopy(
            CONTRACT.RECONSTRUCTION_PREFIX_CUSTODY
        ),
        "controller_execution_custody": copy.deepcopy(
            CONTRACT.CONTROLLER_EXECUTION_CUSTODY
        ),
        "execution_watchdog_status": execution_watchdog_status,
        "runtime_s": phase_runtime,
        "storage": {},
        "prohibition_counters": prohibition_counters(),
        "row_reproduction": row_reproduction,
        "nothing_running": True,
    }
    base_storage = _actual_storage(output_root)
    storage: dict[str, Any] = {}
    for _ in range(12):
        result_core["storage"] = storage
        candidate_result = attach_digest(result_core, "result_content_sha256")
        report_bytes = _markdown_report(candidate_result, aggregates).encode("utf-8")
        candidate_persistence = _build_persistence_receipt(
            source_freeze_commit,
            output_root,
            row_reproduction,
            report_payload=report_bytes,
            execution_watchdog_status=execution_watchdog_status,
        )
        result_bytes = canonical_json_bytes(candidate_result)
        persistence_bytes = canonical_json_bytes(candidate_persistence)
        final_bytes = (
            int(base_storage["bytes"])
            + len(result_bytes)
            + len(report_bytes)
            + len(persistence_bytes)
        )
        next_storage = {
            **base_storage,
            "files": int(base_storage["files"]) + 3,
            "bytes": final_bytes,
            "gb_decimal": final_bytes / 1e9,
            "peak_vram_bytes": int(latent["peak_vram_bytes"]),
            "within_final_ceiling": final_bytes
            <= int(base_storage["final_ceiling_bytes"]),
        }
        if next_storage == storage:
            result = candidate_result
            report = report_bytes
            persistence = candidate_persistence
            break
        storage = next_storage
    else:
        raise QualificationError("terminal result/report storage fixed point did not converge")
    if not result["storage"]["within_final_ceiling"]:
        raise QualificationError("final output exceeds 12 GB ceiling")
    _validate_schema_value("result", result)
    _validate_schema_value("persistence_receipt", persistence)
    atomic_bytes(output_root / REPORT_REL, report)
    atomic_json(output_root / PERSISTENCE_REL, persistence)
    atomic_json(output_root / RESULT_REL, result)
    if _actual_storage(output_root)["bytes"] != result["storage"]["bytes"]:
        raise QualificationError("final actual storage differs from result receipt")
    return result


def _build_persistence_receipt(
    source_freeze_commit: str,
    output_root: Path,
    row_reproduction: Mapping[str, Any],
    *,
    report_payload: bytes,
    execution_watchdog_status: Mapping[str, Any],
) -> dict[str, Any]:
    # Result/self/RUNNING are outside the noncircular manifest.  The report is
    # supplied as bytes and receives an exact virtual manifest row before the
    # three terminal files are each written once.
    excluded = {PERSISTENCE_REL, RUNNING_REL, RESULT_REL, REPORT_REL}
    files = [
        path
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.relative_to(output_root) not in excluded
    ]
    manifest = [
        {
            "path": str(path.relative_to(output_root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    manifest.append(
        {
            "path": str(REPORT_REL),
            "sha256": hashlib.sha256(report_payload).hexdigest(),
            "bytes": len(report_payload),
        }
    )
    manifest.sort(key=lambda row: str(row["path"]))
    return attach_digest(
        {
            **_phase_core(
                "jepa_local_waypoint_planning_cost_persistence_v1",
                source_freeze_commit,
            ),
            "artifact_manifest": manifest,
            "artifact_manifest_exclusions": [
                str(PERSISTENCE_REL),
                str(RESULT_REL),
                str(RUNNING_REL),
            ],
            "artifact_count": len(manifest),
            "total_bytes": sum(int(row["bytes"]) for row in manifest),
            "row_counts": {"candidate": 1728, "selection": 720, "paired": 32},
            "row_reproduction": dict(row_reproduction),
            "context_reconstruction_index_binding": _binding(
                CONTEXT_INDEX_REL, output_root
            ),
            "dense_route_replay_input_index_binding": _binding(
                DENSE_REPLAY_INPUT_INDEX_REL, output_root
            ),
            "cpu_runtime_input_inventory_binding": _binding(
                CPU_RUNTIME_INPUT_INVENTORY_REL, output_root
            ),
            "oracle_admissibility_fanout_index_binding": _binding(
                FANOUT_INDEX_REL, output_root
            ),
            "latent_tensor_index_binding": _binding(LATENT_INDEX_REL, output_root),
            "gpu_inference_receipt_binding": _binding(
                GPU_INFERENCE_REL, output_root
            ),
            "gpu_environment_receipt_binding": _binding(
                GPU_ENVIRONMENT_REL, output_root
            ),
            "external_input_index_bindings": validate_input_hashes(),
            "all_shard_hash_shape_dtype_byte_validation": True,
            "tensor_to_cost_row_reproduction": True,
            "cost_row_to_aggregate_reproduction": True,
            "contract_schema_validation": True,
            "execution_watchdog_status": copy.deepcopy(
                dict(execution_watchdog_status)
            ),
            "prohibition_counters": prohibition_counters(),
            "nothing_running": True,
        }
    )


def _validate_result_cross_bindings(
    result: Mapping[str, Any],
    loaded: Mapping[str, Mapping[str, Any]],
    *,
    output_root: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Bind every reportable result field to its canonical persisted authority."""

    aggregate = loaded["aggregate_metrics"]
    classification = aggregate["classification"]
    fanout = loaded["oracle_admissibility_fanout_index"]
    latent = loaded["latent_tensor_index"]
    goal = loaded["goal_view_index"]
    persistence = loaded["persistence_receipt"]
    execution_watchdog_status = _successful_execution_watchdog_status(
        fanout, loaded["gpu_inference_receipt"]
    )
    exact_bindings = {
        "metrics": _binding(AGGREGATE_REL, output_root),
        "gpu_environment_receipt_binding": _binding(
            GPU_ENVIRONMENT_REL, output_root
        ),
        "gpu_inference_custody": _binding(GPU_INFERENCE_REL, output_root),
        "dense_route_replay_input_index_binding": _binding(
            DENSE_REPLAY_INPUT_INDEX_REL, output_root
        ),
        "cpu_runtime_input_inventory_binding": _binding(
            CPU_RUNTIME_INPUT_INVENTORY_REL, output_root
        ),
        "gates": aggregate["gates"],
        "paired_materiality": aggregate["paired_comparisons"],
        "true_future_gate_classification": classification[
            "true_future_gate_classification"
        ],
        "two_step_gate_passed": classification["two_step_gate_passed"],
        "two_step_gate_signal_or_null": classification[
            "two_step_gate_signal_or_null"
        ],
        "predicted_base_screens": classification["predicted_base_screens"],
        "diagnostic_flags": classification["diagnostic_flags"],
        "primary_classification": classification["primary_classification"],
        "secondary_classifications": classification[
            "secondary_classifications"
        ],
        "next_experiment": classification["next_experiment"],
        "requirements_custody": CONTRACT.build_contract()["requirements_custody"],
        "historical_renderer_limitations": copy.deepcopy(
            CONTRACT.HISTORICAL_RENDERER_LIMITATIONS
        ),
        "reconstruction_prefix_custody": copy.deepcopy(
            CONTRACT.RECONSTRUCTION_PREFIX_CUSTODY
        ),
        "controller_execution_custody": copy.deepcopy(
            CONTRACT.CONTROLLER_EXECUTION_CUSTODY
        ),
        "execution_watchdog_status": execution_watchdog_status,
    }
    for key, expected_value in exact_bindings.items():
        if result.get(key) != expected_value:
            raise QualificationError(f"result authority cross-binding drift: {key}")

    expected_materialisation = {
        "states": int(fanout["states"]),
        "candidates": int(fanout["states"]) * CANDIDATES_PER_STATE,
        "current_blocks": int(fanout["current_blocks"]),
        "successor_blocks": int(fanout["successor_blocks"]),
        "total_oracle_blocks": int(fanout["current_blocks"])
        + int(fanout["successor_blocks"]),
        "physics_frames": int(fanout["physics_frames"]),
        "reconstruction_prefix_blocks": 48 * 40,
        "reconstruction_physics_frames": 48 * 40 * PHYSICS_STEPS_PER_BLOCK,
        "oracle_fanout_blocks": int(fanout["current_blocks"])
        + int(fanout["successor_blocks"]),
        "oracle_fanout_physics_frames": int(fanout["physics_frames"]),
        "total_simulator_blocks": 48 * 40
        + int(fanout["current_blocks"])
        + int(fanout["successor_blocks"]),
        "total_simulator_physics_frames": 48 * 40 * PHYSICS_STEPS_PER_BLOCK
        + int(fanout["physics_frames"]),
        "snapshot_reproductions": int(
            loaded["context_reconstruction_index"]
            ["branch_snapshot_authority_validation"]["passed"]
        ),
        "latent_tensors": int(latent["total_records"]),
        "predicted_tensors": int(latent["counts_by_kind"]["ONE_STEP_PREDICTED"])
        + int(latent["counts_by_kind"]["TWO_STEP_PREDICTED"]),
        "candidate_evidence_rows": len(candidate_rows),
        "selection_evidence_rows": len(selection_rows),
        "paired_effect_evidence_rows": len(paired_rows),
    }
    if result.get("materialisation_counts") != expected_materialisation:
        raise QualificationError("result materialisation-count authority drift")
    expected_goal_counts = {
        "states": int(goal["states"]),
        "views": len(goal["records"]),
        "failed": len(goal["failed_state_ids"]),
    }
    if result.get("goal_view_counts") != expected_goal_counts:
        raise QualificationError("result goal-view count authority drift")
    expected_reproduction = {
        "tensor_to_cost_rows_without_inference": True,
        "cost_rows_to_aggregates_without_inference": True,
        "candidate_rows": len(candidate_rows),
        "selection_rows": len(selection_rows),
        "paired_effect_rows": len(paired_rows),
    }
    if result.get("row_reproduction") != expected_reproduction or persistence.get(
        "row_reproduction"
    ) != expected_reproduction:
        raise QualificationError("result/persistence row-reproduction drift")
    expected_prohibitions = prohibition_counters()
    if result.get("prohibition_counters") != expected_prohibitions or persistence.get(
        "prohibition_counters"
    ) != expected_prohibitions:
        raise QualificationError("result/persistence prohibition custody drift")
    if persistence.get("execution_watchdog_status") != execution_watchdog_status:
        raise QualificationError("persistence execution-watchdog custody drift")
    if result.get("head") != result.get("source_freeze_commit"):
        raise QualificationError("result head/source-freeze binding drift")
    if result.get("experiment_id") != CONTRACT.EXPERIMENT_ID:
        raise QualificationError("result experiment identity drift")
    if result.get("contract_sha256") != CONTRACT.CONTRACT_SHA256:
        raise QualificationError("result contract digest drift")
    if result.get("output_schema_sha256") != CONTRACT.OUTPUT_SCHEMA_SHA256:
        raise QualificationError("result output-schema digest drift")
    if result.get("seed") != CONTRACT.SEED:
        raise QualificationError("result deterministic seed drift")
    if result.get("fixture_sha256") != sha256_file(
        ROOT / CONTRACT.TRACKED_FIXTURE_PATH
    ) or result.get("source_closure_sha256") != sha256_file(
        ROOT / CONTRACT.TRACKED_SOURCE_CLOSURE_PATH
    ):
        raise QualificationError("result fixture/source-closure binding drift")


def _validate_encoder_source_repository_binding(
    observed: Mapping[str, Any],
) -> None:
    encoder = CONTRACT.build_contract()["latent_bindings"]["encoder"]
    repository = encoder["source_repository"]
    root = Path(repository["path"]).resolve()
    backbones = root / repository["backbones_path"]
    if not root.is_dir() or not backbones.is_file():
        raise QualificationError("frozen encoder source repository is missing")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    expected = {
        "path": str(root),
        "git_commit": repository["git_commit"],
        "worktree_clean": bool(repository["worktree_clean_required"]),
        "constructor": encoder["constructor"],
        "backbones_path": repository["backbones_path"],
        "backbones_sha256": repository["backbones_sha256"],
        "backbones_bytes": repository["backbones_bytes"],
    }
    actual = {
        "path": str(root),
        "git_commit": commit,
        "worktree_clean": status == "",
        "constructor": encoder["constructor"],
        "backbones_path": repository["backbones_path"],
        "backbones_sha256": sha256_file(backbones),
        "backbones_bytes": backbones.stat().st_size,
    }
    if actual != expected or dict(observed) != expected:
        raise QualificationError("encoder source repository custody drift")


def check(
    *,
    source_freeze_commit: str | None = None,
    output_root: Path = OUTPUT_ROOT,
    deep: bool = True,
    require_tracked: bool = True,
) -> dict[str, Any]:
    """Deep terminal reproduction; immutable phases never bind to live HEAD."""

    _validate_cpu_interpreter_entrypoint()
    validate_frozen_receipts()
    validate_input_hashes()
    result = load_json(output_root / RESULT_REL)
    validate_digest(result, "result_content_sha256")
    _validate_schema_value("result", result)
    frozen_commit = str(result.get("source_freeze_commit"))
    if source_freeze_commit is not None and source_freeze_commit != frozen_commit:
        raise QualificationError("requested source-freeze commit differs from result")
    validate_source_freeze_commit(frozen_commit, require_live_head=False)
    current_input_bindings = validate_input_hashes()
    validate_cpu_runtime_input_inventory(
        frozen_commit,
        output_root=output_root,
        persist_if_absent=False,
    )
    phase_files = (
        ("preexecution_receipt", PREEXEC_REL),
        ("environment_receipt", ENVIRONMENT_REL),
        ("gpu_environment_receipt", GPU_ENVIRONMENT_REL),
        ("cpu_runtime_input_inventory", CPU_RUNTIME_INPUT_INVENTORY_REL),
        ("context_reconstruction_index", CONTEXT_INDEX_REL),
        ("dense_route_replay_input_index", DENSE_REPLAY_INPUT_INDEX_REL),
        ("oracle_admissibility_fanout_index", FANOUT_INDEX_REL),
        ("goal_view_index", GOAL_INDEX_REL),
        ("latent_tensor_index", LATENT_INDEX_REL),
        ("gpu_inference_receipt", GPU_INFERENCE_REL),
        ("aggregate_metrics", AGGREGATE_REL),
        ("persistence_receipt", PERSISTENCE_REL),
    )
    loaded: dict[str, dict[str, Any]] = {}
    for file_id, relative in phase_files:
        value = load_json(output_root / relative)
        validate_phase_binding(value, frozen_commit)
        _validate_schema_value(file_id, value)
        loaded[file_id] = value
    cpu_runtime_binding = _binding(CPU_RUNTIME_INPUT_INVENTORY_REL, output_root)
    expected_panel_bindings = {
        **copy.deepcopy(current_input_bindings),
        "cpu_runtime_input_inventory": cpu_runtime_binding,
    }
    if loaded["preexecution_receipt"].get("panel_bindings") != (
        expected_panel_bindings
    ) or loaded["persistence_receipt"].get(
        "external_input_index_bindings"
    ) != current_input_bindings:
        raise QualificationError("frozen external-input binding custody drift")
    for file_id in (
        "context_reconstruction_index",
        "oracle_admissibility_fanout_index",
    ):
        if loaded[file_id].get("cpu_runtime_input_inventory_binding") != (
            cpu_runtime_binding
        ):
            raise QualificationError(
                f"{file_id} CPU runtime-input binding drift"
            )
    if loaded["preexecution_receipt"].get(
        "cpu_runtime_input_inventory_binding"
    ) != cpu_runtime_binding or loaded["environment_receipt"].get(
        "cpu_runtime_input_inventory_binding"
    ) != cpu_runtime_binding:
        raise QualificationError("preexecution/environment CPU runtime custody drift")
    if loaded["preexecution_receipt"].get("execution_watchdog_config") != (
        CONTRACT.EXECUTION_WATCHDOGS
    ):
        raise QualificationError("preexecution watchdog configuration drift")
    if Path(
        str(loaded["preexecution_receipt"].get("canonical_output_root"))
    ).resolve() != Path(OUTPUT_ROOT).resolve() or loaded[
        "preexecution_receipt"
    ].get(
        "canonical_output_fresh"
    ) is not True:
        raise QualificationError("preexecution canonical-output custody drift")
    cpu_package_inventory = loaded["environment_receipt"].get(
        "package_inventory", {}
    ).get("cpu", {})
    if (
        cpu_package_inventory.get("foundational_package_roots")
        != loaded["cpu_runtime_input_inventory"]["foundational_package_roots"]
        or cpu_package_inventory.get("foundational_package_closure_policy")
        != CONTRACT.FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
    ):
        raise QualificationError("combined environment CPU foundational custody drift")
    latent = loaded["latent_tensor_index"]
    validate_latent_tensor_index(latent, output_root=output_root, deep=deep)
    fanout = loaded["oracle_admissibility_fanout_index"]
    if len(fanout["records"]) != 48:
        raise QualificationError("fanout index does not contain 48 states")
    raw_policy = fanout.get("raw_continuation_policy_validation")
    if not isinstance(raw_policy, Mapping) or {
        "captures": raw_policy.get("captures"),
        "expected_captures": raw_policy.get("expected_captures"),
        "successor_restores": raw_policy.get("successor_restores"),
        "expected_successor_restores": raw_policy.get(
            "expected_successor_restores"
        ),
        "capture_mode": raw_policy.get("capture_mode"),
        "fall_tip_out_of_bounds_permitted_and_persisted": raw_policy.get(
            "fall_tip_out_of_bounds_permitted_and_persisted"
        ),
        "nan_rejected": raw_policy.get("nan_rejected"),
        "production_reset_checks_suppressed": raw_policy.get(
            "production_reset_checks_suppressed"
        ),
        "evaluation_only": raw_policy.get("evaluation_only"),
        "pass": raw_policy.get("pass"),
    } != {
        "captures": 432,
        "expected_captures": 432,
        "successor_restores": 3888,
        "expected_successor_restores": 3888,
        "capture_mode": "RAW_CONTINUATION_AFTER_CURRENT_H1",
        "fall_tip_out_of_bounds_permitted_and_persisted": True,
        "nan_rejected": True,
        "production_reset_checks_suppressed": True,
        "evaluation_only": True,
        "pass": True,
    }:
        raise QualificationError("fanout raw-continuation policy custody drift")
    for record in fanout["records"]:
        raw_rows = record.get("raw_continuation_snapshots")
        if not isinstance(raw_rows, list) or len(raw_rows) != 9:
            raise QualificationError("fanout raw-continuation record cardinality drift")
        for index, raw in enumerate(raw_rows):
            if (
                raw.get("current_primitive_index") != index
                or raw.get("capture_mode")
                != "RAW_CONTINUATION_AFTER_CURRENT_H1"
                or raw.get("restored_for_successors") != 9
                or raw.get("production_reset_checks_suppressed") is not True
                or raw.get("evaluation_only") is not True
                or raw.get("terminal_flags", {}).get("nan") is not False
            ):
                raise QualificationError("fanout raw-continuation record drift")
        current, successor = _load_fanout_arrays(record, output_root=output_root)
        mapping = record["current_action_mapping"][
            "macro_candidate_to_primitive_index"
        ]
        safe = np.sum(~np.any(successor, axis=2), axis=1, dtype=np.int64)
        derived = [int(safe[int(mapping[str(index)])]) for index in range(12)]
        if derived != record["successor_safe_action_count"]:
            raise QualificationError("fanout safe-count derivation drift")
        viable = [value > 0 for value in derived]
        if viable != record["successor_viable"]:
            raise QualificationError("fanout successor-viability derivation drift")
        immediate = [
            bool(np.any(current[int(mapping[str(index)])])) for index in range(12)
        ]
        admissible = [
            (not contact) and successor_viable
            for contact, successor_viable in zip(immediate, viable, strict=True)
        ]
        if admissible != record["oracle_viability_admissible"]:
            raise QualificationError("fanout oracle-admissibility derivation drift")
    gpu = loaded["gpu_inference_receipt"]
    _validate_encoder_source_repository_binding(
        gpu["encoder_source_repository_binding"]
    )
    gpu_environment = load_json(output_root / GPU_ENVIRONMENT_REL)
    validate_phase_binding(gpu_environment, frozen_commit)
    if gpu.get("gpu_environment_receipt_binding") != _binding(
        GPU_ENVIRONMENT_REL, output_root
    ):
        raise QualificationError("GPU inference environment-receipt binding drift")
    if gpu.get("environment") != {
        "python": gpu_environment["python"],
        "torch": gpu_environment["torch"],
        **gpu_environment["packages"],
    } or gpu.get("device") != {
        "requested": "cuda:0",
        "resolved": "cuda:0",
        **gpu_environment["device"],
    }:
        raise QualificationError("GPU inference live environment/device custody drift")
    if gpu.get("gpu_environment_revalidation") != {
        "interpreter_exact": True,
        "versions_exact": True,
        "device_exact": True,
        "foundational_roots_exact": True,
        "precheckpoint": True,
        "pass": True,
    }:
        raise QualificationError("GPU inference environment-revalidation drift")
    expected_gpu_watchdog = CONTRACT.GPU_WATCHDOG_STATUS_SUCCESS
    if gpu.get("gpu_watchdog_status") != expected_gpu_watchdog:
        raise QualificationError("GPU inference watchdog custody drift")
    _validate_encoder_source_repository_binding(
        gpu_environment["encoder_source_repository_binding"]
    )
    expected_gpu_imports = [
        "torch",
        "numpy",
        "scipy",
        "PIL",
        "yaml",
        "scripts.dev_frozen_dense_representation_encoders_v1",
        "scripts.dev_proprio_predictor_v1",
        "scripts.run_dev_v03_temporal_action_jepa_v1",
        "scripts.build_dev_v03_proprio_action_manifest_v1",
        "scripts.dev_action_slew_reconstruction_v1",
        "scripts.dev_checkpoint_v1",
        "lewm.safety.jepa_local_waypoint_planning_cost_qualification_v1_contract",
    ]
    if gpu_environment.get("import_closure") != expected_gpu_imports:
        raise QualificationError("GPU import-closure module set/order drift")
    required_local_modules = set(expected_gpu_imports[5:])
    source_bindings = gpu_environment.get("import_source_bindings")
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != required_local_modules:
        raise QualificationError("GPU import-source binding domain drift")
    for module_id, binding in source_bindings.items():
        path = Path(str(binding["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(binding["bytes"])
            or sha256_file(path) != binding["sha256"]
        ):
            raise QualificationError(f"GPU import-source binding drift: {module_id}")
    if loaded["environment_receipt"]["import_closure"]["gpu"] != expected_gpu_imports:
        raise QualificationError("combined environment GPU import closure drift")
    three = {"encoder": True, "one_step": True, "two_step": True}
    for key in (
        "strict_state_dict_load",
        "eval_mode",
        "requires_grad_all_false",
        "parameter_state_unchanged",
    ):
        if gpu[key] != three:
            raise QualificationError(f"GPU inference custody drift: {key}")
    if (
        gpu["training_steps"] != 0
        or gpu["optimizer_absent"] is not True
        or gpu["inference_mode"] is not True
        or gpu["dynamic_oom_batch_fallback_used"] is not False
        or gpu["future_input_fields"] != []
    ):
        raise QualificationError("GPU inference prohibition custody drift")
    expected_calls = {
        "ONE_STEP_PREDICTED": {
            "unroll_calls": 48,
            "model_forward_calls": 144,
            "candidate_horizon_outputs": 1728,
        },
        "TWO_STEP_PREDICTED": {
            "unroll_calls": 48,
            "model_forward_calls": 144,
            "candidate_horizon_outputs": 1728,
        },
    }
    if gpu["predictor_call_counts_by_source"] != expected_calls:
        raise QualificationError("GPU predictor call-count custody drift")
    preprocessing = gpu["preprocessing_digest"]
    expected_preprocessing = CONTRACT.build_contract()["latent_bindings"]["encoder"][
        "preprocessing_digest"
    ]
    if preprocessing != {
        "expected": expected_preprocessing,
        "observed": expected_preprocessing,
        "match": True,
    }:
        raise QualificationError("GPU preprocessing custody drift")
    if gpu["encoder_call_counts"] != {
        "frames": 192,
        "batch_size": 16,
        "batches": 12,
        "preprocessing_digest": expected_preprocessing,
    }:
        raise QualificationError("GPU encoder call-count custody drift")
    closure_path = ROOT / CONTRACT.TRACKED_SOURCE_CLOSURE_PATH
    closure = load_json(closure_path)
    if gpu["source_closure_binding"] != {
        "path": CONTRACT.TRACKED_SOURCE_CLOSURE_PATH,
        "sha256": sha256_file(closure_path),
        "content_digest": closure["content_digest"],
        "complete": True,
    }:
        raise QualificationError("GPU source-closure binding drift")
    if gpu["latent_tensor_index_binding"] != _binding(
        LATENT_INDEX_REL, output_root
    ):
        raise QualificationError("GPU latent-index binding drift")
    gpu_terminal_process = _run_gpu_child(
        [
            "check",
            "--source-freeze-commit",
            frozen_commit,
            "--output-root",
            str(output_root),
            "--shallow",
        ]
    )
    try:
        gpu_terminal = json.loads(gpu_terminal_process.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationError("GPU terminal checker returned invalid JSON") from exc
    if gpu_terminal != {
        "pass": True,
        "source_freeze_commit": frozen_commit,
        "tensor_records": 5424,
        "checkpoint_opened_during_check": 0,
        "predictor_inference_calls_during_check": 0,
        "genesis_imported": False,
    }:
        raise QualificationError("GPU terminal checker custody drift")
    persistence = loaded["persistence_receipt"]
    manifest_paths: set[str] = set()
    for row in persistence["artifact_manifest"]:
        relative = str(row["path"])
        if relative in manifest_paths:
            raise QualificationError(f"duplicate persistence artifact {relative}")
        manifest_paths.add(relative)
        path = output_root / relative
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise QualificationError(f"persistence artifact byte drift: {relative}")
        if deep and sha256_file(path) != row["sha256"]:
            raise QualificationError(f"persistence artifact hash drift: {relative}")
    for relative in (PERSISTENCE_REL, RESULT_REL, RUNNING_REL):
        if str(relative) in manifest_paths:
            raise QualificationError("noncircular persistence exclusion was violated")
    expected_manifest_paths = {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file()
        and path.relative_to(output_root)
        not in {PERSISTENCE_REL, RESULT_REL, RUNNING_REL}
    }
    if manifest_paths != expected_manifest_paths:
        raise QualificationError("persistence artifact-manifest set is incomplete")
    if persistence["artifact_manifest_exclusions"] != [
        str(PERSISTENCE_REL),
        str(RESULT_REL),
        str(RUNNING_REL),
    ]:
        raise QualificationError("persistence exclusion contract drift")
    if int(persistence["artifact_count"]) != len(persistence["artifact_manifest"]):
        raise QualificationError("persistence artifact-count drift")
    if int(persistence["total_bytes"]) != sum(
        int(row["bytes"]) for row in persistence["artifact_manifest"]
    ):
        raise QualificationError("persistence total-byte drift")
    if persistence["row_counts"] != {
        "candidate": 1728,
        "selection": 720,
        "paired": 32,
    }:
        raise QualificationError("persistence row-count drift")
    report_text = (output_root / REPORT_REL).read_text(encoding="utf-8")
    for section in CONTRACT.build_output_schema()["files"]["report"][
        "required_sections"
    ]:
        if f"## {section}\n" not in report_text:
            raise QualificationError(f"terminal report lacks section {section!r}")
    if (output_root / REPORT_REL).read_bytes() != _markdown_report(
        result, loaded["aggregate_metrics"]
    ).encode("utf-8"):
        raise QualificationError("terminal Markdown report regeneration drift")
    for key in (
        "all_shard_hash_shape_dtype_byte_validation",
        "tensor_to_cost_row_reproduction",
        "cost_row_to_aggregate_reproduction",
        "contract_schema_validation",
    ):
        if persistence[key] is not True:
            raise QualificationError(f"persistence validation is false: {key}")
    for key, relative in (
        ("context_reconstruction_index_binding", CONTEXT_INDEX_REL),
        ("dense_route_replay_input_index_binding", DENSE_REPLAY_INPUT_INDEX_REL),
        ("cpu_runtime_input_inventory_binding", CPU_RUNTIME_INPUT_INVENTORY_REL),
        ("oracle_admissibility_fanout_index_binding", FANOUT_INDEX_REL),
        ("latent_tensor_index_binding", LATENT_INDEX_REL),
        ("gpu_environment_receipt_binding", GPU_ENVIRONMENT_REL),
        ("gpu_inference_receipt_binding", GPU_INFERENCE_REL),
    ):
        if persistence[key] != _binding(relative, output_root):
            raise QualificationError(f"persistence binding drift: {key}")
    if (output_root / RUNNING_REL).exists():
        raise QualificationError("running marker remains after terminal result")
    if any(result["prohibition_counters"].values()):
        raise QualificationError("terminal result reports a prohibited action")
    if result.get("nothing_running") is not True or persistence.get("nothing_running") is not True:
        raise QualificationError("terminal nothing-running assertion is false")
    candidate_rows = load_gzip_jsonl(output_root / CANDIDATE_LEDGER_REL)
    selection_rows = load_gzip_jsonl(output_root / SELECTION_LEDGER_REL)
    paired_rows = load_gzip_jsonl(output_root / PAIRED_LEDGER_REL)
    _validate_evidence_rows(candidate_rows, selection_rows, paired_rows)
    _validate_result_cross_bindings(
        result,
        loaded,
        output_root=output_root,
        candidate_rows=candidate_rows,
        selection_rows=selection_rows,
        paired_rows=paired_rows,
    )
    if deep:
        (
            row_only_candidate,
            row_only_selection,
            row_only_paired,
            row_only_aggregate,
            row_only_classification,
        ) = _reduce_candidate_rows_without_tensors(
            frozen_commit,
            candidate_rows,
            loaded["goal_view_index"],
        )
        if row_only_candidate != candidate_rows:
            raise QualificationError("row-only candidate custody drift")
        if deterministic_gzip_jsonl_bytes(row_only_selection) != (
            output_root / SELECTION_LEDGER_REL
        ).read_bytes():
            raise QualificationError("row-only selection reproduction drift")
        if deterministic_gzip_jsonl_bytes(row_only_paired) != (
            output_root / PAIRED_LEDGER_REL
        ).read_bytes():
            raise QualificationError("row-only paired reproduction drift")
        if canonical_json_bytes(row_only_aggregate) != (
            output_root / AGGREGATE_REL
        ).read_bytes():
            raise QualificationError("row-only aggregate reproduction drift")
        if row_only_classification != loaded["aggregate_metrics"]["classification"]:
            raise QualificationError("row-only classification reproduction drift")
        (
            regenerated_candidate,
            regenerated_selection,
            regenerated_paired,
            regenerated_aggregate,
            regenerated_classification,
        ) = _write_evidence_and_metrics(
            frozen_commit,
            output_root,
            persist_auxiliary_indices=False,
        )
        if deterministic_gzip_jsonl_bytes(regenerated_candidate) != (
            output_root / CANDIDATE_LEDGER_REL
        ).read_bytes():
            raise QualificationError("tensor-to-candidate-row reproduction drift")
        if deterministic_gzip_jsonl_bytes(regenerated_selection) != (
            output_root / SELECTION_LEDGER_REL
        ).read_bytes():
            raise QualificationError("selection-row reproduction drift")
        if deterministic_gzip_jsonl_bytes(regenerated_paired) != (
            output_root / PAIRED_LEDGER_REL
        ).read_bytes():
            raise QualificationError("paired-effect-row reproduction drift")
        if canonical_json_bytes(regenerated_aggregate) != (
            output_root / AGGREGATE_REL
        ).read_bytes():
            raise QualificationError("cost-row-to-aggregate reproduction drift")
        if regenerated_classification != loaded["aggregate_metrics"]["classification"]:
            raise QualificationError("terminal classification reproduction drift")
    active = _active_experiment_processes()
    if active:
        raise QualificationError(f"terminal experiment processes remain: {active}")
    if _actual_storage(output_root)["bytes"] != result["storage"]["bytes"]:
        raise QualificationError("terminal actual storage drift")
    if require_tracked:
        tracked_result = ROOT / CONTRACT.TRACKED_RESULT_PATH
        tracked_report = ROOT / CONTRACT.TRACKED_REPORT_PATH
        if (
            not tracked_result.is_file()
            or tracked_result.read_bytes() != (output_root / RESULT_REL).read_bytes()
            or not tracked_report.is_file()
            or tracked_report.read_bytes() != (output_root / REPORT_REL).read_bytes()
        ):
            raise QualificationError("tracked/external result publication drift")
    return {
        "pass": True,
        "source_freeze_commit": frozen_commit,
        "live_head": git_output("rev-parse", "HEAD"),
        "live_head_may_be_result_commit": True,
        "latent_tensors": latent["total_records"],
        "candidate_rows": len(candidate_rows),
        "selection_rows": len(selection_rows),
        "paired_effect_rows": len(paired_rows),
        "primary_classification": result["primary_classification"],
        "tracked_publication_required": require_tracked,
        "nothing_running": True,
    }


def _archive_failed_attempt(
    attempt_root: Path,
    *,
    source_freeze_commit: str,
    phase: str,
    error: BaseException,
) -> Path | None:
    archive = attempt_root.with_name(
        f".{OUTPUT_ROOT.name}.failed-{time.time_ns()}-{os.getpid()}"
    )
    if attempt_root.exists():
        os.replace(attempt_root, archive)
    else:
        archive.mkdir(parents=True, exist_ok=False)
    running = archive / RUNNING_REL
    if running.exists():
        os.replace(running, archive / "receipts/FAILED_RUNNING_MARKER.json")
    active = _active_experiment_processes()
    receipt = attach_digest(
        {
            **_phase_core(
                "jepa_local_waypoint_planning_cost_failed_attempt_v1",
                source_freeze_commit,
            ),
            "phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "archive_path": str(archive),
            "partial_artifacts_reusable": False,
            "active_experiment_processes": active,
            "nothing_running": not active,
            "prohibition_counters": prohibition_counters(),
        }
    )
    atomic_json(archive / "receipts/failure.json", receipt)
    return archive


def execute(
    source_freeze_commit: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run all phases in a hidden sibling and atomically publish once valid."""

    validate_source_freeze_commit(source_freeze_commit, require_live_head=True)
    canonical = Path(output_root)
    if canonical.resolve() != Path(OUTPUT_ROOT).resolve():
        raise QualificationError("scientific execution output root differs from frozen path")
    if canonical.exists():
        raise QualificationError("canonical output root must be completely absent")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    attempt = canonical.with_name(
        f".{canonical.name}.attempt-{source_freeze_commit[:12]}-{time.time_ns()}-{os.getpid()}"
    )
    if attempt.exists():
        raise QualificationError("hidden attempt namespace collision")
    phase = "PREFLIGHT"
    published = False
    try:
        preflight(
            source_freeze_commit,
            output_root=attempt,
            canonical_output_root=canonical,
            archive_on_failure=False,
        )
        phase = "MATERIALIZATION"
        materialize(
            source_freeze_commit,
            output_root=attempt,
            workers=os.cpu_count() if workers is None else workers,
        )
        phase = "EVALUATION"
        result = evaluate(source_freeze_commit, output_root=attempt)
        phase = "DEEP_PREPUBLICATION_CHECK"
        check(
            source_freeze_commit=source_freeze_commit,
            output_root=attempt,
            deep=True,
            require_tracked=False,
        )
        if canonical.exists():
            raise QualificationError("canonical output appeared during hidden execution")
        if os.stat(attempt.parent).st_dev != os.stat(canonical.parent).st_dev:
            raise QualificationError("attempt and canonical roots are not on one filesystem")
        phase = "ATOMIC_EXTERNAL_PUBLICATION"
        os.replace(attempt, canonical)
        published = True
        phase = "TRACKED_RESULT_PUBLICATION"
        tracked_result = ROOT / CONTRACT.TRACKED_RESULT_PATH
        tracked_report = ROOT / CONTRACT.TRACKED_REPORT_PATH
        atomic_bytes(tracked_result, (canonical / RESULT_REL).read_bytes())
        atomic_bytes(tracked_report, (canonical / REPORT_REL).read_bytes())
        phase = "POSTPUBLICATION_CHECK"
        terminal = check(
            source_freeze_commit=source_freeze_commit,
            output_root=canonical,
            deep=True,
            require_tracked=True,
        )
        return {
            "pass": True,
            "source_freeze_commit": source_freeze_commit,
            "canonical_output_root": str(canonical),
            "atomic_publication": True,
            "hidden_attempt_removed": not attempt.exists(),
            "tracked_result_path": str(
                tracked_result.relative_to(ROOT)
                if tracked_result.is_relative_to(ROOT)
                else tracked_result
            ),
            "tracked_report_path": str(
                tracked_report.relative_to(ROOT)
                if tracked_report.is_relative_to(ROOT)
                else tracked_report
            ),
            "result_content_sha256": result["result_content_sha256"],
            "primary_classification": terminal["primary_classification"],
            "nothing_running": terminal["nothing_running"],
        }
    except BaseException as error:
        archive_root = canonical if published and canonical.exists() else attempt
        archived = _archive_failed_attempt(
            archive_root,
            source_freeze_commit=source_freeze_commit,
            phase=phase,
            error=error,
        )
        for tracked, external in (
            (ROOT / CONTRACT.TRACKED_RESULT_PATH, RESULT_REL),
            (ROOT / CONTRACT.TRACKED_REPORT_PATH, REPORT_REL),
        ):
            try:
                if archived is not None and tracked.is_file() and archived.joinpath(external).is_file() and (
                    tracked.read_bytes() == archived.joinpath(external).read_bytes()
                ):
                    tracked.unlink()
            except OSError:
                pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--source-freeze-commit", required=True)
    execute_parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    execute_parser.add_argument("--workers", type=int, default=os.cpu_count())
    for command in ("preflight", "materialize", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--source-freeze-commit", required=True)
        child.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
        if command == "materialize":
            child.add_argument("--workers", type=int, default=os.cpu_count())
    worker = subparsers.add_parser("cpu-state-worker")
    worker.add_argument("--state-index", type=int, required=True)
    worker.add_argument("--source-freeze-commit", required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    terminal = subparsers.add_parser("check")
    terminal.add_argument("--source-freeze-commit")
    terminal.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    terminal.add_argument("--shallow", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        value = freeze()
    elif args.command == "execute":
        value = execute(
            args.source_freeze_commit,
            output_root=args.output_root,
            workers=args.workers,
        )
    elif args.command == "preflight":
        value = preflight(args.source_freeze_commit, output_root=args.output_root)
    elif args.command == "materialize":
        value = materialize(
            args.source_freeze_commit,
            output_root=args.output_root,
            workers=args.workers,
        )
    elif args.command == "evaluate":
        value = evaluate(args.source_freeze_commit, output_root=args.output_root)
    elif args.command == "cpu-state-worker":
        value = _materialize_state(
            args.state_index,
            output_root=args.output_root,
            source_freeze_commit=args.source_freeze_commit,
        )
    elif args.command == "check":
        value = check(
            source_freeze_commit=args.source_freeze_commit,
            output_root=args.output_root,
            deep=not args.shallow,
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise QualificationError(f"unknown command {args.command}")
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
