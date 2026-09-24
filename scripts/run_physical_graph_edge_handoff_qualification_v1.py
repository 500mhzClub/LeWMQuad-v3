#!/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
"""Direct runner for ``PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V1``.

The scientific path is intentionally conventional: one foreground process
collects one physical state, a later foreground process assembles the frozen
shards, and subsequent foreground stages encode, select the development
target contract, evaluate held-out rows, and report.  Importing this module
does not initialize Genesis, load a checkpoint, open the output root, or
inspect an outcome.

The physical collector uses the reviewed full Genesis snapshot overlay in
``run_go2_oracle_branch_pilot_v1``.  It never substitutes cloned lanes or an
analytic trajectory for the required restore authority.  Tests may inject a
small fake backend, encoder, and ranker through the public stage functions.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from lewm.safety import physical_graph_edge_handoff_qualification_v1_contract as CONTRACT
from lewm.safety import physical_graph_edge_handoff_qualification_v1_metrics as METRICS


EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V1"
PARENT_COMMIT = "3a784118b461d693d5dbf07035b3f85ee1553598"
SOURCE_BASELINE_COMMIT = "6250d282f1d1edeee411f8ef11407cc7c7a445d3"
V2_FREEZE_COMMIT = "de2e320617bdf47d78e5c142bc3bd3e1faf80d80"
FREEZE_SUBJECT = "Freeze physical graph edge handoff qualification"
RESULT_SUBJECT = "Evaluate physical graph edge handoff qualification"

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v1"
)
MATERIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v1_material"
EXTERNAL_REGENERATION_RECEIPT = OUTPUT_ROOT.parent / (
    "physical_graph_edge_handoff_qualification_v1_regeneration_receipt.json"
)
GENESIS_PYTHON = REPO_ROOT / ".generated/venvs/genesis_rocm_0_4_6_v1/bin/python"
VJEPA_PYTHON = Path("/home/andrewknowles/TinyQuadJEPA/bin/python")
REQUIRED_PROCESS_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

FROZEN_ENCODER_SHA256 = (
    "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
)
FROZEN_RANKER_SHA256 = (
    "e1a2a58ff527b4d2bc210f6b1c8fd1d00d87873691db64e8c3ccb2257fa6c127"
)
FROZEN_RANKER_PATH = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "non_greedy_local_subgoal_jepa_planning_v1/checkpoints/"
    "current_visual_reactive_ranker_final_epoch_060.pt"
)
V2_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "occluded_goal_topological_belief_v2"
)

FAMILIES = (
    "STRAIGHT_PASSAGE",
    "TURNING_JUNCTION",
    "OFFSET_OPENING",
    "ROOM_OR_LOOP_EXIT",
)
ROLES = ("DEVELOPMENT", "DEVELOPMENT_HELDOUT")
TARGET_IDS = (
    "TARGET_NODE_CENTRE",
    "DIRECTED_EDGE_PORT",
    "ROUTE_LOOKAHEAD",
)
HELDOUT_CONDITIONS = (
    "DETERMINISTIC_KINEMATICS",
    "FROZEN_CURRENT_VISUAL_RANKER",
    "ORACLE_BEST_ADMISSIBLE_CANDIDATE",
    "TEACHER_TRACE",
)
CANDIDATE_IDS = (
    "straight_fast",
    "straight_medium",
    "straight_slow",
    "arc_left",
    "arc_right",
    "turn_left",
    "turn_right",
    "turn_left_then_go",
    "turn_right_then_go",
    "go_then_turn_left",
    "reverse_then_turn",
    "hold",
)

STATE_COUNT = 64
PROSPECTIVE_POOL_COUNT = 256
DEVELOPMENT_COUNT = 48
HELDOUT_COUNT = 16
CANDIDATE_COUNT = 12
FANOUT_ROW_COUNT = STATE_COUNT * CANDIDATE_COUNT
WAYPOINT_ROW_COUNT = STATE_COUNT * len(TARGET_IDS)
HELDOUT_SCORE_ROW_COUNT = HELDOUT_COUNT * len(HELDOUT_CONDITIONS)
REPEAT_ROW_COUNT = HELDOUT_COUNT * 2 * 2
HORIZON_TICKS = (5, 10, 15)
PHYSICS_STEPS_PER_COMMAND_TICK = 50
PHYSICS_STEPS_PER_BRANCH = HORIZON_TICKS[-1] * PHYSICS_STEPS_PER_COMMAND_TICK
RESET_FIXTURE_TRACE_COUNT = STATE_COUNT * 2
CANDIDATE_TRACE_COUNT = RESET_FIXTURE_TRACE_COUNT + FANOUT_ROW_COUNT + REPEAT_ROW_COUNT
RESET_FIXTURE_PHYSICS_STEPS = PHYSICS_STEPS_PER_BRANCH

SCIENTIFIC_LEAVES = (
    "contract.json",
    "panel_manifest.json",
    "split_manifest.json",
    "graph_manifest.json",
    "state_snapshots.npz",
    "state_snapshot_index.json",
    "teacher_traces.npz",
    "teacher_trace_index.json",
    "edge_port_index.json",
    "waypoint_contracts.json",
    "rgb_observations.npz",
    "pixel_index.json",
    "canonical_latents.npz",
    "latent_index.json",
    "candidate_traces.npz",
    "candidate_fanout.jsonl",
    "development_target_selection.json",
    "heldout_ranker_scores.jsonl",
    "repeated_execution.jsonl",
    "metrics.json",
)
PUBLICATION_LEAVES = ("result.json", "result.md", "file_hashes.json")
ALL_OUTPUT_LEAVES = SCIENTIFIC_LEAVES + PUBLICATION_LEAVES

DOC_PATHS = {
    "contract": REPO_ROOT
    / "docs/lewm_go2_physical_graph_edge_handoff_qualification_v1_contract_2026-09-01.json",
    "fixture": REPO_ROOT
    / "docs/lewm_go2_physical_graph_edge_handoff_qualification_v1_fixture_2026-09-01.json",
    "output_schema": REPO_ROOT
    / "docs/lewm_go2_physical_graph_edge_handoff_qualification_v1_output_schema_2026-09-01.json",
    "preregistration": REPO_ROOT
    / "docs/lewm_go2_physical_graph_edge_handoff_qualification_v1_preregistration_2026-09-01.md",
    "source_closure": REPO_ROOT
    / "docs/lewm_go2_physical_graph_edge_handoff_qualification_v1_source_closure_2026-09-01.json",
}


class ExperimentError(RuntimeError):
    """A source, custody, physical, or scientific invariant failed."""


def canonical_bytes(value: Any) -> bytes:
    helper = getattr(CONTRACT, "canonical_json_bytes", None)
    if callable(helper):
        raw = helper(value)
        if not isinstance(raw, bytes):
            raise ExperimentError("contract canonicalizer did not return bytes")
        return raw.rstrip(b"\n") + b"\n"
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def content_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_digest", None)
    # Document content digests deliberately exclude the storage LF.  Array
    # digests have their own compact-header/NUL/bytes domain below.
    return hashlib.sha256(canonical_bytes(payload)[:-1]).hexdigest()


def attach_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("content_digest", None)
    payload["content_digest"] = content_digest(payload)
    return payload


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    header = canonical_bytes(
        {"shape": list(array.shape), "dtype": array.dtype.str, "layout": "C"}
    )[:-1]
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_regular(path: Path, *, absent_ok: bool = False) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if absent_ok:
            return None
        raise ExperimentError(f"required file is absent: {path}") from None
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ExperimentError(f"path is not an ordinary regular file: {path}")
    return info


def _require_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ExperimentError(f"required directory is absent: {path}") from exc
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise ExperimentError(f"path is not an ordinary directory: {path}")


def atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or path.exists():
        raise ExperimentError(f"output target is not fresh: {path}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_bytes(path, canonical_bytes(dict(value)))


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_bytes(path, b"".join(canonical_bytes(dict(row)) for row in rows))


def atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or path.exists():
        raise ExperimentError(f"NPZ target is not fresh: {path}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    _require_regular(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ExperimentError(f"noncanonical JSON document: {path}")
    if "content_digest" in value and content_digest(value) != value["content_digest"]:
        raise ExperimentError(f"content digest drift: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    _require_regular(path)
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for index, raw in enumerate(stream):
            value = json.loads(raw)
            if not isinstance(value, dict) or raw != canonical_bytes(value):
                raise ExperimentError(f"noncanonical JSONL row {index}: {path}")
            rows.append(value)
    return rows


def binding(path: Path, *, role: str, kind: str, relative_to: Path | None = None) -> dict[str, Any]:
    info = _require_regular(path)
    assert info is not None
    name = str(path if relative_to is None else path.relative_to(relative_to))
    return {
        "role": str(role),
        "path": name,
        "bytes": int(info.st_size),
        "sha256": sha256_file(path),
        "kind": str(kind),
    }


def git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ExperimentError(f"git {' '.join(arguments)} failed: {exc.output}") from exc


def _normalise_version(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return ".".join(str(item) for item in value)
    return str(value)


def _physical_runtime_core(
    executable: Path, *, fake: bool, backend: str = "cpu",
) -> dict[str, Any]:
    """Capture the observed physical interpreter/package/device authority."""

    authority = CONTRACT.RUNTIME_ENVIRONMENT_AUTHORITY["physical"]
    if fake:
        versions = {
            "torch_version": str(authority["torch_version"]),
            "torch_hip_version": str(authority["torch_hip_version"]),
            "genesis_version": str(authority["genesis_version"]),
            "quadrants_version": str(authority["quadrants_version"]),
            "visible_device_count": int(authority["visible_device_count"]),
        }
    else:
        import genesis
        import quadrants
        import torch

        versions = {
            "torch_version": str(torch.__version__),
            "torch_hip_version": str(torch.version.hip),
            "genesis_version": _normalise_version(genesis.__version__),
            "quadrants_version": _normalise_version(quadrants.__version__),
            "visible_device_count": int(torch.cuda.device_count()),
        }
    row = {
        "stage_id": str(authority["stage_id"]),
        "python_executable": str(executable),
        "python_version": sys.version.split()[0],
        **versions,
        "device": "cpu",
        "backend": str(backend),
        "deterministic_environment": {
            key: (
                str(value) if fake else os.environ.get(key)
            )
            for key, value in sorted(REQUIRED_PROCESS_ENV.items())
        },
        "fake_runtime": bool(fake),
    }
    if set(row) != set(CONTRACT.PHYSICAL_RUNTIME_CORE_FIELDS):
        raise ExperimentError("physical runtime core field drift")
    return row


def _visual_runtime_environment(
    executable: Path, *, runtime_role: str, fake: bool,
) -> dict[str, Any]:
    """Capture one frozen encoder/ranker interpreter and model binding."""

    if runtime_role not in {"encoder", "ranker"}:
        raise ExperimentError("visual runtime requires encoder or ranker role")
    authority = CONTRACT.RUNTIME_ENVIRONMENT_AUTHORITY[runtime_role]
    if fake:
        observed = {
            "python_executable": str(executable),
            "python_version": sys.version.split()[0],
            "torch_version": str(authority["torch_version"]),
            "torch_hip_version": str(authority["torch_hip_version"]),
            "visible_device_count": int(authority["visible_device_count"]),
            "device": str(authority["device"]),
            "device_name": copy.deepcopy(authority["device_name"]),
            "device_capability": copy.deepcopy(authority["device_capability"]),
            "backend": str(authority["backend"]),
        }
    else:
        import torch

        device_name = None
        device_capability = None
        if runtime_role == "encoder":
            if not torch.cuda.is_available():
                raise ExperimentError("frozen encoder runtime lacks cuda:0")
            device_name = str(torch.cuda.get_device_name(0))
            device_capability = [int(value) for value in torch.cuda.get_device_capability(0)]
        observed = {
            "python_executable": str(executable),
            "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__),
            "torch_hip_version": str(torch.version.hip),
            "visible_device_count": int(torch.cuda.device_count()),
            "device": str(authority["device"]),
            "device_name": device_name,
            "device_capability": device_capability,
            "backend": str(authority["backend"]),
        }
        model_path = REPO_ROOT / str(authority["model_source_path"])
        info = _require_regular(model_path)
        assert info is not None
        if sha256_file(model_path) != str(authority["model_source_sha256"]):
            raise ExperimentError(f"{runtime_role} model source drift")
    row = {
        "stage_id": str(authority["stage_id"]),
        **observed,
        "fake_runtime": bool(fake),
        "model_role": str(authority["model_role"]),
        "checkpoint_sha256": str(authority["checkpoint_sha256"]),
        "model_source_path": str(authority["model_source_path"]),
        "model_source_sha256": str(authority["model_source_sha256"]),
        "external_repository_path": copy.deepcopy(authority["external_repository_path"]),
        "external_repository_commit": copy.deepcopy(authority["external_repository_commit"]),
        "external_worktree_clean": copy.deepcopy(authority["external_worktree_clean"]),
    }
    return METRICS.validate_visual_runtime_environment(row, runtime_role=runtime_role)


def _bind_physical_backend_runtime(
    core: Mapping[str, Any], backend_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the process observation to the backend that produced a shard."""

    row = copy.deepcopy(dict(core))
    backend = backend_runtime.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ExperimentError("physical backend omitted its runtime identity")
    row["backend"] = backend
    if not bool(row["fake_runtime"]):
        authority = CONTRACT.RUNTIME_ENVIRONMENT_AUTHORITY["physical"]
        if backend != authority["backend"]:
            raise ExperimentError("real physical backend runtime drift")
        policy_device = backend_runtime.get("policy_device")
        if policy_device != row["device"]:
            raise ExperimentError("physical policy device/runtime drift")
    if set(row) != set(CONTRACT.PHYSICAL_RUNTIME_CORE_FIELDS):
        raise ExperimentError("bound physical runtime core field drift")
    return row


def require_stage_runtime(
    kind: str, *, fake: bool = False, visual_role: str | None = None,
) -> dict[str, Any]:
    """Fail before model/simulator creation when an interpreter or env drifts."""

    if kind not in {"ordinary", "physical", "visual"}:
        raise ExperimentError(f"unknown stage runtime kind: {kind}")
    # Persist the invoked interpreter path (the contract binds the venv path),
    # while using resolved identity only for the executable preflight.
    executable = Path(sys.executable)
    if not fake:
        expected = GENESIS_PYTHON if kind == "physical" else VJEPA_PYTHON if kind == "visual" else None
        if expected is not None and executable.resolve() != expected.resolve():
            raise ExperimentError(
                f"{kind} stage requires {expected}, observed {executable}"
            )
        if kind == "physical":
            drift = {
                key: {"expected": value, "observed": os.environ.get(key)}
                for key, value in REQUIRED_PROCESS_ENV.items()
                if os.environ.get(key) != value
            }
            if drift:
                raise ExperimentError(
                    f"physical-stage deterministic environment drift: {drift}"
                )
    if kind == "physical":
        if visual_role is not None:
            raise ExperimentError("physical runtime cannot carry a visual role")
        return _physical_runtime_core(executable, fake=fake)
    if kind == "visual":
        if visual_role is None:
            raise ExperimentError("visual runtime role is required")
        return _visual_runtime_environment(
            executable, runtime_role=visual_role, fake=fake,
        )
    if visual_role is not None:
        raise ExperimentError("ordinary runtime cannot carry a visual role")
    return {
        "kind": kind,
        "python_executable": str(executable),
        "python_version": sys.version.split()[0],
        "environment": {
            key: os.environ.get(key) for key in sorted(REQUIRED_PROCESS_ENV)
        },
        "fake": bool(fake),
    }


def require_runtime_source_freeze() -> str:
    head = git("rev-parse", "HEAD")
    if git("show", "-s", "--format=%s", head) != FREEZE_SUBJECT:
        raise ExperimentError("runtime HEAD is not the required contract-freeze subject")
    if git("rev-parse", f"{head}^") != PARENT_COMMIT:
        raise ExperimentError("contract freeze is not a direct child of the bound result")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ExperimentError("runtime worktree is not clean at the source freeze")
    tracked = tuple(getattr(CONTRACT, "TRACKED_SOURCE_PATHS", ()))
    if not tracked:
        tracked = (
            "lewm/safety/physical_graph_edge_handoff_qualification_v1_contract.py",
            "lewm/safety/physical_graph_edge_handoff_qualification_v1_metrics.py",
            "scripts/run_physical_graph_edge_handoff_qualification_v1.py",
            "scripts/evaluate_physical_graph_edge_handoff_qualification_v1.py",
            "lewm/tests/test_physical_graph_edge_handoff_qualification_v1.py",
            "lewm/tests/test_run_physical_graph_edge_handoff_qualification_v1.py",
            "lewm/tests/test_evaluate_physical_graph_edge_handoff_qualification_v1.py",
        )
    for relative in tracked:
        path = REPO_ROOT / str(relative)
        _require_regular(path)
        if path.read_bytes() != subprocess.check_output(
            ["git", "show", f"{head}:{relative}"], cwd=REPO_ROOT
        ):
            raise ExperimentError(f"runtime source differs from frozen HEAD: {relative}")
    _validate_runtime_source_closure()
    return head


def _validate_runtime_source_closure() -> dict[str, Any]:
    """Re-hash every executable dependency named by the frozen closure doc."""

    closure_path = DOC_PATHS["source_closure"]
    closure_info = _require_regular(closure_path)
    assert closure_info is not None
    if closure_info.st_nlink != 1:
        raise ExperimentError("source-closure document is not single-linked")
    closure = load_json(closure_path)
    try:
        CONTRACT.validate_content_digest(closure)
    except Exception as exc:
        raise ExperimentError("source-closure content digest drift") from exc
    if set(closure) != {
        "schema", "parent_commit", "row_count", "rows", "content_digest"
    } or closure["schema"] != (
        "physical_graph_edge_handoff_qualification_v1.source_closure.v1"
    ) or closure["parent_commit"] != PARENT_COMMIT:
        raise ExperimentError("source-closure identity drift")
    expected_paths = [str(value) for value in CONTRACT.SOURCE_CLOSURE_PATHS]
    rows = closure["rows"]
    if (
        not isinstance(rows, list)
        or int(closure["row_count"]) != len(expected_paths)
        or len(rows) != len(expected_paths)
    ):
        raise ExperimentError("source-closure cardinality drift")
    observed_paths: list[str] = []
    validated: list[dict[str, Any]] = []
    repo = REPO_ROOT.resolve(strict=True)
    for index, (row_value, expected_relative) in enumerate(zip(rows, expected_paths)):
        if not isinstance(row_value, Mapping) or set(row_value) != {
            "path", "bytes", "sha256"
        }:
            raise ExperimentError(f"source-closure row field drift: {index}")
        relative = str(row_value["path"])
        candidate = Path(relative)
        if (
            relative != expected_relative
            or not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != relative
        ):
            raise ExperimentError(f"source-closure path/order drift: {index}")
        path = REPO_ROOT / candidate
        info = _require_regular(path)
        assert info is not None
        if info.st_nlink != 1 or path.resolve(strict=True) != repo / candidate:
            raise ExperimentError(f"source-closure path is linked or escapes: {relative}")
        observed = {
            "path": relative,
            "bytes": int(info.st_size),
            "sha256": sha256_file(path),
        }
        if dict(row_value) != observed:
            raise ExperimentError(f"source-closure live binding drift: {relative}")
        observed_paths.append(relative)
        validated.append(observed)
    if observed_paths != expected_paths or len(set(observed_paths)) != len(observed_paths):
        raise ExperimentError("source-closure ordered inventory drift")
    return {
        "path": str(closure_path),
        "row_count": len(validated),
        "content_digest": str(closure["content_digest"]),
        "validated": True,
    }


def _prospective_pool_specs() -> list[dict[str, Any]]:
    """Build only role-free, outcome-free identities and frozen balance knobs."""

    builder = getattr(CONTRACT, "build_candidate_specs", None)
    if not callable(builder):
        raise ExperimentError("contract-owned build_candidate_specs is absent")
    rows = [copy.deepcopy(dict(value)) for value in builder()]
    expected = list(range(int(getattr(CONTRACT, "PROSPECTIVE_POOL_COUNT", PROSPECTIVE_POOL_COUNT))))
    if len(rows) != len(expected):
        raise ExperimentError("prospective pool cardinality/order drift")
    expected_digest = CONTRACT.build_contract()["panel"]["candidate_specs_sha256"]
    if hashlib.sha256(canonical_bytes(rows)[:-1]).hexdigest() != expected_digest:
        raise ExperimentError("contract candidate-spec population digest drift")
    if set(str(row["family"]) for row in rows) != set(FAMILIES):
        raise ExperimentError("pool family identity drift")
    if any(set(row) != set(rows[0]) or row.get("role") is not None for row in rows):
        raise ExperimentError("prospective pool field or role drift")
    if len({str(row["scene_id"]) for row in rows}) != len(rows):
        raise ExperimentError("scene identities are not unique")
    turns = [row for row in rows if row["family"] == "TURNING_JUNCTION"]
    expected_per_side = len(turns) // 2
    if sum(row["route_direction"] == "LEFT" for row in turns) != expected_per_side:
        raise ExperimentError("prospective TURNING_JUNCTION balance drift")
    return rows


def _scene_exclusion_audit(specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reproduce the frozen broad predecessor projection, then compare the pool.

    Nothing is optional here: every byte/hash-bound predecessor authority must
    exist and agree before a simulator may be constructed.
    """

    from lewm.safety import occluded_goal_topological_belief_v1_contract as V1C
    from scripts import run_occluded_goal_topological_belief_v1 as V1RUN

    inherited = dict(CONTRACT.PRIOR_SCENE_EXCLUSION_AUTHORITY["inherited_authority"])
    if inherited != dict(V1C.PRIOR_PANEL_EXCLUSION_AUTHORITY):
        raise ExperimentError("inherited ten-authority exclusion contract drift")
    documents = V1RUN._load_bound_prior_authorities()  # noqa: SLF001
    prior = V1RUN._project_prior_identities(documents)  # noqa: SLF001

    v2_binding = dict(CONTRACT.PRIOR_SCENE_EXCLUSION_AUTHORITY["v2_panel_binding"])
    v2_path = Path(str(v2_binding["path"]))
    info = _require_regular(v2_path)
    assert info is not None
    if info.st_size != int(v2_binding["bytes"]) or sha256_file(v2_path) != str(
        v2_binding["sha256"]
    ):
        raise ExperimentError("bound V2 panel exclusion authority drift")
    v2_panel = json.loads(v2_path.read_bytes())
    if not isinstance(v2_panel, Mapping):
        raise ExperimentError("bound V2 panel is not an object")
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

    current_scenes = {str(row["scene_id"]) for row in specs}
    current_scene_hashes = {
        hashlib.sha256(value.encode("utf-8")).hexdigest() for value in current_scenes
    }
    current_identities = {
        str(row[field]) for row in specs for field in ("state_id", "episode_id", "graph_id")
    }
    current_seeds = {int(row["procedural_seed"]) for row in specs}
    current_geometry = {str(row["candidate_spec_id"]) for row in specs}
    current_structured = {
        canonical_bytes(
            [
                str(row["family"]),
                str(row["route_direction"]),
                int(row["stratum_index"]),
                int(row["variant_index"]),
            ]
        )[:-1]
        for row in specs
    }
    prior_structured = {
        canonical_bytes(list(value))[:-1]
        for value in prior["structured_waypoint_or_sequence_path"]
    }
    broad = CONTRACT.PRIOR_SCENE_EXCLUSION_AUTHORITY["broad_prior_plus_v2_projection"]
    checks = (
        ("scene_identity", "scene_identity_count", "scene_identity_canonical_json_sha256", False),
        ("scene_identity_sha256", "scene_identity_sha256_count", "scene_identity_sha256_canonical_json_sha256", False),
        ("episode_or_state_identity", "episode_state_or_graph_identity_count", "episode_state_or_graph_identity_canonical_json_sha256", False),
        ("numeric_seed", "numeric_seed_count", "numeric_seed_canonical_json_sha256", True),
        ("textual_path_geometry_or_source_identity", "textual_path_geometry_or_source_identity_count", "textual_path_geometry_or_source_identity_canonical_json_sha256", False),
    )
    for key, count_key, digest_key, numeric in checks:
        values = list(prior[key])
        if len(values) != int(broad[count_key]) or V1RUN._identity_projection_digest(  # noqa: SLF001
            values, numeric=numeric
        ) != str(broad[digest_key]):
            raise ExperimentError(f"broad predecessor projection drift: {key}")
    structured_values = list(prior["structured_waypoint_or_sequence_path"])
    if (
        len(structured_values) != int(broad["structured_waypoint_or_sequence_path_count"])
        or V1RUN._structured_identity_projection_digest(structured_values)  # noqa: SLF001
        != str(broad["structured_waypoint_or_sequence_path_canonical_json_sha256"])
    ):
        raise ExperimentError("broad predecessor structured projection drift")
    counts = {
        "scene_overlap_count": len(current_scenes & prior["scene_identity"]),
        "scene_hash_overlap_count": len(
            current_scene_hashes & prior["scene_identity_sha256"]
        ),
        "state_or_episode_overlap_count": len(
            current_identities & prior["episode_or_state_identity"]
        ),
        "seed_overlap_count": len(current_seeds & prior["numeric_seed"]),
        "path_or_geometry_overlap_count": len(
            current_geometry & prior["textual_path_geometry_or_source_identity"]
        ),
        "structured_path_overlap_count": len(current_structured & prior_structured),
    }
    if any(counts.values()):
        raise ExperimentError(f"prospective pool overlaps prior authorities: {counts}")
    return {
        "authority_digest": hashlib.sha256(
            canonical_bytes(CONTRACT.PRIOR_SCENE_EXCLUSION_AUTHORITY)[:-1]
        ).hexdigest(),
        "checked_before_simulator_creation": True,
        **counts,
        "all_zero": True,
    }


def build_freeze_documents() -> dict[str, Any]:
    if git("rev-parse", "HEAD") != PARENT_COMMIT:
        raise ExperimentError("freeze documents must be generated on the bound result commit")
    specs = _prospective_pool_specs()
    scientific = CONTRACT.build_contract()
    document = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.contract_document.v1",
            "status": "FROZEN_BEFORE_PHYSICAL_COLLECTION",
            "parent_commit": PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "v2_freeze_commit": V2_FREEZE_COMMIT,
            "required_freeze_subject": FREEZE_SUBJECT,
            "required_result_subject": RESULT_SUBJECT,
            "scientific_contract": scientific,
        }
    )
    fixture = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.fixture.v1",
            "prospective_pool_states": len(specs),
            "selected_states": STATE_COUNT,
            "development": DEVELOPMENT_COUNT,
            "development_heldout": HELDOUT_COUNT,
            "fanout_rows": FANOUT_ROW_COUNT,
            "waypoint_rows": WAYPOINT_ROW_COUNT,
            "heldout_rows": HELDOUT_SCORE_ROW_COUNT,
            "repeat_rows": REPEAT_ROW_COUNT,
            "restore_authority": "full_mutable_genesis_solver_controller_harness_rng_snapshot",
            "restore_repetitions_before_fanout": 2,
            "candidate_physics_steps": PHYSICS_STEPS_PER_BRANCH,
        }
    )
    output_schema = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.output_schema.v1",
            "root": str(OUTPUT_ROOT),
            "scientific_leaves": list(SCIENTIFIC_LEAVES),
            "publication_leaves": list(PUBLICATION_LEAVES),
            "external_regeneration_receipt": str(EXTERNAL_REGENERATION_RECEIPT),
            "all_leaves_unconditional": True,
        }
    )
    preregistration = "\n".join(
        [
            "# PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V1",
            "",
            "Development-only physical graph-edge handoff qualification on 64 fresh scene-disjoint Genesis states.",
            "",
            "Every candidate is executed from a full reviewed solver/controller/harness/RNG restore. Two exact restore-and-replay witnesses must pass before a state's twelve-candidate fanout is admissible.",
            "",
            "Three target representations are selected on 48 development states only. The selected target is frozen before any 16-state development-held-out candidate outcome is opened. No model is trained.",
            "",
        ]
    )
    atomic_json(DOC_PATHS["contract"], document)
    atomic_json(DOC_PATHS["fixture"], fixture)
    atomic_json(DOC_PATHS["output_schema"], output_schema)
    atomic_bytes(DOC_PATHS["preregistration"], preregistration.encode("utf-8"))
    source_paths = tuple(getattr(CONTRACT, "SOURCE_CLOSURE_PATHS", ()))
    if not source_paths:
        raise ExperimentError("contract-owned SOURCE_CLOSURE_PATHS is absent")
    rows = []
    for relative in source_paths:
        path = REPO_ROOT / str(relative)
        info = _require_regular(path)
        assert info is not None
        rows.append({"path": str(relative), "bytes": info.st_size, "sha256": sha256_file(path)})
    atomic_json(
        DOC_PATHS["source_closure"],
        attach_digest(
            {
                "schema": "physical_graph_edge_handoff_qualification_v1.source_closure.v1",
                "parent_commit": PARENT_COMMIT,
                "row_count": len(rows),
                "rows": rows,
            }
        ),
    )
    return document


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _artifact_from_contract(value: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    path = Path(str(value["checkpoint_path"]))
    info = _require_regular(path)
    assert info is not None
    if info.st_size != int(value["checkpoint_size_bytes"]) or sha256_file(path) != str(
        value["checkpoint_sha256"]
    ):
        raise ExperimentError(f"frozen external artifact drift: {role}")
    return {
        "role": role,
        "path": str(path),
        "bytes": int(info.st_size),
        "sha256": str(value["checkpoint_sha256"]),
        "kind": "frozen_checkpoint_read_only",
    }


def _runtime_contract(freeze: str) -> dict[str, Any]:
    runtime = CONTRACT.build_runtime_contract(freeze)
    for row in runtime["external_artifact_bindings"]:
        path = Path(str(row["path"]))
        info = _require_regular(path)
        assert info is not None
        if info.st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]):
            raise ExperimentError(f"frozen external artifact drift: {row['role']}")
    predecessor = dict(runtime["predecessor_result_binding"])
    path = Path(str(predecessor["path"]))
    info = _require_regular(path)
    assert info is not None
    if info.st_size != int(predecessor["bytes"]) or sha256_file(path) != str(predecessor["sha256"]):
        raise ExperimentError("bound V2 predecessor result changed")
    CONTRACT.validate_runtime_contract(runtime, source_freeze_commit=freeze)
    return runtime


def initialize_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    """Bind source/models and the complete role-free pool before simulation."""

    require_stage_runtime("ordinary", fake=fake_runtime)
    freeze = require_runtime_source_freeze()
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise ExperimentError(f"official output root is not fresh: {OUTPUT_ROOT}")
    if MATERIAL_ROOT.exists() or MATERIAL_ROOT.is_symlink():
        raise ExperimentError(f"material root is not fresh: {MATERIAL_ROOT}")
    specs = _prospective_pool_specs()
    exclusion = _scene_exclusion_audit(specs)
    runtime_contract = _runtime_contract(freeze)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    MATERIAL_ROOT.mkdir(parents=True, exist_ok=False)
    for leaf in ("qualification", "selected", "fanout", "repeat", "scenes"):
        (MATERIAL_ROOT / leaf).mkdir()
    atomic_json(OUTPUT_ROOT / "contract.json", runtime_contract)
    pool = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.prospective_pool.v1",
            "experiment_id": EXPERIMENT_ID,
            "source_freeze_commit": freeze,
            "selection_rule": copy.deepcopy(
                CONTRACT.GEOMETRY_AUTHORITY["teacher_only_scan"]
            ),
            "prior_exclusion_evidence": exclusion,
            "specs": specs,
        }
    )
    atomic_json(MATERIAL_ROOT / "prospective_pool.json", pool)
    atomic_json(
        MATERIAL_ROOT / "material_contract.json",
        attach_digest(
            {
                "schema": "physical_graph_edge_handoff_qualification_v1.material_contract.v1",
                "experiment_id": EXPERIMENT_ID,
                "source_freeze_commit": freeze,
                "official_contract_sha256": sha256_file(OUTPUT_ROOT / "contract.json"),
                "prospective_pool_sha256": sha256_file(MATERIAL_ROOT / "prospective_pool.json"),
                "expected_qualification_shards": len(specs),
                "expected_selected_states": STATE_COUNT,
                "scratch_is_not_scientific_result_identity": True,
                "started_at_unix_s": float(time.time()),
            }
        ),
    )
    return pool


def _require_initialized() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = require_runtime_source_freeze()
    runtime = load_json(OUTPUT_ROOT / "contract.json")
    material = load_json(MATERIAL_ROOT / "material_contract.json")
    pool = load_json(MATERIAL_ROOT / "prospective_pool.json")
    if runtime.get("source_freeze_commit") != freeze:
        raise ExperimentError("runtime source-freeze binding drift")
    validator = getattr(CONTRACT, "validate_runtime_contract", None)
    if callable(validator):
        validator(runtime, source_freeze_commit=freeze)
    if (
        material.get("official_contract_sha256") != sha256_file(OUTPUT_ROOT / "contract.json")
        or material.get("prospective_pool_sha256") != sha256_file(MATERIAL_ROOT / "prospective_pool.json")
    ):
        raise ExperimentError("material/runtime binding drift")
    return runtime, pool


def body_from_world(world_pose: Sequence[float], body_pose: Sequence[float]) -> list[float]:
    wx, wy, wyaw = (float(value) for value in world_pose)
    bx, by, byaw = (float(value) for value in body_pose)
    cosine, sine = math.cos(byaw), math.sin(byaw)
    dx, dy = wx - bx, wy - by
    return [cosine * dx + sine * dy, -sine * dx + cosine * dy, _wrap_angle(wyaw - byaw)]


def world_from_body(body_target: Sequence[float], body_pose: Sequence[float]) -> list[float]:
    dx, dy, dyaw = (float(value) for value in body_target)
    bx, by, byaw = (float(value) for value in body_pose)
    cosine, sine = math.cos(byaw), math.sin(byaw)
    return [bx + cosine * dx - sine * dy, by + sine * dx + cosine * dy, _wrap_angle(byaw + dyaw)]


def _segment_intersection_fraction(
    start: Sequence[float], end: Sequence[float], left: Sequence[float], right: Sequence[float]
) -> float | None:
    px, py = float(start[0]), float(start[1])
    rx, ry = float(end[0]) - px, float(end[1]) - py
    qx, qy = float(left[0]), float(left[1])
    sx, sy = float(right[0]) - qx, float(right[1]) - qy
    cross = rx * sy - ry * sx
    if abs(cross) <= 1.0e-12:
        return None
    qpx, qpy = qx - px, qy - py
    t = (qpx * sy - qpy * sx) / cross
    u = (qpx * ry - qpy * rx) / cross
    return float(t) if -1.0e-12 <= t <= 1.0 + 1.0e-12 and -1.0e-12 <= u <= 1.0 + 1.0e-12 else None


def first_teacher_port(
    base_pose_world: Any,
    source_region_member: Any,
    edge_region_member: Any,
    source_boundary_polygon_world: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Derive the port only from the first observed teacher boundary exit."""

    import numpy as np

    poses = np.asarray(base_pose_world, dtype=np.float64)
    source = np.asarray(source_region_member, dtype=np.uint8)
    edge = np.asarray(edge_region_member, dtype=np.uint8)
    polygon = [list(map(float, point)) for point in source_boundary_polygon_world]
    if poses.ndim != 2 or poses.shape[1] != 7 or source.shape != (len(poses),) or edge.shape != source.shape:
        raise ExperimentError("teacher port input shape drift")
    if len(polygon) < 3:
        raise ExperimentError("source boundary polygon is not physical")
    for index in range(1, len(poses)):
        if source[index - 1] != 1 or source[index] != 0 or edge[index] != 1:
            continue
        candidates: list[float] = []
        for vertex, next_vertex in zip(polygon, polygon[1:] + polygon[:1]):
            fraction = _segment_intersection_fraction(
                poses[index - 1, :2], poses[index, :2], vertex, next_vertex
            )
            if fraction is not None:
                candidates.append(fraction)
        if not candidates:
            raise ExperimentError("teacher membership exit has no geometric boundary crossing")
        fraction = min(candidates)
        xy = poses[index - 1, :2] + fraction * (
            poses[index, :2] - poses[index - 1, :2]
        )
        # Quaternion yaw interpolation is deliberately avoided.  The local port
        # tangent is the physical segment direction at the crossing.
        delta = poses[index, :2] - poses[index - 1, :2]
        heading = math.atan2(float(delta[1]), float(delta[0]))
        return {
            "sample_before": index - 1,
            "sample_after": index,
            "fraction": float(fraction),
            "port_world": [float(xy[0]), float(xy[1]), float(heading)],
        }
    raise ExperimentError("teacher trace never exits source through the directed edge")


def canonical_port_crossing(
    base_pose_world: Any,
    target_region_member: Any,
    opening_segment_world: Sequence[Sequence[float]],
    opening_normal_world: Sequence[float],
    competing_edges: Sequence[Mapping[str, Any]],
    *,
    sustained_samples: int = 100,
) -> dict[str, Any]:
    """Find the first positive-normal transverse-port crossing from physics rows."""

    import numpy as np

    poses = np.asarray(base_pose_world, dtype=np.float64)
    target = np.asarray(target_region_member, dtype=np.uint8)
    opening = np.asarray(opening_segment_world, dtype=np.float64)
    normal = np.asarray(opening_normal_world, dtype=np.float64)
    if (
        poses.ndim != 2
        or poses.shape[1] != 7
        or target.shape != (len(poses),)
        or opening.shape != (2, 2)
        or normal.shape != (2,)
        or not np.isfinite(poses).all()
        or not np.isfinite(opening).all()
        or not np.isfinite(normal).all()
    ):
        raise ExperimentError("canonical port geometry/trace shape drift")
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise ExperimentError("canonical port normal is degenerate")
    normal = normal / norm
    midpoint = opening.mean(axis=0)
    tangent = opening[1] - opening[0]
    width = float(np.linalg.norm(tangent))
    if width <= 1.0e-12:
        raise ExperimentError("canonical port opening is degenerate")
    tangent = tangent / width

    registered = _first_registered_port_crossing(
        poses,
        {
            "edge_id": "selected-edge",
            "opening_segment_world": opening.tolist(),
            "opening_normal_world": normal.tolist(),
        },
        competing_edges,
    )
    if registered is None:
        raise ExperimentError("teacher trace never crosses the canonical directed port")
    if not bool(registered["is_selected_edge"]):
        raise ExperimentError("teacher enters a competing physical port first")
    selected = registered
    count, target_early = _consecutive_beyond_samples(
        poses, int(selected["sample_after"]), opening, normal, target,
    )
    if count < int(sustained_samples) and not target_early:
        raise ExperimentError("teacher did not remain beyond the port for 100 physics samples")
    heading = math.atan2(float(normal[1]), float(normal[0]))
    return {
        **selected,
        "port_world": [*selected["point_world"], heading],
        "competing_crossing_before": False,
        "sustained_beyond_samples": int(count),
        "sustained_or_target_reached": True,
    }


def _first_registered_port_crossing(
    base_pose_world: Any,
    selected_edge: Mapping[str, Any],
    competing_edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Project the pure all-port crossing authority into runner row names."""

    import numpy as np

    poses = np.asarray(base_pose_world, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ExperimentError("registered-port pose shape drift")
    try:
        raw = METRICS.first_registered_port_crossing(
            poses[:, :2].tolist(), selected_edge, competing_edges,
            tolerance_m=float(CONTRACT.NUMERICAL_TOLERANCES["se2_position_m"]),
        )
    except Exception as exc:
        raise ExperimentError("pure registered-port crossing authority failed") from exc
    if raw is None:
        return None
    edge_id = str(raw["edge_id"])
    edge = selected_edge if bool(raw["is_selected_edge"]) else next(
        value for value in competing_edges if str(value["edge_id"]) == edge_id
    )
    opening = np.asarray(edge["opening_segment_world"], dtype=np.float64)
    tangent = opening[1] - opening[0]
    width = float(np.linalg.norm(tangent))
    tangent /= width
    midpoint = opening.mean(axis=0)
    point = np.asarray(raw["crossing_point_world_xy"], dtype=np.float64)
    return {
        "edge_id": edge_id,
        "is_selected_edge": bool(raw["is_selected_edge"]),
        "sample_before": int(raw["crossing_sample_before"]),
        "sample_after": int(raw["crossing_sample_after"]),
        "fraction": float(raw["crossing_fraction"]),
        "normal_dot_displacement_m": float(raw["directed_normal_displacement_m"]),
        "lateral_coordinate_m": float((point - midpoint) @ tangent),
        "lateral_fraction": float(raw["lateral_fraction"]),
        "point_world": [float(point[0]), float(point[1])],
    }


def _first_positive_port_crossing(
    base_pose_world: Any,
    opening_segment_world: Sequence[Sequence[float]],
    opening_normal_world: Sequence[float],
) -> dict[str, Any] | None:
    """Return the first directed transverse crossing, without dwell semantics."""

    import numpy as np

    poses = np.asarray(base_pose_world, dtype=np.float64)
    opening = np.asarray(opening_segment_world, dtype=np.float64)
    normal = np.asarray(opening_normal_world, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or opening.shape != (2, 2) or normal.shape != (2,):
        raise ExperimentError("directed-port crossing input shape drift")
    norm = float(np.linalg.norm(normal))
    width = float(np.linalg.norm(opening[1] - opening[0]))
    if norm <= 1.0e-12 or width <= 1.0e-12:
        raise ExperimentError("directed-port geometry is degenerate")
    normal /= norm
    tangent = (opening[1] - opening[0]) / width
    midpoint = opening.mean(axis=0)
    for after in range(1, len(poses)):
        before_xy = poses[after - 1, :2]
        after_xy = poses[after, :2]
        try:
            crossing = METRICS.transverse_port_crossing(
                before_xy.tolist(), after_xy.tolist(), opening.tolist(), normal.tolist(),
                tolerance_m=float(CONTRACT.NUMERICAL_TOLERANCES["se2_position_m"]),
            )
        except Exception as exc:
            raise ExperimentError("pure directed-port crossing authority failed") from exc
        if crossing is None:
            continue
        fraction = float(crossing["crossing_fraction"])
        displacement = after_xy - before_xy
        normal_dot = float(crossing["directed_normal_displacement_m"])
        point = np.asarray(crossing["crossing_point_world_xy"], dtype=np.float64)
        lateral_coordinate = float((point - midpoint) @ tangent)
        return {
            "sample_before": after - 1,
            "sample_after": after,
            "fraction": fraction,
            "normal_dot_displacement_m": normal_dot,
            "lateral_coordinate_m": lateral_coordinate,
            "lateral_fraction": float(crossing["lateral_fraction"]),
            "point_world": [float(point[0]), float(point[1])],
        }
    return None


def _first_competing_crossing(
    base_pose_world: Any, competing_edges: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    result: list[tuple[int, float, str, dict[str, Any]]] = []
    for edge in competing_edges:
        crossing = _first_positive_port_crossing(
            base_pose_world,
            edge["opening_segment_world"],
            edge["opening_normal_world"],
        )
        if crossing is not None:
            result.append(
                (
                    int(crossing["sample_after"]),
                    float(crossing["fraction"]),
                    str(edge["edge_id"]),
                    crossing,
                )
            )
    if not result:
        return None
    _after, _fraction, edge_id, crossing = min(result)
    return {"edge_id": edge_id, **crossing}


def _teacher_route_progress_m(
    base_pose_world: Any, opening_segment_world: Sequence[Sequence[float]]
) -> float:
    """Frozen candidate-blind progress: reduction in distance to port midpoint."""

    import numpy as np

    poses = np.asarray(base_pose_world, dtype=np.float64)
    opening = np.asarray(opening_segment_world, dtype=np.float64)
    midpoint = opening.mean(axis=0)
    distances = np.linalg.norm(poses[:, :2] - midpoint[None, :], axis=1)
    return float(distances[0] - distances.min())


TEACHER_TRACE_MEMBERS = (
    "timestamp_s",
    "base_pose_world",
    "base_twist_world",
    "joint_position",
    "joint_velocity",
    "applied_command",
    "requested_command",
    "physics_contact",
    "source_region_member",
    "edge_region_member",
    "target_region_member",
)
CANDIDATE_TRACE_MEMBERS = (
    "timestamp_s",
    "base_pose_world",
    "base_twist_world",
    "joint_position",
    "joint_velocity",
    "requested_command",
    "post_slew_applied_command",
    "physics_contact",
    "source_region_member",
    "correct_edge_region_member",
    "wrong_edge_region_member",
    "target_region_member",
)
TRACE_DT_S = 0.002


def _collect_solver_fields_compat(scene: Any) -> list[tuple[str, Any]]:
    """Quadrants-compatible port of the reviewed pilot field inventory.

    Genesis 0.4.6 in the bound CPU runtime is backed by Quadrants 0.6.2, not
    the historical ``gstaichi`` module imported by the reviewed pilot.  The
    walk, depth, ordering, and two immutable geometry exclusions are kept
    byte-for-byte equivalent in meaning; an unknown field runtime fails
    before snapshot capture.
    """

    import numpy as np
    import quadrants as qd

    field_types = (qd.Field, qd.Ndarray)
    static_markers = (
        "._sdf_info.geoms_sdf_",
        "._support_field_info.support_",
    )
    found: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def walk(obj: Any, path: str, depth: int) -> None:
        if id(obj) in seen or depth > 4:
            return
        seen.add(id(obj))
        for name, value in sorted(getattr(obj, "__dict__", {}).items()):
            if name.startswith("__"):
                continue
            sub = f"{path}.{name}"
            if isinstance(value, field_types):
                if not any(marker in sub for marker in static_markers):
                    if not callable(getattr(value, "to_numpy", None)) or not callable(
                        getattr(value, "from_numpy", None)
                    ):
                        raise ExperimentError(
                            f"unsupported mutable Quadrants field API at {sub}"
                        )
                    found.append((sub, value))
            elif hasattr(value, "__dict__") and not isinstance(
                value, (str, bytes, np.ndarray, type)
            ):
                walk(value, sub, depth + 1)

    walk(scene, type(scene).__name__, 0)
    for solver in scene.active_solvers:
        walk(solver, type(solver).__name__, 0)
    if not found or len({path for path, _field in found}) != len(found):
        raise ExperimentError("Quadrants mutable solver-field inventory is empty or ambiguous")
    return found


def _reset_robot_to_fixed_spawn_compat(runner: Any) -> None:
    """Graph-free exact fixed-spawn reset for the synthetic in-memory pack.

    This is the fixed-spawn subset of RolloutRunner._reset_robot_to_spawn:
    exact pack pose, production reset stance, zero joint velocity, and policy
    latency reset.  It intentionally never invokes route collectors, a
    planning grid, randomized spawn selection, or a fabricated manifest.
    """

    import numpy as np

    if int(runner.n_envs) != 1 or bool(runner.config.randomize_spawn_pose):
        raise ExperimentError("fixed-spawn compatibility reset requires one nonrandom env")
    envs = [0]
    xyz = np.asarray(runner.pack.robot.spawn_xyz_m, dtype=np.float32).reshape(1, 3)
    quat = np.asarray(runner.pack.robot.spawn_quat_wxyz, dtype=np.float32).reshape(1, 4)
    stance = np.asarray(
        getattr(runner.policy, "reset_stance_rad", runner._stance), dtype=np.float32
    ).reshape(1, 12)
    runner._spawn_xyz_per_env[:] = xyz
    runner._spawn_quat_wxyz_per_env[:] = quat
    runner.build.robot.set_pos(xyz, envs_idx=envs, zero_velocity=True)
    runner.build.robot.set_quat(quat, envs_idx=envs, zero_velocity=False)
    runner.build.robot.set_dofs_position(
        stance, runner._leg_dof_idx.tolist(), envs_idx=envs
    )
    runner.build.robot.set_dofs_velocity(
        np.zeros_like(stance), runner._leg_dof_idx.tolist(), envs_idx=envs
    )
    reset = getattr(runner.policy, "reset", None)
    if callable(reset):
        reset(envs)
    runner._last_executed.fill(0.0)
    runner._blocks_in_episode.fill(0)
    runner._consecutive_tipped_blocks.fill(0)
    runner._recovery_interlock_blocks_remaining.fill(0)


def _normalise_trace(value: Mapping[str, Any], *, kind: str, exact_samples: int | None) -> dict[str, Any]:
    import numpy as np

    members = TEACHER_TRACE_MEMBERS if kind == "teacher" else CANDIDATE_TRACE_MEMBERS
    if set(value) != set(members):
        raise ExperimentError(f"{kind} trace member inventory drift")
    widths = {
        "timestamp_s": (),
        "base_pose_world": (7,),
        "base_twist_world": (6,),
        "joint_position": (12,),
        "joint_velocity": (12,),
        "applied_command": (3,),
        "requested_command": (3,),
        "post_slew_applied_command": (3,),
        "physics_contact": (),
        "source_region_member": (),
        "edge_region_member": (),
        "correct_edge_region_member": (),
        "wrong_edge_region_member": (),
        "target_region_member": (),
    }
    byte_members = {
        "physics_contact",
        "source_region_member",
        "edge_region_member",
        "correct_edge_region_member",
        "wrong_edge_region_member",
        "target_region_member",
    }
    result: dict[str, Any] = {}
    sample_count: int | None = None
    for member in members:
        dtype = np.uint8 if member in byte_members else np.float64
        array = np.ascontiguousarray(np.asarray(value[member], dtype=dtype))
        expected_tail = widths[member]
        if array.ndim != 1 + len(expected_tail) or tuple(array.shape[1:]) != expected_tail:
            raise ExperimentError(f"{kind}.{member} shape drift: {array.shape}")
        if sample_count is None:
            sample_count = int(array.shape[0])
        elif int(array.shape[0]) != sample_count:
            raise ExperimentError(f"{kind} trace sample alignment drift")
        if member in byte_members:
            if not np.isin(array, [0, 1]).all():
                raise ExperimentError(f"{kind}.{member} is not binary")
        elif not np.isfinite(array).all():
            raise ExperimentError(f"{kind}.{member} contains non-finite values")
        result[member] = array
    assert sample_count is not None
    if sample_count < 2 or (exact_samples is not None and sample_count != exact_samples):
        raise ExperimentError(
            f"{kind} trace sample count drift: {sample_count}, expected {exact_samples}"
        )
    timestamps = result["timestamp_s"]
    if not np.allclose(np.diff(timestamps), TRACE_DT_S, rtol=0.0, atol=1.0e-12):
        raise ExperimentError(f"{kind} trace is not sampled after every 2 ms physics step")
    return result


def _trace_digest_projection(trace: Mapping[str, Any]) -> dict[str, str]:
    return {name: canonical_array_sha256(trace[name]) for name in sorted(trace)}


def _reset_trace_pair_matches(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[bool, dict[str, float]]:
    """Apply the frozen full-trace reset comparison, never a backend flag."""

    import numpy as np

    authority = dict(CONTRACT.RESET_TRACE_PAIR_COMPARISON_AUTHORITY)
    if len(first["timestamp_s"]) != int(authority["physics_samples_per_trial"]):
        raise ExperimentError("reset trace length differs from frozen comparison authority")
    maxima: dict[str, float] = {}
    for member in authority["exact_members"]:
        if not np.array_equal(first[member], second[member]):
            return False, {f"exact:{member}": float("inf")}
    checks = (
        ("base_pose_world_position_xyz", "base_pose_world", slice(0, 3)),
        ("base_pose_world_quaternion_xyzw", "base_pose_world", slice(3, 7)),
        ("base_twist_world", "base_twist_world", slice(None)),
        ("joint_position", "joint_position", slice(None)),
        ("joint_velocity", "joint_velocity", slice(None)),
    )
    tolerances = dict(authority["samplewise_tolerances"])
    for tolerance_id, member, columns in checks:
        delta = np.abs(
            np.asarray(first[member])[:, columns]
            - np.asarray(second[member])[:, columns]
        )
        maximum = float(delta.max(initial=0.0))
        maxima[tolerance_id] = maximum
        if maximum > float(tolerances[tolerance_id]):
            return False, maxima
    return True, maxima


def _reset_pair_comparison(
    state_id: str,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    first_termination_reason: str,
    second_termination_reason: str,
    first_stuck: bool,
    second_stuck: bool,
) -> dict[str, Any]:
    """Emit the independently recomputable full reset-pair projection."""

    import numpy as np

    authority = CONTRACT.RESET_TRACE_PAIR_COMPARISON_AUTHORITY
    exact = {
        member: bool(np.array_equal(first[member], second[member]))
        for member in authority["exact_members"]
    }
    delta_pose = np.abs(first["base_pose_world"] - second["base_pose_world"])
    endpoint_delta = first["base_pose_world"][-1, :3] - second["base_pose_world"][-1, :3]
    first_yaw = _pose_yaw_xyzw(first["base_pose_world"][-1])
    second_yaw = _pose_yaw_xyzw(second["base_pose_world"][-1])
    row = {
        "state_id": str(state_id),
        "trial_indices": [0, 1],
        "physics_sample_count": int(len(first["timestamp_s"])),
        "maximum_base_position_error_m": float(delta_pose[:, :3].max(initial=0.0)),
        "maximum_base_quaternion_component_error": float(delta_pose[:, 3:7].max(initial=0.0)),
        "maximum_base_twist_error": float(np.abs(first["base_twist_world"] - second["base_twist_world"]).max(initial=0.0)),
        "maximum_joint_position_error_rad": float(np.abs(first["joint_position"] - second["joint_position"]).max(initial=0.0)),
        "maximum_joint_velocity_error_rad_s": float(np.abs(first["joint_velocity"] - second["joint_velocity"]).max(initial=0.0)),
        "exact_member_equal": exact,
        "endpoint_position_error_m": math.sqrt(math.fsum(float(value) ** 2 for value in endpoint_delta)),
        "endpoint_heading_error_rad": abs(_wrap_angle(first_yaw - second_yaw)),
        "termination_reason_equal": first_termination_reason == second_termination_reason,
        "stuck_equal": bool(first_stuck) is bool(second_stuck),
    }
    tolerance_pairs = (
        ("maximum_base_position_error_m", "reset_base_position_m"),
        ("maximum_base_quaternion_component_error", "reset_base_quaternion_component"),
        ("maximum_base_twist_error", "reset_base_twist"),
        ("maximum_joint_position_error_rad", "reset_joint_position_rad"),
        ("maximum_joint_velocity_error_rad_s", "reset_joint_velocity_rad_s"),
        ("endpoint_position_error_m", "repeat_endpoint_position_m"),
        ("endpoint_heading_error_rad", "repeat_endpoint_heading_rad"),
    )
    row["passed"] = bool(
        all(row[field] <= CONTRACT.NUMERICAL_TOLERANCES[tolerance] for field, tolerance in tolerance_pairs)
        and all(exact.values())
        and row["termination_reason_equal"]
        and row["stuck_equal"]
    )
    return row


def _write_material_shard(directory: Path, metadata: Mapping[str, Any], arrays: Mapping[str, Any]) -> None:
    if directory.exists() or directory.is_symlink():
        raise ExperimentError(f"material shard is not fresh: {directory}")
    directory.mkdir(parents=False, exist_ok=False)
    atomic_npz(directory / "payload.npz", **dict(arrays))
    payload = attach_digest(
        {
            **dict(metadata),
            "payload": binding(
                directory / "payload.npz",
                role="material_shard_payload",
                kind="npz",
                relative_to=MATERIAL_ROOT,
            ),
        }
    )
    atomic_json(directory / "metadata.json", payload)


def _load_material_shard(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    metadata = load_json(directory / "metadata.json")
    payload = dict(metadata["payload"])
    path = directory / "payload.npz"
    info = _require_regular(path)
    assert info is not None
    if (
        payload["path"] != str(path.relative_to(MATERIAL_ROOT))
        or int(payload["bytes"]) != info.st_size
        or payload["sha256"] != sha256_file(path)
    ):
        raise ExperimentError(f"material shard payload binding drift: {directory}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.ascontiguousarray(data[name]) for name in data.files}
    return metadata, arrays


def _qualification_directory(pool_index: int) -> Path:
    return MATERIAL_ROOT / "qualification" / f"pool-{int(pool_index):03d}"


def _qualification_backend_default() -> Any:
    return GenesisGo2PhysicalBackend()


def _point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    try:
        return bool(METRICS.point_in_polygon_inclusive(
            [float(point[0]), float(point[1])],
            [[float(value[0]), float(value[1])] for value in polygon],
            tolerance_m=float(CONTRACT.NUMERICAL_TOLERANCES["se2_position_m"]),
        ))
    except Exception as exc:
        raise ExperimentError("pure polygon-membership authority failed") from exc


def _quat_wxyz_to_rpy(quaternion: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, _wrap_angle(yaw)


def _candidate_requested_commands(candidate_index: int) -> list[list[float]]:
    if not 0 <= int(candidate_index) < CANDIDATE_COUNT:
        raise ExperimentError("candidate index outside frozen bank")
    candidate_id, primitives = CONTRACT.CANDIDATE_BANK[int(candidate_index)]
    if candidate_id != CANDIDATE_IDS[int(candidate_index)]:
        raise ExperimentError("candidate bank/order contract drift")
    commands: list[list[float]] = []
    for primitive in primitives[:3]:
        command = [float(value) for value in CONTRACT.CANDIDATE_PRIMITIVES[primitive]]
        commands.extend([command] * int(CONTRACT.COMMAND_TICKS_PER_BLOCK))
    if len(commands) != 15:
        raise ExperimentError("candidate H3 requested-command length drift")
    return commands


class _GenesisPhysicalSession:
    """One ordinary single-scene CPU Genesis process with reviewed restore."""

    def __init__(self, spec: Mapping[str, Any], *, backend: str) -> None:
        import numpy as np
        from lewm_genesis.go2_adapter import resolve_go2_urdf
        from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits
        from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner
        from lewm_genesis.scene_builder import build_scene_from_pack
        from lewm_genesis.scene_loader import (
            DEFAULT_GO2_FOOT_LINKS_LEWM_ORDER,
            LightingSpec,
            MaterialOverride,
            PhysicsRandomization,
            RobotSpec,
            ScenePack,
            StaticObject,
            VisualRandomization,
            camera_mount_from_platform,
            load_platform_manifest,
            physics_timing_from_platform,
        )
        from scripts.run_go2_oracle_branch_pilot_v1 import BranchContext

        self.spec = copy.deepcopy(dict(spec))
        self.geometry = copy.deepcopy(dict(spec["geometry"]))
        self.backend = str(backend)
        if self.backend != "cpu":
            raise ExperimentError(
                "physical contact qualification requires the reviewed CPU manifold path"
            )
        platform_path = REPO_ROOT / "config/go2_platform_manifest.yaml"
        registry_path = REPO_ROOT / "config/go2_primitive_registry.yaml"
        platform = load_platform_manifest(platform_path)
        spawn = self.geometry["spawn_se2_world"]
        yaw = float(spawn[2])
        static_objects = tuple(
            StaticObject(
                object_id=str(row["wall_id"]),
                kind="wall",
                center_xyz_m=tuple(float(value) for value in row["centre_xyz"]),
                size_xyz_m=tuple(float(value) for value in row["size_xyz"]),
                yaw_rad=float(row["yaw_rad"]),
                material_id=str(row["material_id"]),
            )
            for row in self.geometry["wall_boxes"]
        )
        pack = ScenePack(
            scene_id=str(spec["scene_id"]),
            family=str(spec["family"]),
            split="DEVELOPMENT_UNASSIGNED",
            difficulty_tier="PHYSICAL_GRAPH_EDGE_HANDOFF",
            manifest_sha256=hashlib.sha256(canonical_bytes(self.geometry)[:-1]).hexdigest(),
            physics_seed=int(spec["procedural_seed"]),
            topology_seed=int(spec["procedural_seed"]),
            visual_seed=int(spec["procedural_seed"]),
            world_bounds_xy_m=((-4.0, -4.0), (4.0, 4.0)),
            static_objects=static_objects,
            robot=RobotSpec(
                urdf_path=resolve_go2_urdf(platform, REPO_ROOT),
                spawn_xyz_m=(float(spawn[0]), float(spawn[1]), 0.375),
                spawn_quat_wxyz=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
                foot_links_in_lewm_order=DEFAULT_GO2_FOOT_LINKS_LEWM_ORDER,
            ),
            camera=camera_mount_from_platform(platform),
            timing=physics_timing_from_platform(platform),
            camera_constraints={"min_camera_clearance_m": 0.10, "min_wall_thickness_m": 0.08},
            source_dir=REPO_ROOT,
            visual_randomization=VisualRandomization(
                material_overrides=(
                    MaterialOverride("NEUTRAL_FLOOR", (0.50, 0.50, 0.50, 1.0)),
                    MaterialOverride("NEUTRAL_WALL", (0.35, 0.35, 0.35, 1.0)),
                ),
                lighting=LightingSpec(
                    direction=(0.0, 0.0, -1.0), diffuse_rgb=(0.8, 0.8, 0.8),
                    specular_rgb=(0.2, 0.2, 0.2), ambient_rgb=(0.25, 0.25, 0.25),
                ),
            ),
            physics_randomization=PhysicsRandomization(1.0, 0.0, 1.0, 0.0),
        )
        build = build_scene_from_pack(
            pack, n_envs=1, backend=self.backend, show_viewer=False,
            render_robot=False, apply_textures=False,
        )
        registry = PrimitiveRegistry.from_yaml(registry_path)
        safety = SafetyLimits.from_manifest(platform)
        policy = GenesisGo2PPOPolicy.from_platform_manifest(
            platform, REPO_ROOT, device="cpu"
        )
        runner = RolloutRunner(
            build, policy, registry, safety,
            config=RolloutConfig(
                n_blocks=1, fall_z_threshold_m=0.15, rgb_capture_per_block=False,
                seed=int(spec["procedural_seed"]), log_progress_every_blocks=0,
                foot_contact_source="zero", randomize_spawn_pose=False,
            ),
        )
        self.ctx = BranchContext(
            runner=runner, policy=policy, build=build, pack=pack,
            scene_graph=None, manifest=None, grid=None,
            solver_fields=_collect_solver_fields_compat(build.scene),
        )
        # These are controller inputs/state, not reconstructed report fields.
        # Keep a rolling copy at the same command/policy boundaries used by
        # the production loop so that the frozen ranker can consume the exact
        # persisted predecessor state and exact restore can reinstate it.
        self._command_history = np.zeros((15, 3), dtype=np.float64)
        self._control_history = np.zeros((15, 2), dtype=np.float64)
        self._last_controller_observation = np.zeros((45,), dtype=np.float64)
        self._previous_policy_action = np.zeros((12,), dtype=np.float64)
        self._low_level_policy_state = np.zeros((12,), dtype=np.float64)
        self._contact_topology = self._build_contact_topology()
        self._runtime = {
            "backend": self.backend,
            "n_envs": 1,
            "physics_dt_s": float(pack.timing.physics_dt_s),
            "policy_dt_s": float(pack.timing.policy_dt_s),
            "command_dt_s": float(pack.timing.command_dt_s),
            "policy_evaluation_mode": True,
            "policy_device": "cpu",
            "simulate_action_latency": bool(getattr(policy, "simulate_action_latency", False)),
            "contact_api": "robot.get_contacts(exclude_self_contact=False)",
            "forbidden_net_force_api_used": False,
            "snapshot_restore_source": "scripts/run_go2_oracle_branch_pilot_v1.py",
            "solver_field_collector": (
                "runner-owned exact reviewed walk port for quadrants 0.6.2"
            ),
            "ppo_observation_contact_inputs": "none_in_frozen_45_element_contract",
            "foot_contact_source_zero_scope": (
                "rollout telemetry placeholder only; physical contact labels use "
                "robot.get_contacts at every 2 ms physics step"
            ),
        }

    def _build_contact_topology(self) -> dict[str, set[int]]:
        robot = self.ctx.build.robot
        robot_links = {int(link.idx) for link in robot.links}
        support_links = {
            int(link.idx) for link in robot.links
            if "foot" in str(link.name).lower() or "calf" in str(link.name).lower()
        }
        ground_links: set[int] = set()
        for entity in self.ctx.build.scene.entities:
            if entity is robot:
                continue
            if type(getattr(entity, "morph", None)).__name__ == "Plane":
                ground_links.update(int(link.idx) for link in getattr(entity, "links", ()))
        if not robot_links or not support_links or not ground_links:
            raise ExperimentError("Genesis contact-link topology is incomplete")
        return {"robot": robot_links, "support": support_links, "ground": ground_links}

    def begin_and_settle(self) -> None:
        runner = self.ctx.runner
        _reset_robot_to_fixed_spawn_compat(runner)
        for state in runner.episode_states:
            state.episode_step = 0
        self.ctx.episode_ticks = 0
        self.ctx.ticks_executed = 0
        self.ctx.policy_steps = 0
        self.ctx.episode_start_reset_count = int(runner.episode_states[0].reset_count)
        self.ctx.reset_in_last_block = False
        hold = [[0.0, 0.0, 0.0] for _ in range(15)]
        self.execute_requested_ticks(hold, record=False)

    def _disallowed_contact(self) -> bool:
        import numpy as np
        from lewm.safety.contact_hazard_ontology_v1 import is_disallowed_contact

        contacts = self.ctx.build.robot.get_contacts(exclude_self_contact=False)
        if not contacts:
            return False
        arrays = {
            str(key): np.asarray(self.ctx.runner._as_np(value))
            for key, value in contacts.items()
        }
        link_a = arrays.get("link_a", np.empty(0, dtype=np.int64)).reshape(-1)
        link_b = arrays.get("link_b", np.empty(0, dtype=np.int64)).reshape(-1)
        for index, (a_raw, b_raw) in enumerate(zip(link_a, link_b, strict=True)):
            a, b = int(a_raw), int(b_raw)
            a_robot, b_robot = a in self._contact_topology["robot"], b in self._contact_topology["robot"]
            if a_robot == b_robot:
                continue
            robot_id, environment_id = (a, b) if a_robot else (b, a)
            force_key = "force_a" if a_robot else "force_b"
            force_array = arrays.get(force_key)
            magnitude = None
            if force_array is not None and index < len(force_array):
                magnitude = float(np.linalg.norm(np.asarray(force_array[index], dtype=np.float64)))
            if is_disallowed_contact(
                robot_link_id=robot_id,
                environment_link_id=environment_id,
                foot_link_ids=self._contact_topology["support"],
                ground_link_ids=self._contact_topology["ground"],
                self_contact=False,
                force_magnitude_n=magnitude,
            ):
                return True
        return False

    def _sample(self, requested: Sequence[float], applied: Sequence[float], timestamp_s: float) -> dict[str, Any]:
        import numpy as np

        runner = self.ctx.runner
        robot = self.ctx.build.robot
        pos = np.asarray(runner._as_np(robot.get_pos()), dtype=np.float64).reshape(-1, 3)[0]
        quat = np.asarray(runner._as_np(robot.get_quat()), dtype=np.float64).reshape(-1, 4)[0]
        vel = np.asarray(runner._as_np(robot.get_vel()), dtype=np.float64).reshape(-1, 3)[0]
        ang = np.asarray(runner._as_np(robot.get_ang()), dtype=np.float64).reshape(-1, 3)[0]
        joints = np.asarray(
            runner._as_np(robot.get_dofs_position(runner._leg_dof_idx.tolist())),
            dtype=np.float64,
        ).reshape(-1, 12)[0]
        joint_vel = np.asarray(
            runner._as_np(robot.get_dofs_velocity(runner._leg_dof_idx.tolist())),
            dtype=np.float64,
        ).reshape(-1, 12)[0]
        xy = pos[:2]
        source = self.geometry["source_node"]["boundary_polygon_world"]
        target = self.geometry["target_node"]["boundary_polygon_world"]
        correct = self.geometry["selected_directed_edge"]["edge_region_polygon_world"]
        wrong = [row["edge_region_polygon_world"] for row in self.geometry["competing_directed_edges"]]
        return {
            "timestamp_s": float(timestamp_s),
            "base_pose_world": np.asarray([*pos, quat[1], quat[2], quat[3], quat[0]], dtype=np.float64),
            "base_twist_world": np.asarray([*vel, *ang], dtype=np.float64),
            "joint_position": joints,
            "joint_velocity": joint_vel,
            "requested_command": np.asarray(requested, dtype=np.float64),
            "post_slew_applied_command": np.asarray(applied, dtype=np.float64),
            "applied_command": np.asarray(applied, dtype=np.float64),
            "physics_contact": np.uint8(self._disallowed_contact()),
            "source_region_member": np.uint8(_point_in_polygon(xy, source)),
            "correct_edge_region_member": np.uint8(_point_in_polygon(xy, correct)),
            "edge_region_member": np.uint8(_point_in_polygon(xy, correct)),
            "wrong_edge_region_member": np.uint8(any(_point_in_polygon(xy, polygon) for polygon in wrong)),
            "target_region_member": np.uint8(_point_in_polygon(xy, target)),
        }

    def execute_requested_ticks(
        self, requested_ticks: Sequence[Sequence[float]], *, record: bool = True
    ) -> dict[str, Any] | None:
        import numpy as np

        if len(requested_ticks) % 5 != 0:
            raise ExperimentError("physical command tape is not block aligned")
        runner = self.ctx.runner
        rows: list[dict[str, Any]] = []
        for block_start in range(0, len(requested_ticks), 5):
            requested = np.asarray(requested_ticks[block_start:block_start + 5], dtype=np.float32)[None, ...]
            block = runner._clip_block(requested)
            for tick_offset in range(5):
                command = np.asarray(block.executed[0, tick_offset], dtype=np.float32)
                for _policy_step in range(runner._policy_steps_per_command_tick):
                    observation = runner._build_observation(command[None, :])
                    cached_action = self.ctx.policy._last_actions
                    if cached_action is None:
                        # Exact first-act initialization performed by
                        # GenesisGo2PPOPolicy.act before it builds the 45-D
                        # policy observation.
                        self.ctx.policy._last_actions = np.zeros(
                            (1, 12), dtype=np.float32
                        )
                    previous_policy_action = (
                        np.zeros((12,), dtype=np.float64)
                        if cached_action is None
                        else np.asarray(cached_action, dtype=np.float64)
                        .reshape(1, 12)[0].copy()
                    )
                    controller_observation = np.asarray(
                        self.ctx.policy._build_policy_observation(observation),
                        dtype=np.float64,
                    ).reshape(1, 45)[0].copy()
                    targets = self.ctx.policy.act(observation)
                    self._last_controller_observation = controller_observation
                    self._previous_policy_action = previous_policy_action
                    self._low_level_policy_state = (
                        previous_policy_action.copy()
                        if self.ctx.policy.simulate_action_latency
                        else np.asarray(self.ctx.policy._last_actions, dtype=np.float64)
                        .reshape(1, 12)[0].copy()
                    )
                    runner._apply_joint_targets(targets)
                    base_time = float(runner._sim_time_ns) / 1.0e9
                    for physics_step in range(runner._physics_steps_per_policy):
                        self.ctx.build.scene.step()
                        if record:
                            rows.append(self._sample(
                                requested[0, tick_offset], command,
                                base_time + (physics_step + 1) * TRACE_DT_S,
                            ))
                    runner._sim_time_ns += runner._policy_dt_ns
                for state in runner.episode_states:
                    state.step()
                self._command_history[:-1] = self._command_history[1:]
                self._command_history[-1] = np.asarray(command, dtype=np.float64)
                self._control_history[:-1] = self._control_history[1:]
                self._control_history[-1] = np.asarray(
                    [command[0], command[2]], dtype=np.float64
                )
            runner._last_executed = np.asarray(block.executed[:, -1, :], dtype=np.float32).copy()
            runner._blocks_in_episode += 1
            self.ctx.ticks_executed += 5
            self.ctx.episode_ticks += 5
            self.ctx.policy_steps += 25
            self.ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()
        if not record:
            return None
        members = CANDIDATE_TRACE_MEMBERS
        return {
            member: np.ascontiguousarray(np.stack([row[member] for row in rows], axis=0))
            if member not in {"timestamp_s", "physics_contact", "source_region_member", "correct_edge_region_member", "wrong_edge_region_member", "target_region_member"}
            else np.ascontiguousarray(np.asarray([row[member] for row in rows]))
            for member in members
        }

    def execute_teacher(self) -> dict[str, Any]:
        import numpy as np

        authority = CONTRACT.TEACHER_CONTROLLER_AUTHORITY
        route = self.geometry["teacher_route_polyline_world"]
        rows: list[dict[str, Any]] = []
        runner = self.ctx.runner
        for _tick in range(int(authority["maximum_command_ticks"])):
            pos = np.asarray(runner._as_np(self.ctx.build.robot.get_pos()), dtype=np.float64).reshape(-1, 3)[0]
            quat = np.asarray(runner._as_np(self.ctx.build.robot.get_quat()), dtype=np.float64).reshape(-1, 4)[0]
            yaw = _quat_wxyz_to_rpy(quat)[2]
            lookahead, _clipped, _remaining = _lookahead_from_port(
                [float(pos[0]), float(pos[1]), yaw], route,
                float(authority["lookahead_distance_m"]),
            )
            heading = math.atan2(lookahead[1] - pos[1], lookahead[0] - pos[0])
            error = _wrap_angle(heading - yaw)
            yaw_rate = max(-float(authority["maximum_absolute_yaw_rate_rad_s"]), min(
                float(authority["maximum_absolute_yaw_rate_rad_s"]),
                float(authority["yaw_gain"]) * error,
            ))
            if abs(error) >= float(authority["turn_in_place_heading_error_rad"]):
                vx = 0.0
            else:
                vx = max(
                    float(authority["minimum_forward_command_mps"]),
                    min(float(authority["maximum_forward_command_mps"]), float(authority["linear_gain"]) * math.cos(error)),
                )
            requested = np.asarray([[vx, 0.0, yaw_rate]] * 5, dtype=np.float32)[None, ...]
            clipped = runner._clip_block(requested)
            command = np.asarray(clipped.executed[0, 0], dtype=np.float32)
            for _policy_step in range(runner._policy_steps_per_command_tick):
                observation = runner._build_observation(command[None, :])
                targets = self.ctx.policy.act(observation)
                runner._apply_joint_targets(targets)
                base_time = float(runner._sim_time_ns) / 1.0e9
                for physics_step in range(runner._physics_steps_per_policy):
                    self.ctx.build.scene.step()
                    rows.append(self._sample(
                        [vx, 0.0, yaw_rate], command,
                        base_time + (physics_step + 1) * TRACE_DT_S,
                    ))
                runner._sim_time_ns += runner._policy_dt_ns
            runner._last_executed = command[None, :].copy()
            if len(rows) >= 2:
                partial = {
                    member: np.ascontiguousarray(np.asarray([row[member] for row in rows]))
                    for member in TEACHER_TRACE_MEMBERS
                }
                try:
                    canonical_port_crossing(
                        partial["base_pose_world"], partial["target_region_member"],
                        self.geometry["selected_directed_edge"]["opening_segment_world"],
                        self.geometry["selected_directed_edge"]["opening_normal_world"],
                        self.geometry["competing_directed_edges"],
                    )
                except ExperimentError:
                    pass
                else:
                    if not partial["physics_contact"].any():
                        return partial
        return {
            member: np.ascontiguousarray(np.asarray([row[member] for row in rows]))
            for member in TEACHER_TRACE_MEMBERS
        }

    def render_rgb_and_transform(self) -> tuple[Any, Any]:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from lewm_genesis.camera_safety import camera_safety_config_from_pack, safe_camera_pose_from_base
        from lewm_genesis.scene_loader import effective_camera_mount_xyz_rpy

        runner = self.ctx.runner
        pos = np.asarray(runner._as_np(self.ctx.build.robot.get_pos()), dtype=np.float64).reshape(-1, 3)[0]
        quat_wxyz = np.asarray(runner._as_np(self.ctx.build.robot.get_quat()), dtype=np.float64).reshape(-1, 4)[0]
        quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        mount_xyz, mount_rpy = effective_camera_mount_xyz_rpy(self.ctx.pack)
        pose, _safety = safe_camera_pose_from_base(
            pos, quat_xyzw, mount_xyz_body=mount_xyz, mount_rpy_body=mount_rpy,
            objects=self.ctx.pack.static_objects,
            config=camera_safety_config_from_pack(self.ctx.pack),
        )
        self.ctx.build.camera.set_pose(pos=pose.position, lookat=pose.lookat, up=pose.up)
        native = runner._extract_rgb(self.ctx.build.camera.render())
        if native is None:
            raise ExperimentError("Genesis camera returned no RGB")
        native = np.asarray(native)
        if native.ndim == 4:
            native = native[0]
        native = native[..., :3]
        if native.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(native)) <= 1.0 else 1.0
            native = np.rint(np.clip(native * scale, 0.0, 255.0)).astype(np.uint8)
        if native.shape != (480, 640, 3):
            raise ExperimentError(f"native Genesis RGB shape drift: {native.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(native)).permute(2, 0, 1)[None].float()
        resized = F.interpolate(tensor, size=(168, 224), mode="area")[0].permute(1, 2, 0)
        rgb = resized.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        forward = np.asarray(pose.lookat, dtype=np.float64) - np.asarray(pose.position, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        up = np.asarray(pose.up, dtype=np.float64); up /= np.linalg.norm(up)
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.stack([forward, right, up], axis=1)
        transform[:3, 3] = np.asarray(pose.position, dtype=np.float64)
        return np.ascontiguousarray(rgb), transform

    def capture_snapshot(self) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
        import pickle
        import numpy as np
        import torch
        from scripts import run_go2_oracle_branch_pilot_v1 as PILOT

        snapshot = PILOT.capture_branch_state(
            self.ctx, goal={"directed_edge_id": "selected-edge"},
            identity={"experiment_id": EXPERIMENT_ID, "state_id": self.spec["state_id"]},
        )
        device_states = [state.clone().cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        snapshot.rng["torch_devices"] = device_states
        snapshot.rng["torch_device_count"] = int(torch.cuda.device_count())
        snapshot.harness["pgehq_controller_arrays"] = {
            "command_history": self._command_history.copy(),
            "control_history": self._control_history.copy(),
            "controller_observation": self._last_controller_observation.copy(),
            "previous_policy_action": self._previous_policy_action.copy(),
            "low_level_policy_state": self._low_level_policy_state.copy(),
        }
        payload = pickle.dumps(snapshot, protocol=4)
        runner, robot = self.ctx.runner, self.ctx.build.robot
        pos = np.asarray(runner._as_np(robot.get_pos()), dtype=np.float64).reshape(-1, 3)[0]
        quat = np.asarray(runner._as_np(robot.get_quat()), dtype=np.float64).reshape(-1, 4)[0]
        vel = np.asarray(runner._as_np(robot.get_vel()), dtype=np.float64).reshape(-1, 3)[0]
        ang = np.asarray(runner._as_np(robot.get_ang()), dtype=np.float64).reshape(-1, 3)[0]
        joint = np.asarray(runner._as_np(robot.get_dofs_position(runner._leg_dof_idx.tolist())), dtype=np.float64).reshape(-1, 12)[0]
        joint_vel = np.asarray(runner._as_np(robot.get_dofs_velocity(runner._leg_dof_idx.tolist())), dtype=np.float64).reshape(-1, 12)[0]
        rgb, camera_transform = self.render_rgb_and_transform()
        last_action = np.asarray(self.ctx.policy._last_actions, dtype=np.float64).reshape(1, 12)[0]
        controller_observation = self._last_controller_observation.copy()
        previous_policy_action = self._previous_policy_action.copy()
        command_history = self._command_history.copy()
        control_history = self._control_history.copy()
        low_level_policy_state = self._low_level_policy_state.copy()
        rng_projection = {
            key: hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()
            for key, value in sorted(snapshot.rng.items())
        }
        metadata = {
            "serialized_solver_state_sha256": PILOT._solver_state_digest(snapshot.solver_state, snapshot.step_index),
            "serialized_controller_state_sha256": hashlib.sha256(pickle.dumps(snapshot.harness, protocol=4)).hexdigest(),
            "serialized_rng_state_sha256": hashlib.sha256(pickle.dumps(snapshot.rng, protocol=4)).hexdigest(),
            "policy_last_action_sha256": canonical_array_sha256(last_action),
            "capture_timestamp_s": float(runner._sim_time_ns) / 1.0e9,
            "controller_observation_sha256": canonical_array_sha256(controller_observation),
            "previous_policy_action_sha256": canonical_array_sha256(previous_policy_action),
            "torch_cpu_rng_state_sha256": canonical_array_sha256(snapshot.rng["torch"].numpy()),
            "torch_device_rng_state_sha256s": [canonical_array_sha256(value.numpy()) for value in device_states],
            "torch_device_count": len(device_states),
            "previous_applied_command_sha256": canonical_array_sha256(np.asarray(runner._last_executed)[0]),
            "command_history_sha256": canonical_array_sha256(command_history),
            "control_history_sha256": canonical_array_sha256(control_history),
            "low_level_policy_state_sha256": canonical_array_sha256(low_level_policy_state),
            "solver_field_inventory": sorted(snapshot.solver_state),
            "controller_field_inventory": sorted([
                *snapshot.harness["arrays"], *snapshot.harness["objects"],
                "policy._last_actions",
                "pgehq_controller_arrays.controller_observation_before_final_policy_act",
                "pgehq_controller_arrays.previous_policy_action_before_final_policy_act",
                "pgehq_controller_arrays.low_level_policy_state_applied_under_latency",
                "pgehq_controller_arrays.command_history_15x3_post_slew",
                "pgehq_controller_arrays.control_history_15x2_vx_yaw",
            ]),
            "rng_field_inventory": sorted(snapshot.rng),
        }
        arrays = {
            "payload_bytes": payload,
            "base_pose_world": np.asarray([*pos, quat[1], quat[2], quat[3], quat[0]], dtype=np.float64),
            "base_twist_world": np.asarray([*vel, *ang], dtype=np.float64),
            "joint_position": joint,
            "joint_velocity": joint_vel,
            "controller_observation": controller_observation,
            "policy_last_action": last_action,
            "previous_policy_action": previous_policy_action,
            "previous_applied_command": np.asarray(runner._last_executed, dtype=np.float64)[0],
            "command_history": command_history,
            "control_history": control_history,
            "low_level_policy_state": low_level_policy_state,
            "camera_world_transform": camera_transform,
        }
        return payload, {**arrays, **metadata}, {"rgb": rgb}

    def restore_snapshot(self, payload: bytes) -> Any:
        import pickle
        import numpy as np
        import torch
        from scripts import run_go2_oracle_branch_pilot_v1 as PILOT

        snapshot = pickle.loads(payload)
        PILOT.restore_branch_state(self.ctx, snapshot)
        device_states = snapshot.rng.get("torch_devices")
        expected_count = int(snapshot.rng.get("torch_device_count", -1))
        if not isinstance(device_states, list) or expected_count != int(torch.cuda.device_count()) or len(device_states) != expected_count:
            raise ExperimentError("serialized snapshot torch-device RNG inventory drift")
        if device_states:
            torch.cuda.set_rng_state_all(device_states)
        controller_arrays = snapshot.harness.get("pgehq_controller_arrays")
        if not isinstance(controller_arrays, Mapping) or set(controller_arrays) != {
            "command_history", "control_history", "controller_observation",
            "previous_policy_action", "low_level_policy_state",
        }:
            raise ExperimentError("serialized snapshot controller-array inventory drift")
        self._command_history = np.asarray(
            controller_arrays["command_history"], dtype=np.float64
        ).reshape(15, 3).copy()
        self._control_history = np.asarray(
            controller_arrays["control_history"], dtype=np.float64
        ).reshape(15, 2).copy()
        self._last_controller_observation = np.asarray(
            controller_arrays["controller_observation"], dtype=np.float64
        ).reshape(45).copy()
        self._previous_policy_action = np.asarray(
            controller_arrays["previous_policy_action"], dtype=np.float64
        ).reshape(12).copy()
        self._low_level_policy_state = np.asarray(
            controller_arrays["low_level_policy_state"], dtype=np.float64
        ).reshape(12).copy()
        return snapshot

    def graph(self) -> dict[str, Any]:
        geometry = self.geometry
        nodes = [
            {"node_id": "source", "node_kind": "SOURCE", "centre_world": geometry["source_node"]["centre_world"], "boundary_polygon_world": geometry["source_node"]["boundary_polygon_world"]},
            {"node_id": "target", "node_kind": "TARGET", "centre_world": geometry["target_node"]["centre_world"], "boundary_polygon_world": geometry["target_node"]["boundary_polygon_world"]},
        ]
        edges: list[dict[str, Any]] = []
        selected = geometry["selected_directed_edge"]
        route = geometry["teacher_route_polyline_world"]
        edges.append({
            "edge_id": "selected-edge", "source_node_id": "source", "target_node_id": "target",
            "port_label": str(selected["route_direction"]), "route_polyline_world": route,
            "edge_length_m": _polyline_length(route), "oracle_reachable": True,
            "physically_executable": True,
            "opening_segment_world": copy.deepcopy(selected["opening_segment_world"]),
            "opening_normal_world": copy.deepcopy(selected["opening_normal_world"]),
            "edge_kind": "SELECTED",
        })
        for index, edge in enumerate(geometry["competing_directed_edges"]):
            target_id = f"competing-node-{index}"
            polygon = edge["edge_region_polygon_world"]
            centre = [sum(float(point[axis]) for point in polygon) / len(polygon) for axis in (0, 1)]
            nodes.append({"node_id": target_id, "node_kind": "COMPETING", "centre_world": centre, "boundary_polygon_world": polygon})
            midpoint = [sum(float(point[axis]) for point in edge["opening_segment_world"]) / 2.0 for axis in (0, 1)]
            competitor_route = [geometry["source_node"]["centre_world"], midpoint, centre]
            edges.append({
                "edge_id": str(edge["edge_id"]), "source_node_id": "source", "target_node_id": target_id,
                "port_label": str(edge.get("boundary_side", edge["edge_id"])),
                "route_polyline_world": competitor_route, "edge_length_m": _polyline_length(competitor_route),
                "oracle_reachable": True, "physically_executable": True,
                "opening_segment_world": copy.deepcopy(edge["opening_segment_world"]),
                "opening_normal_world": copy.deepcopy(edge["opening_normal_world"]),
                "edge_kind": "COMPETING",
            })
        return {
            "graph_id": str(self.spec["graph_id"]), "source_node_id": "source",
            "target_node_id": "target", "directed_edge_id": "selected-edge",
            "nodes": nodes, "edges": edges, "goal_reachable": True,
            "graph_edge_physically_executable": True,
            "teacher_positive_route_progress": False,
            "teacher_competing_port_entered": False,
        }


class GenesisGo2PhysicalBackend:
    """Adapter boundary for the real Genesis collector.

    The pure contract owns geometry construction.  A missing geometry factory
    is a hard pre-simulation error; this class never fabricates a port, teacher
    route, collision region, or analytic candidate trajectory.
    """

    def __init__(self, *, backend: str = "cpu") -> None:
        self.backend = str(backend)
        builder = getattr(CONTRACT, "build_candidate_specs", None)
        if not callable(builder):
            raise ExperimentError("contract-owned embedded physical geometry is absent")
        self._spec_sha_by_id = {
            str(row["candidate_spec_id"]): str(row["canonical_spec_sha256"])
            for row in builder()
        }

    def qualify(self, candidate_spec: Mapping[str, Any]) -> Mapping[str, Any]:
        spec = copy.deepcopy(dict(candidate_spec))
        if self._spec_sha_by_id.get(str(spec["candidate_spec_id"])) != str(spec["canonical_spec_sha256"]):
            raise ExperimentError("physical candidate spec is outside frozen authority")
        session = _GenesisPhysicalSession(spec, backend=self.backend)
        session.begin_and_settle()
        payload, snapshot, auxiliary = session.capture_snapshot()
        session.restore_snapshot(payload)
        teacher = session.execute_teacher()
        graph = session.graph()
        graph["teacher_positive_route_progress"] = _teacher_route_progress_m(
            teacher["base_pose_world"], spec["geometry"]["selected_directed_edge"]["opening_segment_world"]
        ) > 0.0
        graph["teacher_competing_port_entered"] = _first_competing_crossing(
            teacher["base_pose_world"], spec["geometry"]["competing_directed_edges"]
        ) is not None
        snapshot_payload_sha = hashlib.sha256(payload).hexdigest()
        return {
            "candidate_spec_id": spec["candidate_spec_id"],
            "initial_decision_state_sha256": snapshot_payload_sha,
            "graph": graph,
            "teacher_trace": teacher,
            "rgb": auxiliary["rgb"],
            "snapshot": snapshot,
            "contact_instrumentation": {
                "api": "robot.get_contacts", "sample_period_s": TRACE_DT_S,
                "forbidden_net_force_api_used": False,
                "ontology_sha256": CONTRACT.CONTACT_AUTHORITY["ontology_sha256"],
            },
            "runtime_evidence": {
                **session._runtime,
                "snapshot_captured_before_teacher": True,
                "teacher_restored_from_serialized_snapshot": True,
                "teacher_snapshot_sha256": snapshot_payload_sha,
            },
        }

    def fanout(self, state: Mapping[str, Any], snapshot_payload: bytes) -> Sequence[Mapping[str, Any]]:
        session = _GenesisPhysicalSession(state, backend=self.backend)
        rows = []
        for candidate_index in range(CANDIDATE_COUNT):
            session.restore_snapshot(snapshot_payload)
            trace = session.execute_requested_ticks(_candidate_requested_commands(candidate_index))
            rows.append({
                "candidate_index": candidate_index,
                "snapshot_payload_sha256": hashlib.sha256(snapshot_payload).hexdigest(),
                "trace": trace,
                "runtime_evidence": dict(session._runtime),
            })
        return rows

    def reset_fixture(
        self, candidate_spec: Mapping[str, Any], snapshot_payload: bytes
    ) -> Mapping[str, Any]:
        import numpy as np

        session = _GenesisPhysicalSession(candidate_spec, backend=self.backend)
        trials = []
        snapshot_sha = hashlib.sha256(snapshot_payload).hexdigest()
        commands = [list(CONTRACT.RESET_FIXTURE_COMMAND)] * int(CONTRACT.RESET_FIXTURE_COMMAND_TICKS)
        for trial_index in range(2):
            snapshot = session.restore_snapshot(snapshot_payload)
            rgb, _transform = session.render_rgb_and_transform()
            restored_state_sha = hashlib.sha256(
                canonical_bytes({
                    "solver": snapshot.digest,
                    "last_action": canonical_array_sha256(session.ctx.policy._last_actions),
                    "last_command": canonical_array_sha256(session.ctx.runner._last_executed),
                })[:-1]
            ).hexdigest()
            trace = session.execute_requested_ticks(commands)
            final_pose = trace["base_pose_world"][-1]
            activity = max(max(abs(value) for value in command) for command in commands)
            start_yaw = _pose_yaw_xyzw(trace["base_pose_world"][0])
            end_yaw = _pose_yaw_xyzw(final_pose)
            stuck = bool(
                activity > CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["command_activity_threshold"]
                and np.linalg.norm(final_pose[:2] - trace["base_pose_world"][0, :2])
                < CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["h3_translation_threshold_m"]
                and abs(_wrap_angle(end_yaw - start_yaw))
                < CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["h3_heading_threshold_rad"]
            )
            trials.append({
                "metadata": {
                    "trial_index": trial_index,
                    "serialized_restore_used": True,
                    "clone_equivalence_used": False,
                    "restored_snapshot_sha256": snapshot_sha,
                    "post_restore_state_sha256": restored_state_sha,
                    "current_rgb_sha256": canonical_array_sha256(rgb),
                    "base_pose_world": [float(value) for value in final_pose],
                    "joint_position_sha256": canonical_array_sha256(trace["joint_position"][-1]),
                    "joint_velocity_sha256": canonical_array_sha256(trace["joint_velocity"][-1]),
                    "controller_state_sha256": canonical_array_sha256(session.ctx.policy._last_actions),
                    "rng_state_sha256": hashlib.sha256(__import__("pickle").dumps(snapshot.rng, protocol=4)).hexdigest(),
                    "requested_command_sequence_sha256": canonical_array_sha256(trace["requested_command"]),
                    "post_slew_applied_command_sequence_sha256": canonical_array_sha256(trace["post_slew_applied_command"]),
                    "contact_sequence_sha256": canonical_array_sha256(trace["physics_contact"]),
                    "termination_reason": "H3_COMPLETE",
                    "stuck": stuck,
                },
                "trace": trace,
            })
        return {
            "candidate_spec_id": candidate_spec["candidate_spec_id"],
            "snapshot_payload_sha256": snapshot_sha,
            "reset_trials": trials,
            "runtime_evidence": dict(session._runtime),
        }

    def repeat(
        self, state: Mapping[str, Any], snapshot_payload: bytes, candidate_indices: Sequence[int]
    ) -> Sequence[Mapping[str, Any]]:
        session = _GenesisPhysicalSession(state, backend=self.backend)
        rows = []
        for selector_index, candidate_index in enumerate(candidate_indices):
            session.restore_snapshot(snapshot_payload)
            rows.append({
                "selector_index": selector_index,
                "candidate_index": int(candidate_index),
                "snapshot_payload_sha256": hashlib.sha256(snapshot_payload).hexdigest(),
                "trace": session.execute_requested_ticks(_candidate_requested_commands(int(candidate_index))),
                "runtime_evidence": dict(session._runtime),
            })
        return rows


def _normalise_snapshot(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the selected-decision snapshot and separate bytes/arrays/metadata."""

    import numpy as np

    expected = {"payload_bytes", *SNAPSHOT_NUMERIC_FIELDS, *SNAPSHOT_METADATA_FIELDS}
    if set(value) != expected:
        raise ExperimentError("serialized snapshot field drift")
    payload = value["payload_bytes"]
    if not isinstance(payload, bytes) or not payload:
        raise ExperimentError("serialized snapshot payload is empty")
    arrays: dict[str, Any] = {
        "snapshot_payload_bytes": np.frombuffer(payload, dtype=np.uint8).copy()
    }
    for field, shape in SNAPSHOT_NUMERIC_FIELDS.items():
        array = np.ascontiguousarray(np.asarray(value[field], dtype=np.float64))
        if array.shape != shape or not np.isfinite(array).all():
            raise ExperimentError(f"serialized snapshot {field} shape/value drift")
        arrays[f"snapshot__{field}"] = array
    metadata = {field: copy.deepcopy(value[field]) for field in SNAPSHOT_METADATA_FIELDS}
    for field in (
        "solver_field_inventory", "controller_field_inventory", "rng_field_inventory"
    ):
        if not isinstance(metadata[field], list) or not metadata[field]:
            raise ExperimentError(f"serialized snapshot {field} is incomplete")
    device_digests = metadata["torch_device_rng_state_sha256s"]
    if not isinstance(device_digests, list) or len(device_digests) != int(
        metadata["torch_device_count"]
    ):
        raise ExperimentError("serialized snapshot omits a torch device RNG state")
    metadata["snapshot_payload_sha256"] = hashlib.sha256(payload).hexdigest()
    return arrays, metadata


def qualify_pool_state_stage(
    pool_index: int,
    *,
    backend: Any | None = None,
    fake_runtime: bool = False,
) -> dict[str, Any]:
    """Collect one teacher-only pool trace; no candidate/reset outcome is opened."""

    runtime = require_stage_runtime("physical", fake=fake_runtime)
    _runtime_contract_doc, pool = _require_initialized()
    specs = [dict(value) for value in pool["specs"]]
    if not 0 <= int(pool_index) < len(specs):
        raise ExperimentError("pool index is outside the frozen prospective population")
    if any((MATERIAL_ROOT / "fanout").iterdir()):
        raise ExperimentError("teacher-only qualification cannot run after fanout")
    spec = specs[int(pool_index)]
    collector = _qualification_backend_default() if backend is None else backend
    raw = collector.qualify(copy.deepcopy(spec))
    required = {
        "candidate_spec_id", "initial_decision_state_sha256", "graph",
        "teacher_trace", "rgb", "snapshot", "contact_instrumentation",
        "runtime_evidence",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ExperimentError("teacher-only backend result field drift")
    if raw["candidate_spec_id"] != spec["candidate_spec_id"]:
        raise ExperimentError("teacher-only candidate identity drift")
    import numpy as np

    teacher = _normalise_trace(dict(raw["teacher_trace"]), kind="teacher", exact_samples=None)
    snapshot_arrays, snapshot_metadata = _normalise_snapshot(dict(raw["snapshot"]))
    if raw["initial_decision_state_sha256"] != snapshot_metadata["snapshot_payload_sha256"]:
        raise ExperimentError("teacher initial state is not the captured serialized snapshot")
    rgb = np.ascontiguousarray(np.asarray(raw["rgb"], dtype=np.uint8))
    if rgb.shape != (168, 224, 3):
        raise ExperimentError("teacher decision RGB shape drift")
    graph = dict(raw["graph"])
    frozen_geometry = dict(spec["geometry"])
    selected_edge = dict(frozen_geometry["selected_directed_edge"])
    competing_edges = [dict(value) for value in frozen_geometry["competing_directed_edges"]]
    reasons: list[str] = []
    try:
        crossing = canonical_port_crossing(
            teacher["base_pose_world"], teacher["target_region_member"],
            selected_edge["opening_segment_world"], selected_edge["opening_normal_world"],
            competing_edges,
        )
    except ExperimentError as exc:
        crossing = None
        reasons.append(f"PORT_CROSSING_INVALID:{exc}")
    contact_free = not bool(teacher["physics_contact"].any())
    left_source = bool((teacher["source_region_member"] == 0).any())
    competing_crossing = _first_competing_crossing(
        teacher["base_pose_world"], competing_edges
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
    route_progress_m = _teacher_route_progress_m(
        teacher["base_pose_world"], selected_edge["opening_segment_world"]
    )
    positive = route_progress_m > 0.0
    goal_reachable = bool(graph.get("goal_reachable", False))
    physically_executable = bool(graph.get("graph_edge_physically_executable", False))
    if not contact_free: reasons.append("TEACHER_PHYSICS_CONTACT")
    if not left_source: reasons.append("TEACHER_DID_NOT_LEAVE_SOURCE")
    if not positive: reasons.append("NO_POSITIVE_ROUTE_PROGRESS")
    if competing: reasons.append("COMPETING_PORT_ENTERED")
    if not goal_reachable: reasons.append("GOAL_NOT_REACHABLE")
    if not physically_executable: reasons.append("EDGE_NOT_PHYSICALLY_EXECUTABLE")
    contact = dict(raw["contact_instrumentation"])
    if (
        contact.get("api") != "robot.get_contacts"
        or contact.get("sample_period_s") != TRACE_DT_S
        or contact.get("forbidden_net_force_api_used") is not False
        or contact.get("ontology_sha256") != CONTRACT.CONTACT_AUTHORITY["ontology_sha256"]
    ):
        raise ExperimentError("teacher contact instrumentation drift")
    backend_runtime = dict(raw["runtime_evidence"])
    runtime = _bind_physical_backend_runtime(runtime, backend_runtime)
    if (
        backend_runtime.get("snapshot_captured_before_teacher") is not True
        or backend_runtime.get("teacher_restored_from_serialized_snapshot") is not True
        or backend_runtime.get("teacher_snapshot_sha256")
        != snapshot_metadata["snapshot_payload_sha256"]
    ):
        raise ExperimentError("teacher did not execute from the bound serialized snapshot")
    teacher_valid = bool(
        crossing is not None and contact_free and left_source and positive
        and not competing and physically_executable
    )
    qualified = bool(teacher_valid and goal_reachable)
    arrays = {
        "rgb": rgb,
        **snapshot_arrays,
        **{f"teacher__{key}": value for key, value in teacher.items()},
    }
    metadata = {
        "schema": "physical_graph_edge_handoff_qualification_v1.teacher_pool_shard.v1",
        "experiment_id": EXPERIMENT_ID,
        "pool_index": int(pool_index),
        "candidate_spec": spec,
        "candidate_spec_sha256": str(spec["canonical_spec_sha256"]),
        "initial_decision_state_sha256": str(raw["initial_decision_state_sha256"]),
        "snapshot": snapshot_metadata,
        "graph": graph,
        "teacher": {
            "sample_count": int(len(teacher["timestamp_s"])),
            "trace_digests": _trace_digest_projection(teacher),
            "contact_free": contact_free,
            "left_source_region": left_source,
            "positive_route_progress": positive,
            "route_progress_m": route_progress_m,
            "competing_port_entered": competing,
            "competing_crossing": competing_crossing,
            "reached_target_node": bool(teacher["target_region_member"].any()),
            "crossing": crossing,
            "teacher_valid": teacher_valid,
        },
        "current_rgb_sha256": canonical_array_sha256(rgb),
        "goal_reachable": goal_reachable,
        "graph_edge_physically_executable": physically_executable,
        "qualified": qualified,
        "rejection_reason": "QUALIFIED" if qualified else "|".join(reasons),
        "contact_instrumentation": contact,
        "stage_runtime": runtime,
        "backend_runtime": backend_runtime,
        "reset_or_candidate_outcome_opened": False,
    }
    _write_material_shard(_qualification_directory(pool_index), metadata, arrays)
    return metadata


def _file_binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    info = _require_regular(path)
    assert info is not None
    root = OUTPUT_ROOT if relative_to is None else relative_to
    return {
        "path": str(path.relative_to(root)),
        "bytes": int(info.st_size),
        "sha256": sha256_file(path),
    }


def _slice_binding(
    path: Path, member: str, start: int, stop: int, array_slice: Any
) -> dict[str, Any]:
    return {
        "file_path": str(path.relative_to(OUTPUT_ROOT)),
        "file_sha256": sha256_file(path),
        "member": str(member),
        "start": int(start),
        "stop": int(stop),
        "slice_sha256": canonical_array_sha256(array_slice),
    }


def _concat_traces(
    traces: Sequence[Mapping[str, Any]], members: Sequence[str]
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    import numpy as np

    offsets = [0]
    result: dict[str, list[Any]] = {member: [] for member in members}
    spans: list[tuple[int, int]] = []
    for trace in traces:
        start = offsets[-1]
        length = int(len(trace["timestamp_s"]))
        stop = start + length
        spans.append((start, stop))
        offsets.append(stop)
        for member in members:
            result[member].append(np.asarray(trace[member]))
    arrays = {
        member: np.ascontiguousarray(np.concatenate(parts, axis=0))
        for member, parts in result.items()
    }
    arrays["trace_offsets"] = np.asarray(offsets, dtype=np.int64)
    return arrays, spans


def _teacher_record(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    trace_index: int,
    span: tuple[int, int],
    selected: bool,
    teacher_file: Path,
) -> dict[str, Any]:
    spec = dict(metadata["candidate_spec"])
    teacher = dict(metadata["teacher"])
    crossing = teacher.get("crossing")
    crossed = isinstance(crossing, Mapping)
    graph = dict(metadata["graph"])
    start, stop = span
    trace = {member: arrays[f"teacher__{member}"] for member in TEACHER_TRACE_MEMBERS}
    opening = spec["geometry"]["selected_directed_edge"]["opening_segment_world"]
    opening_array = __import__("numpy").asarray(opening, dtype=float)
    width = float(__import__("numpy").linalg.norm(opening_array[1] - opening_array[0]))
    lateral_fraction = (
        float(crossing["lateral_coordinate_m"]) / width + 0.5
        if crossed else None
    )
    crossing_velocity = None
    if crossed:
        before = int(crossing["sample_before"])
        after = int(crossing["sample_after"])
        alpha = float(crossing["fraction"])
        velocity = (
            (1.0 - alpha) * trace["base_twist_world"][before, :2]
            + alpha * trace["base_twist_world"][after, :2]
        )
        crossing_velocity = [float(value) for value in velocity]
    crossing_velocity_heading = (
        math.atan2(crossing_velocity[1], crossing_velocity[0])
        if crossed and math.hypot(*crossing_velocity) > 0.0 else None
    )
    import numpy as np

    source = np.asarray(trace["source_region_member"], dtype=np.uint8)
    source_exit_indices = np.flatnonzero((source[:-1] != 0) & (source[1:] == 0)) + 1
    target = np.asarray(trace["target_region_member"], dtype=np.uint8)
    target_indices = np.flatnonzero(target != 0)
    endpoint_pose = np.asarray(trace["base_pose_world"][-1], dtype=np.float64)
    midpoint = opening_array.mean(axis=0)
    tangent = opening_array[1] - opening_array[0]
    tangent /= np.linalg.norm(tangent)
    normal = np.asarray(
        spec["geometry"]["selected_directed_edge"]["opening_normal_world"],
        dtype=np.float64,
    )
    normal /= np.linalg.norm(normal)
    endpoint_lateral = abs(float((endpoint_pose[:2] - midpoint) @ tangent))
    endpoint_angular = abs(_wrap_angle(
        _pose_yaw_xyzw(endpoint_pose) - math.atan2(float(normal[1]), float(normal[0]))
    ))
    final_quat = endpoint_pose[3:7]
    roll, pitch, _yaw = _quat_wxyz_to_rpy(
        [final_quat[3], final_quat[0], final_quat[1], final_quat[2]]
    )
    finite = bool(all(np.isfinite(value).all() for value in trace.values()))
    successor = bool(
        finite
        and float(endpoint_pose[2]) >= float(
            CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["minimum_base_height_m"]
        )
        and abs(roll) <= float(
            CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["maximum_absolute_roll_rad"]
        )
        and abs(pitch) <= float(
            CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["maximum_absolute_pitch_rad"]
        )
        and not bool(trace["physics_contact"][-1])
    )
    start_pose = np.asarray(trace["base_pose_world"][0], dtype=np.float64)
    body_endpoint = body_from_world(
        [float(endpoint_pose[0]), float(endpoint_pose[1]), _pose_yaw_xyzw(endpoint_pose)],
        [float(start_pose[0]), float(start_pose[1]), _pose_yaw_xyzw(start_pose)],
    )
    activity = float(np.max(np.abs(np.asarray(trace["requested_command"], dtype=np.float64))))
    stuck = bool(
        activity > float(CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["command_activity_threshold"])
        and math.hypot(body_endpoint[0], body_endpoint[1])
        < float(CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["h3_translation_threshold_m"])
        and abs(body_endpoint[2])
        < float(CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["h3_heading_threshold_rad"])
    )
    target_early = bool(
        crossed and len(target_indices)
        and int(target_indices[0]) - int(crossing["sample_after"]) + 1 < 100
    )
    return {
        "candidate_spec_id": str(spec["candidate_spec_id"]),
        "state_id": str(spec["state_id"]),
        "family": str(spec["family"]),
        "stratum_index": int(spec["stratum_index"]),
        "variant_index": int(spec["variant_index"]),
        "canonical_spec_sha256": str(spec["canonical_spec_sha256"]),
        "teacher_trace_id": f"{spec['candidate_spec_id']}:teacher",
        "trace_index": int(trace_index),
        "selected": bool(selected),
        "initial_decision_state_sha256": str(metadata["initial_decision_state_sha256"]),
        "current_rgb_sha256": str(metadata["current_rgb_sha256"]),
        "current_rgb_valid": True,
        "goal_reachable": bool(metadata["goal_reachable"]),
        "directed_port_defined": teacher.get("crossing") is not None,
        "graph_edge_physically_executable": bool(metadata["graph_edge_physically_executable"]),
        "source_node_id": str(graph["source_node_id"]),
        "target_node_id": str(graph["target_node_id"]),
        "directed_edge_id": str(graph["directed_edge_id"]),
        "trace_slice": _slice_binding(
            teacher_file, "timestamp_s", start, stop, trace["timestamp_s"]
        ),
        "trace_array_slice_sha256s": _trace_digest_projection(trace),
        "sample_count": int(stop - start),
        "contact_free": bool(teacher["contact_free"]),
        "left_source_region": bool(teacher["left_source_region"]),
        "first_source_exit_sample_index": (
            int(source_exit_indices[0]) if len(source_exit_indices) else None
        ),
        "first_crossing_sample_index": int(crossing["sample_after"]) if crossed else None,
        "crossing_segment_fraction": float(crossing["fraction"]) if crossed else None,
        "crossed_directed_port": crossed,
        "positive_route_progress": bool(teacher["positive_route_progress"]),
        "route_progress_m": float(teacher["route_progress_m"]),
        "crossing_directed_normal_dot": (
            float(crossing["normal_dot_displacement_m"]) if crossed else None
        ),
        "crossing_lateral_fraction": lateral_fraction,
        "crossing_velocity_world_xy": crossing_velocity,
        "crossing_velocity_heading_world_rad": crossing_velocity_heading,
        "beyond_port_consecutive_physics_samples": (
            int(crossing["sustained_beyond_samples"]) if crossed else 0
        ),
        "target_entered_before_dwell_complete": target_early,
        "competing_port_entered": bool(teacher["competing_port_entered"]),
        "reached_target_node": bool(teacher["reached_target_node"]),
        "first_target_entry_sample_index": (
            int(target_indices[0]) if len(target_indices) else None
        ),
        "where_reached": "TARGET_NODE" if teacher["reached_target_node"] else "BEYOND_DIRECTED_PORT",
        "endpoint_lateral_error_m": endpoint_lateral,
        "endpoint_angular_error_rad": endpoint_angular,
        "successor_viable": successor,
        "stuck": stuck,
        "teacher_valid": bool(teacher["teacher_valid"]),
    }


def select_teacher_pool_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    """Freeze the 256 teacher outcomes and select one hash-min state/stratum."""

    require_stage_runtime("ordinary", fake=fake_runtime)
    _runtime, pool = _require_initialized()
    if (MATERIAL_ROOT / "teacher_selection.json").exists():
        raise ExperimentError("teacher selection is already frozen")
    if any((MATERIAL_ROOT / "fanout").iterdir()):
        raise ExperimentError("fanout outcome exists before teacher selection")
    specs = [dict(value) for value in pool["specs"]]
    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    traces: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        metadata, arrays = _load_material_shard(_qualification_directory(index))
        if metadata["candidate_spec"] != spec or metadata["pool_index"] != index:
            raise ExperimentError("teacher shard/spec order drift")
        loaded.append((metadata, arrays))
        traces.append({member: arrays[f"teacher__{member}"] for member in TEACHER_TRACE_MEMBERS})
    teacher_arrays, spans = _concat_traces(traces, TEACHER_TRACE_MEMBERS)
    teacher_path = OUTPUT_ROOT / "teacher_traces.npz"
    atomic_npz(teacher_path, **teacher_arrays)

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (metadata, _arrays) in enumerate(loaded):
        spec = metadata["candidate_spec"]
        grouped[(str(spec["family"]), int(spec["stratum_index"]))].append(index)
    selected_indices: set[int] = set()
    ranks: dict[int, int] = {}
    for family in FAMILIES:
        for stratum in range(16):
            candidates = grouped[(family, stratum)]
            qualified = [index for index in candidates if loaded[index][0]["qualified"]]
            ordered = sorted(
                qualified,
                key=lambda index: (
                    str(loaded[index][0]["candidate_spec_sha256"]),
                    str(loaded[index][0]["candidate_spec"]["candidate_spec_id"]),
                ),
            )
            if not ordered:
                raise ExperimentError(
                    f"teacher-only scan has no qualified candidate for {family}/{stratum}"
                )
            selected_indices.add(ordered[0])
            for rank, index in enumerate(ordered):
                ranks[index] = rank
    if len(selected_indices) != STATE_COUNT:
        raise ExperimentError("teacher-only selection cardinality drift")

    teacher_records = [
        _teacher_record(
            metadata, arrays, trace_index=index, span=spans[index],
            selected=index in selected_indices, teacher_file=teacher_path,
        )
        for index, (metadata, arrays) in enumerate(loaded)
    ]
    qualification_rows: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    for index, (metadata, _arrays) in enumerate(loaded):
        spec = metadata["candidate_spec"]
        teacher = metadata["teacher"]
        crossing = teacher.get("crossing") or {}
        qualified = bool(metadata["qualified"])
        selected = index in selected_indices
        if selected:
            reason = None
        elif qualified:
            reason = "HASH_ORDER_NOT_SELECTED"
        else:
            components = {
                "goal_reachable": bool(metadata["goal_reachable"]),
                "teacher_trace_contact_free": bool(teacher["contact_free"]),
                "teacher_left_source_region": bool(teacher["left_source_region"]),
                "teacher_crossed_directed_port": bool(crossing),
                "teacher_positive_route_progress": bool(teacher["positive_route_progress"]),
                "teacher_no_competing_port": not bool(teacher["competing_port_entered"]),
                "teacher_normal_positive": bool(
                    crossing and crossing["normal_dot_displacement_m"] > 0
                ),
                "teacher_within_lateral_bounds": bool(
                    crossing and abs(crossing["lateral_coordinate_m"])
                    <= float(spec["passage_width_m"]) / 2.0 + 1e-9
                ),
                "teacher_dwell_satisfied": bool(
                    crossing and crossing["sustained_or_target_reached"]
                ),
                "directed_port_defined": bool(crossing),
                "current_rgb_valid": True,
                "graph_edge_physically_executable": bool(
                    metadata["graph_edge_physically_executable"]
                ),
            }
            reason = ";".join(sorted(key for key, value in components.items() if not value))
            rejection_counts[reason] += 1
        qualification_rows.append(
            {
                "candidate_spec_id": spec["candidate_spec_id"],
                "state_id": spec["state_id"],
                "family": spec["family"],
                "stratum_index": int(spec["stratum_index"]),
                "variant_index": int(spec["variant_index"]),
                "canonical_spec_sha256": spec["canonical_spec_sha256"],
                "teacher_trace_id": teacher_records[index]["teacher_trace_id"],
                "teacher_trace_index": index,
                "teacher_trace_slice_sha256": teacher_records[index][
                    "trace_array_slice_sha256s"
                ]["base_pose_world"],
                "initial_decision_state_sha256": metadata["initial_decision_state_sha256"],
                "goal_reachable": bool(metadata["goal_reachable"]),
                "teacher_trace_contact_free": bool(teacher["contact_free"]),
                "teacher_left_source_region": bool(teacher["left_source_region"]),
                "teacher_crossed_directed_port": crossing != {},
                "teacher_positive_route_progress": bool(teacher["positive_route_progress"]),
                "teacher_competing_port_entered": bool(teacher["competing_port_entered"]),
                "teacher_normal_positive": bool(crossing and crossing["normal_dot_displacement_m"] > 0),
                "teacher_within_lateral_bounds": bool(crossing and abs(crossing["lateral_coordinate_m"]) <= float(spec["passage_width_m"]) / 2.0 + 1e-9),
                "teacher_dwell_satisfied": bool(crossing and crossing["sustained_or_target_reached"]),
                "directed_port_defined": crossing != {},
                "current_rgb_valid": True,
                "graph_edge_physically_executable": bool(metadata["graph_edge_physically_executable"]),
                "teacher_valid": bool(teacher["teacher_valid"]),
                "qualified": qualified,
                "rejection_reason": reason,
                "selection_key_sha256": str(metadata["candidate_spec_sha256"]),
                "rank_within_stratum": int(ranks[index]) if qualified else None,
                "selected": selected,
            }
        )
    selected_specs = [specs[index] for index in sorted(selected_indices)]
    selection = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.teacher_selection.v1",
            "experiment_id": EXPERIMENT_ID,
            "candidate_specs_sha256": hashlib.sha256(
                canonical_bytes(CONTRACT.build_candidate_specs())[:-1]
            ).hexdigest(),
            "teacher_traces_file": _file_binding(teacher_path),
            "teacher_records": teacher_records,
            "qualification_rows": qualification_rows,
            "qualification_projection_sha256": hashlib.sha256(
                canonical_bytes(qualification_rows)[:-1]
            ).hexdigest(),
            "selected_specs": selected_specs,
            "selected_pool_indices": sorted(selected_indices),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "fanout_or_ranker_outcomes_opened": 0,
        }
    )
    atomic_json(MATERIAL_ROOT / "teacher_selection.json", selection)
    return selection


SNAPSHOT_NUMERIC_FIELDS = {
    "base_pose_world": (7,),
    "base_twist_world": (6,),
    "joint_position": (12,),
    "joint_velocity": (12,),
    "controller_observation": (45,),
    "policy_last_action": (12,),
    "previous_policy_action": (12,),
    "previous_applied_command": (3,),
    "command_history": (15, 3),
    "control_history": (15, 2),
    "low_level_policy_state": (12,),
    "camera_world_transform": (4, 4),
}
SNAPSHOT_METADATA_FIELDS = {
    "serialized_solver_state_sha256",
    "serialized_controller_state_sha256",
    "serialized_rng_state_sha256",
    "policy_last_action_sha256",
    "capture_timestamp_s",
    "controller_observation_sha256",
    "previous_policy_action_sha256",
    "torch_cpu_rng_state_sha256",
    "torch_device_rng_state_sha256s",
    "torch_device_count",
    "previous_applied_command_sha256",
    "command_history_sha256",
    "control_history_sha256",
    "low_level_policy_state_sha256",
    "solver_field_inventory",
    "controller_field_inventory",
    "rng_field_inventory",
}


def _selected_directory(state_id: str) -> Path:
    return MATERIAL_ROOT / "selected" / str(state_id)


def capture_selected_state_stage(
    state_id: str,
    *,
    backend: Any | None = None,
    fake_runtime: bool = False,
) -> dict[str, Any]:
    """Qualify two restores of the exact snapshot captured before its teacher."""

    runtime = require_stage_runtime("physical", fake=fake_runtime)
    _require_initialized()
    selection = load_json(MATERIAL_ROOT / "teacher_selection.json")
    selected = {
        str(value["state_id"]): dict(value) for value in selection["selected_specs"]
    }
    if str(state_id) not in selected:
        raise ExperimentError("snapshot capture requested for a non-selected pool state")
    if any((MATERIAL_ROOT / "fanout").iterdir()):
        raise ExperimentError("selected reset qualification cannot run after fanout")
    spec = selected[str(state_id)]
    teacher_row = next(
        row for row in selection["teacher_records"] if row["state_id"] == str(state_id)
    )
    pool_index = int(teacher_row["trace_index"])
    qualification_metadata, qualification_arrays = _load_material_shard(
        _qualification_directory(pool_index)
    )
    if qualification_metadata["candidate_spec"] != spec:
        raise ExperimentError("selected snapshot qualification/spec binding drift")
    snapshot_payload = bytes(
        qualification_arrays["snapshot_payload_bytes"].astype("uint8").tobytes()
    )
    snapshot_sha = hashlib.sha256(snapshot_payload).hexdigest()
    if snapshot_sha != teacher_row["initial_decision_state_sha256"]:
        raise ExperimentError("selected snapshot does not bind the teacher initial state")
    collector = _qualification_backend_default() if backend is None else backend
    raw = collector.reset_fixture(copy.deepcopy(spec), snapshot_payload)
    if not isinstance(raw, Mapping) or set(raw) != {
        "candidate_spec_id", "snapshot_payload_sha256", "reset_trials",
        "runtime_evidence",
    }:
        raise ExperimentError("selected reset-fixture result field drift")
    if (
        raw["candidate_spec_id"] != spec["candidate_spec_id"]
        or raw["snapshot_payload_sha256"] != snapshot_sha
    ):
        raise ExperimentError("reset fixture did not use the teacher decision snapshot")
    backend_runtime = dict(raw["runtime_evidence"])
    runtime = _bind_physical_backend_runtime(runtime, backend_runtime)
    import numpy as np

    arrays: dict[str, Any] = {
        "snapshot_payload_bytes": qualification_arrays["snapshot_payload_bytes"].copy(),
        "rgb": qualification_arrays["rgb"].copy(),
    }
    if arrays["rgb"].shape != (168, 224, 3) or canonical_array_sha256(arrays["rgb"]) != teacher_row["current_rgb_sha256"]:
        raise ExperimentError("selected snapshot RGB no longer binds the teacher capture")
    for field, shape in SNAPSHOT_NUMERIC_FIELDS.items():
        value = np.ascontiguousarray(
            np.asarray(qualification_arrays[f"snapshot__{field}"], dtype=np.float64)
        )
        if value.shape != shape or not np.isfinite(value).all():
            raise ExperimentError(f"selected snapshot {field} shape/value drift")
        arrays[f"snapshot__{field}"] = value
    trials = [dict(value) for value in raw["reset_trials"]]
    if len(trials) != 2:
        raise ExperimentError("selected state lacks two serialized restore trials")
    trial_rows: list[dict[str, Any]] = []
    trial_traces: list[dict[str, Any]] = []
    snapshot = dict(qualification_metadata["snapshot"])
    for trial_index, trial in enumerate(trials):
        if set(trial) != {"metadata", "trace"}:
            raise ExperimentError("selected reset trial wrapper drift")
        metadata = dict(trial["metadata"])
        trace = _normalise_trace(
            dict(trial["trace"]), kind="candidate", exact_samples=RESET_FIXTURE_PHYSICS_STEPS
        )
        if (
            metadata.get("serialized_restore_used") is not True
            or metadata.get("clone_equivalence_used") is not False
            or metadata.get("restored_snapshot_sha256") != snapshot_sha
        ):
            raise ExperimentError("selected reset trial is not an exact serialized restore")
        for member, value in trace.items():
            arrays[f"fixture_{trial_index}__{member}"] = value
        # ``exact_match`` is computed from the two traces below.  It is never
        # accepted as a free assertion from the physical backend.
        metadata.pop("exact_match", None)
        trial_rows.append({**metadata, "trace_digests": _trace_digest_projection(trace)})
        trial_traces.append(trace)
    reset_match, reset_maxima = _reset_trace_pair_matches(
        trial_traces[0], trial_traces[1]
    )
    if not reset_match:
        raise ExperimentError(
            f"selected reset trials violate frozen full-trace comparison: {reset_maxima}"
        )
    reset_pair = _reset_pair_comparison(
        str(state_id), trial_traces[0], trial_traces[1],
        first_termination_reason=str(trial_rows[0]["termination_reason"]),
        second_termination_reason=str(trial_rows[1]["termination_reason"]),
        first_stuck=bool(trial_rows[0]["stuck"]),
        second_stuck=bool(trial_rows[1]["stuck"]),
    )
    if not reset_pair["passed"]:
        raise ExperimentError("selected reset-pair comparison did not pass")
    metadata = {
        "schema": "physical_graph_edge_handoff_qualification_v1.selected_state_shard.v1",
        "experiment_id": EXPERIMENT_ID,
        "state_id": str(state_id),
        "candidate_spec": spec,
        "source_teacher_trace_id": teacher_row["teacher_trace_id"],
        "teacher_initial_state_sha256": teacher_row["initial_decision_state_sha256"],
        "snapshot": {field: snapshot[field] for field in sorted(SNAPSHOT_METADATA_FIELDS)},
        "snapshot_payload_sha256": snapshot_sha,
        "reset_trials": trial_rows,
        "reset_pair_comparison": reset_pair,
        "reset_fixture_passed": True,
        "rgb_sha256": canonical_array_sha256(arrays["rgb"]),
        "stage_runtime": runtime,
        "backend_runtime": backend_runtime,
        "fanout_or_ranker_outcome_opened": False,
    }
    _write_material_shard(_selected_directory(str(state_id)), metadata, arrays)
    return metadata


def _pose_yaw_xyzw(base_pose_world: Sequence[float]) -> float:
    if len(base_pose_world) != 7:
        raise ExperimentError("base pose must be xyz plus quaternion xyzw")
    x, y, z, w = (float(value) for value in base_pose_world[3:7])
    return _wrap_angle(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _interpolated_pose_se2(
    before: Sequence[float], after: Sequence[float], fraction: float
) -> list[float]:
    fraction = float(fraction)
    yaw_before = _pose_yaw_xyzw(before)
    yaw_after = _pose_yaw_xyzw(after)
    return [
        float(before[0]) + fraction * (float(after[0]) - float(before[0])),
        float(before[1]) + fraction * (float(after[1]) - float(before[1])),
        _wrap_angle(yaw_before + fraction * _wrap_angle(yaw_after - yaw_before)),
    ]


def _polyline_length(polyline: Sequence[Sequence[float]]) -> float:
    return float(sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for a, b in zip(polyline, polyline[1:])
    ))


def _lookahead_from_port(
    port_pose: Sequence[float], route: Sequence[Sequence[float]], distance_m: float
) -> tuple[list[float], bool, float]:
    """Walk a directed physical route from its closest point to the actual port."""

    points = [[float(value[0]), float(value[1])] for value in route]
    if len(points) < 2:
        raise ExperimentError("directed route polyline is degenerate")
    px, py = float(port_pose[0]), float(port_pose[1])
    candidates: list[tuple[float, int, float, list[float]]] = []
    for index, (left, right) in enumerate(zip(points, points[1:])):
        dx, dy = right[0] - left[0], right[1] - left[1]
        denom = dx * dx + dy * dy
        if denom <= 0.0:
            continue
        fraction = max(0.0, min(1.0, ((px - left[0]) * dx + (py - left[1]) * dy) / denom))
        projected = [left[0] + fraction * dx, left[1] + fraction * dy]
        candidates.append((math.hypot(px - projected[0], py - projected[1]), index, fraction, projected))
    if not candidates:
        raise ExperimentError("directed route has no positive-length segment")
    _distance, segment_index, fraction, projected = min(candidates)
    remaining_segments: list[tuple[list[float], list[float]]] = [
        (projected, points[segment_index + 1])
    ] + list(zip(points[segment_index + 1 :], points[segment_index + 2 :]))
    remaining = float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in remaining_segments))
    desired = min(float(distance_m), remaining)
    travelled = 0.0
    target = projected
    heading = float(port_pose[2])
    for left, right in remaining_segments:
        length = math.hypot(right[0] - left[0], right[1] - left[1])
        if length <= 0.0:
            continue
        heading = math.atan2(right[1] - left[1], right[0] - left[0])
        if travelled + length >= desired:
            alpha = (desired - travelled) / length
            target = [left[0] + alpha * (right[0] - left[0]), left[1] + alpha * (right[1] - left[1])]
            break
        travelled += length
        target = right
    return [float(target[0]), float(target[1]), _wrap_angle(heading)], remaining < float(distance_m), remaining


def _split_assignments(selected_specs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    assignments: list[dict[str, Any]] = []
    population: list[dict[str, Any]] = []
    for family in FAMILIES:
        ranked: list[tuple[str, str, Mapping[str, Any]]] = []
        for spec in selected_specs:
            if spec["family"] != family:
                continue
            projection = {
                "experiment_id": EXPERIMENT_ID,
                "state_id": spec["state_id"],
                "family": family,
                "candidate_spec_id": spec["candidate_spec_id"],
                "canonical_spec_sha256": spec["canonical_spec_sha256"],
            }
            assignment_sha = hashlib.sha256(canonical_bytes(projection)[:-1]).hexdigest()
            ranked.append((assignment_sha, str(spec["state_id"]), spec))
        ranked.sort(key=lambda value: (value[0], value[1]))
        if len(ranked) != 16:
            raise ExperimentError(f"selected family cardinality drift for {family}")
        for rank, (assignment_sha, state_id, spec) in enumerate(ranked):
            role = "DEVELOPMENT" if rank < 12 else "DEVELOPMENT_HELDOUT"
            assignments.append(
                {
                    "state_id": state_id,
                    "role": role,
                    "assignment_sha256": assignment_sha,
                    "rank_within_family": rank,
                }
            )
            population.append(
                {
                    "state_id": state_id,
                    "family": family,
                    "candidate_spec_id": spec["candidate_spec_id"],
                    "assignment_sha256": assignment_sha,
                }
            )
    population.sort(key=lambda row: row["state_id"])
    assignments.sort(key=lambda row: row["state_id"])
    return assignments, hashlib.sha256(canonical_bytes(population)[:-1]).hexdigest()


def _graph_record(
    state_id: str, spec: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    graph = dict(metadata["graph"])
    required = {
        "graph_id", "source_node_id", "target_node_id", "directed_edge_id",
        "nodes", "edges", "goal_reachable", "graph_edge_physically_executable",
        "teacher_positive_route_progress", "teacher_competing_port_entered",
    }
    if set(graph) != required:
        raise ExperimentError("physical graph backend schema drift")
    return {
        "state_id": state_id,
        "graph_id": str(graph["graph_id"]),
        "source_node_id": str(graph["source_node_id"]),
        "target_node_id": str(graph["target_node_id"]),
        "directed_edge_id": str(graph["directed_edge_id"]),
        "nodes": copy.deepcopy(list(graph["nodes"])),
        "edges": copy.deepcopy(list(graph["edges"])),
    }


def _edge_port_record(
    spec: Mapping[str, Any], teacher_row: Mapping[str, Any],
    teacher_trace: Mapping[str, Any], graph_record: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    after = int(teacher_row["first_crossing_sample_index"])
    before = after - 1
    fraction = float(teacher_row["crossing_segment_fraction"])
    port = _interpolated_pose_se2(
        teacher_trace["base_pose_world"][before],
        teacher_trace["base_pose_world"][after],
        fraction,
    )
    edge = next(
        row for row in graph_record["edges"]
        if row["edge_id"] == graph_record["directed_edge_id"]
    )
    route = copy.deepcopy(edge["route_polyline_world"])
    lookahead, clipped, remaining = _lookahead_from_port(
        port, route, float(CONTRACT.ROUTE_LOOKAHEAD_DISTANCE_M)
    )
    source = next(
        row for row in graph_record["nodes"]
        if row["node_id"] == graph_record["source_node_id"]
    )
    del np
    return {
        "state_id": str(spec["state_id"]),
        "directed_edge_id": str(graph_record["directed_edge_id"]),
        "source_node_id": str(graph_record["source_node_id"]),
        "target_node_id": str(graph_record["target_node_id"]),
        "teacher_trace_id": str(teacher_row["teacher_trace_id"]),
        "source_boundary_polygon_world": copy.deepcopy(source["boundary_polygon_world"]),
        "route_polyline_world": route,
        "crossing_sample_before": before,
        "crossing_sample_after": after,
        "crossing_fraction": fraction,
        "teacher_crossing_velocity_world_xy": copy.deepcopy(
            teacher_row["crossing_velocity_world_xy"]
        ),
        "teacher_crossing_velocity_heading_world_rad": float(
            teacher_row["crossing_velocity_heading_world_rad"]
        ),
        "directed_port_world": port,
        "route_lookahead_world": lookahead,
        "route_lookahead_clipped": clipped,
        "remaining_route_length_m": remaining,
        "port_definition": "first actual teacher source-boundary crossing through selected transverse opening",
    }


def _waypoint_row(
    state_id: str, target_id: str, source_pose: Sequence[float],
    target_pose: Sequence[float], route_intent_world: Sequence[float],
) -> dict[str, Any]:
    body = body_from_world(target_pose, source_pose)
    roundtrip = world_from_body(body, source_pose)
    position_error = math.hypot(roundtrip[0] - float(target_pose[0]), roundtrip[1] - float(target_pose[1]))
    heading_error = abs(_wrap_angle(roundtrip[2] - float(target_pose[2])))
    c, s = math.cos(float(source_pose[2])), math.sin(float(source_pose[2]))
    route_dx = c * float(route_intent_world[0]) + s * float(route_intent_world[1])
    route_dy = -s * float(route_intent_world[0]) + c * float(route_intent_world[1])
    bearing = math.atan2(body[1], body[0])
    return {
        "state_id": state_id,
        "target_id": target_id,
        "source_body_pose_world": [float(value) for value in source_pose],
        "target_world_pose": [float(value) for value in target_pose],
        "target_body_pose": body,
        "dx_m": body[0],
        "dy_m": body[1],
        "distance_m": math.hypot(body[0], body[1]),
        "relative_heading_rad": bearing,
        "relative_heading_sin": math.sin(bearing),
        "relative_heading_cos": math.cos(bearing),
        "target_tangent_heading_rad": body[2],
        "target_tangent_heading_sin": math.sin(body[2]),
        "target_tangent_heading_cos": math.cos(body[2]),
        "route_intent_dx": route_dx,
        "route_intent_dy": route_dy,
        "roundtrip_world_pose": roundtrip,
        "position_transform_error_m": position_error,
        "heading_transform_error_rad": heading_error,
        "transform_valid": bool(
            position_error <= float(CONTRACT.NUMERICAL_TOLERANCES["inverse_transform_position_m"])
            and heading_error <= float(CONTRACT.NUMERICAL_TOLERANCES["inverse_transform_heading_rad"])
        ),
    }


def _physical_runtime_environment_from_shards(
    selected_specs: Sequence[Mapping[str, Any]],
    selected_shards: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Prove one physical core across all 256 teacher and 64 reset shards."""

    qualification_digests: list[str] = []
    qualification_cores: list[dict[str, Any]] = []
    for pool_index in range(PROSPECTIVE_POOL_COUNT):
        metadata, _arrays = _load_material_shard(
            _qualification_directory(pool_index)
        )
        core = copy.deepcopy(dict(metadata["stage_runtime"]))
        if set(core) != set(CONTRACT.PHYSICAL_RUNTIME_CORE_FIELDS):
            raise ExperimentError("qualification physical runtime field drift")
        backend_runtime = dict(metadata["backend_runtime"])
        if backend_runtime.get("backend") != core["backend"]:
            raise ExperimentError("qualification backend/runtime binding drift")
        qualification_cores.append(core)
        qualification_digests.append(METRICS.runtime_environment_sha256(core))

    selected_digests: list[str] = []
    selected_cores: list[dict[str, Any]] = []
    for spec in selected_specs:
        state_id = str(spec["state_id"])
        metadata = dict(selected_shards[state_id][0])
        core = copy.deepcopy(dict(metadata["stage_runtime"]))
        if set(core) != set(CONTRACT.PHYSICAL_RUNTIME_CORE_FIELDS):
            raise ExperimentError("selected-reset physical runtime field drift")
        backend_runtime = dict(metadata["backend_runtime"])
        if backend_runtime.get("backend") != core["backend"]:
            raise ExperimentError("selected-reset backend/runtime binding drift")
        selected_cores.append(core)
        selected_digests.append(METRICS.runtime_environment_sha256(core))

    all_cores = qualification_cores + selected_cores
    if len(all_cores) != PROSPECTIVE_POOL_COUNT + STATE_COUNT:
        raise ExperimentError("physical runtime shard cardinality drift")
    core = all_cores[0]
    if any(value != core for value in all_cores[1:]):
        raise ExperimentError("physical runtime changed across collection shards")
    runtime_core_sha256 = METRICS.runtime_environment_sha256(core)
    environment = {
        **copy.deepcopy(core),
        "runtime_core_sha256": runtime_core_sha256,
        "qualification_runtime_sha256s": qualification_digests,
        "selected_snapshot_runtime_sha256s": selected_digests,
    }
    return METRICS.validate_physical_runtime_environment(environment)


def freeze_panel_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    """Freeze the exact-reset-qualified 64-state panel and role assignment."""

    import numpy as np

    require_stage_runtime("ordinary", fake=fake_runtime)
    _runtime, pool = _require_initialized()
    selection = load_json(MATERIAL_ROOT / "teacher_selection.json")
    if any((MATERIAL_ROOT / "fanout").iterdir()):
        raise ExperimentError("candidate outcome exists before panel/split freeze")
    selected_specs = [dict(value) for value in selection["selected_specs"]]
    if len(selected_specs) != STATE_COUNT:
        raise ExperimentError("selected panel cardinality drift")
    selected_shards: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for spec in selected_specs:
        state_id = str(spec["state_id"])
        selected_shards[state_id] = _load_material_shard(_selected_directory(state_id))
        if selected_shards[state_id][0].get("reset_fixture_passed") is not True:
            raise ExperimentError(f"serialized reset fixture failed: {state_id}")

    assignments, population_sha = _split_assignments(selected_specs)
    role_by_state = {row["state_id"]: row["role"] for row in assignments}
    teacher_records = [dict(value) for value in selection["teacher_records"]]
    teacher_by_state = {
        row["state_id"]: row for row in teacher_records if row["selected"]
    }
    qualification_by_state = {
        row["state_id"]: row for row in selection["qualification_rows"] if row["selected"]
    }
    graph_records: list[dict[str, Any]] = []
    panel_states: list[dict[str, Any]] = []
    for spec in selected_specs:
        state_id = str(spec["state_id"])
        pool_index = int(teacher_by_state[state_id]["trace_index"])
        qualification_metadata, _qualification_arrays = _load_material_shard(
            _qualification_directory(pool_index)
        )
        graph_record = _graph_record(state_id, spec, qualification_metadata)
        graph_records.append(graph_record)
        evidence = {
            "snapshot_complete": True,
            "exact_reset_fixture_passed": True,
            "teacher_trace_contact_free": True,
            "teacher_valid": True,
            "directed_port_defined": True,
            "current_rgb_valid": True,
            "graph_edge_physically_executable": True,
        }
        qualification = qualification_by_state[state_id]
        if not all(qualification[field] is True for field in (
            "teacher_trace_contact_free", "teacher_valid", "directed_port_defined",
            "current_rgb_valid", "graph_edge_physically_executable",
        )):
            raise ExperimentError("selected panel contains an ineligible teacher state")
        panel_states.append(
            {
                "state_id": state_id,
                "scene_id": str(spec["scene_id"]),
                "episode_id": str(spec["episode_id"]),
                "graph_id": str(spec["graph_id"]),
                "family": str(spec["family"]),
                "turn_direction": (
                    str(spec["route_direction"])
                    if spec["family"] == "TURNING_JUNCTION" else None
                ),
                "route_direction": str(spec["route_direction"]),
                "passage_width_id": str(spec["passage_width_id"]),
                "port_distance_id": str(spec["port_distance_id"]),
                "stratum_index": int(spec["stratum_index"]),
                "candidate_spec_id": str(spec["candidate_spec_id"]),
                "role": role_by_state[state_id],
                "procedural_seed": int(spec["procedural_seed"]),
                "geometry_sha256": hashlib.sha256(
                    canonical_bytes(spec["geometry"])[:-1]
                ).hexdigest(),
                "source_node_id": str(graph_record["source_node_id"]),
                "target_node_id": str(graph_record["target_node_id"]),
                "directed_edge_id": str(graph_record["directed_edge_id"]),
                "goal_reachable": True,
                "eligible": True,
                "eligibility_evidence": evidence,
            }
        )

    eligible_counts = {
        family: sum(
            bool(row["qualified"]) and row["family"] == family
            for row in selection["qualification_rows"]
        )
        for family in FAMILIES
    }
    selected_projection = {
        family: [
            next(
                str(spec["candidate_spec_id"]) for spec in selected_specs
                if spec["family"] == family and int(spec["stratum_index"]) == stratum
            )
            for stratum in range(16)
        ]
        for family in FAMILIES
    }
    physical_runtime_environment = _physical_runtime_environment_from_shards(
        selected_specs, selected_shards,
    )
    panel = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.panel_manifest.v1",
            "experiment_id": EXPERIMENT_ID,
            "constructed_set_caveat": CONTRACT.CONSTRUCTED_SET_CAVEAT,
            "identity_domain": CONTRACT.IDENTITY_DOMAIN,
            "prior_exclusion_evidence": pool["prior_exclusion_evidence"],
            "prospective_pool_selection": {
                "candidate_spec_count": PROSPECTIVE_POOL_COUNT,
                "candidate_specs_sha256": selection["candidate_specs_sha256"],
                "selection_rule": CONTRACT.GEOMETRY_AUTHORITY["teacher_only_scan"]["selection"],
                "teacher_scan_completed_before_role_assignment": True,
                "ranker_or_fanout_outcomes_opened": 0,
                "per_family_scanned_counts": {family: 64 for family in FAMILIES},
                "per_family_teacher_eligible_counts": eligible_counts,
                "selected_candidate_spec_ids_by_family_and_stratum": selected_projection,
                "rejection_reason_counts": selection["rejection_reason_counts"],
                "qualification_rows": selection["qualification_rows"],
                "qualification_projection_sha256": selection["qualification_projection_sha256"],
                "selected_state_ids": [str(spec["state_id"]) for spec in selected_specs],
                "complete": True,
            },
            "physical_runtime_environment": physical_runtime_environment,
            "states": panel_states,
        }
    )
    split = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.split_manifest.v1",
            "experiment_id": EXPERIMENT_ID,
            "assignment_algorithm": (
                "after the complete teacher-qualified 16-state family population is frozen, "
                "sort each family by SHA256(state canonical identity); first 12 DEVELOPMENT, "
                "last 4 DEVELOPMENT_HELDOUT"
            ),
            "complete_eligible_population_sha256": population_sha,
            "heldout_outcomes_opened_before_assignment": 0,
            "role_counts": copy.deepcopy(CONTRACT.ROLE_COUNTS),
            "family_role_counts": copy.deepcopy(CONTRACT.FAMILY_ROLE_COUNTS),
            "assignments": assignments,
        }
    )
    graphs = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.graph_manifest.v1",
            "experiment_id": EXPERIMENT_ID,
            "graphs": graph_records,
        }
    )

    # Consolidate exactly the selected snapshot arrays.  Opaque bytes remain
    # byte-for-byte those captured before each corresponding teacher rollout.
    snapshot_offsets = [0]
    snapshot_payload_parts: list[Any] = []
    snapshot_arrays: dict[str, list[Any]] = {
        field: [] for field in SNAPSHOT_NUMERIC_FIELDS
    }
    rgb_rows: list[Any] = []
    for spec in selected_specs:
        _metadata, arrays = selected_shards[str(spec["state_id"])]
        payload = np.asarray(arrays["snapshot_payload_bytes"], dtype=np.uint8)
        snapshot_payload_parts.append(payload)
        snapshot_offsets.append(snapshot_offsets[-1] + len(payload))
        for field in SNAPSHOT_NUMERIC_FIELDS:
            snapshot_arrays[field].append(arrays[f"snapshot__{field}"])
        rgb_rows.append(arrays["rgb"])
    atomic_npz(
        OUTPUT_ROOT / "state_snapshots.npz",
        snapshot_payload_bytes=np.ascontiguousarray(np.concatenate(snapshot_payload_parts)),
        snapshot_offsets=np.asarray(snapshot_offsets, dtype=np.int64),
        **{
            field: np.ascontiguousarray(np.stack(rows, axis=0)).astype(np.float64)
            for field, rows in snapshot_arrays.items()
        },
    )
    rgb = np.ascontiguousarray(np.stack(rgb_rows, axis=0)).astype(np.uint8)
    atomic_npz(OUTPUT_ROOT / "rgb_observations.npz", rgb=rgb)

    pixel_hashes = [canonical_array_sha256(row) for row in rgb]
    canonical_hashes = sorted(set(pixel_hashes))
    canonical_index = {value: index for index, value in enumerate(canonical_hashes)}
    pixel = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.pixel_index.v1",
            "experiment_id": EXPERIMENT_ID,
            "rgb_file": _file_binding(OUTPUT_ROOT / "rgb_observations.npz"),
            "hash_domain": CONTRACT.CANONICAL_ENCODING_AUTHORITY["pixel_identity"],
            "records": [
                {
                    "state_id": str(spec["state_id"]),
                    "capture_id": f"{spec['state_id']}:decision-rgb",
                    "rgb_row_index": index,
                    "pixel_sha256": pixel_hashes[index],
                    "canonical_pixel_index": canonical_index[pixel_hashes[index]],
                    "row_sha256": pixel_hashes[index],
                }
                for index, spec in enumerate(selected_specs)
            ],
            "unique_pixel_count": len(canonical_hashes),
        }
    )

    # Derive actual ports only from the persisted teacher trajectories.
    with np.load(OUTPUT_ROOT / "teacher_traces.npz", allow_pickle=False) as data:
        teacher_payload = {name: np.ascontiguousarray(data[name]) for name in data.files}
    port_rows: list[dict[str, Any]] = []
    waypoint_rows: list[dict[str, Any]] = []
    graph_by_state = {row["state_id"]: row for row in graph_records}
    spec_by_state = {str(row["state_id"]): row for row in selected_specs}
    for state_index, state in enumerate(panel_states):
        state_id = state["state_id"]
        teacher = teacher_by_state[state_id]
        start, stop = int(teacher["trace_slice"]["start"]), int(teacher["trace_slice"]["stop"])
        trace = {
            member: teacher_payload[member][start:stop]
            for member in TEACHER_TRACE_MEMBERS
        }
        port = _edge_port_record(
            spec_by_state[state_id], teacher, trace, graph_by_state[state_id]
        )
        port_rows.append(port)
        snapshot_pose = snapshot_arrays["base_pose_world"][state_index]
        source_pose = [
            float(snapshot_pose[0]), float(snapshot_pose[1]), _pose_yaw_xyzw(snapshot_pose)
        ]
        normal = spec_by_state[state_id]["geometry"]["selected_directed_edge"]["opening_normal_world"]
        graph = graph_by_state[state_id]
        target_node = next(row for row in graph["nodes"] if row["node_id"] == graph["target_node_id"])
        route = next(row for row in graph["edges"] if row["edge_id"] == graph["directed_edge_id"])["route_polyline_world"]
        last_a, last_b = route[-2], route[-1]
        target_heading = math.atan2(last_b[1] - last_a[1], last_b[0] - last_a[0])
        targets = {
            "TARGET_NODE_CENTRE": [*target_node["centre_world"], target_heading],
            "DIRECTED_EDGE_PORT": port["directed_port_world"],
            "ROUTE_LOOKAHEAD": port["route_lookahead_world"],
        }
        for target_id in TARGET_IDS:
            waypoint_rows.append(
                _waypoint_row(state_id, target_id, source_pose, targets[target_id], normal)
            )
    ports = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.edge_port_index.v1",
            "experiment_id": EXPERIMENT_ID,
            "records": port_rows,
        }
    )
    support = CONTRACT.ORIGINAL_RANKER_TRAINING_SUPPORT
    support_counts = {target_id: {"inside": 0, "outside": 0} for target_id in TARGET_IDS}
    for row in waypoint_rows:
        inside = all(
            interval[0] - support["support_tolerance"] <= value <= interval[1] + support["support_tolerance"]
            for value, interval in (
                (row["dx_m"], support["body_dx_m"]),
                (row["dy_m"], support["body_dy_m"]),
                (row["distance_m"], support["distance_m"]),
                (row["relative_heading_rad"], support["relative_heading_rad"]),
                (row["relative_heading_sin"], support["relative_heading_sin"]),
                (row["relative_heading_cos"], support["relative_heading_cos"]),
            )
        )
        support_counts[row["target_id"]]["inside" if inside else "outside"] += 1
    waypoints = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.waypoint_contracts.v1",
            "experiment_id": EXPERIMENT_ID,
            "target_ids": list(TARGET_IDS),
            "feature_order": list(CONTRACT.TARGET_FEATURE_ORDER),
            "rows": waypoint_rows,
            "training_contract_binding": copy.deepcopy(CONTRACT.CURRENT_VISUAL_RANKER_BINDING),
            "original_ranker_support": copy.deepcopy(CONTRACT.ORIGINAL_RANKER_TRAINING_SUPPORT),
            "contract_difference_inventory": list(CONTRACT.RANKER_CONTRACT_DIFFERENCE_INVENTORY),
            "target_support_counts": support_counts,
        }
    )
    teacher_index = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.teacher_trace_index.v1",
            "experiment_id": EXPERIMENT_ID,
            "traces_file": _file_binding(OUTPUT_ROOT / "teacher_traces.npz"),
            "records": teacher_records,
        }
    )
    for path, document in (
        (OUTPUT_ROOT / "panel_manifest.json", panel),
        (OUTPUT_ROOT / "split_manifest.json", split),
        (OUTPUT_ROOT / "graph_manifest.json", graphs),
        (OUTPUT_ROOT / "teacher_trace_index.json", teacher_index),
        (OUTPUT_ROOT / "edge_port_index.json", ports),
        (OUTPUT_ROOT / "waypoint_contracts.json", waypoints),
        (OUTPUT_ROOT / "pixel_index.json", pixel),
    ):
        atomic_json(path, document)
    context = attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.panel_context.v1",
            "experiment_id": EXPERIMENT_ID,
            "selected_state_ids": [row["state_id"] for row in panel_states],
            "role_by_state": role_by_state,
            "panel_sha256": sha256_file(OUTPUT_ROOT / "panel_manifest.json"),
            "split_sha256": sha256_file(OUTPUT_ROOT / "split_manifest.json"),
            "heldout_outcomes_opened": 0,
        }
    )
    atomic_json(MATERIAL_ROOT / "panel_context.json", context)
    return panel


def _token_layernorm_l2(value: Any) -> Any:
    import numpy as np

    raw = np.asarray(value, dtype=np.float32)
    if raw.shape != (768, 1024) or not np.isfinite(raw).all():
        raise ExperimentError(f"V-JEPA token contract drift: {raw.shape}")
    mean = raw.mean(axis=-1, keepdims=True, dtype=np.float32)
    variance = np.square(raw - mean, dtype=np.float32).mean(
        axis=-1, keepdims=True, dtype=np.float32
    )
    normalized = (raw - mean) / np.sqrt(variance + np.float32(1.0e-5))
    normalized /= np.maximum(
        np.linalg.norm(normalized, axis=-1, keepdims=True), np.float32(1.0e-12)
    )
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _preprocess_singleton_image(arm: Any, image: Any, temporary_root: Path) -> Any:
    """Use the audited V2 array-or-exact-PNG preprocessing compatibility path."""

    import numpy as np
    import torch

    rgb = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if rgb.shape != (168, 224, 3):
        raise ExperimentError("singleton encoder RGB input contract drift")
    from PIL import Image

    path = temporary_root / "canonical-frame.png"
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")
    value = arm.preprocess(str(path))
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (3, 384, 512)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise ExperimentError("encoder preprocessing output contract drift")
    return value.detach().cpu().contiguous()


def _verify_external_vjepa_source(repository: Path, expected_commit: str) -> dict[str, Any]:
    from scripts import run_occluded_goal_topological_belief_v1 as V1

    try:
        observed = dict(V1.verify_external_vjepa_source())
    except Exception as exc:
        raise ExperimentError("frozen external V-JEPA repository cannot be verified") from exc
    if (
        observed != {
            "path": str(repository),
            "commit": str(expected_commit),
            "worktree_clean": True,
        }
    ):
        raise ExperimentError("frozen external V-JEPA source commit/worktree drift")
    return {
        "repository_path": str(repository),
        "commit": str(expected_commit),
        "worktree_clean": True,
    }


class _FrozenSingletonEncoder:
    def __init__(self) -> None:
        import torch
        from scripts.dev_frozen_dense_representation_encoders_v1 import (
            VJEPA_REPOSITORY, VJepa21Arm,
        )

        if not torch.cuda.is_available():
            raise ExperimentError("supported frozen V-JEPA GPU path is unavailable")
        self.source_binding = _verify_external_vjepa_source(
            Path(VJEPA_REPOSITORY),
            str(CONTRACT.VJEPA_ENCODER_BINDING["external_repository_commit"]),
        )
        self.torch = torch
        self.arm = VJepa21Arm()
        checkpoint = Path(str(self.arm.checkpoint))
        if sha256_file(checkpoint) != FROZEN_ENCODER_SHA256:
            raise ExperimentError("frozen V-JEPA checkpoint binding drift")
        self.device = torch.device("cuda:0")
        built = self.arm.build(self.device, torch.float32)
        module = built if built is not None else getattr(self.arm, "_module", None)
        if (
            module is None
            or bool(getattr(module, "training", True))
            or any(bool(parameter.requires_grad) for parameter in module.parameters())
        ):
            raise ExperimentError("frozen V-JEPA encoder is not in evaluation mode")

    def encode_singleton(self, image: Any) -> Mapping[str, Any]:
        import numpy as np
        import tempfile

        with tempfile.TemporaryDirectory(prefix="pgehq-vjepa-singleton-") as directory:
            value = _preprocess_singleton_image(
                self.arm, image, Path(directory)
            )
        preprocessed = value.numpy()
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", dtype=self.torch.bfloat16
        ):
            tokens = self.arm.tokens(value.unsqueeze(0).to(self.device)).detach().float().cpu().numpy()[0]
        return {
            "raw_tokens": np.ascontiguousarray(tokens, dtype=np.float16),
            "spatial_descriptor": _token_layernorm_l2(tokens),
            "preprocessed_tensor_sha256": canonical_array_sha256(preprocessed),
            "checkpoint_sha256": FROZEN_ENCODER_SHA256,
            "batch_size": 1,
        }


def encode_canonical_pixels_stage(
    *, encoder: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    """Encode each unique exact pixel once, in sorted SHA order."""

    import numpy as np

    encoder_runtime = require_stage_runtime(
        "visual", fake=fake_runtime, visual_role="encoder",
    )
    _require_initialized()
    if any((MATERIAL_ROOT / "fanout").iterdir()):
        raise ExperimentError("physical candidate outcome exists before canonical encoding")
    panel = load_json(OUTPUT_ROOT / "panel_manifest.json")
    pixel = load_json(OUTPUT_ROOT / "pixel_index.json")
    with np.load(OUTPUT_ROOT / "rgb_observations.npz", allow_pickle=False) as data:
        rgb = np.ascontiguousarray(data["rgb"])
    if rgb.shape != (64, 168, 224, 3) or rgb.dtype != np.uint8:
        raise ExperimentError("selected RGB payload contract drift")
    by_hash: dict[str, Any] = {}
    for row in pixel["records"]:
        image = rgb[int(row["rgb_row_index"])]
        digest = canonical_array_sha256(image)
        if digest != row["pixel_sha256"]:
            raise ExperimentError("pixel index/image bytes drift")
        if digest in by_hash and not np.array_equal(by_hash[digest], image):
            raise ExperimentError("canonical pixel SHA collision")
        by_hash.setdefault(digest, image)
    order = sorted(by_hash)
    if len(order) != int(pixel["unique_pixel_count"]):
        raise ExperimentError("canonical pixel cardinality drift")
    runtime_encoder = _FrozenSingletonEncoder() if encoder is None else encoder
    source_authority = CONTRACT.CANONICAL_ENCODING_AUTHORITY["external_encoder_source"]
    expected_source = {
        "repository_path": str(source_authority["repository_path"]),
        "commit": str(source_authority["commit"]),
        "worktree_clean": True,
    }
    if encoder is None:
        external_source = copy.deepcopy(runtime_encoder.source_binding)
    elif fake_runtime:
        external_source = copy.deepcopy(expected_source)
    else:
        external_source = copy.deepcopy(getattr(runtime_encoder, "source_binding", None))
    if external_source != expected_source:
        raise ExperimentError("singleton encoder external-source binding drift")
    raw_rows: list[Any] = []
    descriptor_rows: list[Any] = []
    receipts: list[dict[str, Any]] = []
    for index, pixel_sha in enumerate(order):
        encoded = dict(runtime_encoder.encode_singleton(by_hash[pixel_sha]))
        if set(encoded) != {
            "raw_tokens", "spatial_descriptor", "preprocessed_tensor_sha256",
            "checkpoint_sha256", "batch_size",
        }:
            raise ExperimentError("singleton encoder result field drift")
        raw = np.ascontiguousarray(np.asarray(encoded["raw_tokens"], dtype=np.float16))
        descriptor = np.ascontiguousarray(
            np.asarray(encoded["spatial_descriptor"], dtype=np.float32)
        )
        if raw.shape != (768, 1024) or descriptor.shape != (768, 1024):
            raise ExperimentError("singleton latent shape drift")
        if (
            encoded["checkpoint_sha256"] != FROZEN_ENCODER_SHA256
            or int(encoded["batch_size"]) != 1
            or not np.isfinite(raw).all()
            or not np.isfinite(descriptor).all()
        ):
            raise ExperimentError("singleton frozen-encoder contract drift")
        raw_rows.append(raw); descriptor_rows.append(descriptor)
        receipts.append({
            "canonical_pixel_index": index,
            "pixel_sha256": pixel_sha,
            "preprocessed_tensor_sha256": str(encoded["preprocessed_tensor_sha256"]),
            "raw_token_sha256": canonical_array_sha256(raw),
            "spatial_descriptor_sha256": canonical_array_sha256(descriptor),
        })
    atomic_npz(
        OUTPUT_ROOT / "canonical_latents.npz",
        raw_tokens=np.ascontiguousarray(np.stack(raw_rows)).astype(np.float16),
        spatial_descriptors=np.ascontiguousarray(np.stack(descriptor_rows)).astype(np.float32),
    )
    latent = attach_digest({
        "schema": "physical_graph_edge_handoff_qualification_v1.latent_index.v1",
        "experiment_id": EXPERIMENT_ID,
        "latents_file": _file_binding(OUTPUT_ROOT / "canonical_latents.npz"),
        "encoder_binding": copy.deepcopy(CONTRACT.VJEPA_ENCODER_BINDING),
        "preprocessing_authority": copy.deepcopy(
            CONTRACT.CANONICAL_ENCODING_AUTHORITY["preprocessing"]
        ),
        "external_encoder_source": external_source,
        "encoder_runtime_environment": encoder_runtime,
        "records": [
            {
                "canonical_pixel_index": row["canonical_pixel_index"],
                "pixel_sha256": row["pixel_sha256"],
                "raw_token_row_index": row["canonical_pixel_index"],
                "raw_token_sha256": row["raw_token_sha256"],
                "spatial_descriptor_row_index": row["canonical_pixel_index"],
                "spatial_descriptor_sha256": row["spatial_descriptor_sha256"],
                "preprocessed_tensor_sha256": row["preprocessed_tensor_sha256"],
            }
            for row in receipts
        ],
    })
    atomic_json(OUTPUT_ROOT / "latent_index.json", latent)
    atomic_json(
        MATERIAL_ROOT / "encoding_receipt.json",
        attach_digest({
            "schema": "physical_graph_edge_handoff_qualification_v1.encoding_receipt.v1",
            "experiment_id": EXPERIMENT_ID,
            "singleton_count": len(order),
            "pixel_order": order,
            "records": receipts,
            "preprocessing_authority": copy.deepcopy(
                CONTRACT.CANONICAL_ENCODING_AUTHORITY["preprocessing"]
            ),
            "external_encoder_source": external_source,
            "encoder_runtime_environment": encoder_runtime,
            "fanout_outcomes_opened": 0,
        }),
    )
    del panel
    return latent


def _consecutive_beyond_samples(
    poses: Any, crossing_after: int, opening: Sequence[Sequence[float]],
    normal: Sequence[float], target_member: Any,
) -> tuple[int, bool]:
    import numpy as np

    midpoint = np.asarray(opening, dtype=np.float64).mean(axis=0)
    direction = np.asarray(normal, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    beyond = (np.asarray(poses, dtype=np.float64)[crossing_after:, :2] - midpoint) @ direction
    epsilon = float(CONTRACT.NUMERICAL_TOLERANCES["se2_position_m"])
    count = 0
    for value in beyond:
        if float(value) <= epsilon:
            break
        count += 1
    target = np.asarray(target_member, dtype=np.uint8)[crossing_after:]
    hits = np.flatnonzero(target != 0)
    target_early = bool(len(hits) and int(hits[0]) + 1 < 100)
    return int(count), target_early


def _command_tracking_rows(
    trace: Mapping[str, Any], requested_commands: Sequence[Sequence[float]],
    applied_commands: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    import numpy as np

    rows: list[dict[str, Any]] = []
    threshold = float(CONTRACT.COMMAND_TRACKING_AUTHORITY["active_command_threshold"])
    for tick in range(15):
        start = tick * 50 + 20
        stop = tick * 50 + 50
        values: list[list[float]] = []
        for index in range(start, stop):
            yaw = _pose_yaw_xyzw(trace["base_pose_world"][index])
            vxw, vyw = (float(value) for value in trace["base_twist_world"][index, :2])
            values.append([
                math.cos(yaw) * vxw + math.sin(yaw) * vyw,
                -math.sin(yaw) * vxw + math.cos(yaw) * vyw,
                float(trace["base_twist_world"][index, 5]),
            ])
        mean = np.asarray(values, dtype=np.float64).mean(axis=0)
        applied = [float(value) for value in applied_commands[tick]]
        rows.append({
            "command_tick_index": tick,
            "requested_command": [float(value) for value in requested_commands[tick]],
            "post_slew_command": applied,
            "mean_achieved_body_velocity": [float(value) for value in mean],
            "active_vx": abs(applied[0]) > threshold,
            "active_yaw": abs(applied[2]) > threshold,
        })
    return rows


def derive_candidate_outcome(
    spec: Mapping[str, Any], trace: Mapping[str, Any],
    reset_pose_world: Sequence[float], candidate_index: int,
) -> dict[str, Any]:
    """Derive every physical label from the 750 persisted 2-ms samples."""

    import numpy as np

    trace = _normalise_trace(trace, kind="candidate", exact_samples=PHYSICS_STEPS_PER_BRANCH)
    geometry = spec["geometry"]
    selected_edge = geometry["selected_directed_edge"]
    competitors = geometry["competing_directed_edges"]
    correct_crossing = _first_positive_port_crossing(
        trace["base_pose_world"], selected_edge["opening_segment_world"],
        selected_edge["opening_normal_world"],
    )
    wrong_crossing = _first_competing_crossing(trace["base_pose_world"], competitors)
    registered_crossing = _first_registered_port_crossing(
        trace["base_pose_world"], selected_edge, competitors
    )
    correct_first = bool(
        registered_crossing is not None
        and bool(registered_crossing["is_selected_edge"])
    )
    dwell = 0
    target_early = False
    if correct_first:
        dwell, target_early = _consecutive_beyond_samples(
            trace["base_pose_world"], int(correct_crossing["sample_after"]),
            selected_edge["opening_segment_world"], selected_edge["opening_normal_world"],
            trace["target_region_member"],
        )
    entered_correct = bool(correct_first and (dwell >= 100 or target_early))
    entered_wrong = bool(
        not entered_correct
        and registered_crossing is not None
        and not bool(registered_crossing["is_selected_edge"])
    )
    start_se2 = [float(reset_pose_world[0]), float(reset_pose_world[1]), _pose_yaw_xyzw(reset_pose_world)]
    endpoints_world: list[list[float]] = []
    for sample in (249, 499, 749):
        pose = trace["base_pose_world"][sample]
        endpoints_world.append([float(pose[0]), float(pose[1]), _pose_yaw_xyzw(pose)])
    endpoints_body = [body_from_world(value, start_se2) for value in endpoints_world]
    opening = np.asarray(selected_edge["opening_segment_world"], dtype=np.float64)
    midpoint = opening.mean(axis=0)
    normal = np.asarray(selected_edge["opening_normal_world"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tangent = opening[1] - opening[0]; tangent /= np.linalg.norm(tangent)
    positions = trace["base_pose_world"][:, :2]
    initial_distance = float(np.linalg.norm(np.asarray(start_se2[:2]) - midpoint))
    port_progress = initial_distance - float(np.linalg.norm(positions - midpoint[None, :], axis=1).min())
    endpoint_xy = positions[-1]
    lateral_error = abs(float((endpoint_xy - midpoint) @ tangent))
    target_heading = math.atan2(float(normal[1]), float(normal[0]))
    angular_error = abs(_wrap_angle(endpoints_world[-1][2] - target_heading))
    final_quat_xyzw = trace["base_pose_world"][-1, 3:7]
    roll, pitch, _yaw = _quat_wxyz_to_rpy([
        final_quat_xyzw[3], final_quat_xyzw[0], final_quat_xyzw[1], final_quat_xyzw[2]
    ])
    finite = bool(all(np.isfinite(value).all() for value in trace.values()))
    physics_contact = bool(trace["physics_contact"].any())
    final_contact = bool(trace["physics_contact"][-1])
    successor = bool(
        finite
        and float(trace["base_pose_world"][-1, 2])
        >= float(CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["minimum_base_height_m"])
        and abs(roll) <= float(CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["maximum_absolute_roll_rad"])
        and abs(pitch) <= float(CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["maximum_absolute_pitch_rad"])
        and not final_contact
    )
    requested_commands = _candidate_requested_commands(candidate_index)
    applied_commands = [
        [float(value) for value in trace["post_slew_applied_command"][tick * 50]]
        for tick in range(15)
    ]
    activity = max(max(abs(value) for value in command) for command in requested_commands)
    stuck = bool(
        activity > float(CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["command_activity_threshold"])
        and math.hypot(endpoints_body[-1][0], endpoints_body[-1][1])
        < float(CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["h3_translation_threshold_m"])
        and abs(endpoints_body[-1][2])
        < float(CONTRACT.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["h3_heading_threshold_rad"])
    )
    oracle_admissible = bool(not physics_contact and finite and successor and not stuck)
    if trace["target_region_member"][-1]:
        endpoint_node_id, endpoint_edge_id = "target", "selected-edge"
    elif entered_correct or trace["correct_edge_region_member"][-1]:
        endpoint_node_id, endpoint_edge_id = None, "selected-edge"
    elif entered_wrong:
        endpoint_node_id, endpoint_edge_id = None, str(wrong_crossing["edge_id"])
    elif trace["source_region_member"][-1]:
        endpoint_node_id, endpoint_edge_id = "source", None
    else:
        endpoint_node_id, endpoint_edge_id = None, None
    correct_evidence = correct_crossing if entered_correct else None
    wrong_evidence = wrong_crossing if entered_wrong else None
    source_exits = np.flatnonzero(np.asarray(trace["source_region_member"], dtype=np.uint8) == 0)
    left_source = bool(len(source_exits))
    target_entries = np.flatnonzero(np.asarray(trace["target_region_member"], dtype=np.uint8) != 0)
    reached_target = bool(len(target_entries))

    def crossing_direction(evidence: Mapping[str, Any] | None) -> tuple[list[float] | None, float | None]:
        if evidence is None:
            return None, None
        before = int(evidence["sample_before"])
        after = int(evidence["sample_after"])
        displacement = (
            np.asarray(trace["base_pose_world"][after, :2], dtype=np.float64)
            - np.asarray(trace["base_pose_world"][before, :2], dtype=np.float64)
        )
        values = [float(value) for value in displacement]
        return values, math.atan2(values[1], values[0])

    correct_displacement, correct_direction = crossing_direction(correct_evidence)
    wrong_displacement, wrong_direction = crossing_direction(wrong_evidence)
    return {
        "requested_commands": requested_commands,
        "post_slew_applied_commands": applied_commands,
        "physics_sample_count": 750,
        "port_crossing_sample_before": None if correct_evidence is None else int(correct_evidence["sample_before"]),
        "port_crossing_sample_after": None if correct_evidence is None else int(correct_evidence["sample_after"]),
        "port_crossing_fraction": None if correct_evidence is None else float(correct_evidence["fraction"]),
        "port_crossing_directed_normal_dot": None if correct_evidence is None else float(correct_evidence["normal_dot_displacement_m"]),
        "port_crossing_lateral_fraction": None if correct_evidence is None else float(correct_evidence["lateral_fraction"]),
        "port_crossing_displacement_world_xy": correct_displacement,
        "port_crossing_direction_heading_world_rad": correct_direction,
        "beyond_port_consecutive_physics_samples": int(dwell),
        "target_entered_before_dwell_complete": bool(target_early),
        "competing_port_crossing_first": bool(entered_wrong),
        "first_wrong_edge_id": None if wrong_evidence is None else str(wrong_evidence["edge_id"]),
        "wrong_port_crossing_sample_before": None if wrong_evidence is None else int(wrong_evidence["sample_before"]),
        "wrong_port_crossing_sample_after": None if wrong_evidence is None else int(wrong_evidence["sample_after"]),
        "wrong_port_crossing_fraction": None if wrong_evidence is None else float(wrong_evidence["fraction"]),
        "wrong_port_crossing_directed_normal_dot": None if wrong_evidence is None else float(wrong_evidence["normal_dot_displacement_m"]),
        "wrong_port_crossing_lateral_fraction": None if wrong_evidence is None else float(wrong_evidence["lateral_fraction"]),
        "wrong_port_crossing_displacement_world_xy": wrong_displacement,
        "wrong_port_crossing_direction_heading_world_rad": wrong_direction,
        "h1_endpoint_body": endpoints_body[0], "h2_endpoint_body": endpoints_body[1],
        "h3_endpoint_body": endpoints_body[2], "h3_endpoint_world": endpoints_world[2],
        "h3_base_height_m": float(trace["base_pose_world"][-1, 2]),
        "h3_roll_rad": roll, "h3_pitch_rad": pitch, "h3_solver_finite": finite,
        "h3_disallowed_contact": final_contact,
        "physics_contact": physics_contact, "stuck": stuck, "successor_viable": successor,
        "entered_correct_edge": entered_correct, "entered_wrong_edge": entered_wrong,
        "no_edge": not entered_correct and not entered_wrong,
        "port_progress_m": port_progress, "lateral_error_m": lateral_error,
        "angular_error_rad": angular_error, "oracle_admissible": oracle_admissible,
        "positive_port_progress": port_progress > 0.0,
        "endpoint_node_id": endpoint_node_id, "endpoint_edge_id": endpoint_edge_id,
        "left_source_region": left_source,
        "first_source_exit_sample_index": int(source_exits[0]) if left_source else None,
        "reached_target_node": reached_target,
        "first_target_entry_sample_index": int(target_entries[0]) if reached_target else None,
        "command_tracking_rows": _command_tracking_rows(trace, requested_commands, applied_commands),
    }


def _panel_material() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    panel = load_json(OUTPUT_ROOT / "panel_manifest.json")
    if len(panel.get("states", ())) != STATE_COUNT:
        raise ExperimentError("frozen panel is incomplete")
    specs = {
        str(row["state_id"]): dict(row)
        for row in _prospective_pool_specs()
        if str(row["state_id"]) in {str(state["state_id"]) for state in panel["states"]}
    }
    if len(specs) != STATE_COUNT:
        raise ExperimentError("frozen panel cannot be rebound to prospective geometry")
    return [dict(row) for row in panel["states"]], specs


def _panel_physical_runtime_core() -> tuple[dict[str, Any], str]:
    panel = load_json(OUTPUT_ROOT / "panel_manifest.json")
    environment = METRICS.validate_physical_runtime_environment(
        panel.get("physical_runtime_environment")
    )
    core = {
        field: copy.deepcopy(environment[field])
        for field in CONTRACT.PHYSICAL_RUNTIME_CORE_FIELDS
    }
    digest = METRICS.runtime_environment_sha256(core)
    if digest != environment["runtime_core_sha256"]:
        raise ExperimentError("panel physical runtime digest drift")
    return core, digest


def fanout_state_stage(
    state_id: str, *, backend: Any | None = None, fake_runtime: bool = False,
) -> dict[str, Any]:
    """Execute the exact twelve branches for one frozen state.

    Development-held-out state fanout is inaccessible until the three-way
    development target choice has been frozen on disk.
    """

    runtime = require_stage_runtime("physical", fake=fake_runtime)
    _require_initialized()
    states, specs = _panel_material()
    state_by_id = {str(row["state_id"]): row for row in states}
    if str(state_id) not in state_by_id:
        raise ExperimentError("fanout requested for state outside frozen panel")
    state = state_by_id[str(state_id)]
    selection_path = OUTPUT_ROOT / "development_target_selection.json"
    if state["role"] == "DEVELOPMENT_HELDOUT":
        if not selection_path.is_file():
            raise ExperimentError("held-out fanout opened before target selection freeze")
        selection = load_json(selection_path)
        if selection.get("selection_frozen") is not True:
            raise ExperimentError("held-out fanout opened before target selection freeze")
    elif selection_path.exists():
        # Development fanout is complete before selection.  Reopening one
        # afterwards would make the frozen selection evidence mutable.
        raise ExperimentError("development fanout cannot be changed after target selection")
    selected_metadata, selected_arrays = _load_material_shard(
        _selected_directory(str(state_id))
    )
    snapshot_payload = bytes(
        selected_arrays["snapshot_payload_bytes"].astype("uint8").tobytes()
    )
    snapshot_sha = hashlib.sha256(snapshot_payload).hexdigest()
    if snapshot_sha != selected_metadata["snapshot_payload_sha256"]:
        raise ExperimentError("fanout snapshot binding drift")
    collector = _qualification_backend_default() if backend is None else backend
    raw_rows = list(collector.fanout(copy.deepcopy(specs[str(state_id)]), snapshot_payload))
    if len(raw_rows) != CANDIDATE_COUNT:
        raise ExperimentError("physical fanout does not contain twelve branches")
    arrays: dict[str, Any] = {}
    outcome_rows: list[dict[str, Any]] = []
    reset_pose = selected_arrays["snapshot__base_pose_world"]
    backend_runtime: dict[str, Any] | None = None
    for candidate_index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "candidate_index", "snapshot_payload_sha256", "trace", "runtime_evidence",
        }:
            raise ExperimentError("physical fanout backend row field drift")
        if (
            int(raw["candidate_index"]) != candidate_index
            or raw["snapshot_payload_sha256"] != snapshot_sha
        ):
            raise ExperimentError("physical fanout candidate/snapshot order drift")
        trace = _normalise_trace(
            dict(raw["trace"]), kind="candidate", exact_samples=PHYSICS_STEPS_PER_BRANCH
        )
        for member, value in trace.items():
            arrays[f"candidate_{candidate_index:02d}__{member}"] = value
        outcome_rows.append(
            derive_candidate_outcome(
                specs[str(state_id)], trace, reset_pose, candidate_index
            )
        )
        runtime_row = dict(raw["runtime_evidence"])
        if backend_runtime is None:
            backend_runtime = runtime_row
        elif backend_runtime != runtime_row:
            raise ExperimentError("backend runtime changed within one fanout")
    assert backend_runtime is not None
    runtime = _bind_physical_backend_runtime(runtime, backend_runtime)
    panel_runtime, physical_runtime_core_sha256 = _panel_physical_runtime_core()
    if runtime != panel_runtime:
        raise ExperimentError("fanout physical runtime differs from frozen panel")
    metadata = {
        "schema": "physical_graph_edge_handoff_qualification_v1.fanout_state_shard.v1",
        "experiment_id": EXPERIMENT_ID,
        "state_id": str(state_id),
        "role": str(state["role"]),
        "family": str(state["family"]),
        "candidate_spec_id": str(state["candidate_spec_id"]),
        "snapshot_id": f"{CONTRACT.SNAPSHOT_ID_PREFIX}{state_id}",
        "snapshot_payload_sha256": snapshot_sha,
        "outcome_rows": outcome_rows,
        "stage_runtime": runtime,
        "backend_runtime": backend_runtime,
        "physical_runtime_core_sha256": physical_runtime_core_sha256,
        "target_selection_sha256": (
            sha256_file(selection_path) if state["role"] == "DEVELOPMENT_HELDOUT" else None
        ),
    }
    _write_material_shard(MATERIAL_ROOT / "fanout" / str(state_id), metadata, arrays)
    return metadata


class _FrozenPhysicalRanker:
    """Exact V2 frozen ranker with corrected bearing feature semantics."""

    def __init__(self) -> None:
        from scripts.run_occluded_goal_topological_belief_v1 import (
            FrozenCurrentVisualLocalRanker,
        )

        self._ranker = FrozenCurrentVisualLocalRanker()
        if self._ranker.binding["checkpoint_sha256"] != FROZEN_RANKER_SHA256:
            raise ExperimentError("frozen physical ranker checkpoint drift")

    def score(
        self, current_tokens: Any, waypoint: Mapping[str, Any],
        previous_command: Sequence[float], control_history: Sequence[Sequence[float]],
    ) -> list[float]:
        # The third legacy input is the body-frame target bearing.  Port or
        # route tangent yaw is deliberately retained only as audit evidence.
        relative = [
            float(waypoint["dx_m"]), float(waypoint["dy_m"]),
            math.atan2(float(waypoint["dy_m"]), float(waypoint["dx_m"])),
        ]
        result = self._ranker(
            current_tokens, relative, previous_command=previous_command,
            control_history=control_history,
        )
        if list(result["candidate_names"]) != list(CANDIDATE_IDS):
            raise ExperimentError("frozen ranker candidate order drift")
        return [float(value) for value in result["candidate_scores"]]


def _rank_scores(scores: Sequence[float]) -> list[int]:
    values = [float(value) for value in scores]
    if len(values) != CANDIDATE_COUNT or not all(math.isfinite(value) for value in values):
        raise ExperimentError("candidate score vector drift")
    return sorted(range(CANDIDATE_COUNT), key=lambda index: (-values[index], index))


def _correct_candidate(row: Mapping[str, Any]) -> bool:
    return bool(
        row["oracle_admissible"] and row["entered_correct_edge"]
        and row["successor_viable"] and row["positive_port_progress"]
        and not row["physics_contact"] and not row["stuck"]
    )


def _oracle_ranking(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted(
        range(CANDIDATE_COUNT),
        key=lambda index: (
            0 if _correct_candidate(rows[index]) else 1,
            -float(rows[index]["port_progress_m"]),
            float(rows[index]["lateral_error_m"]),
            float(rows[index]["angular_error_rad"]), index,
        ),
    )


def _kinematic_scores(
    rows: Sequence[Mapping[str, Any]], target_body: Sequence[float],
) -> list[float]:
    start_distance = math.hypot(float(target_body[0]), float(target_body[1]))
    scores: list[float] = []
    for row in rows:
        x = y = yaw = 0.0
        for vx, vy, yaw_rate in row["post_slew_applied_commands"]:
            x += 0.1 * (float(vx) * math.cos(yaw) - float(vy) * math.sin(yaw))
            y += 0.1 * (float(vx) * math.sin(yaw) + float(vy) * math.cos(yaw))
            yaw = _wrap_angle(yaw + 0.1 * float(yaw_rate))
        scores.append(
            start_distance
            - math.hypot(float(target_body[0]) - x, float(target_body[1]) - y)
        )
    return scores


def _selection_metrics(
    rows: Sequence[Mapping[str, Any]], ranking: Sequence[int],
    scores: Sequence[float] | None,
) -> dict[str, Any]:
    ranking = [int(value) for value in ranking]
    selected = ranking[0]
    correct = [index for index, row in enumerate(rows) if _correct_candidate(row)]
    if not any(row["oracle_admissible"] for row in rows):
        raise ExperimentError("fanout state has no oracle-admissible candidate")
    first_rank = next((rank for rank, index in enumerate(ranking, 1) if index in correct), None)
    incorrect = [index for index in range(CANDIDATE_COUNT) if index not in correct]
    positions = {index: rank for rank, index in enumerate(ranking)}
    credits: list[float] = []
    for left in correct:
        for right in incorrect:
            if scores is None:
                credits.append(float(positions[left] < positions[right]))
            else:
                credits.append(
                    1.0 if float(scores[left]) > float(scores[right])
                    else 0.5 if float(scores[left]) == float(scores[right]) else 0.0
                )
    admissible_progress = [
        float(row["port_progress_m"]) for row in rows if row["oracle_admissible"]
    ]
    best, minimum = max(admissible_progress), min(admissible_progress)
    selected_row = rows[selected]
    span = best - minimum
    regret = 0.0 if span <= float(CONTRACT.NUMERICAL_TOLERANCES["division_floor"]) else min(
        1.0, max(0.0, (best - float(selected_row["port_progress_m"])) / span)
    )
    return {
        "candidate_ids": list(CANDIDATE_IDS),
        "scores": None if scores is None else [float(value) for value in scores],
        "ranking": ranking,
        "eligible_correct_edge_candidate_indices": correct,
        "selected_candidate_index": selected,
        "correct_edge_top1": selected in correct,
        "correct_edge_top3": any(value in correct for value in ranking[:3]),
        "correct_edge_mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "selected_correct_edge_execution": bool(selected_row["entered_correct_edge"]),
        "selected_port_progress_m": float(selected_row["port_progress_m"]),
        "oracle_best_port_progress_m": best,
        "minimum_admissible_port_progress_m": minimum,
        "normalized_port_regret": regret,
        "pairwise_correct_edge_ordering": (
            math.fsum(credits) / len(credits) if credits else 0.0
        ),
        "selected_wrong_edge": bool(selected_row["entered_wrong_edge"]),
        "selected_no_edge": bool(selected_row["no_edge"]),
        "selected_lateral_error_m": float(selected_row["lateral_error_m"]),
        "selected_angular_error_rad": float(selected_row["angular_error_rad"]),
        "selected_contact": bool(selected_row["physics_contact"]),
        "selected_stuck": bool(selected_row["stuck"]),
        "selected_successor_viable": bool(selected_row["successor_viable"]),
    }


def _load_state_ranker_material(
    state_id: str,
) -> tuple[Any, list[float], list[list[float]]]:
    """Load exact persisted current tokens and predecessor controller state."""

    import numpy as np

    pixel = load_json(OUTPUT_ROOT / "pixel_index.json")
    latent = load_json(OUTPUT_ROOT / "latent_index.json")
    pixel_row = next(row for row in pixel["records"] if row["state_id"] == state_id)
    canonical = int(pixel_row["canonical_pixel_index"])
    latent_row = next(
        row for row in latent["records"] if row["canonical_pixel_index"] == canonical
    )
    with np.load(OUTPUT_ROOT / "canonical_latents.npz", allow_pickle=False) as data:
        tokens = np.ascontiguousarray(data["raw_tokens"][int(latent_row["raw_token_row_index"])]).copy()
    states, _specs = _panel_material()
    row_index = next(index for index, state in enumerate(states) if state["state_id"] == state_id)
    with np.load(OUTPUT_ROOT / "state_snapshots.npz", allow_pickle=False) as data:
        previous = [float(value) for value in data["previous_applied_command"][row_index]]
        history = [[float(value) for value in item] for item in data["control_history"][row_index]]
    return tokens, previous, history


def _load_fanout_outcomes(state_id: str) -> list[dict[str, Any]]:
    metadata, _arrays = _load_material_shard(MATERIAL_ROOT / "fanout" / state_id)
    rows = [dict(value) for value in metadata["outcome_rows"]]
    if len(rows) != CANDIDATE_COUNT:
        raise ExperimentError("fanout state row count drift")
    return rows


def _target_summary(rows: Sequence[Mapping[str, Any]], target_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row["target_id"] == target_id]
    mean = lambda field: math.fsum(float(row[field]) for row in selected) / len(selected)
    result = {
        "target_id": target_id,
        "state_count": len(selected),
        "selected_correct_edge_execution_rate": mean("selected_correct_edge_execution"),
        "correct_edge_top3_rate": mean("correct_edge_top3"),
        "correct_edge_top1_rate": mean("correct_edge_top1"),
        "correct_edge_mrr": mean("correct_edge_mrr"),
        "normalized_port_regret": mean("normalized_port_regret"),
        "mean_selected_port_progress_m": mean("selected_port_progress_m"),
        "mean_target_transform_error_m": mean("target_transform_error_m"),
        "pairwise_correct_edge_ordering": mean("pairwise_correct_edge_ordering"),
        "selected_wrong_edge_rate": mean("selected_wrong_edge"),
        "selected_no_edge_rate": mean("selected_no_edge"),
        "mean_selected_lateral_error_m": mean("selected_lateral_error_m"),
        "mean_selected_angular_error_rad": mean("selected_angular_error_rad"),
        "selected_contact_rate": mean("selected_contact"),
        "selected_stuck_rate": mean("selected_stuck"),
        "selected_successor_viable_rate": mean("selected_successor_viable"),
    }
    result["selection_key"] = [
        -result["selected_correct_edge_execution_rate"],
        -result["correct_edge_top3_rate"], result["normalized_port_regret"],
        -result["mean_selected_port_progress_m"],
        result["mean_target_transform_error_m"], TARGET_IDS.index(target_id),
    ]
    return result


def development_target_selection_stage(
    *, ranker: Any | None = None, fake_runtime: bool = False,
) -> dict[str, Any]:
    """Score three frozen targets on DEVELOPMENT and irrevocably freeze one."""

    ranker_runtime = require_stage_runtime(
        "visual", fake=fake_runtime, visual_role="ranker",
    )
    _require_initialized()
    if (OUTPUT_ROOT / "development_target_selection.json").exists():
        raise ExperimentError("development target selection is already frozen")
    if (OUTPUT_ROOT / "heldout_ranker_scores.jsonl").exists() or any(
        (MATERIAL_ROOT / "fanout" / state_id).exists()
        for state_id in [
            row["state_id"] for row in load_json(OUTPUT_ROOT / "panel_manifest.json")["states"]
            if row["role"] == "DEVELOPMENT_HELDOUT"
        ]
    ):
        raise ExperimentError("held-out outcome exists before target selection")
    states, _specs = _panel_material()
    development = [row for row in states if row["role"] == "DEVELOPMENT"]
    for state in development:
        if not (MATERIAL_ROOT / "fanout" / state["state_id"] / "metadata.json").is_file():
            raise ExperimentError("development target selection lacks complete fanout")
    waypoints = load_json(OUTPUT_ROOT / "waypoint_contracts.json")["rows"]
    waypoint_by_identity = {
        (row["state_id"], row["target_id"]): row for row in waypoints
    }
    runtime_ranker = _FrozenPhysicalRanker() if ranker is None else ranker
    state_target_rows: list[dict[str, Any]] = []
    for state in development:
        state_id = str(state["state_id"])
        tokens, previous, history = _load_state_ranker_material(state_id)
        fanout = _load_fanout_outcomes(state_id)
        for target_id in TARGET_IDS:
            waypoint = waypoint_by_identity[(state_id, target_id)]
            scores = [float(value) for value in runtime_ranker.score(
                tokens, waypoint, previous, history
            )]
            row = _selection_metrics(fanout, _rank_scores(scores), scores)
            row.update({
                "state_id": state_id,
                "target_id": target_id,
                "target_transform_error_m": max(
                    float(waypoint["position_transform_error_m"]),
                    float(waypoint["heading_transform_error_rad"]),
                ),
            })
            state_target_rows.append(row)
    summaries = [_target_summary(state_target_rows, target) for target in TARGET_IDS]
    selected_index = min(
        range(len(summaries)), key=lambda index: tuple(summaries[index]["selection_key"])
    )
    document = attach_digest({
        "schema": "physical_graph_edge_handoff_qualification_v1.development_target_selection.v1",
        "experiment_id": EXPERIMENT_ID,
        "role": "DEVELOPMENT",
        "target_ids": list(TARGET_IDS),
        "selection_lexicographic": list(CONTRACT.DEVELOPMENT_TARGET_SELECTION_LEXICOGRAPHIC),
        "heldout_outcome_documents_opened": 0,
        "state_target_rows": state_target_rows,
        "target_summaries": summaries,
        "selected_target_id": TARGET_IDS[selected_index],
        "selected_target_index": selected_index,
        "selection_frozen": True,
        "ranker_runtime_environment": ranker_runtime,
    })
    # Pure validation before the held-out namespace becomes accessible.
    development_fanout = [
        {"state_id": state["state_id"], "role": state["role"], "family": state["family"], **row}
        for state in development for row in _load_fanout_outcomes(state["state_id"])
    ]
    # The strict public validator expects the full ledger cardinality, so row
    # validation happens again after held-out assembly.  At this boundary the
    # target summaries are derived exclusively from the 48 development states.
    del development_fanout
    atomic_json(OUTPUT_ROOT / "development_target_selection.json", document)
    return document


def _candidate_score_row(
    state: Mapping[str, Any], condition_id: str, target_id: str,
    fanout: Sequence[Mapping[str, Any]], scores: Sequence[float] | None,
    ranking: Sequence[int],
) -> dict[str, Any]:
    values = _selection_metrics(fanout, ranking, scores)
    values.update({
        "score_row_id": f"{state['state_id']}::{condition_id}",
        "state_id": str(state["state_id"]),
        "family": str(state["family"]),
        "condition_id": condition_id,
        "target_id": target_id,
        "teacher_trace_id": None,
        "teacher_correct_execution": None,
        "ranker_checkpoint_sha256": (
            FROZEN_RANKER_SHA256
            if condition_id == "FROZEN_CURRENT_VISUAL_RANKER" else None
        ),
    })
    return values


def heldout_ranker_scores_stage(
    *, ranker: Any | None = None, fake_runtime: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate the one frozen development-selected target on held-out states."""

    ranker_runtime = require_stage_runtime(
        "visual", fake=fake_runtime, visual_role="ranker",
    )
    _require_initialized()
    selection = load_json(OUTPUT_ROOT / "development_target_selection.json")
    if selection.get("selection_frozen") is not True:
        raise ExperimentError("held-out scoring lacks a frozen development target")
    states, _specs = _panel_material()
    heldout = [row for row in states if row["role"] == "DEVELOPMENT_HELDOUT"]
    for state in heldout:
        if not (MATERIAL_ROOT / "fanout" / state["state_id"] / "metadata.json").is_file():
            raise ExperimentError("held-out scoring lacks complete physical fanout")
    waypoint_rows = load_json(OUTPUT_ROOT / "waypoint_contracts.json")["rows"]
    selected_target = str(selection["selected_target_id"])
    if selection.get("ranker_runtime_environment") != ranker_runtime:
        raise ExperimentError("ranker runtime changed after development selection")
    ranker_runtime_sha256 = METRICS.runtime_environment_sha256(ranker_runtime)
    waypoint_by_state = {
        row["state_id"]: row for row in waypoint_rows
        if row["target_id"] == selected_target
    }
    teacher_by_state = {
        row["state_id"]: row
        for row in load_json(OUTPUT_ROOT / "teacher_trace_index.json")["records"]
        if row["selected"]
    }
    runtime_ranker = _FrozenPhysicalRanker() if ranker is None else ranker
    rows: list[dict[str, Any]] = []
    for state in heldout:
        state_id = str(state["state_id"])
        fanout = _load_fanout_outcomes(state_id)
        waypoint = waypoint_by_state[state_id]
        tokens, previous, history = _load_state_ranker_material(state_id)
        ranker_scores = [float(value) for value in runtime_ranker.score(
            tokens, waypoint, previous, history
        )]
        for condition_id in HELDOUT_CONDITIONS:
            if condition_id == "DETERMINISTIC_KINEMATICS":
                scores = _kinematic_scores(fanout, waypoint["target_body_pose"])
                rows.append(_candidate_score_row(
                    state, condition_id, selected_target, fanout, scores,
                    _rank_scores(scores),
                ))
            elif condition_id == "FROZEN_CURRENT_VISUAL_RANKER":
                rows.append(_candidate_score_row(
                    state, condition_id, selected_target, fanout, ranker_scores,
                    _rank_scores(ranker_scores),
                ))
            elif condition_id == "ORACLE_BEST_ADMISSIBLE_CANDIDATE":
                rows.append(_candidate_score_row(
                    state, condition_id, selected_target, fanout, None,
                    _oracle_ranking(fanout),
                ))
            else:
                teacher = teacher_by_state[state_id]
                rows.append({
                    "score_row_id": f"{state_id}::{condition_id}",
                    "state_id": state_id,
                    "family": str(state["family"]),
                    "condition_id": condition_id,
                    "target_id": selected_target,
                    "candidate_ids": [],
                    "scores": None,
                    "ranking": None,
                    "eligible_correct_edge_candidate_indices": [],
                    "selected_candidate_index": None,
                    "correct_edge_top1": None,
                    "correct_edge_top3": None,
                    "correct_edge_mrr": None,
                    "selected_correct_edge_execution": None,
                    "selected_port_progress_m": float(teacher["route_progress_m"]),
                    "oracle_best_port_progress_m": None,
                    "minimum_admissible_port_progress_m": None,
                    "normalized_port_regret": None,
                    "teacher_trace_id": str(teacher["teacher_trace_id"]),
                    "teacher_correct_execution": bool(teacher["teacher_valid"]),
                    "ranker_checkpoint_sha256": None,
                    "pairwise_correct_edge_ordering": None,
                    "selected_wrong_edge": bool(teacher["competing_port_entered"]),
                    "selected_no_edge": bool(
                        not teacher["crossed_directed_port"]
                        and not teacher["competing_port_entered"]
                    ),
                    "selected_lateral_error_m": float(teacher["endpoint_lateral_error_m"]),
                    "selected_angular_error_rad": float(teacher["endpoint_angular_error_rad"]),
                    "selected_contact": not bool(teacher["contact_free"]),
                    "selected_stuck": bool(teacher["stuck"]),
                    "selected_successor_viable": bool(teacher["successor_viable"]),
                })
    if len(rows) != HELDOUT_SCORE_ROW_COUNT:
        raise ExperimentError("held-out score row count drift")
    for row in rows:
        row["ranker_runtime_environment_sha256"] = ranker_runtime_sha256
    atomic_jsonl(OUTPUT_ROOT / "heldout_ranker_scores.jsonl", rows)
    return rows


def repeat_state_stage(
    state_id: str, *, backend: Any | None = None, fake_runtime: bool = False,
) -> dict[str, Any]:
    """Replay ranker-selected and oracle candidates twice from exact restore."""

    runtime = require_stage_runtime("physical", fake=fake_runtime)
    _require_initialized()
    states, specs = _panel_material()
    state = next(
        (row for row in states if row["state_id"] == str(state_id)), None
    )
    if state is None or state["role"] != "DEVELOPMENT_HELDOUT":
        raise ExperimentError("repeat requested outside held-out panel")
    scores = load_jsonl(OUTPUT_ROOT / "heldout_ranker_scores.jsonl")
    score_by_condition = {
        row["condition_id"]: row for row in scores if row["state_id"] == str(state_id)
    }
    selected_indices = [
        int(score_by_condition["FROZEN_CURRENT_VISUAL_RANKER"]["selected_candidate_index"]),
        int(score_by_condition["FROZEN_CURRENT_VISUAL_RANKER"]["selected_candidate_index"]),
        int(score_by_condition["ORACLE_BEST_ADMISSIBLE_CANDIDATE"]["selected_candidate_index"]),
        int(score_by_condition["ORACLE_BEST_ADMISSIBLE_CANDIDATE"]["selected_candidate_index"]),
    ]
    selected_metadata, selected_arrays = _load_material_shard(
        _selected_directory(str(state_id))
    )
    snapshot_payload = bytes(
        selected_arrays["snapshot_payload_bytes"].astype("uint8").tobytes()
    )
    collector = _qualification_backend_default() if backend is None else backend
    raw_rows = list(collector.repeat(
        copy.deepcopy(specs[str(state_id)]), snapshot_payload, selected_indices
    ))
    if len(raw_rows) != 4:
        raise ExperimentError("repeat backend row count drift")
    arrays: dict[str, Any] = {}
    outcomes: list[dict[str, Any]] = []
    backend_runtime: dict[str, Any] | None = None
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "selector_index", "candidate_index", "snapshot_payload_sha256",
            "trace", "runtime_evidence",
        }:
            raise ExperimentError("repeat backend row field drift")
        if (
            int(raw["selector_index"]) != index
            or int(raw["candidate_index"]) != selected_indices[index]
            or raw["snapshot_payload_sha256"] != selected_metadata["snapshot_payload_sha256"]
        ):
            raise ExperimentError("repeat selector/snapshot binding drift")
        trace = _normalise_trace(
            dict(raw["trace"]), kind="candidate", exact_samples=PHYSICS_STEPS_PER_BRANCH
        )
        for member, value in trace.items():
            arrays[f"repeat_{index:02d}__{member}"] = value
        outcomes.append(derive_candidate_outcome(
            specs[str(state_id)], trace,
            selected_arrays["snapshot__base_pose_world"], selected_indices[index],
        ))
        runtime_row = dict(raw["runtime_evidence"])
        if backend_runtime is None:
            backend_runtime = runtime_row
        elif backend_runtime != runtime_row:
            raise ExperimentError("backend runtime changed within repeat stage")
    assert backend_runtime is not None
    runtime = _bind_physical_backend_runtime(runtime, backend_runtime)
    panel_runtime, physical_runtime_core_sha256 = _panel_physical_runtime_core()
    if runtime != panel_runtime:
        raise ExperimentError("repeat physical runtime differs from frozen panel")
    metadata = {
        "schema": "physical_graph_edge_handoff_qualification_v1.repeat_state_shard.v1",
        "experiment_id": EXPERIMENT_ID,
        "state_id": str(state_id),
        "family": str(state["family"]),
        "snapshot_id": f"{CONTRACT.SNAPSHOT_ID_PREFIX}{state_id}",
        "snapshot_payload_sha256": selected_metadata["snapshot_payload_sha256"],
        "selected_candidate_indices": selected_indices,
        "outcome_rows": outcomes,
        "heldout_scores_sha256": sha256_file(OUTPUT_ROOT / "heldout_ranker_scores.jsonl"),
        "stage_runtime": runtime,
        "backend_runtime": backend_runtime,
        "physical_runtime_core_sha256": physical_runtime_core_sha256,
    }
    _write_material_shard(MATERIAL_ROOT / "repeat" / str(state_id), metadata, arrays)
    return metadata


def _candidate_trace_binding(
    candidate_path: Path, trace: Mapping[str, Any], trace_index: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    start = int(trace_index) * PHYSICS_STEPS_PER_BRANCH
    stop = start + PHYSICS_STEPS_PER_BRANCH
    return (
        _slice_binding(candidate_path, "timestamp_s", start, stop, trace["timestamp_s"]),
        _trace_digest_projection(trace),
    )


def assemble_row_evidence_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    """Consolidate the 960 physical traces and all row-level bindings."""

    import numpy as np

    require_stage_runtime("ordinary", fake=fake_runtime)
    _require_initialized()
    states, _specs = _panel_material()
    _panel_runtime_core, physical_runtime_core_sha256 = (
        _panel_physical_runtime_core()
    )
    heldout = [row for row in states if row["role"] == "DEVELOPMENT_HELDOUT"]
    if len(load_jsonl(OUTPUT_ROOT / "heldout_ranker_scores.jsonl")) != HELDOUT_SCORE_ROW_COUNT:
        raise ExperimentError("row assembly lacks complete held-out scores")
    for state in states:
        if not (MATERIAL_ROOT / "fanout" / state["state_id"] / "metadata.json").is_file():
            raise ExperimentError("row assembly lacks a fanout shard")
    for state in heldout:
        if not (MATERIAL_ROOT / "repeat" / state["state_id"] / "metadata.json").is_file():
            raise ExperimentError("row assembly lacks a repeat shard")

    traces: list[dict[str, Any]] = []
    selected_shards: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    fanout_shards: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    repeat_shards: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for state in states:
        state_id = str(state["state_id"])
        selected_shards[state_id] = _load_material_shard(_selected_directory(state_id))
        metadata, arrays = selected_shards[state_id]
        for trial in range(2):
            traces.append({
                member: arrays[f"fixture_{trial}__{member}"]
                for member in CANDIDATE_TRACE_MEMBERS
            })
    for state in states:
        state_id = str(state["state_id"])
        fanout_shards[state_id] = _load_material_shard(
            MATERIAL_ROOT / "fanout" / state_id
        )
        _metadata, arrays = fanout_shards[state_id]
        for candidate in range(CANDIDATE_COUNT):
            traces.append({
                member: arrays[f"candidate_{candidate:02d}__{member}"]
                for member in CANDIDATE_TRACE_MEMBERS
            })
    for state in heldout:
        state_id = str(state["state_id"])
        repeat_shards[state_id] = _load_material_shard(
            MATERIAL_ROOT / "repeat" / state_id
        )
        _metadata, arrays = repeat_shards[state_id]
        for repeat_index in range(4):
            traces.append({
                member: arrays[f"repeat_{repeat_index:02d}__{member}"]
                for member in CANDIDATE_TRACE_MEMBERS
            })
    if len(traces) != CANDIDATE_TRACE_COUNT:
        raise ExperimentError("candidate trace consolidation cardinality drift")
    candidate_arrays, spans = _concat_traces(traces, CANDIDATE_TRACE_MEMBERS)
    if any(stop - start != PHYSICS_STEPS_PER_BRANCH for start, stop in spans):
        raise ExperimentError("candidate trace consolidation duration drift")
    candidate_path = OUTPUT_ROOT / "candidate_traces.npz"
    atomic_npz(candidate_path, **candidate_arrays)

    # Snapshot index, including the two adjacent restore traces per state.
    snapshot_path = OUTPUT_ROOT / "state_snapshots.npz"
    with np.load(snapshot_path, allow_pickle=False) as data:
        snapshot_npz = {name: np.ascontiguousarray(data[name]) for name in data.files}
    snapshot_file_sha = sha256_file(snapshot_path)
    snapshot_rows: list[dict[str, Any]] = []
    for state_index, state in enumerate(states):
        state_id = str(state["state_id"])
        metadata, arrays = selected_shards[state_id]
        start = int(snapshot_npz["snapshot_offsets"][state_index])
        stop = int(snapshot_npz["snapshot_offsets"][state_index + 1])
        payload_slice = snapshot_npz["snapshot_payload_bytes"][start:stop]
        trial_rows: list[dict[str, Any]] = []
        for trial_index, material_trial in enumerate(metadata["reset_trials"]):
            trace_index = 2 * state_index + trial_index
            trace = traces[trace_index]
            trace_slice, digests = _candidate_trace_binding(
                candidate_path, trace, trace_index
            )
            trial_rows.append({
                "trial_index": trial_index,
                "serialized_restore_used": bool(material_trial["serialized_restore_used"]),
                "restored_snapshot_sha256": str(material_trial["restored_snapshot_sha256"]),
                "post_restore_state_sha256": str(material_trial["post_restore_state_sha256"]),
                "current_rgb_sha256": str(material_trial["current_rgb_sha256"]),
                "base_pose_world": [float(value) for value in material_trial["base_pose_world"]],
                "joint_position_sha256": str(material_trial["joint_position_sha256"]),
                "joint_velocity_sha256": str(material_trial["joint_velocity_sha256"]),
                "controller_state_sha256": str(material_trial["controller_state_sha256"]),
                "rng_state_sha256": str(material_trial["rng_state_sha256"]),
                "trace_index": trace_index,
                "trace_slice": trace_slice,
                "trace_array_slice_sha256s": digests,
                "requested_command_sequence_sha256": str(material_trial["requested_command_sequence_sha256"]),
                "post_slew_applied_command_sequence_sha256": str(material_trial["post_slew_applied_command_sequence_sha256"]),
                "contact_sequence_sha256": str(material_trial["contact_sequence_sha256"]),
                "termination_reason": str(material_trial["termination_reason"]),
                "stuck": bool(material_trial["stuck"]),
            })
        snapshot_meta = dict(metadata["snapshot"])
        snapshot_rows.append({
            "state_id": state_id,
            "snapshot_id": str(metadata.get("snapshot_id", f"{CONTRACT.SNAPSHOT_ID_PREFIX}{state_id}")),
            "snapshot_payload": {
                "file_path": "state_snapshots.npz",
                "file_sha256": snapshot_file_sha,
                "member": "snapshot_payload_bytes",
                "start": start, "stop": stop,
                "slice_sha256": hashlib.sha256(payload_slice.tobytes(order="C")).hexdigest(),
            },
            "array_row_index": state_index,
            "source_teacher_trace_id": str(metadata["source_teacher_trace_id"]),
            "teacher_initial_state_sha256": str(metadata["teacher_initial_state_sha256"]),
            **{
                field: copy.deepcopy(snapshot_meta[field])
                for field in SNAPSHOT_METADATA_FIELDS
            },
            "base_pose_world_sha256": canonical_array_sha256(snapshot_npz["base_pose_world"][state_index]),
            "base_twist_world_sha256": canonical_array_sha256(snapshot_npz["base_twist_world"][state_index]),
            "joint_position_sha256": canonical_array_sha256(snapshot_npz["joint_position"][state_index]),
            "joint_velocity_sha256": canonical_array_sha256(snapshot_npz["joint_velocity"][state_index]),
            "camera_world_transform_sha256": canonical_array_sha256(snapshot_npz["camera_world_transform"][state_index]),
            "reset_trials": trial_rows,
            "reset_pair_comparison": copy.deepcopy(metadata["reset_pair_comparison"]),
        })
    snapshot_index = attach_digest({
        "schema": "physical_graph_edge_handoff_qualification_v1.state_snapshot_index.v1",
        "experiment_id": EXPERIMENT_ID,
        "snapshots_file": _file_binding(snapshot_path),
        "reset_fixture": {
            "fixture_id": "SERIALIZED_RESTORE_TWO_TRIAL_V1",
            "serialized_restore_used": True,
            "clone_equivalence_used": False,
            "passed": True,
        },
        "records": snapshot_rows,
    })

    fanout_rows: list[dict[str, Any]] = []
    for state_index, state in enumerate(states):
        state_id = str(state["state_id"])
        metadata, arrays = fanout_shards[state_id]
        if metadata.get("physical_runtime_core_sha256") != physical_runtime_core_sha256:
            raise ExperimentError("fanout shard physical runtime binding drift")
        for candidate_index, outcome in enumerate(metadata["outcome_rows"]):
            trace_index = RESET_FIXTURE_TRACE_COUNT + state_index * CANDIDATE_COUNT + candidate_index
            trace = {
                member: arrays[f"candidate_{candidate_index:02d}__{member}"]
                for member in CANDIDATE_TRACE_MEMBERS
            }
            trace_slice, digests = _candidate_trace_binding(
                candidate_path, trace, trace_index
            )
            fanout_rows.append({
                "branch_id": f"{state_id}::{CANDIDATE_IDS[candidate_index]}",
                "state_id": state_id,
                "role": str(state["role"]),
                "family": str(state["family"]),
                "candidate_index": candidate_index,
                "candidate_id": CANDIDATE_IDS[candidate_index],
                "snapshot_id": f"{CONTRACT.SNAPSHOT_ID_PREFIX}{state_id}",
                "restored_snapshot_sha256": str(metadata["snapshot_payload_sha256"]),
                "trace_index": trace_index,
                "trace_slice": trace_slice,
                "trace_array_slice_sha256s": digests,
                "physical_runtime_core_sha256": physical_runtime_core_sha256,
                **copy.deepcopy(dict(outcome)),
            })
    if len(fanout_rows) != FANOUT_ROW_COUNT:
        raise ExperimentError("candidate fanout ledger cardinality drift")
    atomic_jsonl(OUTPUT_ROOT / "candidate_fanout.jsonl", fanout_rows)

    heldout_scores = load_jsonl(OUTPUT_ROOT / "heldout_ranker_scores.jsonl")
    score_by_identity = {
        (row["state_id"], row["condition_id"]): row for row in heldout_scores
    }
    fanout_by_identity = {
        (row["state_id"], row["candidate_index"]): row for row in fanout_rows
    }
    repeat_rows: list[dict[str, Any]] = []
    repeat_trace_base = RESET_FIXTURE_TRACE_COUNT + FANOUT_ROW_COUNT
    global_repeat_index = 0
    selector_ids = list(CONTRACT.REPEAT_BRANCH_IDS)
    for state in heldout:
        state_id = str(state["state_id"])
        metadata, arrays = repeat_shards[state_id]
        if metadata.get("physical_runtime_core_sha256") != physical_runtime_core_sha256:
            raise ExperimentError("repeat shard physical runtime binding drift")
        for selector_index, selector_id in enumerate(selector_ids):
            condition = (
                "FROZEN_CURRENT_VISUAL_RANKER" if selector_index == 0
                else "ORACLE_BEST_ADMISSIBLE_CANDIDATE"
            )
            candidate_index = int(
                score_by_identity[(state_id, condition)]["selected_candidate_index"]
            )
            source = fanout_by_identity[(state_id, candidate_index)]
            for repeat_index in range(2):
                within_state = selector_index * 2 + repeat_index
                outcome = dict(metadata["outcome_rows"][within_state])
                trace = {
                    member: arrays[f"repeat_{within_state:02d}__{member}"]
                    for member in CANDIDATE_TRACE_MEMBERS
                }
                trace_index = repeat_trace_base + global_repeat_index
                trace_slice, digests = _candidate_trace_binding(
                    candidate_path, trace, trace_index
                )
                source_endpoint = [float(value) for value in source["h3_endpoint_body"]]
                repeat_endpoint = [float(value) for value in outcome["h3_endpoint_body"]]
                position_error = math.hypot(
                    source_endpoint[0] - repeat_endpoint[0],
                    source_endpoint[1] - repeat_endpoint[1],
                )
                heading_error = abs(_wrap_angle(source_endpoint[2] - repeat_endpoint[2]))
                source_applied = str(source["trace_array_slice_sha256s"]["post_slew_applied_command"])
                repeat_applied = str(digests["post_slew_applied_command"])
                success = bool(
                    source_applied == repeat_applied
                    and bool(source["entered_correct_edge"]) is bool(outcome["entered_correct_edge"])
                    and source["endpoint_edge_id"] == outcome["endpoint_edge_id"]
                    and bool(source["physics_contact"]) is bool(outcome["physics_contact"])
                    and bool(source["stuck"]) is bool(outcome["stuck"])
                    and position_error <= float(CONTRACT.NUMERICAL_TOLERANCES["repeat_endpoint_position_m"])
                    and heading_error <= float(CONTRACT.NUMERICAL_TOLERANCES["repeat_endpoint_heading_rad"])
                )
                repeat_rows.append({
                    "repeat_id": f"{state_id}::{selector_id}::{repeat_index}",
                    "state_id": state_id,
                    "family": str(state["family"]),
                    "branch_selector_id": selector_id,
                    "repeat_index": repeat_index,
                    "source_candidate_index": candidate_index,
                    "source_branch_id": str(source["branch_id"]),
                    "snapshot_id": f"{CONTRACT.SNAPSHOT_ID_PREFIX}{state_id}",
                    "restored_snapshot_sha256": str(metadata["snapshot_payload_sha256"]),
                    "trace_index": trace_index,
                    "trace_slice": trace_slice,
                    "trace_array_slice_sha256s": digests,
                    "source_correct_edge_execution": bool(source["entered_correct_edge"]),
                    "repeat_correct_edge_execution": bool(outcome["entered_correct_edge"]),
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
                    "repeat_success": success,
                    "physical_runtime_core_sha256": physical_runtime_core_sha256,
                })
                global_repeat_index += 1
    if len(repeat_rows) != REPEAT_ROW_COUNT:
        raise ExperimentError("repeat ledger cardinality drift")
    atomic_jsonl(OUTPUT_ROOT / "repeated_execution.jsonl", repeat_rows)
    atomic_json(OUTPUT_ROOT / "state_snapshot_index.json", snapshot_index)
    return {
        "candidate_trace_count": len(traces),
        "candidate_fanout_count": len(fanout_rows),
        "repeat_count": len(repeat_rows),
        "candidate_traces_sha256": sha256_file(candidate_path),
    }


def _inspect_npz_for_pure_metrics(path: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    """Build the same canonical inspection projection as the independent reducer."""

    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(authority):
            raise ExperimentError(f"NPZ member inventory drift: {path.name}")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    members: dict[str, Any] = {}
    for name, spec_value in authority.items():
        spec = dict(spec_value)
        array = arrays[name]
        if array.dtype.str != spec["descr"] or not array.flags.c_contiguous:
            raise ExperimentError(f"NPZ dtype/layout drift: {path.name}:{name}")
        mode = str(spec["hash_mode"])
        hashes: list[str] = []
        if mode == "whole":
            hashes = [canonical_array_sha256(array)]
        elif mode == "rows_axis0":
            hashes = [canonical_array_sha256(array[index]) for index in range(len(array))]
        elif mode == "offset_slices":
            offsets = arrays[str(spec["offsets_member"])]
            for start, stop in zip(offsets[:-1], offsets[1:]):
                payload = array[int(start):int(stop)]
                if path.name == "state_snapshots.npz" and name == "snapshot_payload_bytes":
                    hashes.append(hashlib.sha256(payload.tobytes(order="C")).hexdigest())
                else:
                    hashes.append(canonical_array_sha256(payload))
        else:
            raise ExperimentError(f"unknown NPZ hash mode: {path.name}:{name}")
        offset_values = None
        if any(
            isinstance(other, Mapping) and other.get("offsets_member") == name
            for other in authority.values()
        ):
            offset_values = [int(value) for value in array]
        members[name] = {
            "descr": array.dtype.str,
            "digest_dtype": str(spec["digest_dtype"]),
            "shape": list(array.shape),
            "c_contiguous": True,
            "object_dtype": False,
            "member_sha256": canonical_array_sha256(array),
            "row_or_slice_sha256s": hashes,
            "offset_values": offset_values,
        }
    info = _require_regular(path)
    assert info is not None
    return {
        "path": path.name,
        "bytes": int(info.st_size),
        "sha256": sha256_file(path),
        "members": members,
    }


def recompute_and_persist_metrics_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    """Run the pure metric authority over persisted row/payload evidence."""

    require_stage_runtime("ordinary", fake=fake_runtime)
    _require_initialized()
    documents = {
        "panel_manifest": load_json(OUTPUT_ROOT / "panel_manifest.json"),
        "split_manifest": load_json(OUTPUT_ROOT / "split_manifest.json"),
        "graph_manifest": load_json(OUTPUT_ROOT / "graph_manifest.json"),
        "state_snapshot_index": load_json(OUTPUT_ROOT / "state_snapshot_index.json"),
        "teacher_trace_index": load_json(OUTPUT_ROOT / "teacher_trace_index.json"),
        "edge_port_index": load_json(OUTPUT_ROOT / "edge_port_index.json"),
        "waypoint_contracts": load_json(OUTPUT_ROOT / "waypoint_contracts.json"),
        "pixel_index": load_json(OUTPUT_ROOT / "pixel_index.json"),
        "latent_index": load_json(OUTPUT_ROOT / "latent_index.json"),
        "development_target_selection": load_json(
            OUTPUT_ROOT / "development_target_selection.json"
        ),
    }
    ledgers = {
        "candidate_fanout": load_jsonl(OUTPUT_ROOT / "candidate_fanout.jsonl"),
        "heldout_ranker_scores": load_jsonl(OUTPUT_ROOT / "heldout_ranker_scores.jsonl"),
        "repeated_execution": load_jsonl(OUTPUT_ROOT / "repeated_execution.jsonl"),
    }
    inspections = [
        _inspect_npz_for_pure_metrics(
            OUTPUT_ROOT / leaf, CONTRACT.NPZ_PAYLOAD_AUTHORITY[leaf]
        )
        for leaf in (
            "state_snapshots.npz", "teacher_traces.npz", "rgb_observations.npz",
            "canonical_latents.npz", "candidate_traces.npz",
        )
    ]
    metrics = METRICS.recompute_metrics({
        **documents, **ledgers, "npz_inspections": inspections,
    })
    atomic_json(OUTPUT_ROOT / "metrics.json", metrics)
    return metrics


def _scientific_bindings() -> list[dict[str, Any]]:
    return [
        _file_binding(OUTPUT_ROOT / leaf)
        for leaf in sorted(SCIENTIFIC_LEAVES)
    ]


def _write_publication(metrics: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    runtime = load_json(OUTPUT_ROOT / "contract.json")
    material = load_json(MATERIAL_ROOT / "material_contract.json")
    elapsed_seconds = float(time.time() - float(material["started_at_unix_s"]))
    scientific_bytes = sum(int(row["bytes"]) for row in _scientific_bindings())
    result = attach_digest({
        "schema": "physical_graph_edge_handoff_qualification_v1.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": str(runtime["source_freeze_commit"]),
        "source_baseline_commit": SOURCE_BASELINE_COMMIT,
        "predecessor_result_commit": PARENT_COMMIT,
        "development_only": True,
        "primary_classification": metrics["primary_classification"],
        "secondary_classifications": metrics["secondary_classifications"],
        "next_experiment": metrics["next_experiment"],
        "selected_target_id": metrics["development"]["selected_target_id"],
        "evidence_counts": copy.deepcopy(metrics["evidence_counts"]),
        "panel_metrics": copy.deepcopy(metrics["panel"]),
        "development_metrics": copy.deepcopy(metrics["development"]),
        "heldout_metrics": copy.deepcopy(metrics["heldout"]),
        "repeatability": copy.deepcopy(metrics["repeatability"]),
        "command_tracking": copy.deepcopy(metrics["command_tracking"]),
        "runtime_environments": copy.deepcopy(metrics["runtime_environments"]),
        "stratified_metrics": copy.deepcopy(metrics["stratified"]),
        "gate": copy.deepcopy(metrics["gate"]),
        "component_failures": copy.deepcopy(metrics["component_failures"]),
        "metrics_sha256": sha256_file(OUTPUT_ROOT / "metrics.json"),
        "independent_reducer_receipt_sha256": hashlib.sha256(
            canonical_bytes(receipt)
        ).hexdigest(),
        "runtime_seconds": elapsed_seconds,
        "scientific_storage_bytes": scientific_bytes,
        "models_trained": 0,
        "prohibited_components_trained_or_implemented": [],
    })
    atomic_json(OUTPUT_ROOT / "result.json", result)
    gate = metrics["gate"]
    condition_lines = "\n".join(
        f"- {row['condition_id']}: {json.dumps(row, sort_keys=True)}"
        for row in metrics["heldout"]["condition_summaries"]
    )
    target_lines = "\n".join(
        f"- {row['target_id']}: {json.dumps(row, sort_keys=True)}"
        for row in metrics["development"]["target_summaries"]
    )
    report = (
        f"# {EXPERIMENT_ID}\n\n"
        f"Primary classification: {metrics['primary_classification']}\n\n"
        f"Secondary classifications: {', '.join(metrics['secondary_classifications']) or 'none'}\n\n"
        f"Selected target: {metrics['development']['selected_target_id']}\n\n"
        f"Handoff gate passed: {str(bool(gate['passed'])).lower()}\n\n"
        f"Next experiment: {metrics['next_experiment']}\n\n"
        "## Physical evidence\n\n"
        f"Evidence counts: {json.dumps(metrics['evidence_counts'], sort_keys=True)}\n\n"
        "All 64 selected decision states retained exact serialized solver, controller, "
        "command-history, and RNG snapshots. Each passed two independent 750-sample "
        "serialized-restore fixtures. All 256 prospective teacher traces, 64 actual "
        "teacher-derived directed ports, 768 candidate fanouts, and 64 repeat traces "
        "are persisted at the 2 ms physics rate.\n\n"
        f"Held-out candidate coverage: {json.dumps(metrics['panel'], sort_keys=True)}\n\n"
        "## Development target selection\n\n"
        f"{target_lines}\n\n"
        "## Held-out conditions\n\n"
        f"{condition_lines}\n\n"
        "## Reset, repeat, and controller qualification\n\n"
        f"Repeatability: {json.dumps(metrics['repeatability'], sort_keys=True)}\n\n"
        f"Command tracking: {json.dumps(metrics['command_tracking'], sort_keys=True)}\n\n"
        "## Runtime environments\n\n"
        f"{json.dumps(metrics['runtime_environments'], sort_keys=True)}\n\n"
        f"Stratified outcomes: {json.dumps(metrics['stratified'], sort_keys=True)}\n\n"
        f"Runtime seconds: {elapsed_seconds:.6f}; scientific storage bytes: {scientific_bytes}.\n\n"
        "## Claims boundary\n\n"
        "This is a development-only physical graph-edge handoff qualification. "
        "No model was trained. The frozen V-JEPA encoder and frozen current-visual "
        "ranker were used only for registered inference. No predictor, new local "
        "ranker, memory model, safety model, novelty mechanism, beacon discovery, "
        "or online graph-construction model was trained or implemented.\n"
    )
    atomic_bytes(OUTPUT_ROOT / "result.md", report.encode("utf-8"))
    files = [_file_binding(OUTPUT_ROOT / leaf) for leaf in sorted(
        set(ALL_OUTPUT_LEAVES) - {"file_hashes.json"}
    )]
    manifest = attach_digest({
        "schema": "physical_graph_edge_handoff_qualification_v1.file_hashes.v1",
        "root": str(OUTPUT_ROOT),
        "files": files,
        "file_count_excluding_self": len(files),
        "bytes_excluding_self": sum(int(row["bytes"]) for row in files),
        "file_hashes_self_sha256_excluded": True,
    })
    atomic_json(OUTPUT_ROOT / "file_hashes.json", manifest)
    return result


def report_stage(*, fake_runtime: bool = False, evaluator_module: Any | None = None) -> dict[str, Any]:
    """Persist metrics, run the independent reducer, and publish presentation."""

    require_stage_runtime("ordinary", fake=fake_runtime)
    metrics = recompute_and_persist_metrics_stage(fake_runtime=fake_runtime)
    if set(path.name for path in OUTPUT_ROOT.iterdir()) != set(SCIENTIFIC_LEAVES):
        raise ExperimentError("scientific root inventory drift before independent reduction")
    if evaluator_module is None:
        from scripts import evaluate_physical_graph_edge_handoff_qualification_v1 as evaluator
    else:
        evaluator = evaluator_module
    receipt = evaluator.verify_and_emit(
        OUTPUT_ROOT, EXTERNAL_REGENERATION_RECEIPT, metrics_module=METRICS,
    )
    result = _write_publication(metrics, receipt)
    evaluator.validate_existing_regeneration_receipt(
        OUTPUT_ROOT, EXTERNAL_REGENERATION_RECEIPT, metrics_module=METRICS,
    )
    if set(path.name for path in OUTPUT_ROOT.iterdir()) != set(ALL_OUTPUT_LEAVES):
        raise ExperimentError("final official root inventory drift")
    return result


def physical_backend_smoke_stage() -> dict[str, Any]:
    """Non-panel, non-persisted build/capture/restore/first-act smoke."""

    require_stage_runtime("physical")
    if OUTPUT_ROOT.exists() or MATERIAL_ROOT.exists() or EXTERNAL_REGENERATION_RECEIPT.exists():
        raise ExperimentError("physical smoke requires all experiment roots absent")
    spec = _prospective_pool_specs()[0]
    session = _GenesisPhysicalSession(spec, backend="cpu")
    session.begin_and_settle()
    payload, snapshot, auxiliary = session.capture_snapshot()
    session.restore_snapshot(payload)
    trace = session.execute_requested_ticks([[0.0, 0.0, 0.0]] * 5)
    trace = _normalise_trace(trace, kind="candidate", exact_samples=250)
    return {
        "schema": "physical_graph_edge_handoff_qualification_v1.physical_smoke.v1",
        "candidate_spec_id": str(spec["candidate_spec_id"]),
        "snapshot_payload_bytes": len(payload),
        "solver_field_count": len(snapshot["solver_field_inventory"]),
        "controller_observation_shape": list(snapshot["controller_observation"].shape),
        "command_history_shape": list(snapshot["command_history"].shape),
        "rgb_shape": list(auxiliary["rgb"].shape),
        "physics_samples": len(trace["timestamp_s"]),
        "disallowed_contact_samples": int(trace["physics_contact"].sum()),
        "serialized_restore_used": True,
        "first_policy_act_completed": True,
        "official_root_created": OUTPUT_ROOT.exists(),
        "material_root_created": MATERIAL_ROOT.exists(),
        "pass": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("freeze-docs")
    subparsers.add_parser("initialize")
    qualify = subparsers.add_parser("qualify-pool-state")
    qualify.add_argument("--pool-index", type=int, required=True)
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
    import faulthandler

    faulthandler.enable(all_threads=True)
    arguments = build_parser().parse_args(argv)
    stage = arguments.stage
    if stage == "freeze-docs":
        result: Any = build_freeze_documents()
    elif stage == "initialize":
        result = initialize_stage()
    elif stage == "qualify-pool-state":
        result = qualify_pool_state_stage(arguments.pool_index)
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
    else:  # pragma: no cover - argparse prevents this
        raise ExperimentError(f"unknown stage: {stage}")
    print(canonical_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as exc:
        print(f"{EXPERIMENT_ID}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
