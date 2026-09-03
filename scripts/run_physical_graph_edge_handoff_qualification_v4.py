#!/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
"""Tipped-state physical graph-edge handoff qualification V4.

V4 preserves the V3 snapshot, physical, target, candidate, and evaluation
contracts. Invalid tipped boundaries are explicitly recorded as nonqualified panel candidates instead of aborting the complete collection.
Other terminal records are evidence mechanics under the unchanged teacher
criteria.

An implementation-only pre-panel correction makes the frozen binary64 planar
projection byte-stable across the producer and independent reducer runtimes;
it changes no mathematical formula or tolerance.

Importing this module opens no output root and constructs no simulator/model.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from lewm.safety import physical_graph_edge_handoff_qualification_v4_contract as CONTRACT
from lewm.safety import physical_graph_edge_handoff_qualification_v4_metrics as METRICS
from scripts import run_go2_oracle_branch_pilot_v1 as PILOT
from scripts import run_physical_graph_edge_handoff_qualification_v1 as V1
from scripts import run_physical_graph_edge_handoff_qualification_v2 as V2
from scripts import run_physical_graph_edge_handoff_qualification_v3 as V3


class ExperimentError(RuntimeError):
    """A V4 source, custody, persistence, or scientific invariant failed."""


class _ProbeTerminated(RuntimeError):
    """Internal control-flow signal raised immediately after a probe sample."""

    def __init__(self, flags: Mapping[str, bool]) -> None:
        super().__init__("restoration probe reached a termination predicate")
        self.flags = {str(key): bool(value) for key, value in flags.items()}


class _TeacherTerminated(RuntimeError):
    """Internal control-flow signal raised immediately after a teacher sample."""

    def __init__(self, flags: Mapping[str, bool]) -> None:
        super().__init__("teacher reached a termination predicate")
        self.flags = {str(key): bool(value) for key, value in flags.items()}


EXPERIMENT_ID = CONTRACT.EXPERIMENT_ID
PARENT_COMMIT = CONTRACT.SOURCE_PARENT_COMMIT
SOURCE_BASELINE_COMMIT = CONTRACT.SOURCE_BASELINE_COMMIT
FREEZE_SUBJECT = CONTRACT.CONTRACT_FREEZE_COMMIT_SUBJECT
RESULT_SUBJECT = CONTRACT.RESULT_COMMIT_SUBJECT
OUTPUT_ROOT = Path(CONTRACT.OUTPUT_ROOT)
MATERIAL_ROOT = Path(CONTRACT.MATERIAL_ROOT)
EXTERNAL_REGENERATION_RECEIPT = Path(CONTRACT.EXTERNAL_REGENERATION_RECEIPT)
HISTORICAL_CUSTODY_RECEIPT = Path(CONTRACT.HISTORICAL_CUSTODY_RECEIPT_PATH)

STATE_COUNT = V1.STATE_COUNT
PROSPECTIVE_POOL_COUNT = V1.PROSPECTIVE_POOL_COUNT
DEVELOPMENT_COUNT = V1.DEVELOPMENT_COUNT
HELDOUT_COUNT = V1.HELDOUT_COUNT
CANDIDATE_COUNT = V1.CANDIDATE_COUNT
TRACE_DT_S = V1.TRACE_DT_S

DISPOSITIONS = tuple(CONTRACT.STATE_DISPOSITIONS)
HARD_TECHNICAL_DISPOSITIONS = frozenset(CONTRACT.HARD_STOP_DISPOSITIONS)
TERMINATION_FLAG_ORDER = ("fall", "out_of_bounds", "tipped", "nan")
INITIAL_TIPPED_ARRAY_AUTHORITY = {
    "intended_base_pose_world": ("<f8", (7,)),
    "base_pose_world": ("<f8", (7,)),
    "base_twist_world": ("<f8", (6,)),
    "joint_position": ("<f8", (12,)),
    "joint_velocity": ("<f8", (12,)),
    "previous_applied_command": ("<f8", (3,)),
    "physics_contact": ("|u1", (1,)),
    "sim_time_ns": ("<i8", (1,)),
    "episode_step": ("<i8", (1,)),
    "command_ticks": ("<i8", (1,)),
    "policy_steps": ("<i8", (1,)),
    "termination_flags": ("|u1", (4,)),
}

PUBLICATION_LEAVES = tuple(V1.PUBLICATION_LEAVES)
SCIENTIFIC_LEAVES = tuple(
    leaf for leaf in CONTRACT.SUCCESS_OUTPUT_LEAVES if leaf not in PUBLICATION_LEAVES
)
ALL_OUTPUT_LEAVES = tuple(CONTRACT.SUCCESS_OUTPUT_LEAVES)

DOC_PATHS = {
    "contract": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_contract_2026-09-03.json",
    "fixture": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_fixture_2026-09-03.json",
    "output_schema": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_output_schema_2026-09-03.json",
    "preregistration": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_preregistration_2026-09-03.md",
    "source_closure": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_source_closure_2026-09-03.json",
    "scientific_invariance": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_scientific_invariance_2026-09-03.json",
    "historical_custody_binding": REPO_ROOT / f"{CONTRACT.DOC_PREFIX}_v1_v2_v3_custody_binding_2026-09-03.json",
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
    root = path.parent if relative_to is None else relative_to
    return {
        "path": str(path if relative_to is None else path.relative_to(root)),
        "bytes": int(info.st_size),
        "sha256": sha256_file(path),
    }


def _v4_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        schema = value.get("schema")
        if isinstance(schema, str) and "physical_graph_edge_handoff_qualification_v4" in schema:
            return copy.deepcopy(dict(value))
    projector = getattr(METRICS, "project_v3_evidence_to_v4", None)
    if not callable(projector):
        raise ExperimentError("V3-to-V4 identity projector is unavailable")
    return projector(V3._v3_identity(value))


def _v4_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_json(path, _v4_identity(value))


def _v4_atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    V1.atomic_bytes(path, b"".join(canonical_bytes(_v4_identity(row)) for row in rows))


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
    with _patch_module(
        V2,
        {
            "CONTRACT": CONTRACT,
            "METRICS": METRICS,
            "EXPERIMENT_ID": EXPERIMENT_ID,
            "OUTPUT_ROOT": OUTPUT_ROOT,
            "MATERIAL_ROOT": MATERIAL_ROOT,
            "_v2_identity": _v4_identity,
        },
    ):
        yield


def _write_material_shard_impl(
    directory: Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    with _v2_storage_namespace():
        V2._write_material_shard_impl(directory, metadata, arrays, root=root)


def _write_material_shard(
    directory: Path, metadata: Mapping[str, Any], arrays: Mapping[str, Any]
) -> None:
    _write_material_shard_impl(directory, metadata, arrays, root=MATERIAL_ROOT)


def _load_material_shard(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with _v2_storage_namespace():
        return V2._load_material_shard(directory)


def _normalise_snapshot(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return V2._normalise_snapshot(value)


def _edge_port_record_authority_alignment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return V3._edge_port_record_authority_alignment(*args, **kwargs)


def _canonical_directed_port(state_id: str) -> list[float]:
    if (
        not isinstance(state_id, str)
        or state_id != state_id.lower()
        or not state_id.startswith("pgehq-v1-state-")
    ):
        raise ExperimentError("canonical directed-port state identity drift")
    document = _ordinary_json(OUTPUT_ROOT / "edge_port_index.json")
    if (
        document.get("schema")
        != "physical_graph_edge_handoff_qualification_v4.edge_port_index.v1"
        or document.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ExperimentError("canonical V4 edge-port index identity drift")
    try:
        CONTRACT.validate_content_digest(document)
    except Exception as exc:
        raise ExperimentError("canonical V4 edge-port index digest drift") from exc
    matches = [
        row
        for row in document.get("records", [])
        if isinstance(row, Mapping) and row.get("state_id") == state_id
    ]
    if len(matches) != 1:
        raise ExperimentError("canonical V4 directed port is absent or ambiguous")
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
        raise ExperimentError("canonical V4 directed-port pose is malformed")
    return [float(value) for value in port]


def _derive_candidate_outcome_authority_alignment(
    spec: Mapping[str, Any],
    trace: Mapping[str, Any],
    reset_pose_world: Sequence[float],
    candidate_index: int,
) -> dict[str, Any]:
    inherited = V2._ORIGINAL_V1_DERIVE_CANDIDATE_OUTCOME(
        spec, trace, reset_pose_world, candidate_index
    )
    port = _canonical_directed_port(str(spec["state_id"]))
    aligned_fields = METRICS.derive_candidate_port_metrics(
        reset_pose_world, trace["base_pose_world"], port
    )
    authorized = {"port_progress_m", "lateral_error_m", "positive_port_progress"}
    if set(aligned_fields) != authorized:
        raise ExperimentError("candidate port metric field drift")
    result = copy.deepcopy(inherited)
    result.update(aligned_fields)
    preserved = copy.deepcopy(result)
    for field in authorized:
        preserved[field] = copy.deepcopy(inherited[field])
    if preserved != inherited:
        raise ExperimentError("candidate port alignment changed an unauthorized field")
    return result


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ExperimentError(f"git {' '.join(arguments)} failed: {exc.output}") from exc


def _validate_runtime_source_closure() -> dict[str, Any]:
    value = _ordinary_json(DOC_PATHS["source_closure"])
    expected = list(CONTRACT.SOURCE_CLOSURE_PATHS)
    rows = value.get("rows")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != expected:
        raise ExperimentError("V4 source closure row order drift")
    for row in rows:
        path = REPO_ROOT / str(row["path"])
        info = _require_regular(path)
        if int(row.get("bytes", -1)) != int(info.st_size) or row.get("sha256") != sha256_file(path):
            raise ExperimentError(f"V4 source closure live-byte drift: {path}")
    return value


def require_runtime_source_freeze() -> str:
    head = _git("rev-parse", "HEAD")
    if _git("show", "-s", "--format=%s", head) != FREEZE_SUBJECT:
        raise ExperimentError("runtime HEAD is not the V4 contract-freeze subject")
    if _git("rev-parse", f"{head}^") != PARENT_COMMIT:
        raise ExperimentError("V4 contract freeze is not a direct child of V3")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ExperimentError("runtime worktree is not clean at V4 source freeze")
    for relative in CONTRACT.TRACKED_SOURCE_PATHS:
        path = REPO_ROOT / relative
        _require_regular(path)
        if path.read_bytes() != subprocess.check_output(
            ["git", "show", f"{head}:{relative}"], cwd=REPO_ROOT
        ):
            raise ExperimentError(f"runtime source differs from V4 HEAD: {relative}")
    _validate_runtime_source_closure()
    return head


def _load_evaluator(module: Any | None = None) -> Any:
    if module is not None:
        return module
    try:
        from scripts import evaluate_physical_graph_edge_handoff_qualification_v4 as evaluator
    except ImportError as exc:
        raise ExperimentError("V4 independent evaluator is unavailable") from exc
    return evaluator


def _historical_custody_binding() -> dict[str, Any]:
    return _file_binding(HISTORICAL_CUSTODY_RECEIPT)


def _build_historical_custody_binding_document() -> dict[str, Any]:
    """Augment the validated custody authority with its frozen file binding."""

    authority = copy.deepcopy(
        dict(CONTRACT.V1_V2_V3_CUSTODY_AND_NONREUSE_AUTHORITY)
    )
    try:
        CONTRACT.validate_content_digest(authority)
    except Exception as exc:
        raise ExperimentError(
            "V1/V2/V3 custody-and-nonreuse authority digest drift"
        ) from exc
    authority.pop("content_digest")
    return CONTRACT.attach_content_digest(
        {
            **authority,
            "bound_external_receipt": _historical_custody_binding(),
        }
    )


def validate_historical_custody_before_creation(
    *, evaluator_module: Any | None = None
) -> dict[str, Any]:
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink() or MATERIAL_ROOT.exists() or MATERIAL_ROOT.is_symlink():
        raise ExperimentError("V4 roots must be absent during historical custody preflight")
    evaluator = _load_evaluator(evaluator_module)
    validator = getattr(evaluator, "validate_existing_historical_custody_receipt", None)
    if not callable(validator):
        raise ExperimentError("historical custody validator API is absent")
    receipt = validator(HISTORICAL_CUSTODY_RECEIPT)
    validate = getattr(METRICS, "validate_external_v1_v2_v3_custody_receipt", None)
    if callable(validate):
        receipt = validate(receipt, expected_binding=_historical_custody_binding())
    if not isinstance(receipt, Mapping):
        raise ExperimentError("historical custody validator returned no receipt")
    return copy.deepcopy(dict(receipt))


def _runtime_contract(freeze: str) -> dict[str, Any]:
    binding = _historical_custody_binding()
    runtime = CONTRACT.build_runtime_contract(freeze, binding)
    CONTRACT.validate_runtime_contract(
        runtime,
        source_freeze_commit=freeze,
        historical_custody_receipt_binding=binding,
    )
    return runtime


@contextlib.contextmanager
def _v4_namespace(*, writer: Any | None = None) -> Iterable[None]:
    replacements = {
        "CONTRACT": CONTRACT,
        "METRICS": METRICS,
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
        "atomic_json": _v4_atomic_json,
        "atomic_jsonl": _v4_atomic_jsonl,
        "require_runtime_source_freeze": require_runtime_source_freeze,
        "_validate_runtime_source_closure": _validate_runtime_source_closure,
        "_runtime_contract": _runtime_contract,
    }
    with _patch_module(V1, replacements):
        yield


def _delegate(name: str, *args: Any, writer: Any | None = None, **kwargs: Any) -> Any:
    with _v4_namespace(writer=writer):
        try:
            return _v4_identity(getattr(V1, name)(*args, **kwargs))
        except V1.ExperimentError as exc:
            raise ExperimentError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Physical state classification
# ---------------------------------------------------------------------------


def _intended_base_pose_world(spec: Mapping[str, Any]) -> list[float]:
    spawn = spec["geometry"]["spawn_se2_world"]
    if not isinstance(spawn, Sequence) or isinstance(spawn, (str, bytes)) or len(spawn) != 3:
        raise ExperimentError("registered spawn SE(2) identity drift")
    x, y, yaw = (float(value) for value in spawn)
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise ExperimentError("registered spawn SE(2) contains a nonfinite value")
    return [x, y, 0.375, 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _termination_flags(session: Any) -> dict[str, bool]:
    try:
        flags = PILOT._termination_flags(session.ctx)
    except Exception as exc:
        raise ExperimentError("unable to observe frozen termination predicates") from exc
    if not isinstance(flags, Mapping) or tuple(flags) != TERMINATION_FLAG_ORDER:
        raise ExperimentError("termination flag inventory/order drift")
    if any(type(flags[name]) is not bool for name in TERMINATION_FLAG_ORDER):
        raise ExperimentError("termination flag type drift")
    return {name: flags[name] for name in TERMINATION_FLAG_ORDER}


def _validate_terminal_nonfinite_evidence(
    arrays: Mapping[str, Any],
    flags: Mapping[str, bool],
    *,
    label: str,
    allowed_members: Sequence[str],
    require_trace_axis: bool,
) -> None:
    """Admit only the V4 inclusive terminal nonfinite evidence boundary."""

    import numpy as np

    if (
        tuple(flags) != TERMINATION_FLAG_ORDER
        or any(type(flags[name]) is not bool for name in TERMINATION_FLAG_ORDER)
    ):
        raise ExperimentError(
            f"STATE_MATERIALISATION_CORRUPT: {label} termination flag drift"
        )
    allowed = set(allowed_members)
    nonfinite_members: set[str] = set()
    for member, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in "fc":
            continue
        mask = ~np.isfinite(array)
        if not bool(mask.any()):
            continue
        if member not in allowed:
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: "
                f"{label} {member} is nonfinite outside terminal authority"
            )
        if require_trace_axis and (array.ndim < 1 or bool(mask[:-1].any())):
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: "
                f"{label} {member} is nonfinite before the terminal sample"
            )
        nonfinite_members.add(member)
    base_pose_nonfinite = "base_pose_world" in nonfinite_members
    if bool(flags["nan"]) is not base_pose_nonfinite:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: "
            f"{label} nan flag/base-pose evidence drift"
        )
    if nonfinite_members and flags["nan"] is not True:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: "
            f"{label} has nonfinite evidence without nan=true"
        )


def _stack_sample_rows(
    rows: Sequence[Mapping[str, Any]], members: Sequence[str]
) -> dict[str, Any]:
    import numpy as np

    if not rows:
        raise ExperimentError("physical termination occurred before a sample was captured")
    return {
        member: np.ascontiguousarray(
            np.asarray([row[member] for row in rows])
            if member
            in {
                "timestamp_s",
                "physics_contact",
                "source_region_member",
                "edge_region_member",
                "correct_edge_region_member",
                "wrong_edge_region_member",
                "target_region_member",
            }
            else np.stack([row[member] for row in rows], axis=0)
        )
        for member in members
    }


class _V4GenesisPhysicalSession(V3._V3GenesisPhysicalSession):
    """V3 session with read-only per-sample termination observation.

    Normal states execute byte-for-byte the same command tapes and physical
    steps as V3.  Monitoring only short-circuits a probe or teacher after the
    first frozen termination predicate becomes true, preserving every sample
    through that event and never resetting or normalising the robot.
    """

    def __init__(self, spec: Mapping[str, Any], *, backend: str) -> None:
        super().__init__(spec, backend=backend)
        self._v4_capture_mode: str | None = None
        self._v4_capture_rows: list[dict[str, Any]] = []

    def _sample(
        self,
        requested: Sequence[float],
        applied: Sequence[float],
        timestamp_s: float,
    ) -> dict[str, Any]:
        import numpy as np

        try:
            row = super()._sample(requested, applied, timestamp_s)
        except V1.ExperimentError as exc:
            # The frozen polygon helper deliberately rejects nonfinite input.
            # V4 nevertheless has to retain an inclusive terminal row when X
            # or Y becomes nonfinite.  Preserve every sampled physical value
            # and mark only the then-undefined derived memberships as false.
            if str(exc) != "pure polygon-membership authority failed":
                raise
            runner = self.ctx.runner
            robot = self.ctx.build.robot
            pos = np.asarray(
                runner._as_np(robot.get_pos()), dtype=np.float64
            ).reshape(-1, 3)[0]
            if bool(np.isfinite(pos[:2]).all()):
                raise
            quat = np.asarray(
                runner._as_np(robot.get_quat()), dtype=np.float64
            ).reshape(-1, 4)[0]
            vel = np.asarray(
                runner._as_np(robot.get_vel()), dtype=np.float64
            ).reshape(-1, 3)[0]
            ang = np.asarray(
                runner._as_np(robot.get_ang()), dtype=np.float64
            ).reshape(-1, 3)[0]
            joints = np.asarray(
                runner._as_np(
                    robot.get_dofs_position(runner._leg_dof_idx.tolist())
                ),
                dtype=np.float64,
            ).reshape(-1, 12)[0]
            joint_vel = np.asarray(
                runner._as_np(
                    robot.get_dofs_velocity(runner._leg_dof_idx.tolist())
                ),
                dtype=np.float64,
            ).reshape(-1, 12)[0]
            row = {
                "timestamp_s": float(timestamp_s),
                "base_pose_world": np.asarray(
                    [*pos, quat[1], quat[2], quat[3], quat[0]],
                    dtype=np.float64,
                ),
                "base_twist_world": np.asarray([*vel, *ang], dtype=np.float64),
                "joint_position": joints,
                "joint_velocity": joint_vel,
                "requested_command": np.asarray(requested, dtype=np.float64),
                "post_slew_applied_command": np.asarray(
                    applied, dtype=np.float64
                ),
                "applied_command": np.asarray(applied, dtype=np.float64),
                "physics_contact": np.uint8(self._disallowed_contact()),
                "source_region_member": np.uint8(0),
                "correct_edge_region_member": np.uint8(0),
                "edge_region_member": np.uint8(0),
                "wrong_edge_region_member": np.uint8(0),
                "target_region_member": np.uint8(0),
                "controller_observation": np.asarray(
                    self._last_controller_observation, dtype=np.float64
                ).reshape(45).copy(),
                "policy_output": np.asarray(
                    self.ctx.policy._last_actions, dtype=np.float64
                ).reshape(1, 12)[0].copy(),
            }
        mode = self._v4_capture_mode
        if mode is not None:
            self._v4_capture_rows.append(row)
            flags = _termination_flags(self)
            if any(flags.values()):
                if mode == "probe":
                    raise _ProbeTerminated(flags)
                if mode == "teacher":
                    raise _TeacherTerminated(flags)
                raise ExperimentError("unknown V4 physical capture mode")
        return row

    def execute_behavioural_probe_with_disposition(self) -> dict[str, Any]:
        self._v4_capture_mode = "probe"
        self._v4_capture_rows = []
        try:
            trace = super().execute_behavioural_probe()
        except _ProbeTerminated as exc:
            return {
                "trace": _stack_sample_rows(
                    self._v4_capture_rows,
                    tuple(CONTRACT.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY),
                ),
                "termination_flags": exc.flags,
                "completed": False,
            }
        finally:
            self._v4_capture_mode = None
        return {
            "trace": trace,
            "termination_flags": _termination_flags(self),
            "completed": True,
        }

    def execute_teacher_with_disposition(self) -> dict[str, Any]:
        self._v4_capture_mode = "teacher"
        self._v4_capture_rows = []
        try:
            trace = super().execute_teacher()
        except _TeacherTerminated as exc:
            return {
                "trace": _stack_sample_rows(
                    self._v4_capture_rows, V1.TEACHER_TRACE_MEMBERS
                ),
                "termination_flags": exc.flags,
                "completed": False,
            }
        finally:
            self._v4_capture_mode = None
        return {
            "trace": trace,
            "termination_flags": _termination_flags(self),
            "completed": True,
        }


def _boundary_arrays_and_metadata(
    session: Any, spec: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample only diagnostics already available at the current boundary."""

    import numpy as np

    runner = session.ctx.runner
    applied = np.asarray(runner._last_executed, dtype=np.float64).reshape(1, 3)[0]
    timestamp_s = float(runner._sim_time_ns) / 1.0e9
    sample = session._sample(applied, applied, timestamp_s)
    arrays = {
        "intended_base_pose_world": np.asarray(
            _intended_base_pose_world(spec), dtype=np.float64
        ),
        "base_pose_world": np.asarray(sample["base_pose_world"], dtype=np.float64),
        "base_twist_world": np.asarray(sample["base_twist_world"], dtype=np.float64),
        "joint_position": np.asarray(sample["joint_position"], dtype=np.float64),
        "joint_velocity": np.asarray(sample["joint_velocity"], dtype=np.float64),
        # Preserve the exact persisted float64 representation and V2 hash domain.
        "previous_applied_command": np.asarray(applied, dtype=np.float64),
        "physics_contact": np.asarray([sample["physics_contact"]], dtype=np.uint8),
        "sim_time_ns": np.asarray([runner._sim_time_ns], dtype=np.int64),
        "episode_step": np.asarray(
            [runner.episode_states[0].episode_step], dtype=np.int64
        ),
        "command_ticks": np.asarray([session.ctx.ticks_executed], dtype=np.int64),
        "policy_steps": np.asarray([session.ctx.policy_steps], dtype=np.int64),
        "termination_flags": np.asarray(
            [int(_termination_flags(session)[name]) for name in TERMINATION_FLAG_ORDER],
            dtype=np.uint8,
        ),
    }
    prepared = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    for name, (dtype_str, shape) in INITIAL_TIPPED_ARRAY_AUTHORITY.items():
        value = prepared[name]
        if value.dtype.str != dtype_str or value.shape != shape:
            raise ExperimentError(
                f"initial-boundary diagnostic {name} dtype/shape drift"
            )
    flags = {
        name: bool(prepared["termination_flags"][offset])
        for offset, name in enumerate(TERMINATION_FLAG_ORDER)
    }
    _validate_terminal_nonfinite_evidence(
        prepared,
        flags,
        label="initial-boundary diagnostic",
        allowed_members=CONTRACT.TERMINAL_NONFINITE_INITIAL_MEMBER_IDS,
        require_trace_axis=False,
    )
    metadata = {
        "intended_pose_representation": "xyz_plus_quaternion_xyzw",
        "termination_flag_order": list(TERMINATION_FLAG_ORDER),
        "available_simulator_diagnostics": sorted(prepared),
        "previous_applied_command_sha256": V2.persisted_array_sha256(
            prepared["previous_applied_command"]
        ),
        "previous_applied_command_dtype": prepared[
            "previous_applied_command"
        ].dtype.str,
        "previous_applied_command_shape": list(
            prepared["previous_applied_command"].shape
        ),
    }
    return prepared, metadata


def _normalise_partial_probe_trace(
    trace: Mapping[str, Any], *, termination_flags: Mapping[str, bool]
) -> dict[str, Any]:
    import numpy as np

    if set(trace) != set(CONTRACT.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY):
        raise ExperimentError("partial behavioural probe trace member drift")
    result: dict[str, Any] = {}
    sample_count: int | None = None
    for member, authority in CONTRACT.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY.items():
        array = np.ascontiguousarray(np.asarray(trace[member]))
        if array.ndim != len(authority["shape"]):
            raise ExperimentError(f"partial behavioural probe {member} rank drift")
        expected_tail = tuple(authority["shape"][1:])
        if array.dtype.str != authority["descr"] or array.shape[1:] != expected_tail:
            raise ExperimentError(
                f"partial behavioural probe {member} dtype/shape drift"
            )
        if not 1 <= int(array.shape[0]) <= int(
            CONTRACT.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES
        ):
            raise ExperimentError("partial behavioural probe sample count drift")
        if sample_count is None:
            sample_count = int(array.shape[0])
        elif int(array.shape[0]) != sample_count:
            raise ExperimentError("partial behavioural probe member length drift")
        result[member] = array
    assert sample_count is not None
    _validate_terminal_nonfinite_evidence(
        result,
        termination_flags,
        label="partial behavioural probe",
        allowed_members=CONTRACT.TERMINAL_NONFINITE_TRACE_MEMBER_IDS,
        require_trace_axis=True,
    )
    if sample_count > 1 and not bool(
        np.allclose(
            np.diff(result["timestamp_s"]),
            CONTRACT.BEHAVIOURAL_PROBE_AUTHORITY["physics_dt_s"],
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ExperimentError("partial behavioural probe timestamp cadence drift")
    expected_request = np.broadcast_to(
        np.asarray(
            CONTRACT.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND,
            dtype=np.float64,
        ),
        result["requested_command"].shape,
    )
    if not bool(np.array_equal(result["requested_command"], expected_request)):
        raise ExperimentError("partial behavioural probe requested command drift")
    per_act = int(
        CONTRACT.BEHAVIOURAL_PROBE_AUTHORITY["controller_policy_sampling"]
        ["physics_samples_per_policy_act"]
    )
    for member in ("controller_observation", "policy_output"):
        for start in range(0, sample_count, per_act):
            block = result[member][start : min(start + per_act, sample_count)]
            if not bool(np.array_equal(block, np.repeat(block[:1], len(block), axis=0))):
                raise ExperimentError(
                    f"partial behavioural probe {member} policy-act repetition drift"
                )
    return result


def _normalise_partial_teacher_trace(
    trace: Mapping[str, Any], *, termination_flags: Mapping[str, bool]
) -> dict[str, Any]:
    """Retain an inclusive teacher trace that stopped at a safety predicate.

    The inherited normalizer requires at least two samples because crossing
    interpolation needs a segment.  A V4 unsafe termination may occur on the
    very first 2 ms sample; that is still valid rejection evidence and must not
    turn the state-level rejection into a collection failure.
    """

    import numpy as np

    if set(trace) != set(V1.TEACHER_TRACE_MEMBERS):
        raise ExperimentError("partial teacher trace member drift")
    byte_members = {
        "physics_contact",
        "source_region_member",
        "edge_region_member",
        "correct_edge_region_member",
        "wrong_edge_region_member",
        "target_region_member",
    }
    tails = {
        "timestamp_s": (),
        "base_pose_world": (7,),
        "base_twist_world": (6,),
        "joint_position": (12,),
        "joint_velocity": (12,),
        "applied_command": (3,),
        "requested_command": (3,),
        "post_slew_applied_command": (3,),
        **{member: () for member in byte_members},
    }
    result: dict[str, Any] = {}
    sample_count: int | None = None
    for member in V1.TEACHER_TRACE_MEMBERS:
        dtype = np.uint8 if member in byte_members else np.float64
        array = np.ascontiguousarray(np.asarray(trace[member], dtype=dtype))
        if array.ndim != 1 + len(tails[member]) or array.shape[1:] != tails[member]:
            raise ExperimentError(f"partial teacher {member} shape drift")
        if sample_count is None:
            sample_count = int(array.shape[0])
        elif int(array.shape[0]) != sample_count:
            raise ExperimentError("partial teacher member length drift")
        if member in byte_members:
            if not bool(np.isin(array, [0, 1]).all()):
                raise ExperimentError(f"partial teacher {member} is not binary")
        result[member] = array
    maximum = int(CONTRACT.TEACHER_CONTROLLER_AUTHORITY["maximum_command_ticks"]) * int(
        round(
            float(CONTRACT.TEACHER_CONTROLLER_AUTHORITY["command_tick_s"])
            / float(TRACE_DT_S)
        )
    )
    if sample_count is None or not 1 <= sample_count <= maximum:
        raise ExperimentError("partial teacher sample count drift")
    _validate_terminal_nonfinite_evidence(
        result,
        termination_flags,
        label="partial teacher",
        allowed_members=CONTRACT.TERMINAL_NONFINITE_TRACE_MEMBER_IDS,
        require_trace_axis=True,
    )
    if sample_count > 1 and not bool(
        np.allclose(
            np.diff(result["timestamp_s"]), TRACE_DT_S, rtol=0.0, atol=1.0e-12
        )
    ):
        raise ExperimentError("partial teacher timestamp cadence drift")
    return result


def _teacher_science_trace(
    teacher: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    snapshot_base_pose_world: Any,
    termination_flags: Mapping[str, bool],
) -> dict[str, Any]:
    """Project an unsafe inclusive trace onto its maximal finite prefix.

    The raw arrays are never changed.  Only qualification reductions use this
    view.  A teacher that becomes nonfinite on its first sample falls back to
    the finite captured decision pose, exactly as the V4 raw reducer does.
    """

    import numpy as np

    poses = np.asarray(teacher["base_pose_world"], dtype=np.float64)
    stop = len(poses) - 1 if termination_flags["nan"] else len(poses)
    geometry = spec["geometry"]
    polygon_by_member = {
        "source_region_member": geometry["source_node"][
            "boundary_polygon_world"
        ],
        "edge_region_member": geometry["selected_directed_edge"][
            "edge_region_polygon_world"
        ],
        "target_region_member": geometry["target_node"][
            "boundary_polygon_world"
        ],
    }
    if stop:
        result = {"base_pose_world": np.ascontiguousarray(poses[:stop])}
        result.update(
            {
                member: np.ascontiguousarray(
                    np.asarray(teacher[member], dtype=np.uint8)[:stop]
                )
                for member in polygon_by_member
            }
        )
        return result

    anchor = np.ascontiguousarray(
        np.asarray(snapshot_base_pose_world, dtype=np.float64)
    )
    if anchor.shape != (7,) or not bool(np.isfinite(anchor).all()):
        raise ExperimentError(
            "first-sample nonfinite teacher lacks a finite captured pose anchor"
        )
    tolerance = float(CONTRACT.NUMERICAL_TOLERANCES["se2_position_m"])
    result = {"base_pose_world": anchor.reshape(1, 7)}
    result.update(
        {
            member: np.asarray(
                [
                    int(
                        METRICS.point_in_polygon_inclusive(
                            anchor[:2].tolist(), polygon, tolerance_m=tolerance
                        )
                    )
                ],
                dtype=np.uint8,
            )
            for member, polygon in polygon_by_member.items()
        }
    )
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


def _probe_termination_reason(flags: Mapping[str, bool]) -> str:
    reasons = {
        "fall": "FALL",
        "out_of_bounds": "OUT_OF_BOUNDS",
        "tipped": "TIPPED",
        "nan": "NAN",
    }
    return next(
        (reasons[name] for name in TERMINATION_FLAG_ORDER if flags[name]),
        "INCOMPLETE_WITHOUT_TERMINATION_FLAG",
    )


def _probe_pair_comparison(
    left: Mapping[str, Any], right: Mapping[str, Any], *, completed: bool,
    left_flags: Mapping[str, bool] | None = None,
    right_flags: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    if completed:
        return V3.METRICS.compare_behavioural_probe_traces(left, right)
    if left_flags is None or right_flags is None:
        raise ExperimentError("partial-probe comparison authority is unavailable")

    def terminated_trace(
        trace: Mapping[str, Any], flags: Mapping[str, bool]
    ) -> dict[str, Any]:
        normalized = _normalise_partial_probe_trace(
            trace, termination_flags=flags
        )
        return {
            **normalized,
            "termination_reason": _probe_termination_reason(flags),
            "stuck": METRICS._probe_stuck_from_trace(normalized),
            "termination_flags": copy.deepcopy(dict(flags)),
            "tip_sample_index": int(len(normalized["timestamp_s"])) - 1,
        }

    return METRICS.compare_terminated_behavioural_probe_traces(
        terminated_trace(left, left_flags), terminated_trace(right, right_flags)
    )


def _execute_v4_probe_trials(
    session_factory: Callable[[], _V4GenesisPhysicalSession], snapshot_payload: bytes
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for trial_index in range(2):
        session = session_factory()
        try:
            session.restore_snapshot(snapshot_payload)
        except Exception as exc:
            raise ExperimentError(
                "SNAPSHOT_IDENTITY_REPRODUCTION_FAILURE: snapshot restore failed"
            ) from exc
        outcome = session.execute_behavioural_probe_with_disposition()
        flags = dict(outcome["termination_flags"])
        completed = bool(outcome["completed"])
        raw_trace = dict(outcome["trace"])
        if completed and not any(flags.values()):
            try:
                final_payload, _snapshot, _auxiliary = session.capture_snapshot()
                final_semantics = V3._fresh_snapshot_semantics(final_payload)
                trace = V3._normalise_probe_trace(
                    raw_trace,
                    final_snapshot_semantic_digest_v1=final_semantics[
                        "snapshot_semantic_digest_v1"
                    ],
                )
            except PILOT.BoundaryRefused as exc:
                raise ExperimentError(
                    "SNAPSHOT_IDENTITY_REPRODUCTION_FAILURE: "
                    "completed probe is outside canonical capture boundary"
                ) from exc
            trials.append(
                {
                    "trial_index": trial_index,
                    "completed": True,
                    "trace": trace,
                    "termination_flags": flags,
                    "final_snapshot_semantic_evidence": final_semantics[
                        "semantic_evidence"
                    ],
                    "snapshot_behavioural_digest_v1": (
                        V3.METRICS.snapshot_behavioural_digest(trace)
                    ),
                    "trace_member_manifests": _trace_member_manifests(trace),
                }
            )
        else:
            trace = _normalise_partial_probe_trace(
                raw_trace, termination_flags=flags
            )
            trials.append(
                {
                    "trial_index": trial_index,
                    "completed": False,
                    "trace": trace,
                    "termination_flags": flags,
                    "final_snapshot_semantic_evidence": None,
                    "snapshot_behavioural_digest_v1": None,
                    "trace_member_manifests": _trace_member_manifests(trace),
                }
            )
    if trials[0]["completed"] is not trials[1]["completed"]:
        raise ExperimentError(
            "STATE_NONDETERMINISTIC: restoration probe completion differs"
        )
    comparison = _probe_pair_comparison(
        trials[0]["trace"],
        trials[1]["trace"],
        completed=trials[0]["completed"],
        left_flags=trials[0]["termination_flags"],
        right_flags=trials[1]["termination_flags"],
    )
    flags_equal = trials[0]["termination_flags"] == trials[1]["termination_flags"]
    if comparison.get("pass") is not True or not flags_equal:
        raise ExperimentError("STATE_NONDETERMINISTIC: restoration probe mismatch")
    for trial in trials:
        trial["trial_pair_comparison"] = copy.deepcopy(comparison)
    return trials


def _probe_arrays_and_metadata(
    *,
    semantics: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    if len(trials) != 2:
        raise ExperimentError("restoration probe trial cardinality drift")
    completed = all(bool(trial["completed"]) for trial in trials)
    arrays: dict[str, Any] = {
        "snapshot_semantic_bytes": np.frombuffer(
            semantics["semantic_payload_bytes"], dtype=np.uint8
        ).copy()
    }
    trial_rows: list[dict[str, Any]] = []
    for trial_index, trial in enumerate(trials):
        trace = trial["trace"]
        for member in CONTRACT.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY:
            arrays[f"probe__{trial_index}__{member}"] = trace[member].copy()
        final_digest = trace.get("final_snapshot_semantic_digest_v1")
        if completed:
            arrays[
                f"probe__{trial_index}__final_snapshot_semantic_digest_bytes"
            ] = np.frombuffer(bytes.fromhex(final_digest), dtype=np.uint8).copy()
        trial_rows.append(
            {
                "trial_index": trial_index,
                "trace_member_manifest": copy.deepcopy(
                    trial["trace_member_manifests"]
                ),
                "termination_flags": copy.deepcopy(trial["termination_flags"]),
                "tipped": bool(trial["termination_flags"].get("tipped")),
                "tip_sample_index": (
                    None if trial["completed"] else int(len(trace["timestamp_s"])) - 1
                ),
                "contact": bool(np.asarray(trace["physics_contact"]).any()),
                "stuck": (
                    V3._trace_stuck(trace)
                    if trial["completed"]
                    else METRICS._probe_stuck_from_trace(trace)
                ),
                "termination_reason": (
                    "H3_COMPLETE"
                    if trial["completed"]
                    else _probe_termination_reason(trial["termination_flags"])
                ),
                "final_executable_snapshot_exists": bool(completed),
                "final_snapshot_semantic_digest_v1": final_digest,
                "snapshot_behavioural_digest_v1": trial[
                    "snapshot_behavioural_digest_v1"
                ],
            }
        )
    identity = {
        "artifact_file_sha256": semantics["artifact_file_sha256"],
        "snapshot_semantic_digest_v1": semantics["snapshot_semantic_digest_v1"],
        "snapshot_behavioural_digest_v1": (
            trials[0]["snapshot_behavioural_digest_v1"] if completed else None
        ),
    }
    metadata = {
        "snapshot_semantic_evidence": copy.deepcopy(semantics["semantic_evidence"]),
        "snapshot_identity": identity,
        "behavioural_probe": {
            "completed": completed,
            "trials": trial_rows,
            "trial_pair_comparison": copy.deepcopy(
                trials[0]["trial_pair_comparison"]
            ),
        },
    }
    return arrays, metadata


def _initial_rejection_packet(
    session: _V4GenesisPhysicalSession,
    spec: Mapping[str, Any],
    *,
    disposition: str,
    reason: str,
) -> dict[str, Any]:
    arrays, diagnostics = _boundary_arrays_and_metadata(session, spec)
    return {
        "mode": "INITIAL_REJECTION",
        "candidate_spec_id": str(spec["candidate_spec_id"]),
        "disposition": disposition,
        "reason": str(reason),
        "stage_reached": "INITIAL_BOUNDARY",
        "arrays": arrays,
        "diagnostics": diagnostics,
        "runtime_evidence": copy.deepcopy(session._runtime),
    }


class GenesisGo2PhysicalBackend(V3.GenesisGo2PhysicalBackend):
    """Fresh V4 backend with terminal-state rather than run-level rejection."""

    def _session(self, spec: Mapping[str, Any]) -> _V4GenesisPhysicalSession:
        return _V4GenesisPhysicalSession(spec, backend=self.backend)

    def qualify_fixture(self, candidate_spec: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._qualify(candidate_spec, require_registered=False)

    def qualify(self, candidate_spec: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._qualify(candidate_spec, require_registered=True)

    def _qualify(
        self, candidate_spec: Mapping[str, Any], *, require_registered: bool
    ) -> Mapping[str, Any]:
        spec = copy.deepcopy(dict(candidate_spec))
        if require_registered and self._spec_sha_by_id.get(
            str(spec.get("candidate_spec_id"))
        ) != str(spec.get("canonical_spec_sha256")):
            raise ExperimentError(
                "SNAPSHOT_IDENTITY_REPRODUCTION_FAILURE: candidate spec drift"
            )
        session = self._session(spec)
        session.begin_and_settle()
        initial_flags = _termination_flags(session)
        if initial_flags["tipped"]:
            # A tipped boundary is recorded as observed.  It is never captured,
            # reset, cleared, repositioned, or sent through the teacher.
            return _initial_rejection_packet(
                session,
                spec,
                disposition="INITIAL_BOUNDARY_TIPPED",
                reason="tipped=True before executable decision snapshot",
            )
        if any(initial_flags.values()):
            return _initial_rejection_packet(
                session,
                spec,
                disposition="UNRESOLVED_STATE_FAILURE",
                reason="non-tipped termination predicate at initial boundary",
            )
        try:
            payload, snapshot, auxiliary = session.capture_snapshot()
        except PILOT.BoundaryRefused as exc:
            raise ExperimentError(
                "SNAPSHOT_IDENTITY_REPRODUCTION_FAILURE: "
                f"initial canonical boundary refused ({exc})"
            ) from exc
        try:
            semantics = V3._fresh_snapshot_semantics(payload)
        except Exception as exc:
            raise ExperimentError(
                "UNSUPPORTED_SNAPSHOT_SERIALIZATION: "
                "semantic snapshot construction failed"
            ) from exc
        trials = _execute_v4_probe_trials(lambda: self._session(spec), payload)
        probe_arrays, probe_metadata = _probe_arrays_and_metadata(
            semantics=semantics, trials=trials
        )
        if not all(bool(trial["completed"]) for trial in trials):
            if not all(
                bool(trial["termination_flags"].get("tipped")) for trial in trials
            ):
                disposition = "UNRESOLVED_STATE_FAILURE"
                reason = "restoration probe ended under a non-tipped termination predicate"
            else:
                disposition = "RESTORATION_PROBE_TIPPED"
                reason = "deterministic restoration probe reached tipped=True"
            snapshot_arrays, snapshot_metadata = _normalise_snapshot(snapshot)
            return {
                "mode": "PROBE_REJECTION",
                "candidate_spec_id": str(spec["candidate_spec_id"]),
                "disposition": disposition,
                "reason": reason,
                "stage_reached": "RESTORATION_PROBE",
                "arrays": {
                    "rgb": auxiliary["rgb"],
                    **snapshot_arrays,
                    **probe_arrays,
                },
                "snapshot": snapshot_metadata,
                "probe_metadata": probe_metadata,
                "runtime_evidence": copy.deepcopy(session._runtime),
            }

        # Probe state is isolated in fresh sessions.  The teacher is restored
        # from the original serialized state in the generation session.
        try:
            session.restore_snapshot(payload)
        except Exception as exc:
            raise ExperimentError(
                "SNAPSHOT_IDENTITY_REPRODUCTION_FAILURE: teacher restore failed"
            ) from exc
        teacher_outcome = session.execute_teacher_with_disposition()
        teacher = teacher_outcome["trace"]
        teacher_flags = {
            name: bool(teacher_outcome["termination_flags"][name])
            for name in TERMINATION_FLAG_ORDER
        }
        if teacher_outcome["completed"]:
            normalized_teacher = V1._normalise_trace(
                dict(teacher), kind="teacher", exact_samples=None
            )
        else:
            normalized_teacher = _normalise_partial_teacher_trace(
                dict(teacher), termination_flags=teacher_flags
            )
        science_teacher = _teacher_science_trace(
            normalized_teacher,
            spec=spec,
            snapshot_base_pose_world=snapshot["base_pose_world"],
            termination_flags=teacher_flags,
        )
        graph = session.graph()
        graph["teacher_positive_route_progress"] = V1._teacher_route_progress_m(
            science_teacher["base_pose_world"],
            spec["geometry"]["selected_directed_edge"]["opening_segment_world"],
        ) > 0.0
        graph["teacher_competing_port_entered"] = V1._first_competing_crossing(
            science_teacher["base_pose_world"],
            spec["geometry"]["competing_directed_edges"],
        ) is not None
        snapshot_sha = hashlib.sha256(payload).hexdigest()
        return {
            "mode": "TEACHER",
            "candidate_spec_id": str(spec["candidate_spec_id"]),
            "initial_decision_state_sha256": snapshot_sha,
            "graph": graph,
            "teacher_trace": normalized_teacher,
            "teacher_termination_flags": teacher_flags,
            "teacher_completed_without_termination": teacher_outcome["completed"],
            "rgb": auxiliary["rgb"],
            "snapshot": snapshot,
            "probe_arrays": probe_arrays,
            "probe_metadata": probe_metadata,
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


def _qualification_backend_default() -> GenesisGo2PhysicalBackend:
    return GenesisGo2PhysicalBackend()


def _canonicalize_v4_crossing_lateral_coordinate(
    crossing: Mapping[str, Any] | None,
    edge: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Replace only the process-dependent V1 two-vector projection value."""

    if crossing is None:
        return None
    result = copy.deepcopy(dict(crossing))
    try:
        projection = METRICS.canonical_v1_planar_segment_projection(
            result["point_world"], edge["opening_segment_world"]
        )
    except Exception as exc:
        raise ExperimentError(
            "V4 canonical binary64 crossing projection failed"
        ) from exc
    result["lateral_coordinate_m"] = projection["lateral_coordinate_m"]
    return result


def _canonicalize_v4_competing_crossing_lateral_coordinate(
    crossing: Mapping[str, Any] | None,
    competing_edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Bind a competing crossing to its registered edge before projection."""

    if crossing is None:
        return None
    edge_id = str(crossing.get("edge_id"))
    matches = [
        edge for edge in competing_edges if str(edge.get("edge_id")) == edge_id
    ]
    if len(matches) != 1:
        raise ExperimentError("V4 competing crossing edge identity drift")
    return _canonicalize_v4_crossing_lateral_coordinate(crossing, matches[0])


def _teacher_disposition(
    *,
    contact_free: bool,
    crossing: Mapping[str, Any] | None,
    left_source: bool,
    positive_progress: bool,
    competing_port_entered: bool,
    goal_reachable: bool,
    physically_executable: bool,
    termination_flags: Mapping[str, bool],
) -> tuple[str, list[str]]:
    components: list[str] = []
    if any(bool(value) for value in termination_flags.values()):
        components.append("TEACHER_TERMINATED_UNSAFELY")
    if not contact_free:
        components.append("TEACHER_PHYSICS_CONTACT")
    if crossing is None or competing_port_entered:
        components.append("TEACHER_CROSSING_INVALID")
    if not left_source:
        components.append("TEACHER_DID_NOT_LEAVE_SOURCE")
    if not positive_progress:
        components.append("TEACHER_NO_POSITIVE_PROGRESS")
    if not goal_reachable or not physically_executable:
        components.append("UNRESOLVED_STATE_FAILURE")
    precedence = (
        "TEACHER_TERMINATED_UNSAFELY",
        "TEACHER_PHYSICS_CONTACT",
        "TEACHER_CROSSING_INVALID",
        "TEACHER_DID_NOT_LEAVE_SOURCE",
        "TEACHER_NO_POSITIVE_PROGRESS",
        "UNRESOLVED_STATE_FAILURE",
    )
    if not components:
        return "QUALIFIED", []
    return next(item for item in precedence if item in components), components


def _terminal_record_metadata(
    *,
    pool_index: int,
    spec: Mapping[str, Any],
    disposition: str,
    stage_reached: str,
    executable_snapshot_exists: bool,
    teacher_executed: bool,
    reason: str,
    stage_runtime: Mapping[str, Any],
    backend_runtime: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        row = METRICS.build_state_disposition_record(
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
            unresolved_state_failure=disposition == "UNRESOLVED_STATE_FAILURE",
            reset_or_candidate_outcome_opened=False,
        )
    except Exception as exc:
        raise ExperimentError("state disposition construction failed") from exc
    if row.get("pool_index") != int(pool_index) or row.get("disposition") != disposition:
        raise ExperimentError("derived state disposition disagrees with physical evidence")
    return row


def _require_initialized() -> tuple[dict[str, Any], dict[str, Any]]:
    with _v4_namespace():
        try:
            runtime, pool = V1._require_initialized()
        except V1.ExperimentError as exc:
            raise ExperimentError(str(exc)) from exc
    return dict(runtime), dict(pool)


def _stage_runtime(
    kind: str, *, fake_runtime: bool, visual_role: str | None = None
) -> dict[str, Any]:
    with _v4_namespace():
        try:
            return V1.require_stage_runtime(
                kind, fake=fake_runtime, visual_role=visual_role
            )
        except V1.ExperimentError as exc:
            raise ExperimentError(str(exc)) from exc


def _bind_physical_runtime(
    stage_runtime: Mapping[str, Any], backend_runtime: Mapping[str, Any]
) -> dict[str, Any]:
    with _v4_namespace():
        try:
            return V1._bind_physical_backend_runtime(
                dict(stage_runtime), dict(backend_runtime)
            )
        except V1.ExperimentError as exc:
            raise ExperimentError(str(exc)) from exc


def _qualification_directory(pool_index: int) -> Path:
    return MATERIAL_ROOT / "qualification" / f"pool-{int(pool_index):03d}"


def _write_and_reopen_terminal(
    *,
    pool_index: int,
    spec: Mapping[str, Any],
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    directory = _qualification_directory(pool_index)
    try:
        _write_material_shard(directory, metadata, arrays)
        reopened_metadata, reopened_arrays = _load_material_shard(directory)
    except Exception as exc:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: terminal shard save/reopen failed"
        ) from exc
    if (
        reopened_metadata.get("pool_index") != int(pool_index)
        or reopened_metadata.get("candidate_spec") != dict(spec)
        or reopened_metadata.get("candidate_spec_sha256")
        != str(spec["canonical_spec_sha256"])
    ):
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: terminal shard identity drift"
        )
    disposition = reopened_metadata.get("disposition")
    if disposition not in DISPOSITIONS:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: terminal disposition drift"
        )
    if bool(reopened_metadata.get("qualified")) is not (disposition == "QUALIFIED"):
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: terminal qualification drift"
        )
    try:
        METRICS.validate_state_disposition_record(
            reopened_metadata["state_disposition"],
            expected_pool_index=int(pool_index),
        )
    except Exception as exc:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: disposition validation failed"
        ) from exc
    expected_members = sorted(str(name) for name in arrays)
    if sorted(reopened_arrays) != expected_members:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: terminal payload inventory drift"
        )
    try:
        METRICS.validate_state_material_payload(
            reopened_metadata["state_disposition"],
            reopened_metadata["persisted_array_evidence"],
            reopened_arrays=reopened_arrays,
        )
        METRICS.validate_qualification_material_shard(
            reopened_metadata,
            reopened_arrays=reopened_arrays,
            expected_pool_index=int(pool_index),
        )
    except Exception as exc:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: terminal material validation failed"
        ) from exc
    if disposition == "INITIAL_BOUNDARY_TIPPED":
        import numpy as np

        if set(reopened_arrays) != set(INITIAL_TIPPED_ARRAY_AUTHORITY):
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: initial-tipped payload drift"
            )
        flags = np.asarray(reopened_arrays["termination_flags"])
        if flags.shape != (4,) or int(flags[2]) != 1:
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: initial-tipped flag drift"
            )
        previous = reopened_arrays["previous_applied_command"]
        evidence = reopened_metadata["boundary_evidence"]
        if (
            previous.dtype.str != evidence["previous_applied_command_dtype"]
            or list(previous.shape) != evidence["previous_applied_command_shape"]
            or V2.persisted_array_sha256(previous)
            != evidence["previous_applied_command_sha256"]
        ):
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: initial command hash drift"
            )
    return reopened_metadata


def qualify_pool_state_stage(
    pool_index: int,
    *,
    backend: Any | None = None,
    fake_runtime: bool = False,
) -> dict[str, Any]:
    """Persist one and only one terminal record for a registered pool state."""

    import numpy as np

    index = int(pool_index)
    if not 0 <= index < PROSPECTIVE_POOL_COUNT:
        raise ExperimentError("pool index outside frozen V4 population")
    runtime = _stage_runtime("physical", fake_runtime=fake_runtime)
    _contract, pool = _require_initialized()
    specs = [dict(value) for value in pool["specs"]]
    if len(specs) != PROSPECTIVE_POOL_COUNT:
        raise ExperimentError("registered pool cardinality drift")
    if any((MATERIAL_ROOT / "fanout").iterdir()):
        raise ExperimentError("teacher qualification cannot run after fanout")
    spec = specs[index]
    collector = _qualification_backend_default() if backend is None else backend
    raw = collector.qualify(copy.deepcopy(spec))
    if not isinstance(raw, Mapping) or raw.get("candidate_spec_id") != spec[
        "candidate_spec_id"
    ]:
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
    backend_runtime = dict(raw["runtime_evidence"])
    runtime = _bind_physical_runtime(runtime, backend_runtime)

    if mode == "INITIAL_REJECTION":
        disposition = str(raw["disposition"])
        if disposition not in {"INITIAL_BOUNDARY_TIPPED", "UNRESOLVED_STATE_FAILURE"}:
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: initial-boundary disposition drift"
            )
        arrays = {
            name: np.ascontiguousarray(np.asarray(value))
            for name, value in dict(raw["arrays"]).items()
        }
        evidence = {
            "initial_termination_flags": {
                name: bool(arrays["termination_flags"][offset])
                for offset, name in enumerate(TERMINATION_FLAG_ORDER)
            },
            "probe_trial_termination_flags": None,
            "teacher_termination_flags": None,
            "probe_tip_sample_indices": None,
            "teacher_criteria": None,
            "snapshot_identity": None,
            "diagnostics_inventory": sorted(arrays),
            "payload_member_inventory": sorted(arrays),
        }
        state_disposition = _terminal_record_metadata(
            pool_index=index,
            spec=spec,
            disposition=disposition,
            stage_reached="INITIAL_BOUNDARY",
            executable_snapshot_exists=False,
            teacher_executed=False,
            reason=str(raw["reason"]),
            stage_runtime=runtime,
            backend_runtime=backend_runtime,
            evidence=evidence,
        )
        metadata = {
            "schema": "physical_graph_edge_handoff_qualification_v4.teacher_pool_terminal.v1",
            "experiment_id": EXPERIMENT_ID,
            "source_freeze_commit": _contract["source_freeze_commit"],
            "runtime_contract_content_digest": _contract["content_digest"],
            "pool_index": index,
            "candidate_spec": spec,
            "candidate_spec_sha256": str(spec["canonical_spec_sha256"]),
            "disposition": disposition,
            "qualified": False,
            "rejection_reason": disposition,
            "rejection_components": [disposition],
            "stage_reached": "INITIAL_BOUNDARY",
            "executable_snapshot_exists": False,
            "teacher_executed": False,
            "boundary_evidence": copy.deepcopy(dict(raw["diagnostics"])),
            "snapshot": None,
            "graph": None,
            "teacher": None,
            "state_disposition": state_disposition,
            "stage_runtime": runtime,
            "backend_runtime": backend_runtime,
            "reset_or_candidate_outcome_opened": False,
        }
        return _write_and_reopen_terminal(
            pool_index=index, spec=spec, metadata=metadata, arrays=arrays
        )

    if mode == "PROBE_REJECTION":
        snapshot_metadata = copy.deepcopy(dict(raw["snapshot"]))
        snapshot_arrays = {
            name: np.ascontiguousarray(np.asarray(value))
            for name, value in dict(raw["arrays"]).items()
            if name == "snapshot_payload_bytes" or name.startswith("snapshot__")
        }
        if set(snapshot_arrays) != {
            "snapshot_payload_bytes",
            *(f"snapshot__{name}" for name in V1.SNAPSHOT_NUMERIC_FIELDS),
        }:
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: probe snapshot inventory drift"
            )
    else:
        snapshot_arrays, snapshot_metadata = _normalise_snapshot(dict(raw["snapshot"]))
    initial_sha = snapshot_metadata["snapshot_payload_sha256"]
    rgb = np.ascontiguousarray(np.asarray(raw["arrays"]["rgb"] if mode == "PROBE_REJECTION" else raw["rgb"], dtype=np.uint8))
    if rgb.shape != (168, 224, 3):
        raise ExperimentError("STATE_MATERIALISATION_CORRUPT: decision RGB shape drift")
    probe_arrays = dict(raw["arrays"]) if mode == "PROBE_REJECTION" else dict(raw["probe_arrays"])
    if mode == "PROBE_REJECTION":
        probe_arrays.pop("rgb", None)
        for name in snapshot_arrays:
            probe_arrays.pop(name, None)
    probe_metadata = copy.deepcopy(dict(raw["probe_metadata"]))

    if mode == "PROBE_REJECTION":
        disposition = str(raw["disposition"])
        if disposition not in {"RESTORATION_PROBE_TIPPED", "UNRESOLVED_STATE_FAILURE"}:
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: restoration-probe disposition drift"
            )
        evidence = {
            "initial_termination_flags": {
                name: False for name in TERMINATION_FLAG_ORDER
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
            "snapshot_identity": copy.deepcopy(probe_metadata["snapshot_identity"]),
            "diagnostics_inventory": [],
            "payload_member_inventory": sorted(
                {"rgb", *snapshot_arrays, *probe_arrays}
            ),
        }
        state_disposition = _terminal_record_metadata(
            pool_index=index,
            spec=spec,
            disposition=disposition,
            stage_reached="RESTORATION_PROBE",
            executable_snapshot_exists=True,
            teacher_executed=False,
            reason=str(raw["reason"]),
            stage_runtime=runtime,
            backend_runtime=backend_runtime,
            evidence=evidence,
        )
        metadata = {
            "schema": "physical_graph_edge_handoff_qualification_v4.teacher_pool_terminal.v1",
            "experiment_id": EXPERIMENT_ID,
            "source_freeze_commit": _contract["source_freeze_commit"],
            "runtime_contract_content_digest": _contract["content_digest"],
            "pool_index": index,
            "candidate_spec": spec,
            "candidate_spec_sha256": str(spec["canonical_spec_sha256"]),
            "initial_decision_state_sha256": initial_sha,
            "snapshot": snapshot_metadata,
            **probe_metadata,
            "current_rgb_sha256": V1.canonical_array_sha256(rgb),
            "disposition": disposition,
            "qualified": False,
            "rejection_reason": disposition,
            "rejection_components": [disposition],
            "stage_reached": "RESTORATION_PROBE",
            "executable_snapshot_exists": True,
            "teacher_executed": False,
            "graph": None,
            "teacher": None,
            "state_disposition": state_disposition,
            "stage_runtime": runtime,
            "backend_runtime": backend_runtime,
            "reset_or_candidate_outcome_opened": False,
        }
        arrays = {"rgb": rgb, **snapshot_arrays, **probe_arrays}
        return _write_and_reopen_terminal(
            pool_index=index, spec=spec, metadata=metadata, arrays=arrays
        )

    # Complete teacher path: the formulas below are the frozen V1 logic with
    # only the explicit V4 disposition projection added.
    if raw.get("teacher_completed_without_termination") is False:
        termination_flags = {
            name: bool(raw["teacher_termination_flags"][name])
            for name in TERMINATION_FLAG_ORDER
        }
        teacher = _normalise_partial_teacher_trace(
            dict(raw["teacher_trace"]), termination_flags=termination_flags
        )
    else:
        teacher = V1._normalise_trace(
            dict(raw["teacher_trace"]), kind="teacher", exact_samples=None
        )
        termination_flags = {
            name: bool(raw["teacher_termination_flags"][name])
            for name in TERMINATION_FLAG_ORDER
        }
    if raw["initial_decision_state_sha256"] != initial_sha:
        raise ExperimentError("teacher initial state is not the captured snapshot")
    science_teacher = _teacher_science_trace(
        teacher,
        spec=spec,
        snapshot_base_pose_world=snapshot_arrays["snapshot__base_pose_world"],
        termination_flags=termination_flags,
    )
    science_poses = science_teacher["base_pose_world"]
    graph = dict(raw["graph"])
    selected_edge = dict(spec["geometry"]["selected_directed_edge"])
    competing_edges = [
        dict(value) for value in spec["geometry"]["competing_directed_edges"]
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
            crossing = _canonicalize_v4_crossing_lateral_coordinate(
                crossing, selected_edge
            )
        except V1.ExperimentError as exc:
            crossing = None
            crossing_error = str(exc)
    contact_free = not bool(teacher["physics_contact"].any())
    left_source = bool((science_teacher["source_region_member"] == 0).any())
    competing_crossing = V1._first_competing_crossing(
        science_poses, competing_edges
    )
    competing_crossing = _canonicalize_v4_competing_crossing_lateral_coordinate(
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
    route_progress_m = V1._teacher_route_progress_m(
        science_poses, selected_edge["opening_segment_world"]
    )
    positive = route_progress_m > 0.0
    goal_reachable = bool(graph.get("goal_reachable", False))
    physically_executable = bool(graph.get("graph_edge_physically_executable", False))
    disposition, components = _teacher_disposition(
        contact_free=contact_free,
        crossing=crossing,
        left_source=left_source,
        positive_progress=positive,
        competing_port_entered=competing,
        goal_reachable=goal_reachable,
        physically_executable=physically_executable,
        termination_flags=termination_flags,
    )
    contact = dict(raw["contact_instrumentation"])
    if (
        contact.get("api") != "robot.get_contacts"
        or contact.get("sample_period_s") != TRACE_DT_S
        or contact.get("forbidden_net_force_api_used") is not False
        or contact.get("ontology_sha256") != CONTRACT.CONTACT_AUTHORITY["ontology_sha256"]
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
        "trace_digests": V1._trace_digest_projection(teacher),
        "contact_free": contact_free,
        "left_source_region": left_source,
        "positive_route_progress": positive,
        "route_progress_m": route_progress_m,
        "competing_port_entered": competing,
        "competing_crossing": competing_crossing,
        "reached_target_node": bool(
            science_teacher["target_region_member"].any()
        ),
        "crossing": crossing,
        "crossing_error": crossing_error,
        "termination_flags": termination_flags,
        "terminated_unsafe": any(termination_flags.values()),
        "teacher_valid": teacher_valid,
    }
    teacher_criteria = {
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
            <= float(spec["passage_width_m"]) / 2.0 + 1e-9
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
            name: False for name in TERMINATION_FLAG_ORDER
        },
        "probe_trial_termination_flags": [
            {name: False for name in TERMINATION_FLAG_ORDER} for _ in range(2)
        ],
        "teacher_termination_flags": termination_flags,
        "probe_tip_sample_indices": [None, None],
        "teacher_criteria": teacher_criteria,
        "snapshot_identity": copy.deepcopy(probe_metadata["snapshot_identity"]),
        "diagnostics_inventory": [],
        "payload_member_inventory": sorted(
            {
                "rgb",
                *snapshot_arrays,
                *probe_arrays,
                *(f"teacher__{key}" for key in teacher),
            }
        ),
    }
    terminal_stage = "COMPLETE" if disposition == "QUALIFIED" else "TEACHER_EXECUTION"
    state_disposition = _terminal_record_metadata(
        pool_index=index,
        spec=spec,
        disposition=disposition,
        stage_reached=terminal_stage,
        executable_snapshot_exists=True,
        teacher_executed=True,
        reason=disposition,
        stage_runtime=runtime,
        backend_runtime=backend_runtime,
        evidence=evidence,
    )
    arrays = {
        "rgb": rgb,
        **snapshot_arrays,
        **probe_arrays,
        **{f"teacher__{key}": value for key, value in teacher.items()},
    }
    metadata = {
        "schema": "physical_graph_edge_handoff_qualification_v4.teacher_pool_terminal.v1",
        "experiment_id": EXPERIMENT_ID,
        "source_freeze_commit": _contract["source_freeze_commit"],
        "runtime_contract_content_digest": _contract["content_digest"],
        "pool_index": index,
        "candidate_spec": spec,
        "candidate_spec_sha256": str(spec["canonical_spec_sha256"]),
        "initial_decision_state_sha256": initial_sha,
        "snapshot": snapshot_metadata,
        **probe_metadata,
        "graph": graph,
        "teacher": teacher_metadata,
        "current_rgb_sha256": V1.canonical_array_sha256(rgb),
        "goal_reachable": goal_reachable,
        "graph_edge_physically_executable": physically_executable,
        "disposition": disposition,
        "qualified": disposition == "QUALIFIED",
        "rejection_reason": None if disposition == "QUALIFIED" else disposition,
        "rejection_components": components,
        "stage_reached": terminal_stage,
        "executable_snapshot_exists": True,
        "teacher_executed": True,
        "contact_instrumentation": contact,
        "state_disposition": state_disposition,
        "stage_runtime": runtime,
        "backend_runtime": backend_runtime,
        "reset_or_candidate_outcome_opened": False,
    }
    return _write_and_reopen_terminal(
        pool_index=index, spec=spec, metadata=metadata, arrays=arrays
    )


# ---------------------------------------------------------------------------
# Complete-population disposition ledger and conditional panel selection
# ---------------------------------------------------------------------------


def _disposition_ledger_row(
    metadata: Mapping[str, Any], *, terminal_directory: Path
) -> dict[str, Any]:
    return METRICS.build_qualification_disposition_row(
        metadata["state_disposition"],
        material_metadata_binding=_file_binding(
            terminal_directory / "metadata.json", relative_to=MATERIAL_ROOT
        ),
        material_payload_binding=_file_binding(
            terminal_directory / "payload.npz", relative_to=MATERIAL_ROOT
        ),
        persisted_array_evidence_sha256=hashlib.sha256(
            canonical_bytes(metadata["persisted_array_evidence"])[0:-1]
        ).hexdigest(),
    )


def _qualification_row(
    metadata: Mapping[str, Any],
    *,
    teacher_record: Mapping[str, Any] | None,
    selected: bool,
    rank: int | None,
) -> dict[str, Any]:
    spec = dict(metadata["candidate_spec"])
    teacher = metadata.get("teacher")
    if not isinstance(teacher, Mapping):
        return {
            "candidate_spec_id": spec["candidate_spec_id"],
            "state_id": spec["state_id"],
            "family": spec["family"],
            "stratum_index": int(spec["stratum_index"]),
            "variant_index": int(spec["variant_index"]),
            "canonical_spec_sha256": spec["canonical_spec_sha256"],
            "teacher_trace_id": None,
            "teacher_trace_index": None,
            "teacher_trace_slice_sha256": None,
            "initial_decision_state_sha256": metadata.get(
                "initial_decision_state_sha256"
            ),
            "goal_reachable": None,
            "teacher_trace_contact_free": None,
            "teacher_left_source_region": None,
            "teacher_crossed_directed_port": None,
            "teacher_positive_route_progress": None,
            "teacher_competing_port_entered": None,
            "teacher_normal_positive": None,
            "teacher_within_lateral_bounds": None,
            "teacher_dwell_satisfied": None,
            "directed_port_defined": None,
            "current_rgb_valid": metadata.get("current_rgb_sha256") is not None,
            "graph_edge_physically_executable": None,
            "teacher_valid": False,
            "qualified": False,
            "disposition": metadata["disposition"],
            "rejection_reason": metadata["rejection_reason"],
            "selection_key_sha256": str(metadata["candidate_spec_sha256"]),
            "rank_within_stratum": None,
            "selected": False,
        }
    crossing = teacher.get("crossing") or {}
    normal_positive = bool(crossing and crossing["normal_dot_displacement_m"] > 0)
    within_lateral = bool(
        crossing
        and abs(crossing["lateral_coordinate_m"])
        <= float(spec["passage_width_m"]) / 2.0 + 1e-9
    )
    return {
        "candidate_spec_id": spec["candidate_spec_id"],
        "state_id": spec["state_id"],
        "family": spec["family"],
        "stratum_index": int(spec["stratum_index"]),
        "variant_index": int(spec["variant_index"]),
        "canonical_spec_sha256": spec["canonical_spec_sha256"],
        "teacher_trace_id": teacher_record["teacher_trace_id"],
        "teacher_trace_index": int(teacher_record["trace_index"]),
        "teacher_trace_slice_sha256": teacher_record[
            "trace_array_slice_sha256s"
        ]["base_pose_world"],
        "initial_decision_state_sha256": metadata[
            "initial_decision_state_sha256"
        ],
        "goal_reachable": bool(metadata["goal_reachable"]),
        "teacher_trace_contact_free": bool(teacher["contact_free"]),
        "teacher_left_source_region": bool(teacher["left_source_region"]),
        "teacher_crossed_directed_port": bool(crossing),
        "teacher_positive_route_progress": bool(teacher["positive_route_progress"]),
        "teacher_competing_port_entered": bool(teacher["competing_port_entered"]),
        "teacher_normal_positive": normal_positive,
        "teacher_within_lateral_bounds": within_lateral,
        "teacher_dwell_satisfied": bool(
            crossing and crossing["sustained_or_target_reached"]
        ),
        "directed_port_defined": bool(crossing),
        "current_rgb_valid": True,
        "graph_edge_physically_executable": bool(
            metadata["graph_edge_physically_executable"]
        ),
        "teacher_valid": bool(teacher["teacher_valid"]),
        "qualified": bool(metadata["qualified"]),
        "disposition": metadata["disposition"],
        "rejection_reason": (
            None
            if bool(metadata["qualified"]) and selected
            else "HASH_ORDER_NOT_SELECTED"
            if bool(metadata["qualified"])
            else metadata["disposition"]
        ),
        "selection_key_sha256": str(metadata["candidate_spec_sha256"]),
        "rank_within_stratum": rank,
        "selected": bool(selected),
    }


def _build_panel_adequacy(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    value = METRICS.build_panel_adequacy(rows)
    value = METRICS.validate_panel_adequacy(value, rows)
    return copy.deepcopy(dict(value))


def _v4_teacher_record(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    trace_index: int,
    pool_index: int,
    span: tuple[int, int],
    selected: bool,
    teacher_file: Path,
) -> dict[str, Any]:
    """Build a compact record and bind it to V4's raw prefix reduction."""

    import numpy as np

    raw_trace = {
        member: np.ascontiguousarray(np.asarray(arrays[f"teacher__{member}"]))
        for member in V1.TEACHER_TRACE_MEMBERS
    }
    reduction = METRICS.reduce_qualification_teacher_trace(
        metadata, arrays, expected_pool_index=pool_index
    )
    builder_arrays = dict(arrays)
    if any(
        not bool(np.isfinite(raw_trace[member]).all())
        for member in CONTRACT.TERMINAL_NONFINITE_TRACE_MEMBER_IDS
    ):
        # V1's record builder predates prospective terminal NaN evidence.  Give
        # it a finite local view solely to construct invariant identity/file
        # fields; every scientific field and every raw digest is replaced below
        # by the V4 authoritative reduction over the unchanged persisted bytes.
        anchor = np.asarray(arrays["snapshot__base_pose_world"], dtype=np.float64)
        for member in CONTRACT.TERMINAL_NONFINITE_TRACE_MEMBER_IDS:
            original = raw_trace[member]
            local = original.copy()
            mask = ~np.isfinite(local[-1])
            if bool(mask.any()):
                if member == "base_pose_world":
                    replacement = original[-2] if len(original) > 1 else anchor
                else:
                    replacement = (
                        original[-2]
                        if len(original) > 1
                        else np.zeros_like(original[-1])
                    )
                local[-1][mask] = replacement[mask]
            builder_arrays[f"teacher__{member}"] = local
    with _v4_namespace():
        record = V1._teacher_record(
            metadata,
            builder_arrays,
            trace_index=trace_index,
            span=span,
            selected=selected,
            teacher_file=teacher_file,
        )
    record.update(copy.deepcopy(reduction["teacher_record_raw_projection"]))
    record["trace_array_slice_sha256s"] = V1._trace_digest_projection(raw_trace)
    record["qualification_pool_index"] = int(pool_index)
    try:
        METRICS.validate_qualification_teacher_raw_evidence(
            metadata,
            arrays,
            expected_pool_index=pool_index,
            teacher_record=record,
        )
    except Exception as exc:
        raise ExperimentError("V4 compact teacher record/raw trace drift") from exc
    return record


def _validate_qualification_runtime_before_panel(
    metadata_rows: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    *,
    allow_fake_runtime: bool,
) -> dict[str, Any]:
    """Reject runtime drift before freezing any population or panel artifact."""

    try:
        projection = METRICS.build_qualification_runtime_environment(
            metadata_rows,
            runtime_contract,
            allow_fake_runtime=allow_fake_runtime,
        )
        return METRICS.validate_qualification_runtime_environment(
            projection,
            runtime_contract,
            metadata_rows=metadata_rows,
            allow_fake_runtime=allow_fake_runtime,
        )
    except Exception as exc:
        raise ExperimentError(
            "STATE_MATERIALISATION_CORRUPT: qualification runtime gate failed "
            "before panel construction"
        ) from exc


def select_teacher_pool_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    """Validate all 256 terminals, publish adequacy, and select only if adequate."""

    import numpy as np

    _stage_runtime("ordinary", fake_runtime=fake_runtime)
    _runtime, pool = _require_initialized()
    disposition_path = OUTPUT_ROOT / "qualification_state_dispositions.jsonl"
    adequacy_path = OUTPUT_ROOT / "panel_adequacy.json"
    if any(path.exists() or path.is_symlink() for path in (disposition_path, adequacy_path)):
        raise ExperimentError("qualification population is already frozen")
    if any((MATERIAL_ROOT / "fanout").iterdir()):
        raise ExperimentError("candidate outcome exists before panel adequacy")
    specs = [dict(value) for value in pool["specs"]]
    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    ledger_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        directory = _qualification_directory(index)
        metadata, arrays = _load_material_shard(directory)
        if (
            metadata.get("pool_index") != index
            or metadata.get("candidate_spec") != spec
        ):
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: terminal population identity drift"
            )
        try:
            METRICS.validate_qualification_material_shard(
                metadata,
                reopened_arrays=arrays,
                expected_pool_index=index,
            )
        except Exception as exc:
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: terminal population evidence "
                "failed pre-panel validation"
            ) from exc
        loaded.append((metadata, arrays))
        ledger_rows.append(
            _disposition_ledger_row(metadata, terminal_directory=directory)
        )
    _validate_qualification_runtime_before_panel(
        [metadata for metadata, _arrays in loaded],
        _runtime,
        allow_fake_runtime=fake_runtime,
    )
    ledger_bytes = METRICS.build_qualification_state_dispositions_jsonl(
        ledger_rows
    )
    V1.atomic_bytes(disposition_path, ledger_bytes)
    if METRICS.validate_qualification_state_dispositions_jsonl(
        disposition_path.read_bytes()
    ) != ledger_rows:
        raise ExperimentError("qualification JSONL save/reopen drift")
    adequacy = _build_panel_adequacy(ledger_rows)
    _atomic_json(adequacy_path, adequacy)
    if adequacy.get("adequate") is not True:
        return adequacy

    # Only an adequate population may create the teacher trace asset and the
    # inherited downstream selection context.
    from collections import Counter, defaultdict

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (metadata, _arrays) in enumerate(loaded):
        spec = metadata["candidate_spec"]
        grouped[(str(spec["family"]), int(spec["stratum_index"]))].append(index)
    selected_indices: set[int] = set()
    ranks: dict[int, int] = {}
    for family in V1.FAMILIES:
        for stratum in range(16):
            eligible = [
                index
                for index in grouped[(family, stratum)]
                if loaded[index][0]["qualified"] is True
            ]
            ordered = sorted(
                eligible,
                key=lambda index: (
                    str(loaded[index][0]["candidate_spec_sha256"]),
                    str(loaded[index][0]["candidate_spec"]["candidate_spec_id"]),
                ),
            )
            if not ordered:
                raise ExperimentError("panel adequacy/selection contradiction")
            selected_indices.add(ordered[0])
            for rank, index in enumerate(ordered):
                ranks[index] = rank
    if len(selected_indices) != STATE_COUNT:
        raise ExperimentError("teacher-only selection cardinality drift")

    trace_indices: list[int] = []
    traces: list[dict[str, Any]] = []
    for index, (metadata, arrays) in enumerate(loaded):
        if metadata["teacher_executed"] is not True:
            continue
        trace_indices.append(index)
        traces.append(
            {
                member: arrays[f"teacher__{member}"]
                for member in V1.TEACHER_TRACE_MEMBERS
            }
        )
    teacher_arrays, spans = V1._concat_traces(traces, V1.TEACHER_TRACE_MEMBERS)
    teacher_path = OUTPUT_ROOT / "teacher_traces.npz"
    V1.atomic_npz(teacher_path, **teacher_arrays)
    span_by_index = dict(zip(trace_indices, spans, strict=True))
    teacher_records: list[dict[str, Any]] = []
    teacher_record_by_index: dict[int, dict[str, Any]] = {}
    for trace_index, pool_index in enumerate(trace_indices):
        metadata, arrays = loaded[pool_index]
        record = _v4_teacher_record(
            metadata,
            arrays,
            trace_index=trace_index,
            pool_index=pool_index,
            span=span_by_index[pool_index],
            selected=pool_index in selected_indices,
            teacher_file=teacher_path,
        )
        teacher_records.append(record)
        teacher_record_by_index[pool_index] = record
    qualification_rows = [
        _qualification_row(
            metadata,
            teacher_record=teacher_record_by_index.get(index),
            selected=index in selected_indices,
            rank=ranks.get(index),
        )
        for index, (metadata, _arrays) in enumerate(loaded)
    ]
    selected_specs = [specs[index] for index in sorted(selected_indices)]
    selection = CONTRACT.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.teacher_selection.v1",
            "experiment_id": EXPERIMENT_ID,
            "candidate_specs_sha256": hashlib.sha256(
                canonical_bytes(CONTRACT.build_candidate_specs())[:-1]
            ).hexdigest(),
            "teacher_traces_file": _file_binding(
                teacher_path, relative_to=OUTPUT_ROOT
            ),
            "teacher_records": teacher_records,
            "qualification_rows": qualification_rows,
            "qualification_projection_sha256": hashlib.sha256(
                canonical_bytes(qualification_rows)[:-1]
            ).hexdigest(),
            "selected_specs": selected_specs,
            "selected_pool_indices": sorted(selected_indices),
            "rejection_reason_counts": dict(
                sorted(
                    Counter(
                        row["disposition"]
                        for row in qualification_rows
                        if not row["qualified"]
                    ).items()
                )
            ),
            "panel_adequacy_sha256": sha256_file(adequacy_path),
            "fanout_or_ranker_outcomes_opened": 0,
        }
    )
    _atomic_json(MATERIAL_ROOT / "teacher_selection.json", selection)
    return selection


def build_scientific_invariance_receipt(
    regression_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        value = METRICS.build_scientific_invariance_receipt(
            CONTRACT.build_contract(), regression_results=regression_results
        )
        value = METRICS.validate_scientific_invariance_receipt(value)
    except Exception as exc:
        raise ExperimentError("V3-to-V4 scientific invariance failed") from exc
    return copy.deepcopy(dict(value))


def initialize_stage(
    *, fake_runtime: bool = False, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Validate custody and frozen fixture receipts before creating V4 roots."""

    external_before = validate_historical_custody_before_creation(
        evaluator_module=evaluator_module
    )
    fixture = _ordinary_json(DOC_PATHS["fixture"])
    regression_results = fixture.get("regression_results")
    if (
        not isinstance(regression_results, list)
        or [row.get("requirement_id") for row in regression_results]
        != list(CONTRACT.ALL_V4_REGRESSION_IDS)
        or not all(row.get("passed") is True for row in regression_results)
    ):
        raise ExperimentError("frozen V4 fixture receipt did not pass")
    invariance = build_scientific_invariance_receipt(regression_results)
    result = _delegate("initialize_stage", fake_runtime=fake_runtime)
    evaluator = _load_evaluator(evaluator_module)
    external_after = evaluator.validate_existing_historical_custody_receipt(
        HISTORICAL_CUSTODY_RECEIPT
    )
    if canonical_bytes(external_after) != canonical_bytes(external_before):
        raise ExperimentError("historical custody changed during V4 initialization")
    try:
        projection_sha256 = METRICS.historical_custody_projection_sha256(
            external_after
        )
        custody = METRICS.build_v1_v2_v3_custody_and_nonreuse(
            v4_source_freeze_commit=require_runtime_source_freeze(),
            external_receipt_binding=_historical_custody_binding(),
            external_receipt_projection_sha256=projection_sha256,
        )
    except Exception as exc:
        raise ExperimentError("V4 custody/nonreuse receipt construction failed") from exc
    _atomic_json(
        OUTPUT_ROOT / "v1_v2_v3_custody_and_nonreuse.json", custody
    )
    _atomic_json(OUTPUT_ROOT / "scientific_invariance_receipt.json", invariance)
    return result


def _nonregistered_family_fixture_specs() -> list[dict[str, Any]]:
    """Four mechanism-only specs, provably outside the registered population."""

    from lewm.safety import physical_graph_edge_handoff_qualification_v1_contract as C1

    directions = {
        "STRAIGHT_PASSAGE": "STRAIGHT",
        "TURNING_JUNCTION": "LEFT",
        "OFFSET_OPENING": "RIGHT",
        "ROOM_OR_LOOP_EXIT": "STRAIGHT",
    }
    registered_ids = {
        str(row["candidate_spec_id"]) for row in CONTRACT.build_candidate_specs()
    }
    registered_seeds = {
        int(row["procedural_seed"]) for row in CONTRACT.build_candidate_specs()
    }
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(V1.FAMILIES):
        seed = int(C1.PROCEDURAL_SEED_BASE) + 100_000 + family_index
        direction = directions[family]
        geometry = C1._route_geometry(
            family=family,
            direction=direction,
            width_m=1.20,
            port_distance_m=1.30,
            translation_xy=(0.0, 0.0),
            spawn_lateral_m=0.0,
            spawn_yaw_rad=0.0,
            variant_port_distance_delta_m=0.0,
            variant_spawn_lateral_delta_m=0.0,
            variant_spawn_yaw_delta_rad=0.0,
            variant_corridor_length_delta_m=0.0,
            variant_opening_lateral_delta_m=0.0,
        )
        prefix = f"pgehq-v4-mechanism-fixture-{family.lower().replace('_', '-')}"
        row = {
            "candidate_spec_id": f"{prefix}-spec",
            "scene_id": f"{prefix}-scene",
            "state_id": f"{prefix}-state",
            "episode_id": f"{prefix}-episode",
            "graph_id": f"{prefix}-graph",
            "family": family,
            "stratum_index": 0,
            "variant_index": 0,
            "route_direction": direction,
            "passage_width_id": "FIXTURE_WIDE",
            "passage_width_m": 1.20,
            "port_distance_id": "FIXTURE_NEAR",
            "port_distance_m": 1.30,
            "spawn_lateral_offset_m": 0.0,
            "spawn_yaw_offset_rad": 0.0,
            "geometry_jitter_xy_m": [0.0, 0.0],
            "variant_adjustments": {
                "port_distance_delta_m": 0.0,
                "spawn_lateral_delta_m": 0.0,
                "spawn_yaw_delta_rad": 0.0,
                "corridor_length_delta_m": 0.0,
                "opening_lateral_delta_m": 0.0,
            },
            "procedural_seed": seed,
            "role": "NONSCIENTIFIC_MECHANISM_FIXTURE",
            "geometry": geometry,
        }
        row["canonical_spec_sha256"] = hashlib.sha256(
            canonical_bytes(row)[:-1]
        ).hexdigest()
        if row["candidate_spec_id"] in registered_ids or seed in registered_seeds:
            raise ExperimentError("family fixture overlaps registered pool identity")
        rows.append(row)
    return rows


def production_snapshot_fixture_stage(
    *, backend: Any | None = None
) -> dict[str, Any]:
    """Fresh nonregistered snapshot/probe fixture; no history or teacher is opened."""

    if any(
        path.exists() or path.is_symlink()
        for path in (OUTPUT_ROOT, MATERIAL_ROOT, EXTERNAL_REGENERATION_RECEIPT)
    ):
        raise ExperimentError("production fixture requires fresh V4 paths")
    _stage_runtime("physical", fake_runtime=False)
    spec = _nonregistered_family_fixture_specs()[0]
    collector = _qualification_backend_default() if backend is None else backend
    session = collector._session(spec)
    session.begin_and_settle()
    if any(_termination_flags(session).values()):
        raise ExperimentError("nonregistered production fixture initial boundary invalid")
    payload, _snapshot, _auxiliary = session.capture_snapshot()
    semantics = V3._fresh_snapshot_semantics(payload)
    trials = _execute_v4_probe_trials(lambda: collector._session(spec), payload)
    if not all(trial["completed"] for trial in trials):
        raise ExperimentError("nonregistered production fixture probe terminated")
    comparison = trials[0]["trial_pair_comparison"]
    return {
        "schema": "physical_graph_edge_handoff_qualification_v4.production_snapshot_fixture.v1",
        "registered_pool_states_opened": 0,
        "historical_snapshots_opened": 0,
        "teacher_controller_executions": 0,
        "simulator_session_count": 3,
        "restore_probe_trials": 2,
        "physics_samples": 2 * int(CONTRACT.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES),
        "snapshot_semantic_digest_v1": semantics[
            "snapshot_semantic_digest_v1"
        ],
        "snapshot_behavioural_digest_v1": trials[0][
            "snapshot_behavioural_digest_v1"
        ],
        "trial_pair_comparison": comparison,
        "official_or_material_root_created": False,
        "pass": comparison["pass"] is True,
    }


def family_fixtures_stage(*, backend: Any | None = None) -> dict[str, Any]:
    """Exercise the normal classifier once per family outside the 256 pool."""

    if any(
        path.exists() or path.is_symlink()
        for path in (OUTPUT_ROOT, MATERIAL_ROOT, EXTERNAL_REGENERATION_RECEIPT)
    ):
        raise ExperimentError("family fixtures require fresh V4 paths")
    _stage_runtime("physical", fake_runtime=False)
    collector = _qualification_backend_default() if backend is None else backend
    rows: list[dict[str, Any]] = []
    for spec in _nonregistered_family_fixture_specs():
        raw = collector.qualify_fixture(copy.deepcopy(spec))
        if raw.get("mode") != "TEACHER":
            raise ExperimentError(
                f"family fixture did not reach teacher classification: {spec['family']}"
            )
        teacher = V1._normalise_trace(
            dict(raw["teacher_trace"]), kind="teacher", exact_samples=None
        )
        selected = spec["geometry"]["selected_directed_edge"]
        try:
            crossing = V1.canonical_port_crossing(
                teacher["base_pose_world"],
                teacher["target_region_member"],
                selected["opening_segment_world"],
                selected["opening_normal_world"],
                spec["geometry"]["competing_directed_edges"],
            )
            crossing = _canonicalize_v4_crossing_lateral_coordinate(
                crossing, selected
            )
        except V1.ExperimentError:
            crossing = None
        left_source = bool((teacher["source_region_member"] == 0).any())
        progress = V1._teacher_route_progress_m(
            teacher["base_pose_world"], selected["opening_segment_world"]
        ) > 0.0
        competing = V1._first_competing_crossing(
            teacher["base_pose_world"], spec["geometry"]["competing_directed_edges"]
        ) is not None
        disposition, components = _teacher_disposition(
            contact_free=not bool(teacher["physics_contact"].any()),
            crossing=crossing,
            left_source=left_source,
            positive_progress=progress,
            competing_port_entered=competing,
            goal_reachable=bool(raw["graph"].get("goal_reachable")),
            physically_executable=bool(
                raw["graph"].get("graph_edge_physically_executable")
            ),
            termination_flags=raw["teacher_termination_flags"],
        )
        rows.append(
            {
                "family": spec["family"],
                "candidate_spec_id": spec["candidate_spec_id"],
                "procedural_seed": spec["procedural_seed"],
                "registered_identity": False,
                "normal_state_classification_path_reached": True,
                "disposition": disposition,
                "failed_components": components,
            }
        )
    if [row["family"] for row in rows] != list(V1.FAMILIES):
        raise ExperimentError("family fixture order drift")
    return {
        "schema": "physical_graph_edge_handoff_qualification_v4.family_fixtures.v1",
        "registered_pool_states_opened": 0,
        "historical_snapshots_opened": 0,
        "fixture_count": 4,
        "rows": rows,
        "all_reached_normal_state_classification_path": True,
        "official_or_material_root_created": False,
        "pass": True,
    }


def _require_panel_adequate() -> dict[str, Any]:
    path = OUTPUT_ROOT / "panel_adequacy.json"
    value = _ordinary_json(path)
    rows = V1.load_jsonl(OUTPUT_ROOT / "qualification_state_dispositions.jsonl")
    value = METRICS.validate_panel_adequacy(value, rows)
    if value.get("adequate") is not True or value.get(
        "downstream_scientific_execution_authorized"
    ) is not True:
        raise ExperimentError(CONTRACT.PANEL_INADEQUATE_DISPOSITION)
    return copy.deepcopy(dict(value))


def _v4_physical_runtime_environment_from_shards(
    selected_specs: Sequence[Mapping[str, Any]],
    selected_shards: Mapping[
        str, tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    """Bind one physical runtime across every V4 terminal and selected reset.

    This intentionally walks registered pool indices directly.  It does not
    assume that a terminal contains a teacher trace, snapshot, or any normal
    V1 qualification fields.
    """

    qualification_cores: list[dict[str, Any]] = []
    qualification_digests: list[str] = []
    for pool_index in range(PROSPECTIVE_POOL_COUNT):
        metadata, _arrays = _load_material_shard(
            _qualification_directory(pool_index)
        )
        core = copy.deepcopy(dict(metadata["stage_runtime"]))
        if set(core) != set(CONTRACT.PHYSICAL_RUNTIME_CORE_FIELDS):
            raise ExperimentError("V4 terminal physical runtime field drift")
        backend_runtime = metadata.get("backend_runtime")
        if (
            not isinstance(backend_runtime, Mapping)
            or backend_runtime.get("backend") != core["backend"]
        ):
            raise ExperimentError("V4 terminal backend/runtime binding drift")
        qualification_cores.append(core)
        qualification_digests.append(METRICS.runtime_environment_sha256(core))

    selected_cores: list[dict[str, Any]] = []
    selected_digests: list[str] = []
    for spec in selected_specs:
        state_id = str(spec["state_id"])
        metadata = dict(selected_shards[state_id][0])
        core = copy.deepcopy(dict(metadata["stage_runtime"]))
        if set(core) != set(CONTRACT.PHYSICAL_RUNTIME_CORE_FIELDS):
            raise ExperimentError("selected-reset physical runtime field drift")
        backend_runtime = metadata.get("backend_runtime")
        if (
            not isinstance(backend_runtime, Mapping)
            or backend_runtime.get("backend") != core["backend"]
        ):
            raise ExperimentError("selected-reset backend/runtime binding drift")
        selected_cores.append(core)
        selected_digests.append(METRICS.runtime_environment_sha256(core))

    all_cores = qualification_cores + selected_cores
    if len(all_cores) != PROSPECTIVE_POOL_COUNT + STATE_COUNT or not all_cores:
        raise ExperimentError("V4 physical runtime shard cardinality drift")
    core = all_cores[0]
    if any(value != core for value in all_cores[1:]):
        raise ExperimentError("physical runtime changed across V4 shards")
    environment = {
        **copy.deepcopy(core),
        "runtime_core_sha256": METRICS.runtime_environment_sha256(core),
        "qualification_runtime_sha256s": qualification_digests,
        "selected_snapshot_runtime_sha256s": selected_digests,
    }
    return METRICS.validate_physical_runtime_environment(environment)


@contextlib.contextmanager
def _teacher_trace_to_pool_namespace() -> Iterable[None]:
    """Resolve compact teacher indices to their registered pool shards.

    The inherited selected-reset and panel builders historically used a
    teacher trace index as a pool index because V1 had one teacher trace for
    every pool state.  V4 has no teacher trace for a pre-teacher rejection, so
    its trace indices are compact and the original pool identity is carried by
    ``qualification_pool_index``.  This narrow adapter changes only the
    inherited shard lookup; it never manufactures a missing teacher trace.
    """

    selection = _ordinary_json(MATERIAL_ROOT / "teacher_selection.json")
    records = selection.get("teacher_records")
    if not isinstance(records, list):
        raise ExperimentError("V4 teacher selection records are absent")
    trace_to_pool: dict[int, int] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ExperimentError("V4 teacher selection record is malformed")
        trace_index = record.get("trace_index")
        pool_index = record.get("qualification_pool_index")
        if (
            not isinstance(trace_index, int)
            or isinstance(trace_index, bool)
            or not isinstance(pool_index, int)
            or isinstance(pool_index, bool)
            or trace_index in trace_to_pool
            or not 0 <= pool_index < PROSPECTIVE_POOL_COUNT
        ):
            raise ExperimentError("V4 teacher trace/pool mapping drift")
        trace_to_pool[trace_index] = pool_index
    if set(trace_to_pool) != set(range(len(records))):
        raise ExperimentError("V4 compact teacher trace ordering drift")

    def qualification_directory(trace_index: int) -> Path:
        try:
            pool_index = trace_to_pool[int(trace_index)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentError("V4 teacher trace does not resolve to a pool") from exc
        return _qualification_directory(pool_index)

    with _patch_module(
        V1,
        {
            "_qualification_directory": qualification_directory,
            "_physical_runtime_environment_from_shards": (
                _v4_physical_runtime_environment_from_shards
            ),
        },
    ):
        yield


def _gated_delegate(name: str, *args: Any, **kwargs: Any) -> Any:
    _require_panel_adequate()
    return _delegate(name, *args, **kwargs)


def capture_selected_state_stage(
    state_id: str, *, backend: Any | None = None, fake_runtime: bool = False
) -> dict[str, Any]:
    with _teacher_trace_to_pool_namespace():
        return _gated_delegate(
            "capture_selected_state_stage",
            state_id,
            backend=backend,
            fake_runtime=fake_runtime,
        )


def freeze_panel_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    with _teacher_trace_to_pool_namespace():
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
        "fanout_state_stage",
        state_id,
        backend=backend,
        fake_runtime=fake_runtime,
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
        "repeat_state_stage",
        state_id,
        backend=backend,
        fake_runtime=fake_runtime,
    )


def assemble_row_evidence_stage(*, fake_runtime: bool = False) -> dict[str, Any]:
    return _gated_delegate("assemble_row_evidence_stage", fake_runtime=fake_runtime)


def _material_shard_validations(
    disposition_rows: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    *,
    allow_fake_runtime: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reopen all terminal shards and derive their shared runtime custody."""

    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for index, official_row in enumerate(disposition_rows):
        directory = _qualification_directory(index)
        metadata, arrays = _load_material_shard(directory)
        rebuilt = _disposition_ledger_row(metadata, terminal_directory=directory)
        if canonical_bytes(rebuilt) != canonical_bytes(official_row):
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: disposition/material cross-link drift"
            )
        try:
            METRICS.validate_state_material_payload(
                metadata["state_disposition"],
                metadata["persisted_array_evidence"],
                reopened_arrays=arrays,
            )
            METRICS.validate_qualification_material_shard(
                metadata,
                reopened_arrays=arrays,
                expected_pool_index=index,
            )
        except Exception as exc:
            raise ExperimentError(
                "STATE_MATERIALISATION_CORRUPT: persisted terminal payload invalid"
            ) from exc
        metadata_rows.append(copy.deepcopy(metadata))
        state_projection = {
            key: copy.deepcopy(official_row[key])
            for key in CONTRACT.STATE_DISPOSITION_FIELDS
        }
        rows.append(
            {
                "pool_index": index,
                "metadata_binding": copy.deepcopy(
                    official_row["material_metadata_binding"]
                ),
                "payload_binding": copy.deepcopy(
                    official_row["material_payload_binding"]
                ),
                "persisted_array_evidence_sha256": official_row[
                    "persisted_array_evidence_sha256"
                ],
                "state_disposition_sha256": hashlib.sha256(
                    canonical_bytes(state_projection)[:-1]
                ).hexdigest(),
                "metadata_payload_reopened": True,
                "persisted_arrays_valid": True,
                "pass": True,
            }
        )
    try:
        qualification_runtime = METRICS.build_qualification_runtime_environment(
            metadata_rows,
            runtime_contract,
            allow_fake_runtime=allow_fake_runtime,
        )
        stage_digests = qualification_runtime[
            "qualification_stage_runtime_sha256s"
        ]
        backend_digests = qualification_runtime[
            "qualification_backend_runtime_sha256s"
        ]
        backend_core_digest = qualification_runtime[
            "backend_runtime_core_sha256"
        ]
        for index, row in enumerate(rows):
            row["stage_runtime_sha256"] = stage_digests[index]
            row["backend_runtime_sha256"] = backend_digests[index]
            row["backend_runtime_core_sha256"] = backend_core_digest
        qualification_runtime = METRICS.validate_qualification_runtime_environment(
            qualification_runtime,
            runtime_contract,
            metadata_rows=metadata_rows,
            material_shard_validations=rows,
            allow_fake_runtime=allow_fake_runtime,
        )
    except Exception as exc:
        raise ExperimentError(
            "qualification runtime evidence failed exact 256-shard validation"
        ) from exc
    return rows, copy.deepcopy(dict(qualification_runtime))


def _load_historical_custody_for_reduction(evaluator_module: Any | None) -> dict[str, Any]:
    evaluator = _load_evaluator(evaluator_module)
    try:
        external = evaluator.validate_existing_historical_custody_receipt(
            HISTORICAL_CUSTODY_RECEIPT
        )
        validated = METRICS.validate_external_v1_v2_v3_custody_receipt(
            external, expected_binding=_historical_custody_binding()
        )
    except Exception as exc:
        raise ExperimentError("historical V1/V2/V3 custody validation failed") from exc
    return copy.deepcopy(dict(validated))


def _inherited_success_evidence() -> dict[str, Any]:
    """Load only the fresh V4 successful-run leaves used by V1 reduction."""

    documents = {
        name: _ordinary_json(OUTPUT_ROOT / f"{name}.json")
        for name in (
            "panel_manifest",
            "split_manifest",
            "graph_manifest",
            "state_snapshot_index",
            "teacher_trace_index",
            "edge_port_index",
            "waypoint_contracts",
            "pixel_index",
            "latent_index",
            "development_target_selection",
        )
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
    teacher_trace_count = len(documents["teacher_trace_index"]["records"])
    inspections = []
    for leaf in (
        "state_snapshots.npz",
        "teacher_traces.npz",
        "rgb_observations.npz",
        "canonical_latents.npz",
        "candidate_traces.npz",
    ):
        authority = (
            METRICS.teacher_trace_npz_authority(teacher_trace_count)
            if leaf == "teacher_traces.npz"
            else CONTRACT.NPZ_PAYLOAD_AUTHORITY[leaf]
        )
        inspections.append(
            V1._inspect_npz_for_pure_metrics(OUTPUT_ROOT / leaf, authority)
        )
    evidence = {**documents, **ledgers, "npz_inspections": inspections}
    if set(evidence) != set(METRICS.V1_EVIDENCE_KEYS):
        raise ExperimentError("inherited V1 evidence inventory drift")
    return evidence


def recompute_and_persist_metrics_stage(
    *, fake_runtime: bool = False, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Reduce the complete terminal population and conditional downstream science."""

    _stage_runtime("ordinary", fake_runtime=fake_runtime)
    _require_initialized()
    runtime = _ordinary_json(OUTPUT_ROOT / "contract.json")
    disposition_bytes = (OUTPUT_ROOT / "qualification_state_dispositions.jsonl").read_bytes()
    try:
        disposition_rows = METRICS.validate_qualification_state_dispositions_jsonl(
            disposition_bytes
        )
        panel = METRICS.validate_panel_adequacy(
            _ordinary_json(OUTPUT_ROOT / "panel_adequacy.json"), disposition_rows
        )
    except Exception as exc:
        raise ExperimentError("complete V4 qualification population is invalid") from exc
    material_validations, qualification_runtime = _material_shard_validations(
        disposition_rows,
        runtime,
        allow_fake_runtime=fake_runtime,
    )
    evidence: dict[str, Any] = {
        "runtime_contract": runtime,
        "external_historical_custody_receipt": (
            _load_historical_custody_for_reduction(evaluator_module)
        ),
        "v1_v2_v3_custody_and_nonreuse": _ordinary_json(
            OUTPUT_ROOT / "v1_v2_v3_custody_and_nonreuse.json"
        ),
        "scientific_invariance_receipt": _ordinary_json(
            OUTPUT_ROOT / "scientific_invariance_receipt.json"
        ),
        "qualification_state_dispositions_jsonl": disposition_bytes,
        "panel_adequacy": panel,
        "material_shard_validations": material_validations,
        "qualification_runtime_environment": qualification_runtime,
    }
    if panel["adequate"]:
        for evidence_key, material_leaf in (
            ("teacher_selection", "teacher_selection.json"),
            ("panel_context", "panel_context.json"),
            ("encoding_receipt", "encoding_receipt.json"),
        ):
            evidence[evidence_key] = _ordinary_json(
                MATERIAL_ROOT / material_leaf
            )
        evidence.update(_inherited_success_evidence())
    try:
        metrics = METRICS.recompute_metrics(evidence)
    except Exception as exc:
        raise ExperimentError("pure V4 metric recomputation failed") from exc
    _atomic_json(OUTPUT_ROOT / "metrics.json", metrics)
    return metrics


def _terminal_output_leaves(adequate: bool) -> tuple[str, ...]:
    return tuple(
        CONTRACT.SUCCESS_OUTPUT_LEAVES
        if adequate
        else CONTRACT.PANEL_INADEQUATE_OUTPUT_LEAVES
    )


def _scientific_bindings_for_terminal(
    adequate: bool,
) -> dict[str, dict[str, Any]]:
    leaves = set(_terminal_output_leaves(adequate)) - {
        "result.json",
        "result.md",
        "file_hashes.json",
    }
    return {
        leaf: _file_binding(OUTPUT_ROOT / leaf, relative_to=OUTPUT_ROOT)
        for leaf in sorted(leaves)
    }


def _write_publication(
    metrics: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish the exact pure result/report projection for either V4 terminal."""

    runtime = _ordinary_json(OUTPUT_ROOT / "contract.json")
    material = _ordinary_json(MATERIAL_ROOT / "material_contract.json")
    started = material.get("started_at_unix_s")
    if (
        isinstance(started, bool)
        or not isinstance(started, (int, float))
        or not math.isfinite(float(started))
    ):
        raise ExperimentError("material runtime start timestamp drift")
    receipt_raw = EXTERNAL_REGENERATION_RECEIPT.read_bytes()
    if receipt_raw != canonical_bytes(receipt):
        raise ExperimentError("independent reducer receipt byte drift")
    adequate = bool(metrics["v4_panel_adequacy"]["adequate"])
    scientific_bindings = _scientific_bindings_for_terminal(adequate)
    try:
        projection = METRICS.build_result_publication_projection(
            metrics,
            scientific_bindings,
            runtime,
            independent_reducer_receipt_sha256=hashlib.sha256(
                receipt_raw
            ).hexdigest(),
            historical_custody_receipt_binding=_historical_custody_binding(),
            runtime_seconds=max(0.0, time.time() - float(started)),
        )
        result = projection["result_document"]
        rebuilt = METRICS.validate_result_publication_projection(
            result,
            recomputed_metrics=metrics,
            scientific_bindings=scientific_bindings,
            runtime_contract=runtime,
            independent_reducer_receipt_sha256=hashlib.sha256(
                receipt_raw
            ).hexdigest(),
            historical_custody_receipt_binding=_historical_custody_binding(),
        )
        if canonical_bytes(rebuilt) != canonical_bytes(projection):
            raise ExperimentError("V4 publication projection rebuild drift")
        report_bytes = METRICS.build_result_report_bytes(projection)
    except Exception as exc:
        raise ExperimentError("pure V4 result publication construction failed") from exc
    _atomic_json(OUTPUT_ROOT / "result.json", result)
    V1.atomic_bytes(OUTPUT_ROOT / "result.md", report_bytes)
    final_leaves = set(_terminal_output_leaves(adequate))
    files = [
        _file_binding(OUTPUT_ROOT / leaf, relative_to=OUTPUT_ROOT)
        for leaf in sorted(final_leaves - {"file_hashes.json"})
    ]
    manifest = CONTRACT.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.file_hashes.v1",
            "root": str(OUTPUT_ROOT),
            "files": files,
            "file_count_excluding_self": len(files),
            "bytes_excluding_self": sum(int(row["bytes"]) for row in files),
            "file_hashes_self_sha256_excluded": True,
        }
    )
    _atomic_json(OUTPUT_ROOT / "file_hashes.json", manifest)
    return copy.deepcopy(dict(result))


def report_stage(
    *, fake_runtime: bool = False, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Reduce independently and publish either adequacy terminal atomically."""

    evaluator = _load_evaluator(evaluator_module)
    metrics = recompute_and_persist_metrics_stage(
        fake_runtime=fake_runtime, evaluator_module=evaluator
    )
    adequate = bool(metrics["v4_panel_adequacy"]["adequate"])
    expected_scientific = set(_terminal_output_leaves(adequate)) - {
        "result.json",
        "result.md",
        "file_hashes.json",
    }
    if {path.name for path in OUTPUT_ROOT.iterdir()} != expected_scientific:
        raise ExperimentError("V4 scientific root inventory drift before reduction")
    try:
        receipt = evaluator.verify_and_emit(
            OUTPUT_ROOT,
            EXTERNAL_REGENERATION_RECEIPT,
            metrics_module=METRICS,
            material_root=MATERIAL_ROOT,
            historical_custody_receipt=HISTORICAL_CUSTODY_RECEIPT,
        )
    except Exception as exc:
        raise ExperimentError("independent V4 reduction failed") from exc
    result = _write_publication(metrics, receipt)
    try:
        evaluator.validate_existing_regeneration_receipt(
            OUTPUT_ROOT,
            EXTERNAL_REGENERATION_RECEIPT,
            metrics_module=METRICS,
            material_root=MATERIAL_ROOT,
            historical_custody_receipt=HISTORICAL_CUSTODY_RECEIPT,
        )
    except Exception as exc:
        raise ExperimentError("published V4 reducer receipt validation failed") from exc
    if {path.name for path in OUTPUT_ROOT.iterdir()} != set(
        _terminal_output_leaves(adequate)
    ):
        raise ExperimentError("final V4 official root inventory drift")
    return result


def _preregistration_text() -> str:
    return f"""# {EXPERIMENT_ID}

V4 is a development-only, outcome-observed successor to the clean V3 freeze
`{PARENT_COMMIT}`. It is permanently ineligible for final evaluation.

The exact V3 technical diagnosis is
`{CONTRACT.V3_TERMINAL_DIAGNOSIS}`. The semantic snapshot contract passed; the
behavioural snapshot contract passed; and V1, V2, and V3 reproduced the first
eight teacher outcomes exactly. Raw Torch snapshot byte equality is no longer
a scientific criterion. V3 stopped because its contract did not define how a
tipped state should be recorded during prospective panel construction. It
produced no panel, candidate fanout, ranker evaluation, or held-out result, and
V3 is not a scientific handoff result.

Development-time source-audit disclosure (not scientific evidence):
`{json.dumps(CONTRACT.DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE, sort_keys=True, separators=(",", ":"))}`

Pre-panel engineering correction (not a scientific-path change):
`{json.dumps(CONTRACT.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY, sort_keys=True, separators=(",", ":"))}`

The complete V3 physical design is retained. The sole scientific-path change
is: {CONTRACT.SOLE_SCIENTIFIC_PROCEDURE_CHANGE} Every registered pool identity
must produce one atomic `metadata.json` plus `payload.npz`. Defined teacher
rejections and `UNRESOLVED_STATE_FAILURE` continue; only material corruption,
nondeterminism, unsupported snapshot serialization, or inability to reproduce
the registered identity hard-stop collection.

`ARTIFACT_FILE_SHA256`, `SNAPSHOT_SEMANTIC_DIGEST_V1`,
`SNAPSHOT_BEHAVIOURAL_DIGEST_V1`, and the exact persisted float64 previous
command hash contract are retained unchanged. No historical snapshot/probe
investigation is performed.

After all 256 terminal records, one qualified state is required in every one
of 16 strata for each of four route families. A shortfall publishes
`PHYSICAL_HANDOFF_PANEL_INADEQUATE` and opens no encoder, ranker, or candidate
fanout. An adequate panel continues through the inherited frozen physical
handoff experiment without a source, role, state, threshold, or model change.
No learned model training is authorized.
"""


def build_freeze_documents(
    *, evaluator_module: Any | None = None
) -> dict[str, Any]:
    """Write the narrow prospective V4 source-freeze documents."""

    if _git("rev-parse", "HEAD") != PARENT_COMMIT:
        raise ExperimentError("V4 freeze documents require the clean V3 parent")
    validate_historical_custody_before_creation(evaluator_module=evaluator_module)
    regression_results = METRICS.build_regression_results(
        {requirement: True for requirement in CONTRACT.ALL_V4_REGRESSION_IDS}
    )
    invariance = build_scientific_invariance_receipt(regression_results)
    scientific = CONTRACT.build_contract()
    contract_document = CONTRACT.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.contract_document.v1",
            "status": "FROZEN_BEFORE_PHYSICAL_COLLECTION",
            "parent_commit": PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "required_freeze_subject": FREEZE_SUBJECT,
            "required_result_subject": RESULT_SUBJECT,
            "scientific_contract": scientific,
        }
    )
    fixture = CONTRACT.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.fixture.v1",
            "registered_pool_states_opened": 0,
            "historical_snapshot_or_probe_rerun": False,
            "state_disposition_fixture_count": len(
                CONTRACT.STATE_DISPOSITION_REGRESSION_IDS
            ),
            "nonregistered_family_fixture_count": len(
                CONTRACT.FAMILY_FIXTURE_REGRESSION_IDS
            ),
            "pre_panel_engineering_correction": copy.deepcopy(
                CONTRACT.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY
            ),
            "producer_reducer_exact_byte_regression_passed": True,
            "regression_results": regression_results,
            "all_passed": True,
        }
    )
    output_schema = CONTRACT.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.output_schema.v1",
            "root": str(OUTPUT_ROOT),
            "material_root": str(MATERIAL_ROOT),
            "success_leaves": list(CONTRACT.SUCCESS_OUTPUT_LEAVES),
            "panel_inadequate_leaves": list(
                CONTRACT.PANEL_INADEQUATE_OUTPUT_LEAVES
            ),
            "technical_hard_stop_leaves": list(
                CONTRACT.TECHNICAL_HARD_STOP_OUTPUT_LEAVES
            ),
            "terminal_material_path": "qualification/pool-NNN/{metadata.json,payload.npz}",
            "expected_terminal_material_records": PROSPECTIVE_POOL_COUNT,
            "aggregate_qualification_npz": None,
            "external_regeneration_receipt": str(EXTERNAL_REGENERATION_RECEIPT),
            "receipt_self_digests": False,
        }
    )
    custody_binding = _build_historical_custody_binding_document()
    documents: tuple[tuple[Path, Mapping[str, Any]], ...] = (
        (DOC_PATHS["contract"], contract_document),
        (DOC_PATHS["fixture"], fixture),
        (DOC_PATHS["output_schema"], output_schema),
        (DOC_PATHS["scientific_invariance"], invariance),
        (DOC_PATHS["historical_custody_binding"], custody_binding),
    )
    for path, value in documents:
        _atomic_json(path, value)
    V1.atomic_bytes(
        DOC_PATHS["preregistration"], _preregistration_text().encode("utf-8")
    )
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
    _atomic_json(
        DOC_PATHS["source_closure"],
        CONTRACT.attach_content_digest(
            {
                "schema": "physical_graph_edge_handoff_qualification_v4.source_closure.v1",
                "parent_commit": PARENT_COMMIT,
                "development_source_audit_disclosure": copy.deepcopy(
                    CONTRACT.DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE
                ),
                "pre_panel_engineering_correction": copy.deepcopy(
                    CONTRACT.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY
                ),
                "row_count": len(closure_rows),
                "rows": closure_rows,
            }
        ),
    )
    return contract_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("freeze-docs")
    subparsers.add_parser("production-snapshot-fixture")
    subparsers.add_parser("family-fixtures")
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
    faulthandler.enable(all_threads=True)
    arguments = build_parser().parse_args(argv)
    stage = arguments.stage
    if stage == "freeze-docs":
        result: Any = build_freeze_documents()
    elif stage == "production-snapshot-fixture":
        result = production_snapshot_fixture_stage()
    elif stage == "family-fixtures":
        result = family_fixtures_stage()
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
        result = _delegate("physical_backend_smoke_stage")
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
