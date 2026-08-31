#!/home/andrewknowles/TinyQuadJEPA/bin/python
"""Direct runner for NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1.

This is intentionally a conventional development runner: explicit stages,
ordinary files, one foreground process, and no audit hook, launcher hierarchy,
retry quota, or terminal-custody machinery.  Panel construction is entirely
geometry/candidate based and completes before a V-JEPA encoder or ranker is
opened.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import heapq
import json
import math
import os
import random
import shutil
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

for _thread_variable, _thread_value in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
):
    os.environ.setdefault(_thread_variable, _thread_value)

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "lewm_worlds", REPO_ROOT / "lewm_genesis"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from lewm.safety import non_greedy_local_subgoal_jepa_planning_metrics_v1 as METRICS
from lewm.safety import non_greedy_local_subgoal_jepa_planning_v1_contract as CONTRACT


OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "non_greedy_local_subgoal_jepa_planning_v1"
)
CACHE_ROOT = OUTPUT_ROOT / "cache"
SCENE_ROOT = CACHE_ROOT / "scenes"
RGB_ROOT = CACHE_ROOT / "rgb"
LATENT_ROOT = CACHE_ROOT / "latents"
PREDICTOR_ROOT = CACHE_ROOT / "predictors"

RECOVERY_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "plan_aware_monotone_jepa_cost_development_recovery_v1"
)
PARENT_COMMIT = "507577dcb62044fe449c833eef38dcf883fa71c2"

FAMILIES = tuple(CONTRACT.FAMILY_IDS)
ROLES = dict(CONTRACT.SPLIT_FAMILY_STATE_COUNTS)
ROLE_TOTALS = dict(CONTRACT.SPLIT_STATE_COUNTS)
STATE_COUNT = int(CONTRACT.STATE_COUNT)
CANDIDATE_COUNT = int(CONTRACT.CANDIDATE_COUNT)
TICKS_PER_BLOCK = int(CONTRACT.TICKS_PER_BLOCK)
HORIZON_BLOCKS = int(CONTRACT.HORIZON_COUNT)
COMMAND_DT_S = float(CONTRACT.PANEL_GEOMETRY_AUTHORITY["command_tick_seconds"])
PHYSICS_DT_S = float(CONTRACT.PANEL_GEOMETRY_AUTHORITY["physics_step_seconds"])
ROBOT_RADIUS_M = float(CONTRACT.PANEL_GEOMETRY_AUTHORITY["robot_radius_m"])
GRID_RESOLUTION_M = float(
    CONTRACT.PANEL_GEOMETRY_AUTHORITY["occupancy_grid_resolution_m"]
)
WORLD_HALF_M = float(CONTRACT.PANEL_GEOMETRY_AUTHORITY["world_half_extent_m"])

PRIMITIVES: dict[str, tuple[float, float, float]] = {
    name: tuple(float(value) for value in command)
    for name, command in CONTRACT.PRIMITIVES.items()
}
CANDIDATE_BANK: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (name, tuple(blocks)) for name, blocks in CONTRACT.CANDIDATE_BANK
)


class ExperimentError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def content_digest(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_digest"}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def reducer_content_digest(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_digest"}
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def attach_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_digest"] = content_digest(result)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_bytes(path, canonical_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ExperimentError(f"expected JSON object: {path}")
    return value


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def require_execution_runtime() -> dict[str, Any]:
    authority = CONTRACT.EXECUTION_RUNTIME_AUTHORITY
    if sys.executable != authority["runner_interpreter"]:
        raise ExperimentError(
            f"runner interpreter drift: {sys.executable!r}"
        )
    observed_version = ".".join(str(value) for value in sys.version_info[:3])
    if observed_version != authority["python_version"]:
        raise ExperimentError(f"runner Python version drift: {observed_version}")
    required_environment = authority["required_thread_environment"]
    observed_environment = {
        name: os.environ.get(name) for name in required_environment
    }
    if observed_environment != required_environment:
        raise ExperimentError(
            f"runner numerical thread environment drift: {observed_environment}"
        )
    return {
        "runner_interpreter": sys.executable,
        "python_version": observed_version,
        "thread_environment": observed_environment,
        "numpy_version": np.__version__,
    }


def _validate_execution_contract(
    execution_contract: Mapping[str, Any],
    *,
    head: str,
    expected_recovery_binding: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "source_freeze_commit",
        "source_freeze_parent",
        "source_freeze_subject",
        "scientific_contract",
        "recovery_binding",
        "output_root",
        "execution_runtime",
        "content_digest",
    }
    if (
        set(execution_contract) != expected_fields
        or execution_contract.get("content_digest")
        != content_digest(execution_contract)
        or execution_contract.get("schema")
        != "non_greedy_local_subgoal_jepa_planning_v1.execution_contract.v1"
        or execution_contract.get("source_freeze_commit") != head
        or execution_contract.get("source_freeze_parent") != PARENT_COMMIT
        or execution_contract.get("source_freeze_subject")
        != "Freeze non-greedy local subgoal JEPA planning experiment"
        or execution_contract.get("output_root") != str(OUTPUT_ROOT)
        or execution_contract.get("scientific_contract")
        != CONTRACT.build_contract()
        or execution_contract.get("recovery_binding")
        != expected_recovery_binding
        or execution_contract.get("execution_runtime")
        != require_execution_runtime()
    ):
        raise ExperimentError(
            "execution output is not bound to the current source freeze"
        )
    return dict(execution_contract)


def require_source_freeze(*, require_execution_contract: bool) -> str:
    head = git("rev-parse", "HEAD")
    if (
        git("rev-parse", "HEAD^") != PARENT_COMMIT
        or git("show", "-s", "--format=%s", "HEAD")
        != "Freeze non-greedy local subgoal JEPA planning experiment"
        or git("status", "--porcelain", "--untracked-files=all")
    ):
        raise ExperimentError("scientific stage requires the exact clean source-freeze commit")
    closure_path = REPO_ROOT / (
        "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_"
        "source_closure_2026-08-31.json"
    )
    closure = load_json(closure_path)
    if (
        closure.get("schema")
        != "non_greedy_local_subgoal_jepa_planning_v1.source_closure.v1"
        or closure.get("parent_commit") != PARENT_COMMIT
        or closure.get("content_digest") != content_digest(closure)
        or closure.get("row_count") != len(SOURCE_PATHS)
        or [row.get("path") for row in closure.get("rows", ())] != list(SOURCE_PATHS)
    ):
        raise ExperimentError("source-closure parent drift")
    changed_paths = set(
        git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()
    )
    if changed_paths != set(CONTRACT.TRACKED_SOURCE_PATHS):
        raise ExperimentError("source-freeze changed-path allow-list drift")
    for row in closure.get("rows", ()):
        path = REPO_ROOT / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ExperimentError(f"source-closure row drift: {row.get('path')}")
    if Path(CONTRACT.OUTPUT_ROOT) != OUTPUT_ROOT:
        raise ExperimentError("runtime output root differs from the frozen contract")
    execution_contract_path = OUTPUT_ROOT / "contract.json"
    if require_execution_contract:
        execution_contract = load_json(execution_contract_path)
        _validate_execution_contract(
            execution_contract,
            head=head,
            expected_recovery_binding=verify_recovery_bundle(),
        )
    return head


def verify_recovery_bundle() -> dict[str, Any]:
    observed: list[dict[str, Any]] = []
    expected = getattr(
        CONTRACT,
        "RECOVERY_FILE_BINDINGS",
        getattr(CONTRACT, "RECOVERY_FILES", {}),
    )
    if not expected:
        expected = {
            "recovered_development_result.json": {"bytes": 6470922, "sha256": "510661e5f7da88db7a7c17f06e328cfeb9c9e41e596ea761bf0d10642d0c0b1a"},
            "result.md": {"bytes": 7457, "sha256": "56387c2b7c2cd648a3bca32d98efbaaf3ad548242a9752a4a3bf9e13436533d8"},
            "row_evidence_index.json": {"bytes": 944137, "sha256": "86fd98ac5ffdd7acf04f31d68586fcf688927ef282f5959698a26d6c22ae3201"},
            "regeneration_receipt.json": {"bytes": 13754, "sha256": "4e3576242263bfeebd1533b053bd23cf6d40e060d524782f50ad902bce27f5d8"},
            "file_hashes.json": {"bytes": 1139, "sha256": "77de5e94ed96ebb33e624b2fee8df256aca1d958a72f19a520869770588b0226"},
        }
    captured: dict[str, bytes] = {}
    root_fd = os.open(RECOVERY_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if set(os.listdir(root_fd)) != set(expected):
            raise ExperimentError("recovery authority inventory drift")
        for name, record in expected.items():
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise ExperimentError(f"recovery authority is not regular nlink-1: {name}")
                chunks: list[bytes] = []
                while True:
                    block = os.read(fd, 1 << 20)
                    if not block:
                        break
                    chunks.append(block)
                after = os.fstat(fd)
            finally:
                os.close(fd)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            ):
                raise ExperimentError(f"recovery authority changed during read: {name}")
            raw = b"".join(chunks)
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) != int(record["bytes"]) or digest != record["sha256"]:
                raise ExperimentError(f"recovery authority binding mismatch: {name}")
            captured[name] = raw
            observed.append({"path": str(RECOVERY_ROOT / name), "bytes": len(raw), "sha256": digest})
    finally:
        os.close(root_fd)
    result = json.loads(captured["recovered_development_result.json"])
    if not isinstance(result, Mapping):
        raise ExperimentError("recovered result is not a JSON object")
    if result.get("status") != "DEVELOPMENT_SCIENTIFIC_PAYLOAD_RECOVERED":
        raise ExperimentError("recovered result status drift")
    if result.get("decision", {}).get("primary_classification") != "KINEMATIC_BASELINE_DOMINANT":
        raise ExperimentError("recovered result primary classification drift")
    if (
        result.get("preserved_prior_development_status")
        != CONTRACT.RECOVERY_DECISION_AUTHORITY[
            "exact_preserved_prior_development_status"
        ]
        or result.get("safety_workstream")
        != CONTRACT.RECOVERY_DECISION_AUTHORITY["exact_safety_workstream"]
    ):
        raise ExperimentError("recovered development governance projection drift")
    return attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.recovery_binding.v1",
        "root": str(RECOVERY_ROOT), "files": observed,
        "governance_only": True, "scientific_rows_or_tensors_reused": 0,
        "status": result["status"], "prior_primary": result["decision"]["primary_classification"],
        "next_experiment": result["decision"]["next_experiment"],
        "caveat": result["caveat"], "confirmation_policy": result["confirmation_policy"],
        "preserved_prior_development_status": result[
            "preserved_prior_development_status"
        ],
        "safety_workstream": result["safety_workstream"],
    })


@dataclasses.dataclass(frozen=True)
class Rectangle:
    cx: float
    cy: float
    sx: float
    sy: float
    kind: str = "wall"

    def expanded(self, margin: float) -> "Rectangle":
        return Rectangle(self.cx, self.cy, self.sx + 2 * margin, self.sy + 2 * margin, self.kind)

    def contains(self, x: float, y: float) -> bool:
        return abs(x - self.cx) <= self.sx / 2 and abs(y - self.cy) <= self.sy / 2

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def boundary_walls() -> list[Rectangle]:
    h = WORLD_HALF_M
    t = 0.12
    return [
        Rectangle(0.0, h, 2 * h + 2 * t, t), Rectangle(0.0, -h, 2 * h + 2 * t, t),
        Rectangle(h, 0.0, t, 2 * h + 2 * t), Rectangle(-h, 0.0, t, 2 * h + 2 * t),
    ]


def _jitter(seed: int, label: str, amplitude: float) -> float:
    raw = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return ((int.from_bytes(raw[:8], "big") / (2**64 - 1)) * 2 - 1) * amplitude


def scene_spec(family: str, seed: int) -> dict[str, Any]:
    """Build one candidate-blind procedural task geometry."""
    if family not in FAMILIES:
        raise ExperimentError(f"unknown family {family}")
    # The body-relative goal pose and every deterministic command/endpoint
    # feature are exactly counterbalanced.  Only the obstacle geometry varies,
    # so a no-latent ranker cannot recover family or route side from registered
    # nonvisual inputs.
    profile = seed % 100_000
    side = 1 if profile % 2 == 0 else -1
    jy = _jitter(profile, "wall-y", 0.08)
    jl = _jitter(profile, "length", 0.10)
    start = (-0.48, 0.0, math.pi / 2.0)
    goal = (2.18, 0.0)
    walls = boundary_walls()
    if family == "WALL_DETOUR":
        walls += [
            Rectangle(0.08, -0.12 * side + jy, 0.16, 2.08 + jl, family)
        ]
    elif family == "U_ESCAPE":
        walls += [
            Rectangle(-0.04, -0.08 * side + jy, 0.16, 1.62 + jl, family),
            Rectangle(-1.22, -0.08 * side + jy, 0.16, 1.62 + jl, family),
            Rectangle(-0.63, -0.88 * side + jy, 1.34, 0.16, family),
        ]
    elif family == "DEAD_END_LURE":
        walls += [
            Rectangle(0.10, -0.10 * side + jy, 0.16, 1.82 + jl, family),
            Rectangle(0.60, -1.03 * side + jy, 1.20, 0.16, family),
            Rectangle(0.72, 0.63 * side + jy, 1.35, 0.16, family),
            Rectangle(-0.92, 0.82, 0.30, 0.16, family),
        ]
    else:  # OFFSET_PASSAGE
        opening = 0.92 * side + jy
        gap = 0.66
        lower_end = opening - gap / 2.0
        upper_start = opening + gap / 2.0
        walls += [
            Rectangle(0.10, (-WORLD_HALF_M + lower_end) / 2, 0.16, lower_end + WORLD_HALF_M, family),
            Rectangle(0.10, (upper_start + WORLD_HALF_M) / 2, 0.16, WORLD_HALF_M - upper_start, family),
            Rectangle(-0.92, 0.82, 0.30, 0.16, family),
        ]
    geometry = {
        "family": family, "seed": int(seed), "start": [float(v) for v in start],
        "goal": [float(v) for v in goal], "walls": [wall.as_dict() for wall in walls],
        "world_bounds": [[-WORLD_HALF_M, -WORLD_HALF_M], [WORLD_HALF_M, WORLD_HALF_M]],
    }
    digest = hashlib.sha256(canonical_bytes(geometry)).hexdigest()
    geometry["geometry_digest"] = digest
    geometry["scene_id"] = f"ngls-v1-{family.lower().replace('_', '-')}-{digest[:16]}"
    geometry["episode_id"] = f"episode-{digest[16:32]}"
    return geometry


def rectangles(spec: Mapping[str, Any]) -> list[Rectangle]:
    return [Rectangle(**row) for row in spec["walls"]]


def _segment_intersects_rect(a: tuple[float, float], b: tuple[float, float], rect: Rectangle) -> bool:
    # Liang-Barsky clipping against an axis-aligned rectangle.
    x0, y0 = a; x1, y1 = b
    xmin, xmax = rect.cx - rect.sx / 2, rect.cx + rect.sx / 2
    ymin, ymax = rect.cy - rect.sy / 2, rect.cy + rect.sy / 2
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if abs(p) < 1e-15:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True


def pose_collides(spec: Mapping[str, Any], x: float, y: float) -> bool:
    return any(wall.expanded(ROBOT_RADIUS_M).contains(x, y) for wall in rectangles(spec))


def _grid(spec: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(-WORLD_HALF_M, WORLD_HALF_M + GRID_RESOLUTION_M / 2, GRID_RESOLUTION_M)
    xs, ys = np.meshgrid(axis, axis, indexing="xy")
    blocked = np.zeros(xs.shape, dtype=bool)
    for wall in rectangles(spec):
        expanded = wall.expanded(ROBOT_RADIUS_M)
        blocked |= (np.abs(xs - expanded.cx) <= expanded.sx / 2) & (np.abs(ys - expanded.cy) <= expanded.sy / 2)
    return axis, blocked, np.full(blocked.shape, np.inf, dtype=np.float64)


def _grid_step_allowed(
    blocked: np.ndarray, y: int, x: int, ny: int, nx: int
) -> bool:
    if not (0 <= ny < blocked.shape[0] and 0 <= nx < blocked.shape[1]):
        return False
    if bool(blocked[ny, nx]):
        return False
    dy, dx = ny - y, nx - x
    if abs(dy) > 1 or abs(dx) > 1 or (dy == 0 and dx == 0):
        return False
    if dy != 0 and dx != 0 and (
        bool(blocked[y, nx]) or bool(blocked[ny, x])
    ):
        return False
    return True


def geodesic_field(spec: Mapping[str, Any]) -> dict[str, Any]:
    axis, blocked, distance = _grid(spec)
    gx, gy = (float(v) for v in spec["goal"])
    gi = int(np.argmin(np.abs(axis - gx))); gj = int(np.argmin(np.abs(axis - gy)))
    if blocked[gj, gi]:
        raise ExperimentError(f"goal is blocked: {spec['scene_id']}")
    distance[gj, gi] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, gj, gi)]
    neighbours = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                  (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)))
    while queue:
        value, y, x = heapq.heappop(queue)
        if value != distance[y, x]:
            continue
        for dy, dx, weight in neighbours:
            ny, nx = y + dy, x + dx
            if not _grid_step_allowed(blocked, y, x, ny, nx):
                continue
            candidate = value + weight * GRID_RESOLUTION_M
            if candidate + 1e-12 < distance[ny, nx]:
                distance[ny, nx] = candidate
                heapq.heappush(queue, (candidate, ny, nx))
    return {"axis": axis, "blocked": blocked, "distance": distance}


def field_distance(field: Mapping[str, Any], x: float, y: float) -> float:
    axis = field["axis"]
    ix = int(np.argmin(np.abs(axis - x))); iy = int(np.argmin(np.abs(axis - y)))
    distance = field["distance"]
    direct = float(distance[iy, ix])
    if math.isfinite(direct):
        return direct
    # A physics-rate collision can leave the body infinitesimally outside an
    # inflated boundary while the nearest 5-cm cell centre lies just inside.
    # Resolve only that discretisation case by the closest reachable cell.
    best: tuple[float, float] | None = None
    for radius in range(1, 7):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = iy + dy, ix + dx
                if not (0 <= ny < distance.shape[0] and 0 <= nx < distance.shape[1]):
                    continue
                value = float(distance[ny, nx])
                if not math.isfinite(value):
                    continue
                projected = math.hypot(float(axis[nx]) - x, float(axis[ny]) - y)
                key = (projected, value)
                if best is None or key < best:
                    best = key
        if best is not None:
            return float(best[1] + best[0])
    return math.inf


def heading_error_to_descent(field: Mapping[str, Any], x: float, y: float, yaw: float) -> float:
    axis = field["axis"]; distance = field["distance"]
    ix = int(np.argmin(np.abs(axis - x))); iy = int(np.argmin(np.abs(axis - y)))
    best: tuple[float, float, float] | None = None
    neighbours = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    blocked = field["blocked"]
    for dy, dx, weight in neighbours:
        ny, nx = iy + dy, ix + dx
        if not _grid_step_allowed(blocked, iy, ix, ny, nx) or not math.isfinite(
            distance[ny, nx]
        ):
            continue
        key = (
            float(distance[ny, nx]) + weight * GRID_RESOLUTION_M,
            float(distance[ny, nx]),
            math.atan2(float(dy), float(dx)),
        )
        if best is None or key < best:
            best = key
    if best is None:
        return math.pi
    angle = (best[2] - yaw + math.pi) % (2 * math.pi) - math.pi
    return abs(float(angle))


def _slew_commands(primitives: Sequence[str], *, initial: Sequence[float] = (0.0, 0.0, 0.0)) -> tuple[list[list[float]], list[list[float]]]:
    requested: list[list[float]] = []
    applied: list[list[float]] = []
    previous = np.asarray(initial, dtype=np.float64)
    slew = CONTRACT.PANEL_GEOMETRY_AUTHORITY["slew_limits_per_command_tick"]
    delta = np.asarray(
        [
            slew["delta_vx_max_mps"],
            slew["delta_vy_max_mps"],
            slew["delta_yaw_rate_max_radps"],
        ],
        dtype=np.float64,
    )
    lo = np.asarray([-0.3, 0.0, -0.5]); hi = np.asarray([0.3, 0.0, 0.5])
    for primitive in primitives[:HORIZON_BLOCKS]:
        command = np.asarray(PRIMITIVES[primitive], dtype=np.float64)
        for _ in range(TICKS_PER_BLOCK):
            requested.append(command.tolist())
            bounded = np.clip(command, lo, hi)
            bounded = np.clip(bounded, previous - delta, previous + delta)
            applied.append(bounded.tolist())
            previous = bounded
    return requested, applied


def _simulate_commands(spec: Mapping[str, Any], pose: Sequence[float], commands: Sequence[Sequence[float]]) -> dict[str, Any]:
    x, y, yaw = (float(v) for v in pose)
    ticks: list[dict[str, Any]] = []
    any_contact = False
    substeps = int(round(COMMAND_DT_S / PHYSICS_DT_S))
    for tick_index, command in enumerate(commands):
        vx, vy, yaw_rate = (float(v) for v in command)
        contact = False
        for _ in range(substeps):
            next_yaw = yaw + yaw_rate * PHYSICS_DT_S
            nx = x + (math.cos(yaw) * vx - math.sin(yaw) * vy) * PHYSICS_DT_S
            ny = y + (math.sin(yaw) * vx + math.cos(yaw) * vy) * PHYSICS_DT_S
            if pose_collides(spec, nx, ny):
                contact = True
            else:
                x, y = nx, ny
            yaw = next_yaw
        any_contact |= contact
        ticks.append({"tick": tick_index + 1, "pose": [x, y, yaw], "contact": contact, "applied_command": [vx, vy, yaw_rate]})
    return {"pose": [x, y, yaw], "ticks": ticks, "contact": any_contact}


def successor_viability(
    spec: Mapping[str, Any],
    pose: Sequence[float],
    previous_applied_command: Sequence[float],
) -> tuple[bool, int]:
    safe = 0
    for primitive in PRIMITIVES:
        requested, applied = _slew_commands(
            (primitive,), initial=previous_applied_command
        )
        del requested
        outcome = _simulate_commands(spec, pose, applied[:TICKS_PER_BLOCK])
        if not outcome["contact"]:
            safe += 1
    return safe > 0, safe


def _completed_at_goal(spec: Mapping[str, Any], pose: Sequence[float]) -> bool:
    gx, gy = (float(value) for value in spec["goal"])
    return math.hypot(gx - float(pose[0]), gy - float(pose[1])) <= (
        CONTRACT.COMPLETION_RADIUS_M
    )


def _stuck_outcome(
    start: Sequence[float], end: Sequence[float], applied: Sequence[Sequence[float]]
) -> bool:
    displacement = math.hypot(
        float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
    )
    nontrivial_command = any(
        abs(float(command[0])) > CONTRACT.STUCK_APPLIED_VX_THRESHOLD_MPS
        or abs(float(command[2]))
        > CONTRACT.STUCK_APPLIED_ABS_YAW_RATE_THRESHOLD_RADPS
        for command in applied
    )
    return bool(
        displacement < CONTRACT.STUCK_ENDPOINT_DISPLACEMENT_THRESHOLD_M
        and nontrivial_command
    )


def simulate_candidate(spec: Mapping[str, Any], field: Mapping[str, Any], candidate_index: int) -> dict[str, Any]:
    name, primitive_blocks = CANDIDATE_BANK[candidate_index]
    requested, applied = _slew_commands(primitive_blocks)
    start = spec["start"]
    outcome = _simulate_commands(spec, start, applied)
    decimals = int(
        CONTRACT.GEODESIC_TARGET_AUTHORITY[
            "persisted_numeric_quantization_decimal_places"
        ]
    )
    start_geo = round(
        field_distance(field, float(start[0]), float(start[1])), decimals
    )
    end = outcome["pose"]
    end_geo = round(
        field_distance(field, float(end[0]), float(end[1])), decimals
    )
    gx, gy = (float(v) for v in spec["goal"])
    start_euclid = math.hypot(gx - float(start[0]), gy - float(start[1]))
    end_euclid = math.hypot(gx - float(end[0]), gy - float(end[1]))
    h1_pose = outcome["ticks"][TICKS_PER_BLOCK - 1]["pose"]
    h1_euclid = math.hypot(gx - float(h1_pose[0]), gy - float(h1_pose[1]))
    viable, successor_count = successor_viability(spec, end, applied[-1])
    h1_contact = any(row["contact"] for row in outcome["ticks"][:TICKS_PER_BLOCK])
    stuck = _stuck_outcome(start, end, applied)
    return {
        "candidate_index": candidate_index, "candidate": name,
        "primitive_blocks": list(primitive_blocks), "requested_commands": requested,
        "post_slew_applied_commands": applied,
        "horizon_poses": {f"H{h}": outcome["ticks"][h * TICKS_PER_BLOCK - 1]["pose"] for h in (1, 2, 3)},
        "successor_pose": end, "start_geodesic_distance": start_geo,
        "endpoint_geodesic_distance": end_geo,
        "geodesic_progress": round(start_geo - end_geo, decimals),
        "euclidean_progress": round(start_euclid - end_euclid, decimals),
        "completed": _completed_at_goal(spec, end),
        "h1_euclidean_progress": round(start_euclid - h1_euclid, decimals),
        "heading_change": round(
            (float(end[2]) - float(start[2]) + math.pi) % (2 * math.pi)
            - math.pi,
            decimals,
        ),
        "heading_error_to_next_shortest_segment": round(
            heading_error_to_descent(
                field, float(end[0]), float(end[1]), float(end[2])
            ),
            decimals,
        ),
        "connectivity": math.isfinite(end_geo), "route_family": spec["family"],
        "immediate_contact": h1_contact, "committed_prefix_contact": bool(outcome["contact"]),
        "successor_viability": viable, "successor_safe_action_count": successor_count,
        "oracle_viability_admissible": bool(not outcome["contact"] and viable),
        "stuck": bool(stuck),
        "dead_end": bool(
            end_geo
            >= start_geo - CONTRACT.DEAD_END_GEODESIC_NO_PROGRESS_TOLERANCE_M
            and end_euclid
            < start_euclid - CONTRACT.DEAD_END_EUCLIDEAN_CLOSER_THRESHOLD_M
        ),
        "physics_substeps": len(outcome["ticks"]) * int(round(COMMAND_DT_S / PHYSICS_DT_S)),
    }


def _winner(rows: Sequence[Mapping[str, Any]], key: str, *, admissible_only: bool = False) -> int:
    candidates = [row for row in rows if not admissible_only or row["oracle_viability_admissible"]]
    if not candidates:
        return -1
    if key == "geodesic_progress":
        ordered = sorted(candidates, key=lambda row: (-float(row[key]), float(row["endpoint_geodesic_distance"]), float(row["heading_error_to_next_shortest_segment"]), int(row["candidate_index"])))
    else:
        ordered = sorted(candidates, key=lambda row: (-float(row[key]), int(row["candidate_index"])))
    return int(ordered[0]["candidate_index"])


def _route_structure_visible(
    spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> bool:
    poses: list[Sequence[float]] = [spec["start"]]
    for row in rows:
        poses.extend(row["horizon_poses"][horizon] for horizon in ("H1", "H2", "H3"))
    field_of_view = math.radians(
        float(
            CONTRACT.PANEL_GEOMETRY_AUTHORITY["renderer"][
                "horizontal_fov_degrees"
            ]
        )
    )
    relevant_wall_indices: set[int] = set()
    for pose in poses:
        x, y, yaw = (float(value) for value in pose)
        for sample in range(65):
            angle = yaw + (sample / 64.0 - 0.5) * field_of_view
            distance, wall_index = _ray_distance(spec, x, y, angle)
            if wall_index >= 4 and distance < 2.0 * WORLD_HALF_M:
                relevant_wall_indices.add(wall_index)
    required_visible_parts = {
        "WALL_DETOUR": 1,
        "U_ESCAPE": 2,
        "DEAD_END_LURE": 2,
        "OFFSET_PASSAGE": 2,
    }
    return len(relevant_wall_indices) >= required_visible_parts[str(spec["family"])]


def eligible_state(spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], field: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    sx, sy, _yaw = (float(v) for v in spec["start"]); gx, gy = (float(v) for v in spec["goal"])
    start_geo = field_distance(field, sx, sy); euclid = math.hypot(gx - sx, gy - sy)
    line_obstructed = any(_segment_intersects_rect((sx, sy), (gx, gy), wall) for wall in rectangles(spec)[4:])
    admissible = [row for row in rows if row["oracle_viability_admissible"]]
    geo_best = _winner(rows, "geodesic_progress", admissible_only=True)
    direct_best = _winner(rows, "euclidean_progress", admissible_only=True)
    values = sorted({round(float(row["geodesic_progress"]), 6) for row in admissible})
    ordered_geo = sorted(admissible, key=lambda row: (-float(row["geodesic_progress"]), float(row["endpoint_geodesic_distance"]), float(row["heading_error_to_next_shortest_segment"]), int(row["candidate_index"])))
    def nonindex_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
        return (
            round(-float(row["geodesic_progress"]), 9),
            round(float(row["endpoint_geodesic_distance"]), 9),
            round(float(row["heading_error_to_next_shortest_segment"]), 9),
        )
    unique_best = len(ordered_geo) >= 2 and nonindex_key(ordered_geo[0]) != nonindex_key(ordered_geo[1])
    checks = {
        "goal_reachable": math.isfinite(start_geo),
        "direct_goal_ray_obstructed": line_obstructed,
        "shortest_path_over_euclidean_ge_1_25": start_geo >= 1.25 * euclid,
        "at_least_two_oracle_admissible": len(admissible) >= 2,
        "geodesic_top1_differs_direct_top1": geo_best != direct_best,
        "candidate_utilities_nondegenerate": len(values) >= 3,
        "route_structure_visible_current_or_h1_h3": _route_structure_visible(spec, rows),
        "best_not_candidate_index_only": unique_best,
    }
    return all(checks.values()), {
        "checks": checks, "start_geodesic_distance": start_geo, "start_euclidean_distance": euclid,
        "path_ratio": start_geo / euclid if euclid > 0 else math.inf,
        "admissible_count": len(admissible), "geodesic_top1": geo_best, "direct_top1": direct_best,
        "oracle_best": _winner(rows, "geodesic_progress", admissible_only=True),
    }


def build_prior_scene_exclusion() -> dict[str, Any]:
    sources = {
        ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json": "da67309c073f60d74e4b85427237b19691552a542136e6ddb95939f14b4c5c37",
        ".generated/go2_oracle_branch_pilot_v1_2/state_manifest.json": "1f76afa94a66eaec0049559f9a47d48a4b50543c0ad4c6cec5060ff5b5ab0d9e",
        ".generated/go2_oracle_branch_pilot_v1/identity_manifest.json": "7ebc6aac4eed73d38dec1a6f2be8272f442522d3c43f7a1d2f491b3eafe996c6",
        "docs/lewm_go2_world_model_counterfactual_calibration_scene_panel_v1_2026-08-02.json": "ad4467e54427c661834755e8062ccea1276602eeab5d1abafa4db9ac79d78581",
        "config/go2_generalization_v4/scene_role_commitments.json": "82c4a9a382452031febb712faaa90bb52c8fc5e2fab2a33d8a7ea2d447413b75",
    }
    scene_ids: set[str] = set()
    scene_id_sha256: set[str] = set()
    records: list[dict[str, Any]] = []
    for relative, expected_sha256 in sources.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ExperimentError(f"registered predecessor scene authority drift: {relative}")
        value = load_json(path)
        def walk(item: Any) -> None:
            if isinstance(item, Mapping):
                if "scene_id" in item:
                    scene_id = str(item["scene_id"])
                    scene_ids.add(scene_id)
                    scene_id_sha256.add(hashlib.sha256(scene_id.encode("utf-8")).hexdigest())
                for _key, child in item.items():
                    walk(child)
            elif isinstance(item, list):
                for child in item:
                    walk(child)
        walk(value)
        memberships = value.get("scene_id_sha256_by_role")
        if isinstance(memberships, Mapping):
            for role, raw_hashes in memberships.items():
                if not isinstance(role, str) or not isinstance(raw_hashes, list):
                    raise ExperimentError(
                        f"malformed hashed scene-role authority: {relative}"
                    )
                for digest in raw_hashes:
                    if (
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)
                    ):
                        raise ExperimentError(
                            f"malformed scene-id SHA-256 authority: {relative}:{role}"
                        )
                    scene_id_sha256.add(digest)
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": expected_sha256})
    return attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.scene_exclusion.v1",
        "sources": records,
        "scene_ids": sorted(scene_ids),
        "scene_id_sha256": sorted(scene_id_sha256),
        "comparison": (
            "reject every new scene ID whose exact text or plain UTF-8 SHA-256 "
            "is present in the registered predecessor authorities; source digests "
            "are custody bindings and are not mislabelled as geometry digests"
        ),
    })


def build_panel() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if OUTPUT_ROOT.exists():
        raise ExperimentError("official output root already exists; this experiment is one-shot")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    recovery = verify_recovery_bundle()
    exclusion = build_prior_scene_exclusion()
    head = git("rev-parse", "HEAD")
    subject = git("show", "-s", "--format=%s", "HEAD")
    parent = git("rev-parse", "HEAD^")
    expected_subject = "Freeze non-greedy local subgoal JEPA planning experiment"
    if parent != PARENT_COMMIT or subject != expected_subject or git("status", "--porcelain", "--untracked-files=all"):
        raise ExperimentError("panel generation requires the exact clean descendant source-freeze commit")
    contract_builder = getattr(CONTRACT, "build_contract", None)
    scientific_contract = contract_builder() if callable(contract_builder) else getattr(CONTRACT, "CONTRACT", {})
    atomic_json(OUTPUT_ROOT / "contract.json", attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.execution_contract.v1",
        "source_freeze_commit": head, "source_freeze_parent": parent,
        "source_freeze_subject": subject, "scientific_contract": scientific_contract,
        "recovery_binding": recovery, "output_root": str(OUTPUT_ROOT),
        "execution_runtime": require_execution_runtime(),
    }))
    excluded_ids = set(exclusion["scene_ids"])
    excluded_id_hashes = set(exclusion["scene_id_sha256"])
    selected: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    pool_records: list[dict[str, Any]] = []
    eligible_authority_rows: list[dict[str, Any]] = []
    base_seed = int(getattr(CONTRACT, "PROCEDURAL_SEED", 2026083100))
    pool_counts_by_family: dict[str, int] = {}
    for family_index, family in enumerate(FAMILIES):
        eligible: list[tuple[str, dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
        next_ordinal = 0
        family_panel_ready = False
        while not family_panel_ready and next_ordinal < CONTRACT.PANEL_MAXIMUM_CANDIDATES_PER_FAMILY:
            block_end = min(
                next_ordinal + CONTRACT.PANEL_CANDIDATE_BLOCK_SIZE,
                CONTRACT.PANEL_MAXIMUM_CANDIDATES_PER_FAMILY,
            )
            for ordinal in range(next_ordinal, block_end):
                seed = base_seed + family_index * 100_000 + ordinal
                spec = scene_spec(family, seed)
                if (
                    spec["scene_id"] in excluded_ids
                    or hashlib.sha256(spec["scene_id"].encode("utf-8")).hexdigest()
                    in excluded_id_hashes
                ):
                    continue
                field = geodesic_field(spec)
                rows = [
                    simulate_candidate(spec, field, index)
                    for index in range(CANDIDATE_COUNT)
                ]
                ok, evidence = eligible_state(spec, rows, field)
                pool_records.append(
                    {
                        "scene_id": spec["scene_id"],
                        "family": family,
                        "seed": seed,
                        "eligible": ok,
                        "evidence": evidence,
                        "geometry_digest": spec["geometry_digest"],
                    }
                )
                if ok:
                    order = CONTRACT.deterministic_scene_hash(
                        family=family,
                        scene_id=spec["scene_id"],
                        scene_manifest_sha256=spec["geometry_digest"],
                    )
                    eligible.append((order, spec, rows, evidence))
                    eligible_authority_rows.append(
                        {
                            "family": family,
                            "scene_id": spec["scene_id"],
                            "scene_manifest_sha256": spec["geometry_digest"],
                            "procedural_seed": seed,
                        }
                    )
            next_ordinal = block_end
            ordered_preview = sorted(
                eligible,
                key=lambda item: (item[0], str(item[1]["scene_id"])),
            )[: CONTRACT.STATES_PER_FAMILY]
            if len(ordered_preview) == CONTRACT.STATES_PER_FAMILY:
                weak_or_negative = 0
                useful_initially_away = False
                for _order, _spec, preview_rows, preview_evidence in ordered_preview:
                    oracle = next(
                        row
                        for row in preview_rows
                        if row["candidate_index"] == preview_evidence["oracle_best"]
                    )
                    direct = next(
                        row
                        for row in preview_rows
                        if row["candidate_index"] == preview_evidence["direct_top1"]
                    )
                    weak_or_negative += bool(
                        float(oracle["euclidean_progress"]) <= 0.0
                        or float(oracle["euclidean_progress"])
                        < 0.5 * max(0.0, float(direct["euclidean_progress"]))
                    )
                    useful_initially_away |= any(
                        bool(candidate["oracle_viability_admissible"])
                        and float(candidate["geodesic_progress"]) > 0.0
                        and float(candidate["h1_euclidean_progress"]) < 0.0
                        for candidate in preview_rows
                    )
                family_panel_ready = (
                    weak_or_negative >= 15 and useful_initially_away
                )
        pool_counts_by_family[family] = next_ordinal
        eligible.sort(key=lambda item: (item[0], str(item[1]["scene_id"])))
        if len(eligible) < CONTRACT.STATES_PER_FAMILY or not family_panel_ready:
            raise ExperimentError(
                f"{family}: deterministic pool did not satisfy quota and panel "
                f"adequacy after {next_ordinal} candidates"
            )
        for family_slot, (_order, spec, rows, evidence) in enumerate(eligible[:24]):
            role = (
                "FIT"
                if family_slot < 16
                else "CALIBRATION"
                if family_slot < 20
                else "DEVELOPMENT_HELDOUT"
            )
            state_id = f"ngls-{family_index:02d}-{family_slot:02d}"
            spec = dict(spec, state_id=state_id, role=role, family_slot=family_slot)
            selected.append({
                "state_id": state_id, "scene_id": spec["scene_id"], "episode_id": spec["episode_id"],
                "family": family, "family_slot": family_slot, "role": role, "procedural_seed": spec["seed"],
                "geometry_digest": spec["geometry_digest"], "start": spec["start"], "goal": spec["goal"],
                "eligibility": evidence,
            })
            scene_path = SCENE_ROOT / f"{state_id}.json"
            atomic_json(scene_path, attach_digest({"schema": "non_greedy_scene_v1", **spec}))
            for row in rows:
                branch_rows.append({
                    "schema": "non_greedy_branch_row_v1", "state_id": state_id,
                    "scene_id": spec["scene_id"], "episode_id": spec["episode_id"],
                    "family": family, "role": role, "geometry_digest": spec["geometry_digest"], **row,
                })
    if len(selected) != STATE_COUNT or len(branch_rows) != STATE_COUNT * CANDIDATE_COUNT:
        raise ExperimentError("selected panel cardinality drift")
    contract_split = CONTRACT.deterministic_split_manifest(eligible_authority_rows)
    contract_rows = [
        (
            str(row["family"]),
            str(row["scene_id"]),
            str(row["state_id"]),
            str(row["split_role"]),
        )
        for row in contract_split["states"]
    ]
    runtime_rows = [
        (
            str(row["family"]),
            str(row["scene_id"]),
            str(row["state_id"]),
            str(row["role"]),
        )
        for row in selected
    ]
    if runtime_rows != contract_rows:
        raise ExperimentError("runtime split differs from the frozen pure split authority")
    for identity_field in ("state_id", "scene_id", "episode_id", "geometry_digest"):
        values = [str(row[identity_field]) for row in selected]
        if len(set(values)) != STATE_COUNT:
            raise ExperimentError(f"selected panel {identity_field} identities are not unique")
    # Deterministic role and adequacy audit after the entire eligible pool is fixed.
    family_role_counts = {
        family: {role: sum(row["family"] == family and row["role"] == role for row in selected) for role in ROLES}
        for family in FAMILIES
    }
    direct_diff = np.mean([row["eligibility"]["geodesic_top1"] != row["eligibility"]["direct_top1"] for row in selected])
    by_state = {row["state_id"]: [] for row in selected}
    for row in branch_rows:
        by_state[row["state_id"]].append(row)
    for state in selected:
        regenerated_spec = scene_spec(
            str(state["family"]), int(state["procedural_seed"])
        )
        if any(
            regenerated_spec[key] != state[key]
            for key in ("scene_id", "episode_id", "geometry_digest", "start", "goal")
        ):
            raise ExperimentError(
                f"deterministic scene regeneration drift: {state['state_id']}"
            )
        regenerated_field = geodesic_field(regenerated_spec)
        regenerated_rows = [
            simulate_candidate(regenerated_spec, regenerated_field, candidate_index)
            for candidate_index in range(CANDIDATE_COUNT)
        ]
        observed_rows = sorted(
            by_state[str(state["state_id"])],
            key=lambda row: int(row["candidate_index"]),
        )
        for regenerated, observed in zip(regenerated_rows, observed_rows):
            if any(observed.get(key) != value for key, value in regenerated.items()):
                raise ExperimentError(
                    f"deterministic route-label regeneration drift: {state['state_id']}"
                )
    nonvisual_signatures = {"base": set(), "query": set(), "anchor": set()}
    for state in selected:
        base, query, anchor, _target = _feature_arrays(
            state, by_state[str(state["state_id"])]
        )
        nonvisual_signatures["base"].add(
            hashlib.sha256(base.tobytes(order="C")).hexdigest()
        )
        nonvisual_signatures["query"].add(
            hashlib.sha256(query.tobytes(order="C")).hexdigest()
        )
        nonvisual_signatures["anchor"].add(
            hashlib.sha256(anchor.tobytes(order="C")).hexdigest()
        )
    nonvisual_counterbalance = {
        "unique_base_feature_signatures": len(nonvisual_signatures["base"]),
        "unique_query_feature_signatures": len(nonvisual_signatures["query"]),
        "unique_anchor_signatures": len(nonvisual_signatures["anchor"]),
        "base_feature_sha256": sorted(nonvisual_signatures["base"]),
        "query_feature_sha256": sorted(nonvisual_signatures["query"]),
        "anchor_sha256": sorted(nonvisual_signatures["anchor"]),
    }
    counterbalance_authority = CONTRACT.NONVISUAL_COUNTERBALANCE_AUTHORITY
    nonvisual_counterbalance["pass"] = (
        nonvisual_counterbalance["unique_base_feature_signatures"]
        == counterbalance_authority["required_unique_base_feature_signatures"]
        and nonvisual_counterbalance["unique_query_feature_signatures"]
        == counterbalance_authority["required_unique_query_feature_signatures"]
        and nonvisual_counterbalance["unique_anchor_signatures"]
        == counterbalance_authority["required_unique_anchor_signatures"]
    )
    if not nonvisual_counterbalance["pass"]:
        raise ExperimentError(
            f"registered nonvisual counterbalance failed: {nonvisual_counterbalance}"
        )
    weak_direct = []
    moved_away_families: set[str] = set()
    for state in selected:
        rows = by_state[state["state_id"]]
        oracle = next(row for row in rows if row["candidate_index"] == state["eligibility"]["oracle_best"])
        direct = next(row for row in rows if row["candidate_index"] == state["eligibility"]["direct_top1"])
        weak = float(oracle["euclidean_progress"]) <= 0 or float(oracle["euclidean_progress"]) < 0.5 * max(0.0, float(direct["euclidean_progress"]))
        weak_direct.append(weak)
        if any(
            bool(candidate["oracle_viability_admissible"])
            and float(candidate["geodesic_progress"]) > 0.0
            and float(candidate["h1_euclidean_progress"]) < 0.0
            for candidate in rows
        ):
            moved_away_families.add(state["family"])
    adequacy = {
        "reachable_goals": sum(bool(row["eligibility"]["checks"]["goal_reachable"]) for row in selected),
        "obstructed_direct_goal_rays": sum(bool(row["eligibility"]["checks"]["direct_goal_ray_obstructed"]) for row in selected),
        "two_or_more_admissible_fraction": float(np.mean([row["eligibility"]["admissible_count"] >= 2 for row in selected])),
        "direct_top1_differs_geodesic_fraction": float(direct_diff),
        "oracle_best_weak_or_negative_euclidean_fraction": float(np.mean(weak_direct)),
        "registered_nonvisual_counterbalance": nonvisual_counterbalance,
        "families_with_initial_distance_increase_example": sorted(moved_away_families),
        "no_family_single_utility_class": all(len({round(float(r["geodesic_progress"]), 6) for sid, rows in by_state.items() for r in rows if next(s for s in selected if s["state_id"] == sid)["family"] == family}) > 1 for family in FAMILIES),
        "deterministic_identity_and_label_regeneration": True,
    }
    adequacy["pass"] = (
        adequacy["reachable_goals"] == 96 and adequacy["obstructed_direct_goal_rays"] == 96
        and adequacy["two_or_more_admissible_fraction"] >= 0.90
        and adequacy["direct_top1_differs_geodesic_fraction"] >= 0.80
        and adequacy["oracle_best_weak_or_negative_euclidean_fraction"] >= 0.60
        and adequacy["registered_nonvisual_counterbalance"]["pass"]
        and set(adequacy["families_with_initial_distance_increase_example"]) == set(FAMILIES)
        and adequacy["no_family_single_utility_class"]
    )
    if not adequacy["pass"]:
        raise ExperimentError(f"panel adequacy gate failed: {adequacy}")
    panel = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.panel_manifest.v1",
        "status": "PASS", "development_only": True,
        "constructed_challenge_caveat": "This is a constructed non-greedy challenge set, not an estimate of natural task prevalence.",
        "recovery_binding": recovery, "predecessor_scene_exclusion": exclusion,
        "procedural_pool": {
            "candidate_block_size": CONTRACT.PANEL_CANDIDATE_BLOCK_SIZE,
            "maximum_candidates_per_family": (
                CONTRACT.PANEL_MAXIMUM_CANDIDATES_PER_FAMILY
            ),
            "candidates_examined_by_family": pool_counts_by_family,
            "selected_per_family": 24,
            "eligible_population": pool_records,
        },
        "states": selected, "family_role_counts": family_role_counts, "role_totals": ROLE_TOTALS,
        "candidate_bank": [{"candidate_index": i, "name": name, "primitive_blocks": list(blocks)} for i, (name, blocks) in enumerate(CANDIDATE_BANK)],
        "route_target": "G_i = d_geo(s_t,g) - d_geo(s_{t+H3}^i,g); ties by remaining geodesic distance, heading error to next shortest-path segment, then candidate index",
        "oracle_population": "ORACLE_VIABILITY_ADMISSIBLE = complete current H1-H3 committed prefix contact-free AND actual H3 successor has at least one contact-free next primitive",
        "adequacy": adequacy,
    })
    split = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.split_manifest.v1",
        "assignment": "within each family, deterministic SHA-256 order after the complete eligible population was fixed; first 16 FIT, next 4 CALIBRATION, final 4 DEVELOPMENT_HELDOUT",
        "roles": {role: [row["state_id"] for row in selected if row["role"] == role] for role in ROLES},
        "family_role_counts": family_role_counts,
        "contract_split_manifest": contract_split,
    })
    atomic_json(OUTPUT_ROOT / "panel_manifest.json", panel)
    atomic_json(OUTPUT_ROOT / "split_manifest.json", split)
    write_jsonl(OUTPUT_ROOT / "branch_ledger.jsonl", branch_rows)
    write_jsonl(OUTPUT_ROOT / "route_labels.jsonl", [
        {key: row[key] for key in (
            "schema", "state_id", "scene_id", "family", "role", "candidate_index", "candidate",
            "start_geodesic_distance", "endpoint_geodesic_distance", "geodesic_progress", "euclidean_progress",
            "h1_euclidean_progress",
            "heading_change", "connectivity", "route_family", "immediate_contact", "successor_viability",
            "committed_prefix_contact", "successor_safe_action_count",
            "oracle_viability_admissible", "stuck", "dead_end", "completed",
        )} for row in branch_rows
    ])
    return panel, branch_rows


def _ray_distance(spec: Mapping[str, Any], x: float, y: float, angle: float) -> tuple[float, int]:
    dx, dy = math.cos(angle), math.sin(angle)
    best = 20.0; best_index = 0
    for index, wall in enumerate(rectangles(spec)):
        xmin, xmax = wall.cx - wall.sx / 2, wall.cx + wall.sx / 2
        ymin, ymax = wall.cy - wall.sy / 2, wall.cy + wall.sy / 2
        candidates: list[float] = []
        if abs(dx) > 1e-12:
            for xx in (xmin, xmax):
                t = (xx - x) / dx
                yy = y + t * dy
                if t > 0 and ymin <= yy <= ymax:
                    candidates.append(t)
        if abs(dy) > 1e-12:
            for yy in (ymin, ymax):
                t = (yy - y) / dy
                xx = x + t * dx
                if t > 0 and xmin <= xx <= xmax:
                    candidates.append(t)
        if candidates and min(candidates) < best:
            best = min(candidates); best_index = index
    return best, best_index


def render_rgb(spec: Mapping[str, Any], pose: Sequence[float]) -> np.ndarray:
    """Deterministic obstacle-visible pinhole rendering at the frozen 224x168 size."""
    renderer = CONTRACT.PANEL_GEOMETRY_AUTHORITY["renderer"]
    width = int(renderer["width_pixels"])
    height = int(renderer["height_pixels"])
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[: height // 2] = np.asarray([112, 154, 190], dtype=np.uint8)
    for row in range(height // 2, height):
        shade = int(80 + 70 * (row - height // 2) / (height // 2))
        image[row] = np.asarray([shade, shade, max(60, shade - 18)], dtype=np.uint8)
    x, y, yaw = (float(v) for v in pose)
    fov = math.radians(float(renderer["horizontal_fov_degrees"]))
    for column in range(width):
        angle = yaw + ((column + 0.5) / width - 0.5) * fov
        distance, _wall_index = _ray_distance(spec, x, y, angle)
        corrected = max(0.05, distance * math.cos(angle - yaw))
        wall_height = min(height, int(95 / corrected))
        top = max(0, height // 2 - wall_height // 2)
        bottom = min(height, height // 2 + wall_height // 2)
        base = np.asarray([154.0, 132.0, 105.0], dtype=np.float64)
        illumination = max(0.35, min(1.0, 1.3 / math.sqrt(corrected + 0.2)))
        image[top:bottom, column] = np.clip(base * illumination, 0, 255).astype(np.uint8)
    return image


def render_panel() -> dict[str, Any]:
    from PIL import Image
    if RGB_ROOT.exists() or (CACHE_ROOT / "rgb_index.json").exists():
        raise ExperimentError("render stage output already exists")
    panel = load_json(OUTPUT_ROOT / "panel_manifest.json")
    branch_rows = [json.loads(line) for line in (OUTPUT_ROOT / "branch_ledger.jsonl").read_text().splitlines() if line]
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in branch_rows:
        by_state.setdefault(str(row["state_id"]), []).append(row)
    frames: list[dict[str, Any]] = []
    for index, state in enumerate(panel["states"]):
        sid = str(state["state_id"]); spec = load_json(SCENE_ROOT / f"{sid}.json")
        state_dir = RGB_ROOT / sid; state_dir.mkdir(parents=True, exist_ok=True)
        current_path = state_dir / "current.png"
        Image.fromarray(render_rgb(spec, spec["start"]), mode="RGB").save(current_path)
        frames.append({"state_id": sid, "candidate_index": None, "horizon": "CURRENT", "path": str(current_path), "sha256": sha256_file(current_path)})
        for row in sorted(by_state[sid], key=lambda item: int(item["candidate_index"])):
            for horizon in ("H1", "H2", "H3"):
                path = state_dir / f"candidate_{int(row['candidate_index']):02d}_{horizon.lower()}.png"
                Image.fromarray(render_rgb(spec, row["horizon_poses"][horizon]), mode="RGB").save(path)
                frames.append({"state_id": sid, "candidate_index": int(row["candidate_index"]), "horizon": horizon, "path": str(path), "sha256": sha256_file(path)})
        print(json.dumps({"stage": "render", "state": index + 1, "total": len(panel["states"]), "state_id": sid}), flush=True)
    receipt = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.rgb_index.v1",
        "renderer": "deterministic obstacle-visible pinhole renderer over the exact frozen axis-aligned scene geometry",
        "input_size": [168, 224], "goal_marker_rendered": False, "frames": frames,
    })
    atomic_json(CACHE_ROOT / "rgb_index.json", receipt)
    return receipt


TOKEN_SHAPE = (768, 1024)


def _valid_token(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != int(np.prod(TOKEN_SHAPE)) * 2:
        return False
    value = np.memmap(path, mode="r", dtype=np.float16, shape=TOKEN_SHAPE)
    return bool(np.isfinite(value).all())


def encode_panel(batch_size: int = 8) -> dict[str, Any]:
    import torch
    from scripts.dev_frozen_dense_representation_encoders_v1 import (
        VJepa21Arm, preprocessing_hash,
    )
    if LATENT_ROOT.exists() or (OUTPUT_ROOT / "latent_index.json").exists():
        raise ExperimentError("encode stage output already exists")

    rgb_index = load_json(CACHE_ROOT / "rgb_index.json")
    checkpoint = Path.home() / ".cache/vjepa2_1_vitl_dist_vitG_384.pt"
    expected = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
    if sha256_file(checkpoint) != expected:
        raise ExperimentError("frozen V-JEPA encoder checkpoint binding mismatch")
    unique: dict[str, dict[str, Any]] = {}
    occurrences: list[dict[str, Any]] = []
    for frame in rgb_index["frames"]:
        digest = str(frame["sha256"])
        if sha256_file(Path(frame["path"])) != digest:
            raise ExperimentError(f"RGB frame binding drift: {frame['path']}")
        unique.setdefault(digest, {"rgb_path": frame["path"], "rgb_sha256": digest})
        occurrences.append({
            "state_id": frame["state_id"], "candidate_index": frame["candidate_index"],
            "horizon": frame["horizon"], "rgb_sha256": digest,
        })
    arm = VJepa21Arm()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arm.build(device, torch.float32)
    pending: list[tuple[str, dict[str, Any], Path]] = []
    for digest, record in sorted(unique.items()):
        path = LATENT_ROOT / digest[:2] / f"{digest}.f16"
        record["token_path"] = str(path)
        if not _valid_token(path):
            pending.append((digest, record, path))
    started = time.time(); peak = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        pixels = torch.stack([arm.preprocess(item[1]["rgb_path"]) for item in batch]).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            values = arm.tokens(pixels).float().cpu().numpy().astype(np.float16)
        for (_digest, _record, path), value in zip(batch, values):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            np.ascontiguousarray(value).tofile(temporary)
            os.replace(temporary, path)
            if not _valid_token(path):
                raise ExperimentError(f"invalid V-JEPA token shard: {path}")
        if device.type == "cuda":
            peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
        print(json.dumps({"stage": "encode", "encoded": min(offset + len(batch), len(pending)), "total": len(pending)}), flush=True)
    for record in unique.values():
        path = Path(record["token_path"])
        record.update({"token_sha256": sha256_file(path), "shape": list(TOKEN_SHAPE), "dtype": "float16"})
    for occurrence in occurrences:
        occurrence["token_sha256"] = unique[occurrence["rgb_sha256"]]["token_sha256"]
        occurrence["token_path"] = unique[occurrence["rgb_sha256"]]["token_path"]
    index = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.latent_index.v1",
        "complete": True, "encoder_checkpoint_sha256": expected,
        "encoder_constructor": "vjepa2_1_vit_large_384", "preprocessing_digest": preprocessing_hash(arm),
        "preprocessing": "224x168 RGB, bicubic 512x384, ImageNet normalization, full field of view",
        "token_contract": {"shape": list(TOKEN_SHAPE), "grid": [24, 32], "storage_order": "row-major", "dtype": "float16"},
        "device": str(device), "batch_size": batch_size, "unique_frames": len(unique),
        "frame_occurrences": len(occurrences), "runtime_s": time.time() - started,
        "peak_vram_bytes": peak, "cache_bytes": sum(Path(row["token_path"]).stat().st_size for row in unique.values()),
        "records": [unique[key] for key in sorted(unique)], "occurrences": occurrences,
    })
    atomic_json(OUTPUT_ROOT / "latent_index.json", index)
    del arm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return index


def _load_token(path: str) -> np.ndarray:
    value = np.memmap(path, mode="r", dtype=np.float16, shape=TOKEN_SHAPE)
    if not np.isfinite(value).all():
        raise ExperimentError(f"non-finite token shard: {path}")
    return np.asarray(value, dtype=np.float32)


def _occurrence_index(
    latent_index: Mapping[str, Any],
    *,
    state_ids: set[str] | None = None,
) -> dict[tuple[str, int | None, str], str]:
    result: dict[tuple[str, int | None, str], str] = {}
    for row in latent_index["occurrences"]:
        if state_ids is not None and str(row["state_id"]) not in state_ids:
            continue
        key = (str(row["state_id"]), None if row["candidate_index"] is None else int(row["candidate_index"]), str(row["horizon"]))
        if key in result:
            raise ExperimentError(f"duplicate latent occurrence {key}")
        if sha256_file(Path(row["token_path"])) != row["token_sha256"]:
            raise ExperimentError(f"latent shard hash drift: {row['token_path']}")
        result[key] = str(row["token_path"])
    return result


def _feature_arrays(state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sx, sy, syaw = (float(v) for v in state["start"]); gx, gy = (float(v) for v in state["goal"])
    dx, dy = gx - sx, gy - sy
    c, s = math.cos(syaw), math.sin(syaw)
    body_dx, body_dy = c * dx + s * dy, -s * dx + c * dy
    goal_heading = math.atan2(body_dy, body_dx)
    goal_features = np.asarray([body_dx, body_dy, math.sin(goal_heading), math.cos(goal_heading)], dtype=np.float32)
    base_rows: list[np.ndarray] = []; query_rows: list[np.ndarray] = []
    anchors: list[float] = []
    ordered_rows = sorted(rows, key=lambda item: int(item["candidate_index"]))
    decimals = int(
        CONTRACT.GEODESIC_TARGET_AUTHORITY[
            "persisted_numeric_quantization_decimal_places"
        ]
    )
    raw_targets = [
        round(float(row["geodesic_progress"]), decimals) for row in ordered_rows
    ]
    positive_gaps = sorted(
        left - right
        for left in set(raw_targets)
        for right in set(raw_targets)
        if left > right
    )
    tie_unit = min(1.0e-7, positive_gaps[0] / (4 * (CANDIDATE_COUNT + 1))) if positive_gaps else 1.0e-7
    if tie_unit <= 1.0e-12:
        raise ExperimentError("geodesic target separation is too small for exact tie encoding")
    targets = list(raw_targets)
    for value in sorted(set(raw_targets)):
        tied = [index for index, target in enumerate(raw_targets) if target == value]
        tied.sort(
            key=lambda index: (
                float(ordered_rows[index]["endpoint_geodesic_distance"]),
                float(ordered_rows[index]["heading_error_to_next_shortest_segment"]),
                int(ordered_rows[index]["candidate_index"]),
            )
        )
        for reverse_rank, index in enumerate(reversed(tied)):
            targets[index] += reverse_rank * tie_unit
    for row in ordered_rows:
        requested = np.asarray(row["requested_commands"], dtype=np.float32).reshape(15, 3)
        applied = np.asarray(row["post_slew_applied_commands"], dtype=np.float32).reshape(15, 3)
        previous = np.zeros(3, dtype=np.float32)
        history = np.zeros((15, 2), dtype=np.float32)
        endpoint = np.asarray([
            float(row["successor_pose"][0]) - sx,
            float(row["successor_pose"][1]) - sy,
            float(row["heading_change"]),
        ], dtype=np.float32)
        base = np.concatenate([
            goal_features, requested.reshape(-1), applied.reshape(-1), previous,
            history.reshape(-1), np.asarray([float(row["euclidean_progress"])], dtype=np.float32), endpoint,
        ])
        query = np.concatenate([
            goal_features, applied[:, (0, 2)].reshape(-1), previous[[0, 2]], history.reshape(-1),
        ])
        if base.size != 131 or query.size != 66:
            raise ExperimentError(f"feature width drift: {base.size}/{query.size}")
        base_rows.append(base); query_rows.append(query)
        anchors.append(float(row["euclidean_progress"]))
    return (
        np.stack(base_rows),
        np.stack(query_rows),
        np.asarray(anchors, dtype=np.float32),
        np.asarray(targets, dtype=np.float64),
    )


def _parameter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_panel_rows() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    panel = load_json(OUTPUT_ROOT / "panel_manifest.json")
    states = {str(row["state_id"]): row for row in panel["states"]}
    by_state: dict[str, list[dict[str, Any]]] = {sid: [] for sid in states}
    for line in (OUTPUT_ROOT / "branch_ledger.jsonl").read_text().splitlines():
        if line:
            row = json.loads(line); by_state[str(row["state_id"])].append(row)
    if any(len(rows) != CANDIDATE_COUNT for rows in by_state.values()):
        raise ExperimentError("branch ledger state cardinality drift")
    return panel, states, by_state


def _build_rankers() -> dict[str, Any]:
    if hasattr(CONTRACT, "build_matched_rankers"):
        built = CONTRACT.build_matched_rankers(seed=2026083101)
        if isinstance(built, Mapping):
            return dict(built)
        names = ("NO_LATENT_NON_GREEDY_RANKER", "CURRENT_VISUAL_REACTIVE_RANKER", "TRUE_FUTURE_JEPA_TRAJECTORY_RANKER")
        return dict(zip(names, built))
    raise ExperimentError("contract module does not expose build_matched_rankers")


def _model_score(model: Any, model_id: str, base: Any, query: Any, anchor: Any, current: Any, future: Any) -> Any:
    if current is not None and current.ndim == 2:
        current = current.unsqueeze(0).expand(base.shape[0], -1, -1)
    if model_id == "NO_LATENT_NON_GREEDY_RANKER":
        output = model(
            base_features=base, query_features=query, kinematic_anchor=anchor,
            current_tokens=None, future_tokens=None,
        )
        return output.score if hasattr(output, "score") else output
    if model_id == "CURRENT_VISUAL_REACTIVE_RANKER":
        output = model(
            base_features=base, query_features=query, kinematic_anchor=anchor,
            current_tokens=current, future_tokens=None,
        )
        return output.score if hasattr(output, "score") else output
    output = model(
        base_features=base, query_features=query, kinematic_anchor=anchor,
        current_tokens=current, future_tokens=future,
    )
    return output.score if hasattr(output, "score") else output


def _route_loss(model: Any, model_id: str, base: Any, query: Any, anchor: Any, current: Any, future: Any, targets: Any, mask: Any) -> Any:
    scores = _model_score(model, model_id, base, query, anchor, current, future)
    import torch
    active = torch.nonzero(mask, as_tuple=False).flatten()
    if active.numel() < 2:
        return None
    active_scores = scores[active]; active_targets = targets[active]
    if hasattr(CONTRACT, "matched_ranker_loss"):
        components = CONTRACT.matched_ranker_loss(
            scores=active_scores,
            target_utilities=active_targets,
            state_indices=torch.zeros_like(active, dtype=torch.int64),
            residuals=active_scores - anchor[active],
        )
        return components["total"]
    diff = active_targets[:, None] - active_targets[None, :]
    labels = torch.sign(diff); score_diff = active_scores[:, None] - active_scores[None, :]
    ordered = labels != 0
    pair = torch.nn.functional.softplus(-labels[ordered] * score_diff[ordered]).mean()
    listwise = -(torch.softmax(active_targets, dim=0) * torch.log_softmax(active_scores, dim=0)).sum()
    residual = ((active_scores - anchor[active]) ** 2).mean()
    return pair + 0.5 * listwise + 1e-3 * residual


def _state_latents(
    state_id: str,
    occurrence: Mapping[tuple[str, int | None, str], str],
    *,
    candidates: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    current = _load_token(occurrence[(state_id, None, "CURRENT")])
    indices = list(range(CANDIDATE_COUNT)) if candidates is None else [int(value) for value in candidates]
    future = np.stack([
        np.stack([_load_token(occurrence[(state_id, candidate, f"H{h}")]) for h in (1, 2, 3)])
        for candidate in indices
    ])
    return current, future


def train_rankers() -> dict[str, Any]:
    import torch

    if any(
        path.exists()
        for path in (
            OUTPUT_ROOT / "checkpoints",
            OUTPUT_ROOT / "training_ledger.jsonl",
            CACHE_ROOT / "training_receipt.json",
        )
    ):
        raise ExperimentError("training stage output already exists")

    panel, states, by_state = _load_panel_rows()
    fit_ids = sorted(row["state_id"] for row in panel["states"] if row["role"] == "FIT")
    if len(fit_ids) != 64:
        raise ExperimentError("fit split cardinality drift")
    latent_index = load_json(OUTPUT_ROOT / "latent_index.json")
    occurrence = _occurrence_index(latent_index, state_ids=set(fit_ids))
    rankers = _build_rankers()
    expected_ids = {
        "NO_LATENT_NON_GREEDY_RANKER",
        "CURRENT_VISUAL_REACTIVE_RANKER",
        "TRUE_FUTURE_JEPA_TRAJECTORY_RANKER",
    }
    if set(rankers) != expected_ids:
        raise ExperimentError(f"ranker identity drift: {sorted(rankers)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in rankers.values():
        model.to(device=device, dtype=torch.float32)
        model.train()
    counts = {name: sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) for name, model in rankers.items()}
    if any(count >= 500_000 for count in counts.values()):
        raise ExperimentError(f"ranker parameter cap exceeded: {counts}")
    initial = {name: _parameter_digest(model) for name, model in rankers.items()}
    optimizer_arguments = CONTRACT.TRAINING_POLICY["optimizer_arguments"]
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=CONTRACT.TRAINING_POLICY["learning_rate"],
            weight_decay=CONTRACT.TRAINING_POLICY["weight_decay"],
            betas=tuple(optimizer_arguments["betas"]),
            eps=optimizer_arguments["eps"],
            amsgrad=optimizer_arguments["amsgrad"],
            maximize=optimizer_arguments["maximize"],
            foreach=optimizer_arguments["foreach"],
            fused=optimizer_arguments["fused"],
            capturable=optimizer_arguments["capturable"],
            differentiable=optimizer_arguments["differentiable"],
        )
        for name, model in rankers.items()
    }
    training_rows: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, 61):
        order = sorted(fit_ids, key=lambda sid: hashlib.sha256(f"2026083101:{epoch}:{sid}".encode()).hexdigest())
        epoch_sums = {name: {"loss": 0.0, "steps": 0} for name in rankers}
        for state_index, sid in enumerate(order):
            rows = sorted(by_state[sid], key=lambda item: int(item["candidate_index"]))
            active = [index for index, row in enumerate(rows) if bool(row["oracle_viability_admissible"])]
            if len(active) < 2:
                continue
            base_np, query_np, anchor_np, target_np = _feature_arrays(states[sid], rows)
            base = torch.from_numpy(base_np[active]).to(device)
            query = torch.from_numpy(query_np[active]).to(device)
            anchor = torch.from_numpy(anchor_np[active]).to(device)
            targets = torch.from_numpy(target_np[active]).to(device)
            mask = torch.ones(len(active), dtype=torch.bool, device=device)
            # Latents are opened only after panel adequacy passed and only for the
            # registered fit state currently being optimized.
            current_np, future_np = _state_latents(sid, occurrence, candidates=active)
            current = torch.from_numpy(current_np).to(device)
            future = torch.from_numpy(future_np).to(device)
            for name, model in rankers.items():
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                loss = _route_loss(model, name, base, query, anchor, current, future, targets, mask)
                if loss is None or not torch.isfinite(loss):
                    raise ExperimentError(f"non-finite route loss {name}:{sid}:epoch{epoch}")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise ExperimentError(f"non-finite ranker gradient {name}:{sid}:epoch{epoch}")
                optimizer.step()
                epoch_sums[name]["loss"] += float(loss.detach().cpu())
                epoch_sums[name]["steps"] += 1
            if state_index % 8 == 0:
                print(json.dumps({"stage": "train", "epoch": epoch, "state": state_index + 1, "states": len(order)}), flush=True)
        for name in sorted(rankers):
            steps = int(epoch_sums[name]["steps"])
            training_rows.append({
                "schema": "non_greedy_training_epoch_v1", "model_id": name,
                "epoch": epoch, "optimizer_steps": steps,
                "loss": epoch_sums[name]["loss"] / steps if steps else None,
                "pairwise_weight": 1.0, "listwise_weight": 0.5, "residual_l2_weight": 1e-3,
                "fit_states_total": len(fit_ids),
            })
    checkpoint_records: dict[str, dict[str, Any]] = {}
    checkpoint_dir = OUTPUT_ROOT / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name, model in rankers.items():
        model.eval()
        path = checkpoint_dir / f"{name.lower()}_final_epoch_060.pt"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        torch.save({
            "schema": "non_greedy_ranker_checkpoint_v1", "model_id": name,
            "seed": 2026083101, "epoch": 60, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizers[name].state_dict(),
            "parameter_count": counts[name], "initial_parameter_digest": initial[name],
            "final_parameter_digest": _parameter_digest(model),
        }, temporary)
        os.replace(temporary, path)
        checkpoint_records[name] = {
            "path": str(path.relative_to(OUTPUT_ROOT)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "parameter_count": counts[name],
            "initial_parameter_digest": initial[name], "final_parameter_digest": _parameter_digest(model),
            "epoch": 60,
        }
    write_jsonl(OUTPUT_ROOT / "training_ledger.jsonl", training_rows)
    receipt = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.training_receipt.v1",
        "seed_family": 2026083101, "optimizer": "AdamW", "learning_rate": 1e-3,
        "weight_decay": 1e-4, "epochs": 60, "final_epoch_only": True,
        "optimizer_arguments": optimizer_arguments,
        "runtime": {
            **require_execution_runtime(),
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
        "no_sweep": True, "no_second_seed": True, "fit_states": fit_ids,
        "calibration_or_heldout_used_in_training": False,
        "parameter_counts": counts, "initial_parameter_digests": initial,
        "checkpoints": checkpoint_records,
        "final_losses": {row["model_id"]: row for row in training_rows if row["epoch"] == 60},
        "runtime_s": time.time() - started,
    })
    atomic_json(CACHE_ROOT / "training_receipt.json", receipt)
    del rankers, optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt


def _load_rankers_from_checkpoints(device: Any) -> dict[str, Any]:
    import torch
    receipt = load_json(CACHE_ROOT / "training_receipt.json")
    rankers = _build_rankers()
    for name, model in rankers.items():
        record = receipt["checkpoints"][name]
        path = OUTPUT_ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ExperimentError(f"ranker checkpoint hash drift: {name}")
        value = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(value["model_state_dict"], strict=True)
        model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if _parameter_digest(model) != record["final_parameter_digest"]:
            raise ExperimentError(f"ranker parameter digest drift: {name}")
    return rankers


def _derangement(state_id: str) -> list[int]:
    shift = 1 + int(hashlib.sha256(f"trajectory:{state_id}".encode()).hexdigest()[:8], 16) % 11
    return [(index + shift) % 12 for index in range(12)]


def _time_permutation(state_id: str, candidate: int) -> list[int]:
    choices = ([1, 2, 0], [2, 0, 1], [2, 1, 0], [1, 0, 2])
    index = int(hashlib.sha256(f"time:{state_id}:{candidate}".encode()).hexdigest()[:8], 16) % len(choices)
    return list(choices[index])


def _score_rows_for_state(
    sid: str, state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    rankers: Mapping[str, Any], occurrence: Mapping[tuple[str, int | None, str], str], device: Any,
) -> list[dict[str, Any]]:
    import torch
    ordered = sorted(rows, key=lambda item: int(item["candidate_index"]))
    base_np, query_np, anchor_np, _targets = _feature_arrays(state, ordered)
    current_np, future_np = _state_latents(sid, occurrence)
    base = torch.from_numpy(base_np).to(device); query = torch.from_numpy(query_np).to(device)
    anchor = torch.from_numpy(anchor_np).to(device); current = torch.from_numpy(current_np).to(device)
    future = torch.from_numpy(future_np).to(device)
    scores: dict[str, np.ndarray] = {"DETERMINISTIC_KINEMATICS": anchor_np.astype(np.float64)}
    with torch.inference_mode():
        for name, model in rankers.items():
            scores[name] = _model_score(model, name, base, query, anchor, current, future).float().cpu().numpy().astype(np.float64)
        true_model = rankers["TRUE_FUTURE_JEPA_TRAJECTORY_RANKER"]
        mapping = _derangement(sid)
        scores["FUTURE_TRAJECTORY_DERANGEMENT"] = _model_score(
            true_model, "TRUE_FUTURE_JEPA_TRAJECTORY_RANKER", base, query, anchor, current, future[mapping]
        ).float().cpu().numpy().astype(np.float64)
        time_future = torch.stack([future[index, _time_permutation(sid, index)] for index in range(12)])
        scores["FUTURE_TIME_ORDER_DERANGEMENT"] = _model_score(
            true_model, "TRUE_FUTURE_JEPA_TRAJECTORY_RANKER", base, query, anchor, current, time_future
        ).float().cpu().numpy().astype(np.float64)
    output: list[dict[str, Any]] = []
    expected_pair_by_model = dict(CONTRACT.STAGE_A_MODEL_SOURCE_PAIRS)
    if set(scores) != set(expected_pair_by_model):
        raise ExperimentError("Stage-A model/source authority drift")
    for model_id, source_id in CONTRACT.STAGE_A_MODEL_SOURCE_PAIRS:
        values = np.asarray(scores[model_id])
        if values.shape != (CANDIDATE_COUNT,) or not np.isfinite(values).all():
            raise ExperimentError(f"invalid Stage-A score vector: {sid}:{model_id}")
        for row, score in zip(ordered, values):
            output.append({
                "stage_id": "STAGE_A", "state_id": sid, "family": state["family"],
                "split_role": "DEVELOPMENT_HELDOUT", "candidate_index": int(row["candidate_index"]),
                "model_id": model_id, "source_id": source_id, "score": float(score),
                "geodesic_progress_m": float(row["geodesic_progress"]),
                "remaining_geodesic_m": float(row["endpoint_geodesic_distance"]),
                "heading_error_to_next_shortest_segment_rad": float(
                    row["heading_error_to_next_shortest_segment"]
                ),
                "euclidean_progress_m": float(row["euclidean_progress"]),
                "oracle_admissible": bool(row["oracle_viability_admissible"]),
                "immediate_contact": bool(row["immediate_contact"]),
                "committed_prefix_contact": bool(row["committed_prefix_contact"]),
                "successor_viable": bool(row["successor_viability"]),
                "stuck": bool(row["stuck"]),
                "dead_end": bool(row["dead_end"]),
                "completed": bool(row["completed"]),
            })
    return output


def evaluate_stage_a() -> dict[str, Any]:
    import torch
    if (OUTPUT_ROOT / "stage_a_scores.jsonl").exists() or (CACHE_ROOT / "stage_a_metrics.json").exists():
        raise ExperimentError("Stage-A output already exists")
    _panel, states, by_state = _load_panel_rows()
    heldout = sorted(sid for sid, state in states.items() if state["role"] == "DEVELOPMENT_HELDOUT")
    occurrence = _occurrence_index(
        load_json(OUTPUT_ROOT / "latent_index.json"), state_ids=set(heldout)
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rankers = _load_rankers_from_checkpoints(device)
    if len(heldout) != 16:
        raise ExperimentError("development heldout cardinality drift")
    score_rows: list[dict[str, Any]] = []
    for index, sid in enumerate(heldout):
        score_rows.extend(_score_rows_for_state(sid, states[sid], by_state[sid], rankers, occurrence, device))
        print(json.dumps({"stage": "stage_a", "state": index + 1, "total": len(heldout), "state_id": sid}), flush=True)
    if len(score_rows) != 16 * CANDIDATE_COUNT * 6:
        raise ExperimentError("Stage-A score-row cardinality drift")
    write_jsonl(OUTPUT_ROOT / "stage_a_scores.jsonl", score_rows)
    stage_a_reducer = getattr(METRICS, "recompute_stage_a_metrics_and_gate", None)
    if not callable(stage_a_reducer):
        raise ExperimentError("metrics module lacks the frozen Stage-A reducer")
    metrics = stage_a_reducer(score_rows)
    metrics = dict(metrics)
    atomic_json(CACHE_ROOT / "stage_a_metrics.json", metrics)
    del rankers
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def _stage_a_pass(metrics: Mapping[str, Any]) -> bool:
    decision = metrics.get("decision")
    if not isinstance(decision, Mapping) or type(
        decision.get("stage_a_authorizes_predictor_substitution")
    ) is not bool:
        raise ExperimentError("metrics lack the exact Stage-A authorization decision")
    return bool(decision["stage_a_authorizes_predictor_substitution"])


def _predictor_parameter_digest(model: Any) -> str:
    return _parameter_digest(model)


def _load_frozen_predictor(source: str, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from scripts import dev_proprio_predictor_v1 as P

    bindings = {
        "R1": (
            Path.home() / ".cache/lewm_go2_temporal_v03/factorial_v1/seed_2026080901/seed_2026080901_rgb_one_step_epoch21.pt",
            "20b6e3fa2a2d3c3ec2c20ea37e524f9c2872fdcfd5226b114822efa26872261a",
            "rgb_one_step", False,
        ),
        "RR": (
            Path.home() / ".cache/lewm_go2_temporal_v03/factorial_v1/seed_2026080901/seed_2026080901_rgb_rollout_epoch21.pt",
            "75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4",
            "rgb_rollout", True,
        ),
    }
    path, expected_sha, cell, rollout = bindings[source]
    if sha256_file(path) != expected_sha:
        raise ExperimentError(f"frozen predictor checkpoint drift: {source}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_config = {"cell": cell, "use_proprio": False, "rollout": rollout, "width": 384}
    if checkpoint.get("model_config") != expected_config:
        raise ExperimentError(f"frozen predictor model config drift: {source}")
    model = P.build_paired(2026080901, use_proprio=False, width=384, depth=6, heads=6)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    del checkpoint
    return model, {
        "source": source, "path": str(path), "sha256": expected_sha,
        "model_config": expected_config, "seed": 2026080901,
        "parameter_digest_before": _predictor_parameter_digest(model),
    }


def materialize_predictors() -> dict[str, Any]:
    import torch
    from scripts import dev_proprio_predictor_v1 as P
    from scripts import run_dev_v03_temporal_action_jepa_v1 as T

    if (OUTPUT_ROOT / "predictor_index.json").exists() or PREDICTOR_ROOT.exists():
        raise ExperimentError("predictor-substitution stage output already exists")

    stage_a_metrics = load_json(CACHE_ROOT / "stage_a_metrics.json")
    if not _stage_a_pass(stage_a_metrics):
        conditional = OUTPUT_ROOT / "conditional_stage_b_scores.jsonl"
        if conditional.exists():
            raise ExperimentError(
                "conditional Stage-B evidence exists after a failed Stage-A gate"
            )
        index = attach_digest({
            "schema": "non_greedy_local_subgoal_jepa_planning_v1.predictor_index.v1",
            "executed": False, "reason": "STAGE_A_TRUE_FUTURE_SIGNAL_OR_INCREMENTAL_VALUE_GATE_FAILED",
            "predictor_training_executed": False, "records": [],
        })
        atomic_json(OUTPUT_ROOT / "predictor_index.json", index)
        return index
    panel, states, by_state = _load_panel_rows()
    del panel
    heldout = sorted(sid for sid, state in states.items() if state["role"] == "DEVELOPMENT_HELDOUT")
    occurrence = _occurrence_index(
        load_json(OUTPUT_ROOT / "latent_index.json"), state_ids=set(heldout)
    )
    device = torch.device("cuda:0")
    records: list[dict[str, Any]] = []; custody: dict[str, Any] = {}
    started = time.time()
    for source in ("R1", "RR"):
        model, source_custody = _load_frozen_predictor(source, device)
        for state_index, sid in enumerate(heldout):
            current = torch.from_numpy(_load_token(occurrence[(sid, None, "CURRENT")])).to(device=device, dtype=torch.float32)
            # The task supplies one current image.  The frozen predictor expects
            # a three-frame context, so the exact current encoding is repeated;
            # this is a prospectively frozen missing-history convention.
            context = torch.stack([current, current, current], dim=0)
            context = T.normalise(context).unsqueeze(0).repeat(12, 1, 1, 1)
            ordered = sorted(by_state[sid], key=lambda item: int(item["candidate_index"]))
            active = np.asarray([
                np.asarray(row["post_slew_applied_commands"], dtype=np.float32).reshape(3, 5, 3)[:, :, (0, 2)].reshape(3, 10)
                for row in ordered
            ], dtype=np.float32)
            if active.shape != (12, 3, 10):
                raise ExperimentError(f"predictor action tensor shape drift: {sid}")
            action_blocks = [torch.from_numpy(active[:, horizon]).to(device) for horizon in range(3)]
            control = torch.zeros((12, 3, 5, 2), dtype=torch.float32, device=device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted = P.unroll(model, context, action_blocks, control=control, max_h=3)
            if len(predicted) != 3:
                raise ExperimentError(f"predictor horizon count drift: {source}:{sid}")
            value = np.stack([tensor.float().cpu().numpy().astype(np.float16) for tensor in predicted], axis=1)
            if value.shape != (12, 3, 768, 1024) or not np.isfinite(value).all():
                raise ExperimentError(f"predictor output shape/value drift: {source}:{sid}")
            path = PREDICTOR_ROOT / source.lower() / f"{sid}.f16"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            value.tofile(temporary); os.replace(temporary, path)
            records.append({
                "state_id": sid, "family": states[sid]["family"], "role": states[sid]["role"],
                "source_id": source, "path": str(path), "bytes": path.stat().st_size,
                "sha256": sha256_file(path), "shape": list(value.shape), "dtype": "float16",
                "context_convention": "three exact repeated CURRENT target-encoder token grids",
            })
            print(json.dumps({"stage": "predictor", "source": source, "state": state_index + 1, "total": len(heldout)}), flush=True)
        after = _predictor_parameter_digest(model)
        if after != source_custody["parameter_digest_before"]:
            raise ExperimentError(f"frozen predictor parameters changed: {source}")
        source_custody.update({"parameter_digest_after": after, "parameters_unchanged": True})
        custody[source] = source_custody
        del model
        torch.cuda.empty_cache()
    index = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.predictor_index.v1",
        "executed": True, "predictor_training_executed": False,
        "sources": custody, "records": records, "runtime_s": time.time() - started,
    })
    atomic_json(OUTPUT_ROOT / "predictor_index.json", index)
    return index


def _load_prediction(record: Mapping[str, Any]) -> np.ndarray:
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ExperimentError(f"predictor tensor drift: {path}")
    shape = tuple(int(value) for value in record["shape"])
    value = np.memmap(path, mode="r", dtype=np.float16, shape=shape)
    if shape != (12, 3, 768, 1024) or not np.isfinite(value).all():
        raise ExperimentError(f"predictor tensor invalid: {path}")
    return np.asarray(value, dtype=np.float32)


def evaluate_stage_b() -> dict[str, Any]:
    import torch
    if (OUTPUT_ROOT / "metrics.json").exists() or (OUTPUT_ROOT / "conditional_stage_b_scores.jsonl").exists():
        raise ExperimentError("conditional Stage-B output already exists")
    stage_a_rows = [json.loads(line) for line in (OUTPUT_ROOT / "stage_a_scores.jsonl").read_text().splitlines() if line]
    predictor_index = load_json(OUTPUT_ROOT / "predictor_index.json")
    if predictor_index.get("executed") is not True:
        metrics = METRICS.recompute_metrics_and_gates(stage_a_rows, None)
        atomic_json(OUTPUT_ROOT / "metrics.json", metrics)
        return metrics
    _panel, states, by_state = _load_panel_rows()
    device = torch.device("cuda:0")
    rankers = _load_rankers_from_checkpoints(device)
    model = rankers["TRUE_FUTURE_JEPA_TRAJECTORY_RANKER"]
    record_index = {(row["state_id"], row["source_id"]): row for row in predictor_index["records"]}
    heldout = sorted(sid for sid, state in states.items() if state["role"] == "DEVELOPMENT_HELDOUT")
    occurrence = _occurrence_index(
        load_json(OUTPUT_ROOT / "latent_index.json"), state_ids=set(heldout)
    )
    if set(record_index) != {(sid, source) for sid in heldout for source in ("R1", "RR")}:
        raise ExperimentError("predictor substitution identity coverage drift")
    stage_b_rows: list[dict[str, Any]] = []
    for state_index, sid in enumerate(heldout):
        ordered = sorted(by_state[sid], key=lambda item: int(item["candidate_index"]))
        base_np, query_np, anchor_np, _target = _feature_arrays(states[sid], ordered)
        base = torch.from_numpy(base_np).to(device); query = torch.from_numpy(query_np).to(device)
        anchor = torch.from_numpy(anchor_np).to(device)
        current = torch.from_numpy(_load_token(occurrence[(sid, None, "CURRENT")])).to(device)
        for source in ("R1", "RR"):
            future = torch.from_numpy(_load_prediction(record_index[(sid, source)])).to(device)
            with torch.inference_mode():
                scores = _model_score(model, "TRUE_FUTURE_JEPA_TRAJECTORY_RANKER", base, query, anchor, current, future).float().cpu().numpy()
            if scores.shape != (CANDIDATE_COUNT,) or not np.isfinite(scores).all():
                raise ExperimentError(f"invalid Stage-B score vector: {sid}:{source}")
            for row, score in zip(ordered, scores):
                stage_b_rows.append({
                    "stage_id": "CONDITIONAL_STAGE_B", "state_id": sid, "family": states[sid]["family"],
                    "split_role": "DEVELOPMENT_HELDOUT", "candidate_index": int(row["candidate_index"]),
                    "model_id": "TRUE_FUTURE_JEPA_TRAJECTORY_RANKER", "source_id": source,
                    "score": float(score), "geodesic_progress_m": float(row["geodesic_progress"]),
                    "remaining_geodesic_m": float(row["endpoint_geodesic_distance"]),
                    "heading_error_to_next_shortest_segment_rad": float(
                        row["heading_error_to_next_shortest_segment"]
                    ),
                    "euclidean_progress_m": float(row["euclidean_progress"]),
                    "oracle_admissible": bool(row["oracle_viability_admissible"]),
                    "immediate_contact": bool(row["immediate_contact"]),
                    "committed_prefix_contact": bool(row["committed_prefix_contact"]),
                    "successor_viable": bool(row["successor_viability"]),
                    "stuck": bool(row["stuck"]),
                    "dead_end": bool(row["dead_end"]),
                    "completed": bool(row["completed"]),
                })
        print(json.dumps({"stage": "stage_b", "state": state_index + 1, "total": len(heldout)}), flush=True)
    if len(stage_b_rows) != 16 * CANDIDATE_COUNT * 2:
        raise ExperimentError("conditional Stage-B score-row cardinality drift")
    write_jsonl(OUTPUT_ROOT / "conditional_stage_b_scores.jsonl", stage_b_rows)
    metrics = METRICS.recompute_metrics_and_gates(stage_a_rows, stage_b_rows)
    metrics = dict(metrics)
    atomic_json(OUTPUT_ROOT / "metrics.json", metrics)
    del rankers
    torch.cuda.empty_cache()
    return metrics


DOC_PATHS = {
    "contract": REPO_ROOT / "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_contract_2026-08-31.json",
    "fixture": REPO_ROOT / "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_fixture_2026-08-31.json",
    "output_schema": REPO_ROOT / "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_output_schema_2026-08-31.json",
    "preregistration": REPO_ROOT / "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_preregistration_2026-08-31.md",
    "source_closure": REPO_ROOT / "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_source_closure_2026-08-31.json",
}
SOURCE_PATHS = (
    "lewm/safety/non_greedy_local_subgoal_jepa_planning_metrics_v1.py",
    "lewm/safety/non_greedy_local_subgoal_jepa_planning_v1_contract.py",
    "scripts/run_non_greedy_local_subgoal_jepa_planning_v1.py",
    "scripts/evaluate_non_greedy_local_subgoal_jepa_planning_v1.py",
    "lewm/tests/test_non_greedy_local_subgoal_jepa_planning_metrics_v1.py",
    "lewm/tests/test_non_greedy_local_subgoal_jepa_planning_v1_contract.py",
    "lewm/tests/test_run_non_greedy_local_subgoal_jepa_planning_v1.py",
    "lewm/tests/test_evaluate_non_greedy_local_subgoal_jepa_planning_v1.py",
    "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "scripts/dev_proprio_predictor_v1.py",
    "scripts/run_dev_v03_temporal_action_jepa_v1.py",
    "scripts/dev_action_slew_reconstruction_v1.py",
    "scripts/build_dev_v03_proprio_action_manifest_v1.py",
    "scripts/dev_checkpoint_v1.py",
)


def build_freeze_documents() -> dict[str, Any]:
    if git("rev-parse", "HEAD") != PARENT_COMMIT:
        raise ExperimentError("freeze documents must be built on the registered parent")
    recovery = verify_recovery_bundle()
    exclusion = build_prior_scene_exclusion()
    contract_builder = getattr(CONTRACT, "build_contract", None)
    scientific_contract = contract_builder() if callable(contract_builder) else {
        key: getattr(CONTRACT, key) for key in dir(CONTRACT) if key.isupper() and not key.startswith("_")
        and isinstance(getattr(CONTRACT, key), (str, int, float, bool, tuple, list, dict, type(None)))
    }
    contract_doc = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.contract_document.v1",
        "status": "FROZEN_BEFORE_PANEL_MATERIALIZATION", "parent_commit": PARENT_COMMIT,
        "required_freeze_subject": "Freeze non-greedy local subgoal JEPA planning experiment",
        "required_result_subject": "Evaluate non-greedy local subgoal JEPA planning experiment",
        "recovery_authority": recovery, "scientific_contract": scientific_contract,
        "claims": {
            "positive_exact": "JEPA route selection under oracle admissibility.",
            "prohibited": "JEPA safety.", "development_only": True,
            "constructed_panel": "This is a constructed non-greedy challenge set, not an estimate of natural task prevalence.",
        },
    })
    fixture = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.fixture.v1",
        "predecessor_scene_exclusion": exclusion,
        "candidate_bank": [{"index": index, "name": name, "blocks": list(blocks)} for index, (name, blocks) in enumerate(CANDIDATE_BANK)],
        "primitives": {key: list(value) for key, value in PRIMITIVES.items()},
        "example_scene_digests": {family: scene_spec(family, 2026083100 + index * 100000)["geometry_digest"] for index, family in enumerate(FAMILIES)},
        "panel_counts": {
            "states": 96,
            "per_family": 24,
            "FIT": 64,
            "CALIBRATION": 16,
            "DEVELOPMENT_HELDOUT": 16,
        },
    })
    output_schema = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.output_schema.v1",
        "required_files": [
            "contract.json", "panel_manifest.json", "split_manifest.json", "branch_ledger.jsonl",
            "route_labels.jsonl", "latent_index.json", "predictor_index.json", "training_ledger.jsonl",
            "checkpoints/no_latent_non_greedy_ranker_final_epoch_060.pt",
            "checkpoints/current_visual_reactive_ranker_final_epoch_060.pt",
            "checkpoints/true_future_jepa_trajectory_ranker_final_epoch_060.pt",
            "stage_a_scores.jsonl", "metrics.json",
            "independent_regeneration_receipt.json", "result.json", "result.md",
            "file_hashes.json",
        ],
        "conditional_files": {
            "conditional_stage_b_scores.jsonl": "present exactly when the frozen Stage-A gate passes",
        },
        "report_publication_part_of_scientific_identity": False,
        "independent_reducer": "scripts/evaluate_non_greedy_local_subgoal_jepa_planning_v1.py",
    })
    prereg = """# NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1\n\nDevelopment-only, exploratory, one-seed, scene-disjoint non-greedy local-subgoal experiment.\n\n> This is a constructed non-greedy challenge set, not an estimate of natural task prevalence.\n\nThe frozen question is whether current, true-future, R1, and RR V-JEPA trajectories change route selection when immediate Euclidean progress and command kinematics are insufficient. The primary target is H3 geodesic progress. Contact and successor viability define reported oracle populations and are never learned targets. Positive wording is **JEPA route selection under oracle admissibility.** It is never **JEPA safety.**\n\nThe panel has 96 unique scenes: 24 per WALL_DETOUR, U_ESCAPE, DEAD_END_LURE, and OFFSET_PASSAGE; each family contributes 16 fit, 4 calibration, and 4 development-heldout states. The complete candidate-blind eligible pool is fixed before role assignment and before encoder/model opening.\n\nExactly three rankers use seed 2026083101, AdamW 1e-3/1e-4, 60 epochs, final epoch only, one seed. Stage A and conditional Stage B gates are the exact contract literals. No predictor is trained. No closed-loop control, safety model, memory, routing graph input, novelty, or beacon system is run.\n"""
    for path, value in ((DOC_PATHS["contract"], contract_doc), (DOC_PATHS["fixture"], fixture), (DOC_PATHS["output_schema"], output_schema)):
        atomic_json(path, value)
    atomic_bytes(DOC_PATHS["preregistration"], prereg.encode())
    rows = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ExperimentError(f"source closure path absent: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    closure = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.source_closure.v1",
        "parent_commit": PARENT_COMMIT, "rows": rows, "row_count": len(rows),
        "recovery_bundle_sha256": {row["path"].split("/")[-1]: row["sha256"] for row in recovery["files"]},
    })
    atomic_json(DOC_PATHS["source_closure"], closure)
    return {"contract": contract_doc, "fixture": fixture, "output_schema": output_schema, "source_closure": closure}


def _decision_value(metrics: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in metrics:
        return metrics[key]
    decision = metrics.get("decision")
    if isinstance(decision, Mapping) and key in decision:
        return decision[key]
    return default


def publish_result() -> dict[str, Any]:
    if any((OUTPUT_ROOT / name).exists() for name in ("result.json", "result.md", "file_hashes.json")):
        raise ExperimentError("final result output already exists")
    metrics_path = OUTPUT_ROOT / "metrics.json"
    if not metrics_path.is_file():
        raise ExperimentError("report publication requires persisted independently reducible metrics.json")
    metrics = load_json(metrics_path)
    from scripts import evaluate_non_greedy_local_subgoal_jepa_planning_v1 as REDUCER

    regeneration = REDUCER.validate_existing_regeneration_receipt(
        OUTPUT_ROOT,
        metrics_module=METRICS,
    )
    if (
        regeneration.get("pass") is not True
        or regeneration.get("content_digest") != reducer_content_digest(regeneration)
        or regeneration.get("metrics_and_gates_recomputed") is not True
        or regeneration.get("supplied_metrics_exact_byte_equal") is not True
    ):
        raise ExperimentError("independent regeneration receipt is invalid")
    inputs = regeneration.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ExperimentError("independent regeneration receipt lacks input bindings")
    for key, relative in (
        ("stage_a_scores", "stage_a_scores.jsonl"),
        ("metrics", "metrics.json"),
    ):
        record = inputs.get(key)
        path = OUTPUT_ROOT / relative
        if (
            not isinstance(record, Mapping)
            or record.get("path") != relative
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ExperimentError(f"independent regeneration binding drift: {key}")
    conditional_path = OUTPUT_ROOT / "conditional_stage_b_scores.jsonl"
    expected_stage_b = _stage_a_pass(metrics)
    conditional_record = inputs.get("conditional_stage_b_scores")
    if bool(regeneration.get("conditional_stage_b_present")) != expected_stage_b:
        raise ExperimentError("independent regeneration Stage-B disposition drift")
    if expected_stage_b:
        if (
            not conditional_path.is_file()
            or not isinstance(conditional_record, Mapping)
            or conditional_record.get("path") != conditional_path.name
            or conditional_record.get("bytes") != conditional_path.stat().st_size
            or conditional_record.get("sha256") != sha256_file(conditional_path)
        ):
            raise ExperimentError("conditional Stage-B regeneration binding drift")
    elif conditional_path.exists() or conditional_record is not None:
        raise ExperimentError("conditional Stage-B evidence exists after a failed Stage-A gate")
    panel = load_json(OUTPUT_ROOT / "panel_manifest.json")
    training = load_json(CACHE_ROOT / "training_receipt.json")
    predictor = load_json(OUTPUT_ROOT / "predictor_index.json")
    primary = _decision_value(metrics, "primary_classification", "UNRESOLVED")
    secondaries = _decision_value(metrics, "secondary_classifications", [])
    next_experiment = _decision_value(metrics, "next_experiment", "UNRESOLVED")
    result = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.result.v1",
        "status": "DEVELOPMENT_EXPLORATORY_COMPLETE", "development_only": True,
        "claims_boundary": "JEPA route selection under oracle admissibility.",
        "not_a_claim": "JEPA safety.",
        "constructed_challenge_caveat": panel["constructed_challenge_caveat"],
        "source_freeze_commit": load_json(OUTPUT_ROOT / "contract.json")["source_freeze_commit"],
        "recovery_binding_digest": panel["recovery_binding"]["content_digest"],
        "recovered_development_context": CONTRACT.RECOVERY_DECISION_AUTHORITY,
        "panel_digest": panel["content_digest"], "panel_adequacy": panel["adequacy"],
        "training": training, "metrics": metrics,
        "independent_regeneration_receipt_digest": regeneration["content_digest"],
        "predictor_substitution_executed": bool(predictor.get("executed")),
        "primary_classification": primary, "secondary_classifications": secondaries,
        "next_experiment": next_experiment,
        "prohibited_action_counters": {
            "predictor_training": 0, "safety_model_training": 0, "closed_loop_control": 0,
            "topological_memory": 0, "graph_input_to_ranker": 0, "novelty": 0,
            "beacon_discovery": 0, "custom_python_audit_hooks": 0,
            "custom_startup_or_forensic_framework": 0,
        },
        "safety_workstream": "REQUIREMENTS_ACQUISITION_REQUIRED",
    })
    atomic_json(CACHE_ROOT / "result.json", result)

    def display(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, bool):
            return "PASS" if value else "FAIL"
        if isinstance(value, (int, float)):
            return f"{float(value):.4f}"
        return str(value)

    def metric_row(label: str, source: Mapping[str, Any]) -> str:
        aggregate = source["aggregate"]
        return "| " + " | ".join(
            [
                label,
                display(aggregate["pairwise_accuracy"]),
                display(aggregate["spearman"]),
                display(aggregate["kendall"]),
                display(aggregate["best_route_top1"]),
                display(aggregate["best_route_top3"]),
                display(aggregate["mean_reciprocal_rank"]),
                display(aggregate["normalized_regret"]),
                display(aggregate["oracle_progress_fraction"]),
                display(aggregate["selected_geodesic_progress_m"]),
                display(aggregate["selected_euclidean_progress_m"]),
            ]
        ) + " |"

    stage_a_sources = metrics["stage_a"]["sources"]
    stage_a_rows = [
        metric_row(
            f"{model_id} / {source_id}",
            stage_a_sources[f"{model_id}::{source_id}"],
        )
        for model_id, source_id in CONTRACT.STAGE_A_MODEL_SOURCE_PAIRS
    ]
    stage_a_gate = metrics["stage_a"]["gate"]
    conditional = metrics.get("conditional_stage_b")
    report = [
        "# NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
        "", "Development-only exploratory result.", "",
        "> JEPA route selection under oracle admissibility.", "",
        "This is not JEPA safety and does not establish deployment safety, learned contact avoidance, physical Go2 safety, hidden beacon discovery, topological localisation, persistent memory, or complete maze navigation.",
        "", f"Primary classification: `{primary}`.",
        f"Secondary classifications: {', '.join(f'`{value}`' for value in secondaries) if secondaries else 'none'}.",
        f"Exact next experiment: `{next_experiment}`.", "",
        "## Panel", "",
        panel["constructed_challenge_caveat"],
        f"The panel contains {len(panel['states'])} unique scene/episode identities with role totals {panel['role_totals']}. All registered adequacy checks passed.",
        "",
        f"Adequacy: obstructed rays {panel['adequacy']['obstructed_direct_goal_rays']}/96; "
        f"two-or-more admissible {display(panel['adequacy']['two_or_more_admissible_fraction'])}; "
        f"direct/geodesic top-1 disagreement {display(panel['adequacy']['direct_top1_differs_geodesic_fraction'])}; "
        f"weak-or-negative direct progress for oracle-best {display(panel['adequacy']['oracle_best_weak_or_negative_euclidean_fraction'])}.",
        "", "## Stage A", "",
        "| Condition | Pairwise | Spearman | Kendall | Top-1 | Top-3 | MRR | Regret | Oracle fraction | Selected geo (m) | Selected Euclid (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *stage_a_rows,
        "",
        f"Stage-A classification: `{stage_a_gate['classification']}`; predictor substitution authorized: `{stage_a_gate['pass']}`.",
        f"Candidate-trajectory derangement material: `{stage_a_gate['candidate_future_derangement']['material']}`; "
        f"time-order derangement material (descriptive): `{stage_a_gate['time_order_derangement_descriptive']['material']}`.",
        "", "## Conditional Stage B", "",
    ]
    if conditional is None:
        report.extend([
            "Stage B did not run because the frozen Stage-A gate did not pass.",
        ])
    else:
        stage_b_sources = conditional["sources"]
        report.extend([
            "| Source | Pairwise | Spearman | Kendall | Top-1 | Top-3 | MRR | Regret | Oracle fraction | Selected geo (m) | Selected Euclid (m) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[
                metric_row(
                    source_id,
                    stage_b_sources[f"{model_id}::{source_id}"],
                )
                for model_id, source_id in CONTRACT.STAGE_B_MODEL_SOURCE_PAIRS
            ],
            "",
            f"Stage-B classification: `{conditional['gate']['classification']}`.",
            "RR minus R1 effects: `"
            + json.dumps(
                conditional["gate"]["rr_over_r1"]["comparison"],
                sort_keys=True,
                separators=(",", ":"),
            )
            + "`.",
        ])
    report.extend([
        "", "## Scientific interpretation", "",
        "The machine-readable metrics and independently regenerated raw score ledgers are authoritative. True-future trajectories are an observability upper bound; R1/RR are frozen predictor substitutions and no predictor was trained.",
        "", "## Prior development binding", "",
        "The authoritative recovered predecessor remains `DEVELOPMENT_SCIENTIFIC_PAYLOAD_RECOVERED` / `KINEMATIC_BASELINE_DOMINANT`. Its artifacts were governance-only and supplied zero rows, tensors, or checkpoints to this experiment.",
        "",
        CONTRACT.RECOVERY_DECISION_AUTHORITY["benchmark_conclusion"],
    ])
    atomic_bytes(OUTPUT_ROOT / "result.md", ("\n".join(report) + "\n").encode())
    # Public result JSON is the exact final metric/decision object, separate
    # from report publication and not an additional scientific computation.
    atomic_json(OUTPUT_ROOT / "result.json", result)
    required = [
        "contract.json", "panel_manifest.json", "split_manifest.json", "branch_ledger.jsonl",
        "route_labels.jsonl", "latent_index.json", "predictor_index.json", "training_ledger.jsonl",
        "stage_a_scores.jsonl", "metrics.json", "independent_regeneration_receipt.json",
        "result.json", "result.md",
    ] + [record["path"] for record in training["checkpoints"].values()]
    conditional_stage_b = OUTPUT_ROOT / "conditional_stage_b_scores.jsonl"
    if conditional_stage_b.is_file():
        required.append("conditional_stage_b_scores.jsonl")
    for relative in required:
        path = OUTPUT_ROOT / relative
        if not path.is_file():
            raise ExperimentError(f"required final output absent: {relative}")
    rows = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path == OUTPUT_ROOT / "file_hashes.json":
            continue
        relative = str(path.relative_to(OUTPUT_ROOT))
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ExperimentError(f"output inventory contains a non-regular/nlink-1 leaf: {relative}")
        rows.append({"path": relative, "bytes": info.st_size, "sha256": sha256_file(path)})
    manifest = attach_digest({
        "schema": "non_greedy_local_subgoal_jepa_planning_v1.file_hashes.v1",
        "root": str(OUTPUT_ROOT), "files": rows,
        "file_count_excluding_self": len(rows),
        "bytes_excluding_self": sum(row["bytes"] for row in rows),
        "file_hashes_self_sha256_excluded": True,
    })
    atomic_json(OUTPUT_ROOT / "file_hashes.json", manifest)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze-docs", "panel", "render", "encode", "train", "stage-a", "predict", "stage-b", "report"))
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import faulthandler
    faulthandler.enable()
    require_execution_runtime()
    args = build_parser().parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    stage = args.stage
    if stage == "freeze-docs":
        value = build_freeze_documents()
    else:
        require_source_freeze(require_execution_contract=stage != "panel")
        if stage == "panel":
            value = build_panel()[0]
        elif stage == "render":
            value = render_panel()
        elif stage == "encode":
            value = encode_panel(args.batch_size)
        elif stage == "train":
            value = train_rankers()
        elif stage == "stage-a":
            value = evaluate_stage_a()
        elif stage == "predict":
            value = materialize_predictors()
        elif stage == "stage-b":
            value = evaluate_stage_b()
        else:
            value = publish_result()
    print(json.dumps({"stage": stage, "status": "PASS", "content_digest": value.get("content_digest") if isinstance(value, Mapping) else None}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        raise
    raise SystemExit(exit_code)
