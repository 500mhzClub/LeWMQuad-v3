#!/usr/bin/env python3
"""No-training body-centric range-coverage qualification.

This executable consumes only the repaired two-ply corpus and the frozen
per-link geometry shards.  It never steps Genesis, imports Torch, opens a JEPA
checkpoint, trains a model, or changes any state/action/contact identity.

The lifecycle is deliberately split:

``freeze``
    Validate the prospective contract and write tracked contract, fixture,
    schema, and source-closure receipts before scientific materialisation.
``preflight``
    Bind the committed source, immutable inputs, environment, filesystems, and
    storage limits into the external output namespace.
``materialize``
    Stream all four sensor conditions and both evidence modes into per-state
    structured shards plus the complete row ledger.
``evaluate``
    Calibrate on the frozen internal-calibration role, score the development
    held-out role, attribute errors, benchmark the strongest condition, and
    write the result/report.
``execute``
    Run preflight, materialisation, and evaluation in order after the contract
    freeze commit.
``check``
    Revalidate all receipts and result bindings without recomputation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import gzip
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import platform
import resource
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import body_centric_range_coverage_qualification_v1_contract as CONTRACT


EXPERIMENT = "BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1"
PREDECESSOR_HEAD = "034c2fb902997ac29e2742fc4ddc2c28ad1706b6"
EXPECTED_BRANCH = "jepa-spatial-world-model-nav"
SOURCE_LINEAGE = "10b3a190d506830e6a87e04a0f1c832b92295bd7"
CORPUS = ROOT / ".generated/two_ply_successor_transition_corpus_repaired_v1/corpus_index.json"
GEOMETRY_INDEX = ROOT / ".generated/explicit_per_link_geometric_micro_state_upper_bound_v1/geometry_index.json"
SPLIT = ROOT / ".generated/two_ply_successor_transition_corpus_repaired_v1/development_internal_calibration_repaired_v1.json"
ACTION_CONTRACT = ROOT / ".generated/two_ply_successor_transition_corpus_repaired_v1/canonical_fourteen_action_contract.json"
OUTPUT_ROOT = Path("/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/body_centric_range_coverage_qualification_v1")
TRACKED_CONTRACT = ROOT / "docs/lewm_go2_body_centric_range_coverage_qualification_v1_contract_2026-08-25.json"
TRACKED_SCHEMA = ROOT / "docs/lewm_go2_body_centric_range_coverage_qualification_v1_output_schema_2026-08-25.json"
TRACKED_CLOSURE = ROOT / "docs/lewm_go2_body_centric_range_coverage_qualification_v1_source_closure_2026-08-25.json"
TRACKED_FIXTURE = ROOT / "docs/lewm_go2_body_centric_range_coverage_qualification_v1_fixture_2026-08-25.json"
TRACKED_PREREG = ROOT / "docs/lewm_go2_body_centric_range_coverage_qualification_v1_preregistration_2026-08-25.md"
RESULT_JSON = ROOT / "docs/lewm_go2_body_centric_range_coverage_qualification_v1_result_2026-08-25.json"
RESULT_MD = ROOT / "docs/lewm_go2_body_centric_range_coverage_qualification_v1_result_2026-08-25.md"

EXPECTED = {
    "corpus_index_sha256": "c1055724ebffd71c67b5424b4e447d223a828d3bc4da01369486bff73ad265f0",
    "corpus_logical_digest": "e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223",
    "action_contract_sha256": "cf8df092e8eff61d04348ebfe22b5e6a0cd31b5f39a4e05e45242752b3e5dc06",
    "predecessor_row_ledger_sha256": "63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94",
    "split_sha256": "eb2b41ca3ca4d4f7d2d2fc41495944e306e39798ede8865dd0904fa6c3d88021",
    "geometry_index_sha256": "67b473e6d0e0c422e4b2916b800399c98d5ec60383d398ff907fe2b8909a1d7f",
    "states": 176,
    "transitions": 29470,
    "current_transitions": 2464,
    "successor_transitions": 27006,
    "physics_steps": 1473500,
}
CONDITIONS = tuple(CONTRACT.CONDITION_IDS)
MODES = tuple(CONTRACT.EVIDENCE_MODE_IDS)
LINKS = 13
STEPS = 50
PHYSICS_TIMESTAMPS_S = np.arange(STEPS + 1, dtype=np.float64) * 0.002
PLATFORM_EMITTING_HOUSING_GEOM_INDICES = (1, 2)
PLATFORM_EMITTING_HOUSING_LINK_NAME = "base"
FLOAT_EVIDENCE_FIELDS = (
    "clearance_m",
    "point_age_s",
    "nearest_point_range_m",
    "support_point_age_s",
    "support_nearest_range_m",
    "obstacle_direction_body_rad",
    "target_azimuth_deg",
    "target_elevation_deg",
    "target_range_m",
    "target_event_azimuth_deg",
    "target_event_elevation_deg",
    "target_event_range_m",
)
BOOL_EVIDENCE_FIELDS = (
    "support",
    "event_time_support",
    "nominal_fov",
    "horizontal_fov",
    "vertical_fov",
    "direct_visibility",
    "self_occluded",
    "environment_occluded",
    "near_blind",
    "finite_scan_support_inherited",
)
INT16_EVIDENCE_FIELDS = (
    "responsible_object_index",
    "support_object_index",
    "point_support_count",
    "responsible_acquisition_index",
    "support_acquisition_index",
    "self_occluder_geom_index",
    "self_occluder_link_index",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any, *, canonical: bool = False) -> None:
    if canonical:
        payload = canonical_bytes(value)
    else:
        payload = (json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_bytes(path, payload)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def filesystem_receipt(path: Path) -> dict[str, Any]:
    path = path.resolve()
    stat = os.stat(path)
    usage = shutil.disk_usage(path)
    entry = subprocess.check_output(["findmnt", "-T", str(path), "-no", "SOURCE,FSTYPE,TARGET,FSROOT"], text=True).strip().split()
    return {
        "path": str(path),
        "device": int(stat.st_dev),
        "source": entry[0],
        "fstype": entry[1],
        "mount": entry[2],
        "fsroot": entry[3] if len(entry) > 3 else "/",
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def running_scientific_processes() -> list[dict[str, Any]]:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    needles = ("body_centric_range_coverage", "egomotion_bev_jepa", "jepa_phase3a", "train_")
    own_lineage: set[int] = set()
    cursor = os.getpid()
    while cursor > 1 and cursor not in own_lineage:
        own_lineage.add(cursor)
        try:
            status = Path(f"/proc/{cursor}/status").read_text()
            parent_line = next(line for line in status.splitlines() if line.startswith("PPid:"))
            cursor = int(parent_line.split()[1])
        except (FileNotFoundError, StopIteration, ValueError):
            break
    rows = []
    for line in output.splitlines():
        pid_text, _, command = line.strip().partition(" ")
        if not pid_text or int(pid_text) in own_lineage:
            continue
        if any(needle in command.lower() for needle in needles) and "ps -eo" not in command:
            rows.append({"pid": int(pid_text), "command": command})
    return rows


def role_map(split: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for role, key in (
        ("training", "development_training_state_ids"),
        ("calibration", "internal_calibration_state_ids"),
        ("heldout", "development_heldout_state_ids"),
    ):
        for state_id in split[key]:
            if state_id in mapping:
                raise RuntimeError(f"duplicate role assignment: {state_id}")
            mapping[state_id] = role
    return mapping


def validate_frozen_inputs(*, full_shards: bool = False) -> dict[str, Any]:
    observed = {
        "corpus_index_sha256": sha256_file(CORPUS),
        "action_contract_sha256": sha256_file(ACTION_CONTRACT),
        "split_sha256": sha256_file(SPLIT),
        "geometry_index_sha256": sha256_file(GEOMETRY_INDEX),
    }
    for key, expected in EXPECTED.items():
        if key in observed and observed[key] != expected:
            raise RuntimeError(f"{key}: expected {expected}, observed {observed[key]}")
    corpus = json.loads(CORPUS.read_text())
    geometry = json.loads(GEOMETRY_INDEX.read_text())
    split = json.loads(SPLIT.read_text())
    if corpus["corpus_logical_digest"] != EXPECTED["corpus_logical_digest"]:
        raise RuntimeError("corpus logical digest mismatch")
    if len(corpus["records"]) != EXPECTED["states"] or len(geometry["records"]) != EXPECTED["states"]:
        raise RuntimeError("state count mismatch")
    if geometry["transitions"] != EXPECTED["transitions"]:
        raise RuntimeError("transition count mismatch")
    predecessor_ledger_binding = geometry["bindings"].get("predecessor_row_ledger_sha256")
    if predecessor_ledger_binding != EXPECTED["predecessor_row_ledger_sha256"]:
        raise RuntimeError("predecessor row-ledger binding mismatch")
    observed["predecessor_row_ledger_sha256"] = predecessor_ledger_binding
    roles = role_map(split)
    if set(roles) != {row["state_id"] for row in corpus["records"]}:
        raise RuntimeError("role/corpus identity mismatch")
    geometry_by_state = {row["state_id"]: row for row in geometry["records"]}
    if set(geometry_by_state) != set(roles):
        raise RuntimeError("geometry/corpus identity mismatch")
    shard_bytes = 0
    shard_hashes_checked = 0
    for row in geometry["records"]:
        shard = Path(row["shard_path"]).expanduser()
        if not shard.is_file() or shard.stat().st_size != row["storage_bytes"]:
            raise RuntimeError(f"missing or changed geometry shard: {shard}")
        shard_bytes += shard.stat().st_size
        if full_shards:
            if sha256_file(shard) != row["shard_sha256"]:
                raise RuntimeError(f"geometry shard SHA mismatch: {shard}")
            shard_hashes_checked += 1
    counts = {role: sum(value == role for value in roles.values()) for role in ("training", "calibration", "heldout")}
    receipt = {
        "bindings": observed,
        "corpus_logical_digest": corpus["corpus_logical_digest"],
        "states": len(roles),
        "role_counts": counts,
        "transitions": geometry["transitions"],
        "current_transitions": geometry["materialized_current_transitions"],
        "successor_transitions": geometry["materialized_successor_transitions"],
        "physics_steps": geometry["physics_steps"],
        "geometry_shard_bytes": shard_bytes,
        "geometry_shard_hashes_checked": shard_hashes_checked,
        "pass": True,
    }
    return receipt


def environment_receipt() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "genesis-world", "torch", "gstaichi"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    first_party_modules = (
        "lewm",
        "lewm.safety",
        "lewm.safety.body_centric_range_coverage_qualification_v1_contract",
        "lewm.safety.body_centric_range_coverage_v1",
        "lewm.safety.body_centric_range_coverage_metrics_v1",
        "lewm.safety.body_centric_range_coverage_corpus_v1",
        "lewm.safety.body_centric_range_coverage_reporting_v1",
        "lewm.safety.body_centric_range_coverage_analysis_v1",
    )
    missing_first_party = [name for name in first_party_modules if name not in sys.modules]
    if missing_first_party:
        raise RuntimeError(f"first-party import closure is incomplete: {missing_first_party}")
    imports = {
        "stdlib": [
            "argparse", "collections", "collections.abc", "concurrent.futures", "copy", "dataclasses",
            "gzip", "hashlib", "importlib.metadata", "json", "math", "numbers",
            "multiprocessing", "os", "pathlib", "pickle", "platform", "resource", "shutil",
            "subprocess", "sys", "time", "types", "typing",
        ],
        "first_party": list(first_party_modules),
        "third_party": ["numpy", "scipy", "scipy.spatial"],
        "not_imported": ["genesis", "torch", "rsl_rl", "tensordict", "JEPA predictor"],
    }
    if platform.python_version() != "3.12.3":
        raise RuntimeError(f"Python version drift: {platform.python_version()}")
    if packages["numpy"] != "2.4.6" or packages["scipy"] != "1.17.1":
        raise RuntimeError(f"NumPy/SciPy version drift: {packages}")
    if packages["genesis-world"] != "0.3.14" or packages["torch"] != "2.12.0":
        raise RuntimeError(f"installed Genesis/Torch binding drift: {packages}")
    expected_prefix = Path(
        "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
        "genesis_render_vulkan"
    )
    expected_resolved_prefix = Path(
        "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
        "genesis_render_vulkan"
    )
    if Path(sys.prefix) != expected_prefix or Path(sys.prefix).resolve() != expected_resolved_prefix:
        raise RuntimeError(f"preserved environment path drift: {sys.prefix}")
    recovery_environment = Path(
        "/home/andrewknowles/recovery/REPOSITORY_BACKUP_SPACE_AND_ENVIRONMENT_RECOVERY_V1/"
        "environment_receipt.json"
    )
    pip_freeze = recovery_environment.with_name("genesis_render_vulkan_pip_freeze.txt")
    if sha256_file(recovery_environment) != "c56018c80c47dcc09350a9dc6b4d4930837ded2d1c5a1693e996b91f3777b8f5":
        raise RuntimeError("preserved recovery environment receipt drift")
    if sha256_file(pip_freeze) != "324b3b3b9d90ed85dd11e5d563d767489fbd928cef3602712812493eeccd5a90":
        raise RuntimeError("preserved pip-freeze receipt drift")
    torch_version_path = expected_prefix / "lib/python3.12/site-packages/torch/version.py"
    torch_version_sha256 = "a3f452de1a9dcee621e34adc683ab3b1e20836c6a69e40df94f93de21fced9df"
    if sha256_file(torch_version_path) != torch_version_sha256:
        raise RuntimeError("installed Torch build-version source drift")
    if "__version__ = '2.12.0+cu130'" not in torch_version_path.read_text():
        raise RuntimeError("installed Torch build suffix drift")
    import numpy
    import scipy
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(np.asarray([[0.0, 0.0, 0.0]])).query([[1.0, 0.0, 0.0]])
    if float(distance[0]) != 1.0:
        raise RuntimeError("SciPy cKDTree deterministic smoke failed")
    package_sources = {
        "numpy": {"path": str(Path(numpy.__file__).resolve()), "sha256": sha256_file(Path(numpy.__file__))},
        "scipy": {"path": str(Path(scipy.__file__).resolve()), "sha256": sha256_file(Path(scipy.__file__))},
    }
    first_party_sources = {}
    for name in first_party_modules:
        module_path = Path(sys.modules[name].__file__).resolve()
        first_party_sources[name] = {
            "path": str(module_path),
            "sha256": sha256_file(module_path),
        }
    return {
        "schema": "body_centric_range_coverage_environment_v1",
        "executable": sys.executable,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "package_sources": package_sources,
        "torch_build_version": "2.12.0+cu130",
        "torch_build_version_source": {
            "path": str(torch_version_path),
            "sha256": torch_version_sha256,
        },
        "first_party_sources": first_party_sources,
        "environment_prefix": sys.prefix,
        "environment_prefix_resolved": str(Path(sys.prefix).resolve()),
        "executable_resolved": str(Path(sys.executable).resolve()),
        "import_closure": imports,
        "scipy_ckdtree_smoke": "PASS",
        "torch_loaded": "torch" in sys.modules,
        "cpu_only": True,
        "genesis_steps": 0,
        "model_checkpoints_opened": 0,
        "training_packages_required": False,
        "tinyquadjepa_required": False,
        "environment_classification": "EXISTING_ENVIRONMENT_REUSABLE",
        "environment_name": "genesis_render_vulkan",
        "recovery_environment_receipt": {
            "path": str(recovery_environment),
            "sha256": sha256_file(recovery_environment),
        },
        "pip_freeze_receipt": {
            "path": str(pip_freeze),
            "sha256": sha256_file(pip_freeze),
        },
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "output_schema_sha256": sha256_file(TRACKED_SCHEMA),
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
    }


def rpy_quaternion_wxyz(rpy_rad: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy_rad)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    value = np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return value / np.linalg.norm(value)


def normalize_quaternions(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norm <= 1e-15) or not np.isfinite(array).all():
        raise ValueError("invalid quaternion array")
    return array / norm


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_quaternions(left)
    right = normalize_quaternions(np.broadcast_to(right, left.shape))
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    output = np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )
    return normalize_quaternions(output)


def rotate_vectors(quaternion_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternions(quaternion_wxyz)
    vector = np.asarray(vectors, dtype=np.float64)
    qvec = quaternion[..., 1:]
    scalar = quaternion[..., :1]
    temporary = 2.0 * np.cross(qvec, vector)
    return vector + scalar * temporary + np.cross(qvec, temporary)


def mounted_sensor_poses(
    base_position: np.ndarray,
    base_quaternion: np.ndarray,
    *,
    translation_m: Iterable[float],
    rotation_rpy_rad: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(base_position, dtype=np.float64)
    quaternion = normalize_quaternions(base_quaternion)
    local_position = np.broadcast_to(np.asarray(tuple(translation_m), np.float64), position.shape)
    local_quaternion = np.asarray(rpy_quaternion_wxyz(rotation_rpy_rad), np.float64)
    return position + rotate_vectors(quaternion, local_position), quaternion_multiply(quaternion, local_quaternion)


def yaw_only_base_poses(base_position: np.ndarray, base_quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quaternion = normalize_quaternions(base_quaternion)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw / 2.0
    output = np.stack((np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)), axis=-1)
    return np.asarray(base_position, np.float64), output


def sensor_mount(condition_id: str) -> dict[str, Any]:
    contract = CONTRACT.build_contract()
    mount_name = next(row["mount"] for row in contract["conditions"] if row["id"] == condition_id)
    return dict(contract["mounts"][mount_name])


def condition_contract(condition_id: str) -> dict[str, Any]:
    contract = CONTRACT.build_contract()
    return dict(next(row for row in contract["conditions"] if row["id"] == condition_id))


def applied_action_id(row: dict[str, Any]) -> str:
    values = ",".join(f"{float(value):.7f}" for value in row["applied_action"])
    return f"{row['controller']}:[{values}]"


def inverse_rotate_vectors(quaternion_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternions(quaternion_wxyz).copy()
    quaternion[..., 1:] *= -1.0
    return rotate_vectors(quaternion, vectors)


def closest_surface_on_obb(
    points_world: np.ndarray,
    *,
    center: np.ndarray,
    half_extent: np.ndarray,
    yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closest deterministic OBB surface witness for every world point."""

    points = np.asarray(points_world, np.float64)
    cosine, sine = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    rotation = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    local = (points - np.asarray(center, np.float64)) @ rotation
    half = np.asarray(half_extent, np.float64)
    closest = np.clip(local, -half, half)
    inside = np.all(np.abs(local) <= half + 1e-12, axis=1)
    if inside.any():
        rows = np.flatnonzero(inside)
        slack = half[None, :] - np.abs(local[rows])
        axis = np.argmin(slack, axis=1)
        sign = np.where(local[rows, axis] >= 0.0, 1.0, -1.0)
        closest[rows, axis] = sign * half[axis]
    witness = closest @ rotation.T + np.asarray(center, np.float64)
    distance = np.linalg.norm(points - witness, axis=1)
    return witness, distance


def primitive_bounding_radii(specs: list[Any], geom_indices: np.ndarray) -> np.ndarray:
    output = np.empty(len(geom_indices), np.float64)
    for index, geom_index in enumerate(np.asarray(geom_indices, int)):
        spec = specs[int(geom_index)]
        data = np.asarray(spec.data, np.float64)
        if spec.kind == "sphere":
            output[index] = data[0]
        elif spec.kind == "capsule":
            output[index] = data[0] + data[1] * 0.5
        else:
            output[index] = np.linalg.norm(data[:3] * 0.5)
    return output


def target_clearance_for_selected_geoms(
    target_points: np.ndarray,
    geom_positions: np.ndarray,
    geom_quaternions: np.ndarray,
    specs: list[Any],
    geom_indices: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_points, np.float64)
    positions = np.asarray(geom_positions, np.float64)
    quaternions = np.asarray(geom_quaternions, np.float64)
    indices = np.asarray(geom_indices, int)
    local = inverse_rotate_vectors(quaternions, target - positions)
    output = np.empty(len(indices), np.float64)
    for kind in ("sphere", "capsule", "box"):
        mask = np.asarray([specs[int(index)].kind == kind for index in indices])
        if not mask.any():
            continue
        data = np.zeros((int(mask.sum()), 3), np.float64)
        for row_index, geom_index in enumerate(indices[mask]):
            values = np.asarray(specs[int(geom_index)].data, np.float64)
            data[row_index, : min(3, len(values))] = values[:3]
        if kind == "sphere":
            output[mask] = np.linalg.norm(local[mask], axis=1) - data[:, 0]
        elif kind == "capsule":
            z = np.clip(local[mask, 2], -data[:, 1] * 0.5, data[:, 1] * 0.5)
            delta = local[mask].copy()
            delta[:, 2] -= z
            output[mask] = np.linalg.norm(delta, axis=1) - data[:, 0]
        else:
            half = data[:, :3] * 0.5
            q = np.abs(local[mask]) - half
            output[mask] = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
    return output


def closest_scene_targets(
    geom_positions: np.ndarray,
    geom_quaternions: np.ndarray,
    specs: list[Any],
    environment_boxes: list[Any],
) -> dict[str, np.ndarray]:
    """Construct exact, outcome-independent primitive--OBB closest witnesses."""

    from lewm.safety import body_centric_range_coverage_v1 as geometry

    positions = np.asarray(geom_positions, np.float64)
    quaternions = np.asarray(geom_quaternions, np.float64)
    if positions.shape != (STEPS, 27, 3) or quaternions.shape != (STEPS, 27, 4):
        raise ValueError("unexpected frozen geometry trace shape")
    if len(specs) != 27:
        raise ValueError("the frozen geometry contract must contain 27 specs")
    ordered_boxes = sorted(environment_boxes, key=lambda row: int(row.object_index))
    if [int(box.object_index) for box in ordered_boxes] != list(range(len(ordered_boxes))):
        raise RuntimeError("environment object indices are not contiguous")
    if not ordered_boxes:
        raise RuntimeError("the frozen scene must contain at least one environment OBB")

    flat_position = positions.reshape(-1, 3)
    flat_quaternion = quaternions.reshape(-1, 4)
    geom_indices = np.tile(np.arange(27, dtype=np.int16), STEPS)
    radii = primitive_bounding_radii(specs, geom_indices)
    lower_bounds = np.full(
        (len(flat_position), len(ordered_boxes)), np.inf, np.float64
    )
    for object_index, box in enumerate(ordered_boxes):
        yaw = math.atan2(
            2.0
            * (
                box.quaternion_wxyz[0] * box.quaternion_wxyz[3]
                + box.quaternion_wxyz[1] * box.quaternion_wxyz[2]
            ),
            1.0
            - 2.0
            * (
                box.quaternion_wxyz[2] ** 2
                + box.quaternion_wxyz[3] ** 2
            ),
        )
        _, surface_distance = closest_surface_on_obb(
            flat_position,
            center=np.asarray(box.center_xyz_m),
            half_extent=np.asarray(box.half_extents_xyz_m),
            yaw_rad=yaw,
        )
        cosine, sine = math.cos(yaw), math.sin(yaw)
        rotation = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            np.float64,
        )
        local_center = (
            flat_position - np.asarray(box.center_xyz_m, np.float64)
        ) @ rotation
        inside = np.all(
            np.abs(local_center)
            <= np.asarray(box.half_extents_xyz_m, np.float64)[None, :]
            + 1.0e-12,
            axis=1,
        )
        lower_bounds[:, object_index] = (
            np.where(inside, 0.0, surface_distance) - radii
        )
    pair_clearance = np.full(
        (len(flat_position), len(ordered_boxes)), np.inf, np.float64
    )
    pair_target = np.full(
        (len(flat_position), len(ordered_boxes), 3), np.nan, np.float64
    )
    pair_intersects = np.zeros(
        (len(flat_position), len(ordered_boxes)), dtype=bool
    )

    box_rows: list[int] = []
    for flat_index, geom_index in enumerate(geom_indices):
        if specs[int(geom_index)].kind == "box":
            box_rows.append(flat_index)
            continue
        best_value = math.inf
        candidates = np.argsort(lower_bounds[flat_index], kind="stable")
        for object_index in candidates:
            stopping_bound = best_value if best_value >= 0.0 else 0.0
            if (
                lower_bounds[flat_index, object_index]
                > stopping_bound + 1.0e-12
            ):
                break
            box = ordered_boxes[int(object_index)]
            witness = geometry.primitive_spec_at_geom_pose_obb_closest_witness(
                specs[int(geom_index)],
                flat_position[flat_index],
                flat_quaternion[flat_index],
                box,
            )
            pair_clearance[flat_index, object_index] = float(
                witness.signed_clearance_m
            )
            pair_target[flat_index, object_index] = np.asarray(
                witness.environment_point_world_xyz_m, np.float64
            )
            pair_intersects[flat_index, object_index] = bool(witness.intersects)
            best_value = min(best_value, float(witness.signed_clearance_m))

    def evaluate_box_pairs(
        box_flat_index: np.ndarray, object_index: np.ndarray
    ) -> None:
        if not len(box_flat_index):
            return
        box_geom_index = geom_indices[box_flat_index].astype(int)
        full_extents = np.asarray(
            [specs[index].data[:3] for index in box_geom_index], np.float64
        )
        environment_center = np.asarray(
            [ordered_boxes[int(index)].center_xyz_m for index in object_index],
            np.float64,
        )
        environment_quaternion = np.asarray(
            [ordered_boxes[int(index)].quaternion_wxyz for index in object_index],
            np.float64,
        )
        environment_half_extent = np.asarray(
            [
                ordered_boxes[int(index)].half_extents_xyz_m
                for index in object_index
            ],
            np.float64,
        )
        batch = geometry.batch_box_obb_closest_witness(
            flat_position[box_flat_index],
            flat_quaternion[box_flat_index],
            full_extents,
            environment_center,
            environment_quaternion,
            environment_half_extent,
        )
        pair_clearance[box_flat_index, object_index] = batch.signed_clearance_m
        pair_target[box_flat_index, object_index] = (
            batch.environment_point_world_xyz_m
        )
        pair_intersects[box_flat_index, object_index] = batch.intersects

    if box_rows:
        box_row_array = np.asarray(box_rows, dtype=np.int32)
        box_lower_bounds = lower_bounds[box_row_array]
        first_object = np.argmin(box_lower_bounds, axis=1).astype(np.int16)
        evaluate_box_pairs(box_row_array, first_object)

        first_clearance = pair_clearance[box_row_array, first_object]
        stopping_bound = np.where(first_clearance >= 0.0, first_clearance, 0.0)
        candidate = box_lower_bounds <= stopping_bound[:, None] + 1.0e-12
        candidate[np.arange(len(box_row_array)), first_object] = False
        candidate_row, additional_object = np.nonzero(candidate)
        evaluate_box_pairs(
            box_row_array[candidate_row],
            additional_object.astype(np.int16, copy=False),
        )

    # Environment object index prospectively resolves exact ties.
    best_object = np.argmin(pair_clearance, axis=1).astype(np.int16)
    best_value = pair_clearance[np.arange(len(flat_position)), best_object]
    best_target = pair_target[np.arange(len(flat_position)), best_object]
    geom_clearance = best_value.reshape(STEPS, 27)
    geom_object = best_object.reshape(STEPS, 27)
    geom_target = best_target.reshape(STEPS, 27, 3)
    link_to_geoms = {
        link: [index for index, spec in enumerate(specs) if int(spec.link_index) == link]
        for link in range(LINKS)
    }
    selected = np.empty((STEPS, LINKS), np.int16)
    for link, indices in link_to_geoms.items():
        if not indices:
            raise RuntimeError(f"protected link {link} has no collision geometry")
        values = geom_clearance[:, indices]
        selected[:, link] = np.asarray(indices, np.int16)[np.argmin(values, axis=1)]
    step_index = np.repeat(np.arange(STEPS), LINKS)
    flat_selected = selected.reshape(-1).astype(int)
    selected_flat_trace = step_index * 27 + flat_selected
    selected_object = geom_object[step_index, flat_selected].reshape(-1).astype(int)
    selected_target = geom_target[step_index, flat_selected].reshape(-1, 3).copy()
    selected_clearance = geom_clearance[step_index, flat_selected].reshape(-1).copy()

    # Intersecting boxes can have multiple exact feature witnesses.  Bind the
    # scalar witness identity only for the final 650 per-link winners.
    selected_intersects = pair_intersects[
        selected_flat_trace, selected_object
    ]
    intersecting_witness_cache: dict[
        tuple[int, int, bytes, bytes], tuple[np.ndarray, float]
    ] = {}
    for result_index in np.flatnonzero(selected_intersects):
        flat_index = int(selected_flat_trace[result_index])
        geom_index = int(geom_indices[flat_index])
        if specs[geom_index].kind != "box":
            continue
        object_index = int(selected_object[result_index])
        key = (
            geom_index,
            object_index,
            np.ascontiguousarray(flat_position[flat_index]).tobytes(),
            np.ascontiguousarray(flat_quaternion[flat_index]).tobytes(),
        )
        cached = intersecting_witness_cache.get(key)
        if cached is None:
            witness = geometry.primitive_spec_at_geom_pose_obb_closest_witness(
                specs[geom_index],
                flat_position[flat_index],
                flat_quaternion[flat_index],
                ordered_boxes[object_index],
            )
            cached = (
                np.asarray(witness.environment_point_world_xyz_m, np.float64),
                float(witness.signed_clearance_m),
            )
            intersecting_witness_cache[key] = cached
        selected_target[result_index] = cached[0]
        selected_clearance[result_index] = cached[1]

    return {
        "target_points": selected_target.reshape(STEPS, LINKS, 3),
        "object_index": selected_object.reshape(STEPS, LINKS).astype(np.int16),
        "geom_index": selected,
        "clearance_m": selected_clearance.reshape(STEPS, LINKS),
    }


def interpolate_transition_poses(
    boundary_qpos: np.ndarray,
    boundary_geom_transform: np.ndarray,
    qpos: np.ndarray,
    geom_transform: np.ndarray,
    query_timestamps_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from lewm.safety import body_centric_range_coverage_v1 as geometry

    boundary_qpos = np.asarray(boundary_qpos, np.float64)
    boundary_geom = np.asarray(boundary_geom_transform, np.float64)
    qpos = np.asarray(qpos, np.float64)
    geom = np.asarray(geom_transform, np.float64)
    if boundary_qpos.shape != (19,) or boundary_geom.shape != (27, 7):
        raise ValueError("bad boundary pose shapes")
    if qpos.shape != (STEPS, 19) or geom.shape != (STEPS, 27, 7):
        raise ValueError("bad transition pose shapes")
    base_position = np.concatenate((boundary_qpos[None, :3], qpos[:, :3]), axis=0)
    base_quaternion = np.concatenate((boundary_qpos[None, 3:7], qpos[:, 3:7]), axis=0)
    geom_position = np.concatenate((boundary_geom[None, :, :3], geom[:, :, :3]), axis=0)
    geom_quaternion = np.concatenate((boundary_geom[None, :, 3:], geom[:, :, 3:]), axis=0)
    sampled_base_position, sampled_base_quaternion = geometry.interpolate_transform_series_vectorized(
        PHYSICS_TIMESTAMPS_S, base_position, base_quaternion, query_timestamps_s
    )
    sampled_geom_position, sampled_geom_quaternion = geometry.interpolate_transform_series_vectorized(
        PHYSICS_TIMESTAMPS_S, geom_position, geom_quaternion, query_timestamps_s
    )
    return sampled_base_position, sampled_base_quaternion, sampled_geom_position, sampled_geom_quaternion


def sensor_pose_for_condition(
    condition_id: str,
    base_position: np.ndarray,
    base_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mount = sensor_mount(condition_id)
    if condition_id == "CURRENT_SPARSE_RANGE_BASELINE":
        base_position, base_quaternion = yaw_only_base_poses(base_position, base_quaternion)
    return mounted_sensor_poses(
        base_position,
        base_quaternion,
        translation_m=mount["translation_m"],
        rotation_rpy_rad=mount["rotation_rpy_rad"],
    )


def self_occlusion_geometry_for_condition(
    condition_id: str,
    robot_specs: list[Any],
    robot_position: np.ndarray,
    robot_quaternion: np.ndarray,
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    """Apply the frozen ray-only emitting-housing exemption.

    The stock ``base->radar`` optical origin lies inside the two coarse head
    housing collision primitives (geometry indices 1 and 2).  Those two
    primitives remain protected contact/clearance geometry, but cannot
    physically occlude rays emitted through their own aperture.  No other
    primitive is exempt and the body-centric condition has no exemption.
    """

    positions = np.asarray(robot_position, np.float64)
    quaternions = np.asarray(robot_quaternion, np.float64)
    if positions.shape[-2:] != (len(robot_specs), 3):
        raise ValueError("robot position/spec alignment mismatch")
    if quaternions.shape[-2:] != (len(robot_specs), 4):
        raise ValueError("robot quaternion/spec alignment mismatch")
    if condition_id not in (
        "REALISTIC_PLATFORM_SCAN",
        "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
    ):
        return list(robot_specs), positions, quaternions
    if not robot_specs:
        return [], positions, quaternions
    exempt = set(PLATFORM_EMITTING_HOUSING_GEOM_INDICES)
    observed = {
        int(spec.geom_index)
        for spec in robot_specs
        if int(spec.geom_index) in exempt
    }
    if observed != exempt:
        raise RuntimeError(
            f"platform emitting-housing geometry mismatch: {sorted(observed)}"
        )
    for spec in robot_specs:
        if int(spec.geom_index) in exempt and str(spec.link_name) != PLATFORM_EMITTING_HOUSING_LINK_NAME:
            raise RuntimeError("platform emitting-housing parent-link mismatch")
    keep = np.asarray(
        [int(spec.geom_index) not in exempt for spec in robot_specs], bool
    )
    return (
        [spec for spec, retained in zip(robot_specs, keep, strict=True) if retained],
        positions[..., keep, :],
        quaternions[..., keep, :],
    )


def render_sparse_cloud(
    *,
    condition_id: str,
    mode_id: str,
    state_id: str,
    transition_identity: str,
    boundary_snapshot_digest: str,
    boundary_qpos: np.ndarray,
    boundary_geom_transform: np.ndarray,
    qpos: np.ndarray,
    geom_transform: np.ndarray,
    environment_boxes: list[Any],
    robot_specs: list[Any],
) -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_v1 as geometry

    if condition_id == "CURRENT_SPARSE_RANGE_BASELINE":
        pattern = geometry.generate_sparse_scan_pattern(instantaneous=True)
        if mode_id == "PLANNING_TIME_CAUSAL_CLOUD":
            query = np.zeros(pattern.ray_count, np.float64)
            physical_time = query.copy()
        else:
            pattern_directions = np.concatenate((pattern.directions_sensor_fru, pattern.directions_sensor_fru), axis=0)
            query = np.concatenate((np.zeros(pattern.ray_count), np.full(pattern.ray_count, 0.1)))
            physical_time = query.copy()
            pattern = geometry.ScanPattern(
                identity=f"{pattern.identity}_BOUNDARY_UNION_V1",
                directions_sensor_fru=pattern_directions,
                timestamps_s=query,
                azimuth_rad=np.concatenate((pattern.azimuth_rad, pattern.azimuth_rad)),
                elevation_rad=np.concatenate((pattern.elevation_rad, pattern.elevation_rad)),
                approximation_class="DETERMINISTIC_BOUNDARY_UNION",
                parameters={"boundary_times_s": [0.0, 0.1], "source_pattern": pattern.identity},
            )
        near, far = 0.05, 10.0
        phase_receipt = None
    elif condition_id == "REALISTIC_PLATFORM_SCAN":
        phase_receipt = CONTRACT.derive_l2_scan_phases(
            boundary_snapshot_digest=boundary_snapshot_digest,
        )
        horizontal_phase = float(phase_receipt["horizontal_phase_cycles"])
        vertical_phase = float(phase_receipt["vertical_phase_cycles"])
        causal = mode_id == "PLANNING_TIME_CAUSAL_CLOUD"
        if causal:
            horizontal_phase -= 5.55 * 0.1
            vertical_phase -= 216.0 * 0.1
        pattern = geometry.generate_l2_scan_pattern(
            horizontal_phase_cycles=horizontal_phase,
            vertical_phase_cycles=vertical_phase,
        )
        if causal:
            query = np.zeros(pattern.ray_count, np.float64)
            physical_time = pattern.timestamps_s - 0.1
        else:
            query = np.asarray(pattern.timestamps_s, np.float64)
            physical_time = query.copy()
        near, far = 0.05, 30.0
    else:
        raise ValueError(f"not a sparse condition: {condition_id}")

    if mode_id == "PLANNING_TIME_CAUSAL_CLOUD":
        base_position = np.repeat(np.asarray(boundary_qpos[:3], np.float64)[None], pattern.ray_count, axis=0)
        base_quaternion = np.repeat(np.asarray(boundary_qpos[3:7], np.float64)[None], pattern.ray_count, axis=0)
        robot_position = np.repeat(np.asarray(boundary_geom_transform[:, :3], np.float64)[None], pattern.ray_count, axis=0)
        robot_quaternion = np.repeat(np.asarray(boundary_geom_transform[:, 3:], np.float64)[None], pattern.ray_count, axis=0)
    else:
        base_position, base_quaternion, robot_position, robot_quaternion = interpolate_transition_poses(
            boundary_qpos, boundary_geom_transform, qpos, geom_transform, query
        )
    sensor_position, sensor_quaternion = sensor_pose_for_condition(condition_id, base_position, base_quaternion)
    direction_world = rotate_vectors(sensor_quaternion, np.asarray(pattern.directions_sensor_fru, np.float64))
    occluder_specs, occluder_position, occluder_quaternion = (
        self_occlusion_geometry_for_condition(
            condition_id,
            robot_specs,
            robot_position,
            robot_quaternion,
        )
    )
    raycast = geometry.raycast_moving_scene(
        sensor_position,
        direction_world,
        near,
        far,
        environment_boxes,
        occluder_specs,
        occluder_position,
        occluder_quaternion,
        ground=True,
    )
    environment = raycast.valid_return & (raycast.hit_kind == "ENVIRONMENT")
    self_return = np.asarray(raycast.self_return, bool)
    ray_index = np.flatnonzero(environment).astype(np.int32)
    return {
        "points": np.asarray(raycast.point_world_xyz_m[environment], np.float64),
        "object_index": np.asarray(raycast.object_index[environment], np.int16),
        "ray_index": ray_index,
        "point_time_s": np.asarray(physical_time[environment], np.float64),
        "point_range_m": np.asarray(raycast.distance_m[environment], np.float64),
        "self_ray_index": np.flatnonzero(self_return).astype(np.int32),
        "self_geom_index": np.asarray(raycast.geom_index[self_return], np.int16),
        "self_link_index": np.asarray(raycast.link_index[self_return], np.int16),
        "self_distance_m": np.asarray(raycast.raw_distance_m[self_return], np.float64),
        "pattern": pattern.to_serializable(include_rays=False),
        "phase": phase_receipt,
        "ray_count": pattern.ray_count,
        "environment_return_count": int(environment.sum()),
        "self_return_count": int(raycast.self_return.sum()),
        "near_blind_count": int(raycast.near_blind.sum()),
        "ground_return_count": int((raycast.valid_return & (raycast.hit_kind == "GROUND")).sum()),
        "no_hit_count": int(raycast.no_hit.sum()),
        "transition_identity": transition_identity,
    }


def sparse_per_link_evidence(
    *,
    cloud: dict[str, Any],
    robot_specs: list[Any],
    geom_transform: np.ndarray,
    qpos: np.ndarray,
    targets: dict[str, np.ndarray],
    evaluation_time_s: np.ndarray,
) -> dict[str, np.ndarray]:
    from lewm.safety import body_centric_range_coverage_v1 as geometry

    reduction = geometry.trajectory_point_cloud_per_link_clearance(
        np.asarray(cloud["points"], np.float64),
        robot_specs,
        np.asarray(geom_transform[:, :, :3], np.float64),
        np.asarray(geom_transform[:, :, 3:], np.float64),
        candidate_k=64,
    )
    responsible_point = np.asarray(reduction.responsible_point_index, np.int32)
    clearance = np.asarray(reduction.minimum_clearance_m, np.float64)
    object_index = np.full((STEPS, LINKS), -1, np.int16)
    nearest_ray = np.full((STEPS, LINKS), -1, np.int32)
    point_age = np.full((STEPS, LINKS), np.nan, np.float64)
    point_range = np.full((STEPS, LINKS), np.nan, np.float64)
    obstacle_direction = np.full((STEPS, LINKS), np.nan, np.float64)
    points = np.asarray(cloud["points"], np.float64)
    valid = responsible_point >= 0
    if valid.any():
        selected = responsible_point[valid]
        object_index[valid] = np.asarray(cloud["object_index"], np.int16)[selected]
        nearest_ray[valid] = np.asarray(cloud["ray_index"], np.int32)[selected]
        step_time = np.broadcast_to(np.asarray(evaluation_time_s, np.float64)[:, None], (STEPS, LINKS))
        point_age[valid] = step_time[valid] - np.asarray(cloud["point_time_s"], np.float64)[selected]
        point_range[valid] = np.asarray(cloud["point_range_m"], np.float64)[selected]
        step_indices, _link_indices = np.nonzero(valid)
        observed_points = points[selected]
        body_delta = observed_points - np.asarray(qpos, np.float64)[step_indices, :3]
        body_direction = inverse_rotate_vectors(
            np.asarray(qpos, np.float64)[step_indices, 3:7], body_delta
        )
        obstacle_direction[valid] = np.arctan2(
            body_direction[:, 1], body_direction[:, 0]
        )
    support = np.zeros((STEPS, LINKS), bool)
    support_count = np.zeros((STEPS, LINKS), np.int16)
    support_ray = np.full((STEPS, LINKS), -1, np.int32)
    support_age = np.full((STEPS, LINKS), np.nan, np.float64)
    support_range = np.full((STEPS, LINKS), np.nan, np.float64)
    support_object = np.full((STEPS, LINKS), -1, np.int16)
    event_time_support = np.zeros((STEPS, LINKS), bool)
    cloud_object = np.asarray(cloud["object_index"], np.int16)
    for object_id in np.unique(targets["object_index"]):
        target_mask = targets["object_index"] == object_id
        candidate_indices = np.flatnonzero(cloud_object == object_id)
        candidates = points[candidate_indices]
        if not len(candidate_indices):
            continue
        query = targets["target_points"][target_mask]
        # 0.10 m local witness support is prospectively frozen in the contract.
        from scipy.spatial import cKDTree
        tree = cKDTree(candidates)
        distance, local_index = tree.query(query, k=1)
        selected = candidate_indices[np.asarray(local_index, int)]
        local_support = distance <= 0.10 + 1e-12
        support[target_mask] = local_support
        neighbours = tree.query_ball_point(query, 0.10 + 1e-12)
        support_count[target_mask] = np.asarray(
            [len(values) for values in neighbours], np.int16
        )
        target_steps = np.argwhere(target_mask)[:, 0]
        point_times = np.asarray(cloud["point_time_s"], np.float64)
        event_values = []
        provenance_indices: list[int] = []
        for query_point, nearest_source, local_neighbours, target_step in zip(
            query, selected, neighbours, target_steps, strict=True
        ):
            source_indices = candidate_indices[np.asarray(local_neighbours, int)]
            event_time = float(evaluation_time_s[int(target_step)])
            prior = source_indices[
                point_times[source_indices] <= event_time + 1e-12
            ]
            event_values.append(bool(len(prior)))
            if len(prior):
                chosen_pool = prior
                preferred_time = float(np.max(point_times[prior]))
                chosen_pool = chosen_pool[
                    np.abs(point_times[chosen_pool] - preferred_time) <= 1e-12
                ]
            elif len(source_indices):
                preferred_time = float(np.min(point_times[source_indices]))
                chosen_pool = source_indices[
                    np.abs(point_times[source_indices] - preferred_time) <= 1e-12
                ]
            else:
                chosen_pool = np.asarray([int(nearest_source)], int)
            chosen = min(
                chosen_pool,
                key=lambda index: (
                    float(np.linalg.norm(points[int(index)] - query_point)),
                    int(np.asarray(cloud["ray_index"], np.int32)[int(index)]),
                    int(index),
                ),
            )
            provenance_indices.append(int(chosen))
        event_time_support[target_mask] = np.asarray(event_values, bool)
        provenance = np.asarray(provenance_indices, int)
        support_ray[target_mask] = np.asarray(cloud["ray_index"], np.int32)[provenance]
        step_time = np.broadcast_to(np.asarray(evaluation_time_s, np.float64)[:, None], (STEPS, LINKS))
        support_age[target_mask] = step_time[target_mask] - np.asarray(
            cloud["point_time_s"], np.float64
        )[provenance]
        support_range[target_mask] = np.asarray(cloud["point_range_m"], np.float64)[provenance]
        support_object[target_mask] = int(object_id)
    return {
        "clearance_m": clearance,
        "responsible_point_index": responsible_point,
        "responsible_geom_index": np.asarray(reduction.responsible_geom_index, np.int16),
        "responsible_object_index": object_index,
        "nearest_ray_index": nearest_ray,
        "point_age_s": point_age,
        "nearest_point_range_m": point_range,
        "obstacle_direction_body_rad": obstacle_direction,
        "support": support,
        "event_time_support": event_time_support,
        "point_support_count": support_count,
        "support_nearest_ray_index": support_ray,
        "support_point_age_s": support_age,
        "support_nearest_range_m": support_range,
        "support_object_index": support_object,
        "responsible_acquisition_index": np.full((STEPS, LINKS), -1, np.int16),
        "support_acquisition_index": np.full((STEPS, LINKS), -1, np.int16),
        "self_occluder_geom_index": np.full((STEPS, LINKS), -1, np.int16),
        "self_occluder_link_index": np.full((STEPS, LINKS), -1, np.int16),
        "finite_scan_support_inherited": np.zeros((STEPS, LINKS), bool),
    }


def dense_per_link_evidence(
    *,
    condition_id: str,
    mode_id: str,
    boundary_qpos: np.ndarray,
    boundary_geom_transform: np.ndarray,
    qpos: np.ndarray,
    geom_transform: np.ndarray,
    environment_boxes: list[Any],
    robot_specs: list[Any],
    targets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    from lewm.safety import body_centric_range_coverage_v1 as geometry

    if condition_id not in (
        "CURRENT_SPARSE_RANGE_BASELINE",
        "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
        "DENSE_BODY_CENTRIC_SINGLE_ORIGIN",
    ):
        raise ValueError(condition_id)
    contract_id = (
        "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT"
        if condition_id == "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT"
        else condition_id
    )
    row = condition_contract(contract_id)
    if condition_id == "CURRENT_SPARSE_RANGE_BASELINE":
        horizontal_fov = (-180.0, 180.0)
        vertical_fov = (-15.0, 15.0)
        near, far = 0.05, 10.0
    else:
        horizontal_fov = (-180.0, 180.0)
        vertical_fov = (float(row["elevation_min_deg"]), float(row["elevation_max_deg"]))
        near, far = float(row["minimum_range_m"]), float(row["maximum_range_m"])
    target = np.asarray(targets["target_points"], np.float64).reshape(-1, 3)
    target_object = np.asarray(targets["object_index"], np.int16).reshape(-1)
    target_count = len(target)
    if target_count != STEPS * LINKS:
        raise ValueError("dense target cardinality mismatch")
    event_time = np.repeat(
        np.arange(1, STEPS + 1, dtype=np.float64) * 0.002, LINKS
    )
    event_base_position = np.repeat(np.asarray(qpos[:, :3], np.float64), LINKS, axis=0)
    event_base_quaternion = np.repeat(
        np.asarray(qpos[:, 3:7], np.float64), LINKS, axis=0
    )
    event_body_direction = inverse_rotate_vectors(
        event_base_quaternion, target - event_base_position
    )

    if mode_id == "PLANNING_TIME_CAUSAL_CLOUD":
        acquisition_time = np.asarray([0.0], np.float64)
        acquisition_base_position = np.asarray(boundary_qpos[:3], np.float64)[None]
        acquisition_base_quaternion = np.asarray(boundary_qpos[3:7], np.float64)[None]
        acquisition_robot_position = np.asarray(
            boundary_geom_transform[:, :3], np.float64
        )[None]
        acquisition_robot_quaternion = np.asarray(
            boundary_geom_transform[:, 3:], np.float64
        )[None]
    elif condition_id == "CURRENT_SPARSE_RANGE_BASELINE":
        # Preserve A's frozen boundary-union timing while removing only its
        # finite angular sparsity for attribution.
        acquisition_time = np.asarray([0.0, 0.1], np.float64)
        acquisition_base_position = np.stack(
            (np.asarray(boundary_qpos[:3], np.float64), np.asarray(qpos[-1, :3], np.float64))
        )
        acquisition_base_quaternion = np.stack(
            (np.asarray(boundary_qpos[3:7], np.float64), np.asarray(qpos[-1, 3:7], np.float64))
        )
        acquisition_robot_position = np.stack(
            (
                np.asarray(boundary_geom_transform[:, :3], np.float64),
                np.asarray(geom_transform[-1, :, :3], np.float64),
            )
        )
        acquisition_robot_quaternion = np.stack(
            (
                np.asarray(boundary_geom_transform[:, 3:], np.float64),
                np.asarray(geom_transform[-1, :, 3:], np.float64),
            )
        )
    else:
        # The dense true-future cloud is the union of analytic continuum
        # returns at the boundary and all 50 committed physics endpoints.
        acquisition_time = PHYSICS_TIMESTAMPS_S.copy()
        acquisition_base_position = np.concatenate(
            (np.asarray(boundary_qpos[:3], np.float64)[None], np.asarray(qpos[:, :3], np.float64)),
            axis=0,
        )
        acquisition_base_quaternion = np.concatenate(
            (np.asarray(boundary_qpos[3:7], np.float64)[None], np.asarray(qpos[:, 3:7], np.float64)),
            axis=0,
        )
        acquisition_robot_position = np.concatenate(
            (
                np.asarray(boundary_geom_transform[:, :3], np.float64)[None],
                np.asarray(geom_transform[:, :, :3], np.float64),
            ),
            axis=0,
        )
        acquisition_robot_quaternion = np.concatenate(
            (
                np.asarray(boundary_geom_transform[:, 3:], np.float64)[None],
                np.asarray(geom_transform[:, :, 3:], np.float64),
            ),
            axis=0,
        )
    sensor_condition = (
        "CURRENT_SPARSE_RANGE_BASELINE"
        if condition_id == "CURRENT_SPARSE_RANGE_BASELINE"
        else condition_id
    )
    event_sensor_position, event_sensor_quaternion = sensor_pose_for_condition(
        sensor_condition, event_base_position, event_base_quaternion
    )
    event_sensor_delta = target - event_sensor_position
    event_target_range = np.linalg.norm(event_sensor_delta, axis=1)
    event_direction_world = np.zeros_like(event_sensor_delta)
    event_direction_world[:, 0] = 1.0
    event_nonzero = event_target_range > 1.0e-12
    event_direction_world[event_nonzero] = (
        event_sensor_delta[event_nonzero]
        / event_target_range[event_nonzero, None]
    )
    event_local_direction = inverse_rotate_vectors(
        event_sensor_quaternion, event_direction_world
    )
    event_target_azimuth = np.degrees(
        np.arctan2(event_local_direction[:, 1], event_local_direction[:, 0])
    )
    event_target_elevation = np.degrees(
        np.arctan2(
            event_local_direction[:, 2],
            np.hypot(event_local_direction[:, 0], event_local_direction[:, 1]),
        )
    )
    acquisition_sensor_position, acquisition_sensor_quaternion = sensor_pose_for_condition(
        sensor_condition, acquisition_base_position, acquisition_base_quaternion
    )
    acquisition_count = len(acquisition_time)
    matrix_shape = (acquisition_count, target_count)
    horizontal_matrix = np.zeros(matrix_shape, bool)
    vertical_matrix = np.zeros(matrix_shape, bool)
    in_range_matrix = np.zeros(matrix_shape, bool)
    nominal_matrix = np.zeros(matrix_shape, bool)
    support_matrix = np.zeros(matrix_shape, bool)
    self_matrix = np.zeros(matrix_shape, bool)
    environment_matrix = np.zeros(matrix_shape, bool)
    direct_matrix = np.zeros(matrix_shape, bool)
    near_matrix = np.zeros(matrix_shape, bool)
    self_geom_matrix = np.full(matrix_shape, -1, np.int16)
    self_link_matrix = np.full(matrix_shape, -1, np.int16)
    distance_matrix = np.full(matrix_shape, np.nan, np.float64)
    azimuth_matrix = np.full(matrix_shape, np.nan, np.float64)
    elevation_matrix = np.full(matrix_shape, np.nan, np.float64)

    # Ten acquisition poses gives 6,500 paired rays per chunk and keeps each
    # worker's articulated-pose temporaries bounded.
    for start in range(0, acquisition_count, 10):
        stop = min(start + 10, acquisition_count)
        count = stop - start
        sensor_position = np.repeat(
            acquisition_sensor_position[start:stop], target_count, axis=0
        )
        sensor_quaternion = np.repeat(
            acquisition_sensor_quaternion[start:stop], target_count, axis=0
        )
        repeated_target = np.tile(target, (count, 1))
        repeated_object = np.tile(target_object, count)
        delta = repeated_target - sensor_position
        target_distance = np.linalg.norm(delta, axis=1)
        nonzero = target_distance > 1.0e-12
        direction_world = np.zeros_like(delta)
        direction_world[:, 0] = 1.0
        direction_world[nonzero] = (
            delta[nonzero] / target_distance[nonzero, None]
        )
        local_direction = inverse_rotate_vectors(sensor_quaternion, direction_world)
        azimuth = np.degrees(
            np.arctan2(local_direction[:, 1], local_direction[:, 0])
        )
        elevation = np.degrees(
            np.arctan2(
                local_direction[:, 2],
                np.hypot(local_direction[:, 0], local_direction[:, 1]),
            )
        )
        horizontal = (
            np.ones(len(repeated_target), bool)
            if horizontal_fov == (-180.0, 180.0)
            else (
                (azimuth >= horizontal_fov[0] - 1e-10)
                & (azimuth <= horizontal_fov[1] + 1e-10)
            )
        )
        vertical = (
            (elevation >= vertical_fov[0] - 1e-10)
            & (elevation <= vertical_fov[1] + 1e-10)
        )
        in_range = (
            nonzero
            & (target_distance >= near - 1e-12)
            & (target_distance <= far + 1e-12)
        )
        nominal = horizontal & vertical & in_range
        robot_position = np.repeat(
            acquisition_robot_position[start:stop], target_count, axis=0
        )
        robot_quaternion = np.repeat(
            acquisition_robot_quaternion[start:stop], target_count, axis=0
        )
        occluder_specs, occluder_position, occluder_quaternion = (
            self_occlusion_geometry_for_condition(
                condition_id,
                robot_specs,
                robot_position,
                robot_quaternion,
            )
        )
        raycast = geometry.raycast_moving_scene(
            sensor_position,
            direction_world,
            0.0,
            max(far, float(np.nanmax(target_distance, initial=far)) + 1e-6),
            environment_boxes,
            occluder_specs,
            occluder_position,
            occluder_quaternion,
            ground=True,
        )
        self_before = raycast.self_return & (
            raycast.raw_distance_m <= target_distance + 1e-6
        )
        environment_before = (
            (
                (raycast.hit_kind == "ENVIRONMENT")
                | (raycast.hit_kind == "GROUND")
            )
            & (raycast.raw_distance_m < target_distance - 1e-6)
        )
        target_hit = (
            (raycast.hit_kind == "ENVIRONMENT")
            & (raycast.object_index == repeated_object)
            & (raycast.raw_distance_m >= near - 1e-12)
            & (np.abs(raycast.raw_distance_m - target_distance) <= 1e-6)
        )
        support_pair = nominal & target_hit & ~raycast.self_return
        direct_pair = nominal & ~self_before & ~environment_before
        destination = slice(start, stop)
        reshape = (count, target_count)
        horizontal_matrix[destination] = horizontal.reshape(reshape)
        vertical_matrix[destination] = vertical.reshape(reshape)
        in_range_matrix[destination] = in_range.reshape(reshape)
        nominal_matrix[destination] = nominal.reshape(reshape)
        support_matrix[destination] = support_pair.reshape(reshape)
        self_matrix[destination] = self_before.reshape(reshape)
        self_geom_matrix[destination] = np.where(
            self_before, raycast.geom_index, -1
        ).astype(np.int16).reshape(reshape)
        self_link_matrix[destination] = np.where(
            self_before, raycast.link_index, -1
        ).astype(np.int16).reshape(reshape)
        environment_matrix[destination] = environment_before.reshape(reshape)
        direct_matrix[destination] = direct_pair.reshape(reshape)
        near_matrix[destination] = (
            (horizontal & vertical)
            & (
                (target_distance < near - 1e-12)
                | (
                    np.isfinite(raycast.raw_distance_m)
                    & (raycast.raw_distance_m < near - 1e-12)
                )
            )
        ).reshape(reshape)
        distance_matrix[destination] = target_distance.reshape(reshape)
        azimuth_matrix[destination] = azimuth.reshape(reshape)
        elevation_matrix[destination] = elevation.reshape(reshape)

    support = np.any(support_matrix, axis=0)
    event_support_matrix = support_matrix & (
        acquisition_time[:, None] <= event_time[None, :] + 1e-12
    )
    event_time_support = np.any(event_support_matrix, axis=0)
    support_count = np.sum(support_matrix, axis=0).astype(np.int16)
    selected_acquisition = np.full(target_count, -1, np.int16)
    diagnostic_acquisition = np.zeros(target_count, np.int16)
    for target_index in range(target_count):
        prior = np.flatnonzero(event_support_matrix[:, target_index])
        if len(prior):
            selected_acquisition[target_index] = int(prior[-1])
        else:
            later = np.flatnonzero(support_matrix[:, target_index])
            if len(later):
                selected_acquisition[target_index] = int(later[0])
        stage = (
            horizontal_matrix[:, target_index].astype(np.int16)
            + 2 * vertical_matrix[:, target_index].astype(np.int16)
            + 4 * in_range_matrix[:, target_index].astype(np.int16)
            + 8 * (~self_matrix[:, target_index]).astype(np.int16)
            + 16 * (~environment_matrix[:, target_index]).astype(np.int16)
            + 32 * support_matrix[:, target_index].astype(np.int16)
        )
        best = np.flatnonzero(stage == stage.max())
        order = sorted(
            best,
            key=lambda index: (
                abs(float(acquisition_time[int(index)] - event_time[target_index])),
                int(index),
            ),
        )
        diagnostic_acquisition[target_index] = int(order[0])
    provenance_acquisition = np.where(
        selected_acquisition >= 0, selected_acquisition, diagnostic_acquisition
    ).astype(np.int16)
    target_indices = np.arange(target_count)
    selected_distance = distance_matrix[provenance_acquisition, target_indices]
    selected_azimuth = azimuth_matrix[provenance_acquisition, target_indices]
    selected_elevation = elevation_matrix[provenance_acquisition, target_indices]
    point_age = event_time - acquisition_time[provenance_acquisition]
    clearance = np.where(
        support,
        np.asarray(targets["clearance_m"], np.float64).reshape(-1),
        np.inf,
    )
    nominal = np.any(nominal_matrix, axis=0)
    horizontal = np.any(horizontal_matrix, axis=0)
    vertical = np.any(vertical_matrix, axis=0)
    direct = np.any(direct_matrix, axis=0)
    # A window is self-occlusion-limited only when every nominal opportunity
    # remains blocked by robot geometry. Mixed blockers remain unresolved.
    nonself_nominal = nominal_matrix & ~self_matrix
    self_occluded = nominal & ~np.any(nonself_nominal, axis=0)
    environment_occluded = (
        np.any(nonself_nominal, axis=0)
        & ~direct
        & np.any(environment_matrix & nonself_nominal, axis=0)
    )
    near_blind = ~np.any(in_range_matrix & horizontal_matrix & vertical_matrix, axis=0) & np.any(
        near_matrix, axis=0
    )
    witness_id = (
        provenance_acquisition.astype(np.int32) * target_count
        + np.arange(target_count, dtype=np.int32)
    )
    responsible_object = np.where(support, target_object, -1).astype(np.int16)
    support_object = responsible_object.copy()
    responsible_witness = np.where(support, witness_id, -1).astype(np.int32)
    selected_age = np.where(support, point_age, np.nan)
    selected_range = np.where(support, selected_distance, np.nan)
    selected_self_geom = self_geom_matrix[
        diagnostic_acquisition, target_indices
    ]
    selected_self_link = self_link_matrix[
        diagnostic_acquisition, target_indices
    ]
    for target_index in np.flatnonzero(self_occluded):
        candidates = np.flatnonzero(
            nominal_matrix[:, target_index] & self_matrix[:, target_index]
        )
        if len(candidates):
            chosen = min(
                candidates,
                key=lambda index: (
                    abs(float(acquisition_time[int(index)] - event_time[target_index])),
                    int(index),
                ),
            )
            selected_self_geom[target_index] = self_geom_matrix[
                int(chosen), target_index
            ]
            selected_self_link[target_index] = self_link_matrix[
                int(chosen), target_index
            ]
    shape = (STEPS, LINKS)
    return {
        "clearance_m": clearance.reshape(shape),
        "responsible_object_index": responsible_object.reshape(shape),
        "nearest_ray_index": responsible_witness.reshape(shape),
        "support_nearest_ray_index": witness_id.astype(np.int32).reshape(shape),
        "point_age_s": selected_age.reshape(shape),
        "nearest_point_range_m": selected_range.reshape(shape),
        "support_point_age_s": point_age.reshape(shape),
        "support_nearest_range_m": selected_distance.reshape(shape),
        "support_object_index": target_object.reshape(shape),
        "support": support.reshape(shape),
        "event_time_support": event_time_support.reshape(shape),
        "point_support_count": support_count.reshape(shape),
        "responsible_acquisition_index": np.where(
            support, selected_acquisition, -1
        ).astype(np.int16).reshape(shape),
        "support_acquisition_index": provenance_acquisition.reshape(shape),
        "self_occluder_geom_index": selected_self_geom.reshape(shape),
        "self_occluder_link_index": selected_self_link.reshape(shape),
        "nominal_fov": nominal.reshape(shape),
        "horizontal_fov": horizontal.reshape(shape),
        "vertical_fov": vertical.reshape(shape),
        "direct_visibility": direct.reshape(shape),
        "self_occluded": self_occluded.reshape(shape),
        "environment_occluded": environment_occluded.reshape(shape),
        "near_blind": near_blind.reshape(shape),
        "finite_scan_support_inherited": np.zeros(shape, bool),
        "obstacle_direction_body_rad": np.arctan2(
            event_body_direction[:, 1], event_body_direction[:, 0]
        ).reshape(shape),
        "target_azimuth_deg": selected_azimuth.reshape(shape),
        "target_elevation_deg": selected_elevation.reshape(shape),
        "target_range_m": selected_distance.reshape(shape),
        "target_event_azimuth_deg": event_target_azimuth.reshape(shape),
        "target_event_elevation_deg": event_target_elevation.reshape(shape),
        "target_event_range_m": event_target_range.reshape(shape),
        "acquisition_times_s": acquisition_time,
        "acquisition_support": support_matrix,
        "acquisition_nominal_fov": nominal_matrix,
        "acquisition_self_occluded": self_matrix,
        "acquisition_self_geom_index": self_geom_matrix,
        "acquisition_self_link_index": self_link_matrix,
        "acquisition_environment_occluded": environment_matrix,
    }


def inherit_realistic_support_into_dense_platform(
    dense: dict[str, np.ndarray], sparse: dict[str, np.ndarray]
) -> int:
    """Make C a certified coverage superset of the finite B scan.

    A valid B same-object return within the frozen 0.10 m witness radius is a
    ray that exists inside C's identical mount/FOV/range continuum.  It is
    therefore a conservative local-surface support witness even when the
    exact closest point itself lies on the occluded face of that object.
    """

    sparse_support = np.asarray(sparse["support"], bool)
    dense_support = np.asarray(dense["support"], bool)
    inherited = sparse_support & ~dense_support
    if inherited.any():
        for field in (
            "clearance_m",
            "responsible_object_index",
            "nearest_ray_index",
            "point_age_s",
            "nearest_point_range_m",
            "support_nearest_ray_index",
            "support_point_age_s",
            "support_nearest_range_m",
            "support_object_index",
            "point_support_count",
            "responsible_geom_index",
            "obstacle_direction_body_rad",
        ):
            if field in dense and field in sparse:
                dense[field][inherited] = sparse[field][inherited]
        dense["support"][inherited] = True
        dense["event_time_support"][inherited] = np.asarray(
            sparse["event_time_support"], bool
        )[inherited]
        dense["nominal_fov"][inherited] = True
        dense["horizontal_fov"][inherited] = True
        dense["vertical_fov"][inherited] = True
        dense["direct_visibility"][inherited] = True
        dense["self_occluded"][inherited] = False
        dense["environment_occluded"][inherited] = False
        dense["near_blind"][inherited] = False
        dense["self_occluder_geom_index"][inherited] = -1
        dense["self_occluder_link_index"][inherited] = -1
        dense["responsible_acquisition_index"][inherited] = -1
        dense["support_acquisition_index"][inherited] = -1
        dense["finite_scan_support_inherited"][inherited] = True
    if np.any(sparse_support & ~np.asarray(dense["support"], bool)):
        raise RuntimeError("REALISTIC_PLATFORM_SCAN support is not contained by dense platform C")
    return int(inherited.sum())


def runtime_geometry_contract(state: Any) -> tuple[list[Any], list[Any]]:
    from lewm.safety import body_centric_range_coverage_v1 as geometry

    boxes = []
    for index, name in enumerate(state.scene_boxes.object_names):
        yaw = float(state.scene_boxes.yaw_rad[index])
        boxes.append(
            geometry.OrientedBox(
                identity=str(name),
                center_xyz_m=tuple(float(value) for value in state.scene_boxes.centers_m[index]),
                half_extents_xyz_m=tuple(float(value) for value in state.scene_boxes.half_extents_m[index]),
                quaternion_wxyz=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
                object_index=index,
            )
        )
    specs = []
    for row in state.geometry_contract:
        specs.append(
            geometry.RobotPrimitiveSpec(
                identity=f"{row['link_name']}:{int(row['geom_index']):02d}",
                kind=str(row["kind"]),
                data=tuple(float(value) for value in row["data"]),
                local_position_xyz_m=tuple(float(value) for value in row["local_pos"]),
                local_quaternion_wxyz=tuple(float(value) for value in row["local_quat"]),
                geom_index=int(row["geom_index"]),
                link_index=int(row["link_index"]),
                link_name=str(row["link_name"]),
            )
        )
    return boxes, specs


def platform_mount_housing_preflight(
    context: Any, corpus_adapter: Any, geometry: Any
) -> dict[str, Any]:
    """Bind the stock optical origin to its two coarse host primitives."""

    state_id = context.state_ids[0]
    state = corpus_adapter.load_state(context, state_id, shard_fields=())
    _boxes, specs = runtime_geometry_contract(state)
    boundary = state.boundaries.current
    sensor_position, _sensor_quaternion = sensor_pose_for_condition(
        "REALISTIC_PLATFORM_SCAN",
        np.asarray(boundary.qpos[:3], np.float64)[None],
        np.asarray(boundary.qpos[3:7], np.float64)[None],
    )
    clearances: dict[int, float] = {}
    for spec in specs:
        index = int(spec.geom_index)
        primitive = geometry.RobotPrimitive(
            identity=str(spec.identity),
            kind=str(spec.kind),
            data=tuple(spec.data),
            position_xyz_m=tuple(
                float(value)
                for value in boundary.contract_geom_transform[index, :3]
            ),
            quaternion_wxyz=tuple(
                float(value)
                for value in boundary.contract_geom_transform[index, 3:]
            ),
            geom_index=index,
            link_index=int(spec.link_index),
            link_name=str(spec.link_name),
        )
        clearances[index] = float(
            geometry.point_to_primitive_clearance(sensor_position, primitive)[0]
        )
    containing = sorted(index for index, value in clearances.items() if value <= 0.0)
    if containing != list(PLATFORM_EMITTING_HOUSING_GEOM_INDICES):
        raise RuntimeError(
            f"platform origin containment drift: expected {PLATFORM_EMITTING_HOUSING_GEOM_INDICES}, "
            f"observed {containing}"
        )
    nonexempt_inside = [
        index
        for index, value in clearances.items()
        if index not in PLATFORM_EMITTING_HOUSING_GEOM_INDICES and value <= 0.0
    ]
    if nonexempt_inside:
        raise RuntimeError(f"platform origin lies inside nonexempt geometry: {nonexempt_inside}")
    return {
        "state_id": state_id,
        "sensor_origin_world_m": [float(value) for value in sensor_position[0]],
        "containing_geom_indices": containing,
        "exempt_geom_identities": [
            str(spec.identity)
            for spec in specs
            if int(spec.geom_index) in PLATFORM_EMITTING_HOUSING_GEOM_INDICES
        ],
        "nonexempt_inside_geom_indices": nonexempt_inside,
        "minimum_nonexempt_clearance_m": min(
            value
            for index, value in clearances.items()
            if index not in PLATFORM_EMITTING_HOUSING_GEOM_INDICES
        ),
        "ray_only": True,
        "protected_contact_geometry_unchanged": True,
        "pass": True,
    }


def transition_boundary(state: Any, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    if row["level"] == "current":
        boundary = state.boundaries.current
    else:
        boundary = state.boundaries.successors[int(row["current_action_index"])]
    return (
        np.asarray(boundary.qpos, np.float64),
        np.asarray(boundary.contract_geom_transform, np.float64),
        str(boundary.snapshot_digest),
    )


def audit_selection_sha256(
    *, transition_uid: str, role: str, family: str, transition_kind: str
) -> str:
    namespace = "BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1/RAW_AUDIT_V1"
    rank_value = {
        "transition_uid": str(transition_uid),
        "role": str(role),
        "family": str(family),
        "transition_kind": str(transition_kind),
    }
    return hashlib.sha256(
        namespace.encode("utf-8")
        + b"\x00"
        + json.dumps(rank_value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def frozen_audit_subset(context: Any) -> dict[str, list[str]]:
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in context.iter_transition_identity_rows():
        row = dict(source)
        key = (str(row["role"]), str(row["family"]), str(row["level"]))
        rank = audit_selection_sha256(
            transition_uid=str(row["identity"]),
            role=key[0],
            family=key[1],
            transition_kind=key[2],
        )
        strata[key].append({"identity": str(row["identity"]), "rank": rank})
    selected: dict[str, list[str]] = {role: [] for role in ("training", "calibration", "heldout")}
    for (role, _family, _kind), rows in sorted(strata.items()):
        winner = min(rows, key=lambda row: (row["rank"], row["identity"]))
        selected[role].append(winner["identity"])
    for role in selected:
        selected[role].sort()
    return selected


def representative_geometry_is_equal(state: Any, representative: int, copy: int) -> bool:
    arrays = state.shard.arrays
    return bool(
        np.allclose(arrays["qpos"][representative], arrays["qpos"][copy], rtol=0.0, atol=2e-6)
        and np.allclose(arrays["geom_transform"][representative], arrays["geom_transform"][copy], rtol=0.0, atol=2e-6)
        and bool(arrays["frozen_contact_label"][representative]) == bool(arrays["frozen_contact_label"][copy])
    )


def raw_audit_artifact(
    *,
    transition_identity: str,
    role: str,
    family: str,
    transition_kind: str,
    source_representative_transition_uid: str,
    boundary_snapshot_digest: str,
    condition_id: str,
    mode_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    safe_identity = hashlib.sha256(transition_identity.encode()).hexdigest()[:16]
    relative = Path("raw_audit") / safe_identity / f"{condition_id}__{mode_id}.npz"
    path = OUTPUT_ROOT / relative
    arrays: dict[str, np.ndarray] = {}
    if "points" in payload:
        arrays = {
            "point_world_xyz_m": np.asarray(payload["points"], np.float32),
            "object_index": np.asarray(payload["object_index"], np.int16),
            "ray_index": np.asarray(payload["ray_index"], np.int32),
            "point_time_s": np.asarray(payload["point_time_s"], np.float32),
            "point_range_m": np.asarray(payload["point_range_m"], np.float32),
            "self_ray_index": np.asarray(payload["self_ray_index"], np.int32),
            "self_geom_index": np.asarray(payload["self_geom_index"], np.int16),
            "self_link_index": np.asarray(payload["self_link_index"], np.int16),
            "self_distance_m": np.asarray(payload["self_distance_m"], np.float32),
        }
        kind = "RANGE_POINT_CLOUD"
        count = len(arrays["point_world_xyz_m"])
    else:
        arrays = {
            "target_world_xyz_m": np.asarray(payload["target_points"], np.float32),
            "target_object_index": np.asarray(payload["target_object_index"], np.int16),
            "support": np.asarray(payload["support"], np.uint8),
            "clearance_m": np.asarray(payload["clearance_m"], np.float32),
            "finite_scan_support_inherited": np.asarray(
                payload.get("finite_scan_support_inherited", []), np.uint8
            ),
            "acquisition_times_s": np.asarray(
                payload.get("acquisition_times_s", []), np.float32
            ),
            "acquisition_support": np.asarray(
                payload.get("acquisition_support", []), np.uint8
            ),
            "acquisition_nominal_fov": np.asarray(
                payload.get("acquisition_nominal_fov", []), np.uint8
            ),
            "acquisition_self_occluded": np.asarray(
                payload.get("acquisition_self_occluded", []), np.uint8
            ),
            "acquisition_self_geom_index": np.asarray(
                payload.get("acquisition_self_geom_index", []), np.int16
            ),
            "acquisition_self_link_index": np.asarray(
                payload.get("acquisition_self_link_index", []), np.int16
            ),
            "acquisition_environment_occluded": np.asarray(
                payload.get("acquisition_environment_occluded", []), np.uint8
            ),
        }
        kind = "DENSE_CONTINUUM_TARGET_WITNESS"
        count = int(np.asarray(payload["support"]).size)
    atomic_npz(path, **arrays)
    selection = audit_selection_sha256(
        transition_uid=transition_identity,
        role=role,
        family=family,
        transition_kind=transition_kind,
    )
    return {
        "transition_uid": transition_identity,
        "role": role,
        "family": family,
        "transition_kind": transition_kind,
        "source_representative_transition_uid": source_representative_transition_uid,
        "boundary_snapshot_digest": boundary_snapshot_digest,
        "condition_id": condition_id,
        "evidence_mode": mode_id,
        "pattern_identity": (
            payload.get("pattern", {}).get("identity")
            if "points" in payload
            else f"{condition_id}/ANALYTIC_TARGET_DIRECTED_CONTINUUM_V1"
        ),
        "phase_digest_sha256": (
            None
            if payload.get("phase") is None
            else payload["phase"].get("phase_digest_sha256")
        ),
        "selection_sha256": selection,
        "representation_kind": kind,
        "artifact_relative_path": str(relative),
        "artifact_sha256": sha256_file(path),
        "point_or_witness_count": count,
        "bytes": path.stat().st_size,
    }


def _empty_evidence_arrays(transitions: int) -> dict[str, np.ndarray]:
    shape = (transitions, len(CONDITIONS), len(MODES), STEPS, LINKS)
    arrays: dict[str, np.ndarray] = {}
    for field in FLOAT_EVIDENCE_FIELDS:
        arrays[field] = np.full(shape, np.nan if field != "clearance_m" else np.inf, np.float32)
    for field in BOOL_EVIDENCE_FIELDS:
        arrays[field] = np.zeros(shape, np.uint8)
    for field in INT16_EVIDENCE_FIELDS:
        fill = -1 if field in (
            "responsible_object_index",
            "support_object_index",
            "responsible_acquisition_index",
            "support_acquisition_index",
            "self_occluder_geom_index",
            "self_occluder_link_index",
        ) else 0
        arrays[field] = np.full(shape, fill, np.int16)
    arrays["nearest_ray_index"] = np.full(shape, -1, np.int32)
    arrays["support_nearest_ray_index"] = np.full(shape, -1, np.int32)
    arrays["responsible_geom_index"] = np.full(shape, -1, np.int16)
    return arrays


def _store_evidence(
    arrays: dict[str, np.ndarray],
    transition: int,
    condition_index: int,
    mode_index: int,
    evidence: dict[str, np.ndarray],
) -> None:
    for field in FLOAT_EVIDENCE_FIELDS:
        if field in evidence:
            arrays[field][transition, condition_index, mode_index] = np.asarray(evidence[field], np.float32)
    for field in BOOL_EVIDENCE_FIELDS:
        arrays[field][transition, condition_index, mode_index] = np.asarray(evidence[field], np.uint8)
    for field in INT16_EVIDENCE_FIELDS:
        arrays[field][transition, condition_index, mode_index] = np.asarray(evidence[field], np.int16)
    arrays["nearest_ray_index"][transition, condition_index, mode_index] = np.asarray(
        evidence["nearest_ray_index"], np.int32
    )
    if "support_nearest_ray_index" in evidence:
        arrays["support_nearest_ray_index"][transition, condition_index, mode_index] = np.asarray(
            evidence["support_nearest_ray_index"], np.int32
        )
    if "responsible_geom_index" in evidence:
        arrays["responsible_geom_index"][transition, condition_index, mode_index] = np.asarray(
            evidence["responsible_geom_index"], np.int16
        )


def _copy_evidence(arrays: dict[str, np.ndarray], representative: int, copy: int) -> None:
    for value in arrays.values():
        value[copy] = value[representative]


def _validate_state_receipt_artifacts(
    receipt_path: Path, *, context: Any | None = None
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text())
    core = dict(receipt)
    declared_digest = core.pop("content_digest", None)
    if declared_digest != content_digest(core) or receipt.get("status") != "PASS":
        raise RuntimeError(f"state receipt self-digest/pass failure: {receipt_path}")
    if receipt.get("contract_sha256") != sha256_file(TRACKED_CONTRACT):
        raise RuntimeError(f"state contract binding drift: {receipt_path}")
    if receipt.get("source_closure_sha256") != sha256_file(TRACKED_CLOSURE):
        raise RuntimeError(f"state source-closure binding drift: {receipt_path}")
    if receipt.get("source_freeze_commit") != git("rev-parse", "HEAD"):
        raise RuntimeError(f"state source-freeze binding drift: {receipt_path}")
    shard = Path(receipt["shard_path"])
    if (
        not shard.is_file()
        or shard.stat().st_size != int(receipt["storage_bytes"])
        or sha256_file(shard) != receipt["shard_sha256"]
    ):
        raise RuntimeError(f"state shard drift: {shard}")
    if receipt.get("applied_action_copy_validation", {}).get("pass") is not True:
        raise RuntimeError(f"state copy validation is absent or failed: {receipt_path}")
    if len(receipt.get("scan_receipts", [])) != int(receipt["representatives"]) * 4:
        raise RuntimeError(f"state scan-receipt cardinality drift: {receipt_path}")
    for raw in receipt.get("raw_audit", []):
        artifact = OUTPUT_ROOT / raw["artifact_relative_path"]
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(raw["bytes"])
            or sha256_file(artifact) != raw["artifact_sha256"]
        ):
            raise RuntimeError(f"state raw-audit artifact drift: {artifact}")
    if context is not None:
        state_id = str(receipt["state_id"])
        geometry_record = context.geometry_record(state_id)
        corpus_record = context.corpus_record(state_id)
        if receipt["source_geometry_shard_sha256"] != geometry_record["shard_sha256"]:
            raise RuntimeError(f"state source geometry binding drift: {state_id}")
        if receipt["snapshot_sha256"] != corpus_record["snapshot_sha256"]:
            raise RuntimeError(f"state snapshot binding drift: {state_id}")
    return receipt


def materialize_state(context: Any, state_id: str, audit_identities: set[str]) -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    receipt_path = OUTPUT_ROOT / "states" / f"{state_id}.json"
    if receipt_path.is_file():
        receipt = _validate_state_receipt_artifacts(receipt_path, context=context)
        return {**receipt, "status": "REUSED"}
    started = time.time()
    state = corpus_adapter.load_state(
        context,
        state_id,
        shard_fields=("approximate_scene_clearance", "lidar_clearance"),
    )
    copy_validation = dict(
        corpus_adapter.validate_representative_geometry(
            state.shard, state.action_copy_map
        )
    )
    if copy_validation.get("pass") is not True:
        raise RuntimeError(f"{state_id}: applied-action copy geometry is not exact")
    for representative, copies in state.action_copy_map.copies_by_representative.items():
        representative_digest = transition_boundary(
            state, dict(state.transition_rows[representative])
        )[2]
        for copy in copies:
            copy_digest = transition_boundary(state, dict(state.transition_rows[copy]))[2]
            if copy_digest != representative_digest:
                raise RuntimeError(
                    f"{state_id}: scan-phase boundary differs for copy "
                    f"{representative}->{copy}"
                )
    boxes, specs = runtime_geometry_contract(state)
    transitions = state.shard.transition_count
    evidence = _empty_evidence_arrays(transitions)
    target_points = np.full((transitions, STEPS, LINKS, 3), np.nan, np.float32)
    target_object = np.full((transitions, STEPS, LINKS), -1, np.int16)
    target_geom = np.full((transitions, STEPS, LINKS), -1, np.int16)
    oracle_clearance = np.full((transitions, STEPS, LINKS), np.inf, np.float32)
    scan_receipts: list[dict[str, Any]] = []
    raw_audit: list[dict[str, Any]] = []
    planning_cloud_cache: dict[tuple[str, str], dict[str, Any]] = {}
    copy_map = state.action_copy_map
    representatives = sorted(copy_map.copies_by_representative)
    for ordinal, representative in enumerate(representatives, 1):
        row = dict(state.transition_rows[representative])
        boundary_qpos, boundary_geom, boundary_digest = transition_boundary(state, row)
        qpos = np.asarray(state.shard.arrays["qpos"][representative], np.float64)
        geom_transform = np.asarray(state.shard.arrays["geom_transform"][representative], np.float64)
        targets = closest_scene_targets(
            geom_transform[:, :, :3],
            geom_transform[:, :, 3:],
            specs,
            boxes,
        )
        target_points[representative] = targets["target_points"]
        target_object[representative] = targets["object_index"]
        target_geom[representative] = targets["geom_index"]
        oracle_clearance[representative] = targets["clearance_m"]

        dense_a: dict[str, dict[str, np.ndarray]] = {}
        dense_c: dict[str, dict[str, np.ndarray]] = {}
        dense_d: dict[str, dict[str, np.ndarray]] = {}
        for mode in MODES:
            dense_a[mode] = dense_per_link_evidence(
                condition_id="CURRENT_SPARSE_RANGE_BASELINE",
                mode_id=mode,
                boundary_qpos=boundary_qpos,
                boundary_geom_transform=boundary_geom,
                qpos=qpos,
                geom_transform=geom_transform,
                environment_boxes=boxes,
                robot_specs=specs,
                targets=targets,
            )
            dense_c[mode] = dense_per_link_evidence(
                condition_id="DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
                mode_id=mode,
                boundary_qpos=boundary_qpos,
                boundary_geom_transform=boundary_geom,
                qpos=qpos,
                geom_transform=geom_transform,
                environment_boxes=boxes,
                robot_specs=specs,
                targets=targets,
            )
            dense_d[mode] = dense_per_link_evidence(
                condition_id="DENSE_BODY_CENTRIC_SINGLE_ORIGIN",
                mode_id=mode,
                boundary_qpos=boundary_qpos,
                boundary_geom_transform=boundary_geom,
                qpos=qpos,
                geom_transform=geom_transform,
                environment_boxes=boxes,
                robot_specs=specs,
                targets=targets,
            )
            dense_c[mode]["responsible_geom_index"] = targets["geom_index"]
            dense_d[mode]["responsible_geom_index"] = targets["geom_index"]
            _store_evidence(evidence, representative, 2, MODES.index(mode), dense_c[mode])
            _store_evidence(evidence, representative, 3, MODES.index(mode), dense_d[mode])

        identity = str(row["identity"])
        for condition_index, condition_id in enumerate(CONDITIONS[:2]):
            auxiliary = dense_a if condition_index == 0 else dense_c
            for mode_index, mode in enumerate(MODES):
                cache_key = (condition_id, boundary_digest)
                cache_reused = bool(
                    mode == "PLANNING_TIME_CAUSAL_CLOUD"
                    and cache_key in planning_cloud_cache
                )
                if cache_reused:
                    cloud = planning_cloud_cache[cache_key]
                else:
                    cloud = render_sparse_cloud(
                        condition_id=condition_id,
                        mode_id=mode,
                        state_id=state.state_id,
                        transition_identity=identity,
                        boundary_snapshot_digest=boundary_digest,
                        boundary_qpos=boundary_qpos,
                        boundary_geom_transform=boundary_geom,
                        qpos=qpos,
                        geom_transform=geom_transform,
                        environment_boxes=boxes,
                        robot_specs=specs,
                    )
                    if mode == "PLANNING_TIME_CAUSAL_CLOUD":
                        planning_cloud_cache[cache_key] = cloud
                values = sparse_per_link_evidence(
                    cloud=cloud,
                    robot_specs=specs,
                    geom_transform=geom_transform,
                    qpos=qpos,
                    targets=targets,
                    evaluation_time_s=np.arange(1, STEPS + 1, dtype=np.float64) * 0.002,
                )
                for field in (
                    "nominal_fov",
                    "horizontal_fov",
                    "vertical_fov",
                    "direct_visibility",
                    "self_occluded",
                    "environment_occluded",
                    "near_blind",
                    "self_occluder_geom_index",
                    "self_occluder_link_index",
                    "target_azimuth_deg",
                    "target_elevation_deg",
                    "target_range_m",
                    "target_event_azimuth_deg",
                    "target_event_elevation_deg",
                    "target_event_range_m",
                ):
                    values[field] = auxiliary[mode][field]
                _store_evidence(evidence, representative, condition_index, mode_index, values)
                inherited_dense_support = 0
                if condition_id == "REALISTIC_PLATFORM_SCAN":
                    inherited_dense_support = inherit_realistic_support_into_dense_platform(
                        dense_c[mode], values
                    )
                    _store_evidence(
                        evidence,
                        representative,
                        CONDITIONS.index("DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT"),
                        mode_index,
                        dense_c[mode],
                    )
                scan_receipts.append(
                    {
                        "transition_index": representative,
                        "condition_id": condition_id,
                        "evidence_mode": mode,
                        "ray_count": cloud["ray_count"],
                        "environment_return_count": cloud["environment_return_count"],
                        "self_return_count": cloud["self_return_count"],
                        "near_blind_count": cloud["near_blind_count"],
                        "ground_return_count": cloud["ground_return_count"],
                        "pattern": cloud["pattern"],
                        "phase": cloud["phase"],
                        "render_cache_reused": cache_reused,
                        "dense_platform_inherited_support_witnesses": inherited_dense_support,
                    }
                )
                selected_copy_identities = [
                    str(state.transition_rows[copy]["identity"])
                    for copy in copy_map.copies_by_representative[representative]
                    if str(state.transition_rows[copy]["identity"]) in audit_identities
                ]
                for selected_identity in selected_copy_identities:
                    raw_audit.append(
                        raw_audit_artifact(
                            transition_identity=selected_identity,
                            role=state.role,
                            family=state.family,
                            transition_kind=str(row["level"]),
                            source_representative_transition_uid=identity,
                            boundary_snapshot_digest=boundary_digest,
                            condition_id=condition_id,
                            mode_id=mode,
                            payload=cloud,
                        )
                    )

        selected_copy_identities = [
            str(state.transition_rows[copy]["identity"])
            for copy in copy_map.copies_by_representative[representative]
            if str(state.transition_rows[copy]["identity"]) in audit_identities
        ]
        for selected_identity in selected_copy_identities:
            for condition_id, dense_values in (
                ("DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT", dense_c),
                ("DENSE_BODY_CENTRIC_SINGLE_ORIGIN", dense_d),
            ):
                for mode in MODES:
                    raw_audit.append(
                        raw_audit_artifact(
                            transition_identity=selected_identity,
                            role=state.role,
                            family=state.family,
                            transition_kind=str(row["level"]),
                            source_representative_transition_uid=identity,
                            boundary_snapshot_digest=boundary_digest,
                            condition_id=condition_id,
                            mode_id=mode,
                            payload={
                                "target_points": targets["target_points"],
                                "target_object_index": targets["object_index"],
                                "support": dense_values[mode]["support"],
                                "clearance_m": dense_values[mode]["clearance_m"],
                                "finite_scan_support_inherited": dense_values[mode]["finite_scan_support_inherited"],
                                "acquisition_times_s": dense_values[mode]["acquisition_times_s"],
                                "acquisition_support": dense_values[mode]["acquisition_support"],
                                "acquisition_nominal_fov": dense_values[mode]["acquisition_nominal_fov"],
                                "acquisition_self_occluded": dense_values[mode]["acquisition_self_occluded"],
                                "acquisition_self_geom_index": dense_values[mode]["acquisition_self_geom_index"],
                                "acquisition_self_link_index": dense_values[mode]["acquisition_self_link_index"],
                                "acquisition_environment_occluded": dense_values[mode]["acquisition_environment_occluded"],
                            },
                        )
                    )

        for copy in copy_map.copies_by_representative[representative]:
            if copy == representative:
                continue
            _copy_evidence(evidence, representative, copy)
            target_points[copy] = target_points[representative]
            target_object[copy] = target_object[representative]
            target_geom[copy] = target_geom[representative]
            oracle_clearance[copy] = oracle_clearance[representative]
        if ordinal % 25 == 0:
            print(
                json.dumps(
                    {
                        "state_id": state_id,
                        "representatives_done": ordinal,
                        "representatives_total": len(representatives),
                    }
                ),
                flush=True,
            )

    critical_step = np.empty(transitions, np.int16)
    critical_link = np.empty(transitions, np.int16)
    oracle_contact_step = np.full(transitions, -1, np.int16)
    oracle_contact_link = np.full(transitions, -1, np.int16)
    oracle_contact_object = np.full(transitions, -1, np.int16)
    oracle_attribution_source = np.full(transitions, "UNRESOLVED", dtype="<U24")
    oracle_attribution_timing_mismatch = np.zeros(transitions, np.uint8)
    for transition in range(transitions):
        attribution = corpus_adapter.contact_attribution(state.shard, transition)
        frozen = attribution["frozen"]
        if frozen.get("contact"):
            frozen_step = frozen.get("first_contact_step")
            if frozen_step is not None:
                oracle_contact_step[transition] = int(frozen_step)
            chosen = None
            for source_name in ("native_replay", "exact"):
                candidate = attribution[source_name]
                candidate_step = candidate.get("first_contact_step")
                timing_agrees = (
                    frozen_step is not None
                    and candidate_step is not None
                    and int(candidate_step) == int(frozen_step)
                )
                if bool(candidate.get("contact")) == bool(frozen.get("contact")) and not timing_agrees:
                    oracle_attribution_timing_mismatch[transition] = 1
                if (
                    bool(candidate.get("contact")) == bool(frozen.get("contact"))
                    and timing_agrees
                ):
                    link_name = candidate.get("robot_link_name")
                    if link_name in state.protected_link_names:
                        chosen = (source_name, candidate)
                        break
            if chosen is not None:
                source_name, candidate = chosen
                link_name = str(candidate["robot_link_name"])
                link = state.protected_link_names.index(link_name)
                oracle_contact_link[transition] = link
                object_name = candidate.get("other_link_name")
                oracle_attribution_source[transition] = source_name.upper()
            else:
                link = -1
                object_name = None
            if object_name in state.scene_boxes.object_names:
                oracle_contact_object[transition] = state.scene_boxes.object_names.index(object_name)
            flat = int(np.argmin(oracle_clearance[transition]))
            fallback_step, _fallback_link = divmod(flat, LINKS)
            critical_step[transition] = (
                oracle_contact_step[transition]
                if oracle_contact_step[transition] >= 0
                else fallback_step
            )
            # Preserve unresolved oracle link attribution as -1.  A diagnostic
            # minimum-clearance witness must never be relabelled as the
            # authoritative contact link.
            critical_link[transition] = link
        else:
            flat = int(np.argmin(oracle_clearance[transition]))
            critical_step[transition], critical_link[transition] = divmod(flat, LINKS)

    shard_path = OUTPUT_ROOT / "states" / f"{state_id}.npz"
    identity_rows = list(state.transition_rows)
    arrays = {
        **evidence,
        "target_points_world_m": target_points,
        "target_object_index": target_object,
        "target_geom_index": target_geom,
        "oracle_clearance_m": oracle_clearance,
        "critical_step": critical_step,
        "critical_link": critical_link,
        "oracle_contact_step": oracle_contact_step,
        "oracle_contact_link": oracle_contact_link,
        "oracle_contact_object": oracle_contact_object,
        "oracle_attribution_source": oracle_attribution_source,
        "oracle_attribution_timing_mismatch": oracle_attribution_timing_mismatch,
        "frozen_contact": np.asarray(state.shard.arrays["frozen_contact_label"], np.uint8),
        "exact_contact": np.any(np.asarray(state.shard.arrays["exact_contact"], bool), axis=1).astype(np.uint8),
        "native_contact": np.any(np.asarray(state.shard.arrays["native_contact"], bool), axis=1).astype(np.uint8),
        "transition_level": np.asarray([row["level"] for row in identity_rows]),
        "transition_current_action": np.asarray([row["current_action_index"] for row in identity_rows], np.int16),
        "transition_action": np.asarray([row["action_index"] for row in identity_rows], np.int16),
        "representative_transition": np.asarray(copy_map.representative_by_transition, np.int32),
        "legacy_sparse_global_clearance_m": np.min(
            np.asarray(state.shard.arrays["lidar_clearance"], np.float32), axis=(1, 2)
        ),
    }
    atomic_npz(shard_path, **arrays)
    receipt = {
        "schema": "body_centric_range_coverage_state_v1",
        "status": "PASS",
        "state_id": state_id,
        "scene_id": state.scene_id,
        "family": state.family,
        "role": state.role,
        "transitions": transitions,
        "representatives": len(representatives),
        "current_representatives": state.action_copy_map.current_representative_count,
        "successor_representatives": state.action_copy_map.successor_representative_count,
        "shard_path": str(shard_path),
        "shard_sha256": sha256_file(shard_path),
        "storage_bytes": shard_path.stat().st_size,
        "source_geometry_shard_sha256": state.geometry_record["shard_sha256"],
        "snapshot_sha256": state.source_record["snapshot_sha256"],
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "source_freeze_commit": git("rev-parse", "HEAD"),
        "applied_action_copy_validation": copy_validation,
        "scan_receipts": scan_receipts,
        "raw_audit": raw_audit,
        "oracle_attribution_timing_mismatch_count": int(
            oracle_attribution_timing_mismatch.sum()
        ),
        "runtime_s": time.time() - started,
        "simulator_steps": 0,
        "model_training_steps": 0,
        "jepa_predictor_opens": 0,
    }
    receipt["content_digest"] = content_digest(receipt)
    atomic_json(receipt_path, receipt)
    return receipt


def _materialize_state_worker(payload: tuple[str, tuple[str, ...]]) -> dict[str, Any]:
    """Spawn-safe isolated state materializer used by the frozen process pool."""

    state_id, audit_identities = payload
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    context = corpus_adapter.load_corpus_context(ROOT)
    return materialize_state(context, state_id, set(audit_identities))


def source_paths() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "config/go2_platform_manifest.yaml",
        ROOT / "lewm/__init__.py",
        ROOT / "lewm/safety/__init__.py",
        ROOT / "lewm/safety/body_centric_range_coverage_qualification_v1_contract.py",
        ROOT / "lewm/safety/body_centric_range_coverage_v1.py",
        ROOT / "lewm/safety/body_centric_range_coverage_metrics_v1.py",
        ROOT / "lewm/safety/body_centric_range_coverage_corpus_v1.py",
        ROOT / "lewm/safety/body_centric_range_coverage_reporting_v1.py",
        ROOT / "lewm/safety/body_centric_range_coverage_analysis_v1.py",
        ROOT / "scripts/evaluate_body_centric_range_coverage_qualification_v1.py",
        ROOT / "lewm/tests/test_body_centric_range_coverage_qualification_v1_contract.py",
        ROOT / "lewm/tests/test_body_centric_range_coverage_v1.py",
        ROOT / "lewm/tests/test_body_centric_range_coverage_metrics_v1.py",
        ROOT / "lewm/tests/test_body_centric_range_coverage_corpus_v1.py",
        ROOT / "lewm/tests/test_body_centric_range_coverage_reporting_v1.py",
        ROOT / "lewm/tests/test_body_centric_range_coverage_analysis_v1.py",
        ROOT / "lewm/tests/test_evaluate_body_centric_range_coverage_qualification_v1.py",
        TRACKED_PREREG,
        TRACKED_CONTRACT,
        TRACKED_SCHEMA,
    ]


def evaluator_integration_fixture_receipt() -> dict[str, Any]:
    """Deterministic executable-level fixtures absent from the geometry core."""

    from lewm.safety import body_centric_range_coverage_v1 as geometry

    specs = [
        geometry.RobotPrimitiveSpec(
            identity=f"base:{index:02d}",
            kind="sphere",
            data=(0.1,),
            local_position_xyz_m=(0.0, 0.0, 0.0),
            local_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            geom_index=index,
            link_index=0,
            link_name="base",
        )
        for index in (1, 2, 3)
    ]
    positions = np.zeros((2, 3, 3), np.float64)
    quaternions = np.zeros((2, 3, 4), np.float64)
    quaternions[..., 0] = 1.0
    retained, _, _ = self_occlusion_geometry_for_condition(
        "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
        specs,
        positions,
        quaternions,
    )
    housing_pass = [int(spec.geom_index) for spec in retained] == [3]

    qpos = np.zeros((STEPS, 19), np.float64)
    qpos[:, 3] = 1.0
    boundary_qpos = qpos[0].copy()
    blocking = np.asarray([[0.45, 0.0, 0.067, 1.0, 0.0, 0.0, 0.0]])
    boundary_visible = blocking.copy()
    boundary_visible[0, :3] = (0.0, 2.0, 0.067)
    blocked_trace = np.repeat(blocking[None], STEPS, axis=0)
    target = np.zeros((STEPS, LINKS, 3), np.float64)
    target[..., 0] = 0.9
    target[..., 2] = 0.067
    dense = dense_per_link_evidence(
        condition_id="DENSE_BODY_CENTRIC_SINGLE_ORIGIN",
        mode_id="TRUE_FUTURE_OBSERVABILITY_CLOUD",
        boundary_qpos=boundary_qpos,
        boundary_geom_transform=boundary_visible,
        qpos=qpos,
        geom_transform=blocked_trace,
        environment_boxes=[
            geometry.OrientedBox(
                "wall", (1.0, 0.0, 0.067), (0.1, 1.0, 1.0), object_index=0
            )
        ],
        robot_specs=[
            geometry.RobotPrimitiveSpec(
                identity="occluder",
                kind="sphere",
                data=(0.2,),
                local_position_xyz_m=(0.0, 0.0, 0.0),
                local_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                geom_index=0,
                link_index=0,
                link_name="base",
            )
        ],
        targets={
            "target_points": target,
            "object_index": np.zeros((STEPS, LINKS), np.int16),
            "clearance_m": np.full((STEPS, LINKS), 0.3),
        },
    )
    accumulation_pass = bool(
        dense["support"].all()
        and dense["event_time_support"].all()
        and np.all(dense["support_acquisition_index"] == 0)
        and np.all(dense["support_point_age_s"] > 0.0)
    )
    value = {
        "schema": "body_centric_range_coverage_evaluator_integration_fixture_v1",
        "pass": bool(housing_pass and accumulation_pass),
        "emitting_housing_exemption": {
            "input_geom_indices": [1, 2, 3],
            "retained_occluder_geom_indices": [int(spec.geom_index) for spec in retained],
            "pass": housing_pass,
        },
        "dense_accumulation": {
            "acquisition_count": int(len(dense["acquisition_times_s"])),
            "supported_witnesses": int(np.asarray(dense["support"], bool).sum()),
            "event_supported_witnesses": int(
                np.asarray(dense["event_time_support"], bool).sum()
            ),
            "selected_acquisition_indices": sorted(
                int(value)
                for value in np.unique(dense["support_acquisition_index"])
            ),
            "pass": accumulation_pass,
        },
    }
    value["content_digest"] = content_digest(value)
    return value


def combined_fixture_receipt(geometry: Any, metrics: Any) -> dict[str, Any]:
    geometry_receipt = geometry.run_fixtures(metrics_module=metrics)
    integration_receipt = evaluator_integration_fixture_receipt()
    value = {
        "schema": "body_centric_range_coverage_combined_fixture_v1",
        "pass": bool(
            geometry_receipt.get("pass") and integration_receipt.get("pass")
        ),
        "geometry_core": geometry_receipt,
        "evaluator_integration": integration_receipt,
    }
    value["content_digest"] = content_digest(value)
    return value


def build_source_closure() -> dict[str, Any]:
    rows = []
    for path in source_paths():
        if not path.is_file():
            raise RuntimeError(f"source closure path missing: {path}")
        rows.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    value = {
        "schema": "body_centric_range_coverage_source_closure_v1",
        "experiment": EXPERIMENT,
        "predecessor_head": PREDECESSOR_HEAD,
        "source_lineage": SOURCE_LINEAGE,
        "files": rows,
        "excludes": ["sealed material", "G2", "JEPA predictor", "model checkpoints", "training codepaths"],
        "source_freeze_commit": "RECORDED_BY_POST_COMMIT_PREFLIGHT",
    }
    value["content_digest"] = content_digest(value)
    return value


def write_freeze_receipts() -> dict[str, Any]:
    CONTRACT.write_contract(TRACKED_CONTRACT)
    CONTRACT.write_output_schema(TRACKED_SCHEMA)
    contract = CONTRACT.load_and_validate_contract(TRACKED_CONTRACT)
    output_schema = CONTRACT.load_and_validate_output_schema(TRACKED_SCHEMA)
    # The fixture module is imported only here so contract inspection remains
    # independent of any geometry computation.
    from lewm.safety import body_centric_range_coverage_v1 as geometry
    from lewm.safety import body_centric_range_coverage_metrics_v1 as metrics
    from lewm.safety import body_centric_range_coverage_analysis_v1 as _analysis
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter
    from lewm.safety import body_centric_range_coverage_reporting_v1 as _reporting
    fixture = combined_fixture_receipt(geometry, metrics)
    if not fixture.get("pass"):
        raise RuntimeError("deterministic evaluator fixtures failed")
    atomic_json(TRACKED_FIXTURE, fixture, canonical=True)
    closure = build_source_closure()
    atomic_json(TRACKED_CLOSURE, closure, canonical=True)
    return {
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "contract_content_digest": contract["contract_sha256"],
        "schema_sha256": sha256_file(TRACKED_SCHEMA),
        "schema_content_digest": output_schema["output_schema_sha256"],
        "fixture_sha256": sha256_file(TRACKED_FIXTURE),
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "source_closure_content_digest": closure["content_digest"],
    }


def preflight(*, full_shards: bool = True) -> dict[str, Any]:
    if git("status", "--porcelain=v1"):
        raise RuntimeError("preflight requires a clean worktree")
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"wrong execution branch: {branch}")
    predecessor_is_ancestor = (
        subprocess.call(
            ["git", "merge-base", "--is-ancestor", PREDECESSOR_HEAD, head], cwd=ROOT
        )
        == 0
    )
    if not predecessor_is_ancestor:
        raise RuntimeError("bound predecessor result commit is not an ancestor")
    processes = running_scientific_processes()
    if processes:
        raise RuntimeError(f"scientific process already active: {processes}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    workspace_fs = filesystem_receipt(ROOT)
    output_fs = filesystem_receipt(OUTPUT_ROOT)
    if workspace_fs["device"] == output_fs["device"]:
        raise RuntimeError("output root is not on an independent filesystem")
    if output_fs["fstype"] != "ext4":
        raise RuntimeError(f"output filesystem is not the frozen ext4 target: {output_fs['fstype']}")
    if workspace_fs["free_bytes"] < 20_000_000_000:
        raise RuntimeError("workspace has less than 20 GB free")
    if output_fs["free_bytes"] < 40_000_000_000:
        raise RuntimeError("output filesystem has less than 40 GB free")
    contract = CONTRACT.load_and_validate_contract(TRACKED_CONTRACT)
    schema = CONTRACT.load_and_validate_output_schema(TRACKED_SCHEMA)
    recovery_lineage = contract["recovery_lineage"]
    for path_key, digest_key in (
        ("final_recovery_receipt", "final_recovery_receipt_sha256"),
        ("scientific_execution_gate_receipt", "scientific_execution_gate_receipt_sha256"),
    ):
        recovery_path = Path(recovery_lineage[path_key])
        if not recovery_path.is_file() or sha256_file(recovery_path) != recovery_lineage[digest_key]:
            raise RuntimeError(f"recovery-lineage binding drift: {recovery_path}")
    hardware = contract["hardware_binding"]
    local_hardware_sources = {
        hardware["excluded_local_l1_scaffold"]["robot_xacro"]: hardware["excluded_local_l1_scaffold"]["robot_xacro_sha256"],
        hardware["excluded_local_l1_scaffold"]["lidar_xacro"]: hardware["excluded_local_l1_scaffold"]["lidar_xacro_sha256"],
        hardware["local_selection_authorities"]["platform_manifest"]: hardware["local_selection_authorities"]["platform_manifest_sha256"],
        hardware["local_selection_authorities"]["repository_readme"]: hardware["local_selection_authorities"]["repository_readme_sha256"],
    }
    for relative, expected_digest in local_hardware_sources.items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected_digest:
            raise RuntimeError(f"local hardware authority drift: {source}")
    genesis_urdf = Path(sys.prefix) / "lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf"
    if sha256_file(genesis_urdf) != contract["mounts"]["platform"]["local_genesis_urdf_sha256"]:
        raise RuntimeError("local Genesis Go2 URDF mount authority drift")
    from lewm.safety import body_centric_range_coverage_v1 as geometry
    from lewm.safety import body_centric_range_coverage_metrics_v1 as metrics
    from lewm.safety import body_centric_range_coverage_analysis_v1 as _analysis
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter
    from lewm.safety import body_centric_range_coverage_reporting_v1 as _reporting
    regenerated_fixture = combined_fixture_receipt(geometry, metrics)
    if regenerated_fixture.get("pass") is not True:
        raise RuntimeError("regenerated fixture gate failed")
    if TRACKED_FIXTURE.read_bytes() != canonical_bytes(regenerated_fixture):
        raise RuntimeError("tracked fixture receipt is not byte-identical to regeneration")
    closure = json.loads(TRACKED_CLOSURE.read_text())
    closure_without_digest = dict(closure)
    declared_closure_digest = closure_without_digest.pop("content_digest", None)
    if declared_closure_digest != content_digest(closure_without_digest):
        raise RuntimeError("source-closure content digest mismatch")
    for row in closure["files"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"source closure mismatch: {path}")
    environment = environment_receipt()
    environment["content_digest"] = content_digest(environment)
    inputs = validate_frozen_inputs(full_shards=full_shards)
    corpus_context = corpus_adapter.load_corpus_context(ROOT)
    boundary_smoke = corpus_context.load_boundary_transforms(corpus_context.state_ids[0])
    if boundary_smoke.current.qpos.shape != (19,) or boundary_smoke.current.contract_geom_transform.shape != (27, 7):
        raise RuntimeError("snapshot boundary import-closure smoke failed")
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported by the no-training boundary closure")
    platform_housing = platform_mount_housing_preflight(
        corpus_context, corpus_adapter, geometry
    )
    receipt = {
        "schema": "body_centric_range_coverage_preexecution_receipt_v1",
        "experiment": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "head": head,
        "branch": branch,
        "worktree_clean": True,
        "predecessor_head_is_ancestor": predecessor_is_ancestor,
        "contract_path": str(TRACKED_CONTRACT),
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "contract_content_digest": contract["contract_sha256"],
        "output_schema_sha256": sha256_file(TRACKED_SCHEMA),
        "output_schema_content_digest": schema["output_schema_sha256"],
        "fixture_sha256": sha256_file(TRACKED_FIXTURE),
        "fixture_content_digest": regenerated_fixture["content_digest"],
        "source_closure_path": str(TRACKED_CLOSURE),
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "source_freeze_commit": head,
        "inputs": inputs,
        "environment": environment,
        "boundary_snapshot_smoke": {
            "state_id": boundary_smoke.state_id,
            "current_snapshot_digest": boundary_smoke.current.snapshot_digest,
            "qpos_shape": list(boundary_smoke.current.qpos.shape),
            "geometry_shape": list(boundary_smoke.current.contract_geom_transform.shape),
            "pass": True,
        },
        "platform_emitting_housing_smoke": platform_housing,
        "recovery_lineage": recovery_lineage,
        "hardware_source_bindings": {
            "local_files": local_hardware_sources,
            "genesis_go2_urdf": {
                "path": str(genesis_urdf),
                "sha256": sha256_file(genesis_urdf),
            },
        },
        "workspace_filesystem": workspace_fs,
        "output_filesystem": output_fs,
        "temporary_storage_ceiling_bytes": 30_000_000_000,
        "final_storage_ceiling_bytes": 20_000_000_000,
        "predicted_temporary_storage_bytes": 18_000_000_000,
        "predicted_final_storage_bytes": 14_000_000_000,
        "scientific_processes_before": processes,
        "training_authorized": False,
        "jepa_predictor_access_authorized": False,
        "untouched_g2_access_authorized": False,
        "pass": True,
    }
    receipt["content_digest"] = content_digest(receipt)
    atomic_json(OUTPUT_ROOT / "receipts" / "environment_receipt.json", environment)
    atomic_json(OUTPUT_ROOT / "preexecution_receipt.json", receipt)
    return receipt


def validate_preexecution_receipt() -> dict[str, Any]:
    path = OUTPUT_ROOT / "preexecution_receipt.json"
    if not path.is_file():
        raise RuntimeError("preexecution receipt is missing; run preflight after the freeze commit")
    receipt = json.loads(path.read_text())
    if receipt.get("pass") is not True:
        raise RuntimeError("preexecution receipt does not pass")
    if receipt.get("head") != git("rev-parse", "HEAD"):
        raise RuntimeError("preexecution HEAD no longer matches the current source")
    if receipt.get("contract_sha256") != sha256_file(TRACKED_CONTRACT):
        raise RuntimeError("preexecution contract binding drift")
    if receipt.get("source_closure_sha256") != sha256_file(TRACKED_CLOSURE):
        raise RuntimeError("preexecution source-closure binding drift")
    if int(receipt.get("inputs", {}).get("geometry_shard_hashes_checked", -1)) != EXPECTED["states"]:
        raise RuntimeError("scientific execution requires a full 176-shard preflight")
    if git("status", "--porcelain=v1"):
        raise RuntimeError("scientific execution requires a clean worktree")
    if running_scientific_processes():
        raise RuntimeError("another scientific process is active")
    workspace_fs = filesystem_receipt(ROOT)
    output_fs = filesystem_receipt(OUTPUT_ROOT)
    if workspace_fs["device"] == output_fs["device"] or output_fs["fstype"] != "ext4":
        raise RuntimeError("execution filesystem binding no longer satisfies the contract")
    if workspace_fs["free_bytes"] < 20_000_000_000:
        raise RuntimeError("workspace has less than 20 GB free at execution time")
    if output_fs["free_bytes"] < 40_000_000_000:
        raise RuntimeError("output filesystem has less than 40 GB free at execution time")
    return receipt


def _directory_allocated_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    output = subprocess.check_output(["du", "-x", "-B1", "-s", str(path)], text=True)
    return int(output.split()[0])


def _validate_materialization_index_artifacts(
    context: Any, index: dict[str, Any]
) -> dict[str, Any]:
    core = dict(index)
    declared_digest = core.pop("content_digest", None)
    if declared_digest != content_digest(core) or index.get("status") != "PASS":
        raise RuntimeError("materialization index self-digest/pass failure")
    for key, expected in (
        ("head", git("rev-parse", "HEAD")),
        ("contract_sha256", sha256_file(TRACKED_CONTRACT)),
        ("source_closure_sha256", sha256_file(TRACKED_CLOSURE)),
        ("states", EXPECTED["states"]),
        ("transitions", EXPECTED["transitions"]),
    ):
        if index.get(key) != expected:
            raise RuntimeError(f"materialization {key} binding drift")
    records = list(index.get("records", []))
    if [row.get("state_id") for row in records] != list(context.state_ids):
        raise RuntimeError("materialization state-record order/cardinality drift")
    expected_raw_rows: list[dict[str, Any]] = []
    for record in records:
        receipt = _validate_state_receipt_artifacts(
            OUTPUT_ROOT / "states" / f"{record['state_id']}.json",
            context=context,
        )
        for key in (
            "scene_id", "family", "role", "transitions", "representatives",
            "shard_path", "shard_sha256", "storage_bytes",
        ):
            if record.get(key) != receipt.get(key):
                raise RuntimeError(
                    f"materialization/state receipt mismatch: {record['state_id']}:{key}"
                )
        expected_raw_rows.extend(receipt["raw_audit"])
    manifest = Path(index["raw_audit_manifest_path"])
    if not manifest.is_file() or sha256_file(manifest) != index["raw_audit_manifest_sha256"]:
        raise RuntimeError("materialization raw-audit manifest drift")
    observed_raw_rows = [
        json.loads(line) for line in manifest.read_text().splitlines() if line
    ]
    if (
        observed_raw_rows != expected_raw_rows
        or len(observed_raw_rows) != int(index["raw_audit_artifacts"])
    ):
        raise RuntimeError("materialization raw-audit manifest content/cardinality drift")
    frozen_selection = frozen_audit_subset(context)
    if index.get("audit_selection_by_role") != frozen_selection:
        raise RuntimeError("materialization raw-audit frozen selection drift")
    selected_identities = {
        identity for identities in frozen_selection.values() for identity in identities
    }
    for row in observed_raw_rows:
        if row["transition_uid"] not in selected_identities:
            raise RuntimeError("raw-audit row is outside the frozen selected subset")
        expected_rank = audit_selection_sha256(
            transition_uid=row["transition_uid"],
            role=row["role"],
            family=row["family"],
            transition_kind=row["transition_kind"],
        )
        if row["selection_sha256"] != expected_rank:
            raise RuntimeError("raw-audit selection-rank binding drift")
    return index


def materialize() -> dict[str, Any]:
    """Stream all frozen states into compact per-state sensor evidence shards."""

    preexecution = validate_preexecution_receipt()
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    context = corpus_adapter.load_corpus_context(ROOT)

    existing_path = OUTPUT_ROOT / "materialization_index.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text())
        _validate_materialization_index_artifacts(context, existing)
        return {**existing, "status": "REUSED"}
    audit_by_role = frozen_audit_subset(context)
    audit_identities = {value for rows in audit_by_role.values() for value in rows}
    marker = OUTPUT_ROOT / "MATERIALIZATION_RUNNING.json"
    atomic_json(
        marker,
        {
            "schema": "body_centric_range_coverage_materialization_running_v1",
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "head": preexecution["head"],
        },
    )
    started = time.time()
    records: list[dict[str, Any]] = []
    try:
        worker_count = 12
        numeric_thread_environment = {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        os.environ.update(numeric_thread_environment)
        records_by_state: dict[str, dict[str, Any]] = {}
        process_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count, mp_context=process_context
        ) as executor:
            futures = {
                executor.submit(
                    _materialize_state_worker,
                    (state_id, tuple(sorted(audit_identities))),
                ): state_id
                for state_id in context.state_ids
            }
            for ordinal, future in enumerate(
                concurrent.futures.as_completed(futures), 1
            ):
                state_id = futures[future]
                receipt = future.result()
                records_by_state[state_id] = {
                    key: receipt[key]
                    for key in (
                        "state_id",
                        "scene_id",
                        "family",
                        "role",
                        "transitions",
                        "representatives",
                        "shard_path",
                        "shard_sha256",
                        "storage_bytes",
                        "runtime_s",
                    )
                }
                allocated = _directory_allocated_bytes(OUTPUT_ROOT)
                if allocated > 20_000_000_000:
                    raise RuntimeError(
                        "final experiment storage ceiling exceeded during materialization"
                    )
                print(
                    json.dumps(
                        {
                            "materialized_states": ordinal,
                            "total_states": len(context.state_ids),
                            "state_id": state_id,
                            "output_allocated_bytes": allocated,
                        }
                    ),
                    flush=True,
                )
        records = [records_by_state[state_id] for state_id in context.state_ids]
        raw_rows: list[dict[str, Any]] = []
        scan_totals: dict[str, dict[str, dict[str, int]]] = {
            condition_id: {
                mode_id: {
                    "representative_scans": 0,
                    "unique_rendered_scans": 0,
                    "rays": 0,
                    "unique_rendered_rays": 0,
                    "environment_returns": 0,
                    "robot_self_returns": 0,
                    "near_blind_first_hits": 0,
                    "ground_returns": 0,
                }
                for mode_id in MODES
            }
            for condition_id in CONDITIONS[:2]
        }
        for state_id in context.state_ids:
            state_receipt = json.loads((OUTPUT_ROOT / "states" / f"{state_id}.json").read_text())
            raw_rows.extend(state_receipt["raw_audit"])
            for scan in state_receipt["scan_receipts"]:
                total = scan_totals[scan["condition_id"]][scan["evidence_mode"]]
                total["representative_scans"] += 1
                total["unique_rendered_scans"] += int(
                    not bool(scan.get("render_cache_reused", False))
                )
                total["rays"] += int(scan["ray_count"])
                total["unique_rendered_rays"] += int(scan["ray_count"]) * int(
                    not bool(scan.get("render_cache_reused", False))
                )
                total["environment_returns"] += int(scan["environment_return_count"])
                total["robot_self_returns"] += int(scan["self_return_count"])
                total["near_blind_first_hits"] += int(scan["near_blind_count"])
                total["ground_returns"] += int(scan["ground_return_count"])
        raw_manifest = OUTPUT_ROOT / "raw_audit" / "manifest.jsonl"
        raw_manifest.parent.mkdir(parents=True, exist_ok=True)
        atomic_bytes(raw_manifest, b"".join(canonical_bytes(row) for row in raw_rows))
        index = {
            "schema": "body_centric_range_coverage_materialization_index_v1",
            "experiment": EXPERIMENT,
            "status": "PASS",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "head": preexecution["head"],
            "contract_sha256": sha256_file(TRACKED_CONTRACT),
            "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
            "corpus_binding": dict(context.binding_receipt),
            "states": len(records),
            "transitions": sum(int(row["transitions"]) for row in records),
            "representatives": sum(int(row["representatives"]) for row in records),
            "conditions": len(CONDITIONS),
            "evidence_modes": len(MODES),
            "physics_steps": EXPECTED["physics_steps"],
            "protected_links": LINKS,
            "audit_selection_by_role": audit_by_role,
            "audit_transition_count": len(audit_identities),
            "raw_audit_artifacts": len(raw_rows),
            "scan_materialization_counts": scan_totals,
            "materialization_workers": worker_count,
            "process_start_method": "spawn",
            "numeric_thread_environment": numeric_thread_environment,
            "raw_audit_manifest_path": str(raw_manifest),
            "raw_audit_manifest_sha256": sha256_file(raw_manifest),
            "records": records,
            "runtime_s": time.time() - started,
            "storage_bytes": _directory_allocated_bytes(OUTPUT_ROOT),
            "simulator_steps": 0,
            "model_training_steps": 0,
            "jepa_predictor_opens": 0,
            "g2_opens": 0,
        }
        if index["states"] != EXPECTED["states"] or index["transitions"] != EXPECTED["transitions"]:
            raise RuntimeError("materialization cardinality mismatch")
        index["content_digest"] = content_digest(index)
        atomic_json(existing_path, index)
        return index
    finally:
        marker.unlink(missing_ok=True)


def load_materialization_index() -> dict[str, Any]:
    path = OUTPUT_ROOT / "materialization_index.json"
    if not path.is_file():
        raise RuntimeError("materialization index is missing")
    index = json.loads(path.read_text())
    preexecution = validate_preexecution_receipt()
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    context = corpus_adapter.load_corpus_context(ROOT)
    for key, expected in (
        ("head", preexecution["head"]),
        ("contract_sha256", sha256_file(TRACKED_CONTRACT)),
        ("source_closure_sha256", sha256_file(TRACKED_CLOSURE)),
    ):
        if index.get(key) != expected:
            raise RuntimeError(f"materialization {key} binding drift")
    return _validate_materialization_index_artifacts(context, index)


def _state_evidence_path(state_id: str) -> Path:
    return OUTPUT_ROOT / "states" / f"{state_id}.npz"


def _identity_rows_with_oracle(context: Any, state_id: str) -> list[dict[str, Any]]:
    rows = []
    for source in context.transition_identity_rows(state_id):
        row = dict(source)
        row["oracle_contact"] = bool(row["frozen_contact"])
        rows.append(row)
    return rows


def validate_threshold_freeze_for_heldout() -> dict[str, Any]:
    path = OUTPUT_ROOT / "calibration" / "thresholds_frozen.json"
    if not path.is_file():
        raise RuntimeError("heldout load attempted before aggregate threshold freeze")
    receipt = json.loads(path.read_text())
    core = dict(receipt)
    declared = core.pop("content_digest", None)
    if receipt.get("pass") is not True or declared != content_digest(core):
        raise RuntimeError("aggregate threshold-freeze receipt is invalid")
    materialization_path = OUTPUT_ROOT / "materialization_index.json"
    if receipt.get("materialization_index_sha256") != sha256_file(materialization_path):
        raise RuntimeError("threshold-freeze materialization binding drift")
    for row in receipt.get("calibration_state_evidence", []):
        shard = Path(row["path"])
        if not shard.is_file() or sha256_file(shard) != row["sha256"]:
            raise RuntimeError(f"threshold calibration shard drift: {shard}")
    return receipt


def load_condition_mode_records(
    context: Any,
    *,
    condition_index: int,
    mode_index: int,
    roles: tuple[str, ...] = ("calibration", "heldout"),
) -> list[dict[str, Any]]:
    """Load one analysis slice without retaining unrelated conditions/modes."""

    from lewm.safety import body_centric_range_coverage_analysis_v1 as analysis
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    if "heldout" in roles:
        validate_threshold_freeze_for_heldout()

    output: list[dict[str, Any]] = []
    for role in roles:
        for state_id in context.state_ids_for_role(role):
            identities = _identity_rows_with_oracle(context, state_id)
            with np.load(_state_evidence_path(state_id), allow_pickle=False) as archive:
                clearance = np.asarray(
                    archive["clearance_m"][:, condition_index, mode_index], np.float64
                )
                support = np.asarray(
                    archive["support"][:, condition_index, mode_index], bool
                )
                nominal = np.asarray(
                    archive["nominal_fov"][:, condition_index, mode_index], bool
                )
                direct = np.asarray(
                    archive["direct_visibility"][:, condition_index, mode_index], bool
                )
                point_count = np.asarray(
                    archive["point_support_count"][:, condition_index, mode_index], np.float64
                )
                oracle_link = np.asarray(archive["oracle_contact_link"], np.int16)
                oracle_step = np.asarray(archive["oracle_contact_step"], np.int16)
            global_clearance = np.min(clearance, axis=(1, 2))
            unsupported = ~np.all(support, axis=(1, 2))
            link_contact = np.zeros(clearance.shape, bool)
            for transition, (step, link) in enumerate(zip(oracle_step, oracle_link, strict=True)):
                if step >= 0 and link >= 0:
                    link_contact[transition, int(step), int(link)] = True
            names = context.protected_link_names(state_id)
            records = analysis.transition_records_from_arrays(
                identities,
                clearance_m=global_clearance,
                unsupported=unsupported,
                link_names=names,
                per_link_clearance_m=clearance,
                per_link_support=support,
                per_link_contact=link_contact,
                per_link_nominal_fov=nominal,
                per_link_direct_visibility=direct,
                per_link_point_support_count=point_count,
                body_region_by_link=corpus_adapter.BODY_REGION_BY_LINK,
            )
            for transition, record in enumerate(records):
                record["oracle_contact_link_resolved"] = bool(
                    not identities[transition]["oracle_contact"]
                    or int(oracle_link[transition]) >= 0
                )
            if not np.array_equal(
                unsupported,
                np.asarray(
                    [
                        any(
                            float(link["unsupported_swept_volume_fraction"]) > 0.0
                            for link in row["per_link"].values()
                        )
                        for row in records
                    ],
                    bool,
                ),
            ):
                raise RuntimeError("transition/per-link unsupported aggregation mismatch")
            output.extend(records)
    return output


def _persist_calibration_frontier(
    condition_id: str, mode_id: str, frontier: dict[str, Any]
) -> dict[str, Any]:
    relative = Path("calibration") / f"{condition_id}__{mode_id}.json.gz"
    path = OUTPUT_ROOT / relative
    payload = canonical_bytes(frontier)
    atomic_bytes(path, gzip.compress(payload, compresslevel=6, mtime=0))
    return {
        "condition_id": condition_id,
        "evidence_mode": mode_id,
        "selected_threshold_m": float(frontier["selected_threshold_m"]),
        "frontier_points": int(frontier["frontier_points"]),
        "eligible_points": int(frontier["eligible_points"]),
        "selection_key": list(frontier["selection_key"]),
        "selected": frontier["selected"],
        "artifact_relative_path": str(relative),
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": path.stat().st_size,
        "uncompressed_content_sha256": hashlib.sha256(payload).hexdigest(),
    }


def analyze_all_conditions(context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from lewm.safety import body_centric_range_coverage_analysis_v1 as analysis

    thresholds: dict[str, Any] = {}
    condition_metrics: dict[str, Any] = {}
    # Pass 1 is calibration-only.  Persist every selected frontier and one
    # aggregate freeze receipt before any development-held-out row is loaded.
    for condition_index, condition_id in enumerate(CONDITIONS):
        thresholds[condition_id] = {}
        condition_metrics[condition_id] = {}
        for mode_index, mode_id in enumerate(MODES):
            print(
                json.dumps({"calibration": "START", "condition_id": condition_id, "evidence_mode": mode_id}),
                flush=True,
            )
            records = load_condition_mode_records(
                context,
                condition_index=condition_index,
                mode_index=mode_index,
                roles=("calibration",),
            )
            frontier = analysis.calibrate_threshold(records, role="calibration")
            threshold_receipt = _persist_calibration_frontier(condition_id, mode_id, frontier)
            thresholds[condition_id][mode_id] = threshold_receipt
            print(
                json.dumps(
                    {
                        "calibration": "PASS",
                        "condition_id": condition_id,
                        "evidence_mode": mode_id,
                        "threshold_m": threshold_receipt["selected_threshold_m"],
                    }
                ),
                flush=True,
            )
    threshold_freeze = {
        "schema": "body_centric_range_coverage_threshold_freeze_v1",
        "experiment": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "head": git("rev-parse", "HEAD"),
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "materialization_index_sha256": sha256_file(
            OUTPUT_ROOT / "materialization_index.json"
        ),
        "calibration_state_evidence": [
            {
                "state_id": state_id,
                "path": str(_state_evidence_path(state_id)),
                "sha256": sha256_file(_state_evidence_path(state_id)),
            }
            for state_id in context.state_ids_for_role("calibration")
        ],
        "calibration_role": "internal_calibration",
        "calibration_state_count": len(context.state_ids_for_role("calibration")),
        "development_heldout_rows_used_by_threshold_selection": 0,
        "heldout_label_materialization_precedes_threshold_selection": True,
        "thresholds": thresholds,
        "pass": True,
    }
    threshold_freeze["content_digest"] = content_digest(threshold_freeze)
    atomic_json(OUTPUT_ROOT / "calibration" / "thresholds_frozen.json", threshold_freeze)

    # Pass 2 begins only after the complete threshold receipt is persisted and
    # revalidated against materialization and every calibration evidence shard.
    validate_threshold_freeze_for_heldout()
    expected_oracle_progress = float(
        CONTRACT.build_contract()["predecessor_bindings"]
        ["development_heldout_exact_unique_action_h3_progress_m"]
    )
    for condition_index, condition_id in enumerate(CONDITIONS):
        for mode_index, mode_id in enumerate(MODES):
            print(
                json.dumps({"heldout_evaluation": "START", "condition_id": condition_id, "evidence_mode": mode_id}),
                flush=True,
            )
            records = load_condition_mode_records(
                context,
                condition_index=condition_index,
                mode_index=mode_index,
                roles=("heldout",),
            )
            evaluation = analysis.summarize_condition_mode(
                records,
                None,
                float(thresholds[condition_id][mode_id]["selected_threshold_m"]),
                role="heldout",
            )
            observed_oracle_progress = float(
                evaluation["viability"]["oracle_h3_route_progress_m"]
            )
            if not math.isclose(
                observed_oracle_progress,
                expected_oracle_progress,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "development-heldout exact H3 authority drift: "
                    f"{observed_oracle_progress} != {expected_oracle_progress}"
                )
            condition_metrics[condition_id][mode_id] = evaluation
            print(
                json.dumps({"heldout_evaluation": "PASS", "condition_id": condition_id, "evidence_mode": mode_id}),
                flush=True,
            )
    return thresholds, condition_metrics


def _boundary_digest_for_identity(context: Any, state_id: str, row: dict[str, Any]) -> str:
    source = context.corpus_record(state_id)
    if row["level"] == "current":
        return str(source["current_boundary"]["snapshot_digest"])
    prefix = int(row["current_action_index"])
    successor = next(
        value for value in source["successor_rows"] if int(value["current_action_index"]) == prefix
    )
    return str(successor["boundary"]["snapshot_digest"])


def _obstacle_sector(angle_rad: float) -> str | None:
    if not math.isfinite(float(angle_rad)):
        return None
    names = (
        "FRONT",
        "FRONT_LEFT",
        "LEFT",
        "REAR_LEFT",
        "REAR",
        "REAR_RIGHT",
        "RIGHT",
        "FRONT_RIGHT",
    )
    wrapped = (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
    index = int(math.floor((wrapped + math.pi / 8.0) / (math.pi / 4.0))) % 8
    return names[index]


def _finite_float_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _state_decision_lookup(condition_metrics: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for condition_id in CONDITIONS:
        for mode_id in MODES:
            for row in condition_metrics[condition_id][mode_id]["per_state_decisions"]:
                output[(condition_id, mode_id, str(row["state_id"]))] = row
    return output


def persist_row_level_evidence(
    context: Any,
    thresholds: dict[str, Any],
    condition_metrics: dict[str, Any],
) -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_metrics_v1 as metrics
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    evidence_root = OUTPUT_ROOT / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    transition_path = evidence_root / "transition_evidence.jsonl"
    link_path = evidence_root / "per_link_evidence.jsonl"
    transition_tmp = transition_path.with_name(f".{transition_path.name}.tmp-{os.getpid()}")
    link_tmp = link_path.with_name(f".{link_path.name}.tmp-{os.getpid()}")
    decisions = _state_decision_lookup(condition_metrics)
    transition_count = 0
    link_count = 0
    with transition_tmp.open("wb") as transition_stream, link_tmp.open("wb") as link_stream:
        for state_ordinal, state_id in enumerate(context.state_ids, 1):
            identities = _identity_rows_with_oracle(context, state_id)
            action_copy_map = context.action_copy_map(state_id)
            successor_prefixes = {
                int(row["current_action_index"])
                for row in context.corpus_record(state_id)["successor_rows"]
            }
            link_names = context.protected_link_names(state_id)
            scene_object_names = context.scene_obbs(state_id).object_names
            with np.load(_state_evidence_path(state_id), allow_pickle=False) as archive:
                for condition_index, condition_id in enumerate(CONDITIONS):
                    for mode_index, mode_id in enumerate(MODES):
                        threshold = float(
                            thresholds[condition_id][mode_id]["selected_threshold_m"]
                        )
                        clearance = np.asarray(
                            archive["clearance_m"][:, condition_index, mode_index], np.float64
                        )
                        support = np.asarray(
                            archive["support"][:, condition_index, mode_index], bool
                        )
                        global_clearance = np.min(clearance, axis=(1, 2))
                        unsupported = ~np.all(support, axis=(1, 2))
                        predicted = metrics.contact_predictions(
                            global_clearance, threshold, unsupported
                        )
                        direction = np.asarray(
                            archive["obstacle_direction_body_rad"][:, condition_index, mode_index],
                            np.float64,
                        )
                        responsible_object = np.asarray(
                            archive["responsible_object_index"][:, condition_index, mode_index],
                            np.int16,
                        )
                        nominal = np.asarray(
                            archive["nominal_fov"][:, condition_index, mode_index], bool
                        )
                        direct = np.asarray(
                            archive["direct_visibility"][:, condition_index, mode_index], bool
                        )
                        point_count = np.asarray(
                            archive["point_support_count"][:, condition_index, mode_index],
                            np.int16,
                        )
                        inherited_support = np.asarray(
                            archive["finite_scan_support_inherited"][:, condition_index, mode_index],
                            bool,
                        )
                        support_ray = np.asarray(
                            archive["support_nearest_ray_index"][:, condition_index, mode_index],
                            np.int32,
                        )
                        support_age = np.asarray(
                            archive["support_point_age_s"][:, condition_index, mode_index],
                            np.float64,
                        )
                        support_range = np.asarray(
                            archive["support_nearest_range_m"][:, condition_index, mode_index],
                            np.float64,
                        )
                        responsible_ray = np.asarray(
                            archive["nearest_ray_index"][:, condition_index, mode_index],
                            np.int32,
                        )
                        responsible_age = np.asarray(
                            archive["point_age_s"][:, condition_index, mode_index],
                            np.float64,
                        )
                        responsible_range = np.asarray(
                            archive["nearest_point_range_m"][:, condition_index, mode_index],
                            np.float64,
                        )
                        responsible_acquisition = np.asarray(
                            archive["responsible_acquisition_index"][:, condition_index, mode_index],
                            np.int16,
                        )
                        support_acquisition = np.asarray(
                            archive["support_acquisition_index"][:, condition_index, mode_index],
                            np.int16,
                        )
                        oracle_link = np.asarray(archive["oracle_contact_link"], np.int16)
                        decision = decisions.get((condition_id, mode_id, state_id))
                        for transition_index, identity in enumerate(identities):
                            boundary_digest = _boundary_digest_for_identity(context, state_id, identity)
                            phase = (
                                CONTRACT.derive_l2_scan_phases(
                                    boundary_snapshot_digest=boundary_digest
                                )
                                if condition_id == "REALISTIC_PLATFORM_SCAN"
                                else None
                            )
                            current = identity["level"] == "current"
                            action_index = int(identity["action_index"])
                            predicted_safe_count = None
                            oracle_safe_count = None
                            admitted = None
                            successor_set_available = (
                                int(identity["action_index"]) in successor_prefixes
                                if current
                                else None
                            )
                            decision_role_scored = bool(current and decision is not None)
                            if current and decision is not None:
                                representative_action = int(
                                    action_copy_map.current_action_representative[
                                        action_index
                                    ]
                                )
                                raw_predicted_safe_count = int(
                                    decision["predicted_safe_counts"][representative_action]
                                )
                                raw_oracle_safe_count = int(
                                    decision["true_safe_counts"][representative_action]
                                )
                                successor_set_available = bool(
                                    raw_predicted_safe_count >= 0
                                    and raw_oracle_safe_count >= 0
                                )
                                if successor_set_available:
                                    predicted_safe_count = raw_predicted_safe_count
                                    oracle_safe_count = raw_oracle_safe_count
                                    admitted = bool(
                                        decision["admitted"][representative_action]
                                    )
                            transition_row = {
                                "transition_uid": identity["identity"],
                                "state_id": state_id,
                                "transition_index": transition_index,
                                "transition_level": identity["level"],
                                "current_action_index": int(identity["current_action_index"]),
                                "action_index": action_index,
                                "successor_state_id": identity.get("successor_identity") if current else None,
                                "applied_action_id": applied_action_id(identity),
                                "boundary_snapshot_digest": boundary_digest,
                                "phase_digest_sha256": None if phase is None else phase["phase_digest_sha256"],
                                "role": identity["role"],
                                "family": identity["family"],
                                "condition_id": condition_id,
                                "evidence_mode": mode_id,
                                "oracle_contact": bool(identity["oracle_contact"]),
                                "sensor_global_minimum_clearance_m": float(global_clearance[transition_index]),
                                "unsupported_risk": bool(unsupported[transition_index]),
                                "threshold_m": threshold,
                                "predicted_contact": bool(predicted[transition_index]),
                                "successor_set_available": successor_set_available,
                                "decision_role_scored": decision_role_scored,
                                "predicted_safe_next_action_count": predicted_safe_count,
                                "oracle_safe_next_action_count": oracle_safe_count,
                                "admitted": admitted,
                                "per_link_row_count": LINKS,
                            }
                            transition_stream.write(canonical_bytes(transition_row))
                            transition_count += 1
                            for link_index, link_name in enumerate(link_names):
                                series = clearance[transition_index, :, link_index]
                                finite = np.isfinite(series)
                                minimum_step = int(np.argmin(series)) if finite.any() else -1
                                crossings = np.flatnonzero(finite & (series <= threshold))
                                first_crossing = None if not len(crossings) else int(crossings[0])
                                object_index = (
                                    -1
                                    if minimum_step < 0
                                    else int(responsible_object[transition_index, minimum_step, link_index])
                                )
                                object_name = (
                                    None
                                    if object_index < 0
                                    else scene_object_names[object_index]
                                )
                                angle = (
                                    math.nan
                                    if minimum_step < 0
                                    else float(direction[transition_index, minimum_step, link_index])
                                )
                                link_row = {
                                    "transition_uid": identity["identity"],
                                    "state_id": state_id,
                                    "transition_index": transition_index,
                                    "condition_id": condition_id,
                                    "evidence_mode": mode_id,
                                    "protected_link": link_name,
                                    "collision_region": corpus_adapter.BODY_REGION_BY_LINK[link_name],
                                    "minimum_observed_environment_clearance_m": (
                                        None if minimum_step < 0 else float(series[minimum_step])
                                    ),
                                    "time_to_minimum_clearance_s": (
                                        None if minimum_step < 0 else (minimum_step + 1) * 0.002
                                    ),
                                    "first_threshold_crossing_time_s": (
                                        None if first_crossing is None else (first_crossing + 1) * 0.002
                                    ),
                                    "obstacle_direction_body_rad": None if not math.isfinite(angle) else angle,
                                    "obstacle_sector": _obstacle_sector(angle),
                                    "observation_support": bool(
                                        np.all(support[transition_index, :, link_index])
                                    ),
                                    "unsupported_swept_volume_fraction": float(
                                        1.0 - support[transition_index, :, link_index].mean()
                                    ),
                                    "responsible_environment_object": object_name,
                                    "oracle_contact_link": (
                                        link_name
                                        if int(oracle_link[transition_index]) == link_index
                                        else None
                                    ),
                                    "nominal_fov_inclusion": float(
                                        nominal[transition_index, :, link_index].mean()
                                    ),
                                    "direct_visibility_after_self_occlusion": float(
                                        direct[transition_index, :, link_index].mean()
                                    ),
                                    "point_support_count": int(
                                        point_count[transition_index, :, link_index].sum()
                                    ),
                                    "finite_scan_support_inherited_fraction": float(
                                        inherited_support[
                                            transition_index, :, link_index
                                        ].mean()
                                    ),
                                    "responsible_ray_or_point": (
                                        None
                                        if minimum_step < 0
                                        or responsible_ray[transition_index, minimum_step, link_index] < 0
                                        else int(responsible_ray[transition_index, minimum_step, link_index])
                                    ),
                                    "responsible_point_age_s": (
                                        None
                                        if minimum_step < 0
                                        else _finite_float_or_none(
                                            responsible_age[transition_index, minimum_step, link_index]
                                        )
                                    ),
                                    "responsible_point_range_m": (
                                        None
                                        if minimum_step < 0
                                        else _finite_float_or_none(
                                            responsible_range[transition_index, minimum_step, link_index]
                                        )
                                    ),
                                    "responsible_acquisition_index": (
                                        None
                                        if minimum_step < 0
                                        or responsible_acquisition[
                                            transition_index, minimum_step, link_index
                                        ] < 0
                                        else int(
                                            responsible_acquisition[
                                                transition_index, minimum_step, link_index
                                            ]
                                        )
                                    ),
                                    "support_ray_or_point": (
                                        None
                                        if minimum_step < 0
                                        or support_ray[transition_index, minimum_step, link_index] < 0
                                        else int(support_ray[transition_index, minimum_step, link_index])
                                    ),
                                    "support_point_age_s": (
                                        None
                                        if minimum_step < 0
                                        else _finite_float_or_none(
                                            support_age[transition_index, minimum_step, link_index]
                                        )
                                    ),
                                    "support_point_range_m": (
                                        None
                                        if minimum_step < 0
                                        else _finite_float_or_none(
                                            support_range[transition_index, minimum_step, link_index]
                                        )
                                    ),
                                    "support_acquisition_index": (
                                        None
                                        if minimum_step < 0
                                        or support_acquisition[
                                            transition_index, minimum_step, link_index
                                        ] < 0
                                        else int(
                                            support_acquisition[
                                                transition_index, minimum_step, link_index
                                            ]
                                        )
                                    ),
                                }
                                link_stream.write(canonical_bytes(link_row))
                                link_count += 1
            print(
                json.dumps(
                    {"evidence_states": state_ordinal, "total_states": len(context.state_ids)}
                ),
                flush=True,
            )
    os.replace(transition_tmp, transition_path)
    os.replace(link_tmp, link_path)
    expected_transition_rows = EXPECTED["transitions"] * len(CONDITIONS) * len(MODES)
    expected_link_rows = expected_transition_rows * LINKS
    if transition_count != expected_transition_rows or link_count != expected_link_rows:
        raise RuntimeError("row-level evidence cardinality mismatch")
    return {
        "schema": "body_centric_range_coverage_evidence_persistence_v1",
        "materialization_index_sha256": sha256_file(
            OUTPUT_ROOT / "materialization_index.json"
        ),
        "threshold_freeze_sha256": sha256_file(
            OUTPUT_ROOT / "calibration" / "thresholds_frozen.json"
        ),
        "transition_evidence": {
            "path": str(transition_path),
            "rows": transition_count,
            "bytes": transition_path.stat().st_size,
            "sha256": sha256_file(transition_path),
        },
        "per_link_evidence": {
            "path": str(link_path),
            "rows": link_count,
            "bytes": link_path.stat().st_size,
            "sha256": sha256_file(link_path),
        },
    }


def _realistic_spatial_support(
    *,
    boundary_snapshot_digest: str,
    mode_id: str,
    target_azimuth_deg: float,
    target_elevation_deg: float,
    target_range_m: float,
    dense_direct_support: bool,
) -> bool:
    if not dense_direct_support or not all(
        math.isfinite(value)
        for value in (target_azimuth_deg, target_elevation_deg, target_range_m)
    ):
        return False
    from lewm.safety import body_centric_range_coverage_v1 as geometry

    phase = CONTRACT.derive_l2_scan_phases(
        boundary_snapshot_digest=boundary_snapshot_digest
    )
    horizontal = float(phase["horizontal_phase_cycles"])
    vertical = float(phase["vertical_phase_cycles"])
    if mode_id == "PLANNING_TIME_CAUSAL_CLOUD":
        horizontal -= 5.55 * 0.1
        vertical -= 216.0 * 0.1
    pattern = geometry.generate_l2_scan_pattern(
        horizontal_phase_cycles=horizontal,
        vertical_phase_cycles=vertical,
    )
    azimuth = math.radians(target_azimuth_deg)
    elevation = math.radians(target_elevation_deg)
    target_direction = np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        np.float64,
    )
    maximum_dot = float(np.max(pattern.directions_sensor_fru @ target_direction))
    if maximum_dot <= 0.0:
        return False
    line_distance = float(target_range_m) * math.sqrt(max(0.0, 1.0 - maximum_dot**2))
    return line_distance <= 0.10 + 1e-12


def _select_error_witness(
    *,
    oracle_contact: bool,
    predicted_contact: bool,
    sensor_clearance: np.ndarray,
    support: np.ndarray,
    oracle_clearance: np.ndarray,
    oracle_contact_step: int,
    oracle_contact_link: int,
) -> dict[str, Any]:
    """Select the deterministic witness that actually causes a disagreement."""

    if predicted_contact and not oracle_contact:
        unsupported_rows = np.argwhere(~np.asarray(support, bool))
        if len(unsupported_rows):
            candidates = sorted(
                (
                    float(oracle_clearance[int(step), int(link)]),
                    int(step),
                    int(link),
                )
                for step, link in unsupported_rows
            )
            _, step, link = candidates[0]
            return {
                "step": step,
                "link": link,
                "link_resolved": True,
                "mechanism": "UNSUPPORTED_SWEEP_RISK",
            }
        step, link = np.unravel_index(
            int(np.argmin(sensor_clearance)), sensor_clearance.shape
        )
        return {
            "step": int(step),
            "link": int(link),
            "link_resolved": True,
            "mechanism": "FINITE_CLEARANCE_THRESHOLD",
        }
    if oracle_contact and not predicted_contact:
        if oracle_contact_step >= 0:
            step = int(oracle_contact_step)
        else:
            step = int(np.unravel_index(int(np.argmin(oracle_clearance)), oracle_clearance.shape)[0])
        if oracle_contact_link >= 0:
            link = int(oracle_contact_link)
            resolved = True
        else:
            link = int(np.argmin(oracle_clearance[step]))
            resolved = False
        return {
            "step": step,
            "link": link,
            "link_resolved": resolved,
            "mechanism": "FROZEN_ORACLE_CONTACT",
        }
    raise ValueError("error witness requested for an agreeing prediction")


def _classify_condition_error(
    *,
    condition_id: str,
    evidence: dict[str, Any],
    link_resolved: bool,
    reporting: Any,
) -> dict[str, Any]:
    """Apply a condition-specific deterministic attribution hierarchy."""

    if not link_resolved:
        return {
            "error_class": "UNRESOLVED",
            "reason": "the repaired oracle contact has no agreeing per-link attribution",
        }
    if condition_id == "REALISTIC_PLATFORM_SCAN":
        return reporting.classify_coverage_error(evidence)

    prefix = {
        "CURRENT_SPARSE_RANGE_BASELINE": "condition",
        "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT": "platform",
        "DENSE_BODY_CENTRIC_SINGLE_ORIGIN": "body",
    }[condition_id]
    near = bool(evidence[f"{prefix}_inside_near_blind_region"])
    self_occluded = bool(evidence[f"{prefix}_robot_self_occluded"])
    if near:
        return {"error_class": "NEAR_BLIND_REGION", "reason": f"{prefix} witness is inside the blind region"}
    if self_occluded:
        return {"error_class": "ROBOT_SELF_OCCLUSION", "reason": f"robot geometry occludes the {prefix} witness"}
    if prefix != "body":
        if not bool(evidence[f"{prefix}_nominal_vertical_fov"]):
            return {"error_class": "VERTICAL_COVERAGE_LIMITATION", "reason": f"{prefix} witness is outside vertical FOV"}
        if not bool(evidence[f"{prefix}_nominal_horizontal_fov"]):
            return {"error_class": "HORIZONTAL_COVERAGE_LIMITATION", "reason": f"{prefix} witness is outside horizontal FOV"}
    if condition_id == "CURRENT_SPARSE_RANGE_BASELINE":
        if bool(evidence["condition_dense_direct_support"]) and not bool(
            evidence["condition_temporal_ray_support"]
        ):
            return {"error_class": "SCAN_PATTERN_SPARSITY", "reason": "baseline target is directly visible but no finite baseline ray supports it"}
        if bool(evidence["dense_body_support"]):
            return {"error_class": "PLATFORM_MOUNT_LIMITATION", "reason": "the baseline origin fails where the body-centric origin supports the witness"}
    elif condition_id == "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT":
        if bool(evidence["dense_body_support"]):
            return {"error_class": "PLATFORM_MOUNT_LIMITATION", "reason": "dense platform mount fails where the body-centric origin supports the witness"}
    if not bool(evidence["dense_body_support"]):
        return {"error_class": "SINGLE_ORIGIN_LIMITATION", "reason": "the dense body-centric single origin cannot support the witness"}
    return {"error_class": "UNRESOLVED", "reason": "coverage evidence does not isolate the disagreement"}


def persist_coverage_errors(
    context: Any,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_metrics_v1 as metrics
    from lewm.safety import body_centric_range_coverage_reporting_v1 as reporting
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    path = OUTPUT_ROOT / "evidence" / "coverage_errors.jsonl"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    counts = {label: 0 for label in CONTRACT.COVERAGE_ERROR_CLASSES}
    condition_mode_counts: dict[str, dict[str, dict[str, int]]] = {
        condition_id: {
            mode_id: {label: 0 for label in CONTRACT.COVERAGE_ERROR_CLASSES}
            for mode_id in MODES
        }
        for condition_id in CONDITIONS
    }
    region_counts: dict[str, int] = defaultdict(int)
    rows = 0
    with temporary.open("wb") as stream:
        for state_id in context.state_ids_for_role("heldout"):
            identities = _identity_rows_with_oracle(context, state_id)
            link_names = context.protected_link_names(state_id)
            with np.load(_state_evidence_path(state_id), allow_pickle=False) as archive:
                all_clearance = np.asarray(archive["clearance_m"], np.float64)
                all_support = np.asarray(archive["support"], bool)
                oracle_clearance = np.asarray(archive["oracle_clearance_m"], np.float64)
                oracle_contact_step = np.asarray(archive["oracle_contact_step"], np.int16)
                oracle_contact_link = np.asarray(archive["oracle_contact_link"], np.int16)
                for mode_index, mode_id in enumerate(MODES):
                    a_support = all_support[:, 0, mode_index]
                    b_support = all_support[:, 1, mode_index]
                    c_support = all_support[:, 2, mode_index]
                    d_support = all_support[:, 3, mode_index]
                    b_event_support = np.asarray(
                        archive["event_time_support"][:, 1, mode_index], bool
                    )
                    a_self = np.asarray(archive["self_occluded"][:, 0, mode_index], bool)
                    b_self = np.asarray(archive["self_occluded"][:, 1, mode_index], bool)
                    c_self = np.asarray(archive["self_occluded"][:, 2, mode_index], bool)
                    d_self = np.asarray(archive["self_occluded"][:, 3, mode_index], bool)
                    a_near = np.asarray(archive["near_blind"][:, 0, mode_index], bool)
                    c_near = np.asarray(archive["near_blind"][:, 2, mode_index], bool)
                    d_near = np.asarray(archive["near_blind"][:, 3, mode_index], bool)
                    a_horizontal = np.asarray(archive["horizontal_fov"][:, 0, mode_index], bool)
                    a_vertical = np.asarray(archive["vertical_fov"][:, 0, mode_index], bool)
                    a_direct = np.asarray(archive["direct_visibility"][:, 0, mode_index], bool)
                    c_horizontal = np.asarray(archive["horizontal_fov"][:, 2, mode_index], bool)
                    c_vertical = np.asarray(archive["vertical_fov"][:, 2, mode_index], bool)
                    c_azimuth = np.asarray(
                        archive["target_event_azimuth_deg"][:, 2, mode_index],
                        np.float64,
                    )
                    c_elevation = np.asarray(
                        archive["target_event_elevation_deg"][:, 2, mode_index],
                        np.float64,
                    )
                    c_range = np.asarray(
                        archive["target_event_range_m"][:, 2, mode_index],
                        np.float64,
                    )
                    for condition_index, condition_id in enumerate(CONDITIONS):
                        threshold = float(thresholds[condition_id][mode_id]["selected_threshold_m"])
                        clearance = all_clearance[:, condition_index, mode_index]
                        support = all_support[:, condition_index, mode_index]
                        global_clearance = np.min(clearance, axis=(1, 2))
                        unsupported = ~np.all(support, axis=(1, 2))
                        predicted = metrics.contact_predictions(
                            global_clearance, threshold, unsupported
                        )
                        for transition, identity in enumerate(identities):
                            oracle = bool(identity["oracle_contact"])
                            if bool(predicted[transition]) == oracle:
                                continue
                            witness = _select_error_witness(
                                oracle_contact=oracle,
                                predicted_contact=bool(predicted[transition]),
                                sensor_clearance=clearance[transition],
                                support=support[transition],
                                oracle_clearance=oracle_clearance[transition],
                                oracle_contact_step=int(oracle_contact_step[transition]),
                                oracle_contact_link=int(oracle_contact_link[transition]),
                            )
                            step = int(witness["step"])
                            link = int(witness["link"])
                            boundary_digest = _boundary_digest_for_identity(
                                context, state_id, identity
                            )
                            spatial = _realistic_spatial_support(
                                boundary_snapshot_digest=boundary_digest,
                                mode_id=mode_id,
                                target_azimuth_deg=float(c_azimuth[transition, step, link]),
                                target_elevation_deg=float(c_elevation[transition, step, link]),
                                target_range_m=float(c_range[transition, step, link]),
                                dense_direct_support=bool(c_support[transition, step, link]),
                            )
                            temporal = bool(
                                b_event_support[transition, step, link]
                            )
                            spatial = bool(spatial or temporal)
                            evidence = {
                                "point_accumulation_error": False,
                                "condition_inside_near_blind_region": bool(
                                    a_near[transition, step, link]
                                ),
                                "condition_robot_self_occluded": bool(
                                    a_self[transition, step, link]
                                ),
                                "condition_nominal_vertical_fov": bool(
                                    a_vertical[transition, step, link]
                                ),
                                "condition_nominal_horizontal_fov": bool(
                                    a_horizontal[transition, step, link]
                                ),
                                "condition_dense_direct_support": bool(
                                    a_direct[transition, step, link]
                                ),
                                "condition_temporal_ray_support": bool(
                                    archive["event_time_support"][
                                        transition, 0, mode_index, step, link
                                    ]
                                ),
                                "platform_inside_near_blind_region": bool(
                                    c_near[transition, step, link]
                                ),
                                "body_inside_near_blind_region": bool(
                                    d_near[transition, step, link]
                                ),
                                "platform_robot_self_occluded": bool(
                                    c_self[transition, step, link]
                                    or b_self[transition, step, link]
                                ),
                                "body_robot_self_occluded": bool(
                                    d_self[transition, step, link]
                                ),
                                "platform_nominal_vertical_fov": bool(
                                    c_vertical[transition, step, link]
                                ),
                                "platform_nominal_horizontal_fov": bool(
                                    c_horizontal[transition, step, link]
                                ),
                                "realistic_spatial_ray_support": spatial,
                                "realistic_temporal_ray_support": temporal,
                                "dense_platform_support": bool(
                                    c_support[transition, step, link]
                                ),
                                "dense_body_support": bool(
                                    d_support[transition, step, link]
                                ),
                            }
                            attribution = _classify_condition_error(
                                condition_id=condition_id,
                                evidence=evidence,
                                link_resolved=bool(witness["link_resolved"]),
                                reporting=reporting,
                            )
                            label = str(attribution["error_class"])
                            if bool(witness["link_resolved"]):
                                link_name = link_names[link]
                                region = str(corpus_adapter.BODY_REGION_BY_LINK[link_name])
                                link_region = f"{link_name}/{region}"
                            else:
                                link_name = "UNRESOLVED"
                                region = "UNRESOLVED"
                                link_region = "UNRESOLVED/UNRESOLVED"
                            responsible_mechanism = (
                                witness["mechanism"] != "UNSUPPORTED_SWEEP_RISK"
                            )
                            ray_field = (
                                "nearest_ray_index"
                                if responsible_mechanism
                                else "support_nearest_ray_index"
                            )
                            age_field = (
                                "point_age_s"
                                if responsible_mechanism
                                else "support_point_age_s"
                            )
                            range_field = (
                                "nearest_point_range_m"
                                if responsible_mechanism
                                else "support_nearest_range_m"
                            )
                            acquisition_field = (
                                "responsible_acquisition_index"
                                if responsible_mechanism
                                else "support_acquisition_index"
                            )
                            nearest_ray = int(
                                archive[ray_field][
                                    transition, condition_index, mode_index, step, link
                                ]
                            )
                            point_age = float(
                                archive[age_field][
                                    transition, condition_index, mode_index, step, link
                                ]
                            )
                            point_range = float(
                                archive[range_field][
                                    transition, condition_index, mode_index, step, link
                                ]
                            )
                            acquisition_index = int(
                                archive[acquisition_field][
                                    transition, condition_index, mode_index, step, link
                                ]
                            )
                            sensor_clearance = float(
                                clearance[transition, step, link]
                            )
                            self_geom_index = int(
                                archive["self_occluder_geom_index"][
                                    transition, condition_index, mode_index, step, link
                                ]
                            )
                            self_link_index = int(
                                archive["self_occluder_link_index"][
                                    transition, condition_index, mode_index, step, link
                                ]
                            )
                            row = {
                                "transition_uid": identity["identity"],
                                "state_id": state_id,
                                "transition_level": identity["level"],
                                "current_action_index": int(identity["current_action_index"]),
                                "action_index": int(identity["action_index"]),
                                "candidate_action_id": applied_action_id(identity),
                                "condition_id": condition_id,
                                "evidence_mode": mode_id,
                                "robot_link_or_body_region": link_region,
                                "contact_or_minimum_clearance_physics_step": step,
                                "nominal_fov_inclusion": bool(
                                    archive["nominal_fov"][
                                        transition, condition_index, mode_index, step, link
                                    ]
                                ),
                                "self_occlusion": bool(
                                    archive["self_occluded"][
                                        transition, condition_index, mode_index, step, link
                                    ]
                                ),
                                "self_occluder_geom_index": (
                                    None if self_geom_index < 0 else self_geom_index
                                ),
                                "self_occluder_link": (
                                    None
                                    if self_link_index < 0
                                    or self_link_index >= len(link_names)
                                    else link_names[self_link_index]
                                ),
                                "scan_point_availability": bool(
                                    support[transition, step, link]
                                ),
                                "nearest_ray_or_point": None if nearest_ray < 0 else nearest_ray,
                                "point_age_s": _finite_float_or_none(point_age),
                                "nearest_point_range_m": _finite_float_or_none(point_range),
                                "acquisition_index": (
                                    None if acquisition_index < 0 else acquisition_index
                                ),
                                "exact_clearance_m": float(
                                    oracle_clearance[transition, step, link]
                                ),
                                "sensor_derived_clearance_m": sensor_clearance,
                                "oracle_contact": oracle,
                                "predicted_contact": bool(predicted[transition]),
                                "attribution_mechanism": witness["mechanism"],
                                "oracle_contact_link_resolved": bool(
                                    witness["link_resolved"]
                                ),
                                "error_class": label,
                                "error_reason": attribution["reason"],
                                "attribution_evidence": evidence,
                            }
                            stream.write(canonical_bytes(row))
                            counts[label] += 1
                            condition_mode_counts[condition_id][mode_id][label] += 1
                            region_counts[region] += 1
                            rows += 1
    os.replace(temporary, path)
    return {
        "schema": "body_centric_range_coverage_error_persistence_v1",
        "materialization_index_sha256": sha256_file(
            OUTPUT_ROOT / "materialization_index.json"
        ),
        "threshold_freeze_sha256": sha256_file(
            OUTPUT_ROOT / "calibration" / "thresholds_frozen.json"
        ),
        "path": str(path),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "counts": counts,
        "per_condition_mode_counts": condition_mode_counts,
        "per_region_counts": dict(sorted(region_counts.items())),
        "population": CONTRACT.build_contract()["coverage_attribution"]["error_population"],
    }


def benchmark_worker(threshold_m: float) -> dict[str, Any]:
    """Fresh-process complete held-out-set reduction benchmark."""

    from lewm.safety import body_centric_range_coverage_analysis_v1 as analysis
    from lewm.safety import body_centric_range_coverage_metrics_v1 as metrics
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    context = corpus_adapter.load_corpus_context(ROOT)
    payloads: list[dict[str, Any]] = []
    condition_index = CONDITIONS.index("DENSE_BODY_CENTRIC_SINGLE_ORIGIN")
    mode_index = MODES.index("TRUE_FUTURE_OBSERVABILITY_CLOUD")
    for state_id in context.state_ids_for_role("heldout"):
        with np.load(_state_evidence_path(state_id), allow_pickle=False) as archive:
            payloads.append(
                {
                    "state_id": state_id,
                    "identity_rows": _identity_rows_with_oracle(context, state_id),
                    "clearance": np.asarray(
                        archive["clearance_m"][:, condition_index, mode_index], np.float64
                    ),
                    "support": np.asarray(
                        archive["support"][:, condition_index, mode_index], bool
                    ),
                }
            )

    def reduce_complete_set() -> dict[str, Any]:
        predicted_states: list[dict[str, Any]] = []
        for payload in payloads:
            clearance = payload["clearance"]
            support = payload["support"]
            # This array reduction is deliberately timed: per-link minimum
            # and unsupported swept-witness fraction precede the global
            # decision for every transition in every held-out state.
            per_link_minimum = np.min(clearance, axis=1)
            unsupported_fraction = 1.0 - support.mean(axis=1)
            global_minimum = np.min(per_link_minimum, axis=1)
            unsupported = np.any(unsupported_fraction > 0.0, axis=1)
            records = []
            for index, identity in enumerate(payload["identity_rows"]):
                row = dict(identity)
                row["clearance_m"] = float(global_minimum[index])
                row["unsupported"] = bool(unsupported[index])
                records.append(row)
            states = analysis.build_two_ply_states(records)
            predicted_states.extend(
                analysis.inject_contact_predictions(states, records, threshold_m)
            )
        return metrics.reduce_two_ply_states(predicted_states)

    warmups = 30
    iterations = 1000
    values: list[float] = []
    for iteration in range(warmups + iterations):
        started = time.perf_counter_ns()
        reduce_complete_set()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if iteration >= warmups:
            values.append(elapsed_ms)
    array = np.asarray(values, np.float64)
    current_counts = [
        sum(row["level"] == "current" for row in payload["identity_rows"])
        for payload in payloads
    ]
    successor_counts = [
        sum(row["level"] == "successor" for row in payload["identity_rows"])
        for payload in payloads
    ]
    result = {
        "schema": "body_centric_range_coverage_complete_heldout_set_benchmark_v1",
        "condition_id": "DENSE_BODY_CENTRIC_SINGLE_ORIGIN",
        "evidence_mode": "TRUE_FUTURE_OBSERVABILITY_CLOUD",
        "threshold_m": float(threshold_m),
        "scope": (
            "complete 24-state held-out set: deterministic step/link per-link reduction, "
            "every current action, every next-action set, safe counts, threshold decisions, H3 selection"
        ),
        "includes_ray_generation": False,
        "includes_future_trajectory_acquisition": False,
        "heldout_states": len(payloads),
        "complete_set_each_iteration": True,
        "current_action_rows": int(sum(current_counts)),
        "successor_action_rows": int(sum(successor_counts)),
        "current_action_rows_per_state_min": min(current_counts),
        "current_action_rows_per_state_max": max(current_counts),
        "successor_action_rows_per_state_min": min(successor_counts),
        "successor_action_rows_per_state_max": max(successor_counts),
        "warmups": warmups,
        "iterations": iterations,
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
        "misses_50ms": int((array > 50.0).sum()),
        "misses_80ms": int((array > 80.0).sum()),
        "misses_100ms": int((array > 100.0).sum()),
        "miss_fraction_50ms": float((array > 50.0).mean()),
        "miss_fraction_80ms": float((array > 80.0).mean()),
        "miss_fraction_100ms": float((array > 100.0).mean()),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_vram_bytes": 0,
        "device": "CPU",
    }
    return result


def launch_benchmark(threshold_m: float) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "benchmark-worker",
            "--threshold-m",
            repr(float(threshold_m)),
        ],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def _planning_future_comparison(condition_metrics: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition_id in CONDITIONS:
        causal = condition_metrics[condition_id]["PLANNING_TIME_CAUSAL_CLOUD"]
        future = condition_metrics[condition_id]["TRUE_FUTURE_OBSERVABILITY_CLOUD"]
        output[condition_id] = {
            "planning_time": {
                "current_auc": causal["current_contact"]["auc"],
                "successor_auc": causal["successor_contact"]["auc"],
                "combined_recall": causal["combined_contact"]["recall"],
                "combined_fnr": causal["combined_contact"]["fnr"],
                "safe_count_zero_nonzero_accuracy": causal["safe_action_count"][
                    "zero_vs_nonzero_accuracy"
                ],
                "viable_states_retained": causal["viability"][
                    "states_retaining_admitted_action"
                ],
            },
            "true_future": {
                "current_auc": future["current_contact"]["auc"],
                "successor_auc": future["successor_contact"]["auc"],
                "combined_recall": future["combined_contact"]["recall"],
                "combined_fnr": future["combined_contact"]["fnr"],
                "safe_count_zero_nonzero_accuracy": future["safe_action_count"][
                    "zero_vs_nonzero_accuracy"
                ],
                "viable_states_retained": future["viability"][
                    "states_retaining_admitted_action"
                ],
            },
        }
    return output


def sparse_baseline_regression(context: Any) -> dict[str, Any]:
    """Compare A true-future clearance with the frozen legacy sparse reducer."""

    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    condition_index = CONDITIONS.index("CURRENT_SPARSE_RANGE_BASELINE")
    mode_index = MODES.index("TRUE_FUTURE_OBSERVABILITY_CLOUD")
    for state_id in context.state_ids:
        role = context.role_for_state(state_id)
        with np.load(_state_evidence_path(state_id), allow_pickle=False) as archive:
            legacy = np.asarray(
                archive["legacy_sparse_global_clearance_m"], np.float64
            )
            current = np.min(
                np.asarray(
                    archive["clearance_m"][:, condition_index, mode_index],
                    np.float64,
                ),
                axis=(1, 2),
            )
        grouped[role].append((legacy, current))

    def summarize(rows: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
        legacy = np.concatenate([row[0] for row in rows])
        current = np.concatenate([row[1] for row in rows])
        both = np.isfinite(legacy) & np.isfinite(current)
        difference = np.abs(legacy[both] - current[both])
        return {
            "rows": len(legacy),
            "both_finite_rows": int(both.sum()),
            "legacy_nonfinite_rows": int((~np.isfinite(legacy)).sum()),
            "current_nonfinite_rows": int((~np.isfinite(current)).sum()),
            "exact_within_1e_6_fraction": (
                None if not len(difference) else float((difference <= 1e-6).mean())
            ),
            "mean_absolute_clearance_difference_m": (
                None if not len(difference) else float(difference.mean())
            ),
            "p95_absolute_clearance_difference_m": (
                None if not len(difference) else float(np.percentile(difference, 95))
            ),
            "maximum_absolute_clearance_difference_m": (
                None if not len(difference) else float(difference.max())
            ),
        }

    all_rows = [item for values in grouped.values() for item in values]
    return {
        "schema": "body_centric_range_sparse_baseline_regression_v1",
        "scope": "non-primary comparison against predecessor lidar_clearance global minimum",
        "known_semantic_differences": [
            "V1 applies exact articulated robot self-occlusion",
            "V1 uses the corrected exact point-to-collision-primitive reducer",
        ],
        "all_roles": summarize(all_rows),
        "per_role": {role: summarize(rows) for role, rows in sorted(grouped.items())},
    }


def _secondary_region_input(
    condition_metrics: dict[str, Any], coverage_errors: dict[str, Any]
) -> dict[str, Any]:
    source = condition_metrics["DENSE_BODY_CENTRIC_SINGLE_ORIGIN"][
        "TRUE_FUTURE_OBSERVABILITY_CLOUD"
    ]["per_region"]
    output: dict[str, Any] = {}
    mapping = {
        "front_limb": "FRONT_LIMB",
        "rear_limb": "REAR_LIMB",
        "calf": "CALF",
        "trunk": "TRUNK",
    }
    for source_name, target_name in mapping.items():
        if source_name not in source:
            continue
        combined = source[source_name]["combined_contact"]
        output[target_name] = {
            "contact_positives": int(combined["positives"]),
            "combined_contact_recall": combined["recall"],
            "coverage_error_count": int(
                coverage_errors["per_region_counts"].get(source_name, 0)
            ),
        }
    return output


def _next_decision(primary: str, gates: dict[str, Any]) -> dict[str, Any]:
    passing = [condition for condition in CONDITIONS if gates[condition]["pass"]]
    if passing:
        return {
            "decision": "SPECIFY_PER_LINK_CLEARANCE_PREDICTOR_V1_ONLY_AFTER_PHYSICAL_SENSOR_CONTRACT",
            "passing_sensor_conditions": passing,
            "training_authorized_now": False,
            "predictor_specification": {
                "outputs": [
                    "per-link minimum clearance through the committed tick",
                    "time to first clearance violation",
                    "obstacle/body sector",
                    "observation support",
                    "lower confidence bound or uncertainty interval",
                ],
                "inputs": [
                    "planning-time range observation",
                    "articulated embodied state",
                    "one-tick candidate action",
                    "control history",
                ],
                "deterministic_outputs": ["contact", "successor viability"],
                "forbidden_scalar_only_outputs": [
                    "binary contact",
                    "binary nonviability",
                    "utility score",
                ],
            },
            "qualification": (
                "a passing upper-bound condition does not authorize training until the physical "
                "sensor/mount and a fresh claim-bearing panel are frozen"
            ),
        }
    return {
        "decision": (
            "SPECIFY_MULTIPLE_SENSOR_ORIGINS_OR_NARROW_PROTECTED_CONTACT_SCOPE"
            if primary == "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO"
            else "RESOLVE_RANGE_SENSOR_CONTRACT"
        ),
        "passing_sensor_conditions": [],
        "training_authorized_now": False,
        "per_link_predictor_authorized": False,
    }


def render_markdown_report(result: dict[str, Any]) -> str:
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    contract = CONTRACT.build_contract()
    environment = result["environment_receipt"]
    material = result["materialisation_counts"]
    storage = result["storage"]
    benchmark = result["compute_benchmark"]
    lines = [
        "# Body-centric range coverage qualification V1 result",
        "",
        f"Primary classification: `{result['primary_classification']}`.",
        "",
        "This was a deterministic, CPU-only, no-training development qualification. "
        "True-future clouds use actual transition geometry as an observability upper bound; "
        "they do not establish pre-action prediction or deployment safety.",
        "",
        "## Bindings and execution",
        "",
        f"- Source-freeze commit: `{result['source_freeze_commit']}`",
        f"- Source lineage: `{result['source_lineage']}`",
        f"- Contract SHA-256: `{result['contract_sha256']}` (content `{result['contract_content_digest']}`)",
        f"- Source-closure SHA-256: `{result['source_closure_sha256']}`",
        f"- Environment: Python {environment['python']}, NumPy {environment['packages']['numpy']}, SciPy {environment['packages']['scipy']}; CPU-only; Torch/Genesis/JEPA not imported",
        f"- Materialisation: {material['states']} states, {material['transitions']} transitions, {material['representatives']} unique applied-action representatives, {material['physics_steps']} physics frames, {material['raw_audit_artifacts']} raw-audit artifacts",
        f"- Materialisation runtime/storage: {fmt(material['runtime_s'])} s / {material['storage_bytes']} bytes",
        f"- Output storage before result: {storage['allocated_bytes_before_result']} bytes of {storage['final_ceiling_bytes']} byte ceiling on `{storage['filesystem']['source']}` ({storage['filesystem']['fstype']})",
        f"- Fixture gate: {'PASS' if result['fixture_results'].get('pass') else 'FAIL'}; content digest `{result['fixture_results'].get('content_digest')}`",
        "",
        "## Frozen sensor and mounts",
        "",
        "The realistic condition is the explicit development assumption "
        "`ASSUMED_GO2_HEAD_LIDAR_L2` using `APPROXIMATED_REALISTIC_PLATFORM_SCAN`; "
        "it is not a final deployment selection. The approximation uses 6,400 ideal rays per 100 ms, "
        "64 kpoints/s effective rate, 5.55 Hz azimuth, 216 Hz triangular elevation over −6° to +90°, "
        "0.05–30 m range, and planning-boundary-hash phases.",
        "",
        f"Platform mount (`base`→`radar`): translation `{contract['mounts']['platform']['translation_m']}` m, RPY `{contract['mounts']['platform']['rotation_rpy_rad']}` rad. Body-centric mount: translation `{contract['mounts']['body_centric']['translation_m']}` m, level RPY `{contract['mounts']['body_centric']['rotation_rpy_rad']}` rad.",
        "",
        "Specifications: [Unitree Go2](https://www.unitree.com/go2/), "
        "[Unitree L2](https://www.unitree.com/L2/), "
        "[L2 manual](https://oss-global-cdn.unitree.com/static/Unitree%204D%20LiDAR%20L2%20User%20Manual.pdf).",
        "",
        "## Contact metrics and frozen calibration thresholds",
        "",
        "| Condition | Mode | Threshold m | Cur AUC/AP | Succ AUC/AP | Combined recall/FNR | Negative retention | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for condition_id in CONDITIONS:
        for mode_id in MODES:
            metrics_row = result["condition_metrics"][condition_id][mode_id]
            gate = result["gate_results"]["true_future"].get(condition_id)
            gate_text = (
                "n/a (causal diagnostic)"
                if mode_id == "PLANNING_TIME_CAUSAL_CLOUD"
                else ("PASS" if gate["pass"] else "FAIL")
            )
            values = [
                condition_id,
                mode_id,
                fmt(metrics_row["threshold_m"]),
                f"{fmt(metrics_row['current_contact']['auc'])}/{fmt(metrics_row['current_contact']['average_precision'])}",
                f"{fmt(metrics_row['successor_contact']['auc'])}/{fmt(metrics_row['successor_contact']['average_precision'])}",
                f"{fmt(metrics_row['combined_contact']['recall'])}/{fmt(metrics_row['combined_contact']['fnr'])}",
                fmt(metrics_row["combined_contact"]["negative_retention"]),
                gate_text,
            ]
            lines.append("| " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Safe-count and two-ply viability",
        "",
        "| Condition | Mode | Count MAE/Spearman/exact/zero-nonzero | False zero/nonzero | Viable retained | Nonviable abstain | Unsafe selected (now/next) | Progress fraction/regret | Top-1/Top-3 | Margins ≥1/2/3 |",
        "|---|---|---|---|---:|---:|---|---|---|---|",
    ])
    for condition_id in CONDITIONS:
        for mode_id in MODES:
            row = result["condition_metrics"][condition_id][mode_id]
            safe = row["safe_action_count"]
            viability = row["viability"]
            lines.append(
                "| " + " | ".join([
                    condition_id,
                    mode_id,
                    "/".join(fmt(safe[key]) for key in ("mae", "spearman", "exact_count_accuracy", "zero_vs_nonzero_accuracy")),
                    f"{fmt(safe['false_zero_rate'])}/{fmt(safe['false_nonzero_rate'])}",
                    f"{viability['states_retaining_admitted_action']}/{viability['oracle_viable_states']}",
                    f"{viability['correct_abstentions']}/{viability['oracle_nonviable_states']}",
                    f"{viability['selected_immediate_contacts']}/{viability['selected_oracle_nonviable_successors']}",
                    f"{fmt(viability['oracle_progress_fraction'])}/{fmt(viability['normalized_regret'])}",
                    f"{fmt(viability['best_admissible_top1'])}/{fmt(viability['best_admissible_top3'])}",
                    f"{viability['selected_safe_margin_ge_1']}/{viability['selected_safe_margin_ge_2']}/{viability['selected_safe_margin_ge_3']}",
                ]) + " |"
            )

    lines.extend([
        "",
        "## True-future per-family coverage",
        "",
        "| Condition | Family | Contact AUC/recall | Support | Mean unsupported | Viable retained | Nonviable abstain |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for condition_id in CONDITIONS:
        families = result["condition_metrics"][condition_id]["TRUE_FUTURE_OBSERVABILITY_CLOUD"]["per_family"]
        for family, row in sorted(families.items()):
            lines.append(
                "| " + " | ".join([
                    condition_id,
                    family,
                    f"{fmt(row['combined_contact']['auc'])}/{fmt(row['combined_contact']['recall'])}",
                    fmt(row["coverage"]["support_fraction"]),
                    fmt(row["coverage"]["mean_unsupported_swept_volume_fraction"]),
                    f"{row['states_retaining_admitted_action']}/{row['oracle_viable_states']}",
                    f"{row['correct_abstentions']}/{row['oracle_nonviable_states']}",
                ]) + " |"
            )

    lines.extend([
        "",
        "## True-future per-link coverage",
        "",
        "| Condition | Link | Region | Contact AUC/recall | Support | Direct visibility | Mean unsupported | Unresolved oracle rows |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    for condition_id in CONDITIONS:
        metrics_row = result["condition_metrics"][condition_id]["TRUE_FUTURE_OBSERVABILITY_CLOUD"]
        for link, row in sorted(metrics_row["per_link"].items()):
            coverage = row["coverage"]
            region = str(corpus_adapter.BODY_REGION_BY_LINK.get(link, "UNRESOLVED"))
            lines.append(
                "| " + " | ".join([
                    condition_id,
                    link,
                    region,
                    f"{fmt(row['combined_contact']['auc'])}/{fmt(row['combined_contact']['recall'])}",
                    fmt(coverage["support_fraction"]),
                    fmt(coverage["direct_visibility_fraction"]),
                    fmt(coverage["mean_unsupported_swept_volume_fraction"]),
                    str(row.get("unresolved_oracle_contact_rows_excluded", 0)),
                ]) + " |"
            )

    lines.extend([
        "",
        "## Coverage errors, regression, and compute",
        "",
        f"Coverage-error rows: {result['coverage_error_receipt']['rows']}; aggregate classes: `{json.dumps(result['coverage_error_counts'], sort_keys=True)}`.",
        "",
        f"Per-condition/mode attribution: `{json.dumps(result['coverage_error_receipt']['per_condition_mode_counts'], sort_keys=True)}`.",
        "",
        f"Legacy sparse-regression summary: `{json.dumps(result['sparse_baseline_regression']['all_roles'], sort_keys=True)}`.",
        "",
        f"Complete-heldout reduction benchmark: P50/P90/P95/P99/max = {fmt(benchmark['p50_ms'])}/{fmt(benchmark['p90_ms'])}/{fmt(benchmark['p95_ms'])}/{fmt(benchmark['p99_ms'])}/{fmt(benchmark['max_ms'])} ms; misses at 50/80/100 ms = {benchmark['misses_50ms']}/{benchmark['misses_80ms']}/{benchmark['misses_100ms']}; peak RSS {benchmark['peak_rss_bytes']} bytes; peak VRAM {benchmark['peak_vram_bytes']} bytes.",
        "",
        "Scan materialisation totals: `" + json.dumps(material["scan_materialization_counts"], sort_keys=True) + "`.",
        "",
        "## Classification and custody",
        "",
        f"Secondary classifications: {', '.join(f'`{value}`' for value in result['secondary_classifications']) or 'none'}.",
        "",
        f"Next decision: `{result['next_decision']['decision']}`. No training is authorized.",
        "",
        "No model was trained; no fresh panel, G2 evaluation, JEPA predictor, checkpoint, memory, navigation, routing, or beacon system was opened or executed. Superseded occupancy and recurrent-memory experiments were not restarted.",
        "",
        "Machine-readable authority: the external `result.json`, calibration receipts, transition ledger, per-link ledger, coverage-error ledger, materialisation index, raw-audit manifest, and persistence receipt.",
        "",
    ])
    return "\n".join(lines)


def evaluate() -> dict[str, Any]:
    preexecution = validate_preexecution_receipt()
    materialization = load_materialization_index()
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter
    from lewm.safety import body_centric_range_coverage_reporting_v1 as reporting

    context = corpus_adapter.load_corpus_context(ROOT)
    started = time.time()
    thresholds, condition_metrics = analyze_all_conditions(context)
    true_future_metrics = {
        condition: condition_metrics[condition]["TRUE_FUTURE_OBSERVABILITY_CLOUD"]
        for condition in CONDITIONS
    }
    planning_metrics = {
        condition: condition_metrics[condition]["PLANNING_TIME_CAUSAL_CLOUD"]
        for condition in CONDITIONS
    }
    future_gates = reporting.evaluate_condition_gates(true_future_metrics)
    planning_gates = {
        condition: {
            **reporting.evaluate_true_future_gate(planning_metrics[condition]),
            "mode": "PLANNING_TIME_CAUSAL_CLOUD_DIAGNOSTIC",
        }
        for condition in CONDITIONS
    }
    primary = reporting.select_primary_classification(
        future_gates, realistic_platform_sensor_resolved=True
    )
    row_evidence = persist_row_level_evidence(context, thresholds, condition_metrics)
    coverage_errors = persist_coverage_errors(context, thresholds)
    secondary = reporting.derive_secondary_classifications(
        true_future_gate_results=future_gates,
        planning_time_gate_results=planning_gates,
        sensor_binding_class="ASSUMED_GO2_HEAD_LIDAR_L2",
        per_region_metrics=_secondary_region_input(condition_metrics, coverage_errors),
        coverage_error_counts=coverage_errors["counts"],
    )
    strongest_threshold = thresholds["DENSE_BODY_CENTRIC_SINGLE_ORIGIN"][
        "TRUE_FUTURE_OBSERVABILITY_CLOUD"
    ]["selected_threshold_m"]
    benchmark = launch_benchmark(float(strongest_threshold))
    baseline_regression = sparse_baseline_regression(context)
    fixture = json.loads(TRACKED_FIXTURE.read_text())
    contract = CONTRACT.load_and_validate_contract(TRACKED_CONTRACT)
    schema = CONTRACT.load_and_validate_output_schema(TRACKED_SCHEMA)
    storage_before_result = _directory_allocated_bytes(OUTPUT_ROOT)
    result = {
        "schema_version": "body_centric_range_coverage_qualification_v1.result.v1",
        "experiment_id": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_freeze_commit": preexecution["head"],
        "predecessor_result_commit": PREDECESSOR_HEAD,
        "source_lineage": SOURCE_LINEAGE,
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "contract_content_digest": contract["contract_sha256"],
        "output_schema_sha256": sha256_file(TRACKED_SCHEMA),
        "output_schema_content_digest": schema["output_schema_sha256"],
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "environment_receipt": preexecution["environment"],
        "storage": {
            "output_root": str(OUTPUT_ROOT),
            "filesystem": preexecution["output_filesystem"],
            "temporary_ceiling_bytes": 30_000_000_000,
            "final_ceiling_bytes": 20_000_000_000,
            "allocated_bytes_before_result": storage_before_result,
            "ceiling_pass": storage_before_result <= 20_000_000_000,
        },
        "materialisation_counts": {
            "states": materialization["states"],
            "transitions": materialization["transitions"],
            "representatives": materialization["representatives"],
            "physics_steps": materialization["physics_steps"],
            "protected_links": materialization["protected_links"],
            "raw_audit_artifacts": materialization["raw_audit_artifacts"],
            "runtime_s": materialization["runtime_s"],
            "storage_bytes": materialization["storage_bytes"],
            "scan_materialization_counts": materialization[
                "scan_materialization_counts"
            ],
        },
        "fixture_results": fixture,
        "calibration_thresholds": thresholds,
        "condition_metrics": condition_metrics,
        "sparse_baseline_regression": baseline_regression,
        "planning_time_true_future_comparison": _planning_future_comparison(
            condition_metrics
        ),
        "gate_results": {"true_future": future_gates, "planning_time_diagnostic": planning_gates},
        "coverage_error_counts": coverage_errors["counts"],
        "coverage_error_receipt": coverage_errors,
        "row_level_evidence_receipt": row_evidence,
        "compute_benchmark": benchmark,
        "primary_classification": primary,
        "secondary_classifications": secondary,
        "next_decision": _next_decision(primary, future_gates),
        "runtime_s": time.time() - started,
        "custody": {
            "training_steps": 0,
            "fresh_panel_rows": 0,
            "simulator_steps": 0,
            "jepa_predictor_opens": 0,
            "model_checkpoint_opens": 0,
            "untouched_g2_opens": 0,
            "memory_navigation_routing_beacon_executions": 0,
            "superseded_v3_restarted": False,
            "completed_v4_restarted_or_reinterpreted": False,
            "recurrent_memory_retrained": False,
        },
    }
    if not result["storage"]["ceiling_pass"]:
        raise RuntimeError("final storage ceiling exceeded before result persistence")
    result["result_content_sha256"] = content_digest(result)
    payload = (json.dumps(json_ready(result), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    report = render_markdown_report(result).encode()
    atomic_bytes(OUTPUT_ROOT / "result.json", payload)
    atomic_bytes(OUTPUT_ROOT / "report.md", report)
    atomic_bytes(RESULT_JSON, payload)
    atomic_bytes(RESULT_MD, report)
    persistence = {
        "schema": "body_centric_range_coverage_persistence_receipt_v1",
        "experiment": EXPERIMENT,
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "result": {
            "path": str(OUTPUT_ROOT / "result.json"),
            "sha256": sha256_file(OUTPUT_ROOT / "result.json"),
            "bytes": (OUTPUT_ROOT / "result.json").stat().st_size,
        },
        "report": {
            "path": str(OUTPUT_ROOT / "report.md"),
            "sha256": sha256_file(OUTPUT_ROOT / "report.md"),
            "bytes": (OUTPUT_ROOT / "report.md").stat().st_size,
        },
        "row_level_evidence": row_evidence,
        "coverage_errors": coverage_errors,
        "raw_audit_manifest": {
            "path": materialization["raw_audit_manifest_path"],
            "sha256": materialization["raw_audit_manifest_sha256"],
            "rows": materialization["raw_audit_artifacts"],
        },
        "preexecution_receipt": {
            "path": str(OUTPUT_ROOT / "preexecution_receipt.json"),
            "sha256": sha256_file(OUTPUT_ROOT / "preexecution_receipt.json"),
        },
        "environment_receipt": {
            "path": str(OUTPUT_ROOT / "receipts" / "environment_receipt.json"),
            "sha256": sha256_file(OUTPUT_ROOT / "receipts" / "environment_receipt.json"),
        },
        "materialization_index": {
            "path": str(OUTPUT_ROOT / "materialization_index.json"),
            "sha256": sha256_file(OUTPUT_ROOT / "materialization_index.json"),
        },
        "threshold_freeze_receipt": {
            "path": str(OUTPUT_ROOT / "calibration" / "thresholds_frozen.json"),
            "sha256": sha256_file(OUTPUT_ROOT / "calibration" / "thresholds_frozen.json"),
        },
        "allocated_bytes_final": _directory_allocated_bytes(OUTPUT_ROOT),
        "final_storage_ceiling_bytes": 20_000_000_000,
    }
    persistence["pass"] = persistence["allocated_bytes_final"] <= 20_000_000_000
    persistence["content_digest"] = content_digest(persistence)
    if not persistence["pass"]:
        raise RuntimeError("final storage ceiling exceeded")
    atomic_json(OUTPUT_ROOT / "persistence_receipt.json", persistence)
    return result


def check_result() -> dict[str, Any]:
    if git("status", "--porcelain=v1"):
        raise RuntimeError("result custody check requires a clean worktree")
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("result custody check is on the wrong branch")
    if running_scientific_processes():
        raise RuntimeError("a scientific process remains active during custody check")
    if not (OUTPUT_ROOT / "persistence_receipt.json").is_file():
        raise RuntimeError("persistence receipt missing")
    persistence = json.loads((OUTPUT_ROOT / "persistence_receipt.json").read_text())
    persistence_core = dict(persistence)
    persistence_digest = persistence_core.pop("content_digest", None)
    if persistence_digest != content_digest(persistence_core) or persistence.get("pass") is not True:
        raise RuntimeError("persistence receipt self-digest/pass failure")
    if persistence.get("contract_sha256") != sha256_file(TRACKED_CONTRACT):
        raise RuntimeError("persistence contract binding drift")
    if persistence.get("source_closure_sha256") != sha256_file(TRACKED_CLOSURE):
        raise RuntimeError("persistence source-closure binding drift")
    for section in (
        "preexecution_receipt",
        "environment_receipt",
        "materialization_index",
        "threshold_freeze_receipt",
    ):
        bound = persistence[section]
        if sha256_file(Path(bound["path"])) != bound["sha256"]:
            raise RuntimeError(f"{section} binding drift")

    contract = CONTRACT.load_and_validate_contract(TRACKED_CONTRACT)
    schema = CONTRACT.load_and_validate_output_schema(TRACKED_SCHEMA)
    closure = json.loads(TRACKED_CLOSURE.read_text())
    closure_core = dict(closure)
    closure_digest = closure_core.pop("content_digest", None)
    if closure_digest != content_digest(closure_core):
        raise RuntimeError("source closure self-digest failure")
    for source in closure["files"]:
        source_path = ROOT / source["path"]
        if source_path.stat().st_size != source["bytes"] or sha256_file(source_path) != source["sha256"]:
            raise RuntimeError(f"source closure drift: {source_path}")

    preexecution = json.loads((OUTPUT_ROOT / "preexecution_receipt.json").read_text())
    preexecution_core = dict(preexecution)
    preexecution_digest = preexecution_core.pop("content_digest", None)
    if preexecution_digest != content_digest(preexecution_core) or preexecution.get("pass") is not True:
        raise RuntimeError("preexecution receipt self-digest/pass failure")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", preexecution["head"], git("rev-parse", "HEAD")],
        cwd=ROOT,
    ) != 0:
        raise RuntimeError("source-freeze commit is not an ancestor of the result commit")
    environment_path = OUTPUT_ROOT / "receipts" / "environment_receipt.json"
    environment = json.loads(environment_path.read_text())
    environment_core = dict(environment)
    environment_digest = environment_core.pop("content_digest", None)
    if environment_digest != content_digest(environment_core) or environment != preexecution["environment"]:
        raise RuntimeError("environment receipt drift")

    result_path = OUTPUT_ROOT / "result.json"
    result = json.loads(result_path.read_text())
    result_core = dict(result)
    result_digest = result_core.pop("result_content_sha256", None)
    if result_digest != content_digest(result_core):
        raise RuntimeError("result self-digest failure")
    if RESULT_JSON.read_bytes() != result_path.read_bytes():
        raise RuntimeError("tracked and external result JSON differ")
    if RESULT_MD.read_bytes() != (OUTPUT_ROOT / "report.md").read_bytes():
        raise RuntimeError("tracked and external reports differ")
    for section in ("result", "report"):
        row = persistence[section]
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"{section} artifact drift")
    for section in ("transition_evidence", "per_link_evidence"):
        row = persistence["row_level_evidence"][section]
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"{section} artifact drift")
    errors = persistence["coverage_errors"]
    error_path = Path(errors["path"])
    if error_path.stat().st_size != errors["bytes"] or sha256_file(error_path) != errors["sha256"]:
        raise RuntimeError("coverage-error artifact drift")

    materialization_path = OUTPUT_ROOT / "materialization_index.json"
    materialization = json.loads(materialization_path.read_text())
    materialization_core = dict(materialization)
    materialization_digest = materialization_core.pop("content_digest", None)
    if materialization_digest != content_digest(materialization_core) or materialization.get("status") != "PASS":
        raise RuntimeError("materialization index self-digest/pass failure")
    for record in materialization["records"]:
        state_id = record["state_id"]
        state_receipt_path = OUTPUT_ROOT / "states" / f"{state_id}.json"
        state_receipt = json.loads(state_receipt_path.read_text())
        state_core = dict(state_receipt)
        state_digest = state_core.pop("content_digest", None)
        if state_digest != content_digest(state_core) or state_receipt.get("status") != "PASS":
            raise RuntimeError(f"state receipt self-digest/pass failure: {state_id}")
        state_shard = Path(state_receipt["shard_path"])
        if sha256_file(state_shard) != state_receipt["shard_sha256"]:
            raise RuntimeError(f"state evidence drift: {state_id}")

    threshold_freeze_path = OUTPUT_ROOT / "calibration" / "thresholds_frozen.json"
    threshold_freeze = json.loads(threshold_freeze_path.read_text())
    threshold_core = dict(threshold_freeze)
    threshold_digest = threshold_core.pop("content_digest", None)
    if threshold_digest != content_digest(threshold_core) or threshold_freeze.get("pass") is not True:
        raise RuntimeError("threshold freeze self-digest/pass failure")
    for condition_id in CONDITIONS:
        for mode_id in MODES:
            frontier = threshold_freeze["thresholds"][condition_id][mode_id]
            frontier_path = OUTPUT_ROOT / frontier["artifact_relative_path"]
            if sha256_file(frontier_path) != frontier["artifact_sha256"]:
                raise RuntimeError("calibration frontier artifact drift")
            if hashlib.sha256(gzip.decompress(frontier_path.read_bytes())).hexdigest() != frontier["uncompressed_content_sha256"]:
                raise RuntimeError("calibration frontier content drift")

    raw_manifest = persistence["raw_audit_manifest"]
    raw_manifest_path = Path(raw_manifest["path"])
    if sha256_file(raw_manifest_path) != raw_manifest["sha256"]:
        raise RuntimeError("raw-audit manifest drift")
    raw_rows = [json.loads(line) for line in raw_manifest_path.read_text().splitlines() if line]
    if len(raw_rows) != int(raw_manifest["rows"]):
        raise RuntimeError("raw-audit manifest cardinality drift")
    for row in raw_rows:
        artifact = OUTPUT_ROOT / row["artifact_relative_path"]
        if sha256_file(artifact) != row["artifact_sha256"]:
            raise RuntimeError(f"raw-audit artifact drift: {artifact}")

    fixture = json.loads(TRACKED_FIXTURE.read_text())
    from lewm.safety import body_centric_range_coverage_v1 as geometry
    from lewm.safety import body_centric_range_coverage_metrics_v1 as metrics
    regenerated_fixture = combined_fixture_receipt(geometry, metrics)
    if regenerated_fixture.get("pass") is not True:
        raise RuntimeError("fixture regeneration failed")
    if TRACKED_FIXTURE.read_bytes() != canonical_bytes(regenerated_fixture):
        raise RuntimeError("fixture regeneration drift")
    inputs = validate_frozen_inputs(full_shards=True)
    return {
        "schema": "body_centric_range_coverage_result_check_v1",
        "pass": True,
        "head": git("rev-parse", "HEAD"),
        "contract_content_digest": contract["contract_sha256"],
        "output_schema_content_digest": schema["output_schema_sha256"],
        "source_closure_content_digest": closure["content_digest"],
        "result_content_sha256": result["result_content_sha256"],
        "persistence_receipt_sha256": sha256_file(
            OUTPUT_ROOT / "persistence_receipt.json"
        ),
        "validated_states": len(materialization["records"]),
        "validated_raw_audit_artifacts": len(raw_rows),
        "validated_input_shards": inputs["geometry_shard_hashes_checked"],
        "scientific_processes_running": 0,
    }


def not_implemented(command: str) -> None:
    raise RuntimeError(f"{command} implementation is not yet installed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    pre = sub.add_parser("preflight")
    pre.add_argument("--quick", action="store_true", help="skip full geometry-shard SHA validation")
    sub.add_parser("materialize")
    sub.add_parser("evaluate")
    sub.add_parser("execute")
    sub.add_parser("check")
    benchmark_parser = sub.add_parser("benchmark-worker", help=argparse.SUPPRESS)
    benchmark_parser.add_argument("--threshold-m", required=True, type=float)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(write_freeze_receipts(), indent=2, sort_keys=True))
    elif args.command == "preflight":
        print(json.dumps(preflight(full_shards=not args.quick), indent=2, sort_keys=True))
    elif args.command == "materialize":
        print(json.dumps(json_ready(materialize()), indent=2, sort_keys=True))
    elif args.command == "evaluate":
        print(json.dumps(json_ready(evaluate()), indent=2, sort_keys=True))
    elif args.command == "execute":
        if not (OUTPUT_ROOT / "preexecution_receipt.json").is_file():
            preflight(full_shards=True)
        materialize()
        print(json.dumps(json_ready(evaluate()), indent=2, sort_keys=True))
    elif args.command == "check":
        print(json.dumps(check_result(), indent=2, sort_keys=True))
    elif args.command == "benchmark-worker":
        print(json.dumps(json_ready(benchmark_worker(args.threshold_m)), sort_keys=True))
    else:
        not_implemented(args.command)


if __name__ == "__main__":
    main()
