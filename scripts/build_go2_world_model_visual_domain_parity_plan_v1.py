#!/usr/bin/env python3
"""Build the one-shot 8-family textured-v03 implementation-parity plan.

The plan is RGB-only and contains no physics, policy, training, held-out, or
promotion authority.  It deterministically binds one ordinary historical
textured-v03 TRAIN scene per family and the first four recorded base poses in
each scene.  A separately authorized producer must render every pose twice
through the shared candidate RGB helper and preserve all 32 historical RGB
references read-only.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import build_go2_world_model_bounded_branch_scene_panel_v1 as scene_inventory  # noqa: E402
from scripts import render_replay_v03 as reference_renderer  # noqa: E402
from scripts.build_go2_world_model_counterfactual_calibration_plan_v1 import (  # noqa: E402
    RUNTIME_CONTRACT_SCHEMA,
    _validate_runtime_contract,
)


PLAN_SCHEMA = "lewm_go2_world_model_visual_domain_parity_generation_plan_v1"
PLAN_STATUS = "FROZEN_EXACT_8_FAMILY_32_POSE_RGB_PARITY_PLAN"
SOURCE_PANEL_SCHEMA = "lewm_go2_world_model_visual_domain_parity_panel_v2"
SOURCE_LINEAGE_SCHEMA = "lewm_go2_world_model_visual_domain_parity_source_lineage_v2"
SOURCE_DOMAIN = "historical_textured_v03_reference"
PURPOSE = "textured_v03_deterministic_implementation_parity_v1"
POSES_PER_SCENE = 4
EXPECTED_SCENES = len(pilot.FAMILIES)
EXPECTED_POSES = EXPECTED_SCENES * POSES_PER_SCENE
SOURCE_RGB_ROOT = (
    REPO_ROOT / ".generated/datagen_full/render_textured_v03"
).resolve()
ROLLOUT_ROOT = (REPO_ROOT / ".generated/datagen_full/rollout").resolve()
DEVELOPMENT_ROOT = (REPO_ROOT / ".generated/dev").resolve()
REFERENCE_RENDERER = (REPO_ROOT / "scripts/render_replay_v03.py").resolve()
REFERENCE_TEXTURE_SOURCE = (
    REPO_ROOT / "lewm_genesis/lewm_genesis/textures.py"
).resolve()
ATTEMPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")

COMPARISON_CONTRACT = {
    "kind": "deterministic_textured_v03_implementation_equivalence",
    "families": list(pilot.FAMILIES),
    "scenes_per_family": 1,
    "poses_per_scene": POSES_PER_SCENE,
    "source_frames": EXPECTED_POSES,
    "candidate_frames": EXPECTED_POSES,
    "candidate_duplicate_frames": EXPECTED_POSES,
    "source_pose_selection": "first_four_fixed_frames_in_bound_frames_jsonl",
    "candidate_pose_input": "bound_base_position_and_quaternion_not_stored_camera_pose",
    "candidate_capture": (
        "shared_exact_historical_pose_plus_rgb_only_render_helper"
    ),
    "required_camera_render_call": {"rgb": True, "depth": False},
    "candidate_duplicate_render_required": True,
    "pixel_exact_source_candidate_required": True,
    "pixel_exact_candidate_duplicate_required": True,
    "missing_or_extra_rows_allowed": False,
    "statistical_inference": False,
    "protected_material_allowed": False,
}
EXPECTED_COUNTS = {
    "families": len(pilot.FAMILIES),
    "scenes": EXPECTED_SCENES,
    "poses": EXPECTED_POSES,
    "source_rgb_frames": EXPECTED_POSES,
    "candidate_rgb_frames": EXPECTED_POSES,
    "duplicate_rgb_frames": EXPECTED_POSES,
    "rgb_render_calls": EXPECTED_POSES * 2,
    "auxiliary_depth_render_calls": 0,
    "physics_steps": 0,
}
PLAN_FIELDS = frozenset({
    "schema",
    "status",
    "attempt_id",
    "purpose",
    "citable_as_scientific_evidence",
    "authorizes_retry_or_resume",
    "allows_refill",
    "allows_overwrite",
    "output_root",
    "render_contract",
    "comparison_contract",
    "expected_counts",
    "runtime_bindings",
    "execution_contract",
    "texture_asset_bindings",
    "source_panel_binding",
    "scene_corpus_manifest_bindings",
    "scenes",
    "mesh_asset_bindings",
})
SOURCE_PANEL_FIELDS = frozenset({
    "schema",
    "domain",
    "rgb_root",
    "render_contract",
    "producer_source_binding",
    "renderer_source_binding",
    "texture_source_binding",
    "selected_texture_asset_bindings_by_scene",
    "mesh_asset_bindings_by_scene",
    "producer_lineage",
    "rows",
})
SOURCE_LINEAGE_FIELDS = frozenset({
    "schema",
    "scene_genesis_bindings_by_scene",
    "render_summary_bindings_by_scene",
    "render_plan_bindings_by_scene",
    "frames_jsonl_bindings_by_scene",
})
SOURCE_ROW_FIELDS = frozenset({
    "pair_id",
    "scene_id",
    "family",
    "pose_index",
    "camera_pose_world",
    "scene_manifest_binding",
    "producer_frame_identity",
    "rgb_binding",
    "raw_rgb_sha256",
})
SCENE_FIELDS = frozenset({
    "family",
    "scene_id",
    "scene_manifest_binding",
    "scene_genesis_binding",
    "render_summary_binding",
    "render_plan_binding",
    "frames_jsonl_binding",
    "selected_texture_asset_bindings",
    "mesh_asset_bindings",
    "poses",
})
POSE_FIELDS = frozenset({
    "pair_id",
    "pose_index",
    "source_frame_index",
    "source_env_index",
    "base_position_xyz_m",
    "base_quaternion_wxyz",
    "historical_camera_pose_world",
    "source_rgb_binding",
    "source_raw_pixel_sha256",
    "source_frame_record_sha256",
    "producer_frame_identity",
})


class VisualDomainParityPlanError(RuntimeError):
    """Raised before mutable or biased source evidence can become a plan."""


def _protected(path: Path) -> bool:
    return any(
        part.lower() == "sealed_test.json"
        or part.lower() == "sealed"
        or part.lower().startswith("sealed_")
        or part.lower() in {"heldout", "held_out", "held-out"}
        or part.lower().startswith("heldout_")
        or part.lower().startswith("held_out_")
        or part.lower().startswith("held-out-")
        or part.lower() == "protected"
        or part.lower().startswith("protected_")
        for part in Path(path).parts
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _nofollow_regular(path: Path, *, label: str) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    if _protected(selected) or not selected.is_absolute():
        raise VisualDomainParityPlanError(f"{label} names protected/non-absolute input")
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except OSError as exc:
            raise VisualDomainParityPlanError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(mode):
            raise VisualDomainParityPlanError(f"{label} traverses a symlink")
    if not selected.is_file() or selected.resolve(strict=True) != selected:
        raise VisualDomainParityPlanError(f"{label} is not a canonical regular file")
    return selected


def _binding(path: Path, *, label: str) -> dict[str, Any]:
    selected = _nofollow_regular(path, label=label)
    try:
        bound = pilot.file_binding(selected)
    except (OSError, pilot.PilotContractError) as exc:
        raise VisualDomainParityPlanError(str(exc)) from exc
    if Path(str(bound["path"])) != selected:
        raise VisualDomainParityPlanError(f"{label} binding path changed")
    return bound


def _require_exact_binding(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise VisualDomainParityPlanError(f"{label} binding fields changed")
    actual = _binding(Path(str(value.get("path"))), label=label)
    if actual != dict(value):
        raise VisualDomainParityPlanError(f"{label} binding changed")
    return actual


def _read_bound_json(
    binding: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _nofollow_regular(Path(str(binding.get("path"))), label=label)
    try:
        value, actual = pilot.read_bound_json(
            path,
            expected_sha256=str(binding.get("file_sha256")),
            expected_byte_count=int(binding.get("byte_count", -1)),
            label=label,
        )
    except (OSError, TypeError, ValueError, pilot.PilotContractError) as exc:
        raise VisualDomainParityPlanError(str(exc)) from exc
    if actual != dict(binding) or not isinstance(value, Mapping):
        raise VisualDomainParityPlanError(f"{label} binding/document changed")
    return dict(value), actual


def _raw_rgb_sha256(binding: Mapping[str, Any], *, label: str) -> str:
    path = _nofollow_regular(Path(str(binding["path"])), label=label)
    try:
        payload, actual = pilot.read_bound_bytes(
            path,
            expected_sha256=str(binding.get("file_sha256")),
            expected_byte_count=int(binding.get("byte_count", -1)),
            label=label,
        )
    except (OSError, TypeError, ValueError, pilot.PilotContractError) as exc:
        raise VisualDomainParityPlanError(f"cannot read {label}") from exc
    if actual != dict(binding):
        raise VisualDomainParityPlanError(f"{label} file binding changed")
    try:
        with Image.open(BytesIO(payload)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise VisualDomainParityPlanError(f"{label} is not decodable RGB") from exc
    if rgb.shape != (224, 224, 3):
        raise VisualDomainParityPlanError(f"{label} is not exact 224x224 RGB")
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _finite_vector(value: object, *, length: int, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise VisualDomainParityPlanError(f"{label} vector changed")
    return [float(item) for item in value]


def _historical_pose(value: object, *, label: str) -> dict[str, list[float]]:
    if not isinstance(value, Mapping) or set(value) != {"position", "lookat", "up"}:
        raise VisualDomainParityPlanError(f"{label} camera pose changed")
    return {
        name: _finite_vector(value[name], length=3, label=f"{label} {name}")
        for name in ("position", "lookat", "up")
    }


def _source_pose_from_frame(
    frame: Mapping[str, Any], *, scene_id: str, pose_index: int
) -> tuple[list[float], list[float], dict[str, list[float]], str]:
    """Derive every candidate pose input from one immutable historical row."""

    frame_index = frame.get("frame_index")
    env_index = frame.get("env_index")
    base_pose = frame.get("base_pose_world")
    base_position = base_pose.get("position") if isinstance(base_pose, Mapping) else None
    base_orientation = (
        base_pose.get("orientation") if isinstance(base_pose, Mapping) else None
    )
    quat_xyzw = _finite_vector(
        frame.get("base_quat_world_xyzw"),
        length=4,
        label=f"{scene_id} pose {pose_index} base quaternion",
    )
    if (
        type(frame_index) is not int
        or frame_index != pose_index
        or type(env_index) is not int
        or env_index != pose_index
        or not isinstance(base_position, Mapping)
        or set(base_position) != {"x", "y", "z"}
        or not isinstance(base_orientation, Mapping)
        or set(base_orientation) != {"w", "x", "y", "z"}
    ):
        raise VisualDomainParityPlanError(
            f"{scene_id} first-four frame identity changed"
        )
    position = _finite_vector(
        [base_position[name] for name in ("x", "y", "z")],
        length=3,
        label=f"{scene_id} pose {pose_index} base position",
    )
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    if quat_wxyz != [
        float(base_orientation[name]) for name in ("w", "x", "y", "z")
    ]:
        raise VisualDomainParityPlanError(
            f"{scene_id} base quaternion encodings disagree"
        )
    camera_pose = _historical_pose(
        frame.get("camera_pose_world"),
        label=f"{scene_id} pose {pose_index}",
    )
    return position, quat_wxyz, camera_pose, _canonical_sha256(frame)


def _read_first_bound_frames(
    binding: Mapping[str, Any], *, scene_id: str
) -> list[dict[str, Any]]:
    """Read four rows while hashing the complete immutable JSONL source."""

    path = _nofollow_regular(
        Path(str(binding["path"])), label=f"{scene_id} historical frames JSONL"
    )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VisualDomainParityPlanError(
            f"cannot open {scene_id} historical frames JSONL"
        ) from exc
    digest = hashlib.sha256()
    byte_count = 0
    buffer = b""
    rows: list[dict[str, Any]] = []
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if len(rows) < POSES_PER_SCENE:
                buffer += chunk
                while b"\n" in buffer and len(rows) < POSES_PER_SCENE:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise VisualDomainParityPlanError(
                            f"{scene_id} historical frame row is invalid JSON"
                        ) from exc
                    if not isinstance(value, Mapping):
                        raise VisualDomainParityPlanError(
                            f"{scene_id} historical frame row is not an object"
                        )
                    rows.append(dict(value))
        if len(rows) < POSES_PER_SCENE and buffer.strip():
            try:
                value = json.loads(buffer)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise VisualDomainParityPlanError(
                    f"{scene_id} historical final frame row is invalid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise VisualDomainParityPlanError(
                    f"{scene_id} historical final frame row is not an object"
                )
            rows.append(dict(value))
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or byte_count != int(binding["byte_count"])
        or digest.hexdigest() != binding["file_sha256"]
        or len(rows) != POSES_PER_SCENE
    ):
        raise VisualDomainParityPlanError(
            f"{scene_id} historical frames JSONL changed during read"
        )
    return rows


def _read_expected_mesh_binding(
    path: Path, *, expected_bytes: bytes, label: str
) -> dict[str, Any]:
    """Validate content and derive identity from one no-follow file read."""

    selected = _nofollow_regular(path, label=label)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(selected, flags)
    except OSError as exc:
        raise VisualDomainParityPlanError(f"cannot open {label}") from exc
    digest = hashlib.sha256()
    byte_count = 0
    payload_chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VisualDomainParityPlanError(f"{label} is not a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            payload_chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = selected.stat(follow_symlinks=False)
    except OSError as exc:
        raise VisualDomainParityPlanError(f"{label} changed after read") from exc
    identity = (before.st_dev, before.st_ino, before.st_size)
    if (
        identity != (after.st_dev, after.st_ino, after.st_size)
        or identity != (path_after.st_dev, path_after.st_ino, path_after.st_size)
        or not stat.S_ISREG(path_after.st_mode)
        or byte_count != before.st_size
        or b"".join(payload_chunks) != expected_bytes
    ):
        raise VisualDomainParityPlanError(f"{label} content or identity changed")
    return {
        "path": str(selected),
        "file_sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _mesh_bindings_for_manifest(
    manifest: Mapping[str, Any], *, scene_id: str
) -> list[dict[str, Any]]:
    expected_by_path: dict[str, str] = {}
    for field in ("walls", "obstacles", "landmarks"):
        objects = manifest.get(field)
        if not isinstance(objects, list):
            raise VisualDomainParityPlanError(
                f"{scene_id} manifest {field} inventory changed"
            )
        for value in objects:
            if not isinstance(value, Mapping):
                raise VisualDomainParityPlanError(
                    f"{scene_id} manifest object is malformed"
                )
            category = reference_renderer.category_for_kind(
                str(value.get("kind") or "")
            )
            if category is None:
                continue
            size = _finite_vector(
                value.get("size_xyz_m"),
                length=3,
                label=f"{scene_id} textured object size",
            )
            rounded = tuple(round(number, 3) for number in size)
            sx, sy, sz = rounded
            tiles_per_m = reference_renderer._textures._DEFAULT_TILES_PER_M  # noqa: SLF001
            mesh_name = (
                f"box_{sx:.3f}x{sy:.3f}x{sz:.3f}_"
                f"t{float(tiles_per_m):.2f}.obj"
            )
            mesh_path = (
                REPO_ROOT / ".generated/box_meshes" / mesh_name
            ).resolve()
            expected_text = reference_renderer._textures.box_obj_text(  # noqa: SLF001
                rounded,
                tiles_per_m=tiles_per_m,
            )
            expected_by_path[str(mesh_path)] = expected_text
    bindings = []
    mesh_root = (REPO_ROOT / ".generated/box_meshes").resolve()
    for path_text, expected_text in sorted(expected_by_path.items()):
        path = _nofollow_regular(Path(path_text), label=f"{scene_id} derived mesh")
        try:
            path.relative_to(mesh_root)
        except ValueError as exc:
            raise VisualDomainParityPlanError(
                f"{scene_id} derived mesh escaped the fixed cache"
            ) from exc
        bindings.append(_read_expected_mesh_binding(
            path,
            expected_bytes=expected_text.encode("utf-8"),
            label=f"{scene_id} derived mesh",
        ))
    if not bindings:
        raise VisualDomainParityPlanError(
            f"{scene_id} has no textured structural mesh closure"
        )
    return bindings


def _ranked_source_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        corpus_bindings, inventory = scene_inventory._load_inventory()  # noqa: SLF001
    except scene_inventory.BoundedBranchScenePanelError as exc:
        raise VisualDomainParityPlanError(str(exc)) from exc
    by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in pilot.FAMILIES
    }
    for row in inventory:
        family = str(row.get("family"))
        scene_id = str(row.get("scene_id"))
        if family not in by_family:
            continue
        render_dir = SOURCE_RGB_ROOT / scene_id
        summary = render_dir / "summary.json"
        if (
            render_dir.is_symlink()
            or not render_dir.is_dir()
            or summary.is_symlink()
            or not summary.is_file()
        ):
            continue
        candidate = dict(row)
        candidate["parity_selection_rank"] = _canonical_sha256(
            [PLAN_SCHEMA, "source-scene", family, scene_id, row["manifest_sha256"]]
        )
        by_family[family].append(candidate)
    selected = []
    for family in pilot.FAMILIES:
        candidates = sorted(
            by_family[family],
            key=lambda row: (row["parity_selection_rank"], row["scene_id"]),
        )
        if not candidates:
            raise VisualDomainParityPlanError(
                f"no complete historical textured-v03 source exists for {family}"
            )
        selected.append(candidates[0])
    return corpus_bindings, selected


def _validate_historical_summary_and_plan(
    *, family: str, scene_id: str
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    render_root = SOURCE_RGB_ROOT / scene_id
    summary_binding = _binding(
        render_root / "summary.json", label=f"{scene_id} render summary"
    )
    summary, _ = _read_bound_json(
        summary_binding, label=f"{scene_id} render summary"
    )
    plan_path = Path(str(summary.get("plan")))
    if (
        summary.get("schema") != "lewm_rendered_vision_v03"
        or summary.get("render_status") != "complete"
        or summary.get("scene_id") != scene_id
        or summary.get("family") != family
        or summary.get("split") != "train"
        or summary.get("resolution") != 224
        or summary.get("textures_enabled") is not True
        or summary.get("visuals") != "textured_v03"
        or type(summary.get("frame_count")) is not int
        or int(summary["frame_count"]) < POSES_PER_SCENE
        or not plan_path.is_absolute()
    ):
        raise VisualDomainParityPlanError(
            f"{scene_id} historical render summary changed"
        )
    plan_binding = _binding(plan_path, label=f"{scene_id} historical render plan")
    render_plan, _ = _read_bound_json(
        plan_binding, label=f"{scene_id} historical render plan"
    )
    frames_path = Path(str(render_plan.get("frames_jsonl")))
    if (
        render_plan.get("schema") != "lewm_render_replay_plan_v0"
        or render_plan.get("scene_id") != scene_id
        or render_plan.get("scene_family") != family
        or render_plan.get("split") != "train"
        or render_plan.get("raw_contract_audit_pass") is not True
        or render_plan.get("raw_data_quality_audit_pass") is not True
        or render_plan.get("frame_count") != summary["frame_count"]
        or not frames_path.is_absolute()
    ):
        raise VisualDomainParityPlanError(
            f"{scene_id} historical render plan changed"
        )
    try:
        plan_path.relative_to(ROLLOUT_ROOT / "train" / family)
        frames_path.relative_to(ROLLOUT_ROOT / "train" / family)
    except ValueError as exc:
        raise VisualDomainParityPlanError(
            f"{scene_id} historical plan/frames escaped ordinary TRAIN rollout"
        ) from exc
    camera = render_plan.get("camera")
    if (
        not isinstance(camera, Mapping)
        or camera.get("fov_axis") != "horizontal"
        or not math.isclose(float(camera.get("fov_deg", -1)), 78.323, abs_tol=1e-12)
        or camera.get("training_resolution") != [224, 224]
    ):
        raise VisualDomainParityPlanError(
            f"{scene_id} historical sensor metadata changed"
        )
    frames_binding = _binding(
        frames_path, label=f"{scene_id} historical frames JSONL"
    )
    return summary, summary_binding, render_plan, plan_binding, frames_binding


def _source_scene(
    inventory_row: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    family = str(inventory_row["family"])
    scene_id = str(inventory_row["scene_id"])
    try:
        scene_root = scene_inventory._validated_selected_scene_root(  # noqa: SLF001
            campaign_root=str(inventory_row["campaign_root"]),
            relative_dir=str(inventory_row["relative_dir"]),
        )
    except scene_inventory.BoundedBranchScenePanelError as exc:
        raise VisualDomainParityPlanError(str(exc)) from exc
    manifest_binding = _binding(
        scene_root / "manifest.json", label=f"{scene_id} ordinary scene manifest"
    )
    manifest, _ = _read_bound_json(
        manifest_binding, label=f"{scene_id} ordinary scene manifest"
    )
    genesis_binding = _binding(
        scene_root / "genesis_scene.json", label=f"{scene_id} Genesis scene"
    )
    if (
        manifest.get("scene_id") != scene_id
        or manifest.get("family") != family
        or manifest.get("split") != "train"
        or manifest.get("manifest_sha256") != inventory_row["manifest_sha256"]
    ):
        raise VisualDomainParityPlanError(
            f"{scene_id} ordinary manifest disagrees with its inventory"
        )
    (
        _summary,
        summary_binding,
        render_plan,
        render_plan_binding,
        frames_binding,
    ) = _validate_historical_summary_and_plan(family=family, scene_id=scene_id)
    if render_plan.get("manifest_sha256") != inventory_row["manifest_sha256"]:
        raise VisualDomainParityPlanError(
            f"{scene_id} historical render used a different scene manifest"
        )
    frame_rows = _read_first_bound_frames(frames_binding, scene_id=scene_id)
    try:
        selected_textures = scene_inventory._selected_texture_asset_bindings(  # noqa: SLF001
            manifest
        )
    except scene_inventory.BoundedBranchScenePanelError as exc:
        raise VisualDomainParityPlanError(str(exc)) from exc
    mesh_bindings = _mesh_bindings_for_manifest(manifest, scene_id=scene_id)
    poses = []
    panel_rows = []
    for pose_index, frame in enumerate(frame_rows):
        frame_index = frame.get("frame_index")
        env_index = frame.get("env_index")
        position, quat_wxyz, camera_pose, frame_record_sha256 = (
            _source_pose_from_frame(
                frame, scene_id=scene_id, pose_index=pose_index
            )
        )
        source_path = (
            SOURCE_RGB_ROOT
            / scene_id
            / "rgb"
            / f"frame_{frame_index:06d}_env_{env_index:02d}.png"
        )
        source_binding = _binding(
            source_path, label=f"{scene_id} pose {pose_index} historical RGB"
        )
        raw_pixel_sha256 = _raw_rgb_sha256(
            source_binding,
            label=f"{scene_id} pose {pose_index} historical RGB",
        )
        pair_id = f"{scene_id}/pose_{pose_index:02d}"
        producer_frame_identity = f"frame_{frame_index:06d}_env_{env_index:02d}"
        pose_row = {
            "pair_id": pair_id,
            "pose_index": pose_index,
            "source_frame_index": frame_index,
            "source_env_index": env_index,
            "base_position_xyz_m": position,
            "base_quaternion_wxyz": quat_wxyz,
            "historical_camera_pose_world": camera_pose,
            "source_rgb_binding": source_binding,
            "source_raw_pixel_sha256": raw_pixel_sha256,
            "source_frame_record_sha256": frame_record_sha256,
            "producer_frame_identity": producer_frame_identity,
        }
        poses.append(pose_row)
        panel_rows.append({
            "pair_id": pair_id,
            "scene_id": scene_id,
            "family": family,
            "pose_index": pose_index,
            "camera_pose_world": camera_pose,
            "scene_manifest_binding": manifest_binding,
            "producer_frame_identity": producer_frame_identity,
            "rgb_binding": source_binding,
            "raw_rgb_sha256": raw_pixel_sha256,
        })
    return ({
        "family": family,
        "scene_id": scene_id,
        "scene_manifest_binding": manifest_binding,
        "scene_genesis_binding": genesis_binding,
        "render_summary_binding": summary_binding,
        "render_plan_binding": render_plan_binding,
        "frames_jsonl_binding": frames_binding,
        "selected_texture_asset_bindings": selected_textures,
        "mesh_asset_bindings": mesh_bindings,
        "poses": poses,
    }, panel_rows)


def derive_source_evidence_v1() -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    corpus_bindings, selected_inventory = _ranked_source_rows()
    scenes = []
    panel_rows = []
    for expected_family, inventory_row in zip(
        pilot.FAMILIES, selected_inventory, strict=True
    ):
        if inventory_row["family"] != expected_family:
            raise VisualDomainParityPlanError(
                "historical source family ordering changed"
            )
        scene, rows = _source_scene(inventory_row)
        scenes.append(scene)
        panel_rows.extend(rows)
    panel_rows.sort(key=lambda row: row["pair_id"])
    scene_ids = [scene["scene_id"] for scene in scenes]
    if (
        len(scenes) != EXPECTED_SCENES
        or len(panel_rows) != EXPECTED_POSES
        or len(set(scene_ids)) != EXPECTED_SCENES
        or [scene["family"] for scene in scenes] != list(pilot.FAMILIES)
        or len({row["pair_id"] for row in panel_rows}) != EXPECTED_POSES
    ):
        raise VisualDomainParityPlanError(
            "deterministic historical parity panel count changed"
        )
    source_panel = {
        "schema": SOURCE_PANEL_SCHEMA,
        "domain": SOURCE_DOMAIN,
        "rgb_root": str(SOURCE_RGB_ROOT),
        "render_contract": dict(pilot.TEXTURED_V03_RENDER_CONTRACT),
        "producer_source_binding": _binding(
            REFERENCE_RENDERER, label="historical textured-v03 producer"
        ),
        "renderer_source_binding": _binding(
            REFERENCE_RENDERER, label="historical textured-v03 renderer"
        ),
        "texture_source_binding": _binding(
            REFERENCE_TEXTURE_SOURCE, label="historical texture source"
        ),
        "selected_texture_asset_bindings_by_scene": {
            scene["scene_id"]: scene["selected_texture_asset_bindings"]
            for scene in scenes
        },
        "mesh_asset_bindings_by_scene": {
            scene["scene_id"]: scene["mesh_asset_bindings"]
            for scene in scenes
        },
        "producer_lineage": {
            "schema": SOURCE_LINEAGE_SCHEMA,
            "scene_genesis_bindings_by_scene": {
                scene["scene_id"]: scene["scene_genesis_binding"]
                for scene in scenes
            },
            "render_summary_bindings_by_scene": {
                scene["scene_id"]: scene["render_summary_binding"]
                for scene in scenes
            },
            "render_plan_bindings_by_scene": {
                scene["scene_id"]: scene["render_plan_binding"]
                for scene in scenes
            },
            "frames_jsonl_bindings_by_scene": {
                scene["scene_id"]: scene["frames_jsonl_binding"]
                for scene in scenes
            },
        },
        "rows": panel_rows,
    }
    return source_panel, scenes, corpus_bindings


def _texture_asset_closure() -> list[dict[str, Any]]:
    return [
        _binding(REPO_ROOT / relative, label=f"texture asset {relative}")
        for relative in pilot.TEXTURED_V03_TEXTURE_RELATIVE_PATHS
    ]


def _validate_runtime(
    runtime_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        runtime, execution = _validate_runtime_contract(runtime_contract)
    except Exception as exc:
        raise VisualDomainParityPlanError(str(exc)) from exc
    expected_runtime = {
        "platform_manifest",
        "primitive_registry",
        "policy_checkpoint",
        "policy_config",
        "go2_urdf",
        "python_executable_target",
        "python_environment_config",
        "eglinfo_executable",
        "vulkaninfo_executable",
    }
    if set(runtime) != expected_runtime:
        raise VisualDomainParityPlanError("parity runtime binding closure changed")
    invocation = Path(str(execution.get("python_invocation_path")))
    invocation_target = Path(str(runtime["python_executable_target"]["path"]))
    try:
        resolved_invocation = invocation.resolve(strict=True)
    except OSError as exc:
        raise VisualDomainParityPlanError(
            "parity Python invocation is unavailable"
        ) from exc
    if (
        execution.get("backend") != "vulkan"
        or execution.get("policy_device") != "cpu"
        or execution.get("environment") != pilot.EXECUTION_ENVIRONMENT
        or execution.get("graphics_preflight") != pilot.GRAPHICS_PREFLIGHT_EXPECTATION
        or not invocation.is_absolute()
        or _protected(invocation)
        or resolved_invocation != invocation_target
        or invocation
        != (REPO_ROOT / ".generated/venvs/genesis_render_vulkan/bin/python")
    ):
        raise VisualDomainParityPlanError("parity execution contract changed")
    return runtime, execution


def _validate_output_root(path: Path, *, require_fresh: bool) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    if (
        not selected.is_absolute()
        or _protected(selected)
        or not DEVELOPMENT_ROOT.is_dir()
        or DEVELOPMENT_ROOT.is_symlink()
        or DEVELOPMENT_ROOT.resolve(strict=True) != DEVELOPMENT_ROOT
        or not selected.is_relative_to(DEVELOPMENT_ROOT)
        or selected == DEVELOPMENT_ROOT
    ):
        raise VisualDomainParityPlanError(
            "parity output root must be one canonical path below .generated/dev"
        )
    cursor = DEVELOPMENT_ROOT
    for part in selected.relative_to(DEVELOPMENT_ROOT).parts[:-1]:
        cursor = cursor / part
        if cursor.exists() and (cursor.is_symlink() or not cursor.is_dir()):
            raise VisualDomainParityPlanError(
                "parity output root traverses mutable/non-directory state"
            )
    if selected.resolve(strict=False) != selected:
        raise VisualDomainParityPlanError(
            "parity output root resolves outside its canonical spelling"
        )
    if (
        not selected.parent.is_dir()
        or selected.parent.is_symlink()
        or selected.parent.resolve(strict=True) != selected.parent
    ):
        raise VisualDomainParityPlanError(
            "parity output root parent must already be canonical"
        )
    if require_fresh:
        if selected.exists() or selected.is_symlink():
            raise VisualDomainParityPlanError(
                "parity output root is not fresh"
            )
    elif (
        not selected.is_dir()
        or selected.is_symlink()
        or selected.resolve(strict=True) != selected
    ):
        raise VisualDomainParityPlanError(
            "parity output root is not the canonical reserved directory"
        )
    return selected


def build_plan_v1(
    *,
    attempt_id: str,
    output_root: Path,
    source_panel_binding: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
    scene_corpus_manifest_bindings: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise VisualDomainParityPlanError("parity attempt_id is invalid")
    selected_root = _validate_output_root(output_root, require_fresh=True)
    source_panel_bound = _require_exact_binding(
        source_panel_binding, label="visual-domain parity source panel"
    )
    source_panel, actual_source_panel = _read_bound_json(
        source_panel_bound, label="visual-domain parity source panel"
    )
    if actual_source_panel != source_panel_bound:
        raise VisualDomainParityPlanError("parity source panel binding changed")
    normalized_scenes = [dict(scene) for scene in scenes]
    if (
        source_panel.get("schema") != SOURCE_PANEL_SCHEMA
        or source_panel.get("domain") != SOURCE_DOMAIN
        or len(normalized_scenes) != EXPECTED_SCENES
        or [scene.get("family") for scene in normalized_scenes]
        != list(pilot.FAMILIES)
        or any(
            not isinstance(scene.get("poses"), list)
            or len(scene["poses"]) != POSES_PER_SCENE
            for scene in normalized_scenes
        )
    ):
        raise VisualDomainParityPlanError("parity source evidence changed")
    runtime, execution = _validate_runtime(runtime_contract)
    mesh_by_path = {
        str(binding["path"]): dict(binding)
        for scene in normalized_scenes
        for binding in scene["mesh_asset_bindings"]
    }
    mesh_bindings = [mesh_by_path[path] for path in sorted(mesh_by_path)]
    corpus_bindings = [dict(binding) for binding in scene_corpus_manifest_bindings]
    if not corpus_bindings or any(
        _require_exact_binding(binding, label="ordinary scene corpus manifest")
        != binding
        for binding in corpus_bindings
    ):
        raise VisualDomainParityPlanError("ordinary corpus binding closure changed")
    return {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "attempt_id": attempt_id,
        "purpose": PURPOSE,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "output_root": str(selected_root),
        "render_contract": dict(pilot.TEXTURED_V03_RENDER_CONTRACT),
        "comparison_contract": dict(COMPARISON_CONTRACT),
        "expected_counts": dict(EXPECTED_COUNTS),
        "runtime_bindings": runtime,
        "execution_contract": execution,
        "texture_asset_bindings": _texture_asset_closure(),
        "source_panel_binding": source_panel_bound,
        "scene_corpus_manifest_bindings": corpus_bindings,
        "scenes": normalized_scenes,
        "mesh_asset_bindings": mesh_bindings,
    }


def validate_plan_v1(
    value: object, *, require_fresh_output: bool = True
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PLAN_FIELDS:
        raise VisualDomainParityPlanError("parity generation plan fields changed")
    if (
        value.get("schema") != PLAN_SCHEMA
        or value.get("status") != PLAN_STATUS
        or ATTEMPT_RE.fullmatch(str(value.get("attempt_id"))) is None
        or value.get("purpose") != PURPOSE
        or value.get("citable_as_scientific_evidence") is not False
        or value.get("authorizes_retry_or_resume") is not False
        or value.get("allows_refill") is not False
        or value.get("allows_overwrite") is not False
        or value.get("render_contract") != pilot.TEXTURED_V03_RENDER_CONTRACT
        or value.get("comparison_contract") != COMPARISON_CONTRACT
        or value.get("expected_counts") != EXPECTED_COUNTS
    ):
        raise VisualDomainParityPlanError("parity generation plan identity changed")
    _validate_output_root(
        Path(str(value.get("output_root"))), require_fresh=require_fresh_output
    )
    runtime, execution = _validate_runtime({
        "schema": RUNTIME_CONTRACT_SCHEMA,
        "runtime_bindings": value["runtime_bindings"],
        "execution_contract": value["execution_contract"],
    })
    texture_bindings = value["texture_asset_bindings"]
    expected_texture_bindings = _texture_asset_closure()
    if texture_bindings != expected_texture_bindings:
        raise VisualDomainParityPlanError("parity texture inventory changed")
    source_panel_binding = _require_exact_binding(
        value["source_panel_binding"], label="parity source panel"
    )
    source_panel, actual_source_panel = _read_bound_json(
        source_panel_binding, label="parity source panel"
    )
    if actual_source_panel != source_panel_binding:
        raise VisualDomainParityPlanError("parity source panel changed")
    source_rows = source_panel.get("rows")
    if (
        set(source_panel) != SOURCE_PANEL_FIELDS
        or source_panel.get("schema") != SOURCE_PANEL_SCHEMA
        or source_panel.get("domain") != SOURCE_DOMAIN
        or source_panel.get("rgb_root") != str(SOURCE_RGB_ROOT)
        or source_panel.get("render_contract")
        != pilot.TEXTURED_V03_RENDER_CONTRACT
        or not isinstance(source_rows, list)
        or len(source_rows) != EXPECTED_POSES
        or any(
            not isinstance(row, Mapping) or set(row) != SOURCE_ROW_FIELDS
            for row in source_rows
        )
        or [row.get("pair_id") for row in source_rows]
        != sorted(str(row.get("pair_id")) for row in source_rows)
    ):
        raise VisualDomainParityPlanError("parity source panel identity changed")
    if (
        source_panel.get("producer_source_binding")
        != _binding(REFERENCE_RENDERER, label="parity historical producer")
        or source_panel.get("renderer_source_binding")
        != _binding(REFERENCE_RENDERER, label="parity historical renderer")
        or source_panel.get("texture_source_binding")
        != _binding(REFERENCE_TEXTURE_SOURCE, label="parity texture source")
    ):
        raise VisualDomainParityPlanError("parity source implementation changed")
    source_by_pair = {
        str(row.get("pair_id")): row
        for row in source_rows
        if isinstance(row, Mapping)
    }
    if len(source_by_pair) != EXPECTED_POSES:
        raise VisualDomainParityPlanError("parity source pair inventory changed")
    scenes = value.get("scenes")
    if (
        not isinstance(scenes, list)
        or len(scenes) != EXPECTED_SCENES
        or any(
            not isinstance(scene, Mapping) or set(scene) != SCENE_FIELDS
            for scene in scenes
        )
    ):
        raise VisualDomainParityPlanError("parity scene inventory changed")
    observed_meshes: dict[str, dict[str, Any]] = {}
    observed_pairs = []
    selected_texture_map = source_panel.get(
        "selected_texture_asset_bindings_by_scene"
    )
    mesh_map = source_panel.get("mesh_asset_bindings_by_scene")
    if not isinstance(selected_texture_map, Mapping):
        raise VisualDomainParityPlanError("parity source texture map changed")
    lineage = source_panel.get("producer_lineage")
    if not isinstance(lineage, Mapping):
        raise VisualDomainParityPlanError("parity source lineage changed")
    summary_map = lineage.get("render_summary_bindings_by_scene")
    plan_map = lineage.get("render_plan_bindings_by_scene")
    frames_map = lineage.get("frames_jsonl_bindings_by_scene")
    genesis_map = lineage.get("scene_genesis_bindings_by_scene")
    if (
        set(lineage) != SOURCE_LINEAGE_FIELDS
        or lineage.get("schema") != SOURCE_LINEAGE_SCHEMA
        or not isinstance(mesh_map, Mapping)
        or not all(
            isinstance(item, Mapping)
            for item in (summary_map, plan_map, frames_map, genesis_map)
        )
    ):
        raise VisualDomainParityPlanError("parity source lineage maps changed")
    scene_ids = {str(scene["scene_id"]) for scene in scenes}
    if (
        set(selected_texture_map) != scene_ids
        or set(mesh_map) != scene_ids
        or any(set(item) != scene_ids for item in (summary_map, plan_map, frames_map, genesis_map))
    ):
        raise VisualDomainParityPlanError("parity source lineage scene closure changed")
    for family, scene in zip(pilot.FAMILIES, scenes, strict=True):
        if scene["family"] != family or not isinstance(scene["scene_id"], str):
            raise VisualDomainParityPlanError("parity scene family order changed")
        scene_id = scene["scene_id"]
        for name in (
            "scene_manifest_binding",
            "scene_genesis_binding",
            "render_summary_binding",
            "render_plan_binding",
            "frames_jsonl_binding",
        ):
            if _require_exact_binding(
                scene[name], label=f"parity {scene_id} {name}"
            ) != scene[name]:
                raise VisualDomainParityPlanError(
                    f"parity {scene_id} input binding changed"
                )
        manifest, manifest_actual = _read_bound_json(
            scene["scene_manifest_binding"],
            label=f"parity {scene_id} scene manifest",
        )
        genesis, genesis_actual = _read_bound_json(
            scene["scene_genesis_binding"],
            label=f"parity {scene_id} Genesis scene",
        )
        _summary, summary_actual = _read_bound_json(
            scene["render_summary_binding"],
            label=f"parity {scene_id} render summary",
        )
        _render_plan, render_plan_actual = _read_bound_json(
            scene["render_plan_binding"],
            label=f"parity {scene_id} render plan",
        )
        if (
            manifest_actual != scene["scene_manifest_binding"]
            or genesis_actual != scene["scene_genesis_binding"]
            or summary_actual != scene["render_summary_binding"]
            or render_plan_actual != scene["render_plan_binding"]
            or manifest.get("scene_id") != scene_id
            or manifest.get("family") != family
            or manifest.get("split") != "train"
            or genesis.get("scene_id") != scene_id
            or summary_map.get(scene_id) != scene["render_summary_binding"]
            or plan_map.get(scene_id) != scene["render_plan_binding"]
            or frames_map.get(scene_id) != scene["frames_jsonl_binding"]
            or genesis_map.get(scene_id) != scene["scene_genesis_binding"]
            or selected_texture_map.get(scene_id)
            != scene["selected_texture_asset_bindings"]
            or mesh_map.get(scene_id) != scene["mesh_asset_bindings"]
        ):
            raise VisualDomainParityPlanError(
                f"parity {scene_id} source-panel lineage changed"
            )
        for binding in scene["selected_texture_asset_bindings"].values():
            if binding not in texture_bindings:
                raise VisualDomainParityPlanError(
                    f"parity {scene_id} selected texture escaped closure"
                )
        recomputed_meshes = _mesh_bindings_for_manifest(manifest, scene_id=scene_id)
        if recomputed_meshes != scene["mesh_asset_bindings"]:
            raise VisualDomainParityPlanError(
                f"parity {scene_id} derived mesh closure changed"
            )
        for binding in recomputed_meshes:
            actual = _require_exact_binding(
                binding, label=f"parity {scene_id} derived mesh"
            )
            observed_meshes[str(actual["path"])] = actual
        poses = scene["poses"]
        frame_rows = _read_first_bound_frames(
            scene["frames_jsonl_binding"], scene_id=scene_id
        )
        if (
            not isinstance(poses, list)
            or len(poses) != POSES_PER_SCENE
            or any(
                not isinstance(pose, Mapping) or set(pose) != POSE_FIELDS
                for pose in poses
            )
        ):
            raise VisualDomainParityPlanError(
                f"parity {scene_id} pose inventory changed"
            )
        for pose_index, (pose, frame) in enumerate(
            zip(poses, frame_rows, strict=True)
        ):
            pair_id = str(pose["pair_id"])
            source_row = source_by_pair.get(pair_id)
            base_position, base_quaternion, camera_pose, frame_record_sha256 = (
                _source_pose_from_frame(
                    frame, scene_id=scene_id, pose_index=pose_index
                )
            )
            if (
                pair_id != f"{scene_id}/pose_{pose_index:02d}"
                or pose["pose_index"] != pose_index
                or pose["source_frame_index"] != frame.get("frame_index")
                or pose["source_env_index"] != frame.get("env_index")
                or pose["base_position_xyz_m"] != base_position
                or pose["base_quaternion_wxyz"] != base_quaternion
                or pose["historical_camera_pose_world"] != camera_pose
                or pose["source_frame_record_sha256"] != frame_record_sha256
                or not isinstance(source_row, Mapping)
                or source_row.get("scene_id") != scene_id
                or source_row.get("family") != family
                or source_row.get("pose_index") != pose_index
                or source_row.get("camera_pose_world")
                != pose["historical_camera_pose_world"]
                or source_row.get("scene_manifest_binding")
                != scene["scene_manifest_binding"]
                or source_row.get("producer_frame_identity")
                != pose["producer_frame_identity"]
                or source_row.get("rgb_binding") != pose["source_rgb_binding"]
                or source_row.get("raw_rgb_sha256")
                != pose["source_raw_pixel_sha256"]
                or _raw_rgb_sha256(
                    pose["source_rgb_binding"],
                    label=f"parity {pair_id} source RGB",
                )
                != pose["source_raw_pixel_sha256"]
            ):
                raise VisualDomainParityPlanError(
                    f"parity {pair_id} pose/source lineage changed"
                )
            observed_pairs.append(pair_id)
    expected_meshes = [
        observed_meshes[path] for path in sorted(observed_meshes)
    ]
    if value["mesh_asset_bindings"] != expected_meshes:
        raise VisualDomainParityPlanError("parity global mesh closure changed")
    corpus_bindings = value.get("scene_corpus_manifest_bindings")
    if not isinstance(corpus_bindings, list) or not corpus_bindings:
        raise VisualDomainParityPlanError("parity corpus closure changed")
    expected_corpus_bindings, expected_inventory = _ranked_source_rows()
    if corpus_bindings != expected_corpus_bindings:
        raise VisualDomainParityPlanError("parity corpus binding changed")
    if [
        (scene["family"], scene["scene_id"])
        for scene in scenes
    ] != [
        (row["family"], row["scene_id"])
        for row in expected_inventory
    ]:
        raise VisualDomainParityPlanError("parity deterministic scene selection changed")
    if (
        len(observed_pairs) != EXPECTED_POSES
        or len(set(observed_pairs)) != EXPECTED_POSES
        or set(observed_pairs) != set(source_by_pair)
    ):
        raise VisualDomainParityPlanError("parity pair closure changed")
    normalized = dict(value)
    normalized["runtime_bindings"] = runtime
    normalized["execution_contract"] = execution
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--runtime-contract", required=True, type=Path)
    parser.add_argument("--expected-runtime-contract-sha256", required=True)
    parser.add_argument(
        "--expected-runtime-contract-byte-count", required=True, type=int
    )
    parser.add_argument("--source-panel-output", required=True, type=Path)
    parser.add_argument("--plan-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_contract, _runtime_binding = pilot.read_bound_json(
        args.runtime_contract,
        expected_sha256=args.expected_runtime_contract_sha256,
        expected_byte_count=args.expected_runtime_contract_byte_count,
        label="visual-domain parity runtime contract",
    )
    source_panel, scenes, corpus_bindings = derive_source_evidence_v1()
    source_panel_binding = pilot.write_json_exclusive(
        args.source_panel_output, source_panel
    )
    plan = build_plan_v1(
        attempt_id=args.attempt_id,
        output_root=args.output_root,
        source_panel_binding=source_panel_binding,
        scenes=scenes,
        scene_corpus_manifest_bindings=corpus_bindings,
        runtime_contract=runtime_contract,
    )
    normalized = validate_plan_v1(plan)
    plan_binding = pilot.write_json_exclusive(args.plan_output, normalized)
    print(json.dumps({
        "plan": plan_binding,
        "source_panel": source_panel_binding,
        "expected_counts": normalized["expected_counts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARISON_CONTRACT",
    "EXPECTED_COUNTS",
    "PLAN_SCHEMA",
    "PLAN_STATUS",
    "PLAN_FIELDS",
    "POSE_FIELDS",
    "POSES_PER_SCENE",
    "PURPOSE",
    "SOURCE_DOMAIN",
    "SOURCE_LINEAGE_SCHEMA",
    "SOURCE_LINEAGE_FIELDS",
    "SOURCE_PANEL_FIELDS",
    "SOURCE_PANEL_SCHEMA",
    "SOURCE_ROW_FIELDS",
    "SCENE_FIELDS",
    "VisualDomainParityPlanError",
    "build_plan_v1",
    "derive_source_evidence_v1",
    "validate_plan_v1",
]
