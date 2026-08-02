#!/usr/bin/env python3
"""Recompute deterministic historical textured-v03 render equivalence.

The fixed panel is eight ordinary TRAIN scenes (one per family) and four
pre-bound poses per scene.  Every historical reference frame is compared with
an independently produced candidate frame and an independent duplicate of that
candidate render.  All 32 reference/candidate and candidate/duplicate pairs
must be byte-decoded pixel-exact.

This tool emits evidence only.  It never emits an independent review or an
execution authority.  Both panels must carry transitive producer lineage;
caller-authored frame paths or pose declarations alone are rejected.
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
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as collector  # noqa: E402
from scripts import build_go2_world_model_visual_domain_parity_plan_v1 as parity_plan  # noqa: E402
from scripts import build_go2_world_model_visual_domain_parity_authority_v1 as parity_authority  # noqa: E402
from scripts import render_replay_v03 as reference_renderer  # noqa: E402


PANEL_SCHEMA = parity_plan.SOURCE_PANEL_SCHEMA
RESULT_SCHEMA = "lewm_go2_world_model_bounded_branch_visual_domain_parity_result_v2"
PASS_STATUS = "PASS_EXACT_TEXTURED_V03_IMPLEMENTATION_EQUIVALENCE"
FAIL_STATUS = "FAIL_EXACT_TEXTURED_V03_IMPLEMENTATION_EQUIVALENCE"
SOURCE_LINEAGE_SCHEMA = parity_plan.SOURCE_LINEAGE_SCHEMA
CANDIDATE_LINEAGE_SCHEMA = (
    "lewm_go2_world_model_visual_domain_parity_candidate_lineage_v1"
)
CANDIDATE_GENERATION_RECEIPT_SCHEMA = (
    "lewm_go2_world_model_visual_domain_parity_generation_receipt_v1"
)
CANDIDATE_GENERATION_STATUS = "COMPLETE_EXACT_8_SCENE_32_POSE_DOUBLE_RENDER"
RENDER_RECEIPT_SCHEMA = (
    "lewm_go2_world_model_visual_domain_parity_rgb_render_receipt_v1"
)
RENDER_RECEIPT_STATUS = "RENDER_COMPLETE"
SOURCE_DOMAIN = parity_plan.SOURCE_DOMAIN
CANDIDATE_DOMAIN = "independent_candidate_double_render"
SOURCE_RGB_ROOT = REPO_ROOT / ".generated/datagen_full/render_textured_v03"
CANDIDATE_ROOT = REPO_ROOT / ".generated/dev"
REFERENCE_RENDERER = REPO_ROOT / "scripts/render_replay_v03.py"
REFERENCE_TEXTURE_SOURCE = REPO_ROOT / "lewm_genesis/lewm_genesis/textures.py"
CANDIDATE_COLLECTOR = (
    REPO_ROOT / "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"
)
CANDIDATE_RENDERER = REFERENCE_RENDERER
CANDIDATE_CAMERA_POSE_HELPER = (
    REPO_ROOT / "lewm_genesis/lewm_genesis/render_replay.py"
)
TEXTURE_CATEGORIES = ("floor", "wall", "obstacle")
FAMILIES = tuple(pilot.FAMILIES)
RESOLUTION = (224, 224)
SCENE_COUNT = len(FAMILIES)
POSES_PER_SCENE = 4
FRAME_COUNT = SCENE_COUNT * POSES_PER_SCENE
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SCENE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
_SOURCE_FRAME = re.compile(r"^frame_([0-9]{6,})_env_([0-9]{2,})$")

COMPARISON_CONTRACT = {
    "source_domain": "exact_historical_textured_v03_training_rgb_implementation",
    "source_rgb_root": ".generated/datagen_full/render_textured_v03",
    "source_renderer_path": "scripts/render_replay_v03.py",
    "candidate_domain": "independent_bounded_collector_textured_v03_implementation",
    "candidate_collector_path": (
        "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"
    ),
    "candidate_renderer_path": "scripts/render_replay_v03.py",
    "candidate_camera_pose_helper_path": (
        "lewm_genesis/lewm_genesis/render_replay.py"
    ),
    "evaluator_source_path": "scripts/evaluate_go2_world_model_visual_domain_parity_v1.py",
    "comparison_unit": "prebound_ordinary_train_scene_camera_pose",
    "pairing": "same_scene_manifest_same_camera_pose_same_pair_id",
    "exact_scene_count": SCENE_COUNT,
    "exact_poses_per_scene": POSES_PER_SCENE,
    "exact_family_coverage": list(FAMILIES),
    "candidate_duplicate_render_required": True,
    "reference_candidate_pixel_exact_required": True,
    "candidate_duplicate_pixel_exact_required": True,
    "producer_receipt_and_transitive_source_lineage_required": True,
    "required_native_resolution": [224, 224],
    "required_stored_resolution": [224, 224],
    "required_camera_render_call": {"rgb": True, "depth": False},
    "required_raw_manifest_fov_deg": 78.323,
    "genesis_fov_contract": "pass_raw_fov_deg_directly_as_yfov",
    "horizontal_to_vertical_fov_conversion_allowed": False,
    "native_downsampling_allowed": False,
    "required_scene_builder": "render_replay_v03.build_scene",
    "textures_required": True,
    "required_texture_contract": (
        "render_replay_v03_texture_helpers_and_default_options_exact"
    ),
    "structural_texture_categories": list(TEXTURE_CATEGORIES),
    "rendered_geometry_roles": ["walls", "obstacles", "landmarks"],
    "landmarks_textured": False,
    "distractors_rendered": False,
    "manifest_lighting_applied": False,
    "manifest_camera_extrinsic_jitter_applied": False,
    "camera_mount_contract": "nominal_platform_camera_mount",
    "sensor_flag_equality_alone_sufficient": False,
    "statistical_inference_claimed": False,
    "protected_material_allowed": False,
    "posthoc_contract_changes_allowed": False,
}
THRESHOLDS = {
    "exact_scenes_per_domain": SCENE_COUNT,
    "exact_poses_per_scene": POSES_PER_SCENE,
    "exact_frames_per_domain": FRAME_COUNT,
    "required_family_count": SCENE_COUNT,
    "required_reference_candidate_exact_match_count": FRAME_COUNT,
    "required_candidate_duplicate_exact_match_count": FRAME_COUNT,
    "maximum_reference_candidate_normalized_l1": 0.0,
    "minimum_reference_candidate_rgb_ssim": 1.0,
}


class VisualDomainParityError(RuntimeError):
    """Raised before mutable or unproven pixels can become parity evidence."""


def _protected(path: Path) -> bool:
    return any(
        part.lower() == "sealed_test.json"
        or part.lower() == "sealed"
        or part.lower().startswith("sealed_")
        or part.lower() in {"heldout", "held_out", "held-out"}
        or part.lower().startswith("heldout_")
        or part.lower().startswith("held_out_")
        or part.lower().startswith("held-out-")
        for part in Path(path).parts
    )


def _no_symlink_regular(path: Path, *, label: str) -> Path:
    selected = Path(path)
    if _protected(selected):
        raise VisualDomainParityError(f"{label} names protected material")
    if not selected.is_absolute():
        raise VisualDomainParityError(f"{label} must be an absolute path")
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor /= part
        try:
            mode = cursor.lstat().st_mode
        except OSError as exc:
            raise VisualDomainParityError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(mode):
            raise VisualDomainParityError(f"{label} contains a symlink")
    if not selected.is_file() or selected.resolve(strict=True) != selected:
        raise VisualDomainParityError(f"{label} is not a canonical regular file")
    return selected


def _binding_shape(value: object, *, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "file_sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("file_sha256"), str)
        or _SHA.fullmatch(str(value.get("file_sha256"))) is None
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) < 0
    ):
        raise VisualDomainParityError(f"{label} binding is malformed")
    path = _no_symlink_regular(Path(str(value["path"])), label=label)
    return {
        "path": str(path),
        "file_sha256": str(value["file_sha256"]),
        "byte_count": int(value["byte_count"]),
    }


def _binding(value: object, *, label: str) -> dict[str, Any]:
    declared = _binding_shape(value, label=label)
    try:
        actual = pilot.require_binding(declared, label=label)
    except (OSError, pilot.PilotContractError) as exc:
        raise VisualDomainParityError(str(exc)) from exc
    return actual


def _read_bound_json(
    binding: object, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    bound = _binding(binding, label=label)
    try:
        value, actual = pilot.read_bound_json(
            Path(str(bound["path"])),
            expected_sha256=str(bound["file_sha256"]),
            expected_byte_count=int(bound["byte_count"]),
            label=label,
        )
    except (OSError, pilot.PilotContractError) as exc:
        raise VisualDomainParityError(str(exc)) from exc
    if actual != bound or not isinstance(value, Mapping):
        raise VisualDomainParityError(f"{label} binding or JSON object changed")
    return dict(value), bound


def _read_bound_rgb(
    binding: object, *, label: str
) -> tuple[np.ndarray, dict[str, Any]]:
    bound = _binding(binding, label=label)
    selected = Path(str(bound["path"]))
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(selected, flags)
    except OSError as exc:
        raise VisualDomainParityError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        payload_parts: list[bytes] = []
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            payload_parts.append(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or byte_count != int(bound["byte_count"])
        or digest.hexdigest() != bound["file_sha256"]
    ):
        raise VisualDomainParityError(f"{label} changed while read")
    try:
        with Image.open(BytesIO(b"".join(payload_parts))) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise VisualDomainParityError(f"{label} is not decodable RGB") from exc
    if rgb.shape != (RESOLUTION[1], RESOLUTION[0], 3):
        raise VisualDomainParityError(f"{label} is not exact 224x224 RGB")
    if pilot.file_binding(selected) != bound:
        raise VisualDomainParityError(f"{label} changed after decode")
    return rgb, bound


def _raw_rgb_sha256(rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _pose(value: object, *, label: str) -> dict[str, list[float]]:
    required = ("position", "lookat", "up")
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise VisualDomainParityError(f"{label} fields changed")
    result: dict[str, list[float]] = {}
    for name in required:
        raw = value[name]
        if (
            not isinstance(raw, list)
            or len(raw) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in raw
            )
        ):
            raise VisualDomainParityError(f"{label} {name} is invalid")
        result[name] = [float(item) for item in raw]
    return result


def _source_pose(value: object, *, label: str) -> dict[str, list[float]]:
    return _pose(value, label=label)


def _texture_map(
    value: object, *, scene_ids: Sequence[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(value, Mapping) or set(value) != set(scene_ids):
        raise VisualDomainParityError("selected texture scene inventory changed")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for scene_id in scene_ids:
        row = value[scene_id]
        if not isinstance(row, Mapping) or set(row) != set(TEXTURE_CATEGORIES):
            raise VisualDomainParityError("selected texture categories changed")
        result[scene_id] = {}
        for category in TEXTURE_CATEGORIES:
            bound = _binding(row[category], label=f"{scene_id} {category} texture")
            asset = Path(str(bound["path"])).resolve()
            root = (REPO_ROOT / "assets/textures" / category).resolve()
            try:
                relative = asset.relative_to(root)
            except ValueError as exc:
                raise VisualDomainParityError("texture escaped exact category") from exc
            if (
                len(relative.parts) != 1
                or relative.suffix.lower() not in {".jpg", ".jpeg", ".png"}
            ):
                raise VisualDomainParityError("texture is not a category leaf image")
            result[scene_id][category] = bound
    return result


def _mesh_map(
    value: object, *, scene_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, Mapping) or set(value) != set(scene_ids):
        raise VisualDomainParityError("derived mesh scene inventory changed")
    mesh_root = (REPO_ROOT / ".generated/box_meshes").resolve()
    result: dict[str, list[dict[str, Any]]] = {}
    for scene_id in scene_ids:
        rows = value[scene_id]
        if not isinstance(rows, list) or not rows:
            raise VisualDomainParityError(
                f"{scene_id} has no derived textured-mesh closure"
            )
        normalized = [
            _binding(row, label=f"{scene_id} derived textured mesh")
            for row in rows
        ]
        paths = [Path(str(row["path"])).resolve() for row in normalized]
        if (
            paths != sorted(paths)
            or len(paths) != len(set(paths))
            or any(
                path.suffix.lower() != ".obj" or not path.is_relative_to(mesh_root)
                for path in paths
            )
        ):
            raise VisualDomainParityError(
                f"{scene_id} derived textured-mesh inventory changed"
            )
        result[scene_id] = normalized
    return result


def _exact_panel_shape(
    value: object, *, domain: str, binding: Mapping[str, Any]
) -> dict[str, Any]:
    panel_binding = _binding(binding, label=f"{domain} panel")
    required = {
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
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise VisualDomainParityError(f"{domain} panel fields changed")
    if (
        value.get("schema") != PANEL_SCHEMA
        or value.get("domain") != domain
        or value.get("render_contract") != pilot.TEXTURED_V03_RENDER_CONTRACT
    ):
        raise VisualDomainParityError(f"{domain} panel render identity changed")
    rgb_root = Path(str(value.get("rgb_root")))
    if (
        not rgb_root.is_absolute()
        or _protected(rgb_root)
        or rgb_root.is_symlink()
        or not rgb_root.is_dir()
        or rgb_root.resolve(strict=True) != rgb_root
        or (domain == SOURCE_DOMAIN and rgb_root != SOURCE_RGB_ROOT.resolve())
        or (
            domain == CANDIDATE_DOMAIN
            and not rgb_root.is_relative_to(CANDIDATE_ROOT.resolve())
        )
    ):
        raise VisualDomainParityError(f"{domain} RGB root changed")
    producer = _binding(
        value["producer_source_binding"], label=f"{domain} producer source"
    )
    renderer = _binding(
        value["renderer_source_binding"], label=f"{domain} renderer source"
    )
    textures = _binding(
        value["texture_source_binding"], label=f"{domain} texture source"
    )
    expected_producer = REFERENCE_RENDERER if domain == SOURCE_DOMAIN else CANDIDATE_COLLECTOR
    expected_renderer = REFERENCE_RENDERER if domain == SOURCE_DOMAIN else CANDIDATE_RENDERER
    if (
        producer != pilot.file_binding(expected_producer)
        or renderer != pilot.file_binding(expected_renderer)
        or textures != pilot.file_binding(REFERENCE_TEXTURE_SOURCE)
    ):
        raise VisualDomainParityError(f"{domain} source identity changed")
    rows = value.get("rows")
    source_row_fields = {
        "pair_id",
        "scene_id",
        "family",
        "pose_index",
        "camera_pose_world",
        "scene_manifest_binding",
        "producer_frame_identity",
        "rgb_binding",
        "raw_rgb_sha256",
    }
    candidate_row_fields = source_row_fields | {
        "duplicate_producer_frame_identity",
        "duplicate_rgb_binding",
        "duplicate_raw_rgb_sha256",
    }
    expected_fields = source_row_fields if domain == SOURCE_DOMAIN else candidate_row_fields
    if not isinstance(rows, list) or len(rows) != FRAME_COUNT:
        raise VisualDomainParityError(f"{domain} panel must contain exactly 32 rows")
    normalized_rows: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    seen_frame_identities: set[tuple[str, str]] = set()
    seen_rgb_paths: set[Path] = set()
    scene_manifests: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise VisualDomainParityError(f"{domain} panel row fields changed")
        pair_id = raw.get("pair_id")
        scene_id = raw.get("scene_id")
        family = raw.get("family")
        pose_index = raw.get("pose_index")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or pair_id in seen_pairs
            or not isinstance(scene_id, str)
            or _SCENE_ID.fullmatch(scene_id) is None
            or _protected(Path(scene_id))
            or family not in FAMILIES
            or type(pose_index) is not int
            or not 0 <= int(pose_index) < POSES_PER_SCENE
            or pair_id != f"{scene_id}/pose_{pose_index:02d}"
            or not isinstance(raw.get("producer_frame_identity"), str)
            or not raw["producer_frame_identity"]
            or (scene_id, raw["producer_frame_identity"]) in seen_frame_identities
            or not isinstance(raw.get("raw_rgb_sha256"), str)
            or _SHA.fullmatch(str(raw["raw_rgb_sha256"])) is None
        ):
            raise VisualDomainParityError(f"{domain} panel row identity changed")
        manifest_binding = _binding(
            raw["scene_manifest_binding"],
            label=f"{domain} {scene_id} scene manifest",
        )
        if scene_id not in scene_manifests:
            manifest, actual_manifest = _read_bound_json(
                manifest_binding, label=f"{domain} {scene_id} scene manifest"
            )
            if (
                manifest.get("scene_id") != scene_id
                or manifest.get("family") != family
                or manifest.get("split") != "train"
            ):
                raise VisualDomainParityError(
                    f"{domain} scene manifest identity changed"
                )
            scene_manifests[scene_id] = (manifest, actual_manifest)
        elif scene_manifests[scene_id][1] != manifest_binding:
            raise VisualDomainParityError(
                f"{domain} scene manifest varies within scene"
            )
        rgb_binding = _binding(raw["rgb_binding"], label=f"{domain} {pair_id} RGB")
        rgb_path = Path(str(rgb_binding["path"])).resolve()
        try:
            rgb_path.relative_to(rgb_root)
        except ValueError as exc:
            raise VisualDomainParityError(f"{domain} RGB escaped panel root") from exc
        if rgb_path in seen_rgb_paths:
            raise VisualDomainParityError(f"{domain} panel repeats an RGB leaf")
        normalized = {
            **dict(raw),
            "camera_pose_world": _pose(
                raw["camera_pose_world"], label=f"{domain} {pair_id} camera pose"
            ),
            "scene_manifest_binding": manifest_binding,
            "rgb_binding": rgb_binding,
        }
        if domain == CANDIDATE_DOMAIN:
            duplicate_identity = raw.get("duplicate_producer_frame_identity")
            if (
                not isinstance(duplicate_identity, str)
                or not duplicate_identity
                or (str(scene_id), duplicate_identity) in seen_frame_identities
                or duplicate_identity == raw["producer_frame_identity"]
            ):
                raise VisualDomainParityError("candidate duplicate identity changed")
            duplicate = _binding(
                raw["duplicate_rgb_binding"], label=f"candidate {pair_id} duplicate RGB"
            )
            if (
                not isinstance(raw.get("duplicate_raw_rgb_sha256"), str)
                or _SHA.fullmatch(str(raw["duplicate_raw_rgb_sha256"])) is None
            ):
                raise VisualDomainParityError(
                    "candidate duplicate raw RGB identity changed"
                )
            duplicate_path = Path(str(duplicate["path"])).resolve()
            try:
                duplicate_path.relative_to(rgb_root)
            except ValueError as exc:
                raise VisualDomainParityError(
                    "candidate duplicate RGB escaped panel root"
                ) from exc
            if duplicate_path == rgb_path or duplicate_path in seen_rgb_paths:
                raise VisualDomainParityError(
                    "candidate duplicate must be a unique independently written leaf"
                )
            normalized["duplicate_rgb_binding"] = duplicate
            seen_frame_identities.add((str(scene_id), duplicate_identity))
            seen_rgb_paths.add(duplicate_path)
        seen_pairs.add(pair_id)
        seen_frame_identities.add(
            (str(scene_id), str(raw["producer_frame_identity"]))
        )
        seen_rgb_paths.add(rgb_path)
        normalized_rows.append(normalized)
    if [row["pair_id"] for row in normalized_rows] != sorted(seen_pairs):
        raise VisualDomainParityError(f"{domain} rows are not in fixed pair order")
    scene_ids = sorted(scene_manifests)
    if (
        len(scene_ids) != SCENE_COUNT
        or sorted({str(row["family"]) for row in normalized_rows}) != sorted(FAMILIES)
    ):
        raise VisualDomainParityError(f"{domain} panel lacks exact family coverage")
    for scene_id in scene_ids:
        scene_rows = [row for row in normalized_rows if row["scene_id"] == scene_id]
        if [row["pose_index"] for row in scene_rows] != list(range(POSES_PER_SCENE)):
            raise VisualDomainParityError(f"{domain} scene pose coverage changed")
    texture_map = _texture_map(
        value["selected_texture_asset_bindings_by_scene"], scene_ids=scene_ids
    )
    mesh_map = _mesh_map(
        value["mesh_asset_bindings_by_scene"], scene_ids=scene_ids
    )
    for scene_id, (manifest, _manifest_binding) in scene_manifests.items():
        selected = reference_renderer.select_scene_textures(
            visual_seed=int(manifest.get("visual_seed") or 0), scene_id=scene_id
        )
        for category in TEXTURE_CATEGORIES:
            if (
                not isinstance(selected.get(category), str)
                or pilot.file_binding(Path(str(selected[category])))
                != texture_map[scene_id][category]
            ):
                raise VisualDomainParityError(
                    "selected texture binding is not deterministic"
                )
    return {
        "binding": panel_binding,
        "rgb_root": rgb_root,
        "rows": normalized_rows,
        "scene_ids": scene_ids,
        "scene_manifests": scene_manifests,
        "texture_map": texture_map,
        "mesh_map": mesh_map,
        "producer": producer,
        "renderer": renderer,
        "textures": textures,
        "producer_lineage": value["producer_lineage"],
    }


def _read_source_frame_records(
    binding: object,
    *,
    wanted: Mapping[tuple[int, int], str],
    label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Hash a complete historical JSONL while extracting four pre-bound rows."""

    bound = _binding_shape(binding, label=label)
    selected = Path(str(bound["path"]))
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(selected, flags)
    except OSError as exc:
        raise VisualDomainParityError(f"cannot open {label}") from exc
    records: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    byte_count = 0
    buffer = b""
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            buffer += chunk
            lines = buffer.split(b"\n")
            buffer = lines.pop()
            for line in lines:
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise VisualDomainParityError(f"{label} is invalid JSONL") from exc
                key = (row.get("frame_index"), row.get("env_index"))
                if key in wanted:
                    pair_id = wanted[key]
                    if pair_id in records:
                        raise VisualDomainParityError(
                            f"{label} repeats selected frame identity"
                        )
                    records[pair_id] = row
        if buffer.strip():
            try:
                row = json.loads(buffer)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise VisualDomainParityError(f"{label} has an invalid final row") from exc
            key = (row.get("frame_index"), row.get("env_index"))
            if key in wanted:
                records[wanted[key]] = row
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or byte_count != bound["byte_count"]
        or digest.hexdigest() != bound["file_sha256"]
        or set(records) != set(wanted.values())
    ):
        raise VisualDomainParityError(f"{label} binding or selected rows changed")
    return records, bound


def _validate_source_lineage(panel: Mapping[str, Any]) -> dict[str, Any]:
    lineage = panel["producer_lineage"]
    required = {
        "schema",
        "scene_genesis_bindings_by_scene",
        "render_summary_bindings_by_scene",
        "render_plan_bindings_by_scene",
        "frames_jsonl_bindings_by_scene",
    }
    scene_ids = panel["scene_ids"]
    if (
        not isinstance(lineage, Mapping)
        or set(lineage) != required
        or lineage.get("schema") != SOURCE_LINEAGE_SCHEMA
        or any(
            not isinstance(lineage[name], Mapping)
            or set(lineage[name]) != set(scene_ids)
            for name in (
                "scene_genesis_bindings_by_scene",
                "render_summary_bindings_by_scene",
                "render_plan_bindings_by_scene",
                "frames_jsonl_bindings_by_scene",
            )
        )
    ):
        raise VisualDomainParityError("historical source lineage fields changed")
    rows_by_scene = {
        scene_id: [row for row in panel["rows"] if row["scene_id"] == scene_id]
        for scene_id in scene_ids
    }
    normalized = {
        "schema": SOURCE_LINEAGE_SCHEMA,
        "scene_genesis_bindings_by_scene": {},
        "render_summary_bindings_by_scene": {},
        "render_plan_bindings_by_scene": {},
        "frames_jsonl_bindings_by_scene": {},
    }
    for scene_id in scene_ids:
        family = str(rows_by_scene[scene_id][0]["family"])
        genesis_binding = _binding(
            lineage["scene_genesis_bindings_by_scene"][scene_id],
            label=f"historical {scene_id} Genesis scene",
        )
        manifest_binding = panel["scene_manifests"][scene_id][1]
        expected_genesis = Path(str(manifest_binding["path"])).parent / "genesis_scene.json"
        if Path(str(genesis_binding["path"])) != expected_genesis:
            raise VisualDomainParityError(
                "historical Genesis scene escaped the manifest directory"
            )
        summary, summary_binding = _read_bound_json(
            lineage["render_summary_bindings_by_scene"][scene_id],
            label=f"historical {scene_id} render summary",
        )
        plan, plan_binding = _read_bound_json(
            lineage["render_plan_bindings_by_scene"][scene_id],
            label=f"historical {scene_id} render plan",
        )
        summary_path = Path(str(summary_binding["path"]))
        expected_summary = (SOURCE_RGB_ROOT / scene_id / "summary.json").resolve()
        plan_path = Path(str(plan_binding["path"])).resolve()
        rollout_root = (REPO_ROOT / ".generated/datagen_full/rollout/train").resolve()
        declared_frames_binding = _binding_shape(
            lineage["frames_jsonl_bindings_by_scene"][scene_id],
            label=f"historical {scene_id} frames JSONL",
        )
        frames_path = Path(str(declared_frames_binding["path"])).resolve()
        camera = plan.get("camera")
        if (
            summary_path != expected_summary
            or not plan_path.is_relative_to(rollout_root)
            or summary
            != {
                **summary,
                "schema": "lewm_rendered_vision_v03",
                "render_status": "complete",
                "scene_id": scene_id,
                "split": "train",
                "family": family,
                "resolution": 224,
                "visuals": "textured_v03",
                "textures_enabled": True,
            }
            or summary.get("plan") != str(plan_path)
            or plan.get("schema") != "lewm_render_replay_plan_v0"
            or plan.get("scene_id") != scene_id
            or plan.get("scene_family") != family
            or plan.get("split") != "train"
            or plan.get("frame_count") != summary.get("frame_count")
            or not isinstance(camera, Mapping)
            or float(camera.get("fov_deg", -1.0)) != 78.323
            or camera.get("mount_body")
            != {
                "parent_link": "camera_link",
                "rpy_body_rad": [0.0, 0.0, 0.0],
                "xyz_body_m": [0.326, 0.0, 0.043],
            }
            or plan.get("raw_contract_audit_pass") is not True
            or plan.get("raw_data_quality_audit_pass") is not True
            or plan.get("frames_jsonl")
            != str(frames_path)
            or frames_path != plan_path.parent / "frames.jsonl"
        ):
            raise VisualDomainParityError(
                "historical textured-v03 summary/plan identity changed"
            )
        manifest, _manifest_binding = panel["scene_manifests"][scene_id]
        if plan.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise VisualDomainParityError(
                "historical render plan changed scene-manifest semantic identity"
            )
        wanted: dict[tuple[int, int], str] = {}
        for row in rows_by_scene[scene_id]:
            match = _SOURCE_FRAME.fullmatch(str(row["producer_frame_identity"]))
            if match is None:
                raise VisualDomainParityError("historical frame identity changed")
            key = (int(match.group(1)), int(match.group(2)))
            if key[0] % 48 != key[1]:
                raise VisualDomainParityError(
                    "historical frame/env interleave identity changed"
                )
            if key in wanted:
                raise VisualDomainParityError("historical frame selection repeats a row")
            wanted[key] = str(row["pair_id"])
            expected_rgb = (
                SOURCE_RGB_ROOT
                / scene_id
                / "rgb"
                / f"{row['producer_frame_identity']}.png"
            ).resolve()
            if Path(str(row["rgb_binding"]["path"])) != expected_rgb:
                raise VisualDomainParityError(
                    "historical RGB leaf is not the deterministic renderer output"
                )
        records, frames_binding = _read_source_frame_records(
            lineage["frames_jsonl_bindings_by_scene"][scene_id],
            wanted=wanted,
            label=f"historical {scene_id} frames JSONL",
        )
        for row in rows_by_scene[scene_id]:
            record = records[str(row["pair_id"])]
            if (
                not isinstance(record.get("episode"), Mapping)
                or record["episode"].get("split") != "train"
                or record["episode"].get("manifest_sha256")
                != plan.get("manifest_sha256")
                or _source_pose(
                    record.get("camera_pose_world"),
                    label=f"historical {row['pair_id']} camera pose",
                )
                != row["camera_pose_world"]
            ):
                raise VisualDomainParityError(
                    "historical frame record does not prove the declared pose"
                )
        normalized["render_summary_bindings_by_scene"][scene_id] = summary_binding
        normalized["scene_genesis_bindings_by_scene"][scene_id] = genesis_binding
        normalized["render_plan_bindings_by_scene"][scene_id] = plan_binding
        normalized["frames_jsonl_bindings_by_scene"][scene_id] = frames_binding
    return normalized


def _source_binding_by_name(
    rows: object, *, name: str, expected_path: Path, label: str
) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise VisualDomainParityError(f"{label} source closure is malformed")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and set(row) == {"name", "binding"}
        and row.get("name") == name
    ]
    if len(matches) != 1:
        raise VisualDomainParityError(f"{label} lacks exact source {name}")
    binding = _binding(matches[0]["binding"], label=f"{label} source {name}")
    if binding != pilot.file_binding(expected_path):
        raise VisualDomainParityError(f"{label} source {name} identity changed")
    return binding


def _reject_protected_binding_tree(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        if set(value) == {"path", "file_sha256", "byte_count"}:
            _binding_shape(value, label=label)
            return
        for name, item in value.items():
            _reject_protected_binding_tree(item, label=f"{label} {name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_protected_binding_tree(item, label=f"{label} {index}")


def _validate_render_receipt(
    binding: object,
    *,
    expected: Mapping[str, Any],
    ordinal: str,
) -> dict[str, Any]:
    receipt, actual_binding = _read_bound_json(
        binding, label=f"{expected['pair_id']} {ordinal} render receipt"
    )
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_granted_by_this_document",
        "development_only",
        "protected_material_opened",
        "attempt_id",
        "render_ordinal",
        "pair_id",
        "scene_id",
        "family",
        "pose_index",
        "base_position_xyz_m",
        "base_quaternion_wxyz",
        "historical_camera_pose_world",
        "computed_camera_pose_world",
        "scene_manifest_binding",
        "scene_genesis_binding",
        "source_panel_binding",
        "plan_binding",
        "authority_binding",
        "source_commit",
        "source_bindings",
        "render_contract",
        "runtime_bindings",
        "execution_contract",
        "producer_source_binding",
        "renderer_source_binding",
        "camera_pose_helper_source_binding",
        "texture_source_binding",
        "selected_texture_asset_bindings",
        "mesh_asset_bindings",
        "rgb_render_call",
        "physics_steps",
        "producer_frame_identity",
        "rgb_binding",
        "raw_rgb_sha256",
        "rgb_render_wall_seconds",
    }
    if (
        set(receipt) != required
        or receipt.get("schema") != RENDER_RECEIPT_SCHEMA
        or receipt.get("status") != RENDER_RECEIPT_STATUS
        or receipt.get("authority_granted_by_this_document") is not False
        or receipt.get("scientific_claim_granted_by_this_document") is not False
        or receipt.get("development_only") is not True
        or receipt.get("protected_material_opened") is not False
        or receipt.get("render_ordinal") != ordinal
        or any(
            receipt.get(field) != expected[field]
            for field in (
                "attempt_id",
                "pair_id",
                "scene_id",
                "family",
                "pose_index",
                "base_position_xyz_m",
                "base_quaternion_wxyz",
                "historical_camera_pose_world",
                "computed_camera_pose_world",
                "scene_manifest_binding",
                "scene_genesis_binding",
                "source_panel_binding",
                "plan_binding",
                "authority_binding",
                "source_commit",
                "source_bindings",
                "render_contract",
                "runtime_bindings",
                "execution_contract",
                "producer_source_binding",
                "renderer_source_binding",
                "camera_pose_helper_source_binding",
                "texture_source_binding",
                "selected_texture_asset_bindings",
                "mesh_asset_bindings",
            )
        )
        or receipt.get("rgb_render_call") != {"rgb": True, "depth": False}
        or receipt.get("physics_steps") != 0
        or receipt.get("producer_frame_identity")
        != expected[f"{ordinal}_producer_frame_identity"]
        or receipt.get("rgb_binding") != expected[f"{ordinal}_rgb_binding"]
        or receipt.get("raw_rgb_sha256")
        != expected[f"{ordinal}_raw_rgb_sha256"]
        or isinstance(receipt.get("rgb_render_wall_seconds"), bool)
        or not isinstance(receipt.get("rgb_render_wall_seconds"), (int, float))
        or not math.isfinite(float(receipt["rgb_render_wall_seconds"]))
        or float(receipt["rgb_render_wall_seconds"]) < 0.0
    ):
        raise VisualDomainParityError(
            f"{expected['pair_id']} {ordinal} render provenance changed"
        )
    return actual_binding


def _validate_candidate_lineage(
    panel: Mapping[str, Any], *, source_panel_binding: Mapping[str, Any]
) -> dict[str, Any]:
    lineage = panel["producer_lineage"]
    if (
        not isinstance(lineage, Mapping)
        or set(lineage) != {"schema", "generation_receipt_binding"}
        or lineage.get("schema") != CANDIDATE_LINEAGE_SCHEMA
    ):
        raise VisualDomainParityError("candidate lineage fields changed")
    generation, generation_binding = _read_bound_json(
        lineage["generation_receipt_binding"],
        label="candidate parity generation receipt",
    )
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_granted_by_this_document",
        "development_only",
        "protected_material_opened",
        "attempt_id",
        "output_root",
        "plan_binding",
        "authority_binding",
        "source_review_binding",
        "source_commit",
        "source_panel_binding",
        "render_contract",
        "comparison_contract",
        "expected_counts",
        "runtime_bindings",
        "execution_contract",
        "scene_corpus_manifest_bindings",
        "texture_asset_bindings",
        "mesh_asset_bindings",
        "producer_source_binding",
        "renderer_source_binding",
        "camera_pose_helper_source_binding",
        "texture_source_binding",
        "selected_texture_asset_bindings_by_scene",
        "mesh_asset_bindings_by_scene",
        "source_bindings",
        "render_rows",
        "observed_counts",
        "wall_seconds",
    }
    if (
        set(generation) != required
        or generation.get("schema") != CANDIDATE_GENERATION_RECEIPT_SCHEMA
        or generation.get("status") != CANDIDATE_GENERATION_STATUS
        or generation.get("authority_granted_by_this_document") is not False
        or generation.get("scientific_claim_granted_by_this_document") is not False
        or generation.get("development_only") is not True
        or generation.get("protected_material_opened") is not False
        or generation.get("render_contract") != pilot.TEXTURED_V03_RENDER_CONTRACT
        or generation.get("producer_source_binding") != panel["producer"]
        or generation.get("renderer_source_binding") != panel["renderer"]
        or generation.get("camera_pose_helper_source_binding")
        != pilot.file_binding(CANDIDATE_CAMERA_POSE_HELPER)
        or generation.get("texture_source_binding") != panel["textures"]
        or generation.get("selected_texture_asset_bindings_by_scene")
        != panel["texture_map"]
        or generation.get("mesh_asset_bindings_by_scene") != panel["mesh_map"]
        or generation.get("source_panel_binding") != source_panel_binding
        or not isinstance(generation.get("attempt_id"), str)
        or not generation["attempt_id"]
        or isinstance(generation.get("wall_seconds"), bool)
        or not isinstance(generation.get("wall_seconds"), (int, float))
        or not math.isfinite(float(generation["wall_seconds"]))
        or float(generation["wall_seconds"]) <= 0.0
    ):
        raise VisualDomainParityError("candidate generation receipt changed")
    plan, plan_binding = _read_bound_json(
        generation["plan_binding"], label="candidate parity producer plan"
    )
    try:
        normalized_plan = parity_plan.validate_plan_v1(
            plan, require_fresh_output=False
        )
    except parity_plan.VisualDomainParityPlanError as exc:
        raise VisualDomainParityError(str(exc)) from exc
    _reject_protected_binding_tree(normalized_plan, label="candidate producer plan")
    output_root = Path(str(normalized_plan["output_root"])).resolve()
    if (
        normalized_plan.get("schema") != parity_plan.PLAN_SCHEMA
        or normalized_plan.get("purpose") != parity_plan.PURPOSE
        or normalized_plan.get("render_contract") != pilot.TEXTURED_V03_RENDER_CONTRACT
        or normalized_plan.get("attempt_id") != generation["attempt_id"]
        or generation.get("output_root") != str(output_root)
        or panel["rgb_root"] != output_root / "scenes"
        or Path(str(panel["binding"]["path"])) != output_root / "candidate_panel.json"
        or Path(str(generation_binding["path"]))
        != output_root / "generation_receipt.json"
        or normalized_plan.get("source_panel_binding") != source_panel_binding
        or generation.get("comparison_contract")
        != normalized_plan["comparison_contract"]
        or generation.get("expected_counts") != normalized_plan["expected_counts"]
        or generation.get("runtime_bindings") != normalized_plan["runtime_bindings"]
        or generation.get("execution_contract")
        != normalized_plan["execution_contract"]
        or generation.get("scene_corpus_manifest_bindings")
        != normalized_plan["scene_corpus_manifest_bindings"]
        or generation.get("texture_asset_bindings")
        != normalized_plan["texture_asset_bindings"]
        or generation.get("mesh_asset_bindings")
        != normalized_plan["mesh_asset_bindings"]
    ):
        raise VisualDomainParityError("candidate parity producer plan scope changed")
    authority, authority_binding = _read_bound_json(
        generation["authority_binding"], label="candidate parity producer authority"
    )
    review, review_binding = _read_bound_json(
        generation["source_review_binding"],
        label="candidate parity source review",
    )
    if generation["authority_binding"] != authority_binding:
        raise VisualDomainParityError("candidate authority binding changed")
    try:
        normalized_authority = parity_authority.validate_authority_v1(
            authority,
            plan=normalized_plan,
            plan_binding=plan_binding,
            review=review,
            review_binding=review_binding,
            require_fresh_output=False,
        )
    except parity_authority.VisualDomainParityAuthorityError as exc:
        raise VisualDomainParityError(str(exc)) from exc
    if (
        generation.get("source_review_binding") != review_binding
        or generation.get("source_commit") != normalized_authority["source_commit"]
        or generation.get("source_bindings")
        != normalized_authority["source_bindings"]
        or generation.get("wall_seconds")
        > normalized_authority["caps"]["wall_seconds"]
    ):
        raise VisualDomainParityError("candidate authority/source closure changed")
    _source_binding_by_name(
        normalized_authority["source_bindings"],
        name="collector",
        expected_path=CANDIDATE_COLLECTOR,
        label="candidate authority",
    )
    _source_binding_by_name(
        normalized_authority["source_bindings"],
        name="genesis_render_replay",
        expected_path=CANDIDATE_CAMERA_POSE_HELPER,
        label="candidate authority",
    )
    _source_binding_by_name(
        normalized_authority["source_bindings"],
        name="historical_textured_v03_renderer",
        expected_path=CANDIDATE_RENDERER,
        label="candidate authority",
    )
    _source_binding_by_name(
        normalized_authority["source_bindings"],
        name="textures",
        expected_path=REFERENCE_TEXTURE_SOURCE,
        label="candidate authority",
    )
    render_rows = generation.get("render_rows")
    required_row = {
        "pair_id",
        "scene_id",
        "family",
        "pose_index",
        "base_position_xyz_m",
        "base_quaternion_wxyz",
        "camera_pose_world",
        "scene_manifest_binding",
        "scene_genesis_binding",
        "selected_texture_asset_bindings",
        "mesh_asset_bindings",
        "candidate_producer_frame_identity",
        "duplicate_producer_frame_identity",
        "candidate_rgb_binding",
        "duplicate_rgb_binding",
        "candidate_raw_rgb_sha256",
        "duplicate_raw_rgb_sha256",
        "candidate_render_receipt_binding",
        "duplicate_render_receipt_binding",
    }
    if (
        not isinstance(render_rows, list)
        or len(render_rows) != FRAME_COUNT
        or any(not isinstance(row, Mapping) or set(row) != required_row for row in render_rows)
        or [row.get("pair_id") for row in render_rows]
        != [row["pair_id"] for row in panel["rows"]]
    ):
        raise VisualDomainParityError("candidate generation row inventory changed")
    planned_pairs = {
        pose["pair_id"]: (scene_index, scene, pose)
        for scene_index, scene in enumerate(normalized_plan["scenes"])
        for pose in scene["poses"]
    }
    stored_rgb_bytes = sum(
        int(row[name]["byte_count"])
        for row in render_rows
        for name in ("candidate_rgb_binding", "duplicate_rgb_binding")
    )
    if generation.get("observed_counts") != {
        "scenes": SCENE_COUNT,
        "poses": FRAME_COUNT,
        "candidate_rgb_frames": FRAME_COUNT,
        "duplicate_rgb_frames": FRAME_COUNT,
        "rgb_render_calls": FRAME_COUNT * 2,
        "auxiliary_depth_render_calls": 0,
        "physics_steps": 0,
        "stored_rgb_bytes": stored_rgb_bytes,
    } or stored_rgb_bytes > parity_authority.MAX_STORED_RGB_BYTES:
        raise VisualDomainParityError("candidate generation counts/caps changed")
    normalized_render_receipts = []
    for panel_row, generated in zip(panel["rows"], render_rows, strict=True):
        if panel_row["pair_id"] not in planned_pairs:
            raise VisualDomainParityError("candidate pair is outside producer plan")
        scene_index, planned_scene, planned_pose = planned_pairs[panel_row["pair_id"]]
        expected_common = {
            "pair_id": panel_row["pair_id"],
            "scene_id": panel_row["scene_id"],
            "family": panel_row["family"],
            "pose_index": panel_row["pose_index"],
            "base_position_xyz_m": planned_pose["base_position_xyz_m"],
            "base_quaternion_wxyz": planned_pose["base_quaternion_wxyz"],
            "camera_pose_world": panel_row["camera_pose_world"],
            "scene_manifest_binding": panel_row["scene_manifest_binding"],
            "scene_genesis_binding": planned_scene["scene_genesis_binding"],
            "selected_texture_asset_bindings": panel["texture_map"][panel_row["scene_id"]],
            "mesh_asset_bindings": panel["mesh_map"][panel_row["scene_id"]],
            "candidate_producer_frame_identity": panel_row["producer_frame_identity"],
            "duplicate_producer_frame_identity": panel_row["duplicate_producer_frame_identity"],
            "candidate_rgb_binding": panel_row["rgb_binding"],
            "duplicate_rgb_binding": panel_row["duplicate_rgb_binding"],
            "candidate_raw_rgb_sha256": panel_row["raw_rgb_sha256"],
            "duplicate_raw_rgb_sha256": panel_row["duplicate_raw_rgb_sha256"],
        }
        if any(generated.get(name) != value for name, value in expected_common.items()):
            raise VisualDomainParityError(
                "candidate panel row is not the generation-receipt row"
            )
        scene_root = output_root / "scenes" / f"{scene_index:02d}_{panel_row['scene_id']}"
        pose_root = scene_root / "rows" / f"pose_{panel_row['pose_index']:02d}"
        if (
            Path(str(panel_row["rgb_binding"]["path"]))
            != pose_root / "candidate.png"
            or Path(str(panel_row["duplicate_rgb_binding"]["path"]))
            != pose_root / "duplicate.png"
            or Path(str(generated["candidate_render_receipt_binding"]["path"]))
            != pose_root / "candidate_receipt.json"
            or Path(str(generated["duplicate_render_receipt_binding"]["path"]))
            != pose_root / "duplicate_receipt.json"
        ):
            raise VisualDomainParityError("candidate output layout changed")
        receipt_expected = {
            "attempt_id": generation["attempt_id"],
            "pair_id": panel_row["pair_id"],
            "scene_id": panel_row["scene_id"],
            "family": panel_row["family"],
            "pose_index": panel_row["pose_index"],
            "base_position_xyz_m": planned_pose["base_position_xyz_m"],
            "base_quaternion_wxyz": planned_pose["base_quaternion_wxyz"],
            "historical_camera_pose_world": panel_row["camera_pose_world"],
            "computed_camera_pose_world": panel_row["camera_pose_world"],
            "scene_manifest_binding": panel_row["scene_manifest_binding"],
            "scene_genesis_binding": planned_scene["scene_genesis_binding"],
            "source_panel_binding": source_panel_binding,
            "plan_binding": plan_binding,
            "authority_binding": authority_binding,
            "source_commit": normalized_authority["source_commit"],
            "source_bindings": normalized_authority["source_bindings"],
            "render_contract": pilot.TEXTURED_V03_RENDER_CONTRACT,
            "runtime_bindings": normalized_plan["runtime_bindings"],
            "execution_contract": normalized_plan["execution_contract"],
            "producer_source_binding": panel["producer"],
            "renderer_source_binding": panel["renderer"],
            "camera_pose_helper_source_binding": generation[
                "camera_pose_helper_source_binding"
            ],
            "texture_source_binding": panel["textures"],
            "selected_texture_asset_bindings": panel["texture_map"][panel_row["scene_id"]],
            "mesh_asset_bindings": panel["mesh_map"][panel_row["scene_id"]],
            "candidate_producer_frame_identity": panel_row["producer_frame_identity"],
            "duplicate_producer_frame_identity": panel_row["duplicate_producer_frame_identity"],
            "candidate_rgb_binding": panel_row["rgb_binding"],
            "duplicate_rgb_binding": panel_row["duplicate_rgb_binding"],
            "candidate_raw_rgb_sha256": panel_row["raw_rgb_sha256"],
            "duplicate_raw_rgb_sha256": panel_row["duplicate_raw_rgb_sha256"],
        }
        normalized_render_receipts.append(
            {
                "candidate": _validate_render_receipt(
                    generated["candidate_render_receipt_binding"],
                    expected=receipt_expected,
                    ordinal="candidate",
                ),
                "duplicate": _validate_render_receipt(
                    generated["duplicate_render_receipt_binding"],
                    expected=receipt_expected,
                    ordinal="duplicate",
                ),
            }
        )
    return {
        "schema": CANDIDATE_LINEAGE_SCHEMA,
        "generation_receipt_binding": generation_binding,
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "source_review_binding": review_binding,
        "render_receipt_bindings": normalized_render_receipts,
    }


def _global_ssim(source: np.ndarray, candidate: np.ndarray) -> float:
    """Deterministic diagnostic; exact equality, not SSIM, is the pass gate."""

    if np.array_equal(source, candidate):
        return 1.0
    left = source.astype(np.float64) / 255.0
    right = candidate.astype(np.float64) / 255.0
    scores = []
    for channel in range(3):
        x = left[..., channel].reshape(-1)
        y = right[..., channel].reshape(-1)
        mx, my = float(x.mean()), float(y.mean())
        vx, vy = float(x.var()), float(y.var())
        covariance = float(((x - mx) * (y - my)).mean())
        numerator = (2.0 * mx * my + 0.01**2) * (
            2.0 * covariance + 0.03**2
        )
        denominator = (mx * mx + my * my + 0.01**2) * (
            vx + vy + 0.03**2
        )
        scores.append(numerator / denominator if denominator > 0.0 else 1.0)
    return float(np.mean(scores))


def _measure(source: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    if (
        source["scene_ids"] != candidate["scene_ids"]
        or source["texture_map"] != candidate["texture_map"]
        or source["mesh_map"] != candidate["mesh_map"]
        or len(source["rows"]) != FRAME_COUNT
        or len(candidate["rows"]) != FRAME_COUNT
    ):
        raise VisualDomainParityError("paired panels changed scenes or texture leaves")
    reference_candidate_exact = 0
    candidate_duplicate_exact = 0
    maximum_l1 = 0.0
    minimum_ssim = 1.0
    for reference_row, candidate_row in zip(
        source["rows"], candidate["rows"], strict=True
    ):
        identity_fields = (
            "pair_id",
            "scene_id",
            "family",
            "pose_index",
            "camera_pose_world",
            "scene_manifest_binding",
        )
        if any(
            reference_row[field] != candidate_row[field]
            for field in identity_fields
        ):
            raise VisualDomainParityError("paired scene/camera identity changed")
        reference_rgb, _ = _read_bound_rgb(
            reference_row["rgb_binding"],
            label=f"reference {reference_row['pair_id']} RGB",
        )
        candidate_rgb, _ = _read_bound_rgb(
            candidate_row["rgb_binding"],
            label=f"candidate {candidate_row['pair_id']} RGB",
        )
        duplicate_rgb, _ = _read_bound_rgb(
            candidate_row["duplicate_rgb_binding"],
            label=f"candidate {candidate_row['pair_id']} duplicate RGB",
        )
        if (
            _raw_rgb_sha256(reference_rgb) != reference_row["raw_rgb_sha256"]
            or _raw_rgb_sha256(candidate_rgb) != candidate_row["raw_rgb_sha256"]
            or _raw_rgb_sha256(duplicate_rgb)
            != candidate_row["duplicate_raw_rgb_sha256"]
        ):
            raise VisualDomainParityError(
                "decoded raw RGB hash disagrees with a panel row"
            )
        reference_candidate_exact += int(np.array_equal(reference_rgb, candidate_rgb))
        candidate_duplicate_exact += int(np.array_equal(candidate_rgb, duplicate_rgb))
        maximum_l1 = max(
            maximum_l1,
            float(
                np.abs(
                    reference_rgb.astype(np.float64)
                    - candidate_rgb.astype(np.float64)
                ).mean()
                / 255.0
            ),
        )
        minimum_ssim = min(minimum_ssim, _global_ssim(reference_rgb, candidate_rgb))
    return {
        "scene_count": SCENE_COUNT,
        "poses_per_scene": POSES_PER_SCENE,
        "reference_frame_count": FRAME_COUNT,
        "candidate_frame_count": FRAME_COUNT,
        "duplicate_frame_count": FRAME_COUNT,
        "families": list(FAMILIES),
        "reference_candidate_exact_match_count": reference_candidate_exact,
        "candidate_duplicate_exact_match_count": candidate_duplicate_exact,
        "maximum_reference_candidate_normalized_l1": maximum_l1,
        "minimum_reference_candidate_rgb_ssim": minimum_ssim,
    }


def _passes(measurements: Mapping[str, Any]) -> bool:
    return measurements == {
        "scene_count": THRESHOLDS["exact_scenes_per_domain"],
        "poses_per_scene": THRESHOLDS["exact_poses_per_scene"],
        "reference_frame_count": THRESHOLDS["exact_frames_per_domain"],
        "candidate_frame_count": THRESHOLDS["exact_frames_per_domain"],
        "duplicate_frame_count": THRESHOLDS["exact_frames_per_domain"],
        "families": list(FAMILIES),
        "reference_candidate_exact_match_count": THRESHOLDS[
            "required_reference_candidate_exact_match_count"
        ],
        "candidate_duplicate_exact_match_count": THRESHOLDS[
            "required_candidate_duplicate_exact_match_count"
        ],
        "maximum_reference_candidate_normalized_l1": THRESHOLDS[
            "maximum_reference_candidate_normalized_l1"
        ],
        "minimum_reference_candidate_rgb_ssim": THRESHOLDS[
            "minimum_reference_candidate_rgb_ssim"
        ],
    }


def evaluate_v1(
    *,
    source_panel: Mapping[str, Any],
    source_panel_binding: Mapping[str, Any],
    candidate_panel: Mapping[str, Any],
    candidate_panel_binding: Mapping[str, Any],
) -> dict[str, Any]:
    source = _exact_panel_shape(
        source_panel, domain=SOURCE_DOMAIN, binding=source_panel_binding
    )
    candidate = _exact_panel_shape(
        candidate_panel, domain=CANDIDATE_DOMAIN, binding=candidate_panel_binding
    )
    source_lineage = _validate_source_lineage(source)
    candidate_lineage = _validate_candidate_lineage(
        candidate, source_panel_binding=source["binding"]
    )
    measurements = _measure(source, candidate)
    result = {
        "schema": RESULT_SCHEMA,
        "status": PASS_STATUS if _passes(measurements) else FAIL_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "development_only": True,
        "protected_material_opened": False,
        "comparison_contract": dict(COMPARISON_CONTRACT),
        "thresholds": dict(THRESHOLDS),
        "measurements": measurements,
        "evidence_scene_ids": list(source["scene_ids"]),
        "source_rgb_reference_binding": dict(source["binding"]),
        "candidate_rgb_panel_binding": dict(candidate["binding"]),
        "source_producer_lineage": source_lineage,
        "candidate_producer_lineage": candidate_lineage,
        "candidate_collector_source_binding": dict(candidate["producer"]),
        "candidate_renderer_source_binding": dict(candidate["renderer"]),
        "reference_renderer_source_binding": dict(source["renderer"]),
        "reference_texture_source_binding": dict(source["textures"]),
        "evaluator_source_binding": pilot.file_binding(Path(__file__)),
        "selected_texture_asset_bindings_by_scene": source["texture_map"],
    }
    # Terminal rehashes: panels, manifests, source files, and texture leaves.
    for panel in (source, candidate):
        if _binding(panel["binding"], label="terminal parity panel") != panel["binding"]:
            raise VisualDomainParityError("parity panel changed during evaluation")
        for _scene, (_document, binding) in panel["scene_manifests"].items():
            if _binding(binding, label="terminal scene manifest") != binding:
                raise VisualDomainParityError("scene manifest changed during evaluation")
        for assets in panel["texture_map"].values():
            for binding in assets.values():
                if _binding(binding, label="terminal texture asset") != binding:
                    raise VisualDomainParityError("texture asset changed during evaluation")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-panel", "candidate-panel"):
        parser.add_argument(f"--{name}", required=True, type=Path)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
        parser.add_argument(f"--expected-{name}-byte-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded = {}
    for key, label in (
        ("source_panel", "historical textured-v03 parity panel"),
        ("candidate_panel", "candidate double-render parity panel"),
    ):
        path = _no_symlink_regular(getattr(args, key), label=label)
        caller_binding = {
            "path": str(path),
            "file_sha256": getattr(args, f"expected_{key}_sha256"),
            "byte_count": getattr(args, f"expected_{key}_byte_count"),
        }
        loaded[key] = _read_bound_json(caller_binding, label=label)
    result = evaluate_v1(
        source_panel=loaded["source_panel"][0],
        source_panel_binding=loaded["source_panel"][1],
        candidate_panel=loaded["candidate_panel"][0],
        candidate_panel_binding=loaded["candidate_panel"][1],
    )
    binding = pilot.write_json_exclusive(args.output, result)
    print(json.dumps({"result": binding, "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_DOMAIN",
    "CANDIDATE_GENERATION_RECEIPT_SCHEMA",
    "CANDIDATE_LINEAGE_SCHEMA",
    "COMPARISON_CONTRACT",
    "FAIL_STATUS",
    "PANEL_SCHEMA",
    "PASS_STATUS",
    "RENDER_RECEIPT_SCHEMA",
    "RESULT_SCHEMA",
    "SOURCE_DOMAIN",
    "SOURCE_LINEAGE_SCHEMA",
    "THRESHOLDS",
    "VisualDomainParityError",
    "_global_ssim",
    "_measure",
    "_passes",
    "evaluate_v1",
]
