#!/usr/bin/env python3
"""Supervise one exactly authorized textured-v03 RGB parity generation.

The public supervisor validates the committed source review, immutable plan,
runtime, textures, derived meshes, graphics device, and fresh output boundary
before atomically consuming the attempt.  It then starts exactly eight
plan-bound child workers, one per scene.  Each worker makes two independent
RGB-only calls through the shared historical-pose helper for each of four
pre-bound poses.  There is no retry, resume, refill, overwrite, depth render,
physics step, training, held-out access, or promotion path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
for _package_root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import build_go2_world_model_visual_domain_parity_authority_v1 as authority_builder  # noqa: E402
from scripts import build_go2_world_model_visual_domain_parity_plan_v1 as plan_builder  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as runtime_kernel  # noqa: E402
from scripts import evaluate_go2_world_model_visual_domain_parity_v1 as evaluator  # noqa: E402
from scripts import run_go2_world_model_counterfactual_calibration_authorized_v1 as graphics_supervisor  # noqa: E402


RESERVATION_SCHEMA = "lewm_go2_world_model_visual_domain_parity_reservation_v1"
RESERVATION_STATUS = "ATTEMPT_CONSUMED_NO_RETRY_OR_RESUME"
SCENE_RESULT_SCHEMA = "lewm_go2_world_model_visual_domain_parity_scene_result_v1"
SCENE_RESULT_STATUS = "COMPLETE_EXACT_4_POSE_DOUBLE_RENDER"
TERMINAL_SCHEMA = pilot.TEXTURED_V03_PARITY_TERMINAL_SCHEMA
TERMINAL_SUCCESS_STATUS = pilot.TEXTURED_V03_PARITY_TERMINAL_SUCCESS_STATUS
TERMINAL_FAILURE_STATUS = "TERMINAL_FAILURE_NO_RETRY_OR_RESUME"
CAPABILITY_ENV = "LEWM_VISUAL_PARITY_WORKER_CAPABILITY"
NETWORK_ENV_NAMES = {
    "ALL_PROXY",
    "FTP_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "all_proxy",
    "ftp_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


class VisualDomainParitySupervisionError(RuntimeError):
    """Raised when the one-shot RGB parity boundary fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(pilot.canonical_json_bytes(value)).hexdigest()


def _raw_rgb_sha256(rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _protected(path: Path) -> bool:
    return any(
        part.lower() == "sealed_test.json"
        or part.lower() == "sealed"
        or part.lower().startswith("sealed_")
        or part.lower() in {"heldout", "held_out", "held-out", "protected"}
        or part.lower().startswith("heldout_")
        or part.lower().startswith("held_out_")
        or part.lower().startswith("held-out-")
        or part.lower().startswith("protected_")
        for part in Path(path).parts
    )


def _nofollow_regular(path: Path, *, label: str) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    if not selected.is_absolute() or _protected(selected):
        raise VisualDomainParitySupervisionError(f"{label} path is forbidden")
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor /= part
        try:
            mode = cursor.lstat().st_mode
        except OSError as exc:
            raise VisualDomainParitySupervisionError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(mode):
            raise VisualDomainParitySupervisionError(f"{label} contains a symlink")
    if not selected.is_file() or selected.resolve(strict=True) != selected:
        raise VisualDomainParitySupervisionError(f"{label} is not a regular file")
    return selected


def _read_bound_document(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _SHA.fullmatch(expected_sha256) is None or expected_byte_count <= 0:
        raise VisualDomainParitySupervisionError(f"{label} caller binding is malformed")
    selected = _nofollow_regular(path, label=label)
    try:
        value, binding = pilot.read_bound_json(
            selected,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
            label=label,
        )
    except pilot.PilotContractError as exc:
        raise VisualDomainParitySupervisionError(str(exc)) from exc
    if not isinstance(value, Mapping):
        raise VisualDomainParitySupervisionError(f"{label} is not a JSON object")
    return dict(value), binding


def _read_binding_document(
    binding: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        normalized = pilot.require_binding(binding, label=label)
    except pilot.PilotContractError as exc:
        raise VisualDomainParitySupervisionError(str(exc)) from exc
    selected = _nofollow_regular(Path(str(normalized["path"])), label=label)
    try:
        value, actual = pilot.read_bound_json(
            selected,
            expected_sha256=str(normalized["file_sha256"]),
            expected_byte_count=int(normalized["byte_count"]),
            label=label,
        )
    except pilot.PilotContractError as exc:
        raise VisualDomainParitySupervisionError(str(exc)) from exc
    if actual != normalized or not isinstance(value, Mapping):
        raise VisualDomainParitySupervisionError(f"{label} changed")
    return dict(value), actual


def load_and_validate_chain_v1(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    expected_plan_byte_count: int,
    authority_path: Path,
    expected_authority_sha256: str,
    expected_authority_byte_count: int,
    require_fresh_output: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    plan, plan_binding = _read_bound_document(
        plan_path,
        expected_sha256=expected_plan_sha256,
        expected_byte_count=expected_plan_byte_count,
        label="visual-domain parity plan",
    )
    authority, authority_binding = _read_bound_document(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
        label="visual-domain parity authority",
    )
    review_binding = authority.get("review_binding")
    if not isinstance(review_binding, Mapping):
        raise VisualDomainParitySupervisionError("authority review binding changed")
    review, actual_review_binding = _read_binding_document(
        review_binding, label="visual-domain parity source review"
    )
    try:
        normalized_authority = authority_builder.validate_authority_v1(
            authority,
            plan=plan,
            plan_binding=plan_binding,
            review=review,
            review_binding=actual_review_binding,
            require_fresh_output=require_fresh_output,
        )
        normalized_plan = plan_builder.validate_plan_v1(
            plan, require_fresh_output=require_fresh_output
        )
    except (
        authority_builder.VisualDomainParityAuthorityError,
        plan_builder.VisualDomainParityPlanError,
    ) as exc:
        raise VisualDomainParitySupervisionError(str(exc)) from exc
    return (
        normalized_plan,
        plan_binding,
        normalized_authority,
        authority_binding,
        review,
        actual_review_binding,
    )


def _rehash_chain(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    review: Mapping[str, Any],
    review_binding: Mapping[str, Any],
    require_fresh_output: bool,
) -> None:
    try:
        if pilot.require_binding(authority_binding, label="parity authority") != authority_binding:
            raise VisualDomainParitySupervisionError("parity authority changed")
        authority_builder.validate_authority_v1(
            authority,
            plan=plan,
            plan_binding=plan_binding,
            review=review,
            review_binding=review_binding,
            require_fresh_output=require_fresh_output,
        )
    except (pilot.PilotContractError, authority_builder.VisualDomainParityAuthorityError) as exc:
        raise VisualDomainParitySupervisionError(str(exc)) from exc


def _fresh_attempt_root(path: Path) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    development = (REPO_ROOT / ".generated/dev").resolve()
    if (
        not candidate.is_absolute()
        or _protected(candidate)
        or not candidate.is_relative_to(development)
        or candidate == development
    ):
        raise VisualDomainParitySupervisionError("attempt root escaped .generated/dev")
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise VisualDomainParitySupervisionError("attempt path contains a symlink")
        if not cursor.exists():
            break
    if candidate.exists() or candidate.is_symlink():
        raise VisualDomainParitySupervisionError("attempt root is not fresh")
    parent = candidate.parent
    if not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise VisualDomainParitySupervisionError("attempt parent is not canonical")
    return candidate


def _disk_preflight(
    *, output_parent: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    caps = authority["caps"]
    required = int(caps["required_preflight_free_bytes"])
    if (
        int(caps["maximum_parity_output_bytes"])
        != authority_builder.MAX_PARITY_OUTPUT_BYTES
        or int(caps["projected_pipeline_new_bytes"])
        != authority_builder.PROJECTED_PIPELINE_NEW_BYTES
        or int(caps["free_space_margin_bytes"])
        != authority_builder.FREE_SPACE_MARGIN_BYTES
        or required != authority_builder.REQUIRED_PREFLIGHT_FREE_BYTES
        or required != int(caps["projected_pipeline_new_bytes"])
        + int(caps["free_space_margin_bytes"])
        or required > 4 * 1024**3
    ):
        raise VisualDomainParitySupervisionError("disk authority caps changed")
    try:
        stats = os.statvfs(output_parent)
    except OSError as exc:
        raise VisualDomainParitySupervisionError("cannot inspect output filesystem") from exc
    available = int(stats.f_bavail) * int(stats.f_frsize)
    if available < required:
        raise VisualDomainParitySupervisionError(
            f"insufficient pre-reservation disk: {available} < {required} bytes"
        )
    return {
        "filesystem_path": str(output_parent),
        "available_bytes_before_reservation": available,
        "projected_pipeline_new_bytes": int(caps["projected_pipeline_new_bytes"]),
        "free_space_margin_bytes": int(caps["free_space_margin_bytes"]),
        "required_preflight_free_bytes": required,
        "maximum_parity_output_bytes": int(caps["maximum_parity_output_bytes"]),
        "passed": True,
    }


def _source_binding(authority: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    matches = [
        row["binding"]
        for row in authority["source_bindings"]
        if row["name"] == name
    ]
    if len(matches) != 1:
        raise VisualDomainParitySupervisionError(
            f"reviewed source closure has {len(matches)} entries for {name}"
        )
    return dict(matches[0])


def _write_png_exclusive(path: Path, rgb: np.ndarray) -> dict[str, Any]:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.shape != (224, 224, 3):
        raise VisualDomainParitySupervisionError("candidate RGB shape changed")
    buffer = BytesIO()
    image = Image.fromarray(array)
    if image.mode != "RGB":
        raise VisualDomainParitySupervisionError("candidate RGB mode changed")
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o640)
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    except OSError as exc:
        raise VisualDomainParitySupervisionError(f"cannot exclusively write {path}") from exc
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    return pilot.file_binding(path)


def _pose(value: Mapping[str, Any]) -> dict[str, list[float]]:
    if set(value) != {"position", "lookat", "up"}:
        raise VisualDomainParitySupervisionError("computed camera pose fields changed")
    result = {}
    for name in ("position", "lookat", "up"):
        vector = np.asarray(value[name], dtype=np.float32)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise VisualDomainParitySupervisionError("computed camera pose changed")
        result[name] = [float(item) for item in vector]
    return result


def _expected_scene_dir(plan: Mapping[str, Any], scene_index: int) -> Path:
    scene = plan["scenes"][scene_index]
    return Path(str(plan["output_root"])) / "scenes" / f"{scene_index:02d}_{scene['scene_id']}"


def _canonical_directory(path: Path, *, label: str) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    if _protected(selected) or selected.is_symlink():
        raise VisualDomainParitySupervisionError(f"{label} is forbidden or a symlink")
    try:
        mode = selected.lstat().st_mode
    except OSError as exc:
        raise VisualDomainParitySupervisionError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(mode) or selected.resolve(strict=True) != selected:
        raise VisualDomainParitySupervisionError(f"{label} is not canonical")
    return selected


def _validate_reservation(
    *,
    binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    capability: str | None,
) -> dict[str, Any]:
    reservation, actual = _read_binding_document(binding, label="parity reservation")
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_granted_by_this_document",
        "attempt_id",
        "output_root",
        "plan_binding",
        "authority_binding",
        "review_binding",
        "source_commit",
        "root_creation_consumes_attempt",
        "reservation_records_consumed_attempt",
        "retry_or_resume_allowed",
        "reserved_at",
        "worker_capability_sha256",
    }
    if (
        set(reservation) != required
        or reservation.get("schema") != RESERVATION_SCHEMA
        or reservation.get("status") != RESERVATION_STATUS
        or reservation.get("authority_granted_by_this_document") is not False
        or reservation.get("scientific_claim_granted_by_this_document") is not False
        or reservation.get("attempt_id") != plan["attempt_id"]
        or reservation.get("output_root") != plan["output_root"]
        or reservation.get("plan_binding") != plan_binding
        or reservation.get("authority_binding") != authority_binding
        or reservation.get("review_binding") != authority["review_binding"]
        or reservation.get("source_commit") != authority["source_commit"]
        or reservation.get("root_creation_consumes_attempt") is not True
        or reservation.get("reservation_records_consumed_attempt") is not True
        or reservation.get("retry_or_resume_allowed") is not False
        or _SHA.fullmatch(str(reservation.get("worker_capability_sha256") or ""))
        is None
        or (
            capability is not None
            and reservation.get("worker_capability_sha256")
            != hashlib.sha256(capability.encode("utf-8")).hexdigest()
        )
    ):
        raise VisualDomainParitySupervisionError("parity reservation changed")
    try:
        reserved_at = datetime.fromisoformat(
            str(reservation["reserved_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise VisualDomainParitySupervisionError(
            "parity reservation time changed"
        ) from exc
    if reserved_at.tzinfo is None:
        raise VisualDomainParitySupervisionError(
            "parity reservation time lacks timezone"
        )
    return actual


def _assert_worker_inventory(plan: Mapping[str, Any], *, scene_index: int) -> None:
    root = _canonical_directory(
        Path(str(plan["output_root"])), label="parity attempt root"
    )
    if sorted(path.name for path in root.iterdir()) != ["reservation.json", "scenes"]:
        raise VisualDomainParitySupervisionError("attempt root inventory changed")
    _nofollow_regular(root / "reservation.json", label="parity reservation")
    scenes_root = _canonical_directory(root / "scenes", label="parity scenes root")
    expected_prior = {
        _expected_scene_dir(plan, index).name for index in range(scene_index)
    }
    observed = {path.name for path in scenes_root.iterdir()}
    if observed != expected_prior:
        raise VisualDomainParitySupervisionError("worker scene inventory changed")
    for index in range(scene_index):
        _canonical_directory(
            _expected_scene_dir(plan, index), label=f"prior parity scene {index}"
        )


def _render_scene_worker(
    *,
    scene_index: int,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    review: Mapping[str, Any],
    review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if type(scene_index) is not int or not 0 <= scene_index < plan_builder.EXPECTED_SCENES:
        raise VisualDomainParitySupervisionError("worker scene index changed")
    _assert_worker_inventory(plan, scene_index=scene_index)
    for prior_index in range(scene_index):
        _validate_scene_result(
            plan=plan,
            plan_binding=plan_binding,
            authority_binding=authority_binding,
            scene_index=prior_index,
        )
    scene_plan = plan["scenes"][scene_index]
    scene_dir = _expected_scene_dir(plan, scene_index)
    scene_dir.mkdir(mode=0o750, parents=False, exist_ok=False)
    rows_dir = scene_dir / "rows"
    rows_dir.mkdir(mode=0o750, exist_ok=False)
    _rehash_chain(
        plan=plan,
        plan_binding=plan_binding,
        authority=authority,
        authority_binding=authority_binding,
        review=review,
        review_binding=review_binding,
        require_fresh_output=False,
    )
    manifest, actual_manifest = _read_binding_document(
        scene_plan["scene_manifest_binding"],
        label=f"parity scene {scene_plan['scene_id']} manifest",
    )
    if actual_manifest != scene_plan["scene_manifest_binding"]:
        raise VisualDomainParitySupervisionError("scene manifest binding changed")

    from lewm_genesis.render_replay import _camera_pose_from_payload
    from lewm_genesis.scene_builder import _import_genesis
    from lewm_genesis.scene_loader import load_platform_manifest, load_scene_pack
    from lewm_genesis.textures import available_textures, select_scene_textures
    from scripts.render_replay_v03 import _to_hwc_uint8, build_scene

    platform = load_platform_manifest(plan["runtime_bindings"]["platform_manifest"]["path"])
    pack = load_scene_pack(
        Path(str(scene_plan["scene_manifest_binding"]["path"])).parent,
        platform_manifest=platform,
        workspace_root=REPO_ROOT,
    )
    if (
        pack.scene_id != scene_plan["scene_id"]
        or pack.family != scene_plan["family"]
        or tuple(pack.camera.training_resolution) != (224, 224)
        or pack.camera.fov_axis != "horizontal"
        or float(pack.camera.fov_deg) != 78.323
        or float(pack.camera.near_m) != 0.05
        or float(pack.camera.far_m) != 200.0
    ):
        raise VisualDomainParitySupervisionError("loaded scene/sensor identity changed")
    expected_texture_paths = tuple(
        str((REPO_ROOT / relative).resolve(strict=True))
        for relative in pilot.TEXTURED_V03_TEXTURE_RELATIVE_PATHS
    )
    observed_texture_paths = tuple(
        path
        for category in ("floor", "obstacle", "wall")
        for path in available_textures(category)
    )
    selected_textures = select_scene_textures(
        visual_seed=int(pack.visual_seed), scene_id=str(pack.scene_id)
    )
    if (
        observed_texture_paths != expected_texture_paths
        or {
            name: pilot.file_binding(Path(path))
            for name, path in selected_textures.items()
        }
        != scene_plan["selected_texture_asset_bindings"]
    ):
        raise VisualDomainParitySupervisionError("runtime texture selection changed")
    for binding in scene_plan["mesh_asset_bindings"]:
        if pilot.require_binding(binding, label="parity derived mesh") != binding:
            raise VisualDomainParitySupervisionError("derived mesh changed before render")
    gs = _import_genesis()
    gs.init(backend=gs.vulkan, logging_level="error")
    render_scene, render_camera = build_scene(
        gs,
        manifest,
        fov=float(pilot.TEXTURED_V03_RENDER_CONTRACT["genesis_yfov_deg"]),
        near=float(pack.camera.near_m),
        far=float(pack.camera.far_m),
        res=tuple(pilot.TEXTURED_V03_RENDER_CONTRACT["native_resolution"]),
        textures=True,
    )
    visible_ids = {
        str(item["object_id"])
        for field in ("walls", "obstacles", "landmarks")
        for item in manifest.get(field, [])
    }
    render_build = runtime_kernel._HistoricalTexturedV03RenderBuild(  # noqa: SLF001
        scene=render_scene,
        camera=render_camera,
        pack=pack,
        visible_objects=tuple(
            item for item in pack.static_objects if item.object_id in visible_ids
        ),
        to_hwc_uint8=_to_hwc_uint8,
    )
    rendered_rows = []
    for pose in scene_plan["poses"]:
        pair_id = str(pose["pair_id"])
        pose_dir = rows_dir / f"pose_{int(pose['pose_index']):02d}"
        pose_dir.mkdir(mode=0o750, exist_ok=False)
        candidate = runtime_kernel._render_textured_v03_rgb_from_base_pose(  # noqa: SLF001
            render_build,
            base_position_xyz_m=pose["base_position_xyz_m"],
            base_quaternion_wxyz=pose["base_quaternion_wxyz"],
            historical_camera_pose_from_payload=_camera_pose_from_payload,
        )
        duplicate = runtime_kernel._render_textured_v03_rgb_from_base_pose(  # noqa: SLF001
            render_build,
            base_position_xyz_m=pose["base_position_xyz_m"],
            base_quaternion_wxyz=pose["base_quaternion_wxyz"],
            historical_camera_pose_from_payload=_camera_pose_from_payload,
        )
        candidate_pose = _pose(candidate["camera_pose"])
        duplicate_pose = _pose(duplicate["camera_pose"])
        if (
            candidate_pose != pose["historical_camera_pose_world"]
            or duplicate_pose != pose["historical_camera_pose_world"]
        ):
            raise VisualDomainParitySupervisionError(
                f"computed camera pose changed for {pair_id}"
            )
        candidate_rgb = np.asarray(candidate["rgb"], dtype=np.uint8)
        duplicate_rgb = np.asarray(duplicate["rgb"], dtype=np.uint8)
        candidate_binding = _write_png_exclusive(
            pose_dir / "candidate.png", candidate_rgb
        )
        duplicate_binding = _write_png_exclusive(
            pose_dir / "duplicate.png", duplicate_rgb
        )
        candidate_raw = _raw_rgb_sha256(candidate_rgb)
        duplicate_raw = _raw_rgb_sha256(duplicate_rgb)
        common_receipt = {
            "schema": evaluator.RENDER_RECEIPT_SCHEMA,
            "status": evaluator.RENDER_RECEIPT_STATUS,
            "authority_granted_by_this_document": False,
            "scientific_claim_granted_by_this_document": False,
            "development_only": True,
            "protected_material_opened": False,
            "attempt_id": plan["attempt_id"],
            "pair_id": pair_id,
            "scene_id": scene_plan["scene_id"],
            "family": scene_plan["family"],
            "pose_index": pose["pose_index"],
            "base_position_xyz_m": pose["base_position_xyz_m"],
            "base_quaternion_wxyz": pose["base_quaternion_wxyz"],
            "historical_camera_pose_world": pose["historical_camera_pose_world"],
            "computed_camera_pose_world": candidate_pose,
            "scene_manifest_binding": scene_plan["scene_manifest_binding"],
            "scene_genesis_binding": scene_plan["scene_genesis_binding"],
            "source_panel_binding": plan["source_panel_binding"],
            "plan_binding": plan_binding,
            "authority_binding": authority_binding,
            "source_commit": authority["source_commit"],
            "source_bindings": authority["source_bindings"],
            "render_contract": plan["render_contract"],
            "runtime_bindings": plan["runtime_bindings"],
            "execution_contract": plan["execution_contract"],
            "producer_source_binding": _source_binding(authority, name="collector"),
            "renderer_source_binding": _source_binding(
                authority, name="historical_textured_v03_renderer"
            ),
            "camera_pose_helper_source_binding": _source_binding(
                authority, name="genesis_render_replay"
            ),
            "texture_source_binding": _source_binding(authority, name="textures"),
            "selected_texture_asset_bindings": scene_plan[
                "selected_texture_asset_bindings"
            ],
            "mesh_asset_bindings": scene_plan["mesh_asset_bindings"],
            "rgb_render_call": {"rgb": True, "depth": False},
            "physics_steps": 0,
        }
        receipt_rows = {}
        for ordinal, rgb_binding, raw_sha, wall_seconds, computed_pose in (
            (
                "candidate",
                candidate_binding,
                candidate_raw,
                float(candidate["rgb_render_wall_seconds"]),
                candidate_pose,
            ),
            (
                "duplicate",
                duplicate_binding,
                duplicate_raw,
                float(duplicate["rgb_render_wall_seconds"]),
                duplicate_pose,
            ),
        ):
            identity = f"{pair_id}:{ordinal}"
            receipt = {
                **common_receipt,
                "render_ordinal": ordinal,
                "computed_camera_pose_world": computed_pose,
                "producer_frame_identity": identity,
                "rgb_binding": rgb_binding,
                "raw_rgb_sha256": raw_sha,
                "rgb_render_wall_seconds": wall_seconds,
            }
            receipt_binding = pilot.write_json_exclusive(
                pose_dir / f"{ordinal}_receipt.json", receipt
            )
            receipt_rows[ordinal] = {
                "identity": identity,
                "rgb_binding": rgb_binding,
                "raw_rgb_sha256": raw_sha,
                "receipt_binding": receipt_binding,
            }
        rendered_rows.append({
            "pair_id": pair_id,
            "scene_id": scene_plan["scene_id"],
            "family": scene_plan["family"],
            "pose_index": pose["pose_index"],
            "base_position_xyz_m": pose["base_position_xyz_m"],
            "base_quaternion_wxyz": pose["base_quaternion_wxyz"],
            "camera_pose_world": pose["historical_camera_pose_world"],
            "scene_manifest_binding": scene_plan["scene_manifest_binding"],
            "scene_genesis_binding": scene_plan["scene_genesis_binding"],
            "selected_texture_asset_bindings": scene_plan[
                "selected_texture_asset_bindings"
            ],
            "mesh_asset_bindings": scene_plan["mesh_asset_bindings"],
            "candidate_producer_frame_identity": receipt_rows["candidate"]["identity"],
            "duplicate_producer_frame_identity": receipt_rows["duplicate"]["identity"],
            "candidate_rgb_binding": receipt_rows["candidate"]["rgb_binding"],
            "duplicate_rgb_binding": receipt_rows["duplicate"]["rgb_binding"],
            "candidate_raw_rgb_sha256": receipt_rows["candidate"]["raw_rgb_sha256"],
            "duplicate_raw_rgb_sha256": receipt_rows["duplicate"]["raw_rgb_sha256"],
            "candidate_render_receipt_binding": receipt_rows["candidate"][
                "receipt_binding"
            ],
            "duplicate_render_receipt_binding": receipt_rows["duplicate"][
                "receipt_binding"
            ],
        })
    _rehash_chain(
        plan=plan,
        plan_binding=plan_binding,
        authority=authority,
        authority_binding=authority_binding,
        review=review,
        review_binding=review_binding,
        require_fresh_output=False,
    )
    scene_result = {
        "schema": SCENE_RESULT_SCHEMA,
        "status": SCENE_RESULT_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "attempt_id": plan["attempt_id"],
        "scene_index": scene_index,
        "scene_id": scene_plan["scene_id"],
        "family": scene_plan["family"],
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "render_rows": rendered_rows,
        "observed_counts": {
            "poses": plan_builder.POSES_PER_SCENE,
            "candidate_rgb_frames": plan_builder.POSES_PER_SCENE,
            "duplicate_rgb_frames": plan_builder.POSES_PER_SCENE,
            "rgb_render_calls": plan_builder.POSES_PER_SCENE * 2,
            "auxiliary_depth_render_calls": 0,
            "physics_steps": 0,
        },
    }
    pilot.write_json_exclusive(scene_dir / "scene_result.json", scene_result)
    return scene_result


def _validate_scene_result(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    scene_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_path = _expected_scene_dir(plan, scene_index) / "scene_result.json"
    binding = pilot.file_binding(scene_path)
    result, actual = _read_binding_document(
        binding, label=f"parity scene {scene_index} result"
    )
    scene = plan["scenes"][scene_index]
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_granted_by_this_document",
        "attempt_id",
        "scene_index",
        "scene_id",
        "family",
        "plan_binding",
        "authority_binding",
        "render_rows",
        "observed_counts",
    }
    if (
        set(result) != required
        or result.get("schema") != SCENE_RESULT_SCHEMA
        or result.get("status") != SCENE_RESULT_STATUS
        or result.get("authority_granted_by_this_document") is not False
        or result.get("scientific_claim_granted_by_this_document") is not False
        or result.get("attempt_id") != plan["attempt_id"]
        or result.get("scene_index") != scene_index
        or result.get("scene_id") != scene["scene_id"]
        or result.get("family") != scene["family"]
        or result.get("plan_binding") != plan_binding
        or result.get("authority_binding") != authority_binding
        or result.get("observed_counts")
        != {
            "poses": plan_builder.POSES_PER_SCENE,
            "candidate_rgb_frames": plan_builder.POSES_PER_SCENE,
            "duplicate_rgb_frames": plan_builder.POSES_PER_SCENE,
            "rgb_render_calls": plan_builder.POSES_PER_SCENE * 2,
            "auxiliary_depth_render_calls": 0,
            "physics_steps": 0,
        }
        or not isinstance(result.get("render_rows"), list)
        or len(result["render_rows"]) != plan_builder.POSES_PER_SCENE
        or [row.get("pair_id") for row in result["render_rows"]]
        != [pose["pair_id"] for pose in scene["poses"]]
    ):
        raise VisualDomainParitySupervisionError("parity scene result changed")
    return result, actual


def _validate_completed_inventory(
    plan: Mapping[str, Any], *, allow_terminal: bool = False
) -> int:
    root = _canonical_directory(
        Path(str(plan["output_root"])), label="completed parity attempt root"
    )
    expected_top = {
        "reservation.json",
        "scenes",
        "generation_receipt.json",
        "candidate_panel.json",
        "parity_result.json",
    }
    if allow_terminal:
        expected_top.add("terminal.json")
    if {path.name for path in root.iterdir()} != expected_top:
        raise VisualDomainParitySupervisionError("completed attempt inventory changed")
    files = [
        _nofollow_regular(root / name, label=f"completed parity {name}")
        for name in expected_top - {"scenes", "terminal.json"}
    ]
    if allow_terminal:
        _nofollow_regular(root / "terminal.json", label="completed parity terminal")
    scenes_root = _canonical_directory(root / "scenes", label="completed scenes root")
    expected_scenes = {
        _expected_scene_dir(plan, index).name
        for index in range(plan_builder.EXPECTED_SCENES)
    }
    if {path.name for path in scenes_root.iterdir()} != expected_scenes:
        raise VisualDomainParitySupervisionError("completed scene inventory changed")
    for scene_index, scene in enumerate(plan["scenes"]):
        scene_root = _canonical_directory(
            _expected_scene_dir(plan, scene_index),
            label=f"completed scene {scene_index}",
        )
        if {path.name for path in scene_root.iterdir()} != {"rows", "scene_result.json"}:
            raise VisualDomainParitySupervisionError("completed scene files changed")
        files.append(
            _nofollow_regular(
                scene_root / "scene_result.json",
                label=f"completed scene {scene_index} result",
            )
        )
        rows_root = _canonical_directory(
            scene_root / "rows", label=f"completed scene {scene_index} rows"
        )
        expected_poses = {
            f"pose_{int(pose['pose_index']):02d}" for pose in scene["poses"]
        }
        if {path.name for path in rows_root.iterdir()} != expected_poses:
            raise VisualDomainParitySupervisionError("completed pose inventory changed")
        for pose in scene["poses"]:
            pose_index = int(pose["pose_index"])
            pose_root = _canonical_directory(
                rows_root / f"pose_{pose_index:02d}",
                label=f"completed scene {scene_index} pose {pose_index}",
            )
            expected_files = {
                "candidate.png",
                "duplicate.png",
                "candidate_receipt.json",
                "duplicate_receipt.json",
            }
            if {path.name for path in pose_root.iterdir()} != expected_files:
                raise VisualDomainParitySupervisionError("completed render inventory changed")
            files.extend(
                _nofollow_regular(
                    pose_root / name,
                    label=f"completed scene {scene_index} pose {pose_index} {name}",
                )
                for name in expected_files
            )
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > authority_builder.MAX_PARITY_OUTPUT_BYTES:
        raise VisualDomainParitySupervisionError("parity total-output hard cap exceeded")
    return total_bytes


def _require_parity_pass_result(result: Mapping[str, Any]) -> None:
    measurements = result.get("measurements")
    if (
        result.get("schema") != evaluator.RESULT_SCHEMA
        or result.get("status") != evaluator.PASS_STATUS
        or result.get("authority_granted_by_this_document") is not False
        or result.get("scientific_claim_granted_by_this_document") is not False
        or result.get("development_only") is not True
        or result.get("protected_material_opened") is not False
        or not isinstance(measurements, Mapping)
        or not evaluator._passes(measurements)  # noqa: SLF001
    ):
        raise VisualDomainParitySupervisionError(
            "visual-domain parity evaluator did not pass exactly"
        )


def _terminal_revalidate(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    review: Mapping[str, Any],
    review_binding: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    result_binding: Mapping[str, Any],
    scene_result_bindings: Sequence[Mapping[str, Any]],
    allow_terminal: bool = False,
) -> tuple[dict[str, Any], int]:
    _rehash_chain(
        plan=plan,
        plan_binding=plan_binding,
        authority=authority,
        authority_binding=authority_binding,
        review=review,
        review_binding=review_binding,
        require_fresh_output=False,
    )
    candidate, actual_candidate = _read_binding_document(
        candidate_binding, label="terminal candidate panel"
    )
    result, actual_result = _read_binding_document(
        result_binding, label="terminal parity result"
    )
    source, source_binding = _read_binding_document(
        plan["source_panel_binding"], label="terminal historical source panel"
    )
    if actual_candidate != candidate_binding or actual_result != result_binding:
        raise VisualDomainParitySupervisionError("terminal panel/result binding changed")
    recomputed = evaluator.evaluate_v1(
        source_panel=source,
        source_panel_binding=source_binding,
        candidate_panel=candidate,
        candidate_panel_binding=actual_candidate,
    )
    if pilot.canonical_json_bytes(recomputed) != pilot.canonical_json_bytes(result):
        raise VisualDomainParitySupervisionError("terminal parity recomputation changed")
    _require_parity_pass_result(result)
    generation_binding = candidate["producer_lineage"]["generation_receipt_binding"]
    generation, actual_generation = _read_binding_document(
        generation_binding, label="terminal generation receipt"
    )
    combined_rows = []
    actual_scene_bindings = []
    if len(scene_result_bindings) != plan_builder.EXPECTED_SCENES:
        raise VisualDomainParitySupervisionError("terminal scene result count changed")
    for scene_index, declared in enumerate(scene_result_bindings):
        scene_result, actual_scene = _validate_scene_result(
            plan=plan,
            plan_binding=plan_binding,
            authority_binding=authority_binding,
            scene_index=scene_index,
        )
        if actual_scene != declared:
            raise VisualDomainParitySupervisionError("terminal scene result changed")
        combined_rows.extend(scene_result["render_rows"])
        actual_scene_bindings.append(actual_scene)
    combined_rows.sort(key=lambda row: row["pair_id"])
    if generation.get("render_rows") != combined_rows:
        raise VisualDomainParitySupervisionError("terminal generation rows changed")
    total_bytes = _validate_completed_inventory(
        plan, allow_terminal=allow_terminal
    )
    return {
        "candidate_panel_binding": actual_candidate,
        "parity_result_binding": actual_result,
        "generation_receipt_binding": actual_generation,
        "scene_result_bindings": actual_scene_bindings,
    }, total_bytes


def _require_aware_iso8601(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise VisualDomainParitySupervisionError(f"{label} is empty")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VisualDomainParitySupervisionError(
            f"{label} is not ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise VisualDomainParitySupervisionError(f"{label} lacks timezone")


def _validate_recorded_preflights(
    *,
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> None:
    graphics = terminal.get("graphics_preflight")
    graphics_fields = {
        "phase",
        "status",
        "environment",
        "expectation",
        "vulkan_stdout_sha256",
        "egl_stdout_sha256",
        "egl_stderr_sha256",
        "egl_exit_code",
    }
    if (
        not isinstance(graphics, Mapping)
        or set(graphics) != graphics_fields
        or graphics.get("phase") != "graphics_preflight"
        or graphics.get("status") != "PASS"
        or graphics.get("environment")
        != plan["execution_contract"]["environment"]
        or graphics.get("expectation")
        != plan["execution_contract"]["graphics_preflight"]
        or type(graphics.get("egl_exit_code")) is not int
        or graphics.get("egl_exit_code")
        != plan["execution_contract"]["graphics_preflight"][
            "eglinfo_expected_exit_code"
        ]
        or any(
            _SHA.fullmatch(str(graphics.get(name) or "")) is None
            for name in (
                "vulkan_stdout_sha256",
                "egl_stdout_sha256",
                "egl_stderr_sha256",
            )
        )
    ):
        raise VisualDomainParitySupervisionError(
            "recorded parity graphics preflight changed"
        )

    disk = terminal.get("disk_preflight")
    disk_fields = {
        "filesystem_path",
        "available_bytes_before_reservation",
        "projected_pipeline_new_bytes",
        "free_space_margin_bytes",
        "required_preflight_free_bytes",
        "maximum_parity_output_bytes",
        "passed",
    }
    caps = authority["caps"]
    expected_parent = str(Path(str(plan["output_root"])).parent.resolve(strict=True))
    if (
        not isinstance(disk, Mapping)
        or set(disk) != disk_fields
        or disk.get("filesystem_path") != expected_parent
        or disk.get("passed") is not True
        or any(
            type(disk.get(name)) is not int or int(disk[name]) < 0
            for name in disk_fields - {"filesystem_path", "passed"}
        )
        or disk.get("projected_pipeline_new_bytes")
        != caps["projected_pipeline_new_bytes"]
        or disk.get("free_space_margin_bytes") != caps["free_space_margin_bytes"]
        or disk.get("required_preflight_free_bytes")
        != caps["required_preflight_free_bytes"]
        or disk.get("maximum_parity_output_bytes")
        != caps["maximum_parity_output_bytes"]
        or disk.get("available_bytes_before_reservation")
        < disk.get("required_preflight_free_bytes")
    ):
        raise VisualDomainParitySupervisionError(
            "recorded parity disk preflight changed"
        )


def validate_success_terminal_v1(
    *,
    terminal_binding: Mapping[str, Any],
    expected_result_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen and fully revalidate one completed parity attempt.

    This is intentionally stronger than validating the terminal's top-level
    fields.  It reopens the committed plan, authority, source review,
    reservation, generation receipt, every scene/leaf receipt, both RGB
    panels, and the parity result; recomputes the evaluator; and requires the
    exact post-terminal inventory while excluding ``terminal.json`` from the
    recorded pre-terminal byte count.
    """

    terminal, actual_terminal = _read_binding_document(
        terminal_binding, label="visual-domain parity success terminal"
    )
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_granted_by_this_document",
        "authorizes_retry_or_resume",
        "root_creation_consumes_attempt",
        "reservation_records_consumed_attempt",
        "attempt_id",
        "plan_binding",
        "authority_binding",
        "reservation_binding",
        "source_review_binding",
        "source_commit",
        "scene_result_bindings",
        "generation_receipt_binding",
        "candidate_panel_binding",
        "parity_result_binding",
        "graphics_preflight",
        "disk_preflight",
        "wall_seconds",
        "wall_ceiling_seconds",
        "total_output_bytes_before_terminal",
        "completed_at",
        "terminal_reviewer",
    }
    if (
        set(terminal) != required
        or terminal.get("schema") != TERMINAL_SCHEMA
        or terminal.get("status") != TERMINAL_SUCCESS_STATUS
        or terminal.get("authority_granted_by_this_document") is not False
        or terminal.get("scientific_claim_granted_by_this_document") is not False
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("root_creation_consumes_attempt") is not True
        or terminal.get("reservation_records_consumed_attempt") is not True
        or not isinstance(terminal.get("attempt_id"), str)
        or not terminal["attempt_id"]
        or not isinstance(terminal.get("scene_result_bindings"), list)
        or len(terminal["scene_result_bindings"]) != plan_builder.EXPECTED_SCENES
        or not isinstance(terminal.get("terminal_reviewer"), str)
        or not terminal["terminal_reviewer"].strip()
        or not isinstance(terminal.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", terminal["source_commit"]) is None
    ):
        raise VisualDomainParitySupervisionError(
            "visual-domain parity success terminal changed"
        )
    for name in ("wall_seconds", "wall_ceiling_seconds"):
        value = terminal[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise VisualDomainParitySupervisionError(
                "visual-domain parity terminal wall changed"
            )
    if float(terminal["wall_seconds"]) > float(terminal["wall_ceiling_seconds"]):
        raise VisualDomainParitySupervisionError(
            "visual-domain parity terminal exceeded wall ceiling"
        )
    if (
        type(terminal["total_output_bytes_before_terminal"]) is not int
        or terminal["total_output_bytes_before_terminal"] < 0
    ):
        raise VisualDomainParitySupervisionError(
            "visual-domain parity terminal byte count changed"
        )
    _require_aware_iso8601(terminal["completed_at"], label="parity completion time")

    try:
        plan_binding = pilot.require_binding(
            terminal["plan_binding"], label="terminal parity plan"
        )
        authority_binding = pilot.require_binding(
            terminal["authority_binding"], label="terminal parity authority"
        )
        result_binding = pilot.require_binding(
            terminal["parity_result_binding"], label="terminal parity result"
        )
    except pilot.PilotContractError as exc:
        raise VisualDomainParitySupervisionError(str(exc)) from exc
    if expected_result_binding is not None:
        try:
            expected_result = pilot.require_binding(
                expected_result_binding, label="expected terminal parity result"
            )
        except pilot.PilotContractError as exc:
            raise VisualDomainParitySupervisionError(str(exc)) from exc
        if result_binding != expected_result:
            raise VisualDomainParitySupervisionError(
                "terminal parity result differs from required result"
            )

    (
        plan,
        actual_plan_binding,
        authority,
        actual_authority_binding,
        review,
        review_binding,
    ) = load_and_validate_chain_v1(
        plan_path=Path(str(plan_binding["path"])),
        expected_plan_sha256=str(plan_binding["file_sha256"]),
        expected_plan_byte_count=int(plan_binding["byte_count"]),
        authority_path=Path(str(authority_binding["path"])),
        expected_authority_sha256=str(authority_binding["file_sha256"]),
        expected_authority_byte_count=int(authority_binding["byte_count"]),
        require_fresh_output=False,
    )
    root = Path(str(plan["output_root"])).resolve(strict=True)
    expected_paths = {
        "terminal": root / "terminal.json",
        "reservation_binding": root / "reservation.json",
        "generation_receipt_binding": root / "generation_receipt.json",
        "candidate_panel_binding": root / "candidate_panel.json",
        "parity_result_binding": root / "parity_result.json",
    }
    if (
        actual_plan_binding != plan_binding
        or actual_authority_binding != authority_binding
        or Path(str(actual_terminal["path"])).resolve(strict=True)
        != expected_paths["terminal"]
        or terminal.get("attempt_id") != plan["attempt_id"]
        or terminal.get("source_review_binding") != review_binding
        or terminal.get("source_commit") != authority["source_commit"]
        or terminal.get("terminal_reviewer")
        != authority["external_supervisor"]["terminal_reviewer"]
        or float(terminal["wall_ceiling_seconds"])
        != float(authority["caps"]["wall_seconds"])
        or terminal["total_output_bytes_before_terminal"]
        > int(authority["caps"]["maximum_parity_output_bytes"])
    ):
        raise VisualDomainParitySupervisionError(
            "visual-domain parity terminal chain changed"
        )
    for name in (
        "reservation_binding",
        "generation_receipt_binding",
        "candidate_panel_binding",
        "parity_result_binding",
    ):
        try:
            normalized = pilot.require_binding(
                terminal[name], label=f"terminal {name}"
            )
        except pilot.PilotContractError as exc:
            raise VisualDomainParitySupervisionError(str(exc)) from exc
        if Path(str(normalized["path"])).resolve(strict=True) != expected_paths[name]:
            raise VisualDomainParitySupervisionError(
                f"terminal {name} escaped the attempt root"
            )

    reservation = _validate_reservation(
        binding=terminal["reservation_binding"],
        plan=plan,
        plan_binding=plan_binding,
        authority=authority,
        authority_binding=authority_binding,
        capability=None,
    )
    if reservation != terminal["reservation_binding"]:
        raise VisualDomainParitySupervisionError(
            "visual-domain parity reservation binding changed"
        )
    recomputed, total_bytes = _terminal_revalidate(
        plan=plan,
        plan_binding=plan_binding,
        authority=authority,
        authority_binding=authority_binding,
        review=review,
        review_binding=review_binding,
        candidate_binding=terminal["candidate_panel_binding"],
        result_binding=result_binding,
        scene_result_bindings=terminal["scene_result_bindings"],
        allow_terminal=True,
    )
    if (
        recomputed["candidate_panel_binding"]
        != terminal["candidate_panel_binding"]
        or recomputed["parity_result_binding"] != result_binding
        or recomputed["generation_receipt_binding"]
        != terminal["generation_receipt_binding"]
        or recomputed["scene_result_bindings"]
        != terminal["scene_result_bindings"]
        or total_bytes != terminal["total_output_bytes_before_terminal"]
    ):
        raise VisualDomainParitySupervisionError(
            "visual-domain parity terminal recomputation changed"
        )
    validated_result, validated_result_binding = _read_binding_document(
        result_binding, label="deeply validated parity result"
    )
    if validated_result_binding != result_binding:
        raise VisualDomainParitySupervisionError(
            "deeply validated parity result binding changed"
        )
    _require_parity_pass_result(validated_result)
    _validate_recorded_preflights(
        plan=plan, authority=authority, terminal=terminal
    )
    return {
        "terminal": terminal,
        "terminal_binding": actual_terminal,
        "result_binding": result_binding,
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "source_review_binding": review_binding,
    }


def _generation_documents(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    review_binding: Mapping[str, Any],
    render_rows: list[dict[str, Any]],
    wall_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_textures = {
        scene["scene_id"]: scene["selected_texture_asset_bindings"]
        for scene in plan["scenes"]
    }
    mesh_map = {
        scene["scene_id"]: scene["mesh_asset_bindings"] for scene in plan["scenes"]
    }
    stored_bytes = sum(
        int(row[name]["byte_count"])
        for row in render_rows
        for name in ("candidate_rgb_binding", "duplicate_rgb_binding")
    )
    if stored_bytes > authority_builder.MAX_STORED_RGB_BYTES:
        raise VisualDomainParitySupervisionError("stored RGB hard cap exceeded")
    generation = {
        "schema": evaluator.CANDIDATE_GENERATION_RECEIPT_SCHEMA,
        "status": evaluator.CANDIDATE_GENERATION_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "development_only": True,
        "protected_material_opened": False,
        "attempt_id": plan["attempt_id"],
        "output_root": plan["output_root"],
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "source_review_binding": review_binding,
        "source_commit": authority["source_commit"],
        "source_panel_binding": plan["source_panel_binding"],
        "render_contract": plan["render_contract"],
        "comparison_contract": plan["comparison_contract"],
        "expected_counts": plan["expected_counts"],
        "runtime_bindings": plan["runtime_bindings"],
        "execution_contract": plan["execution_contract"],
        "scene_corpus_manifest_bindings": plan["scene_corpus_manifest_bindings"],
        "texture_asset_bindings": plan["texture_asset_bindings"],
        "mesh_asset_bindings": plan["mesh_asset_bindings"],
        "selected_texture_asset_bindings_by_scene": selected_textures,
        "mesh_asset_bindings_by_scene": mesh_map,
        "producer_source_binding": _source_binding(authority, name="collector"),
        "renderer_source_binding": _source_binding(
            authority, name="historical_textured_v03_renderer"
        ),
        "camera_pose_helper_source_binding": _source_binding(
            authority, name="genesis_render_replay"
        ),
        "texture_source_binding": _source_binding(authority, name="textures"),
        "source_bindings": authority["source_bindings"],
        "render_rows": render_rows,
        "observed_counts": {
            "scenes": plan_builder.EXPECTED_SCENES,
            "poses": plan_builder.EXPECTED_POSES,
            "candidate_rgb_frames": plan_builder.EXPECTED_POSES,
            "duplicate_rgb_frames": plan_builder.EXPECTED_POSES,
            "rgb_render_calls": plan_builder.EXPECTED_POSES * 2,
            "auxiliary_depth_render_calls": 0,
            "physics_steps": 0,
            "stored_rgb_bytes": stored_bytes,
        },
        "wall_seconds": float(wall_seconds),
    }
    generation_binding = pilot.write_json_exclusive(
        Path(str(plan["output_root"])) / "generation_receipt.json", generation
    )
    candidate_panel = {
        "schema": evaluator.PANEL_SCHEMA,
        "domain": evaluator.CANDIDATE_DOMAIN,
        "rgb_root": str((Path(str(plan["output_root"])) / "scenes").resolve()),
        "render_contract": plan["render_contract"],
        "producer_source_binding": generation["producer_source_binding"],
        "renderer_source_binding": generation["renderer_source_binding"],
        "texture_source_binding": generation["texture_source_binding"],
        "selected_texture_asset_bindings_by_scene": selected_textures,
        "mesh_asset_bindings_by_scene": mesh_map,
        "producer_lineage": {
            "schema": evaluator.CANDIDATE_LINEAGE_SCHEMA,
            "generation_receipt_binding": generation_binding,
        },
        "rows": [
            {
                "pair_id": row["pair_id"],
                "scene_id": row["scene_id"],
                "family": row["family"],
                "pose_index": row["pose_index"],
                "camera_pose_world": row["camera_pose_world"],
                "scene_manifest_binding": row["scene_manifest_binding"],
                "producer_frame_identity": row[
                    "candidate_producer_frame_identity"
                ],
                "duplicate_producer_frame_identity": row[
                    "duplicate_producer_frame_identity"
                ],
                "rgb_binding": row["candidate_rgb_binding"],
                "raw_rgb_sha256": row["candidate_raw_rgb_sha256"],
                "duplicate_rgb_binding": row["duplicate_rgb_binding"],
                "duplicate_raw_rgb_sha256": row["duplicate_raw_rgb_sha256"],
            }
            for row in render_rows
        ],
    }
    return candidate_panel, generation_binding


def _terminal_failure(
    *,
    root: Path,
    plan_binding: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any] | None,
    started: float,
    failure: BaseException,
) -> None:
    document = {
        "schema": TERMINAL_SCHEMA,
        "status": TERMINAL_FAILURE_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "authorizes_retry_or_resume": False,
        "plan_binding": dict(plan_binding),
        "authority_binding": dict(authority_binding),
        "reservation_binding": (
            dict(reservation_binding) if reservation_binding is not None else None
        ),
        "reservation_path": str(root / "reservation.json"),
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": reservation_binding is not None,
        "wall_seconds": float(time.monotonic() - started),
        "failed_at": _utc_now(),
        "failure": {
            "type": type(failure).__name__,
            "message": str(failure),
        },
    }
    try:
        pilot.write_json_exclusive(root / "terminal_failure.json", document)
    except Exception:
        pass


def supervise_v1(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    expected_plan_byte_count: int,
    authority_path: Path,
    expected_authority_sha256: str,
    expected_authority_byte_count: int,
) -> dict[str, Any]:
    started = time.monotonic()
    (
        plan,
        plan_binding,
        authority,
        authority_binding,
        review,
        review_binding,
    ) = load_and_validate_chain_v1(
        plan_path=plan_path,
        expected_plan_sha256=expected_plan_sha256,
        expected_plan_byte_count=expected_plan_byte_count,
        authority_path=authority_path,
        expected_authority_sha256=expected_authority_sha256,
        expected_authority_byte_count=expected_authority_byte_count,
        require_fresh_output=True,
    )
    wall_ceiling = float(authority["caps"]["wall_seconds"])
    child_env = graphics_supervisor._child_environment(plan)  # noqa: SLF001
    for name in NETWORK_ENV_NAMES:
        child_env.pop(name, None)
    python = graphics_supervisor._validate_python_invocation(plan)  # noqa: SLF001
    graphics_preflight = graphics_supervisor._run_graphics_preflight(  # noqa: SLF001
        plan,
        child_env=child_env,
        wall_started=started,
        wall_ceiling=wall_ceiling,
    )
    _rehash_chain(
        plan=plan,
        plan_binding=plan_binding,
        authority=authority,
        authority_binding=authority_binding,
        review=review,
        review_binding=review_binding,
        require_fresh_output=True,
    )
    root = _fresh_attempt_root(Path(str(plan["output_root"])))
    disk_preflight = _disk_preflight(output_parent=root.parent, authority=authority)
    os.mkdir(root, mode=0o750)
    reservation_binding = None
    try:
        (root / "scenes").mkdir(mode=0o750, exist_ok=False)
        capability = secrets.token_hex(32)
        reservation = {
            "schema": RESERVATION_SCHEMA,
            "status": RESERVATION_STATUS,
            "authority_granted_by_this_document": False,
            "scientific_claim_granted_by_this_document": False,
            "attempt_id": plan["attempt_id"],
            "output_root": plan["output_root"],
            "plan_binding": plan_binding,
            "authority_binding": authority_binding,
            "review_binding": review_binding,
            "source_commit": authority["source_commit"],
            "root_creation_consumes_attempt": True,
            "reservation_records_consumed_attempt": True,
            "retry_or_resume_allowed": False,
            "reserved_at": _utc_now(),
            "worker_capability_sha256": hashlib.sha256(
                capability.encode("utf-8")
            ).hexdigest(),
        }
        reservation_binding = pilot.write_json_exclusive(
            root / "reservation.json", reservation
        )
        child_env[CAPABILITY_ENV] = capability
        supervisor_source = _source_binding(authority, name="external_supervisor")
        source_path = Path(str(supervisor_source["path"]))
        for scene_index in range(plan_builder.EXPECTED_SCENES):
            remaining = wall_ceiling - (time.monotonic() - started)
            if remaining <= 0.0:
                raise VisualDomainParitySupervisionError("hard wall ceiling exhausted")
            command = [
                str(python),
                str(source_path),
                "worker",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                expected_plan_sha256,
                "--expected-plan-byte-count",
                str(expected_plan_byte_count),
                "--authority",
                str(authority_path),
                "--expected-authority-sha256",
                expected_authority_sha256,
                "--expected-authority-byte-count",
                str(expected_authority_byte_count),
                "--reservation",
                str(reservation_binding["path"]),
                "--expected-reservation-sha256",
                str(reservation_binding["file_sha256"]),
                "--expected-reservation-byte-count",
                str(reservation_binding["byte_count"]),
                "--scene-index",
                str(scene_index),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=remaining,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise VisualDomainParitySupervisionError(
                    f"scene {scene_index} worker exceeded hard wall"
                ) from exc
            if completed.returncode != 0:
                stderr = completed.stderr[-4000:]
                raise VisualDomainParitySupervisionError(
                    f"scene {scene_index} worker failed: {stderr}"
                )
        render_rows = []
        scene_result_bindings = []
        for scene_index in range(plan_builder.EXPECTED_SCENES):
            scene_result, scene_binding = _validate_scene_result(
                plan=plan,
                plan_binding=plan_binding,
                authority_binding=authority_binding,
                scene_index=scene_index,
            )
            render_rows.extend(scene_result["render_rows"])
            scene_result_bindings.append(scene_binding)
        render_rows.sort(key=lambda row: row["pair_id"])
        if [row["pair_id"] for row in render_rows] != sorted(
            row["pair_id"] for row in render_rows
        ):
            raise VisualDomainParitySupervisionError("render row order changed")
        _rehash_chain(
            plan=plan,
            plan_binding=plan_binding,
            authority=authority,
            authority_binding=authority_binding,
            review=review,
            review_binding=review_binding,
            require_fresh_output=False,
        )
        candidate_panel, generation_binding = _generation_documents(
            plan=plan,
            plan_binding=plan_binding,
            authority=authority,
            authority_binding=authority_binding,
            review_binding=review_binding,
            render_rows=render_rows,
            wall_seconds=time.monotonic() - started,
        )
        candidate_binding = pilot.write_json_exclusive(
            root / "candidate_panel.json", candidate_panel
        )
        source_panel, source_binding = _read_binding_document(
            plan["source_panel_binding"], label="historical parity source panel"
        )
        result = evaluator.evaluate_v1(
            source_panel=source_panel,
            source_panel_binding=source_binding,
            candidate_panel=candidate_panel,
            candidate_panel_binding=candidate_binding,
        )
        result_binding = pilot.write_json_exclusive(root / "parity_result.json", result)
        terminal_bindings, total_output_bytes = _terminal_revalidate(
            plan=plan,
            plan_binding=plan_binding,
            authority=authority,
            authority_binding=authority_binding,
            review=review,
            review_binding=review_binding,
            candidate_binding=candidate_binding,
            result_binding=result_binding,
            scene_result_bindings=scene_result_bindings,
        )
        terminal = {
            "schema": TERMINAL_SCHEMA,
            "status": TERMINAL_SUCCESS_STATUS,
            "authority_granted_by_this_document": False,
            "scientific_claim_granted_by_this_document": False,
            "authorizes_retry_or_resume": False,
            "root_creation_consumes_attempt": True,
            "reservation_records_consumed_attempt": True,
            "attempt_id": plan["attempt_id"],
            "plan_binding": plan_binding,
            "authority_binding": authority_binding,
            "reservation_binding": reservation_binding,
            "source_review_binding": review_binding,
            "source_commit": authority["source_commit"],
            "scene_result_bindings": terminal_bindings["scene_result_bindings"],
            "generation_receipt_binding": terminal_bindings[
                "generation_receipt_binding"
            ],
            "candidate_panel_binding": terminal_bindings[
                "candidate_panel_binding"
            ],
            "parity_result_binding": terminal_bindings["parity_result_binding"],
            "graphics_preflight": graphics_preflight,
            "disk_preflight": disk_preflight,
            "wall_seconds": float(time.monotonic() - started),
            "wall_ceiling_seconds": wall_ceiling,
            "total_output_bytes_before_terminal": total_output_bytes,
            "completed_at": _utc_now(),
            "terminal_reviewer": authority["external_supervisor"][
                "terminal_reviewer"
            ],
        }
        pilot.write_json_exclusive(root / "terminal.json", terminal)
        return terminal
    except BaseException as exc:
        _terminal_failure(
            root=root,
            plan_binding=plan_binding,
            authority_binding=authority_binding,
            reservation_binding=reservation_binding,
            started=started,
            failure=exc,
        )
        raise


def worker_v1(args: argparse.Namespace) -> int:
    capability = os.environ.pop(CAPABILITY_ENV, "")
    if not capability or len(capability) != 64:
        raise VisualDomainParitySupervisionError("worker capability is absent")
    (
        plan,
        plan_binding,
        authority,
        authority_binding,
        review,
        review_binding,
    ) = load_and_validate_chain_v1(
        plan_path=args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_plan_byte_count=args.expected_plan_byte_count,
        authority_path=args.authority,
        expected_authority_sha256=args.expected_authority_sha256,
        expected_authority_byte_count=args.expected_authority_byte_count,
        require_fresh_output=False,
    )
    expected_python = graphics_supervisor._validate_python_invocation(plan)  # noqa: SLF001
    if Path(sys.executable).resolve(strict=True) != expected_python.resolve(strict=True):
        raise VisualDomainParitySupervisionError("worker Python is not plan-bound")
    expected_environment = plan["execution_contract"]["environment"]
    for name in runtime_kernel._SANITIZED_SELECTOR_KEYS:  # noqa: SLF001
        if name in expected_environment:
            if os.environ.get(name) != str(expected_environment[name]):
                raise VisualDomainParitySupervisionError(
                    f"worker selector {name} is not plan-bound"
                )
        elif name in os.environ:
            raise VisualDomainParitySupervisionError(
                f"worker inherited forbidden selector {name}"
            )
    if any(name in os.environ for name in NETWORK_ENV_NAMES):
        raise VisualDomainParitySupervisionError("worker inherited network proxy state")
    reservation, reservation_binding = _read_bound_document(
        args.reservation,
        expected_sha256=args.expected_reservation_sha256,
        expected_byte_count=args.expected_reservation_byte_count,
        label="parity reservation",
    )
    actual_reservation = _validate_reservation(
        binding=reservation_binding,
        plan=plan,
        plan_binding=plan_binding,
        authority=authority,
        authority_binding=authority_binding,
        capability=capability,
    )
    if actual_reservation != reservation_binding or not reservation:
        raise VisualDomainParitySupervisionError("worker reservation changed")
    _render_scene_worker(
        scene_index=args.scene_index,
        plan=plan,
        plan_binding=plan_binding,
        authority=authority,
        authority_binding=authority_binding,
        review=review,
        review_binding=review_binding,
    )
    return 0


def _add_bound_chain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-plan-byte-count", required=True, type=int)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", required=True, type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    supervise = subparsers.add_parser("supervise")
    _add_bound_chain_arguments(supervise)
    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    _add_bound_chain_arguments(worker)
    worker.add_argument("--reservation", required=True, type=Path)
    worker.add_argument("--expected-reservation-sha256", required=True)
    worker.add_argument("--expected-reservation-byte-count", required=True, type=int)
    worker.add_argument("--scene-index", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        return worker_v1(args)
    terminal = supervise_v1(
        plan_path=args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_plan_byte_count=args.expected_plan_byte_count,
        authority_path=args.authority,
        expected_authority_sha256=args.expected_authority_sha256,
        expected_authority_byte_count=args.expected_authority_byte_count,
    )
    print(json.dumps({"terminal": terminal}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RESERVATION_SCHEMA",
    "SCENE_RESULT_SCHEMA",
    "TERMINAL_SCHEMA",
    "VisualDomainParitySupervisionError",
    "load_and_validate_chain_v1",
    "supervise_v1",
    "validate_success_terminal_v1",
]
