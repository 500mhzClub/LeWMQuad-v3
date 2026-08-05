#!/usr/bin/env python3
"""Audit perfect camera-ray fields against the exact 320-frame train fit panel.

``--dry-run`` uses one synthetic train scene and opens no generated fit data.
``--run-exact-fit`` reuses the frozen N32 audit's allowlisted fit-only source
loader, but all ray construction and rasterization are performed by the new
independent NumPy implementation.  Neither mode reads RGB, checkpoints, G2,
holdout, runtime, or sealed payloads, and neither mode grants a model license.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LEWM_WORLDS = ROOT / "lewm_worlds"
if str(LEWM_WORLDS) not in sys.path:
    sys.path.insert(0, str(LEWM_WORLDS))

OUTPUT_PATH = (
    ROOT / ".generated/go2_perfect_camera_ray_field_fit_audit/v1/result.json"
)
SCRIPT_PATH = Path(__file__).resolve()
CORE_PATH = (
    ROOT / "lewm/benchmarks/go2_perfect_camera_ray_field_audit.py"
).resolve()


def _load_independent_ray_audit() -> Any:
    """Load the new NumPy audit without importing the protected lewm graph."""

    name = "go2_perfect_camera_ray_field_audit_independent"
    spec = importlib.util.spec_from_file_location(name, CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load independent perfect-ray audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ray_audit = _load_independent_ray_audit()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **core,
        "content_sha256": hashlib.sha256(_canonical_json_bytes(core)).hexdigest(),
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.resolve()
    try:
        destination.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PermissionError("audit output must remain inside the repository") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_bytes(payload) + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable audit output already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _synthetic_manifest() -> tuple[Any, tuple[Any, ...]]:
    from lewm_worlds.manifest import (
        BoxObject,
        CameraValidityConstraints,
        SceneManifest,
        SpawnSpec,
    )

    obstacle = BoxObject(
        object_id="dry_run_obstacle",
        kind="obstacle",
        center_xyz_m=(2.0, 0.1, 0.5),
        size_xyz_m=(0.45, 0.8, 1.0),
        yaw_rad=0.21,
        material_id="wall",
    )
    manifest = SceneManifest(
        scene_id="perfect_ray_dry_run",
        family="open_obstacle_field",
        difficulty_tier="unit_test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-5.0, -5.0), (7.0, 5.0)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(obstacle,),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split="train",
    )
    return manifest, (obstacle,)


def run_dry_run() -> dict[str, Any]:
    from lewm.datasets import go2_paired_navigation as labels_v3
    from lewm_worlds.planning_grid import InflatedOccupancyGrid

    manifest, boxes = _synthetic_manifest()
    physical_grid = InflatedOccupancyGrid(
        manifest,
        cell_size_m=0.05,
        inflation_m=0.0,
        treat_landmarks_as_obstacles=True,
        treat_distractors_as_obstacles=True,
    )
    camera = labels_v3.CameraObservation(
        position_xyz_m=(0.326, 0.0, 0.393),
        lookat_xyz_m=(1.326, 0.0, 0.393),
        up_xyz=(0.0, 0.0, 1.0),
        horizontal_fov_deg=78.323,
        vertical_fov_deg=62.8370386364,
        near_m=0.05,
        image_width_px=224,
        image_height_px=168,
        obstacle_ray_stride_px=2,
    )
    expected, supervision, _observed = (
        labels_v3._observable_physical_raster_and_output_labels(
            physical_grid,
            rendered_obstacle_boxes=boxes,
            collision_obstacle_boxes=boxes,
            base_xy_yaw=(0.0, 0.0, 0.17),
            camera=camera,
            local_grid=labels_v3.DEFAULT_LOCAL_GRID,
        )
    )
    kwargs = {
        "camera": ray_audit.CameraRaySpec.from_camera_observation(camera),
        "rendered_obstacle_boxes": boxes,
        "collision_obstacle_boxes": boxes,
        "base_xy_yaw": (0.0, 0.0, 0.17),
        "physical_free_mask": physical_grid.free_mask,
        "physical_origin_xy_m": physical_grid.origin_xy,
        "physical_cell_size_m": physical_grid.cell_size_m,
    }
    first = ray_audit.reconstruct_frame_from_perfect_rays(**kwargs)
    second = ray_audit.reconstruct_frame_from_perfect_rays(**kwargs)
    report = ray_audit.audit_frame_labels(
        authoritative_labels=expected,
        supervision_mask=supervision,
        reconstruction=first,
        frame_key={"dataset_role": "synthetic_train", "frame_index": 0},
    )
    return {
        "schema": "lewm_go2_perfect_camera_ray_field_dry_run_v1",
        "dry_run": True,
        "gpu_required": False,
        "generated_fit_payload_opened": False,
        "g2_or_holdout_payload_opened": False,
        "contract_parity": report["contract_mismatch_cell_count"] == 0,
        "contract_mismatch_cell_count": report["contract_mismatch_cell_count"],
        "ray_only_mismatch_cell_count": report["ray_only_mismatch_cell_count"],
        "deterministic": bool(
            first.field_sha256 == second.field_sha256
            and np.array_equal(first.contract_labels, second.contract_labels)
            and np.array_equal(first.ray_only_labels, second.ray_only_labels)
        ),
        "field_sha256": first.field_sha256,
        "contract_labels_sha256": report["contract_labels_sha256"],
        "ray_only_labels_sha256": report["ray_only_labels_sha256"],
        "ordinary_pixel_depth_sufficiency_proved": False,
        "reason": (
            "the exact field includes off-pixel five-point ground-support queries "
            "in addition to the registered pixel first-hit lattice"
        ),
    }


def _load_exact_fit_materials(machine_manifest_sha256: str) -> tuple[Any, ...]:
    """Reuse the frozen fit-only reader; return only in-memory train materials."""

    from scripts import audit_go2_n32_camera_frustum_observability as source_access

    ledger = source_access.new_access_ledger()
    spec = source_access.AuditSpec()
    source_hashes = source_access._source_hashes(spec.sources(), ledger=ledger)
    if source_hashes["binding"]["sha256"] != source_access.EXECUTION_BINDING_SHA256:
        raise ValueError("frozen fit-only source binding changed")
    machine_manifest = source_access._load_machine_manifest(
        machine_manifest_sha256,
        source_hashes=source_hashes,
        ledger=ledger,
    )
    source_access._load_authorized_semantics(source_hashes)
    records, panel_metadata = source_access._load_panel(spec, ledger)
    if len(records) != 320 or any(str(row.get("dataset_role", "train")) != "train" for row in records):
        raise PermissionError("selected panel is not exactly 320 physical-train frames")
    shard_entries, grouped_shards = source_access._label_shard_manifest(
        records, spec=spec, ledger=ledger
    )
    authorized_inputs = machine_manifest["authorized_inputs"]
    if authorized_inputs["label_shards"] != source_access._canonical_manifest(
        shard_entries
    ):
        raise ValueError("fit label shards differ from the authorized inventory")
    selected_labels = source_access._read_selected_labels_once(
        grouped_shards, ledger=ledger
    )
    source_frames, scenes, geometry, source_entries = (
        source_access._read_source_geometry(
            records,
            panel_metadata,
            spec=spec,
            ledger=ledger,
            authorized_source_entries=authorized_inputs["source_geometry"]["entries"],
        )
    )
    if authorized_inputs["source_geometry"] != source_access._canonical_manifest(
        source_entries
    ):
        raise ValueError("fit source geometry differs from the authorized inventory")
    if any(int(ledger[name]) != 0 for name in source_access.FORBIDDEN_ACCESS_FIELDS):
        raise PermissionError("the fit loader crossed a forbidden access boundary")
    if int(ledger["unexpected_path_attempts"]) != 0 or int(ledger["denied_attempts_total"]) != 0:
        raise PermissionError("the fit loader made an unexpected or denied path attempt")
    if len(selected_labels) != 320 or len(source_frames) != 320:
        raise ValueError("fit labels/source frames do not reconcile to 320 frames")
    return (
        source_access,
        records,
        selected_labels,
        source_frames,
        scenes,
        geometry,
        ledger,
        source_hashes,
        machine_manifest,
    )


def run_exact_fit(machine_manifest_sha256: str) -> dict[str, Any]:
    (
        source_access,
        records,
        selected_labels,
        source_frames,
        scenes,
        _geometry,
        ledger,
        source_hashes,
        machine_manifest,
    ) = _load_exact_fit_materials(machine_manifest_sha256)
    reports: list[dict[str, Any]] = []
    for record in records:
        identity = tuple(source_access._frame_identity_values(record))
        target, supervision = selected_labels[identity]
        frame = source_frames[identity]
        scene = scenes[str(record["scene_id"])]
        projection = scene["camera_projection"]
        camera = source_access.labels_v3._camera_observation(
            frame,
            horizontal_fov_deg=float(projection["horizontal_fov_deg"]),
            near_m=float(projection["near_m"]),
            vertical_fov_deg=float(projection["vertical_fov_deg"]),
            require_recorded_up=True,
            image_width_px=int(projection["resolution_wh"][0]),
            image_height_px=int(projection["resolution_wh"][1]),
            obstacle_ray_stride_px=2,
        )
        base_xy_yaw = source_access.labels_v3._base_xy_yaw(frame)
        physical_grid = scene["physical_grid"]
        reconstruction = ray_audit.reconstruct_frame_from_perfect_rays(
            camera=ray_audit.CameraRaySpec.from_camera_observation(camera),
            rendered_obstacle_boxes=scene["rendered_boxes"],
            collision_obstacle_boxes=scene["collision_boxes"],
            base_xy_yaw=base_xy_yaw,
            physical_free_mask=physical_grid.free_mask,
            physical_origin_xy_m=physical_grid.origin_xy,
            physical_cell_size_m=physical_grid.cell_size_m,
            output_grid=ray_audit.OutputGridSpec.from_local_grid(
                source_access.labels_v3.DEFAULT_LOCAL_GRID
            ),
        )
        reports.append(
            ray_audit.audit_frame_labels(
                authoritative_labels=target,
                supervision_mask=supervision,
                reconstruction=reconstruction,
                frame_key=source_access._frame_key(record),
            )
        )
    summary = ray_audit.summarize_exact_fit(reports)
    core = {
        "schema": "lewm_go2_perfect_camera_ray_field_fit_result_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset_role": "train",
            "frame_count": 320,
            "cell_count": 320 * 64 * 64,
            "learning_performed": False,
            "rgb_opened": False,
            "gpu_required": False,
        },
        "implementation": {
            "script_path": str(SCRIPT_PATH),
            "script_sha256": _sha256_file(SCRIPT_PATH),
            "core_path": str(CORE_PATH),
            "core_sha256": _sha256_file(CORE_PATH),
            "independent_rasterizer": True,
            "production_label_builder_called_by_rasterizer": False,
        },
        "input_authorization": {
            "frozen_binding_sha256": source_access.EXECUTION_BINDING_SHA256,
            "machine_manifest_file_sha256": machine_manifest_sha256,
            "machine_manifest_content_sha256": machine_manifest["content_sha256"],
            "source_hashes": source_hashes,
        },
        "fit_audit": summary,
        "frame_reports": reports,
        "interpretation": {
            "perfect_prescribed_ray_field_plus_physical_contract_is_sufficient": summary[
                "contract_assisted"
            ]["exact"],
            "perfect_prescribed_ray_field_without_physical_free_prior_is_sufficient": summary[
                "ray_only"
            ]["exact"],
            "ordinary_pixel_depth_sufficiency_proved": False,
            "ordinary_pixel_depth_reason": (
                "the tested perfect field includes five off-lattice ground-support "
                "queries per 0.05 m source cell"
            ),
        },
        "access_ledger": ledger,
        "licenses": {
            "model_output_authorized": False,
            "g2_authorized": False,
            "holdout_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    return _with_content_sha256(core)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-exact-fit", action="store_true")
    parser.add_argument("--machine-manifest-sha256")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.machine_manifest_sha256 is not None or args.output != OUTPUT_PATH:
            parser.error("dry-run forbids fit authorization and output overrides")
    elif args.machine_manifest_sha256 is None:
        parser.error("--run-exact-fit requires --machine-manifest-sha256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        print(json.dumps(run_dry_run(), sort_keys=True), flush=True)
        return 0
    result = run_exact_fit(str(args.machine_manifest_sha256))
    _write_json_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "content_sha256": result["content_sha256"],
                "contract_assisted_exact": result["fit_audit"][
                    "contract_assisted"
                ]["exact"],
                "ray_only_exact": result["fit_audit"]["ray_only"]["exact"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
