#!/usr/bin/env python3
"""Decompose V2 perfect-ray fit mismatches using authorized train geometry.

The runner preserves V1/V2 artifacts, opens only the same authorized 320-frame
train fit scope, converts source objects to primitive scene jobs, and performs
the pure decomposition in at most six forked CPU workers.  It reads no RGB,
model, G2, holdout, runtime, or sealed payload.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing
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
    ROOT
    / ".generated/go2_perfect_camera_ray_field_mismatch_decomposition/v1/result.json"
)
V2_RESULT_PATH = (
    ROOT / ".generated/go2_perfect_camera_ray_field_fit_audit/v2/result.json"
)
V2_RESULT_FILE_SHA256 = (
    "388313d4d01ee7f30107b537504638af20cb580e949645499be0d7a6b292f244"
)
V2_RESULT_CONTENT_SHA256 = (
    "a1e597dbb57517939800aca6b753e23fd3d89582f378409c817f57c97e1e67a3"
)
SCRIPT_PATH = Path(__file__).resolve()
CORE_PATH = (
    ROOT
    / "lewm/benchmarks/go2_perfect_camera_ray_field_mismatch_decomposition.py"
).resolve()
V1_RUNNER_PATH = (
    ROOT / "scripts/audit_go2_perfect_camera_ray_field_fit.py"
).resolve()
MAX_WORKERS = 6
THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _load_neutral(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


decomposition = _load_neutral(
    "go2_perfect_ray_mismatch_decomposition_independent", CORE_PATH
)
v1_runner = _load_neutral(
    "go2_perfect_camera_ray_field_fit_v1_reader_for_decomposition",
    V1_RUNNER_PATH,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**core, "content_sha256": _canonical_json_sha256(core)}


def _write_json_exclusive(payload: Mapping[str, Any]) -> None:
    destination = OUTPUT_PATH.resolve()
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
        raise FileExistsError(
            f"immutable decomposition output already exists: {destination}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_v2_result() -> dict[str, Any]:
    raw = V2_RESULT_PATH.read_bytes()
    if _sha256_bytes(raw) != V2_RESULT_FILE_SHA256:
        raise ValueError("immutable V2 result file SHA-256 changed")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("immutable V2 result is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("immutable V2 result is not an object")
    core = dict(payload)
    declared = str(core.pop("content_sha256", ""))
    if (
        declared != V2_RESULT_CONTENT_SHA256
        or _canonical_json_sha256(core) != declared
    ):
        raise ValueError("immutable V2 result content SHA-256 changed")
    fit = payload.get("fit_audit")
    if (
        payload.get("schema") != "lewm_go2_perfect_camera_ray_field_fit_result_v2"
        or not isinstance(fit, Mapping)
        or int(fit.get("frame_count", -1)) != 320
        or int(fit.get("collision_vetoed_ray_only", {}).get("mismatch_cell_count", -1))
        != 98_473
        or int(fit.get("observable_ray_only", {}).get("mismatch_cell_count", -1))
        != 100_730
        or int(fit.get("collision_veto_effect_on_ray_only_cell_count", -1))
        != 2_257
    ):
        raise ValueError("immutable V2 result decision fields changed")
    return payload


def _runtime_contract() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    affinity = sorted(os.sched_getaffinity(0))
    thread_environment = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if executable.parent != Path("/usr/bin"):
        raise RuntimeError("decomposition requires the system Python interpreter")
    if np.__version__ != "1.26.4":
        raise RuntimeError("decomposition requires NumPy 1.26.4")
    if not 1 <= len(affinity) <= MAX_WORKERS:
        raise RuntimeError("decomposition requires affinity capped to 1-6 CPUs")
    if any(value != "1" for value in thread_environment.values()):
        raise RuntimeError("decomposition requires all numeric thread caps set to 1")
    return {
        "python_executable": str(executable),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "cpu_affinity": affinity,
        "numeric_thread_environment": thread_environment,
        "maximum_scene_workers": MAX_WORKERS,
        "multiprocessing_start_method": "fork",
    }


def _box_mapping(box: Any) -> dict[str, Any]:
    return {
        "center_xyz_m": [float(value) for value in box.center_xyz_m],
        "size_xyz_m": [float(value) for value in box.size_xyz_m],
        "roll_rad": float(getattr(box, "roll_rad", 0.0)),
        "pitch_rad": float(getattr(box, "pitch_rad", 0.0)),
        "yaw_rad": float(getattr(box, "yaw_rad", 0.0)),
    }


def _camera_mapping(camera: Any) -> dict[str, Any]:
    if camera.vertical_fov_deg is None:
        raise ValueError("source camera lacks vertical FOV")
    return {
        "position_xyz_m": [float(value) for value in camera.position_xyz_m],
        "lookat_xyz_m": [float(value) for value in camera.lookat_xyz_m],
        "up_xyz": [float(value) for value in camera.up_xyz],
        "horizontal_fov_deg": float(camera.horizontal_fov_deg),
        "vertical_fov_deg": float(camera.vertical_fov_deg),
        "near_m": float(camera.near_m),
        "ground_plane_z_m": float(camera.ground_plane_z_m),
        "image_width_px": int(camera.image_width_px),
        "image_height_px": int(camera.image_height_px),
        "obstacle_ray_stride_px": int(camera.obstacle_ray_stride_px),
    }


def _output_grid_mapping(grid: Any) -> dict[str, Any]:
    return {
        "rows": int(grid.rows),
        "cols": int(grid.cols),
        "cell_size_m": float(grid.cell_size_m),
        "forward_min_edge_m": float(grid.forward_min_edge_m),
        "left_min_edge_m": float(grid.left_min_edge_m),
    }


def _manifest_collision_metadata(manifest: Any) -> list[tuple[str, Any]]:
    records: list[tuple[str, Any]] = []
    records.extend(("wall", box) for box in manifest.walls)
    records.extend(("obstacle", box) for box in manifest.obstacles)
    records.extend(("landmark", box) for box in manifest.landmarks)
    visual = manifest.visual_randomization
    if visual is not None:
        records.extend(("distractor", box) for box in visual.distractor_objects)
    return records


def _scene_job(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    selected_records: Sequence[tuple[int, Mapping[str, Any]]],
    selected_labels: Mapping[tuple[Any, ...], tuple[np.ndarray, np.ndarray]],
    source_frames: Mapping[tuple[Any, ...], Mapping[str, Any]],
    source_access: Any,
) -> dict[str, Any]:
    manifest = scene["manifest"]
    declared = _manifest_collision_metadata(manifest)
    collision_boxes = tuple(scene["collision_boxes"])
    rendered_boxes = tuple(scene["rendered_boxes"])
    if len(declared) != len(collision_boxes):
        raise ValueError("manifest/collision box count changed")
    for (_group, declared_box), collision_box in zip(declared, collision_boxes):
        if source_access._box_geometry(declared_box) != source_access._box_geometry(
            collision_box
        ):
            raise ValueError("manifest/collision box order or geometry changed")
    matching = scene["box_matching"]
    collision_to_rendered = {
        int(collision_index): int(rendered_index)
        for rendered_index, collision_index in matching["matches"]
    }
    parity_complete = bool(
        len(collision_to_rendered) == len(collision_boxes) == len(rendered_boxes)
        and not matching["unmatched_collision_indices"]
        and not matching["unmatched_rendered_indices"]
    )
    if not parity_complete:
        raise ValueError("exact fit scene lacks rendered/collision box parity")
    collision_records = [
        {
            "box": _box_mapping(box),
            "group": group,
            "kind": str(box.kind),
            "object_id": str(box.object_id),
            "rendered_index": collision_to_rendered[index],
        }
        for index, ((group, box), collision_box) in enumerate(
            zip(declared, collision_boxes)
        )
        if source_access._box_geometry(box)
        == source_access._box_geometry(collision_box)
    ]
    projection = scene["camera_projection"]
    frames: list[dict[str, Any]] = []
    for order, record in selected_records:
        identity = tuple(source_access._frame_identity_values(record))
        target, supervision = selected_labels[identity]
        frame = source_frames[identity]
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
        frames.append(
            {
                "order": int(order),
                "authoritative_labels": np.array(target, dtype=np.uint8, copy=True),
                "supervision_mask": np.array(supervision, dtype=bool, copy=True),
                "frame_key": source_access._frame_key(record),
                "camera": _camera_mapping(camera),
                "base_xy_yaw": list(source_access.labels_v3._base_xy_yaw(frame)),
            }
        )
    physical_grid = scene["physical_grid"]
    return {
        "scene_id": scene_id,
        "family": str(scene["family"]),
        "rendered_boxes": [_box_mapping(box) for box in rendered_boxes],
        "collision_records": collision_records,
        "rendered_collision_parity_complete": parity_complete,
        "world_bounds_xy_m": [
            [float(value) for value in pair] for pair in manifest.world_bounds_xy_m
        ],
        "physical_free_mask": np.array(
            physical_grid.free_mask, dtype=bool, order="C", copy=True
        ),
        "physical_origin_xy_m": [float(value) for value in physical_grid.origin_xy],
        "physical_cell_size_m": float(physical_grid.cell_size_m),
        "output_grid": _output_grid_mapping(source_access.labels_v3.DEFAULT_LOCAL_GRID),
        "frames": frames,
    }


def _decompose_scene_job(job: Mapping[str, Any]) -> dict[str, Any]:
    reports = []
    for frame in job["frames"]:
        report = decomposition.decompose_frame(
            authoritative_labels=frame["authoritative_labels"],
            supervision_mask=frame["supervision_mask"],
            frame_key=frame["frame_key"],
            camera=frame["camera"],
            rendered_boxes=job["rendered_boxes"],
            collision_records=job["collision_records"],
            base_xy_yaw=frame["base_xy_yaw"],
            physical_free_mask=job["physical_free_mask"],
            physical_origin_xy_m=job["physical_origin_xy_m"],
            physical_cell_size_m=job["physical_cell_size_m"],
            world_bounds_xy_m=job["world_bounds_xy_m"],
            rendered_collision_parity_complete=job[
                "rendered_collision_parity_complete"
            ],
            output_grid=job["output_grid"],
        )
        reports.append({"order": int(frame["order"]), "report": report})
    return {
        "scene_id": str(job["scene_id"]),
        "family": str(job["family"]),
        "worker_pid": os.getpid(),
        "reports": reports,
    }


def _frame_key_token(frame_key: Mapping[str, Any]) -> str:
    return json.dumps(frame_key, sort_keys=True, separators=(",", ":"))


def _aggregate_category(
    reports: Sequence[Mapping[str, Any]],
    *,
    section: str,
    category: str,
) -> dict[str, Any]:
    identities: list[list[Any]] = []
    for report in reports:
        coords = report["_private_identities"][section][category]
        for row, column in coords:
            identities.append([report["frame_key"], int(row), int(column)])
    return {
        "count": len(identities),
        "frame_count": sum(
            bool(report["_private_identities"][section][category])
            for report in reports
        ),
        "ordered_cell_identities_sha256": _canonical_json_sha256(identities),
        "sample": identities[:64],
    }


def _aggregate_reports(
    reports: Sequence[dict[str, Any]],
    *,
    v2_result: Mapping[str, Any],
) -> dict[str, Any]:
    if len(reports) != 320:
        raise ValueError("decomposition requires exactly 320 frame reports")
    if any(report.get("schema") != decomposition.FRAME_SCHEMA for report in reports):
        raise ValueError("one decomposition frame schema changed")
    v2_by_key = {
        _frame_key_token(report["frame_key"]): report
        for report in v2_result["frame_reports"]
    }
    if len(v2_by_key) != 320:
        raise ValueError("immutable V2 frame keys are not unique")
    for report in reports:
        key = _frame_key_token(report["frame_key"])
        expected = v2_by_key.get(key)
        if expected is None:
            raise ValueError("decomposition frame is outside immutable V2")
        physical = int(report["physical_prior_mismatch_cell_count"])
        collision = int(report["collision_veto_delta_cell_count"])
        if (
            report["authoritative_labels_sha256"]
            != expected["authoritative_labels_sha256"]
            or physical != int(expected["ray_only_mismatch_cell_count"])
            or collision
            != int(expected["collision_veto_effect_on_ray_only_cell_count"])
            or physical + collision
            != int(expected["observable_ray_only_mismatch_cell_count"])
        ):
            raise ValueError("decomposition frame does not reconcile to immutable V2")
    physical_total = sum(
        int(report["physical_prior_mismatch_cell_count"]) for report in reports
    )
    collision_total = sum(
        int(report["collision_veto_delta_cell_count"]) for report in reports
    )
    if physical_total != 98_473 or collision_total != 2_257:
        raise ValueError("decomposition totals changed from immutable V2")
    physical_categories = {
        category: _aggregate_category(
            reports, section="physical_prior", category=category
        )
        for category in decomposition.PHYSICAL_CATEGORIES
    }
    collision_categories = {
        category: _aggregate_category(
            reports, section="collision_veto", category=category
        )
        for category in decomposition.COLLISION_CATEGORIES
    }
    if sum(item["count"] for item in physical_categories.values()) != physical_total:
        raise AssertionError("physical aggregate categories do not reconcile")
    if sum(item["count"] for item in collision_categories.values()) != collision_total:
        raise AssertionError("collision aggregate categories do not reconcile")
    families = {}
    for family in decomposition.v2.EXPECTED_FAMILIES:
        family_reports = [
            report for report in reports if report["frame_key"]["family"] == family
        ]
        families[family] = {
            "frame_count": len(family_reports),
            "physical_prior_mismatch_cell_count": sum(
                int(report["physical_prior_mismatch_cell_count"])
                for report in family_reports
            ),
            "collision_veto_delta_cell_count": sum(
                int(report["collision_veto_delta_cell_count"])
                for report in family_reports
            ),
            "physical_prior_categories": {
                category: _aggregate_category(
                    family_reports, section="physical_prior", category=category
                )
                for category in decomposition.PHYSICAL_CATEGORIES
            },
            "collision_veto_categories": {
                category: _aggregate_category(
                    family_reports, section="collision_veto", category=category
                )
                for category in decomposition.COLLISION_CATEGORIES
            },
        }
    physical_flags = {
        name: sum(
            int(report["physical_prior_overlapping_evidence_flags"][name])
            for report in reports
        )
        for name in reports[0]["physical_prior_overlapping_evidence_flags"]
    }
    collision_flags = {
        name: sum(
            int(report["collision_veto_overlapping_evidence_flags"][name])
            for report in reports
        )
        for name in reports[0]["collision_veto_overlapping_evidence_flags"]
    }
    public_reports = []
    for report in reports:
        public = dict(report)
        del public["_private_identities"]
        public_reports.append(public)
    return {
        "frame_count": 320,
        "cell_count": 1_310_720,
        "physical_prior_mismatch_cell_count": physical_total,
        "collision_veto_delta_cell_count": collision_total,
        "observable_ray_only_mismatch_cell_count": physical_total + collision_total,
        "output_collision_overlap_within_physical_prior_count": 0,
        "physical_prior_category_precedence": list(
            decomposition.PHYSICAL_CATEGORIES
        ),
        "physical_prior_categories": physical_categories,
        "physical_prior_overlapping_evidence_flags": physical_flags,
        "collision_veto_categories": collision_categories,
        "collision_veto_overlapping_evidence_flags": collision_flags,
        "families": families,
        "frame_reports": public_reports,
    }


def _synthetic_dry_run() -> dict[str, Any]:
    from lewm.datasets import go2_paired_navigation as labels_v3
    from lewm_worlds.planning_grid import InflatedOccupancyGrid

    manifest, boxes = v1_runner._synthetic_manifest()
    grid = InflatedOccupancyGrid(
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
    target, supervision, _observed = (
        labels_v3._observable_physical_raster_and_output_labels(
            grid,
            rendered_obstacle_boxes=boxes,
            collision_obstacle_boxes=boxes,
            base_xy_yaw=(0.0, 0.0, 0.17),
            camera=camera,
            local_grid=labels_v3.DEFAULT_LOCAL_GRID,
        )
    )
    report = decomposition.decompose_frame(
        authoritative_labels=target,
        supervision_mask=supervision,
        frame_key={"family": "open_obstacle_field", "frame": 0},
        camera=_camera_mapping(camera),
        rendered_boxes=[_box_mapping(box) for box in boxes],
        collision_records=[
            {
                "box": _box_mapping(box),
                "group": "obstacle",
                "kind": str(box.kind),
                "object_id": str(box.object_id),
                "rendered_index": index,
            }
            for index, box in enumerate(boxes)
        ],
        base_xy_yaw=(0.0, 0.0, 0.17),
        physical_free_mask=grid.free_mask,
        physical_origin_xy_m=grid.origin_xy,
        physical_cell_size_m=grid.cell_size_m,
        world_bounds_xy_m=manifest.world_bounds_xy_m,
        rendered_collision_parity_complete=True,
        output_grid=_output_grid_mapping(labels_v3.DEFAULT_LOCAL_GRID),
    )
    private = report.pop("_private_identities")
    return {
        "schema": "lewm_go2_perfect_ray_mismatch_decomposition_dry_run_v1",
        "dry_run": True,
        "generated_fit_payload_opened": False,
        "g2_or_holdout_payload_opened": False,
        "gpu_required": False,
        "physical_prior_mismatch_cell_count": report[
            "physical_prior_mismatch_cell_count"
        ],
        "collision_veto_delta_cell_count": report[
            "collision_veto_delta_cell_count"
        ],
        "physical_partition_count": sum(
            len(value) for value in private["physical_prior"].values()
        ),
        "collision_partition_count": sum(
            len(value) for value in private["collision_veto"].values()
        ),
        "frame_report_sha256": _canonical_json_sha256(report),
    }


def run_exact(machine_manifest_sha256: str) -> dict[str, Any]:
    runtime = _runtime_contract()
    v2_result = _load_v2_result()
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
    ) = v1_runner._load_exact_fit_materials(machine_manifest_sha256)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for order, record in enumerate(records):
        grouped.setdefault(str(record["scene_id"]), []).append((order, record))
    jobs = [
        _scene_job(
            scene_id=scene_id,
            scene=scenes[scene_id],
            selected_records=grouped[scene_id],
            selected_labels=selected_labels,
            source_frames=source_frames,
            source_access=source_access,
        )
        for scene_id in sorted(grouped)
    ]
    context = multiprocessing.get_context("fork")
    worker_count = min(MAX_WORKERS, len(jobs))
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        scene_results = list(pool.map(_decompose_scene_job, jobs))
    ordered: list[tuple[int, dict[str, Any]]] = []
    for scene_result in scene_results:
        ordered.extend(
            (int(item["order"]), item["report"])
            for item in scene_result["reports"]
        )
    ordered.sort(key=lambda item: item[0])
    if [order for order, _report in ordered] != list(range(320)):
        raise ValueError("parallel scene jobs did not preserve exact frame order")
    reports = [report for _order, report in ordered]
    decomposition_result = _aggregate_reports(reports, v2_result=v2_result)
    if _sha256_file(V2_RESULT_PATH) != V2_RESULT_FILE_SHA256:
        raise RuntimeError("immutable V2 result changed during decomposition")
    worker_pids = sorted({int(result["worker_pid"]) for result in scene_results})
    core = {
        "schema": "lewm_go2_perfect_ray_mismatch_decomposition_result_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset_role": "train",
            "frame_count": 320,
            "learning_performed": False,
            "rgb_opened": False,
            "gpu_required": False,
        },
        "immutable_v2_result": {
            "path": str(V2_RESULT_PATH.resolve()),
            "file_sha256": V2_RESULT_FILE_SHA256,
            "content_sha256": V2_RESULT_CONTENT_SHA256,
            "preserved_exactly": True,
        },
        "implementation": {
            "runner_path": str(SCRIPT_PATH),
            "runner_sha256": _sha256_file(SCRIPT_PATH),
            "core_path": str(CORE_PATH),
            "core_sha256": _sha256_file(CORE_PATH),
            "v1_fit_reader_path": str(V1_RUNNER_PATH),
            "v1_fit_reader_sha256": _sha256_file(V1_RUNNER_PATH),
        },
        "runtime_environment": {
            **runtime,
            "scene_job_count": len(jobs),
            "configured_worker_count": worker_count,
            "observed_worker_process_count": len(worker_pids),
            "worker_pids": worker_pids,
        },
        "input_authorization": {
            "frozen_binding_sha256": source_access.EXECUTION_BINDING_SHA256,
            "machine_manifest_file_sha256": machine_manifest_sha256,
            "machine_manifest_content_sha256": machine_manifest["content_sha256"],
            "source_hashes": source_hashes,
        },
        "decomposition": decomposition_result,
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
    args = parser.parse_args(argv)
    if args.dry_run and args.machine_manifest_sha256 is not None:
        parser.error("dry-run forbids fit authorization")
    if args.run_exact_fit and args.machine_manifest_sha256 is None:
        parser.error("--run-exact-fit requires --machine-manifest-sha256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        print(json.dumps(_synthetic_dry_run(), sort_keys=True), flush=True)
        return 0
    result = run_exact(str(args.machine_manifest_sha256))
    _write_json_exclusive(result)
    decomposition_result = result["decomposition"]
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH.resolve()),
                "content_sha256": result["content_sha256"],
                "physical_prior_categories": {
                    name: value["count"]
                    for name, value in decomposition_result[
                        "physical_prior_categories"
                    ].items()
                },
                "collision_veto_categories": {
                    name: value["count"]
                    for name, value in decomposition_result[
                        "collision_veto_categories"
                    ].items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
