#!/usr/bin/env python3
"""Run the additive V2 perfect-ray audit on the exact 320 train-fit frames.

V2 preserves the immutable V1 result and adds ``observable_ray_only``, which
uses neither the physical-free prior nor the privileged collision veto.  The
exact mode requires system Python, NumPy 1.26.4, one-thread numeric libraries,
and a process affinity capped to at most eight CPUs.
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
    ROOT / ".generated/go2_perfect_camera_ray_field_fit_audit/v2/result.json"
)
V1_RESULT_PATH = (
    ROOT / ".generated/go2_perfect_camera_ray_field_fit_audit/v1/result.json"
)
V1_RESULT_FILE_SHA256 = (
    "bfb159a168cf4284d99934e40c00fdf3aab2a705e545e00159622f22aac616ba"
)
V1_RESULT_CONTENT_SHA256 = (
    "d32cd3ae37b6171dff623cf4a15759264cba288b705064e1ee095c110b6cf174"
)
SCRIPT_PATH = Path(__file__).resolve()
V2_CORE_PATH = (
    ROOT / "lewm/benchmarks/go2_perfect_camera_ray_field_audit_v2.py"
).resolve()
V1_CORE_PATH = (
    ROOT / "lewm/benchmarks/go2_perfect_camera_ray_field_audit.py"
).resolve()
V1_RUNNER_PATH = (
    ROOT / "scripts/audit_go2_perfect_camera_ray_field_fit.py"
).resolve()
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


ray_audit = _load_neutral(
    "go2_perfect_camera_ray_field_audit_v2_independent", V2_CORE_PATH
)
v1_runner = _load_neutral(
    "go2_perfect_camera_ray_field_fit_v1_reader_for_v2", V1_RUNNER_PATH
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    return {**core, "content_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.resolve()
    if destination != OUTPUT_PATH.resolve():
        raise PermissionError("V2 audit output path is frozen")
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
        raise FileExistsError(f"immutable V2 audit output already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_and_validate_v1_result() -> dict[str, Any]:
    raw = V1_RESULT_PATH.read_bytes()
    if _sha256_bytes(raw) != V1_RESULT_FILE_SHA256:
        raise ValueError("immutable V1 result file SHA-256 changed")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("immutable V1 result is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("immutable V1 result is not an object")
    core = dict(payload)
    declared = str(core.pop("content_sha256", ""))
    if (
        declared != V1_RESULT_CONTENT_SHA256
        or _sha256_bytes(_canonical_json_bytes(core)) != declared
    ):
        raise ValueError("immutable V1 result content SHA-256 changed")
    fit = payload.get("fit_audit")
    if (
        payload.get("schema") != "lewm_go2_perfect_camera_ray_field_fit_result_v1"
        or not isinstance(fit, Mapping)
        or int(fit.get("frame_count", -1)) != 320
        or int(fit.get("cell_count", -1)) != 1_310_720
        or fit.get("contract_assisted", {}).get("exact") is not True
        or int(fit.get("contract_assisted", {}).get("mismatch_cell_count", -1)) != 0
        or fit.get("ray_only", {}).get("exact") is not False
        or int(fit.get("ray_only", {}).get("mismatch_cell_count", -1)) != 98_473
        or int(fit.get("ray_only", {}).get("mismatch_frame_count", -1)) != 111
    ):
        raise ValueError("immutable V1 result decision fields changed")
    return payload


def _runtime_contract() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    affinity = sorted(os.sched_getaffinity(0))
    thread_environment = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if executable.parent != Path("/usr/bin"):
        raise RuntimeError("V2 exact fit requires the system Python interpreter")
    if np.__version__ != "1.26.4":
        raise RuntimeError("V2 exact fit requires NumPy 1.26.4")
    if not 1 <= len(affinity) <= 8:
        raise RuntimeError("V2 exact fit requires process affinity capped to 1-8 CPUs")
    if any(value != "1" for value in thread_environment.values()):
        raise RuntimeError("V2 exact fit requires all numeric thread caps set to 1")
    return {
        "python_executable": str(executable),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "cpu_affinity": affinity,
        "numeric_thread_environment": thread_environment,
    }


def _synthetic_inputs() -> tuple[Any, ...]:
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
    expected, supervision, _observed = (
        labels_v3._observable_physical_raster_and_output_labels(
            grid,
            rendered_obstacle_boxes=boxes,
            collision_obstacle_boxes=boxes,
            base_xy_yaw=(0.0, 0.0, 0.17),
            camera=camera,
            local_grid=labels_v3.DEFAULT_LOCAL_GRID,
        )
    )
    return labels_v3, grid, camera, boxes, expected, supervision


def run_dry_run() -> dict[str, Any]:
    _labels_v3, grid, camera, boxes, expected, supervision = _synthetic_inputs()
    kwargs = {
        "camera": ray_audit.CameraRaySpec.from_camera_observation(camera),
        "rendered_obstacle_boxes": boxes,
        "collision_obstacle_boxes": boxes,
        "base_xy_yaw": (0.0, 0.0, 0.17),
        "physical_free_mask": grid.free_mask,
        "physical_origin_xy_m": grid.origin_xy,
        "physical_cell_size_m": grid.cell_size_m,
    }
    first = ray_audit.reconstruct_frame_from_perfect_rays(**kwargs)
    second = ray_audit.reconstruct_frame_from_perfect_rays(**kwargs)
    report = ray_audit.audit_frame_labels(
        authoritative_labels=expected,
        supervision_mask=supervision,
        reconstruction=first,
        frame_key={"dataset_role": "synthetic_train", "frame_index": 0},
    )
    collision_only = ray_audit.reconstruct_frame_from_perfect_rays(
        **{
            **kwargs,
            "rendered_obstacle_boxes": (),
            "collision_obstacle_boxes": boxes,
        }
    )
    return {
        "schema": "lewm_go2_perfect_camera_ray_field_dry_run_v2",
        "dry_run": True,
        "gpu_required": False,
        "generated_fit_payload_opened": False,
        "g2_or_holdout_payload_opened": False,
        "contract_parity": report["contract_mismatch_cell_count"] == 0,
        "deterministic": bool(
            first.field_sha256 == second.field_sha256
            and np.array_equal(first.contract_labels, second.contract_labels)
            and np.array_equal(
                first.observable_ray_only_labels,
                second.observable_ray_only_labels,
            )
        ),
        "observable_ray_only_omits_collision_veto": bool(
            np.any(
                collision_only.observable_ray_only_labels
                != collision_only.collision_vetoed_ray_only_labels
            )
        ),
        "contract_mismatch_cell_count": report["contract_mismatch_cell_count"],
        "collision_vetoed_ray_only_mismatch_cell_count": report[
            "ray_only_mismatch_cell_count"
        ],
        "observable_ray_only_mismatch_cell_count": report[
            "observable_ray_only_mismatch_cell_count"
        ],
        "field_sha256": first.field_sha256,
        "observable_ray_only_labels_sha256": report[
            "observable_ray_only_labels_sha256"
        ],
    }


def _assert_additive_v1_parity(
    *,
    v1_result: Mapping[str, Any],
    v2_summary: Mapping[str, Any],
) -> None:
    v1_fit = v1_result["fit_audit"]
    if (
        v2_summary["contract_assisted"]["exact"]
        != v1_fit["contract_assisted"]["exact"]
        or v2_summary["contract_assisted"]["mismatch_cell_count"]
        != v1_fit["contract_assisted"]["mismatch_cell_count"]
        or v2_summary["collision_vetoed_ray_only"]["exact"]
        != v1_fit["ray_only"]["exact"]
        or v2_summary["collision_vetoed_ray_only"]["mismatch_cell_count"]
        != v1_fit["ray_only"]["mismatch_cell_count"]
        or v2_summary["collision_vetoed_ray_only"]["mismatch_frame_count"]
        != v1_fit["ray_only"]["mismatch_frame_count"]
        or v2_summary["ordered_authoritative_label_hashes_sha256"]
        != v1_fit["ordered_authoritative_label_hashes_sha256"]
        or v2_summary["ordered_contract_label_hashes_sha256"]
        != v1_fit["ordered_contract_label_hashes_sha256"]
        or v2_summary[
            "ordered_collision_vetoed_ray_only_label_hashes_sha256"
        ]
        != v1_fit["ordered_ray_only_label_hashes_sha256"]
    ):
        raise RuntimeError("V2 did not preserve the immutable V1 arms exactly")


def run_exact_fit(machine_manifest_sha256: str) -> dict[str, Any]:
    runtime = _runtime_contract()
    v1_result = _load_and_validate_v1_result()
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
    _assert_additive_v1_parity(v1_result=v1_result, v2_summary=summary)
    if _sha256_file(V1_RESULT_PATH) != V1_RESULT_FILE_SHA256:
        raise RuntimeError("immutable V1 result changed during V2 execution")
    observable = summary["observable_ray_only"]
    core = {
        "schema": "lewm_go2_perfect_camera_ray_field_fit_result_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset_role": "train",
            "frame_count": 320,
            "cell_count": 1_310_720,
            "learning_performed": False,
            "rgb_opened": False,
            "gpu_required": False,
        },
        "runtime_environment": runtime,
        "immutable_v1_result": {
            "path": str(V1_RESULT_PATH.resolve()),
            "file_sha256": V1_RESULT_FILE_SHA256,
            "content_sha256": V1_RESULT_CONTENT_SHA256,
            "preserved_exactly": True,
        },
        "implementation": {
            "v2_script_path": str(SCRIPT_PATH),
            "v2_script_sha256": _sha256_file(SCRIPT_PATH),
            "v2_core_path": str(V2_CORE_PATH),
            "v2_core_sha256": _sha256_file(V2_CORE_PATH),
            "v1_runner_path": str(V1_RUNNER_PATH),
            "v1_runner_sha256": _sha256_file(V1_RUNNER_PATH),
            "v1_core_path": str(V1_CORE_PATH),
            "v1_core_sha256": _sha256_file(V1_CORE_PATH),
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
            "contract_assisted_reconstruction_exact": summary[
                "contract_assisted"
            ]["exact"],
            "tested_prescribed_observable_ray_arm_determines_current_target": (
                observable["exact"]
            ),
            "target_cells_not_determined_by_observable_ray_field": observable[
                "mismatch_cell_count"
            ],
            "affected_frame_count": observable["mismatch_frame_count"],
            "mismatch_class_transitions": observable[
                "mismatch_class_transitions"
            ],
            "ordinary_pixel_depth_sufficiency_proved": False,
            "deployment_rgb_sufficiency_proved": False,
            "reason": (
                "even observable_ray_only uses five off-lattice ground-support "
                "queries in addition to the registered pixel first-hit lattice"
            ),
            "audit_registration_caveat": (
                "the mechanical comparison retains the frozen output lattice and "
                "exact audit-time pose registration; neither is claimed as an RGB "
                "model input"
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
    args = parser.parse_args(argv)
    if args.dry_run and args.machine_manifest_sha256 is not None:
        parser.error("dry-run forbids fit authorization")
    if args.run_exact_fit and args.machine_manifest_sha256 is None:
        parser.error("--run-exact-fit requires --machine-manifest-sha256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        print(json.dumps(run_dry_run(), sort_keys=True), flush=True)
        return 0
    result = run_exact_fit(str(args.machine_manifest_sha256))
    _write_json_exclusive(OUTPUT_PATH, result)
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH.resolve()),
                "content_sha256": result["content_sha256"],
                "contract_assisted_exact": result["fit_audit"][
                    "contract_assisted"
                ]["exact"],
                "observable_ray_only_exact": result["fit_audit"][
                    "observable_ray_only"
                ]["exact"],
                "observable_ray_only_mismatch_cell_count": result["fit_audit"][
                    "observable_ray_only"
                ]["mismatch_cell_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
