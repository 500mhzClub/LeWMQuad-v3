"""Rendered counterfactual artifact accounting for Phase 2D spatial joins."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .phase2_data import CONFIRMATORY_SPLIT_REQUIREMENTS
from .phase2d_readiness import canonical_split_name


PLAN_SUMMARY_SCHEMA = "jepa_counterfactual_render_plan_summary_v0"
RENDER_ROOT_SCHEMA = "jepa_counterfactual_render_root_summary_v0"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _resolved_matches(value: object, expected: Path) -> bool:
    if value is None:
        return False
    return Path(str(value)).resolve() == expected.resolve()


def _resolve_from(parent: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else parent / path


def _resolve_existing_render_path(parent: Path, value: object) -> Path | None:
    if value is None:
        return None
    raw_path = Path(str(value))
    candidates = [_resolve_from(parent, str(value))]
    if raw_path.name:
        candidates.append(parent / raw_path.name)
    if raw_path.parent.name and raw_path.name:
        candidates.append(parent / raw_path.parent.name / raw_path.name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].resolve()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _audit_scene_metadata(render_root: Path, scenes: list) -> dict:
    scene_summary_count = 0
    scene_summary_invalid_schema_count = 0
    scene_metadata_present_count = 0
    scene_metadata_missing_count = 0
    scene_metadata_relocated_count = 0
    scene_metadata_frame_sum = 0
    for scene in scenes:
        if not isinstance(scene, Mapping):
            scene_metadata_missing_count += 1
            continue
        output_value = scene.get("output")
        if output_value is None:
            scene_metadata_missing_count += 1
            continue
        output_path = Path(str(output_value))
        if not output_path.is_absolute():
            output_path = render_root / output_path
        scene_summary = _load_json(output_path / "summary.json")
        if scene_summary is None:
            scene_metadata_missing_count += 1
            continue
        scene_summary_count += 1
        if scene_summary.get("schema") != "lewm_rendered_vision_v0":
            scene_summary_invalid_schema_count += 1
        recorded_metadata = _resolve_from(
            output_path,
            str(scene_summary.get("frames_rendered_jsonl")),
        ).resolve()
        metadata_path = _resolve_existing_render_path(
            output_path,
            scene_summary.get("frames_rendered_jsonl"),
        )
        if metadata_path is None or not metadata_path.is_file():
            scene_metadata_missing_count += 1
            continue
        scene_metadata_present_count += 1
        if metadata_path != recorded_metadata:
            scene_metadata_relocated_count += 1
        scene_metadata_frame_sum += _line_count(metadata_path)
    return {
        "scene_summary_count": scene_summary_count,
        "scene_summary_invalid_schema_count": scene_summary_invalid_schema_count,
        "scene_metadata_present_count": scene_metadata_present_count,
        "scene_metadata_missing_count": scene_metadata_missing_count,
        "scene_metadata_relocated_count": scene_metadata_relocated_count,
        "scene_metadata_frame_sum": scene_metadata_frame_sum,
    }


def _audit_one_render_root(
    *,
    split_name: str,
    plan_root: Path,
    render_root: Path,
) -> dict:
    plan_summary_path = plan_root / "summary.json"
    render_summary_path = render_root / "root_summary.json"
    plan_summary = _load_json(plan_summary_path)
    render_summary = _load_json(render_summary_path)

    if plan_summary is None:
        return {
            "split": split_name,
            "plan_root": str(plan_root.resolve()),
            "render_root": str(render_root.resolve()),
            "plan_summary_present": False,
            "render_root_summary_present": render_summary is not None,
            "passed": False,
        }

    expected_scene_count = int(plan_summary.get("scene_count") or 0)
    expected_frame_count = int(plan_summary.get("frame_count") or 0)
    expected_candidate_count = int(plan_summary.get("candidate_count") or 0)
    plan_paths = plan_summary.get("plans", [])
    if not isinstance(plan_paths, list):
        plan_paths = []

    if render_summary is None:
        return {
            "split": split_name,
            "plan_root": str(plan_root.resolve()),
            "render_root": str(render_root.resolve()),
            "plan_summary_present": True,
            "render_root_summary_present": False,
            "plan_schema_valid": plan_summary.get("schema") == PLAN_SUMMARY_SCHEMA,
            "expected_scene_count": expected_scene_count,
            "expected_frame_count": expected_frame_count,
            "expected_candidate_count": expected_candidate_count,
            "planned_scene_reports": len(plan_paths),
            "passed": False,
        }

    scenes = render_summary.get("scenes", [])
    if not isinstance(scenes, list):
        scenes = []
    rendered_scene_count = int(render_summary.get("scene_count") or 0)
    rendered_frame_count = int(render_summary.get("frame_count") or 0)
    invalid_frame_count = int(render_summary.get("invalid_frame_count") or 0)
    scene_report_invalid_frames = sum(
        int(scene.get("invalid_frame_count") or 0)
        for scene in scenes
        if isinstance(scene, Mapping)
    )
    scene_report_frames = sum(
        int(scene.get("frame_count") or 0)
        for scene in scenes
        if isinstance(scene, Mapping)
    )
    scene_report_unexpected_return_count = sum(
        int(scene.get("render_return_code") not in (0, 2, None))
        for scene in scenes
        if isinstance(scene, Mapping)
    )
    scene_metadata = _audit_scene_metadata(render_root, scenes)

    checks = {
        "plan_schema_valid": plan_summary.get("schema") == PLAN_SUMMARY_SCHEMA,
        "render_schema_valid": render_summary.get("schema") == RENDER_ROOT_SCHEMA,
        "render_summary_plan_root_matches": _resolved_matches(
            render_summary.get("plan_root"),
            plan_root,
        ),
        "render_summary_output_root_matches": _resolved_matches(
            render_summary.get("output_root"),
            render_root,
        ),
        "planned_scene_report_count_matches": len(plan_paths) == expected_scene_count,
        "rendered_scene_count_matches_plan": rendered_scene_count
        == expected_scene_count,
        "rendered_frame_count_matches_plan": rendered_frame_count
        == expected_frame_count,
        "scene_report_count_matches_plan": len(scenes) == expected_scene_count,
        "scene_report_frame_sum_matches_root": scene_report_frames
        == rendered_frame_count,
        "scene_report_invalid_frame_sum_matches_root": scene_report_invalid_frames
        == invalid_frame_count,
        "scene_reports_return_accepted_status": (
            scene_report_unexpected_return_count == 0
        ),
        "scene_summaries_present": scene_metadata["scene_summary_count"]
        == expected_scene_count,
        "scene_summary_schemas_valid": (
            scene_metadata["scene_summary_invalid_schema_count"] == 0
        ),
        "scene_render_metadata_present": (
            scene_metadata["scene_metadata_present_count"] == expected_scene_count
        ),
        "scene_render_metadata_frame_count_matches_root": (
            scene_metadata["scene_metadata_frame_sum"] == rendered_frame_count
        ),
    }
    return {
        "split": split_name,
        "plan_root": str(plan_root.resolve()),
        "render_root": str(render_root.resolve()),
        "plan_summary_path": str(plan_summary_path.resolve()),
        "render_root_summary_path": str(render_summary_path.resolve()),
        "plan_summary_present": True,
        "render_root_summary_present": True,
        "expected_scene_count": expected_scene_count,
        "expected_frame_count": expected_frame_count,
        "expected_candidate_count": expected_candidate_count,
        "planned_scene_reports": len(plan_paths),
        "rendered_scene_count": rendered_scene_count,
        "rendered_frame_count": rendered_frame_count,
        "invalid_frame_count": invalid_frame_count,
        "scene_report_count": len(scenes),
        "scene_report_frame_sum": scene_report_frames,
        "scene_report_invalid_frame_sum": scene_report_invalid_frames,
        "scene_report_unexpected_return_count": scene_report_unexpected_return_count,
        **scene_metadata,
        "all_rendered_frames_valid": invalid_frame_count == 0,
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_phase2d_render_readiness(
    *,
    plan_roots: Mapping[str, Path],
    render_roots: Mapping[str, Path],
) -> dict:
    """Audit whether rendered counterfactual roots are ready for spatial joins."""

    canonical_plan_roots = {
        canonical_split_name(name): Path(path) for name, path in plan_roots.items()
    }
    canonical_render_roots = {
        canonical_split_name(name): Path(path) for name, path in render_roots.items()
    }
    if len(canonical_plan_roots) != len(plan_roots):
        raise ValueError("plan-root split names must be unique after canonicalization")
    if len(canonical_render_roots) != len(render_roots):
        raise ValueError("render-root split names must be unique after canonicalization")

    required_splits = set(CONFIRMATORY_SPLIT_REQUIREMENTS)
    missing_plan_roots = sorted(required_splits - set(canonical_plan_roots))
    missing_render_roots = sorted(required_splits - set(canonical_render_roots))
    split_reports = {}
    for split_name in sorted(required_splits & set(canonical_plan_roots)):
        render_root = canonical_render_roots.get(split_name)
        if render_root is None:
            split_reports[split_name] = {
                "split": split_name,
                "plan_root": str(canonical_plan_roots[split_name].resolve()),
                "render_root_present": False,
                "passed": False,
            }
            continue
        split_reports[split_name] = _audit_one_render_root(
            split_name=split_name,
            plan_root=canonical_plan_roots[split_name],
            render_root=render_root,
        )

    checks = {
        "all_required_plan_roots_present": not missing_plan_roots,
        "all_required_render_roots_present": not missing_render_roots,
        "all_split_renders_complete_and_accounted": bool(split_reports)
        and all(report.get("passed", False) for report in split_reports.values())
        and not missing_plan_roots
        and not missing_render_roots,
    }
    return {
        "schema": "jepa_phase2d_render_readiness_v0",
        "required_splits": list(CONFIRMATORY_SPLIT_REQUIREMENTS),
        "missing_plan_roots": missing_plan_roots,
        "missing_render_roots": missing_render_roots,
        "splits": split_reports,
        "checks": checks,
        "all_rendered_frames_valid": bool(split_reports)
        and all(
            report.get("all_rendered_frames_valid", False)
            for report in split_reports.values()
        ),
        "ready_for_spatial_future_join": all(checks.values()),
    }
