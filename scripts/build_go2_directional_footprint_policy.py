#!/usr/bin/env python3
"""Build a content-addressed Go2 directional-support footprint policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT, ROOT / "lewm_worlds"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from lewm.planning.oriented_footprint import DirectionalSupportFootprint  # noqa: E402


DEFAULT_CALIBRATION = (
    ROOT / ".generated/go2_footprint_calibration/geometry_v1_calibration.json"
)
DEFAULT_OUTPUT_DIR = ROOT / ".generated/go2_footprint_calibration"
SCHEMA = "lewm_go2_directional_footprint_policy_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _profile(
    directional_statistics: dict[str, dict[str, float]],
    *,
    statistic: str,
    margin_m: float,
) -> dict[str, Any]:
    raw_support = {
        float(angle): float(statistics[statistic])
        for angle, statistics in directional_statistics.items()
    }
    footprint = DirectionalSupportFootprint.from_directional_support(
        raw_support,
        margin_m=margin_m,
    )
    support_planes = [
        {
            "angle_deg": float(angle),
            "raw_support_m": raw_support[float(angle)],
            "support_with_margin_m": float(support),
        }
        for angle, support in zip(
            footprint.support_angles_deg,
            footprint.support_values_m,
            strict=True,
        )
    ]
    return {
        "statistic": statistic,
        "margin_m": margin_m,
        "construction": "intersection_of_directional_support_halfspaces",
        "support_planes": support_planes,
        "vertices_xy_body_m": [list(vertex) for vertex in footprint.vertices_xy_m],
        "vertex_count": len(footprint.vertices_xy_m),
        "maximum_vertex_radius_m": footprint.maximum_vertex_radius_m,
        "cardinal_support_m": {
            "forward": footprint.support_m(0.0),
            "left": footprint.support_m(90.0),
            "rear": footprint.support_m(180.0),
            "right": footprint.support_m(270.0),
        },
    }


def build_policy(calibration_path: Path) -> dict[str, Any]:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("schema") != "lewm_go2_swept_footprint_calibration_v1":
        raise ValueError("unsupported swept-footprint calibration schema")
    if not calibration.get("coverage_gate", {}).get("pass"):
        raise ValueError("primitive coverage gate has not passed")
    if calibration.get("additional_genesis_rollout", {}).get("required") is not False:
        raise ValueError("calibration still requires an additional Genesis rollout")
    margin_m = float(calibration["safety_margin"]["minimum_unmodeled_margin_m"])
    directional = calibration["executed_states"]["all_primitives"][
        "directional_support_m"
    ]
    q99_profile = _profile(
        directional,
        statistic="q99",
        margin_m=margin_m,
    )
    maximum_profile = _profile(
        directional,
        statistic="maximum",
        margin_m=margin_m,
    )
    return {
        "schema": SCHEMA,
        "policy_id": "go2-directional-observed-max-margin-v1",
        "status": "recommended_pending_physical_validation",
        "reference_frame": "go2_base_xy",
        "angle_convention": "degrees_ccw_from_body_forward",
        "recommended_profile": "observed_max_plus_margin",
        "profiles": {
            "q99_plus_margin": q99_profile,
            "observed_max_plus_margin": maximum_profile,
        },
        "selection_policy": {
            "planning_and_collision": "observed_max_plus_margin",
            "development_sensitivity_only": "q99_plus_margin",
            "rationale": (
                "collision feasibility must retain every accepted observed gait "
                "state; q99 deliberately excludes the upper one-percent tail"
            ),
        },
        "limitations": {
            "physical_measurement_status": calibration["safety_margin"][
                "physical_measurement_status"
            ],
            "support_angle_step_deg": 15.0,
            "margin_basis": calibration["safety_margin"]["note"],
            "promotion_gate": (
                "hardware dimensions and controller gait envelope remain required"
            ),
        },
        "source_artifacts": {
            "calibration_report": {
                "path": _relative_or_absolute(calibration_path),
                "sha256": _sha256_file(calibration_path),
            },
            "calibration_core": {
                "path": "lewm/planning/swept_footprint_calibration.py",
                "sha256": _sha256_file(
                    ROOT / "lewm/planning/swept_footprint_calibration.py"
                ),
            },
            "directional_geometry": {
                "path": "lewm/planning/oriented_footprint.py",
                "sha256": _sha256_file(ROOT / "lewm/planning/oriented_footprint.py"),
            },
            "builder": {
                "path": "scripts/build_go2_directional_footprint_policy.py",
                "sha256": _sha256_file(Path(__file__)),
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = build_policy(args.calibration.resolve())
    content_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    artifact = dict(payload)
    artifact["content_sha256"] = content_sha256
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"go2_directional_footprint_policy_v1_{content_sha256}.json"
    serialized = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise ValueError(f"content-address collision at {output}")
    output.write_text(serialized, encoding="utf-8")
    print(output)
    print(
        "recommended=observed_max_plus_margin "
        f"radius={artifact['profiles']['observed_max_plus_margin']['maximum_vertex_radius_m']:.6f}m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
