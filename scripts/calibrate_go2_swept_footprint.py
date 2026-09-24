#!/usr/bin/env python3
"""Calibrate the Go2 base-frame collision envelope from non-sealed data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.planning.swept_footprint_calibration import (
    CollisionKinematicModel,
    SourceArtifact,
    build_calibration_report,
    load_open_field_rollout,
    load_policy_nominal_stance,
)

DEFAULT_URDF = (
    ROOT
    / ".generated/venvs/genesis_render_vulkan/lib/python3.12/site-packages"
    / "genesis/assets/urdf/go2/urdf/go2.urdf"
)
DEFAULT_POLICY_CFG = (
    ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl"
)
DEFAULT_ROLLOUTS = (
    ROOT
    / ".generated/genesis_bulk_rollouts"
    / "cpu_physics_scale_512env_writer_20260517_raw"
    / "open_obstacle_field_36c57d3baa8d/messages.jsonl",
    ROOT
    / ".generated/smoke_multienv_rgb/raw"
    / "open_obstacle_field_0b58a3924f08/messages.jsonl",
    ROOT
    / ".generated/go2_footprint_calibration/open_plane_v2/raw/messages.jsonl",
)
DEFAULT_REFERENCE_ARTIFACTS = (
    ROOT / "config/go2_platform_manifest.yaml",
    ROOT
    / "third_party/unitree_go2_ros2/unitree_go2_description/urdf/const.xacro",
    ROOT
    / "third_party/unitree_go2_ros2/unitree_go2_description/urdf/leg.xacro",
    ROOT
    / "third_party/unitree_go2_ros2/unitree_go2_description/urdf"
    / "unitree_go2_robot.xacro",
)
DEFAULT_OUTPUT = (
    ROOT / ".generated/go2_footprint_calibration/geometry_v1_calibration.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the full projected Go2 collision envelope from the actual "
            "Genesis URDF and train/open-field primitive gait states."
        )
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--policy-cfg", type=Path, default=DEFAULT_POLICY_CFG)
    parser.add_argument(
        "--rollout",
        action="append",
        type=Path,
        default=None,
        help=(
            "Raw messages.jsonl. Repeat for multiple sources. Only train split "
            "open_obstacle_field summaries and stream metadata are accepted."
        ),
    )
    parser.add_argument(
        "--reference-artifact",
        action="append",
        type=Path,
        default=None,
        help="Additional URDF/platform reference artifact to hash.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-safety-margin-m", type=float, default=0.03)
    parser.add_argument("--rounding-increment-m", type=float, default=0.01)
    parser.add_argument("--radial-step-deg", type=float, default=1.0)
    parser.add_argument(
        "--maximum-abs-roll-pitch-deg",
        type=float,
        default=25.0,
    )
    parser.add_argument("--minimum-base-z-m", type=float, default=0.20)
    parser.add_argument("--minimum-blocks-per-primitive", type=int, default=10)
    parser.add_argument("--minimum-samples-per-primitive", type=int, default=40)
    parser.add_argument(
        "--minimum-noninitial-blocks-per-primitive",
        type=int,
        default=3,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = CollisionKinematicModel.from_urdf(args.urdf)
    nominal_stance = load_policy_nominal_stance(
        args.policy_cfg,
        required_joint_names=model.actuated_joint_names,
    )
    rollout_paths = tuple(args.rollout or DEFAULT_ROLLOUTS)
    datasets = [
        load_open_field_rollout(
            path,
            required_joint_names=model.actuated_joint_names,
            workspace_root=ROOT,
            maximum_abs_roll_pitch_rad=(
                args.maximum_abs_roll_pitch_deg * 3.141592653589793 / 180.0
            ),
            minimum_base_z_m=args.minimum_base_z_m,
        )
        for path in rollout_paths
    ]
    references = tuple(args.reference_artifact or DEFAULT_REFERENCE_ARTIFACTS)
    source_artifacts = [
        SourceArtifact("policy_nominal_stance_config", args.policy_cfg),
        SourceArtifact("calibration_cli", Path(__file__)),
        SourceArtifact(
            "calibration_core",
            ROOT / "lewm/planning/swept_footprint_calibration.py",
        ),
        *(
            SourceArtifact("platform_or_urdf_reference", path)
            for path in references
        ),
    ]
    report = build_calibration_report(
        model,
        nominal_joint_positions=nominal_stance,
        datasets=datasets,
        source_artifacts=source_artifacts,
        minimum_safety_margin_m=args.minimum_safety_margin_m,
        output_rounding_m=args.rounding_increment_m,
        radial_step_deg=args.radial_step_deg,
        minimum_blocks_per_primitive=args.minimum_blocks_per_primitive,
        minimum_samples_per_primitive=args.minimum_samples_per_primitive,
        minimum_noninitial_blocks_per_primitive=(
            args.minimum_noninitial_blocks_per_primitive
        ),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    recommendation = report["recommendation"]
    probe = recommendation["action_probe"]
    print(f"wrote {output}")
    print(
        "recommended: "
        f"radius={recommendation['static_configuration_space_radius_m']:.3f}m "
        f"probe=+{probe['forward_m']:.3f}/-{probe['rear_m']:.3f}m "
        f"half_width={probe['half_width_m']:.3f}m"
    )
    additional = report["additional_genesis_rollout"]
    print(
        "additional Genesis rollout: "
        f"required={additional['required']} "
        f"missing={','.join(additional['primitives_missing_coverage']) or 'none'}"
    )


if __name__ == "__main__":
    main()
