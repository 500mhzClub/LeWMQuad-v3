#!/usr/bin/env python3
"""Freeze audited development and sealed Go2 generalization scene manifests."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPO_ROOT, REPO_ROOT / "lewm_worlds"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from lewm.benchmarks.generalization_protocol import (  # noqa: E402
    SceneSplitCounts,
    audited_scene_record,
    build_hashed_scene_role_commitment,
    build_scene_disjoint_manifests,
    fixed_spawn_audit_config_from_geometry_contract,
    write_scene_disjoint_manifests,
)
from lewm.benchmarks.go2_physical_eligibility import (  # noqa: E402
    REQUIRED_GEOMETRY_SCHEMA,
    audit_physical_scene_eligibility,
    physical_config_from_geometry_contract,
    policy_from_geometry_contract,
)
from lewm.planning.geometry_contract import (  # noqa: E402
    DEFAULT_GEOMETRY_CONTRACT,
    load_geometry_contract,
)
from lewm_worlds.families import build_family_manifest, registered_families  # noqa: E402
from lewm_worlds.fixed_spawn_audit import audit_fixed_spawn  # noqa: E402
from lewm_worlds.splits import plan_corpus, plan_sha256  # noqa: E402


DEFAULT_FAMILIES = (
    "medium_enclosed_maze",
)
REQUIRED_TARGET_COLORS = ("blue", "green", "red", "yellow")
DEPLOYMENT_FAMILY = "go2_deployment_medium_maze"
DEPLOYMENT_DISC_RADIUS_M = 0.47


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-id",
        default="go2-jepa-online-memory-generalization-v1",
    )
    parser.add_argument("--candidate-plan-seed", type=int, default=2026070901)
    parser.add_argument("--split-seed", type=int, default=2026070902)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--candidates-per-family", type=int, default=96)
    parser.add_argument("--validation-per-family", type=int, default=8)
    parser.add_argument("--sealed-test-per-family", type=int, default=10)
    parser.add_argument("--minimum-train-per-family", type=int, default=48)
    parser.add_argument(
        "--required-target-colors",
        default=",".join(REQUIRED_TARGET_COLORS),
        help="Require exactly one landmark for each listed task color.",
    )
    parser.add_argument(
        "--geometry-contract",
        type=Path,
        default=DEFAULT_GEOMETRY_CONTRACT,
    )
    parser.add_argument(
        "--development-output",
        type=Path,
        default=Path("config/go2_generalization_v1/development.json"),
    )
    parser.add_argument(
        "--sealed-test-output",
        type=Path,
        default=Path("config/go2_generalization_v1/sealed_test.json"),
    )
    parser.add_argument(
        "--creation-report-output",
        type=Path,
        default=Path("config/go2_generalization_v1/creation_report.json"),
    )
    parser.add_argument(
        "--exclude-scene-id-file",
        type=Path,
        action="append",
        default=[],
        help="Text file containing one known train/development scene ID per line.",
    )
    parser.add_argument(
        "--exclude-scene-glob",
        action="append",
        default=[],
        help="Glob whose file names or result provenance identify known scenes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing commitment (never use after evaluation).",
    )
    parser.add_argument(
        "--require-physical-eligibility",
        action="store_true",
        help=(
            "Require geometry-v2 0.47 m disc eligibility plus exact observed-max "
            "directional-polygon SE(2) claim reachability. Geometry v2 enables "
            "this gate automatically."
        ),
    )
    parser.add_argument(
        "--physical-footprint-policy",
        type=Path,
        default=None,
        help=(
            "Optional explicit policy path; it must resolve to the artifact bound "
            "by geometry v2."
        ),
    )
    parser.add_argument("--physical-yaw-bins", type=int, default=16)
    parser.add_argument("--physical-rotation-subsamples", type=int, default=5)
    parser.add_argument("--physical-mask-validation-samples", type=int, default=32)
    parser.add_argument(
        "--scene-role-commitment-output",
        type=Path,
        default=None,
        help=(
            "Required for physical benchmark creation. Writes only role-tagged "
            "hashes of scene IDs so dataset builders can exclude development and "
            "sealed roles without opening the sealed manifest."
        ),
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_id_from_json(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    result = payload.get("result", payload) if isinstance(payload, dict) else {}
    if isinstance(result, dict):
        for key in ("scene_id", "scene"):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    argv = provenance.get("argv", []) if isinstance(provenance, dict) else []
    if isinstance(argv, list) and "--scene-id" in argv:
        index = argv.index("--scene-id")
        if index + 1 < len(argv):
            return str(argv[index + 1])
    return None


def scene_ids_from_paths(paths: Iterable[Path]) -> set[str]:
    """Extract scene IDs without opening binary training shards."""

    scene_ids: set[str] = set()
    for path in paths:
        if path.name == "manifest.json":
            scene_id = _scene_id_from_json(path)
        elif path.suffix.lower() == ".json":
            scene_id = _scene_id_from_json(path)
        else:
            scene_id = path.stem
        if scene_id:
            scene_ids.add(scene_id)
    return scene_ids


def validate_task_landmarks(manifest: object, required_colors: Iterable[str]) -> None:
    """Reject scene families that do not implement the same target task."""

    required = tuple(sorted(str(color).strip().lower() for color in required_colors))
    colors: list[str] = []
    for landmark in getattr(manifest, "landmarks"):
        text = f"{landmark.material_id} {landmark.object_id}".lower()
        matches = [color for color in required if color in text]
        if len(matches) != 1:
            raise ValueError(
                f"scene {getattr(manifest, 'scene_id', '<unknown>')} has an "
                f"unrecognized or ambiguous task landmark: {landmark.object_id}"
            )
        colors.append(matches[0])
    if tuple(sorted(colors)) != required:
        raise ValueError(
            f"scene {getattr(manifest, 'scene_id', '<unknown>')} target colors "
            f"{sorted(colors)} do not match exactly-one-per-color task {list(required)}"
        )


def _known_scene_ids(args: argparse.Namespace) -> set[str]:
    known: set[str] = set()
    for path in args.exclude_scene_id_file:
        resolved = _resolve(path)
        for line in resolved.read_text().splitlines():
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                known.add(scene_id)
    for pattern in args.exclude_scene_glob:
        known.update(scene_ids_from_paths(REPO_ROOT.glob(str(pattern))))
    return known


def main() -> int:
    args = _parse_args()
    families = tuple(
        item.strip() for item in str(args.families).split(",") if item.strip()
    )
    unknown = sorted(set(families) - set(registered_families()))
    if not families or unknown:
        raise SystemExit(f"invalid families; unknown={unknown}")
    required_colors = tuple(
        color.strip().lower()
        for color in str(args.required_target_colors).split(",")
        if color.strip()
    )
    if len(required_colors) != len(set(required_colors)) or not required_colors:
        raise SystemExit("--required-target-colors must be unique and non-empty")
    for name in (
        "candidates_per_family",
        "validation_per_family",
        "sealed_test_per_family",
        "minimum_train_per_family",
    ):
        if int(getattr(args, name)) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")

    geometry_path = _resolve(args.geometry_contract)
    geometry = load_geometry_contract(
        geometry_path,
        repository_root=REPO_ROOT,
    )
    audit_config = fixed_spawn_audit_config_from_geometry_contract(geometry)
    physical_required = bool(
        args.require_physical_eligibility
        or geometry.schema == REQUIRED_GEOMETRY_SCHEMA
    )
    physical_policy = None
    physical_config = None
    physical_rejection_counts: Counter[str] = Counter()
    disc_eligible_count = 0
    polygon_eligible_count = 0
    combined_eligible_count = 0
    if physical_required:
        if geometry.schema != REQUIRED_GEOMETRY_SCHEMA:
            raise SystemExit(
                "--require-physical-eligibility requires geometry contract v2"
            )
        if families != (DEPLOYMENT_FAMILY,):
            raise SystemExit(
                "geometry v2 benchmark creation requires exactly family "
                f"{DEPLOYMENT_FAMILY!r}"
            )
        if not math.isclose(
            float(geometry.configuration_space.body_inflation_radius_m),
            DEPLOYMENT_DISC_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit(
                "geometry v2 physical benchmark requires a 0.47 m planning disc"
            )
        if args.scene_role_commitment_output is None:
            raise SystemExit(
                "physical benchmark creation requires "
                "--scene-role-commitment-output"
            )
        physical_policy = policy_from_geometry_contract(
            geometry,
            repository_root=REPO_ROOT,
            policy_override=(
                _resolve(args.physical_footprint_policy)
                if args.physical_footprint_policy is not None
                else None
            ),
        )
        physical_config = physical_config_from_geometry_contract(
            geometry,
            yaw_bins=int(args.physical_yaw_bins),
            rotation_subsamples=int(args.physical_rotation_subsamples),
            mask_validation_samples=int(args.physical_mask_validation_samples),
        )
    plan = plan_corpus(
        int(args.candidate_plan_seed),
        {
            "candidate": {
                family: int(args.candidates_per_family) for family in families
            }
        },
        validate=False,
    )
    known_scene_ids = _known_scene_ids(args)
    records = []
    generated_known_overlap: list[str] = []
    for index, assignment in enumerate(plan.assignments, start=1):
        manifest = build_family_manifest(
            scene_seed=int(assignment.scene_seed),
            family=str(assignment.family),
            split=str(assignment.split),
            difficulty_tier=None,
        )
        try:
            validate_task_landmarks(manifest, required_colors)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if manifest.scene_id in known_scene_ids:
            generated_known_overlap.append(manifest.scene_id)
            continue
        audit = audit_fixed_spawn(manifest, config=audit_config)
        if audit.fully_reachable:
            disc_eligible_count += 1
        record = audited_scene_record(manifest, audit)
        if physical_required:
            assert physical_policy is not None and physical_config is not None
            physical = audit_physical_scene_eligibility(
                manifest,
                policy=physical_policy,
                config=physical_config,
            )
            if physical.eligible:
                polygon_eligible_count += 1
            combined_eligible = bool(audit.fully_reachable and physical.eligible)
            if combined_eligible:
                combined_eligible_count += 1
            failures = []
            if not audit.fully_reachable:
                failures.append("disc:" + audit.failure_reason)
            if not physical.eligible:
                failures.append("polygon:" + physical.failure_reason)
                for reason in physical.failure_reason.split(";"):
                    if reason:
                        physical_rejection_counts[reason.split(":", 1)[0]] += 1
            record = replace(
                record,
                fully_reachable=combined_eligible,
                failure_reason=";".join(failures),
                physical_eligible=bool(physical.eligible),
                physical_eligibility_sha256=physical.sha256,
            )
        records.append(record)
        if index % 24 == 0:
            print(
                f"audited {index}/{len(plan.assignments)} candidates"
                + (
                    f" disc={disc_eligible_count} polygon={polygon_eligible_count} "
                    f"combined={combined_eligible_count}"
                    if physical_required
                    else ""
                ),
                flush=True,
            )

    if generated_known_overlap:
        raise SystemExit(
            "fresh candidate plan overlaps known scenes: "
            + ",".join(sorted(generated_known_overlap))
        )
    allocations = {
        family: SceneSplitCounts(
            validation=int(args.validation_per_family),
            sealed_test=int(args.sealed_test_per_family),
        )
        for family in families
    }
    manifests = build_scene_disjoint_manifests(
        records,
        benchmark_id=str(args.benchmark_id),
        split_seed=int(args.split_seed),
        geometry_contract=geometry,
        allocations=allocations,
    )
    train_counts = {
        family: sum(
            record["family"] == family
            for record in manifests.development["train_scenes"]
        )
        for family in families
    }
    too_small = {
        family: count
        for family, count in train_counts.items()
        if count < int(args.minimum_train_per_family)
    }
    if too_small:
        raise SystemExit(
            "insufficient audited training scenes; increase --candidates-per-family: "
            f"{too_small}"
        )

    development_path = _resolve(args.development_output)
    sealed_path = _resolve(args.sealed_test_output)
    report_path = _resolve(args.creation_report_output)
    role_commitment_path = (
        _resolve(args.scene_role_commitment_output)
        if args.scene_role_commitment_output is not None
        else None
    )
    development_screening_path = (
        role_commitment_path.with_name("development_scene_ids.sha256")
        if role_commitment_path is not None
        else None
    )
    sealed_screening_path = (
        role_commitment_path.with_name("sealed_scene_ids.sha256")
        if role_commitment_path is not None
        else None
    )
    output_paths = [development_path, sealed_path, report_path]
    if role_commitment_path is not None:
        assert development_screening_path is not None
        assert sealed_screening_path is not None
        output_paths.extend(
            (
                role_commitment_path,
                development_screening_path,
                sealed_screening_path,
            )
        )
    if physical_required and any(
        "go2_generalization_v3" in path.parts for path in output_paths
    ):
        raise SystemExit("geometry v2 outputs may not overwrite the v3 benchmark")
    for path in output_paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    write_scene_disjoint_manifests(
        manifests,
        development_path=development_path,
        sealed_test_path=sealed_path,
        overwrite=bool(args.overwrite),
    )
    role_commitment_provenance = None
    if role_commitment_path is not None:
        assert development_screening_path is not None
        assert sealed_screening_path is not None
        role_commitment = build_hashed_scene_role_commitment(manifests)
        role_commitment_path.parent.mkdir(parents=True, exist_ok=True)
        role_commitment_path.write_text(
            json.dumps(role_commitment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        screening_files = {}
        for role, path in (
            ("development", development_screening_path),
            ("sealed_test", sealed_screening_path),
        ):
            tokens = role_commitment["scene_id_sha256_by_role"][role]
            path.write_text(
                "" if not tokens else "\n".join(tokens) + "\n",
                encoding="ascii",
            )
            screening_files[role] = {
                "path": _relative_or_absolute(path),
                "file_sha256": _sha256_file(path),
                "scene_id_set_sha256": role_commitment[
                    "set_sha256_by_role"
                ][role]["scene_id_sha256"],
                "count": len(tokens),
                "format": "newline_sha256_scene_id",
                "contains_raw_scene_ids": False,
            }
        role_commitment_provenance = {
            "path": _relative_or_absolute(role_commitment_path),
            "file_sha256": _sha256_file(role_commitment_path),
            "content_sha256": role_commitment["content_sha256"],
            "contains_raw_scene_ids": False,
            "counts": role_commitment["counts"],
            "set_sha256_by_role": role_commitment["set_sha256_by_role"],
            "screening_files": screening_files,
        }
    report = {
        "schema": "lewm_navigation_benchmark_creation_report_v0",
        "benchmark_id": str(args.benchmark_id),
        "candidate_plan_seed": int(args.candidate_plan_seed),
        "candidate_plan_sha256": plan_sha256(plan),
        "split_seed": int(args.split_seed),
        "geometry_contract": {
            "path": str(geometry_path.relative_to(REPO_ROOT)),
            "sha256": geometry.sha256,
        },
        "families": list(families),
        "required_target_colors": sorted(required_colors),
        "candidates_per_family": int(args.candidates_per_family),
        "known_scene_count": len(known_scene_ids),
        "known_scene_ids_sha256": _sha256_payload(sorted(known_scene_ids)),
        "train_count_by_family": train_counts,
        "validation_scene_count": len(
            manifests.development["validation_scenes"]
        ),
        "excluded_scene_count": len(manifests.development["excluded_scenes"]),
        "sealed_test_scene_count": manifests.development["sealed_test"][
            "scene_count"
        ],
        "sealed_test_commitment_sha256": manifests.development["sealed_test"][
            "commitment_sha256"
        ],
    }
    if physical_required:
        assert physical_policy is not None and physical_config is not None
        report["physical_eligibility"] = {
            "required": True,
            "gate": "0.47m_disc_and_exact_actual_yaw_directional_polygon",
            "planning_disc_radius_m": DEPLOYMENT_DISC_RADIUS_M,
            "policy": physical_policy.provenance_dict(
                repository_root=REPO_ROOT
            ),
            "config": asdict(physical_config),
            "evaluator": {
                "path": "lewm/benchmarks/go2_physical_eligibility.py",
                "sha256": _sha256_file(
                    REPO_ROOT / "lewm/benchmarks/go2_physical_eligibility.py"
                ),
            },
            "candidate_count": len(records),
            "disc_eligible_count": disc_eligible_count,
            "polygon_eligible_count": polygon_eligible_count,
            "combined_eligible_count": combined_eligible_count,
            "combined_acceptance_rate": (
                combined_eligible_count / len(records) if records else 0.0
            ),
            "polygon_rejection_counts": dict(
                sorted(physical_rejection_counts.items())
            ),
        }
    if role_commitment_provenance is not None:
        report["hashed_scene_role_commitment"] = role_commitment_provenance
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "benchmark frozen: "
        f"train={sum(train_counts.values())} "
        f"validation={report['validation_scene_count']} "
        f"sealed={report['sealed_test_scene_count']} "
        f"excluded={report['excluded_scene_count']}",
        flush=True,
    )
    print(
        f"sealed_test_commitment_sha256={report['sealed_test_commitment_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
