#!/usr/bin/env python3
"""Strictly score and aggregate a development directory of Go2 results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPO_ROOT, REPO_ROOT / "lewm_worlds"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from lewm.benchmarks.strict_result_scorer import score_result_payload  # noqa: E402
from lewm.planning.geometry_contract import load_geometry_contract  # noqa: E402
from lewm_worlds.manifest import parse_scene_manifest_dict  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--scene-corpus", type=Path, required=True)
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument(
        "--geometry-contract",
        type=Path,
        default=Path("config/go2_generalization_geometry_v1.json"),
    )
    parser.add_argument("--glob", default="*_result.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _scene_id(payload: dict[str, Any]) -> str:
    result = payload.get("result", payload)
    for key in ("scene_id", "scene"):
        value = result.get(key) if isinstance(result, dict) else None
        if isinstance(value, str) and value:
            return value
    argv = payload.get("provenance", {}).get("argv", [])
    if "--scene-id" in argv:
        index = argv.index("--scene-id")
        return str(argv[index + 1])
    raise ValueError("result has no scene identity")


def _arm(path: Path) -> str:
    suffix = "_result"
    stem = path.stem
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem.rsplit("_", 1)[-1]


def main() -> int:
    args = _parse_args()
    forbidden = ("sealed", "final_test", "final-test", "final_eval")
    for path in (args.result_dir, args.scene_corpus, args.output):
        lowered = str(path).lower()
        if any(token in lowered for token in forbidden):
            raise SystemExit(f"batch scorer is development-only: {path}")
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    contract_path = (
        args.geometry_contract
        if args.geometry_contract.is_absolute()
        else REPO_ROOT / args.geometry_contract
    )
    contract = load_geometry_contract(contract_path, repository_root=REPO_ROOT)
    result_dir = args.result_dir if args.result_dir.is_absolute() else REPO_ROOT / args.result_dir
    scene_corpus = (
        args.scene_corpus if args.scene_corpus.is_absolute() else REPO_ROOT / args.scene_corpus
    )
    records = []
    for result_path in sorted(result_dir.glob(args.glob)):
        payload = json.loads(result_path.read_text())
        scene_id = _scene_id(payload)
        manifest_path = (
            scene_corpus / args.split / args.family / scene_id / "manifest.json"
        )
        manifest = parse_scene_manifest_dict(json.loads(manifest_path.read_text()))
        score = score_result_payload(
            payload,
            scene_manifest=manifest,
            geometry_contract=contract,
        )
        records.append(
            {
                "arm": _arm(result_path),
                "result_path": str(result_path),
                "score": score.to_dict(),
            }
        )

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_arm[record["arm"]].append(record["score"])
    aggregates = {}
    for arm, scores in sorted(by_arm.items()):
        coverage = [
            float(score["coverage_final_fraction"])
            for score in scores
            if score["coverage_final_fraction"] is not None
        ]
        auc = [
            float(score["coverage_normalized_auc"])
            for score in scores
            if score["coverage_normalized_auc"] is not None
        ]
        aggregates[arm] = {
            "scene_count": len(scores),
            "physically_accepted_claim_events": sum(
                int(score["strict_accepted_claim_event_count"]) for score in scores
            ),
            "target_count": sum(int(score["target_count"]) for score in scores),
            "physical_four_of_four_scenes": sum(
                score["strict_four_of_four_complete"] is True for score in scores
            ),
            "complete_scores": sum(bool(score["score_complete"]) for score in scores),
            "median_final_coverage_fraction": (
                None if not coverage else statistics.median(coverage)
            ),
            "mean_final_coverage_fraction": (
                None if not coverage else statistics.fmean(coverage)
            ),
            "mean_normalized_coverage_auc": (
                None if not auc else statistics.fmean(auc)
            ),
            "canonical_collision_ticks": sum(
                len(score["canonical_geometry_collision_ticks"] or ())
                for score in scores
            ),
            "proxy_strict_discrepancies": sum(
                len(score["discrepancies"]) for score in scores
            ),
        }
    output = {
        "schema": "lewm_go2_strict_result_batch_v1",
        "development_only": True,
        "geometry_contract_sha256": contract.sha256,
        "result_dir": str(result_dir),
        "scene_corpus": str(scene_corpus),
        "aggregates": aggregates,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregates, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
