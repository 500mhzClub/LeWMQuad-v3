#!/usr/bin/env python3
"""Bind the no-training explicit per-link geometry result and evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".generated/explicit_per_link_geometric_micro_state_upper_bound_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/explicit_per_link_geometric_micro_state_upper_bound_v1"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): value.update(block)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def directory_bytes(path: Path, excluded: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and item != excluded) if path.exists() else 0


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    index_path = OUT / "geometry_index.json"; result_path = OUT / "result.json"
    checker_path = OUT / "result_checker.json"; ledger_path = CACHE / "row_level_evidence.jsonl"
    index = json.loads(index_path.read_text()); result = json.loads(result_path.read_text()); checker = json.loads(checker_path.read_text())
    if index["status"] != "PASS" or not checker["pass"]: raise RuntimeError("geometry result is not complete")
    if sha(ledger_path) != result["row_level_evidence"]["sha256"]: raise RuntimeError("row ledger binding failed")
    active = subprocess.run(["pgrep", "-af", "materialize_explicit_per_link|evaluate_explicit_per_link|check_explicit_per_link"], capture_output=True, text=True).stdout.splitlines()
    active = [line for line in active if "pgrep -af" not in line and "finalize_explicit_per_link" not in line]
    if active: raise RuntimeError(f"experiment processes remain active: {active}")
    receipt_path = OUT / "persistence_receipt.json"
    value = {"schema": "explicit_per_link_geometric_micro_state_upper_bound_v1_persistence_receipt",
             "source_commit": "10b3a190d506830e6a87e04a0f1c832b92295bd7",
             "primary_classification": result["primary_classification"],
             "secondary_classifications": result["secondary_classifications"],
             "index": {"path": str(index_path), "sha256": sha(index_path), "content_digest": index["content_digest"]},
             "result": {"path": str(result_path), "sha256": sha(result_path), "content_digest": result["content_digest"]},
             "checker": {"path": str(checker_path), "sha256": sha(checker_path), "content_digest": checker["content_digest"]},
             "row_ledger": {"path": str(ledger_path), "sha256": sha(ledger_path), "rows": result["row_level_evidence"]["rows"], "bytes": ledger_path.stat().st_size},
             "runtime": {"collection_wall_s": index["runtime"]["parallel_wall_s"], "per_state_runtime_sum_s": index["runtime"]["per_state_runtime_sum_s"], "evaluation_s": result["runtime"]["evaluation_s"]},
             "storage_bytes": directory_bytes(OUT, receipt_path) + directory_bytes(CACHE, receipt_path),
             "execution": {"model_training": 0, "experimental_learned_inference": 0, "fresh_panel": 0, "jepa_access": 0,
                           "successor_predictor": 0, "memory_navigation_novelty_routing": 0,
                           "fixed_low_level_controller_plant_replays": index["transitions"]},
             "active_processes": active, "next_step": result["next_step"]}
    value["content_digest"] = digest(value); atomic_json(receipt_path, value)
    print(json.dumps(value, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
