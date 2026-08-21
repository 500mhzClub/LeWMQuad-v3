#!/usr/bin/env python3
"""Bind the repaired corpus, model, ledger, and terminal development decision."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CORPUS_OUT = ROOT / ".generated/two_ply_successor_transition_corpus_repaired_v1"
MODEL_OUT = ROOT / ".generated/two_ply_successor_transition_corpus_repair_and_model_v1"
CORPUS_CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/two_ply_successor_transition_corpus_repaired_v1"
MODEL_CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/two_ply_successor_transition_corpus_repair_and_model_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def directory_bytes(path: Path) -> int:
    if not path.exists(): return 0
    excluded = MODEL_OUT / "persistence_receipt.json"
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and item != excluded)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(temporary, path)


def main() -> int:
    corpus_path = CORPUS_OUT / "corpus_index.json"; result_path = MODEL_OUT / "result.json"
    corpus = json.loads(corpus_path.read_text()); result = json.loads(result_path.read_text())
    if corpus["status"] != "PASS" or not all(corpus["gate"].values()): raise RuntimeError("corpus repair gate not complete")
    if result["classification"] != "TRUE_SUCCESSOR_SET_VIABILITY_NO_SIGNAL": raise RuntimeError("unexpected terminal")
    checkpoint = Path(result["model"]["checkpoint"]); ledger = Path(result["row_ledger"]["path"])
    if sha(checkpoint) != result["model"]["checkpoint_sha256"] or sha(ledger) != result["row_ledger"]["sha256"]:
        raise RuntimeError("model evidence binding failed")
    collection = json.loads((CORPUS_OUT / "collection_runtime.json").read_text())
    validation = json.loads((CORPUS_OUT / "deterministic_validation.json").read_text())
    active = subprocess.run(["pgrep", "-af", "materialize_two_ply_successor_transition|run_two_ply_successor_transition"], capture_output=True, text=True).stdout.splitlines()
    active = [line for line in active if "pgrep -af" not in line and "finalize_two_ply" not in line]
    result_value = {"schema": "two_ply_successor_transition_corpus_repair_and_model_v1_persistence_receipt",
        "source_commit": "400b00604873449ed587c05c6209ca596b93fd33", "classification": result["classification"],
        "compute_classification": result["compute"]["classification"], "corpus": {"path": str(corpus_path), "sha256": sha(corpus_path),
            "logical_digest": corpus["corpus_logical_digest"], "rows": sum(role["current_transitions"] + role["successor_action_transitions"] for role in corpus["inventory"].values())},
        "result": {"path": str(result_path), "sha256": sha(result_path), "content_digest": result["content_digest"]},
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint), "bytes": checkpoint.stat().st_size},
        "row_ledger": {"path": str(ledger), "sha256": sha(ledger), "bytes": ledger.stat().st_size, "rows": result["row_ledger"]["rows"]},
        "execution": {"seeds_trained": 1, "seed": result["model"]["seed"], "fresh_panel_collected": False, "jepa_accessed": False,
            "successor_predictor_trained": False, "direct_nonviability_classifier_trained": False, "memory_or_navigation_executed": False},
        "runtime": {"initial_smoke_state_s": corpus["records"][0]["runtime_s"], "collection_wall_s": collection["wall_runtime_s"],
            "independent_validation_wall_s": validation["runtime_s"], "training_s": result["model"]["runtime_s"],
            "evaluation_and_benchmark_s": result["runtime_s"], "corpus_state_runtime_sum_s": corpus["runtime_s_sum"]},
        "storage_bytes": directory_bytes(CORPUS_OUT) + directory_bytes(MODEL_OUT) + directory_bytes(CORPUS_CACHE) + directory_bytes(MODEL_CACHE),
        "active_processes": active, "next_decision": result["decision"],
        "preserved": result["preserved"]}
    result_value["content_digest"] = digest(result_value); atomic_json(MODEL_OUT / "persistence_receipt.json", result_value)
    print(json.dumps(result_value, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
