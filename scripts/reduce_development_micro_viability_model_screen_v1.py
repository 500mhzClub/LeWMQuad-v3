#!/usr/bin/env python3
"""Validate and summarize the completed development-only model screen."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from lewm.safety import lightweight_one_tick_viability_model_v1 as CORE
from scripts import run_development_micro_viability_model_screen_v1 as SCREEN


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp"); temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); os.replace(temporary, path)


def main() -> int:
    SCREEN.validate_frozen_inputs(); result = json.loads(SCREEN.RESULT.read_text())
    timing_path = SCREEN.OUT / "compute_benchmark.json"; timing = json.loads(timing_path.read_text())
    if result["classification"] != "DEVELOPMENT_MICRO_VIABILITY_NO_SIGNAL": raise RuntimeError("unexpected screen classification")
    if timing["classification"] != "MICRO_VIABILITY_COMPUTE_SIGNAL": raise RuntimeError("unexpected compute classification")
    ledger = Path(result["row_level_ledger"]["path"])
    if SCREEN.sha(ledger) != result["row_level_ledger"]["sha256"]: raise RuntimeError("model ledger binding failed")
    summary = {"schema": "development_micro_viability_model_screen_v1_summary", "source_commit": SCREEN.SOURCE_COMMIT,
        "predecessor_terminal": "FRESH_MICRO_VIABILITY_PANEL_INADEQUATE", "model_classification": result["classification"],
        "secondary_classification": result["secondary_classification"], "compute_classification": timing["classification"],
        "replanning_interface": "REPLANNING_INTERFACE_UNRESOLVED", "checkpoint_sha256": result["training"]["checkpoint_sha256"],
        "one_seed_trained": True, "seed": CORE.SEED, "fresh_panel_generated": False, "learned_closed_loop_ran": False,
        "jepa_predictor_opened_or_executed": False, "fresh_panel_v2_justified": False,
        "next_decision": "Close LIGHTWEIGHT_ONE_TICK_VIABILITY_MODEL_V1; do not collect FRESH_MICRO_VIABILITY_PANEL_V2 for this architecture.",
        "runtime_s": {"training": result["training"]["runtime_s"], "compute_benchmark": timing["wall_runtime_s"]},
        "artifacts": {"split": {"path": str(SCREEN.SPLIT), "sha256": SCREEN.sha(SCREEN.SPLIT)},
            "checkpoint": {"path": str(SCREEN.CHECKPOINT), "sha256": SCREEN.sha(SCREEN.CHECKPOINT), "bytes": SCREEN.CHECKPOINT.stat().st_size},
            "result": {"path": str(SCREEN.RESULT), "sha256": SCREEN.sha(SCREEN.RESULT), "bytes": SCREEN.RESULT.stat().st_size},
            "compute": {"path": str(timing_path), "sha256": SCREEN.sha(timing_path), "bytes": timing_path.stat().st_size},
            "row_ledger": result["row_level_ledger"]}}
    summary["content_digest"] = CORE.digest(summary); atomic_json(SCREEN.OUT / "screen_summary.json", summary)
    receipt = {"schema": "development_micro_viability_model_screen_v1_persistence_receipt",
        "generated_bytes": tree_bytes(SCREEN.OUT), "external_cache_bytes": tree_bytes(SCREEN.CACHE),
        "summary_sha256": SCREEN.sha(SCREEN.OUT / "screen_summary.json")}
    receipt["content_digest"] = CORE.digest(receipt); atomic_json(SCREEN.OUT / "persistence_receipt.json", receipt)
    print(json.dumps({"summary": summary, "persistence": receipt}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
