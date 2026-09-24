#!/usr/bin/env python3
"""Persist the terminal pre-training panel-adequacy result for V1.

This reducer is deliberately simulation- and model-free.  It consumes the
already materialised oracle-tree index, validates its bound shards, and emits
row-level evidence plus a compact terminal result.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.safety import lightweight_one_tick_viability_model_v1 as core


OUT = ROOT / ".generated" / "lightweight_one_tick_viability_model_and_interface_v1"
CACHE = Path.home() / ".cache" / "lewm_go2_temporal_v03" / "lightweight_one_tick_viability_model_and_interface_v1"
INDEX = OUT / "oracle_tree_index.json"
MANIFEST = OUT / "panel_manifest.json"


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    index = json.loads(INDEX.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if index["manifest_digest"] != manifest["content_digest"]:
        raise RuntimeError("manifest binding mismatch")
    if index["panel_adequate"] or index["classification"] != "FRESH_MICRO_VIABILITY_PANEL_INADEQUATE":
        raise RuntimeError("this reducer is only valid for the frozen inadequate-panel terminal")

    fixture = core.fixture_payload()
    if not fixture["pass"]:
        raise RuntimeError("evaluator fixture failed")
    atomic_json(OUT / "evaluator_fixture.json", fixture)

    rows: list[dict] = []
    for state in index["records"]:
        shard = Path(state["shard_path"])
        if not shard.is_file() or sha256(shard) != state["shard_sha256"]:
            raise RuntimeError(f"invalid state shard: {state['state_id']}")
        for candidate in state["candidates"]:
            rows.append({
                "state_id": state["state_id"],
                "scene_id": state["scene_id"],
                "family": state["family"],
                "role": state["role"],
                "source_kind": state["source_kind"],
                "input_shard_path": str(shard),
                "input_shard_sha256": state["shard_sha256"],
                **candidate,
            })
    ledger = CACHE / "row_level_evidence_v1.jsonl"
    atomic_bytes(ledger, b"".join(canonical(row) for row in rows))

    result = {
        "schema": "lightweight_one_tick_viability_model_and_interface_v1_result",
        "status": "completed_at_preregistered_pretraining_stop",
        "source_commit": core.SOURCE_COMMIT,
        "primary_classification": "FRESH_MICRO_VIABILITY_PANEL_INADEQUATE",
        "claims_boundary": "SIMULATED_ONE_TICK_CONTACT_AND_SUCCESSOR_VIABILITY development evidence only",
        "panel_manifest_sha256": sha256(MANIFEST),
        "oracle_tree_index_sha256": sha256(INDEX),
        "panel_manifest_digest": manifest["content_digest"],
        "inventory": index["inventory"],
        "per_family_inventory": index["per_family_inventory"],
        "panel_adequacy": index["panel_adequacy"],
        "generated_branches": index["generated_branches"],
        "collection_runtime": {
            **index["runtime"],
            "parallel_processes_scope": "parent collector",
            "additional_disjoint_helper_processes": 8,
            "maximum_concurrent_state_processes": 12,
        },
        "model_training": {"ran": False, "seed_count": 0, "reason": "fresh panel failed Section 9 adequacy gate"},
        "calibration": {"ran": False},
        "heldout_model_evaluation": {"ran": False},
        "micro_interface_benchmark": {"ran": False, "reason": "no qualified offline model exists"},
        "learned_closed_loop": {"ran": False, "reason": "offline model and interface gates were not reached"},
        "next_decision": "Freeze a new prospective panel design that yields at least 22/24 oracle-viable states per split and contact plus nonviability examples in every family; do not replace or reuse these observed evaluation states.",
        "row_level_ledger": {"path": str(ledger), "rows": len(rows), "bytes": ledger.stat().st_size, "sha256": sha256(ledger)},
        "preserved_classifications": [
            "LATERAL_AUGMENTED_STATE_ELIGIBILITY_SIGNAL",
            "LATERAL_RECOVERY_CONTROLLER_QUALIFIED",
            "LATERAL_CONTROLLER_SIGNAL_VIABILITY_NO_GO",
            "ONE_TICK_FULL_JEPA_COMPUTE_NO_GO",
            "TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED",
            "REPLANNING_INTERFACE_UNRESOLVED",
            "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
        ],
    }
    result["content_digest"] = core.digest(result)
    atomic_json(OUT / "result.json", result)

    receipt = {
        "schema": "lightweight_one_tick_viability_model_and_interface_v1_persistence_receipt",
        "result_sha256": sha256(OUT / "result.json"),
        "fixture_sha256": sha256(OUT / "evaluator_fixture.json"),
        "manifest_sha256": sha256(MANIFEST),
        "oracle_tree_index_sha256": sha256(INDEX),
        "row_level_ledger": result["row_level_ledger"],
        "generated_tree_bytes": tree_bytes(OUT),
        "external_cache_bytes": tree_bytes(CACHE),
    }
    receipt["content_digest"] = core.digest(receipt)
    atomic_json(OUT / "persistence_receipt.json", receipt)
    print(json.dumps({"result": result, "persistence": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
