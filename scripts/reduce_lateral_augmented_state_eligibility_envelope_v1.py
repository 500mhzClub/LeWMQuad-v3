#!/usr/bin/env python3
"""Persist replay-free evidence and digests for the augmented eligibility run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".generated/lateral_augmented_state_eligibility_envelope_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/lateral_augmented_state_eligibility_envelope_v1"


def read(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    selection = read(OUT / "frozen_selection.json")
    fixture = read(OUT / "fixture.json")
    result = read(OUT / "result.json")
    residuals = [read(path) for path in sorted((OUT / "residual_states").glob("*.json"))]
    controls = [read(path) for path in sorted((OUT / "matched_controls").glob("*.json"))]
    CACHE.mkdir(parents=True, exist_ok=True)
    ledger = CACHE / "row_level_evidence_v1.jsonl"
    with ledger.open("w") as stream:
        stream.write(json.dumps({"record_type": "selection", **selection}, sort_keys=True) + "\n")
        stream.write(json.dumps({"record_type": "fixture", **fixture}, sort_keys=True) + "\n")
        for record in residuals:
            summary = {key: value for key, value in record.items() if key not in {"predecessors", "recovery_rollout"}}
            stream.write(json.dumps({"record_type": "residual_summary", **summary}, sort_keys=True) + "\n")
            for predecessor in record["predecessors"]:
                stream.write(json.dumps({
                    "record_type": "predecessor_boundary", "state_id": record["state_id"],
                    **predecessor,
                }, sort_keys=True) + "\n")
            if record["recovery_rollout"] is not None:
                stream.write(json.dumps({
                    "record_type": "recovery_rollout", "state_id": record["state_id"],
                    **record["recovery_rollout"],
                }, sort_keys=True) + "\n")
        for record in controls:
            stream.write(json.dumps({"record_type": "matched_control_rollout", **record}, sort_keys=True) + "\n")
        stream.write(json.dumps({"record_type": "aggregate_result", **result}, sort_keys=True) + "\n")

    evidence = [
        OUT / "frozen_selection.json", OUT / "fixture.json", OUT / "collection_receipt.json",
        OUT / "result.json", OUT / "index.json", ledger,
    ]
    digests = {
        str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in evidence
    }
    write(CACHE / "content_digests.json", digests)
    receipt = {
        "schema": "lateral_augmented_state_eligibility_persistence_receipt_v1",
        "row_ledger": str(ledger), "row_ledger_sha256": sha256(ledger),
        "row_ledger_bytes": ledger.stat().st_size,
        "records": 2 + len(residuals) + sum(len(row["predecessors"]) for row in residuals)
                   + sum(row["recovery_rollout"] is not None for row in residuals) + len(controls) + 1,
        "primary_classification": result["primary_classification"],
        "content_digests": str(CACHE / "content_digests.json"),
    }
    write(OUT / "persistence_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
