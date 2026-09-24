#!/usr/bin/env python3
"""Qualify the frozen controller's lateral authority before scientific use."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "lewm_genesis"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import h1_candidate_bank_viability_successor_v1 as SUBJECT
from lewm_genesis.lewm_contract import SafetyLimits


SOURCE_COMMIT = "4b655f054ffa1e7322d81a78a7920e260a8283bd"
MANIFEST = ROOT / "config/go2_platform_manifest.yaml"
REGISTRY = ROOT / "config/go2_primitive_registry.yaml"
POLICY_CFG = ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl"
OUT = ROOT / ".generated/h1_candidate_bank_viability_successor_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/h1_candidate_bank_viability_successor_v1"
RESULT = OUT / "controller_authority_result.json"
LEDGER = CACHE / "training_fixture_command_adapter_ledger_v1.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    started = time.perf_counter()
    manifest = yaml.safe_load(MANIFEST.read_text())
    registry = yaml.safe_load(REGISTRY.read_text())
    with POLICY_CFG.open("rb") as stream:
        configs = pickle.load(stream)
    policy_command_config = configs[3]
    audit = SUBJECT.controller_authority_audit(manifest, policy_command_config, registry)
    limits = SafetyLimits.from_manifest(manifest)
    rows = SUBJECT.command_adapter_fixture_rows(limits)
    fixture = SUBJECT.fixture_reduction(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    classification = "LATERAL_RETREAT_CONTROLLER_AUTHORITY_NO_GO"
    if audit["nonzero_lateral_controller_supported"] or fixture["qualification_pass"]:
        raise RuntimeError("frozen no-authority result unexpectedly changed")
    result = {
        "schema": "h1_candidate_bank_viability_successor_v1_controller_authority_result",
        "source_commit": SOURCE_COMMIT,
        "primary_classification": classification,
        "claim_boundary": "development-only simulated micro-viability mechanism qualification",
        "controller_audit": audit,
        "qualification_probe": {
            "registry_magnitude_m_s": SUBJECT.PROBE_MAGNITUDE_M_S,
            "scientific_magnitude_frozen": None,
            "reason": "no nonzero lateral magnitude is supported by the frozen manifest or policy training distribution",
        },
        "training_fixture_adapter_gate": fixture,
        "scientific_execution": {
            "entered": False,
            "frozen_states_touched": 0,
            "simulator_branches_generated": 0,
            "current_state_lateral_branches": 0,
            "successor_lateral_branches": 0,
            "multi_cycle_rollout_branches": 0,
            "stop_reason": "applied vy is deterministically zero before environment dynamics",
        },
        "bank_availability": {
            "before_states_with_viability_action": 40,
            "before_total_states": 48,
            "after": "not_evaluated_no_operational_augmentation",
            "operational_bank_remains_historical_twelve": True,
            "classification": "CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO",
        },
        "preserved_results": [
            "STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION",
            "CANDIDATE_BANK_MULTI_CYCLE_VIABILITY_NO_GO",
            "ONE_TICK_VIABILITY_KERNEL_NO_GO",
            "ONE_TICK_FULL_JEPA_COMPUTE_NO_GO",
            "TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED",
            "REPLANNING_INTERFACE_UNRESOLVED",
            "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
        ],
        "predictor_contract": {
            "jepa_opened": False,
            "historical_action_contract_modified": False,
            "lateral_vy_qualified": False,
            "consequence": "lateral retreat cannot enter either micro execution or macro JEPA scoring under the frozen controller",
        },
        "next_implementation": {
            "name": "DEPLOYMENT_VALID_LATERAL_LOCOMOTION_CONTROLLER_V1",
            "requirements": [
                "prospectively train or bind a controller whose command distribution includes mirrored nonzero vy",
                "qualify tracking, stability, torque, contact, and deterministic command application on training-only fixtures",
                "update the platform safety envelope only after that qualification",
                "rerun H1_CANDIDATE_BANK_VIABILITY_SUCCESSOR_V1 without changing the historical macro bank",
            ],
            "not_authorized_in_this_pass": True,
        },
        "bindings": {
            "platform_manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": sha256(MANIFEST)},
            "primitive_registry": {"path": str(REGISTRY.relative_to(ROOT)), "sha256": sha256(REGISTRY)},
            "policy_config": {"path": str(POLICY_CFG.relative_to(ROOT)), "sha256": sha256(POLICY_CFG)},
            "policy_checkpoint_opened": False,
        },
        "runtime_s": time.perf_counter() - started,
        "storage": {
            "fixture_ledger_path": str(LEDGER),
            "fixture_ledger_rows": len(rows),
            "fixture_ledger_bytes": LEDGER.stat().st_size,
            "fixture_ledger_sha256": sha256(LEDGER),
        },
        "prohibitions_observed": {
            "model_training": False,
            "learned_model_execution": False,
            "jepa_access": False,
            "predictor_contract_modification": False,
            "scientific_state_execution": False,
            "memory": False,
            "navigation": False,
        },
    }
    result["content_digest"] = SUBJECT.digest(result)
    atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
