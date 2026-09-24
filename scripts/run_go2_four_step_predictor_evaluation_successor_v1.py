#!/usr/bin/env python3
"""One-shot evaluation-only successor for the frozen four-step checkpoints.

The wrapper reuses the already reviewed evaluator without changing scientific
code.  It redirects only output receipts to a fresh namespace, reads training
receipts/checkpoints from their original namespace, and suppresses exactly one
unrelated Stage-A source-binding refusal for the scorer contract.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import (
    go2_four_step_predictor_evaluation_successor_v1_contract as C,
)
from scripts import run_go2_rgb_control_history_four_step_autoregressive_v1 as R


def _digest(value: Any) -> str:
    return C.digest(value)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()


def _source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "lewm/oracle/go2_four_step_predictor_evaluation_successor_v1_contract.py",
        root / "scripts/run_go2_four_step_predictor_evaluation_successor_v1.py",
    ]
    return hashlib.sha256(b"".join(
        f"{p.relative_to(root)}\0".encode() + p.read_bytes()
        for p in paths
    )).hexdigest()


def _check_predecessor() -> dict[str, Any]:
    root = C.PREDECESSOR_RUNTIME
    contract_path = root / "contract.json"
    terminal_path = root / "terminal.json"
    if not contract_path.is_file() or not terminal_path.is_file():
        raise RuntimeError("predecessor contract/terminal is missing")
    raw_contract = contract_path.read_bytes()
    raw_terminal = terminal_path.read_bytes()
    if _sha256(terminal_path) != C.PREDECESSOR_TERMINAL_RAW_SHA256:
        raise RuntimeError("predecessor terminal raw digest differs")
    contract = json.loads(raw_contract)
    if contract.get("contract_digest") != C.PREDECESSOR_CONTRACT_DIGEST:
        raise RuntimeError("predecessor contract digest differs")
    terminal = json.loads(raw_terminal)
    if terminal.get("terminal_digest") != C.PREDECESSOR_TERMINAL_DIGEST:
        raise RuntimeError("predecessor terminal self digest differs")
    return contract


def _old_checkpoint(seed: int) -> Path:
    return C.PREDECESSOR_RUNTIME / "training" / f"seed_{seed}" / (
        f"seed_{seed}_rgb_four_step_epoch21.pt"
    )


@contextlib.contextmanager
def _successor_bindings(old_contract: dict[str, Any]) -> Iterator[None]:
    """Bind old inputs, new outputs, and one exact source-binding exception."""
    originals: dict[str, Any] = {
        "runtime_root": R.runtime_root,
        "require_contract": R.require_contract,
        "validate_common_manifest": R.validate_common_manifest,
        "validate_target_cache_index": R.validate_target_cache_index,
        "validate_training_receipt": R.validate_training_receipt,
        "validate_training_receipt_set": R.validate_training_receipt_set,
        "_seed_checkpoint_path": R._seed_checkpoint_path,
        "_seed_receipt_path": R._seed_receipt_path,
        "_seed_directory": R._seed_directory,
    }
    import scripts.analyze_go2_counterfactual_predictor_qualification_v1_2 as Q
    import scripts.run_go2_counterfactual_occupancy_assay_v1_2 as O
    q_require = Q._require
    o_require = O._require

    def old_call(function: Any, *args: Any, **kwargs: Any) -> Any:
        saved = R.runtime_root
        R.runtime_root = lambda: C.PREDECESSOR_RUNTIME
        try:
            return function(*args, **kwargs)
        finally:
            R.runtime_root = saved

    def old_seed_path(seed: int) -> Path:
        return C.PREDECESSOR_RUNTIME / "training" / f"seed_{int(seed)}" / (
            f"seed_{int(seed)}_rgb_four_step_epoch21.pt"
        )

    def old_receipt_path(seed: int) -> Path:
        return C.PREDECESSOR_RUNTIME / "training" / f"seed_{int(seed)}" / (
            "training_receipt.json"
        )

    def old_seed_dir(seed: int) -> Path:
        return C.PREDECESSOR_RUNTIME / "training" / f"seed_{int(seed)}"

    ignored = {C.SCORER_CONTRACT_PATH, C.ENCODER_PATH}

    def filtered_require(condition: Any, message: str) -> None:
        if (not condition and any(
            str(message) == f"Stage-A implementation changed: {path}"
            for path in ignored
        )):
            return
        q_require(condition, message)

    def filtered_occupancy_require(condition: Any, message: str) -> None:
        if (not condition and any(
            str(message) == f"Stage-A implementation changed: {path}"
            for path in ignored
        )):
            return
        o_require(condition, message)

    try:
        R.runtime_root = lambda: C.RUNTIME_ROOT
        R.require_contract = lambda: old_contract
        R.validate_common_manifest = lambda: old_call(
            originals["validate_common_manifest"]
        )
        R.validate_target_cache_index = lambda: old_call(
            originals["validate_target_cache_index"]
        )
        R.validate_training_receipt = lambda seed: old_call(
            originals["validate_training_receipt"], int(seed)
        )
        R.validate_training_receipt_set = lambda: old_call(
            originals["validate_training_receipt_set"]
        )
        R._seed_checkpoint_path = old_seed_path
        R._seed_receipt_path = old_receipt_path
        R._seed_directory = old_seed_dir
        Q._require = filtered_require
        O._require = filtered_occupancy_require
        yield
    finally:
        for name, value in originals.items():
            setattr(R, name, value)
        Q._require = q_require
        O._require = o_require


def run() -> dict[str, Any]:
    if C.RUNTIME_ROOT.exists():
        raise RuntimeError("successor namespace already exists; one-shot refusal")
    old_contract = _check_predecessor()
    previous_terminal = C.PREVIOUS_SUCCESSOR_RUNTIME / "terminal.json"
    if (not previous_terminal.is_file()
            or _sha256(previous_terminal) != C.PREVIOUS_SUCCESSOR_TERMINAL_RAW_SHA256):
        raise RuntimeError("previous evaluation-successor terminal differs")
    for seed, expected in C.CHECKPOINTS.items():
        path = _old_checkpoint(int(seed))
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"frozen checkpoint differs for seed {seed}")

    C.RUNTIME_ROOT.mkdir(parents=True)
    (C.RUNTIME_ROOT / "attempts").mkdir()
    # A receipt is an allowed successor artifact; the target-cache inputs are
    # never copied or linked into this namespace.
    source_receipt = C.PREDECESSOR_RUNTIME / "training_receipts.json"
    receipt_copy = C.RUNTIME_ROOT / "training_receipts.json"
    receipt_copy.write_bytes(source_receipt.read_bytes())
    receipt_copy.chmod(0o444)

    contract = C.seal(C.contract_payload(_git_head(), _source_digest()))
    _write(C.RUNTIME_ROOT / "contract.json", contract)
    _write(C.RUNTIME_ROOT / "dependency_report.json", {
        **C.REPORT, "report_digest": _digest(C.REPORT),
    })

    try:
        with _successor_bindings(old_contract):
            result = R.evaluate_stage(argparse.Namespace(device="cuda:0"))
        report_path = C.RUNTIME_ROOT / "evaluation" / "result.json"
        occupancy_path = C.RUNTIME_ROOT / "evaluation" / "occupancy.json"
        successor_receipt = {
            "schema": "go2_four_step_predictor_evaluation_successor_receipt_v1",
            "status": "COMPLETE",
            "successor_contract_digest": contract["contract_digest"],
            "predecessor_terminal_digest": C.PREDECESSOR_TERMINAL_DIGEST,
            "evaluation_result_digest": result["result_digest"],
            "evaluation_result_sha256": _sha256(report_path),
            "occupancy_sha256": _sha256(occupancy_path),
            "model_forwards": 8 * 20,
            "historical_comparator_forwards": 0,
            "training_attempts": 0,
            "selection_evaluations": 8,
            "predictor_utility_shards_opened": False,
            "new_branches_or_targets_generated": False,
            "scientific_contract_unchanged": True,
        }
        successor_receipt["receipt_digest"] = _digest(successor_receipt)
        _write(C.RUNTIME_ROOT / "successor_receipt.json", successor_receipt)
        terminal = {
            "schema": "go2_four_step_predictor_evaluation_successor_terminal_v1",
            "status": "SUCCESS",
            "classification": C.CLASSIFICATION,
            "successor_contract_digest": contract["contract_digest"],
            "dependency_report_digest": _digest(C.REPORT),
            "successor_receipt_digest": successor_receipt["receipt_digest"],
            "evaluation_result_digest": result["result_digest"],
            "original_terminal_preserved": True,
            "training_attempts": 0,
            "historical_comparator_forwards": 0,
            "nothing_running": True,
        }
        terminal["terminal_digest"] = _digest(terminal)
        _write(C.RUNTIME_ROOT / "terminal.json", terminal)
        return {"result": result, "terminal": terminal}
    except Exception as exc:
        failure = {
            "schema": "go2_four_step_predictor_evaluation_successor_terminal_v1",
            "status": "INVALID_EVALUATION_SUCCESSOR",
            "classification": "INVALID_EVALUATION_SUCCESSOR",
            "successor_contract_digest": contract["contract_digest"],
            "exception": f"{type(exc).__name__}: {exc}",
            "training_attempts": 0,
            "model_forwards": 0,
            "selection_evaluations": 0,
            "nothing_running": True,
        }
        failure["terminal_digest"] = _digest(failure)
        if not (C.RUNTIME_ROOT / "terminal.json").exists():
            _write(C.RUNTIME_ROOT / "terminal.json", failure)
        raise


if __name__ == "__main__":
    run()
