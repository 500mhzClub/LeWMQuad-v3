#!/usr/bin/env python3
"""Permanently freeze the rollout arm at its selected converged checkpoint.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only over the arm; writes only a
receipt.  The rollout arm is NOT resumed after this.

Records the checkpoint SHA-256, the selection receipt, the convergence windows,
the hashes of every script that produced or evaluated it, and the complete
one-step and two-step battery at the selected epoch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
TWO = CACHE / "two_step"
ARM = TWO / "arms" / "arm_rollout"
EVALDIR = TWO / "evaluation"
OUT = TWO / "rollout_frozen"

SOURCES = [
    "scripts/run_dev_v03_two_step_rollout_v1.py",
    "scripts/eval_dev_v03_two_step_rollout_v1.py",
    "scripts/aggregate_dev_v03_two_step_decision_v1.py",
    "scripts/build_dev_v03_two_step_sequences_v1.py",
    "scripts/run_dev_v03_temporal_action_jepa_v1.py",
    "scripts/dev_checkpoint_v1.py",
    "scripts/dev_frozen_dense_representation_encoders_v1.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    decision = json.loads((EVALDIR / "decision.json").read_text())
    evaluation = json.loads((EVALDIR / "result.json").read_text())
    training = json.loads((ARM / "result.json").read_text())

    selected = decision["checkpoint_selection"]["rollout"]["selected_epoch"]
    if selected is None:
        raise RuntimeError("rollout has no selected checkpoint; nothing to freeze")
    checkpoint = ARM / f"checkpoint_epoch{selected}.pt"
    battery = decision["both_arms_at_each_selected_epoch"]["epochs"][str(selected)]["rollout"]
    curve = next(e for e in evaluation["curves"]["rollout"] if e["epoch"] == selected)

    receipt = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING", "claim_bearing": False,
        "frozen": True,
        "do_not_resume": (
            "this arm is permanently frozen at its selected converged checkpoint; "
            "it must not be trained further"
        ),
        "arm": "rollout-supervision bundle, 1.5*e1 + 0.5*e2, fixed sliding context",
        "selected_epoch": selected,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "selection_receipt": decision["checkpoint_selection"]["rollout"],
        "selection_rule": (
            "within epochs 21-23, highest step-one occupied IoU that ALSO beats matched "
            "persistence, has correct-minus-shuffled margin >= +0.0586, beats "
            "open_obstacle_field persistence, and passes occupied-volume calibration; "
            "fixed before the resumed epochs were read"
        ),
        "convergence": decision["convergence"]["per_arm"]["rollout"],
        "convergence_rule": decision["convergence"]["rule"],
        "convergence_windows": {"middle": [18, 19, 20], "late": [21, 22, 23]},
        "training_record": {
            "epochs": training["epochs"],
            "initial_weight_sha256": training["predictor"]["initial_weight_sha256"],
            "seed": training["schedule"]["seed"],
            "predictor": training["predictor"],
            "loss_reduction": training["loss_reduction"],
            "step2_context": training["step2_context"],
            "rollout_gradient_assertion": training.get("rollout_gradient_assertion"),
        },
        "battery_at_selected_epoch": battery,
        "curve_at_selected_epoch": curve,
        "reference": {
            "persistence_step1": evaluation["reference"]["persistence_step1_spatial"],
            "true_future_step1": evaluation["reference"]["true_future_step1_spatial"],
            "persistence_step2_latent": evaluation["reference"]["persistence_step2_latent"],
        },
        "masks": evaluation["masks"],
        "comparison_contract": evaluation["comparison_contract"],
        "evaluator_and_runner_hashes": {
            s: sha256(ROOT / s) for s in SOURCES if (ROOT / s).is_file()
        },
        "all_epoch_checkpoint_sha256": {
            p.name: sha256(p) for p in sorted(ARM.glob("checkpoint_epoch*.pt"),
                                              key=lambda q: int(q.stem.split("epoch")[1]))
        },
    }
    (OUT / "frozen_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({
        "selected_epoch": selected,
        "checkpoint_sha256": receipt["checkpoint"]["sha256"],
        "bytes": receipt["checkpoint"]["bytes"],
        "converged": receipt["convergence"]["converged"],
        "late_minus_middle_iou": receipt["convergence"]["late_minus_middle_iou"],
        "abs_margin_change": receipt["convergence"]["abs_margin_change"],
        "one_step_gate": receipt["selection_receipt"]["one_step_gate"],
        "evaluator_hashes_recorded": len(receipt["evaluator_and_runner_hashes"]),
        "checkpoints_hashed": len(receipt["all_epoch_checkpoint_sha256"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
