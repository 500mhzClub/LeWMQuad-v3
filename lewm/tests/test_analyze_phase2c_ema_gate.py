from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_jepa_phase2c_ema_gate import analyze


def _report(*, ratio: float, ema: bool) -> dict:
    step = {
        "free_running_vs_persistence_mse_ratio": ratio,
        "real_action_beats_zero": True,
        "real_action_beats_shuffled": True,
        "meaningful_real_action_beats_zero": True,
        "meaningful_real_action_beats_shuffled": True,
        "free_running_beats_persistence": ratio < 1.0,
    }
    return {
        "trainable_parameters": 10,
        "scene_overlap": [],
        "train_input_audit": {"usable_complete_valid_sequences": 12},
        "eval_input_audit": {"usable_complete_valid_sequences": 8},
        "target_encoder": {
            "mode": "stop_gradient_ema" if ema else "online_joint",
        },
        "final": {
            "eval": {
                "per_horizon_step": [step, step],
                "representation": {
                    "collapse_warning": False,
                    "mean_feature_std": 0.5,
                },
                "selection": {
                    "safe_positive_progress_rate": 0.5,
                    "selected_enters_grid_unsafe_rate": 0.1,
                },
            }
        },
    }


def test_phase2c_analysis_promotes_passing_ema_target(tmp_path: Path) -> None:
    phase2b_root = tmp_path / "phase2b"
    phase2b_root.mkdir()
    (phase2b_root / "spatial_var.json").write_text(
        json.dumps(_report(ratio=2.0, ema=False))
    )
    phase2c_report = tmp_path / "ema.json"
    phase2c_report.write_text(json.dumps(_report(ratio=0.8, ema=True)))

    report = analyze(phase2b_root, phase2c_report)

    assert report["passes_all_gates"]
    assert report["decision"] == "retain_ema_target_and_replicate"
