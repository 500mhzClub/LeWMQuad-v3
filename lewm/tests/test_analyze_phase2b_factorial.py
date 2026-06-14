from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_jepa_phase2b_factorial import analyze


def _report(*, pooled: bool, collapsed: bool = False) -> dict:
    ratio = 0.9 if pooled else 0.8
    return {
        "trainable_parameters": 100,
        "scene_overlap": [],
        "train_input_audit": {"usable_complete_valid_sequences": 20},
        "eval_input_audit": {"usable_complete_valid_sequences": 20},
        "final": {
            "eval": {
                "representation": {
                    "collapse_warning": collapsed,
                    "mean_feature_std": 1.0,
                },
                "per_horizon_step": [
                    {
                        "real_action_beats_zero": True,
                        "real_action_beats_shuffled": True,
                        "meaningful_real_action_beats_zero": True,
                        "meaningful_real_action_beats_shuffled": True,
                        "free_running_beats_persistence": True,
                        "free_running_vs_persistence_mse_ratio": ratio,
                    },
                    {"free_running_vs_persistence_mse_ratio": ratio},
                ],
                "selection": {
                    "safe_positive_progress_rate": 0.5,
                    "selected_enters_grid_unsafe_rate": 0.1,
                },
            }
        },
    }


def test_phase2b_analysis_promotes_spatial_cell_that_passes_gates(tmp_path: Path) -> None:
    for name in ("pooled", "spatial_var", "spatial_no_var"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(_report(pooled=name == "pooled"))
        )

    report = analyze(tmp_path)

    assert report["passes_all_gates"]
    assert report["decision"] == "promote_to_full_capacity_replication"
