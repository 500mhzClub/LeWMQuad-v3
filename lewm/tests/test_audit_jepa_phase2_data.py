from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_jepa_phase2_data import build_audit


def _write_dataset(path: Path, *, split: str, scene: str) -> None:
    rows = []
    for candidate_index in range(81):
        first_action = candidate_index // 9
        second_action = candidate_index % 9
        rows.append(
            {
                "scene_id": scene,
                "family": "family",
                "split": split,
                "source_index": 0,
                "start_frame": "start.png",
                "primitive_sequence": [
                    f"action_{first_action}",
                    f"action_{second_action}",
                ],
                "active_blocks": [[float(first_action)], [float(second_action)]],
                "future_frames": [
                    f"future_{candidate_index}_0.png",
                    f"future_{candidate_index}_1.png",
                ],
                "future_observations": [
                    {
                        "rgb_path": f"future_{candidate_index}_0.png",
                        "observation_valid": True,
                    },
                    {
                        "rgb_path": f"future_{candidate_index}_1.png",
                        "observation_valid": True,
                    },
                ],
                "complete_valid_future_sequence": True,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_phase2_audit_separates_preliminary_checks_from_registered_gate(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    _write_dataset(train, split="train", scene="train_scene")
    _write_dataset(test, split="test", scene="test_scene")

    report = build_audit({"train": train, "test": test}, legacy_batch_size=8)

    assert report["foundation_gate_passed"]
    assert all(report["confirmatory_data_checks"].values())
    assert not report["confirmatory_data_gate_passed"]
    assert not report["registered_confirmatory_gate"]["checks"][
        "all_required_splits_present"
    ]
    assert not report["registered_confirmatory_gate"]["checks"][
        "artifact_and_seed_lineage_verified"
    ]
    assert report["datasets"]["train"]["full_81_candidate_sources"] == 1
