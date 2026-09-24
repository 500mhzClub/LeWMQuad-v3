from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/create_go2_generalization_benchmark.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("create_generalization_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_id_extraction_uses_shard_name_and_result_provenance(
    tmp_path: Path,
) -> None:
    module = _load_script()
    shard = tmp_path / "medium_enclosed_maze_train123.npz"
    shard.write_bytes(b"not opened")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "provenance": {
                    "argv": ["benchmark.py", "--scene-id", "maze_dev456"]
                }
            }
        )
    )

    assert module.scene_ids_from_paths((shard, result)) == {
        "medium_enclosed_maze_train123",
        "maze_dev456",
    }


def test_task_contract_rejects_duplicate_color_landmarks() -> None:
    module = _load_script()

    def landmark(color: str, index: int):
        return SimpleNamespace(
            material_id=f"landmark_{color}",
            object_id=f"landmark_{index}_{color}",
        )

    valid = SimpleNamespace(
        scene_id="valid",
        landmarks=[landmark(color, index) for index, color in enumerate(
            ("red", "blue", "green", "yellow")
        )],
    )
    invalid = SimpleNamespace(
        scene_id="six_target",
        landmarks=[
            landmark(color, index)
            for index, color in enumerate(
                ("red", "blue", "green", "yellow", "red", "blue")
            )
        ],
    )

    module.validate_task_landmarks(valid, ("red", "yellow", "blue", "green"))
    with pytest.raises(ValueError, match="exactly-one-per-color"):
        module.validate_task_landmarks(
            invalid,
            ("red", "yellow", "blue", "green"),
        )
