from __future__ import annotations

from pathlib import Path

from scripts.render_jepa_counterfactual_plan_root import _output_dir


def test_counterfactual_render_root_preserves_scene_relative_path(tmp_path: Path) -> None:
    plan_root = tmp_path / "plans"
    output_root = tmp_path / "renders"
    plan = plan_root / "train" / "family" / "scene" / "render_replay_plan.json"

    assert _output_dir(plan_root, output_root, plan) == (
        output_root / "train" / "family" / "scene"
    )
