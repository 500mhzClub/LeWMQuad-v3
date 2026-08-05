from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import build_go2_scene_diversity_recurrent_replication_plan_v1 as builder


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _predecessor_panel() -> dict:
    scenes = []
    for role in builder.ROLE_NAMES:
        for family in pilot.FAMILIES:
            for slot in range(2):
                scenes.append(
                    {
                        "role": role,
                        "family": family,
                        "scene_id": f"predecessor-{role}-{family}-{slot}",
                    }
                )
    return {
        "schema": builder.predecessor_panel_builder.PANEL_SCHEMA,
        "scenes": scenes,
    }


@pytest.fixture
def derived_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    campaign = tmp_path / "ordinary-campaign"
    campaign.mkdir()
    monkeypatch.setattr(
        builder.predecessor_panel_builder, "SCENE_CORPUS_ROOT", tmp_path
    )

    exclusions = {
        "schema": "lewm_go2_world_model_bounded_branch_scene_exclusions_v1",
        "scene_ids": [f"base-excluded-{family}" for family in pilot.FAMILIES],
    }
    predecessor = _predecessor_panel()
    exclusions_path = tmp_path / "base-exclusions.json"
    predecessor_path = tmp_path / "predecessor-panel.json"
    exclusions_path.write_text(json.dumps(exclusions) + "\n")
    predecessor_path.write_text(json.dumps(predecessor) + "\n")

    texture_paths = {}
    for category in ("floor", "wall", "obstacle"):
        path = tmp_path / f"{category}.png"
        path.write_bytes(category.encode("utf-8"))
        texture_paths[category] = path

    inventory = []
    for family in pilot.FAMILIES:
        excluded_ids = (
            f"base-excluded-{family}",
            f"predecessor-train-{family}-0",
        )
        for scene_id in excluded_ids:
            inventory.append(
                {
                    "family": family,
                    "scene_id": scene_id,
                    "manifest_sha256": _digest(f"manifest/{scene_id}"),
                    "campaign_root": str(campaign),
                    "relative_dir": f"train/{family}/{scene_id}",
                    "inventory_rank": _digest(f"inventory/{scene_id}"),
                }
            )
        for index in range(8):
            scene_id = f"{family}-fresh-{index}"
            manifest_digest = _digest(f"manifest/{scene_id}")
            scene_root = campaign / "train" / family / scene_id
            scene_root.mkdir(parents=True)
            (scene_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "scene_id": scene_id,
                        "family": family,
                        "split": "train",
                        "manifest_sha256": manifest_digest,
                        "visual_seed": index,
                        "landmarks": [
                            {
                                "object_id": "target-0",
                                "center_xyz_m": [float(index), 1.0, 0.0],
                            }
                        ],
                    }
                )
                + "\n"
            )
            (scene_root / "genesis_scene.json").write_text("{}\n")
            inventory.append(
                {
                    "family": family,
                    "scene_id": scene_id,
                    "manifest_sha256": manifest_digest,
                    "campaign_root": str(campaign),
                    "relative_dir": f"train/{family}/{scene_id}",
                    "inventory_rank": _digest(f"inventory/{scene_id}"),
                }
            )

    current_inventory = list(inventory)
    monkeypatch.setattr(
        builder.predecessor_panel_builder,
        "_load_inventory",
        lambda: ([], list(current_inventory)),
    )
    monkeypatch.setattr(
        builder.predecessor_panel_builder,
        "_selected_texture_asset_bindings",
        lambda _manifest: {
            category: pilot.file_binding(path)
            for category, path in texture_paths.items()
        },
    )
    panel = builder.derive_scene_panel_v1(
        base_exclusions=exclusions,
        base_exclusions_binding=pilot.file_binding(exclusions_path),
        predecessor_panel=predecessor,
        predecessor_panel_binding=pilot.file_binding(predecessor_path),
    )
    current_inventory.reverse()
    repeated = builder.derive_scene_panel_v1(
        base_exclusions=exclusions,
        base_exclusions_binding=pilot.file_binding(exclusions_path),
        predecessor_panel=predecessor,
        predecessor_panel_binding=pilot.file_binding(predecessor_path),
    )
    assert panel == repeated
    return panel


def test_panel_is_fresh_balanced_and_history_complete(derived_panel: dict) -> None:
    scenes = derived_panel["scenes"]
    assert derived_panel["selection_contract"]["seed"] == 20260804
    assert len(scenes) == 64
    assert len({scene["scene_id"] for scene in scenes}) == 64
    assert not any("base-excluded" in scene["scene_id"] for scene in scenes)
    assert not any("predecessor" in scene["scene_id"] for scene in scenes)

    scene_balance = Counter((scene["role"], scene["family"]) for scene in scenes)
    history_balance = Counter()
    for scene in scenes:
        assert len(scene["states"]) == 4
        for state in scene["states"]:
            history_balance[
                (
                    scene["role"],
                    scene["family"],
                    tuple(state["history_action_ids"]),
                )
            ] += 1
    for role in builder.ROLE_NAMES:
        for family in pilot.FAMILIES:
            assert scene_balance[(role, family)] == 4
            for history in builder.HISTORY_PANEL:
                assert history_balance[(role, family, tuple(history))] == 2


def test_plan_changes_only_identity_states_and_counts(derived_panel: dict) -> None:
    base_plan = json.loads(builder.FROZEN_BASE_PLAN.read_text())
    output_root = (
        builder.REPO_ROOT
        / ".generated/dev"
        / f"scene-diversity-plan-test-{uuid.uuid4().hex}"
    )
    plan = builder.build_plan_v1(
        base_plan=base_plan,
        attempt_id=builder.DEFAULT_ATTEMPT_ID,
        output_root=output_root,
        scene_panel=derived_panel,
    )
    assert plan == pilot.validate_plan(plan)
    assert plan["states_per_scene"] == 4
    assert plan["expected_counts"] == {
        "scenes": 64,
        "states": 256,
        "roles": {"eval": 128, "train": 128},
        "actions": 9,
        "candidate_branches": 2304,
        "sentinel_branches": 0,
        "total_branches": 2304,
        "context_frames": 768,
        "target_frames": 2304,
    }
    mutable = {
        "attempt_id",
        "output_root",
        "states_per_scene",
        "states",
        "expected_counts",
    }
    assert all(plan[key] == base_plan[key] for key in set(base_plan) - mutable)
    assert all(state["candidate_action_ids"] == list(range(9)) for state in plan["states"])


def test_history_tamper_is_rejected(derived_panel: dict) -> None:
    changed = copy.deepcopy(derived_panel)
    changed["scenes"][0]["states"][0]["history_action_ids"] = [0, 0]
    base_plan = json.loads(builder.FROZEN_BASE_PLAN.read_text())
    output_root = (
        builder.REPO_ROOT
        / ".generated/dev"
        / f"scene-diversity-plan-test-{uuid.uuid4().hex}"
    )
    with pytest.raises(builder.SceneDiversityPlanError, match="history/target"):
        builder.build_plan_v1(
            base_plan=base_plan,
            attempt_id=builder.DEFAULT_ATTEMPT_ID,
            output_root=output_root,
            scene_panel=changed,
        )
