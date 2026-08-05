from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import collect_go2_scene_diversity_recurrent_replication_v1 as collector


def _states() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    group_index = 0
    for role in ("train", "eval"):
        for family in pilot.FAMILIES:
            for scene_slot in range(4):
                panel_indices = (0, 2, 4, 6) if scene_slot % 2 == 0 else (1, 3, 5, 7)
                scene_id = f"scene-diversity-{role}-{family}-{scene_slot}"
                scene_dir = Path("/synthetic/nonprotected") / scene_id
                for state_index, panel_index in enumerate(panel_indices):
                    result.append(
                        {
                            "state_id": f"state-{group_index:04d}",
                            "role": role,
                            "family": family,
                            "scene_id": scene_id,
                            "scene_manifest_binding": {
                                "path": str(scene_dir / "manifest.json"),
                                "file_sha256": "1" * 64,
                                "byte_count": 1,
                            },
                            "scene_genesis_binding": {
                                "path": str(scene_dir / "genesis_scene.json"),
                                "file_sha256": "2" * 64,
                                "byte_count": 1,
                            },
                            "scene_generation": None,
                            "group_index": group_index,
                            "state_index_in_scene": state_index,
                            "history_action_ids": list(
                                collector.EXPECTED_HISTORY_PANEL[panel_index]
                            ),
                            "candidate_action_ids": list(range(9)),
                            "sentinel_duplicate_action_id": None,
                            "target_xy_m": [1.0, 0.0],
                        }
                    )
                    group_index += 1
    return result


def _plan(*, output_root: Path) -> dict[str, object]:
    return {
        "attempt_id": "scene-diversity-recurrent-replication-v1",
        "purpose": "bounded_wm_a_pilot",
        "branch_mechanism": pilot.BRANCH_MECHANISM,
        "states_per_scene": 4,
        "history_blocks": 2,
        "output_root": str(output_root),
        "states": _states(),
        "expected_counts": copy.deepcopy(collector.EXPECTED_COUNTS),
        "action_catalog": [{"action_id": value} for value in range(9)],
        "runtime_bindings": {},
        "execution_contract": {},
    }


def _authority(
    *, attempt_root: Path, collection_root: Path, plan_binding: dict[str, object]
) -> dict[str, object]:
    return {
        "schema": collector.AUTHORITY_SCHEMA,
        "status": collector.AUTHORITY_STATUS,
        "attempt_id": "scene-diversity-recurrent-replication-v1",
        "attempt_root": str(attempt_root),
        "collection_root": str(collection_root),
        "plan_binding": collector._standard_binding(plan_binding),  # noqa: SLF001
        "preregistration_binding": {"bound": True},
        "source_review_binding": {"bound": True},
        "source_bindings": {"collector": {"bound": True}},
        "dino": {},
        "config": {},
        "caps": copy.deepcopy(collector.EXPECTED_CAPS),
        "permissions": copy.deepcopy(collector.EXPECTED_PERMISSIONS),
    }


def test_exact_64_scene_four_state_history_panel_is_accepted(tmp_path: Path) -> None:
    plan = _plan(output_root=tmp_path / "collection")
    normalized = collector._validate_scene_diversity_plan_v1(plan)  # noqa: SLF001
    assert normalized["expected_counts"] == collector.EXPECTED_COUNTS

    histories = copy.deepcopy(plan)
    histories["states"][0]["history_action_ids"] = [0, 0]
    with pytest.raises(pilot.PilotContractError, match="history tape"):
        collector._validate_scene_diversity_plan_v1(histories)  # noqa: SLF001

    scenes = copy.deepcopy(plan)
    scenes["states"][0]["scene_id"] = "foreign-fifth-scene"
    with pytest.raises(pilot.PilotContractError, match="four scenes"):
        collector._validate_scene_diversity_plan_v1(scenes)  # noqa: SLF001


def test_authority_is_exact_and_collection_root_must_be_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path)
    attempt_root = tmp_path / ".generated/dev/attempt-v1"
    attempt_root.mkdir(parents=True)
    collection_root = attempt_root / "collection"
    plan = _plan(output_root=collection_root)
    plan_binding = {
        "path": str(tmp_path / "plan.json"),
        "file_sha256": "a" * 64,
        "byte_count": 10,
    }
    authority = _authority(
        attempt_root=attempt_root,
        collection_root=collection_root,
        plan_binding=plan_binding,
    )
    authority_binding = {
        "path": str(tmp_path / "authority.json"),
        "sha256": "b" * 64,
        "byte_count": 10,
    }
    collector._validate_authority_v1(  # noqa: SLF001
        authority,
        authority_binding=authority_binding,
        plan=plan,
        plan_binding=plan_binding,
    )

    changed = copy.deepcopy(authority)
    changed["permissions"]["retry_resume_overwrite"] = True
    with pytest.raises(pilot.PilotContractError, match="authority changed"):
        collector._validate_authority_v1(  # noqa: SLF001
            changed,
            authority_binding=authority_binding,
            plan=plan,
            plan_binding=plan_binding,
        )

    collection_root.mkdir()
    with pytest.raises(pilot.PilotContractError, match="fresh direct child"):
        collector._validate_authority_v1(  # noqa: SLF001
            authority,
            authority_binding=authority_binding,
            plan=plan,
            plan_binding=plan_binding,
        )


def test_collection_calls_low_level_scene_once_and_preserves_plan_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path)
    attempt_root = tmp_path / ".generated/dev/attempt-v1"
    attempt_root.mkdir(parents=True)
    collection_root = attempt_root / "collection"
    plan = _plan(output_root=collection_root)
    urdf = tmp_path / "go2.urdf"
    urdf.write_text("<robot/>\n")
    plan["runtime_bindings"] = {
        "platform_manifest": {"path": str(tmp_path / "platform.yaml")},
        "primitive_registry": {"path": str(tmp_path / "primitives.yaml")},
        "go2_urdf": pilot.file_binding(urdf),
    }
    plan["execution_contract"] = {"backend": "vulkan"}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n")
    plan_binding = pilot.file_binding(plan_path)
    authority = _authority(
        attempt_root=attempt_root,
        collection_root=collection_root,
        plan_binding=plan_binding,
    )
    authority_binding = {
        "path": str(tmp_path / "authority.json"),
        "sha256": "b" * 64,
        "byte_count": 10,
    }
    monkeypatch.setattr(
        collector,
        "load_and_validate_v1",
        lambda **_kwargs: (authority, authority_binding, plan, plan_binding),
    )
    monkeypatch.setattr(pilot, "require_plan_bindings", lambda _plan: None)
    monkeypatch.setattr(collector.kernel, "_validate_python_runtime", lambda _plan: None)
    monkeypatch.setattr(
        collector.kernel, "_validate_execution_environment", lambda _plan: None
    )
    monkeypatch.setattr(collector.kernel, "_capture_runtime_versions", lambda: {"v": "1"})

    registry_type = SimpleNamespace(from_yaml=lambda _path: object())
    runtime = {
        "load_platform_manifest": lambda _path: {},
        "resolve_go2_urdf": lambda _platform, _root: urdf,
        "PrimitiveRegistry": registry_type,
        "expand_primitive_to_block": lambda *_args: None,
    }
    monkeypatch.setattr(collector.kernel, "_runtime_imports", lambda **_kwargs: runtime)
    monkeypatch.setattr(collector.kernel, "_load_action_blocks", lambda **_kwargs: [])
    calls: list[list[str]] = []

    def collect_scene(*, states, **_kwargs):
        calls.append([str(state["state_id"]) for state in states])
        receipts = [
            {
                "state": {
                    "state_id": state["state_id"],
                    "role": state["role"],
                    "scene_id": state["scene_id"],
                },
                "branches": [{"kind": "candidate"} for _ in range(9)],
                "context": {"frame_identities": [1, 2, 3]},
            }
            for state in states
        ]
        frames = [{"byte_count": 1} for _ in range(48)]
        metrics = {
            "native_render_calls": 48,
            "rgb_render_calls": 48,
            "auxiliary_depth_render_calls": 48,
            "stored_rgb_frames": 48,
        }
        return receipts, frames, [], [], metrics

    monkeypatch.setattr(collector.kernel, "_collect_scene", collect_scene)
    monkeypatch.setattr(
        collector.bounded,
        "_validated_render_receipt_identity_v1",
        lambda **_kwargs: {"render_contract": "mocked"},
    )

    result, result_path = collector.collect_v1(
        plan_path=plan_path,
        expected_plan_byte_count=plan_binding["byte_count"],
        expected_plan_sha256=plan_binding["file_sha256"],
        authority_path=tmp_path / "authority.json",
        expected_authority_byte_count=10,
        expected_authority_sha256="b" * 64,
    )
    assert result_path == collection_root / "physics_result.json"
    assert result["status"] == "PHYSICS_COMPLETE"
    assert len(calls) == 64
    assert all(len(call) == 4 for call in calls)
    expected_ids = [str(state["state_id"]) for state in plan["states"]]
    receipt_ids = [Path(row["path"]).stem for row in result["state_receipt_bindings"]]
    assert receipt_ids == expected_ids
    assert result["observed_counts"] == collector.EXPECTED_COUNTS
    assert len(result["render_receipt_bindings"]) == 64

