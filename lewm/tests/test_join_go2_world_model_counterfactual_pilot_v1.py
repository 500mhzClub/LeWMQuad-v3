from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
JOINER_PATH = ROOT / "scripts/join_go2_world_model_counterfactual_pilot_v1.py"


def _load_joiner():
    spec = importlib.util.spec_from_file_location(
        "counterfactual_pilot_joiner_v1", JOINER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(identity: str) -> dict[str, object]:
    return {
        "artifact_id": identity,
        "frame_identity": identity,
        "path": f"frames/{identity.replace(':', '_')}.png",
        "file_sha256": f"{len(identity) % 16:x}" * 64,
        "byte_count": 10,
        "width": 224,
        "height": 224,
        "mode": "RGB",
        "format": "PNG",
        "camera_valid": True,
    }


def _collection() -> dict[str, object]:
    frame_receipts = {}
    states = []
    action_catalog = [
        {
            "action_id": action,
            "name": f"action-{action}",
            "requested_block": [[float(action), 0.0, 0.0]] * 5,
        }
        for action in range(9)
    ]
    for group_index, role in enumerate(("train", "eval")):
        state_id = f"{role}:state"
        context_ids = [f"{state_id}:context:{index}" for index in range(3)]
        for identity in context_ids:
            frame_receipts[identity] = _frame(identity)
        branches = []
        for action in range(9):
            identity = f"{state_id}:candidate:{action}"
            receipt = _frame(identity)
            frame_receipts[identity] = receipt
            branches.append({
                "lane_index": group_index * 9 + action,
                "lane_offset": action,
                "kind": "candidate",
                "action_id": action,
                "action_name": f"action-{action}",
                "duplicates_candidate_action_id": None,
                "requested_block": [[float(action), 0.0, 0.0]] * 5,
                # Deliberately differs from the requested block.  The join must
                # preserve it as an outcome without using it as action identity.
                "executed_block": [[float(action) + 0.25, 0.0, 0.0]] * 5,
                "executed_block_sha256": f"{action:x}" * 64,
                "clipped": action == 0,
                "trajectory_policy_step_samples": [{"step": 0}],
                "endpoint_state": {"base_pos_world": [float(action), 0.0, 0.3]},
                "physical_fell": False,
                "physical_tipped": False,
                "physical_path_length_m": float(action),
                "physical_target_progress_m": float(action),
                "render_frame_identity": identity,
                "frame_receipt": receipt,
            })
        states.append({
            "state": {
                "state_id": state_id,
                "role": role,
                "family": "large_enclosed_maze",
                "scene_id": f"{role}-scene",
                "group_index": group_index,
                "state_index_in_scene": 0,
            },
            "context": {
                "rgb_artifact_ids": context_ids,
                "frame_identities": context_ids,
                "history_action_ids": [0, 1],
                "history_executed_blocks": [[[0.0, 0.0, 0.0]] * 5] * 2,
                "executed_block_sha256s": ["a" * 64, "b" * 64],
                "endpoint_command_ticks": [0, 5, 10],
                "prebranch_state_sha256": "c" * 64,
            },
            "relative_target_xy_body_m": [1.0, 0.0],
            "document": {
                "synchronization_audit": {"passed": True},
            },
            "branches": branches,
        })
    return {
        "purpose": "bounded_wm_a_pilot",
        "states": states,
        "frame_receipts": frame_receipts,
        "plan": {"document": {"action_catalog": action_catalog}},
    }


def _calibration() -> dict[str, object]:
    return {
        "decision": "FREEZE_PILOT_CONTRACT",
        "calibration_contract": {
            "excluded_scene_ids": ["calibration-scene"],
            "progress_tolerance_m": 1e-6,
            "path_length_tolerance_m": 1e-6,
        },
    }


def test_join_builds_receipt_rows_without_future_executed_tape_leakage() -> None:
    joiner = _load_joiner()
    rgb_manifest, rows, metadata = joiner.build_joined_documents_v1(
        _collection(), _calibration()
    )
    assert len(rgb_manifest["artifacts"]) == 24
    assert len(rows["train"]) == len(rows["eval"]) == 1
    branch = rows["train"][0]["branches"][0]
    assert branch["action_id"] == 0
    assert branch["requested_block"] == [[0.0, 0.0, 0.0]] * 5
    assert branch["executed_block"] == [[0.25, 0.0, 0.0]] * 5
    assert branch["declared_oracle_dense_rank"] == 8
    assert metadata["scene_ids"] == {
        "train": ["train-scene"],
        "eval": ["eval-scene"],
    }


def test_join_rejects_failed_calibration_before_emitting_rows() -> None:
    joiner = _load_joiner()
    failed = {**_calibration(), "decision": "STOP_SOURCE_REDESIGN"}
    with pytest.raises(joiner.PilotJoinError, match="did not freeze"):
        joiner.build_joined_documents_v1(_collection(), failed)
