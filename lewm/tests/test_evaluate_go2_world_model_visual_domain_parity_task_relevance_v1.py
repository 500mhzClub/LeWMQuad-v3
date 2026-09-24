from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts import (
    evaluate_go2_world_model_visual_domain_parity_task_relevance_v1 as subject,
)
from scripts import run_go2_world_model_visual_domain_parity_authorized_v1 as supervisor


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _progression_analysis(tmp_path: Path) -> dict:
    panel = {}
    for seed in subject.SEEDS:
        panel[str(seed)] = {}
        for arm in subject.ARMS:
            path = tmp_path / f"{seed}_{arm}.pt"
            path.write_bytes(f"{seed}/{arm}".encode())
            panel[str(seed)][arm] = {
                "path": str(path),
                "byte_count": path.stat().st_size,
                "sha256": _sha(path),
            }
    return {
        "schema": subject.PROGRESSION_SCHEMA,
        "status": subject.PROGRESSION_STATUS,
        "development_only": True,
        "citable_as_world_model_usefulness_evidence": False,
        "input_result": {},
        "configuration": {},
        "decoder_anchor_by_seed": {},
        "contrasts": {},
        "proxy_routing": {},
        "terminal_snapshot_bindings": panel,
        "uncertainty_limit": "development-only",
    }


def _inventory_plan(tmp_path: Path) -> dict:
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "scenes").mkdir()
    for name in (
        "reservation.json",
        "generation_receipt.json",
        "candidate_panel.json",
        "parity_result.json",
        "terminal_failure.json",
    ):
        (root / name).write_text(name)
    scenes = []
    for scene_index in range(8):
        scene_id = f"scene_{scene_index}"
        scene_root = root / "scenes" / f"{scene_index:02d}_{scene_id}"
        rows_root = scene_root / "rows"
        rows_root.mkdir(parents=True)
        poses = []
        for pose_index in range(4):
            pose_root = rows_root / f"pose_{pose_index:02d}"
            pose_root.mkdir()
            for name in (
                "candidate.png",
                "duplicate.png",
                "candidate_receipt.json",
                "duplicate_receipt.json",
            ):
                (pose_root / name).write_text(
                    f"{scene_index}/{pose_index}/{name}"
                )
            poses.append({"pose_index": pose_index})
        (scene_root / "scene_result.json").write_text(scene_id)
        scenes.append({"scene_id": scene_id, "poses": poses})
    return {"output_root": str(root), "scenes": scenes}


def test_descriptor_retrieval_passes_exact_pair_panel() -> None:
    reference = np.repeat(
        np.arange(32, dtype=np.float64)[:, None],
        subject.DESCRIPTOR_DIMENSION,
        axis=1,
    )
    candidate = reference + 0.001
    metrics = subject.descriptor_retrieval_metrics_v1(reference, candidate)
    assert metrics["paired_nearest_neighbour_retrieval_count"] == 32
    assert metrics[
        "worst_paired_to_nearest_nonself_descriptor_distance_ratio"
    ] < 0.1
    assert metrics["descriptor_dimension"] == 1536


def test_descriptor_retrieval_exposes_ambiguous_nonself_panel() -> None:
    reference = np.zeros((32, subject.DESCRIPTOR_DIMENSION))
    metrics = subject.descriptor_retrieval_metrics_v1(reference, reference.copy())
    assert metrics["paired_nearest_neighbour_retrieval_count"] == 1
    assert metrics[
        "worst_paired_to_nearest_nonself_descriptor_distance_ratio"
    ] is None


def test_progression_rehashes_exact_twelve_snapshots(tmp_path: Path) -> None:
    analysis = _progression_analysis(tmp_path)
    bindings = subject.progression_snapshot_bindings_v1(analysis)
    assert len(bindings) == 12
    assert {(row["arm"], row["seed"]) for row in bindings} == {
        (arm, seed) for seed in subject.SEEDS for arm in subject.ARMS
    }


def test_progression_rehash_rejects_changed_snapshot(tmp_path: Path) -> None:
    analysis = _progression_analysis(tmp_path)
    binding = analysis["terminal_snapshot_bindings"][str(subject.SEEDS[0])][
        subject.ARMS[0]
    ]
    Path(binding["path"]).write_bytes(b"changed")
    with pytest.raises(subject.TaskRelevanceEvaluationError):
        subject.progression_snapshot_bindings_v1(analysis)


def test_consumed_inventory_requires_exact_141_files(tmp_path: Path) -> None:
    plan = _inventory_plan(tmp_path)
    inventory = subject.consumed_inventory_v1(plan)
    assert len(inventory) == 141
    assert inventory[0]["relative_path"] == "candidate_panel.json"
    assert inventory[-1]["relative_path"] == "terminal_failure.json"
    assert len({row["relative_path"] for row in inventory}) == 141


def test_consumed_inventory_rejects_extra_leaf(tmp_path: Path) -> None:
    plan = _inventory_plan(tmp_path)
    (Path(plan["output_root"]) / "unexpected.json").write_text("unexpected")
    with pytest.raises(subject.TaskRelevanceEvaluationError):
        subject.consumed_inventory_v1(plan)


def test_terminal_failure_preserves_exact_parity_failure(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    plan_binding = {"path": str(tmp_path / "plan.json"), "file_sha256": "a" * 64, "byte_count": 1}
    authority_binding = {"path": str(tmp_path / "authority.json"), "file_sha256": "b" * 64, "byte_count": 1}
    terminal_binding = {"path": str(root / "terminal_failure.json"), "file_sha256": "c" * 64, "byte_count": 1}
    terminal = {
        "schema": supervisor.TERMINAL_SCHEMA,
        "status": supervisor.TERMINAL_FAILURE_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "authorizes_retry_or_resume": False,
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "reservation_binding": {"path": str(root / "reservation.json")},
        "reservation_path": str(root / "reservation.json"),
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "wall_seconds": 10.0,
        "failed_at": "2026-08-02T15:29:25Z",
        "failure": {
            "type": "VisualDomainParitySupervisionError",
            "message": "visual-domain parity evaluator did not pass exactly",
        },
    }
    subject._validate_terminal_failure(
        terminal,
        terminal_binding=terminal_binding,
        plan={"output_root": str(root)},
        plan_binding=plan_binding,
        authority={"caps": {"wall_seconds": 3600}},
        authority_binding=authority_binding,
    )
    terminal["failure"]["message"] = "pass"
    with pytest.raises(subject.TaskRelevanceEvaluationError):
        subject._validate_terminal_failure(
            terminal,
            terminal_binding=terminal_binding,
            plan={"output_root": str(root)},
            plan_binding=plan_binding,
            authority={"caps": {"wall_seconds": 3600}},
            authority_binding=authority_binding,
        )


def test_source_has_no_render_or_training_entrypoint() -> None:
    source = Path(subject.__file__).read_text()
    assert "render_replay_v03.build_scene" not in source
    assert "optimizer.step" not in source
    assert "torch.cuda" not in source
    assert 'torch.device("cpu")' in source
