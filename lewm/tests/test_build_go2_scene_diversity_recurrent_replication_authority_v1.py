from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import build_go2_scene_diversity_recurrent_replication_authority_v1 as authority


@pytest.fixture
def reviewed_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    attempt_root = tmp_path / "attempt_v1"
    collection_root = attempt_root / "collection"
    monkeypatch.setattr(authority, "ATTEMPT_ROOT", attempt_root)
    monkeypatch.setattr(authority, "COLLECTION_ROOT", collection_root)

    plan = json.loads(authority.EXACT_PLAN.read_text())
    plan["output_root"] = str(collection_root.resolve())
    plan_path = tmp_path / "exact-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(authority, "EXACT_PLAN", plan_path)

    source_path = tmp_path / "runner-source.py"
    source_path.write_text("VALUE = 1\n")
    monkeypatch.setattr(authority.runner, "SOURCE_PATHS", {"runner_source": source_path})
    source_bindings = authority.source_bindings_v1()

    preregistration_binding = authority.file_binding_v1(authority.PREREGISTRATION)
    panel_binding = authority.file_binding_v1(authority.SCENE_PANEL)
    plan_binding = authority.file_binding_v1(plan_path)
    review = {
        "schema": authority.SOURCE_REVIEW_SCHEMA,
        "status": authority.SOURCE_REVIEW_STATUS,
        "reviewer": "independent-test-reviewer",
        "protected_material_opened": False,
        "findings": [],
        "preregistration_binding": preregistration_binding,
        "scene_panel_binding": panel_binding,
        "plan_binding": plan_binding,
        "source_bindings": source_bindings,
        "checks": {"focused_tests_passed": True},
    }
    review_path = tmp_path / "source-review.json"
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(authority, "SOURCE_REVIEW", review_path)
    review_binding = authority.file_binding_v1(review_path)

    dino_checkpoint = tmp_path / "dino-checkpoint.pth"
    dino_checkpoint.write_bytes(b"dino")
    dino_repository = tmp_path / "dino-repository"
    dino_repository.mkdir()
    dino = {
        "repository_path": str(dino_repository.resolve()),
        "repository_commit": "a" * 40,
        "checkpoint_binding": authority.file_binding_v1(dino_checkpoint),
    }
    monkeypatch.setattr(authority, "dino_declaration_v1", lambda: copy.deepcopy(dino))
    return {
        "plan": plan,
        "plan_binding": plan_binding,
        "preregistration_binding": preregistration_binding,
        "scene_panel_binding": panel_binding,
        "review": review,
        "review_binding": review_binding,
        "source_bindings": source_bindings,
        "dino": dino,
        "attempt_root": attempt_root,
        "collection_root": collection_root,
    }


def test_authority_binds_exact_review_plan_sources_and_limits(reviewed_inputs: dict) -> None:
    result = authority.build_authority_v1(
        preregistration_binding=reviewed_inputs["preregistration_binding"],
        scene_panel_binding=reviewed_inputs["scene_panel_binding"],
        plan=reviewed_inputs["plan"],
        plan_binding=reviewed_inputs["plan_binding"],
        source_review=reviewed_inputs["review"],
        source_review_binding=reviewed_inputs["review_binding"],
    )
    assert set(result) == authority.collector.AUTHORITY_FIELDS
    assert result["schema"] == authority.collector.AUTHORITY_SCHEMA
    assert result["status"] == authority.collector.AUTHORITY_STATUS
    assert result["source_bindings"] == reviewed_inputs["source_bindings"]
    assert result["dino"] == reviewed_inputs["dino"]
    assert result["config"] == authority.benchmark.config_v1()
    assert result["caps"] == authority.collector.EXPECTED_CAPS
    assert result["permissions"] == authority.collector.EXPECTED_PERMISSIONS
    assert result["attempt_root"] == str(reviewed_inputs["attempt_root"].resolve())
    assert result["collection_root"] == str(reviewed_inputs["collection_root"].resolve())

    reviewed_inputs["attempt_root"].mkdir()
    with pytest.raises(authority.SceneDiversityAuthorityError, match="not fresh"):
        authority.build_authority_v1(
            preregistration_binding=reviewed_inputs["preregistration_binding"],
            scene_panel_binding=reviewed_inputs["scene_panel_binding"],
            plan=reviewed_inputs["plan"],
            plan_binding=reviewed_inputs["plan_binding"],
            source_review=reviewed_inputs["review"],
            source_review_binding=reviewed_inputs["review_binding"],
        )


def test_review_must_bind_the_exact_plan_and_panel(
    reviewed_inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = copy.deepcopy(reviewed_inputs["review"])
    changed["plan_binding"] = dict(changed["plan_binding"])
    changed["plan_binding"]["sha256"] = "0" * 64
    review_path = tmp_path / "changed-review.json"
    review_path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(authority, "SOURCE_REVIEW", review_path)
    with pytest.raises(authority.SceneDiversityAuthorityError, match="review changed"):
        authority.build_authority_v1(
            preregistration_binding=reviewed_inputs["preregistration_binding"],
            scene_panel_binding=reviewed_inputs["scene_panel_binding"],
            plan=reviewed_inputs["plan"],
            plan_binding=reviewed_inputs["plan_binding"],
            source_review=changed,
            source_review_binding=authority.file_binding_v1(review_path),
        )


def test_authority_write_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "authority.json"
    binding = authority._write_json_exclusive(output, {"status": "test"})  # noqa: SLF001
    assert binding == authority.file_binding_v1(output)
    with pytest.raises(authority.SceneDiversityAuthorityError, match="not fresh"):
        authority._write_json_exclusive(output, {"status": "again"})  # noqa: SLF001
