from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import (
    build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_authority
    as authority,
)


@pytest.fixture
def reviewed_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    attempt_root = tmp_path / "attempt_v1"
    collection_root = attempt_root / "collection"
    monkeypatch.setattr(authority, "ATTEMPT_ROOT", attempt_root)
    monkeypatch.setattr(authority, "COLLECTION_ROOT", collection_root)

    preregistration_path = tmp_path / "preregistration.md"
    preregistration_path.write_text("# replacement preregistration\n")
    monkeypatch.setattr(authority, "PREREGISTRATION", preregistration_path)

    frozen = json.loads(authority.plan_builder.FROZEN_V1_EXACT_PLAN.read_text())
    plan = copy.deepcopy(frozen)
    plan["attempt_id"] = authority.plan_builder.DEFAULT_ATTEMPT_ID
    plan["output_root"] = str(collection_root.resolve())
    plan_path = tmp_path / "exact-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(authority, "EXACT_PLAN", plan_path)

    source_path = tmp_path / "replacement-source.py"
    source_path.write_text("VALUE = 1\n")
    monkeypatch.setattr(authority.runner, "SOURCE_PATHS", {"replacement": source_path})
    monkeypatch.setattr(
        authority.runner, "predecessor_failure_bindings_v1", lambda: {}
    )
    source_bindings = authority.source_bindings_v1()

    preregistration_binding = authority.file_binding_v1(preregistration_path)
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
    monkeypatch.setattr(
        authority, "dino_declaration_v1", lambda: copy.deepcopy(dino)
    )
    return {
        "attempt_root": attempt_root,
        "collection_root": collection_root,
        "dino": dino,
        "panel_binding": panel_binding,
        "plan": plan,
        "plan_binding": plan_binding,
        "preregistration_binding": preregistration_binding,
        "review": review,
        "review_binding": review_binding,
        "source_bindings": source_bindings,
    }


def _build(reviewed_inputs: dict) -> dict:
    return authority.build_authority_v1(
        preregistration_binding=reviewed_inputs["preregistration_binding"],
        scene_panel_binding=reviewed_inputs["panel_binding"],
        plan=reviewed_inputs["plan"],
        plan_binding=reviewed_inputs["plan_binding"],
        source_review=reviewed_inputs["review"],
        source_review_binding=reviewed_inputs["review_binding"],
    )


def test_authority_binds_science_identical_reviewed_fresh_attempt(
    reviewed_inputs: dict,
) -> None:
    result = _build(reviewed_inputs)

    assert set(result) == authority.collector.AUTHORITY_FIELDS
    assert result["schema"] == authority.collector.AUTHORITY_SCHEMA
    assert result["status"] == authority.collector.AUTHORITY_STATUS
    assert result["attempt_id"] == authority.plan_builder.DEFAULT_ATTEMPT_ID
    assert result["attempt_root"] == str(reviewed_inputs["attempt_root"].resolve())
    assert result["collection_root"] == str(
        reviewed_inputs["collection_root"].resolve()
    )
    assert result["source_bindings"] == reviewed_inputs["source_bindings"]
    assert result["config"] == authority.benchmark.config_v1()
    assert result["caps"] == authority.collector.EXPECTED_CAPS
    assert result["permissions"] == authority.collector.EXPECTED_PERMISSIONS
    assert result["permissions"]["retry_resume_overwrite"] is False
    assert result["dino"] == reviewed_inputs["dino"]


def test_authority_rejects_scientific_drift_or_nonfresh_root(
    reviewed_inputs: dict,
) -> None:
    valid_drift = copy.deepcopy(reviewed_inputs["plan"])
    valid_drift["execution_contract"]["seed"] += 1
    with pytest.raises(
        authority.SceneDiversityReplacementAuthorityError,
        match="not science-identical",
    ):
        authority._validate_science_identical_plan_v1(valid_drift)  # noqa: SLF001

    changed = copy.deepcopy(reviewed_inputs["plan"])
    changed["states"][0]["candidate_action_ids"] = list(reversed(range(9)))
    with pytest.raises(authority.SceneDiversityReplacementAuthorityError):
        authority.build_authority_v1(
            preregistration_binding=reviewed_inputs["preregistration_binding"],
            scene_panel_binding=reviewed_inputs["panel_binding"],
            plan=changed,
            plan_binding=reviewed_inputs["plan_binding"],
            source_review=reviewed_inputs["review"],
            source_review_binding=reviewed_inputs["review_binding"],
        )

    reviewed_inputs["attempt_root"].mkdir()
    with pytest.raises(
        authority.SceneDiversityReplacementAuthorityError,
        match="not fresh",
    ):
        _build(reviewed_inputs)


def test_source_closure_requires_exact_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n")
    evidence = tmp_path / "terminal.json"
    evidence.write_text("{}\n")
    evidence_binding = authority.file_binding_v1(evidence)
    monkeypatch.setattr(
        authority.runner,
        "SOURCE_PATHS",
        {"predecessor_failure_terminal": source},
    )
    monkeypatch.setattr(
        authority.runner,
        "predecessor_failure_bindings_v1",
        lambda: {"predecessor_failure_terminal": evidence_binding},
    )

    with pytest.raises(
        authority.SceneDiversityReplacementAuthorityError,
        match="not exact",
    ):
        authority.source_bindings_v1()


def test_authority_write_is_exclusive_and_does_not_create_attempt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "authority.json"
    attempt = tmp_path / "attempt_v1"

    binding = authority._write_json_exclusive(  # noqa: SLF001
        output, {"status": "test"}
    )

    assert binding == authority.file_binding_v1(output)
    assert not attempt.exists()
    with pytest.raises(
        authority.SceneDiversityReplacementAuthorityError,
        match="not fresh",
    ):
        authority._write_json_exclusive(  # noqa: SLF001
            output, {"status": "again"}
        )
