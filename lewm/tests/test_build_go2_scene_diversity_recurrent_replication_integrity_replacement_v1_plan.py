from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import (
    build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_plan
    as builder,
)


def _fresh_output_root() -> Path:
    return builder.DEFAULT_OUTPUT_ROOT


def _frozen_plan() -> dict:
    return json.loads(builder.FROZEN_V1_EXACT_PLAN.read_text())


def test_replacement_plan_changes_only_attempt_and_output_identity() -> None:
    frozen = _frozen_plan()
    output_root = _fresh_output_root()

    replacement = builder.build_plan_v1(
        frozen_plan=frozen,
        output_root=output_root,
    )

    assert replacement == pilot.validate_plan(replacement)
    assert replacement["attempt_id"] == builder.DEFAULT_ATTEMPT_ID
    assert replacement["output_root"] == str(output_root.resolve(strict=False))
    assert all(
        replacement[field] == frozen[field]
        for field in set(frozen) - {"attempt_id", "output_root"}
    )
    assert replacement["expected_counts"] == frozen["expected_counts"]
    assert replacement["states"] == frozen["states"]
    assert replacement == json.loads(builder.DEFAULT_PLAN_OUTPUT.read_text())


def test_changed_science_or_old_identity_is_rejected() -> None:
    changed = copy.deepcopy(_frozen_plan())
    changed["execution_contract"]["seed"] += 1
    with pytest.raises(
        builder.SceneDiversityReplacementPlanError,
        match="binding or content changed",
    ):
        builder.build_plan_v1(
            frozen_plan=changed,
            output_root=_fresh_output_root(),
        )

    with pytest.raises(
        builder.SceneDiversityReplacementPlanError,
        match="attempt identifier",
    ):
        builder.build_plan_v1(
            frozen_plan=_frozen_plan(),
            attempt_id="another-attempt",
            output_root=_fresh_output_root(),
        )


def test_output_root_must_be_fresh_and_development_only(tmp_path: Path) -> None:
    with pytest.raises(
        builder.SceneDiversityReplacementPlanError,
        match="exact fresh replacement collection path",
    ):
        builder.build_plan_v1(
            frozen_plan=_frozen_plan(),
            output_root=tmp_path / "outside-repository",
        )

    alternate = builder.REPO_ROOT / ".generated/dev/alternate/attempt_v1/collection"
    with pytest.raises(
        builder.SceneDiversityReplacementPlanError,
        match="exact fresh replacement collection path",
    ):
        builder.build_plan_v1(
            frozen_plan=_frozen_plan(),
            output_root=alternate,
        )


def test_exact_attempt_root_must_still_be_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_repo = tmp_path / "repo"
    development = fake_repo / ".generated/dev"
    development.mkdir(parents=True)
    attempt = development / "replacement/attempt_v1"
    collection = attempt / "collection"
    attempt.mkdir(parents=True)
    monkeypatch.setattr(builder, "REPO_ROOT", fake_repo)
    monkeypatch.setattr(builder, "DEFAULT_ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(builder, "DEFAULT_OUTPUT_ROOT", collection)

    with pytest.raises(
        builder.SceneDiversityReplacementPlanError,
        match="exact fresh replacement collection path",
    ):
        builder.build_plan_v1(
            frozen_plan=_frozen_plan(),
            output_root=collection,
        )
