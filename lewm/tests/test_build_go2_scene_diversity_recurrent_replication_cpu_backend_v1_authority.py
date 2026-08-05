from __future__ import annotations

import json

import pytest

from scripts import (
    build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_authority
    as authority,
)


def test_scientific_authority_binds_exact_pass_and_not_probe_outputs(
    monkeypatch,
) -> None:
    science = json.loads(authority.plan_builder.DEFAULT_PLAN_OUTPUT.read_text())
    bindings = {
        "prereg": {"path": "/prereg", "sha256": "a" * 64, "byte_count": 1},
        "panel": {"path": "/panel", "sha256": "b" * 64, "byte_count": 1},
        "science": {"path": "/science", "sha256": "c" * 64, "byte_count": 1},
        "qualification": {"path": "/qualification", "sha256": "d" * 64, "byte_count": 1},
        "review": {"path": "/review", "sha256": "e" * 64, "byte_count": 1},
        "result": {"path": "/result", "sha256": "f" * 64, "byte_count": 1},
    }
    monkeypatch.setattr(
        authority.qualification_authority,
        "_require_binding",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        authority.qualification_authority,
        "_load_json",
        lambda binding, **_kwargs: science if binding["path"] == "/science" else {"review": True},
    )
    monkeypatch.setattr(
        authority.qualification_authority,
        "source_bindings",
        lambda: {"source": bindings["science"]},
    )
    monkeypatch.setattr(
        authority.qualification_authority,
        "_validate_review",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority.runner,
        "validate_qualification_result_binding",
        lambda value: ({"status": "PASS"}, dict(value)),
    )
    monkeypatch.setattr(
        authority.predecessor_authority,
        "dino_declaration_v2",
        lambda: {"checkpoint_binding": bindings["science"]},
    )

    result = authority.build_scientific_authority(
        preregistration_binding=bindings["prereg"],
        scene_panel_binding=bindings["panel"],
        scientific_plan=science,
        scientific_plan_binding=bindings["science"],
        qualification_plan_binding=bindings["qualification"],
        source_review={"review": True},
        source_review_binding=bindings["review"],
        qualification_result_binding=bindings["result"],
    )

    assert set(result) == authority.runner.collector.AUTHORITY_FIELDS
    assert result["qualification_result_binding"] == bindings["result"]
    assert "qualification_contract" not in result
    assert result["attempt_id"] == authority.plan_builder.DEFAULT_ATTEMPT_ID
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()


def test_scientific_authority_is_impossible_before_pass_result(monkeypatch) -> None:
    monkeypatch.setattr(
        authority.runner,
        "validate_qualification_result_binding",
        lambda _value: (_ for _ in ()).throw(
            authority.runner.SceneDiversityRunnerError("qualification absent")
        ),
    )
    science = json.loads(authority.plan_builder.DEFAULT_PLAN_OUTPUT.read_text())
    dummy = {"path": "/dummy", "sha256": "a" * 64, "byte_count": 1}
    monkeypatch.setattr(
        authority.qualification_authority,
        "_require_binding",
        lambda value, **_kwargs: dict(value),
    )
    loaded = iter((science, {}))
    monkeypatch.setattr(
        authority.qualification_authority,
        "_load_json",
        lambda *_args, **_kwargs: next(loaded),
    )
    monkeypatch.setattr(authority.qualification_authority, "source_bindings", lambda: {})
    monkeypatch.setattr(
        authority.qualification_authority, "_validate_review", lambda *_args, **_kwargs: None
    )

    with pytest.raises(authority.runner.SceneDiversityRunnerError, match="qualification absent"):
        authority.build_scientific_authority(
            preregistration_binding=dummy,
            scene_panel_binding=dummy,
            scientific_plan=science,
            scientific_plan_binding=dummy,
            qualification_plan_binding=dummy,
            source_review={},
            source_review_binding=dummy,
            qualification_result_binding=dummy,
        )


def test_scientific_authority_and_attempt_are_absent() -> None:
    assert not authority.AUTHORITY_OUTPUT.exists()
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
