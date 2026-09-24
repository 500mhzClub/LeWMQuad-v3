from __future__ import annotations

from pathlib import Path

import pytest

from scripts import (
    build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_authority
    as authority,
)


def _bindings() -> dict[str, dict[str, object]]:
    return {
        name: {"path": f"/{name}", "sha256": character * 64, "byte_count": 1}
        for name, character in zip(
            ("prereg", "panel", "science", "qualification", "review", "result"),
            "abcdef",
            strict=True,
        )
    }


def _prepare(tmp_path: Path, monkeypatch, *, qualification_passes: bool):
    science = {"plan_role": "scientific"}
    bindings = _bindings()
    qa = authority.qualification_authority
    monkeypatch.setattr(qa, "_require_binding", lambda value, **_kwargs: dict(value))
    monkeypatch.setattr(
        qa,
        "_load_json",
        lambda binding, **_kwargs: (
            science if binding["path"] == "/science" else {"review": True}
        ),
    )
    monkeypatch.setattr(
        authority.plan_builder,
        "validate_rocm_plan",
        lambda plan, **_kwargs: dict(plan),
    )
    monkeypatch.setattr(qa, "source_bindings", lambda: {"source": bindings["science"]})
    monkeypatch.setattr(qa, "_validate_review", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        authority.predecessor_authority,
        "dino_declaration_v2",
        lambda: {"checkpoint_binding": bindings["science"]},
    )
    monkeypatch.setattr(
        authority.plan_builder,
        "DEFAULT_ATTEMPT_ROOT",
        tmp_path / "scientific_attempt",
    )
    monkeypatch.setattr(
        authority.plan_builder,
        "DEFAULT_OUTPUT_ROOT",
        tmp_path / "scientific_attempt/collection",
    )
    if qualification_passes:
        monkeypatch.setattr(
            authority.runner,
            "validate_qualification_result_binding",
            lambda value: ({"status": "PASS"}, dict(value)),
        )
    else:
        monkeypatch.setattr(
            authority.runner,
            "validate_qualification_result_binding",
            lambda _value: (_ for _ in ()).throw(
                authority.runner.SceneDiversityRunnerError(
                    "V3 qualification absent"
                )
            ),
        )
    return science, bindings


def test_scientific_authority_binds_exact_pass_and_review_only_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    science, bindings = _prepare(tmp_path, monkeypatch, qualification_passes=True)
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
    assert result["predecessor_v2_qualification_terminal_review_binding"] == (
        authority.qualification_authority._v2_terminal_review_binding()  # noqa: SLF001
    )
    assert "qualification_contract" not in result
    assert "probes" not in result
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()


def test_scientific_authority_is_impossible_before_exact_v3_pass(
    tmp_path: Path, monkeypatch,
) -> None:
    science, bindings = _prepare(tmp_path, monkeypatch, qualification_passes=False)
    with pytest.raises(
        authority.runner.SceneDiversityRunnerError,
        match="V3 qualification absent",
    ):
        authority.build_scientific_authority(
            preregistration_binding=bindings["prereg"],
            scene_panel_binding=bindings["panel"],
            scientific_plan=science,
            scientific_plan_binding=bindings["science"],
            qualification_plan_binding=bindings["qualification"],
            source_review={"review": True},
            source_review_binding=bindings["review"],
            qualification_result_binding=bindings["result"],
        )


def test_no_v3_scientific_authority_or_attempt_exists() -> None:
    assert not authority.AUTHORITY_OUTPUT.exists()
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
