from __future__ import annotations

from pathlib import Path

import pytest

from scripts import (
    build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_authority
    as authority,
)


def _bindings() -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": f"/{name}",
            "sha256": character * 64,
            "byte_count": 1,
        }
        for name, character in zip(
            (
                "prereg",
                "panel",
                "science",
                "qualification",
                "review",
                "result",
            ),
            "abcdef",
            strict=True,
        )
    }


def _prepare(
    tmp_path: Path, monkeypatch, *, qualification_passes: bool
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    science: dict[str, object] = {"plan_role": "scientific"}
    bindings = _bindings()
    monkeypatch.setattr(
        authority.qualification_authority,
        "_require_binding",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        authority.qualification_authority,
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
                    "ROCm qualification absent"
                )
            ),
        )
    return science, bindings


def test_scientific_authority_binds_exact_pass_and_no_probe_payloads(
    tmp_path: Path, monkeypatch,
) -> None:
    science, bindings = _prepare(
        tmp_path, monkeypatch, qualification_passes=True
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
    assert result["predecessor_cpu_terminal_review_binding"] == (
        authority.qualification_authority._cpu_terminal_review_binding()  # noqa: SLF001
    )
    assert result["attempt_id"] == authority.plan_builder.DEFAULT_ATTEMPT_ID
    assert "qualification_contract" not in result
    assert "probes" not in result
    assert "scene_result_binding" not in result
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()


def test_scientific_authority_is_impossible_before_exact_pass(
    tmp_path: Path, monkeypatch,
) -> None:
    science, bindings = _prepare(
        tmp_path, monkeypatch, qualification_passes=False
    )
    with pytest.raises(
        authority.runner.SceneDiversityRunnerError,
        match="ROCm qualification absent",
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


def test_scientific_authority_and_attempt_are_absent() -> None:
    assert not authority.AUTHORITY_OUTPUT.exists()
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
