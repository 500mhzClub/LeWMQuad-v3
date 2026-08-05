from __future__ import annotations

from pathlib import Path

from scripts import (
    build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_qualification_authority
    as authority,
)


def test_source_audit_declares_material_rocm_backend_and_frozen_science() -> None:
    audit = authority.ROCM_BACKEND_SOURCE_AUDIT
    assert audit["material_backend_successor_not_vulkan_v4_or_cpu_retry"] is True
    assert audit["genesis_version_exact_0_4_6"] is True
    assert audit["genesis_backend_exact_amdgpu"] is True
    assert audit["hip_device_exact_r9700_gfx1201"] is True
    assert audit["host_rocm_ld_library_path_forbidden"] is True
    assert audit["qualification_outputs_forbidden_from_scientific_reuse"] is True
    assert audit["consumed_cpu_qualification_terminal_review_exactly_bound"] is True


def test_qualification_authority_is_separate_and_cannot_authorize_science(
    tmp_path: Path, monkeypatch,
) -> None:
    science = {"plan_role": "scientific"}
    qualification = {"plan_role": "qualification"}
    bindings = {
        "prereg": {"path": "/prereg", "sha256": "a" * 64, "byte_count": 1},
        "panel": {"path": "/panel", "sha256": "b" * 64, "byte_count": 1},
        "science": {"path": "/science", "sha256": "c" * 64, "byte_count": 1},
        "qualification": {
            "path": "/qualification",
            "sha256": "d" * 64,
            "byte_count": 1,
        },
        "review": {"path": "/review", "sha256": "e" * 64, "byte_count": 1},
    }
    monkeypatch.setattr(
        authority, "_require_binding", lambda value, **_kwargs: dict(value)
    )
    documents = {
        "/science": science,
        "/qualification": qualification,
        "/review": {"review": True},
    }
    monkeypatch.setattr(
        authority,
        "_load_json",
        lambda binding, **_kwargs: documents[str(binding["path"])],
    )
    monkeypatch.setattr(
        authority.plan_builder,
        "validate_rocm_plan",
        lambda plan, **_kwargs: dict(plan),
    )
    monkeypatch.setattr(
        authority, "source_bindings", lambda: {"source": bindings["science"]}
    )
    monkeypatch.setattr(
        authority, "_validate_review", lambda *_args, **_kwargs: None
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
        "QUALIFICATION_ATTEMPT_ROOT",
        tmp_path / "qualification_attempt",
    )
    monkeypatch.setattr(
        authority.plan_builder,
        "DEFAULT_OUTPUT_ROOT",
        tmp_path / "scientific_attempt/collection",
    )
    monkeypatch.setattr(
        authority.plan_builder,
        "QUALIFICATION_OUTPUT_ROOT",
        tmp_path / "qualification_attempt/collection",
    )

    result = authority.build_qualification_authority(
        preregistration_binding=bindings["prereg"],
        scene_panel_binding=bindings["panel"],
        scientific_plan=science,
        scientific_plan_binding=bindings["science"],
        qualification_plan=qualification,
        qualification_plan_binding=bindings["qualification"],
        source_review={"review": True},
        source_review_binding=bindings["review"],
    )

    assert set(result) == authority.qualifier.QUALIFICATION_AUTHORITY_FIELDS
    assert result["schema"] == authority.qualifier.QUALIFICATION_AUTHORITY_SCHEMA
    assert result["attempt_id"] == authority.plan_builder.QUALIFICATION_ATTEMPT_ID
    assert result["qualification_contract"] == authority.qualifier.QUALIFICATION_CONTRACT
    assert result["predecessor_cpu_terminal_review_binding"] == (
        authority._cpu_terminal_review_binding()  # noqa: SLF001
    )
    assert "qualification_result_binding" not in result
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
    assert not authority.plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()


def test_runner_source_closure_names_every_new_builder_runner_and_test() -> None:
    names = set(authority.runner.SOURCE_PATHS)
    assert {
        "rocm_backend_qualification_authority_builder",
        "rocm_backend_scientific_authority_builder",
        "rocm_backend_runner",
        "rocm_backend_qualification_authority_test",
        "rocm_backend_scientific_authority_test",
        "rocm_backend_runner_test",
        "predecessor_cpu_qualification_terminal_review",
    } <= names
