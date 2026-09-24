from __future__ import annotations

from pathlib import Path

from scripts import (
    build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_qualification_authority
    as authority,
)


def test_source_audit_declares_only_fresh_driver_successor() -> None:
    audit = authority.ROCM_BACKEND_SOURCE_AUDIT
    assert audit["material_fresh_v2_ld_lld_driver_successor_not_v1_retry"] is True
    assert audit["exact_lexical_ld_lld_driver_in_plan_and_source"] is True
    assert audit["resolved_regular_lld_target_separately_bound"] is True
    assert audit["direct_regular_lld_target_production_invocation_forbidden"] is True
    assert audit["v1_terminal_review_document_exactly_bound"] is True
    assert audit["v1_runtime_metadata_and_payload_reuse_forbidden"] is True


def test_qualification_authority_binds_v1_review_and_cannot_authorize_science(
    tmp_path: Path, monkeypatch,
) -> None:
    science = {"plan_role": "scientific"}
    qualification = {"plan_role": "qualification"}
    bindings = {
        name: {"path": f"/{name}", "sha256": character * 64, "byte_count": 1}
        for name, character in zip(
            ("prereg", "panel", "science", "qualification", "review"),
            "abcde",
            strict=True,
        )
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
    monkeypatch.setattr(authority, "_validate_review", lambda *_args, **_kwargs: None)
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
    assert result["attempt_id"] == authority.plan_builder.QUALIFICATION_ATTEMPT_ID
    assert result["predecessor_v1_qualification_terminal_review_binding"] == (
        authority._v1_terminal_review_binding()  # noqa: SLF001
    )
    assert "qualification_result_binding" not in result
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
    assert not authority.plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()


def test_source_closure_names_full_v2_chain_and_review_only_evidence() -> None:
    names = set(authority.runner.SOURCE_PATHS)
    assert {
        "rocm_backend_v2_qualification_authority_builder",
        "rocm_backend_v2_scientific_authority_builder",
        "rocm_backend_v2_runner",
        "rocm_backend_v2_qualification_authority_test",
        "rocm_backend_v2_scientific_authority_test",
        "predecessor_v1_qualification_terminal_review",
    } <= names
    assert all(
        "genesis_rocm_backend_v1_qualification/attempt_v1" not in str(path)
        for path in authority.runner.SOURCE_PATHS.values()
    )
