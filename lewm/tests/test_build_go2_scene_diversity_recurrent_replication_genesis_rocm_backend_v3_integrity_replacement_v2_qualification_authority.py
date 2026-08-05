from __future__ import annotations

from pathlib import Path

from scripts import (
    build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_qualification_authority
    as authority,
)


def test_source_audit_declares_exact_closure_only_successor() -> None:
    audit = authority.ROCM_BACKEND_SOURCE_AUDIT
    assert audit["science_identical_v3_integrity_replacement_not_retry"] is True
    assert audit["complete_v3_source_closure_frozen_before_predecessor_overlay"] is True
    assert audit["v2_preregistration_source_is_replacement_owned_literal"] is True
    assert audit["qualifier_first_isolated_process_regression_passed"] is True
    assert audit["poisoned_predecessor_cache_regression_passed"] is True
    assert audit["required_host_home_literal"] == "/home/andrewknowles"
    assert audit["required_host_home_is_not_ambient_derived"] is True
    assert audit["required_host_home_checked_before_reservation"] is True
    assert audit["required_host_home_overwrites_child_ambient_value"] is True
    assert audit["user_logname_and_lang_remain_absent"] is True
    assert audit["v2_terminal_review_document_exactly_bound"] is True
    assert (
        audit["v2_terminal_reservation_runtime_and_payload_reuse_forbidden"]
        is True
    )
    assert audit["v3_terminal_review_document_exactly_bound"] is True
    assert audit["v3_authority_command_runtime_and_payload_reuse_forbidden"] is True
    assert audit["replacement_v1_terminal_review_document_exactly_bound"] is True
    assert (
        audit[
            "replacement_v1_authority_command_runtime_and_payload_reuse_forbidden"
        ]
        is True
    )
    assert audit["closed_collector_qualifier_runner_interface_matrix_passed"] is True


def test_qualification_authority_adds_only_replacement_v1_terminal_review(
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
    assert result["predecessor_v1_qualification_terminal_review_binding"] == (
        authority._v1_terminal_review_binding()  # noqa: SLF001
    )
    assert result["predecessor_v2_qualification_terminal_review_binding"] == (
        authority._v2_terminal_review_binding()  # noqa: SLF001
    )
    assert result["predecessor_v3_qualification_terminal_review_binding"] == (
        authority._v3_terminal_review_binding()  # noqa: SLF001
    )
    assert result[
        "predecessor_replacement_v1_qualification_terminal_review_binding"
    ] == authority._replacement_v1_terminal_review_binding()  # noqa: SLF001
    assert "qualification_result_binding" not in result
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
    assert not authority.plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()


def test_source_closure_has_no_v3_authority_command_or_runtime_payload() -> None:
    names = set(authority.runner.SOURCE_PATHS)
    assert {
        "v2_rocm_plan_builder_source",
        "v2_rocm_runner_source",
        "predecessor_v2_qualification_terminal_review",
        "predecessor_v3_qualification_terminal_review",
        "rocm_backend_v3_integrity_replacement_v1_qualification_authority_builder",
    } <= names
    serialized = "\n".join(str(path) for path in authority.runner.SOURCE_PATHS.values())
    assert "backend_v3_qualification_command_receipt" not in serialized
    assert "backend_v3_qualification_authority_2026" not in serialized
    assert ".generated/dev/go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3/" not in serialized


def test_no_replacement_authority_or_attempt_exists() -> None:
    assert not authority.AUTHORITY_OUTPUT.exists()
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
    assert not authority.plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()
