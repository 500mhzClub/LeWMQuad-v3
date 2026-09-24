from __future__ import annotations

import json

from scripts import (
    build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_qualification_authority
    as authority,
)


def test_source_audit_declares_material_backend_and_frozen_science() -> None:
    audit = authority.CPU_BACKEND_SOURCE_AUDIT
    assert audit["material_successor_not_v4_integrity_replacement"] is True
    assert audit["genesis_backend_exact_cpu"] is True
    assert audit["egl_r9700_rendering_selectors_and_preflight_unchanged"] is True
    assert audit["data_panel_model_arms_seeds_updates_evaluation_and_gates_unchanged"] is True
    assert audit["cpu_physics_numerics_may_differ_from_vulkan"] is True
    assert audit["qualification_outputs_forbidden_from_scientific_reuse"] is True


def test_qualification_authority_is_separate_and_cannot_authorize_science(
    monkeypatch,
) -> None:
    science = json.loads(authority.plan_builder.DEFAULT_PLAN_OUTPUT.read_text())
    qualification = json.loads(
        authority.plan_builder.QUALIFICATION_PLAN_OUTPUT.read_text()
    )
    bindings = {
        "prereg": {"path": "/prereg", "sha256": "a" * 64, "byte_count": 1},
        "panel": {"path": "/panel", "sha256": "b" * 64, "byte_count": 1},
        "science": {"path": "/science", "sha256": "c" * 64, "byte_count": 1},
        "qualification": {"path": "/qualification", "sha256": "d" * 64, "byte_count": 1},
        "review": {"path": "/review", "sha256": "e" * 64, "byte_count": 1},
    }
    monkeypatch.setattr(
        authority,
        "_require_binding",
        lambda value, **_kwargs: dict(value),
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
    monkeypatch.setattr(authority, "source_bindings", lambda: {"source": bindings["science"]})
    monkeypatch.setattr(authority, "_validate_review", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        authority.predecessor_authority,
        "dino_declaration_v2",
        lambda: {"checkpoint_binding": bindings["science"]},
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
    assert "qualification_result_binding" not in result
    assert not authority.plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
    assert not authority.plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()


def test_actual_source_closure_contains_both_authority_builders_and_v3_evidence() -> None:
    bindings = authority.source_bindings()
    assert "cpu_backend_qualification_authority_builder" in bindings
    assert "cpu_backend_scientific_authority_builder" in bindings
    assert bindings["predecessor_v3_failure_terminal"]["sha256"] == (
        authority.runner.PREDECESSOR_V3_TERMINAL_SHA256
    )
