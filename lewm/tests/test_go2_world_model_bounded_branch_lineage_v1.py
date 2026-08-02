from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import build_go2_world_model_bounded_branch_experiment_authority_v1 as authority
from scripts import build_go2_world_model_bounded_branch_experiment_plan_v1 as plan_builder
from scripts import build_go2_world_model_bounded_branch_scene_panel_v1 as selector
from scripts import evaluate_go2_world_model_visual_domain_parity_v1 as parity_evaluator


def _binding(path: str, marker: str = "a") -> dict[str, object]:
    return {"path": path, "file_sha256": marker * 64, "byte_count": 1}


def _normalized_panel() -> list[dict[str, object]]:
    result = []
    for role in plan_builder.ROLE_NAMES:
        for family in pilot.FAMILIES:
            for slot in range(2):
                scene_id = f"{role}-{family}-{slot}"
                result.append(
                    {
                        "role": role,
                        "family": family,
                        "scene_id": scene_id,
                        "scene_manifest_binding": _binding(
                            f"/tmp/{scene_id}/manifest.json"
                        ),
                        "scene_genesis_binding": _binding(
                            f"/tmp/{scene_id}/genesis_scene.json"
                        ),
                        "states": [
                            {
                                "state_id": f"{scene_id}-state-{index}",
                                "history_action_ids": list(
                                    plan_builder.HISTORY_PANEL[index]
                                ),
                                "target_xy_m": [float(slot), float(index)],
                            }
                            for index in range(8)
                        ],
                    }
                )
    return result


def _plan_states(panel: list[dict[str, object]]) -> list[dict[str, object]]:
    states = []
    for scene in panel:
        for state_index, state in enumerate(scene["states"]):
            states.append(
                {
                    "state_id": state["state_id"],
                    "role": scene["role"],
                    "family": scene["family"],
                    "scene_id": scene["scene_id"],
                    "scene_manifest_binding": scene["scene_manifest_binding"],
                    "scene_genesis_binding": scene["scene_genesis_binding"],
                    "scene_generation": None,
                    "group_index": len(states),
                    "state_index_in_scene": state_index,
                    "history_action_ids": state["history_action_ids"],
                    "candidate_action_ids": list(range(pilot.ACTION_COUNT)),
                    "sentinel_duplicate_action_id": None,
                    "target_xy_m": state["target_xy_m"],
                }
            )
    return states


def test_plan_state_lineage_rejects_one_tampered_selected_state():
    panel = _normalized_panel()
    plan = {"states": _plan_states(panel)}
    plan_builder._validate_plan_scene_panel_match_v1(
        plan, normalized_panel=panel
    )
    tampered = copy.deepcopy(plan)
    tampered["states"][137]["target_xy_m"] = [999.0, 999.0]
    with pytest.raises(plan_builder.BoundedBranchPlanError, match="exactly match"):
        plan_builder._validate_plan_scene_panel_match_v1(
            tampered, normalized_panel=panel
        )


def test_scene_panel_freeze_field_contract_is_exact_and_unique():
    assert len(plan_builder.SCENE_PANEL_FREEZE_FIELDS) == len(
        set(plan_builder.SCENE_PANEL_FREEZE_FIELDS)
    )
    assert plan_builder.SCENE_PANEL_FREEZE_FIELDS == (
        "scene_panel_binding",
        "scene_panel_schema",
        "scene_selection_contract",
        "scene_corpus_manifest_bindings",
        "scene_inventory_unique_train_scenes",
        "scene_eligible_counts_by_family",
        "scene_excluded_scene_ids_sha256",
        "scene_selection_rows",
    )


def test_scene_panel_receipt_must_match_exact_bound_document(tmp_path, monkeypatch):
    panel = {
        "schema": selector.PANEL_SCHEMA,
        "selection_contract": {"seed": selector.SELECTION_SEED},
        "corpus_manifest_bindings": [],
        "inventory_unique_train_scenes": 32,
        "eligible_counts_by_family": {family: 4 for family in pilot.FAMILIES},
        "excluded_scene_ids_sha256": "b" * 64,
        "scenes": [],
    }
    path = tmp_path / "scene-panel.json"
    path.write_text(json.dumps(panel, sort_keys=True) + "\n", encoding="utf-8")
    bound = pilot.file_binding(path)
    normalized = _normalized_panel()
    monkeypatch.setattr(
        plan_builder,
        "_validate_panel",
        lambda value, excluded_scene_ids: normalized,
    )
    reopened, freeze = plan_builder._validate_scene_panel_receipt_v1(
        panel,
        binding=bound,
        excluded_scene_ids=set(),
    )
    assert reopened == normalized
    assert freeze["scene_panel_binding"] == bound
    assert set(freeze) == set(plan_builder.SCENE_PANEL_FREEZE_FIELDS)
    tampered = copy.deepcopy(panel)
    tampered["inventory_unique_train_scenes"] = 31
    with pytest.raises(plan_builder.BoundedBranchPlanError, match="document/binding"):
        plan_builder._validate_scene_panel_receipt_v1(
            tampered,
            binding=bound,
            excluded_scene_ids=set(),
        )


def test_scene_selector_rejects_symlinked_selected_scene_path(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    escape = tmp_path / "outside-selected-scene"
    escape.mkdir()
    (escape / "genesis_scene.json").write_text("{}\n", encoding="utf-8")
    inventory = []
    for family_index, family in enumerate(pilot.FAMILIES):
        for index in range(4):
            scene_id = f"{family}-ordinary-{index}"
            semantic_digest = hashlib.sha256(scene_id.encode()).hexdigest()
            manifest_document = {
                "scene_id": scene_id,
                "family": family,
                "split": "train",
                "manifest_sha256": semantic_digest,
            }
            scene_root = campaign / "train" / family / scene_id
            scene_root.parent.mkdir(parents=True, exist_ok=True)
            if family_index == 0 and index == 0:
                scene_root.symlink_to(escape, target_is_directory=True)
                manifest_path = escape / "manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest_document) + "\n", encoding="utf-8"
                )
            else:
                scene_root.mkdir()
                (scene_root / "manifest.json").write_text(
                    json.dumps(manifest_document) + "\n", encoding="utf-8"
                )
                (scene_root / "genesis_scene.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                manifest_path = scene_root / "manifest.json"
            inventory.append(
                {
                    "family": family,
                    "scene_id": scene_id,
                    "manifest_sha256": semantic_digest,
                    "campaign_root": str(campaign),
                    "relative_dir": f"train/{family}/{scene_id}",
                    "inventory_rank": hashlib.sha256(
                        f"inventory/{scene_id}".encode()
                    ).hexdigest(),
                }
            )
    monkeypatch.setattr(selector, "SCENE_CORPUS_ROOT", tmp_path)
    monkeypatch.setattr(selector, "_load_inventory", lambda: ([], inventory))
    with pytest.raises(selector.BoundedBranchScenePanelError, match="symlink"):
        selector.derive_scene_panel_v1(excluded_scene_ids=set())


def test_authority_gate_rejects_fabricated_scene_selection_evidence(monkeypatch):
    calibration = {
        "calibration_receipt_binding": _binding("/tmp/calibration.json", "1"),
        "calibration_terminal_binding": _binding("/tmp/terminal.json", "2"),
        "calibration_terminal_review_binding": _binding("/tmp/review.json", "3"),
        "excluded_scene_ids": ["calibration-scene"],
        "calibration_wall_seconds": 80.0,
        "calibration_stored_rgb_bytes": 1000,
        "calibration_gpu_baseline_used_bytes": 100,
        "calibration_gpu_peak_used_bytes": 200,
        "calibration_gpu_peak_delta_bytes": 100,
        "selected_device_total_vram_bytes": 10_000,
    }
    model = {
        "progression_analysis_binding": _binding("/tmp/analysis.json", "4"),
        "training_result_binding": _binding("/tmp/training.json", "5"),
        "progression_proxy_routing": {"decision": "DELTA_PROXY_MEANINGFUL"},
        "checkpoint_panel_bindings": {},
        "model_observational_scene_ids": ["observational-scene"],
        "model_observational_scene_count": 1,
        "predecessor_terminal_access_binding": _binding("/tmp/access.json", "6"),
        "predecessor_index_bindings": {},
        "predecessor_place_manifest_binding": _binding("/tmp/place.json", "7"),
        "training_pack_role_bindings": {},
        "training_pack_metadata_bindings": {},
    }
    scene = {
        "scene_panel_binding": _binding("/tmp/scene-panel.json", "8"),
        "scene_panel_schema": selector.PANEL_SCHEMA,
        "scene_selection_contract": {"seed": selector.SELECTION_SEED},
        "scene_corpus_manifest_bindings": [],
        "scene_inventory_unique_train_scenes": 64,
        "scene_eligible_counts_by_family": {family: 8 for family in pilot.FAMILIES},
        "scene_excluded_scene_ids_sha256": "9" * 64,
        "scene_selection_rows": [],
    }
    visual_parity = {
        "result_binding": _binding("/tmp/visual-result.json", "a"),
        "terminal_binding": _binding("/tmp/visual-terminal.json", "9"),
        "review_binding": _binding("/tmp/visual-review.json", "b"),
        "source_rgb_reference_binding": _binding("/tmp/source-rgb.json", "c"),
        "candidate_rgb_panel_binding": _binding("/tmp/candidate-rgb.json", "d"),
        "source_producer_lineage": {"schema": "source-lineage"},
        "candidate_producer_lineage": {"schema": "candidate-lineage"},
        "candidate_collector_source_binding": _binding(
            str(
                plan_builder.REPO_ROOT
                / "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"
            ),
            "1",
        ),
        "candidate_renderer_source_binding": _binding(
            str(
                plan_builder.REPO_ROOT
                / "scripts/render_replay_v03.py"
            ),
            "2",
        ),
        "reference_renderer_source_binding": _binding(
            str(plan_builder.REPO_ROOT / "scripts/render_replay_v03.py"), "e"
        ),
        "reference_texture_source_binding": _binding(
            str(
                plan_builder.REPO_ROOT
                / "lewm_genesis/lewm_genesis/textures.py"
            ),
            "0",
        ),
        "evaluator_source_binding": _binding("/tmp/evaluator.py", "f"),
        "selected_texture_asset_bindings_by_scene": {},
        "evidence_scene_ids": ["visual-parity-scene"],
        "comparison_contract": {},
        "thresholds": {},
        "measurements": {},
    }
    calibration.update({
        "visual_domain_parity_result_binding": visual_parity["result_binding"],
        "visual_domain_parity_terminal_binding": visual_parity[
            "terminal_binding"
        ],
        "visual_domain_parity_review_binding": visual_parity["review_binding"],
    })
    gate = {
        "schema": authority.GATE_SCHEMA,
        "status": "PASS",
        "authority_granted_by_this_document": False,
        **calibration,
        **model,
        **scene,
        "visual_domain_parity_result_binding": visual_parity["result_binding"],
        "visual_domain_parity_terminal_binding": visual_parity[
            "terminal_binding"
        ],
        "visual_domain_parity_review_binding": visual_parity["review_binding"],
        "visual_domain_parity_freeze": visual_parity,
    }
    documents = {
        calibration["calibration_receipt_binding"]["path"]: {},
        calibration["calibration_terminal_binding"]["path"]: {},
        calibration["calibration_terminal_review_binding"]["path"]: {},
        model["progression_analysis_binding"]["path"]: {},
        scene["scene_panel_binding"]["path"]: {},
        visual_parity["result_binding"]["path"]: {},
        visual_parity["terminal_binding"]["path"]: {},
        visual_parity["review_binding"]["path"]: {},
    }
    all_bindings = {
        value["path"]: value
        for value in (
            *calibration.values(),
            *model.values(),
            *scene.values(),
            *visual_parity.values(),
        )
        if isinstance(value, dict) and "path" in value
    }

    monkeypatch.setattr(
        authority.pilot,
        "require_binding",
        lambda value, label: dict(value),
    )

    def fake_read(path, *, expected_sha256, expected_byte_count, label):
        binding = all_bindings[str(path)]
        return documents[str(path)], binding

    monkeypatch.setattr(authority.pilot, "read_bound_json", fake_read)
    monkeypatch.setattr(
        plan_builder,
        "_validate_calibration_gate",
        lambda *args, **kwargs: calibration,
    )
    monkeypatch.setattr(
        plan_builder,
        "_validate_model_panel_freeze",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        plan_builder,
        "_validate_scene_panel_receipt_v1",
        lambda *args, **kwargs: (_normalized_panel(), scene),
    )
    monkeypatch.setattr(
        plan_builder,
        "_validate_visual_domain_parity_gate_v1",
        lambda *args, **kwargs: visual_parity,
    )
    validated, _panel = authority._validate_gate(
        gate, binding=_binding("/tmp/gate.json", "f")
    )
    assert validated["scene_selection_contract"] == scene[
        "scene_selection_contract"
    ]
    fabricated = copy.deepcopy(gate)
    fabricated["scene_inventory_unique_train_scenes"] = 63
    with pytest.raises(authority.BoundedBranchAuthorityError, match="derivation"):
        authority._validate_gate(
            fabricated, binding=_binding("/tmp/gate.json", "f")
        )


def _write_json(path: Path, value: object) -> dict[str, object]:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return pilot.file_binding(path)


def _visual_parity_documents(
    tmp_path: Path,
    *,
    comparison_contract: dict[str, object] | None = None,
    reference_candidate_exact_match_count: int = 32,
    evaluator_source: Path | None = None,
):
    source_reference = _write_json(tmp_path / "source-rgb-panel.json", {})
    candidate_panel = _write_json(tmp_path / "candidate-rgb-panel.json", {})
    evidence_scene_ids = [
        f"visual-parity-scene-{index:02d}" for index in range(8)
    ]
    category_assets = {}
    for category in ("floor", "wall", "obstacle"):
        candidates = sorted(
            path
            for path in (plan_builder.REPO_ROOT / "assets/textures" / category).iterdir()
            if path.is_file() and not path.is_symlink()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        assert candidates
        category_assets[category] = pilot.file_binding(candidates[0])
    selected_texture_assets = {
        scene_id: copy.deepcopy(category_assets)
        for scene_id in evidence_scene_ids
    }
    result = {
        "schema": plan_builder.VISUAL_DOMAIN_PARITY_RESULT_SCHEMA,
        "status": plan_builder.VISUAL_DOMAIN_PARITY_RESULT_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "development_only": True,
        "protected_material_opened": False,
        "comparison_contract": copy.deepcopy(
            comparison_contract
            if comparison_contract is not None
            else plan_builder.VISUAL_DOMAIN_PARITY_COMPARISON_CONTRACT
        ),
        "thresholds": copy.deepcopy(
            plan_builder.VISUAL_DOMAIN_PARITY_THRESHOLDS
        ),
        "measurements": {
            "scene_count": 8,
            "poses_per_scene": 4,
            "reference_frame_count": 32,
            "candidate_frame_count": 32,
            "duplicate_frame_count": 32,
            "families": list(pilot.FAMILIES),
            "reference_candidate_exact_match_count": (
                reference_candidate_exact_match_count
            ),
            "candidate_duplicate_exact_match_count": 32,
            "maximum_reference_candidate_normalized_l1": (
                0.0 if reference_candidate_exact_match_count == 32 else 1.0 / 255.0
            ),
            "minimum_reference_candidate_rgb_ssim": (
                1.0 if reference_candidate_exact_match_count == 32 else 0.99
            ),
        },
        "evidence_scene_ids": evidence_scene_ids,
        "source_rgb_reference_binding": source_reference,
        "candidate_rgb_panel_binding": candidate_panel,
        "source_producer_lineage": {"schema": "synthetic-source-lineage"},
        "candidate_producer_lineage": {"schema": "synthetic-candidate-lineage"},
        "candidate_collector_source_binding": pilot.file_binding(
            plan_builder.REPO_ROOT
            / "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"
        ),
        "candidate_renderer_source_binding": pilot.file_binding(
            plan_builder.REPO_ROOT / "scripts/render_replay_v03.py"
        ),
        "reference_renderer_source_binding": pilot.file_binding(
            plan_builder.REPO_ROOT / "scripts/render_replay_v03.py"
        ),
        "reference_texture_source_binding": pilot.file_binding(
            plan_builder.REPO_ROOT / "lewm_genesis/lewm_genesis/textures.py"
        ),
        "evaluator_source_binding": pilot.file_binding(
            evaluator_source or Path(parity_evaluator.__file__)
        ),
        "selected_texture_asset_bindings_by_scene": selected_texture_assets,
    }
    result_binding = _write_json(tmp_path / "visual-parity-result.json", result)
    terminal = {
        "schema": pilot.TEXTURED_V03_PARITY_TERMINAL_SCHEMA,
        "status": pilot.TEXTURED_V03_PARITY_TERMINAL_SUCCESS_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "authorizes_retry_or_resume": False,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "attempt_id": "synthetic-parity-attempt",
        "plan_binding": _binding("/tmp/parity-plan.json", "1"),
        "authority_binding": _binding("/tmp/parity-authority.json", "2"),
        "reservation_binding": _binding("/tmp/parity-reservation.json", "3"),
        "source_review_binding": _binding("/tmp/parity-source-review.json", "4"),
        "source_commit": "5" * 40,
        "scene_result_bindings": [
            _binding(f"/tmp/parity-scene-{index}.json", "6")
            for index in range(len(pilot.FAMILIES))
        ],
        "generation_receipt_binding": _binding("/tmp/generation.json", "7"),
        "candidate_panel_binding": candidate_panel,
        "parity_result_binding": result_binding,
        "graphics_preflight": {"passed": True},
        "disk_preflight": {"passed": True},
        "wall_seconds": 10.0,
        "wall_ceiling_seconds": 100.0,
        "total_output_bytes_before_terminal": 100,
        "completed_at": "2026-08-02T11:59:00Z",
        "terminal_reviewer": "independent-terminal-reviewer",
    }
    terminal_binding = _write_json(
        tmp_path / "visual-parity-terminal.json", terminal
    )
    review = {
        "schema": plan_builder.VISUAL_DOMAIN_PARITY_REVIEW_SCHEMA,
        "status": plan_builder.VISUAL_DOMAIN_PARITY_REVIEW_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "result_binding": result_binding,
        "terminal_binding": terminal_binding,
        "reviewer": {
            "identity": "independent-visual-domain-reviewer",
            "independence_basis": "separate recomputation and source review",
        },
        "reviewed_at": "2026-08-02T12:00:00Z",
        "checks": {
            "result_recomputed_from_bound_panels": True,
            "historical_source_lineage_verified": True,
            "candidate_authority_and_source_closure_verified": True,
            "all_eight_families_exactly_once": True,
            "all_32_reference_candidate_pairs_pixel_exact": True,
            "all_32_candidate_duplicate_pairs_pixel_exact": True,
            "candidate_frames_independently_rendered_not_copied": True,
            "scene_pose_texture_and_source_lineage_exact": True,
            "sensor_geometry_flags_alone_rejected_as_insufficient": True,
            "no_statistical_inference_claimed": True,
            "no_protected_material": True,
        },
        "remaining_findings": [],
    }
    review_binding = _write_json(tmp_path / "visual-parity-review.json", review)
    return result, result_binding, review, review_binding


def test_visual_parity_gate_requires_exact_textured_v03_camera_and_pixels(
    tmp_path, monkeypatch
):
    result, result_binding, review, review_binding = _visual_parity_documents(
        tmp_path
    )
    monkeypatch.setattr(
        parity_evaluator, "evaluate_v1", lambda **_kwargs: copy.deepcopy(result)
    )
    monkeypatch.setattr(
        pilot,
        "validate_textured_v03_parity_prerequisites",
        lambda **_kwargs: {
            "result_binding": result_binding,
            "terminal_binding": review["terminal_binding"],
            "review_binding": review_binding,
        },
    )
    freeze = plan_builder._validate_visual_domain_parity_gate_v1(
        result,
        result_binding=result_binding,
        review=review,
        review_binding=review_binding,
    )
    assert freeze["comparison_contract"]["required_native_resolution"] == [224, 224]
    assert freeze["comparison_contract"]["required_raw_manifest_fov_deg"] == 78.323
    assert freeze["comparison_contract"]["genesis_fov_contract"] == (
        "pass_raw_fov_deg_directly_as_yfov"
    )
    assert freeze["comparison_contract"]["horizontal_to_vertical_fov_conversion_allowed"] is False
    assert freeze["comparison_contract"]["native_downsampling_allowed"] is False
    assert freeze["comparison_contract"]["required_camera_render_call"] == {
        "rgb": True,
        "depth": False,
    }
    assert freeze["comparison_contract"]["pairing"] == (
        "same_scene_manifest_same_camera_pose_same_pair_id"
    )
    assert freeze["thresholds"]["exact_frames_per_domain"] == 32
    assert freeze["thresholds"][
        "required_reference_candidate_exact_match_count"
    ] == 32
    assert freeze["reference_renderer_source_binding"] == pilot.file_binding(
        plan_builder.REPO_ROOT / "scripts/render_replay_v03.py"
    )
    assert freeze["reference_texture_source_binding"] == pilot.file_binding(
        plan_builder.REPO_ROOT / "lewm_genesis/lewm_genesis/textures.py"
    )


def test_v3_prerequisite_rejects_shallow_valid_pass_terminal_with_forged_chain(
    tmp_path,
):
    _result, result_binding, review, review_binding = _visual_parity_documents(
        tmp_path
    )
    with pytest.raises(
        pilot.PilotContractError,
        match="terminal lineage did not validate",
    ):
        pilot.validate_textured_v03_parity_prerequisites(
            result_binding=result_binding,
            terminal_binding=review["terminal_binding"],
            review_binding=review_binding,
        )


def test_visual_parity_gate_rejects_current_640x480_converted_fov_contract(
    tmp_path
):
    altered = copy.deepcopy(plan_builder.VISUAL_DOMAIN_PARITY_COMPARISON_CONTRACT)
    altered["required_native_resolution"] = [640, 480]
    altered["genesis_fov_contract"] = "convert_horizontal_fov_to_vertical_yfov"
    result, result_binding, review, review_binding = _visual_parity_documents(
        tmp_path, comparison_contract=altered
    )
    with pytest.raises(plan_builder.BoundedBranchPlanError, match="fixed exact contract"):
        plan_builder._validate_visual_domain_parity_gate_v1(
            result,
            result_binding=result_binding,
            review=review,
            review_binding=review_binding,
        )


def test_candidate_render_contract_rejects_historical_native_downsample_path():
    historical = copy.deepcopy(pilot.RENDER_CONTRACT)
    historical["native_resolution"] = [640, 480]
    historical["stored_resolution"] = [224, 224]
    with pytest.raises(plan_builder.BoundedBranchPlanError, match="640x480"):
        plan_builder._validate_candidate_render_domain_contract_v1(historical)


def test_visual_parity_gate_rejects_sensor_flags_without_pixel_evidence(
    tmp_path
):
    result = {"same_sensor_flags": True, "textures_requested": True}
    review = {"independently_reviewed": True}
    result_binding = _write_json(tmp_path / "flags-only-result.json", result)
    review_binding = _write_json(tmp_path / "flags-only-review.json", review)
    with pytest.raises(plan_builder.BoundedBranchPlanError, match="result fields"):
        plan_builder._validate_visual_domain_parity_gate_v1(
            result,
            result_binding=result_binding,
            review=review,
            review_binding=review_binding,
        )


def test_visual_parity_gate_rejects_failed_paired_pixel_equivalence(
    tmp_path
):
    result, result_binding, review, review_binding = _visual_parity_documents(
        tmp_path, reference_candidate_exact_match_count=31
    )
    with pytest.raises(plan_builder.BoundedBranchPlanError, match="fixed exact"):
        plan_builder._validate_visual_domain_parity_gate_v1(
            result,
            result_binding=result_binding,
            review=review,
            review_binding=review_binding,
        )


def test_visual_parity_gate_rejects_generic_unreviewed_evaluator_source(tmp_path):
    result, result_binding, review, review_binding = _visual_parity_documents(
        tmp_path, evaluator_source=Path(plan_builder.__file__)
    )
    with pytest.raises(plan_builder.BoundedBranchPlanError, match="source identity"):
        plan_builder._validate_visual_domain_parity_gate_v1(
            result,
            result_binding=result_binding,
            review=review,
            review_binding=review_binding,
        )


def test_visual_parity_gate_rejects_fabricated_pass_metrics_without_recompute(
    tmp_path, monkeypatch
):
    result, result_binding, review, review_binding = _visual_parity_documents(
        tmp_path
    )
    independently_recomputed = copy.deepcopy(result)
    independently_recomputed["status"] = parity_evaluator.FAIL_STATUS
    independently_recomputed["measurements"][
        "reference_candidate_exact_match_count"
    ] = 31
    monkeypatch.setattr(
        parity_evaluator,
        "evaluate_v1",
        lambda **_kwargs: independently_recomputed,
    )
    with pytest.raises(plan_builder.BoundedBranchPlanError, match="exact evaluator recomputation"):
        plan_builder._validate_visual_domain_parity_gate_v1(
            result,
            result_binding=result_binding,
            review=review,
            review_binding=review_binding,
        )


def _synthetic_pixel_panels() -> tuple[dict[str, object], dict[str, object]]:
    source_rows = []
    candidate_rows = []
    scene_ids = []
    for family_index, family in enumerate(pilot.FAMILIES):
        scene_id = f"{family}-parity-scene"
        scene_ids.append(scene_id)
        manifest = _binding(f"/tmp/{scene_id}/manifest.json", str(family_index + 1))
        for pose_index in range(parity_evaluator.POSES_PER_SCENE):
            pair_id = f"{scene_id}/pose_{pose_index:02d}"
            common = {
                "pair_id": pair_id,
                "scene_id": scene_id,
                "family": family,
                "pose_index": pose_index,
                "camera_pose_world": {
                    "position": [0.0, 0.0, 1.0],
                    "lookat": [1.0, 0.0, 1.0],
                    "up": [0.0, 0.0, 1.0],
                },
                "scene_manifest_binding": manifest,
            }
            source_rows.append(
                {
                    **common,
                    "producer_frame_identity": f"reference-{pair_id}",
                    "rgb_binding": _binding(f"/tmp/reference/{pair_id}.png", "a"),
                    "raw_rgb_sha256": "0" * 64,
                }
            )
            candidate_rows.append(
                {
                    **common,
                    "producer_frame_identity": f"candidate-{pair_id}",
                    "duplicate_producer_frame_identity": f"duplicate-{pair_id}",
                    "rgb_binding": _binding(f"/tmp/candidate/{pair_id}.png", "b"),
                    "raw_rgb_sha256": "0" * 64,
                    "duplicate_rgb_binding": _binding(
                        f"/tmp/duplicate/{pair_id}.png", "c"
                    ),
                    "duplicate_raw_rgb_sha256": "0" * 64,
                }
            )
    common_panel = {
        "scene_ids": scene_ids,
        "texture_map": {scene_id: {} for scene_id in scene_ids},
        "mesh_map": {scene_id: [] for scene_id in scene_ids},
    }
    return (
        {**common_panel, "rows": source_rows},
        {**common_panel, "rows": candidate_rows},
    )


def test_visual_parity_evaluator_is_exact_not_statistical(monkeypatch):
    source, candidate = _synthetic_pixel_panels()
    reference = np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3)
    reference = np.asarray(reference % 251, dtype=np.uint8)

    def exact_rgb(_binding_value, *, label):
        return reference.copy(), {}

    monkeypatch.setattr(parity_evaluator, "_read_bound_rgb", exact_rgb)
    monkeypatch.setattr(parity_evaluator, "_raw_rgb_sha256", lambda _rgb: "0" * 64)
    measurements = parity_evaluator._measure(source, candidate)
    assert measurements["reference_candidate_exact_match_count"] == 32
    assert measurements["candidate_duplicate_exact_match_count"] == 32
    assert measurements["maximum_reference_candidate_normalized_l1"] == 0.0
    assert measurements["minimum_reference_candidate_rgb_ssim"] == 1.0
    assert parity_evaluator._passes(measurements)


def test_visual_parity_evaluator_rejects_one_pixel_difference(monkeypatch):
    source, candidate = _synthetic_pixel_panels()
    reference = np.zeros((224, 224, 3), dtype=np.uint8)
    changed_pair = candidate["rows"][17]["pair_id"]

    def one_changed_rgb(_binding_value, *, label):
        rgb = reference.copy()
        if label == f"candidate {changed_pair} RGB":
            rgb[0, 0, 0] = 1
        return rgb, {}

    monkeypatch.setattr(parity_evaluator, "_read_bound_rgb", one_changed_rgb)
    monkeypatch.setattr(parity_evaluator, "_raw_rgb_sha256", lambda _rgb: "0" * 64)
    measurements = parity_evaluator._measure(source, candidate)
    assert measurements["reference_candidate_exact_match_count"] == 31
    assert measurements["candidate_duplicate_exact_match_count"] == 31
    assert measurements["maximum_reference_candidate_normalized_l1"] > 0.0
    assert not parity_evaluator._passes(measurements)


def test_copied_candidate_pixels_without_generation_receipt_cannot_pass(tmp_path):
    fabricated_receipt = {
        "schema": parity_evaluator.CANDIDATE_GENERATION_RECEIPT_SCHEMA,
        "status": parity_evaluator.CANDIDATE_GENERATION_STATUS,
        "render_rows": [],
    }
    receipt_binding = _write_json(tmp_path / "fabricated-generation.json", fabricated_receipt)
    panel = {
        "producer_lineage": {
            "schema": parity_evaluator.CANDIDATE_LINEAGE_SCHEMA,
            "generation_receipt_binding": receipt_binding,
        }
    }
    with pytest.raises(
        parity_evaluator.VisualDomainParityError,
        match="generation receipt changed",
    ):
        parity_evaluator._validate_candidate_lineage(
            panel, source_panel_binding=_binding("/tmp/source-panel.json", "d")
        )


def test_plan_parity_constants_are_owned_by_fixed_evaluator():
    assert plan_builder.VISUAL_DOMAIN_PARITY_RESULT_SCHEMA == parity_evaluator.RESULT_SCHEMA
    assert plan_builder.VISUAL_DOMAIN_PARITY_RESULT_STATUS == parity_evaluator.PASS_STATUS
    assert (
        plan_builder.VISUAL_DOMAIN_PARITY_COMPARISON_CONTRACT
        == parity_evaluator.COMPARISON_CONTRACT
    )
    assert plan_builder.VISUAL_DOMAIN_PARITY_THRESHOLDS == parity_evaluator.THRESHOLDS


def test_calibration_gate_invokes_full_v3_authority_successor_validator(monkeypatch):
    fake = lambda path, marker: _binding(path, marker)  # noqa: E731
    receipt_binding = fake("/tmp/calibration.json", "1")
    terminal_binding = fake("/tmp/terminal.json", "2")
    review_binding = fake("/tmp/review.json", "3")
    authority_binding = fake("/tmp/authority.json", "4")
    parity_prerequisites = {
        "result_binding": fake("/tmp/parity-result.json", "a"),
        "terminal_binding": fake("/tmp/parity-terminal.json", "b"),
        "review_binding": fake("/tmp/parity-review.json", "c"),
    }
    terminal = {
        "schema": plan_builder.calibration_supervisor.TEXTURED_V03_TERMINAL_SCHEMA,
        "status": "COMPLETE_PENDING_TERMINAL_REVIEW",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "scientific_verdict_emitted": False,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "authority_binding": authority_binding,
        "plan_binding": fake("/tmp/plan.json", "5"),
        "predecessor_failure_binding": fake("/tmp/predecessor.json", "6"),
        "source_commit": "a" * 40,
        "attempt_root": "/tmp/calibration-v2",
        "wall_elapsed_seconds": 80.0,
        "wall_ceiling_seconds": 1200.0,
        "phase_receipts": [],
        "physics_result_binding": fake("/tmp/physics.json", "7"),
        "receipt_check_binding": fake("/tmp/check.json", "8"),
        "calibration_receipt_binding": receipt_binding,
        "calibration_decision": "FREEZE_PILOT_CONTRACT",
        "gpu_memory_measurement": {
            "scope": "selected_device_global_vram_not_process_attributed",
            "attribution_limitation": "global",
            "vendor_id": "0x1002",
            "device_id": "0x7551",
            "used_counter_path": "/tmp/used",
            "total_counter_path": "/tmp/total",
            "sample_interval_seconds": 0.05,
            "sample_count": 2,
            "read_errors": 0,
            "baseline_used_bytes": 100,
            "peak_used_bytes": 200,
            "peak_delta_above_baseline_bytes": 100,
            "device_total_bytes": 10_000,
        },
        "failure": None,
        "terminal_reviewer": "reviewer",
        "supervisor_nonce": "9" * 64,
        "visual_domain_parity_prerequisites": parity_prerequisites,
    }
    receipt = {
        "schema": plan_builder.calibration.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA,
        "decision": "FREEZE_PILOT_CONTRACT",
        "calibration_contract": {"excluded_scene_ids": []},
        "resource_measurements": {"stored_rgb_png": {"total_bytes": 1000}},
        "calibration_collection_receipt": terminal["physics_result_binding"],
        "visual_domain_parity_prerequisites": parity_prerequisites,
    }
    monkeypatch.setattr(
        plan_builder.calibration,
        "validate_calibration_receipt_v1",
        lambda value, verify_external_bindings: dict(value),
    )
    monkeypatch.setattr(
        plan_builder,
        "_binding",
        lambda value, label: dict(value),
    )

    def reject_successor(*args, **kwargs):
        raise plan_builder.calibration_supervisor.CalibrationSupervisionError(
            "synthetic successor-boundary rejection"
        )

    monkeypatch.setattr(
        plan_builder.calibration_supervisor,
        "load_and_validate_authority",
        reject_successor,
    )
    with pytest.raises(
        plan_builder.BoundedBranchPlanError,
        match="authority/successor boundary.*synthetic successor",
    ):
        plan_builder._validate_calibration_gate(
            receipt,
            receipt_binding=receipt_binding,
            terminal=terminal,
            terminal_binding=terminal_binding,
            terminal_review={},
            terminal_review_binding=review_binding,
        )


@pytest.mark.parametrize(
    "receipt_schema",
    [
        plan_builder.calibration.CALIBRATION_RECEIPT_SCHEMA,
        plan_builder.calibration.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA,
    ],
)
@pytest.mark.parametrize(
    "mixed_field",
    [
        "terminal_schema",
        "authority_schema",
        "authority_status",
        "plan_purpose",
        "physics_purpose",
        "check_purpose",
    ],
)
def test_calibration_gate_profile_rejects_every_cross_version_mix(
    receipt_schema, mixed_field
):
    selected = plan_builder._calibration_gate_profile({"schema": receipt_schema})
    other_schema = (
        plan_builder.calibration.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA
        if receipt_schema == plan_builder.calibration.CALIBRATION_RECEIPT_SCHEMA
        else plan_builder.calibration.CALIBRATION_RECEIPT_SCHEMA
    )
    other = plan_builder._calibration_gate_profile({"schema": other_schema})
    links = {
        "terminal_schema": selected["terminal_schema"],
        "authority_schema": selected["authority_schema"],
        "authority_status": selected["authority_status"],
        "plan_purpose": selected["purpose"],
        "physics_purpose": selected["purpose"],
        "check_purpose": selected["purpose"],
    }
    replacement_key = (
        "purpose" if mixed_field.endswith("purpose") else mixed_field
    )
    links[mixed_field] = other[replacement_key]
    with pytest.raises(
        plan_builder.BoundedBranchPlanError,
        match="cross-version evidence mix rejected",
    ):
        plan_builder._require_calibration_profile_links(selected, **links)


@pytest.mark.parametrize(
    "receipt_schema",
    [
        plan_builder.calibration.CALIBRATION_RECEIPT_SCHEMA,
        plan_builder.calibration.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA,
    ],
)
def test_calibration_gate_profile_accepts_only_its_exact_version(receipt_schema):
    profile = plan_builder._calibration_gate_profile({"schema": receipt_schema})
    plan_builder._require_calibration_profile_links(
        profile,
        terminal_schema=profile["terminal_schema"],
        authority_schema=profile["authority_schema"],
        authority_status=profile["authority_status"],
        plan_purpose=profile["purpose"],
        physics_purpose=profile["purpose"],
        check_purpose=profile["purpose"],
    )


def test_bounded_textured_plan_rejects_legacy_v2_calibration_receipt(monkeypatch):
    receipt = {
        "schema": plan_builder.calibration.CALIBRATION_RECEIPT_SCHEMA,
        "decision": "FREEZE_PILOT_CONTRACT",
    }
    monkeypatch.setattr(
        plan_builder.calibration,
        "validate_calibration_receipt_v1",
        lambda value, verify_external_bindings: dict(value),
    )
    with pytest.raises(
        plan_builder.BoundedBranchPlanError,
        match="requires the exact textured-v03 V3 calibration",
    ):
        plan_builder._validate_calibration_gate(
            receipt,
            receipt_binding={},
            terminal={},
            terminal_binding={},
            terminal_review={},
            terminal_review_binding={},
        )


@pytest.mark.parametrize(
    ("gate_key", "freeze_key"),
    [
        ("visual_domain_parity_result_binding", "result_binding"),
        ("visual_domain_parity_terminal_binding", "terminal_binding"),
        ("visual_domain_parity_review_binding", "review_binding"),
    ],
)
def test_bounded_plan_rejects_calibration_vs_bounded_parity_a_b_mix(
    gate_key, freeze_key
):
    result = _binding("/tmp/parity-result.json", "a")
    terminal = _binding("/tmp/parity-terminal.json", "b")
    review = _binding("/tmp/parity-review.json", "c")
    gate = {
        "visual_domain_parity_result_binding": result,
        "visual_domain_parity_terminal_binding": terminal,
        "visual_domain_parity_review_binding": review,
    }
    freeze = {
        "result_binding": result,
        "terminal_binding": terminal,
        "review_binding": review,
    }
    gate[gate_key] = _binding(f"/tmp/other-{gate_key}.json", "d")
    with pytest.raises(
        plan_builder.BoundedBranchPlanError,
        match="differs from the V3 calibration prerequisite",
    ):
        plan_builder._require_calibration_parity_identity(gate, freeze)
