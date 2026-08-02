from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = (
    ROOT / "scripts/analyze_go2_world_model_counterfactual_calibration_v1.py"
)


def _load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "counterfactual_calibration_analyzer_v1", ANALYZER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding(marker: str) -> dict[str, object]:
    return {
        "path": f"/synthetic/{marker}.json",
        "file_sha256": f"{len(marker) % 16:x}" * 64,
        "byte_count": len(marker) + 1,
    }


def _collection(*, collapse_last_repeat_action: bool = False) -> dict[str, object]:
    states = []
    frame_receipts = {}
    for group_index in range(16):
        state_id = f"state-{group_index}"
        branches = []
        for action in range(9):
            branches.append({
                "action_id": action,
                "clipped": False,
                "physical_fell": False,
                "physical_tipped": False,
                "physical_target_progress_m": action * 0.1,
                "physical_path_length_m": 1.0 + action * 0.01,
                "endpoint_state": {
                    "base_pos_world": [action * 0.1, 0.0, 0.3],
                    "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            })
        repeat_action = group_index % 9
        if collapse_last_repeat_action and group_index == 8:
            repeat_action = 0
        sentinel = copy.deepcopy(branches[repeat_action])
        sentinel["action_id"] = repeat_action
        states.append({
            "state": {"scene_id": f"scene-{group_index // 2}"},
            "document": {"state": {"state_index_in_scene": group_index % 2}},
            "branches": [*branches, sentinel],
        })
        for context_index in range(3):
            identity = f"{state_id}:context:{context_index}"
            low_info_reasons = (
                [
                    "near_forward_geometry",
                    "low_rgb_texture",
                    "near_wall_depth",
                ]
                if group_index == 0 and context_index == 0
                else []
            )
            frame_receipts[identity] = {
                "frame_identity": identity,
                "byte_count": 100 + context_index,
                "low_information": bool(low_info_reasons),
                "low_info_reasons": low_info_reasons,
            }
        for target_index in range(10):
            identity = f"{state_id}:candidate:{target_index}"
            frame_receipts[identity] = {
                "frame_identity": identity,
                "byte_count": 200 + target_index,
                "low_information": False,
                "low_info_reasons": [],
            }
    scene_metrics = [
        {
            "physics_build_wall_seconds": 0.1,
            "render_scene_build_wall_seconds": 0.1,
            "common_prefix_step_wall_seconds": 0.2,
            "branch_step_wall_seconds": 0.1,
            "native_render_wall_seconds": 0.2,
            "camera_quality_resize_wall_seconds": 0.1,
            "png_encode_write_hash_wall_seconds": 0.1,
            "post_lockstep_receipt_wall_seconds": 0.2,
            "scene_total_wall_seconds": 1.0,
        }
        for _ in range(8)
    ]
    return {
        "purpose": "sizing_calibration_only",
        "counts": {
            "scenes": 8,
            "states": 16,
            "roles": {"calibration": 16},
            "candidate_branches": 144,
            "sentinel_branches": 16,
            "total_branches": 160,
        },
        "states": states,
        "frame_receipts": frame_receipts,
        "document": {
            "attempt_id": "calibration-attempt-v1",
            "scene_metrics": scene_metrics,
            "collection_wall_seconds": 8.5,
        },
    }


def _partition(*, unique_count: int, salt: int) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    for group_index in range(unique_count):
        groups.append({
            "identity_sha256": f"{salt * 16 + group_index + 1:064x}",
            "action_ids": [
                action_id
                for action_id in range(9)
                if action_id % unique_count == group_index
            ],
        })
    return {
        "unique_count": unique_count,
        "collapsed": unique_count < 9,
        "groups": groups,
    }


def _partition_from_action_groups(
    groups: list[list[int]], *, salt: int
) -> dict[str, object]:
    return {
        "unique_count": len(groups),
        "collapsed": len(groups) < 9,
        "groups": [
            {
                "identity_sha256": f"{salt + index + 1:064x}",
                "action_ids": action_ids,
            }
            for index, action_ids in enumerate(groups)
        ],
    }


def _textured_v03_collection(
    *,
    identifiable: bool,
    identifiable_state_indices: set[int] | None = None,
) -> dict[str, object]:
    analyzer = _load_analyzer()
    collection = _collection()
    collection["purpose"] = analyzer.TEXTURED_V03_CALIBRATION_PURPOSE
    plan_states = []
    for state_index, state in enumerate(collection["states"]):
        state_is_identifiable = (
            identifiable
            if identifiable_state_indices is None
            else state_index in identifiable_state_indices
        )
        scene_index = state_index // 2
        family = analyzer.producer_contract.FAMILIES[scene_index]
        history = [state_index % 3, (state_index + 1) % 3]
        state_id = f"state-{state_index}"
        state["state"] = {
            "state_id": state_id,
            "family": family,
            "scene_id": f"scene-{scene_index}",
        }
        state["context"] = {"history_action_ids": history}
        partition_counts = {
            "executed_tape": 9 if state_is_identifiable else 1,
            "physical_trajectory": 1,
            "endpoint_pose": 2 if state_is_identifiable else 1,
            "physical_outcome": 3 if state_is_identifiable else 1,
            "stored_rgb_file": 1,
            "stored_rgb_pixels": 9 if state_is_identifiable else 1,
        }
        state["document"] = {
            "schema": analyzer.producer_contract.TEXTURED_V03_STATE_RECEIPT_SCHEMA,
            "candidate_response_audit": {
                "schema": (
                    analyzer.producer_contract
                    .TEXTURED_V03_CANDIDATE_RESPONSE_AUDIT_SCHEMA
                ),
                "candidate_action_ids": list(range(9)),
                **{
                    name: _partition(
                        unique_count=unique_count,
                        salt=state_index * 8 + partition_index,
                    )
                    for partition_index, (name, unique_count) in enumerate(
                        partition_counts.items()
                    )
                },
            },
            "sentinel_audit": {"passed": True},
            "render_sentinel_audit": {
                "passed": True,
                "stored_rgb_equal": True,
            },
        }
        plan_states.append({
            "state_id": state_id,
            "family": family,
            "scene_id": f"scene-{scene_index}",
            "history_action_ids": history,
        })
    collection["plan"] = {
        "document": {
            "purpose": analyzer.TEXTURED_V03_CALIBRATION_PURPOSE,
            "render_contract": analyzer.producer_contract.TEXTURED_V03_RENDER_CONTRACT,
            "visual_domain_parity_result_binding": _binding("parity-result"),
            "visual_domain_parity_terminal_binding": _binding("parity-terminal"),
            "visual_domain_parity_review_binding": _binding("parity-review"),
        },
        "states": plan_states,
    }
    collection["document"]["attempt_id"] = "calibration-textured-v03-v3"
    for frame_index, frame in enumerate(collection["frame_receipts"].values()):
        frame["camera_valid"] = True
        frame["pixel_sha256"] = f"{frame_index + 1:064x}"
    first_frame = next(iter(collection["frame_receipts"].values()))
    first_frame["low_info_reasons"] = [
        "camera_safety_unresolved",
        *first_frame["low_info_reasons"],
    ]
    first_frame["low_information"] = True
    return collection


def test_analyzer_derives_tolerances_and_all_action_repeat_coverage() -> None:
    analyzer = _load_analyzer()
    receipt = analyzer.derive_calibration_receipt_v1(
        _collection(),
        collection_binding=_binding("collection"),
        analyzer_binding=_binding("analyzer"),
        checker_binding=_binding("checker"),
        joiner_binding=_binding("joiner"),
    )
    assert receipt["decision"] == "FREEZE_PILOT_CONTRACT"
    assert receipt["calibration_contract"]["progress_tolerance_m"] == 1e-6
    assert receipt["calibration_contract"]["path_length_tolerance_m"] == 1e-6
    assert receipt["repeatability_analysis"]["repeated_action_ids"] == [
        index % 9 for index in range(16)
    ]
    assert receipt["repeatability_analysis"]["all_requested_primitives_covered"] is True
    assert receipt["repeatability_analysis"]["interpretation"] == (
        "deterministic_replay_gate_not_empirical_noise_estimate"
    )
    assert receipt["calibration_contract"]["tolerance_derivation"][
        "empirical_noise_scale_estimated"
    ] is False
    assert receipt["visual_validation"]["visual_domain_fidelity_claimed"] is False
    assert receipt["resource_measurements"]["stored_rgb_png"]["total_frames"] == 208
    assert receipt["resource_measurements"]["outcome_counts"][
        "camera_invalid_frames"
    ] == 0
    assert receipt["resource_measurements"]["low_information_strata"] == {
        "total_frames": 1,
        "context_frames": 1,
        "target_frames": 0,
        "reason_counts": {
            "low_rgb_texture": 1,
            "near_wall_depth": 1,
            "near_forward_geometry": 1,
        },
        "context_reason_counts": {
            "low_rgb_texture": 1,
            "near_wall_depth": 1,
            "near_forward_geometry": 1,
        },
        "target_reason_counts": {
            "low_rgb_texture": 0,
            "near_wall_depth": 0,
            "near_forward_geometry": 0,
        },
        "frame_receipt_tags_present": True,
        "hard_invalid_frames": 0,
    }
    analyzer.validate_calibration_receipt_v1(
        receipt, verify_external_bindings=False
    )

    inconsistent = copy.deepcopy(receipt)
    inconsistent["resource_measurements"]["low_information_strata"][
        "reason_counts"
    ]["near_wall_depth"] = 0
    with pytest.raises(
        analyzer.CalibrationAnalysisError, match="resource"
    ):
        analyzer.validate_calibration_receipt_v1(
            inconsistent, verify_external_bindings=False
        )


def test_analyzer_rejects_repeat_panel_that_misses_a_primitive() -> None:
    analyzer = _load_analyzer()
    with pytest.raises(analyzer.CalibrationAnalysisError, match="all nine"):
        analyzer.derive_calibration_receipt_v1(
            _collection(collapse_last_repeat_action=True),
            collection_binding=_binding("collection"),
            analyzer_binding=_binding("analyzer"),
            checker_binding=_binding("checker"),
            joiner_binding=_binding("joiner"),
        )


def test_analyzer_rejects_nonexact_repeat_as_contract_drift() -> None:
    analyzer = _load_analyzer()
    collection = _collection()
    collection["states"][0]["branches"][-1]["physical_target_progress_m"] += 0.01
    with pytest.raises(analyzer.CalibrationAnalysisError, match="exact deterministic"):
        analyzer.derive_calibration_receipt_v1(
            collection,
            collection_binding=_binding("collection"),
            analyzer_binding=_binding("analyzer"),
            checker_binding=_binding("checker"),
            joiner_binding=_binding("joiner"),
        )


def test_textured_v03_measures_collapses_and_freezes_identifiable_states() -> None:
    analyzer = _load_analyzer()
    receipt = analyzer.derive_calibration_receipt_v1(
        _textured_v03_collection(identifiable=True),
        collection_binding=_binding("textured-collection"),
        analyzer_binding=_binding("textured-analyzer"),
        checker_binding=_binding("textured-checker"),
        joiner_binding=_binding("textured-joiner"),
    )

    assert receipt["schema"] == analyzer.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA
    assert receipt["decision"] == "FREEZE_PILOT_CONTRACT"
    assert receipt["calibration_contract"]["progress_tolerance_m"] == 0.01
    assert receipt["calibration_contract"]["path_length_tolerance_m"] == 0.01
    assert receipt["calibration_contract"]["tolerance_derivation"]["schema"] == (
        analyzer.TOLERANCE_DERIVATION_V2_SCHEMA
    )
    support = receipt["candidate_branch_support_analysis"]
    assert support["overall"]["identifiability"]["identifiable_state_count"] == 16
    assert support["overall"]["equivalence_unique_count_distributions"][
        "physical_trajectory"
    ]["minimum_unique_count"] == 1
    assert support["overall"]["equivalence_unique_count_distributions"][
        "stored_rgb_file"
    ]["collapsed_state_count"] == 16
    assert len(support["per_family"]) == 8
    assert len(support["per_history"]) == 3
    assert receipt["physics_validation"][
        "candidate_equivalence_measured_not_rejected"
    ] is True
    assert "nine_unique_executed_tapes_per_state" not in str(receipt)
    assert set(
        receipt["resource_measurements"]["low_information_strata"][
            "reason_counts"
        ]
    ) == {
        "camera_safety_unresolved",
        "low_rgb_texture",
        "near_wall_depth",
        "near_forward_geometry",
    }
    analyzer.validate_calibration_receipt_v1(
        receipt, verify_external_bindings=False
    )


def test_textured_v03_stops_when_no_state_is_identifiable() -> None:
    analyzer = _load_analyzer()
    receipt = analyzer.derive_calibration_receipt_v1(
        _textured_v03_collection(identifiable=False),
        collection_binding=_binding("collapsed-collection"),
        analyzer_binding=_binding("collapsed-analyzer"),
        checker_binding=_binding("collapsed-checker"),
        joiner_binding=_binding("collapsed-joiner"),
    )

    assert receipt["decision"] == (
        "STOP_INSUFFICIENT_JOINT_COUNTERFACTUAL_DISCRIMINATION_SUPPORT"
    )
    assert receipt["candidate_branch_support_analysis"]["overall"][
        "identifiability"
    ]["identifiable_state_count"] == 0
    analyzer.validate_calibration_receipt_v1(
        receipt, verify_external_bindings=False
    )


def test_textured_v03_freezes_at_one_nontrivial_state_per_family() -> None:
    analyzer = _load_analyzer()
    receipt = analyzer.derive_calibration_receipt_v1(
        _textured_v03_collection(
            identifiable=False,
            identifiable_state_indices=set(range(0, 16, 2)),
        ),
        collection_binding=_binding("half-coverage-collection"),
        analyzer_binding=_binding("half-coverage-analyzer"),
        checker_binding=_binding("half-coverage-checker"),
        joiner_binding=_binding("half-coverage-joiner"),
    )

    coverage = receipt["candidate_branch_support_analysis"][
        "calibrated_discrimination_query_coverage"
    ]
    assert receipt["decision"] == "FREEZE_PILOT_CONTRACT"
    assert coverage["overall"]["eligible_query_count"] == 72
    assert coverage["overall"]["total_query_count"] == 144
    assert coverage["overall"]["discrimination_query_coverage"] == 0.5
    assert coverage["all_families_passed"] is True
    assert all(
        item["eligible_query_count"] == 9
        for item in coverage["per_family"].values()
    )


def test_textured_v03_rejects_missing_family_despite_overall_half_coverage() -> None:
    analyzer = _load_analyzer()
    # Family 0 has no nontrivial state, family 1 has both, and every other
    # family has one.  Overall coverage is still exactly 8/16.
    nontrivial = {2, 3, 4, 6, 8, 10, 12, 14}
    receipt = analyzer.derive_calibration_receipt_v1(
        _textured_v03_collection(
            identifiable=False,
            identifiable_state_indices=nontrivial,
        ),
        collection_binding=_binding("unbalanced-coverage-collection"),
        analyzer_binding=_binding("unbalanced-coverage-analyzer"),
        checker_binding=_binding("unbalanced-coverage-checker"),
        joiner_binding=_binding("unbalanced-coverage-joiner"),
    )

    coverage = receipt["candidate_branch_support_analysis"][
        "calibrated_discrimination_query_coverage"
    ]
    first_family = analyzer.producer_contract.FAMILIES[0]
    assert coverage["overall"]["eligible_query_count"] == 72
    assert coverage["overall"]["passed"] is True
    assert coverage["per_family"][first_family]["passed"] is False
    assert coverage["all_families_passed"] is False
    assert coverage["passed"] is False
    assert receipt["decision"] == (
        "STOP_INSUFFICIENT_JOINT_COUNTERFACTUAL_DISCRIMINATION_SUPPORT"
    )


def test_textured_v03_aggregate_nontriviality_cannot_substitute_for_joint_support() -> None:
    analyzer = _load_analyzer()
    collection = _textured_v03_collection(identifiable=True)
    for state_index, state in enumerate(collection["states"]):
        branches = state["branches"]
        for action_id, branch in enumerate(branches[:9]):
            branch["physical_target_progress_m"] = 0.0 if action_id % 2 == 0 else 0.1
            branch["physical_path_length_m"] = 1.0
        repeated_action_id = branches[-1]["action_id"]
        branches[-1] = copy.deepcopy(branches[repeated_action_id])
        audit = state["document"]["candidate_response_audit"]
        audit["executed_tape"] = _partition_from_action_groups(
            [[0], list(range(1, 9))], salt=10_000 + state_index * 10
        )
        audit["stored_rgb_pixels"] = _partition_from_action_groups(
            [[1], [0, *range(2, 9)]], salt=20_000 + state_index * 10
        )

    receipt = analyzer.derive_calibration_receipt_v1(
        collection,
        collection_binding=_binding("misaligned-aggregate-collection"),
        analyzer_binding=_binding("misaligned-aggregate-analyzer"),
        checker_binding=_binding("misaligned-aggregate-checker"),
        joiner_binding=_binding("misaligned-aggregate-joiner"),
    )
    support = receipt["candidate_branch_support_analysis"]
    coverage = support["calibrated_discrimination_query_coverage"]

    assert support["overall"]["identifiability"]["identifiable_state_count"] == 16
    assert all(row["identifiable"] is True for row in support["state_measurements"])
    assert all(
        row["eligible_action_ids"] == []
        and row["eligible_action_count"] == 0
        for row in support["state_measurements"]
    )
    assert coverage["overall"]["eligible_query_count"] == 0
    assert coverage["overall"]["discrimination_query_coverage"] == 0.0
    assert coverage["definition"][
        "aggregate_partition_nontriviality_is_diagnostic_only"
    ] is True
    assert receipt["decision"] == (
        "STOP_INSUFFICIENT_JOINT_COUNTERFACTUAL_DISCRIMINATION_SUPPORT"
    )
    analyzer.validate_calibration_receipt_v1(
        receipt, verify_external_bindings=False
    )


def test_textured_v03_reports_class_diversity_without_calling_actions_separable() -> None:
    analyzer = _load_analyzer()
    collection = _textured_v03_collection(identifiable=True)
    state = collection["states"][0]
    for action_id, branch in enumerate(state["branches"][:9]):
        class_value = 0.0 if action_id < 4 else 0.1
        branch["physical_target_progress_m"] = class_value
        branch["physical_path_length_m"] = 1.0
    state["branches"][-1] = copy.deepcopy(state["branches"][0])

    receipt = analyzer.derive_calibration_receipt_v1(
        collection,
        collection_binding=_binding("two-class-collection"),
        analyzer_binding=_binding("two-class-analyzer"),
        checker_binding=_binding("two-class-checker"),
        joiner_binding=_binding("two-class-joiner"),
    )
    coverage = receipt["candidate_branch_support_analysis"][
        "calibrated_discrimination_query_coverage"
    ]

    assert coverage["overall"]["discrimination_query_coverage"] == 1.0
    assert coverage["overall"]["physical_outcome_class_count"] == 137
    assert coverage["overall"]["maximum_physical_outcome_class_count"] == 144
    assert coverage["overall"]["physical_outcome_class_coverage"] == 137 / 144
    assert coverage["definition"][
        "physical_outcome_class_coverage_is_diagnostic_only"
    ] is True
    assert "separable_action" not in str(coverage)


def test_textured_v03_validator_recomputes_discrimination_coverage() -> None:
    analyzer = _load_analyzer()
    receipt = analyzer.derive_calibration_receipt_v1(
        _textured_v03_collection(identifiable=True),
        collection_binding=_binding("coverage-recompute-collection"),
        analyzer_binding=_binding("coverage-recompute-analyzer"),
        checker_binding=_binding("coverage-recompute-checker"),
        joiner_binding=_binding("coverage-recompute-joiner"),
    )
    forged = copy.deepcopy(receipt)
    forged["candidate_branch_support_analysis"][
        "calibrated_discrimination_query_coverage"
    ]["overall"]["physical_outcome_class_count"] -= 1
    with pytest.raises(
        analyzer.CalibrationAnalysisError,
        match="coverage",
    ):
        analyzer.validate_calibration_receipt_v1(
            forged, verify_external_bindings=False
        )

    forged_state = copy.deepcopy(receipt)
    state_row = forged_state["candidate_branch_support_analysis"][
        "state_measurements"
    ][0]
    state_row["eligible_action_ids"] = []
    state_row["eligible_action_count"] = 0
    with pytest.raises(
        analyzer.CalibrationAnalysisError,
        match="joint eligibility",
    ):
        analyzer.validate_calibration_receipt_v1(
            forged_state, verify_external_bindings=False
        )

    forged_partition_count = copy.deepcopy(receipt)
    signatures = forged_partition_count["candidate_branch_support_analysis"][
        "state_measurements"
    ][0]["joint_contrast_signatures_by_action"]
    signatures[1]["executed_tape_class_sha256"] = signatures[0][
        "executed_tape_class_sha256"
    ]
    with pytest.raises(
        analyzer.CalibrationAnalysisError,
        match="signature partition counts",
    ):
        analyzer.validate_calibration_receipt_v1(
            forged_partition_count, verify_external_bindings=False
        )


def test_textured_v03_one_centimeter_bins_have_explicit_boundary_caveat() -> None:
    analyzer = _load_analyzer()
    assert analyzer._quantize(0.004, 0.01) == 0
    assert analyzer._quantize(0.006, 0.01) == 1
    assert 0.006 - 0.004 < 0.01

    receipt = analyzer.derive_calibration_receipt_v1(
        _textured_v03_collection(identifiable=True),
        collection_binding=_binding("boundary-collection"),
        analyzer_binding=_binding("boundary-analyzer"),
        checker_binding=_binding("boundary-checker"),
        joiner_binding=_binding("boundary-joiner"),
    )
    derivation = receipt["calibration_contract"]["tolerance_derivation"]
    assert derivation["schema"] == analyzer.TOLERANCE_DERIVATION_V2_SCHEMA
    assert derivation["outcome_equivalence_quantization_caveat"] == (
        "1cm_rounding_bins_have_boundary_artifacts_and_are_not_"
        "pairwise_distance_le_1cm_equivalence"
    )


def test_textured_v03_dispatch_fails_closed_on_partial_v2_identity() -> None:
    analyzer = _load_analyzer()
    collection = _collection()
    collection["purpose"] = analyzer.TEXTURED_V03_CALIBRATION_PURPOSE
    with pytest.raises(
        analyzer.CalibrationAnalysisError,
        match="only partially present",
    ):
        analyzer.derive_calibration_receipt_v1(
            collection,
            collection_binding=_binding("partial-collection"),
            analyzer_binding=_binding("partial-analyzer"),
            checker_binding=_binding("partial-checker"),
            joiner_binding=_binding("partial-joiner"),
        )


def test_bound_calibration_receipt_uses_one_fd_without_path_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = _load_analyzer()
    collection_path = tmp_path / "collection.json"
    analyzer_path = tmp_path / "analyzer.py"
    checker_path = tmp_path / "checker.py"
    joiner_path = tmp_path / "joiner.py"
    for path, raw in (
        (collection_path, b"collection\n"),
        (analyzer_path, b"analyzer\n"),
        (checker_path, b"checker\n"),
        (joiner_path, b"joiner\n"),
    ):
        path.write_bytes(raw)
    receipt = analyzer.derive_calibration_receipt_v1(
        _collection(),
        collection_binding=analyzer._binding(collection_path),
        analyzer_binding=analyzer._binding(analyzer_path),
        checker_binding=analyzer._binding(checker_path),
        joiner_binding=analyzer._binding(joiner_path),
    )
    receipt_path = tmp_path / "receipt.json"
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt_path.write_bytes(raw)

    original_binding = analyzer._binding
    receipt_binding_calls = 0

    def reject_receipt_rebinding(path: Path) -> dict[str, object]:
        nonlocal receipt_binding_calls
        if Path(path).resolve() == receipt_path.resolve():
            receipt_binding_calls += 1
            raise AssertionError("receipt pathname was rebound")
        return original_binding(path)

    monkeypatch.setattr(analyzer, "_binding", reject_receipt_rebinding)
    loaded, actual, loaded_raw = analyzer.load_bound_calibration_receipt_v1(
        receipt_path,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_byte_count=len(raw),
    )
    assert loaded == receipt
    assert loaded_raw == raw
    assert actual == {
        "path": str(receipt_path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    assert receipt_binding_calls == 0


def test_analyze_calibration_carries_checker_consumed_binding_across_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = _load_analyzer()
    collection_path = tmp_path / "collection.json"
    collection_a = b'{"a":1}'
    collection_b = b'{"b":2}'
    assert len(collection_a) == len(collection_b)
    collection_path.write_bytes(collection_a)
    expected_sha = hashlib.sha256(collection_a).hexdigest()
    captured: dict[str, object] = {}

    def checked_then_swapped(
        path: Path,
        *,
        expected_file_sha256: str,
        expected_byte_count: int,
        verify_textured_pixels: bool,
    ) -> dict[str, object]:
        assert path == collection_path
        assert expected_file_sha256 == expected_sha
        assert expected_byte_count == len(collection_a)
        assert verify_textured_pixels is True
        collection_path.write_bytes(collection_b)
        return {"validated_from": "A"}

    def capture_derivation(
        collection: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        captured["collection"] = collection
        captured["binding"] = kwargs["collection_binding"]
        return {"schema": "synthetic-calibration-receipt"}

    monkeypatch.setattr(
        analyzer.checker,
        "load_bound_collection_receipts",
        checked_then_swapped,
    )
    monkeypatch.setattr(
        analyzer, "derive_calibration_receipt_v1", capture_derivation
    )
    output_path = tmp_path / "calibration-receipt.json"
    analyzer.analyze_calibration(
        collection_path=collection_path,
        expected_collection_sha256=expected_sha,
        expected_collection_byte_count=len(collection_a),
        output_path=output_path,
    )

    assert captured["collection"] == {"validated_from": "A"}
    assert captured["binding"] == {
        "path": str(collection_path),
        "file_sha256": expected_sha,
        "byte_count": len(collection_a),
    }
    assert hashlib.sha256(collection_path.read_bytes()).hexdigest() != expected_sha
