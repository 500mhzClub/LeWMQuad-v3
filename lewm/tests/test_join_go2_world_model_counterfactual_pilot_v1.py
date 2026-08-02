from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as producer_contract
from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as dataset_consumer
from lewm.tests.test_go2_world_model_counterfactual_pilot_v1 import _smoke_plan
from scripts import analyze_go2_world_model_counterfactual_calibration_v1 as calibration
from scripts import build_go2_world_model_bounded_branch_experiment_plan_v1 as bounded_plan_builder
from scripts import build_go2_world_model_counterfactual_calibration_authority_v1 as calibration_authority_builder
from scripts import build_go2_world_model_counterfactual_calibration_plan_v1 as calibration_plan_builder


ROOT = Path(__file__).resolve().parents[2]
JOINER_PATH = ROOT / "scripts/join_go2_world_model_counterfactual_pilot_v1.py"


def _load_joiner():
    spec = importlib.util.spec_from_file_location(
        "counterfactual_pilot_joiner_v1", JOINER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relative_binding_uses_one_nofollow_leaf_read(tmp_path: Path) -> None:
    joiner = _load_joiner()
    root = tmp_path / "pilot"
    nested = root / "joined"
    nested.mkdir(parents=True)
    leaf = nested / "manifest.json"
    leaf.write_bytes(b"exact-joined-bytes")

    binding = joiner._relative_binding(leaf, root=root)  # noqa: SLF001
    assert binding == {
        "path": "joined/manifest.json",
        "file_sha256": hashlib.sha256(b"exact-joined-bytes").hexdigest(),
        "byte_count": len(b"exact-joined-bytes"),
    }

    symlink_leaf = nested / "symlink.json"
    symlink_leaf.symlink_to(leaf)
    with pytest.raises(joiner.PilotJoinError, match="safely bind"):
        joiner._relative_binding(symlink_leaf, root=root)  # noqa: SLF001

    symlink_directory = root / "symlink-dir"
    symlink_directory.symlink_to(nested, target_is_directory=True)
    with pytest.raises(joiner.PilotJoinError, match="safely bind"):
        joiner._relative_binding(  # noqa: SLF001
            symlink_directory / "manifest.json", root=root
        )


def test_joiner_collection_loader_requires_decoded_pixel_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joiner = _load_joiner()
    observed: dict[str, object] = {}

    def load(path: Path, **kwargs):
        observed.update({"path": path, **kwargs})
        return {"verified": True}

    monkeypatch.setattr(joiner.checker, "load_bound_collection_receipts", load)
    result = joiner._load_pixel_verified_collection(  # noqa: SLF001
        Path("/synthetic/collection.json"),
        expected_file_sha256="a" * 64,
        expected_byte_count=7,
    )
    assert result == {"verified": True}
    assert observed == {
        "path": Path("/synthetic/collection.json"),
        "expected_file_sha256": "a" * 64,
        "expected_byte_count": 7,
        "verify_textured_pixels": True,
    }


def _frame(identity: str) -> dict[str, object]:
    return {
        "artifact_id": identity,
        "frame_identity": identity,
        "path": f"frames/{identity.replace(':', '_')}.png",
        "file_sha256": f"{len(identity) % 16:x}" * 64,
        "byte_count": 10,
        "width": 224,
        "height": 224,
        "mode": "RGB",
        "format": "PNG",
        "camera_valid": True,
        "low_information": False,
        "low_info_reasons": [],
    }


def _collection() -> dict[str, object]:
    frame_receipts = {}
    states = []
    action_catalog = [
        {
            "action_id": action,
            "name": f"action-{action}",
            "requested_block": [[float(action), 0.0, 0.0]] * 5,
        }
        for action in range(9)
    ]
    for group_index, role in enumerate(("train", "eval")):
        state_id = f"{role}:state"
        context_ids = [f"{state_id}:context:{index}" for index in range(3)]
        for identity in context_ids:
            frame_receipts[identity] = _frame(identity)
        branches = []
        for action in range(9):
            identity = f"{state_id}:candidate:{action}"
            receipt = _frame(identity)
            frame_receipts[identity] = receipt
            branches.append({
                "lane_index": group_index * 9 + action,
                "lane_offset": action,
                "kind": "candidate",
                "action_id": action,
                "action_name": f"action-{action}",
                "duplicates_candidate_action_id": None,
                "requested_block": [[float(action), 0.0, 0.0]] * 5,
                # Deliberately differs from the requested block.  The join must
                # preserve it as an outcome without using it as action identity.
                "executed_block": [[float(action) + 0.25, 0.0, 0.0]] * 5,
                "executed_block_sha256": f"{action:x}" * 64,
                "clipped": action == 0,
                "trajectory_policy_step_samples": [{"step": 0}],
                "endpoint_state": {"base_pos_world": [float(action), 0.0, 0.3]},
                "physical_fell": False,
                "physical_tipped": False,
                "physical_path_length_m": float(action),
                "physical_target_progress_m": float(action),
                "render_frame_identity": identity,
                "frame_receipt": receipt,
            })
        states.append({
            "state": {
                "state_id": state_id,
                "role": role,
                "family": "large_enclosed_maze",
                "scene_id": f"{role}-scene",
                "group_index": group_index,
                "state_index_in_scene": 0,
            },
            "context": {
                "rgb_artifact_ids": context_ids,
                "frame_identities": context_ids,
                "history_action_ids": [0, 1],
                "history_executed_blocks": [[[0.0, 0.0, 0.0]] * 5] * 2,
                "executed_block_sha256s": ["a" * 64, "b" * 64],
                "endpoint_command_ticks": [0, 5, 10],
                "prebranch_state_sha256": "c" * 64,
            },
            "relative_target_xy_body_m": [1.0, 0.0],
            "document": {
                "schema": producer_contract.STATE_RECEIPT_SCHEMA,
                "synchronization_audit": {"passed": True},
            },
            "branches": branches,
        })
    return {
        "purpose": "bounded_wm_a_pilot",
        "states": states,
        "frame_receipts": frame_receipts,
        "plan": {"document": {
            "action_catalog": action_catalog,
            "render_contract": dict(producer_contract.RENDER_CONTRACT),
        }},
    }


def _calibration() -> dict[str, object]:
    return {
        "schema": calibration.CALIBRATION_RECEIPT_SCHEMA,
        "decision": "FREEZE_PILOT_CONTRACT",
        "calibration_collection_receipt": {
            "path": "/synthetic/calibration-collection.json",
            "file_sha256": "c" * 64,
            "byte_count": 1,
        },
        "calibration_contract": {
            "excluded_scene_ids": ["calibration-scene"],
            "progress_tolerance_m": 1e-6,
            "path_length_tolerance_m": 1e-6,
        },
    }


def _parity_binding(name: str, digit: str) -> dict[str, object]:
    return {
        "path": f"/synthetic/{name}.json",
        "file_sha256": digit * 64,
        "byte_count": 1,
    }


def _textured_collection(
    result_digit: str = "d",
    terminal_digit: str = "e",
    review_digit: str = "f",
) -> dict[str, object]:
    collection = _collection()
    collection["plan"]["document"]["render_contract"] = dict(
        producer_contract.TEXTURED_V03_RENDER_CONTRACT
    )
    collection["plan"]["document"].update({
        "visual_domain_parity_result_binding": _parity_binding(
            "parity-result", result_digit
        ),
        "visual_domain_parity_terminal_binding": _parity_binding(
            "parity-terminal", terminal_digit
        ),
        "visual_domain_parity_review_binding": _parity_binding(
            "parity-review", review_digit
        ),
    })
    for state in collection["states"]:
        state["document"]["schema"] = (
            producer_contract.TEXTURED_V03_STATE_RECEIPT_SCHEMA
        )
    return collection


def _textured_calibration(
    result_digit: str = "d",
    terminal_digit: str = "e",
    review_digit: str = "f",
) -> dict[str, object]:
    return {
        **_calibration(),
        "schema": calibration.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA,
        "visual_domain_parity_prerequisites": {
            "result_binding": _parity_binding("parity-result", result_digit),
            "terminal_binding": _parity_binding(
                "parity-terminal", terminal_digit
            ),
            "review_binding": _parity_binding("parity-review", review_digit),
        },
    }


def test_join_builds_receipt_rows_without_future_executed_tape_leakage() -> None:
    joiner = _load_joiner()
    rgb_manifest, rows, metadata = joiner.build_joined_documents_v1(
        _collection(), _calibration()
    )
    assert len(rgb_manifest["artifacts"]) == 24
    assert len(rows["train"]) == len(rows["eval"]) == 1
    branch = rows["train"][0]["branches"][0]
    assert branch["action_id"] == 0
    assert branch["requested_block"] == [[0.0, 0.0, 0.0]] * 5
    assert branch["executed_block"] == [[0.25, 0.0, 0.0]] * 5
    assert branch["declared_oracle_dense_rank"] == 8
    assert metadata["scene_ids"] == {
        "train": ["train-scene"],
        "eval": ["eval-scene"],
    }


def test_join_rejects_failed_calibration_before_emitting_rows() -> None:
    joiner = _load_joiner()
    failed = {**_calibration(), "decision": "STOP_SOURCE_REDESIGN"}
    with pytest.raises(joiner.PilotJoinError, match="did not freeze"):
        joiner.build_joined_documents_v1(_collection(), failed)


def test_join_rejects_legacy_collection_with_textured_calibration() -> None:
    joiner = _load_joiner()
    with pytest.raises(joiner.PilotJoinError, match="render profiles differ"):
        joiner.build_joined_documents_v1(
            _collection(),
            _textured_calibration(),
            calibration_visual_domain_parity_result_binding=_parity_binding(
                "parity-result", "d"
            ),
            calibration_visual_domain_parity_terminal_binding=_parity_binding(
                "parity-terminal", "e"
            ),
            calibration_visual_domain_parity_review_binding=_parity_binding(
                "parity-review", "f"
            ),
        )


def test_join_rejects_textured_collection_with_legacy_calibration() -> None:
    joiner = _load_joiner()
    with pytest.raises(joiner.PilotJoinError, match="render profiles differ"):
        joiner.build_joined_documents_v1(
            _textured_collection(), _calibration()
        )


def test_join_rejects_correct_textured_schema_with_wrong_parity_lineage() -> None:
    joiner = _load_joiner()
    with pytest.raises(joiner.PilotJoinError, match="parity lineage differs"):
        joiner.build_joined_documents_v1(
            _textured_collection(result_digit="d"),
            _textured_calibration(),
            calibration_visual_domain_parity_result_binding=_parity_binding(
                "parity-result-wrong", "a"
            ),
            calibration_visual_domain_parity_terminal_binding=_parity_binding(
                "parity-terminal", "e"
            ),
            calibration_visual_domain_parity_review_binding=_parity_binding(
                "parity-review", "f"
            ),
        )


def test_join_rejects_correct_textured_result_with_wrong_terminal_lineage() -> None:
    joiner = _load_joiner()
    with pytest.raises(joiner.PilotJoinError, match="parity lineage differs"):
        joiner.build_joined_documents_v1(
            _textured_collection(),
            _textured_calibration(),
            calibration_visual_domain_parity_result_binding=_parity_binding(
                "parity-result", "d"
            ),
            calibration_visual_domain_parity_terminal_binding=_parity_binding(
                "parity-terminal-wrong", "a"
            ),
            calibration_visual_domain_parity_review_binding=_parity_binding(
                "parity-review", "f"
            ),
        )


def test_join_rejects_correct_textured_result_terminal_with_wrong_review() -> None:
    joiner = _load_joiner()
    with pytest.raises(joiner.PilotJoinError, match="parity lineage differs"):
        joiner.build_joined_documents_v1(
            _textured_collection(),
            _textured_calibration(),
            calibration_visual_domain_parity_result_binding=_parity_binding(
                "parity-result", "d"
            ),
            calibration_visual_domain_parity_terminal_binding=_parity_binding(
                "parity-terminal", "e"
            ),
            calibration_visual_domain_parity_review_binding=_parity_binding(
                "parity-review-wrong", "a"
            ),
        )


@pytest.mark.parametrize(
    ("result_digit", "terminal_digit", "review_digit"),
    (("a", "e", "f"), ("d", "a", "f"), ("d", "e", "a")),
)
def test_join_rejects_calibration_receipt_prerequisite_swap(
    result_digit: str,
    terminal_digit: str,
    review_digit: str,
) -> None:
    joiner = _load_joiner()
    with pytest.raises(joiner.PilotJoinError, match="parity lineage differs"):
        joiner.build_joined_documents_v1(
            _textured_collection(),
            _textured_calibration(
                result_digit=result_digit,
                terminal_digit=terminal_digit,
                review_digit=review_digit,
            ),
            calibration_visual_domain_parity_result_binding=_parity_binding(
                "parity-result", "d"
            ),
            calibration_visual_domain_parity_terminal_binding=_parity_binding(
                "parity-terminal", "e"
            ),
            calibration_visual_domain_parity_review_binding=_parity_binding(
                "parity-review", "f"
            ),
        )


def test_join_accepts_textured_profiles_only_with_exact_parity_lineage() -> None:
    joiner = _load_joiner()
    result = _parity_binding("parity-result", "d")
    terminal = _parity_binding("parity-terminal", "e")
    review = _parity_binding("parity-review", "f")
    _rgb, _rows, metadata = joiner.build_joined_documents_v1(
        _textured_collection(),
        _textured_calibration(),
        calibration_visual_domain_parity_result_binding=result,
        calibration_visual_domain_parity_terminal_binding=terminal,
        calibration_visual_domain_parity_review_binding=review,
    )
    assert metadata["render_profile"] == joiner.TEXTURED_V03_RENDER_PROFILE
    assert metadata["visual_domain_parity_result_binding"] == result
    assert metadata["visual_domain_parity_terminal_binding"] == terminal
    assert metadata["visual_domain_parity_review_binding"] == review


def test_point7_exact_parity_triple_survives_every_consuming_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the one identity across the independently tested stage APIs."""

    def write_bound(name: str, document: object) -> dict[str, object]:
        path = tmp_path / name
        path.write_text(
            json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
        )
        return producer_contract.file_binding(path)

    result_binding = write_bound(
        "parity-result.json",
        {
            "schema": producer_contract.TEXTURED_V03_PARITY_RESULT_SCHEMA,
            "status": producer_contract.TEXTURED_V03_PARITY_PASS_STATUS,
        },
    )
    terminal_binding = write_bound(
        "parity-terminal.json",
        {
            "schema": producer_contract.TEXTURED_V03_PARITY_TERMINAL_SCHEMA,
            "status": (
                producer_contract.TEXTURED_V03_PARITY_TERMINAL_SUCCESS_STATUS
            ),
            "root_creation_consumes_attempt": True,
            "reservation_records_consumed_attempt": True,
        },
    )
    review_binding = write_bound(
        "parity-review.json",
        {
            "schema": producer_contract.TEXTURED_V03_PARITY_REVIEW_SCHEMA,
            "status": producer_contract.TEXTURED_V03_PARITY_REVIEW_PASS_STATUS,
            "result_binding": result_binding,
            "terminal_binding": terminal_binding,
        },
    )
    parity = {
        "result_binding": result_binding,
        "terminal_binding": terminal_binding,
        "review_binding": review_binding,
    }

    # Deep terminal/result/review reopening has its own full-tree tests. This
    # integration fixture pins that validated output and checks that no later
    # stage drops, aliases, or replaces one member of the triple.
    def validated_parity(**bindings):
        assert bindings == {
            "result_binding": result_binding,
            "terminal_binding": terminal_binding,
            "review_binding": review_binding,
        }
        return dict(parity)

    monkeypatch.setattr(
        producer_contract,
        "validate_textured_v03_parity_prerequisites",
        validated_parity,
    )

    smoke = _smoke_plan(tmp_path / "runtime")
    scenes = []
    for family_index, family in enumerate(producer_contract.FAMILIES):
        scene_root = tmp_path / "scenes" / family
        scene_root.mkdir(parents=True)
        manifest = scene_root / "manifest.json"
        genesis = scene_root / "genesis_scene.json"
        target = [1.0, float(family_index)]
        manifest.write_text(
            json.dumps({
                "scene_id": f"calibration-{family_index}",
                "family": family,
                "landmarks": [{
                    "object_id": "target-000",
                    "center_xyz_m": [*target, 0.5],
                }],
            }),
            encoding="utf-8",
        )
        genesis.write_text("{}\n", encoding="utf-8")
        scenes.append({
            "family": family,
            "scene_id": f"calibration-{family_index}",
            "scene_manifest_binding": producer_contract.file_binding(manifest),
            "scene_genesis_binding": producer_contract.file_binding(genesis),
            "states": [
                {
                    "state_id": f"calibration-{family_index}:{state_index}",
                    "history_action_ids": [state_index, state_index + 1],
                    "target_xy_m": target,
                }
                for state_index in range(2)
            ],
        })
    v3_plan = calibration_plan_builder.build_calibration_plan_v1(
        attempt_id=calibration_authority_builder.collector.CALIBRATION_V3_ATTEMPT_ID,
        output_root=Path(
            calibration_authority_builder.collector.CALIBRATION_V3_ROOT
        ),
        scene_panel={
            "schema": calibration_plan_builder.SCENE_PANEL_SCHEMA,
            "scenes": scenes,
        },
        runtime_contract={
            "schema": calibration_plan_builder.RUNTIME_CONTRACT_SCHEMA,
            "runtime_bindings": smoke["runtime_bindings"],
            "execution_contract": smoke["execution_contract"],
        },
        textured_v03=True,
        visual_domain_parity_result_binding=result_binding,
        visual_domain_parity_terminal_binding=terminal_binding,
        visual_domain_parity_review_binding=review_binding,
    )
    assert {
        "result_binding": v3_plan["visual_domain_parity_result_binding"],
        "terminal_binding": v3_plan["visual_domain_parity_terminal_binding"],
        "review_binding": v3_plan["visual_domain_parity_review_binding"],
    } == parity

    plan_binding = write_bound("calibration-v3-plan.json", v3_plan)
    predecessor_binding = producer_contract.file_binding(
        ROOT
        / calibration_authority_builder.collector.CALIBRATION_V2_FAILURE_RELATIVE
    )
    source_bindings = [
        {
            "name": name,
            "binding": producer_contract.file_binding(ROOT / relative),
        }
        for name, relative in calibration_authority_builder
        .canonical_runtime_source_paths_v1(textured_v03=True)
        .items()
    ]
    review = {
        "schema": producer_contract.SOURCE_REVIEW_SCHEMA,
        "status": "PASS_SOURCE_ONLY_NOT_AUTHORITY",
        "authority_granted_by_this_document": False,
        "reviewed_source_commit": "c" * 40,
        "reviewed_source_bindings": source_bindings,
        "remaining_findings": [],
        "reviewer": {
            "identity": "independent-point7-integration-reviewer",
            "independence_basis": "synthetic cross-stage identity audit",
        },
        "reviewed_at": "2026-08-02T12:00:00+00:00",
        "review_method": ["reopened every synthetic stage identity"],
        "test_evidence": ["this exact integration test"],
        "accepted_limitations": ["synthetic lineage test only"],
    }
    review_source_binding = write_bound("calibration-source-review.json", review)
    v3_authority = calibration_authority_builder.build_authority_v1(
        plan=v3_plan,
        plan_binding=plan_binding,
        review=review,
        review_binding=review_source_binding,
        predecessor_failure_binding=predecessor_binding,
        authorizer_identity="explicit-point7-integration-authorizer",
        authorizer_basis="synthetic lineage test",
        issued_at="2026-08-02T12:01:00+00:00",
        terminal_reviewer="independent-terminal-reviewer",
        wall_seconds=3600.0,
        platform_basis="synthetic resolved-gate fixture",
    )
    assert v3_authority["plan_binding"] == plan_binding
    assert v3_authority["attempt"]["root_creation_consumes_attempt"] is True

    calibration_collection_binding = _parity_binding(
        "calibration-collection", "c"
    )
    v3_receipt = {
        **_textured_calibration(),
        "calibration_collection_receipt": calibration_collection_binding,
        "visual_domain_parity_prerequisites": dict(parity),
    }
    calibration_gate = {
        "visual_domain_parity_result_binding": result_binding,
        "visual_domain_parity_terminal_binding": terminal_binding,
        "visual_domain_parity_review_binding": review_binding,
    }
    bounded_plan_builder._require_calibration_parity_identity(  # noqa: SLF001
        calibration_gate,
        {
            "result_binding": result_binding,
            "terminal_binding": terminal_binding,
            "review_binding": review_binding,
        },
    )
    bounded_plan = {
        "visual_domain_parity_result_binding": calibration_gate[
            "visual_domain_parity_result_binding"
        ],
        "visual_domain_parity_terminal_binding": calibration_gate[
            "visual_domain_parity_terminal_binding"
        ],
        "visual_domain_parity_review_binding": calibration_gate[
            "visual_domain_parity_review_binding"
        ],
    }

    collection = _textured_collection()
    collection["plan"]["document"].update(bounded_plan)
    joiner = _load_joiner()
    _rgb, _rows, joined = joiner.build_joined_documents_v1(
        collection,
        v3_receipt,
        calibration_visual_domain_parity_result_binding=result_binding,
        calibration_visual_domain_parity_terminal_binding=terminal_binding,
        calibration_visual_domain_parity_review_binding=review_binding,
    )
    consumed = dataset_consumer._validate_joined_render_lineage_v1(  # noqa: SLF001
        collection_lineage={
            "render_profile": joined["render_profile"],
            "visual_domain_parity_result_binding": joined[
                "visual_domain_parity_result_binding"
            ],
            "visual_domain_parity_terminal_binding": joined[
                "visual_domain_parity_terminal_binding"
            ],
            "visual_domain_parity_review_binding": joined[
                "visual_domain_parity_review_binding"
            ],
        },
        calibration_collection_lineage={
            "calibration_collection_receipt_binding": (
                calibration_collection_binding
            ),
            "render_profile": joined["render_profile"],
            "visual_domain_parity_result_binding": result_binding,
            "visual_domain_parity_terminal_binding": terminal_binding,
            "visual_domain_parity_review_binding": review_binding,
        },
        calibration_receipt={"document": v3_receipt},
        manifest_render_profile=joined["render_profile"],
        manifest_visual_domain_parity_result_binding=joined[
            "visual_domain_parity_result_binding"
        ],
        manifest_visual_domain_parity_terminal_binding=joined[
            "visual_domain_parity_terminal_binding"
        ],
        manifest_visual_domain_parity_review_binding=joined[
            "visual_domain_parity_review_binding"
        ],
        manifest_calibration_collection_receipt_binding=joined[
            "calibration_collection_receipt_binding"
        ],
        synthetic_test_mode=True,
    )
    assert dict(consumed["visual_domain_parity_result_binding"]) == result_binding
    assert dict(consumed["visual_domain_parity_terminal_binding"]) == terminal_binding
    assert dict(consumed["visual_domain_parity_review_binding"]) == review_binding
