from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v3_contract as C


def test_v3_identity_inventory_and_inherited_science_are_exact() -> None:
    contract = C.validate_contract(C.build_contract())
    assert C.EXPERIMENT_ID == "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V3"
    assert C.SOURCE_PARENT_COMMIT == "10117870fe00bbfd8709932cab7b6e9df3be9bfc"
    assert C.SOURCE_BASELINE_COMMIT == "6250d282f1d1edeee411f8ef11407cc7c7a445d3"
    assert C.CONTRACT_FREEZE_COMMIT_SUBJECT == (
        "Freeze semantic physical graph edge handoff qualification V3"
    )
    assert C.RESULT_COMMIT_SUBJECT == (
        "Evaluate semantic physical graph edge handoff qualification V3"
    )
    assert len(C.SUCCESS_OUTPUT_LEAVES) == len(set(C.SUCCESS_OUTPUT_LEAVES)) == 28
    assert len(C.REPRODUCTION_MISMATCH_LEAVES) == 4
    assert contract["scientific_invariance_authority"][
        "all_metric_formulas_gates_classes_precedence_and_next_decisions_unchanged"
    ] is True
    assert C.scientific_invariance_projection(contract) == C.V2_SCIENTIFIC_PROJECTION
    assert C.build_candidate_specs() == C.V2.build_candidate_specs()
    assert tuple(C.SOURCE_DEPENDENCY_PATHS) == tuple(C.V2.SOURCE_DEPENDENCY_PATHS)


def test_snapshot_serializer_and_behavioural_authorities_are_exact() -> None:
    serializer = C.SEMANTIC_SERIALIZER_AUTHORITY
    assert serializer["reference_policy"] == (
        "CANONICAL_FIRST_TRAVERSAL_REFERENCE_AND_STORAGE_GRAPH_V1"
    )
    assert serializer["torch_rule"].find("byte storage offset") >= 0
    assert len(serializer["nonfinite_sentinel_authority"]["allowed_numpy_arrays"]) == 2
    assert len(serializer["regression_requirements"]) == 14
    probe = C.BEHAVIOURAL_PROBE_AUTHORITY
    assert probe["trace_count"] == 544
    assert C.BEHAVIOURAL_PROBE_TOTAL_SAMPLES == 408_000
    assert probe["snapshot_behavioural_digest_v1_designated_trial"] == 0
    assert probe["command"] == [0.2, 0.0, 0.0]
    assert probe["requested_command_trace_value"] == [
        0.20000000298023224,
        0.0,
        0.0,
    ]
    assert probe["requested_command_trace_value"] != probe["command"]
    assert probe["fresh_simulator_instance_per_restoration_trial"] is True
    assert probe["probe_session_topology"]["production_fixture_sessions"] == 7
    assert probe["probe_session_topology"][
        "same_session_second_restore_forbidden"
    ] is True
    assert probe["controller_policy_sampling"] == {
        "command_ticks": 15,
        "policy_acts_per_command_tick": 5,
        "physics_samples_per_policy_act": 10,
        "policy_acts_per_trial": 75,
        "physics_samples_per_trial": 750,
        "controller_observation": probe["controller_policy_sampling"][
            "controller_observation"
        ],
        "policy_output": probe["controller_policy_sampling"]["policy_output"],
        "excluded_policy_output_interpretations": probe[
            "controller_policy_sampling"
        ]["excluded_policy_output_interpretations"],
    }
    assert C.BEHAVIOURAL_PROBE_NPZ_AUTHORITY["policy_output"] == {
        "descr": "<f8",
        "digest_dtype": "float64",
        "shape": [408_000, 12],
        "hash_mode": "offset_slices",
        "offsets_member": "trace_offsets",
    }
    assert C.FIRST_EIGHT_REPRODUCTION_AUTHORITY["first_eight_material_probe"] == {
        "path": "reproduction/first_eight_behavioural_probes.npz",
        "trace_count": 48,
        "physics_samples": 36_000,
        "immutable_binding_required_on_success_or_mismatch": True,
        "official_final_npz_not_written_before_pool_8": True,
    }


def test_equivalence_custody_result_and_source_closure_authorities_are_closed() -> None:
    assert C.SNAPSHOT_EQUIVALENCE_INDEX_FIELDS >= {
        "behavioural_probe_npz_binding",
        "first_eight_material_probe_binding",
        "first_eight_prefix_exact",
    }
    assert C.V3_ADDITIONAL_RECOMPUTE_EVIDENCE_KEYS == {
        "external_historical_custody_receipt",
        "v1_v2_custody_and_nonreuse",
        "scientific_invariance_receipt",
        "v1_v2_v3_first_eight_reproduction",
        "snapshot_equivalence_index",
        "snapshot_behavioural_probe_arrays",
        "first_eight_behavioural_probe_arrays",
    }
    assert C.RESULT_PUBLICATION_AUTHORITY["report_section_order"] == list(
        C.V3_RESULT_REPORT_SECTION_ORDER
    )
    assert C.RESULT_PUBLICATION_AUTHORITY["scientific_input_binding_count"] == 25
    assert C.RESULT_PUBLICATION_AUTHORITY["build_api"] == (
        "build_result_publication_projection"
    )
    assert C.RESULT_PUBLICATION_AUTHORITY["validate_api"] == (
        "validate_result_publication_projection"
    )
    assert C.RESULT_PUBLICATION_AUTHORITY["report_bytes_api"] == (
        "build_result_report_bytes"
    )
    assert len(C.SOURCE_CLOSURE_PATHS) == len(set(C.SOURCE_CLOSURE_PATHS)) == 81
    assert tuple(C.SOURCE_CLOSURE_PATHS) == (
        C.TRACKED_SOURCE_PATHS[7:] + C.V3_WRAPPER_DEPENDENCY_PATHS
    )
    for relative in C.SOURCE_CLOSURE_PATHS:
        assert (Path(__file__).resolve().parents[2] / relative).is_file(), relative


def test_runtime_contract_requires_exact_external_custody_binding() -> None:
    binding = C.HISTORICAL_CUSTODY_RECEIPT_BINDING
    runtime = C.build_runtime_contract("b" * 40, binding)
    assert C.validate_runtime_contract(
        runtime,
        source_freeze_commit="b" * 40,
        historical_custody_receipt_binding=binding,
    ) == runtime
    bad = dict(binding, sha256="a" * 64)
    with pytest.raises(C.PhysicalGraphEdgeHandoffV3ContractError):
        C.build_runtime_contract("b" * 40, bad)
    assert C.HISTORICAL_CUSTODY_RECEIPT_PATH_AUTHORITY["exact_binding"] == binding


def test_runtime_genesis_urdf_binding_uses_exact_ordinary_resolved_path() -> None:
    runtime = C.validate_runtime_contract(
        C.build_runtime_contract("b" * 40, C.HISTORICAL_CUSTODY_RECEIPT_BINDING)
    )
    rows = runtime["external_artifact_bindings"]
    urdf = [row for row in rows if row["role"] == "genesis_go2_urdf"]
    assert urdf == [
        {
            "role": "genesis_go2_urdf",
            "path": C.GENESIS_GO2_URDF_RESOLVED_ORDINARY_PATH,
            "bytes": 24170,
            "sha256": (
                "4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4"
            ),
            "kind": "frozen_physical_robot_asset",
        }
    ]
    path = Path(urdf[0]["path"])
    inherited = Path(C.GENESIS_GO2_URDF_INHERITED_LEXICAL_PATH)
    assert inherited.resolve(strict=True) == path
    assert path.resolve(strict=True) == path
    assert path.is_file() and not path.is_symlink()
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor /= component
        assert not cursor.is_symlink(), cursor
    payload = path.read_bytes()
    assert len(payload) == urdf[0]["bytes"]
    assert hashlib.sha256(payload).hexdigest() == urdf[0]["sha256"]
    authority = C.EXTERNAL_ARTIFACT_RUNTIME_PATH_CORRECTION_AUTHORITY
    assert authority["same_immutable_artifact_bytes_and_sha256"] is True
    assert authority["strict_no_symlink_external_binding_validator_unchanged"] is True
    assert C.build_contract()[
        "external_artifact_runtime_path_correction_authority"
    ] == authority
    invariance = C.SCIENTIFIC_INVARIANCE_AUTHORITY
    assert invariance[
        "external_artifact_runtime_path_correction_authority_content_digest"
    ] == authority["content_digest"]
    assert invariance[
        "external_artifact_runtime_path_changes_bytes_or_scientific_identity"
    ] is False
