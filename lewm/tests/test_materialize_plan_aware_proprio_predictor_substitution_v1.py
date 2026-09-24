from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import materialize_plan_aware_proprio_predictor_substitution_v1 as M


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _gate_fixture(tmp_path: Path) -> tuple[Path, dict]:
    evaluation = tmp_path / "receipts/evaluation_contract.json"
    evidence = tmp_path / "aggregates/stage_a_gate.json"
    checkpoint = tmp_path / "checkpoints/latent.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"final-epoch-route-ranker")
    evaluation_value = M.attach_digest(
        {
            "experiment_contract_digest": (
                M.CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
            ),
            "final_epoch_only": True,
            "predictor_inference_authorised_before_true_gate": False,
            "checkpoint_bindings": {
                "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL": M.artifact_binding(
                    checkpoint, root=tmp_path
                )
            },
            "pass": True,
        }
    )
    _write_json(evaluation, evaluation_value)
    _write_json(
        evidence,
        M.attach_digest(
            {
                "schema": M.STAGE_A_GATE_EVIDENCE_SCHEMA,
                "experiment_id": M.EXPERIMENT_ID,
                "contract_sha256": (
                    M.CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
                ),
                "source_freeze_commit": "a" * 40,
                "evaluation_contract_content_digest": evaluation_value[
                    "content_digest"
                ],
                "true_future_gate": {
                    "classification": "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL",
                    "pass": True,
                },
                "pass": True,
            }
        ),
    )
    value = M.build_stage_b_gate_receipt(
        contract_freeze_commit="a" * 40,
        evaluation_contract_path=evaluation,
        stage_a_gate_evidence_path=evidence,
        latent_ranker_checkpoint_path=checkpoint,
        output_root=tmp_path,
    )
    path = tmp_path / "receipts/stage_b_gate.json"
    M.atomic_json(path, value)
    return path, value


def test_child_environment_scrubs_python_path_and_preserves_venv_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/usr/lib/python3/dist-packages")
    monkeypatch.setenv("PYTHONHOME", "/synthetic/wrong-prefix")
    monkeypatch.setenv("PYTHONUSERBASE", "/synthetic/wrong-user-base")
    environment = M.build_child_environment(M.CPU_INTERPRETER)
    for key in M.SCRUBBED_PYTHON_ENVIRONMENT_KEYS:
        assert key not in environment
    assert environment["VIRTUAL_ENV"] == str(M.CPU_INTERPRETER.parent.parent)
    assert environment["PATH"].split(M.os.pathsep)[0] == str(
        M.CPU_INTERPRETER.parent
    )
    assert M.child_environment_contract(M.CPU_INTERPRETER)[
        "isolated_python_environment_flags"
    ] == ["-E", "-s"]
    # Resolving bin/python would collapse the venv launcher to the system
    # interpreter on this host; the custody contract must retain lexical bytes.
    assert M.child_environment_contract(M.CPU_INTERPRETER)["interpreter"] == str(
        M.CPU_INTERPRETER.absolute()
    )
    policy = M.CONTRACT.EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
        "only_authorised_environment_change"
    ]["per_interpreter"]
    for authority_id, interpreter in (
        ("cpu_child", M.CPU_INTERPRETER),
        ("gpu_child", M.GPU_INTERPRETER),
    ):
        contract = M.child_environment_contract(interpreter)
        assert contract["authority_id"] == authority_id
        assert contract["virtual_env"] == policy[authority_id]["VIRTUAL_ENV"]
        assert contract["path_first_entry"] == policy[authority_id]["PATH_prepend"]


def test_cpu_child_import_preflight_defeats_inherited_system_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/usr/lib/python3/dist-packages")
    value = M.probe_child_interpreter(M.CPU_INTERPRETER, require_genesis=True)
    venv_root = M.CPU_INTERPRETER.parent.parent.resolve()
    assert value["sentinel_available"] is True
    assert Path(value["typing_extensions"]["path"]).resolve().is_relative_to(
        venv_root
    )
    assert Path(value["pydantic_core_or_null"]["path"]).resolve().is_relative_to(
        venv_root
    )
    assert Path(value["genesis_or_null"]["path"]).resolve().is_relative_to(venv_root)


def test_sensed_proprio_feature_applies_only_frozen_gravity_offset() -> None:
    deployment = np.arange(42, dtype=np.float32) / 10
    deployment[:3] = (0.1, -0.2, -0.9)
    observed = M.sensed_proprio_feature(deployment)
    assert observed.shape == (30,)
    assert observed.dtype == np.float32
    np.testing.assert_allclose(
        observed[:3], np.asarray((0.1, -0.2, 0.1), np.float32), rtol=0, atol=3e-8
    )
    np.testing.assert_array_equal(observed[3:], deployment[3:30])


def test_normalize_proprio_history_is_channelwise_and_shape_strict() -> None:
    raw = np.arange(np.prod(M.PROPRIO_SHAPE), dtype=np.float32).reshape(M.PROPRIO_SHAPE)
    mean = np.arange(30, dtype=np.float32)
    std = np.arange(1, 31, dtype=np.float32)
    result = M.normalize_proprio_history(raw, mean, std)
    np.testing.assert_allclose(result, (raw - mean) / std, rtol=0, atol=0)
    with pytest.raises(M.MaterialisationError):
        M.normalize_proprio_history(raw[:, :, :-1], mean, std)
    with pytest.raises(M.MaterialisationError):
        M.normalize_proprio_history(raw, mean, np.zeros(30, np.float32))


def test_control_history_matches_applied_k_minus_one_window() -> None:
    blocks = {
        index: np.arange((index - 37) * 15, (index - 36) * 15, dtype=np.float32).reshape(5, 3)
        for index in (37, 38, 39, 40)
    }
    rows = np.concatenate([blocks[index] for index in (37, 38, 39, 40)], axis=0)
    expected = rows[4:19, (0, 2)].reshape(3, 5, 2)
    np.testing.assert_array_equal(M._control_history_from_blocks(blocks), expected)


def test_stage_b_gate_requires_explicit_true_future_pass_and_bound_files(
    tmp_path: Path,
) -> None:
    path, value = _gate_fixture(tmp_path)
    assert M.validate_stage_b_gate_receipt(path, output_root=tmp_path) == value

    failed = dict(value)
    failed["true_future_gate"] = {
        "classification": "PLAN_AWARE_JEPA_COST_NO_SIGNAL",
        "pass": False,
    }
    failed = M.attach_digest(failed)
    M.atomic_json(path, failed)
    with pytest.raises(M.MaterialisationError, match="did not explicitly pass"):
        M.validate_stage_b_gate_receipt(path, output_root=tmp_path)


def test_stage_b_entrypoint_refuses_absent_explicit_flag() -> None:
    with pytest.raises(M.MaterialisationError, match="stage-b-authorised"):
        M.main(
            [
                "context-state",
                "--gate-receipt",
                "/does/not/exist",
                "--execution-correction-replay-receipt",
                "/does/not/exist",
                "--output-root",
                "/does/not/exist",
                "--state-index",
                "0",
            ]
        )


def test_execution_correction_replay_receipt_is_exact_and_tamper_closed(
    tmp_path: Path,
) -> None:
    inventory = {
        str(row["path"]): copy.deepcopy(row)
        for row in M.CONTRACT.EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS
    }
    value = M.attach_digest(
        {
            "schema": M.CONTRACT.EXECUTION_CORRECTION_REPLAY_SCHEMA_VERSION,
            "experiment_id": M.EXPERIMENT_ID,
            "amendment": copy.deepcopy(
                M.CONTRACT.EXECUTION_CORRECTION_AMENDMENT_BINDING
            ),
            "failed_archive": str(
                M.CONTRACT.EXECUTION_CORRECTION_FAILED_ARCHIVE
            ),
            "fresh_attempt": str(tmp_path),
            "files_reused": 0,
            "byte_exact_replay": [
                inventory[path]
                for path in M.CONTRACT.EXECUTION_CORRECTION_BYTE_EXACT_REPLAY_PATHS
            ],
            "normalized_scientific_replay": [
                {
                    "path": path,
                    "excluded_paths": list(
                        M.CONTRACT.EXECUTION_CORRECTION_NORMALIZED_REPLAY_EXCLUSIONS[
                            path
                        ]
                    ),
                    "scientific_content_digest": digest,
                }
                for path, digest in (
                    M.CONTRACT.EXECUTION_CORRECTION_NORMALIZED_REPLAY_DIGESTS.items()
                )
            ],
            "stage_b_started_before_replay_gate": False,
            "stage_c_started_before_replay_gate": False,
            "pass": True,
        }
    )
    path = tmp_path / "receipts/execution_correction_replay.json"
    M.atomic_json(path, value)
    assert M.validate_execution_correction_replay_receipt(
        path, output_root=tmp_path
    ) == value

    tampered = copy.deepcopy(value)
    tampered["files_reused"] = 1
    M.atomic_json(path, M.attach_digest(tampered))
    with pytest.raises(M.MaterialisationError, match="replay receipt drift"):
        M.validate_execution_correction_replay_receipt(path, output_root=tmp_path)


def test_stage_c_has_independent_contribution_gate(tmp_path: Path) -> None:
    stage_b_path, _stage_b = _gate_fixture(tmp_path)
    metrics = tmp_path / "aggregates/stage_b_metrics.json"
    _write_json(
        metrics,
        M.attach_digest(
            {
                "schema": M.STAGE_B_GATE_EVIDENCE_SCHEMA,
                "experiment_id": M.EXPERIMENT_ID,
                "contract_sha256": (
                    M.CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
                ),
                "source_freeze_commit": "a" * 40,
                "proprioception_gate": {
                    "classification": "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION",
                    "pass": True,
                },
                "pass": True,
            }
        ),
    )
    evaluation = tmp_path / "receipts/evaluation_contract.json"
    value = M.build_stage_c_gate_receipt(
        contract_freeze_commit="a" * 40,
        stage_b_gate_receipt_path=stage_b_path,
        stage_b_metrics_path=metrics,
        evaluation_contract_path=evaluation,
        output_root=tmp_path,
    )
    path = tmp_path / "receipts/stage_c_gate.json"
    M.atomic_json(path, value)
    assert M.validate_stage_c_gate_receipt(
        path, output_root=tmp_path, stage_b_gate_path=stage_b_path
    ) == value

    failed = dict(value)
    failed["proprioceptive_route_contribution"] = {
        "classification": "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION_NOT_SUPPORTED",
        "pass": False,
    }
    M.atomic_json(path, M.attach_digest(failed))
    with pytest.raises(M.MaterialisationError, match="was not supported"):
        M.validate_stage_c_gate_receipt(
            path, output_root=tmp_path, stage_b_gate_path=stage_b_path
        )


def test_donor_component_selection_changes_exactly_one_input() -> None:
    mapping = {"recipient": "donor"}
    assert M.donor_input_sources(
        recipient="recipient",
        ablation="PR_VISUAL_CONTEXT_DERANGED",
        mapping=mapping,
    ) == {"visual": "donor", "proprio": "recipient", "control": "recipient"}
    assert M.donor_input_sources(
        recipient="recipient",
        ablation="PR_PROPRIO_HISTORY_DERANGED",
        mapping=mapping,
    ) == {"visual": "recipient", "proprio": "donor", "control": "recipient"}
    assert M.donor_input_sources(
        recipient="recipient",
        ablation="PR_CONTROL_HISTORY_DERANGED",
        mapping=mapping,
    ) == {"visual": "recipient", "proprio": "recipient", "control": "donor"}


def test_stage_c_mapping_enforces_same_family_split_and_derangement() -> None:
    states = {
        "purpose-0": {"family": "A"},
        "purpose-1": {"family": "A"},
    }
    contexts = {
        "purpose-0": {"split_role": "fit"},
        "purpose-1": {"split_role": "fit"},
    }
    mapping = {"purpose-0": "purpose-1", "purpose-1": "purpose-0"}
    evaluation = {"stage_c_input_donor_mappings": {name: mapping for name in M.STAGE_C_ABLATIONS}}
    assert M._stage_c_mapping(
        ablation="PR_VISUAL_CONTEXT_DERANGED",
        evaluation_contract=evaluation,
        states=states,
        contexts=contexts,
    ) == mapping
    contexts["purpose-1"]["split_role"] = "heldout"
    with pytest.raises(M.MaterialisationError, match="crosses frozen split"):
        M._stage_c_mapping(
            ablation="PR_VISUAL_CONTEXT_DERANGED",
            evaluation_contract=evaluation,
            states=states,
            contexts=contexts,
        )


def test_logical_prediction_loader_uses_persisted_state_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(M, "TENSOR_SHAPE", (2, 3))
    monkeypatch.setattr(M, "PREDICTION_STATE_SHAPE", (2, 3, 2, 3))
    monkeypatch.setattr(M, "CANDIDATE_COUNT", 2)
    monkeypatch.setattr(M, "HORIZONS", (1, 2, 3))
    array = np.arange(36, dtype=np.float16).reshape(2, 3, 2, 3)
    path = tmp_path / "state.npy"
    M.atomic_npy(path, array)
    record = {
        **M.artifact_binding(path, root=tmp_path),
        "array_index": [1, 2],
    }
    np.testing.assert_array_equal(
        M.load_logical_prediction(record, output_root=tmp_path), array[1, 2]
    )


def test_source_is_inference_only_and_uses_frozen_absence_unroll() -> None:
    source = Path(M.__file__).read_text(encoding="utf-8")
    assert "P.unroll(" in source
    assert "P.rollout_validity(" in source
    assert "future_proprioception_read_count\"] = 0" in source
    assert "optimizer.step(" not in source
    assert "optimiser.step(" not in source
    assert "backward(" not in source
    assert M.FROZEN_WORKERS == 24
