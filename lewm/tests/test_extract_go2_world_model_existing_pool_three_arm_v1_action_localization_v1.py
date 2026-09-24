from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from lewm.benchmarks import go2_world_model_existing_pool_three_arm_v1 as three_arm
from lewm.benchmarks import go2_world_model_v3_action_localization_v1 as metrics
from scripts import (
    check_go2_world_model_existing_pool_three_arm_v1_action_localization_v1 as checker,
)
from scripts import (
    extract_go2_world_model_existing_pool_three_arm_v1_action_localization_v1 as worker,
)


def _snapshot_payload(*, arm: str = "conditioned", state_value: float = 1.0) -> dict[str, object]:
    count = worker.h6.VALIDATION_INDEX_ROWS
    candidate = torch.full((count, 9), 2.0, dtype=torch.float64)
    candidate[:, 0] = 1.0
    return {
        "schema": worker.SNAPSHOT_SCHEMA,
        "status": "INERT_AUDIT_SNAPSHOT",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "arm": arm,
        "update": 700,
        "authority_binding": worker.V3_INTERNAL_AUTHORITY_BINDING,
        "plan_binding": worker.V3_INTERNAL_PLAN_BINDING,
        "substrate": {
            "encoder_sha256": worker.V3_SUBSTRATE_SHA256,
            "target_sha256": worker.V3_SUBSTRATE_SHA256,
        },
        "schedule": {"updates": 700},
        "metric_vectors": {
            "validation_row_indices": list(range(count)),
            "validation_factual_energy": torch.ones(count, dtype=torch.float64),
            "validation_persistence_energy": torch.full(
                (count,), 1.5, dtype=torch.float64
            ),
            "validation_wrong_history_energy": torch.full(
                (count,), 2.0, dtype=torch.float64
            ),
            "validation_candidate_energy": candidate,
            "prediction_tokens": torch.tensor([3.0]),
            "target_tokens": torch.tensor([4.0]),
            "training_row_indices": [],
            "training_factual_energy": torch.tensor([5.0]),
        },
        "arm_state_dict": {"weight": torch.tensor([state_value])},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
    }


def _write_snapshot(path: Path, payload: dict[str, object]) -> dict[str, object]:
    torch.save(payload, path)
    return worker.file_binding(path)


def test_bound_snapshot_loader_hashes_before_weights_only_load_and_ignores_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_load = worker.torch.load
    load_calls: list[dict[str, object]] = []

    def recording_load(payload: object, **kwargs: object) -> object:
        load_calls.append(dict(kwargs))
        return real_load(payload, **kwargs)

    monkeypatch.setattr(worker.torch, "load", recording_load)
    first_path = tmp_path / "first.pt"
    first_binding = _write_snapshot(first_path, _snapshot_payload(state_value=1.0))
    monkeypatch.setattr(worker, "SNAPSHOT_BINDING", first_binding)
    first, contract = worker.load_bound_snapshot_metric_vectors()
    assert load_calls == [{"map_location": "cpu", "weights_only": True}]
    assert contract["arm"] == "conditioned"
    assert contract["update"] == 700
    assert not contract["model_or_optimizer_state_consumed_computationally"]
    assert set(first) == {
        "validation_row_indices",
        "validation_factual_energy",
        "validation_persistence_energy",
        "validation_wrong_history_energy",
        "validation_candidate_energy",
    }

    second_path = tmp_path / "second.pt"
    second_binding = _write_snapshot(second_path, _snapshot_payload(state_value=99.0))
    monkeypatch.setattr(worker, "SNAPSHOT_BINDING", second_binding)
    second, second_contract = worker.load_bound_snapshot_metric_vectors()
    assert load_calls == [
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu", "weights_only": True},
    ]
    assert second_contract == contract
    for name in first:
        if isinstance(first[name], torch.Tensor):
            assert torch.equal(first[name], second[name])
        else:
            assert first[name] == second[name]

    called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("deserializer ran before binding validation")

    invalid = dict(second_binding)
    invalid["file_sha256"] = "0" * 64
    monkeypatch.setattr(worker, "SNAPSHOT_BINDING", invalid)
    monkeypatch.setattr(worker.torch, "load", forbidden_load)
    with pytest.raises(worker.LocalizationWorkerError, match="identity changed"):
        worker.load_bound_snapshot_metric_vectors()
    assert not called


def test_bound_snapshot_loader_rejects_wrong_arm_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong_path = tmp_path / "wrong.pt"
    binding = _write_snapshot(wrong_path, _snapshot_payload(arm="blind"))
    monkeypatch.setattr(worker, "SNAPSHOT_BINDING", binding)
    with pytest.raises(worker.LocalizationWorkerError, match="envelope"):
        worker.load_bound_snapshot_metric_vectors()

    link_path = tmp_path / "link.pt"
    link_path.symlink_to(wrong_path)
    link_binding = {**binding, "path": str(link_path)}
    monkeypatch.setattr(worker, "SNAPSHOT_BINDING", link_binding)
    with pytest.raises(OSError):
        worker.load_bound_snapshot_metric_vectors()


def _localization_fixture() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for family in three_arm.REGISTERED_FAMILIES:
        for scene_number in range(2):
            for action in range(9):
                rows.append(
                    {
                        "index": len(rows),
                        "role": "val",
                        "family": family,
                        "scene_id": f"{family}_scene_{scene_number}",
                        "actions": [0, 1, action, 3, 4, 5],
                    }
                )
    actions = np.asarray([int(row["actions"][2]) for row in rows])
    candidate = np.full((len(rows), 9), 2.0)
    candidate[np.arange(len(rows)), actions] = 1.0
    candidate[actions == 2, 0] = 0.8
    factual = candidate[np.arange(len(rows)), actions]
    persistence = np.full(len(rows), 1.5)
    persistence[actions == 2] = 0.9
    return metrics.localize_action_and_controls(
        candidate_energies=candidate,
        factual_energy=factual,
        persistence_energy=persistence,
        wrong_history_energy=np.full(len(rows), 2.0),
        validation_rows=rows,
    )


def test_receipt_checker_recomputes_aggregate_arithmetic_without_payloads() -> None:
    localization = _localization_fixture()
    checker._check_localization(
        localization,
        expected_rows=144,
        expected_scenes=16,
    )

    bad_confusion = copy.deepcopy(localization)
    bad_confusion["action_identification"]["confusion_matrix"][0][0] += 1
    with pytest.raises(checker.LocalizationCheckError, match="confusion accounting"):
        checker._check_localization(
            bad_confusion,
            expected_rows=144,
            expected_scenes=16,
        )

    bad_matrix = copy.deepcopy(localization)
    bad_matrix["pairwise_family_equal_scene_macro_margin_matrix"]["values"][0][0] = 1.0
    with pytest.raises(checker.LocalizationCheckError, match="action diagnostic"):
        checker._check_localization(
            bad_matrix,
            expected_rows=144,
            expected_scenes=16,
        )

    mutations: list[tuple[dict[str, object], str]] = []
    bad_persistence = copy.deepcopy(localization)
    bad_persistence["failure_topology"][
        "persistence_lower_failure_action_ids"
    ] = []
    mutations.append((bad_persistence, "failure topology"))

    bad_scope = copy.deepcopy(localization)
    bad_scope["failure_topology"]["alignment_point_failure_scope"] = "broad"
    mutations.append((bad_scope, "failure topology"))

    bad_family = copy.deepcopy(localization)
    del bad_family["persistence_localization"]["per_action"][0]["point_by_family"][
        three_arm.REGISTERED_FAMILIES[0]
    ]
    mutations.append((bad_family, "action 0 contract"))

    bad_weight = copy.deepcopy(localization)
    bad_weight["action_diagnostics"][0]["inverse_uniform_train_weight"] = -999.0
    mutations.append((bad_weight, "action diagnostic"))

    bad_route = copy.deepcopy(localization)
    bad_route["routing_decision"]["alignment_route"] = "ALIGNMENT_PASSED"
    mutations.append((bad_route, "routing"))

    bad_seed = copy.deepcopy(localization)
    bad_seed["wrong_history_localization"]["bootstrap_seed"] += 1
    mutations.append((bad_seed, "bootstrap contract"))

    bad_support = copy.deepcopy(localization)
    support = bad_support["action_margin_localization"]["per_action"][0][
        "supporting_scene_count_by_family"
    ]
    for family in support:
        support[family] = 7
    bad_support["action_margin_localization"]["per_action"][0][
        "minimum_supporting_scene_count"
    ] = 7
    bad_support["action_margin_localization"]["per_action"][0][
        "total_supporting_scene_count"
    ] = 56
    mutations.append((bad_support, "action 0 contract"))

    bad_mrr = copy.deepcopy(localization)
    bad_mrr["action_diagnostics"][0][
        "row_weighted_factual_mean_reciprocal_rank"
    ] = 0.123
    mutations.append((bad_mrr, "action diagnostic"))

    extra = copy.deepcopy(localization)
    extra["persistence_localization"]["unexpected_scene_values"] = []
    mutations.append((extra, "bootstrap contract"))

    bad_unique = copy.deepcopy(localization)
    bad_unique["action_identification"]["unique_winner_accuracy"] = 0.123456
    mutations.append((bad_unique, "tie accounting"))

    for changed, message in mutations:
        with pytest.raises(checker.LocalizationCheckError, match=message):
            checker._check_localization(
                changed,
                expected_rows=144,
                expected_scenes=16,
            )
