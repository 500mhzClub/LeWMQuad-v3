from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.benchmarks import go2_world_model_existing_pool_three_arm_v1 as metrics
from lewm.models import (
    rgb_single_frame_multiblock_masked_spatial_jepa_v1 as spatial_model,
)
from scripts import execute_go2_world_model_existing_pool_three_arm_v1 as worker


@dataclass(frozen=True)
class _MetadataRow:
    index: int | str
    role: str
    family: str
    scene_id: str
    actions: tuple[int, ...]


class _SmallTemporalTemplate(nn.Module):
    """Small CPU-only object exposing the exact attributes copied by ArmCore."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            spatial_token_count=4,
            feature_dim=3,
            target_token_count=2,
            action_count=9,
            time_embedding_count=3,
            temporal_hidden_dim=3,
            decoder_token_count=6,
            normalization_epsilon=1.0e-8,
        )
        self.predictor_position = nn.Parameter(torch.randn(4, 3))
        self.predictor_mask_token = nn.Parameter(torch.randn(1, 1, 3))
        self.predictor_blocks = nn.ModuleList([_TokenMixer()])
        self.predictor_norm = nn.LayerNorm(3)
        self.predictor_output = nn.Linear(3, 3)
        self.action_embedding = nn.Embedding(9, 3)
        self.time_embedding = nn.Embedding(3, 3)
        self.temporal_gru = nn.GRU(3, 3, num_layers=1, batch_first=True)


class _TokenMixer(nn.Module):
    """Tiny attention stand-in that lets memory tokens affect query tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.projection(value.mean(dim=1, keepdim=True))


def _small_arm() -> tuple[_SmallTemporalTemplate, worker.ArmCore]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260731)
        template = _SmallTemporalTemplate()
    return template, worker.ArmCore(template)


def test_complete_spatial_predecessor_loader_and_temporal_migration_boundary(
    tmp_path: Path,
) -> None:
    with torch.random.fork_rng(devices=[]):
        encoder = spatial_model._construct_n320_encoder_without_rng_draw()
        predecessor = spatial_model.SingleFrameMultiblockMaskedSpatialJepaV1(
            encoder.state_dict()
        )
    valid_state = {
        name: value.detach().clone()
        for name, value in predecessor.state_dict().items()
    }
    for name in tuple(valid_state):
        if name.startswith("target_encoder."):
            online_name = f"encoder.{name.removeprefix('target_encoder.')}"
            valid_state[name] = valid_state[online_name] + 1.0
    valid_state["ema_update_count"] = torch.tensor(1_000, dtype=torch.long)
    del predecessor, encoder

    accepted = {
        name
        for name in valid_state
        if worker.temporal_model.temporal_v1_accepts_predecessor_key(name)
    }
    rejected = set(valid_state) - accepted
    assert len(valid_state) == 187
    assert (
        sum(value.dtype == torch.float32 for value in valid_state.values()) == 186
    )
    assert sum(value.dtype == torch.long for value in valid_state.values()) == 1
    assert len(accepted) == 108
    assert len(rejected) == 79
    assert sum(name.startswith("target_encoder.") for name in rejected) == 78
    assert {
        name for name in rejected if not name.startswith("target_encoder.")
    } == {"ema_update_count"}

    checkpoint_path = tmp_path / "spatial_v1.pt"
    torch.save({"model_state_dict": valid_state}, checkpoint_path)
    loaded = worker.load_predecessor_state(worker.file_binding(checkpoint_path))
    assert tuple(loaded) == tuple(valid_state)
    for name in valid_state:
        torch.testing.assert_close(
            loaded[name], valid_state[name], rtol=0.0, atol=0.0
        )
    assert loaded["ema_update_count"].dtype == torch.long
    assert loaded["ema_update_count"].shape == torch.Size([])
    assert loaded["ema_update_count"].item() == 1_000

    temporal = worker.temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1(
        loaded
    )
    migrated = temporal.state_dict()
    for name in accepted:
        torch.testing.assert_close(
            migrated[name], loaded[name], rtol=0.0, atol=0.0
        )
    for target_name in sorted(
        name for name in rejected if name.startswith("target_encoder.")
    ):
        online_name = f"encoder.{target_name.removeprefix('target_encoder.')}"
        torch.testing.assert_close(
            migrated[target_name], migrated[online_name], rtol=0.0, atol=0.0
        )
        assert not torch.equal(migrated[target_name], loaded[target_name])
    assert int(temporal.ema_update_count.detach().cpu().item()) == 0

    invalid_state = dict(valid_state)
    invalid_state["ema_update_count"] = torch.tensor(999, dtype=torch.long)
    invalid_path = tmp_path / "spatial_v1_wrong_ema_count.pt"
    torch.save({"model_state_dict": invalid_state}, invalid_path)
    with pytest.raises(
        worker.ThreeArmWorkerError,
        match="predecessor model state is invalid",
    ):
        worker.load_predecessor_state(worker.file_binding(invalid_path))

    sparse_path = tmp_path / "spatial_v1_sparse.pt"
    torch.save(
        {
            "model_state_dict": {
                "encoder.weight": torch.eye(2, dtype=torch.float32).to_sparse(),
                "ema_update_count": torch.tensor(1_000, dtype=torch.long),
            }
        },
        sparse_path,
    )
    with pytest.raises(
        worker.ThreeArmWorkerError,
        match="predecessor model state is invalid",
    ):
        worker.load_predecessor_state(worker.file_binding(sparse_path))


def test_worker_failure_receipt_uses_only_replacement_attempt_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement_root = tmp_path / "replacement" / "attempt_v1"
    replacement_root.mkdir(parents=True)
    consumed_root = tmp_path / "consumed" / "attempt_v1"
    consumed_root.mkdir(parents=True)
    consumed_marker = consumed_root / "unchanged.json"
    consumed_marker.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        worker.supervisor_contract,
        "ATTEMPT_ROOT",
        replacement_root,
    )

    def fail_before_result(**_kwargs: object) -> int:
        raise worker.ThreeArmWorkerError("synthetic replacement failure")

    monkeypatch.setattr(worker, "execute_authorized_experiment", fail_before_result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--authority",
            str(tmp_path / "authority.json"),
            "--expected-authority-byte-count",
            "1",
            "--expected-authority-sha256",
            "a" * 64,
        ],
    )
    with pytest.raises(
        worker.ThreeArmWorkerError,
        match="synthetic replacement failure",
    ):
        worker.main()

    failure_path = replacement_root / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["schema"] == (
        "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_"
        "replacement_v1_worker_failure_v1"
    )
    assert failure["schema"] != (
        "lewm_go2_world_model_existing_pool_three_arm_worker_failure_v1"
    )
    assert failure["status"] == "ATTEMPT_CONSUMED_WORKER_FAILURE"
    assert failure["authorizes_retry_or_resume"] is False
    assert failure["error"] == "synthetic replacement failure"
    assert consumed_marker.read_text(encoding="utf-8") == "{}\n"
    assert not (consumed_root / "failure.json").exists()


def _reference_conditioned_prediction(
    arm: worker.ArmCore,
    encoded_history: torch.Tensor,
    actions: torch.Tensor,
    target_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent compact transcription of Temporal-V1's conditioned route."""

    batch, steps, spatial_tokens, feature_dim = encoded_history.shape
    times = torch.arange(steps, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    conditioning = arm.action_embedding(actions) + arm.time_embedding(times)
    recurrent_input = encoded_history + conditioning.unsqueeze(2)
    streams = recurrent_input.permute(0, 2, 1, 3).reshape(
        batch * spatial_tokens, steps, feature_dim
    )
    initial_hidden = torch.zeros(
        1,
        batch * spatial_tokens,
        arm.config.temporal_hidden_dim,
        dtype=streams.dtype,
    )
    recurrent_streams, _ = arm.temporal_gru(streams, initial_hidden)
    recurrent_memory = recurrent_streams[:, -1].reshape(
        batch, spatial_tokens, arm.config.temporal_hidden_dim
    )
    positions = arm.predictor_position.unsqueeze(0).expand(batch, -1, -1)
    query_positions = torch.gather(
        positions,
        1,
        target_indices.unsqueeze(-1).expand(-1, -1, positions.shape[-1]),
    )
    queries = arm.predictor_mask_token.expand(
        batch, arm.config.target_token_count, -1
    ) + query_positions
    predictor = torch.cat(
        (recurrent_memory + positions, queries),
        dim=1,
    )
    for block in arm.predictor_blocks:
        predictor = block(predictor)
    raw = arm.predictor_output(
        arm.predictor_norm(predictor[:, -arm.config.target_token_count :])
    )
    normalized = F.normalize(
        raw,
        p=2.0,
        dim=-1,
        eps=arm.config.normalization_epsilon,
    )
    return raw, normalized, recurrent_memory


def _full_support_rows() -> tuple[list[_MetadataRow], list[_MetadataRow]]:
    train: list[_MetadataRow] = []
    index = 0
    for first in range(metrics.ACTION_COUNT):
        for second in range(metrics.ACTION_COUNT):
            for third in range(metrics.ACTION_COUNT):
                family = metrics.REGISTERED_FAMILIES[index % len(metrics.REGISTERED_FAMILIES)]
                train.append(
                    _MetadataRow(
                        index=index,
                        role="train",
                        family=family,
                        scene_id=f"train-{family}-{index % 17}",
                        actions=(first, second, third, first, second, third),
                    )
                )
                index += 1
    validation = [
        _MetadataRow(
            index=f"val-{family_index}-{action}",
            role="val",
            family=family,
            scene_id=f"val-{family}",
            actions=(action,) * 6,
        )
        for family_index, family in enumerate(metrics.REGISTERED_FAMILIES)
        for action in range(metrics.ACTION_COUNT)
    ]
    return train, validation


def _shuffle_rows() -> list[_MetadataRow]:
    return [
        _MetadataRow(
            index=family_index * 100 + action * 2 + repeat,
            role="train",
            family=family,
            scene_id=f"shuffle-{family_index}-{action}-{repeat}",
            actions=(8, 7, action, 6, 5, 4),
        )
        for family_index, family in enumerate(metrics.REGISTERED_FAMILIES)
        for action in range(metrics.ACTION_COUNT)
        for repeat in range(2)
    ]


def test_bound_schedule_is_exact_deterministic_and_carries_epoch_tails() -> None:
    schedule, receipt = worker.build_bound_training_schedule(
        row_count=7,
        updates=5,
        batch_size=3,
    )
    repeated, repeated_receipt = worker.build_bound_training_schedule(
        row_count=7,
        updates=5,
        batch_size=3,
    )
    expected_epochs = [
        sorted(
            range(7),
            key=lambda row: (
                hashlib.sha256(
                    f"{worker.TRAIN_ORDER_NAMESPACE}/{epoch}/{row}".encode("ascii")
                ).digest(),
                row,
            ),
        )
        for epoch in range(3)
    ]
    expected = [row for epoch in expected_epochs for row in epoch][:15]

    assert schedule.dtype == torch.long
    assert tuple(schedule.shape) == (5, 3)
    assert schedule.reshape(-1).tolist() == expected
    assert torch.equal(schedule, repeated)
    assert receipt == repeated_receipt
    assert set(schedule.reshape(-1)[:7].tolist()) == set(range(7))
    assert set(schedule.reshape(-1)[7:14].tolist()) == set(range(7))
    assert schedule.reshape(-1)[14].item() == expected_epochs[2][0]
    assert receipt == {
        "seed": worker.TRAINING_SEED,
        "namespace": worker.TRAIN_ORDER_NAMESPACE,
        "algorithm": "per_epoch_sha256_sort_then_contiguous_tail_carry_v1",
        "row_count": 7,
        "updates": 5,
        "batch_size": 3,
        "presentations": 15,
        "epochs_touched": 3,
        "epoch_order_sha256": [
            worker.canonical_sha256(epoch) for epoch in expected_epochs
        ],
        "ordered_uint32be_sha256": hashlib.sha256(
            b"".join(struct.pack(">I", row) for row in expected)
        ).hexdigest(),
    }
    assert f"seed-{worker.TRAINING_SEED}" in receipt["namespace"]


def test_pack_artifacts_receive_six_direct_standard_bindings(tmp_path: Path) -> None:
    attempt_root = tmp_path / "attempt"
    pack_root = attempt_root / "pack"
    pack_root.mkdir(parents=True)
    expected: dict[str, dict[str, dict[str, object]]] = {}
    for role, artifacts in worker.PACK_ARTIFACT_RELATIVE_PATHS.items():
        expected[role] = {}
        for name, relative in artifacts.items():
            payload = f"{role}:{name}".encode("ascii")
            path = attempt_root / relative
            path.write_bytes(payload)
            expected[role][name] = {
                "path": relative,
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
            }

    bindings = worker.build_pack_artifact_bindings(
        pack_root=pack_root,
        attempt_root=attempt_root,
    )
    assert bindings == expected
    assert {
        binding["path"]
        for role in bindings.values()
        for binding in role.values()
    } == {
        relative
        for artifacts in worker.PACK_ARTIFACT_RELATIVE_PATHS.values()
        for relative in artifacts.values()
    }

    (pack_root / "train_frames.u8").unlink()
    with pytest.raises(worker.ThreeArmWorkerError, match="absent or unsafe"):
        worker.build_pack_artifact_bindings(
            pack_root=pack_root,
            attempt_root=attempt_root,
        )


def test_packer_publication_is_exclusive_and_never_overwrites(tmp_path: Path) -> None:
    temporary = tmp_path / "payload.partial"
    destination = tmp_path / "payload.bin"
    temporary.write_bytes(b"new")
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        worker.packer._publish_exclusive(temporary, destination)
    assert destination.read_bytes() == b"existing"
    assert temporary.read_bytes() == b"new"

    destination.unlink()
    worker.packer._publish_exclusive(temporary, destination)
    assert destination.read_bytes() == b"new"
    assert not temporary.exists()


def test_packer_manifest_source_closure_includes_imported_package_initializers() -> None:
    bindings = worker.packer.pack_source_bindings()
    assert set(bindings) == {
        "lewm_package",
        "benchmarks_package",
        "counterfactual_metrics",
        "datasets_package",
        "packer",
        "h6_dataset",
        "h6_main_pool_census",
        "h6_sequence_contract_v2",
        "h6_sequence_contract_v1",
    }
    for name, binding in bindings.items():
        required = worker.REQUIRED_SOURCE_PATHS[name]
        assert binding["path"] == required.as_posix()


def test_arm_core_partition_and_optimizer_are_exact_and_disjoint() -> None:
    template, arm = _small_arm()
    template_parameters = {id(parameter) for parameter in template.parameters()}
    arm_parameters = {id(parameter) for parameter in arm.parameters()}
    assert template_parameters.isdisjoint(arm_parameters)
    assert template.state_dict().keys() == arm.state_dict().keys()
    for name, value in template.state_dict().items():
        torch.testing.assert_close(value, arm.state_dict()[name], rtol=0.0, atol=0.0)

    optimizer, partition = worker.build_arm_optimizer(arm)
    named = dict(arm.named_parameters())
    assert set(partition.predictor_names) == {
        name
        for name in named
        if name in worker._PREDICTOR_EXACT
        or name.startswith(worker._PREDICTOR_PREFIXES)
    }
    assert set(partition.memory_names) == {
        name for name in named if name.startswith(worker._MEMORY_PREFIXES)
    }
    assert {id(parameter) for parameter in partition.all} == arm_parameters
    assert len({id(parameter) for parameter in partition.all}) == len(partition.all)
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "predictor",
        "memory",
    ]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        worker.PREDICTOR_BASE_LR * worker.LR_SCALE
    )
    assert optimizer.param_groups[1]["lr"] == pytest.approx(
        worker.MEMORY_BASE_LR * worker.LR_SCALE
    )
    for group in optimizer.param_groups:
        assert group["betas"] == (0.9, 0.999)
        assert group["eps"] == 1.0e-8
        assert group["weight_decay"] == worker.WEIGHT_DECAY
        assert group["amsgrad"] is False

    _template, invalid = _small_arm()
    invalid.unregistered_parameter = nn.Parameter(torch.zeros(1))
    with pytest.raises(worker.ThreeArmWorkerError, match="unregistered arm parameter"):
        worker.partition_arm_parameters(invalid)


def test_shared_encoding_helper_matches_conditioned_route_and_blinds_only_candidate() -> None:
    _template, arm = _small_arm()
    arm.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        encoded = torch.randn(2, 3, 4, 3, dtype=torch.float32)
    factual_actions = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    changed_candidate = factual_actions.clone()
    changed_candidate[:, 2] = torch.tensor([8, 0])
    target_indices = torch.tensor([[0, 3], [1, 2]], dtype=torch.long)

    conditioned = worker.predict_from_shared_encoding(
        arm,
        encoded,
        factual_actions,
        target_indices,
        candidate_blind=False,
    )
    raw, normalized, memory = _reference_conditioned_prediction(
        arm, encoded, factual_actions, target_indices
    )
    torch.testing.assert_close(conditioned.raw, raw, rtol=0.0, atol=0.0)
    torch.testing.assert_close(conditioned.normalized, normalized, rtol=0.0, atol=0.0)
    torch.testing.assert_close(conditioned.recurrent_memory, memory, rtol=0.0, atol=0.0)

    conditioned_changed = worker.predict_from_shared_encoding(
        arm,
        encoded,
        changed_candidate,
        target_indices,
        candidate_blind=False,
    )
    assert not torch.equal(conditioned.raw, conditioned_changed.raw)

    blind = worker.predict_from_shared_encoding(
        arm,
        encoded,
        factual_actions,
        target_indices,
        candidate_blind=True,
    )
    blind_changed = worker.predict_from_shared_encoding(
        arm,
        encoded,
        changed_candidate,
        target_indices,
        candidate_blind=True,
    )
    torch.testing.assert_close(blind.raw, blind_changed.raw, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        blind.recurrent_memory,
        blind_changed.recurrent_memory,
        rtol=0.0,
        atol=0.0,
    )


def test_overlap_and_shuffle_audit_wrappers_accept_synthetic_metric_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, validation = _full_support_rows()
    overlap = worker.build_overlap_audit(train, validation)
    assert overlap["schema"] == worker.OVERLAP_AUDIT_SCHEMA
    assert overlap["status"] == "PASS"
    assert overlap["role_scene_overlap_count"] == 0
    assert overlap["role_row_counts"] == {
        "train": len(train),
        "val": len(validation),
    }

    shuffle_rows = _shuffle_rows()
    derangement = metrics.build_candidate_action_derangement(shuffle_rows)
    monkeypatch.setattr(worker, "EXPECTED_TRAIN_ROWS", len(shuffle_rows))
    shuffle = worker.build_shuffle_audit(derangement, shuffle_rows)
    assert shuffle == derangement.to_dict()
    assert shuffle["schema"] == worker.SHUFFLE_AUDIT_SCHEMA
    assert shuffle["status"] == "PASS"
    assert shuffle["checks"]["role_family_action_marginals_exact"]


def test_action_identification_receipt_propagates_bayesian_audit_evidence() -> None:
    support = {
        family: tuple(
            2 + int(family_index == 7 and action == 8) for action in range(9)
        )
        for family_index, family in enumerate(metrics.REGISTERED_FAMILIES)
    }
    summary = SimpleNamespace(
        row_count=2_048,
        exact_tie_row_count=0,
        unique_winner_count=2_048,
        bootstrap_algorithm=metrics.ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM,
        bootstrap_interpretation=(
            metrics.ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION
        ),
        bootstrap_seed=metrics.ACTION_IDENTIFICATION_BOOTSTRAP_SEED,
        bootstrap_replicates=metrics.BOOTSTRAP_REPLICATES,
        bootstrap_lower_index=500,
        family_action_supporting_scene_counts=support,
        minimum_family_action_supporting_scene_count=2,
        scene_family_balanced_accuracy=0.5,
        balanced_accuracy_bootstrap_lower_95=0.3,
        exact_tie_rate=0.0,
        unique_winner_accuracy=0.5,
        hardest_action_margin=0.2,
        hardest_margin_bootstrap_lower_95=0.1,
    )

    receipt = worker._action_identification_receipt(summary)

    assert receipt == {
        "bootstrap_algorithm": metrics.ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM,
        "bootstrap_interpretation": (
            metrics.ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION
        ),
        "bootstrap_seed": 20_260_803,
        "bootstrap_replicates": 10_000,
        "bootstrap_lower_index": 500,
        "family_action_supporting_scene_counts": {
            family: list(counts) for family, counts in support.items()
        },
        "minimum_family_action_supporting_scene_count": 2,
        "balanced_accuracy": 0.5,
        "balanced_accuracy_one_sided_95_lower_bound": 0.3,
        "balanced_chance": 1.0 / 9.0,
        "exact_tie_count": 0,
        "exact_tie_rate": 0.0,
        "unique_winner_count": 2_048,
        "unique_winner_accuracy": 0.5,
        "hardest_wrong_action_margin": 0.2,
        "hardest_wrong_action_margin_one_sided_95_lower_bound": 0.1,
    }
    assert all(
        type(counts) is list
        for counts in receipt["family_action_supporting_scene_counts"].values()
    )

    inconsistent = SimpleNamespace(**vars(summary))
    inconsistent.unique_winner_count = 2_047
    with pytest.raises(worker.ThreeArmWorkerError, match="unique-winner accounting"):
        worker._action_identification_receipt(inconsistent)


def test_training_fit_uses_row_then_family_metric_and_each_arms_own_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = metrics.REGISTERED_FAMILIES
    family_ids = [families[0], families[0], families[0], *families[1:]]
    scene_ids = [
        "family-zero-large-scene",
        "family-zero-large-scene",
        "family-zero-small-scene",
        *(f"scene-{family}" for family in families[1:]),
    ]
    rows = [
        _MetadataRow(
            index=index,
            role="train",
            family=family,
            scene_id=scene,
            actions=(0, 1, 2, 3, 4, 5),
        )
        for index, (family, scene) in enumerate(zip(family_ids, scene_ids, strict=True))
    ]
    conditioned = torch.ones(len(rows), dtype=torch.float64)
    blind_log_advantages = torch.tensor(
        [1.0, 1.0, -1.0, *([0.2] * 7)], dtype=torch.float64
    )
    shuffled_log_advantages = torch.tensor(
        [2.0, 2.0, -2.0, *([0.4] * 7)], dtype=torch.float64
    )
    blind = blind_log_advantages.exp()
    shuffled = shuffled_log_advantages.exp()
    vectors = worker.EvaluationVectors(
        role="train",
        row_indices=tuple(range(len(rows))),
        factual_energy={
            "conditioned": conditioned,
            "blind": blind,
            "shuffled": shuffled,
        },
        persistence_energy=None,
        wrong_history_energy={},
        candidate_energy={},
        prediction_tokens={},
        target_tokens=None,
        blind_candidate_max_spread=None,
    )
    monkeypatch.setattr(worker, "EXPECTED_TRAIN_ROWS", len(rows))

    summaries, comparisons = worker.analyze_training_fit(vectors, rows=rows)
    expected_blind = ((1.0 + 1.0 - 1.0) / 3.0 + 7.0 * 0.2) / 8.0
    expected_shuffled = ((2.0 + 2.0 - 2.0) / 3.0 + 7.0 * 0.4) / 8.0
    assert set(comparisons) == {"blind", "shuffled"}
    assert all(
        isinstance(value, metrics.FamilyEqualLogEnergyAdvantage)
        for value in comparisons.values()
    )
    assert comparisons["blind"].macro_log_advantage == pytest.approx(expected_blind)
    assert comparisons["shuffled"].macro_log_advantage == pytest.approx(
        expected_shuffled
    )
    # Equal-scene weighting would make family zero's blind value zero, not 1/3.
    assert comparisons["blind"].macro_log_advantage != pytest.approx(
        (7.0 * 0.2) / 8.0
    )
    assert summaries["conditioned"]["factual_mean_energy"] == 1.0
    assert summaries["blind"]["factual_mean_energy"] == pytest.approx(
        float(blind.mean())
    )
    assert summaries["shuffled"]["factual_mean_energy"] == pytest.approx(
        float(shuffled.mean())
    )
    assert summaries["blind"][
        "conditioned_vs_blind_family_equal_log_energy_advantage"
    ] == pytest.approx(expected_blind)


def test_measurement_payload_has_the_exact_registered_shape() -> None:
    authority = {"path": "docs/synthetic_authority.json", "sha256": "a" * 64}
    plan = {"path": "docs/synthetic_plan.json", "sha256": "b" * 64}
    substrate = {"encoder_sha256": "c" * 64, "target_sha256": "d" * 64}
    validation = {"factual_mean_energy": 1.25, "row_count": 2_048}
    payload = worker.measurement_payload(
        arm_name="conditioned",
        update=0,
        authority_binding=authority,
        plan_binding=plan,
        substrate_receipt=substrate,
        validation=validation,
        training=None,
        loss=None,
        learning_rate={"predictor": 0.0, "memory": 0.0},
    )
    assert payload == {
        "schema": worker.MEASUREMENT_SCHEMA,
        "status": "COMPLETE",
        "arm": "conditioned",
        "update": 0,
        "authority_binding": authority,
        "plan_binding": plan,
        "encoder_sha256": "c" * 64,
        "target_sha256": "d" * 64,
        "panel": {
            "kind": "scene_disjoint_factual_validation",
            "row_count": worker.EXPECTED_VALIDATION_ROWS,
            "row_indices_sha256": worker.canonical_sha256(
                list(range(worker.EXPECTED_VALIDATION_ROWS))
            ),
        },
        "validation": validation,
        "training": None,
        "optimization": {
            "completed_updates": 0,
            "optimizer_steps": 0,
            "loss": None,
            "learning_rate_fraction": 0.0,
            "predictor_learning_rate": 0.0,
            "memory_learning_rate": 0.0,
            "warmup_updates": worker.WARMUP_UPDATES,
            "schedule_horizon_updates": worker.COSINE_SCHEDULE_UPDATES,
        },
        "integrity": {
            "candidate_blind_treatment_exact": True,
            "shuffled_derangement_exact": True,
            "factual_evaluation_exact": True,
            "frozen_substrate_exact": True,
            "no_gradient_during_evaluation": True,
            "finite": True,
        },
    }
    with pytest.raises(worker.ThreeArmWorkerError, match="update-zero loss"):
        worker.measurement_payload(
            arm_name="conditioned",
            update=0,
            authority_binding=authority,
            plan_binding=plan,
            substrate_receipt=substrate,
            validation=validation,
            training=None,
            loss=0.0,
            learning_rate={"predictor": 0.0, "memory": 0.0},
        )
    with pytest.raises(worker.ThreeArmWorkerError, match="loss is invalid"):
        worker.measurement_payload(
            arm_name="conditioned",
            update=100,
            authority_binding=authority,
            plan_binding=plan,
            substrate_receipt=substrate,
            validation=validation,
            training=None,
            loss=None,
            learning_rate={"predictor": 0.0, "memory": 0.0},
        )


def test_exact_accounting_matches_the_preregistered_budget() -> None:
    assert worker.exact_accounting() == {
        "bound_h6_rows": 18_048,
        "initial_rgb_leaf_opens": 72_192,
        "verification_rgb_leaf_reopens": 192,
        "total_rgb_leaf_opens": 72_384,
        "forbidden_future_rgb_leaf_opens": 0,
        "packed_frame_bytes": 2_716_729_344,
        "training_schedule_row_presentations": 179_200,
        "sequence_presentations_per_arm": 179_200,
        "total_arm_head_sequence_presentations": 537_600,
        "shared_online_context_frame_encodings": 537_600,
        "shared_future_target_frame_encodings": 179_200,
        "actual_training_frame_encodings": 716_800,
        "optimizer_steps_per_arm": 700,
        "total_optimizer_steps": 2_100,
        "target_ema_steps": 0,
        "validation_row_panels_per_arm": 16_384,
        "shared_validation_frame_encodings": 65_536,
        "nine_way_arm_candidate_row_queries": 442_368,
        "validation_backward_calls": 0,
        "validation_optimizer_steps": 0,
        "train_fit_rows": 16_000,
        "train_fit_shared_frame_encodings": 64_000,
        "train_fit_arm_factual_row_queries": 48_000,
        "train_fit_backward_calls": 0,
        "train_fit_optimizer_steps": 0,
        "total_shared_frame_encodings": 846_336,
        "measurement_receipts": 24,
        "snapshot_bindings": 24,
        "sealed_open_count": 0,
        "heldout_open_count": 0,
        "network_access_count": 0,
        "training_consumed_pack_only": True,
    }


def test_isolated_worker_import_has_no_unbound_local_python_dependency() -> None:
    probe = r"""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from scripts import execute_go2_world_model_existing_pool_three_arm_v1 as worker

observed = set()
for module in tuple(sys.modules.values()):
    filename = getattr(module, "__file__", None)
    if not filename:
        continue
    selected = Path(filename)
    if not selected.is_absolute():
        continue
    try:
        relative = selected.resolve().relative_to(root)
    except ValueError:
        continue
    if relative.suffix == ".py" and relative.parts[0] in {"lewm", "scripts"}:
        observed.add(relative.as_posix())
required = {path.as_posix() for path in worker.REQUIRED_SOURCE_PATHS.values()}
checker = worker.REQUIRED_SOURCE_PATHS["checker"].as_posix()
expected = observed | {checker}
unbound = sorted(expected - required)
missing = sorted(required - expected)
supervisor_required = set(worker.supervisor_contract.REQUIRED_SOURCE_PATHS.values())
maps_differ = required != supervisor_required
print(json.dumps({
    "observed": sorted(observed),
    "unbound": unbound,
    "missing": missing,
    "maps_differ": maps_differ,
}))
raise SystemExit(bool(unbound or missing or maps_differ))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe, str(Path(worker.REPO_ROOT))],
        cwd=worker.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    diagnostic = json.loads(completed.stdout)
    assert completed.returncode == 0, completed.stderr or diagnostic
    assert diagnostic["unbound"] == []
    assert diagnostic["missing"] == []
    assert diagnostic["maps_differ"] is False
    for required in (
        "lewm/__init__.py",
        "lewm/benchmarks/__init__.py",
        "lewm/benchmarks/counterfactual.py",
        "lewm/datasets/__init__.py",
        "lewm/models/__init__.py",
        "lewm/models/lewm.py",
        "lewm/models/phase2d_spatial_lewm.py",
        "lewm/models/predictor.py",
        "lewm/models/primitive_affordance.py",
        "lewm/models/sigreg.py",
        "lewm/models/source_action_utility.py",
        "lewm/models/spatial_lewm.py",
        "lewm/models/spatial_predictor.py",
    ):
        assert required in {
            path.as_posix() for path in worker.REQUIRED_SOURCE_PATHS.values()
        }
