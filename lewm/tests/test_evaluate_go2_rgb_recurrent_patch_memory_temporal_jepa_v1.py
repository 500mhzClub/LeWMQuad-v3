from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from PIL import Image
import pytest
import torch

from lewm.benchmarks import (
    go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as metrics,
)
from lewm.datasets import (
    go2_explicit_plan_discounted_successor_state_v27 as h6,
)
from scripts import (
    evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as evaluation,
)


def _scene(family: str, value: int) -> str:
    return f"{family}_{value:012x}"


def _row(
    index: int,
    *,
    role: str,
    family: str | None = None,
    scene_value: int | None = None,
    actions: tuple[int, ...] = (0, 1, 2, 7, 8, 0),
) -> h6.H6V2Row:
    selected_family = family or metrics.REGISTERED_FAMILIES[index % 8]
    selected_scene = _scene(
        selected_family,
        index + (0 if role == "train" else 1_000_000)
        if scene_value is None
        else scene_value,
    )
    return h6.H6V2Row(
        index=index,
        role=role,
        family=selected_family,
        scene_id=selected_scene,
        rgb=tuple(
            (
                f"{selected_scene}/rgb/"
                f"frame_{position:06d}_env_{position:02d}.png"
            )
            for position in range(7)
        ),
        actions=actions,
    )


def _write_png(
    runtime_root: Path,
    row: h6.H6V2Row,
    position: int,
    *,
    level: int,
) -> Path:
    destination = (
        runtime_root
        / evaluation.RGB_ROOT_RELATIVE_PATH_V1
        / row.rgb[position]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.new(
        "RGB",
        (224, 224),
        color=(level, (level + 17) % 256, (level + 31) % 256),
    ).save(destination, format="PNG")
    return destination


def _position_count(
    receipt: dict[str, int],
    *,
    role: str,
    kind: str,
    position: int,
    operation: str = "open_success",
) -> int:
    return receipt[
        f"{role}_{kind}_rgb_position_{position}_{operation}_count"
    ]


def test_descriptor_loader_opens_only_registered_factual_and_donor_positions(
    tmp_path: Path,
) -> None:
    train = _row(0, role="train")
    donor = _row(0, role="val")
    for position in range(4):
        _write_png(tmp_path, train, position, level=20 + position)
    for position in range(2):
        _write_png(tmp_path, donor, position, level=80 + position)

    with evaluation.SafeTemporalH6RGBLoaderV1(
        tmp_path,
        (train, donor),
    ) as loader:
        factual = loader.load_factual(train)
        donor_history = loader.load_donor_history(donor)
        with pytest.raises(PermissionError, match="outside V1 authority"):
            loader._decode_position(
                train,
                position=4,
                access_kind="factual",
            )
        with pytest.raises(PermissionError, match="outside V1 authority"):
            loader._decode_position(
                donor,
                position=2,
                access_kind="donor",
            )
        receipt = loader.access_snapshot()

    assert factual.shape == (4, 3, 112, 112)
    assert donor_history.shape == (2, 3, 112, 112)
    assert receipt["rgb_open_success_count"] == 6
    assert receipt["rgb_decode_success_count"] == 6
    assert receipt["denied_rgb_position_request_count"] == 2
    assert evaluation.forbidden_rgb_open_count_v1(receipt) == 0
    for position in range(7):
        assert _position_count(
            receipt,
            role="train",
            kind="factual",
            position=position,
        ) == (1 if position < 4 else 0)
        assert _position_count(
            receipt,
            role="val",
            kind="donor",
            position=position,
        ) == (1 if position < 2 else 0)
    assert all(
        _position_count(
            receipt,
            role="train",
            kind="donor",
            position=position,
        )
        == 0
        for position in range(7)
    )


def test_descriptor_loader_rejects_symlink_leaf_without_successful_open(
    tmp_path: Path,
) -> None:
    row = _row(0, role="train")
    _write_png(tmp_path, row, 0, level=40)
    source = _write_png(tmp_path, row, 1, level=41)
    destination = (
        tmp_path / evaluation.RGB_ROOT_RELATIVE_PATH_V1 / row.rgb[0]
    )
    destination.unlink()
    destination.symlink_to(source.name)

    with evaluation.SafeTemporalH6RGBLoaderV1(tmp_path, (row,)) as loader:
        with pytest.raises(
            evaluation.TemporalEvaluationContractError,
            match="no-follow temporal RGB leaf open failed",
        ):
            loader._decode_position(
                row,
                position=0,
                access_kind="factual",
            )
        receipt = loader.access_snapshot()

    assert receipt["rgb_open_attempt_count"] == 1
    assert receipt["rgb_open_success_count"] == 0
    assert receipt["rgb_decode_success_count"] == 0


def test_sequence_and_control_batches_never_tensorize_later_actions_or_rgb(
    tmp_path: Path,
) -> None:
    validation = tuple(
        _row(
            index,
            role="val",
            actions=(index % 9, (index + 1) % 9, 2, 8, 8, 8),
        )
        for index in range(evaluation.VALIDATION_ROW_COUNT_V1)
    )
    factual = validation[0]
    donor = validation[8]
    assert factual.family == donor.family
    assert factual.scene_id != donor.scene_id
    for position in range(4):
        _write_png(tmp_path, factual, position, level=30 + position)
    for position in range(2):
        _write_png(tmp_path, donor, position, level=90 + position)

    with evaluation.SafeTemporalH6RGBLoaderV1(
        tmp_path,
        (factual, donor),
    ) as loader:
        controls = evaluation.build_control_batch_v1(
            (factual,),
            validation_rows=validation,
            donor_indices=(donor.index,),
            loader=loader,
            device="cpu",
        )
        receipt = loader.access_snapshot()

    assert controls.factual.action_sequence.tolist() == [[0, 1, 2]]
    assert controls.wrong_history_action_sequence.tolist() == [[8, 0, 2]]
    assert controls.wrong_action_sequence.tolist() == [[0, 1, 3]]
    assert controls.wrong_action_eligible.tolist() == [True]
    assert receipt["rgb_open_success_count"] == 6
    assert evaluation.forbidden_rgb_open_count_v1(receipt) == 0
    assert all(
        _position_count(
            receipt,
            role="val",
            kind="factual",
            position=position,
        )
        == 0
        for position in range(4, 7)
    )
    assert all(
        _position_count(
            receipt,
            role="val",
            kind="donor",
            position=position,
        )
        == 0
        for position in range(2, 7)
    )


class _SyntheticLoader:
    def __init__(self) -> None:
        self._access = evaluation._initial_access_counters()

    def _record(
        self,
        row: h6.H6V2Row,
        kind: str,
        positions: Sequence[int],
    ) -> None:
        for position in positions:
            self._access["rgb_tensor_request_count"] += 1
            self._access["rgb_open_attempt_count"] += 1
            self._access["rgb_open_success_count"] += 1
            self._access["rgb_decode_success_count"] += 1
            for operation in (
                "request",
                "open_attempt",
                "open_success",
                "decode_success",
            ):
                self._access[
                    (
                        f"{row.role}_{kind}_rgb_position_{position}_"
                        f"{operation}_count"
                    )
                ] += 1

    @staticmethod
    def _frames(
        row: h6.H6V2Row,
        positions: Sequence[int],
    ) -> torch.Tensor:
        base = 0.015 * row.index + (
            0.2 if row.role == "val" else -0.2
        )
        return torch.stack(
            tuple(
                torch.full(
                    (3, 112, 112),
                    base + 0.08 * position,
                    dtype=torch.float32,
                )
                for position in positions
            )
        )

    def load_factual(self, row: h6.H6V2Row) -> torch.Tensor:
        self._record(row, "factual", range(4))
        return self._frames(row, range(4))

    def load_donor_history(self, row: h6.H6V2Row) -> torch.Tensor:
        self._record(row, "donor", range(2))
        return self._frames(row, range(2))

    def access_snapshot(self) -> dict[str, int]:
        return dict(self._access)

    def close(self) -> None:
        return None


def _runtime() -> evaluation.TemporalH6RuntimeV1:
    train = tuple(
        _row(
            index,
            role="train",
            actions=(index % 9, (index + 1) % 9, 0, 7, 8, 1),
        )
        for index in range(evaluation.TRAIN_ROW_COUNT_V1)
    )
    validation = tuple(
        _row(
            index,
            role="val",
            actions=(index % 9, (index + 1) % 9, 0, 7, 8, 1),
        )
        for index in range(evaluation.VALIDATION_ROW_COUNT_V1)
    )
    donors = tuple(
        (index + 8) % evaluation.VALIDATION_ROW_COUNT_V1
        for index in range(evaluation.VALIDATION_ROW_COUNT_V1)
    )
    return evaluation.TemporalH6RuntimeV1(
        train,
        validation,
        train_schedule_indices=tuple(
            range(evaluation.TRAIN_SCHEDULE_ROW_COUNT_V1)
        ),
        sentinel_indices=tuple(range(256)),
        donor_indices=donors,
        loader=_SyntheticLoader(),  # type: ignore[arg-type]
        device="cpu",
        panel_identity={"synthetic": True},
    )


def test_runtime_exposes_exact_schedule_microbatches_and_access_audit() -> None:
    runtime = _runtime()
    schedule_slice = runtime.training_schedule[10:20]
    batches = runtime.load_training_microbatches(schedule_slice, "cpu")

    assert runtime.train_rows[10].index == 10
    assert runtime.val_rows[10].role == "val"
    assert len(batches) == 5
    assert tuple(
        index for batch in batches for index in batch.row_indices
    ) == schedule_slice
    assert all(batch.context_rgb.shape == (2, 3, 3, 112, 112) for batch in batches)
    audit = runtime.access_audit()
    assert audit["passed"] is True
    assert audit["forbidden_rgb_open_count"] == 0
    assert audit["exact_request_attempt_success_decode_counts"] is True
    assert audit["counters"]["rgb_open_success_count"] == 40
    runtime.loader._access["denied_rgb_position_request_count"] += 1
    assert runtime.access_audit()["passed"] is False
    runtime.loader._access["denied_rgb_position_request_count"] -= 1
    runtime.loader._access["rgb_open_success_count"] -= 1
    mismatch = runtime.access_audit()
    assert mismatch["exact_request_attempt_success_decode_counts"] is False
    assert mismatch["passed"] is False
    with pytest.raises(
        evaluation.TemporalEvaluationContractError,
        match="exactly ten",
    ):
        runtime.load_training_microbatches(schedule_slice[:-1], "cpu")


class _SyntheticTemporalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.target_encoder = torch.nn.Linear(1, 1, bias=False)
        self.target_encoder.requires_grad_(False)
        self.register_buffer(
            "ema_update_count",
            torch.zeros((), dtype=torch.long),
        )
        self.eval()

    @staticmethod
    def _tokens(scalar: torch.Tensor, count: int) -> torch.Tensor:
        feature = torch.linspace(
            -1.0,
            1.0,
            evaluation.FEATURE_DIMENSION_V1,
            device=scalar.device,
            dtype=torch.float32,
        )
        token = torch.linspace(
            -0.3,
            0.3,
            count,
            device=scalar.device,
            dtype=torch.float32,
        )
        return torch.sin(
            scalar[:, None, None]
            + token[None, :, None]
            + feature[None, None, :]
        )

    def _prediction(
        self,
        context_rgb: torch.Tensor,
        actions: torch.Tensor,
    ) -> Any:
        frame_values = context_rgb.mean(dim=(2, 3, 4))
        action_values = actions.to(torch.float32) * 0.011
        state_scalars = torch.cumsum(
            frame_values + action_values,
            dim=1,
        )
        states = torch.stack(
            tuple(
                self._tokens(state_scalars[:, step], 256)
                for step in range(context_rgb.shape[1])
            ),
            dim=1,
        )
        prediction = self._tokens(state_scalars[:, -1] * 0.43 + 0.21, 64)
        return SimpleNamespace(
            raw_predicted_target_tokens=prediction,
            recurrent_memory=states[:, -1],
            recurrent_step_states=states,
        )

    def predict_future(
        self,
        context_rgb: torch.Tensor,
        actions: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        capture_intermediates: bool = False,
    ) -> Any:
        del target_indices, capture_intermediates
        return self._prediction(context_rgb, actions)

    def predict_current_only(
        self,
        current_rgb: torch.Tensor,
        action: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        capture_intermediates: bool = False,
    ) -> Any:
        del target_indices, capture_intermediates
        return self._prediction(
            current_rgb.unsqueeze(1),
            action.unsqueeze(1),
        )

    def encode_target(
        self,
        rgb: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> Any:
        del target_indices
        scalar = rgb.mean(dim=(1, 2, 3))
        return SimpleNamespace(raw_target_tokens=self._tokens(scalar, 64))


def test_streamed_controls_capture_temporal_and_causal_diagnostics() -> None:
    runtime = _runtime()
    model = _SyntheticTemporalModel()
    result = evaluation.stream_temporal_panel_v1(
        model,
        runtime,
        tuple(range(8)),
    )

    assert set(result.controls) == set(metrics.CONTROL_NAMES)
    assert result.recurrent_temporal_change > 0.0
    assert (
        result.causal_deltas["wrong_history"]["prediction"][
            "mean_squared_delta"
        ]
        > 0.0
    )
    assert (
        result.causal_deltas["wrong_action_non_hold"]["recurrent_state"][
            "mean_squared_delta"
        ]
        > 0.0
    )
    assert result.integrity["passed"] is True
    assert result.access_receipt["rgb_open_success_count"] == 48
    assert evaluation.forbidden_rgb_open_count_v1(result.access_receipt) == 0

    access_before_slice = runtime.access_snapshot()
    sliced = evaluation.slice_temporal_result_v1(
        result,
        runtime,
        tuple(reversed(range(8))),
    )
    assert runtime.access_snapshot() == access_before_slice
    assert sliced.row_indices == tuple(reversed(range(8)))
    assert sliced.source_row_indices == tuple(range(8))
    assert sliced.integrity["derived_from_source_panel"] is True
    assert sliced.integrity["additional_rgb_open_count"] == 0
    assert sliced.recurrent_health == result.recurrent_health
    assert sliced.prediction_health == result.prediction_health
    assert sliced.target_health == result.target_health


def test_predecessor_retention_adapter_preserves_underlying_update_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Any, ...]] = []

    class _Runtime:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    runtime = _Runtime()

    def _open(
        root: Path,
        *,
        device: Any,
        include_place: bool,
    ) -> tuple[_Runtime, dict[str, Any]]:
        calls.append(("open", root, device, include_place))
        return runtime, {"synthetic": True}

    def _evaluate(
        model: Any,
        selected_runtime: _Runtime,
        update: int,
        device: Any,
    ) -> dict[str, Any]:
        calls.append(("evaluate", model, selected_runtime, update, device))
        return {
            "schema": "synthetic_spatial_checkpoint_evaluation_v1",
            "update": update,
            "controls": {},
        }

    monkeypatch.setattr(
        evaluation.spatial_evaluation,
        "open_bound_runtime",
        _open,
    )
    monkeypatch.setattr(
        evaluation.spatial_evaluation,
        "evaluate_checkpoint",
        _evaluate,
    )
    model = object()
    result = evaluation.evaluate_predecessor_retention_panel_v1(
        model,
        tmp_path,
        200,
        "cpu",
    )

    assert calls == [
        ("open", tmp_path, "cpu", True),
        ("evaluate", model, runtime, 0, "cpu"),
    ]
    assert runtime.closed is True
    assert result["temporal_update"] == 200
    assert result["underlying_spatial_evaluator_update"] == 0
    assert result["evaluation"]["update"] == 0
    with pytest.raises(
        evaluation.TemporalEvaluationContractError,
        match="temporal 0/200/400",
    ):
        evaluation.evaluate_predecessor_retention_panel_v1(
            model,
            tmp_path,
            100,
            "cpu",
        )


def test_update_zero_full_and_sentinel_uses_one_temporal_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    row_count = evaluation.VALIDATION_ROW_COUNT_V1
    energies = {
        "real": torch.full((row_count,), 0.5, dtype=torch.float64),
        "persistence": torch.full((row_count,), 1.0, dtype=torch.float64),
        "current_only_reset": torch.full(
            (row_count,), 1.1, dtype=torch.float64
        ),
        "wrong_history": torch.full((row_count,), 1.2, dtype=torch.float64),
        "wrong_action": torch.full((row_count,), 1.3, dtype=torch.float64),
    }
    summary = metrics.ControlSummary(
        correct_macro_mean=0.5,
        control_macro_mean=1.0,
        primary_ratio=0.5,
        advantage_macro_mean=0.5,
        advantage_bootstrap_lower_95=0.5,
        positive_family_count=8,
        correct_by_scene={},
        control_by_scene={},
        advantage_by_scene={},
        advantage_by_family={},
    )
    health = metrics.RepresentationHealth(
        row_count=row_count,
        token_count=64,
        feature_dimension=192,
        effective_rank=8.0,
        cross_sample_variance=1.0,
        finite=True,
    )
    receipt = dict(evaluation._initial_access_counters())
    receipt.update(
        {
            "rgb_tensor_request_count": 6 * row_count,
            "rgb_open_attempt_count": 6 * row_count,
            "rgb_open_success_count": 6 * row_count,
            "rgb_decode_success_count": 6 * row_count,
        }
    )
    for position in range(4):
        for operation in (
            "request",
            "open_attempt",
            "open_success",
            "decode_success",
        ):
            receipt[
                f"val_factual_rgb_position_{position}_{operation}_count"
            ] = row_count
    for position in range(2):
        for operation in (
            "request",
            "open_attempt",
            "open_success",
            "decode_success",
        ):
            receipt[
                f"val_donor_rgb_position_{position}_{operation}_count"
            ] = row_count
    prediction = torch.zeros(1, 64, 192).expand(row_count, -1, -1)
    recurrent = torch.zeros(1, 256, 192).expand(row_count, -1, -1)
    diagnostic_vectors = {
        name: torch.full((row_count,), 0.1, dtype=torch.float64)
        for name in (
            "wrong_history_prediction_squared",
            "wrong_history_prediction_absolute",
            "wrong_history_state_squared",
            "wrong_history_state_absolute",
            "wrong_action_prediction_squared",
            "wrong_action_prediction_absolute",
            "wrong_action_state_squared",
            "wrong_action_state_absolute",
            "recurrent_temporal_change",
        )
    }
    source = evaluation.TemporalEvaluationResultV1(
        row_indices=tuple(range(row_count)),
        energies=evaluation.TemporalEnergyVectorsV1(
            real=energies["real"],
            persistence=energies["persistence"],
            current_only_reset=energies["current_only_reset"],
            wrong_history=energies["wrong_history"],
            wrong_action=energies["wrong_action"],
            wrong_action_eligible=torch.ones(row_count, dtype=torch.bool),
        ),
        controls={name: summary for name in metrics.CONTROL_NAMES},
        recurrent_health=metrics.RepresentationHealth(
            row_count=row_count,
            token_count=256,
            feature_dimension=192,
            effective_rank=8.0,
            cross_sample_variance=1.0,
            finite=True,
        ),
        prediction_health=health,
        target_health=health,
        recurrent_temporal_change=0.1,
        causal_deltas={},
        token_populations={
            "recurrent": recurrent,
            "prediction": prediction,
            "target": prediction,
        },
        diagnostic_vectors=diagnostic_vectors,
        source_row_indices=tuple(range(row_count)),
        access_receipt=receipt,
        panel_identity={},
        integrity={"finite_energies": True},
    )
    stream_calls: list[tuple[int, ...]] = []

    def _stream(
        model: Any,
        selected_runtime: evaluation.TemporalH6RuntimeV1,
        indices: Sequence[int],
    ) -> evaluation.TemporalEvaluationResultV1:
        del model
        assert selected_runtime is runtime
        stream_calls.append(tuple(indices))
        return source

    monkeypatch.setattr(evaluation, "stream_temporal_panel_v1", _stream)

    result = evaluation.evaluate_update_zero_full_and_sentinel_v1(
        _SyntheticTemporalModel(),
        runtime,
        "cpu",
    )

    assert result["single_temporal_rgb_and_model_pass"] is True
    assert stream_calls == [tuple(range(row_count))]
    assert result["full"]["row_count"] == row_count
    assert result["sentinel"]["row_count"] == 256
    assert result["sentinel"]["access_provenance"] == {
        "source_panel_row_count": row_count,
        "derived_from_source_panel": True,
        "additional_rgb_open_count": 0,
        "additional_model_call_count": 0,
    }
    assert result["full"]["access"]["rgb_open_success_count"] == 6 * row_count
    assert (
        result["sentinel"]["access"]["rgb_open_success_count"]
        == 6 * row_count
    )


def test_training_accounting_is_exact_at_cap() -> None:
    assert evaluation.expected_training_access_v1(400) == {
        "updates": 400,
        "sequence_presentations": 4_000,
        "logical_rgb_frame_presentations": 16_000,
        "online_encoder_frame_presentations": 12_000,
        "ema_target_encoder_frame_presentations": 4_000,
        "microbatch_graphs": 2_000,
        "backward_calls": 2_000,
        "global_gradient_clips": 400,
        "optimizer_steps": 400,
        "ema_steps": 400,
        "forbidden_rgb_open_count": 0,
    }
