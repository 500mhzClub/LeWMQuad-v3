from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts import (
    launch_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16 as launcher,
)
from scripts import (
    run_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16 as training,
)


def _camera_rows() -> dict[str, torch.Tensor]:
    return {
        "camera_origin": torch.zeros((4, 3), dtype=torch.float32),
        "camera_basis": torch.eye(3).expand(4, -1, -1).clone(),
        "ground": torch.zeros((4,), dtype=torch.float32),
        "pixel_hit": torch.zeros((4, 1), dtype=torch.bool),
        "pixel_distance": torch.ones((4, 1), dtype=torch.float32),
        "ground_in_frustum": torch.ones((4, 1), dtype=torch.bool),
        "ground_clear": torch.ones((4, 1), dtype=torch.bool),
    }


def _runtime(rows: list[list[float]]) -> SimpleNamespace:
    pairs = [
        {
            "current_endpoint_sha256": f"current-{index}",
            "next_endpoint_sha256": f"next-{index}",
            "relative_se2_current_frame": row,
        }
        for index, row in enumerate(rows)
    ]

    def base_batch(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {
            key: torch.zeros((4,), dtype=torch.float32)
            for key in training._base._v3.REQUIRED_BATCH_KEYS
        }

    return SimpleNamespace(
        v1_training=SimpleNamespace(build_microbatch_v1=base_batch),
        loader=object(),
        pairs={"train": pairs},
        labels={"train": object()},
        raw_inputs=object(),
        device=torch.device("cpu"),
        torch=torch,
        training_module=training,
    )


def test_batch_adapter_carries_exact_bound_motion_without_new_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def stack(*_args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        calls.append(tuple(_args[1]))
        assert kwargs["role"] == "train"
        return _camera_rows()

    monkeypatch.setattr(launcher._BASE_LAUNCHER, "_stack_camera_rows_v13", stack)
    rows = [[0.1, 0.0, 0.01], [0.0, 0.2, -0.02], [0.3, 0.1, 0.0], [0.0, 0.0, 0.0]]
    batch = launcher._build_one_microbatch_v16(
        runtime=_runtime(rows),
        indices=(0, 1, 2, 3),
        stage="synthetic",
    )
    assert tuple(batch) == training.REQUIRED_BATCH_KEYS
    assert torch.equal(
        batch[training.REALIZED_RELATIVE_SE2_KEY],
        torch.tensor(rows, dtype=torch.float32),
    )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "row",
    ([0.0, 0.0], [0.0, float("nan"), 0.0], "bad"),
)
def test_batch_adapter_rejects_malformed_motion(
    monkeypatch: pytest.MonkeyPatch,
    row: object,
) -> None:
    monkeypatch.setattr(
        launcher._BASE_LAUNCHER,
        "_stack_camera_rows_v13",
        lambda *_args, **_kwargs: _camera_rows(),
    )
    with pytest.raises(PermissionError, match=r"SE\(2\)"):
        launcher._build_one_microbatch_v16(
            runtime=_runtime([row, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            indices=(0, 1, 2, 3),
            stage="synthetic",
        )


def test_launcher_remains_denied_without_authority() -> None:
    assert launcher.main(()) == 4
