from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_go2_recurrent_h4_joint_jepa_v1.py"


def _runner():
    name = "_test_run_go2_recurrent_h4_joint_jepa_v1"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_presentation_cap_and_frozen_observations() -> None:
    runner = _runner()
    assert runner.UPDATES == 1_000
    assert runner.BATCH_SIZE == 16
    assert runner.PRESENTATIONS == 16_000
    assert runner.OBSERVATION_UPDATES == (0, 250, 500, 750, 1_000)


def test_jepa_error_sums_feature_distance() -> None:
    runner = _runner()
    predicted = torch.tensor([1.0, 0.0]).view(1, 1, 1, 2).expand(1, 4, 1, 2)
    target = torch.tensor([0.0, 1.0]).view(1, 1, 1, 2).expand(1, 4, 1, 2)
    result = runner._normalized_error(
        predicted,
        target,
        SimpleNamespace(torch=torch),
    )
    torch.testing.assert_close(result, torch.full((1, 4), 2.0))


def test_noncollapse_sampling_preserves_spatial_features() -> None:
    runner = _runner()
    tokens = torch.arange(2 * 4 * 256 * 3).reshape(2, 4, 256, 3)
    sampled = runner._pool_features(tokens, time_index=3)
    assert sampled.shape == (32, 3)
    torch.testing.assert_close(sampled[0], tokens[0, 3, 0])
    torch.testing.assert_close(sampled[16], tokens[1, 3, 0])
