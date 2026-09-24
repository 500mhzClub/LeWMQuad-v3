from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import torch

from lewm.tests import (
    test_run_go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux
    as v2_fixtures,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux.py"
)


def _load_runner() -> Any:
    name = "_test_go2_swept_progress_survival_v3_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_v2_remains_one_and_v3_is_exactly_half_the_same_auxiliary() -> None:
    generator = torch.Generator().manual_seed(20260728)
    current = torch.randn((3, 3, 2, 3), generator=generator, dtype=torch.float64)
    next_logits = torch.randn(
        (3, 3, 2, 3), generator=generator, dtype=torch.float64
    )
    current_labels = torch.tensor(
        [
            [[0, 1, 2], [2, 1, 0]],
            [[2, 2, 2], [2, 2, 2]],
            [[0, 1, 0], [1, 0, 1]],
        ]
    )
    next_labels = current_labels.roll(1, dims=0)

    v2 = runner._v2.occupied_safety_aux_loss_v2(
        current, current_labels, next_logits, next_labels
    )
    v3 = runner.occupied_safety_aux_loss_v3(
        current, current_labels, next_logits, next_labels
    )

    assert runner._v2.OCCUPIED_SAFETY_AUX_COEFFICIENT == 1.0
    assert runner.OCCUPIED_SAFETY_AUX_COEFFICIENT == 0.5
    assert torch.equal(v3.current_per_row, v2.current_per_row)
    assert torch.equal(v3.next_per_row, v2.next_per_row)
    assert torch.allclose(v3.loss, 0.5 * v2.loss, rtol=1e-15, atol=0.0)


def test_matched_v2_v3_joint_updates_preserve_terms_l_identity_and_accounting() -> None:
    torch.manual_seed(71)
    model_v2 = v2_fixtures._TinyJointModel()
    torch.manual_seed(71)
    model_v3 = v2_fixtures._TinyJointModel()
    model_v3.load_state_dict(model_v2.state_dict())
    optimizer_v2 = runner.build_frozen_optimizer_v1(
        runner.partition_parameters_v1(model_v2)
    )
    optimizer_v3 = runner.build_frozen_optimizer_v1(
        runner.partition_parameters_v1(model_v3)
    )
    microbatches = v2_fixtures._microbatches()

    result_v2 = runner._v2.joint_training_update_v2(
        model_v2, optimizer_v2, microbatches
    )
    result_v3 = runner.joint_training_update_v3(
        model_v3, optimizer_v3, microbatches
    )

    expected_accounting = runner.JointTrainingAccountingV1(
        updates=1,
        presentations=16,
        microbatch_graphs=4,
        backward_calls=4,
        optimizer_steps=1,
        ema_steps=1,
        predictor_forwards=4,
        predictor_objectives=4,
    )
    assert result_v2.accounting == result_v3.accounting == expected_accounting
    assert set(result_v2.mean_losses) == set(result_v3.mean_losses) == {
        "S",
        "P",
        "U",
        "R",
        "O",
        "L",
    }
    for name in ("S", "P", "U", "R"):
        assert result_v3.mean_losses[name] == result_v2.mean_losses[name]
    assert math.isclose(
        result_v3.mean_losses["O"],
        0.5 * result_v2.mean_losses["O"],
        rel_tol=2e-7,
        abs_tol=0.0,
    )
    for result in (result_v2, result_v3):
        assert math.isclose(
            result.mean_losses["L"],
            sum(result.mean_losses[name] for name in ("S", "P", "U", "R", "O")),
            rel_tol=2e-6,
            abs_tol=2e-6,
        )


def test_shared_fixed_driver_keeps_exact_cap_without_mutable_monkeypatching() -> None:
    labels = runner.freeze_role_labels_v1(
        v2_fixtures._label_rows(), role="train", np=__import__("numpy")
    )
    pair = {
        "dataset_role": "train",
        "content_sha256": "a" * 64,
        "current_endpoint_sha256": "b" * 64,
        "next_endpoint_sha256": "c" * 64,
        "scene_id": "scene-a",
        "family": "small_enclosed_maze",
        "primitive": "arc_left",
    }
    built = 0

    def build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal built
        del args, kwargs
        built += 1
        return {}

    def update(
        model: Any,
        optimizer: Any,
        microbatches: Any,
        *,
        accounting: runner.JointTrainingAccountingV1,
    ) -> runner.JointUpdateResultV3:
        del model, optimizer
        assert len(microbatches) == 4
        losses = {name: 1.0 for name in ("S", "P", "U", "R")}
        losses.update(O=0.5, L=4.5)
        return runner.JointUpdateResultV3(
            accounting=runner._v2._v1._base._advanced_accounting(accounting),
            mean_losses=losses,
            gradient_l2={
                name: 1.0 for name in ("encoder", "lift_semantic", "predictor")
            },
            representation_clip_pre_l2=1.0,
            predictor_clip_pre_l2=1.0,
            ranking_active_microbatches=2,
            ranking_eligible_pairs=3,
            survival_supervised_decisions=4,
        )

    accounting, trace, diagnostics = runner._v2._run_fixed_training_core_v2(
        object(),
        object(),
        object(),
        (pair,),
        labels,
        (0,) * runner.MAXIMUM_PRESENTATIONS,
        object(),
        microbatch_builder=build,
        joint_update=update,
    )
    assert accounting == runner.JointTrainingAccountingV1(
        updates=1_000,
        presentations=16_000,
        microbatch_graphs=4_000,
        backward_calls=4_000,
        optimizer_steps=1_000,
        ema_steps=1_000,
        predictor_forwards=4_000,
        predictor_objectives=4_000,
    )
    assert built == 4_000
    assert len(trace) == 1_000
    assert trace[0]["losses"] == {
        "S": 1.0,
        "P": 1.0,
        "U": 1.0,
        "R": 1.0,
        "O": 0.5,
        "L": 4.5,
    }
    assert diagnostics["ranking_active_microbatch_count"] == 2_000


def test_v3_reuses_all_noncoefficient_v2_training_identities() -> None:
    assert runner.ACTION_ORDER is runner._v2.ACTION_ORDER
    assert runner.REQUIRED_BATCH_KEYS is runner._v2.REQUIRED_BATCH_KEYS
    assert runner.build_microbatch_v1 is runner._v2.build_microbatch_v1
    assert runner.build_frozen_optimizer_v1 is runner._v2.build_frozen_optimizer_v1
    assert runner.partition_parameters_v1 is runner._v2.partition_parameters_v1
    assert runner.score_full_control_v1 is runner._v2.score_full_control_v1
    assert runner.MAXIMUM_UPDATES == runner._v2.MAXIMUM_UPDATES == 1_000
    assert runner.MAXIMUM_PRESENTATIONS == runner._v2.MAXIMUM_PRESENTATIONS == 16_000
    assert (
        runner.OCCUPIED_SAFETY_AUX_NORMALIZATION
        == runner._v2.OCCUPIED_SAFETY_AUX_NORMALIZATION
        == math.log(2.0)
    )
