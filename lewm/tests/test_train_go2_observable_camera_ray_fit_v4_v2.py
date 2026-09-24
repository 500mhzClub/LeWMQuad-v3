from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from scripts import train_go2_observable_camera_ray_fit_v4_v2 as trainer_v2


ROOT = Path(__file__).resolve().parents[2]
OLD_TRAINER = ROOT / "scripts/train_go2_observable_camera_ray_fit_v4.py"
NEW_TRAINER = ROOT / "scripts/train_go2_observable_camera_ray_fit_v4_v2.py"


def _top_level_functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_v2_preserves_every_computational_function_byte_equivalent_in_ast() -> None:
    old = _top_level_functions(OLD_TRAINER)
    new = _top_level_functions(NEW_TRAINER)
    assert set(new) == set(old)
    allowed_execution_shell_changes = {
        "decode_selected_rgb",
        "_require_captured_private_trainer",
        "_run_captured_exact",
        "parse_args",
        "main",
    }
    changed = {
        name
        for name in old
        if ast.dump(old[name], include_attributes=False)
        != ast.dump(new[name], include_attributes=False)
    }
    assert changed == allowed_execution_shell_changes
    for name in {
        "load_exact_inputs",
        "validate_exact_target_partition_v4",
        "compute_four_equal_v4_losses",
        "_deterministic_training_batches",
        "train_v4_fit",
        "evaluate_v4_fit",
        "validate_gpu0_r9700_runtime",
        "configure_determinism",
        "reserve_exact_attempt",
        "publish_reserved_exact_attempt",
    }:
        assert ast.dump(old[name], include_attributes=False) == ast.dump(
            new[name], include_attributes=False
        )


def test_v2_imports_only_successor_launcher_and_finalizer_for_execution() -> None:
    source = NEW_TRAINER.read_text()
    assert (
        "from scripts import launch_go2_observable_camera_ray_fit_v4_v2 as "
        "preauth_launcher"
    ) in source
    assert (
        "from scripts.finalize_go2_observable_camera_ray_fit_v4_ladder_v2 import"
    ) in source
    assert "scripts/launch_go2_observable_camera_ray_fit_v4_v2.py" in source
    assert "worker_successor_review_file_sha256" in source


def _args(*, seed: int, fit_size: int) -> argparse.Namespace:
    previous_size = {16: 5, 32: 16, 320: 32}.get(fit_size)
    return argparse.Namespace(
        seed=seed,
        fit_size=fit_size,
        previous_stage_gate=(
            None
            if previous_size is None
            else trainer_v2.CANONICAL_GATE_ROOT
            / f"stage_seed_{seed}_n{previous_size}.json"
        ),
        previous_stage_gate_sha256=None if previous_size is None else "1" * 64,
        seed_20260710_gate=(
            None
            if seed == 20260710
            else trainer_v2.CANONICAL_GATE_ROOT / "seed_20260710.json"
        ),
        seed_20260710_gate_sha256=None if seed == 20260710 else "2" * 64,
    )


@pytest.mark.parametrize("fit_size", [16, 32, 320])
def test_later_rungs_use_finalizer_v2_stage_validation(
    fit_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def stage(path: Path, digest: str, **kwargs: int) -> dict[str, object]:
        calls.append((path, digest, kwargs))
        return {"kind": "stage", **kwargs}

    monkeypatch.setattr(trainer_v2, "validate_canonical_stage_gate_for_execution", stage)
    result = trainer_v2.load_exact_prerequisite_gate_bindings(
        _args(seed=20260710, fit_size=fit_size)
    )
    assert result["previous_stage_gate"] == {
        "kind": "stage",
        "expected_seed": 20260710,
        "expected_next_fit_size": fit_size,
    }
    assert len(calls) == 1
    assert result["seed_20260710_gate"] is None


@pytest.mark.parametrize("fit_size", [5, 16, 32, 320])
def test_second_seed_uses_finalizer_v2_seed_validation(
    fit_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_calls = []
    stage_calls = []

    monkeypatch.setattr(
        trainer_v2,
        "validate_canonical_seed_gate_for_execution",
        lambda path, digest: seed_calls.append((path, digest)) or {"kind": "seed"},
    )
    monkeypatch.setattr(
        trainer_v2,
        "validate_canonical_stage_gate_for_execution",
        lambda path, digest, **kwargs: stage_calls.append((path, digest, kwargs))
        or {"kind": "stage"},
    )
    result = trainer_v2.load_exact_prerequisite_gate_bindings(
        _args(seed=20260711, fit_size=fit_size)
    )
    assert result["seed_20260710_gate"] == {"kind": "seed"}
    assert len(seed_calls) == 1
    assert len(stage_calls) == (0 if fit_size == 5 else 1)
