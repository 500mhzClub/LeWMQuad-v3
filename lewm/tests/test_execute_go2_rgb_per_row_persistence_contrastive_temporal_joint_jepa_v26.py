from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26.py"
)
TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26.py"
)
FROZEN_V25_EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25.py"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _executor(name: str) -> Any:
    return _load(EXECUTOR_PATH, name)


def _training(name: str) -> Any:
    return _load(TRAINING_PATH, name)


def _full_schema_sentinel_batches(executor: Any) -> tuple[dict[str, Any], ...]:
    prior_key = executor._v25._v24._v23.ACTION_PRIOR_M_KEY
    negative_key = (
        executor._v25._v24._v23._base.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
    )
    batches = []
    for _ in range(executor.MICROBATCHES_PER_UPDATE):
        batch = {
            name: torch.zeros(1, dtype=torch.float32)
            for name in executor.TRAINING_REQUIRED_BATCH_KEYS_V26
        }
        batch[negative_key] = torch.tensor((1, 2, 3, 0), dtype=torch.int64)
        batch[prior_key] = torch.zeros(9, dtype=torch.float32)
        batches.append(batch)
    return tuple(batches)


def test_denied_v26_identity_and_exact_v25_evidence_bindings(capsys) -> None:
    executor = _executor("_v26_executor_identity")
    assert executor.main([]) == 4
    denied = capsys.readouterr().out
    assert "DENIED_SOURCE_ONLY" in denied
    assert "joint_jepa_v26" in denied
    assert hashlib.sha256(FROZEN_V25_EXECUTOR_PATH.read_bytes()).hexdigest() == (
        executor.V25_EXECUTOR_FILE_SHA256
    )
    assert FROZEN_V25_EXECUTOR_PATH.stat().st_size == executor.V25_EXECUTOR_BYTE_COUNT
    assert executor.PREREGISTRATION_COMMIT == (
        "0c277fd7350931a7993d5affc2d1d4633ffed916"
    )
    assert executor.OUTPUT_ROOT_RELATIVE_PATH.endswith(
        "joint_jepa_v26/attempt_v1"
    )
    assert executor.BOUND_PARENT_SOURCES[executor.PREREGISTRATION_PATH] == (
        executor.PREREGISTRATION_FILE_SHA256,
        executor.PREREGISTRATION_BYTE_COUNT,
    )
    assert executor.BOUND_PARENT_SOURCES[
        executor.V25_TERMINAL_FAILURE_RESULT_PATH
    ] == (
        executor.V25_TERMINAL_FAILURE_RESULT_FILE_SHA256,
        executor.V25_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
    )
    receipt = executor.private_adapter_receipt_v26()
    assert receipt["scientific_behavior_delegated_exactly_to_v25"] is True
    assert receipt["v25_recovery_writer_delegated_exactly"] is True
    assert receipt["retry_authorized"] is False
    assert receipt["resume_authorized"] is False


def test_training_api_accepts_only_the_six_preregistered_schema_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor("_v26_executor_training_api")
    training = _training("_v26_training_for_executor_api")
    receipt = executor.validate_training_api_v26(training)
    assert receipt["science_identical_to_v25"] is True
    assert receipt["schema_integrity_alias_correction_only"] is True
    full = training._v25._validate_microbatches_v25
    projected_v21 = training._v25._v24._validate_microbatches_v21
    projected_v23 = training._v25._v24._validate_microbatches_v23
    assert training._validate_microbatches_v13 is full
    assert training._validate_microbatches_v21 is projected_v21
    assert training._validate_microbatches_v23 is projected_v23
    assert training._validate_microbatches_v24 is full
    assert training._validate_microbatches_v25 is full
    assert training._validate_microbatches_v26 is full

    # Recreate both incorrect V25 aliases.  The V26 runtime preflight must
    # reject this before an experiment can reach a training update.
    monkeypatch.setattr(training, "_validate_microbatches_v21", full)
    monkeypatch.setattr(training, "_validate_microbatches_v23", full)
    with pytest.raises(RuntimeError, match="schema compatibility aliases"):
        executor.validate_training_api_v26(training)


def test_actual_v25_v24_v23_v21_executor_projection_uses_v21_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor("_v26_executor_projection")
    training = _training("_v26_training_for_projection")
    batches = _full_schema_sentinel_batches(executor)
    v24_executor = executor._v25._v24
    v23_executor = v24_executor._v23
    v21_executor = v23_executor._base
    assert executor.validate_microbatches_for_engine_v26 is (
        executor._v25.validate_microbatches_for_engine_v25
    )
    assert v24_executor._original_validate_microbatches is (
        v23_executor.validate_microbatches_for_engine_v23
    )
    assert v23_executor._original_validate_microbatches is (
        v21_executor.validate_microbatches_for_engine_v21
    )

    seen: dict[str, tuple[tuple[str, ...], ...]] = {}
    actual_v23_boundary = v24_executor._original_validate_microbatches
    actual_v21_boundary = v23_executor._original_validate_microbatches

    def record_v23(runtime: Any, model: Any, values: Any) -> None:
        seen["v23"] = tuple(tuple(batch) for batch in values)
        actual_v23_boundary(runtime, model, values)

    def record_v21(runtime: Any, model: Any, values: Any) -> None:
        seen["v21"] = tuple(tuple(batch) for batch in values)
        actual_v21_boundary(runtime, model, values)

    monkeypatch.setattr(v24_executor, "_original_validate_microbatches", record_v23)
    monkeypatch.setattr(v23_executor, "_original_validate_microbatches", record_v21)
    monkeypatch.setattr(
        executor._engine,
        "_validate_batch_query_identity_v13",
        lambda model, batch: None,
    )
    deepest_training = training._v25._v24._v23._v21._base
    monkeypatch.setattr(
        deepest_training,
        "_validate_microbatches_v13",
        lambda torch_module, values: seen.update(
            runtime_v13=tuple(tuple(batch) for batch in values)
        ),
    )
    runtime = SimpleNamespace(torch=torch, training_module=training)
    executor.validate_microbatches_for_engine_v26(runtime, object(), batches)

    full_keys = tuple(executor.TRAINING_REQUIRED_BATCH_KEYS_V26)
    v21_keys = tuple(v23_executor.TRAINING_REQUIRED_BATCH_KEYS_V21)
    v13_keys = tuple(training._v25._v24._v23._v21.INHERITED_REQUIRED_BATCH_KEYS_V21)
    assert seen["v23"] == (full_keys,) * 4
    assert seen["v21"] == (v21_keys,) * 4
    assert seen["runtime_v13"] == (v13_keys,) * 4
    assert training._validate_microbatches_v21 is (
        training._v25._v24._validate_microbatches_v21
    )
    training._validate_microbatches_v23(torch, batches)

    # The consumed V25 attempt failed exactly here: its full-schema validator
    # was exposed under the projected V21 name.  The same substitution must
    # still reproduce the pre-update schema rejection in this regression.
    monkeypatch.setattr(
        training,
        "_validate_microbatches_v21",
        training._v25._validate_microbatches_v25,
    )
    with pytest.raises(ValueError, match="V25 microbatch schema changed"):
        executor.validate_microbatches_for_engine_v26(runtime, object(), batches)

    reordered = tuple(dict(reversed(tuple(batch.items()))) for batch in batches)
    with pytest.raises(PermissionError, match="V25 engine microbatch schema changed"):
        executor.validate_microbatches_for_engine_v26(runtime, object(), reordered)
