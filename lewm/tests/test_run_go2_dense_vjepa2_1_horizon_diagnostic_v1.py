from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts import run_go2_dense_vjepa2_1_horizon_diagnostic_v1 as horizon


def _binding(path: Path) -> dict[str, object]:
    return horizon.predecessor.file_binding_v1(path)


def _write_json(path: Path, value: object) -> dict[str, object]:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return _binding(path)


def _index(count: int = 1_536, index_sha256: str = "0" * 64) -> object:
    contexts = []
    targets = []
    histories = []
    for state in range(128):
        offset = state * 12
        contexts.append((offset, offset + 1, offset + 2))
        targets.append(tuple(range(offset + 3, offset + 12)))
        histories.append((state % 9, (state + 1) % 9))
    return horizon.predecessor.ScreenIndexV1(
        state_ids=tuple(f"state-{state}" for state in range(128)),
        family_ids=tuple(f"family-{state % 8}" for state in range(128)),
        scene_ids=tuple(f"scene-{state // 8}" for state in range(128)),
        artifact_ids=tuple(f"artifact-{item}" for item in range(count)),
        context_indices=torch.tensor(contexts, dtype=torch.long),
        target_indices=torch.tensor(targets, dtype=torch.long),
        history_actions=torch.tensor(histories, dtype=torch.long),
        index_sha256=index_sha256,
    )


def _metrics(*, ratio: float, retrieval: float, margin: float) -> dict[str, float]:
    persistence = 0.1
    matched = persistence * ratio
    return {
        "matched_cosine_error": matched,
        "persistence_cosine_error": persistence,
        "error_to_persistence_ratio": ratio,
        "branch_retrieval_accuracy": retrieval,
        "cyclic_deranged_cosine_error": matched + margin,
        "action_intervention_margin": margin,
    }


class _TinyDense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.01))

    def forward(
        self,
        context: torch.Tensor,
        history: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        del history
        action = (candidate.float() + 1.0).view(-1, 1, 1)
        return context[:, -1] + self.scale * action


def test_config_is_the_single_fixed_horizon_change() -> None:
    config = horizon.horizon_config_v1()
    assert config["arms"] == ["dense_vjepa2_1"]
    assert config["seed"] == 2_026_080_301
    assert config["updates"] == 3_200
    assert config["trace_updates"] == [0, 800, 1_600, 2_400, 3_200]
    assert config["futility_update"] == 1_600
    assert config["futility_maximum_error_to_persistence_ratio"] == pytest.approx(
        (0.9164363539053353 + 0.8) / 2
    )
    assert config["futility_minimum_branch_retrieval_accuracy"] == pytest.approx(
        (0.2803819444444444 + 0.5) / 2
    )


def test_authority_rejects_changed_caller_binding(tmp_path: Path) -> None:
    authority = tmp_path / "authority.json"
    authority.write_text("{}\n")
    with pytest.raises(horizon.HorizonError, match="caller binding"):
        horizon._read_authority(  # noqa: SLF001
            authority,
            expected_sha256="0" * 64,
            expected_byte_count=3,
        )


def test_update_800_drift_witness_is_exact() -> None:
    horizon._require_update_800_witness(dict(horizon.UPDATE_800_WITNESS))  # noqa: SLF001
    changed = dict(horizon.UPDATE_800_WITNESS)
    changed["branch_retrieval_accuracy"] += 1.0e-15
    with pytest.raises(horizon.HorizonError, match="drift witness"):
        horizon._require_update_800_witness(changed)  # noqa: SLF001


def test_bound_inputs_reuse_cache_without_any_rgb_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(horizon.predecessor, "ARTIFACT_COUNT", 2)
    monkeypatch.setattr(horizon, "INDEX_SHA256", "3" * 64)
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"bound-cache")
    cache_binding = _binding(cache)
    receipt = {
        "binding": cache_binding,
        "eval_artifact_open_count": 0,
        "train_artifact_open_count": 2,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_binding = _write_json(receipt_path, receipt)
    frozen_sources = {
        label: {"path": f"/{label}", "sha256": "4" * 64, "byte_count": 1}
        for label in horizon.predecessor.SOURCE_LABELS
    }
    result = {
        "schema": horizon.predecessor.SCHEMA,
        "status": "COMPLETE_ENGINEERING_SCREEN",
        "collection_justified": False,
        "navigation_usefulness_established": False,
        "screen_index": {"index_sha256": "3" * 64, "eval_rgb_leaf_open_count": 0},
        "feature_caches": {"vjepa2_1": receipt},
        "authority": {"source_bindings": frozen_sources},
    }
    result_path = tmp_path / "result.json"
    result_binding = _write_json(result_path, result)
    terminal_path = tmp_path / "terminal.json"
    terminal = {
        "schema": horizon.predecessor.TERMINAL_SCHEMA,
        "status": "COMPLETE_COLLECTION_NOT_JUSTIFIED",
        "result_binding": result_binding,
        "collection_justified": False,
    }
    terminal_binding = _write_json(terminal_path, terminal)
    review_path = tmp_path / "review.json"
    review = {
        "schema": "lewm_go2_matched_branch_successor_screen_terminal_review_v1",
        "status": "PASS_WITH_MINOR_NON_DECISION_RELEVANT_REPORTING_DEVIATION",
        "result_binding": result_binding,
        "terminal_binding": terminal_binding,
        "protected_material_opened": False,
        "evaluation_rgb_opened": False,
        "findings": [],
    }
    review_binding = _write_json(review_path, review)
    authority = {
        "source_bindings": {
            f"predecessor_{label}": binding
            for label, binding in frozen_sources.items()
        },
        "predecessor_bindings": {
            "feature_cache": cache_binding,
            "feature_receipt": receipt_binding,
            "result": result_binding,
            "terminal": terminal_binding,
            "terminal_review": review_binding,
        }
    }
    index = _index(count=2, index_sha256="3" * 64)
    bundle = SimpleNamespace(access_audit={"rgb_leaf_open_count": 0})
    features = F.normalize(torch.ones(2, 256, 768), dim=-1).to(torch.float16)
    monkeypatch.setattr(
        horizon.predecessor.screen_data,
        "load_bound_posthoc_bundle_v1",
        lambda: bundle,
    )
    monkeypatch.setattr(horizon.predecessor, "build_screen_index_v1", lambda _bundle: index)
    monkeypatch.setattr(
        horizon.predecessor,
        "_load_feature_cache",
        lambda *_args, **_kwargs: features,
    )
    monkeypatch.setattr(
        horizon.predecessor,
        "read_bound_rgb_bytes_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("RGB opened")),
    )

    loaded, loaded_index, loaded_result = horizon.load_bound_inputs_v1(authority)
    assert loaded is features
    assert loaded_index is index
    assert loaded_result == result


def _short_config(*, updates: int = 4) -> dict[str, object]:
    config = horizon.horizon_config_v1()
    config.update(
        {
            "updates": updates,
            "trace_updates": [0, 2, 4],
            "futility_update": 2,
            "futility_maximum_error_to_persistence_ratio": 0.85,
            "futility_minimum_branch_retrieval_accuracy": 0.39,
        }
    )
    return config


def _script_evaluations(
    monkeypatch: pytest.MonkeyPatch, values: list[dict[str, float]]
) -> None:
    iterator = iter(values)
    monkeypatch.setattr(
        horizon.predecessor,
        "evaluate_arm_v1",
        lambda *_args, **_kwargs: dict(next(iterator)),
    )
    monkeypatch.setattr(
        horizon.predecessor,
        "_build_model",
        lambda *_args, **_kwargs: _TinyDense(),
    )


def test_training_stops_at_fixed_futility_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = _metrics(ratio=2.0, retrieval=0.11, margin=0.0)
    failed = _metrics(ratio=0.86, retrieval=0.38, margin=0.02)
    _script_evaluations(monkeypatch, [initial, failed, failed, failed])
    features = F.normalize(torch.randn(1_536, 256, 4), dim=-1).to(torch.float16)

    result = horizon.train_horizon_v1(
        features,
        _index(),
        config=_short_config(),
        device=torch.device("cpu"),
        output_root=tmp_path,
        require_update_800_witness=False,
    )
    assert result["completed_updates"] == 2
    assert result["futility_passed"] is False
    assert result["training_set_capacity_established"] is False
    assert set(result["checkpoint_bindings"]) == {"update_1600"}


def test_training_continues_to_capacity_only_after_futility_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = _metrics(ratio=2.0, retrieval=0.11, margin=0.0)
    midpoint = _metrics(ratio=0.84, retrieval=0.40, margin=0.03)
    final = _metrics(ratio=0.79, retrieval=0.51, margin=0.05)
    _script_evaluations(monkeypatch, [initial, midpoint, midpoint, final, final])
    features = F.normalize(torch.randn(1_536, 256, 4), dim=-1).to(torch.float16)

    result = horizon.train_horizon_v1(
        features,
        _index(),
        config=_short_config(updates=4),
        device=torch.device("cpu"),
        output_root=tmp_path,
        require_update_800_witness=False,
    )
    assert result["completed_updates"] == 4
    assert result["futility_passed"] is True
    assert result["training_set_capacity_established"] is True
    assert set(result["checkpoint_bindings"]) == {"update_1600", "update_3200"}


def test_new_output_gets_terminal_on_midrun_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "attempt"
    monkeypatch.setattr(
        horizon,
        "_read_authority",
        lambda *args, **kwargs: {"output_root": str(output)},
    )

    def fail(_authority: object) -> object:
        output.mkdir()
        raise horizon.HorizonError("expected failure")

    monkeypatch.setattr(horizon, "execute_v1", fail)
    with pytest.raises(horizon.HorizonError, match="expected failure"):
        horizon.main(
            [
                "--authority",
                str(tmp_path / "unused.json"),
                "--expected-authority-sha256",
                "0" * 64,
                "--expected-authority-byte-count",
                "1",
            ]
        )
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE"
    assert terminal["authorizes_rgb_access"] is False
