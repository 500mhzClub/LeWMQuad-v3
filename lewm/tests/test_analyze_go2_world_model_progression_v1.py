from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts import analyze_go2_world_model_progression_v1 as analyzer


def _interval(lower: float = 0.12) -> dict[str, float | int]:
    return {
        "point": 0.16,
        "lower_95": lower,
        "upper_95": 0.20,
        "requested_resamples": 2_000,
        "valid_resamples": 2_000,
        "seed": 20260802,
        "scene_clusters": 150,
    }


def _metrics(value: float) -> dict:
    return {
        "row_count": 2_048,
        "factual_energy_mean": 0.20 - value,
        "persistence_energy_mean": 0.22,
        "persistence_advantage_mean": 0.02 + value,
        "hardest_wrong_action_margin_mean": -0.008 + value,
        "hardest_wrong_action_margin_q05": -0.010 + value,
        "nine_way_action_balanced_accuracy": 0.15 + value,
        "candidate_energy_spread_mean": 0.01,
        "per_action": {
            str(action): {
                "rows": 228 if action < 8 else 224,
                "hardest_margin_mean": -0.008 + value,
                "persistence_advantage_mean": 0.02 + value,
                "recall": 0.15 + value,
            }
            for action in range(9)
        },
    }


def _payload(root: Path) -> tuple[dict, Path]:
    result_path = root / "result.json"
    result_path.write_text("{}\n")
    values = {
        "masked_plain": 0.000,
        "masked_delta": 0.004,
        "full_plain": 0.002,
        "full_delta": 0.008,
    }
    seeds = {}
    for seed in analyzer.SEEDS:
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir()
        for arm in analyzer.ARMS:
            (seed_root / f"{arm}_update_000700.pt").write_bytes(
                f"{seed}/{arm}".encode("ascii")
            )
        seeds[str(seed)] = {
            "build": {
                "core_initial_sha256": {arm: "a" * 64 for arm in analyzer.ARMS},
                "dynamic_registered_parity_max_abs_error": 0.0,
                "decoder_frozen_sha256": "b" * 64,
            },
            "decoder_anchor_balanced_accuracy": {
                "masked": _interval(),
                "full": _interval(),
            },
            "terminal": {arm: _metrics(value) for arm, value in values.items()},
            "terminal_decoder_sha256": "b" * 64,
        }
    pack_role = lambda role: {
        "manifest_sha256": analyzer.EXPECTED_PACK["manifest_sha256"],
        "row_identity_sha256": analyzer.EXPECTED_PACK[role]["row_identity_sha256"],
        "frames": {"sha256": analyzer.EXPECTED_PACK[role]["frames_sha256"]},
        "actions": {"sha256": analyzer.EXPECTED_PACK[role]["actions_sha256"]},
        "metadata": {"sha256": analyzer.EXPECTED_PACK[role]["metadata_sha256"]},
    }
    payload = {
        "schema": analyzer.RUNNER_SCHEMA,
        "status": analyzer.RUNNER_STATUS,
        "citable_as_scientific_evidence": False,
        "protected_material_opened": False,
        "configuration": copy.deepcopy(analyzer.EXPECTED_CONFIGURATION),
        "source_bindings": copy.deepcopy(list(analyzer.EXPECTED_SOURCE_BINDINGS)),
        "inputs": {
            "predecessor": copy.deepcopy(analyzer.EXPECTED_PREDECESSOR),
            "train": pack_role("train"),
            "val": pack_role("val"),
        },
        "seed_results": seeds,
    }
    return payload, result_path


def test_analyzer_computes_fixed_factorial_contrasts_and_snapshot_hashes(tmp_path) -> None:
    payload, result_path = _payload(tmp_path)

    result = analyzer.analyze(payload, result_path=result_path)

    margin = result["contrasts"]["hardest_wrong_action_margin_mean"]
    per_seed = margin["per_seed"][str(analyzer.SEEDS[0])]
    assert per_seed["factorial_effects"] == {
        "delta_main": pytest.approx(0.005),
        "spatial_main": pytest.approx(0.003),
        "interaction": pytest.approx(0.002),
    }
    assert margin["across_seed"]["delta_main"]["positive_seed_count"] == 3
    assert result["proxy_routing"]["decision"] == "DELTA_PROXY_MEANINGFUL"
    assert len(result["terminal_snapshot_bindings"]) == 3
    assert all(
        len(result["terminal_snapshot_bindings"][str(seed)]) == 4
        for seed in analyzer.SEEDS
    )


def test_analyzer_rejects_anchor_whose_lower_bound_does_not_clear_chance(tmp_path) -> None:
    payload, result_path = _payload(tmp_path)
    payload["seed_results"][str(analyzer.SEEDS[1])][
        "decoder_anchor_balanced_accuracy"
    ]["masked"] = _interval(analyzer.CHANCE)

    with pytest.raises(analyzer.AnalysisError, match="did not clear chance"):
        analyzer.analyze(payload, result_path=result_path)


def test_analyzer_rejects_nonfixed_training_configuration(tmp_path) -> None:
    payload, result_path = _payload(tmp_path)
    payload["configuration"]["updates"] = 701

    with pytest.raises(analyzer.AnalysisError, match="configuration changed"):
        analyzer.analyze(payload, result_path=result_path)


def test_analyzer_rejects_missing_terminal_snapshot(tmp_path) -> None:
    payload, result_path = _payload(tmp_path)
    target = (
        tmp_path
        / f"seed_{analyzer.SEEDS[2]}"
        / "full_delta_update_000700.pt"
    )
    target.unlink()

    with pytest.raises(analyzer.AnalysisError, match="terminal snapshot"):
        analyzer.analyze(payload, result_path=result_path)


def test_analyzer_rejects_protected_path_before_access(tmp_path) -> None:
    protected = tmp_path / "sealed" / "result.json"

    with pytest.raises(analyzer.AnalysisError, match="protected path"):
        analyzer._reject_protected_path(protected)
