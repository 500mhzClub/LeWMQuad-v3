from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "joint_decision_frontier",
    ROOT / "scripts/run_joint_decision_level_safety_calibration_frontier_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_threshold_values_cover_strict_admission_boundaries():
    probability = np.asarray([.2, .2, .8])
    thresholds = MODULE.threshold_values(probability)
    assert thresholds[0] < 0 and thresholds[-1] > 1
    assert set(thresholds[1:-1]) == {.2, .8}
    assert not np.any(probability < .2)
    assert np.sum(probability < .8) == 2


def test_primary_selection_uses_frozen_lexicographic_order():
    fields = {name: np.ones(2) for name in MODULE.FRONTIER_FIELDS}
    fields.update({
        "contact_threshold": np.asarray([.4, .5]),
        "stuck_threshold": np.asarray([.4, .5]),
        "aggregate_unsafe_recall": np.asarray([.96, .96]),
        "aggregate_false_negative_rate": np.asarray([.04, .04]),
        "contact_recall": np.asarray([.91, .91]),
        "stuck_recall": np.asarray([.91, .91]),
        "states_only_unsafe_admitted": np.zeros(2),
        "selected_unsafe_count": np.zeros(2),
        "states_retaining_safe": np.asarray([6., 7.]),
        "safe_candidate_retention": np.asarray([.8, .4]),
        "mean_selected_route_progress_m": np.asarray([.3, .2]),
        "normalized_safe_progress_regret": np.asarray([.1, .1]),
        "false_abstentions": np.asarray([0., 1.]),
        "best_safe_top3": np.asarray([1., .8]),
    })
    assert MODULE.choose_primary(fields) == 1


def test_exact_binomial_and_zero_miss_sample_size():
    interval = MODULE.clopper_pearson(0, 58)
    assert interval[0] == 0.0 and 0.0 < interval[1] < .1
    assert MODULE.one_sided_zero_miss_sample_size() == 59


def test_pareto_mask_removes_dominated_points():
    mask = MODULE.pareto_mask(np.asarray([.95, .96, .97]), np.asarray([.5, .4, .6]))
    assert mask.tolist() == [False, False, True]


def test_frozen_ledger_binding_if_present():
    if not MODULE.LEDGER.is_file():
        return
    arrays, index = MODULE.load_ledger()
    assert len(arrays["branch_id"]) == 576
    assert index["array_content_digest"] == MODULE.EXPECTED_CONTENT_DIGEST


def test_completed_result_reproduces_ledger_and_persists_both_frontiers():
    path = MODULE.OUT / "result.json"
    if not path.is_file():
        return
    result = json.loads(path.read_text())
    assert result["ledger_reproduction"]["passed"]
    assert result["calibration_frontier"]["threshold_pairs"] > 0
    assert result["heldout_oracle_frontier"]["threshold_pairs"] > 0
    assert result["custody"]["model_inference"] is False
