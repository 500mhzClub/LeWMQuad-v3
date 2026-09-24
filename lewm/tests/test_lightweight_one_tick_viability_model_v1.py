import numpy as np

from lewm.safety import lightweight_one_tick_viability_model_v1 as subject


def test_fixture_passes():
    assert subject.fixture_payload()["pass"]


def test_model_contract_and_cap():
    model = subject.LightweightOneTickViabilityModel()
    assert subject.parameter_count(model) < 750_000


def test_auc_and_ap_are_exact_for_perfect_order():
    labels = np.asarray([0, 1, 0, 1], bool)
    probability = np.asarray([0.1, 0.8, 0.2, 0.9])
    assert subject.auc(labels, probability) == 1.0
    assert subject.average_precision(labels, probability) == 1.0


def test_threshold_tie_is_rejected():
    rows = [{"action_index": index, "contact": False, "n_safe": 2,
             "h3_progress_m": 1.0 - index / 100 if index < 12 else None,
             "h3_heading_improvement_rad": 0.0 if index < 12 else None,
             "decision_progress_m": 1.0 - index / 100 if index < 12 else 0.0}
            for index in range(14)]
    admitted = np.asarray([0.5 < 0.5] * 14)
    assert subject.select_candidate(rows, admitted, np.zeros(14)) is None
