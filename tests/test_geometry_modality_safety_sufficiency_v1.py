import json
from pathlib import Path

import numpy as np
import pytest

from scripts import materialize_geometry_modality_safety_sufficiency_v1 as geo


def test_nested_state_binding_is_exact():
    rows = geo.state_sources()
    assert len(rows) == 240
    assert len({row["state_id"] for row in rows}) == 240
    assert {name: sum(row["split"] == name for row in rows) for name in ("fit", "calibration", "heldout")} == {
        "fit": 192, "calibration": 24, "heldout": 24}


def test_oriented_box_raycast_hits_front_face():
    center = np.asarray([[2.0, 0.0, 0.5]])
    half = np.asarray([[0.5, 1.0, 0.5]])
    distance, index = geo.horizontal_hits(np.asarray([0.0, 0.0]), np.asarray([0.0]), center, half, np.asarray([0.0]), 10.0)
    assert distance[0] == pytest.approx(1.5)
    assert index[0] == 0


def test_geometry_contract_is_explicit_changed_modality():
    assert geo.DEPTH_NEAR_M > 0
    assert geo.DEPTH_FAR_M == geo.LIDAR_FAR_M == 10.0
    assert len(geo.LIDAR_VERTICAL_DEG) == 4
    assert geo.LIDAR_AZIMUTH_BINS == 180


def test_result_custody_and_gates_when_present():
    path = geo.OUT / "result.json"
    if not path.is_file():
        pytest.skip("scientific result not generated yet")
    result = json.loads(path.read_text())
    assert result["preserved_terminal"] == "FACTORISED_MICRO_SAFETY_DATA_SCALING_NO_SIGNAL"
    assert result["custody"]["jepa_predictor_opened_or_trained"] is False
    assert result["custody"]["new_state_or_candidate_identities"] == 0
    assert set(result["conditions"]) == {"DEPTH_ONLY", "LIDAR_ONLY", "DEPTH_PLUS_EMBODIED"}
    assert all(value["training"]["epochs"] == 60 for value in result["conditions"].values())
