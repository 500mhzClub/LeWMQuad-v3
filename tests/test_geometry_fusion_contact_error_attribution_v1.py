import json

import numpy as np

from scripts import audit_geometry_fusion_contact_error_attribution_v1 as audit


def test_frozen_checkpoint_bindings():
    assert set(audit.EXPECTED) == {"DEPTH_ONLY", "LIDAR_ONLY", "DEPTH_PLUS_EMBODIED"}
    assert all(len(value) == 64 for value in audit.EXPECTED.values())


def test_proximity_and_family_contracts():
    assert audit.PROXIMITY_M == 0.35
    assert len(audit.FAMILIES) == 4


def test_result_is_read_only_and_preserves_terminal_when_present():
    path = audit.OUT / "result.json"
    if not path.is_file(): return
    result = json.loads(path.read_text())
    assert result["preserved_result"] == "GEOMETRY_MODALITY_POSITIVE_TENDENCY"
    assert result["claims_boundary"]["heldout_threshold_not_adopted"] is True
    assert all(value is False for value in result["custody"].values())
    assert result["primary_classification"] in {
        "CALIBRATION_PANEL_OR_METHOD_BOTTLENECK", "WIDE_AREA_GEOMETRY_REQUIRED",
        "CONTACT_REQUIREMENT_ONTOLOGY_REQUIRES_REVISION", "GEOMETRY_FUSION_SCORE_NO_GO"}


def test_sensor_visibility_does_not_invent_contact_point():
    row = {"future_depth": np.full((15, 48, 64), 1.0), "future_lidar": np.full((15, 4, 180), 1.0)}
    value = audit.sensor_visibility(row, 4)
    assert value["front_depth"]["contact_point_inside_horizontal_vertical_fov"] is None
    assert value["lidar"]["contact_point_inside_scan_coverage"] is None
