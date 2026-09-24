from __future__ import annotations

import json
from pathlib import Path

import pytest

from lewm.planning.geometry_contract import (
    DEFAULT_GEOMETRY_CONTRACT,
    DEPLOYMENT_GEOMETRY_CONTRACT,
    load_geometry_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_checked_in_geometry_contract_loads_and_verifies_sources() -> None:
    contract = load_geometry_contract(
        REPO_ROOT / DEFAULT_GEOMETRY_CONTRACT,
        repository_root=REPO_ROOT,
    )

    assert contract.configuration_space.oracle_cell_size_m == pytest.approx(0.05)
    assert contract.configuration_space.body_inflation_radius_m == pytest.approx(0.20)
    assert contract.camera.nominal_xyz_body_m[0] == pytest.approx(0.326)
    assert contract.visibility_and_claim.claim_radius_m == pytest.approx(1.20)
    assert len(contract.sha256) == 64
    assert not contract.physical_promotion_ready


def test_geometry_contract_rejects_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("changed\n")
    payload = json.loads((REPO_ROOT / DEFAULT_GEOMETRY_CONTRACT).read_text())
    payload["source_artifacts"] = {
        "source": {"path": "source.txt", "sha256": "0" * 64}
    }
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="hash mismatch"):
        load_geometry_contract(path, repository_root=tmp_path)


def test_geometry_contract_rejects_standoff_outside_claim_radius(
    tmp_path: Path,
) -> None:
    payload = json.loads((REPO_ROOT / DEFAULT_GEOMETRY_CONTRACT).read_text())
    payload["visibility_and_claim"]["standoff_m"] = 2.0
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="standoff_m"):
        load_geometry_contract(path, repository_root=REPO_ROOT, verify_sources=False)


def test_deployment_geometry_binds_polygon_and_enclosing_disc() -> None:
    contract = load_geometry_contract(
        REPO_ROOT / DEPLOYMENT_GEOMETRY_CONTRACT,
        repository_root=REPO_ROOT,
    )

    assert contract.schema == "lewm_go2_generalization_geometry_v2"
    assert contract.configuration_space.body_inflation_radius_m == pytest.approx(0.47)
    assert contract.configuration_space.connectivity == 4
    assert contract.swept_footprint.directional_profile == "observed_max_plus_margin"
    assert contract.swept_footprint.maximum_vertex_radius_m == pytest.approx(
        0.4617711967569951
    )
    assert contract.swept_footprint.planning_disc_radius_m == pytest.approx(0.47)
    assert not contract.physical_promotion_ready
