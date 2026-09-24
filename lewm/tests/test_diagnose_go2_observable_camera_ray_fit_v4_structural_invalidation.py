from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as gate
from scripts import (
    diagnose_go2_observable_camera_ray_fit_v4_structural_invalidation as diagnostic,
)


ROOT = Path(__file__).resolve().parents[2]


def _artifact_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((diagnostic.N5_ATTEMPT_PATH / name).read_bytes()).hexdigest()
        for name in diagnostic.N5_ARTIFACTS
    }


def test_cpu_diagnostic_reproduces_terminal_structural_failure() -> None:
    before = _artifact_hashes()
    value = diagnostic.build_diagnostic()
    after = _artifact_hashes()
    assert before == after == {
        name: expected["file_sha256"]
        for name, expected in diagnostic.N5_ARTIFACTS.items()
    }
    assert value["schema"] == diagnostic.SCHEMA
    assert value["status"] == "terminal_prepublication_structural_invalidation"
    assert value["cpu_only"] is True
    assert value["writes_artifacts"] is False

    findings = value["structural_findings"]
    matched = findings["matched_rgb"]
    assert matched["stored_total"] == 0.27940133213996887
    assert matched["computed_quarter_component_sum"] == 0.27940132907242515
    assert matched["stored_minus_computed_delta"] == 3.067543719037502e-09
    assert matched["absolute_tolerance"] == 1e-9
    assert matched["within_tolerance"] is False
    assert matched["frozen_validator"] == {
        "passed": False,
        "exception": "ValueError: V4 matched evaluation losses are inconsistent",
    }

    wrong = findings["wrong_rgb_with_target_calibration"]
    assert wrong["stored_total"] == 2.0213493436574934
    assert wrong["computed_quarter_component_sum"] == 2.021349344518967
    assert wrong["stored_minus_computed_delta"] == -8.614735591550016e-10
    assert wrong["absolute_tolerance"] == 1e-9
    assert wrong["within_tolerance"] is True
    assert wrong["frozen_validator"] == {"passed": True, "exception": None}
    assert findings["immutable_full_result_validation"] == {
        "passed": False,
        "exception": "ValueError: V4 matched evaluation losses are inconsistent",
    }

    counterfactual = findings["counterfactual_single_field_repair"]
    assert counterfactual["counterfactual_only"] is True
    assert counterfactual["mutation_authorized"] is False
    assert counterfactual["changed_semantic_paths"] == [
        "$.evaluation.matched_rgb.losses.total"
    ]
    assert counterfactual["full_frozen_validator"] == {
        "passed": True,
        "exception": None,
    }
    assert counterfactual["other_failing_invariants_after_counterfactual"] == []
    assert all(value["authority"][key] is False for key in value["authority"])
    core = dict(value)
    declared = core.pop("content_sha256")
    assert gate.canonical_json_sha256(core) == declared


def test_secondary_hashes_are_exact_and_script_has_no_accelerator_imports(
    capsys: object,
) -> None:
    value = diagnostic.build_diagnostic()
    assert value["secondary_sorted_vector_findings"]["stable_differences"] == {
        "matched_rgb": {
            "immutable_result_sorted_values_sha256": (
                "a8ec842a10766b724b9ee4835c0e6866ce4b2323ccb7c33757c9f9d04ac20326"
            ),
            "stable_read_only_recomputed_sorted_values_sha256": (
                "6014597b1c286c42e5e7caa0643a98141b9545809c325a40763c82caf99d9f08"
            ),
        },
        "wrong_rgb_with_target_calibration": {
            "immutable_result_sorted_values_sha256": (
                "6ec4af60dd8f684bf6ef74339e4e439e7235d1a5fdf632aca0b79e77e95e1c86"
            ),
            "stable_read_only_recomputed_sorted_values_sha256": (
                "1e161762ff2158664cee260ff65b903864e14cce3c7bc09a405336140eee5ec8"
            ),
        },
    }

    source = (ROOT / "scripts/diagnose_go2_observable_camera_ray_fit_v4_structural_invalidation.py").read_text()
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imported_roots
    assert "numpy" not in imported_roots
    assert "cuda" not in source.lower()
    assert "hip_visible_devices" not in source.lower()

    assert diagnostic.main() == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    parsed = json.loads(output)
    assert parsed == value
