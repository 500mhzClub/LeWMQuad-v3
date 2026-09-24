from __future__ import annotations

from copy import deepcopy
import math

import pytest

from lewm.benchmarks import (
    go2_dinov2_physical_readout_calibration_integrity_replacement_v1 as subject,
)


def _document() -> dict[str, object]:
    return {
        "schema": subject.TASK_RELEVANCE_SCHEMA,
        "status": subject.TASK_RELEVANCE_PASS_STATUS,
        "thresholds": {
            "minimum_reference_candidate_rgb_ssim": 0.99,
            "required_paired_nearest_neighbour_retrieval_count": 32,
        },
        "measurements": {
            "pixels": {
                "minimum_reference_candidate_rgb_ssim": 0.999873849744854,
                "candidate_frame_count": 32,
            },
            "frozen_predecessor_descriptor_retrieval": {
                "paired_nearest_neighbour_retrieval_count": 32,
                "maximum_paired_descriptor_distance": 0.0014817728354341111,
            },
            "consumed_inventory_file_count": 141,
        },
        "immutable_exact_parity_failure": {
            "preserved": True,
            "status": "FAIL_VISUAL_DOMAIN_PARITY",
        },
        "bindings": {
            "source_panel": {
                "path": "/development/source-panel.json",
                "file_sha256": "a" * 64,
                "byte_count": 10,
            },
            "consumed_inventory": [
                {
                    "relative_path": "candidate_panel.json",
                    "path": "/development/candidate-panel.json",
                    "file_sha256": "b" * 64,
                    "byte_count": 20,
                }
            ],
        },
    }


def _set_path(document: dict[str, object], path: tuple[str, ...], value: object) -> None:
    panel = document
    for part in path[:-1]:
        child = panel[part]
        assert isinstance(child, dict)
        panel = child
    panel[path[-1]] = value


def test_canonical_exact_document_passes_and_returns_same_stored_object() -> None:
    stored = _document()
    recomputed = deepcopy(stored)

    admitted, evidence = subject.admit_task_relevance_result_v1(
        stored=stored, recomputed=recomputed
    )

    assert admitted is stored
    assert evidence == {
        "schema": subject.COMPATIBILITY_EVIDENCE_SCHEMA,
        "status": subject.COMPATIBILITY_PASS_STATUS,
        "stored_status": subject.TASK_RELEVANCE_PASS_STATUS,
        "recomputed_status": subject.TASK_RELEVANCE_PASS_STATUS,
        "allowed_differing_paths": [subject.SSIM_DOTTED_PATH],
        "differing_paths": [],
        "canonical_exact": True,
        "all_other_fields_canonical_exact": True,
        "stored_minimum_reference_candidate_rgb_ssim": 0.999873849744854,
        "recomputed_minimum_reference_candidate_rgb_ssim": 0.999873849744854,
        "absolute_difference": 0.0,
        "absolute_tolerance": 1.0e-12,
        "relative_tolerance": 0.0,
        "minimum_ssim_gate": 0.99,
        "both_values_finite": True,
        "both_values_at_or_above_gate": True,
        "returns_reviewed_stored_document": True,
    }


def test_singleton_ssim_difference_within_tolerance_passes_with_evidence() -> None:
    stored = _document()
    recomputed = deepcopy(stored)
    recomputed_ssim = 0.9998738497448542
    _set_path(recomputed, subject.SSIM_PATH, recomputed_ssim)

    admitted, evidence = subject.admit_task_relevance_result_v1(
        stored=stored, recomputed=recomputed
    )

    assert admitted is stored
    assert evidence["canonical_exact"] is False
    assert evidence["differing_paths"] == [subject.SSIM_DOTTED_PATH]
    assert evidence["all_other_fields_canonical_exact"] is True
    assert evidence["recomputed_minimum_reference_candidate_rgb_ssim"] == (
        recomputed_ssim
    )
    assert evidence["absolute_difference"] == abs(
        0.999873849744854 - recomputed_ssim
    )


@pytest.mark.parametrize(
    ("stored_ssim", "recomputed_ssim"),
    [
        (1.0, 1.0 - 1.1e-12),
        (0.989999999999, 0.9899999999995),
        (math.nan, 1.0),
        (1.0, math.inf),
        (True, 1.0),
    ],
)
def test_ssim_fails_above_tolerance_below_gate_nonfinite_or_nonnumeric(
    stored_ssim: object, recomputed_ssim: object
) -> None:
    stored = _document()
    recomputed = deepcopy(stored)
    _set_path(stored, subject.SSIM_PATH, stored_ssim)
    _set_path(recomputed, subject.SSIM_PATH, recomputed_ssim)

    with pytest.raises(subject.CompatibilityAdmissionError):
        subject.admit_task_relevance_result_v1(
            stored=stored, recomputed=recomputed
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (
            (
                "measurements",
                "frozen_predecessor_descriptor_retrieval",
                "maximum_paired_descriptor_distance",
            ),
            0.0014817737103522426,
        ),
        (("status",), "FAIL_TASK_RELEVANT_INPUT_ADEQUACY_DEVELOPMENT_ONLY"),
        (("schema",), "changed-schema"),
        (("thresholds", "minimum_reference_candidate_rgb_ssim"), 0.98),
        (("bindings", "source_panel", "file_sha256"), "c" * 64),
        (("bindings", "consumed_inventory",), []),
        (("measurements", "consumed_inventory_file_count"), 140),
        (("immutable_exact_parity_failure", "preserved"), False),
    ],
)
def test_any_second_changed_field_is_rejected(
    path: tuple[str, ...], value: object
) -> None:
    stored = _document()
    recomputed = deepcopy(stored)
    _set_path(recomputed, subject.SSIM_PATH, 0.9998738497448542)
    _set_path(recomputed, path, value)

    with pytest.raises(subject.CompatibilityAdmissionError):
        subject.admit_task_relevance_result_v1(
            stored=stored, recomputed=recomputed
        )


def test_key_or_list_length_drift_is_rejected() -> None:
    stored = _document()
    extra_key = deepcopy(stored)
    extra_key["unexpected"] = True
    with pytest.raises(subject.CompatibilityAdmissionError):
        subject.admit_task_relevance_result_v1(
            stored=stored, recomputed=extra_key
        )

    longer_inventory = deepcopy(stored)
    inventory = longer_inventory["bindings"]["consumed_inventory"]
    assert isinstance(inventory, list)
    inventory.append(deepcopy(inventory[0]))
    with pytest.raises(subject.CompatibilityAdmissionError):
        subject.admit_task_relevance_result_v1(
            stored=stored, recomputed=longer_inventory
        )


def test_inputs_are_not_mutated_on_success_or_failure() -> None:
    stored = _document()
    recomputed = deepcopy(stored)
    _set_path(recomputed, subject.SSIM_PATH, 0.9998738497448542)
    before_stored = deepcopy(stored)
    before_recomputed = deepcopy(recomputed)
    subject.admit_task_relevance_result_v1(
        stored=stored, recomputed=recomputed
    )
    assert stored == before_stored
    assert recomputed == before_recomputed

    recomputed["bindings"] = {}
    before_failure = deepcopy(recomputed)
    with pytest.raises(subject.CompatibilityAdmissionError):
        subject.admit_task_relevance_result_v1(
            stored=stored, recomputed=recomputed
        )
    assert stored == before_stored
    assert recomputed == before_failure
