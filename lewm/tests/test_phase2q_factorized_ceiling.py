from __future__ import annotations

from lewm.benchmarks.phase2q_factorized_ceiling import (
    phase2q_factorized_affordance_ceiling_audit,
)
from scripts.check_jepa_phase2m_primitive_affordance_gate import check_gate


def _labels(
    *,
    progress: float,
    p05: float = 0.2,
    minimum: float = 0.3,
    unsafe_fraction: float = 0.0,
    enters_unsafe: bool = False,
    ends_unsafe: bool = False,
    recoverable: bool = True,
    heading_error: float = 0.0,
) -> dict:
    return {
        "target_progress_m": progress,
        "p05_swept_configuration_clearance_m": p05,
        "minimum_swept_configuration_clearance_m": minimum,
        "unsafe_sample_fraction": unsafe_fraction,
        "enters_grid_unsafe": enters_unsafe,
        "ends_grid_unsafe": ends_unsafe,
        "target_recoverable": recoverable,
        "target_heading_error_rad": heading_error,
    }


def _row(
    *,
    scene_id: str,
    source_index: int,
    sequence: tuple[str, str],
    progress: float,
    p05: float = 0.2,
    unsafe_fraction: float = 0.0,
    enters_unsafe: bool = False,
) -> dict:
    return {
        "scene_id": scene_id,
        "family": "family",
        "source_index": source_index,
        "start_frame": f"{scene_id}_{source_index}.png",
        "primitive_sequence": list(sequence),
        "active_blocks": [[1.0], [2.0]],
        "future_frames": ["future_0.png", "future_1.png"],
        "consequence_labels": _labels(
            progress=progress,
            p05=p05,
            unsafe_fraction=unsafe_fraction,
            enters_unsafe=enters_unsafe,
        ),
    }


def _two_source_rows(prefix: str) -> list[dict]:
    return [
        _row(
            scene_id=f"{prefix}_scene_a",
            source_index=1,
            sequence=("forward_slow", "hold"),
            progress=0.3,
            p05=0.3,
        ),
        _row(
            scene_id=f"{prefix}_scene_a",
            source_index=1,
            sequence=("backward", "hold"),
            progress=-0.3,
            p05=-0.2,
            unsafe_fraction=1.0,
            enters_unsafe=True,
        ),
        _row(
            scene_id=f"{prefix}_scene_b",
            source_index=2,
            sequence=("forward_slow", "hold"),
            progress=-0.3,
            p05=-0.2,
            unsafe_fraction=1.0,
            enters_unsafe=True,
        ),
        _row(
            scene_id=f"{prefix}_scene_b",
            source_index=2,
            sequence=("backward", "hold"),
            progress=0.3,
            p05=0.3,
        ),
    ]


def test_phase2q_true_factor_ceiling_passes_primitive_gate() -> None:
    report = phase2q_factorized_affordance_ceiling_audit(
        train_rows=_two_source_rows("train"),
        validation_rows=_two_source_rows("validation"),
        seed=20260615,
    )
    gate = check_gate(
        report,
        min_primitive_match_rate=0.50,
        max_selected_primitive_excess=0.20,
    )

    assert report["schema"] == "jepa_phase2q_factorized_affordance_ceiling_audit_v0"
    assert report["validation_data"]["true_factor_selection_diagnostic"][
        "selection_summary"
    ]["primitive_match_rate"] == 1.0
    assert gate["passed"]


def test_phase2q_true_factor_diagnostic_records_selection_rule() -> None:
    report = phase2q_factorized_affordance_ceiling_audit(
        train_rows=_two_source_rows("train"),
        validation_rows=_two_source_rows("validation"),
        seed=20260615,
        selection_kwargs={"heading_weight": 0.0},
    )
    diagnostic = report["validation_data"]["true_factor_selection_diagnostic"]

    assert diagnostic["selection_rule"]["heading_weight"] == 0.0
    assert diagnostic["selection_summary"]["mean_target_utility_regret"] == 0.0
