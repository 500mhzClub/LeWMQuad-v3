from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from lewm.benchmarks import go2_swept_progress_survival_labels_v1 as labels
from scripts import diagnose_go2_swept_progress_selection_v1 as selection_census


_DIGEST = "a" * 64


@dataclass(frozen=True)
class _Report:
    feasible: bool

    @property
    def inside_world_bounds(self) -> bool:
        return True

    @property
    def colliding_object_ids(self) -> tuple[str, ...]:
        return () if self.feasible else ("wall",)


class _Checker:
    def __init__(self, _manifest: object, _footprint: object) -> None:
        pass

    def interpolated_sweep(
        self,
        start: object,
        end: object,
        *,
        maximum_corner_step_m: float,
        maximum_yaw_step_rad: float,
    ) -> tuple[tuple[float, object], ...]:
        assert maximum_corner_step_m == labels.v4.MAXIMUM_CORNER_STEP_M
        assert maximum_yaw_step_rad == labels.v4.MAXIMUM_YAW_STEP_RAD
        return ((0.0, start), (1.0, end))

    def pose_feasibility(self, pose: object) -> _Report:
        return _Report(feasible=pose.x_m < 0.35)


def _commands() -> dict[str, tuple[tuple[float, float, float], ...]]:
    zero = ((0.0, 0.0, 0.0),) * 5
    result = {action: zero for action in labels.ACTION_ORDER}
    result["forward_fast"] = ((0.2, 0.0, 0.0),) * 5
    return result


def _state_rows(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, object], ...]:
    monkeypatch.setattr(
        labels.v4, "ManifestDirectionalFootprintFeasibility", _Checker
    )
    pair = {
        "dataset_role": "train",
        "scene_id": "scene_a",
        "family": "large_enclosed_maze",
        "global_row": 7,
        "content_sha256": _DIGEST,
        "current_endpoint_sha256": _DIGEST,
        "primitive": "forward_slow",
        "frames_jsonl_sha256": _DIGEST,
        "scene_manifest_sha256": _DIGEST,
    }
    endpoint = {
        "endpoint_identity_sha256": _DIGEST,
        "content_sha256": _DIGEST,
    }
    return labels.label_state_v1(
        pair=pair,
        endpoint=endpoint,
        source_pose_world=labels.v4.Pose2D(0.0, 0.0, 0.0),
        source_line_number=3,
        scene_manifest=SimpleNamespace(
            scene_id="scene_a", family="large_enclosed_maze"
        ),
        footprint=object(),
        commands_by_action=_commands(),
        source_bindings={"source_frames_jsonl": {"file_sha256": _DIGEST}},
        role_state_index=0,
    )


def _synthetic_state(
    *,
    role: str,
    state_index: int,
    family: str,
    prefixes: dict[str, int],
) -> list[dict[str, object]]:
    non_hold_prefixes = [prefixes[action] for action in labels.NON_HOLD_ACTIONS]
    best = max(non_hold_prefixes)
    informative = best > 0 and len(set(non_hold_prefixes)) >= 2
    rows: list[dict[str, object]] = []
    for action_index, action in enumerate(labels.ACTION_ORDER):
        prefix = prefixes[action]
        participates = action != "hold" and any(
            prefix != other for other in non_hold_prefixes
        )
        rows.append(
            labels.v4.with_content_sha256(
                {
                    "schema": labels.ROW_SCHEMA,
                    "dataset_role": role,
                    "role_state_index": state_index,
                    "global_row": state_index,
                    "pair_content_sha256": _DIGEST,
                    "current_endpoint_sha256": _DIGEST,
                    "scene_id": f"{role}_scene",
                    "family": family,
                    "action_index": action_index,
                    "action": action,
                    "nominal_post_action_se2_current_frame": [0.0, 0.0, 0.0],
                    "immediate_primitive_feasible": prefix > 0,
                    "swept_progress_prefix_length": prefix,
                    "best_non_hold_prefix_length": best,
                    "informative_state": informative,
                    "action_participates_in_unequal_prefix": participates,
                    "provenance": {"synthetic": True},
                }
            )
        )
    return rows


def test_reuses_committed_target_and_labels_hold_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        labels.swept_progress_prefix_v1
        is selection_census.swept_progress_prefix_v1
    )
    rows = _state_rows(monkeypatch)

    assert [row["action"] for row in rows] == list(labels.ACTION_ORDER)
    hold = rows[labels.ACTION_ORDER.index("hold")]
    assert hold["nominal_post_action_se2_current_frame"] == [0.0, 0.0, 0.0]
    assert hold["immediate_primitive_feasible"] is True
    assert hold["swept_progress_prefix_length"] == 3
    assert hold["action_participates_in_unequal_prefix"] is False
    assert all(row["informative_state"] is True for row in rows)
    assert all(labels._valid_hash(row) for row in rows)


def test_hold_must_be_exact_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _commands()
    commands["hold"] = ((0.01, 0.0, 0.0),) + commands["hold"][1:]
    with pytest.raises(labels.SweptProgressLabelError, match="exact zero primitive"):
        labels._require_zero_hold(commands)


def test_role_validation_rejects_content_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = list(_state_rows(monkeypatch))
    rows[0] = {**rows[0], "swept_progress_prefix_length": 15}
    with pytest.raises(labels.SweptProgressLabelError, match="content hash"):
        labels._state_groups(rows, role="train", frozen=False)


def test_synthetic_preflight_counts_roles_families_and_schedule() -> None:
    different = {
        action: (3 if index == 0 else 1)
        for index, action in enumerate(labels.ACTION_ORDER)
    }
    different["hold"] = 2
    equal = {action: 1 for action in labels.ACTION_ORDER}
    rows_by_role = {
        "train": [
            *_synthetic_state(
                role="train",
                state_index=0,
                family="large_enclosed_maze",
                prefixes=different,
            ),
            *_synthetic_state(
                role="train",
                state_index=1,
                family="large_enclosed_maze",
                prefixes=equal,
            ),
        ],
        "probability_calibration": _synthetic_state(
            role="probability_calibration",
            state_index=0,
            family="medium_enclosed_maze",
            prefixes=different,
        ),
        "checkpoint_selection": _synthetic_state(
            role="checkpoint_selection",
            state_index=0,
            family="small_enclosed_maze",
            prefixes=different,
        ),
    }

    checks = labels.summarize_preflight_v1(
        rows_by_role, (0, 1, 0), enforce_frozen_gates=False
    )

    assert checks["informative_state_counts"] == {
        "train": 1,
        "probability_calibration": 1,
        "checkpoint_selection": 1,
    }
    assert checks["selection_family_informative_counts"]["small_enclosed_maze"] == 1
    schedule = checks["frozen_schedule"]
    assert schedule["informative_presentation_count"] == 2
    assert all(
        value == 2
        for value in schedule[
            "unequal_prefix_participation_presentations_by_action"
        ].values()
    )


def test_frozen_gate_requires_each_non_hold_action_participation() -> None:
    checks = {
        "state_counts": {
            role: labels.v4.ROLE_STATE_COUNTS[role] for role in labels.ROLE_ORDER
        },
        "action_row_counts": {
            role: labels.v4.ROLE_STATE_COUNTS[role] * len(labels.ACTION_ORDER)
            for role in labels.ROLE_ORDER
        },
        "informative_state_counts": dict(labels.INFORMATIVE_FLOORS),
        "selection_family_informative_counts": {
            family: labels.SELECTION_FAMILY_FLOOR
            for family in labels.v4.REGISTERED_SELECTION_FAMILIES
        },
        "frozen_schedule": {
            "presentation_count": labels.SCHEDULE_PRESENTATION_COUNT,
            "presentation_indices_sha256": labels.v4.SCHEDULE_PREFIX_SHA256,
            "informative_presentation_count": labels.SCHEDULE_INFORMATIVE_FLOOR,
            "unequal_prefix_participation_presentations_by_action": {
                action: labels.SCHEDULE_ACTION_PARTICIPATION_FLOOR
                for action in labels.NON_HOLD_ACTIONS
            },
        },
    }
    labels.enforce_preflight_gates_v1(checks)
    checks["frozen_schedule"][
        "unequal_prefix_participation_presentations_by_action"
    ]["yaw_right"] = 31
    with pytest.raises(labels.SweptProgressLabelError, match="preflight failed"):
        labels.enforce_preflight_gates_v1(checks)


def test_access_ledger_has_only_exact_model_free_opens() -> None:
    ledger = labels.expected_access_ledger_v1()
    assert ledger["scene_join_calls_started"] == 88
    assert ledger["schedule_opens"] == 1
    for protected in (
        "rgb_opens",
        "checkpoint_opens",
        "runtime_output_opens",
        "g2_opens",
        "navigation_opens",
        "heldout_opens",
        "sealed_opens",
        "production_opens",
    ):
        assert ledger[protected] == 0


def test_gate_failure_receipt_preserves_computed_counts() -> None:
    error = labels.SweptProgressLabelError("gate failed")
    error.checks = {"informative_state_counts": {"train": 511}}

    receipt = labels._failure_receipt(
        phase="enforce_scientific_gates",
        error=error,
        ledger=labels.v4.new_access_ledger_v1(),
    )

    assert receipt["preflight"] == error.checks
    assert labels._valid_hash(receipt)
