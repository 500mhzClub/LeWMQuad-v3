from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn

from lewm.datasets.go2_memory_role_place_triplets_v1 import (
    PlaceTripletRow,
    RGBReference,
    RGBTriplet,
)
from scripts import evaluate_go2_rgb_memory_role_factorized_joint_jepa_v1 as evaluation


def _sha(value: int) -> str:
    return f"{value:064x}"


def _place_rows() -> tuple[PlaceTripletRow, ...]:
    rows = []
    for family in evaluation.FAMILIES_V1:
        scene = f"{family}_selection"
        for _key in range(evaluation.PLACE_FAMILY_ROW_COUNTS_V1[family]):
            index = len(rows)
            rows.append(
                PlaceTripletRow(
                    index=index,
                    role=evaluation.CHECKPOINT_SELECTION_ROLE_V1,
                    family=family,
                    scene_id=scene,
                    anchor=RGBReference(_sha(1_000 + index), f"a/{index}", _sha(1)),
                    positive=RGBReference(_sha(2_000 + index), f"p/{index}", _sha(2)),
                    negative=RGBReference(_sha(3_000 + index), f"n/{index}", _sha(3)),
                    content_sha256=_sha(4_000 + index),
                )
            )
    return tuple(rows)


_KEY_IMAGES = []
for _key in range(64):
    _value = torch.zeros(3, 112, 112, dtype=torch.float32)
    _value[0].reshape(-1)[_key] = 1.0
    _KEY_IMAGES.append(_value)
_KEY_IMAGES = tuple(_KEY_IMAGES)


def _load_triplet(row: PlaceTripletRow) -> RGBTriplet:
    family_index = evaluation.FAMILIES_V1.index(row.family)
    start = sum(
        evaluation.PLACE_FAMILY_ROW_COUNTS_V1[family]
        for family in evaluation.FAMILIES_V1[:family_index]
    )
    key = row.index - start
    quota = evaluation.PLACE_FAMILY_ROW_COUNTS_V1[row.family]
    negative_key = (key + 1) % 64 if quota == 64 else (quota + key) % 64
    return RGBTriplet(
        anchor_rgb=_KEY_IMAGES[key],
        positive_rgb=_KEY_IMAGES[key],
        negative_rgb=_KEY_IMAGES[negative_key],
    )


class _TinyLocalPredictor(nn.Module):
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        selected = action.argmax(dim=1).to(dtype=torch.float32)
        return state + (selected + 1.0)[:, None, None, None] * 0.3


class _TinyRoleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.target_stub = nn.Linear(1, 1, bias=False)
        self.target_stub.requires_grad_(False)
        self.place_predictor = nn.Identity()
        self.local_predictor = _TinyLocalPredictor()
        self.model_input_shapes: list[tuple[int, ...]] = []
        self.train(True)

    def train(self, mode: bool = True) -> _TinyRoleModel:
        super().train(mode)
        self.target_stub.eval()
        return self

    def target_modules(self) -> tuple[nn.Module, ...]:
        return (self.target_stub,)

    def _roles(self, rgb: torch.Tensor, *, detached: bool) -> SimpleNamespace:
        self.model_input_shapes.append(tuple(rgb.shape))
        place = rgb[:, 0].reshape(rgb.shape[0], -1)[:, :64]
        place = torch.nn.functional.normalize(place, dim=1, eps=1.0e-6)
        scalar = rgb[:, 1, 0, 0] + self.anchor * 0.0
        local = scalar[:, None, None, None].expand(-1, 32, 16, 16)
        if detached:
            place = place.detach()
            local = local.detach()
        return SimpleNamespace(place_key=place, local_control=local)

    def encode_online_roles(self, rgb: torch.Tensor) -> SimpleNamespace:
        return self._roles(rgb, detached=False)

    def encode_target_roles(self, rgb: torch.Tensor) -> SimpleNamespace:
        return self._roles(rgb, detached=True)


def _local_rows() -> tuple[evaluation.LocalSelectionRowV1, ...]:
    rows = []
    actions = (0, 1, 2, evaluation.HOLD_ACTION_INDEX_V1)
    for family_index, family in enumerate(evaluation.FAMILIES_V1):
        for action in actions:
            rows.append(
                evaluation.LocalSelectionRowV1(
                    index=len(rows),
                    role=evaluation.CHECKPOINT_SELECTION_ROLE_V1,
                    family=family,
                    scene_id=f"{family}_local_selection_{family_index}",
                    action=action,
                )
            )
    return tuple(rows)


def _load_local(row: evaluation.LocalSelectionRowV1) -> dict[str, object]:
    current = torch.zeros(3, 112, 112, dtype=torch.float32)
    current[1, 0, 0] = 0.2 + 0.01 * (row.index % 4)
    next_rgb = current.clone()
    next_rgb[1, 0, 0] += 0.3 * (row.action + 1)
    return {
        evaluation.LOCAL_CURRENT_RGB_KEY_V1: current,
        evaluation.LOCAL_NEXT_RGB_KEY_V1: next_rgb,
        evaluation.LOCAL_ACTION_KEY_V1: row.action,
    }


def test_place_evaluation_is_bounded_retrieval_aware_and_noncollapsed() -> None:
    model = _TinyRoleModel()
    result = evaluation.evaluate_place_checkpoint_selection_v1(
        model,
        _place_rows(),
        load_triplet=_load_triplet,
        device="cpu",
        training_scene_ids={"train_scene"},
        update=0,
    )

    assert result["passed"] is True
    assert result["row_count"] == 320
    assert result["energy"]["negative_minus_positive_mean"] == pytest.approx(1.0)
    assert result["energy"]["negative_minus_positive_bootstrap_lower_95"] > 0.0
    assert result["energy"]["positive_family_count"] == 8
    assert result["retrieval"]["minimum_candidate_count"] == 60
    assert result["retrieval"]["maximum_candidate_count"] == 64
    assert result["retrieval"]["recall_at_5"] == 1.0
    assert result["retrieval"]["chance_multiple"] > 3.0
    assert result["retrieval"]["scene_count_above_chance"] == 8
    assert result["noncollapse"]["target_place_key_effective_rank"] > 50.0
    assert result["access"] == {
        "triplet_loader_call_count": 320,
        "rgb_tensor_count": 960,
        "privileged_label_fields_passed_to_model": 0,
        "retained_place_key_rows": 1280,
        "retained_non_scalar_local_rows": 0,
    }
    assert len(result["per_row"]) == 320
    assert all(set(row) == {
        "index",
        "family",
        "scene_id",
        "positive_energy",
        "negative_energy",
        "negative_minus_positive",
        "pessimistic_retrieval_rank",
    } for row in result["per_row"])
    assert model.training is True
    assert all(type(shape) is tuple for shape in model.model_input_shapes)


def test_local_evaluation_uses_cyclic_action_and_non_hold_persistence() -> None:
    model = _TinyRoleModel()
    result = evaluation.evaluate_local_checkpoint_selection_v1(
        model,
        _local_rows(),
        load_pair=_load_local,
        device="cpu",
        training_scene_ids={"train_scene"},
        update=400,
    )

    assert result["passed"] is True
    assert result["row_count"] == 32
    assert result["action"]["wrong_minus_correct_mean"] > 0.05
    assert result["action"]["wrong_minus_correct_bootstrap_lower_95"] > 0.0
    assert result["action"]["positive_family_count"] == 8
    assert result["persistence"]["non_hold_row_count"] == 24
    assert result["persistence"]["correct_to_no_update_energy_ratio"] == pytest.approx(0.0)
    assert result["persistence"]["no_update_minus_correct_bootstrap_lower_95"] > 0.0
    assert result["access"]["retained_non_scalar_local_rows"] == 0
    assert all(
        row["wrong_action"] == (row["action"] + 1) % 9
        for row in result["per_row"]
    )
    assert sum(row["non_hold_persistence_gate_row"] for row in result["per_row"]) == 24


def test_selection_boundary_rejects_role_overlap_and_bad_loader() -> None:
    model = _TinyRoleModel()
    rows = list(_place_rows())
    rows[0] = PlaceTripletRow(
        index=0,
        role="train",
        family=rows[0].family,
        scene_id=rows[0].scene_id,
        anchor=rows[0].anchor,
        positive=rows[0].positive,
        negative=rows[0].negative,
        content_sha256=rows[0].content_sha256,
    )
    with pytest.raises(evaluation.MemoryRoleEvaluationContractError, match="role"):
        evaluation.evaluate_place_checkpoint_selection_v1(
            model,
            rows,
            load_triplet=_load_triplet,
            device="cpu",
            training_scene_ids={"train_scene"},
            update=0,
        )

    local = _local_rows()
    with pytest.raises(evaluation.MemoryRoleEvaluationContractError, match="overlap"):
        evaluation.evaluate_local_checkpoint_selection_v1(
            model,
            local,
            load_pair=_load_local,
            device="cpu",
            training_scene_ids={local[0].scene_id},
            update=0,
        )

    def wrong_action(row: evaluation.LocalSelectionRowV1) -> dict[str, object]:
        result = _load_local(row)
        result[evaluation.LOCAL_ACTION_KEY_V1] = (row.action + 1) % 9
        return result

    with pytest.raises(evaluation.MemoryRoleEvaluationContractError, match="action changed"):
        evaluation.evaluate_local_checkpoint_selection_v1(
            model,
            local,
            load_pair=wrong_action,
            device="cpu",
            training_scene_ids={"train_scene"},
            update=0,
        )


def test_terminal_gate_keeps_depth_tail_prior_and_memory_controls_diagnostic() -> None:
    model = _TinyRoleModel()
    place0 = evaluation.evaluate_place_checkpoint_selection_v1(
        model,
        _place_rows(),
        load_triplet=_load_triplet,
        device="cpu",
        training_scene_ids={"train_scene"},
        update=0,
    )
    place400 = copy.deepcopy(place0)
    place400["update"] = 400
    local400 = evaluation.evaluate_local_checkpoint_selection_v1(
        model,
        _local_rows(),
        load_pair=_load_local,
        device="cpu",
        training_scene_ids={"train_scene"},
        update=400,
    )
    controls = {
        control: {
            check: True for check in evaluation.PHYSICAL_CONTROL_CHECK_NAMES_V1
        }
        for control in evaluation.PHYSICAL_CONTROL_NAMES_V1
    }
    physical = {
        "margin_count": 189,
        "passed_margin_count": 73,
        "rough_motion": {"depth_p95_m": 999.0},
    }
    gate = evaluation.evaluate_terminal_gate_v1(
        update0_place=place0,
        update400_place=place400,
        update400_local=local400,
        physical_summary=physical,
        controls=controls,
        integrity_pass=True,
        diagnostics={"tail_energy": 1.0e9, "mean_prior_energy": -1.0e9},
    )

    assert gate["passed"] is True
    assert gate["diagnostic_only"]["rough_depth_is_a_gate"] is False
    assert gate["diagnostic_only"]["tail_metric_is_a_gate"] is False
    assert gate["diagnostic_only"]["prior_metric_is_a_gate"] is False
    assert gate["memory_reset_reverse_shuffle_required"] is False
    assert gate["navigation_evaluation_required"] is False
    assert len(gate["causal_control_checks"]) == 12

    physical["passed_margin_count"] = 72
    failed = evaluation.evaluate_terminal_gate_v1(
        update0_place=place0,
        update400_place=place400,
        update400_local=local400,
        physical_summary=physical,
        controls=controls,
        integrity_pass=True,
    )
    assert failed["passed"] is False
    assert failed["action"] == "FAIL_TERMINAL_NO_MEMORY_INTEGRATION"

    with pytest.raises(evaluation.MemoryRoleEvaluationContractError, match="nonfinite"):
        evaluation.evaluate_terminal_gate_v1(
            update0_place=place0,
            update400_place=place400,
            update400_local=local400,
            physical_summary={**physical, "passed_margin_count": 73},
            controls=controls,
            integrity_pass=True,
            diagnostics={"tail_energy": float("nan")},
        )
