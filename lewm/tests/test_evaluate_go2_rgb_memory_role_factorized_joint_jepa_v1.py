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


_FROZEN_FAMILY_CANDIDATE_COUNTS = (63, 64, 59, 59, 64, 64, 32, 48)


def _place_rows(
    candidate_counts: tuple[int, ...] = _FROZEN_FAMILY_CANDIDATE_COUNTS,
) -> tuple[PlaceTripletRow, ...]:
    assert len(candidate_counts) == len(evaluation.FAMILIES_V1)
    rows = []
    for family_index, (family, candidate_count) in enumerate(
        zip(evaluation.FAMILIES_V1, candidate_counts, strict=True)
    ):
        scene = f"{family}_selection"
        quota = evaluation.PLACE_FAMILY_ROW_COUNTS_V1[family]
        extra_negative_count = candidate_count - quota
        assert 0 <= extra_negative_count <= quota
        identity_base = 100_000 * (family_index + 1)
        positive_ids = [_sha(identity_base + key) for key in range(quota)]
        for key in range(quota):
            index = len(rows)
            negative_identity = (
                _sha(identity_base + 10_000 + key % extra_negative_count)
                if extra_negative_count
                else positive_ids[(key + 2) % quota]
            )
            rows.append(
                PlaceTripletRow(
                    index=index,
                    role=evaluation.CHECKPOINT_SELECTION_ROLE_V1,
                    family=family,
                    scene_id=scene,
                    anchor=RGBReference(
                        positive_ids[(key + 1) % quota], f"a/{index}", _sha(1)
                    ),
                    positive=RGBReference(
                        positive_ids[key], f"p/{index}", _sha(2)
                    ),
                    negative=RGBReference(
                        negative_identity, f"n/{index}", _sha(3)
                    ),
                    content_sha256=_sha(1_000_000 + index),
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
    assert result["retrieval"]["minimum_candidate_count"] == 32
    assert result["retrieval"]["maximum_candidate_count"] == 64
    assert result["retrieval"]["by_scene"][
        "small_enclosed_maze_selection"
    ]["exact_chance_recall_at_5"] == pytest.approx(5.0 / 32.0)
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


def test_place_candidate_preflight_matches_frozen_metadata_topology() -> None:
    result = evaluation.preflight_place_retrieval_candidate_counts_v1(_place_rows())

    assert result["ordered_family_candidate_counts"] == list(
        _FROZEN_FAMILY_CANDIDATE_COUNTS
    )
    assert result["candidate_counts_by_family"] == dict(
        zip(
            evaluation.FAMILIES_V1,
            _FROZEN_FAMILY_CANDIDATE_COUNTS,
            strict=True,
        )
    )
    assert result["minimum_candidate_count"] == 32
    assert result["maximum_candidate_count"] == 64
    assert result["all_candidate_counts_within_registered_bounds"] is True
    assert result["all_paired_positives_present"] is True
    assert result["rgb_open_count"] == 0


def test_place_evaluation_rejects_31_candidates_after_bounded_loads() -> None:
    counts = list(_FROZEN_FAMILY_CANDIDATE_COUNTS)
    counts[evaluation.FAMILIES_V1.index("small_enclosed_maze")] = 31
    rows = _place_rows(tuple(counts))
    preflight = evaluation.preflight_place_retrieval_candidate_counts_v1(rows)
    assert preflight["minimum_candidate_count"] == 31
    assert preflight["rgb_open_count"] == 0

    load_count = 0

    def counted_load(row: PlaceTripletRow) -> RGBTriplet:
        nonlocal load_count
        load_count += 1
        return _load_triplet(row)

    model = _TinyRoleModel()
    with pytest.raises(
        evaluation.MemoryRoleEvaluationContractError, match=r"\[32,64\]"
    ):
        evaluation.evaluate_place_checkpoint_selection_v1(
            model,
            rows,
            load_triplet=counted_load,
            device="cpu",
            training_scene_ids={"train_scene"},
            update=0,
        )
    assert load_count == evaluation.PLACE_SELECTION_ROW_COUNT_V1
    assert model.training is True


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


def _all_controls_pass() -> dict[str, dict[str, bool]]:
    return {
        control: {
            check: True for check in evaluation.PHYSICAL_CONTROL_CHECK_NAMES_V1
        }
        for control in evaluation.PHYSICAL_CONTROL_NAMES_V1
    }


def test_update100_continuation_gate_applies_exact_v3_thresholds() -> None:
    model = _TinyRoleModel()
    place100 = evaluation.evaluate_place_checkpoint_selection_v1(
        model,
        _place_rows(),
        load_triplet=_load_triplet,
        device="cpu",
        training_scene_ids={"train_scene"},
        update=100,
    )
    chance = place100["retrieval"]["exact_chance_recall_at_5"]
    place100["retrieval"]["recall_at_5"] = 1.5 * chance
    place100["noncollapse"]["target_place_key_effective_rank"] = 2.0
    place100["energy"]["negative_minus_positive_bootstrap_lower_95"] = 1.0e-9
    place100["energy"]["positive_family_count"] = 6
    local100 = evaluation.evaluate_local_checkpoint_selection_v1(
        model,
        _local_rows(),
        load_pair=_load_local,
        device="cpu",
        training_scene_ids={"train_scene"},
        update=100,
    )
    physical = {
        "margin_count": 189,
        "passed_margin_count": 60,
        "rough_motion": {"depth_p95_m": 999.0},
    }

    gate = evaluation.evaluate_update100_continuation_gate_v3(
        update100_place=place100,
        update100_local=local100,
        physical_summary=physical,
        integrity_pass=True,
    )
    assert gate["schema"].startswith(
        "lewm_go2_rgb_memory_role_factorized_joint_jepa_v4_"
    )
    assert gate["passed"] is True
    assert gate["action"] == "CONTINUE_SAME_ATTEMPT_TO_UPDATE_400"
    assert gate["retry_authorized"] is False
    assert gate["resume_authorized"] is False

    failures = []
    below_chance = copy.deepcopy(place100)
    below_chance["retrieval"]["recall_at_5"] = 1.5 * chance - 1.0e-9
    failures.append((below_chance, physical, True))
    low_rank = copy.deepcopy(place100)
    low_rank["noncollapse"]["target_place_key_effective_rank"] = 1.999
    failures.append((low_rank, physical, True))
    zero_bootstrap = copy.deepcopy(place100)
    zero_bootstrap["energy"]["negative_minus_positive_bootstrap_lower_95"] = 0.0
    failures.append((zero_bootstrap, physical, True))
    five_families = copy.deepcopy(place100)
    five_families["energy"]["positive_family_count"] = 5
    failures.append((five_families, physical, True))
    low_physical = {**physical, "passed_margin_count": 59}
    failures.append((place100, low_physical, True))
    failures.append((place100, physical, False))
    bad_access = copy.deepcopy(place100)
    bad_access["access"]["rgb_tensor_count"] -= 1
    failures.append((bad_access, physical, True))

    for failed_place, failed_physical, integrity in failures:
        failed = evaluation.evaluate_update100_continuation_gate_v3(
            update100_place=failed_place,
            update100_local=local100,
            physical_summary=failed_physical,
            integrity_pass=integrity,
        )
        assert failed["passed"] is False
        assert failed["action"] == "STOP_VALID_SCIENTIFIC_FAILURE_AT_UPDATE_100"

    bad_local_access = copy.deepcopy(local100)
    bad_local_access["access"]["pair_loader_call_count"] -= 1
    failed = evaluation.evaluate_update100_continuation_gate_v3(
        update100_place=place100,
        update100_local=bad_local_access,
        physical_summary=physical,
        integrity_pass=True,
    )
    assert failed["passed"] is False


def test_terminal_gate_uses_lean_v3_memory_entry_checks() -> None:
    model = _TinyRoleModel()
    place0 = evaluation.evaluate_place_checkpoint_selection_v1(
        model,
        _place_rows(),
        load_triplet=_load_triplet,
        device="cpu",
        training_scene_ids={"train_scene"},
        update=0,
    )
    place0["noncollapse"]["target_place_key_effective_rank"] = 100.0
    place400 = copy.deepcopy(place0)
    place400["update"] = 400
    chance = place400["retrieval"]["exact_chance_recall_at_5"]
    place400["retrieval"]["recall_at_5"] = 3.0 * chance
    place400["retrieval"]["scene_count_above_chance"] = 6
    place400["noncollapse"]["target_place_key_effective_rank"] = 4.0
    place400["energy"]["negative_minus_positive_mean"] = 0.0
    for name in place400["checks"]:
        if name != "target_integrity_pass":
            place400["checks"][name] = False
    place400["passed"] = False
    local400 = evaluation.evaluate_local_checkpoint_selection_v1(
        model,
        _local_rows(),
        load_pair=_load_local,
        device="cpu",
        training_scene_ids={"train_scene"},
        update=400,
    )
    for name in local400["checks"]:
        if name != "target_integrity_pass":
            local400["checks"][name] = False
    local400["passed"] = False
    controls = _all_controls_pass()
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
    assert gate["diagnostic_only"]["place_fixed_point10_energy_gap_is_a_gate"] is False
    assert gate["diagnostic_only"]["place_absolute_r5_point40_is_a_gate"] is False
    assert gate["diagnostic_only"]["place_rank_retention_is_a_gate"] is False
    assert gate["diagnostic_only"]["local_metrics_are_a_gate"] is False
    assert gate["diagnostic_only"]["place_legacy_conjunction_passed"] is False
    assert gate["diagnostic_only"]["local_legacy_conjunction_passed"] is False
    assert gate["observed"]["target_place_key_rank_retention_ratio"] == pytest.approx(
        0.04
    )
    assert gate["memory_reset_reverse_shuffle_required"] is False
    assert gate["navigation_evaluation_required"] is False
    assert len(gate["causal_control_checks"]) == 12
    assert gate["action"] == (
        "PASS_MEMORY_ENTRY_ELIGIBLE_FOR_LEARNED_MEMORY_INTEGRATION"
    )

    failing_inputs = []
    below_chance = copy.deepcopy(place400)
    below_chance["retrieval"]["recall_at_5"] = 3.0 * chance - 1.0e-9
    failing_inputs.append((below_chance, physical, controls, True))
    five_scenes = copy.deepcopy(place400)
    five_scenes["retrieval"]["scene_count_above_chance"] = 5
    failing_inputs.append((five_scenes, physical, controls, True))
    low_rank = copy.deepcopy(place400)
    low_rank["noncollapse"]["target_place_key_effective_rank"] = 3.999
    failing_inputs.append((low_rank, physical, controls, True))
    failing_inputs.append(
        (place400, {**physical, "passed_margin_count": 72}, controls, True)
    )
    failed_controls = copy.deepcopy(controls)
    failed_controls["wrong_rgb"]["positive_family_count"] = False
    failing_inputs.append((place400, physical, failed_controls, True))
    failing_inputs.append((place400, physical, controls, False))

    for failed_place, failed_physical, failed_controls, integrity in failing_inputs:
        failed = evaluation.evaluate_terminal_gate_v1(
            update0_place=place0,
            update400_place=failed_place,
            update400_local=local400,
            physical_summary=failed_physical,
            controls=failed_controls,
            integrity_pass=integrity,
        )
        assert failed["passed"] is False
        assert failed["action"] == "FAIL_TERMINAL_NO_MEMORY_INTEGRATION"

    with pytest.raises(evaluation.MemoryRoleEvaluationContractError, match="nonfinite"):
        evaluation.evaluate_terminal_gate_v1(
            update0_place=place0,
            update400_place=place400,
            update400_local=local400,
            physical_summary=physical,
            controls=controls,
            integrity_pass=True,
            diagnostics={"tail_energy": float("nan")},
        )
