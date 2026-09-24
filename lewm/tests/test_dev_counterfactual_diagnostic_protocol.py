from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts import dev_probe_counterfactual_action_fidelity as fidelity
from scripts import dev_probe_counterfactual_overfit_capacity as capacity


def _context(scene: str = "scene") -> list[str]:
    return [
        f"{scene}/rgb/frame_{index:06d}_env_00.png"
        for index in (0, 240, 480)
    ]


def _candidate_rows(
    *,
    source_role: str,
    physics_validated: bool = True,
    include_provenance: bool = True,
) -> list[dict]:
    split = fidelity.SOURCE_SPLITS[source_role]
    context = _context(f"{source_role}_scene")
    rows = []
    for action_index, primitive in enumerate(("arc_left", "forward_fast", "yaw_right")):
        row = {
            "complete_valid_future_sequence": True,
            "future_frame_physics_validated": [physics_validated],
            "split": split,
            "scene_id": f"{source_role}_scene",
            "source_index": 7,
            "family": "open_obstacle_field",
            "start_frame": context[-1],
            "primitive_sequence": [primitive],
            "future_frames": [f"future_{source_role}_{action_index}.png"],
            "consequence_labels": {"target_progress_m": float(action_index)},
        }
        if include_provenance:
            row["h6_context_frames"] = context
            row["h6_historical_actions"] = ["backward", "forward_slow"]
        rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_context_provenance_requires_five_tick_h6_cadence_and_actions() -> None:
    valid = fidelity.ContextProvenance(
        source_role="train",
        scene_id="scene",
        start_frame=_context()[-1],
        context_frames=tuple(_context()),
        historical_actions=(2, 5),
    )
    fidelity.validate_context_provenance(valid)

    adjacent_tick_context = fidelity.ContextProvenance(
        source_role="train",
        scene_id="scene",
        start_frame="scene/rgb/frame_000096_env_00.png",
        context_frames=(
            "scene/rgb/frame_000000_env_00.png",
            "scene/rgb/frame_000048_env_00.png",
            "scene/rgb/frame_000096_env_00.png",
        ),
        historical_actions=(2, 5),
    )
    with pytest.raises(
        fidelity.CounterfactualProtocolError,
        match="five-tick H6 endpoints",
    ):
        fidelity.validate_context_provenance(adjacent_tick_context)


def test_group_loading_preserves_train_eval_and_actual_history(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_rows(train_path, _candidate_rows(source_role="train"))
    _write_rows(eval_path, _candidate_rows(source_role="eval"))

    loaded = fidelity.load_groups(
        3,
        sources={"train": train_path, "eval": eval_path},
        provenance_by_role={},
        allow_unbound_embedded_provenance=True,
    )

    assert len(loaded.groups_by_role["train"]) == 1
    assert len(loaded.groups_by_role["eval"]) == 1
    train_group = loaded.groups_by_role["train"][0]
    eval_group = loaded.groups_by_role["eval"][0]
    assert train_group["source_role"] == "train"
    assert eval_group["source_role"] == "eval"
    assert train_group["scene_id"] != eval_group["scene_id"]
    assert train_group["historical_actions"] == [2, 5]
    assert train_group["context_frames"] == _context("train_scene")
    assert train_group["target_evidence_class"] == "physics_validated"
    train_binding = loaded.audit["roles"]["train"]["source_binding"]
    assert train_binding["byte_count"] == train_path.stat().st_size
    assert len(train_binding["sha256"]) == 64
    fidelity.assert_counterfactual_sources_unchanged(loaded.audit)


def test_group_loading_rejects_train_eval_scene_overlap(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_rows = _candidate_rows(source_role="train")
    eval_rows = _candidate_rows(source_role="eval")
    for row in eval_rows:
        row["scene_id"] = "train_scene"
    _write_rows(train_path, train_rows)
    _write_rows(eval_path, eval_rows)

    with pytest.raises(
        fidelity.CounterfactualProtocolError,
        match="train/eval scene overlap",
    ):
        fidelity.load_groups(
            3,
            sources={"train": train_path, "eval": eval_path},
            provenance_by_role={},
            allow_unbound_embedded_provenance=True,
        )


def test_group_loading_fails_closed_without_h6_provenance(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    _write_rows(
        train_path,
        _candidate_rows(source_role="train", include_provenance=False),
    )

    with pytest.raises(
        fidelity.CounterfactualProtocolError,
        match="no reset-safe H6 block-cadence/action provenance",
    ):
        fidelity.load_groups(
            3,
            sources={"train": train_path},
            provenance_by_role={},
            allow_unbound_embedded_provenance=True,
        )


def test_kinematic_rows_are_rejected_by_default_and_explicit_when_allowed(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    _write_rows(
        train_path,
        _candidate_rows(source_role="train", physics_validated=False),
    )

    claim_bearing = fidelity.load_groups(
        3,
        sources={"train": train_path},
        provenance_by_role={},
        allow_unbound_embedded_provenance=True,
    )
    assert claim_bearing.groups_by_role["train"] == ()
    assert claim_bearing.audit["roles"]["train"]["nonphysics_rows_rejected"] == 3

    diagnostic = fidelity.load_groups(
        3,
        sources={"train": train_path},
        provenance_by_role={},
        evidence_scope="kinematic_diagnostic",
        allow_unbound_embedded_provenance=True,
    )
    assert len(diagnostic.groups_by_role["train"]) == 1
    assert (
        diagnostic.groups_by_role["train"][0]["target_evidence_class"]
        == "kinematic_render_only"
    )


def _summary_row(*, group_id: str, family: str, k: int, hits: int) -> dict:
    return {
        "group_id": group_id,
        "source_role": "train",
        "scene_id": group_id,
        "source_index": 0,
        "family": family,
        "target_evidence_class": "physics_validated",
        "actions": list(range(k)),
        "k": k,
        "fidelity_hit_count": hits,
        "discrimination_hit_count": hits,
        "fidelity_rate": hits / k,
        "discrimination_rate": hits / k,
        "fidelity_strict_win_count": hits,
        "discrimination_strict_win_count": hits,
        "fidelity_margin_sum": float(hits),
        "discrimination_margin_sum": float(hits),
        "fidelity_margin_mean": hits / k,
        "discrimination_margin_mean": hits / k,
        "chance": 1.0 / k,
        "diag_mean": 0.1,
        "offdiag_mean": 0.2,
        "pred_spread": 1.0,
    }


def test_summary_uses_matching_macro_and_micro_chance_weights() -> None:
    summary = fidelity.summarize(
        [
            _summary_row(group_id="g3", family="f3", k=3, hits=3),
            _summary_row(group_id="g5", family="f5", k=5, hits=0),
        ],
        "synthetic",
    )

    assert summary["macro"]["fidelity_rate"] == pytest.approx(0.5)
    assert summary["macro"]["chance"] == pytest.approx((1 / 3 + 1 / 5) / 2)
    assert summary["micro"]["fidelity_rate"] == pytest.approx(3 / 8)
    assert summary["micro"]["chance"] == pytest.approx(2 / 8)


def test_action_controls_preserve_candidate_cardinality() -> None:
    actions = [0, 3, 8]
    assert fidelity.conditioned_candidate_actions(actions, "factual") == actions
    assert fidelity.conditioned_candidate_actions(actions, "action_blind") == [6, 6, 6]
    assert fidelity.conditioned_candidate_actions(actions, "action_shuffled") == [3, 8, 0]


def test_development_output_cannot_escape_registered_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development output must remain"):
        fidelity.require_development_output(tmp_path / "result.json")

    inside = fidelity.DEV_OUTPUT_ROOT / "counterfactual" / "result.json"
    assert fidelity.require_development_output(inside) == inside


def test_checkpoint_and_rgb_inputs_cannot_escape_development_contract(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "update_000100.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    with pytest.raises(ValueError, match="development checkpoint must remain"):
        fidelity.require_development_checkpoint(checkpoint)
    checkpoint_link = tmp_path / "checkpoint_link.pt"
    checkpoint_link.symlink_to(checkpoint)
    with pytest.raises(ValueError, match="must be a non-symlink file"):
        fidelity.require_development_checkpoint(checkpoint_link)

    rgb = tmp_path / "frame_000240_env_00.png"
    rgb.write_bytes(b"not opened")
    with pytest.raises(
        fidelity.CounterfactualProtocolError,
        match="escapes the bound render root",
    ):
        fidelity.decode(str(rgb), torch.device("cpu"))


def test_counterfactual_checkpoint_loading_is_pinned_and_weights_only() -> None:
    source = Path(fidelity.__file__).read_text()
    assert "weights_only=False" not in source
    assert source.count("weights_only=True") == 2
    assert fidelity.PREDECESSOR_BYTE_COUNT == 52_282_877
    assert fidelity.PREDECESSOR_SHA256 == (
        "f5aac23cf275d73b92ce5609a583dea89"
        "f6686a624d4889d9762740535aab873"
    )


def test_counterfactual_json_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    fidelity.write_json_atomic(path, {"status": "complete"})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fidelity.write_json_atomic(path, {"status": "replacement"})


def test_diagnostic_source_binding_detects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("first\n")
    binding = fidelity.file_binding(source)
    fidelity.assert_file_bindings_unchanged([binding], kind="test source")

    source.write_text("second\n")
    with pytest.raises(RuntimeError, match="test source changed"):
        fidelity.assert_file_bindings_unchanged([binding], kind="test source")


def test_unbound_embedded_provenance_is_rejected_by_default(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    _write_rows(train_path, _candidate_rows(source_role="train"))

    with pytest.raises(
        fidelity.CounterfactualProtocolError,
        match="no reset-safe H6 block-cadence/action provenance",
    ):
        fidelity.load_groups(
            3,
            sources={"train": train_path},
            provenance_by_role={},
        )


def test_embedded_provenance_must_match_bound_h6_index(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    rows = _candidate_rows(source_role="train")
    _write_rows(train_path, rows)
    context = _context("train_scene")
    conflicting = fidelity.ContextProvenance(
        source_role="train",
        scene_id="train_scene",
        start_frame=context[-1],
        context_frames=tuple(context),
        historical_actions=(0, 1),
    )

    with pytest.raises(
        fidelity.CounterfactualProtocolError,
        match="disagrees with the bound H6 index",
    ):
        fidelity.load_groups(
            3,
            sources={"train": train_path},
            provenance_by_role={
                "train": {
                    fidelity._render_leaf(context[-1]): conflicting,
                }
            },
        )


def test_matrix_metrics_distinguish_fidelity_from_discrimination() -> None:
    energy = torch.tensor([[0.0, 10.0], [1.0, 2.0]], dtype=torch.float64)
    group = {
        "group_id": "g",
        "source_role": "eval",
        "scene_id": "scene",
        "source_index": 0,
        "family": "family",
        "target_evidence_class": "physics_validated",
        "actions": [0, 1],
    }

    result = fidelity._matrix_metrics(energy, group)

    assert result["fidelity_rate"] == pytest.approx(0.5)
    assert result["discrimination_rate"] == pytest.approx(1.0)
    assert result["fidelity_strict_win_count"] == 1
    assert result["discrimination_strict_win_count"] == 2
    assert [branch["fidelity_hit"] for branch in result["branch_results"]] == [
        True,
        False,
    ]


def test_capacity_partition_evaluation_is_one_frozen_snapshot(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1)
    model.train()
    observations: list[tuple[bool, float]] = []

    def fake_forward_group(
        live_model,
        _context,
        _targets,
        actions,
        _historical_actions,
        _mask,
        _device,
        *,
        action_mode,
    ):
        observations.append((live_model.training, float(live_model.weight.item())))
        assert action_mode == "factual"
        return torch.eye(len(actions), dtype=torch.float32).neg()

    monkeypatch.setattr(capacity, "forward_group", fake_forward_group)
    groups = []
    for index in range(2):
        group = {
            "group_id": f"g{index}",
            "source_role": "train",
            "scene_id": f"scene{index}",
            "source_index": index,
            "family": "open_obstacle_field",
            "target_evidence_class": "physics_validated",
            "actions": [0, 1, 2],
            "historical_actions": [3, 4],
        }
        groups.append((None, None, group["actions"], group))

    result = capacity.evaluate_partition(
        model,
        groups,
        mask=None,
        device=torch.device("cpu"),
        action_mode="factual",
    )

    assert len(result["group_results"]) == 2
    assert observations[0] == observations[1]
    assert observations[0][0] is False
    assert model.training is True
