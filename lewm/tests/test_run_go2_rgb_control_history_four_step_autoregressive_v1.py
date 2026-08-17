"""Synthetic CPU/source tests for the bounded four-step rollout runner."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scripts import (  # noqa: E402
    run_go2_rgb_control_history_four_step_autoregressive_v1 as R,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/run_go2_rgb_control_history_four_step_autoregressive_v1.py"


def _function_source(name: str) -> str:
    text = SOURCE.read_text()
    node = next(
        item for item in ast.parse(text).body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    return "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno])


def test_four_step_objective_is_the_exact_mean_of_four_frozen_l1_losses() -> None:
    outputs = tuple(
        torch.full((2, 3), float(horizon), requires_grad=True)
        for horizon in range(1, 5)
    )
    targets = tuple(torch.zeros_like(value) for value in outputs)
    loss, components = R.four_step_objective(outputs, targets)
    assert [float(value.detach()) for value in components] == [1.0, 2.0, 3.0, 4.0]
    assert float(loss.detach()) == 2.5
    loss.backward()
    for output in outputs:
        assert output.grad is not None
        assert torch.allclose(output.grad, torch.full_like(output, 1.0 / 24.0))
    with pytest.raises(RuntimeError, match="exactly four"):
        R.four_step_objective(outputs[:3], targets[:3])


def test_objective_component_separation_is_diagonal_with_quarter_derivatives() -> None:
    outputs = tuple(torch.full((2, 3), float(index + 1)) for index in range(4))
    targets = tuple(torch.zeros_like(output) for output in outputs)
    result = R.objective_component_separation(outputs, targets)
    identity = [[row == column for column in range(4)] for row in range(4)]
    assert result["changed_component_matrix"] == identity
    assert result["expected_changed_component_matrix"] == identity
    assert result["only_registered_component_changes"] is True
    assert result["combined_loss_derivative_per_component"] == [0.25] * 4
    assert result["all_derivatives_exactly_one_quarter"] is True


class _LatestFrameModel(torch.nn.Module):
    """Tiny differentiable model used only to witness attached autoregression."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([0.75, 1.25, 1.75]))

    def forward(self, context, action, mask, proprio, valid, control):
        del action, mask, proprio, valid, control
        return context[:, -1] * self.scale


def test_h3_and_h4_losses_backpropagate_through_the_attached_chain_on_cpu() -> None:
    generator = torch.Generator().manual_seed(41)
    model = _LatestFrameModel()
    context = torch.randn(2, 3, 2, 3, generator=generator)
    actions = tuple(torch.randn(2, 10, generator=generator) for _ in range(4))
    control = torch.randn(2, 3, 5, 2, generator=generator)
    targets = tuple(torch.randn(2, 2, 3, generator=generator) for _ in range(4))
    outputs = R.P.unroll(
        model, context, actions, proprio=None, control=control, max_h=4)
    assert len(outputs) == 4
    loss, components = R.four_step_objective(outputs, targets)
    h3_via_h1 = torch.autograd.grad(
        components[2], outputs[0], retain_graph=True)[0]
    h4_via_h1 = torch.autograd.grad(
        components[3], outputs[0], retain_graph=True)[0]
    assert torch.isfinite(h3_via_h1).all() and torch.count_nonzero(h3_via_h1)
    assert torch.isfinite(h4_via_h1).all() and torch.count_nonzero(h4_via_h1)
    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad).all()
    assert torch.count_nonzero(model.scale.grad)


def test_forward_path_is_rgb_control_history_only_and_exactly_four_steps() -> None:
    source = _function_source("forward_four_step")
    assert "P.unroll(" in source
    assert "proprio=None" in source
    assert 'control=batch["control"]' in source
    assert "max_h=4" in source
    assert "four_step_objective(outputs, batch[\"targets\"])" in source
    assert ".detach(" not in source


def test_runner_imports_no_utility_or_sealed_consumer() -> None:
    tree = ast.parse(SOURCE.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert all("utility" not in name.lower() for name in imported)
    assert all("sealed" not in name.lower() for name in imported)


def _rendered_fixture(
    root: Path, scene: str, source: int, env: str = "00"
) -> dict[int, dict]:
    metadata = {}
    for horizon in range(5):
        frame = source + 240 * horizon
        metadata[frame] = {
            "env": 0,
            "episode_id": 7,
            "reset_count": 2,
            "block_size": 5,
            "sequence_id": 100 + horizon,
            "primitive": "h1" if horizon == 0 else "h2",
        }
        if horizon:
            path = root / scene / "rgb" / f"frame_{frame:06d}_env_{env}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic-not-an-image")
    return metadata


def test_common_manifest_reducer_preserves_order_and_localises_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(R.HSEQ, "V03", tmp_path)
    factorial_rows = []
    map_entries = []
    proprio_rows = []
    two_step_rows = []
    frame_metadata = {}
    for position, source in enumerate((0, 2_000, 4_000, 6_000)):
        pair = f"pair-{position}"
        scene = f"scene-{position}"
        factorial_rows.append({
            "position": position,
            "stable_row_id": f"stable-{position}",
            "pair_sha256": pair,
            "manifest_row_index": position,
            "split": "train",
            "family": "fixture_family",
        })
        map_entries.append({"manifest_row_index": position})
        blocks = [[0.0] * 10 for _ in range(4)]
        if position == 2:
            blocks = blocks[:3]
        proprio_rows.append({"pair_sha256": pair, "action_blocks": blocks})
        two_step_rows.append({
            "pair_sha256": pair,
            "scene": scene,
            "t": source,
            "env": "00",
            "env_index": 0,
            "episode_id": 7,
            "reset_count": 2,
            "action_step1": "h1",
            "action_step2": "h2",
        })
        frame_metadata[scene] = _rendered_fixture(tmp_path, scene, source)
    # Position 1 crosses a reset boundary; position 2 lacks action block H4.
    frame_metadata["scene-1"][2_000 + 2 * 240]["reset_count"] = 3
    included, excluded = R.build_common_manifest_rows(
        {"rows": list(reversed(factorial_rows))},
        {"entries": map_entries},
        proprio_rows,
        two_step_rows,
        frame_metadata,
    )
    assert [row["position"] for row in included] == [0, 3]
    assert [row["stable_row_id"] for row in included] == ["stable-0", "stable-3"]
    assert [row["first_exclusion_reason"] for row in excluded] == [
        "reset_or_episode_boundary",
        "fewer_than_four_verified_post_slew_action_blocks",
    ]
    assert all(row["max_horizon"] == 4 for row in included)
    assert all(row["first_exclusion_reason"] is None for row in included)


def test_create_only_receipts_are_read_only_and_cannot_be_replaced(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    R._write_json_once(receipt, {"attempt": 1})
    assert receipt.stat().st_mode & 0o777 == 0o444
    with pytest.raises(RuntimeError, match="create-only"):
        R._write_json_once(receipt, {"attempt": 2})


def test_failed_stage_attempt_is_durable_and_cannot_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    success = runtime / "smoke.json"
    monkeypatch.setattr(R, "runtime_root", lambda: runtime)
    monkeypatch.setattr(
        R, "require_contract", lambda: {"contract_digest": "c" * 64})
    monkeypatch.setattr(
        R, "environment_record", lambda require_exact=False: {
            "fixture": True, "exact": require_exact})
    R._start_once("smoke", success)
    attempt = runtime / "attempts/smoke.json"
    assert attempt.is_file()
    assert attempt.stat().st_mode & 0o777 == 0o444
    with pytest.raises(RuntimeError, match="already attempted without success"):
        R._start_once("smoke", success)
    success.write_text("{}\n")
    R._start_once("smoke", success)  # completed stages are read-only consumers


def test_registered_base_checks_both_file_and_state_digests() -> None:
    source = _function_source("_registered_base")
    assert "seed not in C.FROZEN_SEEDS" in source
    assert "C.BASE_WEIGHT_SHA256[seed]" in source
    assert "C.BASE_STATE_DIGEST[seed]" in source
    assert "F.state_digest(state)" in source
    fresh = _function_source("_fresh_model")
    assert '"rgb_one_step"' in fresh
    assert "_model_state_digest(model)" in fresh
    assert "F.assert_no_active_dropout(model)" in fresh


def test_smoke_warmup_chain_probe_and_checkpoint_state_are_discarded() -> None:
    source = _function_source("smoke_stage")
    assert "initial_loss, initial_components" in source
    assert "exact_equal_weight_formula" in source
    assert "CK.save(" in source and "CK.load_for_resume(" in source
    assert "_model_state_digest(resumed) == updated_model_digest" in source
    assert "_optimizer_state_digest(resumed_optimizer)" in source
    assert "warmup_steps = 50" in source
    assert "range(2, warmup_steps)" in source
    assert "original_next = _one_batch_step(model, optimizer, next_batch, device)" \
        in source
    assert "resumed_next = _one_batch_step(resumed, resumed_optimizer, next_batch, device)" \
        in source
    assert '"next_batch_update_state_equal": next_exact' in source
    assert "output.retain_grad()" in source
    assert "h3 = chain_probe(3)" in source
    assert "h4 = chain_probe(4)" in source
    assert source.rindex("discarded, _, _ = _fresh_model(seed, device)") > \
        source.index("h4 = chain_probe(4)")
    assert '"warmup_state_discarded": True' in source
    assert '"scientific_optimizer_step_performed": False' in source
    assert '"calibration_or_counterfactual_corpus_opened": False' in source


def test_manifest_receipt_discloses_historical_sample_mismatch() -> None:
    source = _function_source("manifest_stage")
    assert '"historical_control_train_rows": 3_922' in source
    assert '"historical_control_train_row_difference": 68' in source
    assert '"historical_controls_sample_matched": False' in source
    assert '"historical_controls_retrained_or_reselected": False' in source


def test_common_plan_filters_the_historical_order_then_rechunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"split": "train", "position": position}
        for position in (0, 2, 3, 4, 5)
    ] + [{"split": "checkpoint_selection", "position": 6}]
    calls = []

    def historical_plan(seed: int, epoch: int, count: int, batch: int):
        calls.append((seed, epoch, count, batch))
        return [[5, 1, 4, 0], [3, 2]]

    monkeypatch.setitem(R.C.DATA_ORDER_CONTRACT, "historical_train_rows", 6)
    monkeypatch.setattr(R.F, "batch_plan", historical_plan)
    plan = R.common_plan_from_rows(17, 9, rows)
    assert calls == [(17, 9, 6, 4)]
    # Historical row 1 is the sole H4-invalid row.  All other identities keep
    # their historical relative order and are remapped to common-row positions.
    assert plan == [[4, 3, 0, 2], [1]]
    assert [index for batch in plan for index in batch] == [4, 3, 0, 2, 1]


def test_resource_gate_uses_strict_vram_ram_and_filesystem_inequalities() -> None:
    gib = 2**30
    passing = R.resource_gate(
        peak_vram_bytes=28 * gib - 1,
        minimum_mem_available_bytes=20 * gib + 1,
        filesystem_free_bytes=12 * gib + 1,
        projected_remaining_bytes=10 * gib,
    )
    assert passing["pass"] is True
    assert all(passing["checks"].values())
    assert R.resource_gate(
        peak_vram_bytes=28 * gib,
        minimum_mem_available_bytes=20 * gib + 1,
        filesystem_free_bytes=12 * gib + 1,
        projected_remaining_bytes=10 * gib,
    )["pass"] is False
    assert R.resource_gate(
        peak_vram_bytes=28 * gib - 1,
        minimum_mem_available_bytes=20 * gib,
        filesystem_free_bytes=12 * gib + 1,
        projected_remaining_bytes=10 * gib,
    )["pass"] is False
    assert R.resource_gate(
        peak_vram_bytes=28 * gib - 1,
        minimum_mem_available_bytes=20 * gib + 1,
        filesystem_free_bytes=12 * gib,
        projected_remaining_bytes=10 * gib,
    )["pass"] is False


def test_preflight_is_one_full_epoch_and_discards_its_updated_state() -> None:
    source = _function_source("preflight_stage")
    assert "epoch = _train_epoch(model, optimizer, loader, seed, 0, device, monitor)" \
        in source
    assert '"full_epochs_measured": 1' in source
    assert '"projected_eight_run_wall_seconds"' in source
    assert "epoch[\"wall_seconds\"] * EPOCHS * len(C.FROZEN_SEEDS)" in source
    assert "peak_vram_bytes=peak_reserved" in source
    assert "del model, optimizer" in source
    assert "reloaded, _, _ = _fresh_model(seed, device)" in source
    assert '"preflight_weights_discarded": True' in source
    assert '"scientific_epoch_completed": False' in source


def test_training_runs_all_24_epochs_and_retains_epoch21_without_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = R.C.FROZEN_SEEDS[0]
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "resource_preflight.json").write_text('{"valid": true}\n')
    epochs_run = []
    checkpoint_history = []

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    model = Tiny()
    base = {"state_digest": R.C.BASE_STATE_DIGEST[seed]}
    monkeypatch.setattr(R, "runtime_root", lambda: runtime)
    monkeypatch.setattr(
        R, "require_contract", lambda: {"contract_digest": "c" * 64})
    monkeypatch.setattr(R, "environment_record", lambda require_exact=False: {})
    monkeypatch.setattr(R, "_start_once", lambda *args, **kwargs: None)
    monkeypatch.setattr(R, "validate_preflight_receipt", lambda: {"valid": True})
    monkeypatch.setattr(R, "validate_training_input_files", lambda: {})
    monkeypatch.setattr(R, "FourStepLoader", lambda: SimpleNamespace())
    monkeypatch.setattr(R, "resolve_device", lambda name: torch.device("cuda"))
    monkeypatch.setattr(
        R, "_fresh_model", lambda requested_seed, device:
        (model, tmp_path / "base.pt", base))
    monkeypatch.setattr(R, "_model_state_digest", lambda value: "m" * 64)
    monkeypatch.setattr(R, "_optimizer_state_digest", lambda value: "o" * 64)
    monkeypatch.setattr(R, "_all_finite_model_optimizer", lambda *args: True)
    monkeypatch.setattr(R.F, "state_digest", lambda state: "m" * 64)
    monkeypatch.setattr(
        R, "validate_common_manifest", lambda: {"common_rows_digest": "r" * 64})
    monkeypatch.setattr(
        R, "validate_target_cache_index",
        lambda: {"target_cache_index_digest": "t" * 64})
    monkeypatch.setattr(
        R.F, "terminal_window", lambda history: {
            "epochs": [item["epoch"] for item in history[19:24]]})
    monkeypatch.setattr(R.torch.cuda, "empty_cache", lambda: None)

    def train_epoch(model, optimizer, loader, requested_seed, epoch, device):
        del model, optimizer, loader, device
        assert requested_seed == seed
        epochs_run.append(epoch)
        return {
            "epoch": epoch, "batches": 1,
            "e1": 1.0, "e2": 2.0, "e3": 3.0, "e4": 4.0,
            "loss": 2.5, "wall_seconds": 0.01,
        }

    def save_checkpoint(path: Path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_history.extend(item["epoch"] for item in kwargs["extra"]["history"])
        torch.save({
            "schema": R.CK.SCHEMA,
            "epoch": kwargs["epoch"],
            "seed": kwargs["seed"],
            "model_state_dict": kwargs["model"].state_dict(),
            "common_rows_digest": kwargs["extra"]["common_rows_digest"],
            "target_cache_index_digest": kwargs["extra"][
                "target_cache_index_digest"],
            "base_state_digest": kwargs["extra"]["base_state_digest"],
        }, path)
        sha256 = R._sha256_file(path)
        ledger = path.parent / "checkpoint_receipts.jsonl"
        ledger.write_text('{"sha256":"' + sha256 + '"}\n')
        return {"sha256": sha256, "bytes": path.stat().st_size}

    monkeypatch.setattr(R, "_train_epoch", train_epoch)
    monkeypatch.setattr(R.CK, "save", save_checkpoint)
    monkeypatch.setattr(R.CK, "load_for_resume", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        R, "validate_training_receipt",
        lambda requested_seed: R._read_json(R._seed_receipt_path(requested_seed)))

    result = R.train_seed_stage(SimpleNamespace(seed=seed, device="cuda:0"))
    assert epochs_run == list(range(24))
    assert checkpoint_history == list(range(22))
    assert epochs_run[-2:] == [22, 23]
    assert result["epochs_trained"] == 24
    assert result["checkpoint_epoch"] == 21
    assert result["best_epoch_selected"] is False
    assert result["finite_weak_run_retained"] is True
    assert result["extension_or_retry"] is False
    assert [item["epoch"] for item in result["history"]] == list(range(24))
    assert result["terminal_window"]["epochs"] == [19, 20, 21, 22, 23]


def test_t_interval_uses_eight_paired_seeds_sample_sd_and_df7() -> None:
    result = R.t_interval([1, 2, 3, 4, 5, 6, 7, 8])
    assert result["n"] == 8
    assert result["mean"] == 4.5
    assert result["sample_standard_deviation"] == pytest.approx(math.sqrt(6.0))
    half_width = 2.3646242510102993 * math.sqrt(6.0) / math.sqrt(8.0)
    assert result["two_sided_95_t_interval"] == pytest.approx(
        [4.5 - half_width, 4.5 + half_width])
    with pytest.raises(RuntimeError, match="eight finite"):
        R.t_interval([1.0] * 7)
    with pytest.raises(RuntimeError, match="eight finite"):
        R.t_interval([1.0] * 7 + [float("nan")])


def test_paired_effect_sign_is_positive_for_benefit_in_both_directions() -> None:
    higher = R.paired_effect_summary(
        [1.0] * 8, [2.0] * 8, [3.0] * 8, higher_is_better=True)
    assert higher["benefit_orientation"] == "four_step - comparator"
    assert higher["four_step_minus_two_step_benefit"]["values"] == [1.0] * 8
    assert higher["four_step_minus_two_step_benefit"]["eight_seed_effects"] == [
        {"seed": int(seed), "effect": 1.0} for seed in R.C.FROZEN_SEEDS]
    assert higher["four_step_minus_one_step_benefit"]["mean"] == 2.0
    lower = R.paired_effect_summary(
        [3.0] * 8, [2.0] * 8, [1.0] * 8, higher_is_better=False)
    assert lower["benefit_orientation"] == "comparator - four_step"
    assert lower["four_step_minus_two_step_benefit"]["values"] == [1.0] * 8
    assert lower["four_step_minus_one_step_benefit"]["mean"] == 2.0
    assert R.EFFECT_METRICS["normalized_error_reduction"] == (
        "normalised_error_vs_persistence", False)


def _effect(mean: float, lower: float | None = None,
            upper: float | None = None) -> dict:
    return {
        "mean": mean,
        "two_sided_95_t_interval": [
            mean - 0.1 if lower is None else lower,
            mean + 0.1 if upper is None else upper,
        ],
    }


def _interpretation_fixture() -> dict:
    direct = (
        "changed_token_correct_future_cosine", "normalized_error_reduction")
    result = {"equal_family": {}}
    for horizon in ("H1", "H2", "H4"):
        result["equal_family"][horizon] = {
            endpoint: {"four_step_minus_two_step_benefit": _effect(0.1)}
            for endpoint in direct
        }
    result["equal_family"]["H4"].update({
        "correct_branch_top1_retrieval": {
            "four_step_minus_two_step_benefit": _effect(-0.1)},
        "pairwise_branch_discrimination": {
            "four_step_minus_two_step_benefit": _effect(0.01)},
    })
    return result


def test_usefulness_requires_both_direct_h4_means_and_top1_or_pairwise() -> None:
    analysis = _interpretation_fixture()
    result = R.interpretation_from_effects(analysis)
    assert result["direct_fidelity_improved"] is True
    assert result["retrieval_improved"] is True  # pairwise, despite top-1 < 0
    assert result["useful"] is True
    assert result["planning_or_utility_claim"] is False
    analysis["equal_family"]["H4"]["normalized_error_reduction"][
        "four_step_minus_two_step_benefit"] = _effect(-0.01)
    result = R.interpretation_from_effects(analysis)
    assert result["direct_fidelity_improved"] is False
    assert result["useful"] is False
    assert result["classification"] == (
        "DIRECT_FIDELITY_EVIDENCE_DISCORDANT_OR_MIXED")


def test_material_h1_h2_regression_requires_ci_wholly_below_zero() -> None:
    analysis = _interpretation_fixture()
    h1 = analysis["equal_family"]["H1"][
        "changed_token_correct_future_cosine"]
    h1["four_step_minus_two_step_benefit"] = _effect(-0.2, -0.3, -0.01)
    result = R.interpretation_from_effects(analysis)
    assert result["horizon_tradeoff"] is True
    assert [(row["horizon"], row["endpoint"])
            for row in result["H1_H2_material_regressions"]] == [
        ("H1", "changed_token_correct_future_cosine")]
    h1["four_step_minus_two_step_benefit"] = _effect(-0.2, -0.3, 0.0)
    result = R.interpretation_from_effects(analysis)
    assert result["horizon_tradeoff"] is False
    assert result["H1_H2_material_regressions"] == []


def test_occupancy_is_h2_h3_h4_only_and_h1_fails_closed() -> None:
    assert R.occupancy_horizons([2, 3, 4]) == (2, 3, 4)
    with pytest.raises(RuntimeError, match="occupancy horizon"):
        R.occupancy_horizons([1, 2, 3, 4])
    with pytest.raises(RuntimeError, match="occupancy horizon"):
        R.occupancy_horizons([2, 4])
    source = _function_source("_score_occupancy_state")
    assert "for horizon in (2, 3, 4)" in source
    assert "targets[index, horizon - 1]" in source
    assert "horizon in (1, 2, 3, 4)" not in source
    analysis_source = _function_source("_occupancy_analysis")
    assert '"1": {' in analysis_source
    assert '"horizon": 1, "available": False' in analysis_source
    assert '"predictor_latents_scored": False' in analysis_source
    assert '"H1_not_reinterpreted": True' in analysis_source
    assert '"qualified_true_target_horizons": [2, 3, 4]' in analysis_source
    assert '"probe_refit": False' in analysis_source


def test_evaluation_reuses_persisted_controls_and_forwards_only_new_models() -> None:
    source = _function_source("evaluate_stage")
    assert 'frozen_result["cells_by_seed"]' in source
    assert '"one_step": frozen_cells["rgb_one_step"]' in source
    assert '"two_step": frozen_cells["rgb_rollout"]' in source
    assert '"four_step": four' in source
    assert source.count("_load_epoch21_model(seed, device)") == 1
    assert '"historical_comparator_model_forwards": 0' in source
    assert '"new_four_step_model_forward_states": 8 * 20' in source
    assert 'analysis["equal_family"]["H4"]' in source
    assert '"no_predictor_utility_shards_opened": True' in source
    assert '"no_branches_targets_labels_or_masks_regenerated": True' in source
    assert source.count("prediction = Q.predict_state(") == 1
    assert "Q.score_state_predictions(bundle, state, prediction, device)" in source
    assert "_score_occupancy_state(\n                prediction," in source


def test_every_stage_has_one_implementation_and_no_placeholder() -> None:
    text = SOURCE.read_text()
    tree = ast.parse(text)
    definitions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    for name in (
        "issue_stage", "manifest_stage", "encode_stage", "smoke_stage",
        "preflight_stage", "train_seed_stage", "train_all_stage",
        "evaluate_stage", "validate_stage",
    ):
        assert definitions.count(name) == 1, f"duplicate or missing {name}"
    assert "NotImplementedError" not in text


def test_encode_batch_cli_argument_is_parsed_without_touching_runtime() -> None:
    args = R.parse_args(["encode", "--encode-batch", "7"])
    assert args.stage == "encode"
    assert args.encode_batch == 7


def _write_read_only(path: Path, value: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    path.chmod(0o444)


def test_validate_terminal_closes_the_exact_namespace_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    contract = {
        "contract_digest": "c" * 64,
        "source_closure": {
            "source_repository_commit": "1" * 40,
            "four_step_source_closure_digest": "s" * 64,
        },
    }
    expected = {
        "contract.json", "target_availability.json", "common_h4_rows.jsonl",
        "common_h4_manifest.json", "training_input_verification.json",
        R.TARGET_BLOBS[3], R.TARGET_BLOBS[4], "target_cache_index.json",
        "attempts/smoke.json", "smoke.json", "attempts/preflight.json",
        "resource_preflight.json", "training_receipts.json",
        "attempts/evaluation.json", "evaluation/result.json",
        "evaluation/occupancy.json",
    }
    for seed in R.C.FROZEN_SEEDS:
        expected.update({
            f"attempts/train_seed_{seed}.json",
            f"training/seed_{seed}/seed_{seed}_rgb_four_step_epoch21.pt",
            f"training/seed_{seed}/checkpoint_receipts.jsonl",
            f"training/seed_{seed}/training_receipt.json",
            f"evaluation/prediction_ledgers/seed_{seed}.jsonl",
        })
    attempt_paths = {
        "attempts/smoke.json", "attempts/preflight.json",
        "attempts/evaluation.json",
        *{f"attempts/train_seed_{seed}.json" for seed in R.C.FROZEN_SEEDS},
    }
    for relative in expected - attempt_paths:
        _write_read_only(runtime / relative)
    for name in ("smoke", "preflight"):
        _write_read_only(runtime / "attempts" / f"{name}.json", R.json.dumps({
            "schema": "go2_rgb_control_history_four_step_attempt_v1",
            "stage": name,
            "four_step_contract_digest": "c" * 64,
        }) + "\n")
    for seed in R.C.FROZEN_SEEDS:
        name = f"train_seed_{seed}"
        _write_read_only(runtime / "attempts" / f"{name}.json", R.json.dumps({
            "schema": "go2_rgb_control_history_four_step_attempt_v1",
            "stage": name,
            "four_step_contract_digest": "c" * 64,
        }) + "\n")
    _write_read_only(runtime / "attempts/evaluation.json", R.json.dumps({
        "schema": "go2_rgb_control_history_four_step_evaluation_attempt_v1",
        "resumable": False,
        "retry_authorised": False,
    }) + "\n")

    monkeypatch.setattr(R, "runtime_root", lambda: runtime)
    monkeypatch.setattr(R, "require_contract", lambda: contract)
    monkeypatch.setattr(R, "environment_record", lambda require_exact=False: {})
    monkeypatch.setattr(R, "validate_common_manifest", lambda: {
        "counts": {"train": 3_854, "checkpoint_selection": 466},
        "family_counts": {}, "exclusion_counts": {},
        "common_rows_digest": "m" * 64, "manifest_digest": "n" * 64,
    })
    monkeypatch.setattr(R, "validate_training_input_files", lambda hash_files=False: {
        "verification_digest": "i" * 64})
    monkeypatch.setattr(R, "validate_target_cache_index", lambda: {
        "target_cache_index_digest": "t" * 64, "wall_seconds": 1.0})
    monkeypatch.setattr(R, "validate_smoke_receipt", lambda: {
        "smoke_digest": "k" * 64})
    monkeypatch.setattr(R, "validate_preflight_receipt", lambda: {
        "preflight_digest": "p" * 64, "wall_seconds_per_epoch": 2.0})
    monkeypatch.setattr(R, "validate_training_receipt_set", lambda: {
        "training_receipt_set_digest": "r" * 64,
        "receipt_digests": [str(index) * 64 for index in range(1, 9)],
        "total_wall_seconds": 3.0,
    })
    monkeypatch.setattr(R, "validate_evaluation_result", lambda: {
        "result_digest": "e" * 64,
        "occupancy_co_outcome": {"occupancy_digest": "o" * 64},
        "interpretation": {"useful": False},
        "primary_H4_equal_family": {}, "runtime_seconds": 4.0,
    })

    terminal = R.validate_stage(SimpleNamespace())
    assert terminal["classification"] == (
        "COMPLETE_FOUR_STEP_ROLLOUT_OBJECTIVE_RESULT")
    assert terminal["historical_control_comparability"] == {
        "historical_control_train_rows": 3922,
        "new_four_step_train_rows": 3854,
        "row_difference": 68,
        "historical_controls_sample_matched": False,
        "historical_controls_retrained_or_reselected": False,
    }
    assert terminal["predictor_utility_scoring_or_shards_opened"] is False
    assert terminal["final_200_state_corpus_generated"] is False
    assert terminal["nothing_remains_running"] is True
    assert terminal["terminal_digest"] == R._digest({
        key: value for key, value in terminal.items() if key != "terminal_digest"})
    assert (runtime / "terminal.json").stat().st_mode & 0o777 == 0o444
    with pytest.raises(RuntimeError, match="second validation forbidden"):
        R.validate_stage(SimpleNamespace())


def test_failure_terminal_is_immutable_and_does_not_authorise_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    contract_path = runtime / "contract.json"
    _write_read_only(contract_path, R.json.dumps({
        "contract_digest": "c" * 64,
        "source_closure": {"source_repository_commit": "1" * 40},
    }) + "\n")
    monkeypatch.setattr(R, "runtime_root", lambda: runtime)
    monkeypatch.setattr(R.C, "contract_path", lambda root=R.C.ROOT: contract_path)
    monkeypatch.setattr(R, "_mem_total_bytes", lambda: 100)
    monkeypatch.setattr(R, "_mem_available_bytes", lambda: 50)
    monkeypatch.setattr(R.torch.cuda, "is_available", lambda: False)
    terminal = R.record_failure_terminal("smoke", RuntimeError("fixture failure"))
    assert terminal is not None
    assert terminal["classification"] == "INVALID_SMOKE"
    assert terminal["failed_stage"] == "smoke"
    assert terminal["exception_type"] == "RuntimeError"
    assert terminal["retry_resume_or_replacement_authorised"] is False
    assert terminal["automatic_follow_on_experiment_started"] is False
    assert terminal["predictor_utility_or_final_corpus_access"] is False
    assert terminal["nothing_remains_running"] is True
    assert terminal["terminal_digest"] == R._digest({
        key: value for key, value in terminal.items() if key != "terminal_digest"})
    terminal_path = runtime / "terminal.json"
    assert terminal_path.stat().st_mode & 0o777 == 0o444
    original = terminal_path.read_bytes()
    assert R.record_failure_terminal("smoke", RuntimeError("second")) is None
    assert terminal_path.read_bytes() == original


def test_main_refuses_every_invocation_after_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _write_read_only(runtime / "terminal.json")
    called = []
    monkeypatch.setattr(R, "runtime_root", lambda: runtime)
    monkeypatch.setitem(R.STAGES, "validate", lambda args: called.append(args))
    assert R.main(["validate"]) == 2
    assert called == []
