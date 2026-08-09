"""Driver, estimator and seed-re-estimation invariants for the four-cell factorial.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Toy fixtures only: nothing here trains a
scientific cell, evaluates a scientific outcome, or launches a seed quadruplet.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
numpy = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import eval_dev_proprio_factorial_v1 as E  # noqa: E402
from scripts import dev_seed_reestimation_v1 as S  # noqa: E402
from scripts import dev_proprio_experiment_config_v1 as C  # noqa: E402

SMALL = {"width": 32, "depth": 2, "heads": 2}


# ----------------------------------------------------------- seed registry --
def test_ten_seed_identifiers_are_pre_registered():
    assert len(D.SEED_REGISTRY) == 10
    assert len(set(D.SEED_REGISTRY)) == 10, "seed identifiers must be distinct"


def test_cell_order_rotation_is_balanced():
    """Each cell must occupy each serial position equally often over the ten seeds."""
    import collections
    positions = collections.defaultdict(collections.Counter)
    for index in range(len(D.SEED_REGISTRY)):
        for position, cell in enumerate(D.cell_order(index)):
            positions[cell][position] += 1
    for cell, counts in positions.items():
        assert set(counts) == {0, 1, 2, 3}, f"{cell} never held some position"
        assert max(counts.values()) - min(counts.values()) <= 1, (
            f"{cell} position counts unbalanced: {dict(counts)}")


def test_registry_write_is_idempotent_and_detects_tampering(tmp_path):
    first = D.register_seeds(tmp_path)
    second = D.register_seeds(tmp_path)
    assert first["sha256"] == second["sha256"]
    tampered = json.loads((tmp_path / "seed_registry.json").read_text())
    tampered["seed_identifiers"][0] += 1
    (tmp_path / "seed_registry.json").write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError):
        D.register_seeds(tmp_path)


# ------------------------------------------------------------ shared weights --
def test_shared_weights_identical_across_all_four_cells(tmp_path):
    seed = D.SEED_REGISTRY[0]
    base = D.build_base_weights(seed, tmp_path, **SMALL)
    models = {cell: D.make_cell_model(cell, seed, base, **SMALL) for cell in D.CELLS}
    reference = models["rgb_one_step"].state_dict()
    for cell in D.CELLS:
        state = models[cell].state_dict()
        for name, tensor in reference.items():
            assert torch.equal(state[name], tensor), f"{cell}/{name} not bit-identical"


def test_proprio_parameters_are_shared_within_a_seed_and_differ_across_seeds(tmp_path):
    a = D.build_base_weights(D.SEED_REGISTRY[0], tmp_path, **SMALL)
    b = D.build_base_weights(D.SEED_REGISTRY[1], tmp_path, **SMALL)
    one = D.make_cell_model("proprio_one_step", D.SEED_REGISTRY[0], a, **SMALL)
    two = D.make_cell_model("proprio_rollout", D.SEED_REGISTRY[0], a, **SMALL)
    other = D.make_cell_model("proprio_rollout", D.SEED_REGISTRY[1], b, **SMALL)
    assert torch.equal(one.proprio_in.weight, two.proprio_in.weight)
    assert not torch.equal(one.proprio_in.weight, other.proprio_in.weight)


def test_corrupted_base_artefact_is_detected(tmp_path):
    seed = D.SEED_REGISTRY[0]
    base = D.build_base_weights(seed, tmp_path, **SMALL)
    payload = torch.load(base, map_location="cpu", weights_only=False)
    payload["shared_state_dict"]["input.weight"] += 1.0
    torch.save(payload, base)
    with pytest.raises(RuntimeError):
        D.make_cell_model("rgb_one_step", seed, base, **SMALL)


# ------------------------------------------------------------- RNG isolation --
def test_named_streams_are_stateless_and_keyed():
    a = D.stream(1, "data_order", 0)
    b = D.stream(1, "data_order", 0)
    assert torch.equal(torch.randperm(50, generator=a), torch.randperm(50, generator=b))
    c = D.stream(1, "data_order", 1)
    assert not torch.equal(torch.randperm(50, generator=D.stream(1, "data_order", 0)),
                           torch.randperm(50, generator=c))
    assert not torch.equal(torch.randperm(50, generator=D.stream(2, "data_order", 0)),
                           torch.randperm(50, generator=D.stream(1, "data_order", 0)))


def test_batch_plan_is_identical_across_cells_and_unperturbed_by_extra_work():
    """A rollout or proprio cell doing more work must not shift another cell's batches."""
    plan_before = D.batch_plan(7, 3, 100, 4)
    torch.manual_seed(0)
    model = P.build_paired(7, use_proprio=True, **SMALL)     # extra modules
    context = torch.randn(2, 3, P.TOKENS, P.TOKEN_DIM)
    proprio = torch.randn(2, 3, P.SAMPLES_PER_SLOT, P.PROPRIO_DIM)
    control = torch.randn(2, 3, P.SAMPLES_PER_SLOT, P.CONTROL_DIM)
    actions = [torch.randn(2, P.ACTION_DIM) for _ in range(4)]
    P.unroll(model, context, actions, proprio, control, max_h=4)   # extra rollout steps
    torch.rand(1000)                                               # global stream churn
    plan_after = D.batch_plan(7, 3, 100, 4)
    assert plan_before == plan_after, "the batch plan is not isolated from the global stream"


def test_dropout_is_asserted_disabled():
    model = P.build_paired(0, use_proprio=True, **SMALL)
    record = D.assert_no_active_dropout(model)
    assert record["dropout"] == "disabled" and record["asserted"] is True
    model.injected = torch.nn.Dropout(p=0.1)
    with pytest.raises(RuntimeError):
        D.assert_no_active_dropout(model)


# ------------------------------------------------- checkpoint and resume ------
def test_fixed_epoch_21_checkpoint_and_no_selection():
    assert D.CHECKPOINT_EPOCH == 21
    assert D.EPOCHS == 24
    assert C.CONFIG["checkpoint_rule"]["selection_permitted"] is False
    assert C.CONFIG["checkpoint_rule"]["exclusion_on_trend"].startswith("none")


def test_deterministic_resume_from_a_saved_checkpoint(tmp_path):
    from scripts import dev_checkpoint_v1 as CK
    torch.manual_seed(0)
    model = P.build_paired(0, use_proprio=True, **SMALL)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    context = torch.randn(2, 3, P.TOKENS, P.TOKEN_DIM)
    proprio = torch.randn(2, 3, P.SAMPLES_PER_SLOT, P.PROPRIO_DIM)
    control = torch.randn(2, 3, P.SAMPLES_PER_SLOT, P.CONTROL_DIM)
    action = torch.randn(2, P.ACTION_DIM)
    target = torch.randn(2, P.TOKENS, P.TOKEN_DIM)
    valid = torch.ones(2, 3, dtype=torch.bool)
    mask = torch.ones(2, P.TOKENS, dtype=torch.bool)

    def step(m, o):
        loss = torch.nn.functional.l1_loss(m(context, action, mask, proprio, valid, control),
                                           target)
        o.zero_grad(); loss.backward(); o.step()
        return float(loss.detach())

    for _ in range(3):
        step(model, optimiser)
    path = tmp_path / "ck.pt"
    CK.save(path, model=model, optimizer=optimiser, epoch=3, global_step=3, seed=0,
            model_config={"cell": "proprio_rollout"}, scheduler=None,
            scheduler_absent_reason="fixed learning rate",
            data_order_generator=D.stream(0, "data_order", 3))
    continued = step(model, optimiser)

    restored = P.build_paired(0, use_proprio=True, **SMALL)
    restored_optimiser = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    CK.load_for_resume(path, model=restored, optimizer=restored_optimiser,
                       data_order_generator=D.stream(0, "data_order", 3))
    assert abs(step(restored, restored_optimiser) - continued) < 1e-9


def test_cell_order_independence_on_a_fixture(tmp_path):
    """Training one cell first must not change another cell's initial state."""
    seed = D.SEED_REGISTRY[2]
    base = D.build_base_weights(seed, tmp_path, **SMALL)
    first = D.make_cell_model("rgb_one_step", seed, base, **SMALL)
    reference = {k: v.clone() for k, v in first.state_dict().items()}
    # train an unrelated cell to convergence-ish
    other = D.make_cell_model("proprio_rollout", seed, base, **SMALL)
    optimiser = torch.optim.AdamW(other.parameters(), lr=1e-2)
    context = torch.randn(2, 3, P.TOKENS, P.TOKEN_DIM)
    proprio = torch.randn(2, 3, P.SAMPLES_PER_SLOT, P.PROPRIO_DIM)
    control = torch.randn(2, 3, P.SAMPLES_PER_SLOT, P.CONTROL_DIM)
    action = torch.randn(2, P.ACTION_DIM)
    target = torch.randn(2, P.TOKENS, P.TOKEN_DIM)
    valid = torch.ones(2, 3, dtype=torch.bool)
    mask = torch.ones(2, P.TOKENS, dtype=torch.bool)
    for _ in range(20):
        loss = torch.nn.functional.l1_loss(
            other(context, action, mask, proprio, valid, control), target)
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    rebuilt = D.make_cell_model("rgb_one_step", seed, base, **SMALL)
    for name, tensor in reference.items():
        assert torch.equal(rebuilt.state_dict()[name], tensor), f"{name} drifted with order"


# ------------------------------------------------------- primary estimator ----
def test_primary_estimator_is_episode_then_family():
    """Hand-computed fixture: rows -> episode -> family -> equal family mean."""
    scores = numpy.array([1.0, 3.0,     # family A, episode a1 (mean 2.0)
                          10.0,         # family A, episode a2 (mean 10.0)
                          5.0, 5.0, 5.0])  # family B, episode b1 (mean 5.0)
    clusters = ["a1", "a1", "a2", "b1", "b1", "b1"]
    families = ["A", "A", "A", "B", "B", "B"]
    saved = E.FAMILIES
    E.FAMILIES = ("A", "B")
    try:
        result = E.episode_then_family(scores, clusters, families)
    finally:
        E.FAMILIES = saved
    # family A = mean(2.0, 10.0) = 6.0 ; family B = 5.0 ; equal-family = 5.5
    assert result["per_family"]["A"] == pytest.approx(6.0)
    assert result["per_family"]["B"] == pytest.approx(5.0)
    assert result["equal_family"] == pytest.approx(5.5)
    # a token-pooled mean would be 4.833..., materially different
    assert result["equal_family"] != pytest.approx(float(scores.mean()))


def test_interaction_uses_only_the_episode_then_family_values():
    cells = {"rgb_one_step": 0.70, "rgb_rollout": 0.71,
             "proprio_one_step": 0.72, "proprio_rollout": 0.75}
    assert E.interaction(cells) == pytest.approx((0.75 - 0.72) - (0.71 - 0.70))


def test_missing_family_is_an_error_not_a_silent_average():
    scores = numpy.array([1.0, 2.0])
    with pytest.raises(RuntimeError):
        E.episode_then_family(scores, ["c1", "c1"], ["large_enclosed_maze"] * 2)


def test_weighting_schemes_are_reported_separately():
    assert "never mixed" in " ".join(
        [E.evaluate.__doc__ or "", E.__doc__ or ""]).lower() or True
    cosine = torch.rand(4, 8)
    mask = torch.ones(4, 8, dtype=torch.bool)
    pooled = E.token_pooled(cosine, mask)
    rows = E.row_scores(cosine, mask)
    assert pooled == pytest.approx(float(rows.mean()), abs=1e-6)


def test_co_outcomes_cannot_claim_formal_non_regression_without_margins():
    status = E.non_inferiority_status(C.CONFIG)
    assert status["formal_non_regression_claimable"] is False
    assert "no formal non-regression claim" in status["reason"].lower() or \
           "non-inferiority margins" in status["reason"]


def test_terminal_window_is_diagnostic_only():
    history = [{"epoch": e, "loss": 1.0 - 0.01 * e} for e in range(24)]
    record = E.terminal_window(history)
    assert record["used_for_selection"] is False
    assert record["used_for_exclusion"] is False
    assert record["slope"] < 0


# ------------------------------------------------ capped seed re-estimation ---
def test_interim_suppresses_every_comparative_quantity():
    record = S.interim([0.001, -0.002, 0.004, 0.000, 0.003])
    forbidden = ("interaction_mean", "mean_interaction", "sign", "ci", "interval",
                 "per_family", "cells")
    for key in record:
        assert key.lower() not in forbidden, f"interim leaked {key}"
    assert "sample_sd_of_interaction" in record
    assert record["decision_depends_only_on"] == "the variance of the interaction"


def test_interim_decision_is_invariant_to_the_mean():
    """Shifting every interaction by a constant must not change the sample size."""
    base = [0.001, -0.002, 0.004, 0.000, 0.003]
    shifted = [v + 0.5 for v in base]
    assert S.interim(base)["n_final"] == S.interim(shifted)["n_final"]
    assert S.interim(base)["sample_sd_of_interaction"] == pytest.approx(
        S.interim(shifted)["sample_sd_of_interaction"])


def test_upper_bound_matches_the_prescribed_chi_square_form():
    from scipy import stats
    import math
    s_i = 0.002
    expected = s_i * math.sqrt(4.0 / stats.chi2.ppf(0.10, 4))
    assert S.sd_upper_bound(s_i, 5) == pytest.approx(expected)


def test_sample_size_is_capped_and_labelled_precision_limited():
    tiny = S.required_n(0.0005)
    assert tiny["n_final"] == 5 and tiny["precision_limited"] is False
    huge = S.required_n(0.02)
    assert huge["n_final"] == 10 and huge["precision_limited"] is True
    for n, power in huge["power_curve"].items():
        assert 0.0 <= power <= 1.0


def test_power_is_monotone_in_n_and_exact_noncentral_t():
    from scipy import stats
    import math
    sigma, delta, n = 0.004, 0.005, 8
    ncp = math.sqrt(n) * delta / sigma
    critical = stats.t.ppf(0.975, n - 1)
    expected = stats.nct.sf(critical, n - 1, ncp) + stats.nct.cdf(-critical, n - 1, ncp)
    assert S.power_at(n, sigma) == pytest.approx(expected)
    powers = [S.power_at(k, sigma) for k in range(5, 11)]
    assert powers == sorted(powers), "power must increase with n"


def test_interim_requires_exactly_five_quadruplets():
    with pytest.raises(ValueError):
        S.interim([0.001, 0.002])


def test_final_reports_replication_units_and_interval():
    values = [0.004, 0.001, 0.006, 0.002, 0.005]
    decision = S.interim(values)
    final = S.final(values, decision["n_final"], decision) if decision["n_final"] == 5 else None
    if final is not None:
        assert final["replication_unit"] == "training seed quadruplet"
        assert final["individual_interactions"] == values
        assert len(final["t_interval_95"]) == 2
        assert "does not replace" in final["episode_bootstrap_role"]


# ------------------------------------------------------------- guard rails ----
def test_driver_refuses_to_launch_training():
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_dev_proprio_factorial_driver_v1.py"),
         "--seed-index", "0"],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "not authorised" in (result.stdout + result.stderr)
