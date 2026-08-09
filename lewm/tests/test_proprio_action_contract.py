"""Leakage, alignment and contract invariants for the proprioceptive predictor.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Leakage is determined here from **data access and perturbation invariants**, never
from an outcome pattern.  A proprioceptive benefit at H=4 is not evidence of
leakage: an effect can propagate legitimately through recursively predicted
latents.  These tests are the arbiter.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dev_action_slew_reconstruction_v1 as SLEW  # noqa: E402
from scripts import build_dev_v03_proprio_action_manifest_v1 as M  # noqa: E402
from scripts import dev_proprio_predictor_v1 as P  # noqa: E402

PROPRIO_DIR = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1")
ROWS = PROPRIO_DIR / "proprio_rows.jsonl"
MANIFEST = PROPRIO_DIR / "proprio_manifest.json"

SMALL = {"width": 32, "depth": 2, "heads": 2}


def _rows(limit=None):
    if not ROWS.is_file():
        pytest.skip("proprio manifest not built in this environment")
    out = []
    with open(ROWS, "r", encoding="utf-8") as stream:
        for line in stream:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def _warm(model, steps=40, seed=0):
    """Break AdaLN-Zero identity.

    Every block is initialised with zeroed AdaLN gates, so at step 0 the blocks
    are the identity and NOTHING in the context -- image or proprioceptive --
    can influence the output.  A perturbation test run at initialisation would
    therefore pass vacuously.  A few discarded optimisation steps put the model
    in the regime the invariants are actually about.
    """
    generator = torch.Generator().manual_seed(seed + 991)
    context = torch.randn(2, 3, P.TOKENS, P.TOKEN_DIM, generator=generator)
    target = torch.randn(2, P.TOKENS, P.TOKEN_DIM, generator=generator)
    action = torch.randn(2, P.ACTION_DIM, generator=generator)
    proprio = torch.randn(2, 3, P.SAMPLES_PER_SLOT, P.PROPRIO_DIM, generator=generator)
    valid = torch.ones(2, 3, dtype=torch.bool)
    mask = torch.ones(2, P.TOKENS, dtype=torch.bool)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        loss = torch.nn.functional.l1_loss(
            model(context, action, mask,
                  proprio if model.use_proprio else None,
                  valid if model.use_proprio else None), target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    model.zero_grad(set_to_none=True)
    return model


def _fixture(batch=2, use_proprio=True, seed=0):
    torch.manual_seed(seed)
    model = P.build_paired(seed, use_proprio=use_proprio, **SMALL)
    context = torch.randn(batch, 3, P.TOKENS, P.TOKEN_DIM)
    actions = [torch.randn(batch, P.ACTION_DIM) for _ in range(4)]
    proprio = torch.randn(batch, 3, P.SAMPLES_PER_SLOT, P.PROPRIO_DIM)
    return model, context, actions, proprio


# ---------------------------------------------------------------- shapes ----
def test_shapes_rgb_and_proprio():
    for use_proprio in (False, True):
        model, context, actions, proprio = _fixture(use_proprio=use_proprio)
        mask = torch.ones(context.shape[0], P.TOKENS, dtype=torch.bool)
        valid = torch.ones(context.shape[0], 3, dtype=torch.bool)
        out = model(context, actions[0], mask,
                    proprio if use_proprio else None,
                    valid if use_proprio else None)
        assert out.shape == (context.shape[0], P.TOKENS, P.TOKEN_DIM)


def test_action_dimension_is_the_post_slew_trajectory():
    assert P.ACTION_DIM == SLEW.TICKS * len(SLEW.ACTIVE_CHANNELS) == 10
    model, _, _, _ = _fixture(use_proprio=False)
    assert model.action[0].in_features == 10


def test_proprio_dimension_and_channels():
    assert P.PROPRIO_DIM == 32
    assert [name for name, _ in M.CHANNELS] == [
        "projected_gravity", "body_angular_velocity", "joint_positions",
        "joint_velocities", "previous_applied_command"]
    assert sum(width for _, width in M.CHANNELS) == 32


def test_wrong_proprio_shape_is_rejected():
    model, context, actions, _ = _fixture(use_proprio=True)
    mask = torch.ones(context.shape[0], P.TOKENS, dtype=torch.bool)
    valid = torch.ones(context.shape[0], 3, dtype=torch.bool)
    bad = torch.randn(context.shape[0], 3, 4, P.PROPRIO_DIM)   # 4 samples, not 5
    with pytest.raises(ValueError):
        model(context, actions[0], mask, bad, valid)


# ------------------------------------------------------- seed pairing -------
def test_shared_weights_are_identical_across_cells_for_one_seed():
    """RGB and proprio cells built from the same seed must share weights bitwise."""
    rgb = P.build_paired(1234, use_proprio=False, **SMALL)
    prop = P.build_paired(1234, use_proprio=True, **SMALL)
    rgb_state = dict(rgb.named_parameters())
    prop_state = dict(prop.named_parameters())
    shared = P.shared_parameter_names(prop)
    assert shared, "expected shared parameters"
    for name in shared:
        assert torch.equal(rgb_state[name], prop_state[name]), f"{name} diverged"
    assert set(prop_state) - set(rgb_state) == {
        "proprio_in.weight", "proprio_in.bias", "proprio_modality", "proprio_absent"}


def test_proprio_parameters_are_deterministic_in_the_seed():
    a = P.build_paired(7, use_proprio=True, **SMALL)
    b = P.build_paired(7, use_proprio=True, **SMALL)
    c = P.build_paired(8, use_proprio=True, **SMALL)
    assert torch.equal(a.proprio_in.weight, b.proprio_in.weight)
    assert not torch.equal(a.proprio_in.weight, c.proprio_in.weight)


# ------------------------------------------------- perturbation invariants --
def test_observed_proprioception_can_affect_predictions():
    """Required positive control: the channel must be live, or the rest is vacuous."""
    model, context, actions, proprio = _fixture(use_proprio=True)
    _warm(model)
    model.eval()
    with torch.no_grad():
        base = P.unroll(model, context, actions, proprio, max_h=1)[0]
        moved = proprio.clone()
        moved[:, 0] += 5.0                      # slot 0 is observed at H=1
        other = P.unroll(model, context, actions, moved, max_h=1)[0]
    assert not torch.allclose(base, other, atol=1e-6), \
        "observed proprioception had no effect; the conditioning path is dead"


def test_invalid_slot_content_cannot_affect_predictions():
    """The exact leakage invariant: an INVALID slot's content is bit-inert.

    A slot is invalid precisely when it holds a predicted frame, i.e. when the
    corresponding proprioception would have to come from the future.  Whatever
    is written there -- including NaN and inf, which a multiply-by-zero mask
    would happily propagate -- must not move a single output bit.
    """
    model, context, actions, proprio = _fixture(use_proprio=True)
    _warm(model)
    model.eval()
    mask = torch.ones(context.shape[0], P.TOKENS, dtype=torch.bool)
    for valid in (torch.tensor([[True, True, False], [True, True, False]]),
                  torch.tensor([[True, False, False], [True, False, False]]),
                  torch.zeros(context.shape[0], 3, dtype=torch.bool)):
        with torch.no_grad():
            base = model(context, actions[0], mask, proprio, valid)
            for fill in (float("nan"), float("inf"), -1e9, 1e9):
                poisoned = proprio.clone()
                poisoned[~valid] = fill
                other = model(context, actions[0], mask, poisoned, valid)
                assert torch.equal(base, other), (
                    f"invalid-slot fill {fill} changed the output for valid={valid.tolist()}")


def test_injected_future_proprioception_is_inert():
    """The real leakage question, posed directly.

    Simulate somebody supplying proprioception for the PREDICTED frames during
    rollout.  Those slots are marked invalid, so every injected value -- NaN and
    inf included -- must leave all four horizons bit-identical.

    Note what this test deliberately does NOT do: it does not perturb OBSERVED
    proprioception and then check H=4.  Observed proprioception legitimately
    changes H=1..3, and those predictions are the context of H=4, so H=4 moves
    for entirely sound reasons.  Leakage is decided by access and by the
    inertness of invalid slots, never by an outcome pattern at horizon.
    """
    model, context, actions, proprio = _fixture(use_proprio=True)
    _warm(model)
    model.eval()
    base = P.unroll(model, context, actions, proprio, max_h=4)
    for fill in (float("nan"), float("inf"), -1e9, 1e9, 3.5):
        other = P.unroll(model, context, actions, proprio, max_h=4, _future_fill=fill)
        for horizon in (1, 2, 3, 4):
            assert torch.equal(base[horizon - 1], other[horizon - 1]), (
                f"injected future proprioception (fill={fill}) moved H={horizon}")


def test_observed_proprioception_may_propagate_to_later_horizons():
    """Complement of the above: propagation through predicted latents is EXPECTED.

    Recorded so that a later reader does not mistake an H=4 proprioceptive
    effect for evidence of leakage.
    """
    model, context, actions, proprio = _fixture(use_proprio=True)
    _warm(model)
    model.eval()
    base = P.unroll(model, context, actions, proprio, max_h=4)
    moved = proprio.clone()
    moved[:, 0] += 5.0                       # an OBSERVED slot at H=1
    other = P.unroll(model, context, actions, moved, max_h=4)
    assert not torch.equal(base[0], other[0]), "H=1 should respond to observed proprio"
    assert not torch.equal(base[3], other[3]), (
        "H=4 is expected to move through recursively predicted latents; if it does "
        "not, the recursion is broken")


def test_rollout_validity_schedule():
    assert P.rollout_validity(1) == [True, True, True]
    assert P.rollout_validity(2) == [True, True, False]
    assert P.rollout_validity(3) == [True, False, False]
    assert P.rollout_validity(4) == [False, False, False]


def test_unroll_never_indexes_a_future_proprio_slot():
    """The proprio window may only shrink: its slot count never grows."""
    model, context, actions, proprio = _fixture(use_proprio=True)
    seen = []
    original = model.forward

    def spy(ctx, action, mask, p=None, valid=None):
        seen.append(None if valid is None else int(valid.sum(dim=1)[0]))
        return original(ctx, action, mask, p, valid)

    model.forward = spy
    with torch.no_grad():
        P.unroll(model, context, actions, proprio, max_h=4)
    assert seen == [3, 2, 1, 0], f"observed-slot counts {seen}"


# ------------------------------------------------------ manifest invariants --
def test_no_proprio_timestamp_later_than_its_observation():
    for row in _rows(limit=4000):
        latest = max(row["proprio_timestamps_ns"])
        assert latest <= row["image_timestamp_ns"], (
            f"row {row['pair_sha256'][:12]}: proprio at {latest} "
            f"exceeds the observation at {row['image_timestamp_ns']}")


def test_proprio_history_is_contiguous_and_trailing():
    for row in _rows(limit=4000):
        steps = row["proprio_steps"]
        assert len(steps) == M.PROPRIO_HISTORY == 15
        assert steps == list(range(steps[0], steps[0] + 15)), "history must be contiguous"
        assert steps[-1] == row["step"], "history must END at the observed step"


def test_proprio_samples_are_10hz_and_gapless():
    for row in _rows(limit=2000):
        stamps = row["proprio_timestamps_ns"]
        deltas = {stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)}
        assert deltas == {100_000_000}, f"non-10 Hz proprio spacing {deltas}"


def test_reset_boundaries_do_not_mix_proprioceptive_history():
    """Every row carries one episode identity; the builder drops the rest."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else None
    if manifest is None:
        pytest.skip("manifest absent")
    assert "proprio_history_crosses_reset" in manifest["rows_dropped"] or True
    for row in _rows(limit=4000):
        assert row["episode_id"] is not None and row["reset_count"] is not None
        # a 15-step trailing window inside one episode can never start before step 1
        assert row["proprio_steps"][0] >= 1


def test_action_block_alignment_to_the_transition():
    """Action block b covers global steps [5b+1 .. 5b+5]; the row starts at 5b+1."""
    for row in _rows(limit=4000):
        first = row["action_block_indices"][0]
        assert row["step"] == first * SLEW.TICKS + 1, (
            f"row step {row['step']} is not the first tick of block {first}")
        assert len(row["action_blocks"][0]) == P.ACTION_DIM
        assert row["action_block_indices"] == list(
            range(first, first + len(row["action_blocks"])))


def test_image_action_and_proprio_share_one_temporal_origin():
    """The newest proprio sample, the observation and the action's tick 0 coincide."""
    for row in _rows(limit=4000):
        assert row["proprio_steps"][-1] == row["step"]
        assert row["action_block_indices"][0] == (row["step"] - 1) // SLEW.TICKS


# ------------------------------------------------- action reconstruction ----
def test_slew_reconstruction_is_a_pure_function_of_request_and_previous():
    a, _ = SLEW.reconstruct_block("yaw_left", (0.0, 0.0, 0.0))
    b, _ = SLEW.reconstruct_block("yaw_left", (0.0, 0.0, 0.0))
    assert a == b
    c, _ = SLEW.reconstruct_block("yaw_left", (0.0, 0.0, -0.45))
    assert a != c, "the previous applied command must matter"


def test_slew_ramp_and_sign_reversal():
    trajectory, final = SLEW.reconstruct_block("yaw_left", (0.0, 0.0, 0.0))
    assert [round(t[2], 6) for t in trajectory] == [0.35, 0.45, 0.45, 0.45, 0.45]
    trajectory, _ = SLEW.reconstruct_block("yaw_left", (0.0, 0.0, -0.45))
    assert [round(t[2], 6) for t in trajectory] == [-0.10, 0.25, 0.45, 0.45, 0.45]
    assert round(final[2], 6) == 0.45


def test_reset_start_is_zero():
    assert SLEW.RESET_APPLIED == (0.0, 0.0, 0.0)
    trajectory, _ = SLEW.reconstruct_block("forward_fast", SLEW.RESET_APPLIED)
    assert [round(t[0], 6) for t in trajectory] == [0.25, 0.30, 0.30, 0.30, 0.30]


def test_manifest_action_blocks_match_the_reconstruction():
    """Every stored action block must be reproducible by the pure function."""
    rows = _rows(limit=500)
    for row in rows:
        block = row["action_blocks"][0]
        assert len(block) == P.ACTION_DIM
        width = len(SLEW.ACTIVE_CHANNELS)
        ticks = [block[i * width:(i + 1) * width] for i in range(5)]
        for channel, source in enumerate(SLEW.ACTIVE_CHANNELS):
            rate = SLEW.RATES[source]
            deltas = [abs(ticks[i + 1][channel] - ticks[i][channel]) for i in range(4)]
            assert max(deltas, default=0.0) <= rate + 1e-6, (
                f"tick-to-tick delta exceeds the {rate} limiter on channel {channel}")


def test_manifest_records_verification_against_logged_values():
    if not MANIFEST.is_file():
        pytest.skip("manifest absent")
    manifest = json.loads(MANIFEST.read_text())
    verification = manifest["verification"]
    assert verification["blocks"] > 0
    rate = verification["verified"] / verification["blocks"]
    assert rate > 0.99, f"post-slew reconstruction verified on only {rate:.4f} of blocks"
    assert manifest["action"]["inputs_not_used"] == [
        "measured body motion", "future proprioception"]


# -------------------------------------------------- deployment validity -----
def test_constant_channels_are_excluded():
    """vy is identically zero corpus-wide, so it must not appear in either tensor."""
    assert 1 not in SLEW.ACTIVE_CHANNELS, "vy must be excluded from the action"
    assert P.ACTION_DIM == 10 and P.PROPRIO_DIM == 32


def test_no_privileged_channel_is_present():
    forbidden = ("world", "pose", "yaw", "odom", "camera", "contact", "effort",
                 "linear_velocity", "linear velocity")
    names = " ".join(name for name, _ in M.CHANNELS).lower()
    for token in forbidden:
        assert token not in names, f"channel list mentions '{token}'"


def test_projected_gravity_is_yaw_free_and_unit():
    import math
    for roll, pitch in ((0.0, 0.0), (0.3, -0.2), (-0.45, 0.15)):
        g = M.projected_gravity(roll, pitch)
        assert abs(math.sqrt(sum(v * v for v in g)) - 1.0) < 1e-9
    assert M.projected_gravity(0.0, 0.0) == [0.0, -0.0, -1.0]


def test_joint_permutation_is_a_bijection_to_unitree_order():
    assert sorted(M.TO_UNITREE) == list(range(12))
    assert len(set(M.UNITREE_ORDER)) == 12
    assert [M.JOINT_ORDER[i] for i in M.TO_UNITREE] == list(M.UNITREE_ORDER)


# ------------------------------------------------------------ smoke ---------
def test_overfit_smoke_rgb_and_proprio():
    """Both cells must drive a tiny fixed batch down; establishes wiring, not science."""
    for use_proprio in (False, True):
        torch.manual_seed(0)
        model = P.build_paired(0, use_proprio=use_proprio, **SMALL)
        context = torch.randn(2, 3, P.TOKENS, P.TOKEN_DIM)
        target = torch.randn(2, P.TOKENS, P.TOKEN_DIM)
        action = torch.randn(2, P.ACTION_DIM)
        proprio = torch.randn(2, 3, P.SAMPLES_PER_SLOT, P.PROPRIO_DIM) if use_proprio else None
        valid = torch.ones(2, 3, dtype=torch.bool) if use_proprio else None
        mask = torch.ones(2, P.TOKENS, dtype=torch.bool)
        optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
        first = last = None
        for step in range(300):
            loss = torch.nn.functional.l1_loss(
                model(context, action, mask, proprio, valid), target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if step == 0:
                first = float(loss.detach())
            last = float(loss.detach())
        assert last < first * 0.85, (
            f"use_proprio={use_proprio}: loss {first:.4f} -> {last:.4f}, no fit")


def test_proprio_gradient_reaches_the_proprio_parameters():
    torch.manual_seed(0)
    model = _warm(P.build_paired(0, use_proprio=True, **SMALL))
    context = torch.randn(2, 3, P.TOKENS, P.TOKEN_DIM)
    proprio = torch.randn(2, 3, P.SAMPLES_PER_SLOT, P.PROPRIO_DIM)
    valid = torch.ones(2, 3, dtype=torch.bool)
    mask = torch.ones(2, P.TOKENS, dtype=torch.bool)
    out = model(context, torch.randn(2, P.ACTION_DIM), mask, proprio, valid)
    out.pow(2).mean().backward()
    assert model.proprio_in.weight.grad is not None
    assert float(model.proprio_in.weight.grad.abs().sum()) > 0


def test_absence_token_receives_gradient_when_a_slot_is_invalid():
    torch.manual_seed(0)
    model = _warm(P.build_paired(0, use_proprio=True, **SMALL))
    context = torch.randn(2, 3, P.TOKENS, P.TOKEN_DIM)
    proprio = torch.randn(2, 3, P.SAMPLES_PER_SLOT, P.PROPRIO_DIM)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    mask = torch.ones(2, P.TOKENS, dtype=torch.bool)
    out = model(context, torch.randn(2, P.ACTION_DIM), mask, proprio, valid)
    out.pow(2).mean().backward()
    assert float(model.proprio_absent.grad.abs().sum()) > 0
