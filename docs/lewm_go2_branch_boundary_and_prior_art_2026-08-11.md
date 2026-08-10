# Canonical branch-capture boundary; prior branch infrastructure found incompatible

Date: 2026-08-11
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.**

Frozen baseline `081eb3d` and all listed digests preserved. No world-model
checkpoint loaded, no scorer trained, no scorer-fit or evaluation corpus generated,
no frozen result altered.

---

# 1. Canonical branch-capture boundary — `1faae05f843e6f02f0f354c63ab3bcad9404111140146b1355d025da3d0c7a92`

**One** canonical point: immediately before the first tick of candidate action
block 0, at a command-block boundary.

**Must hold before capture:** the visual and proprioceptive history for the branch
state is fixed and emitted; the snapshot-time goal/landmark binding is fixed and
recorded; the previous applied post-slew command is known; low-level controller
state is internally consistent.

**Must not yet have happened:** any tick of the candidate applied; any candidate
modification of slew-limiter state; any policy or simulator advance past the
boundary.

## Asserted phase invariants

| field | expected | rationale |
|---|---|---|
| command-block tick | **0** | the boundary is the first of a 5-tick block |
| low-level decimation phase | **0** | policy_dt 0.02 s = 10 physics steps; the boundary aligns to a policy step |
| camera / proprioception sampling phase | **0** | observations emit at 10 Hz command-tick cadence; the boundary coincides with an emission instant |
| reset / termination / truncation flags | **False** | |
| episode step counter | **(s−1) mod 5 == 0** | the same block alignment the factorial manifest already enforces on every row |
| global step counter | recorded, not constrained | |

**Refusal rule:** snapshot creation is refused outside this boundary. The
implementation must **not** silently normalise, advance or rewind a state into the
expected phase; arbitrary-phase capture needs its own implementation and its own
qualification first. Restoration must leave the branch at exactly this boundary
with no hidden simulation or policy step.

---

# 2. Prior branch infrastructure exists — and is incompatible

While tracing the capture point I found an existing counterfactual-pilot stack:
`lewm/benchmarks/go2_world_model_counterfactual_pilot_v1.py` plus a 3,487-line
collector and a 1,757-line test suite, from commit `aa87bf6`.

**It declares `BRANCH_MECHANISM = "parallel_lockstep_envs_no_restore"`.**

| verdict | **INCOMPATIBLE with design v1.1; must not be reused** |
|---|---|

Reasons, precisely:

1. **It does not snapshot or restore at all** — the mechanism name says so. It
   branches by running parallel environments in lockstep.
2. `_initialize_exact_clones` puts every environment at the **manifest spawn**,
   zeroes `policy._last_actions`, zeroes `runner._last_executed`, and sets
   `_sim_time_ns = 0`. It branches from a **fresh episode start**, not a mid-episode
   state.
3. Design v1.1 requires slew limiting seeded from the **restored previous applied
   command**. A zeroed `_last_executed` cannot supply it — and, per the previous
   pass, `_last_actions` is simultaneously observation dims [33:45] and the applied
   action under latency, so zeroing it changes both.
4. Branch states drawn mid-episode across eight families are unreachable by
   respawning.

This is exactly the situation your earlier standing instruction anticipated: do not
reuse a component merely because it exists. I am reporting it as found and rejected
rather than adapting it.

**One genuinely useful corroboration:** its `_capture_components` enumerates qpos,
DOF velocities, base pose/quat/linear/angular velocity, leg joint position and
velocity, `runner._last_executed` and `policy._last_actions` — an independent
confirmation of the field set in inventory v2, including that `_last_actions` is
live state.

---

# 3. What was NOT done

Parts 2–7 — the snapshot implementation in the deployed adapter, integrity and
corruption tests, omission-sensitivity controls, frozen identity manifests,
deterministic replay qualification, branch-order invariance, and the 20-state
oracle pilot — **were not executed**.

The gating chain is sequential and each stage depends on a working snapshot in
`GenesisGo2PPOPolicy` and the rollout harness. That implementation does not exist,
and the prior art cannot be adapted into it for the reasons above. I stopped at the
boundary specification rather than produce a partial snapshot that would pass its
own tests while omitting state — the failure mode the last two passes were
specifically built to prevent.

Consequently there are **no** integrity, corruption, omission-sensitivity, replay,
branch-order, pilot, identifiability or spatial-coverage results, and **no measured
runtime**. The cost model therefore stays at **≈ 40 h / ~22 GB**, unrevised, since
part 7 permits updating it only from measured execution.

# 4. Blockers

1. **Snapshot capture/restore is unimplemented** in the deployed path. Specification,
   inventory and boundary are now complete and mutually consistent, so this is
   bounded engineering rather than an open question — but it is the whole of the
   remaining gate.
2. **Prior branch infrastructure cannot be reused**, so the implementation starts
   from the deployed adapter rather than from `aa87bf6`.
3. Downstream and unchanged: no leakage-free predictor score; no H=2–4 spatial
   labels in the temporal corpus.

## Stopping condition

Nothing is running. No scorer-fit or evaluation corpus generated, no scorer trained,
no predictor scored or retrained, no frozen result altered.
