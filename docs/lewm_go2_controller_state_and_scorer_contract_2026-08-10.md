# Controller-state audit, snapshot gate, and prospective scorer contract

Date: 2026-08-10
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.**

Frozen and untouched: `012cea6`, snapshot schema `b18c49ea…`, design v1.1
`e3176d93…`, and every earlier confirmatory, spatial and audit artefact. No
predictor checkpoint was loaded or scored in this pass.

---

# 1. Controller-statefulness audit

Inspected the loaded checkpoint and config directly, not the name.

| property | finding |
|---|---|
| actor architecture | `MLPModel`, 45 → 512 → 256 → 128 → 12, ELU |
| **recurrent hidden state** | **none** — `actor_state_dict` holds only `mlp.{0,2,4,6}.{weight,bias}` and `distribution.std_param`; no gru/lstm/rnn/hidden parameter exists |
| inference updates hidden tensors | resolved by architecture: there are none |
| runtime observation normalisation | **none** — no running mean/var/rms buffers; `obs_scales` are fixed constants |
| dropout | none |
| command jitter RNG | none (`command_jitter_std = 0.0`) |
| **stochastic action sampling** | **UNRESOLVED** — `GaussianDistribution`, `std_type scalar`, and `distribution.std_param (12,)` is present. If the deployed inference path *samples* rather than taking the mean, policy-side RNG is live state |

A useful consistency check fell out: `lewm_command_bank` in the policy config is
**exactly** the nine primitive set-points used by the frozen action contract, and
`lin_vel_y_range` is `[0.0, 0.0]` — independently confirming the lateral-command
exclusion made earlier from the corpus side.

# 2. State inventory — `82c034d77b02d0291169ea77eb383b460b9328b2f601aabae56f7581dd8e8a77`

Every mutable field classified as GENESIS / HARNESS / RECONSTRUCTED / IRRELEVANT /
UNRESOLVED. Highlights:

- **GENESIS** (verified bit-identical): root pose and orientation, root linear and
  angular velocity, joint positions and velocities, contact/collision solver state,
  environment-object state, solver caches, physics RNG.
- **HARNESS**: **previous applied post-slew command** (critical — it determines the
  first tick of every branch), command-block tick index and sequence id, low-level
  decimation phase, numpy/torch/Python RNG, episode/reset/step counters, beacon and
  landmark state, task-completion and termination state, collision/stuck/contact
  accumulators, camera timing.
- **RECONSTRUCTED**: slew-limiter carry — memoryless given the previous applied
  command.
- **IRRELEVANT**: controller hidden state (feedforward); multi-step observation
  history (a 45-D observation admits none).

## Four fields remain UNRESOLVED, and that gates the work

| field | why unresolved |
|---|---|
| previous policy action (12-D) | 45 = 3 ang-vel + 3 projected gravity + 3 commands + 12 dof-pos + 12 dof-vel + **12 actions** matches exactly, which strongly implies a previous-action buffer — but the observation-assembly code was not located and I will not assume it |
| action-latency / filtering buffers | not located; if a latency queue exists it is live state |
| policy-side RNG | live only if inference samples the Gaussian |
| gait-phase state | no gait phase appears in the 45-D observation, but the harness may keep one |

> Your instruction was explicit: **do not proceed while a state-affecting field
> remains unresolved.** Four are. So parts 2, 3 and 4 — the branch snapshot
> implementation, deterministic replay qualification, and the 20-state pilot —
> were **not started**.

This is not a shortfall I could have engineered around: a previous-action buffer or
a sampling RNG would pass a solver-only replay check and then silently change a
replayed trajectory, which is exactly the failure the qualification exists to
catch. Resolution path: locate the runtime inference wrapper and observation
assembly used by the branch-generation path, re-run this audit, and land each field
in a resolved class before any snapshot is implemented.

# 3–4. Snapshot implementation, replay qualification, oracle pilot — **NOT RUN**

Gated as above. No world-model checkpoint was loaded; no model-side candidate score
was computed. No H=1–4 spatial-label coverage is available because no branch was
generated.

# 5. Prospective scorer contract — `d32118552b6fd373aefab143917bb04e63ffbe196129266a1546affc08f763ff`

Specification only; nothing trained.

`U_hat = 1.0·progress_hat − 2.0·safety_hat + 0.5·completion_hat`, weights reused
unchanged from design v1.1.

- **Consumes the full H=1–4 latent trajectory**, not H=4 alone, because safety and
  completion are path-dependent — a branch that collides at H=2 and recovers by H=4
  is invisible to a terminal-only score. Per-horizon shared encoder, learned
  attention pool over h, then three separate heads.
- The candidate action sequence **is** supplied, in the same 10-D five-tick
  post-slew representation, alongside goal context.
- **Goal binding is explicit**: a landmark identifier plus its planning-time
  observable bearing and range, recorded per row and per branch, **assigned at
  snapshot time**. Inferring the goal after collection from whichever landmark
  yields the most favourable progress is prohibited. BFS distance is a **label**
  only, never an input.
- **Safety preserves path events**: max over ticks of a per-tick hazard indicator
  (contact, fall, stuck, clearance < 0.15 m) across H=1–4 — explicitly not terminal
  state. **Completion** indicates reaching the bound landmark *at or before* the
  horizon.
- Fixed epoch budget, **final-epoch weights, no best-epoch selection**; fit and
  calibration split **by scene**.
- **Qualification on true latent trajectories, scene-disjoint, before any predicted
  latent is scored**: progress Spearman ≥ 0.50; safety AUC ≥ 0.75 with calibration
  error ≤ 0.10; completion AUC ≥ 0.75, calibration ≤ 0.10; composite pairwise
  ordering accuracy ≥ 0.65; and it must **beat a no-latent baseline** (action
  identity + goal context only) by ≥ 0.05 pairwise accuracy. Thresholds come from
  prior work and the baseline, never from the future 200-state corpus.

# 6. Refined scorer-fit corpus — `a587b1de264dfb54176aa231e5183ae4b7b4229bbf65c02d62438f86af5e7116`

120 states (15/family) × 6 branches = 720, on a third pool disjoint from the
factorial scenes, the pilot states and the evaluation states — **at scene, episode
cluster, state and branch level**, enforced by a hashed per-pool allow-list.

Six branches cannot cover 12 candidates within one state, so allocation is a
**frozen rotation over the 12 candidates indexed by state number**: each candidate
appears exactly **60 times** across 120 states, 7–8 times per family
(integer-balanced). Candidate identity is not confounded with family, with the
fit/calibration split (by scene, rotation by state index), or with goal type
(separate frozen rotation). Each 6-subset includes at least one forward, one
turning and one reversing candidate so all three utility components vary within
every state. Split 96 fit / 24 calibration, **by scene**.

# 7. Compute and storage

**Unchanged from the previous estimate — ≈ 40 h, ~22 GB — because the pilot did not
run and produced no measured throughput to refine it with.**

# 8. Blockers

1. **Four unresolved controller/harness state fields** (previous policy action,
   latency buffers, policy-side RNG, gait phase) — gating the snapshot, the
   qualification and the pilot.
2. **Runtime inference wrapper not located** in this repository, which is what
   would resolve them.
3. Downstream and unchanged: no leakage-free predictor score exists yet; H=2–4
   spatial labels are absent.

## Stopping condition

Nothing is running. No 120-state scorer-fit corpus, no 200-state evaluation corpus,
no scorer trained, no predictor checkpoint scored, no predictor retrained, no frozen
result altered.
