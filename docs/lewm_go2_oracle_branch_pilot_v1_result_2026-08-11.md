# 20-state oracle branch pilot — executed; identifiability gate **FAILS**

Date: 2026-08-11
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.**

No world-model checkpoint was loaded. No scorer was trained, no scorer-fit or
evaluation corpus was generated, no frozen confirmatory result was altered.
H=1–4 spatial labels remain deferred and are not part of this gate.

Artefacts: `.generated/go2_oracle_branch_pilot_v1/{identity_manifest.json,
replay_qualification.json, pilot_branches.jsonl, gate_report.json}`.

---

# 1. Digests

| item | digest |
|---|---|
| candidate bank | `85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9` (**unchanged**) |
| oracle implementation (final) | `03dbe01100870cb4cc082f936bc3d0b62aa1e5d23e8eedb44188e88992acbb53` |
| oracle implementation (pre-run) | `4849e6dd…` — **SUPERSEDED_PRE_RUN** |
| qualification contract | `bb2100e785ffd190be23271a13ec9742b04dbea22d309056796cca3d243ad289` |
| state-selection contract | `722fba73fdc3b9d6fce474997e2723b4dd82e5ad8fc8480e91348ca7290ae845` |
| pilot identity manifest | `015eb0bb4ccb9da28ce4b055771975fc68ac0c986e462d9c3af0a61ef45a9ea2` |

The oracle digest **did** supersede the fall-only pre-run version. The design-v1.1
weights (1.0 / −2.0 / +0.5), the tie tolerance (0.02), the horizon, the candidate
bank and all five gate thresholds are unchanged; only the safety *implementation*
was corrected, and it was corrected before any qualification or pilot branch ran.

## 1.1 Path-level safety, from production definitions

`safety = max over every command tick of the four-block branch` of a per-tick
hazard indicator — never terminal-only:

| evidence | production source |
|---|---|
| clearance | `SceneGraph.clearance_to_walls` < 0.15 m |
| contact | `analyze_go2_closed_loop_quality._body_probe_configuration_clearance_m` over `InflatedOccupancyGrid`, body 0.35 / 0.18 / margin 0.03, hazard at ≤ 1e-4 |
| stuck | `lewm_worlds.labels.derived.DerivedLabelComputer.stuck_label` |
| termination | `RolloutRunner._check_and_reset_fallen_envs` predicates (fall-z, out-of-bounds, tip), evaluated per tick |

`PoseStep.last_command` is aligned to the executed command of the tick that
produced the displacement being measured. Production's own message-timestamp join
pairs a pose with the *following* tick's command; that off-by-one would have
measured the stuck label against the wrong command, so the computer is reused with
a corrected alignment. This is recorded in the oracle digest.

# 2. Canonical boundary — real checks replaced `_assert_boundary`

Asserted, with **refusal** (never normalisation) outside the boundary: single-env
build; command-block tick phase 0; low-level decimation phase 0; observation-emission
phase 0 ns; the exact clock relation `_sim_time_ns == policy_steps × policy_dt_ns`;
no reset in the preceding block and `reset_count` unmoved; all production
termination predicates false; the tipped accumulator zero; `(s−1) mod 5 == 0` on the
per-episode source step with `episode_step` agreeing with the driver's counter; and
`_last_actions` / `_last_executed` initialised, finite, inside the safety envelope
and consistent with the final tick of the last executed block.

The boundary check earned its place: it **refused** a state produced by a driver
bug (a re-drive path that rewound the per-episode counters while the global sim
clock kept running) rather than capturing a misaligned state. The driver was fixed
by capturing inline during the selection scan; the global sim clock is never
rewound.

# 3. Replay qualification — **PASS**, after one decisive correction

## 3.1 First attempt failed; the cause was a genuine snapshot omission

All three repetitions of every case diverged. First divergent field, exactly:

```
case sign_reversal, block 0, tick 0, policy step 0, field root_pos
  a z = 0.35557469725608826
  b z = 0.35557463765144350
```

x and y identical; z differs by ~1 ULP of float32 at the **first** physics step.
Isolation showed the policy is exactly repeatable for a fixed observation, and the
solver is exactly repeatable for fixed joint targets — so the fault was in what
`restore` failed to restore.

**Genesis's `Scene.save_checkpoint` walks only `solver.__dict__` and the solver's
`data_manager`.** A field-level audit found 586 reachable gstaichi fields, of which
the checkpoint misses the collider state (`contact_cache`, `contact_data`,
`broad_collision_pairs`, `n_contacts`, `first_time`), the whole constraint solver
(including `qacc_ws`, `jac`, `Jaref`, `quad`), the contact islands, and the GJK
state. Those are carried between steps, so two branches from the "same" snapshot
started from different solver internals.

**Correction:** the snapshot now captures every gstaichi field reachable from the
scene and its solvers, minus the static SDF and support-field lookup tables. Of the
1,899 MB reachable, only **0.9 MB actually mutates**; the excluded tables are
1,889 MB of immutable geometry. Restoration is validated by the qualification
itself, not by the exclusion argument.

A second finding fell out: on this scene the vulkan backend was non-deterministic
at 1 ULP even with full restoration, while the CPU backend is exactly deterministic
and ~13× faster for a single environment. **The pilot runs on the CPU backend.**
Nothing is rendered by the pilot, so no render path is affected.

## 3.2 Qualified result

Four cases × 3 repetitions, all quantities compared exactly (policy observation,
raw actor output, `_last_actions` before/after, action applied under latency,
requested and post-slew commands, root pose and velocity, joint positions and
velocities, contact/safety events, and the per-tick oracle evidence):

| case | family | source step | 3/3 identical | slew clipped |
|---|---|---|---|---|
| sign reversal | large_enclosed_maze | 201 | **yes** | yes |
| sustained turn | loop_alias_stress | 201 | **yes** | no |
| obstacle-adjacent | medium_enclosed_maze | 101 | **yes** | no (body clearance −0.030 m) |
| slew limited | visual_sensor_stress | 201 | **yes** | yes |

Sensitivity controls: perturbing `_last_actions` changes **both** the policy
observation and the action applied under latency; perturbing `_last_executed`
changes the first post-slew command (`[0.30, 0.00, 0.00]` → `[0.30, 0.20, 0.00]`,
the lateral channel because its per-tick delta limit is 0). Both required
divergences observed.

Branch order: 12 candidates in declared order and in reverse, restoring before
each — every candidate's complete branch result identical between orders, 0
mismatches.

Total qualification wall time 35.7 s.

# 4. The pilot — 240 branches, all executed

20 frozen states, one per scene, `val` split, disjoint from the 80-scene factorial
panel and from the four development replay states; all 8 families, ≥ 2 states each
(3/3/3/3/2/2/2/2). One mid-episode snapshot per state, 12 independent restores,
four blocks each: **240 attempted, 203 valid, 37 invalid**. 170 s wall, 2.2 MB
stored (snapshots are in-memory, ~11 MB live, one at a time).

## Gate — **FAIL (2 of 5 components pass)**

| component | threshold | measured | verdict |
|---|---|---|---|
| uniquely separated best candidate | ≥ 70 % | **47.1 %** | fail |
| median distinct utility levels | ≥ 5 | **2.0** | fail |
| median best-to-worst spread | ≥ 0.10 | **0.55** | pass |
| invalid-branch rate | ≤ 20 % | **15.4 %** | pass |
| all 8 families with ≥ 2 valid states | 8 | **7** | fail |

Overall identifiability: **FAIL**. Nothing was retuned after observing outcomes.

# 5. Why it failed — three separable causes

**(a) The horizon does not move the robot a BFS cell.** Over the frozen four-block
(2.0 s) branch, ΔBFS was 0 in 162/203 valid branches, ±1 in the remaining 41.
Divided by the frozen `PROGRESS_SCALE = 20`, progress takes only the three values
{−0.05, 0, +0.05}. The progress term therefore cannot separate candidates.

**(b) Safety is saturated, and stuck/contact are the reason.** 96.1 % of valid
branches carry safety = 1: `stuck` fires in 186/203 and `contact` in 116/203, while
`low_clearance` fires only once. The per-tick production stuck label (commanded
magnitude > 1e-3 and speed < 0.05 m/s) is almost always true somewhere in 20 ticks
of a trotting gait, and the 0.35 × 0.18 m body probe exceeds maze corridor
half-widths, so the geometric contact proxy is nearly always true indoors. With
safety binary and near-constant and progress ≈ 0, the composite utility collapses
onto six values overall — `[−2.05, −2.00, −1.95, −1.45, 0.00, 0.05]` — hence a
median of 2 distinct levels and only 47 % of states with a separated best.

**(c) Three states are geometrically unavailable, not defective.** All 36
`no_eligible_canonical_boundary` invalids come from three states
(`large_enclosed_maze-0`, `medium_enclosed_maze-1`, `small_enclosed_maze-1`) whose
episodes spend the entire 40–120 block window in low-index cells (0–5) from which
**no landmark is BFS-reachable** under `transit_blocked=nav_blocked_cells`. This is
the spawn-caged / disconnected-pocket geometry already documented in the Go2 eval
geometry audit. It costs `small_enclosed_maze` its second valid state, which is the
sole reason the family-coverage component fails. The single remaining invalid is one
branch whose final pose was unlocatable.

# 6. What this does and does not license

The **mechanism is qualified**: mid-episode capture at a real canonical boundary,
bit-identical restore across repetitions and across branch orders, demonstrated
sensitivity to both controller fields, and a path-level oracle built from
production definitions. That was the blocker in every previous pass and it is now
resolved and measured.

The **oracle as frozen does not identify a best action** at this horizon in these
scenes. The failure is not the snapshot, the boundary, or the branch runner — it is
that a 2 s horizon produces almost no BFS movement while a binary path-level safety
term dominates. Any redesign (longer horizon, graded safety, finer progress metric,
reachability-filtered state pool) is a **new frozen design**, and must be registered
before it is run. It must not be derived by tuning against this corpus.

Measured cost, for the first time: **~3.5 minutes and ~2.2 MB for 240 branches**
(qualification 35.7 s; pilot 170 s; branch execution 49.4 s of that). The standing
≈ 40 h / ~22 GB estimate is now known to be very conservative for branch generation
itself.

# 7. Blockers

1. **The frozen oracle is not identifiable at H=4 in these scenes** — cause
   located (§5a, §5b), remedy requires a new registered design, not a retune.
2. **Scene-pocket unreachability** removes states from the pool; the selection rule
   needs a landmark-reachability precondition applied at scene level, not only at
   the capture boundary.
3. Downstream and unchanged: no leakage-free predictor-side candidate score;
   H=2–4 spatial labels absent.

## Stopping condition

Nothing is running. No scorer-fit or evaluation corpus generated, no scorer
trained, no predictor scored or retrained, no frozen result altered. The 20 pilot
state identities are permanently marked unavailable for scorer-fit and final
evaluation in `identity_manifest.json`.
