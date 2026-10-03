# Plan: next harness version — reserve exit and aligned C2 clearance, 2 October 2026

**Status: approved by Andrew (2 October, evening); built 3 October (`lewm/dev_harness_reserve_exit_development.py`, harness `reserve_exit_v1`).** Build starts after tonight's queues finish, before the dynamics-perturbation experiment. Development mode; all results PRELIMINARY.

## Why

The preliminary run's one clear separation was C2 without recovery: 11/20 against C1's 20/20. That result is mostly a shared harness trap, so the report now carries a correction. The trap is recorded in the [known limitations](go2_navigation_harness_v4_known_limitations_2026-09-29.md), and the analysis tool is `scripts/analyse_go2_reserve_trap_development.py`.

**The trap.** Once the base centre is within 0.45 m of a remembered wall, no translation can pass the clearance check, even one moving straight away from the wall.
- **C2** makes every action ineligible there, turns included. This explains 8 of its 9 recovery-off stalls.
- **The forecast controllers** still turn there. But the gait drifts the body 5–7 cm while turning, and that carries them inside. This explains C3's maze-31 deadlock and the C1 stage-2 stalls seen so far.

## Changes (new files only; nothing else changes)

**(a) Reserve exit, for every forecast-based clearance check.** A translation is also allowed if its forecast centre-path clearance to remembered walls:
- never decreases along the path, with a 1 mm tolerance; and
- ends higher than it starts;

even when it starts inside the reserve (below 0.48 m) or inside the disc (at or below 0.45 m). Everything else about the check is unchanged.

**(b) C2 uses the same clearance semantics as C1, C3 and C4.** C2's all-actions block at 0.45 m goes.
- C2 has no forecast. Each candidate is checked on its nominal command path: the requested command integrated over the same horizon, as the dispatch check already projects it.
- The check is the same one the others use, including the reserve-recovery modes and the new exit.
- C2's own selection rule is unchanged: the nearest primitive to the waypoint feedback, among eligible actions.
- **This gives C2 a crude kinematic forecast for its safety check.** The nominal command path has no learned motion model and no command history. C2's selection still uses no forecast. Approved by Andrew.

**(c) Nothing else changes.** Unchanged:
- the dispatch-time depth stop (0.45 m disk on the current depth images);
- the stopping projection, the coverage rule and routing;
- the margins and the pessimistic-unknown rule, which stay off.

**Residual to watch.** If the wall the robot is leaving is visible ahead within 0.45 m, the depth stop can still veto an exit. The re-run reports any such vetoes.

**Protocol note.** The V4 protocol's one-change-per-version rule does not apply in development mode (Andrew). Changes (a) and (b) are recorded together as one version.

## Gate

The V4 protocol's harness-iteration gate, on the dev_tune mazes 0–9, episodes 0 and 1 (20 episodes), with recovery off and the coverage fix:
1. C1 passes 9 of 10 on the first episodes and 9 of 10 on the second, with zero contacts and zero hard violations.
2. Then C0 (oracle forecasts) passes at least 19 of 20.

The version is used only if it passes. The gate runs are labelled, and their failures kept.

## Re-run

Recovery off, on prelim_test_v1 mazes 30–49 (the preliminary run's recovery-off set):
- C1, C2, C3 and C4, with the same models as the preliminary run (C3 large past-frames decoder, C4 refit, one checkpoint file).
- Pinned launches.

**Reported:**
- Each controller against C1 on the same mazes: success difference with a paired bootstrap, discordant counts and McNemar; SPL; time.
- Every remaining stall, with its mechanism: the reserve-trap analysis plus the binding attribution (remembered wall, scan mode, other).
- Contacts and clearance for every run.
- Exits taken: how often the new exit allowed a translation, and the clearance before and after.

## Cost and order

**Build:** about half a day, in new files. It covers the exit rule as a mixin over the memory-forecast check, C2's nominal-path check, a synthetic test of each, and a v3 pinned launcher option.

**Compute:**

| Run | Estimate |
|---|---|
| C1 gate (20) | about 1 h on the CPU |
| C0 gate (20) | about 2 h on the CPU |
| C1, C2, C4 re-run | about 4 h on the CPU |
| C3 re-run | about 9–10 h on 2 GPU lanes, first on the GPU once the gates pass. It also serves as C3's margin reference; C3 at the p95 margin on this harness follows. |

## Interaction with the calibrated-margin experiment

**The trap may confound it.** A larger margin means more time inside the reserve, where the trap applies. So the stage 1 and stage 2 write-ups mark every reserve-trap stall, using `scripts/analyse_go2_reserve_trap_development.py`. If the trap explains a margin's liveness cost, the margin comparison is re-run on this version.

## Decisions (Andrew, 3 October, morning; superseded by the re-scope below)

1. **Margin repeats wait for today's diagnosis** of the alternating-turn stalls in margin stage 1 ([results](go2_navigation_calibrated_margin_results_2026-10-02.md)).
   - **If the route-versus-check clash is confirmed,** margin-aware routing is added to this version: when a margin is active, routing clearance grows by the same bound.
   - Nominal behaviour (no margin) is unchanged, so the gates still apply as written.
   - Then stage 1 is repeated on this version: C1 and C4 at p95 and p99, and C3 at p95.
2. **C1 stage 2 is re-run on this version** at the lowest priority, when CPU is free.
3. **The forecast-sensitivity curve is re-measured on this version, trimmed:** clean, noise 20, 40 and 80 mm, and forward/arcs × 0.25, 20 mazes each. The write-up states that the old curve's cliff was mostly reserve-trap stalls.
4. **The shared results page is updated once,** after Sunday's re-runs.
5. **The dynamics experiment stays next** after these.

## Re-scope (Andrew, 3 October, midday)

**The goal, that the JEPA controller can navigate, is established.** Safety analysis is a nice-to-have, and that thread is closed ([old-harness findings](go2_navigation_old_harness_safety_findings_closed_2026-10-03.md)).

**Kept:**
- this version's two changes: (a) the reserve exit and (b) C2's aligned clearance;
- the C1 gate, then the C0 gate;
- one recovery-off re-run of C1–C4 on mazes 30–49, the clean baseline.

**Dropped:**
- margin-aware routing (not built);
- the margin repeats (C1, C4 and C3);
- the C1 never-seen-cells re-run;
- the sensitivity re-measure.

**Next, straight after the re-run:** the [dynamics experiment](go2_navigation_dynamics_perturbation_plan_2026-10-02.md). Friction first, then low-friction patches with a visual marker. C1, C3 and C4, recovery off.

## Built (3 October)

**Files:**
- `lewm/dev_harness_reserve_exit_development.py` (harness `reserve_exit_v1`);
- `scripts/run_go2_dev_mission_pinned_v3_development.py` and `scripts/launch_go2_dev_cohort_pinned_v3_development.py` (`--harness reserve_exit_v1`);
- tests in `scripts/test_go2_dev_reserve_exit_development.py`: 13 synthetic tests, all passing.

**(a) The exit sits at the innermost check**, `ReserveRecoveryLookaheadRuntime`'s call to `select_clear_prediction`. Every outer layer (turn reserve, latch, stopping projection, terminal rules) sees an exit as an ordinary clear candidate.
- A translation passes if its eight segment clearances never fall more than 1 mm below the previous one or below the start, and the last ends above the first.
- With the exit off, the result is bit-identical to the frozen check. With it on and no exit applying, the action is identical.

**(b) C2's mixin replaces only the all-actions block.**
- Eligibility = C2's own rule AND the same check (translation reserve, reserve recovery, exit, stepwise turn reserve) on the nominal command path: the committed prefix, the 400-ms commit and one stopped tick, unicycle-integrated at 10 ms.
- Selection is still the nearest primitive to the waypoint feedback.
- C2's terminal heading-first rule keeps its own guard: it stays off while the centre is within 0.45 m.

**Residual (unchanged by design).** Inside the 0.45-m disc and facing the wall, nothing passes for any controller:
- turns are blocked inside the disc, as before;
- every translation primitive moves forward, and none reverses.

The re-run reports any such stall.

## Gates and launch (3 October)

**C1 gate: passed.** Cohort `rexit_gate_c1`, dev_tune mazes 0–9, launch pin `963fdb42`.
- 10/10 on episode 0 and 10/10 on episode 1.
- 0 contacts and 0 hard violations; minimum separation 7.0 cm.
- Every decision carries the exit receipt.

**C0 gate:** cohort `rexit_gate_c0`, running (4/4 so far). The re-run counts only if C0 reaches at least 19/20.

**C2 smoke test** (cohort `rexit_smoke_c2`, dev_tune mazes 0 and 1): not a gate, run because the gates do not cover C2.
- **Maze 0:** success, with one exit taken.
- **Maze 1:** failed on the residual.
  - At 41.6 s C2 sat at 0.484 m, where every translation would lose clearance. By 57.6 s turning drift had carried it to 0.4499 m, 0.1 mm inside the disc.
  - It then stayed in scan mode (`ADDITIONAL_VIEW_REQUIRED`), which allows only turns and hold. Inside the disc the turn reserve blocks both turns, for every controller, so it held for 420 s.
  - The old harness's all-actions block left it in the same place, so this is not a regression. It is the inside-the-disc residual, here met while scanning rather than facing a wall.

**Re-run launched** at 12:32 BST: cohort `rexit_rerun`, pin `3b173833` (runtime files as `963fdb42`).
- C1, C2, C3 and C4 × prelim_test_v1 mazes 30–49, recovery off.
- Models as in the preliminary run: one checkpoint holding the large past-frames decoder and its matched C4.
- 7 workers, 2 of them C3 GPU lanes.

## Correction: C2's nominal path, reserve_exit_v1_1 (3 October, 12:55)

**The error.** v1 integrated each primitive's 40-entry list: 300 ms of motion, then a stop. But a 400-ms commit dispatches the primitive's first command, held for four 100-ms ticks (`ScheduledCommand.prepare`). So v1's C2 path under-predicted C2's moves by up to 25%, which is not the plan's "requested command integrated as dispatched".

**The fix,** in a new standalone module so that the pinned v1 file stays untouched while the re-run launches: `lewm/dev_harness_reserve_exit_v1_1_development.py`, harness `reserve_exit_v1_1`.
- C2's path is now exactly the dispatched sequence, `command_sequences(prefix, pulse)`: the committed prefix, the candidate command for four ticks (one tick for a translation in C2's terminal pulse mode), and a stopped tick. These are the sequences the forecast controllers' forecasters receive.
- The exit rule is byte-identical, so C0, C1, C3 and C4 behave exactly as under v1.
- Tests: `scripts/test_go2_dev_reserve_exit_v1_1_development.py`, 13 passing.

**Consequences for the re-run:**
- The baseline's C1, C3 and C4 come from `rexit_rerun` (v1, identical for them).
- C2 comes from a separate cohort, `rexit_rerun_c2`, on v1.1, launched with the v4 pinned launcher.
- The `rexit_rerun` C2 missions (v1 path) still run because the cohort's queue cannot be changed without editing pinned files. They are kept, labelled superseded.
- The C1 and C0 gates are unaffected: neither uses C2's path.
- The dynamics runs use v1.1.
