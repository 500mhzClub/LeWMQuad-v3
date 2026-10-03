# Plan: next harness version — reserve exit and aligned C2 clearance, 2 October 2026

**Status: approved by Andrew (2 October, evening); not yet built.** Build starts after tonight's queues finish, before the dynamics-perturbation experiment. Development mode; all results PRELIMINARY.

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
