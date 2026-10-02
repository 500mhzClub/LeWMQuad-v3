# Calibrated margins and pessimistic unknown cells: results (in progress), 2 October 2026

**PRELIMINARY.** Development mode, prelim_test_v1 mazes 30–49, recovery off, coverage-rule fix. The plan is in [go2_navigation_calibrated_margin_experiment_plan_2026-10-02.md](go2_navigation_calibrated_margin_experiment_plan_2026-10-02.md).
- Stage 1 (C1 and C4 margins, old harness) is still running and will be added here.
- C3 now runs on the next harness version ([plan](go2_navigation_harness_reserve_exit_plan_2026-10-02.md)).

## Stage 2, C1: never-observed cells block moves (look-around exemption, p95 bound)

**Run.**
- Cohort `stage2_c1_lookaround_p95`, pinned to `0cb7cddb`.
- Never-observed cells block a move within 0.425 m + C1's p95 bound (2.10 cm) of the forecast centre path.
- Seeded free: the 0.5 m start disc (the operating precondition) and the traversed track.
- The scripted look-around is exempt.
- Remembered walls are at nominal, with no margin.

**Result.**
- **Round trips: 9 of 20,** against 20 of 20 for the clean baseline (`sens_base`, same mazes). 0 contacts.
- **Stalls: 13 of 20** (a span of at least 120 s without translation): 10 at the start, 3 later.
  - Maze 48 recovered and completed.
  - Maze 37 resumed at 232 s but failed.
  - Maze 36's stall (64–302 s) ended and it completed.
- **The ≤ 10% condition for reinstating C3 stage 2 is not met.** C3 stage 2 stays cancelled.

**What blocked translation.** Two scripts, sampled every 5 s inside each stall:
- `scripts/analyse_go2_stage2_stalls_development.py` matches each stalled decision's logged clearance against the true-wall clearance and the reconstructed unseen-cell clearance.
- `scripts/analyse_go2_reserve_trap_development.py` reports clearance at onset and whether any translation increased it.

| Mechanism | Mazes | Notes |
|---|---|---|
| **Reserve trap, remembered wall** | 31, 32, 35, 37, 39, 44, 46, 48 | Logged clearance matched the true wall less the map's 1.5 cm bias. The robot sat inside the 0.48 m reserve or the 0.45 m disc, and no translation increased clearance or met the recovery conditions. Maze 39 is the exception: a translation would have increased clearance in 81 of 93 samples, but failed recovery. |
| **Unseen cells (live map)** | 33, 34, 41 | Logged clearance was 10–12 cm below the true-wall clearance, so live-map unseen cells bound. The reconstruction marks those areas as observed, so whether they were unobservable from the stall position is **not established**. |
| **Hold outscored every move** | 45 | Clearance 0.479 m on every candidate, at the reserve edge. Translations were clear but never chosen (1139 "movement outscored" holds). |
| **Recovered** | 36 | Stall 64–302 s, then completed. Translations were clear throughout and a translation would have increased clearance at every sample. |

**Reading.**
- **Most of the rule's liveness cost runs through the shared reserve trap.** It accounts for 8 of 13 stalls.
- The rule delays translation at the start: blocked moves mean longer turning. Turning drifts the body 5–7 cm, which carries it into the reserve, and then it cannot translate out.
- Only 3 stalls were bound by unseen cells, and for those the unobservability of the blocking cells could not be checked offline.
- **The sensor-coverage framing stays on hold,** as Andrew asked. Establishing it needs the live map's unseen cells logged at stalled decisions; the reconstruction does not reproduce them.
- **Suggested next step (needs Andrew's decision).** Re-run this C1 stage 2 on the next harness version, where translations that increase clearance are allowed. That separates the rule's own cost from the trap. Cost: about 1.5 h of CPU.

**Reserve-trap stalls are marked** in every stage write-up (Andrew, 2 October): a larger margin, or a stricter unseen-cell rule, means more time inside the reserve.
