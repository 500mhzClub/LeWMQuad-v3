# Clean baseline on the reserve-exit harness: recovery-off C1–C4, mazes 30–49, 3 October 2026

**PRELIMINARY.** Development mode, prelim_test_v1 mazes 30–49, episode 0, recovery off, coverage-rule fix. Harness `reserve_exit_v1` ([plan](go2_navigation_harness_reserve_exit_plan_2026-10-02.md)).

**The harness version:**
- (a) the reserve exit for every forecast-based clearance check;
- (b) C2's clearance check on its dispatched command path in place of its all-actions block;
- nothing else changed.

**Gates passed:** C1 20/20 and C0 20/20 on the development mazes.

**Models,** as in the preliminary run: C3 uses the large past-frames decoder, and C4 its matched refit (one checkpoint file).

**Cohorts:**
- C1, C3 and C4: `rexit_rerun` (pin `3b173833`).
- C2: `rexit_rerun_c2` (mazes 30–44) and `rexit_rerun_c2b` (45–49), on `reserve_exit_v1_1`, the version with C2's path as dispatched.
- The `rexit_rerun` C2 missions used v1's 40-entry path, are superseded and are not in the tables (10/20).

**Scripts:**
- `scripts/report_go2_prelim_results_development.py` (per-mission rows);
- `scripts/report_go2_reserve_exit_rerun_development.py` (exits and stalls).

## Results

| Ctrl | Missions | Round trips (Wilson 95%) | SPL | Median round trip (s, successes) | Outbound hold rate | Contacts · hard · operating | Min clearance (cm) | Preliminary run (old harness, recovery off) |
|---|---:|---|---:|---:|---:|---|---:|---|
| C1 | 20 | 20/20 · 1.00 (0.84–1.00) | 0.86 | 155 | 0.015 | 0 · 0 · 0 | 7.5 | 20/20 |
| C2 | 20 | 9/20 · 0.45 (0.26–0.66) | 0.37 | 172 | 0.392 | 0 · 0 · 0 | 5.9 | 11/20 |
| C3 | 20 | 19/20 · 0.95 (0.76–0.99) | 0.82 | 160 | 0.095 | 0 · 0 · 0 | 9.1 | 19/20 |
| C4 | 20 | 20/20 · 1.00 (0.84–1.00) | 0.84 | 164 | 0.028 | 0 · 0 · 0 | 5.8 | 20/20 |

**Paired comparisons.** Maze-level paired bootstrap 95% intervals. SPL is compared on mazes both succeed; time to beacon on all mazes, with non-arrival set to 480 s.

| Pair | Mazes | Success | Difference (95% CI) | Only first / only second | McNemar exact p | SPL difference, both succeed (95% CI) | Time to beacon, all mazes, non-arrival 480 s (95% CI) |
|---|---:|---|---|---|---:|---|---|
| C2 vs C1 | 20 | 9 / 20 | -0.55 (-0.75 to -0.35) | 0  / 11 [33, 34, 36, 38, 39, 42, 43, 44, 45, 47, 48] | 0.001 | -0.02 (-0.05 to +0.01) | +187.8 s (+108.9 to +268.4) |
| C3 vs C1 | 20 | 19 / 20 | -0.05 (-0.15 to +0.00) | 0  / 1 [31] | 1.000 | -0.01 (-0.02 to +0.01) | +22.7 s (-0.8 to +62.3) |
| C4 vs C1 | 20 | 20 / 20 | +0.00 (+0.00 to +0.00) | 0  / 0  | 1.000 | -0.01 (-0.04 to +0.00) | +14.3 s (+1.1 to +31.0) |
| C3 vs C4 | 20 | 19 / 20 | -0.05 (-0.15 to +0.00) | 0  / 1 [31] | 1.000 | +0.01 (-0.01 to +0.03) | +8.3 s (-22.0 to +51.5) |

## Reading

- **Safety:** 0 contacts and 0 hard or operating violations, in all 80 missions.
- **C1, C3 and C4 still do not separate.**
  - C4 equals C1 (20/20).
  - C3 is 19/20, failing only maze 31, where it also failed in the preliminary run.
  - SPL differences are 0.01–0.02.
- **C2's separation remains: 9/20 against C1's 20/20.** The difference is −0.55 (−0.75 to −0.35), 11 discordant mazes to 0, McNemar p = 0.001.
- **The harness change did not close C2's gap,** and the mechanism has changed.
  - In the preliminary run, C2's all-actions block was the cause.
  - Now C2 is checked like the others, and the remaining causes are the two below.

## Every remaining failure

| Ctrl | Maze | Mechanism |
|---|---:|---|
| C2 | 33, 34, 36, 38, 39, 43, 44, 48 | **Reserve trap, turning variant.** Onset clearance 0.48–0.51 m true, inside the reserve or the disc. C2's dispatched path for an in-place turn never moves the centre, so it can never show the clearance gain the turn reserve requires inside the reserve: both turns were blocked at every decision sampled. Every translation loses clearance, so none qualifies for the exit, and C2 holds. |
| C2 | 42, 45, 47 | **Terminal spin.** On the route to the goal cell, the waypoint is 2–4 cm from the robot at a bearing of −28° to −47°, just outside the 2-cm arrival radius. C2's heading-first terminal rule turns right (about 700 consecutive right turns), and the bearing rotates with the body, so the turn never completes. This is the known terminal heading limit cycle ("trap 2"), which recovery's spin break handled in the preliminary run. |
| C3 | 31 | **Reserve trap, turning variant**, as in the preliminary run. The stall starts at 47.5 s at 0.49 m true clearance. Inside the reserve, an exit the check selected was withheld by the coverage rule: unknown cells in the footprint require a view, and the view needs a turn. Inside the disc, from 70 s, the unchanged stopping projection vetoed the exit, because its path starts below 0.45 m. |

**Exits taken** (final command a translation that passed only by the exit):
- C2: 11 in 7 missions; C3: 1; C4: 2; C1: 0.
- **None was stopped by the depth stop.** At the median, true centre clearance rose from 48 to 49–50 cm one second after an exit.

**One root.** Every remaining reserve-trap failure needs a turn inside the reserve, and the turn reserve allows that only when the forecast centre path shows a clearance gain. The same rule froze every stage-1 dynamics mission at μ = 0.2 ([dynamics plan](go2_navigation_dynamics_perturbation_plan_2026-10-02.md)). The candidate change is recorded there for Andrew's decision; it is not applied in this version.
