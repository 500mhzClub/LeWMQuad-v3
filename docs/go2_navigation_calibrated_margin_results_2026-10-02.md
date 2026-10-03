# Calibrated margins and pessimistic unknown cells: results, 2–3 October 2026

> **Closed (Andrew, 3 October).** Old-harness findings, summarised with the trap caveat in [go2_navigation_old_harness_safety_findings_closed_2026-10-03.md](go2_navigation_old_harness_safety_findings_closed_2026-10-03.md). Not re-run or extended.

**PRELIMINARY.** Development mode, prelim_test_v1 mazes 30–49, recovery off, coverage-rule fix. The plan is in [go2_navigation_calibrated_margin_experiment_plan_2026-10-02.md](go2_navigation_calibrated_margin_experiment_plan_2026-10-02.md).
- Stage 1 (C1 and C4 margins, old harness) finished overnight; results below.
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

## Stage 1: calibrated margins on remembered walls (C1 and C4, old harness; added 3 October)

**Runs.**
- Cohorts `margin_p95`, `margin_p99` and `margin_c4_nominal_b` (C4's no-margin reference). C1's reference is the clean sensitivity baseline `sens_base`, same mazes.
- The check subtracts the controller's calibrated bound from every remembered-wall distance. Unknown cells count as free. Routing is not inflated.
- Margins: C1 2.10 cm (p95) and 2.89 cm (p99); C4 2.64 and 3.44 cm.

| Ctrl | Margin | Round trips (Wilson 95%) | SPL | Outbound hold rate | Contacts · hard · operating | Min clearance (cm) | vs own no-margin run: success diff (95% CI) · only no-margin / only margin | Time to beacon, non-arrival = 480 s (95% CI) |
|---|---|---|---:|---:|---|---:|---|---|
| C1 | none | 20/20 (0.84–1.00) | 0.86 | 0.015 | 0 · 0 · 0 | 7.5 | – | – |
| C1 | p95 | 10/20 (0.30–0.70) | 0.43 | 0.276 | 0 · 0 · 0 | 6.5 | −0.50 (−0.70 to −0.30) · 10 / 0 | +204 s (+125 to +283) |
| C1 | p99 | 11/20 (0.34–0.74) | 0.46 | 0.282 | 0 · 0 · 0 | 8.2 | −0.45 (−0.65 to −0.25) · 9 / 0 | +189 s (+115 to +262) |
| C4 | none | 20/20 (0.84–1.00) | 0.84 | 0.028 | 0 · 0 · 0 | 5.8 | – | – |
| C4 | p95 | 15/20 (0.53–0.89) | 0.60 | 0.151 | 0 · 0 · 0 | 7.1 | −0.25 (−0.45 to −0.10) · 5 / 0 | +119 s (+53 to +190) |
| C4 | p99 | 12/20 (0.39–0.78) | 0.48 | 0.137 | 0 · 0 · 0 | 7.6 | −0.40 (−0.60 to −0.20) · 8 / 0 | +170 s (+101 to +243) |

**Realised exceedance** (near-wall decisions, true path clearance < 0.60 m, interventions excluded). Calibration on mazes 50–89 with recovery on targets 5% at p95 and 1% at p99.

| Run | Ctrl | Near-wall decisions | e_f above the p95 bound | e_f above the p99 bound |
|---|---|---:|---:|---:|
| no margin (`sens_base`) | C1 | 5,155 | 4.8% | 0.72% |
| no margin | C4 | 5,711 | 4.4% | 0.79% |
| p95 margin | C1 | 7,879 | 2.3% | 0.06% |
| p95 margin | C4 | 8,705 | 2.7% | 0.68% |
| p99 margin | C1 | 8,351 | 2.5% | 0.72% |
| p99 margin | C4 | 11,286 | 2.4% | 0.74% |

- **The calibration transferred.** Without a margin, exceedance on the evaluation mazes is at the target despite the shift from recovery on to recovery off.
- **With a margin it is below target.** The margin changes behaviour (more holding, more turning), and those decisions carry smaller errors.

**Every failure** (32; `scripts/analyse_go2_reserve_trap_development.py`, corrected on 3 October so that a margin shifts the disc and requirement out by the margin):

| Mechanism | C1 p95 | C1 p99 | C4 p95 | C4 p99 |
|---|---|---|---|---|
| **Reserve trap, widened by the margin** | 32, 38, 41, 42, 44 | 35, 40, 42, 44, 48 | 43, 46 | 39 |
| **A translation passes the check but is not selected;** the robot alternates left and right turns for minutes | 30, 31, 35 | 30, 31, 32, 43 | 30, 31, 37 | 30, 31, 35, 37, 40, 48 |
| Other | 33 (stall unclassified), 48 (pose loss after turning, 130 s) | – | – | 45 (no stall; slow, turning-dominated) |

**Reading.**
- **The margins bought almost no clearance and cost a lot of liveness.** Minimum clearance moved by about 1 cm, and 0 contacts was already the case without them. Success fell by 45–50 points for C1 and 25–40 for C4.
- **About half the cost is the reserve trap, widened by the margin** (13 of 30 stalls). Holding inside the widened reserve, no translation passes.
- **The other half is a different, new mechanism**: alternating turns while some translation is clear. **Candidate, not yet verified:** a route-versus-check clash. Routing uses nominal clearance, so it leads into gaps that the margin-inflated check then refuses, and the planner turns back and forth toward a route it cannot follow. The next harness version (reserve exit) does not address it.
- **Per the GPU-order decision, C1/C4 stage 1 is repeated on the new harness**, because their stalls show the trap. Before that, the alternating-turn stalls are diagnosed offline (route direction against the clear translations at each stalled decision), so the repeat is interpretable.

### Diagnosis: the alternating-turn stalls are a route-versus-check clash (3 October)

**Script.** `scripts/diagnose_go2_margin_route_check_clash_development.py`.

**Mechanism.**
- The route is planned at nominal clearance.
- The route-target lookahead and the action check both see wall distances reduced by the margin.
- The lookahead walks along the route and stops at the first point whose straight segment from the robot falls below min(0.48 m, current clearance).
- Where the route runs with nominal clearance between 0.48 m and 0.48 m plus the margin, the lookahead stops short, often at the robot's own cell. The planner then scores every translation as overshooting a target at or behind the robot, so turns win.

| Run | Route-following decisions | Target shortened by the lookahead | Target collapsed (within 10 cm of the robot) | Where determinable, original target blocked only by the margin |
|---|---:|---:|---:|---:|
| C1 no margin | 6,161 | 1% | 0% | – |
| C4 no margin | 6,941 | 2% | 0% | – |
| C1 p95 | 13,149 | 44% | 22% | 927 of 1,048 |
| C4 p95 | 12,396 | 33% | 22% | 1,262 of 1,265 |
| C1 p99 | 13,087 | 37% | 15% | 1,776 of 1,822 |
| C4 p99 | 13,662 | 41% | 24% | 2,194 of 2,310 |

**The 16 alternating-turn stalls:**
- **12: the lookahead shortened or collapsed the target** in at least 63% of stalled decisions, almost always because of the margin alone: C1 p95 30, 31, 35; C1 p99 30, 32; C4 p95 30, 31, 37; C4 p99 30, 31, 35, 37.
- **2: the same clash at the action check** (C4 p99 40, 48). When the robot faced its target, forward was blocked, and in 21 of 24 such decisions it would have passed without the margin.
- **2: the margin-widened reserve trap, turning variant** (C1 p99 31, 43). The robot hovers at 0.48 m plus the margin with its target behind it. The turn toward the target fails the check by under a millimetre in about half the decisions, so the filter picks the opposite turn; translations are mostly blocked.

**Conclusion: confirmed.** Per Andrew's decision, margin-aware routing is added to the next harness version: when a margin is active, routing clearance grows by the same bound. With no margin, behaviour is unchanged. Stage 1 (C1 and C4 at p95 and p99, C3 at p95) is then repeated on that version.
