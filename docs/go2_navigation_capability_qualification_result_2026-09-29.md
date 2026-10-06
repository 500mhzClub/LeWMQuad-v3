# Navigation capability qualification result, 29 September 2026

*Capability qualification, not paper results. The training renders behind C3 and C4 have unverified provenance. The frozen harness is `v4_completed_support` (`82b7b604…`), which passed its C0 gate 20/20.*

## Capability

**Two learned or kinematic controllers qualify: C4 (supervised predictor) at 19/20 and C1 (command history) at 18/20. The JEPA controller C3 (13/20) and the reactive controller C2 (11/20) do not.** Every validation mission had zero disallowed contacts, zero hard-clearance violations and zero operating-margin violations.

| Controller | Round trips | 95% CI (maze bootstrap) | Beacon | Home | Contacts | Capable (≥16/20 and 0 contacts) |
|---|---|---|---|---|---|---|
| C1 command history | 18/20 (90%) | [75, 100] | 19/20 | 18/20 | 0 | **yes** |
| C2 reactive | 11/20 (55%) | [35, 75] | 12/20 | 11/20 | 0 | **no** |
| C3 JEPA (maze-data head) | 13/20 (65%) | [45, 85] | 13/20 | 13/20 | 0 | **no** |
| C4 supervised predictor | 19/20 (95%) | [85, 100] | 19/20 | 19/20 | 0 | **yes** |
| C0 oracle (diagnostic, 10 mazes) | 10/10 (100%) | [100, 100] | 10/10 | 10/10 | 0 | diagnostic |

Paired round-trip differences against C1, over the same 20 mazes, in percentage points with 95% bootstrap intervals:
- C4: **+5 [−10, +20]**
- C3: **−25 [−50, 0]**
- C2: **−35 [−65, −5]**

**Design.** Episodes 10/0–29/0, one per maze, for C1–C4; 10/0–19/0 for C0. The paired maze-cluster bootstrap uses 10,000 replicates with seed 2026092519. Classification is by point estimate, as [pre-registered](go2_navigation_capability_qualification_plan_2026-09-28.md). C3's upper bound (85%) reaches past the criterion, so its non-capability rests on the point estimate. C2's does not.

## Metrics

| Controller | SPL out [CI] | SPL return [CI] | Time to beacon, median (IQR) | Time to home, median (IQR) | Stall out / return (holds / decisions) | Wall per mission (median) |
|---|---|---|---|---|---|---|
| C1 | 0.71 [0.61, 0.79] | 0.83 [0.69, 0.92] | 105 s (86–142) | 58 s (47–67) | 14% (1733/7130) / 3% (82/2842) | 413 s |
| C2 | 0.41 [0.26, 0.57] | 0.50 [0.32, 0.69] | 108 s (78–132) | 53 s (48–59) | 23% (5510/13211) / 8% (880/2434) | 817 s |
| C3 | 0.44 [0.29, 0.58] | 0.56 [0.38, 0.74] | 211 s (150–254) | 81 s (65–103) | 49% (9086/14733) / 22% (960/3203) | 4,409 s |
| C4 | 0.73 [0.63, 0.81] | 0.86 [0.77, 0.92] | 97 s (76–123) | 64 s (51–85) | 6% (427/6474) / 8% (279/3046) | 1,469 s |
| C0 | 0.77 [0.72, 0.83] | 0.93 [0.92, 0.94] | 129 s (100–189) | 58 s (48–71) | 15% (811/3600) / 1% (12/1494) | 1,388 s |

Times cover successful legs only. SPL scores a failed leg as zero. Wall time was measured with up to five concurrent owners on one GPU.

**Decision latency.** Median / p95, physics paused. These isolated figures come from replaying a fixed sample (the first 100 decisions of 10/0, 11/0 and 12/0), one controller at a time with nothing else running, checked against the log. The in-run figures, measured under concurrency, are kept for reference.

| Controller | Isolated median / p95 | In-run (under concurrency) median / p95 |
|---|---|---|
| C1 | 0.20 / 0.22 s | 0.23 / 0.25 s |
| C2 | 0.19 / 0.21 s | 0.22 / 0.24 s |
| C3 | **2.66 / 2.71 s** | 8.13 / 8.81 s |
| C4 | **0.83 / 0.85 s** | 3.03 / 3.48 s |
| C0 | 1.73 / 1.76 s (physics branches) | 2.00 / 2.23 s |

**Safety.** All five controllers had zero hard violations, zero unresolved native samples, zero operating-margin violations and zero FK interval failures. Minimum articulated separation: C1 63 mm, C2 69 mm, C3 28 mm, C4 50 mm, C0 46 mm.

## Per-maze results

RT is a verified round trip; "B only" means the beacon was reached but not home. Times are mission simulated seconds.

| Maze | C1 | C2 | C3 | C4 | C0 |
|---|---|---|---|---|---|
| 10/0 | B only 480 | RT 231 | RT 291 | RT 144 | RT 118 |
| 11/0 | RT 125 | RT 407 | RT 352 | RT 187 | RT 134 |
| 12/0 | RT 165 | RT 141 | fail 480 | RT 165 | RT 150 |
| 13/0 | fail 480 | RT 255 | fail 480 | RT 332 | RT 294 |
| 14/0 | RT 341 | fail 480 | fail 480 | RT 230 | RT 207 |
| 15/0 | RT 241 | fail 480 | fail 480 | RT 190 | RT 220 |
| 16/0 | RT 106 | RT 100 | RT 318 | RT 101 | RT 302 |
| 17/0 | RT 244 | fail 480 | fail 480 | RT 272 | RT 197 |
| 18/0 | RT 169 | fail 480 | RT 301 | RT 179 | RT 170 |
| 19/0 | RT 144 | RT 154 | RT 397 | RT 142 | RT 293 |
| 20/0 | RT 156 | fail 480 | RT 207 | RT 139 | |
| 21/0 | RT 194 | RT 173 | RT 340 | RT 213 | |
| 22/0 | RT 175 | RT 193 | RT 315 | fail 480 | |
| 23/0 | RT 111 | RT 111 | RT 316 | RT 154 | |
| 24/0 | RT 126 | RT 130 | RT 182 | RT 126 | |
| 25/0 | RT 126 | RT 109 | RT 208 | RT 99 | |
| 26/0 | RT 231 | B only 480 | RT 456 | RT 334 | |
| 27/0 | RT 155 | fail 480 | fail 480 | RT 140 | |
| 28/0 | RT 157 | fail 480 | RT 198 | RT 149 | |
| 29/0 | RT 152 | fail 480 | fail 480 | RT 121 | |

## Failure taxonomy by mechanism

Every failure is a 480-s timeout; none involves contact or pose loss. Each was diagnosed from its preserved records: the remaining true-map shortest path at 480 s, hold fraction and 30-s progress windows (`timeout_diagnosis.json`).

| Mechanism | C1 | C2 | C3 | C4 | Episodes |
|---|---:|---:|---:|---:|---|
| **No eligible movement under the view requirement.** Translations are view-restricted, turns clearance-blocked, and the robot holds. | 0 | 6 | 4 | 0 | C2 15, 17, 18, 20, 26 (return), 28; C3 12, 13, 14, 15 |
| **Hold outscores every movement** (C3's own predicted progress) | 0 | 0 | 2 | 0 | C3 27, 29 |
| **Latched recovery turn blocked by forecast clearance** | 1 | 0 | 1 | 0 | C1 13; C3 17 |
| **Terminal heading limit cycle at the goal.** The robot turns in place 3–5 cm from the target, and arrival never confirms. | 1 | 3 | 0 | 0 | C1 10 (home); C2 14, 27, 29 |
| **Turn oscillation without progress** | 0 | 0 | 0 | 1 | C4 22 |
| **Total** | 2 | 9 | 7 | 1 | |

- **Budget.** No failure shows steady progress with little holding, the handoff's budget-flag test. The 480-s budget is not implicated.
- **Two rule labels to read alongside the table.** The fixed rules label C3 27/0 "slow but progressing" (it made 4.2 m of progress in the final 2 minutes, but only after a 330-s stall at 73% hold) and C3 29/0 "other stall" (95% hold). Both are the hold-outscores-movement stall above; the rules had no category for it.
- **What the shared harness contributes.** The view-requirement deadlock also traps C2, which has no motion predictor. The latched recovery turn also traps C1. The terminal limit cycle traps both C1 and C2. So these are harness mechanisms that each controller hits at a different rate. C0 met them too but always escaped: 19/0 had 67 no-eligible holds and 16/0 had 489 view-restriction holds.

## Why C3 and C4 differ: prediction-slot evidence

These analyses read logged decisions only. They are for interpretation and change nothing.

**Translation predictions at 800 ms.** Medians over each controller's own validation states (C0 5,094, C3 17,936 and C4 9,520 decisions), with C1's logged forecast for the same inputs:

| | Hold | Forward | Arcs | Turns |
|---|---|---|---|---|
| C0 (truth) vs C1 forecast | 61 vs 58 mm | 129 vs 124 mm | 116 vs 107–115 mm | 58–65 vs 54–62 mm |
| C4 vs C1 forecast | 19 vs 20 mm | 57 vs 82 mm | 40–56 vs 65–73 mm | 14–28 vs 17–24 mm |
| C3 vs C1 forecast | 9 vs 2 mm | **10 vs 66 mm** | **8–12 vs 49–57 mm** | 7–10 vs 4 mm |

- **C1 is a good stand-in for truth.** C0's true motion matches C1's forecast.
- **C4 is close.** It under-predicts forward progress by about 30%.
- **C3 barely separates movement from holding** in the states it visits: forward and arcs are about one-sixth of kinematic. This fits the earlier finding that its readout underestimates progress. It explains the "hold outscores movement" stalls, and C3's 49% outbound hold rate against C1's 14%. In the deadlocks, C3's small spurious translation for holds and turns also puts their predicted footprints into wall margins that clearance forbids.

**Is C4 mostly kinematic?** It receives the candidate command tape directly.
- **Agreement with C1.** Over all 9,520 C4 decisions, C4 and C1 agree to 2–4 mm on the shared committed prefix. After the candidates diverge, they differ by a median 18 mm at 800 ms, which is about the size of the between-candidate spread. Their forward-progress rankings match (Spearman 0.94), and they pick the same best-forward candidate 63% of the time.
- **Visual swap.** This covers 241 decisions from 10/0, 11/0 and 12/0, with inputs regenerated by verified replay; C4 with its own features reproduces the log exactly. Swapping in the visual features of another maze, or of the same maze 16 s later, changes C4's 800-ms prediction by a median **5.9 mm** (p90 17–18 mm; about 0.8° of yaw). Swapping its command history instead changes it by **16.1 mm** (p90 60 mm). C4's own candidate spread is 18.5 mm.
- **Reading.** C4 is predominantly a kinematic model. Vision moves its predictions by about a third of the between-candidate differences.

## Proposed intervention for C3 (one; not implemented, needs approval)

**Recalibrate C3's motion readout on standstill starts and in-place turns.** The encoder and predictor stay frozen. The readout is refit only on training-role data that covers starts from rest and pure rotations, the situations where C3's translation predictions collapse.

**Acceptance is offline and precedes any closed-loop run.** On development-role logged states with C0 physics truth, which the qualified branch tooling supplies, the refit readout's 800-ms translation for forward and arc candidates from rest must come within 25% of truth. Its spurious translation for hold and in-place turns must not exceed C1's. Validation must not be reused for this: it has now served qualification.

**Rationale.** This targets the measured mechanism behind all seven C3 failures: collapsed translation predictions leading to hold-dominated selection and clearance-blocked turns. It isolates the readout-gap hypothesis (progress report H1) without touching the representation.

**Not proposed here.** Harness escapes for the shared traps would change every controller, so they are harness decisions (see the E1 proposal), not a C3 intervention.

## Videos (replay-verified)

**Selection.** The pre-registered rule picks the lowest-ID validation episode that C1–C4 all completed: **11/0**. C1 failed 10/0, so 11/0 is the lowest common success. Each video comes from a deterministic replay on the frozen V4 session and controller runtime. Every consumed packet record, decision, dispatch command and reason, applied command, published pose and mission row was identical, the native trace was exact, and the error was 0.0 mm / 0.0°.

The format is 1920×1080 at 30 fps, H.264 yuv420p, with a labelled 4× cut, a contact sheet and a metadata JSON.

| Controller | Packets / decisions / dispatch steps verified | Frames | Files (under `videos/` on RecoveryStorage) |
|---|---|---|---|
| C1 (18/20) | 1,250 / 304 / 6,246 | 3,748 | `capability_C1_val11_ep0_attempt001/` |
| C2 (11/20) | 4,069 / 997 / 20,341 | 12,205 | `capability_C2_val11_ep0_attempt001/` |
| C3 (13/20) | 3,518 / 870 / 17,586 | 10,552 | `capability_C3_val11_ep0_attempt001/` |
| C4 (19/20) | 1,868 / 454 / 9,336 | 5,602 | `capability_C4_val11_ep0_attempt001/` |
| Composite 2×2 (C1, C2 / C3, C4) | Built from the four verified videos; shorter missions hold their final frame | 12,205 (406.8 s) + 4× cut | `capability_composite_val11_ep0_attempt001/` (`3b9389b9922e…`) |

## Records, errata and caveats

**Records:**
- [Gate result](go2_navigation_capability_completed_support_v4_gate_result_2026-09-28.md)
- [Version ledger](go2_navigation_capability_version_ledger_2026-09-28.md)
- [Harness change log](go2_navigation_capability_harness_change_log_2026-09-28.md)
- [Qualification plan](go2_navigation_capability_qualification_plan_2026-09-28.md)
- [C0 prefix erratum](go2_navigation_capability_oracle_prefix_erratum_2026-09-28.md)
- Analysis outputs under `analysis/qualification_v4_2026-09-28/` on RecoveryStorage: `qualification_analysis.json`, `timeout_diagnosis.json`, `isolated_latency.json` and `c4_inputs/`.

**Reactive-hold reader erratum.** The frozen hold classifier could not read C2's reactive records; the [evaluator-only correction](go2_navigation_capability_reactive_hold_reader_erratum_2026-09-28.md) handled them. 76 scored episodes and 7,092 hold rows reproduced exactly. Validation paused for about 25 minutes, and no mission was rerun.

**Concurrency.** Validation ran up to five owners at once. The four video replays reproduced their concurrently run missions bit for bit, which confirms concurrency did not change outputs.

**Sensor idealisation** (from the ground-truth audit):
- No C1–C4 decision reads true pose, map or contacts.
- The simulated sensors are noise-free RGB, depth with 2 mm noise, and an ideal gyro.
- An **ideal accelerometer** sets the initial gravity direction once at the start. It is not in the handoff's sensor list and should be declared.
- An evaluator-only native guard (contact, speed or domain) could end an episode. It never fired.

**Provenance.** The training renders behind C3's predictor and readout, and behind C4, are unverified. Carry this caveat into any attribution of C3 or C4 outcomes.

**Budget.** 86.2 of 160 active hours were used by the end of validation, and about 87.0 after the latency benchmark, the swap test and the videos.
