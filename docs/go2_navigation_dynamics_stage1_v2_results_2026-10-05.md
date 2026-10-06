# Dynamics stage 1 on the fixed harness: results (5 October 2026)

**PRELIMINARY.** prelim_test_v1 mazes 30–49, episode 0, recovery off, harness `reserve_exit_v2` (the in-place turn exit), pinned launch `b5e70ae9`. Uniform floor friction μ, zero-shot: C1, C3 and C4 as they are.

**Question:** when commands no longer determine motion, does C3's or C4's visual prediction beat C1's command-only prediction?

## Answer

**No, not zero-shot.** At μ = 0.3 all three forecasters have the same error by movement type.
- They over-predict straight travel by 16–22% and under-predict the translation during turns by about half.
- C3 and C4 were trained at nominal friction, and nothing in the image signals a uniform friction change.
- Stage 2 (marked patches, matched refit) is the test of whether a visual cue can be learned.

## Outcomes

| Cohort | C1 | C4 | C3 | C2 |
|---|---|---|---|---|
| Normal friction, v2 reference (`rexit2_ref`) | 20/20 | 19/20 | – (not run, Andrew 4 Oct) | 15/20, 1 contact stop |
| μ = 0.3 (`dyn1v2_mu030_cpu`, `dyn1v2_mu030_c3`) | 16/20, 4 contact stops | 15/20, 4 contact stops | 13/20, 5 contact stops, 2 budget exhaustions | – |
| μ = 0.25 (`dyn1v2_mu025_cpu`) | 1/20, 19 contact stops | 1/20, 18 contact stops | skipped (Andrew, 5 Oct) | – |

**Contacts and clearance.**
- Every completed run had 0 contacts.
- All 51 contact stops were scored with the failure reader.
- Minimum wall clearance on completed runs:

| Cohort | C1 | C4 | C3 | C2 |
|---|---|---|---|---|
| Normal friction | 7.5 cm | 2.9 cm | – | 5.9 cm |
| μ = 0.3 | 5.5 cm | 0.6 cm | 0.4 cm | – |
| μ = 0.25 | 7.7 cm | 0.6 cm | – | – |

- The C3 near misses: maze 48 (0.4 cm, a budget exhaustion) and maze 43 (0.6 cm, a success).

**C3's two non-contact failures** (mazes 44 and 48) were budget exhaustion after about 1,000 blocked-recovery stalls each. The robot made progress, then stayed held near a wall.

## Every contact stop has one cause: the v2 turn exit at low friction

- **Which part:** a rear calf against a wall in all 51 stops (40 rear-left, 11 rear-right), a median 65 s into the mission.
- **The motion:** an in-place turn every time, with base speed at most 0.10 m/s.
- **What allowed it:** every stop came within 2 s of a turn that only the v2 turn exit passed. The turn started with the centre 0.36–0.44 m from the wall, inside the 0.45-m disc, mostly while scanning.
- **Why the check missed it:** the exit checks the centre path. The legs sweep outward, and at low friction the rear feet slide.
- **At normal friction, C2 hit the same mechanism once.**
- **Status:** recorded as a known limitation, with no change mid-experiment. A v3 turn check on the legs' swept footprint is required before the confirmatory sealed run.

## Forecast error by movement type

Median 700-ms XY error (mm) and median predicted/true translation ratio, on decisions whose requested commands ran as planned (`scripts/score_go2_dev_closed_loop_prediction_development.py`).

| | All | Turn | Cruise | Arc | Switch | Decisions |
|---|---|---|---|---|---|---|
| **Normal friction (v2 reference)** | | | | | | |
| C1 | 6 ×0.96 | 5 ×0.75 | 4 ×1.00 | 7 ×0.95 | 7 ×0.96 | 7,178 |
| C4 | 5 ×1.01 | 3 ×0.88 | 6 ×1.02 | 6 ×1.03 | 7 ×1.01 | 8,934 |
| **μ = 0.3** | | | | | | |
| C1 | 19 ×0.92 | 12 ×0.54 | 40 ×1.17 | 21 ×0.91 | 25 ×1.01 | 6,831 |
| C4 | 19 ×1.00 | 15 ×0.47 | 43 ×1.22 | 17 ×0.98 | 25 ×1.06 | 6,714 |
| C3 | 16 ×0.89 | 14 ×0.56 | 41 ×1.16 | 19 ×0.97 | 27 ×0.96 | 9,440 |
| C1A v2 (adaptive C1, validation) | 18 ×0.89 | 12 ×0.55 | 37 ×1.11 | 20 ×0.92 | 25 ×0.98 | 7,117 |
| **μ = 0.25** | | | | | | |
| C1 | 13 ×0.63 | 13 ×0.44 | 60 ×1.42 | 22 ×1.07 | 34 ×1.18 | 4,748 |
| C4 | 17 ×0.88 | 15 ×0.40 | 71 ×1.51 | 25 ×1.18 | 35 ×1.23 | 4,193 |

Notes:
- **C3's lower "all" figure is a mix effect.** C3 has 2,969 hold decisions (12 mm), most from the two stalled missions. By type, C3 is no better than C1.
- **At μ = 0.25 the "all" figures are low** because most missions stop early, in scan-heavy openings (holds and turns).
- **C1A v2,** the adaptive baseline, gains little at uniform friction. Its single translation ratio stays near 1.0 because the error depends on the movement (see the plan, "C1A v2 validation").
- **Context:** the μ = 0.2 run on v1.1 froze every mission before prediction could matter ([results](go2_navigation_dynamics_stage1_results_2026-10-04.md)).

## Decisions taken (Andrew, 5 October)

- C3 at μ = 0.25 skipped.
- Turn-exit contacts recorded as a limitation; v3 swept-footprint turn check required before the confirmatory sealed run.
- Stage 2 go:
  - patches on straight segments;
  - primary measure: forecast error by distance to the patch edge, marked versus unmarked;
  - an adaptive C1 baseline (C1A).
- Stage-2 layout sets: a selected family, "routes with long straights" (24 fit, 6 held-out, 20 evaluation), plus a no-patch reference on the evaluation mazes.

See [the plan](go2_navigation_dynamics_perturbation_plan_2026-10-02.md) for the stage-2 build, the layout registration and the open custody question: the new layouts cannot be checked against `sealed_test_v2` before training.
