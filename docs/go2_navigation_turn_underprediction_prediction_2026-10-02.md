# Prediction: turns-only forecast under-prediction (written before results), 2 October 2026

**Status: prediction recorded at 12:12 BST, 2 October 2026, before reading any result.** One mission of each turns-only cohort had finished, and neither had been opened. PRELIMINARY: development mode, prelim_test_v1 mazes 30–49, C1, recovery off, coverage-rule fix.

**Cohorts.**
- `sens_turnscale0p5` and `sens_turnscale0p25`.
- C1's forecast for the two in-place turns is scaled by 0.5 and 0.25 (displacement and heading change). Every other move is unchanged.

## Prediction (Andrew, 2 October)

**The reasoning.**
- Every planner-stage clearance check is a 0.45 m disc on the forecast's centre path, and it ignores heading.
- An in-place turn barely moves the centre: its forecast centre drift is about 1 cm over a 400 ms commit.
- So under-predicting turns should change **what the planner selects, and arrival** (turning to face the goal, scanning, terminal heading settling). It should **not** change clearance.

## Criteria (fixed now; each condition is compared with the clean baseline `sens_base`, same mazes)

**Clearance unchanged (predicted to hold at both levels):**
1. No disallowed contacts, and no hard (5 mm) or operating (20 mm) clearance violations.
2. Close approaches while moving (worst 5 per mission, at least 1 s apart):
   - the median is within 1.5 cm of the clean baseline's 16.2 cm;
   - the worst is at least 5.5 cm (the clean worst, 7.5 cm, minus 2 cm).
3. The disc holds against the true walls at every approach: the base centre is at least 0.45 m from the nearest true wall.
4. In the safety error budget, the centre-forecast error term for in-place turns grows by less than 1 cm.

**Selection and arrival change (predicted):**
5. Choice-change rate at least 0.05. The clean rate is 0 by construction.
6. Arrival is affected by at least one of the following, against the clean baseline:
   - success drops by at least 2 mazes;
   - the strict-vs-reached gap grows;
   - the median round trip lengthens, with the paired 95% interval excluding zero.

## Competing mechanism (stated in advance)

**Exposure.** Under-predicted turns may make the planner turn longer or more often near walls: the real turn goes 2–4× further than predicted, so heading targets overshoot and turns oscillate. Forward-only under-prediction already did this: turns went from 32% to 51% of moving time at forward × 0.25, and close approaches below the clean 10th percentile went from 10 to 35.

If criterion 2 fails while criteria 3 and 4 hold and turning time has risen, read it this way: the clearance prediction failed through more exposure to near-wall turning, not through the check. That would still be a failure of "not clearance", and it will be reported as one.

## Result (added 14:05 BST, after both cohorts finished; the prediction above is unchanged)

**Sources.**
- `scripts/diagnose_go2_forecast_sensitivity_failures_development.py`
- `scripts/analyse_go2_forecast_sensitivity_close_approaches_development.py`
- `scripts/analyse_go2_forecast_sensitivity_error_budget_development.py`
- `scripts/report_go2_forecast_sensitivity_development.py` (choice change)
- `scripts/report_go2_prelim_results_development.py` (paired bootstrap)

All are compared with the clean baseline `sens_base` on the same 20 mazes.

| Criterion | Turns × 0.5 | Turns × 0.25 |
|---|---|---|
| 1. No contacts or violations | **held** (0 · 0 · 0) | **held** (0 · 0 · 0) |
| 2. Close-approach median within 1.5 cm of 16.2 cm; worst ≥ 5.5 cm | **held**: 15.4 cm · 8.2 cm | **held**: 16.2 cm · 8.5 cm |
| 3. Disc held against true walls at every approach | **held** (0 of 100 broken) | **held** (0 of 100 broken) |
| 4. Turn centre-forecast error term (p95) up by less than 1 cm (clean 1.3 cm) | **held**: 1.8 cm (+0.5) | **failed**: 4.0 cm (+2.7) |
| 5. Choice-change rate ≥ 0.05 | **failed, narrowly**: 0.048 | **held**: 0.080 |
| 6. Arrival affected | **held**: 3 mazes lost (2 pose losses after in-place turning, 1 hold), paired success −0.15 (−0.30 to 0.00); round trip +23 s (+8 to +41) | **failed**: no success change; round trip +10 s (−3 to +25); no strict-vs-reached gap |

**Reading.**
- **Clearance held at both levels.** Criteria 1–3 hold: no contacts, no violations, approach clearances within the clean range, the disc never broken.
- **The prediction's mechanism was quantitatively wrong at × 0.25.** In-place turns move the centre by a few centimetres, not about 1 cm. Under-predicting that drift raised the turn term of the error budget by 2.7 cm at p95. The disc check still absorbed it, and the error budget had no violations: slack min 7.0 cm, tight-bound slack min 0.0 cm.
- **The selection-and-arrival side held only in part, and not monotonically.**
  - × 0.5: 4.8% of choices changed. It cost 3 successes (2 of them shared-system pose losses after in-place turning) and 23 s per round trip.
  - × 0.25: more choices changed (8.0%), but there was no detectable arrival effect. Turn moving-time share was 0.42 at × 0.5, against 0.33 clean and at × 0.25.
- **The exposure risk named in advance appeared only at × 0.5,** as more near-wall turning. Approach clearance still stayed within criterion 2.
