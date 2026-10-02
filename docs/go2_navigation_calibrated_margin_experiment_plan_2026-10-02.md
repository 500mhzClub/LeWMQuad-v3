# Plan: calibrated clearance margins and pessimistic unknown cells (DRAFT, not run), 2 October 2026

**Status: approved (Andrew, 2 October) with staging and adjustments; see the final section.** Stage 1 for C1 and C4 is running. Requested by Andrew (2 October) after the [safety error budget](go2_navigation_forecast_sensitivity_safety_budget_2026-10-02.md). Development mode: all results will be PRELIMINARY.

## Why

**The disc's spare margin is smaller than the forecast error.**
- Every planner-stage clearance check is a heading-blind 0.45 m disc on the forecast centre path. It leaves only 2.5 cm over the body's 42.5 cm reach (5.5 cm with the reserve).
- The 95th-percentile centre forecast error near walls is 2.1 cm (C1), 2.6 cm (C4) and 5.3 cm (C3).
- So at p95 the budget is negative for every controller: −0.3, −0.8 and −3.7 cm.

**Today's safety is therefore implicit.** It comes from the planner usually working above the disc.

**What a calibrated margin does.** It inflates the disc by each controller's own conformal bound on its forecast error, using the same rule for all. That makes the forecast term explicit and moves its cost onto liveness, where controllers can be compared on success, holds and time.

**The pessimistic-unknown variant closes the other open term:** never-observed cells counting as free.

## 1. Calibration

**Score.** e_f for each executed moving decision: the largest error, over the 800 ms check horizon, between the controller's applied forecast centre path and the true centre path, both anchored at the true pose. This is the measure from the budget.

**Stratum.** Near-wall decisions only: the checked path's true clearance below 60 cm, where the check binds. All-decision quantiles are reported alongside.

**Quantile.** The split-conformal bound is the ⌈(n+1)(1−α)⌉-th smallest near-wall score, for α = 0.05 (p95) and α = 0.01 (p99).
- Decisions within a mission are correlated, so each bound also gets a maze-block bootstrap interval.
- The guarantee assumes exchangeability. The evaluation runs therefore report realised coverage: the share of their own near-wall decisions with e_f at or below the bound.

**Which data serve for what (disjoint mazes):**

| Role | Data | Why |
|---|---|---|
| **Calibration** | Preliminary run, recovery on, mazes 50–89 (prelim_test_v1). Already recorded: 40 missions per controller, about 10,600–11,600 near-wall decisions each. Decisions issued by recovery interventions (backups, escapes, latch releases) are excluded. | Same controllers, model versions and harness, held-out mazes, no new compute. |
| **Evaluation** | New runs on mazes 30–49, recovery off, coverage-rule fix. | The same 20 mazes as the forecast-sensitivity experiment, so the clean C1 baseline and the error budget line up. |
| **Reference only** | Preliminary run, recovery off, mazes 30–49. | Predates the coverage-rule fix, so not a baseline. Used only to check that the bounds transfer. |

**Provisional bounds** (computed now on the calibration split; the final ones are recomputed and committed before any evaluation run):

| Controller | Near-wall decisions | Conformal p95 (block-bootstrap 95%) | Conformal p99 (block-bootstrap 95%) | Same quantiles on mazes 30–49 (reference) |
|---|---:|---|---|---|
| C1 | 11,568 | 2.11 cm (2.02–2.24) | 2.90 cm (2.54–3.04) | 2.08 · 2.63 cm |
| C4 | 10,614 | 2.64 cm (2.58–2.68) | 3.44 cm (3.34–3.53) | 2.60 · 3.47 cm |
| C3 | 10,572 | 5.39 cm (5.27–5.50) | 7.19 cm (6.96–7.52) | 5.26 · 7.13 cm |

## 2. The margin rule (the same for every controller)

**What changes.** Every forecast-based planner-stage clearance requirement r becomes r + q̂_α(controller):
- the memory forecast clearance (0.45 m, or 0.48 m with the turn and translation reserves);
- the reserve-recovery thresholds;
- the hold-relative recovery footprints;
- the route planner's connector radius, so that routes and checks agree.

**What stays the same:** the forecast-independent checks.
- the dispatch-time depth stop (0.45 m requested-speed connector);
- the stopping projection (requested speed);
- the coverage rule.

**Feasibility, checked now.** All 20 evaluation mazes remain routable from home to beacon at centre clearances up to 0.57 m, which is 0.48 m plus 9 cm. So even C3's p99 margin makes no maze geometrically impossible.

**Implementation.**
- A development mixin that binds the margin into the frozen functions through a context variable, as the coverage fix does.
- A synthetic test confirming the requirement is r + q̂ and the forecast-independent checks are unchanged.
- A `--margin p95|p99` option on the mission and cohort entries.
- A calibration script that writes the bounds JSON, committed before evaluation.

## 3. Pessimistic-unknown variant

**Rule.** In the same planner-stage checks, a never-observed 5 cm cell within reach counts as occupied. Within reach means inside the checked disc of the forecast path and within 0.5 m of the robot. Two areas are seeded as known free:
- the start disc of 0.50 m, because the episode generator rejects spawns closer than 0.5 m to a wall;
- the robot's own track: cells within 0.20 m of any past estimated base-centre position, under the body core.

Without seeding, the rule would deadlock the initial look-around: the floor under and around the start pose can't be seen from there.

**Estimated effect**, from the offline reconstruction ([budget doc](go2_navigation_forecast_sensitivity_safety_budget_2026-10-02.md), `scripts/analyse_go2_unknown_cells_development.py`):

| Run | Controller | Decisions the variant would newly block (all · turns · after the first 30 s) | Unsafe passes it would remove |
|---|---|---|---:|
| preliminary, recovery off | C1 | 0.7% · 2.2% · 0.0% | 0 |
| preliminary, recovery off | C3 | 1.2% · 3.4% · 0.2% | 0 |
| preliminary, recovery off | C4 | 0.8% · 2.2% · 0.0% | 0 |
| sensitivity, noise 160 mm | C1 | 12.5% · 13.2% · 6.0% | 4.7% of decisions |

**Expected outcome.** Little liveness cost and little safety change at nominal forecasts, because no hidden wall was ever within reach. It guards the blind zone when forecasts are poor. These figures are the share of decisions whose selected move would be blocked. The planner would then pick another move, so the effect on liveness should be smaller.

**Implementation.** The map snapshot's floor and occupied cells define what has been observed; the track comes from the estimated poses. A mixin plus tests, about half a day.

## 4. Design

- **Controllers:** C1, C3 (the C3-v2 decoder) and C4 (C4-v2), as in the preliminary run.
- **Common settings:** mazes 30–49, recovery off, coverage-rule fix, and the same episode seeds as the sensitivity experiment.
- **Stage 1, margins** (unknown cells free): nominal, q̂95 and q̂99 per controller.
  - The nominal C1 baseline already exists (`sens_base`).
  - Nominal C3 and C4 need new runs, because the preliminary recovery-off runs predate the coverage fix.
- **Stage 2, pessimistic unknown:** at nominal, and at the margin chosen in stage 1.

## 5. What is reported

**Safety side:**
- contacts, and hard (5 mm) and operating (20 mm) violations;
- close-approach clearance (worst, p5, median), the rear-calf share, and the share out of the cameras' view;
- error-budget slack and violations, recomputed with the inflated disc;
- realised conformal coverage on the evaluation decisions;
- depth-stop catches on planned translations.

**Liveness side:**
- strict and reached success, SPL and median round trip;
- hold rate and time shares (progressing, holding, turning in place, wandering);
- forecast-rejected moves per decision, and freeze failures;
- pose loss, reported separately as a shared-system failure.

**Comparisons:**
- Each condition against the same controller's nominal run on the same mazes: maze-level paired bootstrap, with discordant counts.
- Across controllers, the safety–liveness trade-off: clearance p5 against success and time, over the margins.
- The question it answers: does pricing in its larger forecast error cost C3 more liveness than C1 and C4?

## 6. Cost

**Inputs:**
- **Mission wall time** (preliminary run, median and max): C1 7 min (20); C4 28 min (90); C3 55 min (204), at most 2 C3 missions at once on the one GPU.
- **Memory:** launches wait for 17 GiB free.

| Step | Engineering | Compute |
|---|---|---|
| Calibration script and committed bounds | 0.5 day | minutes (existing logs) |
| Margin mixin, option, synthetic tests | 0.5–1 day | — |
| Pessimistic-unknown mixin, tests | 0.5 day | — |
| Stage 1: C1 2 runs, C3 3, C4 3 (× 20 mazes) | — | **C3 ≈ 27 h on 2 GPU lanes (critical path)**; C4 ≈ 6 h and C1 ≈ 1 h in parallel on CPU |
| Stage 2: 2 conditions × 3 controllers × 20 | — | C3 ≈ 18 h; C1 and C4 ≈ 4 h in parallel |
| Disk | — | about 2–4 GB per 20-mission run. 14 new runs need about 40 GB against today's 58 GB free and the 12 GiB reserve, so a retention review comes before stage 2. |

**Stage 1 exceeds the 24 h threshold for asking first**, because of C3. Cheaper options:
- **(a)** Run stage 1 without q̂99. C3 needs nominal plus q̂95, about 18 h.
- **(b)** Run C3 on 10 of the 20 mazes, about 14 h for all three conditions.
- **(c)** Run C1 and C4 through both stages first (about a day of CPU), then C3 only at nominal and q̂95.

**Recommendation:** (c). C1 and C4 test the rule cheaply. C3 is where the liveness cost is expected, and its two runs fit in about 18 h.

## Approved: staging, adjustments and deviations (2 October)

**Staging.**
- C1 and C4 go through both stages first, on CPU, starting now behind the memory gate.
- C3 runs at nominal and at the p95 margin once the sensitivity queues free the GPU.

**Adjustments.**
1. **Both levels for C1 and C4.** Both p95 and p99 margins are reported.
   - The bound is per decision, not per mission. A mission makes hundreds of near-wall decisions, so at p95 several exceedances per mission are expected.
   - **p99 is the safety-relevant level.**
2. **Realised exceedance** on every evaluation run: the share of near-wall decisions whose actual e_f exceeded the controller's bound. This is the validity check for the calibration under the changed closed-loop behaviour.
3. **Possible shift.** Calibration ran with recovery on and the nominal disc; evaluation runs with recovery off and the inflated disc.
4. **Stage 2 seeding.** Seeded as known free are only the robot's own reach disc at the start (0.425 m) and its traversed track (cells within 0.20 m of past planning positions). The generator's 0.5 m spawn guarantee is privileged and is not used. The report states whether the initial look-around still works.

**Implementation as built.**

| Item | Commit | Notes |
|---|---|---|
| Bounds | `bce6b386` | Committed before any evaluation run: C1 2.10 / 2.89 cm, C4 2.64 / 3.44 cm, C3 5.39 / 7.19 cm |
| Margin mixin and `--margin p95\|p99` | `e99a6bde` | |
| Pessimistic-unknown mixin and `--pessimistic-unknown` | `17a55901` | Synthetic test at the start pose: the unseen ring between 0.425 m and 0.5 m blocks an in-place turn, which predicts that the look-around deadlocks. Being checked live. |

**Deviation from the draft: the routing graph is not inflated.**
- Its 0.45 m radius is spread across six routing layers.
- The margin applies where the safety decision is made: the forecast-based action check (inherited by the turn reserve and recovery modes) and the route-target lookahead.
- A route/check conflict would show up as forecast-clearance holds, which are reported.

**Code identity.** Missions launched after `e99a6bde` load the new module.
- Outside the new options the new code is inert (synthetic test).
- One live identity check (C1, maze 30, no new options) is compared decision by decision with the clean baseline mission.

**Cohorts.**

| Cohort | Content |
|---|---|
| `margin_c4_nominal` | C4 × 20, the nominal reference with the coverage fix |
| `margin_p99` | C1 and C4 × 20 each |
| `margin_p95` | C1 and C4 × 20 each |
| `identity_margin_code` | the identity check |
| `pessimistic_smoke` | C1, mazes 30–31 |

The C1 nominal reference is `sens_base`.


## Stage 2 revised and process changes (Andrew, 2 October, afternoon)

**Stage 2 runs for C1 and C3 only.**

**Smoke tests that led here** (C1, mazes 30–31):
- **Seeding only the 0.425 m start reach disc deadlocked**, under both the first and the revised rule.
- **First rule, unseen cells counted as walls:** confirmed live on maze 30. No clear move in 1197 of 1197 decisions; 18° of turning in a minute; never translated.
- **Revised rule, unseen cells block within reach + bound:** the synthetic test blocks the start turn. Its live smoke test was still running when this was written.

**Seeding now** (`lewm/dev_pessimistic_unknown_seeded_development.py`): an **operating precondition, the same for every controller**. The robot is placed in a cleared 0.5 m area. The start disc of that radius is seeded at 1 cm resolution and never beyond 0.5 m, together with the traversed track. The [limitations note](go2_navigation_harness_v4_known_limitations_2026-09-29.md) records this, and that the physical Go2's wide-view lidar would make it unnecessary.

**Why 0.5 m and not exactly the blocking radius:**
- C1's start turn forecasts drift about 1 cm.
- Seeding at C1's blocking radius (0.454 m) would still block its start turn.
- 0.5 m covers every controller's blocking radius (0.45–0.50 m).

**Prediction for C3** (synthetic test):
- At its p99 bound (blocking radius 0.497 m), C3's start turn is blocked even without drift.
- At p95 (0.479 m) it passes without drift but not with 1 cm.
- C3's initial-panorama turn forecasts drift 0.9 cm at the median and 6.6 cm at p99.
- So C3 may not leave the start under this rule. That would be a result about C3's forecast under the rule.

**Process: pinned launches, from `6a83b937`.**
- **No edits to live harness code while missions can launch;** new behaviour goes in new files.
- **Every batch launches through `scripts/launch_go2_dev_cohort_pinned_development.py`.** It refuses unless the runtime files equal HEAD, and records the commit and file hashes in `<name>_launch_pin.json`, `config.json` and `result.json`.
- **Missions run through `scripts/run_go2_dev_mission_pinned_development.py`.** It re-verifies the hashes before any repository import, refuses on mismatch, and records `runs/<assignment>/launch_pin.json`.
- **Launches are staggered,** and the wall-clock ledger is locked.
- **No git worktree:** AGENTS.md forbids worktree or checkout copies while legacy sealed blobs remain tracked.

**Batches launched before the pin.** Identify them post hoc by the module and entry hashes in each `dev_run.json`. Their behaviour without the new options is unchanged: the identity re-run of clean maze 30 was decision- and physics-identical.

**C4 nominal relaunched.**
- `margin_c4_nominal` never started: launched at the same instant as `margin_p99`, it failed the shared wall-ledger write. Its `config.json` is kept as the record.
- The reference runs as `margin_c4_nominal_b` (pinned).
- `margin_p95`, and the three remaining sensitivity cohorts (noise 80 mm, scale 0.75×, turns × 2.0), also run pinned.
