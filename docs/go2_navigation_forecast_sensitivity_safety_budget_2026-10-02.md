# Safety as an error budget: forecast-sensitivity cohorts (interim), 2 October 2026

**PRELIMINARY.** Development mode, prelim_test_v1 mazes 30–49, C1 with degraded forecasts, recovery off, coverage-rule fix.

**Interim.**
- The uniform-scale cohorts are partial: 0.25× has 4 of 20 missions and 0.5× has 5.
- The turns-only cohorts are excluded until they finish. They are checked against a prediction committed beforehand (commit `ddc4e5cc`, [prediction](go2_navigation_turn_underprediction_prediction_2026-10-02.md)).
- Regenerate with `scripts/analyse_go2_forecast_sensitivity_error_budget_development.py --markdown OUT`.

## The budget

    min body clearance ≥ (disc radius − max body reach) − e_f − e_p − e_m

| Term | What it is | Measured value |
|---|---|---|
| **Disc radius** | Every planner-stage clearance check requires the forecast centre path to stay more than this far from remembered obstacle cells. The heading is not used. | 45 cm; 48 cm with the turn/translation reserve (45 cm in the reserve-recovery modes) |
| **Max body reach** | Largest horizontal distance of any of the 27 collision primitives from the base centre, over all moving samples in all cohorts | **42.5 cm**. At close approaches: median 41.4 cm in any direction, 41.2 cm toward the wall |
| **Static margin** | Disc − reach | **2.5 cm** (5.5 cm at full reserve) |
| **e_f**, centre-position forecast error | The forecast centre path the check used (degraded where applicable) against the true centre path, both from the true pose; largest error over the 800 ms horizon | Per cohort, below: p95 2.1 cm clean, up to 32 cm at 160 mm noise |
| **e_p**, tracker error | Registered visual pose against the true pose at the decision: position error plus heading error times the path's extent | p95 0.5–1.2 cm in every cohort, max 2.6 cm. Heading error p95 below 0.1° |
| **e_m**, map error | Remembered minus true clearance of the same path, minus e_p, clipped at 0. Cells are 1 cm squares and distances are measured to the squares, so quantisation and inflation add only conservative error. What remains is walls misplaced at observation, or never observed. | Median 0. Usually negative before clipping: the map is about 1.5 cm conservative. Heavy tail of 40–155 cm when a nearby wall was never seen |

**Per-cohort terms** (every executed moving decision):

| Condition | Decisions | e_f: median · p95 · max (cm) | e_f p95: in-place turns · forward/arcs (cm) | e_p: median · p95 · max (cm) | e_m: median · p95 · max (cm) | Budget at p95 terms (cm) | Observed worst approach (cm) |
|---|---:|---|---|---|---|---:|---:|
| clean | 7314 | 1.0 · 2.1 · 3.5 | 1.3 · 2.3 | 0.3 · 0.8 · 1.1 | 0.0 · 0.0 · 130.6 | −0.4 | 7.5 |
| noise 10 mm | 7558 | 1.5 · 3.2 · 5.2 | 2.7 · 3.3 | 0.3 · 0.6 · 1.1 | 0.0 · 0.0 · 130.6 | −1.3 | 6.6 |
| noise 20 mm (partial) | 4779 | 2.4 · 4.9 · 8.3 | 4.7 · 5.1 | 0.3 · 0.7 · 1.0 | 0.0 · 0.0 · 40.8 | −3.2 | 8.2 |
| noise 40 mm | 10423 | 4.5 · 9.0 · 16.7 | 9.1 · 9.0 | 0.3 · 1.0 · 1.3 | 0.0 · 0.0 · 132.2 | −7.5 | 3.9 |
| noise 160 mm | 8039 | 14.0 · 32.3 · 55.3 | 32.4 · 31.2 | 0.1 · 0.5 · 0.9 | 0.0 · 11.7 · 155.5 | −42.0 | 4.2 |
| scale 0.25× (4/20) | 1666 | 9.6 · 11.3 · 13.0 | 5.2 · 11.4 | 0.5 · 0.9 · 1.2 | 0.0 · 0.0 · 40.4 | −9.7 | 9.2 |
| scale 0.5× (5/20) | 2123 | 6.7 · 8.4 · 9.5 | 3.4 · 8.6 | 0.3 · 0.6 · 0.8 | 0.0 · 0.0 · 40.4 | −6.5 | 14.1 |
| forward × 0.25 | 10912 | 1.7 · 10.2 · 13.2 | 2.0 · 10.6 | 0.3 · 1.2 · 2.6 | 0.0 · 0.1 · 130.6 | −8.9 | 6.1 |
| forward × 0.5 | 7491 | 6.8 · 8.5 · 9.8 | 1.5 · 8.7 | 0.3 · 0.7 · 1.0 | 0.0 · 0.1 · 130.6 | −6.8 | 7.4 |
| forward × 0.75 | 7519 | 3.9 · 5.2 · 6.4 | 1.2 · 5.3 | 0.3 · 0.8 · 1.2 | 0.0 · 0.0 · 130.6 | −3.5 | 6.1 |

## Checked against every close approach

**Method.** For each close approach (worst 5 per mission while moving, at least 1 s apart), the bound is computed from that decision's actual errors:

    (45 − 42.5 cm) − e_f at the approach time − e_p − e_m

- **The executing decision** is the last dispatched non-zero command. When the planner requests a hold, the slew limiter winds the previous move down over the next ticks. So the closest approaches at 40–160 mm noise, which happen in that wind-down after a turn, belong to that turn.
- **When the bound applies:** that move passed the check (remembered clearance above 45 cm), and the approach falls within its 800 ms forecast.
- **The tight form:** true clearance of the checked path − e_f − the body's largest reach at that instant. It is a strict geometric lower bound.

| Condition | Approaches | Bound applies (in slew wind-down) | Just past the 800 ms forecast | Violations | Slack: min · median (cm) | Tight-bound slack: min · median (cm) |
|---|---:|---:|---:|---:|---|---|
| clean | 100 | 100 (0) | 0 | 0 | 5.4 · 14.5 | 0.2 · 0.9 |
| noise 10 mm | 100 | 100 (0) | 0 | 0 | 5.1 · 14.2 | 0.1 · 2.2 |
| noise 20 mm (partial) | 50 | 50 (1) | 0 | 0 | 6.4 · 13.8 | 0.3 · 2.6 |
| noise 40 mm | 100 | 98 (11) | 2 | 0 | 4.2 · 12.6 | 0.5 · 4.4 |
| noise 160 mm | 100 | 93 (16) | 7 | 0 | 4.3 · 18.2 | 0.9 · 12.2 |
| scale 0.25× (4/20) | 20 | 20 (2) | 0 | 0 | 7.9 · 14.5 | 0.1 · 1.1 |
| scale 0.5× (5/20) | 25 | 25 (1) | 0 | 0 | 12.2 · 14.4 | 0.1 · 0.9 |
| forward × 0.25 | 100 | 100 (1) | 0 | 0 | 4.2 · 13.4 | 0.0 · 1.0 |
| forward × 0.5 | 100 | 100 (0) | 0 | 0 | 5.6 · 13.8 | 0.2 · 1.2 |
| forward × 0.75 | 100 | 100 (2) | 0 | 0 | 4.0 · 13.9 | 0.2 · 1.6 |

**No violations.** Every one of the 693 approaches the bound applies to sits at or above it, whether the lower or upper separation bound is used. The 9 approaches just past their forecast are the tail of a turn's wind-down, at ages of 0.80 s. They kept at least 5.0 cm.

**The budget explains the observed clearance closely.** The strict tight bound sits within 0–1 cm of the observed clearance at its minimum, and within 1–4 cm at the median (12 cm at 160 mm noise).

## What the budget says

1. **The static margin is thin, and the check alone guarantees nothing at p95 errors.**
   - The disc leaves 2.5 cm over the body's reach (5.5 cm at full reserve).
   - With p95 error terms, the budget is negative in every cohort: −0.4 cm clean, −7.5 cm at 40 mm noise, −9.7 cm at scale 0.25×.
2. **Observed clearance stays positive mainly because the planner usually works above the disc.**
   - The remembered clearance at close approaches has a median of 55 cm clean; only 1% of approaches are at 48–50 cm.
   - A secondary reason: the body's reach toward the wall (median 41.2 cm) is below its 42.5 cm maximum.
   - Errors rarely point at the wall.
3. **Degraded forecasts erode clearance by pushing the planner to the limit.**
   - The forecast-based check then rejects the other moves, so the moves left are the ones that barely pass.
   - The remembered clearance at approaches falls to a median of 50 cm at 40 mm noise and 48 cm at 160 mm. 52–63% of approaches sit at 48–50 cm.
   - The body's reach is unchanged (41.4 cm). This matches the close-approach analysis: the centre gets nearer the walls during turns.
4. **e_f is the term that grows with degradation.** Tracker error stays below about 1 cm. The map is conservative, except for walls never observed.
5. **The map's tail, walls never seen, only mattered at 160 mm noise.**
   - The question is how often a decision believed its path cleared the disc while, against the true walls, it didn't.
   - That never happened at 40 mm noise or less, or under any scale cohort; the closest such path still cleared by 45.1 cm.
   - At 160 mm noise it happened in 375 of 8039 decisions, with true clearance down to 14.9 cm. Those noisy forecast paths pointed into unobserved space, mostly beside and behind the robot, where the map holds no walls.
   - The robot did not follow those paths: its real motion was the nominal one, and its closest approach was 4.2 cm.
   - This is the same blind zone as the depth cameras (see the [known-limitations note](go2_navigation_harness_v4_known_limitations_2026-09-29.md)).

## Caveats

- **Separation** is the frozen reader's 500 Hz lower bound, with its upper bound used for definite violations. Near wall corners it can understate true separation by a few centimetres.
- **The tracker term** is measured in the initial body frame. The map frame differs from it by a fixed transform, which cancels in relative geometry.
- **The map term** is inferred from the logged remembered clearance, because remembered cells are not logged per decision. It combines walls misplaced at observation with walls never observed.

## C3 and C4 on the budget (preliminary run)

**Source.** Per executed moving decision, from the preliminary-run logs, using the forecast the planner applied (`applied_prediction_after_yaw_ablation`).
- **Near walls** = the checked path's true clearance below 60 cm, within 15 cm of the disc, where the check binds.
- The preliminary run predates the coverage-rule fix. That does not change how forecasts are scored.
- Script: `analyse_go2_forecast_sensitivity_error_budget_development.py --by-controller`.

| Run | Controller | Decisions | e_f overall p50 · p95 · p99 (cm) | e_f near walls p50 · p95 · p99 (cm) | e_p near walls p95 (cm) | Budget near walls at p95 · p99 (cm) |
|---|---|---:|---|---|---:|---|
| recovery off, mazes 30–49 | C1 | 7575 | 1.0 · 2.3 · 3.0 | 0.9 · 2.1 · 2.6 | 0.7 | −0.3 · −4.8 |
| recovery off, mazes 30–49 | C3 | 7355 | 2.0 · 5.2 · 7.0 | 2.1 · 5.3 · 7.1 | 0.7 | −3.7 · −9.3 |
| recovery off, mazes 30–49 | C4 | 7676 | 1.1 · 2.6 · 3.4 | 1.1 · 2.6 · 3.5 | 0.5 | −0.8 · −4.6 |
| recovery on, mazes 30–89 | C1 | 22392 | 1.0 · 2.2 · 3.0 | 0.9 · 2.1 · 2.8 | 0.8 | −0.3 · −5.0 |
| recovery on, mazes 30–89 | C3 | 21120 | 2.1 · 5.3 · 7.1 | 2.1 · 5.3 · 7.1 | 0.7 | −3.6 · −9.4 |
| recovery on, mazes 30–89 | C4 | 21846 | 1.1 · 2.6 · 3.4 | 1.1 · 2.6 · 3.4 | 0.8 | −0.9 · −7.0 |
| recovery on (10 mazes) | C0 oracle | 2768 | 0.2 · 0.5 · 0.6 | 0.2 · 0.5 · 0.6 | 0.7 | +1.2 · −4.8 |

**Placement against the C1 cohorts.**
- C4 sits with clean C1.
- C3's forecast error is about twice C1's. It falls between the 10 mm noise cohort (p95 3.2, p99 3.9 cm) and the 20 mm cohort (4.9 / 6.2 cm), at a level that left C1's success unchanged.
- C0's 0.5 cm is the measurement floor: timing and sampling of the true path.

**Map term near walls.** The 99th percentile is 3–6 cm in every run: walls never observed. The p99 budget is negative for every controller, including the oracle.

## Unknown cells counted as free

**Method.**
- The clearance checks measure distance to remembered occupied cells only, so a never-observed cell counts as free.
- Neither the map nor the depth images were kept. Observation coverage is reconstructed by ray-casting both depth cameras along the true trajectory against the true scene, at 10 Hz.
- The reconstruction is validated against the planner's logged remembered clearance: it is 1.2–1.8 cm larger at the median (p95 absolute 2.3–2.7 cm), as expected from the 1 cm cell squares.
- Reach = 5 cm cells within the checked disc (0.45 m of the selected move's forecast centre path) and within 0.5 m of the robot.
- Script: `scripts/analyse_go2_unknown_cells_development.py`.

| Run | Controller | Moving decisions | Unknown in reach: all · turns · forward/arcs · after the first 30 s | Unknown cells by direction: front · side · rear | Hidden wall in reach | Passed only because unknown is free |
|---|---|---:|---|---|---:|---:|
| preliminary, recovery off | C1 | 7575 | 15.0% · 37.4% · 5.3% · 0.2% | 0.24 · 0.49 · 0.27 | 0.0% | 0.0% |
| preliminary, recovery off | C3 | 7355 | 15.5% · 33.3% · 5.9% · 0.2% | 0.23 · 0.49 · 0.27 | 0.0% | 0.0% |
| preliminary, recovery off | C4 | 7676 | 14.6% · 30.2% · 5.8% · 0.0% | 0.23 · 0.49 · 0.27 | 0.0% | 0.0% |
| clean | C1 | 7314 | 15.3% · 36.6% · 5.6% · 0.0% | 0.23 · 0.49 · 0.27 | 0.0% | 0.0% |
| noise 40 mm | C1 | 10423 | 20.2% · 30.7% · 6.0% · 9.3% | 0.23 · 0.50 · 0.27 | 0.0% | 0.0% |
| noise 160 mm | C1 | 8039 | 84.0% · 86.3% · 52.6% · 81.6% | 0.23 · 0.51 · 0.27 | 0.0% | 4.7% |
| scale 0.25× (partial) | C1 | 2822 | 15.4% · 41.1% · 5.8% · 0.4% | 0.23 · 0.49 · 0.27 | 0.0% | 0.0% |
| forward × 0.25 | C1 | 10912 | 14.9% · 20.2% · 9.8% · 4.5% | 0.23 · 0.49 · 0.28 | 0.0% | 0.0% |

The remaining cohorts are in the script output. Forward × 0.5 and × 0.75, 10 mm and 20 mm noise and scale 0.5× all fall between the clean and forward × 0.25 rows.

**What this shows.**
- **About 15% of moving decisions plan through never-observed cells within 0.5 m**, a third of in-place turns, for C1, C3 and C4 alike.
- **It is almost entirely a start-of-mission effect:** at most 0.2% after the first 30 s. The cells are the floor under and around the start pose, beside and behind the robot, which no camera can see from there.
- **No hidden wall within reach in any cohort:** walls near the robot are always observed before the robot gets close.
- **The one real hazard is at 160 mm noise:** 4.7% of decisions passed only because unknown counts as free, because the noisy path pointed into unseen space. This matches the logged-map analysis exactly: 375 of 8039 decisions.
- Persistent unknown cells in reach after 30 s track freezing: 9% at 40 mm noise and 82% at 160 mm, where the robot never turned to look.

