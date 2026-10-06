# Dynamics stage 1: open-loop friction characterisation and the chosen μ, 3 October 2026

**PRELIMINARY.** Development mode. The rule was fixed in the [dynamics plan](go2_navigation_dynamics_perturbation_plan_2026-10-02.md) ("Stage 1 μ selection, made concrete") before any characterisation run.

**Script:** `scripts/characterise_go2_friction_open_loop_development.py` (commit 517e4037).

**Outputs:** `dynamics_stage1_characterisation_v1/` under the capability root (`summary.json`, and one `result.json` per run).

## Answer

**μ = 0.2.**
- **It is the lowest level that is stable with one grid step of margin.** The next lower level, 0.15, is also stable.
- **C1's commands-only forecast error there is clearly above nominal:**
  - the 700-ms median is 29.8 mm against 5.9 mm nominal (5.1×);
  - that is above the nominal 95th percentile of 16.6 mm.
- **No level fell or stumbled.** From 1.0 down to 0.15 there were:
  - no falls and no non-foot contacts;
  - a lowest base height of 29.6 cm (nominal 30.7 cm);
  - maximum roll 7.1° and pitch 3.9° (nominal 6.3° and 2.1°).

**What low friction does to motion.** It mostly removes forward progress, not turning.
- At μ = 0.2 the robot covers 62% of the commanded translation, while the realised turn rate is 1.01× the commanded rate.
- So C1, which predicts nominal motion from commands, over-predicts cruise by 1.67×: cruise error is 88 mm and arc error 36 mm.

**Slip sets in between 0.4 and 0.3.** Above 0.4 the speed ratio stays at 1.0–1.08. At 0.3 it drops to 0.78.

**How these levels compare with the old-harness forecast-sensitivity curve** (a guide only, since that curve's cliff was mostly reserve-trap stalls):
- 1.67× over-prediction is beyond the largest bias tested there (1.5×, which was tolerated);
- a 30-mm median error lies inside that curve's 20–40 mm cliff for noise.

## Method

- **Grid and repeats:** μ = 1.0 (nominal), 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2 and 0.15. Three repeats each, with spawn yaw 0, +0.3 and −0.3 rad and distinct seeds.
- **Tape:** a fixed tape of the harness's six primitives, 66 commits of 400 ms:
  - rest starts, cruise, steady arcs, in-place turns and command switches;
  - each primitive dispatched as the harness dispatches it: its first command held for the commit, with decisions every 400 ms and a 300-ms committed prefix.
- **Arena:** a 12-m square room, because the scene builder needs a bounded wall roster.
- **Friction** is set on the floor and all 27 robot geometries (`lewm/dev_dynamics_friction_development.py`). Genesis takes the maximum of a contact pair's two coefficients, so both surfaces must change.
- **Gains:** the locomotion checkpoint's PD gains, set as the owner sets them.
- **C1's forecast** comes from the deployed command-history model, fed exactly its runtime inputs: the last four policy packets' applied-command records and the committed prefix.
- **Truth** is the base displacement over 700 ms in the decision's body frame. Movement types use the feature-cache rule.
- **Validation at nominal:** C1's open-loop error is 5.9 mm median, against 6 mm measured while driving on the C1 gate. Per type: hold 2.4 against 0, cruise 5.5 against 4, arc 6.2 against 7 mm.

## Results

Nominal (mu = 1.0): C1 command-only 700-ms error median 5.9 mm, p95 16.6 mm; base height min 30.7 cm; roll max 6.3 deg, pitch max 2.1 deg.

| mu | Decisions | C1 error median · p95 (mm) | Clearly above nominal | Fall | Stumble | Next lower stable | Qualifies | Speed ratio | Yaw-rate ratio | Base height min (cm) | Roll · pitch max (deg) | Non-foot contact samples |
|---:|---:|---|---|---|---|---|---|---:|---:|---:|---|---:|
| 1.00 | 195 | 5.9 · 16.6 | False | False | False | True | False | 0.95 | 1.01 | 30.7 | 6.3 · 2.1 | 0 |
| 0.80 | 195 | 6.6 · 17.2 | False | False | False | True | False | 1.00 | 1.02 | 30.9 | 6.1 · 2.2 | 0 |
| 0.60 | 195 | 9.7 · 20.1 | False | False | False | True | False | 1.05 | 1.03 | 30.9 | 5.9 · 2.3 | 0 |
| 0.50 | 195 | 12.1 · 26.3 | False | False | False | True | False | 1.07 | 1.05 | 30.9 | 5.9 · 2.2 | 0 |
| 0.40 | 195 | 13.4 · 32.4 | False | False | False | True | False | 1.08 | 1.06 | 30.8 | 5.9 · 2.2 | 0 |
| 0.30 | 195 | 18.6 · 52.5 | True | False | False | True | True | 0.78 | 1.06 | 30.3 | 6.2 · 2.8 | 0 |
| 0.25 | 195 | 22.1 · 87.2 | True | False | False | True | True | 0.71 | 1.05 | 30.1 | 6.4 · 3.1 | 0 |
| 0.20 | 195 | 29.8 · 100.9 | True | False | False | True | True | 0.62 | 1.01 | 30.1 | 7.0 · 3.4 | 0 |
| 0.15 | 195 | 41.2 · 107.5 | True | False | False | False | False | 0.54 | 0.96 | 29.6 | 7.1 · 3.9 | 0 |

**C1 command-only 700-ms error by movement type (median mm · median predicted/true).**

| mu | hold | rest_start | turn | cruise | arc_steady | switch |
|---:|---|---|---|---|---|---|
| 1.00 | 2.4 · 0.88 (n 18) | 4.3 · 1.08 (n 6) | 4.9 · 0.68 (n 42) | 5.5 · 1.02 (n 24) | 6.2 · 0.99 (n 18) | 7.8 · 0.97 (n 87) |
| 0.80 | 3.5 · 0.73 (n 18) | 2.8 · 1.12 (n 6) | 5.8 · 0.63 (n 42) | 6.4 · 0.96 (n 24) | 7.3 · 0.97 (n 18) | 9.1 · 0.95 (n 87) |
| 0.60 | 4.0 · 0.93 (n 18) | 1.4 · 1.06 (n 6) | 8.4 · 0.51 (n 42) | 13.5 · 0.91 (n 24) | 8.7 · 0.95 (n 18) | 11.9 · 0.91 (n 87) |
| 0.50 | 3.8 · 0.66 (n 18) | 0.7 · 1.34 (n 6) | 9.1 · 0.50 (n 42) | 18.8 · 0.90 (n 24) | 11.1 · 0.93 (n 18) | 14.0 · 0.89 (n 87) |
| 0.40 | 4.1 · 0.76 (n 18) | 13.0 · 1.52 (n 6) | 9.8 · 0.48 (n 42) | 22.3 · 0.89 (n 24) | 14.9 · 0.91 (n 18) | 16.8 · 0.87 (n 87) |
| 0.30 | 2.8 · 0.80 (n 18) | 16.2 · 1.65 (n 6) | 10.8 · 0.42 (n 42) | 44.2 · 1.28 (n 24) | 18.1 · 0.96 (n 18) | 23.8 · 1.20 (n 87) |
| 0.25 | 6.6 · 0.69 (n 18) | 19.4 · 2.39 (n 6) | 12.7 · 0.42 (n 42) | 77.8 · 1.70 (n 24) | 20.8 · 1.10 (n 18) | 30.5 · 1.34 (n 87) |
| 0.20 | 6.5 · 0.70 (n 18) | 19.7 · 3.01 (n 6) | 15.7 · 0.34 (n 42) | 87.8 · 1.67 (n 24) | 36.3 · 1.33 (n 18) | 39.3 · 1.47 (n 87) |
| 0.15 | 6.3 · 0.66 (n 18) | 19.9 · 3.96 (n 6) | 20.0 · 0.32 (n 42) | 94.5 · 1.96 (n 24) | 52.7 · 1.62 (n 18) | 50.4 · 1.63 (n 87) |

**Chosen μ: 0.2.** lowest mu stable with one grid step of margin (mu 0.15 also stable) and C1's command-only 700-ms error clearly above nominal: median 29.8 mm vs nominal 5.9 mm (x5.1) and above nominal p95 16.6 mm

## Notes

- **Rest-start windows are few** (n = 6 per level), so that row is noisy.
- **"Turn" ratios measure centre displacement during in-place turns,** which is a few millimetres. The yaw-rate ratio is the turning measure.
- **μ = 0.3 also qualifies** (3.2× nominal error) and would give an intermediate dose. The rule picks one level, so stage 1 runs at 0.2 unless Andrew wants both.
