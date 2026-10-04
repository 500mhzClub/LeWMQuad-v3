# Dynamics stage 1, uniform floor friction μ = 0.2: results, 4 October 2026 (interim; C3 10 of 20)

**PRELIMINARY.** Development mode, prelim_test_v1 mazes 30–49, episode 0, recovery off, harness `reserve_exit_v1_1`. μ = 0.2 was chosen by the pre-declared rule ([characterisation](go2_navigation_dynamics_stage1_characterisation_2026-10-03.md)); friction is set on the floor and all 27 robot geometries. Controllers C1, C3 and C4 run zero-shot, unchanged from the [clean baseline](go2_navigation_reserve_exit_rerun_results_2026-10-03.md).

**Cohorts:**
- C1 and C4: `dyn1_friction02_cpu`, with C4 mazes 48–49 in `dyn1_friction02_c4b`, rerun after the 4 October pause.
- C3: `dyn1_friction02_c3b`. It is still running; the interrupted `dyn1_friction02_c3` directories are kept as incomplete records.
- Contact stops are scored with the failure reader.

## Driving outcomes

| Controller | Missions | Round trips | Beacon reached | Ended on contact | Hard-clearance samples, missions with any | Clean baseline (μ = 1.0) |
|---|---:|---:|---:|---:|---:|---|
| C1 | 20 | 0 | 0 | 3 (mazes 30, 33, 42) | 3 | 20/20 |
| C3 | 10 of 20 so far | 0 | 0 | 1 (maze 39) | 2 | 19/20 |
| C4 | 20 | 0 | 0 | 2 (mazes 31, 32) | 3 | 20/20 |

- **No controller completes a round trip at μ = 0.2,** so outcomes cannot separate them.
- **Every contact is the rear-left calf against a wall,** after long hard-clearance stretches of turning or holding pressed against it.

## Why: the harness freezes all three

- **Most missions freeze 19–36 s in,** during the scripted opening look-around.
- **Slip carries the body inside the 0.45-m disc.** Turning in place on μ = 0.2 slides the body much further than the nominal 5–7 cm; true clearance at onset is 0.40–0.44 m.
- **Nothing is then allowed.** The turn reserve blocks both turns inside the reserve or disc unless the forecast path gains clearance, and the look-around (scan mode) excludes translation. The robot holds until the budget runs out.
- **Holds dominate:** "memory clearance" and "view restriction" holds are 700–1,100 of about 1,200 decisions per mission, and robots travel 2–6 m in total.
- **The reserve exit cannot act here,** because translation is excluded while scanning.

## Forecast error by movement type (closed loop, decisions actually made)

`scripts/score_go2_dev_closed_loop_prediction_development.py`. Each cell is the median 700-ms predicted/true translation ratio · median XY error (mm), with the decision count. The μ = 0.2 decisions come mostly from before or around each freeze.

| Controller, condition | Cruise | Steady arc | Command switch | In-place turn | Rest start | Hold |
|---|---|---|---|---|---|---|
| C1, μ = 1.0 (clean baseline) | 1.00 · 4 (483) | 0.95 · 7 (887) | 0.96 · 7 (3,757) | 0.75 · 5 (2,008) | – · 14 (1) | 0.88 · 5 (42) |
| C1, μ = 0.2 | 1.38 · 75 (41) | 1.40 · 41 (116) | 1.32 · 43 (645) | 0.32 · 19 (1,367) | 1.87 · 15 (34) | 0.02 · 1 (20,506) |
| C4, μ = 1.0 | 1.02 · 6 (519) | 1.03 · 6 (848) | 1.01 · 8 (3,829) | 0.88 · 3 (2,635) | 0.91 · 3 (9) | 1.15 · 3 (118) |
| C4, μ = 0.2 | 1.41 · 76 (32) | 1.41 · 39 (80) | 1.32 · 45 (533) | 0.31 · 19 (1,441) | 0.97 · 7 (23) | 0.36 · 3 (19,220) |
| C3, μ = 1.0 | 1.00 · 12 (586) | 1.03 · 10 (814) | 0.97 · 15 (3,513) | 0.84 · 4 (2,295) | – · 8 (8) | 0.52 · 12 (1,375) |
| C3, μ = 0.2 (10 of 20) | 1.26 · 67 (11) | 1.33 · 37 (61) | 1.11 · 35 (442) | 0.27 · 21 (717) | 1.02 · 13 (11) | 0.78 · 9 (10,093) |

(The C4 μ = 0.2 row is `dyn1_friction02_cpu` only; the two `c4b` missions agree: cruise 1.98 · 105 (3), arc 1.43 · 49 (11), switch 1.34 · 51 (65).)

**Reading.**
- **Low friction makes every forecaster over-predict translation.** C1 and C4 over-predict by about 1.3–1.4×, and errors are 5–10× their clean values (cruise 75–76 mm against 4–6).
- **C4's visual inputs give it no zero-shot advantage over C1.** Its errors and ratios match C1's within a few millimetres.
- **C3 is somewhat closer on the moving types** in this partial sample: cruise 1.26 · 67 mm, arc 1.33 · 37, switch 1.11 · 35, against C1's 1.38 · 75, 1.40 · 41 and 1.32 · 43. The cruise and arc counts are small (11 and 61), and C3 has finished only 10 of 20 missions.
- **These are the decisions taken before the freezes.** The robots barely drive, so few cruise and arc decisions exist.

## Status and next

- **C3 mazes 40–49 are running,** two GPU lanes at about 100 min per mission. Expected to finish around 03:00 on 5 October.
- **Open for Andrew:**
  - the turn-reserve fix: let an in-place turn proceed inside the reserve or disc when its forecast centre path does not lose clearance. This is the rule behind every stage-1 freeze, and C2's and C3's remaining baseline failures;
  - the stage-2 speed guard.
- **Stage 1 at μ = 0.2 on this harness can report forecast error but not driving differences.**
