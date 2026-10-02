# Harness `v4_completed_support`: known limitations (shared traps), 29 September 2026

**Decision (Andrew, 29 September 2026).** The three shared traps are not fixed now. The harness stays frozen for E1 (sha256 `82b7b604…`, commit 7da82b23). The traps will be revisited with the E2 harness work, which needs its own oracle gate. Until then, every E1 result measures each controller together with these traps, and E1 reports must say so.

## Counts by controller (capability validation, 20 episodes per controller)

The source is the preserved validation records, read with the fixed mechanism rules in `scripts/diagnose_go2_capability_validation_timeouts_development.py` (`analysis/qualification_v4_2026-09-28/timeout_diagnosis.json`). Every failure was a 480-s timeout with zero contacts. Nothing was re-run.

| Trap | C1 | C2 | C3 | C4 | Episodes |
|---|---:|---:|---:|---:|---|
| **1. No eligible movement under the view requirement.** Translations are view-restricted, turns are clearance-blocked, and the robot holds. | 0 | 6 | 4 | 0 | C2 15, 17, 18, 20, 26 (return), 28; C3 12, 13, 14, 15 |
| **2. Terminal heading limit cycle at the goal.** The robot turns in place 3–5 cm from the target, and arrival never confirms. | 1 | 3 | 0 | 0 | C1 10 (home); C2 14, 27, 29 |
| **3. Latched clearance (recovery) turn with its preferred direction blocked by forecast clearance** | 1 | 0 | 1 | 1* | C1 13; C3 17; C4 22* |
| **All trap failures** | **2** | **9** | **5** | **1** | |
| *Validation failures in total* | *2* | *9* | *7* | *1* | |

**Not harness traps.** C3's two remaining failures (27/0 and 29/0) are its own "hold outscores every movement" stalls. They come from C3's collapsed translation predictions, not from the harness.

**C0.** The oracle met traps 1 and 3 but always escaped. For example, 19/0 had 67 no-eligible holds and 16/0 had 489 view-restriction holds.

## \*Which category C4's oscillation falls in

**It is trap 3, the latched clearance turn, in a turning form rather than a holding form.**

The fixed rules label C4 22/0 "turn oscillation without progress". That rule keys on the `LATCHED_RECOVERY_TURN_BLOCKED` hold override, which never fired here, because the harness substituted the opposite turn instead of a hold. The logged decisions show it is the same machinery:

**The stuck period (30–450 s, 1,050 decisions):**
- The harness's latched clearance turn was active in 997 decisions (95%). Its preferred direction was blocked by forecast clearance: right turn blocked 792 times, left turn 205.
- The clearance-memory filter replaced C4's own choice in 642 decisions (61%).
- The early heading release was suppressed 424 times.
- The robot alternated left and right turns in place (519 and 524) with no translation for 420 s. The remaining path grew from 8.5 m to 9.6 m.

**The escape:**
- At about 450 s the latch released, and the clearance turn was active in 0 of the next 75 decisions.
- C4 translated at once, covering 4.5 m in 30 s, and the budget ran out 5.1 m from the beacon.

**The holding form, for comparison.** C1 13/0 and C3 17/0 had the latch active in 100% of their final-window decisions, with the filter replacing every choice and 0 turns. There the substitute was a hold.

**The rules stay frozen.** This attribution is a reading of the logged records; the rules and their labels are unchanged. For E1, the trap-3 count is reported both ways: the frozen-rule label, and the latch-active fraction in the final window.

**Only C4 hit this on 22/0.** C1, C2 and C3 all completed round trips on the same episode (RT 175, 193 and 315 s).

## Visual pose loss (fresh check, 29 September)

**C1 on fresh-check maze 09 failed with lost visual pose tracking.** 165.3 s into the outbound leg, 8.96 m travelled, the tracking stage raised `ValueError('measured visual pose unavailable')`. The frozen runtime turns this into a controller fault (`RuntimeError`), which ends the mission.
- There were zero disallowed contacts and zero hard violations.
- The reader's taxonomy records 1 pose loss, 8 movement-outscored holds, 2 blocked-recovery holds and 2 view-restriction holds. The failure is preserved at `runs/c3v2_check_C1_chk09_ep0_attempt001` (`failure.json`).

**This is the first pose-loss failure on `v4_completed_support`.**
- None occurred in the gate, the development screens or validation qualification.
- It is a shared-harness perception limitation, not specific to C1. The camera-based tracker loses registration, and no controller can recover a mission once the tracker faults.
- It is outside traps 1–3. E1 reports it as its own category (pose-loss controller failures) for every controller.

### Long in-place turning breaks the tracker (forecast sensitivity, 2 October; updated 19:20 with the final cohorts)

**What happened.** The C1 forecast-sensitivity cohorts had **twelve** pose losses, all `VISUAL_TERMINAL_FAILURE` from the tracking stage. PRELIMINARY, prelim_test_v1 mazes 30–49, recovery off.

| Cohort | Mazes | Loss time | Motion before the loss |
|---|---|---|---|
| noise 40 mm | 38, 44, 45 | 125–298 s | sustained in-place turning: last 10 s 82–94% turn-only, no translation |
| noise 80 mm | 38, 45 | 387–421 s | sustained: last 10 s 66–75% turn-only, no translation |
| noise 80 mm | 39 | 185 s | mixed: last 10 s 55% turn-only, 36% translating |
| noise 160 mm | 41 | 123 s | sustained: 47% of the last 10 s, 56% of the 20 s before |
| noise 160 mm | 47 | 458 s | a short burst: 20 s holding, then about 5 s of turning (51% of the last 10 s) |
| uniform scale × 1.25 | 39 | 281 s | sustained: last 10 s 93% turn-only, 4% translating |
| forward × 0.25 | 34 | 135 s | sustained: 100% turn-only for the last 30 s |
| turns × 0.5 | 32, 43 | 134–165 s | sustained: 93–100% turn-only for the last 30 s |

- There were none in the clean, 10 mm, 20 mm, other uniform-scale, forward × 0.5 / × 0.75, turns × 0.25, × 1.25, × 1.5 or × 2.0 cohorts.
- There were none in the preliminary run's C1 missions.
- **Every loss followed turning-dominated motion.** Ten followed sustained in-place turning with at most 4% translation (the planner alternating left and right turns). One (noise 160 mm, maze 47) followed a short burst of turning after a hold. One (noise 80 mm, maze 39) was mixed: 55% turning and 36% translating in its last 10 s.
- The eight losses listed at 14:30 were those in the cohorts finished by then; noise 80 mm and uniform × 1.25 added four.

**The tracker never sees the forecast.** It uses camera, depth and gyro only. A degraded forecast causes the loss only indirectly, by producing in-place turning.

**Limitation.** Long in-place turning breaks the visual tracker, whatever produces it: scan mode, a terminal heading limit cycle (trap 2), a latched clearance turn (trap 3), or turn under-prediction. Any controller that turns in place for long risks ending its mission this way.

**How it is reported.** The sensitivity tables count pose loss as its own category, a shared-system failure, not a forecast failure, as E1 does. The diagnosis script is `scripts/diagnose_go2_forecast_sensitivity_failures_development.py`.

## Rear and side clearance during turns rests on the remembered map and the forecast only (2 October)

**Both depth cameras face forward.**
- **Primary:** level, 78° × 63°.
- **Auxiliary:** the same lens pitched 45° down.
- Both read depth from 0.2 m to 5 m.

The last-moment depth stop uses only their current images, so it cannot see a wall beside or behind the robot. Everything that protects the rear and sides comes from the planner's remembered-map check, applied to the forecast.

**That check is a disc, not the articulated body.** Every planner-stage clearance filter in the frozen chain tests a 0.45 m disc centred on the base. That covers memory clearance, the turn and translation reserves, the stopping projection and the recovery modes. The disc is swept along the forecast's centre positions only, against remembered obstacle cells, and the heading is not used. Turns and translations require 0.48 m.

**The disc contains the whole body, with a thin margin at the rear.**
- The articulated body never extends more than 42.5 cm from the base centre, in any sensitivity cohort or move (sampled at 10 Hz).
- So a centre held at 0.45 m leaves the rear calves about 3 cm, or about 6 cm at 0.48 m.

**Evidence (forecast-sensitivity cohorts, PRELIMINARY).**
- In the clean C1 baseline, 89 of the 100 closest approaches while moving (5 per mission) were a rear calf, nearly all during in-place turns.
- 93% of all approaches had the nearest wall point outside both cameras' view.
- In-place turns carried 79–94% of the clearance loss in every cohort.
- The disc held against the true walls at all approaches but one.
- The clearance lost under degraded forecasts came from the base centre sitting closer to walls during turns: median 57 cm clean, 50–52 cm at 40–160 mm noise.

Script: `scripts/analyse_go2_forecast_sensitivity_close_approaches_development.py`.

**A difference between simulation and the real robot.** The physical Go2 carries a wide-view lidar (Unitree's 4D LiDAR L1, a hemispherical field of view) that would cover the rear and side zone. So the simulated stack's rear and side protection is weaker than the real robot's sensing allows. This blind zone is specific to the simulated sensor set, and results about it should not be read as limits of the real platform.

## What each trap means for E1 comparisons

- **Trap 1 mostly hits controllers that hold or predict little translation.** C2 has no motion predictor, and C3 predicts collapsed translation. So part of C3's E1 deficit against C1 and C4 will be this trap and not the representation alone. E1 reports it per controller.
- **Trap 2 is a goal-settling artefact.** It turns a completed approach into a timeout. It hits C1 and C2 and not the learned predictors, so it slightly favours C3 and C4.
- **Trap 3 is rare,** with one failure each for C1, C3 and C4, and it is controller-agnostic.

## Operating precondition for the pessimistic-unknown condition (2 October)

**The rule.** In stage 2 of the [calibrated-margin experiment](go2_navigation_calibrated_margin_experiment_plan_2026-10-02.md), never-observed cells block a move. They do so when they lie within body reach (0.425 m) plus the controller's calibrated forecast-error bound of the forecast centre path.

**Why that needs a precondition.** At the start pose, both forward depth cameras see the floor only from about 0.45 m ahead. Nothing beside or behind the robot is observed until it turns, and a turn is a move the rule must allow.

**What was tried first.** Seeding only the robot's own 0.425 m reach disc deadlocked the initial look-around. In the live smoke test (C1, maze 30), no move was clear in any of 1197 decisions and the robot turned 18° in a minute. The blocking radius (0.45–0.50 m) exceeds that seed, and the turn's own forecast drifts about 1 cm.

**Operating precondition.** The robot is placed in a cleared area of 0.5 m radius, the same for every controller. The start disc of that radius is seeded as known free, at 1 cm resolution and never beyond 0.5 m.

**Simulated episodes satisfy it.** The episode generator rejects spawns closer than 0.5 m to a wall.

**C3 at its p99 bound may not leave the start.** Its blocking radius (0.497 m) nearly fills the precondition, and its initial-panorama turn forecasts drift up to 6.6 cm at p99. In the synthetic test its start turn is blocked even with no drift. This is a consequence of C3's forecast error under the rule, and it is reported as such.

**Look-around exemption (option A, 2 October; the same for all controllers).** The 0.5 m seed alone still stalled half the smoke mazes. The body drifts 4.9–6.8 cm (median 5.8 cm) while turning on the spot for the scripted look-around. That is a gait property, the same for every controller, and it exceeds the room between the blocking radius and 0.5 m.
- So, during the scripted look-around only, never-observed cells do not block a move.
- **Justification:** under the precondition, the body cannot leave the cleared disc while its centre stays within 0.5 − 0.425 = 0.075 m of the start, 0.425 m being the body's largest reach.
- The exemption ends when the look-around completes, or as soon as the centre drifts more than 0.07 m. The rule then applies in full.
- Remembered walls are checked throughout.

**A difference between simulation and the real robot.** The physical Go2's wide-view lidar (Unitree's 4D LiDAR L1, a hemispherical field of view) would observe the start surroundings directly. On the real robot this precondition, and the look-around exemption, would be unnecessary.


## Can't translate out once inside the reserve (shared-system trap, 2 October)

**What it is.** Once the base centre is within 0.45 m of a remembered wall (inside the clearance disc), no translation can pass the planner's check, even one that moves directly away from the wall.
- **Forecast controllers (C1, C3, C4).** A translation that starts inside the 0.48 m requirement passes only through reserve recovery. That needs its starting clearance above 0.45 m, no decrease after the start, and an end above 0.48 m.
- **C2.** Its reactive rule makes *every* action ineligible, turns included, once the stored clearance is 0.45 m or less.

**How robots get inside.** Turns stay allowed because their forecast barely moves the centre (about 1 cm). The gait actually drifts the body 5–7 cm while turning on the spot, and that carries the centre into the disc.

**Evidence** (`scripts/analyse_go2_reserve_trap_development.py`; stalls of 120 s or more without translation, sampled every 5 s). PRELIMINARY:
- **C2, preliminary run, recovery off: 8 of 9 stalls.**
  - Mazes 30, 34, 40, 43, 46, 48 and 49 started inside the disc (remembered centre clearance 0.413–0.445 m). Maze 44 started outside and drifted in.
  - In most samples a 0.1 m step at the current heading would have *increased* true clearance. C2's rule forbids it.
  - The ninth stall (maze 36) is different: clearance 0.58 m, still progressing at budget end.
- **C3, preliminary run, recovery off, maze 31** (its only failure, the deadlock described in the preliminary report).
  - The stall began outside the disc (remembered 0.542 m at 70 s).
  - The robot drifted inside while turning in scan mode and stayed trapped for 73 of 79 samples. In 15 of them a translation would have increased clearance.
- **C1, stage 2 of the calibrated-margin experiment, mazes 31 and 32** (the first stage 2 stalls). Same ending: inside the disc, or inside the reserve with no translation increasing clearance, while scanning.
  - The full 20-maze count will be added when stage 2 finishes.

**Status.** This is a shared-system limitation of the frozen harness, not specific to any forecast.

**Candidate fix for the next harness version — NOT applied now.** Allow a translation whose forecast clearance never decreases from its start, and ends higher, even inside the disc or reserve: one that leaves the wall rather than approaching it. For C2, allow turns and translations that increase clearance. It needs its own oracle gate before use.
