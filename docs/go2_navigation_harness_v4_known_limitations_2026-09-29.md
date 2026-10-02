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

### Long in-place turning breaks the tracker (forecast sensitivity, 2 October)

**What happened.** The C1 forecast-sensitivity cohorts had five pose losses, all `VISUAL_TERMINAL_FAILURE` from the tracking stage. PRELIMINARY, prelim_test_v1 mazes 30–49, recovery off.
- They were: noise 40 mm on mazes 38, 44 and 45; noise 160 mm on maze 41; and forward × 0.25 on maze 34.
- The losses came 123–298 s into each mission.
- There were none in the clean, 10 mm, forward × 0.5 or forward × 0.75 cohorts, and none in the preliminary run's C1 missions.

**Every one followed sustained turning in place.**
- In the last 10 s before the loss, the robot applied turn-only commands 47–100% of the time and no translation at all.
- The planner was alternating left and right turns.

**The tracker never sees the forecast.** It uses camera, depth and gyro only. A degraded forecast causes the loss only indirectly, by producing long in-place turning.

**Limitation.** Long in-place turning breaks the visual tracker, whatever produces it: scan mode, a terminal heading limit cycle (trap 2), or a latched clearance turn (trap 3). Any controller that turns in place for long risks ending its mission this way.

**How it is reported.** The sensitivity tables count pose loss as its own category, a shared-system failure, not a forecast failure, as E1 does. The diagnosis script is `scripts/diagnose_go2_forecast_sensitivity_failures_development.py`.

## What each trap means for E1 comparisons

- **Trap 1 mostly hits controllers that hold or predict little translation.** C2 has no motion predictor, and C3 predicts collapsed translation. So part of C3's E1 deficit against C1 and C4 will be this trap and not the representation alone. E1 reports it per controller.
- **Trap 2 is a goal-settling artefact.** It turns a completed approach into a timeout. It hits C1 and C2 and not the learned predictors, so it slightly favours C3 and C4.
- **Trap 3 is rare,** with one failure each for C1, C3 and C4, and it is controller-agnostic.
