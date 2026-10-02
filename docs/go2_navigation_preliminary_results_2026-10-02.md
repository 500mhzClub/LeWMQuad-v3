# Go2 navigation capability: PRELIMINARY results (2 October 2026)

> **PRELIMINARY.** Development mode; one decoder/C4 seed. These are results on **prelim_test_v1**, the 60 former sealed mazes that Andrew declassified on 1 October. They are not sealed-set results: the rigorous phase uses the freshly generated, untouched `sealed_test_v2`. Full tables: [go2_navigation_preliminary_results_tables_2026-10-02.md](go2_navigation_preliminary_results_tables_2026-10-02.md).

## Headline

- **Every controller except C2 without recovery succeeds on 95–100% of missions.** The static benchmark does not separate C1 (command history, no vision), C3 (JEPA) or C4 (supervised predictor). C0 (perfect oracle forecasts, run on 10 mazes) also reaches the ceiling, so these mazes test the shared harness more than prediction quality.
- **C3 and C4 are indistinguishable.**
  - Success difference: +0.02 with recovery on (60 mazes), −0.05 with it off (20).
  - SPL difference: 0.00.
  - Time to beacon, mazes both succeed: within 1 s.
- **The one clear separation is C2 (reactive) without recovery:** 11/20 against C1's 20/20. Difference −0.45 (−0.65 to −0.25), 9 discordant mazes to 0, McNemar p = 0.004. All 9 failures are no-movement stalls.
- **Recovery helps C2 (9 stalls fixed) and caused both recovery-on failures of C1 and C4.** With recovery off, C1 and C4 complete every maze tried.
- **Safety:** 0 contacts, 0 hard violations and 0 operating violations across all 332 missions. Minimum wall clearance is smaller with recovery on (2.2–3.3 cm for C1/C2/C3) than off (5.7–9.3 cm), because the escapes and back-ups use reduced radii.

## What was run

- **Recovery on:** all 60 preliminary mazes (IDs 30–89, episode 0), C1–C4.
- **Recovery off:** the first 20 (30–49), C1–C4.
- **C0:** mazes 30–39, recovery on only.
- **Supplement:** C1 and C4 recovery off on maze 55, to check a recovery-on failure.
- **Missions:** 332 in total, none with errors. The 3-maze trial's recovery-on missions (30–32) are part of the recovery-on set: same code, and runs are verified deterministic.
- **Models:**
  - C3: the frozen V-JEPA 2.1 encoder and action-conditioned predictor, with the **large past-frames decoder** (12.9M parameters; `dev_decoder_fits/p3_large_past_frames_s2026093011.pt`), chosen by the drive test.
  - C4: the matched supervised predictor from the same fit (17.4M).
  - C1 (command-history kinematics) and C2 (reactive) are unchanged.
- **System frozen during the run**, including the six development trap fixes:
  - **Recovery on** = terminal, latch, deadlock (with the C2 reactive escape), stall and back-up.
  - **Recovery off** = the controller's own choices on the frozen V4 harness. The pose-loss record runs in both and changes no decision.
- **Determinism:** decisions do not depend on machine load, verified by an identical C3 mission run alone and alongside another run.
- **C0\*:** C0 ran on a copy of the owner's harness in which only the C0 maze-ID check is relaxed; the frozen owner limits C0 to IDs below 20.

## Per controller

| Ctrl | Recovery | Missions | Success (Wilson 95%) | SPL | Median round trip (s) | Missions using recovery | Outbound hold rate | Min clearance |
|---|---|---:|---|---:|---:|---:|---:|---:|
| C0\* | on | 10 | 10 · 1.00 (0.72–1.00) | 0.84 | 169 | 1 | 0.041 | 8.8 cm |
| C1 | on | 60 | 59 · 0.98 (0.91–1.00) | 0.81 | 157 | 25 | 0.091 | 3.0 cm |
| C2 | on | 60 | 60 · 1.00 (0.94–1.00) | 0.81 | 152 | 25 | 0.008 | 3.3 cm |
| C3 | on | 60 | 60 · 1.00 (0.94–1.00) | 0.84 | 156 | 20 | 0.057 | 2.2 cm |
| C4 | on | 60 | 59 · 0.98 (0.91–1.00) | 0.82 | 158 | 16 | 0.062 | 5.6 cm |
| C1 | off | 20 | 20 · 1.00 (0.84–1.00) | 0.85 | 157 | — | 0.044 | 7.5 cm |
| C2 | off | 20 | 11 · 0.55 (0.34–0.74) | 0.46 | 154 | — | 0.203 | 5.9 cm |
| C3 | off | 20 | 19 · 0.95 (0.76–0.99) | 0.82 | 160 | — | 0.098 | 9.3 cm |
| C4 | off | 20 | 20 · 1.00 (0.84–1.00) | 0.85 | 161 | — | 0.041 | 5.7 cm |

Recovery counts per mission are in the full tables.
- **C2 with recovery on:** 1.43 deadlock escapes and 0.77 terminal spin breaks per mission.
- **C1 with recovery on:** 0.85 latch timeouts per mission.
- **C4 with recovery on:** 1.07 latch timeouts per mission.

## Paired comparison against C1 (same mazes and recovery setting)

Differences are other minus C1. Maze-level paired bootstrap 95% intervals (B = 10,000). "Both succeed" columns are conditional on both controllers succeeding. **Survivorship caution:** that conditioning drops each controller's failed mazes, which biases a controller with many failures (C2 off) in its favour. The last column counts every maze, with non-arrival set to the 480-s budget.

| vs C1 | Recovery | Mazes | Success | Difference (95% CI) | Only C1 / only ctrl | McNemar p | SPL diff, both succeed | Time-to-beacon diff, both succeed (s) | Time-to-beacon diff, all mazes, non-arrival = 480 s |
|---|---|---:|---|---|---|---:|---|---|---|
| C0\* | on | 10 | 10 / 10 | 0.00 | 0 / 0 | – | −0.01 (−0.05 to +0.01) | +15.5 (−10.0 to +57.8) | +15.5 (−10.0 to +57.8) |
| C2 | on | 60 | 60 / 59 | +0.02 (0.00 to +0.05) | 0 / 1 [55] | 1.00 | −0.01 (−0.03 to 0.00) | −21.4 (−31.1 to −12.5) | −27.5 (−43.8 to −14.5) |
| C3 | on | 60 | 60 / 59 | +0.02 (0.00 to +0.05) | 0 / 1 [55] | 1.00 | +0.01 (0.00 to +0.02) | −7.0 (−15.3 to +1.0) | −12.9 (−28.8 to −0.7) |
| C4 | on | 60 | 59 / 59 | 0.00 (−0.05 to +0.05) | 1 [45] / 1 [55] | 1.00 | +0.01 (0.00 to +0.02) | −8.0 (−16.6 to −0.5) | −7.7 (−27.1 to +11.9) |
| C2 | off | 20 | 11 / 20 | **−0.45 (−0.65 to −0.25)** | **9 / 0** | **0.004** | −0.01 (−0.03 to 0.00) | −10.3 (−18.0 to −4.2) | **+85.6 (+20.9 to +160.5)** |
| C3 | off | 20 | 19 / 20 | −0.05 (−0.15 to 0.00) | 1 [31] / 0 | 1.00 | 0.00 (−0.01 to +0.02) | +2.0 (−13.7 to +15.9) | +19.4 (−9.1 to +59.2) |
| C4 | off | 20 | 20 / 20 | 0.00 | 0 / 0 | – | 0.00 (−0.02 to +0.02) | +1.5 (−12.5 to +16.5) | +1.5 (−12.5 to +16.5) |

**C3 versus C4 directly:**
- Recovery on, 60 mazes: 60/59, +0.02 (0.00 to +0.05); SPL +0.00; time to beacon +0.9 s (−5.5 to +7.3).
- Recovery off, 20 mazes: 19/20, −0.05 (−0.15 to 0.00); SPL +0.00; time to beacon +0.5 s (−7.7 to +8.5).

## Closed-loop prediction accuracy while driving

The forecasts each controller actually made while driving were scored against physics truth by `scripts/score_go2_dev_closed_loop_prediction_development.py`:
- the selected candidate's 700-ms XY forecast against the true displacement;
- only decisions whose executed commands match the forecast's command sequence (about 95% of them).

Each cell is the median predicted/true ratio, then the median error. Recovery on, all 60 mazes. C2 is omitted, because its selector uses no forecasts.

| Ctrl | All | Cruise | Steady arc | Command switch | In-place turn | Hold |
|---|---|---|---|---|---|---|
| C0\* (oracle) | 1.00 · 0 mm | 1.00 · 0 | 1.00 · 0 | 1.00 · 0 | 1.00 · 0 | 1.00 · 0 |
| C1 (kinematic) | 0.95 · 6 mm | 0.99 · 4 | 0.95 · 7 | 0.96 · 7 | 0.73 · 5 | 0.76 · 0 |
| C3 (JEPA, large decoder) | 0.97 · **10 mm** | 1.00 · 11 | 1.03 · 10 | 0.97 · 14 | 0.83 · 4 | 0.54 · 9 |
| C4 (supervised) | 1.01 · 6 mm | 1.02 · 6 | 1.03 · 6 | 1.01 · 7 | 0.84 · 3 | 0.87 · 3 |

**C3's forecasts are about 1.7–2× less accurate than C4's or C1's in closed loop, yet all three drive the same.** The harness tolerates centimetre-scale forecast error through its clearance reserves and replanning every 400 ms. That is further evidence these mazes do not test prediction quality.

## Recovery on versus off (mazes 30–49, plus maze 55)

| Ctrl | Mazes | On | Off | Only on succeeds | Only off succeeds |
|---|---:|---:|---:|---|---|
| C1 | 20 + 55 | 20 + **0** | 20 + **1** | — | **55** |
| C2 | 20 | 20 | 11 | 30, 34, 36, 40, 43, 44, 46, 48, 49 | — |
| C3 | 20 | 20 | 19 | 31 | — |
| C4 | 20 + 55 | 19 + 1 | 20 + 1 | — | **45** |

Runs are deterministic, so a recovery-on run and its recovery-off counterpart are identical until recovery first intervenes. Both recovery-on failures diverge at that first intervention:

- **C4, maze 45** (diverges at frame 684).
  - The latch timeout released a latched left turn after 10 decisions with less than 0.05 rad of progress. Left alone, that latch completes: the recovery-off run reaches the beacon at frame 2,080.
  - A release/re-latch cycle followed (39 timeouts, 221 cool-downs), and the budget ran out 1.15 m from the beacon.
- **C1, maze 55** (diverges at frame 912).
  - The stall watchdog (30 s within 15 cm) called a back-up on the exact decision at which C1, left alone, set off and reached the beacon 34 s later.
  - It was followed by 25 latch timeouts, 10 escapes, 6 frontier exclusions (2 undone) and a second back-up, with no progress.
- **C4, maze 55:** no intervention fired, and the two runs are bit-identical.

**Reading.**
- For C1, C3 and C4 the recovery thresholds are too eager: they intervene on situations these controllers resolve themselves.
- For C2, recovery is essential.
- Recovery also lowers minimum clearance (see the table above).

## C3's recovery-off holds

These come from `scripts/analyse_go2_prelim_c3_holds_development.py`, run per mission and pooled over all 20 recovery-off missions: 1,390 hold decisions.
- **72% (1,000) are the maze-31 deadlock, C3's only recovery-off failure.**
  - The robot sits inside the 0.48-m forecast-clearance requirement: 0.45-m disk plus a 0.03-m reserve. Hold's own predicted clearance is 0.429 m and fails the gate at 959 of 999 decisions, so no move is eligible.
  - Turns are excluded by forecast clearance (998/999).
  - Forward and arcs are mostly excluded by the view restriction during a scan (909) and otherwise by clearance (89).
  - It is a geometric harness deadlock (the same trap the deadlock escape fixes for C2), not a forecast error.
- **18% (248) are planned arrival settling** at the goal.
- **7% (101) are coverage-rule holds** (mostly maze 30; see Known issues).
- **Predicted travel.** C3's 700-ms forward forecast at those decisions is close to the command-history reference where the robot is moving: 0.048 against 0.055 m (maze 30), 0.013–0.016 against 0.018 m near the goal. From rest at maze 31 it is short: 0.023 against 0.055 m, so C3 still under-predicts starts from rest in closed loop. That did not cause the hold.

## Why C0 is slower to the beacon than C1 (+15.5 s)

From `scripts/analyse_go2_prelim_c0_speed_development.py`, outbound leg, mazes 30–39, recovery on:
- **One maze carries the mean.** The paired mean is +15.5 s but the **median is −0.6 s**. Maze 32 alone carries it: C0 took 289 s against C1's 97 s.
  - It spent 69% of decisions turning, against 41%.
  - It had 21 recovery events, 17 of them latch timeouts.
  - Its route was 2.15× the shortest path, against 1.49×.
  - It spent 79% of decisions routing to frontiers.
  - It is also the maze where C0's oracle checker raised its end-of-mission erratum (see Known issues).
- **The other nine mazes are within about ±10 s, with no consistent sign.** Both controllers spend:
  - about 20% of decisions on straight forward moves and about 75% on turns and arcs;
  - about 25% in scan or view mode (translations excluded);
  - about 50–70% routing to frontiers.

  C0 chooses forward slightly less often (median −4 points); detours are equal (median difference −0.005).
- **Conclusion: the selector and routing limit speed, not prediction accuracy.** With exact forecasts C0 is no faster. The 400-ms commit objective plus its heading-alignment term favours turns and arcs, required camera views block translations, frontier exploration dominates the outbound leg, and occasional latch/recovery loops add more.

## Known issues and caveats

- **Ceiling.** C0, C1, C3 and C4 are all at 95–100% success. With 60 mazes, the success-difference interval is about ±5 points: that bounds a hidden difference, it does not establish equivalence. No equivalence margin was declared.
- **Coverage-rule holds (frozen V4 rule; to be fixed after the preliminary results, per Andrew).**
  - The translation-footprint rule treats observed **occupied** coarse cells as unknown, and requests a camera view only for unobserved cells. The robot therefore holds until its utilities drift.
  - Scale: 1–3% of mission time for C0/C1/C3/C4 and none for C2 (its selector bypasses the rule). Mid-run, 11 of 130 missions had a hold of 10 s or more; the longest was 32 s.
- **C0 on maze 32:** the frozen executed-prefix checker raised at the end. It passes under the committed erratum: 1 of 831 decisions had no matching branch, and all 4,980 comparable rows show 0.0 error.
- **What C3 versus C4 tests.** Both share the frozen pretrained V-JEPA encoder. C3 versus C4 compares a latent world model plus decoder against direct supervised regression on the same features. It does not compare a foundation model with a task-specific model trained from scratch.
- **Single seed** for the decoder and C4. The decoder was chosen on development data; the validation mazes were used for tuning and are excluded here.

## Still weak

1. The static benchmark does not discriminate prediction quality: oracle, kinematic, JEPA and supervised all reach the ceiling.
2. The recovery thresholds hurt the predictive controllers and reduce clearance; C2 depends on recovery.
3. The coverage-rule hold bug.
4. C3 under-predicts starts from rest in closed loop (0.023 against 0.055 m at maze 31).
5. The no-movement clearance deadlock (C3 maze 31; C2's stalls) is a harness-geometry trap, fixed only by the recovery escape.

## Videos (labelled PRELIMINARY; every replay verified identical to its logged run)

All are in `<capability root>/videos/`:
- `prelim_C1_prelim30_recovery-off`, `prelim_C2_prelim31_recovery-off`, `prelim_C3_prelim30_recovery-off` and `prelim_C4_prelim30_recovery-off`: each controller's lowest-ID recovery-off success.
- `prelim_C0_prelim30_recovery-on` (C0\*, recovery on: C0 has no recovery-off runs). It has no 4× version, because the renderer only makes one for missions over 120 s and this one took 111 s.
- `prelim_C2_prelim30_recovery-off_stall`: labelled failure, a return-leg stall.

## Reproduce

```
scripts/report_go2_prelim_results_development.py prelim_trial prelim_on prelim_off
scripts/analyse_go2_prelim_c3_holds_development.py <C3 recovery-off runs> --pooled
scripts/analyse_go2_prelim_c0_speed_development.py
scripts/score_go2_dev_closed_loop_prediction_development.py prelim_trial prelim_on prelim_off
```
