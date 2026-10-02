# Current research brief

**Start here:** [the 28 September handoff](go2_navigation_capability_handoff_2026-09-28.md). It sets the goal, current state, rules in force and immediate task, and it overrides older instructions where they conflict. Where it disagrees with the records on a fact, the records win.

The authoritative reference is [LeWMQuad-v3 agent brief: navigation testbed and controller capability, 25 September 2026](go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md), as amended by the handoff.

It supersedes the decision-headroom programme. V4.2 is closed; its recommended study-design revision is not pursued. Follow the new brief's authorised scope, gates, budgets and stop conditions. Earlier protocols and results remain historical records.

## Active goal — explicitly started 25 September 2026

**Programme goal:** determine when JEPA representations carry decision-relevant information for navigation beyond kinematic prediction, reactive control and explicit memory. Start with a validated testbed, then the E1 benchmark, then moving obstacles. E1 execution and moving obstacles require separate approval.

**Current goal:** establish a validated Go2 simulation navigation testbed. For each controller type (C0 oracle, C1 command history, C2 reactive, C3 JEPA, C4 supervised predictor), determine whether it completes beacon retrieval and return in unseen mazes, report against the pre-registered capability criterion, and produce one example video per controller (C0 optional under the brief).

**Current (2 October): after the preliminary results, per Andrew.**
- **Settings.**
  - **Recovery off is the default** of the dev entries. The switch stays, and recovery on is a secondary condition.
  - **Coverage-rule fix** (`coverage`) is on in both settings: observed obstacle cells count as observed in the frozen footprint rule.
- **Preliminary report published as a private page:** https://claude.ai/artifact/BAKFyFaXryXYoLjHnxmNbX (share it from its Share menu).
- **Budget rescore (free):** [budget rescore](go2_navigation_preliminary_budget_rescore_2026-10-02.md).
  - Tighter budgets separate only C2 with recovery on (the fastest) from C1: +0.10 at 240 s.
  - C1, C3 and C4 stay within noise at every budget.
- **Running: forecast-sensitivity experiment.**
  - C1 on prelim mazes 30–49, recovery off, coverage fix, with its forecasts degraded (`--degrade`): noise of 10–160 mm (heading 1–16°) at 700 ms, and scale 0.25–1.5×. Plus an undegraded baseline.
  - Dose-response: `scripts/report_go2_forecast_sensitivity_development.py`, with C3 and C4 marked by their measured closed-loop error.
- **Dynamics-perturbation plan drafted, not run:** [plan](go2_navigation_dynamics_perturbation_plan_2026-10-02.md). Perturbation sizes are finalised after the sensitivity results.
- **E2 (moving obstacles)** waits until the dynamics work is done.

**Mode (30 September, 16:45): DEVELOPMENT, per Andrew.** The aim is a fixed, working system with preliminary results. Rigour comes later, once the system is frozen: pre-declaration, multiple seeds and the E1 protocol.
- **Dropped:** acceptance gates, stop-for-approval points, version caps, one-change-per-version, bitwise replay checks and formal budget stops.
- **Kept:**
  - the sealed test set stays untouched;
  - ask before deleting data or before starting more than about 24 h of compute;
  - the storage reserve;
  - one git commit per meaningful change, saying what changed and why;
  - contacts and wall clearance reported on every run.
- **Goals, in order:**
  1. Fix C3's decoder over-prediction on turns and slow movement while keeping its cruising accuracy, judged by closed-loop prediction accuracy by movement type and then by driving; refit C4 on the same data each time.
  2. Fix the shared-system traps (no-movement deadlock, latched recovery turn, goal oscillation, pose loss), re-checking C1 after each change.
  3. Run C0 (10 episodes) and C1–C4 on validation plus the 10 unused round mazes (C3-v3 round layouts 22–31), then make the videos.
- **Round mazes 22–31:** the stopped exploratory safety check partly ran there. It is kept, and not used for development.
- **Order of work (Andrew, 30 September):**
  1. decoder fix;
  2. harness trap fixes;
  3. **freeze the system**;
  4. one full run of all 60 sealed test mazes on every controller (C0 on a subset) as **preliminary results**, plus validation and videos.
  The sealed mazes are not run before the freeze. Afterwards that set is relabelled **"preliminary test"**.
- **The rigorous phase will use a freshly generated sealed set.**
- **Preliminary-test set (Andrew, 1 October):** Andrew declassified the 60 capability test mazes (layouts 30–89). They are renamed `sets/prelim_test_v1`, with an `AGENTS.md` exception (commit 657ca4c8), and are run through the dev entry's `prelim_test` set (hash-verified against the capability registry). Every result from them is labelled **preliminary**.
- **Rigorous-phase sealed set generated (1 October):** `sets/sealed_test_v2`, 60 mazes × 2 episodes. It uses the same generator and exclusions, extended to all 257 earlier graphs, and random seeds that were never displayed. Structural checks only; all passed. Sealed and untouched until the rigorous phase. See [the registration note](go2_navigation_sealed_test_v2_registration_2026-10-01.md) (public receipt sha256 `45460286…`).
  - The seeds are recorded inside the sealed folder, in its sealed registry.
  - One hash-verified backup is on the workspace NVMe, off RecoveryStorage: `/mnt/workspace_drive/LeWMQuad-v3_sealed_backups/sealed_test_v2`.
- **Preliminary run, as approved by Andrew (1 October):**
  - **Recovery on:** all 60 preliminary mazes (IDs 30–89, episode 0), C1–C4, with the drive-test-chosen C3 decoder and its matched C4.
  - **Recovery off:** the first 20 of those (IDs 30–49), C1–C4.
  - **C0:** 10 mazes (IDs 30–39), recovery on. C0's results are labelled as run on a copy of the owner's harness in which only the C0 maze-ID check is relaxed; the frozen owner limits C0 to IDs below 20.
  - **No validation or round mazes;** validation was used for tuning, so its numbers would be optimistic.
  - **Sequence:** first a 3-maze trial on all controllers, then the full run. A short progress report at about 25%, and keep going unless something is broken.
  - **Beforehand:** the 1-versus-2 concurrency identity check, and deletion of the feature cache once the decoder is chosen (authorised).
  - **Results:** per controller, success, SPL, times, recovery counts per mission, stall rates, contacts and clearance, plus recovery on versus off on the 20 mazes. All labelled preliminary.
  - **COMPLETE (2 October, 07:11):** 332 missions, no errors.
    - Report: [preliminary results](go2_navigation_preliminary_results_2026-10-02.md), with [full tables](go2_navigation_preliminary_results_tables_2026-10-02.md).
    - Options note: [making the benchmark discriminate](go2_navigation_benchmark_discrimination_options_2026-10-02.md). Nothing started.
    - Headline: C0, C1, C3 and C4 reach 95–100% success. C3 and C4 are indistinguishable even though C3's closed-loop forecasts are about 1.7–2× less accurate. C2 without recovery is the only separation (11/20). Recovery caused both of C1's and C4's recovery-on failures.
  - **Trial (1 October, 03:25–04:49):** mazes 30–32, recovery on, C0–C4, with the chosen decoder and its matched C4.
    - 15/15 round trips, 0 contacts, 0 hard violations, minimum clearance 7.4 cm.
    - C0 on maze 32: the frozen executed-prefix checker raised at the end. It passes under the committed erratum: 1 of 831 decisions had no matching branch, and all 4,980 comparable rows show 0.0 error.
  - **Full run launched (04:50):**
    - `prelim_on`: C1–C4 on mazes 33–89, plus C0 on 33–39; 235 missions.
    - `prelim_off`: C1–C4 on mazes 30–49, recovery off; 80 missions.
    - The trial's recovery-on missions on 30–32 are reused rather than re-run: same code, and runs are verified deterministic.
    - Results: `scripts/report_go2_prelim_results_development.py prelim_trial prelim_on prelim_off`.
  - **Recovery-on failures, checked without recovery (Andrew, 1 October; supplement cohort `prelim_off_extra55`: C1 and C4 on maze 55, recovery off).** Both recovery-on failures by C1, C3 or C4 disappear with recovery off, and in both the first recovery intervention caused them:
    - **C4, maze 45.** Runs are identical until frame 684. There the latch timeout (10 decisions with less than 0.05 rad of progress) released a latched left turn that completes if left alone (the recovery-off run reaches the beacon at frame 2,080). A release/re-latch cycle followed (39 timeouts, 221 cool-downs), and the budget ran out 1.15 m from the beacon.
    - **C1, maze 55.** Runs are identical until frame 912. There the stall watchdog (30 s within 15 cm) called a back-up on the exact decision at which C1, left alone, set off (left arc) and reached the beacon 34 s later. After that: 25 latch timeouts, 10 escapes, 6 frontier exclusions (2 undone), another back-up, and no progress.
    - **C4, maze 55:** no intervention fired, and the recovery-on and recovery-off runs are bit-identical.
    - **Implication:** for C1, C3 and C4 the recovery thresholds are too eager; they intervene on situations the controllers resolve themselves. Recovery is essential for C2 (all 9 of its recovery-off stalls).
  - **Videos (Andrew, 1 October), one per controller.** Each is the lowest-ID recovery-off success, else recovery on and labelled, plus one labelled C2 recovery-off stall.
    - Rendered by `scripts/render_go2_prelim_video_development.py`: the unchanged V4 renderer, replaying with the run's own fixes and PRELIMINARY labels.
    - Every replay verified identical to its log.
    - In `<capability root>/videos/`:
      - `prelim_C1_prelim30_recovery-off`, `prelim_C2_prelim31_recovery-off`, `prelim_C3_prelim30_recovery-off`, `prelim_C4_prelim30_recovery-off`;
      - `prelim_C0_prelim30_recovery-on` (C0*, recovery on: C0 has no recovery-off runs);
      - `prelim_C2_prelim30_recovery-off_stall` (labelled failure: return-leg stall).
- **Decoder inputs per controller (current system):**
  - C1: command history plus the candidate tape (kinematic forecast; no images).
  - C2: reactive, no motion predictor.
  - C3 (JEPA): the frozen V-JEPA 2.1 encoder and frozen action-conditioned predictor, which sees three causal frames (t−1 s, t−0.5 s, t), the 1.5-s command history and the candidate tape. The motion decoder sees **only** the current-frame and predicted-future pooled features.
  - C4 (supervised): the pooled features of the three causal frames, the command history and the candidate tape, all given directly to the predictor.
  - The C3-vs-C4 input asymmetry is being addressed with decoder input variants (past-frame summaries, or command history), and this entry will be updated with the chosen inputs.
- **Decoder selection rule (Andrew, 30 September evening; fixed before any phase-1 result existed):** choose on one held-out group, report on others, so the choice does not inflate the reported result.
  - **Choose on** `eval_onpolicy`: held-out closed-loop C1 decisions from the C3-v3 round's held-out layouts 16–21.
  - **Score:** the C3 decoder's median 800-ms XY error, averaged over the six movement types (hold, rest start, in-place turn, steady cruise, steady arc, command switch). Lower is better. Scores within 1 mm are split by the mean |log(median predicted/true)| over the moving types.
  - **Seeds (added 21:55, before any fit's score was looked at):** each fit takes about a minute, so every candidate is fitted with 3 seeds and scored by its mean over them.
  - **Applies to** both the phase-1 training-mix choice and the variant choice. Between the input variants, (a) past frames is preferred if its score is within **3 mm** of (b) command history.
  - **Report on**, never used for choosing:
    - `eval_transfer` (700 ms);
    - `eval_offline` (the offline rest-start and turn recordings);
    - `eval_fresh_c3` (C3-v2's own fresh-check decisions).
  - The matched C4 refit is reported alongside and plays no part in the choice.
- **Recovery reporting (Andrew, 30 September evening):** every table reports, per mission and per controller, deadlock escapes, stall reroutes, latch timeouts and terminal spin breaks next to success, contacts and clearance (`scripts/summarise_go2_dev_cohorts_development.py`). Recovery must not hide weak prediction.
- **Decoder progress (30 September, late), all under the committed rule, seed means:**
  - Choice-set score: mean over movement types of median 800-ms XY error on held-out closed-loop C1 decisions.
  - **Phase 1 (training mix) is a null:** all mixes score 13.4–16.5 mm, with a seed spread of ±2–3 mm; the C3-v3 checkpoint scores 12.7 mm. Rebalancing turns C3-v3's over-prediction on turns and rest starts into under-prediction when cruising.
  - **Phase 2 (inputs, small decoder):** past frames 14.4 mm, command history 11.7 mm. Neither closes most of the gap to C4 (about 5 mm).
  - **Phase 3 (larger decoder, 12.8M parameters, starting equal to the base):**
    - scores: history 7.9 mm, base 8.3 mm, past frames 8.5 mm;
    - rule pick: **large, past frames**. It is inside the 1-mm tie band, has the smallest ratio error, and is the preferred visual-only variant under the 3-mm rule;
    - report sets: transfer 0.97 · 5 mm, and on C3's own fresh-check decisions cruise 0.96, switch 0.93, turn 0.89, rest start 1.09.
  - **Chosen by the drive test (1 October, 02:56): the large past-frames decoder** (`dev_decoder_fits/p3_large_past_frames_s2026093011.pt`), with its matched C4 from the same file.
    - Dev mazes 0–4, recovery on, large decoder against C3-v3:
      - 5/5 each;
      - SPL 0.87 vs 0.85;
      - median time 159 s vs 165 s;
      - 2,020 vs 2,636 decisions;
      - hold rates 0.045/0.079 vs 0.093/0.164;
      - deadlock escapes 0.2 vs 1.8 per mission;
      - minimum clearance 11.6 vs 3.4 cm.
    - Closed-loop forecast error: 9 vs 11 mm overall (cruise 11 vs 16, arc 9 vs 17, switch 14 vs 18 mm). The large decoder under-predicts in-place turns (ratio 0.75, error 5 mm).
    - The feature cache's two arrays (46.7 GiB) were then deleted, as authorised; see the storage log.
  - **Drive test before choosing (Andrew):** the median-seed large past-frames decoder (`dev_decoder_fits/p3_large_past_frames_s2026093011.pt`) against C3-v3, on dev mazes 0–4, recovery on. At most 2 C3 missions run at once, because measured compute sets the simulated clock. Scored by driving and by `scripts/score_go2_dev_closed_loop_prediction_development.py`.
- **Model sizes (to be matched properly in the rigorous phase):**
  - shared frozen V-JEPA 2.1 ViT-L encoder, about 304M;
  - C3's frozen action-conditioned predictor, 17,204,608;
  - C3 motion decoder: v1–v3 852,515; small past-frames variant 918,051; **large past-frames variant 12,883,267**;
  - C4 DirectMotionPredictor, 17,397,283.
- **Machine load and simulated time (1 October):** C3 and C4 decisions do not depend on wall time, by construction:
  - the owner uses `UntimedSimulationClock`, which returns simulated time and charges no compute to it (`service_cost_charged_to_simulation=False`);
  - after every 100-ms camera frame the owner drains all controller queues (tracking, registration, obstacles, mapping, planning) before physics continues;
  - planning waits for that frame's map (`queues['mapping'].join()`, "avoiding host scheduling-dependent map choice");
  - dispatch reads that frame's obstacles, because requests run between physics ticks after the drain.

  So frame freshness, observation age, the obstacle veto, the fixed 300-ms dispatch delay and command expiry are all simulated time, the same for every controller. Wall time enters only records, profiles, crash timeouts (a 120-s drain timeout; the pipeline drain deadline), and the owner Budget's 160-hour programme window, which stops missions at 03:00 on 2 October.
  - **My earlier statement that compute time drives the simulated clock was wrong.** That is the parent `MeasuredLatencyClock`, which V4 does not use.
  - Development runs now use `DevBudget`: the owner's filesystem and VRAM reserve checks without the window (formal budget stops were dropped). Development reads go through `scripts/read_go2_dev_mission_development.py` for the same reason.
  - **Verified (1 October, 03:25):** C3 with the chosen decoder on dev maze 0 was run alone on the GPU, and compared with the same mission from the drive test, which ran alongside another C3 run. The two are **identical**: 473 planning records (forecasts included), 9466 20-ms requests, the physics trace and the outcome (`scripts/compare_go2_dev_runs_identity_development.py`). Two C1 runs under different loads were identical too (fresh-check 09).
- **Trap fixes frozen (Andrew, 1 October)** at the current six: terminal (burst after a terminal spin; ignores scan mode), latch, deadlock (with the C2 reactive escape), stall (never retires the last frontier), back-up, and pose (record only). Frozen at commit 2e33bbfe (`lewm/dev_harness_fixes_development.py`).
  - No further tuning on C1 counts: 27/30 with missions flipping is noise at this sample size.
  - **Known harness issue: coverage-rule holds** (found 1 October from the preliminary videos; frozen V4 rule in `lewm/coverage_translation_view_development.py`, not a development fix).
    - The rule rejects a translation whose 0.48-m swept footprint adds any coarse cell that is not floor. That includes cells already observed as **occupied**: obstacle edges that the fine 1-cm clearance gate still passes.
    - Its remedy, a camera view request, only targets *unobserved* cells, so no view is requested. The substitute is the best of hold or a turn by utility, so the robot holds until the utilities drift or a truly unobserved cell enters the footprint.
    - Examples:
      - C3 maze 30 (recovery off) held 24.8 s with forward wanted toward a waypoint 0.36 m ahead; a left turn finally outscored hold.
      - C4 maze 30 held 13.2 s at the same spot; a truly unobserved cell triggered a view request.
    - Preliminary scale, mid-run: 1–3% of mission time for C0/C1/C3/C4, none for C2 (its reactive selector does not use the rule). 11 of 130 missions had a hold of at least 10 s; the longest was 32 s. Recovery rarely breaks it, because the stall watchdog needs 30 s without 15 cm of motion.
    - Not changed during the preliminary run (system frozen). Candidate fix for the rigorous phase: treat occupied cells as observed in the rule and leave them to the clearance gate.
  - **Known failures:**
    - C1 validation 10: pose loss on the outbound leg in the six-fix re-check;
    - C1 validation 13 and fresh-check 09: stuck oscillating at a corridor entrance or during a turnaround.
- **Recovery on/off (Andrew, 30 September evening):** the preliminary run drives every controller twice: recovery on (the full development system) and recovery off (the controller's own choices on the frozen V4 harness). Otherwise recovery can flatten the differences between controllers; for example C2 reached 30/30 with recovery in 17 missions.
  - Switch: `--recovery on|off` on the dev mission and cohort entries (`fixes_for` in `lewm/dev_harness_fixes_development.py`).
  - **Recovery** means all behavioural fixes: terminal, latch, deadlock, stall and back-up. Each overrides a controller choice. `pose` only records the tracker failure chain and stays on in both.
  - `scripts/test_go2_dev_recovery_switch_development.py` checks that recovery-on composes exactly the runtime of the full explicit fix list (same classes, same order).
- **Pose corrections (Andrew, 30 September evening):** tables also report, per mission, jumps in the published pose between consecutive 100-ms frames beyond the Go2's physical limits (more than 0.05 m or 0.15 rad), and the frames in floor-transport re-anchoring mode.
  - Correction: C1 validation 13 had **no** pose corrections. Its earlier success came from about 0.45 m of creep during 125 s of in-place oscillation, which made a route feasible, not from a pose correction.
- **Reverse motion:** a scripted short back-up may be used inside the escape rule (for the narrow-corridor turnaround). Reverse is **not** added to the candidate bank, because no predictor is trained on it.

**Previous status (30 September, 10:30): C3-v3 round complete; stopped for E1 confirmation.** Read [the C3-v3 round report](go2_navigation_c3v3_round_report_2026-09-30.md) and [the E1 launch plan](go2_navigation_e1_launch_plan_2026-09-29.md) first.
- **The round.** On-policy data came from C1. C3-v3 passed the primary closed-loop criteria (moving ratio 0.92 against C3-v2's 0.33) but failed both no-regression criteria (offline held-out groups and transfer, worst on in-place turns).
- **Which pair enters E1.** Under the pre-declared rule, the safety check did not run and **C3-v2 and C4-v2 enter E1**. No further C3 intervention is allowed before E1.
- **Storage.** Cleared, with the approved deletion recorded in [the storage log](storage_manifests/storage_log.md).
- **E1 waits for Andrew's confirmation.** The sealed set is untouched.

**Earlier status (30 September): C3-v2 phase complete; stopped for three decisions before E1.** Read [the C3-v2 phase report](go2_navigation_c3v2_phase_report_2026-09-29.md) first.
- **Offline acceptance and the rule.** C3-v2 (the refit readout) passed all seven pre-declared offline criteria.
- **Fresh check.** 10 new mazes, zero contacts and zero hard violations for every controller; round trips (sanity only) C1 9/10, C3-v2 6/10, C4-v2 7/10.
- **Which pair enters E1.** Under the pre-declared rule the C3-v2 and C4-v2 pair enters.
- **Gap diagnosis.** The offline-to-closed-loop gap was diagnosed on the check mazes.
  - The pipeline is exact: all 14,397 decisions reproduced.
  - The cause is **C3's motion readout lacking coverage of closed-loop states:** it decodes about 0.2 of true forward travel even from actual future frames, while C4 decodes 0.84 on the same states. It is not the predictor.
- **Decisions pending (Andrew):**
  1. the proposed bounded on-policy round (not run);
  2. clearing at least 16 GiB of RecoveryStorage (candidates listed);
  3. E1 confirmation, after the round.
- **Records.** [E1 launch plan](go2_navigation_e1_launch_plan_2026-09-29.md), [model versions](go2_navigation_c3v2_c4v2_model_versions_2026-09-29.md), [harness limitations](go2_navigation_harness_v4_known_limitations_2026-09-29.md), [pre-declaration and Amendment 1](go2_navigation_c3v2_readout_fix_predeclaration_2026-09-29.md). The sealed set is untouched.

**Earlier status (29 September): COMPLETE for this brief, stopped for approval.** The C0 gate passed 20/20 on `v4_completed_support`. Capability qualification (validation 10/0–29/0): **C4 19/20 and C1 18/20 are capable; C3 (JEPA) 13/20 and C2 11/20 are not.** C0 scored 10/10, and there were zero contacts. Replay-verified videos, the capability report and the E1 proposal are delivered. The E1 run, the single proposed C3 intervention and any harness change all need Andrew's approval. See [the capability qualification result](go2_navigation_capability_qualification_result_2026-09-29.md), [the E1 proposal](go2_navigation_e1_proposal_2026-09-29.md), [the gate result](go2_navigation_capability_completed_support_v4_gate_result_2026-09-28.md) and [the handoff](go2_navigation_capability_handoff_2026-09-28.md).

This replaces the 7 September navigation goal and the closed decision-headroom programme; neither is to be resumed. Machine-readable active-goal state is in [go2_navigation_capability_active_goal_2026-09-25.json](go2_navigation_capability_active_goal_2026-09-25.json). The thread goal tool refuses to replace its unfinished legacy goal; it must not be falsely marked achieved to bypass that limitation. This repository record is the current task tracker.

The completed diagnostic screen and current indexing/containment work are recorded in [the grid-correction progress report](go2_navigation_capability_grid_c3_progress_2026-09-26.md). Earlier interim reports remain historical records.


**Completed additional check (20/20 passed; five normal recordings exact):** after the unchanged C1 screen, execute the [startup-output contract](go2_navigation_capability_paired_floor_output_contract_plan_2026-09-26.json) on all 20 development starts, with numerical tolerance frozen from the old primary-path calibration before V1 comparisons. Compare complete 00/03/04/05/07 recordings with C3. Any failed or unresolved start must be reported before further changes. The current version also changes normal-start initialisation, so its charge stands regardless of the conditional correctness exemption.


**Gate-sequence amendment (27 September):** after C1 first reaches 9/10 on the ten first episodes, run the same harness on all ten second development episodes and require another 9/10 before C0. If that check fails, every subsequent version screens all 20 episodes with an aggregate 18/20 requirement. Existing safety rules and caps apply; the running V2 screen is unchanged. See [the approved amendment](go2_navigation_capability_second_episode_gate_amendment_2026-09-27.md). The old gate entry point must not bypass this requirement.
