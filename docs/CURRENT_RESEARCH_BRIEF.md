# Current research brief

**Start here:** [the 28 September handoff](go2_navigation_capability_handoff_2026-09-28.md). It sets the goal, current state, rules in force and immediate task, and it overrides older instructions where they conflict. Where it disagrees with the records on a fact, the records win.

The authoritative reference is [LeWMQuad-v3 agent brief: navigation testbed and controller capability, 25 September 2026](go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md), as amended by the handoff.

It supersedes the decision-headroom programme. V4.2 is closed; its recommended study-design revision is not pursued. Follow the new brief's authorised scope, gates, budgets and stop conditions. Earlier protocols and results remain historical records.

## Active goal — explicitly started 25 September 2026

**Programme goal:** determine when JEPA representations carry decision-relevant information for navigation beyond kinematic prediction, reactive control and explicit memory. Start with a validated testbed, then the E1 benchmark, then moving obstacles. E1 execution and moving obstacles require separate approval.

**Current goal:** establish a validated Go2 simulation navigation testbed. For each controller type (C0 oracle, C1 command history, C2 reactive, C3 JEPA, C4 supervised predictor), determine whether it completes beacon retrieval and return in unseen mazes, report against the pre-registered capability criterion, and produce one example video per controller (C0 optional under the brief).

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
- **Before the preliminary sealed run:**
  - That set lives in a `sealed_*` directory, which `AGENTS.md` forbids the model-facing account to open. Andrew either declassifies and relabels it (rename plus an `AGENTS.md` exception) or sets up the custody launcher (`docs/go2_navigation_e1_sealed_custody_launcher_proposal_2026-09-30.md`).
  - The development feature cache (about 50 GB) must be deleted or shrunk first, to keep the 12-GiB reserve. Andrew is told before anything else is cleared.
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
