# C3-v2 phase report: fresh check, E1 rule, gap diagnosis, 29 September 2026

## Answer and recommendation (30 September, 01:40)

**The offline-to-closed-loop gap is in C3's motion readout.** The readout does not cover the states the robot actually visits in closed loop. It is not a pipeline defect, and the predictor is not the bottleneck.

**The pipeline is cleared.**
- The logged inputs of all 20 C3-v2 and C4-v2 fresh-check missions were re-run through the offline evaluator: 14,397 decisions and 1,171,104 values.
- They match the logged run-time predictions to within 1.9×10⁻⁷ m and 4.0×10⁻⁷ rad, against a tolerance of 10⁻⁵.

**Split test.** This covers fresh-check mazes, executed full forward commands and the same logged decisions throughout. Each cell gives the median predicted/true 800-ms travel, then the median XY error at 800 ms.

| Model and input | From rest (n=6, true 58 mm) | Moving (n=78, true 118 mm) |
|---|---|---|
| C3-v1 readout, (a) actual future frames | 0.14 · 50 mm | 0.12 · 87 mm |
| C3-v1 readout, (b) predicted features, as deployed | 0.05 · 59 mm | 0.13 · 82 mm |
| C3-v2 readout, (a) actual future frames | 0.27 · 42 mm | 0.17 · 90 mm |
| C3-v2 readout, (b) predicted features, as deployed | 0.04 · 56 mm | 0.21 · 70 mm |
| C4-v1 | 0.30 · 41 mm | 0.74 · 32 mm |
| C4-v2 | 0.48 · 29 mm | 0.84 · 21 mm |
| C1 | 1.15 · 11 mm | 1.01 · 11 mm |

**Reading: both (a) and (b) are low, which is Andrew's second branch.**
- **The readout lacks coverage.** C3-v1's readout was trained on actual future features, yet it recovers only 12% of moving travel even when given the actual future frame.
- **The predictor is not what caps it.** Its predicted change in features is about the actual change in size (1.11× moving, 1.22× from rest), with a median cosine of 0.49 between the two changes.
- **At rest the predictor does add some shortfall,** since (a) exceeds (b) there (six decisions). But (a) is itself far below the truth.
- **Sanity check** (`split_heldout_sanity.json`, 120 held-out offline windows with full forward commands). Each readout works on its own input there:

  | | From rest | Moving |
  |---|---|---|
  | C3-v1 on actual futures | 0.65 | 0.51 |
  | C3-v2 on predicted features | 0.81 | 0.75 |

  The computation is therefore sound. Both readouts fail only on the closed-loop states.
- **The information is in the inputs.** C4, which sees the same frames, command history and tape, predicts these states well (0.84 moving).
- **What differs most:** closed-loop forward decisions are taken while cruising.
  - The median forward command over the preceding second is 0.066, against 0.0 in both the held-out and training data (KS 0.58).
  - The command switches at 300 ms in 74% of closed-loop decisions, against 34% offline.
  - Speed reaches an upper quartile of 0.15 m/s, against 0.03–0.04 m/s.
  - Closed-loop starts from rest come after about 5 s of complete stillness, whereas offline rest windows follow 1.2-s holds with the body still settling.
  - Camera pitch and roll differ least.

**Recommendation.**
1. **Approve one bounded on-policy round (§7, proposed, not run).** C1 drives new training-only mazes to collect closed-loop states; the C3 and C4 readouts are refit as a matched pair. Acceptance criteria are declared first and scored on the closed-loop distribution, on new mazes. About 8 h without a safety check, 14 h with one.
2. **Launch E1 after that round, whatever its outcome,** with the version rule in §7. Until then the pre-declared rule stands: the C3-v2 and C4-v2 pair enters.
3. **Approve clearing at least 16 GiB of RecoveryStorage (§8).** One unreferenced 26.8-GiB directory from the closed 10 September programme covers it.
4. **E1 is not ready for confirmation until 1–3 are decided.**

**The rule is applied and E1 is held.**
- Under the pre-declared rule (§6 and Amendment 1), the **C3-v2 and C4-v2 pair enters E1**.
- As Andrew instructed, E1 is **not** ready for confirmation until the offline-to-closed-loop gap is diagnosed. That diagnosis is §5.
- Everything here uses the fresh-check mazes, the new recording mazes and development data. The validation and sealed sets were not re-run or opened. The mechanism check in §4 reads C3-v1's preserved validation logs, read-only, as Andrew asked.

## 1. Fresh check and the E1 rule

Ten fresh mazes, one episode each, registered and excluded from every other set. The harness is the frozen `v4_completed_support`. Cohort `cohorts/c3v2_fresh_check`, complete, with no stops; wall time 7.13 h.

| Controller | Round trips (sanity check only) | Disallowed contacts | Hard violations |
|---|---:|---:|---:|
| C1 | 9/10 | 0 | 0 |
| C3-v2 | 6/10 | 0 | 0 |
| C4-v2 | 7/10 | 0 | 0 |

**Applying the rule.**
- C3-v2 passed every §5 criterion and has zero contacts and zero hard violations, so **C3-v2 enters, and C4-v2 enters with it**.
- C4-v2 has no violation to flag.
- Round-trip counts do not select. Ten mazes cannot rank versions, and these are different mazes from validation.

**Failures, by frozen mechanism rule:**
- **C3-v2:** four 480-s timeouts.
  - Maze 00: no eligible movement on a frontier route (not the view requirement). 9.6 m remained at 480 s.
  - Mazes 01 and 06: latched recovery turn blocked by forecast clearance, with the latch active in 100% of final-window decisions. 8.2 m and 2.8 m remained.
  - Maze 09: turn oscillation near the goal on the return leg, with the latch inactive. 3.2 m remained.
- **C4-v2:** three timeouts, all within 5 cm of the goal.
  - Mazes 06 and 07: terminal heading limit cycle at the goal (0.04 m and 0.05 m remaining).
  - Maze 01: labelled "other stall" by the frozen rules, but it is 0.03 m from the goal with 741 planned-arrival-settling holds.
  - So all three C4-v2 failures are arrival failures at the goal, the trap-2 family.
- **C1:** one controller failure. Maze 09 lost measured visual pose at 165 s (`measured visual pose unavailable`), with zero contacts. This is the first pose-loss failure on the V4 harness; qualification had none.

## 2. Model versions

C3-v2 (readout `11a61e41…`) and C4-v2 (`68de16af…`) are recorded in [the model-version record](go2_navigation_c3v2_c4v2_model_versions_2026-09-29.md). C3-v1's validation result (13/20) remains the pre-registered capability result.

## 3. Answers to the three questions

**(a) Was C3-v2 fine-tuned from C3-v1, and was its recipe identical?**
- **Neither was fine-tuned from the other.** Both are fine-tuned from the same parent, `mixed_data_final.pt` (`bbbb05fd…`). C3-v1 is `maze_data_final.pt` (`aa853c6f…`).
- **The recipe is identical:**
  - 440 updates of batch 64 (32 "old" + 32 maze-pool);
  - AdamW with lr 1e-3 and weight decay 1e-4, gradient clip 1.0;
  - seed 2026092205, with the same schedule seeds (+0 for the old-context draws, +2 for the maze pool, +10 and +11 for horizons);
  - the fixed final checkpoint.
- **The old half of every batch is identical in both,** including the contexts, their order and the horizons. The 5,966 old contexts and the 2,448 v1 maze contexts are in the same order in both datasets; this was checked.
- **What differs:**
  - the maze pool: 5,216 contexts instead of 2,448, drawn by the same seed, so its draw sequence differs;
  - the future input: predicted features instead of actual future features;
  - v2's features were cached from 8-frame encoder batches, a recorded effect of at most 0.008 in fp16.
- **So extra training cannot explain the 41.3 → 19.5 mm gain on the other held-out windows.** The design cannot separate the two data changes, however.
  - C4-v2, which received only the data change, improved even more on the same held-out windows (27.1 → 8.7 mm). Those windows come from the same collector and tape as the new fit recordings, so a large part of the held-out gains is a match to that distribution.
  - On the disjoint transfer set, C3 improved by 41% (27.0 → 15.9 mm at 700 ms) and C4 by 17% (9.2 → 7.6 mm). That is consistent with the predicted-feature input helping C3 beyond the data change.
  - The queued 2×2 analysis fits (E1 plan §6) will split the two changes.

**(b) Is the transfer set disjoint, by maze, from the training data? How does 27.0 mm relate to 46.62 and 52.15 mm?**
- **It is disjoint.** Audit: `c3v2_data_v1/transfer_set_maze_disjointness.json`, script commit 3be46d17.
  - The 240 windows come from 2 mazes (8 cases).
  - They were compared by exact wall geometry, canonical topology and a rotation-invariant layout identity with every scene used by C3-v2 and C4-v2 training data (six sources, 172 directories).
  - Only two sources are 4×4 mazes: maze-view training (4 mazes) and the new fit recordings (4 mazes). The rest are 5-wall cluster layouts.
  - None is shared, and neither is any scene of C3's frozen predictor training (186 directories, 4 scenes).
- **The figures measure different window sets.**
  - 27.0 mm is the XY error over all 120 windows at 700 ms: 32 translation, 64 turn and 24 hold.
  - 46.62 and 52.15 mm are the translation-only windows on maze 0 and maze 1. The acceptance evaluator reproduces both exactly for C3-v1, which cross-checks the evaluator. Pooled over both mazes, the translation windows give 49.5 mm.
  - Translation windows are 27% of the set but carry 89% of the squared error. Turns are about 9.5 mm and holds about 12 mm.
  - For C3-v2 the same breakdown is 15.9 mm pooled and 27.7 mm on translation windows (18.9 and 34.3 mm by maze).

**(c) Are any of the replay's clock-time fields, or its log ordering, read by a decision?**
- **No.** Static trace:
  - **`added_routing_s`** is a `perf_counter` measurement in `clearance_preferred_route_development.refine_proposal`. It is written into the route receipt and copied to the planning log. The route consumers drop the receipt (`fine_goal_route_development:38`, `cached_fine_connectivity_development:81`), and no other code reads it apart from a test and an offline diagnosis script.
  - **`acquisition_wall_ms`** is set on the camera metadata row (`session.captured_pairs`). The controller receives only the sensor packet tuple and the simulated time; the row is read only by sensor retention for persistence.
  - **`decision_latency_s`** and the reader's `input_sha256` are computed by the frozen reader after the mission.
  - **The stage-log order:** `UntimedSimulationClock.end` appends to `releases`. That clock's `__call__` returns simulated time only; its parent class charged wall time. `releases` is saved at mission end and read only by the reader and the latency benchmark. The runtime writes to `self.planning[-1]` but never reads the logged receipts back.
- **The replay confirms it.** All of these values changed while every one of the 6,521 dispatch commands stayed identical.

## 4. Mechanism check (Andrew's request): C3-v2 on the fresh check versus C3-v1 on validation

**This is a mechanism check on 10 fresh mazes, not a result.** The two sets are different mazes and are not paired. C1 on both sets is a reference for how much the maze sets alone differ, and it suggests the fresh mazes are easier: C1's outbound hold rate is 7% there against 24% on validation. Source: `analysis/c3v2_mechanism_check/result_10of10.json`.

| | C3-v2, fresh (10) | C3-v1, validation (20) | C1, fresh (10) | C1, validation (20) |
|---|---|---|---|---|
| Outbound hold rate, pooled (per-mission mean) | 59% (38%), 3,908/6,603 | 62% (49%), 9,086/14,733 | 7% (6%) | 24% (14%) |
| Return hold rate, pooled (per-mission mean) | 23% (19%), 304/1,314 | 30% (22%), 960/3,203 | 5% (4%) | 3% (3%) |
| Hold decisions per mission: movement lost to hold on score or tie | 59 | 211 | 1 | 2 |
| Hold decisions per mission: no eligible movement | 113 | 197 | 0 | 0 |
| Hold decisions per mission: explicit override (latched recovery) | 250 | 94 | 24 | 89 |

**Deadlocks and hold stalls, by trap (timeouts, frozen rules):**

| Trap | C3-v2, fresh | C3-v1, validation |
|---|---:|---:|
| 1. No eligible movement (view requirement / other route state) | 0 / 1 | 4 / 0 |
| 2. Terminal limit cycle | 0 | 0 |
| 3. Latched recovery turn (latch active in final window) | 2 (100%, 100%) | 1 (100%) |
| Hold outscores movement stall | 0 | 2 |
| Turn oscillation, latch inactive | 1 | 0 |

**Predicted forward travel from rest in closed loop.**
- **Definition:** forward decisions with zero applied commands in the preceding 1.0 s, and with the executed tape equal to the predicted forward tape.
- **Results:**

  | Set | Decisions | Median predicted/true at 800 ms | Predicted | True |
  |---|---:|---:|---:|---:|
  | C3-v2, fresh | 6 | 0.04 | 2.5 mm | 58 mm |
  | C3-v1, validation | 14 | 0.14 | 7.9 mm | 59 mm |

- **All tape-matched forward decisions** (secondary): C3-v2 0.21 (79 decisions) against C3-v1 0.19 (153).
- **The v2 readout was active** in the check: the median logged 800-ms forward prediction is 19 mm against C3-v1's 10 mm. The v2 readout is predicting larger forward travel, but still far below the truth.

## 5. The offline-to-closed-loop gap (diagnosis on the check mazes)

All of this uses fresh-check mazes, offline only. Records are in `c3v2_gap_diagnosis_v1/`; the scripts are in commits 66838325 and later.

**Frames.** The fresh-check runs kept frame hashes only. The consumed RGB of all 20 C3-v2 and C4-v2 missions was regenerated by verified deterministic replay: pass 1 of the capability video renderer, unchanged. Every consumed-packet hash, action, dispatch command, applied command, native trace value, pose and mission row matched the log, with zero pose error. Output: `replays/*/replay_verification.json`, 1.8 GB.

**1. Pipeline test: passed.**
- **What was compared.** The logged inputs (frames, command history, candidate applied tapes) were run through the acceptance evaluator's computation:
  - C3-v2: the readout on the frozen predictor's features for the selected (and executed) candidate on every decision, and all six candidates on every 20th decision;
  - C4-v2: all six candidates on every decision.
- **Scale:** 20 missions, 14,397 decisions and 1,171,104 compared values. Every logged decision could be rebuilt offline.
- **Result:** the largest difference is 1.9×10⁻⁷ m in XY and 4.0×10⁻⁷ rad in yaw, against a tolerance of 10⁻⁵. There is no divergence to report.

**2. Like-for-like on the same decisions.**
- **Truth.** True travel is the physics-trace motion in the decision body frame. It is used only where the executed applied tape equals the scored candidate's tape.
- **Two kinds of forward decision.** Forward decisions whose tape was executed split into:
  - full forward commands (at least four 100-ms steps; 84 decisions);
  - 100-ms pulses (58 decisions, 49 of them from C4 missions: terminal pulses at the goal with about 3 mm of true travel).
  A ratio is meaningless for pulses, so they are reported by XY error only.

| Full forward commands | n | Median true | C3-v1 | C3-v2 | C4-v1 | C4-v2 | C1 |
|---|---:|---:|---|---|---|---|---|
| From rest: ratio · XY error | 6 (all from C3 missions) | 58 mm | 0.05 · 59 mm | 0.04 · 56 mm | 0.30 · 41 mm | 0.48 · 29 mm | 1.15 · 11 mm |
| Moving: ratio · XY error | 78 (66 C3, 12 C4) | 118 mm | 0.13 · 82 mm | 0.21 · 70 mm | 0.74 · 32 mm | 0.84 · 21 mm | 1.01 · 11 mm |

**Pulses** (n=58, true 3 mm). Median XY error: C3-v1 7 mm, C3-v2 8 mm, C4-v1 8 mm, C4-v2 9 mm. C1 is 17 mm, because it over-predicts pulses.

**C4-v2's forward predictions did not fall.** On like-for-like states it beats C4-v1 (0.84 against 0.74 moving; 0.48 against 0.30 from rest). The lower median in the mechanism check came from the mix of states: most executed forward decisions on C4 missions are pulses at the goal.

**3. Conditions.**
- **The comparison.** 14,383 closed-loop decisions against the 1,384 held-out offline contexts, and against the 11,182 training contexts. Continuous features use the KS distance and proportions use their absolute difference; features are ranked by distance. Files: `conditions_20runs.json` and `conditions_20runs_with_training.json`.

| Moving full-forward decisions (closed loop, n=134 executed forward candidates) | Closed loop | Held-out | Training | Distance (held-out / training) |
|---|---|---|---|---|
| Forward command, preceding 1 s (median, IQR) | 0.066 (0.02–0.16) | 0.0 (0–0.064) | 0.0 (0–0.048) | 0.56 / 0.58 |
| Prefix differs from the rest of the tape | 74% | 34% | 34% | 0.40 / 0.40 |
| Speed now (median, upper quartile) | 0.032, 0.153 m/s | 0.016, 0.033 m/s | 0.018, 0.039 m/s | 0.34 / 0.27 |
| Mean speed over preceding 1 s (upper quartile) | 0.147 m/s | 0.060 m/s | 0.041 m/s | 0.29 / 0.35 |
| Camera pitch (median) | −0.72° | −0.50° | −0.66° (sources with camera metadata) | 0.19 / 0.10 |
| Camera roll (median) | 1.94° | 1.87° | 2.02° (sources with camera metadata) | 0.13 / 0.13 |

- **Starts from rest** (8 closed-loop decisions): commands had been zero for at least 1.45 s and the body still for a median of 5 s. Offline rest windows follow 1.2-s holds, with a body speed of 0.005–0.03 m/s.
- **Cruising speed alone is not missing from training.** 19% of training contexts exceed 0.1 m/s, against 34% of the executed closed-loop forward candidates. What training lacks is sustained forward command history with a command switch inside the tape.

**4. Where the failure sits (split test).** The table at the top, from `localisation.json`.

## 7. Proposed bounded round: on-policy readout coverage (not run; needs Andrew's approval)

The purpose is to give C3's readout, and C4 as its matched pair, the closed-loop state distribution it lacks, and then launch E1 whatever the result.

1. **Data.**
   - Register 22 new training-only mazes from the same generator, excluded from every existing set (development, validation, fresh check, recording, sealed).
   - C1 drives each for one episode on the frozen V4 harness: 16 mazes for fitting, 6 held out for acceptance. C1 is deterministic, capable (18/20) and non-learned, so its visits are the controller-independent closed-loop distribution.
   - Frames are regenerated by verified replay, as in this diagnosis.
   - Every decision with a complete causal context and executed tape becomes a context: frames, command history, executed tape and physics-true targets. That gives about 6,000 fit contexts and 2,500 held out.
2. **Fits, a matched pair with the recipe otherwise identical.**
   - **C3-v3 readout:** initialised from `mixed_data_final.pt`, 440 updates, seed 2026092205, predicted-feature input, fixed final checkpoint.
   - **C4-v3:** C4's recipe, 1,760 updates, seed 2026092513.
   - **Batches, declared in advance:** 32 old contexts plus 16 from the C3-v2 maze pool plus 16 on-policy contexts. The fixed on-policy share guarantees exposure within 440 updates; without it each on-policy context would be seen about 0.7 times.
3. **Acceptance, declared and committed before any fit,** on the 6 held-out on-policy mazes (closed-loop distribution, new mazes). Measured on executed full forward commands, against C3-v2 on the same decisions:
   - **A:** moving, median predicted/true 800-ms ratio in [0.75, 1.25], and median XY error at most 50% of C3-v2's.
   - **B:** from rest, median ratio at least 0.5. It gates only if n ≥ 10; below that it is reported only.
   - **C:** no loss on the existing §5 held-out groups or the transfer population (at most 1.05× C3-v2's).
   - C4-v3 is reported, not gating.
4. **Version rule for E1, declared with the criteria.**
   - C3-v3 and C4-v3 enter if C3-v3 passes A–C; otherwise C3-v2 and C4-v2 enter.
   - E1 launches after the round either way.
   - **Optional:** a 10-maze closed-loop safety check of C3-v3 and C4-v3 on new mazes (zero contacts and hard violations required), about 6–7 h. A readout predicting four times more travel will translate more near walls, so I recommend it; it is Andrew's call.
5. **Cost:**
   - C1 missions about 1 h (CPU, 4 in parallel); replays about 1 h;
   - C3-v3 features about 2.3 h; C4-v3 about 2 h (C4's total rises to about 5.4 of its 12 GPU-h);
   - acceptance about 1 h.
   - That is about 8 h, or about 14 h with the safety check.
   - Storage is about 3 GB.
   - This runs before E1, under the programme's active-time accounting. Its missions still carry the owner's calendar stop (2 Oct 03:47 BST): if the round starts after about 1 Oct 12:00, its C1 missions need the same running-time `Budget` swap as E1.

## 8. Storage for E1 (candidates for Andrew's approval; nothing has been touched)

- **The requirement, as I read it:** after E1's projected footprint (64.9 GiB), at least 15 GiB must remain above the 12-GiB reserve. So E1 must start with at least 92.0 GiB free.
- **Current state:** RecoveryStorage has 81.1 GiB free, a **shortfall of 10.8 GiB**. With the bounded round's roughly 3 GB, clearing **at least 16 GiB** is needed.
- **No move target exists.** The only other data disks are the workspace drive (5.5 GiB free) and the Steam drive (0.5 GiB free), so these are clear-only candidates.

**Survey method** (`scripts/survey_go2_e1_storage_candidates_development.py`, result `e1_storage/candidates.json`). A directory counts as referenced if its name appears in:
- the 303 frozen-harness source files;
- the capability, C3-v2 and E1 documents;
- the current training lineage (C3-v2/C4-v2 data, C4-v1 preparation, the C3 readout and predictor training records, the transfer set);
- the E1 seed scripts and everything they import.

**Protected:**
- the V4.2 audit and every decision-headroom directory (`go2_decision_headroom_*`, `go2_headroom_*`), per Andrew;
- the capability artifact root;
- every referenced directory.

| Candidate (not referenced by the current programme) | Size | Date | Note |
|---|---:|---|---|
| `go2_supervised_rollout_mazes_v1_attempt_001` | 26.8 GiB | 10 Sep | **Recommended: covers the need alone.** Closed pre-capability programme. |
| `go2_stop_conditioned_independent_00_frozen_reference_seed_2026091001_full_direct_v1_attempt_001` | 17.3 GiB | 13 Sep | Closed programme; cited by 13 Sep docs |
| `go2_stop_conditioned_independent_00_frozen_reference_seed_2026091001_full_supervised_rollout_v1_attempt_001` | 16.7 GiB | 13 Sep | As above |
| `go2_stop_conditioned_settling_maze02_v1_attempt_001` | 13.9 GiB | 13 Sep | Closed programme |
| `go2_extended_return_budget_maze02_v1_attempt_001` | 13.9 GiB | 12 Sep | Closed programme |
| `go2_measured_plane_chained_maze02_v1_attempt_001` | 11.6 GiB | 12 Sep | Closed programme |
| `go2_clearance_preferred_arc_recovery_learned_round_trip_native_layout03_v1_attempt_001` | 11.2 GiB | 13 Sep | Closed programme |
| `go2_no_rgb_direct_extended_budget_maze02_pilot_v1_attempt_001` | 11.1 GiB | 11 Sep | Closed programme |
| `c3v2_gap_diagnosis_v1/replays/*/ego_frames` | 1.7 GiB | 29 Sep | Regenerable by verified replay; the diagnosis results are kept |

- **Consequence of deleting:** these are historical records of closed programmes. Deleting one destroys its raw evidence, and older documents that cite it lose their backing data. The capability protocol's `no_existing_artifact_retirement` rule means each removal needs Andrew's explicit approval.
- **Recommendation:** approve the first row, which leaves about 26 GiB above the reserve after E1 and the round.

## 6. Other deliverables

- **Harness known limitations:** [the note](go2_navigation_harness_v4_known_limitations_2026-09-29.md), with a C4 column. C4's validation oscillation (22/0) is trap 3 in its turning form. C1's fresh-check pose loss (maze 09, 165 s) is added as a separate shared-perception limitation.
- **Held-out isolation for C4-v2: passed** (`c3v2_data_v1/c4v2_heldout_isolation_audit.json`). The 14,340 encoded frames are:
  - 9,948 carried over from C4-v1;
  - 2,928 from the new fit recordings;
  - 1,464 held-out frames, encoded but never used by a training context.
  The 11,182 training contexts use 12,876 unique frames.
- **E1 launch plan:** [the plan](go2_navigation_e1_launch_plan_2026-09-29.md), with the answer at the top.
- **Budget.**
  - Programme active time is 101.0 h of 160 h, as of 30 September 01:40; the fresh check ended at 97.1 h.
  - E1's own running time so far is 0.23 h (the budget replay checks).
  - The owner's calendar stop (2 October 03:47 BST) is unchanged for this phase.
  - One active-time interval was closed by hand: a scorer was restarted as three parallel subsets. Its ledger end row says so and was recorded conservatively at the restart.
