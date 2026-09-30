# E1 launch plan, 29 September 2026

## Update, 30 September: round approved and running; storage awaiting approval

- **The bounded round is approved**, with the 10-maze safety check on newly generated mazes. It is pre-declared in [the round pre-declaration](go2_navigation_c3v3_onpolicy_round_predeclaration_2026-09-30.md) (commit ec2e34c9, before any work). It is the second and final C3 intervention before E1, and C3-v1 remains the capability result.
- **Which version enters E1:**
  - C3-v3 and C4-v3 enter if C3-v3 passes the offline acceptance and has zero contacts and hard violations in the safety check. A C4-v3 violation is flagged for Andrew.
  - Otherwise C3-v2 and C4-v2 enter.
- **E1 launches only after Andrew's confirmation,** whatever the outcome.
- **Storage.** Nothing is deleted. Provenance tracing shows that no current training set or the transfer set was derived from `go2_supervised_rollout_mazes_v1_attempt_001`. Its manifest and summary are in `docs/storage_manifests/`, and deletion awaits Andrew's go-ahead.

## Status: not ready for confirmation (30 September)

1. **The gap is diagnosed.** C3's motion readout does not cover the closed-loop state distribution.
   - It is not the pipeline: a bitwise-level match on all 14,397 fresh-check decisions.
   - It is not the predictor: the readout trained on actual futures also recovers only 0.12 of true moving travel, even from the actual future frame. C4 recovers 0.84 on the same states. The [phase report](go2_navigation_c3v2_phase_report_2026-09-29.md) has the split test at the top.
2. **Proposed next step (not run):** one bounded on-policy round. C1 drives new training-only mazes; C3 and C4 are refit as a matched pair; acceptance criteria are declared first and scored on the closed-loop distribution, on new mazes. About 8 h, or 14 h with a safety check. E1 launches after the round whatever its outcome. See the report §7.
3. **Storage:** at least 16 GiB must be cleared first. The candidates, for approval, are in the report §8.
4. **What enters E1 now:** until the round decides otherwise, the pre-declared rule puts in C3-v2 and C4-v2.

**This is a plan for approval. Nothing has been launched and the sealed set is untouched.** E1 needs Andrew's confirmation. The design is the one approved in principle on 29 September: 60 sealed mazes × 1 episode × 3 training seeds, with no real-time mode. See the [E1 proposal](go2_navigation_e1_proposal_2026-09-29.md), the [C3-v2 pre-declaration and Amendment 1](go2_navigation_c3v2_readout_fix_predeclaration_2026-09-29.md), the [phase report](go2_navigation_c3v2_phase_report_2026-09-29.md), and the [harness known limitations](go2_navigation_harness_v4_known_limitations_2026-09-29.md).

## 1. What enters E1

**Under the pre-declared rule (§6 and Amendment 1), the C3-v2 and C4-v2 pair enters E1.**
- C3-v2 passed all seven offline criteria.
- In the fresh check it had zero disallowed contacts and zero hard-clearance violations over 10 missions.
- C4-v2 also had none, so there is nothing to flag for C4.

The versions are recorded in [the model-version record](go2_navigation_c3v2_c4v2_model_versions_2026-09-29.md):
- C3-v2 readout `11a61e41…`;
- C4-v2 `68de16af…`.

**The offline-to-closed-loop gap is now diagnosed** (phase report, top and §5). E1 stays unconfirmed until Andrew decides on the bounded round (report §7) and the storage clearance (report §8).

## 2. Design

- **Harness.** The frozen `v4_completed_support` (sha256 `82b7b604…`, commit 7da82b23), with the frozen readers and the prefix and reader errata. No harness file is edited.
- **Episodes.** Episode 0 of each of the 60 sealed mazes, paired across controllers; episode 1 is held in reserve. The sealed packets are opened only after confirmation, by a loader that checks every packet against the capability inventory's registered hashes before use.
- **Controllers:**
  - C1 command history, one instance (its fit is deterministic);
  - C2 reactive, one instance;
  - C3 (JEPA) and C4 (supervised predictor), three training seeds each;
  - C0 on the 10 lowest sealed maze IDs, only as a harness check.
- **Mode.** Physics-paused, as in qualification. Decision latency is reported as a cost.
- **Outcome.** As pre-registered in the proposal §2: the paired per-maze difference in round-trip success, averaged over seeds within each maze, with a maze-cluster bootstrap. The comparisons are H2 (C3 against C1), H4 (C3 against C4), and C3 against C2. SPL, time and stalls are secondary outcomes, and safety is a hard constraint.
- **Mechanism accounting.** Every failure is labelled with the frozen rules in `diagnose_go2_capability_validation_timeouts_development.py`. Trap 3 is reported both ways: the frozen-rule label, and the fraction of final-window decisions with the latched clearance turn active. Pose-loss controller failures (one in the fresh check, C1 maze 09) are reported as their own category.

## 3. Training seeds

- **Seed 1** is the recorded version of each controller in the selected pair.
- **Seeds 2 and 3** rerun that pair's recipe and data with only the seed changed. Every seed that each recipe derives from its base seed (sampling cycles, horizon schedules, torch and numpy seeds) moves with it.

| Part | Seed 1 | Seeds 2 and 3 |
|---|---|---|
| C3 action-conditioned predictor | 2026091802 | 2026091803, 2026091804 |
| C3 motion readout | 2026092205 | 2026092206, 2026092207 |
| C4 | 2026092513 | 2026092514, 2026092515 |

**Each C3-v2 seed is a chain.**
- First, fine-tune the predictor from its fixed parent checkpoint.
- Then re-extract the readout's predicted-future features with that seed's predictor. They cannot be reused across seeds, because the features depend on the predictor.
- Then fit the readout.

**The seeds are asymmetric, and their spreads are not compared.**
- **C3's learned parts are fine-tuned from fixed weights.** The predictor starts from its fixed parent checkpoint and the readout from `mixed_data_final.pt`. A C3 seed changes only the order and horizons of the training samples.
- **C4 trains from random initialisation.** Its seed changes both the initial weights and the sample order.
- So the two seed spreads measure different things. Results are reported per seed, and the C3 and C4 spreads are not compared with each other.

**Every seed enters E1, whatever its offline numbers.** Acceptance selects the version, not the seeds. Each seed's §5 acceptance measures are reported (not gating).

## 4. Running time: accounting, projection and cap

**How the calendar stop is replaced.**
- Each E1 mission runs the frozen owner's `run` with three functions swapped by `bind`, as the fresh check already did for two of them:
  - the episode loader (the sealed loader above);
  - the model loader (the selected version and seed);
  - `Budget`, which becomes `lewm/e1_running_time_budget_development.running_time_budget(cap)`.
- The swapped `Budget` removes only the owner's 160-h calendar condition. It stops at E1's own running-time cap: the union of intervals of jobs named `E1 …` in the active-wall ledger.
- The disk, memory, VRAM and closeout-admission checks are the owner's own code, unchanged.

**No harness file is edited, and no decision code changes.**
- **Bitwise replay (development maze 00/0, C1).** With only `Budget` swapped, attempt 2 of the check (`e1_budget_replay_check_v2`, commit b65d4dd2) reproduced every record exactly: all 6,521 dispatch commands and applied commands, 310 model calls, 1,305 mission rows, poses and consumed frames, the selected actions in all 326 planning rows, and every native physics array.
- **The allowed differences are clock and log-order fields only:** routing compute time, camera acquisition wall time, decision latency, the reader's hash of the planning file, and the order in which parallel services log at equal simulated time.
- **None of these can reach a decision:**
  - The routing time is written into the route receipt and copied to the planning log; route consumers drop the receipt.
  - The acquisition time lives on the camera metadata row, and the controller receives only the sensor packets and the simulated time.
  - Decision latency and the file hash are computed by the reader after the mission.
  - The harness clock returns simulated time only; its stage log is appended and saved, and never read during a run.
- **Attempt 1** failed its pre-declared allow-list because it had missed those fields. It is preserved.

**Projection** (`scripts/project_go2_e1_runtime_development.py`, result `e1_projection/projection_v2_10c3_with_analysis_fits.json`).
- Mission wall times are measured, not assumed:
  - C1, C3-v2 and C4-v2 from the fresh check (10 each);
  - C2 and C0 from capability validation.
- Missions are scheduled with the cohort runners' rule: at most 5 owners and at most 2 C3 owners. The makespan is bootstrapped 1,000 times over the measured times.
- Seed training, per-seed acceptance and the §6 analysis fits run as serial GPU jobs, with measured durations.

| Block | Missions | Median makespan | p90 | Serial GPU jobs |
|---|---:|---:|---:|---:|
| 1 (pilot: seed 1, plus C1, C2 and the C0 subset) | 250 | 38.2 h | 42.4 h | none |
| 2 (seed 2) | 120 | 38.1 h | 42.2 h | 11.5 h: 5.0 h of seed training and acceptance, plus 6.5 h of analysis fits |
| 3 (seed 3) | 120 | 38.1 h | 42.0 h | 5.0 h |
| **Total** | 490 | **130.9 h** | **143.1 h** | |

**The running-time cap is 157 h.** That is the median projection plus the 0.23 h of E1-named jobs already run (the budget replay checks), plus a 20% margin. C3 is the long pole: its median mission takes 3,256 s and its mean 4,524 s (up to 9,683 s for a timeout), with two lanes.

**Storage.**
- **E1's projected footprint is 69.7 GB (64.9 GiB),** using mean bytes per mission:
  - C3 29.8 GB, C4 24.9 GB, C2 8.7 GB, C1 5.3 GB, C0 1.1 GB.
- **RecoveryStorage has 81.1 GiB free.** Keeping 12 GiB free leaves about 4.2 GiB of headroom, so storage is the tighter constraint.
- **E1 needs more room before it starts.** It must start with at least 15 GiB above the reserve after its own footprint, so at least 92.0 GiB free. That is a shortfall of 10.8 GiB, or about 16 GiB including the bounded round. The candidates are in the phase report §8, and deleting anything needs Andrew's approval.

**Stops.**
1. **Hard cap, in every mission.** The swapped `Budget` raises a resource stop when E1 running time reaches the cap minus a 120-s closeout reserve. The E1 cohort runner also refuses a new launch if running time plus the projected remainder exceeds the cap.
2. **Pilot and seed-boundary checkpoints.** After the pilot block and after each later seed block, the remaining running time is re-projected from E1's own completed missions. If the projected total exceeds the cap, E1 stops and reports.
3. **Storage stop at each seed boundary.** Before the next block launches, its projected storage (about 18.2 GB for 60 C3 and 60 C4 missions) plus 12 GiB must fit in RecoveryStorage's free space; otherwise E1 stops and reports. The projection uses the measured mean bytes per mission for each controller. The owner's per-mission admission already refuses any mission that would breach the 12-GiB reserve.
4. **Safety.** Any disallowed contact, hard-clearance violation, technical failure or closeout defect stops new launches, as in qualification.

## 5. Order of work after confirmation

1. Write and commit the E1 entry and cohort scripts: the sealed loader, the seeded model loader and the running-time budget. Hash-check every sealed packet before the first mission.
2. **Pilot block,** then checkpoint (§4).
3. Train seeds 2 and 3 and compute their acceptance measures, then run the §6 analysis fits.
4. **Seed-2 block**, then checkpoint. **Seed-3 block**, then checkpoint.
5. Analyse and report against the pre-registered outcome, with mechanism accounting, per-seed results and latency. Declare the sensor idealisation.

## 6. Queued offline analysis (never enters E1)

Andrew's request of 29 September: two extra C3 readout fits that split C3-v2's gain between its two data changes. They use the identical recipe (initialisation `mixed_data_final.pt`, 440 updates, batch 32 + 32, the same optimiser, seed and schedule seeds, fixed final checkpoint):

| Fit | Data | Future input | Where it sits in the 2×2 |
|---|---|---|---|
| C3-v1 (exists) | Old data | Actual future features | Old × actual |
| **(a)** | Old data plus the new recordings (C3-v2's data) | Actual future features | New × actual |
| **(b)** | Old data only (C3-v1's data) | Predicted features | Old × predicted |
| C3-v2 (exists) | Old data plus the new recordings | Predicted features | New × predicted |

- **Scoring and records.** Both fits are scored with the acceptance evaluator on the same held-out and transfer measures as C3-v1 and C3-v2, and the 2×2 is reported. They are recorded as analysis fits that never enter E1.
- **Timing and cost.** They run after E1 launches, in block 2, and cost about 6.5 GPU-h:
  - (a): actual-future features for about 12,600 frames, at the C3-v1 readout fit's measured rate;
  - (b): predicted features for 8,414 contexts, at the C3-v2 fit's rate;
  - one pass of the acceptance evaluator.
- **In the projection.** The cost is included, so it lengthens E1's wall time rather than hiding it.

## 7. Risks

- **C3's closed-loop motion deficit.** Unless the bounded round fixes the readout's coverage, E1 measures C3 with a readout that predicts about 0.2 of the true forward travel in the states it visits. E1's C3 comparisons would then largely measure that deficit, and the report must say so.
- **Storage.** Even after the recommended clearance, E1 uses most of RecoveryStorage. The seed-boundary storage stop is the guard.
- **One GPU.** C3 is the long pole, with two lanes. A second GPU would roughly halve E1.
- **Shared-harness limitations.** Traps 1–3 and visual pose loss are counted per controller with the frozen rules. See the limitations note.
- **Sensor idealisation.** Noise-free RGB and an ideal gyro and accelerometer (the accelerometer is used for initial gravity alignment) must be declared in any E1 write-up.
- **Maze sets differ in difficulty.** C1's outbound hold rate was 7% on the fresh mazes against 24% on validation. The sealed set's difficulty is unknown until E1 runs, and the paired design is what protects the comparisons.
