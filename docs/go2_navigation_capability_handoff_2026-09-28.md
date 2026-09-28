# LeWMQuad-v3 handoff: navigation testbed and controller capability

**Prepared:** 28 September 2026, for a fresh agent taking over from the current session
**Approver:** Andrew Knowles
**Workspace:** `/home/andrewknowles/Workspace/LeWMQuad-v3`
**Artifact root:** `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/`
**Authoritative brief:** `docs/go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md`, as amended in section 5

This handoff was written from the reports the previous agent sent Andrew, not from the repository itself. Two rules follow from that:

- **Facts:** where this document and the records on disk disagree about a result, version, count or path, the records win. Stop and report the discrepancy before running anything.
- **Instructions:** where this document and older instructions disagree about what to do, this document wins.

Items marked **(confirm)** could not be checked from the reports; resolve them from the records.

> **Corrections, 28 September 2026** (after [the orientation note](go2_navigation_capability_orientation_2026-09-28.md), approved by Andrew Knowles):
> - **§2 and §3.** 02/0 had **one** decision without a matching branch (six branch rows), not six decisions. At the 119.9-s boundary the current observation was missing on the first tick, and vetoes followed. The committed forward prefix shared by all six branches therefore never executed, and the plan was then held without motion.
> - **§2 and §7.** Five changes are charged: startup_c2, V1, V2, V3 and V4. V3 (`live_turn`) was the turn-memory change (8/10). V4 (`completed_support`) is the recovery change that uses the tracker's actual feature count (10/10 on both screens).
> - **§4 and §5.2.** The pre-registration governs the cap: six versions **including V0**. V4 is the last version. If the gate fails, stop and report; Andrew decides whether to allow another change. Earlier records that say "five of six consumed" are corrected by the [version ledger](go2_navigation_capability_version_ledger_2026-09-28.md).
> - **Further rulings.** Causes of no-match decisions are dispatch-layer substitutions (missing current observation, veto or override); see [the erratum](go2_navigation_capability_oracle_prefix_erratum_2026-09-28.md). From the orientation stop onwards, wall time counts only while jobs run (§5.7). All large outputs and temporary files go on RecoveryStorage, including video render intermediates (§5.7).
> - **Confirmed items** are resolved in the orientation note.

---

## 1. Goal

**Programme.** Determine when JEPA representations carry decision-relevant information for navigation, beyond what kinematic motion prediction, reactive control and explicit memory already provide. This is a JEPA-based PhD about useful feature representations for general robotics. The simulated Unitree Go2 navigation stack is the testbed, not the thesis.

**Current phase.** Establish a validated navigation testbed. Then determine, for each controller type, whether it can reach a beacon in an unseen maze and return home. Report each controller against the pre-registered capability criterion, with one example video per controller.

**Done** when all of the following hold:
- the C0 oracle gate has passed on a frozen harness version;
- capability qualification has run on the validation set;
- the capability report, videos and E1 proposal are committed;
- work has stopped.

A JEPA loss is an acceptable result. An unvalidated testbed is not.

---

## 2. Current state (28 September, 09:38)

| Stage | Status |
|---|---|
| C1 screen, first dev-tune episodes | 10/10 round trips, safe |
| Second-episode C1 check | 10/10, safe |
| C0 oracle gate (all 20 dev-tune episodes) | Stopped: 4 qualified successes; 5th attempt (02/0) unqualified, with one decision (six branch rows) lacking a matching executed prefix (section 3) |
| Capability qualification | Not started |
| Example videos | Not started; renderer prepared, pipeline-test video done |
| Capability report, E1 proposal | Not started |

- **Harness:** `completed_support_v4` (`82b7b604…`, frozen at `7da82b23`). Five changes are charged (startup_c2, V1–V4). With V0 that makes six versions, which is the cap under the pre-registration. **V4 is the last version.**
- **Safety:** zero disallowed contacts and zero clearance violations in every run of this programme.
- **Throughput:** C1 runs at about 2 wall-seconds per simulated second, and a ten-episode C1 screen takes 2–3 hours. C0 is much slower, because it branches the physics at every decision.
- **Budget and storage:** hours used against the 160-hour cap are in the latest budget re-projection **(confirm)**. About 100 GiB was usable on RecoveryStorage at the last check.

---

## 3. Immediate task: resolve the C0 stop and resume the gate

**What happened.** In C0 episode 02/0, at one decision (the 119.9-s source boundary), all six of C0's physics branches began with the same committed forward prefix. That prefix was carried over from the 119.5-s selection. On the first tick the shared harness's dispatch layer substituted a hold, because the current observation was missing (`CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE`), and 14 vetoed ticks followed (`COMMAND_WINDOW_VETO_LATCHED`). So that one decision had no executed prefix with the same commands to compare against, which gave six branch rows. The harness then held that plan for 20 ticks (`COMMITTED_PREFIX_NOT_EXECUTED`), so it produced no motion. The same sequence occurs in C1 runs.

All 2,322 comparable prefixes matched exactly, with no pose discrepancy. The agent preserved the attempt and stopped, as its rule required.

Stop report: `docs/go2_navigation_capability_completed_support_v4_oracle_stop_2026-09-28.md` (commit `b5ad9f7d`).

**Disposition (approved with this handoff):**

1. **This is not a fidelity failure.** The fidelity check exists to catch restoration errors, and a vetoed candidate was never executed. The original instruction already scoped the check to "every executed candidate".
2. **Record an erratum clarifying the check.**
   - Compare the realised motion against the branch whose command tape matches what was actually dispatched over the compared prefix. That is usually the selected candidate, but it may be another candidate, such as hold.
   - Any mismatch on a comparable prefix is still a stop.
   - Count decisions with no matching branch per episode, with their cause (veto or override), and report them. They neither qualify nor disqualify an episode.
3. **Resolve the fifth attempt.**
   - **If it reached a terminal mission outcome:** re-evaluate it from its preserved records under the erratum. Don't rerun it. Its navigation outcome stands.
   - **If the check aborted it before a terminal outcome:** keep the aborted attempt and run 02/0 once more, as a new, labelled attempt under the erratum. This is not a retry of a navigation failure. Its trajectory must reproduce the aborted attempt exactly up to the abort point.
4. **Resume the gate** with the remaining episodes, on the same frozen harness and C0 implementation. Don't rerun completed episodes.
5. **Watch for veto loops.** C0's branches don't model the veto, so C0 can keep choosing a move that the veto keeps blocking.
   - Report veto counts per episode.
   - If a C0 failure involves repeated vetoed selections, diagnose it as a harness mechanism, like any other gate failure.
   - Making C0's branches include the veto would change the oracle's definition, and needs Andrew's approval.

Before resuming, confirm from the records that:
- the first C0 gate episode passed its full-frame replay check (section 5.6);
- later C0 runs use hash retention.

---

## 4. After the gate

- **If the gate passes** (C0 completes at least 19 of 20 dev-tune round trips, with zero disallowed contacts and zero hard-criterion violations): run capability qualification, produce the videos, and write the report and the E1 proposal (section 6). Then stop.
- **If the gate fails:**
  1. Classify the failures with the same mechanism accounting used so far.
  2. Make one change under the protocol (section 5.2).
  3. Screen C1 on all 20 dev-tune episodes; at least 18 of 20 must pass.
  4. Rerun the full C0 gate on the new version.

  If the six-version cap is reached without a pass, stop and report. **V4 is already the sixth version, counting V0**, so a failed gate means stop and report. Andrew then decides whether to allow another change.

---

## 5. Rules in force

These are the brief plus every amendment approved since.

### 5.1 Data roles and integrity

- **Maze sets.** All are registered and hashed. Audit layouts 00–07 are excluded from every set.
  - **Dev-tune:** 10 mazes × 2 episodes (NN/0 first, NN/1 second). Used for harness iteration, the second-episode check and the C0 gate.
  - **Validation:** 20 mazes × 2 episodes registered. Capability qualification uses one episode per maze, under the approved time reduction **(confirm which episode)**. Never used for iteration.
  - **Sealed test:** 60 mazes, 120 episodes, reserved for E1. Structural checks are allowed, with sealed results reported only in aggregate. Never run, render or otherwise inspect them.
- **Pre-registration.** Committed before the pilot. It fixes the episode rules, the 480-s mission budget, the arrival criteria, the SPL inflation, the stall definition, the C3 head and the C4 source. Any change needs an erratum, recorded before the next run.
- **Paired design.** Every controller runs the same episodes. The maze is the unit of inference.
- **Records.**
  - No silent retries. Preserve every failure and never overwrite.
  - Use fresh output roots.
  - Hash every maze, episode, harness version, model and configuration.
  - Commit docs and configurations to Git; keep large artifacts under the artifact root.
- **Labelling.** Capability results are capability qualification, not paper results.
- **Provenance caveat.** Training renders for the current predictor and both motion readouts are unverified: the historical re-render check left 16 examples unresolved for missing restoration inputs. Carry this caveat into any attribution of readout or forecast losses.

### 5.2 Harness change protocol

- **One change per version** to the shared harness, identical for all controllers. Freeze and hash each version with its rationale, diff and expected effect.
- **Allowed without approval:**
  - mapping and memory marking;
  - observation and view-seeking requirements;
  - eligibility and clearance rules;
  - recovery behaviour, including pose-loss recovery and falling back to the downward camera;
  - stopping logic;
  - selector tie-breaks.
- **Needs approval:**
  - changes to the visual tracker's estimation;
  - changes to learned models, sensors or the candidate bank;
  - privileged information in the deployed harness;
  - per-controller special cases;
  - the mission budget.
- **Six-version cap.** Stop and report if the cap is reached without passing the gate. The pre-registration governs: six versions **including V0** (IDs 0–5). V0 plus five charged changes means V4 is the last.
- **Correctness fixes don't count toward the cap** when both of these hold:
  - an outcome-independent test fails before the fix and passes after it (for example, a contract on geometry, indexing or initialisation, checked against ground truth);
  - containment passes: episodes the defect couldn't affect reproduce exactly (native arrays, requests and consumed sensor hashes) up to the first frame at which the defect could have mattered.

  If containment fails, the change counts as a version and the cause is reported. Two containment failures in a row stop work.
- **Startup defects.** A failure that happens before the first planning decision, and that is predictable from registered geometry, is a startup defect. It qualifies as a correctness defect.
- **Safety.** A version may become less conservative, but never less safe. Any disallowed contact or hard-criterion violation disqualifies that version.
- **Budget.** Don't change the 480-s mission budget. If a failure shows steady progress with little holding, flag it to Andrew as a possible budget issue.

### 5.3 Gates

- **C1 screens:**
  - first dev-tune episodes: at least 9/10 (passed);
  - second-episode check: at least 9/10 before the C0 gate (passed);
  - after any further harness change: all 20 dev-tune episodes, at least 18/20.
- **C0 gate:** at least 19 of 20 dev-tune episodes, zero disallowed contacts and zero hard-criterion violations.

### 5.4 Task, evaluation and metrics

- **Mission.** Start at home, reach the beacon, and return home within 480 s of simulated time. The robot settles for 1.5 s before the mission starts.
- **Settled-start task transform** (erratum, frozen in `13ebc8e2`).
  - At mission start, after settling, the task layer converts both mission cues once into the robot's settled start frame. It uses the simulator's settled pose, in the tracker's planar origin convention.
  - That pose is logged on the evaluator side only.
  - Controllers receive the converted cues and no true pose at any other time.
  - Settling moves the robot about 15 mm and 0.3°; before this fix, controllers aimed about 30 mm off the beacon.
- **Arrival reader** (evaluator correction). It checks the registered world beacon and home directly, requiring:
  - a 40-mm radius;
  - a 1-s dwell;
  - motion below the 50-mm/s quiet threshold;
  - zero commands.
- **Safety.**
  - Native disallowed contacts, excluding contact with the supporting ground.
  - Articulated clearance over all 27 collision primitives at native 2-ms steps, with a 5-mm hard criterion and a 20-mm operating margin.
  - Report the FK interval robustness bound.
- **Metrics:**
  - beacon, home and round-trip success;
  - SPL per leg, using the shortest path on the true map at the pre-registered inflation;
  - time to beacon and to home;
  - contacts and margin violations;
  - stall rate: hold decisions by mission phase;
  - decision latency: wall time per planning decision, median and 95th percentile, with physics paused;
  - wall time per episode;
  - a failure taxonomy by mechanism.

### 5.5 Controllers

All controllers run on one frozen harness: simulated RGB, depth with the existing noise treatment, an ideal gyro, measured visual pose, mapping, memory, routing, the six-candidate bank (hold, forward, left arc, right arc, left turn, right turn), the safety filter and the selector. They differ only in what fills the prediction slot.

| ID | Controller | Prediction slot |
|---|---|---|
| C0 | Oracle (diagnostic only) | True candidate motion from restored physics branches at each decision. Never receives the true map or pose. |
| C1 | Command history | Existing fitted command-history motion model |
| C2 | Reactive | None; existing reactive selector |
| C3 | JEPA | Frozen V-JEPA 2.1 encoder, action-conditioned predictor and the pre-declared motion head **(confirm which)** |
| C4 | Supervised predictor | Direct regression from the same frozen features and command tape to the same motion targets **(confirm source)** |

### 5.6 Retention, replay and environment

- **C1–C4.** Record a SHA-256 hash of every consumed RGB frame and depth packet, plus full logs. Store no frames and no per-decision snapshots. This was qualified by bitwise replay of 13,278 RGB frames and depth packets.
- **C0.** The first corrected C0 gate episode was to be kept in full, and replayed bitwise before later C0 runs switched to hashes. A mismatch is a C0 validity problem, not a storage problem: stop.
- **Regeneration.** Anything that verified replay can regenerate is regenerated when needed: videos, diagnostics and later audits.
- **Environment pin.** Recorded in `docs/go2_navigation_capability_environment_pin_2026-09-26.json`. No silent changes. After an unavoidable change, record it and re-verify replay on one pilot mission before continuing.
- **Genesis render caches.** After any physical restoration or scene-time rewind, invalidate the visualizer caches with the existing patch before rendering. Otherwise images show stale geometry.
- **JSON output.** Every writer goes through the NumPy-safe converter, with read-back validation.

### 5.7 Budget and resources

- **Wall time:** the cap for this brief is 160 hours. After each cohort, re-project from the measured wall time per mission for each controller, with 15% contingency. From the orientation stop (28 September, 10:00 BST, 70.19 h) onwards, wall time counts only while jobs run; concurrent jobs count once. The ledger is `wall_active_ledger_2026-09-28.jsonl` under the programme artifact root. The frozen run owner still enforces its calendar 160-h closeout, about 2 October 03:47 BST.
- **Workspace headroom:** keep every large output and temporary file on RecoveryStorage, including video render intermediates.
- **C4 training,** if it's ever needed: at most 12 GPU-hours.
- **VRAM:** device capacity minus a 2-GiB reserve, counting all users.
- **Filesystem reserves:** at least 12 GiB free on RecoveryStorage and 4 GiB on the workspace. Recheck before each cohort.
- **Concurrency:** output-preserving concurrency is allowed once equivalence checks pass; they did in the pilot.
- **Hardware:** AMD Radeon AI PRO R9700 GPU, 16-core/32-thread CPU, ROCm.

### 5.8 Working practice

- **Keep working in-session.** After launching a long job, wait on it in-session with brief periodic status checks, then carry on. Milestone updates are progress notes, not stopping points.
- **Report to Andrew at:**
  - the gate result;
  - the capability results, with the videos;
  - the final report and the E1 proposal.
- **Controller failures** (pose loss, timeout) are results to record and classify, not reasons to stop. Batch wrappers must not stop on them.
- **Stop only for:**
  - the version cap;
  - a budget or storage limit;
  - a safety violation;
  - a second consecutive containment failure;
  - a C0 validity problem;
  - a discrepancy between this handoff and the records;
  - anything needing approval.

---

## 6. Capability qualification, videos and deliverables

This follows sections 6, 7 and 9 of the brief, with the amendments above.

**Qualification**

1. On the frozen harness that passed the gate, run C1–C4 on the 20 validation episodes, and C0 on the 10 with the lowest IDs.
2. Report every metric in section 5.4 for each controller. Use per-maze values as the unit, with raw counts and 95% maze-cluster bootstrap intervals.
3. Apply the capability criterion, fixed before runs: round-trip success of at least 80% (16 of 20) with zero disallowed contacts. Classify by point estimate, and show the intervals.
4. If C3 or C4 is not capable while C0 and C1 are, report the mechanisms and propose at most one targeted intervention. Don't implement it.

**Videos**

Make one video per controller type: C1–C4, with C0 optional.

- **Episode selection,** fixed before viewing any output:
  - the lowest-ID validation episode that every included controller completed;
  - if there is none, each controller's lowest-ID success;
  - if a controller has no success, its lowest-ID episode, labelled as a failure.
- **Rendering.**
  - Render from a deterministic replay of the logged episode, never during evaluation.
  - Verify that the replay reproduces the logged trajectory, decisions and consumed-frame hashes. If it doesn't, don't publish that video, and report why.
  - Use the prepared renderer, bound to the passing harness.
  - Don't change simulator or controller code for videos.
- **Panels:**
  - **Egocentric:** the frames the controller actually consumed, at their native cadence. Label any smoother re-render as such.
  - **Exocentric:** a chase camera rendered in a separate pass.
  - **Minimap:** the true maze with walls, home, beacon, pose, heading, and the trajectory coloured by leg. Optionally add the observed map, and the candidate endpoints with the chosen one highlighted.
  - **HUD:** controller, harness version, maze and episode, simulated time, mission phase, current action, a stall indicator and arrival status.
- **Success rate.** Each video states that controller's validation success rate.
- **Format:** 1920×1080 at 30 fps in simulated time, H.264 MP4 (yuv420p). For missions longer than two minutes, add a clearly labelled 4× accelerated cut.
- **Extras:**
  - a side-by-side composite when controllers share an episode;
  - a keyframe contact sheet per video;
  - a metadata JSON binding the episode, harness, model and replay-verification hashes.

**Deliverables, then stop**

1. **Capability report.** Use a new name, for example `docs/go2_navigation_capability_qualification_result_<date>.md`, because `go2_navigation_capability_result_2026-09-25.md` is the pilot report. Lead with the per-controller capability table, then per-maze results and the failure taxonomy.
2. **Harness change log:** every version with its diff, charge status and screen results.
3. **Media:** the videos, composite, contact sheets and metadata.
4. **E1 proposal:** a schedule based on measured throughput, the seeds, multi-seed C4 training, and whether to add a real-time latency mode.

---

## 7. How the harness got here

| Version | Change | Charged | C1 result |
|---|---|---|---|
| v0 | Stack as deployed for the V4.2 audit source runs | Base | Pilot on 00/0: C0–C3 completed round trips. C4 failed beacon verification, which exposed the start-reference defect. |
| `v0_task_c1` (`13ebc8e2`) | Settled-start task transform and corrected arrival reader | No: correctness | 3/10: five startup failures, one timeout, one pose loss |
| `v0_startup_c2` (`9ee22d80`) | Map bound from the 5.5 × 5.5 m generator envelope (±8 m storage, ±7.9 m validation); startup floor fallback to the downward camera; bounded startup rotation | Yes: containment failed because two routing functions kept stale grid offsets | 3/10, diagnostic only |
| `grid_c3` (`f904a9b5`) | Grid-index correction across all call sites | No: correctness, all five containment checks exact | 3/10: three pose losses, four timeouts |
| `paired_floor_v1` (`2469bb79`) | Floor height taken from the paired-camera plane; the fallback path had set it 23–29 cm too high | Yes: normal-start code changed, though outputs were identical | 5/10; beacon reached 9/10 |
| `exhausted_view_v2` | Retire an unsuccessful recovery reference after 1 s of continuously accepted, aligned poses | Yes | 8/10; beacon reached 10/10; no pose losses |
| `v3_live_turn` (`8c825807`) | Release the visual turn-memory latch when its chosen direction becomes ineligible (turn-memory change) | Yes | 8/10; beacon reached 10/10; no pose losses. The `v3c1_live_turn` implementation erratum (`4f503906`) re-bound the deployed memory class after a zero-decision startup failure and was not charged. |
| `completed_support_v4` (`7da82b23`) | Recovery uses the tracker's actual selected-feature count, including sparse corner completion, instead of only the original strong-corner subset | Yes (fifth charged change; sixth version counting V0, which is the cap) | 10/10 on first episodes; 10/10 on second episodes |

What the iterations showed:

- **Random starts changed the task.** About half of starts face a wall with too little floor in view. Most mazes extended past the old ±4.9-m map bound.
- **Failures came from the control logic, not motion prediction.** After startup, most failures came from the recovery and turn-selection logic: latched overrides, holding at headings already reached, and reversals mid-turn.
- **Time was never the binding constraint.** Holds and stalls were.

---

## 8. Research background

**Model lineage.**
- A frozen V-JEPA 2.1 encoder, with native RGB context at −1000, −500 and 0 ms.
- An action-conditioned predictor of dense future features, conditioned on the horizon.
- A supervised readout from current and future features to XY and yaw motion.
- Candidates are six command tapes, scored at eight horizons from 100 to 800 ms, with a 300-ms committed prefix.
- One inference pass takes about 2.55 s, with physics paused.

**Findings before this phase** (`LeWMQuad_JEPA_World_Model_Progress_Report_2026-09-23.md`):
- Action-conditioned feature prediction is real. At 700 ms, feature MSE is 0.234, against 0.312 without future actions and 0.402 for persistence.
- Decoding motion from those features transfers poorly and underestimates progress. Command history predicts short-horizon ego-motion to about 1 cm.
- In earlier navigation cohorts, simpler controllers matched or beat the JEPA controller.

**Decision-headroom audit V4.2** (`docs/go2_decision_headroom_audit_result_2026-09-25.md`, with its memo alongside):
- The result was inconclusive. Point estimates favoured command history over both learned heads.
- Even with true motion, the harness excluded 40–56% of physically safe movement candidates.
- The branch panel is retained for future representation work (`go2_decision_headroom_phase2_v4_attempt_001/branch_panel_v42.json`). Its layout roles: 00–01 fit-eligible, 02–03 selection, 04–07 evaluation-only.

**Why the direction changed.** The question became when JEPA representations carry decision-relevant information. Answering it first needs a harness that can navigate.

**Hypotheses:**
- **H1 (competence):** a JEPA world model in a standard stack completes beacon retrieval and return in unseen static mazes.
- **H2 (boundary):** in static mazes, it doesn't beat kinematic prediction.
- **H3 (niche):** where outcomes depend on scene dynamics, a JEPA model that predicts consequences improves success or safety at matched latency.
- **H4 (representation):** JEPA-trained predictors match or beat supervised predictors trained on the same data.

---

## 9. Lessons for whoever continues

- **Check a correctness fix's output, not just its path.** A correctness fix needs a contract on its output against ground truth, not just proof that the code path runs. The fallback floor initialisation passed "reaches navigation" for two screens while it was 23–29 cm wrong.
- **Test every call site.** A convention change needs a test that covers every call site: two routing functions kept the old grid offsets.
- **Use containment.** Checking against earlier recordings is cheap and catches implementation errors. Use it for every correctness fix.
- **Diagnose before changing anything.** Use logs and exact replay; failures have reproduced exactly under replay.
- **Don't stop at milestones.** Stopping leaves the machine idle for hours; wait on jobs in-session.

---

## 10. Files

**Direction and rules**
- `docs/CURRENT_RESEARCH_BRIEF.md`: pointer to the current brief. Update it to reference this handoff.
- `docs/go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md`: the authoritative brief.

**Capability programme** (all in `docs/`, in order):
- The pre-registration record committed before the pilot **(confirm path)**.
- `go2_navigation_capability_result_2026-09-25.md`: pilot and throughput (`8872d9fc`).
- `go2_navigation_capability_reference_retention_erratum_2026-09-26.md` (`c21881b3`), with `go2_navigation_capability_sensor_regeneration_2026-09-26.json` and `go2_navigation_capability_resume_budget_2026-09-26.json`.
- `go2_navigation_capability_environment_pin_2026-09-26.json`.
- `go2_navigation_capability_corrected_screen_progress_2026-09-26.md`.
- `go2_navigation_capability_startup_erratum_2026-09-26.md` and `go2_navigation_capability_harness_v0_startup_c2_final_2026-09-26.json`.
- `go2_navigation_capability_grid_c3_progress_2026-09-26.md` and `go2_navigation_capability_grid_c3_failure_diagnosis_2026-09-26.md`.
- `go2_navigation_capability_paired_floor_v1_result_2026-09-27.md` and `go2_navigation_capability_paired_floor_leg_diagnosis_2026-09-27.md`.
- `go2_navigation_capability_exhausted_view_v2_result_2026-09-27.md`.
- The `completed_support_v4` change and screen records **(confirm names)**, and the second-episode check requirement (`576e782d`).
- `go2_navigation_capability_completed_support_v4_oracle_stop_2026-09-28.md` (`b5ad9f7d`).

**Background**
- `LeWMQuad_JEPA_World_Model_Progress_Report_2026-09-23.md` **(confirm location)**.
- `docs/go2_decision_headroom_audit_result_2026-09-25.md`, `docs/go2_decision_headroom_decision_memo_2026-09-25.md`, `docs/go2_decision_headroom_protocol_v42_2026-09-23.md`.

**Artifacts** (under the artifact root)
- Capability programme: `go2_navigation_capability_v1_attempt_001/`. Contains cohorts, replay receipts and videos, including the pipeline test.
- V4.2 audit: `go2_decision_headroom_phase2_v4_attempt_001/`, including the branch panel. Don't delete or move it without Andrew's say-so.

**Scripts seen in reports**
- `scripts/check_go2_capability_episode_references_2026_09_26.py`
- `scripts/continue_go2_capability_correctness_screen_development.py`

---

## 11. Later work (context only; not authorised)

- **E1 benchmark.** The 60 sealed test mazes, with paired episodes, at least three training seeds for the learned components, and per-maze analysis. Possible additions:
  - representation comparisons on the V4.2 branch panel: DINO at matched resolution, a small end-to-end JEPA, and a supervised baseline;
  - a real-time latency mode alongside paused physics.
- **E2 moving obstacles.** Needs:
  - a readout that predicts obstacle consequences;
  - obstacle-aware baselines, for example constant-velocity tracking;
  - memory that doesn't store moving obstacles as walls.
- **E3 decision audits.** Reuse the qualified branch tooling to explain E1 and E2 outcomes.
- **E4 hardware subset.** Needs decisions on sensor parity (the simulation currently uses simulated depth and an ideal gyro) and on real-time planning.

---

## 12. Kickoff prompt for a fresh agent

```
You're taking over the LeWMQuad-v3 navigation capability programme from a previous agent session.

1. Read, in order:
   - docs/go2_navigation_capability_handoff_2026-09-28.md: the goal, current state, rules in force and immediate task. It overrides older instructions where they conflict.
   - docs/go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md: the authoritative brief, as amended by the handoff.
   - docs/go2_navigation_capability_completed_support_v4_oracle_stop_2026-09-28.md: the C0 stop.
   - The pre-registration, the version ledger, the latest budget re-projection and the other records in the handoff's file index, as needed.

2. Write a short orientation note, docs/go2_navigation_capability_orientation_<date>.md, confirming from the records:
   - the current harness version and its hash;
   - the number of versions charged against the six-version cap;
   - wall hours used against the 160-hour cap, and usable storage;
   - the C3 head and the C4 source;
   - which C0 gate episodes are complete, and how the fifth attempt ended;
   - whether the first C0 gate episode passed its full-frame replay check.
   Resolve every item the handoff marks "confirm". Update docs/CURRENT_RESEARCH_BRIEF.md to point to the handoff. Post the note as a milestone and continue without waiting for a reply. If anything on disk contradicts the handoff, stop and report the discrepancy before running anything.

3. Carry out the handoff's immediate task: record the C0 fidelity-check erratum, resolve the fifth attempt as specified, and resume the C0 gate. Continue through capability qualification, the example videos, the capability report and the E1 proposal, then stop.

Keep working in this session. Wait on long jobs in-session with brief status checks, treat milestone updates as progress notes, and stop only for the stop conditions in the handoff.
```
