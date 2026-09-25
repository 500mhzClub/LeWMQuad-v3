# LeWMQuad-v3 agent brief: navigation testbed and controller capability

**Prepared:** 25 September 2026
**Approver:** Andrew Knowles
**Supersedes:** the decision-headroom programme. The V4.2 audit is closed and complete; the memo's recommended study-design revision is not pursued.
**Authorised scope:** sections 4–9 of this brief. Everything else requires explicit approval.

---

## 1. Research direction

This is a JEPA-based PhD about useful feature representations for general robotics. The Go2 navigation stack is the testbed, not the thesis.

The new phase builds a validated navigation testbed and uses it to compare controller types on one mission: find and reach a beacon in an unseen maze, then return home. A JEPA loss is an acceptable result. An unvalidated testbed is not: every comparison depends on a harness that lets a controller with good motion prediction actually navigate.

Planned sequence (only steps 1 and 2 are authorised by this brief):

1. **Harness validation.** Show the shared stack completes missions when motion prediction is perfect.
2. **Capability qualification.** Establish, per controller type, whether it navigates across unseen mazes, with results and an example video. ← *this brief*
3. **E1 benchmark.** 60 sealed test mazes, paired episodes, multiple training seeds.
4. **E2 moving obstacles.** Needs a readout that predicts obstacle consequences, obstacle-aware baselines, and memory that does not store moving obstacles as walls.
5. **E3 decision audits.** Reuse the qualified branching tools to explain E1 and E2 outcomes.
6. **E4 hardware subset.**

Working hypotheses:

- **H1 (competence).** Integrated into a standard navigation stack, a JEPA world model can complete beacon retrieval and return in unseen static mazes.
- **H2 (boundary).** In static mazes it does not outperform kinematic motion prediction, because ego-motion is determined by the commands sent.
- **H3 (niche).** Where outcomes depend on scene dynamics, a JEPA model that predicts consequences improves success or safety over kinematic and reactive controllers at matched latency.
- **H4 (representation).** JEPA-trained predictors match or beat supervised predictors trained on the same data.

Carried forward from the audit:

- qualified state restoration, rendering and decision replay;
- mazes are the unit of inference, and designs are paired;
- nothing is tuned on evaluation material;
- the current harness excluded physically safe movement even with true motion, much of it through view-seeking restrictions — expect harness work before any controller can be judged;
- training-render provenance of the current predictor and readouts is unverified; carry that caveat into any reporting.

Out of scope now: model-free RL (infeasible at the current simulator speed), dynamic obstacles, sensor changes, a real-time latency mode, hardware, representation comparisons and end-to-end JEPA training.

## 2. Controller types

Every controller runs on the same harness version: same sensors (simulated RGB, depth with the existing noise treatment, ideal gyro, measured visual pose), mapping, memory, routing, candidate bank, safety filter and selector. They differ only in what fills the prediction slot.

| ID | Controller | Prediction slot |
|---|---|---|
| C0 | Oracle (diagnostic only) | True candidate motion from restored physics branches at each decision, using the qualified branch tooling |
| C1 | Command history | Existing fitted command-history motion model |
| C2 | Reactive | None; existing reactive selector |
| C3 | JEPA | Frozen V-JEPA 2.1 encoder, action-conditioned predictor and one motion head |
| C4 | Supervised predictor | Direct regression from the same frozen features and candidate command tape to the same motion targets |

- **C0** is a diagnostic, not a competitor. Its privilege is confined to the prediction slot; it never receives the true map or pose. It is expensive (one branch per candidate at every decision), so use it only for the gate and the validation subset below.
- **C3.** Before any C3 run, pre-declare one head (old-data or maze-data) with a one-line justification from existing evidence. Do not select a head from new runs.
- **C4.** If an existing supervised or direct predictor loads unchanged in the current harness and matches C3's interface, use it. Otherwise train one instance:
  - same training data and data roles as C3's readout, same feature preprocessing, same targets and horizons;
  - parameter count comparable to C3's predictor plus readout;
  - validation on selection-role data only; one seed; at most 12 GPU-hours.

  Record the render-provenance caveat. Multiple seeds belong to E1.

## 3. Definitions (fix before any run)

- **Mission:** start at the home pose, reach the beacon, return home, within a simulated-time budget. Keep the existing 480-s budget unless the generated mazes are larger than the audit family; in that case set one budget from the shortest round-trip path length and apply it to every controller.
- **Beacon retrieval and home arrival:** the existing verified-arrival criteria from the physical readers. Round-trip success requires both.
- **Collision:** any native disallowed contact (ground support excluded, as now). Also report hard-criterion (5-mm) and operating-margin (20-mm) violations on executed trajectories from the native-step articulated evaluation.
- **SPL per leg:** success × shortest path ÷ max(actual path, shortest path), with the shortest path on the true map at an inflation declared before runs.
- **Stall rate:** fraction of planning decisions that select hold, reported by mission phase.
- **Decision latency:** wall time per planning decision (median and 95th percentile), physics paused as now.

## 4. Maze and episode sets

Generate three new sets with the existing generator and its duplicate registry. Fix every seed before any run.

| Set | Mazes | Episodes per maze | Use |
|---|---:|---:|---|
| Dev-tune | 10 | 2 | Harness iteration (first episode of each maze) and the oracle gate (both episodes) |
| Validation | 20 | 2 | Capability qualification only; never used during iteration |
| Test | 60 | — | E1 only. Generate, register and hash; do not run, render or inspect beyond structural checks |

Exclude the eight audit layouts (00–07) from all three sets, and leave their branch-panel roles unchanged.

Each episode draws, from a fixed seed, a start pose and heading (home) and a beacon location, both in free space with clearance. The beacon must not be visible from the start pose, and the start–beacon geodesic distance must exceed a declared minimum. Every controller runs the same episodes, so all comparisons are paired.

## 5. Harness validation

Goal: a harness in which a controller with perfect motion reliably completes missions. If it can't, the harness is wrong by definition.

**Gate:** C0 completes at least 19 of the 20 dev-tune episodes, with zero disallowed contacts and zero hard-criterion violations.

**Iteration protocol.** C1 stands in for C0 during iteration: it is far cheaper, its motion error is about a centimetre, and the audit found its filter behaviour close to that under true motion.

1. Run C1 on the first episode of each dev-tune maze with the current stack (harness v0).
2. Classify every failure and stall episode by mechanism, using the audit's binding-rule accounting: no eligible movement (observation/view restriction, memory clearance, stopping projection), blocked recovery, pose loss, invalid or unreachable target, contact, budget exhaustion despite progress.
3. Pick the dominant mechanism and make one change to the shared harness that addresses it. Record the rationale, diff and expected effect.
4. Freeze and hash the new version, then re-run step 1.
5. When C1 completes at least 9 of 10, run the C0 gate on all 20 dev-tune episodes. If the gate fails, classify C0's failures and continue from step 3.
6. Stop after six versions if the gate has not passed; report the failure taxonomy and do not lower the gate.

**Allowed changes**, applied identically to every controller: mapping and memory marking, observation and view-seeking requirements, eligibility and clearance rules, recovery behaviour, stopping logic, selector tie-breaks.

**Not allowed:** privileged information in the deployed harness; per-controller special cases; changes to learned models, sensors or the candidate bank; any use of validation or test mazes.

**Safety:** the gate may become less conservative, never less safe. Any disallowed contact or hard-criterion violation in a version's runs disqualifies that version.

## 6. Capability qualification

With the frozen harness version that passed the gate:

1. Run C1–C4 on all 40 validation episodes, and C0 on the 10 validation episodes with the lowest IDs.
2. Report per controller:
   - beacon, home and round-trip success;
   - SPL per leg;
   - time to beacon and home (median, IQR);
   - disallowed contacts and margin violations;
   - stall rate;
   - decision latency and wall time per episode;
   - a failure taxonomy using the section 5 mechanisms.
3. Use per-maze averages as the unit, with raw counts and 95% maze-cluster bootstrap intervals. Label everything as capability qualification, not paper results.
4. **Capability criterion**, fixed before runs: round-trip success of at least 80% on validation, with zero disallowed contacts. Classification uses the point estimate; report intervals so borderline cases are visible.
5. If C3 or C4 is not capable while C0 and C1 are, report the mechanisms and propose at most one targeted intervention for approval. Do not implement it.

## 7. Example videos

Produce one example video per controller type (C1–C4; C0 optional), showing a successful mission.

**Episode selection**, fixed before viewing any output:

1. the lowest-ID validation episode that every included controller completed;
2. if none exists, each controller's lowest-ID successful validation episode;
3. if a controller has no success, its lowest-ID episode, labelled clearly as a failure example.

Each video states the controller's validation success rate, so one run is not mistaken for typical performance.

**Production.** Render from a deterministic replay of the logged episode, never during evaluation, so extra cameras cannot perturb the controller.

- Verify the replay reproduces the logged pose trajectory within the existing restoration tolerances (1 mm, 0.1°) and makes identical decisions. If it doesn't, don't publish that video; report why.
- Do not change simulator or controller code to make videos.

**Frame content:**

- **Egocentric panel:** the frames the controller actually received, shown at their native cadence (held between frames). A smoother 30-fps re-render may be added only if labelled as such.
- **Exocentric panel:** a third-person chase camera at a fixed offset behind and above the robot, smoothed, rendered in a separate pass that the primary camera cannot see.
- **Minimap:** top-down true maze with walls, home, beacon, robot pose and heading, and the trajectory coloured by leg (outbound, return). Optional layers: the robot's observed map, and the candidate endpoints predicted by the controller's motion source, with the chosen candidate highlighted.
- **HUD:** controller, harness version, maze and episode IDs, simulated time, mission phase, current action, stall indicator, beacon and home status.

**Format:** 1920×1080, 30 fps in simulated time, H.264 MP4 (yuv420p). For missions longer than two minutes, add a 4× accelerated cut with a visible time-scale label.

Also produce:

- a side-by-side composite of all controllers when they share an episode;
- a keyframe contact sheet (PNG) per video;
- a metadata JSON binding episode, harness, model and replay-verification hashes.

Use system ffmpeg or imageio-ffmpeg.

## 8. Throughput, budget and discipline

**Throughput pilot first.** After freezing harness v0, run one dev-tune episode per controller and measure wall-seconds per simulated second. Then test two to four concurrent episodes per GPU.

Output-preserving optimisations are allowed — per-frame feature caching, batching, concurrent episodes, additional GPUs — but only after verifying on recorded episodes that decisions and trajectories are identical to the unoptimised path on the same device. Record the device and software identity for every episode.

**Budget caps** (stop and report before exceeding any):

- harness iteration: at most six versions;
- all runs and videos in this brief: at most 120 wall-hours. If the pilot projects more, cut validation to one episode per maze before cutting mazes; if still over, stop and report;
- C4 training: at most 12 GPU-hours;
- VRAM: per device, capacity minus a 2-GiB reserve, counting all users;
- filesystem reserves: RecoveryStorage at least 12 GiB free, workspace at least 4 GiB.

**Discipline carried from V4.2:**

- fresh output roots;
- hashes for every maze, episode, harness version, model and configuration;
- every failure preserved, nothing overwritten;
- the JSON output converter at every writer;
- one frozen configuration per run;
- no silent retries.

## 9. Deliverables, then stop

1. Maze and episode registry (dev-tune, validation, sealed test) with hashes.
2. Harness change log with each version's diff and dev-tune results, and the frozen passing version — or the stop report.
3. Throughput pilot results and equivalence checks.
4. Capability report, `go2_navigation_capability_result_<date>.md`, leading with the per-controller capability table, then per-maze results and the failure taxonomy.
5. The example videos, composite, contact sheets and metadata.
6. A proposal for E1: throughput-based schedule, seeds, multi-seed C4 training, and whether to add a real-time latency mode.

Then stop. This brief does not authorise running the sealed test set, dynamic obstacles, RL, changes to C3's models, sensor changes or hardware work. The brief sets out the new direction and authorises one task. First the shared navigation stack must pass an oracle test. Then the agent qualifies each controller type on 20 unseen mazes and renders an example video for each. Anything beyond that comes back to you for approval.

These defaults are your calls, so check them before sending:

Oracle gate: 19 of 20 development-maze episodes.
Capability criterion: at least 80% round-trip success across 40 validation episodes, with zero collisions.
Caps: at most six harness versions, 120 wall-hours of runs, and 12 GPU-hours if the supervised predictor needs training.
JEPA head: the agent chooses one before any run and can't switch after seeing results.

The videos are rendered from replays of logged runs, checked against the logs, rather than recorded live. That way the extra cameras can't change what the controllers do. Each video also states that controller's overall success rate, so one good run isn't mistaken for typical performance.
