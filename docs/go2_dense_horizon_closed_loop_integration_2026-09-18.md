# Dense horizon predictor: first closed-loop maze integration

The completed horizon-conditioned V-JEPA predictor now supplies all six candidate
action forecasts at 100–800 ms to the existing observed-map navigation controller.
The first completed pilot executed commands in native Go2 physics. It establishes
integration, not goal-reaching or independent maze generalisation.

## Completed full-maze comparison

Both full assignments and their physical readers completed with exit code 0.
The world-model planner failed; the matched reactive controller completed the
goal-and-return mission on the same exposed sparse-corner layout 0.

| Quantity | Dense action-conditioned planner | Reactive feedback |
|---|---:|---:|
| Physically verified goal arrivals | 0 | 1 |
| Physically verified return arrivals | 0 | 1 |
| Disallowed contact samples | 0 | 0 |
| Pipeline faults | 0 | 0 |
| Simulated execution | 480.32 s (budget exhausted) | 249.22 s (round trip) |
| Execution wall time before persistence | 3896.88 s | 1991.17 s |
| Selected hold plans / all plans | 1102 / 1198 | 1 / 613 |
| Longest consecutive zero request | 437.22 s | 1.90 s |
| Median tracking position error | 1.36 mm | 4.17 mm |

The reactive goal arrival at frame 1708 and return arrival at frame 2492 both
passed the native 4-cm distance and one-second quiet-dwell checks. Maximum native
distance during the respective dwells was 11.65 mm and 17.67 mm. All requested
commands during each dwell were zero. These are verified simulation results,
not real-time or hardware qualification, and concern one exposed maze.

The baseline computes the same dense forecasts but does not use their values
to select commands. On its 595 matched executed 700-ms windows, dense XY/yaw
RMSE was 69.12 mm / 10.25 degrees, versus 8.08 mm / 0.75 degrees for command
history. On 379 ordinary translation windows the respective XY errors were
84.63 mm and 8.72 mm. These errors are measured along the reactive trajectory;
they do not describe alternative-policy navigation. The comparison shows that
the shared observed mapping/exploration system can navigate this maze, while
this forecast-driven controller package does not. It does not isolate JEPA
training or prove a benefit from predictive planning.

Authoritative results are the `dense_navigation_readout.json` files in the
action and reactive full-run roots listed below. The reactive recording retains
RGB/physics/commands and depth hashes, without raw depth arrays.

## Completed evidence

| Quantity | Exposed-maze pilot |
|---|---:|
| Simulation duration | 16.1 s |
| Acquired camera pairs / registered poses | 161 / 161 |
| Online dense-model candidate evaluations | 38 |
| 20-ms command steps / nonzero requested steps | 805 / 655 |
| Disallowed contact samples | 0 |
| Pipeline faults | 0 |
| Verified goal / return arrivals | 0 / 0 |
| Median / maximum tracking position error | 1.67 / 2.70 mm |
| Median model inference time | 2,448 ms |
| Execution wall time, excluding final recording persistence | 141.7 s |

The controller selected 34 left turns, three holds and one right turn. Most of
the pilot covered the initial visual survey, followed by observed-floor frontier
routing after a weak-view interruption. This short duration did not establish
translational navigation or a successful mission.

On 35 overlapping executed windows whose requested command tapes matched the
selected forecast through 700 ms:

| Forecast | XY RMSE (mm) | Yaw RMSE (degrees) |
|---|---:|---:|
| Dense predicted features + frozen motion readout | 37.64 | 10.00 |
| Existing command-history reference | 5.07 | 0.69 |
| Nominal command integration | 9.03 | 1.41 |

These are selected turn/hold windows from one exposed layout, not independent
trials or alternative-policy navigation results. The frozen physical readout was
trained only at 500 ms; its 700-ms use is temporal transfer. The poor physical
forecasting result is retained. It does not by itself establish the controller's
full mission outcome or identify whether predicted features or the readout cause
the error.

A subsequent CPU-only readout checked every 20-ms applied command against the
prospective tape. No applied-tape mismatch occurred among the request-matched
windows. At the head's trained **500-ms horizon**, dense XY RMSE was **37.16 mm**
and yaw RMSE **7.23 degrees**, versus **4.21 mm / 0.57 degrees** for the existing
command-history reference. Thus the degradation is not confined to extrapolating
the readout to 700 ms. The dense forecasts have a mean forward-position error
of 34.05 mm at 500 ms. Full horizon results are in
`go2_dense_horizon_pilot_motion_diagnostic_2026-09-18.json`.

Decoding the **actual future image**, using the same frozen head on all 35 matched
500-ms pilot windows, was also inaccurate:

| Features supplied to the frozen motion head | XY RMSE (mm) | Yaw RMSE (degrees) |
|---|---:|---:|
| Dense model's predicted future | 37.16 | 7.23 |
| Actual observed future (offline oracle) | 44.30 | 8.32 |
| Current image repeated as future | 15.22 | 9.99 |

This establishes a transfer failure of this particular physical readout even
with true future inputs. It does not establish that motion information is absent
from V-JEPA features, nor isolate how much additional error the predictor causes.
The oracle is unavailable to the online controller. The CPU-only diagnostic
encoded 70 images in about 100 seconds on cores 4–7 while the unchanged full
mission used cores 8–15/24–31 and the GPU. A retained training example cross-check
against prior GPU decoding differed by less than 1e-7 in the output channels;
the large pilot error is not explained by CPU execution.

A follow-up source check found the same token layer normalisation and spatial
pooling in readout training and native inference. There is a small yaw-definition
difference: training uses yaw from the relative 3-D rotation, while the pilot
diagnostic uses the difference in world headings. Recomputing all 35 matched
500-ms pilot windows with the training definition gives 7.195 degrees dense yaw
RMSE, versus 7.229 degrees with world-heading differences. The definitions differ
by only 0.056 degrees RMS (0.106 maximum) on these windows, so this convention
difference does not explain the large error. This check changes no model or
controller and does not establish agreement for arbitrary body tilts.

Absolute bias is not the whole planning problem: the controller scores change
over **300–700 ms**. On those same 35 executed windows, increment errors were
**14.62 mm / 6.46 degrees** for the dense model, versus **3.34 mm / 0.60 degrees**
for the command-history reference. Shared position bias largely cancels, but
substantial increment error remains. This still evaluates selected executed
windows, not counterfactual action ranking or full navigation success. See
`go2_dense_horizon_pilot_commit_interval_2026-09-18.json` and the pilot's
`observed_future_motion_probe/result.json`.

A fixed spatial-correlation readout experiment completed using only the original
training recordings. It worsened action-conditioned planning-interval error and
is not selected for navigation. It did not change the active full mission.
Its rationale and complete comparison are in
`go2_correlation_motion_readout_2026-09-18.md`.

### Completed physical branch diagnostic

`scripts/collect_go2_dense_stall_turn_branches_development.py --prepare`
completed and fixed frame 444: the first saved planning frame with a full second
of zero requested commands in the final sustained stall. Its frozen plan is
`.generated/navigation_development_artifacts_v1/go2_dense_stall_turn_branches_v1_attempt_001/plan.json`.
After the reactive comparison, three separate processes can run the script with
`--action hold`, `--action left_turn`, and `--action right_turn`. Each replays the
original 2220 command steps, checks the physical prefix and exact three-image
camera context, then executes the saved 800-ms candidate tape. Existing native
initialization, gains, sensors and command limiter are reused. RGB, physics and
depth hashes are retained; raw depth arrays are omitted. A replay mismatch is
retained as a diagnostic failure, not silently accepted as the original state.

`scripts/evaluate_go2_dense_stall_turn_branches_development.py` then compares
the frozen action/blind forecasts with the actual branch images, and compares
motion decoded from predicted versus actual future features at 300/500/700/800
ms and over the 300–700 ms commitment interval. Forecasts are computed before
future images load. This is one post hoc counterfactual diagnostic, with no fit,
no controller changes, no new maze, and no navigation-success claim. It will
help distinguish inaccurate visual branch prediction from poor physical decoding.
Preparation checked syntax, fixed-context selection and recorded command-history
shape. After both navigation owners and readers completed, one sequential
coordinator completed all three branches and CPU evaluation with exit code 0.
All three reproduced the original context poses and RGB exactly and had no
contact. Actual left rotation over 300–700 ms was +9.17 degrees, but the motion
head decoded -1.38 degrees even from actual future images. Correct-action latent
forecasts nevertheless beat both wrong-action forecasts for all three branches
at 500, 700 and 800 ms. Full results and limits are in
`go2_dense_stall_turn_branches_2026-09-18.md`. The evidence points to physical
readout transfer as the next intervention, without establishing predictor or
JEPA navigation superiority.

A subsequent fixed geometric feature/depth readout recovered actual-future yaw
well at this context (0.41-degree RMSE), but predicted-feature translation error
was 79.53 mm. It is not selected for navigation. The synthetic limitation,
matched MLP comparison and output-subspace diagnostic are recorded in
`go2_dense_geometric_readout_results_2026-09-18.md`. Neither decoder replacement
nor an assumed output-width bottleneck alone resolves the observed transfer
problem. Broader training-view collection has completed and matched old/mixed
readout training is running, as recorded in
`go2_full_heading_readout_experiment_2026-09-18.md`.

## Treatment and limitations

- Frozen pretrained encoder, completed action-conditioned horizon predictor and
  frozen motion head; no fitting on this maze or pilot.
- Actual full-resolution RGB packets at -1000/-500/0 ms, plus the causal applied
  command history. Requested candidate tapes pass through the platform limiter
  before conditioning the predictor. Equal applied prefixes share one forecast.
- Existing observed mapping, exploration, local turn memory, arrival checks,
  backtracking, dispatch checks and physical command limiter remain in use.
- Synchronous untimed simulation: physics pauses while acquisition, perception,
  mapping and planning complete. Host durations are recorded but not charged to
  simulation deadlines. Mapping for the acquired frame finishes before planning.
  This treatment is not real-time qualified and must also apply to future controls.
- Contact cost is disabled; the model does not predict collision probability.
  Shared observed-geometry checks remain active.
- Existing synthetic 2-mm depth noise and ideal body gyro. No hardware validation.
- Exposed sparse-corner replication layout 0; the four prospective dense-model
  mazes remain unexecuted.

The native model adapter reproduced both action and no-future-action retained
training-branch motion outputs exactly, with exact equality before the command
branches diverge. See `go2_dense_horizon_navigation_check_2026-09-18.json`.

## Attempts and current next step

All artifacts are under
`.generated/navigation_development_artifacts_v1/` in the workspace:

1. `go2_dense_horizon_untimed_exposed_maze_pilot_v1_attempt_001`: preserved
   initialization failure, zero acquired navigation frames or command steps.
   The legacy stage profiler assumed `encode_history` existed on every model.
   Its fix instruments only implemented stages; existing old-model stages remain
   instrumented. No model or scientific parameter changed for the relaunch.
2. `go2_dense_horizon_untimed_exposed_maze_pilot_v1_attempt_002`: completed,
   owner exit 0. Full RGB, depth, physics, command, model-call and planning records
   retained. Physical results and matched-window diagnostics are in
   `dense_navigation_readout.json`.
3. `go2_dense_horizon_untimed_exposed_maze_full_v1_attempt_001`: full exposed-maze
   round-trip attempt exhausted its 4800-navigation-tick budget without an
   observed arrival. The terminal result records 4804 camera frames, 1198 model
   calls, 480.32 simulated seconds, no disallowed contact and 3896.88 seconds
   of execution wall time before final persistence. This is a navigation failure.
   Owner exit and the queued physical reader both completed with exit code 0.
   The physical reader verified zero arrivals and zero disallowed contact samples;
   all 4804 camera poses were registered, with median position error 1.36 mm and
   maximum 2.70 mm. There were no pipeline faults. Full raw depth is retained.

The recorded decisions identify a sustained forecast-scoring stall. All 1102
hold selections already ranked hold highest before memory filtering. Of these,
1101 were view-turn decisions, and every one had at least one turn passing the
recorded full-reserve clearance check. The last non-hold plan was frame 424;
zero commands continued for 437.22 seconds through the recording end. In the
last plan, the requested view was still 0.878 radians away, but hold scored above
both turns. The decoded 300–700 ms yaw increments were -0.176 degrees for hold,
-2.581 degrees for left turn, and -5.559 degrees for right turn, despite a
positive (leftward) view error of 50.32 degrees. Thus neither turn was predicted
to reduce that view error. Dispatch vetoes alone therefore do not explain this stall. These
records do not prove that an alternative policy would navigate successfully or
isolate the encoder, predictor and motion readout. The queued reactive run is
the next actual closed-loop comparison. Exact counts and source hashes are in
`go2_dense_horizon_full_maze_stall_diagnostic_2026-09-18.json`.

The completed physical/forecast readout is `dense_navigation_readout.json` in
the full-run root. On 1196 overlapping matched 700-ms executed windows, dense
XY/yaw RMSE was 25.13 mm / 2.69 degrees, versus 4.18 mm / 0.46 degrees for the
command-history reference. The aggregate is dominated by 1101 hold windows;
on the 22 translation windows, dense XY RMSE was 61.18 mm versus 12.99 mm for
command history. These are selected executed windows, not alternate navigation
outcomes. Median model inference was 2510 ms. The reader command below has
already completed; do not rerun it over the retained result.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 4-7 .generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/read_go2_dense_horizon_navigation_development.py --root-name go2_dense_horizon_untimed_exposed_maze_full_v1_attempt_001
```

The broader goal remains incomplete: independent dense-model maze missions,
matched online controls, separation of JEPA training and persistent-memory
contributions, and realistic timing/sensing and bounded hardware evidence remain
outstanding. This exposed mission precedes the four-maze prospective comparison.

## Prepared matched controls

The common untimed runner now accepts `--arm action`, `--arm no_future_action`,
`--arm command_history` and `--arm reactive_feedback`. The default action run
completed unchanged. The queued reactive run and its physical reader have also
completed; the verified result is reported above.

All arms retain the same camera/actuator cadence, native context warmup,
same-frame mapping barrier, observed map, physical arrival rules and exposed
layout. The command-history arm uses the existing frozen command-only forecast
with the same predictive consumers. The reactive arm uses the established
current-waypoint/current-clearance selector and bypasses forecast-dependent
selection and recovery rules. Neural forecasts are still computed for workload
control but are not used to select commands in those two controls. Thus the
reactive comparison is a controller-package comparison, not isolation of a
single cost term. The action-blind arm predicts once per horizon and broadcasts
to candidates; computation costs need not match in this untimed treatment.

Launch a full exposed control only after checking current resources and terminal
recording persistence, for example with the same environment and CPU affinity as
the active action run plus `--full-mission --arm reactive_feedback`. The reader
`scripts/read_go2_dense_horizon_navigation_development.py` records which forecast
source actually controlled each arm. Fresh-maze comparison remains a later step.

The completed `reactive_feedback` full round-trip assignment used the same
exposed layout 0. Its owner PID 91941 and session 21463, including the subsequent
physical reader, exited successfully after the action recording was saved. Root:
`go2_dense_horizon_untimed_reactive_feedback_exposed_maze_full_v1_attempt_001`.
Preserve this completed assignment; no repeat to obtain success is scheduled.

This baseline uses `--depth-retention rgb_only`: all RGB, physics, commands,
perception records, camera identities, depth hashes and noise recipes are
retained, including on failure. Raw depth arrays are not archived. This change
applies only after physical execution and does not change any live sensor input,
map, model, command or arrival rule. Direct raw-depth replay is unavailable
without regeneration; bitwise regeneration is not claimed. The existing action
run remains a full-depth reference. The recording writer was checked on a real
pilot frame for exact retained RGB pixels and hashes; no existing artifacts were
deleted. The reader marks the new recording schema explicitly. Two GiB of
headroom is required before launching this lean-recording full run.
