# Planning-cost diagnosis after the latency stress failures

Offline replay reconstructed registered observations and maps from the retained
original JEPA/layout-3 failure. Native state was not read. At frames 300, 700,
and 1100, repeated action-selection calls took median 20.20, 19.77 and 20.80 ms.
Recorded raw neural forecasts were reproduced within 1e-7 absolute / 1e-6
relative tolerance. A warm route-proposer profile took 2.62–3.39 ms. These
component profiles exclude live concurrency and do not reconstruct all frontier
visit/recovery state.

A second replay reproduced all 301 registered poses through frame 300 exactly.
The complete planning call at that observation took median 28.85 ms in
isolation; a pose-evidence read took 0.490 ms. These measurements do not support
assuming that redundant pose checks or the neural forward alone account for
the roughly 100-ms planning service recorded in the live stress run.

Artifacts:
`go2_current_planning_profile_v1_attempt_001`,
`go2_current_routing_profile_v1_attempt_001`, and
`go2_complete_plan_profile_v1_attempt_001`.
Each retains its profiling source, timings and cProfile records. These are
computation diagnostics, not navigation outcomes or hardware timing guarantees.

Next diagnostic is one 800-tick native prefix with the original JEPA controller
and fixed 20-ms added planning delay. Record elapsed and thread-CPU time around
the complete plan, route, action selection, alternative forecasts and pose
reads. Component times are inclusive and must not be summed as independent
work. The whole-plan interval includes result-release waiting. Controller
selection, tracking, deadline, clearance and arrival rules are unchanged;
the mission budget is shortened solely for profiling. This is not a full
navigation/reliability trial and must not be pooled into the 4800-tick studies.
Root: `go2_live_planning_profile_jepa_plus20ms_layout03_800_v1_attempt_001`.
Run alone on the original layout-3 CPU group, then inspect the records after
the owner exits and persistence completes.

## Live diagnostic result and scheduling experiment

The short prefix ended at its declared 800-tick budget with tracking intact,
no arrivals and no live contact report. Across 126 post-survey planning calls,
median action-selection elapsed time was 83.91 ms versus 28.27 ms thread CPU.
Route construction was 13.58 ms elapsed / 10.89 ms CPU. Five pose reads per
plan totalled 3.62 / 3.50 ms. The complete plan including result-release waits
took 239.11 / 41.69 ms. These inclusive timings identify substantial waiting
or descheduling; they do not alone prove which contention mechanism caused it.

Next fixed experiment: full 4800-tick JEPA then supervised missions on the
same exposed layout, retaining 20-ms added planning delay and all original
deadlines/selection/measurement limits, with only the parent process Python
switch interval reduced from 5 ms to 1 ms. Spawned workers retain their default
interval. Keep lightweight live component timings in both runs. This tests
whether allowing the controller threads more frequent interpreter access
restores deadline margin and physical navigation; no benefit is assumed.

Launch with `--control --planning-extra-ms 20 --parent-switch-ms 1`, first
`--arm jepa`, then `--arm supervised_rollout`. Output roots are
`go2_planning_switch1ms_plus20ms_{arm}_noise_2mm_native_layout03_4800_v1_attempt_001`.
Preserve both outcomes and verify physical arrivals/contact plus actual
latency and switch-interval treatment. This is a host scheduling experiment,
not calibrated robot timing or an isolated change to JEPA training.

The JEPA 1-ms-switch run exhausted its full budget with no arrivals, zero
physical contacts and tracking intact. Only 59/1200 plans were on time;
no translating command was requested. The switch-interval receipt confirmed
0.001 s in the parent, with spawned-worker settings unchanged. Actual added
delay was 20 ms on all 1200 jobs. Post-survey action selection remained
83.45 ms elapsed / 28.04 ms thread CPU. The large elapsed/CPU gap was not
resolved. This variant is not adopted as a repair. Physical/treatment/forecast
evaluation completed, and diagnosed depth was retired with all non-depth
records preserved.

The supervised half also completed and was physically evaluated: no arrivals,
zero contacts, tracking intact, and full budget exhausted at 480.92 simulated
seconds. Only 58/1200 plans were on time. The parent switch interval was
confirmed at 1 ms and all 1200 jobs received the intended 20-ms added delay.
Post-survey action selection remained 83.51 ms elapsed / 27.85 ms thread CPU;
whole-plan time including release waits was 236.82 / 40.76 ms. The fixed pair
therefore provides no evidence that reducing the parent switch interval
repairs this timing failure. Both failed outcomes are retained. No simulation
remains running from this experiment.

The next scientific task is to isolate the source of the live elapsed/CPU gap
and test a targeted planning-timing repair while preserving the forecast
execution assumptions. The current measurements do not establish whether
renderer interference, interpreter contention, or another scheduling effect
is responsible. Reliability replication and visual-recovery repair remain
outstanding; these timing diagnostics establish neither a JEPA advantage nor
deployment readiness.

Next bounded diagnosis: repeat the 800-tick JEPA/+20-ms prefix with the
original 5-ms parent switch interval. Alternate elapsed and thread-CPU
cProfile timers on frames 300 through 344 (twelve planning calls). Preserve
normal component timings throughout. This will locate candidate waiting
sites within action selection, with profiling overhead explicitly included
in measured service. It is a diagnostic trajectory, not a reliability or
speed comparison. Root:
`go2_live_planning_call_profile_jepa_plus20ms_layout03_800_v1_attempt_001`.
Run alone on CPUs 8–15,24–31; the previous pair is terminal and roughly
4.5 GiB is free, sufficient for this short recording. No model, sensor,
selection rule, deadline or acceptance threshold changes.

The sparse call-profile trajectory completed (61/200 plans on time, no
arrivals or contacts), but its cProfile data are invalid: they contain
non-planning threads and negative thread-CPU durations. Do not use these
call profiles to attribute the waiting. The failed diagnostic and source
snapshots are preserved. Follow with direct same-thread elapsed/CPU timers
around the model input check, forward, history encoding, transitions,
rollout decoding, scoring and predictive clearance. Use the same 800-tick
configuration, root
`go2_live_planning_stage_profile_jepa_plus20ms_layout03_800_v1_attempt_001`.
This adds measurement only and retains the original 5-ms switch interval.

Direct stage timers succeeded: post-survey history encoding took median
42.47 ms elapsed / 8.29 ms thread CPU, accounting for most of the measured
forward pass (44.99 / 9.74 ms). Predictive clearance took 7.26 / 6.39 ms,
input treatment checks 0.20 / 0.18 ms. Action selection remained 83.23 /
28.13 ms. These are inclusive component medians over completed calls; they
locate the gap but do not identify the underlying scheduling mechanism.
The 800-tick run had 55/200 plans on time, no arrivals or physical contacts.

Targeted optimization: encode the shared four-observation history once for
the six candidates, then retain all six separate action-conditioned latent
rollouts and forecast heads. Apply only in inference when all history tensors
have zero batch stride, proving shared storage; distinct histories retain
the original implementation. There is no cross-observation cache or weight
change. Frozen JEPA and supervised outputs on recorded frames 300/700/1100
passed atol=1e-6, rtol=1e-6 comparisons for every output (maximum absolute
difference 2.39e-6 across all channels). Distinct-history fallback was exact.
Batch-size floating-point changes mean bitwise trajectory identity is not
assumed. Isolated forward time fell from 6.13–6.44 ms to 2.12–2.16 ms.
Evidence: `go2_shared_candidate_history_equivalence_v1_attempt_001`.

Prospective test: full 4800-tick JEPA then supervised missions on the same
layout 3 with the original 5-ms switch interval and +20-ms planning delay.
Keep all selection, acceptance, deadline and physical arrival rules. Record
direct stage timings and counts proving shared encoding was exercised.
Roots: `go2_shared_history_plus20ms_{arm}_noise_2mm_native_layout03_4800_v1_attempt_001`.
Run sequentially on CPUs 8–15,24–31, retiring completed diagnosed depth
between runs. Approximately 4.3 GiB is available before the first launch.
Neither deadline recovery nor navigation success is assumed from the
offline speedup; both require the prospective physical outcomes.

The shared-history JEPA mission exhausted its budget at 480.90 simulated
seconds with no arrivals and tracking intact. Only 129/1200 plans were on
time (original +20-ms JEPA: 56/1200); translating commands totalled 9.20 s.
All 1200 model calls exercised shared encoding, without fallback, and all
1200 planning jobs received exactly the additional 20 ms. The median total
observation-to-plan age remained 306 ms, above the unchanged 300-ms deadline.
Post-survey encoding fell to 14.29 ms elapsed / 2.63 ms CPU, but the complete
model forward still took 41.59 / 4.62 ms, and action selection 81.20 / 22.20 ms.
Thus reduced encoding CPU work did not remove the live waiting; the elapsed
cost also appears outside the encoding sub-call. This supports testing
process isolation rather than assuming further arithmetic optimization will
restore deadlines. It does not prove which simulator operation causes the
waiting. The supervised half remains to be executed with the fixed settings.

The JEPA physical evaluation confirmed zero contacts and no arrivals. Its
diagnosed depth was retired, preserving all non-depth records and source
snapshots. The earlier completed 800-tick component profile's depth was also
retired; its timings and physical evaluation remain. This restores recording
headroom for the supervised half without deleting active failure-replay data.

The shared-history supervised mission also exhausted its full budget without
an arrival, at 480.86 simulated seconds, with tracking intact. It admitted
143/1200 plans; all 1200 calls used shared encoding and all jobs received
exactly the added 20 ms. Median observation-to-plan age remained 306 ms.
Action selection took 81.96 ms elapsed / 22.75 ms CPU, and model forward
40.98 / 4.61 ms. Neither arm demonstrates restored navigation under delay.
Physical evaluation confirmed zero contacts and no arrivals for supervised
as well. The complete six-outcome comparison (original delay, 1-ms switching,
shared history; both models) is saved at
`go2_planning_timing_interventions_v1_attempt_001/result.json`. All six failed
to arrive and recorded zero contacts. Completed supervised depth was retired;
all non-depth outcomes, diagnostics and source snapshots remain.

Next prospective experiment: isolate the frozen shared-history model in one
spawned CPU worker, keeping input preparation, selection, tracking, physics,
deadlines and the added 20 ms unchanged. The parent synchronously awaits each
response, so packing, IPC, inference and response waiting remain charged to
planning service. No forecast is delivered ahead of completion. The worker
receives only causal history tensors and known candidate commands, and its
state digest must match the parent's frozen model. This tests whether process
isolation removes enough of the live waiting to restore physical navigation;
it does not assume the precise cause is the Python interpreter.

Worker-versus-local shared inference was bitwise equal for all outputs on
three recorded frames, both frozen models, normal and short-pulse commands,
plus a distinct-history fallback. The first check is retained at
`go2_isolated_forecast_equivalence_v1_attempt_001`; its isolated transfer
timings are diagnostic only (the preceding native mission was terminal but
still archiving). A runtime integration check additionally exercises the
model-input hook, worker identity, response ordering and treatment receipts.
That runtime check passed for both models, both command-duration modes and
the distinct-history fallback, with bitwise equal outputs. Its evidence is
`go2_isolated_forecast_runtime_equivalence_v1_attempt_001`.

Run full 4800-tick JEPA then supervised missions sequentially, retaining the
5-ms parent switch interval and +20-ms planning delay. Roots:
`go2_isolated_forecast_plus20ms_{arm}_noise_2mm_native_layout03_4800_v1_attempt_001`.
Use the same CPU group 8–15,24–31, with one additional single-threaded inference
worker and no concurrent native run or heavy analysis. Save per-call worker
CPU/wall time and parent timing including transfer. Preserve both outcomes;
assess actual navigation and deadline margins before adopting the change.

The isolated JEPA run exhausted 480.92 simulated seconds with no arrivals,
tracking intact, and 160/1200 plans on time. All 1200 forecast responses came
from the declared single-threaded worker, with the matching frozen state;
all additional planning delays were exactly 20 ms. Worker inference median
was 3.65 ms elapsed / 3.64 ms CPU. Parent inference including transfer fell
to 14.86 ms elapsed, but predictive clearance rose to 33.26 / 7.23 ms.
Action selection remained 79.25 / 18.90 ms and median observation-to-plan
age 306 ms. Waiting moved between parent stages rather than disappearing.

Revision before launching the supervised isolation repeat: defer that run.
The existing original/shared pairs already show both models suffer, and the
isolated JEPA measurements now point to parent-wide interference. Instead,
test a specific candidate mechanism in a fixed native scene. The installed
Genesis renderer's native Numba color/depth readback functions do not request
`nogil`. Compile instance-local variants from the same function bodies with
`nogil=True`, alternate original/nogil/original/nogil over ten paired renders
each, and measure a 1-ms Python heartbeat. Require bitwise identical primary
and auxiliary RGB/depth and no physics advancement. Restore the originals.
This tests responsiveness and pixel equality, not navigation or deployment
timing. Root: `go2_renderer_readback_gil_fixed_pose_v1_attempt_001`. No package
source is edited, no deadline is relaxed, and the isolated failure remains.

The fixed-pose readback test passed bitwise pixel equality on all 40 paired
renders, but did not improve responsiveness: heartbeat p95 gaps remained
14.78–14.84 ms in both variants; paired rendering remained about 58.8 ms.
This rejects readback-only lock release as the needed fix at this pose.
Next use the same A-B-A-B procedure for only the compiled `_forward_pass`
drawing function, whose original native compilation also does not request
lock release. Root: `go2_renderer_forward_pass_gil_fixed_pose_v1_attempt_001`.
Again preserve function bodies, pixels, camera pose, physics, and originals.

The drawing-pass A-B-A-B test succeeded: original heartbeat p95 gaps were
15.05 and 15.04 ms; nogil variants reduced them to 1.41 and 1.20 ms. Maximum
gaps fell from 15.6–16.6 ms to 4.64–4.69 ms. All 40 paired renders retained
bitwise identical primary/auxiliary RGB and depth, with no physics advancement.
Rendering itself stayed about 59 ms per pair. This supports drawing-pass lock
holding as a source of parent-thread interference, but is not yet a navigation
repair or a calibration of deployed hardware.

Prospective full navigation test: JEPA then supervised, each with the original
six-history inference, 5-ms parent switch, +20-ms planning delay, 4800-tick
budget and every original selection/acceptance/arrival rule. Only the camera
instance's compiled drawing pass receives nogil=True. Compile and compare
fixed-pose pixels before the timed mission; keep OpenGL context ownership on
the original scene thread. Record normal direct planner-stage timings.
Roots: `go2_nogil_drawing_plus20ms_{arm}_noise_2mm_native_layout03_4800_v1_attempt_001`.
Run sequentially on CPUs 8–15,24–31. All previous native processes are terminal;
the isolated JEPA physical evaluation confirmed zero contacts and no arrivals,
and its diagnosed depth was retired. Roughly 4.0 GiB is free before JEPA.
Do not combine this with shared encoding or isolated inference: test the
renderer change against the original delayed-controller outcomes first.

The drawing-pass JEPA mission completed a physically verified round trip in
208.40 simulated seconds, with zero contacts. Outbound arrival was frame
1152 and home arrival frame 2080. Over their one-second quiet dwells, maximum
physical distances were 21.11 and 20.86 mm, and maximum 100-ms speeds were
16.80 and 20.76 mm/s; all requested dwell commands were zero. Tracking remained
intact. Of 495 plans, 478 were on time (96.6%), versus 56/1200 (4.7%) in the
original delayed JEPA run. All 495 added delays were exactly 20 ms.

Median observation-to-plan age fell to 254 ms including the extra delay,
compared with 310 ms originally. Post-survey action selection was 32.54 ms
elapsed / 26.63 ms CPU; model forward was 9.50 / 8.88 ms and predictive
clearance 7.04 / 6.49 ms. This is an exercised renderer scheduling change
with a successful closed-loop result, not merely an offline speedup. Retain
its full raw recording as the first successful reference for this change.
It remains one exposed development layout, one execution, ideal gyro, and
non-real-time simulation; independent replication and deployment evidence
remain necessary. Complete the matched supervised drawing-pass run next.

Recording headroom was restored by retiring redundant depth from the older
completed full-direct adapter failure, preserving its strict visibility
failure's exact frame-1317–1321 inputs and all non-depth outcome evidence.
About 9.1 GiB is free before the supervised launch; current failure replays
and the new successful JEPA recording remain full.

The matched supervised drawing-pass mission has now completed and passed physical arrival evaluation: round trip in 160.04 simulated seconds, zero contacts, intact tracking, and no pipeline faults. Both outbound and home arrivals satisfied the one-second stationary dwell and 4-cm physical radius. Of 380 plans, 368 were on time (96.8%); the added delay was exactly 20 ms for every plan. Median observation-to-plan time was 237 ms before the added wait and 257 ms after. Source snapshots and nogil_drawing_summary_v1.json preserve the treatment and timing diagnosis.

The renderer-fix pair is therefore complete: JEPA 208.40 s and supervised 160.04 s, both verified round trips with zero contacts on this one exposed development layout. This supports the scheduling repair, not a JEPA advantage or general reliability claim. Fresh-maze replication remains to be launched; the new layout generator is preparation only. Neither completed run required a visual-recovery plan, so recovery robustness remains unresolved.
