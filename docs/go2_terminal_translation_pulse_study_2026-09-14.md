# Terminal translation pulse development

The preceding full layout-6 attempt verified outbound arrival but ended
53.6 mm from home without a return dwell. Its predictive hold changed only
one committed action; its forecast expected 19–20 mm final goal distance,
while the observed result settled around 28–29 mm. The next hypothesis is
that shorter translation commands permit useful corrections after such a
near miss. Arrival criteria remain 20 mm observed radius, one-second quiet
dwell, and independent 40 mm physical radius.

`TerminalTranslationPulseRuntime` keeps the six candidate actions, 300-ms
dispatch delay, and 400-ms planning cadence. When routing to the exact goal
within the existing 100-mm terminal region, forward and arc candidates
become 100-ms commands followed by zero. Turn commands remain 400 ms.
Forecast command tensors and frozen residual features use the same pulse
sequences. The scoring endpoint stays 700 ms after observation to include
settling. Scheduled translation windows expire after 100 ms, and the next
forecast's prefix consequently contains the actual zero tail. Original
deadline, overlap, actual-prefix, geometric and arrival checks remain.
The unchanged frozen residual has not yet been shown accurate on pulses.

Twelve focused tests pass across the new treatment, existing commitment
ledger and delayed planning. They cover forecast/correction command
agreement, unchanged turn inputs, pulse expiry, following zero prefixes,
actual-request mismatch, missed deadlines and overlapping windows.

The prospective local comparison uses layout-6 scene construction with an
explicit nearby mission: outbound `[0.09, 0]` m, then return `[0, 0]` m.
Both arms use the same 900-camera-interval navigation budget, learned model,
orthonormal gyro perception, fine-goal routing and predictive arrival hold.
This is a terminal-control study, not an independent unseen-maze result.
Run the pulse arm first, then the standard 400-ms arm, one native owner at
a time. The same launcher selects the arm without source edits:
`scripts/run_go2_terminal_pulse_probe_development.py --arm pulse|standard`.

Exclusive roots:
- `go2_terminal_100ms_pulse_local_round_trip_layout06_v1_attempt_001`
- `go2_terminal_standard400ms_local_round_trip_layout06_v1_attempt_001`

Prelaunch: 16 physical/32 logical cores, affinity 32, CPU 1.9%, 77.31 GiB
available RAM, 19.39 GiB free disk, both GPUs idle, no experiment owner.
One native owner and four post-run camera archive writers are used. Each
arm is limited to approximately 900 frames to keep storage bounded. Preserve
both outcomes and independently evaluate arrivals after final archiving.

The first 90-mm-goal probe completed exit 0 with 905 frames, no arrivals
and no contacts. No terminal pulse was enabled or selected: the routing
records contained 72 initial-panorama decisions, 69 frontier-view decisions,
and 84 frontier-route decisions, but no observed route to the goal cell.
The nearby goal was in floor space not observed by the cameras. Consequently
this attempt did not exercise the intended control treatment and supplies
no pulse-versus-standard precision evidence. It is preserved with its
independent arrival evaluation (pose error median/maximum 4.60/6.78 mm).
The uninformative standard-arm run at that same target was not launched.

The revised prospective pair uses `[0.65, 0]` m in the entrance corridor,
retaining the scene, return target and 900-interval budget. This gives the
cameras a down-corridor goal to observe before final approach. No map or
free-space information is supplied to the controller. Select this mission
with `--goal-x-m .65`; `.09` remains available to reproduce the first setup.
The new exclusive roots insert `_goal065` before `_local_round_trip` in
the two names above. Run pulse then standard without source edits between
arms. Before revised pulse launch: CPU 0.3%, 76.93 GiB available RAM,
16.48 GiB free disk, no experiment owner. One native owner, then four
archive writers. This revised experiment has no result yet.

The revised pulse arm completed exit 0 with 907 camera pairs and admitted
poses. Independent evaluation verified outbound arrival at frame 379:
physical goal distance during the one-second dwell was 11.43–18.62 mm,
maximum 100-ms speed was 0.02925 m/s, and all requests were zero. There
were no contacts and no return arrival. Pose error median/maximum was
4.42/5.35 mm. The run exhausted its 90-second budget.

`terminal_pulse_dispatch_diagnostic_v1.json` confirms two selected and
committed pulse plans, at frames 364 and 368. The first executed exactly
five 20-ms forward requests, followed by zero at the 100-ms expiry and
through the remaining 300-ms planning interval. The second was suppressed
by measured mission settling before any motion. Those intentional expired
pulse tails retain the inherited `NO_ON_TIME_PLAN` reason; they must not
be counted as evidence of late planning. Before outbound arrival, 88 plans
were on time and three late. Return decisions all sought observed frontiers
rather than the unobserved home goal cell; the overall 111/222 on-time
count also reflects slower return routing. No round-trip success is claimed.

The standard 400-ms arm is now running on the same 650-mm mission with
unchanged source. Before its launch: CPU 0.3%, 76.90 GiB available RAM,
13.57 GiB free disk, both GPUs idle, no remaining native owner. One native
owner and four archive writers remain the selected concurrency. Comparing
the two completed outcomes is still required before claiming an advantage
from the pulse treatment in this local experiment.

The standard arm also completed exit 0 with 905 admitted camera/pose pairs.
Independent evaluation verified outbound arrival at frame 365 (1.4 seconds
earlier than the pulse arm). Physical dwell distance was 14.98–18.88 mm,
maximum 100-ms speed 0.04750 m/s, and every request was zero. No contacts
or return arrival occurred. Pose error median/maximum was 4.74/5.35 mm.

`go2_terminal_pulse_standard_goal065_comparison_v1_attempt_001/result.json`
contains both independently evaluated summaries and confirms equal source
hashes and matched mission, model-independent timing and geometry settings,
apart from the intended translation-duration treatment. Both controllers
reached the outbound target, and this single pair establishes no pulse
advantage. The pulse implementation did execute the intended command and
zero tail, but has not been shown to fix the harder maze-terminal failures.
These are shared-host measured-timing runs, not deterministic policy
counterfactuals. Neither result establishes unseen-maze transfer or a causal
JEPA-training benefit.

Final physical home distances were 0.446 m (pulse) and 0.496 m (standard).
The return routes sought frontiers rather than the unobserved home cell;
the local setup's return failure therefore should not be attributed to
terminal pulse precision. All three local attempts remain archived.
No native owner remains, and free disk is approximately 10.77 GiB.

The pulse treatment remains experimental rather than promoted on the basis
of this pair. A full-maze pulse test is still needed on the harder terminal
approach, but its recent 16-GiB camera archive does not fit current space.
The existing storage records identify a practical future-format change:
the in-memory capture writer stores native optical depth together with
deterministically derived depth and validity arrays, while `PublicReplay`
already reconstructs those same packets from native depth and compares them
with the stored derived copies. A new native-depth-only archive version can
retain raw pixels, RGB, body/gyro, all execution records and hashes of the
original derived arrays, then reconstruct and verify the packets on read.
The timed live observations need not change at all.

Next implementation targets are
`scripts/in_memory_paired_camera_session_development.py` and
`scripts/in_memory_public_replay_development.py`, with the new format selected
explicitly for future runs. Keep V1 reading and every completed archive
unchanged. Measure actual savings and full public-packet equality on an
existing ordinary development recording before a new native launch; use
temporary regenerated copies so this validation does not retain another
large dataset. This is the next available storage action without retiring
older scientific evidence or relying on unapproved stale cleanup proposals.
