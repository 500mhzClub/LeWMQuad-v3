# Live navigation with synthetic depth noise

The local floor/correspondence-depth tracker completed all four full recorded
journeys with 2-mm independent depth noise. Its clean full-recording controls
are running. Replay leaves the trajectory fixed; the next experiment must
exercise online perception, mapping, action selection and physical execution.

Fix eight native assignments before observing any outcome: layouts 0–3 with
2-mm noise, followed by layouts 0–3 with zero noise. These revisit the four
post-repeatability development layouts; they are not new independent layouts.
Use the same learned controller, model/residual weights, persistent routing
memory, 4,800-tick budget, 20-mm observed/40-mm physical arrivals and quiet dwell,
actions, footprint, measurement thresholds and measured-simulation timing.
Use LocalFeatureDepthConsensusMotion and LocalFeatureDepthRegistration in both
conditions. No tuning or replacement of failures within this population.

Launcher: `scripts/run_go2_live_local_feature_depth_noise_development.py`,
arguments `--layout-index 0|1|2|3 --sigma-mm 0|2`. Output roots are
`go2_live_local_feature_depth_noise_{0|2}mm_native_layoutXX_4800_v1_attempt_001`.
Run native pairs 0/1, then 2/3, at most two owners at once, and wait for both
archives and process exits before the next pair. Keep disjoint physical CPU
groups 0–7/16–23 and 8–15/24–31 and single-thread numerical libraries. Start
after the full-recording replay roster completes. A complete native pair has
recently taken roughly 8–12 minutes plus archival work; this is not a timeout.

Noise uses the exact replay helper and seed 2026091414, independently per valid
pixel, camera, layout and acquisition frame. Original invalid rays stay unknown;
perturbed values outside 0.2–5 m become invalid. RGB, gyro and ideal acquisition
stamps remain unchanged. The live session supplies the resulting packets to
every consumer: tracking, registration, mapping and independent obstacle checks.
Noise generation and packet hashing are charged to acquisition time. Mapping
and independent obstacle algorithms remain unchanged; the local floor estimator
is currently used only by tracking/registration. Failures there are a relevant
outcome rather than justification for changing this fixed experiment mid-run.

Retain the compact unperturbed native depth, RGB and exact delivered noisy
packet digests with the seed/recipe. Do not duplicate the noisy depth arrays.
The recording has a distinct schema: the ordinary PublicReplay rejects it;
`scripts.live_depth_noise_session_development.NoisyPublicReplay` reconstructs
and verifies the actual delivered stream. An actual recorded paired frame
passed both clean and noisy injection/archive round trips, input immutability,
depth validity and invalid-ray preservation checks. Both worker initializers
selected the intended tracker and registration classes. These integration
checks do not count as navigation evidence.

Before dispatch, check live CPU/GPU use, RAM and output space. The last capacity
check found 74 GiB available RAM and 21 GiB free artifact storage. Comparable
retained full recordings use 2.2 GiB for 3,106 frames and 3.4 GiB for 4,805;
the first native pair fits without depth retirement. Reassess capacity between
pairs, using the existing authorized retention policy if necessary.

After each owner exits, evaluate all reported arrivals against saved physics,
including quiet dwell and contacts, and preserve partial/failure trajectories.
Report all eight assignments, goal/round-trip rates, pose error, command vetoes,
planning latency and failure causes. Compare clean/noisy runs using the same
new estimator. This is a development sensor-sensitivity experiment; it does
not identify a JEPA-specific benefit, calibrated hardware robustness, strict
real-time operation or real-robot success.

## First native pair dispatched

All eight full replay assignments finished successfully before dispatch.
The final resource check showed approximately 75 GiB available RAM, 21 GiB
artifact-volume free space and idle GPUs. The same 16 physical CPU cores /
32 logical CPUs are divided into the declared two groups. The first two
2-mm native owners launched: layout 0 session 39452 and layout 1 session 41884; results are pending.

Both owners were confirmed live with actual launch records: PIDs 3527051
and 3527070. Both reached camera frame 100 with registered pose frame 98 and
an active OUTBOUND mission. The noisy session, new tracker/registration and
persistent routing scope are present in both launch records. No live-navigation
outcome is established yet. Finish this pair through archival and owner exit,
then run noisy layouts 2/3 followed by all four clean controls unchanged.

## First noisy pair: navigation phase complete, archival pending

Both layouts exhausted 4,800 navigation ticks with 4,805 captured frames, no
reported arrival, no reported disallowed contact and intact tracking. Plans
were on time in 1,195/1,200 and 1,197/1,200 cases. Independent obstacle-floor
planes were unavailable on 4,675/4,805 and 4,447/4,805 frames respectively;
every unavailable receipt cites insufficient combined measured candidates.
The final frames had only 39 and 69 candidates against the unchanged minimum
of 100. Both initial panoramas completed only two of nine view stages.

Unavailable/stale observations and the resulting command-window vetoes
dominated requests. These records identify independent raw-depth floor
support as a bottleneck despite improved tracking/registration. Preserve
`independent_floor_support_diagnostic_v1.json` per root. Owner exits and
physics evaluation are still pending at this diagnostic stage. The next
hypothesis is to apply the existing local floor-depth estimate in the
independent obstacle detector while retaining raw obstacle points and all
acceptance thresholds. Do not activate it within the fixed current roster.

## First pair fully archived and evaluated; second noisy pair launched

Both owners exited 0 after 682.53 / 712.32 seconds including archival,
with maximum RSS 25,425,696 / 25,408,480 KiB and zero swaps. Independent
physics evaluation found no arrivals or contacts, all 4,805 poses present,
and median/maximum errors 1.412/5.023 mm and 0.196/1.324 mm. The per-root
`continuous_native_arrival_evaluation.json` and `live_navigation_summary_v1.json`
are terminal evidence; pending flags in earlier diagnostic files describe
their earlier creation stage.

The fixed retrospective probe (`scripts/probe_go2_live_independent_floor_development.py`)
uses startup, first unavailable obstacle floor and final frame in each run.
It verified the actual delivered noisy packet digests and reproduced all six
original floor receipts exactly, holding public gyro up fixed. Local averaging
changed candidate counts from 57 to 1,796 and 39 to 1,283 on layout-0 failed
frames 50/4804, and 85 to 4,467 and 69 to 4,582 on layout-1 frames 34/4804.
All four formerly unavailable planes passed the unchanged fit rules; both
startup planes also remained available. These are component counterfactuals,
not executed navigation. Results: `local_floor_obstacle_frame_probe_v1.json`.
An inactive observer variant is in `lewm/local_floor_independent_obstacles_development.py`;
the current native launcher does not import or use it.

With both previous owners exited, 76 GiB RAM available, 13 GiB artifact space
and idle GPUs, noisy layouts 2/3 launched with unchanged source/settings on
the same disjoint CPU groups: sessions 75412 and 2047. Finish both through
archive and exit, then evaluate them and run the four clean controls. Review
space before the clean pairs; the completed first pair used about 7 GiB.

## Observer sequence integration and remaining noisy navigation phases

The inactive local-floor obstacle observer was exercised on the actual first
51 noisy layout-0 frames, through the original first unavailable frame 50.
All 51 original observer receipts reproduced exactly. Original current
obstacles were available on 50 frames; the candidate returned current obstacle
evidence on all 51. Every returned candidate obstacle-cell set was independently
reconstructed from the original noisy depth points using its fitted floor,
the same height band and crop. No raw obstacle points or thresholds changed.
Evidence: `local_floor_observer_sequence_51_v1.json` in the layout-0 root.
This is a sequence integration check, not a native navigation result.

Noisy layouts 2/3 completed their navigation phases with 4,805 frames each, no
reported arrivals or disallowed contacts, and intact tracking. Plans were on
time in 1,187/1,200 and 1,189/1,200 cases. Their independent floor detectors
were unavailable on 4,615 and 4,419 frames respectively, again solely because
of insufficient combined measured candidates. Recording and owner exits remain
pending at this stage. Preserve their per-root floor-support diagnostics and
finish all four clean controls unchanged after archival.

## All four noisy assignments complete; first clean pair launched

Layouts 2/3 exited 0 after 681.26 / 713.54 seconds including recording,
with maximum RSS 25,407,272 / 25,417,896 KiB and zero swaps. Physics evaluation
found no arrivals or contacts, all 4,805 poses present and median/maximum
errors 0.258/1.054 mm and 1.744/2.884 mm. All four noisy assignments therefore
ended without goals or returns, with zero disallowed contacts and complete
tracking. All source-hash sets match. Combined noisy results are in
`go2_live_local_feature_depth_noise_noisy_four_layout_summary_v1_attempt_001/result.json`;
its pending clean fields describe this intermediate stage, not the final
eight-assignment outcome.

The remaining diagnostic probes reproduced original planes exactly on layout-2
frames 0/19/4804 and layout-3 frames 0/34/4804. Local candidates increased
80 to 8,796 and 81 to 9,680 on layout 2; 90 to 4,579 and 80 to 4,896 on
layout 3. Across the twelve selected actual noisy frames, all original
planes reproduce and all local-floor alternatives are available. No
counterfactual navigation is inferred from these component results.

With both previous owners exited, 76 GiB RAM available, 9.9 GiB artifact
space and idle GPUs, clean layouts 0/1 launched unchanged on the two declared
CPU groups: sessions 56457 and 88689. Wait for both archives and exits,
verify the results, then run clean layouts 2/3. The inactive local obstacle
detector is not used by any assignment in this roster.

Both clean owners are confirmed live (PIDs 3532361/3532396). At camera
frame 1400, registered pose frames were 1399/1398 with active OUTBOUND
missions. Their observed goal distances have changed substantially as they
move through the maze, unlike the near-stationary noisy initial-survey runs.
No clean arrival or final outcome is established yet. Reviewed retirement of
the completed unexercised 90-mm terminal probe reclaimed 2.879 GiB; about
12.68 GiB artifact-volume space remained while this pair was running.

## First clean pair: physical speed stops, one verified goal

Both owners exited 1 with `CONTEXT_NATIVE_CONTACT_SPEED_OR_DOMAIN_STOP`,
after 332.96 / 287.97 seconds including recording. Layout 0 acquired 2,300
frames and published 2,299 poses; layout 1 acquired 1,864 and published
1,862. Maximum RSS was 13,431,684 / 11,384,868 KiB; zero swaps.

Physics evaluation verified layout 0 OUTBOUND arrival at frame 1860: all
physical positions during the one-second dwell were 11.663–16.205 mm from
the goal, maximum 100-ms speed 0.01970 m/s and all requested intervals zero.
It subsequently stopped during return. Layout 1 had no arrival. Neither
returned home and neither recorded a disallowed contact. Median/maximum
pose errors were 4.499/8.499 mm and 2.370/5.500 mm.

The terminal trace identifies speed-limit exceedance in both cases: full
3-D translation speeds 0.301642 / 0.300472 m/s against the fixed 0.3-m/s
limit. Both were inside the domain. Horizontal speeds were only 0.288687 /
0.289771 m/s; the guard includes vertical gait motion. Do not reinterpret
this as collision or relax the guard. Whole-recording maximum speed includes
initial settling and is labeled separately in `physical_stop_diagnostic_v1.json`.
The post-camera maximum equals the terminal speed in both runs.

The independent floor remained available on every processed receipt: 2,299/2,299
and 1,863/1,863. Thus clean inputs permitted navigation, but this controller
still lacks reliable execution within its declared speed envelope. Preserve
all failures separately from the noisy floor-support failures. Per-root
physics evaluations, speed diagnostics and `live_navigation_summary_v1.json`
are saved.

Both first-pair recordings are complete. Clean layouts 2/3 launched next
unchanged on the declared CPU groups after a capacity check (76 GiB RAM
available, 9.9 GiB artifact space, idle GPUs). Their current tool sessions
are recorded in the continuation state; evaluate both after archival and
owner exits before activating the independent local-floor follow-up.

Final clean pair handles: layout 2 session 52537; layout 3 session 42799.

Both final clean owners were confirmed live with the intended launch
settings: PIDs 3533509/3533528. Both reached frame 800 with active OUTBOUND
missions and changing observed positions. Their final outcomes remain pending.

The two clean speed stops share the same recent requested-command sequence:
forward, 400-ms left arc, 400-ms left turn, 400-ms hold, then forward. Stops
occurred about 424/428 ms after forward began. Layout 0 continued forward;
layout 1 requested right arc 28 ms before stopping. Both were in a gait
transient with downward body velocity (about 0.087/0.079 m/s), rather than
exceeding 0.3 m/s horizontally. Save `speed_stop_transition_diagnostic_v1.json`
per root. This suggests examining command transitions and the allowed action
amplitudes against the total-speed envelope; it does not establish that any
modified command would pass. No action or guard change is included in the
current roster or the already fixed independent-floor follow-up.

## Final eight-run outcome

Both final clean owners exited 0 with complete recordings: layout 2 took
434.60 seconds, 3,058 camera/pose pairs and maximum RSS 16,920,116 KiB; layout 3
took 411.07 seconds, 2,729 pairs and maximum RSS 15,506,700 KiB. Neither swapped.
Physics verified both round trips, including distance, quiet motion, zero
requests throughout the dwell and zero disallowed contacts. Layout 2 goal/home
arrivals were frames 1952/3056, with maximum physical dwell distances
16.881/10.190 mm. Layout 3 arrivals were 1784/2727, with maxima 19.483/21.275 mm.
Median/maximum pose errors were 3.586/6.607 mm and 2.887/5.863 mm.

All eight assignments are terminal. Noisy: 0/4 verified goals, 0/4 round trips.
Clean: 3/4 verified goals, 2/4 round trips, with the two speed-stop failures
retained. All eight had zero disallowed contacts. Shared source hashes match
and each clean/noisy pair differs only in owner/noise-condition metadata.
Combined result: `go2_live_local_feature_depth_noise_eight_run_summary_v1_attempt_001/result.json`.

The clean controls establish useful navigation with this estimator, but not
reliability within the declared speed envelope. Synthetic depth noise caused
severe independent floor-support loss despite accurate tracking. Proceed to
the already fixed four-run independent-local-floor experiment described in
`docs/go2_live_local_floor_obstacle_noise_2026-09-15.md`; preserve this entire
comparison and do not modify the planned action set or speed guard.

The final clean/noisy trajectory PNG and SVG were generated in the combined
result directory and visually inspected. They show the four near-stationary
noisy failures alongside all clean trajectories, marking the two speed-stop
endpoints and retaining both successes and failures. The independent-floor
follow-up has now launched its first pair after all eight owners exited.
