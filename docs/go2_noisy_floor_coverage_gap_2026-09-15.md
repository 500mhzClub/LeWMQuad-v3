# Noisy floor-map connectivity diagnosis

The completed gyro-height-floor layout 2 had complete independent floor
availability but no goal and 1,005 view-budget-exhausted decisions. It still
lost exploration options after the first standoff panorama. Current inputs
are the fully retained
`go2_live_gyro_height_floor_noise_2mm_native_layout02_4800_v1_attempt_001`.

`scripts/probe_go2_union_floor_coverage_development.py` replayed the recorded
mapping prefix through frame 572, reproducing all 144 saved local-map count
witnesses. Fixed admitted public poses and actual delivered noisy packets
were used. Raw, local and union maps had identical initial floor height and
coarse/fine obstacle geometry. Their final floor sets contained 493, 842 and
1,335 cells; the union exactly preserved both component sets at every frame.
It did not fix the failure. Both local and union maps had a one-cell reachable
component at [18,-3], the same eleven unknown connector cells, and no frontier
after the saved exclusions. Removing exclusions only reselected [18,-3].
The raw-only map had no admissible entry. Do not promote the union from its
larger cell count; no counterfactual navigation was executed.

`scripts/probe_go2_floor_gap_pixels_development.py` examined the original
per-pixel rejection conditions for that connector over the same recorded
prefix, without changing thresholds. For each cell/variant it reports the
visible rectangle with highest original passing fraction; this is a
post-hoc diagnosis, not a new coverage classification. The diagnostic
reconstructs original covered/not-covered decisions exactly for those
rectangles. All best local-depth views of cells [9,-2] through [17,-3] had
zero invalid quadrilaterals and zero height-band rejections. Their failures
were entirely in the ground-mesh predicate. Local passing fractions increased
with distance, from 651/3,540 (18.4%) at [9,-2] to 727/735 (98.9%) at [17,-3].
The existing rule requires every quadrilateral in a floor square's projected
image rectangle to pass. The eight mesh rejections at [17,-3] therefore still
prevented that cell joining the map. Raw-depth mesh tests were much worse;
at [9,-2] only one of 3,540 quads passed despite complete valid height-consistent
measurements. Cell [8,-2] was never entirely in either recorded camera
frustum and must remain distinguished from these observed-but-rejected cells.

Outputs are `union_floor_coverage_prefix_probe_v1.json` and
`floor_gap_pixel_rejection_probe_v1.json` in the current gyro-floor layout-2
root. No native physics, new pose admission, full controller replay or new
commands were used. Both probes ran on CPU 8, separate from the cached live
layout-0 trial on CPUs 0–7,16–23. That trial was not changed.

The next scientific hypothesis is to use an available current paired measured
floor plane for orientation evidence instead of independent noisy pixel-scale
mesh normals, while retaining original raw-pixel validity and the all-pixel
10-mm height check for each projected floor rectangle. Keep unobserved cells
unknown, raw obstacle geometry and pose acceptance unchanged, and explicitly
account for unavailable current planes. This changes floor classification
and needs a recorded-map comparison and a fresh live navigation test. It is
not implemented or justified as hardware support certification by this probe.

## Current-plane candidate and prospective development test

`lewm/current_plane_floor_coverage_development.py` implements the candidate.
Each mapping observation refits the current paired raw height-selected plane
with the existing count, extent, alignment and selected-point residual gates.
When available, its orientation evidence replaces pixel-scale mesh normals;
the original projected-rectangle bounds, raw four-corner validity and all-pixel
10-mm height checks remain. No current plane means the existing local-depth
coverage rule is used. Initial floor-height estimation, pose admission and
raw coarse/fine obstacle sets are unchanged. This is a changed floor
classifier, not unchanged acceptance or a foot-support certificate.

Two focused tests passed: noisy observed floor coverage with invalid-ray,
height-outlier and frustum rejection; and exact original coverage fallback
when the current plane is unavailable. The first recorded-map attempt stopped
at frame 0 because initial specific force was not normalized before its unit
check. Its failure/source identity is preserved in
`current_plane_floor_coverage_initialization_failure_v1.json`; normalization
now follows the original mapping gravity calculation.

The corrected recorded-prefix comparison reproduced all 144 baseline count
witnesses and refit an available current plane on every frame. Floor height
and raw coarse/fine obstacle geometry remained identical. At frame 572, the
candidate had 2,222 floor cells and a 389-cell reachable component with 13
frontiers, versus the baseline's one-cell component and no unexcluded frontier.
Its new route entered at (0.475,-0.075) and ended at (0.475,0.025). The unseen
connector cell [8,-2] remained explicitly unknown. The complete result is
`current_plane_floor_coverage_prefix_probe_v1.json`. Fixed recorded poses were
inputs; no counterfactual controller commands or physics were executed.

A five-frame integration replay through the cached tracker, original full
registration admission and production captured mapping class also passed.
`current_plane_floor_live_chain_smoke_v1.json` records the resulting current
floor/fine-obstacle sets. The source and writer checks are complete.

Proceed with one prospective noisy live layout-2 run using
`scripts/run_go2_current_plane_coverage_noise_development.py`, root
`go2_current_plane_coverage_noise_2mm_native_layout02_4800_v1_attempt_001`.
Use CPU group 0–7,16–23, the existing fixed noise recipe, learned weights,
action bank, speed stop, pose/obstacle observers and 4,800-tick budget.
Use the bounded retained-depth tracker cache, now supported by exact recorded
pose equivalence and the completed cached layout-0 trial. Relative to the
gyro-floor layout-2 predecessor this changes both the routing floor rule and
cache lifetime: report the combined system outcome, not an isolated live
mapping causal effect. Preserve success or failure without retuning/replacing
the assignment. This is a development revisit; hardware, final evaluation
and broad reliability remain unproven. Available RAM was 76 GiB and artifact
free space about 5.0 GiB before launch.

Live owner confirmed: PID 3552349 / session 30437, initialization log
02:41:18 local. The actual launch record identifies the new mapper, cached
tracker, conditional raw/local fallback and both changes relative to the
predecessor. Wait for archive completion and owner exit before evaluation.

`current_plane_floor_connectivity_comparison_v1.png` and `.svg` in the
gyro-floor layout-2 diagnostic root visualize the recorded map comparison.
They show floor cells, the recorded robot position, proposed route and unknown
connector, with obstacle geometry omitted from the drawing. The counts refer
to the full reachable component, not just the displayed close-up. This is a
recorded-map figure and supplies no new navigation outcome.

## Verified targeted live round trip

The layout-2 owner exited 0 after 432.42 s including archive, max RSS
17,030,040 KiB, zero swaps. Independent physical-trajectory evaluation verifies
both arrivals: outbound frame 2056 and home frame 3065. Maximum physical
distance during each one-second dwell was 12.277/18.131 mm; maximum 100-ms
speed was 0.00942/0.02358 m/s, with every requested interval zero. Both dwell
checks passed and the full run had zero contacts. All 3,067 poses were
accepted; median/max position error was 2.435/8.895 mm. This is a successful
closed-loop noisy simulation on the previously stalled development layout,
not a recorded-map navigation claim, unseen final evaluation or hardware
qualification. Cache and floor-classifier changes remain jointly present.

Expand evaluation to the other existing development layouts 0,1,3 using the
same sensing/model/mapping/cache/action/speed/arrival settings. Fix this roster
before launch and preserve every outcome
without tuning or replacement. Layout 2 was the targeted first experiment;
the expanded cohort is a subsequent development check and must not be
described as an originally preregistered four-run study. The scientific
implementation remains fixed; the additional launcher only selects layout
and corresponding noise/physics seeds and output identity.

Recording capacity changes the execution schedule: after reviewed predecessor
depth retirement, about 6.6 GiB is free. Start layout 0 alone, then schedule
layouts 1/3 as recording capacity permits; do not assume two worst-case full
recordings will fit simultaneously. The fixed roster and scientific settings
remain unchanged. Use
`scripts/run_go2_current_plane_coverage_remaining_noise_development.py` with
`--layout-index 0`, `1` or `3`. The targeted layout-2 launcher and all scientific
modules remain unchanged. Its actual native launch metadata verifies layout 0,
the matching public mission/noise recipe, cached tracker and current-plane mapper.

Layout 0 owner confirmed live: PID 3553856 / session 73171, initialization
log 02:53:06 local, CPUs 0–7,16–23. Layouts 1 and 3 have not launched. Wait
for this archive/owner completion and evaluate before scheduling further runs
against available storage. Preserve the successful targeted layout 2 and
every subsequent result as a development expansion, not a sealed test.

The later reviewed retirement of six completed early clean transfer recordings
reclaimed 13.217 GiB and raised free space to about 20 GiB. This resolves the
capacity restriction without retiring current comparison/training inputs.
Layout 1 therefore started while layout 0 continued: PID 3554620 / session
80116, initialization log 02:58:09 local, CPUs 8–15,24–31. Both actual launch
records identify the fixed cached tracker/current-plane mapper and correct
layout references. Layout 0 remains PID 3553856 / session 73171. Layout 3
has not launched; schedule it after a native owner and its archive finish,
with no more than two native owners and the proper CPU allocation.

After this learned-controller development cohort, the existing
`HeadingFirstReactiveRuntime` in
`scripts/run_go2_heading_first_terminal_reactive_development.py` is the
stronger non-predictive control to adapt to the same current sensor/cache/map
pipeline. Its recent-reference-refresh successor changes perception and must
not be used as if sensing were matched. No new reactive implementation or run
has been created here. Such a comparison assesses complete predictive versus
instantaneous action selection; JEPA-specific attribution still needs the
matched training controls, and generality needs further independent layouts.

Layout 0 subsequently exited 0 after 481.89 s including archive, max RSS
18,621,276 KiB, zero swaps. Independent evaluation verifies goal frame 2243
and home frame 3396, with maximum physical dwell distances 18.023/7.498 mm,
quiet speeds 0.00952/0.01383 m/s and all dwell commands zero. No contacts or
pipeline faults occurred. All 3,398 poses were accepted; median/max error
2.815/6.180 mm, floor availability 3,396/3,398 and sampled path 24.297 m.
Of 842 selected plans, 836 were on time. Layouts 0 and 2 are now verified
round trips; learned layout 1 remains live and learned layout 3 is pending.

The idle even CPU group will run the fixed stronger reactive layout-2
comparison while learned layout 1 continues, as specified in
`docs/go2_current_plane_matched_reactive_noise_2026-09-15.md`. This does not
change the remaining learned assignments or their priority on the odd group.

Layout 1 subsequently exited 0 after 524.04 s including archive, max RSS
19,410,440 KiB, zero swaps. Independent evaluation verifies goal frame 2612
and home frame 3543, maximum physical dwell distances 21.906/15.528 mm,
quiet speeds 0.02241/0.02479 m/s, all dwell commands zero and zero contacts.
All 3,545 camera pairs have registered poses; median/max position error
4.685/10.664 mm. Thus learned layouts 0, 1 and 2 have verified noisy
round trips; the fourth result is pending. Layout 3 has now launched on
CPUs 8–15,24–31, PID 3556306 / session 31540, with the fixed remaining
launcher. Matched heading-first reactive layout 2 is concurrently live
on the even CPU group, PID 3555709 / session 80033. RecoveryStorage has
about 16 GiB free at this update; storage is not blocking these runs.

The fixed learned expansion is now complete: 3/4 independently verified
goals and round trips, zero disallowed contacts, floor available on
10,630/10,633 receipts. Layout 3 exited 1 after 111.47 s including archive
(max RSS 5,545,304 KiB, zero swaps), with 624 camera pairs, 623 registered
poses, no arrivals and median/max position error 2.179/4.560 mm. At native
time 63.832 s, full 3D body speed reached 0.301165 m/s against the unchanged
0.300 m/s guard; horizontal speed was 0.288869 m/s and vertical velocity
-0.085178 m/s. The body remained inside the domain and contact evaluation
found zero disallowed contacts. This is an execution-speed failure, not a
successful round trip or evidence that the new floor classifier failed.
Preserve the failure without replacement or changing the guard. The complete
cohort is saved in `go2_current_plane_coverage_noise_four_layout_summary_v1_attempt_001/result.json`;
layout 3 also has `physical_speed_stop_diagnostic_v1.json`.

A compact native speed-margin diagnostic is saved in the complete learned
cohort root as `native_speed_margin_diagnostic_v1.json`. Across the four
learned trajectories and completed reactive layouts 1/2, only learned layout
3 exceeded the fixed 0.3 m/s guard (one recorded 2-ms sample before stopping).
The other five maxima were 0.27655–0.28391 m/s. This localizes the recorded
stop to a brief excursion; it does not establish an acceptable hardware
speed bound or justify changing the guard. Native velocity remains evaluator
only, and all comparison commands and thresholds remain unchanged.
