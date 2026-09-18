# Local routing-floor coverage under depth noise

The independent local-floor detector kept floor evidence available on every
frame of its first two live noisy runs. Both initial panoramas completed, but
navigation then exhausted its view budget. Those failures remain failures;
the predecessor four-run roster must finish unchanged before this trial starts.

`scripts/probe_go2_noisy_routing_floor_development.py` replayed the actual
mapping-frame prefixes through the last selected plan before first view-budget
exhaustion on layouts 0/1. It verified reconstructed delivered noisy pixels and
reproduced every saved floor/fine-obstacle count: 114 and 122 witnesses. Compact
previously admitted public poses were held fixed; full registration acceptance
and controller state were not replayed. No native physics was used.

The counterfactual used the existing 5x5 local inverse-depth estimator only for
floor coverage. Initial floor-height fitting and every coarse/fine obstacle set
remained exactly identical to the raw baseline. The raw maps ended with 1,256 /
847 floor cells and no base-proposer route. The local maps had fewer cells,
496 / 592, but restored a base-proposer route to an unexplored frontier in both
cases. At layout-0 startup, auxiliary current floor coverage changed from zero
to 236 cells. Total cell count alone does not explain route usefulness.
Both proposed entry connectors still included unknown floor cells, as permitted
and reported by the existing routing proposal; this is not a motion grant,
complete floor coverage or counterfactual navigation success. Per-root
`routing_floor_prefix_counterfactual_v1.json` contains all outcomes.

## Fixed prospective experiment

After all four independent-floor runs finish, run one noisy navigation trial
on each layout 0–3 with
`scripts/run_go2_live_local_floor_mapping_noise_development.py --layout-index I`.
Use pairs 0/1 then 2/3 on the existing disjoint CPU groups, waiting for both
archives and owner exits between pairs. Output roots are
`go2_live_local_floor_mapping_noise_2mm_native_layoutXX_4800_v1_attempt_001`.
Preserve every failure; do not tune between assignments or replace failures.

Only routing floor coverage changes. `LocalFloorRoutingMap` retains the original
paired capture, accumulated 1-cm obstacle cells and partial-height pose reader.
`LocalCoverageGeometry.floor_coverage` applies the existing local estimator,
then the existing mesh, height and projected-rectangle coverage predicates.
Original packets remain unchanged; invalid depth is not filled. Local geometry
is explicitly estimated rather than raw measured pixel geometry. Original
initial floor-height fitting, obstacle height bands and raw obstacle point
extraction remain unchanged. Include the extra mapping computation in live
timing.

Keep independent local-floor obstacle detection, tracking/registration,
learned weights, persistent memory, actions, 0.3-m/s total-speed stop, footprint,
arrival requirements, 4,800-tick budget and fixed 2-mm noise recipe unchanged.
Compare with the completed independent-floor four-run population, not with
different clean controllers or new independent mazes. Evaluate goals/returns
against saved physics, contacts, tracking error, view completion, planning
reasons and timing. This development revisit does not establish hardware noise
calibration, strict real-time operation, JEPA-specific benefit or deployment.

The launcher is prepared and has not been activated. Current independent-floor
layouts 2/3 remain the only live native assignments.

Preparation verified the complete launch writer chain, worker initializer and
map method dispatch. The implementation exactly reproduced the counterfactual's
frame-0 floor-cell set and fine-obstacle count. A first writer check omitted the
declared CPU affinity and rejected before execution; the corrected check passed.
No native mapping trial has run yet.

## First pair launched

The independent-floor predecessor is now terminal and evaluated: 0/4 goals
and round trips, zero contacts, 19,221/19,221 floor receipts available and
4/4 initial panoramas completed. All common source hashes and non-treatment
settings matched. Its combined result is
`go2_live_local_floor_obstacle_noise_four_layout_summary_v1_attempt_001/result.json`.

After both final owners exited, the host had 76 GiB RAM available, about
13 GiB artifact space, idle GPUs and no other native/replay/probe owners.
Reviewed retirement of three diagnosed predecessor noisy-depth recordings
reclaimed 9.704 GiB; their outcomes and all current recordings remain retained.
Mapping layouts 0/1 launched on the declared CPU groups in sessions 85721/33916.
Results are pending. Preserve both through archive and exit, evaluate them,
then run fixed layouts 2/3 unchanged.

The same prefix diagnostic subsequently completed on predecessor layouts 2/3,
without changing the live roster. All 97/105 saved map-count witnesses
reproduced; the raw base proposer had no route and the local-coverage proposer
found a frontier route in both. Final raw/local floor counts were 507/486 and
1,716/489. Across all four prefixes, 438 saved count witnesses reproduced.
The combined component result is
`go2_noisy_routing_floor_prefix_four_layout_summary_v1_attempt_001/result.json`.
The counterfactuals keep poses and original obstacle sets fixed; live navigation
outcomes are still pending and are the relevant test of usefulness.

## First pair complete; layouts 2/3 launched unchanged

Layout 0 exited 1 after 347.11 seconds, 2,381 acquisitions and 2,379 admitted
poses. Registration rejected a current candidate against its transported floor
reference. Physics evaluation found no arrivals or contacts; median/maximum
position error was 5.458/8.286 mm. Final native distance from home was 0.570 m,
goal distance 1.323 m, and sampled horizontal path length 2.061 m. The nearly
constant goal distance in progress logs did not mean no translation occurred.
Maximum RSS was 13,925,656 KiB, zero swaps.

Independent floor evidence was available on only 566/2,381 frames, first
unavailable at frame 560. The initial panorama completed, but the first
frontier standoff panorama remained unfinished. There were 510 selected
standoff-view plans, 12 frontier routes and 72 initial-view plans; 472/594
selected plans were on time. Preserve the main registration failure and the
subsequent clock-closed tracking shutdown fault separately.

Layout 1 exited 0 after 702.13 seconds with all 4,805 poses/acquisitions. It
exhausted the mission budget without arrivals or contacts; median/maximum pose
error was 5.085/11.972 mm. It travelled a sampled 23.220 m, approached within
2.631 m of the goal and ended 2.514 m from home. Its independent floor stayed
available on every frame. Selected plans included 499 frontier routes and
610 standoff views; 1,136/1,200 selected plans were on time. Maximum RSS was
25,462,912 KiB, zero swaps. Mapping enables exploration here but has not
established successful goal finding. Both per-root arrival evaluations,
navigation summaries and floor/route diagnoses are saved.

With both archives complete and owners exited, unchanged runtime source
hashes, 76 GiB RAM available, about 11 GiB artifact space, idle GPUs and no
other native/replay owners, fixed layouts 2/3 launched in sessions 70992/84633.
The last original noisy-depth recording was reviewed for retirement, reclaiming
2.947 GiB; all current independent-floor/mapping recordings remain retained.
Do not tune these final two assignments or replace failures.

## Exact layout-0 registration failure and raw height support

`scripts/diagnose_go2_live_mapping_registration_development.py` replayed every
delivered noisy packet through failure. All seven recorded pose/identity fields
for all 2,379 raw and registered poses matched exactly; registration failed at
frame 2379 with the same structured live stage/reason. The floor detector had
66 auxiliary candidates, below its 100-point minimum, and used a full-plane
anchor from frame 1569 (81 seconds earlier). One candidate exceeded the 3-mm
transport limit: maximum 3.0633 mm, RMS 2.0331 mm, mean signed residual -2.0027 mm.
Raw snapshots, prior full registered evidence, anchor state and residual
diagnosis are retained under `sensor_failure_replay_v1`. Replay took 319.10 s.
Its initial substring comparison against nested exception text reported false
because of escaping; `live_failure_comparison_v1.json` records the exact
structured match without modifying the original result. The reporter is fixed
for future diagnoses; no scientific replay was repeated for this reporting fix.

`scripts/probe_go2_noisy_floor_height_support_development.py` used the recorded
independent public gyro normal and original noisy depth points. It measured
the densest 6-mm height slab (twice the existing 3-mm residual limit), replacing
the pixel-mesh-normal candidate rule only in this diagnostic. At frames
0/559/560/2379/2380, existing candidate counts were 11781/166/91/66/49;
the slab contained 23748/4427/3965/3625/3652 raw points, all with at least
2-cm second spatial extent. At the first floor failure its extent was 3.28 cm;
at registration failure, 2.94 cm. Later slabs contained only auxiliary points.
They represent 44–48% of below-body points near the wall, not all raw data.
This supports investigating noise-tolerant candidate selection rather than
assuming the floor disappeared. It does not certify that a selected slab is
the floor, validate a new estimator, or establish counterfactual navigation.
The live mapping roster and its estimator thresholds remain unchanged.

## Final four-run mapping result

Layout 2 exited 0 after 653.90 seconds with 4,805 acquisitions/poses, no arrivals
or contacts and median/maximum position errors 6.053/7.571 mm. It exhausted
the mission budget after 978 view-budget-exhausted planning records; 212/222
selected plans were on time. Floor was available on 4,799/4,805 receipts, so
its remaining failure is not persistent floor loss. It finished 0.485 m from
home, with a sampled path of 3.434 m. Maximum RSS was 25,385,488 KiB; zero swaps.

Layout 3 exited 1 after 183.14 seconds with 1,124 acquisitions and 1,121 admitted
poses. Registration failed against the transported floor reference. Physics
found no arrivals or contacts, median/maximum error 3.412/4.997 mm, final home
distance 1.436 m and sampled path 2.706 m. Independent floor availability was
538/1,123 receipts, first missing at frame 538. Plans included 43 frontier routes
and 167 standoff views; 269/280 were on time. Maximum RSS was 7,945,076 KiB;
zero swaps. An exact replay reproduced all 1,121 pose-field rows and the same
registration failure at frame 1121: 51 candidates, anchor frame 537, maximum
transport residual 3.0520 mm, one candidate above 3 mm. Its snapshots and result
are retained under `sensor_failure_replay_v1`.

All four mapping assignments are terminal and evaluated: 0/4 goals and round
trips, zero contacts; registration failures on 0/3 and mission-budget failures
on 1/2. All common source hashes, treatment source sets and non-treatment
settings match. Combined evidence is
`go2_live_local_floor_mapping_noise_four_layout_summary_v1_attempt_001/result.json`.
The candidate enables some exploration but does not establish reliable noisy
navigation. Current work tests noise-tolerant floor candidate selection on the
two failed recordings, described in
`docs/go2_robust_height_floor_candidates_2026-09-15.md`. No new native candidate
has launched.
# Post-panorama frontier exclusion diagnosis

`scripts/probe_go2_completed_frontier_exclusion_development.py` reconstructed
the layout-2 map through frame 620, reproducing all 156 saved floor/fine-obstacle
count witnesses from actual delivered noisy packets and fixed recorded public
poses. The first no-frontier plan was frame 624, immediately after the first
standoff panorama completed and excluded five viewed frontier cells.

With those saved exclusions the base fine-obstacle proposer reproduces
`OBSERVED_COMPONENT_HAS_NO_FRONTIER`. Removing exclusions only proposes the
same previously viewed cell [18,-3] at (0.925,-0.125), with a one-cell reachable
component and eight unknown connector cells. Thus simply clearing the visited
set does not demonstrate new explored space or resolved floor connectivity;
it would repropose the old target. No controller, new commands, native physics,
or counterfactual navigation were executed. The complete diagnostic is saved
as `completed_frontier_exclusion_probe_v1.json` under mapping layout 2. Keep
the ongoing gyro-height-floor live study unchanged.
