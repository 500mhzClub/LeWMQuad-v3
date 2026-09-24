# Declared floor-pose gap in live navigation

The natural-noise reacquisition trial on reactive layout 3 completed a verified
round trip, but no floor measurement was rejected. Its success therefore did
not test live holding or reacquisition. The recorded predecessor replay did
reproduce rejection on frames 405–408, and the runtime's four focused tests
passed.

Fix two separate development fault-injection probes: stronger reactive on layout
3 and the learned JEPA controller on layout 0, one run each. These are mechanism
tests on different layouts, not a matched comparison between controllers. Both
withhold floor-registration poses on the declared frames 405–408. RGB/gyro
tracking and independent raw-depth obstacle observation continue. This is a
floor-pose publication outage, not simulated loss of all camera data or calibrated
hardware uncertainty. Do not add these outcomes to the natural-noise cohort.

`DeclaredFloorGapRegistration` validates incoming frames using the existing
registration implementation, consumes their clock slots, returns no floor pose,
and retains the pre-gap anchor/reference. Withheld poses cannot become hidden
anchors for later registration. A focused test passed this state behavior and
resumption on the next frame. No thresholds, raw pixels, commands, planner,
model, correction, map, mission budget or physical guards change.

Use `scripts/run_go2_declared_floor_gap_noise_development.py --arm reactive`
on CPUs 8–15,24–31 and `--arm learned` on CPUs 0–7,16–23. At most two owners,
including archives, may coexist. Both have the same four-frame reacquisition
history requirement; the learned probe additionally exercises rebuilding the
motion-correction model's actual four-pose history before prediction resumes.
No native owners were active when preparing this pair. Approximately 23 GiB
free storage and the existing two-owner RAM envelope are adequate.

Roots:
`go2_declared_floor_gap_<arm>_noise_2mm_native_layoutXX_4800_v1_attempt_001`.
After owner exit, verify the four declared missing frames, omitted pose/map
updates, reset arrival dwell, actual zero commands from rejection publication
until reacquisition, exclusion of old in-flight plans, and at least four
consecutive accepted poses before planning resumes. Independently evaluate
arrivals, contacts and full navigation outcome. Preserve failures without
replacement. Passing a command-gap check alone does not establish navigation
reliability, a JEPA advantage, or hardware validity.

## Completed outcomes

Both native owners have exited. The learned layout-0 probe exited 0 after
447.18 s including archive, max RSS 17,564,368 KiB, zero swaps. Independent
evaluation verifies goal frame 1993 and home frame 3164, maximum dwell distances
11.485/21.858 mm, quiet speeds 0.01840/0.01252 m/s and all dwell requests zero.
No contacts or pipeline faults. Of 3,166 camera pairs, 3,162 poses were published:
exactly frames 405–408 were withheld. Median/max pose error 4.152/8.404 mm;
path 23.753 m; selected/on-time plans 784/775.

The learned gap mechanism passed. Rejection published at 42.112 s; reacquisition
published at 42.828 s after accepted frames 409–412. All 36 request samples
strictly inside this interval were zero (exact timestamp ties excluded). No
withheld pose reached mapping or arrival history, no plan committed during the
hold, and no old observation plan committed after rejection. First subsequent
nonzero request: 43.000 s. All 784 correction records used complete four-pose
histories, including 683 after the gap; none used a withheld frame. Evidence:
`floor_gap_execution_diagnostic_v1.json`. The command/pose timeline is saved as
`declared_floor_gap_command_recovery_v1.png` and `.svg` and was visually inspected.
This is end-to-end fault-injection evidence, not calibrated sensor loss or a
controlled comparison with the earlier trajectory.

Reactive layout 3 exited 1 after 287.98 s including archive, max RSS 11,368,784
KiB, zero swaps. It recorded 1,861 camera pairs and 681 registered poses, no
arrivals, zero contacts, median/max pose error 3.411/9.984 mm and path 2.810 m.
Besides the four declared frames, there were 1,171 natural floor rejections.
Checked waiting intervals issued zero commands, with no pose/map/arrival
invariant violations. Eighteen episodes reached four accepted poses (first at
frame 629), but movement never resumed; last nonzero request was at 41.940 s.
The probe fails recovery-to-execution and navigation. Terminal pipeline failure
was visual tracking unavailable. No replacement trial was made.

`scripts/diagnose_go2_declared_gap_tracking_failure_development.py` reproduced
tracking failure at frame 1856 and matched all 681 published raw-pose witnesses
exactly. Replay took 171.02 s on CPU 8 using original noisy sensor packets,
without navigation or physics. Evidence is in the reactive root under
`terminal_tracking_replay_v1/`. Its `terminal_raw_snapshot.json` contains the
actual terminal failure chain; the generic `failure` key is absent, so
`result.json` records that key as null. The separate
`terminal_tracking_reason_readout_v1.json` records the actual reason. At frame
1856, primary depth has zero valid pixels and auxiliary depth has 149,398.
Some auxiliary direct/chained pairs are marked geometrically qualified, but
higher-level pose admission failed. The exact later candidate/temporal rejection
remains unresolved; do not conclude that every auxiliary correspondence was
unusable. Next diagnose the outer candidate rejection before selecting another
perception change.

These probes remain separate from the completed natural-noise matched cohort.
No native or replay jobs remain active. The larger training comparison is
prepared but unstarted.

## Completed outer-candidate diagnosis

The additional replay in `terminal_candidate_trace_v1/` reproduces failure at
1856 and all 681 published raw poses exactly (166.93 s). At frame 1855 four
auxiliary candidates pass outer admission. At 1856 no candidate passes:
20 auxiliary attempts fail the gyro consensus minimum-match/strict-majority
requirement, 12 lack rigid matches, and four joint attempts lack three inliers
from each view. Primary attempts likewise fail rigid or joint-camera support.
Thus initial fit qualification did not imply a usable final pose. The trace
does not establish a numerical bug or justify relaxing admission thresholds;
it also does not yet distinguish minimum count from majority loss inside the
combined gyro-consensus rejection. Recovery-to-movement remains unresolved.

The frozen training-method comparison can measure the current system with this
known limitation. Proceed with its unchanged pipeline and count failures,
keeping these fault-injection probes separate. No further perception change
is included in that comparison.

## Why pose reacquisition did not restore dispatch

Postprocessing the completed reactive recording found 35 selected left-turn
plans after the first reacquisition, 33 committed. Actual requests from 64.600 s
onward contain 5,372 floor-reacquisition holds, 420 latched command-window vetoes,
324 missing-plan holds, 29 unavailable/stale-observation vetoes and one terminal
pipeline hold. No motion resumed. Thus the selector did propose motion after
reacquisition; obstacle dispatch was a separate obstacle to execution.

Twelve sampled original noisy packet pairs (frames 404,405,628,629,630,632–636,
1852,1856) have zero primary valid pixels and substantial auxiliary returns.
Eleven of these have an accepted independent floor plane. The independent
observer nevertheless returns None whenever either camera's raw point cloud
is empty. The packet reconstruction verified both delivered noisy digests.
This is a targeted sample diagnosis, not a full framewise observer replay.
Evidence: `post_reacquisition_dispatch_diagnostic_v1.json`.

Raw native optical-depth inspection, used only for diagnosis, identifies a
near-range blind spot: all 307,200 primary pixels are positive but below the
declared 0.2 m minimum at frames 404,632,1852. Their min/median/max depths are
0.155/0.172/0.194 m, 0.146/0.164/0.186 m and 0.142/0.161/0.186 m respectively.
Evidence: `primary_camera_near_range_diagnostic_v1.json`. These invalid depths
remain excluded from the controller. A subsequent prospective intervention
should test preserving usable camera views before entering this configuration;
floor-pose reacquisition alone cannot remove the independent-observer veto.
The frozen training comparison is unchanged.

The saved panorama starts at frame 352 with stored obstacle clearance 0.549 m.
The last nonzero translation request is at 36.980 s, after that observation
(36.700 s) because of the unchanged planning delay. Native horizontal motion
from frame 352 reaches 59.3 mm at 356, 86.3 mm at 376 and 102.2 mm at 404.
Stored clearance falls to 0.495 m at 376 and 0.488 m at 404. These are measured
recorded motions; they do not establish a counterfactual success. For an
upright camera facing a perpendicular flat wall, the 0.326-m forward mounting
offset plus 0.200-m minimum optical depth gives a 0.526-m body-distance near
clip boundary. This simplified calculation is not a general visibility or
collision certificate. Evidence: `panorama_near_range_motion_diagnostic_v1.json`.

The primary view is already entirely below range at sampled frame 404, before
the declared 405–408 publication gap. Consequently the injected gap alone
cannot be credited as the cause of the blind spot. A concrete later scientific
test is whether panorama placement that accounts for remaining committed
translation and measured turn motion preserves usable views. Keep the six
actions, valid-depth range and physical guards intact and evaluate actual
navigation, including failures. This is a follow-up hypothesis, not yet an
implemented or successful intervention.
