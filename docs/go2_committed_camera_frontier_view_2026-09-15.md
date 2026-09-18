# Complete a started frontier viewing turn

## Completed learned-motion follow-ups

Both fixed learned follow-ups have exited with floor-registration conflicts;
all four standard evaluations are saved for each. Neither completed a round
trip, and both had zero disallowed contacts. The original four-maze totals
remain unchanged because these are exposed-layout development follow-ups.

- Layout 2: 3,956 accepted poses from 3,958 captured frames. Its goal at frame
  3088 passed the physical quiet-dwell check (maximum distance 36.908 mm),
  improving on the original no-goal budget exhaustion. The return stopped on
  floor registration. Median/maximum position errors were 8.757/32.817 mm;
  957/982 plans were on time. Wall time including recording was 9:27.61.
- Layout 3: 3,591 accepted poses from 3,593 captured frames. Its observed goal
  at frame 3472 is invalid: physical quiet-dwell distances were 51.482–62.578 mm,
  outside the unchanged 40 mm criterion. Median/maximum pose errors were
  39.716/47.400 mm; 851/893 plans were on time. Wall time was 9:03.35.

Both used learned corrected XY and learned yaw with the original tracker.
Pair reports and inspected PNG/SVG figures are saved under
`go2_committed_camera_view_learned_layout02_summary_v1_attempt_001` and the
corresponding layout-3 root. Each pair has 145 identical common source bindings
and matching shared settings; final scored learned XY/yaw channels were checked.
Layout-2 turn-only command time fell from 434.98 to 301.44 seconds, but its
longest completed view event increased from 58.4 to 100.8 seconds. Layout-3
turn-only time increased from 234.70 to 249.76 seconds and its longest event
from 138.8 to 177.2 seconds. The viewing change therefore does not uniformly
shorten viewing tasks or resolve navigation failures.

Their saved executed-window XY RMSEs were 5.745/6.306 mm (942/826 overlapping
windows), while learned yaw RMSEs were 4.372/3.996 degrees versus command-yaw
alternatives 3.089/2.737 degrees. These conditional forecast measurements do
not establish alternative navigation outcomes. Both recordings are retained.

## Verified native follow-up

The one exposed-layout follow-up completed a physically verified round trip:
2,298 accepted poses, goal frame 1577 and home frame 2296, zero disallowed
contacts. Goal/home quiet-dwell maximum distances were 13.686/24.893 mm;
median/maximum pose errors were 1.641/6.275 mm. It travelled 26.251 m and had
536 of 553 selected plans on time. Owner wall time including recording was
5:44.45, with no swap activity. All four standard evaluations are saved.

The original control reached the goal but exhausted the 480-second mission
budget before completing the return dwell. The follow-up finished in about
230 simulated seconds. Its 18 completed view tasks all ended with actual mapped
patch observations; the longest lasted 36.4 seconds, versus 212.4 seconds in the
original. The original target cell (21,20) was not itself a requested event in
the new trajectory, so the comparison must not claim an exact same-task
counterfactual. Both pose drift and elapsed navigation time improved on this
exposed follow-up; repeatability and fresh-layout transfer remain unproven.

The paired report and PNG/SVG are under
`go2_committed_camera_view_layout01_summary_v1_attempt_001`.
Common source and shared mission/sensing/model/fit settings were checked; the
original failed outcome is retained. Historical host load and asynchronous
trajectories differ. Keep both full recordings for diagnosis.

There are 145 matching common source bindings. Turn-only completed command
time decreased from 264.42 to 61.64 seconds; translation command time decreased
from 168.22 to 142.56 seconds. Three of the follow-up's 18 completed view events
actually entered the committed-turn branch. The other patches were observed
before a committed viewing turn was needed.

Next, apply the same fixed viewing change to the failed learned-motion layouts
2 and 3 as exposed development follow-ups. Neither will change the original
four-maze learned/control totals. Keep the tracker unchanged in those native
follow-ups so any concurrent offline reference-retention experiment remains
separate.

The two learned-motion follow-ups are now launched in parallel via
`scripts/run_go2_committed_camera_view_learned_followups_development.py`:
layout 2 in session 39254 on CPUs 0–7,16–23, and layout 3 in session 98753 on
CPUs 8–15,24–31. Roots are
`go2_committed_camera_view_learned_noise_2mm_native_layout02_4800_v1_attempt_001`
and the corresponding `layout03` root. Each uses learned corrected XY and
learned yaw, disabled contact scoring, the same fixed maze/sensors/physical
limits, and the original accepted-reference-refresh tracker. The viewing helper
is identical to the successful control follow-up. No local-view reference bank
is used in these native tests. Their two-assignment annotations were checked
before launch; no outcome is available yet. Retain both outcomes without tuning.

Their actual launch records confirm learned XY/yaw, committed views, original
accepted-reference refresh and two planned assignments. Native owner PIDs are
3638756 (layout 2) and 3638757 (layout 3). Both were observed advancing past
1,000 camera frames; no completed outcome is yet available.

## Diagnosis and implementation history

The combined-perception pose/command controller on fresh development layout 1
spent 212.4 seconds pursuing one camera view, then reached the mission goal too
late to complete the return dwell within its unchanged budget. The original
failure remains in the eight-assignment comparison.

`scripts/diagnose_go2_long_camera_view_development.py` replayed 671 actual mapping
updates from that retained recording and matched the map-count receipts for all
531 plans during the episode. It used the recorded accepted visual poses and
reconstructed delivered noisy depth. It did not feed native state to the map or
policy, revalidate tracking, or execute an alternative trajectory.

Results are in the ordinary artifact root
`go2_combined_perception_motion_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001/long_camera_view_geometry_replay_v1/`.

- The viewpoint was (1.375,0.625); the requested unknown cell was (21,20).
- The view/approach condition switched 53 times. There were 107 eligible view
  plans, 82 close-enough plans whose directed camera projection was infeasible,
  and 342 plans outside the 0.10 m arrival radius.
- All 107 recorded directed-view statuses matched the reconstructed view
  condition. The remaining plans were approach or translation-veto recovery.
- Distance to the chosen viewpoint ranged from 38.7 to 257.0 mm. No eligible
  view plan reached the 0.10 rad alignment tolerance; only one plan's actual
  camera projection fully contained the patch.
- The task eventually completed when the requested patch became observed.
  No unknown floor was admitted from a projection hypothesis.

This identifies repeated interruptions of the viewing turn. It does not prove
that fixing that transition alone will solve the run: obstacle recovery and
predicted-clearance turn selection can also affect the executed heading.

The separate successor source
`lewm/committed_camera_frontier_view_development.py` starts views under the
original conditions, then retains the view task while turning. It no longer
returns to approach solely because the body leaves the arrival radius or a
hypothetical directed projection changes during the turn. Once the measured
heading aligns, it checks the actual camera projection. If the patch falls
outside the actual image, it marks that viewpoint attempted and chooses another
known-floor viewpoint. If visible, it waits for fresh mapped evidence; an
unresolved patch remains unknown. Actual mapped observation and an available
mission-goal route still end the task immediately. Physical dispatch guards,
prediction filters, sensing, models and correction fits are unchanged.

Five focused tests passed in 2.89 s: body drift during the turn, aligned but blind
view retry, actual mapped-patch completion, fresh-map waiting without unknown
admission, and mission-goal priority.

All eight frozen combined-perception assignments are now complete. The single
follow-up launched on the exposed layout-1 pose/command case, retaining its
original failure and all diagnostics. Session 56522, owner PID 3636339, has now
exited successfully; its verified outcome is reported above.
Success on that development follow-up would need a subsequent fresh-layout
comparison; it would not establish learned-model benefit or broad reliability.

The single-follow-up launcher is ready:
`scripts/run_go2_committed_camera_view_development.py`.
It requires the completed eight-run aggregate, uses the original layout-1 CPU
group (8–15,24–31), and writes
`go2_committed_camera_view_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001`.
Import/MRO and an in-memory launch-annotation check passed: one exposed-layout
assignment, pose/command XY, command yaw, accepted-reference refresh and the
committed-view change. No simulation or output root was created by those checks.
The launcher retains the exact combined-cohort layout/physics/appearance seeds,
synthetic depth noise, model/fits, action bank, timing and physical criteria.

The actual launch record confirms one layout-1 assignment, pose/command XY,
command yaw, accepted-reference refresh, the committed viewing turn, and the
correct original reference root. After this owner exits including recording,
run the standard physical arrival/navigation/XY/yaw evaluations. Compare its
view events and approach/view/recovery durations with the original 212.4-second
episode. Shared runtime/model settings remain fixed, but historical host load
and the resulting asynchronous trajectory need not be identical; one exposed
follow-up does not establish a general causal or reliability claim.
