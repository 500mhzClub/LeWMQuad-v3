# Camera-aware frontier viewing positions

The learned-yaw layout-0 failure retained a diagonal gap in the observed floor
near its upper passage. Saved registered-pose and calibration projection shows
that cells (11,52), (12,52), (13,52) never fit wholly inside either camera image
during the recorded standoff and close viewing manoeuvres. Closer approach did
not observe the floor beneath and near the forward-mounted cameras.

`lewm/camera_frontier_viewpoint_development.py` proposes a viewing position along
an already computed observed-floor route. It projects the requested unknown
floor square using both calibrated cameras, current measured height/roll/pitch
and a directed candidate yaw. It selects the route point nearest the frontier
with full image/range coverage and no known coarse obstacle along the camera's
horizontal sight line. Projection is only a hypothesis: it neither fills the
map nor authorizes motion or certifies true visibility. A new observation and
the existing action-clearance checks remain necessary.

On the saved upper-frontier revisit routes at frames 3180 and 3636, it selects
(0.075, 2.525) to observe cell (11,52), approximately 0.43 m behind the prior
close-view position. The original camera-failure diagnosis and the separate
query receipts remain in the learned-yaw layout-0 root:
`camera_frontier_upper_revisit_queries_v1.json`. These retrospective queries
explicitly remove the upper exclusions; they are not navigation outcomes.
A first query using the original unexcluded-target selection found no candidate
because that route selected another frontier or was already too close; that
readout also remains preserved.

Three focused tests passed in 0.09 s: near-body/behind-camera invisibility,
observed-route preservation with no unknown-floor admission, and known-occluder
rejection. The helper is not yet connected to the live controller. Next work
must connect route-to-viewpoint, actual arrival/current projection, acquired
floor evidence, and revisit handling without permanently discarding unseen
patches merely because a turn completed. Test that behaviour on saved-map
scenarios, then in a separate live development probe. Do not alter the active
reference-refresh intervention while it runs.

## Controller integration and fixed live probe

`lewm/camera_frontier_visits_development.py` now routes to the selected viewpoint
and dispatches a directed turn through the existing frontier-view path. The
current measured pose must project the requested floor patch before a fresh
mapped view is evaluated. Only actual mapped floor or obstacle membership
resolves that patch. A fresh unsuccessful view records the tried position and
permits another viewpoint. If no candidate remains, the frontier's temporary
search exclusion clears when mapped floor/obstacle counts grow; it is not a
claim that the unknown patch was observed. Already-too-close cases search up
to 128 nearby observed-floor candidates within 1.25 m of the patch and require
the original observed-route proposer to connect them. All action-clearance,
command and physical guards remain active.

Seven focused geometry/state tests passed in 1.70 s, including fresh-view
failure without unknown-floor admission, actual mapped-patch completion,
known-floor backtracking from the near-body blind region and mission-goal
route priority. Saved-state queries at frames 640, 808, 1364 and 3180 all return
routes to camera viewpoints, taking 15.9–37.4 ms each in this small diagnostic.
These are state-level queries, not full policy replay or a wall-time guarantee.

Fix one live development follow-up on the original stalled layout 0: learned
yaw, pose/command XY, disabled contact, original coherent tracker (without the
separate reference-refresh treatment), same model/fit/noise/action/physics and
4,800-tick budget. Change the frontier viewing policy only. Launcher:
`scripts/run_go2_camera_frontier_viewpoint_development.py`; root:
`go2_camera_frontier_viewpoint_learned_yaw_noise_2mm_native_layout00_4800_v1_attempt_001`.
Preserve the original stall and the full new result, including any failure.
A success would support this development intervention, not general reliability
or a world-model training advantage.

The final route substitution also preserves the new connector's entry point,
unknown-cell list and observation requirement. The four visit-state tests
repassed in 1.68 s after that receipt correction. The single native probe
launched in session 87084, owner PID 3626118, on layout 0's original CPU group
(0–7,16–23). Actual launch metadata confirms learned yaw, pose/command XY,
disabled contact, camera-selected viewpoints, fresh mapped-patch completion,
the original coherent tracker and one fixed native assignment. The worker is
live at camera frame 100. No native navigation outcome is available yet.
Wait for owner exit including archive before independent arrival/XY/yaw
assessment. Keep all full inputs and any failure for diagnosis.

## Verified physical goal, return and mapped patch evidence

The native owner exited 0 after 250.18 s including archive, peak RSS
10,725,796 KiB, zero swaps. All 1,706 camera pairs have registered poses.
Independent physical evaluation verifies goal frame 1012 and home frame 1704,
maximum dwell distances 18.941/18.116 mm and maximum 100 ms speeds
0.021487/0.009813 m/s, with all commands zero during both one-second dwells.
Zero disallowed contacts. Position error median/max is 1.561/4.473 mm, final
native home distance 18.113 mm, travelled path 21.638 m. The original control
exhausted its 480-second budget without reaching the goal.

There are 420 selected plans, 413 on time and seven late. Maximum host/simulation
lag is 371.030 ms; maximum acquisition time 214.648 ms. The 407 executed windows
have XY RMSE 7.104 mm, maximum 26.621 mm, none above 30 mm. Applied learned-yaw
endpoint RMSE is 3.374 degrees versus 1.770 degrees for the command alternative
on the same windows. This remains measured-simulation evidence.

All fifteen requested-patch completion events report observed patches; there
are no unresolved fresh-view events and no excluded frontier targets. A separate
replay in `camera_frontier_observation_replay_v1/` consumes all 427 recorded
mapping updates with the delivered noisy packets and saved registered poses.
It matches all 420 planning-map count receipts and confirms all fifteen requested
patches in observed floor (none in obstacles). Elapsed replay time is 48.676 s.
This verifies actual map membership, not hypothetical projection as free space.
Patches may become observed while approaching a viewpoint, before any fixed
turn sequence completes. The replay does not revalidate raw tracking.

Comparison:
`go2_camera_frontier_viewpoint_layout00_summary_v1_attempt_001/result.json`.
The 138 common predecessor runtime sources match; model, fitted XY, learned
yaw, disabled contact, original tracker and physical settings remain fixed.
The new policy jointly changes viewpoint selection, directed viewing and
unresolved-patch handling. This one exposed-layout follow-up does not isolate
which subchange was necessary or demonstrate general reliability. The original
stall and both full sensor recordings remain. The comparison PNG/SVG has been
rendered and visually inspected. No native process remains from this probe.

The next scientific step is to combine the independently tested reference-refresh
and camera-viewing changes, then evaluate fresh development mazes with frozen
learned-motion and simpler motion-prediction controls. Preserve all outcomes and
keep the learning, planning and memory claims separate. Multiple training seeds,
real-time qualification and hardware evidence are still outstanding.
