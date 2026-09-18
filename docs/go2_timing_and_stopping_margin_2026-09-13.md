# Timing and stopping-distance development

The completed continuous mission in attempt 005 failed to reach the goal.
Its pose estimates were accurate on the recorded trajectory, but usually too
late for planning. Its conservative circular obstacle check also left it
unable to execute the requested turn after approaching a wall.

## Timing experiments

`go2_feature100_150_201frames_recorded_v1_attempt_001` compared both trackers
on the same 201 recorded independent-maze frames, alternating execution order.
Both accepted every frame. The 100-feature arm used 13.121 seconds versus
14.008 seconds after the first three frames, a 6.33% reduction. Median service
was 63.44 versus 69.14 ms. Maximum position error was 7.53 versus 5.17 mm;
median was 6.10 versus 4.22 mm. Native poses were read after estimation ended.
The candidate retains at most nine features per image cell and 100 total;
the constructor may compute 108 descriptors before trimming. Pose-admission
thresholds remain unchanged. Full-journey robustness is not established.

`go2_continuous_round_trip_native_layout00_v1_attempt_006` tried local tracking
with the existing 150-feature estimator, eliminating camera-array process
transfers. It failed after 115 acquisitions and 573 policy services when the
tracking input queue filled. Tracking averaged 139.20 ms over 82 frames.
This was worse than separate-process tracking; the attempt is preserved and
local tracking is not adopted.

Attempt 007 completed its prospective 601-frame test of the 100-feature tracker
in its separate process. It retained attempt 005's original clearance rule,
separating the timing treatment from the stopping change. There were 117
on-time and 33 late plans, with 260 nonzero services and no contact or tracker
failure. Tracking averaged 86.94 ms, with 460 ms 95th-percentile age. The
robot still stalled: XY displacement was 0.5264 m and there were no arrivals.
All 601 registered poses had median/max native position error 8.52/10.32 mm.
The timed interval was 60.43 host seconds and 60.54 simulated seconds including
zero draining. The process exited zero with all 601 image pairs saved.
Result SHA-256: `04a99d9c1640bdb6cb1c27649de2385b5cdf7d953ef03974e3d5453fcb88ca5d`.

## Stopping hypothesis

At the first transition from forward to zero in attempt 005, simulation time
6.6 s including settling, native speed was 0.2303 m/s. The robot travelled
up to 2.84 cm in XY during the following second and first fell below 0.05 m/s
after 188 ms. These post-estimation measurements diagnose the controller;
they are not fed back into its sensor packets.

The existing nominal check projects only the remainder of the requested
command from the latest depth observation. It omits displacement during the
observation's age and after a stop request. The next development guard extends
the requested-speed projection by the observation age plus a fixed 0.5 s
stopping allowance. It retains the same 0.45 m circle and every original
freshness and obstacle veto. The allowance is a prospective development
hypothesis, not a calibrated physical stopping bound. It does not claim
whole-body or unknown-space clearance.

Two focused tests passed in 1.88 s: the extended check stops a translating
command earlier while retaining room to turn, and preserves the old blocked
turn/freshness behavior and the command-history layer. No native success from
the stopping change had been observed at that point. The full-budget prospective
test in `go2_stopping_margin_round_trip_native_layout00_v1_attempt_001`, session
57398, subsequently completed and exited zero with all data persisted. It used
the 100-feature tracker in its separate process. There were 1,805 frames,
420 on-time and 30 late plans, 393 nonzero services, no contact and no arrivals.
The navigation budget expired. XY displacement was 0.6450 m; all registered
poses had median/max position error 7.78/8.94 mm. The margin alone was insufficient.

The first margin veto at 7.5 s stopped translation while the original projected
connector still had 0.5061 m clearance. The extended connector had 0.4061 m.
The planner nevertheless repeatedly selected the same forward action toward
a nominal floor frontier, instead of requesting a new view. It resumed forward
at 38.2 s, moved closer to the obstacle, and later turn requests were vetoed.
This isolates a missing feedback path from action rejection to exploration,
in addition to any limitations of the circular footprint or stopping allowance.

The next runtime requests a measured 45-degree left observation turn after an
actual translation veto. It uses the existing model-scored hold/left/right
viewing behavior. The recovery remains active until a post-veto visual pose
shows the target heading within the existing 0.1 rad tolerance. It retains map,
tracker, model, command-prefix accounting and every obstacle veto. One focused
test passed in 1.91 s, including no recovery completion from a pre-veto pose and
no mutation of earlier request receipts. The full-budget experiment is launched
in `go2_veto_view_round_trip_native_layout00_v1_attempt_001`, session 24669.
That run also completed the budget without arrival or contact, and exited
zero after saving all 1,805 image pairs. It had 430 on-time and 20 late plans,
521 nonzero services, and 394 first obstacle vetoes. The measured-view recovery
did trigger, but turns remained blocked. This is another negative result.

Inspection of the actual camera images and saved depth isolated a discretization
effect. At frame 90 the nearest sampled obstacle return was 0.5021 m from the
body origin, but its 5 cm cell extended to 0.4472 m, tripping the 0.45 m circle.
At frame 96 the corresponding distances were 0.4951 and 0.4472 m. Rebinning the
same points into 1 cm cells gave 0.4924 m in both cases. The diagnostic is saved
as `near_obstacle_quantization_diagnostic.json` in the veto-view run root.

The next candidate uses 1 cm cells only for the independent current-depth veto;
the historical routing map remains unchanged. The nominal circle remains 0.45 m.
The fine grid retains cells inside the body-centred two-metre square. Its
connector rejects endpoints beyond 0.5 m from the origin, so that square covers
the complete supported connector plus the 0.45 m circle. Original freshness,
plane and observed-point rules remain in force. This refines the numerical
representation of observed obstacles; it is not a whole-body clearance proof.

A focused test passed in 1.84 s: the finer cells resolve the observed aliasing
case, still veto closer returns, reject incompatible grid frames and reject
connectors outside the cropped domain. The new full-budget native test is
`go2_fine_obstacle_round_trip_native_layout00_v1_attempt_001`, session 7042.
It exited with registration failure after 193 acquisitions and 965 policy
services, with no contact or arrivals. There were 45 on-time plans out of 47.
No original near-obstacle veto occurred, but current obstacle observations became
unavailable when the visible floor lost two-axis extent. Registration then
rejected a current-point conflict with its transported floor reference.
The last admitted pose was frame 190; maximum admitted position error was
10.62 mm. XY displacement was 0.4974 m. All 193 image pairs were saved.
Failure SHA-256: `fc2ca482221537a167be164adb14b5e88efc993ec9f9e1b07d84ba9f6ebf55fd`.

Public-sensor replay of this exact trajectory found the same registration
failure for both trackers: 100 features at frame 191, 150 at frame 189.
Both retained a last full floor anchor from frame 82. The primary camera had
zero current floor candidates, while the auxiliary camera still had 4,109
(100-feature arm) or 4,106 (150-feature arm). The patch lacked two-axis spatial
extent. Maximum residual against the transported reference was 3.0021 mm
and 3.0476 mm respectively, versus the unchanged 3 mm limit.

The diagnostic therefore tests partial observability, not merely a larger
feature budget. For the 100-feature arm, a current scalar height update of
0.630 mm reduces maximum residual to 2.653 mm without changing the retained
normal; the corresponding 150-feature scalar update alone does not pass.
Separately, using the uninterrupted gyro-derived normal and fitting only the
height of the current points keeps obstacle observations available over all
193 frames, versus 83 under the complete-plane requirement. It retains the
3 mm residual limit and at least 100 samples. This is offline feasibility only:
the resulting normal is a gyro prior, not a newly measured full plane, and no
new native navigation or hardware qualification follows from this result.

The next science step is to integrate an explicitly partial floor-height
observation into both registration and the current-depth classifier, preserve
the distinction from a fully observed plane, replay the failed sequence, and
then run a prospective mission. Do not loosen the residual threshold or label
the existing failures as successful. All continuous native sessions above are
terminal; only the separate comparison queue remains active.

## Comparison queue

The original direct-model collection exhausted 8,000 navigation ticks without
arrival. Its later audit was interrupted at frame 2,429 by the old
`PHASE_DISK_ALLOWANCE` rule. That rule attributed filesystem-wide free-space
changes to this one audit; concurrent experiments could therefore trip its
8 GiB allowance despite hundreds of GiB of actual free space. The interrupted
audit and its failure remain intact and are not relabelled as verified.

For future runs, removed only this invalid attribution-based stop condition.
Actual free-space reserve, available RAM and worker RSS limits remain enforced.
Resource guard/audit tests passed (34 tests, 2.18 s). The next never-started
nominal comparator launched in queue V3, session 2509, followed by the two
contact-horizon cases, layout 1 JEPA/supervised, memory ablation and six-action
reactive case. This queue does not retry the interrupted direct-model audit.
