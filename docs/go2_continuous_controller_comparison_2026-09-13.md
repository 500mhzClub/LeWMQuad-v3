# Continuous controller comparison

The current learned controller has one verified layout-0 round trip and one
layout-1 speed-limit failure. Layout 2 is the next transfer experiment with
the same controller and frozen models. These are development results, not
sealed evaluation. Preserve unsuccessful attempts in the comparison.

The first comparison replaces learned action selection with current-waypoint
reactive feedback. Both arms must use the current paired-camera tracker,
registered floor/obstacle map, exact goal target, initial and frontier surveys,
frontier exclusion, route lookahead, mission dwells and independent current
depth command veto. Both retain six actions, 300 ms dispatch delay, 400 ms
commitment, actual-prefix checks, 100 ms cameras and 20 ms command service.
Charge each arm's actual computation; do not pad the reactive arm to match
neural inference time. Keep the same physical limits and navigation budget.

The reactive arm uses no world-model checkpoint, learned motion residual,
future outcome tensor or predicted-path clearance filter. It chooses the
nearest existing action to instantaneous measured waypoint/heading feedback,
with the existing nominal footprint test at the current measured position.
This compares the complete predictive action-selection method against a
reactive selection method with the same observed-map navigation machinery.
It does not isolate neural scoring from the predicted-path filter, and it
does not establish the contribution of persistent memory.

Record physical goal/home arrivals, contacts, speed-limit stops, perception
failures, travel time, deadline misses and exhausted budgets. Compare actual
native execution, not retrospective action scores. Layout-0 results are
development comparisons; broader reliability needs additional layouts and
repeat attempts under a fixed treatment.

Subsequent comparisons must separate prediction use, JEPA training and
memory. In particular, the current motion residual was fit for the JEPA
forecasts. Applying it unchanged to another predictor would not isolate the
training objective. Use an explicitly matched residual treatment or a
no-residual comparison when comparing predictor training methods. Realistic
sensing and hardware validation remain separate outstanding work.

## Frozen correction transfer evidence

The layout-1 failure retained 56 selected translation forecast windows whose
actual requested commands matched the forecast prefix through 700 ms.
Evaluation against native physical displacement gives 25.48 mm raw JEPA XY
RMSE and 9.26 mm corrected RMSE. The correction was frozen from layout-0
development data; no refit was performed. This is evidence that the correction
transfers to these executed motions, despite the later speed-limit stop.

`scripts/evaluate_saved_motion_forecasts_development.py` reads the online
forecast records and actual requests, admitting only matching prefixes for
each horizon. It does not rerun the predictor or evaluate counterfactual
actions. The report is `saved_motion_forecast_native_evaluation.json` in each
completed attempt root. It includes command-matched late plans and is a
forecast accuracy analysis, not a navigation-benefit comparison or a bound
on instantaneous physical speed.

## Layout 2 transfer

The unchanged learned controller exhausted 3,600 navigation ticks with no
arrivals, no physical stop and no disallowed contacts. There were 875 on-time
plans out of 900 (97.2%). Final observed goal distance was 1.701 m. The artifact
is `go2_indexed_geometry_precise_goal_round_trip_native_layout02_v1_attempt_001`.

The late run held because every forecast violated the nominal 0.45 m stored
obstacle clearance. At frame 644 the selected right turn still passed with
predicted minimum clearance 0.4511 m; at frame 648 all candidates failed.
The final estimated minimum was 0.4291 m, and the desired right turn remained
blocked. This was a physical drift and stopping problem, not merely an
unseen-obstacle map discontinuity: evaluator-only native wall distances
decreased from 0.5248 m at frame 600 to 0.4749 m at 644, 0.4528 m at 656 and
0.4413 m at 3600. Registered position errors stayed near 13 mm in this period.
See `native_stall_wall_distance_diagnostic.json`. These are centre-to-wall
distances; the nominal disk violation does not imply an actual robot contact.

Turning currently has no extra forecast error reserve. A future treatment
should investigate turn drift, stopping motion and relocation to a measured
clearer viewing position. Preserve this failure before changing those rules.
The reactive arm now shares the common planning scheduler and initial survey
through a selection hook and survey mixin. It constructs without a world
model or motion-residual fit; 12 focused tests pass. Its first native trial is
described below on development layout 0.

## First matched reactive execution

`go2_continuous_reactive_round_trip_native_layout00_v1_attempt_001` terminated
with visual tracking unavailable at frame 1039, after 1,041 camera pairs had
been acquired. The owner exited 1 and all camera pairs were saved. There
were 1,039 registered poses, no disallowed contacts and no arrivals. The last
registered outbound goal distance was 1.560 m; the last plan was a left turn
during the ninth heading of a frontier panorama. There were 249 on-time plans
out of 259 (96.1%). Independent native evaluation found median/maximum
position error 13.61/21.72 mm before tracking failed.

The launch records `model_assignment=reactive`, no world model, no motion
residual and no predicted-path filter. Actual action records confirm the
instantaneous selection rule. Thus the present layout-0 comparison is one
learned-controller round-trip success and one reactive perception failure.
The trajectories differ; one comparison does not establish general JEPA
advantage or isolate which prediction component mattered. Replaying the
preserved sensor sequence reproduced the failure exactly, with no raw-pose
differences on the accepted prefix.

The first conflict was a primary-camera fit from reference frame 1034 to
current frame 1039: 38 of 49 matched points were initial image inliers.
The floor-constrained solution kept 37 of those 38 inside the same image
gates. Requiring all 38 caused it to be discarded; the original image fit
then had a 3.0010275 mm floor-height residual against the 3 mm requirement.
Evaluator-only native comparison found original/constrained translation
errors 3.525/0.811 mm. Both measured planes agreed with the native floor to
less than one micrometre in height and 0.000007 rad in normal direction.
See `plane_image_conflict_diagnostic.json` and
`plane_image_conflict_native_evaluation.json`.

The explicit `plane_consensus_tracker_development.py` treatment tries a
constrained refit with monotonic inlier pruning after this height conflict.
It retains at least 12 points, a strict majority of the original matches,
both-camera support for pooled fits, and the existing absolute image,
conditioning, motion, gyro and continuity thresholds. Failed alternatives
preserve the original conflict. This changes the estimator's consensus rule;
it does not relax the floor-height threshold or retain the conflicting pose.
The saved conflicting pair converges in two rounds with 37/49 matches.
Full-sequence replay passed all 1,041 frames, with no differences in the
previously accepted raw-pose prefix and consensus refinement selected only
at frame 1039. Median/maximum registered position error was 13.62/21.87 mm.
The retained set occupies the same 9 image cells in both views of the primary
pair as the original set. A fresh reactive trial uses
`go2_plane_consensus_reactive_round_trip_native_layout00_v1_attempt_001` to
test navigation with the new estimator. Neither original failed attempt is
replaced. A learned-versus-reactive comparison under this new estimator also
requires a learned arm with the same treatment.

## Prepared turn-drift treatment

`lewm/clearance_turn_recovery_development.py` applies the existing 3 cm
translation reserve to turns as well. If the preferred pure turn is blocked
but its opposite is clear, it keeps that alternative direction until the
measured target heading is reached or crossed. It holds whenever the latched
turn's forecast is not clear and cancels the latch on a mission change.
This prevents immediately reversing toward the blocked shorter turn.
It is a separate planner treatment, not part of the reactive comparison.

Three focused tests cover directional persistence across angle wrap,
clearance rejection, target crossing and mission changes. Applied to the
saved layout-2 frame 624, it changes right turn to left turn, with no native
input. That counterfactual selection is recorded in
`turn_reserve_counterfactual_frame624.json`; it does not establish navigation
benefit or a calibrated turning/stopping bound. A prospective run remains
necessary.

## Full reactive run with plane consensus

The new reactive attempt completed 3,600 navigation ticks with no arrivals,
no tracking failure and no disallowed contacts. It completed six frontier
surveys, reached a minimum observed goal distance of 1.485 m, and finished
5.195 m from the goal. It did not simply remain stopped: 16,360 policy requests
were nonzero. There were 859 on-time plans out of 897 (95.8%). Timed wall and
simulated durations were 361.35 and 360.98 seconds. Camera persistence and
the independent arrival evaluation follow timed execution.

The matching learned arm is
`go2_plane_consensus_learned_round_trip_native_layout00_v1_attempt_001`, using
the same plane-consensus estimator and original predictive controller, without
the prepared turn-recovery treatment. It starts while the completed reactive
run saves images: 48 GiB memory was available on a 32-logical-CPU host, and no
reactive physics or perception workers remained active. Both arms record
measured service costs; timing remains shared-host development evidence.

Both owners subsequently exited 0 with all 3,605 camera pairs and poses saved
per arm. Independent arrival checks found no confirmed arrivals in either.
The learned arm first found a goal route at frame 3336 and reached a minimum
native goal distance of 23.81 mm, but did not establish a stopped arrival and
return before the budget expired. Reactive minimum native goal distance was
1.465 m, and its final distance was 5.190 m. Native 10 Hz trajectory lengths
were 12.94 m learned and 20.59 m reactive. Registered position median/maximum
errors were 6.10/15.32 mm learned and 32.45/80.64 mm reactive. Learned plans
were on time for 768/900 decisions, compared with reactive 859/897.

The comparison artifact is
`go2_plane_consensus_controller_comparison_layout00_v1_attempt_001`, with
`result.json` and native trajectory figures in PNG and SVG. This is one
development pair: better goal progress does not establish completed missions,
repeatability, JEPA-specific benefit or unseen-layout reliability. The
original earlier successful round trip remains valid, but was not repeated
by this attempt within the budget.

The next active native experiment is
`go2_clearance_turn_recovery_learned_round_trip_native_layout02_v1_attempt_001`.
It tests the prepared turn reserve and direction latch, with the plane
consensus estimator and the same 3,600-tick budget. It starts after the
comparison owners exit, without overlapping image persistence. Future
attempts use lossless compression level 1 for camera archives: a ten-depth-
frame sample took 0.264 s versus 0.411 s at the old setting, with 5.7% more
bytes and byte-identical decoded arrays. This only changes post-run saving.

## Turn recovery outcome and next treatment

The first turn-recovery attempt exited 0 with all 3,605 pairs and poses saved,
no contacts and no arrivals. It exhausted the budget while holding about
1.76 m from the goal. Plans were on time for 832/900 decisions; maximum
registered position error was 16.74 mm. A right-turn latch began at frame
704, but later became infeasible. At the final planning observation, the
latched right turn predicted minimum clearance 0.4675 m. The opposite left
turn preserved the 0.4708 m prefix clearance and improved it to 0.4752 m,
but could not restore the entire 0.48 m reserve inside the short horizon.

`StepwiseClearanceTurnRecoveryRuntime` is the next explicit treatment. Within
an existing turn-reserve deficit, it allows a turn only when the entire
forecast remains above the nominal 0.45 m footprint, the controllable tail
does not reduce prefix clearance, and terminal clearance improves by at
least 1 mm or 10% of the remaining deficit, whichever is larger. A blocked
latch may switch to such an improving opposite turn while preserving its
measured target heading. The point forecasts are not calibrated bounds.

Five focused tests pass. The saved final stalled decision changes from hold
to left turn in `stepwise_recovery_counterfactual.json`, with no native input.
The fresh native attempt is
`go2_stepwise_clearance_turn_recovery_learned_round_trip_native_layout02_v1_attempt_001`.
It keeps the same physical limits, estimator and 3,600-tick navigation budget.

The stepwise attempt has now finished and its owner is absent. It saved all
3,605 camera pairs and registered poses. Independent evaluation found zero
disallowed contact samples and no arrivals; the 3,600-tick budget expired
about 1.58 m from the outbound goal. There were 856/900 on-time plans (95.1%).
Median/maximum registered position errors were 15.17/15.40 mm. Allowing
incremental predicted clearance recovery did not resolve the navigation stall.

## Clearance-preferring route treatment

The next treatment changes the path to the existing observed route target.
It applies four-neighbour A* to the same traversable floor graph, with a soft
penalty for proximity to observed occupied cells. Preferred clearance is
0.60 m; the cost is 1 + 2 * (max(0, 0.60 - distance) / 0.15)^2, where distance
is the centre-to-occupied-centre Euclidean distance less a coarse-cell half
diagonal, clamped at zero. Entry selection, frontier exclusions, route target,
unknown connector accounting and exact action clearance checks are preserved.
The original 0.48 m shortcut lookahead cap is unchanged. No extra space is
inferred free. Runtime planning time includes the added routing work.

`clearance_preferred_routes_diagnostic.json` in the original indexed-geometry
layout-2 root reconstructs maps from recorded public paired depth and recorded
estimator poses. It does not read native truth. At planning frame 400, path
minimum exact fine-obstacle clearance rises from 0.4962 to 0.5652 m and median
clearance from 0.5550 to 0.6239 m, for 0.10 m additional path length. Added A*
time was 1.39 ms. Across frames 400/500/600/624 it took 0.40–1.43 ms. Late
snapshots already start near the wall, so their minimum clearance cannot
improve simply by replacing the remaining path. This supports prevention
before reaching the tight pose, not successful escape from it. The diagnostic
uses the same entry and target within each pair but does not replay historical
frontier exclusions; it is not an exact replay of every online route decision.

The prospective native arm is
`go2_clearance_preferred_route_learned_round_trip_native_layout02_v1_attempt_001`.
It adds this path preference to the stepwise turn-recovery controller, keeping
the estimator, frozen world model/residual, action set and 3,600-tick budget.

Eight focused routing/turn tests passed before launch, including runtime
frontier exclusion and connector-receipt preservation. Available resources
were 77 GiB RAM and 276 GiB disk; one native owner was launched, without a
second simulator competing for those resources.

A subsequent read of the completed stepwise arm identified a second specific
stall mechanism: its direction latch can force hold even when the opposite
turn satisfies the full 0.48 m forecast reserve. Direction switching currently
accepts only the special stepwise-recovery flag, excluding full-clear turns.
`blocked_turn_latch_diagnostic.json` preserves 64 such decisions (62 on time),
from frames 676 through 3596. This is a controller restriction, not evidence
that those turns would physically succeed. The running path-preference arm
retains the same rule so its intervention remains path preference alone. If
it also reaches a blocked latch, accepting a full-clear opposite direction is
a concrete next correction, without weakening any clearance threshold.

The path-preference arm completed timed execution with an observed outbound
arrival at frame 1961 (196.1 navigation seconds), measured goal distance
7.65 mm and ten quiet intervals. The return leg stalled about 3.95 m from
home and the six-minute budget expired. No physical fault or disallowed
contact was reported. Plans were on time for 828/896 decisions (92.4%).
824 planning records contain the routing-treatment receipt; measured added
routing time had median/p95/maximum 2.18/3.47/13.28 ms. Camera persistence
and independent physical arrival verification were still pending when this
paragraph was written; the observed arrival alone is not physical proof.

The final 100 planning decisions were holds despite an observed route to the
home goal. This time no turn latch was active: the raw best action was a
right arc, all translating actions and the right turn failed clearance, and
the left turn passed stepwise recovery. Its predicted minimum clearance was
0.4646 m, rising to 0.4673 m at the horizon end. Its negative immediate
alignment utility made it lose to hold. The earlier recovery trigger only
handled a raw preferred pure turn, so it never started the long-way turn.

The next controller correction extends that trigger to a blocked translating
preference when the filtered result is hold and waypoint heading error is
larger than 0.1 rad. It uses the same target heading and existing clear
opposite-turn checks. A blocked latch may also switch to a full-reserve
opposite turn, retaining direction when the current turn remains clear.
No nominal footprint, reserve or stepwise gain threshold changes.
`blocked_arc_recovery_counterfactual.json` records hold -> left turn on the
saved final decision, with no native state input. Ten focused tests pass,
including preservation of frontier exclusions, persistent turning when clear,
full-reserve direction recovery, and the blocked-arc case. The new synthetic
arc fixture initially marked blocked translations clear; that fixture flag
was corrected before the successful test run.

The prepared prospective arm is
`go2_clearance_preferred_arc_recovery_learned_round_trip_native_layout02_v1_attempt_001`.
It retains the same route preference, model, estimator and six-minute budget.

The path-preference owner subsequently exited 0 and all 3,605 paired camera
observations and poses were saved. Independent native evaluation confirms
the outbound arrival: the full one-second dwell stayed 9.45–13.36 mm from
the goal, all actual requested command intervals were zero, and maximum
100 ms native translational speed was 0.01042 m/s. There were zero
disallowed contact samples. Median/maximum registered position error was
2.27/9.34 mm. The return remained incomplete; this is verified goal-reaching
on development layout 2, not a verified round trip or unseen reliability.
The next arm starts after this owner's exit and completed archival.

## Arc-recovery trial and prospective transfer

The revised layout-2 arm finished timed execution at 271.64 simulated seconds
with 2,714 camera pairs and two observed arrivals: outbound frame 1672 and
return frame 2712. It reported no physical fault or disallowed contact,
622/670 on-time plans (92.8%), and terminal status
`OBSERVED_ROUND_TRIP_CANDIDATE`. Physical dwell evaluation follows archival.

There were **no recorded turn-recovery latch events in this successful
candidate run**. Therefore its completion cannot be attributed specifically
to the newly extended recovery trigger. Closed-loop timing and trajectory
differ between attempts; repeated prospective outcomes are needed. The
saved-decision counterfactual remains evidence that the logical stall case
was changed, not proof that this change caused the native improvement.

`ClearancePreferredReactiveRuntime` now combines the identical route mixin
with instantaneous reactive action selection. Five focused tests passed:
reactive planning remains executable without a model/history/forecasts, rejects
model use, and produces the same route proposal as the learned arm for the
same map, mission and frontier exclusions. Launcher support is prepared;
no new reactive navigation outcome is claimed.

The next fixed transfer is layout index 3, using the current learned runtime,
model, residual, estimator, route preference, all physical bounds and the
same 3,600-tick budget. It is chosen before inspecting its generated geometry.
No prior layout-3-or-higher launch was found in the ordinary artifact root's
native round-trip and stop-conditioned-independent experiment families. This
is a scoped development inventory check, not a universal historical claim
or a sealed final evaluation. The source generator defines eight layouts
and enforces disjointness from its explicit 48-layout source registry.
The prospective output is
`go2_clearance_preferred_arc_recovery_learned_round_trip_native_layout03_v1_attempt_001`.

The layout-2 arc-recovery owner exited 0 and all 2,714 pairs and poses were
saved. Independent evaluation **verifies the round trip**: outbound one-second
native dwell distance 12.53–17.37 mm, return 13.00–16.58 mm, with every requested
command interval zero. Maximum 100 ms native speeds during those dwells were
0.01737 and 0.03442 m/s, both below 0.05 m/s. There were zero disallowed
contact samples. Median/maximum registered position error was 2.90/14.57 mm.
This adds a verified continuous development layout-2 round trip to the earlier
layout-0 success; it does not establish across-layout repeatability, a specific
recovery-code benefit, JEPA advantage or deployment readiness.

The layout-3 transfer launched after the completed owner's exit and archival.
Hardware inspection found 16 physical/32 logical CPUs with full affinity,
3.6% CPU utilization, about 257 GiB free artifact space, and both GPUs idle.
There was 55.3 GiB RAM available while the predecessor was still archiving,
with additional memory released on its exit. Native concurrency remains one
to avoid simulator/rendering contention during the timing-sensitive transfer.

## Layout-3 matched comparison

Before the learned transfer outcome was known, the next arm was selected as
the reactive controller on the same layout 3 with the same six-minute budget,
plane-consensus sensing, persistent map, frontier bookkeeping, clearance-
preferring routing, mission and current-depth dispatch checks. Reactive uses
instantaneous six-action waypoint feedback, with no neural model, motion
residual, future-path clearance filter or forecast-based turn recovery. Thus
this compares the complete predictive-selection method against reactive
selection; it does not isolate JEPA training or the individual future filter.

The learned arm's timed result exhausted 3,600 ticks with an observed outbound
arrival at frame 2441. Return progress reached about 0.091 m from home at
frame 3600. No return arrival was reported. There was no physical stop or
disallowed contact; 834/891 plans were on time (93.6%), with 3,605 camera pairs
and poses recorded for persistence. This remains an incomplete round trip,
although it made substantial return progress before the budget expired.
Physical arrival verification follows completion of the camera archive.

The prepared reactive root is
`go2_clearance_preferred_reactive_round_trip_native_layout03_v1_attempt_001`.
No controller parameters or time budget are changed in response to the
learned arm's result. The legacy six-minute failures remain preserved if a
later prospectively matched pair is given a longer budget.

The learned layout-3 owner exited 0 with all 3,605 pairs and poses saved.
Independent evaluation verifies the outbound dwell: native distance
14.92–21.24 mm for the complete one-second interval, all requested commands
zero, and maximum 100 ms native speed 0.02053 m/s. There were zero disallowed
contact samples. Median/maximum registered position error was 13.70/47.53 mm.
There was no return arrival. The last twelve planning decisions included
forward, arc and turn commands, consistent with continued progress toward
home rather than a sustained final hold. No recovery latch events occurred.

The reactive layout-3 arm starts after learned archival and owner exit, with
77 GiB RAM and about 245 GiB disk available. Native concurrency remains one.
`scripts/compare_continuous_navigation_arms_development.py` is prepared to
compare completed arms using independently checked arrivals, native goal/home
distances, 10 Hz horizontal path length and planning deadlines. It checks the
matched mission/timing/routing settings and common implementation identities,
and labels the comparison as complete predictive versus reactive selection.
The actual comparison will be produced after the reactive outcome is saved.

The completed learned arm's native summary gives a final saved-camera home
distance of 20.11 mm, after the earlier progress log at frame 3600 reported
about 91 mm using pose frame 3598. These are different time boundaries and
pose sources. The final native proximity does not establish a one-second
quiet return arrival, and mission completion remains unconfirmed. Minimum
native outbound goal distance was 4.01 mm; total 10 Hz native horizontal
path length was 28.28 m. These values are preserved in
`native_navigation_summary.json` and will be used consistently for both arms.

The reactive owner exited 1 after 376 camera acquisitions (about 37.5 seconds)
with `tracking: measured visual pose unavailable`, while turning for a frontier
view near the initial location. It saved all camera observations and 374
accepted poses. Independent evaluation found no arrivals, zero disallowed
contact samples, and median/maximum accepted-pose error 1.87/5.20 mm. The
matched comparison is saved in
`go2_clearance_preferred_controller_comparison_layout03_v1_attempt_001/result.json`.
Reactive minimum native goal distance was 1.263 m and 10 Hz path length 0.657 m;
87/93 plans were on time. This early perception failure limits interpretation:
the result compares complete pipelines and does not establish a clean planning
or JEPA-specific advantage.

`plane_consensus_failure_replay.json` reproduces the tracking failure exactly
at frame 374, with no raw-pose differences over all 374 accepted frames. The
failure chain is a retained-image fit conflicting with measured floor height.
The current plane/image consensus fallback does not resolve that pair. Once
this conflict is latched, every further retained-reference attempt reports the
same conflict, so those repeated reasons are not independent geometric tests.
The next diagnosis captures the actual failed pair and consensus rejection.

The captured failure is a primary image registration from reference frame 369
to current frame 374, with 14/26 inliers. Plane-constrained consensus cannot
retain the original strict majority. The configured diagnostic is
`plane_consensus_rejection_configured_diagnostic.json`. The earlier
`plane_consensus_rejection_diagnostic.json` omitted the process's 2 cm floor-
extent configuration and therefore reported a validation mismatch; that
unconfigured result is preserved but is not an estimator failure diagnosis.

`PairLocalPlaneConsensusPose` is an explicit new estimator treatment. After
an image pair fails the existing plane-height consensus check, it rejects
that pair and lets the original reference/camera search continue. It does
not admit the conflicting image pose, lower match counts, weaken geometric
thresholds, reset history, or supply inertial/native translation. Any selected
alternative must pass the existing image, plane, gyro, displacement and
continuity checks; no qualifying alternative still produces terminal failure.
The change is the scope of rejection (pair versus whole observation), not the
acceptance rule for a candidate. A full replay of the 376-frame failure record
is underway, retaining rejected-pair records and native evaluation afterward.

The pair-local replay accepts all 376 recorded frames with no failure and no
raw-pose differences on the original 374-frame accepted prefix. At frame 374,
it rejects reference 369's 14/26-inlier pair, then accepts reference 367 with
26 preserved inliers and constrained residual RMS 2.40 mm. The image-to-plane
pose correction is only 0.244 mm. Frame 375 uses reference 374 normally.
Maximum evaluated position error over the replay is 5.55 mm. This is evidence
for the alternative measured reference, not a resumed native navigation run.

A new prospective matched pair will use the pair-local estimator for both
reactive and learned controllers, with 4,800 navigation ticks (eight minutes)
for both. The prior learned arm was still completing its return at six minutes;
all six-minute outcomes remain preserved. This new pair changes estimator and
budget together and cannot isolate their separate effects. All physical
limits, arrival requirements, routing preferences and learned weights remain
unchanged. The existing mission implementation supports up to 8,000 ticks.

The new reactive root is
`go2_pair_local_plane_preferred_reactive_round_trip_native_layout03_4800_v1_attempt_001`.
It starts first, followed by the learned arm under the same configuration.
Prelaunch resources: 77.55 GiB available RAM, 243 GiB free disk, 16 physical/
32 logical CPUs with full affinity and 0.7% CPU utilization, both GPUs idle.
The eight-minute camera population is expected to use about 40 GiB based on
the measured roughly 30 GiB six-minute population. One native owner runs at
a time; this is a capacity estimate, not an enforced reservation.

The eight-minute reactive arm finished timed execution without a tracking
failure, physical fault or disallowed contact. It recorded 4,805 camera pairs,
1,117/1,200 on-time plans (93.1%), and no arrivals. It stalled about 3.26 m
from the outbound goal. The final 100 decisions were holds with
`ADDITIONAL_VIEW_REQUIRED`; the stored current clearance was 0.4166 m,
below the unchanged nominal 0.45 m disk.

The independent saved-trace diagnosis
`reactive_stall_clearance_native_evaluation.json` shows real loss of wall
clearance, rather than large tracking drift: native nearest-wall distance
falls from 0.5014 m at frame 1468 to 0.4652 m at 1472 (first blocked disk),
then 0.4399 m at frame 1600 and 0.4350 m at 4800. Registered position errors
at these points are about 5 mm. Stored clearance is about 17–18 mm more
conservative than the physical wall distance. The controller requested a
right arc at frame 1468, then hold at 1472; continuing physical motion after
that contributes to the nominal-clearance deficit. The existing candidate
set and current-disk rule offer no reactive escape once the disk is blocked.
The learned arm will use the previously declared identical eight-minute
protocol and estimator; no thresholds are adjusted to this result.

## Position-forecast contribution diagnostic

An independent small study uses the original residual fit's existing four
layout-0 training caches (1,207 windows) and its previously used development
validation cache (448 windows). It changes no current native inputs or model.
`base_forecast_sensitivity_diagnostic.json` in the original residual-study
root records the analytic derivative of corrected XY with respect to neural
XY and yaw forecasts. The fitted correction substantially attenuates some
neural XY directions; these derivatives are not fractions of navigation
performance attributable to the neural model.

`go2_pose_action_xy_forecast_ablation_study_v1_attempt_001` fits two absolute
XY predictors with the same fixed ridge penalty 1: all 42 original features,
or only the 38 pose-history/known-command/nominal-integral features (excluding
neural XY and sine/cosine yaw forecasts). Both fits are saved before reading
validation. On the 54 command-matched 700 ms translation windows, RMSE is
10.31 mm with all features and 10.63 mm without neural forecast features;
the original hybrid is 10.16 mm and original neural XY alone is 26.76 mm.
On 65 turn-only windows, the two absolute fits are 5.49 and 5.48 mm, versus
5.69 mm for the original hybrid. These are correlated windows from previously
used development data, not new independent navigation trials. The two new
absolute fits share an objective; the old hybrid uses a different residual-
centred prior. This motivates a prospective dynamics ablation. It does not
measure heading/contact prediction benefit or establish JEPA-specific value.

The eight-minute reactive owner exited 0, with all 4,805 paired observations
and poses archived. Independent evaluation confirms no arrivals and zero
disallowed contact samples; median/maximum registered position error is
4.92/11.75 mm. The matched learned arm now starts in
`go2_pair_local_plane_preferred_learned_round_trip_native_layout03_4800_v1_attempt_001`,
with the same estimator, route preference and eight-minute budget. Available
resources after reactive exit were 77 GiB RAM and about 228 GiB disk.

A bounded post-run archival benchmark used 12 already saved paired frames.
One writer took 0.570 s; four writers took 0.160 s (3.57x). Every decoded RGB
and depth array was identical. The report is
`camera_archive_thread_benchmark.json` in the verified layout-2 arc-recovery
root. The sample is small and the predecessor was finishing archival around
this measurement, so this is a throughput indication rather than a complete
run-time forecast. Neither arm of the current pair changes its archive code;
parallel post-run writing is a candidate improvement after the pair closes.

The two absolute XY fits now have prospective runtime implementations in
`lewm/absolute_xy_forecast_ablation_development.py`. Both retain the same
learned yaw/contact predictions, route control and four causal registered
poses; only the XY predictor changes. The all-feature fit SHA-256 is
`1bcdb395bf9a16b216368ecef61052349065464f56253e8b21652b53e15c45d6`;
the pose/command-only fit SHA-256 is
`b509184a7bb52a18701ba248e8dcd770c87daf72b346d4116061ac6a3b04074b`.
Three focused checks pass: removed neural inputs cannot affect ablated XY,
future commands cannot affect earlier prediction horizons, and the runtime
replaces only XY while preserving heading/contact outputs. These variants
are prepared, not yet integrated into the native launcher or tested online.
The current eight-minute learned arm continues with its original hybrid.

The eight-minute learned arm finished timed execution with 4,805 observations,
no physical fault or disallowed contact, and no arrivals. Plans were on time
for 1,090/1,178 decisions (92.5%). It reached the goal vicinity, then repeatedly
turned while estimated goal distance varied around 2–4 cm. Its final 100
planning decisions were right turns, not translational progress. Archival and
independent evaluation are pending at this point.

At final frame 4800 all six candidates pass full clearance (minimum predicted
clearance at least 0.552 m). The right arc has position/contact utility
+0.00531 m, versus -0.00419 m for the selected right turn, but its heading
penalty is -0.04136 m, making it lose overall. The goal waypoint is only
42.9 mm away and requires no final orientation. This diagnoses excessive
heading priority near a positional mission endpoint; it is not a clearance
or recovery-latch failure.

`TerminalPositionPriorityRuntime` is prepared as a separate prospective
controller treatment. It activates only when the exact public goal is the
selected route target and lies within 0.10 m (one nominal 0.08 m forward
commitment plus the 0.02 m observed arrival radius). After unchanged prediction
and clearance filtering, it prefers an eligible translating candidate only
if its predicted position progress is positive and its position/contact utility
exceeds both hold and the original selected action. Otherwise it preserves
heading guidance. Active clearance-recovery latches retain control. Arrival
requirements, action set, commitment length and contact/clearance checks remain
unchanged. Three focused tests passed, and
`terminal_position_priority_counterfactual.json` changes the saved final
right turn to a right arc without native input. No navigation benefit is yet
claimed. The XY-input ablation remains prepared for later prospective testing.

The learned owner subsequently exited 0 and archived all 4,805 pairs and
poses. Independent evaluation confirms no arrivals and zero disallowed
contact samples, with median/maximum position error 19.16/27.86 mm. The
completed matched result is
`go2_pair_local_plane_preferred_controller_comparison_layout03_4800_v1_attempt_001/result.json`.
Learned minimum/final native goal distances are 0.368/25.32 mm, versus
1.251/3.257 m reactive. Native 10 Hz path lengths are 20.99 and 8.93 m.
Both missions remain incomplete; momentary native proximity is not the
required stopped arrival and does not establish a round trip or JEPA benefit.

The next prospective learned arm is
`go2_terminal_position_priority_learned_round_trip_native_layout03_4800_v1_attempt_001`.
It adds only the declared terminal-position action-selection treatment to
the timed controller. The post-run archive implementation now uses four
independent frame writers, preserving metadata order with executor.map and
keeping all rendering/native identity checks on the owner thread. Compression,
pixels and acquisition timing remain unchanged; the change occurs after
timed execution. The preceding matched pair was fully closed before this
archive revision. The twelve-pair benchmark established decoded-array identity
and indicated 3.57x writer throughput; full-run archival performance will be
observed in the new trial. The prepared XY predictor variants remain inactive.

The terminal-position run completed its eight-minute budget, archived all
4,805 camera pairs and poses, and its owner is no longer live. Independent
physics evaluation verifies the outbound arrival at frame 3578: the complete
one-second dwell stayed 28.57–32.71 mm from the goal, all requested commands
were zero, and maximum 100 ms speed was 0.02767 m/s. There were zero disallowed
contact samples. It did not arrive home; final native home distance was
2.213 m. Plans were on time for 1,127/1,197 decisions (94.2%); median/maximum
position error was 10.48/49.30 mm. The four-writer archive completed and
preserved static camera identity. The original exec session exit code was
not recovered; terminal process state, completed archive and absence of a
failure document were checked. No full-run archive speedup is inferred from
the earlier small benchmark.

Crucially, the terminal-position override changed zero actions in this run.
The verified outbound arrival therefore does not establish that treatment's
benefit. Measured scheduling and trajectories differed from the previous arm.
The eight-minute round-trip mission remains a failure.

The recovery latch was active for 374 planning decisions. The saved public
decisions expose 20 instances where the normal clearance-filtered planner
preferred a translating action with full reserve, positive predicted position
progress, and position/contact utility above hold, but recovery still forced
a turn toward its old heading. `progress_rejoining_counterfactual.json`
records these independent per-frame counterfactuals; it is not a closed-loop
replay. The first is frame 1396: right turn becomes left arc, predicting
8.15 mm progress. The preceding learned run also contains this behavior.

The next prospective treatment, `ProgressRejoiningTerminalRuntime`, releases
an existing recovery latch only for that full-reserve translating choice.
It retains recovery during panorama obligations, for turns, for non-progress
translations and for translations lacking full reserve. Original runtime
classes default to the old behavior. Eleven focused recovery/terminal tests
pass; no clearance, arrival, action-duration or physical limit changes.
The next output is
`go2_progress_rejoining_learned_round_trip_native_layout03_4800_v1_attempt_001`.
Before launch the prior owner was absent, CPU utilization was 3.4%, 77.3 GiB
RAM and 195.9 GiB artifact space were available, and GPU utilization was 0%
with about 1.84 GB of 34.21 GB VRAM occupied. One native owner and four post-run
archive writers are retained; overlapping native sessions would contend for
timing and large camera buffers. The XY-input ablation remains inactive.

The progress-rejoining layout-3 owner exited 0 with 4,752 paired camera
observations and poses. Both arrivals are independently verified: outbound
frame 3170 (317.0 simulated seconds), home frame 4750 (475.0 seconds). The
one-second physical dwell maximum distances were 17.73 mm outbound and
27.51 mm at home; maximum 100 ms speeds were 0.01784/0.02092 m/s and every
requested dwell command was zero. There were zero disallowed contact samples.
Median/maximum registered position error was 9.10/20.97 mm. This establishes
one complete development round trip under the current controller, not
repeatability or JEPA-specific benefit.

Recovery released once, at frame 1836, for an on-time left arc. Terminal
position priority changed 17 decisions, of which 15 were on time; the saved
`controller_treatment_events.json` records them. Plans were on time for
997/1,175 decisions (84.9%). A one-thread saved-map reconstruction and small
offline diagnostics/tests overlapped portions of this shared-host trial;
the measured deadline and late-plan rejection remained active. These are
not controlled estimates of each treatment's causal contribution.

Separately, the preceding terminal-position run's return failure was not
just a short budget. Replaying the exact recorded mapping update frames
reproduced `OBSERVED_COMPONENT_HAS_NO_FRONTIER` at frames 4520 and 4760.
The home cell was observed, and exact 1 cm obstacle geometry admitted a
continuous nominal-radius path. Coarse obstacle inflation falsely closed
the route. `return_route_connectivity_diagnostic.json` and three small saved
map snapshots preserve this public-sensor diagnosis; no native pose was used.

`lewm/fine_goal_route_development.py` prepares an observed-goal fallback:
when the coarse proposal fails to reach an observed goal, it searches the
same observed floor with exact fine-cell clearance on connectors and every
edge, retaining the 0.45 m nominal radius and soft clearance preference.
On both saved stalled maps it found a path with 0.515 m minimum continuous
clearance in about 48 ms. Five focused fine/coarse routing tests passed,
including true obstacles and missing floor preventing a route. This fallback
is not active in the launcher. A conditional planning receipt for it was
added; that branch does not execute in the current controller.

Next is a prospective transfer of the successful progress-rejoining controller
to layout 4 with the same eight-minute budget, before any generated layout-4
geometry inspection. No prior `go2_*native_layout04_*` launch was found in
the ordinary native artifact inventory; the generator is disjoint from its
explicit 48-layout source registry. This is a scoped development novelty
claim, not a universal historical or final held-out claim. Output:
`go2_progress_rejoining_learned_round_trip_native_layout04_4800_v1_attempt_001`.
The prepared fine-goal route and XY-predictor variants remain inactive.

Before the layout-4 outcome is known, the next transfer cohort is fixed to
layouts 4, 5, 6 and 7, one first attempt each, using the progress-rejoining
controller, `seed_2026091001_full_jepa`, and 4,800 navigation ticks. Every
failure counts. Do not tune the controller between these four attempts;
the prepared fine-goal fallback remains inactive through this cohort.
This separates prospective transfer evidence from the earlier layout-by-layout
development. A scoped ordinary native launch inventory found only the new
layout-4 owner among `go2_*native_layout04_*` through `layout07_*`; generated
geometry for layouts 5–7 has not been inspected here. These remain development
experiments, not sealed final evaluation. After this fixed cohort, run the
matched reactive comparisons and use failures to choose the next development
treatment. Repeatability within each layout remains a separate question.

Fixed-cohort layout 4 finished with no arrivals, exhausted its eight-minute
budget, and exited 0 after all 4,805 paired camera frames and poses were saved.
Independent evaluation confirms zero disallowed contact samples and no
round trip. Pose median/maximum error was 36.78/51.07 mm. Plans were on time
for 1,144/1,200 decisions (95.3%), so this failure is not simply missed planning
deadlines. Minimum/final native goal distances were 1.258/2.593 m; native
10 Hz path length was 8.900 m.

The final 99 nonterminal planning records all requested hold during
`FRONTIER_STANDOFF_REQUIRES_VIEW`. All five moving candidates failed forecast
clearance; the final turn forecasts reached minima of 0.4674/0.4682 m,
below the 0.48 m full reserve and without the required stepwise improvement.
Recovery never released to a progressing translation in this run. There were
200 decisions with no clear moving candidate; the first was frame 1404.
The final native distance to the nearest physical wall was 0.4903 m. This
physical evaluator measurement does not validate a moving candidate or
authorize relaxing the forecast checks. The failure and its diagnosis are
preserved in `fixed_transfer_failure_diagnostic.json`,
`native_stall_wall_clearance.json`, and `native_navigation_summary.json`.

The cohort now has 0/1 verified round trips, with layouts 5–7 remaining.
Proceed to layout 5 with the same controller and budget, retaining all failures.
Before the next launch the prior owner was absent, 78.2 GiB RAM and 164.7 GiB
artifact space were available, CPU utilization was 0.5%, and GPU utilization
was 0%. Retain one native owner and four post-run archive writers. No fallback,
action-set, sensor, model, clearance or timing treatment changes between these
cohort trials.

Fixed-cohort layout 5 exited 0 after its budget expired, with 4,813 paired
camera frames and poses archived. It reported an outbound arrival at frame
3566, but independent physics evaluation REJECTED that arrival: the complete
one-second dwell was 48.05–54.49 mm from the true public goal, outside the
40 mm physical radius. The estimated distance was 13.15 mm. Commands were
all zero and maximum 100 ms speed was 0.01734 m/s, so the rejection is geometric,
not a failure to stop. This is a false arrival caused by pose/goal-registration
error, not a verified outbound success. There was no return arrival.

Median/maximum position error was 30.52/61.85 mm. Zero disallowed contact
samples were recorded; 1,122/1,197 plans were on time (93.7%). Minimum native
distance before the reported outbound arrival was 44.88 mm; final native
home distance was 1.266 m. The 10 Hz native path was 31.41 m. Final nonterminal
planning still followed frontier routes rather than an all-hold deadlock.
Neither recovery release nor terminal-position priority changed an action
in this trial. `fixed_transfer_treatment_events.json` and
`native_navigation_summary.json` preserve these outcomes.

The fixed cohort has 0/2 verified round trips and no verified outbound arrivals:
layout 4 stalled, layout 5 falsely reported its outbound arrival and then
exhausted the budget during return. Continue unchanged to layouts 6 and 7.
Before the layout-6 launch, the prior owner was absent, 77.5 GiB RAM and
148.9 GiB artifact space were available, CPU utilization was 0.3%, and GPU
utilization was 0%. One native owner and four archive writers remain selected.

During layout 6, small read-only calculations on the completed layout-5
records decomposed the false arrival without changing the running estimator.
At frame 3566, raw visual XY error was approximately (-25.85, -24.61) mm;
floor registration changed XY by only (+0.24, +0.44) mm. The horizontal
failure therefore precedes the floor-registration step. Visual yaw error
was -0.5521 degrees, while the recorded simulated gyro orientation error
was +0.00745 degrees. Native state was used only for these evaluations.

A causal diagnostic carried each consecutive saved visual translation
increment through the previous measured gyro orientation instead of the
previous visual orientation. All candidate positions were fixed before
opening native evaluator data. XY median/maximum error changed from
30.48/62.25 mm to 23.06/28.37 mm on that saved trajectory; error at the reported
arrival changed from 35.69 to 24.79 mm. This is not a refit of image
correspondences, an online estimator test, a navigation counterfactual, or a
real-IMU calibration. It motivates a visual/gyro fusion experiment after the
fixed cohort and matched baseline, while also showing residual translation
error that gyro orientation alone does not remove. Reports are
`false_arrival_pose_error_decomposition.json`,
`false_arrival_gyro_visual_heading_comparison.json`, and
`gyro_carried_visual_increment_diagnostic.json` in the layout-5 root.

Fixed-cohort layout 6 exited 0 after reporting both arrivals, with all 3,354
paired camera frames and poses saved. Independent evaluation accepts outbound
frame 2551: the one-second stopped dwell remained 31.34–39.53 mm from the goal,
all requests were zero, and maximum 100 ms speed was 0.01703 m/s. It rejects
return frame 3352: native home distance during the dwell was 33.28–44.09 mm,
exceeding the 40 mm radius, although observed distance was only 4.80 mm.
Commands were zero and maximum 100 ms speed was 0.02738 m/s. Thus the reported
round trip is a false terminal success and MUST NOT count as verified.

There were zero disallowed contact samples; median/maximum pose error was
11.48/71.04 mm. Plans were on time for 789/828 decisions (95.3%). Native 10 Hz
path length was 22.19 m; final home distance was 43.78 mm. Recovery never
released; terminal-position priority changed 62 decisions, all on time.
The fixed transfer cohort is 0/3 verified round trips, with one verified
outbound arrival (layout 6). Layout 7 follows with the same controller and
budget. Neither a near miss nor the controller's terminal-success report
changes the independent physical arrival requirement.

The same causal gyro-carried visual-increment diagnostic was evaluated on
three additional completed runs, without changing any native controller.
Horizontal median/maximum errors in millimetres were:

| Saved layout | Original visual increments | Gyro-carried increments |
| --- | ---: | ---: |
| 3 | 9.10 / 21.12 | 10.94 / 17.45 |
| 4 | 36.64 / 51.00 | 35.35 / 48.89 |
| 5, preceding diagnostic | 30.48 / 62.25 | 23.06 / 28.37 |
| 6 | 11.51 / 70.70 | 10.27 / 25.85 |

These are raw-pose horizontal errors, not the registered 3D errors reported
by the arrival evaluator. Orientation drift explains part of the error on
layouts 5 and 6, but barely explains layout 4; layout 3's median error slightly
worsens. Each saved root contains `gyro_carried_visual_increment_diagnostic.json`.
This diagnostic neither refits image correspondences nor predicts the result
of a changed controller. No claim of universally improved localization,
independent unseen performance, or realistic-IMU robustness follows from it.

Fixed-cohort layout 7 exited 0 after exhausting its budget, with 4,806 paired
camera frames and poses archived. Outbound frame 3840 is independently
verified: its stopped one-second dwell stayed 18.31–31.98 mm from the goal,
all requested commands were zero, and maximum 100 ms speed was 0.02436 m/s.
There was no return arrival; final native home distance was 1.467 m. Zero
disallowed contact samples were recorded. Pose median/maximum error was
15.23/56.38 mm; 1,123/1,193 plans were on time (94.1%). Recovery released once
at frame 2804 for an on-time left arc; terminal position priority changed
eight decisions, all on time. Final nonterminal plans followed the observed
route to the home goal cell.

The unchanged-controller transfer cohort is complete:

| Layout | Verified outbound | Verified round trip | Main outcome |
| --- | --- | --- | --- |
| 4 | No | No | Frontier-view clearance stall |
| 5 | No | No | False outbound arrival; return budget exhausted |
| 6 | Yes | No | False home arrival, maximum dwell distance 44.09 mm |
| 7 | Yes | No | Return budget exhausted |

Totals: 2/4 verified outbound arrivals, 0/4 verified round trips, zero
disallowed contact samples in all four trials. Machine-readable results:
`go2_fixed_transfer_learned_cohort_layout04_07_v1_attempt_001/result.json`.
The earlier layout-3 success was development evidence preceding this cohort
and is not included in its denominator. Reliable transfer is not established.

Next run the matched reactive arm on layouts 4–7, one first attempt each with
the same public missions, pose estimator, stored map, route preference,
eight-minute budget, action library, arrival thresholds and measured timing.
Only the predictive selection/recovery treatments are disabled. The first
output is
`go2_fixed_transfer_preferred_reactive_round_trip_native_layout04_4800_v1_attempt_001`.
Keep common implementations unchanged until all matched comparisons close.
The fine-goal fallback and gyro-based estimator experiments remain inactive.
This comparison concerns complete predictive versus instantaneous reactive
selection; it does not by itself isolate JEPA training or persistent memory.

Matched reactive layout 4 exited 0 and archived 4,805 paired camera frames and
poses. Independent evaluation confirms no arrivals and zero disallowed contact
samples. Pose median/maximum error was 17.21/19.37 mm. Plans were on time for
1,131/1,200 decisions (94.3%). The final 99 nonterminal decisions were holds
with `ADDITIONAL_VIEW_REQUIRED`: stored current clearance was 0.3943 m, below
the nominal 0.45 m disk. Native evaluator wall distance was 0.4057 m at frame
4800, so this was not solely a false mapped obstacle. Earlier native wall
distances were 0.5037 m at frame 1000 and 0.4127 m at frame 1200. No actual
contact is inferred from nominal-disk encroachment.

The first matched comparison is complete in
`go2_fixed_transfer_controller_comparison_layout04_4800_v1_attempt_001/result.json`.
Settings and common source hashes match. Both arms have zero arrivals; native
minimum goal distances were 1.258 m learned and 1.217 m reactive. Learned ended
0.4903 m from the nearest physical wall, reactive 0.4057 m. The learned
arm ended with more clearance in this trial but did not yield
mission completion; this pair establishes neither a navigation advantage nor
a JEPA-specific contribution. Reactive layouts 5–7 remain, with the same
configuration and common implementation. Next output:
`go2_fixed_transfer_preferred_reactive_round_trip_native_layout05_4800_v1_attempt_001`.

Matched reactive layout 5 terminated early at approximately 173 simulated
seconds and exited 1. The registration stage raised
`current measured candidate conflicts with transported floor reference`.
All 1,733 acquired paired camera frames were archived; 1,731 registered poses
were admitted. Independent evaluation confirms no arrivals, zero disallowed
contact samples, and pose median/maximum error 17.25/24.65 mm over the admitted
population. The failure is retained as the first attempt; no retry or gate
change is made within the matched series.

The final 100 decisions contained 93 turns and seven right arcs; routes
alternated between an observed frontier and a required post-veto view.
The last stored current clearance was 0.4869 m. This differs from layout 4's
nominal-disk blockage. Plans were on time for 416/432 decisions (96.3%).

The second comparison is complete in
`go2_fixed_transfer_controller_comparison_layout05_4800_v1_attempt_001/result.json`;
settings and common source hashes match. Neither arm has a verified arrival.
Minimum native goal distance was 44.88 mm learned versus 1.258 m reactive;
the learned reported arrival remains independently rejected. Native path
lengths were 31.41 m learned and 4.774 m reactive, with the reactive attempt
ending early on a sensor-registration fault. This is evidence about complete
pipeline outcomes, not an isolated prediction or JEPA-training effect.
Continue unchanged to reactive layouts 6 and 7. Next output:
`go2_fixed_transfer_preferred_reactive_round_trip_native_layout06_4800_v1_attempt_001`.

Matched reactive layout 6 exhausted the eight-minute budget and exited 0,
with 4,805 paired camera frames and poses archived. Independent evaluation
confirms no arrivals and zero disallowed contact samples. Pose median/maximum
error was 5.79/14.32 mm; 1,147/1,200 plans were on time (95.6%). The final
99 nonterminal decisions were holds with `ADDITIONAL_VIEW_REQUIRED`; stored
current clearance was 0.4273 m, below the nominal 0.45 m radius. The native
nearest-wall distance at frame 4800 was 0.4399 m. The pose error alone does
not explain that nominal-clearance deficit.

The third pair is complete in
`go2_fixed_transfer_controller_comparison_layout06_4800_v1_attempt_001/result.json`.
Settings and common source hashes match. Learned has a verified outbound
arrival and an independently rejected home arrival; reactive has neither
arrival and got no closer than 0.9572 m to the outbound goal. Both have zero
verified round trips and zero disallowed contact samples. The different
outbound outcomes support further comparison of complete pipelines, not a
JEPA-specific or repeatability claim. Reactive layout 7 is the final matched
trial, with common code and settings retained. Next output:
`go2_fixed_transfer_preferred_reactive_round_trip_native_layout07_4800_v1_attempt_001`.

Matched reactive layout 7 exited 0 after exhausting the eight-minute budget.
All 4,805 paired frames and registered poses were archived. Independent
evaluation found no arrivals and zero disallowed contact samples. Pose
median/maximum error was 12.04/17.23 mm; 1,127/1,200 plans were on time
(93.9%). Minimum native outbound goal distance was 1.2671 m. The final
nonterminal decision was a hold with `ADDITIONAL_VIEW_REQUIRED`, with stored
clearance 0.39777 m below the nominal 0.45 m radius.

The complete four-layout matched comparison is now recorded in
`go2_fixed_transfer_controller_comparison_cohort_layout04_07_v1_attempt_001/result.json`.
All four pairs have matching common source hashes and comparison settings.
Learned: 2/4 verified outbound arrivals and 0/4 verified round trips.
Reactive: 0/4 verified outbound arrivals and 0/4 verified round trips.
Neither arm recorded disallowed contact samples. Reactive layout 5's early
registration failure remains in the denominator. The two outbound differences
justify further development of the predictive pipeline, but the small cohort
and zero round trips establish neither reliable transfer nor a JEPA-specific
effect. The previous layout-3 success remains outside this denominator.

The next perception experiment refits translation to accepted RGB-D
correspondences while using measured gyro rotation, preserving image residual
and motion limits. It begins as a raw-estimator replay on recorded layout 5
sensor packets. It is not selected by the native launcher, and does not claim
the old image-only rotation semantics. Replaying recorded trajectories cannot
establish closed-loop navigation success or realistic gyro noise tolerance.

Four focused analytic geometry tests passed for the new refit: primary,
auxiliary and pooled-camera body translation, plus rejection of inconsistent
gyro rotation by image reprojection. Four independent raw-estimator replays
are running with the same implementation on learned layouts 3, 4, 5 and 6.
This includes the earlier successful trajectory and both false-arrival cases;
it is development diagnosis on reused trajectories, not a fresh-layout test.
Before parallel launch: 16 physical/32 logical cores, affinity 32, CPU 3.4%,
76.63 GiB available RAM, 72.88 GiB free output storage, GPU 0% busy. Each
replay uses one compute thread and about 1.3 GB resident memory. Measured
parallel progress was approximately 9–10 frames/s per process; native
simulation had already terminated. No source changes are made to the running
estimator during this replay set.

The first completed gyro-refit replay (layout 3) failed at frame 1045:
neither a retained anchor nor the previous frame supported the current pose.
On its 1,045 admitted frames, raw horizontal median/maximum error was
1.71/3.69 mm versus the original raw estimator's 5.70/9.32 mm over the same
prefix. Lower prefix error does not offset the new tracking failure. The
variant is not ready for navigation. A bounded identical replay is collecting
the per-reference rejection causes; the other three full replays continue.

The first gyro-refit replay set is complete. Errors below are raw horizontal
pose error, compared with the original raw estimator on exactly the same
admitted prefix; they exclude downstream floor registration.

| Layout | Frames / requested | Original median / max (mm) | Gyro refit median / max (mm) | Outcome |
| --- | --- | --- | --- | --- |
| 3 | 1045 / 4752 | 5.70 / 9.32 | 1.71 / 3.69 | Tracking failure at 1045 |
| 4 | 4805 / 4805 | 36.64 / 51.00 | 28.38 / 53.65 | Completed replay |
| 5 | 3779 / 4813 | 27.97 / 57.52 | 7.57 / 13.42 | Image/plane conflict at 3779 |
| 6 | 3354 / 3354 | 11.51 / 70.70 | 10.94 / 16.99 | Completed replay |

At layout 5's original false outbound declaration (frame 3566), the refitted
raw pose had 7.63 mm horizontal error and reported 40.68 mm to the public
goal. At layout 6's false home declaration (frame 3352), those values were
14.34 mm and 39.69 mm. Both distances exceed the controller's 20 mm observed
arrival threshold, but this is a recorded-trajectory observation, not proof
of a changed closed-loop arrival outcome. Layout 4 retains substantial drift,
and its maximum raw horizontal error worsened slightly.

The bounded layout-3 diagnostic reproduced the same prefix exactly. Several
otherwise qualified pairs failed because gyro refitting lost an accepted
RGB-D correspondence; other pairs lacked enough matches. The next variant
uses monotonically pruned gyro/image consensus, retaining the same absolute
image residual, reprojection, conditioning, motion and strict-majority limits.
It also treats image/plane conflicts as pair-local rejections, so another
independently qualified pair can be considered; conflicting poses are never
admitted. Four new focused tests passed, covering outlier rejection with
primary, auxiliary and pooled cameras and refusal to lose strict majority.

This variant is `lewm/gyro_consensus_pair_pose_development.py`. Four new raw
replays are running on the same layouts, with exclusive output name
`gyro_consensus_pair_pose_replay_v1.json`. Before launch: CPU 0.5%, 77.58 GiB
available RAM, 72.87 GiB free disk, both GPUs idle; no native owner remained.
Use four one-thread replay processes, as measured in the preceding set.
No native launch or claim of navigation improvement has been made for either
gyro variant. Next inspect full replay survival and accuracy, then integrate
the supported estimator with explicit gyro-derived rotation semantics and
evaluate it with the actual floor-registration stage before a native trial.

All four robust gyro-consensus raw replays completed without tracking failure:

| Layout | Frames | Original raw median / max XY error (mm) | Gyro consensus median / max XY error (mm) |
| --- | --- | --- | --- |
| 3 | 4752 | 9.10 / 21.12 | 6.67 / 11.69 |
| 4 | 4805 | 36.64 / 51.00 | 37.93 / 63.58 |
| 5 | 4813 | 30.48 / 62.25 | 6.24 / 11.15 |
| 6 | 3354 | 11.51 / 70.70 | 9.72 / 15.24 |

This resolves the two new tracking failures from the first gyro-refit variant
on these recorded trajectories and improves three of four error profiles.
Layout 4 worsens and remains an unresolved translation-drift case. These are
reused development trajectories, not navigation or generalization results.

`GyroConsensusVisualMotion` now connects the estimator to the real pose-reader
and floor-registration pipeline. The new mode is explicitly `gyro_rgbd_refit`
with `gyro_role=rotation_estimator`. The original `current_joint_pose` accessor
still requires image-derived joint rotation; the new supported-pose dispatcher
admits the separately identified gyro variant. Floor-registered outputs retain
the actual original estimator mode and distinguish its witnesses. Image/depth
identity, continuity, floor residual, correction and motion thresholds remain.
The completed fixed-controller comparison predates these integration changes.

Twenty-eight focused existing dual-camera/floor-transport tests passed. A
ten-frame real recorded camera/gyro/floor-registration prefix also passed.
Its first evaluation attempt used the wrong saved-pose key and is preserved
as `gyro_consensus_registered_prefix_v1.json`; the corrected evaluator result
is `gyro_consensus_registered_prefix_v2.json`. No long registered replay is
required before the next development simulation: the native trial will
exercise that path with actual closed-loop decisions.

The native experiment is now running in
`go2_gyro_consensus_progress_rejoining_learned_round_trip_native_layout06_4800_v1_attempt_001`.
It changes perception while keeping the learned model, motion correction,
progress-rejoining controller, action library, mission, eight-minute budget
and 20 mm observed / 40 mm physical arrival requirements from learned layout
6. This is a development revisit of a known false-home-arrival case. It is
not part of the completed four-layout fixed-controller denominator.

Before native launch: all replay owners were terminal, 78.38 GiB RAM and
72.86 GiB disk were available, CPU 0.5%, GPUs idle. Run one native owner with
the existing four post-run archive writers. No concurrent replay or training
is planned during the timed navigation. Independent arrival evaluation must
follow terminal execution and archive completion; no success is yet claimed.

A brief evaluator-only summary of the completed layout-4 raw replay is saved
as `gyro_consensus_translation_drift_diagnostic_v1.json`. Horizontal error
grows mainly along initial X: approximately -6.8/+3.5 mm XY at frame 1200,
-37.3/+6.8 mm at 2400, and -56.9/+25.5 mm at 4800. Sampled gyro yaw errors
remain within about 0.034 degrees. Several sampled selected reference ages
are only two or three camera frames. Thus accurate gyro heading alone does
not resolve the remaining accumulated translation error. This small saved-
result calculation ran during the native trial; it did not replay sensors,
alter the online estimator or use evaluator poses for control.
