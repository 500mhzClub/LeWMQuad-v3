# Fresh-maze replication after the renderer scheduling repair

**Completed: all 20 assignments evaluated, 14 verified round trips, zero contacts.**
See [the complete scientific result](go2_nogil_navigation_replication_complete_result_2026-09-16.md)
for the full comparison and remaining limitations. Earlier progress entries below
are chronological and do not represent the final batch state.

Four new procedural maze topologies and five fixed arms (20 native missions).
The inventory and executable plan were saved before assignment 1 started.
Construction seed 2026091617 accepted four of six structural candidates; both
abstract topology and grid embedding differ from the explicit 88-layout source
registry and from each other. This is prospective development within the same
maze family, not sealed final evaluation or broader environment-type transfer.

The arms are JEPA, same-data supervised prediction, fitted pose-command
prediction, instantaneous ranking with predictive guards, and reactive feedback
with forecast selection disabled. The reactive arm still computes the model for
workload control; it changes the controller package, not only a scoring term.
Layouts rotate arm order. Four layouts cannot perfectly balance five positions.
All models, controller sources and assignments remain fixed throughout the batch.
A failed attempt remains in the denominator; no topology replacement or tuning
from these outcomes is permitted within this comparison.

All arms use the unchanged native drawing function compiled with interpreter
lock release, the original six-candidate inference batch, the same direct-stage
timers, and 20 ms added planning delay. The 300-ms deadline remains unchanged.
Native missions run sequentially with fixed per-layout CPU allocation. Raw
physics is evaluator-only; policy inputs remain visual/depth observations,
ideal gyro and command history. Synthetic depth noise is 2 mm. This is measured
simulation timing, not real-time or hardware qualification.

Primary outcome: physically verified goal-and-home round trip without disallowed
contact. Secondary outcomes: goal arrival, contacts, tracking failures, simulated
completion time, on-time planning fraction, and forecast errors on executed
windows. Compare JEPA against supervised, both against pose-command prediction,
and supervised against each action-selection control. Report all four pairs
and failure modes; one execution per layout/arm and one training seed limit
reliability and algorithm-superiority claims.

Run `scripts/run_go2_nogil_navigation_replication_development.py --assignment N`
with the existing native environment and assigned CPU group, then `--evaluate
--assignment N` after owner exit and persistence. The JSON plan and inventory
have the same document prefix. Retire only completed diagnosed depth under the
existing retention policy; keep active failure replay inputs and the first
successful renderer-fix reference. Preserve every result, physics trace, RGB,
command, forecast, timing receipt, source identity and failure record.

Assignment 1 (JEPA, fresh layout 0) failed in visual tracking with 365 camera
frames, no arrivals and zero contacts. Of 90 plans, 89 were on time; recorded
added delay was exactly 20 ms. There were 34.5 seconds of nonzero commands but
no requested translation, and 38 recovery-active plans. This is an early visual
recovery/navigation failure despite adequate measured planning deadlines.
Full depth is retained for diagnosis. Continue the fixed matched comparisons;
do not alter the controller or exclude this maze because of its outcome.

Assignment 2 (supervised, fresh layout 0) also failed in visual tracking, after
454 successfully tracked frames; 110/113 plans were on time. It requested no
translation and made 51 recovery-active plans. Both neural arms therefore
failed before goal-directed translation, with zero arrivals or contacts. Their
last recovery records request a previously well-supported heading while the
current view has few selected features. Different failure times prevent a claim
of identical execution; the shared failure mode remains to be diagnosed after
the fixed comparisons. Both full raw recordings are retained.

Assignment 3 (pose-command, fresh layout 0) failed in visual tracking with 407
acquired frames, zero arrivals/contacts, no translation requests, and 97/100
plans on time. Examination of all three completed `initial_survey.json` records
shows the same unfinished survey stage: three views completed (approximately
0, 45 and 90 degrees), target heading 2.357100476470606 radians (135 degrees),
`complete=false`. Recovery takes precedence when image support becomes weak,
then the survey requests the same uncompleted heading again. These records
identify a shared startup-survey/recovery interaction to investigate after the
fixed comparisons; they do not by themselves prove a repair or isolate the
tracking algorithm from the commanded trajectory.

The first fresh layout is complete across all five arms: zero round trips,
zero arrivals, zero contacts, and five visual-tracking failures before any
requested translation. Instantaneous ranking had 88/88 on-time plans; reactive
feedback had 89/89. All five stopped with the initial panorama unfinished at
the fourth requested heading. The aggregate readout is
`go2_nogil_replication_readout_v1_attempt_001/result.json`; it explicitly reports
5/20 evaluated assignments and remains IN_PROGRESS. All five raw failure
recordings are retained. Proceed to fresh layout 1 without controller changes.

User steering during assignment 6: continue this comparison unchanged; consider
larger candidate sets later. Six candidates remain fixed here. A future study
could vary candidate speeds/turn rates or action sequences while measuring both
navigation benefit and missed deadlines. Increasing candidate count is a planner
change, not a JEPA retraining requirement by itself, but new command sequences
would require checking the frozen model's coverage. No expanded-candidate test
has been started or added to this batch.

Assignment 6 (supervised, fresh layout 1) passed physical round-trip evaluation: 248.02 simulated seconds, zero contacts, intact tracking, 586/605 plans on time (96.9%), and 14 recovery-active plans. Both one-second quiet arrivals stayed within 17.8 mm of the physical targets; maximum 100-ms speeds were 21.5 and 14.4 mm/s. The added delay remained exactly 20 ms. Retain its full depth as the first successful fresh-maze reference. The first-maze failures remain in the comparison.

Assignment 7 (pose-command, fresh layout 1) also passed physical round-trip evaluation: 379.36 simulated seconds, zero contacts, intact tracking, and 916/940 plans on time (97.4%). Supervised finished 131.34 s earlier in this single pair. Observed outbound arrival frames were 1631 versus 2945; both return legs took 845 camera intervals. This is not yet a replicated speed advantage.

Exploratory post-hoc inspection of the pose-command stall (frames 1600–2799) found 282/300 turn selections and 294/300 on-time plans. All 300 plans targeted an observed-floor frontier. Turn-recovery events included 62 alternative-turn latches and 58 releases to the preferred heading. This points toward forecast/clearance/recovery interaction rather than a long deadline outage, without proving a counterfactual action would have succeeded. The interval was selected after observing the stall.

Post-hoc physical corridor readout found 11 return edges reversing observed outbound edges in each successful run, with no invalid maze-graph transitions or outside-grid frames. Supervised had 11 unique directed outbound edges; pose-command had 13. These are physical backtracking observations, not a causal memory ablation. Source: scripts/read_go2_nogil_replication_backtracking_development.py; per-root physical_return_corridor_readout_v1.json. The first successful fresh-maze recording remains full; only redundant diagnosed pose-command success depth was retired under the existing policy.

Assignment 8 (instantaneous ranking, fresh layout 1) passed physical round-trip evaluation: 249.62 simulated seconds, zero contacts, intact tracking, 589/613 plans on time (96.1%), and five recovery-active plans. It was 1.60 s slower overall than supervised prediction: outbound arrival 7.1 s earlier, return leg 8.7 s longer. This single pair shows similar completion time, not a substantial ranking benefit. Both arrivals passed the one-second quiet physical-radius checks.

The instantaneous-ranking receipt changed the pre-guard preferred action in 174/613 plans; subsequent guards changed its preferred action in 15 plans. Predictive guards remain, so this is not prediction-free navigation. Physical return reversed all 11 observed outward corridors with no invalid graph transitions. Source snapshots, timing, forecast and physical evidence were preserved; redundant successful depth was retired after diagnosis. Continue with assignment 9 (reactive feedback), then assignment 10 (JEPA), on this same fixed layout.

Assignment 9 (reactive feedback, fresh layout 1) passed physical round-trip evaluation: 385.28 simulated seconds, zero contacts, intact tracking, 924/949 plans on time (97.4%), and 14 recovery-active plans. Both arrivals passed the one-second quiet physical-radius checks. The return leg took 83.7 s and reversed all 11 outward corridors; the longer total primarily arose near the outward goal.

The post-hoc terminal interval, frames 1600–2999, contained 341 right turns, two holds and one forward selection; 336/344 plans were on time. Across 1400 camera observations it reached at most two consecutive quiet intervals, with 20 observations within the 2-cm observed arrival radius. Native heading accumulated -9.639 revolutions; physical goal distance ranged 18.5–52.4 mm (median 35.2 mm). It eventually escaped and arrived. This is a controller-specific terminal orbit, not evidence that prediction is necessary for every reactive controller. The interval and physical analysis source are saved in the root; no alternative action outcome was inferred. Redundant successful depth was retired after evaluation and diagnosis; all non-depth evidence and other retained failure/reference inputs remain.

Assignment 10 (JEPA, fresh layout 1) exhausted 480.90 simulated seconds with no
arrivals, zero contacts, intact tracking, 1176/1200 plans on time (98.0%), and
14 recovery-active plans. This completes the second layout: all other four
arms passed their physical round-trip checks. Retain JEPA's full depth for
investigation; do not remove the failed arm or change the remaining batch.

A post-hoc frontier-view diagnosis found 965/1200 plans with route status
FRONTIER_STANDOFF_REQUIRES_VIEW. Frames 600–3199 contained 650 such plans:
330 left turns, 314 right turns, six holds, no translations, and 638 on-time
plans. Frames 3200–4299 contained another 275 frontier-view plans, with 252
turns, 21 holds, two left arcs, and 272 on time. Later motion resumed too late
to reach the goal within budget. This points toward a prediction/clearance/view
control interaction; it does not isolate the exact causal defect.

Independent comparison of all 4805 recorded registered positions against the
native camera-time positions found horizontal error median 1.90 mm, p95 4.06 mm,
and maximum 5.43 mm. In the post-hoc stall interval (frames 600–4299), median
was 1.87 mm and maximum 4.28 mm. This makes a large accumulated position error
an unlikely explanation. Native truth was evaluator-only and no online error
bound is claimed calibrated. Records: frontier_view_stall_readout_v1.json and
registered_pose_posthoc_accuracy_v1.json in the JEPA root.

## Midpoint: two of four fresh layouts complete

| Arm | Layout 0 | Layout 1 |
| --- | --- | --- |
| JEPA | Tracking failure before translation | Budget exhausted, no arrival |
| Supervised rollout | Tracking failure before translation | Round trip, 248.02 s |
| Pose-command | Tracking failure before translation | Round trip, 379.36 s |
| Instantaneous ranking | Tracking failure before translation | Round trip, 249.62 s |
| Reactive feedback | Tracking failure before translation | Round trip, 385.28 s |

All ten runs had zero contacts. This is 4/10 verified round trips so far, with
all failures retained. It does not demonstrate reliable transfer or a JEPA
advantage. Complete the remaining two fixed layouts before interpreting the
full comparison. Models, controller, six candidate actions, noise and added
planning delay remain unchanged.

To preserve the new JEPA failure recording and continue the batch, retired only
the superseded old no-RGB direct adapter case's depth, under the existing
retention policy. Its completed raw sensor/model/command and visibility audits,
verified outward arrival, failed return/budget outcome, RGB, physics, commands,
models and other metadata remain unchanged. Reclaimed 8,564,809,728 allocated
bytes from 9042 depth leaves; all 3037 case JSON hashes, 9075 non-depth identities
and 33 parent JSON hashes matched. Inventory:
.generated/depth_retirement_old_adapter_no_rgb_direct_2026-09-16/.

Assignment 11 (pose-command, fresh layout 2) passed physical round-trip evaluation in 180.00 simulated seconds with zero contacts, intact tracking, 417/433 plans on time (96.3%), and 12 recovery-active plans. Both one-second quiet arrivals passed; their maximum physical distances were 10.0 mm outbound and 19.8 mm home. The physical return reversed all seven observed outward corridors, with no invalid graph transitions or outside-grid frames. The long frontier stall seen for this arm on layout 1 did not recur. Source/timing/forecast/physical records were preserved and redundant successful depth retired after diagnosis. Continue assignment 12, instantaneous ranking, on the unchanged layout and controller.

Assignment 12 (instantaneous ranking, fresh layout 2) passed physical round-trip evaluation in 165.68 simulated seconds, with zero contacts, intact tracking, 392/406 plans on time (96.6%), and five recovery-active plans. Both one-second quiet arrivals passed; maximum physical distances were 5.4 mm outbound and 22.5 mm home. Physical return retraced all seven outward corridors with no invalid graph transitions. This is 14.32 s faster than pose-command on this layout; the supervised and JEPA matched runs remain pending. Exact added delay and selector behavior were verified. Redundant successful depth was retired after diagnosis, preserving all non-depth evidence.

Additional post-hoc readout of JEPA layout 1, frames 600–4299, found 104 new alternative-turn latches. All 104 preferred turns passed the nominal footprint check but missed the extra reserve; their predicted reserve shortfall ranged 0.009–11.39 mm (median 2.65 mm). The selected turn opposed the preferred turn in 455/925 plans; 910/925 plans were on time. Only 11 plans excluded a full-reserve-clear translating action from scan scoring, and none coincided with those opposite-turn selections. This narrows the observed stall toward repeated reserve-boundary turn selection; it does not establish safe alternative execution or justify reducing the reserve. Existing analysis script: scripts/diagnose_saved_scan_recovery_development.py; saved result: saved_scan_recovery_600_4299_v1.json in the JEPA layout-1 root. Controller and thresholds remain unchanged.

Assignment 13 (reactive feedback, fresh layout 2) passed physical round-trip evaluation in 207.66 simulated seconds, zero contacts, intact tracking, 500/505 plans on time (99.0%), and 12 recovery-active plans. Both one-second quiet arrivals passed, with maximum physical distances 19.0 mm outbound and 16.4 mm home. Outbound arrival was frame 1492 and home frame 2069; the temporary goal-area delay resolved within budget. Physical return reversed all seven outward corridors with no invalid graph transitions. Compared with instantaneous ranking, outbound was 46.4 s later but return was 4.6 s shorter. This is a controller-package comparison. Exact 20-ms delay and forecast-unused selector were verified; redundant successful depth retired after diagnosis. Continue JEPA, then supervised, on this same fixed layout.

Assignment 14 (JEPA, fresh layout 2) passed physical round-trip evaluation in 174.02 simulated seconds with zero contacts, intact tracking, 411/425 plans on time (96.7%), and 15 recovery-active plans. Both one-second quiet arrivals passed, maximum physical distances 7.6 mm outward and 15.5 mm home. Outward arrival frame 1073, home frame 1736. All seven outward corridors were physically reversed on return, with no invalid graph transitions. The long frontier-view stall from layout 1 did not recur here. This first JEPA success in the new cohort is 8.34 s slower than instantaneous ranking, 5.98 s faster than pose-command, and 33.64 s faster than reactive feedback on this layout; one execution does not establish a speed advantage. Redundant successful depth retired after diagnosis; the layout-1 JEPA failure and full comparison references remain retained. Proceed to supervised rollout, the final arm on layout 2.

Assignment 15 (supervised rollout, fresh layout 2) passed physical round-trip evaluation in 162.08 simulated seconds, zero contacts, intact tracking, 380/396 plans on time (96.0%), and 16 recovery-active plans. Both one-second quiet arrivals passed, with maximum physical distances 8.5 mm outward and 14.9 mm home. Outward/home frames were 1004/1617. All seven outward corridors were reversed with no invalid graph transitions. Redundant successful depth retired after diagnosis.

All five arms completed fresh layout 2 without contact: supervised 162.08 s, instantaneous ranking 165.68 s, JEPA 174.02 s, pose-command 180.00 s, reactive feedback 207.66 s. Supervised was 3.60 s faster than instantaneous ranking and 11.94 s faster than JEPA in these single runs. Across the first three layouts, JEPA has 1/3 verified round trips and each other arm 2/3; all 15 runs have zero contacts. This does not establish reliable transfer or a JEPA advantage. The remaining fixed assignments are instantaneous, reactive feedback, JEPA, supervised, and pose-command on fresh layout 3. Continue unchanged.

Assignment 16 (instantaneous ranking, fresh layout 3) passed physical round-trip evaluation in 210.76 simulated seconds with zero contacts, intact tracking, 500/519 plans on time (96.3%), and five recovery-active plans. Both one-second quiet arrivals passed; maximum physical distances were 10.9 mm outward and 7.6 mm home. Outward/home arrival frames were 1467/2104. The physical corridor readout is saved beside the result. Instantaneous ranking now completes 3/4 fresh layouts; its first-layout tracking failure remains included. Exact added delay and assigned selector were verified. Redundant successful depth retired after diagnosis; proceed to reactive feedback on the unchanged final layout.

Assignment 17 (reactive feedback, fresh layout 3) passed physical round-trip evaluation in 204.90 simulated seconds, zero contacts, intact tracking, 490/504 plans on time (97.2%), and eight recovery-active plans. Both quiet arrivals passed; maximum physical distances were 14.7 mm outbound and 16.4 mm home. Outward/home frames were 1435/2043. The physical corridor readout is preserved. Reactive feedback now finishes 3/4 fresh layouts; it was 5.86 s faster than instantaneous ranking on this final layout, while slower on layouts 1 and 2. Exact added delay and forecast-unused selector were verified; redundant successful depth retired after diagnosis. Proceed to JEPA on the fixed final layout.

Assignment 18 (JEPA, fresh layout 3) passed physical round-trip evaluation in 237.20 simulated seconds, zero contacts, intact tracking, 558/578 plans on time (96.5%), and eight recovery-active plans. Both one-second quiet arrivals passed; maximum physical distances were 13.6 mm outward and 18.9 mm home. Arrival frames 1776/2368. JEPA finishes 2/4 fresh layouts, with the shared first-layout tracking failure and its separate second-layout budget failure retained. The physical corridor readout is preserved; redundant successful depth retired after diagnosis.

Exploratory terminal interval frames 1400–1776 contained 84 plans, 83 on time: 61 forward translation pulses and 23 holds. The 994 request records labeled NO_ON_TIME_PLAN here must not all be counted as deadline misses: short translation commitments deliberately expire after 100 ms and leave a 300-ms quiet tail before the next dispatch, which can receive that same label. This was prolonged goal settling with mostly timely planning, not evidence of a long compute outage. Saved terminal_pulse_settling_readout_v1.json; no controller, reserve or arrival threshold changed. Proceed to supervised and pose-command for the remaining two fixed assignments.

Assignment 19 (supervised rollout, fresh layout 3) passed physical round-trip evaluation in 230.06 simulated seconds, zero contacts, intact tracking, 539/559 plans on time (96.4%), and 28 recovery-active plans. Both quiet arrivals passed; maximum physical distances were 15.7 mm outbound and 7.0 mm home. Arrival frames 1716/2297. The physical corridor readout is preserved. Supervised finishes 3/4 fresh layouts, faster than JEPA by 11.94 s and 7.14 s on the two jointly successful layouts; JEPA alone failed the layout-1 budget. Supervised was 19.30 s slower than instantaneous ranking and 25.16 s slower than reactive feedback on this final layout. Keep all matched outcomes rather than selecting the favorable speed comparisons.

The requested added delay remained 20 ms; actual publication delay was 20 ms for 558 plans and 22 ms for one plan, a one-physics-step overshoot, recorded rather than normalized away. No controller or deadline was changed. Redundant successful depth retired after diagnosis. Assignment 20, pose-command on the same layout, is the only mission remaining.

Assignment 20 (pose-command, fresh layout 3) passed physical round-trip evaluation in 194.86 simulated seconds, zero contacts, intact tracking, 461/481 plans on time (95.8%), and eight recovery-active plans. Both quiet arrivals passed; maximum physical distances were 15.3 mm outbound and 9.4 mm home. Arrival frames 1395/1945. The physical return corridor readout is preserved. Pose-command finishes 3/4 fresh layouts and was fastest on the final layout. Exact 20-ms added delay was verified. Redundant successful depth retired after diagnosis; all twenty fixed assignments are now evaluated, including every failure.
