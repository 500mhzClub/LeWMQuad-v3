# Next structurally independent development layouts

Prepare four new mazes while the fixed twelve-run training-method comparison
finishes. The existing sixteen-cell tree generator, metric geometry and
farthest-cell goal rule are unchanged. Construction seed 2026091517 accepts
the first four eligible candidates from six examined; two fail the existing
route-structure rule. No candidate is selected using navigation outcomes.

All four abstract topologies and grid embeddings differ from each other and
the explicit 64-layout source registry, including the four layouts used in the
current training comparison. This is not a claim about every historical maze
or generalization beyond the generator's family. These are ordinary prospective
development layouts, not sealed final evaluation.

Source: `lewm/post_training_comparison_layouts_development.py`.
Inventory: `docs/go2_post_training_comparison_layout_inventory_2026-09-15.json`.
SHA-256: `7ee6653289d294189abb0f7641f91b8d13f26d5670c952eaae0f45c7acda731e`.
Physics seeds are 2026097600–2026097603; appearance seeds 2026097700–2026097703.
The existing scene-pack function produced the expected scene identities and
wall-object counts for each specification. The public mission contains only
goal coordinates, home coordinates and the return requirement. Topology and
shortest-route data remain within scene construction and evaluation.

No model, training input or native simulation was run on these layouts.
Finish all twelve current assignments and the retained JEPA layout-3 tracking
diagnosis before choosing the next frozen controller/control-arm comparison.
These prepared layouts do not change the running study or authorize calling
its exposed layouts unseen again. Future navigation must preserve all assigned
outcomes; construction and scene-pack checks alone prove no navigation success.

## Fixed prospective comparison after the perception probe

The preceding twelve training assignments are complete (JEPA 2/4 round trips,
direct 2/4, supervised rollout 4/4). The JEPA layout-3 tracking failure was
reproduced and diagnosed as an inconsistent independent-plane offset constraint
under gyro rotation. A shared-normal paired-floor revision completed all 450
recorded frames and then achieved a physically verified live round trip on that
exposed layout, with zero disallowed contacts. The original failure is preserved.
Details: `docs/go2_gyro_coherent_floor_constraint_2026-09-15.md`.

Before any native execution on the new layouts, fix all sixteen assignments:
JEPA, direct, supervised rollout and heading-first reactive on each index 0–3.
Use the existing seed-2026091001 frozen models and their matched frozen visual
motion corrections for the three predictive arms. No new training, fit, layout
replacement or controller tuning occurs within this cohort. Every failure
counts. All arms use `GyroCoherentFloorMotion`, the existing raw-height floor
registration, independent raw-depth obstacle observer, current-plane floor map,
persistent routing memory, 2-mm noise recipe, six actions, 4,800 navigation ticks,
arrival tolerances and measured-simulation timing. The floor-reacquisition and
declared-gap treatments are absent.

The three predictive arms share planning/recovery code; reactive uses the
existing stronger heading-first instantaneous controller. This compares complete
controllers and frozen training pipelines with condition-specific corrections.
It does not isolate predictive scoring from recovery rules, numerical correction
fits from training, or establish a JEPA-specific advantage with one training seed.

Launcher: `scripts/run_go2_post_training_transfer_noise_development.py`, with
`--layout-index 0..3 --condition jepa|direct|supervised_rollout|reactive`.
Output template:
`go2_post_training_transfer_<condition>_noise_2mm_native_layoutXX_4800_v1_attempt_001`.
The native physical initializer binds both the new specification and scene-pack
function; all four packs and public mission fields were checked. Source-only
writer checks confirmed each model/correction assignment and shared tracker.
The first exploratory pack check passed an index instead of a specification;
the corrected source-only check passed, with no simulation or artifact created.

At most two native owners, including archive stages, run concurrently on the
established disjoint physical CPU groups. Even layouts run 0 then 2, odd layouts
1 then 3; condition order within each layout is JEPA, direct, supervised rollout,
reactive. The odd group may start first while the already-running layout-1
sensor replay finishes on CPU 0. No native owner shares that replay's CPU group.
About 60 GiB was free before the completed probe's archive; the sixteen-run
cohort is expected to fit at recent compressed-depth sizes. Record launch source
hashes and retain all sixteen outcomes and sensor recordings.

Execution began with JEPA layout 1 in session 54005 on the odd group. Its
launch records the new inventory hash and shared gyro-consistent tracker.
The longer sensor replay then exited successfully, freeing CPU 0 and the
even group for JEPA layout 0. Both are first assignments on their new scenes;
all sixteen outcomes remain pending at this entry.

## First independently verified new-maze outcome

JEPA layout 1 exited 0 after 428.81 s including archive, maximum RSS 15,861,256
KiB and zero swaps. Both physical arrivals pass: goal frame 1644 and home frame
2748. Maximum one-second dwell distances 10.175/21.684 mm, maximum 100-ms speeds
0.025245/0.019832 m/s, and all dwell requests zero. All 2,750 camera frames
published registered poses. Median/max position error 3.335/6.177 mm, zero
disallowed contacts; selected/on-time plans 672/655. Its normal summary and
executed-forecast analysis are saved in the run root. This is one of sixteen
fixed assignments; no controller ranking is established yet.

Direct layout 1 started next in session 40954 on the freed odd group. JEPA
layout 0 reported a round-trip candidate and is finishing its recording; owner
exit and independent physical evaluation remain pending at this entry.

`scripts/compare_go2_post_training_transfer_development.py --layout-index I`
will collect all four conditions after that layout is complete. It reuses the
physical-arrival summaries and checks shared mission/sensor settings, common
recorded source hashes and each predictive arm's actual correction bindings.
It preserves failures and reports the reactive recovery-rule difference.

## First two JEPA assignments verified

JEPA layout 0 exited 0 after 394.15 s including archive, maximum RSS 15,293,636
KiB and zero swaps. Independent evaluation verifies goal frame 1575 and home
frame 2628. Maximum dwell distances were 21.662/10.090 mm, quiet speeds
0.012506/0.016492 m/s, and all dwell requests zero. All 2,630 camera frames
published registered poses; median/max error 2.279/5.244 mm, zero disallowed
contacts. Selected/on-time plans 651/640. All selected-plan correction bindings
match the frozen JEPA fit. Direct layout 0 started next in session 41264.

The first two JEPA outcomes on the new layouts are therefore two goals and two
verified round trips, with zero disallowed contacts. Fourteen assignments remain,
including JEPA layouts 2/3 and all direct, supervised and reactive controls.
Direct layouts 0/1 are active on the two disjoint groups. No ranking or reliable
four-layout transfer claim follows from the two initial successes.

JEPA layout 1 travelled 19.863 m and finished 21.861 mm from home. Its 647
executed 700-ms forecast windows have corrected endpoint XY RMSE 7.654 mm,
maximum 25.715 mm, and zero windows with any corrected-path error above 30 mm.
Every correction binding matches the assigned fit. Forecast windows overlap
and are conditional on the executed trajectory; these are not calibrated
clearance guarantees.

## Direct layouts 0 and 1 verified; supervised trials started

Direct layout 1 exited 0 after 547.01 s including archive, maximum RSS
20,125,852 KiB and zero swaps. Goal frame 2143 and home frame 3680 pass physical
evaluation. Maximum dwell distances 25.090/26.887 mm, quiet speeds
0.007563/0.017379 m/s, all dwell requests zero. All 3,682 poses published,
median/max error 8.389/10.847 mm, zero disallowed contacts. Path 20.160 m,
selected/on-time plans 904/886, final home distance 27.064 mm. Its 879 executed
forecast windows have corrected endpoint XY RMSE 11.069 mm, maximum 41.662 mm,
and 18 windows with any corrected-path error over 30 mm. All correction
bindings match. Supervised rollout layout 1 started next in session 87535.

Direct layout 0 exited 0 after 547.66 s including archive, maximum RSS
21,077,972 KiB and zero swaps. Goal frame 2177 and home frame 3880 both pass
physical evaluation. Maximum dwell distances 21.228/21.772 mm, quiet speeds
0.022679/0.008818 m/s, all dwell requests zero. All 3,883 poses published,
median/max error 5.063/8.893 mm, zero disallowed contacts. Selected/on-time plans
961/955. All actual correction bindings match. Supervised rollout layout 0
started next in session 87880 on the freed even CPU group.

Four of sixteen outcomes are now evaluated: JEPA and direct each have two goals
and two round trips on layouts 0/1, all with zero disallowed contacts. JEPA
reached home 125.2 and 93.2 simulated seconds earlier on the two respective
layouts. These two pairwise observations do not establish general superiority;
the remaining twelve assigned outcomes, including the other two layouts, stay
fixed and pending.

Separate future-control work is recorded in
`docs/go2_command_motion_prediction_controls_2026-09-15.md`. Its command-only
analysis and pose/command fit do not modify any model, correction, source or
controller used by this native cohort. They address the unresolved contribution
of neural XY prediction after the current comparison finishes.

## Six new-maze round trips verified; reactive comparisons running

Supervised rollout layouts 0/1 both exited successfully and passed independent
physical goal and home checks, with zero disallowed contacts. Layout 0 published
all 3,664 poses, reached goal/home at frames 1889/3662, and had median/max
position errors 1.704/4.737 mm. Owner elapsed time including archive was 515.74 s,
maximum RSS 20,035,844 KiB, zero swaps. Layout 1 published all 3,358 poses,
reached goal/home at 2068/3356, and had median/max errors 2.853/5.728 mm.
Its 794 executed forecast windows have corrected endpoint XY RMSE 10.097 mm,
maximum 36.748 mm, with nine windows exceeding 30 mm anywhere on the path.

Six of sixteen assignments are now independently evaluated: JEPA, direct and
supervised rollout each completed both round trips on layouts 0/1. All six
have zero disallowed contacts. Reactive layout 1 is running in session 46075;
reactive layout 0 started in session 30128 on the freed even CPU group.
Supervised layout 0 summary/forecast processing is session 15747. All eight
assignments on layouts 2/3 remain pending. Storage has approximately 46 GiB
available; there is no current storage blocker. The fixed cohort is unchanged.

## Both reactive outcomes retained; second pair of layouts started

Reactive layout 0 exited 1 after 145.68 s, maximum RSS 6,954,744 KiB, zero
swaps. The registration worker rejected a current floor measurement as
conflicting with the transported reference. It acquired 925 camera pairs and
published 923 poses. Independent evaluation finds no arrivals, zero disallowed
contacts, median/max position errors 3.189/10.147 mm. This remains a failed
assignment; no replacement or controller change occurs within the cohort.

Reactive layout 1 exited 0 after 708.96 s including archive, maximum RSS
25,377,956 KiB, zero swaps. It exhausted the navigation budget with no arrivals.
All 4,805 poses published; independent evaluation finds zero disallowed contacts,
median/max position errors 4.488/12.042 mm. Its saved
`terminal_turning_diagnostic_v1.json` describes frames 2000–4800: 690 right-turn
plans and five holds, all 695 with nominal clearance and a route to the goal
cell; seven plans were late. Mission observations were inside the 20-mm radius
for at most six consecutive samples, and accumulated at most two quiet
intervals. This is a descriptive terminal-turning failure, not proof of its
precise dynamical cause or of a learned-prediction advantage.

Eight of sixteen assignments are now evaluated: each predictive method has
two goals and two round trips; reactive has neither on these two layouts. All
eight have zero disallowed contacts. Layout 0's four-condition comparison and
PNG/SVG trajectory plots are saved; its displayed trajectories were inspected.
JEPA layout 2 started in session 83540 on the even CPU group; JEPA layout 3
started in session 57228 on the odd group after reactive layout 1's owner exited.
The remaining order is direct, supervised rollout and reactive on each layout.

Supervised layout 0's completed forecast analysis contains 886 executed windows,
corrected endpoint XY RMSE 11.063 mm, maximum 64.489 mm, with nine windows
exceeding 30 mm anywhere along the corrected path. The large individual error
is retained; average accuracy is not a clearance or safety guarantee.

Both layout 0/1 four-condition comparison reports and PNG/SVG trajectory figures
are now complete and visually inspected. Shared recorded settings and all 128
common source bindings match within each comparison; every predictive plan's
correction binding matches its frozen assignment. Both reactive normal summaries
are also saved. These complete-controller results still do not isolate learned
XY prediction, reactive recovery behavior or the contribution of memory.

## JEPA layouts 2 and 3: verified goals, return speed stops

Both remaining JEPA owners exited 1 at the unchanged physical guard. Layout 2
owner elapsed time was 376.35 s, maximum RSS 14,754,996 KiB, zero swaps. It
recorded 2,520 camera pairs and 2,519 registered poses; independent evaluation
verifies goal frame 2040, no home arrival, zero disallowed contacts, median/max
position errors 7.619/10.549 mm. The terminal full 3-D instantaneous body speed
was 0.301030 m/s at physical time 253.492 s, exceeding the 0.3-m/s limit. Its
horizontal speed was 0.291266 m/s and 100-ms displacement speed 0.253417 m/s.

Layout 3 owner elapsed time was 253.99 s, maximum RSS 9,890,336 KiB, zero swaps.
It recorded 1,496 camera pairs and 1,494 registered poses; independent evaluation
verifies goal frame 1320, no home arrival, zero disallowed contacts, median/max
position errors 2.244/4.601 mm. Terminal full body speed was 0.300391 m/s at
151.012 s, horizontal speed 0.286451 m/s and 100-ms displacement speed
0.248394 m/s. Both were inside the domain and requesting/applied forward 0.2 m/s.
Each root retains `physical_speed_stop_diagnostic_v1.json`; neither failure is
converted into a round-trip success or rerun with a relaxed threshold.

JEPA's four fixed assignments are complete: four verified goals, two verified
round trips, two return-phase speed stops, zero disallowed contacts. There are
ten evaluated outcomes across the cohort, with eight verified goals and six
round trips. Direct layout 2/3 started next in sessions 30621/48948 on their
respective physical CPU groups. Six outcomes remain pending. About 40 GiB is
free on artifact storage.

The two failed JEPA roots also have normal summaries and executed-forecast
analyses. Layout 2: 625 selected plans, 611 fully executed windows, corrected
700-ms endpoint XY RMSE 9.101 mm, maximum 24.466 mm, maximum path error
27.241 mm. Layout 3: 369 selected plans, 360 windows, corrected endpoint RMSE
8.190 mm, maximum/path maximum 25.071 mm. Neither has a window with path error
over 30 mm. These XY errors do not predict or certify the instantaneous 3-D
speed guard, which both navigation assignments failed.

Separate completed-recording timing analysis is in
`docs/go2_post_training_host_timing_2026-09-15.md`; it changes no current clocks
or assignments. Reactive layout 0's delivered-depth sample diagnostic confirms
zero primary valid pixels on frames 920–924 while auxiliary depth remains
available. Full registration replay remains pending; this failure is distinct
from the earlier raw-tracker paired-floor inconsistency that the coherent
perception revision addressed.

The matching raw-depth follow-up
`terminal_primary_near_range_diagnostic_v1.json` shows all 307,200 primary
pixels finite and positive but below the 0.2-m near limit in each sampled frame
920–924 (frame medians 0.187 down to 0.178 m). Captured pixel hashes match.
These invalid subrange values remain evaluator-only. Thus the lost primary
coverage is a confirmed near-range blind spot; reproducing the downstream
registration conflict and testing a remedy are still separate work.

All 625/369 selected JEPA plans on layouts 2/3 match the frozen correction
root, hash and base-model assignment.

## Direct layout 3 round trip verified; supervised layout 3 started

Direct layout 3 exited 0 after 468.44 s including archive, maximum RSS
17,630,820 KiB and zero swaps. All 3,150 camera frames have registered poses.
Independent evaluation verifies goal frame 1783 and home frame 3148, with
maximum dwell distances 11.835/19.253 mm and speeds 0.004832/0.009653 m/s;
all dwell requests are zero. Median/max pose error 8.817/11.782 mm, zero
disallowed contacts, path 19.194 m, final home distance 19.214 mm. Selected/on-time
plans 780/760; every actual correction binding matches the frozen direct fit.
Its 764 executed forecast windows have corrected endpoint XY RMSE 10.997 mm,
maximum/path maximum 37.915 mm, and eleven windows exceeding 30 mm on the path.

Eleven assignments are independently evaluated: nine goals and seven round
trips, zero disallowed contacts across those eleven. Supervised rollout layout 3
started in session 17494. Direct layout 2 reports a goal at frame 2480 but
exhausted its 4,800-tick budget during return (4,805 recorded camera pairs).
Its owner session 30621 is still archiving; exit and physical evaluation remain
pending at this entry. Once complete, supervised layout 2 is next on that CPU
group; reactive follows supervised on each layout.

The additional speed-stop analysis is in
`docs/go2_post_training_restart_speed_diagnosis_2026-09-15.md`. The two JEPA
return failures share a 0.4-s arc, 0.4-s turn, 0.4-s hold and forward restart
pattern, but five other recorded instances do not stop during the observed
forward segment. None of the final actions invoked reserve recovery. This
narrows the transition/gait-state hypothesis without proving a causal remedy,
and no current controller or speed guard is changed.

## Direct layout 2 goal verified; both supervised trials running

Direct layout 2 exited 0 after 668.19 s including archive, maximum RSS
25,489,668 KiB, zero swaps. Independent evaluation verifies goal frame 2480;
the return remains incomplete at the navigation budget. All 4,805 camera
frames published poses, median/max position errors 4.232/9.696 mm, zero
disallowed contacts. Goal dwell maximum distance 6.373 mm and speed
0.018585 m/s, all requests zero. Final home distance 2.115 m, path 23.123 m,
selected/on-time plans 1196/1187. Every actual correction binding matches the
frozen direct fit. Its 1,188 executed forecast windows have corrected endpoint
XY RMSE 9.194 mm, maximum/path maximum 43.448 mm, and ten windows exceeding
30 mm anywhere on the path. Normal summary and forecast evaluation are saved.

The saved `return_model_hold_diagnostic_v1.json` counts 422 holds among 575
return-generation selected plans. In all 422, every candidate passes its full
geometric reserve check, and forward scores above hold if the contact-penalty
term is removed arithmetically. The actual 1.2-m penalty coefficient is unchanged.
This identifies the learned score's contribution to the recorded hesitation;
it does not establish that any unexecuted forward action was contact-free,
that the score is calibrated, or that a changed controller would finish.

Twelve assignments are now independently evaluated: ten goals, seven round
trips and zero disallowed contacts. JEPA has four goals/two round trips; direct
has four goals/three round trips. Supervised layout 2 started in session 62180;
supervised layout 3 remains active in session 17494. Reactive layouts 2/3 follow
on their corresponding groups. Four outcomes remain pending.

## Supervised layout 3 verified; reactive layout 3 started

Supervised rollout layout 3 exited 0 after 461.20 s including archive, maximum
RSS 17,370,296 KiB, zero swaps. All 3,102 camera frames published registered
poses. Independent evaluation verifies goal frame 1775 and home frame 3100;
maximum dwell distances 7.321/26.276 mm and speeds 0.033447/0.016477 m/s,
all dwell requests zero. Median/max pose error 2.046/7.136 mm, zero disallowed
contacts, path 19.626 m, final home distance 26.400 mm. Selected/on-time plans
761/734; every actual correction binding matches its frozen supervised fit.
Normal summary and forecast analysis are saved: 736 executed windows,
corrected endpoint XY RMSE 10.736 mm, maximum/path maximum 45.295 mm, eight
windows above 30 mm anywhere on the corrected path.

Thirteen assignments are independently evaluated: eleven goals, eight round
trips and zero disallowed contacts. Reactive layout 3 started in session 55498
on the odd CPU group. Supervised layout 2 remains active in session 62180,
in its return phase; reactive layout 2 follows after its owner exits and the
outcome is evaluated. No source, model, correction, threshold or assignment
in the fixed sixteen-run comparison has changed.

## Supervised completes four round trips; reactive layout 3 also succeeds

Supervised layout 2 exited 0 after 581.74 s including archive, maximum RSS
22,250,900 KiB, zero swaps. All 4,130 camera frames published poses. Independent
evaluation verifies goal frame 2544 and home frame 4128; maximum dwell
distances 13.577/16.697 mm and speeds 0.020385/0.014359 m/s, all dwell requests
zero. Median/max pose error 5.050/9.159 mm, zero disallowed contacts, path
25.800 m and final home distance 16.952 mm. Selected/on-time plans 1013/1001;
all actual correction bindings match. Its 985 executed windows give corrected
endpoint XY RMSE 10.978 mm, maximum/path maximum 65.113 mm, eleven windows
above 30 mm on the path. The normal summary and forecast evaluation are saved.

All twelve predictive assignments are complete. Each method reached all four
goals; round trips are JEPA 2/4, direct 3/4 and supervised rollout 4/4. These
are one-seed, four-layout development observations, not a statistical training
advantage claim. Reactive layout 2 launched next in session 92495.

Reactive layout 3 exited 0 after 304.02 s including archive, maximum RSS
11,919,848 KiB, zero swaps. All 1,971 camera frames published poses. Goal
frame 1319 and home frame 1969 pass independent evaluation; maximum dwell
distances 16.919/10.407 mm and speeds 0.011481/0.020977 m/s, all dwell requests
zero. Median/max pose error 2.683/5.764 mm, zero disallowed contacts, path
18.991 m, final home distance 8.451 mm, selected/on-time plans 483/475.
It returned sooner than the successful direct and supervised controllers on
this maze; JEPA had reached the goal but stopped during return. Layout 3's
complete comparison, reactive summary and PNG/SVG figures are saved and the
figure was visually inspected. Shared sources/settings and predictive correction
bindings match within the comparison.

Fifteen outcomes are independently evaluated: thirteen goals and ten round
trips, with zero disallowed contacts. Only reactive layout 2 remains live.
The freed odd CPU group is running a targeted coherent-tracker/floor-registration
replay of reactive layout 0's retained failure, session 49025. It changes no
native assignment and must finish before that CPU group starts another job.

`scripts/summarize_go2_post_training_transfer_development.py` is prepared to
collect all four completed comparison reports, including every failure and
predictive forecast analysis. It has been syntax-checked but not run on an
incomplete population. The summary output is
`go2_post_training_transfer_four_layout_summary_v1_attempt_001/result.json`.

The targeted reactive-layout-0 replay exited 0 after 107.525 s of recorded
sensor processing. It exactly matches all 923 published raw poses and reproduces
the same registration conflict at frame 923, with identical registered pose
error statistics. Results are retained under that root's
`gyro_coherent_floor_coherent_replay_v1/`. Native truth was loaded only after
estimation. The odd CPU group is now free; no new native assignment is launched
there before the fixed sixteen-trial population finishes. The rejection is in
the missing-plane transport check against the retained floor reference; the
already-confirmed primary near-range blind spot is present, but a corrective
intervention has not yet been tested.

## Complete sixteen-assignment result

Reactive layout 2 exited 0 after 350.33 s including archive, maximum RSS
14,139,168 KiB, zero swaps. All 2,441 camera frames published poses. Independent
evaluation verifies goal frame 1679 and home frame 2439; maximum dwell
distances 16.226/13.168 mm and speeds 0.002752/0.017466 m/s, all dwell requests
zero. Median/max pose error 3.962/9.245 mm, zero disallowed contacts, path
25.296 m, final home distance 13.026 mm, selected/on-time plans 599/594.
Its normal summary and layout-2 four-condition comparison/figures are saved;
the figure was visually inspected. All four comparisons and the complete
aggregate are now saved. Shared implementation bindings match across layouts.

| Controller | Verified goals | Verified round trips | Disallowed contacts |
| --- | ---: | ---: | ---: |
| JEPA | 4/4 | 2/4 | 0 |
| Direct | 4/4 | 3/4 | 0 |
| Supervised rollout | 4/4 | 4/4 | 0 |
| Heading-first reactive | 2/4 | 2/4 | 0 |

All sixteen original outcomes remain, including two JEPA return speed stops,
direct layout 2 return-budget exhaustion, reactive layout 0 registration failure
and reactive layout 1 terminal-turning budget exhaustion. JEPA and reactive
succeed on opposite pairs of layouts. Reactive returned sooner than the
successful predictive runs on layouts 2/3. Thus the study does not establish
a general JEPA advantage; it identifies supervised rollout as the strongest
observed complete controller in this small development population. Models have
one training seed and condition-specific corrections; reactive recovery differs.
Host real-time, calibrated real sensing and hardware validation remain open.

The next eight-run XY-source intervention is documented in
`docs/go2_command_motion_prediction_controls_2026-09-15.md`. Before any ablation
run, its reference was changed from the earlier unexecuted JEPA proposal to
supervised rollout, based on the completed controller outcomes. All current
sixteen recordings stay retained. The first new learned-XY reference assignments
on layouts 0/1 started in sessions 70329/7058 on the even/odd groups respectively.
They compute both forecast alternatives, keeping the later pose/command arms
matched in computation structure. No result from these new trials is available
at this entry, and none changes the completed sixteen-run comparison.
