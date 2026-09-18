# Learned yaw versus integrated command yaw

The saved contact-pilot trajectories show lower executed yaw error from command
integration than from the neural yaw output, but those retrospective errors do
not establish a navigation benefit. The repaired frontier follow-up has now
completed a physical goal/return with pose/command XY, learned yaw and disabled
contact on the previously stalled layout. Keep that exploration policy fixed
and test the remaining neural motion channel in live navigation.

Before dispatch, fix four assignments: learned and command yaw on each of the
already exposed development layouts 0 and 1. Both use the same frozen supervised
network, fitted pose/command XY, disabled contact score, repaired
arrival-conditioned frontier views, coherent perception, persistent routing
memory, 2 mm depth noise, action bank, terminal pulses, physical guards and
4,800-tick budget. No training, refitting or parameter selection occurs within
this comparison. Preserve all four outcomes, including any failures.

`lewm/yaw_source_ablation_development.py` computes both yaw alternatives in both
arms, then changes only forecast channels 2:4 (sine/cosine yaw) in the command
arm. It uses the existing SE(2) command integral and actual planned pulse mode.
The pulse mode is read from the runtime field: its saved receipt is added only
after selection returns. Actual final forecasts and both alternatives are
recorded separately from the earlier XY/contact correction stages.

The command arm's scored forecasts contain fitted pose/command XY, integrated
yaw and zero contact scoring. The network is still evaluated in both arms.
This isolates neural yaw conditional on this controller; it does not test the
benefit of predictive planning itself and is not a model-free comparison because
the XY predictor is fitted. Integrated command yaw models neither inertia nor
slip. Two exposed-maze pairs cannot establish broad superiority or reliability.

Launcher: `scripts/run_go2_yaw_source_ablation_development.py`, arguments
`--layout-index 0|1 --yaw-source learned|command`.
Root template:
`go2_yaw_source_ablation_<source>_noise_2mm_native_layoutXX_4800_v1_attempt_001`.
Run the two learned-yaw references first, then their command-yaw counterparts.
Keep each layout on its original physical CPU group: layout 0 on 0–7,16–23;
layout 1 on 8–15,24–31. At most two native owners, including archiving, run
concurrently. The inherited writer's CPU restriction is honored.

Independently evaluate physical goal/return, contacts, failures, completed
frontier views, executed motion forecasts and applied command/late-plan counts
after each owner exits. Compare actual scored yaw/motion channels and the fixed
shared settings across each pair. Keep earlier contact/frontier outcomes intact;
fresh references are required because the exploration policy and asynchronous
trajectories have changed. Measured-simulation timing remains explicit; no
hardware or real-time qualification follows from these runs.

The focused tests check command-arm invariance to neural-yaw perturbations,
unchanged XY/contact/input values, learned-arm identity, and actual returned and
recorded pulse yaw. The four yaw/frontier tests passed before source inspection
identified the pulse-receipt timing detail; the yaw adapter/test were corrected
to use the actual runtime pulse field and are rechecked before dispatch.

The retention review releases only seven completed predecessor success depth
recordings. All original outcomes and other sensor/physics/command records
remain, together with the five failed and four supervised full references and
all newer XY/contact/frontier recordings. The retirement reclaims 11.263 GiB
to make room for these four native archives under the existing user authorization.

The corrected yaw adapter's two tests passed (1.78 s). All four launch-writer
assignments passed in memory under their prescribed affinities, checking actual
yaw/contact/XY flags, layout identity, repaired view metadata and fixed four-run
scope. About 17 GiB of artifact space and 76 GiB RAM are available before launch.

The learned-yaw references launched on layouts 0 and 1 in sessions 81517 and
2677, owners 3615988 and 3616024, respectively. Both owners are confirmed live;
actual launch records match learned yaw, disabled contact, pose/command XY and
the fixed four-assignment scope. Command-yaw counterparts remain queued until
the corresponding reference owner finishes its archive and evaluation.

The yaw evaluator now reads the final `applied_prediction_after_yaw_ablation`
for the actual applied source and retains the upstream neural prediction as a
saved alternative in the command arm. It reports applied, neural and command
errors on identical executed windows. All original neural/command aggregate
values matched the saved 927-window pilot reference exactly in a memory-only
legacy check. The paired comparator also verifies final scored forecast
channels, actual pulse yaw, fixed fits, shared settings and runtime sources.
These are analysis-only changes; no running controller source changed.

## Layout-1 reference failed; fixed command counterpart dispatched

The learned-yaw layout-1 owner exited 1 after 114.85 s including archive, peak
RSS 5,731,484 KiB, zero swaps. It acquired 648 camera pairs and published 632
poses before `tracking: ValueError('measured visual pose unavailable')` stopped
the run. Independent evaluation records no arrivals, zero disallowed contacts,
median/max position error 4.055/6.134 mm and final goal distance 2.692 m.
The precise perception cause is not yet diagnosed; the full recording remains.

Its 155 executed windows have XY RMSE 5.302 mm, maximum 18.437 mm and no path
error above 30 mm. Same-window yaw endpoint RMSE is 3.158 degrees neural versus
1.925 command; commitment-yaw RMSE is 3.068 versus 1.767 degrees. The applied
yaw was neural. These trajectory-conditional errors do not establish a different
navigation outcome.

The layout-1 command-yaw counterpart started in session 83388 on its unchanged
CPU group after reference evaluation. Layout-0 learned yaw remains active.
All four assignments remain fixed; neither the perception failure nor the
slow layout-0 exploration prompts an in-cohort controller change or retry.

## Three outcomes recorded; final command-yaw assignment launched

Both previously live owners have now exited. Learned yaw on layout 0 exhausted
its 4,800-tick mission budget without either arrival: 4,805 registered poses,
zero disallowed contacts, median/max position error 2.999/11.547 mm and final
goal distance 3.441 m. Its 830 executed windows have corrected XY RMSE 5.254 mm.
Command yaw on layout 1 stopped after 1,172 camera pairs with a registration
failure: `current measured candidate conflicts with transported floor reference`.
Independent evaluation and saved XY/yaw evaluations are recorded for both.
The exact causes of the current exploration stall and perception failures remain
to be diagnosed after the fixed comparison.

The layout-1 paired comparison completed: neither yaw source achieved a round
trip and both had zero disallowed contacts. Source/channel matching passed.
Its report is `go2_yaw_source_ablation_comparison_layout01_v1_attempt_001/result.json`.
The final fixed command-yaw layout-0 assignment launched in session 24614 after
its reference owner exited and evaluation completed. Its launch record confirms
command yaw, pose/command XY, four fixed assignments and measured-simulation
timing. No runtime or parameter changes were made within the study. About
12 GiB of artifact storage remained before this final launch.

## Recorded failure diagnosis while the last assignment runs

Exact coherent sensor replays reproduced both layout-1 failures. All 632 saved
raw poses match for learned yaw, which fails at frame 632; all 1,170 saved raw
poses match for command yaw, which fails at frame 1170. Reports are under each
root's `gyro_coherent_floor_coherent_replay_v1/`. The layout-1 comparison plot
has been rendered and visually inspected.

At command frame 1170, delivered primary depth has zero valid pixels: all
307,200 native pixels are below the fixed 0.2 m near limit (median 0.181588 m).
Auxiliary depth retains 183,046 valid pixels. This establishes a primary blind
spot coincident with the reproduced registration conflict; the residual-level
cause and a successful recovery intervention remain untested.

At learned frame 632, both cameras retain all 307,200 valid pixels. All eight
retained anchors (frames 609–616) fail rigid-pose matching. The measured bridge
budget is exhausted despite valid consecutive-frame fits: primary/auxiliary
inliers 57/42, fractions 0.950/0.977, RMS residuals 0.767/0.671 mm, each spanning
12 image cells. Their incremental translation estimates differ by 0.100 mm.
This local agreement does not bound accumulated drift or prove that promotion
would succeed. `tracking_failure_diagnosis_v1.json` preserves the compact
readout; the complete failure snapshot remains available.

The learned layout-0 map reconstruction consumes 1,202 actual delivered mapping
frames, matches all 847 saved planning-map count receipts, all 14 view exclusion
events and the exact 38-cell final exclusion set. At planning frames 3180 and
3636, all 15 reachable frontiers are excluded. Removing only the upper-branch
exclusions restores a route to (0.525, 2.625). The upper passage is observed and
coarsely traversable but disconnected from the robot's observed component.
Thus arrival-conditioned views alone did not prevent premature exhaustion on
this trajectory. Saved route queries are not alternative navigation evidence.
The map diagnosis PNG/SVG was inspected. Analysis now handles deferred view
events; no running controller source changed.

## Complete fixed four-run result

The final command-yaw layout-0 owner exited 0 after 457.73 s including archive,
peak RSS 17,764,816 KiB, zero swaps. All 3,174 camera pairs have registered poses.
Independent physical evaluation verifies goal at frame 2061 and home at 3172;
maximum native distances during the quiet dwells are 13.252 and 13.475 mm,
maximum 100 ms speeds 0.017077 and 0.008794 m/s, all dwell commands zero.
Zero disallowed contacts. Position error median/max 5.068/7.507 mm, travelled
22.955 m, final home distance 13.428 mm. Its 662 executed windows have XY RMSE
6.413 mm and maximum error 28.466 mm. Neural versus command yaw endpoint RMSE
on these same windows is 3.062 versus 1.562 degrees. Actual applied yaw is command.
There were 786 selected plans, 669 on time and 117 late; maximum host/simulation
lag was 10.357 s. This successful measured-simulation run is not real-time evidence.

| Yaw source | Verified goals | Verified round trips | Disallowed contacts |
| --- | ---: | ---: | ---: |
| Learned | 0/2 | 0/2 | 0 |
| Command integration | 1/2 | 1/2 | 0 |

All four assignments and both comparisons are complete. Final scored channel
bindings and 138 common runtime source hashes match. Both pair PNG/SVG plots
were inspected. Aggregate:
`go2_yaw_source_ablation_two_layout_summary_v1_attempt_001/result.json`.
The command yaw alternative has lower same-window endpoint error on all four
recorded trajectories, but the small exposed-layout comparison does not prove
statistical superiority, equivalence or general reliability. Both arms retain
fitted XY prediction, predictive selection and persistent routing memory. The
network is evaluated in both, while its outcome channels do not score actions
in the command arm. The observed result establishes no learned-yaw advantage.
All four full recordings and failures remain; about 9.6 GiB artifact space is
free after completion. No native owner remains from this cohort.

Next scientific work should target the diagnosed bottlenecks: test visual
reference renewal with measured drift evaluation, and determine why a close
frontier view leaves a traversable passage disconnected before changing the
exploration rule again. Primary near-range loss also needs a prospective
observation-recovery test. Preserve these fixed results; subsequent changes
belong to a separate development intervention.

## Why the close upper-branch views miss the floor gap

A posthoc projection of four missing floor cells along map row 52 uses only the
saved registered poses, fixed floor hypothesis and calibrated camera transforms.
Across the recorded standoff-view frames 808–1080 and approach/close-view frames
1084–1508, cells (11,52), (12,52) and (13,52) are never wholly inside either
camera image at a mapping observation. During standoff views their body-relative
horizontal distances span approximately 0.324–0.507 m. Closer views reduce these
distances further. Cell (14,52) is fully projected in 14 standoff camera views
and one approach view. Raw-pixel coverage probing over frames 1100–1508 finds
no fully visible projected square for any of the four cells.

The three inner cells therefore fail before a floor-height residual or noise
threshold can admit them. Approaching and rotating closer is not sufficient to
observe this gap with the forward-mounted cameras. This supports testing a
camera-visibility-aware viewing position, rather than merely extending views
or clearing exclusions. It does not certify the projected floor as free space
or establish a successful new navigation policy. Detailed receipts:
`upper_passage_camera_visibility_v1.json` and
`upper_passage_pixel_coverage_diagnosis_v1.json` in the learned-yaw layout-0 root.
