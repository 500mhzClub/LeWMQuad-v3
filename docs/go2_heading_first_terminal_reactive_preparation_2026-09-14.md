# Inactive stronger terminal-reactive comparison

Finish all eight assignments of the current post-repeatability transfer study
unchanged. This separate control hypothesis responds to terminal-convergence
failures on its first two reactive layouts; it is not selected by a native
launcher and provides no navigation result.

Both reactive trials physically entered the goal vicinity but failed the
required settled arrival. Their minimum physical distances were 13.588 and
29.768 mm. Minimum observed distances were 15.834 and 23.740 mm. Layout 0 had
only five observed frames inside the unchanged 20-mm radius, at most three
consecutively; layout 1 had none. They first came within 100 observed mm at
frames 1762 and 1514, leaving substantial budget for terminal control.

The current feedback law requests forward speed proportional to the cosine of
heading error, without scaling that desired speed by target distance. It then
chooses the nearest existing primitive. Near the goal it alternates arcs and
pure turns; shortening translation to the existing 100-ms pulse did not prevent
circling. The native matched comparisons therefore do not establish that
prediction was necessary for exploration. A stronger model-free terminal
controller is an appropriate additional control, even if it removes the
observed learned advantage.

`lewm/heading_first_terminal_reactive_development.py` prepares one fixed variant:
within the existing exact-goal terminal approach, turn toward the currently
observed target until absolute heading error is at most 0.1 rad, then choose
the existing forward pulse. The 0.1-rad tolerance is the existing panorama
alignment tolerance, not selected by a search. Inside the configured observed
arrival radius choose hold; the unchanged mission still requires its measured
quiet dwell. Preserve nominal current clearance and action eligibility, scan
behavior, every nonterminal selection, amplitudes, 400-ms turn duration,
100-ms translation pulse, 300-ms dispatch delay and zero tail. Use no model,
residual, integrated-command pose or future-state prediction.

Three focused tests passed in 2.38 seconds: misaligned turn versus aligned pulse
with correct durations, hold without declaring arrival, and preservation of
clearance veto, survey and nonterminal behavior. The same helper changes 388 of
758 terminal selections on layout 0 and 417 of 804 on layout 1. Every terminal
state in these original trajectories requests right turn under the new rule;
the saved trajectories never closely aligned with the target. These are saved
states, not executed counterfactual trajectories or independent samples.
Per-root results: `heading_first_terminal_saved_selection_v1.json`.

After completing the frozen current study, specify a prospective roster and
compare native outcomes without tuning between its runs. Preserve the original
reactive failures and report this treatment separately. Realistic sensing,
timing, broader repeatability, memory causality and hardware evidence remain
outstanding regardless of this control's result.

The separate launcher is now prepared:
`scripts/run_go2_heading_first_terminal_reactive_development.py --layout-index 0|1|2|3`.
It reuses the current scene, compact acquisition, CPU groups, perception,
mapping, mission and writer chain. All four writer configurations were
exercised without native execution, preserving every non-owner baseline field
and original source identity before adding the new treatment metadata and
its two source files. These are revisits to the fixed development layouts,
not four additional independent mazes. No native trial has been launched
with this terminal variant; finish the current learned pair first.

## Fixed four-run native follow-up

The preceding eight-run study is complete: learned 4/4 round trips, original
reactive 0/4, with the original terminal and close-wall failures retained.
Run the prepared heading-first reactive controller once on each of the same
four layouts, in two parallel batches: 0/1, then 2/3 after both first-batch
archives and owner exits. Complete all four without tuning or substituting
layouts according to outcomes. These are development revisits, not additional
independent layouts. Keep the existing 4,800-tick budget, arrivals, sensors,
mapping, action amplitudes, pulse durations, CPU allocation and source settings.

Compare all three conditions per layout: learned, original reactive and this
stronger reactive control. Record verified goals/round trips, contacts, timing
and terminal-rule activations. A better reactive outcome is evidence about
baseline strength; it does not prove JEPA or memory causality. The terminal-only
change does not directly address the pre-goal primary-depth blindness on layout
2. Any changed trajectory there requires separate interpretation.

Before first dispatch, RecoveryStorage has about 15 GiB free and available RAM
is about 76 GiB. This is sufficient for the first pair. Review eligible completed
diagnostic depth retirement before the later pair if needed; preserve the full
original eight-run comparison and its unresolved perception failure.

First pair dispatched around 21:08 local log time: layout 0 session 8492 /
PID 3465920; layout 1 session 15840 / PID 3465946. Both owners were confirmed
live and their actual launch records identify the heading-first reactive
condition, 0.1-rad tolerance and fixed per-layout CPU groups. Results pending.
Reviewed retirement of the completed terminal-priority predecessor probe's
depth reclaimed a further 2.6607 GiB while preserving its diagnostic records.

## First pair complete

Both owners exited 0 with all 4,805 camera pairs. Independent physics evaluation
verified outbound arrivals: layout 0 frame 2529, 12.712–15.801-mm dwell;
layout 1 frame 1808, 11.586–13.876-mm dwell. Both satisfied quiet motion and
zero requested commands, with zero disallowed contacts. Neither returned home
within the budget. Final home distances were 2.491 and 1.432 m; paths 26.953
and 24.841 m. Maximum pose errors were 9.548 and 13.914 mm.

The terminal rule ran on 10/39 plans and changed 7/22 actions respectively.
The return failures differ. Layout 0 had 413 missing current-observation
requests, followed by latched vetoes despite available nominal stored-map
clearance. Layout 1 selected hold for all 451 plans in the final 180 seconds:
stored clearance was 0.421–0.437 m, below the unchanged 0.45-m nominal radius;
its missing-observation count was zero. These recorded causes remain separate
from the solved outbound settling problem. Full depth remains available for
further physical/perception diagnosis. The per-root
`heading_first_terminal_and_return_diagnostic_v1.json` was saved before arrival
verification; its pending field describes that earlier diagnostic time only.

Both three-condition comparison results passed shared settings/source checks
and are saved in `go2_heading_first_reactive_comparison_layoutXX_v1_attempt_001`.
Owner elapsed times were 672.39/700.21 seconds, peak RSS
25,205,260/25,351,812 KiB, zero swaps. With both owners absent, about 11 GiB
disk and 76 GiB RAM available, proceed with the fixed layout-2/3 pair unchanged.

Both three-condition PNG/SVG comparisons were generated and visually inspected.
The trajectories show completed outbound settling followed by distinct return
stalls. Physics over frames 3000–4804 puts layout 0's base centre 0.495–0.534 m
from the nearest wall, with only 32 mm net displacement. Its primary depth was
still partly valid at frame 3100, but had zero valid pixels at sampled frames
4000/4800 while auxiliary depth and floor fits remained available. Layout 1's
actual base-centre wall distance was 0.443–0.459 m (final 0.445 m), so the
stored-map stop is not explained simply by a fictitious obstacle. Its current
sampled-obstacle distances were 0.492–0.515 m, illustrating that currently seen
points need not include the nearest physical surface. Neither measurement is
an articulated clearance certificate. Per-root physical-clearance and primary
depth availability diagnostics retain the evidence.

Final pair dispatched around 21:21 local log time: layout 2 session 7897 /
PID 3470601; layout 3 session 32094 / PID 3470637. Both owners and fixed
heading-first launch settings were confirmed live. Results remain pending.

## Layout-3 tracking failure retained

Layout 3 exited 1 after 1,018 acquired pairs and 996 registered poses, with
`tracking: measured visual pose unavailable`. No arrival completed and no
disallowed contact occurred. Independent evaluation of the admitted poses
found median/maximum error 2.972/5.891 mm. Minimum physical goal distance
was 2.916 mm, but proximity alone did not meet the settled-arrival requirement.
Path length was 10.797 m; 244/248 plans were on time. The terminal rule marked
seven plans and changed zero selected actions, so this failure cannot be
attributed to a changed terminal action. Asynchronous trajectories can differ
before any intervention. The original reactive arm reached its goal on this
layout; both outcomes remain in the comparison.

The archive, failure, physics evaluation, summary and resource record are saved.
Owner elapsed time was 165.17 s, maximum RSS 7,484,128 KiB, zero swaps. Its
three-condition comparison passed the shared setting/source checks and its
PNG/SVG plots were visually inspected. A public-sensor replay using the
unchanged tracker is running on the released CPU group to identify the detailed
failure; no source or active layout-2 trial was changed.

The unchanged public tracking replay completed in 85.50 s: all 996 original
accepted raw poses matched exactly, then failure recurred at frame 996.
Incremental visual motion was still available, but all eight retained references
(frames 972–979) lacked enough rigid-pose matches, exhausting the existing
bounded measured bridge. The replay read no native physics and did not replay
registration. Full evidence is in `tracking_failure_public_replay_v1/result.json`;
the concise per-root `tracking_failure_public_replay_summary_v1.json` records
the failure chain. No bridge allowance or admission threshold was changed.

## Complete four-run stronger-reactive result

Layout 2 exited 0 with 4,805 pairs and a verified outbound arrival at frame
3564: physical dwell 15.229–23.115 mm, maximum 100-ms speed 0.02016 m/s,
all requested commands zero. No round trip or disallowed contact occurred.
Path length was 18.883 m, final home distance 2.748 m, maximum position error
15.287 mm, and 1168/1179 plans were on time. The terminal rule marked 414
plans and changed 383 actions. Its return tail selected mainly right turns,
with 264 missing current-observation requests over the run. Owner elapsed time
was 670.02 s, maximum RSS 25,064,944 KiB, zero swaps. The return failure remains
available for further depth diagnosis.

All four fixed stronger-reactive assignments are now terminal, with every
failure retained. Stronger reactive: 3/4 verified goals, 0/4 round trips.
Original reactive: 1/4 goals, 0/4 round trips. Learned: 4/4 goals and round trips.
All twelve recorded runs had zero disallowed contacts. The eight original runs
are reused comparison context, not eight new samples. All four three-condition
comparisons passed shared settings/source checks, and all PNG/SVG trajectory
figures were visually inspected. Combined result:
`go2_heading_first_reactive_four_layout_summary_v1_attempt_001/result.json`.

Better terminal feedback improved outbound completion in this development
follow-up, but did not close the round-trip gap. Observation loss, insufficient
clearance and tracking failure remain. The complete-controller comparison still
does not isolate predictive planning, JEPA training or memory causality; broad
reliability and realistic sensing/timing/hardware validation remain outstanding.
Proceed with the already fixed eight-run routing-memory experiment.
