# Decision-headroom audit: Phase 0 and Phase 1 working record

The [final handoff](go2_decision_headroom_agent_handoff_final_2026-09-23.md)
is the authoritative research specification. It is already saved at the
requested `docs/` path. Andrew explicitly authorised Phase 0 and the bounded,
blinded Phase 1 on 23 September. It supersedes the prior audit draft and
Stages B–D of the progress-report roadmap. No further fitting, controller
repair, candidate expansion or artifact retirement is authorised. The previous
navigation objective does not extend this scope.

**Current state, 19:36 BST:** the native pilot is stopped and closed out after
an RGB-restoration defect. Both completed sources passed physical restoration
at all eight states; none of their 384 source-replay images matched bitwise.
The third source was interrupted before physics. There is no live pilot owner.
See the [complete Phase 1 result and bounded recheck proposal](go2_decision_headroom_phase1_result_2026-09-23.md)
and [hashed technical-stop configuration](go2_decision_headroom_phase1_technical_stop_2026-09-23.json).
The proposed renderer-cache patch is not applied, and no further physics is
being scheduled without explicit approval of the new fixed replay population.
The full checkpoint-(a) Phase 2 protocol remains unqualified; earlier entries
below retain the execution history.

## Stage A complete and unchanged

All four original assignments and their physical readers completed by 19:22 BST.
The coordinator's terminal result is `COMPLETE`; both coordinator and final
native owner exited. The existing-data and maze-data matched heads each finished
0/2 round trips, with no goal/home arrivals, disallowed contacts or pipeline
faults in any cell. All runs exhausted 480.32 simulated seconds. The full table,
fourth-run diagnostics and retained initial-head comparisons are consolidated
in [the Stage A record](go2_maze_view_readout_navigation_2026-09-23.md).
Earlier timestamped observations below are the execution history.

## Pilot prephysics qualification stopped; diagnosis retained

**Update, 19:28 BST:** the two erroneous unsafe-forward labels now have an
explicit analytical erratum in `reference_sanity_v1/corrected_qualification.json`
(SHA-256 `6fc574d27e549e743ec7833ccaef77bb706e67cae2030a2f579579815ccefda5`).
All 24 original optimal sets passed; the two safety labels were corrected after
execution, and must not be described as prewritten successes. No cost,
geometry, evaluator parameter or original expectation file changed; no case
was re-executed. The exact original strict failure remains at
`reference_sanity_v1/result.json`.

The narrow [prephysics continuation](../scripts/continue_go2_decision_headroom_prephysics_development.py)
requires zero prior source, snapshot, branch and component-timing attempts,
the exact documented correction, unchanged caps/inputs/controller sources,
and current resource admission. It archives the first failed closeout intact
under `prephysics_closeout_v1/`, carries its wall/CPU/cache accounting forward,
and counts retained original evidence against the same storage cap. It cannot
restart a native source trial or run a second continuation. Source collection
started as PID 304282, creation time 1790188129.85 (execution session 70238),
with the original first command-history assignment. Revalidate this live fact
before acting. No Phase 2 selections or rankings are authorised.

The queued launcher entered admission after Stage A closed, then exited with
the recorded reference-sanity failure. No source rollout, snapshot, physics
branch or component-timing pass was attempted. Owner runtime was 2.775 s,
aggregate CPU 2.12 s, and retained bytes at resource closeout were 153324.
The reader reports `PILOT_INVALID_OR_INCOMPLETE`: the planned 24 physical slots
are uncollected, not observed restoration failures. Physical repeat, RGB
restoration and per-state compute/storage measurements remain unavailable.

The analytical panel passed all 24 prewritten optimal-action sets and all
geodesic requirements. It failed two additional unsafe-action assertions:
`wall_ahead_2` and `wall_ahead_3` incorrectly expected forward to be unsafe.
Their geometry leaves 10/20 mm at the endpoint, with conservative swept
lower bounds 9.718/19.718 mm, exceeding the specified 5-mm clearance.
This is a fixture-label error, not evidence of a wrong selected optimum.
The [diagnosis and original result binding](go2_decision_headroom_reference_fixture_diagnosis_2026-09-23.json)
preserve both failures. Do not tune clearance or the reference formula to force
the incorrect labels, or silently present corrected labels as prewritten.

The original cap configuration, panel, evaluator and failed attempt remain
unchanged at this checkpoint in the working record. A routine audit correction
must preserve these observations, keep the physical source/branch caps and
cumulative compute/storage accounting, and avoid rerunning completed native
trials. No Phase 2 or additional artifact-retirement authority exists.

Pilot root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_decision_headroom_phase1_v1_attempt_001`.
Its `reference_sanity_v1/result.json`, `failure.json`, `resource_result.json`
and `pilot_validity_report.json` are the authoritative failed-prephysics
records. Later draft-preparation statements below describe the state before
this first execution, not current qualification.

## Stage A execution history

The retained plan fixes layout/readout order as 00/old data, 00/maze data,
02/maze data, 02/old data. Both layout-00 owners and physical readers are
complete: budget exhaustion with no goal/home arrival, no disallowed contact
and no pipeline fault. Their full outcomes and executed-window errors are in
[the Stage A study record](go2_maze_view_readout_navigation_2026-09-23.md).

Latest check at 18:12 BST: coordinator PID 250817 completed the layout-02
maze-data assignment and its physical reader, then started layout-02 old data.
The third result exhausted 480.32 simulated seconds with no arrivals, contacts
or pipeline faults, and 1140 holds among 1198 plans. Its executed-window results
are consolidated in the Stage A record. These are timestamped observations,
not permanent ownership facts. Recheck process handles and result records before acting.
No duplicate, restarted or extended trial has been launched.

At 18:29 BST the fourth owner, PID 293613 (creation time 1790183473.7),
had reached frame 1220 and 303 model calls. The coordinator remained live.
All four source digests in Stage A's retained plan still matched the workspace.

At 19:00 BST the same fourth owner and coordinator were live; the native log
had reached frame 3400/4800 and 848 model calls, still outbound without an
arrival. At 19:02 BST a single waiting launcher was started (execution session
9081). It performs no model or physics work while either original owner is
live. It requires all four assignment results and the coordinator's final
result, then enters the existing bounded pilot's admission exactly once.
Missing terminal results or an existing pilot directory stop that launcher;
neither condition permits restarting a trial. The pilot's own resource and
Stage A checks run again after this wait. This is queued execution, not pilot
validity evidence. No audit/controller source was changed for the launcher.

Source root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_maze_view_readout_navigation_v1_attempt_001`.

## Numerical pilot limits recorded before new physics

The [working protocol draft](go2_decision_headroom_protocol_draft_2026-09-23.md)
now records the sampling, information-boundary and analysis structure before
pilot outcomes. It explicitly leaves numerical decisions requiring technical
measurements open. It is neither the frozen checkpoint-(a) protocol nor
permission to execute comparative rows.

The [Phase 1 cap configuration](go2_decision_headroom_phase1_caps_v1_2026-09-23.json)
sets hard maxima, including failed attempts:

| Resource | Ceiling |
|---|---:|
| Exposed same-family pilot layouts | 2: existing indices 0 and 2 |
| Source runs | 6: two per frozen source-controller class |
| Source camera frames per run | 161 |
| Source snapshot frames per run | 12, 52, 92, 132 |
| Snapshots | 24 |
| Candidate branches | 6 candidates × 3 repeats × 24 states = 432 |
| Recorded-trace restoration attempts | 3 × 24 = 72 |
| Total branch/replay attempts | 504, each exactly 800 ms including prefix |
| All source/branch physics, including settling | 508.8 simulated seconds |
| Independent reference sanity panel | 24 prewritten analytical physical cases |
| Pilot execution wall time | 4 hours |
| Aggregate CPU time | 64 core-hours |
| GPU-owner wall time | 4 hours; one owner |
| Aggregate resident RAM / GPU allocation | 48 / 24 GiB |
| Minimum available RAM / VRAM | 16 / 4 GiB |
| Retained data / peak additional writes | 8 / 12 GiB |
| RecoveryStorage / workspace reserve | 12 / 4 GiB |

The proposed source runs retain the source controller's normal mission budget
and settings but stop collection after the declared short prefix; they are
audit-state collection, not navigation-performance trials. No mission outcome
comparison is permitted. Pilot layouts remain exposed and cannot enter a
never-examined subset. The analytical sanity panel qualifies the independent
cost only; it cannot substitute for physical restoration/repeat evidence.

No Phase 1 physics or new GPU work has started. These caps are not a measured
Phase 2 budget and do not constitute the frozen checkpoint-(a) protocol.
Source/branch ownership will remain sequential. Admission must wait for all
Stage A assignments and readers to terminate, then recheck actual free space,
memory, resolved input/output/temp/cache paths and owners.

## Tooling preparation

The existing native machinery already inventories mutable solver fields,
including contact/constraint state omitted by a simple scene-state copy.
The audit adapter in
[decision_headroom_snapshot_development.py](../lewm/decision_headroom_snapshot_development.py)
reuses that inventory and serialization machinery, with the current paced
100-ms observation / 20-ms command / 2-ms physics boundary. It adds the paced
limiter window and previous-command state, locomotion action history and RNG
state. Controller/model sources are unchanged.

This adapter has passed parsing only. It is **not fidelity-qualified**.
The old oracle runner's block-boundary check does not describe the current
paced loop; the audit adapter has a separate boundary identity. Physical
qualification must reproduce the complete source applied-command trace at
all eight horizons, not just its first chosen action. Renderer state, branch
sampling and the complete frozen decision-input packet still need to be
integrated and verified. No historical trajectory is assumed restorable.

Decision packets must retain native RGB/context, causal body/command history,
observed pose/map, route and mission state, commitment ledger, actual six
candidate tapes and all selector inputs/masks. Privileged physics/geometry
must remain in separate evaluator records. No raw depth or dense features
will be retained; if this prevents faithful reproduction, stop for the scoped
amendment required by the handoff rather than silently omit a needed input.

The snapshot adapter now also captures the body/gyro sensor buffers and sensor
timestamp state. Those wrappers enforce ordered samples, so restoring only
physics would leave their clocks in the future when replaying a branch. This
is an audit-state addition; the source sensing remains unchanged.

The [24-case sanity panel](go2_decision_headroom_reference_sanity_v1_2026-09-23.json)
now contains expected optimal sets and required exclusions written before any
cost execution. It covers the six requested categories, with four rotations
and small variations each. The preparatory-turn cases require a geodesic
detour around a wall, so a direct Euclidean shortcut cannot qualify them.
These analytical component cases are not simulator rollouts or extra layouts.

The draft [independent physical reference](../lewm/decision_headroom_reference_development.py)
uses true wall geometry, disk-inflated geodesic distance, heading alignment,
opposing-motion braking and phase-appropriate settling, all in seconds.
Safety is an acceptability constraint, not a finite penalty. It retains cost
components, contacts and swept-clearance bounds. Incomplete horizons have no
finite cost. Its provisional 0.46-m disk follows the existing geometry
calibration's conservative recommendation; this is not hardware validation.

The [sanity qualification runner](../scripts/qualify_go2_decision_headroom_reference_development.py)
consumes only the analytical cases and has no comparative model/scorer path.
It checks Stage A completion and live ownership before executing. The new
files have passed parsing only: **no sanity costs, rankings, restoration
measurements or pilot physics have been produced yet**. Geometry resolution,
near ties and the complete reference formulation still require qualification
before checkpoint (a); none is claimed frozen for Phase 2.

## Next bounded actions and stop

The source/branch orchestration is now drafted in
[the source collector](../scripts/run_go2_decision_headroom_source_development.py)
and [physics branch runner](../scripts/run_go2_decision_headroom_branches_development.py).
The [decision capture adapter](../lewm/decision_headroom_packet_development.py)
records the selector's entry arguments, native RGB context and controller state,
then lets the unchanged source selector run normally. It retains the resulting
actual six candidate tapes, canonical ordering, prefix and ordinary source
eligibility information. It executes no alternative selector.

The pilot source readout is fixed as `maze_view_old_data` for all three source
controller classes, including the workload-only model calls of the simple
controls. This chooses the named matched control before pilot outcomes; it
does not choose a winning head. Both existing heads remain required for the
eventual Phase 2 comparisons. Source assignment order is layout 0 command,
reactive, action; then layout 2 command, reactive, action.

After a source prefix is persisted and its controller stops, the drafted runner
reuses the same native scene for restoration and candidate branches. It thus
adds no extra settling rollout. Each replay uses all forty recorded applied
20-ms commands; each candidate uses its exact eight requested 100-ms slots,
including the committed prefix, through the existing limiter and policy.
RGB is retained at all eight horizons. Physical stops preserve partial traces
and do not fabricate later images or endpoints. Restoration is compared with
the source, while repeat variability compares repeats of the same candidate.

These are **physically unexecuted drafts**. The
[single-owner budget monitor](../scripts/run_go2_decision_headroom_pilot_development.py)
now admits only the fixed six source assignments, 24 snapshot identities and
504 branch/replay identities, including failed attempts. It checks cumulative
physics reservations, wall/CPU time, sampled RAM/VRAM, retained bytes, known
cache growth and filesystem reserves. Larger snapshot, source-recording and
branch-image writes require available space before execution. Admission binds
model/config inputs and audit/controller sources; temporary writes go under
the pilot root. Memory measurements are sampled, not operating-system limits.

The branch recorder includes per-sample and interpolated swept-disk clearance
with the declared footprint, preserving the limitation between native 2-ms
samples. Recorded-trace replays also compare native RGB pixel hashes with the
original source acquisition; physical and rendering evidence remain distinct.
No cost or alternative method ranking is computed on pilot branch states.

Five [synthetic audit checks](../scripts/check_go2_decision_headroom_pilot_development.py)
passed without physics, GPU allocation or model construction: systematic
restoration mismatch versus repeat agreement, intermediate contact/missing
horizon detection, between-sample obstacle crossing, prohibited retained inputs,
and exact source/snapshot/branch budget boundaries. These checks do not establish
physical restoration validity or qualify the reference cost. The actual pilot
root still does not exist; no pilot execution or new fitting has started.

A separate CPU-only constructor check serialized both captured controller
classes' initial non-plumbing state: 54 fields and 167461 bytes each. No
observations were submitted and no learned model or physics was executed.
Two preliminary check-harness calls failed before that result: the first
omitted required executor arguments; the second used a placeholder without
the required `modules()` interface. Supplying null, unused executors and an
evaluation-mode identity module corrected the check harness. Controller source
was unchanged. Initial-state serialization is not a claim that the complete
post-observation packet or native restoration has already qualified.

Subsequent preparation adds direct capture of the exact source route and map-
frame target. Terminal mission-coordinate targets retain their original initial-
frame coordinates. A view-seeking decision with no XY target is explicitly
flagged rather than assigned an invented positional target. The evaluator's
coordinate helper uses a single physical anchor at the first source RGB frame;
it never moves the target using the current true robot pose. A sixth synthetic
check covers that distinction. All six checks passed; all ten audit modules
parsed. No native scene or model was constructed during these checks.

The [component timer](../scripts/time_go2_decision_headroom_components_development.py)
uses at most two extra encoder and predictor calls per source and two readout
calls per existing head per source: 12, 12, and 12 per head across the pilot.
It records durations and discards decoded outputs without running a selector.
The [technical reader](../scripts/read_go2_decision_headroom_pilot_development.py)
consolidates the fixed population, all three pairwise repeat comparisons,
rendering identity, resources and missing records. Neither tool has executed
on pilot data. Three invalid/missing states among the fixed 24 make the declared
90% restoration requirement unreachable and stop further source assignments.

Before any sanity cost was evaluated, the four reversal examples were made
unambiguously face away from their target (2.8 radians plus the fixed variant),
replacing an oblique-approach draft. The 24 cases and expected optimal sets are
unchanged; this pre-execution clarification is recorded in the panel JSON.

## Precision preparation, with no pilot method comparisons

The [planning scenarios](go2_decision_headroom_precision_scenarios_2026-09-23.json)
use assumed layout-level standard deviations of 0.25, 0.5 and 1 second,
not variance estimated from pilot methods. A proposed practical margin of
0.25 seconds corresponds to 5 cm of nominal travel or 0.1125 radians of
nominal turning under the reference units. These are local proxy-cost scales,
not a claim that local regret sums predict mission-time savings.

For the five proposed primary contrasts (G, H_scorer, H_motion and D_learned
for both heads), the scenarios use 99% individual t intervals as a conservative
planning approximation for a 95% Bonferroni family. At assumed layout SD 0.5 s,
eight layouts give approximately 0.619 s half-width; targeting 0.125 s would
require approximately 110 layouts under that approximation. Actual layout
variance and interval coverage remain unknown. No sample size has been selected
from method outcomes, and these scenarios authorise no collection. The final
sampling design must combine these assumptions with the measured pilot budget,
retain an inconclusive outcome and identify an exploratory design honestly.

The file also distinguishes zero observed harm from proof of negligible harm.
Its binomial event-bound illustration is for any-observed-harm per independent
layout, not an exact interval for weighted state-level harm or paired excess
harm. The final estimand, uncertainty procedure and harm thresholds must be
fixed in the protocol before any comparative audit.

1. Finish Stage A unchanged and consolidate the four outcomes, including
   retained initial-head comparators with their exposure clearly labelled.
2. Complete the audit-only decision-packet/branch adapter and independent
   physical reference evaluator. Write sanity-panel expected answers before
   computing its rankings. Do not compute comparative audit rows in the pilot.
3. Run source capture, faithful restoration and three-repeat checks within
   the declared caps, measuring all resource use. Preserve failures; stop on
   a cap or an unresolved validity limitation.
4. Commit the proposed protocol and hashed configuration, including the fixed
   Phase 2 sampling/precision design and measured budget. Submit all evidence
   and **stop at checkpoint (a) for Andrew's explicit approval**.

Phase 2 collection, comparative scoring and the eventual recommended next
intervention remain unauthorised. Passing the pilot or committing its protocol
does not change that boundary.
