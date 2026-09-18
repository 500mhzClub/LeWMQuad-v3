# Matched surface-memory native goal probe V1

Prospective experiment: test whether retained surface evidence changes actual
goal-reaching/contact outcomes relative to current-frame surface evidence. Use
the fixed final family full-JEPA snapshot, SHA-256
`bcb8874e2adf89053463206267a4ccb90380909c324e303734a59b038f5b1821`, admitted by
the complete six-fit result
`ed3e2f6385991439fd390ffc64e647f6763fb3576b35f7c767fab19e4a29398c`.
No training, checkpoint search, variant-dependent model input or retuning occurs.

At each five-tick replanning decision, obtain all six original four-second learned
predictions and original distance-reduction-minus-1.2-times-contact-score utilities.
For each candidate query all 27 current-joint URDF primitive boxes at its predicted
half-second base displacement and yaw. Reject undefined predicted yaw. Filter any
candidate with a possible measured surface-voxel intersection. Select the highest
unchanged utility among remaining candidates, breaking ties by original bank
order (hold first). If none remain, latch `ALL_CANDIDATES_HAVE_SURFACE_INTERSECTION`
and perform the original ten zero-command terminal intervals. Do not treat such
a stop as a navigation success. Do not replace the target with a nearer one.

Both variants update the same memory with each accepted visual pose, so observation
processing is matched. Selection queries use either its current-frame index or
persistent index. This isolates use of retained surfaces in decisions, not memory
allocation/computation cost. Both retain all surface-check witnesses and original
utilities/predictions. No-hit means UNKNOWN, never verified clearance; the outer
controller explicitly performs an unqualified exploratory simulation experiment.
No floor classification or foot exemption is introduced. Footprint checks are
discrete, current-joint projections, not swept gait or calibrated safety bounds.
The observer and memory update every tick, but filtering occurs at the unchanged
five-tick replanning boundary. No new current-posture emergency gate is added.

Use fixed known integration layouts `family_episode_052` and `family_episode_039`,
with source-defined physics/appearance seeds, initial goal (1.2, 0) m, six original
candidate commands, three warmup ticks, 240 navigation ticks, original arrival/
quiet rule and terminal drain. Freeze the counterbalanced order:
current-frame 052, persistent 052, persistent 039, current-frame 039. Each gets
a fresh scene, model reload, observer, memory and process. No case shares state.
The exclusive attempt root is `go2_surface_memory_goal_probe_v1_attempt_001` under
the owned navigation-development artifact root. Case-named subdirectories retain
the original scientific trial identity separately.

Admission requires the exact complete surface-memory replay result
`d19b8254779c7aafb7ef4d0bee612a0d266da5236f1bea831f376d775c79dcfe`, all its
source/artifact bindings, full original native/input checks, the verified robot
URDF, and focused controller tests. Freeze every new source before native work.
Assess hardware, competing jobs and output reserve before launch. Use one fresh
worker serially: this four-case test measures complete-loop timing, for which
native-worker contention would confound comparison. The existing family throughput
benchmark does not override this measurement purpose. Plan 8 GiB of output above
the existing 40 GiB reserve and require at least 32 GiB available RAM.

Audit every frame, exact requested/applied command, model state and sensor-to-
observer-to-memory-to-selection replay. Retain all footprint/depth failures and
native contact records. Continue to the next fixed case after scientific failure;
stop subsequent cases after infrastructure/raw audit failure. Never retry/resume
an existing case. At completion compare native verified goals, terminal/minimum
goal distance, contact/visual/filter/budget stops, action choices and recorded
timing for each matched pair. Fewer contacts with no goal-reaching is insufficient
for success. One mirrored layout pair and one model seed give no independent-maze
or statistical benefit claim. Physics remains paused during computation.

This experiment does not add frontier exploration, route execution, physical
backtracking, realistic sensing delays or hardware motion. Those remain required
parts of the active objective; they are not replaced by a veto-counting assay.
