# Whole-task pilot: faithful execution, zero discoveries or returns

The corrected collector completed all four trials. Full raw audit passed all
four and reproduced all400 controller decisions. Every trial failed the first
local visual-change arrival check. None discovered the hidden marker, completed
a local leg, changed to return, or claimed home. There were no native contacts,
body-limit stops or sensor faults. The whole-task scientific objective remains
unachieved; this is an informative failed integration experiment.

## Fixed outcomes

| Layout | Memory | Task success | Terminal reason | Elapsed | Final distance from home |
|---|---|---|---|---|---|
| North dogleg | Episodic | No | Local no visual change |10.4 s|1.572881 m|
| North dogleg | Local only | No | Local no visual change |10.4 s|1.572881 m|
| South branch | Episodic | No | Local no visual change |10.4 s|1.572881 m|
| South branch | Local only | No | Local no visual change |10.4 s|1.572881 m|

The initial RGB non-detection and physical marker-center occlusion checks pass
in all four, as does the actual250-row zero-command release. A raw-array identity
diagnostic places all four trials in **one exact physical trajectory group**.
The images differ between layouts, but the resulting commands and physics do not.
These are not four independent demonstrations of generalization. The memory
ablation never reaches a completed outward route or a return choice and cannot
establish whether memory helps. No learned method was used in this pilot.

## Failure mechanism and what not to infer

At the terminal decision, global time11.4 s, the body was quiet and the command
progress proxy was1.498330 m against the local required1.084688 m. The remaining
failed predicate was the unchanged floor-mask-change threshold0.1: final three
values were .063958/.063333/.062917 in north dogleg and
.074375/.073750/.073542 in south branch. Even the maximum over each local
attempt remained below threshold (.082708 and .079167 respectively).

Independent diagnostic geometry places the terminal base at
(1.570837,-.027944,.314684) m. The instantaneous nominal articulated robot's
rear is .458099 m beyond the first construction-cell boundary x=.72 m. Thus
there was substantial real translation despite rejected image change. This
post-hoc boundary diagnostic is **not** a new success metric, an observed portal,
a qualified arrival, or proof of clearance for a future turn. It must not be
fed back as a hidden controller input. The original whole-task outcomes stay0/4.

The inspected terminal RGB is largely untextured floor and walls. Global floor
mask difference is not a measurement of where a portal lies or whether the
whole body has crossed it. Earlier fixtures already demonstrated the converse:
visual change could pass before whole-body clearance. Lowering the threshold
on these outcomes would not supply the missing geometric state.

Episodic arms correctly retain the failed attempt and enter
UNCERTAIN_AFTER_FAILURE, route depth0 and zero trusted graph edges. They do not
turn movement distance into invented place identity or claim a return.

## Evidence and integrity

Original collector72844 remains terminal0/4 on the preserved legacy sampling
infrastructure failure. Its237 source bindings and failed root are unchanged.
The separate correction changes only unused region annotations and settling
trace aggregation, preserving the original scientific protocol and controller.
Nine focused correction tests95938 pass; final full60058 passes1,026 tests
across89 files in74.64 s. Preflight26669 passes243source/186input/2gait bindings.

Corrected collector95031: COMPLETE4, exit0. Full audit14299: PASS4/400 decisions,
exit0. Evidence includes23,800 native physics/fast-gyro samples,2,380 ordinary
sensor samples and420 actual RGB packets. Audit checks raw contacts, state-derived
sensors, causal histories, camera geometry, physical object readback, command
tape and slew, exact controller/marker/memory replay, independent physical
return reduction and paired settling identity. PASS certifies faithful evidence,
not scientific task success.

Read-only diagnostic47707 passes bindings and computes trajectory groups,
terminal nominal articulated support, novelty history and memory failure state.
The first diagnostic50422 stopped on a nonexistent snapshot key `route_depth`;
the corrected read uses `hypothesized_route_depth`. Neither diagnostic writes
artifacts, changes scores, or reruns physics.

Corrected root: `.generated/go2_whole_task_navigation_sampling_correction_development_v1_attempt_001`.
Launch SHA-256: `a446d92c2cf5e4fd80391fb52073d5dd8ccfad25780a471a8610100abf322043`.
Result SHA-256: `dc87600fc6fa6ce96c5109b951db715f4ffff9b444dfcab5e7f353731db2b7da`.
Full audit SHA-256: `a9f92c26f66dbf79922b238dc0053dbd71f7b33d9e17aff04f2633cc7fb796d1`.
All six correction source/test/protocol/witness paths are now bound. No collector
or audit remains running; do not edit or restart completed attempts.

Post-document guard85730 passes all243 source,186 input and two gait bindings,
plus the three exact corrected launch/result/audit identities, exit0. No live
experiment, audit or verification process remains.

Next: [observed local geometry feeding the continuous mission](go2_observed_geometry_whole_task_next_steps_2026-09-05.md).
