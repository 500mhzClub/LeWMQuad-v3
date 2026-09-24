# Robot-visible primary sensor compatibility replay V1

Replay the complete 20-frame primary RGB/depth/body/gyro prefix from the completed
robot-visible auxiliary sensor capture using the two preselected corrected
seed-2026091001 full-JEPA/full-direct assignments. Auxiliary depth must not enter
this replay. Preserve the observer, route/view logic, models, correction buffers,
candidate ranking, geometry checks and all existing controller gates.

For each assignment, independently reload a fresh model/controller twice and
require exact new-decision replay and unchanged full model state. Verify native
and public physical prefixes against each original model case. Report observer
and controller failure states, observed XY errors against evaluator-only native
pose, and forecast changes over the common executed causal prefix. Native pose
may enter only post-replay error scoring, never controller input.

Compare requested commands against the 19 commands actually executed. Include
the observation before the first different command or terminal, and exclude
affected later observations from causal forecast comparisons. The twentieth
observation has a proposed command that was not executed; compare that proposal
separately. Preserve every shadow observation and difference, even after an
earlier mismatch, without claiming a counterfactual executed outcome. A matching
prefix establishes neither full-mission nor prospective-navigation compatibility.

Freeze source/tests, the sensor result report, input bindings and model assignment
before `go2_visible_robot_primary_compatibility_v1_attempt_001`. Use one CPU
process and one numerical thread for this bounded 2-model, 2-pass, 20-frame
read-only check, with 8 GiB available RAM and 256 MiB output allowance above the
40-GiB reserve. No large training or scene workload is launched. Reverify source,
input and model/correction identities afterward, preserving any failure. No
training, native execution, auxiliary integration, independent-maze, hardware or
overall-goal qualification is performed.
