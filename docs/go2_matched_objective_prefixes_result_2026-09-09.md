# Matched training-objective prefixes: all three cases complete

The actual JEPA and supervised-rollout models produce different requested
commands at the first planning observation on all three fixed layouts. Their
preceding actual commands and current observed state match. This establishes
a prospective control-decision difference between the two matched fitted
pipelines, not a JEPA navigation advantage or a physical alternative outcome.

Session58952 exited0. Result
`edf680896e11ef22096cf56323094be2de24b712405466149cdeeb59bab2a6fd`;
launch `665fc0db005929ce763e132b2161d210cb32bdbc9c8f753cc162cc9f81897fc6`.
Root `go2_matched_objective_prefixes_v1_attempt_001`;1,664source bindings,
182.68536946782842s after launch. All cohort/upstream/source/fit/artifact
checks completed, all fixed cases executed, no model training or simulator.

| Layout | Consumed observations | Earlier actual commands exact | First command difference | JEPA request | Supervised-rollout request |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 4 | 3 | 3 | [0.16,0,0.45], left arc | [0,0,-0.45], right turn |
| 2 | 4 | 3 | 3 | [0.16,0,0.45], left arc | [0,0,-0.45], right turn |
| 3 | 4 | 3 | 3 | [0.16,0,0.45], left arc | [0,0,-0.45], right turn |

Each case uses a fresh pair of loaded models and unchanged observed controller
instances from episode start. All four JEPA decisions reproduce the original
native record exactly. Both arms share their raw/registered pose evidence,
mapping and mission state. Their complete6x8x5prediction banks first differ
at3, with no terminal in either arm. The same rollout head and100–800ms clocks
are used. All input arrays remain unchanged. No observation4was consumed in
any case, and no unexecuted physical outcome is inferred. The12consumed
observations and three paired planning contexts are not12navigation trials.

Both neural base states and all correction buffers matched their authenticated
first-seed assignments. Corrected model-state SHA-256:

- JEPA:`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.
- Supervised-rollout:`171c8576d2c3fcfd0ce698351acf86a05f2ea29bdf829041e98641cdac778a73`.

Both remain unchanged after each case, with no gradients. The two arms share
initialization, training data/schedule,1200updates and architecture/head; each
retains its own intercepts fitted with the same training-only correction
procedure. Thus the comparison changes the fitted training pipeline, including
its resulting correction values; it does not hold unequal numerical biases
artificially equal. Only the declared JEPA arm adds the latent EMA-target loss.
One optimization seed and three initial planning contexts do not establish
generalization, optimization robustness or which policy navigates better.

Saved paired decision-stream SHA-256 for layouts1/2/3:
`4c3348e2a0aa1c684aa45166049c125c05878f6cb9ce6a75218af429fbc47fb5`,
`0fc047c473a81bd0665349a983b02c0d58e52e784f2374026b77bad224172265`,
`bb0992931296db7b4385cd98a063a5c2da55d04961eead53cc36172944707d8f`.
Each stream is under its original `full_jepa_novel_maze_0N` case directory;
`jepa_decision` stores JEPA and `decision` stores the alternative. Directory
names identify the source episode, not the alternative model treatment.

Launch hardware80,009,658,368availableRAM bytes,88,065,753,088artifactfree bytes,
CPU6.4%,bothGPUs0%,all32affinity. It ran as one CPU process alongside immutable
memory readout and residual-native input validation, with no second scene.

Next physical comparison must use all fixed layouts1,2,3 with the same original
observer/controller and exact supervised corrected state. It must reproduce
900physics samples, four paired observations and all saved alternative decisions
through3for each layout, and audit all newly executed outcomes afterward.
Retain the original learned cohort as the paired comparator; do not select
models or layouts from these new results. The supervised native cohort and
its prefix checker/launcher are not yet implemented. It follows the already
queued residual then tracking native work unless a later explicit goal decision
changes order. No native, round-trip, real-time or hardware claim is made here.
