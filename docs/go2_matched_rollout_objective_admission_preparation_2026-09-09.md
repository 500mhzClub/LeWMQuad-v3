# Matched JEPA and supervised-rollout admission

Added a pure metadata admission helper for the existing first-seed full-RGB
JEPA and supervised-rollout fits. This prepares the missing training-objective
comparison under the current maze controller; it does not execute a new model,
change the active cohort, or establish a JEPA navigation advantage.

Sources:

- `lewm/matched_rollout_objective_admission_development.py`, SHA-256
  `d7d523aff27f9d4ee095a9da91055ed4d3de0f06d6f86faabfe80d43f7a396b3`.
- `lewm/tests/test_matched_rollout_objective_admission_development.py`, SHA-256
  `19ff02d144fca794ccaa2dfe0cd58f81f2b9cc51c5fae9312e6f47c3b3d47dcf`.

## Comparison supported by the frozen training implementation

`lewm/observation_horizon_learning_development.py` applies identical direct
outcome, rollout outcome, variance and covariance losses to these two arms.
Only JEPA adds the latent-prediction loss against the EMA target. Both maintain
the EMA after each optimizer update. The shared `active_parameters` function
in`lewm/pulse_timed_learning_development.py` assigns both arms the same encoder,
history, direct-plan/decoder, transition and rollout-decoder modules.

Both native assignments therefore use`rollout_outcomes` with full predictor
RGB. A direct-head model would change the inference head as well as training;
the reactive comparator changes the planning method. Those remain distinct
comparisons. The earlier direct comparator assignment is preserved; this
additional supervised-rollout comparison does not replace it or claim that
an objective effect has already been measured in mazes.

The fixed pair is`seed_2026091001_full_jepa` and
`seed_2026091001_full_supervised_rollout`. They share initial state
`ed2c1f096b430b424cf6e381047eeee7672934bcac2fdbe198d5f85c2cead607`, the same
training experiment/data/schedule binding, latent width32, learning rate.001,
EMA.99 and1,200updates. Their frozen snapshot file hashes are respectively
`abf5272ab9cde930dd5408a3ae46a426bcf8453a896013c17de9d3c87d9091e8` and
`1c3eb0b1b02f6c594cccf68f68e9f56490d6c126bec496fcdff9ff004393b9bb`.

Retain each model's own already fitted training-only XY intercepts. Require
the same estimator,408training contexts,7,200draws, horizon support and
weighting, while allowing the learned bias values to differ. Forcing equal
bias values would change the existing fitted inference pipeline. Neither
correction may use native maze data, change yaw/contact, or claim probability
calibration. Thus the proposed comparison concerns the two training pipelines
with their identically specified training-only correction procedure, not an
effect of neural weights with all numerical correction values held constant.

## Verification completed

All21focused tests pass in0.11s. Tests reject altered initialization, schedules,
RGB variants, update counts, selected/substituted checkpoints, a direct-head
assignment, mismatched correction ownership/population/weighting, native-fitted
bias, contact changes, nonfinite corrections and changed forecast clocks.
They also check that the result and its nested records do not mutate evidence.

The helper accepted the actual authenticated fit metadata. The inspection
verified completed native result
`1597adbb2291ac0a7e6c3dcd5699bb233567e17ed8fd09f8104eabbd623d8374`, its bound
launch, and both exact fit JSON files against the inherited frozen fit artifact
map, then rechecked the input metadata bindings. The unchanged native launch
supplies the already completed eighteen-fit and training-correction admission.
This small inspection loaded no checkpoint, tensor dataset or simulator.

The helper is deliberately not a byte verifier: callers must authenticate the
complete upstream evidence and fit files before calling it, then use the
existing evaluation-only assigned loader. Its output explicitly retains this
requirement. It does not supersede the original eighteen-fit audit.

## Remaining prospective work

Bind this helper and the existing model loader into a separate declared
comparison. Verify actual corrected model states and a causal recorded-input
decision intervention before attributing changed commands to the objective.
For physical evidence, use fresh native execution with the same controller,
sensor pipeline, mission, action menu, geometry and evaluator; preserve each
arm's complete raw replay and all failures. Pair all fixed layouts rather
than selecting layouts or models by new outcomes. One optimization seed alone
does not establish robustness across seeds. No native comparison, broad JEPA
advantage, independent-maze success or hardware qualification is claimed here.

The current learned layouts1,2,3cohort and queued reactive/planning-memory
execution order remain unchanged. No frozen running source was modified.
