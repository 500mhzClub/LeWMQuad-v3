# Continuation audit: reader-population-only correction

Physical collection completed all16 trials with result SHA-256
`003147bede55befe019362335faefe5173ae6e98c4b1bbb429fb22ffa44ff6bf`.
The original audit passed the first four short contact episodes, then failed on
the first completed episode before loading its first observation. Its existing
route reader imposes a341-frame population cap. Completed continuation episodes
contain431 frames; the fixed continuation protocol and original trial auditor
already allow at most806. This is an incompatible legacy reader limit, not a
physical task failure, missing capture or permission to extend a task budget.

Preserve the original audit FAIL unchanged:
`df3ed99d49c66e4161e6054358f4f4dd05309b041f95e54f8d29ee3e02dccd2c`.
Preserve all physics, sensors, images, controller decisions, outcome criteria,
models, original audit source and protocol. Do not rerun physical trials.

The separate `continuation_rgb_dataset_development.py` reader changes only the
function name and population cap341→806. All policy schema, history, image,
index, path, calibration, timing, tensor and privilege checks are identical.
The legacy route reader remains unchanged and continues rejecting populations
above341. AST equality tests verify the narrow change and that the corrected
`audit_trial` function changes only its reader calls. Synthetic tests cover the
342/431/806 boundary, reject807 and malformed policy inputs, and compare actual
short-episode packets exactly. Actual long-episode reads include indices340/341
and the final430, followed by the full independent raw replay/reconstruction.

The new audit binds its four correction source/test/document paths, two imported
synthetic-fixture sources, and the exact
original launch, result and failed audit before replay. Outputs are separately
named `audit_population_v2_binding.json` and
`raw_artifact_audit_population_v2.json`. Neither may overwrite the original FAIL
or be retried in place. No source or physical outcome is altered to obtain PASS.
This correction alone makes no claim that the full audit has succeeded.
