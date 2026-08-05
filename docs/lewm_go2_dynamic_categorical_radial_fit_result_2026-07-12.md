# Dynamic categorical radial fit diagnostic result

Date: 2026-07-12

## Scope and verdict

This development-only compatibility diagnostic is complete. It tested whether
the legacy dynamic categorical radial target/model could satisfy the frozen
320-frame aggregate-plus-all-family fit gate before any holdout, G2, shared-JEPA,
or runtime work.

The answer is **no**. `fit_gate_passes=false`, `qualifying_branch=null`, and all
licenses remain false. This closes the legacy-target diagnostic. It does not
justify a second seed, train-role holdouts, G2 access, checkpoint promotion, or
runtime use. The observable camera-evidence V4 target remains the active path.

## Immutable result

- result: `.generated/go2_dynamic_categorical_radial_fit/v1/seed_20260710_result.json`
- result file SHA-256:
  `bc374656c8a871bd111cba916553eb128249aa31f03031e85741206e3c5c0959`
- canonical content SHA-256:
  `5c1255df0f36bb5b1053dd546aa2323e3ea083be85237812b18138dc7b633086`
- seed: `20260710`
- fit rows: 320 frames / 160 transitions / 20 scene clusters
- model: `DynamicCategoricalRadialPerceptionFullRay`, 2,887,067 parameters
- device: GPU0, AMD Radeon AI PRO R9700, exact `HIP_VISIBLE_DEVICES=0`
- GPU1/Raphael iGPU: rejected by the runner and observed at 0% compute
- elapsed wall time: approximately 3 h 22 min

## Branch outcomes

The production-faithful branch ran its complete 2,000-update schedule. Its
aggregate panel passed every frozen check:

- hierarchical balanced NLL: `0.0121012` (required `<=0.03`)
- UNKNOWN/known balanced accuracy: `0.992071` (required `>=0.99`)
- FREE/OCCUPIED balanced accuracy: `0.999823` (required `>=0.99`)
- FREE recall: `0.999360` (required `>=0.98`)
- OCCUPIED recall: `0.986605` (required `>=0.98`)
- UNKNOWN recall: `0.985547` (required `>=0.98`)
- cross-scene wrong-RGB NLL delta: `4.30384` (required `>=0.25`)
- same-scene wrong-view NLL delta: `3.17507` (required `>=0.25`)

That aggregate result was insufficient because the frozen gate requires all
five scene families. Large enclosed, medium enclosed, and small enclosed mazes
passed. Two families failed:

- `open_obstacle_field`: UNKNOWN/known balanced accuracy `0.989535`, OCCUPIED
  recall `0.975450`, and UNKNOWN recall `0.979809`;
- `rough_local_dynamics`: UNKNOWN/known balanced accuracy `0.985765` and UNKNOWN
  recall `0.972340`.

No evaluation in the production branch passed the aggregate-plus-all-family
gate, including the exact terminal evaluations at steps 1,800, 1,900, and
2,000.

The conditional ceiling branch then ran its registered 5,000 updates from the
verified identical initialization. It also never passed. Its terminal aggregate
hierarchical balanced NLL was `0.0427696`, UNKNOWN/known balanced accuracy was
`0.974521`, OCCUPIED recall was `0.962358`, and UNKNOWN recall was `0.956058`.

## Access and promotion accounting

The result records:

- 320 train-role RGB decodes;
- zero non-fit image or label payload opens;
- zero checkpoint-selection or probability-calibration sidecar opens;
- zero G2 payload opens and zero G2 model outputs;
- zero non-fit model outputs;
- `g2=false`, `runtime=false`, `shared_jepa=false`, and
  `heldout_claim=false`.

The failure is therefore a valid train-only capacity/target diagnostic, not a
held-out evaluation. It also reinforces the first-principles correction already
established by the camera-observability audit: optimizing the old world-boundary
target further is not the route to deployable generalization. The next learned
perception result must come from the independently audited, camera-observable V4
evidence target and its separately reviewed ladder.
