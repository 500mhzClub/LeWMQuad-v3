# Counterfactual corpus V2: explicit render-variation recovery

V1 stopped after 19/120 collected branches when its exact RGB hash requirement
failed. It remains `INFRASTRUCTURE_FAILURE`; no V1 result or artifact is rewritten.
Complete physical traces and sensor-history bindings match within each layout.
The triggering branch-start image differs from its reference at one pixel
(three channels, maximum difference 15/255), with identical camera transform.
Across 497 inspected prefix-frame comparisons, 40 differ; at most three pixels
change, maximum channel difference 16 and maximum raw 8-bit RMS 0.046875.
This supports bounded rendering variation, not stochastic physical branches.

V2 is a separately identified recovery assembly, not a resumption of the V1
process or a relabeling of its result. Retain and verify the 19 existing branches
at their original paths and source bindings. Execute only the 101 not-yet-run
layout/action pairs under the same fixed scene specs, roles, seeds, actions,
physics, gait, stopping rules and labels. No completed physical trial is rerun.
The composite manifest binds each member to its actual source root and hashes.
All 120 originally planned unique pairs remain in scope. Stop on any further
integrity/infrastructure error; no automatic retries or tolerance searches.

Require exact complete physical/command prefix arrays, sensor-history arrays,
timestamps and per-frame camera transforms. For corresponding prefix RGB frames,
allow raw 8-bit RMS <=1.0 and changed-pixel fraction <=0.001. Both conditions
must hold; a broad brightness change or wrong scene remains a failure. Preserve
every original image and report the actual differences, not just a pass flag.
These prospective tolerances address the demonstrated acquisition discrepancy;
they do not change physical task outcomes or observation validity.

For all five action examples from a layout, the model's branch-start context is
the same actual observed packet from that layout's first (`stop`) branch. It was
captured before action branching and represents the exactly matched physical
state. No image is edited or synthesized. Future observations and outcome labels
remain those of each independently executed branch. Thus the learner sees an
exact shared context, not action-correlated tiny rendering differences.

Audit original member hashes and source/gait bindings before adoption. Verify
complete raw contacts, causal sensor/image histories, prefix/tape integrity and
censored labels before fitting any model. Training/validation layout roles remain
unchanged (16/8); no validation branch has yet been collected or used for fitting.
The corpus remains development-only. No JEPA, generalization or hardware claim
follows from recovery or from dataset completion alone.
