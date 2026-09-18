# Full-population causal input validation for expanded training contexts

Validate the complete4800-slot/4010-available training target result
4d300f77849d174cc9d7bd2a35d276e0996d1b5ac8795bee4f419291ada6b328.
The old complete input check is
73e11e168f933633dbbd3b82668a5b021c076e18bbfc7704987008b4d290c1b5.
Require every one of408 shared available training contexts to reproduce the
old RGB/body/control tensor fingerprints, normalized action tensors, validity
masks and target offsets exactly. Preserve the original geometry-transfer data.

For every available expanded context, independently construct inference inputs
using only four actual past packets and the known original command suffix.
Then materialize the private training sample with a separate future reader,
and require every input tensor to equal its inference counterpart. Future
packets are training-only, actual same-episode timed observations with the
original target masks. Native target values never enter policy inputs.
Keep original RGB/body/control normalization and eight100ms action blocks.

The stream only admits original training rows. Exact source/trial/offset,
availability, history and derivation clocks are checked. Every consumed
policy manifest/history/RGB leaf must have its original collection binding
verified before and after materialization. Native physics, scene geometry,
contact tables and target labels are not policy-reader inputs. Missing
contexts remain accounted for; no future tensor is materialized for them.

Training-cache allowance is at most8GiB, measured by actual tensor bytes, with
LRU eviction and fresh stacked batches so callers cannot mutate cached samples.
The full-population checker uses zero sample cache. It retains only six initial
samples for untrained direct/supervised/JEPA model interface, finite-gradient
and unchanged-parameter checks. No optimizer is created, no parameter update
occurs, and no checkpoint is selected or saved. These checks are not fitting.

Use one CPU input-check process and one-thread Torch/BLAS beside the existing
single native scene. Require8GiB checker plus32GiB concurrent-native memory
allowance,40GiB artifact reserve and64MiB new metadata allowance. Inspect fresh
hardware before launch and record progress every128 materialized contexts.
The output go2_all_phase_training_inputs_v1_attempt_001 is exclusive; preserve
any terminal failure. Reverify source, target, original-input and consumed
policy bindings after the complete pass.

This validates a new training-input population. It does not add command-switch
phase diversity, independent trials, fitted weights, geometry-transfer gains,
native navigation, JEPA advantage or deployment qualifications. Matched
retraining schedules and assignments must be frozen separately before fitting.
