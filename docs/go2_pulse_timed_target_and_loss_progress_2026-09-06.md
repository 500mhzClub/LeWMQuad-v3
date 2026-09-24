# Pulse-timed JEPA adapter: actual labels and matched objectives

The objective remains useful RGB-plus-sensor prediction for novel-maze
navigation. No model was fitted, optimizer stepped, checkpoint created or
learned controller deployed by this work.

## Actual recorded targets

The observation-pairing index retained all185 old coupled-room pulse windows.
The target-only native derivation now provides917 observed motion/contact
targets,8 censored requested-prefix targets and555 unknown-plan slots. All917
contact targets are negative. There are no positive contact examples in these
room streams; they cannot establish collision-risk discrimination or maze
generalization. Windows and repeated frames are not independent environments.

Original derivation attempt
`.generated/go2_pulse_native_targets_v1_attempt_001` failed before labels because
it expected boolean contacts, whereas the collector serializes binary uint8.
That failure is preserved. The explicit binary decoder rejects other encodings
and validates all three actual traces before the successor starts. The completed
successor is `.generated/go2_recorded_pulse_native_targets_v1_attempt_001`.
Result SHA256: `10418dec703a41947bc47b1f5f1851069ed376e5f6c4f1565cff7899fe59a4fe`.
Target SHA256: `a5dde045d15b7fd1269a147a24ed1decbc379b1a09ea0cdfb0f1e268d20e6b0f`.

Labels use full current-body displacement and projected R_current^T R_future
yaw, with exact2.2/2.5 s endpoints. Contact before command divergence can be an
observed positive even when the later endpoint is interrupted; noncontact
stops and events after divergence do not establish future safety/risk. Native
labels remain target-only. Fourteen semantic tests and eight recorded-format
tests passed, including3D coordinate invariance and all three actual traces.

## Matched training interface, still untrained

`lewm/pulse_timed_learning_development.py` provides a direct supervised,
supervised-rollout and JEPA objective for the pulse-timed model. It validates
the true target offsets and keeps input and target dictionaries separate.
Each arm exposes its online encoder to the same past and valid future images
through the shared regularizer. Only the JEPA arm adds EMA latent prediction;
the EMA encoder remains gradient-free. Recursive dynamics and rollout losses
are absent from the direct arm. Direct heads consume action values, validity
and exact cumulative time, not fabricated half-second completion.

Future-image and native-outcome masks are independent: a contact image may
exist without a valid collision-free motion label, and observed native motion
may exist without RGB. Invalid NaN placeholders are masked before arithmetic.
Outcome losses are averaged per window; latent prediction is averaged per
window with actual future images. Layout-balanced sampling remains a separate
required data/schedule decision, not something this loss proves.

Fourteen focused loss tests passed, covering gradients, equal image exposure,
censoring, timestamp errors, target-invariant inference, and an actual recorded
RGB/body/native-label join for two old windows. This last check evaluates
finite objectives with untrained weights; it is not a training result or
performance comparison.

## Next

Complete the fresh execution audit and fix demonstrated execution failures
without relaxing native acceptance. Then collect adequate action/state/scene
coverage in actual connected-maze workflows and freeze independent layout
splits before fitting or selection. The existing room traces can support a
development dynamics check but cannot replace those splits or collision data.
Use identical samples, initialization seeds, action vocabularies and budgets
across model arms; report coverage, contact-class imbalance, scene/action
ablations, task-relevant prediction error and actual action/mission outcomes.
Keep online rollout and memory ablations distinct from predictive-training
effects. Common sensing/local-control substrates must be identical across
navigation arms; adding depth to only one arm would confound a JEPA claim.
Realistic sensing, timing, clearance and bounded hardware remain required.
