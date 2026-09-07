# Frozen-objective loss/gradient diagnostic V1

Posthoc read-only analysis of all nine completed pulse-timed fits, at reconstructed
initial weights and at their saved final weights. Use the identical first six-row
batch of the frozen 72-draw training schedule for all 18 measurements, including
its actual past/future observations and independent target masks. This batch is
not a representative independent data sample or a reconstruction of all 108 updates.

Decompose each outcome loss into XY, sin/cos angle and contact. Preserve the
original four-channel SmoothL1 denominator (XY and angle each have coefficient
0.5), per-window reduction, unit beta, BCE, weighted variance 0.1, covariance
0.01 and JEPA latent term. Check the sum and every active-parameter gradient
against the unchanged frozen training_loss implementation before reporting.
Report weighted scalar loss, Euclidean parameter/encoder gradient norms and
cosine with the total gradient before clipping. These gradient contributions
are not additive norms, Adam update magnitudes or proof of causal interference.

No optimizer steps, EMA updates, batch selection, checkpoint selection, resume,
new physics or controller integration. Bind source, raw inputs, schedule and
all nine checkpoint hashes before loading. Verify model tensor identity and
absence of parameter gradients before/after each diagnostic. CPU one thread,
deterministic algorithms, 10 GiB storage reserve, exclusive output directory
`.generated/go2_pulse_loss_scale_diagnostic_v1_attempt_001`.

Use the measurements to inform a distinct prospectively fixed objective/adequate
budget comparison, not to extend or rewrite the completed pilot. Small XY loss
alone is not proof that scaling it fixes prediction. Independent layout/state/
support/contact coverage and matched sensor/training/rollout/memory comparisons
remain required for the full scientific goal.
