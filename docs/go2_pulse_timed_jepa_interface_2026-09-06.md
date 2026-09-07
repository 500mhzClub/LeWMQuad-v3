# Pulse-timed RGB/body JEPA interface: source only, untrained

The running coupled controller uses a fixed empirical dynamics table, not this
model. No JEPA training, checkpoint, GPU job or online learned prediction was
launched by this work.

The existing `TemporalRGBBodyJEPA` accepts only complete half-second action
blocks. A 2-tick pulse followed by20 minimum braking ticks has a known prefix
of2.2 s; a 5-tick pulse has2.5 s. Padding the former through2.5 s would invent
three future commands, while discarding the partial block would omit the
actual minimum-braking endpoint. Quiet-dependent additional braking and the
next selected pulse are not known at departure.

The distinct `PulseTimedRGBBodyJEPA` retains the four-frame chronological
RGB/body/control encoder and frozen EMA target, but conditions both recursive
latent prediction and the direct outcome head on command values AND per-tick
validity. Only the final known block may be partial. Outputs carry exact
cumulative target offsets: .5,1,1.5,2,2.2 s for a short pulse, or .5,1,1.5,2,2.5 s
for a long pulse. Unknown horizons are zero placeholders with false validity,
not stop-conditioned predictions. The encoder sees no native pose or future
target observations in forward. The declared normalized command scale remains
[.3,1,.5]; actual pulse support remains forward.20 and yaw+/-.45.

Thirteen focused tests cover actual timing, missing-prefix validation, causal
invariance of earlier predictions under later plan extension, the difference
between unknown padding and known zero commands, public rollout helpers,
gradient flow and a frozen target encoder. These are analytic/tensor tests,
not learned predictive quality or physical execution evidence.

This is not a drop-in replacement for old checkpoints or training scripts.
The action/mask transition and direct heads have new weights. The old training
loss/sampler assumes full blocks and a five-action/sixteen-layout experiment;
it must not be reused blindly for these six pulse-duration cells. The next
training adapter must bind actual target image/body packets at the returned
timestamps, verify the executed known prefix, censor physical stops/missing
observations, and keep target-side motion/contact labels outside the model
input signature. An estimator failure need not mean missing raw RGB; keep those
distinct and do not silently discard difficult observations.

Freeze independent layout splits before fitting/selection and collect diverse
scene/body/action histories. Open-room dynamics with only two nominal samples
per action cannot establish that RGB or JEPA contributes useful scene-dependent
information. Compare empirical/geometric, direct supervised, supervised rollout
and JEPA under matched input exposure, action choices and budgets. Separately
test online multistep rollout and persistent memory in completed maze missions.
No latent-loss or tensor-test improvement substitutes for that experiment.
