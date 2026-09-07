# Fixed matched pulse-training plumbing pilot V1

Train the pulse-timed direct, supervised-rollout and JEPA reference models on
the existing185-window development corpus. This is the first optimizer pilot
for this particular pulse-timed dataset/model interface, not the first learned
model anywhere in the repository. It is not a navigation-policy experiment.
All three trajectories share one room layout, have zero positive contacts and
provide no independent selection/evaluation layouts. Every reported before/
after score is train-role resubstitution, not generalization or JEPA benefit.

## Fixed training and scoring

Run seeds2026090721,2026090722,2026090723, each in direct, supervised_rollout,
JEPA order. Each fit starts from identical full weights for its paired seed,
latent width32, CPU deterministic algorithms and one Torch thread. Reuse the
already frozen12-update/6-sample schedule (seed2026090711), identical across
all arms/seeds. Thus there are9 fits and108 optimizer updates; no extension,
coefficient sweep, alternate seed, best-checkpoint selection or retry.

AdamW learning rate.001, zero weight decay, gradient norm clip1, EMA.99 updated
after each optimizer step. Use the frozen matched partial-time losses and
their existing weights. Direct excludes recursive modules from optimization;
supervised-rollout trains outcome dynamics; JEPA additionally predicts EMA
future embeddings. All expose their online encoder to the same actual past
and valid future images through the shared regularizer. EMA maintenance is
identical, but only JEPA consumes its latent target in the training loss.
This matches samples/updates/initialization, not wall time or active parameter
count. Finite gradients/model state are required; errors latch and preserve
partial logs rather than restarting a failed optimizer step.

Evaluate all185 eligible windows before and after training using actual target
offsets, including2.2s and2.5s final partial blocks. Report planar and wrapped
yaw error, contact Brier/accuracy/monotonicity, counts and per-offset metrics.
Contact accuracy is not hazard discrimination on an all-negative corpus.
Include zero-motion/no-contact as a simple control. Direct reports its trained
direct head; other arms report direct and recursive heads. Do not report the
direct arm's untrained recursive head as a model result. No future/native
targets enter inference inputs. No ranking or promotion follows from scores.

Save only the fixed final checkpoint, full optimizer state, initialization/
final tensor identities, exact sample-index update log and before/after scores.
Reload with weights_only=True, verify tensor identity and score restored
weights. This checks persistence, not scientific resume authority. Rare cells
are repeatedly sampled; two short-forward examples do not become12 independent
examples. Layout/action balancing does not balance support/body-state coverage.

## Integrity, resources and subsequent science

Exclusive output `.generated/go2_pulse_training_pilot_v1_attempt_001`.
Bind completed dataset launch/result/schedule, all inherited source/input/raw
artifacts and this new runner/launcher/test/protocol closure. Freeze at launch,
verify before and after. Require1GiB artifact allowance plus10GiB reserve;
check reserve before every update. No existing evidence is moved or deleted.
No source export, protected material, external messages, physical execution,
GPU training, checkpoint promotion or navigation use is part of this pilot.

The result must preserve negative outcomes and distinguish working training
plumbing from useful predictive control. The scientific next stage requires
new independently split maze layouts, support/body/action coverage and censored
obstacle/contact outcomes; multiple matched training seeds; RGB/history/action
ablations; task-relevant predictions and online decisions. Then separately
test online rollout and memory with the same sensing/local execution substrate.
Reliable navigation, realistic sensing/deadlines/body sweep and bounded hardware
remain unachieved requirements, irrespective of this pilot's loss curves.
