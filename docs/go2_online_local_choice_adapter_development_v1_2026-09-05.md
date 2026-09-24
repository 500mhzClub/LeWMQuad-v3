# Strict online local-choice adapter, development V1

New implementation only; completed learning/model sources remain unchanged.
The adapter is a one-shot conditional local-decision boundary, not a maze planner
or a receding-horizon policy. It accepts only the strict causal RGB/body/control
packet, body-start xy intent of norm at most 0.8 m, and the current sensor clock.
It rejects extra privileged fields, wrong/reused episode/reset identities,
repeat selection, stale images/packets, off-command-clock requests, incomplete
or stale latest histories, and invalid intents. Invalid inputs issue no tape;
the caller must retain its independent stop/failsafe responsibility.

The fitted reference used complete zero-latency simulated histories. This adapter
therefore requires that exact regime instead of silently treating arbitrary
real sensors as qualified. It does not estimate odometry or use simulator state.
Its command-clock time is separate from measured wall-clock inference latency;
a synchronous simulation can pause physics during inference. Real-time delay
handling on hardware remains open.

The production factory within this development module has explicit immutable
hashes for the learning launch, completed prediction audit and all six selected
checkpoints. It verifies source bindings, weights and seed/condition/update
identity before use, rejects symlinked material, and loads the verified bytes
with weights-only loading. It does not load training/validation observations or
select a checkpoint using performance metrics. The all-stop comparator needs
no checkpoint but uses the same packet/intent validation.

Each learned condition uses **all three fixed seeds**. Five known candidate plans
are reconstructed from the previously applied command and the original slew
contract; no future measured commands enter. Every member predicts its eight
0.5-s rollout endpoints. Ensemble displacement is the arithmetic mean; contact
probability is the mean of member sigmoid probabilities, not the sigmoid of
mean logits. At 4 s, rank `10 * mean_probability + distance(mean_xy, intent)`.
Resolve ties in the original action order. The ensemble is not claimed calibrated.

Return the selected 40-tick command plus five zero-command release ticks, the
expected post-slew tape, all candidate plans/predictions/costs, image/tensor
hashes, model identities and measured latency. The code currently calls the
existing full model forward (including the unused direct head); latency reports
must include that actual work. Inference makes no weight updates.

Synthetic tests cover plan/release timing, five-candidate input separation,
mean-probability aggregation, deterministic ties, identity resets, malformed and
stale inputs, one-shot behavior, stop parity and model-byte/symlink guards.

The fixed replay qualification uses all eight development-validation canonical
contexts, three intents and three methods (72 adapter calls). It reloads the
actual bound models, compares every per-seed candidate prediction against the
saved intact rollout witness with maximum absolute tolerance 1e−5, independently
reconstructs ensemble costs/selection, and retains actual adapter outputs/timing.
Batch size changes from the original 40-row scoring to five candidates, hence
the fixed floating-point tolerance. This is an interface equivalence test—not
fresh generalization evidence, new physics or a learned online navigation trial.
No tolerance search, fitting, seed replacement or checkpoint change is allowed.
