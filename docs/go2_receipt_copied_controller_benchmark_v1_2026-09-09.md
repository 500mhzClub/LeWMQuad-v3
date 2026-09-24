# Paired original and receipt-copied controller replay

The completed earlier phase diagnosis placed most controller time in geometry,
map operations and selection, with model forward near7ms. The current controller
already inherits the per-observation floor cache. The separately tested receipt
copier improved component copying on saved receipts but has not been integrated
into a native controller. This experiment tests an isolated selector integration.

Use the complete514observations of original learned development maze2, result
a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720,
including warmup and terminal drain. Authenticate complete artifacts, source,
environment and original raw audit. Use two fresh instances of the assigned
seed_2026091001_full_jepa corrected model, both state
4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6,
and separate original/candidate controller histories. No training or new scene.

The candidate uses identical original selector code objects, closures, defaults
and initialization. Fifteen explicit function forks substitute only the existing
receipt copier and dependency bindings to other declared forks. A mirrored class
hierarchy routes original super() calls through those forks. No original module,
function namespace, source file or live controller is patched. Controller,
observer, map, mission, model, forecasts and decision schema remain original.

Replay each observation in alternating order: original first on even frames,
candidate first on odd frames. Both complete decisions must equal the saved
original, not merely its command. Check public input fingerprints before and
after each controller. Check every original dispatched request and completion;
the final observation without a following command must already be terminal.
Stop immediately on mismatch, preserving the partial timing stream and failure.
Never follow a changed command with another recorded observation. Verify both
model states and absent gradients, then all source/input/artifact bindings again.

Time only controller.observe with monotone wall timers. Acquisition, input
reconstruction, hashing, output comparison and receipt writing are outside these
timers. No cProfile instrumentation. Report paired reductions and both medians
for non-warmup, nonterminal observations, plus both execution-order subgroups.
These measurements can demonstrate a controller-replay difference on this fixed
trajectory; they cannot establish a new navigation outcome, complete-loop speedup,
real-time scheduling, generalization or hardware qualification. Keep failures and
negative or negligible speed differences.

One sequential CPU replay process, two models/controllers, one numerical thread,
16GiB available-memory admission,40GiB storage reserve plus64MiB output allowance.
Refresh CPU topology/affinity/load, RAM, GPU/VRAM, storage and competitors before
launch and periodically during replay. It may overlap the existing sole native
scene with measured headroom. Capacity admissions are not enforced OS quotas.
Output:go2_receipt_copied_controller_benchmark_v1_attempt_001 under the established
external development artifact root. Preserve exclusive output and sealed custody.
