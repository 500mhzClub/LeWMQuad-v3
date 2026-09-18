# Prospective expanded-context matched fitting and execution benchmark

Use the completed all-phase input result
ef48950b7987eaf9310ba8124a00c2e6e13c9b84b7cda6a362ac7ddb6ecd63fb and explicitly
admit scope correction8fcfdc678b721a84ed53c2545ec55284e2185d7b1e6bc5ca0dc26da4832e2dca.
Preserve the4010 available training contexts in4800 planned slots and all420
available original geometry-transfer contexts in456 original slots. Native
labels are targets only. Transfer inference uses the unchanged original stream,
with no future-image reader. No additional native collection occurs here.

Freeze18 fresh fits: seeds2026091001/2026091401/2026091402, input variants
full/no_rgb, conditions direct/supervised_rollout/jepa. Every fit uses the
original width32 CPU float32 trainer,1200 updates of six samples, learning
rate0.001, AdamW weight decay0, gradient clip1, EMA0.99, original loss and
0.06m XY normalization. Use the exact schedules in
docs/go2_all_phase_training_schedules_preparation_2026-09-10.json:600 batches
per source,75 draws per original family trial and50 per original switch trial.
Within-trial shuffled cycles cover every available context. This adds contexts
from the same120 recordings; it creates neither independent episodes nor new
command-switch phases. No outcome-dependent schedule, early stopping, best
checkpoint selection, warm start, resume or replacement attempt is allowed.

For subsequent prospective navigation, the primary candidate is fixed as
seed_2026091001_full_jepa. Its matched training controls are
seed_2026091001_full_supervised_rollout and seed_2026091001_full_direct; the
three seed2026091001 no_rgb fits are fixed sensor-ablation controls. Remaining
seeds provide optimization-repeat evidence, without substituting a winning
checkpoint. All fit predictions are persisted before scoring. No native
launch is authorized by this runner: model admission, any training-only
correction derivation, online planner/memory controls and the fixed layout
roster require a separately bound prospective navigation protocol. Existing
native queue/contact owners and the already prepared maze3 pilot are preserved.

Benchmark root:go2_all_phase_fit_benchmark_v1_attempt_001.
Fit root:go2_all_phase_matched_fits_v1_attempt_001.
Each root is exclusive; preserve partial artifacts and any failure. The fitting
phase requires the exact successful benchmark result and identical frozen source
and scientific definitions. Benchmark weights are never saved for reuse.

Benchmark one versus three independent CPU processes. Each process uses one
Torch/OpenCV/BLAS thread and first materializes all4010 admitted training samples
into a private cache (7,359,601,120 tensor bytes,8GiB configured maximum). The cache
is reused across fresh models in that process; batches remain fresh stacks and
each cache hit rechecks consumed policy bindings. Each benchmark phase has three
process assignments, each running two20-update full-JEPA fits: benchmark seeds
2026091810..2026091815, using the fixed2026091001 data schedule. Run the serial
phase with one process at a time, then the same three assignments concurrently.
All120 updates per phase, six final model hashes, full fit records and six ledger
hashes must match exactly across phases. Require every process to warm all4010
samples and retain the exact full-cache byte count. Select three fitting workers
only if measured full-phase speedup is at least1.25 and each measured process
peak RSS is at most10GiB; otherwise select one. Nonfinite timings or unequal
models/ledgers terminate the benchmark rather than silently choosing serial.

The same worker implementation handles full fits. Assign the fixed18-fit roster
round-robin to the selected one or three processes. Each process warms its cache
once and executes its assigned fresh models sequentially; no model/optimizer is
shared between fits. A worker failure retains its ledger, completed siblings and
terminal reason, and stops its remaining fits. Other already launched workers
finish their assignments. The parent never retries a worker or creates a native
scene. Direct subprocesses provide explicit output ownership and lifecycle;
they do not use the native queue's multiprocessing scene-worker mechanism.

Assess current CPU/RAM/GPU/storage and competitors before launch. Require10GiB
per planned training worker plus32GiB native headroom and2GiB parent headroom;
three workers therefore require64GiB available RAM at dispatch. These are launch
allowances, not an operating-system reservation. Each worker checks its actual
RSS against10GiB during warming and every optimizer receipt. Retain40GiB free
on the artifact filesystem, with2GiB output allowance. The parent records
hardware and child-process state at most15 seconds apart. Full original source,
collection, target and input admission runs before and after each phase; every
worker verifies launch/source/input/correction identities and its consumed
policy leaves. Do not repeat full collection hashing per optimizer step.

The current trainer, admitted input tensors, prediction export and checkpoint
contract are CPU-based. This study retains those numerical settings. GPU fitting
is not an admitted interchangeable implementation and no GPU speed claim is
made. Any later GPU implementation requires separate validation before use.

Full fits write all1200 durable update receipts, one final receipt-bound snapshot
and evaluation-only reload, complete4010 training/420 transfer predictions for
all active heads, then source/cluster scores. Benchmark and fitted snapshots
cannot be resumed. Model identity, input variant, dataset/correction identity,
schedule, update count and initialization must remain bound. Every seed must
have six identical initial model hashes. All21600 full-fit updates must be
accounted for before completion. Prediction loss does not establish navigation,
calibration, real-time performance, independent-layout reliability or deployment.
