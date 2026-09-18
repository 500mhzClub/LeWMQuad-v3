# Observation-horizon eighteen-fit protocol V1

Authenticate the complete short-horizon input check and target derivation,
both original recorded populations, all source/data bindings and all three
unchanged mixed-context schedules. Bind the exact target-result and new
input-check hashes as the snapshot dataset identity. No source role,
availability mask or schedule changes.

Train eighteen fresh CPU models: seeds 2026091001, 2026091401 and 2026091402
crossed with full/no-predictor-RGB inputs and direct/supervised-rollout/JEPA
objectives. The explicit new model has eight one-command blocks and targets
at 100–800 ms. Keep latent width 32, AdamW 0.001, EMA 0.99, gradient clip 1,
the existing loss coefficients and 0.06-m position scale, 1,200 updates and
batch size six. Use exactly the original 600-family/600-branch batch mixture.
Pair initialization across each seed's six arms; old temporal-model snapshots
are incompatible. No warm start, chosen intermediate step or resume.

Before scientific fitting, benchmark four serial then four parallel fresh
full-JEPA models for twenty updates each. Benchmark seeds are 2026091610–13;
each uses the first twenty batches of science seed 2026091001's schedule.
Require exact paired update ledgers and final model identities. Select four
workers only with speedup at least 1.25 and every parallel worker's peak RSS
at most 8 GiB; otherwise select one. Benchmark weights are never reused.

Use the exclusive benchmark root
`go2_observation_horizon_fit_benchmark_v1_attempt_001`, then scientific root
`go2_observation_horizon_fits_v1_attempt_001`. Freeze all training, prediction,
scoring, snapshot, admission and focused-test sources at benchmark launch;
scientific fitting requires exactly those bindings and settings. Recheck
hardware and competing work, require 40 GiB available RAM and 2 GiB output
allowance above the unchanged 40-GiB reserve, and monitor resources. Use fresh
one-task processes with one OpenCV/PyTorch/BLAS thread each and bounded batches.
Retain running siblings and halt later batches on a failed worker. No retry,
replacement, checkpoint choice or resume is part of either phase.

Persist all 21,600 scientific updates, including exact sample indices,
schedule identity, loss parts and model state. Save one final receipt-bound
snapshot per fit and reload it through the new evaluation-only validator,
which requires the exact 100-ms/800-ms/one-command temporal contract and all
optimizer state. Old snapshots and changed clocks must fail admission.

Predict both roles from past-only inputs and prospective short plans, save
both complete raw arrays and their completion receipt, then score. Reconstruct
all 18 raw score populations, ledgers and snapshots before any native use.
Report source/cluster/role and repeat/switch or initial/moving strata, including
100-ms first-observation and 500-ms shared-horizon errors separately. Keep
contact denominators, censored targets and undefined-yaw failures explicit.

The fixed first-seed full-JEPA native candidate and same-seed full-direct
comparator remain assigned before results. Three-seed variation is optimization
variation, not independent-maze evidence. No fitting result establishes
navigation, probability calibration, real-time control or hardware deployment.
