# Six matched transition fits and separate execution benchmark V1

Use the completed new-scope input check at result
`92b5fa6066715bad218b9d2cecdd853a10ff2448e93ab886d5af1214de09a01e`.
Preserve the original family task-design/readiness failures, both native goal
failures, all missing contexts and the complete fixed train/transfer roles.
This is a local transition-prediction bootstrap, not independent-maze evidence.

Freeze all sources, this protocol, inputs and schedule before either phase.
Six scientific fits use seed2026091001 and the common validated1200-update,
batch6 schedule, crossing full/no_rgb with direct/supervised_rollout/jepa.
Use the unchanged cumulative-event trainer, latent32, AdamW0.001, EMA0.99,
position scale, gradient clipping and losses. Remove RGB from both input and
future teacher observations for no_rgb. No transfer optimization or sampling
by outcome. Each fit starts from fresh identical seeded model weights and runs
exactly1200 updates. No resume, retry, partial-success comparison or selection.

Before the scientific fits, run four separate full-JEPA20-update benchmark
cases with seeds2026091010–2026091013, each using the first20 common schedule
batches. Run those four fresh cases serially, then in four fresh spawned
workers. Keep their complete update ledgers and final model identities;
benchmark weights are never saved as eligible checkpoints or resumed. Every
paired update record and final state must match exactly. Select four workers
only if wall-speedup is at least1.25 and each parallel worker peak RSS is at
most8GiB; otherwise select one. Any numerical mismatch fails the benchmark.
All source/scientific settings must remain identical for the six-fit phase.

Inspect actual CPU/RAM/GPU activity, affinity, competing processes and output
space before each phase. Require40GiB available RAM and2GiB output allowance
in addition to40GiB free-space reserve. CPU/OpenCV/Torch/BLAS thread count is
one per worker; enable deterministic Torch algorithms. Use fresh spawned
processes, maximum one task per process. Monitor resources every approximately
16 seconds while work is pending. The memory allowances are observations and
admission checks, not OS-enforced memory limits. No GPU or physics execution.

Each worker owns unique exclusive request/log/ledger/terminal files. Persist
and fsync every actual optimizer update with its sample indices, schedule,
treatment, loss, gradient norm and model-state hash. A recording failure after
an optimizer step retains that actual update and fails the attempt. Complete
fits save exact receipt-bound snapshots and reload evaluation-only, verifying
full model/optimizer state before scoring. Sources, input artifacts and output
bytes are verified before and after work. On infrastructure failure, complete
already-running siblings and leave later batches unlaunched; retain failures.

Infer on all336 training and348 transfer contexts using their actual recorded
past packets and exact prospective family suffix plans. Save ordered raw
predictions, clocks and masks for the direct head and, where trained, rollout
head. Primary head is direct for direct fits and rollout otherwise. Re-read
saved prediction bytes and require exact score reconstruction. Report position
Euclidean error, wrapped yaw error and uncalibrated contact Brier score by
parameter cluster for all/initial/moving windows. Invalid near-zero yaw vectors
are explicitly missing; no finite yaw summary if any relevant target has an
undefined prediction. Preserve motion/contact censoring and denominators.
These are descriptive dependent local-layout metrics with no significance,
probability-calibration, navigation or contribution claim. Further scientific
readout and fresh native controls require their own receipts.

The fixed first native bootstrap candidate is full-JEPA seed2026091001,
regardless of transfer ranking. The corner observer is separately eligible by
complete recorded-trace replay. New controller/model/mission bindings and an
actual native audit are required before claiming any goal-reaching. Shared
RGB-D localization remains present in any no_rgb predictor comparison.

Exclusive roots under the navigation artifact base:
`go2_family_transition_fit_benchmark_v1_attempt_001` and
`go2_family_transition_fits_v1_attempt_001`. The fit phase requires the exact
complete benchmark result SHA-256 and recomputes its concurrency decision.
Frozen predecessor source and artifacts remain unchanged.
