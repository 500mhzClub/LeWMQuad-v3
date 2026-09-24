# Augmented family/switch matched fits V1

This prospective implementation follows the separately recorded augmented-fit
design. Authenticate the unchanged completed original family input check and the
complete passing new 144-cell input check, every bound source and data artifact,
the full role accounting and all three fixed branch schedules. Bind their exact
result hashes as an input pair; its canonical JSON SHA-256 is the snapshot's
dataset identity. Inconsistent shared native/input identities fail admission.

Use eighteen fresh CPU models: seeds 2026091001, 2026091401 and 2026091402 crossed
with full/no-predictor-RGB inputs and direct/supervised-rollout/JEPA objectives.
The existing cumulative-contact trainer, architecture, outcome scaling, losses,
optimizer, gradient clipping and EMA implementation remain unchanged. Latent
width is 32, AdamW learning rate 0.001, EMA momentum 0.99, batch size six and
1,200 updates per model. Each seed shares its initial state and data schedule
across all six arms. No model receives a warm start or chosen intermediate step.

For each optimization seed, derive both source schedules at 1,200 updates using
that seed, then take the first 600 batches from each. Interleave their 600 pairs,
with the source order in each pair shuffled using seed plus 1,000,000,000. This
gives 75 draws per original training episode and 50 per new training cell.
Each new-source batch contains six suffix siblings from one matching context.
All transfer labels stay outside optimization. Keep complete censored target
masks and all original/new population denominators.

Before scientific fitting, use the exclusive
`go2_augmented_family_switch_fit_benchmark_v1_attempt_001` root for four serial
then four parallel short full-JEPA fits. Benchmark initialization seeds are
2026091410 through 2026091413; each uses the same first twenty batches of the
mixed optimization-seed-2026091001 schedule (ten from each source). Require exact
paired update ledgers and final model identities, twenty updates each, measured
positive wall times and no failed worker. Four workers require at least 1.25
speedup and each observed worker peak RSS at most 8 GiB; otherwise select one
worker. Benchmark weights are excluded from scientific fitting.

Check hardware/competing jobs and require 40 GiB available RAM and 2 GiB output
allowance above the unchanged 40-GiB artifact reserve. Use one OpenCV/PyTorch/
BLAS thread per worker and fresh processes with one task each. Monitor resources
and dispatch bounded batches. A failed worker terminates later batches; running
siblings and all ledgers remain preserved. No retry, replacement or resume.

Freeze executable source bindings at benchmark launch. Scientific root
`go2_augmented_family_switch_fits_v1_attempt_001` must use exactly those sources,
input-pair identity and science settings and the measured worker count. Persist
every one of the 21,600 scientific optimizer steps durably, including global
sample indices, schedule hash, loss components and model state hash. Require a
common initial hash across the six arms of each seed. Save one final snapshot
per fit, bound to experiment, input pair, schedule and input treatment. Reload
it through the unchanged evaluation-only snapshot validator.

Inference uses only source-specific past policy packets and prospective plans.
Save complete raw predictions for both roles and their completion receipt before
scoring either role for that model; no score
affects a fit, schedule or checkpoint. Report source, parameter cluster and
initial/continuing old windows or repeat/switch new branches, including the
first 500-ms horizon separately. Preserve motion/yaw/contact counts, censoring
and undefined-yaw failures. Resubstitution is explicit for training scores;
neither transfer parameter clusters nor repeated horizons are independent mazes.

The fixed first full-JEPA seed remains the primary native candidate; a matched
full-direct comparison is also required. No favorable score establishes
navigation, probability calibration, realistic timing or deployment. Native
control requires a separately frozen protocol and full eighteen-fit admission.
The previous near-wall observer failure and all earlier failed arrivals remain
unchanged and unresolved by fitting alone.
