# Moving-action switch augmented fitting design V1

Design recorded while the fixed 144-cell collection is still running, before
new predictive scores or optimization. Collection and input gates must pass
before this design can become an executable fitting protocol.

Retain the existing audited family-transition training windows and add the new
matched moving-prefix branches. The new population covers only one appearance
and left-open geometry per cluster, so replacing all old data would discard
existing mirrored-layout, appearance and time-offset coverage. Use the new data
as an augmentation, with the original 48 training episodes and 72 new training
cells kept in their fixed roles. Both sources use the same two training clusters
and two reused development transfer clusters. Neither supplies independent
novel mazes.

Use the unchanged latent-width-32 cumulative-contact model, AdamW learning rate
0.001, EMA 0.99, 1,200 optimizer updates and batch size six. Cross direct,
supervised-rollout and JEPA objectives with full/no-predictor-RGB inputs and
optimization seeds 2026091001, 2026091401 and 2026091402: eighteen fresh final
models. All arms of a seed share initial state, data schedule and numerical
settings. No warm start, checkpoint choice, continuation or outcome-based
schedule change.

Balance 600 batches from the old family windows and 600 batches from the new
moving-action branches. Use the unchanged old family view's seeded balanced
schedule for its first 600 batches: 75 draws per original training episode.
Use the new branch view's seeded balanced schedule for its first 600 batches:
50 draws per new training cell, six suffix siblings per branch batch. Alternate
old/new batch pairs, with each pair's source order seeded prospectively. Keep
the actual plans, availability masks, native target conventions and policy-only
inference paths from each authenticated source. Do not fabricate additional
targets from new episodes or reuse benchmark trajectories.

Before scientific optimization, measure serial/four-worker fitting throughput
with separately seeded short JEPA fits and require exact update/model equality,
bounded memory and useful measured speedup. Preserve all hardware and ledger
receipts. Evaluation must save complete per-source train/transfer predictions
before computing motion, yaw and contact scores. Report initial/continuing old
windows and repeat/switch new branches separately, with parameter clusters as
the sampling units; repeated horizons and suffix siblings are not independent.

Compare new branch predictions against all six unchanged original fits using
the same contexts and labels. Favorable predictive loss cannot select a hidden
checkpoint or establish a navigation result. Follow with separately frozen
matched closed-loop probes, then the still-outstanding independent-maze,
reactive/non-predictive, planning/memory, sensing/timing and hardware evidence.
The existing observer's near-wall failure remains unresolved by this design.
