# Training-only translation correction readout V1

Require the completed exclusive correction fit and reconstruct all thirty
trained-head coefficient records exactly from original training predictions,
targets and draw schedules. Verify the original full eighteen-model admission.
No coefficient is chosen or refitted using this readout.

Apply each frozen float32 XY correction to both recorded roles for every model
and head. Save both roles and a prediction-completion receipt before scoring
either. Use the original score function, retain all strata, masks, target counts,
contact censoring and undefined-yaw accounting, and require original primary
scores to reproduce. Require unchanged raw yaw/contact components and unchanged
non-position scores. Report all before/after results, not just improvements.

Freeze source and input identities before the exclusive
`go2_training_translation_bias_readout_v1_attempt_001` root. Save complete corrected
prediction arrays, scores, per-model metrics and descriptive three-seed primary
summaries. Reverify inputs afterward; preserve any failure. Use one CPU process
and one numerical thread, 8 GiB available RAM and 256 MiB output allowance above
the 40-GiB reserve. No new RGB/future tensor materialization, neural inference,
model fitting, optimizer step or native execution is performed by this readout.

Training scores are resubstitution. Optimization-seed variation is not independent
maze replication. All contact logits remain uncalibrated. Keep the fixed first-seed
full-JEPA and full-direct assignments for the next prospective native probe,
regardless of which correction looks better. No navigation, real-time or hardware
qualification follows from improved prediction scores.
