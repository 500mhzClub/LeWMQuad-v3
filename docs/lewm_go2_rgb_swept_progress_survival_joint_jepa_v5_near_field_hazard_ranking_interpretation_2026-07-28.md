# RGB Swept-Progress Survival Joint-JEPA V5 — Frozen Implementation Interpretation

- Status: frozen after independent preregistration review and before V5 source completion, runtime access, or training.
- Preregistration commit: `7fe075d752b5d14c539eaed213c9f28510659c79`.
- This note resolves implementation wording only. It does not add a loss variant, parameter, coefficient, margin, range, data source, retry, or evaluation change.

## Exact interpretation

- A “raster row” is one batch sample's complete `[64,64]` raster. Hazard pairs never cross samples or current/next views.
- Compute each available view mean over its eligible samples. If both current and next have eligible samples, average their two means equally. If only one view is available, use that view mean with weight one. If neither is available, return an exact zero graph-connected to both current and next semantic logits.
- The near mask has axes `[forward,left]`, shape `[64,64]`, and exactly 1,016 true cells. It is not intersected with another visibility or anchor mask.
- Hazard receipts use a distinct `hazard_*` namespace. The inherited `ranking_active_microbatch_count` and `ranking_eligible_pair_count` continue to describe the JEPA swept-progress objective `R` only.
- If V5 passes all unchanged V4 development gates, the separately frozen V5 calibration step will fit one new four-parameter calibration artifact from V5 calibration-role logits, search the unchanged 2,016 threshold tuples once on that role, and apply the selected V5 artifact and tuple unchanged to V5 selection-role logits. It will not reuse the V4 fitted artifact or read the V4 candidate.

## Required implementation invariants

- The fixed near mask contains 1,016 cells; class indices remain UNKNOWN/FREE/OCCUPIED = `0/1/2`; and the hazard score remains `occupied_logit - logsumexp(unknown_logit, free_logit)`.
- Per eligible sample, the receipt count is exactly `near_occupied_count * near_free_count`, and the loss is the mean of every registered Cartesian-pair term.
- Equal hazard scores produce normalized loss one; current-only, next-only, both-view, and neither-view aggregation are tested.
- Inherited `S/P/U/R/O`, optimizer/EMA behavior, model parameters, and terminal accounting remain unchanged; total loss is exactly `S+P+U+R+O+H`, with inherited `O` coefficient `0.5` and `H` coefficient `1.0`.
