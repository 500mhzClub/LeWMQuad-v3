# SAFE_LOCAL_WAYPOINT_PLANNER_PURPOSE_BUILT_V1

Status: `PURPOSE_BUILT_LOCAL_WAYPOINT_DATA_NO_GO`

This run preserved the prior `TRUE_FUTURE_LOCAL_WAYPOINT_PLANNER_NO_GO` as a
four-fit/two-held-out result whose branch ledger lacked the local-waypoint
geometry contract. It did not reuse that result for the present gate.

## Prior metric reconciliation

The prior `selected_unsafe_rate=0.9583333333` used two selected held-out
states as its denominator. It measured the safety value of selected branches,
not all admissibility decisions or unsafe candidates admitted by a safety
filter. The old normalized-regret implementation was mean absolute regret
divided by mean absolute progress plus `1e-6`; it used the best candidate over
the full candidate set, not the best safe candidate, and therefore was not
bounded to `[0,1]`. Per-state selected candidates were state 6 → 5 and state
7 → 17; selected progress was `0.2889870978` and `-0.2624094162`, with
reported regrets `0.6351838935` and `0.3326693165`. The old ledger did not
retain safe-candidate counts or safe-positive-progress counts, so those fields
remain unavailable rather than being reconstructed.

## Purpose-built collection

The deterministic pre-outcome selection froze 48 states (12 per family),
with 12 candidates each and 576 valid branch rows. The split is 32 fit, 8
calibration, and 8 held-out states (8/2/2 per family), digest
`ebef7db828a4c754432375818fd6b1eff0731cc3bc546ff2b69667b03abe56a8`.
The state manifest digest is
`da67309c073f60d74e4b85427237b19691552a542136e6ddb95939f14b4c5c37`; the
branch-label ledger digest is
`9b25b227c3e4de11e68e4abee454c4251399fafb468458a4e0d65f89bc6cdf7c`.

At H3, safe-positive-progress states were 83.33% in large and medium
enclosed mazes, 66.67% in small enclosed mazes, and 50.00% in loop-alias
stress; pooled coverage was 70.83%. States with at least two safe candidates
were 66.67% pooled. The required gates were 75% and 50%, respectively, so the
primary adequacy gate fails. Unsafe labels were nondegenerate (70.66% of
branches pooled), and each family contained both safe/unsafe and positive/
non-positive progress examples. Progress ranges were nondegenerate in every
family.

The collector produced H1–H3 poses, body-frame motion, waypoint distance and
progress, completion, clearance and tick-level safety traces. It did not
produce true RGB/ViT-L target latents: the purpose-built collector has no
rendered visual-target shard, so visual supervision required by the planner
contract is also unavailable. No planner training was therefore authorized.

## Evaluator fixture

The fixture digest is
`6d3457472444353f508a83158cc26afbafd8e370e9a03c626a0865ef550bae5e`.
It verifies perfect and reversed rankings, unsafe high-progress rejection,
all-unsafe abstention, deterministic tie handling, and normalized regret on
the safe candidate set. The corrected evaluator defines

`(best_safe_progress - selected_progress) /
max(best_safe_progress - worst_safe_progress, 1e-8)`

only when at least two safe candidates exist; unsafe selections are reported as
safety violations, not folded into that denominator.

## Decision

`PURPOSE_BUILT_LOCAL_WAYPOINT_DATA_NO_GO`. The declared data gate failed before
planner training, and visual target latents were not present. No planner seed,
predictor seed, predictor checkpoint, or simulator predictor evaluation was
opened. No global beacon memory, novelty layer, or routing was implemented.

Generated artefacts remain under `.generated/safe_local_waypoint_purpose_built_v1/`:

* `state_manifest.json` — `da67309c073f60d74e4b85427237b19691552a542136e6ddb95939f14b4c5c37`
* `branch_labels.jsonl` — `9b25b227c3e4de11e68e4abee454c4251399fafb468458a4e0d65f89bc6cdf7c`
* `data_adequacy.json` — `80721c838c75dc852335f30bd80becae59703de6c0da064ebecde5d72e31cefc`
* `evaluator_fixture.json` — `6d3457472444353f508a83158cc26afbafd8e370e9a03c626a0865ef550bae5e`
* `split.json` — `ebef7db828a4c754432375818fd6b1eff0731cc3bc546ff2b69667b03abe56a8`
* `old_metric_reconciliation.json` — `4563734550709d49cf84a720830253ab973c68c8fbf2972c1eb163f05370e724`
